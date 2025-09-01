#include "../include/distributed_transformer.hpp"
#include <stdexcept>
#include <chrono>
#include <iostream>
#include <algorithm>

DistributedTransformer::DistributedTransformer(const TransformerConfig& config,
                                             std::shared_ptr<TiktokenTokenizer> tokenizer,
                                             int* argc, char*** argv)
    : config_(config), tokenizer_(tokenizer) {
    
    // Initialize MPI first
    initialize_mpi(argc, argv);
    
    // Create base transformer on each rank
    base_transformer_ = std::make_unique<Transformer>(config_, tokenizer_);
    
    // Initialize NCCL for GPU communication if available
#ifdef CUDA_AVAILABLE
    // Create CUDA streams
    cudaStreamCreate(&compute_stream_);
    cudaStreamCreate(&comm_stream_);
    
#ifdef NCCL_AVAILABLE
    if (world_size_ > 1) {
        initialize_nccl();
    }
#endif
#endif
    
    // Initialize gradient buffers for synchronization
    initialize_gradient_buffers();
    
    if (world_rank_ == 0) {
        std::cout << "DistributedTransformer initialized with " << world_size_ 
                  << " ranks, local rank " << local_rank_ << std::endl;
#ifdef CUDA_AVAILABLE
        std::cout << "NCCL communication: " << (nccl_initialized_ ? "enabled" : "disabled") << std::endl;
#endif
    }
}

DistributedTransformer::~DistributedTransformer() {
#ifdef CUDA_AVAILABLE
    cudaStreamDestroy(compute_stream_);
    cudaStreamDestroy(comm_stream_);
    
#ifdef NCCL_AVAILABLE
    if (nccl_initialized_) {
        ncclCommDestroy(nccl_comm_);
    }
#endif
#endif
    
    if (mpi_initialized_) {
        // Only finalize if we initialized MPI
        int finalized;
        MPI_Finalized(&finalized);
        if (!finalized) {
            MPI_Finalize();
        }
    }
}

DistributedTransformer::DistributedTransformer(DistributedTransformer&& other) noexcept
    : base_transformer_(std::move(other.base_transformer_)),
      config_(std::move(other.config_)),
      tokenizer_(std::move(other.tokenizer_)),
      world_rank_(other.world_rank_),
      world_size_(other.world_size_),
      local_rank_(other.local_rank_),
      local_size_(other.local_size_),
      mpi_initialized_(other.mpi_initialized_),
      gradient_buffers_(std::move(other.gradient_buffers_)),
      total_parameters_(other.total_parameters_),
      gradients_synchronized_(other.gradients_synchronized_),
      last_sync_time_(other.last_sync_time_),
      total_sync_time_(other.total_sync_time_),
      sync_count_(other.sync_count_) {
    
#ifdef CUDA_AVAILABLE
    compute_stream_ = other.compute_stream_;
    comm_stream_ = other.comm_stream_;
    
#ifdef NCCL_AVAILABLE
    nccl_comm_ = other.nccl_comm_;
    nccl_initialized_ = other.nccl_initialized_;
    
    // Reset other's NCCL state to prevent double cleanup
    other.nccl_initialized_ = false;
#endif
#endif
    
    // Reset other's MPI state to prevent double cleanup
    other.mpi_initialized_ = false;
}

DistributedTransformer& DistributedTransformer::operator=(DistributedTransformer&& other) noexcept {
    if (this != &other) {
        base_transformer_ = std::move(other.base_transformer_);
        config_ = std::move(other.config_);
        tokenizer_ = std::move(other.tokenizer_);
        world_rank_ = other.world_rank_;
        world_size_ = other.world_size_;
        local_rank_ = other.local_rank_;
        local_size_ = other.local_size_;
        mpi_initialized_ = other.mpi_initialized_;
        gradient_buffers_ = std::move(other.gradient_buffers_);
        total_parameters_ = other.total_parameters_;
        gradients_synchronized_ = other.gradients_synchronized_;
        last_sync_time_ = other.last_sync_time_;
        total_sync_time_ = other.total_sync_time_;
        sync_count_ = other.sync_count_;
        
#ifdef CUDA_AVAILABLE
        compute_stream_ = other.compute_stream_;
        comm_stream_ = other.comm_stream_;
        
#ifdef NCCL_AVAILABLE
        nccl_comm_ = other.nccl_comm_;
        nccl_initialized_ = other.nccl_initialized_;
        
        // Reset other's NCCL state
        other.nccl_initialized_ = false;
#endif
#endif
        
        // Reset other's MPI state
        other.mpi_initialized_ = false;
    }
    return *this;
}

void DistributedTransformer::initialize_mpi(int* argc, char*** argv) {
    int provided;
    
    // Check if MPI is already initialized
    int initialized;
    MPI_Initialized(&initialized);
    
    if (!initialized) {
        // Initialize MPI with thread support
        MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided);
        mpi_initialized_ = true;
        
        if (provided < MPI_THREAD_MULTIPLE) {
            std::cerr << "Warning: MPI does not support MPI_THREAD_MULTIPLE" << std::endl;
        }
    }
    
    // Get rank and size information
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
    
    // Determine local rank for GPU assignment
#ifdef CUDA_AVAILABLE
    int device_count;
    cudaGetDeviceCount(&device_count);
    local_rank_ = world_rank_ % device_count;
    local_size_ = device_count;
    
    // Set CUDA device for this rank
    cudaSetDevice(local_rank_);
    
    if (world_rank_ == 0) {
        std::cout << "Found " << device_count << " CUDA devices" << std::endl;
    }
#else
    local_rank_ = world_rank_;
    local_size_ = world_size_;
#endif
}

#ifdef NCCL_AVAILABLE
void DistributedTransformer::initialize_nccl() {
    if (world_size_ <= 1) {
        return; // No need for NCCL with single rank
    }
    
    try {
        // Generate NCCL unique ID on rank 0 and broadcast to all ranks
        ncclUniqueId nccl_id;
        if (world_rank_ == 0) {
            ncclGetUniqueId(&nccl_id);
        }
        
        // Broadcast NCCL ID to all ranks
        MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);
        
        // Initialize NCCL communicator
        ncclResult_t nccl_result = ncclCommInitRank(&nccl_comm_, world_size_, nccl_id, world_rank_);
        if (nccl_result != ncclSuccess) {
            throw std::runtime_error("Failed to initialize NCCL: " + std::string(ncclGetErrorString(nccl_result)));
        }
        
        nccl_initialized_ = true;
        
        if (world_rank_ == 0) {
            std::cout << "NCCL initialized successfully for " << world_size_ << " ranks" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Rank " << world_rank_ << ": Failed to initialize NCCL: " << e.what() << std::endl;
        nccl_initialized_ = false;
    }
}
#endif

void DistributedTransformer::initialize_gradient_buffers() {
    if (world_size_ <= 1) {
        return; // No synchronization needed for single rank
    }
    
    // Get all parameters from base transformer
    auto& parameters = base_transformer_->parameters();
    
    total_parameters_ = 0;
    gradient_buffers_.clear();
    
    for (auto& param : parameters) {
        total_parameters_ += param.size();
        gradient_buffers_.push_back(param.data());
    }
    
    if (world_rank_ == 0) {
        std::cout << "Initialized gradient buffers for " << total_parameters_ 
                  << " parameters across " << gradient_buffers_.size() << " tensors" << std::endl;
    }
}

void DistributedTransformer::backward(const Matrix& logits,
                                    const Matrix& target_distribution,
                                    float learning_rate) {
    // Perform local backward pass using base transformer
    base_transformer_->backward(logits, target_distribution, learning_rate);
    
    // Synchronize gradients across all ranks
    if (world_size_ > 1) {
        synchronize_gradients();
    }
    
    gradients_synchronized_ = true;
}

void DistributedTransformer::synchronize_gradients() {
    if (world_size_ <= 1) {
        return;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
#ifdef NCCL_AVAILABLE
    if (nccl_initialized_) {
        // Use NCCL for GPU-to-GPU communication
        auto& parameters = base_transformer_->parameters();
        
        for (auto& param : parameters) {
            if (param.empty()) continue;
            
            // AllReduce gradients across all ranks
            ncclAllReduce(param.data(), param.data(), param.size(),
                         ncclFloat, ncclSum, nccl_comm_, comm_stream_);
        }
        
        // Wait for communication to complete
        cudaStreamSynchronize(comm_stream_);
        
        // Scale gradients by world size (average instead of sum)
        for (auto& param : parameters) {
            if (param.empty()) continue;
            param *= (1.0f / world_size_);
        }
        
    } else {
#endif
        // Fallback to MPI for CPU communication
        auto& parameters = base_transformer_->parameters();
        
        for (auto& param : parameters) {
            if (param.empty()) continue;
            
            // Create temporary buffer for AllReduce
            std::vector<float> temp_buffer(param.size());
            std::copy(param.data(), param.data() + param.size(), temp_buffer.begin());
            
            // AllReduce using MPI
            MPI_Allreduce(temp_buffer.data(), param.data(), param.size(),
                         MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            
            // Scale by world size
            param *= (1.0f / world_size_);
        }
        
#ifdef NCCL_AVAILABLE
    }
#endif
    
    auto end_time = std::chrono::high_resolution_clock::now();
    last_sync_time_ = std::chrono::duration<double>(end_time - start_time).count();
    total_sync_time_ += last_sync_time_;
    sync_count_++;
    
    if (world_rank_ == 0 && sync_count_ % 100 == 0) {
        std::cout << "Gradient sync #" << sync_count_ 
                  << " completed in " << last_sync_time_ * 1000.0 << "ms" << std::endl;
    }
}

void DistributedTransformer::train(const std::vector<std::vector<int>>& input_tokens,
                                 const std::vector<std::vector<int>>& target_tokens,
                                 size_t num_epochs,
                                 float learning_rate) {
    
    if (input_tokens.size() != target_tokens.size()) {
        throw std::invalid_argument("Input and target token vectors must have the same size");
    }
    
    size_t total_samples = input_tokens.size();
    
    // Split data across ranks
    auto local_inputs = split_batch(input_tokens);
    auto local_targets = split_batch(target_tokens);
    
    if (world_rank_ == 0) {
        std::cout << "Starting distributed training:" << std::endl;
        std::cout << "  Total samples: " << total_samples << std::endl;
        std::cout << "  Local samples per rank: " << local_inputs.size() << std::endl;
        std::cout << "  Epochs: " << num_epochs << std::endl;
        std::cout << "  Learning rate: " << learning_rate << std::endl;
    }
    
    // Train using local data - base transformer handles the training loop
    base_transformer_->train(local_inputs, local_targets, num_epochs, learning_rate);
    
    // Final synchronization to ensure all ranks have the same model
    if (world_size_ > 1) {
        synchronize_gradients();
    }
    
    if (world_rank_ == 0) {
        std::cout << "Distributed training completed!" << std::endl;
        std::cout << "Total gradient synchronizations: " << sync_count_ << std::endl;
        std::cout << "Average sync time: " << get_average_sync_time() * 1000.0 << "ms" << std::endl;
    }
}

template<typename T>
std::vector<T> DistributedTransformer::split_batch(const std::vector<T>& data) const {
    if (world_size_ <= 1) {
        return data; // No splitting needed for single rank
    }
    
    size_t total_size = data.size();
    size_t local_size = get_local_batch_size(total_size);
    size_t start_idx = get_local_start_index(total_size);
    
    std::vector<T> local_data;
    local_data.reserve(local_size);
    
    for (size_t i = 0; i < local_size && (start_idx + i) < total_size; ++i) {
        local_data.push_back(data[start_idx + i]);
    }
    
    return local_data;
}

template<typename T>
void DistributedTransformer::broadcast_vector(std::vector<T>& vec, int root) const {
    // First broadcast the size
    size_t vec_size = vec.size();
    MPI_Bcast(&vec_size, 1, MPI_UNSIGNED_LONG, root, MPI_COMM_WORLD);
    
    // Resize vector on non-root ranks
    if (world_rank_ != root) {
        vec.resize(vec_size);
    }
    
    // Broadcast the data
    if (vec_size > 0) {
        MPI_Bcast(vec.data(), vec_size * sizeof(T), MPI_BYTE, root, MPI_COMM_WORLD);
    }
}

// Explicit template instantiations for common types
template std::vector<std::vector<int>> DistributedTransformer::split_batch(const std::vector<std::vector<int>>&) const;
template std::vector<int> DistributedTransformer::split_batch(const std::vector<int>&) const;
template void DistributedTransformer::broadcast_vector(std::vector<int>&, int) const;
template void DistributedTransformer::broadcast_vector(std::vector<float>&, int) const;

std::unique_ptr<DistributedTransformer> create_distributed_transformer(
    const TransformerConfig& config,
    std::shared_ptr<TiktokenTokenizer> tokenizer,
    int* argc,
    char*** argv) {
    
    return std::make_unique<DistributedTransformer>(config, tokenizer, argc, argv);
}
