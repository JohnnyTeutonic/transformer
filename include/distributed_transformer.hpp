#pragma once

#include <mpi.h>
#include <memory>
#include <vector>
#include <string>

#ifdef CUDA_AVAILABLE
#include <cuda_runtime.h>
#ifdef NCCL_AVAILABLE
#include <nccl.h>
#endif
#endif

#include "transformer.hpp"
#include "config.hpp"
#include "tiktoken_tokenizer.hpp"
#include "matrix.hpp"

/**
 * @brief Distributed wrapper for Transformer that adds MPI support
 * 
 * Uses composition pattern to wrap existing Transformer without breaking it.
 * Provides seamless distributed training across multiple nodes/GPUs.
 */
class DistributedTransformer {
private:
    // Core components - composition not inheritance
    std::unique_ptr<Transformer> base_transformer_;
    TransformerConfig config_;
    std::shared_ptr<TiktokenTokenizer> tokenizer_;
    
    // MPI state
    int world_rank_ = 0;
    int world_size_ = 1;
    int local_rank_ = 0;
    int local_size_ = 1;
    bool mpi_initialized_ = false;
    
#ifdef CUDA_AVAILABLE
    cudaStream_t compute_stream_;
    cudaStream_t comm_stream_;
#ifdef NCCL_AVAILABLE
    // NCCL for GPU communication
    ncclComm_t nccl_comm_;
    bool nccl_initialized_ = false;
#endif
#endif
    
    // Distributed training state
    std::vector<float*> gradient_buffers_;
    size_t total_parameters_ = 0;
    bool gradients_synchronized_ = false;
    
    // Performance tracking
    double last_sync_time_ = 0.0;
    double total_sync_time_ = 0.0;
    size_t sync_count_ = 0;

public:
    /**
     * @brief Initialize distributed transformer
     * @param config Transformer configuration
     * @param tokenizer Shared tokenizer instance
     * @param argc Command line argument count (for MPI_Init)
     * @param argv Command line arguments (for MPI_Init)
     */
    DistributedTransformer(const TransformerConfig& config, 
                          std::shared_ptr<TiktokenTokenizer> tokenizer,
                          int* argc = nullptr, char*** argv = nullptr);
    
    /**
     * @brief Destructor - handles MPI/NCCL cleanup
     */
    ~DistributedTransformer();
    
    // Disable copy operations to prevent MPI/NCCL handle issues
    DistributedTransformer(const DistributedTransformer&) = delete;
    DistributedTransformer& operator=(const DistributedTransformer&) = delete;
    
    // Enable move operations
    DistributedTransformer(DistributedTransformer&& other) noexcept;
    DistributedTransformer& operator=(DistributedTransformer&& other) noexcept;
    
    /**
     * @brief Forward pass - delegates to base transformer (no MPI needed)
     * @param input_tokens Input token sequence
     * @param original_query Original query string
     * @param tokenizer Tokenizer instance
     * @return Output logits
     */
    Matrix forward(const std::vector<int>& input_tokens, 
                   const std::string& original_query, 
                   const TiktokenTokenizer& tokenizer) {
        return base_transformer_->forward(input_tokens, original_query, tokenizer);
    }
    
    /**
     * @brief Distributed backward pass with gradient synchronization
     * @param logits Output logits from forward pass
     * @param target_distribution Target distribution
     * @param learning_rate Learning rate
     */
    void backward(const Matrix& logits, 
                  const Matrix& target_distribution, 
                  float learning_rate);
    
    /**
     * @brief Distributed training with automatic batch splitting
     * @param input_tokens All input sequences (will be split across ranks)
     * @param target_tokens All target sequences (will be split across ranks)
     * @param num_epochs Number of training epochs
     * @param learning_rate Learning rate
     */
    void train(const std::vector<std::vector<int>>& input_tokens,
               const std::vector<std::vector<int>>& target_tokens,
               size_t num_epochs,
               float learning_rate);
    
    /**
     * @brief Generate text using the distributed model
     * @param input_tokens Initial token sequence
     * @param max_length Maximum generation length
     * @param temperature Sampling temperature
     * @return Generated token sequence
     */
    std::vector<int> generate(const std::vector<int>& input_tokens,
                             size_t max_length = 100,
                             float temperature = 1.0f) {
        // Generation only needs to run on rank 0, broadcast result
        std::vector<int> result;
        if (world_rank_ == 0) {
            result = base_transformer_->generate(input_tokens, max_length, temperature);
        }
        
        // Broadcast result to all ranks
        broadcast_vector(result, 0);
        return result;
    }
    
    /**
     * @brief Save model (only rank 0 saves to prevent conflicts)
     * @param path Path to save model
     */
    void save_model(const std::string& path) const {
        if (world_rank_ == 0) {
            base_transformer_->save_model(path);
        }
        // Barrier to ensure save completes before continuing
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    /**
     * @brief Load model (all ranks load the same model)
     * @param path Path to load model from
     */
    void load_model(const std::string& path) {
        // All ranks load the same model
        *base_transformer_ = Transformer::load_model(path);
        
        // Reinitialize gradient buffers after loading
        initialize_gradient_buffers();
    }
    
    // Getters for distributed info
    int get_world_rank() const { return world_rank_; }
    int get_world_size() const { return world_size_; }
    int get_local_rank() const { return local_rank_; }
    int get_local_size() const { return local_size_; }
    
    // Performance metrics
    double get_average_sync_time() const { 
        return sync_count_ > 0 ? total_sync_time_ / sync_count_ : 0.0; 
    }
    double get_total_sync_time() const { return total_sync_time_; }
    size_t get_sync_count() const { return sync_count_; }
    
    // Access to underlying transformer (for compatibility)
    Transformer* get_base_transformer() { return base_transformer_.get(); }
    const Transformer* get_base_transformer() const { return base_transformer_.get(); }
    
    /**
     * @brief Check if distributed training is properly initialized
     * @return True if MPI and optionally NCCL are ready
     */
    bool is_distributed_ready() const {
        return mpi_initialized_ && world_size_ > 1;
    }

private:
    /**
     * @brief Initialize MPI communication
     * @param argc Command line argument count
     * @param argv Command line arguments
     */
    void initialize_mpi(int* argc, char*** argv);
    
    /**
     * @brief Initialize NCCL for GPU communication
     */
    void initialize_nccl();
    
    /**
     * @brief Initialize gradient buffers for synchronization
     */
    void initialize_gradient_buffers();
    
    /**
     * @brief Synchronize gradients across all ranks
     */
    void synchronize_gradients();
    
    /**
     * @brief Split batch data across ranks
     * @param data Input data to split
     * @return Local portion of data for this rank
     */
    template<typename T>
    std::vector<T> split_batch(const std::vector<T>& data) const;
    
    /**
     * @brief Broadcast vector from root rank to all ranks
     * @param vec Vector to broadcast (modified in-place on non-root ranks)
     * @param root Root rank to broadcast from
     */
    template<typename T>
    void broadcast_vector(std::vector<T>& vec, int root) const;
    
    /**
     * @brief Get local batch size for this rank
     * @param total_batch_size Total batch size across all ranks
     * @return Local batch size for this rank
     */
    size_t get_local_batch_size(size_t total_batch_size) const {
        size_t base_size = total_batch_size / world_size_;
        size_t remainder = total_batch_size % world_size_;
        return base_size + (world_rank_ < remainder ? 1 : 0);
    }
    
    /**
     * @brief Get starting index for this rank's data
     * @param total_batch_size Total batch size across all ranks
     * @return Starting index for this rank
     */
    size_t get_local_start_index(size_t total_batch_size) const {
        size_t base_size = total_batch_size / world_size_;
        size_t remainder = total_batch_size % world_size_;
        return world_rank_ * base_size + std::min(static_cast<size_t>(world_rank_), remainder);
    }
};

/**
 * @brief Helper function to create distributed transformer
 * @param config Transformer configuration  
 * @param tokenizer Tokenizer instance
 * @param argc Command line argument count
 * @param argv Command line arguments
 * @return Unique pointer to distributed transformer
 */
std::unique_ptr<DistributedTransformer> create_distributed_transformer(
    const TransformerConfig& config,
    std::shared_ptr<TiktokenTokenizer> tokenizer,
    int* argc = nullptr,
    char*** argv = nullptr
);
