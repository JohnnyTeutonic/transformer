#include "../include/lm_head.hpp"
#include "../include/token_constants.hpp"
#include "../include/cuda/matrix_ops.cuh"
#include "../include/cuda/matrix_multiply.cuh"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <random>
#include <deque>
#include <cassert>
#include <vector>
#include <memory>

// Only include CUDA headers if CUDA is available
#ifdef USE_CUDA
#include "/usr/local/cuda/include/cuda_runtime.h"
#include "/usr/local/cuda/include/cublas_v2.h"
#endif

// Add minimum active tokens constant
constexpr size_t MIN_ACTIVE_TOKENS = 1000;  // Reasonable default value

Matrix LanguageModelHead::backward_pass(const Matrix& grad_output, const Matrix& hidden_states) {
    // Verify input dimensions
    if (grad_output.cols() != vocab_size_) {
        throw std::runtime_error("Gradient output dimension mismatch in backward pass");
    }
    if (hidden_states.cols() != hidden_size_) {
        throw std::runtime_error("Hidden states dimension mismatch in backward pass");
    }

    std::cout << "\nLM Head Backward Pass Dimensions:" << std::endl
              << "- grad_output: [" << grad_output.rows() << " x " << grad_output.cols() << "]" << std::endl
              << "- hidden_states: [" << hidden_states.rows() << " x " << hidden_states.cols() << "]" << std::endl
              << "- projection: [" << projection.rows() << " x " << projection.cols() << "]" << std::endl;
    
    // Initialize output gradient with correct dimensions [batch_size x hidden_size]
    Matrix grad_proj(grad_output.rows(), hidden_size_);
    
#ifdef USE_CUDA
    // First declare device pointers
    float *d_grad_output = nullptr, *d_projection = nullptr, *d_grad_proj = nullptr;

    // Then define cleanup lambda using the now-declared pointers
    auto cleanup = [&]() {
        if (d_grad_output) cudaFree(d_grad_output);
        if (d_projection) cudaFree(d_projection);
        if (d_grad_proj) cudaFree(d_grad_proj);
        cudaDeviceSynchronize();
    };
    
    try {
        // Calculate memory sizes
        size_t grad_output_size = grad_output.rows() * grad_output.cols() * sizeof(float);
        size_t projection_size = projection.rows() * projection.cols() * sizeof(float);
        size_t grad_proj_size = grad_proj.rows() * grad_proj.cols() * sizeof(float);
        
        // Print memory requirements
        std::cout << "\nMemory requirements:" << std::endl
                  << "- grad_output: " << grad_output_size << " bytes" << std::endl
                  << "- projection: " << projection_size << " bytes" << std::endl
                  << "- grad_proj: " << grad_proj_size << " bytes" << std::endl;
        
        // Check GPU memory
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        std::cout << "GPU memory before allocation: Free=" << (free_mem/1024/1024) 
                  << "MB, Total=" << (total_mem/1024/1024) << "MB" << std::endl;
        
        // Allocate device memory
        cudaError_t err = cudaMalloc(&d_grad_output, grad_output_size);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate d_grad_output: " + std::string(cudaGetErrorString(err)));
        }
        
        err = cudaMalloc(&d_projection, projection_size);
        if (err != cudaSuccess) {
            cleanup();
            throw std::runtime_error("Failed to allocate d_projection: " + std::string(cudaGetErrorString(err)));
        }
        
        err = cudaMalloc(&d_grad_proj, grad_proj_size);
        if (err != cudaSuccess) {
            cleanup();
            throw std::runtime_error("Failed to allocate d_grad_proj: " + std::string(cudaGetErrorString(err)));
        }
        
        // Copy data to device
        err = cudaMemcpy(d_grad_output, grad_output.data(), grad_output_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cleanup();
            throw std::runtime_error("Failed to copy grad_output to device");
        }
        
        err = cudaMemcpy(d_projection, projection.data(), projection_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cleanup();
            throw std::runtime_error("Failed to copy projection to device");
        }
        
        // Launch our custom matrix multiplication kernel
        err = launch_matrix_multiply(
            d_projection,
            d_grad_output,
            d_grad_proj,
            grad_output.rows(),  // batch_size
            vocab_size_,         // vocab_size
            hidden_size_,        // hidden_size
            nullptr             // use default stream
        );

        if (err != cudaSuccess) {
            cleanup();
            throw std::runtime_error("CUDA matrix multiplication failed: " + std::string(cudaGetErrorString(err)));
        }

        // Synchronize to ensure computation is complete
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            cleanup();
            throw std::runtime_error("CUDA synchronization failed: " + std::string(cudaGetErrorString(err)));
        }
        
        // Copy result back to host
        err = cudaMemcpy(grad_proj.data(), d_grad_proj, grad_proj_size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            cleanup();
            throw std::runtime_error("Failed to copy result back to host");
        }
        
        cleanup();
        return grad_proj;
        
    } catch (const std::exception& e) {
        std::cout << "Exception in backward_pass: " << e.what() << std::endl;
        cleanup();
        throw;
    }
#else
    throw std::runtime_error("CUDA support not enabled");
#endif
}

LanguageModelHead::LanguageModelHead(size_t hidden_size, size_t vocab_size)
    : hidden_size_(hidden_size), vocab_size_(vocab_size), 
      projection(vocab_size, hidden_size),  // Fixed dimensions: [vocab_size x hidden_size]
      bias(vocab_size, 0.0f),  // [vocab_size] stays the same
      token_frequencies(vocab_size, 0.0f),
      pruning_threshold(1e-6f),
      active_tokens(vocab_size, 1),
      training_steps(0),
      is_training_(false),
      m_proj(vocab_size, hidden_size, 0.0f),  // Match projection dimensions
      v_proj(vocab_size, hidden_size, 0.0f),  // Match projection dimensions
      m_bias(vocab_size, 0.0f),
      v_bias(vocab_size, 0.0f),
      t(0),
      beta1(0.9f),
      beta2(0.999f),
      eps(1e-8f),
      current_lr(0.001f),
      min_lr(0.0001f),
      max_lr(0.01f),
      lr_decay(0.99f) {
    
    // Initialize projection matrix with Xavier/Glorot initialization
    float scale = std::sqrt(2.0f / (hidden_size + vocab_size));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, scale);
    
    // Initialize projection weights
    for (size_t i = 0; i < projection.rows(); i++) {
        for (size_t j = 0; j < projection.cols(); j++) {
            projection(i, j) = dist(gen);
        }
    }
    
    std::cout << "Initialized LM Head with:" << std::endl;
    std::cout << "- Hidden size: " << hidden_size << std::endl;
    std::cout << "- Vocab size: " << vocab_size << std::endl;
    std::cout << "- Projection matrix: " << projection.rows() << "x" << projection.cols() << std::endl;
    std::cout << "- Projection matrix shape: [vocab_size x hidden_size] = [" << vocab_size << " x " << hidden_size << "]" << std::endl;
}

Matrix LanguageModelHead::forward(const Matrix& hidden_states, bool training) {
    // Cache hidden states for backward pass
    hidden_states_ = hidden_states;
    
    // Use project_to_vocab for the actual projection
    return project_to_vocab(hidden_states);
}

Matrix LanguageModelHead::project_to_vocab(const Matrix& hidden_states) {
    this->hidden_states = hidden_states;
    
    size_t total_size = hidden_states.rows();
    size_t hidden_dim = hidden_states.cols();
    
    if (hidden_dim != hidden_size_) {
        throw std::runtime_error("Hidden dimension mismatch: " + std::to_string(hidden_dim) +
                               " != " + std::to_string(hidden_size_));
    }
    
    // Project hidden states to vocabulary space using CUDA matrix multiplication
    std::cout << "\nProjecting to vocabulary space:" << std::endl << std::flush;
    std::cout << "Input hidden_states: [" << hidden_states.rows() << " x " << hidden_states.cols() << "]" << std::endl << std::flush;
    std::cout << "Projection matrix: [" << projection.rows() << " x " << projection.cols() << "]" << std::endl << std::flush;
    
    Matrix logits(hidden_states.rows(), vocab_size_);  // [batch_size x vocab_size]
    std::cout << "Allocated logits: [" << logits.rows() << " x " << logits.cols() << "]" << std::endl << std::flush;
    
#ifdef USE_CUDA
    float* d_hidden = nullptr, *d_proj = nullptr, *d_logits = nullptr;
    std::cout << "Starting CUDA operations..." << std::endl << std::flush;
    
    try {
        size_t batch_size = hidden_states.rows();
        
        // Print the actual sizes we're working with
        std::cout << "Matrix dimensions:" << std::endl
                  << "hidden_states: " << hidden_states.rows() << "x" << hidden_states.cols() << std::endl
                  << "projection: " << projection.rows() << "x" << projection.cols() << std::endl
                  << "logits: " << logits.rows() << "x" << logits.cols() << std::endl;

        // Calculate memory sizes with proper casting
        size_t hidden_size = static_cast<size_t>(hidden_states.rows()) * 
                            static_cast<size_t>(hidden_states.cols()) * 
                            sizeof(float);
        
        size_t proj_size = static_cast<size_t>(projection.rows()) * 
                          static_cast<size_t>(projection.cols()) * 
                          sizeof(float);
        
        size_t logits_size = static_cast<size_t>(logits.rows()) * 
                            static_cast<size_t>(logits.cols()) * 
                            sizeof(float);

        // Print allocated sizes
        std::cout << "Allocation sizes:" << std::endl
                  << "hidden_size: " << hidden_size << " bytes" << std::endl
                  << "proj_size: " << proj_size << " bytes" << std::endl
                  << "logits_size: " << logits_size << " bytes" << std::endl;

        // Check if we have enough GPU memory
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        std::cout << "GPU memory available: " << free_mem << " bytes" << std::endl;

        if (hidden_size + proj_size + logits_size > free_mem) {
            throw std::runtime_error("Not enough GPU memory for allocation");
        }

        // Remove duplicate declarations
        std::cout << "Allocating d_hidden..." << std::endl << std::flush;
        cudaError_t err = cudaMalloc(&d_hidden, hidden_size);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate d_hidden: " + std::string(cudaGetErrorString(err)));
        }
        
        std::cout << "Allocating d_proj..." << std::endl << std::flush;
        err = cudaMalloc(&d_proj, proj_size);
        if (err != cudaSuccess) {
            cudaFree(d_hidden);  // Clean up previous allocation
            throw std::runtime_error("Failed to allocate d_proj: " + std::string(cudaGetErrorString(err)));
        }
        
        std::cout << "Allocating d_logits..." << std::endl << std::flush;
        err = cudaMalloc(&d_logits, logits_size);
        if (err != cudaSuccess) {
            cudaFree(d_hidden);  // Clean up previous allocations
            cudaFree(d_proj);
            throw std::runtime_error("Failed to allocate d_logits: " + std::string(cudaGetErrorString(err)));
        }

        // Ensure cuBLAS handle is created before use
        if (!cublas_handle) {
            cublasStatus_t status = cublasCreate(&cublas_handle);
            if (status != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("Failed to create cuBLAS handle");
            }
        }

        // Copy data to device (no transpose needed yet)
        err = cudaMemcpy(d_hidden, hidden_states.data(), hidden_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to copy hidden states to device");
        }

        err = cudaMemcpy(d_proj, projection.data(), proj_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to copy projection to device");
        }

        std::cout << "\n=== Starting cuBLAS Matrix Multiplication ===" << std::endl;
        
        std::cout << "Setting alpha and beta..." << std::endl;
        float alpha = 1.0f;
        float beta = 0.0f;
        std::cout << "Set alpha=" << alpha << " and beta=" << beta << std::endl;

        std::cout << "Calculating leading dimensions..." << std::endl;
        int lda = batch_size;      // Leading dimension of A is now rows of transposed matrix
        std::cout << "Set lda=" << lda << std::endl;
        
        int ldb = hidden_size_;    // Leading dimension of B is now rows of transposed matrix
        std::cout << "Set ldb=" << ldb << std::endl;
        
        int ldc = batch_size;      // Leading dimension of C remains the same
        std::cout << "Set ldc=" << ldc << std::endl;
        
        std::cout << "Verifying cuBLAS handle before operation..." << std::endl;
        if (!cublas_handle) {
            std::cout << "cuBLAS handle is null!" << std::endl;
            throw std::runtime_error("cuBLAS handle is null before matrix multiplication");
        }
        std::cout << "cuBLAS handle verified" << std::endl;

        std::cout << "Verifying device pointers..." << std::endl;
        if (!d_hidden || !d_proj || !d_logits) {
            std::cout << "Device pointers: d_hidden=" << d_hidden << ", d_proj=" << d_proj << ", d_logits=" << d_logits << std::endl;
            throw std::runtime_error("One or more device pointers are null");
        }
        std::cout << "All device pointers verified" << std::endl;

        std::cout << "Starting cublasSgemm call..." << std::endl;
        cublasStatus_t status = cublasSgemm(cublas_handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,     // Now we transpose A and don't transpose B
                    batch_size,                    // M: rows of output
                    vocab_size_,                   // N: cols of output  
                    hidden_size_,                  // K: inner dimension
                    &alpha,
                    d_hidden, lda,                 // Input matrix (transposed)
                    d_proj, ldb,                   // Projection matrix (transposed)
                    &beta,
                    d_logits, ldc);               // Output matrix
        std::cout << "cublasSgemm call completed with status: " << status << std::endl;

        std::cout << "Checking for CUDA errors..." << std::endl;
        cudaError_t cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            std::cout << "CUDA error detected: " << cudaGetErrorString(cuda_status) << std::endl;
        } else {
            std::cout << "No CUDA errors detected" << std::endl;
        }

        std::cout << "Synchronizing CUDA device..." << std::endl;
        cuda_status = cudaDeviceSynchronize();
        if (cuda_status != cudaSuccess) {
            std::cout << "CUDA synchronization failed: " << cudaGetErrorString(cuda_status) << std::endl;
        } else {
            std::cout << "CUDA synchronization successful" << std::endl;
        }

        // Copy result back to host
        cudaMemcpy(logits.data(), d_logits, logits_size, cudaMemcpyDeviceToHost);
        
        std::cout << "Freeing CUDA memory..." << std::endl << std::flush;
        cudaFree(d_hidden);
        cudaFree(d_proj);
        cudaFree(d_logits);
        
        std::cout << "CUDA operations completed successfully" << std::endl << std::flush;
        
    } catch (const std::exception& e) {
        // Clean up on error
        if (d_hidden) cudaFree(d_hidden);
        if (d_proj) cudaFree(d_proj);
        if (d_logits) cudaFree(d_logits);
        throw;
    }
#else
    throw std::runtime_error("CUDA support not enabled");
#endif
    
    // Add bias
    for (size_t i = 0; i < logits.rows(); ++i) {
        for (size_t j = 0; j < logits.cols(); ++j) {
            logits(i, j) += bias[j];
        }
    }
    
    std::cout << "Final logits dimensions: [" << logits.rows() << " x " << logits.cols() << "]" << std::endl;
    return logits;
}

Matrix LanguageModelHead::backward(const Matrix& grad_output, const Matrix& target_distribution) {
    return backward_pass(grad_output, hidden_states);  // Use the existing backward_pass implementation
}

void LanguageModelHead::backward_linear(const Matrix& grad_output) {
    // Use backward_pass which already has Adam optimization
    backward_pass(grad_output, hidden_states_);
}

void LanguageModelHead::update_learning_rate(float current_loss) {
    // Add loss to history
    loss_history.push_back(current_loss);
    if (loss_history.size() > LOSS_HISTORY_SIZE) {
        loss_history.pop_front();
    }
    
    // Only adjust learning rate if we have enough history
    if (loss_history.size() >= 2) {
        float avg_recent_loss = 0.0f;
        float avg_old_loss = 0.0f;
        size_t recent_count = loss_history.size() / 2;
        
        // Calculate average of recent and older losses
        for (size_t i = 0; i < loss_history.size(); i++) {
            if (i >= loss_history.size() - recent_count) {
                avg_recent_loss += loss_history[i];
            } else {
                avg_old_loss += loss_history[i];
            }
        }
        
        if (recent_count > 0) {
        avg_recent_loss /= recent_count;
        }
        
        if (loss_history.size() > recent_count) {
        avg_old_loss /= (loss_history.size() - recent_count);
        }
        
        // Adjust learning rate based on loss trend
        if (avg_recent_loss < avg_old_loss) {
            // Loss is decreasing, increase learning rate slightly
            current_lr = std::min(max_lr, current_lr * lr_growth);
        } else {
            // Loss is increasing or stagnant, decrease learning rate
            current_lr = std::max(min_lr, current_lr * lr_decay);
        }
    }
    
    prev_loss = current_loss;
}

void LanguageModelHead::update_token_frequencies(const std::vector<int>& tokens) {
    // Reset frequencies periodically to prevent over-accumulation
    if (training_steps % 1000 == 0) {  // Reset every 1000 steps
        #pragma omp parallel for
        for (size_t i = 0; i < token_frequencies.size(); i++) {
            token_frequencies[i] = 0.0f;
        }
    }
    
    #pragma omp parallel for
    for (size_t i = 0; i < tokens.size(); i++) {
        int token = tokens[i];
        if (token >= 0 && static_cast<size_t>(token) < vocab_size_) {
            #pragma omp atomic
            token_frequencies[token] += 1.0f;
        }
    }
    training_steps++;
    
    // Normalize frequencies to prevent extreme values
    if (!token_frequencies.empty()) {
        float max_freq = *std::max_element(token_frequencies.begin(), token_frequencies.end());
        if (max_freq > 0) {
            #pragma omp parallel for
            for (size_t i = 0; i < token_frequencies.size(); i++) {
                token_frequencies[i] /= max_freq;  // Normalize to [0,1] range
            }
        }
    }
}

void LanguageModelHead::update_active_tokens() {
    const float decay = 0.99f;
    
    // Parallelize frequency decay
    #pragma omp parallel for
    for (size_t i = 0; i < vocab_size_; i++) {
        token_frequencies[i] *= decay;
    }
    
    // Use vector of pairs to avoid multiple passes
    std::vector<std::pair<float, size_t>> freq_pairs(vocab_size_);
    
    #pragma omp parallel for
    for (size_t i = 0; i < vocab_size_; i++) {
        freq_pairs[i] = {token_frequencies[i], i};
    }
    
    // Partial sort only what we need
    std::partial_sort(freq_pairs.begin(), 
                     freq_pairs.begin() + std::min(MIN_ACTIVE_TOKENS, vocab_size_),
                     freq_pairs.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Reset active tokens
    std::fill(active_tokens.begin(), active_tokens.end(), 0);
    active_token_indices.clear();
    active_token_indices.reserve(std::min(MIN_ACTIVE_TOKENS, vocab_size_));
    
    // Set active tokens based on sorted frequencies
    for (size_t i = 0; i < std::min(MIN_ACTIVE_TOKENS, vocab_size_); i++) {
        size_t idx = freq_pairs[i].second;
        active_tokens[idx] = 1;
        active_token_indices.push_back(idx);
    }
}

#ifdef USE_CUDA
// Add the new GPU kernel for FP16 conversion
__global__ void convert_projection_to_fp16_kernel(
    half* output, const float* input, const unsigned char* active_tokens,
    size_t hidden_size, size_t vocab_size) {
    
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_size * vocab_size && active_tokens[idx / hidden_size]) {
        output[idx] = __float2half(input[idx]);
    }
}
#endif

LanguageModelHead::~LanguageModelHead() {
#ifdef USE_CUDA
    if (cublas_handle) {
        cublasDestroy(cublas_handle);
    }
    if (d_projection) cudaFree(d_projection);
    if (d_bias) cudaFree(d_bias);
    if (d_projection_fp16) cudaFree(d_projection_fp16);
    if (d_hidden_states_fp16) cudaFree(d_hidden_states_fp16);
    if (d_output_fp16) cudaFree(d_output_fp16);
    if (d_output) cudaFree(d_output);
    if (d_active_tokens) cudaFree(d_active_tokens);
    if (d_active_token_indices) cudaFree(d_active_token_indices);
    if (compute_stream) cudaStreamDestroy(compute_stream);
#endif
}

void LanguageModelHead::set_training(bool training_mode) {
    is_training_ = training_mode;
}

void LanguageModelHead::bias_completion_format(Matrix& logits) {
    if (!tokenizer) {
        return;  // Skip biasing if tokenizer is not set
    }

    // Get special token IDs from tokenizer
    const int sep_token_id = tokenizer->get_sep_token_id();
    
    // Get the last predicted token
    int last_token = -1;  // You'll need to track this
    
    // After separator token, boost probability of tokens that commonly start completions
    if (last_token == sep_token_id) {
        // Boost tokens that typically start completions (e.g., space token)
        // This helps enforce the format where completions start with a space
        const float boost_factor = 2.0f;
        for (size_t i = 0; i < logits.rows(); i++) {
            std::string token = tokenizer->decode({static_cast<int>(i)});
            if (!token.empty() && token[0] == ' ') {
                logits.data()[i] *= boost_factor;
            }
        }
    }
} 
