#include "../include/lm_head.hpp"
#include "../include/token_constants.hpp"
#include "../include/cuda/matrix_ops.cuh"  // Add CUDA matrix operations header
#include <cmath>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <random>
#include <deque>
#include <cassert>

// Add minimum active tokens constant
constexpr size_t MIN_ACTIVE_TOKENS = 1000;  // Reasonable default value

// Only include CUDA headers if CUDA is available
#if defined(USE_CUDA) && defined(CUDA_AVAILABLE)
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#endif

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
    if (hidden_states.cols() != hidden_size_) {
        throw std::runtime_error("Hidden dimension mismatch: " + std::to_string(hidden_states.cols()) +
                               " != " + std::to_string(hidden_size_));
    }
    
    // Cache hidden states for backward pass
    hidden_states_ = hidden_states;
    
    // Project hidden states to vocabulary space using CUDA matrix multiplication
    // hidden_states: [batch_size x hidden_size]
    // projection: [vocab_size x hidden_size]
    // result: [batch_size x vocab_size]
    std::cout << "\nLM Head matrix dimensions:" << std::endl;
    std::cout << "hidden_states: " << hidden_states.rows() << "x" << hidden_states.cols() << std::endl;
    std::cout << "projection: " << projection.rows() << "x" << projection.cols() << std::endl;
    
    Matrix logits(hidden_states.rows(), vocab_size_);  // Initialize with correct dimensions
    std::cout << "pre-allocated logits: " << logits.rows() << "x" << logits.cols() << std::endl;
    
    // Use cuBLAS's built-in transpose operation by setting CUBLAS_OP_T
    float* d_hidden, *d_proj, *d_logits;
    size_t hidden_size = hidden_states.rows() * hidden_states.cols() * sizeof(float);
    size_t proj_size = projection.rows() * projection.cols() * sizeof(float);
    size_t logits_size = logits.rows() * logits.cols() * sizeof(float);
    
#ifdef USE_CUDA
    cudaMalloc(&d_hidden, hidden_size);
    cudaMalloc(&d_proj, proj_size);
    cudaMalloc(&d_logits, logits_size);
    
    cudaMemcpy(d_hidden, hidden_states.data(), hidden_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_proj, projection.data(), proj_size, cudaMemcpyHostToDevice);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Ensure we have a valid cuBLAS handle
    if (!cublas_handle) {
        cublasCreate(&cublas_handle);
    }
    
    // Use cuBLAS to perform the matrix multiplication with transpose
    cublasSgemm(cublas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                vocab_size_, hidden_states.rows(), hidden_size_,
                &alpha,
                d_proj, hidden_size_,
                d_hidden, hidden_size_,
                &beta,
                d_logits, vocab_size_);
    
    cudaDeviceSynchronize();
    cudaMemcpy(logits.data(), d_logits, logits_size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_hidden);
    cudaFree(d_proj);
    cudaFree(d_logits);
#else
    throw std::runtime_error("CUDA support not enabled");
#endif
    
    std::cout << "after matmul logits: " << logits.rows() << "x" << logits.cols() << std::endl;
    
    // Add bias
    for (size_t i = 0; i < logits.rows(); ++i) {
        for (size_t j = 0; j < logits.cols(); ++j) {
            logits(i, j) += bias[j];
        }
    }
    
    return logits;
}

Matrix LanguageModelHead::project_to_vocab(const Matrix& hidden_states) {
    this->hidden_states = hidden_states;
    size_t total_size = hidden_states.rows();
    size_t hidden_dim = hidden_states.cols();
    
    if (hidden_dim != hidden_size_) {
        throw std::runtime_error("Hidden dimension mismatch: " + std::to_string(hidden_dim) +
                               " != " + std::to_string(hidden_size_));
    }
    
    return forward(hidden_states, false);
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
        avg_recent_loss /= recent_count;
        avg_old_loss /= (loss_history.size() - recent_count);
        
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
    
    size_t active_count = 0;
    active_token_indices.clear();
    
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

Matrix LanguageModelHead::backward_pass(const Matrix& grad_output, const Matrix& hidden_states) {
    // Verify input dimensions
    if (grad_output.cols() != vocab_size_) {
        throw std::runtime_error("Gradient output dimension mismatch in backward pass. Expected vocab_size: " + 
                               std::to_string(vocab_size_) + ", got: " + std::to_string(grad_output.cols()));
    }
    if (hidden_states.cols() != hidden_size_) {
        throw std::runtime_error("Hidden states dimension mismatch in backward pass. Expected hidden_size: " + 
                               std::to_string(hidden_size_) + ", got: " + std::to_string(hidden_states.cols()));
    }

    std::cout << "\nLM Head Backward Pass Dimensions:" << std::endl;
    std::cout << "- grad_output: [" << grad_output.rows() << " x " << grad_output.cols() << "]" << std::endl;
    std::cout << "- hidden_states: [" << hidden_states.rows() << " x " << hidden_states.cols() << "]" << std::endl;
    std::cout << "- projection: [" << projection.rows() << " x " << projection.cols() << "]" << std::endl;

    // Compute gradients for projection matrix using cuBLAS
    Matrix grad_proj(vocab_size_, hidden_size_);
    
    // Allocate device memory
    float *d_hidden = nullptr, *d_grad_output = nullptr, *d_grad_proj = nullptr;
    
    try {
        // First operation: compute grad_proj
        {
            size_t hidden_size = hidden_states.rows() * hidden_states.cols() * sizeof(float);
            size_t grad_output_size = grad_output.rows() * grad_output.cols() * sizeof(float);
            size_t grad_proj_size = grad_proj.rows() * grad_proj.cols() * sizeof(float);
            
#ifdef USE_CUDA
            // Allocate device memory for first operation
            cudaMalloc(&d_hidden, hidden_size);
            cudaMalloc(&d_grad_output, grad_output_size);
            cudaMalloc(&d_grad_proj, grad_proj_size);
            
            if (!d_hidden || !d_grad_output || !d_grad_proj) {
                throw std::runtime_error("CUDA memory allocation failed");
            }
            
            // Copy data to device
            cudaMemcpy(d_hidden, hidden_states.data(), hidden_size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_grad_output, grad_output.data(), grad_output_size, cudaMemcpyHostToDevice);
            
            float alpha = 1.0f;
            float beta = 0.0f;
            
            // Ensure we have a valid cuBLAS handle
            if (!cublas_handle) {
                cublasCreate(&cublas_handle);
            }
            
            // Compute grad_proj = grad_output.T @ hidden_states
            cublasSgemm(cublas_handle,
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        vocab_size_, hidden_size_, grad_output.rows(),
                        &alpha,
                        d_grad_output, grad_output.cols(),
                        d_hidden, hidden_states.cols(),
                        &beta,
                        d_grad_proj, vocab_size_);
            
            cudaDeviceSynchronize();
            
            // Copy result back to host
            cudaMemcpy(grad_proj.data(), d_grad_proj, grad_proj_size, cudaMemcpyDeviceToHost);
            
            // Free memory from first operation
            cudaFree(d_hidden);
            cudaFree(d_grad_proj);
            // Keep d_grad_output as we need it for the second operation
            d_hidden = nullptr;
            d_grad_proj = nullptr;
#else
            throw std::runtime_error("CUDA support not enabled");
#endif
        }
        
        // Update parameters using Adam optimizer
        t++;  // Increment time step
        
        // Constants for gradient clipping and stability
        const float clip_threshold = 5.0f;
        const float max_allowed_value = 100.0f;
        const float scale_factor = std::sqrt(1.0f / hidden_size_);
        const float max_update = 0.05f * scale_factor;
        
        bool has_unstable_update = false;
        
        // Update projection matrix using Adam optimizer
        #pragma omp parallel for collapse(2) reduction(|:has_unstable_update)
        for (size_t i = 0; i < grad_proj.rows(); ++i) {
            for (size_t j = 0; j < grad_proj.cols(); ++j) {
                if (!std::isfinite(grad_proj(i, j))) {
                    continue;
                }
                
                // Clip gradient
                float clipped_grad = grad_proj(i, j);
                if (std::abs(clipped_grad) > clip_threshold) {
                    clipped_grad = (clipped_grad > 0) ? clip_threshold : -clip_threshold;
                }
                
                // Update first moment estimate
                m_proj(i, j) = beta1 * m_proj(i, j) + (1 - beta1) * clipped_grad;
                
                // Update second moment estimate
                v_proj(i, j) = beta2 * v_proj(i, j) + (1 - beta2) * clipped_grad * clipped_grad;
                
                // Compute bias-corrected first moment estimate
                float m_hat = m_proj(i, j) / (1 - std::pow(beta1, t));
                
                // Compute bias-corrected second moment estimate
                float v_hat = v_proj(i, j) / (1 - std::pow(beta2, t));
                
                // Compute update
                float update = current_lr * m_hat / (std::sqrt(v_hat) + eps);
                
                // Clip update
                if (std::abs(update) > max_update) {
                    update = (update > 0) ? max_update : -max_update;
                }
                
                // Apply update
                projection(i, j) -= update;
                
                // Check for instability
                if (std::abs(projection(i, j)) > max_allowed_value) {
                    has_unstable_update = true;
                }
            }
        }
        
        // Update bias using Adam optimizer
        #pragma omp parallel for reduction(|:has_unstable_update)
        for (size_t i = 0; i < grad_output.row_sum().size(); ++i) {
            if (!std::isfinite(grad_output.row_sum()[i])) {
                continue;
            }
            
            // Clip gradient
            float clipped_grad = grad_output.row_sum()[i];
            if (std::abs(clipped_grad) > clip_threshold) {
                clipped_grad = (clipped_grad > 0) ? clip_threshold : -clip_threshold;
            }
            
            // Update first moment estimate
            m_bias[i] = beta1 * m_bias[i] + (1 - beta1) * clipped_grad;
            
            // Update second moment estimate
            v_bias[i] = beta2 * v_bias[i] + (1 - beta2) * clipped_grad * clipped_grad;
            
            // Compute bias-corrected first moment estimate
            float m_hat = m_bias[i] / (1 - std::pow(beta1, t));
            
            // Compute bias-corrected second moment estimate
            float v_hat = v_bias[i] / (1 - std::pow(beta2, t));
            
            // Compute update
            float update = current_lr * m_hat / (std::sqrt(v_hat) + eps);
            
            // Clip update
            if (std::abs(update) > max_update) {
                update = (update > 0) ? max_update : -max_update;
            }
            
            // Apply update
            bias[i] -= update;
            
            // Check for instability
            if (std::abs(bias[i]) > max_allowed_value) {
                has_unstable_update = true;
            }
        }
        
        if (has_unstable_update) {
            std::cout << "Warning: Unstable updates detected in backward pass" << std::endl;
        }
        
        // Second operation: compute grad_input
        {
            Matrix grad_input(grad_output.rows(), hidden_size_);
            size_t grad_input_size = grad_input.rows() * grad_input.cols() * sizeof(float);
            size_t proj_size = projection.rows() * projection.cols() * sizeof(float);
            
            float *d_proj = nullptr, *d_grad_input = nullptr;
            
#ifdef USE_CUDA
            // Allocate memory for second operation
            cudaMalloc(&d_proj, proj_size);
            cudaMalloc(&d_grad_input, grad_input_size);
            
            if (!d_proj || !d_grad_input) {
                if (d_proj) cudaFree(d_proj);
                if (d_grad_input) cudaFree(d_grad_input);
                if (d_grad_output) cudaFree(d_grad_output);
                throw std::runtime_error("CUDA memory allocation failed");
            }
            
            // Copy projection matrix to device
            cudaMemcpy(d_proj, projection.data(), proj_size, cudaMemcpyHostToDevice);
            
            float alpha = 1.0f;
            float beta = 0.0f;
            
            // Compute grad_input = grad_output @ projection
            cublasSgemm(cublas_handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        grad_output.rows(), hidden_size_, vocab_size_,
                        &alpha,
                        d_grad_output, grad_output.rows(),
                        d_proj, vocab_size_,
                        &beta,
                        d_grad_input, grad_output.rows());
            
            cudaDeviceSynchronize();
            
            // Copy result back to host
            cudaMemcpy(grad_input.data(), d_grad_input, grad_input_size, cudaMemcpyDeviceToHost);
            
            // Cleanup all device memory
            cudaFree(d_grad_output);
            cudaFree(d_proj);
            cudaFree(d_grad_input);
            
            return grad_input;
#else
            throw std::runtime_error("CUDA support not enabled");
#endif
        }
        
    } catch (const std::exception& e) {
        // Clean up on error
        if (d_hidden) cudaFree(d_hidden);
        if (d_grad_output) cudaFree(d_grad_output);
        if (d_grad_proj) cudaFree(d_grad_proj);
        throw;  // Re-throw the exception
    }
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
