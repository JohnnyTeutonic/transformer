#pragma once
#include "../matrix.hpp"
#include <cuda_runtime.h>

namespace cuda {
    // Initialize and cleanup
    void initialize_cuda();
    void cleanup_cuda();

    // Matrix operations
    void add(const Matrix& A, const Matrix& B, Matrix& C);
    
    // Matrix multiplication operations
    Matrix matmul(const Matrix& A, const Matrix& B);  // Returns new matrix
    void matmul(const Matrix& A, const Matrix& B, Matrix& C, cudaStream_t stream = nullptr);  // In-place version with optional stream
    
    // GELU operations
    void gelu_forward(Matrix& x);
    void gelu_backward(Matrix& grad_output, const Matrix& input);

    // Beam search operations
    void topk(const std::vector<float>& input, Matrix& top_k_values, 
             std::vector<int>& top_k_indices, int k);
    void beam_search_step(const Matrix& model_output, const Matrix& beam_scores,
                         Matrix& next_scores, std::vector<int>& next_tokens,
                         int beam_width);

    // Layer normalization operations
    void layer_norm_forward(const Matrix& input, const Matrix& gamma, const Matrix& beta,
                          Matrix& output, float eps);
    void layer_norm_backward(const Matrix& grad_output, const Matrix& input,
                           const Matrix& gamma, Matrix& grad_gamma,
                           Matrix& grad_beta, float eps);

    // Bias operations
    void launch_add_bias(float* output, const float* bias, int batch_size, int hidden_size);
    void launch_row_sum(const float* input, float* output, int rows, int cols);
} 