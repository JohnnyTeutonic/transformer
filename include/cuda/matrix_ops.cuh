#pragma once
#include "../matrix.hpp"

#ifdef _WIN32
    #ifdef BUILD_CUDA_LIB
        #define CUDA_API __declspec(dllexport)
    #else
        #define CUDA_API __declspec(dllimport)
    #endif
#else
    #define CUDA_API
#endif

namespace cuda {
    // Initialize and cleanup
    CUDA_API __host__ void initialize_cuda();
    CUDA_API __host__ void cleanup_cuda();

    // Matrix operations
    CUDA_API __host__ void matmul(const Matrix& A, const Matrix& B, Matrix& C);
    CUDA_API __host__ void add(const Matrix& A, const Matrix& B, Matrix& C);
    
    // GELU operations
    CUDA_API __host__ void gelu_forward(Matrix& x);
    CUDA_API __host__ void gelu_backward(Matrix& grad_output, const Matrix& input);

    // Beam search operations
    CUDA_API __host__ void topk(const std::vector<float>& input, Matrix& top_k_values, 
                              std::vector<int>& top_k_indices, int k);
    CUDA_API __host__ void beam_search_step(const Matrix& model_output, const Matrix& beam_scores,
                                          Matrix& next_scores, std::vector<int>& next_tokens,
                                          int beam_width);

    // Layer normalization operations
    CUDA_API __host__ void layer_norm_forward(const Matrix& input, const Matrix& gamma, const Matrix& beta,
                                           Matrix& output, float eps);
    CUDA_API __host__ void layer_norm_backward(const Matrix& grad_output, const Matrix& input,
                                            const Matrix& gamma, Matrix& grad_gamma,
                                            Matrix& grad_beta, float eps);
} 