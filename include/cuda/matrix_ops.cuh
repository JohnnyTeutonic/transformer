#pragma once

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#define CUDA_API __host__ __device__
#else
#define CUDA_API
#endif

namespace cuda {
    // Initialize and cleanup
    CUDA_API void initialize_cuda();
    CUDA_API void cleanup_cuda();

    // Matrix operations
    CUDA_API void matmul(const Matrix& A, const Matrix& B, Matrix& C);
    CUDA_API void add(const Matrix& A, const Matrix& B, Matrix& C);
    
    // GELU operations
    CUDA_API void gelu_forward(Matrix& x);
    CUDA_API void gelu_backward(Matrix& grad_output, const Matrix& input);

    // Beam search operations
    CUDA_API void topk(const std::vector<float>& input, Matrix& top_k_values, 
                      std::vector<int>& top_k_indices, int k);
    CUDA_API void beam_search_step(const Matrix& model_output, const Matrix& beam_scores,
                                  Matrix& next_scores, std::vector<int>& next_tokens,
                                  int beam_width);

    // Layer normalization operations
    CUDA_API void layer_norm_forward(const Matrix& input, const Matrix& gamma, const Matrix& beta,
                                   Matrix& output, float eps);
    CUDA_API void layer_norm_backward(const Matrix& grad_output, const Matrix& input,
                                    const Matrix& gamma, Matrix& grad_gamma,
                                    Matrix& grad_beta, float eps);
} 