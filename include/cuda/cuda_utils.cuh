#pragma once
#include <cuda_runtime.h>

// Forward declare Matrix class
class Matrix;

namespace cuda {
    bool is_initialized();
    void initialize_cuda();
    void cleanup_cuda();
    void launch_softmax_kernel(float* scores, int seq_len, cudaStream_t stream);
    
    // Add CUDA host device specifier for Matrix operations
    __host__ Matrix matmul(const Matrix& A, const Matrix& B, Matrix* output = nullptr);
    __host__ void launch_attention_scores(const float* Q, const float* K, float* scores, float scale,
                                int seq_len, int head_dim, cudaStream_t stream);
    __host__ void launch_softmax(float* scores, int seq_len, cudaStream_t stream);
}

// Include Matrix definition after forward declarations
#include "../matrix.hpp"