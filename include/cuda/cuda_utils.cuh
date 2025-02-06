#pragma once
#include "../matrix.hpp"
#include "../vector.hpp"
#include <cuda_runtime.h>

using FloatVector = Vector;

namespace cuda {
    void initialize_cuda();
    void cleanup_cuda();
    
    // Utility functions
    void launch_softmax_kernel(float* scores, int seq_len, cudaStream_t stream);
    void launch_attention_scores(const float* Q, const float* K, float* scores, float scale,
                               int seq_len, int head_dim, cudaStream_t stream);
    void launch_softmax(float* scores, int seq_len, cudaStream_t stream);
    
    // Bias operations
    void add_bias(Matrix& matrix, const FloatVector& bias);
    void compute_bias_gradients(FloatVector& bias_grad, const Matrix& grad);
}