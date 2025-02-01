#pragma once
#include "../matrix.hpp"
#include <cuda_runtime.h>

namespace cuda {
    // Attention operation wrappers
    void compute_attention_scores(const Matrix& Q, const Matrix& K, Matrix& scores, float scale, int num_heads = 1);
    void apply_softmax(Matrix& matrix);
    
    void launch_attention_kernel(const float* Q, const float* K, const float* V,
                               float* output, const float* mask,
                               int batch_size, int num_heads, int seq_len, int head_dim,
                               float scale, cudaStream_t stream);
} 