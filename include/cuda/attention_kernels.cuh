#pragma once
#include <cuda_runtime.h>

namespace cuda {
    // Kernel declarations
    __global__ void attention_scores_kernel(const float* Q, const float* K, float* scores,
                                          float scale, int seq_len, int head_dim);
    __global__ void softmax_kernel(float* scores, int seq_len);
    __global__ void attention_kernel(const float* Q, const float* K, const float* V,
                                   float* output, int batch_size, int num_heads,
                                   int seq_len, int head_dim);
    __global__ void scaled_dot_product_attention_kernel(const float* Q, const float* K,
                                                      const float* V, float* output,
                                                      const float* mask, int batch_size,
                                                      int num_heads, int seq_len,
                                                      int head_dim, float scale);

    // Launch functions
    void launch_attention_scores(const float* Q, const float* K, float* scores,
                               float scale, int seq_len, int head_dim,
                               cudaStream_t stream = nullptr);
    void launch_softmax(float* scores, int seq_len, cudaStream_t stream = nullptr);
    void launch_attention(const float* Q, const float* K, const float* V,
                         float* output, int batch_size, int num_heads,
                         int seq_len, int head_dim, cudaStream_t stream = nullptr);
    void launch_scaled_dot_product_attention(const float* Q, const float* K,
                                           const float* V, float* output,
                                           const float* mask, int batch_size,
                                           int num_heads, int seq_len,
                                           int head_dim, float scale,
                                           cudaStream_t stream = nullptr);
}