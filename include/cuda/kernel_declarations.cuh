#pragma once
#include <cuda_runtime.h>

// Ensure proper linkage for CUDA kernels
#ifdef __CUDACC__
#define CUDA_KERNEL __global__
#else
#define CUDA_KERNEL
#endif

namespace cuda {
    // Attention kernels
    CUDA_KERNEL void attention_scores_kernel(const float* Q, const float* K, float* scores,
                                           float scale, int seq_len, int head_dim);
    
    CUDA_KERNEL void softmax_kernel(float* matrix, int rows, int cols);
    
    CUDA_KERNEL void attention_kernel(const float* Q, const float* K, const float* V,
                                    float* output, const float* mask,
                                    int batch_size, int seq_len, int head_dim, int hidden_dim);

    CUDA_KERNEL void scaled_dot_product_attention_kernel(
        const float* Q, const float* K, const float* V,
        float* output, const float* mask,
        int batch_size, int num_heads, int seq_len, int head_dim,
        float scale);

    // Feed forward kernels
    CUDA_KERNEL void feed_forward_backward_kernel_1(const float* grad, const float* w2,
                                                  float* d_intermediate, int batch_size,
                                                  int hidden_size, int intermediate_size);

    CUDA_KERNEL void feed_forward_backward_kernel_2(const float* d_intermediate, const float* w1,
                                                  float* dx, int batch_size,
                                                  int hidden_size, int intermediate_size);

    CUDA_KERNEL void gelu_backward_kernel(const float* d_intermediate, float* d_input,
                                        const int num_elements);

    CUDA_KERNEL void gelu_activation_kernel(float* data, int size);

    // Matrix operation kernels
    CUDA_KERNEL void add_bias_kernel(float* output, const float* bias,
                                   int batch_size, int hidden_size);

    CUDA_KERNEL void matrix_multiply_kernel(const float* A, const float* B, float* C,
                                          int M, int N, int K);

    // Beam search kernels
    CUDA_KERNEL void topk_kernel(const float* scores, float* output_scores,
                                int* output_indices, int n, int k);

    CUDA_KERNEL void beam_search_step_kernel(const float* current_scores,
                                           const float* next_scores,
                                           float* output_scores, int* output_indices,
                                           int batch_size, int vocab_size, int beam_width);

    // Tokenizer kernels
    CUDA_KERNEL void parallel_tokenize_kernel(const char* text, size_t text_len,
                                            const char* vocab_data, const int* vocab_lengths,
                                            size_t vocab_size, int* output_tokens,
                                            size_t* positions);
} 