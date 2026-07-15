#pragma once
#include "../../include/matrix.hpp"
#include <cuda_runtime.h>
#include "kernel_declarations.cuh"

namespace cuda {
    // Attention operation wrappers
    void compute_attention_scores(const Matrix& Q, const Matrix& K, Matrix& scores, float scale, int num_heads = 1);
    void apply_softmax(Matrix& matrix);
    void attention_forward(const Matrix& Q, const Matrix& K, const Matrix& V, 
                         Matrix& output, int batch_size, int num_heads, int seq_len);

    // CUDA kernel launcher
    void launch_attention_scores_kernel(const float* Q, const float* K, float* scores, float scale,
                                      int seq_len, int head_dim, cudaStream_t stream);

    /**
     * @brief Batched multi-head attention using cuBLAS strided batched GEMM.
     * Replaces 512 CPU matmuls with single batched GPU call.
     * 
     * @param Q Query tensor [batch_size * seq_len, hidden_size] (host)
     * @param K Key tensor [batch_size * seq_len, hidden_size] (host)
     * @param V Value tensor [batch_size * seq_len, hidden_size] (host)
     * @param output Output tensor [batch_size * seq_len, hidden_size] (host)
     * @param attn_weights Attention weights [batch_size * num_heads * seq_len, seq_len] (host, for caching)
     * @param batch_size Number of sequences
     * @param seq_len Sequence length
     * @param num_heads Number of attention heads
     * @param head_dim Dimension per head (hidden_size / num_heads)
     * @param scale Attention scale factor (1/sqrt(head_dim))
     */
    void batched_attention_forward(
        const float* h_Q,
        const float* h_K,
        const float* h_V,
        float* h_output,
        float* h_attn_weights,
        int batch_size,
        int seq_len,
        int num_heads,
        int head_dim,
        float scale
    );

    /**
     * @brief Exact attention backward on GPU (cuBLAS), mirroring
     *        batched_attention_forward's layout. Replaces the scalar CPU
     *        exact-backward loops (~40x per-step speedup at d=512 configs).
     *
     * dV_h = P^T @ dOut_h ; dP = dOut_h @ V_h^T ;
     * dS = P .* (dP - rowsum(dP .* P)) ; dQ_h = scale * dS @ K_h ;
     * dK_h = scale * dS^T @ Q_h. Q/K are the post-RoPE caches; the caller
     * un-rotates dQ/dK before the weight-gradient matmuls.
     */
    void batched_attention_backward(
        const float* h_dOut,
        const float* h_attn_weights,
        const float* h_Q,
        const float* h_K,
        const float* h_V,
        float* h_dQ,
        float* h_dK,
        float* h_dV,
        int batch_size,
        int seq_len,
        int num_heads,
        int head_dim,
        float scale
    );
}

// Declare kernels outside namespace
extern "C" {
    CUDA_KERNEL void attention_scores_kernel(const float* Q, const float* K, float* scores,
                                           float scale, int seq_len, int head_dim);
    CUDA_KERNEL void softmax_kernel(float* matrix, int rows, int cols);
    CUDA_KERNEL void attention_kernel(const float* Q, const float* K, const float* V,
                                    float* output, int batch_size, int seq_len, 
                                    int head_dim, int hidden_dim);
} 