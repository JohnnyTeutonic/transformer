#pragma once

#include <cuda_runtime.h>

namespace cuda {

/**
 * Launch fused attention kernel that combines QKV projection, attention computation,
 * and output projection in an optimized pipeline.
 * 
 * @param input Input tensor [batch_size, seq_len, hidden_size]
 * @param qkv_weights Combined Q,K,V weight matrix [hidden_size, 3*hidden_size]
 * @param qkv_bias Combined Q,K,V bias vector [3*hidden_size]
 * @param output_weights Output projection weights [hidden_size, hidden_size]
 * @param output Output tensor [batch_size, seq_len, hidden_size]
 * @param mask Attention mask [seq_len, seq_len] or nullptr
 * @param batch_size Batch size
 * @param seq_len Sequence length
 * @param hidden_size Hidden dimension size
 * @param num_heads Number of attention heads
 * @param stream CUDA stream for async execution
 */
void launch_fused_attention_kernel(
    const float* input,
    const float* qkv_weights,
    const float* qkv_bias,
    const float* output_weights,
    float* output,
    const float* mask,
    int batch_size,
    int seq_len,
    int hidden_size,
    int num_heads,
    cudaStream_t stream = 0
);

} // namespace cuda
