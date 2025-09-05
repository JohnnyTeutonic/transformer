#include "../../include/cuda/fused_attention_kernels.cuh"
#include "../../include/cuda/cuda_check.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace cuda {

__global__ void fused_qkv_projection_kernel(
    const float* input,           // [batch_size, seq_len, hidden_size]
    const float* qkv_weights,     // [hidden_size, 3 * hidden_size] (Q,K,V combined)
    const float* qkv_bias,        // [3 * hidden_size]
    float* qkv_output,            // [batch_size, seq_len, 3 * hidden_size]
    int batch_size,
    int seq_len,
    int hidden_size
) {
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int hidden_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len || hidden_idx >= 3 * hidden_size) {
        return;
    }
    
    float sum = 0.0f;
    int input_offset = batch_idx * seq_len * hidden_size + seq_idx * hidden_size;
    
    // Compute QKV projection in single pass
    for (int i = 0; i < hidden_size; i++) {
        sum += input[input_offset + i] * qkv_weights[i * 3 * hidden_size + hidden_idx];
    }
    
    // Add bias and store
    int output_offset = batch_idx * seq_len * 3 * hidden_size + seq_idx * 3 * hidden_size;
    qkv_output[output_offset + hidden_idx] = sum + qkv_bias[hidden_idx];
}

__global__ void fused_attention_scores_kernel(
    const float* Q,               // [batch_size, num_heads, seq_len, head_dim]
    const float* K,               // [batch_size, num_heads, seq_len, head_dim]
    float* scores,                // [batch_size, num_heads, seq_len, seq_len]
    const float scale,
    const float* mask,            // [seq_len, seq_len] or nullptr
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int q_seq_idx = blockIdx.z;
    int k_seq_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || head_idx >= num_heads || 
        q_seq_idx >= seq_len || k_seq_idx >= seq_len) {
        return;
    }
    
    // Compute attention score
    float score = 0.0f;
    int q_offset = batch_idx * num_heads * seq_len * head_dim + 
                   head_idx * seq_len * head_dim + 
                   q_seq_idx * head_dim;
    int k_offset = batch_idx * num_heads * seq_len * head_dim + 
                   head_idx * seq_len * head_dim + 
                   k_seq_idx * head_dim;
    
    for (int d = 0; d < head_dim; d++) {
        score += Q[q_offset + d] * K[k_offset + d];
    }
    
    score *= scale;
    
    // Apply mask if provided
    if (mask != nullptr) {
        if (mask[q_seq_idx * seq_len + k_seq_idx] == 0.0f) {
            score = -INFINITY;
        }
    }
    
    // Store score
    int score_offset = batch_idx * num_heads * seq_len * seq_len + 
                       head_idx * seq_len * seq_len + 
                       q_seq_idx * seq_len;
    scores[score_offset + k_seq_idx] = score;
}

__global__ void fused_softmax_kernel(
    float* scores,                // [batch_size, num_heads, seq_len, seq_len]
    int batch_size,
    int num_heads,
    int seq_len
) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int seq_idx = blockIdx.z;
    
    if (batch_idx >= batch_size || head_idx >= num_heads || seq_idx >= seq_len) {
        return;
    }
    
    int offset = batch_idx * num_heads * seq_len * seq_len + 
                 head_idx * seq_len * seq_len + 
                 seq_idx * seq_len;
    
    // Find max for numerical stability
    float max_val = -INFINITY;
    for (int i = 0; i < seq_len; i++) {
        max_val = fmaxf(max_val, scores[offset + i]);
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < seq_len; i++) {
        scores[offset + i] = expf(scores[offset + i] - max_val);
        sum += scores[offset + i];
    }
    
    // Normalize
    for (int i = 0; i < seq_len; i++) {
        scores[offset + i] /= sum;
    }
}

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
    cudaStream_t stream
) {
    int head_dim = hidden_size / num_heads;
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    
    // Allocate temporary buffers
    float* d_qkv_output;
    float* d_q, *d_k, *d_v;
    float* d_scores;
    float* d_attention_output;
    
    size_t qkv_size = batch_size * seq_len * 3 * hidden_size * sizeof(float);
    size_t qkv_split_size = batch_size * seq_len * hidden_size * sizeof(float);
    size_t scores_size = batch_size * num_heads * seq_len * seq_len * sizeof(float);
    size_t attention_output_size = batch_size * seq_len * hidden_size * sizeof(float);
    
    CUDA_CHECK(cudaMallocAsync(&d_qkv_output, qkv_size, stream));
    CUDA_CHECK(cudaMallocAsync(&d_q, qkv_split_size, stream));
    CUDA_CHECK(cudaMallocAsync(&d_k, qkv_split_size, stream));
    CUDA_CHECK(cudaMallocAsync(&d_v, qkv_split_size, stream));
    CUDA_CHECK(cudaMallocAsync(&d_scores, scores_size, stream));
    CUDA_CHECK(cudaMallocAsync(&d_attention_output, attention_output_size, stream));
    
    // Step 1: Fused QKV projection
    dim3 qkv_grid(batch_size, seq_len);
    dim3 qkv_block(min(3 * hidden_size, 1024));
    
    fused_qkv_projection_kernel<<<qkv_grid, qkv_block, 0, stream>>>(
        input, qkv_weights, qkv_bias, d_qkv_output,
        batch_size, seq_len, hidden_size
    );
    
    // Step 2: Split QKV and reshape for multi-head attention
    // (This would typically be done with a kernel, simplified here)
    CUDA_CHECK(cudaMemcpyAsync(d_q, d_qkv_output, qkv_split_size, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_k, d_qkv_output + batch_size * seq_len * hidden_size, qkv_split_size, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_v, d_qkv_output + 2 * batch_size * seq_len * hidden_size, qkv_split_size, cudaMemcpyDeviceToDevice, stream));
    
    // Step 3: Compute attention scores
    dim3 scores_grid(batch_size, num_heads, seq_len);
    dim3 scores_block(seq_len);
    
    fused_attention_scores_kernel<<<scores_grid, scores_block, 0, stream>>>(
        d_q, d_k, d_scores, scale, mask,
        batch_size, num_heads, seq_len, head_dim
    );
    
    // Step 4: Apply softmax
    dim3 softmax_grid(batch_size, num_heads, seq_len);
    dim3 softmax_block(1);
    
    fused_softmax_kernel<<<softmax_grid, softmax_block, 0, stream>>>(
        d_scores, batch_size, num_heads, seq_len
    );
    
    // Step 5: Apply attention to values (using cuBLAS for efficiency)
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    cublasSetStream(cublas_handle, stream);
    
    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                hidden_size, batch_size * seq_len, seq_len,
                &alpha,
                d_v, hidden_size,
                d_scores, seq_len,
                &beta,
                d_attention_output, hidden_size);
    
    // Step 6: Output projection
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                hidden_size, batch_size * seq_len, hidden_size,
                &alpha,
                output_weights, hidden_size,
                d_attention_output, hidden_size,
                &beta,
                output, hidden_size);
    
    // Cleanup
    cublasDestroy(cublas_handle);
    CUDA_CHECK(cudaFreeAsync(d_qkv_output, stream));
    CUDA_CHECK(cudaFreeAsync(d_q, stream));
    CUDA_CHECK(cudaFreeAsync(d_k, stream));
    CUDA_CHECK(cudaFreeAsync(d_v, stream));
    CUDA_CHECK(cudaFreeAsync(d_scores, stream));
    CUDA_CHECK(cudaFreeAsync(d_attention_output, stream));
}

} // namespace cuda
