#include "../../include/cuda/attention_kernels.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace cuda {

__global__ void attention_scores_kernel(const float* Q, const float* K, float* scores,
                                        const float scale, int seq_len, int head_dim) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < seq_len && col < seq_len) {
        float sum = 0.0f;
        for (int i = 0; i < head_dim; i++) {
            sum += Q[row * head_dim + i] * K[col * head_dim + i];
        }
        scores[row * seq_len + col] = sum * scale;
    }
}

__global__ void softmax_kernel(float* scores, int seq_len) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < seq_len) {
        // Find max for numerical stability
        float max_val = scores[row * seq_len];
        for (int i = 1; i < seq_len; i++) {
            max_val = max(max_val, scores[row * seq_len + i]);
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < seq_len; i++) {
            scores[row * seq_len + i] = expf(scores[row * seq_len + i] - max_val);
            sum += scores[row * seq_len + i];
        }

        // Normalize
        for (int i = 0; i < seq_len; i++) {
            scores[row * seq_len + i] /= (sum + 1e-6f);  // Add small epsilon for stability
        }
    }
}

__global__ void attention_kernel(const float* Q, const float* K, const float* V,
                               float* output, int batch_size, int num_heads,
                               int seq_len, int head_dim) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (b < batch_size && h < num_heads && i < seq_len) {
        int batch_offset = b * num_heads * seq_len * head_dim;
        int head_offset = h * seq_len * head_dim;
        
        for (int d = 0; d < head_dim; d++) {
            float sum = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                int q_idx = batch_offset + head_offset + i * head_dim + d;
                int k_idx = batch_offset + head_offset + j * head_dim + d;
                int v_idx = batch_offset + head_offset + j * head_dim + d;
                sum += Q[q_idx] * K[k_idx] * V[v_idx];
            }
            output[batch_offset + head_offset + i * head_dim + d] = sum;
        }
    }
}

__global__ void scaled_dot_product_attention_kernel(const float* Q, const float* K,
                                                  const float* V, float* output,
                                                  const float* mask, int batch_size,
                                                  int num_heads, int seq_len,
                                                  int head_dim, float scale) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (b < batch_size && h < num_heads && i < seq_len) {
        int batch_head_offset = (b * num_heads + h) * seq_len * seq_len;
        
        // Compute attention scores
        for (int j = 0; j < seq_len; j++) {
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                int qk_idx = (b * num_heads + h) * seq_len * head_dim + i * head_dim + d;
                score += Q[qk_idx] * K[qk_idx];
            }
            score *= scale;
            
            // Apply mask if provided
            if (mask != nullptr) {
                score += mask[batch_head_offset + i * seq_len + j];
            }
            
            // Store in output temporarily
            output[batch_head_offset + i * seq_len + j] = score;
        }
        
        // Apply softmax
        float max_val = output[batch_head_offset + i * seq_len];
        for (int j = 1; j < seq_len; j++) {
            max_val = max(max_val, output[batch_head_offset + i * seq_len + j]);
        }
        
        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            float val = expf(output[batch_head_offset + i * seq_len + j] - max_val);
            output[batch_head_offset + i * seq_len + j] = val;
            sum += val;
        }
        
        for (int j = 0; j < seq_len; j++) {
            output[batch_head_offset + i * seq_len + j] /= (sum + 1e-6f);
        }
        
        // Compute final output
        for (int d = 0; d < head_dim; d++) {
            float sum = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                sum += output[batch_head_offset + i * seq_len + j] * 
                       V[(b * num_heads + h) * seq_len * head_dim + j * head_dim + d];
            }
            output[(b * num_heads + h) * seq_len * head_dim + i * head_dim + d] = sum;
        }
    }
}

// Launch function implementations
void launch_attention_scores(const float* Q, const float* K, float* scores,
                           float scale, int seq_len, int head_dim,
                           cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((seq_len + block.x - 1) / block.x,
              (seq_len + block.y - 1) / block.y);
    
    attention_scores_kernel<<<grid, block, 0, stream>>>(
        Q, K, scores, scale, seq_len, head_dim);
}

void launch_softmax(float* scores, int seq_len, cudaStream_t stream) {
    dim3 block(256);
    dim3 grid((seq_len + block.x - 1) / block.x);
    
    softmax_kernel<<<grid, block, 0, stream>>>(scores, seq_len);
}

void launch_attention(const float* Q, const float* K, const float* V,
                     float* output, int batch_size, int num_heads,
                     int seq_len, int head_dim, cudaStream_t stream) {
    dim3 block(256);
    dim3 grid(
        (seq_len + block.x - 1) / block.x,
        num_heads,
        batch_size
    );
    
    attention_kernel<<<grid, block, 0, stream>>>(
        Q, K, V, output, batch_size, num_heads, seq_len, head_dim);
}

void launch_scaled_dot_product_attention(const float* Q, const float* K,
                                       const float* V, float* output,
                                       const float* mask, int batch_size,
                                       int num_heads, int seq_len,
                                       int head_dim, float scale,
                                       cudaStream_t stream) {
    dim3 block(256);
    dim3 grid(
        (seq_len + block.x - 1) / block.x,
        num_heads,
        batch_size
    );
    
    scaled_dot_product_attention_kernel<<<grid, block, 0, stream>>>(
        Q, K, V, output, mask, batch_size, num_heads,
        seq_len, head_dim, scale);
}

} // namespace cuda