#include "../../include/attention.hpp"
#include "../../include/cuda/matrix_ops.cuh"
#include "../../include/cuda/cuda_utils.cuh"
#include "../../include/cuda/cuda_check.cuh"
#include "../../include/cuda/attention_ops.cuh"
#include <cuda_runtime.h>

namespace cuda {

__global__ void scaled_dot_product_attention_kernel(
    const float* Q, const float* K, const float* V,
    float* output, const float* mask,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale) {
    
    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (b < batch_size && h < num_heads && i < seq_len) {
        // Calculate attention scores
        float scores[MAX_SEQ_LEN];  // Assume MAX_SEQ_LEN is defined
        
        for (int j = 0; j < seq_len; j++) {
            float sum = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                const int q_idx = ((b * num_heads + h) * seq_len + i) * head_dim + d;
                const int k_idx = ((b * num_heads + h) * seq_len + j) * head_dim + d;
                sum += Q[q_idx] * K[k_idx];
            }
            scores[j] = sum * scale;
            
            // Apply mask if provided
            if (mask != nullptr) {
                scores[j] += mask[i * seq_len + j];
            }
        }
        
        // Softmax
        float max_score = scores[0];
        for (int j = 1; j < seq_len; j++) {
            max_score = max(max_score, scores[j]);
        }
        
        float exp_sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            scores[j] = expf(scores[j] - max_score);
            exp_sum += scores[j];
        }
        
        for (int j = 0; j < seq_len; j++) {
            scores[j] /= exp_sum;
        }
        
        // Multiply with values
        for (int d = 0; d < head_dim; d++) {
            float sum = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                const int v_idx = ((b * num_heads + h) * seq_len + j) * head_dim + d;
                sum += scores[j] * V[v_idx];
            }
            const int out_idx = ((b * num_heads + h) * seq_len + i) * head_dim + d;
            output[out_idx] = sum;
        }
    }
}

} // namespace cuda

Matrix MultiHeadAttention::forward_cuda(const Matrix& input, 
                                      const AttentionMask& mask,
                                      const std::optional<KVCache>& kv_cache) {
    const int batch_size = input.rows();
    const int seq_len = input.cols() / hidden_size_;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    
    // Project input to Q, K, V spaces
    Matrix Q(input.rows(), params_.query_weights.cols());
    cuda::matmul(input, params_.query_weights, Q, nullptr);
    
    Matrix K;
    if (kv_cache) {
        K = kv_cache->get_key();
    } else {
        K = Matrix(input.rows(), params_.key_weights.cols());
        cuda::matmul(input, params_.key_weights, K, nullptr);
    }
    
    Matrix V;
    if (kv_cache) {
        V = kv_cache->get_value();
    } else {
        V = Matrix(input.rows(), params_.value_weights.cols());
        cuda::matmul(input, params_.value_weights, V, nullptr);
    }
    
    // Allocate output
    Matrix output(batch_size * seq_len, hidden_size_);
    
    // Launch attention kernel
    cuda::launch_attention_kernel(
        Q.data(), K.data(), V.data(),
        output.data(), mask.value().data(),
        batch_size, num_heads_, seq_len, head_dim_,
        scale, cuda::get_stream());
    
    // Project output
    Matrix output_proj(output.rows(), params_.output_weights.cols());
    cuda::matmul(output, params_.output_weights, output_proj, nullptr);
    return output_proj;
} 