#include "../../include/attention.hpp"
#include "../../include/cuda/matrix_ops.cuh"
#include "../../include/cuda/cuda_utils.cuh"
#include "../../include/cuda/cuda_check.cuh"
#include "../../include/cuda/attention_ops.cuh"
#include <cuda_runtime.h>

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
        static_cast<int>(batch_size), 
        static_cast<int>(num_heads_), 
        static_cast<int>(seq_len), 
        static_cast<int>(head_dim_),
        scale, cuda::get_stream());
    
    // Project output
    Matrix output_proj(output.rows(), params_.output_weights.cols());
    cuda::matmul(output, params_.output_weights, output_proj, nullptr);
    return output_proj;
} 