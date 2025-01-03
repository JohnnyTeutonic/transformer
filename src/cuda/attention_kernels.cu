#include "../../include/cuda/cuda_utils.cuh"
#include "../../include/attention.hpp"

__global__ void flash_attention_kernel(const float *Q, const float *K,
                                       const float *V, float *output,
                                       const int batch_size,
                                       const int seq_length,
                                       const int head_dim) {
  // Use shared memory for better performance
  extern __shared__ float shared_mem[];
  float* shared_K = shared_mem;
  float* shared_V = &shared_mem[blockDim.x];
  
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int global_idx = bid * blockDim.x + tid;
  
  if (global_idx >= batch_size * seq_length * head_dim)
    return;
        
  const int b = global_idx / (seq_length * head_dim);
  const int s = (global_idx / head_dim) % seq_length;
  const int h = global_idx % head_dim;
  
  // Load query element for this thread
  const float q_val = Q[b * seq_length * head_dim + s * head_dim + h];
  
  float max_val = -INFINITY;
  float sum = 0.0f;
  float denom = 0.0f;
  
  // Process in tiles to maximize shared memory usage
  for (int tile = 0; tile < (seq_length + blockDim.x - 1) / blockDim.x; ++tile) {
    const int tile_start = tile * blockDim.x;
    const int tile_size = min(blockDim.x, seq_length - tile_start);
    
    // Collaboratively load K and V tiles into shared memory
    if (tid < tile_size) {
      const int seq_idx = tile_start + tid;
      shared_K[tid] = K[b * seq_length * head_dim + seq_idx * head_dim + h];
      shared_V[tid] = V[b * seq_length * head_dim + seq_idx * head_dim + h];
    }
    
    __syncthreads();
    
    // First pass: find max for numerical stability
    for (int i = 0; i < tile_size; ++i) {
      float qk = q_val * shared_K[i];
      max_val = max(max_val, qk);
    }
    
    // Second pass: compute attention scores
    for (int i = 0; i < tile_size; ++i) {
      float qk = q_val * shared_K[i];
      float score = __expf(qk - max_val);  // Use fast math
      sum += score * shared_V[i];
      denom += score;
    }
    
    __syncthreads();
  }
  
  // Write final result
  if (denom > 0.0f) {  // Avoid division by zero
    output[global_idx] = sum / denom;
  } else {
    output[global_idx] = 0.0f;
  }
}

Matrix MultiHeadAttention::backward_cuda(const Matrix &grad_output, const Matrix &input) const {
#ifdef USE_CUDA
    // For now, fall back to CPU implementation with empty target distribution
    return backward(grad_output, input, Matrix());
#else
    throw std::runtime_error("CUDA support not enabled");
#endif
}