#include "../../include/lm_head.hpp"
#include "../../include/cuda/cuda_check.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace cuda {

__global__ void convert_projection_to_fp16_kernel(
    half* output, const float* input, const unsigned char* active_tokens,
    size_t hidden_size, size_t vocab_size) {
    
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_size * vocab_size && active_tokens[idx / hidden_size]) {
        output[idx] = __float2half(input[idx]);
    }
}

// Add other CUDA kernels needed for OptimizedLanguageModelHead

} // namespace cuda 