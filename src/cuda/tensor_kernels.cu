#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../../include/cuda/cuda_utils.cuh"
#include "../../include/cuda/tensor_kernels.cuh"

extern "C" {

__global__ void tensor_mul_kernel(const float* a, const float* b, float* result,
                                int d1, int d2, int d3, int d4, int b_d4) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = d1 * d2 * d3 * d4;
    
    if (idx < total_elements) {
        // Calculate indices for each dimension
        const int i4 = idx % d4;
        const int i3 = (idx / d4) % d3;
        const int i2 = (idx / (d4 * d3)) % d2;
        const int i1 = idx / (d4 * d3 * d2);
        
        // Calculate input indices
        const int a_idx = i1 * (d2 * d3 * d4) + i2 * (d3 * d4) + i3 * d4 + i4;
        const int b_idx = i1 * (d2 * d3 * b_d4) + i2 * (d3 * b_d4) + i3 * b_d4 + i4;
        
        result[idx] = a[a_idx] * b[b_idx];
    }
}

void launch_tensor_mul(const float* a, const float* b, float* result,
                      int d1, int d2, int d3, int d4, int b_d4) {
    const int total_elements = d1 * d2 * d3 * d4;
    const int threads_per_block = 256;
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    tensor_mul_kernel<<<blocks, threads_per_block>>>(a, b, result, d1, d2, d3, d4, b_d4);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

} 