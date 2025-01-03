#pragma once
#include <cuda_runtime.h>
#include "cuda_utils.cuh"

// Kernel declarations
__global__ void feed_forward_backward_kernel_1(
    const float* grad_output,
    const float* w2,
    float* d_intermediate,
    size_t batch_size,
    size_t hidden_size,
    size_t intermediate_size
);

__global__ void gelu_backward_kernel(
    const float* d_intermediate,
    float* d_input,
    size_t size
);

__global__ void feed_forward_backward_kernel_2(
    const float* d_intermediate,
    const float* w1,
    float* dx,
    size_t batch_size,
    size_t hidden_size,
    size_t intermediate_size
);

// Add launch wrapper functions
namespace cuda {
    void launch_feed_forward_backward(
        const float* grad_output,
        const float* w2,
        float* d_intermediate,
        float* d_w1,
        float* d_output,
        size_t batch_size,
        size_t hidden_size,
        size_t intermediate_size,
        cudaStream_t stream = nullptr
    );
}