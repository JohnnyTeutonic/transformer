#pragma once

#include <cuda_runtime.h>

// Kernel declarations
__global__ void feed_forward_backward_kernel_1(const float* grad, const float* w2, float* d_intermediate,
                                             int batch_size, int hidden_size, int intermediate_size);

__global__ void gelu_backward_kernel(const float* d_intermediate, float* d_input, int size);

__global__ void feed_forward_backward_kernel_2(const float* d_intermediate, const float* w1, float* dx,
                                             int batch_size, int hidden_size, int intermediate_size); 