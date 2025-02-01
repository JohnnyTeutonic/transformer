#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cuda_fp16.h>
#include "cuda_utils.cuh"  // Include this for the CUDA utility functions

namespace cuda {

// CUDA kernel declarations only
__global__ void add_bias_kernel(float* output, const float* bias, 
                              size_t rows, size_t cols);

__global__ void row_sum_kernel(const float* input, float* output,
                             size_t rows, size_t cols);

__global__ void adam_update_kernel(float* param, const float* grad,
                                 float* m, float* v,
                                 float beta1, float beta2, float eps, float lr,
                                 int size, unsigned long step);

} // namespace cuda 