#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cuda_fp16.h>

namespace cuda {

// CUDA kernel for adding bias to output
__global__ void add_bias_kernel(float* output, const float* bias, 
                              size_t rows, size_t cols);

// CUDA kernel for computing row sums
__global__ void row_sum_kernel(const float* input, float* output,
                             size_t rows, size_t cols);

// CUDA kernel for Adam optimizer update
__global__ void adam_update_kernel(float* param, const float* grad,
                                 float* m, float* v,
                                 float lr, float beta1, float beta2,
                                 float eps, int t, size_t size);

// Host-side wrapper functions
void launch_add_bias(float* output, const float* bias,
                    size_t rows, size_t cols, cudaStream_t stream = nullptr);

void launch_row_sum(const float* input, float* output,
                   size_t rows, size_t cols, cudaStream_t stream = nullptr);

void launch_adam_update(float* param, const float* grad,
                       float* m, float* v,
                       float lr, float beta1, float beta2,
                       float eps, int t, size_t size,
                       cudaStream_t stream = nullptr);

// CUDA utility functions for LM head operations
bool is_available();
cudaStream_t get_stream();
void synchronize();

// Kernel launch functions
void launch_add_bias(float* output, const float* bias, int rows, int cols);
void launch_row_sum(const float* input, float* output, int rows, int cols);
void launch_adam_update(float* params, const float* grads, float* m, float* v,
                      float beta1, float beta2, float lr, float epsilon, int size,
                      cudaStream_t stream);

} // namespace cuda 