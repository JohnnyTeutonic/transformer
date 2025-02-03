#pragma once
#include <cuda_runtime.h>

__global__ void update_params_kernel(
    float* param,
    const float* grad,
    float learning_rate,
    float weight_decay,
    float max_relative_change,
    size_t rows,
    size_t cols
);

__global__ void scale_tensor_kernel(
    float* tensor,
    float scale,
    size_t size
); 