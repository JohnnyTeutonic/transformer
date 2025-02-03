#include "../../include/cuda/parameter_update_kernels.cuh"

__global__ void update_params_kernel(
    float* param,
    const float* grad,
    float learning_rate,
    float weight_decay,
    float max_relative_change,
    size_t rows,
    size_t cols
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int total_elements = rows * cols;

    for (int i = idx; i < total_elements; i += stride) {
        float param_val = param[i];
        float grad_val = grad[i];
        float decay = weight_decay * param_val;
        
        // Compute update with weight decay
        float update = grad_val * learning_rate + decay;
        
        // Limit maximum parameter change
        float param_scale = fabsf(param_val) + 1e-8f;
        float max_update = max_relative_change * param_scale;
        update = fmaxf(-max_update, fminf(max_update, update));
        
        // Apply update
        param_val -= update;
        
        // Clip values
        param_val = fmaxf(-100.0f, fminf(100.0f, param_val));
        
        param[i] = param_val;
    }
}

__global__ void scale_tensor_kernel(
    float* tensor,
    float scale,
    size_t size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size; i += stride) {
        tensor[i] *= scale;
    }
} 