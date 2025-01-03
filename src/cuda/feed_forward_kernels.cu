#include "cuda/feed_forward_kernels.cuh"

__global__ void feed_forward_backward_kernel_1(
    const float* grad_output,
    const float* w2,
    float* d_intermediate,
    size_t batch_size,
    size_t hidden_size,
    size_t intermediate_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * intermediate_size) {
        int batch_idx = idx / intermediate_size;
        int inter_idx = idx % intermediate_size;
        
        float sum = 0.0f;
        for (int i = 0; i < hidden_size; ++i) {
            sum += grad_output[batch_idx * hidden_size + i] * 
                   w2[inter_idx * hidden_size + i];
        }
        d_intermediate[idx] = sum;
    }
}

__global__ void gelu_backward_kernel(
    const float* d_intermediate,
    float* d_input,
    size_t size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = d_input[idx];
        // GELU derivative
        float cdf = 0.5f * (1.0f + tanhf((0.797884f * (x + 0.044715f * x * x * x))));
        float pdf = 0.797884f * (1.0f - tanhf((0.797884f * (x + 0.044715f * x * x * x))) * 
                   tanhf((0.797884f * (x + 0.044715f * x * x * x)))) * 
                   (1.0f + 3.0f * 0.044715f * x * x);
        d_input[idx] = d_intermediate[idx] * (cdf + x * pdf);
    }
}

__global__ void feed_forward_backward_kernel_2(
    const float* d_intermediate,
    const float* w1,
    float* dx,
    size_t batch_size,
    size_t hidden_size,
    size_t intermediate_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * hidden_size) {
        int batch_idx = idx / hidden_size;
        int hidden_idx = idx % hidden_size;
        
        float sum = 0.0f;
        for (int i = 0; i < intermediate_size; ++i) {
            sum += d_intermediate[batch_idx * intermediate_size + i] * 
                   w1[hidden_idx * intermediate_size + i];
        }
        dx[idx] = sum;
    }
}

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
        cudaStream_t stream
    ) {
        const int block_size = 256;
        const int grid_size_1 = (batch_size * intermediate_size + block_size - 1) / block_size;
        const int grid_size_2 = (batch_size * hidden_size + block_size - 1) / block_size;

        feed_forward_backward_kernel_1<<<grid_size_1, block_size, 0, stream>>>(
            grad_output, w2, d_intermediate, batch_size, hidden_size, intermediate_size);
        
        gelu_backward_kernel<<<grid_size_1, block_size, 0, stream>>>(
            d_intermediate, d_intermediate, batch_size * intermediate_size);
        
        feed_forward_backward_kernel_2<<<grid_size_2, block_size, 0, stream>>>(
            d_intermediate, d_w1, d_output, batch_size, hidden_size, intermediate_size);
    }
}