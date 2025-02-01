#include "../../include/feed_forward.hpp"
#include "../include/cuda/cuda_check.cuh"
#include "../include/cuda/cuda_launch.cuh"
#include "../include/cuda/feed_forward_kernels.cuh"
#include "../../include/cuda/cuda_utils.cuh"
#include <cuda_runtime.h>

namespace cuda {

__global__ void gelu_activation_kernel(float* x, int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        const float val = x[idx];
        const float cdf = 0.5f * (1.0f + tanhf(0.797884f * (val + 0.044715f * val * val * val)));
        x[idx] = val * cdf;
    }
}

} // namespace cuda

Matrix FeedForward::backward_cuda(const Matrix& grad, const Matrix& input) const {
    const size_t batch_size = grad.rows();
    const size_t hidden_size = grad.cols();
    const size_t intermediate_size = w1.cols();

    // Allocate device memory
    float *d_grad, *d_w2, *d_intermediate, *d_input, *d_w1, *d_dx;
    CUDA_CHECK(cudaMalloc(&d_grad, batch_size * hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w2, hidden_size * intermediate_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_intermediate, batch_size * intermediate_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_input, batch_size * intermediate_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w1, hidden_size * intermediate_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dx, batch_size * hidden_size * sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_grad, grad.data(), batch_size * hidden_size * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w2, w2.data(), hidden_size * intermediate_size * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Launch kernels
    const int block_size = 256;
    const int grid_size = (batch_size * hidden_size + block_size - 1) / block_size;
    dim3 grid(grid_size);
    dim3 block(block_size);

    // First backward pass through second linear layer
    CUDA_LAUNCH(feed_forward_backward_kernel_1, grid, block, 0, nullptr, d_grad, d_w2,
                d_intermediate, batch_size, hidden_size, intermediate_size);

    // Backward through GELU
    CUDA_LAUNCH(gelu_backward_kernel, grid, block, 0, nullptr, d_intermediate, d_input,
                batch_size * intermediate_size);

    // Second backward pass
    CUDA_LAUNCH(feed_forward_backward_kernel_2, grid, block, 0, nullptr, d_intermediate, d_w1, d_dx,
                batch_size, hidden_size, intermediate_size);

    // Copy result back to host
    Matrix dx(batch_size, hidden_size);
    CUDA_CHECK(cudaMemcpy(dx.data(), d_dx, batch_size * hidden_size * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_grad));
    CUDA_CHECK(cudaFree(d_w2));
    CUDA_CHECK(cudaFree(d_intermediate));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_w1));
    CUDA_CHECK(cudaFree(d_dx));

    return dx;
}

void FeedForward::forward_cuda(const Matrix& input) {
    // First linear layer
    Matrix intermediate = matmul(input, W1);
    
    // Add bias
    cuda::launch_add_bias(intermediate.data(), b1.data(),
                         intermediate.rows(), intermediate.cols(),
                         cuda::get_stream());
    
    // GELU activation
    dim3 block(256);
    dim3 grid((intermediate.size() + block.x - 1) / block.x);
    cuda::gelu_activation_kernel<<<grid, block>>>(
        intermediate.data(), intermediate.size());
    
    // Second linear layer
    Matrix output = matmul(intermediate, W2);
    
    // Add bias
    cuda::launch_add_bias(output.data(), b2.data(),
                         output.rows(), output.cols(),
                         cuda::get_stream());
    
    return output;
}