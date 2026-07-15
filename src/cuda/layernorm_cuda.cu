#define USE_CUDA
#include "../../include/cuda/cuda_check.cuh"
#include "../../include/cuda/cuda_utils.cuh"
#include "../../include/layer_norm.hpp"
#include "../../include/cuda/backward_kernels.cuh"
#include <cuda_runtime.h>

#ifdef USE_CUDA

namespace cuda {

// ============================================================================
// KERNEL DEFINITIONS (must come before functions that use them)
// ============================================================================

__global__ void layer_norm_stats_kernel(const float* input, float* mean, float* variance,
                                       int hidden_size, int batch_size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    float sum = 0.0f;
    float sq_sum = 0.0f;
    
    for (int i = 0; i < hidden_size; ++i) {
        float val = input[idx * hidden_size + i];
        sum += val;
        sq_sum += val * val;
    }
    
    mean[idx] = sum / hidden_size;
    variance[idx] = (sq_sum / hidden_size) - (mean[idx] * mean[idx]);
}

__global__ void layer_norm_kernel(const float* input, const float* mean, const float* variance,
                                 const float* gamma, const float* beta, float* output,
                                 int hidden_size, int batch_size, float eps) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * hidden_size;
    if (idx >= total) return;
    
    const int batch_idx = idx / hidden_size;
    const int feat_idx = idx % hidden_size;
    
    const float mean_val = mean[batch_idx];
    const float var_val = variance[batch_idx];
    const float inv_std = rsqrtf(var_val + eps);
    
    const float normalized = (input[idx] - mean_val) * inv_std;
    output[idx] = gamma[feat_idx] * normalized + beta[feat_idx];
}

__global__ void LayerNormBackwardKernel(
    const float* d_grad_output,
    const float* d_input,
    const float* d_gamma,
    float* d_grad_gamma,
    const int batch_size,
    const int hidden_size,
    const float eps
) {
    // ... kernel implementation ...
}

// ============================================================================
// WRAPPER FUNCTIONS
// ============================================================================

void LayerNormBackwardCUDA(
    const float* d_grad_output,
    const float* d_input,
    const float* d_gamma,
    float* d_grad_gamma,
    const int batch_size,
    const int hidden_size,
    const float eps
) {
    // Calculate grid and block dimensions
    dim3 block(256);
    dim3 grid((batch_size * hidden_size + block.x - 1) / block.x);

    // Launch kernel with correct parameter types
    LayerNormBackwardKernel<<<grid, block>>>(
        d_grad_output,
        d_input,
        d_gamma,
        d_grad_gamma,
        batch_size,
        hidden_size,
        eps
    );
}

void layer_norm_backward(const Matrix& grad_output, const Matrix& input,
                         const Matrix& gamma, Matrix& grad_gamma,
                         Matrix& grad_beta, float eps) {
    int batch_size = input.rows();
    int hidden_size = input.cols();

    float *d_grad_output, *d_input, *d_gamma;
    float *d_grad_gamma, *d_grad_beta;
    
    CUDA_CHECK(cudaMalloc(&d_grad_output, batch_size * hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_input, batch_size * hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_gamma, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_beta, hidden_size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_grad_output, grad_output.data(), 
                        batch_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), 
                        batch_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, gamma.data(), 
                        hidden_size * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((hidden_size + 255) / 256);
    
    CUDA_CHECK(cudaMemset(d_grad_gamma, 0, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grad_beta, 0, hidden_size * sizeof(float)));

    LayerNormBackwardKernel<<<grid, block>>>(
        d_grad_output, d_input, d_gamma, d_grad_gamma, batch_size, hidden_size, eps);
    
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(grad_gamma.data(), d_grad_gamma, hidden_size * sizeof(float),
                        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(grad_beta.data(), d_grad_beta, hidden_size * sizeof(float),
                        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_grad_output));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_grad_gamma));
    CUDA_CHECK(cudaFree(d_grad_beta));
}

void layer_norm_forward(const Matrix& input, const Matrix& gamma, const Matrix& beta,
                        Matrix& output, float eps) {
    const int batch_size = input.rows();
    const int hidden_size = input.cols();
    
    float* d_input, *d_gamma, *d_beta, *d_output;
    float* d_mean, *d_variance;
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_input, input.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, gamma.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta, beta.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, output.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean, batch_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_variance, batch_size * sizeof(float)));
    
    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, gamma.data(), gamma.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, beta.data(), beta.size() * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch kernels
    const int block_size = 256;
    const int grid_size = (batch_size + block_size - 1) / block_size;
    
    // First compute mean and variance
    size_t shared_mem_size = 2 * block_size * sizeof(float);  // For sum and squared sum
    layer_norm_stats_kernel<<<grid_size, block_size, shared_mem_size>>>(
        d_input, d_mean, d_variance, hidden_size, batch_size);
    
    // Then normalize using the computed statistics
    const int total_elements = batch_size * hidden_size;
    const int norm_grid_size = (total_elements + block_size - 1) / block_size;
    layer_norm_kernel<<<norm_grid_size, block_size>>>(
        d_input, d_mean, d_variance, d_gamma, d_beta, d_output,
        hidden_size, batch_size, eps);
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_mean));
    CUDA_CHECK(cudaFree(d_variance));
}

} // namespace cuda

#endif // USE_CUDA