#define USE_CUDA
#include "../../include/cuda/cuda_check.cuh"
#include "../../include/cuda/cuda_utils.cuh"
#include "../../include/layer_norm.hpp"
#include "cuda/layernorm_kernels.cuh"
#include "../../include/cuda/backward_kernels.cuh"
#include <cuda_runtime.h>
#include <iostream>

#ifdef USE_CUDA

namespace cuda {

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
    std::cout << "\n=== LayerNorm Backward Debug ===" << std::endl << std::flush;
    std::cout << "grad_output dims: " << grad_output.rows() << "x" << grad_output.cols() << std::endl << std::flush;
    std::cout << "input dims: " << input.rows() << "x" << input.cols() << std::endl << std::flush;
    std::cout << "gamma size: " << gamma.size() << std::endl;
    std::cout << "grad_gamma size: " << grad_gamma.size() << std::endl;
    std::cout << "grad_beta size: " << grad_beta.size() << std::endl;

    int batch_size = input.rows();
    int hidden_size = input.cols();

    std::cout << "Allocating device memory..." << std::endl;
    // Allocate device memory
    float *d_grad_output, *d_input, *d_gamma;
    float *d_grad_gamma, *d_grad_beta;
    std::cout << "Allocating device memory..." << std::endl;
    std::cout << "batch_size: " << batch_size << ", hidden_size: " << hidden_size << std::endl;
    std::cout << "grad_output size: " << grad_output.size() << std::endl;
    std::cout << "input size: " << input.size() << std::endl;
    std::cout << "gamma size: " << gamma.size() << std::endl;
    std::cout << "grad_gamma size: " << grad_gamma.size() << std::endl;
    std::cout << "grad_beta size: " << grad_beta.size() << std::endl;
    CUDA_CHECK(cudaMalloc(&d_grad_output, batch_size * hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_input, batch_size * hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_gamma, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_beta, hidden_size * sizeof(float)));

    std::cout << "Copying data to device..." << std::endl;
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_grad_output, grad_output.data(), 
                        batch_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), 
                        batch_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, gamma.data(), 
                        hidden_size * sizeof(float), cudaMemcpyHostToDevice));

    std::cout << "Launching kernel..." << std::endl;
    // Launch kernel
    // Use smaller block size to ensure enough blocks for all features
    dim3 block(32, 1);  // 32 threads per block is more reasonable
    // Calculate grid size to cover all features
    int num_blocks = (hidden_size + block.x - 1) / block.x;
    dim3 grid(num_blocks, 1);

    // Calculate shared memory size needed for mean and variance
    // Each block needs space for its own mean and variance arrays
    size_t shared_mem_size = 2 * block.x * sizeof(float);  // 2 arrays of 32 floats each
    
    // Verify shared memory size is sufficient
    size_t max_shared_mem = 48 * 1024;  // 48KB typical limit
    if (shared_mem_size > max_shared_mem) {
        printf("Error: Required shared memory (%zu bytes) exceeds maximum (%zu bytes)\n", 
               shared_mem_size, max_shared_mem);
        return;
    }
    
    // Zero out grad arrays before kernel launch
    CUDA_CHECK(cudaMemset(d_grad_gamma, 0, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grad_beta, 0, hidden_size * sizeof(float)));

    LayerNormBackwardKernel<<<grid, block, shared_mem_size>>>(
        d_grad_output, d_input, d_gamma, d_grad_gamma, batch_size, hidden_size, eps);
    
    // Ensure kernel completion before proceeding
    CUDA_CHECK(cudaDeviceSynchronize());

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Kernel Launch Error: %s\n", cudaGetErrorString(err));
    }

    std::cout << "Copying results back to host..." << std::endl;
    std::cout << "Copying grad_gamma..." << std::endl;
    CUDA_CHECK(cudaMemcpy(grad_gamma.data(), d_grad_gamma, hidden_size * sizeof(float),
                        cudaMemcpyDeviceToHost));
    std::cout << "Copying grad_beta..." << std::endl;
    CUDA_CHECK(cudaMemcpy(grad_beta.data(), d_grad_beta, hidden_size * sizeof(float),
                        cudaMemcpyDeviceToHost));

    std::cout << "Freeing device memory..." << std::endl;
    // Free device memory
    CUDA_CHECK(cudaFree(d_grad_output));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_grad_gamma));
    CUDA_CHECK(cudaFree(d_grad_beta));
    std::cout << "=== LayerNorm Backward Complete ===" << std::endl;
}

__global__ void layer_norm_stats_kernel(const float* input, float* mean, float* variance,
                                      int hidden_size, int batch_size) {
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    // Each thread processes a portion of the hidden dimension
    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;
    
    // Use shared memory for reduction
    extern __shared__ float shared_mem[];
    float* shared_sum = shared_mem;
    float* shared_sq_sum = shared_mem + blockDim.x;
    
    // Each thread processes multiple elements in the hidden dimension
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = input[batch_idx * hidden_size + i];
        local_sum += val;
        local_sq_sum += val * val;
    }
    
    // Store in shared memory
    shared_sum[threadIdx.x] = local_sum;
    shared_sq_sum[threadIdx.x] = local_sq_sum;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
            shared_sq_sum[threadIdx.x] += shared_sq_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // Write results
    if (threadIdx.x == 0) {
        mean[batch_idx] = shared_sum[0] / hidden_size;
        variance[batch_idx] = (shared_sq_sum[0] / hidden_size) - (mean[batch_idx] * mean[batch_idx]);
    }
}

__global__ void layer_norm_kernel(const float* input, const float* mean, const float* variance,
                                const float* gamma, const float* beta, float* output,
                                int hidden_size, int batch_size, float eps) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    // Load mean and variance for this sequence position
    const float mean_val = mean[batch_idx];
    const float var_val = variance[batch_idx];
    const float inv_std = rsqrtf(var_val + eps);
    
    // Normalize each element in the sequence
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        const int idx = batch_idx * hidden_size + i;
        const float normalized = (input[idx] - mean_val) * inv_std;
        output[idx] = gamma[i] * normalized + beta[i];
    }
}

} // namespace cuda

// Define the static GPU memory
namespace {
    struct GPUMemory {
        float *d_input = nullptr;
        float *d_output = nullptr;
        float *d_gamma = nullptr;
        float *d_beta = nullptr;
        float *d_mean = nullptr;
        float *d_variance = nullptr;
        int cached_batch_size = 0;
        int cached_hidden_size = 0;
        
        void free() {
            if (d_input) cudaFree(d_input);
            if (d_output) cudaFree(d_output);
            if (d_gamma) cudaFree(d_gamma);
            if (d_beta) cudaFree(d_beta);
            if (d_mean) cudaFree(d_mean);
            if (d_variance) cudaFree(d_variance);
            d_input = d_output = d_gamma = d_beta = d_mean = d_variance = nullptr;
            cached_batch_size = cached_hidden_size = 0;
        }
        
        ~GPUMemory() {
            free();
        }
    };
    
    GPUMemory gpu_memory;
}

Matrix LayerNorm::forward_cuda(const Matrix& input) {
    int batch_size = input.rows();
    int hidden_size = input.cols();
    
    // Only reallocate if batch size or hidden size changes
    if (batch_size != gpu_memory.cached_batch_size || 
        hidden_size != gpu_memory.cached_hidden_size) {
        // Free old memory
        gpu_memory.free();
        
        // Allocate new memory
        CUDA_CHECK(cudaMalloc(&gpu_memory.d_input, batch_size * hidden_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gpu_memory.d_output, batch_size * hidden_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gpu_memory.d_gamma, hidden_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gpu_memory.d_beta, hidden_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gpu_memory.d_mean, batch_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gpu_memory.d_variance, batch_size * sizeof(float)));
        
        gpu_memory.cached_batch_size = batch_size;
        gpu_memory.cached_hidden_size = hidden_size;
    }
    
    // Create output matrix
    Matrix output(batch_size, hidden_size, 0.0f);
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(gpu_memory.d_input, input.data(), 
                         batch_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_memory.d_gamma, params_.gamma.data(), 
                         hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_memory.d_beta, params_.beta.data(), 
                         hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch kernels
    dim3 stats_grid(batch_size);
    dim3 stats_block(std::min(hidden_size, 256));
    dim3 norm_grid(batch_size);
    dim3 norm_block(std::min(hidden_size, 256));
    
    cuda::layer_norm_stats_kernel<<<stats_grid, stats_block, 2 * stats_block.x * sizeof(float)>>>(
        gpu_memory.d_input, gpu_memory.d_mean, gpu_memory.d_variance,
        hidden_size, batch_size);
    CUDA_CHECK(cudaGetLastError());
    
    cuda::layer_norm_kernel<<<norm_grid, norm_block>>>(
        gpu_memory.d_input, gpu_memory.d_mean, gpu_memory.d_variance,
        gpu_memory.d_gamma, gpu_memory.d_beta, gpu_memory.d_output,
        hidden_size, batch_size, eps_);
    CUDA_CHECK(cudaGetLastError());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(output.data(), gpu_memory.d_output,
                         batch_size * hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    return output;
}

#endif // USE_CUDA