#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "../../include/cuda/backward_ops.cuh"
#include "../../include/cuda/cuda_check.cuh"
#include "../../include/cuda/cuda_utils.cuh"

namespace cuda {
    // Forward declare kernels
    __global__ void layer_norm_backward_kernel(const float* grad, const float* input,
                                             const float* gamma, float* dx, int batch_size,
                                             int hidden_size, float eps);
    __global__ void gelu_backward_kernel(float* grad_output, const float* input, int size);

    void layer_norm_backward(const Matrix& grad, const Matrix& input, const Matrix& gamma,
                           Matrix& dx, float eps) {
        size_t batch_size = input.rows();
        size_t hidden_size = input.cols();
        
        size_t grad_size = grad.size() * sizeof(float);
        size_t input_size = input.size() * sizeof(float);
        size_t gamma_size = gamma.size() * sizeof(float);

        float *d_grad, *d_input, *d_gamma, *d_dx;
        
        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_grad, grad_size));
        CUDA_CHECK(cudaMalloc(&d_input, input_size));
        CUDA_CHECK(cudaMalloc(&d_gamma, gamma_size));
        CUDA_CHECK(cudaMalloc(&d_dx, grad_size));

        // Copy data to device
        CUDA_CHECK(cudaMemcpy(d_grad, grad.get_data(), grad_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_input, input.get_data(), input_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_gamma, gamma.get_data(), gamma_size, cudaMemcpyHostToDevice));

        // Launch kernel
        dim3 block(256);
        dim3 grid((batch_size * hidden_size + block.x - 1) / block.x);
        
        layer_norm_backward_kernel<<<grid, block>>>(d_grad, d_input, d_gamma, d_dx,
                                                  batch_size, hidden_size, eps);

        // Copy result back to host
        CUDA_CHECK(cudaMemcpy(dx.get_data(), d_dx, grad_size, cudaMemcpyDeviceToHost));

        // Free device memory
        CUDA_CHECK(cudaFree(d_grad));
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_gamma));
        CUDA_CHECK(cudaFree(d_dx));
    }
    
    void gelu_backward(Matrix& grad_output, const Matrix& input) {
        float *d_grad, *d_input;
        size_t size = input.size() * sizeof(float);

        CUDA_CHECK(cudaMalloc(&d_grad, size));
        CUDA_CHECK(cudaMalloc(&d_input, size));

        CUDA_CHECK(cudaMemcpy(d_grad, grad_output.get_data(), size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_input, input.get_data(), size, cudaMemcpyHostToDevice));

        dim3 block(256);
        dim3 grid((input.size() + block.x - 1) / block.x);

        gelu_backward_kernel<<<grid, block>>>(d_grad, d_input, input.size());

        CUDA_CHECK(cudaMemcpy(grad_output.get_data(), d_grad, size, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_grad));
        CUDA_CHECK(cudaFree(d_input));
    }

    __global__ void layer_norm_backward_kernel(const float* grad, const float* input,
                                             const float* gamma, float* dx, int batch_size,
                                             int hidden_size, float eps) {
        extern __shared__ float shared_mem[];
        float* mean = shared_mem;
        float* var = shared_mem + blockDim.x;
        float* sum_grad = shared_mem + 2 * blockDim.x;
        float* sum_grad_diff = shared_mem + 3 * blockDim.x;

        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < batch_size) {
            // Compute mean and variance
            float batch_mean = 0.0f;
            float batch_var = 0.0f;
            
            for (int j = 0; j < hidden_size; ++j) {
                batch_mean += input[tid * hidden_size + j];
            }
            batch_mean /= hidden_size;
            
            for (int j = 0; j < hidden_size; ++j) {
                float diff = input[tid * hidden_size + j] - batch_mean;
                batch_var += diff * diff;
            }
            batch_var /= hidden_size;
            
            float std = sqrtf(batch_var + eps);
            
            // Compute gradients
            float batch_sum_grad = 0.0f;
            float batch_sum_grad_diff = 0.0f;
            
            for (int j = 0; j < hidden_size; ++j) {
                float diff = input[tid * hidden_size + j] - batch_mean;
                batch_sum_grad += grad[tid * hidden_size + j] * gamma[j];
                batch_sum_grad_diff += grad[tid * hidden_size + j] * gamma[j] * diff;
            }
            
            // Compute final gradients
            for (int j = 0; j < hidden_size; ++j) {
                float diff = input[tid * hidden_size + j] - batch_mean;
                dx[tid * hidden_size + j] = gamma[j] * 
                    (grad[tid * hidden_size + j] - 
                     (batch_sum_grad + diff * batch_sum_grad_diff / batch_var) / hidden_size) / std;
            }
        }
    }

    __global__ void gelu_backward_kernel(float* grad_output, const float* input, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            float x = input[idx];
            float cdf = 0.5f * (1.0f + erf(x / sqrtf(2.0f)));
            float pdf = exp(-0.5f * x * x) / sqrtf(2.0f * M_PI);
            grad_output[idx] *= (cdf + x * pdf);
        }
    }
} 