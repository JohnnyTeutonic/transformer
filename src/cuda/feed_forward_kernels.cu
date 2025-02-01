#include "../../include/cuda/feed_forward_kernels.cuh"
#include "../../include/cuda/cuda_check.cuh"
#include "../../include/cuda/matrix_ops.cuh"  // Include this for launch_add_bias
#include "../../include/matrix.hpp"
#include "../../include/feed_forward.hpp"
#include "../../include/cuda/cuda_utils.cuh"
#include "../../include/cuda/kernel_declarations.cuh"
#include <cuda_runtime.h>

// Forward declare kernels
namespace cuda {
    __global__ void add_bias_kernel(float* output, const float* bias,
                                  int batch_size, int hidden_size);
    __global__ void gelu_activation_kernel(float* data, int size);
}

namespace cuda {
    __global__ void feed_forward_backward_kernel_1(const float* grad, const float* w2,
                                                 float* d_intermediate, int batch_size,
                                                 int hidden_size, int intermediate_size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_elements = batch_size * intermediate_size;
        
        if (idx < total_elements) {
            int batch = idx / intermediate_size;
            int inter = idx % intermediate_size;
            
            float sum = 0.0f;
            for (int k = 0; k < hidden_size; ++k) {
                sum += grad[batch * hidden_size + k] * w2[inter * hidden_size + k];
            }
            d_intermediate[idx] = sum;
        }
    }

    __global__ void feed_forward_backward_kernel_2(const float* d_intermediate, const float* w1,
                                                 float* dx, int batch_size,
                                                 int hidden_size, int intermediate_size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_elements = batch_size * hidden_size;
        
        if (idx < total_elements) {
            int batch = idx / hidden_size;
            int hidden = idx % hidden_size;
            
            float sum = 0.0f;
            for (int k = 0; k < intermediate_size; ++k) {
                sum += d_intermediate[batch * intermediate_size + k] * w1[hidden * intermediate_size + k];
            }
            dx[idx] = sum;
        }
    }

    void feed_forward_backward(const Matrix& grad, const Matrix& weights, 
                             Matrix& dx, bool is_first_layer) {
        const int batch_size = grad.rows();
        const int hidden_size = weights.cols();
        const int intermediate_size = weights.rows();
        
        float* d_grad, *d_weights, *d_dx, *d_intermediate;
        size_t grad_size = grad.size() * sizeof(float);
        size_t weights_size = weights.size() * sizeof(float);
        size_t dx_size = dx.size() * sizeof(float);
        size_t intermediate_size_bytes = batch_size * intermediate_size * sizeof(float);
        
        CUDA_CHECK(cudaMalloc(&d_grad, grad_size));
        CUDA_CHECK(cudaMalloc(&d_weights, weights_size));
        CUDA_CHECK(cudaMalloc(&d_dx, dx_size));
        CUDA_CHECK(cudaMalloc(&d_intermediate, intermediate_size_bytes));
        
        CUDA_CHECK(cudaMemcpy(d_grad, grad.data(), grad_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_weights, weights.data(), weights_size, cudaMemcpyHostToDevice));
        
        dim3 block(256);
        dim3 grid((batch_size * intermediate_size + 255) / 256);
        
        feed_forward_backward_kernel_1<<<grid, block>>>(
            d_grad, d_weights, d_intermediate, batch_size, hidden_size, intermediate_size);
            
        if (!is_first_layer) {
            gelu_backward_kernel<<<grid, block>>>(
                d_intermediate, d_dx, batch_size * intermediate_size);
                
            feed_forward_backward_kernel_2<<<grid, block>>>(
                d_intermediate, d_weights, d_dx, batch_size, hidden_size, intermediate_size);
        }
        
        CUDA_CHECK(cudaMemcpy(dx.data(), d_dx, dx_size, cudaMemcpyDeviceToHost));
        
        CUDA_CHECK(cudaFree(d_grad));
        CUDA_CHECK(cudaFree(d_weights));
        CUDA_CHECK(cudaFree(d_dx));
        CUDA_CHECK(cudaFree(d_intermediate));
    }

    __global__ void gelu_backward_kernel(const float* d_intermediate, float* d_input,
                                       const int num_elements) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_elements) {
            float x = d_input[idx];
            float cdf = 0.5f * (1.0f + tanhf(0.797884f * (x + 0.044715f * x * x * x)));
            float pdf = 0.797884f * (1.0f - tanhf(0.797884f * x) * tanhf(0.797884f * x));
            d_input[idx] = d_intermediate[idx] * (cdf + x * pdf);
        }
    }

    __global__ void gelu_activation_kernel(float* data, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            float x = data[idx];
            float x3 = x * x * x;
            constexpr float sqrt_2_pi = 0.7978845608028654f;
            data[idx] = 0.5f * x * (1.0f + tanhf(sqrt_2_pi * (x + 0.044715f * x3)));
        }
    }

    void feed_forward_forward(const Matrix& input, const Matrix& W1, const Matrix& W2,
                            Matrix& intermediate, Matrix& output) {
        // Implementation...
    }
    
    void feed_forward_backward(const Matrix& grad_output, const Matrix& input,
                             const Matrix& W1, const Matrix& W2,
                             Matrix& d_input, Matrix& d_W1, Matrix& d_W2) {
        // Implementation...
    }
} // end namespace cuda

// Move FeedForward::forward_cuda implementation outside of cuda namespace
#ifdef CUDA_AVAILABLE
Matrix FeedForward::forward_cuda(const Matrix& input) {
    const int batch_size = input.rows();
    const int hidden_size = input.cols();
    
    // Get weights and biases using accessor methods
    const Matrix& W1 = get_fc1_weights();
    const Matrix& W2 = get_fc2_weights();
    const Vector& b1 = get_fc1_bias();
    const Vector& b2 = get_fc2_bias();
    
    // First linear layer
    Matrix intermediate(batch_size, W1.cols());  // Create intermediate matrix with correct dimensions
    cuda::matmul(input, W1, intermediate, nullptr);
    
    // Add bias - ensure types match exactly
    cuda::launch_add_bias(intermediate.data(), b1.data(),
                         static_cast<int>(batch_size), 
                         static_cast<int>(intermediate.cols()));
    
    // Apply GELU activation
    dim3 block(256);
    dim3 grid((intermediate.size() + block.x - 1) / block.x);
    
    cuda::gelu_activation_kernel<<<grid, block>>>(
        intermediate.data(),
        static_cast<int>(intermediate.size())
    );
    CUDA_CHECK(cudaGetLastError());
    
    // Second linear layer
    Matrix output(batch_size, W2.cols());  // Create output matrix with correct dimensions
    cuda::matmul(intermediate, W2, output, nullptr);
    
    // Add bias - ensure types match exactly
    cuda::launch_add_bias(output.data(), b2.data(),
                         static_cast<int>(batch_size), 
                         static_cast<int>(output.cols()));
    
    return output;
}
#endif