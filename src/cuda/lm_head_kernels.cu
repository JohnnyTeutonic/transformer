#include "../../include/lm_head.hpp"
#include "../../include/cuda/lm_head_kernels.cuh"
#include "../../include/cuda/cuda_check.cuh"

#if defined(USE_CUDA) && defined(CUDA_AVAILABLE)
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace {
// Anonymous namespace for kernel definitions
__global__ void convert_to_fp16_kernel_impl(half* output, const float* input, size_t size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2half(input[idx]);
    }
}

__global__ void convert_and_expand_vocab_kernel_impl(
    float* output, const half* input, const unsigned char* active_tokens,
    size_t batch_size, size_t vocab_size, size_t active_vocab_size) {
    
    const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < batch_size && col < vocab_size) {
        // Find position in compressed vocabulary
        size_t active_pos = 0;
        for (size_t i = 0; i < col; i++) {
            if (active_tokens[i]) {
                active_pos++;
            }
        }
        
        if (active_tokens[col]) {
            output[row * vocab_size + col] = __half2float(input[row * active_vocab_size + active_pos]);
        } else {
            output[row * vocab_size + col] = -INFINITY;
        }
    }
}
} // anonymous namespace

// Device function implementations
__device__ void LanguageModelHead::convert_to_fp16_kernel(
    half* output, const float* input, size_t idx) {
    output[idx] = __float2half(input[idx]);
}

__device__ void LanguageModelHead::convert_and_expand_vocab_kernel(
    float* output, const half* input, const unsigned char* active_tokens,
    size_t row, size_t col, size_t batch_size, size_t vocab_size, size_t active_vocab_size) {
    
    if (row < batch_size && col < vocab_size) {
        size_t active_pos = 0;
        for (size_t i = 0; i < col; i++) {
            if (active_tokens[i]) {
                active_pos++;
            }
        }
        
        if (active_tokens[col]) {
            output[row * vocab_size + col] = __half2float(input[row * active_vocab_size + active_pos]);
        } else {
            output[row * vocab_size + col] = -INFINITY;
        }
    }
}

// Host function implementations
void LanguageModelHead::launch_convert_to_fp16(half* output, const float* input, size_t size) {
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    convert_to_fp16_kernel_impl<<<num_blocks, block_size, 0, compute_stream>>>(output, input, size);
    CUDA_CHECK(cudaGetLastError());
}

void LanguageModelHead::launch_convert_and_expand_vocab(
    float* output, const half* input,
    size_t batch_size, size_t vocab_size, size_t active_vocab_size) {
    
    dim3 block_dim(16, 16);
    dim3 grid_dim(
        (batch_size + block_dim.x - 1) / block_dim.x,
        (vocab_size + block_dim.y - 1) / block_dim.y
    );
    
    convert_and_expand_vocab_kernel_impl<<<grid_dim, block_dim, 0, compute_stream>>>(
        output, input, active_tokens.data(),
        batch_size, vocab_size, active_vocab_size
    );
    CUDA_CHECK(cudaGetLastError());
}

namespace cuda {

// Forward declarations of kernels
__global__ void add_bias_kernel(float* output, const float* bias, int rows, int cols);
__global__ void row_sum_kernel(const float* input, float* output, int rows, int cols);
__global__ void adam_update_kernel(float* params, const float* grads, float* m, float* v,
                                 float beta1, float beta2, float lr, float epsilon, int size);

bool is_available() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    return (error == cudaSuccess && deviceCount > 0);
}

cudaStream_t get_stream() {
    return nullptr; // Returns default stream for now
}

void synchronize() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

void launch_add_bias(float* output, const float* bias, int rows, int cols) {
    dim3 block(256);
    dim3 grid((rows * cols + block.x - 1) / block.x);
    add_bias_kernel<<<grid, block>>>(output, bias, rows, cols);
}

void launch_row_sum(const float* input, float* output, int rows, int cols) {
    dim3 block(256);
    dim3 grid((cols + block.x - 1) / block.x);
    row_sum_kernel<<<grid, block>>>(input, output, rows, cols);
}

void launch_adam_update(float* params, const float* grads, float* m, float* v,
                      float beta1, float beta2, float lr, float epsilon, int size,
                      cudaStream_t stream) {
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    adam_update_kernel<<<grid, block, 0, stream>>>(params, grads, m, v,
                                                  beta1, beta2, lr, epsilon, size);
}

// Kernel implementations
__global__ void add_bias_kernel(float* output, const float* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int col = idx % cols;
        output[idx] += bias[col];
    }
}

__global__ void row_sum_kernel(const float* input, float* output, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        float sum = 0.0f;
        for (int row = 0; row < rows; row++) {
            sum += input[row * cols + col];
        }
        output[col] = sum;
    }
}

__global__ void adam_update_kernel(float* params, const float* grads, float* m, float* v,
                                 float beta1, float beta2, float lr, float epsilon, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Update biased first moment estimate
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * grads[idx];
        
        // Update biased second raw moment estimate
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * grads[idx] * grads[idx];
        
        // Update parameters
        params[idx] -= lr * m[idx] / (sqrtf(v[idx]) + epsilon);
    }
}

} // namespace cuda 

#endif // defined(USE_CUDA) && defined(CUDA_AVAILABLE) 