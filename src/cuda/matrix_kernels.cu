#include "../../include/cuda/cuda_utils.cuh"
#include "../../include/cuda/matrix_kernels.cuh"
#include <cuda_runtime.h>

__global__ void matrix_add_kernel(const float* a, const float* b, float* result,
                                int rows, int cols) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = rows * cols;
    
    if (idx < total_elements) {
        result[idx] = a[idx] + b[idx];
    }
}

__global__ void matrix_sub_kernel(const float* a, const float* b, float* result,
                                int rows, int cols) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = rows * cols;
    
    if (idx < total_elements) {
        result[idx] = a[idx] - b[idx];
    }
}

__global__ void matrix_scalar_mul_kernel(const float* a, float scalar, float* result,
                                       int total_elements) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        result[idx] = a[idx] * scalar;
    }
}

__global__ void matrix_mul_kernel(const float* a, const float* b, float* result,
                                int m, int n, int k) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += a[row * n + i] * b[i * k + col];
        }
        result[row * k + col] = sum;
    }
}

__global__ void matrix_transpose_kernel(const float* a, float* result,
                                      int rows, int cols) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < cols && idy < rows) {
        result[idx * rows + idy] = a[idy * cols + idx];
    }
}

void launch_matrix_add(const float* a, const float* b, float* result,
                      int rows, int cols) {
    const int total_elements = rows * cols;
    const int threads_per_block = 256;
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    matrix_add_kernel<<<blocks, threads_per_block>>>(a, b, result, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void launch_matrix_sub(const float* a, const float* b, float* result,
                      int rows, int cols) {
    const int total_elements = rows * cols;
    const int threads_per_block = 256;
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    matrix_sub_kernel<<<blocks, threads_per_block>>>(a, b, result, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void launch_matrix_scalar_mul(const float* a, float scalar, float* result,
                            int total_elements) {
    const int threads_per_block = 256;
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    matrix_scalar_mul_kernel<<<blocks, threads_per_block>>>(a, scalar, result, total_elements);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void launch_matrix_mul(const float* a, const float* b, float* result,
                      int m, int n, int k) {
    dim3 block_size(16, 16);
    dim3 num_blocks((k + block_size.x - 1) / block_size.x,
                   (m + block_size.y - 1) / block_size.y);
    
    matrix_mul_kernel<<<num_blocks, block_size>>>(a, b, result, m, n, k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void launch_matrix_transpose(const float* a, float* result,
                           int rows, int cols) {
    dim3 block_size(16, 16);
    dim3 num_blocks((cols + block_size.x - 1) / block_size.x,
                   (rows + block_size.y - 1) / block_size.y);
    
    matrix_transpose_kernel<<<num_blocks, block_size>>>(a, result, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
} 