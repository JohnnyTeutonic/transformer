#include "../../include/cuda/cuda_utils.cuh"
#include "../../include/cuda/matrix_kernels.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>

// External cuBLAS handle declaration
extern cublasHandle_t cublas_handle;

void launch_matrix_add(const float* a, const float* b, float* result,
                      int rows, int cols) {
    const int total_elements = rows * cols;
    
    // Copy a to result first
    CUDA_CHECK(cudaMemcpy(result, a, total_elements * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Use cublasSaxpy: result = a + b
    float alpha = 1.0f;
    CUBLAS_CHECK(cublasSaxpy(cublas_handle, total_elements,
                            &alpha, b, 1, result, 1));
}

void launch_matrix_sub(const float* a, const float* b, float* result,
                      int rows, int cols) {
    const int total_elements = rows * cols;
    
    // Copy a to result first
    CUDA_CHECK(cudaMemcpy(result, a, total_elements * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Use cublasSaxpy with negative alpha: result = a - b
    float alpha = -1.0f;
    CUBLAS_CHECK(cublasSaxpy(cublas_handle, total_elements,
                            &alpha, b, 1, result, 1));
}

void launch_matrix_scalar_mul(const float* a, float scalar, float* result,
                            int total_elements) {
    // Copy a to result first
    CUDA_CHECK(cudaMemcpy(result, a, total_elements * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Use cublasSscal: result = scalar * a
    CUBLAS_CHECK(cublasSscal(cublas_handle, total_elements,
                            &scalar, result, 1));
}

void launch_matrix_mul(const float* a, const float* b, float* result,
                      int m, int n, int k) {
    // cuBLAS uses column-major order, so we need to transpose the operation
    // C = A * B becomes C^T = B^T * A^T in column-major order
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Perform C = A * B using cuBLAS
    // Note: cuBLAS expects matrices in column-major order, but our matrices are in row-major order
    // So we compute: C^T = B^T * A^T which is equivalent to C = A * B in row-major order
    CUBLAS_CHECK(cublasSgemm(cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            n,    // Number of rows of C
                            m,    // Number of columns of C
                            k,    // Number of columns of A
                            &alpha,
                            b, n,    // Leading dimension of B
                            a, k,    // Leading dimension of A
                            &beta,
                            result, n));  // Leading dimension of C
}

void launch_matrix_transpose(const float* a, float* result,
                           int rows, int cols) {
    // Use cublasSgeam for matrix transpose
    float alpha = 1.0f;
    float beta = 0.0f;
    
    CUBLAS_CHECK(cublasSgeam(cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            rows, cols,
                            &alpha,
                            a, cols,    // Input matrix
                            &beta,
                            nullptr, rows,    // No second input matrix
                            result, rows));   // Output matrix
} 