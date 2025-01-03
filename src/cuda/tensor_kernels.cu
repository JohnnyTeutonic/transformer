#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../../include/cuda/cuda_utils.cuh"
#include "../../include/cuda/tensor_kernels.cuh"

extern cublasHandle_t cublas_handle;

void launch_tensor_mul(const float* a, const float* b, float* result,
                      int d1, int d2, int d3, int d4, int b_d4) {
    // Treat the tensor multiplication as a batch of matrix multiplications
    // Each batch corresponds to a combination of d1 and d2 indices
    
    const int batch_size = d1 * d2;  // Number of matrix multiplications
    const int m = d3;                // Rows of A
    const int n = b_d4;              // Cols of B
    const int k = d4;                // Cols of A / Rows of B
    
    // Create arrays of pointers for batched operation
    const float** a_array = nullptr;
    const float** b_array = nullptr;
    float** c_array = nullptr;
    
    // Host arrays for storing device pointers
    const float** h_a_array = new const float*[batch_size];
    const float** h_b_array = new const float*[batch_size];
    float** h_c_array = new float*[batch_size];
    
    // Allocate device memory for pointer arrays
    CUDA_CHECK(cudaMalloc(&a_array, batch_size * sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&b_array, batch_size * sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&c_array, batch_size * sizeof(float*)));
    
    // Set up pointers for each batch
    for (int i = 0; i < d1; ++i) {
        for (int j = 0; j < d2; ++j) {
            const int batch_idx = i * d2 + j;
            const size_t offset = (i * d2 * d3 * d4 + j * d3 * d4);
            
            h_a_array[batch_idx] = a + offset;
            h_b_array[batch_idx] = b + offset;
            h_c_array[batch_idx] = result + (i * d2 * d3 * b_d4 + j * d3 * b_d4);
        }
    }
    
    // Copy pointer arrays to device
    CUDA_CHECK(cudaMemcpy(a_array, h_a_array, batch_size * sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_array, h_b_array, batch_size * sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(c_array, h_c_array, batch_size * sizeof(float*), cudaMemcpyHostToDevice));
    
    // Perform batched matrix multiplication
    float alpha = 1.0f;
    float beta = 0.0f;
    
    CUBLAS_CHECK(cublasSgemmBatched(cublas_handle,
                                   CUBLAS_OP_N, CUBLAS_OP_N,
                                   n, m, k,
                                   &alpha,
                                   b_array, n,    // Leading dimension of each B
                                   a_array, k,    // Leading dimension of each A
                                   &beta,
                                   c_array, n,    // Leading dimension of each C
                                   batch_size));  // Number of batches
    
    // Cleanup
    CUDA_CHECK(cudaFree(a_array));
    CUDA_CHECK(cudaFree(b_array));
    CUDA_CHECK(cudaFree(c_array));
    
    delete[] h_a_array;
    delete[] h_b_array;
    delete[] h_c_array;
} 