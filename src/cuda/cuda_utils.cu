#include "../../include/cuda/cuda_utils.cuh"
#include "../../include/cuda/cuda_check.cuh"
#include "../../include/cuda/cuda_init.cuh"
#include "../../include/matrix.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace cuda {
    __global__ void softmax_kernel(float* scores, int seq_len) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < seq_len) {
            // Find max value for numerical stability
            float max_val = scores[row * seq_len];
            for (int i = 1; i < seq_len; i++) {
                max_val = max(max_val, scores[row * seq_len + i]);
            }

            // Compute exp and sum
            float sum = 0.0f;
            for (int i = 0; i < seq_len; i++) {
                scores[row * seq_len + i] = expf(scores[row * seq_len + i] - max_val);
                sum += scores[row * seq_len + i];
            }

            // Normalize
            for (int i = 0; i < seq_len; i++) {
                scores[row * seq_len + i] /= sum;
            }
        }
    }

    void launch_softmax_kernel(float* scores, int seq_len, cudaStream_t stream) {
        if (!is_initialized()) {
            initialize_cuda();
        }

        dim3 block_dim(256);
        dim3 grid_dim((seq_len + block_dim.x - 1) / block_dim.x);

        softmax_kernel<<<grid_dim, block_dim, 0, stream>>>(scores, seq_len);
        CUDA_CHECK(cudaGetLastError());
    }

    Matrix matmul(const Matrix& A, const Matrix& B, Matrix* output) {
        if (!is_initialized()) {
            initialize_cuda();
        }

        // Double check cuBLAS is initialized
        if (cublas_handle == nullptr) {
            throw std::runtime_error("cuBLAS handle is null after initialization");
        }
        
        printf("GPU matrix multiplication initialized\n");
        
        // Get dimensions
        const int M = A.rows();
        const int N = B.cols();
        const int K = A.cols();
        printf("Dimensions: M=%d, N=%d, K=%d\n", M, N, K);
        
        // Verify dimensions
        if (A.cols() != B.rows()) {
            throw std::runtime_error("Matrix dimension mismatch: A.cols() != B.rows()");
        }

        // Use provided output matrix or create new one
        Matrix& C = output ? *output : *(new Matrix(M, N));

        // Transfer matrices to GPU if needed and store the GPU matrices
        Matrix A_gpu = A.is_cuda() ? A : A.to_gpu();
        Matrix B_gpu = B.is_cuda() ? B : B.to_gpu();
        if (!C.is_cuda()) {
            C = C.to_gpu();
        }

        // Verify GPU pointers after transfer
        printf("cuBLAS handle: %p\n", (void*)cublas_handle);
        printf("GPU pointers after transfer - A: %p, B: %p, C: %p\n", 
               (void*)A_gpu.get_data(), (void*)B_gpu.get_data(), (void*)C.get_data());

        if (!A_gpu.get_data() || !B_gpu.get_data() || !C.get_data()) {
            throw std::runtime_error("Invalid GPU pointers after transfer");
        }

        // Set scaling factors
        const float alpha = 1.0f;
        const float beta = 0.0f;
        
        // For row-major matrices A(M,K) and B(K,N), to compute C = A*B in column-major cuBLAS,
        // we compute C^T = B^T * A^T which is equivalent to computing (A*B)^T
        // This means we swap M and N, and swap the order of matrices
        CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                N, M, K, &alpha,
                                B_gpu.get_data(), N,  // Leading dimension is N for column-major B^T
                                A_gpu.get_data(), K,  // Leading dimension is K for column-major A^T
                                &beta,
                                C.get_data(), N));    // Leading dimension is N for column-major C^T
        printf("Matrix multiplication completed\n");

        return C;
    }
}