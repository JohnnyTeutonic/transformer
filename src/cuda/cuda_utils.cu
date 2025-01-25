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

    Matrix gpu_matmul(const Matrix& A, const Matrix& B) {
        if (!is_initialized()) {
            initialize_cuda();
        }
        printf("GPU matrix multiplication initialized\n");
        // Get dimensions
        const int M = A.rows();
        const int N = B.cols();
        const int K = A.cols();
        printf("Dimensions: %d, %d, %d\n", M, N, K);
        // Create result matrix
        Matrix C(M, N);

        // Set scaling factors
        const float alpha = 1.0f;
        const float beta = 0.0f;
        printf("Scaling factors set\n");
        // Get the global cuBLAS handle
        extern cublasHandle_t cublas_handle;
        printf("cuBLAS handle obtained\n");
        // Perform matrix multiplication
        CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                N, M, K, &alpha,
                                B.get_data(), N,
                                A.get_data(), K,
                                &beta,
                                C.get_data(), N));

        return C;
    }
}

// Move these outside of any namespace
namespace {
    bool cuda_initialized = false;
}

bool is_initialized() {
    return cuda_initialized;
}

bool initialize_cuda() {
    if (!is_initialized()) {
        cudaError_t error = cudaSetDevice(0);
        if (error != cudaSuccess) {
            // Handle error
            return false;
        }
        cuda_initialized = true;
    }
    return true;
}

void cleanup_cuda() {
    if (cuda_initialized) {
        cudaDeviceReset();
        cuda_initialized = false;
    }
}