// Include Matrix definition first
#include "../../include/matrix.hpp"

#ifdef USE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include "../../include/cuda/matrix_ops.cuh"
#include "../../include/cuda/cuda_check.cuh"
#include "../../include/cuda/cuda_utils.cuh"
#include "../../include/cuda/memory_manager.cuh"
#include "../../include/cuda/cuda_init.cuh"

// Forward declare kernels
__global__ void gelu_forward_kernel(float* x, int size);

// Helper functions outside of cuda namespace
static void throw_runtime_error(const char* msg) {
    throw ::std::runtime_error(msg);
}

namespace cuda {
    Matrix matmul(const Matrix& A, const Matrix& B) {
        // Verify dimensions
        if (A.cols() != B.rows()) {
            throw_runtime_error("Matrix multiplication dimension mismatch");
        }
        printf("Matrix multiplication dimensions verified\n");
        // Initialize CUDA if needed
        if (!is_initialized()) {
            initialize_cuda();
        }
        printf("CUDA initialized\n");
        // Create output matrix with correct dimensions
        Matrix C(A.rows(), B.cols());

        // First ensure all matrices are on GPU
        printf("Ensuring matrices are on GPU\n");
        Matrix A_gpu = A.is_cuda() ? A : A.to_gpu();
        Matrix B_gpu = B.is_cuda() ? B : B.to_gpu();
        Matrix C_gpu(C.rows(), C.cols(), nullptr, true);  // Create GPU matrix
        printf("GPU matrices created\n");
        // Get dimensions
        const int M = A.rows();
        const int N = B.cols();
        const int K = A.cols();
        printf("Dimensions: %d, %d, %d\n", M, N, K);
        // Set scaling factors
        const float alpha = 1.0f;
        const float beta = 0.0f;
        printf("Scaling factors set\n");
        // Get raw pointers
        float* d_A = A_gpu.get_data();
        float* d_B = B_gpu.get_data();
        float* d_C = C_gpu.get_data();

        // Get the global cuBLAS handle
        extern cublasHandle_t cublas_handle;
        printf("cuBLAS handle obtained\n");
        // Perform matrix multiplication using cuBLAS
        // Note: cuBLAS uses column-major order, so we compute C = B^T * A^T
        printf("Performing matrix multiplication\n");
        CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                N, M, K, &alpha,
                                d_B, N,  // Leading dimension of B is N
                                d_A, K,  // Leading dimension of A is K
                                &beta,
                                d_C, N));  // Leading dimension of C is N
        printf("Matrix multiplication completed\n");
        return C_gpu;
    }

    void gelu_forward(Matrix& x) {
        if (!x.is_cuda()) {
            x = x.to_gpu();
        }
        
        dim3 block(256);
        dim3 grid((x.size() + block.x - 1) / block.x);
        
        gelu_forward_kernel<<<grid, block>>>(x.get_data(), x.size());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

// Kernel implementations
__global__ void gelu_forward_kernel(float* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        float cdf = 0.5f * (1.0f + tanhf(0.797884f * (val + 0.044715f * val * val * val)));
        x[idx] = val * cdf;
    }
}
#endif // USE_CUDA