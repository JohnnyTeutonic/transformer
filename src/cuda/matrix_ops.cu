#ifdef USE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "../../include/cuda/matrix_ops.cuh"
#include "../../include/cuda/cuda_check.cuh"
#include "../../include/cuda/cuda_utils.cuh"
#include "../../include/cuda/memory_manager.cuh"

// Forward declare all kernels
__global__ void matrix_multiply_kernel(const float* A, const float* B, float* C, 
                                      int M, int N, int K);
__global__ void gelu_forward_kernel(float* x, int size);

// Global cuBLAS handle
static cublasHandle_t cublas_handle;
static bool cublas_initialized = false;

namespace cuda {
    __host__ void initialize_cuda() {
        CUDA_CHECK(cudaSetDevice(0));
        if (!cublas_initialized) {
            CUBLAS_CHECK(cublasCreate(&cublas_handle));
            cublas_initialized = true;
        }
    }

    __host__ void cleanup_cuda() {
        if (cublas_initialized) {
            CUBLAS_CHECK(cublasDestroy(cublas_handle));
            cublas_initialized = false;
        }
    }

    __host__ void matmul(const Matrix& A, const Matrix& B, Matrix& C) {
        // Verify dimensions
        if (A.cols() != B.rows()) {
            throw std::runtime_error("Matrix multiplication dimension mismatch: " +
                std::to_string(A.rows()) + "x" + std::to_string(A.cols()) + " * " +
                std::to_string(B.rows()) + "x" + std::to_string(B.cols()));
        }
        // Ensure output matrix has correct dimensions
        if (C.rows() != A.rows() || C.cols() != B.cols()) {
            throw std::runtime_error("Output matrix has wrong dimensions: expected " +
                std::to_string(A.rows()) + "x" + std::to_string(B.cols()) + " got " +
                std::to_string(C.rows()) + "x" + std::to_string(C.cols()));
        }

        // Initialize cuBLAS if needed
        if (!cublas_initialized) {
            initialize_cuda();
        }

        // First ensure all matrices are on GPU
        Matrix A_gpu = A.is_cuda() ? A : A.to_gpu();
        Matrix B_gpu = B.is_cuda() ? B : B.to_gpu();
        if (!C.is_cuda()) {
            C = Matrix(C.rows(), C.cols(), nullptr, false);  // Create GPU matrix
        }

        // cuBLAS uses column-major order, while we use row-major order
        // So we compute C = B^T * A^T to get the correct result
        const float alpha = 1.0f;
        const float beta = 0.0f;
        
        // Get raw pointers
        float* d_A = A_gpu.get_data();
        float* d_B = B_gpu.get_data();
        float* d_C = C.get_data();

        // Perform matrix multiplication: C = alpha * (B^T * A^T) + beta * C
        CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                B.cols(), A.rows(), A.cols(),
                                &alpha,
                                d_B, B.cols(),
                                d_A, A.cols(),
                                &beta,
                                d_C, C.cols()));
        
        // Synchronize to catch any errors
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    __host__ void gelu_forward(Matrix& x) {
        float* d_x;
        size_t size = x.size() * sizeof(float);
        
        CUDA_CHECK(cudaMalloc(&d_x, size));
        CUDA_CHECK(cudaMemcpy(d_x, x.get_data(), size, cudaMemcpyHostToDevice));
        
        dim3 block(256);
        dim3 grid((x.size() + 255) / 256);
        
        gelu_forward_kernel<<<grid, block>>>(d_x, x.size());
        
        CUDA_CHECK(cudaMemcpy(x.get_data(), d_x, size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_x));
    }
}

// Kernel implementations
__global__ void matrix_multiply_kernel(const float* A, const float* B, float* C, 
                                     int M, int N, int K) {
    // Use shared memory for better performance
    __shared__ float shared_A[32][32];
    __shared__ float shared_B[32][32];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    for (int tile = 0; tile < (K + 31) / 32; ++tile) {
        // Load data into shared memory
        if (row < M && tile * 32 + threadIdx.x < K)
            shared_A[threadIdx.y][threadIdx.x] = A[row * K + tile * 32 + threadIdx.x];
        else
            shared_A[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && tile * 32 + threadIdx.y < K)
            shared_B[threadIdx.y][threadIdx.x] = B[(tile * 32 + threadIdx.y) * N + col];
        else
            shared_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial dot product
        if (row < M && col < N) {
            for (int k = 0; k < 32; ++k) {
                sum += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];
            }
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// GELU kernel implementations
__global__ void gelu_forward_kernel(float* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        float cdf = 0.5f * (1.0f + tanhf(0.797884f * (val + 0.044715f * val * val * val)));
        x[idx] = val * cdf;
    }
}
#endif // USE_CUDA