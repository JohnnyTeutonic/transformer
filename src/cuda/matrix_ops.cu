// Include Matrix definition first
#include "../../include/matrix.hpp"
#include <stdexcept>

#ifdef USE_CUDA
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
        if (A.rows() == 0 || A.cols() == 0 || B.rows() == 0 || B.cols() == 0) {
            throw_runtime_error("Matrix multiplication with zero dimensions");
        }
        printf("Matrix multiplication dimensions verified: (%zu x %zu) * (%zu x %zu)\n", 
               A.rows(), A.cols(), B.rows(), B.cols());
        
        // Initialize CUDA if needed
        if (!is_initialized()) {
            initialize_cuda();
        }
        printf("CUDA initialized\n");
        
        // Create output matrix with correct dimensions
        Matrix C(A.rows(), B.cols());
        if (C.rows() == 0 || C.cols() == 0) {
            throw_runtime_error("Output matrix has zero dimensions");
        }

        // First ensure all matrices are on GPU
        printf("Ensuring matrices are on GPU\n");
        Matrix A_gpu = A.is_cuda() ? A : A.to_gpu();
        Matrix B_gpu = B.is_cuda() ? B : B.to_gpu();
        Matrix C_gpu(C.rows(), C.cols(), nullptr, true);  // Create GPU matrix
        
        // Verify GPU matrices
        if (!A_gpu.is_cuda() || !B_gpu.is_cuda() || !C_gpu.is_cuda()) {
            throw_runtime_error("Failed to move matrices to GPU");
        }
        if (!A_gpu.get_data() || !B_gpu.get_data() || !C_gpu.get_data()) {
            throw_runtime_error("Null GPU data pointers");
        }
        printf("GPU matrices created and verified\n");

        // Use our custom matrix multiplication
        if (!customMatrixMultiply(A_gpu, B_gpu, C_gpu)) {
            throw_runtime_error("Matrix multiplication failed");
        }
        printf("Matrix multiplication completed\n");
        
        return C_gpu;
    }

    void gelu_forward(Matrix& x) {
        if (!x.is_cuda()) {
            x = x.to_gpu();
        }
        
        if (!x.is_cuda() || !x.get_data()) {
            throw_runtime_error("Failed to prepare matrix for GELU");
        }
        
        dim3 block(256);
        dim3 grid((x.size() + block.x - 1) / block.x);
        
        gelu_forward_kernel<<<grid, block>>>(x.get_data(), x.size());
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            throw_runtime_error(cudaGetErrorString(error));
        }
        error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            throw_runtime_error(cudaGetErrorString(error));
        }
    }
}

// Kernel implementations
__global__ void gelu_forward_kernel(float* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        // Clamp input to avoid numerical instability
        val = max(-20.0f, min(20.0f, val));
        float cdf = 0.5f * (1.0f + tanhf(0.797884f * (val + 0.044715f * val * val * val)));
        x[idx] = val * cdf;
    }
}
#endif // USE_CUDA