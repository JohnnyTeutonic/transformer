#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdexcept>
#include <string>

// Global cuBLAS handle
extern cublasHandle_t cublas_handle;

// CUDA initialization/cleanup
void initialize_cuda();
void cleanup_cuda();

// Define CUDA kernel launch macro first
#ifdef __CUDACC__
    #define CUDA_KERNEL_LAUNCH(grid, block, shared, stream) <<<grid, block, shared, stream>>>
#else
    #define CUDA_KERNEL_LAUNCH(grid, block, shared, stream)
#endif

namespace cuda {
    template<typename KernelFunc, typename... Args>
    __host__ inline void launch_kernel(KernelFunc kernel, dim3 grid, dim3 block, 
                                     size_t shared_mem, cudaStream_t stream, 
                                     Args... args) {
        kernel CUDA_KERNEL_LAUNCH(grid, block, shared_mem, stream)(args...);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            throw std::runtime_error("CUDA kernel launch failed: " + 
                                   std::string(cudaGetErrorString(error)));
        }
    }
}

// Macro for launching CUDA kernels
#define CUDA_LAUNCH(kernel, grid, block, shared_mem, stream, ...) \
    cuda::launch_kernel(kernel, grid, block, shared_mem, stream, __VA_ARGS__)

// Error checking macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(error))); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            throw std::runtime_error("cuBLAS error: " + std::to_string(status)); \
        } \
    } while(0)