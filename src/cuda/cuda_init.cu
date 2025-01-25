#include "../../include/cuda/cuda_check.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "../../include/cuda/cuda_init.cuh"
#include <stdexcept>
#include <string>

// Global cuBLAS handle
cublasHandle_t cublas_handle = nullptr;
static bool cuda_initialized = false;

namespace cuda {

void initialize_cuda() {
    if (cuda_initialized) {
        return;  // Already initialized
    }

    try {
        // Set device
        CUDA_CHECK(cudaSetDevice(0));

        // Initialize cuBLAS
        CUBLAS_CHECK(cublasCreate(&cublas_handle));
        
        // Set stream
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));

        cuda_initialized = true;
    } catch (const std::exception& e) {
        cleanup_cuda();  // Clean up on failure
        throw std::runtime_error(std::string("CUDA initialization failed: ") + e.what());
    }
}

void cleanup_cuda() {
    if (!cuda_initialized) {
        return;  // Already cleaned up
    }

    if (cublas_handle != nullptr) {
        cublasDestroy(cublas_handle);
        cublas_handle = nullptr;
    }

    cudaDeviceReset();
    cuda_initialized = false;
}

bool is_initialized() {
    return cuda_initialized;
}

} // namespace cuda