#include "../../include/cuda/cuda_check.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "../../include/cuda/cuda_init.cuh"
#include "../../include/cuda/cuda_utils.cuh"
#include <stdexcept>
#include <string>

// Move handle into cuda namespace
namespace cuda {
    cublasHandle_t cublas_handle = nullptr;
}

// Move this inside namespace cuda
namespace cuda {
    static bool cuda_initialized = false;

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

// Keep these outside namespace
bool init_cublas() {
    cublasStatus_t status = cublasCreate(&cuda::cublas_handle);  // Use namespace qualified handle
    return (status == CUBLAS_STATUS_SUCCESS);
}

void cleanup_cublas() {
    if (cuda::cublas_handle != nullptr) {  // Use namespace qualified handle
        cublasDestroy(cuda::cublas_handle);
        cuda::cublas_handle = nullptr;
    }
}

// Add any other CUDA initialization code here...

// When shutting down
void shutdown() {
    cleanup_cublas();
    cleanup_cuda();  // Now this should be found since we included cuda_utils.cuh
}