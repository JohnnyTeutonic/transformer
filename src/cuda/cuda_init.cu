#include "../../include/cuda/cuda_init.cuh"
#include "../../include/cuda/cuda_check.cuh"
#include <cuda_runtime.h>
#include <stdexcept>

namespace cuda {
    static bool cuda_initialized = false;

    bool is_initialized() {
        return cuda_initialized;
    }

    void initialize_cuda() {
        if (cuda_initialized) {
            return;
        }

        // Get device properties
        int deviceCount;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0) {
            throw std::runtime_error("No CUDA devices available");
        }

        // Select and initialize the first available device
        CUDA_CHECK(cudaSetDevice(0));
        
        // Get and print device properties
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        printf("\nInitializing CUDA device: %s\n", prop.name);
        printf("Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("Max threads in X-dimension: %d\n", prop.maxThreadsDim[0]);
        printf("Max grid size in X-dimension: %d\n", prop.maxGridSize[0]);
        printf("Total global memory: %zu MB\n\n", prop.totalGlobalMem / (1024*1024));

        cuda_initialized = true;
    }

    void cleanup_cuda() {
        if (!cuda_initialized) {
            return;
        }

        // Reset device
        CUDA_CHECK(cudaDeviceReset());
        cuda_initialized = false;
    }
}

// When shutting down
void shutdown() {
    cuda::cleanup_cuda();  // This handles CUDA cleanup
}