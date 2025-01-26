#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t status = call; \
    if (status != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + \
            cudaGetErrorString(status)); \
    } \
} while(0)

namespace cuda {
    bool is_initialized();
    void initialize_cuda();
    void cleanup_cuda();
} 