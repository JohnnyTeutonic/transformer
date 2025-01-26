#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace cuda {
    extern cublasHandle_t cublas_handle;  // Move into cuda namespace
    bool init_cublas();  // Add this declaration
}

// Remove or update this conflicting declaration
// void initialize_cuda();  // <-- Remove this line since it's already declared in cuda_utils.cuh

void cleanup_cublas();
// Note: cleanup_cuda is now declared in cuda_utils.cuh