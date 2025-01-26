#pragma once
#include <cuda_runtime.h>

namespace cuda {
    // CUDA initialization and cleanup
    bool is_initialized();
    void initialize_cuda();
    void cleanup_cuda();
}

// Remove or update this conflicting declaration
// void initialize_cuda();  // <-- Remove this line since it's already declared in cuda_utils.cuh

// Note: cleanup_cuda is now declared in cuda_utils.cuh