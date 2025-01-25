#include "../../include/cuda/cuda_utils.cuh"
#include "../../include/cuda/cuda_check.cuh"
#include "../../include/cuda/cuda_init.cuh"

namespace cuda {

void initialize_memory_manager() {
    if (!is_initialized()) {
        initialize_cuda();
    }
}

void cleanup_memory_manager() {
    cleanup_cuda();
}

} // namespace cuda 