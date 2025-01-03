#include "cuda/cuda_utils.cuh"

cublasHandle_t cublas_handle;

void initialize_cuda() {
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
}

void cleanup_cuda() {
    if (cublas_handle != nullptr) {
        CUBLAS_CHECK(cublasDestroy(cublas_handle));
    }
}