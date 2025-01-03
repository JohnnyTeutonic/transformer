#include "../../include/cuda/cuda_utils.cuh"

cublasHandle_t cublas_handle;

void initialize_cuda() {
    // Get the device with the highest compute capability
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    if (device_count == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        exit(EXIT_FAILURE);
    }
    
    int max_compute = 0;
    int selected_device = 0;
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        int compute = prop.major * 10 + prop.minor;
        if (compute > max_compute) {
            max_compute = compute;
            selected_device = i;
        }
    }
    
    CUDA_CHECK(cudaSetDevice(selected_device));
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    
    // Set optimal flags for maximum performance
    CUBLAS_CHECK(cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH));
}

void cleanup_cuda() {
    if (cublas_handle != nullptr) {
        CUBLAS_CHECK(cublasDestroy(cublas_handle));
        cublas_handle = nullptr;
    }
    CUDA_CHECK(cudaDeviceReset());
}