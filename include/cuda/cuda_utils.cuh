#pragma once
#include "../matrix.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace cuda {
    void initialize_cuda();
    void cleanup_cuda();
    void launch_softmax_kernel(float* scores, int seq_len, cudaStream_t stream);

    // Check if CUDA is available and initialized
    bool is_available();

    // Initialize CUDA resources
    void initialize();

    // Cleanup CUDA resources
    void cleanup();

    // Get CUDA stream
    cudaStream_t get_stream();

    // Get cuBLAS handle
    cublasHandle_t get_cublas_handle();

    // Synchronize CUDA stream
    void synchronize();

    // Memory management helpers
    template<typename T>
    T* device_malloc(size_t size);

    template<typename T>
    void device_free(T* ptr);

    template<typename T>
    void copy_to_device(T* dst, const T* src, size_t size);

    template<typename T>
    void copy_to_host(T* dst, const T* src, size_t size);
}

#ifdef CUDA_AVAILABLE
Matrix cuda_matmul(const Matrix& A, const Matrix& B);
void launch_attention_scores(const float* Q, const float* K, float* scores, float scale,
                             int seq_len, int head_dim, cudaStream_t stream);
void launch_softmax(float* scores, int seq_len, cudaStream_t stream);
#endif