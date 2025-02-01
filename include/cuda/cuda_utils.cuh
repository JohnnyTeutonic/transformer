#pragma once
#include "../matrix.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace cuda {
    // Initialization and cleanup
    void initialize_cuda();
    void cleanup_cuda();
    bool is_available();
    void initialize();
    void cleanup();
    
    // Stream and handle management
    cudaStream_t get_stream();
    cublasHandle_t get_cublas_handle();
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
    
    // Attention operations
    void launch_attention_kernel(const float* Q, const float* K, const float* V,
                               float* output, const float* mask,
                               int batch_size, int num_heads, int seq_len, int head_dim,
                               float scale);
    
    void launch_softmax_kernel(float* scores, int seq_len, cudaStream_t stream = nullptr);

    // CUDA kernel launches
    void launch_add_bias(float* output, const float* bias, int batch_size, int hidden_size);
    
    void launch_row_sum(const float* input, float* output, 
                       unsigned long rows, unsigned long cols, 
                       cudaStream_t stream = nullptr);
    
    void launch_adam_update(float* params, const float* grads, float* m, float* v,
                           float beta1, float beta2, float eps, float lr, int size,
                           unsigned long step, cudaStream_t stream = nullptr);

    // Declare common CUDA utilities here
}