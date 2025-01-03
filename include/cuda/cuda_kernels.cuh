#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

// Forward declarations of CUDA kernels
__global__ void attention_kernel(float *Q, float *K, float *V, float *output,
                               int batch_size, int seq_len, int head_dim);

__global__ void softmax_kernel(float *matrix, int rows, int cols);

// Launch wrapper functions
void launch_attention(float *Q, float *K, float *V, float *output,
                     int batch_size, int seq_len, int head_dim,
                     cudaStream_t stream = nullptr);

void launch_softmax(float *matrix, int rows, int cols,
                   cudaStream_t stream = nullptr); 