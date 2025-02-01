#include "cuda_kernels.hpp"
#include "cuda/cuda_check.cuh"
#include <cuda_fp16.h>

namespace cuda {

__global__ void softmax_kernel(float* matrix, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    // Find max value
    float max_val = -INFINITY;
    for (int i = 0; i < cols; i++) {
        max_val = max(max_val, matrix[row * cols + i]);
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < cols; i++) {
        float val = exp(matrix[row * cols + i] - max_val);
        matrix[row * cols + i] = val;
        sum += val;
    }

    // Normalize
    for (int i = 0; i < cols; i++) {
        matrix[row * cols + i] /= sum;
    }
}

__global__ void relu_kernel(float* matrix, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        matrix[idx] = max(0.0f, matrix[idx]);
    }
}

__global__ void attention_scores_kernel(const float* Q, const float* K, float* scores,
                                      float scale, int seq_len, int head_dim) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < seq_len && col < seq_len) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += Q[row * head_dim + d] * K[col * head_dim + d];
        }
        scores[row * seq_len + col] = score * scale;
    }
}

__global__ void attention_kernel(float* Q, float* K, float* V, float* output, 
                               int batch_size, int seq_len, int head_dim) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int i = threadIdx.x;
    
    if (b >= batch_size || i >= seq_len) return;

    __shared__ float scores[1024]; // Assuming max sequence length

    // Compute attention scores
    for (int j = 0; j < seq_len; j++) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += Q[b * seq_len * head_dim + i * head_dim + d] *
                     K[b * seq_len * head_dim + j * head_dim + d];
        }
        scores[j] = score / sqrt(float(head_dim));
    }

    // Apply softmax
    __syncthreads();
    float max_score = -INFINITY;
    for (int j = 0; j < seq_len; j++) {
        max_score = max(max_score, scores[j]);
    }

    float sum = 0.0f;
    for (int j = 0; j < seq_len; j++) {
        scores[j] = exp(scores[j] - max_score);
        sum += scores[j];
    }

    for (int j = 0; j < seq_len; j++) {
        scores[j] /= sum;
    }

    // Compute weighted sum
    for (int d = 0; d < head_dim; d++) {
        float weighted_sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            weighted_sum += scores[j] * V[b * seq_len * head_dim + j * head_dim + d];
        }
        output[b * seq_len * head_dim + i * head_dim + d] = weighted_sum;
    }
}

__global__ void add_bias_kernel(float* output, const float* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        const int col = idx % cols;
        output[idx] += bias[col];
    }
}

__global__ void row_sum_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float sum = 0.0f;
        for (int col = 0; col < cols; col++) {
            sum += input[row * cols + col];
        }
        output[row] = sum;
    }
}

void launch_softmax(float* matrix, int rows, int cols, cudaStream_t stream) {
    dim3 grid(rows);
    dim3 block(1);
    softmax_kernel<<<grid, block, 0, stream>>>(matrix, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    if (stream == nullptr) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

void launch_attention(float* Q, float* K, float* V, float* output,
                     int batch_size, int seq_len, int head_dim,
                     cudaStream_t stream) {
    dim3 grid(batch_size, 1);
    dim3 block(seq_len);
    attention_kernel<<<grid, block, 0, stream>>>(Q, K, V, output, 
                                               batch_size, seq_len, head_dim);
    CUDA_CHECK(cudaGetLastError());
    if (stream == nullptr) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

void launch_add_bias(float* output, const float* bias,
                    int rows, int cols,
                    cudaStream_t stream) {
    const int block_size = 256;
    const int num_blocks = (rows * cols + block_size - 1) / block_size;
    add_bias_kernel<<<num_blocks, block_size, 0, stream>>>(output, bias, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    if (stream == nullptr) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

void launch_row_sum(const float* input, float* output,
                   int rows, int cols,
                   cudaStream_t stream) {
    const int block_size = 256;
    const int num_blocks = (rows + block_size - 1) / block_size;
    row_sum_kernel<<<num_blocks, block_size, 0, stream>>>(input, output, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    if (stream == nullptr) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

void launch_adam_update(float* param, const float* grad,
                       float* m, float* v,
                       float beta1, float beta2,
                       float eps, float lr,
                       int step, int size,
                       cudaStream_t stream) {
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    adam_update_kernel<<<num_blocks, block_size, 0, stream>>>(
        param, grad, m, v, beta1, beta2, eps, lr, step, size);
    CUDA_CHECK(cudaGetLastError());
    if (stream == nullptr) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

} // namespace cuda