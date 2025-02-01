#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "../../include/attention.hpp"
#include "../../include/cuda/attention_ops.cuh"
#include "../../include/cuda/cuda_check.cuh"
#include "../../include/cuda/matrix_ops.cuh"
#include "../../include/cuda/cuda_utils.cuh"
#include "../../include/cuda/kernel_declarations.cuh"

#ifndef MAX_SEQ_LEN
#define MAX_SEQ_LEN 2048  // or whatever maximum sequence length you want to support
#endif

// Kernel declarations in extern "C" to match header
extern "C" {
    __global__ void attention_kernel(const float* Q, const float* K, const float* V,
                                   float* output, const float* mask,
                                   int batch_size, int seq_len, int head_dim, int hidden_dim);
    
    __global__ void attention_scores_kernel(const float* Q, const float* K, float* scores,
                                          float scale, int seq_len, int head_dim);
    
    __global__ void softmax_kernel(float* matrix, int rows, int cols);
}

namespace cuda {
    // Host functions only in namespace
    void compute_attention_scores(const Matrix& Q, const Matrix& K, Matrix& scores, float scale, int num_heads) {
        // Synchronize before starting
        CUDA_CHECK(cudaDeviceSynchronize());
        
        int batch_size = Q.rows();
        int hidden_dim = Q.cols();
        int head_dim = hidden_dim / num_heads;
        int seq_len = batch_size;

        // Verify all dimensions are valid
        if (batch_size <= 0 || hidden_dim <= 0 || head_dim <= 0 || seq_len <= 0) {
            throw std::runtime_error("Invalid dimensions detected");
        }

        // Memory allocation with error checking
        float* d_Q = nullptr;
        float* d_K = nullptr;
        float* d_scores = nullptr;
        
        try {
            CUDA_CHECK(cudaMalloc(&d_Q, Q.size() * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_K, K.size() * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_scores, scores.size() * sizeof(float)));
            
            // Zero initialize the scores buffer
            CUDA_CHECK(cudaMemset(d_scores, 0, scores.size() * sizeof(float)));

            CUDA_CHECK(cudaMemcpy(d_Q, Q.data(), Q.size() * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_K, K.data(), K.size() * sizeof(float), cudaMemcpyHostToDevice));
            
            // Synchronize to ensure memory transfers are complete
            CUDA_CHECK(cudaDeviceSynchronize());

            dim3 block(16, 16);
            dim3 grid((seq_len + block.x - 1) / block.x, (seq_len + block.y - 1) / block.y);
            
            attention_scores_kernel<<<grid, block>>>(d_Q, d_K, d_scores,
                scale, seq_len, head_dim);
                
            // Check for kernel launch errors
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("Kernel launch failed: ") + 
                                       cudaGetErrorString(err));
            }

            // Synchronize after kernel
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy(scores.data(), d_scores, scores.size() * sizeof(float), 
                                cudaMemcpyDeviceToHost));

        } catch (const std::exception& e) {
            printf("CUDA error caught: %s\n", e.what());
            // Clean up on error
            if (d_Q) cudaFree(d_Q);
            if (d_K) cudaFree(d_K);
            if (d_scores) cudaFree(d_scores);
            throw;  // Re-throw the exception
        }

        // Clean up
        CUDA_CHECK(cudaFree(d_Q));
        CUDA_CHECK(cudaFree(d_K));
        CUDA_CHECK(cudaFree(d_scores));
        
        // Final synchronize
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void apply_softmax(Matrix& matrix) {
        float* d_matrix;
        size_t size = matrix.size() * sizeof(float);

        CUDA_CHECK(cudaMalloc(&d_matrix, size));
        CUDA_CHECK(cudaMemcpy(d_matrix, matrix.data(), size, cudaMemcpyHostToDevice));

        softmax_kernel<<<matrix.rows(), 1>>>(d_matrix, matrix.rows(), matrix.cols());

        CUDA_CHECK(cudaMemcpy(matrix.data(), d_matrix, size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_matrix));
    }

    void attention_forward(const Matrix& Q, const Matrix& K, const Matrix& V, 
                         Matrix& output, int batch_size, int num_heads, int seq_len) {
        // Configure grid for batch and head parallelism
        dim3 block(32, 1);
        dim3 grid((batch_size + block.x - 1) / block.x, num_heads);
        
        int head_dim = Q.cols() / num_heads;
        int hidden_dim = Q.cols();  // Store the full hidden dimension

        float *d_Q, *d_K, *d_V, *d_output;
        size_t QKV_size = Q.size() * sizeof(float);
        size_t output_size = output.size() * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_Q, QKV_size));
        CUDA_CHECK(cudaMalloc(&d_K, QKV_size));
        CUDA_CHECK(cudaMalloc(&d_V, QKV_size));
        CUDA_CHECK(cudaMalloc(&d_output, output_size));

        CUDA_CHECK(cudaMemcpy(d_Q, Q.data(), QKV_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K, K.data(), QKV_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V, V.data(), QKV_size, cudaMemcpyHostToDevice));

        // Allocate shared memory for scores
        size_t shared_mem_size = seq_len * sizeof(float);
        attention_kernel<<<grid, block, shared_mem_size>>>(d_Q, d_K, d_V, d_output,
                                                         batch_size, seq_len, head_dim, hidden_dim);

        CUDA_CHECK(cudaMemcpy(output.data(), d_output, output_size, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_Q));
        CUDA_CHECK(cudaFree(d_K));
        CUDA_CHECK(cudaFree(d_V));
        CUDA_CHECK(cudaFree(d_output));
    }

    void launch_attention_scores_kernel(const float* Q, const float* K, float* scores, float scale,
                                      int seq_len, int head_dim, cudaStream_t stream) {
        dim3 block_dim(16, 16);
        dim3 grid_dim((seq_len + block_dim.x - 1) / block_dim.x,
                      (seq_len + block_dim.y - 1) / block_dim.y);

        attention_scores_kernel<<<grid_dim, block_dim, 0, stream>>>(Q, K, scores, scale, seq_len,
                                                                 head_dim);
    }

    void launch_attention_kernel(const float* Q, const float* K, const float* V,
                               float* output, const float* mask,
                               int batch_size, int num_heads, int seq_len, int head_dim,
                               float scale, cudaStream_t stream) {
        // Calculate grid and block dimensions
        dim3 block(256);
        dim3 grid((seq_len + block.x - 1) / block.x, num_heads, batch_size);
        
        // Launch kernel
        scaled_dot_product_attention_kernel<<<grid, block, 0, stream>>>(
            Q, K, V, output, mask,
            batch_size, num_heads, seq_len, head_dim,
            scale);
        
        CUDA_CHECK(cudaGetLastError());
    }
}

// Kernel implementations
extern "C" {
    __global__ void attention_kernel(const float* Q, const float* K, const float* V,
                                   float* output, const float* mask,
                                   int batch_size, int seq_len, int head_dim, int hidden_dim) {
        int b = blockIdx.x * blockDim.x + threadIdx.x;  // batch index
        int h = blockIdx.y;  // head index

        if (b < batch_size) {
            // Process this batch element for the current head
            int head_offset = h * head_dim;
            int batch_offset = b * hidden_dim;
            
            // Allocate scores in shared memory
            extern __shared__ float scores[];
            
            // Compute attention scores for this head
            for (int j = 0; j < seq_len; j++) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    int q_idx = batch_offset + head_offset + d;
                    int k_idx = j * hidden_dim + head_offset + d;
                    
                    if (q_idx < batch_size * hidden_dim && k_idx < seq_len * hidden_dim) {
                        score += Q[q_idx] * K[k_idx];
                    }
                }
                scores[j] = score / sqrtf(float(head_dim));
                
                // Apply mask if provided
                if (mask != nullptr) {
                    scores[j] += mask[b * seq_len + j];
                }
            }
            
            // Apply softmax
            float max_score = scores[0];
            for (int j = 1; j < seq_len; j++) {
                max_score = max(max_score, scores[j]);
            }
            
            float sum = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                scores[j] = expf(scores[j] - max_score);
                sum += scores[j];
            }
            
            for (int j = 0; j < seq_len; j++) {
                scores[j] /= sum;
            }
            
            // Compute weighted sum
            for (int d = 0; d < head_dim; d++) {
                float weighted_sum = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    int v_idx = j * hidden_dim + head_offset + d;
                    if (v_idx < seq_len * hidden_dim) {
                        weighted_sum += scores[j] * V[v_idx];
                    }
                }
                int out_idx = batch_offset + head_offset + d;
                if (out_idx < batch_size * hidden_dim) {
                    output[out_idx] = weighted_sum;
                }
            }
        }
    }

    __global__ void attention_scores_kernel(const float* Q, const float* K, float* scores,
                                           float scale, int seq_len, int head_dim) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        int col = blockIdx.y * blockDim.y + threadIdx.y;

        if (row < seq_len && col < seq_len) {
            float sum = 0.0f;
            for (int i = 0; i < head_dim; i++) {
                int q_idx = row * head_dim + i;
                int k_idx = col * head_dim + i;
                if (q_idx < seq_len * head_dim && k_idx < seq_len * head_dim) {
                    sum += Q[q_idx] * K[k_idx];
                }
            }
            scores[row * seq_len + col] = sum * scale;
        }
    }

    __global__ void softmax_kernel(float* matrix, int rows, int cols) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < rows) {
            // Find max for numerical stability
            float max_val = matrix[row * cols];
            for (int i = 1; i < cols; i++) {
                max_val = max(max_val, matrix[row * cols + i]);
            }

            // Compute exp and sum
            float sum = 0.0f;
            for (int i = 0; i < cols; i++) {
                matrix[row * cols + i] = expf(matrix[row * cols + i] - max_val);
                sum += matrix[row * cols + i];
            }

            // Normalize
            for (int i = 0; i < cols; i++) {
                matrix[row * cols + i] /= (sum + 1e-6f);
            }
        }
    }

    __global__ void scaled_dot_product_attention_kernel(
        const float* Q, const float* K, const float* V,
        float* output, const float* mask,
        int batch_size, int num_heads, int seq_len, int head_dim,
        float scale) {
        
        const int b = blockIdx.z;  // batch index
        const int h = blockIdx.y;  // head index
        const int i = blockIdx.x * blockDim.x + threadIdx.x;  // sequence position
        
        if (b < batch_size && h < num_heads && i < seq_len) {
            // Calculate attention scores for this position
            float scores[MAX_SEQ_LEN];  // Assume MAX_SEQ_LEN is defined
            
            // Compute attention scores
            for (int j = 0; j < seq_len; j++) {
                float sum = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    const int q_idx = ((b * num_heads + h) * seq_len + i) * head_dim + d;
                    const int k_idx = ((b * num_heads + h) * seq_len + j) * head_dim + d;
                    sum += Q[q_idx] * K[k_idx];
                }
                scores[j] = sum * scale;
                
                // Apply mask if provided
                if (mask != nullptr) {
                    scores[j] += mask[i * seq_len + j];
                }
            }
            
            // Apply softmax
            float max_score = scores[0];
            for (int j = 1; j < seq_len; j++) {
                max_score = max(max_score, scores[j]);
            }
            
            float sum = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                scores[j] = expf(scores[j] - max_score);
                sum += scores[j];
            }
            
            for (int j = 0; j < seq_len; j++) {
                scores[j] /= sum;
            }
            
            // Compute weighted sum of values
            for (int d = 0; d < head_dim; d++) {
                float weighted_sum = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    const int v_idx = ((b * num_heads + h) * seq_len + j) * head_dim + d;
                    weighted_sum += scores[j] * V[v_idx];
                }
                const int out_idx = ((b * num_heads + h) * seq_len + i) * head_dim + d;
                output[out_idx] = weighted_sum;
            }
        }
    }
} 

// Implementation of MultiHeadAttention::forward_cuda
#ifdef CUDA_AVAILABLE
Matrix MultiHeadAttention::forward_cuda(const Matrix& input, 
                                      const AttentionMask& mask,
                                      const std::optional<KVCache>& kv_cache) {
    const int batch_size = input.rows();
    const int seq_len = input.cols() / hidden_size;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    // Project input to Q, K, V using CUDA matmul
    Matrix Q(input.rows(), get_query_weights().cols());
    cuda::matmul(input, get_query_weights(), Q, nullptr);
    
    Matrix K;
    if (kv_cache) {
        K = kv_cache->get_key();
    } else {
        K = Matrix(input.rows(), get_key_weights().cols());
        cuda::matmul(input, get_key_weights(), K, nullptr);
    }
    
    Matrix V;
    if (kv_cache) {
        V = kv_cache->get_value();
    } else {
        V = Matrix(input.rows(), get_value_weights().cols());
        cuda::matmul(input, get_value_weights(), V, nullptr);
    }
    
    // Reshape matrices for attention
    Matrix Q_reshaped = Q.reshape(batch_size, num_heads, seq_len, head_dim);
    Matrix K_reshaped = K.reshape(batch_size, num_heads, seq_len, head_dim);
    Matrix V_reshaped = V.reshape(batch_size, num_heads, seq_len, head_dim);
    
    // Allocate output
    Matrix output(batch_size * seq_len, hidden_size);
    
    // Launch attention kernel
    cuda::launch_attention_kernel(
        Q_reshaped.data(), K_reshaped.data(), V_reshaped.data(),
        output.data(), mask.value().data(),
        batch_size, num_heads, seq_len, head_dim,
        scale);
    
    // Project output
    Matrix final_output(output.rows(), get_output_weights().cols());
    cuda::matmul(output, get_output_weights(), final_output, nullptr);
    
    return final_output;
}
#endif 