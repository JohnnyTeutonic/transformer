#include "../../include/cuda/cuda_utils.cuh"
#include "../../include/cuda/cuda_check.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace cuda {
    __global__ void softmax_kernel(float* scores, int seq_len) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < seq_len) {
            float max_val = scores[row * seq_len];
            for (int i = 1; i < seq_len; i++) {
                max_val = max(max_val, scores[row * seq_len + i]);
            }

            float sum = 0.0f;
            for (int i = 0; i < seq_len; i++) {
                scores[row * seq_len + i] = expf(scores[row * seq_len + i] - max_val);
                sum += scores[row * seq_len + i];
            }

            for (int i = 0; i < seq_len; i++) {
                scores[row * seq_len + i] /= sum;
            }
        }
    }

    void launch_softmax_kernel(float* scores, int seq_len, cudaStream_t stream) {
        dim3 block_dim(256);
        dim3 grid_dim((seq_len + block_dim.x - 1) / block_dim.x);

        softmax_kernel<<<grid_dim, block_dim, 0, stream>>>(scores, seq_len);
    }

    __global__ void add_bias_kernel(float* matrix, const float* bias, int rows, int cols) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < rows * cols) {
            int col = idx % cols;
            matrix[idx] += bias[col];
        }
    }

    void add_bias(Matrix& matrix, const FloatVector& bias) {
        // Check dimensions first
        if (bias.size() != matrix.cols()) {
            throw std::invalid_argument("Bias size (" + std::to_string(bias.size()) + 
                                      ") must match matrix columns (" + std::to_string(matrix.cols()) + ")");
        }

        float* d_matrix = matrix.get_data();
        float* d_bias;
        CUDA_CHECK(cudaMalloc(&d_bias, bias.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_bias, bias.data(), bias.size() * sizeof(float), cudaMemcpyHostToDevice));

        dim3 block(256);
        dim3 grid((matrix.size() + block.x - 1) / block.x);
        add_bias_kernel<<<grid, block>>>(d_matrix, d_bias, matrix.rows(), matrix.cols());

        CUDA_CHECK(cudaFree(d_bias));
    }

    __global__ void compute_bias_gradients_kernel(float* bias_grad, const float* grad, int rows, int cols) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col < cols) {
            float sum = 0.0f;
            for (int row = 0; row < rows; ++row) {
                sum += grad[row * cols + col];
            }
            bias_grad[col] = sum;
        }
    }

    void compute_bias_gradients(FloatVector& bias_grad, const Matrix& grad) {
        float* d_bias_grad;
        CUDA_CHECK(cudaMalloc(&d_bias_grad, bias_grad.size() * sizeof(float)));
        
        dim3 block(256);
        dim3 grid((grad.cols() + block.x - 1) / block.x);
        compute_bias_gradients_kernel<<<grid, block>>>(d_bias_grad, grad.get_data(), grad.rows(), grad.cols());

        CUDA_CHECK(cudaMemcpy(bias_grad.data(), d_bias_grad, bias_grad.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_bias_grad));
    }
}