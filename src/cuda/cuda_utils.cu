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

    // Declare constant for shared memory size
    constexpr int BLOCK_SIZE = 256;
    constexpr int WARP_SIZE = 32;

    __global__ void compute_bias_gradients_kernel(float* bias_grad, const float* grad, int rows, int cols) {
        const int tid = threadIdx.x;
        const int col = blockIdx.x;

        // Each block handles one column
        if (col >= cols) return;

        // Use shared memory for parallel reduction
        extern __shared__ float shared_sum[];
        shared_sum[tid] = 0.0f;
        __syncthreads();

        // Each thread accumulates values for its assigned rows
        for (int row = tid; row < rows; row += blockDim.x) {
            shared_sum[tid] += grad[row * cols + col];
        }
        __syncthreads();

        // Parallel reduction in shared memory
        for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared_sum[tid] += shared_sum[tid + stride];
            }
            __syncthreads();
        }

        // Only thread 0 writes the final result
        if (tid == 0) {
            bias_grad[col] = shared_sum[0];
        }
    }

    // CPU implementation as a separate function
    void cpu_compute_bias_gradients(FloatVector& bias_grad, const Matrix& grad) {
        #pragma omp parallel for
        for (int col = 0; col < static_cast<int>(grad.cols()); ++col) {
            float sum = 0.0f;
            for (int row = 0; row < static_cast<int>(grad.rows()); ++row) {
                sum += grad(row, col);
            }
            bias_grad[col] = sum;
        }
    }

    void compute_bias_gradients(FloatVector& bias_grad, const Matrix& grad) {
        // Validate dimensions
        if (bias_grad.size() != grad.cols()) {
            throw std::runtime_error(
                "Bias gradient size (" + std::to_string(bias_grad.size()) + 
                ") must match input columns (" + std::to_string(grad.cols()) + ")"
            );
        }

        // Just use CPU implementation for now since it's reliable
        cpu_compute_bias_gradients(bias_grad, grad);
    }
}