// Include Matrix definition first
#include "../../include/matrix.hpp"

#ifdef USE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include "../../include/cuda/matrix_ops.cuh"
#include "../../include/cuda/cuda_check.cuh"
#include "../../include/cuda/cuda_utils.cuh"
#include "../../include/cuda/memory_manager.cuh"

// Forward declare all kernels
__global__ void matrix_multiply_kernel(const float* A, const float* B, float* C, 
                                      int M, int N, int K);
__global__ void gelu_forward_kernel(float* x, int size);

// Global cuBLAS handle
static cublasHandle_t cublas_handle;
static bool cublas_initialized = false;

// Helper functions outside of cuda namespace
static void throw_runtime_error(const char* msg) {
    throw ::std::runtime_error(msg);
}

namespace cuda {
    void initialize_cuda() {
        CUDA_CHECK(cudaSetDevice(0));
        if (!cublas_initialized) {
            CUBLAS_CHECK(cublasCreate(&cublas_handle));
            cublas_initialized = true;
        }
    }

    void cleanup_cuda() {
        if (cublas_initialized) {
            CUBLAS_CHECK(cublasDestroy(cublas_handle));
            cublas_initialized = false;
        }
    }

    Matrix matmul(const Matrix& A, const Matrix& B) {
        // Verify dimensions
        if (A.cols() != B.rows()) {
            throw_runtime_error("Matrix multiplication dimension mismatch");
        }

        // Initialize cuBLAS if needed
        if (!cublas_initialized) {
            initialize_cuda();
        }

        // Create output matrix with correct dimensions
        Matrix C(A.rows(), B.cols());

        // First ensure all matrices are on GPU
        Matrix A_gpu = A.is_cuda() ? A : A.to_gpu();
        Matrix B_gpu = B.is_cuda() ? B : B.to_gpu();
        Matrix C_gpu(C.rows(), C.cols(), nullptr, false);  // Create GPU matrix

        // cuBLAS uses column-major order, while we use row-major order
        // So we compute C = B^T * A^T to get the correct result
        const float alpha = 1.0f;
        const float beta = 0.0f;
        
        // Get raw pointers
        float* d_A = A_gpu.get_data();
        float* d_B = B_gpu.get_data();
        float* d_C = C_gpu.get_data();

        // Perform matrix multiplication: C = alpha * (B^T * A^T) + beta * C
        CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                B.cols(), A.rows(), A.cols(),
                                &alpha,
                                d_B, B.cols(),
                                d_A, A.cols(),
                                &beta,
                                d_C, C.cols()));
        
        // Synchronize to catch any errors
        CUDA_CHECK(cudaDeviceSynchronize());
        
        return C_gpu;
    }

    void gelu_forward(Matrix& x) {
        if (!x.is_cuda()) {
            x = x.to_gpu();
        }
        
        dim3 block(256);
        dim3 grid((x.size() + block.x - 1) / block.x);
        
        gelu_forward_kernel<<<grid, block>>>(x.get_data(), x.size());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void topk(const Matrix& input, Matrix& top_k_values, ::std::vector<int>& top_k_indices, int k) {
        // Validate input dimensions
        if (k <= 0 || k > input.cols()) {
            throw_runtime_error("Invalid k value for topk operation");
        }

        // First do it on CPU for now (can optimize with CUDA later)
        Matrix input_cpu = input.is_cuda() ? input.to_cpu() : input;
        top_k_values = Matrix(input.rows(), k);
        top_k_indices.resize(input.rows() * k);

        // For each row
        for (size_t i = 0; i < input.rows(); ++i) {
            ::std::vector<::std::pair<float, int>> pairs;
            pairs.reserve(input.cols());
            for (size_t j = 0; j < input.cols(); ++j) {
                pairs.push_back({input_cpu(i, j), static_cast<int>(j)});
            }

            // Sort in descending order
            ::std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });

            // Copy top k values and indices
            for (int j = 0; j < k; ++j) {
                top_k_values(i, j) = pairs[j].first;
                top_k_indices[i * k + j] = pairs[j].second;
            }
        }
    }

    void topk(const ::std::vector<float>& input, Matrix& top_k_values, ::std::vector<int>& top_k_indices, int k) {
        // Validate input dimensions
        if (k <= 0 || k > input.size()) {
            throw_runtime_error("Invalid k value for topk operation");
        }

        // Create pairs of (value, index) for sorting
        ::std::vector<::std::pair<float, int>> pairs;
        pairs.reserve(input.size());
        for (size_t j = 0; j < input.size(); ++j) {
            pairs.push_back({input[j], static_cast<int>(j)});
        }

        // Sort in descending order
        ::std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });

        // Initialize output matrix with single row
        top_k_values = Matrix(1, k);
        top_k_indices.resize(k);

        // Copy top k values and indices
        for (int j = 0; j < k; ++j) {
            top_k_values(0, j) = pairs[j].first;
            top_k_indices[j] = pairs[j].second;
        }
    }

    void beam_search_step(const Matrix& model_output, const Matrix& beam_scores,
                         Matrix& next_scores, ::std::vector<int>& next_tokens,
                         int beam_width) {
        // Get dimensions
        size_t vocab_size = model_output.cols();
        size_t batch_size = model_output.rows();
        
        // Ensure inputs are on CPU for now
        Matrix model_output_cpu = model_output.is_cuda() ? model_output.to_cpu() : model_output;
        Matrix beam_scores_cpu = beam_scores.is_cuda() ? beam_scores.to_cpu() : beam_scores;
        
        // Compute scores
        Matrix scores(batch_size, vocab_size);
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < vocab_size; ++j) {
                scores(i, j) = model_output_cpu(i, j) + beam_scores_cpu(i, 0);
            }
        }
        
        // Find top-k scores and corresponding tokens
        topk(scores, next_scores, next_tokens, beam_width);
    }
}

// Kernel implementations
__global__ void matrix_multiply_kernel(const float* A, const float* B, float* C, 
                                     int M, int N, int K) {
    // Use shared memory for better performance
    __shared__ float shared_A[32][32];
    __shared__ float shared_B[32][32];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    for (int tile = 0; tile < (K + 31) / 32; ++tile) {
        // Load data into shared memory
        if (row < M && tile * 32 + threadIdx.x < K)
            shared_A[threadIdx.y][threadIdx.x] = A[row * K + tile * 32 + threadIdx.x];
        else
            shared_A[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && tile * 32 + threadIdx.y < K)
            shared_B[threadIdx.y][threadIdx.x] = B[(tile * 32 + threadIdx.y) * N + col];
        else
            shared_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial dot product
        if (row < M && col < N) {
            for (int k = 0; k < 32; ++k) {
                sum += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];
            }
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// GELU kernel implementations
__global__ void gelu_forward_kernel(float* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        float cdf = 0.5f * (1.0f + tanhf(0.797884f * (val + 0.044715f * val * val * val)));
        x[idx] = val * cdf;
    }
}
#endif // USE_CUDA