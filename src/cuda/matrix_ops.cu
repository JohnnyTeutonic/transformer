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
#include "../../include/cuda/cuda_init.cuh"

// Forward declare kernels
__global__ void gelu_forward_kernel(float* x, int size);

// Helper functions outside of cuda namespace
static void throw_runtime_error(const char* msg) {
    throw ::std::runtime_error(msg);
}

namespace cuda {
    Matrix matmul(const Matrix& A, const Matrix& B) {
        // Verify dimensions
        if (A.cols() != B.rows()) {
            throw_runtime_error("Matrix multiplication dimension mismatch");
        }

        // Initialize CUDA if needed
        if (!is_initialized()) {
            initialize_cuda();
        }

        // Create output matrix with correct dimensions
        Matrix C(A.rows(), B.cols());

        // First ensure all matrices are on GPU
        Matrix A_gpu = A.is_cuda() ? A : A.to_gpu();
        Matrix B_gpu = B.is_cuda() ? B : B.to_gpu();
        Matrix C_gpu(C.rows(), C.cols(), nullptr, true);  // Create GPU matrix

        // Get dimensions
        const int M = A.rows();
        const int N = B.cols();
        const int K = A.cols();

        // Set scaling factors
        const float alpha = 1.0f;
        const float beta = 0.0f;

        // Get raw pointers
        float* d_A = A_gpu.get_data();
        float* d_B = B_gpu.get_data();
        float* d_C = C_gpu.get_data();

        // Get the global cuBLAS handle
        extern cublasHandle_t cublas_handle;
        printf("cuBLAS handle obtained\n");
        // Perform matrix multiplication using cuBLAS
        // Note: cuBLAS uses column-major order, so we compute C = B^T * A^T
        CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                N, M, K, &alpha,
                                d_B, N,  // Leading dimension of B is N
                                d_A, K,  // Leading dimension of A is K
                                &beta,
                                d_C, N));  // Leading dimension of C is N
        printf("Matrix multiplication completed\n");
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
__global__ void gelu_forward_kernel(float* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        float cdf = 0.5f * (1.0f + tanhf(0.797884f * (val + 0.044715f * val * val * val)));
        x[idx] = val * cdf;
    }
}
#endif // USE_CUDA