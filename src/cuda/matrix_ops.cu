#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "../../include/cuda/matrix_ops.cuh"
#include "../../include/cuda/cuda_check.cuh"
#include "../../include/cuda/cuda_utils.cuh"

// Forward declare all kernels
__global__ void matrix_multiply_kernel(const float* A, const float* B, float* C, 
                                      int M, int N, int K);
__global__ void gelu_forward_kernel(float* x, int size);
__global__ void add_bias_kernel(float* output, const float* bias, int rows, int cols);
__global__ void row_sum_kernel(const float* input, float* output, int rows, int cols);
__global__ void adam_update_kernel(float* params, const float* grads, float* m, float* v,
                                 float beta1, float beta2, float lr, float epsilon, int size);

namespace cuda {
    // Global cuBLAS handle with proper initialization
    static cublasHandle_t cublas_handle = nullptr;
    static bool cuda_initialized = false;

    void initialize_cuda() {
        if (cuda_initialized) {
            return;
        }

        // Set CUDA device
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to set CUDA device: " + std::string(cudaGetErrorString(err)));
        }
        std::cout << "CUDA device set successfully" << std::endl;

        // Print CUDA device properties
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "Using CUDA device: " << prop.name << std::endl;
        std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;

        // Initialize cuBLAS
        cublasStatus_t status = cublasCreate(&cublas_handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cuBLAS handle: " + std::to_string(status));
        }
        std::cout << "cuBLAS handle created successfully" << std::endl;

        cuda_initialized = true;
    }

    void cleanup_cuda() {
        if (cublas_handle != nullptr) {
            cublasDestroy(cublas_handle);
            cublas_handle = nullptr;
            cuda_initialized = false;
            std::cout << "cuBLAS handle destroyed successfully" << std::endl;
        }
    }

    void matmul(const Matrix& A, const Matrix& B, Matrix& C) {
        // Ensure CUDA is initialized
        if (!cuda_initialized || cublas_handle == nullptr) {
            initialize_cuda();
        }

        // Verify dimensions
        if (A.cols() != B.rows()) {
            throw std::runtime_error("Matrix multiplication dimension mismatch: " +
                std::to_string(A.rows()) + "x" + std::to_string(A.cols()) + " * " +
                std::to_string(B.rows()) + "x" + std::to_string(B.cols()));
        }
        // Ensure output matrix has correct dimensions
        if (C.rows() != A.rows() || C.cols() != B.cols()) {
            throw std::runtime_error("Output matrix has wrong dimensions: expected " +
                std::to_string(A.rows()) + "x" + std::to_string(B.cols()) + " got " +
                std::to_string(C.rows()) + "x" + std::to_string(C.cols()));
        }

        float* d_A, *d_B, *d_C;
        size_t A_size = A.rows() * A.cols() * sizeof(float);
        size_t B_size = B.rows() * B.cols() * sizeof(float);
        size_t C_size = A.rows() * B.cols() * sizeof(float);

        CUDA_CHECK(cudaMalloc(&d_A, A_size));
        CUDA_CHECK(cudaMalloc(&d_B, B_size));
        CUDA_CHECK(cudaMalloc(&d_C, C_size));

        CUDA_CHECK(cudaMemcpy(d_A, A.data(), A_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, B.data(), B_size, cudaMemcpyHostToDevice));

        float alpha = 1.0f;
        float beta = 0.0f;

        // For row-major matrices A[m,k] * B[k,n] = C[m,n], we compute:
        // C = A * B in column-major order
        cublasStatus_t status = cublasSgemm(cublas_handle,
                                          CUBLAS_OP_N, CUBLAS_OP_N,  // No transposition needed
                                          B.cols(), A.rows(), A.cols(),  // Dimensions for the operation
                                          &alpha,
                                          d_B, B.cols(),  // Leading dimension is cols for B
                                          d_A, A.cols(),  // Leading dimension is cols for A
                                          &beta,
                                          d_C, B.cols()); // Leading dimension is cols for C

        // Print dimensions for debugging
        std::cout << "Matrix multiplication dimensions:" << std::endl;
        std::cout << "A: " << A.rows() << "x" << A.cols() << std::endl;
        std::cout << "B: " << B.rows() << "x" << B.cols() << std::endl;
        std::cout << "C: " << C.rows() << "x" << C.cols() << std::endl;

        if (status != CUBLAS_STATUS_SUCCESS) {
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
            throw std::runtime_error("cuBLAS matrix multiplication failed with status: " + std::to_string(status));
        }

        CUDA_CHECK(cudaMemcpy(C.data(), d_C, C_size, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    }

    void gelu_forward(Matrix& x) {
        float* d_x;
        size_t size = x.size() * sizeof(float);
        
        CUDA_CHECK(cudaMalloc(&d_x, size));
        CUDA_CHECK(cudaMemcpy(d_x, x.data(), size, cudaMemcpyHostToDevice));
        
        dim3 block(256);
        dim3 grid((x.size() + 255) / 256);
        
        gelu_forward_kernel<<<grid, block>>>(d_x, x.size());
        
        CUDA_CHECK(cudaMemcpy(x.data(), d_x, size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_x));
    }

    // Add new utility functions
    bool is_available() {
        int deviceCount = 0;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        return (error == cudaSuccess && deviceCount > 0);
    }

    cudaStream_t get_stream() {
        return nullptr; // Returns default stream for now
    }

    void synchronize() {
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void launch_add_bias(float* output, const float* bias, int rows, int cols) {
        dim3 block(256);
        dim3 grid((rows * cols + block.x - 1) / block.x);
        add_bias_kernel<<<grid, block>>>(output, bias, rows, cols);
    }

    void launch_row_sum(const float* input, float* output, int rows, int cols) {
        dim3 block(256);
        dim3 grid((cols + block.x - 1) / block.x);
        row_sum_kernel<<<grid, block>>>(input, output, rows, cols);
    }

    void launch_adam_update(float* params, const float* grads, float* m, float* v,
                          float beta1, float beta2, float lr, float epsilon, int size,
                          cudaStream_t stream) {
        dim3 block(256);
        dim3 grid((size + block.x - 1) / block.x);
        adam_update_kernel<<<grid, block, 0, stream>>>(params, grads, m, v,
                                                      beta1, beta2, lr, epsilon, size);
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

// Add kernel implementations at the bottom of the file
__global__ void add_bias_kernel(float* output, const float* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int col = idx % cols;
        output[idx] += bias[col];
    }
}

__global__ void row_sum_kernel(const float* input, float* output, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        float sum = 0.0f;
        for (int row = 0; row < rows; row++) {
            sum += input[row * cols + col];
        }
        output[col] = sum;
    }
}

__global__ void adam_update_kernel(float* params, const float* grads, float* m, float* v,
                                 float beta1, float beta2, float lr, float epsilon, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Update biased first moment estimate
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * grads[idx];
        
        // Update biased second raw moment estimate
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * grads[idx] * grads[idx];
        
        // Update parameters
        params[idx] -= lr * m[idx] / (sqrtf(v[idx]) + epsilon);
    }
}