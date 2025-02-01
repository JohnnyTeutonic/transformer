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
                                 float beta1, float beta2, float epsilon, float lr, int size,
                                 int step);

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

    Matrix matmul(const Matrix& A, const Matrix& B) {
        // Create output matrix
        Matrix C(A.rows(), B.cols());
        // Call the stream version with nullptr stream
        matmul(A, B, C, nullptr);
        return C;
    }

    void gelu_forward(Matrix& x) {
        size_t size = x.size() * sizeof(float);
        float* d_x;
        CUDA_CHECK(cudaMalloc(&d_x, size));
        CUDA_CHECK(cudaMemcpy(d_x, x.data(), size, cudaMemcpyHostToDevice));
        
        dim3 block(256);
        dim3 grid((x.size() + 255) / 256);
        
        gelu_forward_kernel<<<grid, block>>>(d_x, x.size());
        
        CUDA_CHECK(cudaMemcpy(x.data(), d_x, size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_x));
    }

    void launch_add_bias(float* output, const float* bias, int batch_size, int hidden_size) {
        dim3 block(256);
        dim3 grid((batch_size * hidden_size + block.x - 1) / block.x);
        
        add_bias_kernel<<<grid, block>>>(output, bias, batch_size, hidden_size);
        CUDA_CHECK(cudaGetLastError());
    }

    void launch_row_sum(const float* input, float* output, int rows, int cols, cudaStream_t stream) {
        const int block_size = 256;
        const int num_blocks = (rows + block_size - 1) / block_size;
        row_sum_kernel<<<num_blocks, block_size, 0, stream>>>(input, output, rows, cols);
        CUDA_CHECK(cudaGetLastError());
        if (stream == nullptr) {
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    void matmul(const Matrix& a, const Matrix& b, Matrix& c, cudaStream_t stream) {
        // Ensure CUDA is initialized
        if (!cuda_initialized || cublas_handle == nullptr) {
            initialize_cuda();
        }

        // Verify dimensions
        if (a.cols() != b.rows()) {
            throw std::runtime_error("Matrix multiplication dimension mismatch: " +
                std::to_string(a.rows()) + "x" + std::to_string(a.cols()) + " * " +
                std::to_string(b.rows()) + "x" + std::to_string(b.cols()));
        }

        // Verify output matrix dimensions
        if (c.rows() != a.rows() || c.cols() != b.cols()) {
            throw std::runtime_error("Output matrix dimensions mismatch");
        }

        std::cout << "\n=== Matrix Multiplication Debug Info ===" << std::endl;
        std::cout << "Matrix A: " << a.rows() << "x" << a.cols() << std::endl;
        std::cout << "Matrix B: " << b.rows() << "x" << b.cols() << std::endl;
        std::cout << "Matrix C: " << c.rows() << "x" << c.cols() << std::endl;

        float* d_A, *d_B, *d_C;
        size_t A_size = a.rows() * a.cols() * sizeof(float);
        size_t B_size = b.rows() * b.cols() * sizeof(float);
        size_t C_size = c.rows() * c.cols() * sizeof(float);

        std::cout << "Allocating device memory..." << std::endl;
        std::cout << "A_size: " << A_size << " bytes" << std::endl;
        std::cout << "B_size: " << B_size << " bytes" << std::endl;
        std::cout << "C_size: " << C_size << " bytes" << std::endl;

        CUDA_CHECK(cudaMalloc(&d_A, A_size));
        CUDA_CHECK(cudaMalloc(&d_B, B_size));
        CUDA_CHECK(cudaMalloc(&d_C, C_size));

        // Use stream for async operations if provided
        cudaStream_t compute_stream = stream ? stream : cudaStream_t(nullptr);
        
        CUDA_CHECK(cudaMemcpyAsync(d_A, a.data(), A_size, cudaMemcpyHostToDevice, compute_stream));
        CUDA_CHECK(cudaMemcpyAsync(d_B, b.data(), B_size, cudaMemcpyHostToDevice, compute_stream));

        float alpha = 1.0f;
        float beta = 0.0f;

        std::cout << "cuBLAS parameters:" << std::endl;
        std::cout << "alpha: " << alpha << ", beta: " << beta << std::endl;
        std::cout << "Leading dimensions:" << std::endl;
        std::cout << "lda (A): " << a.rows() << " (rows)" << std::endl;
        std::cout << "ldb (B): " << b.rows() << " (rows)" << std::endl;
        std::cout << "ldc (C): " << c.cols() << " (cols)" << std::endl;
        std::cout << "Operation dimensions (m,n,k): " << c.cols() << "," << c.rows() << "," << a.cols() << std::endl;

        // Set stream for cuBLAS operations
        if (stream) {
            CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
        }

        // Using transposed operations to handle row-major to column-major conversion
        cublasStatus_t status = cublasSgemm(cublas_handle,
                                          CUBLAS_OP_T, CUBLAS_OP_T,  // Use transposed operations
                                          c.cols(), c.rows(), a.cols(),  // m, n, k dimensions
                                          &alpha,
                                          d_B, ((b.rows() + 31) / 32) * 32,  // Pad leading dimension to multiple of 32
                                          d_A, ((a.rows() + 31) / 32) * 32,  // Pad leading dimension to multiple of 32
                                          &beta,
                                          d_C, ((c.cols() + 31) / 32) * 32); // Pad leading dimension to multiple of 32

        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cout << "cuBLAS error status: " << status << std::endl;
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
            throw std::runtime_error("cuBLAS matrix multiplication failed with status: " + std::to_string(status));
        }

        std::cout << "Matrix multiplication completed, copying results back..." << std::endl;
        CUDA_CHECK(cudaMemcpyAsync(c.data(), d_C, C_size, cudaMemcpyDeviceToHost, compute_stream));
        
        // Ensure all operations are complete
        if (stream) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
        } else {
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Cleanup
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
        
        std::cout << "Matrix multiplication completed successfully" << std::endl;
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
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float sum = 0.0f;
        for (int col = 0; col < cols; ++col) {
            sum += input[row * cols + col];
        }
        output[row] = sum;
    }
}

__global__ void adam_update_kernel(float* params, const float* grads, float* m, float* v,
                                 float beta1, float beta2, float epsilon, float lr, int size,
                                 int step) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Update biased first moment estimate
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * grads[idx];
        
        // Update biased second raw moment estimate
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * grads[idx] * grads[idx];
        
        // Compute bias correction terms
        float m_hat = m[idx] / (1.0f - powf(beta1, step));
        float v_hat = v[idx] / (1.0f - powf(beta2, step));
        
        // Update parameters with bias correction
        params[idx] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
    }
}