#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "../../include/cuda/matrix_ops.cuh"
#include "../../include/cuda/cuda_check.cuh"
#include "../../include/cuda/cuda_utils.cuh"
#include <unordered_map>

// Forward declare all kernels
__global__ void matrix_multiply_kernel(const float* A, const float* B, float* C, 
                                      int M, int N, int K);
__global__ void gelu_forward_kernel(float* x, int size);

namespace cuda {
    // Add persistent streams
    static const int NUM_STREAMS = 4;
    static cudaStream_t streams[NUM_STREAMS];
    static int current_stream = 0;
    
    // Memory pool for GPU buffers
    struct MemoryPool {
        std::unordered_map<size_t, std::vector<float*>> free_buffers;
        std::unordered_map<float*, size_t> buffer_sizes;
        size_t total_allocated = 0;
        size_t reuse_count = 0;
        
        float* allocate(size_t size) {
            // Check if we have a free buffer of the right size
            auto& buffers = free_buffers[size];
            if (!buffers.empty()) {
                float* buffer = buffers.back();
                buffers.pop_back();
                reuse_count++;
                return buffer;
            }
            
            // Allocate new buffer
            float* buffer;
            CUDA_CHECK(cudaMalloc(&buffer, size * sizeof(float)));
            buffer_sizes[buffer] = size;
            total_allocated += size;
            return buffer;
        }
        
        void free(float* buffer) {
            if (buffer == nullptr) return;
            auto size = buffer_sizes[buffer];
            free_buffers[size].push_back(buffer);
        }
        
        void cleanup() {
            size_t total_freed = 0;
            for (auto& pair : free_buffers) {
                total_freed += pair.first * pair.second.size();
                for (float* buffer : pair.second) {
                    cudaFree(buffer);
                }
            }
            std::cout << "Memory pool cleanup:"
                      << "\n- Total allocated: " << total_allocated << " elements"
                      << "\n- Buffer reuse count: " << reuse_count
                      << "\n- Freed buffers: " << total_freed << " elements" << std::endl;
            free_buffers.clear();
            buffer_sizes.clear();
            total_allocated = 0;
            reuse_count = 0;
        }
    };
    
    static MemoryPool memory_pool;
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

        // Create persistent streams
        for (int i = 0; i < NUM_STREAMS; i++) {
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
        }

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

        cuda_initialized = true;
    }

    void cleanup_cuda() {
        if (cublas_handle != nullptr) {
            memory_pool.cleanup();
            
            // Cleanup streams
            for (int i = 0; i < NUM_STREAMS; i++) {
                cudaStreamDestroy(streams[i]);
            }
            
            cublasDestroy(cublas_handle);
            cublas_handle = nullptr;
            cuda_initialized = false;
        }
    }

    // Helper to get next stream in round-robin fashion
    cudaStream_t get_next_stream() {
        cudaStream_t stream = streams[current_stream];
        current_stream = (current_stream + 1) % NUM_STREAMS;
        return stream;
    }

    void matmul(const Matrix& A, const Matrix& B, Matrix& C) {
        // A: [batch_size x hidden_size]
        // B: [hidden_size x vocab_size]
        // C: [batch_size x vocab_size]
        
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

        // Use memory pool instead of direct allocation
        float *d_A = memory_pool.allocate(A.rows() * A.cols());
        float *d_B = memory_pool.allocate(B.rows() * B.cols());
        float *d_C = memory_pool.allocate(C.rows() * C.cols());

        size_t A_size = A.rows() * A.cols() * sizeof(float);
        size_t B_size = B.rows() * B.cols() * sizeof(float);
        size_t C_size = C.rows() * C.cols() * sizeof(float);

        // Use persistent stream
        cudaStream_t stream = get_next_stream();
        
        CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), A_size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), B_size, cudaMemcpyHostToDevice, stream));

        float alpha = 1.0f;
        float beta = 0.0f;

        cublasSetStream(cublas_handle, stream);

        cublasStatus_t status = cublasSgemm(cublas_handle,
                                          CUBLAS_OP_N, CUBLAS_OP_N,
                                          C.cols(), C.rows(), A.cols(),
                                          &alpha,
                                          d_B, B.cols(),
                                          d_A, A.cols(),
                                          &beta,
                                          d_C, C.cols());

        if (status != CUBLAS_STATUS_SUCCESS) {
            memory_pool.free(d_A);
            memory_pool.free(d_B);
            memory_pool.free(d_C);
            throw std::runtime_error("cuBLAS matrix multiplication failed: " + std::to_string(status));
        }

        CUDA_CHECK(cudaMemcpyAsync(C.data(), d_C, C_size, cudaMemcpyDeviceToHost, stream));
        
        // Only synchronize the specific stream
        cudaStreamSynchronize(stream);

        memory_pool.free(d_A);
        memory_pool.free(d_B);
        memory_pool.free(d_C);
    }

    void matmul_transposed(const Matrix& A, const Matrix& B, Matrix& C) {
        // Ensure CUDA is initialized
        if (!cuda_initialized || cublas_handle == nullptr) {
            initialize_cuda();
        }

        // Verify dimensions for transposed multiplication
        if (A.cols() != B.cols()) {
            throw std::runtime_error("Matrix multiplication dimension mismatch for transposed operation: " +
                std::to_string(A.rows()) + "x" + std::to_string(A.cols()) + " * " +
                std::to_string(B.rows()) + "x" + std::to_string(B.cols()));
        }
        
        if (C.rows() != A.rows() || C.cols() != B.rows()) {
            throw std::runtime_error("Output matrix has wrong dimensions: expected " +
                std::to_string(A.rows()) + "x" + std::to_string(B.rows()) + " got " +
                std::to_string(C.rows()) + "x" + std::to_string(C.cols()));
        }

        // Use memory pool instead of direct allocation
        float *d_A = memory_pool.allocate(A.rows() * A.cols());
        float *d_B = memory_pool.allocate(B.rows() * B.cols());
        float *d_C = memory_pool.allocate(C.rows() * C.cols());

        size_t A_size = A.rows() * A.cols() * sizeof(float);
        size_t B_size = B.rows() * B.cols() * sizeof(float);
        size_t C_size = C.rows() * C.cols() * sizeof(float);

        // Use persistent stream
        cudaStream_t stream = get_next_stream();
        
        CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), A_size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), B_size, cudaMemcpyHostToDevice, stream));

        float alpha = 1.0f;
        float beta = 0.0f;

        cublasSetStream(cublas_handle, stream);

        cublasStatus_t status = cublasSgemm(cublas_handle,
                                          CUBLAS_OP_T, CUBLAS_OP_N,
                                          B.rows(), A.rows(), A.cols(),
                                          &alpha,
                                          d_B, B.cols(),
                                          d_A, A.cols(),
                                          &beta,
                                          d_C, B.rows());

        if (status != CUBLAS_STATUS_SUCCESS) {
            memory_pool.free(d_A);
            memory_pool.free(d_B);
            memory_pool.free(d_C);
            throw std::runtime_error("cuBLAS matrix multiplication failed: " + std::to_string(status));
        }

        CUDA_CHECK(cudaMemcpyAsync(C.data(), d_C, C_size, cudaMemcpyDeviceToHost, stream));
        
        // Only synchronize the specific stream
        cudaStreamSynchronize(stream);

        memory_pool.free(d_A);
        memory_pool.free(d_B);
        memory_pool.free(d_C);
    }

    void gelu_forward(Matrix& x) {
        // Use memory pool instead of direct allocation
        float* d_x = memory_pool.allocate(x.size());
        size_t size = x.size() * sizeof(float);
        
        // Use persistent stream
        cudaStream_t stream = get_next_stream();
        
        CUDA_CHECK(cudaMemcpyAsync(d_x, x.data(), size, cudaMemcpyHostToDevice, stream));
        
        dim3 block(256);
        dim3 grid((x.size() + 255) / 256);
        
        gelu_forward_kernel<<<grid, block, 0, stream>>>(d_x, x.size());
        
        CUDA_CHECK(cudaMemcpyAsync(x.data(), d_x, size, cudaMemcpyDeviceToHost, stream));
        
        cudaStreamSynchronize(stream);
        
        memory_pool.free(d_x);
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