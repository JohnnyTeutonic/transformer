#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "../../include/cuda/matrix_ops.cuh"
#include "../../include/cuda/cuda_check.cuh"
#include "../../include/cuda/cuda_utils.cuh"
#include <unordered_map>
#include <string>
#include <chrono>
#include <cstdlib>
#include <cstring>

// Forward declare all kernels
__global__ void matrix_multiply_kernel(const float* A, const float* B, float* C, 
                                      int M, int N, int K);
__global__ void gelu_forward_kernel(float* x, int size);
__global__ void gelu_backward_kernel(float* grad_output, const float* input, int size);
__global__ void add_bias_kernel(float* matrix, const float* bias, int rows, int cols);

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
            // Ensure all in-flight GPU work has completed before freeing any
            // device memory or destroying streams/handles. Freeing buffers that
            // are still referenced by pending kernels/copies corrupts the
            // allocator and causes an access violation at teardown.
            cudaDeviceSynchronize();

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

    // Expose the shared cuBLAS handle so other translation units (e.g. the
    // device-resident LM head projection in loss_kernels.cu) can reuse it
    // instead of creating a second handle. Creating a second handle led to a
    // crash during process teardown (cuBLAS atexit cleanup vs. context destroy).
    cublasHandle_t get_cublas_handle() {
        if (!cuda_initialized || cublas_handle == nullptr) {
            initialize_cuda();
        }
        return cublas_handle;
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

        // ---- Optional phase profiling (set MATMUL_PROFILE=1) ----
        static const bool s_prof = (std::getenv("MATMUL_PROFILE") != nullptr);
        static long long s_calls = 0;
        static double s_alloc = 0, s_launch = 0, s_sync = 0, s_free = 0;
        static double s_bytes = 0;
        using clk = std::chrono::steady_clock;
        auto p0 = s_prof ? clk::now() : clk::time_point{};

        // Use memory pool instead of direct allocation
        float *d_A = memory_pool.allocate(A.rows() * A.cols());
        float *d_B = memory_pool.allocate(B.rows() * B.cols());
        float *d_C = memory_pool.allocate(C.rows() * C.cols());

        size_t A_size = A.rows() * A.cols() * sizeof(float);
        size_t B_size = B.rows() * B.cols() * sizeof(float);
        size_t C_size = C.rows() * C.cols() * sizeof(float);

        auto p1 = s_prof ? clk::now() : clk::time_point{};

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

        auto p2 = s_prof ? clk::now() : clk::time_point{};
        
        // Only synchronize the specific stream
        cudaStreamSynchronize(stream);

        auto p3 = s_prof ? clk::now() : clk::time_point{};

        memory_pool.free(d_A);
        memory_pool.free(d_B);
        memory_pool.free(d_C);

        if (s_prof) {
            auto p4 = clk::now();
            auto us = [](clk::time_point a, clk::time_point b){
                return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count(); };
            s_alloc  += us(p0, p1);
            s_launch += us(p1, p2);
            s_sync   += us(p2, p3);
            s_free   += us(p3, p4);
            s_bytes  += (double)(A_size + B_size + C_size);
            if (++s_calls % 200 == 0) {
                printf("[MATMUL_PROFILE] calls=%lld avg(us): alloc=%.1f launch=%.1f sync=%.1f free=%.1f | avgMB=%.2f\n",
                       s_calls, s_alloc/s_calls, s_launch/s_calls, s_sync/s_calls, s_free/s_calls,
                       (s_bytes/s_calls)/1e6);
                fflush(stdout);
            }
        }
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

    void gelu_backward(Matrix& grad_output, const Matrix& input) {
        // GELU derivative: d/dx[x * Φ(x)] where Φ is CDF of standard normal (approximated)
        // grad = grad_output * (Φ(x) + x * φ(x))
        float* d_grad = memory_pool.allocate(grad_output.size());
        float* d_input = memory_pool.allocate(input.size());
        size_t size = grad_output.size() * sizeof(float);
        
        cudaStream_t stream = get_next_stream();
        
        CUDA_CHECK(cudaMemcpyAsync(d_grad, grad_output.data(), size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_input, input.data(), size, cudaMemcpyHostToDevice, stream));
        
        dim3 block(256);
        dim3 grid((grad_output.size() + 255) / 256);
        
        gelu_backward_kernel<<<grid, block, 0, stream>>>(d_grad, d_input, grad_output.size());
        
        CUDA_CHECK(cudaMemcpyAsync(grad_output.data(), d_grad, size, cudaMemcpyDeviceToHost, stream));
        
        cudaStreamSynchronize(stream);
        
        memory_pool.free(d_grad);
        memory_pool.free(d_input);
    }

    void add_bias(Matrix& matrix, const Vector& bias) {
        // Add bias vector to each row of matrix
        // matrix shape: [batch_size, features]
        // bias shape: [features]
        float* d_matrix = memory_pool.allocate(matrix.size());
        float* d_bias = memory_pool.allocate(bias.size());
        
        size_t matrix_size = matrix.size() * sizeof(float);
        size_t bias_size = bias.size() * sizeof(float);
        
        cudaStream_t stream = get_next_stream();
        
        CUDA_CHECK(cudaMemcpyAsync(d_matrix, matrix.data(), matrix_size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_bias, bias.data(), bias_size, cudaMemcpyHostToDevice, stream));
        
        dim3 block(256);
        dim3 grid((matrix.size() + 255) / 256);
        
        add_bias_kernel<<<grid, block, 0, stream>>>(d_matrix, d_bias, matrix.rows(), matrix.cols());
        
        CUDA_CHECK(cudaMemcpyAsync(matrix.data(), d_matrix, matrix_size, cudaMemcpyDeviceToHost, stream));
        
        cudaStreamSynchronize(stream);
        
        memory_pool.free(d_matrix);
        memory_pool.free(d_bias);
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

__global__ void gelu_backward_kernel(float* grad_output, const float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        // GELU(x) = x * Φ(x) where Φ is CDF of standard normal (approximated with tanh)
        // GELU'(x) = Φ(x) + x * φ(x)
        // Using tanh approximation: Φ(x) ≈ 0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
        float x_cubed = x * x * x;
        float inner = 0.797884f * (x + 0.044715f * x_cubed);
        float tanh_inner = tanhf(inner);
        float cdf = 0.5f * (1.0f + tanh_inner);
        
        // Derivative of tanh approximation
        float tanh_deriv = 1.0f - tanh_inner * tanh_inner;
        float inner_deriv = 0.797884f * (1.0f + 0.134145f * x * x);
        float pdf = 0.5f * tanh_deriv * inner_deriv;
        
        // GELU derivative
        float gelu_grad = cdf + x * pdf;
        
        grad_output[idx] *= gelu_grad;
    }
}

__global__ void add_bias_kernel(float* matrix, const float* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int col = idx % cols;
        matrix[idx] += bias[col];
    }
}

__global__ void softmax_kernel_rowwise(float* matrix, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    
    extern __shared__ float shared[];
    float* row_data = shared;
    
    // Load row into shared memory and find max
    float max_val = -INFINITY;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float val = matrix[row * cols + j];
        row_data[j] = val;
        max_val = fmaxf(max_val, val);
    }
    
    // Reduce to find global max across threads
    __shared__ float shared_max;
    if (threadIdx.x == 0) shared_max = -INFINITY;
    __syncthreads();
    atomicMax((int*)&shared_max, __float_as_int(max_val));
    __syncthreads();
    max_val = shared_max;
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float exp_val = expf(row_data[j] - max_val);
        row_data[j] = exp_val;
        sum += exp_val;
    }
    
    // Reduce sum across threads
    __shared__ float shared_sum;
    if (threadIdx.x == 0) shared_sum = 0.0f;
    __syncthreads();
    atomicAdd(&shared_sum, sum);
    __syncthreads();
    sum = shared_sum;
    
    // Normalize and write back
    float inv_sum = 1.0f / (sum + 1e-10f);
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        matrix[row * cols + j] = row_data[j] * inv_sum;
    }
}

namespace cuda {
    void softmax(Matrix& matrix) {
        if (!cuda_initialized) {
            initialize_cuda();
        }
        
        float* d_matrix = memory_pool.allocate(matrix.size());
        size_t size = matrix.size() * sizeof(float);
        
        cudaStream_t stream = get_next_stream();
        
        CUDA_CHECK(cudaMemcpyAsync(d_matrix, matrix.data(), size, cudaMemcpyHostToDevice, stream));
        
        // Launch one block per row
        int block_size = 256;
        if (matrix.cols() < 256) block_size = 128;
        if (matrix.cols() < 128) block_size = 64;
        size_t shared_mem = matrix.cols() * sizeof(float);
        
        softmax_kernel_rowwise<<<matrix.rows(), block_size, shared_mem, stream>>>(
            d_matrix, matrix.rows(), matrix.cols());
        
        CUDA_CHECK(cudaMemcpyAsync(matrix.data(), d_matrix, size, cudaMemcpyDeviceToHost, stream));
        
        cudaStreamSynchronize(stream);
        
        memory_pool.free(d_matrix);
    }
}