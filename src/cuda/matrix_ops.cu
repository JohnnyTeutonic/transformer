#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mutex>
#include <thread>  // Add this for sleep_for
#include <chrono>  // Add this for milliseconds
#include <queue>
#include <vector>
#include "../../include/cuda/matrix_ops.cuh"
#include "../../include/cuda/cuda_check.cuh"
#include "../../include/cuda/cuda_utils.cuh"

// Forward declare all kernels
__global__ void matrix_multiply_kernel(const float* A, const float* B, float* C, 
                                      int M, int N, int K);
__global__ void gelu_forward_kernel(float* x, int size);

namespace cuda {
    namespace internal {  // Internal namespace for resource management
        struct CUDAResources {
            cublasHandle_t cublas_handle;
            bool initialized;
            std::mutex mutex;
            
            CUDAResources() : cublas_handle(nullptr), initialized(false) {}
            
            ~CUDAResources() {
                if (cublas_handle) {
                    cublasDestroy(cublas_handle);
                    cublas_handle = nullptr;
                }
            }
        };
        
        static CUDAResources resources;
    }
    
    // Stream pool management
    static const int MAX_STREAMS = 8;
    static std::vector<cudaStream_t> stream_pool;
    static std::queue<cudaStream_t> available_streams;
    static std::mutex stream_mutex;

    // Add the forward declaration here
    void cleanup_stream_pool();

    // Stream management functions
    bool initialize_stream_pool() {
        std::lock_guard<std::mutex> lock(stream_mutex);
        
        if (!stream_pool.empty()) {
            return true;  // Already initialized
        }

        try {
            // Create all streams upfront
            for (int i = 0; i < MAX_STREAMS; i++) {
                cudaStream_t stream;
                cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
                if (err != cudaSuccess) {
                    std::cout << "Failed to create stream " << i << ": " << cudaGetErrorString(err) << std::endl;
                    cleanup_stream_pool();
                    return false;
                }
                stream_pool.push_back(stream);
                available_streams.push(stream);
            }
            return true;
        } catch (const std::exception& e) {
            std::cout << "Exception in stream pool initialization: " << e.what() << std::endl;
            cleanup_stream_pool();
            return false;
        }
    }

    void cleanup_stream_pool() {
        // Synchronize and destroy all streams in the pool
        for (cudaStream_t& stream : stream_pool) {
            if (stream != nullptr) {
                cudaStreamSynchronize(stream);
                cudaStreamDestroy(stream);
            }
        }
        stream_pool.clear();
    }

    // RAII stream handler
    class StreamGuard {
        cudaStream_t stream;
        bool valid;

    public:
        StreamGuard() : stream(nullptr), valid(false) {
            std::lock_guard<std::mutex> lock(stream_mutex);
            if (!available_streams.empty()) {
                stream = available_streams.front();
                available_streams.pop();
                valid = true;
            }
        }

        ~StreamGuard() {
            if (valid) {
                cudaStreamSynchronize(stream);
                std::lock_guard<std::mutex> lock(stream_mutex);
                available_streams.push(stream);
            }
        }

        bool is_valid() const { return valid; }
        cudaStream_t get() const { return stream; }
    };

    // CPU fallback implementation
    static void cpu_matmul(const Matrix& A, const Matrix& B, Matrix& C) {
        if (A.cols() != B.rows()) {
            throw std::runtime_error("Matrix multiplication dimension mismatch");
        }
        
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < A.rows(); i++) {
            for (size_t j = 0; j < B.cols(); j++) {
                float sum = 0.0f;
                #pragma omp simd reduction(+:sum)
                for (size_t k = 0; k < A.cols(); k++) {
                    sum += A(i, k) * B(k, j);
                }
                C(i, j) = sum;
            }
        }
    }

    // Enhanced context validation
    bool validate_cuda_context() {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cout << "CUDA context validation failed: " << cudaGetErrorString(err) << std::endl;
            return false;
        }

        int device;
        err = cudaGetDevice(&device);
        if (err != cudaSuccess) {
            std::cout << "Failed to get CUDA device: " << cudaGetErrorString(err) << std::endl;
            return false;
        }

        cudaDeviceProp props;
        err = cudaGetDeviceProperties(&props, device);
        if (err != cudaSuccess) {
            std::cout << "Failed to get device properties: " << cudaGetErrorString(err) << std::endl;
            return false;
        }

        size_t free_mem, total_mem;
        err = cudaMemGetInfo(&free_mem, &total_mem);
        if (err != cudaSuccess || free_mem == 0 || total_mem == 0) {
            std::cout << "Cannot access GPU memory: " << cudaGetErrorString(err) << std::endl;
            return false;
        }

        return true;
    }

    bool recover_cuda_context() {
        std::cout << "Attempting to recover CUDA context..." << std::endl;
        
        // First, try to reset the device
        cudaError_t err = cudaDeviceReset();
        if (err != cudaSuccess) {
            std::cout << "Device reset failed: " << cudaGetErrorString(err) << std::endl;
            return false;
        }

        // Clear any existing errors
        cudaGetLastError();

        // Reinitialize CUDA
        err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            std::cout << "Failed to set CUDA device: " << cudaGetErrorString(err) << std::endl;
            return false;
        }

        // Recreate cuBLAS handle
        if (internal::resources.cublas_handle) {
            cublasDestroy(internal::resources.cublas_handle);
            internal::resources.cublas_handle = nullptr;
        }

        cublasStatus_t status = cublasCreate(&internal::resources.cublas_handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cout << "Failed to create cuBLAS handle: " << status << std::endl;
            return false;
        }

        // Reinitialize stream pool
        cleanup_stream_pool();
        if (!initialize_stream_pool()) {
            std::cout << "Failed to reinitialize stream pool" << std::endl;
            return false;
        }

        return validate_cuda_context();
    }

    bool initialize_cuda() {
        std::lock_guard<std::mutex> lock(internal::resources.mutex);
        static int failure_count = 0;
        const int MAX_GLOBAL_FAILURES = 3;
        const int MAX_RETRY_ATTEMPTS = 3;
        const int BASE_DELAY_MS = 100;

        if (failure_count >= MAX_GLOBAL_FAILURES) {
            std::cerr << "Too many CUDA failures, disabling CUDA operations" << std::endl;
            return false;
        }

        if (internal::resources.initialized) {
            if (!validate_cuda_context()) {
                std::cout << "Existing CUDA context invalid, attempting recovery..." << std::endl;
                if (!recover_cuda_context()) {
                    internal::resources.initialized = false;
                    failure_count++;
                    return false;
                }
            }
            return true;
        }

        for (int attempt = 0; attempt < MAX_RETRY_ATTEMPTS; attempt++) {
            try {
                if (attempt > 0) {
                    int delay_ms = BASE_DELAY_MS * (1 << attempt);
                    std::cout << "Attempt " << (attempt + 1) << ", waiting " << delay_ms << "ms..." << std::endl;
                    std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
                }

                // Get number of devices
                int deviceCount;
                cudaError_t error = cudaGetDeviceCount(&deviceCount);
                if (error != cudaSuccess) {
                    std::cerr << "Failed to get CUDA device count: " << cudaGetErrorString(error) << std::endl;
                    continue;
                }

                if (deviceCount == 0) {
                    std::cerr << "No CUDA-capable devices found" << std::endl;
                    continue;
                }

                // Get device properties
                cudaDeviceProp deviceProp;
                error = cudaGetDeviceProperties(&deviceProp, 0); // Use first device
                if (error != cudaSuccess) {
                    std::cerr << "Failed to get device properties: " << cudaGetErrorString(error) << std::endl;
                    continue;
                }

                // Print device info
                std::cout << "Using CUDA Device " << 0 << ": " << deviceProp.name << std::endl;

                // Set device
                error = cudaSetDevice(0);
                if (error != cudaSuccess) {
                    std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(error) << std::endl;
                    continue;
                }

                // Initialize cuBLAS handle
                if (internal::resources.cublas_handle == nullptr) {
                    cublasStatus_t status = cublasCreate(&internal::resources.cublas_handle);
                    if (status != CUBLAS_STATUS_SUCCESS) {
                        std::cerr << "Failed to create cuBLAS handle" << std::endl;
                        continue;
                    }
                }

                // Initialize stream pool
                if (!initialize_stream_pool()) {
                    if (internal::resources.cublas_handle) {
                        cublasDestroy(internal::resources.cublas_handle);
                        internal::resources.cublas_handle = nullptr;
                    }
                    continue;
                }

                // Ensure device is ready
                error = cudaDeviceSynchronize();
                if (error != cudaSuccess) {
                    std::cerr << "Failed to synchronize device: " << cudaGetErrorString(error) << std::endl;
                    cleanup_cuda();
                    continue;
                }

                internal::resources.initialized = true;
                failure_count = 0;
                return true;

            } catch (const std::exception& e) {
                std::cout << "CUDA initialization attempt " << (attempt + 1) 
                         << " failed: " << e.what() << std::endl;
                cleanup_cuda();
            }
        }

        failure_count++;
        return false;
    }

    void cleanup_cuda() {
        std::lock_guard<std::mutex> lock(internal::resources.mutex);
        
        // First synchronize all streams
        for (auto& stream : stream_pool) {
            if (stream != nullptr) {
                cudaStreamSynchronize(stream);
            }
        }

        // Then cleanup streams
        cleanup_stream_pool();

        // Finally destroy cuBLAS handle
        if (internal::resources.cublas_handle != nullptr) {
            cublasDestroy(internal::resources.cublas_handle);
            internal::resources.cublas_handle = nullptr;
        }

        internal::resources.initialized = false;
        
        // Reset device last
        cudaDeviceReset();
        
        // Clear any remaining errors
        cudaGetLastError();
    }

    // Add a function to check if CUDA is usable
    bool is_cuda_available() {
        if (!internal::resources.initialized) {
            try {
                initialize_cuda();
            } catch (const std::exception& e) {
                std::cout << "CUDA initialization check failed: " << e.what() << std::endl;
                return false;
            }
        }
        return internal::resources.initialized;
    }

    void matmul(const Matrix& A, const Matrix& B, Matrix& C) {
        std::cout << "\nEntering cuda::matmul..." << std::endl;
        
        // First check if CUDA is available
        if (!is_cuda_available()) {
            std::cout << "CUDA not available, falling back to CPU implementation" << std::endl;
            cpu_matmul(A, B, C);
            return;
        }

        // Get a stream from the pool
        StreamGuard stream_guard;
        if (!stream_guard.is_valid()) {
            std::cout << "No available CUDA streams, falling back to CPU implementation" << std::endl;
            cpu_matmul(A, B, C);
            return;
        }

        // Verify dimensions
        if (A.cols() != B.rows()) {
            throw std::runtime_error("Matrix multiplication dimension mismatch");
        }

        float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

        // Helper lambda for cleanup
        auto cleanup = [&]() {
            if (d_A) cudaFree(d_A);
            if (d_B) cudaFree(d_B);
            if (d_C) cudaFree(d_C);
        };

        try {
            // Add synchronization point before validation
            cudaStreamSynchronize(stream_guard.get());
            
            if (!validate_cuda_context()) {
                std::cout << "CUDA context validation failed, resetting device..." << std::endl;
                cleanup_cuda();  // Proper cleanup before reset
                initialize_cuda();  // Reinitialize everything
                
                if (!validate_cuda_context()) {
                    throw std::runtime_error("CUDA context validation failed after reset");
                }
            }

            // Calculate sizes and allocate memory
            size_t A_size = A.rows() * A.cols() * sizeof(float);
            size_t B_size = B.rows() * B.cols() * sizeof(float);
            size_t C_size = C.rows() * C.cols() * sizeof(float);

            CUDA_CHECK(cudaMalloc(&d_A, A_size));
            CUDA_CHECK(cudaMalloc(&d_B, B_size));
            CUDA_CHECK(cudaMalloc(&d_C, C_size));

            // Copy data using the stream
            CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), A_size, cudaMemcpyHostToDevice, stream_guard.get()));
            CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), B_size, cudaMemcpyHostToDevice, stream_guard.get()));

            // Set stream for cuBLAS
            cublasStatus_t cublas_status = cublasSetStream(
                internal::resources.cublas_handle, 
                stream_guard.get()
            );
            if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("Failed to set cuBLAS stream");
            }

            float alpha = 1.0f;
            float beta = 0.0f;

            // Perform multiplication
            cublasStatus_t status = cublasSgemm(
                internal::resources.cublas_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                B.cols(), A.rows(), A.cols(),
                &alpha,
                d_B, B.cols(),
                d_A, A.cols(),
                &beta,
                d_C, C.cols()
            );

            if (status != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("cuBLAS matrix multiplication failed");
            }

            // Copy result back
            CUDA_CHECK(cudaMemcpyAsync(C.data(), d_C, C_size, cudaMemcpyDeviceToHost, stream_guard.get()));

            // Add synchronization point after computation
            cudaStreamSynchronize(stream_guard.get());
            
            cleanup();

        } catch (const std::exception& e) {
            std::cout << "Exception in matrix multiplication: " << e.what() << std::endl;
            cleanup();
            
            // Try to recover context and fall back to CPU if needed
            if (!recover_cuda_context()) {
                std::cout << "Failed to recover CUDA context, falling back to CPU implementation" << std::endl;
                cpu_matmul(A, B, C);
            } else {
                throw; // Re-throw if recovery succeeded but operation still failed
            }
        }
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