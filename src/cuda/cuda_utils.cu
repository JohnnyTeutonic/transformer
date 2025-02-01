#include "../../include/cuda/cuda_utils.cuh"
#include "../../include/cuda/cuda_check.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

namespace cuda {

static bool cuda_initialized = false;
static cudaStream_t compute_stream = nullptr;
static cublasHandle_t cublas_handle = nullptr;

bool is_available() {
    int device_count;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    return (error == cudaSuccess && device_count > 0 && cuda_initialized);
}

void initialize() {
    if (cuda_initialized) return;

    try {
        // Set device
        CUDA_CHECK(cudaSetDevice(0));

        // Create stream
        CUDA_CHECK(cudaStreamCreate(&compute_stream));

        // Initialize cuBLAS
        CUBLAS_CHECK(cublasCreate(&cublas_handle));
        CUBLAS_CHECK(cublasSetStream(cublas_handle, compute_stream));

        cuda_initialized = true;
        std::cout << "CUDA initialized successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "CUDA initialization failed: " << e.what() << std::endl;
        cleanup();
        throw;
    }
}

void cleanup() {
    if (cublas_handle) {
        cublasDestroy(cublas_handle);
        cublas_handle = nullptr;
    }
    if (compute_stream) {
        cudaStreamDestroy(compute_stream);
        compute_stream = nullptr;
    }
    cuda_initialized = false;
}

cudaStream_t get_stream() {
    return compute_stream;
}

cublasHandle_t get_cublas_handle() {
    return cublas_handle;
}

void synchronize() {
    if (compute_stream) {
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    }
}

template<typename T>
T* device_malloc(size_t size) {
    T* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, size * sizeof(T)));
    return ptr;
}

template<typename T>
void device_free(T* ptr) {
    if (ptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

template<typename T>
void copy_to_device(T* dst, const T* src, size_t size) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, size * sizeof(T), 
                              cudaMemcpyHostToDevice, compute_stream));
}

template<typename T>
void copy_to_host(T* dst, const T* src, size_t size) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, size * sizeof(T), 
                              cudaMemcpyDeviceToHost, compute_stream));
}

// Explicit template instantiations
template float* device_malloc<float>(size_t);
template void device_free<float>(float*);
template void copy_to_device<float>(float*, const float*, size_t);
template void copy_to_host<float>(float*, const float*, size_t);

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

Matrix cuda_matmul(const Matrix& A, const Matrix& B) {
    std::cout << "Starting CUDA matrix multiplication..." << std::endl;
    std::cout << "Matrix A: " << A.rows() << "x" << A.cols() << std::endl;
    std::cout << "Matrix B: " << B.rows() << "x" << B.cols() << std::endl;

    cublasHandle_t handle;
    cublasStatus_t status;
    cudaError_t err;

    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to create cuBLAS handle");
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    Matrix C(A.rows(), B.cols(), 0.0f);
    Matrix C_gpu = C.to_gpu();
    std::cout << "Created output matrix C: " << C.rows() << "x" << C.cols() << std::endl;

    try {
        status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, B.cols(), A.rows(), A.cols(), &alpha,
                             B.get_data(), B.cols(), A.get_data(), A.cols(), &beta,
                             C_gpu.get_data(), C_gpu.cols());

        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cuBLAS SGEMM failed with status: " + std::to_string(status));
        }

        // Synchronize to catch any asynchronous errors
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA sync failed: " + std::string(cudaGetErrorString(err)));
        }

        std::cout << "CUDA matrix multiplication completed successfully" << std::endl;
        C = C_gpu.to_cpu();
    } catch (const std::exception& e) {
        cublasDestroy(handle);
        throw;
    }

    cublasDestroy(handle);
    return C;
}

} // namespace cuda