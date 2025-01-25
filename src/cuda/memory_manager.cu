#include "../../include/cuda/memory_manager.hpp"
#include "../../include/matrix.hpp"
#include <cuda_runtime.h>

namespace cuda {

std::unique_ptr<Matrix> MemoryManager::allocate_matrix(size_t rows, size_t cols) {
    // Check if CUDA device is available and initialized
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to get CUDA device count: " + 
                               std::string(cudaGetErrorString(err)));
    }
    if (deviceCount == 0) {
        throw std::runtime_error("No CUDA devices available");
    }

    // Get and print device properties
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to get device properties: " + 
                               std::string(cudaGetErrorString(err)));
    }
    
    std::cout << "CUDA Device: " << prop.name << std::endl;
    std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    
    // Calculate size and check against available memory
    size_t size = rows * cols * sizeof(float);
    size_t free_mem, total_mem;
    err = cudaMemGetInfo(&free_mem, &total_mem);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to get memory info: " + 
                               std::string(cudaGetErrorString(err)));
    }
    
    std::cout << "Allocating " << size / 1024.0f << " KB" << std::endl;
    std::cout << "Free memory: " << free_mem / (1024*1024) << " MB" << std::endl;
    
    if (size > free_mem) {
        throw std::runtime_error("Not enough GPU memory. Required: " + 
                               std::to_string(size/1024.0f) + " KB, Available: " + 
                               std::to_string(free_mem/1024.0f) + " KB");
    }

    // Attempt allocation
    float* device_ptr = nullptr;
    printf("Attempting to allocate %zu bytes\n", size);
    err = cudaMalloc(&device_ptr, size);
    printf("cudaMalloc returned %d\n", err);
    if (err != cudaSuccess || device_ptr == nullptr) {
        throw std::runtime_error("CUDA memory allocation failed: " + 
                               std::string(cudaGetErrorString(err)));
    }
    
    // Clear any existing errors
    cudaGetLastError();
    printf("cudaGetLastError returned %d\n", err);
    
    // Create matrix with device memory ownership
    auto matrix = std::make_unique<Matrix>(rows, cols, device_ptr, false);  // false for device memory
    matrix->set_owns_data(true);  // Use setter instead of direct access
    return matrix;
}

void MemoryManager::free(void* ptr) {
    if (ptr) {
        cudaError_t err = cudaFree(ptr);
        if (err != cudaSuccess) {
            // Log error but don't throw in destructor
            std::cerr << "CUDA memory free failed: " << cudaGetErrorString(err) << std::endl;
        }
    }
}

} // namespace cuda 