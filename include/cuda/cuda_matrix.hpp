#pragma once

#ifdef USE_CUDA

// CRITICAL: Include this FIRST to fix CUDA 12.6 + GCC math conflicts
#include "cuda_math_fix.hpp"

#include "../matrix.hpp"
#include <memory>

namespace cuda {

/**
 * @brief GPU-accelerated matrix wrapper for CUDA operations
 * 
 * Provides efficient GPU memory management and data transfer
 * for transformer operations.
 */
class CudaMatrix {
public:
    // Constructors
    CudaMatrix() : rows_(0), cols_(0), data_(nullptr), owns_data_(true) {}
    
    CudaMatrix(size_t rows, size_t cols);
    
    explicit CudaMatrix(const Matrix& host_matrix);
    
    // Copy constructor
    CudaMatrix(const CudaMatrix& other);
    
    // Move constructor
    CudaMatrix(CudaMatrix&& other) noexcept;
    
    // Assignment operators
    CudaMatrix& operator=(const CudaMatrix& other);
    CudaMatrix& operator=(CudaMatrix&& other) noexcept;
    
    // Destructor
    ~CudaMatrix();
    
    // Accessors
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t size() const { return rows_ * cols_; }
    float* data() { return data_; }
    const float* data() const { return data_; }
    
    // Data transfer
    void from_host(const Matrix& host_matrix);
    Matrix to_matrix() const;
    
    // Memory management
    void allocate(size_t rows, size_t cols);
    void free();
    
    // Utilities
    void fill(float value);
    void zero() { fill(0.0f); }
    
private:
    size_t rows_;
    size_t cols_;
    float* data_;
    bool owns_data_;
};

} // namespace cuda

#endif // USE_CUDA

