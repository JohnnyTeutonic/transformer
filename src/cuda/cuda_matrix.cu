#ifdef USE_CUDA

#include "../../include/cuda/cuda_matrix.hpp"
#include "../../include/cuda/cuda_check.cuh"
#include <cstring>
#include <stdexcept>

namespace cuda {

CudaMatrix::CudaMatrix(size_t rows, size_t cols) 
    : rows_(rows), cols_(cols), owns_data_(true) {
    allocate(rows, cols);
}

CudaMatrix::CudaMatrix(const Matrix& host_matrix)
    : rows_(host_matrix.rows()), cols_(host_matrix.cols()), owns_data_(true) {
    allocate(rows_, cols_);
    from_host(host_matrix);
}

CudaMatrix::CudaMatrix(const CudaMatrix& other)
    : rows_(other.rows_), cols_(other.cols_), owns_data_(true) {
    allocate(rows_, cols_);
    if (other.data_) {
        CUDA_CHECK(cudaMemcpy(data_, other.data_, size() * sizeof(float), 
                             cudaMemcpyDeviceToDevice));
    }
}

CudaMatrix::CudaMatrix(CudaMatrix&& other) noexcept
    : rows_(other.rows_), cols_(other.cols_), 
      data_(other.data_), owns_data_(other.owns_data_) {
    other.data_ = nullptr;
    other.owns_data_ = false;
}

CudaMatrix& CudaMatrix::operator=(const CudaMatrix& other) {
    if (this != &other) {
        free();
        rows_ = other.rows_;
        cols_ = other.cols_;
        owns_data_ = true;
        allocate(rows_, cols_);
        if (other.data_) {
            CUDA_CHECK(cudaMemcpy(data_, other.data_, size() * sizeof(float),
                                 cudaMemcpyDeviceToDevice));
        }
    }
    return *this;
}

CudaMatrix& CudaMatrix::operator=(CudaMatrix&& other) noexcept {
    if (this != &other) {
        free();
        rows_ = other.rows_;
        cols_ = other.cols_;
        data_ = other.data_;
        owns_data_ = other.owns_data_;
        other.data_ = nullptr;
        other.owns_data_ = false;
    }
    return *this;
}

CudaMatrix::~CudaMatrix() {
    free();
}

void CudaMatrix::allocate(size_t rows, size_t cols) {
    rows_ = rows;
    cols_ = cols;
    if (size() > 0) {
        CUDA_CHECK(cudaMalloc(&data_, size() * sizeof(float)));
    }
}

void CudaMatrix::free() {
    if (data_ && owns_data_) {
        cudaFree(data_);
    }
    data_ = nullptr;
}

void CudaMatrix::from_host(const Matrix& host_matrix) {
    if (rows_ != host_matrix.rows() || cols_ != host_matrix.cols()) {
        throw std::runtime_error("CudaMatrix dimensions don't match host matrix");
    }
    if (host_matrix.size() > 0) {
        CUDA_CHECK(cudaMemcpy(data_, host_matrix.data(), 
                             size() * sizeof(float), cudaMemcpyHostToDevice));
    }
}

Matrix CudaMatrix::to_matrix() const {
    Matrix result(rows_, cols_);
    if (size() > 0 && data_) {
        CUDA_CHECK(cudaMemcpy(result.data(), data_, 
                             size() * sizeof(float), cudaMemcpyDeviceToHost));
    }
    return result;
}

void CudaMatrix::fill(float value) {
    if (size() > 0) {
        // Simple kernel to fill matrix with value
        // For now, do it on CPU and copy (not optimal but works)
        std::vector<float> temp(size(), value);
        CUDA_CHECK(cudaMemcpy(data_, temp.data(), 
                             size() * sizeof(float), cudaMemcpyHostToDevice));
    }
}

} // namespace cuda

#endif // USE_CUDA

