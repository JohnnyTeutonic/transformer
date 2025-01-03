#include "../include/tensor.hpp"
#include "../include/cuda/cuda_utils.cuh"
#ifdef USE_CUDA
#include "../include/cuda/tensor_kernels.cuh"
#endif
#include <stdexcept>
#include <numeric>
#include <omp.h>

void tensormul(const float* a, const float* b, float* result,
               int d1, int d2, int d3, int d4, int b_d4) {
#ifdef USE_CUDA
    launch_tensor_mul(a, b, result, d1, d2, d3, d4, b_d4);
#else
    // CPU fallback implementation
    for (int i1 = 0; i1 < d1; ++i1) {
        for (int i2 = 0; i2 < d2; ++i2) {
            for (int i3 = 0; i3 < d3; ++i3) {
                for (int i4 = 0; i4 < d4; ++i4) {
                    const int a_idx = i1 * (d2 * d3 * d4) + i2 * (d3 * d4) + i3 * d4 + i4;
                    const int b_idx = i1 * (d2 * d3 * b_d4) + i2 * (d3 * b_d4) + i3 * b_d4 + i4;
                    result[a_idx] = a[a_idx] * b[b_idx];
                }
            }
        }
    }
#endif
}

Tensor::Tensor(unsigned long d1, unsigned long d2, unsigned long d3, unsigned long d4) 
    : dims_{d1, d2, d3, d4} {
    size_t total_size = d1 * d2 * d3 * d4;
    data_.resize(total_size, 0.0f);
}

Tensor::Tensor(const Matrix& mat, const std::vector<unsigned long>& shape) {
    if (shape.size() != 4) {
        throw std::runtime_error("Tensor shape must have 4 dimensions");
    }
    
    size_t total_size = std::accumulate(shape.begin(), shape.end(), 1UL, std::multiplies<unsigned long>());
    if (total_size != mat.size()) {
        throw std::runtime_error("Matrix size does not match tensor shape");
    }
    
    dims_ = shape;
    data_.resize(total_size);
    std::copy(mat.data(), mat.data() + total_size, data_.begin());
}

float& Tensor::at(unsigned long i, unsigned long j, unsigned long k, unsigned long l) {
    size_t index = i * (dims_[1] * dims_[2] * dims_[3]) +
                   j * (dims_[2] * dims_[3]) +
                   k * dims_[3] + l;
    if (index >= data_.size()) {
        throw std::out_of_range("Tensor index out of bounds");
    }
    return data_[index];
}

float Tensor::at(unsigned long i, unsigned long j, unsigned long k, unsigned long l) const {
    size_t index = i * (dims_[1] * dims_[2] * dims_[3]) +
                   j * (dims_[2] * dims_[3]) +
                   k * dims_[3] + l;
    if (index >= data_.size()) {
        throw std::out_of_range("Tensor index out of bounds");
    }
    return data_[index];
}

Tensor Tensor::transpose(const std::vector<unsigned long>& perm) const {
    if (perm.size() != 4) {
        throw std::runtime_error("Transpose permutation must have 4 dimensions");
    }
    
    std::vector<unsigned long> new_dims = {
        dims_[perm[0]], dims_[perm[1]], dims_[perm[2]], dims_[perm[3]]
    };
    
    Tensor result(new_dims[0], new_dims[1], new_dims[2], new_dims[3]);
    
    for (size_t i = 0; i < dims_[0]; ++i) {
        for (size_t j = 0; j < dims_[1]; ++j) {
            for (size_t k = 0; k < dims_[2]; ++k) {
                for (size_t l = 0; l < dims_[3]; ++l) {
                    std::vector<size_t> old_idx = {i, j, k, l};
                    std::vector<size_t> new_idx = {
                        old_idx[perm[0]], old_idx[perm[1]], 
                        old_idx[perm[2]], old_idx[perm[3]]
                    };
                    result.at(new_idx[0], new_idx[1], new_idx[2], new_idx[3]) = 
                        at(i, j, k, l);
                }
            }
        }
    }
    
    return result;
}

Tensor Tensor::permute(const std::vector<unsigned long>& perm) const {
    if (perm.size() != 4) {
        throw std::runtime_error("Permutation must be of size 4");
    }
    
    std::vector<unsigned long> new_dims = {
        dims_[perm[0]], dims_[perm[1]], dims_[perm[2]], dims_[perm[3]]
    };
    
    Tensor result(new_dims[0], new_dims[1], new_dims[2], new_dims[3]);
    
    #pragma omp parallel for collapse(4)
    for (size_t i = 0; i < dims_[0]; ++i) {
        for (size_t j = 0; j < dims_[1]; ++j) {
            for (size_t k = 0; k < dims_[2]; ++k) {
                for (size_t l = 0; l < dims_[3]; ++l) {
                    std::vector<size_t> old_idx = {i, j, k, l};
                    std::vector<size_t> new_idx = {
                        old_idx[perm[0]], old_idx[perm[1]], 
                        old_idx[perm[2]], old_idx[perm[3]]
                    };
                    result.at(new_idx[0], new_idx[1], new_idx[2], new_idx[3]) = 
                        at(i, j, k, l);
                }
            }
        }
    }
    
    return result;
}

Tensor Tensor::tensormul(const Tensor& other) const {
    if (dims_[3] != other.dims_[2]) {
        throw std::runtime_error("Incompatible dimensions for tensor multiplication");
    }
    
    Tensor result(dims_[0], dims_[1], dims_[2], other.dims_[3]);
    
    float *d_a, *d_b, *d_result;
    const size_t size_a = data_.size() * sizeof(float);
    const size_t size_b = other.data_.size() * sizeof(float);
    const size_t size_result = result.data_.size() * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_a, size_a));
    CUDA_CHECK(cudaMalloc(&d_b, size_b));
    CUDA_CHECK(cudaMalloc(&d_result, size_result));
    
    CUDA_CHECK(cudaMemcpy(d_a, data_.data(), size_a, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, other.data_.data(), size_b, cudaMemcpyHostToDevice));
    
    launch_tensor_mul(d_a, d_b, d_result,
                     dims_[0], dims_[1], dims_[2], dims_[3],
                     other.dims_[3]);
    
    CUDA_CHECK(cudaMemcpy(result.data_.data(), d_result, size_result, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_result));
    
    return result;
}

Matrix Tensor::to_matrix() const {
    size_t rows = this->rows();
    size_t cols = this->cols();
    
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) = data_[i * cols + j];
        }
    }
    
    return result;
}

Tensor Tensor::safe_tensormul(const Tensor& a, const Tensor& b) {
    try {
        return a.tensormul(b);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Safe tensor multiplication failed: ") + e.what());
    }
}
