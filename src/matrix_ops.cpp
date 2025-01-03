#include "../include/components.hpp"
#include "../include/cuda/cuda_utils.cuh"
#include "../include/cuda/matrix_kernels.cuh"
#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>
#include <string>

Matrix::Matrix(size_t rows, size_t cols, const float* data) 
    : rows_(rows), cols_(cols) {
    data_.resize(rows * cols);
    if (data) {
        std::copy(data, data + (rows * cols), data_.begin());
    }
}

Vector Matrix::row(size_t row_idx) const {
  Vector result(cols_);
  for (size_t i = 0; i < cols_; ++i) {
    result[i] = (*this)(row_idx, i);
  }
  return result;
}

Matrix &Matrix::operator+=(const Matrix &other) {
  if (rows_ != other.rows_ || cols_ != other.cols_) {
    throw std::invalid_argument("Matrix dimensions don't match for addition");
  }
  for (size_t i = 0; i < data_.size(); ++i) {
    data_[i] += other.data_[i];
  }
  return *this;
}

Matrix &Matrix::operator-=(const Matrix &other) {
  if (rows_ != other.rows_ || cols_ != other.cols_) {
    throw std::runtime_error("Matrix dimensions don't match for subtraction");
  }

  for (size_t i = 0; i < data_.size(); ++i) {
    data_[i] -= other.data_[i];
  }
  return *this;
}

void Matrix::apply_softmax() {
  for (size_t i = 0; i < rows_; ++i) {
    float max_val = -std::numeric_limits<float>::infinity();
    for (size_t j = 0; j < cols_; ++j) {
      max_val = std::max(max_val, (*this)(i, j));
    }

    float sum = 0.0f;
    for (size_t j = 0; j < cols_; ++j) {
      float &val = (*this)(i, j);
      val = std::exp(val - max_val);
      sum += val;
    }

    for (size_t j = 0; j < cols_; ++j) {
      (*this)(i, j) /= sum;
    }
  }
}

Matrix Matrix::transpose() const {
  Matrix result(cols_, rows_);
  for (size_t i = 0; i < rows_; ++i) {
    for (size_t j = 0; j < cols_; ++j) {
      result(j, i) = (*this)(i, j);
    }
  }
  return result;
}

Matrix operator*(const Matrix &m, float scalar) {
  Matrix result(m.rows(), m.cols());
  
  float *d_a, *d_result;
  const size_t size = m.size() * sizeof(float);
  
  CUDA_CHECK(cudaMalloc(&d_a, size));
  CUDA_CHECK(cudaMalloc(&d_result, size));
  
  CUDA_CHECK(cudaMemcpy(d_a, m.data(), size, cudaMemcpyHostToDevice));
  
  launch_matrix_scalar_mul(d_a, scalar, d_result, m.size());
  
  CUDA_CHECK(cudaMemcpy(result.data(), d_result, size, cudaMemcpyDeviceToHost));
  
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_result));
  
  return result;
}

Matrix operator*(float scalar, const Matrix &m) { return m * scalar; }

Matrix operator+(const Matrix &a, const Matrix &b) {
  if (a.rows() != b.rows() || a.cols() != b.cols()) {
    throw std::runtime_error("Matrix dimensions don't match for addition");
  }
  Matrix result(a.rows(), a.cols());
  
  float *d_a, *d_b, *d_result;
  const size_t size = a.size() * sizeof(float);
  
  CUDA_CHECK(cudaMalloc(&d_a, size));
  CUDA_CHECK(cudaMalloc(&d_b, size));
  CUDA_CHECK(cudaMalloc(&d_result, size));
  
  CUDA_CHECK(cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice));
  
  launch_matrix_add(d_a, d_b, d_result, a.rows(), a.cols());
  
  CUDA_CHECK(cudaMemcpy(result.data(), d_result, size, cudaMemcpyDeviceToHost));
  
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_result));
  
  return result;
}

Matrix operator-(const Matrix &a, const Matrix &b) {
  if (a.rows() != b.rows() || a.cols() != b.cols()) {
    throw std::runtime_error("Matrix dimensions don't match for subtraction");
  }
  Matrix result(a.rows(), a.cols());
  
  float *d_a, *d_b, *d_result;
  const size_t size = a.size() * sizeof(float);
  
  CUDA_CHECK(cudaMalloc(&d_a, size));
  CUDA_CHECK(cudaMalloc(&d_b, size));
  CUDA_CHECK(cudaMalloc(&d_result, size));
  
  CUDA_CHECK(cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice));
  
  launch_matrix_sub(d_a, d_b, d_result, a.rows(), a.cols());
  
  CUDA_CHECK(cudaMemcpy(result.data(), d_result, size, cudaMemcpyDeviceToHost));
  
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_result));
  
  return result;
}

Matrix operator/(const Matrix &m, float scalar) { return m * (1.0f / scalar); }

Matrix matmul(const Matrix &a, const Matrix &b) {
  if (a.cols() != b.rows()) {
    throw std::runtime_error("Matrix dimensions don't match for multiplication");
  }
  
  Matrix result(a.rows(), b.cols());
  
  float *d_a, *d_b, *d_result;
  const size_t size_a = a.size() * sizeof(float);
  const size_t size_b = b.size() * sizeof(float);
  const size_t size_result = result.size() * sizeof(float);
  
  CUDA_CHECK(cudaMalloc(&d_a, size_a));
  CUDA_CHECK(cudaMalloc(&d_b, size_b));
  CUDA_CHECK(cudaMalloc(&d_result, size_result));
  
  CUDA_CHECK(cudaMemcpy(d_a, a.data(), size_a, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, b.data(), size_b, cudaMemcpyHostToDevice));
  
  launch_matrix_mul(d_a, d_b, d_result, a.rows(), b.cols(), a.cols());
  
  CUDA_CHECK(cudaMemcpy(result.data(), d_result, size_result, cudaMemcpyDeviceToHost));
  
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_result));
  
  return result;
}

Matrix &Matrix::operator*=(float scalar) {
  for (float &val : data_) {
    val *= scalar;
  }
  return *this;
}

void Matrix::save(std::ostream &os) const {
  os.write(reinterpret_cast<const char *>(&rows_), sizeof(rows_));
  os.write(reinterpret_cast<const char *>(&cols_), sizeof(cols_));
  os.write(reinterpret_cast<const char *>(data_.data()),
           data_.size() * sizeof(float));
}

Matrix Matrix::load(std::istream &is) {
  size_t rows, cols;
  is.read(reinterpret_cast<char *>(&rows), sizeof(rows));
  is.read(reinterpret_cast<char *>(&cols), sizeof(cols));

  Matrix result(rows, cols);
  is.read(reinterpret_cast<char *>(result.data_.data()),
          result.data_.size() * sizeof(float));
  return result;
}