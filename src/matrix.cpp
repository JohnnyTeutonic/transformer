#include "../include/matrix.hpp"
#include <random>
#include <algorithm>
#include <cstddef>
#include <stdexcept>

Matrix::Matrix() : rows_(0), cols_(0), owns_data_(true) {
    data_.clear();
}

Matrix::Matrix(unsigned long rows, unsigned long cols, float init_val) 
    : rows_(rows), cols_(cols), owns_data_(true) {
    data_.resize(rows * cols, init_val);
}

Matrix::~Matrix() {
    if (owns_data_) {
        data_.clear();
    }
}

float& Matrix::operator()(unsigned long row, unsigned long col) {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix indices out of bounds");
    }
    return data_[row * cols_ + col];
}

const float& Matrix::operator()(unsigned long row, unsigned long col) const {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix indices out of bounds");
    }
    return data_[row * cols_ + col];
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        rows_ = other.rows_;
        cols_ = other.cols_;
        owns_data_ = true;
        data_ = other.data_;
    }
    return *this;
}

void Matrix::initialize_random(float scale) {
    if (!owns_data_) {
        throw std::runtime_error("Cannot initialize external data buffer");
    }
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-scale, scale);
    
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            data_[i * cols_ + j] = dis(gen);
        }
    }
}

void Matrix::initialize_constant(float value) {
    if (!owns_data_) {
        throw std::runtime_error("Cannot initialize external data buffer");
    }
    
    std::fill(data_.begin(), data_.end(), value);
} 