#include "../include/matrix.hpp"
#include <random>
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <cmath>

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

void Matrix::resize(size_t new_rows, size_t new_cols) {
    if (new_rows == rows_ && new_cols == cols_) {
        return;
    }
    if (new_rows > SIZE_MAX / new_cols) {
        throw std::runtime_error("Matrix dimensions would cause overflow");
    }
    data_.resize(new_rows * new_cols);
    rows_ = new_rows;
    cols_ = new_cols;
}

float& Matrix::operator()(size_t row, size_t col) {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    return data_[row * cols_ + col];
}

const float& Matrix::operator()(size_t row, size_t col) const {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    return data_[row * cols_ + col];
}

Matrix& Matrix::operator+=(const Matrix& other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }
    for (size_t i = 0; i < data_.size(); ++i) {
        data_[i] += other.data_[i];
    }
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }
    for (size_t i = 0; i < data_.size(); ++i) {
        data_[i] -= other.data_[i];
    }
    return *this;
}

Matrix& Matrix::operator*=(float scalar) {
    for (float& val : data_) {
        val *= scalar;
    }
    return *this;
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

void Matrix::fill(float value) {
    std::fill(data_.begin(), data_.end(), value);
}

void Matrix::randomize(float min_val, float max_val) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    for (float& val : data_) {
        val = dis(gen);
    }
}

Vector Matrix::row_sum() const {
    Vector result(cols_, 0.0f);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result[j] += (*this)(i, j);
        }
    }
    return result;
}

void Matrix::save(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(&rows_), sizeof(rows_));
    os.write(reinterpret_cast<const char*>(&cols_), sizeof(cols_));
    os.write(reinterpret_cast<const char*>(data_.data()), data_.size() * sizeof(float));
}

Matrix Matrix::load(std::istream& is) {
    size_t rows, cols;
    is.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    is.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    Matrix result(rows, cols);
    is.read(reinterpret_cast<char*>(result.data_.data()), result.data_.size() * sizeof(float));
    return result;
}

// Non-member operators
Matrix operator+(const Matrix& a, const Matrix& b) {
    Matrix result = a;
    result += b;
    return result;
}

Matrix operator-(const Matrix& a, const Matrix& b) {
    Matrix result = a;
    result -= b;
    return result;
}

Matrix operator*(const Matrix& m, float scalar) {
    Matrix result = m;
    result *= scalar;
    return result;
}

Matrix operator*(float scalar, const Matrix& m) {
    return m * scalar;
}

Matrix matmul(const Matrix& A, const Matrix& B) {
    if (A.cols() != B.rows()) {
        throw std::invalid_argument("Matrix dimensions must match for multiplication");
    }
    Matrix C(A.rows(), B.cols(), 0.0f);
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < B.cols(); ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < A.cols(); ++k) {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }
    return C;
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