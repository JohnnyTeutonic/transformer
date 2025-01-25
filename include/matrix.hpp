#pragma once

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include "cuda/memory_manager.cuh"
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>
#include "vector.hpp"  // Include Vector definition from vector.hpp
#define M_PI 3.14159265358979323846
// Forward declarations
class Matrix;
class Vector;

// Forward declare cuda namespace
namespace cuda {
    class MatrixOps;  // Forward declare any classes that need friend access
    class MemoryManager;  // Forward declare MemoryManager
}

/**
 * @brief A 2D matrix class optimized for neural network operations.
 * 
 * The Matrix class provides a fundamental building block for neural network computations,
 * supporting both CPU and GPU operations. Features include:
 * - Basic matrix operations (addition, multiplication, transposition)
 * - Neural network specific operations (ReLU, GELU, Softmax)
 * - CUDA acceleration support
 * - Memory management for both CPU and GPU
 * - Efficient data access patterns
 */
class Matrix {
  private:
    std::vector<float> data_;        ///< Matrix data storage on CPU
    size_t rows_;                    ///< Number of rows
    size_t cols_;                    ///< Number of columns
    std::tuple<size_t, size_t> shape_; ///< Matrix shape as (rows, cols)
    bool owns_data_ = true;          ///< Whether this matrix owns its data or views external data

#ifdef USE_CUDA
    float* gpu_data_ = nullptr;      ///< Matrix data storage on GPU
    bool is_on_gpu_ = false;         ///< Whether the data is currently on GPU
    bool is_cuda_ = false;
#endif

  public:
    // Make CUDA operations a friend of Matrix
    friend class cuda::MatrixOps;
    
    /**
     * @brief Check if matrix is using CUDA memory
     * @return true if matrix is using CUDA memory
     */
    bool is_cuda() const { return is_cuda_; }

    /**
     * @brief Default constructor.
     */
    Matrix();

    /**
     * @brief Constructs a matrix with specified dimensions.
     * @param rows Number of rows
     * @param cols Number of columns
     * @param init_val Initial value for all elements (default: 0.0f)
     */
    Matrix(size_t rows, size_t cols, float init_val = 0.0f);

    /**
     * @brief Constructs a matrix using external data.
     * @param rows Number of rows
     * @param cols Number of columns
     * @param external_data Pointer to external data
     */
    Matrix(size_t rows, size_t cols, float* external_data);

    /**
     * @brief Constructs a matrix using external data with ownership control.
     * @param rows Number of rows
     * @param cols Number of columns
     * @param external_data Pointer to external data (can be host or device memory)
     * @param is_owner Whether this matrix should own the data (for host memory)
     *                 or whether it's CUDA memory (for device memory)
     */
    Matrix(size_t rows, size_t cols, float* external_data, bool is_owner)
        : rows_(rows), cols_(cols), shape_(std::make_tuple(rows, cols)) {
        if (is_owner) {
            // Host memory case
            owns_data_ = true;
            is_cuda_ = false;
            data_.assign(external_data, external_data + (rows * cols));
        } else {
            // Device memory case
            owns_data_ = true;
            is_cuda_ = true;
#ifdef USE_CUDA
            gpu_data_ = external_data;
            is_on_gpu_ = true;
#else
            throw std::runtime_error("CUDA support not enabled");
#endif
        }
    }

    /**
     * @brief Gets the number of rows.
     * @return Number of rows
     */
    size_t rows() const {
        return rows_;
    }

    /**
     * @brief Gets the number of columns.
     * @return Number of columns
     */
    size_t cols() const {
        return cols_;
    }

    /**
     * @brief Gets the total number of elements.
     * @return Number of elements
     */
    size_t size() const {
        return data_.size();
    }

    /**
     * @brief Gets the total size in bytes.
     * @return Size in bytes
     */
    size_t bytes() const {
        return size() * sizeof(float);
    }

    /**
     * @brief Gets the matrix shape.
     * @return Tuple of (rows, cols)
     */
    std::tuple<size_t, size_t> shape() const {
        return shape_;
    }

    /**
     * @brief Checks if the matrix is empty.
     * @return True if empty
     */
    bool empty() const {
        return data_.empty();
    }

    /**
     * @brief Gets the minimum value in the matrix.
     * @return Minimum value
     */
    float min() const {
        return *std::min_element(data_.begin(), data_.end());
    }

    /**
     * @brief Gets the maximum value in the matrix.
     * @return Maximum value
     */
    float max() const {
        return *std::max_element(data_.begin(), data_.end());
    }

    /**
     * @brief Gets a pointer to the underlying data.
     * @return Const pointer to data (CPU or GPU based on current location)
     */
#ifdef USE_CUDA
    const float* get_data() const {
        return is_on_gpu_ ? gpu_data_ : data_.data();
    }
    float* get_data() {
        return is_on_gpu_ ? gpu_data_ : data_.data();
    }
#else
    const float* get_data() const {
        return data_.data();
    }
    float* get_data() {
        return data_.data();
    }
#endif

    /**
     * @brief Resizes the matrix.
     * @param new_rows New number of rows
     * @param new_cols New number of columns
     */
    void resize(size_t new_rows, size_t new_cols);

    /**
     * @brief Element access operator.
     * @param row Row index
     * @param col Column index
     * @return Reference to the element
     */
    float& operator()(size_t row, size_t col);
    const float& operator()(size_t row, size_t col) const;

    /**
     * @brief Safe element access with bounds checking.
     * @param row Row index
     * @param col Column index
     * @return Reference to the element
     * @throws std::out_of_range if indices are invalid
     */
    float& at(size_t row, size_t col);
    const float& at(size_t row, size_t col) const;

    /**
     * @brief Gets a row as a vector.
     * @param row Row index
     * @return Vector containing the row
     */
    Vector row(size_t row) const;

    /**
     * @brief Sets a row from a vector.
     * @param row Row index
     * @param vec Vector containing new values
     */
    void set_row(size_t row, const Vector& vec);

    /**
     * @brief Computes the matrix transpose.
     * @return Transposed matrix
     */
    Matrix transpose() const;

    /**
     * @brief Applies ReLU activation function element-wise.
     */
    void apply_relu();

    /**
     * @brief Applies GELU activation function element-wise.
     */
    void apply_gelu();

    /**
     * @brief Applies GELU derivative for backpropagation.
     * @param x Input matrix
     */
    void apply_gelu_derivative(const Matrix& x);

    /**
     * @brief Applies softmax function row-wise.
     */
    void apply_softmax();

    /**
     * @brief Adds a bias vector to each row.
     * @param bias Bias vector to add
     */
    void add_bias(const Vector& bias);

    /**
     * @brief Matrix addition assignment operator.
     * @param other Matrix to add
     * @return Reference to this matrix
     */
    Matrix& operator+=(const Matrix& other);

    Matrix& operator-=(const Matrix& other);
    Matrix& operator*=(float scalar);
    Matrix& operator/=(float scalar);
    Matrix& operator*=(const Matrix& other);
    void save(std::ostream& os) const;
    static Matrix load(std::istream& is);
    void randomize(float min_val, float max_val);
    Vector row_sum() const;
    void fill(float value);
    void fill(const Matrix& m, float value);

    // Add element-wise multiplication (Hadamard product)
    Matrix hadamard(const Matrix& other) const {
        if (rows() != other.rows() || cols() != other.cols()) {
            throw std::runtime_error("Matrix dimensions must match for hadamard product");
        }

        Matrix result(rows(), cols());
        const float* this_data = get_data();
        const float* other_data = other.get_data();
        float* result_data = result.get_data();
        
        for (size_t i = 0; i < size(); ++i) {
            result_data[i] = this_data[i] * other_data[i];
        }
        return result;
    }

    // Returns a view into a block of the matrix
    Matrix block(size_t start_row, size_t start_col, size_t num_rows, size_t num_cols) const {
        // Validate input parameters
        if (start_row >= rows_ || start_col >= cols_) {
            throw std::out_of_range("Block start position out of matrix bounds");
        }
        if (start_row + num_rows > rows_ || start_col + num_cols > cols_) {
            throw std::out_of_range("Block dimensions exceed matrix bounds");
        }
        if (num_rows == 0 || num_cols == 0) {
            throw std::invalid_argument("Block dimensions cannot be zero");
        }

        // Create result matrix with proper size
        Matrix result(num_rows, num_cols);
        
        // Copy data with bounds checking
        try {
            for (size_t i = 0; i < num_rows; ++i) {
                for (size_t j = 0; j < num_cols; ++j) {
                    result(i, j) = (*this)(start_row + i, start_col + j);
                }
            }
        } catch (const std::exception& e) {
            throw std::runtime_error("Error creating matrix block: " + std::string(e.what()));
        }
        
        return result;
    }

    // Only declare copy constructor and assignment operator
    Matrix(const Matrix& other) {
        data_ = other.data_;
        rows_ = other.rows_;
        cols_ = other.cols_;
        shape_ = other.shape_;
        owns_data_ = other.owns_data_;
#ifdef USE_CUDA
        if (other.is_on_gpu_) {
            cudaMalloc(&gpu_data_, data_.size() * sizeof(float));
            cudaMemcpy(gpu_data_, other.gpu_data_, data_.size() * sizeof(float),
                       cudaMemcpyDeviceToDevice);
            is_on_gpu_ = true;
        }
#endif
    }

    Matrix& operator=(const Matrix& other); // Declaration only

    // Move constructor
    Matrix(Matrix&& other) noexcept 
        : data_(std::move(other.data_)),
          rows_(other.rows_),
          cols_(other.cols_),
          shape_(other.shape_),
          owns_data_(other.owns_data_),
          is_cuda_(other.is_cuda_) {
#ifdef USE_CUDA
        gpu_data_ = other.gpu_data_;
        is_on_gpu_ = other.is_on_gpu_;
        // Important: null out the other's GPU pointer
        other.gpu_data_ = nullptr;
        other.is_on_gpu_ = false;
#endif
        other.rows_ = 0;
        other.cols_ = 0;
        other.owns_data_ = false;
        other.is_cuda_ = false;
    }

    // Move assignment
    Matrix& operator=(Matrix&& other) noexcept {
        if (this != &other) {
            // Free existing resources
            if (owns_data_) {
#ifdef USE_CUDA
                if (is_cuda_ && gpu_data_) {
                    cuda::MemoryManager::get_instance().free(gpu_data_);
                }
#endif
            }

            // Move resources
            data_ = std::move(other.data_);
            rows_ = other.rows_;
            cols_ = other.cols_;
            shape_ = other.shape_;
            owns_data_ = other.owns_data_;
            is_cuda_ = other.is_cuda_;
#ifdef USE_CUDA
            gpu_data_ = other.gpu_data_;
            is_on_gpu_ = other.is_on_gpu_;
            // Important: null out the other's GPU pointer
            other.gpu_data_ = nullptr;
            other.is_on_gpu_ = false;
#endif
            // Reset other
            other.rows_ = 0;
            other.cols_ = 0;
            other.owns_data_ = false;
            other.is_cuda_ = false;
        }
        return *this;
    }

    // Destructor
    ~Matrix();

    static Matrix from_vector(const std::vector<float>& vec) {
        Matrix mat(1, vec.size());
        std::copy(vec.begin(), vec.end(), mat.get_data());
        return mat;
    }

    // Forward method for compatibility with neural network layers
    Matrix& forward(const Matrix& input) {
        *this = input;  // Copy input to this matrix
        return *this;
    }

    /**
     * @brief Initialize matrix with random values using Xavier/Glorot initialization
     * @param scale Scaling factor for initialization
     * @throws std::runtime_error if matrix doesn't own its data
     */
    void initialize_random(float scale);

    /**
     * @brief Initialize matrix with a constant value
     * @param value Value to initialize all elements with
     * @throws std::runtime_error if matrix doesn't own its data
     */
    void initialize_constant(float value);

    /**
     * @brief Transfers the matrix data to GPU memory.
     * @return A new Matrix object that points to GPU memory
     */
    Matrix to_gpu() const {
        #ifdef USE_CUDA
        float* gpu_ptr;
        size_t total_size = rows_ * cols_ * sizeof(float);
        cudaMalloc(&gpu_ptr, total_size);
        cudaMemcpy(gpu_ptr, data_.data(), total_size, cudaMemcpyHostToDevice);
        return Matrix(rows_, cols_, gpu_ptr, false);  // false indicates GPU ownership
        #else
        throw std::runtime_error("CUDA support not enabled");
        #endif
    }

    /**
     * @brief Transfers the matrix data from GPU to CPU memory.
     * @return A new Matrix object in CPU memory
     */
    Matrix to_cpu() const {
        #ifdef USE_CUDA
        if (!is_cuda_) {
            return *this;  // Already on CPU
        }
        Matrix cpu_matrix(rows_, cols_);
        size_t total_size = rows_ * cols_ * sizeof(float);
        cudaMemcpy(cpu_matrix.get_data(), gpu_data_, total_size, cudaMemcpyDeviceToHost);
        return cpu_matrix;
        #else
        return *this;  // Always on CPU when CUDA is not enabled
        #endif
    }

    // Add friend declaration for gelu function
    friend void gelu(Matrix& x);
    friend void gelu_derivative(Matrix& x);

    // Add accessor methods
    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }
    
    // Get raw data size
    size_t data_size() const { return data_.size(); }

#ifdef USE_CUDA
    // CUDA-specific methods
    float* gpu_data() { return gpu_data_; }
    const float* gpu_data() const { return gpu_data_; }
#endif
};

// Make to_vector inline to allow multiple definitions
inline std::vector<int> to_vector(const Matrix& m) {
    const float* data = m.get_data();
    return std::vector<int>(data, data + m.size());
}

// Non-member operators
Matrix operator+(const Matrix& a, const Matrix& b);
Matrix operator-(const Matrix& a, const Matrix& b);
Matrix operator*(const Matrix& m, float scalar);
Matrix operator*(float scalar, const Matrix& m);
Matrix operator/(const Matrix& m, float scalar);
Matrix operator*(const Matrix& a, const Matrix& b);

// Matrix multiplication function
Matrix matmul(const Matrix& a, const Matrix& b);

inline std::ostream& operator<<(std::ostream& os, const std::tuple<size_t, size_t>& shape) {
    os << std::get<0>(shape) << "x" << std::get<1>(shape);
    return os;
}