#include "../include/feed_forward.hpp"
#ifdef USE_CUDA
#include "../include/cuda/cuda_check.cuh"
#include "../include/cuda/cuda_launch.cuh"
#include "../include/cuda/feed_forward_kernels.cuh"
#include "../include/cuda/backward_ops.cuh"
#include "../include/cuda/matrix_ops.cuh"
#include "../include/cuda/memory_manager.cuh"
#endif
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// CPU implementation of matrix multiplication
void matmul(const Matrix& a, const Matrix& b, Matrix& c) {
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < a.rows(); ++i) {
        for (size_t j = 0; j < b.cols(); ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < a.cols(); ++k) {
                sum += a(i, k) * b(k, j);
            }
            c(i, j) = sum;
        }
    }
}

FeedForward::FeedForward(size_t hidden_size, size_t intermediate_size, float dropout)
    : w1(hidden_size, intermediate_size), w2(intermediate_size, hidden_size), b1(intermediate_size),
      b2(hidden_size), dropout_prob(dropout),
      // Don't initialize intermediate_cache with fixed size
      dW1_(hidden_size, intermediate_size), dW2_(intermediate_size, hidden_size),
      db1_(intermediate_size), db2_(hidden_size) {

    // Validate dimensions
    if (hidden_size == 0) {
        throw std::runtime_error("Hidden size cannot be zero");
    }
    if (intermediate_size == 0) {
        throw std::runtime_error("Intermediate size cannot be zero");
    }

    std::cout << "Initializing FeedForward with dimensions:" << std::endl;
    std::cout << "Hidden size: " << hidden_size << std::endl;
    std::cout << "Intermediate size: " << intermediate_size << std::endl;
    std::cout << "Dropout probability: " << dropout_prob << std::endl;

    // Initialize weights with Xavier/Glorot initialization
    std::random_device rd;
    std::mt19937 gen(rd());

    float w1_limit = std::sqrt(6.0f / (hidden_size + intermediate_size));
    float w2_limit = std::sqrt(6.0f / (intermediate_size + hidden_size));

    std::uniform_real_distribution<float> w1_dis(-w1_limit, w1_limit);
    std::uniform_real_distribution<float> w2_dis(-w2_limit, w2_limit);

    // Initialize weights
    for (size_t i = 0; i < w1.rows(); ++i) {
        for (size_t j = 0; j < w1.cols(); ++j) {
            w1(i, j) = w1_dis(gen);
        }
    }

    for (size_t i = 0; i < w2.rows(); ++i) {
        for (size_t j = 0; j < w2.cols(); ++j) {
            w2(i, j) = w2_dis(gen);
        }
    }

    // Initialize biases to small non-zero values to prevent dead neurons
    for (size_t i = 0; i < b1.size(); ++i)
        b1[i] = 0.01f;  // Small positive bias
    for (size_t i = 0; i < b2.size(); ++i)
        b2[i] = 0.01f;  // Small positive bias

    // Initialize gradients to zero
    for (size_t i = 0; i < dW1_.rows(); ++i) {
        for (size_t j = 0; j < dW1_.cols(); ++j) {
            dW1_(i, j) = 0.0f;
        }
    }

    for (size_t i = 0; i < dW2_.rows(); ++i) {
        for (size_t j = 0; j < dW2_.cols(); ++j) {
            dW2_(i, j) = 0.0f;
        }
    }

    for (size_t i = 0; i < db1_.size(); ++i)
        db1_[i] = 0.0f;
    for (size_t i = 0; i < db2_.size(); ++i)
        db2_[i] = 0.0f;

    // Verify initialization
    std::cout << "Verifying matrix dimensions after initialization:" << std::endl;
    std::cout << "W1: " << w1.rows() << "x" << w1.cols() << std::endl;
    std::cout << "W2: " << w2.rows() << "x" << w2.cols() << std::endl;
    std::cout << "b1: " << b1.size() << std::endl;
    std::cout << "b2: " << b2.size() << std::endl;
}

Matrix FeedForward::forward(const Matrix& x, bool training) {
    try {
        // Check input dimensions
        if (x.rows() == 0) {
            throw std::runtime_error("Input matrix has zero rows. Expected positive number of rows, got 0");
        }
        if (x.cols() == 0) {
            throw std::runtime_error("Input matrix has zero columns. Expected " + 
                                   std::to_string(w1.rows()) + " columns, got 0");
        }
        if (x.cols() != w1.rows()) {
            throw std::runtime_error("Input dimension mismatch. Expected " + 
                                   std::to_string(w1.rows()) + " columns, got " + 
                                   std::to_string(x.cols()));
        }

        // Print detailed dimensions for debugging
        std::cout << "\nFeed Forward Dimensions:" << std::endl;
        std::cout << "Input: " << x.rows() << "x" << x.cols() << std::endl;
        std::cout << "W1: " << w1.rows() << "x" << w1.cols() << std::endl;
        std::cout << "W2: " << w2.rows() << "x" << w2.cols() << std::endl;
        std::cout << "b1: " << b1.size() << std::endl;
        std::cout << "b2: " << b2.size() << std::endl;

        // Verify weight matrix dimensions
        if (w1.rows() == 0 || w1.cols() == 0) {
            throw std::runtime_error("W1 has invalid dimensions: " + 
                                   std::to_string(w1.rows()) + "x" + 
                                   std::to_string(w1.cols()));
        }
        if (w2.rows() == 0 || w2.cols() == 0) {
            throw std::runtime_error("W2 has invalid dimensions: " + 
                                   std::to_string(w2.rows()) + "x" + 
                                   std::to_string(w2.cols()));
        }
        if (w2.rows() != w1.cols()) {
            throw std::runtime_error("Weight matrices dimension mismatch: W1 cols (" + 
                                   std::to_string(w1.cols()) + ") != W2 rows (" + 
                                   std::to_string(w2.rows()) + ")");
        }

        // Debug input
        std::cout << "\nFeed Forward Input Stats:" << std::endl;
        float min_x = std::numeric_limits<float>::max();
        float max_x = -std::numeric_limits<float>::max();
        float sum_x = 0.0f;
        size_t nonzero = 0;
        for (size_t i = 0; i < x.rows(); ++i) {
            for (size_t j = 0; j < x.cols(); ++j) {
                float val = x(i, j);
                if (!std::isfinite(val)) {
                    throw std::runtime_error("Non-finite input value detected at (" + 
                                           std::to_string(i) + "," + std::to_string(j) + 
                                           "): " + std::to_string(val));
                }
                min_x = std::min(min_x, val);
                max_x = std::max(max_x, val);
                sum_x += val;
                if (val != 0.0f) nonzero++;
            }
        }
        std::cout << "Input - Min: " << min_x << " Max: " << max_x 
                  << " Mean: " << (sum_x / (x.rows() * x.cols()))
                  << " Nonzero: " << nonzero << "/" << (x.rows() * x.cols()) << std::endl;

        // Cache input for backward pass if training
        if (training) {
            input_cache_ = x;
        }

        // First linear transformation
        Matrix intermediate(x.rows(), w1.cols());
        
        // Verify intermediate dimensions
        if (intermediate.rows() == 0 || intermediate.cols() == 0) {
            throw std::runtime_error("Intermediate matrix has zero dimensions: " + 
                                   std::to_string(intermediate.rows()) + "x" + 
                                   std::to_string(intermediate.cols()));
        }
        
        if (x.is_cuda() && w1.is_cuda()) {
            #ifdef USE_CUDA
            if (!cuda::customMatrixMultiply(x, w1, intermediate)) {
                throw std::runtime_error("CUDA matrix multiplication failed for first layer");
            }
            #endif
        } else {
            matmul(x, w1, intermediate);
        }
        
        // Debug after first multiplication
        std::cout << "\nAfter First Matrix Multiplication:" << std::endl;
        float min_inter = std::numeric_limits<float>::max();
        float max_inter = -std::numeric_limits<float>::max();
        float sum_inter = 0.0f;
        nonzero = 0;
        for (size_t i = 0; i < intermediate.rows(); ++i) {
            for (size_t j = 0; j < intermediate.cols(); ++j) {
                float val = intermediate(i, j);
                if (!std::isfinite(val)) {
                    throw std::runtime_error("Non-finite value after first matrix multiply at (" + 
                                           std::to_string(i) + "," + std::to_string(j) + 
                                           "): " + std::to_string(val));
                }
                min_inter = std::min(min_inter, val);
                max_inter = std::max(max_inter, val);
                sum_inter += val;
                if (val != 0.0f) nonzero++;
            }
        }
        std::cout << "Intermediate - Min: " << min_inter << " Max: " << max_inter 
                  << " Mean: " << (sum_inter / (intermediate.rows() * intermediate.cols()))
                  << " Nonzero: " << nonzero << "/" << (intermediate.rows() * intermediate.cols()) << std::endl;
        
        // Add bias with bounds checking
        for (size_t i = 0; i < intermediate.rows(); ++i) {
            for (size_t j = 0; j < intermediate.cols(); ++j) {
                float val = intermediate(i, j) + b1[j];
                if (!std::isfinite(val)) {
                    throw std::runtime_error("Non-finite value after bias at (" + 
                                           std::to_string(i) + "," + std::to_string(j) + 
                                           "): " + std::to_string(val));
                }
                intermediate(i, j) = val;
            }
        }

        // Apply GELU activation with bounds checking
        const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
        for (size_t i = 0; i < intermediate.rows(); ++i) {
            for (size_t j = 0; j < intermediate.cols(); ++j) {
                float x_val = intermediate(i, j);
                // Clamp input to avoid numerical instability, but use wider range
                x_val = std::max(-20.0f, std::min(20.0f, x_val));
                float gelu = x_val * 0.5f * (1.0f + std::erf(x_val / std::sqrt(2.0f)));
                if (!std::isfinite(gelu)) {
                    throw std::runtime_error("Non-finite value after GELU at (" + 
                                           std::to_string(i) + "," + std::to_string(j) + 
                                           "): " + std::to_string(gelu));
                }
                intermediate(i, j) = gelu;
            }
        }

        // Cache intermediate values for backward pass if training
        if (training) {
            intermediate_cache = intermediate;
        }

        // Second linear transformation
        Matrix output(intermediate.rows(), w2.cols());
        
        // Verify output dimensions
        if (output.rows() == 0 || output.cols() == 0) {
            throw std::runtime_error("Output matrix has zero dimensions: " + 
                                   std::to_string(output.rows()) + "x" + 
                                   std::to_string(output.cols()));
        }
        
        if (intermediate.is_cuda() && w2.is_cuda()) {
            #ifdef USE_CUDA
            if (!cuda::customMatrixMultiply(intermediate, w2, output)) {
                throw std::runtime_error("CUDA matrix multiplication failed for second layer");
            }
            #endif
        } else {
            matmul(intermediate, w2, output);
        }

        // Debug final output
        std::cout << "\nFinal Output Stats:" << std::endl;
        float min_out = std::numeric_limits<float>::max();
        float max_out = -std::numeric_limits<float>::max();
        float sum_out = 0.0f;
        nonzero = 0;
        for (size_t i = 0; i < output.rows(); ++i) {
            for (size_t j = 0; j < output.cols(); ++j) {
                float val = output(i, j);
                if (!std::isfinite(val)) {
                    throw std::runtime_error("Non-finite value in output at (" + 
                                           std::to_string(i) + "," + std::to_string(j) + 
                                           "): " + std::to_string(val));
                }
                min_out = std::min(min_out, val);
                max_out = std::max(max_out, val);
                sum_out += val;
                if (val != 0.0f) nonzero++;
            }
        }
        std::cout << "Output - Min: " << min_out << " Max: " << max_out 
                  << " Mean: " << (sum_out / (output.rows() * output.cols()))
                  << " Nonzero: " << nonzero << "/" << (output.rows() * output.cols()) << std::endl;

        // Add bias with bounds checking
        for (size_t i = 0; i < output.rows(); ++i) {
            for (size_t j = 0; j < output.cols(); ++j) {
                float val = output(i, j) + b2[j];
                if (!std::isfinite(val)) {
                    throw std::runtime_error("Non-finite value after final bias at (" + 
                                           std::to_string(i) + "," + std::to_string(j) + 
                                           "): " + std::to_string(val));
                }
                output(i, j) = val;
            }
        }

        return output;
    } catch (const std::exception& e) {
        throw std::runtime_error("Error in feed forward pass: " + std::string(e.what()));
    }
}

void FeedForward::save(std::ostream& os) const {
    size_t hidden_size = w2.cols();
    size_t intermediate_size = w1.cols();

    os.write(reinterpret_cast<const char*>(&hidden_size), sizeof(hidden_size));
    os.write(reinterpret_cast<const char*>(&intermediate_size), sizeof(intermediate_size));
    os.write(reinterpret_cast<const char*>(&dropout_prob), sizeof(dropout_prob));

    os.write(reinterpret_cast<const char*>(w1.get_data()), w1.rows() * w1.cols() * sizeof(float));
    os.write(reinterpret_cast<const char*>(w2.get_data()), w2.rows() * w2.cols() * sizeof(float));
    os.write(reinterpret_cast<const char*>(b1.data()), b1.size() * sizeof(float));
    os.write(reinterpret_cast<const char*>(b2.data()), b2.size() * sizeof(float));
}

std::unique_ptr<FeedForward> FeedForward::load(std::istream& is) {
    size_t hidden_size, intermediate_size;
    float dropout_prob;

    is.read(reinterpret_cast<char*>(&hidden_size), sizeof(hidden_size));
    is.read(reinterpret_cast<char*>(&intermediate_size), sizeof(intermediate_size));
    is.read(reinterpret_cast<char*>(&dropout_prob), sizeof(dropout_prob));

    auto ffn = std::make_unique<FeedForward>(hidden_size, intermediate_size, dropout_prob);

    is.read(reinterpret_cast<char*>(ffn->w1.get_data()),
            ffn->w1.rows() * ffn->w1.cols() * sizeof(float));
    is.read(reinterpret_cast<char*>(ffn->w2.get_data()),
            ffn->w2.rows() * ffn->w2.cols() * sizeof(float));
    is.read(reinterpret_cast<char*>(ffn->b1.data()), ffn->b1.size() * sizeof(float));
    is.read(reinterpret_cast<char*>(ffn->b2.data()), ffn->b2.size() * sizeof(float));

    return ffn;
}

Matrix FeedForward::backward(const Matrix& grad_output, const Matrix& input) {
    std::cout << "FeedForward::backward dimensions:" << std::endl;
    std::cout << "grad_output: " << grad_output.rows() << "x" << grad_output.cols() << std::endl;
    std::cout << "input: " << input.rows() << "x" << input.cols() << std::endl;
    std::cout << "intermediate_cache: " << intermediate_cache.rows() << "x" << intermediate_cache.cols() << std::endl;

    try {
#ifdef USE_CUDA
        // Compute gradients for second layer
        Matrix d_intermediate(grad_output.rows(), w2.rows());  // [batch_size x intermediate_size]
        std::cout << "d_intermediate dims: " << d_intermediate.rows() << "x" << d_intermediate.cols() << std::endl;
        
        Matrix w2_transposed = w2.transpose();
        if (!cuda::customMatrixMultiply(grad_output, w2_transposed, d_intermediate)) {
            throw std::runtime_error("Matrix multiplication failed");
        }
        
        // Compute gradients for GELU activation
        Matrix gelu_grad = intermediate_cache;  // Create copy for in-place modification
        cuda::gelu_backward(gelu_grad, d_intermediate);  // Compute GELU gradient in-place
        
        if (d_intermediate.rows() != gelu_grad.rows() || d_intermediate.cols() != gelu_grad.cols()) {
            throw std::runtime_error("Dimension mismatch in GELU backward: " + 
                std::to_string(d_intermediate.rows()) + "x" + std::to_string(d_intermediate.cols()) +
                " vs " + std::to_string(gelu_grad.rows()) + "x" + std::to_string(gelu_grad.cols()));
        }
        d_intermediate = d_intermediate.hadamard(gelu_grad);
        
        // Compute input gradients
        Matrix d_input(input.rows(), input.cols());  // [batch_size x hidden_size]
        std::cout << "d_input dims before matmul: " << d_input.rows() << "x" << d_input.cols() << std::endl;
        Matrix w1_transposed = w1.transpose();
        if (!cuda::customMatrixMultiply(d_intermediate, w1_transposed, d_input)) {
            throw std::runtime_error("Matrix multiplication failed");
        }
        std::cout << "d_input dims after matmul: " << d_input.rows() << "x" << d_input.cols() << std::endl;
        
        // Verify output dimensions match input dimensions
        if (d_input.rows() != input.rows() || d_input.cols() != input.cols()) {
            throw std::runtime_error("Output matrix has wrong dimensions: expected " +
                std::to_string(input.rows()) + "x" + std::to_string(input.cols()) +
                " got " + std::to_string(d_input.rows()) + "x" + std::to_string(d_input.cols()));
        }
        
        return d_input;
#else
        throw std::runtime_error("CUDA support not enabled");
#endif
    } catch (const std::exception& e) {
        throw std::runtime_error("FeedForward backward failed: " + std::string(e.what()));
    }
}

void FeedForward::update_parameters(const Matrix& grad) {
    float learning_rate = 0.01f;  // Could be made configurable
    
    w1 -= dW1_ * learning_rate;
    // Scale vector elements individually
    for (size_t i = 0; i < b1.size(); ++i) {
        b1[i] -= db1_[i] * learning_rate;
    }
    
    w2 -= dW2_ * learning_rate;
    // Scale vector elements individually
    for (size_t i = 0; i < b2.size(); ++i) {
        b2[i] -= db2_[i] * learning_rate;
    }
}

void FeedForward::initialize_weights() {
    // Get sizes from weight matrices
    size_t hidden_size = w1.rows();  // Input/output size
    size_t intermediate_size = w1.cols();  // Hidden layer size
    
    float scale = sqrt(2.0f / (hidden_size + intermediate_size));
    
    w1.initialize_random(scale);
    w2.initialize_random(scale);
    
    // Initialize biases to small non-zero values
    b1.initialize_constant(0.01f);
    b2.initialize_constant(0.01f);
}