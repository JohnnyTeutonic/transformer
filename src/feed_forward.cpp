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
      b2(hidden_size), dropout_prob(dropout), intermediate_cache(1, intermediate_size),
      // Initialize gradients with same dimensions as their parameters
      dW1_(hidden_size, intermediate_size), dW2_(intermediate_size, hidden_size),
      db1_(intermediate_size), db2_(hidden_size) {

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

    // Initialize biases to zero
    for (size_t i = 0; i < b1.size(); ++i)
        b1[i] = 0.0f;
    for (size_t i = 0; i < b2.size(); ++i)
        b2[i] = 0.0f;

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
}

Matrix FeedForward::forward(const Matrix& input) {
    std::cout << "\n=== FeedForward::forward START ===" << std::endl;
    std::cout << "Input dimensions: " << input.rows() << "x" << input.cols() << std::endl;
    
    // Declare matrices outside of conditional blocks
    Matrix intermediate, output;
    bool using_cuda = false;
    
#ifdef USE_CUDA
    // Try to initialize CUDA first
    cudaError_t err = cudaFree(0);  // Simple CUDA runtime test
    if (err == cudaSuccess) {
        printf("Using CUDA\n");
        auto& memory_mgr = cuda::MemoryManager::get_instance();
        printf("Memory manager instance obtained\n");
        
        try {
            // Allocate memory for intermediate and output matrices
            size_t intermediate_size = input.rows() * w1.cols();
            size_t output_size = input.rows() * w2.cols();
            
            float* intermediate_data = memory_mgr.get_device_memory(intermediate_size);
            float* output_data = memory_mgr.get_device_memory(output_size);
            
            // Create Matrix objects that wrap the device memory
            Matrix intermediate(input.rows(), w1.cols(), intermediate_data, false);  // false means don't take ownership
            Matrix output(input.rows(), w2.cols(), output_data, false);
            
            // First matrix multiplication
            intermediate = cuda::matmul(input, w1);
            printf("First matmul completed\n");
            
            // Apply bias and ReLU using CUDA kernels
            cuda::FFNOps::add_bias_and_relu(intermediate, b1.data(), b1.size());
            printf("Bias and ReLU applied\n");
            
            // Second matrix multiplication
            output = cuda::matmul(intermediate, w2);
            printf("Second matmul completed\n");
            
            // Now safe to move the final results
            intermediate = std::move(intermediate);
            output = std::move(output);
            using_cuda = true;
            printf("CUDA computations completed successfully\n");
        } catch (const std::exception& e) {
            std::cerr << "Error in CUDA memory allocation: " << e.what() << std::endl;
            std::cout << "Falling back to CPU implementation" << std::endl;
        }
    } else {
        std::cout << "CUDA not available, falling back to CPU implementation" << std::endl;
    }
#endif

    // If CUDA failed or isn't available, use CPU implementation
    if (!using_cuda) {
        intermediate = Matrix(input.rows(), w1.cols());
        output = Matrix(input.rows(), w2.cols());
        std::cout << "Using CPU" << std::endl;
        matmul(input, w1, intermediate);
    }

    // Debug intermediate values before ReLU
    float min_pre_relu = std::numeric_limits<float>::infinity();
    float max_pre_relu = -std::numeric_limits<float>::infinity();
    float sum_pre_relu = 0.0f;
    size_t nonzero_pre_relu = 0;
    std::cout << "Intermediate matrix: " << intermediate.rows() << "x" << intermediate.cols() << std::endl;
    for (size_t i = 0; i < intermediate.rows(); ++i) {
        for (size_t j = 0; j < intermediate.cols(); ++j) {
            float val = intermediate(i, j) + b1[j];
            min_pre_relu = std::min(min_pre_relu, val);
            max_pre_relu = std::max(max_pre_relu, val);
            sum_pre_relu += val;
            if (std::abs(val) > 1e-6) nonzero_pre_relu++;
        }
    }
    
    std::cout << "Before ReLU:" << std::endl;
    std::cout << "Value range: [" << min_pre_relu << ", " << max_pre_relu << "]" << std::endl;
    std::cout << "Mean: " << sum_pre_relu / (intermediate.rows() * intermediate.cols()) << std::endl;
    std::cout << "Nonzero: " << nonzero_pre_relu << "/" 
              << (intermediate.rows() * intermediate.cols()) << std::endl;
    
    // Add bias and apply activation
    for (size_t i = 0; i < intermediate.rows(); ++i) {
        for (size_t j = 0; j < intermediate.cols(); ++j) {
            intermediate(i, j) += b1[j];
            intermediate(i, j) = std::max(0.0f, intermediate(i, j)); // ReLU activation
        }
    }
    
    // Debug intermediate values after ReLU
    float min_post_relu = std::numeric_limits<float>::infinity();
    float max_post_relu = -std::numeric_limits<float>::infinity();
    float sum_post_relu = 0.0f;
    size_t nonzero_post_relu = 0;
    
    for (size_t i = 0; i < intermediate.rows(); ++i) {
        for (size_t j = 0; j < intermediate.cols(); ++j) {
            float val = intermediate(i, j);
            min_post_relu = std::min(min_post_relu, val);
            max_post_relu = std::max(max_post_relu, val);
            sum_post_relu += val;
            if (std::abs(val) > 1e-6) nonzero_post_relu++;
        }
    }
    
    std::cout << "After ReLU:" << std::endl;
    std::cout << "Value range: [" << min_post_relu << ", " << max_post_relu << "]" << std::endl;
    std::cout << "Mean: " << sum_post_relu / (intermediate.rows() * intermediate.cols()) << std::endl;
    std::cout << "Nonzero: " << nonzero_post_relu << "/" 
              << (intermediate.rows() * intermediate.cols()) << std::endl;

    // Second matrix multiplication
    if (using_cuda) {
#ifdef USE_CUDA
        output = cuda::matmul(intermediate, w2);
#endif
    } else {
        matmul(intermediate, w2, output);
    }
    
    // Add bias and debug final output
    float min_output = std::numeric_limits<float>::infinity();
    float max_output = -std::numeric_limits<float>::infinity();
    float sum_output = 0.0f;
    size_t nonzero_output = 0;
    
    for (size_t i = 0; i < output.rows(); ++i) {
        for (size_t j = 0; j < output.cols(); ++j) {
            output(i, j) += b2[j];
            float val = output(i, j);
            min_output = std::min(min_output, val);
            max_output = std::max(max_output, val);
            sum_output += val;
            if (std::abs(val) > 1e-6) nonzero_output++;
        }
    }
    
    std::cout << "Final output:" << std::endl;
    std::cout << "Value range: [" << min_output << ", " << max_output << "]" << std::endl;
    std::cout << "Mean: " << sum_output / (output.rows() * output.cols()) << std::endl;
    std::cout << "Nonzero: " << nonzero_output << "/" 
              << (output.rows() * output.cols()) << std::endl;
    
    std::cout << "=== FeedForward::forward END ===\n" << std::endl;
    return output;
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
        
        d_intermediate = cuda::matmul(grad_output, w2.transpose());
        
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
        d_input = cuda::matmul(d_intermediate, w1.transpose());
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