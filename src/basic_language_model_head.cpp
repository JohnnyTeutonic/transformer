#include "../include/lm_head.hpp"
#include "../include/cuda/matrix_ops.cuh"
#include "../include/matrix.hpp"
#include <cmath>
#include <stdexcept>

BasicLanguageModelHead::BasicLanguageModelHead(size_t input_dim, size_t vocab_size) {
    input_dim_ = input_dim;
    vocab_size_ = vocab_size;
    weights_ = Matrix(input_dim, vocab_size);
    bias_ = Matrix(1, vocab_size);
    
    // Initialize weights with Xavier initialization
    float scale = std::sqrt(2.0f / (input_dim + vocab_size));
    weights_.initialize_random(scale);
    bias_.initialize_constant(0.0f);
}

Matrix BasicLanguageModelHead::project_to_vocab(const Matrix& input) {
    if (input.cols() != input_dim_) {
        throw std::runtime_error("Input dimension mismatch in BasicLanguageModelHead");
    }
    
    // Compute logits using matrix multiplication
    Matrix logits = matmul(input, weights_);
    
    // Add bias
    for (size_t i = 0; i < logits.rows(); ++i) {
        for (size_t j = 0; j < logits.cols(); ++j) {
            logits(i, j) += bias_(0, j);
        }
    }
    
    return logits;
}

void BasicLanguageModelHead::load(std::istream& is) {
    // Read dimensions
    is.read(reinterpret_cast<char*>(&input_dim_), sizeof(input_dim_));
    is.read(reinterpret_cast<char*>(&vocab_size_), sizeof(vocab_size_));
    
    // Load matrices
    weights_ = Matrix(input_dim_, vocab_size_);
    bias_ = Matrix(1, vocab_size_);
    
    // Read the actual data
    for (size_t i = 0; i < weights_.rows(); ++i) {
        for (size_t j = 0; j < weights_.cols(); ++j) {
            float val;
            is.read(reinterpret_cast<char*>(&val), sizeof(float));
            weights_(i, j) = val;
        }
    }
    
    for (size_t j = 0; j < bias_.cols(); ++j) {
        float val;
        is.read(reinterpret_cast<char*>(&val), sizeof(float));
        bias_(0, j) = val;
    }
}

void BasicLanguageModelHead::save(std::ostream& os) const {
    // Write dimensions
    os.write(reinterpret_cast<const char*>(&input_dim_), sizeof(input_dim_));
    os.write(reinterpret_cast<const char*>(&vocab_size_), sizeof(vocab_size_));
    
    // Write matrices
    for (size_t i = 0; i < weights_.rows(); ++i) {
        for (size_t j = 0; j < weights_.cols(); ++j) {
            float val = weights_(i, j);
            os.write(reinterpret_cast<const char*>(&val), sizeof(float));
        }
    }
    
    for (size_t j = 0; j < bias_.cols(); ++j) {
        float val = bias_(0, j);
        os.write(reinterpret_cast<const char*>(&val), sizeof(float));
    }
}

Matrix BasicLanguageModelHead::backward_pass(const Matrix& grad_output, const Matrix& hidden_states) {
    // Simple gradient computation without optimizations
    Matrix grad_hidden = matmul(grad_output, weights_.transpose());
    
    // Update weights and bias
    Matrix grad_weights = matmul(hidden_states.transpose(), grad_output);
    for (size_t i = 0; i < weights_.rows(); ++i) {
        for (size_t j = 0; j < weights_.cols(); ++j) {
            weights_(i, j) -= 0.001f * grad_weights(i, j);  // Simple learning rate
        }
    }
    
    // Update bias
    for (size_t j = 0; j < bias_.cols(); ++j) {
        float grad_bias = 0.0f;
        for (size_t i = 0; i < grad_output.rows(); ++i) {
            grad_bias += grad_output(i, j);
        }
        bias_(0, j) -= 0.001f * grad_bias;
    }
    
    return grad_hidden;
} 