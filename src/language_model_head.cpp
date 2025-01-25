#include "../include/lm_head.hpp"
#include "../include/cuda/matrix_ops.cuh"

LanguageModelHead::LanguageModelHead(unsigned long input_dim, unsigned long vocab_size) 
    : input_dim_(input_dim)
    , vocab_size_(vocab_size)
    , weights_(input_dim, vocab_size)
    , bias_(1, vocab_size) {
    
    // Initialize weights
    weights_.initialize_random(0.02f); // Xavier initialization
    
    // Initialize bias
    bias_.initialize_constant(0.0f);
}

LanguageModelHead::~LanguageModelHead() = default;

Matrix LanguageModelHead::project_to_vocab(const Matrix& input) {
    // Perform matrix multiplication: input * weights + bias
    Matrix output = matrix_multiply(input, weights_);
    
    // Add bias to each row
    for (size_t i = 0; i < output.rows(); ++i) {
        for (size_t j = 0; j < output.cols(); ++j) {
            output(i, j) += bias_(0, j);
        }
    }
    
    return output;
} 