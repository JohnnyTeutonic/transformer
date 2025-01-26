#include "../include/lm_head.hpp"
#include "../include/matrix.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

OptimizedLanguageModelHead::OptimizedLanguageModelHead(size_t hidden_size, size_t vocab_size) {
    input_dim_ = hidden_size;
    vocab_size_ = vocab_size;
    weights_ = Matrix(hidden_size, vocab_size);
    bias_ = Matrix(1, vocab_size);
    
    // Initialize weights with Xavier initialization
    float scale = std::sqrt(2.0f / (hidden_size + vocab_size));
    weights_.initialize_random(scale);
    bias_.initialize_constant(0.0f);
    
    // Initialize other members
    token_frequencies.resize(vocab_size, 0.0f);
    active_tokens.resize(vocab_size, 1);  // Start with all tokens active
    training_steps = 0;
    pruning_threshold = 1e-6f;
    
    // Initialize active token indices
    active_token_indices.reserve(vocab_size);
    for (size_t i = 0; i < vocab_size; i++) {
        active_token_indices.push_back(i);
    }
}

OptimizedLanguageModelHead::~OptimizedLanguageModelHead() {
#ifdef USE_CUDA
    if (cublas_handle) cublasDestroy(cublas_handle);
    if (d_projection) cudaFree(d_projection);
    if (d_bias) cudaFree(d_bias);
    if (d_projection_fp16) cudaFree(d_projection_fp16);
    if (d_hidden_states_fp16) cudaFree(d_hidden_states_fp16);
    if (d_output_fp16) cudaFree(d_output_fp16);
    if (d_output) cudaFree(d_output);
    if (d_active_tokens) cudaFree(d_active_tokens);
    if (d_active_token_indices) cudaFree(d_active_token_indices);
    if (compute_stream) cudaStreamDestroy(compute_stream);
#endif
}

Matrix OptimizedLanguageModelHead::project_to_vocab(const Matrix& input) {
    return forward_impl(input);
}

Matrix OptimizedLanguageModelHead::forward_impl(const Matrix& hidden_states) {
    if (hidden_states.cols() != input_dim_) {
        throw std::runtime_error("Input dimension mismatch in OptimizedLanguageModelHead");
    }
    
    // Compute logits using matrix multiplication
    Matrix logits = matmul(hidden_states, weights_);
    
    // Add bias
    for (size_t i = 0; i < logits.rows(); ++i) {
        for (size_t j = 0; j < logits.cols(); ++j) {
            if (active_tokens[j]) {  // Only compute for active tokens
                logits(i, j) += bias_(0, j);
            }
        }
    }
    
    return logits;
}

void OptimizedLanguageModelHead::update_token_frequencies(const std::vector<int>& tokens) {
    // Reset frequencies periodically to prevent over-accumulation
    if (training_steps % 1000 == 0) {
        std::fill(token_frequencies.begin(), token_frequencies.end(), 0.0f);
    }
    
    for (int token : tokens) {
        if (token >= 0 && static_cast<size_t>(token) < vocab_size_) {
            token_frequencies[token] += 1.0f;
        }
    }
    training_steps++;
    
    // Normalize frequencies
    if (!token_frequencies.empty()) {
        float max_freq = *std::max_element(token_frequencies.begin(), token_frequencies.end());
        if (max_freq > 0) {
            for (float& freq : token_frequencies) {
                freq /= max_freq;
            }
        }
    }
}

void OptimizedLanguageModelHead::update_active_tokens() {
    const float decay = 0.99f;
    
    // Decay frequencies
    for (float& freq : token_frequencies) {
        freq *= decay;
    }
    
    // Sort tokens by frequency
    std::vector<std::pair<float, size_t>> freq_pairs(vocab_size_);
    for (size_t i = 0; i < vocab_size_; i++) {
        freq_pairs[i] = {token_frequencies[i], i};
    }
    
    std::partial_sort(freq_pairs.begin(), 
                     freq_pairs.begin() + MIN_ACTIVE_TOKENS,
                     freq_pairs.end(),
                     std::greater<>());
    
    // Update active tokens
    std::fill(active_tokens.begin(), active_tokens.end(), 0);
    active_token_indices.clear();
    active_token_indices.reserve(MIN_ACTIVE_TOKENS);
    
    for (size_t i = 0; i < MIN_ACTIVE_TOKENS; i++) {
        size_t idx = freq_pairs[i].second;
        active_tokens[idx] = 1;
        active_token_indices.push_back(idx);
    }
}

Matrix OptimizedLanguageModelHead::backward_pass(const Matrix& grad_output, const Matrix& hidden_states) {
    // Compute gradients for weights and bias
    Matrix grad_hidden = matmul(grad_output, weights_.transpose());
    backward_linear(grad_output);
    return grad_hidden;
}

void OptimizedLanguageModelHead::backward_linear(const Matrix& grad_output) {
    // Update weights and bias based on gradients
    const float learning_rate = 0.001f;
    
    // Only update for active tokens
    for (size_t i = 0; i < weights_.rows(); ++i) {
        for (size_t j = 0; j < weights_.cols(); ++j) {
            if (active_tokens[j]) {
                weights_(i, j) -= learning_rate * grad_output(0, j);
            }
        }
    }
    
    for (size_t j = 0; j < bias_.cols(); ++j) {
        if (active_tokens[j]) {
            bias_(0, j) -= learning_rate * grad_output(0, j);
        }
    }
}

void OptimizedLanguageModelHead::load(std::istream& is) {
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
    
    // Initialize other members
    token_frequencies.resize(vocab_size_, 0.0f);
    active_tokens.resize(vocab_size_, 1);  // Start with all tokens active
    training_steps = 0;
    pruning_threshold = 1e-6f;
    
    // Initialize active token indices
    active_token_indices.clear();
    active_token_indices.reserve(vocab_size_);
    for (size_t i = 0; i < vocab_size_; i++) {
        active_token_indices.push_back(i);
    }
}

void OptimizedLanguageModelHead::save(std::ostream& os) const {
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