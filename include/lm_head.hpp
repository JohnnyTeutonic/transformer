#pragma once
#include "components.hpp"
#include "cuda_utils.hpp"
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#endif

/**
 * @brief Language model head for token prediction in transformer models.
 * 
 * The LanguageModelHead class transforms hidden states into logits over the vocabulary,
 * enabling token prediction for language modeling tasks. Features include:
 * - Linear projection to vocabulary size
 * - Bias terms for each token
 * - Adaptive token frequency tracking
 * - Adam optimizer integration
 * - Dropout regularization
 */
class LanguageModelHead {
  private:
    Matrix weights_;
    Matrix bias_;
    unsigned long input_dim_;
    unsigned long vocab_size_;
    float dropout_prob;                  ///< Dropout probability during training
    size_t hidden_size_;                 ///< Size of input hidden states
    Matrix hidden_states;                ///< Cached hidden states for backward pass
    Matrix hidden_states_;               ///< Cached hidden states for forward pass
    std::vector<float> token_frequencies; ///< Tracked frequencies of token usage

    // Vocabulary pruning
    static constexpr size_t PRUNE_INTERVAL = 100;  // Update active tokens every N steps
    static constexpr size_t MIN_ACTIVE_TOKENS = 1000;  // Minimum number of active tokens
    float pruning_threshold;
    std::vector<unsigned char> active_tokens;  // Changed from vector<bool> to vector<unsigned char>
    std::vector<int> active_token_indices;     // List of indices of active tokens
    size_t training_steps;
    
    // Pinned memory for efficient GPU transfers
    float* h_projection = nullptr;
    float* h_bias = nullptr;

    // Device memory buffers
    float* d_projection = nullptr;  // Device copy of projection matrix
    float* d_bias = nullptr;       // Device copy of bias
    half* d_projection_fp16 = nullptr;  // FP16 version of projection
    half* d_hidden_states_fp16 = nullptr;  // FP16 version of input
    half* d_output_fp16 = nullptr;  // FP16 intermediate output
    float* d_output = nullptr;      // Final FP32 output

    /**
     * @brief Computes gradients for the linear projection.
     * @param grad_output Gradient of the loss with respect to the output
     */
    void backward_linear(const Matrix& grad_output);

    /**
     * @brief Implementation of the forward pass computation.
     * @param hidden_states Input hidden states
     * @return Output logits over vocabulary
     */
    Matrix forward_impl(const Matrix& hidden_states);

    void update_active_tokens();

#ifdef USE_CUDA
    // CUDA streams and synchronization
    cudaStream_t compute_stream;

    // Device memory for active tokens and indices
    unsigned char* d_active_tokens = nullptr;
    int* d_active_token_indices = nullptr;

    // Maximum batch size for memory allocation
    static constexpr size_t max_batch_size = 4096;  // Adjust based on your needs

    // CUDA kernel launchers
    __host__ void launch_convert_to_fp16(half* output, const float* input, size_t size);
    __host__ void launch_convert_and_expand_vocab(
        float* output, const half* input, size_t batch_size, size_t vocab_size, size_t active_vocab_size);

    cublasHandle_t cublas_handle;
#endif

  public:
    /**
     * @brief Constructs a language model head.
     * @param hidden_size Size of input hidden states
     * @param vocab_size Size of the vocabulary
     */
    LanguageModelHead(unsigned long input_dim, unsigned long vocab_size);

    ~LanguageModelHead();  // Just declare it here

    /**
     * @brief Performs the forward pass, computing logits from hidden states.
     * @param hidden_states Input hidden states
     * @return Matrix of logits over vocabulary
     */
    Matrix forward(const Matrix& hidden_states) {
        hidden_states_ = hidden_states;  // Cache for backward pass
        return project_to_vocab(hidden_states);
    }

    /**
     * @brief Performs the backward pass with Adam optimization.
     * @param grad_output Gradient of the loss with respect to the output
     * @param hidden_states Original input hidden states
     * @return Gradient with respect to the input
     */
    Matrix backward_pass(const Matrix& grad_output, const Matrix& hidden_states) {
        Matrix grad_proj = matmul(grad_output.transpose(), hidden_states);
        Matrix grad_bias = Matrix(1, vocab_size_, 0.0f);
        
        // Sum gradients for bias
        for (size_t i = 0; i < grad_output.rows(); ++i) {
            for (size_t j = 0; j < grad_output.cols(); ++j) {
                grad_bias(0, j) += grad_output(i, j);
            }
        }

        // Apply weight updates with adaptive learning rate
        float lr = 0.001f;
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float eps = 1e-8f;

        static Matrix m_proj(weights_.rows(), weights_.cols(), 0.0f);
        static Matrix v_proj(weights_.rows(), weights_.cols(), 0.0f);
        static Matrix m_bias(1, vocab_size_, 0.0f);
        static Matrix v_bias(1, vocab_size_, 0.0f);
        static size_t t = 0;
        t++;

        // Update projection matrix
        for (size_t i = 0; i < weights_.rows(); ++i) {
            for (size_t j = 0; j < weights_.cols(); ++j) {
                m_proj(i, j) = beta1 * m_proj(i, j) + (1 - beta1) * grad_proj(i, j);
                v_proj(i, j) = beta2 * v_proj(i, j) + (1 - beta2) * grad_proj(i, j) * grad_proj(i, j);
                float m_hat = m_proj(i, j) / (1 - std::pow(beta1, t));
                float v_hat = v_proj(i, j) / (1 - std::pow(beta2, t));
                weights_(i, j) -= lr * m_hat / (std::sqrt(v_hat) + eps);
            }
        }

        // Update bias matrix
        for (size_t j = 0; j < bias_.cols(); ++j) {
            m_bias(0, j) = beta1 * m_bias(0, j) + (1 - beta1) * grad_bias(0, j);
            v_bias(0, j) = beta2 * v_bias(0, j) + (1 - beta2) * grad_bias(0, j) * grad_bias(0, j);
            float m_hat = m_bias(0, j) / (1 - std::pow(beta1, t));
            float v_hat = v_bias(0, j) / (1 - std::pow(beta2, t));
            bias_(0, j) -= lr * m_hat / (std::sqrt(v_hat) + eps);
        }

        return matmul(grad_output, weights_);
    }

    /**
     * @brief Saves the model head to a stream.
     * @param os Output stream to save to
     */
    void save(std::ostream& os) const {
        weights_.save(os);
        bias_.save(os);
        os.write(reinterpret_cast<const char*>(&dropout_prob), sizeof(dropout_prob));
    }

    /**
     * @brief Loads a model head from a stream.
     * @param is Input stream to load from
     * @return Unique pointer to loaded model head
     */
    static std::unique_ptr<LanguageModelHead> load(std::istream& is) {
        auto lm_head = std::make_unique<LanguageModelHead>(0, 0); // Temporary sizes
        lm_head->weights_ = Matrix::load(is);
        lm_head->bias_ = Matrix::load(is);  // Changed from Vector::load to Matrix::load
        is.read(reinterpret_cast<char*>(&lm_head->dropout_prob), sizeof(lm_head->dropout_prob));
        return lm_head;
    }

    /**
     * @brief Gets references to trainable parameters.
     * @return Vector of parameter references
     */
    std::vector<std::reference_wrapper<Matrix>> get_parameters() {
        std::vector<std::reference_wrapper<Matrix>> params;
        params.push_back(std::ref(weights_));
        params.push_back(std::ref(bias_));  // Now we can include bias since it's a Matrix
        return params;
    }

    /**
     * @brief Gets the bias matrix.
     * @return Reference to bias matrix
     */
    Matrix& get_bias() {
        return bias_;
    }

    /**
     * @brief Projects hidden states to vocabulary space.
     * @param hidden_states Input hidden states
     * @return Matrix of logits over vocabulary
     */
    Matrix project_to_vocab(const Matrix& hidden_states);

    /**
     * @brief Performs backward pass with optional target distribution.
     * @param grad_output Gradient of the loss with respect to the output
     * @param target_distribution Optional target distribution for distillation
     * @return Gradient with respect to the input
     */
    Matrix backward(const Matrix& grad_output, const Matrix& target_distribution = Matrix());

    /**
     * @brief Updates token frequencies based on observed tokens.
     * @param tokens Vector of token indices observed in the current batch
     */
    void update_token_frequencies(const std::vector<int>& tokens);

    /**
     * @brief Prunes vocabulary by removing infrequently used tokens.
     * @param min_frequency_threshold Minimum frequency threshold for keeping tokens
     */
    void prune_vocabulary(float min_frequency_threshold = 1e-5);

    // Add helper function for matrix multiplication
    Matrix matrix_multiply(const Matrix& a, const Matrix& b) {
        if (a.cols() != b.rows()) {
            throw std::runtime_error("Matrix dimensions don't match for multiplication");
        }
        
        Matrix result(a.rows(), b.cols(), 0.0f);
        for (size_t i = 0; i < a.rows(); ++i) {
            for (size_t j = 0; j < b.cols(); ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < a.cols(); ++k) {
                    sum += a(i, k) * b(k, j);
                }
                result(i, j) = sum;
            }
        }
        return result;
    }
};