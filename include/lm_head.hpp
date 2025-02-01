#pragma once
#include "components.hpp"
#include "layer_norm.hpp"
#include "tiktoken_tokenizer.hpp"
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <deque>

// Only include CUDA headers if CUDA is available
#if defined(USE_CUDA) && defined(CUDA_AVAILABLE)
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
using half = __half;  // Define half type alias for CUDA's __half
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
    // Core model components
    Matrix projection;                    ///< Projection matrix to vocabulary space
    Vector bias;                         ///< Bias terms for each token
    float dropout_prob;                  ///< Dropout probability during training
    size_t vocab_size_;                  ///< Size of the vocabulary
    size_t hidden_size_;                 ///< Size of input hidden states
    Matrix hidden_states;                ///< Cached hidden states for backward pass
    Matrix hidden_states_;               ///< Cached hidden states for forward pass
    std::vector<float> token_frequencies; ///< Tracked frequencies of token usage
    float pruning_threshold;
    std::vector<unsigned char> active_tokens;  // Changed from vector<bool> to vector<unsigned char>
    std::vector<int> active_token_indices;     // List of indices of active tokens
    size_t training_steps;
    bool is_training_;  // Add training state member variable
    
    // Adam optimizer state
    Matrix m_proj;  // Momentum for projection
    Matrix v_proj;  // RMSprop for projection
    Vector m_bias;  // Momentum for bias
    Vector v_bias;  // RMSprop for bias
    size_t t;      // Time step
    float beta1;   // Momentum parameter
    float beta2;   // RMSprop parameter
    float eps;     // Small constant for numerical stability
    
    // Learning rate adaptation
    float current_lr;  // Current learning rate
    float min_lr;      // Minimum learning rate
    float max_lr;      // Maximum learning rate
    float lr_decay;    // Learning rate decay factor
    float lr_growth;   // Learning rate growth factor
    std::deque<float> loss_history;
    static constexpr size_t LOSS_HISTORY_SIZE = 100;
    float prev_loss = std::numeric_limits<float>::infinity();
    
    void update_learning_rate(float current_loss);
    
    // Pinned memory for efficient GPU transfers
    float* h_projection = nullptr;
    float* h_bias = nullptr;

    // Regular device memory buffers
    float* d_projection = nullptr;  // Device copy of projection matrix
    float* d_bias = nullptr;       // Device copy of bias
    float* d_output = nullptr;      // Final FP32 output
    unsigned char* d_active_tokens = nullptr;

#if defined(USE_CUDA) && defined(CUDA_AVAILABLE)
    // CUDA-specific members
    cudaStream_t compute_stream;
    cublasHandle_t cublas_handle;
    
    // FP16 device memory buffers
    half* d_projection_fp16 = nullptr;
    half* d_hidden_states_fp16 = nullptr;
    half* d_output_fp16 = nullptr;

    // CUDA kernel declarations
    __device__ static void convert_to_fp16_kernel(half* output, const float* input, size_t idx);
    __device__ static void convert_and_expand_vocab_kernel(
        float* output, const half* input, const unsigned char* active_tokens,
        size_t row, size_t col, size_t batch_size, size_t vocab_size, size_t active_vocab_size);

    // CUDA host functions
    void launch_convert_to_fp16(half* output, const float* input, size_t size);
    void launch_convert_and_expand_vocab(
        float* output, const half* input,
        size_t batch_size, size_t vocab_size, size_t active_vocab_size);
#endif

    // Helper methods
    void bias_completion_format(Matrix& logits);
    
    // Core components
    std::unique_ptr<LayerNorm> layer_norm;  ///< Layer normalization
    std::shared_ptr<TiktokenTokenizer> tokenizer;  ///< Tokenizer instance

    // Private helper functions
    void backward_linear(const Matrix& grad_output);
    void update_active_tokens();
    static constexpr size_t MIN_ACTIVE_TOKENS = 1000;  // Minimum number of active tokens to maintain

#if defined(USE_CUDA) && defined(CUDA_AVAILABLE)
    // Additional CUDA device memory
    int* d_active_token_indices = nullptr;
#endif

  public:
    // Constructors and destructor
    LanguageModelHead(size_t hidden_size, size_t vocab_size);
    LanguageModelHead(const LanguageModelHead& other);  // Copy constructor
    ~LanguageModelHead();

    // Core functionality
    Matrix forward(const Matrix& hidden_states, bool training = false);
    Matrix backward_pass(const Matrix& grad_output, const Matrix& hidden_states);
    Matrix project_to_vocab(const Matrix& hidden_states);
    Matrix backward(const Matrix& grad_output, const Matrix& target_distribution = Matrix());
    
    // Serialization
    void save(std::ostream& os) const;
    static std::unique_ptr<LanguageModelHead> load(std::istream& is);

    // Getters and setters
    std::vector<std::reference_wrapper<Matrix>> get_parameters();
    Matrix& get_weights() { return projection; }
    const Matrix& get_weights() const { return projection; }
    Vector& get_bias() { return bias; }
    const Vector& get_bias() const { return bias; }
    const std::vector<float>& get_token_frequencies() const { return token_frequencies; }
    void set_training(bool training_mode);
    void set_tokenizer(std::shared_ptr<TiktokenTokenizer> tok) { tokenizer = tok; }

    // Token management
    void update_token_frequencies(const std::vector<int>& tokens);
    void prune_vocabulary(float min_frequency_threshold = 1e-5);
};
