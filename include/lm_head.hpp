#pragma once
#include "matrix.hpp"
#include <vector>
#ifdef USE_CUDA
#include <cuda_fp16.h>  // For half type
#endif

class LanguageModelHead {
public:
    virtual ~LanguageModelHead() = default;
    virtual Matrix project_to_vocab(const Matrix& input) = 0;
    virtual Matrix backward_pass(const Matrix& grad_output, const Matrix& hidden_states) = 0;
    virtual void load(std::istream& is) = 0;
    virtual void save(std::ostream& os) const = 0;
    
protected:
    size_t input_dim_;
    size_t vocab_size_;
    Matrix weights_;
    Matrix bias_;
};

class BasicLanguageModelHead : public LanguageModelHead {
public:
    BasicLanguageModelHead(size_t input_dim, size_t vocab_size);
    ~BasicLanguageModelHead() override = default;
    Matrix project_to_vocab(const Matrix& input) override;
    Matrix backward_pass(const Matrix& grad_output, const Matrix& hidden_states) override;
    void load(std::istream& is) override;
    void save(std::ostream& os) const override;
};

class OptimizedLanguageModelHead : public LanguageModelHead {
public:
    OptimizedLanguageModelHead(size_t hidden_size, size_t vocab_size);
    ~OptimizedLanguageModelHead() override;
    Matrix project_to_vocab(const Matrix& input) override;
    Matrix backward_pass(const Matrix& grad_output, const Matrix& hidden_states) override;
    void load(std::istream& is) override;
    void save(std::ostream& os) const override;
    
    // Additional optimized features
    void update_token_frequencies(const std::vector<int>& tokens);
    void update_active_tokens();

private:
    static constexpr size_t MIN_ACTIVE_TOKENS = 1000;
    Matrix hidden_states;
    std::vector<float> token_frequencies;
    std::vector<int> active_tokens;
    std::vector<size_t> active_token_indices;
    size_t training_steps;
    float pruning_threshold;
    
    Matrix forward_impl(const Matrix& hidden_states);
    void backward_linear(const Matrix& grad_output);

#ifdef USE_CUDA
    // CUDA-specific members
    bool is_cuda_{false};
    float* d_projection{nullptr};
    float* d_bias{nullptr};
    __half* d_projection_fp16{nullptr};
    __half* d_hidden_states_fp16{nullptr};
    __half* d_output_fp16{nullptr};
    float* d_output{nullptr};
    unsigned char* d_active_tokens{nullptr};
    int* d_active_token_indices{nullptr};
    cudaStream_t compute_stream{nullptr};
#endif
};