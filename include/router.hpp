#pragma once

#include "matrix.hpp"
#include <vector>
#include <tuple>

#ifdef USE_CUDA
#include "cuda/matrix.cuh"
#endif

struct RouterOutput {
    Matrix top_k_indices;
    Matrix top_k_weights;
    Vector expert_token_fractions;
};

class Router {
public:
    struct Parameters {
        Matrix weights; // [hidden_size, num_experts]
    };

    struct Gradients {
        Matrix weights_grad;
    };

#ifdef USE_CUDA
    struct CudaParameters { cuda::CudaMatrix weights; };
#endif

    Router(size_t hidden_size, size_t num_experts, size_t top_k);

    // Returns a tuple containing:
    // 1. A matrix of top-k expert indices for each token [batch_size * seq_len, top_k]
    // 2. A matrix of corresponding renormalized weights (softmax scores) for the top-k experts [batch_size * seq_len, top_k]
    RouterOutput forward(const Matrix& hidden_states);
    
    // Backward pass to compute gradients for the router's weights
    Matrix backward(const Matrix& grad_output);

    Parameters& parameters() { return params_; }
    Gradients& gradients() { return grads_; }

private:
    void initialize_weights();

    size_t hidden_size_;
    size_t num_experts_;
    size_t top_k_;

    Parameters params_;
    Gradients grads_;
    
    // CPU Cache
    Matrix input_cache_;
    Matrix softmax_cache_;

#ifdef USE_CUDA
    // GPU Cache and parameters
    std::unique_ptr<CudaParameters> params_gpu_;
    cuda::CudaMatrix softmax_cache_gpu_;
#endif
};
