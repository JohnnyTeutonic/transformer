#include "../include/router.hpp"
#include <algorithm>
#include <vector>
#include <numeric>

#ifdef USE_CUDA
#include "../include/cuda/router_kernel.cuh"
#endif

Router::Router(size_t hidden_size, size_t num_experts, size_t top_k)
    : hidden_size_(hidden_size), num_experts_(num_experts), top_k_(top_k) {
    if (top_k > num_experts) {
        throw std::invalid_argument("top_k cannot be greater than num_experts");
    }
    initialize_weights();
}

void Router::initialize_weights() {
    params_.weights = Matrix(hidden_size_, num_experts_);
    grads_.weights_grad = Matrix(hidden_size_, num_experts_, 0.0f);
    
    // Xavier/Glorot initialization
    float scale = std::sqrt(2.0f / (hidden_size_ + num_experts_));
    params_.weights.initialize_random(scale);
}

RouterOutput Router::forward(const Matrix& hidden_states) {
#ifdef USE_CUDA
    // Convert input to CudaMatrix
    cuda::CudaMatrix d_hidden_states(hidden_states);

    // Ensure model parameters are on the GPU
    if (!params_gpu_) {
        params_gpu_ = std::make_unique<CudaParameters>();
        params_gpu_->weights = cuda::CudaMatrix(params_.weights);
    }

    // Allocate GPU memory for outputs
    cuda::CudaMatrix d_logits(hidden_states.rows(), num_experts_);
    cuda::CudaMatrix d_probabilities(hidden_states.rows(), num_experts_);
    cuda::CudaMatrix d_top_k_indices(hidden_states.rows(), top_k_);
    cuda::CudaMatrix d_top_k_weights(hidden_states.rows(), top_k_);
    
    // Launch kernel
    cuda::kernels::router_forward_kernel_launcher(
        d_hidden_states,
        params_gpu_->weights,
        top_k_,
        d_logits,
        d_probabilities,
        d_top_k_indices,
        d_top_k_weights
    );
    
    // Cache probabilities for backward pass
    softmax_cache_gpu_ = std::move(d_probabilities);

    // TODO: The aux loss calculation should also be done on the GPU.
    // For now, we copy back and do it on the CPU.
    Vector expert_token_fractions(num_experts_, 0.0f);
    Matrix top_k_indices_cpu = d_top_k_indices.to_matrix();
    std::vector<int> tokens_per_expert(num_experts_, 0);
    for (size_t i = 0; i < top_k_indices_cpu.rows(); ++i) {
        for (size_t k = 0; k < top_k_; ++k) {
            tokens_per_expert[static_cast<size_t>(top_k_indices_cpu(i, k))]++;
        }
    }
    for (size_t i = 0; i < num_experts_; ++i) {
        expert_token_fractions[i] = static_cast<float>(tokens_per_expert[i]) / (top_k_indices_cpu.rows() * top_k_);
    }

    return {top_k_indices_cpu, d_top_k_weights.to_matrix(), expert_token_fractions};

#else
    // CPU implementation
    input_cache_ = hidden_states;
    // 1. Compute router logits: hidden_states @ router_weights
    Matrix logits = hidden_states * params_.weights;

    // 2. Compute softmax over logits to get probabilities
    softmax_cache_ = logits.softmax(); // Cache for backward pass

    // 3. Select top-k experts for each token
    size_t batch_size = hidden_states.rows();
    Matrix top_k_indices(batch_size, top_k_);
    Matrix top_k_weights(batch_size, top_k_);

    // 4. Renormalize the weights of the top-k experts
    for (size_t i = 0; i < batch_size; ++i) {
        Vector probs = softmax_cache_.row(i);
        std::vector<size_t> indices(num_experts_);
        std::iota(indices.begin(), indices.end(), 0);

        // Partial sort to find the top k experts
        std::partial_sort(indices.begin(), indices.begin() + top_k_, indices.end(),
                          [&](size_t a, size_t b) {
                              return probs[a] > probs[b];
                          });
        
        // 4. Store top-k indices and re-normalize their weights
        float weight_sum = 0.0f;
        for (size_t k = 0; k < top_k_; ++k) {
            top_k_indices(i, k) = static_cast<float>(indices[k]);
            float weight = probs[indices[k]];
            top_k_weights(i, k) = weight;
            weight_sum += weight;
        }

        // Renormalize weights to sum to 1
        if (weight_sum > 1e-6) {
            for (size_t k = 0; k < top_k_; ++k) {
                top_k_weights(i, k) /= weight_sum;
            }
        }
    }

    // 5. Calculate the fraction of tokens dispatched to each expert (for aux loss)
    Vector expert_token_fractions(num_experts_, 0.0f);
    std::vector<int> tokens_per_expert(num_experts_, 0);
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t k = 0; k < top_k_; ++k) {
            tokens_per_expert[static_cast<size_t>(top_k_indices(i, k))]++;
        }
    }
    for (size_t i = 0; i < num_experts_; ++i) {
        expert_token_fractions[i] = static_cast<float>(tokens_per_expert[i]) / (batch_size * top_k_);
    }

    return {top_k_indices, top_k_weights, expert_token_fractions};
#endif
}

Matrix Router::backward(const Matrix& d_router_softmax) {
    // Gradient of softmax is (softmax * (1 - softmax))
    // However, the full jacobian is complex. A common simplification is to
    // compute the gradient w.r.t the logits.
    // d_logits = d_softmax * d(softmax)/d(logits)
    // Here, d_router_softmax is the gradient of the loss w.r.t the softmax output.
    
    // We can directly compute gradient w.r.t logits from d_softmax.
    // Let S = softmax_cache_, P = d_router_softmax (dL/dS)
    // dL/dLogits_i = sum_j(dL/dS_j * dS_j/dLogits_i)
    // dS_j/dLogits_i = S_j * (delta_ij - S_i)
    // dL/dLogits_i = sum_j(P_j * S_j * (delta_ij - S_i))
    // dL/dLogits_i = P_i * S_i - S_i * sum_j(P_j * S_j)

    size_t batch_size = softmax_cache_.rows();
    Matrix d_logits(batch_size, num_experts_, 0.0f);

    for (size_t i = 0; i < batch_size; i++) {
        Vector s = softmax_cache_.row(i);
        Vector p = d_router_softmax.row(i);
        float p_dot_s = 0.0f;
        for (size_t j = 0; j < num_experts_; j++) {
            p_dot_s += p[j] * s[j];
        }
        for (size_t j = 0; j < num_experts_; j++) {
            d_logits(i, j) = s[j] * (p[j] - p_dot_s);
        }
    }

    // 1. Gradient w.r.t weights: input.T @ d_logits
    grads_.weights_grad = input_cache_.transpose() * d_logits;

    // 2. Gradient w.r.t input: d_logits @ weights.T
    Matrix d_input = d_logits * params_.weights.transpose();

    return d_input;
}
