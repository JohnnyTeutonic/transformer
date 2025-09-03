#include "../include/moe.hpp"
#include <stdexcept>

#ifdef USE_CUDA
#include "../include/cuda/moe_kernel.cuh"
#endif

MixtureOfExperts::MixtureOfExperts(size_t num_experts, size_t top_k, size_t hidden_size, size_t intermediate_size, float aux_loss_coefficient)
    : num_experts_(num_experts), 
      top_k_(top_k), 
      hidden_size_(hidden_size), 
      intermediate_size_(intermediate_size),
      aux_loss_coefficient_(aux_loss_coefficient),
      router_(hidden_size, num_experts, top_k) {
    
    if (top_k > num_experts) {
        throw std::runtime_error("top_k cannot be greater than num_experts.");
    }

    experts_.reserve(num_experts);
    experts_ptrs_.reserve(num_experts);
    for (size_t i = 0; i < num_experts; ++i) {
        experts_.push_back(std::make_unique<FeedForward>(hidden_size, intermediate_size));
        experts_ptrs_.push_back(experts_.back().get());
    }
}

Matrix MixtureOfExperts::forward(const Matrix& hidden_states) {
#ifdef USE_CUDA
    // The full MoE forward pass on the GPU is complex.
    // It involves: Router -> Dispatch -> Expert Forward -> Combine.
    // We have the router kernel. The dispatch/combine is a placeholder.
    // For now, we'll perform the routing on GPU, then fallback to CPU for dispatch/combine.
    
    // 1. Perform routing on GPU
    cuda::CudaMatrix d_hidden_states(hidden_states);
    // ... (logic to call router kernel) ...
    
    // 2. Copy routing results and hidden states to CPU for dispatch/combine
    // Matrix top_k_indices = d_top_k_indices.to_matrix();
    // Matrix top_k_weights = d_top_k_weights.to_matrix();
    // ... rest of CPU logic ...
    
    // This hybrid approach is inefficient. Since the dispatch kernel is a placeholder,
    // we will just keep the entire MoE forward pass on the CPU for now when USE_CUDA is defined.
    // This avoids unnecessary data transfer and ensures correctness.
#endif

    // --- CPU Implementation ---
    input_cache_ = hidden_states;
    size_t batch_size = hidden_states.rows();

    // Get routing decisions from the router
    RouterOutput router_output = router_.forward(hidden_states);
    top_k_indices_cache_ = router_output.top_k_indices;
    top_k_weights_cache_ = router_output.top_k_weights;

    // Calculate auxiliary load balancing loss
    float aux_loss = 0.0f;
    for (size_t i = 0; i < num_experts_; ++i) {
        // The loss encourages the product of token fraction and router probability to be high for all experts.
        // We don't have the mean router probability directly here, so we use a simplification based on token fractions.
        // A common formulation is to encourage the sum of squares of the fractions to be minimal, which encourages uniformity.
        // Loss = sum(fraction_i^2)
        aux_loss += router_output.expert_token_fractions[i] * router_output.expert_token_fractions[i];
    }
    // The final loss is scaled by the number of experts and a coefficient
    last_aux_loss_ = aux_loss * num_experts_ * aux_loss_coefficient_;


    Matrix final_output(batch_size, hidden_size_, 0.0f);
    expert_outputs_cache_.assign(batch_size, std::vector<Matrix>(top_k_));


    // This is a naive implementation. A production-ready version would batch tokens
    // by expert to avoid redundant computation and maximize hardware utilization.
    // However, this version is functionally correct and easier to understand.

    // 2. Process each token
    for (size_t i = 0; i < batch_size; ++i) {
        Vector token_hidden_state = hidden_states.row(i);
        Matrix token_input(1, hidden_size_);
        token_input.set_row(0, token_hidden_state);

        Vector combined_expert_output(hidden_size_, 0.0f);

        // 3. Pass token to its assigned top-k experts
        for (size_t k = 0; k < top_k_; ++k) {
            size_t expert_index = static_cast<size_t>(top_k_indices(i, k));
            float weight = top_k_weights(i, k);

            if (expert_index >= experts_.size()) {
                throw std::runtime_error("Expert index out of bounds.");
            }

            // Get expert output
            Matrix expert_output_matrix = experts_[expert_index]->forward(token_input);
            expert_outputs_cache_[i][k] = expert_output_matrix; // Cache for backward pass
            Vector expert_output = expert_output_matrix.row(0);

            // 4. Combine expert outputs with router weights
            for (size_t d = 0; d < hidden_size_; ++d) {
                combined_expert_output[d] += expert_output[d] * weight;
            }
        }
        final_output.set_row(i, combined_expert_output);
    }

    return final_output;
}

Matrix MixtureOfExperts::backward(const Matrix& grad_output) {
    size_t batch_size = input_cache_.rows();
    Matrix d_input(batch_size, hidden_size_, 0.0f);
    Matrix d_router_softmax(batch_size, num_experts_, 0.0f);

    for (size_t i = 0; i < batch_size; ++i) {
        Vector grad_output_token = grad_output.row(i);
        Matrix grad_output_token_matrix(1, hidden_size_);
        grad_output_token_matrix.set_row(0, grad_output_token);
        
        Vector d_input_token(hidden_size_, 0.0f);
        Matrix token_input(1, hidden_size_);
        token_input.set_row(0, input_cache_.row(i));

        for (size_t k = 0; k < top_k_; ++k) {
            size_t expert_idx = static_cast<size_t>(top_k_indices_cache_(i, k));
            float weight = top_k_weights_cache_(i, k);
            
            // 1. Calculate gradient w.r.t router weights (dLoss/dWeight)
            const Matrix& expert_output_matrix = expert_outputs_cache_[i][k];
            float d_weight = 0.0f;
            for(size_t d=0; d<hidden_size_; ++d) {
                d_weight += grad_output_token[d] * expert_output_matrix(0, d);
            }
            d_router_softmax(i, expert_idx) = d_weight;

            // 2. Calculate gradient w.r.t expert output (dLoss/dExpertOutput)
            Matrix grad_to_expert_output = grad_output_token_matrix * weight;
            
            // 3. Backpropagate through the expert
            Matrix d_expert_input_matrix = experts_[expert_idx]->backward(grad_to_expert_output, token_input);

            // 4. Accumulate gradient w.r.t input
            Vector d_expert_input = d_expert_input_matrix.row(0);
            for(size_t d=0; d<hidden_size_; ++d) {
                d_input_token[d] += d_expert_input[d];
            }
        }
        d_input.set_row(i, d_input_token);
    }
    
    // 5. Backpropagate through the router
    router_.backward(d_router_softmax);

    return d_input;
}
