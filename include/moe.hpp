#pragma once

#ifdef USE_CUDA
// CRITICAL: Include math fix FIRST
#include "cuda/cuda_math_fix.hpp"
#endif

#include "feed_forward.hpp"
#include "router.hpp"
#include <vector>
#include <memory>

#ifdef USE_CUDA
#include "cuda/cuda_matrix.hpp"
#endif

class MixtureOfExperts {
public:
    MixtureOfExperts(size_t num_experts, size_t top_k, size_t hidden_size, size_t intermediate_size, float aux_loss_coefficient);
    
    // Delete copy constructor and assignment since we have unique_ptr members
    MixtureOfExperts(const MixtureOfExperts&) = delete;
    MixtureOfExperts& operator=(const MixtureOfExperts&) = delete;
    
    // Allow move operations
    MixtureOfExperts(MixtureOfExperts&&) = default;
    MixtureOfExperts& operator=(MixtureOfExperts&&) = default;

    Matrix forward(const Matrix& hidden_states);
    Matrix backward(const Matrix& grad_output);
    
    // Allow access to underlying parameters for optimizer and saving
    std::vector<FeedForward*>& get_experts() { return experts_ptrs_; }
    Router& get_router() { return router_; }
    float get_aux_loss() const { return last_aux_loss_; }
    
    // Additional methods for TransformerLayer compatibility
    void set_training(bool mode);
    void reset_state();
    void save(std::ostream& os) const;
    static std::unique_ptr<MixtureOfExperts> load(std::istream& is, const struct TransformerConfig& config);
    std::vector<std::reference_wrapper<Matrix>> get_weights();
    void update_parameters(float learning_rate);

private:
    size_t num_experts_;
    size_t top_k_;
    size_t hidden_size_;
    size_t intermediate_size_;
    float aux_loss_coefficient_;

    std::vector<std::unique_ptr<FeedForward>> experts_;
    std::vector<FeedForward*> experts_ptrs_; // For easier access
    Router router_;
    float last_aux_loss_ = 0.0f;

#ifdef USE_CUDA
    std::vector<std::unique_ptr<FeedForward::CudaParameters>> experts_params_gpu_;
#endif

    // Cache for backward pass
    Matrix input_cache_;
    Matrix top_k_indices_cache_;
    Matrix top_k_weights_cache_;
    std::vector<std::vector<Matrix>> expert_outputs_cache_;
};
