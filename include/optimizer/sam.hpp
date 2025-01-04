#pragma once

#include "../types.hpp"
#include "../matrix.hpp"
#include "optimizer.hpp"
#include <memory>
#include <vector>
#include <functional>

// Forward declaration
class LanguageModelHead;

class SAM {
public:
    SAM(float rho = 0.05f, float learning_rate = 0.001f)
        : rho(rho), learning_rate(learning_rate) {}
    
    void first_step(const std::vector<Matrix*>& params,
                   const std::vector<Matrix>& grads);
    
    void second_step(const std::vector<Matrix*>& params,
                    const std::vector<Matrix>& grads);
    
    void update_bias(const std::vector<std::reference_wrapper<FloatVector>>& bias_params,
                    const std::vector<FloatVector>& bias_grads);
    
    void compute_parameter_gradients(const Matrix& logits,
                                   const Matrix& target_distribution,
                                   std::vector<Matrix>& param_grads);
    
    Matrix compute_gradients(const Matrix& logits,
                           const Matrix& hidden_states,
                           LanguageModelHead* lm_head);

private:
    // Helper methods
    float compute_grad_norm(const std::vector<Matrix>& grads);
    void save_parameter_copies(const std::vector<Matrix*>& params);
    void restore_parameters(std::vector<Matrix*>& params);

    // Member variables
    float rho;
    float learning_rate;
    std::vector<Matrix> saved_params;
    std::vector<Matrix> momentum_buffers;
    std::vector<FloatVector> bias_momentum;
    Matrix current_gradients;
    std::vector<Matrix> parameter_copies;  // Added for save/restore functionality
};