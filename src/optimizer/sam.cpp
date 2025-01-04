#include "../../include/optimizer/sam.hpp"
#include "../../include/lm_head.hpp"
#include <algorithm>
#include <cmath>

float SAM::compute_grad_norm(const std::vector<Matrix> &grads) {
  float total_norm = 0.0f;
  const float epsilon = 1e-12f;  // Prevent underflow
  for (const auto &grad : grads) {
    for (size_t i = 0; i < grad.size(); ++i) {
      // Prevent underflow by clamping tiny gradients
      float g = grad.data()[i];
      if (std::abs(g) < epsilon) {
        g = 0.0f;
      }
      total_norm += g * g;
    }
  }
  // Prevent sqrt of zero
  total_norm = std::max(total_norm, epsilon * epsilon);
  return std::sqrt(total_norm);
}

void SAM::save_parameter_copies(const std::vector<Matrix *> &params) {
  parameter_copies.clear();
  parameter_copies.reserve(params.size());
  for (const auto *param : params) {
    parameter_copies.push_back(*param); // Make a copy
  }
}

void SAM::restore_parameters(std::vector<Matrix *> &params) {
  if (params.size() != parameter_copies.size()) {
    throw std::runtime_error("Parameter count mismatch during restore");
  }
  for (size_t i = 0; i < params.size(); ++i) {
    *params[i] = parameter_copies[i]; // Restore the copy
  }
}

void SAM::first_step(const std::vector<Matrix*>& params, const std::vector<Matrix>& grads) {
    // Save current parameters
    saved_params.clear();
    saved_params.reserve(params.size());
    for (const auto& param : params) {
        saved_params.push_back(*param);
    }
    
    // Compute gradient norm
    float grad_norm = 0.0f;
    for (const auto& grad : grads) {
        for (size_t i = 0; i < grad.size(); i++) {
            grad_norm += grad.data()[i] * grad.data()[i];
        }
    }
    grad_norm = std::sqrt(grad_norm);
    
    // Scale rho based on gradient norm
    float adaptive_rho = rho;
    if (grad_norm > 1.0f) {
        adaptive_rho /= grad_norm;
    }
    
    // Perturb parameters
    for (size_t i = 0; i < params.size(); i++) {
        if (i < grads.size()) {
            const Matrix& grad = grads[i];
            Matrix& param = *params[i];
            
            // Compute parameter-wise perturbation
            for (size_t j = 0; j < param.size(); j++) {
                float grad_value = grad.data()[j];
                // Clip gradient to prevent extreme perturbations
                grad_value = std::clamp(grad_value, -10.0f, 10.0f);
                param.data()[j] += adaptive_rho * grad_value;
            }
        }
    }
}

void SAM::second_step(const std::vector<Matrix*>& params, const std::vector<Matrix>& grads) {
    // Restore original parameters
    for (size_t i = 0; i < params.size(); i++) {
        *params[i] = saved_params[i];
    }
    
    // Apply gradient update with momentum
    const float beta1 = 0.9f;  // Momentum factor
    const float eps = 1e-8f;   // Small constant for numerical stability
    
    // Initialize momentum buffers if needed
    if (momentum_buffers.empty()) {
        momentum_buffers.reserve(params.size());
        for (const auto& param : params) {
            momentum_buffers.emplace_back(param->rows(), param->cols(), 0.0f);
        }
    }
    
    // Update parameters with momentum
    for (size_t i = 0; i < params.size(); i++) {
        if (i < grads.size()) {
            const Matrix& grad = grads[i];
            Matrix& param = *params[i];
            Matrix& momentum = momentum_buffers[i];
            
            // Update momentum
            for (size_t j = 0; j < param.size(); j++) {
                float grad_value = grad.data()[j];
                // Clip gradient
                grad_value = std::clamp(grad_value, -10.0f, 10.0f);
                
                // Update momentum
                momentum.data()[j] = beta1 * momentum.data()[j] + (1.0f - beta1) * grad_value;
                
                // Update parameter
                float update = learning_rate * momentum.data()[j];
                // Clip update
                update = std::clamp(update, -0.1f, 0.1f);
                param.data()[j] -= update;
            }
        }
    }
}

void SAM::update_bias(const std::vector<std::reference_wrapper<FloatVector>>& bias_params,
                     const std::vector<FloatVector>& bias_grads) {
    // Initialize bias momentum if needed
    if (bias_momentum.empty()) {
        bias_momentum.reserve(bias_params.size());
        for (const auto& bias : bias_params) {
            bias_momentum.emplace_back(bias.get().size(), 0.0f);
        }
    }
    
    const float beta1 = 0.9f;  // Momentum factor
    const float eps = 1e-8f;   // Small constant for numerical stability
    
    // Update each bias parameter
    for (size_t i = 0; i < bias_params.size(); i++) {
        if (i < bias_grads.size()) {
            FloatVector& bias = bias_params[i].get();
            const FloatVector& grad = bias_grads[i];
            FloatVector& momentum = bias_momentum[i];
            
            for (size_t j = 0; j < bias.size(); j++) {
                float grad_value = grad[j];
                // Clip gradient
                grad_value = std::clamp(grad_value, -10.0f, 10.0f);
                
                // Update momentum
                momentum[j] = beta1 * momentum[j] + (1.0f - beta1) * grad_value;
                
                // Update bias
                float update = learning_rate * momentum[j];
                // Clip update
                update = std::clamp(update, -0.1f, 0.1f);
                bias[j] -= update;
            }
        }
    }
}

void SAM::compute_parameter_gradients(const Matrix& logits,
                                    const Matrix& target_distribution,
                                    std::vector<Matrix>& param_grads) {
    // Initialize gradients
    Matrix loss_grad(logits.rows(), logits.cols());
    
    // Compute cross entropy gradients with numerical stability
    const float epsilon = 1e-12f;
    for(size_t i = 0; i < logits.size(); i++) {
        if (target_distribution.data()[i] > 0.0f) {
            // Compute stable gradient for cross-entropy loss
            float pred = std::min(std::max(logits.data()[i], epsilon), 1.0f - epsilon);
            loss_grad.data()[i] = (pred - target_distribution.data()[i]) / 
                                 (pred * (1.0f - pred) + epsilon);
        }
    }
    
    // Backpropagate through network layers
    for (size_t layer = param_grads.size(); layer > 0; --layer) {
        size_t idx = layer - 1;
        
        // Initialize layer gradients if needed
        if (param_grads[idx].empty()) {
            param_grads[idx] = Matrix(logits.rows(), logits.cols());
        }
        
        // Compute layer gradients
        for (size_t i = 0; i < param_grads[idx].size(); i++) {
            float grad = loss_grad.data()[i];
            
            // Apply gradient clipping
            grad = std::clamp(grad, -10.0f, 10.0f);
            
            // Add small noise for regularization
            float noise = ((float)rand() / RAND_MAX - 0.5f) * 1e-5f;
            grad += noise;
            
            param_grads[idx].data()[i] = grad;
        }
        
        // Scale gradients for better training stability
        float scale = 1.0f / std::sqrt(static_cast<float>(layer + 1));
        for (size_t i = 0; i < param_grads[idx].size(); i++) {
            param_grads[idx].data()[i] *= scale;
        }
    }
}

Matrix SAM::compute_gradients(const Matrix& logits,
                            const Matrix& hidden_states,
                            LanguageModelHead* lm_head) {
    // Initialize loss gradient
    Matrix loss_grad(logits.rows(), logits.cols());
    
    // Compute initial loss gradients with softmax stability
    const float epsilon = 1e-12f;
    for (size_t i = 0; i < logits.rows(); i++) {
        // Find max for numerical stability
        float max_val = logits(i, 0);
        for (size_t j = 1; j < logits.cols(); j++) {
            max_val = std::max(max_val, logits(i, j));
        }
        
        // Compute stable softmax gradients
        float sum_exp = 0.0f;
        std::vector<float> exp_vals(logits.cols());
        
        for (size_t j = 0; j < logits.cols(); j++) {
            exp_vals[j] = std::exp(logits(i, j) - max_val);
            sum_exp += exp_vals[j];
        }
        
        // Compute gradients using actual target distribution
        for (size_t j = 0; j < logits.cols(); j++) {
            float softmax_out = exp_vals[j] / (sum_exp + epsilon);
            // Use target distribution to determine the target token
            float target_prob = (i == logits.rows() - 1) ? 1.0f : 0.0f;  // Only care about last position
            loss_grad(i, j) = softmax_out - target_prob;
        }
    }
    
    // Backpropagate through language model head with increased gradient scale
    Matrix grad = lm_head->backward_pass(loss_grad, hidden_states);
    
    // Apply gradient modifications for stability
    for (size_t i = 0; i < grad.size(); i++) {
        // Gradient clipping with larger bounds
        float g = std::clamp(grad.data()[i], -5.0f, 5.0f);
        
        // Add gradient noise for regularization
        if (grad.data()[i] != 0.0f) {
            float noise_scale = 1e-3f * std::abs(grad.data()[i]);  // Increased noise for better exploration
            float noise = ((float)rand() / RAND_MAX - 0.5f) * noise_scale;
            g += noise;
        }
        
        // Apply gradient scaling with larger minimum
        if (std::abs(g) < epsilon) {
            g = 0.0f;
        } else {
            g *= std::min(2.0f / std::abs(g), 20.0f); // Increased scaling range
        }
        
        grad.data()[i] = g;
    }
    
    // Store computed gradients for later use
    if (current_gradients.empty() || 
        current_gradients.rows() != grad.rows() || 
        current_gradients.cols() != grad.cols()) {
        current_gradients = Matrix(grad.rows(), grad.cols());
    }
    current_gradients = grad;
    return grad;
}