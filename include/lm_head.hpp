#pragma once
#include "components.hpp"
#include "optimizer/sam.hpp"
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

class LanguageModelHead {
private:
  Matrix projection;
  Vector bias;
  float dropout_prob;
  size_t vocab_size_;
  size_t hidden_size_;
  Matrix hidden_states;
  void backward_linear(const Matrix& grad_output);
  Matrix forward_impl(const Matrix &hidden_states) const;

public:
  LanguageModelHead(size_t hidden_size, size_t vocab_size, float dropout = 0.1)
      : projection(Matrix(vocab_size, hidden_size)), bias(Vector(vocab_size)),
        dropout_prob(dropout), vocab_size_(vocab_size),
        hidden_size_(hidden_size) {
    // Xavier/Glorot initialization for better scaling
    float scale = std::sqrt(6.0f / (hidden_size + vocab_size));
    std::cout << "LM Head initialization:" << std::endl;
    std::cout << "Creating projection matrix: [" << vocab_size << " × "
              << hidden_size << "] with scale " << scale << std::endl;
    projection.randomize(-scale, scale);
    // Initialize bias to zero
    for (size_t i = 0; i < vocab_size; ++i) {
        bias[i] = 0.0f;
    }
  }

  LanguageModelHead(const LanguageModelHead &other)
      : projection(other.projection), bias(other.bias),
        dropout_prob(other.dropout_prob) {}

  LanguageModelHead &operator=(const LanguageModelHead &other) {
    if (this != &other) {
      projection = other.projection;
      bias = other.bias;
      dropout_prob = other.dropout_prob;
    }
    return *this;
  }

  Matrix forward(const Matrix &hidden_states) {
    this->hidden_states = hidden_states;
    return forward_impl(hidden_states);
  }

  Matrix backward_pass(const Matrix &grad_output, const Matrix &hidden_states) {
    // Compute gradients for projection and bias
    std::cout << "Computing gradients for projection and bias" << std::endl;
    Matrix grad_proj = matmul(grad_output.transpose(), hidden_states);
    std::cout << "grad projection shape: " << grad_proj.shape() << std::endl;
    Vector grad_bias = grad_output.row_sum();
    std::cout << "grad bias size: " << grad_bias.size() << std::endl;

    // Create vector of parameters and gradients for SAM
    std::vector<Matrix*> params = {&projection};
    std::vector<Matrix> grads = {grad_proj};
    
    // Create SAM optimizer if not already created
    static std::unique_ptr<SAM> sam_optimizer = std::make_unique<SAM>();
    
    // First step of SAM
    sam_optimizer->first_step(params, grads);
    
    // Compute gradients at perturbed point
    Matrix perturbed_logits = forward_impl(hidden_states);
    sam_optimizer->compute_parameter_gradients(perturbed_logits, grad_output, grads);
    Matrix perturbed_grad_proj = matmul(grad_output.transpose(), hidden_states);
    
    // Second step of SAM
    std::vector<Matrix> perturbed_grads = {perturbed_grad_proj};
    sam_optimizer->second_step(params, perturbed_grads);
    
    // Update bias separately using a simpler update rule
    std::vector<std::reference_wrapper<FloatVector>> bias_params = {std::ref(bias)};
    std::vector<FloatVector> bias_grads = {grad_bias};
    sam_optimizer->update_bias(bias_params, bias_grads);

    // Compute gradient with respect to input
    Matrix grad_input = matmul(grad_output, projection);
    if (grad_input.cols() != hidden_states.cols()) {
      throw std::runtime_error(
          "Language model head gradient output dimension (" +
          std::to_string(grad_input.cols()) + ") must match hidden size (" +
          std::to_string(hidden_states.cols()) + ")");
    }
    return grad_input;
  }

  void save(std::ostream &os) const {
    projection.save(os);
    bias.save(os);
    os.write(reinterpret_cast<const char *>(&dropout_prob),
             sizeof(dropout_prob));
  }

  static std::unique_ptr<LanguageModelHead> load(std::istream &is) {
    auto lm_head = std::make_unique<LanguageModelHead>(0, 0); // Temporary sizes
    lm_head->projection = Matrix::load(is);
    lm_head->bias = Vector::load(is);
    is.read(reinterpret_cast<char *>(&lm_head->dropout_prob),
            sizeof(lm_head->dropout_prob));
    return lm_head;
  }

  std::vector<std::reference_wrapper<Matrix>> get_parameters() {
    std::vector<std::reference_wrapper<Matrix>> params;
    params.push_back(std::ref(projection));
    // Note: We'll need to handle bias separately since it's a Vector
    return params;
  }

  Vector &get_bias() { return bias; }

  Matrix project_to_vocab(const Matrix &hidden_states) const;

  const Matrix &get_projection() const { return projection; }

  void backward(const Matrix& grad_output, const Matrix& target_distribution);
};