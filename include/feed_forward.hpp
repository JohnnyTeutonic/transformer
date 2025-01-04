#pragma once
#include "components.hpp"
#ifdef USE_CUDA
#include "cuda/cuda_utils.cuh"
#endif

using FloatVector = Vector;

class FeedForward {
private:
  Matrix w1;
  Matrix w2;
  Vector b1;
  Vector b2;
  float dropout_prob;
  Matrix intermediate_cache;
  Matrix dropout_mask_cache;  // Cache for dropout mask
  
  // Gradient storage
  Matrix w1_grad;
  Matrix w2_grad;
  Vector b1_grad;
  Vector b2_grad;

public:
  virtual ~FeedForward() = default;
  FeedForward() = default;
  FeedForward(size_t hidden_size, size_t intermediate_size, float dropout = 0.1f);  // Declaration only
  Matrix forward(const Matrix &x);
  Matrix backward(const Matrix &grad_output, const Matrix &input);
  Matrix backward_cuda(const Matrix& grad_output, const Matrix& grad_hidden);
  void save(std::ostream &os) const;
  static std::unique_ptr<FeedForward> load(std::istream &is);
  friend class Transformer;

  // Getter for gradients
  std::vector<std::reference_wrapper<Matrix>> get_weight_gradients() {
    return {std::ref(w1_grad), std::ref(w2_grad)};
  }
  
  std::vector<std::reference_wrapper<Vector>> get_bias_gradients() {
    return {std::ref(b1_grad), std::ref(b2_grad)};
  }

  std::vector<std::reference_wrapper<Matrix>> get_weights() {
    return {std::ref(w1), std::ref(w2)};
  }

  friend class TransformerLayer;

  FloatVector &getBias1() { return b1; }
  FloatVector &getBias2() { return b2; }

  FeedForward(const FeedForward &other)
      : w1(other.w1), w2(other.w2), b1(other.b1), b2(other.b2),
        dropout_prob(other.dropout_prob),
        intermediate_cache(other.intermediate_cache),
        dropout_mask_cache(other.dropout_mask_cache),
        w1_grad(other.w1_grad), w2_grad(other.w2_grad),
        b1_grad(other.b1_grad), b2_grad(other.b2_grad) {}

  FeedForward &operator=(const FeedForward &other) {
    if (this != &other) {
      w1 = other.w1;
      w2 = other.w2;
      b1 = other.b1;
      b2 = other.b2;
      dropout_prob = other.dropout_prob;
      intermediate_cache = other.intermediate_cache;
      dropout_mask_cache = other.dropout_mask_cache;
      w1_grad = other.w1_grad;
      w2_grad = other.w2_grad;
      b1_grad = other.b1_grad;
      b2_grad = other.b2_grad;
    }
    return *this;
  }
};