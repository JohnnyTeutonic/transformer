#pragma once
#include "../components.hpp"
#include <memory>
#include <unordered_map>
#include <vector>

// Base optimizer class
class Optimizer {
protected:
  std::vector<Matrix *> parameters;
  std::vector<Matrix> gradients;
  float learning_rate;
  float beta1;
  float beta2;
  float epsilon;
  size_t t; // timestep

public:
  Optimizer(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f)
      : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), t(0) {}
  
  virtual ~Optimizer() = default;
  virtual void step(std::vector<Matrix *> &params, const std::vector<Matrix> &grads) = 0;
  virtual void zero_grad() = 0;
  virtual void update(const std::vector<Matrix> &params, const std::vector<Matrix> &grads) = 0;
  virtual void save(std::ostream &os) const = 0;
  virtual void load(std::istream &is) = 0;
  
  void add_parameter(Matrix &param) {
    parameters.push_back(&param);
    gradients.push_back(Matrix(param.rows(), param.cols(), 0.0f));
  }
};

// Basic SGD Optimizer
class SGD : public Optimizer {
private:
  float momentum;
  std::vector<Matrix> velocity;

public:
  explicit SGD(float lr = 0.001f, float momentum = 0.9f)
      : Optimizer(lr), momentum(momentum) {}

  void step(std::vector<Matrix *> &params,
            const std::vector<Matrix> &grads) override {
    if (velocity.empty()) {
      // Initialize velocity vectors
      for (const auto &grad : grads) {
        velocity.push_back(Matrix(grad.rows(), grad.cols(), 0.0f));
      }
    }

    for (size_t i = 0; i < params.size(); ++i) {
      // Update velocity
      for (size_t r = 0; r < velocity[i].rows(); ++r) {
        for (size_t c = 0; c < velocity[i].cols(); ++c) {
          velocity[i](r, c) =
              momentum * velocity[i](r, c) + learning_rate * grads[i](r, c);
        }
      }

      // Update parameters
      for (size_t r = 0; r < params[i]->rows(); ++r) {
        for (size_t c = 0; c < params[i]->cols(); ++c) {
          (*params[i])(r, c) -= velocity[i](r, c);
        }
      }
    }
  }

  void zero_grad() override { 
    velocity.clear(); 
    t = 0;
  }

  void update(const std::vector<Matrix> &params, const std::vector<Matrix> &grads) override {
    t++;
    std::vector<Matrix*> param_ptrs;
    for (size_t i = 0; i < params.size(); ++i) {
      param_ptrs.push_back(const_cast<Matrix*>(&params[i]));
    }
    step(param_ptrs, grads);
  }

  void save(std::ostream &os) const override {
    // Save optimizer state
    os.write(reinterpret_cast<const char*>(&learning_rate), sizeof(learning_rate));
    os.write(reinterpret_cast<const char*>(&momentum), sizeof(momentum));
    os.write(reinterpret_cast<const char*>(&t), sizeof(t));
    
    // Save velocity matrices
    size_t num_velocities = velocity.size();
    os.write(reinterpret_cast<const char*>(&num_velocities), sizeof(num_velocities));
    for (const auto& v : velocity) {
      v.save(os);
    }
  }

  void load(std::istream &is) override {
    // Load optimizer state
    is.read(reinterpret_cast<char*>(&learning_rate), sizeof(learning_rate));
    is.read(reinterpret_cast<char*>(&momentum), sizeof(momentum));
    is.read(reinterpret_cast<char*>(&t), sizeof(t));
    
    // Load velocity matrices
    size_t num_velocities;
    is.read(reinterpret_cast<char*>(&num_velocities), sizeof(num_velocities));
    velocity.clear();
    velocity.reserve(num_velocities);
    for (size_t i = 0; i < num_velocities; ++i) {
      velocity.push_back(Matrix::load(is));
    }
  }
};