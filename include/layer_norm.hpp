#pragma once
#include "matrix.hpp"
#include "lora.hpp"
#include <cmath>

/**
 * @brief Layer Normalization implementation for neural networks.
 * 
 * Layer Normalization normalizes the inputs across the features, applying a learnable
 * scale (gamma) and shift (beta) parameter. The normalization is computed as:
 * y = ((x - mean) / sqrt(variance + eps)) * gamma + beta
 * 
 * Features:
 * - Per-layer normalization
 * - Learnable scale and shift parameters
 * - CUDA acceleration support
 * - Gradient computation for training
 */
class LayerNorm {
public:
    /**
     * @brief Constructs a layer normalization module.
     * @param hidden_size_ Size of the input features
     * @param eps_ Small constant for numerical stability (default: 1e-5)
     * @param rms_ RMSNorm mode (LLaMA-compatible): no mean subtraction, no beta
     */
    LayerNorm(size_t hidden_size_, float eps_ = 1e-5, bool rms_ = false);

    bool is_rms() const { return rms_mode_; }

    /**
     * @brief Performs the forward pass of layer normalization.
     * @param input Input tensor of shape [batch_size, hidden_size]
     * @return Normalized tensor of the same shape
     */
    Matrix forward(const Matrix& input);

    /**
     * @brief Performs the backward pass to compute gradients.
     * @param grad_output Gradient of the loss with respect to the output
     * @param input Original input tensor
     * @return Gradient with respect to the input
     */
    Matrix backward(const Matrix& grad_output, const Matrix& input) {
        input_cache_ = input;  // Cache input for backward pass
        return compute_gradients(grad_output);
    }

    /**
     * @brief Gets references to all learnable parameters.
     * @return Vector of references to parameter vectors
     */
    const std::vector<std::reference_wrapper<Matrix>> get_parameter_list() {
        std::vector<std::reference_wrapper<Matrix>> params;
        params.push_back(std::ref(params_.gamma));
        params.push_back(std::ref(params_.beta));
        return params;
    }

    /**
     * @brief Gets references to all parameter gradients.
     * @return Vector of references to gradient vectors
     */
    const std::vector<std::reference_wrapper<const Matrix>> parameter_gradients() const {
        std::vector<std::reference_wrapper<const Matrix>> grads;
        grads.push_back(std::cref(grads_.gamma_grad));
        grads.push_back(std::cref(grads_.beta_grad));
        return grads;
    }

    /**
     * @brief Saves the layer parameters to a stream.
     * @param os Output stream to save to
     */
    void save(std::ostream& os) const;

    /**
     * @brief Loads layer parameters from a stream.
     * @param is Input stream to load from
     * @return Unique pointer to the loaded layer
     */
    static std::unique_ptr<LayerNorm> load(std::istream& is);

    /**
     * @brief Gets the size of the input features.
     * @return Hidden size
     */
    size_t get_hidden_size() const {
        return hidden_size_;
    }

    /**
     * @brief Gets the epsilon value.
     * @return Epsilon constant
     */
    float get_eps() const {
        return eps_;
    }

    /**
     * @brief Copy constructor.
     * @param other LayerNorm instance to copy from
     */
    LayerNorm(const LayerNorm& other)
        : hidden_size_(other.hidden_size_), eps_(other.eps_), rms_mode_(other.rms_mode_),
          params_(other.params_),
          input_cache_(other.input_cache_), output_cache_(other.output_cache_),
          grads_(other.grads_) {}

    /**
     * @brief Assignment operator.
     * @param other LayerNorm instance to assign from
     * @return Reference to this instance
     */
    LayerNorm& operator=(const LayerNorm& other) {
        if (this != &other) {
            hidden_size_ = other.hidden_size_;
            eps_ = other.eps_;
            rms_mode_ = other.rms_mode_;
            params_ = other.params_;
            input_cache_ = other.input_cache_;
            output_cache_ = other.output_cache_;
            grads_ = other.grads_;
        }
        return *this;
    }

    Matrix get_combined_gradients() const {
        // Create a matrix that combines both gradients
        Matrix combined(1, hidden_size_ * 2);
        std::copy(grads_.gamma_grad.data(), grads_.gamma_grad.data() + hidden_size_, combined.data());
        std::copy(grads_.beta_grad.data(), grads_.beta_grad.data() + hidden_size_, combined.data() + hidden_size_);
        return combined;
    }

    // Mutable accessors
    Matrix& get_gamma_mut() { return params_.gamma; }
    Matrix& get_beta_mut() { return params_.beta; }

    // Const accessors
    const Matrix& get_gamma() const { return params_.gamma; }
    const Matrix& get_beta() const { return params_.beta; }
    
    // GGUF export accessor (returns gamma as Vector for 1D export)
    Vector getGamma() const {
        Vector v(hidden_size_);
        std::copy(params_.gamma.data(), params_.gamma.data() + hidden_size_, v.data());
        return v;
    }

    // Parameter structure to hold gamma and beta
    struct Parameters {
        Matrix gamma;  // Scale parameter
        Matrix beta;   // Shift parameter
    };

    // Gradient structure to hold gradients
    struct Gradients {
        Matrix gamma_grad;  // Gradient for gamma
        Matrix beta_grad;   // Gradient for beta
    };
    
    // Adam optimizer state (first + second moments; see attention.hpp note)
    struct MomentumState {
        Matrix gamma_m;
        Matrix beta_m;
        Matrix gamma_v;
        Matrix beta_v;
        size_t t = 0;
        bool initialized = false;
    };

    // Parameter accessors
    Parameters& parameters() { return params_; }
    Gradients& param_gradients() { return grads_; }
    const Parameters& parameters() const { return params_; }
    const Gradients& param_gradients() const { return grads_; }

    void update_parameters(float learning_rate) {
        // LoRA fine-tuning freezes everything except the adapters.
        if (lora::settings().enabled) return;
        // Adam (matches the LM head's optimizer; see attention.cpp note)
        const float beta = 0.9f;
        const float beta2 = 0.999f;
        const float adam_eps = 1e-8f;
        const float clip_threshold = 1.0f;

        // Initialize optimizer state on first call
        if (!momentum_.initialized) {
            momentum_.gamma_m = Matrix(1, hidden_size_, 0.0f);
            momentum_.beta_m = Matrix(1, hidden_size_, 0.0f);
            momentum_.gamma_v = Matrix(1, hidden_size_, 0.0f);
            momentum_.beta_v = Matrix(1, hidden_size_, 0.0f);
            momentum_.initialized = true;
        }
        momentum_.t += 1;
        const float bc1 = 1.0f - std::pow(beta, static_cast<float>(momentum_.t));
        const float bc2 = 1.0f - std::pow(beta2, static_cast<float>(momentum_.t));

        auto adam_vec = [&](float* param, float* grad, float* m, float* v) {
            float norm_sq = 0.0f;
            for (size_t i = 0; i < hidden_size_; ++i) {
                norm_sq += grad[i] * grad[i];
            }
            float norm = std::sqrt(norm_sq);
            float scale = (norm > clip_threshold) ? (clip_threshold / (norm + 1e-8f)) : 1.0f;
            for (size_t i = 0; i < hidden_size_; ++i) {
                float g = grad[i] * scale;
                m[i] = beta * m[i] + (1.0f - beta) * g;
                v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
                param[i] -= learning_rate * (m[i] / bc1) / (std::sqrt(v[i] / bc2) + adam_eps);
                grad[i] = 0.0f;
            }
        };

        adam_vec(params_.gamma.data(), grads_.gamma_grad.data(),
                 momentum_.gamma_m.data(), momentum_.gamma_v.data());

        // RMSNorm mode has no beta parameter - keep it frozen at zero
        if (rms_mode_) {
            return;
        }
        adam_vec(params_.beta.data(), grads_.beta_grad.data(),
                 momentum_.beta_m.data(), momentum_.beta_v.data());
    }

private:
    size_t hidden_size_;
    float eps_;
    bool rms_mode_ = false;  // RMSNorm (LLaMA) mode: no mean subtraction, no beta
    Parameters params_;
    Matrix input_cache_;  // Stored for backward pass
    Matrix output_cache_; // Stored for backward pass
    Gradients grads_;
    MomentumState momentum_;  // For stable training

    // Helper method to compute gradients
    Matrix compute_gradients(const Matrix& grad_output);
};