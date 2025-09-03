#include "../include/transformer.hpp"
#include "../include/scope_logger.hpp"
#ifdef USE_CUDA
#include "../include/cuda/cublas_check.cuh"
#include "../include/cuda/cuda_check.cuh"
#include <cublas_v2.h>
#endif
#include "../include/logger.hpp"
#include "../include/half_precision.hpp"
#include "../include/training/dynamic_loss_scaler.hpp"
#include "../include/utils.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <omp.h>
#include <set>
#include <stdexcept>
#include <nlohmann/json.hpp>

#ifdef USE_CUDA
extern cublasHandle_t cublas_handle;
#endif

// Global variable to store the last computed gradient norm
float last_grad_norm = 0.0f;

// LearningRateScheduler implementation
class LearningRateScheduler {
public:
    LearningRateScheduler(float initial_lr = 0.00001f,  // Reduced initial learning rate
                         float peak_lr = 0.0001f,       // Reduced peak learning rate
                         int warmup_steps = 2000,      // More warmup steps
                         float decay_factor = 0.95f)   // Faster decay
        : initial_lr_(initial_lr), peak_lr_(peak_lr), 
          warmup_steps_(warmup_steps), decay_factor_(decay_factor),
          min_lr_(initial_lr * 0.1f)  // Add minimum learning rate
    {
        std::cout << "LR Schedule: initial=" << initial_lr 
                  << ", peak=" << peak_lr 
                  << ", warmup_steps=" << warmup_steps 
                  << ", min_lr=" << min_lr_ << std::endl;
    }

    float get_lr(int step) {
        if (step < warmup_steps_) {
            // Smoother warmup using quadratic growth
            float progress = static_cast<float>(step) / warmup_steps_;
            float warmup_factor = progress * progress;  // Quadratic growth
            return initial_lr_ + (peak_lr_ - initial_lr_) * warmup_factor;
        }
        
        // Cosine decay after warmup with minimum learning rate
        float progress = static_cast<float>(step - warmup_steps_) / 1000.0f;
        progress = std::min(1.0f, progress);
        float decay = 0.5f * (1.0f + std::cos(progress * M_PI));
        float lr = peak_lr_ * decay * std::pow(decay_factor_, (step - warmup_steps_) / 1000.0f);
        
        // Ensure learning rate doesn't go below minimum
        return std::max(lr, min_lr_);
    }

    void set_lr(float new_lr) {
        // Maintain ratio between initial and peak
        float ratio = peak_lr_ / initial_lr_;
        initial_lr_ = new_lr;
        peak_lr_ = new_lr * ratio;
        min_lr_ = initial_lr_ * 0.1f;
        
        std::cout << "LR Schedule: updated initial=" << initial_lr_ 
                  << ", peak=" << peak_lr_
                  << ", min=" << min_lr_ << std::endl;
    }

private:
    float initial_lr_;
    float peak_lr_;
    int warmup_steps_;
    float decay_factor_;
    float min_lr_;  // Add minimum learning rate
};

// TransformerLayer implementation
TransformerLayer::TransformerLayer(const TransformerConfig& config_, size_t idx)
    : config(config_), layer_idx(idx) {
    // Initialize components
    std::cout << "Initializing TransformerLayer " << idx << " with GQA config:" << std::endl;
    std::cout << "- use_gqa: " << (config.use_gqa ? "true" : "false") << std::endl;
    std::cout << "- num_kv_heads: " << config.num_kv_heads << std::endl;
    std::cout << "- hidden_size: " << config.hidden_size << std::endl;
    std::cout << "- intermediate_size: " << config.intermediate_size << std::endl;

    self_attention = std::make_unique<MultiHeadAttention>(
        config.hidden_size, config.num_heads, config.head_dim, config.dropout_rate,
        config.use_flash_attention, config.use_rope, config.use_sliding_window, config.window_size,
        config.use_gqa, config.num_kv_heads, config.max_seq_length, config.use_fp16);

    attention_ln = std::make_unique<LayerNorm>(config.hidden_size);
    
    // Conditionally initialize FeedForward or Mixture of Experts layer
    if (config.moe.enabled) {
        moe_layer = std::make_unique<MixtureOfExperts>(
            config.moe.num_experts, config.moe.top_k, config.hidden_size, config.intermediate_size, config.moe.aux_loss_coefficient);
        feed_forward = nullptr;
    } else {
        feed_forward = std::make_unique<FeedForward>(config.hidden_size, config.intermediate_size);
        moe_layer = nullptr;
    }

    ffn_ln = std::make_unique<LayerNorm>(config.hidden_size);

    // Initialize dropout layers
    attention_dropout = std::make_unique<Dropout>(config.dropout_rate);
    ffn_dropout = std::make_unique<Dropout>(config.dropout_rate);
}

Matrix TransformerLayer::forward(const Matrix& input, const AttentionMask& mask,
                                 const std::optional<KVCache>& kv_cache) {
    std::cout << "=== TransformerLayer::forward START ===" << std::endl;

    // Layer norm before attention
    Matrix normalized = attention_ln->forward(input);
    
    // Debug normalized output and validate statistics
    float min_norm = std::numeric_limits<float>::infinity();
    float max_norm = -std::numeric_limits<float>::infinity();
    float sum_norm = 0.0f;
    float sum_squared = 0.0f;
    size_t nonzero_norm = 0;
    const size_t total_elements = normalized.rows() * normalized.cols();
    
    #pragma omp parallel for collapse(2) reduction(min:min_norm) reduction(max:max_norm) \
                             reduction(+:sum_norm,sum_squared,nonzero_norm)
    for (size_t i = 0; i < normalized.rows(); i++) {
        for (size_t j = 0; j < normalized.cols(); j++) {
            float val = normalized(i, j);
            min_norm = std::min(min_norm, val);
            max_norm = std::max(max_norm, val);
            sum_norm += val;
            sum_squared += val * val;
            if (std::abs(val) > 1e-6) nonzero_norm++;
        }
    }
    
    float mean = sum_norm / total_elements;
    float variance = (sum_squared / total_elements) - (mean * mean);
    
    // Check for layer norm instability
    const float STABILITY_THRESHOLD = 1e3;
    if (std::abs(mean) > 1e-2 || std::abs(variance - 1.0) > 1e-1 || 
        std::abs(min_norm) > STABILITY_THRESHOLD || std::abs(max_norm) > STABILITY_THRESHOLD) {
        std::cerr << "WARNING: Layer normalization statistics outside expected ranges:\n"
                  << "Mean: " << mean << " (expected close to 0)\n"
                  << "Variance: " << variance << " (expected close to 1)\n"
                  << "Min: " << min_norm << "\n"
                  << "Max: " << max_norm << "\n";
                  
        // Clip extreme values if needed
        if (std::abs(min_norm) > STABILITY_THRESHOLD || std::abs(max_norm) > STABILITY_THRESHOLD) {
            for (size_t i = 0; i < normalized.rows(); i++) {
                for (size_t j = 0; j < normalized.cols(); j++) {
                    normalized(i, j) = std::max(-STABILITY_THRESHOLD, 
                                              std::min(STABILITY_THRESHOLD, normalized(i, j)));
                }
            }
            std::cerr << "Applied value clipping for stability\n";
        }
    }
    
    std::cout << "After attention layer norm:\n"
              << "Min norm: " << min_norm << "\n"
              << "Max norm: " << max_norm << "\n"
              << "Mean norm: " << mean << "\n"
              << "Variance: " << variance << "\n"
              << "Nonzero norm: " << nonzero_norm << "/" << total_elements << "\n\n";

    // Cache the normalized input for attention backward pass
    std::string attn_key = "attn_norm_" + std::to_string(layer_idx);
    GradientCheckpoint::cache_activation(attn_key, normalized);

    // Self attention
    Matrix attention_output = self_attention->forward(normalized, mask, kv_cache);
    
    // Debug attention output
    float min_attn = std::numeric_limits<float>::infinity();
    float max_attn = -std::numeric_limits<float>::infinity();
    float sum_attn = 0.0f;
    size_t nonzero_attn = 0;
    
    #pragma omp parallel for collapse(2) reduction(min:min_attn) reduction(max:max_attn) \
                             reduction(+:sum_attn,nonzero_attn)
    for (size_t i = 0; i < attention_output.rows(); i++) {
        for (size_t j = 0; j < attention_output.cols(); j++) {
            float val = attention_output(i, j);
            min_attn = std::min(min_attn, val);
            max_attn = std::max(max_attn, val);
            sum_attn += val;
            if (std::abs(val) > 1e-6) nonzero_attn++;
        }
    }
    
    std::cout << "After self attention:\n"
              << "Min attn: " << min_attn << "\n"
              << "Max attn: " << max_attn << "\n"
              << "Mean attn: " << sum_attn / (attention_output.rows() * attention_output.cols()) << "\n"
              << "Nonzero attn: " << nonzero_attn << "/" 
              << (attention_output.rows() * attention_output.cols()) << "\n\n";
    
    // Apply attention dropout if in training mode
    if (training && attention_dropout) {
        attention_dropout->set_training(true);
        attention_output = attention_dropout->forward(attention_output);
    }
    Matrix residual = attention_output + normalized;
    
    // Debug residual
    float min_res = std::numeric_limits<float>::infinity();
    float max_res = -std::numeric_limits<float>::infinity();
    float sum_res = 0.0f;
    size_t nonzero_res = 0;
    
    for (size_t i = 0; i < residual.rows(); i++) {
        for (size_t j = 0; j < residual.cols(); j++) {
            float val = residual(i, j);
            min_res = std::min(min_res, val);
            max_res = std::max(max_res, val);
            sum_res += val;
            if (std::abs(val) > 1e-6) nonzero_res++;
        }
    }
    
    std::cout << "After residual connection:\n"
              << "Min res: " << min_res << "\n"
              << "Max res: " << max_res << "\n"
              << "Mean res: " << sum_res / (residual.rows() * residual.cols()) << "\n"
              << "Nonzero res: " << nonzero_res << "/" 
              << (residual.rows() * residual.cols()) << "\n\n";
    
    std::cout << "calculating attention ln" << std::endl;
    Matrix norm1 = attention_ln->forward(residual);
    
    // Debug norm1
    float min_norm1 = std::numeric_limits<float>::infinity();
    float max_norm1 = -std::numeric_limits<float>::infinity();
    float sum_norm1 = 0.0f;
    size_t nonzero_norm1 = 0;
    
    for (size_t i = 0; i < norm1.rows(); i++) {
        for (size_t j = 0; j < norm1.cols(); j++) {
            float val = norm1(i, j);
            min_norm1 = std::min(min_norm1, val);
            max_norm1 = std::max(max_norm1, val);
            sum_norm1 += val;
            if (std::abs(val) > 1e-6) nonzero_norm1++;
        }
    }
    
    std::cout << "After second attention layer norm:\n"
              << "Min norm1: " << min_norm1 << "\n"
              << "Max norm1: " << max_norm1 << "\n"
              << "Mean norm1: " << sum_norm1 / (norm1.rows() * norm1.cols()) << "\n"
              << "Nonzero norm1: " << nonzero_norm1 << "/" 
              << (norm1.rows() * norm1.cols()) << "\n\n";

    // Cache the normalized input for feed forward backward pass
    std::string ffn_key = "ffn_norm_" + std::to_string(layer_idx);
    GradientCheckpoint::cache_activation(ffn_key, norm1);
    std::cout << "Cached normalized input for feed forward: " << norm1.rows() << "x"
                  << norm1.cols() << std::endl;
    // Feed forward through MoE or FFN
    Matrix ff_output;
    if (moe_layer) {
        ff_output = moe_layer->forward(norm1);
    } else {
        ff_output = feed_forward->forward(norm1);
    }
    
    // Debug feed forward output
    float min_ff = std::numeric_limits<float>::infinity();
    float max_ff = -std::numeric_limits<float>::infinity();
    float sum_ff = 0.0f;
    size_t nonzero_ff = 0;
    
    for (size_t i = 0; i < ff_output.rows(); i++) {
        for (size_t j = 0; j < ff_output.cols(); j++) {
            float val = ff_output(i, j);
            min_ff = std::min(min_ff, val);
            max_ff = std::max(max_ff, val);
            sum_ff += val;
            if (std::abs(val) > 1e-6) nonzero_ff++;
        }
    }
    
    std::cout << "After feed forward:\n"
              << "Min ff: " << min_ff << "\n"
              << "Max ff: " << max_ff << "\n"
              << "Mean ff: " << sum_ff / (ff_output.rows() * ff_output.cols()) << "\n"
              << "Nonzero ff: " << nonzero_ff << "/" 
              << (ff_output.rows() * ff_output.cols()) << "\n\n";
    
    // Apply feed forward dropout if in training mode
    if (training && ffn_dropout) {
        ffn_dropout->set_training(true);
        ff_output = ffn_dropout->forward(ff_output);
    }
    residual = ff_output + norm1;
    
    // Debug final residual
    float min_final = std::numeric_limits<float>::infinity();
    float max_final = -std::numeric_limits<float>::infinity();
    float sum_final = 0.0f;
    size_t nonzero_final = 0;
    
    for (size_t i = 0; i < residual.rows(); i++) {
        for (size_t j = 0; j < residual.cols(); j++) {
            float val = residual(i, j);
            min_final = std::min(min_final, val);
            max_final = std::max(max_final, val);
            sum_final += val;
            if (std::abs(val) > 1e-6) nonzero_final++;
        }
    }
    
    std::cout << "After final residual:\n"
              << "Min final: " << min_final << "\n"
              << "Max final: " << max_final << "\n"
              << "Mean final: " << sum_final / (residual.rows() * residual.cols()) << "\n"
              << "Nonzero final: " << nonzero_final << "/" 
              << (residual.rows() * residual.cols()) << "\n\n";
    
    std::cout << "Residual dimensions: " << residual.rows() << "x" << residual.cols() << std::endl;
    return ffn_ln->forward(residual);
}

Matrix TransformerLayer::backward(const Matrix& grad_output, const Matrix& input,
                                  const Matrix& target_distribution) {
    std::cout << "=== TransformerLayer::backward START ===" << std::endl;

    try {
        // Ensure dimensions match input
        if (grad_output.cols() != input.cols()) {
            throw std::runtime_error("Gradient output columns (" + std::to_string(grad_output.cols()) + 
                                   ") must match input columns (" + std::to_string(input.cols()) + ")");
        }

        // Get the cached normalized input for feed forward
        std::string ffn_key = "ffn_norm_" + std::to_string(layer_idx);
        Matrix ffn_normalized = GradientCheckpoint::get_activation(ffn_key);

        // Backward through feed forward network with dimension validation
        Matrix ff_dropout_grad = training ? ffn_dropout->backward(grad_output) : grad_output;
        if (ff_dropout_grad.cols() != input.cols()) {
            throw std::runtime_error("FFN dropout gradient columns must match input columns");
        }

        Matrix ffn_grad;
        if (moe_layer) {
            ffn_grad = moe_layer->backward(ff_dropout_grad);
        } else {
            ffn_grad = feed_forward->backward(ff_dropout_grad, ffn_normalized);
        }
        
        if (ffn_grad.cols() != input.cols()) {
            throw std::runtime_error("FFN gradient columns must match input columns");
        }

        // Backward through feed forward layer norm
        Matrix ffn_ln_grad = ffn_ln->backward(ffn_grad, input);
        if (ffn_ln_grad.cols() != input.cols()) {
            throw std::runtime_error("FFN layer norm gradient columns must match input columns");
        }

        // First residual connection - proper gradient flow
        Matrix residual_grad = ffn_ln_grad + grad_output;  // Add both gradient paths
        // Get the cached normalized input for attention
        std::string attn_key = "attn_norm_" + std::to_string(layer_idx);
        Matrix attn_normalized = GradientCheckpoint::get_activation(attn_key);
        // Backward through self attention with dimension validation
        Matrix attn_dropout_grad = training ? attention_dropout->backward(residual_grad) : residual_grad;
        if (attn_dropout_grad.cols() != input.cols()) {
            throw std::runtime_error("Attention dropout gradient columns must match input columns");
        }

        // Project attention gradients to correct dimension before addition
        Matrix attention_grad = self_attention->backward(attn_dropout_grad, attn_normalized, target_distribution);
        if (attention_grad.cols() != input.cols()) {
            throw std::runtime_error("Attention gradient columns must match input columns");
        }

        // Ensure attention gradients have correct dimensions
        if (attention_grad.rows() != residual_grad.rows() || attention_grad.cols() != residual_grad.cols()) {
            throw std::runtime_error("Attention gradient dimensions (" + 
                std::to_string(attention_grad.rows()) + "x" + std::to_string(attention_grad.cols()) + 
                ") must match residual gradient dimensions (" +
                std::to_string(residual_grad.rows()) + "x" + std::to_string(residual_grad.cols()) + ")");
        }

        // Backward through attention layer norm
        Matrix attention_ln_grad = attention_ln->backward(attention_grad, input);
        if (attention_ln_grad.cols() != input.cols()) {
            throw std::runtime_error("Attention layer norm gradient columns must match input columns");
        }

        // Second residual connection - proper gradient flow
        Matrix final_grad = attention_ln_grad + residual_grad;  // Add both gradient paths
        std::cout << "=== TransformerLayer::backward END ===" << std::endl;
        return final_grad;

    } catch (const std::exception& e) {
        std::cerr << "Error in TransformerLayer::backward: " << e.what() << std::endl;
        throw;
    }
}

float TransformerLayer::getAuxLoss() const {
    if (moe_layer) {
        return moe_layer->get_aux_loss();
    }
    return 0.0f;
}

// Transformer implementation
Transformer::Transformer(const TransformerConfig& config_, std::shared_ptr<TiktokenTokenizer> tokenizer) 
    : config(config_), tokenizer_(tokenizer) {
    if (!tokenizer_) {
        throw std::runtime_error("Tokenizer cannot be null");
    }

    // Get and validate vocabulary size from tokenizer
    size_t vocab_size = tokenizer_->vocab_size();
    if (vocab_size == 0) {
        throw std::runtime_error("Tokenizer reports vocabulary size of 0");
    }
    
    // Update config with correct vocab size
    config.update_vocab_size(vocab_size);
    
    std::cout << "Initializing transformer with vocabulary size: " << vocab_size << std::endl;

    // Initialize embeddings with validated vocab size
    token_embedding = std::make_unique<TokenEmbedding>(vocab_size, config.hidden_size);
    if (!token_embedding) {
        throw std::runtime_error("Failed to create token embedding layer");
    }

    // Initialize positional encoding with correct hidden_size dimension
    pos_encoding = std::make_unique<PositionalEncoding>(config.max_seq_length, config.hidden_size);
    if (!pos_encoding) {
        throw std::runtime_error("Failed to create positional encoding layer");
    }
    
    // Create transformer layers
    layers.reserve(config.num_layers);
    for (size_t i = 0; i < config.num_layers; i++) {
        layers.push_back(TransformerLayer::create(config, i));
    }
    
    final_ln = std::make_unique<LayerNorm>(config.hidden_size, config.layer_norm_epsilon);
    lm_head = std::make_unique<LanguageModelHead>(config.hidden_size, vocab_size);
    dropout = std::make_unique<Dropout>(config.dropout_rate);
    
    // Initialize attention layers with tokenizer
    initialize_attention_layers();
    
    // Initialize weights
    initialize_weights();
}

// Add new BatchSequence structure to handle proper sequence boundaries
struct BatchSequence {
    Matrix embeddings;          // [batch_size x seq_len x hidden_size]
    Matrix attention_mask;      // [batch_size x seq_len x seq_len]
    std::vector<size_t> lengths;  // Original sequence lengths
};

TransformerOutput Transformer::forward(const std::vector<int>& input_tokens) {
    // ... (Keep the forward pass implementation, but change the return type and logic at the end)
    
    // ... inside the forward pass ...

    // Process through transformer layers
    float total_aux_loss = 0.0f;
    for (size_t i = 0; i < layers.size(); i++) {
        x = layers[i]->forward(x, mask);
        total_aux_loss += layers[i]->getAuxLoss();
        m_layer_activations.push_back(x);
    }

    // Final layer norm
    x = final_ln->forward(x);
    hidden_states = x;
    last_hidden_states = x;
    GradientCheckpoint::cache_activation("final_hidden_states", x);

    // Project to vocabulary space
    Matrix logits = lm_head->forward(x);

    return {logits, total_aux_loss};
}

void Transformer::clear_kv_cache() {
    SCOPE_LOG();
    for (auto& cache : m_kv_caches) {
        cache.clear();
    }
}

// Original backward method implementation
void Transformer::backward(const Matrix& grad_output, const std::vector<int>& input_tokens, float learning_rate) {
    std::cout << "Entering function 'Transformer::backward'" << std::endl;
    try {
        std::cout << "\nStarting backward pass..." << std::endl;
        std::cout << "Initial gradient dimensions: " << grad_output.rows() << "x" << grad_output.cols() << std::endl;
        
        // First, pass gradient through language model head
        Matrix hidden_states = GradientCheckpoint::get_activation("final_hidden_states");
        if (hidden_states.empty()) {
            throw std::runtime_error("No cached hidden states found for backward pass");
        }
        
        Matrix current_grad = lm_head->backward_pass(grad_output, hidden_states);
        
        const float global_max_grad_norm = config.gradient_clip_threshold;
        static DynamicLossScaler loss_scaler;
        
        // If using FP16, apply loss scaling
        if (config.use_fp16) {
            float scale = loss_scaler.get_scale();
            current_grad *= scale;
            std::cout << "Applied loss scale: " << scale << std::endl;
        }
        
        // Store layer gradients for global norm computation
        std::vector<Matrix> layer_gradients;
        layer_gradients.reserve(layers.size());
        
        bool has_inf_nan = false;
        float total_grad_norm = 0.0f;  // Track total gradient norm
        
        // Backward through layers with proper sequence handling
        for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i) {
            std::string layer_key = "layer_" + std::to_string(i) + "_input";
            Matrix layer_input = GradientCheckpoint::get_activation(layer_key);
            
            if (layer_input.empty()) {
                throw std::runtime_error("No cached activation found for layer " + std::to_string(i));
            }
            
            try {
                Matrix layer_grad = layers[i]->backward(current_grad, layer_input);
                if (!layer_grad.empty()) {
                    if (config.use_fp16 && loss_scaler.has_inf_or_nan(layer_grad)) {
                        has_inf_nan = true;
                        break;
                    }
                    
                    // Compute gradient norm for this layer
                    float layer_grad_norm = 0.0f;
                    for (size_t j = 0; j < layer_grad.size(); ++j) {
                        layer_grad_norm += layer_grad.data()[j] * layer_grad.data()[j];
                    }
                    total_grad_norm += std::sqrt(layer_grad_norm);
                    
                    layer_gradients.push_back(layer_grad);
                    current_grad = layer_grad;
                }
            } catch (const std::exception& e) {
                std::cerr << "Error in layer " << i << " backward pass: " << e.what() << std::endl;
                throw;
            }
        }
        
        // Store the total gradient norm in a static variable that can be accessed by the training monitor
        static float last_grad_norm = 0.0f;
        last_grad_norm = total_grad_norm / layers.size();  // Average across layers
        
        // Handle FP16 loss scaling
        if (config.use_fp16) {
            bool should_skip = !loss_scaler.update_scale(!has_inf_nan);
            if (should_skip) {
                std::cout << "Skipping step due to inf/nan in gradients" << std::endl;
                return;
            }
            
            // Unscale gradients
            float inv_scale = 1.0f / loss_scaler.get_scale();
            for (auto& grad : layer_gradients) {
                grad *= inv_scale;
            }
        }
        
        // Update parameters with proper sequence handling
        for (size_t i = 0; i < layers.size(); i++) {
            auto& layer = layers[i];
            if (auto* attention = layer->getAttention()) {
                update_attention_parameters(attention, learning_rate, config);
            }
            if (auto* moe = layer->getMixtureOfExperts()) {
                // Update experts
                for (auto* expert : moe->get_experts()) {
                    update_ffn_parameters(expert, learning_rate, config);
                }
                // Update router
                auto& router_params = moe->get_router().parameters();
                auto& router_grads = moe->get_router().gradients();
                update_parameter_with_clip(router_params.weights, router_grads.weights_grad, learning_rate, config);

            } else if (auto* ffn = layer->getFeedForward()) {
                 update_ffn_parameters(ffn, learning_rate, config);
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error in backward pass: " << e.what() << std::endl;
        throw;
    }
    std::cout << "Exiting function 'Transformer::backward'" << std::endl;
}

void Transformer::clear_gradients() {
    std::cout << "Entering function 'Transformer::clear_gradients'" << std::endl;
    try {
        // Reset layer gradients
        for (auto& layer : layers) {
            if (auto* attention = layer->getAttention()) {
                attention->param_gradients() = MultiHeadAttention::Gradients();
            }
            if (auto* moe = layer->getMixtureOfExperts()) {
                for (auto* expert : moe->get_experts()) {
                    expert->param_gradients() = FeedForward::Gradients();
                }
                moe->get_router().gradients().weights_grad.initialize_constant(0.0f);
            }
            if (auto* ffn = layer->getFeedForward()) {
                ffn->param_gradients() = FeedForward::Gradients();
            }
            if (auto* ln = layer->getLayerNorm()) {
                ln->param_gradients() = LayerNorm::Gradients();
            }
        }
        
        // Reset embedding gradients
        if (token_embedding) {
            token_embedding->get_gradient_table().initialize_constant(0.0f);
        }
        
        // Reset parameter gradients
        parameter_grads.reset();
    } catch (const std::exception& e) {
        std::cerr << "Error in Transformer::clear_gradients: " << e.what() << std::endl;
        throw;
    }
    std::cout << "Exiting function 'Transformer::clear_gradients'" << std::endl;
}

// New batch backward method implementation
void Transformer::backward(const Matrix& logits, const Matrix& target_distribution, float learning_rate) {
    std::cout << "Entering function 'Transformer::backward (with target distribution)'" << std::endl;
    try {
        // Get cached hidden states
        Matrix last_hidden = GradientCheckpoint::get_activation("final_hidden_states");
        if (last_hidden.empty()) {
            throw std::runtime_error("No cached hidden states found for backward pass");
        }

        // First compute loss gradient with respect to hidden states
        Matrix hidden_grad(logits.rows(), config.hidden_size);
        
        // Backward through final layer norm if present
        if (final_ln) {
            hidden_grad = final_ln->backward(logits, last_hidden);
        } else {
            hidden_grad = logits;  // If no layer norm, use logits directly
        }

        // Now backward through LM head with hidden gradients
        std::cout << "\nStarting LM head backward..." << std::endl;
        Matrix d_hidden = lm_head->backward(hidden_grad, target_distribution);

        // Continue with rest of backward pass...
        
    } catch (const std::exception& e) {
        std::cerr << "Error in Transformer backward: " << e.what() << std::endl;
        throw;
    }
    std::cout << "Exiting function 'Transformer::backward (with target distribution)'" << std::endl;
}

void Transformer::update_parameters(float learning_rate) {
    std::cout << "Entering function 'Transformer::update_parameters'" << std::endl;
    try {
        SCOPE_LOG();
        std::cout << "=== Transformer::update_parameters START ===" << std::endl;

        // Update layer parameters
        for (auto& layer : layers) {
            // Update attention parameters
            if (auto* attention = layer->getAttention()) {
                auto& attn_params = attention->parameters();
                auto& attn_grads = attention->param_gradients();

                // Update weights
                update_parameter_with_clip(attn_params.query_weights, attn_grads.query_grad, learning_rate, config);
                update_parameter_with_clip(attn_params.key_weights, attn_grads.key_grad, learning_rate, config);
                update_parameter_with_clip(attn_params.value_weights, attn_grads.value_grad, learning_rate, config);
                update_parameter_with_clip(attn_params.output_weights, attn_grads.output_grad, learning_rate, config);

                // Update biases
                update_parameter_with_clip(attn_params.query_bias, attn_grads.query_bias_grad, learning_rate, config);
                update_parameter_with_clip(attn_params.key_bias, attn_grads.key_bias_grad, learning_rate, config);
                update_parameter_with_clip(attn_params.value_bias, attn_grads.value_bias_grad, learning_rate, config);
                update_parameter_with_clip(attn_params.output_bias, attn_grads.output_bias_grad, learning_rate, config);
            }

            // Update feed forward parameters
            if (auto* ffn = layer->getFeedForward()) {
                auto& ffn_params = ffn->parameters();
                auto& ffn_grads = ffn->param_gradients();

                // Update weights
                update_parameter_with_clip(ffn_params.ff1_weights, ffn_grads.ff1_grad, learning_rate, config);
                update_parameter_with_clip(ffn_params.ff2_weights, ffn_grads.ff2_grad, learning_rate, config);

                // Update biases
                update_parameter_with_clip(ffn_params.ff1_bias, ffn_grads.ff1_bias_grad, learning_rate, config);
                update_parameter_with_clip(ffn_params.ff2_bias, ffn_grads.ff2_bias_grad, learning_rate, config);
            }

            // Update layer norm parameters
            if (auto* ln = layer->getLayerNorm()) {
                auto& ln_params = ln->parameters();
                auto& ln_grads = ln->param_gradients();
                
                update_parameter_with_clip(ln_params.gamma, ln_grads.gamma_grad, learning_rate, config);
                update_parameter_with_clip(ln_params.beta, ln_grads.beta_grad, learning_rate, config);
            }
        }

        std::cout << "=== Transformer::update_parameters END ===" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error in Transformer::update_parameters: " << e.what() << std::endl;
        throw;
    }
    std::cout << "Exiting function 'Transformer::update_parameters'" << std::endl;
}

std::vector<Matrix>& Transformer::parameters() {
    static std::vector<Matrix> all_params;
    all_params.clear();

    // Token embedding parameters
    if (token_embedding) {
        all_params.push_back(token_embedding->get_weights());
    }

    // Layer parameters
    for (const auto& layer : layers) {
        if (auto* attention = layer->getAttention()) {
            auto& attn_params = attention->parameters();
            all_params.push_back(attn_params.query_weights);
            all_params.push_back(attn_params.key_weights);
            all_params.push_back(attn_params.value_weights);
            all_params.push_back(attn_params.output_weights);
        }

        if (auto* ffn = layer->getFeedForward()) {
            auto& ffn_params = ffn->parameters();
            all_params.push_back(ffn_params.ff1_weights);
            all_params.push_back(ffn_params.ff2_weights);
        }
    }

    return all_params;
}

void Transformer::initialize_weights() {
    SCOPE_LOG();
    // Xavier/Glorot initialization with proper scaling and bounds
    auto init_weights = [](Matrix& weights, size_t fan_in, size_t fan_out, bool is_attention = false) {
        float scale = std::sqrt(2.0f / (fan_in + fan_out));
        if (is_attention) {
            // Attention weights need smaller initialization
            scale *= 0.1f;
        }
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, scale);
        
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < weights.rows(); i++) {
            for (size_t j = 0; j < weights.cols(); j++) {
                float value = dist(gen);
                // Clip extreme values
                value = std::max(-2.0f * scale, std::min(2.0f * scale, value));
                weights(i, j) = value;
            }
        }
    };
    
    // Initialize token embeddings with smaller scale
    if (token_embedding) {
        auto& embedding_weights = token_embedding->get_weights();
        init_weights(embedding_weights, tokenizer_->vocab_size(), config.hidden_size);
        embedding_weights *= 0.1f;  // Scale down embeddings
    }
    
    // Initialize transformer layers
    for (auto& layer : layers) {
        if (layer) {
            // Initialize attention weights
            if (auto* attention = layer->getAttention()) {
                auto& params = attention->parameters();
                
                // Query, Key, Value weights need careful initialization
                init_weights(params.query_weights, config.hidden_size, config.hidden_size, true);
                init_weights(params.key_weights, config.hidden_size, config.hidden_size, true);
                init_weights(params.value_weights, config.hidden_size, config.hidden_size, true);
                init_weights(params.output_weights, config.hidden_size, config.hidden_size, true);
                
                // Initialize attention biases to zero
                params.query_bias.initialize_constant(0.0f);
                params.key_bias.initialize_constant(0.0f);
                params.value_bias.initialize_constant(0.0f);
                params.output_bias.initialize_constant(0.0f);
            }
            
            // Initialize feed forward weights
            if (auto* ff = layer->getFeedForward()) {
                auto& params = ff->parameters();
                
                // FF1 (expansion) and FF2 (projection) weights
                init_weights(params.gate_proj_weights, config.hidden_size, config.intermediate_size);
                init_weights(params.up_proj_weights, config.hidden_size, config.intermediate_size);
                init_weights(params.down_proj_weights, config.intermediate_size, config.hidden_size);
                
                // Initialize FF biases to small positive values for ReLU
                params.gate_proj_bias.initialize_constant(0.01f);
                params.up_proj_bias.initialize_constant(0.01f);
                params.down_proj_bias.initialize_constant(0.01f);
            }
            
            // Initialize layer norm parameters
            if (auto* ln = layer->getLayerNorm()) {
                auto& params = ln->parameters();
                params.gamma.initialize_constant(1.0f);  // Identity scaling
                params.beta.initialize_constant(0.0f);   // No initial shift
            }
        }
    }
    
    // Initialize final layer norm
    if (final_ln) {
        auto& params = final_ln->parameters();
        params.gamma.initialize_constant(1.0f);
        params.beta.initialize_constant(0.0f);
    }
    
    std::cout << "Weights initialized with proper scaling" << std::endl;
}

Transformer::~Transformer() {
    std::cout << "Transformer destructor called" << std::endl;
}

void Transformer::load(std::istream& is) {
    SCOPE_LOG();
    try {
        // Load token embedding
        token_embedding->load(is);

        // Load positional encoding
        pos_encoding->load(is);

        // Load transformer layers
        for (auto& layer : layers) {
            layer->load(is);
        }

        // Load final layer norm
        final_ln->load(is);

        // Load language model head
        lm_head->load(is);

    } catch (const std::exception& e) {
        throw std::runtime_error("Error loading transformer: " + std::string(e.what()));
    }
}

void Transformer::set_training(bool training_mode) {
    SCOPE_LOG();
    training = training_mode;
    // Set training mode for all components that need it
    for (auto& layer : layers) {
        layer->training = training_mode;
    }
    if (lm_head) {
        lm_head->set_training(training_mode);
    }
}

std::pair<std::string, PhraseType> Transformer::predict_final_phrase(
    const std::string& input_text,
    const TiktokenTokenizer& tokenizer) {
    // First predict the phrase type
    PhraseType predicted_type = predict_phrase_type(input_text, tokenizer);
    
    // Tokenize input without delimiter
    std::vector<int> tokens = tokenizer.encode(input_text);
    
    // Forward pass
    Matrix hidden_states = forward(tokens, input_text, tokenizer);
    
    // Extract the prediction based on the predicted type
    std::string predicted_phrase = extract_prediction(hidden_states, predicted_type, tokenizer);
    
    return {predicted_phrase, predicted_type};
}

PhraseType Transformer::predict_phrase_type(
    const std::string& input_text,
    const TiktokenTokenizer& tokenizer) {
    // Tokenize input
    std::vector<int> tokens = tokenizer.encode(input_text);
    
    // Forward pass
    Matrix hidden_states = forward(tokens, input_text, tokenizer);
    
    // Analyze hidden states to determine phrase type
    return analyze_phrase_type(hidden_states, tokenizer);
}

PhraseType Transformer::analyze_phrase_type(
    const Matrix& hidden_states,
    const TiktokenTokenizer& tokenizer) {
    // Get the final token predictions
    Matrix final_hidden_states = Matrix(hidden_states.row(hidden_states.rows() - 1));
    
    // Convert hidden states to probabilities
    float max_hidden_state = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < final_hidden_states.cols(); i++) {
        max_hidden_state = std::max(max_hidden_state, final_hidden_states(0, i));
    }
    
    std::vector<float> probabilities(final_hidden_states.cols());
    float sum_exp = 0.0f;
    for (size_t i = 0; i < final_hidden_states.cols(); i++) {
        probabilities[i] = std::exp(final_hidden_states(0, i) - max_hidden_state);
        sum_exp += probabilities[i];
    }
    for (float& prob : probabilities) {
        prob /= sum_exp;
    }
    
    // Calculate scores for each phrase type
    float verb_score = 0.0f;
    float adj_score = 0.0f;
    float general_score = 0.0f;
    
    // Look at top K tokens for scoring
    const size_t K = 10;
    std::vector<std::pair<float, size_t>> top_tokens;
    for (size_t i = 0; i < probabilities.size(); i++) {
        top_tokens.push_back({probabilities[i], i});
    }
    std::sort(top_tokens.begin(), top_tokens.end(), std::greater<>());
    
    // Analyze top tokens
    for (size_t i = 0; i < std::min(K, top_tokens.size()); i++) {
        float prob = top_tokens[i].first;
        std::string token = tokenizer.decode({static_cast<int>(top_tokens[i].second)});
        
        // Check verb patterns
        if (this->is_likely_verb(token)) {
            verb_score += prob;
        }
        // Check adjective patterns
        else if (this->is_likely_adjective(token)) {
            adj_score += prob;
        }
        else {
            general_score += prob;
        }
    }
    
    // Add context-based scoring
    std::string context = tokenizer.decode(this->last_input_tokens_);
    std::transform(context.begin(), context.end(), context.begin(), ::tolower);
    
    // Common verb context patterns
    const std::vector<std::string> verb_contexts = {
        "want to", "like to", "need to", "try to", "going to",
        "starts to", "begins to", "wants to", "likes to", "needs to"
    };
    
    // Common adjective context patterns
    const std::vector<std::string> adj_contexts = {
        "is", "are", "was", "were", "looks", "seems", "feels",
        "appears", "becomes", "remains", "stays", "the", "very"
    };
    
    // Boost scores based on context
    for (const auto& pattern : verb_contexts) {
        if (context.find(pattern) != std::string::npos) {
            verb_score *= 1.5f;
            break;
        }
    }
    
    for (const auto& pattern : adj_contexts) {
        if (context.find(pattern) != std::string::npos) {
            adj_score *= 1.5f;
            break;
        }
    }
    
    // Debug output
    std::cout << "\nPhrase Type Analysis:\n";
    std::cout << "Verb score: " << verb_score << "\n";
    std::cout << "Adjective score: " << adj_score << "\n";
    std::cout << "General score: " << general_score << "\n";
    std::cout << "Context: '" << context << "'\n";
    
    // Return the type with highest score
    if (verb_score > adj_score && verb_score > general_score) {
        return PhraseType::VERB;
    } else if (adj_score > verb_score && adj_score > general_score) {
        return PhraseType::ADJECTIVE;
    }
    
    return PhraseType::GENERAL;
}

std::string Transformer::extract_prediction(
    const Matrix& hidden_states,
    PhraseType phrase_type,
    const TiktokenTokenizer& tokenizer) {
    
    // Create a local generator with time-based seed for more randomness
    auto time_seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937 local_gen(static_cast<unsigned int>(time_seed));
    
    // Get the final token predictions
    Matrix final_hidden_states = Matrix(hidden_states.row(hidden_states.rows() - 1));
    
    // Dynamic temperature based on input length and phrase type
    float base_temp = 0.8f;
    if (phrase_type == PhraseType::VERB) {
        base_temp = 0.9f;  // Slightly higher for verbs
    } else if (phrase_type == PhraseType::ADJECTIVE) {
        base_temp = 0.7f;  // Slightly lower for adjectives
    }
    
    // Add small random variation to temperature
    std::uniform_real_distribution<float> temp_var(0.9f, 1.1f);
    const float temperature = base_temp * temp_var(local_gen);
    
    // Apply softmax with temperature
    std::vector<float> probabilities(final_hidden_states.cols());
    float max_val = -std::numeric_limits<float>::infinity();
    
    // Find max for numerical stability
    for (size_t i = 0; i < final_hidden_states.cols(); i++) {
        max_val = std::max(max_val, final_hidden_states(0, i));
    }
    
    // Compute softmax with temperature
    float sum_exp = 0.0f;
    for (size_t i = 0; i < final_hidden_states.cols(); i++) {
        float scaled_logit = (final_hidden_states(0, i) - max_val) / temperature;
        probabilities[i] = std::exp(scaled_logit);
        sum_exp += probabilities[i];
    }
    
    // Normalize and apply context-based adjustments
    for (size_t i = 0; i < probabilities.size(); i++) {
        probabilities[i] /= sum_exp;
        
        // Decode token for context checking
        std::string token = tokenizer.decode({static_cast<int>(i)});
        
        // Apply phrase type specific boosts
        switch (phrase_type) {
            case PhraseType::VERB:
                if (this->is_likely_verb(token)) {
                    probabilities[i] *= 1.5f;
                }
                break;
            case PhraseType::ADJECTIVE:
                if (this->is_likely_adjective(token)) {
                    probabilities[i] *= 1.5f;
                }
                break;
            default:
                break;
        }
        
        // Penalize very common tokens slightly
        if (token.length() <= 2) {
            probabilities[i] *= 0.8f;
        }
    }
    
    // Renormalize after adjustments
    sum_exp = std::accumulate(probabilities.begin(), probabilities.end(), 0.0f);
    for (float& prob : probabilities) {
        prob /= sum_exp;
    }
    
    // Apply nucleus (top-p) sampling
    const float p = 0.9f;  // Keep top 90% of probability mass
    float cumsum = 0.0f;
    std::vector<size_t> valid_indices;
    
    // Sort indices by probability
    std::vector<size_t> indices(probabilities.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&](size_t a, size_t b) { return probabilities[a] > probabilities[b]; });
    
    // Keep tokens until we reach the probability threshold
    for (size_t idx : indices) {
        if (cumsum >= p) break;
        valid_indices.push_back(idx);
        cumsum += probabilities[idx];
    }
    
    // Ensure we have at least a few options
    if (valid_indices.size() < 3) {
        valid_indices.push_back(indices[valid_indices.size()]);
    }
    
    // Sample from the valid indices
    std::discrete_distribution<> dist(valid_indices.size(), 0.0, 1.0,
        [&](double i) { return probabilities[valid_indices[static_cast<size_t>(i)]]; });
    
    int predicted_token = valid_indices[dist(local_gen)];
    
    // Decode and return the predicted token
    return tokenizer.decode({predicted_token});
}

void Transformer::boost_verb_probabilities(
    Vector& probabilities,
    const TiktokenTokenizer& tokenizer,
    std::mt19937* gen
) {
    const float verb_boost = 0.3f;
    const float non_verb_penalty = 0.1f;
    
    for (size_t i = 0; i < probabilities.size(); i++) {
        std::string token = tokenizer.decode({static_cast<int>(i)});
        if (is_likely_verb(token)) {
            probabilities[i] *= (1.0f + verb_boost);
        } else {
            probabilities[i] *= (1.0f - non_verb_penalty);
        }
    }
    
    // Add random noise if generator provided
    if (gen) {
        std::normal_distribution<float> noise(0.0f, 0.05f);
        for (size_t i = 0; i < probabilities.size(); i++) {
            probabilities[i] *= (1.0f + noise(*gen));
        }
    }
}

void Transformer::boost_adjective_probabilities(
    Vector& probabilities,
    const TiktokenTokenizer& tokenizer,
    std::mt19937* gen
) {
    const float adj_boost = 0.3f;
    const float non_adj_penalty = 0.1f;
    
    for (size_t i = 0; i < probabilities.size(); i++) {
        std::string token = tokenizer.decode({static_cast<int>(i)});
        if (is_likely_adjective(token)) {
            probabilities[i] *= (1.0f + adj_boost);
        } else {
            probabilities[i] *= (1.0f - non_adj_penalty);
        }
    }
    
    // Add random noise if generator provided
    if (gen) {
        std::normal_distribution<float> noise(0.0f, 0.05f);
        for (size_t i = 0; i < probabilities.size(); i++) {
            probabilities[i] *= (1.0f + noise(*gen));
        }
    }
}

bool Transformer::is_likely_verb(const std::string& token) const {
    // Common verb endings
    const std::vector<std::string> verb_endings = {
        "ing", "ed", "ate", "ize", "ify", "ise", "ect",
        "ent", "age", "ute", "end", "ish", "ade", "ine",
        "ume", "ure", "ide", "ive", "ete", "act"
    };

    // Common verbs that don't follow standard patterns
    const std::unordered_set<std::string> common_verbs = {
        "go", "do", "make", "take", "come", "see", "get",
        "know", "find", "give", "tell", "work", "call", "try",
        "ask", "need", "feel", "let", "put", "mean", "keep",
        "run", "set", "move", "play", "pay", "hear", "help",
        "talk", "turn", "start", "show", "wait", "plan", "learn"
    };

    // First check if it's a common verb
    std::string lower_token = token;
    std::transform(lower_token.begin(), lower_token.end(), lower_token.begin(), ::tolower);
    if (common_verbs.find(lower_token) != common_verbs.end()) {
        return true;
    }

    // Then check for verb endings
    for (const auto& ending : verb_endings) {
        if (lower_token.length() > ending.length() && 
            lower_token.substr(lower_token.length() - ending.length()) == ending) {
            return true;
        }
    }

    return false;
}

bool Transformer::is_likely_adjective(const std::string& token) const {
    // Common adjective endings
    const std::vector<std::string> adj_endings = {
        "ful", "ous", "ible", "able", "al", "ive", "less",
        "ish", "like", "ic", "ian", "en", "ent", "ant",
        "ary", "ing", "ed", "y", "ly", "some"
    };

    // Common adjectives that don't follow standard patterns
    const std::unordered_set<std::string> common_adjectives = {
        "good", "bad", "new", "old", "high", "low", "big",
        "small", "large", "little", "long", "short", "great",
        "hot", "cold", "warm", "cool", "easy", "hard", "fast",
        "slow", "early", "late", "young", "right", "wrong",
        "true", "false", "open", "close", "light", "dark",
        "heavy", "soft", "hard", "weak", "strong", "rich",
        "poor", "safe", "clean", "dirty", "quiet", "loud"
    };

    // First check if it's a common adjective
    std::string lower_token = token;
    std::transform(lower_token.begin(), lower_token.end(), lower_token.begin(), ::tolower);
    if (common_adjectives.find(lower_token) != common_adjectives.end()) {
        return true;
    }

    // Then check for adjective endings
    for (const auto& ending : adj_endings) {
        if (lower_token.length() > ending.length() && 
            lower_token.substr(lower_token.length() - ending.length()) == ending) {
            return true;
        }
    }

    return false;
}

void update_parameter_with_clip(Matrix& param, const Matrix& grad, float learning_rate, const TransformerConfig& config) {
    const float clip_threshold = config.gradient_clip_threshold;
    const float weight_decay = config.weight_decay;
    const float max_relative_change = 0.1f;  // Reduced back to 10% for stability
    
    // Calculate gradient norm
    float grad_norm = 0.0f;
    #pragma omp parallel for reduction(+:grad_norm)
    for (size_t i = 0; i < grad.rows(); ++i) {
        for (size_t j = 0; j < grad.cols(); ++j) {
            grad_norm += grad(i, j) * grad(i, j);
        }
    }
    grad_norm = std::sqrt(grad_norm);
    
    // Apply gradient clipping with smooth transition
    float scaling_factor = 1.0f;
    if (grad_norm > clip_threshold) {
        scaling_factor = clip_threshold / (grad_norm + 1e-8f);
        // Use linear scaling for more stability
        scaling_factor = std::max(0.1f, scaling_factor);
    }
    
    // Use a more stable learning rate adaptation
    float adaptive_lr = learning_rate;
    if (grad_norm > clip_threshold) {
        adaptive_lr *= scaling_factor;
    }
    
    // Update parameters with clipped gradients and proper weight decay
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < param.rows(); ++i) {
        for (size_t j = 0; j < param.cols(); ++j) {
            // Apply weight decay separately from gradient
            float decay_update = -weight_decay * param(i, j);
            float grad_update = -grad(i, j) * scaling_factor;
            
            // Compute total update
            float total_update = adaptive_lr * (grad_update + decay_update);
            
            // Limit relative change
            float max_update = max_relative_change * std::abs(param(i, j) + 1e-8f);
            total_update = std::clamp(total_update, -max_update, max_update);
            
            // Apply update
            param(i, j) += total_update;
        }
    }
}

void update_parameter_with_clip(Vector& param, const Vector& grad, float learning_rate, const TransformerConfig& config) {
    const float clip_threshold = config.gradient_clip_threshold;
    const float weight_decay = config.weight_decay;
    
    float grad_norm = 0.0f;
    #pragma omp parallel for reduction(+:grad_norm)
    for (size_t i = 0; i < grad.size(); ++i) {
        grad_norm += grad[i] * grad[i];
    }
    grad_norm = std::sqrt(grad_norm);
    
    // Apply adaptive clipping with a softer threshold
    float scaling_factor = 1.0f;
    if (grad_norm > clip_threshold) {
        scaling_factor = clip_threshold / (grad_norm + 1e-8f);
        scaling_factor = std::sqrt(scaling_factor);  // Softer scaling
    }
    
    #pragma omp parallel for
    for (size_t i = 0; i < param.size(); ++i) {
        float decay = weight_decay * param[i];
        float grad_update = -grad[i] * scaling_factor;
        float total_update = learning_rate * (grad_update + decay);
        float max_update = 0.1f * std::abs(param[i] + 1e-8f);
        total_update = std::clamp(total_update, -max_update, max_update);
        param[i] += total_update;
    }
}

// Add helper function declarations
void update_attention_parameters(MultiHeadAttention* attention, float learning_rate, const TransformerConfig& config) {
    auto& params = attention->parameters();
    auto& grads = attention->param_gradients();
    
    // Update query parameters
    update_parameter_with_clip(params.query_weights, grads.query_grad, learning_rate, config);
    update_parameter_with_clip(params.query_bias, grads.query_bias_grad, learning_rate, config);
    
    // Update key parameters
    update_parameter_with_clip(params.key_weights, grads.key_grad, learning_rate, config);
    update_parameter_with_clip(params.key_bias, grads.key_bias_grad, learning_rate, config);
    
    // Update value parameters
    update_parameter_with_clip(params.value_weights, grads.value_grad, learning_rate, config);
    update_parameter_with_clip(params.value_bias, grads.value_bias_grad, learning_rate, config);
    
    // Update output parameters
    update_parameter_with_clip(params.output_weights, grads.output_grad, learning_rate, config);
    update_parameter_with_clip(params.output_bias, grads.output_bias_grad, learning_rate, config);
}

void update_ffn_parameters(FeedForward* ffn, float learning_rate, const TransformerConfig& config) {
    auto& params = ffn->parameters();
    auto& grads = ffn->param_gradients();
    
    // Update FF1 parameters
    update_parameter_with_clip(params.ff1_weights, grads.ff1_grad, learning_rate, config);
    update_parameter_with_clip(params.ff1_bias, grads.ff1_bias_grad, learning_rate, config);
    
    // Update FF2 parameters
    update_parameter_with_clip(params.ff2_weights, grads.ff2_grad, learning_rate, config);
    update_parameter_with_clip(params.ff2_bias, grads.ff2_bias_grad, learning_rate, config);
}

// Add as a static member of the Transformer class
static DynamicLossScaler loss_scaler;

// Add learning_rate parameter to backward_pass
void Transformer::backward_pass(const Matrix& output, const Matrix& target_distribution, float learning_rate) {
    std::cout << "Entering function 'Transformer::backward_pass'" << std::endl;
    try {
        // Compute loss gradient with proper sequence handling
        Matrix loss_grad = Utils::compute_loss_gradient(output, target_distribution);
        
        // Retrieve sequence boundaries
        const auto& seq_boundaries = last_seq_boundaries;
        
        // Apply adaptive loss scaling if using FP16
        if (config.use_fp16) {
            // Track gradient statistics for adaptive scaling
            static float running_max_grad = 0.0f;
            static float running_min_grad = std::numeric_limits<float>::max();
            static float scale_growth_factor = 2.0f;
            static float scale_reduction_factor = 0.5f;
            static size_t stable_steps = 0;
            static const size_t STABILITY_THRESHOLD = 1000;
            static const float MAX_SCALE = 32768.0f;
            static const float MIN_SCALE = 1.0f;
            
            // Compute current gradient statistics
            float current_max_grad = 0.0f;
            float current_min_grad = std::numeric_limits<float>::max();
            bool has_inf_nan = false;
            
            #pragma omp parallel for reduction(max:current_max_grad) reduction(min:current_min_grad) reduction(||:has_inf_nan)
            for (size_t i = 0; i < loss_grad.rows(); i++) {
                for (size_t j = 0; j < loss_grad.cols(); j++) {
                    float val = std::abs(loss_grad(i, j));
                    if (std::isfinite(val)) {
                        current_max_grad = std::max(current_max_grad, val);
                        current_min_grad = std::min(current_min_grad, val);
        } else {
                        has_inf_nan = true;
                    }
                }
            }
            
            // Update running statistics with exponential moving average
            const float alpha = 0.95f;
            running_max_grad = alpha * running_max_grad + (1.0f - alpha) * current_max_grad;
            running_min_grad = alpha * running_min_grad + (1.0f - alpha) * current_min_grad;
            
            // Get current scale
            float current_scale = loss_scaler.get_scale();
            
            // Adjust scale based on gradient statistics and stability
            if (has_inf_nan) {
                // Reduce scale on inf/nan
                current_scale *= scale_reduction_factor;
                stable_steps = 0;
            } else {
                stable_steps++;
                
                // Check if gradients are too small
                if (running_max_grad < 1e-4f && current_scale < MAX_SCALE) {
                    current_scale *= scale_growth_factor;
                    stable_steps = 0;
                }
                // Check if gradients are too large
                else if (running_max_grad > 1.0f && current_scale > MIN_SCALE) {
                    current_scale *= scale_reduction_factor;
                    stable_steps = 0;
                }
                // Increase scale if we've been stable for a while
                else if (stable_steps >= STABILITY_THRESHOLD && current_scale < MAX_SCALE) {
                    current_scale *= scale_growth_factor;
                    stable_steps = 0;
                }
            }
            
            // Clamp scale to valid range
            current_scale = std::clamp(current_scale, MIN_SCALE, MAX_SCALE);
            
            // Update loss scaler
            loss_scaler.set_scale(current_scale);
            
            // Apply scale to gradients
            #pragma omp parallel for collapse(2)
            for (size_t i = 0; i < loss_grad.rows(); i++) {
                for (size_t j = 0; j < loss_grad.cols(); j++) {
                    loss_grad(i, j) *= current_scale;
                }
            }
            
            std::cout << "FP16 scaling - Scale: " << current_scale 
                      << ", Max grad: " << running_max_grad 
                      << ", Min grad: " << running_min_grad 
                      << ", Stable steps: " << stable_steps << std::endl;
        }
        
        // Initialize gradient accumulation buffers
        std::vector<Matrix> layer_gradients;
        layer_gradients.reserve(layers.size());
        
        bool has_inf_nan = false;
        
        // Backward through layers with proper sequence handling
        for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i) {
            std::string layer_key = "layer_" + std::to_string(i) + "_input";
            Matrix layer_input = GradientCheckpoint::get_activation(layer_key);
            
            if (layer_input.empty()) {
                throw std::runtime_error("No cached activation found for layer " + std::to_string(i));
            }
            
            try {
                Matrix layer_grad = layers[i]->backward(loss_grad, layer_input);
                if (!layer_grad.empty()) {
                    if (config.use_fp16 && loss_scaler.has_inf_or_nan(layer_grad)) {
                        has_inf_nan = true;
                        break;
                    }
                    
                    layer_gradients.push_back(layer_grad);
                    loss_grad = layer_grad;  // Propagate gradients to next layer
                }
            } catch (const std::exception& e) {
                std::cerr << "Error in layer " << i << " backward pass: " << e.what() << std::endl;
                throw;
            }
        }
        
        // Handle FP16 unscaling at the end
        if (config.use_fp16 && !has_inf_nan) {
            float inv_scale = 1.0f / loss_scaler.get_scale();
            for (auto& grad : layer_gradients) {
                grad *= inv_scale;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in Transformer::backward_pass: " << e.what() << std::endl;
        throw;
    }
    std::cout << "Exiting function 'Transformer::backward_pass'" << std::endl;
}

// Helper function to unscale gradients
void Transformer::unscale_gradients(MultiHeadAttention::Gradients& grads, float scale) {
    std::cout << "Entering function 'Transformer::unscale_gradients (attention)'" << std::endl;
    try {
        // Unscale weight gradients
        grads.query_grad *= scale;
        grads.key_grad *= scale;
        grads.value_grad *= scale;
        grads.output_grad *= scale;

        // Unscale bias gradients
        grads.query_bias_grad *= scale;
        grads.key_bias_grad *= scale;
        grads.value_bias_grad *= scale;
        grads.output_bias_grad *= scale;
    } catch (const std::exception& e) {
        std::cerr << "Error in Transformer::unscale_gradients: " << e.what() << std::endl;
        throw;
    }
    std::cout << "Exiting function 'Transformer::unscale_gradients (attention)'" << std::endl;
}

void Transformer::unscale_gradients(FeedForward::Gradients& grads, float scale) {
    std::cout << "Entering function 'Transformer::unscale_gradients (feedforward)'" << std::endl;
    try {
        // Unscale weight gradients
        grads.ff1_grad *= scale;
        grads.ff2_grad *= scale;

        // Unscale bias gradients
        grads.ff1_bias_grad *= scale;
        grads.ff2_bias_grad *= scale;
    } catch (const std::exception& e) {
        std::cerr << "Error in Transformer::unscale_gradients: " << e.what() << std::endl;
        throw;
    }
    std::cout << "Exiting function 'Transformer::unscale_gradients (feedforward)'" << std::endl;
}

Transformer& Transformer::operator=(const Transformer& other) {
    if (this != &other) {
        // Create a new config instead of assigning
        config = TransformerConfig(other.config);  // Assuming TransformerConfig has a copy constructor

        // Deep copy all components using make_unique and copy constructors
        if (other.token_embedding) {
            token_embedding = std::make_unique<TokenEmbedding>(*other.token_embedding);
        } else {
            token_embedding.reset();
        }
        
        if (other.pos_encoding) {
            pos_encoding = std::make_unique<PositionalEncoding>(*other.pos_encoding);
        } else {
            pos_encoding.reset();
        }
        
        if (other.final_ln) {
            final_ln = std::make_unique<LayerNorm>(*other.final_ln);
        } else {
            final_ln.reset();
        }
        
        if (other.lm_head) {
            lm_head = std::make_unique<LanguageModelHead>(*other.lm_head);
        } else {
            lm_head.reset();
        }
        
        // Copy layers
        layers.clear();
        layers.reserve(other.layers.size());
        for (const auto& layer : other.layers) {
            if (layer) {
                layers.push_back(std::make_unique<TransformerLayer>(*layer));
            } else {
                layers.push_back(nullptr);
            }
        }
        
        // Copy state
        training = other.training;
        hidden_states = other.hidden_states;
        last_hidden_states = other.last_hidden_states;
        m_layer_activations = other.m_layer_activations;
        m_kv_caches = other.m_kv_caches;
        last_seq_boundaries = other.last_seq_boundaries;
        last_input_tokens_ = other.last_input_tokens_;
        last_input_query_ = other.last_input_query_;
    }
    return *this;
}

void Transformer::backward(const Matrix& grad, const Matrix& activation, size_t layer_idx) {
    std::cout << "Entering function 'Transformer::backward (layer-specific)'" << std::endl;
    try {
        if (layer_idx >= layers.size()) {
            throw std::runtime_error("Layer index out of bounds in backward pass");
        }
        
        // Ensure we're in training mode
        if (!training) {
            throw std::runtime_error("Cannot perform backward pass while in inference mode");
        }

        // Store gradients for the current layer
        if (!parameter_grads.has_value()) {
            parameter_grads = std::vector<Matrix>();
        }
        
        // Compute gradients through the layer
        Matrix layer_grads = layers[layer_idx]->backward(grad, activation);
        parameter_grads->push_back(layer_grads);
    } catch (const std::exception& e) {
        std::cerr << "Error in Transformer::backward: " << e.what() << std::endl;
        throw;
    }
    std::cout << "Exiting function 'Transformer::backward (layer-specific)'" << std::endl;
}

float Transformer::get_dynamic_temperature(PhraseType type, const std::mt19937& gen) const {
    std::cout << "Entering function 'Transformer::get_dynamic_temperature'" << std::endl;
    try {
        // Base temperature from config
        float base_temp = config.token_prediction.temperature;
        
        // Add randomness to temperature
        // Create a local copy of the generator since we need to modify it
        std::mt19937 local_gen(gen.default_seed);
        std::uniform_real_distribution<float> temp_noise(0.9f, 1.1f);
        float random_factor = temp_noise(local_gen);
        
        // Adjust temperature based on phrase type
        float type_multiplier = 1.0f;
        switch(type) {
            case PhraseType::VERB:
                type_multiplier = 1.3f;  // Higher temperature for verbs
                break;
            case PhraseType::ADJECTIVE:
                type_multiplier = 1.2f;  // Higher temperature for adjectives
                break;
            case PhraseType::NOUN:
                type_multiplier = 1.1f;  // Slightly higher for nouns
                break;
            default:
                type_multiplier = 1.0f;
        }
        
        // Consider input length - longer inputs get lower temperature
        float length_factor = 1.0f;
        if (!last_input_tokens_.empty()) {
            length_factor = 1.0f + (0.1f * std::min(last_input_tokens_.size(), size_t(5)));
        }
        
        // Calculate final temperature
        float final_temp = base_temp * type_multiplier * random_factor / length_factor;
        
        // Ensure temperature stays within reasonable bounds
        return std::clamp(final_temp, 0.8f, 2.5f);
    } catch (const std::exception& e) {
        std::cerr << "Error in Transformer::get_dynamic_temperature: " << e.what() << std::endl;
        throw;
    }
    std::cout << "Exiting function 'Transformer::get_dynamic_temperature'" << std::endl;
}

std::vector<TokenPrediction> Transformer::predict_next_tokens(const std::string& input, size_t num_predictions) {
    std::cout << "Entering function 'Transformer::predict_next_tokens'" << std::endl;
    try {
        // First get the input tokens
        std::vector<int> input_tokens = tokenize(input);
        
        // Get logits from forward pass
        Matrix logits = forward(input_tokens, input, *tokenizer_);
        
        // Apply diversity penalty to logits
        const float diversity_penalty = 0.9f;  // Penalty factor for repeated tokens
        const float presence_penalty = 0.3f;   // Penalty for tokens present in input
        
        std::unordered_map<size_t, float> token_penalties;
        
        // Initialize penalties based on input tokens
        for (const auto& token : input_tokens) {
            token_penalties[token] = presence_penalty;
        }
        
        // Track used tokens for diversity penalty
        std::set<size_t> used_tokens;
        std::vector<TokenPrediction> predictions;
        
        // Get the final row of logits for prediction
        Vector final_logits = logits.row(logits.rows() - 1);
        
        for (size_t i = 0; i < num_predictions; i++) {
            // Apply penalties to logits
            Vector adjusted_logits = final_logits;
            for (size_t j = 0; j < adjusted_logits.size(); j++) {
                // Apply presence penalty
                if (token_penalties.count(j)) {
                    adjusted_logits[j] -= token_penalties[j];
                }
                
                // Apply diversity penalty for already predicted tokens
                if (used_tokens.count(j)) {
                    adjusted_logits[j] *= diversity_penalty;
                }
            }
            
            // Get temperature for this prediction
            float temp = get_dynamic_temperature(get_phrase_type(input), gen_);
            
            // Get probabilities with temperature
            Vector probs = softmax_with_temperature(adjusted_logits, temp);
            
            // Sample from distribution
            size_t token_id = sample_from_distribution(probs, gen_);
            float prob = probs[token_id];
            float raw_logit = final_logits[token_id];
            
            // Add prediction
            predictions.push_back({
                token_id,
                detokenize({static_cast<int>(token_id)}),
                prob,
                raw_logit,
                get_token_type(token_id)
            });
            
            // Update tracking
            used_tokens.insert(token_id);
            token_penalties[token_id] = presence_penalty;
        }
        
        return predictions;
    } catch (const std::exception& e) {
        std::cerr << "Error in Transformer::predict_next_tokens: " << e.what() << std::endl;
        throw;
    }
    std::cout << "Exiting function 'Transformer::predict_next_tokens'" << std::endl;
}

float Transformer::compute_semantic_similarity(const std::string& token, const std::string& input) const {
    float similarity = 0.0f;
    
    // Check for direct substring match
    if (input.find(token) != std::string::npos) {
        similarity += 0.5f;
    }
    
    // Check for subject-verb agreement
    if (is_likely_verb(token) && is_subject(input)) {
        similarity += 0.3f;
    }
    
    // Check for adjective-noun agreement
    if (is_likely_adjective(token) && input.find(' ') != std::string::npos) {
        std::string last_word = input.substr(input.find_last_of(' ') + 1);
        if (!is_article(last_word) && !is_likely_verb(last_word)) {
            similarity += 0.3f;
        }
    }
    
    // Check for linking verb patterns
    if (is_linking_verb(token) && is_likely_adjective(input)) {
        similarity += 0.4f;
    }
    
    // Consider recent context from generation history
    if (!generation_history.empty()) {
        std::vector<std::string> recent_tokens;
        size_t context_window = std::min(generation_history.size(), size_t(5));
        for (size_t i = generation_history.size() - context_window; i < generation_history.size(); i++) {
            recent_tokens.push_back(tokenizer_->decode({generation_history[i]}));
        }
        
        // Check for semantic coherence with recent tokens
        for (const auto& recent : recent_tokens) {
            if (is_likely_verb(recent) && is_likely_verb(token)) {
                similarity -= 0.2f;  // Penalize consecutive verbs
            }
            if (is_likely_adjective(recent) && is_likely_adjective(token)) {
                similarity -= 0.2f;  // Penalize consecutive adjectives
            }
        }
    }
    
    return std::clamp(similarity, 0.0f, 1.0f);
}

float Transformer::get_context_boost(const std::string& token, const std::string& input) {
    float boost = 0.0f;
    
    // Parse input for key components
    std::istringstream iss(input);
    std::vector<std::string> words;
    std::string word;
    while (iss >> word) {
        words.push_back(word);
    }
    
    // Analyze input structure
    bool has_subject = false;
    bool has_verb = false;
    bool has_adjective = false;
    std::string subject;
    
    for (const auto& w : words) {
        if (is_subject(w)) {
            has_subject = true;
            subject = w;
        }
        if (is_likely_verb(w)) has_verb = true;
        if (is_likely_adjective(w)) has_adjective = true;
    }
    
    // Boost based on grammatical completeness
    if (is_likely_verb(token)) {
        if (has_subject && !has_verb) boost += 0.4f;  // Need a verb
        if (has_verb) boost -= 0.2f;  // Already have a verb
    }
    
    if (is_likely_adjective(token)) {
        if (has_subject && !has_adjective) boost += 0.3f;  // Adjective could help
        if (has_adjective) boost -= 0.2f;  // Already descriptive
    }
    
    // Boost based on semantic relationships
    if (!subject.empty() && !hidden_states.empty()) {
        std::vector<int> subject_tokens = tokenizer_->encode(subject);
        std::vector<int> token_tokens = tokenizer_->encode(token);
        
        if (!subject_tokens.empty() && !token_tokens.empty()) {
            Vector subject_embedding = hidden_states.row(subject_tokens[0]);
            Vector token_embedding = hidden_states.row(token_tokens[0]);
            
            // Compute cosine similarity
            float dot_product = 0.0f;
            float norm1 = 0.0f;
            float norm2 = 0.0f;
            
            for (size_t i = 0; i < subject_embedding.size(); i++) {
                dot_product += subject_embedding[i] * token_embedding[i];
                norm1 += subject_embedding[i] * subject_embedding[i];
                norm2 += token_embedding[i] * token_embedding[i];
            }
            
            float similarity = dot_product / (std::sqrt(norm1) * std::sqrt(norm2) + 1e-6f);
            boost += similarity * 0.3f;
        }
    }
    
    return std::clamp(boost, -0.5f, 0.5f);
}

bool Transformer::is_subject(const std::string& word) const {
    static const std::unordered_set<std::string> subjects = {
        "i", "you", "he", "she", "it", "we", "they",
        "this", "that", "these", "those", "who", "which",
        "the", "a", "an"
    };
    std::string lower_word = word;
    std::transform(lower_word.begin(), lower_word.end(), lower_word.begin(), ::tolower);
    return subjects.find(lower_word) != subjects.end();
}

bool Transformer::is_article(const std::string& word) const {
    static const std::unordered_set<std::string> articles = {"a", "an", "the"};
    std::string lower_word = word;
    std::transform(lower_word.begin(), lower_word.end(), lower_word.begin(), ::tolower);
    return articles.find(lower_word) != articles.end();
}

bool Transformer::is_linking_verb(const std::string& word) const {
    static const std::unordered_set<std::string> linking_verbs = {
        "is", "are", "was", "were", "be", "been", "being",
        "seems", "appears", "looks", "feels", "sounds",
        "becomes", "remains", "stays", "grows", "turns"
    };
    std::string lower_word = word;
    std::transform(lower_word.begin(), lower_word.end(), lower_word.begin(), ::tolower);
    return linking_verbs.find(lower_word) != linking_verbs.end();
}

std::vector<int> Transformer::tokenize(const std::string& text) const {
    if (!tokenizer_) {
        throw std::runtime_error("Tokenizer not initialized");
    }
    return tokenizer_->encode(text);
}

std::string Transformer::detokenize(const std::vector<int>& tokens) const {
    if (!tokenizer_) {
        throw std::runtime_error("Tokenizer not initialized");
    }
    return tokenizer_->decode(tokens);
}

PhraseType Transformer::get_token_type(size_t token_id) const {
    std::cout << "Entering function 'Transformer::get_token_type'" << std::endl;
    try {
        std::string token = detokenize({static_cast<int>(token_id)});
    if (is_likely_verb(token)) {
            return PhraseType::VERB;
        } else if (is_likely_adjective(token)) {
            return PhraseType::ADJECTIVE;
        } else if (token.length() > 2) {  // Simple heuristic for nouns
            return PhraseType::NOUN;
        }
        return PhraseType::OTHER;
    } catch (const std::exception& e) {
        std::cerr << "Error in Transformer::get_token_type: " << e.what() << std::endl;
        throw;
    }
    std::cout << "Exiting function 'Transformer::get_token_type'" << std::endl;
}

size_t Transformer::sample_from_distribution(const Vector& probabilities, std::mt19937& gen) const {
    std::discrete_distribution<size_t> dist(probabilities.begin(), probabilities.end());
    return dist(gen);
}

Vector Transformer::softmax_with_temperature(const Vector& logits, float temperature) const {
    std::cout << "Entering function 'Transformer::softmax_with_temperature'" << std::endl;
    try {
        Vector output(logits.size());
        float max_val = *std::max_element(logits.begin(), logits.end());
        
        float sum_exp = 0.0f;
        for (size_t i = 0; i < logits.size(); i++) {
            output[i] = std::exp((logits[i] - max_val) / temperature);
            sum_exp += output[i];
        }
        
        for (size_t i = 0; i < output.size(); i++) {
            output[i] /= sum_exp;
        }
        
        return output;
    } catch (const std::exception& e) {
        std::cerr << "Error in Transformer::softmax_with_temperature: " << e.what() << std::endl;
        throw;
    }
    std::cout << "Exiting function 'Transformer::softmax_with_temperature'" << std::endl;
}

Vector Transformer::forward(const std::vector<int>& input_tokens, bool update_kv_cache) {
    // ... existing forward pass code ...
    
    // Get logits from the final layer
    // Convert the row vector to a 1xN matrix before passing to lm_head
    Matrix last_hidden_row(1, hidden_states.cols());
    last_hidden_row.row(0) = hidden_states.row(hidden_states.rows() - 1);
    Vector logits = lm_head->forward(last_hidden_row);
    
    
    // If we're generating, update the frequency stats for the last token
    if (!input_tokens.empty() && update_kv_cache) {
        update_token_frequency(input_tokens.back());
    }
    
    return logits;
}

void Transformer::reset_cache() {
    // Reset KV cache in all layers
    for (auto& layer : layers) {
        if (layer && layer->getAttention()) {
            layer->getAttention()->reset_cache();
        }
    }
    
    // Reset frequency statistics
    reset_frequency_stats();
    
    // Reset current context
    current_input_context.clear();
    
    // Reset any other stateful components
    if (lm_head) {
        lm_head->reset_state();
    }
}

// Frequency tracking
struct TokenFrequencyStats {
    size_t total_occurrences = 0;
    size_t recent_occurrences = 0;  // Within recency window
    std::vector<size_t> positions;  // Track positions where token was used
    float frequency_penalty = 0.0f;
    float recency_penalty = 0.0f;
};

std::unordered_map<int, TokenFrequencyStats> token_frequency_stats;
size_t total_tokens_generated = 0;
const size_t RECENCY_WINDOW = 50;  // Consider last 50 tokens for recency
const float BASE_FREQUENCY_PENALTY = 0.1f;  // Reduced from 0.3f
const float BASE_RECENCY_PENALTY = 0.05f;   // Reduced from 0.2f
const float PENALTY_DECAY = 0.98f;  // Increased from 0.95f

// Implementation of update_token_frequency
void Transformer::update_token_frequency(int token_id) {
    auto& stats = token_frequency_stats[token_id];
    
    // Update total occurrences
    stats.total_occurrences++;
    
    // Update positions list
    stats.positions.push_back(total_tokens_generated);
    
    // Update recent occurrences (within recency window)
    stats.recent_occurrences = std::count_if(
        stats.positions.begin(),
        stats.positions.end(),
        [this](size_t pos) {
            return total_tokens_generated - pos <= RECENCY_WINDOW;
        }
    );
    
    // Prune old positions outside recency window
    while (!stats.positions.empty() && 
           total_tokens_generated - stats.positions.front() > RECENCY_WINDOW) {
        stats.positions.erase(stats.positions.begin());
    }
    
    // Update penalties
    stats.frequency_penalty = BASE_FREQUENCY_PENALTY * 
        (static_cast<float>(stats.total_occurrences) / (total_tokens_generated + 1));
    
    stats.recency_penalty = BASE_RECENCY_PENALTY * 
        (static_cast<float>(stats.recent_occurrences) / RECENCY_WINDOW);
    
    // Apply decay to penalties
    stats.frequency_penalty *= PENALTY_DECAY;
    stats.recency_penalty *= PENALTY_DECAY;
    
    total_tokens_generated++;
}

// Add initialization of attention layers
void Transformer::initialize_attention_layers() {
    if (!tokenizer_) {
        throw std::runtime_error("Tokenizer not set during attention layer initialization");
    }
    
    for (auto& layer : layers) {
        if (layer && layer->getAttention()) {
            layer->getAttention()->set_tokenizer(tokenizer_);
        }
    }
}

std::vector<int> Transformer::generate(const std::vector<int>& input_tokens, 
                            size_t max_length,
                            float temperature) {
    std::cout << "\n=== Starting Transformer::generate ===" << std::endl;
    std::cout << "Input tokens: " << input_tokens.size() << std::endl;
    std::cout << "Max length: " << max_length << std::endl;
    std::cout << "Temperature: " << temperature << std::endl;
    
    if (!lm_head) {
        throw std::runtime_error("Language model head is not initialized!");
    }
    
    if (input_tokens.empty()) {
        throw std::runtime_error("Input tokens cannot be empty");
    }
    
    std::vector<int> output_tokens = input_tokens;
    size_t tokens_generated = 0;
    
    // Clear diversity tracking for new generation
    std::cout << "Resetting diversity tracking..." << std::endl;
    reset_diversity_tracking();
    
    // Decode input tokens to set initial context if not already set
    if (current_input_context.empty()) {
        std::cout << "Setting initial context from input tokens..." << std::endl;
        current_input_context = detokenize(input_tokens);
    }
    std::cout << "Current context: '" << current_input_context << "'" << std::endl;
    
    // Initialize KV cache for efficient generation
    std::cout << "Clearing KV cache..." << std::endl;
    clear_kv_cache();
    
    // Set to inference mode
    std::cout << "Setting to inference mode..." << std::endl;
    set_training(false);
    
    try {
        while (output_tokens.size() < max_length) {
            std::cout << "\n--- Generation step " << tokens_generated + 1 << " ---" << std::endl;
            std::cout << "Current sequence length: " << output_tokens.size() << std::endl;
            
            // Forward pass through the model with proper context
            std::cout << "Performing forward pass..." << std::endl;
            Matrix logits = forward(output_tokens, current_input_context, *tokenizer_);
            std::cout << "Logits shape: [" << logits.rows() << " x " << logits.cols() << "]" << std::endl;
            
            if (logits.empty()) {
                throw std::runtime_error("Forward pass returned empty logits");
            }
            
            // Validate logits dimensions
            if (logits.rows() == 0 || logits.cols() == 0) {
                throw std::runtime_error("Invalid logits dimensions: [" + 
                    std::to_string(logits.rows()) + " x " + std::to_string(logits.cols()) + "]");
            }
            
            // Get the last row of logits for next token prediction
            std::cout << "Extracting last row of logits..." << std::endl;
            if (logits.rows() == 0) {
                throw std::runtime_error("Cannot extract last row from empty logits");
            }
            
            Matrix last_logits(1, logits.cols());
            try {
                for (size_t j = 0; j < logits.cols(); j++) {
                    last_logits(0, j) = logits(logits.rows() - 1, j);
                }
            } catch (const std::exception& e) {
                throw std::runtime_error("Error extracting last row: " + std::string(e.what()) + 
                    "\nLogits shape: [" + std::to_string(logits.rows()) + " x " + 
                    std::to_string(logits.cols()) + "]");
            }
            
            // Apply diversity penalties
            std::cout << "Applying diversity penalties..." << std::endl;
            std::cout << "last_logits dimensions: [" << last_logits.rows() << " x " << last_logits.cols() << "]" << std::endl;
            
            // Create a new Vector directly from the data
            Vector logits_vec(last_logits.cols());
            for (size_t i = 0; i < last_logits.cols(); i++) {
                logits_vec[i] = last_logits(0, i);
            }
            
            std::cout << "Created logits vector of size: " << logits_vec.size() << std::endl;
            
            // Get dynamic temperature based on generation state
            std::cout << "Computing dynamic temperature..." << std::endl;
            float dynamic_temp = get_dynamic_temperature(
                predict_phrase_type(current_input_context, *tokenizer_),
                gen_
            );
            std::cout << "Dynamic temperature: " << dynamic_temp << std::endl;
            
            // Combine base and dynamic temperature
            float effective_temp = temperature * dynamic_temp;
            std::cout << "Effective temperature: " << effective_temp << std::endl;
            
            // Use the language model head's sampling function for next token prediction
            std::cout << "Sampling next token..." << std::endl;
            Vector sampled;
            try {
                sampled = lm_head->sample_next_token(last_logits, current_input_context, effective_temp);
            } catch (const std::exception& e) {
                throw std::runtime_error("Error in sample_next_token: " + std::string(e.what()));
            }
            
            // Find the index of the sampled token
            std::cout << "Finding sampled token index..." << std::endl;
            int next_token = -1;
            for (size_t i = 0; i < sampled.size(); i++) {
                if (sampled[i] > 0.5f) {  // Use 0.5 threshold for numerical stability
                    next_token = static_cast<int>(i);
                    break;
                }
            }
            
            if (next_token == -1) {
                throw std::runtime_error("Failed to sample valid token");
            }
            
            // Update token stats for diversity tracking
            std::cout << "Updating token statistics..." << std::endl;
            update_token_stats(next_token);
            
            // Add to output
            output_tokens.push_back(next_token);
            tokens_generated++;
            
            // Update context with the new token
            std::string new_token = tokenizer_->decode({next_token});
            current_input_context += new_token;
            
            // Debug output
            std::cout << "Generated token " << tokens_generated << ": '" << new_token << "'" << std::endl;
            std::cout << "Updated context: '" << current_input_context << "'" << std::endl;
            
            // Check for end of sequence token
            if (next_token == eos_token_) {
                std::cout << "End of sequence token generated, stopping generation." << std::endl;
                break;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in generate: " << e.what() << std::endl;
        throw;  // Re-throw the exception after logging
    }
    
    std::cout << "=== Generation complete ===" << std::endl;
    std::cout << "Total tokens generated: " << tokens_generated << std::endl;
    std::cout << "Final sequence length: " << output_tokens.size() << std::endl;
    std::cout << "Final context: '" << current_input_context << "'" << std::endl;
    
    return output_tokens;
}

Matrix Transformer::create_target_distribution(const std::vector<int>& target_tokens, size_t vocab_size) const {
    // Create target distribution with one-hot encoding for the target token
    Matrix distribution(1, vocab_size, 0.0f);
    
    // Set the probability of the target token to 1.0
    if (!target_tokens.empty()) {
        int target_token = target_tokens.back();  // Use the last token as target
        if (target_token >= 0 && static_cast<size_t>(target_token) < vocab_size) {
            distribution(0, target_token) = 1.0f;
        }
    }
    
    return distribution;
}

float Transformer::train_step(const std::vector<int>& inputs, const std::vector<int>& targets, float learning_rate) {
    // 1. Forward pass to get logits and auxiliary loss
    TransformerOutput output = forward(inputs);

    // 2. Compute main cross-entropy loss
    Matrix target_distribution = create_target_distribution(targets, tokenizer_->vocab_size());
    float main_loss = Utils::compute_loss(output.logits, target_distribution);

    // 3. Compute total loss
    float total_loss = main_loss + output.aux_loss;

    // 4. Backward pass
    // The 'backward' method should be adapted to start from the total loss or handle the two loss components.
    // For now, we'll use the existing backward pass which starts from the logits.
    // A more advanced implementation would compute gradients for both losses and add them.
    backward(output.logits, target_distribution, learning_rate); 

    // 5. Update parameters
    update_parameters(learning_rate);

    return total_loss;
}

void Transformer::set_training(bool mode) {
    training = mode;
    // Set training mode for all components that need it
    for (auto& layer : layers) {
        layer->set_training(mode);
    }
    if (lm_head) {
        lm_head->set_training(mode);
    }
}


