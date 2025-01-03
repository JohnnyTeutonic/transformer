#include "../include/transformer.hpp"
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../include/cuda/cublas_check.cuh"
#include "../include/cuda/cuda_check.cuh"
#endif
#include "../include/logger.hpp"
#include <fstream>
#include <iostream>
#include <omp.h>
#include <stdexcept>

#ifdef USE_CUDA
extern cublasHandle_t cublas_handle;
#endif

// TransformerConfig implementation
TransformerConfig::TransformerConfig(size_t vocab_size, size_t max_seq_length,
                                     size_t hidden_size, size_t num_layers,
                                     size_t num_heads, size_t batch_size,
                                     size_t num_epochs)
    : vocab_size(vocab_size), max_seq_length(max_seq_length),
      hidden_size(hidden_size), num_layers(num_layers), num_heads(num_heads),
      head_dim(hidden_size / num_heads), intermediate_size(4 * hidden_size),
      dropout_prob(0.1f), use_flash_attention(true), use_rope(true),
      use_sliding_window(false), window_size(512), use_gqa(false),
      num_kv_heads(num_heads), use_cuda(true), batch_size(batch_size),
      num_epochs(num_epochs) {
  std::cout << "entering TransformerConfig constructor" << std::endl;
  if (hidden_size % num_heads != 0) {
    throw std::invalid_argument(
        "Hidden size must be divisible by number of heads");
  }
  std::cout << "exiting TransformerConfig constructor" << std::endl;
}

// TransformerLayer implementation
TransformerLayer::TransformerLayer(const TransformerConfig &config, size_t idx)
    : kv_cache(config.max_seq_length), 
      config(config),
      layer_idx(idx) {
  std::cout << "entering TransformerLayer constructor" << std::endl;
  // Initialize attention layer
  self_attention = std::make_unique<MultiHeadAttention>(
      config.hidden_size, config.num_heads, config.head_dim,
      config.dropout_prob, config.use_flash_attention, config.use_rope,
      config.use_sliding_window, config.window_size, config.use_gqa,
      config.num_kv_heads);

  // Initialize layer normalization
  attention_ln = std::make_unique<LayerNorm>(config.hidden_size);
  ffn_ln = std::make_unique<LayerNorm>(config.hidden_size);

  // Initialize feed-forward network
  feed_forward = std::make_unique<FeedForward>(
      config.hidden_size, config.intermediate_size, config.dropout_prob);
  std::cout << "exiting TransformerLayer constructor" << std::endl;
}

Matrix TransformerLayer::forward(const Matrix &input, const AttentionMask &mask,
                               const std::optional<KVCache> &kv_cache) {
    std::cout << "=== TransformerLayer::forward START ===" << std::endl;
    std::cout << "Input matrix shape: " << input.rows() << "x" << input.cols() << std::endl;
    
    // Layer norm before attention
    std::cout << "Applying attention layer normalization..." << std::endl;
    Matrix normalized = attention_ln->forward(input);
    std::cout << "Normalized matrix shape: " << normalized.rows() << "x" << normalized.cols() << std::endl;
    
    // Cache the normalized input for backward pass
    std::cout << "Caching normalized input for layer " << layer_idx << std::endl;
    GradientCheckpoint::cache_activation(std::to_string(layer_idx), normalized);
    
    // Self attention
    std::cout << "Applying self attention..." << std::endl;
    Matrix attention_output;
    if (config.use_flash_attention) {
        std::cout << "=== Using FLASH attention for forward pass ===" << std::endl;
        Matrix Q = matmul(normalized, self_attention->query_proj);
        Matrix K = matmul(normalized, self_attention->key_proj);
        Matrix V = matmul(normalized, self_attention->value_proj);
        attention_output = self_attention->flash_attention(Q, K, V, mask);
    } else {
        std::cout << "=== Using STANDARD attention for forward pass ===" << std::endl;
        attention_output = self_attention->forward(normalized, mask, kv_cache);
    }
    std::cout << "Attention output shape: " << attention_output.rows() << "x" << attention_output.cols() << std::endl;
    
    // Validate dimensions before adding residual
    if (attention_output.rows() != input.rows() || attention_output.cols() != input.cols()) {
        std::cout << "Dimension mismatch! Reshaping attention output..." << std::endl;
        // Create properly sized output
        Matrix reshaped_output(input.rows(), input.cols());
        
        // Copy values, handling potential size differences
        for (size_t i = 0; i < input.rows(); ++i) {
            for (size_t j = 0; j < input.cols(); ++j) {
                // Use modulo to handle cases where attention_output is smaller
                reshaped_output(i, j) = attention_output(i % attention_output.rows(), 
                                                       j % attention_output.cols());
            }
        }
        attention_output = reshaped_output;
    }
    
    // Scale residual connection to prevent value explosion
    const float residual_scale = 0.5f;
    std::cout << "Scaling residual connection with factor " << residual_scale << std::endl;
    for(size_t i = 0; i < attention_output.size(); i++) {
        attention_output.data()[i] *= residual_scale;
    }
    
    // Now dimensions should match for residual connection
    std::cout << "Adding first residual connection..." << std::endl;
    Matrix residual1 = attention_output + input;
    std::cout << "First residual shape: " << residual1.rows() << "x" << residual1.cols() << std::endl;
    
    // Layer norm before feed forward
    std::cout << "Applying feed forward layer normalization..." << std::endl;
    Matrix ffn_normalized = ffn_ln->forward(residual1);
    std::cout << "FFN normalized shape: " << ffn_normalized.rows() << "x" << ffn_normalized.cols() << std::endl;
    
    // Cache the normalized input for feed forward backward pass
    std::cout << "Caching FFN normalized input for layer " << layer_idx << std::endl;
    GradientCheckpoint::cache_activation(std::to_string(layer_idx) + "_ffn", ffn_normalized);
    std::cout << "Cached FFN activation successfully" << std::endl;
    
    // Feed forward
    std::cout << "Applying feed forward network..." << std::endl;
    Matrix ffn_output = feed_forward->forward(ffn_normalized);
    std::cout << "FFN output shape: " << ffn_output.rows() << "x" << ffn_output.cols() << std::endl;
    
    // Scale second residual connection
    std::cout << "Scaling second residual connection..." << std::endl;
    for(size_t i = 0; i < ffn_output.size(); i++) {
        ffn_output.data()[i] *= residual_scale;
    }
    std::cout << "Scaled FFN output" << std::endl;
    
    std::cout << "Adding second residual connection..." << std::endl;
    Matrix residual2 = ffn_output + residual1;
    std::cout << "Second residual shape: " << residual2.rows() << "x" << residual2.cols() << std::endl;
    
    std::cout << "=== TransformerLayer::forward END ===" << std::endl;
    return residual2;
}

void TransformerLayer::clear_cache() { 
  std::cout << "entering TransformerLayer::clear_cache" << std::endl;
  kv_cache.clear(); 
  std::cout << "exiting TransformerLayer::clear_cache" << std::endl;
}

void TransformerLayer::convert_to_fp16() {
  std::cout << "entering TransformerLayer::convert_to_fp16" << std::endl;
#ifdef USE_CUDA
  if (self_attention) {
    auto weights = self_attention->get_weights();
    for (auto &weight : weights) {
      HalfPrecisionTraining::convert_to_fp16(weight);
    }
  }
  if (feed_forward) {
    auto weights = feed_forward->get_weights();
    for (auto &weight : weights) {
      HalfPrecisionTraining::convert_to_fp16(weight);
    }
  }
#endif
  std::cout << "exiting TransformerLayer::convert_to_fp16" << std::endl;
}

// Transformer implementation
Transformer::Transformer(const TransformerConfig &config) : config(config) {
  std::cout << "\n=== Transformer::constructor START ===" << std::endl;
  
  // Print configuration
  std::cout << "Configuration:" << std::endl;
  std::cout << "- Vocabulary size: " << config.vocab_size << std::endl;
  std::cout << "- Max sequence length: " << config.max_seq_length << std::endl;
  std::cout << "- Hidden size: " << config.hidden_size << std::endl;
  std::cout << "- Number of layers: " << config.num_layers << std::endl;
  std::cout << "- Number of heads: " << config.num_heads << std::endl;
  std::cout << "- Head dimension: " << config.head_dim << std::endl;
  std::cout << "- Intermediate size: " << config.intermediate_size << std::endl;
  std::cout << "- Dropout probability: " << config.dropout_prob << std::endl;
  std::cout << "=== ATTENTION CONFIGURATION ===" << std::endl;
  std::cout << "- Flash attention enabled: " << std::boolalpha << config.use_flash_attention << std::endl;
  std::cout << "- Flash attention setting from config: " << std::boolalpha << this->config.use_flash_attention << std::endl;
  std::cout << "- Use RoPE: " << config.use_rope << std::endl;
  std::cout << "- Use sliding window: " << config.use_sliding_window << std::endl;
  std::cout << "- Window size: " << config.window_size << std::endl;
  std::cout << "- Use GQA: " << config.use_gqa << std::endl;
  std::cout << "- Number of KV heads: " << config.num_kv_heads << std::endl;
  std::cout << "- Use CUDA: " << config.use_cuda << std::endl;

  if (config.use_cuda) {
    #ifdef USE_CUDA
    std::cout << "\nInitializing CUDA..." << std::endl;
    initialize_cuda();
    cuda_initialized = true;
    std::cout << "CUDA initialization complete" << std::endl;
    #endif
  }

  // Initialize token embedding with memory pooling
  std::cout << "\nInitializing token embedding..." << std::endl;
  token_embedding = std::make_unique<TokenEmbedding>(config.vocab_size, config.hidden_size);
  std::cout << "Token embedding initialized" << std::endl;

  // Initialize positional encoding
  std::cout << "\nInitializing positional encoding..." << std::endl;
  pos_encoding = std::make_unique<PositionalEncoding>(config.max_seq_length, config.hidden_size);
  std::cout << "Positional encoding initialized" << std::endl;

  // Initialize transformer layers
  std::cout << "\nInitializing transformer layers..." << std::endl;
  layers.reserve(config.num_layers);
  for (size_t i = 0; i < config.num_layers; ++i) {
    std::cout << "Creating layer " << i << "..." << std::endl;
    layers.push_back(TransformerLayer::create(config, i));
    std::cout << "Layer " << i << " created" << std::endl;
  }
  std::cout << "All layers initialized" << std::endl;

  // Initialize final layer normalization
  std::cout << "\nInitializing final layer normalization..." << std::endl;
  final_ln = std::make_unique<LayerNorm>(config.hidden_size);
  std::cout << "Final layer normalization initialized" << std::endl;

  // Initialize language model head
  std::cout << "\nInitializing language model head..." << std::endl;
  lm_head = std::make_unique<LanguageModelHead>(config.hidden_size, config.vocab_size);
  std::cout << "Language model head initialized" << std::endl;

  // Enable half-precision training if configured
  if (config.use_fp16) {
    std::cout << "\nConverting to FP16..." << std::endl;
    for (auto &layer : layers) {
      layer->convert_to_fp16();
    }
    std::cout << "FP16 conversion complete" << std::endl;
  }

  std::cout << "=== Transformer::constructor END ===\n" << std::endl;
}

Matrix Transformer::forward(const std::vector<int> &input_tokens, bool use_cache) {
  std::cout << "\n=== Transformer::forward START ===" << std::endl;
  
  try {
    // Print input information
    std::cout << "Input:" << std::endl;
    std::cout << "- Number of tokens: " << input_tokens.size() << std::endl;
    std::cout << "- Use cache: " << std::boolalpha << use_cache << std::endl;
    
    if (config.use_cuda) {
      std::cout << "\nUsing CUDA for forward pass" << std::endl;
      return forward_cuda(input_tokens, use_cache);
    }

    // Allocate memory for embeddings
    std::cout << "\nAllocating memory for embeddings..." << std::endl;
    size_t embed_size = input_tokens.size() * config.hidden_size;
    float *embed_data = MemoryPool::allocate_static(embed_size * sizeof(float));
    Matrix embeddings(input_tokens.size(), config.hidden_size, embed_data);
    std::cout << "Embeddings matrix allocated: " << embeddings.rows() << "x" << embeddings.cols() << std::endl;

    // Get embeddings using CUDA
    std::cout << "\nComputing embeddings using CUDA..." << std::endl;
    token_embedding->forward_cuda(input_tokens, embeddings);
    std::cout << "CUDA embeddings computed" << std::endl;
    
    // Initialize hidden states with embeddings before processing layers
    hidden_states = embeddings;
    std::cout << "Initialized hidden states with embeddings: " << hidden_states.rows() << "x" << hidden_states.cols() << std::endl;
    
    // Add positional encodings
    std::cout << "\nAdding positional encodings..." << std::endl;
    if (input_tokens.empty()) {
        throw std::runtime_error("Cannot create position IDs for empty input");
    }
    
    size_t seq_length = input_tokens.size();
    Matrix position_ids(seq_length, 1);
    
    // Fill position IDs
    #pragma omp parallel for
    for (size_t i = 0; i < seq_length; ++i) {
        position_ids(i, 0) = static_cast<float>(i);
    }
    
    // Get positional encodings and add them to hidden states
    Matrix pos_encodings = pos_encoding->forward(position_ids);
    if (pos_encodings.rows() == hidden_states.rows() && pos_encodings.cols() == hidden_states.cols()) {
        hidden_states += pos_encodings;
    } else {
        throw std::runtime_error("Positional encoding dimensions do not match hidden states");
    }
    
    // Create attention mask
    AttentionMask mask = use_cache ? AttentionMask() : AttentionMask::create_causal_mask(seq_length);
    
    // Process through transformer layers
    for (size_t i = 0; i < layers.size(); ++i) {
        // Save activation for gradient checkpointing
        GradientCheckpoint::save_activation(hidden_states, i);
        
        std::cout << "\nLayer " << i << " configuration:" << std::endl;
        std::cout << "- Flash attention enabled: " << std::boolalpha << config.use_flash_attention << std::endl;
        
        // Forward through layer
        if (config.use_flash_attention) {
            std::cout << "=== Using FLASH attention for layer " << i << " ===" << std::endl;
            Matrix Q = matmul(hidden_states, layers[i]->self_attention->query_proj);
            std::cout << "Q matrix shape: " << Q.rows() << "x" << Q.cols() << std::endl;
            Matrix K = matmul(hidden_states, layers[i]->self_attention->key_proj);
            std::cout << "K matrix shape: " << K.rows() << "x" << K.cols() << std::endl;
            Matrix V = matmul(hidden_states, layers[i]->self_attention->value_proj);
            std::cout << "V matrix shape: " << V.rows() << "x" << V.cols() << std::endl;
            
            // Use flash attention algorithm
            hidden_states = layers[i]->self_attention->flash_attention(Q, K, V, mask);
            std::cout << "Flash attention output shape: " << hidden_states.rows() << "x" << hidden_states.cols() << std::endl;
        } else {
            std::cout << "=== Using STANDARD attention for layer " << i << " ===" << std::endl;
            hidden_states = layers[i]->forward(hidden_states, mask);
        }
        
        if (config.use_fp16) {
            HalfPrecisionTraining::convert_to_fp16(hidden_states);
        }
    }

    // Final layer normalization using CUDA
    std::cout << "\nApplying final layer normalization..." << std::endl;
    hidden_states = final_ln->forward(hidden_states);
    std::cout << "Final normalization complete" << std::endl;
    std::cout << "Hidden states dimensions: " << hidden_states.rows() << "x" << hidden_states.cols() << std::endl;

    // Project to logits using language model head
    std::cout << "\nComputing logits using language model head..." << std::endl;
    if (!lm_head) {
        throw std::runtime_error("Language model head not initialized");
    }
    Matrix logits = lm_head->forward(hidden_states);
    std::cout << "Logits computed with dimensions: " << logits.rows() << "x" << logits.cols() << std::endl;

    // Make a copy and cleanup
    std::cout << "\nCleaning up..." << std::endl;
    Matrix result = logits;
    MemoryPool::deallocate_static(embed_data, embed_size * sizeof(float));
    std::cout << "Memory cleanup complete" << std::endl;

    std::cout << "=== Transformer::forward END ===\n" << std::endl;
    return result;
    
  } catch (const std::exception& e) {
    std::cerr << "\nERROR in transformer forward pass: " << e.what() << std::endl;
    std::cerr << "=== Transformer::forward FAILED ===\n" << std::endl;
    throw;
  }
}

void Transformer::train(const std::vector<std::vector<int>> &input_tokens,
                        const std::vector<std::vector<int>> &target_tokens,
                        size_t num_epochs, float learning_rate) {
  std::cout << "entering Transformer::train" << std::endl;
  const size_t batch_size = config.batch_size;  // Use batch_size from config

  for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
// Process batches
#pragma omp parallel for
    for (size_t i = 0; i < input_tokens.size(); i += batch_size) {
      size_t batch_end = std::min(i + batch_size, input_tokens.size());

      // Create batch
      std::vector<std::vector<int>> input_batch(
          input_tokens.begin() + i, input_tokens.begin() + batch_end);
      std::vector<std::vector<int>> target_batch(
          target_tokens.begin() + i, target_tokens.begin() + batch_end);

      // Forward pass
      std::vector<Matrix> activations;
      Matrix logits = forward(input_batch[0], &activations);

      // Compute loss and gradients
      Matrix loss_grad = compute_loss_gradients(logits, target_batch[0]);

      // Backward pass
      backward_pass(activations, loss_grad);
      std::cout << "backward pass done" << std::endl;
      // IMPORTANT TO UPDATE PARAMETERS USING OPTIMIZER
      update_parameters(learning_rate);
    }
  }
  std::cout << "exiting Transformer::train" << std::endl;
}

Matrix Transformer::compute_loss_gradients(const Matrix &logits,
                                           const std::vector<int> &targets) {
  std::cout << "entering Transformer::compute_loss_gradients" << std::endl;
  const size_t batch_size = logits.rows();
  const size_t vocab_size = logits.cols();
  Matrix gradients(batch_size, vocab_size);

  // For each sequence position
  for (size_t i = 0; i < batch_size; ++i) {
    // Compute softmax probabilities
    std::vector<float> probs(vocab_size);
    float max_logit = logits(i, 0);  // Initialize with first value
    std::cout << "logit value: " << logits(i, 0) << std::endl;

    // Find max logit for numerical stabilxity
    for (size_t j = 0; j < vocab_size; ++j) {
      max_logit = std::max(max_logit, logits(i, j));
    }

    float sum = 0.0f;
    // Add numerical stability to softmax computation
    const float epsilon = 1e-10f;
    for (size_t j = 0; j < vocab_size; ++j) {
      // Clamp the exponent to prevent overflow
      float exp_val = std::min(logits(i, j) - max_logit, 88.0f);
      probs[j] = std::exp(exp_val);
      sum += probs[j];
    }
    // Prevent division by zero
    sum = std::max(sum, epsilon);

    // Normalize and compute gradients
    for (size_t j = 0; j < vocab_size; ++j) {
      probs[j] /= sum;
      // Gradient is (probability - 1) for correct class, probability for others
      gradients(i, j) = probs[j];
    }
    gradients(i, targets[i]) -= 1.0f; // Subtract 1 from target class
  }

  std::cout << "exiting Transformer::compute_loss_gradients" << std::endl;
  return gradients;
}

void Transformer::backward_pass(const std::vector<Matrix> &activations,
                                const Matrix &loss_grad) {
  std::cout << "entering Transformer::backward_pass" << std::endl;
  Matrix current_grad = loss_grad;
  // Convert gradients to FP16 if enabled
  if (config.use_fp16) {
    HalfPrecisionTraining::convert_to_fp16(current_grad);
  }
  std::cout << "backward pass using cuda done" << std::endl;

  std::cout << "iterating through layers in reverse order" << std::endl;
  // Backward through layers in reverse order
  for (int i = layers.size() - 1; i >= 0; --i) {
    // Retrieve checkpointed activation
    Matrix activation = GradientCheckpoint::get_activation(i);

    if (config.use_cuda) {  
      std::cout << "backward pass using cuda" << std::endl;
      current_grad = layers[i]->backward_cuda(current_grad, activation);
      std::cout << "backward pass using cuda done" << std::endl;
    } else {
      std::cout << "backward pass using cpu" << std::endl;
      current_grad = layers[i]->backward(current_grad, activation);
      std::cout << "backward pass using cpu done" << std::endl;
    }

    // Convert gradients back to FP32 if needed
    if (config.use_fp16) {
      HalfPrecisionTraining::convert_to_fp32(current_grad);
    }
  }
}

void Transformer::update_parameters(float learning_rate) {
    std::cout << "\n=== Transformer::update_parameters START ===" << std::endl;
    
    // Get all trainable parameters and their gradients
    auto& params = this->parameters();
    auto& grads = this->parameter_gradients();
    
    if (params.size() != grads.size()) {
        throw std::runtime_error("Parameter and gradient size mismatch: " + 
                               std::to_string(params.size()) + " vs " + 
                               std::to_string(grads.size()));
    }
    
    std::cout << "Updating " << params.size() << " parameter matrices" << std::endl;
    
    // Implement Adam optimizer
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float epsilon = 1e-8f;
    
    // Initialize momentum and velocity if not already done
    if (momentum_buffers.empty()) {
        momentum_buffers.resize(params.size());
        velocity_buffers.resize(params.size());
        for (size_t i = 0; i < params.size(); i++) {
            momentum_buffers[i] = Matrix(params[i].rows(), params[i].cols(), 0.0f);
            velocity_buffers[i] = Matrix(params[i].rows(), params[i].cols(), 0.0f);
        }
    }
    
    // Update each parameter matrix
    for (size_t i = 0; i < params.size(); i++) {
        Matrix& param = params[i];
        const Matrix& grad = grads[i];
        Matrix& momentum = momentum_buffers[i];
        Matrix& velocity = velocity_buffers[i];
        
        if (param.rows() != grad.rows() || param.cols() != grad.cols()) {
            throw std::runtime_error("Parameter and gradient dimension mismatch at index " + 
                                   std::to_string(i));
        }
        
        std::cout << "Updating parameter matrix " << i << " with shape: " 
                  << param.rows() << "x" << param.cols() << std::endl;
        
        // Update rule with Adam optimizer
        for (size_t j = 0; j < param.size(); j++) {
            // Update momentum
            momentum.data()[j] = beta1 * momentum.data()[j] + (1.0f - beta1) * grad.data()[j];
            
            // Update velocity
            velocity.data()[j] = beta2 * velocity.data()[j] + 
                               (1.0f - beta2) * grad.data()[j] * grad.data()[j];
            
            // Compute bias-corrected estimates
            float m_hat = momentum.data()[j] / (1.0f - std::pow(beta1, update_step + 1));
            float v_hat = velocity.data()[j] / (1.0f - std::pow(beta2, update_step + 1));
            
            // Update parameter
            param.data()[j] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
            
            // Add gradient clipping
            if (param.data()[j] > 5.0f) param.data()[j] = 5.0f;
            if (param.data()[j] < -5.0f) param.data()[j] = -5.0f;
        }
    }
    
    update_step++;
    std::cout << "=== Transformer::update_parameters END ===\n" << std::endl;
}

void Transformer::save_model(const std::string &path) const {
  std::ofstream os(path, std::ios::binary);
  if (!os) {
    throw std::runtime_error("Failed to open file for saving model");
  }

  // Save config
  os.write(reinterpret_cast<const char *>(&config), sizeof(config));

  // Save embeddings
  token_embedding->save(os);
  pos_encoding->save(os);

  // Save layers
  for (const auto &layer : layers) {
    layer->save(os);
  }

  // Save final layer norm
  final_ln->save(os);
}

Transformer Transformer::load_model(const std::string &path) {
  std::ifstream is(path, std::ios::binary);
  if (!is) {
    throw std::runtime_error("Failed to open file for loading model");
  }

  // Load config
  TransformerConfig config;
  is.read(reinterpret_cast<char *>(&config), sizeof(config));

  // Create transformer with loaded config
  Transformer transformer(config);

  // Load embeddings
  transformer.token_embedding = TokenEmbedding::load(is);
  transformer.pos_encoding = PositionalEncoding::load(is);

  // Load layers
  transformer.layers.clear();
  for (size_t i = 0; i < config.num_layers; ++i) {
    auto layer = TransformerLayer::create(config, i);
    layer->load(is);
    transformer.layers.push_back(std::move(layer));
  }

  // Load final layer norm
  transformer.final_ln = LayerNorm::load(is);

  return transformer;
}

void Transformer::clear_kv_cache() {
  for (auto &layer : layers) {
    layer->clear_cache();
  }
}

Matrix Transformer::backward(const Matrix &grad, const Matrix &activation,
                             size_t layer_idx) {
  if (layer_idx >= layers.size()) {
    throw std::out_of_range("Layer index out of range");
  }

  // Compute gradients for the current layer
  Matrix layer_grad = grad;

  // Backward through layer normalization
  if (layer_idx == layers.size() - 1) {
    layer_grad = final_ln->backward(layer_grad, activation);
  }

  // Backward through transformer layer
  // Note: This would require implementing backward methods in TransformerLayer
  // and its components (attention, feed-forward, etc.)

  return layer_grad;
}

Matrix Transformer::backward_cuda(const Matrix &grad, const Matrix &activation,
                                  size_t layer_idx) {
#ifdef USE_CUDA
  if (layer_idx >= layers.size()) {
    throw std::out_of_range("Layer index out of range");
  }

  Matrix current_grad = grad;

  // Convert gradients to FP16 if enabled
  if (config.use_fp16) {
    HalfPrecisionTraining::convert_to_fp16(current_grad);
  }

  // Backward through final layer norm using CUDA
  if (layer_idx == layers.size() - 1) {
    current_grad = final_ln->backward_cuda(current_grad, activation);
  }

  // Backward through layer using CUDA
  Matrix layer_grad =
      layers[layer_idx]->backward_cuda(current_grad, activation);

  // Convert gradients back to FP32 if needed
  if (config.use_fp16) {
    HalfPrecisionTraining::convert_to_fp32(layer_grad);
  }

  return layer_grad;
#else
  return backward(grad, activation, layer_idx);
#endif
}

std::vector<Matrix> &Transformer::parameters() {
  std::cout << "entering Transformer::parameters" << std::endl;
  static std::vector<Matrix> all_params;
  all_params.clear();

  // Add embedding parameters
  all_params.push_back(token_embedding->get_embedding_table());
  std::cout << "Adding attention parameters" << std::endl;
  // Add layer parameters
  for (auto &layer : layers) {
    // Add attention parameters
    all_params.push_back(layer->self_attention->query_proj);
    all_params.push_back(layer->self_attention->key_proj);
    all_params.push_back(layer->self_attention->value_proj);
    all_params.push_back(layer->self_attention->output_proj);
    std::cout << "Adding layer norm parameters" << std::endl;
    // Add layer norm parameters - convert Vector to Matrix
    const Vector &gamma = layer->attention_ln->get_gamma();
    const Vector &beta = layer->attention_ln->get_beta();
    Matrix gamma_matrix(1, gamma.size());
    Matrix beta_matrix(1, beta.size());
    for (size_t i = 0; i < gamma.size(); ++i) {
      gamma_matrix(0, i) = gamma[i];
      beta_matrix(0, i) = beta[i];
    }
    all_params.push_back(gamma_matrix);
    all_params.push_back(beta_matrix);

    // Add feed forward parameters
    all_params.push_back(layer->feed_forward->w1);
    all_params.push_back(layer->feed_forward->w2);
    std::cout << "Added feed forward parameters" << std::endl;
    // Convert feed forward biases to matrices
    std::cout << "Converting feed forward biases to matrices" << std::endl;
    Matrix b1_matrix(1, layer->feed_forward->b1.size());
    Matrix b2_matrix(1, layer->feed_forward->b2.size());
    for (size_t i = 0; i < layer->feed_forward->b1.size(); ++i) {
      b1_matrix(0, i) = layer->feed_forward->b1[i];
    }
    for (size_t i = 0; i < layer->feed_forward->b2.size(); ++i) {
      b2_matrix(0, i) = layer->feed_forward->b2[i];
    }
    all_params.push_back(b1_matrix);
    all_params.push_back(b2_matrix);

    // Add final layer norm parameters
    std::cout << "Adding final layer norm parameters" << std::endl;
    const Vector &ffn_gamma = layer->ffn_ln->get_gamma();
    const Vector &ffn_beta = layer->ffn_ln->get_beta();
    Matrix ffn_gamma_matrix(1, ffn_gamma.size());
    Matrix ffn_beta_matrix(1, ffn_beta.size());
    for (size_t i = 0; i < ffn_gamma.size(); ++i) {
      ffn_gamma_matrix(0, i) = ffn_gamma[i];
      ffn_beta_matrix(0, i) = ffn_beta[i];
    }
    all_params.push_back(ffn_gamma_matrix);
    all_params.push_back(ffn_beta_matrix);
  }

  // Add final layer norm parameters
  std::cout << "Adding final layer norm parameters" << std::endl;
  const Vector &final_gamma = final_ln->get_gamma();
  const Vector &final_beta = final_ln->get_beta();
  Matrix final_gamma_matrix(1, final_gamma.size());
  Matrix final_beta_matrix(1, final_beta.size());
  for (size_t i = 0; i < final_gamma.size(); ++i) {
    final_gamma_matrix(0, i) = final_gamma[i];
    final_beta_matrix(0, i) = final_beta[i];
  }
  all_params.push_back(final_gamma_matrix);
  all_params.push_back(final_beta_matrix);
  std::cout << "Exiting Transformer::parameters" << std::endl;
  return all_params;
}

void Transformer::save(std::ostream &os) const {
  // Save config
  os.write(reinterpret_cast<const char *>(&config), sizeof(config));

  // Save embeddings
  std::cout << "Saving embeddings" << std::endl;
  token_embedding->save(os);
  pos_encoding->save(os);

  // Save layers
  for (const auto &layer : layers) {
    std::cout << "Saving layer" << std::endl;
    layer->save(os);
  }

  // Save final layer norm
  std::cout << "Saving final layer norm" << std::endl;
  final_ln->save(os);
}

void Transformer::load(std::istream &is) {
  // Read config
  size_t vocab_size, max_seq_length, hidden_size, num_layers, num_heads, batch_size;
  std::cout << "Reading config" << std::endl;
  is.read(reinterpret_cast<char *>(&vocab_size), sizeof(vocab_size));
  is.read(reinterpret_cast<char *>(&max_seq_length), sizeof(max_seq_length));
  is.read(reinterpret_cast<char *>(&hidden_size), sizeof(hidden_size));
  is.read(reinterpret_cast<char *>(&num_layers), sizeof(num_layers));
  is.read(reinterpret_cast<char *>(&num_heads), sizeof(num_heads));
  is.read(reinterpret_cast<char *>(&batch_size), sizeof(batch_size));

  TransformerConfig config(vocab_size, max_seq_length, hidden_size, num_layers,
                         num_heads, batch_size);

  // Load layers
  layers.clear();
  for (size_t i = 0; i < num_layers; ++i) {
    auto layer = TransformerLayer::create(config, i);
    std::cout << "Loading layer" << std::endl;
    layer->load(is);
    layers.push_back(std::move(layer));
  }

  // Load embeddings and final layer norm
  token_embedding = std::make_unique<TokenEmbedding>(vocab_size, hidden_size);
  std::cout << "Loading embeddings" << std::endl;
  token_embedding->load(is);
  
  final_ln = std::make_unique<LayerNorm>(hidden_size);
  std::cout << "Loading final layer norm" << std::endl;
  final_ln->load(is);
}

Matrix TransformerLayer::backward(const Matrix& grad_output, 
                                const Matrix& input,
                                const Matrix& target_distribution) {
    try {
        std::cout << "entering TransformerLayer::backward" << std::endl;
        // Get cached activations
        std::string ffn_key = std::to_string(layer_idx) + "_ffn";
        if (!GradientCheckpoint::has_activation(ffn_key)) {
            throw std::runtime_error("Missing feed forward activation cache");
        }
        std::cout << "Getting feed forward activation cache" << std::endl;
        Matrix ffn_normalized = GradientCheckpoint::get_activation(ffn_key);
        
        // Feed forward backward
        Matrix d_ffn;
        // Ensure gradient matches hidden_size dimension
        Matrix reshaped_grad = grad_output;
        if (grad_output.cols() != config.hidden_size) {
            reshaped_grad = Matrix(grad_output.rows(), config.hidden_size);
            for (size_t i = 0; i < grad_output.rows(); ++i) {
                for (size_t j = 0; j < config.hidden_size; ++j) {
                    reshaped_grad(i, j) = grad_output(i, j);
                }
            }
        }
        
        if (config.use_cuda) {
            std::cout << "Feed forward backward using cuda" << std::endl;
            d_ffn = feed_forward->backward_cuda(reshaped_grad, ffn_normalized);
        } else {
            std::cout << "Feed forward backward using cpu" << std::endl;
            d_ffn = feed_forward->backward(reshaped_grad, ffn_normalized);
        }
        std::cout << "Feed forward backward" << std::endl;
        Matrix d_ln2 = ffn_ln->backward(d_ffn, input);
        std::cout << "Feed forward layer norm backward" << std::endl;
        
        // Attention backward
        std::string attn_key = std::to_string(layer_idx);
        if (!GradientCheckpoint::has_activation(attn_key)) {
            throw std::runtime_error("Missing attention activation cache");
        }
        std::cout << "Getting attention activation cache" << std::endl;
        Matrix attn_normalized = GradientCheckpoint::get_activation(attn_key);
        
        Matrix d_residual1 = d_ln2;
        Matrix d_attn;
        if (config.use_flash_attention) {
            // Use flash attention backward pass
            std::cout << "Using flash attention for backward pass" << std::endl;
            Matrix Q = matmul(attn_normalized, self_attention->query_proj);
            Matrix K = matmul(attn_normalized, self_attention->key_proj);
            Matrix V = matmul(attn_normalized, self_attention->value_proj);
            
            // Use flash attention algorithm
            Matrix attention_output = self_attention->flash_attention(Q, K, V, AttentionMask());
            d_attn = self_attention->backward(d_residual1, attn_normalized, attention_output);
        } else {
            // Use standard attention backward pass
            std::cout << "Using standard attention for backward pass" << std::endl;
            d_attn = self_attention->backward(d_residual1, attn_normalized, target_distribution);
        }
        std::cout << "Attention backward" << std::endl;

        return d_attn;
    } catch (const std::exception& e) {
        std::cerr << "Error in transformer backward pass: " << e.what() << std::endl;
        throw;
    }
}

Matrix Transformer::forward_cuda(const std::vector<int> &input_tokens, bool use_cache) {
#ifdef USE_CUDA
    std::cout << "\n=== Transformer::forward_cuda START ===" << std::endl;
    
    try {
        // Validate input
        if (input_tokens.empty()) {
            throw std::runtime_error("Input tokens vector cannot be empty");
        }
        
        // Allocate memory for embeddings
        std::cout << "\nAllocating memory for embeddings..." << std::endl;
        size_t embed_size = input_tokens.size() * config.hidden_size;
        float *embed_data = MemoryPool::allocate_static(embed_size * sizeof(float));
        Matrix embeddings(input_tokens.size(), config.hidden_size, embed_data);
        std::cout << "Embeddings matrix allocated: " << embeddings.rows() << "x" << embeddings.cols() << std::endl;
        
        // Get embeddings using CUDA
        std::cout << "\nComputing embeddings using CUDA..." << std::endl;
        token_embedding->forward_cuda(input_tokens, embeddings);
        std::cout << "CUDA embeddings computed" << std::endl;
        
        // Initialize hidden states with embeddings before processing layers
        hidden_states = embeddings;
        std::cout << "Initialized hidden states with embeddings: " << hidden_states.rows() << "x" << hidden_states.cols() << std::endl;
        
        // Add positional encodings
        std::cout << "\nAdding positional encodings..." << std::endl;
        size_t seq_length = input_tokens.size();
        Matrix position_ids(seq_length, 1);
        
        // Fill position IDs
        #pragma omp parallel for
        for (size_t i = 0; i < seq_length; ++i) {
            position_ids(i, 0) = static_cast<float>(i);
        }
        
        // Get positional encodings and add them to hidden states
        Matrix pos_encodings = pos_encoding->forward(position_ids);
        if (pos_encodings.rows() == hidden_states.rows() && pos_encodings.cols() == hidden_states.cols()) {
            hidden_states += pos_encodings;
        } else {
            throw std::runtime_error("Positional encoding dimensions do not match hidden states");
        }
        
        // Create attention mask
        AttentionMask mask = use_cache ? AttentionMask() : AttentionMask::create_causal_mask(seq_length);
        
        // Process through transformer layers
        for (size_t i = 0; i < layers.size(); ++i) {
            // Save activation for gradient checkpointing
            GradientCheckpoint::save_activation(hidden_states, i);
            
            // Forward through layer
            hidden_states = layers[i]->forward(hidden_states, mask);
            
            if (config.use_fp16) {
                HalfPrecisionTraining::convert_to_fp16(hidden_states);
            }
        }

        // Final layer normalization using CUDA
        std::cout << "\nApplying final layer normalization..." << std::endl;
        hidden_states = final_ln->forward(hidden_states);
        std::cout << "Final normalization complete" << std::endl;
        std::cout << "Hidden states dimensions: " << hidden_states.rows() << "x" << hidden_states.cols() << std::endl;

        // Project to logits using language model head
        std::cout << "\nComputing logits using language model head..." << std::endl;
        if (!lm_head) {
            throw std::runtime_error("Language model head not initialized");
        }
        Matrix logits = lm_head->forward(hidden_states);
        std::cout << "Logits computed with dimensions: " << logits.rows() << "x" << logits.cols() << std::endl;

        // Make a copy and cleanup
        std::cout << "\nCleaning up..." << std::endl;
        Matrix result = logits;
        MemoryPool::deallocate_static(embed_data, embed_size * sizeof(float));
        std::cout << "Memory cleanup complete" << std::endl;
        
        std::cout << "=== Transformer::forward_cuda END ===\n" << std::endl;
        return result;
        
    } catch (const std::exception& e) {
        std::cerr << "\nERROR in transformer CUDA forward pass: " << e.what() << std::endl;
        std::cerr << "=== Transformer::forward_cuda FAILED ===\n" << std::endl;
        throw;
    }
#else
    throw std::runtime_error("CUDA support not enabled");
#endif
}

Matrix TransformerLayer::backward_cuda(const Matrix &grad,
                                     const Matrix &input) const {
#ifdef USE_CUDA
    std::cout << "entering TransformerLayer::backward_cuda" << std::endl;
    
    // Get cached activations
    std::string ffn_key = std::to_string(layer_idx) + "_ffn";
    if (!GradientCheckpoint::has_activation(ffn_key)) {
        throw std::runtime_error("Missing feed forward activation cache");
    }
    Matrix ffn_normalized = GradientCheckpoint::get_activation(ffn_key);
    
    // Feed forward backward using CUDA
    Matrix d_ffn = feed_forward->backward_cuda(grad, ffn_normalized);
    Matrix d_ln2 = ffn_ln->backward_cuda(d_ffn, input);
    
    // Attention backward using CUDA
    std::string attn_key = std::to_string(layer_idx);
    if (!GradientCheckpoint::has_activation(attn_key)) {
        throw std::runtime_error("Missing attention activation cache");
    }
    Matrix attn_normalized = GradientCheckpoint::get_activation(attn_key);
    
    Matrix d_residual1 = d_ln2;
    Matrix d_attn;
    // Create empty target distribution matrix
    Matrix target_distribution(input.rows(), input.cols(), 0.0f);
    
    if (config.use_flash_attention) {
        // Use flash attention backward pass
        std::cout << "Using flash attention for backward pass" << std::endl;
        Matrix Q = matmul(attn_normalized, self_attention->query_proj);
        Matrix K = matmul(attn_normalized, self_attention->key_proj);
        Matrix V = matmul(attn_normalized, self_attention->value_proj);
        
        // Use flash attention algorithm
        Matrix attention_output = self_attention->flash_attention(Q, K, V, AttentionMask());
        d_attn = self_attention->backward(d_residual1, attn_normalized, attention_output);
    } else {
        // Use standard attention backward pass
        std::cout << "Using standard attention for backward pass" << std::endl;
        d_attn = self_attention->backward(d_residual1, attn_normalized, target_distribution);
    }
    std::cout << "Attention backward" << std::endl;

    return d_attn;
#else
    throw std::runtime_error("CUDA support not enabled");
#endif
}

Transformer::~Transformer() {
  // Disable logging before CUDA cleanup
  Logger::getInstance().disableLogging();
  #ifdef USE_CUDA 
  if (cuda_initialized) {
    cleanup_cuda();
    cuda_initialized = false;
  }
  #endif
}

Transformer::Transformer(const Transformer &other) : config(other.config) {
  // Deep copy token embedding
  token_embedding = std::make_unique<TokenEmbedding>(*other.token_embedding);

  // Deep copy positional encoding
  pos_encoding = std::make_unique<PositionalEncoding>(*other.pos_encoding);

  // Deep copy layers
  layers.reserve(other.layers.size());
  for (const auto &layer : other.layers) {
    auto new_layer = std::make_unique<TransformerLayer>(*layer);
    layers.push_back(std::move(new_layer));
  }

  // Deep copy final layer norm
  final_ln = std::make_unique<LayerNorm>(*other.final_ln);

  // Deep copy language model head if it exists
  if (other.lm_head) {
    lm_head = std::make_unique<LanguageModelHead>(*other.lm_head);
  }
}

Transformer &Transformer::operator=(const Transformer &other) {
  if (this != &other) {
    config = other.config;
    std::cout << "Copying config" << std::endl;
    // Deep copy token embedding
    token_embedding = std::make_unique<TokenEmbedding>(*other.token_embedding);
    std::cout << "Copying token embedding" << std::endl;
    // Deep copy positional encoding
    pos_encoding = std::make_unique<PositionalEncoding>(*other.pos_encoding);
    std::cout << "Copying positional encoding" << std::endl;
    // Deep copy layers
    layers.clear();
    std::cout << "Clearing layers" << std::endl;
    layers.reserve(other.layers.size());
    std::cout << "Reserving layers" << std::endl;
    for (const auto &layer : other.layers) {
      layers.push_back(std::make_unique<TransformerLayer>(*layer));
    }
    std::cout << "Copying final layer norm" << std::endl;
    // Deep copy final layer norm
    final_ln = std::make_unique<LayerNorm>(*other.final_ln);
    std::cout << "Copying language model head" << std::endl;
    // Deep copy language model head if it exists
    if (other.lm_head) {
      lm_head = std::make_unique<LanguageModelHead>(*other.lm_head);
    } else {
      lm_head.reset();
    }
    std::cout << "Exiting operator=" << std::endl;
  }
  return *this;
}

void Transformer::backward(const Matrix &grad_output, const std::vector<int> &input_tokens, float learning_rate) {
    try {
        std::cout << "\n=== Transformer::backward START ===" << std::endl;
        
        // Add gradient scaling factor
        const float grad_scale = 16.0f;  // Start with larger scale to combat vanishing gradients
        Matrix scaled_grad = grad_output;
        
        // Scale up gradients
        for (size_t i = 0; i < scaled_grad.size(); ++i) {
            scaled_grad.data()[i] *= grad_scale;
        }
        
        // Gradient clipping and normalization
        const float max_grad_norm = 1.0f;
        Matrix clipped_grad = scaled_grad;
        
        // Calculate gradient norm
        float grad_norm = 0.0f;
        for (size_t i = 0; i < clipped_grad.size(); ++i) {
            grad_norm += clipped_grad.data()[i] * clipped_grad.data()[i];
        }
        grad_norm = std::sqrt(grad_norm);
        
        // Add dynamic gradient scaling based on norm
        float dynamic_scale = 1.0f;
        if (grad_norm < 0.1f) {  // Gradients too small
            dynamic_scale = 2.0f;
        } else if (grad_norm > 10.0f) {  // Gradients too large
            dynamic_scale = 0.5f;
        }
        
        // Apply dynamic scaling
        for (size_t i = 0; i < clipped_grad.size(); ++i) {
            clipped_grad.data()[i] *= dynamic_scale;
        }

        // Get cached activations for each layer
        std::vector<Matrix> layer_activations;
        layer_activations.reserve(layers.size());
        for (size_t i = 0; i < layers.size(); ++i) {
            if (!GradientCheckpoint::has_activation(std::to_string(i))) {
                throw std::runtime_error("Missing activation cache for layer " + std::to_string(i));
            }
            layer_activations.push_back(GradientCheckpoint::get_activation(std::to_string(i)));
        }

        // Backward through final layer norm
        Matrix current_grad = final_ln->backward(clipped_grad, layer_activations.back());

        // Backward through transformer layers in reverse order
        for (int i = layers.size() - 1; i >= 0; --i) {
            std::cout << "Processing backward pass for layer " << i << std::endl;
            
            // Get the input to this layer from cached activations
            const Matrix& layer_input = layer_activations[i];
            
            // Create target distribution for attention mechanism
            Matrix target_distribution(layer_input.rows(), layer_input.cols(), 0.0f);
            // Fill target distribution based on input tokens if needed
            // This is used by attention mechanism for computing gradients
            
            // Backward through transformer layer
            current_grad = layers[i]->backward(current_grad, layer_input, target_distribution);
            
            // Apply gradient clipping per layer if needed
            float layer_grad_norm = 0.0f;
            for (size_t j = 0; j < current_grad.size(); ++j) {
                layer_grad_norm += current_grad.data()[j] * current_grad.data()[j];
            }
            layer_grad_norm = std::sqrt(layer_grad_norm);
            
            if (layer_grad_norm > max_grad_norm) {
                float scale_factor = max_grad_norm / (layer_grad_norm + 1e-6f);
                for (size_t j = 0; j < current_grad.size(); ++j) {
                    current_grad.data()[j] *= scale_factor;
                }
            }
        }

        // Backward through embeddings
        token_embedding->backward(current_grad, input_tokens);

        update_parameters(learning_rate);

        std::cout << "=== Transformer::backward END ===\n" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\nERROR in transformer backward pass: " << e.what() << std::endl;
        std::cerr << "=== Transformer::backward FAILED ===\n" << std::endl;
        throw;
    }
}

// Add member to store last hidden states for backward pass
Matrix hidden_states;