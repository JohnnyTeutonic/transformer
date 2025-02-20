#pragma once

#include <memory>
#include <vector>
#include <string>
#include <optional>
#include <functional>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <iostream>

#include "matrix.hpp"
#include "embeddings.hpp"  // Includes TokenEmbedding and PositionalEncoding
#include "attention.hpp"
#include "layer_norm.hpp"
#include "feed_forward.hpp"
#include "dropout.hpp"
#include "lm_head.hpp"
#include "config.hpp"
#include "cache.hpp"
#include "components.hpp"
#include "gradient_checkpoint.hpp"
#include "half_precision.hpp"
#include "memory_pool.hpp"
#include "phrase_types.hpp"
#include "tiktoken_tokenizer.hpp"

// Forward declarations
class TransformerLayer;
class LayerNorm;
class LanguageModelHead;
class Dropout;
class KVCache;
class Matrix;
class MultiHeadAttention;
class FeedForward;

// Add helper function declarations
void update_attention_parameters(MultiHeadAttention* attention, float learning_rate, const TransformerConfig& config);
void update_ffn_parameters(FeedForward* ffn, float learning_rate, const TransformerConfig& config);

// Add loss computation declarations
float compute_loss(const Matrix& output, const Matrix& target_distribution);
Matrix compute_loss_gradient(const Matrix& output, const Matrix& target_distribution);

// Make update_parameter_with_clip global functions instead of member functions
void update_parameter_with_clip(Matrix& param, const Matrix& grad, float learning_rate, const TransformerConfig& config);
void update_parameter_with_clip(Vector& param, const Vector& grad, float learning_rate, const TransformerConfig& config);

// Token prediction result structure
struct TokenPrediction {
    size_t token_id;
    std::string token_text;
    float probability;
    float raw_logit;
    PhraseType type;
};

/**
 * @brief A single layer of the Transformer model implementing the standard Transformer architecture.
 * 
 * Each TransformerLayer consists of:
 * - Multi-head self-attention mechanism
 * - Layer normalization for attention
 * - Feed-forward neural network
 * - Layer normalization for feed-forward
 * - Dropout layers for regularization
 * - Key-Value cache for efficient inference
 */
class TransformerLayer {
  private:
    std::unique_ptr<MultiHeadAttention> self_attention;  ///< Multi-head self-attention mechanism
    std::unique_ptr<LayerNorm> attention_ln;            ///< Layer normalization for attention output
    std::unique_ptr<LayerNorm> ffn_ln;                 ///< Layer normalization for feed-forward output
    std::unique_ptr<FeedForward> feed_forward;         ///< Feed-forward neural network
    std::unique_ptr<Dropout> attention_dropout;        ///< Dropout for attention
    std::unique_ptr<Dropout> ffn_dropout;             ///< Dropout for feed-forward
    KVCache kv_cache;                                ///< Cache for key-value pairs in attention
    const TransformerConfig& config;                 ///< Reference to model configuration
    size_t layer_idx;                              ///< Index of this layer in the transformer
    bool training = false;                        ///< Whether the layer is in training mode

  public:
    virtual ~TransformerLayer() = default;
    TransformerLayer() = default;

    /**
     * @brief Constructs a transformer layer with the given configuration and layer index.
     * @param config_ Configuration parameters for the transformer
     * @param idx Index of this layer in the transformer stack
     */
    TransformerLayer(const TransformerConfig& config_, size_t idx);

    /**
     * @brief Performs the forward pass through the transformer layer.
     * @param input Input tensor of shape [batch_size, seq_len, hidden_size]
     * @param mask Attention mask to prevent attending to future tokens
     * @param kv_cache Optional key-value cache for efficient inference
     * @return Output tensor of shape [batch_size, seq_len, hidden_size]
     */
    Matrix forward(const Matrix& input, const AttentionMask& mask,
                   const std::optional<KVCache>& kv_cache = std::nullopt);

    /**
     * @brief Clears all cached states and resets layer components.
     */
    void clear_cache() {
        kv_cache.clear();
        if (self_attention) {
            self_attention->reset_state();
        }
        if (feed_forward) {
            feed_forward->reset_state();
        }
        if (attention_dropout) {
            attention_dropout->reset_mask();
        }
        if (ffn_dropout) {
            ffn_dropout->reset_mask();
        }
    }

    void set_training(bool mode) {
        training = mode;
    }

    /**
     * @brief Saves the layer's parameters to an output stream.
     * @param os Output stream to save to
     */
    void save(std::ostream& os) const {
        self_attention->save(os);
        attention_ln->save(os);
        feed_forward->save(os);
        ffn_ln->save(os);
    }

    /**
     * @brief Creates a new transformer layer with the given configuration.
     * @param config Configuration parameters for the transformer
     * @param idx Index of this layer in the transformer stack
     * @return Unique pointer to the created layer
     */
    static std::unique_ptr<TransformerLayer> create(const TransformerConfig& config, size_t idx) {
        return std::make_unique<TransformerLayer>(config, idx);
    }

    void load(std::istream& is) {
        self_attention = MultiHeadAttention::load(is, config);
    }
    Matrix backward(const Matrix& grad_output, const Matrix& input,
                    const Matrix& target_distribution = Matrix());
    Matrix backward_cuda(const Matrix& grad, const Matrix& input) const;
    std::vector<std::reference_wrapper<Matrix>> get_weights() {
        std::vector<std::reference_wrapper<Matrix>> weights;
        auto attention_weights = self_attention->get_weights();
        auto ff_weights = feed_forward->get_weights();

        weights.insert(weights.end(), attention_weights.begin(), attention_weights.end());
        weights.insert(weights.end(), ff_weights.begin(), ff_weights.end());

        return weights;
    }
    friend class Transformer;

    MultiHeadAttention* getAttention() {
        return self_attention.get();
    }

    const MultiHeadAttention* getAttention() const {
        return self_attention.get();
    }

    FeedForward* getFeedForward() {
        return feed_forward.get();
    }
    LayerNorm* getLayerNorm() {
        return attention_ln.get();
    }
    void convert_to_fp16();

    TransformerLayer(const TransformerLayer& other)
        : config(other.config), kv_cache(other.kv_cache), layer_idx(other.layer_idx) {
        if (other.self_attention) {
            self_attention = std::make_unique<MultiHeadAttention>(*other.self_attention);
        }
        if (other.attention_ln) {
            attention_ln = std::make_unique<LayerNorm>(*other.attention_ln);
        }
        if (other.feed_forward) {
            feed_forward = std::make_unique<FeedForward>(*other.feed_forward);
        }
        if (other.ffn_ln) {
            ffn_ln = std::make_unique<LayerNorm>(*other.ffn_ln);
        }
    }

    TransformerLayer& operator=(const TransformerLayer& other) {
        if (this != &other) {
            kv_cache = other.kv_cache;
            layer_idx = other.layer_idx;

            if (other.self_attention) {
                self_attention = std::make_unique<MultiHeadAttention>(*other.self_attention);
            }
            if (other.attention_ln) {
                attention_ln = std::make_unique<LayerNorm>(*other.attention_ln);
            }
            if (other.feed_forward) {
                feed_forward = std::make_unique<FeedForward>(*other.feed_forward);
            }
            if (other.ffn_ln) {
                ffn_ln = std::make_unique<LayerNorm>(*other.ffn_ln);
            }
        }
        return *this;
    }

    void update_parameters(float learning_rate) {
        if (self_attention) {
            self_attention->update_parameters(learning_rate);
        }
        if (feed_forward) {
            feed_forward->update_parameters(learning_rate);
        }
        if (attention_ln) {
            attention_ln->update_parameters(learning_rate);
        }
        if (ffn_ln) {
            ffn_ln->update_parameters(learning_rate);
        }
    }

    void set_tokenizer(const std::shared_ptr<TiktokenTokenizer>& tokenizer) {
        if (self_attention) {
            self_attention->set_tokenizer(tokenizer);
        }
    }
};

/**
 * @brief Main Transformer model implementing the standard Transformer architecture.
 * 
 * The Transformer consists of:
 * - Token embedding layer
 * - Positional encoding
 * - Multiple transformer layers
 * - Final layer normalization
 * - Language model head for token prediction
 * 
 * Supports both training and inference modes, with features like:
 * - Key-Value caching for efficient inference
 * - CUDA acceleration
 * - Half-precision (FP16) computation
 * - Gradient checkpointing
 * - Various optimization algorithms
 */
class Transformer {
private:
    // Change from reference to value to avoid const issues
    TransformerConfig config;  // Store by value instead of reference
    std::shared_ptr<TiktokenTokenizer> tokenizer_;  // Add shared_ptr to tokenizer
    std::mt19937 gen_;  // Random number generator
    static constexpr int DEFAULT_EOS_TOKEN = 50256;  // Default EOS token ID for GPT-style tokenizers
    int eos_token_ = DEFAULT_EOS_TOKEN;  // End of sequence token ID

    // Add helper function for phrase type detection
    PhraseType get_phrase_type(const std::string& text) const {
        return PhraseTypeHandler::detect_phrase_type(text);
    }

    // Add analyze_phrase_type method for internal phrase type analysis
    PhraseType analyze_phrase_type(const Matrix& hidden_states, const TiktokenTokenizer& tokenizer);

    // Add initialization method for attention layers
    void initialize_attention_layers();

    // Components
    std::unique_ptr<TokenEmbedding> token_embedding;
    std::unique_ptr<PositionalEncoding> pos_encoding;
    std::vector<std::unique_ptr<TransformerLayer>> layers;
    std::unique_ptr<LayerNorm> final_ln;
    std::unique_ptr<LanguageModelHead> lm_head;
    std::unique_ptr<Dropout> dropout;

    // State
    bool training = true;
    Matrix hidden_states;
    Matrix last_hidden_states;
    std::vector<Matrix> m_layer_activations;
    std::vector<KVCache> m_kv_caches;
    std::vector<std::pair<size_t, size_t>> last_seq_boundaries;
    std::vector<int> last_input_tokens_;
    std::string last_input_query_;

    // Optimizer state
    std::vector<Matrix> momentum_buffers;
    std::vector<Matrix> velocity_buffers;
    size_t update_step = 0;
    std::optional<std::vector<Matrix>> parameter_grads;

    // Randomization helpers
    void add_random_noise(Matrix& logits, std::mt19937& gen) const {
        std::normal_distribution<float> noise_dist(0.0f, 0.1f);
        for (size_t i = 0; i < logits.rows(); i++) {
            for (size_t j = 0; j < logits.cols(); j++) {
                logits(i, j) += noise_dist(gen);
            }
        }
    }

    void add_random_noise(Vector& vec, std::mt19937& gen) const {
        std::normal_distribution<float> noise_dist(0.0f, 0.1f);
        for (size_t i = 0; i < vec.size(); i++) {
            vec[i] += noise_dist(gen);
        }
    }

    std::vector<float> apply_nucleus_sampling(
        const std::vector<float>& probabilities,
        float p,
        std::mt19937& gen
    ) const;

    void apply_random_boost(
        std::vector<float>& probabilities,
        std::mt19937& gen,
        float min_boost = 0.8f,
        float max_boost = 1.2f
    ) const;

    // Private methods
    void unscale_gradients(MultiHeadAttention::Gradients& grads, float scale);
    void unscale_gradients(FeedForward::Gradients& grads, float scale);
    Matrix compute_loss_gradients(const Matrix& logits, const std::vector<int>& targets);
    void backward_pass(const Matrix& output, const Matrix& target_distribution, float learning_rate);
    std::vector<Matrix>& parameter_gradients();
    void clear_gradients();

    // Helper methods for phrase prediction
    void boost_verb_probabilities(
        Vector& probabilities,
        const TiktokenTokenizer& tokenizer,
        std::mt19937* gen = nullptr
    );

    void boost_adjective_probabilities(
        Vector& probabilities,
        const TiktokenTokenizer& tokenizer,
        std::mt19937* gen = nullptr
    );

    bool is_likely_verb(const std::string& token) const;
    bool is_likely_adjective(const std::string& token) const;

    std::string extract_prediction(
        const Matrix& hidden_states,
        PhraseType phrase_type,
        const TiktokenTokenizer& tokenizer
    );

    // Helper method to ensure tokenizer is available
    void check_tokenizer() const {
        if (!tokenizer_) {
            throw std::runtime_error("Tokenizer not set. Call set_tokenizer before using this method.");
        }
    }

    std::string current_input_context;  // Track current input context

    // Add these function declarations in the private section
    float get_context_boost(const std::string& token, const std::string& input);
    bool is_subject(const std::string& word) const;
    bool is_article(const std::string& word) const;
    bool is_linking_verb(const std::string& word) const;
    float compute_semantic_similarity(const std::string& token, const std::string& input) const;
    std::vector<int> tokenize(const std::string& text) const;
    std::string detokenize(const std::vector<int>& tokens) const;
    PhraseType get_token_type(size_t token_id) const;
    size_t sample_from_distribution(const Vector& probabilities, std::mt19937& gen) const;
    Vector softmax_with_temperature(const Vector& logits, float temperature) const;

    // Token diversity tracking
    struct TokenStats {
        float presence_count;     // How many times token has been generated
        float recency_penalty;    // Penalty based on how recently token was used
        std::vector<size_t> positions;  // Positions where token was generated
    };
    std::unordered_map<int, TokenStats> generated_token_stats;
    std::vector<int> generation_history;  // Track sequence of generated tokens
    
    // Diversity control parameters
    const float presence_penalty = 0.3f;    // Base penalty for token reuse
    const float recency_decay = 0.85f;      // How quickly recency penalty decays
    const float frequency_penalty = 0.2f;    // Penalty based on frequency
    const float diversity_boost = 0.15f;     // Boost for novel tokens
    const size_t recency_window = 10;       // Window for recency calculations
    
    // Helper methods for diverse prediction
    void apply_diversity_penalties(Vector& logits) {
        // Get current position in generation
        size_t current_pos = generation_history.size();
        
        // Update recency penalties for all tracked tokens
        for (auto& [token_id, stats] : generated_token_stats) {
            // Decay recency penalty based on distance from last use
            if (!stats.positions.empty()) {
                size_t distance = current_pos - stats.positions.back();
                stats.recency_penalty *= std::pow(recency_decay, distance);
            }
            
            // Apply penalties to logits
            float total_penalty = 0.0f;
            
            // Presence penalty based on total usage
            total_penalty += presence_penalty * stats.presence_count;
            
            // Recency penalty
            total_penalty += stats.recency_penalty;
            
            // Frequency penalty based on local concentration
            size_t recent_uses = std::count_if(
                stats.positions.begin(), 
                stats.positions.end(),
                [current_pos, this](size_t pos) {
                    return current_pos - pos <= recency_window;
                }
            );
            total_penalty += frequency_penalty * recent_uses;
            
            // Apply combined penalty
            logits[token_id] -= total_penalty;
        }
        
        // Add diversity boost for unused tokens
        for (size_t i = 0; i < logits.size(); i++) {
            if (generated_token_stats.find(i) == generated_token_stats.end()) {
                logits[i] += diversity_boost;
            }
        }
    }

    void update_token_stats(int token_id) {
        size_t current_pos = generation_history.size();
        
        // Add to generation history
        generation_history.push_back(token_id);
        
        // Update or create token stats
        auto& stats = generated_token_stats[token_id];
        stats.presence_count += 1.0f;
        stats.recency_penalty += 1.0f;
        stats.positions.push_back(current_pos);
        
        // Prune old positions outside recency window
        while (!stats.positions.empty() && 
               current_pos - stats.positions.front() > recency_window) {
            stats.positions.erase(stats.positions.begin());
        }
    }

    float get_adaptive_temperature(size_t tokens_generated) {
        // Start with higher temperature and gradually decrease
        float base_temp = 1.2f;
        float min_temp = 0.7f;
        
        // Adjust temperature based on repetition
        float repetition_factor = 1.0f;
        if (!generation_history.empty()) {
            // Check for recent repetitions
            std::unordered_map<int, size_t> recent_counts;
            size_t window_start = (generation_history.size() > recency_window) ? 
                                 generation_history.size() - recency_window : 0;
            
            for (size_t i = window_start; i < generation_history.size(); i++) {
                recent_counts[generation_history[i]]++;
            }
            
            // Increase temperature if seeing repetition
            size_t max_count = 0;
            for (const auto& [token, count] : recent_counts) {
                max_count = std::max(max_count, count);
            }
            
            if (max_count > 1) {
                repetition_factor = 1.0f + (max_count - 1) * 0.2f;
            }
        }
        
        // Combine decay and repetition factors
        float decay = std::exp(-static_cast<float>(tokens_generated) / 20.0f);
        float temp = min_temp + (base_temp - min_temp) * decay;
        return temp * repetition_factor;
    }

    void reset_diversity_tracking() {
        generated_token_stats.clear();
        generation_history.clear();
    }

    // Sampling parameters
    struct SamplingParams {
        float base_temperature = 1.2f;
        float min_temperature = 0.7f;
        float nucleus_p = 0.9f;          // Default top-p for nucleus sampling
        float min_p = 0.1f;              // Minimum nucleus p value
        float max_p = 0.95f;             // Maximum nucleus p value
        float p_adjustment_rate = 0.05f;  // How quickly to adjust p based on entropy
    } sampling_params;

    // Helper method for nucleus sampling
    Vector apply_nucleus_sampling(const Vector& logits, float p, float temperature) {
        // Sort token indices by probability
        std::vector<std::pair<float, size_t>> token_probs;
        token_probs.reserve(logits.size());
        
        // Apply temperature and convert to probabilities
        Vector scaled_logits = logits;
        for (size_t i = 0; i < logits.size(); i++) {
            scaled_logits[i] /= temperature;
        }
        Vector probs = softmax(scaled_logits);
        
        for (size_t i = 0; i < probs.size(); i++) {
            token_probs.emplace_back(probs[i], i);
        }
        
        // Sort by probability in descending order
        std::sort(token_probs.begin(), token_probs.end(),
                 std::greater<std::pair<float, size_t>>());
        
        // Calculate cumulative probabilities
        float cumsum = 0.0f;
        size_t nucleus_size = 0;
        
        for (size_t i = 0; i < token_probs.size(); i++) {
            cumsum += token_probs[i].first;
            if (cumsum > p) {
                nucleus_size = i + 1;
                break;
            }
        }
        
        if (nucleus_size == 0) nucleus_size = token_probs.size();
        
        // Create new distribution with only nucleus tokens
        Vector nucleus_probs(logits.size(), 0.0f);
        float nucleus_sum = 0.0f;
        
        for (size_t i = 0; i < nucleus_size; i++) {
            nucleus_probs[token_probs[i].second] = token_probs[i].first;
            nucleus_sum += token_probs[i].first;
        }
        
        // Renormalize probabilities
        if (nucleus_sum > 0.0f) {
            for (size_t i = 0; i < nucleus_probs.size(); i++) {
                nucleus_probs[i] /= nucleus_sum;
            }
        }
        
        return nucleus_probs;
    }

    // Helper method to compute distribution entropy
    float compute_entropy(const Vector& probs) {
        float entropy = 0.0f;
        for (size_t i = 0; i < probs.size(); i++) {
            if (probs[i] > 0.0f) {
                entropy -= probs[i] * std::log2(probs[i]);
            }
        }
        return entropy;
    }

    // Get adaptive sampling parameters based on generation state
    std::pair<float, float> get_adaptive_sampling_params(size_t tokens_generated) {
        float temperature = sampling_params.min_temperature;
        float nucleus_p = sampling_params.nucleus_p;
        
        // Get base adaptive temperature
        float base_temp = get_adaptive_temperature(tokens_generated);
        
        // Adjust nucleus p based on recent entropy
        if (!generation_history.empty()) {
            // Calculate entropy of recent generations
            std::unordered_map<int, float> token_freqs;
            size_t window_start = (generation_history.size() > recency_window) ? 
                                 generation_history.size() - recency_window : 0;
            
            for (size_t i = window_start; i < generation_history.size(); i++) {
                token_freqs[generation_history[i]]++;
            }
            
            // Convert frequencies to probabilities
            Vector recent_probs(token_freqs.size());
            float sum = 0.0f;
            for (const auto& [token, freq] : token_freqs) {
                sum += freq;
            }
            for (const auto& [token, freq] : token_freqs) {
                recent_probs[token] = freq / sum;
            }
            
            // Compute entropy
            float recent_entropy = compute_entropy(recent_probs);
            
            // Adjust nucleus p based on entropy
            // Lower entropy (more repetitive) -> increase p for more diversity
            float entropy_factor = std::clamp(1.0f - recent_entropy / 4.0f, 0.0f, 1.0f);
            nucleus_p = std::clamp(
                sampling_params.nucleus_p + entropy_factor * sampling_params.p_adjustment_rate,
                sampling_params.min_p,
                sampling_params.max_p
            );
        }
        
        // Combine temperature adjustments
        temperature = base_temp;
        
        return {temperature, nucleus_p};
    }

    // Add separator-aware methods
    Matrix adjust_position_encodings(const Matrix& pos_enc, const std::vector<int>& tokens) const {
        // Get first attention layer to use its separator handling
        if (layers.empty() || !layers[0] || !layers[0]->getAttention()) {
            throw std::runtime_error("No attention layer available for position encoding adjustment");
        }
        return layers[0]->getAttention()->adjust_position_encodings(pos_enc, tokens);
    }

    AttentionMask create_separator_mask(const std::vector<int>& tokens) const {
        // Get first attention layer to use its separator handling
        if (layers.empty() || !layers[0] || !layers[0]->getAttention()) {
            throw std::runtime_error("No attention layer available for mask creation");
        }
        return layers[0]->getAttention()->create_separator_mask(tokens);
    }

    size_t find_separator_position(const std::vector<int>& tokens) const {
        // Get first attention layer to use its separator handling
        if (layers.empty() || !layers[0] || !layers[0]->getAttention()) {
            throw std::runtime_error("No attention layer available for separator detection");
        }
        return layers[0]->getAttention()->find_separator_position(tokens);
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
    const float BASE_FREQUENCY_PENALTY = 0.3f;
    const float BASE_RECENCY_PENALTY = 0.2f;
    const float PENALTY_DECAY = 0.95f;

    // Frequency tracking methods
    void update_token_frequency(int token_id);

    void reset_frequency_stats() {
        token_frequency_stats.clear();
        total_tokens_generated = 0;
    }

public:
    Transformer() = default;

    /**
     * @brief Initialize the weights of the transformer model
     */
    void initialize_weights();

    /**
     * @brief Sets the training mode for the transformer and all its components.
     * @param mode True for training mode, false for inference mode
     */
    void set_training(bool mode);

    /**
     * @brief Constructs a transformer with the given configuration and tokenizer.
     * @param config_ Configuration parameters for the transformer
     * @param tokenizer The tokenizer to use for this transformer
     */
    Transformer(const TransformerConfig& config_, std::shared_ptr<TiktokenTokenizer> tokenizer);

    /**
     * @brief Performs the forward pass through the transformer.
     * @param input_tokens Input token sequence
     * @param original_query The original input query string
     * @param tokenizer The tokenizer instance to use for decoding
     * @param use_cache Whether to use key-value caching for inference
     * @return Output logits for each position
     */
    Matrix forward(const std::vector<int>& input_tokens, const std::string& original_query, const TiktokenTokenizer& tokenizer);

    /**
     * @brief Performs backward pass and updates model parameters
     * @param logits Output logits from forward pass
     * @param target_distribution Target probability distribution
     * @param learning_rate Learning rate for parameter updates
     */
    void backward(const Matrix& logits, const Matrix& target_distribution, float learning_rate);

    /**
     * @brief Trains the transformer on the given dataset.
     * @param input_tokens Batch of input token sequences
     * @param target_tokens Batch of target token sequences
     * @param num_epochs Number of training epochs
     * @param learning_rate Learning rate for optimization
     */
    void train(const std::vector<std::vector<int>>& input_tokens,
               const std::vector<std::vector<int>>& target_tokens, size_t num_epochs,
               float learning_rate);

    /**
     * @brief Saves the model parameters to a file.
     * @param path Path to save the model to
     */
    void save_model(const std::string& path) const;

    /**
     * @brief Loads a model from a file.
     * @param path Path to load the model from
     * @return The loaded transformer model
     */
    static Transformer load_model(const std::string& path);

    /**
     * @brief Clears all key-value caches in the model.
     */
    void clear_kv_cache();

    /**
     * @brief Performs backward pass through a specific layer
     * @param grad Gradient tensor from the next layer
     * @param activation Input activation for this layer
     * @param layer_idx Index of the layer to perform backward pass on
     */
    void backward(const Matrix& grad, const Matrix& activation, size_t layer_idx);

    std::vector<Matrix>& parameters();
    void save(std::ostream& os) const;
    void load(std::istream& is);

    std::vector<std::vector<std::reference_wrapper<Matrix>>> get_layer_weights() const {
        std::vector<std::vector<std::reference_wrapper<Matrix>>> all_weights;
        for (const auto& layer : layers) {
            all_weights.push_back(layer->get_weights());
        }
        return all_weights;
    }

    friend class QuantizationAwareTraining;

    const TransformerConfig& getConfig() const {
        return config;
    }
    const std::vector<std::unique_ptr<TransformerLayer>>& getLayers() const {
        return layers;
    }
    std::vector<std::unique_ptr<TransformerLayer>>& getLayers() {
        return layers;
    }
    virtual ~Transformer();

    // Add copy constructor and assignment operator
    Transformer(const Transformer& other);
    Transformer& operator=(const Transformer& other);

    // Move constructor and assignment operator
    Transformer(Transformer&& other) noexcept = default;
    Transformer& operator=(Transformer&& other) noexcept = default;

    // Keep the original backward method for single sample training
    void backward(const Matrix& grad_output, const std::vector<int>& input_tokens, float learning_rate);
    
    // Add new backward method for batch training
    void backward(std::vector<Matrix>& outputs, const Matrix& target_distribution, float learning_rate);

    const Matrix& get_hidden_states() const {
        return hidden_states;
    }
    LanguageModelHead* get_lm_head() {
        return lm_head.get();
    }
    void set_lm_head(std::unique_ptr<LanguageModelHead> head) {
        lm_head = std::move(head);
    }

    bool verify_state() const {
        return token_embedding && pos_encoding && final_ln && lm_head && !layers.empty() &&
               std::all_of(layers.begin(), layers.end(), [](const auto& layer) { return layer != nullptr; });
    }

    bool is_training() const {
        return training;
    }

    void save_checkpoint(const std::string& path);

    const std::vector<int>& get_last_input() const {
        return last_input_tokens_;
    }

    const std::string& get_last_query() const {
        return last_input_query_;
    }

    /**
     * @brief Gets the tokenizer associated with this transformer.
     * @return Shared pointer to the tokenizer
     */
    std::shared_ptr<TiktokenTokenizer> get_tokenizer() const {
        return tokenizer_;
    }

    /**
     * @brief Updates model parameters using computed gradients.
     * @param learning_rate Learning rate for the update
     */
    void update_parameters(float learning_rate);

    /**
     * @brief Predicts the final phrase for a given input text without delimiters
     * @param input_text The input text without delimiters
     * @param tokenizer The tokenizer instance
     * @return A pair containing the predicted phrase and its type
     */
    std::pair<std::string, PhraseType> predict_final_phrase(
        const std::string& input_text,
        const TiktokenTokenizer& tokenizer
    );

    /**
     * @brief Predicts the most likely phrase type for the given input
     * @param input_text The input text
     * @param tokenizer The tokenizer instance
     * @return The predicted phrase type
     */
    PhraseType predict_phrase_type(
        const std::string& input_text,
        const TiktokenTokenizer& tokenizer
    );

    // Add softmax helper function
    Vector softmax(const Vector& input) const {
        Vector output(input.size());
        float max_val = *std::max_element(input.begin(), input.end());
        float sum = 0.0f;
        
        // Compute exp and sum
        for (size_t i = 0; i < input.size(); i++) {
            output[i] = std::exp(input[i] - max_val);
            sum += output[i];
        }
        
        // Normalize
        for (size_t i = 0; i < output.size(); i++) {
            output[i] /= sum;
        }
        
        return output;
    }

    // Add context tracking
    std::string get_current_context() const { return current_input_context; }
    void set_current_context(const std::string& context) { current_input_context = context; }

    /**
     * @brief Generates a sequence of tokens given an input sequence
     * @param input_tokens Initial sequence of tokens
     * @param max_length Maximum length of the generated sequence
     * @param temperature Sampling temperature (higher = more random)
     * @return Generated sequence of tokens
     */
    std::vector<int> generate(const std::vector<int>& input_tokens, 
                            size_t max_length = 100,
                            float temperature = 1.0f);

    // Add this function declaration in the public section
    std::vector<TokenPrediction> predict_next_tokens(const std::string& input, size_t num_predictions);

    /**
     * @brief Forward pass through the transformer model with KV cache update option
     * @param input_tokens Input token IDs
     * @param update_kv_cache Whether to update the key-value cache
     * @return Vector of logits for next token prediction
     */
    Vector forward(const std::vector<int>& input_tokens, bool update_kv_cache);

    /**
     * @brief Reset the transformer's key-value cache and frequency statistics
     */
    void reset_cache();

    /**
     * @brief Get dynamic temperature for sampling based on phrase type and current state
     * @param type The type of phrase being generated
     * @param gen Random number generator for adding noise
     * @return Dynamic temperature value
     */
    float get_dynamic_temperature(PhraseType type, const std::mt19937& gen) const;

    /**
     * @brief Creates a target distribution matrix for training
     * @param target_tokens Target token sequence
     * @param vocab_size Size of the vocabulary
     * @return Matrix containing the target distribution
     */
    Matrix create_target_distribution(const std::vector<int>& target_tokens, size_t vocab_size) const;

};

class PositionalEncoding;  // Forward declaration is enough since we include embeddings.hpp