#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <cstddef>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

// Forward declarations of configuration structures
struct BeamSearchConfig {
    bool use_beam_search = true;
    size_t beam_size = 4;
    size_t beams_per_group = 4;
    size_t num_groups = 3;
    float length_penalty = 1.5f;
    float temperature = 2.5f;
    float top_p = 0.98f;
    size_t max_length = 4;
    float initial_temperature = 3.0f;
    float initial_noise_scale = 0.8f;
    float diversity_strength = 4.0f;
    size_t top_k = 100;
    float token_noise_scale = 0.1f;
};

struct TokenizerConfig {
    bool use_subword = true;
    size_t vocab_size = 0;
    std::string model_path;
    std::vector<std::string> special_tokens;
};

struct TokenPredictionConfig {
    float temperature = 1.0f;
    int top_k = 5;
    float top_p = 0.9f;
    float frequency_penalty = 0.1f;
    float presence_penalty = 0.0f;
    float min_token_prob = 0.05f;
    
    struct CategoryBonus {
        float verb = 0.2f;
        float adjective = 0.2f;
        float noun = 0.3f;
    } category_bonus;
};

struct PathConfig {
    std::string save_directory;
    std::string model_name;
    size_t checkpoint_frequency;
};

/**
 * @brief Configuration class for transformer model architecture and training settings.
 * 
 * This class encapsulates all configuration parameters needed to define and train
 * a transformer model, including architectural choices, optimization settings,
 * and various hyperparameters. The configuration is divided into several categories:
 * - Model architecture parameters
 * - Attention mechanism settings
 * - Training and optimization parameters
 * - File paths and checkpointing
 * - Generation and beam search settings
 */
struct TransformerConfig {
    // Model architecture
    size_t vocab_size;
    size_t hidden_size;
    size_t num_heads;
    size_t num_layers;
    size_t head_dim;
    size_t intermediate_size;
    size_t num_kv_heads;
    size_t max_seq_length;

    // Training parameters
    size_t batch_size;
    size_t num_epochs;
    float initial_lr;
    float dropout_rate;
    float weight_decay;
    
    // Early stopping parameters
    size_t early_stopping_patience;
    float early_stopping_threshold;
    
    // Cross validation parameters
    size_t num_folds = 5;
    
    // Learning rate parameters
    float peak_lr;
    size_t warmup_steps;
    float decay_factor;
    
    // Optimization parameters
    float gradient_clip_threshold;
    float layer_norm_epsilon;
    size_t gradient_accumulation_steps;
    bool use_gradient_checkpointing;
    bool use_fp16;
    size_t memory_pool_size;

    // Attention parameters
    bool use_flash_attention;
    bool use_rope;
    bool use_sliding_window;
    size_t window_size;
    bool use_gqa;

    // Paths
    PathConfig paths;

    // Component configs
    TokenizerConfig tokenizer;
    BeamSearchConfig beam_search;
    TokenPredictionConfig token_prediction;

    // Checkpoint loading
    bool load_from_checkpoint = false;
    std::string checkpoint_to_load;

    // Optimizer settings
    bool use_momentum = false;
    bool use_adam = false;
    float momentum = 0.9f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;

    struct TrainingConfig {
        size_t batch_size = 16;
        size_t num_epochs = 3;
        float dropout_rate = 0.1f;
        float weight_decay = 0.01f;
        
        struct CrossValidationConfig {
            size_t num_folds = 2;
            size_t validation_frequency = 1;
            float early_stopping_threshold = 1.5f;
            size_t early_stopping_patience = 2;
            size_t num_epochs = 10;
        } cross_validation;
        
        // Learning rate parameters
        struct LearningRateConfig {
            float initial_lr = 1e-4f;
            float peak_lr = 2e-3f;
            size_t warmup_steps = 15;
            float decay_factor = 0.98f;
            size_t decay_steps = 250;
            float min_lr = 1e-4f;
        } learning_rate;

        // Hyperparameter tuning
        struct TuningConfig {
            size_t num_trials = 10;
            size_t evaluation_steps = 50;
            bool enabled = true;
        } tuning;
    } training;

    /**
     * @brief Constructs a transformer configuration with default values.
     * @param vocab_size Size of the vocabulary (default: 32000)
     * @param max_seq_length Maximum sequence length (default: 512)
     * @param hidden_size Dimension of hidden states (default: 768)
     * @param num_layers Number of transformer layers (default: 12)
     * @param num_heads Number of attention heads (default: 12)
     * @param batch_size Training batch size (default: 32)
     * @param num_epochs Number of training epochs (default: 10)
     */
    TransformerConfig(size_t vocab_size = 32000, size_t max_seq_length = 512,
                     size_t hidden_size = 768, size_t num_layers = 12, size_t num_heads = 12,
                     size_t batch_size = 32, size_t num_epochs = 10);

    /**
     * @brief Compares two configurations for inequality.
     * @param other Configuration to compare against
     * @return true if configurations differ, false otherwise
     */
    bool operator!=(const TransformerConfig& other) const;

    /**
     * @brief Loads configuration from a JSON file.
     */
    void load_from_json(const std::string& config_path);
};

// JSON serialization declarations
void to_json(nlohmann::json& j, const TokenizerConfig& t);
void from_json(const nlohmann::json& j, TokenizerConfig& t);

void to_json(nlohmann::json& j, const BeamSearchConfig& b);
void from_json(const nlohmann::json& j, BeamSearchConfig& b);

void to_json(nlohmann::json& j, const TokenPredictionConfig& t);
void from_json(const nlohmann::json& j, TokenPredictionConfig& t);

void to_json(nlohmann::json& j, const TransformerConfig& t);
void from_json(const nlohmann::json& j, TransformerConfig& t);

#endif // CONFIG_HPP