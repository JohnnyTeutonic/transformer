#pragma once
#include <cstddef>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

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
class TransformerConfig {
  public:
    // Model parameters
    size_t vocab_size;           ///< Size of the vocabulary
    size_t max_seq_length;       ///< Maximum sequence length the model can handle
    size_t hidden_size;          ///< Dimension of the model's hidden states
    size_t num_layers;           ///< Number of transformer layers
    size_t num_heads;            ///< Number of attention heads
    size_t head_dim;             ///< Dimension of each attention head
    size_t intermediate_size;     ///< Size of the feedforward network

    // Training parameters
    float dropout_prob = 0.1f;    ///< Dropout probability
    bool use_flash_attention = true;  ///< Whether to use flash attention
    bool use_rope = true;         ///< Whether to use rotary position embeddings
    bool use_sliding_window = false;  ///< Whether to use sliding window attention
    size_t window_size = 512;     ///< Size of the sliding window
    bool use_gqa = false;         ///< Whether to use grouped query attention
    size_t num_kv_heads;          ///< Number of key-value heads for GQA
    bool use_fp16 = false;        ///< Whether to use half-precision
    bool use_gradient_checkpointing = true;  ///< Whether to use gradient checkpointing
    size_t memory_pool_size = 1024;  ///< Size of memory pool in MB
    size_t batch_size;            ///< Training batch size
    size_t num_epochs;            ///< Number of training epochs
    float dropout_rate;           ///< Dropout rate
    float weight_decay;           ///< Weight decay rate

    // Debug and logging settings
    bool debug_mode = false;      ///< Global debug mode flag
    size_t log_frequency = 100;   ///< How often to log training stats

    // Checkpoint settings
    bool load_from_checkpoint;    ///< Whether to load from checkpoint
    std::string checkpoint_to_load;  ///< Path to checkpoint to load

    // Path settings
    struct {
        std::string save_directory;  ///< Directory to save models
        std::string model_name;      ///< Name of the model
        size_t checkpoint_frequency;  ///< How often to save checkpoints
    } paths;

    // Beam search settings
    struct {
        bool use_beam_search;      ///< Whether to use beam search
        size_t beam_size;          ///< Size of beam
        size_t beams_per_group;    ///< Number of beams per group
        size_t num_groups;         ///< Number of groups
        float length_penalty;      ///< Length penalty
        float temperature;         ///< Temperature for sampling
        float top_p;              ///< Top-p sampling threshold
        size_t max_length;        ///< Maximum generation length
        float initial_temperature;  ///< Initial temperature
        float initial_noise_scale;  ///< Initial noise scale
        float diversity_strength;   ///< Diversity strength
        size_t top_k;             ///< Top-k sampling threshold
        float token_noise_scale;   ///< Token noise scale
    } beam_search;

    // Tokenizer settings
    struct {
        bool use_subword;
        size_t vocab_size;
        std::string model_path;
        std::vector<std::string> special_tokens;
    } tokenizer;

    /**
     * @brief Constructs a transformer configuration with default values.
     */
    TransformerConfig(size_t vocab_size = 32000, size_t max_seq_length = 512,
                      size_t hidden_size = 768, size_t num_layers = 12, size_t num_heads = 12,
                      size_t batch_size = 32, size_t num_epochs = 10);

    bool operator!=(const TransformerConfig& other) const;

    /**
     * @brief Loads configuration from a JSON object
     * @param j JSON object containing configuration
     */
    void from_json(const nlohmann::json& j);
};

// Add nlohmann::json support
namespace nlohmann {
    template <>
    struct adl_serializer<TransformerConfig> {
        static void from_json(const json& j, TransformerConfig& config) {
            config.from_json(j);
        }
    };
}