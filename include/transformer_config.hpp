#pragma once
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

struct TransformerConfig {
    struct ModelConfig {
        int vocab_size;
        int hidden_size;
        int num_heads;
        int num_layers;
        int head_dim;
        int intermediate_size;
        int max_seq_length;
    };

    struct TrainingConfig {
        int batch_size;
        int num_epochs;
        float dropout_rate;
        float weight_decay;
    };

    struct AttentionConfig {
        bool use_flash_attention;
        bool use_rope;
        bool use_sliding_window;
        int window_size;
        bool use_gqa;
        int num_kv_heads;
    };

    struct OptimizationConfig {
        bool use_fp16;
        bool use_gradient_checkpointing;
        int memory_pool_size;
    };

    struct PathsConfig {
        std::string save_directory;
        std::string model_name;
        int checkpoint_frequency;
    };

    struct BeamSearchConfig {
        bool use_beam_search;
        int beam_size;
        int beams_per_group;
        int num_groups;
        float length_penalty;
        float temperature;
        float initial_temperature;
        float diversity_strength;
        int top_k;
        float top_p;
        int max_length;
        float initial_noise_scale;
        float token_noise_scale;
    };

    struct TokenizerConfig {
        bool use_subword;
        int vocab_size;
        std::string model_path;
        std::vector<std::string> special_tokens;
    };

    struct DebugConfig {
        bool verbose_logging;
        bool log_matrix_stats;
        int log_frequency;
    };

    ModelConfig model;
    TrainingConfig training;
    AttentionConfig attention;
    OptimizationConfig optimization;
    PathsConfig paths;
    BeamSearchConfig beam_search;
    TokenizerConfig tokenizer;
    DebugConfig debug;
    
    bool load_from_checkpoint;
    std::string checkpoint_to_load;
    int pad_token_id;
    int unk_token_id;
    int bos_token_id;
    int eos_token_id;
    int mask_token_id;

    // Update the load function to parse debug settings
    static TransformerConfig from_json(const nlohmann::json& j) {
        TransformerConfig config;
        // ... existing parsing ...

        if (j.contains("debug")) {
            const auto& debug = j["debug"];
            config.debug.verbose_logging = debug.value("verbose_logging", false);
            config.debug.log_matrix_stats = debug.value("log_matrix_stats", false);
            config.debug.log_frequency = debug.value("log_frequency", 100);
        }

        return config;
    }
};

// JSON serialization functions
void from_json(const nlohmann::json& j, TransformerConfig::ModelConfig& m);
void from_json(const nlohmann::json& j, TransformerConfig::TrainingConfig& t);
void from_json(const nlohmann::json& j, TransformerConfig::AttentionConfig& a);
void from_json(const nlohmann::json& j, TransformerConfig::OptimizationConfig& o);
void from_json(const nlohmann::json& j, TransformerConfig::PathsConfig& p);
void from_json(const nlohmann::json& j, TransformerConfig::BeamSearchConfig& b);
void from_json(const nlohmann::json& j, TransformerConfig::TokenizerConfig& t);
void from_json(const nlohmann::json& j, TransformerConfig::DebugConfig& d);
void from_json(const nlohmann::json& j, TransformerConfig& c); 