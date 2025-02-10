#include "../include/config.hpp"
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <nlohmann/json.hpp>

TransformerConfig::TransformerConfig(size_t max_seq_length_, size_t hidden_size_,
                                   size_t num_layers_, size_t num_heads_, size_t samples_per_iteration_,
                                   size_t num_epochs_)
    : hidden_size(hidden_size_),
      num_heads(num_heads_),
      num_layers(num_layers_),
      head_dim(hidden_size_ / num_heads_),
      intermediate_size(4 * hidden_size_),
      max_seq_length(max_seq_length_) {
    
    // Initialize training config
    training.samples_per_iteration = samples_per_iteration_;
    training.num_epochs = num_epochs_;
    
    if (hidden_size % num_heads != 0) {
        throw std::invalid_argument("Hidden size must be divisible by number of heads");
    }
}

bool TransformerConfig::operator!=(const TransformerConfig& other) const {
    return max_seq_length != other.max_seq_length ||
           hidden_size != other.hidden_size ||
           num_layers != other.num_layers ||
           num_heads != other.num_heads ||
           head_dim != other.head_dim ||
           intermediate_size != other.intermediate_size ||
           training.samples_per_iteration != other.training.samples_per_iteration ||
           training.num_epochs != other.training.num_epochs ||
           dropout_rate != other.dropout_rate ||
           weight_decay != other.weight_decay ||
           use_gradient_checkpointing != other.use_gradient_checkpointing ||
           use_fp16 != other.use_fp16 ||
           use_flash_attention != other.use_flash_attention ||
           use_rope != other.use_rope ||
           use_sliding_window != other.use_sliding_window ||
           window_size != other.window_size ||
           use_gqa != other.use_gqa ||
           num_kv_heads != other.num_kv_heads;
}

void TransformerConfig::load_from_json(const std::string& config_path) {
    try {
        std::ifstream file(config_path);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open config file: " + config_path);
        }

        nlohmann::json j;
        file >> j;

        std::cout << "\nLoading configuration from: " << config_path << std::endl;

        // Load model parameters
        if (j.contains("model")) {
            const auto& model = j["model"];
            hidden_size = model.value("hidden_size", hidden_size);
            num_heads = model.value("num_heads", num_heads);
            num_layers = model.value("num_layers", num_layers);
            head_dim = model.value("head_dim", head_dim);
            intermediate_size = model.value("intermediate_size", intermediate_size);
            max_seq_length = model.value("max_seq_length", max_seq_length);

            // Print loaded configuration for debugging
            std::cout << "Loaded model configuration:" << std::endl;
            std::cout << "- hidden_size: " << hidden_size << std::endl;
            std::cout << "- num_heads: " << num_heads << std::endl;
            std::cout << "- num_layers: " << num_layers << std::endl;
            std::cout << "- head_dim: " << head_dim << std::endl;
            std::cout << "- intermediate_size: " << intermediate_size << std::endl;
            std::cout << "- max_seq_length: " << max_seq_length << std::endl;
        }

        // Load training parameters
        if (j.contains("training")) {
            const auto& training_json = j["training"];
            
            // Debug output before loading
            std::cout << "\nBefore loading training config:" << std::endl;
            std::cout << "- samples_per_iteration: " << training.samples_per_iteration << std::endl;
            std::cout << "- num_epochs: " << training.num_epochs << std::endl;
            std::cout << "- tuning.enabled: " << std::boolalpha << training.tuning.enabled << std::endl;
            
            training.samples_per_iteration = training_json.value("samples_per_iteration", training.samples_per_iteration);
            training.num_epochs = training_json.value("num_epochs", training.num_epochs);
            training.dropout_rate = training_json.value("dropout_rate", training.dropout_rate);
            training.weight_decay = training_json.value("weight_decay", training.weight_decay);

            // Load tuning configuration
            if (training_json.contains("tuning")) {
                const auto& tuning_json = training_json["tuning"];
                std::cout << "\nFound tuning configuration in JSON:" << std::endl;
                std::cout << "JSON tuning.enabled = " << std::boolalpha 
                          << tuning_json.value("enabled", true) << std::endl;
                
                training.tuning.enabled = tuning_json.value("enabled", training.tuning.enabled);
                training.tuning.num_trials = tuning_json.value("num_trials", training.tuning.num_trials);
                training.tuning.evaluation_steps = tuning_json.value("evaluation_steps", training.tuning.evaluation_steps);
                
                std::cout << "\nAfter loading tuning config:" << std::endl;
                std::cout << "- enabled: " << std::boolalpha << training.tuning.enabled << std::endl;
                std::cout << "- num_trials: " << training.tuning.num_trials << std::endl;
                std::cout << "- evaluation_steps: " << training.tuning.evaluation_steps << std::endl;
            } else {
                std::cout << "No tuning configuration found in JSON, using defaults" << std::endl;
            }

            // Load cross validation configuration
            if (training_json.contains("cross_validation")) {
                const auto& cv = training_json["cross_validation"];
                training.cross_validation.num_folds = cv.value("num_folds", training.cross_validation.num_folds);
                training.cross_validation.validation_frequency = cv.value("validation_frequency", training.cross_validation.validation_frequency);
                training.cross_validation.early_stopping_threshold = cv.value("early_stopping_threshold", training.cross_validation.early_stopping_threshold);
                training.cross_validation.early_stopping_patience = cv.value("early_stopping_patience", training.cross_validation.early_stopping_patience);
                training.cross_validation.num_epochs = cv.value("num_epochs", training.cross_validation.num_epochs);
            }

            // Load learning rate configuration
            if (training_json.contains("learning_rate")) {
                const auto& lr = training_json["learning_rate"];
                training.learning_rate.initial_lr = lr.value("initial_lr", training.learning_rate.initial_lr);
                training.learning_rate.peak_lr = lr.value("peak_lr", training.learning_rate.peak_lr);
                training.learning_rate.warmup_steps = lr.value("warmup_steps", training.learning_rate.warmup_steps);
                training.learning_rate.decay_factor = lr.value("decay_factor", training.learning_rate.decay_factor);
                training.learning_rate.decay_steps = lr.value("decay_steps", training.learning_rate.decay_steps);
                training.learning_rate.min_lr = lr.value("min_lr", training.learning_rate.min_lr);
            }
        }

        // Load attention parameters
        if (j.contains("attention")) {
            const auto& attention = j["attention"];
            use_flash_attention = attention.value("use_flash_attention", use_flash_attention);
            use_rope = attention.value("use_rope", use_rope);
            use_sliding_window = attention.value("use_sliding_window", use_sliding_window);
            window_size = attention.value("window_size", window_size);
            use_gqa = attention.value("use_gqa", use_gqa);
            num_kv_heads = attention.value("num_kv_heads", num_kv_heads);
        }

        // Load optimization parameters
        if (j.contains("optimization")) {
            const auto& opt = j["optimization"];
            use_fp16 = opt.value("use_fp16", use_fp16);
            use_gradient_checkpointing = opt.value("use_gradient_checkpointing", use_gradient_checkpointing);
            memory_pool_size = opt.value("memory_pool_size", memory_pool_size);
            gradient_clip_threshold = opt.value("gradient_clip_threshold", gradient_clip_threshold);
            layer_norm_epsilon = opt.value("layer_norm_epsilon", layer_norm_epsilon);
            gradient_accumulation_steps = opt.value("gradient_accumulation_steps", gradient_accumulation_steps);
            use_momentum = opt.value("use_momentum", use_momentum);
            use_adam = opt.value("use_adam", use_adam);
            momentum = opt.value("momentum", momentum);
            beta1 = opt.value("beta1", beta1);
            beta2 = opt.value("beta2", beta2);
            epsilon = opt.value("epsilon", epsilon);
        }

        // Load paths
        if (j.contains("paths")) {
            const auto& p = j["paths"];
            paths.save_directory = p.value("save_directory", paths.save_directory);
            paths.model_name = p.value("model_name", paths.model_name);
            paths.checkpoint_frequency = p.value("checkpoint_frequency", paths.checkpoint_frequency);
        }

        // Load tokenizer configuration
        if (j.contains("tokenizer")) {
            const auto& tok = j["tokenizer"];
            tokenizer.use_subword = tok.value("use_subword", tokenizer.use_subword);
            tokenizer.model_path = tok.value("model_path", tokenizer.model_path);
            tokenizer.special_tokens = tok.value("special_tokens", tokenizer.special_tokens);
        }

        // Load beam search configuration
        if (j.contains("beam_search")) {
            const auto& beam = j["beam_search"];
            beam_search.use_beam_search = beam.value("use_beam_search", beam_search.use_beam_search);
            beam_search.beam_size = beam.value("beam_size", beam_search.beam_size);
            beam_search.beams_per_group = beam.value("beams_per_group", beam_search.beams_per_group);
            beam_search.num_groups = beam.value("num_groups", beam_search.num_groups);
            beam_search.length_penalty = beam.value("length_penalty", beam_search.length_penalty);
            beam_search.temperature = beam.value("temperature", beam_search.temperature);
            beam_search.top_p = beam.value("top_p", beam_search.top_p);
            beam_search.max_length = beam.value("max_length", beam_search.max_length);
            beam_search.initial_temperature = beam.value("initial_temperature", beam_search.initial_temperature);
            beam_search.initial_noise_scale = beam.value("initial_noise_scale", beam_search.initial_noise_scale);
            beam_search.diversity_strength = beam.value("diversity_strength", beam_search.diversity_strength);
            beam_search.top_k = beam.value("top_k", beam_search.top_k);
            beam_search.token_noise_scale = beam.value("token_noise_scale", beam_search.token_noise_scale);
        }

        // Load token prediction configuration
        if (j.contains("token_prediction")) {
            const auto& tp = j["token_prediction"];
            token_prediction.temperature = tp.value("temperature", token_prediction.temperature);
            token_prediction.top_k = tp.value("top_k", token_prediction.top_k);
            token_prediction.top_p = tp.value("top_p", token_prediction.top_p);
            token_prediction.frequency_penalty = tp.value("frequency_penalty", token_prediction.frequency_penalty);
            token_prediction.presence_penalty = tp.value("presence_penalty", token_prediction.presence_penalty);
            token_prediction.min_token_prob = tp.value("min_token_prob", token_prediction.min_token_prob);
            
            if (tp.contains("category_bonus")) {
                const auto& cb = tp["category_bonus"];
                token_prediction.category_bonus.verb = cb.value("verb", token_prediction.category_bonus.verb);
                token_prediction.category_bonus.adjective = cb.value("adjective", token_prediction.category_bonus.adjective);
                token_prediction.category_bonus.noun = cb.value("noun", token_prediction.category_bonus.noun);
            }
        }

        // Load checkpoint settings
        load_from_checkpoint = j.value("load_from_checkpoint", load_from_checkpoint);
        checkpoint_to_load = j.value("checkpoint_to_load", checkpoint_to_load);

    } catch (const std::exception& e) {
        throw std::runtime_error("Error loading config from JSON: " + std::string(e.what()));
    }
}