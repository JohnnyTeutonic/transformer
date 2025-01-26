#include "../include/utils.hpp"
#include "../include/beam_search.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <queue>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <set>
#include <unordered_set>
#include "../include/data_augmentation.hpp"
#include "components.hpp"
#include <map>
#include <limits>

struct JobCategory {
    std::string category;
    std::vector<std::string> job_titles;
    std::vector<std::string> common_verbs;
    std::vector<std::string> common_locations;
};

bool starts_with(const std::string& str, const std::string& prefix) {
    return str.size() >= prefix.size() && 
           str.compare(0, prefix.size(), prefix) == 0;
}

bool ends_with(const std::string& str, const std::string& suffix) {
    if (str.length() < suffix.length()) {
        return false;
    }
    return str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
}

float Utils::adjust_learning_rate(float current_lr, float loss_ratio, size_t step) {
    const size_t WARMUP_STEPS = 50;
    const float PEAK_LR = 5e-4;
    const float MIN_LR = 1e-5;

    if (step < WARMUP_STEPS) {
        return MIN_LR + (PEAK_LR - MIN_LR) * (static_cast<float>(step) / WARMUP_STEPS);
    }

    const size_t DECAY_STEPS = 5000;
    float progress = static_cast<float>(step - WARMUP_STEPS) / DECAY_STEPS;
    progress = std::min(1.0f, progress);

    float decay_factor = 0.5f * (1.0f + std::cos(progress * M_PI));
    float lr = MIN_LR + (PEAK_LR - MIN_LR) * decay_factor;

    const float LOSS_SPIKE_THRESHOLD = 1.5f;
    if (loss_ratio > LOSS_SPIKE_THRESHOLD) {
        lr *= 0.1f;
    }

    return std::clamp(lr, MIN_LR, PEAK_LR);
}

Matrix Utils::create_batch_target_distribution(const std::vector<std::vector<int>>& target_tokens,
                                               const Tokenizer& tokenizer, size_t vocab_size,
                                               size_t input_max_seq_len) {
    // Calculate total size based on input sequence length
    size_t batch_size = target_tokens.size();
    size_t total_tokens = batch_size * input_max_seq_len;
    
    // Create target distribution for all token positions
    Matrix target_distribution(total_tokens, vocab_size, 0.0f);
    
    // Set target distribution for each token in each sequence
    size_t current_pos = 0;
    for (size_t seq = 0; seq < target_tokens.size(); seq++) {
        
        // Set actual tokens
        for (size_t i = 0; i < target_tokens[seq].size(); i++) {
            target_distribution(current_pos, target_tokens[seq][i]) = 1.0f;
            current_pos++;
        }
        
        // Pad remaining positions with pad token
        for (size_t i = target_tokens[seq].size(); i < input_max_seq_len; i++) {
            target_distribution(current_pos, tokenizer.get_pad_token_id()) = 1.0f;
            current_pos++;
        }
    }
    
    std::cout << "Final target distribution shape: " 
              << target_distribution.rows() << "x" << target_distribution.cols() << std::endl;
    std::cout << "Final current_pos: " << current_pos << "\n";
    std::cout << "=== Target Distribution Creation Complete ===\n\n";
    
    return target_distribution;
}

float Utils::compute_batch_loss(const Matrix& logits, const Matrix& target_distribution, 
                              const Tokenizer& tokenizer) {
    float total_loss = 0.0f;
    size_t valid_predictions = 0;
    
    for (size_t i = 0; i < logits.rows(); ++i) {
        // Get predicted token sequence
        std::vector<int> pred_tokens;
        float max_logit = -std::numeric_limits<float>::infinity();
        int pred_token = -1;
        for (size_t j = 0; j < logits.cols(); ++j) {
            if (logits(i, j) > max_logit) {
                max_logit = logits(i, j);
                pred_token = j;
            }
        }
        
        // Only compute loss for valid word completions
        std::string decoded = tokenizer.decode({pred_token});
        if (!decoded.empty() && decoded[0] != ' ') {  // Likely a subword
            valid_predictions++;
            
            // Compute cross entropy loss
            float token_loss = 0.0f;
            float sum_exp = 0.0f;
            float max_val = -std::numeric_limits<float>::infinity();
            
            // Find max for numerical stability
            for (size_t j = 0; j < logits.cols(); j++) {
                max_val = std::max(max_val, logits(i, j));
            }
            
            // Compute softmax denominator
            for (size_t j = 0; j < logits.cols(); j++) {
                sum_exp += std::exp(logits(i, j) - max_val);
            }
            
            // Compute loss for this token
            for (size_t j = 0; j < logits.cols(); j++) {
                if (target_distribution(i, j) > 0.0f) {
                    token_loss -= target_distribution(i, j) * 
                                 (logits(i, j) - max_val - std::log(sum_exp));
                }
            }
            
            total_loss += token_loss;
        }
    }
    
    return valid_predictions > 0 ? total_loss / valid_predictions : 0.0f;
}

TransformerConfig Utils::load_config(const std::string& config_path) {
    TransformerConfig config;
    try {
        std::ifstream file(config_path);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open config file: " + config_path);
        }

        nlohmann::json j;
        file >> j;

        // Parse model settings
        auto& model = j["model"];
        config.vocab_size = model["vocab_size"];
        config.hidden_size = model["hidden_size"];
        config.num_heads = model["num_heads"];
        config.num_layers = model["num_layers"];
        config.head_dim = model["head_dim"];
        config.intermediate_size = model["intermediate_size"];

        // Parse training settings
        auto& training = j["training"];
        config.batch_size = training["batch_size"];
        config.num_epochs = training["num_epochs"];
        config.dropout_rate = training["dropout_rate"];
        config.weight_decay = training["weight_decay"];

        // Parse paths
        if (j.contains("paths")) {
            auto& paths = j["paths"];
            config.paths.save_directory = paths["save_directory"];
            config.paths.model_name = paths["model_name"];
            config.paths.checkpoint_frequency = paths["checkpoint_frequency"];
        }

        // Parse attention settings
        auto& attention = j["attention"];
        config.use_flash_attention = attention["use_flash_attention"];
        config.use_rope = attention["use_rope"];
        config.use_sliding_window = attention["use_sliding_window"];
        config.window_size = attention["window_size"];
        if (attention.contains("use_gqa")) {
            config.use_gqa = attention["use_gqa"].get<bool>();
            std::cout << "Loaded use_gqa from config: " << config.use_gqa << std::endl;
            if (config.use_gqa) {
                if (attention.contains("num_kv_heads")) {
                    config.num_kv_heads = attention["num_kv_heads"].get<size_t>();
                } else {
                    config.num_kv_heads = config.num_heads / 2; // Default to half the heads
                }
                std::cout << "Using GQA with num_heads=" << config.num_heads
                          << " and num_kv_heads=" << config.num_kv_heads << std::endl;
            } else {
                config.num_kv_heads = config.num_heads; // No GQA, use same number
            }
        }

        // Parse optimization settings
        auto& optimization = j["optimization"];
        config.use_fp16 = optimization["use_fp16"];
        config.use_gradient_checkpointing = optimization["use_gradient_checkpointing"];
        config.memory_pool_size = optimization["memory_pool_size"];

        // Parse beam search settings
        if (j.contains("beam_search")) {
            auto& beam = j["beam_search"];
            config.beam_search.use_beam_search = beam.value("use_beam_search", true);
            config.beam_search.beam_size = beam["beam_size"];
            config.beam_search.beams_per_group = beam.value("beams_per_group", 4);
            config.beam_search.num_groups = beam.value("num_groups", 3);
            config.beam_search.length_penalty = beam["length_penalty"];
            config.beam_search.temperature = beam["temperature"];
            config.beam_search.top_p = beam["top_p"];
            config.beam_search.max_length = beam["max_length"];
            config.beam_search.initial_temperature = beam.value("initial_temperature", 3.0f);
            config.beam_search.initial_noise_scale = beam.value("initial_noise_scale", 0.8f);
            config.beam_search.diversity_strength = beam.value("diversity_strength", 4.0f);
            config.beam_search.top_k = beam.value("top_k", 100);
            config.beam_search.token_noise_scale = beam.value("token_noise_scale", 0.1f);
        } else {
            // Default values if not specified
            config.beam_search.use_beam_search = true;
            config.beam_search.beam_size = 5;
            config.beam_search.beams_per_group = 4;
            config.beam_search.num_groups = 3;
            config.beam_search.length_penalty = 0.6f;
            config.beam_search.temperature = 1.0f;
            config.beam_search.top_p = 0.9f;
            config.beam_search.max_length = 20;
            config.beam_search.initial_temperature = 3.0f;
            config.beam_search.initial_noise_scale = 0.8f;
            config.beam_search.diversity_strength = 4.0f;
            config.beam_search.top_k = 100;
            config.beam_search.token_noise_scale = 0.1f;
        }

        // Add checkpoint loading settings
        if (j.contains("load_from_checkpoint")) {
            config.load_from_checkpoint = j["load_from_checkpoint"].get<bool>();
            if (config.load_from_checkpoint && j.contains("checkpoint_to_load")) {
                config.checkpoint_to_load = j["checkpoint_to_load"].get<std::string>();
                std::cout << "Will load checkpoint from: " << config.checkpoint_to_load
                          << std::endl;
            }
        }

        // Load tokenizer settings
        if (j.contains("tokenizer")) {
            const auto& tok = j["tokenizer"];
            config.tokenizer.use_subword = tok.value("use_subword", true);
            config.tokenizer.vocab_size = tok.value("vocab_size", 32000);
            config.tokenizer.model_path = tok.value("model_path", "model/tokenizer.model");
            config.tokenizer.special_tokens = tok.value("special_tokens", 
                std::vector<std::string>{"<pad>", "", " ", "</s>", "<mask>"});
        }

    } catch (const std::exception& e) {
        throw std::runtime_error("Error parsing config file: " + std::string(e.what()));
    }
    return config;
}

std::vector<std::pair<std::string, std::string>> Utils::create_training_data() {
    std::vector<std::pair<std::string, std::string>> training_pairs;
    std::filesystem::path exe_path = std::filesystem::current_path().parent_path();
    std::filesystem::path data_dir = exe_path / "data";
    std::filesystem::path file_path = data_dir / "training_pairs.txt";

    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open training data file: " + file_path.string());
    }

    std::string line;
    while (std::getline(file, line)) {
        size_t delimiter_pos = line.find('|');
        if (delimiter_pos != std::string::npos) {
            std::string input = line.substr(0, delimiter_pos);
            std::string output = line.substr(delimiter_pos + 1);
            training_pairs.emplace_back(input, output);
        }
    }

    // Apply data augmentation
    DataAugmentation augmenter(0.3f, 0.3f);
    auto augmented_pairs = augmenter.augmentDataset(training_pairs);
    
    std::cout << "\n=== Data Augmentation Analysis ===\n";
    std::cout << "Original pairs: " << training_pairs.size() << "\n";
    std::cout << "Augmented pairs: " << augmented_pairs.size() << "\n";
    
    // Compare original and augmented examples
    std::cout << "\nSample augmentations:\n";
    for (size_t i = 0; i < std::min(size_t(5), training_pairs.size()); i++) {
        std::cout << "\nOriginal: '" << training_pairs[i].first 
                  << "' -> '" << training_pairs[i].second << "'\n";
        
        // Find augmentations of this example
        std::cout << "Augmentations:\n";
        for (const auto& [aug_input, aug_output] : augmented_pairs) {
            if (aug_input.find(training_pairs[i].first) != std::string::npos ||
                training_pairs[i].first.find(aug_input) != std::string::npos) {
                std::cout << "  '" << aug_input << "' -> '" << aug_output << "'\n";
            }
        }
    }
    
    // Keep the malformed pairs check
    int malformed_count = 0;
    std::cout << "\nChecking for malformed pairs...\n";
    for (const auto& [input, output] : training_pairs) {
        if (input.empty() || output.empty()) {
            std::cout << "Empty input or output found\n";
            malformed_count++;
            continue;
        }
        
        if (input.length() < 5 || output.length() < 2) {
            std::cout << "Suspiciously short pair: '" << input << "' -> '" << output << "'\n";
            malformed_count++;
        }
    }
    
    if (malformed_count > 0) {
        std::cout << "Warning: Found " << malformed_count << " potentially malformed pairs\n";
    }
    
    analyze_training_patterns(training_pairs);
    
    return training_pairs;
}

void Utils::analyze_token_mappings(
    const std::vector<std::pair<std::string, std::string>>& training_data,
    const Tokenizer& tokenizer) {
    std::cout << "\n=== Analyzing Token Mappings ===\n";
    size_t total_words = 0;
    size_t unknown_tokens = 0;
    std::unordered_map<std::string, int> unknown_words;

    for (const auto& pair : training_data) {
        std::string processed_input = pair.first;
        tokenizer.preprocess_text(processed_input);
        std::vector<int> tokens = tokenizer.encode(processed_input);

        for (int token : tokens) {
            if (!tokenizer.is_special_token(token)) {
                total_words++;
                if (tokenizer.decode({token}) == " ") {
                    unknown_tokens++;
                    unknown_words[tokenizer.decode({token})]++;
                }
            }
        }
    }

    std::cout << "Token Mapping Statistics:\n"
              << "Total words: " << total_words << "\n"
              << "Unknown tokens: " << unknown_tokens << " ("
              << (100.0f * unknown_tokens / total_words) << "%)\n";
}

std::vector<std::pair<std::string, std::string>> Utils::load_validation_data() {
    std::vector<std::pair<std::string, std::string>> validation_pairs;
    std::filesystem::path exe_path = std::filesystem::current_path().parent_path();
    std::filesystem::path data_dir = exe_path / "data";
    std::filesystem::path file_path = data_dir / "validation_pairs.txt";

    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open validation data file: " + file_path.string());
    }

    std::string line;
    while (std::getline(file, line)) {
        size_t delimiter_pos = line.find('|');
        if (delimiter_pos != std::string::npos) {
            validation_pairs.emplace_back(line.substr(0, delimiter_pos),
                                          line.substr(delimiter_pos + 1));
        }
    }
    return validation_pairs;
}

bool Utils::validate_input_sequence(const std::vector<int>& tokens, size_t vocab_size,
                                    size_t max_seq_length) {
    // For target sequences, we allow empty sequences
    if (tokens.empty()) {
        return true;  // Empty sequences are valid for targets
    }

    // For non-empty sequences, check length if max_seq_length is specified
    if (max_seq_length > 0 && tokens.size() > max_seq_length) {
        std::cout << "Invalid sequence: too long (length: " << tokens.size() 
                  << ", max: " << max_seq_length << ")" << std::endl;
        return false;
    }

    // Validate each token
    for (int token : tokens) {
        if (token < 0 || static_cast<size_t>(token) >= vocab_size) {
            std::cout << "Invalid token " << token << " (vocab size: " << vocab_size << ")" << std::endl;
            return false;
        }
    }
    return true;
}

void Utils::print_matrix(const Matrix& m, const std::string& name, size_t max_rows,
                         size_t max_cols) {
    std::cout << "\n" << name << " (" << m.rows() << "x" << m.cols() << "):\n";
    for (size_t i = 0; i < std::min(max_rows, m.rows()); ++i) {
        for (size_t j = 0; j < std::min(max_cols, m.cols()); ++j) {
            std::cout << std::fixed << std::setprecision(4) << m(i, j) << " ";
        }
        std::cout << (m.cols() > max_cols ? "..." : "") << "\n";
    }
    if (m.rows() > max_rows) {
        std::cout << "...\n";
    }
}

// Helper function to get multi-token predictions
std::vector<std::pair<std::string, float>> Utils::get_multi_token_predictions(
    const Matrix& logits, const Tokenizer& tokenizer, int beam_width) {
    
    const int last_pos = logits.rows() - 1;
    std::vector<std::pair<std::string, float>> predictions;
    
    // Get top tokens and their probabilities
    std::vector<std::pair<float, int>> token_probs;
    for (int j = 0; j < logits.cols(); j++) {
        token_probs.push_back({logits(last_pos, j), j});
    }
    
    // Sort by probability
    std::sort(token_probs.begin(), token_probs.end(), std::greater<>());
    
    // Take top beam_width tokens
    for (int i = 0; i < std::min(beam_width, static_cast<int>(token_probs.size())); i++) {
        int token_id = token_probs[i].second;
        float prob = token_probs[i].first;
        
        // Skip special tokens
        if (tokenizer.is_special_token(token_id)) continue;
        
        // Decode token
        std::vector<int> token_seq = {token_id};
        std::string decoded = tokenizer.decode(token_seq);
        
        if (!decoded.empty()) {
            predictions.push_back({decoded, prob});
        }
    }
    
    return predictions;
}

void Utils::print_top_predictions(const Matrix& logits, const Tokenizer& tokenizer, 
                                Transformer& transformer, int k) {
    // Add subword analysis
    std::cout << "\nInput tokenization analysis:\n";
    std::string test_input = "The weather is";
    std::vector<int> tokens = tokenizer.encode(test_input);
    std::cout << "Tokens: ";
    for (int token : tokens) {
        std::cout << "'" << tokenizer.decode({token}) << "' ";
    }
    std::cout << "\n";

    // Track subword combinations
    std::map<std::string, float> complete_words;
    std::vector<std::pair<float, std::vector<int>>> top_sequences;

    // Try to build complete words from subword combinations
    for (size_t i = 0; i < logits.cols(); ++i) {
        float score = logits(logits.rows() - 1, i);
        std::string token = tokenizer.decode({static_cast<int>(i)});
        
        // Look ahead for potential next subwords
        std::vector<int> sequence = {static_cast<int>(i)};
        std::string current_word = token;
        
        // Try combining with next likely tokens
        Matrix next_hidden = transformer.forward(sequence, "", tokenizer);
        Matrix next_logits = transformer.get_lm_head()->project_to_vocab(next_hidden);
        
        // Get top 5 next tokens
        std::vector<std::pair<float, int>> next_tokens;
        for (size_t j = 0; j < next_logits.cols(); ++j) {
            next_tokens.push_back({next_logits(0, j), j});
        }
        std::partial_sort(next_tokens.begin(), next_tokens.begin() + 5, next_tokens.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
            
        // Try combining with each next token
        for (size_t j = 0; j < 5; ++j) {
            std::string combined = current_word + tokenizer.decode({next_tokens[j].first});
            if (!combined.empty() && combined[0] != ' ') {  // Likely a subword continuation
                complete_words[combined] = score + next_tokens[j].first;
            }
        }
    }

    // Sort and print complete word predictions
    std::vector<std::pair<float, std::string>> sorted_words;
    for (const auto& [word, score] : complete_words) {
        sorted_words.push_back({score, word});
    }
    std::sort(sorted_words.rbegin(), sorted_words.rend());

    std::cout << "\nTop complete word predictions:\n";
    for (size_t i = 0; i < std::min(size_t(k), sorted_words.size()); ++i) {
        std::cout << i + 1 << ". \"" << sorted_words[i].second 
                  << "\" (score=" << sorted_words[i].first << ")\n";
    }
}

float Utils::evaluate_validation(
    Transformer& transformer, const Tokenizer& tokenizer,
    const std::vector<std::pair<std::string, std::string>>& validation_data) {
    std::cout << "\n=== Evaluating Validation Data ===\n";

    float total_loss = 0.0f;
    size_t correct_predictions = 0;
    size_t total_predictions = 0;

    // Validate we have data to process
    if (validation_data.empty()) {
        std::cout << "Warning: Empty validation data\n";
        return 0.0f;
    }

    transformer.set_training(false); // Set model to evaluation mode

    for (const auto& pair : validation_data) {
        // Preprocess input
        std::string processed_input = pair.first;
        std::cout << "Processing input: '" << processed_input << "'\n";
        tokenizer.preprocess_text(processed_input);
        std::cout << "Preprocessed input: '" << processed_input << "'\n";

        std::vector<int> input_tokens = tokenizer.encode(processed_input);
        std::cout << "Encoded input tokens: ";
        for (int token : input_tokens) {
            std::cout << token << " ";
        }
        std::cout << "\n";

        // Skip empty sequences
        if (input_tokens.empty()) {
            std::cout << "Warning: Empty input tokens, skipping\n";
            continue;
        }

        // Validate input tokens
        if (!Utils::validate_input_sequence(input_tokens, tokenizer.vocab_size())) {
            std::cout << "Warning: Invalid input sequence, skipping\n";
            continue;
        }

        try {
            // Get model prediction
            std::cout << "Calling transformer.forward with " << input_tokens.size() << " tokens\n";
            Matrix output = transformer.forward(input_tokens, "", tokenizer);
            std::cout << "Forward pass output shape: " << output.rows() << "x" << output.cols()
                      << "\n";

            if (output.rows() == 0 || output.cols() == 0) {
                std::cout << "Warning: Empty output from transformer, skipping\n";
                continue;
            }

            auto lm_head = transformer.get_lm_head();
            if (!lm_head) {
                std::cerr << "Error: Language model head not initialized. Initializing now...\n";
                std::cout << "Error: Null language model head\n";
                continue;
            }

            Matrix logits = lm_head->project_to_vocab(output);
            std::cout << "Logits shape: " << logits.rows() << "x" << logits.cols() << "\n";

            // For single token prediction, we don't need beam search
            // Just get the highest probability token directly
            int predicted_token = -1;
            float max_logit = -std::numeric_limits<float>::infinity();
            
            for (size_t i = 0; i < logits.cols(); ++i) {
                float val = logits(logits.rows() - 1, i);
                if (val > max_logit) {
                    max_logit = val;
                    predicted_token = i;
                }
            }

            // Get target
            std::string processed_target = pair.second;
            tokenizer.preprocess_text(processed_target);
            std::vector<int> target_tokens = tokenizer.encode(processed_target);

            // Create target distribution
            Matrix target_distribution(1, tokenizer.vocab_size(), 0.0f);
            if (!target_tokens.empty()) {
                target_distribution(0, target_tokens.back()) = 1.0f;
            }

            // Compute loss using only the last token's prediction
            Matrix last_token_logits(1, logits.cols());
            for (size_t i = 0; i < logits.cols(); ++i) {
                last_token_logits(0, i) = logits(logits.rows() - 1, i);
            }

            float batch_loss = Utils::compute_batch_loss(last_token_logits, target_distribution, tokenizer);
            total_loss += batch_loss;

            // Check if prediction matches target
            if (!target_tokens.empty() && predicted_token == target_tokens.back()) {
                correct_predictions++;
            }
            total_predictions++;

        } catch (const std::exception& e) {
            std::cout << "Error evaluating validation: " << e.what() << "\n";
        }
    }

    transformer.set_training(true); // Reset to training mode
    return total_predictions > 0 ? total_loss / total_predictions : 0.0f;
}

void Utils::apply_sampling_parameters(std::vector<float>& logits, float temperature, float top_p) {
    // Apply temperature scaling first
    if (temperature != 1.0f) {
        for (auto& logit : logits) {
            logit /= temperature;
        }
    }

    // Apply top-p (nucleus) sampling if enabled
    if (top_p < 1.0f) {
        // Convert logits to probabilities
        std::vector<std::pair<float, size_t>> probs_with_indices;
        float max_logit = *std::max_element(logits.begin(), logits.end());
        float sum_exp = 0.0f;

        for (size_t i = 0; i < logits.size(); i++) {
            float prob = std::exp(logits[i] - max_logit);
            sum_exp += prob;
            probs_with_indices.push_back({prob, i});
        }

        // Normalize probabilities
        for (auto& pair : probs_with_indices) {
            pair.first /= sum_exp;
        }

        // Sort by probability in descending order
        std::sort(probs_with_indices.begin(), probs_with_indices.end(),
                  std::greater<std::pair<float, size_t>>());

        // Find cutoff index for top-p
        float cumsum = 0.0f;
        size_t cutoff_idx = probs_with_indices.size() - 1;
        for (size_t i = 0; i < probs_with_indices.size(); i++) {
            cumsum += probs_with_indices[i].first;
            if (cumsum > top_p) {
                cutoff_idx = i;
                break;
            }
        }

        // Create mask for filtered tokens
        std::vector<bool> keep_token(logits.size(), false);
        for (size_t i = 0; i <= cutoff_idx; i++) {
            keep_token[probs_with_indices[i].second] = true;
        }

        // Apply mask to logits
        for (size_t i = 0; i < logits.size(); i++) {
            if (!keep_token[i]) {
                logits[i] = -std::numeric_limits<float>::infinity();
            }
        }
    }
}

std::vector<std::string>& Utils::get_vocabulary(const Tokenizer& tokenizer) {
    static std::vector<std::string> vocabulary;
    if (vocabulary.empty()) {
        vocabulary.reserve(tokenizer.vocab_size());
        
        // Fill vocabulary with all possible token strings
        for (size_t i = 0; i < tokenizer.vocab_size(); i++) {
            vocabulary.push_back(tokenizer.decode({static_cast<int>(i)}));
        }
        std::cout << "Loaded vocabulary with " << vocabulary.size() << " tokens" << std::endl;
    }
    return vocabulary;
}

std::vector<size_t> topKSampling(const std::vector<float>& probabilities, size_t k) {
    std::vector<std::pair<float, size_t>> prob_idx;
    for (size_t i = 0; i < probabilities.size(); i++) {
        prob_idx.push_back({probabilities[i], i});
    }
    
    // Sort by probability in descending order
    std::partial_sort(prob_idx.begin(), prob_idx.begin() + k, prob_idx.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Return top k indices
    std::vector<size_t> result;
    for (size_t i = 0; i < k; i++) {
        result.push_back(prob_idx[i].second);
    }
    return result;
}

std::vector<size_t> nucleusSampling(const std::vector<float>& probabilities, float p) {
    std::vector<std::pair<float, size_t>> sorted_probs;
    for (size_t i = 0; i < probabilities.size(); i++) {
        sorted_probs.push_back({probabilities[i], i});
    }
    
    std::sort(sorted_probs.begin(), sorted_probs.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });
    
    float cumsum = 0.0f;
    std::vector<size_t> result;
    for (const auto& pair : sorted_probs) {
        cumsum += pair.first;
        result.push_back(pair.second);
        if (cumsum >= p) break;
    }
    return result;
}

struct PatternInfo {
    std::string pattern;
    float frequency_weight;  // Based on common usage
    std::string example;
};

const std::vector<PatternInfo> weighted_patterns = {
    {" to the", 1.0f, "most common"},
    {" in the", 0.9f, "very common"},
    {" according to", 0.5f, "less common"},
    // ...
};

PatternMetrics Utils::evaluate_pattern_completion(
    Transformer& transformer,
    const Tokenizer& tokenizer,
    const std::vector<std::pair<std::string, std::string>>& validation_data) {
    
    PatternMetrics metrics{0.0f, 0.0f, {}};
    size_t total = 0;
    size_t pattern_correct = 0;
    size_t destination_correct = 0;
    std::map<std::string, int> error_counts;

    // Define the expected pattern structure
    const std::vector<std::string> action_verbs = {
        "work", "assist", "organize", "serve", "create", "operate", "study",
        "design", "build", "test", "analyze", "develop", "research", "manage",
        "monitor", "code", "train", "teach", "maintain", "process"
    };

    // Function to check if a string follows job title pattern (typically 1-3 words)
    auto is_job_title = [](const std::string& s) {
        std::stringstream ss(s);
        std::string word;
        int word_count = 0;
        while (ss >> word && word_count < 4) {
            word_count++;
        }
        return word_count >= 1 && word_count <= 3;
    };

    // Function to check if a string is a specialized location (typically 2-3 words)
    auto is_specialized_location = [](const std::string& s) {
        std::stringstream ss(s);
        std::string word;
        std::vector<std::string> words;
        while (ss >> word) {
            words.push_back(word);
        }
        // Location should be 1-3 words and often ends with: lab, room, center, studio, etc.
        static const std::vector<std::string> common_endings = {
            "lab", "room", "center", "studio", "office", "facility", "station"
        };
        if (words.size() < 1 || words.size() > 3) return false;
        
        // Check if last word is a common location ending
        std::string last_word = words.back();
        return std::find(common_endings.begin(), common_endings.end(), last_word) 
               != common_endings.end();
    };

    transformer.set_training(false);
    
    for (const auto& [input, target] : validation_data) {
        // Add debug prints
        std::cout << "\nProcessing example:"
                  << "\nInput: '" << input << "'"
                  << "\nTarget: '" << target << "'" << std::endl;
        
        // Only process inputs that end with "in the"
        if (!ends_with(input, " in the")) continue;

        // Process and get prediction
        std::string processed_input = input;
        tokenizer.preprocess_text(processed_input);
        std::vector<int> input_tokens = tokenizer.encode(processed_input);
        Matrix hidden = transformer.forward(input_tokens, input, tokenizer);
        Matrix logits = transformer.get_lm_head()->project_to_vocab(hidden);
        
        // Get prediction
        int predicted_token = -1;
        float max_logit = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < logits.cols(); ++i) {
            if (logits(logits.rows() - 1, i) > max_logit) {
                max_logit = logits(logits.rows() - 1, i);
                predicted_token = i;
            }
        }
        
        std::string prediction = tokenizer.decode({predicted_token});
        std::string expected = target;

        // Check if prediction follows the specialized location pattern
        bool follows_pattern = is_specialized_location(prediction);
        
        if (follows_pattern) {
            pattern_correct++;
        } else {
            error_counts[prediction]++;
            std::cout << "Pattern error - Input: \"" << input << "\", Predicted: \"" 
                      << prediction << "\", Expected: \"" << expected << "\"\n";
        }
        
        // Check exact match
        if (prediction == expected) {
            destination_correct++;
        }
        
        total++;

        // Print top predictions for debugging
        std::cout << "Top 5 predictions for '" << input << "':\n";
        std::vector<std::pair<float, int>> top_logits;
        for (size_t i = 0; i < logits.cols(); ++i) {
            top_logits.push_back({logits(logits.rows() - 1, i), i});
        }
        std::partial_sort(top_logits.begin(), top_logits.begin() + 5, top_logits.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
        for (int i = 0; i < 5; ++i) {
            std::cout << "  " << tokenizer.decode({top_logits[i].second}) 
                      << " (score: " << top_logits[i].first << ")\n";
        }
    }
    
    transformer.set_training(true);
    
    // Calculate metrics
    metrics.pattern_accuracy = total > 0 ? static_cast<float>(pattern_correct) / total : 0.0f;
    metrics.destination_accuracy = total > 0 ? static_cast<float>(destination_correct) / total : 0.0f;
    
    // Get top 3 most common mistakes
    std::vector<std::pair<int, std::string>> sorted_errors;
    for (const auto& [error, count] : error_counts) {
        sorted_errors.push_back({count, error});
    }
    std::sort(sorted_errors.rbegin(), sorted_errors.rend());
    
    for (size_t i = 0; i < std::min(size_t(3), sorted_errors.size()); ++i) {
        metrics.common_mistakes.push_back(sorted_errors[i].second);
    }
    
    // Print detailed metrics
    std::cout << "\nPattern Completion Metrics:\n"
              << "Total examples evaluated: " << total << "\n"
              << "Pattern accuracy: " << (metrics.pattern_accuracy * 100.0f) << "%\n"
              << "Destination accuracy: " << (metrics.destination_accuracy * 100.0f) << "%\n"
              << "Most common mistakes:\n";
    
    for (const auto& mistake : metrics.common_mistakes) {
        std::cout << "  - \"" << mistake << "\" (count: " << error_counts[mistake] << ")\n";
    }
    std::cout << std::endl;
    
    return metrics;
}

void Utils::analyze_training_patterns(const std::vector<std::pair<std::string, std::string>>& pairs) {
    std::cout << "\n=== Detailed Training Data Analysis ===\n";
    
    // Categories for different types of jobs
    std::map<std::string, std::vector<std::string>> job_patterns = {
        {"tech", {"engineer", "developer", "programmer", "architect", "analyst"}},
        {"science", {"scientist", "researcher", "specialist", "expert"}},
        {"creative", {"designer", "artist", "creator", "composer"}},
        {"service", {"worker", "assistant", "manager", "coordinator"}},
        {"craft", {"maker", "crafter", "builder", "smith"}}
    };

    // Initialize categories with proper JobCategory structs
    std::map<std::string, JobCategory> categories;
    for (const auto& [category, patterns] : job_patterns) {
        JobCategory cat;
        cat.category = category;
        cat.job_titles = patterns;  // Initialize with the patterns
        categories[category] = cat;
    }
    
    // Track statistics
    std::map<std::string, int> verb_counts;
    std::map<std::string, std::set<std::string>> verb_locations;  // what locations each verb is used with
    std::map<std::string, std::set<std::string>> location_endings;  // categorize by ending (lab, room, etc)
    std::map<std::string, JobCategory> job_stats;
    
    for (const auto& [input, output] : pairs) {
        std::stringstream ss(input);
        std::vector<std::string> words;
        std::string word;
        while (ss >> word) {
            words.push_back(word);
        }
        
        if (words.size() < 4) continue;  // Skip malformed entries
        
        // Extract components
        std::string job_title;
        std::string verb;
        size_t verb_pos = 0;
        
        // Find the verb (word before "in the")
        for (size_t i = 0; i < words.size(); i++) {
            if (words[i] == "in" && i + 1 < words.size() && words[i + 1] == "the") {
                verb = words[i - 1];
                verb_pos = i - 1;
                break;
            }
        }
        
        // Get job title (everything before the verb)
        for (size_t i = 0; i < verb_pos; i++) {
            if (i > 0) job_title += " ";
            job_title += words[i];
        }
        
        // Categorize job
        std::string job_category = "other";
        for (const auto& [category, patterns] : categories) {
            for (const auto& pattern : patterns.job_titles) {
                if (job_title.find(pattern) != std::string::npos) {
                    job_category = category;
                    break;
                }
            }
            if (job_category != "other") break;
        }
        
        // Update statistics
        verb_counts[verb]++;
        verb_locations[verb].insert(output);
        
        // Analyze location endings
        std::stringstream loc_ss(output);
        std::vector<std::string> loc_words;
        while (loc_ss >> word) {
            loc_words.push_back(word);
        }
        if (!loc_words.empty()) {
            std::string ending = loc_words.back();
            location_endings[ending].insert(output);
        }
        
        // Update job category stats
        job_stats[job_category].job_titles.push_back(job_title);
        job_stats[job_category].common_verbs.push_back(verb);
        job_stats[job_category].common_locations.push_back(output);
    }
    
    // Print analysis
    std::cout << "\n1. Verb Analysis:\n";
    std::cout << "Top 10 most common verbs:\n";
    std::vector<std::pair<int, std::string>> sorted_verbs;
    for (const auto& [verb, count] : verb_counts) {
        sorted_verbs.push_back({count, verb});
    }
    std::sort(sorted_verbs.rbegin(), sorted_verbs.rend());
    for (size_t i = 0; i < std::min(size_t(10), sorted_verbs.size()); i++) {
        std::cout << sorted_verbs[i].second << ": " << sorted_verbs[i].first 
                  << " uses\n";
        std::cout << "  Sample locations: ";
        size_t loc_count = 0;
        for (const auto& loc : verb_locations[sorted_verbs[i].second]) {
            if (loc_count++ < 3) std::cout << "'" << loc << "' ";
        }
        std::cout << "\n";
    }
    
    std::cout << "\n2. Location Analysis:\n";
    std::cout << "Common location endings:\n";
    for (const auto& [ending, locations] : location_endings) {
        std::cout << ending << ": " << locations.size() << " unique locations\n";
        std::cout << "  Examples: ";
        size_t count = 0;
        for (const auto& loc : locations) {
            if (count++ < 3) std::cout << "'" << loc << "' ";
        }
        std::cout << "\n";
    }
    
    std::cout << "\n3. Job Category Analysis:\n";
    for (const auto& [category, stats] : job_stats) {
        std::cout << "\nCategory: " << category << "\n";
        std::cout << "  Total jobs: " << stats.job_titles.size() << "\n";
        
        // Most common job titles
        std::map<std::string, int> title_counts;
        for (const auto& title : stats.job_titles) {
            title_counts[title]++;
        }
        std::cout << "  Common titles: ";
        std::vector<std::pair<int, std::string>> sorted_titles;
        for (const auto& [title, count] : title_counts) {
            sorted_titles.push_back({count, title});
        }
        std::sort(sorted_titles.rbegin(), sorted_titles.rend());
        for (size_t i = 0; i < std::min(size_t(3), sorted_titles.size()); i++) {
            std::cout << "'" << sorted_titles[i].second << "' (" 
                      << sorted_titles[i].first << ") ";
        }
        std::cout << "\n";
        
        // Most common verbs for this category
        std::map<std::string, int> cat_verb_counts;
        for (const auto& verb : stats.common_verbs) {
            cat_verb_counts[verb]++;
        }
        std::cout << "  Common verbs: ";
        std::vector<std::pair<int, std::string>> sorted_cat_verbs;
        for (const auto& [verb, count] : cat_verb_counts) {
            sorted_cat_verbs.push_back({count, verb});
        }
        std::sort(sorted_cat_verbs.rbegin(), sorted_cat_verbs.rend());
        for (size_t i = 0; i < std::min(size_t(3), sorted_cat_verbs.size()); i++) {
            std::cout << "'" << sorted_cat_verbs[i].second << "' (" 
                      << sorted_cat_verbs[i].first << ") ";
        }
        std::cout << "\n";
    }
}

bool is_valid_completion(const std::string& text, const Tokenizer& tokenizer) {
    // Tokenize and check if it forms valid words
    std::vector<int> tokens = tokenizer.encode(text);
    std::string reconstructed = tokenizer.decode(tokens);
    
    // Check if reconstruction matches original (indicates valid words)
    if (reconstructed != text) {
        return false;
    }
    
    // Check if it follows location pattern
    std::vector<std::string> words;
    std::stringstream ss(text);
    std::string word;
    while (ss >> word) {
        words.push_back(word);
    }
    
    // Check for common location endings
    static const std::vector<std::string> endings = {
        "lab", "room", "center", "studio", "office", "facility"
    };
    
    if (!words.empty()) {
        std::string last_word = words.back();
        return std::find(endings.begin(), endings.end(), last_word) != endings.end();
    }
    return false;
}