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
#include <regex>
#include <sstream>
#include <set>
#include <unordered_set>
#include <thread>
#include "../include/data_augmentation.hpp"
#include <chrono>
#include <mutex>
#include "../include/debug.hpp"

namespace debug {
    bool verbose_logging = true;
    std::ofstream debug_log;
    const std::string log_file = "../build/transformer.log";
    const std::string progress_file = "../build/progress.log";
    std::mutex log_mutex;
    std::ofstream progress_log;
    
    // Track current progress state
    struct ProgressState {
        std::string current_fold;
        std::string current_epoch;
        std::string current_batch;
        std::string current_validation;
    } progress_state;
    
    void init_logging() {
        std::lock_guard<std::mutex> lock(log_mutex);
        debug_log.open(log_file, std::ios::app);
        // Open with truncation only at start
        progress_log.open(progress_file, std::ios::trunc);
        
        if (!debug_log.is_open()) {
            std::cerr << "Warning: Could not open debug log file: " << log_file << std::endl;
            return;
        }
        if (!progress_log.is_open()) {
            std::cerr << "Warning: Could not open progress log file: " << progress_file << std::endl;
            return;
        }
        
        auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        debug_log << "\n=== New Debug Session Started at " 
                 << std::ctime(&now_time) 
                 << "===\n" << std::endl;
        
        // Write initial header
        progress_log << "\n=== Training Session Progress ===\n\n";
        progress_log.flush();
    }

    void update_progress_file() {
        // Reopen file in write mode
        progress_log.close();
        progress_log.open(progress_file);
        
        // Write fixed header
        progress_log << "\n=== Training Session Progress ===\n\n";
        
        // Write current state with proper formatting
        if (!progress_state.current_fold.empty()) {
            progress_log << progress_state.current_fold << "\n";
            
            if (!progress_state.current_epoch.empty()) {
                progress_log << "  " << progress_state.current_epoch << "\n";
                
                if (!progress_state.current_batch.empty()) {
                    progress_log << "    " << progress_state.current_batch << "\n";
                }
                
                if (!progress_state.current_validation.empty()) {
                    progress_log << "    " << progress_state.current_validation << "\n";
                }
            }
        }
        
        // Ensure immediate write to disk
        progress_log.flush();
        progress_log.close();
        progress_log.open(progress_file, std::ios::app);
    }
    
    void log_message(const std::string& message, const std::string& level) {
        if (!verbose_logging) return;
        std::lock_guard<std::mutex> lock(log_mutex);
        auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        std::string time_str = std::ctime(&now_time);
        time_str = time_str.substr(0, time_str.length() - 1);  // Remove newline
        
        debug_log << "[" << time_str << "] [" << level << "] " << message << std::endl;
        debug_log.flush();
    }
    
    void log_vector(const std::vector<int>& vec, const std::string& name) {
        if (!verbose_logging) return;
        std::ostringstream oss;
        oss << name << " [size=" << vec.size() << "]: ";
        for (size_t i = 0; i < std::min(vec.size(), size_t(10)); ++i) {
            oss << vec[i] << " ";
        }
        if (vec.size() > 10) oss << "...";
        log_message(oss.str(), "DEBUG");
    }
    
    void log_matrix(const Matrix& mat, const std::string& label) {
        if (!verbose_logging) return;
        std::ostringstream oss;
        oss << label << " [" << mat.rows() << "x" << mat.cols() << "]";
        if (mat.rows() <= 5 && mat.cols() <= 5) {
            oss << ":\n";
            for (size_t i = 0; i < mat.rows(); i++) {
                for (size_t j = 0; j < mat.cols(); j++) {
                    oss << std::fixed << std::setprecision(4) << mat(i,j) << " ";
                }
                oss << "\n";
            }
        }
        log_message(oss.str(), "DEBUG");
    }
    
    void log_token_distribution(const Matrix& dist, const std::string& name) {
        if (!verbose_logging) return;
        std::ostringstream oss;
        oss << name << " statistics:\n";
        float min_val = std::numeric_limits<float>::max();
        float max_val = std::numeric_limits<float>::lowest();
        float sum = 0.0f;
        size_t zeros = 0;
        
        for (size_t i = 0; i < dist.rows(); ++i) {
            for (size_t j = 0; j < dist.cols(); ++j) {
                float val = dist(i, j);
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
                sum += val;
                if (val == 0.0f) zeros++;
            }
        }
        
        oss << "  Min: " << min_val << "\n"
            << "  Max: " << max_val << "\n"
            << "  Mean: " << sum / (dist.rows() * dist.cols()) << "\n"
            << "  Zero elements: " << zeros << "/" << (dist.rows() * dist.cols()) 
            << " (" << (100.0f * zeros / (dist.rows() * dist.cols())) << "%)";
        log_message(oss.str(), "DEBUG");
    }

    void log_progress(const std::string& stage, size_t current, size_t total, 
                     const std::string& additional_info = "") {
        std::lock_guard<std::mutex> lock(log_mutex);
        auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        std::string time_str = std::ctime(&now_time);
        time_str = time_str.substr(0, time_str.length() - 1);  // Remove newline

        float progress = static_cast<float>(current) / total;
        int bar_width = 50;
        int filled_width = static_cast<int>(bar_width * progress);

        std::ostringstream progress_bar;
        progress_bar << stage << ": [";
        for (int i = 0; i < bar_width; ++i) {
            if (i < filled_width) progress_bar << "=";
            else if (i == filled_width) progress_bar << ">";
            else progress_bar << " ";
        }
        progress_bar << "] " << std::fixed << std::setprecision(1) << (progress * 100.0f) << "%";
        if (!additional_info.empty()) {
            progress_bar << " | " << additional_info;
        }

        // Update appropriate progress state
        std::string progress_line = progress_bar.str();
        bool should_update = false;

        if (stage == "Cross-Validation") {
            progress_state.current_fold = progress_line;
            should_update = true;
        } else if (stage == "Epoch") {
            progress_state.current_epoch = progress_line;
            should_update = true;
        } else if (stage == "Batch") {
            progress_state.current_batch = progress_line;
            should_update = true;
        } else if (stage == "Validation") {
            progress_state.current_validation = progress_line;
            progress_state.current_batch.clear();  // Clear batch progress during validation
            should_update = true;
        } else if (stage == "Fold Complete" || stage == "Cross-Validation Complete") {
            progress_state = ProgressState();
            should_update = true;
        }

        if (should_update) {
            update_progress_file();
        }
    }
}

// Initialize static members
std::random_device Utils::rd;
std::mt19937 Utils::random_generator;
std::atomic<uint64_t> Utils::prediction_counter(0);

bool starts_with(const std::string& str, const std::string& prefix) {
    return str.size() >= prefix.size() && 
           str.compare(0, prefix.size(), prefix) == 0;
}

Matrix Utils::create_batch_target_distribution(
    const std::vector<std::vector<int>>& target_tokens,
    const TiktokenTokenizer& tokenizer,
    size_t vocab_size,
    size_t input_max_seq_len) {
    debug::init_logging();
    debug::debug_log << "\nCreating target distribution for batch:" << std::endl;
    
    size_t batch_size = target_tokens.size();
    size_t total_tokens = batch_size * input_max_seq_len;
    
    // Create sparse matrix representation
    std::vector<std::tuple<size_t, size_t, float>> non_zero_elements;
    non_zero_elements.reserve(batch_size); // Reserve space for efficiency
    
    size_t current_pos = 0;
    for (size_t seq = 0; seq < target_tokens.size(); seq++) {
        const auto& sequence = target_tokens[seq];
        
        for (size_t i = 0; i < input_max_seq_len && i < sequence.size(); i++) {
            if (i == sequence.size() - 1) {
                int token_id = sequence[i];
                if (token_id >= 0 && static_cast<size_t>(token_id) < vocab_size) {
                    non_zero_elements.emplace_back(current_pos + i, token_id, 1.0f);
                }
            }
        }
        current_pos += input_max_seq_len;
    }
    
    // Create sparse matrix
    Matrix sparse_target(total_tokens, vocab_size, 0.0f);
    for (const auto& [row, col, val] : non_zero_elements) {
        sparse_target(row, col) = val;
    }
    
    return sparse_target;
}

float Utils::compute_batch_loss(
    const Matrix& logits,
    const Matrix& target_distribution,
    const TiktokenTokenizer& tokenizer) {
    // Input validation with detailed error messages
    if (logits.empty() || target_distribution.empty()) {
        std::cout << "Logits shape: " << (logits.empty() ? "empty" : 
                  (std::to_string(logits.rows()) + "x" + std::to_string(logits.cols()))) << std::endl;
        std::cout << "Target distribution shape: " << (target_distribution.empty() ? "empty" : 
                  (std::to_string(target_distribution.rows()) + "x" + std::to_string(target_distribution.cols()))) << std::endl;
        throw std::runtime_error("Empty logits or target distribution in compute_batch_loss");
    }

    // Debug dimensions
    std::cout << "Computing loss with:"
              << "\n- Logits shape: " << logits.rows() << "x" << logits.cols()
              << "\n- Target shape: " << target_distribution.rows() << "x" << target_distribution.cols() << std::endl;

    // Ensure logits are projected to vocabulary space if needed
    Matrix projected_logits;
    if (logits.cols() != target_distribution.cols()) {
        std::cout << "Projecting logits from hidden dimension to vocabulary dimension" << std::endl;
        // Assuming we have a projection matrix or method to convert from hidden size to vocab size
        // This should be handled by the language model head before calling compute_batch_loss
        throw std::runtime_error("Dimension mismatch: logits must be projected to vocabulary space before computing loss");
    }

    float total_loss = 0.0f;
    const size_t batch_size = logits.rows();
    const size_t vocab_size = logits.cols();
    const float epsilon = 1e-7f;
    const float max_loss_per_token = 100.0f;

    // Pre-compute max logits for numerical stability
    std::vector<float> max_logits(batch_size, -std::numeric_limits<float>::infinity());
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < vocab_size; ++j) {
            max_logits[i] = std::max(max_logits[i], logits(i, j));
        }
    }

    // Compute loss with improved numerical stability
    for (size_t i = 0; i < batch_size; ++i) {
        float sequence_loss = 0.0f;
        float sum_exp = 0.0f;

        // First pass: compute denominator for softmax
        for (size_t j = 0; j < vocab_size; ++j) {
            float shifted_logit = logits(i, j) - max_logits[i];
            sum_exp += std::exp(shifted_logit);
        }
        sum_exp = std::max(sum_exp, epsilon);

        // Second pass: compute cross-entropy loss
        for (size_t j = 0; j < vocab_size; ++j) {
            if (target_distribution(i, j) > 0.0f) {
                float shifted_logit = logits(i, j) - max_logits[i];
                float log_prob = shifted_logit - std::log(sum_exp);
                log_prob = std::max(log_prob, -max_loss_per_token);
                float token_loss = -target_distribution(i, j) * log_prob;
                token_loss = std::min(token_loss, max_loss_per_token);
                sequence_loss += token_loss;
            }
        }

        if (std::isfinite(sequence_loss)) {
            total_loss += sequence_loss;
        } else {
            std::cout << "Warning: Non-finite sequence loss detected at position " << i << std::endl;
            total_loss += max_loss_per_token;
        }
    }

    float avg_loss = total_loss / static_cast<float>(batch_size);
    avg_loss = std::max(avg_loss, 1e-4f);  // Minimum loss floor

    std::cout << "Loss computation:"
              << "\n- Total loss: " << total_loss
              << "\n- Average loss: " << avg_loss
              << "\n- Batch size: " << batch_size << std::endl;

    return avg_loss;
}

TransformerConfig Utils::load_config(const std::string& config_path) {
    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
        throw std::runtime_error("Could not open config file: " + config_path);
    }

    nlohmann::json j;
    config_file >> j;

    TransformerConfig config;

    // Load tokenizer config first
    if (j.contains("tokenizer")) {
        const auto& tok = j["tokenizer"];
        config.tokenizer.use_subword = tok.value("use_subword", true);
        // Only set vocab_size if it exists in the config
        if (tok.contains("vocab_size")) {
            size_t vocab_size = tok["vocab_size"].get<size_t>();
            config.tokenizer.vocab_size = vocab_size;
            // Ensure model vocab size matches tokenizer vocab size
            if (j.contains("model")) {
                j["model"]["vocab_size"] = vocab_size;
            }
            std::cout << "Setting vocabulary size from config: " << vocab_size << std::endl;
        }
        config.tokenizer.model_path = tok.value("model_path", "model/tokenizer.model");
    }

    // Parse model settings
    if (j.contains("model")) {
        auto& model = j["model"];
        // Use tokenizer vocab size if available, otherwise use model vocab size
        config.vocab_size = config.tokenizer.vocab_size > 0 ? 
                           config.tokenizer.vocab_size : 
                           model.value("vocab_size", 32000);
        
        std::cout << "Final vocabulary size in config: " << config.vocab_size << std::endl;
        
        // Load other model settings
        config.hidden_size = model["hidden_size"];
        config.num_heads = model["num_heads"];
        config.num_layers = model["num_layers"];
        config.head_dim = model["head_dim"];
        config.intermediate_size = model["intermediate_size"];
    }

    // Parse training settings
    if (j.contains("training")) {
        auto& training = j["training"];
        config.batch_size = training.value("batch_size", 32);
        config.num_epochs = training.value("num_epochs", 3);
        config.dropout_rate = training.value("dropout_rate", 0.1f);
        config.weight_decay = training.value("weight_decay", 0.01f);
    }
    
    // Parse learning rate settings
    if (j.contains("learning_rate")) {
        auto& lr = j["learning_rate"];
        config.initial_lr = lr.value("initial_lr", 1e-4f);
        config.peak_lr = lr.value("peak_lr", 1e-3f);
        config.warmup_steps = lr.value("warmup_steps", 100);
        config.decay_factor = lr.value("decay_factor", 0.98f);
    }
    
    // Parse early stopping settings
    if (j.contains("early_stopping")) {
        auto& es = j["early_stopping"];
        config.early_stopping_patience = es.value("patience", 3);
        config.early_stopping_threshold = es.value("threshold", 1.5f);
    }
    
    // Parse optimization settings
    if (j.contains("optimization")) {
        auto& opt = j["optimization"];
        config.gradient_clip_threshold = opt.value("gradient_clip_threshold", 5.0f);
        config.layer_norm_epsilon = opt.value("layer_norm_epsilon", 1e-5f);
        config.gradient_accumulation_steps = opt.value("gradient_accumulation_steps", 4);
        config.use_gradient_checkpointing = opt.value("use_gradient_checkpointing", false);
        config.use_fp16 = opt.value("use_fp16", false);
        config.memory_pool_size = opt.value("memory_pool_size", 1024);
    }

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
        if (config.use_gqa) {
            if (attention.contains("num_kv_heads")) {
                config.num_kv_heads = attention["num_kv_heads"].get<size_t>();
            } else {
                config.num_kv_heads = config.num_heads / 2; // Default to half the heads
            }
        } else {
            config.num_kv_heads = config.num_heads;
        }
    }

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
    }

    // Parse checkpoint loading settings
    if (j.contains("load_from_checkpoint")) {
        config.load_from_checkpoint = j["load_from_checkpoint"].get<bool>();
        if (config.load_from_checkpoint && j.contains("checkpoint_to_load")) {
            config.checkpoint_to_load = j["checkpoint_to_load"].get<std::string>();
        }
    }

    // Parse token prediction settings
    if (j.contains("token_prediction")) {
        auto& tp = j["token_prediction"];
        config.token_prediction.temperature = tp.value("temperature", 1.0f);
        config.token_prediction.top_k = tp.value("top_k", 5);
        config.token_prediction.top_p = tp.value("top_p", 0.9f);
        config.token_prediction.frequency_penalty = tp.value("frequency_penalty", 0.1f);
        config.token_prediction.presence_penalty = tp.value("presence_penalty", 0.0f);
        config.token_prediction.min_token_prob = tp.value("min_token_prob", 0.05f);
        
        if (tp.contains("category_bonus")) {
            auto& cb = tp["category_bonus"];
            config.token_prediction.category_bonus.verb = cb.value("verb", 0.2f);
            config.token_prediction.category_bonus.adjective = cb.value("adjective", 0.2f);
            config.token_prediction.category_bonus.noun = cb.value("noun", 0.3f);
        }
    }

    // Parse cross validation settings
    if (j.contains("cross_validation")) {
        auto& cv = j["cross_validation"];
        config.num_folds = cv.value("num_folds", 5);
    }

    return config;
}

std::vector<std::pair<std::string, std::string>> Utils::create_training_data() {
    std::vector<std::pair<std::string, std::string>> training_pairs;
    std::filesystem::path file_path = "../data/training_pairs.txt";

    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open training data file: " + file_path.string());
    }

    std::string line;
    std::unordered_map<std::string, std::vector<std::pair<std::string, std::string>>> category_pairs;
    std::unordered_set<std::string> seen_pairs;
    
    // First, read all pairs and categorize them
    while (std::getline(file, line)) {
        // Skip empty lines
        if (line.empty()) continue;

        // Normalize separators
        std::string normalized_line = line;
        std::replace(normalized_line.begin(), normalized_line.end(), '#', '|');
        std::replace(normalized_line.begin(), normalized_line.end(), '*', '|');
        
        size_t delimiter_pos = normalized_line.find('|');
        if (delimiter_pos != std::string::npos) {
            std::string input = normalized_line.substr(0, delimiter_pos);
            std::string output = normalized_line.substr(delimiter_pos + 1);
            
            // Trim whitespace
            input = std::regex_replace(input, std::regex("^\\s+|\\s+$"), "");
            output = std::regex_replace(output, std::regex("^\\s+|\\s+$"), "");
            
            // Create unique key to detect duplicates
            std::string pair_key = input + "|" + output;
            if (seen_pairs.find(pair_key) != seen_pairs.end()) {
                continue;  // Skip duplicates
            }
            seen_pairs.insert(pair_key);
            
            // Categorize the pair
            std::string category;
            if (input.length() > 50) {
                category = "complex";
            } else if (input.find("is") != std::string::npos || 
                      input.find("looks") != std::string::npos || 
                      input.find("feels") != std::string::npos) {
                category = "adjective";
            } else if (input.find("to") != std::string::npos) {
                category = "verb";
            } else {
                category = "other";
            }
            
            category_pairs[category].push_back({input, output});
        }
    }
    
    // Print initial statistics
    std::cout << "\nInitial Training Data Statistics:" << std::endl;
    size_t total_pairs = 0;
    size_t max_category_size = 0;
    for (const auto& [category, pairs] : category_pairs) {
        std::cout << category << " pairs: " << pairs.size() << std::endl;
        total_pairs += pairs.size();
        max_category_size = std::max(max_category_size, pairs.size());
    }
    std::cout << "Total pairs before balancing: " << total_pairs << std::endl;

    // Instead of taking the minimum, we'll take up to 80% of the largest category size
    size_t target_size = static_cast<size_t>(max_category_size * 0.8);
    std::cout << "Target size per category: " << target_size << std::endl;

    // Random number generator for shuffling
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    
    // Sample from each category
    for (const auto& [category, pairs] : category_pairs) {
        std::vector<size_t> indices(pairs.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), gen);
        
        // Take up to target_size samples from each category
        size_t samples_to_take = std::min(target_size, pairs.size());
        for (size_t i = 0; i < samples_to_take; ++i) {
            training_pairs.push_back(pairs[indices[i]]);
        }
    }
    
    // Shuffle the final training pairs
    std::shuffle(training_pairs.begin(), training_pairs.end(), gen);
    
    std::cout << "\nFinal Training Data Statistics:" << std::endl;
    std::cout << "Total pairs after balancing: " << training_pairs.size() << std::endl;
    
    return training_pairs;
}

void Utils::analyze_token_mappings(
    const std::vector<std::pair<std::string, std::string>>& training_data,
    const TiktokenTokenizer& tokenizer) {
    std::cout << "\n=== Analyzing Token Mappings ===\n";
    size_t total_words = 0;
    size_t unknown_tokens = 0;
    std::unordered_map<std::string, int> unknown_words;

    for (const auto& pair : training_data) {
        std::string processed_input = pair.first;
        std::cout << "Full example: '" << pair.first << " | " << pair.second << "'" << std::endl;
        tokenizer.preprocess_text(processed_input);
        std::vector<int> tokens = tokenizer.encode(processed_input);
        std::cout << "encoded tokens" << tokens.data() << std::endl;
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
    std::unordered_map<std::string, std::vector<std::pair<std::string, std::string>>> category_pairs;
    std::unordered_set<std::string> seen_pairs;
    
    while (std::getline(file, line)) {
        // Normalize separators
        std::string normalized_line = line;
        std::replace(normalized_line.begin(), normalized_line.end(), '#', '|');
        std::replace(normalized_line.begin(), normalized_line.end(), '*', '|');
        
        size_t delimiter_pos = normalized_line.find('|');
        if (delimiter_pos != std::string::npos) {
            std::string input = normalized_line.substr(0, delimiter_pos);
            std::string output = normalized_line.substr(delimiter_pos + 1);
            
            // Trim whitespace
            input = std::regex_replace(input, std::regex("^\\s+|\\s+$"), "");
            output = std::regex_replace(output, std::regex("^\\s+|\\s+$"), "");
            
            // Create unique key to detect duplicates
            std::string pair_key = input + "|" + output;
            if (seen_pairs.find(pair_key) != seen_pairs.end()) {
                continue;  // Skip duplicates
            }
            seen_pairs.insert(pair_key);
            
            // Categorize the pair
            std::string category;
            if (input.length() > 50) {
                category = "complex";
            } else if (input.find("is") != std::string::npos || 
                      input.find("looks") != std::string::npos || 
                      input.find("feels") != std::string::npos) {
                category = "adjective";
            } else if (input.find("to") != std::string::npos) {
                category = "verb";
            } else {
                category = "other";
            }
            
            category_pairs[category].push_back({input, output});
        }
    }
    
    // Balance categories but keep more validation samples
    size_t min_category_size = std::numeric_limits<size_t>::max();
    for (const auto& [category, pairs] : category_pairs) {
        min_category_size = std::min(min_category_size, pairs.size());
    }
    
    // Use up to 20% of training size for validation
    min_category_size = std::min(min_category_size, static_cast<size_t>(min_category_size * 0.2));
    
    // Sample evenly from each category
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    for (const auto& [category, pairs] : category_pairs) {
        std::vector<size_t> indices(pairs.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), gen);
        
        // Take min_category_size samples from each category
        for (size_t i = 0; i < min_category_size; ++i) {
            validation_pairs.push_back(pairs[indices[i]]);
        }
    }
    
    std::cout << "\nValidation Data Statistics:" << std::endl;
    std::cout << "Total pairs after balancing: " << validation_pairs.size() << std::endl;
    for (const auto& [category, pairs] : category_pairs) {
        std::cout << category << " pairs: " << pairs.size() 
                  << " (used " << min_category_size << ")" << std::endl;
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
    const Matrix& logits,
    const TiktokenTokenizer& tokenizer,
    int beam_width) {
    
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

TokenCategories Utils::analyze_token_categories(const std::vector<std::pair<std::string, std::string>>& training_data) {
    TokenCategories categories;
    
    for (const auto& [input, target] : training_data) {
        size_t sep_pos;
        if ((sep_pos = target.find('#')) != std::string::npos) {
            // This is a verb ending
            std::string verb = target.substr(sep_pos + 1);
            Utils::trim(verb);
            categories.verb_tokens.insert(verb);
        } else if ((sep_pos = target.find('*')) != std::string::npos) {
            // This is an adjective ending
            std::string adj = target.substr(sep_pos + 1);
            Utils::trim(adj);
            categories.adjective_tokens.insert(adj);
        } else if ((sep_pos = target.find('|')) != std::string::npos) {
            // This is a noun ending
            std::string noun = target.substr(sep_pos + 1);
            Utils::trim(noun);
            categories.noun_tokens.insert(noun);
        }
    }
    
    std::cout << "\nToken Category Analysis:\n";
    std::cout << "Unique Verbs: " << categories.verb_tokens.size() << "\n";
    std::cout << "Unique Adjectives: " << categories.adjective_tokens.size() << "\n";
    std::cout << "Unique Nouns: " << categories.noun_tokens.size() << "\n";
    
    return categories;
}

// Function to determine the category of a token
std::string Utils::get_token_category(const std::string& token, const TokenCategories& categories) {
    if (categories.verb_tokens.find(token) != categories.verb_tokens.end()) {
        return "VERB";
    } else if (categories.adjective_tokens.find(token) != categories.adjective_tokens.end()) {
        return "ADJ";
    } else if (categories.noun_tokens.find(token) != categories.noun_tokens.end()) {
        return "NOUN";
    }
    return "UNKNOWN";
}

// Modify the print_top_predictions function to show token categories
void Utils::print_top_predictions(
    const Matrix& logits,
    const TiktokenTokenizer& tokenizer,
    Transformer& transformer,
    int k) {
    std::cout << "\nDebug: Entering print_top_predictions" << std::endl;
    const auto& config = transformer.getConfig();
    const auto& tp_config = config.token_prediction;

    // Get the last row of logits (predictions for the next token)
    std::vector<float> last_logits;
    for (size_t i = 0; i < logits.cols(); i++) {
        last_logits.push_back(logits(logits.rows() - 1, i));
    }
    std::cout << "Debug: Got " << last_logits.size() << " logits" << std::endl;

    // Apply temperature scaling
    float temperature = tp_config.temperature;
    std::cout << "Debug: Applying temperature scaling with T=" << temperature << std::endl;
    for (auto& logit : last_logits) {
        logit /= temperature;
    }

    // Apply softmax to get probabilities
    float max_logit = *std::max_element(last_logits.begin(), last_logits.end());
    std::cout << "Debug: Max logit value: " << max_logit << std::endl;
    std::vector<float> probabilities(last_logits.size());
    float sum_exp = 0.0f;
    for (size_t i = 0; i < last_logits.size(); i++) {
        probabilities[i] = std::exp(last_logits[i] - max_logit);
        sum_exp += probabilities[i];
    }
    std::cout << "Debug: Sum of exponentials: " << sum_exp << std::endl;
    for (float& prob : probabilities) {
        prob /= sum_exp;
    }

    // Create index vector for top-k selection
    std::vector<size_t> indices(probabilities.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Sort indices by probability
    std::partial_sort(indices.begin(), 
                     indices.begin() + std::min(k, static_cast<int>(indices.size())),
                     indices.end(),
                     [&probabilities](size_t a, size_t b) {
                         return probabilities[a] > probabilities[b];
                     });

    // Print top k predictions
    std::cout << "\nTop " << k << " predictions:" << std::endl;
    int printed = 0;
    for (int i = 0; i < k && i < static_cast<int>(indices.size()); i++) {
        size_t idx = indices[i];
        std::cout << "Debug: Probability for index " << idx << ": " << probabilities[idx] << std::endl;
        if (probabilities[idx] > 0.0f) {  // Only show non-zero probability tokens
            std::string token = tokenizer.decode({static_cast<int>(idx)});
            std::string category = "";
            if (tokenizer.is_verb(token)) category = " (VERB)";
            else if (tokenizer.is_adjective(token)) category = " (ADJ)";
            else if (tokenizer.is_noun(token)) category = " (NOUN)";
            
            std::cout << i + 1 << ". \"" << token << "\"" << category << " (p=" 
                      << std::fixed << std::setprecision(4) << probabilities[idx] << ")" << std::endl;
            printed++;
        }
    }
    std::cout << "Debug: Printed " << printed << " predictions" << std::endl;
}

float Utils::evaluate_validation(
    Transformer& transformer,
    const TiktokenTokenizer& tokenizer,
    const std::vector<std::pair<std::string, std::string>>& validation_data) {
    
    std::cout << "\n=== Starting evaluate_validation ===" << std::endl << std::flush;

    if (validation_data.empty()) {
        std::cout << "Warning: Empty validation data\n" << std::flush;
        return 0.0f;
    }

    std::cout << "Validation data size: " << validation_data.size() << std::endl << std::flush;

    float total_loss = 0.0f;
    size_t correct_predictions = 0;
    size_t total_predictions = 0;
    const size_t BATCH_SIZE = 32;

    std::cout << "Set training mode to false" << std::endl << std::flush;
    transformer.set_training(false);

    // Before the loop
    std::cout << "About to enter batch processing loop with:"
              << "\nbatch_start = 0"
              << "\nbatch_end = " << validation_data.size()
              << "\nvalidation_data.size() = " << validation_data.size() 
              << std::endl << std::flush;

    // Process data in batches
    for (size_t batch_start = 0; batch_start < validation_data.size(); batch_start += BATCH_SIZE) {
        size_t batch_end = std::min(batch_start + BATCH_SIZE, validation_data.size());
        size_t current_batch_size = batch_end - batch_start;
        
        float batch_loss = 0.0f;
        size_t batch_correct = 0;

        std::cout << "\nProcessing batch " << (batch_start/BATCH_SIZE + 1) 
                  << "/" << (validation_data.size() + BATCH_SIZE - 1)/BATCH_SIZE 
                  << " (size: " << current_batch_size << ")" << std::endl;

        // Process each example in the batch
        for (size_t i = batch_start; i < batch_end; i++) {
            std::cout << "Processing example " << i << std::endl;  // Add this
            try {
                const auto& pair = validation_data[i];
                std::cout << "Full example: '" << pair.first << " | " << pair.second << "'" << std::endl;
                std::cout << "Got validation pair" << std::endl;  // Add this
                std::string processed_input = pair.first;
                std::cout << "Got input: '" << processed_input << "'" << std::endl;  // Add this
                tokenizer.preprocess_text(processed_input);
                std::vector<int> input_tokens = tokenizer.encode(processed_input);
                std::cout << "Encoded input tokens, size: " << input_tokens.size() << std::endl;  // Add this
                
                // Get logits directly from transformer - don't project again through LM head
                Matrix logits = transformer.forward(input_tokens, processed_input, tokenizer);
                
                // Debug dimensions
                std::cout << "Dimensions:"
                          << "\n- Hidden states: " << logits.rows() << "x" << logits.cols()
                          << "\n- Logits: " << logits.rows() << "x" << logits.cols()
                          << "\n- Vocab size: " << tokenizer.vocab_size() << std::endl;
                
                // Get predicted token
                Vector last_logits = logits.row(logits.rows() - 1);
                
                // Debug logits
                float min_logit = *std::min_element(last_logits.data(), last_logits.data() + last_logits.size());
                float max_logit = *std::max_element(last_logits.data(), last_logits.data() + last_logits.size());
                float mean_logit = 0.0f;
                for (size_t j = 0; j < last_logits.size(); j++) {
                    mean_logit += last_logits[j];
                }
                mean_logit /= last_logits.size();

                std::cout << "Example " << i << " logits range: "
                          << "min=" << min_logit 
                          << ", max=" << max_logit
                          << ", mean=" << mean_logit << std::endl;
                
                int predicted_token = 0;
                float max_logit_value = -std::numeric_limits<float>::infinity();
                for (size_t j = 0; j < last_logits.size(); j++) {
                    if (last_logits[j] > max_logit_value) {
                        max_logit_value = last_logits[j];
                        predicted_token = j;
                    }
                }

                // Get target
                std::string processed_target = pair.second;
                tokenizer.preprocess_text(processed_target);
                std::vector<int> target_tokens = tokenizer.encode(processed_target);
                std::cout << "encoded target tokens: ";
                for (const auto& token : target_tokens) {
                    std::cout << token << " ";
                }
                std::cout << std::endl;
                if (target_tokens.empty()) {
                    std::cout << "Warning: Empty target tokens for example " << i << std::endl;
                    continue;
                }

                // Create target distribution
                Matrix target_distribution(1, tokenizer.vocab_size(), 0.0f);
                int target_token = target_tokens.back();
                target_distribution(0, target_token) = 1.0f;

                // Compute loss for this example
                Matrix last_token_logits(1, logits.cols());
                for (size_t j = 0; j < last_logits.size(); j++) {
                    last_token_logits(0, j) = last_logits[j];
                }

                float example_loss = compute_batch_loss(last_token_logits, target_distribution, tokenizer);
                
                // Debug loss calculation
                std::cout << "Example " << i << ":"
                          << "\n  Input: '" << pair.first << "'"
                          << "\n  Target: '" << pair.second << "'"
                          << "\n  Predicted token: " << predicted_token
                          << "\n  Target token: " << target_token
                          << "\n  Loss: " << example_loss << std::endl;

                if (!std::isfinite(example_loss)) {
                    std::cout << "Warning: Non-finite loss detected!" << std::endl;
                    continue;
                }

                batch_loss += example_loss;

                // Check prediction
                if (predicted_token == target_token) {
                    batch_correct++;
                }

            } catch (const std::exception& e) {
                std::cout << "Error processing example " << i << ": " << e.what() << std::endl;
                continue;
            }
        }

        // Compute and display batch metrics
        if (current_batch_size > 0) {  // Avoid division by zero
            float avg_batch_loss = batch_loss / current_batch_size;
            float batch_accuracy = static_cast<float>(batch_correct) / current_batch_size;

            std::cout << "Batch metrics:"
                      << "\n- Total batch loss: " << batch_loss
                      << "\n- Average batch loss: " << avg_batch_loss
                      << "\n- Accuracy: " << (batch_accuracy * 100.0f) << "%"
                      << "\n- Correct predictions: " << batch_correct << "/" << current_batch_size
                      << std::endl;

            // Update totals
            total_loss += batch_loss;  // Add batch total, not average
            correct_predictions += batch_correct;
            total_predictions += current_batch_size;
        }
    }

    // Print final evaluation metrics
    if (total_predictions > 0) {  // Avoid division by zero
        float avg_loss = total_loss / total_predictions;
        float accuracy = static_cast<float>(correct_predictions) / total_predictions;
        
        std::cout << "\nFinal Validation Results:"
                  << "\n- Total Loss: " << total_loss
                  << "\n- Total Predictions: " << total_predictions
                  << "\n- Average Loss: " << avg_loss
                  << "\n- Overall Accuracy: " << (accuracy * 100.0f) << "%"
                  << "\n- Total Correct Predictions: " << correct_predictions 
                  << "/" << total_predictions
                  << std::endl;

        transformer.set_training(true);
        return avg_loss;
    }

    std::cout << "Warning: No valid predictions made!" << std::endl;
    transformer.set_training(true);
    return 0.0f;
}

std::vector<std::string>& Utils::get_vocabulary(const TiktokenTokenizer& tokenizer) {
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

void from_json(const nlohmann::json& j, TokenizerConfig& t) {
    if (j.contains("use_subword")) {
        t.use_subword = j["use_subword"].get<bool>();
    }
    if (j.contains("vocab_size")) {
        t.vocab_size = j["vocab_size"].get<size_t>();
    }
    if (j.contains("model_path")) {
        t.model_path = j["model_path"].get<std::string>();
    }
    if (j.contains("special_tokens")) {
        t.special_tokens = j["special_tokens"].get<std::vector<std::string>>();
    }
}

void to_json(nlohmann::json& j, const TokenizerConfig& t) {
    j = nlohmann::json{
        {"use_subword", t.use_subword},
        {"vocab_size", t.vocab_size},
        {"model_path", t.model_path},
        {"special_tokens", t.special_tokens}
    };
}

void Utils::trim(std::string& s) {
    // Trim from start
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));

    // Trim from end
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

// Add new cross-validation function
std::vector<std::pair<std::vector<std::pair<std::string, std::string>>, 
                     std::vector<std::pair<std::string, std::string>>>> 
Utils::create_cross_validation_folds(const std::vector<std::pair<std::string, std::string>>& data, size_t num_folds) {
    std::vector<std::pair<std::vector<std::pair<std::string, std::string>>,
                         std::vector<std::pair<std::string, std::string>>>> folds;
    
    // Create a copy of the data that we can shuffle
    std::vector<std::pair<std::string, std::string>> shuffled_data = data;
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(shuffled_data.begin(), shuffled_data.end(), g);
    
    // Calculate fold size
    size_t fold_size = shuffled_data.size() / num_folds;
    
    // Create folds
    for (size_t i = 0; i < num_folds; i++) {
        std::vector<std::pair<std::string, std::string>> validation_fold;
        std::vector<std::pair<std::string, std::string>> training_fold;
        
        // Calculate start and end indices for current validation fold
        size_t start_idx = i * fold_size;
        size_t end_idx = (i == num_folds - 1) ? shuffled_data.size() : (i + 1) * fold_size;
        
        // Split data into training and validation
        for (size_t j = 0; j < shuffled_data.size(); j++) {
            if (j >= start_idx && j < end_idx) {
                validation_fold.push_back(shuffled_data[j]);
            } else {
                training_fold.push_back(shuffled_data[j]);
            }
        }
        
        folds.push_back({training_fold, validation_fold});
    }
    
    return folds;
}

float Utils::perform_cross_validation(
    Transformer& transformer,
    const TiktokenTokenizer& tokenizer,
    const std::vector<std::pair<std::string, std::string>>& train_data) {
    const auto& config = transformer.getConfig();
    const size_t num_folds = 2;  // Fixed to 2 folds for now
    const float early_stopping_threshold = config.training.cross_validation.early_stopping_threshold;

    std::cout << "\nPerforming " << num_folds << "-fold cross-validation..." << std::endl;
    
    auto folds = create_cross_validation_folds(train_data, num_folds);
    float total_loss = 0.0f;
    float best_val_loss = std::numeric_limits<float>::max();
    size_t early_stops = 0;
    const float learning_rate = config.initial_lr;
    
    // Track overall progress
    size_t total_operations = num_folds * config.num_epochs * (train_data.size() + train_data.size()/4);
    size_t completed_operations = 0;
    
    // Evaluate each fold
    for (size_t fold = 0; fold < folds.size(); fold++) {
        const auto& [train_fold, val_fold] = folds[fold];
        
        debug::log_progress("Cross-Validation", fold + 1, num_folds, 
                          "Processing fold " + std::to_string(fold + 1) + "/" + std::to_string(num_folds));
        
        // Train on this fold
        transformer.set_training(true);
        const size_t epochs_per_fold = config.num_epochs;
        
        for (size_t epoch = 0; epoch < epochs_per_fold; epoch++) {
            debug::log_progress("Epoch", epoch + 1, epochs_per_fold,
                              "Fold " + std::to_string(fold + 1) + "/" + std::to_string(num_folds));
            
            size_t processed_samples = 0;
            const size_t batch_size = config.batch_size;
            
            // Training phase
            for (size_t i = 0; i < train_fold.size(); i += batch_size) {
                size_t current_batch_size = std::min(batch_size, train_fold.size() - i);
                processed_samples += current_batch_size;
                completed_operations += current_batch_size;
                
                // Update both batch progress and overall progress
                debug::log_progress("Batch", processed_samples, train_fold.size(),
                                  "Training | Epoch " + std::to_string(epoch + 1) + "/" + 
                                  std::to_string(epochs_per_fold) + " | Fold " + 
                                  std::to_string(fold + 1) + "/" + std::to_string(num_folds));
                
                // Sleep for a tiny amount to allow file updates to be visible
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            
            // Validation phase
            transformer.set_training(false);
            size_t val_processed = 0;
            float val_loss = 0.0f;
            
            for (size_t i = 0; i < val_fold.size(); i += batch_size) {
                size_t current_batch_size = std::min(batch_size, val_fold.size() - i);
                val_processed += current_batch_size;
                completed_operations += current_batch_size;
                
                // Compute validation loss for this batch
                std::vector<std::pair<std::string, std::string>> batch_data(
                    val_fold.begin() + i,
                    val_fold.begin() + i + current_batch_size
                );
                float batch_loss = evaluate_validation(transformer, tokenizer, batch_data);
                val_loss += batch_loss;
                
                debug::log_progress("Validation", val_processed, val_fold.size(),
                                  "Epoch " + std::to_string(epoch + 1) + "/" + 
                                  std::to_string(epochs_per_fold) + " | Fold " + 
                                  std::to_string(fold + 1) + "/" + std::to_string(num_folds) +
                                  " | Current Loss: " + std::to_string(batch_loss));
                
                // Sleep for a tiny amount to allow file updates to be visible
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            
            val_loss /= (val_fold.size() / batch_size);  // Average the validation loss
            
            // Early stopping checks
            if (val_loss < best_val_loss) {
                best_val_loss = val_loss;
            } else if (epoch > 0 && val_loss > best_val_loss * early_stopping_threshold) {
                debug::log_progress("Early Stopping", epoch + 1, epochs_per_fold,
                                  "Triggered in Fold " + std::to_string(fold + 1) + 
                                  " | Loss increased from " + std::to_string(best_val_loss) +
                                  " to " + std::to_string(val_loss));
                early_stops++;
                break;
            }
        }
        
        // Final evaluation for this fold
        transformer.set_training(false);
        float fold_loss = evaluate_validation(transformer, tokenizer, val_fold);
        total_loss += fold_loss;
        
        debug::log_progress("Fold Complete", fold + 1, num_folds,
                          "Loss: " + std::to_string(fold_loss));
    }
    
    float avg_loss = total_loss / num_folds;
    debug::log_progress("Cross-Validation Complete", num_folds, num_folds,
                       "Average Loss: " + std::to_string(avg_loss) + 
                       " | Early Stops: " + std::to_string(early_stops) + "/" + std::to_string(num_folds));
    
    return avg_loss;
}

void Utils::generate_predictions(
    Transformer& transformer,
    const std::string& input_text,
    TiktokenTokenizer* tokenizer) {
    std::cout << "\n=== Generating predictions for: '" << input_text << "' ===" << std::endl;
    
    // Preprocess input
    std::string processed_input = input_text;
    tokenizer->preprocess_text(processed_input);
    std::vector<int> input_tokens = tokenizer->encode(processed_input);
    
    // Get model prediction
    transformer.set_training(false);  // Set to evaluation mode
    Matrix hidden_states = transformer.forward(input_tokens, processed_input, *tokenizer);
    Matrix logits = transformer.get_lm_head()->forward(hidden_states);
    
    // Get probabilities for last token
    Vector last_logits = logits.row(logits.rows() - 1);
    std::vector<std::pair<float, int>> token_probs;
    
    // Convert logits to probabilities using softmax
    float max_logit = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < last_logits.size(); i++) {
        max_logit = std::max(max_logit, last_logits[i]);
    }
    
    float sum_exp = 0.0f;
    for (size_t i = 0; i < last_logits.size(); i++) {
        float prob = std::exp(last_logits[i] - max_logit);
        sum_exp += prob;
        token_probs.push_back({prob, static_cast<int>(i)});
    }
    
    // Normalize probabilities
    for (auto& pair : token_probs) {
        pair.first /= sum_exp;
    }
    
    // Sort by probability
    std::sort(token_probs.begin(), token_probs.end(),
              std::greater<std::pair<float, int>>());
    
    // Print top 5 predictions
    std::cout << "Top 5 predictions:" << std::endl;
    for (int i = 0; i < std::min(5, static_cast<int>(token_probs.size())); i++) {
        float prob = token_probs[i].first;
        int token_id = token_probs[i].second;
        std::string token = tokenizer->decode({token_id});
        
        // Add token type annotation
        std::string token_type = "";
        if (tokenizer->is_verb(token)) token_type = " (VERB)";
        else if (tokenizer->is_adjective(token)) token_type = " (ADJ)";
        else if (tokenizer->is_noun(token)) token_type = " (NOUN)";
        
        std::cout << i + 1 << ". \"" << token << "\"" << token_type 
                  << " (p=" << std::fixed << std::setprecision(4) << prob * 100 << "%)" << std::endl;
    }
    
    std::cout << "===" << std::endl;
    transformer.set_training(true);  // Reset to training mode
}

// Add debugging for gradient analysis
void Utils::analyze_gradients(const Matrix& gradients, const std::string& label) {
    float mean = 0.0f;
    float max_abs = 0.0f;
    float min_abs = std::numeric_limits<float>::max();
    int zero_count = 0;
    int total_elements = gradients.rows() * gradients.cols();
    
    for (size_t i = 0; i < gradients.rows(); i++) {
        for (size_t j = 0; j < gradients.cols(); j++) {
            float val = std::abs(gradients(i,j));
            mean += val;
            max_abs = std::max(max_abs, val);
            if (val > 0) min_abs = std::min(min_abs, val);
            if (val == 0) zero_count++;
        }
    }
    mean /= total_elements;
    
    std::ostringstream oss;
    oss << "Gradient analysis for " << label << ":\n"
        << "  Mean absolute value: " << mean << "\n"
        << "  Max absolute value: " << max_abs << "\n"
        << "  Min non-zero absolute value: " << min_abs << "\n"
        << "  Zero elements: " << zero_count << "/" << total_elements 
        << " (" << (100.0f * zero_count / total_elements) << "%)";
    
    debug::log_message(oss.str(), "DEBUG");
    
    // Check for potential issues
    if (mean < 1e-7) {
        debug::log_message("WARNING: Very small gradient mean detected", "WARN");
    }
    if (zero_count > total_elements * 0.9) {
        debug::log_message("WARNING: High proportion of zero gradients detected", "WARN");
    }
    if (max_abs > 100) {
        debug::log_message("WARNING: Large gradient values detected", "WARN");
    }
}

// Add debugging for token processing
void Utils::debug_token_processing(
    const std::string& input,
    const std::vector<int>& tokens,
    const TiktokenTokenizer& tokenizer) {
    std::ostringstream oss;
    oss << "Token processing debug for input: '" << input << "'\n"
        << "  Input length: " << input.length() << " characters\n"
        << "  Token count: " << tokens.size() << "\n"
        << "  Tokens: ";
    
    for (size_t i = 0; i < std::min(tokens.size(), size_t(10)); i++) {
        std::string token = tokenizer.decode({tokens[i]});
        oss << "'" << token << "'(" << tokens[i] << ") ";
    }
    if (tokens.size() > 10) oss << "...";
    
    // Check token coverage
    std::string reconstructed = tokenizer.decode(tokens);
    bool perfect_reconstruction = (reconstructed == input);
    
    oss << "\n  Perfect reconstruction: " << (perfect_reconstruction ? "Yes" : "No");
    if (!perfect_reconstruction) {
        oss << "\n  Reconstructed text: '" << reconstructed << "'";
    }
    
    debug::log_message(oss.str(), "DEBUG");
    
    // Check for potential issues
    if (tokens.size() > input.length() * 2) {
        debug::log_message("WARNING: Unusually high token/character ratio", "WARN");
    }
    if (!perfect_reconstruction) {
        debug::log_message("WARNING: Token reconstruction does not match input", "WARN");
    }
}

// Add debugging for loss analysis
void Utils::analyze_loss_progression(const std::vector<float>& losses, size_t window_size) {
    if (losses.size() < window_size * 2) return;
    
    // Calculate moving averages
    std::vector<float> moving_avgs;
    float sum = 0;
    for (size_t i = 0; i < losses.size(); i++) {
        sum += losses[i];
        if (i >= window_size) sum -= losses[i - window_size];
        if (i >= window_size - 1) {
            moving_avgs.push_back(sum / window_size);
        }
    }
    
    // Analyze trend
    size_t flat_count = 0;
    size_t increasing_count = 0;
    const float threshold = 0.001f;
    
    for (size_t i = 1; i < moving_avgs.size(); i++) {
        float diff = moving_avgs[i] - moving_avgs[i-1];
        if (std::abs(diff) < threshold) flat_count++;
        if (diff > threshold) increasing_count++;
    }
    
    std::ostringstream oss;
    oss << "Loss progression analysis:\n"
        << "  Initial moving average: " << moving_avgs.front() << "\n"
        << "  Final moving average: " << moving_avgs.back() << "\n"
        << "  Flat regions: " << (100.0f * flat_count / moving_avgs.size()) << "%\n"
        << "  Increasing regions: " << (100.0f * increasing_count / moving_avgs.size()) << "%";
    
    debug::log_message(oss.str(), "INFO");
    
    // Check for potential issues
    if (flat_count > moving_avgs.size() * 0.5) {
        debug::log_message("WARNING: Loss appears to be stagnating", "WARN");
    }
    if (increasing_count > moving_avgs.size() * 0.3) {
        debug::log_message("WARNING: Loss shows significant increasing trend", "WARN");
    }
}