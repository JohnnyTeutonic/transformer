#include "../include/utils.hpp"
#include "../include/beam_search.hpp"
#include "../include/scope_logger.hpp"
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
    bool verbose_logging = false;
    bool scope_logging_enabled = false;
    std::ofstream debug_log;
    const std::string log_file = "transformer.log";
    const std::string progress_file = "progress.log";
    std::mutex log_mutex;
    std::ofstream progress_log;
    
    // Define the global progress_state variable
    ProgressState progress_state;

    void enable_scope_logging(bool enable) {
        scope_logging_enabled = enable;
    }

    void init_logging() {
        std::lock_guard<std::mutex> lock(log_mutex);
        debug_log.open(log_file, std::ios::trunc);
        progress_log.open(progress_file, std::ios::trunc);
        if (scope_logging_enabled) {
            ScopeLogger::init();  // Only initialize scope logger if enabled
        }
    }

    // Implement ProgressState methods
    void ProgressState::update_tuning(size_t trial, size_t total, const std::string& config, float loss) {
        // Only update tuning progress if we're actually tuning
        if (current_stage != Stage::TUNING) {
            return;  // Skip if we're not in tuning phase
        }
        
        current_stage = Stage::TUNING;
        tuning.current_trial = trial;
        tuning.total_trials = total;
        tuning.current_config = config;
        if (loss < tuning.best_loss) {
            tuning.best_loss = loss;
        }
        update_progress_file();
    }

    void ProgressState::update_training(size_t epoch, size_t total_epochs, size_t batch, size_t total_batches, float loss) {
        current_stage = Stage::TRAINING;
        training.current_epoch = epoch;
        training.total_epochs = total_epochs;
        training.current_batch = batch;
        training.total_batches = total_batches;
        training.current_loss = loss;
        if (loss < training.best_loss) {
            training.best_loss = loss;
        }
        update_progress_file();
    }

    void ProgressState::update_cross_validation(size_t fold, size_t total_folds, size_t epoch, size_t total_epochs, float loss) {
        current_stage = Stage::CROSS_VALIDATION;
        training.is_cross_validation = true;
        training.current_fold = fold;
        training.total_folds = total_folds;
        training.current_epoch = epoch;
        training.total_epochs = total_epochs;
        training.current_loss = loss;
        if (loss < training.best_loss) {
            training.best_loss = loss;
        }
        update_progress_file();
    }

    void ProgressState::update_inference(size_t tokens, size_t total, float avg_time) {
        current_stage = Stage::INFERENCE;
        inference.tokens_generated = tokens;
        inference.total_tokens = total;
        inference.average_time_per_token = avg_time;
        update_progress_file();
    }

    void ProgressState::reset() {
        // Store current stage
        Stage previous_stage = current_stage;
        
        // Reset tuning state
        tuning.current_trial = 0;
        tuning.total_trials = 0;
        tuning.current_config.clear();
        tuning.best_loss = std::numeric_limits<float>::max();
        
        // Reset training state
        training.current_epoch = 0;
        training.total_epochs = 0;
        training.current_batch = 0;
        training.total_batches = 0;
        training.current_fold = 0;
        training.total_folds = 0;
        training.current_loss = 0.0f;
        training.best_loss = std::numeric_limits<float>::max();
        training.is_cross_validation = false;
        
        // Reset inference state
        inference.tokens_generated = 0;
        inference.total_tokens = 0;
        inference.average_time_per_token = 0.0f;
        
        // Restore previous stage instead of defaulting to IDLE
        current_stage = previous_stage;
        
        // Update the progress file
        update_progress_file();
    }

    std::string ProgressState::get_stage_string() const {
        switch (current_stage) {
            case Stage::TUNING: return "Hyperparameter Tuning";
            case Stage::TRAINING: return "Training";
            case Stage::CROSS_VALIDATION: return "Cross Validation";
            case Stage::INFERENCE: return "Inference";
            default: return "Idle";
        }
    }

    void ProgressState::update_progress_file() {
        std::lock_guard<std::mutex> lock(log_mutex);
        if (!progress_log.is_open()) {
            progress_log.open(progress_file, std::ios::trunc);
        }
        
        progress_log.seekp(0);
        progress_log << "=== Training Session Progress ===\n\n";
        
        // Write current state with proper formatting
        progress_log << "Current Stage: " << get_stage_string() << "\n\n";
        
        switch (current_stage) {
            case Stage::TUNING:
                write_tuning_progress();
                break;
            case Stage::TRAINING:
            case Stage::CROSS_VALIDATION:
                write_training_progress();
                break;
            case Stage::INFERENCE:
                write_inference_progress();
                break;
            default:
                progress_log << "System idle\n";
        }
        
        progress_log.flush();
    }

    void ProgressState::write_tuning_progress() {
        progress_log << "Hyperparameter Tuning Progress:\n"
                    << "Trial: " << tuning.current_trial + 1 << "/" << tuning.total_trials << "\n"
                    << "Current Configuration:\n" << tuning.current_config << "\n"
                    << "Best Loss: " << tuning.best_loss << "\n\n";
    }

    void ProgressState::write_training_progress() {
        if (training.is_cross_validation) {
            progress_log << "Cross Validation Progress:\n"
                        << "Fold: " << training.current_fold + 1 << "/" << training.total_folds << "\n";
        }
        progress_log << "Training Progress:\n"
                    << "Epoch: " << training.current_epoch + 1 << "/" << training.total_epochs << "\n"
                    << "Batch: " << training.current_batch + 1 << "/" << training.total_batches << "\n"
                    << "Current Loss: " << training.current_loss << "\n"
                    << "Best Loss: " << training.best_loss << "\n\n";
    }

    void ProgressState::write_inference_progress() {
        progress_log << "Inference Progress:\n"
                    << "Tokens Generated: " << inference.tokens_generated << "/" << inference.total_tokens << "\n"
                    << "Average Time per Token: " << inference.average_time_per_token << "ms\n\n";
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
                     const std::string& additional_info) {
        std::lock_guard<std::mutex> lock(log_mutex);
        if (!progress_log.is_open()) {
            progress_log.open(progress_file, std::ios::app);
        }
        
        float progress = static_cast<float>(current) / total * 100.0f;
        progress_log << stage << ": " << std::fixed << std::setprecision(1) 
                    << progress << "% (" << current << "/" << total << ")";
        
        if (!additional_info.empty()) {
            progress_log << " - " << additional_info;
        }
        progress_log << std::endl;
        progress_log.flush();
        
        // Update the main progress file using the class method
        progress_state.update_progress_file();
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
    size_t sequence_length) {
    
    // Create target distribution with correct dimensions
    size_t actual_batch_size = target_tokens.size();
    
    // Debug output
    std::cout << "Creating target distribution:"
              << "\n- Actual batch size: " << actual_batch_size
              << "\n- Vocab size: " << vocab_size << std::endl;
    
    // For single-word prediction, we only need one row per sequence
    Matrix target_distribution(actual_batch_size, vocab_size, 0.0f);
    
    // Fill target distribution only for the final token of each sequence
    for (size_t i = 0; i < actual_batch_size; i++) {
        const auto& sequence = target_tokens[i];
        if (!sequence.empty()) {
            // Get the last token as our target
            int target_token = sequence.back();
            if (target_token >= 0 && static_cast<size_t>(target_token) < vocab_size) {
                target_distribution(i, target_token) = 1.0f;
            }
        }
    }
    
    std::cout << "Created target distribution with shape: " << target_distribution.shape() << std::endl;
    return target_distribution;
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

    // Validate dimensions
    const size_t logits_batch_size = logits.rows();
    const size_t target_batch_size = target_distribution.rows();
    const size_t batch_size = std::min(logits_batch_size, target_batch_size);
    const size_t vocab_size = logits.cols();

    if (logits.cols() != target_distribution.cols()) {
        throw std::runtime_error("Logits and target distribution must have same number of columns. " 
                               "Got " + std::to_string(logits.cols()) + " and " 
                               + std::to_string(target_distribution.cols()));
    }

    // Constants for numerical stability
    const float epsilon = 1e-5f;  // Increased from 1e-7f
    const float max_loss_per_token = 100.0f;
    const float min_log_prob = -50.0f;  // Prevent excessively small log probabilities

    float total_loss = 0.0f;

    // Process each item in the batch
    #pragma omp parallel for reduction(+:total_loss)
    for (size_t i = 0; i < batch_size; ++i) {
        // Find max logit for numerical stability
        float max_logit = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < vocab_size; ++j) {
            max_logit = std::max(max_logit, logits(i, j));
        }

        // Compute log-sum-exp trick for softmax denominator
        float sum_exp = 0.0f;
        for (size_t j = 0; j < vocab_size; ++j) {
            float shifted_logit = logits(i, j) - max_logit;
            // Clamp extremely negative values
            if (shifted_logit > min_log_prob) {
            sum_exp += std::exp(shifted_logit);
        }
        }
        
        // Ensure denominator is valid
        sum_exp = std::max(sum_exp, epsilon);
        float log_denominator = std::log(sum_exp) + max_logit;

        // Compute cross-entropy loss for this item
        float sequence_loss = 0.0f;
        for (size_t j = 0; j < vocab_size; ++j) {
            if (target_distribution(i, j) > 0.0f) {
                // Compute log probability with numerical stability
                float log_prob = logits(i, j) - log_denominator;
                // Clamp log probability to prevent extreme values
                log_prob = std::clamp(log_prob, min_log_prob, 0.0f);
                
                // Compute weighted cross-entropy term
                float term = -target_distribution(i, j) * log_prob;
                // Clamp individual loss terms
                term = std::min(term, max_loss_per_token);
                sequence_loss += term;
            }
        }

        // Add sequence loss to total
        if (std::isfinite(sequence_loss)) {
            total_loss += sequence_loss;
        } else {
            std::cout << "Warning: Non-finite sequence loss detected at position " << i << std::endl;
            total_loss += max_loss_per_token;  // Use max loss as fallback
        }
    }

    // Compute average loss with floor
    float avg_loss = total_loss / static_cast<float>(batch_size);
    avg_loss = std::max(avg_loss, 1e-4f);  // Minimum loss floor

    // Debug output
    if (avg_loss > 10.0f || avg_loss < 1e-3f) {
        std::cout << "Unusual loss value detected: " << avg_loss << std::endl;
        std::cout << "Batch statistics:"
              << "\n- Total loss: " << total_loss
                  << "\n- Batch size: " << batch_size
                  << "\n- Vocab size: " << vocab_size << std::endl;
    }

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
        config.tokenizer.model_path = tok.value("model_path", "model/tokenizer.model");
    }

    // Parse model settings
    if (j.contains("model")) {
        auto& model = j["model"];
        // Load other model settings
        config.hidden_size = model["hidden_size"];
        config.num_heads = model["num_heads"];
        config.num_layers = model["num_layers"];
        config.head_dim = model["head_dim"];
        config.intermediate_size = model["intermediate_size"];
    }

    // Load training parameters
    if (j.contains("training")) {
        const auto& training = j["training"];
        config.training.samples_per_iteration = training.value("samples_per_iteration", 32);
        config.training.num_epochs = training.value("num_epochs", config.training.num_epochs);
        config.training.dropout_rate = training.value("dropout_rate", config.training.dropout_rate);
        config.training.weight_decay = training.value("weight_decay", config.training.weight_decay);
        
        if (training.contains("cross_validation")) {
            auto& cv = training["cross_validation"];
            config.training.cross_validation.num_folds = cv.value("num_folds", 2);
            config.training.cross_validation.validation_frequency = cv.value("validation_frequency", 1);
            config.training.cross_validation.early_stopping_threshold = cv.value("early_stopping_threshold", 1.5f);
            config.training.cross_validation.early_stopping_patience = cv.value("early_stopping_patience", 2);
            config.training.cross_validation.num_epochs = cv.value("num_epochs", 10);
        }
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

PhraseType Utils::detect_phrase_type(const std::string& input, const std::string& output, char delimiter) {
    // Detect phrase type based on input and output patterns
    if (input.find("is") != std::string::npos || 
        input.find("looks") != std::string::npos || 
        input.find("feels") != std::string::npos ||
        delimiter == '*') {
        return PhraseType::ADJECTIVE;
    } else if (input.find("to") != std::string::npos || delimiter == '#') {
        return PhraseType::VERB;
    }
    return PhraseType::GENERAL;  // Use GENERAL instead of NOUN/OTHER
}

float Utils::get_sampling_weight(const std::string& category, 
                               const std::unordered_map<std::string, size_t>& counts) {
    float base_weight = 1.0f;
    size_t count = counts.at(category);
    size_t max_count = 0;
    for (const auto& [_, c] : counts) {
        max_count = std::max(max_count, c);
    }
    return base_weight * (static_cast<float>(max_count) / count);
}

std::vector<TrainingExample> Utils::balance_dataset(
    const std::unordered_map<PhraseType, std::vector<TrainingExample>>& categorized_data) {
    
    // Find target size (95% of largest category)
    size_t max_size = 0;
    for (const auto& [_, examples] : categorized_data) {
        max_size = std::max(max_size, examples.size());
    }
    size_t target_size = static_cast<size_t>(max_size * 0.95);

    // Prepare balanced dataset
    std::vector<TrainingExample> balanced_data;
    std::mt19937 gen(42);  // Fixed seed for reproducibility

    for (const auto& [type, examples] : categorized_data) {
        std::vector<size_t> indices(examples.size());
        std::iota(indices.begin(), indices.end(), 0);
        
        // If category is smaller than target, oversample
        if (examples.size() < target_size) {
            size_t repeats = (target_size + examples.size() - 1) / examples.size();
            std::vector<size_t> expanded_indices;
            for (size_t r = 0; r < repeats; r++) {
                expanded_indices.insert(expanded_indices.end(), indices.begin(), indices.end());
            }
            indices = expanded_indices;
        }
        
        // Shuffle indices
        std::shuffle(indices.begin(), indices.end(), gen);
        
        // Take exactly target_size samples
        for (size_t i = 0; i < target_size; ++i) {
            balanced_data.push_back(examples[indices[i % indices.size()]]);
        }
    }

    // Final shuffle of all balanced data
    std::shuffle(balanced_data.begin(), balanced_data.end(), gen);
    return balanced_data;
}

std::vector<ContextualTrainingExample> Utils::create_training_data() {
    std::filesystem::path file_path = "../data/training_pairs.txt";
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open training data file: " + file_path.string());
    }

    // First pass: collect all examples
    std::vector<TrainingExample> raw_examples;
    size_t total_lines = 0;
    std::unordered_map<char, size_t> separator_counts;  // Track usage of different separators
    
    std::cout << "\nDebug: Starting to load training data..." << std::endl;
    
    std::string line;
    while (std::getline(file, line)) {
        total_lines++;
        if (line.empty()) continue;

        // Find the separator while preserving its type
        size_t separator_pos = std::string::npos;
        char separator = '\0';
        
        // Check for each type of separator while preserving its meaning
        if ((separator_pos = line.find('|')) != std::string::npos) {
            separator = '|';  // General phrase/noun separator
        } else if ((separator_pos = line.find('#')) != std::string::npos) {
            separator = '#';  // Verb phrase separator
        } else if ((separator_pos = line.find('*')) != std::string::npos) {
            separator = '*';  // Adjective phrase separator
        }

        if (separator_pos != std::string::npos) {
            std::string input = line.substr(0, separator_pos);
            std::string output = line.substr(separator_pos + 1);
            
            // Trim whitespace
            trim(input);
            trim(output);
            
            // Detect phrase type based on the separator
            PhraseType type;
            switch (separator) {
                case '#':
                    type = PhraseType::VERB;
                    break;
                case '*':
                    type = PhraseType::ADJECTIVE;
                    break;
                case '|':
                default:
                    type = PhraseType::GENERAL;
                    break;
            }
            
            separator_counts[separator]++;
            raw_examples.emplace_back(input, output, type);
        }
    }

    std::cout << "\nDebug: Training data loading statistics:" << std::endl;
    std::cout << "Total lines read: " << total_lines << std::endl;
    std::cout << "Total examples: " << raw_examples.size() << std::endl;
    std::cout << "Separator distribution:" << std::endl;
    std::cout << "  General phrases (|): " << separator_counts['|'] << std::endl;
    std::cout << "  Verb phrases (#): " << separator_counts['#'] << std::endl;
    std::cout << "  Adjective phrases (*): " << separator_counts['*'] << std::endl;

    // Second pass: create contextual examples
    std::vector<ContextualTrainingExample> contextual_examples;
    const size_t CONTEXT_SIZE = 3;  // Keep context window smaller for stability
    
    // Create sliding windows of examples
    for (size_t i = 0; i < raw_examples.size(); i++) {
        // Collect context from previous examples
        std::vector<std::string> context;
        for (size_t j = 1; j <= CONTEXT_SIZE && i >= j; j++) {
            context.insert(context.begin(), raw_examples[i - j].input);
        }
        
        // Create contextual example
        contextual_examples.emplace_back(
            context,
            raw_examples[i].input,
            raw_examples[i].output,
            raw_examples[i].type,
            CONTEXT_SIZE
        );
    }

    // Print statistics
    std::cout << "\nTraining Data Statistics:" << std::endl;
    std::cout << "Original examples: " << raw_examples.size() << std::endl;
    std::cout << "Contextual examples: " << contextual_examples.size() << std::endl;
    std::cout << "Context window size: " << CONTEXT_SIZE << std::endl;
    
    // Shuffle the examples while maintaining separator distribution
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(contextual_examples.begin(), contextual_examples.end(), g);
    
    return contextual_examples;
}

std::vector<ContextualTrainingExample> Utils::load_validation_data() {
    std::filesystem::path exe_path = std::filesystem::current_path().parent_path();
    std::filesystem::path data_dir = exe_path / "data";
    std::filesystem::path file_path = data_dir / "validation_pairs.txt";

    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open validation data file: " + file_path.string());
    }

    std::unordered_map<PhraseType, std::vector<ContextualTrainingExample>> categorized_data;
    std::unordered_set<std::string> seen_pairs;
    std::vector<std::string> context_buffer;  // Keep track of recent examples for context
    const size_t CONTEXT_SIZE = 3;
    
    std::string line;
    while (std::getline(file, line)) {
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
                continue;
            }
            seen_pairs.insert(pair_key);
            
            // Detect phrase type and categorize
            char delimiter = line[delimiter_pos];
            PhraseType type = detect_phrase_type(input, output, delimiter);
            
            // Create contextual example
            std::vector<std::string> current_context(context_buffer.end() - std::min(context_buffer.size(), CONTEXT_SIZE), 
                                                   context_buffer.end());
            
            ContextualTrainingExample example(current_context, input, output, type, CONTEXT_SIZE);
            categorized_data[type].push_back(example);
            
            // Update context buffer
            context_buffer.push_back(input + " " + output);
            if (context_buffer.size() > CONTEXT_SIZE * 2) {  // Keep buffer size reasonable
                context_buffer.erase(context_buffer.begin());
            }
        }
    }
    
    // For validation, we want a smaller but still balanced dataset
    // Use 20% of the size of the smallest category
    size_t min_category_size = std::numeric_limits<size_t>::max();
    for (const auto& [_, examples] : categorized_data) {
        min_category_size = std::min(min_category_size, examples.size());
    }
    size_t target_size = static_cast<size_t>(min_category_size * 0.2);
    
    std::vector<ContextualTrainingExample> validation_data;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    
    for (const auto& [type, examples] : categorized_data) {
        std::vector<size_t> indices(examples.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), gen);
        
        // Take target_size samples from each category
        for (size_t i = 0; i < target_size; ++i) {
            validation_data.push_back(examples[indices[i]]);
        }
    }
    
    std::cout << "\nValidation Data Statistics:" << std::endl;
    std::cout << "Total validation examples: " << validation_data.size() << std::endl;
    
    return validation_data;
}

ValidationMetrics Utils::evaluate_validation(
    Transformer& transformer,
    const TiktokenTokenizer& tokenizer,
    const std::vector<ContextualTrainingExample>& validation_data) {
    
    transformer.set_training(false);
    ValidationMetrics metrics{0.0f, 0.0f};
    std::unordered_map<PhraseType, float> type_losses;
    std::unordered_map<PhraseType, float> type_correct;
    std::unordered_map<PhraseType, size_t> type_counts;
    
    const size_t BATCH_SIZE = 32;
    size_t total_correct = 0;
    size_t total_samples = 0;

    // Process data in batches
    for (size_t batch_start = 0; batch_start < validation_data.size(); batch_start += BATCH_SIZE) {
        size_t batch_end = std::min(batch_start + BATCH_SIZE, validation_data.size());
        size_t current_batch_size = batch_end - batch_start;
        
        std::vector<std::vector<int>> context_batch;
        std::vector<std::vector<int>> target_batch;
        std::vector<PhraseType> batch_types;
        std::vector<std::string> full_contexts;

        // Prepare batch
        for (size_t i = batch_start; i < batch_end; i++) {
            const auto& example = validation_data[i];
            std::string full_context = example.get_full_context();
            std::string processed_context = full_context;
            std::string processed_target = example.output;
            
            tokenizer.preprocess_text(processed_context);
            tokenizer.preprocess_text(processed_target);
            
            std::vector<int> context_tokens = tokenizer.encode(processed_context);
            std::vector<int> target_tokens = tokenizer.encode(processed_target);
            
            if (context_tokens.empty() || target_tokens.empty()) continue;
            
            context_batch.push_back(context_tokens);
            target_batch.push_back(target_tokens);
            batch_types.push_back(example.type);
            full_contexts.push_back(full_context);
        }

        if (context_batch.empty()) continue;

        // Forward pass for each sequence with context
        std::vector<Matrix> all_logits;
        for (size_t i = 0; i < context_batch.size(); i++) {
            Matrix seq_logits = transformer.forward(context_batch[i], full_contexts[i], tokenizer);
            Matrix last_logits(1, seq_logits.cols());
            for (size_t j = 0; j < seq_logits.cols(); j++) {
                last_logits(0, j) = seq_logits(seq_logits.rows() - 1, j);
            }
            all_logits.push_back(last_logits);
        }
        
        // Combine logits and compute metrics
        Matrix combined_logits(context_batch.size(), tokenizer.vocab_size());
        for (size_t i = 0; i < all_logits.size(); i++) {
            for (size_t j = 0; j < all_logits[i].cols(); j++) {
                combined_logits(i, j) = all_logits[i](0, j);
            }
        }
        
        // Create target distribution
        Matrix target_distribution = create_batch_target_distribution(
            target_batch, tokenizer, tokenizer.vocab_size(), context_batch.size());
        
        // Compute loss and accuracy
        float batch_loss = compute_batch_loss(combined_logits, target_distribution, tokenizer);
        
        // Update metrics for each example
        for (size_t i = 0; i < current_batch_size; i++) {
            PhraseType type = batch_types[i];
            type_counts[type]++;
            
            // Get predicted token
            size_t predicted_token = 0;
            float max_prob = -std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < combined_logits.cols(); j++) {
                if (combined_logits(i, j) > max_prob) {
                    max_prob = combined_logits(i, j);
                    predicted_token = j;
                }
            }
            
            // Check if prediction matches target
            bool is_correct = false;
            for (size_t j = 0; j < combined_logits.cols(); j++) {
                if (target_distribution(i, j) > 0.0f && j == predicted_token) {
                    is_correct = true;
                    total_correct++;
                    type_correct[type]++;
                    break;
                }
            }
            
            type_losses[type] += batch_loss;
            total_samples++;
        }
        
        metrics.loss += batch_loss * current_batch_size;
    }

    // Compute final metrics
    metrics.loss /= total_samples;
    metrics.accuracy = static_cast<float>(total_correct) / total_samples;
    
    // Compute type-specific metrics
    for (const auto& [type, count] : type_counts) {
        metrics.type_specific_loss[type] = type_losses[type] / count;
        metrics.type_specific_accuracy[type] = type_correct[type] / count;
    }

    transformer.set_training(true);
    return metrics;
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

// Modify the print_top_predictions function to filter out delimiter tokens
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

    // Apply dynamic temperature
    std::string current_context = transformer.get_current_context();
    PhraseType current_type = transformer.predict_phrase_type(current_context, tokenizer);
    
    // Create a proper random generator that lives for the duration of the function
    std::random_device rd;
    std::mt19937 gen(rd());
    float temperature = transformer.get_dynamic_temperature(current_type, std::as_const(gen));
    
    std::cout << "Using temperature: " << temperature << std::endl;
    
    for (auto& logit : last_logits) {
        logit /= temperature;
    }

    // Apply softmax to get probabilities
    float max_logit = *std::max_element(last_logits.begin(), last_logits.end());
    std::vector<float> probabilities(last_logits.size());
    float sum_exp = 0.0f;
    for (size_t i = 0; i < last_logits.size(); i++) {
        probabilities[i] = std::exp(last_logits[i] - max_logit);
        sum_exp += probabilities[i];
    }
    for (float& prob : probabilities) {
        prob /= sum_exp;
    }

    // Create index vector for top-k selection
    std::vector<size_t> indices(probabilities.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Only filter out standalone delimiter tokens, not tokens containing delimiters
    const std::unordered_set<std::string> delimiters = {"|", "#", "*"};
    for (size_t i = 0; i < probabilities.size(); i++) {
        std::string token = tokenizer.decode({static_cast<int>(i)});
        if (delimiters.find(token) != delimiters.end()) {
            probabilities[i] = 0.0f;
        }
    }

    // Add diversity bonus for less frequent tokens
    std::unordered_map<std::string, int> token_categories;
    for (size_t i = 0; i < probabilities.size(); i++) {
        if (probabilities[i] > 0.0f) {
            std::string token = tokenizer.decode({static_cast<int>(i)});
            std::string category = "OTHER";
            if (tokenizer.is_verb(token)) category = "VERB";
            else if (tokenizer.is_adjective(token)) category = "ADJ";
            else if (tokenizer.is_noun(token)) category = "NOUN";
            token_categories[category]++;
            
            // Boost underrepresented categories
            if (token_categories[category] < 3) {  // Boost first few tokens of each category
                probabilities[i] *= 1.2f;
            }
        }
    }

    // Renormalize probabilities
    sum_exp = std::accumulate(probabilities.begin(), probabilities.end(), 0.0f);
    if (sum_exp > 0.0f) {
        for (float& prob : probabilities) {
            prob /= sum_exp;
        }
    }

    // Sort indices by probability
    std::partial_sort(indices.begin(), 
                     indices.begin() + std::min(k, static_cast<int>(indices.size())),
                     indices.end(),
                     [&probabilities](size_t a, size_t b) {
                         return probabilities[a] > probabilities[b];
                     });

    // Print top k predictions with detailed information
    std::cout << "\nTop " << k << " predictions:" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    std::cout << std::setw(5) << "Rank" 
              << std::setw(20) << "Token" 
              << std::setw(15) << "Category"
              << std::setw(15) << "Probability"
              << std::setw(15) << "Logit" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    int printed = 0;
    std::unordered_set<std::string> seen_categories;  // Track category diversity
    
    for (int i = 0; i < k && i < static_cast<int>(indices.size()); i++) {
        size_t idx = indices[i];
        if (probabilities[idx] > 0.0f) {
            std::string token = tokenizer.decode({static_cast<int>(idx)});
            std::string category = "OTHER";
            if (tokenizer.is_verb(token)) category = "VERB";
            else if (tokenizer.is_adjective(token)) category = "ADJ";
            else if (tokenizer.is_noun(token)) category = "NOUN";
            
            // Ensure category diversity in top predictions
            if (printed < 3 || seen_categories.find(category) == seen_categories.end()) {
                seen_categories.insert(category);
                
                // Format token for display
                std::string display_token = token;
                if (display_token.length() > 15) {
                    display_token = display_token.substr(0, 12) + "...";
                }
                
                std::cout << std::fixed << std::setprecision(4)
                         << std::setw(5) << (printed + 1)
                         << std::setw(20) << ("\"" + display_token + "\"")
                         << std::setw(15) << category
                         << std::setw(15) << probabilities[idx]
                         << std::setw(15) << last_logits[idx] << std::endl;
                printed++;
            }
        }
    }
    std::cout << std::string(50, '-') << std::endl;

    // Save predictions with actual context
    save_predictions_to_csv(logits, tokenizer, transformer.get_current_context(), "predictions.csv", k);
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
    // num_folds should be passed from config.training.cross_validation.num_folds
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
    const std::vector<std::pair<std::string, std::string>>& data) {
    
    const size_t num_folds = transformer.getConfig().training.cross_validation.num_folds;  // Get from config instead of hardcoding
    const size_t fold_size = data.size() / num_folds;
    float total_loss = 0.0f;

    std::cout << "\nPerforming " << num_folds << "-fold cross-validation..." << std::endl;
    
    // Create shuffled indices
    std::vector<size_t> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    // Pre-compute tokenization for all data to avoid redundant work
    struct ProcessedData {
        std::vector<int> input_tokens;
        std::vector<int> target_tokens;
    };
    std::vector<ProcessedData> processed_data;
    processed_data.reserve(data.size());

    #pragma omp parallel for
    for (size_t i = 0; i < data.size(); i++) {
        const auto& [input_str, target_str] = data[i];
        std::string processed_input = input_str;
        std::string processed_target = target_str;
        tokenizer.preprocess_text(processed_input);
        tokenizer.preprocess_text(processed_target);
        
        ProcessedData proc_data;
        proc_data.input_tokens = tokenizer.encode(processed_input);
        proc_data.target_tokens = tokenizer.encode(processed_target);
        processed_data.push_back(proc_data);
    }

    // For each fold
    for (size_t fold = 0; fold < num_folds; fold++) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Split data into train and validation using pre-computed tokens
        std::vector<ProcessedData> val_data;
        size_t start_idx = fold * fold_size;
        size_t end_idx = (fold == num_folds - 1) ? indices.size() : (fold + 1) * fold_size;
        
        for (size_t i = start_idx; i < end_idx; i++) {
            val_data.push_back(processed_data[indices[i]]);
        }

        // Evaluate on validation fold using pre-computed tokens
        float fold_loss = 0.0f;
        size_t total_samples = 0;
        const size_t BATCH_SIZE = 32;

        transformer.set_training(false);

        // Process validation data in batches
        for (size_t batch_start = 0; batch_start < val_data.size(); batch_start += BATCH_SIZE) {
            size_t batch_end = std::min(batch_start + BATCH_SIZE, val_data.size());
            size_t current_batch_size = batch_end - batch_start;

            // Prepare batch matrices
            std::vector<Matrix> batch_logits;
            batch_logits.reserve(current_batch_size);

            // Forward pass for batch
            for (size_t i = batch_start; i < batch_end; i++) {
                const auto& tokens = val_data[i].input_tokens;
                if (!tokens.empty()) {
                    Matrix logits = transformer.forward(tokens, "", tokenizer);
                    batch_logits.push_back(std::move(logits));
                }
            }

            if (batch_logits.empty()) continue;

            // Create target distributions for batch
            std::vector<std::vector<int>> target_tokens;
            for (size_t i = batch_start; i < batch_end; i++) {
                target_tokens.push_back(val_data[i].target_tokens);
            }

            Matrix target_distribution = create_batch_target_distribution(
                target_tokens, tokenizer, tokenizer.vocab_size(), current_batch_size);

            // Compute batch loss
            for (size_t i = 0; i < batch_logits.size(); i++) {
                float sample_loss = compute_batch_loss(batch_logits[i], target_distribution, tokenizer);
                if (std::isfinite(sample_loss)) {
                    fold_loss += sample_loss;
                    total_samples++;
                }
            }
        }

        transformer.set_training(true);
        
        // Compute average loss for fold
        float avg_fold_loss = total_samples > 0 ? fold_loss / total_samples : 0.0f;
        total_loss += avg_fold_loss;

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        
        std::cout << "Fold " << fold + 1 << "/" << num_folds 
                  << " (Loss: " << avg_fold_loss 
                  << ", Time: " << duration.count() << "s)" << std::endl;
    }
    
    float avg_loss = total_loss / num_folds;
    std::cout << "Cross-validation complete. Average loss: " << avg_loss << std::endl;
    return avg_loss;
}

void Utils::generate_predictions(
    Transformer& transformer,
    const std::string& input_text,
    std::shared_ptr<TiktokenTokenizer> tokenizer) {
    
    if (!tokenizer) {
        std::cerr << "Error: TiktokenTokenizer is null" << std::endl;
        return;
    }

    std::cout << "\n=== Processing input: '" << input_text << "' ===" << std::endl;
    
    // Set the current context in the transformer
    transformer.set_current_context(input_text);
    
    // Preprocess input
    std::string processed_input = input_text;
    tokenizer->preprocess_text(processed_input);
    std::vector<int> input_tokens = tokenizer->encode(processed_input);
    
    // Print input tokens
    std::cout << "Input tokens: ";
    for (const auto& token : input_tokens) {
        std::cout << "'" << tokenizer->decode({token}) << "' ";
    }
    std::cout << std::endl;
    
    // Set generation parameters
    const size_t max_length = input_tokens.size() + 50;  // Generate up to 50 new tokens
    const float base_temperature = 0.8f;  // Lower temperature for more focused generation
    
    // Generate sequence
    std::cout << "\nGenerating sequence..." << std::endl;
    transformer.set_training(false);  // Ensure we're in inference mode
    
    try {
        std::vector<int> generated_tokens = transformer.generate(
            input_tokens,
            max_length,
            base_temperature
        );
        
        // Print the generated sequence
        std::cout << "\nGeneration Results:" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        
        // Print original input
        std::cout << "Original input: " << input_text << std::endl;
        
        // Print full generated text
        std::string generated_text = tokenizer->decode(generated_tokens);
        std::cout << "Generated text: " << generated_text << std::endl;
        
        // Print token-by-token breakdown
        std::cout << "\nToken-by-token breakdown:" << std::endl;
        for (size_t i = input_tokens.size(); i < generated_tokens.size(); i++) {
            std::string token_text = tokenizer->decode({generated_tokens[i]});
            std::cout << i - input_tokens.size() + 1 << ". '" << token_text << "' "
                     << "(ID: " << generated_tokens[i] << ")" << std::endl;
        }
        
        // Save generation results
        std::ofstream output_file("generation_results.txt", std::ios::app);
        output_file << "\n=== Generation Results ===\n";
        output_file << "Input: " << input_text << "\n";
        output_file << "Generated: " << generated_text << "\n";
        output_file << "Parameters:\n";
        output_file << "- Base temperature: " << base_temperature << "\n";
        output_file << "- Max length: " << max_length << "\n";
        output_file << "- Tokens generated: " << (generated_tokens.size() - input_tokens.size()) << "\n";
        output_file << std::string(50, '-') << "\n";
        output_file.close();
        
    } catch (const std::exception& e) {
        std::cerr << "Error during generation: " << e.what() << std::endl;
    }
    
    // Also show top predictions for the next token (for comparison)
    std::cout << "\nTop next token predictions (for reference):" << std::endl;
    Matrix logits = transformer.forward(input_tokens, input_text, *tokenizer);
    print_top_predictions(logits, *tokenizer, transformer, 5);
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

void Utils::save_predictions_to_csv(
    const Matrix& logits,
    const TiktokenTokenizer& tokenizer,
    const std::string& input_text,
    const std::string& csv_path,
    int top_k) {
    
    // Check if file exists to determine if we need to write headers
    bool file_exists = std::filesystem::exists(csv_path);
    
    // Open file in append mode
    std::ofstream csv_file(csv_path, std::ios::app);
    if (!csv_file.is_open()) {
        std::cerr << "Failed to open file: " << csv_path << std::endl;
        return;
    }

    // Write headers if file is new
    if (!file_exists) {
        csv_file << "timestamp,input_text,rank,token,probability,logit,category\n";
    }

    // Get current timestamp
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::string timestamp = std::ctime(&time);
    timestamp = timestamp.substr(0, timestamp.length() - 1); // Remove newline

    // Get the last row of logits (predictions for the next token)
    std::vector<float> last_logits;
    for (size_t i = 0; i < logits.cols(); i++) {
        last_logits.push_back(logits(logits.rows() - 1, i));
    }

    // Apply softmax to get probabilities
    float max_logit = *std::max_element(last_logits.begin(), last_logits.end());
    std::vector<float> probabilities(last_logits.size());
    float sum_exp = 0.0f;
    for (size_t i = 0; i < last_logits.size(); i++) {
        probabilities[i] = std::exp(last_logits[i] - max_logit);
        sum_exp += probabilities[i];
    }
    for (float& prob : probabilities) {
        prob /= sum_exp;
    }

    // Create index vector for top-k selection
    std::vector<size_t> indices(probabilities.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Sort indices by probability
    std::partial_sort(indices.begin(), 
                     indices.begin() + std::min(top_k, static_cast<int>(indices.size())),
                     indices.end(),
                     [&probabilities](size_t a, size_t b) {
                         return probabilities[a] > probabilities[b];
                     });

    // Write predictions to CSV
    for (int i = 0; i < top_k && i < static_cast<int>(indices.size()); i++) {
        size_t idx = indices[i];
        if (probabilities[idx] > 0.0f) {
            std::string token = tokenizer.decode({static_cast<int>(idx)});
            // Clean token for CSV (escape commas and quotes)
            std::string cleaned_token = token;
            std::replace(cleaned_token.begin(), cleaned_token.end(), ',', ';');
            std::replace(cleaned_token.begin(), cleaned_token.end(), '"', '\'');
            
            // Determine token category
            std::string category = "OTHER";
            if (tokenizer.is_verb(token)) category = "VERB";
            else if (tokenizer.is_adjective(token)) category = "ADJ";
            else if (tokenizer.is_noun(token)) category = "NOUN";

            // Clean input text for CSV
            std::string cleaned_input = input_text;
            std::replace(cleaned_input.begin(), cleaned_input.end(), ',', ';');
            std::replace(cleaned_input.begin(), cleaned_input.end(), '"', '\'');

            // Write row to CSV
            csv_file << std::fixed << std::setprecision(6)
                    << timestamp << ","
                    << "\"" << cleaned_input << "\","
                    << (i + 1) << ","
                    << "\"" << cleaned_token << "\","
                    << probabilities[idx] << ","
                    << last_logits[idx] << ","
                    << category << "\n";
        }
    }

    csv_file.close();
    std::cout << "Predictions saved to " << csv_path << std::endl;
}

bool Utils::validate_input_sequence(
    const std::vector<int>& tokens,
    size_t vocab_size,
    size_t max_seq_length) {
    
    // Check sequence length
    if (tokens.size() > max_seq_length) {
        std::cout << "Sequence too long: " << tokens.size() << " > " << max_seq_length << std::endl;
        return false;
    }

    // Check if tokens are within vocabulary range
    for (const int token : tokens) {
        if (token < 0 || static_cast<size_t>(token) >= vocab_size) {
            std::cout << "Token out of vocabulary range: " << token << " (vocab size: " << vocab_size << ")" << std::endl;
            return false;
        }
    }

    return true;
}