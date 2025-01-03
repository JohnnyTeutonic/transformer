#include "../include/utils.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <random>

Timer::Timer() : is_running(false) {}

void Timer::start() {
    start_time = std::chrono::high_resolution_clock::now();
    is_running = true;
}

double Timer::stop() {
    if (!is_running) {
        return 0.0;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    is_running = false;
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    return duration.count() / 1000.0; // Convert to milliseconds
}

void print_timing_stats(const TimingStats& stats) {
    std::cout << "\n=== Performance Timing Statistics ===\n";
    
    // Forward pass statistics
    if (stats.forward_pass_count > 0) {
        double avg_forward = stats.forward_pass_time / stats.forward_pass_count;
        std::cout << "Forward Pass:\n"
                  << "  Total time: " << stats.forward_pass_time << " ms\n"
                  << "  Count: " << stats.forward_pass_count << "\n"
                  << "  Average: " << avg_forward << " ms/pass\n";
    }
    
    // Backward pass statistics
    if (stats.backward_pass_count > 0) {
        double avg_backward = stats.backward_pass_time / stats.backward_pass_count;
        std::cout << "Backward Pass:\n"
                  << "  Total time: " << stats.backward_pass_time << " ms\n"
                  << "  Count: " << stats.backward_pass_count << "\n"
                  << "  Average: " << avg_backward << " ms/pass\n";
    }
    
    // Validation statistics
    if (stats.validation_count > 0) {
        double avg_validation = stats.validation_time / stats.validation_count;
        std::cout << "Validation:\n"
                  << "  Total time: " << stats.validation_time << " ms\n"
                  << "  Count: " << stats.validation_count << "\n"
                  << "  Average: " << avg_validation << " ms/validation\n";
    }
    
    // Checkpoint statistics
    if (stats.checkpoint_count > 0) {
        double avg_checkpoint = stats.checkpoint_time / stats.checkpoint_count;
        std::cout << "Checkpointing:\n"
                  << "  Total time: " << stats.checkpoint_time << " ms\n"
                  << "  Count: " << stats.checkpoint_count << "\n"
                  << "  Average: " << avg_checkpoint << " ms/checkpoint\n";
    }
    
    // Total statistics
    double total_time = stats.forward_pass_time + stats.backward_pass_time + 
                       stats.validation_time + stats.checkpoint_time;
    std::cout << "\nTotal Statistics:\n"
              << "  Total time: " << total_time << " ms\n"
              << "  Forward pass: " << (stats.forward_pass_time / total_time * 100) << "%\n"
              << "  Backward pass: " << (stats.backward_pass_time / total_time * 100) << "%\n"
              << "  Validation: " << (stats.validation_time / total_time * 100) << "%\n"
              << "  Checkpointing: " << (stats.checkpoint_time / total_time * 100) << "%\n";
    
    std::cout << "===================================\n";
}

void clip_gradients(std::vector<Matrix>& gradients, float threshold) {
    float total_norm = 0.0f;
    
    // Compute total gradient norm
    for (const auto& grad : gradients) {
        for (size_t i = 0; i < grad.size(); i++) {
            total_norm += grad.data()[i] * grad.data()[i];
        }
    }
    total_norm = std::sqrt(total_norm);
    
    // Clip if necessary
    if (total_norm > threshold) {
        float scaling_factor = threshold / (total_norm + 1e-6f);
        for (auto& grad : gradients) {
            for (size_t i = 0; i < grad.size(); i++) {
                grad.data()[i] *= scaling_factor;
            }
        }
    }
}

float adjust_learning_rate(float current_lr, float loss_ratio, size_t step) {
    const size_t WARMUP_STEPS = 1000;  // Longer warmup period
    const float PEAK_LR = 1e-4;        // Peak learning rate
    const float MIN_LR = 1e-6;         // Minimum learning rate
    const float LOSS_SPIKE_THRESHOLD = 1.5f;
    
    // Warmup phase
    if (step < WARMUP_STEPS) {
        return MIN_LR + (PEAK_LR - MIN_LR) * (static_cast<float>(step) / WARMUP_STEPS);
    }
    
    // Cosine decay after warmup
    const size_t DECAY_STEPS = 50000;  // Longer decay period
    float progress = static_cast<float>(step - WARMUP_STEPS) / DECAY_STEPS;
    progress = std::min(1.0f, progress);  // Cap progress at 1.0
    
    // Cosine decay from peak_lr to min_lr
    float decay_factor = 0.5f * (1.0f + std::cos(progress * M_PI));
    float lr = MIN_LR + (PEAK_LR - MIN_LR) * decay_factor;
    
    // Adjust based on loss if needed
    if (loss_ratio > LOSS_SPIKE_THRESHOLD) {
        lr *= 0.5f;  // Reduce learning rate if loss spikes
    }
    
    // Debug output
    if (step % 100 == 0) {
        std::cout << "\nLR Debug - Step: " << step 
                  << ", Progress: " << progress 
                  << ", Decay Factor: " << decay_factor << std::endl;
    }
    
    return std::clamp(lr, MIN_LR, PEAK_LR);
}

bool validate_input_sequence(const std::vector<int>& tokens, size_t vocab_size, size_t max_seq_length) {
    if (tokens.empty() || tokens.size() > max_seq_length) {
        std::cerr << "Invalid sequence length: " << tokens.size() << std::endl;
        return false;
    }
    
    for (int token : tokens) {
        if (token < 0 || token >= static_cast<int>(vocab_size)) {
            std::cerr << "Invalid token id: " << token << std::endl;
            return false;
        }
    }
    
    return true;
}

float compute_batch_loss(const Matrix& logits, const Matrix& targets) {
    float loss = 0.0f;
    const float epsilon = 1e-10f;
    const float temperature = 1.2f;;
    
    for (size_t i = 0; i < logits.rows(); i++) {
        float max_logit = logits(i, 0);
        for (size_t j = 0; j < logits.cols(); j++) {
            max_logit = std::max(max_logit, logits(i, j));
        }
        
        float sum_exp = 0.0f;
        std::vector<float> scaled_probs(logits.cols());
        
        for (size_t j = 0; j < logits.cols(); j++) {
            scaled_probs[j] = std::exp((logits(i, j) - max_logit) / temperature);
            sum_exp += scaled_probs[j];
        }
        
        const float smoothing_factor = 0.1f;
        for (size_t j = 0; j < logits.cols(); j++) {
            float prob = scaled_probs[j] / (sum_exp + epsilon);
            float smooth_target = targets(i, j) * (1.0f - smoothing_factor) + 
                                (smoothing_factor / logits.cols());
            loss -= smooth_target * std::log(prob + epsilon);
        }
    }
    
    return loss / logits.rows();
}

Matrix create_batch_target_distribution(const std::vector<std::vector<int>>& token_sequences, size_t vocab_size) {
    if (token_sequences.empty()) {
        throw std::runtime_error("Cannot create target distribution from empty batch");
    }
    
    Matrix distribution(token_sequences.size(), vocab_size, 0.0f);
    
    for (size_t batch_idx = 0; batch_idx < token_sequences.size(); batch_idx++) {
        const auto& tokens = token_sequences[batch_idx];
        
        if (tokens.empty()) {
            throw std::runtime_error("Empty token sequence in batch at position " + 
                std::to_string(batch_idx));
        }
        
        for (int token : tokens) {
            if (token < 0 || token >= static_cast<int>(vocab_size)) {
                throw std::runtime_error("Token " + std::to_string(token) + 
                    " is out of vocabulary range [0, " + std::to_string(vocab_size) + 
                    ") at batch position " + std::to_string(batch_idx));
            }
        }
        
        float weight = 1.0f / tokens.size();
        for (int token : tokens) {
            distribution(batch_idx, token) = weight;
        }
    }
    
    return distribution;
}

void print_matrix(const Matrix& m, const std::string& name, size_t max_rows, size_t max_cols) {
    std::cout << "\n" << name << " (" << m.rows() << "x" << m.cols() << "):\n";
    for (size_t i = 0; i < std::min(max_rows, m.rows()); ++i) {
        for (size_t j = 0; j < std::min(max_cols, m.cols()); ++j) {
            std::cout << std::fixed << std::setprecision(4) << m(i, j) << " ";
        }
        std::cout << (m.cols() > max_cols ? "..." : "") << "\n";
    }
    if (m.rows() > max_rows)
        std::cout << "...\n";
}

void print_top_predictions(const Matrix& logits, const Tokenizer& tokenizer, size_t k) {
    std::vector<std::pair<float, int>> scores;
    for (size_t i = 0; i < logits.cols(); ++i) {
        scores.push_back({logits(logits.rows() - 1, i), static_cast<int>(i)});
    }

    std::partial_sort(
        scores.begin(), scores.begin() + k, scores.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });

    std::cout << "\nTop " << k << " predictions:\n";
    for (size_t i = 0; i < k; ++i) {
        std::string token = tokenizer.decode({scores[i].second});
        std::cout << i + 1 << ". \"" << token << "\" (probability: " << std::fixed
                 << std::setprecision(4) << std::exp(scores[i].first) << ")\n";
    }
}

std::vector<std::pair<std::string, std::string>> create_training_data() {
    std::vector<std::pair<std::string, std::string>> training_pairs;
    std::filesystem::path exe_path = std::filesystem::current_path().parent_path();
    std::filesystem::path data_dir = exe_path / "data";
    std::filesystem::path file_path = data_dir / "training_pairs.txt";

    if (!std::filesystem::exists(data_dir)) {
        std::filesystem::create_directories(data_dir);
    }

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

    if (training_pairs.empty()) {
        throw std::runtime_error("No training pairs loaded from file");
    }

    return training_pairs;
}

void analyze_token_mappings(const std::vector<std::pair<std::string, std::string>>& training_data, 
                          const Tokenizer& tokenizer) {
    std::cout << "\n=== Analyzing Token Mappings ===\n";
    
    size_t total_words = 0;
    size_t unknown_tokens = 0;
    std::unordered_map<std::string, int> unknown_words;
    
    for (const auto& pair : training_data) {
        std::istringstream input_ss(pair.first);
        std::string word;
        while (input_ss >> word) {
            total_words++;
            std::vector<int> tokens = tokenizer.encode(word);
            for (int token : tokens) {
                if (tokenizer.decode({token}) == "<unk>") {
                    unknown_tokens++;
                    unknown_words[word]++;
                }
            }
        }
        
        std::istringstream target_ss(pair.second);
        while (target_ss >> word) {
            total_words++;
            std::vector<int> tokens = tokenizer.encode(word);
            for (int token : tokens) {
                if (tokenizer.decode({token}) == "<unk>") {
                    unknown_tokens++;
                    unknown_words[word]++;
                }
            }
        }
    }
    
    std::cout << "\nToken Mapping Statistics:\n";
    std::cout << "Total words processed: " << total_words << "\n";
    std::cout << "Unknown token occurrences: " << unknown_tokens << " (" 
              << (100.0f * unknown_tokens / total_words) << "%)\n\n";
    
    if (!unknown_words.empty()) {
        std::cout << "Words mapped to <unk> token:\n";
        for (const auto& [word, count] : unknown_words) {
            std::cout << "'" << word << "': " << count << " times\n";
        }
    }
    
    std::cout << "\n=== End Token Mapping Analysis ===\n\n";
}

std::vector<std::pair<std::string, std::string>> load_data_pairs(const std::string& file_path) {
    std::vector<std::pair<std::string, std::string>> data_pairs;
    std::ifstream file(file_path);
    
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + file_path);
    }

    std::string line;
    while (std::getline(file, line)) {
        size_t delimiter_pos = line.find('|');
        if (delimiter_pos != std::string::npos) {
            std::string input = line.substr(0, delimiter_pos);
            std::string output = line.substr(delimiter_pos + 1);
            // Trim any whitespace
            input.erase(0, input.find_first_not_of(" \t\r\n"));
            input.erase(input.find_last_not_of(" \t\r\n") + 1);
            output.erase(0, output.find_first_not_of(" \t\r\n"));
            output.erase(output.find_last_not_of(" \t\r\n") + 1);
            data_pairs.emplace_back(input, output);
        }
    }

    return data_pairs;
}

DataSet create_dataset(const std::string& mode) {
    DataSet dataset;
    std::filesystem::path file_path;
    
    // Get the current working directory and go up one level
    std::filesystem::path current_path = std::filesystem::current_path();
    std::filesystem::path parent_path = current_path.parent_path();
    
    if (mode == "train") {
        file_path = parent_path / "data" / "training_pairs.txt";
    } else if (mode == "validation") {
        file_path = parent_path / "data" / "validation_pairs.txt";
    } else {
        throw std::invalid_argument("Invalid mode. Use 'train' or 'validation'");
    }
    
    // Check if file exists
    if (!std::filesystem::exists(file_path)) {
        throw std::runtime_error("File not found: " + file_path.string() + 
                               "\nCurrent directory: " + current_path.string() +
                               "\nParent directory: " + parent_path.string());
    }
    
    std::cout << "Loading dataset from: " << file_path << std::endl;
    dataset.pairs = load_data_pairs(file_path.string());
    dataset.size = dataset.pairs.size();
    dataset.current_index = 0;
    
    return dataset;
}

std::vector<std::pair<std::string, std::string>> get_batch(DataSet& dataset, size_t batch_size) {
    std::vector<std::pair<std::string, std::string>> batch;
    batch.reserve(batch_size);
    
    for (size_t i = 0; i < batch_size && dataset.current_index < dataset.size; ++i) {
        batch.push_back(dataset.pairs[dataset.current_index++]);
        
        // Reset index if we've reached the end
        if (dataset.current_index >= dataset.size) {
            dataset.current_index = 0;
            // Shuffle the dataset using modern C++ random facilities
            static std::random_device rd;
            static std::mt19937 gen(rd());
            std::shuffle(dataset.pairs.begin(), dataset.pairs.end(), gen);
        }
    }
    
    return batch;
}

float calculate_accuracy(const Matrix& logits, const Matrix& targets) {
    size_t correct = 0;
    size_t total = logits.rows();
    
    for (size_t i = 0; i < total; ++i) {
        size_t predicted = 0;
        float max_val = logits(i, 0);
        
        for (size_t j = 1; j < logits.cols(); ++j) {
            if (logits(i, j) > max_val) {
                max_val = logits(i, j);
                predicted = j;
            }
        }
        
        size_t actual = 0;
        max_val = targets(i, 0);
        for (size_t j = 1; j < targets.cols(); ++j) {
            if (targets(i, j) > max_val) {
                max_val = targets(i, j);
                actual = j;
            }
        }
        
        if (predicted == actual) {
            correct++;
        }
    }
    
    return static_cast<float>(correct) / total;
} 