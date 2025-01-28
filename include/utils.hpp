#pragma once
#include "matrix.hpp"
#include "tokenizer.hpp"
#include "transformer.hpp"
#include <string>
#include <utility>
#include <vector>
#include <chrono>
#include <ctime>

namespace Utils {
    std::string get_current_time();
    float adjust_learning_rate(float current_lr, float loss_ratio, size_t step);
    bool validate_input_sequence(const std::vector<int>& tokens, size_t vocab_size,
                               size_t max_seq_length = 512);
    void print_matrix(const Matrix& m, const std::string& name, size_t max_rows = 5,
                     size_t max_cols = 5);
    void print_top_predictions(const Matrix& logits, const Tokenizer& tokenizer, 
                             Transformer& transformer, int k);
    std::vector<std::pair<std::string, std::string>> create_training_data();
    void analyze_token_mappings(const std::vector<std::pair<std::string, std::string>>& training_data,
                              const Tokenizer& tokenizer);
    std::vector<std::pair<std::string, std::string>> load_validation_data();
    float evaluate_validation(Transformer& transformer, const Tokenizer& tokenizer,
                            const std::vector<std::pair<std::string, std::string>>& validation_data);
    TransformerConfig load_config(const std::string& config_path);
    Matrix create_batch_target_distribution(const std::vector<std::vector<int>>& target_tokens,
                                         const Tokenizer& tokenizer, size_t vocab_size,
                                         size_t input_max_seq_len);
    float compute_batch_loss(const Matrix& logits, const Matrix& target_distribution, const Tokenizer& tokenizer);
    void apply_sampling_parameters(std::vector<float>& logits, float temperature,
                                 float top_p);
    std::vector<std::string>& get_vocabulary(const Tokenizer& tokenizer);
    std::vector<std::pair<std::string, float>> get_multi_token_predictions(
        const Matrix& logits, const Tokenizer& tokenizer, int beam_width);
}