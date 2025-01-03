#pragma once
#include <string>
#include <vector>
#include <utility>
#include "matrix.hpp"
#include "tokenizer.hpp"

// Helper function to clip gradients
void clip_gradients(std::vector<Matrix>& gradients, float threshold);

// Helper function to adjust learning rate
float adjust_learning_rate(float current_lr, float loss_ratio, size_t step);

// Helper function to validate input tokens
bool validate_input_sequence(const std::vector<int>& tokens, size_t vocab_size, size_t max_seq_length = 512);

// Helper function to compute loss with improved numerical stability
float compute_batch_loss(const Matrix& logits, const Matrix& targets);

// Helper function to create target distribution for a batch
Matrix create_batch_target_distribution(const std::vector<std::vector<int>>& token_sequences, size_t vocab_size);

// Helper function to print matrix contents
void print_matrix(const Matrix& m, const std::string& name, size_t max_rows = 5, size_t max_cols = 5);

// Helper function to print top predictions
void print_top_predictions(const Matrix& logits, const Tokenizer& tokenizer, size_t k = 5);

// Helper function to create training data
std::vector<std::pair<std::string, std::string>> create_training_data();

// Helper function to analyze token mappings
void analyze_token_mappings(const std::vector<std::pair<std::string, std::string>>& training_data, 
                          const Tokenizer& tokenizer); 