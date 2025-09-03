#pragma once
#include "tiktoken_tokenizer.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include "matrix.hpp"
#include "phrase_types.hpp"

// Penalty computation functions
float compute_verb_penalty(const Matrix& logits, const std::vector<int>& final_tokens,
                         const TiktokenTokenizer& tokenizer);

float compute_adjective_penalty(const Matrix& logits, const std::vector<int>& final_tokens,
                              const TiktokenTokenizer& tokenizer);

// Gradient factor computation functions
float verb_gradient_factor(size_t position, const std::vector<int>& tokens,
                         const TiktokenTokenizer& tokenizer);

float adjective_gradient_factor(size_t position, const std::vector<int>& tokens,
                              const TiktokenTokenizer& tokenizer); 