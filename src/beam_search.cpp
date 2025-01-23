#include "../include/beam_search.hpp"
#include "../include/cuda/matrix_ops.cuh"
#include "../include/cuda/memory_manager.cuh"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>

BeamSearch::BeamSearch(size_t beam_width, float length_penalty)
    : beam_width_(beam_width), length_penalty_(length_penalty) {
    std::cout << "Initializing BeamSearch with width=" << beam_width
              << ", length_penalty=" << length_penalty << std::endl;
}

float BeamSearch::apply_length_penalty(float score, size_t length) const {
    float penalized_score = score / std::pow(length, length_penalty_);
    std::cout << "Applied length penalty: original_score=" << std::fixed << std::setprecision(4)
              << score << ", length=" << length << ", penalized_score=" << penalized_score
              << std::endl;
    return penalized_score;
}

void BeamSearch::update_beams(std::vector<std::vector<int>>& sequences,
                              Matrix& beam_scores,
                              const Matrix& next_scores,
                              const std::vector<int>& next_tokens) {
    // Create temporary vectors to store candidates
    std::vector<std::pair<float, std::pair<size_t, int>>> candidates;
    candidates.reserve(beam_width_ * beam_width_);
    
    // Gather all candidates from all beams
    for (size_t i = 0; i < beam_width_; i++) {
        for (size_t j = 0; j < beam_width_; j++) {
            float score = beam_scores(i, 0) + next_scores(i, j);
            candidates.push_back({score, {i, next_tokens[j]}});
        }
    }
    
    // Sort candidates by score
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Create new sequences and scores
    std::vector<std::vector<int>> new_sequences;
    Matrix new_scores(beam_width_, 1);
    
    // Take top beam_width_ candidates
    for (size_t i = 0; i < beam_width_; i++) {
        const auto& [score, beam_token] = candidates[i];
        const auto& [beam_idx, token] = beam_token;
        
        // Copy sequence from parent beam
        new_sequences.push_back(sequences[beam_idx]);
        new_sequences.back().push_back(token);
        new_scores(i, 0) = score;
    }
    
    // Update sequences and scores
    sequences = std::move(new_sequences);
    beam_scores = std::move(new_scores);
}

bool BeamSearch::is_search_complete(const std::vector<std::vector<int>>& sequences) {
    // Check if all sequences have reached the end token
    for (const auto& seq : sequences) {
        if (seq.empty()) return false;
        
        // Check if sequence ends with end token (usually 2 for GPT models)
        if (seq.back() == 2) return true;
        
        // Check if sequence has reached maximum length (e.g., 1024 tokens)
        if (seq.size() >= 1024) return true;
    }
    return false;
}

std::vector<int> BeamSearch::get_best_sequence(
    const std::vector<std::vector<int>>& sequences,
    const Matrix& beam_scores) {
    // Find sequence with highest score after length penalty
    float best_score = -std::numeric_limits<float>::infinity();
    size_t best_idx = 0;
    
    for (size_t i = 0; i < sequences.size(); i++) {
        float penalized_score = apply_length_penalty(beam_scores(i, 0), sequences[i].size());
        if (penalized_score > best_score) {
            best_score = penalized_score;
            best_idx = i;
        }
    }
    
    return sequences[best_idx];
}

std::vector<int> BeamSearch::cpu_beam_search(
    const std::vector<float>& initial_logits,
    size_t max_length,
    const std::function<std::vector<float>(const std::vector<int>&)>& next_token_fn) {
    // Initialize with top beam_width_ tokens
    std::vector<std::pair<std::vector<int>, float>> beams;
    std::vector<std::pair<float, int>> top_tokens;
    
    // Get initial top tokens
    for (size_t i = 0; i < initial_logits.size(); i++) {
        top_tokens.push_back({initial_logits[i], static_cast<int>(i)});
    }
    
    std::partial_sort(top_tokens.begin(), 
                      top_tokens.begin() + beam_width_,
                      top_tokens.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Initialize beams
    for (size_t i = 0; i < beam_width_; i++) {
        beams.push_back({{top_tokens[i].second}, top_tokens[i].first});
    }
    
    // Main search loop
    for (size_t step = 1; step < max_length; step++) {
        std::vector<std::pair<std::vector<int>, float>> new_beams;
        
        // Expand each beam
        for (const auto& [sequence, score] : beams) {
            if (sequence.back() == 2) {  // End token
                new_beams.push_back({sequence, score});
                continue;
            }
            
            // Get next token logits from model using callback
            std::vector<float> next_logits = next_token_fn(sequence);
            if (next_logits.empty()) {
                continue;  // Skip if model returns no predictions
            }
            
            // Convert logits to probabilities using softmax
            float max_logit = *std::max_element(next_logits.begin(), next_logits.end());
            std::vector<float> probs(next_logits.size());
            float sum_exp = 0.0f;
            
            for (size_t i = 0; i < next_logits.size(); i++) {
                probs[i] = std::exp(next_logits[i] - max_logit);
                sum_exp += probs[i];
            }
            
            // Get top k candidates for this beam
            std::vector<std::pair<float, int>> candidates;
            for (size_t i = 0; i < probs.size(); i++) {
                float prob = probs[i] / sum_exp;
                float new_score = score + std::log(prob + 1e-10);  // Add log probability
                candidates.push_back({new_score, static_cast<int>(i)});
            }
            
            // Sort and take top candidates
            std::partial_sort(candidates.begin(), 
                            candidates.begin() + beam_width_, 
                            candidates.end(),
                            std::greater<std::pair<float, int>>());
            
            // Add top candidates to new beams
            for (size_t i = 0; i < beam_width_; i++) {
                std::vector<int> new_sequence = sequence;
                new_sequence.push_back(candidates[i].second);
                new_beams.push_back({new_sequence, candidates[i].first});
            }
        }
        
        // Sort and prune beams
        std::partial_sort(new_beams.begin(),
                         new_beams.begin() + beam_width_,
                         new_beams.end(),
                         [](const auto& a, const auto& b) { 
                             return a.second > b.second; 
                         });
        
        beams.clear();
        beams.insert(beams.end(), 
                    new_beams.begin(),
                    new_beams.begin() + std::min(beam_width_, new_beams.size()));
        
        // Check if all beams have ended
        bool all_ended = true;
        for (const auto& [sequence, _] : beams) {
            if (sequence.back() != 2) {
                all_ended = false;
                break;
            }
        }
        if (all_ended) break;
    }
    
    // Return sequence with highest score
    return std::max_element(beams.begin(), beams.end(),
                           [this](const auto& a, const auto& b) {
                               float score_a = apply_length_penalty(a.second, a.first.size());
                               float score_b = apply_length_penalty(b.second, b.first.size());
                               return score_a < score_b;
                           })->first;
}

std::vector<BeamSearch::Hypothesis>
BeamSearch::search(const std::vector<float>& initial_logits,
                   std::function<std::vector<float>(const std::vector<int>&)> next_token_fn,
                   size_t max_length, int eos_token_id) {
    // Separate function for single token prediction
    auto predict_single_token = [](const std::vector<float>& logits) -> Hypothesis {
        if (logits.empty()) {
            throw std::runtime_error("Empty logits vector");
        }
        
        if (logits.size() != 1443) {  // Replace with tokenizer.vocab_size()
            throw std::runtime_error("Logits size mismatch: " + 
                std::to_string(logits.size()) + " vs vocab size 1443");
        }
        
        // Find max logit directly
        auto max_it = std::max_element(logits.begin(), logits.end());
        int predicted_token = std::distance(logits.begin(), max_it);
        
        if (predicted_token < 0 || predicted_token >= static_cast<int>(logits.size())) {
            throw std::runtime_error("Invalid predicted token index");
        }
        
        std::cout << "Single token prediction:" << std::endl;
        // Note: We can't decode the token here since BeamSearch doesn't have access to tokenizer
        // Instead, we'll let the caller handle the decoding
        std::cout << "- Score: " << *max_it << std::endl;
        
        return Hypothesis{{predicted_token}, *max_it};
    };

    // For single token prediction, bypass beam search entirely
    if (max_length == 1) {
        return {predict_single_token(initial_logits)};
    }

    // Only use beam search for sequence generation
    try {
#ifdef USE_CUDA
        try {
            auto& memory_mgr = cuda::MemoryManager::instance();
            printf("accessing memory manager\n");
            
            // Initialize beam scores and sequences (only for sequence generation)
            Matrix beam_scores(beam_width_, 1);
            std::vector<std::vector<int>> sequences(beam_width_);
            printf("initialized beam scores and sequences\n");
            // Get initial top-k candidates
            Matrix top_k_logits(beam_width_, 1);
            std::vector<int> top_k_indices(beam_width_);
            cuda::topk(initial_logits, top_k_logits, top_k_indices, beam_width_);
            printf("got initial top-k candidates\n");
            
            // Initialize sequences with top-k tokens
            for (size_t i = 0; i < beam_width_; i++) {
                sequences[i].push_back(top_k_indices[i]);
                beam_scores(i, 0) = top_k_logits(i, 0);
            }
            printf("initialized sequences with top-k tokens\n");
            // Main beam search loop
            for (size_t step = 1; step < max_length; step++) {
                Matrix next_scores;
                std::vector<int> next_tokens;
                
                // Get next token predictions
                // Get predictions for all sequences in beam
                std::vector<float> all_logits;
                for (const auto& sequence : sequences) {
                    auto logits = next_token_fn(sequence);
                    if (logits.empty()) {
                        throw std::runtime_error("Empty logits returned from next_token_fn");
                    }
                    all_logits.insert(all_logits.end(), logits.begin(), logits.end());
                }
                printf("got all logits\n");
                // Convert to matrix with shape (beam_width, vocab_size)
                size_t vocab_size = all_logits.size() / sequences.size();
                Matrix model_output(sequences.size(), vocab_size);
                std::copy(all_logits.begin(), all_logits.end(), model_output.data());
                printf("converted to matrix\n");
                // Verify dimensions before CUDA call
                if (model_output.rows() == 0 || model_output.cols() == 0) {
                    throw std::runtime_error("Invalid model output dimensions: " + 
                        std::to_string(model_output.rows()) + "x" + 
                        std::to_string(model_output.cols()));
                }
                
                printf("Model output dimensions: %zux%zu\n", model_output.rows(), model_output.cols());
                printf("Beam scores dimensions: %zux%zu\n", beam_scores.rows(), beam_scores.cols());

                cuda::beam_search_step(model_output, beam_scores, 
                                     next_scores, next_tokens, beam_width_);
                
                // Update sequences and scores
                update_beams(sequences, beam_scores, next_scores, next_tokens);
                
                // Check for completion
                if (is_search_complete(sequences)) break;
            }
            printf("beam search loop completed\n");
            // Return best sequence
            std::vector<Hypothesis> hypotheses;
            hypotheses.push_back(Hypothesis{sequences[0], beam_scores(0, 0)});
            return hypotheses;
            
        } catch (const std::runtime_error& e) {
            std::cerr << "CUDA beam search failed, falling back to CPU: " << e.what() << std::endl;
#endif
            // For single token prediction, don't fall back to beam search
            if (max_length == 1) {
                throw std::runtime_error("Single token prediction failed");
            }
            
            // CPU fallback implementation only for sequence generation
            auto result = cpu_beam_search(initial_logits, max_length, next_token_fn);
            std::vector<Hypothesis> hypotheses;
            hypotheses.push_back(Hypothesis{result, 0.0f});
            return hypotheses;
#ifdef USE_CUDA
        }
#endif
    } catch (const std::exception& e) {
        throw std::runtime_error("Beam search failed: " + std::string(e.what()));
    }
}