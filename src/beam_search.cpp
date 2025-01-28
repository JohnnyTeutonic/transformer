#include "../include/beam_search.hpp"
#include "../include/cuda/matrix_ops.cuh"
#include "../include/cuda/memory_manager.cuh"
#include "../include/utils.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

BeamSearch::BeamSearch(const TransformerConfig& config,
                       size_t beam_width, float length_penalty, float temperature,
                       float diversity_strength, size_t top_k, float top_p)
    : beam_width_(beam_width)
    , length_penalty_(length_penalty)
    , temperature(std::min(temperature, 0.5f))  // Lower temperature for more focused predictions
    , diversity_strength(std::min(diversity_strength, 0.5f))  // Lower diversity for more coherent completions
    , top_k(std::min(top_k, size_t(50)))  // Limit top_k to prevent too much diversity
    , top_p(std::min(top_p, 0.9f))  // Lower top_p for more focused sampling
    , max_length_(3)  // Hard limit to 3 tokens for completion
    , config_(config)  // Store reference to config
{
    std::cout << "Initializing BeamSearch with width=" << beam_width
              << ", length_penalty=" << length_penalty 
              << ", temperature=" << temperature
              << ", diversity_strength=" << diversity_strength
              << ", top_k=" << top_k
              << ", top_p=" << top_p 
              << ", max_length=" << max_length_ << std::endl;
    // Use consistent special token IDs
    pad_token_id_ = 0;
    unk_token_id_ = 1;
    bos_token_id_ = 2;
    eos_token_id_ = 3;
    mask_token_id_ = 4;
}

float BeamSearch::apply_length_penalty(float score, size_t length) const {
    // Reduce length penalty impact and favor shorter completions
    float penalized_score = score / std::pow(length, length_penalty_ * 0.3f);
    return penalized_score;
}

void BeamSearch::update_beams(std::vector<std::vector<int>>& sequences,
                              Matrix& beam_scores,
                              const Matrix& next_scores,
                              const std::vector<int>& next_tokens) {
    // Add dimension check to prevent re-projection
    if (next_scores.cols() == config_.vocab_size) {
        // Data is already projected to vocab space, use directly
        process_projected_scores(sequences, beam_scores, next_scores, next_tokens);
        return;
    }
    
    const float MIN_SCORE = -1e2f;  // Less extreme minimum
    
    // Create temporary vectors to store candidates
    std::vector<std::pair<float, std::pair<size_t, int>>> candidates;
    candidates.reserve(beam_width_ * beam_width_);
    
    // Track token frequencies across all beams with reduced impact
    std::unordered_map<int, float> token_counts;
    for (const auto& seq : sequences) {
        for (int token : seq) {
            token_counts[token] += 0.5f;  // Reduced penalty for repeated tokens
        }
    }
    
    // For each beam
    for (size_t i = 0; i < sequences.size(); ++i) {
        // Check if sequence has reached max length
        if (sequences[i].size() >= max_length_) {
            continue;  // Skip this beam if it's already at max length
        }
        
        // Get top-k candidates for this beam
        std::vector<std::pair<float, int>> beam_candidates;
        for (size_t j = 0; j < next_scores.cols(); ++j) {
            float score = next_scores(i, j);
            
            // Apply temperature and diversity penalty
            score /= temperature;
            if (token_counts.count(j)) {
                score -= diversity_strength * token_counts[j];
            }
            
            beam_candidates.emplace_back(score, j);
        }
        
        // Sort and keep top-k
        std::partial_sort(beam_candidates.begin(),
                         beam_candidates.begin() + std::min(top_k, beam_candidates.size()),
                         beam_candidates.end(),
                         std::greater<>());
        
        // Add candidates to the pool
        for (size_t j = 0; j < std::min(top_k, beam_candidates.size()); ++j) {
            float score = beam_candidates[j].first + beam_scores(i, 0);
            candidates.emplace_back(score, std::make_pair(i, beam_candidates[j].second));
        }
    }
    
    // Sort all candidates
    std::sort(candidates.begin(), candidates.end(),
              std::greater<std::pair<float, std::pair<size_t, int>>>());
    
    // Update sequences and scores with top candidates
    std::vector<std::vector<int>> new_sequences;
    Matrix new_scores(beam_width_, 1);
    size_t num_updated = 0;
    
    for (const auto& candidate : candidates) {
        if (num_updated >= beam_width_) break;
        
        size_t beam_idx = candidate.second.first;
        int token = candidate.second.second;
        
        // Create new sequence
        std::vector<int> new_seq = sequences[beam_idx];
        new_seq.push_back(token);
        
        // Only add if sequence is unique and not too long
        if (new_seq.size() <= max_length_ &&
            std::find(new_sequences.begin(), new_sequences.end(), new_seq) == new_sequences.end()) {
            new_sequences.push_back(new_seq);
            new_scores(num_updated, 0) = candidate.first;
            num_updated++;
        }
    }
    
    // Update the beams
    sequences = std::move(new_sequences);
    beam_scores = std::move(new_scores);
}

bool BeamSearch::is_search_complete(const std::vector<std::vector<int>>& sequences) {
    // Check if all sequences have reached the end token or max length
    const size_t MAX_LENGTH = 1024;  // Explicit constant
    for (const auto& seq : sequences) {
        if (seq.empty()) return false;
        
        // Continue searching if any sequence hasn't ended and hasn't reached max length
        if (seq.back() != eos_token_id_ && seq.size() < MAX_LENGTH) {
            return false;
        }
    }
    return true;  // All sequences have ended or reached max length
}

std::vector<BeamSearch::Hypothesis> BeamSearch::get_best_sequence(
    const std::vector<std::vector<int>>& sequences,
    const Matrix& beam_scores
) {
    std::vector<Hypothesis> best_hypotheses;
    
    // Get the final scores from the last position
    Vector row_vector = beam_scores.row(beam_scores.rows() - 1);
    std::vector<float> final_scores(row_vector.begin(), row_vector.end());
    
    // Create hypotheses with sequences and their scores
    for (size_t i = 0; i < sequences.size(); ++i) {
        Hypothesis hyp;
        hyp.sequence = sequences[i];
        hyp.score = final_scores[i];
        best_hypotheses.push_back(hyp);
    }
    
    // Sort hypotheses by score in descending order
    std::sort(best_hypotheses.begin(), best_hypotheses.end(),
        [](const Hypothesis& a, const Hypothesis& b) {
            return a.score > b.score;
        });
    
    // Keep only the top hypotheses based on num_groups
    if (best_hypotheses.size() > config_.beam_search.num_groups) {
        best_hypotheses.resize(config_.beam_search.num_groups);
    }
    
    return best_hypotheses;
}

std::vector<int> BeamSearch::cpu_beam_search(
    const std::vector<float>& initial_logits,
    std::function<std::vector<float>(const std::vector<int>&)> next_token_fn,
    size_t max_length) {
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
            if (sequence.back() == eos_token_id_) {  // Use consistent eos_token_id_
                new_beams.push_back({sequence, score});
                continue;
            }
            
            // Get next token logits from the model
            std::vector<float> next_logits = next_token_fn(sequence);
            
            // Apply temperature scaling and sampling
            std::vector<float> scaled_probs = calculateScores(next_logits);
            
            // Get top-k candidates
            auto candidate_pairs = topKSampling(scaled_probs, top_k);
            std::vector<size_t> candidate_indices;
            for (const auto& pair : candidate_pairs) {
                candidate_indices.push_back(pair.second);
            }
            
            // Apply nucleus sampling if enabled
            if (top_p < 1.0f) {
                auto nucleus_pairs = nucleusSampling(scaled_probs, top_p);
                std::vector<size_t> nucleus_indices;
                for (const auto& pair : nucleus_pairs) {
                    nucleus_indices.push_back(pair.second);
                }
                
                // Use intersection of top-k and nucleus sampling
                std::vector<size_t> filtered_indices;
                std::set_intersection(
                    candidate_indices.begin(), candidate_indices.end(),
                    nucleus_indices.begin(), nucleus_indices.end(),
                    std::back_inserter(filtered_indices)
                );
                candidate_indices = filtered_indices;
            }
            
            // Create beam candidates
            std::vector<BeamCandidate> candidates;
            for (size_t idx : candidate_indices) {
                // Convert int sequence to size_t sequence
                std::vector<size_t> new_sequence;
                new_sequence.reserve(sequence.size() + 1);
                for (int token : sequence) {
                    new_sequence.push_back(static_cast<size_t>(token));
                }
                new_sequence.push_back(idx);
                float new_score = score + std::log(scaled_probs[idx]);
                candidates.push_back(BeamCandidate(new_sequence, new_score));
            }
            
            // Apply diversity penalty
            diversityPenalty(candidates, diversity_strength);
            
            // Add top candidates to new beams
            std::sort(candidates.begin(), candidates.end(),
                     [](const auto& a, const auto& b) { return a.score > b.score; });
            
            size_t num_to_add = std::min(beam_width_, candidates.size());
            for (size_t i = 0; i < num_to_add; i++) {
                // Convert size_t sequence back to int sequence
                std::vector<int> int_sequence;
                int_sequence.reserve(candidates[i].sequence.size());
                for (size_t token : candidates[i].sequence) {
                    int_sequence.push_back(static_cast<int>(token));
                }
                new_beams.push_back({int_sequence, candidates[i].score});
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
            if (sequence.back() != eos_token_id_) {  // Use consistent eos_token_id_
                all_ended = false;
                break;
            }
        }
        if (all_ended) break;
    }
    
    // Return sequence with highest score after length penalty
    return std::max_element(beams.begin(), beams.end(),
                           [this](const auto& a, const auto& b) {
                               float score_a = apply_length_penalty(a.second, a.first.size());
                               float score_b = apply_length_penalty(b.second, b.first.size());
                               return score_a < score_b;
                           })->first;
}

std::vector<BeamSearch::Hypothesis> BeamSearch::search(
    const std::vector<float>& initial_logits,
    std::function<std::vector<float>(const std::vector<int>&)> next_token_fn,
    size_t max_length, int eos_token_id) {
    
    // Get hypotheses from appropriate implementation
    std::vector<Hypothesis> hypotheses;
#ifdef USE_CUDA
    try {
        hypotheses = search_cuda(initial_logits, next_token_fn, max_length, eos_token_id);
    } catch (const std::runtime_error& e) {
        std::cerr << "CUDA search failed, falling back to CPU: " << e.what() << std::endl;
        hypotheses = search_cpu(initial_logits, next_token_fn, max_length, eos_token_id);
    }
#else
    hypotheses = search_cpu(initial_logits, next_token_fn, max_length, eos_token_id);
#endif

    // Sort hypotheses by score
    std::sort(hypotheses.begin(), hypotheses.end(),
              [this](const auto& a, const auto& b) {
                  float score_a = apply_length_penalty(a.score, a.sequence.size());
                  float score_b = apply_length_penalty(b.score, b.sequence.size());
                  return score_a > score_b;
              });

    // Take top 5 hypotheses
    if (hypotheses.size() > 5) {
        hypotheses.resize(5);
    }

    // Print the top 5 predictions with their scores
    std::cout << "\nTop " << hypotheses.size() << " predictions:\n";
    for (size_t i = 0; i < hypotheses.size(); i++) {
        float penalized_score = apply_length_penalty(hypotheses[i].score, hypotheses[i].sequence.size());
        std::cout << i + 1 << ". Score: " << std::fixed << std::setprecision(4) 
                  << penalized_score << "\n";
    }
    std::cout << std::endl;

    return hypotheses;
}

void BeamSearch::diversityPenalty(std::vector<BeamCandidate>& candidates, float strength) {
    // Apply much stronger diversity penalty
    const float base_penalty = strength * 4.0f;  // Increased from 2.0f to 4.0f
    const float unk_penalty = base_penalty * 2.0f;  // Extra penalty for UNK tokens
    
    // Track unique tokens to penalize repetition across beams
    std::unordered_map<size_t, int> global_token_counts;
    
    // First pass: count all tokens across all candidates
    for (size_t i = 0; i < candidates.size(); i++) {
        for (const auto& token : candidates[i].sequence) {
            global_token_counts[token]++;
            // Apply extra penalty for UNK tokens
            if (token == unk_token_id_) {
                candidates[i].score -= unk_penalty * global_token_counts[token];
            }
        }
    }
    
    // Second pass: apply penalties
    for (size_t i = 0; i < candidates.size(); i++) {
        float total_penalty = 0.0f;
        
        // Check for self-repetition within the sequence
        std::unordered_map<size_t, int> local_token_counts;
        for (const auto& token : candidates[i].sequence) {
            local_token_counts[token]++;
            if (local_token_counts[token] > 1) {
                total_penalty += base_penalty * (local_token_counts[token] - 1) * 2.0f;
            }
            
            // Add penalty based on global token frequency
            if (global_token_counts[token] > 1) {
                total_penalty += base_penalty * (global_token_counts[token] - 1);
            }
        }
        
        // Check for overlap with higher-scored candidates
        for (size_t j = 0; j < i; j++) {
            float overlap = calculateOverlap(candidates[i].sequence, candidates[j].sequence);
            total_penalty += base_penalty * overlap * 3.0f;  // Increased overlap penalty
        }
        
        // Apply stronger penalty for first token repetition
        if (i > 0 && !candidates[i].sequence.empty() && !candidates[0].sequence.empty()) {
            if (candidates[i].sequence[0] == candidates[0].sequence[0]) {
                total_penalty += base_penalty * 5.0f;  // Heavy penalty for same first token
            }
        }
        
        // Apply the penalties
        candidates[i].score -= total_penalty;
    }
}

std::vector<float> BeamSearch::calculateScores(const std::vector<float>& logits) {
    // Apply temperature scaling with a more moderate temperature
    const float temperature = 0.8f;  // Less aggressive temperature
    std::vector<float> scores = logits;
    
    float max_score = *std::max_element(scores.begin(), scores.end());
    float sum = 0.0f;
    
    for (float& score : scores) {
        // Completely filter out UNK token by setting its probability to 0
        if (score == logits[unk_token_id_]) {
            score = -std::numeric_limits<float>::infinity();  // Effectively zero probability after softmax
            continue;
        }
        score = std::exp((score - max_score) / temperature);
        sum += score;
    }
    
    // Normalize but prevent division by zero
    if (sum > 1e-6f) {
        for (float& score : scores) {
            score /= sum;
        }
    }
    
    return scores;
}

std::vector<std::pair<float, size_t>> BeamSearch::topKSampling(
    const std::vector<float>& probabilities, size_t k) {
    std::vector<std::pair<float, size_t>> prob_idx;
    prob_idx.reserve(probabilities.size());
    
    // Add random noise to break ties and increase diversity
    std::vector<float> noisy_probs = probabilities;
    for (float& prob : noisy_probs) {
        float noise = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.02f;
        prob = std::max(0.0f, prob + noise);
    }
    
    for (size_t i = 0; i < noisy_probs.size(); i++) {
        prob_idx.push_back({noisy_probs[i], i});
    }
    
    // Sort by probability in descending order
    std::partial_sort(
        prob_idx.begin(),
        prob_idx.begin() + std::min(k, prob_idx.size()),
        prob_idx.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; }
    );
    
    // Return top k pairs
    return std::vector<std::pair<float, size_t>>(
        prob_idx.begin(),
        prob_idx.begin() + std::min(k, prob_idx.size())
    );
}

std::vector<std::pair<float, size_t>> BeamSearch::nucleusSampling(
    const std::vector<float>& probabilities, float p) {
    std::vector<std::pair<float, size_t>> sorted_probs;
    sorted_probs.reserve(probabilities.size());
    
    for (size_t i = 0; i < probabilities.size(); i++) {
        sorted_probs.push_back({probabilities[i], i});
    }
    
    // Sort by probability in descending order
    std::sort(sorted_probs.begin(), sorted_probs.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Find cutoff for cumulative probability >= p
    float cumsum = 0.0f;
    size_t cutoff_idx = sorted_probs.size();
    
    for (size_t i = 0; i < sorted_probs.size(); i++) {
        cumsum += sorted_probs[i].first;
        if (cumsum >= p) {
            cutoff_idx = i + 1;
            break;
        }
    }
    
    return std::vector<std::pair<float, size_t>>(
        sorted_probs.begin(),
        sorted_probs.begin() + cutoff_idx
    );
}

std::vector<BeamSearch::Hypothesis> BeamSearch::search_cuda(
    const std::vector<float>& initial_logits,
    std::function<std::vector<float>(const std::vector<int>&)> next_token_fn,
    size_t max_length,
    int eos_token_id) {
    
    // Initialize sequences and scores
    std::vector<std::vector<int>> sequences(beam_width_);
    Matrix beam_scores(beam_width_, 1);
    
    // Get initial top-k tokens
    std::vector<std::pair<float, int>> initial_candidates;
    for (size_t i = 0; i < initial_logits.size(); i++) {
        initial_candidates.emplace_back(initial_logits[i], i);
    }
    
    // Sort and select top beam_width_ candidates
    std::partial_sort(initial_candidates.begin(),
                     initial_candidates.begin() + beam_width_,
                     initial_candidates.end(),
                     std::greater<std::pair<float, int>>());
    
    // Initialize sequences with top tokens
    for (size_t i = 0; i < beam_width_; i++) {
        sequences[i] = {initial_candidates[i].second};
        beam_scores(i, 0) = initial_candidates[i].first;
    }
    
    // Main search loop
    while (!is_search_complete(sequences)) {
        std::vector<std::vector<float>> next_token_scores;
        std::vector<int> next_tokens;
        
        // Get next token predictions for each sequence
        for (const auto& seq : sequences) {
            auto logits = next_token_fn(seq);
            next_token_scores.push_back(logits);
            next_tokens.push_back(seq.back());
        }
        
        // Convert scores to Matrix
        Matrix next_scores(next_token_scores.size(), next_token_scores[0].size());
        for (size_t i = 0; i < next_token_scores.size(); i++) {
            for (size_t j = 0; j < next_token_scores[i].size(); j++) {
                next_scores(i, j) = next_token_scores[i][j];
            }
        }
        
        // Update beams
        update_beams(sequences, beam_scores, next_scores, next_tokens);
    }
    
    // Return top hypotheses
    return get_best_sequence(sequences, beam_scores);
}

std::vector<BeamSearch::Hypothesis> BeamSearch::search_cpu(
    const std::vector<float>& initial_logits,
    std::function<std::vector<float>(const std::vector<int>&)> next_token_fn,
    size_t max_length,
    int eos_token_id) {
    
    // Initialize sequences and scores
    std::vector<std::vector<int>> sequences(beam_width_);
    Matrix beam_scores(beam_width_, 1);
    
    // Get initial top-k tokens
    std::vector<std::pair<float, int>> initial_candidates;
    for (size_t i = 0; i < initial_logits.size(); i++) {
        initial_candidates.emplace_back(initial_logits[i], i);
    }
    
    // Sort and select top beam_width_ candidates
    std::partial_sort(initial_candidates.begin(),
                     initial_candidates.begin() + beam_width_,
                     initial_candidates.end(),
                     std::greater<std::pair<float, int>>());
    
    // Initialize sequences with top tokens
    for (size_t i = 0; i < beam_width_; i++) {
        sequences[i] = {initial_candidates[i].second};
        beam_scores(i, 0) = initial_candidates[i].first;
    }
    
    // Main search loop
    while (!is_search_complete(sequences)) {
        std::vector<std::vector<float>> next_token_scores;
        std::vector<int> next_tokens;
        
        // Get next token predictions for each sequence
        for (const auto& seq : sequences) {
            auto logits = next_token_fn(seq);
            next_token_scores.push_back(logits);
            next_tokens.push_back(seq.back());
        }
        
        // Convert scores to Matrix
        Matrix next_scores(next_token_scores.size(), next_token_scores[0].size());
        for (size_t i = 0; i < next_token_scores.size(); i++) {
            for (size_t j = 0; j < next_token_scores[i].size(); j++) {
                next_scores(i, j) = next_token_scores[i][j];
            }
        }
        
        // Update beams
        update_beams(sequences, beam_scores, next_scores, next_tokens);
    }
    
    // Return top hypotheses
    return get_best_sequence(sequences, beam_scores);
}

void BeamSearch::process_projected_scores(std::vector<std::vector<int>>& sequences,
                                        Matrix& beam_scores,
                                        const Matrix& projected_scores,
                                        const std::vector<int>& next_tokens) {
    std::vector<std::pair<float, std::pair<size_t, int>>> candidates;
    candidates.reserve(beam_width_ * beam_width_);
    
    // Track token frequencies across all beams with reduced impact
    std::unordered_map<int, float> token_counts;
    for (const auto& seq : sequences) {
        for (int token : seq) {
            token_counts[token] += 0.5f;
        }
    }
    
    // For each beam
    for (size_t i = 0; i < sequences.size(); ++i) {
        if (sequences[i].size() >= max_length_) {
            continue;
        }
        
        // Get top-k candidates directly from projected scores
        std::vector<std::pair<float, int>> beam_candidates;
        for (size_t j = 0; j < projected_scores.cols(); ++j) {
            float score = projected_scores(i, j);
            score /= temperature;
            if (token_counts.count(j)) {
                score -= diversity_strength * token_counts[j];
            }
            beam_candidates.emplace_back(score, j);
        }
        
        // Sort and keep top-k
        std::partial_sort(beam_candidates.begin(),
                         beam_candidates.begin() + std::min(top_k, beam_candidates.size()),
                         beam_candidates.end(),
                         std::greater<>());
        
        // Add candidates to the pool
        for (size_t j = 0; j < std::min(top_k, beam_candidates.size()); ++j) {
            float score = beam_candidates[j].first + beam_scores(i, 0);
            candidates.emplace_back(score, std::make_pair(i, beam_candidates[j].second));
        }
    }
    
    // Sort all candidates
    std::sort(candidates.begin(), candidates.end(),
              std::greater<std::pair<float, std::pair<size_t, int>>>());
              
    // Update sequences and scores based on candidates
    std::vector<std::vector<int>> new_sequences;
    Matrix new_scores(beam_width_, 1);
    
    for (size_t i = 0; i < std::min(beam_width_, candidates.size()); ++i) {
        const auto& candidate = candidates[i];
        std::vector<int> new_seq = sequences[candidate.second.first];
        new_seq.push_back(candidate.second.second);
        new_sequences.push_back(new_seq);
        new_scores(i, 0) = candidate.first;
    }
    
    sequences = std::move(new_sequences);
    beam_scores = std::move(new_scores);
}