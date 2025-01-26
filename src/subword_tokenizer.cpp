#include "../include/subword_tokenizer.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

SubwordTokenizer::SubwordTokenizer(unsigned long vocab_size, unsigned long min_freq)
    : target_vocab_size(vocab_size), minimum_frequency(min_freq) {
    // Initialize special tokens
    token_to_id["<pad>"] = pad_token_id;
    token_to_id["<unk>"] = unk_token_id;
    token_to_id["<s>"] = bos_token_id;
    token_to_id["</s>"] = eos_token_id;
    token_to_id["<mask>"] = mask_token_id;

    // Initialize reverse mapping
    id_to_token[pad_token_id] = "<pad>";
    id_to_token[unk_token_id] = "<unk>";
    id_to_token[bos_token_id] = "<s>";
    id_to_token[eos_token_id] = "</s>";
    id_to_token[mask_token_id] = "<mask>";
}

void SubwordTokenizer::train(const std::vector<std::string>& texts) {
    learn_vocabulary(texts);
}

void SubwordTokenizer::learn_vocabulary(const std::vector<std::string>& texts) {
    // Count initial character frequencies
    std::unordered_map<std::string, size_t> token_freqs;
    
    for (const auto& text : texts) {
        auto chars = split_into_chars(text);
        for (const auto& c : chars) {
            token_freqs[c]++;
        }
    }

    // Learn BPE merge rules until we reach target vocabulary size
    while (token_to_id.size() < target_vocab_size) {
        auto most_freq = find_most_frequent_pair(token_freqs);
        if (most_freq.first.empty() || token_freqs[most_freq.first] < minimum_frequency) {
            break;
        }
        
        merge_pair(most_freq.first, token_freqs);
        merge_rules.push_back({most_freq.first, most_freq.second});
        
        // Add to vocabulary
        if (token_to_id.find(most_freq.second) == token_to_id.end()) {
            int new_id = token_to_id.size();
            token_to_id[most_freq.second] = new_id;
            id_to_token[new_id] = most_freq.second;
        }
    }
}

std::vector<int> SubwordTokenizer::encode(const std::string& text) const {
    std::vector<int> result;
    auto chars = split_into_chars(text);
    auto tokens = apply_merges(chars);
    
    for (const auto& token : tokens) {
        auto it = token_to_id.find(token);
        if (it != token_to_id.end()) {
            result.push_back(it->second);
        } else {
            result.push_back(unk_token_id);
        }
    }
    
    return result;
}

std::string SubwordTokenizer::decode(const std::vector<int>& tokens) const {
    std::string result;
    for (int token : tokens) {
        auto it = id_to_token.find(token);
        if (it != id_to_token.end()) {
            result += it->second;
        } else {
            result += id_to_token.at(unk_token_id);
        }
    }
    return result;
}

void SubwordTokenizer::save_vocabulary(const std::string& path) const {
    json j;
    j["token_to_id"] = token_to_id;
    j["merge_rules"] = merge_rules;
    
    std::ofstream file(path);
    file << j.dump(2);
}

void SubwordTokenizer::load_vocabulary(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open vocabulary file: " + path);
    }
    
    json j;
    file >> j;
    
    token_to_id = j["token_to_id"].get<std::unordered_map<std::string, int>>();
    merge_rules = j["merge_rules"].get<std::vector<std::pair<std::string, std::string>>>();
    
    // Rebuild id_to_token mapping
    id_to_token.clear();
    for (const auto& [token, id] : token_to_id) {
        id_to_token[id] = token;
    }
}

std::vector<std::string> SubwordTokenizer::split_into_chars(const std::string& text) const {
    std::vector<std::string> chars;
    for (char c : text) {
        chars.push_back(std::string(1, c));
    }
    return chars;
}

std::pair<std::string, std::string> SubwordTokenizer::find_most_frequent_pair(
    const std::unordered_map<std::string, size_t>& token_freqs) const {
    std::string best_pair;
    std::string merged;
    size_t max_freq = 0;
    
    for (const auto& [token, freq] : token_freqs) {
        if (token.length() < 2) continue;
        
        size_t space_pos = token.find(' ');
        if (space_pos != std::string::npos && space_pos < token.length() - 1) {
            std::string pair = token;
            std::string merge_result = token.substr(0, space_pos) + token.substr(space_pos + 1);
            
            if (freq > max_freq) {
                max_freq = freq;
                best_pair = pair;
                merged = merge_result;
            }
        }
    }
    
    return {best_pair, merged};
}

void SubwordTokenizer::merge_pair(const std::string& pair, 
                                std::unordered_map<std::string, size_t>& token_freqs) {
    size_t space_pos = pair.find(' ');
    if (space_pos == std::string::npos) return;
    
    std::string merged = pair.substr(0, space_pos) + pair.substr(space_pos + 1);
    token_freqs[merged] = token_freqs[pair];
    token_freqs.erase(pair);
}

std::vector<std::string> SubwordTokenizer::apply_merges(const std::vector<std::string>& chars) const {
    std::vector<std::string> tokens = chars;
    
    for (const auto& [pair, merged] : merge_rules) {
        for (size_t i = 0; i < tokens.size() - 1; ++i) {
            if (tokens[i] + " " + tokens[i + 1] == pair) {
                tokens[i] = merged;
                tokens.erase(tokens.begin() + i + 1);
                --i;
            }
        }
    }
    
    return tokens;
} 