#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <set>

class SubwordTokenizer {
public:
    SubwordTokenizer(size_t vocab_size = 1000, size_t min_freq = 2);
    
    // Training methods
    void train(const std::vector<std::string>& texts);
    void learn_vocabulary(const std::vector<std::string>& texts);
    
    // Tokenization methods
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& tokens) const;
    
    // Vocabulary methods
    size_t vocab_size() const { return token_to_id.size(); }
    void save_vocabulary(const std::string& path) const;
    void load_vocabulary(const std::string& path);
    
    // Special token methods
    int get_pad_token_id() const { return pad_token_id; }
    int get_unk_token_id() const { return unk_token_id; }
    int get_bos_token_id() const { return bos_token_id; }
    int get_eos_token_id() const { return eos_token_id; }
    int get_mask_token_id() const { return mask_token_id; }

private:
    // Vocabulary mappings
    std::unordered_map<std::string, int> token_to_id;
    std::unordered_map<int, std::string> id_to_token;
    
    // BPE merge operations
    std::vector<std::pair<std::string, std::string>> merge_rules;
    
    // Configuration
    size_t target_vocab_size;
    size_t minimum_frequency;
    
    // Special token IDs
    const int pad_token_id = 0;
    const int unk_token_id = 1;
    const int bos_token_id = 2;
    const int eos_token_id = 3;
    const int mask_token_id = 4;
    
    // Helper methods
    std::vector<std::string> split_into_chars(const std::string& text) const;
    std::pair<std::string, std::string> find_most_frequent_pair(
        const std::unordered_map<std::string, size_t>& token_freqs) const;
    void merge_pair(const std::string& pair, std::unordered_map<std::string, size_t>& token_freqs);
    std::vector<std::string> apply_merges(const std::vector<std::string>& chars) const;
}; 