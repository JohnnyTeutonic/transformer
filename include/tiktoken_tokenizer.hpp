#pragma once
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include "tiktoken/tiktoken/tiktoken.hpp"
#include "token_constants.hpp"

class TiktokenTokenizer {
public:
    TiktokenTokenizer();
    ~TiktokenTokenizer() = default;

    // Core tokenization methods
    std::vector<int> encode(const std::string& text, bool add_special_tokens = true) const;
    std::string decode(const std::vector<int>& token_ids, bool skip_special_tokens = true) const;

    // Special token getters - using our constants
    int get_pad_token_id() const { return tokens::PAD_ID; }
    int get_unk_token_id() const { return tokens::UNK_ID; }
    int get_bos_token_id() const { return tokens::BOS_ID; }
    int get_eos_token_id() const { return tokens::EOS_ID; }
    int get_mask_token_id() const { return tokens::MASK_ID; }
    int get_sep_token_id() const { return tokens::SEP_ID; }

    // Special token strings - using literal to match tokens::SEP_TOKEN
    static constexpr const char* SEP_TOKEN = "|";

    // Vocabulary size
    size_t vocab_size() const { return vocab_size_; }

    // Initialize with model type (with default encoding)
    void initialize(const std::string& encoding_name = "gpt2");

    bool is_initialized() const { return is_initialized_; }

    void set_vocab_size(size_t size) {
        target_vocab_size = size;
    }

    static void set_debug_logging(bool enable);  // Add static method to control logging

protected:
    // Helper functions for tokenization
    std::vector<int> tokenize_text(const std::string& text) const;
    std::string decode_token(int token_id) const;

private:
    std::unique_ptr<tiktoken::Encoding> tiktoken_;
    std::vector<bool> filtered_tokens_;  // Tracks which tokens we keep
    std::unordered_map<int, int> old_to_new_id_;  // Maps original token IDs to our new consecutive IDs
    std::unordered_map<int, int> new_to_old_id_;  // Maps our new consecutive IDs back to original token IDs
    std::unordered_map<std::string, size_t> token_frequencies_;  // Track token frequencies for vocabulary selection
    
    // Map between our special token IDs and tiktoken's vocabulary
    void setup_special_tokens();
    
    // Helper to convert between old and new token IDs
    int convert_to_new_id(int old_id) const;
    int convert_to_old_id(int new_id) const;

    // Helper to build frequency-based vocabulary
    void build_vocabulary_from_frequencies();

    // New member variables for custom tokenizer
    std::unordered_map<std::string, int> token_to_id_;  // Maps tokens to their IDs
    std::unordered_map<int, std::string> id_to_token_;  // Maps IDs back to tokens
    bool is_initialized_ = false;
    size_t target_vocab_size = 2500;  // Default vocabulary size
    static bool debug_logging_;  // Add static debug flag
    size_t vocab_size_;  // Add vocab_size_ member variable
}; 