#pragma once
#include "base_tokenizer.hpp"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include "token_constants.hpp"

// Add before the class declaration
std::vector<std::pair<std::string, std::string>> extract_phrase_pairs(const std::string& filepath);

class TiktokenTokenizer : public BaseTokenizer {
private:
    // Forward declare the implementation struct
    struct Impl;
    std::unique_ptr<Impl> pimpl_;

    bool initialized_ = false;
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<int, std::string> id_to_token_;
    size_t vocab_size_ = tokens::NUM_SPECIAL_TOKENS;
    std::unordered_map<std::string, size_t> token_frequencies_;

public:
    explicit TiktokenTokenizer(const std::string& encoding_name = "gpt2");
    ~TiktokenTokenizer() override;

    // Core tokenization methods
    std::vector<int> encode(const std::string& text) const override;
    std::string decode(const std::vector<int>& tokens) const override;
    size_t vocab_size() const override;
    void initialize(const std::string& encoding_name) override;
    bool is_initialized() const override;

    // Special token getters
    int get_pad_token_id() const override { return tokens::PAD_ID; }
    int get_unk_token_id() const override { return tokens::UNK_ID; }
    int get_bos_token_id() const override { return tokens::BOS_ID; }
    int get_eos_token_id() const override { return tokens::EOS_ID; }
    int get_mask_token_id() const override { return tokens::MASK_ID; }
    int get_sep_token_id() const { return tokens::SEP_ID; }

    // Vocabulary management
    void load_vocabulary(const std::string& vocab_file);
    void save_vocabulary(const std::string& vocab_file) const;
    void set_vocab_size(size_t size) override;
    void build_vocabulary_from_data(const std::string& data_file, size_t min_freq);
    void build_vocabulary_from_frequencies();
    
    // Additional methods
    std::vector<int> encode(const std::string& text, bool add_special_tokens) const;
    std::string decode(const std::vector<int>& tokens, bool skip_special_tokens) const;
    void print_vocabulary_mappings() const;
    void print_vocabulary_stats() const;
    void save(std::ostream& os) const;
    std::vector<std::string> get_vocabulary_vector() const;

    const std::unordered_map<std::string, int>& get_vocab() const {
        return token_to_id_;
    }

    bool is_verb(const std::string& token) const {
        return verb_tokens_.find(token) != verb_tokens_.end();
    }

    bool is_adjective(const std::string& token) const {
        return adjective_tokens_.find(token) != adjective_tokens_.end();
    }

    bool is_noun(const std::string& token) const {
        return noun_tokens_.find(token) != noun_tokens_.end();
    }

    // Add these method declarations in the public section
    void learn_bpe(const std::string& training_file, size_t target_vocab_size, size_t min_freq);
    void apply_bpe(std::vector<int>& tokens) const;

protected:
    // Token categories
    std::unordered_set<std::string> verb_tokens_;
    std::unordered_set<std::string> adjective_tokens_;
    std::unordered_set<std::string> noun_tokens_;
    std::unordered_set<std::string> determiner_tokens_;
    void initialize_token_categories();

private:
    // Private helpers
    void setup_special_tokens();
    std::vector<int> tokenize_text(const std::string& text) const;
    std::string decode_token(int token_id) const;
    static std::string preprocess_text(const std::string& text);
    bool is_special_token(int token_id) const {
        return token_id >= 0 && token_id < tokens::NUM_SPECIAL_TOKENS;
    }
}; 