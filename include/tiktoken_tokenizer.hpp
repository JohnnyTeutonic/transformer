#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

class TiktokenTokenizer {
private:
    std::unordered_map<std::string, int> word_to_id_;
    std::unordered_map<int, std::string> id_to_word_;
    bool initialized_ = false;
    static constexpr int UNK_ID = 0;
    static constexpr const char* UNK_TOKEN = "<unk>";
    static constexpr int SEP_ID = 1;  // Add separator token ID
    static constexpr const char* SEP_TOKEN = "|";  // Use | as separator token
    
public:
    // Core tokenization methods
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& tokens) const;
    
    // Text preprocessing
    static std::string preprocess_text(const std::string& text);
    
    // Vocabulary management
    void build_vocabulary_from_file(const std::string& filepath);
    size_t vocab_size() const { return word_to_id_.size(); }
    bool is_initialized() const { return initialized_; }
    
    // Token type checking
    bool is_verb(const std::string& token) const;
    bool is_adjective(const std::string& token) const;
    bool is_noun(const std::string& token) const;
    bool is_special_token(int token_id) const;
    
    // Special token getters
    int get_unk_token_id() const { return UNK_ID; }
    int get_sep_token_id() const { return SEP_ID; }
    
    // Debugging helpers
    void print_vocabulary() const;
    
private:
    static std::vector<std::string> split_into_words(const std::string& text);
}; 