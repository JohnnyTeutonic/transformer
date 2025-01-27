#pragma once
#include "tokenizer.hpp"
#include "tiktoken/tiktoken/tiktoken.hpp"
#include "token_constants.hpp"
#include "utils/hash_utils.hpp"  // Already has pair_hash definition

class TiktokenTokenizer : public Tokenizer {
public:
    TiktokenTokenizer();
    ~TiktokenTokenizer() override = default;

    // Debug logging control
    static void set_debug_logging(bool enable);

    // Implement all pure virtual functions
    std::vector<int> encode(const std::string& text) const override;
    std::string decode(const std::vector<int>& tokens) const override;
    void preprocess_text(std::string& text) const override;
    bool is_special_token(int token_id) const override;
    int get_pad_token_id() const override { return tokens::PAD_ID; }
    int get_unk_token_id() const override { return tokens::UNK_ID; }
    int get_bos_token_id() const override { return tokens::BOS_ID; }
    int get_eos_token_id() const override { return tokens::EOS_ID; }
    int get_mask_token_id() const override { return tokens::MASK_ID; }
    size_t vocab_size() const override;
    bool is_initialized() const override { return tiktoken_ != nullptr; }
    void initialize(const std::string& encoding_name = "cl100k_base");
    void initialize() override { initialize("gpt2"); }
    void print_vocabulary_mappings() const override;

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

    // Validation helper functions
    void validate_tokenization();
    void track_bpe_merges(const std::string& text);

    size_t target_vocab_size = 7000;  // Keep the reduced vocabulary size
    static bool debug_logging_;  // Add static debug flag

    // Change from static to instance member
    std::unordered_map<std::pair<std::string, std::string>, int, pair_hash> bpe_merges;

protected:
    void save_vocabulary(std::ostream& os) const override;
}; 