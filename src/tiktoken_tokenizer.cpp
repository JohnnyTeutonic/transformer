#include "../include/tiktoken_tokenizer.hpp"
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <random>
#include <algorithm>
#include <iomanip>

TiktokenTokenizer::TiktokenTokenizer() = default;

void TiktokenTokenizer::initialize(const std::string& encoding_name) {
    try {
        std::cout << "Initializing tokenizer..." << std::endl;
        
        // Initialize with specified encoding
        tiktoken_ = std::make_unique<tiktoken::Encoding>(encoding_name);
        std::cout << "Loaded " << encoding_name << " vocabulary" << std::endl;
        
        // Setup special tokens and mappings
        setup_special_tokens();
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to initialize tokenizer: " + std::string(e.what()));
    }
}

void TiktokenTokenizer::build_vocabulary_from_frequencies() {
    std::vector<std::pair<std::string, size_t>> freq_pairs(token_frequencies_.begin(), token_frequencies_.end());
    
    // Sort by frequency, highest first
    std::sort(freq_pairs.begin(), freq_pairs.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    size_t current_id = 5;  // Start after special tokens
    size_t training_data_tokens = 0;  // Counter for tokens from training data
    size_t gpt2_tokens = 0;  // Counter for GPT2 tokens
    filtered_tokens_.resize(tiktoken_->get_vocab_size(), false);
    
    // Print more detailed frequency analysis
    if (debug_logging_) {
        std::cout << "\nToken frequency analysis:" << std::endl;
        std::cout << "Top 20 most frequent tokens:" << std::endl;
        for (size_t i = 0; i < std::min(size_t(20), freq_pairs.size()); i++) {
            std::cout << freq_pairs[i].first << ": " << freq_pairs[i].second << std::endl;
        }
        
        // Print frequency distribution
        std::cout << "\nFrequency distribution:" << std::endl;
        size_t very_high_freq = 0, high_freq = 0, medium_freq = 0, low_freq = 0;
        for (const auto& [token, freq] : freq_pairs) {
            if (freq > 10000) very_high_freq++;
            else if (freq > 1000) high_freq++;
            else if (freq > 100) medium_freq++;
            else low_freq++;
        }
        std::cout << "Very high frequency (>10000): " << very_high_freq << std::endl;
        std::cout << "High frequency (1000-10000): " << high_freq << std::endl;
        std::cout << "Medium frequency (100-1000): " << medium_freq << std::endl;
        std::cout << "Low frequency (<100): " << low_freq << std::endl;
    }
    
    // First pass: Prioritize complete words and common subwords
    for (const auto& [token, freq] : freq_pairs) {
        if (current_id >= target_vocab_size) break;
        
        std::vector<int> token_ids = tiktoken_->encode(token);
        std::string decoded = tiktoken_->decode(token_ids);
        
        // Prioritize tokens that represent complete words or common subwords
        bool is_complete_word = (decoded.find(" ") != std::string::npos) || 
                               (decoded.length() > 2 && freq >= 10);
        
        if (token_ids.size() == 1 && (is_complete_word || freq >= 5)) {
            int old_id = token_ids[0];
            filtered_tokens_[old_id] = true;
            old_to_new_id_[old_id] = current_id;
            new_to_old_id_[current_id] = old_id;
            current_id++;
            training_data_tokens++;
        }
    }
    
    // Second pass: Fill remaining slots with most frequent GPT2 tokens
    if (current_id < target_vocab_size) {
        for (size_t i = 0; i < tiktoken_->get_vocab_size() && current_id < target_vocab_size; i++) {
            if (!filtered_tokens_[i]) {
                filtered_tokens_[i] = true;
                old_to_new_id_[i] = current_id;
                new_to_old_id_[current_id] = i;
                current_id++;
                gpt2_tokens++;
            }
        }
    }
    
    std::cout << "\nVocabulary construction complete:" << std::endl;
    std::cout << "- Total tokens: " << current_id << std::endl;
    std::cout << "- Training data tokens: " << training_data_tokens << std::endl;
    std::cout << "- GPT2 tokens: " << gpt2_tokens << std::endl;
    std::cout << "- Special tokens: 5" << std::endl;
    std::cout << "- Original GPT2 vocabulary size: " << tiktoken_->get_vocab_size() << std::endl;

    std::cout << "\nBPE merge statistics:" << std::endl;
    std::vector<std::pair<std::pair<std::string, std::string>, int>> merge_stats;
    for (const auto& [merge_pair, count] : bpe_merges) {
        merge_stats.push_back({merge_pair, count});
    }
    
    // Sort by frequency
    std::sort(merge_stats.begin(), merge_stats.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
              
    // Print top merges
    std::cout << "Top 20 most common BPE merges:" << std::endl;
    for (size_t i = 0; i < std::min(size_t(20), merge_stats.size()); i++) {
        const auto& [pair, count] = merge_stats[i];
        std::cout << "'" << pair.first << "' + '" << pair.second 
                 << "' -> " << count << " occurrences" << std::endl;
    }
}

void TiktokenTokenizer::setup_special_tokens() {
    if (!is_initialized()) {
        throw std::runtime_error("Tokenizer not initialized");
    }

    // Add our special tokens to tiktoken's vocabulary in the same order as defined in token_constants.hpp
    tiktoken_->add_special_token("<pad>", tokens::PAD_ID);    // ID 0
    tiktoken_->add_special_token("<unk>", tokens::UNK_ID);    // ID 1
    tiktoken_->add_special_token("<s>", tokens::BOS_ID);      // ID 2
    tiktoken_->add_special_token("</s>", tokens::EOS_ID);     // ID 3
    tiktoken_->add_special_token("<mask>", tokens::MASK_ID);  // ID 4
}

int TiktokenTokenizer::convert_to_new_id(int old_id) const {
    if (old_id < 5) return old_id;  // Special tokens keep their IDs
    auto it = old_to_new_id_.find(old_id);
    return it != old_to_new_id_.end() ? it->second : tokens::UNK_ID;
}

int TiktokenTokenizer::convert_to_old_id(int new_id) const {
    if (new_id < 5) return new_id;  // Special tokens keep their IDs
    auto it = new_to_old_id_.find(new_id);
    return it != new_to_old_id_.end() ? it->second : tokens::UNK_ID;
}

// Add debug flag as a static member
bool TiktokenTokenizer::debug_logging_ = false;

// Add static method to control debug logging
void TiktokenTokenizer::set_debug_logging(bool enable) {
    debug_logging_ = enable;
}

std::vector<int> TiktokenTokenizer::encode(const std::string& text) const {
    if (!is_initialized()) {
        throw std::runtime_error("Tokenizer not initialized");
    }
    try {
        return tiktoken_->encode(text);
    } catch (const std::exception& e) {
        throw std::runtime_error("Encoding failed: " + std::string(e.what()));
    }
}

std::string TiktokenTokenizer::decode(const std::vector<int>& tokens) const {
    if (!is_initialized()) {
        throw std::runtime_error("Tokenizer not initialized");
    }
    try {
        return tiktoken_->decode(tokens);
    } catch (const std::exception& e) {
        throw std::runtime_error("Decoding failed: " + std::string(e.what()));
    }
}

void TiktokenTokenizer::preprocess_text(std::string& text) const {
    // Apply any necessary text preprocessing
    // For tiktoken, we might not need much preprocessing as it handles most cases
}

bool TiktokenTokenizer::is_special_token(int token_id) const {
    return token_id == tokens::PAD_ID ||
           token_id == tokens::UNK_ID ||
           token_id == tokens::BOS_ID ||
           token_id == tokens::EOS_ID ||
           token_id == tokens::MASK_ID;
}

size_t TiktokenTokenizer::vocab_size() const {
    if (!is_initialized()) {
        throw std::runtime_error("Tokenizer not initialized");
    }
    return tiktoken_->get_vocab_size();
}

void TiktokenTokenizer::print_vocabulary_mappings() const {
    if (!is_initialized()) {
        throw std::runtime_error("Tokenizer not initialized");
    }
    
    std::cout << "\nVocabulary mappings:" << std::endl;
    std::cout << "- Special tokens:" << std::endl;
    std::cout << "  PAD: " << tokens::PAD_ID << std::endl;
    std::cout << "  UNK: " << tokens::UNK_ID << std::endl;
    std::cout << "  BOS: " << tokens::BOS_ID << std::endl;
    std::cout << "  EOS: " << tokens::EOS_ID << std::endl;
    std::cout << "  MASK: " << tokens::MASK_ID << std::endl;
    
    std::cout << "- Vocabulary size: " << vocab_size() << std::endl;
}

void TiktokenTokenizer::save_vocabulary(std::ostream& os) const {
    if (!is_initialized()) {
        throw std::runtime_error("Tokenizer not initialized");
    }
    
    // Save vocabulary size
    size_t vocab_size = tiktoken_->get_vocab_size();
    os.write(reinterpret_cast<const char*>(&vocab_size), sizeof(vocab_size));
    
    // Could save more vocabulary-specific data here if needed
}

void TiktokenTokenizer::validate_tokenization() {
    std::vector<std::string> test_cases = {
        "The cat",
        "I want to",
        "Hello world",
        "This is a test"
    };
    
    std::cout << "\nValidating tokenization:\n";
    for (const auto& text : test_cases) {
        auto tokens = encode(text);
        std::string reconstructed = decode(tokens);
        std::cout << "Original: '" << text << "'\n";
        std::cout << "Reconstructed: '" << reconstructed << "'\n";
        std::cout << "Tokens: ";
        for (int token : tokens) {
            std::cout << token << " ";
        }
        std::cout << "\n\n";
    }
}

void TiktokenTokenizer::track_bpe_merges(const std::string& text) {
    std::vector<int> tokens = tiktoken_->encode(text);
    std::string prev_token;
    
    for (int token_id : tokens) {
        std::string token = tiktoken_->decode({token_id});
        if (!prev_token.empty()) {
            bpe_merges[{prev_token, token}]++;
        }
        prev_token = token;
    }
} 