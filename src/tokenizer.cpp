#include "../include/tokenizer.hpp"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>

// Define static member
const std::unordered_map<char, std::string> Tokenizer::SPECIAL_CHAR_MAP = {
    {'\n', "<newline>"}, {'\t', "<tab>"}, {'.', "<period>"},
    {'!', "<exclamation>"}, {'?', "<question>"}, {',', "<comma>"}
};

Tokenizer::Tokenizer() {
    tokenizer_ = std::make_unique<SubwordTokenizer>();
}

void Tokenizer::initialize(const std::string& encoding_name) {
    if (encoding_name == "custom") {
        // For custom encoding, we'll train on data later
        std::cout << "Using custom subword tokenizer. Call train_on_data() to train.\n";
    } else {
        // Try to load pre-trained vocabulary
        try {
            tokenizer_->load_vocabulary(encoding_name);
            std::cout << "Loaded vocabulary from " << encoding_name << "\n";
        } catch (const std::exception& e) {
            std::cerr << "Failed to load vocabulary: " << e.what() << "\n";
            throw;
        }
    }
}

void Tokenizer::train_on_data(const std::vector<std::string>& texts) {
    if (!tokenizer_) {
        tokenizer_ = std::make_unique<SubwordTokenizer>();
    }
    tokenizer_->train(texts);
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    if (!tokenizer_) {
        throw std::runtime_error("Tokenizer not initialized");
    }
    return tokenizer_->encode(text);
}

std::string Tokenizer::decode(const std::vector<int>& tokens) const {
    if (!tokenizer_) {
        throw std::runtime_error("Tokenizer not initialized");
    }
    return tokenizer_->decode(tokens);
}

void Tokenizer::preprocess_text(std::string& text) const {
    // Replace special characters with their token representations
    for (const auto& [ch, token] : SPECIAL_CHAR_MAP) {
        size_t pos = 0;
        while ((pos = text.find(ch, pos)) != std::string::npos) {
            text.replace(pos, 1, token);
            pos += token.length();
        }
    }
}

bool Tokenizer::is_special_token(int token_id) const {
    return token_id <= tokenizer_->get_mask_token_id();
}

void Tokenizer::print_vocabulary_mappings() const {
    if (!tokenizer_) {
        std::cout << "Tokenizer not initialized\n";
        return;
    }
    
    std::cout << "Vocabulary size: " << vocab_size() << "\n";
    std::cout << "Special tokens:\n";
    std::cout << "  PAD: " << get_pad_token_id() << "\n";
    std::cout << "  UNK: " << get_unk_token_id() << "\n";
    std::cout << "  BOS: " << get_bos_token_id() << "\n";
    std::cout << "  EOS: " << get_eos_token_id() << "\n";
    std::cout << "  MASK: " << get_mask_token_id() << "\n";
}

void Tokenizer::save(std::ostream& os) const {
    if (!tokenizer_) {
        throw std::runtime_error("Tokenizer not initialized");
    }
    std::stringstream ss;
    ss << os.rdbuf();
    tokenizer_->save_vocabulary(ss.str());
}

std::unique_ptr<Tokenizer> Tokenizer::load(std::istream& is) {
    auto tokenizer = std::make_unique<Tokenizer>();
    std::stringstream ss;
    ss << is.rdbuf();
    tokenizer->tokenizer_->load_vocabulary(ss.str());
    return tokenizer;
}

std::vector<std::string> Tokenizer::get_vocabulary_vector() const {
    // This would need to be implemented based on your needs
    return std::vector<std::string>();
}