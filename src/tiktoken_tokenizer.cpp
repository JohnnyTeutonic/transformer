#include "../include/tiktoken_tokenizer.hpp"
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>

TiktokenTokenizer::TiktokenTokenizer() = default;

void TiktokenTokenizer::initialize(const std::string& encoding_name) {
    try {
        std::cout << "Initializing GPT-2 tokenizer" << std::endl;
        
        // Print current working directory
        std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;
        
        // Get the executable path and derive the project root
        std::filesystem::path exe_path = std::filesystem::current_path();
        std::filesystem::path project_root = exe_path.parent_path();  // Go up from build dir
        
        // Load vocabulary file using absolute paths
        std::filesystem::path vocab_path = project_root / "scripts" / "tiktoken_data" / "gpt2.vocab.json";
        std::filesystem::path merges_path = project_root / "scripts" / "tiktoken_data" / "gpt2.merges.json";
        
        // Print absolute paths
        std::cout << "Absolute vocab path: " << vocab_path << std::endl;
        std::cout << "Absolute merges path: " << merges_path << std::endl;
        
        if (!std::filesystem::exists(vocab_path)) {
            throw std::runtime_error("GPT-2 vocabulary file not found at: " + vocab_path.string());
        }
        if (!std::filesystem::exists(merges_path)) {
            throw std::runtime_error("GPT-2 merges file not found at: " + merges_path.string());
        }

        // Create a custom encoding using the vocabulary and merges files
        tiktoken_ = std::make_unique<tiktoken::Encoding>();
        
        // Load vocabulary
        std::ifstream vocab_file(vocab_path);
        if (!vocab_file.is_open()) {
            throw std::runtime_error("Failed to open vocabulary file: " + vocab_path.string());
        }

        // Parse JSON vocabulary
        nlohmann::json vocab_json;
        vocab_file >> vocab_json;
        
        // Add tokens from vocabulary
        for (const auto& [token, id] : vocab_json.items()) {
            tiktoken_->add_special_token(token, id);
        }

        // Load merges
        std::ifstream merges_file(merges_path);
        if (!merges_file.is_open()) {
            throw std::runtime_error("Failed to open merges file: " + merges_path.string());
        }

        // Parse JSON merges
        nlohmann::json merges_json;
        merges_file >> merges_json;
        
        // Process merges
        std::vector<std::string> tokens;
        for (const auto& merge_str : merges_json) {
            if (merge_str.is_string()) {
                std::string merge = merge_str.get<std::string>();
                // Split the merge rule
                size_t space_pos = merge.find(' ');
                if (space_pos != std::string::npos) {
                    std::string first = merge.substr(0, space_pos);
                    std::string second = merge.substr(space_pos + 1);
                    // Add both parts as individual tokens if they're not already in vocab
                    tiktoken_->add_special_token(first, tiktoken_->get_vocab_size());
                    tiktoken_->add_special_token(second, tiktoken_->get_vocab_size());
                    // Add the merged token
                    tiktoken_->add_special_token(first + second, tiktoken_->get_vocab_size());
                }
            }
        }

        std::cout << "Base vocabulary size before special tokens: " << tiktoken_->get_vocab_size() << std::endl;
        setup_special_tokens();
        std::cout << "Final vocabulary size after special tokens: " << tiktoken_->get_vocab_size() << std::endl;
        
        // Test encode a simple string
        std::string test_str = "Hello world";
        auto test_tokens = tiktoken_->encode(test_str);
        std::cout << "Test encoding '" << test_str << "': ";
        for (auto token : test_tokens) {
            std::cout << token << " ";
        }
        std::cout << std::endl;
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to initialize tiktoken: " + std::string(e.what()));
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

std::vector<int> TiktokenTokenizer::encode(const std::string& text) const {
    if (!is_initialized()) {
        throw std::runtime_error("Tokenizer not initialized");
    }

    try {
        if (text.empty()) {
            std::cout << "Warning: Attempting to encode empty string" << std::endl;
            return std::vector<int>();
        }

        // Use tiktoken's encode method directly instead of custom tokenization
        auto tokens = tiktoken_->encode(text);
        
        // Add BOS and EOS tokens
        std::vector<int> result;
        result.reserve(tokens.size() + 2);
        result.push_back(tokens::BOS_ID);
        result.insert(result.end(), tokens.begin(), tokens.end());
        result.push_back(tokens::EOS_ID);

        if (result.empty()) {
            std::cout << "Warning: Encoding produced empty tokens for text: '" << text << "'" << std::endl;
        } 
        return result;
    } catch (const std::exception& e) {
        std::cout << "Error encoding text: '" << text << "': " << e.what() << std::endl;
        throw std::runtime_error("Encoding failed: " + std::string(e.what()));
    }
}

std::string TiktokenTokenizer::decode(const std::vector<int>& tokens) const {
    if (!is_initialized()) {
        throw std::runtime_error("Tokenizer not initialized");
    }

    try {
        // Filter out special tokens
        std::vector<int> filtered_tokens;
        filtered_tokens.reserve(tokens.size());
        for (int token : tokens) {
            if (token != tokens::BOS_ID && token != tokens::EOS_ID) {
                filtered_tokens.push_back(token);
            }
        }
        
        // Use tiktoken's decode method
        return tiktoken_->decode(filtered_tokens);
    } catch (const std::exception& e) {
        throw std::runtime_error("Decoding failed: " + std::string(e.what()));
    }
}

size_t TiktokenTokenizer::vocab_size() const {
    if (!is_initialized()) {
        throw std::runtime_error("Tokenizer not initialized");
    }
    return tiktoken_->get_vocab_size();
} 