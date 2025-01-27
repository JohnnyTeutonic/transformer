#include "../include/tokenizer.hpp"
#include "../include/tiktoken_tokenizer.hpp"
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <regex>

// Define the special character map
const std::unordered_map<char, std::string> Tokenizer::SPECIAL_CHAR_MAP = {
    {'\n', "<newline>"},    {'\t', "<tab>"},     {'.', "<period>"},
    {'!', "<exclamation>"}, {'?', "<question>"}, {',', "<comma>"}};

void Tokenizer::save(std::ostream& os) const {
    try {
        uint32_t version = 1;
        os.write(reinterpret_cast<const char*>(&version), sizeof(version));
        save_vocabulary(os);
        if (!os.good()) {
            throw std::runtime_error("Failed to save tokenizer");
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Error saving tokenizer: " + std::string(e.what()));
    }
}

std::unique_ptr<Tokenizer> Tokenizer::load(std::istream& is) {
    try {
        // Create TiktokenTokenizer as a Tokenizer pointer
        std::unique_ptr<Tokenizer> tokenizer = std::make_unique<TiktokenTokenizer>();

        uint32_t version;
        is.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (version != 1) {
            throw std::runtime_error("Unsupported tokenizer version");
        }

        // Initialize the tokenizer
        tokenizer->initialize();

        if (!is.good()) {
            throw std::runtime_error("Failed to load tokenizer");
        }

        return tokenizer;
    } catch (const std::exception& e) {
        throw std::runtime_error("Error loading tokenizer: " + std::string(e.what()));
    }
}

void Tokenizer::save_vocabulary(std::ostream& os) const {
    throw std::runtime_error("save_vocabulary not implemented in base class");
}

void Tokenizer::initialize() {
    throw std::runtime_error("initialize not implemented in base class");
}

bool Tokenizer::is_initialized() const {
    return false;  // Base class is never initialized
}

void Tokenizer::print_vocabulary_mappings() const {
    throw std::runtime_error("print_vocabulary_mappings not implemented in base class");
}

bool Tokenizer::is_noun(const std::string& token) const {
    // Simple heuristic: consider capitalized words or special tokens as nouns
    return !token.empty() && (
        std::isupper(token[0]) ||
        token.find("<") == 0  // Special tokens
    );
}

bool Tokenizer::is_adjective(const std::string& token) const {
    // Common adjectives
    static const std::unordered_set<std::string> adjectives = {
        "big", "small", "large", "tiny", "huge", "little",
        "good", "bad", "great", "awful", "terrible", "wonderful",
        "beautiful", "ugly", "pretty", "handsome",
        "old", "new", "young", "ancient", "modern",
        "happy", "sad", "angry", "excited", "nervous",
        "red", "blue", "green", "yellow", "black", "white",
        "hot", "cold", "warm", "cool",
        "fast", "slow", "quick", "rapid",
        "hard", "soft", "rough", "smooth",
        "bright", "dark", "dim", "shiny",
        "loud", "quiet", "noisy", "silent",
        "clean", "dirty", "neat", "messy",
        "rich", "poor", "wealthy", "expensive",
        "strong", "weak", "powerful", "feeble",
        "smart", "clever", "intelligent", "wise",
        "brave", "cowardly", "fearless", "timid",
        "kind", "mean", "gentle", "cruel",
        "tall", "short", "high", "low",
        "wide", "narrow", "broad", "thin",
        "deep", "shallow", "thick", "slim"
    };
    
    // Convert token to lowercase for comparison
    std::string lower_token = token;
    std::transform(lower_token.begin(), lower_token.end(), lower_token.begin(), ::tolower);
    
    return adjectives.find(lower_token) != adjectives.end();
}

bool Tokenizer::is_determiner(const std::string& token) const {
    // Common determiners
    static const std::unordered_set<std::string> determiners = {
        "the", "a", "an",                     // Articles
        "this", "that", "these", "those",     // Demonstratives
        "my", "your", "his", "her", "its",    // Possessives
        "our", "their",
        "any", "many", "much", "few",         // Quantifiers
        "several", "some", "all", "both",
        "each", "every", "either", "neither",
        "no", "other", "another"
    };
    
    // Convert token to lowercase for comparison
    std::string lower_token = token;
    std::transform(lower_token.begin(), lower_token.end(), lower_token.begin(), ::tolower);
    
    return determiners.find(lower_token) != determiners.end();
}