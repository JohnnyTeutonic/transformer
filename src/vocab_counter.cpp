#include "../include/vocab_counter.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

size_t VocabularyCounter::count_vocabulary(const std::string& training_file, const std::string& validation_file) {
    std::unordered_set<std::string> vocab;
    
    // Process both files
    process_file(training_file, vocab);
    process_file(validation_file, vocab);
    
    std::cout << "Found vocabulary size: " << vocab.size() << std::endl;
    return vocab.size();
}

void VocabularyCounter::process_file(const std::string& filename, std::unordered_set<std::string>& vocab) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        auto tokens = tokenize_line(line);
        for (const auto& token : tokens) {
            if (!token.empty()) {
                vocab.insert(token);
            }
        }
    }
}

std::vector<std::string> VocabularyCounter::tokenize_line(const std::string& line) {
    std::vector<std::string> tokens;
    std::string current_token;
    
    // First split on delimiters |#*
    std::vector<std::string> parts;
    std::stringstream ss(line);
    std::string part;
    
    size_t pos = 0;
    while (pos < line.length()) {
        // Find next delimiter
        size_t next_pos = line.find_first_of("|#*", pos);
        if (next_pos == std::string::npos) {
            parts.push_back(line.substr(pos));
            break;
        }
        parts.push_back(line.substr(pos, next_pos - pos));
        pos = next_pos + 1;
    }
    
    // Then split each part on spaces
    for (const auto& part : parts) {
        std::stringstream part_ss(part);
        std::string word;
        while (part_ss >> word) {
            if (!word.empty()) {
                tokens.push_back(word);
            }
        }
    }
    
    return tokens;
} 