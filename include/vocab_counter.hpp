#pragma once
#include <string>
#include <unordered_set>
#include <vector>

class VocabularyCounter {
public:
    // Count unique tokens from training and validation files
    static size_t count_vocabulary(const std::string& training_file, const std::string& validation_file);

private:
    // Helper to process a single file and add tokens to the set
    static void process_file(const std::string& filename, std::unordered_set<std::string>& vocab);
    
    // Helper to tokenize a line based on delimiters
    static std::vector<std::string> tokenize_line(const std::string& line);
};