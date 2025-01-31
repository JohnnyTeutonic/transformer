#include "../include/vocab_counter.hpp"
#include <iostream>

int main() {
    try {
        size_t vocab_size = VocabularyCounter::count_vocabulary(
            "data/training_pairs.txt",
            "data/validation_pairs.txt"
        );
        
        std::cout << "Total unique words found: " << vocab_size << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} 