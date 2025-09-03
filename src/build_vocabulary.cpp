#include "../include/tiktoken_tokenizer.hpp"
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <training_data_file> <output_vocab_file>" << std::endl;
        return 1;
    }

    std::string training_file = argv[1];
    std::string vocab_file = argv[2];

    try {
        // Initialize tokenizer
        TiktokenTokenizer tokenizer;
        
        // Build vocabulary from training data
        std::cout << "Building vocabulary from " << training_file << "..." << std::endl;
        tokenizer.build_vocabulary_from_data(training_file, 2); // min frequency of 2
        
        // Learn BPE merges
        std::cout << "Learning BPE merges..." << std::endl;
        tokenizer.learn_bpe(training_file, 50000, 2); // target vocab size of 50k
        
        // Save vocabulary
        std::cout << "Saving vocabulary to " << vocab_file << "..." << std::endl;
        tokenizer.save_vocabulary(vocab_file);
        
        std::cout << "Done! Final vocabulary size: " << tokenizer.vocab_size() << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
} 