#include "../include/tiktoken_tokenizer.hpp"
#include <iostream>
#include <vector>
#include <string>

int main() {
    TiktokenTokenizer tokenizer;
    
    // Build vocabulary from training data
    std::cout << "Building vocabulary from training data..." << std::endl;
    tokenizer.build_vocabulary_from_file("../data/training_pairs.txt");
    
    // Print the vocabulary
    std::cout << "\nVocabulary:" << std::endl;
    tokenizer.print_vocabulary();
    
    // Test some example inputs
    std::vector<std::string> test_inputs = {
        "Food scientists test in the",
        "Buskers perform in the",
        "Synthetic evolution directors guide in the",
        "Project coordinators work in the",
        "Proteomics researchers study in the",
        "Specialists maintain in the",
        "Advisors consult in the",
        "Firefighters work at the",
        "Incident responders work in the",
        "UI designers create in the"
    };
    
    std::cout << "\nTesting tokenization:" << std::endl;
    for (const auto& input : test_inputs) {
        std::cout << "\nInput: '" << input << "'" << std::endl;
        
        // Encode
        auto tokens = tokenizer.encode(input);
        std::cout << "Token IDs: ";
        for (int id : tokens) {
            std::cout << id << " ";
        }
        std::cout << std::endl;
        
        // Decode
        std::string decoded = tokenizer.decode(tokens);
        std::cout << "Decoded: '" << decoded << "'" << std::endl;
    }
    
    return 0;
} 