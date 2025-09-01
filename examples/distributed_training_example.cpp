#include "../include/distributed_transformer.hpp"
#include "../include/config.hpp"
#include "../include/tiktoken_tokenizer.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <algorithm>

/**
 * @brief Example demonstrating distributed transformer training
 * 
 * Compile with MPI support:
 * mkdir -p build && cd build
 * cmake -DCMAKE_BUILD_TYPE=Release ..
 * make -j$(nproc)
 * 
 * Run with MPI:
 * mpirun -np 4 ./distributed_training_example
 * 
 * Or with SLURM:
 * srun -N 2 -n 8 --gres=gpu:4 ./distributed_training_example
 */

int main(int argc, char* argv[]) {
    try {
        // Create transformer configuration
        TransformerConfig config;
        config.vocab_size = 50257;        // GPT-2 vocab size
        config.hidden_size = 768;         // Model dimension
        config.num_layers = 12;           // Number of transformer layers
        config.num_heads = 12;            // Number of attention heads
        config.max_seq_length = 1024;     // Maximum sequence length
        config.dropout_rate = 0.1f;       // Dropout rate
        config.initial_lr = 0.0001f;       // Learning rate
        
        // Initialize tokenizer
        auto tokenizer = std::make_shared<TiktokenTokenizer>();
        try {
            tokenizer->build_vocabulary_from_file("tiktoken_data/cl100k_base.vocab");
        } catch (const std::exception& e) {
            std::cerr << "Failed to load tokenizer: " << e.what() << std::endl;
            std::cerr << "Note: This is just a demo - tokenizer file not required for MPI integration test" << std::endl;
            // Continue anyway for MPI testing
        }
        
        // Create distributed transformer (MPI_Init called automatically)
        auto distributed_model = create_distributed_transformer(config, tokenizer, &argc, &argv);
        
        // Print distributed info on rank 0
        if (distributed_model->get_world_rank() == 0) {
            std::cout << "=== Distributed Transformer Training Example ===" << std::endl;
            std::cout << "World size: " << distributed_model->get_world_size() << std::endl;
            std::cout << "Local rank: " << distributed_model->get_local_rank() << std::endl;
            std::cout << "Distributed ready: " << (distributed_model->is_distributed_ready() ? "Yes" : "No") << std::endl;
        }
        
        // Prepare training data (example sentences)
        std::vector<std::string> training_texts = {
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is revolutionizing artificial intelligence.",
            "Distributed training accelerates model development significantly.",
            "Neural networks learn complex patterns from data.",
            "Transformers have become the dominant architecture in NLP.",
            "Attention mechanisms allow models to focus on relevant information.",
            "Large language models demonstrate emergent capabilities.",
            "Parallel processing enables efficient model training.",
            "Deep learning requires substantial computational resources.",
            "Gradient synchronization ensures consistent model updates."
        };
        
        // Tokenize training data
        std::vector<std::vector<int>> input_tokens;
        std::vector<std::vector<int>> target_tokens;
        
        for (const auto& text : training_texts) {
            auto tokens = tokenizer->encode(text);
            if (tokens.size() > 1) {
                // Input: all tokens except last
                input_tokens.push_back(std::vector<int>(tokens.begin(), tokens.end() - 1));
                // Target: all tokens except first
                target_tokens.push_back(std::vector<int>(tokens.begin() + 1, tokens.end()));
            }
        }
        
        if (distributed_model->get_world_rank() == 0) {
            std::cout << "\nTraining data prepared:" << std::endl;
            std::cout << "  Total sequences: " << input_tokens.size() << std::endl;
            std::cout << "  Average sequence length: " 
                      << (input_tokens.empty() ? 0 : 
                          std::accumulate(input_tokens.begin(), input_tokens.end(), 0,
                                        [](int sum, const auto& seq) { return sum + seq.size(); }) / input_tokens.size())
                      << std::endl;
        }
        
        // Training parameters
        const size_t num_epochs = 5;
        const float learning_rate = 0.0001f;
        
        if (distributed_model->get_world_rank() == 0) {
            std::cout << "\nStarting distributed training..." << std::endl;
        }
        
        // Perform distributed training
        auto start_time = std::chrono::high_resolution_clock::now();
        
        distributed_model->train(input_tokens, target_tokens, num_epochs, learning_rate);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        if (distributed_model->get_world_rank() == 0) {
            std::cout << "\nTraining completed!" << std::endl;
            std::cout << "Total time: " << duration.count() << "ms" << std::endl;
            std::cout << "Average sync time: " << distributed_model->get_average_sync_time() * 1000.0 << "ms" << std::endl;
            std::cout << "Total synchronizations: " << distributed_model->get_sync_count() << std::endl;
        }
        
        // Test generation on rank 0
        if (distributed_model->get_world_rank() == 0) {
            std::cout << "\nTesting text generation..." << std::endl;
            
            std::string prompt = "The future of artificial intelligence";
            auto prompt_tokens = tokenizer->encode(prompt);
            
            auto generated_tokens = distributed_model->generate(prompt_tokens, 50, 0.8f);
            auto generated_text = tokenizer->decode(generated_tokens);
            
            std::cout << "Prompt: " << prompt << std::endl;
            std::cout << "Generated: " << generated_text << std::endl;
        }
        
        // Save model (only rank 0 saves)
        std::string model_path = "distributed_model.bin";
        distributed_model->save_model(model_path);
        
        if (distributed_model->get_world_rank() == 0) {
            std::cout << "\nModel saved to: " << model_path << std::endl;
            std::cout << "=== Training Complete ===" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
