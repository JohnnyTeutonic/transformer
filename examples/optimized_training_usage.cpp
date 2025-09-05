/**
 * @file optimized_training_usage.cpp
 * @brief Simple usage example showing how to enable all optimizations
 */

#include "../include/transformer.hpp"
#include "../include/training/training_state_manager.hpp"
#include "../include/config.hpp"
#include <iostream>

int main() {
    std::cout << "=== Optimized Transformer Usage Example ===" << std::endl;
    
    // Create configuration with all optimizations enabled
    TransformerConfig config;
    config.num_layers = 6;
    config.num_heads = 8;
    config.hidden_size = 512;
    config.head_dim = 64;
    config.intermediate_size = 2048;
    config.max_seq_length = 256;
    config.vocab_size = 10000;
    
    // Enable adaptive batching
    config.batch_size = 32;
    config.min_batch_size = 8;
    config.max_batch_size = 128;
    
    // Enable attention optimizations
    config.use_flash_attention = true;
    config.use_fp16 = true;
    config.use_gqa = true;
    config.num_kv_heads = 4;
    
    // Create transformer (fused attention automatically enabled)
    Transformer transformer(config);
    
    // Create training manager with all optimizations
    TrainingStateManager training_manager;
    
    std::cout << "✅ Transformer created with optimizations:" << std::endl;
    std::cout << "  - Fused Attention Kernels: Enabled" << std::endl;
    std::cout << "  - Flash Attention: " << (config.use_flash_attention ? "Enabled" : "Disabled") << std::endl;
    std::cout << "  - Grouped Query Attention: " << (config.use_gqa ? "Enabled" : "Disabled") << std::endl;
    std::cout << "  - FP16: " << (config.use_fp16 ? "Enabled" : "Disabled") << std::endl;
    
    // Demonstrate adaptive batch scheduling
    size_t optimal_batch = training_manager.get_optimal_batch_size(config);
    std::cout << "\n📊 Adaptive Batch Scheduling:" << std::endl;
    std::cout << "  - Initial batch size: " << config.batch_size << std::endl;
    std::cout << "  - Optimal batch size: " << optimal_batch << std::endl;
    std::cout << "  - Range: " << config.min_batch_size << " - " << config.max_batch_size << std::endl;
    
    // Demonstrate gradient compression
    auto* gradient_manager = training_manager.get_gradient_manager();
    std::vector<Matrix> sample_gradients = {
        Matrix(512, 512, 0.01f),   // Attention gradients
        Matrix(512, 2048, 0.005f)  // FFN gradients
    };
    
    auto compressed = gradient_manager->compress_gradients_for_p2p(sample_gradients);
    std::cout << "\n🗜️ Gradient Compression:" << std::endl;
    std::cout << "  - Compression ratio: " << compressed.compression_ratio << "x" << std::endl;
    std::cout << "  - Bandwidth reduction: " << (100.0f * (1.0f - 1.0f/compressed.compression_ratio)) << "%" << std::endl;
    
    // Example training loop structure
    std::cout << "\n🚀 Training Loop Structure:" << std::endl;
    std::cout << "for (epoch = 0; epoch < num_epochs; ++epoch) {" << std::endl;
    std::cout << "    // Get optimal batch size" << std::endl;
    std::cout << "    size_t batch_size = training_manager.get_optimal_batch_size(config);" << std::endl;
    std::cout << "    " << std::endl;
    std::cout << "    for (batch in batches) {" << std::endl;
    std::cout << "        // Forward pass (uses fused attention automatically)" << std::endl;
    std::cout << "        output = transformer.forward(input, target);" << std::endl;
    std::cout << "        " << std::endl;
    std::cout << "        // Backward pass" << std::endl;
    std::cout << "        transformer.backward(grad_output, input, target);" << std::endl;
    std::cout << "        " << std::endl;
    std::cout << "        // P2P gradient sync with compression" << std::endl;
    std::cout << "        compressed = gradient_manager->compress_gradients_for_p2p(gradients);" << std::endl;
    std::cout << "        // ... exchange with peers ..." << std::endl;
    std::cout << "        gradients = gradient_manager->decompress_gradients_from_p2p(received);" << std::endl;
    std::cout << "        " << std::endl;
    std::cout << "        // Update performance metrics" << std::endl;
    std::cout << "        training_manager.update_batch_performance(metrics);" << std::endl;
    std::cout << "    }" << std::endl;
    std::cout << "}" << std::endl;
    
    std::cout << "\n✨ All optimizations ready for use!" << std::endl;
    
    return 0;
}
