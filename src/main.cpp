#include "../include/attention.hpp"
#ifdef CUDA_AVAILABLE
#include "../include/cuda/cuda_init.cuh"
#endif
#include "../include/lm_head.hpp"
#include "../include/logger.hpp"
#include "../include/model_saver.hpp"
#include "../include/optimizer/sam.hpp"
#include "../include/quantization.hpp"
#include "../include/tokenizer.hpp"
#include "../include/transformer.hpp"
#include "../include/utils/tensor_cache.hpp"
#include "../include/vocabulary.hpp"
#include "../include/matrix.hpp"
#include "../include/preprocessing.hpp"
#include "../include/utils.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <sstream>

// Add necessary forward declarations and structures
class Tokenizer;
std::unique_ptr<Tokenizer> tokenizer;

// Configuration constants
const float INITIAL_LEARNING_RATE = 0.001f;
const float MIN_LEARNING_RATE = 1e-6f;
const float MAX_LEARNING_RATE = 0.1f;
const float GRADIENT_CLIP_THRESHOLD = 1.0f;
const float LOSS_SPIKE_THRESHOLD = 1.5f;
const size_t WARMUP_STEPS = 100;
float learning_rate = INITIAL_LEARNING_RATE;
float prev_loss = std::numeric_limits<float>::max();

struct GradientState {
    Matrix momentum;
    Matrix velocity;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
};

int main(int argc, char *argv[]) {
    try {
        // Initialize training and validation datasets
        DataSet train_dataset = create_dataset("train");
        DataSet val_dataset = create_dataset("validation");
        // Initialize logger
        Logger &logger = Logger::getInstance();
        logger.enableLogging();

auto training_data = std::move(TextPreprocessor::preprocess_training_data(train_dataset.pairs)); // convert to lowercase
auto validation_data = std::move(TextPreprocessor::preprocess_training_data(val_dataset.pairs)); // convert to lowercase
#ifdef CUDA_AVAILABLE
        initialize_cuda();
#endif
        // Initialize tokenizer first to get vocab size
        auto tokenizer = std::make_unique<Tokenizer>();
        tokenizer->print_vocabulary_mappings(); // Print initial mappings
        // Analyze token mappings
        analyze_token_mappings(training_data, *tokenizer);

        // Get vocabulary size from the tokenizer
        size_t actual_vocab_size = tokenizer->vocab_size();
        
        std::cout << "Actual vocabulary size: " << actual_vocab_size << std::endl;

        TransformerConfig config;
        config.vocab_size = actual_vocab_size;
        config.hidden_size = 128;
        config.num_heads = 8;
        config.num_layers = 3;
        config.use_cuda = true;
        config.use_flash_attention = true;
        config.use_rope = true;
        config.use_sliding_window = true;
        config.window_size = 256;
        config.use_fp16 = true;
        config.head_dim = config.hidden_size / config.num_heads;  // Add explicit head_dim calculation
        config.batch_size = 8;
        config.num_epochs = 30;
        config.log_level = LogLevel::DEBUG;  // Set desired log level

        std::cout << "Initializing transformer with configuration:\n"
                  << "- Hidden size: " << config.hidden_size << "\n"
                  << "- Attention heads: " << config.num_heads << "\n"
                  << "- Layers: " << config.num_layers << "\n"
                  << "- Batch size: " << config.batch_size << "\n"
                  << "- Number of epochs: " << config.num_epochs << "\n"
                  << "- Using Flash Attention: " << std::boolalpha
                  << config.use_flash_attention << "\n"
                  << "- Using RoPE: " << config.use_rope << "\n"
                  << "- Using Sliding Window: " << config.use_sliding_window
                  << "\n";

        // Initialize components
        Transformer transformer(config);
        auto lm_head = std::make_unique<LanguageModelHead>(config.vocab_size,
                                                           config.hidden_size);

        // Setup advanced components
        TensorCache<Matrix> activation_cache(1024, CacheReplacementPolicy::ARC);
        QuantizationAwareTraining qat(true);
        auto sam_optimizer = std::make_unique<SAM>(0.05f);

        // Print and verify vocabulary mappings
        std::cout << "\nVerifying vocabulary mappings:\n";
        tokenizer->print_vocabulary_mappings();

        if (!tokenizer->verify_mappings()) {
            std::cerr << "Error: Vocabulary mappings are inconsistent!\n";
            return 1;
        }

        // Training loop
        const size_t NUM_EPOCHS = 30;
        const size_t BATCH_SIZE = 5;
        const size_t VALIDATION_INTERVAL = 100;  // Validate every 100 batches
        size_t global_step = 0;
        
        for (size_t epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
            std::cout << "epoch: " << epoch << std::endl;
            float epoch_loss = 0.0f;
            size_t num_batches = 0;
            
            while (num_batches * BATCH_SIZE < train_dataset.size) {
                // Get training batch
                auto batch = get_batch(train_dataset, BATCH_SIZE);
                
                // Process batch
                std::vector<std::vector<int>> input_tokens;
                std::vector<std::vector<int>> target_tokens;
                
                // Tokenize batch
                for (const auto& [input_str, target_str] : batch) {
                    input_tokens.push_back(tokenizer->encode(input_str));
                    target_tokens.push_back(tokenizer->encode(target_str));
                }
                
                // Create target distribution
                Matrix target_distribution = create_batch_target_distribution(target_tokens, config.vocab_size);
                
                // Forward pass
                float batch_loss = 0.0f;
                Matrix hidden_states;
                Matrix logits;
                
                for (size_t i = 0; i < input_tokens.size(); ++i) {
                    // Forward pass through transformer
                    hidden_states = transformer.forward(input_tokens[i]);
                    logits = lm_head->project_to_vocab(hidden_states);
                    
                    // Create target slice with same dimensions as logits
                    Matrix target_slice(logits.rows(), logits.cols(), 0.0f);
                    
                    // Fill target slice from target distribution
                    for (size_t r = 0; r < logits.rows(); r++) {
                        for (size_t j = 0; j < logits.cols(); j++) {
                            target_slice(r, j) = target_distribution(i, j);
                        }
                    }
                                        
                    // Compute loss
                    batch_loss += compute_batch_loss(logits, target_slice);
                    
                    // Compute gradients and backward pass
                    Matrix grad_output = logits - target_slice;  // Simple gradient for cross-entropy
                    transformer.backward(grad_output, input_tokens[i], learning_rate);
                }
                
                batch_loss /= input_tokens.size();
                std::cout << "batch loss: " << batch_loss << std::endl;
                // Update learning rate
                float loss_ratio = batch_loss / (prev_loss + 1e-10f);
                learning_rate = adjust_learning_rate(learning_rate, loss_ratio, global_step);
                prev_loss = batch_loss;
                
                // Update epoch statistics
                epoch_loss += batch_loss;
                num_batches++;
                std::cout << "global step: " << global_step << std::endl;
                global_step++;
                
                // Print progress
                if (global_step % 10 == 0) {
                    std::cout << "\rStep " << global_step 
                              << " - Loss: " << batch_loss 
                              << " - LR: " << learning_rate << std::endl;
                }
                
                // Validation step
                if (global_step % 5 == 0) {
                    std::cout << "validation step" << std::endl;
                    float val_loss = 0.0f;
                    float val_accuracy = 0.0f;
                    size_t val_batches = 0;
                    const size_t MAX_VAL_BATCHES = 5;  // Limit validation batches for efficiency
                    
                    // Validation loop
                    while (val_batches < MAX_VAL_BATCHES) {
                        std::cout << "validation batch: " << val_batches << std::endl;
                        auto val_batch = get_batch(val_dataset, BATCH_SIZE);
                        
                        // Process validation batch
                        std::vector<std::vector<int>> val_input_tokens;
                        std::vector<std::vector<int>> val_target_tokens;
                        
                        for (const auto& [input_str, target_str] : val_batch) {
                            val_input_tokens.push_back(tokenizer->encode(input_str));
                            val_target_tokens.push_back(tokenizer->encode(target_str));
                        }
                        
                        Matrix val_target_distribution = create_batch_target_distribution(val_target_tokens, config.vocab_size);
                        
                        float batch_val_loss = 0.0f;
                        Matrix val_hidden_states;
                        Matrix val_logits;
                        
                        for (size_t i = 0; i < val_input_tokens.size(); ++i) {
                            val_hidden_states = transformer.forward(val_input_tokens[i]);
                            val_logits = lm_head->project_to_vocab(val_hidden_states);
                            
                            // Create validation target slice with same dimensions as logits
                            Matrix val_target_slice(val_logits.rows(), val_logits.cols(), 0.0f);
                            
                            // Fill validation target slice
                            for (size_t r = 0; r < val_logits.rows(); r++) {
                                for (size_t j = 0; j < val_logits.cols(); j++) {
                                    val_target_slice(r, j) = val_target_distribution(i, j);
                                }
                            }
                            
                            batch_val_loss += compute_batch_loss(val_logits, val_target_slice);
                            val_accuracy += calculate_accuracy(val_logits, val_target_slice);
                            std::cout << "val accuracy: " << val_accuracy << std::endl;
                        }
                        
                        batch_val_loss /= val_input_tokens.size();
                        val_loss += batch_val_loss;
                        val_batches++;
                    }
                    
                    // Report validation metrics
                    val_loss /= val_batches;
                    val_accuracy /= val_batches;
                    std::cout << "\nValidation Step " << global_step 
                              << " - Loss: " << val_loss 
                              << " - Accuracy: " << val_accuracy << std::endl;
                }
            }
            
            // Report epoch metrics
            epoch_loss /= num_batches;
            std::cout << "Epoch " << epoch + 1 << "/" << NUM_EPOCHS 
                      << " - Loss: " << epoch_loss << std::endl;
            
            // Test predictions on sample inputs after each epoch
            std::cout << "\n=== Testing Predictions After Epoch " << epoch + 1 << " ===\n";
            std::vector<std::string> test_inputs = {
                "Students research in the",  // Academic context
                "Doctors work in the",       // Medical context
                "Engineers build in the",    // Technical context
                "Artists create in the",     // Creative context
                "Chefs prepare in the"       // Culinary context
            };

            Matrix final_hidden_states;  // Store the last hidden states
            for (const auto& test_input : test_inputs) {
                std::cout << "\nTesting: '" << test_input << "'\n";
                // Encode input
                std::vector<int> test_tokens = tokenizer->encode(test_input);
                
                // Forward pass
                Matrix test_hidden = transformer.forward(test_tokens);
                Matrix test_logits = lm_head->project_to_vocab(test_hidden);
                final_hidden_states = test_hidden;  // Save the last hidden states
                std::cout << "TEST PREDICTIONS" << std::endl;
                // Get top 5 predictions
                print_top_predictions(test_logits, *tokenizer, 5);
            }
            std::cout << "\n=== End of Test Predictions ===\n";

            // Save checkpoint if needed
            const size_t checkpoint_frequency = 2;  // Save every 2 epochs
            if ((epoch + 1) % checkpoint_frequency == 0) {
                std::string save_directory = "models";
                std::string model_name = "transformer_model";
                std::filesystem::create_directories(save_directory);

                // Save checkpoint using ModelSaver
                ModelSaver model_saver;
                if (!model_saver.saveCheckpoint(transformer, save_directory, model_name,
                                              epoch + 1, epoch_loss)) {
                    logger.log("Failed to save checkpoint", LogLevel::ERROR);
                    return 1;
                }
                logger.log("Successfully saved checkpoint for epoch " + 
                          std::to_string(epoch + 1), LogLevel::INFO);
            }

            // Store last hidden states for gradient checkpointing
            GradientCheckpoint::cache_activation(std::to_string(epoch), final_hidden_states);

            // Clear previous checkpoints to save memory
            if (epoch > 0) {
                GradientCheckpoint::get_activation(std::to_string(epoch - 1));  // This removes the activation from cache
            }
            
            // Continue with next epoch...
        }

        // Save the final model
        std::string save_directory = "models";
        std::string model_name = "transformer_model_final";
        std::filesystem::create_directories(save_directory);
        
        std::cout << "\nSaving final model to " << save_directory << "/" << model_name << "...\n";
        ModelSaver model_saver;
        if (!model_saver.saveModel(transformer, save_directory, model_name)) {
            logger.log("Failed to save final model", LogLevel::ERROR);
            return 1;
        }
        logger.log("Successfully saved final model", LogLevel::INFO);
        std::cout << "Model saved successfully!\n";
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}