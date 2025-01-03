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
  std::cout << "entering main" << std::endl;
  // Initialize logger
  Logger &logger = Logger::getInstance();
  logger.enableLogging();

  try {
#ifdef CUDA_AVAILABLE
    initialize_cuda();
#endif
    // Initialize tokenizer first to get vocab size
    auto tokenizer = std::make_unique<Tokenizer>();
    tokenizer->print_vocabulary_mappings(); // Print initial mappings
    
    // Get vocabulary size from the tokenizer
    size_t actual_vocab_size = tokenizer->vocab_size();
    
    std::cout << "Actual vocabulary size: " << actual_vocab_size << std::endl;

    TransformerConfig config;
    config.vocab_size = actual_vocab_size;
    config.hidden_size = 360;
    config.num_heads = 12;
    config.num_layers = 6;
    config.use_cuda = true;
    config.use_flash_attention = true;
    config.use_rope = true;
    config.use_sliding_window = true;
    config.window_size = 256;
    config.use_fp16 = true;
    config.head_dim = config.hidden_size / config.num_heads;  // Add explicit head_dim calculation
    config.batch_size = 8;
    config.num_epochs = 10;
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

    // Get training data
    std::vector<std::pair<std::string, std::string>> training_data = create_training_data();
    
    // Preprocess the training data (convert to lowercase)
    training_data = TextPreprocessor::preprocess_training_data(training_data);
    
    // Analyze token mappings
    analyze_token_mappings(training_data, *tokenizer);
    
    // Print vocabulary for inspection
    std::cout << "\n=== Full Vocabulary Mapping ===\n";
    tokenizer->print_vocabulary_mappings();
    std::cout << "\n";

    // Training parameters
    const size_t checkpoint_frequency = 2; // Save checkpoint every 2 epochs

    // Initialize model saver
    ModelSaver model_saver;
    std::string save_directory = "models";
    std::string model_name = "transformer_model";

    // Training loop
    size_t global_step = 0;  // Move outside epoch loop
    Matrix last_hidden_states;
    for (size_t epoch = 0; epoch < config.num_epochs; ++epoch) {
        std::cout << "Epoch " << epoch + 1 << "/" << config.num_epochs << "\n";
        float epoch_loss = 0.0f;
        size_t total_batches = (training_data.size() + config.batch_size - 1) / config.batch_size;
        
        // Process batches
        for (size_t batch = 0; batch < total_batches; ++batch) {
            size_t start_idx = batch * config.batch_size;
            size_t end_idx = std::min(start_idx + config.batch_size, training_data.size());
            
            // Create batch with validation
            std::vector<std::vector<int>> input_batch;
            std::vector<std::vector<int>> target_tokens;
            
            // Fill and validate batch
            bool batch_valid = true;
            for (size_t j = start_idx; j < end_idx; ++j) {
                const auto &[input_str, target_str] = training_data[j];
                std::vector<int> input_tokens = tokenizer->encode(input_str);
                std::vector<int> curr_target_tokens = tokenizer->encode(target_str);
                
                if (!validate_input_sequence(input_tokens, config.vocab_size) || 
                    !validate_input_sequence(curr_target_tokens, config.vocab_size)) {
                    std::cerr << "Invalid sequence at position " << j << std::endl;
                    batch_valid = false;
                    break;
                }
                
                input_batch.push_back(input_tokens);
                target_tokens.push_back(curr_target_tokens);
            }
            
            if (!batch_valid) continue;  // Skip invalid batches
            
            // Create target distribution for entire batch at once
            Matrix target_distribution = create_batch_target_distribution(target_tokens, config.vocab_size);
            
            // Forward pass with gradient accumulation
            Matrix accumulated_gradients(config.hidden_size, config.vocab_size, 0.0f);
            float batch_loss = 0.0f;
            
            for (size_t i = 0; i < input_batch.size(); ++i) {
                // Forward pass
                Matrix hidden_states = transformer.forward(input_batch[i]);
                Matrix logits = lm_head->project_to_vocab(hidden_states);
                
                // Initialize accumulated_gradients with correct dimensions to match hidden_states
                if (i == 0) {
                    // Initialize with correct dimensions: should match hidden_states shape
                    accumulated_gradients = Matrix(hidden_states.rows(), hidden_states.cols(), 0.0f);
                }
                
                // Extract corresponding row from target distribution
                Matrix target_slice(logits.rows(), target_distribution.cols());
                for (size_t j = 0; j < target_distribution.cols(); j++) {
                    for (size_t r = 0; r < logits.rows(); r++) {
                        target_slice(r, j) = target_distribution(i, j);
                    }
                }
                
                // Compute loss and gradients
                float sample_loss = compute_batch_loss(logits, target_slice);
                batch_loss += sample_loss;
                
                // Compute gradients using SAM
                std::vector<Matrix> param_grads;
                param_grads.reserve(transformer.getLayers().size());
                sam_optimizer->compute_parameter_gradients(hidden_states, target_slice, param_grads);
                
                // Accumulate gradients with proper dimensions
                for (const auto& grad : param_grads) {
                    // Ensure grad dimensions match hidden_states before accumulating
                    if (grad.rows() == accumulated_gradients.rows() && 
                        grad.cols() == accumulated_gradients.cols()) {
                        for (size_t j = 0; j < grad.size(); j++) {
                            accumulated_gradients.data()[j] += grad.data()[j];
                        }
                    } else {
                        std::cout << "Skipping gradient with mismatched dimensions. Expected: " 
                                  << accumulated_gradients.rows() << "x" << accumulated_gradients.cols()
                                  << ", Got: " << grad.rows() << "x" << grad.cols() << std::endl;
                    }
                }
            }
            
            // Average gradients
            float scale = 1.0f / input_batch.size();
            for (size_t i = 0; i < accumulated_gradients.size(); i++) {
                accumulated_gradients.data()[i] *= scale;
            }
            
            // Update learning rate
            float loss_ratio = batch_loss / (prev_loss + 1e-10f);
            learning_rate = adjust_learning_rate(learning_rate, loss_ratio, global_step);
            global_step++;  // Increment after using it
            
            // Apply gradients
            transformer.backward(accumulated_gradients, input_batch[0], learning_rate);
            
            // Update loss tracking
            prev_loss = batch_loss;
            epoch_loss += batch_loss;
            
            // Print progress with learning rate
            std::cout << "\rBatch " << batch + 1 << "/" << total_batches 
                      << " in epoch " << epoch + 1 
                      << " (Loss: " << batch_loss 
                      << ", LR: " << learning_rate 
                      << ", Step: " << global_step  // Add step counter to output
                      << ")" << std::flush;
        }
        
        std::cout << "\nCompleted epoch " << epoch + 1 << "/" << config.num_epochs 
                  << " (Loss: " << epoch_loss/total_batches << ")" << std::endl;
        
        // Save checkpoint
        if ((epoch + 1) % checkpoint_frequency == 0) {
            if (!model_saver.saveCheckpoint(transformer, save_directory, model_name,
                                            epoch + 1, epoch_loss)) {
                logger.log("Failed to save checkpoint", LogLevel::ERROR);
                return 1;
            }
        }

        // Test prediction on a sample input
        if ((epoch + 1) % 2 == 0) {
            // Test multiple different contexts
            std::vector<std::string> test_inputs = {
                "I go to",                  // Basic location
                "Surgeons operate in the",  // Medical context
                "Athletes train in the",    // Sports context
                "Musicians perform in the", // Entertainment context
                "Students research in the", // Educational context
                "Chefs cook in the",        // Culinary context
                "Artists create in the",    // Creative context
                "Engineers work in the",    // Technical context
                "Lawyers practice in the",  // Legal context
                "Teachers instruct in the", // Educational context
                "Scientists experiment in", // Research context
                "Pilots fly through the",   // Aviation context
                "Dancers rehearse in the",  // Performance context
                "Writers compose in the",   // Literary context
                "Mechanics repair in the"   // Automotive context
            };

            for (const auto &test_input : test_inputs) {
                std::cout << "\nTesting: '" << test_input << "'\n";
                std::vector<int> test_tokens = tokenizer->encode(test_input);
                Matrix test_hidden = transformer.forward(test_tokens);
                Matrix test_logits = lm_head->forward(test_hidden);
                print_top_predictions(test_logits, *tokenizer, 5);
            }
        }
    }

    std::cout << "\nTraining completed!\n";

    // Create directories if they don't exist
    std::filesystem::create_directories(save_directory);

    // Save the trained model
    std::cout << "\nSaving final model to " << save_directory << "/"
              << model_name << "...\n";
    bool save_success =
        model_saver.saveModel(transformer, save_directory, model_name);
    if (save_success) {
        logger.log("Successfully saved model to " + save_directory + "/" +
                   model_name, LogLevel::INFO);
        std::cout << "Model saved successfully!\n";
    } else {
        logger.log("Failed to save model to " + save_directory + "/" + model_name,
                   LogLevel::ERROR);
        return 1;
    }

    // Demonstrate quantization
    std::cout << "\nTesting quantization...\n";
    std::vector<Matrix> calibration_data{
        last_hidden_states}; // Use stored hidden states
    qat.calibrate(transformer, calibration_data);
    Matrix quantized = qat.quantize_weights(last_hidden_states, "layer_0");
    print_matrix(quantized, "Quantized hidden states");

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

#ifdef CUDA_AVAILABLE
  cleanup_cuda(); // Cleanup at program end
#endif
  logger.disableLogging();
  std::cout << "exiting main" << std::endl;
  return 0;
}