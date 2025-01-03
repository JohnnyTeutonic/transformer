#include "../include/main.hpp"

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
        const size_t VALIDATION_INTERVAL = 100;  // Validate every 100 batches
        const size_t PRINT_INTERVAL = 50;       // Print progress every 50 steps
        size_t global_step = 0;
        
        // Initialize timing statistics
        TimingStats timing_stats;
        Timer timer;
        
        for (size_t epoch = 0; epoch < config.num_epochs; ++epoch) {
            std::cout << "EPOCH: " << epoch << std::endl;
            float epoch_loss = 0.0f;
            size_t num_batches = 0;
            
            while (num_batches * config.batch_size < train_dataset.size) {
                // Get training batch
                auto batch = get_batch(train_dataset, config.batch_size);
                
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
                    // Time forward pass
                    timer.start();
                    hidden_states = transformer.forward(input_tokens[i]);
                    logits = lm_head->project_to_vocab(hidden_states);
                    timing_stats.forward_pass_time += timer.stop();
                    timing_stats.forward_pass_count++;
                    
                    Matrix target_slice(logits.rows(), logits.cols(), 0.0f);
                    for (size_t r = 0; r < logits.rows(); r++) {
                        for (size_t j = 0; j < logits.cols(); j++) {
                            target_slice(r, j) = target_distribution(i, j);
                        }
                    }
                    
                    batch_loss += compute_batch_loss(logits, target_slice);
                    Matrix grad_output = logits - target_slice;
                    
                    // Time backward pass
                    timer.start();
                    transformer.backward(grad_output, input_tokens[i], learning_rate);
                    timing_stats.backward_pass_time += timer.stop();
                    timing_stats.backward_pass_count++;
                }
                
                batch_loss /= input_tokens.size();
                
                // Update learning rate
                float loss_ratio = batch_loss / (prev_loss + 1e-10f);
                learning_rate = adjust_learning_rate(learning_rate, loss_ratio, global_step);
                prev_loss = batch_loss;
                
                // Update epoch statistics
                epoch_loss += batch_loss;
                num_batches++;
                global_step++;
                
                // Print progress less frequently
                if (global_step % PRINT_INTERVAL == 0) {
                    std::cout << "\rEpoch " << epoch + 1 << "/" << config.num_epochs 
                              << " - Step " << global_step 
                              << " - Loss: " << batch_loss 
                              << " - LR: " << learning_rate << std::flush;
                }
                
                // Validation step (less frequent)
                if (global_step % VALIDATION_INTERVAL == 0) {
                    timer.start();  // Start timing validation
                    float val_loss = 0.0f;
                    float val_accuracy = 0.0f;
                    size_t val_batches = 0;
                    const size_t MAX_VAL_BATCHES = 5;
                    
                    while (val_batches < MAX_VAL_BATCHES) {
                        auto val_batch = get_batch(val_dataset, config.batch_size);
                        
                        std::vector<std::vector<int>> val_input_tokens;
                        std::vector<std::vector<int>> val_target_tokens;
                        
                        for (const auto& [input_str, target_str] : val_batch) {
                            val_input_tokens.push_back(tokenizer->encode(input_str));
                            val_target_tokens.push_back(tokenizer->encode(target_str));
                        }
                        
                        Matrix val_target_distribution = create_batch_target_distribution(val_target_tokens, config.vocab_size);
                        float batch_val_loss = 0.0f;
                        
                        for (size_t i = 0; i < val_input_tokens.size(); ++i) {
                            Matrix val_hidden_states = transformer.forward(val_input_tokens[i]);
                            Matrix val_logits = lm_head->project_to_vocab(val_hidden_states);
                            
                            Matrix val_target_slice(val_logits.rows(), val_logits.cols(), 0.0f);
                            for (size_t r = 0; r < val_logits.rows(); r++) {
                                for (size_t j = 0; j < val_logits.cols(); j++) {
                                    val_target_slice(r, j) = val_target_distribution(i, j);
                                }
                            }
                            
                            batch_val_loss += compute_batch_loss(val_logits, val_target_slice);
                            val_accuracy += calculate_accuracy(val_logits, val_target_slice);
                        }
                        
                        batch_val_loss /= val_input_tokens.size();
                        val_loss += batch_val_loss;
                        val_batches++;
                    }
                    
                    val_loss /= val_batches;
                    val_accuracy /= val_batches;
                    timing_stats.validation_time += timer.stop();  // Stop timing validation
                    timing_stats.validation_count++;
                    
                    std::cout << "\nValidation Step " << global_step 
                              << " - Loss: " << val_loss 
                              << " - Accuracy: " << val_accuracy << std::endl;
                }
            }
            
            // Report epoch metrics
            epoch_loss /= num_batches;
            std::cout << "Epoch " << epoch + 1 << "/" << config.num_epochs 
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
                timer.start();  // Start timing checkpoint
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
                timing_stats.checkpoint_time += timer.stop();  // Stop timing checkpoint
                timing_stats.checkpoint_count++;
                
                logger.log("Successfully saved checkpoint for epoch " + 
                          std::to_string(epoch + 1), LogLevel::INFO);
            }

            // Store last hidden states for gradient checkpointing
            GradientCheckpoint::cache_activation(std::to_string(epoch), final_hidden_states);

            // Clear previous checkpoints to save memory
            if (epoch > 0) {
                GradientCheckpoint::get_activation(std::to_string(epoch - 1));  // This removes the activation from cache
            }
            
            // Print timing statistics at the end of each epoch
            print_timing_stats(timing_stats);
        }

        // Save the final model
        std::string save_directory = "models";
        std::string model_name = "transformer_model_final";
        std::filesystem::create_directories(save_directory);
        
        timer.start();  // Time final model saving
        std::cout << "\nSaving final model to " << save_directory << "/" << model_name << "...\n";
        ModelSaver model_saver;
        if (!model_saver.saveModel(transformer, save_directory, model_name)) {
            logger.log("Failed to save final model", LogLevel::ERROR);
            return 1;
        }
        timing_stats.checkpoint_time += timer.stop();
        timing_stats.checkpoint_count++;
        
        logger.log("Successfully saved final model", LogLevel::INFO);
        std::cout << "Model saved successfully!\n";
        
        // Print final timing statistics
        std::cout << "\n=== Final Training Statistics ===\n";
        print_timing_stats(timing_stats);
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}