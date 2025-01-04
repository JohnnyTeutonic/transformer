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

        // Process training data
        for (auto& [input_str, target_str] : training_data) {
            std::istringstream iss_input(input_str);
            std::istringstream iss_target(target_str);
            std::string word;
            
            // Process input string
            std::string processed_input;
            while (iss_input >> word) {
                processed_input += word + " ";
            }
            if (!processed_input.empty()) {
                processed_input.pop_back(); // Remove trailing space
            }
            
            // Process target string
            std::string processed_target;
            while (iss_target >> word) {
                processed_target += word + " ";
            }
            if (!processed_target.empty()) {
                processed_target.pop_back(); // Remove trailing space
            }
            
            input_str = processed_input;
            target_str = processed_target;
        }

        // Process validation data
        for (auto& [input_str, target_str] : validation_data) {
            std::istringstream iss_input(input_str);
            std::istringstream iss_target(target_str);
            std::string word;
            
            // Process input string
            std::string processed_input;
            while (iss_input >> word) {
                processed_input += word + " ";
            }
            if (!processed_input.empty()) {
                processed_input.pop_back(); // Remove trailing space
            }
            
            // Process target string
            std::string processed_target;
            while (iss_target >> word) {
                processed_target += word + " ";
            }
            if (!processed_target.empty()) {
                processed_target.pop_back(); // Remove trailing space
            }
            
            input_str = processed_input;
            target_str = processed_target;
        }

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
        static std::unique_ptr<SAM> sam_optimizer = std::make_unique<SAM>(
            0.05f,  // rho (neighborhood size)
            0.001f  // learning rate
        );

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
                    // For next token prediction:
                    // Input should be all tokens except the last one
                    // Target should be just the last token
                    std::vector<int> all_tokens = tokenizer->encode(input_str + " " + target_str);
                    
                    if (all_tokens.size() < 2) continue;  // Skip if too short
                    
                    // Input is everything except the last token
                    std::vector<int> input_seq(all_tokens.begin(), all_tokens.end() - 1);
                    // Target is just the last token
                    std::vector<int> target_seq = {all_tokens.back()};
                    
                    input_tokens.push_back(input_seq);
                    target_tokens.push_back(target_seq);
                }
                
                // Create target distribution for just the last token
                Matrix target_distribution = create_batch_target_distribution(target_tokens, config.vocab_size);
                
                // Forward pass
                float batch_loss = 0.0f;
                Matrix hidden_states;
                Matrix logits;
                Matrix accumulated_gradients;
                
                // Process entire batch at once
                std::vector<int> batch_input_tokens;
                std::vector<int> batch_lengths;
                size_t max_length = 0;
                
                // Find max length and prepare batch
                for (const auto& tokens : input_tokens) {
                    max_length = std::max(max_length, tokens.size());
                    batch_lengths.push_back(tokens.size());
                }
                
                // Pad sequences to max_length
                for (size_t i = 0; i < input_tokens.size(); ++i) {
                    const auto& tokens = input_tokens[i];
                    batch_input_tokens.insert(batch_input_tokens.end(), tokens.begin(), tokens.end());
                    // Pad with PAD token
                    batch_input_tokens.insert(batch_input_tokens.end(), 
                                           max_length - tokens.size(),
                                           tokenizer->get_pad_token_id());
                }
                
                // Forward pass on entire batch
                hidden_states = transformer.forward(batch_input_tokens);
                logits = lm_head->project_to_vocab(hidden_states);
                
                // Compute loss and gradients for each sequence in batch
                size_t offset = 0;
                for (size_t i = 0; i < input_tokens.size(); ++i) {
                    size_t seq_length = batch_lengths[i];
                    
                    // Create full sequence gradients initialized to zero with max_length
                    Matrix sequence_gradients(max_length, logits.cols(), 0.0f);
                    
                    // Extract logits for this sequence's last position
                    Matrix last_position_logits(1, logits.cols());
                    for (size_t j = 0; j < logits.cols(); ++j) {
                        last_position_logits(0, j) = logits(offset + seq_length - 1, j);
                    }
                    
                    // Get target distribution for this sequence
                    Matrix target_slice(1, logits.cols(), 0.0f);
                    for (size_t j = 0; j < logits.cols(); j++) {
                        target_slice(0, j) = target_distribution(i, j);
                    }
                    
                    // Compute loss for this sequence
                    batch_loss += compute_batch_loss(last_position_logits, target_slice);
                    
                    // Compute gradients for the last position
                    Matrix last_position_grads = compute_loss_gradients(last_position_logits, target_slice);
                    
                    // Place the gradients at the last position in the sequence
                    for (size_t j = 0; j < logits.cols(); ++j) {
                        sequence_gradients(seq_length - 1, j) = last_position_grads(0, j);
                    }
                    
                    // Initialize accumulated_gradients with correct dimensions on first sequence
                    if (i == 0) {
                        accumulated_gradients = Matrix(max_length, logits.cols(), 0.0f);
                    }
                    
                    // Add gradients
                    accumulated_gradients += sequence_gradients;
                    
                    offset += max_length;
                }
                
                // Average loss and gradients over batch
                batch_loss /= input_tokens.size();
                accumulated_gradients *= (1.0f / input_tokens.size());
                
                // Get transformer parameters and convert to pointers
                std::vector<Matrix*> param_ptrs;
                auto& params = transformer.parameters();
                for (auto& param : params) {
                    param_ptrs.push_back(&param);
                }
                
                // Create gradient vectors for each parameter
                std::vector<Matrix> grads;
                grads.reserve(params.size());
                for (const auto& param : params) {
                    grads.push_back(Matrix(param.rows(), param.cols(), 0.0f));
                }
                
                // Copy accumulated gradients to first gradient matrix
                if (!grads.empty()) {
                    grads[0] = accumulated_gradients;
                }
                
                // First step of SAM
                sam_optimizer->first_step(param_ptrs, grads);
                
                // Recompute forward pass at the perturbed point
                hidden_states = transformer.forward(batch_input_tokens);
                logits = lm_head->project_to_vocab(hidden_states);
                
                // Recompute loss and gradients at the perturbed point
                Matrix perturbed_gradients(max_length, logits.cols(), 0.0f);
                float perturbed_loss = 0.0f;
                
                offset = 0;
                for (size_t i = 0; i < input_tokens.size(); ++i) {
                    size_t seq_length = batch_lengths[i];
                    
                    // Extract logits for this sequence's last position
                    Matrix last_position_logits(1, logits.cols());
                    for (size_t j = 0; j < logits.cols(); ++j) {
                        last_position_logits(0, j) = logits(offset + seq_length - 1, j);
                    }
                    
                    // Get target distribution for this sequence
                    Matrix target_slice(1, logits.cols(), 0.0f);
                    for (size_t j = 0; j < logits.cols(); j++) {
                        target_slice(0, j) = target_distribution(i, j);
                    }
                    
                    // Compute loss at perturbed point
                    perturbed_loss += compute_batch_loss(last_position_logits, target_slice);
                    
                    // Compute gradients at perturbed point
                    Matrix last_position_grads = compute_loss_gradients(last_position_logits, target_slice);
                    
                    // Place the gradients at the last position
                    for (size_t j = 0; j < logits.cols(); ++j) {
                        perturbed_gradients(seq_length - 1, j) = last_position_grads(0, j);
                    }
                    
                    offset += max_length;
                }
                
                perturbed_loss /= input_tokens.size();
                perturbed_gradients *= (1.0f / input_tokens.size());
                
                // Create perturbed gradient vectors
                std::vector<Matrix> perturbed_grads;
                perturbed_grads.reserve(params.size());
                for (const auto& param : params) {
                    perturbed_grads.push_back(Matrix(param.rows(), param.cols(), 0.0f));
                }
                
                // Copy perturbed gradients to first gradient matrix
                if (!perturbed_grads.empty()) {
                    perturbed_grads[0] = perturbed_gradients;
                }
                
                // Second step of SAM with perturbed gradients
                sam_optimizer->second_step(param_ptrs, perturbed_grads);
                
                // Update learning rate using perturbed loss
                float loss_ratio = perturbed_loss / (prev_loss + 1e-10f);
                learning_rate = adjust_learning_rate(learning_rate, loss_ratio, global_step);
                prev_loss = perturbed_loss;
                
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
            for (auto& test_input : test_inputs) {
                std::cout << "\nTesting: '" << test_input << "'\n";
                // Process test input
                std::istringstream iss(test_input);
                std::string word;
                std::string processed_input;
                
                while (iss >> word) {
                    processed_input += word + " ";
                }
                if (!processed_input.empty()) {
                    processed_input.pop_back(); // Remove trailing space
                }
                
                // Encode processed input
                std::vector<int> test_tokens = tokenizer->encode(processed_input);
                
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