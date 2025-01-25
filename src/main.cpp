#include "../include/main.hpp"
#include <fstream>
#include <nlohmann/json.hpp>
#include <random>
#include "../include/tokenizer.hpp"
#include <chrono>

// Add necessary forward declarations and structures
std::unique_ptr<Tokenizer> tokenizer;
PerformanceMetrics metrics; // Single definition of the global metrics variable

// Configuration constants
const float INITIAL_LEARNING_RATE = 0.001f;
float learning_rate = INITIAL_LEARNING_RATE;
float prev_loss = std::numeric_limits<float>::max();
size_t global_step = 0;
size_t no_improvement_count = 0;
size_t completed_iterations = 0;
const size_t MAX_OUTPUT_ITERATIONS = 2;
const float GRADIENT_CLIP_THRESHOLD = 1.0f;
const float L1_REGULARIZATION = 0.01f;  // Sparsity regularization
const size_t INCREASED_BATCH_SIZE = 32;  // Increased from 1
float batch_loss = 0.0f;

// Add before the gradient scaling code
float calculateGradientMagnitude(const std::vector<Matrix>& gradients) {
    float total_magnitude = 0.0f;
    int count = 0;
    
    for (const auto& grad : gradients) {
        for (size_t i = 0; i < grad.rows(); i++) {
            for (size_t j = 0; j < grad.cols(); j++) {
                total_magnitude += std::abs(grad(i, j));
                count++;
            }
        }
    }
    
    return count > 0 ? total_magnitude / count : 0.0f;
}

int main(int argc, char* argv[]) {
    std::cout << "entering main" << std::endl;
    Logger& logger = Logger::getInstance();
    logger.startLogging();

    try {
        // Load configuration
        std::filesystem::path exe_path = std::filesystem::current_path().parent_path();
        std::filesystem::path config_path = exe_path / "config" / "transformer_config.json";
        TransformerConfig config = Utils::load_config(config_path.string());

        // Initialize random seed
        std::srand(static_cast<unsigned int>(std::time(nullptr)));

#ifdef CUDA_AVAILABLE
        // Initialize CUDA
        if (cudaSetDevice(0) != cudaSuccess) {
            std::cerr << "Failed to initialize CUDA device" << std::endl;
            return 1;
        }

        // Create CUDA stream
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        std::cout << "CUDA is available" << std::endl;
#else
        std::cout << "CUDA is not available" << std::endl;
#endif

        // Load training data first
        auto training_pairs = Utils::create_training_data();
        std::cout << "Loaded " << training_pairs.size() << " training pairs" << std::endl;

        // Initialize tokenizer with config
        std::cout << "Initializing tiktoken with encoding: gpt2" << std::endl;
        tokenizer = std::make_unique<Tokenizer>("gpt2");
        
        try {
            tokenizer->initialize();  // Initialize with default encoding
            std::cout << "Initialized tokenizer. Vocabulary size: " 
                      << tokenizer->vocab_size() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Failed to initialize tokenizer: " << e.what() << std::endl;
            return 1;
        }

        // Update vocabulary size in config based on tokenizer
        config.vocab_size = tokenizer->vocab_size();
        std::cout << "Using vocabulary size: " << config.vocab_size << std::endl;

        // Update batch size in config
        config.batch_size = INCREASED_BATCH_SIZE;
        std::cout << "Using increased batch size: " << config.batch_size << std::endl;

        // Initialize model with updated config
        Transformer transformer(config);
        auto lm_head = std::make_unique<LanguageModelHead>(config.hidden_size, config.vocab_size);

        // Setup advanced components
        TensorCache<Matrix> activation_cache(1024, CacheReplacementPolicy::ARC);
        QuantizationAwareTraining qat(true);
        auto sam_optimizer = std::make_unique<SAM>(0.05f);

        // Print vocabulary mappings
        std::cout << "\nPrinting vocabulary mappings:\n";
        tokenizer->print_vocabulary_mappings();

        // Training parameters
        const size_t checkpoint_frequency =
            config.paths.checkpoint_frequency; // Save checkpoint every 2 epochs

        // Initialize model saver
        ModelSaver model_saver;
        std::string save_directory = config.paths.save_directory;
        std::string model_name = config.paths.model_name;

        // After transformer initialization but before training loop
        if (config.load_from_checkpoint) {
            std::cout << "Attempting to load checkpoint from: " << config.checkpoint_to_load
                      << std::endl;

            try {
                if (!std::filesystem::exists(config.checkpoint_to_load)) {
                    std::cout << "Warning: Checkpoint file does not exist: "
                              << config.checkpoint_to_load << std::endl;
                    std::cout << "Proceeding with training from scratch..." << std::endl;
                } else {
                    // Attempt to load the checkpoint
                    if (!model_saver.loadCheckpoint(transformer, config.checkpoint_to_load)) {
                        std::cerr << "Warning: Failed to load checkpoint from: "
                                  << config.checkpoint_to_load << std::endl;
                        std::cout << "Proceeding with training from scratch..." << std::endl;
                    } else {
                        // Extract epoch number from checkpoint filename
                        std::string filename =
                            std::filesystem::path(config.checkpoint_to_load).filename().string();
                        size_t epoch_pos = filename.find("epoch_");
                        if (epoch_pos != std::string::npos) {
                            std::string epoch_str = filename.substr(epoch_pos + 6);
                            size_t end_pos = epoch_str.find_first_not_of("0123456789");
                            epoch_str = epoch_str.substr(0, end_pos);
                            global_step =
                                std::stoul(epoch_str) * (training_pairs.size() / config.batch_size);
                        }

                        std::cout << "Successfully loaded checkpoint. Resuming from global step: "
                                  << global_step << std::endl;
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "Warning: Error during checkpoint loading: " << e.what() << std::endl;
                std::cout << "Proceeding with training from scratch..." << std::endl;
            }
        }

        // Training loop
        size_t global_step = 0; // Move outside epoch loop
        Matrix last_hidden_states;

        // Load validation data
        auto validation_data = Utils::load_validation_data();
        std::cout << "Loaded " << validation_data.size() << " validation examples\n";

        // Update any hardcoded token references
        int pad_id = tokenizer->get_pad_token_id();    // Should be 0
        std::cout << "pad_id: " << pad_id << std::endl;
        int unk_id = tokenizer->get_unk_token_id();    // Should be 1
        std::cout << "unk_id: " << unk_id << std::endl;
        int bos_id = tokenizer->get_bos_token_id();    // Should be 2
        std::cout << "bos_id: " << bos_id << std::endl;
        int eos_id = tokenizer->get_eos_token_id();    // Should be 3
        std::cout << "eos_id: " << eos_id << std::endl;
        int mask_id = tokenizer->get_mask_token_id();  // Should be 4
        std::cout << "mask_id: " << mask_id << std::endl;
        std::cout << "epochs: " << config.num_epochs << std::endl;

        // Add these test patterns after each validation run
        std::vector<std::string> test_patterns = {
            "I go to the",
            "She walks to the",
            "He drives to the",
            "They run to the",
            "We walk to the"
        };

        for (size_t epoch = 0; epoch < config.num_epochs; ++epoch) {
            std::cout << "Epoch " << epoch + 1 << "/" << config.num_epochs << "\n";
            float epoch_loss = 0.0f;
            size_t num_batches =
                (training_pairs.size() + config.batch_size - 1) / config.batch_size;

            // Process batches
            for (size_t batch = 0; batch < num_batches; ++batch) {
                metrics.start_timer("batch_processing");

                size_t start_idx = batch * config.batch_size;
                size_t end_idx = std::min(start_idx + config.batch_size, training_pairs.size());
                size_t current_batch_size = end_idx - start_idx;

                // Find maximum sequence length in this batch
                size_t max_seq_len = 0;
                for (size_t j = start_idx; j < end_idx; ++j) {
                    const auto& [input_str, target_str] = training_pairs[j];
                    std::vector<int> input_tokens = tokenizer->encode(input_str);
                    std::vector<int> target_tokens = tokenizer->encode(target_str);
                    // Consider both input and target sequence lengths
                    max_seq_len = std::max({max_seq_len, input_tokens.size(), target_tokens.size()});
                }
                std::cout << "\n=== Processing Batch " << batch << " ===\n";

                // Create batch with validation
                std::vector<std::vector<int>> input_batch;
                std::vector<std::vector<int>> target_batch;  // Rename from target_tokens

                // Fill and validate batch with padding
                bool batch_valid = true;
                for (size_t j = start_idx; j < end_idx; ++j) {
                    const auto& [input_str, target_str] = training_pairs[j];

                    // Preprocess text
                    std::string processed_input = input_str;
                    std::string processed_target = target_str;
                    tokenizer->preprocess_text(processed_input);
                    tokenizer->preprocess_text(processed_target);

                    // Encode using appropriate tokenizer
                    std::vector<int> input_tokens = tokenizer->encode(processed_input);
                    std::vector<int> target_tokens = tokenizer->encode(processed_target);

                    // Validate sequences
                    if (!Utils::validate_input_sequence(input_tokens, tokenizer->vocab_size()) ||
                        !Utils::validate_input_sequence(target_tokens, tokenizer->vocab_size())) {
                        std::cerr << "Invalid sequence at position " << j << std::endl;
                        batch_valid = false;
                        break;
                    }

                    // Pad sequences to max_seq_len
                    while (input_tokens.size() < max_seq_len) {
                        input_tokens.push_back(tokenizer->get_pad_token_id());
                    }
                    while (target_tokens.size() < max_seq_len) {  // Add padding for target tokens
                        target_tokens.push_back(tokenizer->get_pad_token_id());
                    }

                    input_batch.push_back(input_tokens);
                    target_batch.push_back(target_tokens);  // Use target_batch instead of target_tokens
                }

                if (!batch_valid)
                    continue; // Skip invalid batches

                std::cout << "Input batch size: " << input_batch.size() << " sequences\n";
                std::cout << "Target batch size: " << target_batch.size() << " sequences\n";

                // First collect valid sequences
                std::vector<std::vector<int>> valid_input_batch;
                std::vector<std::vector<int>> valid_target_batch;

                for (size_t i = 0; i < input_batch.size(); i++) {
                    const auto& input_sequence = input_batch[i];
                    const auto& target_sequence = target_batch[i];
                    
                    if (input_sequence.size() != max_seq_len) {
                        std::cerr << "Error: Input sequence length mismatch. Expected " << max_seq_len 
                                  << " but got " << input_sequence.size() << std::endl;
                        continue;
                    }
                    
                    if (target_sequence.size() != max_seq_len) {
                        std::cerr << "Error: Target sequence length mismatch. Expected " << max_seq_len 
                                  << " but got " << target_sequence.size() << std::endl;
                        continue;
                    }
                    
                    valid_input_batch.push_back(input_sequence);
                    valid_target_batch.push_back(target_sequence);
                }

                if (valid_input_batch.empty()) {
                    std::cerr << "Error: No valid sequences in batch\n";
                    continue;
                }

                auto batch_start_time = std::chrono::high_resolution_clock::now();

                // Create target distribution for entire batch using only valid sequences
                Matrix target_distribution = Utils::create_batch_target_distribution(
                    valid_target_batch, *tokenizer, config.vocab_size, max_seq_len);

                // Process the batch as a single sequence
                std::vector<int> flattened_batch;
                flattened_batch.reserve(valid_input_batch.size() * max_seq_len);
                for (const auto& sequence : valid_input_batch) {
                    flattened_batch.insert(flattened_batch.end(), sequence.begin(), sequence.end());
                }

                // Forward pass with the flattened batch
                transformer.set_training(true);
                metrics.start_timer("forward_pass");
                Matrix hidden_states = transformer.forward(flattened_batch, "", *tokenizer);
                metrics.stop_timer("forward_pass");

                metrics.record_memory_usage(hidden_states.bytes());
                Matrix logits = lm_head->project_to_vocab(hidden_states);

                // Only print predictions every 50 batches to reduce output
                if (batch % 50 == 0) {
                    std::cout << "\n=== Batch " << batch << " Status ===\n";
                    std::cout << "Batch size: " << valid_input_batch.size() << " sequences\n";
                    std::cout << "Total tokens: " << flattened_batch.size() << "\n";
                    std::cout << "Hidden states shape: " << hidden_states.rows() << "x" << hidden_states.cols() << "\n";
                    std::cout << "Memory usage: " << (hidden_states.bytes() / 1024.0f / 1024.0f) << " MB\n";
                    
                    // Print sample predictions
                    std::cout << "\nSample predictions:\n";
                    Utils::print_top_predictions(logits, *tokenizer, transformer, 5);
                }

                float batch_loss = Utils::compute_batch_loss(logits, target_distribution, *tokenizer);

                // Add L1 regularization to encourage sparsity
                float l1_loss = 0.0f;
                for (size_t i = 0; i < hidden_states.rows(); i++) {
                    for (size_t j = 0; j < hidden_states.cols(); j++) {
                        l1_loss += std::abs(hidden_states(i, j));
                    }
                }
                batch_loss += L1_REGULARIZATION * l1_loss;

                // First calculate the loss gradients
                Matrix loss_gradients(current_batch_size, config.vocab_size);
                for (size_t i = 0; i < current_batch_size; i++) {
                    for (size_t j = 0; j < config.vocab_size; j++) {
                        float predicted = logits(i, j);
                        float target = target_distribution(i, j);
                        loss_gradients(i, j) = predicted - target;
                    }
                }

                // Then do the backward pass
                Matrix lm_head_gradients = lm_head->backward(loss_gradients);
                transformer.backward(lm_head_gradients, flattened_batch, learning_rate);

                // Update tracking variables
                float current_loss = batch_loss;
                if (current_loss >= prev_loss) {
                    no_improvement_count++;
                    if (no_improvement_count > 10) {
                        std::cout << "Warning: Loss not improving for " << no_improvement_count 
                                  << " iterations" << std::endl;
                    }
                } else {
                    no_improvement_count = 0;
                }
                prev_loss = current_loss;
                epoch_loss += batch_loss;
                global_step++;

                metrics.stop_timer("batch_processing");

                auto batch_end_time = std::chrono::high_resolution_clock::now();
                auto batch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                    batch_end_time - batch_start_time).count();

                // Only print if we haven't exceeded max iterations
                if (completed_iterations < MAX_OUTPUT_ITERATIONS) {
                    if (batch % 10 == 0) {
                        std::cout << "\rBatch " << batch << " completed in " << batch_duration 
                                  << "ms (Loss: " << batch_loss
                                  << ", LR: " << learning_rate << ")" << std::flush;
                    }

                    // Make predictions after each batch
                    std::string test_input = "I go to";
                    std::string processed_input = test_input;
                    tokenizer->preprocess_text(processed_input);
                    std::vector<int> test_tokens = tokenizer->encode(processed_input);
                    
                    // Get model prediction (in evaluation mode)
                    transformer.set_training(false);
                    Matrix test_hidden = transformer.forward(test_tokens, test_input, *tokenizer);
                    Matrix pred_logits = lm_head->project_to_vocab(test_hidden);
                    transformer.set_training(true);  // Set back to training mode
                    
                    // Show the top predictions
                    std::cout << "\n=== Batch " << batch << " Predictions for '" << test_input << "' ===\n";
                    Utils::print_top_predictions(pred_logits, *tokenizer, transformer, 5);
                    std::cout << "================================================\n";

                    // Test additional queries
                    std::vector<std::string> additional_queries = {
                        "The weather is",
                        "I want to",
                        "The cat",
                        "She likes to"
                    };

                    for (const auto& query : additional_queries) {
                        processed_input = query;
                        tokenizer->preprocess_text(processed_input);
                        test_tokens = tokenizer->encode(processed_input);
                        
                        transformer.set_training(false);
                        test_hidden = transformer.forward(test_tokens, query, *tokenizer);
                        pred_logits = lm_head->project_to_vocab(test_hidden);
                        transformer.set_training(true);
                        
                        std::cout << "\n=== Batch " << batch << " Predictions for '" << query << "' ===\n";
                        Utils::print_top_predictions(pred_logits, *tokenizer, transformer, 5);
                        std::cout << "================================================\n";
                    }
                }

                // Update iteration counter at the end of each epoch
                if (batch == num_batches - 1) {
                    completed_iterations++;
                    if (completed_iterations == MAX_OUTPUT_ITERATIONS) {
                        std::cout << "\nReached " << MAX_OUTPUT_ITERATIONS << " iterations. Suppressing further output.\n";
                    }
                }

                // Print progress and metrics every 10 batches
                if ((batch + 1) % 10 == 0 || batch + 1 == num_batches) {
                    std::cout << "\rBatch " << batch + 1 << "/" << num_batches << " in epoch "
                              << epoch + 1 << " (Loss: " << batch_loss
                              << ", Avg Loss: " << epoch_loss / (batch + 1)
                              << ", LR: " << learning_rate << ")" << std::flush;

                    // Print performance metrics
                    metrics.print_metrics();
                }

                // In the training loop, after processing each batch
                for (const auto& tokens : input_batch) {
                    lm_head->update_token_frequencies(tokens);
                }
            }

            std::cout << "\nCompleted epoch " << epoch + 1 << "/" << config.num_epochs
                      << " (Loss: " << epoch_loss / num_batches << ")" << std::endl;

            // Save checkpoint
            if ((epoch + 1) % checkpoint_frequency == 0) {
                std::cout << "Attempting to save checkpoint to: " << save_directory << "/"
                          << model_name << std::endl;

                // Verify directory exists and is writable
                if (!std::filesystem::exists(save_directory)) {
                    std::cout << "Creating directory: " << save_directory << std::endl;
                    if (!std::filesystem::create_directories(save_directory)) {
                        std::cerr << "Failed to create directory: " << save_directory << std::endl;
                        // Don't exit, just skip checkpoint
                        continue;
                    }
                }

                // Try to save
                if (!model_saver.saveCheckpoint(transformer, save_directory, model_name, 
                                                 global_step, epoch_loss)) {
                    std::cerr << "Failed to save checkpoint, but continuing training" << std::endl;
                    // Don't exit, just continue training
                }
            }

            // Test prediction on a sample input
            if ((epoch + 1) % 2 == 0) {
                std::cout << "\nTesting generation with " 
                          << (config.tokenizer.use_subword ? "subword" : "regular") 
                          << " tokenization:" << std::endl;
                
                // Test a simple input
                std::string test_input = "I go to";
                std::cout << "\n=== Processing prompt: '" << test_input << "' ===" << std::endl;
                
                // Preprocess input
                std::string processed_input = test_input;
                tokenizer->preprocess_text(processed_input);
                std::vector<int> test_tokens = tokenizer->encode(processed_input);
                
                // Get model prediction
                Matrix test_hidden = transformer.forward(test_tokens, "", *tokenizer);
                Matrix logits = lm_head->project_to_vocab(test_hidden);
                
                // For single token prediction, we don't need beam search
                // Just show the top predictions
                std::cout << "\nTop Predictions:\n";
                Utils::print_top_predictions(logits, *tokenizer, transformer, 5);
            }

            if ((epoch + 1) % 5 == 0) { 
                // Cache clearing removed since TiktokenTokenizer doesn't use caching
            }

            // Run validation every 3 epochs
            if ((epoch + 1) % 3 == 0) { // Validate every 3 epochs
                std::cout << "\nRunning validation after epoch " << (epoch + 1) << "...\n";
                float validation_loss =
                    Utils::evaluate_validation(transformer, *tokenizer, validation_data);
                
                // Add pattern completion evaluation
                auto pattern_metrics = Utils::evaluate_pattern_completion(transformer, *tokenizer, validation_data);
                
                std::cout << "Pattern Completion Metrics:\n"
                          << "  Pattern Accuracy: " << (pattern_metrics.pattern_accuracy * 100) << "%\n"
                          << "  Destination Accuracy: " << (pattern_metrics.destination_accuracy * 100) << "%\n"
                          << "  Common mistakes:\n";
                
                for (const auto& mistake : pattern_metrics.common_mistakes) {
                    std::cout << "    - \"" << mistake << "\"\n";
                }
            }
        }

        std::cout << "\nTraining completed!\n";

        // Final prediction test
        std::cout << "\nFinal generation test with " 
                  << (config.tokenizer.use_subword ? "subword" : "regular") 
                  << " tokenization:" << std::endl;
        
        // Test a simple input
        std::string test_input = "I go to";
        std::cout << "\n=== Processing prompt: '" << test_input << "' ===" << std::endl;
        
        // Preprocess input
        std::string processed_input = test_input;
        tokenizer->preprocess_text(processed_input);
        std::vector<int> test_tokens = tokenizer->encode(processed_input);
        
        // Get model prediction
        transformer.set_training(false);  // Set to evaluation mode
        Matrix test_hidden = transformer.forward(test_tokens, "", *tokenizer);
        Matrix logits = lm_head->project_to_vocab(test_hidden);
        
        // Show the top predictions
        std::cout << "\nTop Predictions:\n";
        Utils::print_top_predictions(logits, *tokenizer, transformer, 5);

        // Create directories if they don't exist
        std::filesystem::create_directories(save_directory);

        // Save the trained model
        std::cout << "\nSaving final model to " << save_directory << "/" << model_name << "...\n";
        bool save_success = model_saver.saveModel(transformer, save_directory, model_name);
        if (save_success) {
            std::cout << "Successfully saved model to " + save_directory + "/" + model_name
                      << std::endl;
            std::cout << "Model saved successfully!\n";
        } else {
            std::cout << "Failed to save model to " + save_directory + "/" + model_name
                      << std::endl;
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

#ifdef CUDA_AVAILABLE
    cleanup_cuda(); // Cleanup at program end
#endif
    logger.stopLogging();
    std::cout << "exiting main" << std::endl;
    return 0;
}