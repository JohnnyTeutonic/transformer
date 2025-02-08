#include "../include/main.hpp"
#include <fstream>
#include <nlohmann/json.hpp>
#include <random>
#include "../include/tiktoken_tokenizer.hpp"
#include "../include/utils.hpp"
#include "../include/phrase_analysis.hpp"
#include "../include/training/training.hpp"  // Include unified training header
#include "../include/hyperparameter_tuner.hpp"
#include "../include/count_vocabulary.hpp"
#include "../include/cuda/matrix_ops.cuh"
#include <iostream>
#include <chrono>
#include <ctime>
#include <mutex>
#include "../include/debug.hpp"

// Add necessary forward declarations and structures
std::unique_ptr<TiktokenTokenizer> tokenizer;
PerformanceMetrics metrics;

// Configuration constants
const float INITIAL_LEARNING_RATE = 0.001f;
float learning_rate = INITIAL_LEARNING_RATE;
float prev_loss = std::numeric_limits<float>::max();
size_t global_step = 0;

// Add training components with proper types
TrainingStateManagerPtr training_manager;
TrainingMonitorPtr training_monitor;

// Initialize tokenizer function
bool initialize_tokenizer(TransformerConfig& config) {
    std::cout << "\nInitializing tokenizer..." << std::endl;
    tokenizer = std::make_unique<TiktokenTokenizer>();

    try {
        // Build vocabulary from training data
        tokenizer->build_vocabulary_from_file("../data/training_pairs.txt");
        // Update config with actual vocabulary size
        config.vocab_size = tokenizer->vocab_size();
        std::cout << "Initialized tokenizer with vocabulary size: " << config.vocab_size << std::endl;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error initializing tokenizer: " << e.what() << std::endl;
        return false;
    }
}

// Data structure for preprocessing
struct Data {
    std::vector<std::vector<float>> samples;
    std::vector<int> labels;
};

// Add debug logging to the generate_predictions function
void generate_predictions(Transformer& transformer, const std::string& input_text, TiktokenTokenizer* tokenizer) {
    if (!tokenizer) {
        std::cerr << "Error: TiktokenTokenizer is null" << std::endl;
        return;
    }

    // Tokenize input
    std::vector<int> input_tokens = tokenizer->encode(input_text);
    
    // Generate prediction
    std::vector<int> output_tokens = transformer.generate(input_tokens);
    
    // Decode and print result
    std::string output_text = tokenizer->decode(output_tokens);
    std::cout << "Input: " << input_text << std::endl;
    std::cout << "Output: " << output_text << std::endl;
}

// Add this function before the main training loop
void reinitialize_batch_weights(Transformer& transformer, const TransformerConfig& config, size_t global_step) {
    // Get a random seed based on time and batch number
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Dynamic temperature scaling
    float temperature = 1.0f + (5.0f * std::exp(-global_step / 500.0f));  // Starts at 6.0, decays to 1.0
    
    // Multiple scales for different layers
    float attention_scale = 0.15f * std::exp(-global_step / 2000.0f);  // Slower decay for attention
    float ffn_scale = 0.1f * std::exp(-global_step / 1000.0f);        // Faster decay for feed-forward
    float output_scale = 0.05f * std::exp(-global_step / 800.0f);     // Even faster for output layer
    
    // Minimum scales to maintain exploration
    attention_scale = std::max(0.02f, attention_scale);
    ffn_scale = std::max(0.01f, ffn_scale);
    output_scale = std::max(0.005f, output_scale);
    
    // Get all parameters that need reinitialization
    auto& params = transformer.parameters();
    
    // Reinitialize each parameter matrix with controlled randomness
    for (size_t p = 0; p < params.size(); p++) {
        Matrix& current_param = params[p];
        const size_t rows = current_param.rows();
        const size_t cols = current_param.cols();
        
        // Choose scale based on layer type (inferred from matrix dimensions)
        float scale;
        if (cols == config.hidden_size) {
            scale = attention_scale;  // Attention layers
        } else if (cols == config.intermediate_size) {
            scale = ffn_scale;        // Feed-forward layers
        } else {
            scale = output_scale;     // Output layers
        }
        
        // Process each matrix in parallel
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                // Thread-local random generators
                std::mt19937 local_gen(rd() + i * cols + j);  // Unique seed per element
                
                // Add temperature-based sampling
                std::normal_distribution<float> dist(0.0f, scale * temperature);
                float perturbation = dist(local_gen);
                
                // Current weight magnitude affects perturbation
                float current_value = std::abs(current_param(i, j));
                float adaptive_scale = scale / (1.0f + current_value * temperature);
                
                // Apply perturbation with probability decay
                float apply_prob = std::exp(-global_step / 3000.0f);  // Probability of applying perturbation
                if (std::uniform_real_distribution<float>(0.0f, 1.0f)(local_gen) < apply_prob) {
                    current_param(i, j) += perturbation * adaptive_scale;
                }
                
                // Occasionally flip sign of small weights to explore different patterns
                if (std::abs(current_param(i, j)) < 0.01f && 
                    std::uniform_real_distribution<float>(0.0f, 1.0f)(local_gen) < 0.1f) {
                    current_param(i, j) *= -1.0f;
                }
            }
        }
    }
}

void test_model_predictions(Transformer& transformer, std::unique_ptr<TiktokenTokenizer>& tokenizer) {
    std::vector<std::string> test_queries = {
        "The cat begins to",          // Simple action start
        "The old house looks very",   // Descriptive context
        "I want to quickly",          // Personal intention
        "The bright sun makes me feel", // Sensory experience
        "The computer starts to",     // Technical context
        "The musician skillfully",    // Professional action
        "The food tastes extremely",  // Sensory description
        "The children love to",       // Group action
        "The storm makes the trees",  // Nature action
        "The painting appears"        // Art observation
    };

    for (const auto& test_input : test_queries) {
        std::cout << "\n=== Testing Query: \"" << test_input << "\" ===\n";
        
        std::string processed_input = test_input;
        tokenizer->preprocess_text(processed_input);
        std::vector<int> test_tokens = tokenizer->encode(processed_input);
        
        // Get model prediction
        transformer.set_training(false);  // Set to evaluation mode
        Matrix logits = transformer.forward(test_tokens, "", *tokenizer);  // Already includes LM head projection

        // Show the top predictions
        std::cout << "\nTop Predictions:\n";
        Utils::print_top_predictions(logits, *tokenizer, transformer, 5);
    }
}

// Add debug logging to main function
int main(int argc, char* argv[]) {
    try {
        debug::init_logging();
        debug::log_message("Starting transformer application", "INFO");
        
        // Initialize CUDA at program startup
        cuda::initialize_cuda();
        std::cout << "CUDA initialized successfully" << std::endl;

        std::cout << "entering main" << std::endl;
        Logger& logger = Logger::getInstance();
        logger.startLogging();

        // Initialize random number generation
        Utils::initialize_random();
        std::filesystem::path exe_path = std::filesystem::current_path().parent_path();

        // First, count vocabulary size from training and validation files
        std::cout << "\nCounting unique tokens in training and validation files..." << std::endl;
        size_t custom_vocab_size = transformer::VocabularyCounter::countUniqueTokens(
            exe_path.string() + "/data/training_pairs.txt",
            exe_path.string() + "/data/validation_pairs.txt"
        );
        std::cout << "Number of unique tokens found in data files: " << custom_vocab_size << std::endl;

        // Load and update config with the counted vocabulary size
        std::filesystem::path config_path = exe_path / "config" / "transformer_config.json";
        
        // Read the config file
        std::ifstream config_file(config_path);
        if (!config_file.is_open()) {
            throw std::runtime_error("Could not open config file: " + config_path.string());
        }
        
        nlohmann::json config_json;
        config_file >> config_json;
        config_file.close();
        
        // Update vocabulary size in config
        size_t previous_vocab_size = 0;
        if (config_json.contains("vocab_size") && !config_json["vocab_size"].is_null()) {
            previous_vocab_size = config_json["vocab_size"].get<size_t>();
        }
        std::cout << "Previous vocabulary size in config: " << (previous_vocab_size == 0 ? "Not set" : std::to_string(previous_vocab_size)) << std::endl;
        
        config_json["vocab_size"] = custom_vocab_size;
        std::cout << "Updated vocabulary size in config to: " << custom_vocab_size << std::endl;
        
        // Write updated config back to file
        std::ofstream output_config_file(config_path);
        if (!output_config_file.is_open()) {
            throw std::runtime_error("Could not open config file for writing: " + config_path.string());
        }
        output_config_file << config_json.dump(4);
        output_config_file.close();

        // Now load the updated config for transformer initialization
        TransformerConfig config = Utils::load_config(config_path.string());
        
        // Initialize random seed using hardware entropy
        std::random_device rd;
        std::mt19937 gen(rd());
        
        // Create a seed sequence using multiple entropy sources
        std::vector<std::uint32_t> entropy{
            static_cast<std::uint32_t>(std::time(nullptr)),
            rd(), rd(), rd(), rd()
        };
        std::seed_seq seq(entropy.begin(), entropy.end());
        
        // Create a new generator with the seed sequence
        std::mt19937 global_gen(seq);
        
        // Store the generator in a global context or pass it where needed
        Utils::set_random_generator(global_gen);

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
        if (!initialize_tokenizer(config)) {
            std::cerr << "Failed to initialize tokenizer" << std::endl;
            return 1;
        }

        // Initialize model with updated config
        std::cout << "\nInitializing transformer with custom vocabulary size: " << config.vocab_size << std::endl;
        Transformer transformer(config);
        std::cout << "\nTransformer initialized with language model head" << std::endl << std::flush;
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

        // Load validation data
        auto validation_data = Utils::load_validation_data();
        std::cout << "Loaded " << validation_data.size() << " validation examples\n";

        // Combine training and validation data for cross-validation
        std::vector<std::pair<std::string, std::string>> all_data;
        all_data.insert(all_data.end(), training_pairs.begin(), training_pairs.end());
        all_data.insert(all_data.end(), validation_data.begin(), validation_data.end());
        
        // Perform initial cross-validation to establish baseline
        std::cout << "\n=== Performing Initial Baseline Cross-Validation ===" << std::endl;
        float initial_cv_loss = Utils::perform_cross_validation(
            transformer, 
            *tokenizer, 
            all_data
        );
        std::cout << "=== Initial baseline cross-validation loss: " << initial_cv_loss << " ===" << std::endl;

        // Update any hardcoded token references
        int pad_id = 0;    // UNK_ID
        std::cout << "pad_id: " << pad_id << std::endl;
        int unk_id = 0;    // UNK_ID
        std::cout << "unk_id: " << unk_id << std::endl;
        int bos_id = 0;    // We don't use these in our simple tokenizer
        std::cout << "bos_id: " << bos_id << std::endl;
        int eos_id = 0;    // We don't use these in our simple tokenizer
        std::cout << "eos_id: " << eos_id << std::endl;
        int mask_id = 0;   // We don't use these in our simple tokenizer
        std::cout << "mask_id: " << mask_id << std::endl;
        std::cout << "epochs: " << config.num_epochs << std::endl;

        float best_cv_loss = initial_cv_loss;
        size_t epochs_without_improvement = 0;
        const size_t PATIENCE = 3;

        // After loading data but before training loop
        std::cout << "\nStarting hyperparameter tuning phase..." << std::endl;
        
        // Initialize hyperparameter tuner
        HyperparameterRanges ranges;  // Now defined
        HyperparameterTuner tuner(ranges, config);  // Pass config to constructor
        
        // Run hyperparameter tuning
        std::cout << "Running hyperparameter tuning with " << training_pairs.size() 
                  << " training examples..." << std::endl;
        auto tuning_results = tuner.tune(training_pairs, *tokenizer);
        
        // Get best configuration
        auto best_config = tuner.get_best_config();
        
        // Save tuning results
        std::string tuning_results_path = save_directory + "/tuning_results.json";
        tuner.save_results(tuning_results_path);
        
        std::cout << "\nHyperparameter tuning complete!" << std::endl;
        std::cout << "Best configuration achieved validation loss: " 
                  << tuning_results[0].mean_validation_loss << std::endl;
        
        // Update transformer config with best hyperparameters
        config = best_config.to_transformer_config();
        
        // Reinitialize transformer with best config
        transformer = Transformer(config);
        std::cout << "Reinitialized transformer with best hyperparameters" << std::endl;
        
        // Update training parameters from best config
        const float initial_lr = best_config.initial_lr;
        const float peak_lr = best_config.peak_lr;
        const size_t warmup_steps = best_config.warmup_steps;
        const float decay_factor = best_config.decay_factor;
        const float gradient_clip_threshold = best_config.gradient_clip_threshold;
        const size_t early_stopping_patience = best_config.early_stopping_patience;
        const float early_stopping_threshold = best_config.early_stopping_threshold;
        
        std::cout << "\nStarting main training with best hyperparameters..." << std::endl;

        // After loading config but before training loop
        std::cout << "\nStarting training with:"
                  << "\n- Batch size: " << config.training.batch_size
                  << "\n- Number of epochs: " << config.training.num_epochs
                  << "\n- Learning rate: " << config.training.learning_rate.initial_lr
                  << std::endl;

        // Calculate total batches
        size_t total_batches = (training_pairs.size() + config.training.batch_size - 1) / config.training.batch_size;

        // Training loop
        for (size_t epoch = 0; epoch < config.training.num_epochs; epoch++) {
            std::cout << "\nEpoch " << epoch + 1 << "/" << config.training.num_epochs << std::endl;
            float epoch_loss = 0.0f;
            
            // Process each batch
            for (size_t batch = 0; batch < total_batches; batch++) {
                size_t start_idx = batch * config.training.batch_size;
                size_t end_idx = std::min(start_idx + config.training.batch_size, training_pairs.size());
                size_t current_batch_size = end_idx - start_idx;

                // Create batch with validation
                std::vector<std::vector<int>> input_batch;
                std::vector<std::vector<int>> target_batch;

                // Fill and validate batch with padding
                bool batch_valid = true;
                for (size_t j = start_idx; j < end_idx; ++j) {
                    const auto& [input_str, target_str] = training_pairs[j];
                    std::string processed_input = input_str;
                    std::string processed_target = target_str;
                    tokenizer->preprocess_text(processed_input);
                    tokenizer->preprocess_text(processed_target);
                    std::vector<int> input_tokens = tokenizer->encode(processed_input);
                    std::vector<int> target_tokens = tokenizer->encode(processed_target);
                    
                    if (!Utils::validate_input_sequence(input_tokens, tokenizer->vocab_size()) ||
                        !Utils::validate_input_sequence(target_tokens, tokenizer->vocab_size())) {
                        batch_valid = false;
                        break;
                    }
                    input_batch.push_back(input_tokens);
                    target_batch.push_back(target_tokens);
                }

                if (!batch_valid) continue;

                // Forward pass and compute loss
                Matrix logits = transformer.forward(input_batch[0], "", *tokenizer);
                float batch_loss = Utils::compute_batch_loss(logits, target_distribution, *tokenizer);

                // Update parameters
                transformer.backward(loss_gradients, input_batch[0], current_lr);
                transformer.update_parameters(current_lr);

                // Print progress every 10 batches
                if ((batch + 1) % 10 == 0) {
                    std::cout << "\rBatch " << batch + 1 << "/" << total_batches 
                              << " (Loss: " << batch_loss << ")" << std::flush;
                }
            }

            std::cout << "\nCompleted epoch " << epoch + 1 << "/" << config.training.num_epochs
                      << " (Loss: " << epoch_loss / total_batches << ")" << std::endl;

            // Run validation and make predictions only at end of epoch
            if ((epoch + 1) % 2 == 0) {
                std::cout << "\nRunning validation..." << std::endl;
                float validation_loss = Utils::evaluate_validation(transformer, *tokenizer, validation_data);
                
                // Make a few test predictions
                std::cout << "\nTest predictions:" << std::endl;
                Utils::generate_predictions(transformer, "I go to the", tokenizer.get());
                Utils::generate_predictions(transformer, "The weather is", tokenizer.get());
            }

            // Save regular checkpoint
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
                if (!model_saver.saveCheckpoint(transformer, save_directory, model_name, epoch + 1,
                                                epoch_loss)) {
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
                transformer.set_training(false);  // Set to evaluation mode
                Matrix logits = transformer.forward(test_tokens, "", *tokenizer);  // Already includes LM head projection
                
                // Show the top predictions
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

                // Log validation results
                std::cout << "Epoch " << (epoch + 1) << " Validation Loss: " << validation_loss
                          << std::endl;
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
        Matrix logits = transformer.forward(test_tokens, "", *tokenizer);  // Already includes LM head projection

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

        // Test model predictions
        test_model_predictions(transformer, tokenizer);

        // Cleanup CUDA before exit
        cuda::cleanup_cuda();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}