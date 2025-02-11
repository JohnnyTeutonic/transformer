#include "../include/main.hpp"
#include "../include/scope_logger.hpp"  // Add scope logger header
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
PerformanceMetrics metrics;

// Configuration constants
const float INITIAL_LEARNING_RATE = 0.001f;
float learning_rate = INITIAL_LEARNING_RATE;
float prev_loss = std::numeric_limits<float>::max();
size_t global_step = 0;

// Add training components with proper types
TrainingStateManagerPtr training_manager;
TrainingMonitorPtr training_monitor;

// Data structure for preprocessing
struct Data {
    std::vector<std::vector<float>> samples;
    std::vector<int> labels;
};

// Add debug logging to the generate_predictions function
void generate_predictions(Transformer& transformer, const std::string& input_text, std::shared_ptr<TiktokenTokenizer> tokenizer) {
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

// Update the test_model_predictions function signature to accept shared_ptr
void test_model_predictions(Transformer& transformer, std::shared_ptr<TiktokenTokenizer> tokenizer) {
    if (!tokenizer) {
        std::cerr << "Error: TiktokenTokenizer is null" << std::endl;
        return;
    }
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
        // Initialize logging but don't enable scope logging by default
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

        // Now load the updated config for transformer initialization
        TransformerConfig config;
        std::string config_path = exe_path.string() + "/config/transformer_config.json";
        std::cout << "Loading configuration from: " << config_path << std::endl;
        config.load_from_json(config_path);
        
        // Print loaded configuration for verification
        std::cout << "\nLoaded configuration:" << std::endl;
        std::cout << "- Training:" << std::endl;
        std::cout << "  - Samples per iteration: " << config.training.samples_per_iteration << std::endl;
        std::cout << "  - Num epochs: " << config.training.num_epochs << std::endl;
        std::cout << "  - Tuning enabled: " << std::boolalpha << config.training.tuning.enabled << std::endl;
        std::cout << "  - Tuning trials: " << config.training.tuning.num_trials << std::endl;

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
        std::shared_ptr<TiktokenTokenizer> tokenizer = std::make_shared<TiktokenTokenizer>();
        try {
            tokenizer->build_vocabulary_from_file("../data/training_pairs.txt");
            std::cout << "Successfully initialized tokenizer" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Failed to initialize tokenizer: " << e.what() << std::endl;
            return 1;
        }

        // Initialize model with updated config
        std::cout << "\nInitializing transformer..." << std::endl;
        Transformer transformer(config, tokenizer);  // Pass shared_ptr tokenizer
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
                                std::stoul(epoch_str) * (training_pairs.size() / config.training.samples_per_iteration);
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
        
        float best_cv_loss = std::numeric_limits<float>::max();
        size_t epochs_without_improvement = 0;
        const size_t PATIENCE = 3;

        // Initialize progress state
        debug::progress_state.reset();

        // Only enter tuning stage if tuning is enabled
        if (config.training.tuning.enabled) {
            std::cout << "Entering tuning stage" << std::endl;
            debug::progress_state.current_stage = debug::ProgressState::Stage::TUNING;
            
            // Initialize hyperparameter tuner
            HyperparameterRanges ranges;
            HyperparameterTuner tuner(ranges, config, tokenizer);
            
            // Run tuning and get results
            auto tuning_results = tuner.tune(training_pairs, *tokenizer);
            
            // Only process results if we got any
            if (!tuning_results.empty()) {
                // Process tuning results...
                auto best_config = tuner.get_best_config();
                std::string tuning_results_path = save_directory + "/tuning_results.json";
                tuner.save_results(tuning_results_path);
                config = best_config.to_transformer_config();
                transformer = Transformer(config, tokenizer);
                debug::log_message("Reinitialized transformer with best hyperparameters", "INFO");
            }
        }

        // Enter training stage
        debug::progress_state.current_stage = debug::ProgressState::Stage::TRAINING;
        std::cout << "\nStarting main training..." << std::endl;

        // Calculate total iterations needed
        size_t total_iterations = (training_pairs.size() + config.training.samples_per_iteration - 1) 
                                / config.training.samples_per_iteration;

        std::cout << "\nStarting training with:"
                  << "\n- Samples per iteration: " << config.training.samples_per_iteration
                  << "\n- Total samples: " << training_pairs.size()
                  << "\n- Total iterations: " << total_iterations
                  << "\n- Number of epochs: " << config.training.num_epochs
                  << "\n- Learning rate: " << config.training.learning_rate.initial_lr
                  << std::endl;

        // Extract learning rate parameters from config
        const float initial_lr = config.training.learning_rate.initial_lr;
        const float peak_lr = config.training.learning_rate.peak_lr;
        const size_t warmup_steps = config.training.learning_rate.warmup_steps;
        const float decay_factor = config.training.learning_rate.decay_factor;
        float current_lr = initial_lr;
        size_t global_step = 0;  // Initialize global step counter

        // Initialize training monitor
        auto training_monitor = std::make_unique<TrainingMonitor>();

        // Training loop
        auto training_start = std::chrono::high_resolution_clock::now();
        size_t total_steps = config.training.num_epochs * total_iterations;
        
        debug::progress_state.reset();  // Reset progress state at start of training
        
        for (size_t epoch = 0; epoch < config.training.num_epochs; epoch++) {
            auto epoch_start = std::chrono::high_resolution_clock::now();
            std::cout << "\nEpoch " << epoch + 1 << "/" << config.training.num_epochs << std::endl;
            float epoch_loss = 0.0f;
            
            // Process samples in groups
            for (size_t iter = 0; iter < total_iterations; iter++) {
                auto iteration_start = std::chrono::high_resolution_clock::now();
                
                // Calculate range of samples for this iteration
                size_t start_idx = iter * config.training.samples_per_iteration;
                size_t end_idx = std::min(start_idx + config.training.samples_per_iteration, training_pairs.size());
                size_t samples_this_iteration = end_idx - start_idx;

                float iteration_loss = 0.0f;

                // Process each sample in this iteration sequentially
                for (size_t sample_idx = start_idx; sample_idx < end_idx; ++sample_idx) {
                    const auto& [input_str, target_str] = training_pairs[sample_idx];
                    
                    // Process single sample
                    std::string processed_input = input_str;
                    std::string processed_target = target_str;
                    tokenizer->preprocess_text(processed_input);
                    tokenizer->preprocess_text(processed_target);
                    
                    std::vector<int> input_tokens = tokenizer->encode(processed_input);
                    std::vector<int> target_tokens = tokenizer->encode(processed_target);
                    
                    if (!Utils::validate_input_sequence(input_tokens, tokenizer->vocab_size()) ||
                        !Utils::validate_input_sequence(target_tokens, tokenizer->vocab_size())) {
                        std::cout << "Skipping invalid sample " << sample_idx << std::endl;
                        continue;
                    }

                    // Forward pass for single sample
                    Matrix logits = transformer.forward(input_tokens, "", *tokenizer);

                    // Create target distribution for single sample
                    Matrix target_distribution = Utils::create_batch_target_distribution(
                        {target_tokens}, *tokenizer, tokenizer->vocab_size(), 1
                    );

                    // Compute loss for this sample
                    float sample_loss = Utils::compute_batch_loss(logits, target_distribution, *tokenizer);
                    iteration_loss += sample_loss;

                    // Backward pass for single sample
                    transformer.backward(logits, target_distribution, current_lr);
                }

                // Average loss over samples in this iteration
                iteration_loss /= samples_this_iteration;
                epoch_loss += iteration_loss;

                // Update training monitor with metrics
                RunningStatistics grad_stats;
                grad_stats.mean = iteration_loss;  // Use loss as a proxy for gradient statistics
                grad_stats.variance = 0.0f;        // Initialize to 0 for now

                // Access the gradient norm from the transformer
                extern float last_grad_norm;  // Declare the external variable
                Matrix gradient_matrix(1, 1);
                gradient_matrix(0, 0) = last_grad_norm;  // Use the actual gradient norm

                TrainingMetrics metrics(
                    iteration_loss,           // Current loss
                    gradient_matrix,          // Use actual gradient norm matrix
                    epoch,                    // Current epoch
                    iter,                     // Current step
                    0.0f,                    // Let the monitor compute the trend
                    grad_stats               // Basic statistics
                );
                training_monitor->log_metrics(metrics);

                // Update progress state
                debug::progress_state.update_training(
                    epoch, config.training.num_epochs,
                    iter, total_iterations,
                    epoch_loss / (iter + 1)
                );

                // Calculate timing and progress
                auto iteration_end = std::chrono::high_resolution_clock::now();
                auto iteration_duration = std::chrono::duration_cast<std::chrono::milliseconds>(iteration_end - iteration_start);
                
                // Print progress every few iterations
                if ((iter + 1) % 5 == 0) {
                    float progress = (float)(epoch * total_iterations + iter + 1) / (config.training.num_epochs * total_iterations);
                    std::cout << "\rProgress: " << std::fixed << std::setprecision(1) << (progress * 100) << "% "
                              << "Iteration " << iter + 1 << "/" << total_iterations 
                              << " (Loss: " << iteration_loss 
                              << ", Samples: " << samples_this_iteration
                              << ", Time: " << iteration_duration.count() << "ms)" 
                              << std::flush;
                }
            }

            // Calculate epoch timing
            auto epoch_end = std::chrono::high_resolution_clock::now();
            auto epoch_duration = std::chrono::duration_cast<std::chrono::seconds>(epoch_end - epoch_start);
            
            std::cout << "\nCompleted epoch " << epoch + 1 << "/" << config.training.num_epochs
                      << " (Loss: " << epoch_loss / total_iterations 
                      << ", Time: " << epoch_duration.count() << "s)" << std::endl;

            // Run quick validation check at end of epoch
            if ((epoch + 1) % 2 == 0) {
                std::cout << "\nRunning validation..." << std::endl;
                auto val_start = std::chrono::high_resolution_clock::now();
                
                float validation_loss = Utils::evaluate_validation(transformer, *tokenizer, validation_data);
                
                auto val_end = std::chrono::high_resolution_clock::now();
                auto val_duration = std::chrono::duration_cast<std::chrono::seconds>(val_end - val_start);
                
                std::cout << "Validation complete (Loss: " << validation_loss 
                          << ", Time: " << val_duration.count() << "s)" << std::endl;
                
                // Make a few quick test predictions
                Utils::generate_predictions(transformer, "I go to the", tokenizer);
                Utils::generate_predictions(transformer, "The weather is", tokenizer);
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
            if ((epoch + 1) % 3 == 0) {
                std::cout << "\nRunning validation after epoch " << (epoch + 1) << "...\n";
                float validation_loss = Utils::evaluate_validation(transformer, *tokenizer, validation_data);

                // Update progress state for cross-validation
                debug::progress_state.update_cross_validation(
                    0, 1,  // Single fold for regular validation
                    epoch, config.training.num_epochs,
                    validation_loss
                );

                // Log validation results
                std::cout << "Epoch " << (epoch + 1) << " Validation Loss: " << validation_loss
                          << std::endl;
            }
        }

        // Reset progress state at end of training
        debug::progress_state.reset();

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