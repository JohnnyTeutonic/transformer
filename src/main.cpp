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
        Matrix test_hidden = transformer.forward(test_tokens, "", *tokenizer);
        Matrix logits = transformer.get_lm_head()->forward(test_hidden);
        
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

                // Find maximum sequence length in this batch
                size_t max_seq_len = 0;
                for (size_t j = start_idx; j < end_idx; ++j) {
                    const auto& [input_str, target_str] = training_pairs[j];
                    std::vector<int> input_tokens = tokenizer->encode(input_str);
                    std::vector<int> target_tokens = tokenizer->encode(target_str);
                    // Consider both input and target sequence lengths
                    max_seq_len = std::max({max_seq_len, input_tokens.size(), target_tokens.size()});
                }
                std::cout << "\n=== Processing Batch " << batch + 1 << " ===\n";

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
                        input_tokens.push_back(0);  // Use 0 for padding
                    }
                    while (target_tokens.size() < max_seq_len) {  // Add padding for target tokens
                        target_tokens.push_back(0);  // Use 0 for padding
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

                // Create target distribution for entire batch using only valid sequences
                Matrix target_distribution = Utils::create_batch_target_distribution(
                    valid_target_batch, *tokenizer, custom_vocab_size, max_seq_len);

                // Process the batch as a single sequence
                std::vector<int> flattened_batch;
                flattened_batch.reserve(valid_input_batch.size() * max_seq_len);

                // Flatten the batch into a single sequence
                for (const auto& sequence : valid_input_batch) {
                    flattened_batch.insert(flattened_batch.end(), sequence.begin(), sequence.end());
                }
                std::cout << "Flattened batch size: " << flattened_batch.size() << " tokens\n";

                // Forward pass through the model
                transformer.set_tokenizer(tokenizer.get());
                Matrix hidden_states = transformer.forward(flattened_batch, "", *tokenizer);

                // Project hidden states to vocabulary space using LM head
                Matrix logits = transformer.get_lm_head()->forward(hidden_states);

                // Compute batch loss
                float batch_loss = Utils::compute_batch_loss(logits, target_distribution, *tokenizer);

                // Compute gradient norm
                Matrix loss_gradients = Utils::compute_loss_gradient(logits, target_distribution);
                float grad_norm = 0.0f;
                for (size_t i = 0; i < loss_gradients.size(); i++) {
                    grad_norm += loss_gradients.data()[i] * loss_gradients.data()[i];
                }
                grad_norm = std::sqrt(grad_norm);

                // Calculate learning rate using tuned parameters
                float current_lr;
                if (global_step < warmup_steps) {
                    // Linear warmup
                    current_lr = initial_lr + (peak_lr - initial_lr) * (float)global_step / warmup_steps;
                } else {
                    // Cosine decay with tuned decay factor
                    float steps_after_warmup = global_step - warmup_steps;
                    float decay = std::pow(decay_factor, steps_after_warmup / 1000.0f);  // Decay every 1000 steps
                    current_lr = peak_lr * decay;
                }
                
                // Ensure learning rate doesn't go below initial_lr
                current_lr = std::max(current_lr, initial_lr);

                // Open loss log file in append mode
                std::ofstream loss_log("../build/loss.log", std::ios::app);
                if (loss_log.is_open()) {
                    // Get current timestamp
                    auto now = std::chrono::system_clock::now();
                    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
                    std::string timestamp = std::ctime(&now_time);
                    timestamp.pop_back(); // Remove trailing newline
                    
                    // Write detailed metrics to loss log
                    loss_log << "[" << timestamp << "] "
                            << "Epoch: " << (epoch + 1) << "/" << config.training.num_epochs << ", "
                            << "Batch: " << (batch + 1) << "/" << total_batches << ", "
                            << "Batch Loss: " << batch_loss << ", "
                            << "Avg Epoch Loss: " << (epoch_loss / (batch + 1)) << ", "
                            << "Gradient Norm: " << grad_norm << ", "
                            << "Learning Rate: " << current_lr << ", "
                            << "Global Step: " << global_step << std::endl;
                    
                    // If this is the first batch of the first epoch, write header
                    if (epoch == 0 && batch == 0) {
                        loss_log << "\nTraining Configuration:\n"
                                << "- Batch Size: " << config.training.batch_size << "\n"
                                << "- Initial Learning Rate: " << initial_lr << "\n"
                                << "- Peak Learning Rate: " << peak_lr << "\n"
                                << "- Warmup Steps: " << warmup_steps << "\n"
                                << "- Decay Factor: " << decay_factor << "\n"
                                << "- Gradient Clip Threshold: " << gradient_clip_threshold << "\n"
                                << "- Early Stopping Patience: " << early_stopping_patience << "\n"
                                << "- Early Stopping Threshold: " << early_stopping_threshold << "\n\n";
                    }
                    
                    // Add validation metrics when available
                    if ((batch + 1) % config.training.cross_validation.validation_frequency == 0) {
                        loss_log << "[" << timestamp << "] "
                                << "=== Validation Metrics ===\n"
                                << "Current Validation Loss: " << batch_loss << "\n"
                                << "Best Validation Loss: " << best_cv_loss << "\n"
                                << "Epochs Without Improvement: " << epochs_without_improvement << "\n\n";
                    }
                    
                    loss_log.close();
                }

                // Apply gradient clipping
                if (grad_norm > gradient_clip_threshold) {
                    float scale = gradient_clip_threshold / (grad_norm + 1e-6f);
                    for (size_t i = 0; i < loss_gradients.size(); i++) {
                        loss_gradients.data()[i] *= scale;
                    }
                    grad_norm = gradient_clip_threshold;
                }

                // Backward pass and parameter update
                transformer.backward(loss_gradients, flattened_batch, current_lr);
                transformer.update_parameters(current_lr);

                // Print training statistics
                std::cout << "\nTraining Statistics (Batch " << batch + 1 << "):" << std::endl;
                std::cout << "Batch Loss: " << batch_loss << std::endl;
                std::cout << "Gradient Norm: " << grad_norm << std::endl;
                std::cout << "Learning Rate: " << current_lr << std::endl;
                
                // Print progress and metrics every 10 batches
                if ((batch + 1) % 10 == 0 || batch + 1 == total_batches) {
                    std::cout << "\rBatch " << batch + 1 << "/" << total_batches << " in epoch "
                              << epoch + 1 << " (Loss: " << batch_loss
                              << ", Avg Loss: " << epoch_loss / (batch + 1)
                              << ", LR: " << current_lr << ")" << std::flush;

                    // Print performance metrics
                    metrics.print_metrics();
                }

                // Make predictions every 2 batches with clear separation
                if ((batch + 1) % 2 == 0) {
                    std::cout << "\n\n=== Making Predictions After Batch " << (batch + 1) 
                              << " in Epoch " << (epoch + 1) << " ===\n" << std::endl;
                    
                    // Test verb completions
                    std::cout << "\n--- Testing Verb Completions ---" << std::endl;
                    generate_predictions(transformer, "I go to the", tokenizer.get());
                    generate_predictions(transformer, "I want to", tokenizer.get());
                    generate_predictions(transformer, "The code begins to", tokenizer.get());
                    
                    // Test adjective completions
                    std::cout << "\n--- Testing Adjective Completions ---" << std::endl;
                    generate_predictions(transformer, "The weather is", tokenizer.get());
                    generate_predictions(transformer, "His voice sounds", tokenizer.get());
                    generate_predictions(transformer, "The food tastes", tokenizer.get());
                    
                    // Test mixed contexts
                    std::cout << "\n--- Testing Mixed Contexts ---" << std::endl;
                    generate_predictions(transformer, "The students eagerly", tokenizer.get());
                    generate_predictions(transformer, "The ocean waves", tokenizer.get());
                    generate_predictions(transformer, "The chef skillfully", tokenizer.get());
                    
                    std::cout << "\n=== End of Predictions ===\n" << std::endl;
                    std::cout.flush();  // Ensure output is displayed
                }

                // Update tracking variables
                prev_loss = batch_loss;
                epoch_loss += batch_loss;
                global_step++;
                
                metrics.stop_timer("batch_processing");

                // In the training loop, after processing each batch
                for (const auto& tokens : input_batch) {
                    // Add frequency decay for common tokens
                    static const float FREQ_DECAY_RATE = 0.98f;
                    static const float MIN_FREQ = 0.1f;
                    
                    // Get mean frequency for comparison
                    const auto& frequencies = transformer.get_lm_head()->get_token_frequencies();
                    float mean_freq = 0.0f;
                    if (!frequencies.empty()) {
                        mean_freq = std::accumulate(frequencies.begin(), frequencies.end(), 0.0f) / frequencies.size();
                    }
                    
                    // Create a vector of tokens with adjusted frequencies
                    std::vector<int> update_tokens;
                    
                    for (int token : tokens) {
                        if (token < frequencies.size()) {
                            float current_freq = frequencies[token];
                            
                            // Only include tokens that need frequency adjustment
                            if (current_freq > 2.0f * mean_freq) {
                                // Apply decay to very frequent tokens
                                // We'll handle these separately to reduce their frequency
                                continue;
                            } else {
                                // Add tokens that should get frequency boost
                                update_tokens.push_back(token);
                            }
                        }
                    }
                    
                    // Update frequencies for normal tokens
                    if (!update_tokens.empty()) {
                        transformer.get_lm_head()->update_token_frequencies(update_tokens);
                    }
                }
            }

            std::cout << "\nCompleted epoch " << epoch + 1 << "/" << config.training.num_epochs
                      << " (Loss: " << epoch_loss / total_batches << ")" << std::endl;

            // Perform cross-validation every two epochs
            if ((epoch + 1) % 2 == 0) {
                std::cout << "\n=== Performing Periodic Cross-Validation (Epoch " << epoch + 1 << ") ===" << std::endl;
                float cv_loss = Utils::perform_cross_validation(
                    transformer, 
                    *tokenizer, 
                    all_data
                );
                std::cout << "=== Periodic cross-validation loss: " << cv_loss << " ===" << std::endl;
                
                // Early stopping based on cross-validation with tuned parameters
                if (cv_loss < best_cv_loss) {
                    best_cv_loss = cv_loss;
                    epochs_without_improvement = 0;
                    
                    // Save best model
                    std::cout << "New best model found! Saving checkpoint..." << std::endl;
                    if (!model_saver.saveCheckpoint(transformer, save_directory, 
                                                  model_name + "_best", epoch + 1, cv_loss)) {
                        std::cerr << "Failed to save best model checkpoint" << std::endl;
                    }
                } else {
                    epochs_without_improvement++;
                    if (epochs_without_improvement >= early_stopping_patience) {
                        std::cout << "Early stopping triggered after " << epoch + 1 
                                 << " epochs (patience: " << early_stopping_patience << ")" << std::endl;
                        break;
                    }
                }
                
                // Check for significant overfitting using tuned threshold
                float train_val_ratio = cv_loss / (epoch_loss / total_batches);
                if (train_val_ratio > early_stopping_threshold) {
                    std::cout << "Significant overfitting detected (ratio: " << train_val_ratio 
                             << " > threshold: " << early_stopping_threshold << "). Stopping training." << std::endl;
                    break;
                }
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
                Matrix test_hidden = transformer.forward(test_tokens, "", *tokenizer);
                Matrix logits = transformer.get_lm_head()->forward(test_hidden);
                
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
        Matrix test_hidden = transformer.forward(test_tokens, "", *tokenizer);
        Matrix logits = transformer.get_lm_head()->forward(test_hidden);
        
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