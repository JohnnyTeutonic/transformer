#include "../include/main.hpp"
#include <fstream>
#include <nlohmann/json.hpp>
#include <random>
#include "../include/tokenizer.hpp"
#include "../include/utils.hpp"  // Add include for Utils

// Add necessary forward declarations and structures
std::unique_ptr<Tokenizer> tokenizer;
PerformanceMetrics metrics; // Single definition of the global metrics variable

// Configuration constants
const float INITIAL_LEARNING_RATE = 0.001f;
float learning_rate = INITIAL_LEARNING_RATE;
float prev_loss = std::numeric_limits<float>::max();
size_t global_step = 0;

float compute_loss(const Matrix& logits, const std::vector<int>& target_tokens, const Tokenizer& tokenizer) {
    float loss = 0.0f;
    const int sep_token_id = tokenizer.get_sep_token_id();
    bool after_separator = false;
    
    for (size_t i = 0; i < target_tokens.size() - 1; i++) {
        int current_token = target_tokens[i];
        int next_token = target_tokens[i + 1];
        
        // Track if we're after the separator
        if (current_token == sep_token_id) {
            after_separator = true;
        }
        
        // Get predicted probability distribution
        Vector logits_row = logits.row(i);  // Get as Vector
        std::vector<float> row_data(logits_row.begin(), logits_row.end());  // Convert to std::vector<float>
        
        // Find max for numerical stability
        float max_val = *std::max_element(row_data.begin(), row_data.end());
        
        // Compute softmax with numerical stability
        std::vector<float> probs(row_data.size());
        float sum_exp = 0.0f;
        for (size_t j = 0; j < row_data.size(); j++) {
            probs[j] = std::exp(row_data[j] - max_val);
            sum_exp += probs[j];
        }
        
        // Normalize
        for (float& p : probs) {
            p /= sum_exp;
        }
        
        float token_loss = -std::log(probs[next_token] + 1e-10);
        
        // Apply additional penalty for format violations after separator
        if (after_separator) {
            std::string next_token_str = tokenizer.decode({next_token});
            if (!next_token_str.empty() && next_token_str[0] != ' ') {
                // Penalize tokens that don't start with space after separator
                token_loss *= 1.5f;
            }
        }
        
        loss += token_loss;
    }
    
    return loss / target_tokens.size();
}

// Add these helper functions before train_epoch
size_t get_sequence_length(const std::string& input, const std::string& target) {
    return input.length() + target.length();
}

std::vector<std::vector<std::pair<std::string, std::string>>> create_curriculum_buckets(
    std::vector<std::pair<std::string, std::string>>& training_pairs,
    size_t num_buckets = 5) {
    
    // Sort training pairs by total sequence length
    std::sort(training_pairs.begin(), training_pairs.end(),
        [](const auto& a, const auto& b) {
            return get_sequence_length(a.first, a.second) < get_sequence_length(b.first, b.second);
        });
    
    std::vector<std::vector<std::pair<std::string, std::string>>> buckets(num_buckets);
    size_t pairs_per_bucket = training_pairs.size() / num_buckets;
    
    // Distribute pairs into buckets
    for (size_t i = 0; i < training_pairs.size(); i++) {
        size_t bucket_idx = std::min(i / pairs_per_bucket, num_buckets - 1);
        buckets[bucket_idx].push_back(training_pairs[i]);
    }
    
    return buckets;
}

// Modify train_epoch to use curriculum learning
void train_epoch(Transformer& model, std::vector<std::pair<std::string, std::string>>& training_pairs,
                float learning_rate, const Tokenizer& tokenizer, size_t epoch) {
    
    static std::vector<std::vector<std::pair<std::string, std::string>>> curriculum_buckets;
    static bool buckets_initialized = false;
    
    // Initialize curriculum buckets on first epoch
    if (!buckets_initialized) {
        curriculum_buckets = create_curriculum_buckets(training_pairs);
        buckets_initialized = true;
    }
    
    // Calculate which buckets to use based on training progress
    size_t num_buckets = curriculum_buckets.size();
    size_t available_buckets = std::min(
        num_buckets,
        static_cast<size_t>(1 + (epoch / 2))  // Add a new bucket every 2 epochs
    );
    
    // Combine available buckets for this epoch
    std::vector<std::pair<std::string, std::string>> current_training_pairs;
    for (size_t i = 0; i < available_buckets; i++) {
        current_training_pairs.insert(
            current_training_pairs.end(),
            curriculum_buckets[i].begin(),
            curriculum_buckets[i].end()
        );
    }
    
    // Shuffle available training pairs
    auto rng = std::default_random_engine(std::random_device{}());
    std::shuffle(current_training_pairs.begin(), current_training_pairs.end(), rng);
    
    // Create batches from available training pairs
    size_t batch_size = model.getConfig().batch_size;
    std::vector<std::vector<std::pair<std::string, std::string>>> batches;
    
    // Split into batches
    for (size_t i = 0; i < current_training_pairs.size(); i += batch_size) {
        size_t end_idx = std::min(i + batch_size, current_training_pairs.size());
        std::vector<std::pair<std::string, std::string>> batch(
            current_training_pairs.begin() + i,
            current_training_pairs.begin() + end_idx
        );
        batches.push_back(batch);
    }
    
    // Shuffle the order of batches
    std::shuffle(batches.begin(), batches.end(), rng);
    
    // Process each batch
    for (const auto& batch : batches) {
        // Process each example in the batch
        std::vector<std::vector<int>> input_batch;
        std::vector<std::vector<int>> target_batch;
        size_t max_seq_len = 0;
        
        // First pass: find maximum sequence length in batch
        for (const auto& [input_str, target_str] : batch) {
            std::string processed_input = input_str;
            std::string processed_target = target_str;
            tokenizer.preprocess_text(processed_input);
            tokenizer.preprocess_text(processed_target);
            
            std::vector<int> input_tokens = tokenizer.encode(processed_input);
            std::vector<int> target_tokens = tokenizer.encode(processed_target);
            
            max_seq_len = std::max({max_seq_len, input_tokens.size(), target_tokens.size()});
        }
        
        // Second pass: pad sequences to max_seq_len
        for (const auto& [input_str, target_str] : batch) {
            std::string processed_input = input_str;
            std::string processed_target = target_str;
            tokenizer.preprocess_text(processed_input);
            tokenizer.preprocess_text(processed_target);
            
            std::vector<int> input_tokens = tokenizer.encode(processed_input);
            std::vector<int> target_tokens = tokenizer.encode(processed_target);
            
            // Pad input sequence
            while (input_tokens.size() < max_seq_len) {
                input_tokens.push_back(tokenizer.get_pad_token_id());
            }
            
            // Pad target sequence
            while (target_tokens.size() < max_seq_len) {
                target_tokens.push_back(tokenizer.get_pad_token_id());
            }
            
            input_batch.push_back(input_tokens);
            target_batch.push_back(target_tokens);
        }
        
        // Create target distribution
        Matrix target_distribution = Utils::create_batch_target_distribution(
            target_batch, tokenizer, model.getConfig().vocab_size, max_seq_len);
        
        // Forward pass
        model.set_training(true);
        Matrix hidden_states;
        for (size_t i = 0; i < input_batch.size(); i++) {
            Matrix batch_hidden = model.forward(input_batch[i], "", tokenizer);
            if (hidden_states.empty()) {
                hidden_states = Matrix(input_batch.size() * batch_hidden.rows(), batch_hidden.cols());
            }
            // Copy batch hidden states
            for (size_t j = 0; j < batch_hidden.rows(); j++) {
                for (size_t k = 0; k < batch_hidden.cols(); k++) {
                    hidden_states(i * batch_hidden.rows() + j, k) = batch_hidden(j, k);
                }
            }
        }
        
        // Get logits and compute loss
        Matrix logits = model.get_lm_head()->project_to_vocab(hidden_states);
        float batch_loss = Utils::compute_batch_loss(logits, target_distribution, tokenizer);
        
        // Backward pass
        Matrix loss_gradients(logits.rows(), logits.cols());
        for (size_t i = 0; i < logits.rows(); i++) {
            for (size_t j = 0; j < logits.cols(); j++) {
                loss_gradients(i, j) = logits(i, j) - target_distribution(i, j);
            }
        }
        
        model.backward(loss_gradients, input_batch[0], learning_rate);
        model.update_parameters(learning_rate);
        
        std::cout << "Batch loss: " << batch_loss << std::endl;
    }
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
        size_t epoch = 0;
        while (true) {
            std::cout << "\nStarting epoch " << epoch << std::endl;
            
            // Train one epoch
            train_epoch(transformer, training_pairs, learning_rate, *tokenizer, epoch);
            
            // Save checkpoint if needed
            if ((epoch + 1) % checkpoint_frequency == 0) {
                std::string checkpoint_path = save_directory + "/" + model_name + "_epoch" + 
                                            std::to_string(epoch) + ".ckpt";
                model_saver.saveCheckpoint(transformer, save_directory, model_name, 
                                         static_cast<int>(epoch), prev_loss);
            }
            
            epoch++;
            
            // Optional: Add early stopping based on validation loss
            // ... rest of existing code ...
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