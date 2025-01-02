#include "../include/attention/advanced_attention.hpp"
#include "../include/cuda/cuda_init.cuh"
#include "../include/lm_head.hpp"
#include "../include/logger.hpp"
#include "../include/model_saver.hpp"
#include "../include/optimizer/sam.hpp"
#include "../include/quantization.hpp"
#include "../include/tokenizer.hpp"
#include "../include/transformer.hpp"
#include "../include/utils/tensor_cache.hpp"
#include "../include/vocabulary.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

void print_matrix(const Matrix &m, const std::string &name, size_t max_rows = 5,
                  size_t max_cols = 5) {
  std::cout << "\n" << name << " (" << m.rows() << "x" << m.cols() << "):\n";
  for (size_t i = 0; i < std::min(max_rows, m.rows()); ++i) {
    for (size_t j = 0; j < std::min(max_cols, m.cols()); ++j) {
      std::cout << std::fixed << std::setprecision(4) << m(i, j) << " ";
    }
    std::cout << (m.cols() > max_cols ? "..." : "") << "\n";
  }
  if (m.rows() > max_rows)
    std::cout << "...\n";
}

void print_top_predictions(const Matrix &logits, const Tokenizer &tokenizer,
                           size_t k = 5) {
  std::vector<std::pair<float, int>> scores;
  for (size_t i = 0; i < logits.cols(); ++i) {
    scores.push_back({logits(logits.rows() - 1, i), static_cast<int>(i)});
  }

  std::partial_sort(
      scores.begin(), scores.begin() + k, scores.end(),
      [](const auto &a, const auto &b) { return a.first > b.first; });

  std::cout << "\nTop " << k << " predictions:\n";
  for (size_t i = 0; i < k; ++i) {
    std::string token = tokenizer.decode({scores[i].second});
    std::cout << i + 1 << ". \"" << token << "\" (probability: " << std::fixed
              << std::setprecision(4) << std::exp(scores[i].first) << ")\n";
  }
}

// Add this helper function to create a simple dataset
std::vector<std::pair<std::string, std::string>> create_training_data() {
  std::vector<std::pair<std::string, std::string>> training_pairs;
  // Get the executable directory
  std::filesystem::path exe_path =
      std::filesystem::current_path().parent_path();
  std::filesystem::path data_dir = exe_path / "data";
  std::filesystem::path file_path = data_dir / "training_pairs.txt";

  // Create data directory if it doesn't exist
  if (!std::filesystem::exists(data_dir)) {
    std::filesystem::create_directories(data_dir);
  }

  std::ifstream file(file_path);

  if (!file.is_open()) {
    throw std::runtime_error("Could not open training data file: " +
                             file_path.string());
  }

  std::string line;
  while (std::getline(file, line)) {
    size_t delimiter_pos = line.find('|');
    if (delimiter_pos != std::string::npos) {
      std::string input = line.substr(0, delimiter_pos);
      std::string output = line.substr(delimiter_pos + 1);
      training_pairs.emplace_back(input, output);
    }
  }

  if (training_pairs.empty()) {
    throw std::runtime_error("No training pairs loaded from file");
  }

  return training_pairs;
}

Matrix create_target_distribution(const std::vector<int> &targets,
                                  size_t vocab_size) {
  // Create distribution matrix with same number of rows as sequence length
  Matrix distribution(targets.size(), vocab_size, 0.0f);
  std::cout << "Creating target distribution with shape: "
            << distribution.rows() << "x" << distribution.cols() << std::endl;
  for (int target : targets) {
    if (target >= static_cast<int>(vocab_size)) {
      throw std::runtime_error("Target ID exceeds vocabulary size");
    }
    // Set target positions to 1.0 in the last row (for next token prediction)
    distribution(distribution.rows() - 1, target) = 1.0f;
  }
  return distribution;
}

Matrix compute_cross_entropy_gradient(const Matrix &logits,
                                      const Matrix &targets,
                                      const LanguageModelHead *lm_head) {

  // Ensure logits and targets have matching dimensions
  if (logits.rows() != targets.rows() || logits.cols() != targets.cols()) {
    throw std::runtime_error(
        "Logits dimensions (" + std::to_string(logits.rows()) + "x" +
        std::to_string(logits.cols()) + ") must match targets dimensions (" +
        std::to_string(targets.rows()) + "x" + std::to_string(targets.cols()) +
        ")");
  }

  // First compute the vocab-space gradient
  Matrix vocab_grad = logits;
  std::cout << "vocab_grad dimensions: " << vocab_grad.rows() << "x"
            << vocab_grad.cols() << std::endl;
  std::cout << "applying softmax" << std::endl;
  vocab_grad.apply_softmax();
  std::cout << "subtracting targets" << std::endl;
  vocab_grad = vocab_grad - targets;

  return vocab_grad; // Return the gradient in vocab space
}

int main(int argc, char *argv[]) {
  // Initialize logger
  Logger &logger = Logger::getInstance();
  logger.startLogging();

  try {
    initialize_cuda(); // Initialize CUDA at program start

    // Configure the transformer
    TransformerConfig config;
    config.vocab_size = 50000;
    config.hidden_size = 768;
    config.num_heads = 12;
    config.num_layers = 6;
    config.use_flash_attention = true;
    config.use_rope = true;
    config.use_sliding_window = true;
    config.window_size = 256;
    config.use_cuda = false;

    std::cout << "Initializing transformer with configuration:\n"
              << "- Hidden size: " << config.hidden_size << "\n"
              << "- Attention heads: " << config.num_heads << "\n"
              << "- Layers: " << config.num_layers << "\n"
              << "- Using Flash Attention: " << std::boolalpha
              << config.use_flash_attention << "\n"
              << "- Using RoPE: " << config.use_rope << "\n"
              << "- Using Sliding Window: " << config.use_sliding_window
              << "\n";

    // Initialize components
    Transformer transformer(config);
    auto tokenizer = std::make_unique<Tokenizer>();
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

    // Create training data
    auto training_pairs = create_training_data();
    std::cout << "\nTraining on " << training_pairs.size() << " examples\n";

    // Training parameters
    const size_t num_epochs = 10;
    const float learning_rate = 0.001f;
    const size_t checkpoint_frequency = 2; // Save checkpoint every 2 epochs

    // Initialize model saver
    ModelSaver model_saver;
    std::string save_directory = "models";
    std::string model_name = "transformer_model";

    // Training loop
    Matrix last_hidden_states; // Add this to store the last hidden states
    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
      std::cout << "Epoch " << epoch + 1 << "/" << num_epochs << "\n";
      float epoch_loss = 0.0f;

      const size_t batch_size = 32; // Adjust based on your GPU memory
      for (size_t i = 0; i < training_pairs.size(); i += batch_size) {
        std::cout << "Processing batch " << i << std::endl;
        // Create batch
        std::vector<std::vector<int>> input_batch;
        std::vector<std::vector<int>> target_batch;

        // Fill batch
        for (size_t j = 0; j < batch_size && (i + j) < training_pairs.size();
             ++j) {
          const auto &[input_text, target_text] = training_pairs[i + j];
          input_batch.push_back(tokenizer->encode(input_text));
          target_batch.push_back(tokenizer->encode(target_text));
        }

        // Get input and target from training pairs
        const auto &[input_text, target_text] = training_pairs[i];
        std::cout << "Processing pair " << i << ": '" << input_text << "' -> '"
                  << target_text << "'\n";

        // Tokenize input and target
        std::vector<int> input_tokens = tokenizer->encode(input_text);
        std::vector<int> target_tokens = tokenizer->encode(target_text);
        std::cout << "Input tokens: " << input_tokens.size() << "\n";
        std::cout << "Target tokens: " << target_tokens.size() << "\n";
        std::cout << "Forward pass for input tokens '" << target_text << "'\n";
        // Forward pass
        Matrix hidden_states = transformer.forward(input_tokens);
        last_hidden_states = hidden_states;
        std::cout << "Forward pass for hidden states '" << target_text << "'\n";
        // Project to vocabulary space
        Matrix logits = lm_head->project_to_vocab(hidden_states);
        std::cout << "Hidden states shape: " << hidden_states.rows() << "x"
                  << hidden_states.cols() << "\n";
        std::cout << "Logits shape: " << logits.rows() << "x" << logits.cols()
                  << "\n";

        // Ensure target_distribution has same sequence length as logits
        Matrix target_distribution =
            create_target_distribution(target_tokens, config.vocab_size);
        if (target_distribution.rows() != logits.rows()) {
          // Resize target_distribution to match logits sequence length
          Matrix resized_targets(logits.rows(), config.vocab_size, 0.0f);
          // Copy the last row of target_distribution to all rows of
          // resized_targets
          for (size_t i = 0; i < logits.rows(); i++) {
            for (size_t j = 0; j < config.vocab_size; j++) {
              resized_targets(i, j) =
                  target_distribution(target_distribution.rows() - 1, j);
            }
          }
          target_distribution = resized_targets;
        }

        std::cout << "Creating target distribution\n";
        // Compute proper loss and gradients
        Matrix loss_grad = compute_cross_entropy_gradient(
            logits, target_distribution, lm_head.get());

        // Backpropagate through the network
        std::cout << "Backward pass for loss gradient\n";
        Matrix hidden_grad = lm_head->backward(loss_grad, hidden_states);
        std::cout << "Backward pass for hidden states\n";
        transformer.backward(hidden_grad, input_tokens);
        std::cout << "Target Matrix calculation" << target_text << "'\n";

        // Compute loss and gradients
        Matrix target_matrix(logits.rows(), logits.cols(), 0.0f);
        std::cout << "Target matrix shape: " << target_matrix.rows() << "x"
                  << target_matrix.cols() << "\n";
        // We only care about the last token's prediction
        size_t last_position = logits.rows() - 1; // Last position in sequence
        std::cout << "Last position: " << last_position << "\n";
        for (int token : target_tokens) {
          target_matrix(last_position, token) =
              1.0f; // One-hot encode only the last position
        }
        std::cout << "Target matrix after one-hot encoding: "
                  << target_matrix.rows() << "x" << target_matrix.cols()
                  << "\n";
        // Cross entropy loss (only for last position)
        float loss = 0.0f;
        for (size_t j = 0; j < logits.cols(); ++j) {
          if (target_matrix(last_position, j) > 0.0f) {
            loss -= std::log(logits(last_position, j) + 1e-10);
          }
        }
        epoch_loss += loss;
        std::cout << "Loss: " << loss << "\n";
        // Backward pass - compute gradients
        Matrix grad_output(logits.rows(), logits.cols(),
                           0.0f); // Initialize with zeros
        std::cout << "Created gradient output matrix\n";

        // Apply softmax derivative: grad * (softmax - target)
        // First, apply softmax to the last position
        float max_val = 0.0001f;
        std::cout << "Initialized max value for softmax\n";

        // Only compute gradients for the last position
        for (size_t j = 0; j < logits.cols(); ++j) {
          max_val = std::max(max_val, logits(last_position, j));
        }
        std::cout << "Found max value: " << max_val << "\n";

        float sum = 0.0f;
        std::cout << "Initialized sum for softmax normalization\n";

        // Store softmax values temporarily
        std::vector<float> softmax_values(logits.cols());
        for (size_t j = 0; j < logits.cols(); ++j) {
          softmax_values[j] = std::exp(logits(last_position, j) - max_val);
          sum += softmax_values[j];
        }
        std::cout << "Computed exponentials and sum: " << sum << "\n";

        // Compute gradients only for last position
        for (size_t j = 0; j < logits.cols(); ++j) {
          float softmax_prob = softmax_values[j] / sum;
          grad_output(last_position, j) =
              softmax_prob - target_matrix(last_position, j);
        }
        std::cout << "Computed gradients for last position\n";

        // Note: Other positions are already zero from initialization

        // Update weights using SAM optimizer
        std::vector<Matrix *> params;
        std::vector<Matrix> grads;
        std::cout << "Updating weights using SAM optimizer\n";

        // Add transformer parameters
        auto transformer_weights = transformer.get_layer_weights();
        for (const auto &layer_weights : transformer_weights) {
          for (auto &weight : layer_weights) {
            params.push_back(&weight.get());
          }
        }
        std::cout << "Transformer parameters added\n";

        // Add language model head parameters
        auto lm_params = lm_head->get_parameters();
        for (auto &param : lm_params) {
          params.push_back(&param.get());
        }
        std::cout << "Language model parameters added\n";

        // Initialize gradients
        for (size_t i = 0; i < params.size(); ++i) {
          grads.push_back(Matrix(params[i]->rows(), params[i]->cols()));
        }

        // First step with initial gradients
        std::cout << "Starting SAM first step with initial gradients\n";
        std::cout << "Number of parameters: " << params.size()
                  << ", Number of gradients: " << grads.size() << "\n";
        sam_optimizer->first_step(params, grads);
        std::cout << "Completed first step\n";

        // Recompute gradients at the perturbed point
        std::cout << "\nRecomputing gradients at perturbed point...\n";
        Matrix new_hidden_states = transformer.forward(input_tokens);
        std::cout << "New hidden states shape: " << new_hidden_states.rows()
                  << "x" << new_hidden_states.cols() << "\n";

        Matrix new_logits = lm_head->forward(new_hidden_states);
        std::cout << "New logits shape: " << new_logits.rows() << "x"
                  << new_logits.cols() << "\n";

        // Recompute grad_output similar to before
        Matrix new_grad_output(new_logits.rows(), new_logits.cols(), 0.0f);
        std::cout << "Created new gradient output matrix: "
                  << new_grad_output.rows() << "x" << new_grad_output.cols()
                  << "\n";

        // Recompute softmax and gradients for last position
        float new_max_val = 0.0001f;
        size_t last_pos = new_logits.rows() - 1;
        for (size_t j = 0; j < new_logits.cols(); ++j) {
          new_max_val = std::max(new_max_val, new_logits(last_pos, j));
        }
        std::cout << "Computed new max value for softmax: " << new_max_val
                  << "\n";

        float new_sum = 0.0f;
        std::vector<float> new_softmax_values(new_logits.cols());
        for (size_t j = 0; j < new_logits.cols(); ++j) {
          new_softmax_values[j] =
              std::exp(new_logits(last_pos, j) - new_max_val);
          new_sum += new_softmax_values[j];
        }

        std::cout << "Computed new softmax normalization sum: " << new_sum
                  << "\n";

        // Create new gradients vector with correct dimensions
        std::vector<Matrix> new_grads;
        std::cout << "Created empty new_grads vector\n";

        // First compute gradients for transformer parameters
        for (size_t i = 0; i < params.size(); ++i) {
          // Initialize each gradient with same dimensions as its parameter
          new_grads.push_back(
              Matrix(params[i]->rows(), params[i]->cols(), 0.0f));
          std::cout << "Created gradient " << i
                    << " with dimensions: " << new_grads.back().rows() << "x"
                    << new_grads.back().cols() << "\n";
        }
        std::cout
            << "Finished initializing all gradients with correct dimensions\n";

        // Compute gradients for the last position
        std::cout << "Computing gradients for last position...\n";
        for (size_t j = 0; j < new_logits.cols(); ++j) {
          float softmax_prob = new_softmax_values[j] / new_sum;
          new_grad_output(last_pos, j) =
              softmax_prob - target_matrix(last_pos, j);
        }
        std::cout << "Computed new gradients for last position\n";

        // Backpropagate the gradients through the network
        std::cout << "Starting gradient backpropagation\n";
        Matrix current_grad = new_grad_output;
        std::cout << "Created current_grad with dimensions: "
                  << current_grad.shape() << "\n";

        // Print dimensions of first few parameters for debugging
        /*std::cout << "First few parameter dimensions:\n";
        for (size_t i = 0; i < std::min(size_t(3), params.size()); ++i) {
          std::cout << "Parameter shape: " << params[i]->shape() << "\n";
        }*/

        // Ensure gradients match parameter dimensions exactly
        for (size_t i = 0; i < params.size(); ++i) {

          // Get parameter dimensions
          size_t param_rows = params[i]->rows();
          size_t param_cols = params[i]->cols();

          /*std::cout << "Parameter dimensions: " << param_rows << "x"
                    << param_cols << "\n";*/

          // Create gradient with matching dimensions
          new_grads[i] = Matrix(param_rows, param_cols, 0.0f);

          // For now, use a very small constant gradient for testing
          // This ensures dimensions match exactly with the parameter
          for (size_t r = 0; r < param_rows; ++r) {
            for (size_t c = 0; c < param_cols; ++c) {
              new_grads[i](r, c) = 1e-4f; // Very small constant gradient
            }
          }
        }
        std::cout << "Completed gradient computation for all parameters\n";

        // Verify gradient dimensions before second step
        std::cout << "\nVerifying gradient dimensions:\n";
        for (size_t i = 0; i < params.size(); ++i) {
          if (params[i]->rows() != new_grads[i].rows() ||
              params[i]->cols() != new_grads[i].cols()) {
            std::cout << "Dimension mismatch at parameter " << i << "!\n";
            std::cout << "Parameter: " << params[i]->shape() << "\n";
            std::cout << "Gradient: " << new_grads[i].shape() << "\n";

            throw std::runtime_error(
                "Gradient dimensions don't match parameters");
          }
        }
        std::cout << "All gradient dimensions verified\n";

        // Second step with new gradients
        std::cout << "\nStarting SAM second step\n";
        std::cout << "Number of parameters: " << params.size()
                  << ", Number of new gradients: " << new_grads.size() << "\n";
        sam_optimizer->second_step(params, new_grads);
        std::cout << "Completed second step\n\n";

        // Handle bias updates separately
        std::vector<std::reference_wrapper<FloatVector>> biases;
        std::vector<FloatVector> bias_grads;

        // Collect biases from transformer layers
        for (const auto &layer : transformer.getLayers()) {
          // Collect attention biases
          auto *attn = layer->getAttention();
          biases.push_back(std::ref(attn->getQueryBias()));
          biases.push_back(std::ref(attn->getKeyBias()));
          biases.push_back(std::ref(attn->getValueBias()));
          biases.push_back(std::ref(attn->getOutputBias()));

          // Collect feed forward biases
          auto *ff = layer->getFeedForward();
          biases.push_back(std::ref(ff->getBias1()));
          biases.push_back(std::ref(ff->getBias2()));
        }

        // Compute bias gradients
        bias_grads.resize(biases.size());
        for (size_t i = 0; i < biases.size(); ++i) {
          const FloatVector &bias = biases[i].get();
          FloatVector &grad = bias_grads[i];
          grad.resize(bias.size());

          // Compute gradients for biases (simplified)
          for (size_t j = 0; j < bias.size(); ++j) {
            grad[j] = 0.0001f; // Small constant gradient for testing
          }
        }

        // Update biases
        try {
          sam_optimizer->update_bias(biases, bias_grads);
          std::cout << "Completed bias updates\n";
        } catch (const std::exception &e) {
          std::cerr << "Error updating biases: " << e.what() << std::endl;
          throw;
        }
      }

      // Print epoch statistics
      epoch_loss /= training_pairs.size();
      std::cout << "Epoch " << epoch + 1 << "/" << num_epochs
                << ", Loss: " << epoch_loss << "\n";

      // Save checkpoint
      if ((epoch + 1) % checkpoint_frequency == 0) {
        if (!model_saver.saveCheckpoint(transformer, save_directory, model_name,
                                        epoch + 1, epoch_loss)) {
          logger.log("Failed to save checkpoint", true);
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
          print_top_predictions(test_logits, *tokenizer, 3);
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
                 model_name);
      std::cout << "Model saved successfully!\n";
    } else {
      logger.log("Failed to save model to " + save_directory + "/" + model_name,
                 true);
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

  cleanup_cuda(); // Cleanup at program end
  logger.stopLogging();
  return 0;
}