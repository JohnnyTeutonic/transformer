#pragma once
#include "tiktoken_tokenizer.hpp"
#include "matrix.hpp"
#include "transformer.hpp"
#include "phrase_types.hpp"
#include <string>
#include <utility>
#include <vector>
#include <unordered_set>
#include <cmath>
#include <random>
#include <atomic>
#include <chrono>
#include <memory>
#include <unordered_map>

// Token category structure
struct TokenCategories {
    std::unordered_set<std::string> verb_tokens;
    std::unordered_set<std::string> adjective_tokens;
    std::unordered_set<std::string> noun_tokens;
};

struct TrainingExample {
    std::string input;
    std::string output;
    PhraseType type;

    // Add conversion operator to pair
    operator std::pair<std::string, std::string>() const {
        return {input, output};
    }

    // Add constructor from pair
    TrainingExample(const std::pair<std::string, std::string>& pair) 
        : input(pair.first), output(pair.second), type(PhraseType::GENERAL) {}

    // Default constructor
    TrainingExample() = default;

    // Regular constructor
    TrainingExample(std::string input_, std::string output_, PhraseType type_)
        : input(std::move(input_)), output(std::move(output_)), type(type_) {}
};

struct ContextualTrainingExample {
    std::vector<std::string> context_window;  // Previous phrases for context
    std::string input;
    std::string output;
    PhraseType type;
    size_t context_size;  // Size of context window

    ContextualTrainingExample(std::vector<std::string> context, 
                            std::string input_, 
                            std::string output_,
                            PhraseType type_,
                            size_t ctx_size = 3)
        : context_window(std::move(context))
        , input(std::move(input_))
        , output(std::move(output_))
        , type(type_)
        , context_size(ctx_size) {}

    // Get full context including current input
    std::string get_full_context() const {
        std::string full_context;
        for (const auto& ctx : context_window) {
            full_context += ctx + " ";
        }
        full_context += input;
        return full_context;
    }

    // Add conversion operators
    operator std::pair<std::string, std::string>() const {
        return {input, output};
    }

    operator TrainingExample() const {
        return TrainingExample(input, output, type);
    }
};

struct ValidationMetrics {
    float loss;
    float accuracy;
    std::unordered_map<PhraseType, float> type_specific_accuracy;
    std::unordered_map<PhraseType, float> type_specific_loss;
};

class Utils {
private:
    static std::random_device rd;  // Hardware random number source
    static std::mt19937 random_generator;
    static std::atomic<uint64_t> prediction_counter;  // Counter for unique seeds

    // Randomization helpers
    static float apply_temperature_scaling(
        std::vector<float>& logits,
        float temperature,
        std::mt19937& gen
    );

    static void add_random_variation(
        std::vector<float>& probabilities,
        std::mt19937& gen,
        float min_var = 0.8f,
        float max_var = 1.2f
    );

    static std::vector<std::pair<float, int>> apply_nucleus_sampling(
        const std::vector<std::pair<float, int>>& token_probs,
        float p,
        std::mt19937& gen
    );

    // Add new private methods for data handling
    static PhraseType detect_phrase_type(const std::string& input, const std::string& output, char delimiter);
    static float get_sampling_weight(const std::string& category, 
                                   const std::unordered_map<std::string, size_t>& counts);
    static std::vector<TrainingExample> balance_dataset(
        const std::unordered_map<PhraseType, std::vector<TrainingExample>>& categorized_data);

public:
    /**
     * @brief Find the index of the maximum value in a Matrix row or Vector
     * @param row The Matrix row or Vector to find the maximum value in
     * @return The index of the maximum value
     */
    static size_t argmax(const Matrix& row) {
        if (row.rows() != 1) {
            throw std::runtime_error("argmax expects a single row matrix");
        }
        return std::distance(
            row.data(),
            std::max_element(row.data(), row.data() + row.cols())
        );
    }

    /**
     * @brief Find the index of the maximum value in a Vector
     * @param vec The Vector to find the maximum value in
     * @return The index of the maximum value
     */
    static size_t argmax(const Vector& vec) {
        return std::distance(
            vec.data(),
            std::max_element(vec.data(), vec.data() + vec.size())
        );
    }

    static bool validate_input_sequence(const std::vector<int>& tokens, size_t vocab_size,
                                        size_t max_seq_length = 512);
    static void print_matrix(const Matrix& m, const std::string& name, size_t max_rows = 5,
                             size_t max_cols = 5);
    static void print_top_predictions(
        const Matrix& logits,
        const TiktokenTokenizer& tokenizer,
        Transformer& transformer,
        int k
    );

    // Add new function to save predictions to CSV
    static void save_predictions_to_csv(
        const Matrix& logits,
        const TiktokenTokenizer& tokenizer,
        const std::string& input_text,
        const std::string& csv_path = "predictions.csv",
        int top_k = 10
    );

    static std::vector<ContextualTrainingExample> create_training_data();
    static void
    analyze_token_mappings(const std::vector<std::pair<std::string, std::string>>& training_data,
                           const TiktokenTokenizer& tokenizer);
    static std::vector<ContextualTrainingExample> load_validation_data();
    static ValidationMetrics evaluate_validation(
        Transformer& transformer,
        const TiktokenTokenizer& tokenizer,
        const std::vector<ContextualTrainingExample>& validation_data
    );
    static TransformerConfig load_config(const std::string& config_path);
    static Matrix
    create_batch_target_distribution(const std::vector<std::vector<int>>& target_tokens,
                                     const TiktokenTokenizer& tokenizer, size_t vocab_size,
                                     size_t input_max_seq_len);
    static float compute_batch_loss(const Matrix& logits, const Matrix& target_distribution, const TiktokenTokenizer& tokenizer);
    static std::vector<std::string>& get_vocabulary(const TiktokenTokenizer& tokenizer);
    static std::vector<std::pair<std::string, std::string>> get_multi_token_predictions(
        const Matrix& logits, const TiktokenTokenizer& tokenizer, int beam_width);
    
    // Token category analysis functions
    static TokenCategories analyze_token_categories(const std::vector<std::pair<std::string, std::string>>& training_data);
    static std::string get_token_category(const std::string& token, const TokenCategories& categories);
    static void trim(std::string& s);

    // Cross-validation functions
    static std::vector<std::pair<std::vector<std::pair<std::string, std::string>>, 
                                std::vector<std::pair<std::string, std::string>>>> 
    create_cross_validation_folds(const std::vector<std::pair<std::string, std::string>>& data, 
                                size_t num_folds);

    static float perform_cross_validation(
        Transformer& transformer,
        const TiktokenTokenizer& tokenizer,
        const std::vector<std::pair<std::string, std::string>>& train_data
    );

    // Add inline utility functions for gradient computation
    static inline float compute_grad_norm(const Matrix& grad) {
        float norm = 0.0f;
        #pragma omp parallel for reduction(+:norm)
        for (size_t i = 0; i < grad.rows(); ++i) {
            for (size_t j = 0; j < grad.cols(); ++j) {
                norm += grad(i, j) * grad(i, j);
            }
        }
        return std::sqrt(norm);
    }

    static inline size_t count_params(const Matrix& param) {
        return param.rows() * param.cols();
    }

    // Add loss computation functions
    static inline float compute_loss(const Matrix& output, const Matrix& target_distribution) {
        if (output.size() != target_distribution.size()) {
            throw std::runtime_error("Output and target distribution must have the same size");
        }

        const size_t batch_size = output.rows();
        const size_t vocab_size = output.cols();
        const float epsilon = 1e-5f;  // Increased from 1e-10f for better stability
        float total_loss = 0.0f;

        #pragma omp parallel for reduction(+:total_loss)
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < vocab_size; ++j) {
                if (target_distribution(i, j) > 0.0f) {
                    float pred = std::clamp(output(i, j), epsilon, 1.0f - epsilon);
                    total_loss -= target_distribution(i, j) * std::log(pred);
                }
            }
        }

        return total_loss / static_cast<float>(batch_size);
    }

    static inline Matrix compute_loss_gradient(const Matrix& output, const Matrix& target_distribution) {
        if (output.size() != target_distribution.size()) {
            throw std::runtime_error("Output and target distribution must have the same size");
        }

        const size_t batch_size = output.rows();
        const size_t vocab_size = output.cols();
        const float epsilon = 1e-5f;  // Increased epsilon for stability
        Matrix gradient(batch_size, vocab_size);

        // Pre-compute max values for numerical stability
        std::vector<float> max_logits(batch_size, -std::numeric_limits<float>::infinity());
        #pragma omp parallel for
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < vocab_size; ++j) {
                max_logits[i] = std::max(max_logits[i], output(i, j));
            }
        }

        // Pre-compute denominator terms
        std::vector<float> denominators(batch_size, 0.0f);
        #pragma omp parallel for
        for (size_t i = 0; i < batch_size; ++i) {
            float sum_exp = 0.0f;
            for (size_t j = 0; j < vocab_size; ++j) {
                float shifted_logit = output(i, j) - max_logits[i];
                sum_exp += std::exp(shifted_logit);
            }
            denominators[i] = std::max(sum_exp, epsilon);
        }

        // Compute final gradients
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < vocab_size; ++j) {
                float shifted_logit = output(i, j) - max_logits[i];
                float softmax_output = std::exp(shifted_logit) / denominators[i];
                
                // Stabilized gradient computation
                gradient(i, j) = std::clamp(
                    softmax_output - target_distribution(i, j),
                    -1.0f,  // Prevent extreme gradients
                    1.0f
                );
            }
        }

        return gradient;
    }

    // Random number generation utilities
    static void set_random_generator(const std::mt19937& gen) {
        random_generator = gen;
    }
    
    static std::mt19937& get_random_generator() {
        return random_generator;
    }
    
    static float random_float(float min = 0.0f, float max = 1.0f) {
        std::uniform_real_distribution<float> dist(min, max);
        return dist(random_generator);
    }
    
    static int random_int(int min, int max) {
        std::uniform_int_distribution<int> dist(min, max);
        return dist(random_generator);
    }
    
    // Get a new random generator with unique seed
    static std::mt19937 get_new_generator() {
        // Combine multiple entropy sources
        auto time_seed = static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        auto counter = prediction_counter.fetch_add(1, std::memory_order_relaxed);
        auto hw_rand = static_cast<uint64_t>(rd());
        
        // Create seed sequence from multiple sources
        std::seed_seq seq{
            static_cast<uint32_t>(time_seed),
            static_cast<uint32_t>(time_seed >> 32),
            static_cast<uint32_t>(counter),
            static_cast<uint32_t>(hw_rand),
            static_cast<uint32_t>(hw_rand >> 32),
            static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&counter))  // Use address as additional entropy
        };
        
        return std::mt19937(seq);
    }

    // Initialize random generator with time-based seed
    static void initialize_random() {
        auto time_seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        std::seed_seq seq{static_cast<uint32_t>(time_seed & 0xFFFFFFFF)};
        random_generator = std::mt19937(seq);
        prediction_counter = 0;
    }

    static void generate_predictions(
        Transformer& transformer,
        const std::string& input_text,
        std::shared_ptr<TiktokenTokenizer> tokenizer
    );

    // Debugging utilities
    static void analyze_gradients(const Matrix& gradients, const std::string& label);
    static void analyze_loss_progression(const std::vector<float>& losses, size_t window_size);
    static void debug_token_processing(const std::string& input, const std::vector<int>& tokens, 
                                     const TiktokenTokenizer& tokenizer);
};