#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <mutex>
#include "matrix.hpp"
#include "scope_logger.hpp"  // Include the scope logger header

namespace debug {
    extern bool verbose_logging;
    extern bool scope_logging_enabled;
    extern std::ofstream debug_log;
    extern const std::string log_file;
    extern const std::string progress_file;
    extern std::mutex log_mutex;
    extern std::ofstream progress_log;

    // Track current progress state
    struct ProgressState {
        enum class Stage {
            TUNING,
            TRAINING,
            CROSS_VALIDATION,
            INFERENCE,
            IDLE
        } current_stage = Stage::IDLE;

        // Hyperparameter tuning state
        struct TuningProgress {
            size_t current_trial = 0;
            size_t total_trials = 0;
            std::string current_config;  // JSON string of current hyperparameters
            float best_loss = std::numeric_limits<float>::max();
        } tuning;

        // Training/Cross-validation state
        struct TrainingProgress {
            size_t current_epoch = 0;
            size_t total_epochs = 0;
            size_t current_batch = 0;
            size_t total_batches = 0;
            size_t current_fold = 0;
            size_t total_folds = 0;
            float current_loss = 0.0f;
            float best_loss = std::numeric_limits<float>::max();
            bool is_cross_validation = false;
        } training;

        // Inference state
        struct InferenceProgress {
            size_t tokens_generated = 0;
            size_t total_tokens = 0;
            float average_time_per_token = 0.0f;
        } inference;

        void update_tuning(size_t trial, size_t total, const std::string& config, float loss);
        void update_training(size_t epoch, size_t total_epochs, size_t batch, size_t total_batches, float loss);
        void update_cross_validation(size_t fold, size_t total_folds, size_t epoch, size_t total_epochs, float loss);
        void update_inference(size_t tokens, size_t total, float avg_time);
        void reset();
        std::string get_stage_string() const;
        void update_progress_file();

    private:
        void write_tuning_progress();
        void write_training_progress();
        void write_inference_progress();
    };

    // Declare the global progress_state variable
    extern ProgressState progress_state;

    // Function declarations
    void init_logging();
    void enable_scope_logging(bool enable);
    void log_message(const std::string& message, const std::string& level = "INFO");
    void log_vector(const std::vector<int>& vec, const std::string& name);
    void log_matrix(const Matrix& mat, const std::string& label);
    void log_token_distribution(const Matrix& dist, const std::string& name);
    void log_progress(const std::string& stage, size_t current, size_t total, 
                     const std::string& additional_info = "");
}

// Define a macro for easy usage that checks if scope logging is enabled
#define SCOPE_LOG() \
    debug::ScopeLogger scope_logger_instance = \
        (debug::scope_logging_enabled ? \
         debug::ScopeLogger(__FUNCTION__, __FILE__, __LINE__) : \
         debug::ScopeLogger()) 