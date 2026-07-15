#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <mutex>
#include <memory>
#include <chrono>
#include <atomic>

namespace convergence {

struct GradientState {
    std::string node_id;
    std::vector<float> gradients;        // Current gradients
    std::vector<float> control_variates; // SCAFFOLD control variates
    uint64_t step_number;                // Global step when this gradient was computed
    uint64_t local_steps;                // Number of local steps taken
    float learning_rate;                 // Learning rate used
    std::chrono::steady_clock::time_point timestamp;
};

struct ModelSnapshot {
    std::vector<float> parameters;
    uint64_t global_step;
    float validation_loss;
    float training_loss;
    std::chrono::steady_clock::time_point timestamp;
    std::unordered_map<std::string, float> layer_statistics; // Per-layer stats
};

struct ConvergenceMetrics {
    float gradient_norm;
    float parameter_norm;
    float loss_reduction_rate;
    float convergence_score;            // 0.0 to 1.0, higher = better convergence
    uint32_t steps_since_improvement;
    bool is_converged;
    bool is_diverging;
    std::string convergence_status;     // "converging", "converged", "diverging", "stalled"
};

struct FedProxConfig {
    float proximal_mu = 0.01f;          // Proximal term coefficient
    bool enable_adaptive_mu = true;     // Adapt mu based on heterogeneity
    float mu_adaptation_rate = 0.1f;    // How quickly to adapt mu
    float max_mu = 1.0f;               // Maximum mu value
    float min_mu = 0.001f;             // Minimum mu value
};

struct SCAFFOLDConfig {
    float server_learning_rate = 1.0f;  // Server learning rate
    bool enable_momentum = true;        // Use momentum for control variates
    float momentum_beta = 0.9f;         // Momentum coefficient
    bool adaptive_server_lr = true;     // Adapt server learning rate
};

struct ConvergenceConfig {
    // Algorithm selection
    bool enable_fedprox = true;
    bool enable_scaffold = true;
    bool enable_gradient_compression = false;
    
    // Staleness handling
    uint32_t max_staleness_steps = 10;     // Maximum allowed gradient staleness
    float staleness_penalty_factor = 0.9f; // Weight reduction for stale gradients
    bool use_age_weighted_averaging = true;
    
    // Convergence detection
    float convergence_threshold = 1e-6f;   // Loss improvement threshold for convergence
    uint32_t convergence_window = 100;     // Steps to check for convergence
    float divergence_threshold = 2.0f;     // Loss increase indicating divergence
    uint32_t divergence_window = 20;       // Steps to check for divergence
    
    // Validation checkpoints
    uint32_t validation_frequency = 50;    // Run validation every N steps
    float validation_split = 0.1f;         // Fraction of data for validation
    bool enable_early_stopping = true;
    uint32_t early_stopping_patience = 200; // Steps without improvement before stopping
    
    // Quality monitoring
    bool enable_gradient_monitoring = true;
    bool enable_parameter_monitoring = true;
    float anomaly_detection_threshold = 3.0f; // Standard deviations for anomaly detection
    
    FedProxConfig fedprox_config;
    SCAFFOLDConfig scaffold_config;
};

class ConvergenceQualityManager {
public:
    explicit ConvergenceQualityManager(const ConvergenceConfig& config = {});
    ~ConvergenceQualityManager();
    
    // Gradient processing with advanced algorithms
    std::vector<float> process_gradients_fedprox(const std::vector<GradientState>& node_gradients,
                                               const std::vector<float>& global_model);
    std::vector<float> process_gradients_scaffold(const std::vector<GradientState>& node_gradients,
                                                 const std::vector<float>& global_model);
    std::vector<float> combine_gradients_with_staleness_handling(const std::vector<GradientState>& node_gradients);
    
    // Model state management
    bool update_global_model(const std::vector<float>& new_parameters, uint64_t global_step);
    std::vector<float> get_global_model() const;
    bool save_model_checkpoint(const std::string& checkpoint_path, const ModelSnapshot& snapshot);
    bool load_model_checkpoint(const std::string& checkpoint_path, ModelSnapshot& snapshot);
    
    // Convergence monitoring
    ConvergenceMetrics analyze_convergence(const std::vector<float>& current_loss_history,
                                         const std::vector<float>& current_gradients);
    bool is_training_converged() const;
    bool is_training_diverging() const;
    std::string get_convergence_status() const;
    
    // Validation and quality checks
    struct ValidationResult {
        float validation_loss;
        float validation_accuracy;
        std::unordered_map<std::string, float> metrics;
        bool validation_successful;
        std::string error_message;
    };
    
    ValidationResult run_validation_checkpoint(const std::vector<float>& validation_data,
                                             const std::vector<float>& validation_labels);
    bool detect_convergence_anomalies(const std::vector<GradientState>& gradients);
    
    // Advanced aggregation methods
    std::vector<float> weighted_gradient_aggregation(const std::vector<GradientState>& gradients,
                                                   const std::vector<float>& node_weights);
    std::vector<float> momentum_based_aggregation(const std::vector<GradientState>& gradients,
                                                 float momentum_factor = 0.9f);
    std::vector<float> adaptive_aggregation(const std::vector<GradientState>& gradients,
                                           const std::vector<float>& node_reliabilities);
    
    // Control variates management (SCAFFOLD)
    bool initialize_control_variates(const std::vector<std::string>& node_ids,
                                   size_t parameter_count);
    bool update_control_variates(const std::string& node_id,
                               const std::vector<float>& local_gradients,
                               const std::vector<float>& global_gradients);
    std::vector<float> get_control_variates(const std::string& node_id) const;
    
    // Proximal term computation (FedProx)
    std::vector<float> compute_proximal_term(const std::vector<float>& local_parameters,
                                            const std::vector<float>& global_parameters,
                                            float mu);
    float adapt_proximal_mu(const std::vector<GradientState>& gradients);
    
    // Staleness handling
    float calculate_staleness_weight(uint64_t current_step, uint64_t gradient_step);
    std::vector<float> apply_staleness_correction(const std::vector<GradientState>& gradients,
                                                 uint64_t current_global_step);
    
    // Model quality assessment
    struct QualityAssessment {
        float gradient_consistency_score;   // How consistent gradients are across nodes
        float convergence_stability_score;  // How stable the convergence is
        float parameter_quality_score;      // Quality of learned parameters
        float overall_quality_score;        // Overall training quality (0.0-1.0)
        std::vector<std::string> quality_issues; // Identified quality problems
    };
    
    QualityAssessment assess_training_quality(const std::vector<GradientState>& recent_gradients,
                                            const std::vector<float>& recent_losses);
    
    // Configuration and monitoring
    void update_config(const ConvergenceConfig& new_config);
    ConvergenceConfig get_config() const;
    
    struct ConvergenceStats {
        uint64_t total_global_steps;
        uint64_t steps_since_last_improvement;
        float best_validation_loss;
        float current_learning_rate;
        uint32_t active_nodes;
        float average_gradient_staleness;
        float convergence_rate;
        std::chrono::steady_clock::time_point last_checkpoint;
        std::string current_algorithm;          // "FedProx", "SCAFFOLD", "Vanilla"
    };
    
    ConvergenceStats get_convergence_stats() const;
    bool export_convergence_report(const std::string& output_path) const;

private:
    ConvergenceConfig config_;
    mutable std::mutex config_mutex_;
    
    // Global model state
    mutable std::mutex model_mutex_;
    std::vector<float> global_model_;
    std::vector<ModelSnapshot> model_history_;
    uint64_t current_global_step_;
    
    // Control variates for SCAFFOLD
    mutable std::mutex control_variates_mutex_;
    std::unordered_map<std::string, std::vector<float>> node_control_variates_;
    std::vector<float> server_control_variates_;
    
    // Convergence tracking
    mutable std::mutex convergence_mutex_;
    std::vector<float> loss_history_;
    ConvergenceMetrics current_metrics_;
    std::vector<ValidationResult> validation_history_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    ConvergenceStats stats_;
    
    // FedProx state
    float current_proximal_mu_;
    std::vector<float> gradient_heterogeneity_history_;
    
    // SCAFFOLD state
    float current_server_lr_;
    std::vector<float> server_momentum_;
    
    // Helper methods
    float compute_gradient_heterogeneity(const std::vector<GradientState>& gradients);
    std::vector<float> compute_gradient_mean(const std::vector<GradientState>& gradients);
    float compute_gradient_variance(const std::vector<GradientState>& gradients,
                                   const std::vector<float>& mean_gradient);
    
    bool detect_gradient_anomalies(const std::vector<float>& gradients);
    bool detect_parameter_anomalies(const std::vector<float>& parameters);
    
    void update_loss_history(float new_loss);
    void update_convergence_metrics();
    
    // Advanced aggregation helpers
    std::vector<float> apply_byzantine_robust_aggregation(const std::vector<GradientState>& gradients);
    std::vector<float> apply_median_aggregation(const std::vector<GradientState>& gradients);
    std::vector<float> apply_trimmed_mean_aggregation(const std::vector<GradientState>& gradients,
                                                     float trim_fraction = 0.1f);
};

// Specialized algorithms implementation
namespace algorithms {

class FedProxOptimizer {
public:
    static std::vector<float> compute_fedprox_update(const std::vector<float>& local_gradients,
                                                    const std::vector<float>& local_parameters,
                                                    const std::vector<float>& global_parameters,
                                                    float learning_rate,
                                                    float mu);
    
    static float estimate_optimal_mu(const std::vector<GradientState>& gradients);
    static std::vector<float> adaptive_mu_schedule(const std::vector<float>& heterogeneity_history);
};

class SCAFFOLDOptimizer {
public:
    static std::vector<float> compute_scaffold_update(const std::vector<float>& local_gradients,
                                                     const std::vector<float>& client_control_variates,
                                                     const std::vector<float>& server_control_variates,
                                                     float learning_rate);
    
    static std::vector<float> update_client_control_variates(const std::vector<float>& old_variates,
                                                           const std::vector<float>& local_gradients,
                                                           const std::vector<float>& global_gradients,
                                                           float momentum_beta);
    
    static std::vector<float> update_server_control_variates(const std::vector<float>& old_server_variates,
                                                           const std::vector<std::vector<float>>& client_variates,
                                                           float server_learning_rate);
};

class StalenessCompensator {
public:
    static std::vector<float> apply_staleness_correction(const std::vector<GradientState>& gradients,
                                                        uint64_t current_step,
                                                        float penalty_factor);
    
    static std::vector<float> bounded_staleness_aggregation(const std::vector<GradientState>& gradients,
                                                           uint32_t max_staleness,
                                                           uint64_t current_step);
    
    static float compute_staleness_weight(uint64_t gradient_age,
                                         uint32_t max_staleness,
                                         const std::string& weighting_scheme = "exponential");
};

} // namespace algorithms
} // namespace convergence
