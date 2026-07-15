#pragma once

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <random>
#include <atomic>
#include <chrono>
#include <torch/torch.h>
#include "distributed_transformer.hpp"
#include "distributed_rlhf.hpp"

namespace experiments {

// Experiment configuration
struct ExperimentConfig {
    std::string experiment_id;
    std::string experiment_name;
    std::string description;
    std::vector<std::string> variant_names;
    std::unordered_map<std::string, float> traffic_allocation;  // variant -> percentage
    uint64_t start_time;
    uint64_t end_time;
    std::vector<std::string> metrics_to_track;
    bool is_active = true;
};

// Variant information
struct Variant {
    std::string variant_id;
    std::string experiment_id;
    std::unordered_map<std::string, std::any> parameters;
    std::shared_ptr<model::DistributedTransformer> model;
    std::shared_ptr<rlhf::DistributedRLHF> rlhf_trainer;
};

// User assignment
struct UserAssignment {
    std::string user_id;
    std::string experiment_id;
    std::string variant_id;
    uint64_t assignment_time;
    std::unordered_map<std::string, std::string> user_attributes;
};

// Metric data point
struct MetricPoint {
    std::string metric_name;
    double value;
    uint64_t timestamp;
    std::string variant_id;
    std::string user_id;
    std::unordered_map<std::string, std::string> metadata;
};

// A/B Testing Framework
class ABTestingFramework {
public:
    ABTestingFramework();
    ~ABTestingFramework();
    
    // Experiment management
    std::string create_experiment(const ExperimentConfig& config);
    void start_experiment(const std::string& experiment_id);
    void stop_experiment(const std::string& experiment_id);
    void delete_experiment(const std::string& experiment_id);
    
    // Variant management
    void register_variant(const std::string& experiment_id, const Variant& variant);
    void update_variant_allocation(const std::string& experiment_id,
                                  const std::unordered_map<std::string, float>& allocation);
    
    // User assignment
    std::string assign_user_to_variant(const std::string& user_id,
                                      const std::string& experiment_id);
    UserAssignment get_user_assignment(const std::string& user_id,
                                      const std::string& experiment_id);
    
    // Metric tracking
    void track_metric(const MetricPoint& metric);
    void track_batch_metrics(const std::vector<MetricPoint>& metrics);
    
    // Analysis
    struct ExperimentResults {
        std::string experiment_id;
        std::unordered_map<std::string, double> variant_metrics;  // variant -> metric value
        std::unordered_map<std::string, double> confidence_intervals;
        double p_value;
        std::string winning_variant;
        bool is_statistically_significant;
        double effect_size;
    };
    
    ExperimentResults analyze_experiment(const std::string& experiment_id,
                                        const std::string& metric_name);
    
    // Reporting
    std::vector<ExperimentResults> get_all_results(const std::string& experiment_id);
    void export_results_to_csv(const std::string& experiment_id, const std::string& path);
    
    // Multi-armed bandit optimization
    void enable_bandit_optimization(const std::string& experiment_id,
                                   const std::string& optimization_metric);
    
private:
    std::unordered_map<std::string, ExperimentConfig> experiments_;
    std::unordered_map<std::string, std::vector<Variant>> variants_;
    std::unordered_map<std::string, UserAssignment> user_assignments_;
    std::vector<MetricPoint> metrics_;
    
    // Random assignment
    std::mt19937 rng_;
    std::string select_variant_randomly(const ExperimentConfig& config);
    
    // Statistical analysis
    double compute_p_value(const std::vector<double>& control,
                          const std::vector<double>& treatment);
    double compute_confidence_interval(const std::vector<double>& data);
    double compute_effect_size(const std::vector<double>& control,
                              const std::vector<double>& treatment);
    
    // Multi-armed bandit
    struct BanditState {
        std::vector<double> rewards;
        std::vector<int> counts;
        std::string algorithm;  // "epsilon_greedy", "ucb", "thompson_sampling"
    };
    std::unordered_map<std::string, BanditState> bandit_states_;
    
    std::string select_variant_bandit(const std::string& experiment_id);
    void update_bandit(const std::string& experiment_id,
                      const std::string& variant_id,
                      double reward);
};

// Sequential testing for early stopping
class SequentialTesting {
public:
    SequentialTesting(double alpha = 0.05, double beta = 0.2);
    
    // Sequential probability ratio test (SPRT)
    enum class TestResult {
        CONTINUE,
        ACCEPT_NULL,
        REJECT_NULL
    };
    
    TestResult sprt_test(const std::vector<double>& observations);
    
    // Group sequential design
    TestResult group_sequential_test(const std::vector<std::vector<double>>& group_observations,
                                    int current_stage);
    
    // Adaptive designs
    void update_sample_size(const std::vector<double>& interim_results);
    int get_recommended_sample_size();
    
private:
    double alpha_;  // Type I error rate
    double beta_;   // Type II error rate
    double log_likelihood_ratio_;
    
    // Boundaries for sequential testing
    double upper_boundary_;
    double lower_boundary_;
    
    // Adaptive sample size
    int current_sample_size_;
    int max_sample_size_;
};

// Bayesian A/B testing
class BayesianABTesting {
public:
    BayesianABTesting();
    
    // Bayesian analysis
    struct BayesianResults {
        std::unordered_map<std::string, double> posterior_means;
        std::unordered_map<std::string, double> posterior_variances;
        std::unordered_map<std::string, double> probability_of_being_best;
        double expected_loss;
        double value_remaining;
    };
    
    BayesianResults analyze(const std::unordered_map<std::string, std::vector<double>>& variant_data);
    
    // Thompson sampling for exploration
    std::string thompson_sampling(const std::unordered_map<std::string, std::pair<double, double>>& beta_params);
    
    // Expected improvement
    double compute_expected_improvement(const BayesianResults& results);
    
private:
    // Beta distribution parameters
    std::unordered_map<std::string, std::pair<double, double>> beta_params_;
    
    // Normal distribution parameters
    std::unordered_map<std::string, std::pair<double, double>> normal_params_;
    
    // Update posteriors
    void update_beta_posterior(const std::string& variant,
                              int successes, int failures);
    void update_normal_posterior(const std::string& variant,
                                const std::vector<double>& observations);
    
    // Monte Carlo sampling
    std::vector<double> sample_posterior(const std::string& variant, int num_samples);
};

// Feature flagging system
class FeatureFlags {
public:
    FeatureFlags();
    
    // Feature flag configuration
    struct FeatureFlag {
        std::string flag_name;
        bool is_enabled;
        std::unordered_map<std::string, std::any> configuration;
        std::vector<std::string> enabled_for_users;
        double rollout_percentage;
        std::string experiment_id;  // Associated A/B test
    };
    
    // Flag management
    void create_flag(const FeatureFlag& flag);
    void update_flag(const std::string& flag_name, const FeatureFlag& flag);
    void delete_flag(const std::string& flag_name);
    
    // Flag evaluation
    bool is_enabled(const std::string& flag_name, const std::string& user_id = "");
    std::any get_configuration(const std::string& flag_name, const std::string& key);
    
    // Rollout management
    void set_rollout_percentage(const std::string& flag_name, double percentage);
    void add_user_to_rollout(const std::string& flag_name, const std::string& user_id);
    
    // Integration with A/B testing
    void link_to_experiment(const std::string& flag_name, const std::string& experiment_id);
    
private:
    std::unordered_map<std::string, FeatureFlag> flags_;
    std::mt19937 rng_;
    
    bool evaluate_rollout(const FeatureFlag& flag, const std::string& user_id);
};

// Model comparison framework
class ModelComparison {
public:
    ModelComparison();
    
    // Register models for comparison
    void register_model(const std::string& model_id,
                       std::shared_ptr<model::DistributedTransformer> model);
    
    // Head-to-head comparison
    struct ComparisonResult {
        std::string model_a_id;
        std::string model_b_id;
        double model_a_score;
        double model_b_score;
        std::string winner;
        double confidence;
        std::unordered_map<std::string, double> metric_breakdown;
    };
    
    ComparisonResult compare_models(const std::string& model_a_id,
                                   const std::string& model_b_id,
                                   const std::vector<std::string>& test_prompts);
    
    // Tournament-style comparison
    std::vector<std::pair<std::string, double>> run_tournament(
        const std::vector<std::string>& model_ids,
        const std::vector<std::string>& test_prompts);
    
    // Online evaluation
    void start_online_evaluation(const std::vector<std::string>& model_ids);
    ComparisonResult get_online_results(const std::string& model_a_id,
                                       const std::string& model_b_id);
    
private:
    std::unordered_map<std::string, std::shared_ptr<model::DistributedTransformer>> models_;
    std::unordered_map<std::string, std::vector<double>> online_scores_;
    
    // Evaluation metrics
    double compute_perplexity(std::shared_ptr<model::DistributedTransformer> model,
                             const std::string& text);
    double compute_response_quality(std::shared_ptr<model::DistributedTransformer> model,
                                   const std::string& prompt);
    
    // ELO rating system
    std::unordered_map<std::string, double> elo_ratings_;
    void update_elo_ratings(const std::string& winner, const std::string& loser);
};

} // namespace experiments
