#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <mutex>
#include <memory>
#include <chrono>
#include <queue>
#include <atomic>

namespace byzantine {

struct GradientFingerprint {
    std::string node_id;
    std::vector<float> gradient_sample;  // Sampled gradient values for clustering
    float l2_norm;                       // L2 norm of full gradient
    float cosine_similarity;             // Similarity to cluster center
    uint32_t timestamp;                  // When gradient was produced
    size_t gradient_size;                // Total gradient size
    std::vector<float> layer_norms;      // Per-layer gradient norms
};

struct ClusterAnalysis {
    std::vector<std::vector<std::string>> clusters; // Node IDs grouped by similarity
    std::vector<std::string> outlier_nodes;         // Nodes that don't fit any cluster
    float silhouette_score;                         // Quality of clustering
    size_t dominant_cluster_size;                   // Size of largest cluster
    float outlier_threshold;                        // Threshold used for outlier detection
};

struct CrossValidationTask {
    std::string task_id;
    std::vector<float> input_data;    // Known test input
    std::vector<float> expected_output; // Expected gradient output
    float tolerance;                  // Acceptable error tolerance
    std::vector<std::string> assigned_nodes; // Nodes that should validate this
    uint32_t created_timestamp;
};

struct CrossValidationResult {
    std::string node_id;
    std::string task_id;
    std::vector<float> computed_output;
    float error_magnitude;
    bool validation_passed;
    uint32_t completion_time_ms;
};

struct HistoricalBounds {
    float mean_l2_norm;
    float std_l2_norm;
    float max_acceptable_norm;
    float min_acceptable_norm;
    std::vector<float> layer_mean_norms;
    std::vector<float> layer_std_norms;
    uint32_t samples_count;
    std::chrono::steady_clock::time_point last_updated;
};

struct SuspiciousActivity {
    std::string node_id;
    std::string activity_type;     // "gradient_outlier", "cross_val_failure", "magnitude_violation", etc.
    std::string description;       // Human-readable description
    float severity_score;          // 0.0 to 1.0, higher = more suspicious
    uint32_t timestamp;
    std::unordered_map<std::string, float> metadata; // Additional context
};

struct ByzantineDetectionConfig {
    float outlier_threshold = 2.0f;           // Standard deviations for outlier detection
    size_t min_cluster_size = 3;              // Minimum nodes for a valid cluster
    float cosine_similarity_threshold = 0.8f; // Threshold for gradient similarity
    size_t gradient_sample_size = 1000;       // Number of gradient values to sample for clustering
    
    // Cross-validation settings
    uint32_t cross_val_frequency = 10;        // Run cross-validation every N gradient steps
    float cross_val_error_threshold = 0.1f;   // Max acceptable error in cross-validation
    size_t cross_val_tasks_per_node = 3;      // Number of validation tasks per node
    
    // Historical bounds settings
    size_t historical_window_size = 100;      // Number of gradients to track for bounds
    float magnitude_violation_multiplier = 3.0f; // How many std devs constitute violation
    
    // Reputation and quarantine
    float reputation_decay_rate = 0.95f;      // How quickly reputation recovers
    float quarantine_threshold = 0.3f;        // Reputation below this triggers quarantine
    uint32_t quarantine_duration_minutes = 30; // How long to quarantine suspicious nodes
    
    bool enable_gradient_clustering = true;
    bool enable_cross_validation = true;
    bool enable_magnitude_bounds = true;
    bool enable_adaptive_thresholds = true;
};

class ByzantineDetectionEngine {
public:
    explicit ByzantineDetectionEngine(const ByzantineDetectionConfig& config = {});
    ~ByzantineDetectionEngine();
    
    // Main detection methods
    ClusterAnalysis analyze_gradient_similarity(const std::vector<GradientFingerprint>& gradients);
    std::vector<std::string> detect_outlier_nodes(const std::vector<GradientFingerprint>& gradients);
    bool validate_gradient_magnitude(const std::string& node_id, const GradientFingerprint& gradient);
    
    // Cross-validation system
    std::vector<CrossValidationTask> generate_cross_validation_tasks(
        const std::vector<std::string>& participating_nodes);
    void submit_cross_validation_result(const CrossValidationResult& result);
    std::vector<std::string> get_cross_validation_failures() const;
    
    // Historical bounds management
    void update_historical_bounds(const std::vector<GradientFingerprint>& gradients);
    HistoricalBounds get_current_bounds() const;
    bool is_gradient_within_bounds(const GradientFingerprint& gradient) const;
    
    // Reputation system
    float get_node_reputation(const std::string& node_id) const;
    void update_node_reputation(const std::string& node_id, float reputation_delta);
    std::vector<std::string> get_quarantined_nodes() const;
    bool is_node_quarantined(const std::string& node_id) const;
    
    // Suspicious activity tracking
    void report_suspicious_activity(const SuspiciousActivity& activity);
    std::vector<SuspiciousActivity> get_recent_suspicious_activities(uint32_t last_n_minutes = 60) const;
    std::vector<std::string> get_most_suspicious_nodes(size_t top_n = 5) const;
    
    // Detection pipeline
    struct DetectionResult {
        std::vector<std::string> byzantine_nodes;
        std::vector<std::string> suspicious_nodes;
        std::vector<std::string> quarantined_nodes;
        ClusterAnalysis cluster_analysis;
        float detection_confidence;
        std::string detection_summary;
    };
    
    DetectionResult run_full_detection_pipeline(const std::vector<GradientFingerprint>& gradients);
    
    // Configuration and monitoring
    void update_config(const ByzantineDetectionConfig& new_config);
    ByzantineDetectionConfig get_config() const;
    
    struct DetectionStats {
        uint32_t total_gradients_analyzed;
        uint32_t outliers_detected;
        uint32_t cross_validation_tasks_completed;
        uint32_t cross_validation_failures;
        uint32_t nodes_quarantined;
        float average_cluster_quality;
        std::chrono::steady_clock::time_point last_detection_run;
    };
    
    DetectionStats get_detection_stats() const;
    void export_detection_report(const std::string& output_path) const;

private:
    ByzantineDetectionConfig config_;
    mutable std::mutex config_mutex_;
    
    // Historical data
    mutable std::mutex bounds_mutex_;
    HistoricalBounds current_bounds_;
    std::queue<GradientFingerprint> gradient_history_;
    
    // Cross-validation
    mutable std::mutex cross_val_mutex_;
    std::unordered_map<std::string, CrossValidationTask> pending_cross_val_tasks_;
    std::unordered_map<std::string, std::vector<CrossValidationResult>> cross_val_results_;
    uint32_t next_task_id_;
    
    // Reputation system
    mutable std::mutex reputation_mutex_;
    std::unordered_map<std::string, float> node_reputations_;
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> quarantine_expiry_;
    
    // Activity tracking
    mutable std::mutex activity_mutex_;
    std::vector<SuspiciousActivity> suspicious_activities_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    DetectionStats stats_;
    
    // Helper methods
    std::vector<float> sample_gradient_for_clustering(const std::vector<float>& full_gradient) const;
    float calculate_cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) const;
    std::vector<std::vector<size_t>> perform_kmeans_clustering(
        const std::vector<std::vector<float>>& data, size_t k) const;
    float calculate_silhouette_score(const std::vector<std::vector<float>>& data,
                                    const std::vector<std::vector<size_t>>& clusters) const;
    
    // Cross-validation helpers
    std::string generate_cross_validation_task_id();
    CrossValidationTask create_synthetic_validation_task(const std::vector<std::string>& nodes);
    bool validate_cross_validation_result(const CrossValidationResult& result) const;
    
    // Reputation helpers
    void decay_reputations();
    void process_quarantine_expiries();
    
    // Adaptive threshold adjustment
    void adjust_thresholds_based_on_history();
    float calculate_adaptive_outlier_threshold(const std::vector<float>& similarity_scores) const;
};

// Specialized detection algorithms
namespace algorithms {

class GradientClusteringDetector {
public:
    static ClusterAnalysis detect_outliers_dbscan(const std::vector<GradientFingerprint>& gradients,
                                                  float eps = 0.1f, size_t min_samples = 3);
    static ClusterAnalysis detect_outliers_isolation_forest(const std::vector<GradientFingerprint>& gradients,
                                                           float contamination = 0.1f);
    static std::vector<size_t> find_gradient_outliers_statistical(const std::vector<GradientFingerprint>& gradients,
                                                                 float z_threshold = 2.0f);
};

class CrossValidationOracle {
public:
    static std::vector<CrossValidationTask> generate_mathematical_tasks(size_t num_tasks,
                                                                       const std::vector<std::string>& nodes);
    static std::vector<CrossValidationTask> generate_model_consistency_tasks(size_t num_tasks,
                                                                            const std::vector<std::string>& nodes);
    static float evaluate_task_difficulty(const CrossValidationTask& task);
};

class ReputationScorer {
public:
    static float calculate_gradient_quality_score(const GradientFingerprint& gradient,
                                                 const HistoricalBounds& bounds);
    static float calculate_consistency_score(const std::vector<GradientFingerprint>& node_gradients);
    static float calculate_collaboration_score(const std::string& node_id,
                                             const std::vector<CrossValidationResult>& results);
};

} // namespace algorithms
} // namespace byzantine
