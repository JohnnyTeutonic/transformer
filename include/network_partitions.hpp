#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <mutex>
#include <memory>
#include <chrono>
#include <atomic>
#include <thread>
#include <functional>

namespace network_partitions {

struct NodeConnectivity {
    std::string node_id;
    std::unordered_set<std::string> reachable_nodes;
    std::unordered_set<std::string> unreachable_nodes;
    std::chrono::steady_clock::time_point last_connectivity_check;
    uint32_t consecutive_failures;
    bool is_isolated;
    float connection_quality_score;  // 0.0 to 1.0
};

struct PartitionInfo {
    std::string partition_id;
    std::unordered_set<std::string> nodes_in_partition;
    std::string partition_leader;            // Node coordinating this partition
    uint32_t partition_size;
    bool is_majority_partition;              // Contains majority of original nodes
    bool is_training_active;                 // Whether training continues in this partition
    float training_quality_degradation;     // 0.0 to 1.0, higher = more degraded
    std::chrono::steady_clock::time_point partition_detected_time;
    std::chrono::steady_clock::time_point last_model_checkpoint_time;
};

struct NetworkStatus {
    uint32_t total_nodes;
    uint32_t reachable_nodes;
    uint32_t unreachable_nodes;
    uint32_t active_partitions;
    bool is_partitioned;
    std::string largest_partition_id;
    float overall_connectivity_score;       // 0.0 to 1.0
    std::chrono::steady_clock::time_point last_status_update;
};

struct CheckpointMetadata {
    std::string checkpoint_id;
    std::vector<float> model_parameters;
    std::string partition_id;               // Which partition created this checkpoint
    uint64_t global_step;
    float validation_loss;
    uint32_t nodes_contributing;            // Number of nodes that contributed
    std::unordered_set<std::string> contributing_nodes;
    std::chrono::steady_clock::time_point creation_time;
    bool is_reconciliation_checkpoint;     // Created during partition healing
};

enum class PartitionStrategy {
    PAUSE_AND_WAIT,              // Pause training until network heals
    MAJORITY_CONTINUES,          // Only majority partition continues training
    DEGRADED_TRAINING,           // All partitions continue with quality degradation
    CHECKPOINT_AND_RESTART       // Save state and restart when healed
};

struct PartitionHandlingConfig {
    PartitionStrategy strategy = PartitionStrategy::DEGRADED_TRAINING;
    
    // Connectivity monitoring
    uint32_t connectivity_check_interval_ms = 10000;    // How often to check connectivity
    uint32_t ping_timeout_ms = 5000;                    // Timeout for ping operations
    uint32_t max_consecutive_failures = 3;              // Failures before marking node unreachable
    
    // Partition detection
    float partition_detection_threshold = 0.5f;         // Connectivity loss threshold
    uint32_t partition_stability_window_ms = 30000;     // Wait before declaring stable partition
    
    // Training continuation
    float minority_partition_quality_penalty = 0.5f;    // Quality penalty for minority partitions
    uint32_t min_nodes_for_training = 2;                // Minimum nodes to continue training
    bool allow_single_node_training = false;            // Allow training with just one node
    
    // Checkpointing
    uint32_t checkpoint_interval_during_partition_ms = 60000; // Frequent checkpoints during partition
    uint32_t max_checkpoints_per_partition = 10;        // Limit checkpoints per partition
    bool enable_automatic_reconciliation = true;        // Auto-reconcile when partition heals
    
    // Quality degradation
    float max_acceptable_degradation = 0.8f;            // Max training quality loss before pause
    bool enable_adaptive_degradation = true;            // Adjust degradation based on partition size
    
    // Recovery settings
    uint32_t partition_healing_detection_window_ms = 60000; // How long to confirm partition healed
    uint32_t reconciliation_timeout_ms = 300000;        // Max time for reconciliation process
};

class NetworkPartitionHandler {
public:
    explicit NetworkPartitionHandler(const PartitionHandlingConfig& config = {});
    ~NetworkPartitionHandler();
    
    // Node registration and management
    void register_node(const std::string& node_id, const std::string& ip_address, uint16_t port);
    void unregister_node(const std::string& node_id);
    std::vector<std::string> get_registered_nodes() const;
    
    // Connectivity monitoring
    void start_connectivity_monitoring();
    void stop_connectivity_monitoring();
    NodeConnectivity check_node_connectivity(const std::string& node_id);
    NetworkStatus get_network_status() const;
    
    // Partition detection and management
    std::vector<PartitionInfo> detect_network_partitions();
    PartitionInfo get_current_partition(const std::string& node_id) const;
    bool is_network_partitioned() const;
    std::vector<PartitionInfo> get_all_partitions() const;
    
    // Training coordination during partitions
    struct TrainingDecision {
        bool should_continue_training;
        float quality_degradation_factor;      // 0.0 to 1.0
        std::string reasoning;
        std::vector<std::string> participating_nodes;
        uint32_t estimated_performance_impact_percent;
    };
    
    TrainingDecision make_training_decision(const std::string& requesting_node_id);
    void update_training_status(const std::string& partition_id, bool is_active, float current_quality);
    
    // Checkpoint management during partitions
    std::string create_partition_checkpoint(const std::vector<float>& model_parameters,
                                           uint64_t global_step,
                                           const std::string& partition_id);
    bool save_checkpoint_to_persistent_storage(const std::string& checkpoint_id,
                                              const std::string& storage_path);
    std::vector<CheckpointMetadata> get_available_checkpoints() const;
    
    // Partition healing and reconciliation
    struct ReconciliationResult {
        bool reconciliation_successful;
        std::string chosen_checkpoint_id;
        std::vector<std::string> merged_nodes;
        float data_consistency_score;          // 0.0 to 1.0
        std::string reconciliation_summary;
        uint32_t reconciliation_time_ms;
    };
    
    bool detect_partition_healing();
    ReconciliationResult reconcile_partitions(const std::vector<std::string>& healing_partitions);
    std::vector<float> merge_model_states(const std::vector<CheckpointMetadata>& checkpoints);
    
    // Quality assessment during partitions
    struct QualityMetrics {
        float connectivity_quality;            // Network connectivity quality
        float training_consensus_quality;      // How well nodes agree on training
        float model_consistency_quality;       // Consistency of model updates
        float overall_training_quality;        // Combined quality score
        uint32_t effective_batch_size;         // Effective batch size with current nodes
        float expected_convergence_impact;     // Impact on convergence time
    };
    
    QualityMetrics assess_training_quality(const std::string& partition_id) const;
    void update_quality_metrics(const std::string& partition_id, const QualityMetrics& metrics);
    
    // Event callbacks for integration
    using PartitionDetectedCallback = std::function<void(const PartitionInfo&)>;
    using PartitionHealedCallback = std::function<void(const std::vector<std::string>&)>;
    using TrainingQualityCallback = std::function<void(float quality_degradation)>;
    
    void set_partition_detected_callback(PartitionDetectedCallback callback);
    void set_partition_healed_callback(PartitionHealedCallback callback);
    void set_training_quality_callback(TrainingQualityCallback callback);
    
    // Manual control and override
    void force_partition_detection();
    void force_checkpoint_creation(const std::string& reason = "manual");
    void override_training_decision(bool should_continue, const std::string& reason);
    void trigger_manual_reconciliation();
    
    // Configuration and diagnostics
    void update_config(const PartitionHandlingConfig& new_config);
    PartitionHandlingConfig get_config() const;
    
    struct PartitionStats {
        uint32_t total_partitions_detected;
        uint32_t successful_reconciliations;
        uint32_t failed_reconciliations;
        uint32_t total_checkpoints_created;
        float average_partition_duration_minutes;
        float average_reconciliation_time_ms;
        std::chrono::steady_clock::time_point last_partition_event;
    };
    
    PartitionStats get_partition_statistics() const;
    bool export_partition_report(const std::string& output_path) const;

private:
    PartitionHandlingConfig config_;
    mutable std::mutex config_mutex_;
    
    // Node registry
    struct NodeInfo {
        std::string node_id;
        std::string ip_address;
        uint16_t port;
        std::chrono::steady_clock::time_point last_seen;
        bool is_registered;
    };
    
    mutable std::mutex nodes_mutex_;
    std::unordered_map<std::string, NodeInfo> registered_nodes_;
    std::unordered_map<std::string, NodeConnectivity> node_connectivity_;
    
    // Partition state
    mutable std::mutex partitions_mutex_;
    std::vector<PartitionInfo> current_partitions_;
    std::string my_partition_id_;
    bool is_partitioned_;
    
    // Checkpoint management
    mutable std::mutex checkpoints_mutex_;
    std::unordered_map<std::string, CheckpointMetadata> checkpoints_;
    uint64_t next_checkpoint_id_;
    
    // Quality tracking
    mutable std::mutex quality_mutex_;
    std::unordered_map<std::string, QualityMetrics> partition_quality_;
    
    // Background threads
    std::atomic<bool> monitoring_active_{false};
    std::thread connectivity_monitor_thread_;
    std::thread partition_detector_thread_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    PartitionStats stats_;
    
    // Callbacks
    PartitionDetectedCallback partition_detected_callback_;
    PartitionHealedCallback partition_healed_callback_;
    TrainingQualityCallback training_quality_callback_;
    
    // Helper methods
    bool ping_node(const std::string& node_id, uint32_t timeout_ms);
    void run_connectivity_monitoring();
    void run_partition_detection();
    
    std::string generate_partition_id();
    std::string generate_checkpoint_id();
    
    // Connectivity assessment
    void update_node_connectivity(const std::string& node_id);
    float calculate_connectivity_score(const NodeConnectivity& connectivity) const;
    
    // Partition algorithms
    std::vector<std::vector<std::string>> find_connected_components();
    PartitionInfo create_partition_from_nodes(const std::vector<std::string>& nodes);
    bool is_majority_partition(const PartitionInfo& partition) const;
    std::string select_partition_leader(const std::vector<std::string>& nodes);
    
    // Training decision logic
    TrainingDecision make_degraded_training_decision(const PartitionInfo& partition);
    TrainingDecision make_majority_only_decision(const PartitionInfo& partition);
    TrainingDecision make_pause_and_wait_decision(const PartitionInfo& partition);
    TrainingDecision make_checkpoint_restart_decision(const PartitionInfo& partition);
    
    // Quality assessment
    float calculate_batch_size_impact(uint32_t current_nodes, uint32_t original_nodes) const;
    float calculate_consensus_quality(const PartitionInfo& partition) const;
    float estimate_convergence_impact(const PartitionInfo& partition) const;
    
    // Reconciliation algorithms
    CheckpointMetadata select_best_checkpoint(const std::vector<CheckpointMetadata>& candidates);
    std::vector<float> weighted_model_average(const std::vector<CheckpointMetadata>& checkpoints);
    bool validate_model_consistency(const std::vector<CheckpointMetadata>& checkpoints);
    
    // Cleanup and maintenance
    void cleanup_old_checkpoints();
    void cleanup_stale_partitions();
    void update_statistics();
};

// Utility functions
namespace utils {

// Network connectivity testing
bool test_tcp_connection(const std::string& ip_address, uint16_t port, uint32_t timeout_ms = 5000);
float measure_network_latency(const std::string& ip_address, uint32_t timeout_ms = 1000);
std::pair<float, float> measure_network_bandwidth(const std::string& ip_address, uint16_t port);

// Graph algorithms for partition detection
std::vector<std::vector<std::string>> find_connected_components_graph(
    const std::unordered_map<std::string, std::unordered_set<std::string>>& adjacency_list);

// Checkpoint utilities
bool compress_checkpoint_data(const std::vector<float>& data, std::vector<uint8_t>& compressed_data);
bool decompress_checkpoint_data(const std::vector<uint8_t>& compressed_data, std::vector<float>& data);
std::string calculate_checkpoint_hash(const std::vector<float>& data);

} // namespace utils
} // namespace network_partitions
