#pragma once

#include "p2p_network.hpp"
#include "matrix.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <memory>

namespace p2p {

enum class RecoveryStatus {
    INITIATED,
    IN_PROGRESS,
    COMPLETED,
    FAILED,
    TIMEOUT
};

struct ModelStateSnapshot {
    std::vector<Matrix> parameters;
    uint64_t training_step = 0;
    uint64_t epoch = 0;
    std::chrono::steady_clock::time_point timestamp;
    std::string checkpoint_hash;
};

struct HeartbeatInfo {
    std::chrono::steady_clock::time_point last_seen;
    uint64_t last_sequence_number = 0;
    bool is_alive = true;
};

struct HeartbeatMessage {
    std::string sender_id;
    uint64_t timestamp = 0;
    uint64_t sequence_number = 0;
    std::string model_state_hash;
};

struct PartitionEvent {
    std::chrono::steady_clock::time_point timestamp;
    std::vector<std::string> lost_peers;
    std::string partition_id;
};

struct RecoverySession {
    std::string session_id;
    std::vector<std::string> recovered_peers;
    std::chrono::steady_clock::time_point start_time;
    RecoveryStatus status = RecoveryStatus::INITIATED;
    std::string error_message;
};

struct PartitionState {
    bool is_partitioned = false;
    std::chrono::steady_clock::time_point partition_start_time;
    std::vector<std::string> lost_peers;
};

struct ConnectivityMatrix {
    std::chrono::steady_clock::time_point timestamp;
    std::vector<std::string> peer_ids;
    std::vector<std::vector<bool>> connectivity;  // connectivity[i][j] = can peer i reach peer j
};

struct StateReconciliationRequest {
    std::string session_id;
    std::string requester_id;
    std::string target_peer_id;
    std::vector<uint8_t> our_merkle_root;
    std::chrono::steady_clock::time_point timestamp;
};

struct RecoveryConfig {
    uint32_t heartbeat_interval_ms = 5000;        // Send heartbeat every 5 seconds
    uint32_t partition_timeout_ms = 15000;        // Consider peer lost after 15 seconds
    uint32_t recovery_timeout_ms = 60000;         // Recovery timeout after 60 seconds
    uint32_t partition_detection_interval_ms = 10000;  // Check for partitions every 10 seconds
    uint32_t recovery_check_interval_ms = 2000;   // Check recovery progress every 2 seconds
    uint32_t merkle_tree_depth = 10;              // Depth for state verification
    float consensus_threshold = 0.67f;            // Require 67% agreement for state reconciliation
};

struct RecoveryStats {
    uint64_t partitions_detected = 0;
    uint64_t successful_recoveries = 0;
    uint64_t failed_recoveries = 0;
    uint64_t total_recovery_time_ms = 0;
    std::chrono::steady_clock::time_point last_partition_time;
    std::chrono::steady_clock::time_point last_recovery_time;
};

class PartitionRecoveryManager {
public:
    explicit PartitionRecoveryManager(
        std::shared_ptr<P2PNetwork> network,
        const RecoveryConfig& config = RecoveryConfig{});
    
    ~PartitionRecoveryManager();
    
    // Main control interface
    bool start_recovery_monitoring();
    void stop_recovery_monitoring();
    
    // Model state management
    void register_model_state_provider(std::function<ModelStateSnapshot()> provider);
    void register_model_state_applier(std::function<bool(const ModelStateSnapshot&)> applier);
    
    // Partition handling
    bool handle_partition_detected(const std::vector<std::string>& lost_peers);
    bool handle_partition_healed(const std::vector<std::string>& recovered_peers);
    
    // State reconciliation
    bool initiate_state_reconciliation(const std::vector<std::string>& recovered_peers);
    
    // Statistics and monitoring
    RecoveryStats get_recovery_stats() const;
    PartitionState get_current_partition_state() const {
        std::lock_guard<std::mutex> lock(partition_mutex_);
        return current_partition_state_;
    }
    
    // Configuration
    void update_config(const RecoveryConfig& config) { config_ = config; }
    RecoveryConfig get_config() const { return config_; }

private:
    // Core monitoring threads
    void heartbeat_monitor_thread();
    void partition_detection_thread();
    void recovery_coordination_thread();
    
    // Heartbeat management
    void send_heartbeat_to_peers();
    void check_missed_heartbeats();
    
    // Partition detection
    void analyze_connectivity_patterns();
    void detect_network_partitions();
    
    // State reconciliation
    bool wait_for_reconciliation_responses(const std::vector<StateReconciliationRequest>& requests);
    bool perform_state_reconciliation(const std::unordered_map<std::string, ModelStateSnapshot>& peer_states);
    
    // Merkle tree utilities for state verification
    std::vector<uint8_t> build_merkle_tree(const ModelStateSnapshot& state);
    std::string calculate_model_state_hash(const ModelStateSnapshot& state);
    
    // Recovery session management
    void process_recovery_sessions();
    void cleanup_completed_sessions();
    
    // Utility functions
    std::string bytes_to_hex(const std::vector<uint8_t>& bytes);
    std::string generate_partition_id();
    std::string generate_recovery_session_id();
    
    // Serialization helpers
    std::vector<uint8_t> serialize_heartbeat(const HeartbeatMessage& heartbeat);
    std::vector<uint8_t> serialize_reconciliation_request(const StateReconciliationRequest& request);
    
    // Simulation helpers (for testing)
    bool simulate_reconciliation_response(const StateReconciliationRequest& request, 
                                        ModelStateSnapshot& peer_state);
    
    // Network and configuration
    std::shared_ptr<P2PNetwork> network_;
    RecoveryConfig config_;
    
    // Threading and synchronization
    std::atomic<bool> recovery_active_;
    std::thread heartbeat_thread_;
    std::thread partition_detector_thread_;
    std::thread recovery_thread_;
    
    // Heartbeat tracking
    mutable std::mutex heartbeat_mutex_;
    std::condition_variable heartbeat_cv_;
    std::unordered_map<std::string, HeartbeatInfo> peer_heartbeats_;
    std::atomic<uint64_t> heartbeat_sequence_{0};
    
    // Partition tracking
    mutable std::mutex partition_mutex_;
    std::condition_variable partition_cv_;
    PartitionState current_partition_state_;
    std::unordered_map<std::string, PartitionEvent> active_partitions_;
    
    // Connectivity analysis
    mutable std::mutex connectivity_mutex_;
    std::vector<ConnectivityMatrix> connectivity_history_;
    
    // Recovery session management
    mutable std::mutex recovery_mutex_;
    std::condition_variable recovery_cv_;
    std::unordered_map<std::string, RecoverySession> active_recovery_sessions_;
    
    // Model state management
    mutable std::mutex state_mutex_;
    std::function<ModelStateSnapshot()> model_state_provider_;
    std::function<bool(const ModelStateSnapshot&)> model_state_applier_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    RecoveryStats recovery_stats_;
};

// Factory function for easy integration
std::unique_ptr<PartitionRecoveryManager> create_partition_recovery_manager(
    std::shared_ptr<P2PNetwork> network,
    const RecoveryConfig& config = RecoveryConfig{});

} // namespace p2p
