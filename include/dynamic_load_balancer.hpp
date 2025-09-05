#pragma once

#include "performance_profiler.hpp"
#include <vector>
#include <unordered_map>
#include <string>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <memory>

namespace load_balancing {

struct PeerCapabilities {
    uint32_t cpu_cores = 0;
    float memory_gb = 0.0f;
    uint32_t gpu_count = 0;
    float gpu_memory_gb = 0.0f;
    std::string gpu_compute_capability;
    float network_bandwidth_mbps = 0.0f;
    float compute_score = 0.0f;  // Normalized 0-1 score
};

struct PeerPerformance {
    float samples_per_second = 0.0f;
    float average_latency_ms = 0.0f;
    float cpu_usage_percent = 0.0f;
    float memory_usage_percent = 0.0f;
    float gpu_usage_percent = 0.0f;
    float network_bytes_per_sec = 0.0f;
    bool is_healthy = true;
};

struct PeerLoad {
    uint32_t assigned_batch_size = 0;
    float cpu_usage_percent = 0.0f;
    float memory_usage_percent = 0.0f;
    float gpu_usage_percent = 0.0f;
    float network_usage_percent = 0.0f;
};

struct PeerInfo {
    std::string peer_id;
    PeerCapabilities capabilities;
    PeerLoad current_load;
    std::vector<std::pair<std::chrono::steady_clock::time_point, PeerPerformance>> recent_performance;
    std::chrono::steady_clock::time_point last_update;
    bool is_active = true;
    bool needs_load_reduction = false;
    bool needs_load_increase = false;
};

struct PeerAssignment {
    std::string peer_id;
    uint32_t assigned_batch_size = 0;
    float priority = 0.0f;  // Higher = better performance
};

struct WorkloadAssignment {
    std::vector<PeerAssignment> peer_assignments;
    uint32_t total_batch_size = 0;
    std::chrono::steady_clock::time_point timestamp;
};

enum class MigrationStatus {
    PENDING,
    IN_PROGRESS,
    COMPLETED,
    FAILED
};

struct WorkloadMigration {
    std::string migration_id;
    std::string source_peer_id;
    std::string target_peer_id;
    uint32_t workload_size = 0;
    MigrationStatus status = MigrationStatus::PENDING;
    std::chrono::steady_clock::time_point created_time;
    std::chrono::steady_clock::time_point completed_time;
    std::string error_message;
};

struct LoadBalancerConfig {
    uint32_t rebalancing_interval_ms = 30000;        // Rebalance every 30 seconds
    uint32_t performance_monitoring_interval_ms = 5000;  // Monitor performance every 5 seconds
    uint32_t migration_check_interval_ms = 10000;    // Check for migrations every 10 seconds
    uint32_t migration_timeout_ms = 60000;           // Migration timeout after 60 seconds
    uint32_t peer_timeout_ms = 30000;                // Consider peer dead after 30 seconds
    uint32_t performance_history_size = 20;          // Keep last 20 performance samples
    float performance_threshold = 0.7f;              // Target performance level (0-1)
    float load_imbalance_threshold = 0.3f;           // Coefficient of variation threshold
    float performance_degradation_threshold = 0.2f;  // Performance drop threshold
    float performance_improvement_threshold = 0.15f; // Performance improvement threshold
    uint32_t min_batch_size = 1;                     // Minimum batch size per peer
    uint32_t max_batch_size = 1000;                  // Maximum batch size per peer
    bool enable_workload_migration = true;           // Enable automatic workload migration
    bool enable_adaptive_batching = true;            // Enable adaptive batch sizing
};

struct LoadBalancingStats {
    uint64_t total_rebalancing_operations = 0;
    uint64_t total_migrations = 0;
    uint64_t successful_migrations = 0;
    uint64_t failed_migrations = 0;
    uint32_t active_peers = 0;
    uint32_t total_workload = 0;
    std::chrono::steady_clock::time_point last_rebalancing_time;
    std::chrono::steady_clock::time_point last_migration_time;
};

class DynamicLoadBalancer {
public:
    explicit DynamicLoadBalancer(
        std::shared_ptr<profiling::PerformanceProfiler> profiler,
        const LoadBalancerConfig& config = LoadBalancerConfig{});
    
    ~DynamicLoadBalancer();
    
    // Control interface
    bool start_load_balancing();
    void stop_load_balancing();
    bool is_load_balancing() const { return balancing_active_.load(); }
    
    // Peer management
    void register_peer(const std::string& peer_id, const PeerCapabilities& capabilities);
    void unregister_peer(const std::string& peer_id);
    void update_peer_performance(const std::string& peer_id, const PeerPerformance& performance);
    
    // Workload assignment
    WorkloadAssignment get_optimal_workload_assignment(uint32_t total_batch_size);
    std::vector<std::string> get_optimal_peer_selection(uint32_t required_peers);
    
    // Statistics and monitoring
    LoadBalancingStats get_load_balancing_stats() const;
    std::vector<PeerInfo> get_active_peers() const {
        std::lock_guard<std::mutex> lock(peers_mutex_);
        std::vector<PeerInfo> active;
        for (const auto& [peer_id, info] : peer_info_) {
            if (info.is_active) active.push_back(info);
        }
        return active;
    }
    
    // Configuration
    void update_config(const LoadBalancerConfig& config) { config_ = config; }
    LoadBalancerConfig get_config() const { return config_; }

private:
    // Core monitoring and balancing threads
    void performance_monitoring_thread();
    void load_balancing_thread();
    void workload_migration_thread();
    
    // Performance monitoring
    void monitor_peer_health();
    void detect_performance_issues();
    
    // Load balancing
    bool should_rebalance();
    void perform_load_rebalancing();
    
    // Workload migration
    void check_for_migration_opportunities();
    void process_pending_migrations();
    void create_migration_plan(const std::string& source_peer, const std::string& target_peer);
    void execute_migration(WorkloadMigration& migration);
    
    // Performance calculation
    float calculate_peer_performance_score(const PeerInfo& peer) const;
    float calculate_capability_score(const PeerCapabilities& capabilities) const;
    float calculate_performance_score(const PeerPerformance& performance) const;
    float calculate_performance_trend(const PeerInfo& peer) const;
    
    // Utility functions
    bool is_peer_healthy(const PeerInfo& peer) const;
    uint32_t adjust_for_peer_capabilities(const PeerInfo& peer, uint32_t suggested_batch) const;
    void update_peer_load_estimate(PeerInfo& peer);
    std::string find_best_migration_target(const std::string& source_peer, 
                                         const std::vector<std::string>& candidate_peers) const;
    std::string generate_migration_id();
    void update_balancing_statistics();
    
    // Dependencies
    std::shared_ptr<profiling::PerformanceProfiler> profiler_;
    LoadBalancerConfig config_;
    
    // Threading and synchronization
    std::atomic<bool> balancing_active_;
    std::thread performance_monitor_thread_;
    std::thread load_balancer_thread_;
    std::thread workload_migration_thread_;
    
    // Peer management
    mutable std::mutex peers_mutex_;
    std::unordered_map<std::string, PeerInfo> peer_info_;
    
    // Load balancing
    mutable std::mutex balancing_mutex_;
    std::condition_variable balancing_cv_;
    
    // Workload migration
    mutable std::mutex migration_mutex_;
    std::condition_variable migration_cv_;
    std::vector<WorkloadMigration> pending_migrations_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    LoadBalancingStats balancing_stats_;
};

// Factory function for easy integration
std::unique_ptr<DynamicLoadBalancer> create_dynamic_load_balancer(
    std::shared_ptr<profiling::PerformanceProfiler> profiler,
    const LoadBalancerConfig& config = LoadBalancerConfig{});

} // namespace load_balancing
