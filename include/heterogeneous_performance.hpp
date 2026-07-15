#pragma once

#include <vector>
#include <unordered_map>
#include <mutex>
#include <chrono>
#include <memory>
#include <thread>
#include <atomic>

namespace heterogeneous {

struct NodeCapabilities {
    std::string node_id;
    float compute_score;        // Relative compute capability (1.0 = baseline)
    float memory_gb;           // Available memory in GB
    float network_bandwidth;   // Network bandwidth in Mbps
    float storage_speed;       // Storage I/O speed in MB/s
    bool has_gpu;             // GPU availability
    int gpu_memory_gb;        // GPU memory if available
    std::chrono::steady_clock::time_point last_benchmark;
    uint32_t benchmark_version; // Incremented when benchmark changes
};

struct WorkAllocation {
    std::string node_id;
    size_t gradient_shard_size;    // Number of parameters in this shard
    size_t gradient_shard_offset;  // Starting offset in gradient vector
    float expected_completion_time; // Estimated time to complete this shard
    uint32_t priority_level;       // Higher priority = more important work
};

struct SynchronizationConfig {
    float wait_percentile = 0.8f;   // Wait for fastest 80% of nodes
    uint32_t min_nodes_required = 3; // Minimum nodes needed for training
    uint32_t max_wait_time_ms = 30000; // Maximum wait time before proceeding
    bool enable_adaptive_timeout = true; // Adjust timeouts based on historical performance
};

struct BenchmarkResult {
    float flops_per_second;      // Floating point operations per second
    float memory_bandwidth;      // GB/s memory bandwidth
    float gradient_compute_time; // Time to compute gradients (ms)
    float communication_latency; // Network round-trip time (ms)
    float aggregation_speed;     // Gradients aggregated per second
    bool benchmark_valid;        // Whether benchmark completed successfully
};

class HeterogeneousPerformanceManager {
public:
    explicit HeterogeneousPerformanceManager(const SynchronizationConfig& config = {});
    ~HeterogeneousPerformanceManager();

    // Node capability management
    bool register_node(const std::string& node_id, const NodeCapabilities& capabilities);
    bool remove_node(const std::string& node_id);
    bool update_node_capabilities(const std::string& node_id, const NodeCapabilities& capabilities);
    std::vector<NodeCapabilities> get_registered_nodes() const;
    
    // Benchmarking
    BenchmarkResult run_performance_benchmark(const std::string& node_id);
    bool schedule_periodic_benchmarks(bool enable = true);
    void update_benchmark_from_training_data(const std::string& node_id, 
                                           float actual_time_ms, size_t work_size);
    
    // Work allocation
    std::vector<WorkAllocation> allocate_gradient_work(size_t total_gradient_size,
                                                      const std::vector<std::string>& available_nodes);
    std::vector<WorkAllocation> rebalance_work_allocation(const std::vector<WorkAllocation>& current_allocation,
                                                         const std::vector<std::string>& slow_nodes);
    
    // Elastic synchronization
    struct SyncResult {
        std::vector<std::string> completed_nodes;
        std::vector<std::string> pending_nodes;
        float completion_percentage;
        bool should_proceed;
        uint32_t wait_time_ms;
    };
    
    SyncResult wait_for_gradient_completion(const std::vector<std::string>& participating_nodes,
                                          const std::unordered_map<std::string, bool>& completion_status);
    bool should_proceed_with_partial_sync(float completion_percentage, 
                                         uint32_t elapsed_time_ms) const;
    
    // Performance prediction
    float predict_completion_time(const std::string& node_id, size_t work_size) const;
    std::vector<std::string> select_fastest_nodes(const std::vector<std::string>& candidates, 
                                                  size_t max_nodes) const;
    
    // Adaptive configuration
    void update_sync_config(const SynchronizationConfig& new_config);
    SynchronizationConfig get_sync_config() const;
    
    // Monitoring and diagnostics
    struct PerformanceStats {
        uint32_t total_benchmarks_run;
        uint32_t active_nodes;
        float average_node_score;
        float sync_completion_rate;
        float average_wait_time_ms;
        std::chrono::steady_clock::time_point last_rebalance;
    };
    
    PerformanceStats get_performance_stats() const;
    void log_performance_metrics(const std::string& output_path) const;

private:
    mutable std::mutex capabilities_mutex_;
    std::unordered_map<std::string, NodeCapabilities> node_capabilities_;
    
    mutable std::mutex performance_history_mutex_;
    std::unordered_map<std::string, std::vector<BenchmarkResult>> performance_history_;
    
    SynchronizationConfig sync_config_;
    mutable std::mutex config_mutex_;
    
    // Benchmarking
    std::atomic<bool> benchmark_running_{false};
    std::thread benchmark_thread_;
    
    // Performance tracking
    PerformanceStats stats_;
    mutable std::mutex stats_mutex_;
    
    // Helper methods
    float calculate_node_score(const BenchmarkResult& result) const;
    bool is_benchmark_outdated(const NodeCapabilities& capabilities) const;
    void run_benchmark_thread();
    size_t calculate_optimal_shard_size(const NodeCapabilities& capabilities, 
                                       size_t total_work) const;
    
    // Advanced allocation algorithms
    std::vector<WorkAllocation> optimize_allocation_greedy(size_t total_gradient_size,
                                                          const std::vector<NodeCapabilities>& nodes);
    std::vector<WorkAllocation> optimize_allocation_load_balance(size_t total_gradient_size,
                                                               const std::vector<NodeCapabilities>& nodes);
};

// Factory for creating performance managers with different strategies
class PerformanceManagerFactory {
public:
    enum class AllocationStrategy {
        GREEDY_FASTEST_FIRST,
        LOAD_BALANCED,
        MEMORY_AWARE,
        NETWORK_OPTIMIZED
    };
    
    static std::unique_ptr<HeterogeneousPerformanceManager> 
    create_manager(AllocationStrategy strategy, const SynchronizationConfig& config = {});
};

} // namespace heterogeneous
