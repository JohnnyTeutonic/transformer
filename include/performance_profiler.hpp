#pragma once

#include <vector>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <memory>
#include <unordered_map>

namespace profiling {

struct CPUMetrics {
    float usage_percent = 0.0f;
    uint32_t core_count = 0;
    float frequency_mhz = 0.0f;
    float temperature_celsius = 0.0f;
};

struct GPUMetrics {
    float usage_percent = 0.0f;
    float memory_usage_percent = 0.0f;
    float memory_used_gb = 0.0f;
    float memory_total_gb = 0.0f;
    float temperature_celsius = 0.0f;
    float power_usage_watts = 0.0f;
};

struct MemoryMetrics {
    float total_gb = 0.0f;
    float used_gb = 0.0f;
    float available_gb = 0.0f;
    float usage_percent = 0.0f;
};

struct NetworkMetrics {
    float bytes_sent_per_sec = 0.0f;
    float bytes_received_per_sec = 0.0f;
    float packets_sent_per_sec = 0.0f;
    float packets_received_per_sec = 0.0f;
    float latency_ms = 0.0f;
};

struct PerformanceMetrics {
    CPUMetrics cpu;
    GPUMetrics gpu;
    MemoryMetrics memory;
    NetworkMetrics network;
    std::chrono::steady_clock::time_point timestamp;
};

struct TrainingStepMetrics {
    uint64_t step_number = 0;
    uint32_t batch_size = 0;
    float loss = 0.0f;
    uint64_t step_time_ms = 0;
    float samples_per_second = 0.0f;
    float memory_usage_gb = 0.0f;
    std::chrono::steady_clock::time_point timestamp;
};

struct NetworkOperationMetrics {
    std::string operation_type;
    uint64_t bytes_transferred = 0;
    uint64_t operation_time_ms = 0;
    float bandwidth_mbps = 0.0f;
    float latency_ms = 0.0f;
    bool success = true;
    std::chrono::steady_clock::time_point timestamp;
};

struct SystemCapabilities {
    uint32_t cpu_cores = 0;
    float cpu_frequency_mhz = 0.0f;
    float total_memory_gb = 0.0f;
    float available_memory_gb = 0.0f;
    uint32_t gpu_count = 0;
    float gpu_memory_gb = 0.0f;
    std::string gpu_compute_capability;
    float network_bandwidth_mbps = 0.0f;
    float network_latency_ms = 0.0f;
};

struct PerformanceProfile {
    std::chrono::steady_clock::time_point timestamp;
    SystemCapabilities system_capabilities;
    PerformanceMetrics current_metrics;
    float compute_score = 0.0f;      // 0.0 = fully loaded, 1.0 = idle
    float memory_score = 0.0f;       // 0.0 = no memory, 1.0 = plenty available
    float network_score = 0.0f;      // 0.0 = poor network, 1.0 = excellent network
    float overall_score = 0.0f;      // Combined score
    float recent_training_throughput = 0.0f;  // Recent samples/sec
};

struct TrainingStatistics {
    uint64_t total_steps = 0;
    uint64_t total_samples = 0;
    uint64_t total_training_time_ms = 0;
    float average_step_time_ms = 0.0f;
    float average_throughput_samples_per_sec = 0.0f;
};

struct NetworkStatistics {
    uint64_t total_operations = 0;
    uint64_t total_bytes_transferred = 0;
    uint64_t total_operation_time_ms = 0;
    float average_operation_time_ms = 0.0f;
    float average_bandwidth_mbps = 0.0f;
};

struct ProfilerConfig {
    uint32_t sampling_interval_ms = 1000;    // Sample system metrics every second
    uint32_t history_size = 300;             // Keep 5 minutes of history at 1s intervals
    bool enable_gpu_monitoring = true;
    bool enable_network_monitoring = true;
    bool enable_detailed_profiling = false;  // More detailed but higher overhead
};

class PerformanceProfiler {
public:
    explicit PerformanceProfiler(const ProfilerConfig& config = ProfilerConfig{});
    ~PerformanceProfiler();
    
    // Control interface
    bool start_profiling();
    void stop_profiling();
    bool is_profiling() const { return profiling_active_.load(); }
    
    // Data recording interface
    void record_training_step(const TrainingStepMetrics& metrics);
    void record_network_operation(const NetworkOperationMetrics& metrics);
    
    // Query interface
    SystemCapabilities get_system_capabilities() const;
    PerformanceMetrics get_current_metrics() const;
    PerformanceProfile get_performance_profile() const;
    
    // Statistics
    TrainingStatistics get_training_statistics() const {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        return training_stats_;
    }
    
    NetworkStatistics get_network_statistics() const {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        return network_stats_;
    }
    
    // Configuration
    void update_config(const ProfilerConfig& config) { config_ = config; }
    ProfilerConfig get_config() const { return config_; }
    
    // Utility functions
    void reset_counters();
    void export_metrics(const std::string& filename) const;

private:
    // Monitoring threads
    void cpu_monitoring_thread();
    void gpu_monitoring_thread();
    void memory_monitoring_thread();
    void network_monitoring_thread();
    
    // Sampling functions
    CPUMetrics sample_cpu_metrics();
    GPUMetrics sample_gpu_metrics();
    MemoryMetrics sample_memory_metrics();
    NetworkMetrics sample_network_metrics();
    
    // System information queries
    uint32_t get_cpu_core_count() const;
    float get_cpu_frequency() const;
    float get_cpu_temperature() const;
    float get_total_memory_gb() const;
    float get_available_memory_gb() const;
    uint32_t get_gpu_count() const;
    float get_gpu_memory_gb(uint32_t gpu_index) const;
    std::string get_gpu_compute_capability(uint32_t gpu_index) const;
    
    // Performance scoring
    float calculate_compute_score() const;
    float calculate_memory_score() const;
    float calculate_network_score() const;
    
    // Statistics updates
    void update_training_statistics(const TrainingStepMetrics& metrics);
    void update_network_statistics(const NetworkOperationMetrics& metrics);
    
    // Configuration and state
    ProfilerConfig config_;
    std::atomic<bool> profiling_active_;
    
    // Threading
    std::thread cpu_monitor_thread_;
    std::thread gpu_monitor_thread_;
    std::thread memory_monitor_thread_;
    std::thread network_monitor_thread_;
    
    // Synchronization
    mutable std::mutex metrics_mutex_;
    mutable std::mutex profiling_mutex_;
    std::condition_variable profiling_cv_;
    
    // Current metrics
    PerformanceMetrics current_metrics_;
    
    // Historical data
    std::vector<std::pair<std::chrono::steady_clock::time_point, CPUMetrics>> cpu_history_;
    std::vector<std::pair<std::chrono::steady_clock::time_point, GPUMetrics>> gpu_history_;
    std::vector<std::pair<std::chrono::steady_clock::time_point, MemoryMetrics>> memory_history_;
    std::vector<std::pair<std::chrono::steady_clock::time_point, NetworkMetrics>> network_metrics_history_;
    
    // Training and network operation history
    std::vector<TrainingStepMetrics> training_history_;
    std::vector<NetworkOperationMetrics> network_history_;
    
    // Running statistics
    TrainingStatistics training_stats_;
    NetworkStatistics network_stats_;
    
    // Platform-specific state
#ifdef CUDA_AVAILABLE
    bool nvml_initialized_ = false;
#endif
};

// Factory function for easy integration
std::unique_ptr<PerformanceProfiler> create_performance_profiler(
    const ProfilerConfig& config = ProfilerConfig{});

} // namespace profiling
