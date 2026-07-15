#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <mutex>
#include <memory>
#include <chrono>
#include <atomic>
#include <thread>
#include <queue>

namespace debugging {

struct LogEntry {
    std::string node_id;
    std::string component;           // "training", "consensus", "network", "memory", etc.
    std::string log_level;          // "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    std::string message;
    std::chrono::steady_clock::time_point timestamp;
    std::unordered_map<std::string, std::string> metadata; // Additional context
    std::string trace_id;           // For distributed tracing
    std::string span_id;            // For distributed tracing
    bool contains_sensitive_data;   // Flag for privacy filtering
};

struct TraceSpan {
    std::string trace_id;
    std::string span_id;
    std::string parent_span_id;
    std::string operation_name;
    std::string node_id;
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;
    std::unordered_map<std::string, std::string> tags;
    std::vector<LogEntry> logs;
    bool is_error;
    std::string error_message;
};

struct HealthCheckResult {
    std::string node_id;
    std::string check_name;
    bool is_healthy;
    float health_score;             // 0.0 to 1.0
    std::string status_message;
    std::unordered_map<std::string, float> metrics;
    std::chrono::steady_clock::time_point check_timestamp;
    uint32_t consecutive_failures;
};

struct SystemMetrics {
    std::string node_id;
    
    // Resource metrics
    float cpu_usage_percent;
    float memory_usage_percent;
    float disk_usage_percent;
    float network_rx_bytes_per_sec;
    float network_tx_bytes_per_sec;
    
    // Training metrics
    float gradient_computation_time_ms;
    float communication_time_ms;
    uint64_t gradients_processed;
    uint64_t consensus_rounds_participated;
    
    // Quality metrics
    float gradient_norm;
    float loss_value;
    uint32_t byzantine_detections;
    float reputation_score;
    
    std::chrono::steady_clock::time_point collection_timestamp;
};

struct PrivacyConfig {
    bool enable_data_sanitization = true;
    bool hash_sensitive_fields = true;
    std::vector<std::string> sensitive_field_patterns; // Regex patterns for sensitive data
    bool enable_differential_privacy = false;
    float differential_privacy_epsilon = 1.0f;
    uint32_t log_retention_days = 30;
};

struct MonitoringConfig {
    // Logging configuration
    std::string log_level = "INFO";          // Minimum log level to process
    uint32_t max_log_buffer_size = 10000;    // Maximum logs in memory buffer
    uint32_t log_batch_size = 100;           // Batch size for log transmission
    uint32_t log_flush_interval_ms = 5000;   // How often to flush logs
    
    // Tracing configuration
    float trace_sampling_rate = 0.1f;        // Fraction of traces to sample
    uint32_t max_span_duration_ms = 300000;  // Maximum span duration (5 minutes)
    bool enable_distributed_tracing = true;
    
    // Health checking
    uint32_t health_check_interval_ms = 30000; // Health check frequency
    uint32_t health_check_timeout_ms = 5000;   // Timeout for individual checks
    uint32_t max_consecutive_failures = 3;     // Failures before marking unhealthy
    
    // Metrics collection
    uint32_t metrics_collection_interval_ms = 10000; // Metrics collection frequency
    bool enable_system_metrics = true;
    bool enable_training_metrics = true;
    bool enable_network_metrics = true;
    
    PrivacyConfig privacy_config;
};

class DistributedDebuggingMonitor {
public:
    explicit DistributedDebuggingMonitor(const MonitoringConfig& config = {});
    ~DistributedDebuggingMonitor();
    
    // Logging with privacy preservation
    void log_message(const LogEntry& entry);
    void log_debug(const std::string& node_id, const std::string& component, 
                   const std::string& message, const std::unordered_map<std::string, std::string>& metadata = {});
    void log_info(const std::string& node_id, const std::string& component, 
                  const std::string& message, const std::unordered_map<std::string, std::string>& metadata = {});
    void log_warning(const std::string& node_id, const std::string& component, 
                     const std::string& message, const std::unordered_map<std::string, std::string>& metadata = {});
    void log_error(const std::string& node_id, const std::string& component, 
                   const std::string& message, const std::unordered_map<std::string, std::string>& metadata = {});
    
    // Distributed tracing
    std::string start_trace(const std::string& operation_name, const std::string& node_id,
                           const std::string& parent_trace_id = "");
    void add_trace_tag(const std::string& trace_id, const std::string& key, const std::string& value);
    void log_trace_event(const std::string& trace_id, const std::string& event, 
                        const std::unordered_map<std::string, std::string>& metadata = {});
    void finish_trace(const std::string& trace_id, bool is_error = false, 
                     const std::string& error_message = "");
    
    // Health monitoring
    void register_health_check(const std::string& check_name, 
                              std::function<HealthCheckResult()> check_function);
    void run_health_check(const std::string& check_name);
    void run_all_health_checks();
    std::vector<HealthCheckResult> get_health_status() const;
    bool is_system_healthy() const;
    
    // Metrics collection
    void collect_system_metrics(const std::string& node_id);
    void submit_custom_metric(const std::string& node_id, const std::string& metric_name, 
                             float value, const std::unordered_map<std::string, std::string>& tags = {});
    std::vector<SystemMetrics> get_recent_metrics(const std::string& node_id, 
                                                 uint32_t last_n_minutes = 10) const;
    
    // Aggregated monitoring
    struct AggregatedMetrics {
        uint32_t total_nodes;
        uint32_t healthy_nodes;
        uint32_t unhealthy_nodes;
        float average_cpu_usage;
        float average_memory_usage;
        float total_gradients_processed;
        float average_gradient_computation_time;
        uint32_t total_byzantine_detections;
        std::chrono::steady_clock::time_point aggregation_timestamp;
    };
    
    AggregatedMetrics get_aggregated_metrics() const;
    
    // Privacy-preserving log aggregation
    std::vector<LogEntry> get_sanitized_logs(const std::string& filter_component = "",
                                            const std::string& filter_level = "",
                                            uint32_t last_n_minutes = 60) const;
    bool export_sanitized_logs(const std::string& output_path, 
                              const std::string& format = "json") const; // "json" or "csv"
    
    // Anomaly detection
    struct AnomalyAlert {
        std::string node_id;
        std::string anomaly_type;       // "performance", "behavior", "error_rate", "resource"
        std::string description;
        float severity_score;           // 0.0 to 1.0
        std::unordered_map<std::string, float> evidence;
        std::chrono::steady_clock::time_point detection_timestamp;
    };
    
    std::vector<AnomalyAlert> detect_system_anomalies() const;
    void enable_real_time_anomaly_detection(bool enable = true);
    
    // Node isolation and recovery
    void isolate_problematic_node(const std::string& node_id, const std::string& reason);
    void recover_isolated_node(const std::string& node_id);
    std::vector<std::string> get_isolated_nodes() const;
    
    // Configuration and control
    void update_config(const MonitoringConfig& new_config);
    MonitoringConfig get_config() const;
    
    // Centralized log aggregation server
    bool start_log_aggregation_server(uint16_t port = 9090);
    void stop_log_aggregation_server();
    bool is_aggregation_server_running() const;
    
    // Distributed tracing server integration (e.g., Jaeger-compatible)
    bool start_tracing_server(uint16_t port = 14268);
    void stop_tracing_server();
    
    // Export and reporting
    bool export_system_report(const std::string& output_path) const;
    bool export_trace_data(const std::string& output_path, const std::string& trace_id = "") const;

private:
    MonitoringConfig config_;
    mutable std::mutex config_mutex_;
    
    // Logging infrastructure
    mutable std::mutex log_buffer_mutex_;
    std::queue<LogEntry> log_buffer_;
    std::thread log_processing_thread_;
    std::atomic<bool> log_processing_active_{true};
    
    // Distributed tracing
    mutable std::mutex trace_mutex_;
    std::unordered_map<std::string, TraceSpan> active_traces_;
    std::vector<TraceSpan> completed_traces_;
    uint64_t next_trace_id_;
    uint64_t next_span_id_;
    
    // Health monitoring
    mutable std::mutex health_mutex_;
    std::unordered_map<std::string, std::function<HealthCheckResult()>> health_checks_;
    std::unordered_map<std::string, HealthCheckResult> last_health_results_;
    std::thread health_monitoring_thread_;
    std::atomic<bool> health_monitoring_active_{true};
    
    // Metrics collection
    mutable std::mutex metrics_mutex_;
    std::unordered_map<std::string, std::vector<SystemMetrics>> node_metrics_;
    std::thread metrics_collection_thread_;
    std::atomic<bool> metrics_collection_active_{true};
    
    // Node isolation
    mutable std::mutex isolation_mutex_;
    std::unordered_map<std::string, std::string> isolated_nodes_; // node_id -> reason
    
    // Anomaly detection
    mutable std::mutex anomaly_mutex_;
    std::vector<AnomalyAlert> recent_anomalies_;
    std::atomic<bool> real_time_anomaly_detection_{false};
    
    // Server components
    std::atomic<bool> aggregation_server_running_{false};
    std::atomic<bool> tracing_server_running_{false};
    std::thread aggregation_server_thread_;
    std::thread tracing_server_thread_;
    
    // Privacy and sanitization
    LogEntry sanitize_log_entry(const LogEntry& entry) const;
    bool contains_sensitive_data(const std::string& text) const;
    std::string hash_sensitive_field(const std::string& data) const;
    float apply_differential_privacy_noise(float original_value) const;
    
    // Background processing threads
    void process_log_queue();
    void run_health_monitoring_loop();
    void run_metrics_collection_loop();
    void run_log_aggregation_server(uint16_t port);
    void run_tracing_server(uint16_t port);
    
    // Helper methods
    std::string generate_trace_id();
    std::string generate_span_id();
    bool should_sample_trace() const;
    void cleanup_old_traces();
    void cleanup_old_metrics();
    void cleanup_old_logs();
    
    // Anomaly detection algorithms
    std::vector<AnomalyAlert> detect_performance_anomalies() const;
    std::vector<AnomalyAlert> detect_error_rate_anomalies() const;
    std::vector<AnomalyAlert> detect_resource_anomalies() const;
    std::vector<AnomalyAlert> detect_behavioral_anomalies() const;
    
    // System resource monitoring
    SystemMetrics collect_system_resources(const std::string& node_id) const;
    float get_cpu_usage() const;
    float get_memory_usage() const;
    float get_disk_usage() const;
    std::pair<float, float> get_network_usage() const; // rx, tx bytes/sec
};

// Specialized health check implementations
namespace health_checks {

HealthCheckResult check_memory_usage(float warning_threshold = 0.8f, float critical_threshold = 0.9f);
HealthCheckResult check_cpu_usage(float warning_threshold = 0.8f, float critical_threshold = 0.9f);
HealthCheckResult check_disk_space(float warning_threshold = 0.8f, float critical_threshold = 0.9f);
HealthCheckResult check_network_connectivity(const std::vector<std::string>& peer_nodes);
HealthCheckResult check_consensus_participation(uint32_t min_rounds_per_minute = 5);
HealthCheckResult check_gradient_computation_health(float max_acceptable_time_ms = 10000);
HealthCheckResult check_byzantine_detection_rate(float max_acceptable_rate = 0.1f);

} // namespace health_checks

// Distributed tracing utilities
namespace tracing {

class AutoSpan {
public:
    AutoSpan(DistributedDebuggingMonitor* monitor, const std::string& operation_name, 
             const std::string& node_id, const std::string& parent_trace_id = "");
    ~AutoSpan();
    
    void add_tag(const std::string& key, const std::string& value);
    void log_event(const std::string& event, const std::unordered_map<std::string, std::string>& metadata = {});
    void set_error(const std::string& error_message);
    
    std::string get_trace_id() const { return trace_id_; }

private:
    DistributedDebuggingMonitor* monitor_;
    std::string trace_id_;
    bool finished_;
    bool has_error_;
    std::string error_message_;
};

#define TRACE_SPAN(monitor, operation, node_id) \
    tracing::AutoSpan _trace_span(monitor, operation, node_id)

#define TRACE_SPAN_WITH_PARENT(monitor, operation, node_id, parent) \
    tracing::AutoSpan _trace_span(monitor, operation, node_id, parent)

} // namespace tracing
} // namespace debugging
