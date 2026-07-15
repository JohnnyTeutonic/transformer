#include "../include/debugging_monitoring.hpp"
#include "../include/logger.hpp"
#include <algorithm>
#include <random>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <regex>
#include <cmath>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#include <pdh.h>
#else
#include <sys/sysinfo.h>
#include <sys/statvfs.h>
#include <unistd.h>
#include <fstream>
#endif

namespace debugging {

DistributedDebuggingMonitor::DistributedDebuggingMonitor(const MonitoringConfig& config)
    : config_(config), next_trace_id_(1), next_span_id_(1) {
    
    logger::log_info("Initializing Distributed Debugging Monitor");
    logger::log_info("- Log level: " + config_.log_level);
    logger::log_info("- Trace sampling rate: " + std::to_string(config_.trace_sampling_rate * 100) + "%");
    logger::log_info("- Health check interval: " + std::to_string(config_.health_check_interval_ms) + "ms");
    logger::log_info("- Privacy sanitization: " + (config_.privacy_config.enable_data_sanitization ? "enabled" : "disabled"));
    
    // Start background threads
    log_processing_thread_ = std::thread(&DistributedDebuggingMonitor::process_log_queue, this);
    health_monitoring_thread_ = std::thread(&DistributedDebuggingMonitor::run_health_monitoring_loop, this);
    metrics_collection_thread_ = std::thread(&DistributedDebuggingMonitor::run_metrics_collection_loop, this);
    
    // Initialize sensitive data patterns
    if (config_.privacy_config.sensitive_field_patterns.empty()) {
        config_.privacy_config.sensitive_field_patterns = {
            R"(\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b)",  // Credit card numbers
            R"(\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b)", // Email addresses
            R"(\b\d{3}-\d{2}-\d{4}\b)",  // SSN format
            R"(\bpassword\s*[:=]\s*\S+\b)",  // Password fields
            R"(\btoken\s*[:=]\s*\S+\b)",    // Token fields
            R"(\bkey\s*[:=]\s*\S+\b)",      // Key fields
        };
    }
}

DistributedDebuggingMonitor::~DistributedDebuggingMonitor() {
    // Stop background threads
    log_processing_active_ = false;
    health_monitoring_active_ = false;
    metrics_collection_active_ = false;
    
    if (log_processing_thread_.joinable()) {
        log_processing_thread_.join();
    }
    if (health_monitoring_thread_.joinable()) {
        health_monitoring_thread_.join();
    }
    if (metrics_collection_thread_.joinable()) {
        metrics_collection_thread_.join();
    }
    
    // Stop servers
    stop_log_aggregation_server();
    stop_tracing_server();
}

void DistributedDebuggingMonitor::log_message(const LogEntry& entry) {
    // Check if log level is sufficient
    std::vector<std::string> levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"};
    auto config_level_it = std::find(levels.begin(), levels.end(), config_.log_level);
    auto entry_level_it = std::find(levels.begin(), levels.end(), entry.log_level);
    
    if (config_level_it == levels.end() || entry_level_it == levels.end() || 
        std::distance(levels.begin(), entry_level_it) < std::distance(levels.begin(), config_level_it)) {
        return; // Log level too low
    }
    
    // Sanitize entry for privacy
    LogEntry sanitized_entry = sanitize_log_entry(entry);
    
    // Add to buffer
    {
        std::lock_guard<std::mutex> lock(log_buffer_mutex_);
        
        log_buffer_.push(sanitized_entry);
        
        // Limit buffer size
        while (log_buffer_.size() > config_.max_log_buffer_size) {
            log_buffer_.pop();
        }
    }
}

void DistributedDebuggingMonitor::log_info(const std::string& node_id, const std::string& component, 
                                          const std::string& message, 
                                          const std::unordered_map<std::string, std::string>& metadata) {
    LogEntry entry;
    entry.node_id = node_id;
    entry.component = component;
    entry.log_level = "INFO";
    entry.message = message;
    entry.timestamp = std::chrono::steady_clock::now();
    entry.metadata = metadata;
    entry.contains_sensitive_data = contains_sensitive_data(message);
    
    log_message(entry);
}

void DistributedDebuggingMonitor::log_warning(const std::string& node_id, const std::string& component, 
                                             const std::string& message, 
                                             const std::unordered_map<std::string, std::string>& metadata) {
    LogEntry entry;
    entry.node_id = node_id;
    entry.component = component;
    entry.log_level = "WARNING";
    entry.message = message;
    entry.timestamp = std::chrono::steady_clock::now();
    entry.metadata = metadata;
    entry.contains_sensitive_data = contains_sensitive_data(message);
    
    log_message(entry);
}

void DistributedDebuggingMonitor::log_error(const std::string& node_id, const std::string& component, 
                                           const std::string& message, 
                                           const std::unordered_map<std::string, std::string>& metadata) {
    LogEntry entry;
    entry.node_id = node_id;
    entry.component = component;
    entry.log_level = "ERROR";
    entry.message = message;
    entry.timestamp = std::chrono::steady_clock::now();
    entry.metadata = metadata;
    entry.contains_sensitive_data = contains_sensitive_data(message);
    
    log_message(entry);
}

std::string DistributedDebuggingMonitor::start_trace(const std::string& operation_name, 
                                                    const std::string& node_id,
                                                    const std::string& parent_trace_id) {
    if (!config_.enable_distributed_tracing || !should_sample_trace()) {
        return ""; // Not sampling this trace
    }
    
    std::lock_guard<std::mutex> lock(trace_mutex_);
    
    TraceSpan span;
    span.trace_id = generate_trace_id();
    span.span_id = generate_span_id();
    span.parent_span_id = parent_trace_id;
    span.operation_name = operation_name;
    span.node_id = node_id;
    span.start_time = std::chrono::steady_clock::now();
    span.is_error = false;
    
    active_traces_[span.trace_id] = span;
    
    logger::log_debug("Started trace: " + span.trace_id + " for operation: " + operation_name);
    
    return span.trace_id;
}

void DistributedDebuggingMonitor::add_trace_tag(const std::string& trace_id, 
                                               const std::string& key, const std::string& value) {
    if (trace_id.empty()) return;
    
    std::lock_guard<std::mutex> lock(trace_mutex_);
    
    auto it = active_traces_.find(trace_id);
    if (it != active_traces_.end()) {
        it->second.tags[key] = value;
    }
}

void DistributedDebuggingMonitor::log_trace_event(const std::string& trace_id, const std::string& event,
                                                 const std::unordered_map<std::string, std::string>& metadata) {
    if (trace_id.empty()) return;
    
    std::lock_guard<std::mutex> lock(trace_mutex_);
    
    auto it = active_traces_.find(trace_id);
    if (it != active_traces_.end()) {
        LogEntry log_entry;
        log_entry.message = event;
        log_entry.metadata = metadata;
        log_entry.timestamp = std::chrono::steady_clock::now();
        log_entry.trace_id = trace_id;
        log_entry.span_id = it->second.span_id;
        
        it->second.logs.push_back(log_entry);
    }
}

void DistributedDebuggingMonitor::finish_trace(const std::string& trace_id, bool is_error, 
                                              const std::string& error_message) {
    if (trace_id.empty()) return;
    
    std::lock_guard<std::mutex> lock(trace_mutex_);
    
    auto it = active_traces_.find(trace_id);
    if (it != active_traces_.end()) {
        it->second.end_time = std::chrono::steady_clock::now();
        it->second.is_error = is_error;
        it->second.error_message = error_message;
        
        // Move to completed traces
        completed_traces_.push_back(it->second);
        active_traces_.erase(it);
        
        // Limit completed traces history
        if (completed_traces_.size() > 1000) {
            completed_traces_.erase(completed_traces_.begin());
        }
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            it->second.end_time - it->second.start_time);
        
        logger::log_debug("Finished trace: " + trace_id + " duration: " + 
                         std::to_string(duration.count()) + "ms" + 
                         (is_error ? " (ERROR)" : ""));
    }
}

void DistributedDebuggingMonitor::register_health_check(const std::string& check_name,
                                                       std::function<HealthCheckResult()> check_function) {
    std::lock_guard<std::mutex> lock(health_mutex_);
    
    health_checks_[check_name] = check_function;
    
    logger::log_info("Registered health check: " + check_name);
}

void DistributedDebuggingMonitor::run_health_check(const std::string& check_name) {
    std::lock_guard<std::mutex> lock(health_mutex_);
    
    auto it = health_checks_.find(check_name);
    if (it != health_checks_.end()) {
        try {
            HealthCheckResult result = it->second();
            result.check_timestamp = std::chrono::steady_clock::now();
            
            // Update consecutive failure count
            auto last_result_it = last_health_results_.find(check_name);
            if (last_result_it != last_health_results_.end()) {
                if (result.is_healthy) {
                    result.consecutive_failures = 0;
                } else {
                    result.consecutive_failures = last_result_it->second.consecutive_failures + 1;
                }
            } else {
                result.consecutive_failures = result.is_healthy ? 0 : 1;
            }
            
            last_health_results_[check_name] = result;
            
            // Log health check results
            if (!result.is_healthy) {
                log_warning("system", "health_check", "Health check failed: " + check_name + " - " + result.status_message);
                
                // Trigger isolation if too many consecutive failures
                if (result.consecutive_failures >= config_.max_consecutive_failures) {
                    isolate_problematic_node(result.node_id, "Health check failures: " + check_name);
                }
            }
            
        } catch (const std::exception& e) {
            HealthCheckResult error_result;
            error_result.check_name = check_name;
            error_result.is_healthy = false;
            error_result.health_score = 0.0f;
            error_result.status_message = "Health check exception: " + std::string(e.what());
            error_result.check_timestamp = std::chrono::steady_clock::now();
            
            last_health_results_[check_name] = error_result;
            
            log_error("system", "health_check", "Health check exception for " + check_name + ": " + e.what());
        }
    }
}

void DistributedDebuggingMonitor::run_all_health_checks() {
    std::vector<std::string> check_names;
    
    {
        std::lock_guard<std::mutex> lock(health_mutex_);
        for (const auto& [name, func] : health_checks_) {
            check_names.push_back(name);
        }
    }
    
    // Run all health checks
    for (const auto& check_name : check_names) {
        run_health_check(check_name);
    }
}

std::vector<HealthCheckResult> DistributedDebuggingMonitor::get_health_status() const {
    std::lock_guard<std::mutex> lock(health_mutex_);
    
    std::vector<HealthCheckResult> results;
    for (const auto& [name, result] : last_health_results_) {
        results.push_back(result);
    }
    
    return results;
}

bool DistributedDebuggingMonitor::is_system_healthy() const {
    std::lock_guard<std::mutex> lock(health_mutex_);
    
    for (const auto& [name, result] : last_health_results_) {
        if (!result.is_healthy) {
            return false;
        }
    }
    
    return true;
}

void DistributedDebuggingMonitor::collect_system_metrics(const std::string& node_id) {
    SystemMetrics metrics = collect_system_resources(node_id);
    
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    node_metrics_[node_id].push_back(metrics);
    
    // Keep only recent metrics (last 24 hours)
    auto& node_metric_history = node_metrics_[node_id];
    auto cutoff_time = std::chrono::steady_clock::now() - std::chrono::hours(24);
    
    node_metric_history.erase(
        std::remove_if(node_metric_history.begin(), node_metric_history.end(),
                      [cutoff_time](const SystemMetrics& metric) {
                          return metric.collection_timestamp < cutoff_time;
                      }),
        node_metric_history.end());
}

void DistributedDebuggingMonitor::submit_custom_metric(const std::string& node_id, 
                                                      const std::string& metric_name, float value,
                                                      const std::unordered_map<std::string, std::string>& tags) {
    // Create a log entry for the custom metric
    LogEntry metric_log;
    metric_log.node_id = node_id;
    metric_log.component = "metrics";
    metric_log.log_level = "INFO";
    metric_log.message = "Custom metric: " + metric_name + " = " + std::to_string(value);
    metric_log.timestamp = std::chrono::steady_clock::now();
    metric_log.metadata = tags;
    metric_log.metadata["metric_name"] = metric_name;
    metric_log.metadata["metric_value"] = std::to_string(value);
    
    log_message(metric_log);
}

DistributedDebuggingMonitor::AggregatedMetrics DistributedDebuggingMonitor::get_aggregated_metrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    AggregatedMetrics aggregated;
    aggregated.aggregation_timestamp = std::chrono::steady_clock::now();
    
    uint32_t total_nodes = 0;
    float total_cpu = 0.0f, total_memory = 0.0f;
    uint64_t total_gradients = 0;
    float total_gradient_time = 0.0f;
    uint32_t total_byzantine = 0;
    
    // Aggregate across all nodes
    for (const auto& [node_id, metrics_history] : node_metrics_) {
        if (!metrics_history.empty()) {
            const auto& latest = metrics_history.back();
            total_nodes++;
            total_cpu += latest.cpu_usage_percent;
            total_memory += latest.memory_usage_percent;
            total_gradients += latest.gradients_processed;
            total_gradient_time += latest.gradient_computation_time_ms;
            total_byzantine += latest.byzantine_detections;
        }
    }
    
    aggregated.total_nodes = total_nodes;
    
    if (total_nodes > 0) {
        aggregated.average_cpu_usage = total_cpu / total_nodes;
        aggregated.average_memory_usage = total_memory / total_nodes;
        aggregated.average_gradient_computation_time = total_gradient_time / total_nodes;
    }
    
    aggregated.total_gradients_processed = total_gradients;
    aggregated.total_byzantine_detections = total_byzantine;
    
    // Count healthy vs unhealthy nodes
    std::lock_guard<std::mutex> health_lock(health_mutex_);
    std::unordered_set<std::string> unhealthy_nodes;
    
    for (const auto& [check_name, result] : last_health_results_) {
        if (!result.is_healthy) {
            unhealthy_nodes.insert(result.node_id);
        }
    }
    
    aggregated.unhealthy_nodes = static_cast<uint32_t>(unhealthy_nodes.size());
    aggregated.healthy_nodes = total_nodes - aggregated.unhealthy_nodes;
    
    return aggregated;
}

std::vector<DistributedDebuggingMonitor::AnomalyAlert> DistributedDebuggingMonitor::detect_system_anomalies() const {
    std::vector<AnomalyAlert> all_anomalies;
    
    // Detect different types of anomalies
    auto performance_anomalies = detect_performance_anomalies();
    auto error_rate_anomalies = detect_error_rate_anomalies();
    auto resource_anomalies = detect_resource_anomalies();
    auto behavioral_anomalies = detect_behavioral_anomalies();
    
    all_anomalies.insert(all_anomalies.end(), performance_anomalies.begin(), performance_anomalies.end());
    all_anomalies.insert(all_anomalies.end(), error_rate_anomalies.begin(), error_rate_anomalies.end());
    all_anomalies.insert(all_anomalies.end(), resource_anomalies.begin(), resource_anomalies.end());
    all_anomalies.insert(all_anomalies.end(), behavioral_anomalies.begin(), behavioral_anomalies.end());
    
    // Sort by severity score (highest first)
    std::sort(all_anomalies.begin(), all_anomalies.end(),
             [](const AnomalyAlert& a, const AnomalyAlert& b) {
                 return a.severity_score > b.severity_score;
             });
    
    return all_anomalies;
}

void DistributedDebuggingMonitor::isolate_problematic_node(const std::string& node_id, const std::string& reason) {
    std::lock_guard<std::mutex> lock(isolation_mutex_);
    
    isolated_nodes_[node_id] = reason;
    
    log_warning("system", "node_isolation", "Isolated node " + node_id + " due to: " + reason);
    
    // Trigger anomaly alert
    AnomalyAlert alert;
    alert.node_id = node_id;
    alert.anomaly_type = "node_isolation";
    alert.description = "Node isolated: " + reason;
    alert.severity_score = 0.8f;
    alert.detection_timestamp = std::chrono::steady_clock::now();
    
    {
        std::lock_guard<std::mutex> anomaly_lock(anomaly_mutex_);
        recent_anomalies_.push_back(alert);
        
        // Keep only recent anomalies
        if (recent_anomalies_.size() > 100) {
            recent_anomalies_.erase(recent_anomalies_.begin());
        }
    }
}

// Privacy and sanitization methods
LogEntry DistributedDebuggingMonitor::sanitize_log_entry(const LogEntry& entry) const {
    if (!config_.privacy_config.enable_data_sanitization) {
        return entry;
    }
    
    LogEntry sanitized = entry;
    
    // Sanitize message content
    if (sanitized.contains_sensitive_data || contains_sensitive_data(sanitized.message)) {
        if (config_.privacy_config.hash_sensitive_fields) {
            // Hash the entire message if it contains sensitive data
            sanitized.message = hash_sensitive_field(sanitized.message);
        } else {
            // Replace sensitive patterns with placeholders
            std::string sanitized_message = sanitized.message;
            for (const auto& pattern : config_.privacy_config.sensitive_field_patterns) {
                std::regex sensitive_regex(pattern, std::regex_constants::icase);
                sanitized_message = std::regex_replace(sanitized_message, sensitive_regex, "[REDACTED]");
            }
            sanitized.message = sanitized_message;
        }
    }
    
    // Sanitize metadata
    for (auto& [key, value] : sanitized.metadata) {
        if (contains_sensitive_data(value)) {
            if (config_.privacy_config.hash_sensitive_fields) {
                value = hash_sensitive_field(value);
            } else {
                value = "[REDACTED]";
            }
        }
    }
    
    return sanitized;
}

bool DistributedDebuggingMonitor::contains_sensitive_data(const std::string& text) const {
    for (const auto& pattern : config_.privacy_config.sensitive_field_patterns) {
        std::regex sensitive_regex(pattern, std::regex_constants::icase);
        if (std::regex_search(text, sensitive_regex)) {
            return true;
        }
    }
    return false;
}

std::string DistributedDebuggingMonitor::hash_sensitive_field(const std::string& data) const {
    // Simple hash function (in production, use a proper cryptographic hash)
    std::hash<std::string> hasher;
    size_t hash = hasher(data);
    
    std::stringstream ss;
    ss << "HASH_" << std::hex << hash;
    return ss.str();
}

// Background processing methods
void DistributedDebuggingMonitor::process_log_queue() {
    logger::log_info("Started log processing thread");
    
    std::vector<LogEntry> batch;
    batch.reserve(config_.log_batch_size);
    
    while (log_processing_active_.load()) {
        try {
            // Collect batch of logs
            {
                std::lock_guard<std::mutex> lock(log_buffer_mutex_);
                while (!log_buffer_.empty() && batch.size() < config_.log_batch_size) {
                    batch.push_back(log_buffer_.front());
                    log_buffer_.pop();
                }
            }
            
            // Process batch (in a real implementation, would send to centralized logging)
            if (!batch.empty()) {
                logger::log_debug("Processing log batch of size: " + std::to_string(batch.size()));
                
                // Here you would send logs to centralized logging system
                // For now, just clear the batch
                batch.clear();
            }
            
            // Sleep for flush interval
            std::this_thread::sleep_for(std::chrono::milliseconds(config_.log_flush_interval_ms));
            
        } catch (const std::exception& e) {
            logger::log_error("Error in log processing thread: " + std::string(e.what()));
        }
    }
    
    logger::log_info("Log processing thread stopped");
}

void DistributedDebuggingMonitor::run_health_monitoring_loop() {
    logger::log_info("Started health monitoring thread");
    
    while (health_monitoring_active_.load()) {
        try {
            run_all_health_checks();
            
            // Check for real-time anomaly detection
            if (real_time_anomaly_detection_.load()) {
                auto anomalies = detect_system_anomalies();
                
                for (const auto& anomaly : anomalies) {
                    if (anomaly.severity_score > 0.7f) {
                        log_warning("system", "anomaly_detection", 
                                   "High-severity anomaly detected: " + anomaly.description);
                    }
                }
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(config_.health_check_interval_ms));
            
        } catch (const std::exception& e) {
            logger::log_error("Error in health monitoring thread: " + std::string(e.what()));
        }
    }
    
    logger::log_info("Health monitoring thread stopped");
}

void DistributedDebuggingMonitor::run_metrics_collection_loop() {
    logger::log_info("Started metrics collection thread");
    
    while (metrics_collection_active_.load()) {
        try {
            // Collect metrics for all known nodes
            std::vector<std::string> node_ids;
            
            {
                std::lock_guard<std::mutex> lock(metrics_mutex_);
                for (const auto& [node_id, metrics] : node_metrics_) {
                    node_ids.push_back(node_id);
                }
            }
            
            // If no nodes yet, add system node
            if (node_ids.empty()) {
                node_ids.push_back("system");
            }
            
            for (const auto& node_id : node_ids) {
                collect_system_metrics(node_id);
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(config_.metrics_collection_interval_ms));
            
        } catch (const std::exception& e) {
            logger::log_error("Error in metrics collection thread: " + std::string(e.what()));
        }
    }
    
    logger::log_info("Metrics collection thread stopped");
}

// System resource collection
SystemMetrics DistributedDebuggingMonitor::collect_system_resources(const std::string& node_id) const {
    SystemMetrics metrics;
    metrics.node_id = node_id;
    metrics.collection_timestamp = std::chrono::steady_clock::now();
    
    try {
        metrics.cpu_usage_percent = get_cpu_usage();
        metrics.memory_usage_percent = get_memory_usage();
        metrics.disk_usage_percent = get_disk_usage();
        
        auto network_usage = get_network_usage();
        metrics.network_rx_bytes_per_sec = network_usage.first;
        metrics.network_tx_bytes_per_sec = network_usage.second;
        
        // TODO: Collect training-specific metrics from the actual training system
        metrics.gradient_computation_time_ms = 0.0f;
        metrics.communication_time_ms = 0.0f;
        metrics.gradients_processed = 0;
        metrics.consensus_rounds_participated = 0;
        metrics.gradient_norm = 0.0f;
        metrics.loss_value = 0.0f;
        metrics.byzantine_detections = 0;
        metrics.reputation_score = 1.0f;
        
    } catch (const std::exception& e) {
        logger::log_error("Error collecting system metrics: " + std::string(e.what()));
    }
    
    return metrics;
}

float DistributedDebuggingMonitor::get_cpu_usage() const {
#ifdef _WIN32
    // Windows implementation
    static PDH_HQUERY query = nullptr;
    static PDH_HCOUNTER counter = nullptr;
    
    if (query == nullptr) {
        PdhOpenQuery(nullptr, 0, &query);
        PdhAddCounter(query, L"\\Processor(_Total)\\% Processor Time", 0, &counter);
        PdhCollectQueryData(query);
        return 0.0f; // First call always returns 0
    }
    
    PdhCollectQueryData(query);
    PDH_FMT_COUNTERVALUE value;
    PdhGetFormattedCounterValue(counter, PDH_FMT_DOUBLE, nullptr, &value);
    
    return static_cast<float>(value.doubleValue) / 100.0f;
    
#else
    // Linux implementation
    static long long last_total = 0, last_idle = 0;
    
    std::ifstream stat_file("/proc/stat");
    if (!stat_file.is_open()) {
        return 0.0f;
    }
    
    std::string line;
    std::getline(stat_file, line);
    
    std::istringstream iss(line);
    std::string cpu_label;
    long long user, nice, system, idle, iowait, irq, softirq, steal, guest, guest_nice;
    
    iss >> cpu_label >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal >> guest >> guest_nice;
    
    long long current_idle = idle + iowait;
    long long current_total = user + nice + system + idle + iowait + irq + softirq + steal;
    
    if (last_total == 0) {
        last_total = current_total;
        last_idle = current_idle;
        return 0.0f;
    }
    
    long long total_diff = current_total - last_total;
    long long idle_diff = current_idle - last_idle;
    
    last_total = current_total;
    last_idle = current_idle;
    
    if (total_diff == 0) return 0.0f;
    
    return 1.0f - static_cast<float>(idle_diff) / static_cast<float>(total_diff);
#endif
}

float DistributedDebuggingMonitor::get_memory_usage() const {
#ifdef _WIN32
    MEMORYSTATUSEX memory_status;
    memory_status.dwLength = sizeof(memory_status);
    GlobalMemoryStatusEx(&memory_status);
    
    return static_cast<float>(memory_status.dwMemoryLoad) / 100.0f;
    
#else
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        unsigned long total_ram = info.totalram * info.mem_unit;
        unsigned long free_ram = info.freeram * info.mem_unit;
        unsigned long used_ram = total_ram - free_ram;
        
        return static_cast<float>(used_ram) / static_cast<float>(total_ram);
    }
    
    return 0.0f;
#endif
}

float DistributedDebuggingMonitor::get_disk_usage() const {
#ifdef _WIN32
    ULARGE_INTEGER free_bytes, total_bytes;
    if (GetDiskFreeSpaceEx(L"C:\\", &free_bytes, &total_bytes, nullptr)) {
        ULONGLONG used_bytes = total_bytes.QuadPart - free_bytes.QuadPart;
        return static_cast<float>(used_bytes) / static_cast<float>(total_bytes.QuadPart);
    }
    
    return 0.0f;
    
#else
    struct statvfs stat;
    if (statvfs("/", &stat) == 0) {
        unsigned long total_space = stat.f_blocks * stat.f_frsize;
        unsigned long free_space = stat.f_avail * stat.f_frsize;
        unsigned long used_space = total_space - free_space;
        
        return static_cast<float>(used_space) / static_cast<float>(total_space);
    }
    
    return 0.0f;
#endif
}

std::pair<float, float> DistributedDebuggingMonitor::get_network_usage() const {
    // Simplified network usage (would need more sophisticated implementation in production)
    return {0.0f, 0.0f}; // rx, tx bytes/sec
}

// Anomaly detection implementations
std::vector<DistributedDebuggingMonitor::AnomalyAlert> DistributedDebuggingMonitor::detect_performance_anomalies() const {
    std::vector<AnomalyAlert> anomalies;
    
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    for (const auto& [node_id, metrics_history] : node_metrics_) {
        if (metrics_history.size() < 10) continue; // Need enough history
        
        // Calculate average gradient computation time
        float total_gradient_time = 0.0f;
        size_t count = 0;
        
        for (const auto& metric : metrics_history) {
            if (metric.gradient_computation_time_ms > 0) {
                total_gradient_time += metric.gradient_computation_time_ms;
                count++;
            }
        }
        
        if (count > 0) {
            float avg_gradient_time = total_gradient_time / count;
            float latest_gradient_time = metrics_history.back().gradient_computation_time_ms;
            
            // Check if latest time is significantly higher than average
            if (latest_gradient_time > avg_gradient_time * 3.0f && latest_gradient_time > 1000.0f) {
                AnomalyAlert alert;
                alert.node_id = node_id;
                alert.anomaly_type = "performance";
                alert.description = "Gradient computation time significantly increased";
                alert.severity_score = std::min(1.0f, latest_gradient_time / (avg_gradient_time * 5.0f));
                alert.evidence["latest_time_ms"] = latest_gradient_time;
                alert.evidence["average_time_ms"] = avg_gradient_time;
                alert.detection_timestamp = std::chrono::steady_clock::now();
                
                anomalies.push_back(alert);
            }
        }
    }
    
    return anomalies;
}

std::vector<DistributedDebuggingMonitor::AnomalyAlert> DistributedDebuggingMonitor::detect_resource_anomalies() const {
    std::vector<AnomalyAlert> anomalies;
    
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    for (const auto& [node_id, metrics_history] : node_metrics_) {
        if (metrics_history.empty()) continue;
        
        const auto& latest = metrics_history.back();
        
        // High resource usage alerts
        if (latest.cpu_usage_percent > 0.9f) {
            AnomalyAlert alert;
            alert.node_id = node_id;
            alert.anomaly_type = "resource";
            alert.description = "High CPU usage detected";
            alert.severity_score = latest.cpu_usage_percent;
            alert.evidence["cpu_usage"] = latest.cpu_usage_percent;
            alert.detection_timestamp = std::chrono::steady_clock::now();
            anomalies.push_back(alert);
        }
        
        if (latest.memory_usage_percent > 0.9f) {
            AnomalyAlert alert;
            alert.node_id = node_id;
            alert.anomaly_type = "resource";
            alert.description = "High memory usage detected";
            alert.severity_score = latest.memory_usage_percent;
            alert.evidence["memory_usage"] = latest.memory_usage_percent;
            alert.detection_timestamp = std::chrono::steady_clock::now();
            anomalies.push_back(alert);
        }
    }
    
    return anomalies;
}

std::vector<DistributedDebuggingMonitor::AnomalyAlert> DistributedDebuggingMonitor::detect_error_rate_anomalies() const {
    // Simplified error rate detection based on log entries
    return {}; // TODO: Implement based on error log frequency
}

std::vector<DistributedDebuggingMonitor::AnomalyAlert> DistributedDebuggingMonitor::detect_behavioral_anomalies() const {
    // Simplified behavioral anomaly detection
    return {}; // TODO: Implement based on consensus participation patterns
}

// Utility methods
std::string DistributedDebuggingMonitor::generate_trace_id() {
    std::stringstream ss;
    ss << "trace_" << std::hex << next_trace_id_++;
    return ss.str();
}

std::string DistributedDebuggingMonitor::generate_span_id() {
    std::stringstream ss;
    ss << "span_" << std::hex << next_span_id_++;
    return ss.str();
}

bool DistributedDebuggingMonitor::should_sample_trace() const {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    return dis(gen) < config_.trace_sampling_rate;
}

} // namespace debugging
