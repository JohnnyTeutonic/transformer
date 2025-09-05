#pragma once

#include "config.hpp"
#include <vector>
#include <cstddef>

/**
 * Adaptive batch scheduler that dynamically adjusts batch size based on:
 * - Available GPU/CPU memory
 * - System performance metrics
 * - Network conditions (for P2P training)
 * - Training convergence characteristics
 */
class AdaptiveBatchScheduler {
public:
    struct PerformanceMetrics {
        size_t batch_size;
        float samples_per_second;
        float memory_utilization;
        float gpu_utilization;
        float network_latency_ms;
        float loss_value;
        
        PerformanceMetrics() : batch_size(0), samples_per_second(0.0f), 
                              memory_utilization(0.0f), gpu_utilization(0.0f),
                              network_latency_ms(0.0f), loss_value(0.0f) {}
    };
    
    struct SystemMetrics {
        size_t total_memory;
        size_t available_memory;
        float memory_utilization;
        float gpu_utilization;
        float network_bandwidth_mbps;
        float network_latency_ms;
        size_t num_peers;
        
        SystemMetrics() : total_memory(0), available_memory(0), memory_utilization(0.0f),
                         gpu_utilization(0.0f), network_bandwidth_mbps(0.0f),
                         network_latency_ms(0.0f), num_peers(0) {}
    };
    
    /**
     * Constructor
     * @param min_batch_size Minimum allowed batch size
     * @param max_batch_size Maximum allowed batch size
     */
    explicit AdaptiveBatchScheduler(size_t min_batch_size = 1, size_t max_batch_size = 128);
    
    /**
     * Calculate optimal batch size based on current system state
     * @param config Transformer configuration
     * @return Optimal batch size
     */
    size_t calculate_optimal_batch_size(const TransformerConfig& config);
    
    /**
     * Update performance metrics for adaptive scheduling
     * @param metrics Current performance metrics
     */
    void update_performance_metrics(const PerformanceMetrics& metrics);
    
    /**
     * Get current batch size
     * @return Current batch size
     */
    size_t get_current_batch_size() const { return current_batch_size_; }
    
    /**
     * Set memory utilization target (0.0 - 1.0)
     * @param target Target memory utilization ratio
     */
    void set_memory_utilization_target(float target);
    
    /**
     * Set adaptation rate for gradual batch size changes
     * @param rate Adaptation rate (0.01 - 0.5)
     */
    void set_adaptation_rate(float rate);
    
    /**
     * Print current statistics
     */
    void print_statistics() const;
    
    // Getters for monitoring
    float get_average_throughput() const { return avg_throughput_; }
    float get_average_memory_utilization() const { return avg_memory_utilization_; }
    float get_average_gpu_utilization() const { return avg_gpu_utilization_; }

private:
    // Configuration
    size_t min_batch_size_;
    size_t max_batch_size_;
    size_t current_batch_size_;
    float memory_utilization_target_;
    size_t performance_history_size_;
    float adaptation_rate_;
    
    // Performance tracking
    std::vector<PerformanceMetrics> performance_history_;
    float avg_throughput_;
    float avg_memory_utilization_;
    float avg_gpu_utilization_;
    
    /**
     * Get current system metrics
     * @return Current system state
     */
    SystemMetrics get_system_metrics() const;
    
    /**
     * Estimate memory usage per training sample
     * @param config Transformer configuration
     * @return Estimated memory per sample in bytes
     */
    size_t estimate_memory_per_sample(const TransformerConfig& config) const;
    
    /**
     * Calculate maximum batch size based on memory constraints
     * @param available_memory Available memory in bytes
     * @param memory_per_sample Memory required per sample
     * @return Memory-constrained batch size
     */
    size_t calculate_memory_constrained_batch_size(size_t available_memory,
                                                  size_t memory_per_sample) const;
    
    /**
     * Calculate optimal batch size for performance
     * @param metrics Current system metrics
     * @return Performance-optimal batch size
     */
    size_t calculate_performance_optimal_batch_size(const SystemMetrics& metrics) const;
    
    /**
     * Calculate optimal batch size for network efficiency (P2P training)
     * @param metrics Current system metrics
     * @return Network-optimal batch size
     */
    size_t calculate_network_optimal_batch_size(const SystemMetrics& metrics) const;
    
    /**
     * Apply gradual adaptation to avoid sudden batch size changes
     * @param target_batch_size Target batch size
     * @return Gradually adapted batch size
     */
    size_t apply_gradual_adaptation(size_t target_batch_size) const;
    
    /**
     * Update running averages from performance history
     */
    void update_running_averages();
};
