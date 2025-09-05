#pragma once

#include <vector>
#include <unordered_map>
#include <memory>
#include <chrono>
#include <mutex>
#include "types.hpp"
#include "cuda_manager.hpp"

namespace adaptive_batch {

struct MemoryInfo {
    size_t total_memory = 0;
    size_t free_memory = 0;
    size_t used_memory = 0;
    float utilization = 0.0f;
};

struct BatchConfiguration {
    size_t batch_size = 32;
    size_t sequence_length = 512;
    size_t model_parameters = 0;
    float memory_pressure = 0.0f;
    bool oom_detected = false;
};

struct ProbeResult {
    size_t optimal_batch_size = 0;
    float memory_utilization = 0.0f;
    float throughput_estimate = 0.0f;
    bool success = false;
};

class AdaptiveBatchOptimizer {
private:
    // Configuration cache for similar model/sequence combinations
    std::unordered_map<std::string, ProbeResult> configuration_cache_;
    std::mutex cache_mutex_;
    
    // Memory monitoring
    std::unique_ptr<cuda::CudaManager> cuda_manager_;
    MemoryInfo last_memory_info_;
    
    // Performance tracking
    std::vector<float> throughput_history_;
    std::chrono::steady_clock::time_point last_measurement_;
    
    // Safety parameters
    static constexpr float MEMORY_SAFETY_MARGIN = 0.15f; // Keep 15% free
    static constexpr float OOM_RECOVERY_FACTOR = 0.7f;   // Reduce by 30% after OOM
    static constexpr size_t MIN_BATCH_SIZE = 1;
    static constexpr size_t MAX_BATCH_SIZE = 1024;
    
public:
    AdaptiveBatchOptimizer();
    ~AdaptiveBatchOptimizer() = default;
    
    /**
     * Probe for optimal batch size using binary search
     * @param seq_len Sequence length for the batch
     * @param model_params Number of model parameters
     * @param force_reprobe Force reprobing even if cached result exists
     * @return ProbeResult containing optimal batch size and metrics
     */
    ProbeResult probe_optimal_batch_size(size_t seq_len, size_t model_params, bool force_reprobe = false);
    
    /**
     * Adjust batch size during runtime based on memory pressure
     * @param current_config Current batch configuration
     * @param memory_pressure Current memory pressure (0.0 - 1.0)
     * @return Adjusted batch configuration
     */
    BatchConfiguration adjust_batch_size_runtime(const BatchConfiguration& current_config, float memory_pressure);
    
    /**
     * Handle OOM recovery by reducing batch size
     * @param failed_config Configuration that caused OOM
     * @return Safe configuration for recovery
     */
    BatchConfiguration handle_oom_recovery(const BatchConfiguration& failed_config);
    
    /**
     * Get current memory information
     * @return MemoryInfo struct with current GPU memory stats
     */
    MemoryInfo get_memory_info() const;
    
    /**
     * Update throughput measurements for performance tracking
     * @param samples_per_second Current training throughput
     */
    void update_throughput(float samples_per_second);
    
    /**
     * Get recommended batch size for given constraints
     * @param seq_len Sequence length
     * @param model_params Model parameter count
     * @param target_memory_util Target memory utilization (0.0 - 1.0)
     * @return Recommended batch size
     */
    size_t get_recommended_batch_size(size_t seq_len, size_t model_params, float target_memory_util = 0.85f);
    
    /**
     * Clear configuration cache (useful for model changes)
     */
    void clear_cache();
    
    /**
     * Get performance statistics
     * @return Map of performance metrics
     */
    std::unordered_map<std::string, float> get_performance_stats() const;

private:
    /**
     * Generate cache key for configuration
     */
    std::string generate_cache_key(size_t seq_len, size_t model_params) const;
    
    /**
     * Estimate memory usage for given configuration
     */
    size_t estimate_memory_usage(size_t batch_size, size_t seq_len, size_t model_params) const;
    
    /**
     * Perform binary search for optimal batch size
     */
    ProbeResult binary_search_batch_size(size_t seq_len, size_t model_params, size_t min_batch, size_t max_batch);
    
    /**
     * Test if a specific batch configuration fits in memory
     */
    bool test_batch_configuration(size_t batch_size, size_t seq_len, size_t model_params);
    
    /**
     * Calculate throughput estimate based on batch size and memory utilization
     */
    float calculate_throughput_estimate(size_t batch_size, float memory_util) const;
};

} // namespace adaptive_batch
