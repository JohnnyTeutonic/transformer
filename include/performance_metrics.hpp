#pragma once
#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <iostream>
#include "config.hpp"
#include "matrix.hpp"

/**
 * @brief Performance monitoring and profiling for transformer operations.
 * 
 * The PerformanceMetrics class provides comprehensive performance tracking
 * capabilities for transformer model operations, including:
 * - High-resolution timing measurements
 * - Memory usage tracking
 * - FLOPS calculations for attention operations
 * - Statistical aggregation of metrics
 */
class PerformanceMetrics {
  private:
    /// Stores start times for active timing operations
    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> start_times;
    
    /// Accumulates total time spent in each operation
    std::unordered_map<std::string, double> accumulated_times;
    
    /// Tracks number of calls to each operation
    std::unordered_map<std::string, size_t> call_counts;

    // Memory tracking
    size_t peak_memory_usage = 0;        ///< Maximum memory usage observed
    std::vector<size_t> memory_samples; ///< Historical memory usage data
    size_t global_step = 0;            ///< Global training step counter

    // Add memory tracking
    struct MemoryMetrics {
        size_t peak_memory_usage = 0;
        size_t current_memory_usage = 0;
        
        void track_allocation(size_t bytes) {
            current_memory_usage += bytes;
            peak_memory_usage = std::max(peak_memory_usage, current_memory_usage);
        }
        
        void track_deallocation(size_t bytes) {
            current_memory_usage -= bytes;
        }
    };

    const TransformerConfig* config_ptr = nullptr;  // Change to pointer to allow default construction

  public:
    // Add default constructor
    PerformanceMetrics() = default;

    // Modify existing constructor to store pointer
    explicit PerformanceMetrics(const TransformerConfig& config) : config_ptr(&config) {}

    /**
     * @brief Starts timing an operation.
     * 
     * Records the start time for a named operation. Must be paired
     * with a corresponding stop_timer call.
     * 
     * @param name Identifier for the operation being timed
     */
    void start_timer(const std::string& name);

    /**
     * @brief Stops timing an operation.
     * 
     * Calculates elapsed time since the corresponding start_timer
     * call and updates statistics.
     * 
     * @param name Identifier for the operation being timed
     * @throws std::runtime_error if no matching start_timer call exists
     */
    void stop_timer(const std::string& name);

    /**
     * @brief Gets average execution time for an operation.
     * 
     * Calculates the mean execution time across all recorded
     * instances of the named operation.
     * 
     * @param name Identifier for the operation
     * @return Average time in milliseconds
     */
    double get_average_time(const std::string& name) const;

    /**
     * @brief Records current memory usage.
     * 
     * Updates peak memory usage if necessary and stores
     * the sample for statistical analysis.
     * 
     * @param bytes_used Current memory usage in bytes
     */
    void record_memory_usage(size_t bytes_used);

    /**
     * @brief Gets peak memory usage.
     * @return Maximum memory usage observed in bytes
     */
    size_t get_peak_memory() const;

    /**
     * @brief Gets average memory usage.
     * @return Mean memory usage across all samples in bytes
     */
    double get_average_memory() const;

    /**
     * @brief Records FLOPs for an attention operation.
     * 
     * Calculates and accumulates floating point operations
     * for attention computation based on input dimensions.
     * 
     * @param seq_length Sequence length
     * @param num_heads Number of attention heads
     * @param head_dim Dimension of each head
     */
    void record_attention_flops(size_t seq_length, size_t num_heads, size_t head_dim);

    /**
     * @brief Gets attention computation performance.
     * @return Attention computation speed in GFLOPS
     */
    double get_attention_gflops() const;

    /**
     * @brief Prints all collected metrics.
     * 
     * Outputs a formatted report of:
     * - Timing statistics for all operations
     * - Memory usage statistics
     * - Computational performance metrics
     */
    void print_metrics() const;

    /**
     * @brief Resets all metrics.
     * 
     * Clears all accumulated statistics and resets counters
     * to their initial state.
     */
    void reset();

    // Modify log_matrix_stats to check pointer
    void log_matrix_stats(const std::string& name, const Matrix& mat) {
        if (!config_ptr || !config_ptr->debug_mode) {  // Check pointer before use
            return;
        }
        compute_and_log_stats(name, mat);
    }
    
    void log_training_stats(float loss, float accuracy) {
        if (!config_ptr || !config_ptr->debug_mode) {  // Check pointer before use
            return;
        }
        
        if (global_step % config_ptr->log_frequency == 0) {
            std::cout << "Step " << global_step 
                      << " Loss: " << loss 
                      << " Acc: " << accuracy << std::endl;
        }
        global_step++;
    }

    // Add method to set config after construction
    void set_config(const TransformerConfig& config) {
        config_ptr = &config;
    }

  private:
    void compute_and_log_stats(const std::string& name, const Matrix& mat) {
        float min_val = std::numeric_limits<float>::max();
        float max_val = std::numeric_limits<float>::lowest();
        float sum = 0.0f;
        float sum_sq = 0.0f;
        size_t count = mat.rows() * mat.cols();

        for (size_t i = 0; i < mat.rows(); ++i) {
            for (size_t j = 0; j < mat.cols(); ++j) {
                float val = mat(i, j);
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
                sum += val;
                sum_sq += val * val;
            }
        }

        float mean = sum / count;
        float variance = (sum_sq / count) - (mean * mean);
        float std_dev = std::sqrt(variance);

        std::cout << name << " stats:"
                  << " min=" << min_val
                  << " max=" << max_val
                  << " mean=" << mean
                  << " std=" << std_dev << std::endl;
    }
};