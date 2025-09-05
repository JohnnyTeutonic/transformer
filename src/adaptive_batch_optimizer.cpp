#include "adaptive_batch_optimizer.hpp"
#include "logger.hpp"
#include <algorithm>
#include <sstream>
#include <iomanip>

#ifdef CUDA_AVAILABLE
#include <cuda_runtime.h>
#include <nvml.h>
#endif

namespace adaptive_batch {

AdaptiveBatchOptimizer::AdaptiveBatchOptimizer() {
    #ifdef CUDA_AVAILABLE
    cuda_manager_ = std::make_unique<cuda::CudaManager>();
    
    // Initialize NVML for memory monitoring
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        logger::log_warning("Failed to initialize NVML for GPU monitoring");
    }
    #endif
    
    last_measurement_ = std::chrono::steady_clock::now();
    throughput_history_.reserve(100); // Keep last 100 measurements
}

ProbeResult AdaptiveBatchOptimizer::probe_optimal_batch_size(size_t seq_len, size_t model_params, bool force_reprobe) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    std::string cache_key = generate_cache_key(seq_len, model_params);
    
    // Check cache first
    if (!force_reprobe && configuration_cache_.find(cache_key) != configuration_cache_.end()) {
        logger::log_info("Using cached batch size configuration for seq_len=" + std::to_string(seq_len) + 
                         ", model_params=" + std::to_string(model_params));
        return configuration_cache_[cache_key];
    }
    
    logger::log_info("Probing optimal batch size for seq_len=" + std::to_string(seq_len) + 
                     ", model_params=" + std::to_string(model_params));
    
    // Get current memory info
    MemoryInfo mem_info = get_memory_info();
    if (mem_info.total_memory == 0) {
        logger::log_warning("Could not get GPU memory info, using conservative batch size");
        ProbeResult result;
        result.optimal_batch_size = 8; // Conservative fallback
        result.success = false;
        return result;
    }
    
    // Calculate reasonable search bounds
    size_t max_theoretical_batch = static_cast<size_t>(
        (mem_info.free_memory * (1.0f - MEMORY_SAFETY_MARGIN)) / 
        estimate_memory_usage(1, seq_len, model_params)
    );
    
    size_t max_batch = std::min(max_theoretical_batch, MAX_BATCH_SIZE);
    size_t min_batch = MIN_BATCH_SIZE;
    
    logger::log_info("Batch size search range: [" + std::to_string(min_batch) + 
                     ", " + std::to_string(max_batch) + "]");
    
    // Perform binary search
    ProbeResult result = binary_search_batch_size(seq_len, model_params, min_batch, max_batch);
    
    // Cache the result
    if (result.success) {
        configuration_cache_[cache_key] = result;
        logger::log_info("Optimal batch size found: " + std::to_string(result.optimal_batch_size) + 
                         " (memory utilization: " + std::to_string(result.memory_utilization * 100.0f) + "%)");
    }
    
    return result;
}

BatchConfiguration AdaptiveBatchOptimizer::adjust_batch_size_runtime(const BatchConfiguration& current_config, float memory_pressure) {
    BatchConfiguration adjusted_config = current_config;
    adjusted_config.memory_pressure = memory_pressure;
    
    // If memory pressure is high, reduce batch size
    if (memory_pressure > 0.9f) {
        size_t new_batch_size = static_cast<size_t>(current_config.batch_size * 0.8f);
        adjusted_config.batch_size = std::max(new_batch_size, MIN_BATCH_SIZE);
        
        logger::log_warning("High memory pressure (" + std::to_string(memory_pressure * 100.0f) + 
                           "%), reducing batch size from " + std::to_string(current_config.batch_size) + 
                           " to " + std::to_string(adjusted_config.batch_size));
    }
    // If memory pressure is low, we could potentially increase batch size
    else if (memory_pressure < 0.6f && current_config.batch_size < MAX_BATCH_SIZE) {
        size_t new_batch_size = static_cast<size_t>(current_config.batch_size * 1.1f);
        adjusted_config.batch_size = std::min(new_batch_size, MAX_BATCH_SIZE);
        
        logger::log_info("Low memory pressure (" + std::to_string(memory_pressure * 100.0f) + 
                        "%), increasing batch size from " + std::to_string(current_config.batch_size) + 
                        " to " + std::to_string(adjusted_config.batch_size));
    }
    
    return adjusted_config;
}

BatchConfiguration AdaptiveBatchOptimizer::handle_oom_recovery(const BatchConfiguration& failed_config) {
    BatchConfiguration recovery_config = failed_config;
    recovery_config.oom_detected = true;
    
    // Aggressively reduce batch size
    size_t new_batch_size = static_cast<size_t>(failed_config.batch_size * OOM_RECOVERY_FACTOR);
    recovery_config.batch_size = std::max(new_batch_size, MIN_BATCH_SIZE);
    
    logger::log_error("OOM detected! Reducing batch size from " + std::to_string(failed_config.batch_size) + 
                     " to " + std::to_string(recovery_config.batch_size));
    
    // Invalidate cache for this configuration
    std::string cache_key = generate_cache_key(failed_config.sequence_length, failed_config.model_parameters);
    std::lock_guard<std::mutex> lock(cache_mutex_);
    configuration_cache_.erase(cache_key);
    
    return recovery_config;
}

MemoryInfo AdaptiveBatchOptimizer::get_memory_info() const {
    MemoryInfo info;
    
    #ifdef CUDA_AVAILABLE
    size_t free_bytes, total_bytes;
    cudaError_t cuda_status = cudaMemGetInfo(&free_bytes, &total_bytes);
    
    if (cuda_status == cudaSuccess) {
        info.total_memory = total_bytes;
        info.free_memory = free_bytes;
        info.used_memory = total_bytes - free_bytes;
        info.utilization = static_cast<float>(info.used_memory) / static_cast<float>(info.total_memory);
    } else {
        logger::log_warning("Failed to get CUDA memory info: " + std::string(cudaGetErrorString(cuda_status)));
    }
    #else
    logger::log_warning("CUDA not available, cannot get GPU memory info");
    #endif
    
    return info;
}

void AdaptiveBatchOptimizer::update_throughput(float samples_per_second) {
    throughput_history_.push_back(samples_per_second);
    
    // Keep only recent measurements
    if (throughput_history_.size() > 100) {
        throughput_history_.erase(throughput_history_.begin());
    }
    
    last_measurement_ = std::chrono::steady_clock::now();
}

size_t AdaptiveBatchOptimizer::get_recommended_batch_size(size_t seq_len, size_t model_params, float target_memory_util) {
    ProbeResult result = probe_optimal_batch_size(seq_len, model_params);
    
    if (result.success && result.memory_utilization <= target_memory_util) {
        return result.optimal_batch_size;
    }
    
    // Fallback: estimate based on memory constraints
    MemoryInfo mem_info = get_memory_info();
    if (mem_info.total_memory > 0) {
        size_t target_memory = static_cast<size_t>(mem_info.total_memory * target_memory_util);
        size_t per_sample_memory = estimate_memory_usage(1, seq_len, model_params);
        
        if (per_sample_memory > 0) {
            return std::max(MIN_BATCH_SIZE, std::min(MAX_BATCH_SIZE, target_memory / per_sample_memory));
        }
    }
    
    // Conservative fallback
    return 8;
}

void AdaptiveBatchOptimizer::clear_cache() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    configuration_cache_.clear();
    logger::log_info("Cleared batch size configuration cache");
}

std::unordered_map<std::string, float> AdaptiveBatchOptimizer::get_performance_stats() const {
    std::unordered_map<std::string, float> stats;
    
    if (!throughput_history_.empty()) {
        float sum = 0.0f;
        float max_throughput = 0.0f;
        float min_throughput = std::numeric_limits<float>::max();
        
        for (float throughput : throughput_history_) {
            sum += throughput;
            max_throughput = std::max(max_throughput, throughput);
            min_throughput = std::min(min_throughput, throughput);
        }
        
        stats["avg_throughput"] = sum / throughput_history_.size();
        stats["max_throughput"] = max_throughput;
        stats["min_throughput"] = min_throughput;
        stats["throughput_samples"] = static_cast<float>(throughput_history_.size());
    }
    
    MemoryInfo mem_info = get_memory_info();
    stats["memory_utilization"] = mem_info.utilization;
    stats["free_memory_gb"] = static_cast<float>(mem_info.free_memory) / (1024.0f * 1024.0f * 1024.0f);
    stats["total_memory_gb"] = static_cast<float>(mem_info.total_memory) / (1024.0f * 1024.0f * 1024.0f);
    
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        stats["cached_configurations"] = static_cast<float>(configuration_cache_.size());
    }
    
    return stats;
}

std::string AdaptiveBatchOptimizer::generate_cache_key(size_t seq_len, size_t model_params) const {
    std::stringstream ss;
    ss << seq_len << "_" << model_params;
    return ss.str();
}

size_t AdaptiveBatchOptimizer::estimate_memory_usage(size_t batch_size, size_t seq_len, size_t model_params) const {
    // Rough estimation based on transformer memory requirements
    // This is a simplified model - in practice, you'd want more sophisticated estimation
    
    // Activation memory: batch_size * seq_len * hidden_dim * num_layers * sizeof(float)
    // Assuming hidden_dim scales with sqrt(model_params) and reasonable number of layers
    size_t hidden_dim = static_cast<size_t>(std::sqrt(model_params / 12)); // Rough estimate
    size_t num_layers = std::max(1UL, model_params / (hidden_dim * hidden_dim * 4)); // Rough estimate
    
    size_t activation_memory = batch_size * seq_len * hidden_dim * num_layers * sizeof(float);
    
    // Gradient memory (roughly same as activation memory)
    size_t gradient_memory = activation_memory;
    
    // Model parameter memory
    size_t parameter_memory = model_params * sizeof(float);
    
    // Optimizer state memory (Adam: 2x parameters)
    size_t optimizer_memory = parameter_memory * 2;
    
    // Add some overhead (20%)
    size_t total_memory = static_cast<size_t>((activation_memory + gradient_memory + parameter_memory + optimizer_memory) * 1.2f);
    
    return total_memory;
}

ProbeResult AdaptiveBatchOptimizer::binary_search_batch_size(size_t seq_len, size_t model_params, size_t min_batch, size_t max_batch) {
    ProbeResult result;
    result.success = false;
    
    size_t left = min_batch;
    size_t right = max_batch;
    size_t best_batch_size = min_batch;
    
    while (left <= right) {
        size_t mid = left + (right - left) / 2;
        
        logger::log_debug("Testing batch size: " + std::to_string(mid));
        
        if (test_batch_configuration(mid, seq_len, model_params)) {
            // This batch size works, try larger
            best_batch_size = mid;
            left = mid + 1;
            result.success = true;
        } else {
            // This batch size is too large, try smaller
            if (mid == 0) break;
            right = mid - 1;
        }
    }
    
    if (result.success) {
        result.optimal_batch_size = best_batch_size;
        
        // Calculate memory utilization for the optimal batch size
        size_t estimated_usage = estimate_memory_usage(best_batch_size, seq_len, model_params);
        MemoryInfo mem_info = get_memory_info();
        if (mem_info.total_memory > 0) {
            result.memory_utilization = static_cast<float>(estimated_usage) / static_cast<float>(mem_info.total_memory);
        }
        
        result.throughput_estimate = calculate_throughput_estimate(best_batch_size, result.memory_utilization);
    }
    
    return result;
}

bool AdaptiveBatchOptimizer::test_batch_configuration(size_t batch_size, size_t seq_len, size_t model_params) {
    // Estimate memory usage
    size_t estimated_usage = estimate_memory_usage(batch_size, seq_len, model_params);
    
    // Get current memory info
    MemoryInfo mem_info = get_memory_info();
    if (mem_info.total_memory == 0) {
        return false; // Can't determine memory availability
    }
    
    // Check if estimated usage fits within available memory with safety margin
    size_t available_memory = static_cast<size_t>(mem_info.free_memory * (1.0f - MEMORY_SAFETY_MARGIN));
    
    bool fits = estimated_usage <= available_memory;
    
    logger::log_debug("Batch size " + std::to_string(batch_size) + 
                     ": estimated " + std::to_string(estimated_usage / (1024*1024)) + "MB, " +
                     "available " + std::to_string(available_memory / (1024*1024)) + "MB, " +
                     "fits: " + (fits ? "yes" : "no"));
    
    return fits;
}

float AdaptiveBatchOptimizer::calculate_throughput_estimate(size_t batch_size, float memory_util) const {
    // Simple throughput model based on batch size and memory utilization
    // In practice, you'd want to calibrate this based on actual measurements
    
    float base_throughput = 100.0f; // samples per second baseline
    
    // Throughput generally increases with batch size (up to a point)
    float batch_factor = std::min(2.0f, static_cast<float>(batch_size) / 32.0f);
    
    // But decreases if memory utilization is too high (memory bandwidth bottleneck)
    float memory_factor = memory_util < 0.8f ? 1.0f : (1.0f - (memory_util - 0.8f) * 2.0f);
    memory_factor = std::max(0.1f, memory_factor);
    
    return base_throughput * batch_factor * memory_factor;
}

} // namespace adaptive_batch
