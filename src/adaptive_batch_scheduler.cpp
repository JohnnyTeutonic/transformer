#include "../include/adaptive_batch_scheduler.hpp"
#include <algorithm>
#include <iostream>
#include <cmath>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

AdaptiveBatchScheduler::AdaptiveBatchScheduler(size_t min_batch_size, size_t max_batch_size)
    : min_batch_size_(min_batch_size), max_batch_size_(max_batch_size),
      current_batch_size_(min_batch_size), memory_utilization_target_(0.85f),
      performance_history_size_(10), adaptation_rate_(0.1f) {
    
    std::cout << "AdaptiveBatchScheduler initialized:" << std::endl;
    std::cout << "- Min batch size: " << min_batch_size_ << std::endl;
    std::cout << "- Max batch size: " << max_batch_size_ << std::endl;
    std::cout << "- Memory utilization target: " << memory_utilization_target_ << std::endl;
    
    // Initialize performance history
    performance_history_.reserve(performance_history_size_);
}

size_t AdaptiveBatchScheduler::calculate_optimal_batch_size(const TransformerConfig& config) {
    // Get current system state
    SystemMetrics metrics = get_system_metrics();
    
    // Calculate memory requirements per sample
    size_t memory_per_sample = estimate_memory_per_sample(config);
    
    // Calculate maximum batch size based on available memory
    size_t memory_constrained_batch_size = calculate_memory_constrained_batch_size(
        metrics.available_memory, memory_per_sample);
    
    // Calculate performance-optimal batch size
    size_t performance_optimal_batch_size = calculate_performance_optimal_batch_size(metrics);
    
    // Calculate network-optimal batch size (for P2P training)
    size_t network_optimal_batch_size = calculate_network_optimal_batch_size(metrics);
    
    // Choose the most restrictive constraint
    size_t optimal_batch_size = std::min({
        memory_constrained_batch_size,
        performance_optimal_batch_size,
        network_optimal_batch_size,
        max_batch_size_
    });
    
    optimal_batch_size = std::max(optimal_batch_size, min_batch_size_);
    
    // Apply gradual adaptation to avoid sudden changes
    size_t adapted_batch_size = apply_gradual_adaptation(optimal_batch_size);
    
    std::cout << "Batch size calculation:" << std::endl;
    std::cout << "- Memory constrained: " << memory_constrained_batch_size << std::endl;
    std::cout << "- Performance optimal: " << performance_optimal_batch_size << std::endl;
    std::cout << "- Network optimal: " << network_optimal_batch_size << std::endl;
    std::cout << "- Final adapted: " << adapted_batch_size << std::endl;
    
    current_batch_size_ = adapted_batch_size;
    return adapted_batch_size;
}

void AdaptiveBatchScheduler::update_performance_metrics(const PerformanceMetrics& metrics) {
    // Add to performance history
    performance_history_.push_back(metrics);
    
    // Keep only recent history
    if (performance_history_.size() > performance_history_size_) {
        performance_history_.erase(performance_history_.begin());
    }
    
    // Update running averages
    update_running_averages();
    
    // Log performance update
    std::cout << "Performance metrics updated:" << std::endl;
    std::cout << "- Throughput: " << metrics.samples_per_second << " samples/sec" << std::endl;
    std::cout << "- Memory utilization: " << metrics.memory_utilization << std::endl;
    std::cout << "- GPU utilization: " << metrics.gpu_utilization << std::endl;
    std::cout << "- Network latency: " << metrics.network_latency_ms << " ms" << std::endl;
}

AdaptiveBatchScheduler::SystemMetrics AdaptiveBatchScheduler::get_system_metrics() const {
    SystemMetrics metrics;
    
#ifdef USE_CUDA
    // Get GPU memory info
    size_t free_memory, total_memory;
    cudaError_t error = cudaMemGetInfo(&free_memory, &total_memory);
    if (error == cudaSuccess) {
        metrics.total_memory = total_memory;
        metrics.available_memory = free_memory;
        metrics.memory_utilization = 1.0f - (static_cast<float>(free_memory) / total_memory);
    } else {
        // Fallback values
        metrics.total_memory = 8ULL * 1024 * 1024 * 1024; // 8GB default
        metrics.available_memory = metrics.total_memory / 2;
        metrics.memory_utilization = 0.5f;
    }
    
    // Get GPU utilization (simplified - in practice would use NVML)
    metrics.gpu_utilization = 0.8f; // Placeholder
#else
    // CPU-only fallback
    metrics.total_memory = 16ULL * 1024 * 1024 * 1024; // 16GB default
    metrics.available_memory = metrics.total_memory / 2;
    metrics.memory_utilization = 0.5f;
    metrics.gpu_utilization = 0.0f;
#endif
    
    // Network metrics (would be updated by P2P network layer)
    metrics.network_bandwidth_mbps = 100.0f; // Placeholder
    metrics.network_latency_ms = 50.0f;       // Placeholder
    metrics.num_peers = 4;                    // Placeholder
    
    return metrics;
}

size_t AdaptiveBatchScheduler::estimate_memory_per_sample(const TransformerConfig& config) const {
    // Estimate memory usage per sample
    size_t hidden_size = config.hidden_size;
    size_t num_layers = config.num_layers;
    size_t vocab_size = config.vocab_size;
    size_t max_seq_length = config.max_seq_length;
    
    // Forward pass memory
    size_t forward_memory = 0;
    forward_memory += max_seq_length * hidden_size * sizeof(float); // Input embeddings
    forward_memory += num_layers * max_seq_length * hidden_size * sizeof(float); // Layer outputs
    forward_memory += max_seq_length * vocab_size * sizeof(float); // Final logits
    
    // Attention memory (for all heads and layers)
    size_t attention_memory = num_layers * max_seq_length * max_seq_length * sizeof(float);
    
    // Gradient memory (roughly equal to parameter memory)
    size_t gradient_memory = forward_memory;
    
    // Add safety margin
    size_t total_memory = (forward_memory + attention_memory + gradient_memory) * 2;
    
    return total_memory;
}

size_t AdaptiveBatchScheduler::calculate_memory_constrained_batch_size(size_t available_memory,
                                                                     size_t memory_per_sample) const {
    if (memory_per_sample == 0) {
        return max_batch_size_;
    }
    
    // Reserve some memory for system operations
    size_t usable_memory = static_cast<size_t>(available_memory * memory_utilization_target_);
    
    size_t max_samples = usable_memory / memory_per_sample;
    return std::min(max_samples, max_batch_size_);
}

size_t AdaptiveBatchScheduler::calculate_performance_optimal_batch_size(const SystemMetrics& metrics) const {
    if (performance_history_.empty()) {
        return current_batch_size_;
    }
    
    // Find the batch size that maximizes throughput
    size_t best_batch_size = current_batch_size_;
    float best_throughput = 0.0f;
    
    for (const auto& perf : performance_history_) {
        if (perf.samples_per_second > best_throughput) {
            best_throughput = perf.samples_per_second;
            best_batch_size = perf.batch_size;
        }
    }
    
    // If GPU utilization is low, try increasing batch size
    if (metrics.gpu_utilization < 0.7f && best_batch_size < max_batch_size_) {
        best_batch_size = std::min(best_batch_size * 2, max_batch_size_);
    }
    
    // If GPU utilization is too high, decrease batch size
    if (metrics.gpu_utilization > 0.95f && best_batch_size > min_batch_size_) {
        best_batch_size = std::max(best_batch_size / 2, min_batch_size_);
    }
    
    return best_batch_size;
}

size_t AdaptiveBatchScheduler::calculate_network_optimal_batch_size(const SystemMetrics& metrics) const {
    // For P2P training, larger batches reduce communication frequency
    // but increase memory usage and latency
    
    float network_efficiency = metrics.network_bandwidth_mbps / (metrics.network_latency_ms + 1.0f);
    
    // If network is slow, use larger batches to amortize communication cost
    if (network_efficiency < 10.0f) {
        return std::min(current_batch_size_ * 2, max_batch_size_);
    }
    
    // If network is fast, can use smaller batches for better convergence
    if (network_efficiency > 100.0f) {
        return std::max(current_batch_size_ / 2, min_batch_size_);
    }
    
    return current_batch_size_;
}

size_t AdaptiveBatchScheduler::apply_gradual_adaptation(size_t target_batch_size) const {
    // Gradually adapt to avoid sudden changes that could destabilize training
    float adaptation_factor = 1.0f + adaptation_rate_;
    
    if (target_batch_size > current_batch_size_) {
        // Increase gradually
        size_t max_increase = static_cast<size_t>(current_batch_size_ * adaptation_factor);
        return std::min(target_batch_size, max_increase);
    } else if (target_batch_size < current_batch_size_) {
        // Decrease gradually
        size_t max_decrease = static_cast<size_t>(current_batch_size_ / adaptation_factor);
        return std::max(target_batch_size, max_decrease);
    }
    
    return target_batch_size;
}

void AdaptiveBatchScheduler::update_running_averages() {
    if (performance_history_.empty()) {
        return;
    }
    
    // Calculate running averages
    float total_throughput = 0.0f;
    float total_memory_util = 0.0f;
    float total_gpu_util = 0.0f;
    
    for (const auto& perf : performance_history_) {
        total_throughput += perf.samples_per_second;
        total_memory_util += perf.memory_utilization;
        total_gpu_util += perf.gpu_utilization;
    }
    
    size_t history_size = performance_history_.size();
    avg_throughput_ = total_throughput / history_size;
    avg_memory_utilization_ = total_memory_util / history_size;
    avg_gpu_utilization_ = total_gpu_util / history_size;
}

void AdaptiveBatchScheduler::set_memory_utilization_target(float target) {
    memory_utilization_target_ = std::clamp(target, 0.5f, 0.95f);
    std::cout << "Updated memory utilization target to: " << memory_utilization_target_ << std::endl;
}

void AdaptiveBatchScheduler::set_adaptation_rate(float rate) {
    adaptation_rate_ = std::clamp(rate, 0.01f, 0.5f);
    std::cout << "Updated adaptation rate to: " << adaptation_rate_ << std::endl;
}

void AdaptiveBatchScheduler::print_statistics() const {
    std::cout << "\n=== Adaptive Batch Scheduler Statistics ===" << std::endl;
    std::cout << "Current batch size: " << current_batch_size_ << std::endl;
    std::cout << "Average throughput: " << avg_throughput_ << " samples/sec" << std::endl;
    std::cout << "Average memory utilization: " << avg_memory_utilization_ << std::endl;
    std::cout << "Average GPU utilization: " << avg_gpu_utilization_ << std::endl;
    std::cout << "Performance history size: " << performance_history_.size() << std::endl;
    std::cout << "============================================\n" << std::endl;
}
