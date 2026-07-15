#include "../include/heterogeneous_performance.hpp"
#include "../include/logger.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <fstream>

#ifdef CUDA_AVAILABLE
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

namespace heterogeneous {

HeterogeneousPerformanceManager::HeterogeneousPerformanceManager(const SynchronizationConfig& config)
    : sync_config_(config) {
    logger::log_info("Initializing Heterogeneous Performance Manager");
    logger::log_info("- Wait percentile: " + std::to_string(sync_config_.wait_percentile * 100) + "%");
    logger::log_info("- Min nodes required: " + std::to_string(sync_config_.min_nodes_required));
    logger::log_info("- Max wait time: " + std::to_string(sync_config_.max_wait_time_ms) + "ms");
    
    stats_ = {};
    stats_.last_rebalance = std::chrono::steady_clock::now();
}

HeterogeneousPerformanceManager::~HeterogeneousPerformanceManager() {
    benchmark_running_ = false;
    if (benchmark_thread_.joinable()) {
        benchmark_thread_.join();
    }
}

bool HeterogeneousPerformanceManager::register_node(const std::string& node_id, 
                                                   const NodeCapabilities& capabilities) {
    std::lock_guard<std::mutex> lock(capabilities_mutex_);
    
    node_capabilities_[node_id] = capabilities;
    
    {
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        stats_.active_nodes = node_capabilities_.size();
    }
    
    logger::log_info("Registered node " + node_id + " with compute score: " + 
                     std::to_string(capabilities.compute_score));
    
    return true;
}

bool HeterogeneousPerformanceManager::remove_node(const std::string& node_id) {
    std::lock_guard<std::mutex> lock(capabilities_mutex_);
    
    auto it = node_capabilities_.find(node_id);
    if (it != node_capabilities_.end()) {
        node_capabilities_.erase(it);
        
        {
            std::lock_guard<std::mutex> stats_lock(stats_mutex_);
            stats_.active_nodes = node_capabilities_.size();
        }
        
        logger::log_info("Removed node " + node_id);
        return true;
    }
    
    return false;
}

BenchmarkResult HeterogeneousPerformanceManager::run_performance_benchmark(const std::string& node_id) {
    logger::log_info("Running performance benchmark for node " + node_id);
    
    BenchmarkResult result = {};
    auto start_time = std::chrono::steady_clock::now();
    
    try {
        // CPU benchmark: Matrix multiplication
        const size_t matrix_size = 512;
        std::vector<float> a(matrix_size * matrix_size, 1.0f);
        std::vector<float> b(matrix_size * matrix_size, 1.0f);
        std::vector<float> c(matrix_size * matrix_size, 0.0f);
        
        auto cpu_start = std::chrono::steady_clock::now();
        
        // Simple matrix multiplication for FLOPS measurement
        for (size_t i = 0; i < matrix_size; ++i) {
            for (size_t j = 0; j < matrix_size; ++j) {
                for (size_t k = 0; k < matrix_size; ++k) {
                    c[i * matrix_size + j] += a[i * matrix_size + k] * b[k * matrix_size + j];
                }
            }
        }
        
        auto cpu_end = std::chrono::steady_clock::now();
        auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
        
        // Calculate FLOPS (2 * N^3 operations for matrix multiply)
        size_t operations = 2 * matrix_size * matrix_size * matrix_size;
        result.flops_per_second = static_cast<float>(operations) / (cpu_duration.count() * 1e-6f);
        
        // Memory bandwidth benchmark
        const size_t mem_test_size = 64 * 1024 * 1024; // 64 MB
        std::vector<float> mem_src(mem_test_size / sizeof(float));
        std::vector<float> mem_dst(mem_test_size / sizeof(float));
        
        // Fill source with random data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        for (auto& val : mem_src) {
            val = dis(gen);
        }
        
        auto mem_start = std::chrono::steady_clock::now();
        
        // Memory copy benchmark
        for (int i = 0; i < 10; ++i) {
            std::copy(mem_src.begin(), mem_src.end(), mem_dst.begin());
        }
        
        auto mem_end = std::chrono::steady_clock::now();
        auto mem_duration = std::chrono::duration_cast<std::chrono::microseconds>(mem_end - mem_start);
        
        size_t bytes_transferred = mem_test_size * 10 * 2; // Read + Write
        result.memory_bandwidth = static_cast<float>(bytes_transferred) / (mem_duration.count() * 1e-6f * 1e9f); // GB/s
        
        // Gradient computation simulation
        const size_t gradient_size = 1024 * 1024; // 1M parameters
        std::vector<float> gradients(gradient_size);
        std::vector<float> parameters(gradient_size);
        
        // Fill with test data
        for (size_t i = 0; i < gradient_size; ++i) {
            parameters[i] = dis(gen);
        }
        
        auto grad_start = std::chrono::steady_clock::now();
        
        // Simulate gradient computation (element-wise operations)
        for (size_t i = 0; i < gradient_size; ++i) {
            gradients[i] = parameters[i] * 0.5f + std::sin(parameters[i]) * 0.1f;
        }
        
        auto grad_end = std::chrono::steady_clock::now();
        result.gradient_compute_time = std::chrono::duration_cast<std::chrono::milliseconds>(grad_end - grad_start).count();
        
        // Communication latency (simulate network round-trip)
        auto comm_start = std::chrono::steady_clock::now();
        std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Minimum latency
        auto comm_end = std::chrono::steady_clock::now();
        result.communication_latency = std::chrono::duration_cast<std::chrono::microseconds>(comm_end - comm_start).count() / 1000.0f;
        
        // Aggregation speed (vector operations)
        std::vector<float> grad1(gradient_size), grad2(gradient_size), result_grad(gradient_size);
        for (size_t i = 0; i < gradient_size; ++i) {
            grad1[i] = dis(gen);
            grad2[i] = dis(gen);
        }
        
        auto agg_start = std::chrono::steady_clock::now();
        
        for (size_t i = 0; i < gradient_size; ++i) {
            result_grad[i] = (grad1[i] + grad2[i]) * 0.5f;
        }
        
        auto agg_end = std::chrono::steady_clock::now();
        auto agg_duration = std::chrono::duration_cast<std::chrono::microseconds>(agg_end - agg_start);
        result.aggregation_speed = static_cast<float>(gradient_size) / (agg_duration.count() * 1e-6f);
        
        result.benchmark_valid = true;
        
        // Store benchmark result in history
        {
            std::lock_guard<std::mutex> lock(performance_history_mutex_);
            auto& history = performance_history_[node_id];
            history.push_back(result);
            
            // Keep only last 10 benchmarks
            if (history.size() > 10) {
                history.erase(history.begin());
            }
        }
        
        {
            std::lock_guard<std::mutex> stats_lock(stats_mutex_);
            stats_.total_benchmarks_run++;
        }
        
        auto total_time = std::chrono::steady_clock::now() - start_time;
        logger::log_info("Benchmark completed for " + node_id + " in " + 
                        std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(total_time).count()) + "ms");
        logger::log_info("- FLOPS: " + std::to_string(result.flops_per_second / 1e9f) + " GFLOPS");
        logger::log_info("- Memory BW: " + std::to_string(result.memory_bandwidth) + " GB/s");
        logger::log_info("- Gradient time: " + std::to_string(result.gradient_compute_time) + "ms");
        
    } catch (const std::exception& e) {
        logger::log_error("Benchmark failed for node " + node_id + ": " + e.what());
        result.benchmark_valid = false;
    }
    
    return result;
}

std::vector<WorkAllocation> HeterogeneousPerformanceManager::allocate_gradient_work(
    size_t total_gradient_size, const std::vector<std::string>& available_nodes) {
    
    std::lock_guard<std::mutex> lock(capabilities_mutex_);
    
    logger::log_info("Allocating gradient work for " + std::to_string(total_gradient_size) + 
                     " parameters across " + std::to_string(available_nodes.size()) + " nodes");
    
    std::vector<NodeCapabilities> node_caps;
    for (const auto& node_id : available_nodes) {
        auto it = node_capabilities_.find(node_id);
        if (it != node_capabilities_.end()) {
            node_caps.push_back(it->second);
        }
    }
    
    if (node_caps.empty()) {
        logger::log_warning("No node capabilities found for work allocation");
        return {};
    }
    
    // Use load-balanced allocation by default
    return optimize_allocation_load_balance(total_gradient_size, node_caps);
}

std::vector<WorkAllocation> HeterogeneousPerformanceManager::optimize_allocation_load_balance(
    size_t total_gradient_size, const std::vector<NodeCapabilities>& nodes) {
    
    std::vector<WorkAllocation> allocations;
    
    // Calculate total compute capacity
    float total_compute_score = 0.0f;
    for (const auto& node : nodes) {
        total_compute_score += node.compute_score;
    }
    
    if (total_compute_score <= 0.0f) {
        logger::log_error("Invalid total compute score for load balancing");
        return allocations;
    }
    
    size_t allocated_so_far = 0;
    
    for (size_t i = 0; i < nodes.size(); ++i) {
        WorkAllocation allocation;
        allocation.node_id = nodes[i].node_id;
        
        // Calculate this node's share based on compute capability
        float node_share = nodes[i].compute_score / total_compute_score;
        
        if (i == nodes.size() - 1) {
            // Last node gets remaining work to avoid rounding errors
            allocation.gradient_shard_size = total_gradient_size - allocated_so_far;
        } else {
            allocation.gradient_shard_size = static_cast<size_t>(total_gradient_size * node_share);
        }
        
        allocation.gradient_shard_offset = allocated_so_far;
        allocation.expected_completion_time = predict_completion_time(nodes[i].node_id, allocation.gradient_shard_size);
        allocation.priority_level = static_cast<uint32_t>(nodes[i].compute_score * 10); // Scale to reasonable range
        
        allocations.push_back(allocation);
        allocated_so_far += allocation.gradient_shard_size;
        
        logger::log_debug("Allocated " + std::to_string(allocation.gradient_shard_size) + " parameters to " + 
                         allocation.node_id + " (score: " + std::to_string(nodes[i].compute_score) + ")");
    }
    
    return allocations;
}

HeterogeneousPerformanceManager::SyncResult HeterogeneousPerformanceManager::wait_for_gradient_completion(
    const std::vector<std::string>& participating_nodes,
    const std::unordered_map<std::string, bool>& completion_status) {
    
    SyncResult result;
    result.wait_time_ms = 0;
    result.should_proceed = false;
    
    auto start_time = std::chrono::steady_clock::now();
    
    while (true) {
        // Count completed nodes
        uint32_t completed_count = 0;
        for (const auto& node_id : participating_nodes) {
            auto it = completion_status.find(node_id);
            if (it != completion_status.end() && it->second) {
                completed_count++;
                result.completed_nodes.push_back(node_id);
            } else {
                result.pending_nodes.push_back(node_id);
            }
        }
        
        result.completion_percentage = static_cast<float>(completed_count) / participating_nodes.size();
        
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        result.wait_time_ms = static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count());
        
        // Check if we should proceed
        if (should_proceed_with_partial_sync(result.completion_percentage, result.wait_time_ms)) {
            result.should_proceed = true;
            break;
        }
        
        // Check timeout
        if (result.wait_time_ms >= sync_config_.max_wait_time_ms) {
            logger::log_warning("Gradient sync timeout reached after " + std::to_string(result.wait_time_ms) + "ms");
            result.should_proceed = completed_count >= sync_config_.min_nodes_required;
            break;
        }
        
        // Sleep before next check
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Clear vectors for next iteration
        result.completed_nodes.clear();
        result.pending_nodes.clear();
    }
    
    {
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        stats_.sync_completion_rate = result.completion_percentage;
        stats_.average_wait_time_ms = (stats_.average_wait_time_ms + result.wait_time_ms) / 2.0f;
    }
    
    logger::log_info("Sync completed: " + std::to_string(result.completion_percentage * 100) + "% of nodes (" +
                     std::to_string(result.completed_nodes.size()) + "/" + std::to_string(participating_nodes.size()) +
                     ") in " + std::to_string(result.wait_time_ms) + "ms");
    
    return result;
}

bool HeterogeneousPerformanceManager::should_proceed_with_partial_sync(float completion_percentage, 
                                                                      uint32_t elapsed_time_ms) const {
    std::lock_guard<std::mutex> lock(config_mutex_);
    
    // Always proceed if we have the target percentage
    if (completion_percentage >= sync_config_.wait_percentile) {
        return true;
    }
    
    // If adaptive timeout is enabled, use exponential backoff
    if (sync_config_.enable_adaptive_timeout) {
        // Reduce wait percentage over time
        float time_factor = static_cast<float>(elapsed_time_ms) / sync_config_.max_wait_time_ms;
        float adjusted_percentile = sync_config_.wait_percentile * (1.0f - time_factor * 0.3f);
        
        return completion_percentage >= adjusted_percentile;
    }
    
    return false;
}

float HeterogeneousPerformanceManager::predict_completion_time(const std::string& node_id, size_t work_size) const {
    std::lock_guard<std::mutex> lock(performance_history_mutex_);
    
    auto history_it = performance_history_.find(node_id);
    if (history_it == performance_history_.end() || history_it->second.empty()) {
        // No history available, use node capabilities as fallback
        std::lock_guard<std::mutex> cap_lock(capabilities_mutex_);
        auto cap_it = node_capabilities_.find(node_id);
        if (cap_it != node_capabilities_.end()) {
            // Rough estimate based on compute score
            return static_cast<float>(work_size) / (cap_it->second.compute_score * 1000.0f);
        }
        return 1000.0f; // Default fallback
    }
    
    // Use most recent benchmark results
    const auto& recent_benchmark = history_it->second.back();
    
    if (!recent_benchmark.benchmark_valid) {
        return 1000.0f; // Fallback for invalid benchmarks
    }
    
    // Predict based on gradient computation time and work size
    float base_time = recent_benchmark.gradient_compute_time;
    float scale_factor = static_cast<float>(work_size) / (1024.0f * 1024.0f); // Normalize to 1M parameters
    
    return base_time * scale_factor;
}

std::vector<std::string> HeterogeneousPerformanceManager::select_fastest_nodes(
    const std::vector<std::string>& candidates, size_t max_nodes) const {
    
    std::lock_guard<std::mutex> lock(capabilities_mutex_);
    
    std::vector<std::pair<std::string, float>> node_scores;
    
    for (const auto& node_id : candidates) {
        auto it = node_capabilities_.find(node_id);
        if (it != node_capabilities_.end()) {
            node_scores.emplace_back(node_id, it->second.compute_score);
        }
    }
    
    // Sort by compute score (descending)
    std::sort(node_scores.begin(), node_scores.end(),
             [](const auto& a, const auto& b) {
                 return a.second > b.second;
             });
    
    std::vector<std::string> selected_nodes;
    size_t select_count = std::min(max_nodes, node_scores.size());
    
    for (size_t i = 0; i < select_count; ++i) {
        selected_nodes.push_back(node_scores[i].first);
    }
    
    return selected_nodes;
}

bool HeterogeneousPerformanceManager::schedule_periodic_benchmarks(bool enable) {
    if (enable && !benchmark_running_.load()) {
        benchmark_running_ = true;
        benchmark_thread_ = std::thread(&HeterogeneousPerformanceManager::run_benchmark_thread, this);
        logger::log_info("Scheduled periodic benchmarks");
        return true;
    } else if (!enable && benchmark_running_.load()) {
        benchmark_running_ = false;
        if (benchmark_thread_.joinable()) {
            benchmark_thread_.join();
        }
        logger::log_info("Stopped periodic benchmarks");
        return true;
    }
    
    return false;
}

void HeterogeneousPerformanceManager::run_benchmark_thread() {
    logger::log_info("Started benchmark thread");
    
    while (benchmark_running_.load()) {
        try {
            std::vector<std::string> nodes_to_benchmark;
            
            {
                std::lock_guard<std::mutex> lock(capabilities_mutex_);
                for (const auto& [node_id, capabilities] : node_capabilities_) {
                    if (is_benchmark_outdated(capabilities)) {
                        nodes_to_benchmark.push_back(node_id);
                    }
                }
            }
            
            for (const auto& node_id : nodes_to_benchmark) {
                if (!benchmark_running_.load()) break;
                
                logger::log_info("Running scheduled benchmark for " + node_id);
                auto result = run_performance_benchmark(node_id);
                
                if (result.benchmark_valid) {
                    // Update node capabilities with new benchmark
                    std::lock_guard<std::mutex> lock(capabilities_mutex_);
                    auto it = node_capabilities_.find(node_id);
                    if (it != node_capabilities_.end()) {
                        it->second.compute_score = calculate_node_score(result);
                        it->second.last_benchmark = std::chrono::steady_clock::now();
                        it->second.benchmark_version++;
                    }
                }
            }
            
        } catch (const std::exception& e) {
            logger::log_error("Error in benchmark thread: " + std::string(e.what()));
        }
        
        // Sleep for 5 minutes between benchmark cycles
        std::this_thread::sleep_for(std::chrono::minutes(5));
    }
    
    logger::log_info("Benchmark thread stopped");
}

float HeterogeneousPerformanceManager::calculate_node_score(const BenchmarkResult& result) const {
    if (!result.benchmark_valid) {
        return 0.1f; // Minimum score for invalid benchmarks
    }
    
    // Weighted combination of different performance metrics
    float flops_score = result.flops_per_second / 1e9f; // Normalize to GFLOPS
    float memory_score = result.memory_bandwidth; // Already in GB/s
    float gradient_score = 1000.0f / std::max(1.0f, result.gradient_compute_time); // Inverse of time
    float comm_score = 100.0f / std::max(1.0f, result.communication_latency); // Inverse of latency
    float agg_score = result.aggregation_speed / 1e6f; // Normalize to millions/sec
    
    // Weighted average (compute-heavy workload weights)
    float total_score = (flops_score * 0.3f + 
                        memory_score * 0.2f + 
                        gradient_score * 0.3f + 
                        comm_score * 0.1f + 
                        agg_score * 0.1f);
    
    return std::max(0.1f, total_score); // Ensure minimum score
}

bool HeterogeneousPerformanceManager::is_benchmark_outdated(const NodeCapabilities& capabilities) const {
    auto now = std::chrono::steady_clock::now();
    auto time_since_benchmark = std::chrono::duration_cast<std::chrono::minutes>(now - capabilities.last_benchmark);
    
    // Benchmark is outdated if it's older than 30 minutes
    return time_since_benchmark.count() > 30;
}

HeterogeneousPerformanceManager::PerformanceStats HeterogeneousPerformanceManager::get_performance_stats() const {
    std::lock_guard<std::mutex> stats_lock(stats_mutex_);
    
    // Calculate average node score
    float total_score = 0.0f;
    {
        std::lock_guard<std::mutex> cap_lock(capabilities_mutex_);
        for (const auto& [node_id, capabilities] : node_capabilities_) {
            total_score += capabilities.compute_score;
        }
        if (!node_capabilities_.empty()) {
            const_cast<PerformanceStats&>(stats_).average_node_score = total_score / node_capabilities_.size();
        }
    }
    
    return stats_;
}

void HeterogeneousPerformanceManager::update_benchmark_from_training_data(const std::string& node_id, 
                                                                         float actual_time_ms, size_t work_size) {
    // Create a synthetic benchmark result from training data
    BenchmarkResult synthetic_result = {};
    synthetic_result.gradient_compute_time = actual_time_ms;
    synthetic_result.aggregation_speed = static_cast<float>(work_size) / (actual_time_ms / 1000.0f);
    synthetic_result.benchmark_valid = true;
    
    {
        std::lock_guard<std::mutex> lock(performance_history_mutex_);
        auto& history = performance_history_[node_id];
        history.push_back(synthetic_result);
        
        if (history.size() > 10) {
            history.erase(history.begin());
        }
    }
    
    // Update node score based on real performance
    std::lock_guard<std::mutex> cap_lock(capabilities_mutex_);
    auto it = node_capabilities_.find(node_id);
    if (it != node_capabilities_.end()) {
        float new_score = calculate_node_score(synthetic_result);
        // Blend with existing score (moving average)
        it->second.compute_score = (it->second.compute_score * 0.7f) + (new_score * 0.3f);
    }
}

void HeterogeneousPerformanceManager::log_performance_metrics(const std::string& output_path) const {
    try {
        std::ofstream file(output_path);
        if (!file.is_open()) {
            logger::log_error("Failed to open performance metrics file: " + output_path);
            return;
        }
        
        auto stats = get_performance_stats();
        
        file << "Heterogeneous Performance Metrics\n";
        file << "==================================\n";
        file << "Total Benchmarks Run: " << stats.total_benchmarks_run << "\n";
        file << "Active Nodes: " << stats.active_nodes << "\n";
        file << "Average Node Score: " << stats.average_node_score << "\n";
        file << "Sync Completion Rate: " << (stats.sync_completion_rate * 100) << "%\n";
        file << "Average Wait Time: " << stats.average_wait_time_ms << "ms\n";
        
        file << "\nNode Details:\n";
        file << "-------------\n";
        
        std::lock_guard<std::mutex> lock(capabilities_mutex_);
        for (const auto& [node_id, capabilities] : node_capabilities_) {
            file << "Node: " << node_id << "\n";
            file << "  Compute Score: " << capabilities.compute_score << "\n";
            file << "  Memory: " << capabilities.memory_gb << " GB\n";
            file << "  Network BW: " << capabilities.network_bandwidth << " Mbps\n";
            file << "  Has GPU: " << (capabilities.has_gpu ? "Yes" : "No") << "\n";
            if (capabilities.has_gpu) {
                file << "  GPU Memory: " << capabilities.gpu_memory_gb << " GB\n";
            }
            file << "\n";
        }
        
        file.close();
        logger::log_info("Performance metrics logged to: " + output_path);
        
    } catch (const std::exception& e) {
        logger::log_error("Failed to log performance metrics: " + std::string(e.what()));
    }
}

// Factory implementation
std::unique_ptr<HeterogeneousPerformanceManager> 
PerformanceManagerFactory::create_manager(AllocationStrategy strategy, const SynchronizationConfig& config) {
    
    auto manager = std::make_unique<HeterogeneousPerformanceManager>(config);
    
    logger::log_info("Created performance manager with strategy: " + std::to_string(static_cast<int>(strategy)));
    
    return manager;
}

} // namespace heterogeneous
