#include "../include/dynamic_load_balancer.hpp"
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cmath>

namespace load_balancing {

DynamicLoadBalancer::DynamicLoadBalancer(
    std::shared_ptr<profiling::PerformanceProfiler> profiler,
    const LoadBalancerConfig& config)
    : profiler_(profiler), config_(config), balancing_active_(false) {
    
    std::cout << "DynamicLoadBalancer initialized with:" << std::endl;
    std::cout << "- Rebalancing interval: " << config_.rebalancing_interval_ms << "ms" << std::endl;
    std::cout << "- Performance threshold: " << config_.performance_threshold << std::endl;
    std::cout << "- Min batch size: " << config_.min_batch_size << std::endl;
    std::cout << "- Max batch size: " << config_.max_batch_size << std::endl;
    std::cout << "- Workload migration: " << (config_.enable_workload_migration ? "enabled" : "disabled") << std::endl;
}

DynamicLoadBalancer::~DynamicLoadBalancer() {
    stop_load_balancing();
}

bool DynamicLoadBalancer::start_load_balancing() {
    if (balancing_active_.load()) {
        std::cout << "Dynamic load balancing already active" << std::endl;
        return true;
    }
    
    std::cout << "Starting dynamic load balancing..." << std::endl;
    
    balancing_active_.store(true);
    
    // Start load balancing threads
    performance_monitor_thread_ = std::thread(&DynamicLoadBalancer::performance_monitoring_thread, this);
    load_balancer_thread_ = std::thread(&DynamicLoadBalancer::load_balancing_thread, this);
    
    if (config_.enable_workload_migration) {
        workload_migration_thread_ = std::thread(&DynamicLoadBalancer::workload_migration_thread, this);
    }
    
    std::cout << "Dynamic load balancing started" << std::endl;
    return true;
}

void DynamicLoadBalancer::stop_load_balancing() {
    if (!balancing_active_.load()) {
        return;
    }
    
    std::cout << "Stopping dynamic load balancing..." << std::endl;
    
    balancing_active_.store(false);
    
    // Notify all condition variables
    balancing_cv_.notify_all();
    
    // Join threads
    if (performance_monitor_thread_.joinable()) {
        performance_monitor_thread_.join();
    }
    if (load_balancer_thread_.joinable()) {
        load_balancer_thread_.join();
    }
    if (workload_migration_thread_.joinable()) {
        workload_migration_thread_.join();
    }
    
    std::cout << "Dynamic load balancing stopped" << std::endl;
}

void DynamicLoadBalancer::register_peer(const std::string& peer_id, const PeerCapabilities& capabilities) {
    std::lock_guard<std::mutex> lock(peers_mutex_);
    
    PeerInfo info;
    info.peer_id = peer_id;
    info.capabilities = capabilities;
    info.current_load = PeerLoad{};
    info.last_update = std::chrono::steady_clock::now();
    info.is_active = true;
    
    peer_info_[peer_id] = info;
    
    std::cout << "Registered peer: " << peer_id 
              << " (CPU cores: " << capabilities.cpu_cores
              << ", GPU memory: " << capabilities.gpu_memory_gb << " GB)" << std::endl;
    
    // Trigger rebalancing
    balancing_cv_.notify_one();
}

void DynamicLoadBalancer::unregister_peer(const std::string& peer_id) {
    std::lock_guard<std::mutex> lock(peers_mutex_);
    
    auto it = peer_info_.find(peer_id);
    if (it != peer_info_.end()) {
        it->second.is_active = false;
        std::cout << "Unregistered peer: " << peer_id << std::endl;
        
        // Trigger workload migration if this peer had work
        if (it->second.current_load.assigned_batch_size > 0) {
            balancing_cv_.notify_one();
        }
    }
}

void DynamicLoadBalancer::update_peer_performance(const std::string& peer_id, const PeerPerformance& performance) {
    std::lock_guard<std::mutex> lock(peers_mutex_);
    
    auto it = peer_info_.find(peer_id);
    if (it != peer_info_.end()) {
        it->second.recent_performance.push_back({std::chrono::steady_clock::now(), performance});
        
        // Keep only recent performance history
        if (it->second.recent_performance.size() > config_.performance_history_size) {
            it->second.recent_performance.erase(it->second.recent_performance.begin());
        }
        
        it->second.last_update = std::chrono::steady_clock::now();
        
        // Update current load based on performance
        update_peer_load_estimate(it->second);
    }
}

WorkloadAssignment DynamicLoadBalancer::get_optimal_workload_assignment(uint32_t total_batch_size) {
    std::lock_guard<std::mutex> lock(peers_mutex_);
    
    WorkloadAssignment assignment;
    assignment.total_batch_size = total_batch_size;
    assignment.timestamp = std::chrono::steady_clock::now();
    
    // Get active peers
    std::vector<PeerInfo*> active_peers;
    for (auto& [peer_id, info] : peer_info_) {
        if (info.is_active && is_peer_healthy(info)) {
            active_peers.push_back(&info);
        }
    }
    
    if (active_peers.empty()) {
        std::cerr << "No active peers available for workload assignment" << std::endl;
        return assignment;
    }
    
    // Calculate performance scores for each peer
    std::vector<float> performance_scores;
    performance_scores.reserve(active_peers.size());
    
    for (const auto* peer : active_peers) {
        float score = calculate_peer_performance_score(*peer);
        performance_scores.push_back(score);
    }
    
    // Normalize scores
    float total_score = std::accumulate(performance_scores.begin(), performance_scores.end(), 0.0f);
    if (total_score <= 0.0f) {
        // Fallback to equal distribution
        uint32_t batch_per_peer = total_batch_size / active_peers.size();
        for (size_t i = 0; i < active_peers.size(); ++i) {
            PeerAssignment peer_assignment;
            peer_assignment.peer_id = active_peers[i]->peer_id;
            peer_assignment.assigned_batch_size = batch_per_peer;
            peer_assignment.priority = 1.0f;
            assignment.peer_assignments.push_back(peer_assignment);
        }
        return assignment;
    }
    
    // Assign workload proportional to performance scores
    uint32_t assigned_total = 0;
    for (size_t i = 0; i < active_peers.size(); ++i) {
        PeerAssignment peer_assignment;
        peer_assignment.peer_id = active_peers[i]->peer_id;
        
        float proportion = performance_scores[i] / total_score;
        uint32_t assigned_batch = static_cast<uint32_t>(total_batch_size * proportion);
        
        // Apply constraints
        assigned_batch = std::max(config_.min_batch_size, assigned_batch);
        assigned_batch = std::min(config_.max_batch_size, assigned_batch);
        
        // Adjust for peer capabilities
        assigned_batch = adjust_for_peer_capabilities(*active_peers[i], assigned_batch);
        
        peer_assignment.assigned_batch_size = assigned_batch;
        peer_assignment.priority = performance_scores[i];
        
        assignment.peer_assignments.push_back(peer_assignment);
        assigned_total += assigned_batch;
    }
    
    // Handle any remainder due to rounding
    if (assigned_total < total_batch_size) {
        uint32_t remainder = total_batch_size - assigned_total;
        // Assign remainder to the best performing peer
        if (!assignment.peer_assignments.empty()) {
            auto best_peer = std::max_element(assignment.peer_assignments.begin(),
                                            assignment.peer_assignments.end(),
                                            [](const auto& a, const auto& b) {
                                                return a.priority < b.priority;
                                            });
            best_peer->assigned_batch_size += remainder;
        }
    }
    
    std::cout << "Generated workload assignment for " << active_peers.size() 
              << " peers (total batch: " << total_batch_size << ")" << std::endl;
    
    return assignment;
}

std::vector<std::string> DynamicLoadBalancer::get_optimal_peer_selection(uint32_t required_peers) {
    std::lock_guard<std::mutex> lock(peers_mutex_);
    
    std::vector<std::pair<std::string, float>> peer_scores;
    
    for (const auto& [peer_id, info] : peer_info_) {
        if (info.is_active && is_peer_healthy(info)) {
            float score = calculate_peer_performance_score(info);
            peer_scores.emplace_back(peer_id, score);
        }
    }
    
    // Sort by performance score (descending)
    std::sort(peer_scores.begin(), peer_scores.end(),
             [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Select top peers
    std::vector<std::string> selected_peers;
    size_t num_to_select = std::min(static_cast<size_t>(required_peers), peer_scores.size());
    
    for (size_t i = 0; i < num_to_select; ++i) {
        selected_peers.push_back(peer_scores[i].first);
    }
    
    return selected_peers;
}

LoadBalancingStats DynamicLoadBalancer::get_load_balancing_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return balancing_stats_;
}

void DynamicLoadBalancer::performance_monitoring_thread() {
    std::cout << "Performance monitoring thread started" << std::endl;
    
    while (balancing_active_.load()) {
        auto start_time = std::chrono::steady_clock::now();
        
        // Monitor peer health and performance
        monitor_peer_health();
        
        // Detect performance degradation
        detect_performance_issues();
        
        // Sleep until next monitoring cycle
        std::unique_lock<std::mutex> lock(balancing_mutex_);
        balancing_cv_.wait_for(lock, std::chrono::milliseconds(config_.performance_monitoring_interval_ms),
                              [this] { return !balancing_active_.load(); });
    }
    
    std::cout << "Performance monitoring thread stopped" << std::endl;
}

void DynamicLoadBalancer::load_balancing_thread() {
    std::cout << "Load balancing thread started" << std::endl;
    
    while (balancing_active_.load()) {
        auto start_time = std::chrono::steady_clock::now();
        
        // Check if rebalancing is needed
        if (should_rebalance()) {
            perform_load_rebalancing();
        }
        
        // Update load balancing statistics
        update_balancing_statistics();
        
        // Sleep until next rebalancing cycle
        std::unique_lock<std::mutex> lock(balancing_mutex_);
        balancing_cv_.wait_for(lock, std::chrono::milliseconds(config_.rebalancing_interval_ms),
                              [this] { return !balancing_active_.load(); });
    }
    
    std::cout << "Load balancing thread stopped" << std::endl;
}

void DynamicLoadBalancer::workload_migration_thread() {
    std::cout << "Workload migration thread started" << std::endl;
    
    while (balancing_active_.load()) {
        auto start_time = std::chrono::steady_clock::now();
        
        // Check for peers that need workload migration
        check_for_migration_opportunities();
        
        // Process pending migrations
        process_pending_migrations();
        
        // Sleep until next migration cycle
        std::unique_lock<std::mutex> lock(migration_mutex_);
        migration_cv_.wait_for(lock, std::chrono::milliseconds(config_.migration_check_interval_ms),
                              [this] { return !balancing_active_.load(); });
    }
    
    std::cout << "Workload migration thread stopped" << std::endl;
}

void DynamicLoadBalancer::monitor_peer_health() {
    auto now = std::chrono::steady_clock::now();
    
    std::lock_guard<std::mutex> lock(peers_mutex_);
    
    for (auto& [peer_id, info] : peer_info_) {
        if (!info.is_active) continue;
        
        // Check if peer has been silent for too long
        auto time_since_update = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - info.last_update).count();
        
        if (time_since_update > config_.peer_timeout_ms) {
            std::cout << "Peer " << peer_id << " timed out (last update: " 
                     << time_since_update << "ms ago)" << std::endl;
            info.is_active = false;
            
            // Trigger rebalancing if this peer had work
            if (info.current_load.assigned_batch_size > 0) {
                balancing_cv_.notify_one();
            }
        }
    }
}

void DynamicLoadBalancer::detect_performance_issues() {
    std::lock_guard<std::mutex> lock(peers_mutex_);
    
    for (auto& [peer_id, info] : peer_info_) {
        if (!info.is_active || info.recent_performance.empty()) continue;
        
        // Calculate recent performance trend
        float performance_trend = calculate_performance_trend(info);
        
        // Check if performance is degrading
        if (performance_trend < -config_.performance_degradation_threshold) {
            std::cout << "Performance degradation detected for peer " << peer_id 
                     << " (trend: " << performance_trend << ")" << std::endl;
            
            // Mark for potential workload reduction
            info.needs_load_reduction = true;
            balancing_cv_.notify_one();
        }
        
        // Check if performance is improving
        if (performance_trend > config_.performance_improvement_threshold) {
            info.needs_load_reduction = false;
        }
    }
}

bool DynamicLoadBalancer::should_rebalance() {
    std::lock_guard<std::mutex> lock(peers_mutex_);
    
    // Check if any peer needs load adjustment
    for (const auto& [peer_id, info] : peer_info_) {
        if (info.needs_load_reduction || info.needs_load_increase) {
            return true;
        }
    }
    
    // Check load imbalance
    std::vector<float> load_scores;
    for (const auto& [peer_id, info] : peer_info_) {
        if (info.is_active) {
            float score = calculate_peer_performance_score(info);
            load_scores.push_back(score);
        }
    }
    
    if (load_scores.size() < 2) return false;
    
    // Calculate coefficient of variation
    float mean = std::accumulate(load_scores.begin(), load_scores.end(), 0.0f) / load_scores.size();
    float variance = 0.0f;
    for (float score : load_scores) {
        variance += (score - mean) * (score - mean);
    }
    variance /= load_scores.size();
    float cv = std::sqrt(variance) / mean;
    
    return cv > config_.load_imbalance_threshold;
}

void DynamicLoadBalancer::perform_load_rebalancing() {
    std::cout << "Performing load rebalancing..." << std::endl;
    
    std::lock_guard<std::mutex> lock(peers_mutex_);
    
    // Calculate total current workload
    uint32_t total_workload = 0;
    for (const auto& [peer_id, info] : peer_info_) {
        if (info.is_active) {
            total_workload += info.current_load.assigned_batch_size;
        }
    }
    
    if (total_workload == 0) return;
    
    // Generate new optimal assignment
    auto new_assignment = get_optimal_workload_assignment(total_workload);
    
    // Apply the new assignment
    for (const auto& assignment : new_assignment.peer_assignments) {
        auto it = peer_info_.find(assignment.peer_id);
        if (it != peer_info_.end()) {
            uint32_t old_batch = it->second.current_load.assigned_batch_size;
            it->second.current_load.assigned_batch_size = assignment.assigned_batch_size;
            
            std::cout << "Rebalanced peer " << assignment.peer_id 
                     << ": " << old_batch << " -> " << assignment.assigned_batch_size << std::endl;
        }
    }
    
    // Update statistics
    std::lock_guard<std::mutex> stats_lock(stats_mutex_);
    balancing_stats_.total_rebalancing_operations++;
    balancing_stats_.last_rebalancing_time = std::chrono::steady_clock::now();
}

void DynamicLoadBalancer::check_for_migration_opportunities() {
    std::lock_guard<std::mutex> lock(peers_mutex_);
    
    // Find overloaded and underloaded peers
    std::vector<std::string> overloaded_peers;
    std::vector<std::string> underloaded_peers;
    
    for (const auto& [peer_id, info] : peer_info_) {
        if (!info.is_active) continue;
        
        float performance_score = calculate_peer_performance_score(info);
        
        if (performance_score < config_.performance_threshold * 0.5f) {
            overloaded_peers.push_back(peer_id);
        } else if (performance_score > config_.performance_threshold * 1.5f) {
            underloaded_peers.push_back(peer_id);
        }
    }
    
    // Create migration plans
    for (const auto& overloaded_peer : overloaded_peers) {
        if (!underloaded_peers.empty()) {
            // Find best target peer
            std::string target_peer = find_best_migration_target(overloaded_peer, underloaded_peers);
            
            if (!target_peer.empty()) {
                create_migration_plan(overloaded_peer, target_peer);
            }
        }
    }
}

void DynamicLoadBalancer::process_pending_migrations() {
    std::lock_guard<std::mutex> lock(migration_mutex_);
    
    auto now = std::chrono::steady_clock::now();
    
    auto it = pending_migrations_.begin();
    while (it != pending_migrations_.end()) {
        auto& migration = *it;
        
        // Check if migration has timed out
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - migration.created_time).count();
        
        if (elapsed > config_.migration_timeout_ms) {
            std::cout << "Migration timed out: " << migration.migration_id << std::endl;
            migration.status = MigrationStatus::FAILED;
            it = pending_migrations_.erase(it);
            continue;
        }
        
        // Process migration based on status
        if (migration.status == MigrationStatus::PENDING) {
            execute_migration(migration);
        }
        
        ++it;
    }
}

float DynamicLoadBalancer::calculate_peer_performance_score(const PeerInfo& peer) const {
    if (peer.recent_performance.empty()) {
        // Default score based on capabilities
        return calculate_capability_score(peer.capabilities);
    }
    
    // Calculate weighted average of recent performance
    float total_weight = 0.0f;
    float weighted_score = 0.0f;
    auto now = std::chrono::steady_clock::now();
    
    for (const auto& [timestamp, performance] : peer.recent_performance) {
        auto age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - timestamp).count();
        float weight = std::exp(-static_cast<float>(age_ms) / 60000.0f); // Exponential decay over 1 minute
        
        float score = calculate_performance_score(performance);
        weighted_score += score * weight;
        total_weight += weight;
    }
    
    return total_weight > 0.0f ? weighted_score / total_weight : 0.0f;
}

float DynamicLoadBalancer::calculate_capability_score(const PeerCapabilities& capabilities) const {
    // Normalize and combine different capability metrics
    float cpu_score = std::min(1.0f, static_cast<float>(capabilities.cpu_cores) / 32.0f);
    float memory_score = std::min(1.0f, capabilities.memory_gb / 64.0f);
    float gpu_score = capabilities.gpu_count > 0 ? 
                     std::min(1.0f, capabilities.gpu_memory_gb / 24.0f) : 0.0f;
    float network_score = std::min(1.0f, capabilities.network_bandwidth_mbps / 1000.0f);
    
    // Weighted combination (GPU gets higher weight for ML workloads)
    return cpu_score * 0.2f + memory_score * 0.2f + gpu_score * 0.5f + network_score * 0.1f;
}

float DynamicLoadBalancer::calculate_performance_score(const PeerPerformance& performance) const {
    // Convert performance metrics to a 0-1 score
    float throughput_score = std::min(1.0f, performance.samples_per_second / 1000.0f);
    float latency_score = std::max(0.0f, 1.0f - performance.average_latency_ms / 1000.0f);
    float memory_score = std::max(0.0f, 1.0f - performance.memory_usage_percent / 100.0f);
    float cpu_score = std::max(0.0f, 1.0f - performance.cpu_usage_percent / 100.0f);
    
    return throughput_score * 0.4f + latency_score * 0.2f + memory_score * 0.2f + cpu_score * 0.2f;
}

bool DynamicLoadBalancer::is_peer_healthy(const PeerInfo& peer) const {
    auto now = std::chrono::steady_clock::now();
    auto time_since_update = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - peer.last_update).count();
    
    return time_since_update < config_.peer_timeout_ms;
}

uint32_t DynamicLoadBalancer::adjust_for_peer_capabilities(const PeerInfo& peer, uint32_t suggested_batch) const {
    // Adjust batch size based on peer capabilities
    float capability_factor = calculate_capability_score(peer.capabilities);
    
    // Scale batch size by capability (but within bounds)
    uint32_t adjusted_batch = static_cast<uint32_t>(suggested_batch * capability_factor);
    adjusted_batch = std::max(config_.min_batch_size, adjusted_batch);
    adjusted_batch = std::min(config_.max_batch_size, adjusted_batch);
    
    return adjusted_batch;
}

void DynamicLoadBalancer::update_peer_load_estimate(PeerInfo& peer) {
    if (peer.recent_performance.empty()) return;
    
    // Update load estimate based on recent performance
    const auto& latest_perf = peer.recent_performance.back().second;
    
    peer.current_load.cpu_usage_percent = latest_perf.cpu_usage_percent;
    peer.current_load.memory_usage_percent = latest_perf.memory_usage_percent;
    peer.current_load.gpu_usage_percent = latest_perf.gpu_usage_percent;
    peer.current_load.network_usage_percent = 
        (latest_perf.network_bytes_per_sec / 1024.0f / 1024.0f) / peer.capabilities.network_bandwidth_mbps * 100.0f;
}

float DynamicLoadBalancer::calculate_performance_trend(const PeerInfo& peer) const {
    if (peer.recent_performance.size() < 2) return 0.0f;
    
    // Simple linear regression to calculate trend
    size_t n = peer.recent_performance.size();
    float sum_x = 0.0f, sum_y = 0.0f, sum_xy = 0.0f, sum_x2 = 0.0f;
    
    for (size_t i = 0; i < n; ++i) {
        float x = static_cast<float>(i);
        float y = calculate_performance_score(peer.recent_performance[i].second);
        
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_x2 += x * x;
    }
    
    float slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    return slope;
}

std::string DynamicLoadBalancer::find_best_migration_target(
    const std::string& source_peer,
    const std::vector<std::string>& candidate_peers) const {
    
    float best_score = -1.0f;
    std::string best_peer;
    
    for (const auto& candidate : candidate_peers) {
        auto it = peer_info_.find(candidate);
        if (it != peer_info_.end()) {
            float score = calculate_peer_performance_score(it->second);
            if (score > best_score) {
                best_score = score;
                best_peer = candidate;
            }
        }
    }
    
    return best_peer;
}

void DynamicLoadBalancer::create_migration_plan(const std::string& source_peer, const std::string& target_peer) {
    std::lock_guard<std::mutex> lock(migration_mutex_);
    
    WorkloadMigration migration;
    migration.migration_id = generate_migration_id();
    migration.source_peer_id = source_peer;
    migration.target_peer_id = target_peer;
    migration.created_time = std::chrono::steady_clock::now();
    migration.status = MigrationStatus::PENDING;
    
    // Calculate workload to migrate (e.g., 25% of current load)
    auto source_it = peer_info_.find(source_peer);
    if (source_it != peer_info_.end()) {
        migration.workload_size = source_it->second.current_load.assigned_batch_size / 4;
    }
    
    pending_migrations_.push_back(migration);
    
    std::cout << "Created migration plan: " << migration.migration_id 
             << " (" << source_peer << " -> " << target_peer 
             << ", workload: " << migration.workload_size << ")" << std::endl;
}

void DynamicLoadBalancer::execute_migration(WorkloadMigration& migration) {
    std::cout << "Executing migration: " << migration.migration_id << std::endl;
    
    // In a real implementation, this would coordinate with the P2P network
    // to actually move workload between peers
    
    // For now, just update the load assignments
    auto source_it = peer_info_.find(migration.source_peer_id);
    auto target_it = peer_info_.find(migration.target_peer_id);
    
    if (source_it != peer_info_.end() && target_it != peer_info_.end()) {
        uint32_t migrated_workload = std::min(migration.workload_size,
                                             source_it->second.current_load.assigned_batch_size);
        
        source_it->second.current_load.assigned_batch_size -= migrated_workload;
        target_it->second.current_load.assigned_batch_size += migrated_workload;
        
        migration.status = MigrationStatus::COMPLETED;
        migration.completed_time = std::chrono::steady_clock::now();
        
        std::cout << "Migration completed: " << migration.migration_id 
                 << " (migrated: " << migrated_workload << ")" << std::endl;
        
        // Update statistics
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        balancing_stats_.total_migrations++;
        balancing_stats_.successful_migrations++;
    } else {
        migration.status = MigrationStatus::FAILED;
        migration.error_message = "Source or target peer not found";
        
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        balancing_stats_.failed_migrations++;
    }
}

std::string DynamicLoadBalancer::generate_migration_id() {
    static std::atomic<uint64_t> counter{0};
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    return "migration_" + std::to_string(timestamp) + "_" + std::to_string(counter.fetch_add(1));
}

void DynamicLoadBalancer::update_balancing_statistics() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    // Update statistics based on current state
    balancing_stats_.active_peers = 0;
    balancing_stats_.total_workload = 0;
    
    {
        std::lock_guard<std::mutex> peers_lock(peers_mutex_);
        for (const auto& [peer_id, info] : peer_info_) {
            if (info.is_active) {
                balancing_stats_.active_peers++;
                balancing_stats_.total_workload += info.current_load.assigned_batch_size;
            }
        }
    }
}

} // namespace load_balancing
