#include "../include/network_partitions.hpp"
#include "../include/logger.hpp"
#include <algorithm>
#include <random>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <thread>
#include <queue>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#endif

namespace network_partitions {

NetworkPartitionHandler::NetworkPartitionHandler(const PartitionHandlingConfig& config)
    : config_(config), next_checkpoint_id_(1), is_partitioned_(false) {
    
    logger::log_info("Initializing Network Partition Handler");
    logger::log_info("- Strategy: " + std::to_string(static_cast<int>(config_.strategy)));
    logger::log_info("- Connectivity check interval: " + std::to_string(config_.connectivity_check_interval_ms) + "ms");
    logger::log_info("- Min nodes for training: " + std::to_string(config_.min_nodes_for_training));
    logger::log_info("- Automatic reconciliation: " + (config_.enable_automatic_reconciliation ? "enabled" : "disabled"));
    
    // Initialize statistics
    stats_ = {};
    stats_.last_partition_event = std::chrono::steady_clock::now();
    
#ifdef _WIN32
    // Initialize Winsock
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);
#endif
}

NetworkPartitionHandler::~NetworkPartitionHandler() {
    stop_connectivity_monitoring();
    
#ifdef _WIN32
    WSACleanup();
#endif
}

void NetworkPartitionHandler::register_node(const std::string& node_id, 
                                           const std::string& ip_address, uint16_t port) {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    NodeInfo node_info;
    node_info.node_id = node_id;
    node_info.ip_address = ip_address;
    node_info.port = port;
    node_info.last_seen = std::chrono::steady_clock::now();
    node_info.is_registered = true;
    
    registered_nodes_[node_id] = node_info;
    
    // Initialize connectivity info
    NodeConnectivity connectivity;
    connectivity.node_id = node_id;
    connectivity.last_connectivity_check = std::chrono::steady_clock::now();
    connectivity.consecutive_failures = 0;
    connectivity.is_isolated = false;
    connectivity.connection_quality_score = 1.0f;
    
    node_connectivity_[node_id] = connectivity;
    
    logger::log_info("Registered node: " + node_id + " at " + ip_address + ":" + std::to_string(port));
}

void NetworkPartitionHandler::unregister_node(const std::string& node_id) {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    auto it = registered_nodes_.find(node_id);
    if (it != registered_nodes_.end()) {
        it->second.is_registered = false;
        node_connectivity_.erase(node_id);
        
        logger::log_info("Unregistered node: " + node_id);
    }
}

std::vector<std::string> NetworkPartitionHandler::get_registered_nodes() const {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    std::vector<std::string> nodes;
    for (const auto& [node_id, node_info] : registered_nodes_) {
        if (node_info.is_registered) {
            nodes.push_back(node_id);
        }
    }
    
    return nodes;
}

void NetworkPartitionHandler::start_connectivity_monitoring() {
    if (monitoring_active_.load()) {
        logger::log_warning("Connectivity monitoring already active");
        return;
    }
    
    monitoring_active_ = true;
    
    connectivity_monitor_thread_ = std::thread(&NetworkPartitionHandler::run_connectivity_monitoring, this);
    partition_detector_thread_ = std::thread(&NetworkPartitionHandler::run_partition_detection, this);
    
    logger::log_info("Started connectivity monitoring");
}

void NetworkPartitionHandler::stop_connectivity_monitoring() {
    if (!monitoring_active_.load()) {
        return;
    }
    
    monitoring_active_ = false;
    
    if (connectivity_monitor_thread_.joinable()) {
        connectivity_monitor_thread_.join();
    }
    if (partition_detector_thread_.joinable()) {
        partition_detector_thread_.join();
    }
    
    logger::log_info("Stopped connectivity monitoring");
}

NodeConnectivity NetworkPartitionHandler::check_node_connectivity(const std::string& node_id) {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    auto node_it = registered_nodes_.find(node_id);
    if (node_it == registered_nodes_.end()) {
        NodeConnectivity empty_connectivity;
        empty_connectivity.node_id = node_id;
        empty_connectivity.is_isolated = true;
        empty_connectivity.connection_quality_score = 0.0f;
        return empty_connectivity;
    }
    
    const NodeInfo& node_info = node_it->second;
    NodeConnectivity& connectivity = node_connectivity_[node_id];
    
    // Test connectivity to all other registered nodes
    connectivity.reachable_nodes.clear();
    connectivity.unreachable_nodes.clear();
    
    for (const auto& [other_node_id, other_node_info] : registered_nodes_) {
        if (other_node_id == node_id || !other_node_info.is_registered) {
            continue;
        }
        
        bool is_reachable = ping_node(other_node_id, config_.ping_timeout_ms);
        
        if (is_reachable) {
            connectivity.reachable_nodes.insert(other_node_id);
        } else {
            connectivity.unreachable_nodes.insert(other_node_id);
        }
    }
    
    // Update connectivity metrics
    uint32_t total_peers = connectivity.reachable_nodes.size() + connectivity.unreachable_nodes.size();
    if (total_peers > 0) {
        connectivity.connection_quality_score = static_cast<float>(connectivity.reachable_nodes.size()) / total_peers;
    }
    
    // Update isolation status
    connectivity.is_isolated = connectivity.reachable_nodes.empty() && !connectivity.unreachable_nodes.empty();
    connectivity.last_connectivity_check = std::chrono::steady_clock::now();
    
    logger::log_debug("Connectivity check for " + node_id + ": " + 
                     std::to_string(connectivity.reachable_nodes.size()) + " reachable, " +
                     std::to_string(connectivity.unreachable_nodes.size()) + " unreachable");
    
    return connectivity;
}

NetworkStatus NetworkPartitionHandler::get_network_status() const {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    NetworkStatus status;
    status.total_nodes = 0;
    status.reachable_nodes = 0;
    status.unreachable_nodes = 0;
    status.is_partitioned = is_partitioned_;
    
    float total_connectivity = 0.0f;
    uint32_t connectivity_samples = 0;
    
    for (const auto& [node_id, node_info] : registered_nodes_) {
        if (!node_info.is_registered) continue;
        
        status.total_nodes++;
        
        auto connectivity_it = node_connectivity_.find(node_id);
        if (connectivity_it != node_connectivity_.end()) {
            const NodeConnectivity& connectivity = connectivity_it->second;
            
            if (connectivity.is_isolated) {
                status.unreachable_nodes++;
            } else {
                status.reachable_nodes++;
            }
            
            total_connectivity += connectivity.connection_quality_score;
            connectivity_samples++;
        }
    }
    
    if (connectivity_samples > 0) {
        status.overall_connectivity_score = total_connectivity / connectivity_samples;
    }
    
    {
        std::lock_guard<std::mutex> partitions_lock(partitions_mutex_);
        status.active_partitions = static_cast<uint32_t>(current_partitions_.size());
        
        if (!current_partitions_.empty()) {
            // Find largest partition
            auto largest_partition = std::max_element(current_partitions_.begin(), current_partitions_.end(),
                [](const PartitionInfo& a, const PartitionInfo& b) {
                    return a.partition_size < b.partition_size;
                });
            status.largest_partition_id = largest_partition->partition_id;
        }
    }
    
    status.last_status_update = std::chrono::steady_clock::now();
    
    return status;
}

std::vector<PartitionInfo> NetworkPartitionHandler::detect_network_partitions() {
    logger::log_debug("Detecting network partitions");
    
    // Build connectivity graph
    std::unordered_map<std::string, std::unordered_set<std::string>> adjacency_list;
    
    {
        std::lock_guard<std::mutex> lock(nodes_mutex_);
        
        for (const auto& [node_id, connectivity] : node_connectivity_) {
            adjacency_list[node_id] = connectivity.reachable_nodes;
            // Add self to adjacency list
            adjacency_list[node_id].insert(node_id);
        }
    }
    
    // Find connected components
    std::vector<std::vector<std::string>> components = utils::find_connected_components_graph(adjacency_list);
    
    std::vector<PartitionInfo> partitions;
    
    for (const auto& component : components) {
        if (component.empty()) continue;
        
        PartitionInfo partition = create_partition_from_nodes(component);
        partitions.push_back(partition);
    }
    
    // Update partition state
    {
        std::lock_guard<std::mutex> lock(partitions_mutex_);
        
        bool was_partitioned = is_partitioned_;
        is_partitioned_ = partitions.size() > 1;
        
        // Detect new partitions
        if (!was_partitioned && is_partitioned_) {
            logger::log_warning("Network partition detected! " + std::to_string(partitions.size()) + " partitions found");
            
            {
                std::lock_guard<std::mutex> stats_lock(stats_mutex_);
                stats_.total_partitions_detected++;
                stats_.last_partition_event = std::chrono::steady_clock::now();
            }
            
            // Trigger callbacks
            for (const auto& partition : partitions) {
                if (partition_detected_callback_) {
                    partition_detected_callback_(partition);
                }
            }
        }
        
        current_partitions_ = partitions;
        
        // Find our partition
        std::string our_node_id = "current_node"; // TODO: Get actual node ID
        for (const auto& partition : partitions) {
            if (partition.nodes_in_partition.find(our_node_id) != partition.nodes_in_partition.end()) {
                my_partition_id_ = partition.partition_id;
                break;
            }
        }
    }
    
    return partitions;
}

NetworkPartitionHandler::TrainingDecision NetworkPartitionHandler::make_training_decision(const std::string& requesting_node_id) {
    PartitionInfo current_partition = get_current_partition(requesting_node_id);
    
    if (current_partition.partition_id.empty()) {
        // Node not in any known partition, create default decision
        TrainingDecision decision;
        decision.should_continue_training = false;
        decision.quality_degradation_factor = 1.0f;
        decision.reasoning = "Node not found in any partition";
        decision.estimated_performance_impact_percent = 100;
        return decision;
    }
    
    switch (config_.strategy) {
        case PartitionStrategy::DEGRADED_TRAINING:
            return make_degraded_training_decision(current_partition);
            
        case PartitionStrategy::MAJORITY_CONTINUES:
            return make_majority_only_decision(current_partition);
            
        case PartitionStrategy::PAUSE_AND_WAIT:
            return make_pause_and_wait_decision(current_partition);
            
        case PartitionStrategy::CHECKPOINT_AND_RESTART:
            return make_checkpoint_restart_decision(current_partition);
            
        default:
            return make_degraded_training_decision(current_partition);
    }
}

NetworkPartitionHandler::TrainingDecision NetworkPartitionHandler::make_degraded_training_decision(const PartitionInfo& partition) {
    TrainingDecision decision;
    
    // Allow training if we have minimum nodes
    decision.should_continue_training = (partition.partition_size >= config_.min_nodes_for_training) ||
                                       (config_.allow_single_node_training && partition.partition_size >= 1);
    
    if (decision.should_continue_training) {
        // Calculate quality degradation based on partition size
        float original_nodes = static_cast<float>(get_registered_nodes().size());
        float current_nodes = static_cast<float>(partition.partition_size);
        
        // Quality degrades with fewer nodes
        float size_factor = current_nodes / original_nodes;
        decision.quality_degradation_factor = size_factor;
        
        // Additional penalty for minority partitions
        if (!partition.is_majority_partition) {
            decision.quality_degradation_factor *= (1.0f - config_.minority_partition_quality_penalty);
        }
        
        decision.estimated_performance_impact_percent = static_cast<uint32_t>((1.0f - size_factor) * 100);
        
        decision.reasoning = "Training continues with " + std::to_string(partition.partition_size) + 
                            " nodes (quality factor: " + std::to_string(decision.quality_degradation_factor) + ")";
        
        decision.participating_nodes.assign(partition.nodes_in_partition.begin(), partition.nodes_in_partition.end());
    } else {
        decision.quality_degradation_factor = 0.0f;
        decision.estimated_performance_impact_percent = 100;
        decision.reasoning = "Insufficient nodes for training (have " + std::to_string(partition.partition_size) + 
                            ", need " + std::to_string(config_.min_nodes_for_training) + ")";
    }
    
    return decision;
}

NetworkPartitionHandler::TrainingDecision NetworkPartitionHandler::make_majority_only_decision(const PartitionInfo& partition) {
    TrainingDecision decision;
    
    decision.should_continue_training = partition.is_majority_partition && 
                                       (partition.partition_size >= config_.min_nodes_for_training);
    
    if (decision.should_continue_training) {
        // Majority partition continues with minimal degradation
        decision.quality_degradation_factor = 0.9f; // 10% degradation due to missing minority
        decision.estimated_performance_impact_percent = 10;
        decision.reasoning = "Training continues - majority partition (" + 
                            std::to_string(partition.partition_size) + " nodes)";
        decision.participating_nodes.assign(partition.nodes_in_partition.begin(), partition.nodes_in_partition.end());
    } else {
        decision.quality_degradation_factor = 0.0f;
        decision.estimated_performance_impact_percent = 100;
        decision.reasoning = partition.is_majority_partition ? 
                            "Insufficient nodes in majority partition" : 
                            "Training paused - minority partition";
    }
    
    return decision;
}

NetworkPartitionHandler::TrainingDecision NetworkPartitionHandler::make_pause_and_wait_decision(const PartitionInfo& partition) {
    TrainingDecision decision;
    
    // Always pause training during partitions
    decision.should_continue_training = false;
    decision.quality_degradation_factor = 0.0f;
    decision.estimated_performance_impact_percent = 100;
    decision.reasoning = "Training paused due to network partition - waiting for network healing";
    
    return decision;
}

NetworkPartitionHandler::TrainingDecision NetworkPartitionHandler::make_checkpoint_restart_decision(const PartitionInfo& partition) {
    TrainingDecision decision;
    
    // Create checkpoint and then pause
    force_checkpoint_creation("partition_detected");
    
    decision.should_continue_training = false;
    decision.quality_degradation_factor = 0.0f;
    decision.estimated_performance_impact_percent = 100;
    decision.reasoning = "Checkpoint created, training paused until partition heals";
    
    return decision;
}

std::string NetworkPartitionHandler::create_partition_checkpoint(const std::vector<float>& model_parameters,
                                                               uint64_t global_step,
                                                               const std::string& partition_id) {
    std::lock_guard<std::mutex> lock(checkpoints_mutex_);
    
    CheckpointMetadata checkpoint;
    checkpoint.checkpoint_id = generate_checkpoint_id();
    checkpoint.model_parameters = model_parameters;
    checkpoint.partition_id = partition_id;
    checkpoint.global_step = global_step;
    checkpoint.validation_loss = 0.0f; // TODO: Get actual validation loss
    checkpoint.creation_time = std::chrono::steady_clock::now();
    checkpoint.is_reconciliation_checkpoint = false;
    
    // Determine contributing nodes
    auto partition_info = get_current_partition("");
    if (!partition_info.partition_id.empty()) {
        checkpoint.contributing_nodes = partition_info.nodes_in_partition;
        checkpoint.nodes_contributing = static_cast<uint32_t>(partition_info.nodes_in_partition.size());
    }
    
    checkpoints_[checkpoint.checkpoint_id] = checkpoint;
    
    logger::log_info("Created partition checkpoint: " + checkpoint.checkpoint_id + 
                     " (step " + std::to_string(global_step) + ", " + 
                     std::to_string(checkpoint.nodes_contributing) + " nodes)");
    
    // Cleanup old checkpoints
    cleanup_old_checkpoints();
    
    {
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        stats_.total_checkpoints_created++;
    }
    
    return checkpoint.checkpoint_id;
}

bool NetworkPartitionHandler::detect_partition_healing() {
    NetworkStatus status = get_network_status();
    
    // Simple heuristic: healing detected if connectivity is high and no active partitions
    bool is_healed = status.overall_connectivity_score > 0.9f && 
                     status.active_partitions <= 1 && 
                     was_recently_partitioned();
    
    if (is_healed && is_partitioned_) {
        logger::log_info("Network partition healing detected!");
        
        {
            std::lock_guard<std::mutex> lock(partitions_mutex_);
            is_partitioned_ = false;
            
            std::vector<std::string> all_nodes;
            for (const auto& partition : current_partitions_) {
                all_nodes.insert(all_nodes.end(), 
                               partition.nodes_in_partition.begin(), 
                               partition.nodes_in_partition.end());
            }
            
            current_partitions_.clear();
            my_partition_id_.clear();
            
            // Trigger healing callback
            if (partition_healed_callback_) {
                partition_healed_callback_(all_nodes);
            }
        }
        
        // Start reconciliation if enabled
        if (config_.enable_automatic_reconciliation) {
            // Note: In a real implementation, this would trigger reconciliation
            logger::log_info("Starting automatic reconciliation process");
        }
        
        return true;
    }
    
    return false;
}

// Helper methods
bool NetworkPartitionHandler::ping_node(const std::string& node_id, uint32_t timeout_ms) {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    auto it = registered_nodes_.find(node_id);
    if (it == registered_nodes_.end()) {
        return false;
    }
    
    const NodeInfo& node_info = it->second;
    return utils::test_tcp_connection(node_info.ip_address, node_info.port, timeout_ms);
}

void NetworkPartitionHandler::run_connectivity_monitoring() {
    logger::log_info("Started connectivity monitoring thread");
    
    while (monitoring_active_.load()) {
        try {
            std::vector<std::string> nodes = get_registered_nodes();
            
            for (const auto& node_id : nodes) {
                update_node_connectivity(node_id);
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(config_.connectivity_check_interval_ms));
            
        } catch (const std::exception& e) {
            logger::log_error("Error in connectivity monitoring: " + std::string(e.what()));
        }
    }
    
    logger::log_info("Connectivity monitoring thread stopped");
}

void NetworkPartitionHandler::run_partition_detection() {
    logger::log_info("Started partition detection thread");
    
    while (monitoring_active_.load()) {
        try {
            detect_network_partitions();
            
            // Check for partition healing
            if (is_partitioned_) {
                detect_partition_healing();
            }
            
            // Regular maintenance
            cleanup_old_checkpoints();
            update_statistics();
            
            std::this_thread::sleep_for(std::chrono::milliseconds(config_.partition_stability_window_ms));
            
        } catch (const std::exception& e) {
            logger::log_error("Error in partition detection: " + std::string(e.what()));
        }
    }
    
    logger::log_info("Partition detection thread stopped");
}

void NetworkPartitionHandler::update_node_connectivity(const std::string& node_id) {
    NodeConnectivity connectivity = check_node_connectivity(node_id);
    
    // Update failure count
    if (connectivity.connection_quality_score < config_.partition_detection_threshold) {
        connectivity.consecutive_failures++;
    } else {
        connectivity.consecutive_failures = 0;
    }
    
    // Update isolation status
    if (connectivity.consecutive_failures >= config_.max_consecutive_failures) {
        connectivity.is_isolated = true;
    }
    
    {
        std::lock_guard<std::mutex> lock(nodes_mutex_);
        node_connectivity_[node_id] = connectivity;
    }
}

PartitionInfo NetworkPartitionHandler::create_partition_from_nodes(const std::vector<std::string>& nodes) {
    PartitionInfo partition;
    partition.partition_id = generate_partition_id();
    partition.nodes_in_partition = std::unordered_set<std::string>(nodes.begin(), nodes.end());
    partition.partition_size = static_cast<uint32_t>(nodes.size());
    partition.is_majority_partition = is_majority_partition(partition);
    partition.partition_leader = select_partition_leader(nodes);
    partition.is_training_active = false; // Will be determined by training decision
    partition.training_quality_degradation = 0.0f;
    partition.partition_detected_time = std::chrono::steady_clock::now();
    partition.last_model_checkpoint_time = std::chrono::steady_clock::now();
    
    return partition;
}

bool NetworkPartitionHandler::is_majority_partition(const PartitionInfo& partition) const {
    uint32_t total_registered_nodes = static_cast<uint32_t>(get_registered_nodes().size());
    return partition.partition_size > (total_registered_nodes / 2);
}

std::string NetworkPartitionHandler::select_partition_leader(const std::vector<std::string>& nodes) {
    if (nodes.empty()) {
        return "";
    }
    
    // Simple leader selection: lexicographically smallest node ID
    return *std::min_element(nodes.begin(), nodes.end());
}

PartitionInfo NetworkPartitionHandler::get_current_partition(const std::string& node_id) const {
    std::lock_guard<std::mutex> lock(partitions_mutex_);
    
    for (const auto& partition : current_partitions_) {
        if (partition.nodes_in_partition.find(node_id) != partition.nodes_in_partition.end()) {
            return partition;
        }
    }
    
    return PartitionInfo{}; // Empty partition if not found
}

std::string NetworkPartitionHandler::generate_partition_id() {
    static uint64_t counter = 1;
    std::stringstream ss;
    ss << "partition_" << std::hex << counter++;
    return ss.str();
}

std::string NetworkPartitionHandler::generate_checkpoint_id() {
    std::stringstream ss;
    ss << "checkpoint_" << std::hex << next_checkpoint_id_++;
    return ss.str();
}

void NetworkPartitionHandler::cleanup_old_checkpoints() {
    std::lock_guard<std::mutex> lock(checkpoints_mutex_);
    
    // Remove checkpoints older than 24 hours
    auto cutoff_time = std::chrono::steady_clock::now() - std::chrono::hours(24);
    
    auto it = checkpoints_.begin();
    while (it != checkpoints_.end()) {
        if (it->second.creation_time < cutoff_time) {
            logger::log_debug("Removing old checkpoint: " + it->first);
            it = checkpoints_.erase(it);
        } else {
            ++it;
        }
    }
}

bool NetworkPartitionHandler::was_recently_partitioned() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    auto time_since_partition = std::chrono::steady_clock::now() - stats_.last_partition_event;
    return std::chrono::duration_cast<std::chrono::minutes>(time_since_partition).count() < 60;
}

void NetworkPartitionHandler::update_statistics() {
    // Update various statistics
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    // Calculate average partition duration if we have partition events
    // TODO: Track partition durations properly
    
    stats_.last_partition_event = std::chrono::steady_clock::now();
}

// Utility function implementations
namespace utils {

bool test_tcp_connection(const std::string& ip_address, uint16_t port, uint32_t timeout_ms) {
#ifdef _WIN32
    SOCKET sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock == INVALID_SOCKET) {
        return false;
    }
    
    // Set non-blocking mode
    u_long mode = 1;
    ioctlsocket(sock, FIONBIO, &mode);
    
    sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    inet_pton(AF_INET, ip_address.c_str(), &addr.sin_addr);
    
    int result = connect(sock, (sockaddr*)&addr, sizeof(addr));
    if (result == SOCKET_ERROR) {
        if (WSAGetLastError() == WSAEWOULDBLOCK) {
            // Connection in progress
            fd_set write_set;
            FD_ZERO(&write_set);
            FD_SET(sock, &write_set);
            
            timeval tv;
            tv.tv_sec = timeout_ms / 1000;
            tv.tv_usec = (timeout_ms % 1000) * 1000;
            
            result = select(0, nullptr, &write_set, nullptr, &tv);
            if (result > 0) {
                // Check for connection errors
                int error;
                int error_len = sizeof(error);
                getsockopt(sock, SOL_SOCKET, SO_ERROR, (char*)&error, &error_len);
                closesocket(sock);
                return error == 0;
            }
        }
    }
    
    closesocket(sock);
    return result == 0;
    
#else
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        return false;
    }
    
    // Set non-blocking mode
    int flags = fcntl(sock, F_GETFL, 0);
    fcntl(sock, F_SETFL, flags | O_NONBLOCK);
    
    sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    inet_pton(AF_INET, ip_address.c_str(), &addr.sin_addr);
    
    int result = connect(sock, (sockaddr*)&addr, sizeof(addr));
    if (result < 0) {
        if (errno == EINPROGRESS) {
            fd_set write_set;
            FD_ZERO(&write_set);
            FD_SET(sock, &write_set);
            
            timeval tv;
            tv.tv_sec = timeout_ms / 1000;
            tv.tv_usec = (timeout_ms % 1000) * 1000;
            
            result = select(sock + 1, nullptr, &write_set, nullptr, &tv);
            if (result > 0) {
                int error;
                socklen_t error_len = sizeof(error);
                getsockopt(sock, SOL_SOCKET, SO_ERROR, &error, &error_len);
                close(sock);
                return error == 0;
            }
        }
    }
    
    close(sock);
    return result == 0;
#endif
}

std::vector<std::vector<std::string>> find_connected_components_graph(
    const std::unordered_map<std::string, std::unordered_set<std::string>>& adjacency_list) {
    
    std::unordered_set<std::string> visited;
    std::vector<std::vector<std::string>> components;
    
    for (const auto& [node, neighbors] : adjacency_list) {
        if (visited.find(node) != visited.end()) {
            continue;
        }
        
        // DFS to find connected component
        std::vector<std::string> component;
        std::queue<std::string> queue;
        
        queue.push(node);
        visited.insert(node);
        
        while (!queue.empty()) {
            std::string current = queue.front();
            queue.pop();
            component.push_back(current);
            
            auto current_neighbors_it = adjacency_list.find(current);
            if (current_neighbors_it != adjacency_list.end()) {
                for (const std::string& neighbor : current_neighbors_it->second) {
                    if (visited.find(neighbor) == visited.end()) {
                        visited.insert(neighbor);
                        queue.push(neighbor);
                    }
                }
            }
        }
        
        if (!component.empty()) {
            components.push_back(component);
        }
    }
    
    return components;
}

} // namespace utils
} // namespace network_partitions
