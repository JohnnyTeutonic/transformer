#include "../include/partition_recovery.hpp"
#include "../include/serialization.hpp"
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <openssl/sha.h>

namespace p2p {

PartitionRecoveryManager::PartitionRecoveryManager(
    std::shared_ptr<P2PNetwork> network,
    const RecoveryConfig& config)
    : network_(network), config_(config), recovery_active_(false) {
    
    std::cout << "PartitionRecoveryManager initialized with:" << std::endl;
    std::cout << "- Heartbeat interval: " << config_.heartbeat_interval_ms << "ms" << std::endl;
    std::cout << "- Partition timeout: " << config_.partition_timeout_ms << "ms" << std::endl;
    std::cout << "- Recovery timeout: " << config_.recovery_timeout_ms << "ms" << std::endl;
    std::cout << "- Merkle tree depth: " << config_.merkle_tree_depth << std::endl;
}

PartitionRecoveryManager::~PartitionRecoveryManager() {
    stop_recovery_monitoring();
}

bool PartitionRecoveryManager::start_recovery_monitoring() {
    if (recovery_active_.load()) {
        std::cout << "Partition recovery monitoring already active" << std::endl;
        return true;
    }
    
    std::cout << "Starting partition recovery monitoring..." << std::endl;
    
    recovery_active_.store(true);
    
    // Start monitoring threads
    heartbeat_thread_ = std::thread(&PartitionRecoveryManager::heartbeat_monitor_thread, this);
    partition_detector_thread_ = std::thread(&PartitionRecoveryManager::partition_detection_thread, this);
    recovery_thread_ = std::thread(&PartitionRecoveryManager::recovery_coordination_thread, this);
    
    std::cout << "Partition recovery monitoring started" << std::endl;
    return true;
}

void PartitionRecoveryManager::stop_recovery_monitoring() {
    if (!recovery_active_.load()) {
        return;
    }
    
    std::cout << "Stopping partition recovery monitoring..." << std::endl;
    
    recovery_active_.store(false);
    
    // Notify all condition variables
    heartbeat_cv_.notify_all();
    partition_cv_.notify_all();
    recovery_cv_.notify_all();
    
    // Join threads
    if (heartbeat_thread_.joinable()) {
        heartbeat_thread_.join();
    }
    if (partition_detector_thread_.joinable()) {
        partition_detector_thread_.join();
    }
    if (recovery_thread_.joinable()) {
        recovery_thread_.join();
    }
    
    std::cout << "Partition recovery monitoring stopped" << std::endl;
}

void PartitionRecoveryManager::register_model_state_provider(
    std::function<ModelStateSnapshot()> provider) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    model_state_provider_ = provider;
}

void PartitionRecoveryManager::register_model_state_applier(
    std::function<bool(const ModelStateSnapshot&)> applier) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    model_state_applier_ = applier;
}

bool PartitionRecoveryManager::handle_partition_detected(const std::vector<std::string>& lost_peers) {
    std::lock_guard<std::mutex> lock(partition_mutex_);
    
    std::cout << "Partition detected! Lost peers: ";
    for (const auto& peer : lost_peers) {
        std::cout << peer << " ";
    }
    std::cout << std::endl;
    
    // Record partition event
    PartitionEvent event;
    event.timestamp = std::chrono::steady_clock::now();
    event.lost_peers = lost_peers;
    event.partition_id = generate_partition_id();
    
    active_partitions_[event.partition_id] = event;
    
    // Update network state
    current_partition_state_.is_partitioned = true;
    current_partition_state_.partition_start_time = event.timestamp;
    current_partition_state_.lost_peers = lost_peers;
    
    // Notify recovery thread
    partition_cv_.notify_one();
    
    return true;
}

bool PartitionRecoveryManager::handle_partition_healed(const std::vector<std::string>& recovered_peers) {
    std::lock_guard<std::mutex> lock(partition_mutex_);
    
    std::cout << "Partition healed! Recovered peers: ";
    for (const auto& peer : recovered_peers) {
        std::cout << peer << " ";
    }
    std::cout << std::endl;
    
    // Start recovery process
    RecoverySession session;
    session.session_id = generate_recovery_session_id();
    session.recovered_peers = recovered_peers;
    session.start_time = std::chrono::steady_clock::now();
    session.status = RecoveryStatus::INITIATED;
    
    active_recovery_sessions_[session.session_id] = session;
    
    // Clear partition state
    current_partition_state_.is_partitioned = false;
    current_partition_state_.lost_peers.clear();
    
    // Notify recovery thread
    recovery_cv_.notify_one();
    
    return initiate_state_reconciliation(recovered_peers);
}

void PartitionRecoveryManager::heartbeat_monitor_thread() {
    std::cout << "Heartbeat monitor thread started" << std::endl;
    
    while (recovery_active_.load()) {
        auto start_time = std::chrono::steady_clock::now();
        
        // Send heartbeat to all known peers
        send_heartbeat_to_peers();
        
        // Check for missed heartbeats
        check_missed_heartbeats();
        
        // Sleep until next heartbeat interval
        std::unique_lock<std::mutex> lock(heartbeat_mutex_);
        heartbeat_cv_.wait_for(lock, std::chrono::milliseconds(config_.heartbeat_interval_ms),
                              [this] { return !recovery_active_.load(); });
    }
    
    std::cout << "Heartbeat monitor thread stopped" << std::endl;
}

void PartitionRecoveryManager::partition_detection_thread() {
    std::cout << "Partition detection thread started" << std::endl;
    
    while (recovery_active_.load()) {
        auto start_time = std::chrono::steady_clock::now();
        
        // Analyze peer connectivity patterns
        analyze_connectivity_patterns();
        
        // Detect potential partitions
        detect_network_partitions();
        
        // Sleep until next detection cycle
        std::unique_lock<std::mutex> lock(partition_mutex_);
        partition_cv_.wait_for(lock, std::chrono::milliseconds(config_.partition_detection_interval_ms),
                              [this] { return !recovery_active_.load(); });
    }
    
    std::cout << "Partition detection thread stopped" << std::endl;
}

void PartitionRecoveryManager::recovery_coordination_thread() {
    std::cout << "Recovery coordination thread started" << std::endl;
    
    while (recovery_active_.load()) {
        // Process active recovery sessions
        process_recovery_sessions();
        
        // Clean up completed sessions
        cleanup_completed_sessions();
        
        // Sleep until next coordination cycle
        std::unique_lock<std::mutex> lock(recovery_mutex_);
        recovery_cv_.wait_for(lock, std::chrono::milliseconds(config_.recovery_check_interval_ms),
                             [this] { return !recovery_active_.load(); });
    }
    
    std::cout << "Recovery coordination thread stopped" << std::endl;
}

void PartitionRecoveryManager::send_heartbeat_to_peers() {
    if (!network_) return;
    
    auto peers = network_->get_connected_peers();
    if (peers.empty()) return;
    
    // Create heartbeat message
    HeartbeatMessage heartbeat;
    heartbeat.sender_id = network_->get_node_id();
    heartbeat.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    heartbeat.sequence_number = heartbeat_sequence_.fetch_add(1);
    
    // Add current model state hash for consistency checking
    if (model_state_provider_) {
        auto state = model_state_provider_();
        heartbeat.model_state_hash = calculate_model_state_hash(state);
    }
    
    // Serialize and send to all peers
    auto serialized = serialize_heartbeat(heartbeat);
    
    for (const auto& peer : peers) {
        network_->send_message_to_peer(peer, MessageType::HEARTBEAT, serialized);
    }
    
    // Update our own heartbeat record
    std::lock_guard<std::mutex> lock(heartbeat_mutex_);
    peer_heartbeats_[heartbeat.sender_id] = {
        std::chrono::steady_clock::now(),
        heartbeat.sequence_number,
        true  // is_alive
    };
}

void PartitionRecoveryManager::check_missed_heartbeats() {
    auto now = std::chrono::steady_clock::now();
    std::vector<std::string> lost_peers;
    
    {
        std::lock_guard<std::mutex> lock(heartbeat_mutex_);
        
        for (auto& [peer_id, heartbeat_info] : peer_heartbeats_) {
            auto time_since_last = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - heartbeat_info.last_seen).count();
            
            if (time_since_last > config_.partition_timeout_ms && heartbeat_info.is_alive) {
                heartbeat_info.is_alive = false;
                lost_peers.push_back(peer_id);
                
                std::cout << "Peer " << peer_id << " missed heartbeat (last seen " 
                         << time_since_last << "ms ago)" << std::endl;
            }
        }
    }
    
    if (!lost_peers.empty()) {
        handle_partition_detected(lost_peers);
    }
}

void PartitionRecoveryManager::analyze_connectivity_patterns() {
    // Analyze peer connectivity matrix to detect partition patterns
    auto peers = network_->get_connected_peers();
    if (peers.size() < 2) return;
    
    std::lock_guard<std::mutex> lock(connectivity_mutex_);
    
    // Build connectivity matrix
    ConnectivityMatrix matrix;
    matrix.timestamp = std::chrono::steady_clock::now();
    matrix.peer_ids = peers;
    
    // Initialize matrix
    size_t n = peers.size();
    matrix.connectivity.resize(n, std::vector<bool>(n, false));
    
    // Fill diagonal (self-connectivity)
    for (size_t i = 0; i < n; ++i) {
        matrix.connectivity[i][i] = true;
    }
    
    // Query connectivity between peers (simplified - in real implementation,
    // this would involve asking peers about their connections)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            // For now, assume all connected peers can see each other
            // In a real implementation, we'd query peer connectivity
            matrix.connectivity[i][j] = true;
            matrix.connectivity[j][i] = true;
        }
    }
    
    connectivity_history_.push_back(matrix);
    
    // Keep only recent history
    if (connectivity_history_.size() > 100) {
        connectivity_history_.erase(connectivity_history_.begin());
    }
}

void PartitionRecoveryManager::detect_network_partitions() {
    std::lock_guard<std::mutex> lock(connectivity_mutex_);
    
    if (connectivity_history_.empty()) return;
    
    const auto& latest = connectivity_history_.back();
    
    // Use Union-Find to detect connected components
    std::vector<int> parent(latest.peer_ids.size());
    std::iota(parent.begin(), parent.end(), 0);
    
    std::function<int(int)> find = [&](int x) {
        return parent[x] == x ? x : parent[x] = find(parent[x]);
    };
    
    auto unite = [&](int x, int y) {
        x = find(x);
        y = find(y);
        if (x != y) parent[x] = y;
    };
    
    // Unite connected peers
    for (size_t i = 0; i < latest.connectivity.size(); ++i) {
        for (size_t j = i + 1; j < latest.connectivity[i].size(); ++j) {
            if (latest.connectivity[i][j]) {
                unite(static_cast<int>(i), static_cast<int>(j));
            }
        }
    }
    
    // Group peers by connected component
    std::unordered_map<int, std::vector<std::string>> components;
    for (size_t i = 0; i < latest.peer_ids.size(); ++i) {
        components[find(static_cast<int>(i))].push_back(latest.peer_ids[i]);
    }
    
    // If we have multiple components, we have a partition
    if (components.size() > 1) {
        std::cout << "Network partition detected! " << components.size() 
                 << " separate components found" << std::endl;
        
        // Find the component we're not in (lost peers)
        std::string our_id = network_->get_node_id();
        std::vector<std::string> lost_peers;
        
        for (const auto& [component_id, peers] : components) {
            bool we_are_in_this_component = std::find(peers.begin(), peers.end(), our_id) != peers.end();
            if (!we_are_in_this_component) {
                lost_peers.insert(lost_peers.end(), peers.begin(), peers.end());
            }
        }
        
        if (!lost_peers.empty()) {
            handle_partition_detected(lost_peers);
        }
    }
}

bool PartitionRecoveryManager::initiate_state_reconciliation(const std::vector<std::string>& recovered_peers) {
    std::cout << "Initiating state reconciliation with recovered peers..." << std::endl;
    
    if (!model_state_provider_) {
        std::cerr << "No model state provider registered!" << std::endl;
        return false;
    }
    
    // Get our current model state
    auto our_state = model_state_provider_();
    auto our_merkle_root = build_merkle_tree(our_state);
    
    std::cout << "Our model state hash: " << bytes_to_hex(our_merkle_root) << std::endl;
    
    // Request state from recovered peers
    std::vector<StateReconciliationRequest> requests;
    
    for (const auto& peer : recovered_peers) {
        StateReconciliationRequest request;
        request.session_id = generate_recovery_session_id();
        request.requester_id = network_->get_node_id();
        request.target_peer_id = peer;
        request.our_merkle_root = our_merkle_root;
        request.timestamp = std::chrono::steady_clock::now();
        
        requests.push_back(request);
        
        // Send reconciliation request
        auto serialized = serialize_reconciliation_request(request);
        network_->send_message_to_peer(peer, MessageType::STATE_RECONCILIATION_REQUEST, serialized);
    }
    
    // Wait for responses and perform reconciliation
    return wait_for_reconciliation_responses(requests);
}

bool PartitionRecoveryManager::wait_for_reconciliation_responses(
    const std::vector<StateReconciliationRequest>& requests) {
    
    std::cout << "Waiting for reconciliation responses from " << requests.size() << " peers..." << std::endl;
    
    auto timeout = std::chrono::steady_clock::now() + 
                  std::chrono::milliseconds(config_.recovery_timeout_ms);
    
    std::unordered_map<std::string, ModelStateSnapshot> peer_states;
    std::unordered_set<std::string> responded_peers;
    
    // Wait for responses (simplified - in real implementation, this would be event-driven)
    while (std::chrono::steady_clock::now() < timeout && 
           responded_peers.size() < requests.size()) {
        
        // Check for new responses (this would be handled by message handlers in real implementation)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // For demonstration, simulate receiving responses
        for (const auto& request : requests) {
            if (responded_peers.find(request.target_peer_id) == responded_peers.end()) {
                // Simulate response (in real implementation, this would come from network)
                if (simulate_reconciliation_response(request, peer_states[request.target_peer_id])) {
                    responded_peers.insert(request.target_peer_id);
                    std::cout << "Received state from peer: " << request.target_peer_id << std::endl;
                }
            }
        }
    }
    
    if (responded_peers.size() < requests.size()) {
        std::cout << "Warning: Only received " << responded_peers.size() 
                 << " responses out of " << requests.size() << " requests" << std::endl;
    }
    
    // Perform state reconciliation
    return perform_state_reconciliation(peer_states);
}

bool PartitionRecoveryManager::perform_state_reconciliation(
    const std::unordered_map<std::string, ModelStateSnapshot>& peer_states) {
    
    if (peer_states.empty()) {
        std::cout << "No peer states received for reconciliation" << std::endl;
        return false;
    }
    
    std::cout << "Performing state reconciliation with " << peer_states.size() << " peer states..." << std::endl;
    
    // Get our current state
    auto our_state = model_state_provider_();
    
    // Find the most recent state based on training step
    const ModelStateSnapshot* most_recent_state = &our_state;
    uint64_t highest_step = our_state.training_step;
    
    for (const auto& [peer_id, state] : peer_states) {
        if (state.training_step > highest_step) {
            highest_step = state.training_step;
            most_recent_state = &state;
        }
    }
    
    // If we don't have the most recent state, apply it
    if (most_recent_state != &our_state) {
        std::cout << "Applying more recent state from training step " 
                 << most_recent_state->training_step 
                 << " (our step: " << our_state.training_step << ")" << std::endl;
        
        if (model_state_applier_) {
            bool success = model_state_applier_(*most_recent_state);
            if (success) {
                std::cout << "State reconciliation completed successfully" << std::endl;
                return true;
            } else {
                std::cerr << "Failed to apply reconciled state" << std::endl;
                return false;
            }
        } else {
            std::cerr << "No model state applier registered!" << std::endl;
            return false;
        }
    } else {
        std::cout << "Our state is already the most recent, no reconciliation needed" << std::endl;
        return true;
    }
}

std::vector<uint8_t> PartitionRecoveryManager::build_merkle_tree(const ModelStateSnapshot& state) {
    // Build Merkle tree for model state verification
    std::vector<std::vector<uint8_t>> leaf_hashes;
    
    // Hash each parameter matrix
    for (const auto& param : state.parameters) {
        std::vector<uint8_t> param_hash(SHA256_DIGEST_LENGTH);
        SHA256(reinterpret_cast<const unsigned char*>(param.data()), 
               param.size() * sizeof(float), param_hash.data());
        leaf_hashes.push_back(param_hash);
    }
    
    // Build tree bottom-up
    while (leaf_hashes.size() > 1) {
        std::vector<std::vector<uint8_t>> next_level;
        
        for (size_t i = 0; i < leaf_hashes.size(); i += 2) {
            std::vector<uint8_t> combined;
            combined.insert(combined.end(), leaf_hashes[i].begin(), leaf_hashes[i].end());
            
            if (i + 1 < leaf_hashes.size()) {
                combined.insert(combined.end(), leaf_hashes[i + 1].begin(), leaf_hashes[i + 1].end());
            }
            
            std::vector<uint8_t> parent_hash(SHA256_DIGEST_LENGTH);
            SHA256(combined.data(), combined.size(), parent_hash.data());
            next_level.push_back(parent_hash);
        }
        
        leaf_hashes = std::move(next_level);
    }
    
    return leaf_hashes.empty() ? std::vector<uint8_t>(SHA256_DIGEST_LENGTH, 0) : leaf_hashes[0];
}

std::string PartitionRecoveryManager::calculate_model_state_hash(const ModelStateSnapshot& state) {
    auto merkle_root = build_merkle_tree(state);
    return bytes_to_hex(merkle_root);
}

std::string PartitionRecoveryManager::bytes_to_hex(const std::vector<uint8_t>& bytes) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (uint8_t byte : bytes) {
        ss << std::setw(2) << static_cast<int>(byte);
    }
    return ss.str();
}

std::string PartitionRecoveryManager::generate_partition_id() {
    static std::atomic<uint64_t> counter{0};
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    return "partition_" + std::to_string(timestamp) + "_" + std::to_string(counter.fetch_add(1));
}

std::string PartitionRecoveryManager::generate_recovery_session_id() {
    static std::atomic<uint64_t> counter{0};
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    return "recovery_" + std::to_string(timestamp) + "_" + std::to_string(counter.fetch_add(1));
}

// Utility functions for serialization (simplified implementations)
std::vector<uint8_t> PartitionRecoveryManager::serialize_heartbeat(const HeartbeatMessage& heartbeat) {
    serialization::Serializer s;
    s.write_string(heartbeat.sender_id);
    s.write_uint64(heartbeat.timestamp);
    s.write_uint64(heartbeat.sequence_number);
    s.write_string(heartbeat.model_state_hash);
    return s.take_buffer();
}

std::vector<uint8_t> PartitionRecoveryManager::serialize_reconciliation_request(
    const StateReconciliationRequest& request) {
    serialization::Serializer s;
    s.write_string(request.session_id);
    s.write_string(request.requester_id);
    s.write_string(request.target_peer_id);
    s.write_bytes(request.our_merkle_root);
    return s.take_buffer();
}

bool PartitionRecoveryManager::simulate_reconciliation_response(
    const StateReconciliationRequest& request,
    ModelStateSnapshot& peer_state) {
    
    // Simulate receiving a model state from peer
    // In real implementation, this would be received via network
    
    peer_state.training_step = 1000;  // Simulate some training progress
    peer_state.epoch = 5;
    peer_state.timestamp = std::chrono::steady_clock::now();
    
    // Simulate some parameters (empty for now)
    peer_state.parameters.clear();
    
    return true;  // Simulate successful response
}

void PartitionRecoveryManager::process_recovery_sessions() {
    std::lock_guard<std::mutex> lock(recovery_mutex_);
    
    auto now = std::chrono::steady_clock::now();
    
    for (auto& [session_id, session] : active_recovery_sessions_) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - session.start_time).count();
        
        if (elapsed > config_.recovery_timeout_ms && session.status == RecoveryStatus::INITIATED) {
            session.status = RecoveryStatus::TIMEOUT;
            std::cout << "Recovery session " << session_id << " timed out" << std::endl;
        }
    }
}

void PartitionRecoveryManager::cleanup_completed_sessions() {
    std::lock_guard<std::mutex> lock(recovery_mutex_);
    
    auto it = active_recovery_sessions_.begin();
    while (it != active_recovery_sessions_.end()) {
        if (it->second.status == RecoveryStatus::COMPLETED || 
            it->second.status == RecoveryStatus::FAILED ||
            it->second.status == RecoveryStatus::TIMEOUT) {
            
            std::cout << "Cleaning up recovery session: " << it->first 
                     << " (status: " << static_cast<int>(it->second.status) << ")" << std::endl;
            it = active_recovery_sessions_.erase(it);
        } else {
            ++it;
        }
    }
}

RecoveryStats PartitionRecoveryManager::get_recovery_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return recovery_stats_;
}

} // namespace p2p
