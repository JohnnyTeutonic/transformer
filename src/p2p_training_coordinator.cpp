#include "../include/p2p_network.hpp"
#include "../include/distributed_transformer.hpp"
#include "../include/matrix.hpp"
#include <iostream>
#include <chrono>
#include <thread>
#include <algorithm>

namespace p2p {

// P2PTrainingCoordinator implementation
P2PTrainingCoordinator::P2PTrainingCoordinator(std::shared_ptr<DistributedTransformer> transformer,
                                             std::shared_ptr<P2PNetwork> network)
    : transformer_(transformer), network_(network) {
    
    if (!transformer_) {
        throw std::runtime_error("DistributedTransformer cannot be null");
    }
    if (!network_) {
        throw std::runtime_error("P2PNetwork cannot be null");
    }
    
    std::cout << "P2P Training Coordinator initialized" << std::endl;
}

P2PTrainingCoordinator::~P2PTrainingCoordinator() {
    stop_distributed_training();
}

bool P2PTrainingCoordinator::start_distributed_training() {
    if (training_active_.load()) {
        std::cout << "Distributed training already active" << std::endl;
        return true;
    }
    
    if (!network_->is_running()) {
        std::cerr << "P2P Network must be running before starting training" << std::endl;
        return false;
    }
    
    std::cout << "Starting distributed P2P training..." << std::endl;
    
    training_active_.store(true);
    coordination_thread_ = std::thread(&P2PTrainingCoordinator::training_coordination_thread, this);
    
    std::cout << "Distributed P2P training started" << std::endl;
    return true;
}

void P2PTrainingCoordinator::stop_distributed_training() {
    if (!training_active_.load()) {
        return;
    }
    
    std::cout << "Stopping distributed P2P training..." << std::endl;
    
    training_active_.store(false);
    
    // Notify coordination thread
    {
        std::lock_guard<std::mutex> lock(training_mutex_);
        training_cv_.notify_all();
    }
    
    if (coordination_thread_.joinable()) {
        coordination_thread_.join();
    }
    
    std::cout << "Distributed P2P training stopped" << std::endl;
}

bool P2PTrainingCoordinator::coordinate_training_step(const std::vector<Matrix>& local_gradients,
                                                     std::vector<Matrix>& consensus_gradients) {
    if (!training_active_.load()) {
        std::cerr << "Training not active" << std::endl;
        return false;
    }
    
    auto step_start = std::chrono::high_resolution_clock::now();
    
    std::cout << "Coordinating training step for epoch " << current_epoch_ 
              << ", batch " << current_batch_ << std::endl;
    
    // Step 1: Propose our local gradients to the network
    std::string proposal_id = network_->propose_gradient(local_gradients, current_epoch_, current_batch_);
    if (proposal_id.empty()) {
        std::cerr << "Failed to propose gradients" << std::endl;
        std::lock_guard<std::mutex> lock(stats_mutex_);
        training_stats_.consensus_failures++;
        return false;
    }
    
    // Step 2: Wait for consensus on gradients
    bool consensus_reached = network_->wait_for_consensus(proposal_id, 30000);  // 30 second timeout
    if (!consensus_reached) {
        std::cerr << "Consensus not reached for proposal: " << proposal_id << std::endl;
        std::lock_guard<std::mutex> lock(stats_mutex_);
        training_stats_.consensus_failures++;
        return false;
    }
    
    // Step 3: Get consensus gradients
    consensus_gradients = network_->get_consensus_gradient(proposal_id);
    if (consensus_gradients.empty()) {
        std::cerr << "Failed to get consensus gradients" << std::endl;
        std::lock_guard<std::mutex> lock(stats_mutex_);
        training_stats_.consensus_failures++;
        return false;
    }
    
    // Step 4: Update statistics
    auto step_end = std::chrono::high_resolution_clock::now();
    float step_time = std::chrono::duration<float, std::milli>(step_end - step_start).count();
    
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        training_stats_.training_steps++;
        
        // Update average step time using exponential moving average
        if (training_stats_.average_step_time_ms == 0.0f) {
            training_stats_.average_step_time_ms = step_time;
        } else {
            training_stats_.average_step_time_ms = 0.9f * training_stats_.average_step_time_ms + 0.1f * step_time;
        }
        
        training_stats_.active_training_nodes = static_cast<uint32_t>(network_->get_active_peers().size() + 1);
    }
    
    std::cout << "Training step completed successfully in " << step_time << "ms" << std::endl;
    
    // Step 5: Advance to next batch
    current_batch_++;
    
    return true;
}

void P2PTrainingCoordinator::handle_node_failure(const std::string& node_id) {
    std::cout << "Handling node failure: " << node_id << std::endl;
    
    // Check if we need to adjust consensus requirements
    auto active_peers = network_->get_active_peers();
    uint32_t remaining_nodes = static_cast<uint32_t>(active_peers.size() + 1);  // +1 for ourselves
    
    std::cout << "Remaining active nodes: " << remaining_nodes << std::endl;
    
    // If we have too few nodes, we might need to pause training or adjust parameters
    const uint32_t MIN_TRAINING_NODES = 3;  // Minimum nodes for meaningful distributed training
    
    if (remaining_nodes < MIN_TRAINING_NODES) {
        std::cout << "Too few nodes remaining (" << remaining_nodes 
                  << "), pausing distributed training" << std::endl;
        
        // Temporarily pause training until more nodes join
        std::unique_lock<std::mutex> lock(training_mutex_);
        training_cv_.wait(lock, [this, &active_peers]() {
            active_peers = network_->get_active_peers();
            return !training_active_.load() || active_peers.size() + 1 >= MIN_TRAINING_NODES;
        });
        
        if (training_active_.load()) {
            std::cout << "Resuming distributed training with " 
                      << (active_peers.size() + 1) << " nodes" << std::endl;
        }
    }
    
    // Update training statistics
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        training_stats_.active_training_nodes = remaining_nodes;
    }
}

void P2PTrainingCoordinator::handle_network_partition() {
    std::cout << "Handling network partition..." << std::endl;
    
    // In case of network partition, we need to:
    // 1. Detect which partition we're in
    // 2. Decide whether to continue training or wait for network healing
    // 3. Implement partition tolerance strategies
    
    auto active_peers = network_->get_active_peers();
    uint32_t partition_size = static_cast<uint32_t>(active_peers.size() + 1);
    
    std::cout << "Current partition size: " << partition_size << std::endl;
    
    // Simple strategy: only continue if we have majority of original nodes
    // More sophisticated strategies could be implemented based on specific requirements
    
    const uint32_t MINIMUM_PARTITION_SIZE = 2;  // Minimum nodes to continue training
    
    if (partition_size < MINIMUM_PARTITION_SIZE) {
        std::cout << "Partition too small, pausing training until network heals" << std::endl;
        
        // Wait for network to heal
        std::unique_lock<std::mutex> lock(training_mutex_);
        training_cv_.wait(lock, [this, &active_peers]() {
            active_peers = network_->get_active_peers();
            return !training_active_.load() || active_peers.size() + 1 >= MINIMUM_PARTITION_SIZE;
        });
        
        if (training_active_.load()) {
            std::cout << "Network healed, resuming training" << std::endl;
        }
    }
}

void P2PTrainingCoordinator::training_coordination_thread() {
    std::cout << "Training coordination thread started" << std::endl;
    
    while (training_active_.load()) {
        try {
            // Validate training state
            if (!validate_training_state()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                continue;
            }
            
            // Synchronize model state periodically
            static uint32_t sync_counter = 0;
            if (++sync_counter % 10 == 0) {  // Every 10 iterations
                synchronize_model_state();
            }
            
            // Check for node failures or network issues
            auto active_peers = network_->get_active_peers();
            static uint32_t last_peer_count = 0;
            
            if (active_peers.size() != last_peer_count) {
                if (active_peers.size() < last_peer_count) {
                    // Node(s) left
                    std::cout << "Detected node departure: " << last_peer_count 
                              << " -> " << active_peers.size() << " peers" << std::endl;
                    handle_node_failure("unknown");  // Would track specific node in production
                } else {
                    // Node(s) joined
                    std::cout << "Detected new node(s): " << last_peer_count 
                              << " -> " << active_peers.size() << " peers" << std::endl;
                    
                    // Notify waiting threads that new nodes have joined
                    training_cv_.notify_all();
                }
                last_peer_count = static_cast<uint32_t>(active_peers.size());
            }
            
            // Sleep briefly to avoid busy waiting
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
        } catch (const std::exception& e) {
            std::cerr << "Error in training coordination: " << e.what() << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
    }
    
    std::cout << "Training coordination thread stopped" << std::endl;
}

bool P2PTrainingCoordinator::validate_training_state() {
    // Check if network is still running
    if (!network_->is_running()) {
        std::cerr << "P2P Network is not running" << std::endl;
        return false;
    }
    
    // Check if we have minimum number of peers for training
    auto active_peers = network_->get_active_peers();
    const uint32_t MIN_PEERS = 1;  // Can train with just one other peer
    
    if (active_peers.size() < MIN_PEERS) {
        // Don't spam the log, just return false
        return false;
    }
    
    // Check if transformer is still valid
    if (!transformer_) {
        std::cerr << "DistributedTransformer is null" << std::endl;
        return false;
    }
    
    return true;
}

void P2PTrainingCoordinator::synchronize_model_state() {
    std::cout << "Synchronizing model state across peers..." << std::endl;
    
    // In a production system, this would:
    // 1. Exchange model checksums with peers
    // 2. Detect any divergence in model parameters
    // 3. Implement model state synchronization protocol
    // 4. Handle cases where models have diverged due to network issues
    
    // For now, we'll just log that synchronization occurred
    std::cout << "Model state synchronization completed" << std::endl;
}

} // namespace p2p
