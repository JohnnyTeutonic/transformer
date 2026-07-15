#include "../include/pbft_consensus.hpp"
#include "../include/serialization.hpp"
#include <iostream>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <openssl/sha.h>
#include <openssl/evp.h>

using namespace serialization;

namespace p2p {

PBFTConsensus::PBFTConsensus(const std::string& replica_id, const PBFTConfig& config)
    : replica_id_(replica_id)
    , config_(config)
    , current_view_(0)
    , sequence_number_(1)
    , total_replicas_(config.f * 3 + 1)
    , view_change_in_progress_(false)
    , last_checkpoint_sequence_(0)
    , running_(true) {
    
    std::cout << "PBFT Consensus initialized for replica " << replica_id_ 
              << " (f=" << config_.f << ", total_replicas=" << total_replicas_ << ")" << std::endl;
}

PBFTConsensus::~PBFTConsensus() {
    running_ = false;
}

bool PBFTConsensus::submit_request(const PBFTRequest& request) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    if (!is_primary()) {
        // Forward to primary or reject
        return false;
    }
    
    // Add to pending queue
    pending_requests_.push(request);
    process_pending_requests();
    
    return true;
}

void PBFTConsensus::process_pending_requests() {
    while (!pending_requests_.empty() && !view_change_in_progress_) {
        PBFTRequest request = pending_requests_.front();
        pending_requests_.pop();
        
        // Create request state
        RequestState state;
        state.request = request;
        state.phase = PBFTPhase::PRE_PREPARE;
        state.start_time = std::chrono::steady_clock::now();
        
        active_requests_[sequence_number_] = state;
        
        // Send PRE-PREPARE
        send_pre_prepare(request);
        
        sequence_number_++;
    }
}

void PBFTConsensus::send_pre_prepare(const PBFTRequest& request) {
    PBFTPrePrepare pre_prepare;
    pre_prepare.view = current_view_;
    pre_prepare.sequence_number = sequence_number_;
    pre_prepare.digest = compute_digest(request);
    pre_prepare.request = request;
    pre_prepare.primary_signature = sign_message(pre_prepare.digest);
    
    // Store in our own state
    active_requests_[sequence_number_].pre_prepare = pre_prepare;
    
    // Broadcast to all replicas
    std::vector<uint8_t> serialized_data;
    serialize_pbft_pre_prepare(pre_prepare, serialized_data);
    
    if (broadcast_sender_) {
        broadcast_sender_(PBFTMessageType::PRE_PREPARE, serialized_data);
    }
    
    std::cout << "PBFT: Sent PRE-PREPARE for sequence " << sequence_number_ 
              << " in view " << current_view_ << std::endl;
}

void PBFTConsensus::handle_pre_prepare(const PBFTPrePrepare& message, const std::string& sender_id) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    // Verify sender is the primary for this view
    if (sender_id != get_primary_id()) {
        std::cout << "PBFT: Rejecting PRE-PREPARE from non-primary " << sender_id << std::endl;
        return;
    }
    
    // Verify view and sequence number
    if (message.view != current_view_) {
        std::cout << "PBFT: PRE-PREPARE view mismatch: " << message.view << " vs " << current_view_ << std::endl;
        return;
    }
    
    // Verify message signature
    if (!verify_message_signature(message.digest, message.primary_signature, sender_id)) {
        std::cout << "PBFT: PRE-PREPARE signature verification failed" << std::endl;
        return;
    }
    
    // Verify digest matches request
    if (compute_digest(message.request) != message.digest) {
        std::cout << "PBFT: PRE-PREPARE digest mismatch" << std::endl;
        return;
    }
    
    // Accept PRE-PREPARE
    RequestState state;
    state.request = message.request;
    state.pre_prepare = message;
    state.phase = PBFTPhase::PREPARE;
    state.start_time = std::chrono::steady_clock::now();
    
    active_requests_[message.sequence_number] = state;
    
    // Send PREPARE
    send_prepare(message.view, message.sequence_number, message.digest);
    
    std::cout << "PBFT: Accepted PRE-PREPARE and sent PREPARE for sequence " 
              << message.sequence_number << std::endl;
}

void PBFTConsensus::send_prepare(uint32_t view, uint64_t seq_num, const std::string& digest) {
    PBFTPrepare prepare;
    prepare.view = view;
    prepare.sequence_number = seq_num;
    prepare.digest = digest;
    prepare.replica_id = replica_id_;
    prepare.signature = sign_message(digest);
    
    // Store our own prepare
    active_requests_[seq_num].prepare_messages[replica_id_] = prepare;
    
    // Broadcast to all replicas
    std::vector<uint8_t> serialized_data;
    serialize_pbft_prepare(prepare, serialized_data);
    
    if (broadcast_sender_) {
        broadcast_sender_(PBFTMessageType::PREPARE, serialized_data);
    }
}

void PBFTConsensus::handle_prepare(const PBFTPrepare& message, const std::string& sender_id) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    // Verify message
    if (message.view != current_view_) {
        return;
    }
    
    if (!verify_message_signature(message.digest, message.signature, sender_id)) {
        std::cout << "PBFT: PREPARE signature verification failed from " << sender_id << std::endl;
        return;
    }
    
    auto it = active_requests_.find(message.sequence_number);
    if (it == active_requests_.end()) {
        std::cout << "PBFT: Received PREPARE for unknown sequence " << message.sequence_number << std::endl;
        return;
    }
    
    // Store prepare message
    it->second.prepare_messages[sender_id] = message;
    
    // Check if we have enough prepares for quorum
    if (has_quorum_prepares(message.sequence_number)) {
        it->second.phase = PBFTPhase::COMMIT;
        send_commit(message.view, message.sequence_number, message.digest);
        
        std::cout << "PBFT: Quorum reached for PREPARE, sent COMMIT for sequence " 
                  << message.sequence_number << std::endl;
    }
}

void PBFTConsensus::send_commit(uint32_t view, uint64_t seq_num, const std::string& digest) {
    PBFTCommit commit;
    commit.view = view;
    commit.sequence_number = seq_num;
    commit.digest = digest;
    commit.replica_id = replica_id_;
    commit.signature = sign_message(digest);
    
    // Store our own commit
    active_requests_[seq_num].commit_messages[replica_id_] = commit;
    
    // Broadcast to all replicas
    std::vector<uint8_t> serialized_data;
    serialize_pbft_commit(commit, serialized_data);
    
    if (broadcast_sender_) {
        broadcast_sender_(PBFTMessageType::COMMIT, serialized_data);
    }
}

void PBFTConsensus::handle_commit(const PBFTCommit& message, const std::string& sender_id) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    // Verify message
    if (message.view != current_view_) {
        return;
    }
    
    if (!verify_message_signature(message.digest, message.signature, sender_id)) {
        std::cout << "PBFT: COMMIT signature verification failed from " << sender_id << std::endl;
        return;
    }
    
    auto it = active_requests_.find(message.sequence_number);
    if (it == active_requests_.end()) {
        return;
    }
    
    // Store commit message
    it->second.commit_messages[sender_id] = message;
    
    // Check if we have enough commits for execution
    if (has_quorum_commits(message.sequence_number) && !it->second.executed) {
        execute_request(message.sequence_number);
        
        std::cout << "PBFT: Executed request for sequence " << message.sequence_number << std::endl;
    }
}

void PBFTConsensus::execute_request(uint64_t sequence_number) {
    auto it = active_requests_.find(sequence_number);
    if (it == active_requests_.end() || it->second.executed) {
        return;
    }
    
    // Execute the request
    if (request_executor_) {
        std::string result = request_executor_(it->second.request);
        std::cout << "PBFT: Request executed with result digest: " << result.substr(0, 16) << "..." << std::endl;
    }
    
    it->second.executed = true;
    stats_.requests_processed++;
    
    // Update statistics
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - it->second.start_time);
    
    if (stats_.requests_processed == 1) {
        stats_.average_consensus_time_ms = duration.count();
    } else {
        stats_.average_consensus_time_ms = (stats_.average_consensus_time_ms * (stats_.requests_processed - 1) + duration.count()) / stats_.requests_processed;
    }
    
    // Check if we need to create a checkpoint
    if (config_.enable_checkpoints && sequence_number % config_.checkpoint_interval == 0) {
        create_checkpoint();
    }
    
    // Clean up old requests periodically
    if (sequence_number % 10 == 0) {
        cleanup_old_requests();
    }
}

bool PBFTConsensus::has_quorum_prepares(uint64_t sequence_number) const {
    auto it = active_requests_.find(sequence_number);
    if (it == active_requests_.end()) {
        return false;
    }
    
    return it->second.prepare_messages.size() >= get_quorum_size();
}

bool PBFTConsensus::has_quorum_commits(uint64_t sequence_number) const {
    auto it = active_requests_.find(sequence_number);
    if (it == active_requests_.end()) {
        return false;
    }
    
    return it->second.commit_messages.size() >= get_quorum_size();
}

void PBFTConsensus::initiate_view_change() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    if (view_change_in_progress_) {
        return;
    }
    
    view_change_in_progress_ = true;
    uint32_t new_view = current_view_ + 1;
    
    std::cout << "PBFT: Initiating view change from " << current_view_ << " to " << new_view << std::endl;
    
    PBFTViewChange view_change;
    view_change.new_view = new_view;
    view_change.replica_id = replica_id_;
    view_change.last_sequence_number = sequence_number_ - 1;
    
    // Add checkpoint proof and prepared requests
    // (Simplified implementation - in production, would include full proof)
    
    view_change.signature = sign_message(std::to_string(new_view) + replica_id_);
    
    // Store our view change message
    view_change_messages_[new_view].push_back(view_change);
    
    // Broadcast view change
    std::vector<uint8_t> serialized_data;
    serialize_pbft_view_change(view_change, serialized_data);
    
    if (broadcast_sender_) {
        broadcast_sender_(PBFTMessageType::VIEW_CHANGE, serialized_data);
    }
    
    stats_.view_changes++;
    start_view_change_timer();
}

void PBFTConsensus::handle_view_change(const PBFTViewChange& message, const std::string& sender_id) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    if (!verify_message_signature(std::to_string(message.new_view) + message.replica_id, 
                                  message.signature, sender_id)) {
        return;
    }
    
    view_change_messages_[message.new_view].push_back(message);
    
    // Check if we have enough view change messages to proceed
    if (view_change_messages_[message.new_view].size() >= get_quorum_size()) {
        if (get_primary_id() == replica_id_) {  // We are the new primary
            // Send NEW-VIEW message
            PBFTNewView new_view;
            new_view.view = message.new_view;
            new_view.view_change_messages = view_change_messages_[message.new_view];
            new_view.primary_signature = sign_message(std::to_string(message.new_view));
            
            std::vector<uint8_t> serialized_data;
            serialize_pbft_new_view(new_view, serialized_data);
            
            if (broadcast_sender_) {
                broadcast_sender_(PBFTMessageType::NEW_VIEW, serialized_data);
            }
            
            process_new_view(new_view);
        }
    }
}

void PBFTConsensus::handle_new_view(const PBFTNewView& message, const std::string& sender_id) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    // Verify sender is the new primary
    if (sender_id != get_primary_id()) {
        return;
    }
    
    if (!verify_message_signature(std::to_string(message.view), message.primary_signature, sender_id)) {
        return;
    }
    
    process_new_view(message);
}

void PBFTConsensus::process_new_view(const PBFTNewView& new_view) {
    current_view_ = new_view.view;
    view_change_in_progress_ = false;
    
    // Clear view change messages for old views
    auto it = view_change_messages_.begin();
    while (it != view_change_messages_.end()) {
        if (it->first <= current_view_) {
            it = view_change_messages_.erase(it);
        } else {
            ++it;
        }
    }
    
    std::cout << "PBFT: View changed to " << current_view_ << std::endl;
    
    // Resume processing pending requests if we're the new primary
    if (is_primary()) {
        process_pending_requests();
    }
}

void PBFTConsensus::create_checkpoint() {
    PBFTCheckpoint checkpoint;
    checkpoint.sequence_number = sequence_number_ - 1;
    checkpoint.state_digest = compute_state_digest();
    checkpoint.replica_id = replica_id_;
    checkpoint.signature = sign_message(checkpoint.state_digest);
    
    checkpoint_messages_[checkpoint.sequence_number].push_back(checkpoint);
    
    std::vector<uint8_t> serialized_data;
    serialize_pbft_checkpoint(checkpoint, serialized_data);
    
    if (broadcast_sender_) {
        broadcast_sender_(PBFTMessageType::CHECKPOINT, serialized_data);
    }
    
    last_checkpoint_sequence_ = checkpoint.sequence_number;
    stats_.checkpoints_created++;
    
    std::cout << "PBFT: Created checkpoint at sequence " << checkpoint.sequence_number << std::endl;
}

void PBFTConsensus::handle_checkpoint(const PBFTCheckpoint& message, const std::string& sender_id) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    if (!verify_message_signature(message.state_digest, message.signature, sender_id)) {
        return;
    }
    
    checkpoint_messages_[message.sequence_number].push_back(message);
    
    // Check if we have enough checkpoint messages for stability
    if (checkpoint_messages_[message.sequence_number].size() >= get_quorum_size()) {
        last_stable_checkpoint_ = message.state_digest;
        std::cout << "PBFT: Stable checkpoint established at sequence " << message.sequence_number << std::endl;
    }
}

bool PBFTConsensus::is_primary() const {
    return get_primary_id() == replica_id_;
}

std::string PBFTConsensus::get_primary_id() const {
    if (replica_ids_.empty()) {
        return replica_id_;  // Fallback if no replicas registered
    }
    
    size_t primary_index = current_view_ % replica_ids_.size();
    return replica_ids_[primary_index];
}

void PBFTConsensus::add_replica(const std::string& replica_id) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    if (std::find(replica_ids_.begin(), replica_ids_.end(), replica_id) == replica_ids_.end()) {
        replica_ids_.push_back(replica_id);
        replica_status_[replica_id] = true;
        std::sort(replica_ids_.begin(), replica_ids_.end());  // Ensure consistent ordering
        
        std::cout << "PBFT: Added replica " << replica_id << " (total: " << replica_ids_.size() << ")" << std::endl;
    }
}

void PBFTConsensus::remove_replica(const std::string& replica_id) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    auto it = std::find(replica_ids_.begin(), replica_ids_.end(), replica_id);
    if (it != replica_ids_.end()) {
        replica_ids_.erase(it);
        replica_status_.erase(replica_id);
        
        std::cout << "PBFT: Removed replica " << replica_id << " (total: " << replica_ids_.size() << ")" << std::endl;
    }
}

void PBFTConsensus::set_total_replicas(uint32_t total) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    total_replicas_ = total;
}

std::string PBFTConsensus::compute_digest(const PBFTRequest& request) const {
    // Simple SHA-256 hash of request content
    std::string content = request.client_id + std::to_string(request.timestamp) + request.operation;
    
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, content.c_str(), content.length());
    SHA256_Final(hash, &sha256);
    
    std::stringstream ss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
    }
    
    return ss.str();
}

std::string PBFTConsensus::compute_state_digest() const {
    // Simplified state digest - in production would hash entire state
    std::string state = replica_id_ + std::to_string(current_view_) + std::to_string(sequence_number_);
    
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, state.c_str(), state.length());
    SHA256_Final(hash, &sha256);
    
    std::stringstream ss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
    }
    
    return ss.str();
}

bool PBFTConsensus::verify_message_signature(const std::string& message, const std::string& signature, const std::string& sender_id) {
    // Simplified signature verification - in production would use proper cryptographic verification
    return !signature.empty() && !sender_id.empty();
}

std::string PBFTConsensus::sign_message(const std::string& message) {
    // Simplified signing - in production would use proper cryptographic signing
    return replica_id_ + "_signature_" + std::to_string(std::hash<std::string>{}(message));
}

void PBFTConsensus::cleanup_old_requests() {
    auto cutoff_time = std::chrono::steady_clock::now() - std::chrono::milliseconds(config_.request_timeout_ms * 2);
    
    auto it = active_requests_.begin();
    while (it != active_requests_.end()) {
        if (it->second.executed && it->second.start_time < cutoff_time) {
            it = active_requests_.erase(it);
        } else {
            ++it;
        }
    }
}

void PBFTConsensus::start_view_change_timer() {
    view_change_timer_ = std::chrono::steady_clock::now();
}

PBFTConsensus::PBFTStats PBFTConsensus::get_statistics() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    PBFTStats stats = stats_;
    stats.current_view = current_view_;
    stats.last_sequence_number = sequence_number_ - 1;
    
    return stats;
}

void PBFTConsensus::print_statistics() const {
    auto stats = get_statistics();
    
    std::cout << "\n=== PBFT Consensus Statistics ===" << std::endl;
    std::cout << "Replica ID: " << replica_id_ << std::endl;
    std::cout << "Current view: " << stats.current_view << std::endl;
    std::cout << "Last sequence number: " << stats.last_sequence_number << std::endl;
    std::cout << "Requests processed: " << stats.requests_processed << std::endl;
    std::cout << "View changes: " << stats.view_changes << std::endl;
    std::cout << "Checkpoints created: " << stats.checkpoints_created << std::endl;
    std::cout << "Average consensus time: " << stats.average_consensus_time_ms << " ms" << std::endl;
    std::cout << "Is primary: " << (is_primary() ? "Yes" : "No") << std::endl;
    std::cout << "Active requests: " << active_requests_.size() << std::endl;
    std::cout << "Total replicas: " << replica_ids_.size() << std::endl;
    std::cout << "================================\n" << std::endl;
}

} // namespace p2p
