#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <chrono>
#include <atomic>
#include <queue>

namespace p2p {

// PBFT Message Types
enum class PBFTMessageType : uint8_t {
    REQUEST = 0x10,
    PRE_PREPARE = 0x11,
    PREPARE = 0x12,
    COMMIT = 0x13,
    VIEW_CHANGE = 0x14,
    NEW_VIEW = 0x15,
    CHECKPOINT = 0x16
};

// PBFT Phase enumeration
enum class PBFTPhase {
    IDLE,
    PRE_PREPARE,
    PREPARE,
    COMMIT,
    VIEW_CHANGE
};

// PBFT Request structure
struct PBFTRequest {
    std::string client_id;
    uint64_t timestamp;
    std::string operation;  // Serialized gradient data
    std::string signature;
    
    PBFTRequest() : timestamp(0) {}
};

// PBFT Pre-Prepare message
struct PBFTPrePrepare {
    uint32_t view;
    uint64_t sequence_number;
    std::string digest;  // Hash of the request
    PBFTRequest request;
    std::string primary_signature;
    
    PBFTPrePrepare() : view(0), sequence_number(0) {}
};

// PBFT Prepare message
struct PBFTPrepare {
    uint32_t view;
    uint64_t sequence_number;
    std::string digest;
    std::string replica_id;
    std::string signature;
    
    PBFTPrepare() : view(0), sequence_number(0) {}
};

// PBFT Commit message
struct PBFTCommit {
    uint32_t view;
    uint64_t sequence_number;
    std::string digest;
    std::string replica_id;
    std::string signature;
    
    PBFTCommit() : view(0), sequence_number(0) {}
};

// PBFT View Change message
struct PBFTViewChange {
    uint32_t new_view;
    std::string replica_id;
    uint64_t last_sequence_number;
    std::vector<std::pair<uint64_t, std::string>> checkpoint_proof;
    std::vector<PBFTPrePrepare> prepared_requests;
    std::string signature;
    
    PBFTViewChange() : new_view(0), last_sequence_number(0) {}
};

// PBFT New View message
struct PBFTNewView {
    uint32_t view;
    std::vector<PBFTViewChange> view_change_messages;
    std::vector<PBFTPrePrepare> pre_prepare_messages;
    std::string primary_signature;
    
    PBFTNewView() : view(0) {}
};

// PBFT Checkpoint message
struct PBFTCheckpoint {
    uint64_t sequence_number;
    std::string state_digest;
    std::string replica_id;
    std::string signature;
    
    PBFTCheckpoint() : sequence_number(0) {}
};

// PBFT Configuration
struct PBFTConfig {
    uint32_t f;  // Maximum number of Byzantine nodes (total nodes = 3f + 1)
    uint32_t checkpoint_interval = 100;  // Checkpoint every 100 requests
    uint32_t view_change_timeout_ms = 10000;  // 10 seconds
    uint32_t request_timeout_ms = 5000;  // 5 seconds
    uint32_t max_sequence_number = 1000000;  // Sequence number wraparound
    bool enable_checkpoints = true;
    
    PBFTConfig(uint32_t byzantine_nodes) : f(byzantine_nodes) {}
};

// Request state tracking
struct RequestState {
    PBFTRequest request;
    PBFTPrePrepare pre_prepare;
    std::unordered_map<std::string, PBFTPrepare> prepare_messages;
    std::unordered_map<std::string, PBFTCommit> commit_messages;
    PBFTPhase phase;
    std::chrono::steady_clock::time_point start_time;
    bool executed;
    
    RequestState() : phase(PBFTPhase::IDLE), executed(false) {}
};

// PBFT Consensus Engine
class PBFTConsensus {
public:
    PBFTConsensus(const std::string& replica_id, const PBFTConfig& config);
    ~PBFTConsensus();

    // Core PBFT operations
    bool submit_request(const PBFTRequest& request);
    void handle_pre_prepare(const PBFTPrePrepare& message, const std::string& sender_id);
    void handle_prepare(const PBFTPrepare& message, const std::string& sender_id);
    void handle_commit(const PBFTCommit& message, const std::string& sender_id);
    void handle_view_change(const PBFTViewChange& message, const std::string& sender_id);
    void handle_new_view(const PBFTNewView& message, const std::string& sender_id);
    void handle_checkpoint(const PBFTCheckpoint& message, const std::string& sender_id);

    // View management
    void initiate_view_change();
    bool is_primary() const;
    std::string get_primary_id() const;
    uint32_t get_current_view() const { return current_view_; }

    // State management
    void add_replica(const std::string& replica_id);
    void remove_replica(const std::string& replica_id);
    void set_total_replicas(uint32_t total);
    
    // Checkpoint management
    void create_checkpoint();
    bool verify_checkpoint(const PBFTCheckpoint& checkpoint);
    
    // Message verification
    bool verify_message_signature(const std::string& message, const std::string& signature, const std::string& sender_id);
    std::string sign_message(const std::string& message);
    
    // Callbacks for network communication
    using MessageSender = std::function<void(const std::string& recipient, PBFTMessageType type, const std::vector<uint8_t>& data)>;
    using BroadcastSender = std::function<void(PBFTMessageType type, const std::vector<uint8_t>& data)>;
    using RequestExecutor = std::function<std::string(const PBFTRequest& request)>;  // Returns result digest
    
    void set_message_sender(MessageSender sender) { message_sender_ = sender; }
    void set_broadcast_sender(BroadcastSender sender) { broadcast_sender_ = sender; }
    void set_request_executor(RequestExecutor executor) { request_executor_ = executor; }

    // Statistics and monitoring
    struct PBFTStats {
        uint64_t requests_processed = 0;
        uint64_t view_changes = 0;
        uint64_t checkpoints_created = 0;
        float average_consensus_time_ms = 0.0f;
        uint32_t current_view = 0;
        uint64_t last_sequence_number = 0;
    };
    
    PBFTStats get_statistics() const;
    void print_statistics() const;

private:
    // Core state
    std::string replica_id_;
    PBFTConfig config_;
    uint32_t current_view_;
    uint64_t sequence_number_;
    uint32_t total_replicas_;
    
    // Request tracking
    std::unordered_map<uint64_t, RequestState> active_requests_;
    std::queue<PBFTRequest> pending_requests_;
    
    // View change tracking
    std::unordered_map<uint32_t, std::vector<PBFTViewChange>> view_change_messages_;
    std::chrono::steady_clock::time_point view_change_timer_;
    bool view_change_in_progress_;
    
    // Checkpoint tracking
    std::unordered_map<uint64_t, std::vector<PBFTCheckpoint>> checkpoint_messages_;
    uint64_t last_checkpoint_sequence_;
    std::string last_stable_checkpoint_;
    
    // Replica management
    std::vector<std::string> replica_ids_;
    std::unordered_map<std::string, bool> replica_status_;  // true = active, false = suspected Byzantine
    
    // Network callbacks
    MessageSender message_sender_;
    BroadcastSender broadcast_sender_;
    RequestExecutor request_executor_;
    
    // Synchronization
    mutable std::mutex state_mutex_;
    std::atomic<bool> running_;
    
    // Statistics
    mutable PBFTStats stats_;
    
    // Helper methods
    void process_pending_requests();
    void send_pre_prepare(const PBFTRequest& request);
    void send_prepare(uint32_t view, uint64_t seq_num, const std::string& digest);
    void send_commit(uint32_t view, uint64_t seq_num, const std::string& digest);
    void execute_request(uint64_t sequence_number);
    
    bool is_prepared(uint64_t sequence_number) const;
    bool is_committed(uint64_t sequence_number) const;
    bool has_quorum_prepares(uint64_t sequence_number) const;
    bool has_quorum_commits(uint64_t sequence_number) const;
    
    std::string compute_digest(const PBFTRequest& request) const;
    std::string compute_state_digest() const;
    
    void cleanup_old_requests();
    void start_view_change_timer();
    void process_new_view(const PBFTNewView& new_view);
    
    uint32_t get_quorum_size() const { return 2 * config_.f + 1; }
    uint32_t get_total_nodes() const { return 3 * config_.f + 1; }
};

} // namespace p2p
