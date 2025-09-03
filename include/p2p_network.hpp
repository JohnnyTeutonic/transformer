#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <functional>
#include <queue>
#include <future>
#include <random>

// OpenSSL includes
#include <openssl/ssl.h>
#include <openssl/err.h>
#include <openssl/evp.h>

// Forward declarations
class Matrix;
class DistributedTransformer;
class P2PTrainingCoordinator; // Forward-declare the coordinator

// Forward declare OpenSSL structs
struct ssl_ctx_st;
struct ssl_st;
struct evp_pkey_st; // Forward declare for EVP_PKEY

namespace p2p {

// Network message types
enum class MessageType : uint8_t {
    NODE_DISCOVERY = 0x01,
    NODE_ANNOUNCEMENT = 0x02,
    GRADIENT_PROPOSAL = 0x03,
    GRADIENT_VOTE = 0x04, // This will be repurposed as the 'PREPARE' vote
    GRADIENT_COMMIT = 0x05,
    HEARTBEAT = 0x06,
    NODE_LEAVE = 0x07,
    TRAINING_SYNC = 0x08,
    DATA_REQUEST = 0x09,
    DATA_RESPONSE = 0x0A,
    PEER_LIST_REQUEST = 0x0B,
    PEER_LIST_RESPONSE = 0x0C,
    STATE_SYNC_REQUEST = 0x0D,
    STATE_SYNC_RESPONSE = 0x0E
};

// Manages the global SSL context for the application
class SSLContext {
public:
    SSLContext();
    ~SSLContext();

    bool init(const std::string& cert_path, const std::string& key_path);
    SSL_CTX* get() const { return ctx_.get(); }

private:
    struct SSLContextDeleter {
        void operator()(SSL_CTX* ctx) const { SSL_CTX_free(ctx); }
    };
    std::unique_ptr<SSL_CTX, SSLContextDeleter> ctx_;
};


// Node information
struct NodeInfo {
    std::string node_id;
    std::string ip_address;
    uint16_t port;
    uint32_t compute_capability;  // Relative compute power (0-1000)
    uint64_t available_memory;    // Available GPU memory in bytes
    std::chrono::steady_clock::time_point last_seen;
    bool is_trusted;              // For Byzantine fault tolerance
    float reputation_score;       // 0.0 to 1.0 based on past behavior
    
    // SSL session info
    std::shared_ptr<SSL> ssl_session; // Manages the SSL state for this connection
    std::string public_key; // Peer's public key in PEM format

    NodeInfo() : compute_capability(0), available_memory(0), is_trusted(false), reputation_score(0.5f) {}
};

// Network message structure
struct NetworkMessage {
    MessageType type;
    std::string sender_id;
    std::string recipient_id;  // Empty for broadcast
    uint32_t sequence_number;
    uint64_t timestamp;
    std::vector<uint8_t> payload;
    std::string signature;     // For message authentication
    
    NetworkMessage() : type(MessageType::HEARTBEAT), sequence_number(0), timestamp(0) {}
};

// Gradient proposal for consensus
struct GradientProposal {
    std::string proposal_id;
    std::string proposer_id;
    uint32_t epoch;
    uint32_t batch_id;
    std::vector<float> gradient_hash;  // Hash of gradient for verification
    std::vector<float> gradient_data;  // Actual gradient data
    uint64_t timestamp;
    
    GradientProposal() : epoch(0), batch_id(0), timestamp(0) {}
};

// Vote on gradient proposal (now the PREPARE vote)
struct GradientVote {
    std::string proposal_id;
    std::string voter_id;
    bool approve;
    std::string reason;  // Optional reason for rejection
    uint64_t timestamp;
    
    GradientVote() : approve(false), timestamp(0) {}
};

// Represents a COMMIT message for a gradient proposal
struct GradientCommit {
    std::string proposal_id;
    std::string committer_id;
    uint64_t timestamp;
};

// Represents a gradient matrix after quantization
struct QuantizedGradient {
    std::vector<int8_t> quantized_data;
    float scale;
    size_t original_rows;
    size_t original_cols;
};

// Represents a chunk of the model's state for synchronization
struct ModelStateChunk {
    std::string parameter_name;
    uint32_t chunk_index;
    uint32_t total_chunks;
    std::vector<uint8_t> data;
};

// P2P Network configuration
struct P2PConfig {
    std::string node_id;
    std::string bind_address = "0.0.0.0";
    uint16_t bind_port = 8888;
    std::vector<std::string> bootstrap_nodes;  // Initial peers to connect to

    // Security
    std::string tls_cert_path = "p2p.crt";
    std::string tls_key_path = "p2p.key";
    std::string private_key_path = "node.key"; // For message signing
    std::string public_key_path = "node.pub";  // Our public key
    
    // Consensus parameters
    float consensus_threshold = 0.67f;  // 2/3 majority required
    uint32_t max_proposal_age_ms = 30000;  // 30 seconds
    uint32_t heartbeat_interval_ms = 5000;  // 5 seconds
    uint32_t node_timeout_ms = 15000;  // 15 seconds
    
    // Byzantine fault tolerance
    uint32_t max_byzantine_nodes = 1;  // f in (3f+1) nodes
    bool require_signatures = true;
    
    // Performance tuning
    uint32_t max_concurrent_connections = 100;
    uint32_t message_buffer_size = 1024 * 1024;  // 1MB
    
    // Gradient Quantization
    bool enable_gradient_quantization = true;
    int gradient_quantization_bits = 8; // 8-bit quantization
    
    P2PConfig() {
        // Generate random node ID if not provided
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 15);
        
        node_id = "node_";
        for (int i = 0; i < 16; ++i) {
            node_id += "0123456789abcdef"[dis(gen)];
        }
    }
};

// Network statistics
struct NetworkStats {
    uint64_t messages_sent = 0;
    uint64_t messages_received = 0;
    uint64_t bytes_sent = 0;
    uint64_t bytes_received = 0;
    uint64_t consensus_rounds = 0;
    uint64_t failed_consensus = 0;
    float average_consensus_time_ms = 0.0f;
    uint32_t active_peers = 0;
    uint32_t total_peers_seen = 0;
    
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
    
    void print_stats() const;
};

// Main P2P Network class
class P2PNetwork {
public:
    explicit P2PNetwork(const P2PConfig& config);
    ~P2PNetwork();
    
    // Network lifecycle
    bool start();
    void stop();
    bool is_running() const { return running_.load(); }
    
    // Node management
    bool connect_to_peer(const std::string& address, uint16_t port);
    void disconnect_from_peer(const std::string& node_id);
    std::vector<NodeInfo> get_active_peers() const;
    NodeInfo get_node_info(const std::string& node_id) const;
    
    // Gradient consensus protocol
    std::string propose_gradient(const std::vector<Matrix>& gradients, uint32_t epoch, uint32_t batch_id);
    bool vote_on_gradient(const std::string& proposal_id, bool approve, const std::string& reason = "");
    bool wait_for_consensus(const std::string& proposal_id, uint32_t timeout_ms = 30000);
    std::vector<Matrix> get_consensus_gradient(const std::string& proposal_id);
    
    // Message handling
    void send_message(const NetworkMessage& message);
    void broadcast_message(const NetworkMessage& message);
    void register_message_handler(MessageType type, std::function<void(const NetworkMessage&)> handler);
    
    // Statistics and monitoring
    NetworkStats get_stats() const { return stats_; }
    void reset_stats() { stats_ = NetworkStats(); }
    
    // Configuration
    const P2PConfig& get_config() const { return config_; }
    void update_config(const P2PConfig& config);
    
    // Fault tolerance
    void report_malicious_node(const std::string& node_id, const std::string& reason);
    void blacklist_node(const std::string& node_id);
    bool is_node_trusted(const std::string& node_id) const;

    void set_coordinator(std::shared_ptr<P2PTrainingCoordinator> coordinator);
    
private:
    // Core network operations
    void network_thread();
    void heartbeat_thread();
    void consensus_thread();
    void cleanup_thread();
    
    // Message processing
    void handle_incoming_message(const NetworkMessage& message);
    void handle_node_discovery(const NetworkMessage& message);
    void handle_node_announcement(const NetworkMessage& message);
    void handle_gradient_proposal(const NetworkMessage& message);
    void handle_gradient_prepare(const NetworkMessage& message); // Renamed from handle_gradient_vote
    void handle_gradient_commit(const NetworkMessage& message);
    void handle_heartbeat(const NetworkMessage& message);
    void handle_peer_list_request(const NetworkMessage& message);
    void handle_peer_list_response(const NetworkMessage& message);
    void handle_state_sync_request(const NetworkMessage& message);
    void handle_state_sync_response(const NetworkMessage& message);
    
    // Consensus algorithm (Byzantine Fault Tolerant)
    bool validate_gradient_proposal(const GradientProposal& proposal);
    bool check_consensus_reached(const std::string& proposal_id);
    void finalize_consensus(const std::string& proposal_id);
    void cleanup_old_proposals();
    
    // Network utilities
    std::vector<uint8_t> serialize_message(const NetworkMessage& message);
    NetworkMessage deserialize_message(const std::vector<uint8_t>& data);
    std::string calculate_gradient_hash(const std::vector<Matrix>& gradients);
    bool verify_message_signature(const NetworkMessage& message);
    void sign_message(NetworkMessage& message);
    
    // Node management
    void update_node_reputation(const std::string& node_id, float delta);
    void remove_inactive_nodes();
    bool should_accept_new_peer() const;
    
    // State Synchronization
    void request_model_state();
    
    // Configuration and state
    P2PConfig config_;
    std::atomic<bool> running_{false};
    std::atomic<bool> is_synchronized_{false};
    std::unique_ptr<SSLContext> ssl_context_;
    std::unique_ptr<EVP_PKEY, decltype(&EVP_PKEY_free)> private_key_{nullptr, EVP_PKEY_free};
    
    // Network state
    mutable std::mutex nodes_mutex_;
    std::unordered_map<std::string, NodeInfo> known_nodes_;
    std::unordered_map<std::string, std::string> node_connections_;  // node_id -> connection_info
    
    // Consensus state
    mutable std::mutex consensus_mutex_;
    std::unordered_map<std::string, GradientProposal> active_proposals_;
    std::unordered_map<std::string, std::vector<GradientVote>> proposal_prepare_votes_;
    std::unordered_map<std::string, std::vector<GradientCommit>> proposal_commit_votes_;
    std::unordered_map<std::string, std::vector<Matrix>> consensus_gradients_;
    
    // State for assembling model chunks
    mutable std::mutex state_sync_mutex_;
    std::map<std::string, std::vector<ModelStateChunk>> incoming_state_chunks_;
    
    // Message handling
    mutable std::mutex handlers_mutex_;
    std::unordered_map<MessageType, std::function<void(const NetworkMessage&)>> message_handlers_;
    
    // Threading
    std::vector<std::thread> worker_threads_;
    std::atomic<bool> shutdown_requested_{false};
    
    // Statistics
    mutable std::mutex stats_mutex_;
    NetworkStats stats_;
    
    // Blacklist for malicious nodes
    mutable std::mutex blacklist_mutex_;
    std::unordered_set<std::string> blacklisted_nodes_;
    
    // Sequence numbers for message ordering
    std::atomic<uint32_t> sequence_counter_{0};
    
    // Random number generator for various operations
    mutable std::mutex rng_mutex_;
    std::mt19937 rng_;
};

// P2P Training Coordinator - integrates with DistributedTransformer
class P2PTrainingCoordinator {
public:
    P2PTrainingCoordinator(std::shared_ptr<DistributedTransformer> transformer,
                          std::shared_ptr<P2PNetwork> network);
    ~P2PTrainingCoordinator();
    
    // Training coordination
    bool start();
    void stop();
    
    // Submit gradients to the asynchronous queue
    void submit_gradients(const std::vector<Matrix>& local_gradients);
    
    // Training step coordination
    // bool coordinate_training_step(const std::vector<Matrix>& local_gradients,
    //                              std::vector<Matrix>& consensus_gradients);
    
    // Fault tolerance
    void handle_node_failure(const std::string& node_id);
    void handle_network_partition();
    
    // Statistics
    struct TrainingStats {
        uint64_t training_steps = 0;
        uint64_t consensus_failures = 0;
        float average_step_time_ms = 0.0f;
        uint32_t active_training_nodes = 0;
    };
    
    TrainingStats get_training_stats() const { return training_stats_; }

    std::shared_ptr<DistributedTransformer> get_transformer() { return transformer_; }

private:
    void coordination_thread_loop();
    bool validate_training_state();
    void synchronize_model_state();
    
    std::shared_ptr<DistributedTransformer> transformer_;
    std::shared_ptr<P2PNetwork> network_;
    
    std::atomic<bool> training_active_{false};
    std::thread coordination_thread_;
    
    // Thread-safe queue for gradients
    std::queue<std::vector<Matrix>> gradient_queue_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    mutable std::mutex stats_mutex_;
    TrainingStats training_stats_;
    
    // Training synchronization
    std::mutex training_mutex_;
    std::condition_variable training_cv_;
    uint32_t current_epoch_ = 0;
    uint32_t current_batch_ = 0;
};

// Utility functions
namespace utils {
    // Network discovery helpers
    std::vector<std::string> discover_local_peers(uint16_t port_range_start = 8888, 
                                                 uint16_t port_range_end = 8898);
    bool is_port_available(uint16_t port);
    std::string get_local_ip_address();
    
    // Gradient compression/decompression
    std::vector<uint8_t> compress_gradients(const std::vector<Matrix>& gradients, int level = 1);
    std::vector<Matrix> decompress_gradients(const std::vector<uint8_t>& compressed_data);
    
    // Gradient quantization helpers
    QuantizedGradient quantize_matrix(const Matrix& matrix, int bits);
    Matrix dequantize_matrix(const QuantizedGradient& q_grad);

    // Cryptographic utilities
    std::string calculate_sha256(const std::vector<uint8_t>& data);
    std::string generate_keypair();  // Returns private key, public key stored in node info
    std::string sign_data(const std::vector<uint8_t>& data, const std::string& private_key);
    bool verify_signature(const std::vector<uint8_t>& data, const std::string& signature, 
                         const std::string& public_key);
}

} // namespace p2p
