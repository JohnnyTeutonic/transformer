#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <chrono>
#include <atomic>
#include <queue>
#include <array>
#include <functional>

namespace p2p {

// Kademlia configuration constants
constexpr size_t KADEMLIA_KEY_SIZE = 20;  // 160-bit keys (SHA-1 size)
constexpr size_t KADEMLIA_BUCKET_SIZE = 20;  // k parameter
constexpr size_t KADEMLIA_ALPHA = 3;  // Concurrency parameter
constexpr size_t KADEMLIA_B = 5;  // Bits per hop in routing

// Kademlia node ID type
using NodeID = std::array<uint8_t, KADEMLIA_KEY_SIZE>;

// DHT Message types
enum class DHTMessageType : uint8_t {
    PING = 0x20,
    PONG = 0x21,
    FIND_NODE = 0x22,
    FIND_NODE_RESPONSE = 0x23,
    STORE = 0x24,
    STORE_RESPONSE = 0x25,
    FIND_VALUE = 0x26,
    FIND_VALUE_RESPONSE = 0x27
};

// Contact information for a node
struct KademliaContact {
    NodeID node_id;
    std::string ip_address;
    uint16_t port;
    std::chrono::steady_clock::time_point last_seen;
    uint32_t rtt_ms;  // Round-trip time
    uint32_t failures;  // Consecutive failure count
    
    KademliaContact() : port(0), rtt_ms(0), failures(0) {
        node_id.fill(0);
        last_seen = std::chrono::steady_clock::now();
    }
    
    KademliaContact(const NodeID& id, const std::string& ip, uint16_t p)
        : node_id(id), ip_address(ip), port(p), rtt_ms(0), failures(0) {
        last_seen = std::chrono::steady_clock::now();
    }
    
    bool is_stale(uint32_t timeout_ms) const {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_seen);
        return elapsed.count() > timeout_ms;
    }
    
    void update_last_seen() {
        last_seen = std::chrono::steady_clock::now();
        failures = 0;
    }
    
    void record_failure() {
        failures++;
    }
};

// K-bucket for storing contacts
class KBucket {
public:
    KBucket(size_t max_size = KADEMLIA_BUCKET_SIZE);
    
    bool add_contact(const KademliaContact& contact);
    bool remove_contact(const NodeID& node_id);
    bool update_contact(const KademliaContact& contact);
    
    std::vector<KademliaContact> get_contacts() const;
    std::vector<KademliaContact> get_closest_contacts(const NodeID& target, size_t count) const;
    
    bool is_full() const { return contacts_.size() >= max_size_; }
    size_t size() const { return contacts_.size(); }
    bool contains(const NodeID& node_id) const;
    
    void refresh_stale_contacts(uint32_t timeout_ms);
    
private:
    std::vector<KademliaContact> contacts_;
    size_t max_size_;
    mutable std::mutex mutex_;
    
    void move_to_tail(size_t index);
};

// Routing table for Kademlia
class KademliaRoutingTable {
public:
    KademliaRoutingTable(const NodeID& local_node_id);
    
    bool add_contact(const KademliaContact& contact);
    bool remove_contact(const NodeID& node_id);
    bool update_contact(const KademliaContact& contact);
    
    std::vector<KademliaContact> find_closest_contacts(const NodeID& target, size_t count) const;
    std::vector<KademliaContact> get_all_contacts() const;
    
    void refresh_buckets();
    size_t get_bucket_index(const NodeID& node_id) const;
    
    // Statistics
    size_t total_contacts() const;
    void print_statistics() const;
    
private:
    NodeID local_node_id_;
    std::array<std::unique_ptr<KBucket>, KADEMLIA_KEY_SIZE * 8> buckets_;  // 160 buckets
    mutable std::mutex mutex_;
    
    size_t calculate_distance_bits(const NodeID& a, const NodeID& b) const;
};

// DHT Messages
struct DHTMessage {
    DHTMessageType type;
    NodeID sender_id;
    NodeID target_id;  // For lookups
    std::string transaction_id;
    std::vector<uint8_t> payload;
    
    DHTMessage() : type(DHTMessageType::PING) {
        sender_id.fill(0);
        target_id.fill(0);
    }
};

struct DHTStoreItem {
    NodeID key;
    std::vector<uint8_t> value;
    std::chrono::steady_clock::time_point expiry;
    NodeID publisher;
    
    DHTStoreItem() {
        key.fill(0);
        publisher.fill(0);
        expiry = std::chrono::steady_clock::now() + std::chrono::hours(24);
    }
    
    bool is_expired() const {
        return std::chrono::steady_clock::now() > expiry;
    }
};

// Lookup operation state
struct LookupState {
    NodeID target;
    std::vector<KademliaContact> closest_nodes;
    std::unordered_map<std::string, bool> queried_nodes;  // node_id -> queried
    std::unordered_map<std::string, bool> responded_nodes;  // node_id -> responded
    std::chrono::steady_clock::time_point start_time;
    bool completed;
    
    LookupState(const NodeID& t) : target(t), completed(false) {
        start_time = std::chrono::steady_clock::now();
    }
};

// Kademlia DHT Configuration
struct KademliaConfig {
    uint32_t refresh_interval_ms = 3600000;  // 1 hour
    uint32_t republish_interval_ms = 86400000;  // 24 hours
    uint32_t expire_time_ms = 86400000;  // 24 hours
    uint32_t node_timeout_ms = 900000;  // 15 minutes
    uint32_t lookup_timeout_ms = 10000;  // 10 seconds
    size_t max_stored_items = 10000;
    bool enable_value_storage = true;
};

// Main Kademlia DHT class
class KademliaDHT {
public:
    KademliaDHT(const NodeID& node_id, const std::string& bind_ip, uint16_t bind_port, const KademliaConfig& config = KademliaConfig{});
    ~KademliaDHT();
    
    // Core DHT operations
    bool start();
    void stop();
    
    // Node operations
    void bootstrap(const std::vector<std::pair<std::string, uint16_t>>& bootstrap_nodes);
    std::vector<KademliaContact> find_node(const NodeID& target);
    bool ping(const KademliaContact& contact);
    
    // Storage operations
    bool store(const NodeID& key, const std::vector<uint8_t>& value);
    std::vector<uint8_t> find_value(const NodeID& key);
    
    // Network integration
    void handle_message(const DHTMessage& message, const std::string& sender_ip, uint16_t sender_port);
    
    // Callbacks for network communication
    using MessageSender = std::function<void(const std::string& ip, uint16_t port, const DHTMessage& message)>;
    void set_message_sender(MessageSender sender) { message_sender_ = sender; }
    
    // Maintenance operations
    void refresh_buckets();
    void republish_data();
    void expire_data();
    
    // Information and statistics
    NodeID get_node_id() const { return node_id_; }
    std::vector<KademliaContact> get_known_nodes() const;
    size_t get_stored_items_count() const;
    
    struct DHTStats {
        size_t total_nodes = 0;
        size_t stored_items = 0;
        uint64_t lookups_performed = 0;
        uint64_t lookups_successful = 0;
        uint64_t messages_sent = 0;
        uint64_t messages_received = 0;
        float average_lookup_time_ms = 0.0f;
    };
    
    DHTStats get_statistics() const;
    void print_statistics() const;

private:
    // Core state
    NodeID node_id_;
    std::string bind_ip_;
    uint16_t bind_port_;
    KademliaConfig config_;
    
    // Routing and storage
    std::unique_ptr<KademliaRoutingTable> routing_table_;
    std::unordered_map<std::string, DHTStoreItem> stored_data_;  // key -> item
    
    // Active operations
    std::unordered_map<std::string, std::unique_ptr<LookupState>> active_lookups_;
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> pending_pings_;
    
    // Network communication
    MessageSender message_sender_;
    
    // Synchronization and lifecycle
    mutable std::mutex state_mutex_;
    std::atomic<bool> running_;
    std::thread maintenance_thread_;
    
    // Statistics
    mutable DHTStats stats_;
    
    // Message handlers
    void handle_ping(const DHTMessage& message, const std::string& sender_ip, uint16_t sender_port);
    void handle_pong(const DHTMessage& message, const std::string& sender_ip, uint16_t sender_port);
    void handle_find_node(const DHTMessage& message, const std::string& sender_ip, uint16_t sender_port);
    void handle_find_node_response(const DHTMessage& message, const std::string& sender_ip, uint16_t sender_port);
    void handle_store(const DHTMessage& message, const std::string& sender_ip, uint16_t sender_port);
    void handle_store_response(const DHTMessage& message, const std::string& sender_ip, uint16_t sender_port);
    void handle_find_value(const DHTMessage& message, const std::string& sender_ip, uint16_t sender_port);
    void handle_find_value_response(const DHTMessage& message, const std::string& sender_ip, uint16_t sender_port);
    
    // Lookup operations
    std::string start_lookup(const NodeID& target);
    void continue_lookup(const std::string& lookup_id);
    void complete_lookup(const std::string& lookup_id);
    
    // Utility functions
    std::string generate_transaction_id();
    std::string node_id_to_string(const NodeID& id) const;
    NodeID string_to_node_id(const std::string& str) const;
    NodeID calculate_distance(const NodeID& a, const NodeID& b) const;
    
    void send_message(const std::string& ip, uint16_t port, const DHTMessage& message);
    void maintenance_loop();
    
    // Bootstrap helpers
    void perform_bootstrap_lookup();
};

// Utility functions
NodeID generate_random_node_id();
NodeID hash_to_node_id(const std::string& data);
std::string format_node_id(const NodeID& id);
uint32_t calculate_xor_distance_bits(const NodeID& a, const NodeID& b);

} // namespace p2p
