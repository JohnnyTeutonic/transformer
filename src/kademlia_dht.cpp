#include "../include/kademlia_dht.hpp"
#include <iostream>
#include <algorithm>
#include <random>
#include <sstream>
#include <iomanip>
#include <thread>
#include <openssl/sha.h>

namespace p2p {

// Utility functions implementation
NodeID generate_random_node_id() {
    NodeID id;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint8_t> dis(0, 255);
    
    for (size_t i = 0; i < KADEMLIA_KEY_SIZE; ++i) {
        id[i] = dis(gen);
    }
    
    return id;
}

NodeID hash_to_node_id(const std::string& data) {
    NodeID id;
    unsigned char hash[SHA_DIGEST_LENGTH];
    
    SHA1(reinterpret_cast<const unsigned char*>(data.c_str()), data.length(), hash);
    std::copy(hash, hash + KADEMLIA_KEY_SIZE, id.begin());
    
    return id;
}

std::string format_node_id(const NodeID& id) {
    std::stringstream ss;
    for (size_t i = 0; i < std::min(size_t(8), KADEMLIA_KEY_SIZE); ++i) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(id[i]);
    }
    return ss.str();
}

uint32_t calculate_xor_distance_bits(const NodeID& a, const NodeID& b) {
    for (size_t i = 0; i < KADEMLIA_KEY_SIZE; ++i) {
        uint8_t xor_byte = a[i] ^ b[i];
        if (xor_byte != 0) {
            // Find the position of the most significant bit
            uint32_t bit_pos = 0;
            for (int j = 7; j >= 0; --j) {
                if (xor_byte & (1 << j)) {
                    bit_pos = (KADEMLIA_KEY_SIZE - 1 - i) * 8 + j;
                    break;
                }
            }
            return bit_pos;
        }
    }
    return 0;  // Identical nodes
}

// KBucket implementation
KBucket::KBucket(size_t max_size) : max_size_(max_size) {}

bool KBucket::add_contact(const KademliaContact& contact) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Check if contact already exists
    for (size_t i = 0; i < contacts_.size(); ++i) {
        if (contacts_[i].node_id == contact.node_id) {
            // Move to tail (most recently seen)
            move_to_tail(i);
            contacts_.back().update_last_seen();
            return true;
        }
    }
    
    // Add new contact
    if (contacts_.size() < max_size_) {
        contacts_.push_back(contact);
        return true;
    }
    
    // Bucket is full, check if head contact is stale
    if (contacts_.front().is_stale(900000)) {  // 15 minutes
        contacts_.erase(contacts_.begin());
        contacts_.push_back(contact);
        return true;
    }
    
    return false;  // Bucket full with active contacts
}

bool KBucket::remove_contact(const NodeID& node_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = std::find_if(contacts_.begin(), contacts_.end(),
        [&node_id](const KademliaContact& contact) {
            return contact.node_id == node_id;
        });
    
    if (it != contacts_.end()) {
        contacts_.erase(it);
        return true;
    }
    
    return false;
}

bool KBucket::update_contact(const KademliaContact& contact) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (size_t i = 0; i < contacts_.size(); ++i) {
        if (contacts_[i].node_id == contact.node_id) {
            contacts_[i] = contact;
            move_to_tail(i);
            return true;
        }
    }
    
    return false;
}

std::vector<KademliaContact> KBucket::get_contacts() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return contacts_;
}

std::vector<KademliaContact> KBucket::get_closest_contacts(const NodeID& target, size_t count) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<KademliaContact> result = contacts_;
    
    // Sort by XOR distance to target
    std::sort(result.begin(), result.end(),
        [&target](const KademliaContact& a, const KademliaContact& b) {
            return calculate_xor_distance_bits(a.node_id, target) < calculate_xor_distance_bits(b.node_id, target);
        });
    
    if (result.size() > count) {
        result.resize(count);
    }
    
    return result;
}

bool KBucket::contains(const NodeID& node_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return std::any_of(contacts_.begin(), contacts_.end(),
        [&node_id](const KademliaContact& contact) {
            return contact.node_id == node_id;
        });
}

void KBucket::refresh_stale_contacts(uint32_t timeout_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    contacts_.erase(
        std::remove_if(contacts_.begin(), contacts_.end(),
            [timeout_ms](const KademliaContact& contact) {
                return contact.is_stale(timeout_ms) || contact.failures > 3;
            }),
        contacts_.end());
}

void KBucket::move_to_tail(size_t index) {
    if (index < contacts_.size() - 1) {
        KademliaContact contact = contacts_[index];
        contacts_.erase(contacts_.begin() + index);
        contacts_.push_back(contact);
    }
}

// KademliaRoutingTable implementation
KademliaRoutingTable::KademliaRoutingTable(const NodeID& local_node_id) 
    : local_node_id_(local_node_id) {
    
    for (size_t i = 0; i < buckets_.size(); ++i) {
        buckets_[i] = std::make_unique<KBucket>();
    }
}

bool KademliaRoutingTable::add_contact(const KademliaContact& contact) {
    if (contact.node_id == local_node_id_) {
        return false;  // Don't add ourselves
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t bucket_index = get_bucket_index(contact.node_id);
    return buckets_[bucket_index]->add_contact(contact);
}

bool KademliaRoutingTable::remove_contact(const NodeID& node_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t bucket_index = get_bucket_index(node_id);
    return buckets_[bucket_index]->remove_contact(node_id);
}

bool KademliaRoutingTable::update_contact(const KademliaContact& contact) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t bucket_index = get_bucket_index(contact.node_id);
    return buckets_[bucket_index]->update_contact(contact);
}

std::vector<KademliaContact> KademliaRoutingTable::find_closest_contacts(const NodeID& target, size_t count) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<KademliaContact> candidates;
    
    // Start with the bucket that should contain the target
    size_t target_bucket = get_bucket_index(target);
    
    // Add contacts from target bucket and surrounding buckets
    for (int i = static_cast<int>(target_bucket); i >= 0 && candidates.size() < count * 2; --i) {
        auto bucket_contacts = buckets_[i]->get_contacts();
        candidates.insert(candidates.end(), bucket_contacts.begin(), bucket_contacts.end());
    }
    
    for (size_t i = target_bucket + 1; i < buckets_.size() && candidates.size() < count * 2; ++i) {
        auto bucket_contacts = buckets_[i]->get_contacts();
        candidates.insert(candidates.end(), bucket_contacts.begin(), bucket_contacts.end());
    }
    
    // Sort by distance to target
    std::sort(candidates.begin(), candidates.end(),
        [&target](const KademliaContact& a, const KademliaContact& b) {
            return calculate_xor_distance_bits(a.node_id, target) < calculate_xor_distance_bits(b.node_id, target);
        });
    
    if (candidates.size() > count) {
        candidates.resize(count);
    }
    
    return candidates;
}

std::vector<KademliaContact> KademliaRoutingTable::get_all_contacts() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<KademliaContact> all_contacts;
    
    for (const auto& bucket : buckets_) {
        auto bucket_contacts = bucket->get_contacts();
        all_contacts.insert(all_contacts.end(), bucket_contacts.begin(), bucket_contacts.end());
    }
    
    return all_contacts;
}

void KademliaRoutingTable::refresh_buckets() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& bucket : buckets_) {
        bucket->refresh_stale_contacts(900000);  // 15 minutes
    }
}

size_t KademliaRoutingTable::get_bucket_index(const NodeID& node_id) const {
    uint32_t distance_bits = calculate_xor_distance_bits(local_node_id_, node_id);
    return std::min(distance_bits, static_cast<uint32_t>(buckets_.size() - 1));
}

size_t KademliaRoutingTable::calculate_distance_bits(const NodeID& a, const NodeID& b) const {
    return calculate_xor_distance_bits(a, b);
}

size_t KademliaRoutingTable::total_contacts() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t total = 0;
    for (const auto& bucket : buckets_) {
        total += bucket->size();
    }
    
    return total;
}

void KademliaRoutingTable::print_statistics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::cout << "=== Kademlia Routing Table Statistics ===" << std::endl;
    std::cout << "Local Node ID: " << format_node_id(local_node_id_) << std::endl;
    std::cout << "Total contacts: " << total_contacts() << std::endl;
    
    size_t non_empty_buckets = 0;
    for (size_t i = 0; i < buckets_.size(); ++i) {
        if (buckets_[i]->size() > 0) {
            non_empty_buckets++;
        }
    }
    
    std::cout << "Non-empty buckets: " << non_empty_buckets << "/" << buckets_.size() << std::endl;
    std::cout << "=========================================" << std::endl;
}

// KademliaDHT implementation
KademliaDHT::KademliaDHT(const NodeID& node_id, const std::string& bind_ip, uint16_t bind_port, const KademliaConfig& config)
    : node_id_(node_id)
    , bind_ip_(bind_ip)
    , bind_port_(bind_port)
    , config_(config)
    , routing_table_(std::make_unique<KademliaRoutingTable>(node_id))
    , running_(false) {
    
    std::cout << "Kademlia DHT initialized with node ID: " << format_node_id(node_id_) << std::endl;
}

KademliaDHT::~KademliaDHT() {
    stop();
}

bool KademliaDHT::start() {
    if (running_.load()) {
        return false;
    }
    
    running_ = true;
    
    // Start maintenance thread
    maintenance_thread_ = std::thread(&KademliaDHT::maintenance_loop, this);
    
    std::cout << "Kademlia DHT started on " << bind_ip_ << ":" << bind_port_ << std::endl;
    return true;
}

void KademliaDHT::stop() {
    if (!running_.load()) {
        return;
    }
    
    running_ = false;
    
    if (maintenance_thread_.joinable()) {
        maintenance_thread_.join();
    }
    
    std::cout << "Kademlia DHT stopped" << std::endl;
}

void KademliaDHT::bootstrap(const std::vector<std::pair<std::string, uint16_t>>& bootstrap_nodes) {
    std::cout << "Bootstrapping DHT with " << bootstrap_nodes.size() << " nodes" << std::endl;
    
    // Add bootstrap nodes to routing table
    for (const auto& node : bootstrap_nodes) {
        // Generate a temporary node ID for bootstrap nodes
        NodeID bootstrap_id = hash_to_node_id(node.first + std::to_string(node.second));
        KademliaContact contact(bootstrap_id, node.first, node.second);
        
        routing_table_->add_contact(contact);
        
        // Ping bootstrap node
        ping(contact);
    }
    
    // Perform bootstrap lookup for our own ID
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));  // Wait for pings
    perform_bootstrap_lookup();
}

std::vector<KademliaContact> KademliaDHT::find_node(const NodeID& target) {
    std::string lookup_id = start_lookup(target);
    
    // Wait for lookup completion
    auto start_time = std::chrono::steady_clock::now();
    while (running_.load()) {
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            auto it = active_lookups_.find(lookup_id);
            if (it == active_lookups_.end() || it->second->completed) {
                break;
            }
        }
        
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_time);
        
        if (elapsed.count() > config_.lookup_timeout_ms) {
            complete_lookup(lookup_id);
            break;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // Return results
    std::lock_guard<std::mutex> lock(state_mutex_);
    auto it = active_lookups_.find(lookup_id);
    if (it != active_lookups_.end()) {
        auto result = it->second->closest_nodes;
        active_lookups_.erase(it);
        return result;
    }
    
    return {};
}

bool KademliaDHT::ping(const KademliaContact& contact) {
    DHTMessage message;
    message.type = DHTMessageType::PING;
    message.sender_id = node_id_;
    message.transaction_id = generate_transaction_id();
    
    // Record pending ping
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        pending_pings_[message.transaction_id] = std::chrono::steady_clock::now();
    }
    
    send_message(contact.ip_address, contact.port, message);
    
    std::cout << "DHT: Sent PING to " << format_node_id(contact.node_id) << std::endl;
    return true;
}

bool KademliaDHT::store(const NodeID& key, const std::vector<uint8_t>& value) {
    if (!config_.enable_value_storage) {
        return false;
    }
    
    // Find closest nodes to the key
    auto closest_nodes = routing_table_->find_closest_contacts(key, KADEMLIA_BUCKET_SIZE);
    
    if (closest_nodes.empty()) {
        return false;
    }
    
    // Store locally if we're among the closest
    bool stored_locally = false;
    for (const auto& contact : closest_nodes) {
        if (calculate_xor_distance_bits(contact.node_id, key) > calculate_xor_distance_bits(node_id_, key)) {
            stored_locally = true;
            break;
        }
    }
    
    if (stored_locally) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        
        DHTStoreItem item;
        item.key = key;
        item.value = value;
        item.publisher = node_id_;
        item.expiry = std::chrono::steady_clock::now() + std::chrono::milliseconds(config_.expire_time_ms);
        
        stored_data_[node_id_to_string(key)] = item;
        stats_.stored_items = stored_data_.size();
    }
    
    // Send STORE messages to closest nodes
    for (const auto& contact : closest_nodes) {
        DHTMessage message;
        message.type = DHTMessageType::STORE;
        message.sender_id = node_id_;
        message.target_id = key;
        message.transaction_id = generate_transaction_id();
        message.payload = value;
        
        send_message(contact.ip_address, contact.port, message);
    }
    
    return true;
}

std::vector<uint8_t> KademliaDHT::find_value(const NodeID& key) {
    // Check local storage first
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        auto it = stored_data_.find(node_id_to_string(key));
        if (it != stored_data_.end() && !it->second.is_expired()) {
            return it->second.value;
        }
    }
    
    // Perform network lookup
    auto closest_nodes = find_node(key);
    
    // Query closest nodes for the value
    for (const auto& contact : closest_nodes) {
        DHTMessage message;
        message.type = DHTMessageType::FIND_VALUE;
        message.sender_id = node_id_;
        message.target_id = key;
        message.transaction_id = generate_transaction_id();
        
        send_message(contact.ip_address, contact.port, message);
    }
    
    // In a full implementation, would wait for responses
    // For now, return empty if not found locally
    return {};
}

void KademliaDHT::handle_message(const DHTMessage& message, const std::string& sender_ip, uint16_t sender_port) {
    stats_.messages_received++;
    
    // Update routing table with sender info
    KademliaContact sender_contact(message.sender_id, sender_ip, sender_port);
    routing_table_->add_contact(sender_contact);
    
    switch (message.type) {
        case DHTMessageType::PING:
            handle_ping(message, sender_ip, sender_port);
            break;
        case DHTMessageType::PONG:
            handle_pong(message, sender_ip, sender_port);
            break;
        case DHTMessageType::FIND_NODE:
            handle_find_node(message, sender_ip, sender_port);
            break;
        case DHTMessageType::FIND_NODE_RESPONSE:
            handle_find_node_response(message, sender_ip, sender_port);
            break;
        case DHTMessageType::STORE:
            handle_store(message, sender_ip, sender_port);
            break;
        case DHTMessageType::STORE_RESPONSE:
            handle_store_response(message, sender_ip, sender_port);
            break;
        case DHTMessageType::FIND_VALUE:
            handle_find_value(message, sender_ip, sender_port);
            break;
        case DHTMessageType::FIND_VALUE_RESPONSE:
            handle_find_value_response(message, sender_ip, sender_port);
            break;
    }
}

void KademliaDHT::handle_ping(const DHTMessage& message, const std::string& sender_ip, uint16_t sender_port) {
    // Send PONG response
    DHTMessage response;
    response.type = DHTMessageType::PONG;
    response.sender_id = node_id_;
    response.transaction_id = message.transaction_id;
    
    send_message(sender_ip, sender_port, response);
}

void KademliaDHT::handle_pong(const DHTMessage& message, const std::string& sender_ip, uint16_t sender_port) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    auto it = pending_pings_.find(message.transaction_id);
    if (it != pending_pings_.end()) {
        auto rtt = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - it->second);
        
        // Update contact with RTT
        KademliaContact contact(message.sender_id, sender_ip, sender_port);
        contact.rtt_ms = rtt.count();
        contact.update_last_seen();
        
        routing_table_->update_contact(contact);
        pending_pings_.erase(it);
        
        std::cout << "DHT: Received PONG from " << format_node_id(message.sender_id) 
                  << " (RTT: " << rtt.count() << "ms)" << std::endl;
    }
}

void KademliaDHT::handle_find_node(const DHTMessage& message, const std::string& sender_ip, uint16_t sender_port) {
    auto closest_contacts = routing_table_->find_closest_contacts(message.target_id, KADEMLIA_BUCKET_SIZE);
    
    DHTMessage response;
    response.type = DHTMessageType::FIND_NODE_RESPONSE;
    response.sender_id = node_id_;
    response.transaction_id = message.transaction_id;
    
    // Serialize contacts into payload (simplified)
    // In production, would use proper serialization
    
    send_message(sender_ip, sender_port, response);
}

void KademliaDHT::handle_find_node_response(const DHTMessage& message, const std::string& sender_ip, uint16_t sender_port) {
    // Process response for active lookups
    // In production, would deserialize contacts from payload and continue lookup
}

void KademliaDHT::handle_store(const DHTMessage& message, const std::string& sender_ip, uint16_t sender_port) {
    if (!config_.enable_value_storage) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    if (stored_data_.size() >= config_.max_stored_items) {
        return;  // Storage full
    }
    
    DHTStoreItem item;
    item.key = message.target_id;
    item.value = message.payload;
    item.publisher = message.sender_id;
    item.expiry = std::chrono::steady_clock::now() + std::chrono::milliseconds(config_.expire_time_ms);
    
    stored_data_[node_id_to_string(message.target_id)] = item;
    stats_.stored_items = stored_data_.size();
    
    // Send acknowledgment
    DHTMessage response;
    response.type = DHTMessageType::STORE_RESPONSE;
    response.sender_id = node_id_;
    response.transaction_id = message.transaction_id;
    
    send_message(sender_ip, sender_port, response);
}

void KademliaDHT::handle_store_response(const DHTMessage& message, const std::string& sender_ip, uint16_t sender_port) {
    // Handle store acknowledgment
}

void KademliaDHT::handle_find_value(const DHTMessage& message, const std::string& sender_ip, uint16_t sender_port) {
    DHTMessage response;
    response.type = DHTMessageType::FIND_VALUE_RESPONSE;
    response.sender_id = node_id_;
    response.transaction_id = message.transaction_id;
    
    // Check if we have the value
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        auto it = stored_data_.find(node_id_to_string(message.target_id));
        if (it != stored_data_.end() && !it->second.is_expired()) {
            response.payload = it->second.value;
            send_message(sender_ip, sender_port, response);
            return;
        }
    }
    
    // Return closest nodes instead
    auto closest_contacts = routing_table_->find_closest_contacts(message.target_id, KADEMLIA_BUCKET_SIZE);
    // Serialize contacts into payload
    
    send_message(sender_ip, sender_port, response);
}

void KademliaDHT::handle_find_value_response(const DHTMessage& message, const std::string& sender_ip, uint16_t sender_port) {
    // Process find value response
}

std::string KademliaDHT::start_lookup(const NodeID& target) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    std::string lookup_id = generate_transaction_id();
    auto lookup_state = std::make_unique<LookupState>(target);
    
    // Initialize with closest known contacts
    lookup_state->closest_nodes = routing_table_->find_closest_contacts(target, KADEMLIA_ALPHA);
    
    active_lookups_[lookup_id] = std::move(lookup_state);
    
    stats_.lookups_performed++;
    
    return lookup_id;
}

void KademliaDHT::continue_lookup(const std::string& lookup_id) {
    // Continue iterative lookup process
    // In production, would implement full iterative lookup algorithm
}

void KademliaDHT::complete_lookup(const std::string& lookup_id) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    auto it = active_lookups_.find(lookup_id);
    if (it != active_lookups_.end()) {
        it->second->completed = true;
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - it->second->start_time);
        
        // Update statistics
        if (stats_.lookups_performed == 1) {
            stats_.average_lookup_time_ms = duration.count();
        } else {
            stats_.average_lookup_time_ms = (stats_.average_lookup_time_ms * (stats_.lookups_performed - 1) + duration.count()) / stats_.lookups_performed;
        }
        
        stats_.lookups_successful++;
    }
}

std::string KademliaDHT::generate_transaction_id() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<uint32_t> dis;
    
    return std::to_string(dis(gen));
}

std::string KademliaDHT::node_id_to_string(const NodeID& id) const {
    return std::string(reinterpret_cast<const char*>(id.data()), id.size());
}

NodeID KademliaDHT::string_to_node_id(const std::string& str) const {
    NodeID id;
    id.fill(0);
    
    size_t copy_size = std::min(str.size(), id.size());
    std::copy(str.begin(), str.begin() + copy_size, id.begin());
    
    return id;
}

NodeID KademliaDHT::calculate_distance(const NodeID& a, const NodeID& b) const {
    NodeID distance;
    for (size_t i = 0; i < KADEMLIA_KEY_SIZE; ++i) {
        distance[i] = a[i] ^ b[i];
    }
    return distance;
}

void KademliaDHT::send_message(const std::string& ip, uint16_t port, const DHTMessage& message) {
    if (message_sender_) {
        message_sender_(ip, port, message);
        stats_.messages_sent++;
    }
}

void KademliaDHT::maintenance_loop() {
    while (running_.load()) {
        // Refresh buckets
        refresh_buckets();
        
        // Republish data
        if (config_.enable_value_storage) {
            republish_data();
        }
        
        // Expire old data
        expire_data();
        
        // Sleep for a while
        std::this_thread::sleep_for(std::chrono::minutes(5));
    }
}

void KademliaDHT::refresh_buckets() {
    routing_table_->refresh_buckets();
}

void KademliaDHT::republish_data() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    auto now = std::chrono::steady_clock::now();
    
    for (auto& pair : stored_data_) {
        auto& item = pair.second;
        
        // Check if item needs republishing
        auto age = std::chrono::duration_cast<std::chrono::milliseconds>(now - (item.expiry - std::chrono::milliseconds(config_.expire_time_ms)));
        
        if (age.count() > config_.republish_interval_ms) {
            // Republish the item
            store(item.key, item.value);
        }
    }
}

void KademliaDHT::expire_data() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    auto it = stored_data_.begin();
    while (it != stored_data_.end()) {
        if (it->second.is_expired()) {
            it = stored_data_.erase(it);
        } else {
            ++it;
        }
    }
    
    stats_.stored_items = stored_data_.size();
}

std::vector<KademliaContact> KademliaDHT::get_known_nodes() const {
    return routing_table_->get_all_contacts();
}

size_t KademliaDHT::get_stored_items_count() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return stored_data_.size();
}

void KademliaDHT::perform_bootstrap_lookup() {
    // Perform lookup for our own node ID to populate routing table
    find_node(node_id_);
}

KademliaDHT::DHTStats KademliaDHT::get_statistics() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    DHTStats stats = stats_;
    stats.total_nodes = routing_table_->total_contacts();
    
    return stats;
}

void KademliaDHT::print_statistics() const {
    auto stats = get_statistics();
    
    std::cout << "\n=== Kademlia DHT Statistics ===" << std::endl;
    std::cout << "Node ID: " << format_node_id(node_id_) << std::endl;
    std::cout << "Known nodes: " << stats.total_nodes << std::endl;
    std::cout << "Stored items: " << stats.stored_items << std::endl;
    std::cout << "Lookups performed: " << stats.lookups_performed << std::endl;
    std::cout << "Lookups successful: " << stats.lookups_successful << std::endl;
    std::cout << "Messages sent: " << stats.messages_sent << std::endl;
    std::cout << "Messages received: " << stats.messages_received << std::endl;
    std::cout << "Average lookup time: " << stats.average_lookup_time_ms << " ms" << std::endl;
    std::cout << "==============================\n" << std::endl;
}

} // namespace p2p
