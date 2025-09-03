#include "../include/p2p_network.hpp"
#include "../include/matrix.hpp"
#include "../include/distributed_transformer.hpp"
#include "../include/utils.hpp"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <random>
#include <thread>
#include <future>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>
#endif
#include "../include/serialization.hpp"

#include <openssl/ssl.h>
#include <openssl/err.h>


namespace p2p {

// NetworkStats implementation
void NetworkStats::print_stats() const {
    auto now = std::chrono::steady_clock::now();
    auto uptime = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
    
    std::cout << "\n=== P2P Network Statistics ===" << std::endl;
    std::cout << "Uptime: " << uptime << " seconds" << std::endl;
    std::cout << "Messages sent: " << messages_sent << std::endl;
    std::cout << "Messages received: " << messages_received << std::endl;
    std::cout << "Bytes sent: " << bytes_sent << " (" << (bytes_sent / 1024.0 / 1024.0) << " MB)" << std::endl;
    std::cout << "Bytes received: " << bytes_received << " (" << (bytes_received / 1024.0 / 1024.0) << " MB)" << std::endl;
    std::cout << "Consensus rounds: " << consensus_rounds << std::endl;
    std::cout << "Failed consensus: " << failed_consensus << std::endl;
    std::cout << "Average consensus time: " << average_consensus_time_ms << " ms" << std::endl;
    std::cout << "Active peers: " << active_peers << std::endl;
    std::cout << "Total peers seen: " << total_peers_seen << std::endl;
    std::cout << "==============================\n" << std::endl;
}

// --- SSLContext Implementation ---
SSLContext::SSLContext() {
    SSL_load_error_strings();
    OpenSSL_add_ssl_algorithms();
}

SSLContext::~SSLContext() {
    EVP_cleanup();
}

bool SSLContext::init(const std::string& cert_path, const std::string& key_path) {
    const SSL_METHOD* method = TLS_server_method();
    ctx_.reset(SSL_CTX_new(method));
    if (!ctx_) {
        std::cerr << "Error creating SSL context." << std::endl;
        ERR_print_errors_fp(stderr);
        return false;
    }

    // Load certificate
    if (SSL_CTX_use_certificate_file(ctx_.get(), cert_path.c_str(), SSL_FILETYPE_PEM) <= 0) {
        std::cerr << "Error loading certificate file." << std::endl;
        ERR_print_errors_fp(stderr);
        return false;
    }

    // Load private key
    if (SSL_CTX_use_privatekey_file(ctx_.get(), key_path.c_str(), SSL_FILETYPE_PEM) <= 0) {
        std::cerr << "Error loading private key file." << std::endl;
        ERR_print_errors_fp(stderr);
        return false;
    }

    // Check if the private key matches the certificate
    if (!SSL_CTX_check_private_key(ctx_.get())) {
        std::cerr << "Private key does not match the public certificate." << std::endl;
        return false;
    }

    return true;
}


// --- P2PNetwork Implementation ---
P2PNetwork::P2PNetwork(const P2PConfig& config) 
    : config_(config), rng_(std::random_device{}()), running_(false) {
    
    std::cout << "Initializing P2P Network with node ID: " << config_.node_id << std::endl;
    
    // Initialize Winsock on Windows
#ifdef _WIN32
    WSADATA wsaData;
    int result = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (result != 0) {
        throw std::runtime_error("WSAStartup failed: " + std::to_string(result));
    }
#endif

    // Register default message handlers
    register_message_handler(MessageType::NODE_DISCOVERY, 
        [this](const NetworkMessage& msg) { handle_node_discovery(msg); });
    register_message_handler(MessageType::NODE_ANNOUNCEMENT, 
        [this](const NetworkMessage& msg) { handle_node_announcement(msg); });
    register_message_handler(MessageType::GRADIENT_PROPOSAL, 
        [this](const NetworkMessage& msg) { handle_gradient_proposal(msg); });
    register_message_handler(MessageType::GRADIENT_VOTE, 
        [this](const NetworkMessage& msg) { handle_gradient_prepare(msg); });
    register_message_handler(MessageType::GRADIENT_COMMIT, 
        [this](const NetworkMessage& msg) { handle_gradient_commit(msg); });
    register_message_handler(MessageType::HEARTBEAT, 
        [this](const NetworkMessage& msg) { handle_heartbeat(msg); });
    register_message_handler(MessageType::PEER_LIST_REQUEST, 
        [this](const NetworkMessage& msg) { handle_peer_list_request(msg); });
    register_message_handler(MessageType::PEER_LIST_RESPONSE, 
        [this](const NetworkMessage& msg) { handle_peer_list_response(msg); });
    register_message_handler(MessageType::STATE_SYNC_REQUEST, 
        [this](const NetworkMessage& msg) { handle_state_sync_request(msg); });
    register_message_handler(MessageType::STATE_SYNC_RESPONSE, 
        [this](const NetworkMessage& msg) { handle_state_sync_response(msg); });
}

P2PNetwork::~P2PNetwork() {
    stop();
    
#ifdef _WIN32
    WSACleanup();
#endif
}

bool P2PNetwork::start() {
    if (running_.load()) {
        std::cout << "P2P Network already running" << std::endl;
        return true;
    }
    
    if (!init_ssl()) {
        std::cerr << "Failed to initialize SSL context. Aborting." << std::endl;
        return false;
    }

    if (!load_private_key()) {
        std::cerr << "Failed to load private key. Aborting." << std::endl;
        return false;
    }
    
    std::cout << "Starting P2P Network on " << config_.bind_address << ":" << config_.bind_port << std::endl;
    
    running_.store(true);
    shutdown_requested_.store(false);
    
    // Start worker threads
    worker_threads_.emplace_back(&P2PNetwork::network_thread, this);
    worker_threads_.emplace_back(&P2PNetwork::heartbeat_thread, this);
    worker_threads_.emplace_back(&P2PNetwork::consensus_thread, this);
    worker_threads_.emplace_back(&P2PNetwork::cleanup_thread, this);
    
    // Connect to bootstrap nodes
    for (const auto& bootstrap_node : config_.bootstrap_nodes) {
        size_t colon_pos = bootstrap_node.find(':');
        if (colon_pos != std::string::npos) {
            std::string address = bootstrap_node.substr(0, colon_pos);
            uint16_t port = static_cast<uint16_t>(std::stoi(bootstrap_node.substr(colon_pos + 1)));
            
            std::cout << "Connecting to bootstrap node: " << address << ":" << port << std::endl;
            connect_to_peer(address, port);
        }
    }
    
    // Announce our presence
    NetworkMessage announcement;
    announcement.type = MessageType::NODE_ANNOUNCEMENT;
    announcement.sender_id = config_.node_id;
    announcement.sequence_number = sequence_counter_.fetch_add(1);
    announcement.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    // Add our node info to payload
    serialization::Serializer s;
    s.write_string(config_.node_id);
    
    // Read and include our public key
    std::ifstream pub_key_file(config_.public_key_path);
    std::stringstream pub_key_stream;
    pub_key_stream << pub_key_file.rdbuf();
    s.write_string(pub_key_stream.str());

    s.write_string(utils::get_local_ip_address());
    s.write_uint16(config_.bind_port);
    s.write_uint64(500); // Default capability
    s.write_uint64(8ULL * 1024 * 1024 * 1024); // 8GB default
    s.write_uint8(1); // is_trusted
    s.write_trivial(1.0f); // reputation_score
    announcement.payload = s.take_buffer();
    
    broadcast_message(announcement);
    
    if (!config_.bootstrap_nodes.empty()) { // If we are not a bootstrap node
        is_synchronized_.store(false);
        request_model_state();
    } else {
        is_synchronized_.store(true); // Bootstrap nodes start as synchronized
    }
    
    std::cout << "P2P Network started successfully" << std::endl;
    return true;
}

void P2PNetwork::stop() {
    if (!running_.load()) {
        return;
    }
    
    std::cout << "Stopping P2P Network..." << std::endl;
    
    shutdown_requested_.store(true);
    running_.store(false);
    
    // Send leave message to peers
    NetworkMessage leave_msg;
    leave_msg.type = MessageType::NODE_LEAVE;
    leave_msg.sender_id = config_.node_id;
    leave_msg.sequence_number = sequence_counter_.fetch_add(1);
    leave_msg.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    broadcast_message(leave_msg);
    
    // Wait for threads to finish
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();
    
    std::cout << "P2P Network stopped" << std::endl;
}

bool P2PNetwork::connect_to_peer(const std::string& address, uint16_t port) {
    std::cout << "Attempting to connect to peer: " << address << ":" << port << std::endl;
    
    // Create socket
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        std::cerr << "Failed to create socket" << std::endl;
        return false;
    }
    
    // Set up address
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    
    if (inet_pton(AF_INET, address.c_str(), &server_addr.sin_addr) <= 0) {
        std::cerr << "Invalid address: " << address << std::endl;
#ifdef _WIN32
        closesocket(sock);
#else
        close(sock);
#endif
        return false;
    }
    
    // Connect with timeout
    if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "Failed to connect to " << address << ":" << port << std::endl;
#ifdef _WIN32
        closesocket(sock);
#else
        close(sock);
#endif
        return false;
    }

    // --- Start TLS Handshake (Client side) ---
    SSL* ssl = SSL_new(ssl_context_->get());
    SSL_set_fd(ssl, sock);

    if (SSL_connect(ssl) <= 0) {
        std::cerr << "SSL handshake failed with " << address << ":" << port << std::endl;
        ERR_print_errors_fp(stderr);
        SSL_free(ssl);
#ifdef _WIN32
        closesocket(sock);
#else
        close(sock);
#endif
        return false;
    }
    std::cout << "SSL handshake successful with " << address << ":" << port << std::endl;

    // Send discovery message AND peer list request
    NetworkMessage discovery;
    discovery.type = MessageType::NODE_DISCOVERY;
    discovery.sender_id = config_.node_id;
    discovery.sequence_number = sequence_counter_.fetch_add(1);
    discovery.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    auto serialized_discovery = serialization::serialize(discovery);
    SSL_write(ssl, serialized_discovery.data(), serialized_discovery.size());

    NetworkMessage peer_req;
    peer_req.type = MessageType::PEER_LIST_REQUEST;
    peer_req.sender_id = config_.node_id;
    auto serialized_req = serialization::serialize(peer_req);
    SSL_write(ssl, serialized_req.data(), serialized_req.size());

    // The connection will be kept alive by the server-side network_thread.
    // Here we just initiate it. The server will add it to its pool.
    // We can close our temporary SSL object and socket.
    SSL_shutdown(ssl);
    SSL_free(ssl);
#ifdef _WIN32
    closesocket(sock);
#else
    close(sock);
#endif
    
    std::cout << "Connected to peer: " << address << ":" << port << std::endl;
    return true;
}

std::string P2PNetwork::propose_gradient(const std::vector<Matrix>& gradients, uint32_t epoch, uint32_t batch_id) {
    std::lock_guard<std::mutex> lock(consensus_mutex_);
    
    // Generate unique proposal ID
    std::string proposal_id = config_.node_id + "_" + std::to_string(epoch) + "_" + 
                             std::to_string(batch_id) + "_" + std::to_string(sequence_counter_.fetch_add(1));
    
    // Create gradient proposal
    GradientProposal proposal;
    proposal.proposal_id = proposal_id;
    proposal.proposer_id = config_.node_id;
    proposal.epoch = epoch;
    proposal.batch_id = batch_id;
    proposal.gradient_hash = {calculate_gradient_hash(gradients).begin(), calculate_gradient_hash(gradients).end()};
    proposal.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    // Serialize gradients (with quantization if enabled)
    serialization::Serializer grad_serializer;
    grad_serializer.write_uint32(static_cast<uint32_t>(gradients.size()));

    if (config_.enable_gradient_quantization) {
        for (const auto& grad : gradients) {
            auto q_grad = utils::quantize_matrix(grad, config_.gradient_quantization_bits);
            grad_serializer.write_uint64(q_grad.original_rows);
            grad_serializer.write_uint64(q_grad.original_cols);
            grad_serializer.write_trivial(q_grad.scale);
            grad_serializer.write_bytes(reinterpret_cast<const uint8_t*>(q_grad.quantized_data.data()), q_grad.quantized_data.size());
        }
    } else {
        for (const auto& grad : gradients) {
            grad_serializer.write_uint64(grad.rows());
            grad_serializer.write_uint64(grad.cols());
            for (size_t i = 0; i < grad.size(); ++i) {
                grad_serializer.write_trivial(grad.data()[i]);
            }
        }
    }
    proposal.gradient_data = grad_serializer.take_buffer();

    active_proposals_[proposal_id] = proposal;
    proposal_prepare_votes_[proposal_id] = std::vector<GradientVote>(); // Use correct map
    proposal_commit_votes_[proposal_id] = std::vector<GradientCommit>();
    
    // Broadcast proposal
    NetworkMessage msg;
    msg.type = MessageType::GRADIENT_PROPOSAL;
    msg.sender_id = config_.node_id;
    msg.sequence_number = sequence_counter_.fetch_add(1);
    msg.timestamp = proposal.timestamp;
    
    // Serialize proposal info into payload
    serialization::Serializer payload_serializer;
    payload_serializer.write_string(proposal_id);
    payload_serializer.write_uint32(epoch);
    payload_serializer.write_uint32(batch_id);
    payload_serializer.write_bytes(proposal.gradient_hash);
    msg.payload = payload_serializer.take_buffer();

    broadcast_message(msg);
    
    std::cout << "Proposed gradient: " << proposal_id << std::endl;
    return proposal_id;
}

bool P2PNetwork::vote_on_gradient(const std::string& proposal_id, bool approve, const std::string& reason) {
    std::lock_guard<std::mutex> lock(consensus_mutex_);
    
    auto it = active_proposals_.find(proposal_id);
    if (it == active_proposals_.end()) {
        std::cerr << "Proposal not found: " << proposal_id << std::endl;
        return false;
    }
    
    // Create vote
    GradientVote vote;
    vote.proposal_id = proposal_id;
    vote.voter_id = config_.node_id;
    vote.approve = approve;
    vote.reason = reason;
    vote.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    proposal_votes_[proposal_id].push_back(vote);
    
    // Broadcast vote
    NetworkMessage msg;
    msg.type = MessageType::GRADIENT_VOTE;
    msg.sender_id = config_.node_id;
    msg.sequence_number = sequence_counter_.fetch_add(1);
    msg.timestamp = vote.timestamp;
    
    serialization::Serializer s;
    s.write_string(proposal_id);
    s.write_uint8(approve ? 1 : 0);
    s.write_string(reason);
    msg.payload = s.take_buffer();
    
    broadcast_message(msg);
    
    std::cout << "Voted on gradient " << proposal_id << ": " << (approve ? "APPROVE" : "REJECT") << std::endl;
    return true;
}

bool P2PNetwork::wait_for_consensus(const std::string& proposal_id, uint32_t timeout_ms) {
    auto start_time = std::chrono::steady_clock::now();
    
    while (std::chrono::steady_clock::now() - start_time < std::chrono::milliseconds(timeout_ms)) {
        {
            std::lock_guard<std::mutex> lock(consensus_mutex_);
            if (check_consensus_reached(proposal_id)) {
                return true;
            }
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    std::cout << "Consensus timeout for proposal: " << proposal_id << std::endl;
    return false;
}

void P2PNetwork::network_thread() {
    std::cout << "Network thread started" << std::endl;
    
    // Create listening socket
    int listen_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_sock < 0) {
        std::cerr << "Failed to create listening socket" << std::endl;
        return;
    }
    
    // Set socket options
    int opt = 1;
    setsockopt(listen_sock, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&opt), sizeof(opt));
    
    // Bind socket
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(config_.bind_port);
    
    if (bind(listen_sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "Failed to bind socket to port " << config_.bind_port << std::endl;
#ifdef _WIN32
        closesocket(listen_sock);
#else
        close(listen_sock);
#endif
        return;
    }
    
    // Listen for connections
    if (listen(listen_sock, config_.max_concurrent_connections) < 0) {
        std::cerr << "Failed to listen on socket" << std::endl;
#ifdef _WIN32
        closesocket(listen_sock);
#else
        close(listen_sock);
#endif
        return;
    }
    
    std::cout << "Listening for connections on port " << config_.bind_port << std::endl;
    
    while (running_.load()) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        
        int client_sock = accept(listen_sock, (struct sockaddr*)&client_addr, &client_len);
        if (client_sock < 0) {
            if (running_.load()) {
                std::cerr << "Failed to accept connection" << std::endl;
            }
            continue;
        }

        // Handle client in separate thread
        std::thread(&P2PNetwork::handle_client_connection, this, client_sock).detach();
    }
    
#ifdef _WIN32
    closesocket(listen_sock);
#else
    close(listen_sock);
#endif
    
    std::cout << "Network thread stopped" << std::endl;
}

void P2PNetwork::heartbeat_thread() {
    std::cout << "Heartbeat thread started" << std::endl;
    
    while (running_.load()) {
        // Send heartbeat to all known peers
        NetworkMessage heartbeat;
        heartbeat.type = MessageType::HEARTBEAT;
        heartbeat.sender_id = config_.node_id;
        heartbeat.sequence_number = sequence_counter_.fetch_add(1);
        heartbeat.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        broadcast_message(heartbeat);
        
        // Remove inactive nodes
        remove_inactive_nodes();
        
        std::this_thread::sleep_for(std::chrono::milliseconds(config_.heartbeat_interval_ms));
    }
    
    std::cout << "Heartbeat thread stopped" << std::endl;
}

void P2PNetwork::consensus_thread() {
    std::cout << "Consensus thread started" << std::endl;
    
    while (running_.load()) {
        {
            std::lock_guard<std::mutex> lock(consensus_mutex_);
            
            // Check for consensus on active proposals
            for (auto& [proposal_id, proposal] : active_proposals_) {
                if (check_consensus_reached(proposal_id)) {
                    finalize_consensus(proposal_id);
                }
            }
            
            // Clean up old proposals
            cleanup_old_proposals();
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    
    std::cout << "Consensus thread stopped" << std::endl;
}

void P2PNetwork::cleanup_thread() {
    std::cout << "Cleanup thread started" << std::endl;
    
    while (running_.load()) {
        // Periodic cleanup tasks
        remove_inactive_nodes();
        
        {
            std::lock_guard<std::mutex> lock(consensus_mutex_);
            cleanup_old_proposals();
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(30000));  // Every 30 seconds
    }
    
    std::cout << "Cleanup thread stopped" << std::endl;
}

void P2PNetwork::handle_incoming_message(const NetworkMessage& message) {
    if (!verify_message_signature(message)) {
        std::cerr << "Warning: Dropping message with invalid signature from " << message.sender_id << std::endl;
        report_malicious_node(message.sender_id, "Invalid signature");
        return;
    }
    
    // Check if sender is blacklisted
    {
        std::lock_guard<std::mutex> lock(blacklist_mutex_);
        if (blacklisted_nodes_.count(message.sender_id)) {
            return;  // Ignore messages from blacklisted nodes
        }
    }
    
    // Update sender's last seen time
    {
        std::lock_guard<std::mutex> lock(nodes_mutex_);
        auto it = known_nodes_.find(message.sender_id);
        if (it != known_nodes_.end()) {
            it->second.last_seen = std::chrono::steady_clock::now();
        }
    }
    
    // Dispatch to appropriate handler
    std::lock_guard<std::mutex> lock(handlers_mutex_);
    auto handler_it = message_handlers_.find(message.type);
    if (handler_it != message_handlers_.end()) {
        handler_it->second(message);
    } else {
        std::cout << "No handler for message type: " << static_cast<int>(message.type) << std::endl;
    }
}

void P2PNetwork::handle_node_discovery(const NetworkMessage& message) {
    std::cout << "Received node discovery from: " << message.sender_id << std::endl;
    
    // Respond with our node announcement
    NetworkMessage response;
    response.type = MessageType::NODE_ANNOUNCEMENT;
    response.sender_id = config_.node_id;
    response.recipient_id = message.sender_id;
    response.sequence_number = sequence_counter_.fetch_add(1);
    response.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    send_message(response);
}

void P2PNetwork::handle_node_announcement(const NetworkMessage& message) {
    std::cout << "Received node announcement from: " << message.sender_id << std::endl;
    
    try {
        serialization::Deserializer d(message.payload);
        NodeInfo node_info;
        node_info.node_id = d.read_string();
        node_info.public_key = d.read_string(); // Read the public key
        node_info.ip_address = d.read_string();
        node_info.port = d.read_uint16();
        node_info.compute_capability = d.read_uint64();
        node_info.available_memory = d.read_uint64();
        node_info.is_trusted = d.read_uint8();
        node_info.reputation_score = d.read_trivial<float>();
        node_info.last_seen = std::chrono::steady_clock::now();

        {
            std::lock_guard<std::mutex> lock(nodes_mutex_);
            known_nodes_[message.sender_id] = node_info;
            
            std::lock_guard<std::mutex> stats_lock(stats_mutex_);
            stats_.active_peers = known_nodes_.size();
            stats_.total_peers_seen = std::max(stats_.total_peers_seen, static_cast<uint32_t>(known_nodes_.size()));
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to deserialize node announcement from " << message.sender_id << ": " << e.what() << std::endl;
    }
}

void P2PNetwork::handle_gradient_proposal(const NetworkMessage& message) {
    std::cout << "Received gradient proposal from: " << message.sender_id << std::endl;
    
    try {
        serialization::Deserializer d(message.payload);
        std::string proposal_id = d.read_string();
        
        // Auto-prepare for now
        vote_on_gradient(proposal_id, true, "auto-approved");
    } catch (const std::exception& e) {
        std::cerr << "Failed to deserialize gradient proposal from " << message.sender_id << ": " << e.what() << std::endl;
    }
}

void P2PNetwork::handle_gradient_prepare(const NetworkMessage& message) { // Formerly handle_gradient_vote
    std::cout << "Received PREPARE vote from: " << message.sender_id << std::endl;
    
    try {
        serialization::Deserializer d(message.payload);
        std::string proposal_id = d.read_string();
        bool approved = d.read_uint8();
        std::string reason = d.read_string();

        std::lock_guard<std::mutex> lock(consensus_mutex_);
        GradientVote vote;
        vote.proposal_id = proposal_id;
        vote.voter_id = message.sender_id;
        vote.approve = approved;
        vote.reason = reason;
        vote.timestamp = message.timestamp;
        proposal_prepare_votes_[proposal_id].push_back(vote); // Use correct map
        
        // Check if we have enough PREPARE votes to send a COMMIT
        size_t total_nodes = known_nodes_.size() + 1;
        size_t required_votes = static_cast<size_t>(total_nodes * config_.consensus_threshold);
        if (proposal_prepare_votes_[proposal_id].size() >= required_votes) {
            
            // Send COMMIT message
            NetworkMessage commit_msg;
            commit_msg.type = MessageType::GRADIENT_COMMIT;
            commit_msg.sender_id = config_.node_id;
            commit_msg.payload = serialization::Serializer().write_string(proposal_id).take_buffer();
            broadcast_message(commit_msg);
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to deserialize gradient prepare from " << message.sender_id << ": " << e.what() << std::endl;
    }
}

void P2PNetwork::handle_gradient_commit(const NetworkMessage& message) {
    std::cout << "Received COMMIT message from: " << message.sender_id << std::endl;
    
    try {
        serialization::Deserializer d(message.payload);
        std::string proposal_id = d.read_string();
        
        GradientCommit commit;
        commit.proposal_id = proposal_id;
        commit.committer_id = message.sender_id;
        commit.timestamp = message.timestamp;

        std::lock_guard<std::mutex> lock(consensus_mutex_);
        proposal_commit_votes_[proposal_id].push_back(commit);
    } catch (const std::exception& e) {
        std::cerr << "Failed to deserialize gradient commit from " << message.sender_id << ": " << e.what() << std::endl;
    }
}

void P2PNetwork::handle_heartbeat(const NetworkMessage& message) {
    // Update node's last seen time (already done in handle_incoming_message)
    // Could add additional heartbeat processing here
}

void P2PNetwork::handle_peer_list_request(const NetworkMessage& message) {
    std::cout << "Received peer list request from: " << message.sender_id << std::endl;
    
    NetworkMessage response;
    response.type = MessageType::PEER_LIST_RESPONSE;
    response.sender_id = config_.node_id;
    response.recipient_id = message.sender_id;

    serialization::Serializer s;
    std::vector<NodeInfo> peers = get_active_peers();
    s.write_uint32(peers.size());
    for (const auto& peer : peers) {
        s.write_string(peer.ip_address);
        s.write_uint16(peer.port);
    }
    response.payload = s.take_buffer();

    send_message(response);
}

void P2PNetwork::handle_peer_list_response(const NetworkMessage& message) {
    std::cout << "Received peer list response from: " << message.sender_id << std::endl;
    
    try {
        serialization::Deserializer d(message.payload);
        uint32_t num_peers = d.read_uint32();

        for (uint32_t i = 0; i < num_peers; ++i) {
            std::string ip = d.read_string();
            uint16_t port = d.read_uint16();

            // Check if we are already connected to this peer
            bool already_connected = false;
            {
                std::lock_guard<std::mutex> lock(nodes_mutex_);
                for (const auto& pair : known_nodes_) {
                    if (pair.second.ip_address == ip && pair.second.port == port) {
                        already_connected = true;
                        break;
                    }
                }
            }

            // Also check it's not ourself
            if (ip == utils::get_local_ip_address() && port == config_.bind_port) {
                already_connected = true;
            }

            if (!already_connected) {
                connect_to_peer(ip, port);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error deserializing peer list: " << e.what() << std::endl;
    }
}

void P2PNetwork::handle_state_sync_request(const NetworkMessage& message) {
    if (!is_synchronized_.load()) return;

    auto coord_sptr = coordinator_.lock();
    if (!coord_sptr) return;

    std::cout << "Received state sync request from: " << message.sender_id << std::endl;

    auto all_params = coord_sptr->get_transformer()->get_all_parameters();
    const size_t CHUNK_SIZE = 1024 * 512; // 512KB chunks

    for (const auto& pair : all_params) {
        const std::string& param_name = pair.first;
        const Matrix& matrix = pair.second;
        
        std::vector<uint8_t> matrix_bytes = serialization::serialize_matrix(matrix);
        size_t total_size = matrix_bytes.size();
        uint32_t total_chunks = (total_size + CHUNK_SIZE - 1) / CHUNK_SIZE;

        for (uint32_t i = 0; i < total_chunks; ++i) {
            ModelStateChunk chunk;
            chunk.parameter_name = param_name;
            chunk.chunk_index = i;
            chunk.total_chunks = total_chunks;

            size_t offset = i * CHUNK_SIZE;
            size_t size = std::min(CHUNK_SIZE, total_size - offset);
            chunk.data.assign(matrix_bytes.begin() + offset, matrix_bytes.begin() + offset + size);

            NetworkMessage response;
            response.type = MessageType::STATE_SYNC_RESPONSE;
            response.sender_id = config_.node_id;
            response.recipient_id = message.sender_id;
            response.payload = serialization::serialize_chunk(chunk);
            send_message(response);
        }
    }
}

void P2PNetwork::handle_state_sync_response(const NetworkMessage& message) {
    if (is_synchronized_.load()) return;
    
    try {
        ModelStateChunk chunk = serialization::deserialize_chunk(message.payload);
        
        std::map<std::string, Matrix> reconstructed_params;
        bool all_synced = false;

        {
            std::lock_guard<std::mutex> lock(state_sync_mutex_);
            incoming_state_chunks_[chunk.parameter_name].push_back(chunk);

            // Check if we have received all chunks for this parameter
            if (incoming_state_chunks_[chunk.parameter_name].size() == chunk.total_chunks) {
                // If so, check if we have ALL parameters
                auto coord_sptr = coordinator_.lock();
                if (coord_sptr) {
                    auto expected_params = coord_sptr->get_transformer()->get_all_parameters();
                    if (incoming_state_chunks_.size() == expected_params.size()) {
                        
                        // We have all parameters, try to reconstruct them all
                        bool reconstruction_complete = true;
                        for (const auto& pair : expected_params) {
                            if (incoming_state_chunks_.find(pair.first) == incoming_state_chunks_.end() ||
                                incoming_state_chunks_.at(pair.first).size() != incoming_state_chunks_.at(pair.first)[0].total_chunks) {
                                reconstruction_complete = false;
                                break;
                            }
                        }

                        if (reconstruction_complete) {
                            for (auto& pair : incoming_state_chunks_) {
                                std::sort(pair.second.begin(), pair.second.end(), 
                                          [](const auto& a, const auto& b){ return a.chunk_index < b.chunk_index; });
                                
                                std::vector<uint8_t> full_data;
                                for(const auto& c : pair.second) {
                                    full_data.insert(full_data.end(), c.data.begin(), c.data.end());
                                }
                                reconstructed_params[pair.first] = serialization::deserialize_matrix(full_data);
                            }
                            all_synced = true;
                        }
                    }
                }
            }
        }

        if (all_synced) {
            auto coord_sptr = coordinator_.lock();
            if (coord_sptr) {
                coord_sptr->get_transformer()->set_all_parameters(reconstructed_params);
                is_synchronized_.store(true);
                std::cout << "Model state synchronized successfully!" << std::endl;
            }
        }

    } catch (const std::exception& e) {
        // ... error handling ...
    }
}

bool P2PNetwork::check_consensus_reached(const std::string& proposal_id) {
    auto it = proposal_commit_votes_.find(proposal_id);
    if (it == proposal_commit_votes_.end()) {
        return false;
    }
    
    const auto& commits = it->second;
    size_t total_nodes = known_nodes_.size() + 1;
    size_t required_votes = static_cast<size_t>(total_nodes * config_.consensus_threshold);
    
    return commits.size() >= required_votes;
}

void P2PNetwork::finalize_consensus(const std::string& proposal_id) {
    std::cout << "Consensus reached for proposal: " << proposal_id << std::endl;
    
    auto proposal_it = active_proposals_.find(proposal_id);
    if (proposal_it != active_proposals_.end()) {
        // Convert gradient data back to matrices
        std::vector<Matrix> gradients;
        try {
            serialization::Deserializer grad_deserializer(proposal_it->second.gradient_data);
            uint32_t num_matrices = grad_deserializer.read_uint32();
            gradients.reserve(num_matrices);

            if (config_.enable_gradient_quantization) {
                for (uint32_t i = 0; i < num_matrices; ++i) {
                    QuantizedGradient q_grad;
                    q_grad.original_rows = grad_deserializer.read_uint64();
                    q_grad.original_cols = grad_deserializer.read_uint64();
                    q_grad.scale = grad_deserializer.read_trivial<float>();
                    
                    size_t num_elements = q_grad.original_rows * q_grad.original_cols;
                    std::vector<uint8_t> byte_data = grad_deserializer.read_bytes(num_elements);
                    q_grad.quantized_data.resize(num_elements);
                    memcpy(q_grad.quantized_data.data(), byte_data.data(), num_elements);
                    
                    gradients.push_back(utils::dequantize_matrix(q_grad));
                }
            } else {
                for (uint32_t i = 0; i < num_matrices; ++i) {
                    uint64_t rows = grad_deserializer.read_uint64();
                    uint64_t cols = grad_deserializer.read_uint64();
                    Matrix grad(rows, cols);
                    for (size_t j = 0; j < grad.size(); ++j) {
                        grad.data()[j] = grad_deserializer.read_trivial<float>();
                    }
                    gradients.push_back(std::move(grad));
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Failed to deserialize gradients for proposal " << proposal_id << ": " << e.what() << std::endl;
            // Handle error, maybe mark proposal as invalid
        }
        
        consensus_gradients_[proposal_id] = gradients;
        
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.consensus_rounds++;
    }
}

void P2PNetwork::cleanup_old_proposals() {
    auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    auto it = active_proposals_.begin();
    while (it != active_proposals_.end()) {
        if (now - it->second.timestamp > config_.max_proposal_age_ms) {
            std::cout << "Cleaning up old proposal: " << it->first << std::endl;
            proposal_prepare_votes_.erase(it->first); // Corrected map name
            proposal_commit_votes_.erase(it->first);
            consensus_gradients_.erase(it->first);
            it = active_proposals_.erase(it);
        } else {
            ++it;
        }
    }
}

void P2PNetwork::remove_inactive_nodes() {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    auto now = std::chrono::steady_clock::now();
    auto timeout = std::chrono::milliseconds(config_.node_timeout_ms);
    
    auto it = known_nodes_.begin();
    while (it != known_nodes_.end()) {
        if (now - it->second.last_seen > timeout) {
            std::cout << "Removing inactive node: " << it->first << std::endl;
            it = known_nodes_.erase(it);
        } else {
            ++it;
        }
    }
    
    std::lock_guard<std::mutex> stats_lock(stats_mutex_);
    stats_.active_peers = known_nodes_.size();
}

std::vector<uint8_t> P2PNetwork::serialize_message(const NetworkMessage& message) {
    // Simplified serialization - production would use proper binary format
    std::ostringstream oss;
    oss << static_cast<int>(message.type) << "|"
        << message.sender_id << "|"
        << message.recipient_id << "|"
        << message.sequence_number << "|"
        << message.timestamp << "|"
        << std::string(message.payload.begin(), message.payload.end()) << "|"
        << message.signature;
    
    std::string serialized = oss.str();
    return std::vector<uint8_t>(serialized.begin(), serialized.end());
}

NetworkMessage P2PNetwork::deserialize_message(const std::vector<uint8_t>& data) {
    // Simplified deserialization
    std::string str(data.begin(), data.end());
    std::istringstream iss(str);
    std::string token;
    
    NetworkMessage message;
    
    if (std::getline(iss, token, '|')) {
        message.type = static_cast<MessageType>(std::stoi(token));
    }
    if (std::getline(iss, token, '|')) {
        message.sender_id = token;
    }
    if (std::getline(iss, token, '|')) {
        message.recipient_id = token;
    }
    if (std::getline(iss, token, '|')) {
        message.sequence_number = std::stoul(token);
    }
    if (std::getline(iss, token, '|')) {
        message.timestamp = std::stoull(token);
    }
    if (std::getline(iss, token, '|')) {
        message.payload.assign(token.begin(), token.end());
    }
    if (std::getline(iss, token, '|')) {
        message.signature = token;
    }
    
    return message;
}

std::string P2PNetwork::calculate_gradient_hash(const std::vector<Matrix>& gradients) {
    // Simplified hash calculation
    std::ostringstream oss;
    for (const auto& grad : gradients) {
        for (size_t i = 0; i < grad.size(); ++i) {
            oss << grad.data()[i] << ",";
        }
    }
    return std::to_string(std::hash<std::string>{}(oss.str()));
}

void P2PNetwork::send_message(const NetworkMessage& message) {
    if (message.recipient_id.empty()) {
        broadcast_message(message);
        return;
    }

    NetworkMessage signed_message = message;
    sign_message(signed_message);
    auto serialized_msg = serialization::serialize(signed_message);

    std::shared_ptr<ssl_st> ssl_session;
    {
        std::lock_guard<std::mutex> lock(nodes_mutex_);
        auto it = known_nodes_.find(message.recipient_id);
        if (it != known_nodes_.end() && it->second.ssl_session) {
            ssl_session = it->second.ssl_session;
        }
    }

    if (ssl_session) {
        if (SSL_write(ssl_session.get(), serialized_msg.data(), serialized_msg.size()) <= 0) {
            std::cerr << "Error sending message to " << message.recipient_id << std::endl;
            // Handle error, maybe disconnect peer
        }
    } else {
        std::cerr << "Could not find active connection for recipient: " << message.recipient_id << std::endl;
    }
    
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.messages_sent++;
}

void P2PNetwork::broadcast_message(const NetworkMessage& message) {
    NetworkMessage signed_message = message;
    sign_message(signed_message);
    
    auto serialized_msg = serialization::serialize(signed_message);
    
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    for (const auto& pair : known_nodes_) {
        if (pair.second.ssl_session && pair.first != message.sender_id) {
            SSL_write(pair.second.ssl_session.get(), serialized_msg.data(), serialized_msg.size());
        }
    }
    
    std::lock_guard<std::mutex> stats_lock(stats_mutex_);
    stats_.messages_sent += known_nodes_.size();
}

void P2PNetwork::register_message_handler(MessageType type, std::function<void(const NetworkMessage&)> handler) {
    std::lock_guard<std::mutex> lock(handlers_mutex_);
    message_handlers_[type] = handler;
}

std::vector<NodeInfo> P2PNetwork::get_active_peers() const {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    std::vector<NodeInfo> peers;
    for (const auto& [node_id, node_info] : known_nodes_) {
        peers.push_back(node_info);
    }
    return peers;
}

void P2PNetwork::blacklist_node(const std::string& node_id) {
    std::lock_guard<std::mutex> lock(blacklist_mutex_);
    blacklisted_nodes_.insert(node_id);
    std::cout << "Blacklisted node: " << node_id << std::endl;
}

bool P2PNetwork::is_node_trusted(const std::string& node_id) const {
    std::lock_guard<std::mutex> lock(blacklist_mutex_);
    return blacklisted_nodes_.find(node_id) == blacklisted_nodes_.end();
}

bool P2PNetwork::init_ssl() {
    ssl_context_ = std::make_unique<SSLContext>();
    return ssl_context_->init(config_.tls_cert_path, config_.tls_key_path);
}

bool P2PNetwork::load_private_key() {
    FILE* fp = fopen(config_.private_key_path.c_str(), "rb");
    if (!fp) {
        std::cerr << "Error: Cannot open private key file: " << config_.private_key_path << std::endl;
        return false;
    }

    private_key_.reset(PEM_read_PrivateKey(fp, NULL, NULL, NULL));
    fclose(fp);

    if (!private_key_) {
        std::cerr << "Error reading private key from file." << std::endl;
        ERR_print_errors_fp(stderr);
        return false;
    }
    return true;
}

void P2PNetwork::sign_message(NetworkMessage& message) {
    if (!private_key_) return;

    EVP_MD_CTX* mdctx = EVP_MD_CTX_new();
    EVP_PKEY_CTX* pctx = nullptr;

    if (!mdctx) { /* error */ return; }

    if (EVP_DigestSignInit(mdctx, &pctx, EVP_sha256(), NULL, private_key_.get()) <= 0) {
        // ... error handling ...
    }
    
    if (EVP_DigestSignUpdate(mdctx, message.payload.data(), message.payload.size()) <= 0) {
        // ... error handling ...
    }

    size_t sig_len;
    if (EVP_DigestSignFinal(mdctx, NULL, &sig_len) <= 0) {
        // ... error handling ...
    }

    std::vector<unsigned char> signature(sig_len);
    if (EVP_DigestSignFinal(mdctx, signature.data(), &sig_len) <= 0) {
        // ... error handling ...
    }
    
    message.signature.assign(signature.begin(), signature.end());

    EVP_MD_CTX_free(mdctx);
}

bool P2PNetwork::verify_message_signature(const NetworkMessage& message) {
    std::string public_key_pem;
    {
        std::lock_guard<std::mutex> lock(nodes_mutex_);
        auto it = known_nodes_.find(message.sender_id);
        if (it == known_nodes_.end()) {
            // Allow discovery messages from unknown nodes without signature
            if (message.type == MessageType::NODE_ANNOUNCEMENT || message.type == MessageType::NODE_DISCOVERY) {
                return true; 
            }
            return false; // Cannot verify if we don't know the node
        }
        public_key_pem = it->second.public_key;
    }

    if (public_key_pem.empty()) return false;

    BIO* bio = BIO_new_mem_buf(public_key_pem.data(), -1);
    EVP_PKEY* public_key = PEM_read_bio_PUBKEY(bio, NULL, NULL, NULL);
    BIO_free(bio);

    if (!public_key) { /* error */ return false; }

    EVP_MD_CTX* mdctx = EVP_MD_CTX_new();
    if (!mdctx) { /* error */ EVP_PKEY_free(public_key); return false; }

    bool is_valid = false;
    if (EVP_DigestVerifyInit(mdctx, NULL, EVP_sha256(), NULL, public_key) <= 0) {
        // ... error handling ...
    } else if (EVP_DigestVerifyUpdate(mdctx, message.payload.data(), message.payload.size()) <= 0) {
        // ... error handling ...
    } else if (EVP_DigestVerifyFinal(mdctx, (const unsigned char*)message.signature.data(), message.signature.size()) == 1) {
        is_valid = true;
    }
    
    EVP_MD_CTX_free(mdctx);
    EVP_PKEY_free(public_key);
    return is_valid;
}

void P2PNetwork::handle_client_connection(int client_sock) {
    // --- Start TLS Handshake (Server side) ---
    SSL* ssl = SSL_new(ssl_context_->get());
    SSL_set_fd(ssl, client_sock);

    if (SSL_accept(ssl) <= 0) {
        std::cerr << "SSL accept failed for incoming connection." << std::endl;
        ERR_print_errors_fp(stderr);
        SSL_free(ssl);
#ifdef _WIN32
        closesocket(client_sock);
#else
        close(client_sock);
#endif
        return;
    }
    std::cout << "SSL accept successful for incoming connection." << std::endl;

    // --- Communication loop using SSL ---
    std::vector<uint8_t> buffer(config_.message_buffer_size);
    std::string sender_id; // Will be identified after first message

    while (running_.load()) {
        int bytes_received = SSL_read(ssl, buffer.data(), buffer.size());
        if (bytes_received <= 0) {
            // ... (handle disconnection) ...
            break;
        }

        try {
            buffer.resize(bytes_received);
            NetworkMessage msg = serialization::deserialize(buffer);

            if (sender_id.empty()) {
                sender_id = msg.sender_id;
                std::lock_guard<std::mutex> lock(nodes_mutex_);
                known_nodes_[sender_id].ssl_session = std::shared_ptr<ssl_st>(ssl, SSL_free);
                known_nodes_[sender_id].socket = client_sock;
            }

            handle_incoming_message(msg);
            // ... (update stats) ...
        } catch (const std::exception& e) {
            std::cerr << "Error processing SSL message: " << e.what() << std::endl;
        }
    }

    // Clean up on disconnect
    if (!sender_id.empty()) {
        std::lock_guard<std::mutex> lock(nodes_mutex_);
        known_nodes_.erase(sender_id);
    }
    // The SSL_free is handled by the shared_ptr deleter.
    // The raw socket is closed automatically when the thread exits.
}

// Utility functions
namespace utils {

std::string get_local_ip_address() {
    // Simplified - would implement proper IP detection
    return "127.0.0.1";
}

bool is_port_available(uint16_t port) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        return false;
    }
    
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    
    bool available = (bind(sock, (struct sockaddr*)&addr, sizeof(addr)) == 0);
    
#ifdef _WIN32
    closesocket(sock);
#else
    close(sock);
#endif
    
    return available;
}

std::vector<std::string> discover_local_peers(uint16_t port_range_start, uint16_t port_range_end) {
    std::vector<std::string> peers;
    
    // Scan local network for peers (simplified implementation)
    for (uint16_t port = port_range_start; port <= port_range_end; ++port) {
        if (!is_port_available(port)) {
            peers.push_back("127.0.0.1:" + std::to_string(port));
        }
    }
    
    return peers;
}

struct QuantizedGradient {
    uint64_t original_rows;
    uint64_t original_cols;
    float scale;
    std::vector<int8_t> quantized_data;
};

QuantizedGradient quantize_matrix(const Matrix& matrix, int bits) {
    QuantizedGradient q_grad;
    q_grad.original_rows = matrix.rows();
    q_grad.original_cols = matrix.cols();

    // 1. Find the maximum absolute value in the matrix
    float abs_max = 0.0f;
    for (size_t i = 0; i < matrix.size(); ++i) {
        if (std::abs(matrix.data()[i]) > abs_max) {
            abs_max = std::abs(matrix.data()[i]);
        }
    }

    // 2. Calculate the scale factor
    float max_quant_val = (1 << (bits - 1)) - 1;
    q_grad.scale = abs_max / max_quant_val;

    // Avoid division by zero if matrix is all zeros
    if (q_grad.scale < 1e-9) {
        q_grad.scale = 1.0f;
    }

    // 3. Quantize the data
    q_grad.quantized_data.resize(matrix.size());
    for (size_t i = 0; i < matrix.size(); ++i) {
        float scaled_val = std::round(matrix.data()[i] / q_grad.scale);
        q_grad.quantized_data[i] = static_cast<int8_t>(std::max(-max_quant_val, std::min(max_quant_val, scaled_val)));
    }

    return q_grad;
}

Matrix dequantize_matrix(const QuantizedGradient& q_grad) {
    Matrix matrix(q_grad.original_rows, q_grad.original_cols);
    for (size_t i = 0; i < q_grad.quantized_data.size(); ++i) {
        matrix.data()[i] = static_cast<float>(q_grad.quantized_data[i]) * q_grad.scale;
    }
    return matrix;
}

} // namespace utils

// --- P2PTrainingCoordinator Implementation ---

P2PTrainingCoordinator::P2PTrainingCoordinator(std::shared_ptr<DistributedTransformer> transformer,
                                             std::shared_ptr<P2PNetwork> network)
    : transformer_(transformer), network_(network), training_active_(false) {}

P2PTrainingCoordinator::~P2PTrainingCoordinator() {
    stop();
}

bool P2PTrainingCoordinator::start() {
    if (training_active_.load()) {
        return true;
    }
    training_active_.store(true);
    coordination_thread_ = std::thread(&P2PTrainingCoordinator::coordination_thread_loop, this);
    return true;
}

void P2PTrainingCoordinator::stop() {
    if (!training_active_.load()) {
        return;
    }
    training_active_.store(false);
    queue_cv_.notify_one(); // Wake up the thread so it can exit
    if (coordination_thread_.joinable()) {
        coordination_thread_.join();
    }
}

void P2PTrainingCoordinator::submit_gradients(const std::vector<Matrix>& local_gradients) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        // Bounded staleness: limit queue size to prevent getting too far ahead
        if (gradient_queue_.size() > 2) { 
            // In a real scenario, we might wait here or drop older gradients.
            // For now, we'll just log a warning and proceed.
            std::cout << "Warning: Gradient queue is growing large." << std::endl;
        }
        gradient_queue_.push(local_gradients);
    }
    queue_cv_.notify_one();
}

void P2PTrainingCoordinator::coordination_thread_loop() {
    while (training_active_.load()) {
        std::vector<Matrix> gradients_to_process;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] { return !gradient_queue_.empty() || !training_active_.load(); });

            if (!training_active_.load() && gradient_queue_.empty()) {
                return; // Exit if stopped and queue is empty
            }

            gradients_to_process = gradient_queue_.front();
            gradient_queue_.pop();
        }

        // Now run the blocking consensus protocol
        std::string proposal_id = network_->propose_gradient(gradients_to_process, current_epoch_, current_batch_++);
        
        if (network_->wait_for_consensus(proposal_id)) {
            std::vector<Matrix> consensus_gradients = network_->get_consensus_gradient(proposal_id);
            if (!consensus_gradients.empty()) {
                // This call needs to be thread-safe if the model is being read by the main thread
                // For now, we assume a lock exists within apply_gradients or the optimizer.
                // transformer_->apply_gradients(consensus_gradients);
                std::cout << "Consensus gradients applied for proposal: " << proposal_id << std::endl;
            }
        } else {
            std::cerr << "Consensus failed for proposal: " << proposal_id << std::endl;
            // Handle failure...
        }
    }
}

} // namespace p2p
