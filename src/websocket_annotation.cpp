#include "websocket_annotation.hpp"
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <openssl/sha.h>
#include <openssl/bio.h>
#include <openssl/evp.h>
#include <openssl/buffer.h>
#include <json/json.h>
#include <chrono>
#include <algorithm>
#include <random>

namespace web_interface {

// WebSocketConnection implementation
void WebSocketConnection::send_json(const std::string& json_data) {
    std::lock_guard<std::mutex> lock(queue_mutex);
    message_queue.push(json_data);
}

void WebSocketConnection::close() {
    is_connected = false;
    if (socket_fd >= 0) {
        ::close(socket_fd);
        socket_fd = -1;
    }
}

// OperationalTransform implementation
OperationalTransform::Operation OperationalTransform::transform(const Operation& op1, const Operation& op2) {
    Operation result = op1;
    
    if (op1.type == Operation::INSERT && op2.type == Operation::INSERT) {
        if (op1.position < op2.position) {
            // op1 happens first, no change
        } else if (op1.position > op2.position) {
            result.position += op2.text.length();
        } else {
            // Same position - resolve by user_id for consistency
            if (op1.user_id < op2.user_id) {
                // op1 happens first
            } else {
                result.position += op2.text.length();
            }
        }
    } else if (op1.type == Operation::DELETE && op2.type == Operation::INSERT) {
        if (op1.position >= op2.position) {
            result.position += op2.text.length();
        }
    } else if (op1.type == Operation::INSERT && op2.type == Operation::DELETE) {
        if (op1.position > op2.position) {
            result.position -= std::min(op2.length, op1.position - op2.position);
        }
    } else if (op1.type == Operation::DELETE && op2.type == Operation::DELETE) {
        if (op1.position > op2.position) {
            result.position -= std::min(op2.length, op1.position - op2.position);
        }
    }
    
    return result;
}

std::string OperationalTransform::apply_operation(const std::string& text, const Operation& op) {
    std::string result = text;
    
    switch (op.type) {
        case Operation::INSERT:
            if (op.position <= result.length()) {
                result.insert(op.position, op.text);
            }
            break;
        case Operation::DELETE:
            if (op.position < result.length()) {
                result.erase(op.position, std::min(op.length, static_cast<uint32_t>(result.length() - op.position)));
            }
            break;
        case Operation::RETAIN:
            // No change
            break;
    }
    
    return result;
}

std::vector<OperationalTransform::Operation> OperationalTransform::compose_operations(
    const std::vector<Operation>& ops) {
    std::vector<Operation> result;
    
    for (const auto& op : ops) {
        bool composed = false;
        
        if (!result.empty()) {
            auto& last = result.back();
            
            // Try to compose consecutive operations
            if (last.type == op.type && last.user_id == op.user_id) {
                if (last.type == Operation::INSERT && 
                    last.position + last.text.length() == op.position) {
                    last.text += op.text;
                    composed = true;
                } else if (last.type == Operation::DELETE && 
                           last.position == op.position) {
                    last.length += op.length;
                    composed = true;
                }
            }
        }
        
        if (!composed) {
            result.push_back(op);
        }
    }
    
    return result;
}

// WebSocketAnnotationHub implementation
WebSocketAnnotationHub::WebSocketAnnotationHub(uint16_t port) : port_(port) {
    ot_engine_ = std::make_unique<OperationalTransform>();
    stats_ = {};
}

WebSocketAnnotationHub::~WebSocketAnnotationHub() {
    stop();
}

bool WebSocketAnnotationHub::start() {
    if (running_.load()) return true;
    
    // Create server socket
    server_socket_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket_ < 0) return false;
    
    // Allow socket reuse
    int opt = 1;
    setsockopt(server_socket_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    
    // Bind to port
    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port_);
    
    if (bind(server_socket_, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        close(server_socket_);
        return false;
    }
    
    if (listen(server_socket_, 128) < 0) {
        close(server_socket_);
        return false;
    }
    
    running_ = true;
    accept_thread_ = std::thread(&WebSocketAnnotationHub::accept_connections, this);
    
    // Start worker threads
    for (int i = 0; i < 4; ++i) {
        worker_threads_.emplace_back([this]() {
            while (running_.load()) {
                // Process message queues
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                update_stats();
            }
        });
    }
    
    return true;
}

void WebSocketAnnotationHub::stop() {
    if (!running_.load()) return;
    
    running_ = false;
    
    // Close server socket to interrupt accept
    if (server_socket_ >= 0) {
        close(server_socket_);
        server_socket_ = -1;
    }
    
    // Wait for threads
    if (accept_thread_.joinable()) {
        accept_thread_.join();
    }
    
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    // Close all connections
    std::lock_guard<std::mutex> lock(connections_mutex_);
    for (auto& [id, conn] : connections_) {
        conn->close();
    }
    connections_.clear();
}

std::string WebSocketAnnotationHub::add_connection(int socket_fd, const std::string& user_id, 
                                                   const std::string& session_id) {
    std::lock_guard<std::mutex> lock(connections_mutex_);
    
    // Generate connection ID
    std::string connection_id = user_id + "_" + session_id + "_" + 
                                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    
    auto conn = std::make_unique<WebSocketConnection>();
    conn->connection_id = connection_id;
    conn->user_id = user_id;
    conn->session_id = session_id;
    conn->socket_fd = socket_fd;
    
    connections_[connection_id] = std::move(conn);
    user_connections_[user_id] = connection_id;
    
    stats_.active_connections++;
    
    return connection_id;
}

void WebSocketAnnotationHub::remove_connection(const std::string& connection_id) {
    std::lock_guard<std::mutex> lock(connections_mutex_);
    
    auto it = connections_.find(connection_id);
    if (it != connections_.end()) {
        // Remove from user connections
        auto user_it = user_connections_.find(it->second->user_id);
        if (user_it != user_connections_.end() && user_it->second == connection_id) {
            user_connections_.erase(user_it);
        }
        
        // Remove from task connections
        for (auto& [task_id, conns] : task_connections_) {
            conns.erase(std::remove(conns.begin(), conns.end(), connection_id), conns.end());
        }
        
        it->second->close();
        connections_.erase(it);
        
        stats_.active_connections--;
    }
}

std::vector<std::string> WebSocketAnnotationHub::get_active_connections(const std::string& task_id) {
    std::lock_guard<std::mutex> lock(connections_mutex_);
    
    auto it = task_connections_.find(task_id);
    if (it != task_connections_.end()) {
        return it->second;
    }
    
    return {};
}

void WebSocketAnnotationHub::broadcast_annotation_update(const std::string& task_id, 
                                                         const AnnotationUpdate& update) {
    Json::Value msg;
    msg["type"] = "ANNOTATION_UPDATE";
    msg["task_id"] = task_id;
    msg["user_id"] = update.user_id;
    msg["field_name"] = update.field_name;
    msg["new_value"] = update.new_value;
    msg["timestamp"] = static_cast<Json::UInt64>(update.timestamp);
    msg["version"] = update.version;
    
    Json::StreamWriterBuilder builder;
    std::string json_str = Json::writeString(builder, msg);
    
    broadcast_to_task(task_id, json_str);
    
    stats_.messages_sent++;
}

void WebSocketAnnotationHub::handle_cursor_position(const std::string& user_id, 
                                                    const CursorPosition& pos) {
    Json::Value msg;
    msg["type"] = "CURSOR_POSITION";
    msg["user_id"] = user_id;
    msg["task_id"] = pos.task_id;
    msg["line"] = pos.line;
    msg["column"] = pos.column;
    msg["selection_text"] = pos.selection_text;
    
    Json::StreamWriterBuilder builder;
    std::string json_str = Json::writeString(builder, msg);
    
    broadcast_to_task(pos.task_id, json_str);
}

void WebSocketAnnotationHub::handle_live_discussion(const std::string& task_id, 
                                                    const ChatMessage& msg) {
    Json::Value json_msg;
    json_msg["type"] = "CHAT_MESSAGE";
    json_msg["task_id"] = task_id;
    json_msg["user_id"] = msg.user_id;
    json_msg["message"] = msg.message;
    json_msg["timestamp"] = static_cast<Json::UInt64>(msg.timestamp);
    json_msg["is_system_message"] = msg.is_system_message;
    
    Json::StreamWriterBuilder builder;
    std::string json_str = Json::writeString(builder, json_msg);
    
    broadcast_to_task(task_id, json_str);
    
    stats_.messages_sent++;
}

curation::AnnotationLabel WebSocketAnnotationHub::resolve_annotation_conflict(
    const std::vector<curation::AnnotationLabel>& competing) {
    
    if (competing.empty()) {
        return curation::AnnotationLabel{};
    }
    
    // Count votes for each label
    std::map<std::string, int> label_counts;
    std::map<std::string, float> confidence_sum;
    
    for (const auto& label : competing) {
        label_counts[label.text]++;
        confidence_sum[label.text] += label.confidence;
    }
    
    // Find majority vote
    std::string best_label;
    int max_count = 0;
    float max_avg_confidence = 0;
    
    for (const auto& [label, count] : label_counts) {
        float avg_confidence = confidence_sum[label] / count;
        
        if (count > max_count || (count == max_count && avg_confidence > max_avg_confidence)) {
            best_label = label;
            max_count = count;
            max_avg_confidence = avg_confidence;
        }
    }
    
    // Return the winning label
    for (const auto& label : competing) {
        if (label.text == best_label) {
            return label;
        }
    }
    
    return competing[0];  // Fallback
}

void WebSocketAnnotationHub::broadcast_consensus_update(const std::string& task_id, 
                                                        const ConsensusVisualization& viz) {
    Json::Value msg;
    msg["type"] = "CONSENSUS_UPDATE";
    msg["task_id"] = task_id;
    msg["overall_agreement"] = viz.overall_agreement;
    
    Json::Value distributions;
    for (const auto& [label, dist] : viz.label_distributions) {
        Json::Value dist_array;
        for (float val : dist) {
            dist_array.append(val);
        }
        distributions[label] = dist_array;
    }
    msg["label_distributions"] = distributions;
    
    Json::Value disagreements;
    for (const auto& [label1, label2] : viz.disagreements) {
        Json::Value pair;
        pair.append(label1);
        pair.append(label2);
        disagreements.append(pair);
    }
    msg["disagreements"] = disagreements;
    
    Json::Value confidence;
    for (const auto& [annotator, conf] : viz.annotator_confidence) {
        confidence[annotator] = conf;
    }
    msg["annotator_confidence"] = confidence;
    
    Json::StreamWriterBuilder builder;
    std::string json_str = Json::writeString(builder, msg);
    
    broadcast_to_task(task_id, json_str);
}

void WebSocketAnnotationHub::broadcast_reputation_update(const std::string& user_id, 
                                                         float new_score, uint32_t rank) {
    Json::Value msg;
    msg["type"] = "REPUTATION_UPDATE";
    msg["user_id"] = user_id;
    msg["score"] = new_score;
    msg["rank"] = rank;
    
    Json::StreamWriterBuilder builder;
    std::string json_str = Json::writeString(builder, msg);
    
    send_to_connection(user_connections_[user_id], json_str);
}

void WebSocketAnnotationHub::broadcast_achievement(const std::string& user_id, 
                                                   const std::string& achievement) {
    Json::Value msg;
    msg["type"] = "ACHIEVEMENT";
    msg["user_id"] = user_id;
    msg["achievement"] = achievement;
    msg["timestamp"] = static_cast<Json::UInt64>(
        std::chrono::system_clock::now().time_since_epoch().count());
    
    Json::StreamWriterBuilder builder;
    std::string json_str = Json::writeString(builder, msg);
    
    // Send to user
    send_to_connection(user_connections_[user_id], json_str);
    
    // Also broadcast to everyone for celebration
    broadcast_to_all(json_str);
}

void WebSocketAnnotationHub::broadcast_leaderboard_update(
    const std::vector<std::pair<std::string, float>>& top_users) {
    
    Json::Value msg;
    msg["type"] = "LEADERBOARD_UPDATE";
    
    Json::Value leaderboard;
    for (size_t i = 0; i < top_users.size(); ++i) {
        Json::Value entry;
        entry["rank"] = static_cast<Json::UInt>(i + 1);
        entry["user_id"] = top_users[i].first;
        entry["score"] = top_users[i].second;
        leaderboard.append(entry);
    }
    msg["leaderboard"] = leaderboard;
    
    Json::StreamWriterBuilder builder;
    std::string json_str = Json::writeString(builder, msg);
    
    broadcast_to_all(json_str);
}

WebSocketAnnotationHub::HubStats WebSocketAnnotationHub::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void WebSocketAnnotationHub::accept_connections() {
    while (running_.load()) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        
        int client_socket = accept(server_socket_, (struct sockaddr*)&client_addr, &client_len);
        if (client_socket < 0) {
            if (running_.load()) {
                // Error in accept
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            continue;
        }
        
        // Handle WebSocket handshake and create connection
        // This would be done in a worker thread in production
        // For now, simplified implementation
        
        // Read handshake
        char buffer[4096];
        int bytes_read = recv(client_socket, buffer, sizeof(buffer) - 1, 0);
        if (bytes_read > 0) {
            buffer[bytes_read] = '\0';
            
            // Parse WebSocket key (simplified)
            std::string request(buffer);
            size_t key_pos = request.find("Sec-WebSocket-Key: ");
            if (key_pos != std::string::npos) {
                key_pos += 19;  // Length of "Sec-WebSocket-Key: "
                size_t key_end = request.find("\r\n", key_pos);
                std::string client_key = request.substr(key_pos, key_end - key_pos);
                
                // Generate accept key
                std::string accept_key = create_websocket_accept_key(client_key);
                
                // Send handshake response
                std::string response = 
                    "HTTP/1.1 101 Switching Protocols\r\n"
                    "Upgrade: websocket\r\n"
                    "Connection: Upgrade\r\n"
                    "Sec-WebSocket-Accept: " + accept_key + "\r\n"
                    "\r\n";
                
                send(client_socket, response.c_str(), response.length(), 0);
                
                // Create connection (simplified - would parse user_id from request)
                std::string user_id = "user_" + std::to_string(client_socket);
                std::string session_id = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
                add_connection(client_socket, user_id, session_id);
            } else {
                close(client_socket);
            }
        } else {
            close(client_socket);
        }
    }
}

std::string WebSocketAnnotationHub::create_websocket_accept_key(const std::string& client_key) {
    // WebSocket magic string
    const std::string magic_string = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    std::string combined = client_key + magic_string;
    
    // SHA1 hash
    unsigned char hash[SHA_DIGEST_LENGTH];
    SHA1(reinterpret_cast<const unsigned char*>(combined.c_str()), combined.length(), hash);
    
    // Base64 encode
    BIO* b64 = BIO_new(BIO_f_base64());
    BIO* bio = BIO_new(BIO_s_mem());
    bio = BIO_push(b64, bio);
    BIO_set_flags(bio, BIO_FLAGS_BASE64_NO_NL);
    BIO_write(bio, hash, SHA_DIGEST_LENGTH);
    BIO_flush(bio);
    
    BUF_MEM* buffer_ptr;
    BIO_get_mem_ptr(bio, &buffer_ptr);
    std::string accept_key(buffer_ptr->data, buffer_ptr->length);
    
    BIO_free_all(bio);
    
    return accept_key;
}

void WebSocketAnnotationHub::broadcast_to_task(const std::string& task_id, const std::string& message) {
    std::lock_guard<std::mutex> lock(connections_mutex_);
    
    auto it = task_connections_.find(task_id);
    if (it != task_connections_.end()) {
        for (const auto& conn_id : it->second) {
            auto conn_it = connections_.find(conn_id);
            if (conn_it != connections_.end() && conn_it->second->is_connected) {
                conn_it->second->send_json(message);
            }
        }
    }
}

void WebSocketAnnotationHub::broadcast_to_all(const std::string& message) {
    std::lock_guard<std::mutex> lock(connections_mutex_);
    
    for (auto& [id, conn] : connections_) {
        if (conn->is_connected) {
            conn->send_json(message);
        }
    }
}

void WebSocketAnnotationHub::send_to_connection(const std::string& connection_id, const std::string& message) {
    std::lock_guard<std::mutex> lock(connections_mutex_);
    
    auto it = connections_.find(connection_id);
    if (it != connections_.end() && it->second->is_connected) {
        it->second->send_json(message);
    }
}

void WebSocketAnnotationHub::update_stats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    // Update active tasks count
    stats_.active_tasks.clear();
    for (const auto& [task_id, conns] : task_connections_) {
        stats_.active_tasks[task_id] = conns.size();
    }
    
    // Calculate average latency (would need actual measurement in production)
    stats_.average_latency_ms = 5.0f;  // Placeholder
}

// QualityDashboard implementation
QualityDashboard::QualityDashboard(std::shared_ptr<curation::DistributedCurationPlatform> curation_platform)
    : curation_platform_(curation_platform) {
}

QualityDashboard::~QualityDashboard() {
    stop_streaming();
}

void QualityDashboard::start_streaming(std::shared_ptr<WebSocketAnnotationHub> hub) {
    if (streaming_.load()) return;
    
    websocket_hub_ = hub;
    streaming_ = true;
    streaming_thread_ = std::thread(&QualityDashboard::streaming_loop, this);
}

void QualityDashboard::stop_streaming() {
    if (!streaming_.load()) return;
    
    streaming_ = false;
    if (streaming_thread_.joinable()) {
        streaming_thread_.join();
    }
}

QualityDashboard::RealTimeMetrics QualityDashboard::compute_realtime_metrics() {
    RealTimeMetrics metrics;
    
    // Get recent submissions
    auto recent_submissions = curation_platform_->get_recent_submissions(100);
    
    // Calculate IAA
    metrics.current_iaa = compute_iaa(recent_submissions);
    
    // Rank annotators
    metrics.annotator_performance = rank_annotators();
    
    // Analyze task distribution
    metrics.task_difficulty_distribution = analyze_task_distribution();
    
    // Check for quality alerts
    auto anomalies = detect_anomalies();
    for (const auto& anomaly : anomalies) {
        if (anomaly.severity_score > 0.7f) {
            metrics.quality_alerts.push_back(
                "⚠️ " + anomaly.issue_type + " for " + anomaly.annotator_id);
        }
    }
    
    return metrics;
}

std::vector<QualityDashboard::AnomalyDetection> QualityDashboard::detect_anomalies() {
    std::vector<AnomalyDetection> anomalies;
    
    // Get annotator statistics
    auto annotator_stats = curation_platform_->get_annotator_statistics();
    
    for (const auto& stats : annotator_stats) {
        // Check for speed anomalies
        if (is_anomalous_speed(stats.annotator_id, stats.avg_time_per_task)) {
            anomalies.push_back({
                stats.annotator_id,
                "Unusually fast annotation speed",
                0.8f,
                "Review annotations for quality",
                static_cast<uint64_t>(std::chrono::system_clock::now().time_since_epoch().count())
            });
        }
        
        // Check for agreement anomalies
        if (is_anomalous_agreement(stats.annotator_id, stats.agreement_rate)) {
            anomalies.push_back({
                stats.annotator_id,
                "Low agreement with consensus",
                0.7f,
                "Provide additional training",
                static_cast<uint64_t>(std::chrono::system_clock::now().time_since_epoch().count())
            });
        }
    }
    
    return anomalies;
}

float QualityDashboard::compute_task_difficulty(const std::string& task_id) {
    auto task_submissions = curation_platform_->get_task_submissions(task_id);
    
    if (task_submissions.empty()) return 0.5f;
    
    // Calculate variance in annotations
    std::map<std::string, int> label_counts;
    for (const auto& submission : task_submissions) {
        for (const auto& label : submission.labels) {
            label_counts[label.text]++;
        }
    }
    
    // High variance = high difficulty
    float total = task_submissions.size();
    float entropy = 0.0f;
    
    for (const auto& [label, count] : label_counts) {
        float p = count / total;
        if (p > 0) {
            entropy -= p * std::log2(p);
        }
    }
    
    // Normalize entropy to [0, 1]
    float max_entropy = std::log2(label_counts.size());
    return max_entropy > 0 ? entropy / max_entropy : 0.0f;
}

void QualityDashboard::streaming_loop() {
    while (streaming_.load()) {
        auto metrics = compute_realtime_metrics();
        
        // Send metrics to all connected clients
        Json::Value msg;
        msg["type"] = "QUALITY_METRICS";
        msg["iaa"] = metrics.current_iaa;
        
        Json::Value annotator_perf;
        for (const auto& [id, score] : metrics.annotator_performance) {
            annotator_perf[id] = score;
        }
        msg["annotator_performance"] = annotator_perf;
        
        Json::Value task_difficulty;
        for (const auto& [task, diff] : metrics.task_difficulty_distribution) {
            task_difficulty[task] = diff;
        }
        msg["task_difficulty"] = task_difficulty;
        
        Json::Value alerts;
        for (const auto& alert : metrics.quality_alerts) {
            alerts.append(alert);
        }
        msg["quality_alerts"] = alerts;
        
        Json::StreamWriterBuilder builder;
        std::string json_str = Json::writeString(builder, msg);
        
        if (websocket_hub_) {
            // Broadcast to all connections
            // In real implementation, this would use websocket_hub_->broadcast_to_all()
        }
        
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
}

float QualityDashboard::compute_iaa(const std::vector<curation::AnnotationSubmission>& submissions) {
    if (submissions.size() < 2) return 1.0f;
    
    // Simplified Fleiss' Kappa calculation
    std::map<std::string, std::map<std::string, int>> task_label_counts;
    std::map<std::string, int> task_annotator_counts;
    
    for (const auto& submission : submissions) {
        task_annotator_counts[submission.task_id]++;
        for (const auto& label : submission.labels) {
            task_label_counts[submission.task_id][label.text]++;
        }
    }
    
    float total_agreement = 0.0f;
    int task_count = 0;
    
    for (const auto& [task_id, label_counts] : task_label_counts) {
        int n = task_annotator_counts[task_id];
        if (n < 2) continue;
        
        float p_observed = 0.0f;
        for (const auto& [label, count] : label_counts) {
            p_observed += (float)(count * (count - 1)) / (n * (n - 1));
        }
        
        total_agreement += p_observed;
        task_count++;
    }
    
    return task_count > 0 ? total_agreement / task_count : 0.0f;
}

std::vector<std::pair<std::string, float>> QualityDashboard::rank_annotators() {
    auto stats = curation_platform_->get_annotator_statistics();
    
    std::vector<std::pair<std::string, float>> rankings;
    for (const auto& stat : stats) {
        float score = stat.agreement_rate * 0.4f + 
                     stat.completion_rate * 0.3f + 
                     (1.0f - std::min(1.0f, stat.avg_time_per_task / 300.0f)) * 0.3f;
        rankings.push_back({stat.annotator_id, score});
    }
    
    std::sort(rankings.begin(), rankings.end(), 
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    return rankings;
}

std::map<std::string, float> QualityDashboard::analyze_task_distribution() {
    std::map<std::string, float> distribution;
    
    auto all_tasks = curation_platform_->get_all_tasks();
    for (const auto& task : all_tasks) {
        float difficulty = compute_task_difficulty(task.task_id);
        distribution[task.task_id] = difficulty;
    }
    
    return distribution;
}

bool QualityDashboard::is_anomalous_speed(const std::string& annotator_id, float time_spent) {
    // Flag if less than 5 seconds per task
    return time_spent < 5.0f;
}

bool QualityDashboard::is_anomalous_agreement(const std::string& annotator_id, float agreement) {
    // Flag if agreement is below 60%
    return agreement < 0.6f;
}

bool QualityDashboard::is_anomalous_pattern(const std::vector<curation::AnnotationLabel>& labels) {
    // Check for patterns like all same label, sequential patterns, etc.
    if (labels.size() < 10) return false;
    
    // Check if all labels are the same
    std::set<std::string> unique_labels;
    for (const auto& label : labels) {
        unique_labels.insert(label.text);
    }
    
    return unique_labels.size() == 1;
}

} // namespace web_interface
