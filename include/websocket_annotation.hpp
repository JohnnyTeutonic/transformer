#pragma once

#include <string>
#include <map>
#include <vector>
#include <memory>
#include <functional>
#include <thread>
#include <atomic>
#include <mutex>
#include <queue>
#include "distributed_curation.hpp"

namespace web_interface {

// WebSocket message types
enum class WebSocketMessageType {
    ANNOTATION_UPDATE,
    CURSOR_POSITION,
    CHAT_MESSAGE,
    CONSENSUS_UPDATE,
    USER_JOINED,
    USER_LEFT,
    HEARTBEAT,
    ERROR
};

// Real-time annotation update
struct AnnotationUpdate {
    std::string task_id;
    std::string user_id;
    std::string field_name;
    std::string new_value;
    uint64_t timestamp;
    uint32_t version;  // For conflict resolution
};

// Cursor position for collaborative editing
struct CursorPosition {
    std::string user_id;
    std::string task_id;
    uint32_t line;
    uint32_t column;
    std::string selection_text;
};

// Chat message for discussion
struct ChatMessage {
    std::string user_id;
    std::string task_id;
    std::string message;
    uint64_t timestamp;
    bool is_system_message;
};

// WebSocket connection
struct WebSocketConnection {
    std::string connection_id;
    std::string user_id;
    std::string session_id;
    int socket_fd;
    std::atomic<bool> is_connected{true};
    std::queue<std::string> message_queue;
    std::mutex queue_mutex;
    
    void send_json(const std::string& json_data);
    void close();
};

// Operational Transform for collaborative text editing
class OperationalTransform {
public:
    struct Operation {
        enum Type { INSERT, DELETE, RETAIN };
        Type type;
        std::string text;
        uint32_t position;
        uint32_t length;
        std::string user_id;
        uint32_t version;
    };
    
    Operation transform(const Operation& op1, const Operation& op2);
    std::string apply_operation(const std::string& text, const Operation& op);
    std::vector<Operation> compose_operations(const std::vector<Operation>& ops);
    
private:
    uint32_t current_version_ = 0;
    std::map<std::string, std::vector<Operation>> pending_operations_;
};

// Consensus visualization data
struct ConsensusVisualization {
    std::string task_id;
    std::map<std::string, std::vector<float>> label_distributions;
    float overall_agreement;
    std::vector<std::pair<std::string, std::string>> disagreements;
    std::map<std::string, float> annotator_confidence;
};

// WebSocket annotation hub for real-time collaboration
class WebSocketAnnotationHub {
public:
    WebSocketAnnotationHub(uint16_t port = 8081);
    ~WebSocketAnnotationHub();
    
    // Lifecycle
    bool start();
    void stop();
    bool is_running() const { return running_.load(); }
    
    // Connection management
    std::string add_connection(int socket_fd, const std::string& user_id, const std::string& session_id);
    void remove_connection(const std::string& connection_id);
    std::vector<std::string> get_active_connections(const std::string& task_id);
    
    // Real-time updates
    void broadcast_annotation_update(const std::string& task_id, const AnnotationUpdate& update);
    void handle_cursor_position(const std::string& user_id, const CursorPosition& pos);
    void handle_live_discussion(const std::string& task_id, const ChatMessage& msg);
    
    // Consensus and conflict resolution
    curation::AnnotationLabel resolve_annotation_conflict(const std::vector<curation::AnnotationLabel>& competing);
    void broadcast_consensus_update(const std::string& task_id, const ConsensusVisualization& viz);
    
    // Gamification updates
    void broadcast_reputation_update(const std::string& user_id, float new_score, uint32_t rank);
    void broadcast_achievement(const std::string& user_id, const std::string& achievement);
    void broadcast_leaderboard_update(const std::vector<std::pair<std::string, float>>& top_users);
    
    // Statistics
    struct HubStats {
        uint32_t active_connections;
        uint32_t messages_sent;
        uint32_t messages_received;
        std::map<std::string, uint32_t> active_tasks;
        float average_latency_ms;
    };
    HubStats get_stats() const;
    
private:
    uint16_t port_;
    std::atomic<bool> running_{false};
    int server_socket_ = -1;
    
    // Connection management
    std::map<std::string, std::unique_ptr<WebSocketConnection>> connections_;
    std::map<std::string, std::vector<std::string>> task_connections_;  // task_id -> connection_ids
    std::map<std::string, std::string> user_connections_;  // user_id -> connection_id
    mutable std::mutex connections_mutex_;
    
    // Operational transform engine
    std::unique_ptr<OperationalTransform> ot_engine_;
    
    // Message handling
    std::thread accept_thread_;
    std::vector<std::thread> worker_threads_;
    void accept_connections();
    void handle_connection(std::unique_ptr<WebSocketConnection> conn);
    void process_message(const std::string& connection_id, const std::string& message);
    
    // WebSocket protocol
    std::string create_websocket_accept_key(const std::string& client_key);
    std::string encode_websocket_frame(const std::string& payload);
    std::string decode_websocket_frame(const std::string& frame);
    
    // Broadcast helpers
    void broadcast_to_task(const std::string& task_id, const std::string& message);
    void broadcast_to_all(const std::string& message);
    void send_to_connection(const std::string& connection_id, const std::string& message);
    
    // Statistics tracking
    mutable std::mutex stats_mutex_;
    HubStats stats_;
    void update_stats();
};

// Real-time quality metrics dashboard
class QualityDashboard {
public:
    struct RealTimeMetrics {
        float current_iaa;  // Inter-annotator agreement
        std::vector<std::pair<std::string, float>> annotator_performance;
        std::map<std::string, float> task_difficulty_distribution;
        std::vector<std::string> quality_alerts;
        std::map<std::string, ConsensusVisualization> task_consensus;
    };
    
    struct AnomalyDetection {
        std::string annotator_id;
        std::string issue_type;
        float severity_score;
        std::string recommendation;
        uint64_t timestamp;
    };
    
    QualityDashboard(std::shared_ptr<curation::DistributedCurationPlatform> curation_platform);
    ~QualityDashboard();
    
    // Start streaming metrics
    void start_streaming(std::shared_ptr<WebSocketAnnotationHub> hub);
    void stop_streaming();
    
    // Compute metrics
    RealTimeMetrics compute_realtime_metrics();
    std::vector<AnomalyDetection> detect_anomalies();
    float compute_task_difficulty(const std::string& task_id);
    
    // Stream to web interface
    void stream_metrics_to_web(WebSocketConnection& conn);
    
private:
    std::shared_ptr<curation::DistributedCurationPlatform> curation_platform_;
    std::shared_ptr<WebSocketAnnotationHub> websocket_hub_;
    std::atomic<bool> streaming_{false};
    
    std::thread streaming_thread_;
    void streaming_loop();
    
    // Metrics computation
    float compute_iaa(const std::vector<curation::AnnotationSubmission>& submissions);
    std::vector<std::pair<std::string, float>> rank_annotators();
    std::map<std::string, float> analyze_task_distribution();
    
    // Anomaly detection
    bool is_anomalous_speed(const std::string& annotator_id, float time_spent);
    bool is_anomalous_agreement(const std::string& annotator_id, float agreement);
    bool is_anomalous_pattern(const std::vector<curation::AnnotationLabel>& labels);
};

} // namespace web_interface
