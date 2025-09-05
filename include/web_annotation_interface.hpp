#pragma once

#include "distributed_curation.hpp"
#include <string>
#include <memory>
#include <functional>
#include <thread>
#include <atomic>
#include <map>
#include <vector>

namespace web_interface {

// HTTP server configuration
struct WebServerConfig {
    std::string bind_address = "0.0.0.0";
    uint16_t port = 8080;
    std::string static_files_path = "./web";
    uint32_t max_connections = 100;
    uint32_t request_timeout_seconds = 30;
    bool enable_cors = true;
    std::string cors_origin = "*";
    bool enable_ssl = false;
    std::string ssl_cert_path;
    std::string ssl_key_path;
};

// HTTP request/response structures
struct HttpRequest {
    std::string method;
    std::string path;
    std::string query_string;
    std::map<std::string, std::string> headers;
    std::map<std::string, std::string> query_params;
    std::string body;
    std::string client_ip;
};

struct HttpResponse {
    uint16_t status_code = 200;
    std::string status_text = "OK";
    std::map<std::string, std::string> headers;
    std::string body;
    std::string content_type = "text/html";
};

// Session management
struct UserSession {
    std::string session_id;
    std::string user_id;
    std::string annotator_id;
    uint64_t created_timestamp;
    uint64_t last_activity;
    bool is_authenticated;
    std::map<std::string, std::string> session_data;
};

// Annotation UI data structures
struct AnnotationTaskUI {
    std::string task_id;
    std::string title;
    std::string description;
    std::string content;
    std::string content_type; // "text", "conversation", "code", etc.
    std::vector<std::string> label_schema;
    std::map<std::string, std::string> metadata;
    float difficulty_score;
    float estimated_time_minutes;
    uint32_t current_annotators;
    uint32_t required_annotators;
};

struct AnnotationSubmissionUI {
    std::string task_id;
    std::map<std::string, float> label_scores;
    std::map<std::string, std::string> label_comments;
    float confidence_score;
    uint32_t time_spent_seconds;
    std::string feedback;
};

// Web-based annotation interface
class WebAnnotationInterface {
public:
    WebAnnotationInterface(std::shared_ptr<curation::DistributedCurationPlatform> curation_platform,
                          const WebServerConfig& config = WebServerConfig{});
    ~WebAnnotationInterface();

    // Server lifecycle
    bool start();
    void stop();
    bool is_running() const { return running_.load(); }

    // Session management
    std::string create_session(const std::string& user_id, const std::string& annotator_id);
    bool validate_session(const std::string& session_id);
    void destroy_session(const std::string& session_id);
    std::optional<UserSession> get_session(const std::string& session_id);

    // Statistics
    struct InterfaceStats {
        uint32_t active_sessions;
        uint32_t total_requests;
        uint32_t completed_annotations;
        uint32_t active_tasks;
        float average_annotation_time_minutes;
        std::map<std::string, uint32_t> user_activity;
    };
    InterfaceStats get_stats() const;

private:
    std::shared_ptr<curation::DistributedCurationPlatform> curation_platform_;
    WebServerConfig config_;
    std::atomic<bool> running_{false};

    // HTTP server
    std::thread server_thread_;
    void http_server_thread();
    void handle_client_connection(int client_socket);

    // Session management
    std::map<std::string, UserSession> active_sessions_;
    mutable std::mutex sessions_mutex_;
    void cleanup_expired_sessions();

    // Request routing
    using RequestHandler = std::function<HttpResponse(const HttpRequest&, const UserSession*)>;
    std::map<std::string, RequestHandler> route_handlers_;
    void setup_routes();

    // Route handlers
    HttpResponse handle_index(const HttpRequest& request, const UserSession* session);
    HttpResponse handle_login(const HttpRequest& request, const UserSession* session);
    HttpResponse handle_logout(const HttpRequest& request, const UserSession* session);
    HttpResponse handle_dashboard(const HttpRequest& request, const UserSession* session);
    HttpResponse handle_get_tasks(const HttpRequest& request, const UserSession* session);
    HttpResponse handle_get_task_details(const HttpRequest& request, const UserSession* session);
    HttpResponse handle_submit_annotation(const HttpRequest& request, const UserSession* session);
    HttpResponse handle_get_profile(const HttpRequest& request, const UserSession* session);
    HttpResponse handle_update_profile(const HttpRequest& request, const UserSession* session);
    HttpResponse handle_get_stats(const HttpRequest& request, const UserSession* session);
    HttpResponse handle_static_files(const HttpRequest& request, const UserSession* session);

    // API endpoints
    HttpResponse api_get_available_tasks(const HttpRequest& request, const UserSession* session);
    HttpResponse api_submit_annotation(const HttpRequest& request, const UserSession* session);
    HttpResponse api_get_annotator_stats(const HttpRequest& request, const UserSession* session);
    HttpResponse api_get_leaderboard(const HttpRequest& request, const UserSession* session);

    // Utility methods
    HttpRequest parse_http_request(const std::string& raw_request);
    std::string serialize_http_response(const HttpResponse& response);
    std::map<std::string, std::string> parse_query_string(const std::string& query_string);
    std::string url_decode(const std::string& encoded);
    std::string generate_session_id();
    bool is_authenticated(const HttpRequest& request, UserSession*& session);
    
    // JSON helpers
    std::string to_json(const AnnotationTaskUI& task);
    std::string to_json(const std::vector<AnnotationTaskUI>& tasks);
    std::string to_json(const curation::AnnotatorProfile& profile);
    std::string to_json(const InterfaceStats& stats);
    AnnotationSubmissionUI from_json_submission(const std::string& json);

    // HTML template rendering
    std::string render_template(const std::string& template_name, 
                               const std::map<std::string, std::string>& variables = {});
    std::string load_template(const std::string& template_name);
    std::string replace_template_variables(const std::string& template_content,
                                         const std::map<std::string, std::string>& variables);

    // Statistics tracking
    mutable std::mutex stats_mutex_;
    InterfaceStats current_stats_;
    void update_stats();
};

// HTML template generator for annotation interface
class AnnotationTemplateGenerator {
public:
    static std::string generate_index_page();
    static std::string generate_login_page();
    static std::string generate_dashboard_page();
    static std::string generate_annotation_page(const AnnotationTaskUI& task);
    static std::string generate_profile_page(const curation::AnnotatorProfile& profile);
    static std::string generate_leaderboard_page();
    static std::string generate_stats_page();

    // CSS and JavaScript
    static std::string generate_css();
    static std::string generate_javascript();

private:
    static std::string wrap_in_layout(const std::string& content, const std::string& title = "Annotation Interface");
    static std::string generate_navigation(bool is_authenticated = false);
    static std::string generate_task_card(const AnnotationTaskUI& task);
    static std::string generate_annotation_form(const AnnotationTaskUI& task);
};

// Utility functions for web interface
namespace utils {
    // HTTP utilities
    std::string get_mime_type(const std::string& file_extension);
    std::string get_current_timestamp_iso();
    std::string escape_html(const std::string& text);
    std::string escape_json(const std::string& text);
    
    // File utilities
    bool file_exists(const std::string& path);
    std::string read_file(const std::string& path);
    bool write_file(const std::string& path, const std::string& content);
    
    // Validation
    bool is_valid_email(const std::string& email);
    bool is_valid_session_id(const std::string& session_id);
    bool is_safe_filename(const std::string& filename);
    
    // Security
    std::string generate_csrf_token();
    bool validate_csrf_token(const std::string& token, const std::string& session_id);
    std::string hash_password(const std::string& password);
    bool verify_password(const std::string& password, const std::string& hash);
}

} // namespace web_interface
