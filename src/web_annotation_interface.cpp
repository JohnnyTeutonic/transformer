#include "../include/web_annotation_interface.hpp"
#include <iostream>
#include <sstream>
#include <fstream>
#include <regex>
#include <random>
#include <chrono>
#include <algorithm>
#include <thread>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif

namespace web_interface {

WebAnnotationInterface::WebAnnotationInterface(
    std::shared_ptr<curation::DistributedCurationPlatform> curation_platform,
    const WebServerConfig& config)
    : curation_platform_(curation_platform), config_(config) {
    
    std::cout << "Initializing Web Annotation Interface" << std::endl;
    std::cout << "- Bind address: " << config_.bind_address << ":" << config_.port << std::endl;
    std::cout << "- Static files path: " << config_.static_files_path << std::endl;
    std::cout << "- CORS enabled: " << (config_.enable_cors ? "yes" : "no") << std::endl;

#ifdef _WIN32
    // Initialize Winsock
    WSADATA wsaData;
    int result = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (result != 0) {
        throw std::runtime_error("WSAStartup failed: " + std::to_string(result));
    }
#endif

    setup_routes();
    
    // Initialize stats
    current_stats_ = InterfaceStats{};
}

WebAnnotationInterface::~WebAnnotationInterface() {
    stop();
    
#ifdef _WIN32
    WSACleanup();
#endif
}

bool WebAnnotationInterface::start() {
    if (running_.load()) {
        std::cout << "Web annotation interface already running" << std::endl;
        return true;
    }

    std::cout << "Starting web annotation interface..." << std::endl;
    
    running_.store(true);
    server_thread_ = std::thread(&WebAnnotationInterface::http_server_thread, this);
    
    std::cout << "Web annotation interface started on http://" 
              << config_.bind_address << ":" << config_.port << std::endl;
    
    return true;
}

void WebAnnotationInterface::stop() {
    if (!running_.load()) {
        return;
    }
    
    std::cout << "Stopping web annotation interface..." << std::endl;
    running_.store(false);
    
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
    
    std::cout << "Web annotation interface stopped" << std::endl;
}

void WebAnnotationInterface::http_server_thread() {
    std::cout << "HTTP server thread started" << std::endl;
    
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
    server_addr.sin_port = htons(config_.port);
    
    if (bind(listen_sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "Failed to bind socket to port " << config_.port << std::endl;
#ifdef _WIN32
        closesocket(listen_sock);
#else
        close(listen_sock);
#endif
        return;
    }
    
    // Listen for connections
    if (listen(listen_sock, config_.max_connections) < 0) {
        std::cerr << "Failed to listen on socket" << std::endl;
#ifdef _WIN32
        closesocket(listen_sock);
#else
        close(listen_sock);
#endif
        return;
    }
    
    std::cout << "HTTP server listening on port " << config_.port << std::endl;
    
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
        std::thread(&WebAnnotationInterface::handle_client_connection, this, client_sock).detach();
    }
    
#ifdef _WIN32
    closesocket(listen_sock);
#else
    close(listen_sock);
#endif
    
    std::cout << "HTTP server thread stopped" << std::endl;
}

void WebAnnotationInterface::handle_client_connection(int client_socket) {
    // Read HTTP request
    char buffer[8192];
    int bytes_received = recv(client_socket, buffer, sizeof(buffer) - 1, 0);
    
    if (bytes_received <= 0) {
#ifdef _WIN32
        closesocket(client_socket);
#else
        close(client_socket);
#endif
        return;
    }
    
    buffer[bytes_received] = '\0';
    std::string raw_request(buffer);
    
    try {
        // Parse HTTP request
        HttpRequest request = parse_http_request(raw_request);
        
        // Update statistics
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            current_stats_.total_requests++;
        }
        
        // Check for session
        UserSession* session = nullptr;
        auto cookie_it = request.headers.find("cookie");
        if (cookie_it != request.headers.end()) {
            // Parse session ID from cookie
            std::regex session_regex("session_id=([^;]+)");
            std::smatch match;
            if (std::regex_search(cookie_it->second, match, session_regex)) {
                std::string session_id = match[1].str();
                auto session_opt = get_session(session_id);
                if (session_opt) {
                    auto& sess = active_sessions_[session_id];
                    sess.last_activity = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count();
                    session = &sess;
                }
            }
        }
        
        // Route request
        HttpResponse response;
        
        // Find matching route
        bool route_found = false;
        for (const auto& [route_pattern, handler] : route_handlers_) {
            if (request.path == route_pattern || 
                (route_pattern.back() == '*' && 
                 request.path.substr(0, route_pattern.length() - 1) == route_pattern.substr(0, route_pattern.length() - 1))) {
                response = handler(request, session);
                route_found = true;
                break;
            }
        }
        
        if (!route_found) {
            response.status_code = 404;
            response.status_text = "Not Found";
            response.body = "<html><body><h1>404 Not Found</h1></body></html>";
            response.content_type = "text/html";
        }
        
        // Add CORS headers if enabled
        if (config_.enable_cors) {
            response.headers["Access-Control-Allow-Origin"] = config_.cors_origin;
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS";
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With";
        }
        
        // Serialize and send response
        std::string response_str = serialize_http_response(response);
        send(client_socket, response_str.c_str(), response_str.length(), 0);
        
    } catch (const std::exception& e) {
        std::cerr << "Error handling HTTP request: " << e.what() << std::endl;
        
        // Send error response
        HttpResponse error_response;
        error_response.status_code = 500;
        error_response.status_text = "Internal Server Error";
        error_response.body = "<html><body><h1>500 Internal Server Error</h1></body></html>";
        error_response.content_type = "text/html";
        
        std::string error_str = serialize_http_response(error_response);
        send(client_socket, error_str.c_str(), error_str.length(), 0);
    }
    
#ifdef _WIN32
    closesocket(client_socket);
#else
    close(client_socket);
#endif
}

void WebAnnotationInterface::setup_routes() {
    // Main pages
    route_handlers_["/"] = [this](const HttpRequest& req, const UserSession* sess) { 
        return handle_index(req, sess); 
    };
    route_handlers_["/login"] = [this](const HttpRequest& req, const UserSession* sess) { 
        return handle_login(req, sess); 
    };
    route_handlers_["/logout"] = [this](const HttpRequest& req, const UserSession* sess) { 
        return handle_logout(req, sess); 
    };
    route_handlers_["/dashboard"] = [this](const HttpRequest& req, const UserSession* sess) { 
        return handle_dashboard(req, sess); 
    };
    route_handlers_["/tasks"] = [this](const HttpRequest& req, const UserSession* sess) { 
        return handle_get_tasks(req, sess); 
    };
    route_handlers_["/task"] = [this](const HttpRequest& req, const UserSession* sess) { 
        return handle_get_task_details(req, sess); 
    };
    route_handlers_["/submit"] = [this](const HttpRequest& req, const UserSession* sess) { 
        return handle_submit_annotation(req, sess); 
    };
    route_handlers_["/profile"] = [this](const HttpRequest& req, const UserSession* sess) { 
        return handle_get_profile(req, sess); 
    };
    route_handlers_["/stats"] = [this](const HttpRequest& req, const UserSession* sess) { 
        return handle_get_stats(req, sess); 
    };
    
    // API endpoints
    route_handlers_["/api/tasks"] = [this](const HttpRequest& req, const UserSession* sess) { 
        return api_get_available_tasks(req, sess); 
    };
    route_handlers_["/api/submit"] = [this](const HttpRequest& req, const UserSession* sess) { 
        return api_submit_annotation(req, sess); 
    };
    route_handlers_["/api/stats"] = [this](const HttpRequest& req, const UserSession* sess) { 
        return api_get_annotator_stats(req, sess); 
    };
    route_handlers_["/api/leaderboard"] = [this](const HttpRequest& req, const UserSession* sess) { 
        return api_get_leaderboard(req, sess); 
    };
    
    // Static files
    route_handlers_["/static/*"] = [this](const HttpRequest& req, const UserSession* sess) { 
        return handle_static_files(req, sess); 
    };
}

// Route handlers implementation
HttpResponse WebAnnotationInterface::handle_index(const HttpRequest& request, const UserSession* session) {
    HttpResponse response;
    response.content_type = "text/html";
    
    if (session && session->is_authenticated) {
        // Redirect to dashboard
        response.status_code = 302;
        response.status_text = "Found";
        response.headers["Location"] = "/dashboard";
        response.body = "";
    } else {
        response.body = AnnotationTemplateGenerator::generate_index_page();
    }
    
    return response;
}

HttpResponse WebAnnotationInterface::handle_login(const HttpRequest& request, const UserSession* session) {
    HttpResponse response;
    response.content_type = "text/html";
    
    if (request.method == "POST") {
        // Process login
        auto params = parse_query_string(request.body);
        std::string username = params["username"];
        std::string password = params["password"];
        
        // Simple authentication (in production, use proper authentication)
        if (!username.empty() && !password.empty()) {
            // Create session
            std::string session_id = create_session(username, username);
            
            response.status_code = 302;
            response.status_text = "Found";
            response.headers["Location"] = "/dashboard";
            response.headers["Set-Cookie"] = "session_id=" + session_id + "; Path=/; HttpOnly";
            response.body = "";
        } else {
            response.body = AnnotationTemplateGenerator::generate_login_page();
        }
    } else {
        response.body = AnnotationTemplateGenerator::generate_login_page();
    }
    
    return response;
}

HttpResponse WebAnnotationInterface::handle_logout(const HttpRequest& request, const UserSession* session) {
    HttpResponse response;
    
    if (session) {
        destroy_session(session->session_id);
    }
    
    response.status_code = 302;
    response.status_text = "Found";
    response.headers["Location"] = "/";
    response.headers["Set-Cookie"] = "session_id=; Path=/; HttpOnly; Expires=Thu, 01 Jan 1970 00:00:00 GMT";
    response.body = "";
    
    return response;
}

HttpResponse WebAnnotationInterface::handle_dashboard(const HttpRequest& request, const UserSession* session) {
    HttpResponse response;
    response.content_type = "text/html";
    
    if (!session || !session->is_authenticated) {
        response.status_code = 302;
        response.status_text = "Found";
        response.headers["Location"] = "/login";
        response.body = "";
        return response;
    }
    
    response.body = AnnotationTemplateGenerator::generate_dashboard_page();
    return response;
}

HttpResponse WebAnnotationInterface::handle_get_tasks(const HttpRequest& request, const UserSession* session) {
    HttpResponse response;
    response.content_type = "text/html";
    
    if (!session || !session->is_authenticated) {
        response.status_code = 302;
        response.status_text = "Found";
        response.headers["Location"] = "/login";
        response.body = "";
        return response;
    }
    
    // Get available tasks from curation platform
    auto tasks = curation_platform_->get_available_tasks(session->annotator_id);
    
    // Convert to UI format
    std::vector<AnnotationTaskUI> ui_tasks;
    for (const auto& task : tasks) {
        AnnotationTaskUI ui_task;
        ui_task.task_id = task.task_id;
        ui_task.title = "Annotation Task";
        ui_task.description = "Please annotate the following content";
        ui_task.content = task.content;
        ui_task.content_type = task.data_type;
        ui_task.label_schema = task.label_schema;
        ui_task.difficulty_score = task.difficulty_score;
        ui_task.estimated_time_minutes = task.difficulty_score * 10.0f; // Rough estimate
        ui_task.required_annotators = task.required_annotators;
        ui_task.current_annotators = 0; // Would need to query this
        
        ui_tasks.push_back(ui_task);
    }
    
    // Generate HTML with tasks
    std::string tasks_html = "<div class='tasks-container'>";
    for (const auto& task : ui_tasks) {
        tasks_html += AnnotationTemplateGenerator::generate_task_card(task);
    }
    tasks_html += "</div>";
    
    std::map<std::string, std::string> variables;
    variables["tasks_content"] = tasks_html;
    variables["task_count"] = std::to_string(ui_tasks.size());
    
    response.body = render_template("tasks", variables);
    return response;
}

HttpResponse WebAnnotationInterface::handle_get_task_details(const HttpRequest& request, const UserSession* session) {
    HttpResponse response;
    response.content_type = "text/html";
    
    if (!session || !session->is_authenticated) {
        response.status_code = 302;
        response.status_text = "Found";
        response.headers["Location"] = "/login";
        response.body = "";
        return response;
    }
    
    // Get task ID from query parameters
    std::string task_id = request.query_params.count("id") ? request.query_params.at("id") : "";
    
    if (task_id.empty()) {
        response.status_code = 400;
        response.status_text = "Bad Request";
        response.body = "<html><body><h1>400 Bad Request - Missing task ID</h1></body></html>";
        return response;
    }
    
    // Get available tasks and find the requested one
    auto tasks = curation_platform_->get_available_tasks(session->annotator_id);
    
    for (const auto& task : tasks) {
        if (task.task_id == task_id) {
            AnnotationTaskUI ui_task;
            ui_task.task_id = task.task_id;
            ui_task.title = "Annotation Task";
            ui_task.description = "Please annotate the following content";
            ui_task.content = task.content;
            ui_task.content_type = task.data_type;
            ui_task.label_schema = task.label_schema;
            ui_task.difficulty_score = task.difficulty_score;
            ui_task.estimated_time_minutes = task.difficulty_score * 10.0f;
            ui_task.required_annotators = task.required_annotators;
            ui_task.current_annotators = 0;
            
            response.body = AnnotationTemplateGenerator::generate_annotation_page(ui_task);
            return response;
        }
    }
    
    response.status_code = 404;
    response.status_text = "Not Found";
    response.body = "<html><body><h1>404 Task Not Found</h1></body></html>";
    return response;
}

HttpResponse WebAnnotationInterface::handle_submit_annotation(const HttpRequest& request, const UserSession* session) {
    HttpResponse response;
    
    if (!session || !session->is_authenticated) {
        response.status_code = 401;
        response.status_text = "Unauthorized";
        response.body = "{\"error\": \"Not authenticated\"}";
        response.content_type = "application/json";
        return response;
    }
    
    if (request.method != "POST") {
        response.status_code = 405;
        response.status_text = "Method Not Allowed";
        response.body = "{\"error\": \"Method not allowed\"}";
        response.content_type = "application/json";
        return response;
    }
    
    try {
        // Parse annotation submission from request body
        AnnotationSubmissionUI ui_submission = from_json_submission(request.body);
        
        // Convert to curation platform format
        curation::AnnotationSubmission submission;
        submission.task_id = ui_submission.task_id;
        submission.annotator_id = session->annotator_id;
        submission.confidence_score = ui_submission.confidence_score;
        submission.submission_timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        // Convert labels
        for (const auto& [label_type, score] : ui_submission.label_scores) {
            curation::AnnotationLabel label;
            label.label_type = label_type;
            label.score = score;
            label.text_feedback = ui_submission.label_comments.count(label_type) ? 
                                 ui_submission.label_comments.at(label_type) : "";
            label.annotator_id = session->annotator_id;
            label.timestamp = submission.submission_timestamp;
            
            submission.labels.push_back(label);
        }
        
        // Sign the submission (simplified)
        submission.signature = "web_signature_" + submission.task_id + "_" + submission.annotator_id;
        
        // Submit to curation platform
        bool success = curation_platform_->submit_annotation(submission);
        
        if (success) {
            // Update statistics
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                current_stats_.completed_annotations++;
            }
            
            response.status_code = 200;
            response.body = "{\"success\": true, \"message\": \"Annotation submitted successfully\"}";
        } else {
            response.status_code = 400;
            response.body = "{\"success\": false, \"message\": \"Failed to submit annotation\"}";
        }
        
        response.content_type = "application/json";
        
    } catch (const std::exception& e) {
        response.status_code = 400;
        response.status_text = "Bad Request";
        response.body = "{\"error\": \"Invalid request: " + std::string(e.what()) + "\"}";
        response.content_type = "application/json";
    }
    
    return response;
}

HttpResponse WebAnnotationInterface::handle_get_profile(const HttpRequest& request, const UserSession* session) {
    HttpResponse response;
    response.content_type = "text/html";
    
    if (!session || !session->is_authenticated) {
        response.status_code = 302;
        response.status_text = "Found";
        response.headers["Location"] = "/login";
        response.body = "";
        return response;
    }
    
    // Get annotator profile
    auto profile_opt = curation_platform_->get_annotator_profile(session->annotator_id);
    
    if (profile_opt) {
        response.body = AnnotationTemplateGenerator::generate_profile_page(profile_opt.value());
    } else {
        response.body = "<html><body><h1>Profile not found</h1></body></html>";
    }
    
    return response;
}

HttpResponse WebAnnotationInterface::handle_get_stats(const HttpRequest& request, const UserSession* session) {
    HttpResponse response;
    response.content_type = "text/html";
    
    response.body = AnnotationTemplateGenerator::generate_stats_page();
    return response;
}

HttpResponse WebAnnotationInterface::handle_static_files(const HttpRequest& request, const UserSession* session) {
    HttpResponse response;
    
    // Extract filename from path
    std::string filename = request.path.substr(8); // Remove "/static/"
    
    // Security check
    if (!utils::is_safe_filename(filename)) {
        response.status_code = 403;
        response.status_text = "Forbidden";
        response.body = "Access denied";
        return response;
    }
    
    std::string file_path = config_.static_files_path + "/" + filename;
    
    if (utils::file_exists(file_path)) {
        response.body = utils::read_file(file_path);
        
        // Set content type based on file extension
        size_t dot_pos = filename.find_last_of('.');
        if (dot_pos != std::string::npos) {
            std::string extension = filename.substr(dot_pos);
            response.content_type = utils::get_mime_type(extension);
        }
    } else {
        response.status_code = 404;
        response.status_text = "Not Found";
        response.body = "File not found";
    }
    
    return response;
}

// API endpoints
HttpResponse WebAnnotationInterface::api_get_available_tasks(const HttpRequest& request, const UserSession* session) {
    HttpResponse response;
    response.content_type = "application/json";
    
    if (!session || !session->is_authenticated) {
        response.status_code = 401;
        response.body = "{\"error\": \"Not authenticated\"}";
        return response;
    }
    
    auto tasks = curation_platform_->get_available_tasks(session->annotator_id);
    
    // Convert to UI format
    std::vector<AnnotationTaskUI> ui_tasks;
    for (const auto& task : tasks) {
        AnnotationTaskUI ui_task;
        ui_task.task_id = task.task_id;
        ui_task.title = "Annotation Task";
        ui_task.content = task.content;
        ui_task.content_type = task.data_type;
        ui_task.label_schema = task.label_schema;
        ui_task.difficulty_score = task.difficulty_score;
        ui_task.required_annotators = task.required_annotators;
        
        ui_tasks.push_back(ui_task);
    }
    
    response.body = to_json(ui_tasks);
    return response;
}

HttpResponse WebAnnotationInterface::api_submit_annotation(const HttpRequest& request, const UserSession* session) {
    return handle_submit_annotation(request, session);
}

HttpResponse WebAnnotationInterface::api_get_annotator_stats(const HttpRequest& request, const UserSession* session) {
    HttpResponse response;
    response.content_type = "application/json";
    
    if (!session || !session->is_authenticated) {
        response.status_code = 401;
        response.body = "{\"error\": \"Not authenticated\"}";
        return response;
    }
    
    auto profile_opt = curation_platform_->get_annotator_profile(session->annotator_id);
    
    if (profile_opt) {
        response.body = to_json(profile_opt.value());
    } else {
        response.status_code = 404;
        response.body = "{\"error\": \"Profile not found\"}";
    }
    
    return response;
}

HttpResponse WebAnnotationInterface::api_get_leaderboard(const HttpRequest& request, const UserSession* session) {
    HttpResponse response;
    response.content_type = "application/json";
    
    auto top_annotators = curation_platform_->get_top_annotators(50);
    
    std::string json = "[";
    for (size_t i = 0; i < top_annotators.size(); ++i) {
        if (i > 0) json += ",";
        json += to_json(top_annotators[i]);
    }
    json += "]";
    
    response.body = json;
    return response;
}

// Session management
std::string WebAnnotationInterface::create_session(const std::string& user_id, const std::string& annotator_id) {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    
    std::string session_id = generate_session_id();
    
    UserSession session;
    session.session_id = session_id;
    session.user_id = user_id;
    session.annotator_id = annotator_id;
    session.created_timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    session.last_activity = session.created_timestamp;
    session.is_authenticated = true;
    
    active_sessions_[session_id] = session;
    
    // Register annotator with curation platform if not already registered
    auto profile_opt = curation_platform_->get_annotator_profile(annotator_id);
    if (!profile_opt) {
        curation::AnnotatorProfile profile;
        profile.annotator_id = annotator_id;
        profile.public_key = "web_public_key_" + annotator_id;
        profile.reputation_score = 0.5f; // Starting reputation
        profile.total_annotations = 0;
        profile.consensus_agreements = 0;
        profile.average_confidence = 0.0f;
        profile.last_active = session.created_timestamp;
        profile.is_verified = false;
        
        curation_platform_->register_annotator(profile);
    }
    
    std::cout << "Created session for user: " << user_id << " (session: " << session_id << ")" << std::endl;
    return session_id;
}

bool WebAnnotationInterface::validate_session(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    return active_sessions_.find(session_id) != active_sessions_.end();
}

void WebAnnotationInterface::destroy_session(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    active_sessions_.erase(session_id);
    std::cout << "Destroyed session: " << session_id << std::endl;
}

std::optional<UserSession> WebAnnotationInterface::get_session(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    auto it = active_sessions_.find(session_id);
    if (it != active_sessions_.end()) {
        return it->second;
    }
    return std::nullopt;
}

// Utility methods
HttpRequest WebAnnotationInterface::parse_http_request(const std::string& raw_request) {
    HttpRequest request;
    
    std::istringstream iss(raw_request);
    std::string line;
    
    // Parse request line
    if (std::getline(iss, line)) {
        std::istringstream request_line(line);
        std::string version;
        request_line >> request.method >> request.path >> version;
        
        // Parse query string
        size_t query_pos = request.path.find('?');
        if (query_pos != std::string::npos) {
            request.query_string = request.path.substr(query_pos + 1);
            request.path = request.path.substr(0, query_pos);
            request.query_params = parse_query_string(request.query_string);
        }
    }
    
    // Parse headers
    while (std::getline(iss, line) && line != "\r" && !line.empty()) {
        size_t colon_pos = line.find(':');
        if (colon_pos != std::string::npos) {
            std::string key = line.substr(0, colon_pos);
            std::string value = line.substr(colon_pos + 1);
            
            // Trim whitespace
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t\r\n") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t\r\n") + 1);
            
            std::transform(key.begin(), key.end(), key.begin(), ::tolower);
            request.headers[key] = value;
        }
    }
    
    // Parse body
    std::string body_line;
    while (std::getline(iss, body_line)) {
        request.body += body_line + "\n";
    }
    if (!request.body.empty()) {
        request.body.pop_back(); // Remove last newline
    }
    
    return request;
}

std::string WebAnnotationInterface::serialize_http_response(const HttpResponse& response) {
    std::ostringstream oss;
    
    // Status line
    oss << "HTTP/1.1 " << response.status_code << " " << response.status_text << "\r\n";
    
    // Headers
    oss << "Content-Type: " << response.content_type << "\r\n";
    oss << "Content-Length: " << response.body.length() << "\r\n";
    
    for (const auto& [key, value] : response.headers) {
        oss << key << ": " << value << "\r\n";
    }
    
    oss << "\r\n";
    
    // Body
    oss << response.body;
    
    return oss.str();
}

std::map<std::string, std::string> WebAnnotationInterface::parse_query_string(const std::string& query_string) {
    std::map<std::string, std::string> params;
    
    std::istringstream iss(query_string);
    std::string pair;
    
    while (std::getline(iss, pair, '&')) {
        size_t eq_pos = pair.find('=');
        if (eq_pos != std::string::npos) {
            std::string key = url_decode(pair.substr(0, eq_pos));
            std::string value = url_decode(pair.substr(eq_pos + 1));
            params[key] = value;
        }
    }
    
    return params;
}

std::string WebAnnotationInterface::url_decode(const std::string& encoded) {
    std::string decoded;
    
    for (size_t i = 0; i < encoded.length(); ++i) {
        if (encoded[i] == '%' && i + 2 < encoded.length()) {
            int hex_value;
            std::istringstream hex_stream(encoded.substr(i + 1, 2));
            hex_stream >> std::hex >> hex_value;
            decoded += static_cast<char>(hex_value);
            i += 2;
        } else if (encoded[i] == '+') {
            decoded += ' ';
        } else {
            decoded += encoded[i];
        }
    }
    
    return decoded;
}

std::string WebAnnotationInterface::generate_session_id() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);
    
    std::string session_id;
    for (int i = 0; i < 32; ++i) {
        int val = dis(gen);
        session_id += (val < 10) ? ('0' + val) : ('a' + val - 10);
    }
    
    return session_id;
}

// JSON helpers (simplified implementations)
std::string WebAnnotationInterface::to_json(const AnnotationTaskUI& task) {
    std::ostringstream oss;
    oss << "{";
    oss << "\"task_id\":\"" << utils::escape_json(task.task_id) << "\",";
    oss << "\"title\":\"" << utils::escape_json(task.title) << "\",";
    oss << "\"content\":\"" << utils::escape_json(task.content) << "\",";
    oss << "\"content_type\":\"" << utils::escape_json(task.content_type) << "\",";
    oss << "\"difficulty_score\":" << task.difficulty_score << ",";
    oss << "\"required_annotators\":" << task.required_annotators;
    oss << "}";
    return oss.str();
}

std::string WebAnnotationInterface::to_json(const std::vector<AnnotationTaskUI>& tasks) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < tasks.size(); ++i) {
        if (i > 0) oss << ",";
        oss << to_json(tasks[i]);
    }
    oss << "]";
    return oss.str();
}

std::string WebAnnotationInterface::to_json(const curation::AnnotatorProfile& profile) {
    std::ostringstream oss;
    oss << "{";
    oss << "\"annotator_id\":\"" << utils::escape_json(profile.annotator_id) << "\",";
    oss << "\"reputation_score\":" << profile.reputation_score << ",";
    oss << "\"total_annotations\":" << profile.total_annotations << ",";
    oss << "\"consensus_agreements\":" << profile.consensus_agreements << ",";
    oss << "\"average_confidence\":" << profile.average_confidence;
    oss << "}";
    return oss.str();
}

AnnotationSubmissionUI WebAnnotationInterface::from_json_submission(const std::string& json) {
    // Simplified JSON parsing - in production, use a proper JSON library
    AnnotationSubmissionUI submission;
    
    // Extract task_id
    std::regex task_id_regex("\"task_id\"\\s*:\\s*\"([^\"]+)\"");
    std::smatch match;
    if (std::regex_search(json, match, task_id_regex)) {
        submission.task_id = match[1].str();
    }
    
    // Extract confidence_score
    std::regex confidence_regex("\"confidence_score\"\\s*:\\s*([0-9.]+)");
    if (std::regex_search(json, match, confidence_regex)) {
        submission.confidence_score = std::stof(match[1].str());
    }
    
    // Extract label scores (simplified)
    std::regex label_regex("\"([^\"]+)_score\"\\s*:\\s*([0-9.]+)");
    std::sregex_iterator iter(json.begin(), json.end(), label_regex);
    std::sregex_iterator end;
    
    for (; iter != end; ++iter) {
        std::string label_type = (*iter)[1].str();
        float score = std::stof((*iter)[2].str());
        submission.label_scores[label_type] = score;
    }
    
    return submission;
}

WebAnnotationInterface::InterfaceStats WebAnnotationInterface::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    InterfaceStats stats = current_stats_;
    
    // Update active sessions count
    {
        std::lock_guard<std::mutex> sessions_lock(sessions_mutex_);
        stats.active_sessions = active_sessions_.size();
    }
    
    // Get active tasks from curation platform
    auto platform_stats = curation_platform_->get_platform_stats();
    stats.active_tasks = platform_stats.total_tasks - platform_stats.completed_tasks;
    
    return stats;
}

// Template rendering (simplified)
std::string WebAnnotationInterface::render_template(const std::string& template_name, 
                                                   const std::map<std::string, std::string>& variables) {
    std::string template_content = load_template(template_name);
    return replace_template_variables(template_content, variables);
}

std::string WebAnnotationInterface::load_template(const std::string& template_name) {
    // For now, return hardcoded templates
    if (template_name == "tasks") {
        return "<html><body><h1>Available Tasks</h1><p>Task count: {{task_count}}</p>{{tasks_content}}</body></html>";
    }
    return "<html><body><h1>Template not found</h1></body></html>";
}

std::string WebAnnotationInterface::replace_template_variables(const std::string& template_content,
                                                             const std::map<std::string, std::string>& variables) {
    std::string result = template_content;
    
    for (const auto& [key, value] : variables) {
        std::string placeholder = "{{" + key + "}}";
        size_t pos = 0;
        while ((pos = result.find(placeholder, pos)) != std::string::npos) {
            result.replace(pos, placeholder.length(), value);
            pos += value.length();
        }
    }
    
    return result;
}

// AnnotationTemplateGenerator implementation
std::string AnnotationTemplateGenerator::generate_index_page() {
    return wrap_in_layout(R"(
        <div class="hero">
            <h1>Welcome to the Annotation Platform</h1>
            <p>Help improve AI systems by providing high-quality annotations</p>
            <a href="/login" class="btn btn-primary">Get Started</a>
        </div>
        <div class="features">
            <div class="feature">
                <h3>Distributed</h3>
                <p>Decentralized platform with no single point of failure</p>
            </div>
            <div class="feature">
                <h3>Secure</h3>
                <p>Cryptographically signed annotations with reputation system</p>
            </div>
            <div class="feature">
                <h3>Rewarding</h3>
                <p>Earn rewards for high-quality annotations</p>
            </div>
        </div>
    )", "Annotation Platform");
}

std::string AnnotationTemplateGenerator::generate_login_page() {
    return wrap_in_layout(R"(
        <div class="login-form">
            <h2>Login</h2>
            <form method="post" action="/login">
                <div class="form-group">
                    <label for="username">Username:</label>
                    <input type="text" id="username" name="username" required>
                </div>
                <div class="form-group">
                    <label for="password">Password:</label>
                    <input type="password" id="password" name="password" required>
                </div>
                <button type="submit" class="btn btn-primary">Login</button>
            </form>
        </div>
    )", "Login");
}

std::string AnnotationTemplateGenerator::generate_dashboard_page() {
    return wrap_in_layout(R"(
        <div class="dashboard">
            <h1>Dashboard</h1>
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>Available Tasks</h3>
                    <p class="stat-number" id="available-tasks">Loading...</p>
                </div>
                <div class="stat-card">
                    <h3>Completed Annotations</h3>
                    <p class="stat-number" id="completed-annotations">Loading...</p>
                </div>
                <div class="stat-card">
                    <h3>Reputation Score</h3>
                    <p class="stat-number" id="reputation-score">Loading...</p>
                </div>
            </div>
            <div class="actions">
                <a href="/tasks" class="btn btn-primary">View Available Tasks</a>
                <a href="/profile" class="btn btn-secondary">View Profile</a>
            </div>
        </div>
        <script>
            // Load dashboard data
            fetch('/api/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('completed-annotations').textContent = data.total_annotations || 0;
                    document.getElementById('reputation-score').textContent = (data.reputation_score || 0).toFixed(2);
                });
            
            fetch('/api/tasks')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('available-tasks').textContent = data.length || 0;
                });
        </script>
    )", "Dashboard");
}

std::string AnnotationTemplateGenerator::generate_task_card(const AnnotationTaskUI& task) {
    std::ostringstream oss;
    oss << "<div class='task-card'>";
    oss << "<h3>" << utils::escape_html(task.title) << "</h3>";
    oss << "<p class='task-content'>" << utils::escape_html(task.content.substr(0, 200));
    if (task.content.length() > 200) oss << "...";
    oss << "</p>";
    oss << "<div class='task-meta'>";
    oss << "<span class='difficulty'>Difficulty: " << task.difficulty_score << "</span>";
    oss << "<span class='time'>Est. " << task.estimated_time_minutes << " min</span>";
    oss << "<span class='progress'>" << task.current_annotators << "/" << task.required_annotators << " annotators</span>";
    oss << "</div>";
    oss << "<a href='/task?id=" << task.task_id << "' class='btn btn-primary'>Annotate</a>";
    oss << "</div>";
    return oss.str();
}

std::string AnnotationTemplateGenerator::wrap_in_layout(const std::string& content, const std::string& title) {
    std::ostringstream oss;
    oss << "<!DOCTYPE html><html><head>";
    oss << "<title>" << utils::escape_html(title) << "</title>";
    oss << "<style>" << generate_css() << "</style>";
    oss << "</head><body>";
    oss << generate_navigation(true);
    oss << "<main>" << content << "</main>";
    oss << "<script>" << generate_javascript() << "</script>";
    oss << "</body></html>";
    return oss.str();
}

std::string AnnotationTemplateGenerator::generate_navigation(bool is_authenticated) {
    std::ostringstream oss;
    oss << "<nav class='navbar'>";
    oss << "<div class='nav-brand'><a href='/'>Annotation Platform</a></div>";
    oss << "<div class='nav-links'>";
    if (is_authenticated) {
        oss << "<a href='/dashboard'>Dashboard</a>";
        oss << "<a href='/tasks'>Tasks</a>";
        oss << "<a href='/profile'>Profile</a>";
        oss << "<a href='/logout'>Logout</a>";
    } else {
        oss << "<a href='/login'>Login</a>";
    }
    oss << "</div>";
    oss << "</nav>";
    return oss.str();
}

std::string AnnotationTemplateGenerator::generate_css() {
    return R"(
        body { font-family: Arial, sans-serif; margin: 0; padding: 0; background: #f5f5f5; }
        .navbar { background: #333; color: white; padding: 1rem; display: flex; justify-content: space-between; }
        .nav-brand a { color: white; text-decoration: none; font-weight: bold; }
        .nav-links a { color: white; text-decoration: none; margin-left: 1rem; }
        main { padding: 2rem; max-width: 1200px; margin: 0 auto; }
        .btn { padding: 0.5rem 1rem; background: #007bff; color: white; text-decoration: none; border-radius: 4px; border: none; cursor: pointer; }
        .btn:hover { background: #0056b3; }
        .btn-secondary { background: #6c757d; }
        .task-card { background: white; padding: 1rem; margin: 1rem 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .task-meta { display: flex; gap: 1rem; margin: 0.5rem 0; font-size: 0.9em; color: #666; }
        .form-group { margin: 1rem 0; }
        .form-group label { display: block; margin-bottom: 0.5rem; }
        .form-group input { width: 100%; padding: 0.5rem; border: 1px solid #ddd; border-radius: 4px; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 2rem 0; }
        .stat-card { background: white; padding: 1.5rem; border-radius: 8px; text-align: center; }
        .stat-number { font-size: 2rem; font-weight: bold; color: #007bff; }
    )";
}

std::string AnnotationTemplateGenerator::generate_javascript() {
    return R"(
        // Basic JavaScript for the annotation interface
        function submitAnnotation(taskId) {
            const form = document.getElementById('annotation-form');
            const formData = new FormData(form);
            
            const submission = {
                task_id: taskId,
                label_scores: {},
                label_comments: {},
                confidence_score: parseFloat(formData.get('confidence')) || 0.5
            };
            
            // Collect label scores
            for (let [key, value] of formData.entries()) {
                if (key.endsWith('_score')) {
                    submission.label_scores[key.replace('_score', '')] = parseFloat(value);
                } else if (key.endsWith('_comment')) {
                    submission.label_comments[key.replace('_comment', '')] = value;
                }
            }
            
            fetch('/api/submit', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(submission)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('Annotation submitted successfully!');
                    window.location.href = '/tasks';
                } else {
                    alert('Error: ' + data.message);
                }
            })
            .catch(error => {
                alert('Error submitting annotation: ' + error);
            });
        }
    )";
}

// Utility functions
namespace utils {

std::string get_mime_type(const std::string& file_extension) {
    if (file_extension == ".html" || file_extension == ".htm") return "text/html";
    if (file_extension == ".css") return "text/css";
    if (file_extension == ".js") return "application/javascript";
    if (file_extension == ".json") return "application/json";
    if (file_extension == ".png") return "image/png";
    if (file_extension == ".jpg" || file_extension == ".jpeg") return "image/jpeg";
    if (file_extension == ".gif") return "image/gif";
    if (file_extension == ".svg") return "image/svg+xml";
    return "application/octet-stream";
}

std::string escape_html(const std::string& text) {
    std::string escaped;
    for (char c : text) {
        switch (c) {
            case '<': escaped += "&lt;"; break;
            case '>': escaped += "&gt;"; break;
            case '&': escaped += "&amp;"; break;
            case '"': escaped += "&quot;"; break;
            case '\'': escaped += "&#39;"; break;
            default: escaped += c; break;
        }
    }
    return escaped;
}

std::string escape_json(const std::string& text) {
    std::string escaped;
    for (char c : text) {
        switch (c) {
            case '"': escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default: escaped += c; break;
        }
    }
    return escaped;
}

bool file_exists(const std::string& path) {
    std::ifstream file(path);
    return file.good();
}

std::string read_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return "";
    }
    
    std::ostringstream oss;
    oss << file.rdbuf();
    return oss.str();
}

bool is_safe_filename(const std::string& filename) {
    // Basic security check - no path traversal
    return filename.find("..") == std::string::npos && 
           filename.find("/") == std::string::npos &&
           filename.find("\\") == std::string::npos;
}

} // namespace utils

} // namespace web_interface
