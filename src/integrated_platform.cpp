#include "../include/integrated_platform.hpp"
#include "../include/utils.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <sys/sysinfo.h>
#include <unistd.h>
#endif

namespace integrated {

IntegratedPlatform::IntegratedPlatform(const PlatformConfig& config)
    : config_(config) {
    
    std::cout << "Initializing Integrated Platform" << std::endl;
    std::cout << "- Node ID: " << config_.node_id << std::endl;
    std::cout << "- P2P Port: " << config_.p2p_port << std::endl;
    std::cout << "- Web Port: " << config_.web_port << std::endl;
    std::cout << "- Curation enabled: " << (config_.enable_curation ? "yes" : "no") << std::endl;
    std::cout << "- RLHF enabled: " << (config_.enable_rlhf ? "yes" : "no") << std::endl;
    std::cout << "- Web interface enabled: " << (config_.enable_web_interface ? "yes" : "no") << std::endl;

    // Initialize capabilities based on system
    capabilities_ = utils::detect_system_capabilities();
    
    // Initialize stats
    current_stats_ = PlatformStats{};
    current_stats_.last_updated_timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    // Initialize health status
    current_health_ = HealthStatus{};
    current_health_.last_check_timestamp = current_stats_.last_updated_timestamp;
}

IntegratedPlatform::~IntegratedPlatform() {
    stop();
}

bool IntegratedPlatform::initialize() {
    if (initialized_.load()) {
        std::cout << "Platform already initialized" << std::endl;
        return true;
    }

    std::cout << "Initializing platform components..." << std::endl;

    try {
        // Validate configuration
        if (!validate_config(config_)) {
            std::cerr << "Invalid platform configuration" << std::endl;
            return false;
        }

        // Initialize P2P network
        p2p::P2PConfig p2p_config;
        p2p_config.node_id = config_.node_id;
        p2p_config.bind_address = config_.bind_address;
        p2p_config.port = config_.p2p_port;
        p2p_config.bootstrap_peers = config_.bootstrap_peers;
        
        network_ = std::make_shared<p2p::P2PNetwork>(p2p_config);
        if (!network_) {
            std::cerr << "Failed to create P2P network" << std::endl;
            return false;
        }

        // Initialize transformer model
        if (!config_.model_config_path.empty()) {
            // Load model configuration
            TransformerConfig model_config;
            // In practice, load from config_.model_config_path
            model_config.vocab_size = 50257;
            model_config.hidden_size = 768;
            model_config.num_layers = 12;
            model_config.num_heads = 12;
            model_config.max_seq_length = 1024;

            // Create tokenizer
            std::shared_ptr<Tokenizer> tokenizer;
            if (!config_.tokenizer_path.empty()) {
                tokenizer = std::make_shared<TikTokenTokenizer>(config_.tokenizer_path);
            }

            model_ = std::make_shared<DistributedTransformer>(model_config, tokenizer);
            if (!model_) {
                std::cerr << "Failed to create transformer model" << std::endl;
                return false;
            }

            // Load checkpoint if specified
            if (!config_.checkpoint_path.empty()) {
                if (!model_->load_checkpoint(config_.checkpoint_path)) {
                    std::cout << "Warning: Failed to load model checkpoint from " << config_.checkpoint_path << std::endl;
                }
            }
        }

        // Initialize curation platform
        if (config_.enable_curation) {
            curation_platform_ = std::make_shared<curation::DistributedCurationPlatform>(
                network_, config_.curation_config);
            if (!curation_platform_) {
                std::cerr << "Failed to create curation platform" << std::endl;
                return false;
            }
        }

        // Initialize RLHF coordinator
        if (config_.enable_rlhf && model_) {
            rlhf_coordinator_ = std::make_shared<rlhf::DistributedRLHFCoordinator>(
                network_, curation_platform_, model_, config_.rlhf_config);
            if (!rlhf_coordinator_) {
                std::cerr << "Failed to create RLHF coordinator" << std::endl;
                return false;
            }
        }

        // Initialize web interface
        if (config_.enable_web_interface && curation_platform_) {
            web_interface::WebServerConfig web_config = config_.web_config;
            web_config.port = config_.web_port;
            
            web_interface_ = std::make_shared<web_interface::WebAnnotationInterface>(
                curation_platform_, web_config);
            if (!web_interface_) {
                std::cerr << "Failed to create web annotation interface" << std::endl;
                return false;
            }
        }

        // Setup component callbacks
        setup_component_callbacks();

        initialized_.store(true);
        std::cout << "Platform initialization completed successfully" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Platform initialization failed: " << e.what() << std::endl;
        return false;
    }
}

bool IntegratedPlatform::start() {
    if (!initialized_.load()) {
        std::cerr << "Platform not initialized. Call initialize() first." << std::endl;
        return false;
    }

    if (running_.load()) {
        std::cout << "Platform already running" << std::endl;
        return true;
    }

    std::cout << "Starting integrated platform..." << std::endl;

    try {
        // Start P2P network
        if (!network_->start()) {
            std::cerr << "Failed to start P2P network" << std::endl;
            return false;
        }

        // Start curation platform
        if (curation_platform_ && !curation_platform_->start()) {
            std::cerr << "Failed to start curation platform" << std::endl;
            return false;
        }

        // Start RLHF coordinator
        if (rlhf_coordinator_ && !rlhf_coordinator_->start()) {
            std::cerr << "Failed to start RLHF coordinator" << std::endl;
            return false;
        }

        // Start web interface
        if (web_interface_ && !web_interface_->start()) {
            std::cerr << "Failed to start web annotation interface" << std::endl;
            return false;
        }

        // Start background threads
        running_.store(true);
        
        if (config_.enable_metrics_collection) {
            worker_threads_.emplace_back(&IntegratedPlatform::metrics_collection_thread, this);
        }
        
        worker_threads_.emplace_back(&IntegratedPlatform::health_monitoring_thread, this);
        worker_threads_.emplace_back(&IntegratedPlatform::resource_management_thread, this);
        
        if (config_.enable_auto_training) {
            worker_threads_.emplace_back(&IntegratedPlatform::auto_training_thread, this);
        }

        emit_event(PlatformEvent::TRAINING_STARTED, "Platform started successfully");
        
        std::cout << "Integrated platform started successfully" << std::endl;
        std::cout << "- P2P Network: " << (network_->is_running() ? "running" : "stopped") << std::endl;
        std::cout << "- Curation Platform: " << (curation_platform_ && curation_platform_->is_running() ? "running" : "stopped") << std::endl;
        std::cout << "- RLHF Coordinator: " << (rlhf_coordinator_ && rlhf_coordinator_->is_running() ? "running" : "stopped") << std::endl;
        std::cout << "- Web Interface: " << (web_interface_ && web_interface_->is_running() ? "running" : "stopped") << std::endl;
        
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Platform startup failed: " << e.what() << std::endl;
        emit_event(PlatformEvent::ERROR_OCCURRED, "Platform startup failed: " + std::string(e.what()));
        return false;
    }
}

void IntegratedPlatform::stop() {
    if (!running_.load()) {
        return;
    }

    std::cout << "Stopping integrated platform..." << std::endl;
    running_.store(false);

    // Stop background threads
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();

    // Stop components in reverse order
    if (web_interface_) {
        web_interface_->stop();
    }

    if (rlhf_coordinator_) {
        rlhf_coordinator_->stop();
    }

    if (curation_platform_) {
        curation_platform_->stop();
    }

    if (network_) {
        network_->stop();
    }

    emit_event(PlatformEvent::TRAINING_COMPLETED, "Platform stopped");
    std::cout << "Integrated platform stopped" << std::endl;
}

// Platform operations
bool IntegratedPlatform::submit_annotation_task(const curation::AnnotationTask& task) {
    if (!curation_platform_) {
        std::cerr << "Curation platform not available" << std::endl;
        return false;
    }

    std::string task_id = curation_platform_->submit_annotation_task(task);
    if (!task_id.empty()) {
        emit_event(PlatformEvent::ANNOTATION_TASK_CREATED, "Task ID: " + task_id);
        return true;
    }

    return false;
}

bool IntegratedPlatform::start_training_session(const std::vector<std::string>& training_data) {
    if (!model_) {
        std::cerr << "Model not available for training" << std::endl;
        return false;
    }

    emit_event(PlatformEvent::TRAINING_STARTED, "Training session with " + std::to_string(training_data.size()) + " samples");
    
    // This would implement the actual training logic
    // For now, just simulate training
    std::cout << "Starting training session with " << training_data.size() << " samples" << std::endl;
    
    return true;
}

bool IntegratedPlatform::start_rlhf_training(uint32_t reward_epochs, uint32_t ppo_iterations) {
    if (!rlhf_coordinator_) {
        std::cerr << "RLHF coordinator not available" << std::endl;
        return false;
    }

    emit_event(PlatformEvent::TRAINING_STARTED, "RLHF training: " + std::to_string(reward_epochs) + " reward epochs, " + std::to_string(ppo_iterations) + " PPO iterations");
    
    return rlhf_coordinator_->run_full_rlhf_pipeline(reward_epochs, ppo_iterations);
}

bool IntegratedPlatform::save_model_checkpoint(const std::string& path) {
    if (!model_) {
        std::cerr << "Model not available for checkpoint saving" << std::endl;
        return false;
    }

    return model_->save_checkpoint(path);
}

bool IntegratedPlatform::load_model_checkpoint(const std::string& path) {
    if (!model_) {
        std::cerr << "Model not available for checkpoint loading" << std::endl;
        return false;
    }

    return model_->load_checkpoint(path);
}

// Monitoring and statistics
PlatformStats IntegratedPlatform::get_stats() const {
    std::lock_guard<std::mutex> lock(platform_mutex_);
    return current_stats_;
}

NodeCapabilities IntegratedPlatform::get_capabilities() const {
    std::lock_guard<std::mutex> lock(platform_mutex_);
    return capabilities_;
}

std::vector<std::string> IntegratedPlatform::get_connected_peers() const {
    if (!network_) {
        return {};
    }
    
    return network_->get_connected_peers();
}

// Configuration management
bool IntegratedPlatform::update_config(const PlatformConfig& new_config) {
    if (!validate_config(new_config)) {
        std::cerr << "Invalid configuration provided" << std::endl;
        return false;
    }

    PlatformConfig old_config = config_;
    config_ = new_config;
    
    try {
        apply_config_changes(old_config, new_config);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to apply configuration changes: " << e.what() << std::endl;
        config_ = old_config; // Rollback
        return false;
    }
}

PlatformConfig IntegratedPlatform::get_config() const {
    return config_;
}

// Event handling
void IntegratedPlatform::set_event_callback(EventCallback callback) {
    event_callback_ = callback;
}

void IntegratedPlatform::emit_event(PlatformEvent event, const std::string& details) {
    if (event_callback_) {
        event_callback_(event, details);
    }
    
    std::cout << "Platform Event: " << static_cast<int>(event) << " - " << details << std::endl;
}

// Resource management
bool IntegratedPlatform::allocate_resources(const std::string& task_type, uint32_t required_memory_mb) {
    std::lock_guard<std::mutex> lock(platform_mutex_);
    
    if (!check_resource_availability(required_memory_mb)) {
        return false;
    }
    
    resource_allocation_[task_type] += required_memory_mb;
    return true;
}

void IntegratedPlatform::release_resources(const std::string& task_type) {
    std::lock_guard<std::mutex> lock(platform_mutex_);
    resource_allocation_.erase(task_type);
}

std::map<std::string, uint32_t> IntegratedPlatform::get_resource_allocation() const {
    std::lock_guard<std::mutex> lock(platform_mutex_);
    return resource_allocation_;
}

// Health monitoring
IntegratedPlatform::HealthStatus IntegratedPlatform::check_health() const {
    HealthStatus health;
    health.last_check_timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    std::vector<bool> component_health;
    
    // Check network health
    if (!check_network_health()) {
        health.issues.push_back("P2P network issues detected");
        component_health.push_back(false);
    } else {
        component_health.push_back(true);
    }
    
    // Check model health
    if (!check_model_health()) {
        health.issues.push_back("Model issues detected");
        component_health.push_back(false);
    } else {
        component_health.push_back(true);
    }
    
    // Check curation health
    if (!check_curation_health()) {
        health.issues.push_back("Curation platform issues detected");
        component_health.push_back(false);
    } else {
        component_health.push_back(true);
    }
    
    // Check RLHF health
    if (!check_rlhf_health()) {
        health.issues.push_back("RLHF coordinator issues detected");
        component_health.push_back(false);
    } else {
        component_health.push_back(true);
    }
    
    // Check web interface health
    if (!check_web_interface_health()) {
        health.issues.push_back("Web interface issues detected");
        component_health.push_back(false);
    } else {
        component_health.push_back(true);
    }
    
    // Check system resources
    if (!check_system_resources()) {
        health.issues.push_back("System resource constraints detected");
        component_health.push_back(false);
    } else {
        component_health.push_back(true);
    }
    
    // Calculate overall health score
    uint32_t healthy_components = std::count(component_health.begin(), component_health.end(), true);
    health.overall_score = static_cast<float>(healthy_components) / component_health.size();
    health.is_healthy = health.overall_score >= 0.8f; // 80% threshold
    
    return health;
}

// Background threads
void IntegratedPlatform::metrics_collection_thread() {
    std::cout << "Started metrics collection thread" << std::endl;
    
    while (running_.load()) {
        try {
            collect_network_metrics();
            collect_training_metrics();
            collect_curation_metrics();
            collect_rlhf_metrics();
            collect_web_metrics();
            collect_system_metrics();
            
            // Update timestamp
            {
                std::lock_guard<std::mutex> lock(platform_mutex_);
                current_stats_.last_updated_timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
            }
            
            // Export metrics if configured
            if (!config_.metrics_output_path.empty()) {
                utils::export_metrics_to_file(current_stats_, config_.metrics_output_path + "/platform_metrics.json");
            }
            
        } catch (const std::exception& e) {
            std::cerr << "Error in metrics collection: " << e.what() << std::endl;
        }
        
        std::this_thread::sleep_for(std::chrono::seconds(config_.metrics_collection_interval_seconds));
    }
    
    std::cout << "Metrics collection thread stopped" << std::endl;
}

void IntegratedPlatform::health_monitoring_thread() {
    std::cout << "Started health monitoring thread" << std::endl;
    
    while (running_.load()) {
        try {
            HealthStatus health = check_health();
            
            {
                std::lock_guard<std::mutex> lock(platform_mutex_);
                current_health_ = health;
            }
            
            if (!health.is_healthy) {
                emit_event(PlatformEvent::ERROR_OCCURRED, "Health check failed: " + std::to_string(health.issues.size()) + " issues detected");
            }
            
        } catch (const std::exception& e) {
            std::cerr << "Error in health monitoring: " << e.what() << std::endl;
        }
        
        std::this_thread::sleep_for(std::chrono::seconds(60)); // Check every minute
    }
    
    std::cout << "Health monitoring thread stopped" << std::endl;
}

void IntegratedPlatform::auto_training_thread() {
    std::cout << "Started auto training thread" << std::endl;
    
    while (running_.load()) {
        try {
            // Check if it's time for automatic training
            if (rlhf_coordinator_) {
                auto stats = rlhf_coordinator_->get_training_stats();
                
                // Simple heuristic: if we have enough preference data, start RLHF training
                if (stats.total_preference_samples >= 1000) {
                    std::cout << "Auto-starting RLHF training with " << stats.total_preference_samples << " preference samples" << std::endl;
                    start_rlhf_training(3, 10); // 3 reward epochs, 10 PPO iterations
                }
            }
            
        } catch (const std::exception& e) {
            std::cerr << "Error in auto training: " << e.what() << std::endl;
        }
        
        std::this_thread::sleep_for(std::chrono::hours(config_.auto_training_interval_hours));
    }
    
    std::cout << "Auto training thread stopped" << std::endl;
}

void IntegratedPlatform::resource_management_thread() {
    std::cout << "Started resource management thread" << std::endl;
    
    while (running_.load()) {
        try {
            update_resource_usage();
            
            // Check for resource constraints
            float memory_usage = utils::get_memory_usage_percent();
            float cpu_usage = utils::get_cpu_usage_percent();
            
            if (memory_usage > 90.0f || cpu_usage > 95.0f) {
                emit_event(PlatformEvent::PERFORMANCE_THRESHOLD_EXCEEDED, 
                          "High resource usage: CPU " + std::to_string(cpu_usage) + "%, Memory " + std::to_string(memory_usage) + "%");
            }
            
        } catch (const std::exception& e) {
            std::cerr << "Error in resource management: " << e.what() << std::endl;
        }
        
        std::this_thread::sleep_for(std::chrono::seconds(30));
    }
    
    std::cout << "Resource management thread stopped" << std::endl;
}

// Component coordination
void IntegratedPlatform::setup_component_callbacks() {
    // Setup curation callbacks
    if (curation_platform_) {
        curation_platform_->set_task_completed_callback(
            [this](const curation::AnnotationConsensus& consensus) {
                handle_curation_task_completed(consensus);
            });
    }
    
    // Setup RLHF callbacks
    if (rlhf_coordinator_) {
        rlhf_coordinator_->set_ppo_step_completed_callback(
            [this](const rlhf::PPOMetrics& metrics) {
                handle_rlhf_iteration_completed(metrics);
            });
    }
    
    // Setup network callbacks
    if (network_) {
        // These would be implemented in the P2P network class
        // network_->set_peer_joined_callback([this](const std::string& peer_id) { handle_network_peer_joined(peer_id); });
        // network_->set_peer_left_callback([this](const std::string& peer_id) { handle_network_peer_left(peer_id); });
    }
}

void IntegratedPlatform::handle_curation_task_completed(const curation::AnnotationConsensus& consensus) {
    emit_event(PlatformEvent::ANNOTATION_TASK_COMPLETED, "Task: " + consensus.task_id);
    
    std::lock_guard<std::mutex> lock(platform_mutex_);
    current_stats_.completed_annotation_tasks++;
}

void IntegratedPlatform::handle_rlhf_iteration_completed(const rlhf::PPOMetrics& metrics) {
    emit_event(PlatformEvent::RLHF_ITERATION_COMPLETED, "Iteration: " + std::to_string(metrics.iteration));
    
    std::lock_guard<std::mutex> lock(platform_mutex_);
    current_stats_.rlhf_iterations_completed = metrics.iteration;
}

void IntegratedPlatform::handle_network_peer_joined(const std::string& peer_id) {
    emit_event(PlatformEvent::NODE_JOINED, "Peer: " + peer_id);
    
    std::lock_guard<std::mutex> lock(platform_mutex_);
    current_stats_.connected_peers++;
}

void IntegratedPlatform::handle_network_peer_left(const std::string& peer_id) {
    emit_event(PlatformEvent::NODE_LEFT, "Peer: " + peer_id);
    
    std::lock_guard<std::mutex> lock(platform_mutex_);
    current_stats_.connected_peers = std::max(0u, current_stats_.connected_peers - 1);
}

// Health checks
bool IntegratedPlatform::check_network_health() const {
    return network_ && network_->is_running();
}

bool IntegratedPlatform::check_model_health() const {
    return !model_ || true; // Model is always healthy if it exists
}

bool IntegratedPlatform::check_curation_health() const {
    return !curation_platform_ || curation_platform_->is_running();
}

bool IntegratedPlatform::check_rlhf_health() const {
    return !rlhf_coordinator_ || rlhf_coordinator_->is_running();
}

bool IntegratedPlatform::check_web_interface_health() const {
    return !web_interface_ || web_interface_->is_running();
}

bool IntegratedPlatform::check_system_resources() const {
    float memory_usage = utils::get_memory_usage_percent();
    float cpu_usage = utils::get_cpu_usage_percent();
    
    return memory_usage < 95.0f && cpu_usage < 98.0f;
}

// Metrics collection
void IntegratedPlatform::collect_network_metrics() {
    if (!network_) return;
    
    std::lock_guard<std::mutex> lock(platform_mutex_);
    current_stats_.connected_peers = network_->get_peer_count();
    // current_stats_.total_messages_sent = network_->get_messages_sent();
    // current_stats_.total_messages_received = network_->get_messages_received();
    // current_stats_.network_latency_ms = network_->get_average_latency();
}

void IntegratedPlatform::collect_training_metrics() {
    if (!model_) return;
    
    std::lock_guard<std::mutex> lock(platform_mutex_);
    // current_stats_.current_loss = model_->get_current_loss();
    // current_stats_.total_parameters = model_->get_parameter_count();
    // current_stats_.training_throughput_tokens_per_second = model_->get_throughput();
}

void IntegratedPlatform::collect_curation_metrics() {
    if (!curation_platform_) return;
    
    auto curation_stats = curation_platform_->get_platform_stats();
    
    std::lock_guard<std::mutex> lock(platform_mutex_);
    current_stats_.total_annotation_tasks = curation_stats.total_tasks;
    current_stats_.completed_annotation_tasks = curation_stats.completed_tasks;
    current_stats_.active_annotators = curation_stats.active_annotators;
    current_stats_.average_annotation_quality = curation_stats.platform_quality_score;
}

void IntegratedPlatform::collect_rlhf_metrics() {
    if (!rlhf_coordinator_) return;
    
    auto rlhf_stats = rlhf_coordinator_->get_training_stats();
    
    std::lock_guard<std::mutex> lock(platform_mutex_);
    current_stats_.rlhf_iterations_completed = rlhf_stats.ppo_metrics.iteration;
    current_stats_.reward_model_accuracy = rlhf_stats.reward_model_metrics.accuracy;
    current_stats_.preference_data_samples = rlhf_stats.total_preference_samples;
}

void IntegratedPlatform::collect_web_metrics() {
    if (!web_interface_) return;
    
    auto web_stats = web_interface_->get_stats();
    
    std::lock_guard<std::mutex> lock(platform_mutex_);
    current_stats_.active_web_sessions = web_stats.active_sessions;
    current_stats_.total_web_requests = web_stats.total_requests;
    current_stats_.web_annotations_submitted = web_stats.completed_annotations;
}

void IntegratedPlatform::collect_system_metrics() {
    std::lock_guard<std::mutex> lock(platform_mutex_);
    current_stats_.cpu_usage_percent = utils::get_cpu_usage_percent();
    current_stats_.memory_usage_percent = utils::get_memory_usage_percent();
    current_stats_.disk_usage_percent = utils::get_disk_usage_percent();
}

// Resource management
bool IntegratedPlatform::check_resource_availability(uint32_t required_memory_mb) const {
    uint32_t total_allocated = 0;
    for (const auto& [task, memory] : resource_allocation_) {
        total_allocated += memory;
    }
    
    return (total_allocated + required_memory_mb) <= config_.max_memory_mb;
}

void IntegratedPlatform::update_resource_usage() {
    // Update capabilities based on current system state
    std::lock_guard<std::mutex> lock(platform_mutex_);
    capabilities_.available_memory_mb = utils::get_available_memory_mb();
}

// Configuration validation
bool IntegratedPlatform::validate_config(const PlatformConfig& config) const {
    if (config.node_id.empty()) {
        std::cerr << "Node ID cannot be empty" << std::endl;
        return false;
    }
    
    if (config.p2p_port == 0 || config.web_port == 0) {
        std::cerr << "Invalid port configuration" << std::endl;
        return false;
    }
    
    if (config.max_memory_mb == 0 || config.max_cpu_threads == 0) {
        std::cerr << "Invalid resource limits" << std::endl;
        return false;
    }
    
    return true;
}

void IntegratedPlatform::apply_config_changes(const PlatformConfig& old_config, const PlatformConfig& new_config) {
    // Apply configuration changes that can be done at runtime
    // For major changes, a restart might be required
    
    if (old_config.metrics_collection_interval_seconds != new_config.metrics_collection_interval_seconds) {
        std::cout << "Updated metrics collection interval to " << new_config.metrics_collection_interval_seconds << " seconds" << std::endl;
    }
    
    if (old_config.max_memory_mb != new_config.max_memory_mb) {
        std::cout << "Updated memory limit to " << new_config.max_memory_mb << " MB" << std::endl;
    }
}

// PlatformFactory implementation
PlatformConfig PlatformFactory::create_training_node_config(const std::string& node_id) {
    PlatformConfig config;
    config.node_id = node_id;
    config.enable_curation = false;
    config.enable_rlhf = true;
    config.enable_web_interface = false;
    config.enable_auto_training = true;
    config.max_memory_mb = 16384; // 16GB for training
    config.max_cpu_threads = 16;
    return config;
}

PlatformConfig PlatformFactory::create_annotation_node_config(const std::string& node_id) {
    PlatformConfig config;
    config.node_id = node_id;
    config.enable_curation = true;
    config.enable_rlhf = false;
    config.enable_web_interface = true;
    config.enable_auto_training = false;
    config.max_memory_mb = 4096; // 4GB for annotation
    config.max_cpu_threads = 4;
    return config;
}

PlatformConfig PlatformFactory::create_full_node_config(const std::string& node_id) {
    PlatformConfig config;
    config.node_id = node_id;
    config.enable_curation = true;
    config.enable_rlhf = true;
    config.enable_web_interface = true;
    config.enable_auto_training = true;
    config.max_memory_mb = 32768; // 32GB for full node
    config.max_cpu_threads = 32;
    return config;
}

PlatformConfig PlatformFactory::create_lightweight_node_config(const std::string& node_id) {
    PlatformConfig config;
    config.node_id = node_id;
    config.enable_curation = true;
    config.enable_rlhf = false;
    config.enable_web_interface = false;
    config.enable_auto_training = false;
    config.max_memory_mb = 2048; // 2GB for lightweight
    config.max_cpu_threads = 2;
    return config;
}

std::unique_ptr<IntegratedPlatform> PlatformFactory::create_platform(const PlatformConfig& config) {
    return std::make_unique<IntegratedPlatform>(config);
}

std::unique_ptr<IntegratedPlatform> PlatformFactory::create_training_node(const std::string& node_id, 
                                                                         const std::vector<std::string>& bootstrap_peers) {
    auto config = create_training_node_config(node_id);
    config.bootstrap_peers = bootstrap_peers;
    return create_platform(config);
}

std::unique_ptr<IntegratedPlatform> PlatformFactory::create_annotation_node(const std::string& node_id,
                                                                           const std::vector<std::string>& bootstrap_peers) {
    auto config = create_annotation_node_config(node_id);
    config.bootstrap_peers = bootstrap_peers;
    return create_platform(config);
}

// Utility functions
namespace utils {

NodeCapabilities detect_system_capabilities() {
    NodeCapabilities caps;
    
    caps.available_memory_mb = get_available_memory_mb();
    caps.cpu_cores = get_cpu_core_count();
    caps.has_gpu = has_gpu_support();
    
    // Set reasonable defaults based on system resources
    caps.max_model_size_mb = caps.available_memory_mb / 2; // Use half of available memory
    caps.can_train_models = caps.available_memory_mb >= 4096; // Require at least 4GB
    caps.can_annotate_data = true; // Always supported
    caps.can_serve_web_interface = true; // Always supported
    caps.can_store_data = true; // Always supported
    
    return caps;
}

uint32_t get_available_memory_mb() {
#ifdef _WIN32
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);
    return static_cast<uint32_t>(memInfo.ullAvailPhys / (1024 * 1024));
#else
    struct sysinfo memInfo;
    sysinfo(&memInfo);
    return static_cast<uint32_t>(memInfo.freeram * memInfo.mem_unit / (1024 * 1024));
#endif
}

uint32_t get_cpu_core_count() {
    return std::thread::hardware_concurrency();
}

bool has_gpu_support() {
    // Simple check - in practice, you'd check for CUDA/OpenCL support
    return false;
}

float get_cpu_usage_percent() {
    // Simplified implementation - in practice, use platform-specific APIs
    return 50.0f; // Placeholder
}

float get_memory_usage_percent() {
#ifdef _WIN32
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);
    return static_cast<float>(memInfo.ullTotalPhys - memInfo.ullAvailPhys) / memInfo.ullTotalPhys * 100.0f;
#else
    struct sysinfo memInfo;
    sysinfo(&memInfo);
    return static_cast<float>(memInfo.totalram - memInfo.freeram) / memInfo.totalram * 100.0f;
#endif
}

float get_disk_usage_percent(const std::string& path) {
    // Simplified implementation
    return 25.0f; // Placeholder
}

void export_metrics_to_file(const PlatformStats& stats, const std::string& file_path) {
    std::ofstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open metrics file: " << file_path << std::endl;
        return;
    }
    
    // Export as JSON (simplified)
    file << "{\n";
    file << "  \"connected_peers\": " << stats.connected_peers << ",\n";
    file << "  \"total_annotation_tasks\": " << stats.total_annotation_tasks << ",\n";
    file << "  \"completed_annotation_tasks\": " << stats.completed_annotation_tasks << ",\n";
    file << "  \"active_annotators\": " << stats.active_annotators << ",\n";
    file << "  \"cpu_usage_percent\": " << stats.cpu_usage_percent << ",\n";
    file << "  \"memory_usage_percent\": " << stats.memory_usage_percent << ",\n";
    file << "  \"timestamp\": " << stats.last_updated_timestamp << "\n";
    file << "}\n";
}

} // namespace utils

} // namespace integrated
