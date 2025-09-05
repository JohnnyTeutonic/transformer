#pragma once

#include "distributed_transformer.hpp"
#include "distributed_curation.hpp"
#include "distributed_rlhf.hpp"
#include "web_annotation_interface.hpp"
#include "p2p_network.hpp"
#include "p2p_message_types.hpp"
#include <memory>
#include <atomic>
#include <thread>
#include <vector>
#include <map>
#include <functional>

namespace integrated {

// Platform configuration
struct PlatformConfig {
    // Network configuration
    std::string node_id;
    std::string bind_address = "0.0.0.0";
    uint16_t p2p_port = 7777;
    uint16_t web_port = 8080;
    std::vector<std::string> bootstrap_peers;
    
    // Model configuration
    std::string model_config_path;
    std::string tokenizer_path;
    std::string checkpoint_path;
    
    // Curation configuration
    curation::CurationConfig curation_config;
    
    // RLHF configuration
    rlhf::RLHFConfig rlhf_config;
    
    // Web interface configuration
    web_interface::WebServerConfig web_config;
    
    // Platform settings
    bool enable_curation = true;
    bool enable_rlhf = true;
    bool enable_web_interface = true;
    bool enable_auto_training = false;
    uint32_t auto_training_interval_hours = 24;
    
    // Resource limits
    uint32_t max_concurrent_tasks = 100;
    uint32_t max_memory_mb = 8192;
    uint32_t max_cpu_threads = 8;
    
    // Monitoring
    bool enable_metrics_collection = true;
    uint32_t metrics_collection_interval_seconds = 60;
    std::string metrics_output_path = "./metrics";
};

// Node capabilities
struct NodeCapabilities {
    bool can_train_models = true;
    bool can_annotate_data = true;
    bool can_serve_web_interface = true;
    bool can_store_data = true;
    uint32_t max_model_size_mb = 4096;
    uint32_t available_memory_mb = 8192;
    uint32_t cpu_cores = 8;
    bool has_gpu = false;
    uint32_t gpu_memory_mb = 0;
    std::vector<std::string> supported_model_types = {"transformer", "reward_model"};
};

// Platform statistics
struct PlatformStats {
    // Network stats
    uint32_t connected_peers = 0;
    uint32_t total_messages_sent = 0;
    uint32_t total_messages_received = 0;
    float network_latency_ms = 0.0f;
    
    // Training stats
    uint32_t training_iterations_completed = 0;
    float current_loss = 0.0f;
    uint32_t total_parameters = 0;
    float training_throughput_tokens_per_second = 0.0f;
    
    // Curation stats
    uint32_t total_annotation_tasks = 0;
    uint32_t completed_annotation_tasks = 0;
    uint32_t active_annotators = 0;
    float average_annotation_quality = 0.0f;
    
    // RLHF stats
    uint32_t rlhf_iterations_completed = 0;
    float reward_model_accuracy = 0.0f;
    float policy_improvement = 0.0f;
    uint32_t preference_data_samples = 0;
    
    // Web interface stats
    uint32_t active_web_sessions = 0;
    uint32_t total_web_requests = 0;
    uint32_t web_annotations_submitted = 0;
    
    // Resource usage
    float cpu_usage_percent = 0.0f;
    float memory_usage_percent = 0.0f;
    float disk_usage_percent = 0.0f;
    
    uint64_t last_updated_timestamp = 0;
};

// Event types for the platform
enum class PlatformEvent {
    NODE_JOINED,
    NODE_LEFT,
    TRAINING_STARTED,
    TRAINING_COMPLETED,
    TRAINING_FAILED,
    ANNOTATION_TASK_CREATED,
    ANNOTATION_TASK_COMPLETED,
    RLHF_ITERATION_COMPLETED,
    CONSENSUS_REACHED,
    CONSENSUS_FAILED,
    WEB_SESSION_CREATED,
    WEB_SESSION_ENDED,
    ERROR_OCCURRED,
    PERFORMANCE_THRESHOLD_EXCEEDED
};

// Main integrated platform class
class IntegratedPlatform {
public:
    IntegratedPlatform(const PlatformConfig& config);
    ~IntegratedPlatform();

    // Platform lifecycle
    bool initialize();
    bool start();
    void stop();
    bool is_running() const { return running_.load(); }

    // Component access
    std::shared_ptr<p2p::P2PNetwork> get_network() const { return network_; }
    std::shared_ptr<DistributedTransformer> get_model() const { return model_; }
    std::shared_ptr<curation::DistributedCurationPlatform> get_curation() const { return curation_platform_; }
    std::shared_ptr<rlhf::DistributedRLHFCoordinator> get_rlhf() const { return rlhf_coordinator_; }
    std::shared_ptr<web_interface::WebAnnotationInterface> get_web_interface() const { return web_interface_; }

    // Platform operations
    bool submit_annotation_task(const curation::AnnotationTask& task);
    bool start_training_session(const std::vector<std::string>& training_data);
    bool start_rlhf_training(uint32_t reward_epochs, uint32_t ppo_iterations);
    bool save_model_checkpoint(const std::string& path);
    bool load_model_checkpoint(const std::string& path);

    // Monitoring and statistics
    PlatformStats get_stats() const;
    NodeCapabilities get_capabilities() const;
    std::vector<std::string> get_connected_peers() const;
    
    // Configuration management
    bool update_config(const PlatformConfig& new_config);
    PlatformConfig get_config() const;

    // Event handling
    using EventCallback = std::function<void(PlatformEvent, const std::string&)>;
    void set_event_callback(EventCallback callback);

    // Resource management
    bool allocate_resources(const std::string& task_type, uint32_t required_memory_mb);
    void release_resources(const std::string& task_type);
    std::map<std::string, uint32_t> get_resource_allocation() const;

    // Health monitoring
    struct HealthStatus {
        bool is_healthy = true;
        std::vector<std::string> issues;
        float overall_score = 1.0f; // 0.0 to 1.0
        uint64_t last_check_timestamp = 0;
    };
    HealthStatus check_health() const;

private:
    PlatformConfig config_;
    std::atomic<bool> running_{false};
    std::atomic<bool> initialized_{false};

    // Core components
    std::shared_ptr<p2p::P2PNetwork> network_;
    std::shared_ptr<DistributedTransformer> model_;
    std::shared_ptr<curation::DistributedCurationPlatform> curation_platform_;
    std::shared_ptr<rlhf::DistributedRLHFCoordinator> rlhf_coordinator_;
    std::shared_ptr<web_interface::WebAnnotationInterface> web_interface_;

    // Background threads
    std::vector<std::thread> worker_threads_;
    void metrics_collection_thread();
    void health_monitoring_thread();
    void auto_training_thread();
    void resource_management_thread();

    // State management
    mutable std::mutex platform_mutex_;
    PlatformStats current_stats_;
    NodeCapabilities capabilities_;
    std::map<std::string, uint32_t> resource_allocation_;
    HealthStatus current_health_;

    // Event handling
    EventCallback event_callback_;
    void emit_event(PlatformEvent event, const std::string& details = "");

    // Component coordination
    void setup_component_callbacks();
    void handle_curation_task_completed(const curation::AnnotationConsensus& consensus);
    void handle_rlhf_iteration_completed(const rlhf::PPOMetrics& metrics);
    void handle_network_peer_joined(const std::string& peer_id);
    void handle_network_peer_left(const std::string& peer_id);

    // Resource management
    bool check_resource_availability(uint32_t required_memory_mb) const;
    void update_resource_usage();

    // Health checks
    bool check_network_health() const;
    bool check_model_health() const;
    bool check_curation_health() const;
    bool check_rlhf_health() const;
    bool check_web_interface_health() const;
    bool check_system_resources() const;

    // Metrics collection
    void collect_network_metrics();
    void collect_training_metrics();
    void collect_curation_metrics();
    void collect_rlhf_metrics();
    void collect_web_metrics();
    void collect_system_metrics();

    // Configuration validation
    bool validate_config(const PlatformConfig& config) const;
    void apply_config_changes(const PlatformConfig& old_config, const PlatformConfig& new_config);
};

// Platform factory for easy setup
class PlatformFactory {
public:
    // Predefined configurations
    static PlatformConfig create_training_node_config(const std::string& node_id);
    static PlatformConfig create_annotation_node_config(const std::string& node_id);
    static PlatformConfig create_full_node_config(const std::string& node_id);
    static PlatformConfig create_lightweight_node_config(const std::string& node_id);

    // Configuration builders
    static PlatformConfig& with_p2p_port(PlatformConfig& config, uint16_t port);
    static PlatformConfig& with_web_port(PlatformConfig& config, uint16_t port);
    static PlatformConfig& with_bootstrap_peers(PlatformConfig& config, const std::vector<std::string>& peers);
    static PlatformConfig& with_model_path(PlatformConfig& config, const std::string& path);
    static PlatformConfig& with_resource_limits(PlatformConfig& config, uint32_t memory_mb, uint32_t cpu_threads);

    // Platform creation
    static std::unique_ptr<IntegratedPlatform> create_platform(const PlatformConfig& config);
    static std::unique_ptr<IntegratedPlatform> create_training_node(const std::string& node_id, 
                                                                   const std::vector<std::string>& bootstrap_peers = {});
    static std::unique_ptr<IntegratedPlatform> create_annotation_node(const std::string& node_id,
                                                                     const std::vector<std::string>& bootstrap_peers = {});
};

// Utility functions for platform management
namespace utils {
    // Configuration helpers
    bool save_config(const PlatformConfig& config, const std::string& file_path);
    std::optional<PlatformConfig> load_config(const std::string& file_path);
    
    // Network discovery
    std::vector<std::string> discover_local_peers(uint16_t port_range_start = 7777, uint16_t port_range_end = 7787);
    bool is_port_available(uint16_t port);
    
    // System information
    NodeCapabilities detect_system_capabilities();
    uint32_t get_available_memory_mb();
    uint32_t get_cpu_core_count();
    bool has_gpu_support();
    
    // Monitoring helpers
    void export_metrics_to_file(const PlatformStats& stats, const std::string& file_path);
    void export_metrics_to_prometheus(const PlatformStats& stats, const std::string& endpoint);
    
    // Health check utilities
    bool ping_peer(const std::string& peer_address, uint16_t port, uint32_t timeout_ms = 5000);
    float calculate_network_latency(const std::vector<std::string>& peers);
    
    // Resource monitoring
    float get_cpu_usage_percent();
    float get_memory_usage_percent();
    float get_disk_usage_percent(const std::string& path = "./");
}

} // namespace integrated
