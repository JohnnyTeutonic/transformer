#pragma once

#include <memory>
#include <atomic>
#include <thread>
#include <vector>
#include <string>
#include <chrono>

// Forward declarations to avoid circular includes
namespace heterogeneous { class HeterogeneousPerformanceManager; }
namespace byzantine { class ByzantineDetectionEngine; }
namespace incentive { class IncentiveAlignmentSystem; }
namespace convergence { class ConvergenceQualityManager; }
namespace debugging { class DistributedDebuggingMonitor; }
namespace network_partitions { class NetworkPartitionHandler; struct PartitionInfo; }

namespace advanced_distributed {

struct AdvancedPlatformConfig {
    // Basic node configuration
    std::string node_id;
    std::string ip_address = "127.0.0.1";
    uint16_t port = 8080;
    
    // Organization information
    std::string organization_name;
    std::string contact_info;
    bool is_academic = false;
    
    // Network configuration
    uint32_t expected_total_nodes = 10;
    uint32_t min_nodes_for_training = 3;
    float elastic_sync_percentile = 0.8f;  // Wait for fastest 80% of nodes
    
    // Training configuration
    std::string model_name = "distributed_transformer";
    float default_learning_rate = 0.001f;
    uint32_t checkpoint_interval = 1000;
    
    // Advanced features
    bool enable_byzantine_detection = true;
    bool enable_federated_learning = true;
    bool enable_incentive_system = true;
    bool enable_network_partition_handling = true;
    bool enable_privacy_preservation = true;
};

struct TrainingConfig {
    std::string model_name;
    std::string dataset_path;
    uint32_t expected_nodes;
    float learning_rate = 0.001f;
    uint32_t max_epochs = 100;
    uint32_t batch_size = 32;
    bool enable_validation = true;
    float validation_split = 0.1f;
};

struct TrainingStepResult {
    uint64_t step_number;
    bool success = false;
    float loss_value = 0.0f;
    float gradient_norm = 0.0f;
    float convergence_score = 0.0f;
    bool is_converged = false;
    float processing_time_ms = 0.0f;
    std::string error_message;
    std::chrono::steady_clock::time_point timestamp;
};

struct PlatformStatus {
    bool is_running = false;
    bool training_active = false;
    uint64_t current_step = 0;
    
    // Network status
    uint32_t total_nodes = 0;
    uint32_t active_nodes = 0;
    bool is_partitioned = false;
    
    // Health status
    bool system_healthy = true;
    float average_cpu_usage = 0.0f;
    float average_memory_usage = 0.0f;
    
    // Training status
    float convergence_quality = 0.0f;
    
    std::chrono::steady_clock::time_point last_updated;
};

class AdvancedIntegratedPlatform {
public:
    explicit AdvancedIntegratedPlatform(const AdvancedPlatformConfig& config);
    ~AdvancedIntegratedPlatform();
    
    // Platform lifecycle
    bool start_platform();
    void stop_platform();
    bool is_running() const { return is_running_.load(); }
    
    // Training coordination
    bool start_distributed_training(const TrainingConfig& training_config);
    void stop_distributed_training();
    bool is_training_active() const { return training_active_.load(); }
    
    // Training execution
    TrainingStepResult execute_training_step(const std::vector<float>& input_data,
                                           const std::vector<float>& labels);
    
    // Status and monitoring
    PlatformStatus get_platform_status() const;
    
    // Component access (for advanced usage)
    heterogeneous::HeterogeneousPerformanceManager* get_performance_manager() const {
        return performance_manager_.get();
    }
    
    byzantine::ByzantineDetectionEngine* get_byzantine_detector() const {
        return byzantine_detector_.get();
    }
    
    incentive::IncentiveAlignmentSystem* get_incentive_system() const {
        return incentive_system_.get();
    }
    
    convergence::ConvergenceQualityManager* get_convergence_manager() const {
        return convergence_manager_.get();
    }
    
    debugging::DistributedDebuggingMonitor* get_debugging_monitor() const {
        return debugging_monitor_.get();
    }
    
    network_partitions::NetworkPartitionHandler* get_partition_handler() const {
        return partition_handler_.get();
    }

private:
    // Configuration
    AdvancedPlatformConfig config_;
    TrainingConfig current_training_config_;
    
    // State
    std::atomic<bool> is_running_{false};
    std::atomic<bool> training_active_{false};
    std::atomic<uint64_t> current_training_step_{0};
    
    // Main platform thread
    std::thread platform_thread_;
    
    // Advanced distributed systems components
    std::unique_ptr<heterogeneous::HeterogeneousPerformanceManager> performance_manager_;
    std::unique_ptr<byzantine::ByzantineDetectionEngine> byzantine_detector_;
    std::unique_ptr<incentive::IncentiveAlignmentSystem> incentive_system_;
    std::unique_ptr<convergence::ConvergenceQualityManager> convergence_manager_;
    std::unique_ptr<debugging::DistributedDebuggingMonitor> debugging_monitor_;
    std::unique_ptr<network_partitions::NetworkPartitionHandler> partition_handler_;
    
    // Initialization methods
    bool initialize_advanced_components();
    void setup_component_integration();
    void register_health_checks();
    
    // Event handlers
    void handle_partition_detected(const network_partitions::PartitionInfo& partition);
    void handle_partition_healed(const std::vector<std::string>& healed_nodes);
    
    // Platform main loop
    void run_platform_loop();
    
    // Helper methods
    std::vector<std::string> get_available_nodes() const;
    std::vector<float> simulate_gradient_computation(const std::vector<float>& input_data,
                                                   const std::vector<float>& labels);
    float simulate_loss_computation();
    float calculate_l2_norm(const std::vector<float>& vector);
    std::vector<float> sample_gradients(const std::vector<float>& gradients, size_t sample_size);
};

// Utility functions for platform setup
namespace utils {

AdvancedPlatformConfig create_academic_config(const std::string& node_id, 
                                             const std::string& university_name);

AdvancedPlatformConfig create_commercial_config(const std::string& node_id,
                                               const std::string& company_name);

TrainingConfig create_default_training_config(const std::string& model_name,
                                             const std::string& dataset_path);

} // namespace utils
} // namespace advanced_distributed
