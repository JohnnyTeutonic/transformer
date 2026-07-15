#include "../include/advanced_integrated_platform.hpp"
#include "../include/heterogeneous_performance.hpp"
#include "../include/byzantine_detection.hpp"
#include "../include/incentive_alignment.hpp"
#include "../include/convergence_quality.hpp"
#include "../include/debugging_monitoring.hpp"
#include "../include/network_partitions.hpp"
#include "../include/logger.hpp"
#include <algorithm>
#include <thread>
#include <chrono>

namespace advanced_distributed {

AdvancedIntegratedPlatform::AdvancedIntegratedPlatform(const AdvancedPlatformConfig& config)
    : config_(config), is_running_(false), training_active_(false) {
    
    logger::log_info("Initializing Advanced Distributed Training Platform");
    logger::log_info("===============================================");
    
    // Core platform initialization
    logger::log_info("Core Platform Configuration:");
    logger::log_info("- Node ID: " + config_.node_id);
    logger::log_info("- Total expected nodes: " + std::to_string(config_.expected_total_nodes));
    logger::log_info("- Organization: " + config_.organization_name);
    logger::log_info("- Academic mode: " + (config_.is_academic ? "enabled" : "disabled"));
    
    // Initialize advanced distributed systems components
    initialize_advanced_components();
    
    // Set up integration callbacks
    setup_component_integration();
    
    logger::log_info("Advanced Distributed Training Platform initialized successfully");
}

AdvancedIntegratedPlatform::~AdvancedIntegratedPlatform() {
    stop_platform();
}

bool AdvancedIntegratedPlatform::initialize_advanced_components() {
    logger::log_info("Initializing advanced distributed systems components...");
    
    try {
        // 1. Heterogeneous Performance Manager
        heterogeneous::SynchronizationConfig sync_config;
        sync_config.wait_percentile = config_.elastic_sync_percentile;
        sync_config.min_nodes_required = config_.min_nodes_for_training;
        sync_config.max_wait_time_ms = 30000;
        sync_config.enable_adaptive_timeout = true;
        
        performance_manager_ = std::make_unique<heterogeneous::HeterogeneousPerformanceManager>(sync_config);
        logger::log_info("✓ Heterogeneous Performance Manager initialized");
        
        // 2. Byzantine Detection Engine
        byzantine::ByzantineDetectionConfig detection_config;
        detection_config.enable_gradient_clustering = true;
        detection_config.enable_cross_validation = true;
        detection_config.enable_magnitude_bounds = true;
        detection_config.outlier_threshold = 2.0f;
        detection_config.cross_val_frequency = 10;
        detection_config.quarantine_threshold = 0.3f;
        
        byzantine_detector_ = std::make_unique<byzantine::ByzantineDetectionEngine>(detection_config);
        logger::log_info("✓ Byzantine Detection Engine initialized");
        
        // 3. Incentive Alignment System
        incentive::IncentiveConfig incentive_config;
        if (config_.is_academic) {
            incentive_config.academic_reserved_share = 0.6f;
            incentive_config.contributor_reserved_share = 0.2f;
            incentive_config.academic_priority_boost = 2.0f;
        } else {
            incentive_config.academic_reserved_share = 0.3f;
            incentive_config.contributor_reserved_share = 0.4f;
            incentive_config.academic_priority_boost = 1.2f;
        }
        
        incentive_system_ = std::make_unique<incentive::IncentiveAlignmentSystem>(incentive_config);
        
        // Register our organization
        incentive_system_->register_contributor(config_.node_id, 
                                               config_.is_academic ? "university" : "commercial",
                                               config_.contact_info);
        
        logger::log_info("✓ Incentive Alignment System initialized");
        
        // 4. Convergence Quality Manager
        convergence::ConvergenceConfig convergence_config;
        convergence_config.enable_fedprox = true;
        convergence_config.enable_scaffold = true;
        convergence_config.max_staleness_steps = 10;
        convergence_config.enable_early_stopping = true;
        convergence_config.validation_frequency = 50;
        
        convergence_manager_ = std::make_unique<convergence::ConvergenceQualityManager>(convergence_config);
        logger::log_info("✓ Convergence Quality Manager initialized");
        
        // 5. Debugging and Monitoring System
        debugging::MonitoringConfig monitoring_config;
        monitoring_config.enable_distributed_tracing = true;
        monitoring_config.privacy_config.enable_data_sanitization = true;
        monitoring_config.trace_sampling_rate = 0.1f;
        monitoring_config.health_check_interval_ms = 30000;
        
        debugging_monitor_ = std::make_unique<debugging::DistributedDebuggingMonitor>(monitoring_config);
        
        // Register health checks
        register_health_checks();
        
        logger::log_info("✓ Debugging and Monitoring System initialized");
        
        // 6. Network Partition Handler
        network_partitions::PartitionHandlingConfig partition_config;
        partition_config.strategy = network_partitions::PartitionStrategy::DEGRADED_TRAINING;
        partition_config.enable_automatic_reconciliation = true;
        partition_config.min_nodes_for_training = config_.min_nodes_for_training;
        partition_config.max_acceptable_degradation = 0.7f;
        
        partition_handler_ = std::make_unique<network_partitions::NetworkPartitionHandler>(partition_config);
        logger::log_info("✓ Network Partition Handler initialized");
        
        return true;
        
    } catch (const std::exception& e) {
        logger::log_error("Failed to initialize advanced components: " + std::string(e.what()));
        return false;
    }
}

void AdvancedIntegratedPlatform::setup_component_integration() {
    logger::log_info("Setting up component integration...");
    
    // Set up Byzantine detection callbacks
    if (byzantine_detector_) {
        // Integration will notify when nodes are quarantined
    }
    
    // Set up partition detection callbacks
    if (partition_handler_) {
        partition_handler_->set_partition_detected_callback(
            [this](const network_partitions::PartitionInfo& partition) {
                handle_partition_detected(partition);
            });
            
        partition_handler_->set_partition_healed_callback(
            [this](const std::vector<std::string>& healed_nodes) {
                handle_partition_healed(healed_nodes);
            });
    }
    
    // Set up monitoring health checks
    register_health_checks();
}

void AdvancedIntegratedPlatform::register_health_checks() {
    if (!debugging_monitor_) return;
    
    // System health checks
    debugging_monitor_->register_health_check("memory_usage", 
        []() { return debugging::health_checks::check_memory_usage(0.8f, 0.9f); });
    
    debugging_monitor_->register_health_check("cpu_usage",
        []() { return debugging::health_checks::check_cpu_usage(0.8f, 0.9f); });
    
    debugging_monitor_->register_health_check("disk_space",
        []() { return debugging::health_checks::check_disk_space(0.8f, 0.9f); });
    
    // Training-specific health checks
    debugging_monitor_->register_health_check("gradient_computation",
        []() { return debugging::health_checks::check_gradient_computation_health(10000.0f); });
    
    debugging_monitor_->register_health_check("byzantine_detection_rate",
        []() { return debugging::health_checks::check_byzantine_detection_rate(0.1f); });
}

bool AdvancedIntegratedPlatform::start_platform() {
    if (is_running_.load()) {
        logger::log_warning("Platform is already running");
        return true;
    }
    
    logger::log_info("Starting Advanced Distributed Training Platform");
    
    try {
        // Start monitoring first
        if (debugging_monitor_) {
            debugging_monitor_->start_log_aggregation_server(9090);
            debugging_monitor_->start_tracing_server(14268);
            debugging_monitor_->enable_real_time_anomaly_detection(true);
            logger::log_info("✓ Monitoring services started");
        }
        
        // Start network partition monitoring
        if (partition_handler_) {
            // Register this node
            partition_handler_->register_node(config_.node_id, config_.ip_address, config_.port);
            partition_handler_->start_connectivity_monitoring();
            logger::log_info("✓ Network partition monitoring started");
        }
        
        // Start performance monitoring
        if (performance_manager_) {
            performance_manager_->schedule_periodic_benchmarks(true);
            logger::log_info("✓ Performance monitoring started");
        }
        
        is_running_ = true;
        
        // Start main platform loop
        platform_thread_ = std::thread(&AdvancedIntegratedPlatform::run_platform_loop, this);
        
        logger::log_info("Advanced Distributed Training Platform started successfully");
        return true;
        
    } catch (const std::exception& e) {
        logger::log_error("Failed to start platform: " + std::string(e.what()));
        return false;
    }
}

void AdvancedIntegratedPlatform::stop_platform() {
    if (!is_running_.load()) {
        return;
    }
    
    logger::log_info("Stopping Advanced Distributed Training Platform");
    
    is_running_ = false;
    training_active_ = false;
    
    if (platform_thread_.joinable()) {
        platform_thread_.join();
    }
    
    // Stop components
    if (partition_handler_) {
        partition_handler_->stop_connectivity_monitoring();
    }
    
    if (debugging_monitor_) {
        debugging_monitor_->stop_log_aggregation_server();
        debugging_monitor_->stop_tracing_server();
    }
    
    logger::log_info("Advanced Distributed Training Platform stopped");
}

bool AdvancedIntegratedPlatform::start_distributed_training(const TrainingConfig& training_config) {
    if (training_active_.load()) {
        logger::log_warning("Training is already active");
        return false;
    }
    
    logger::log_info("Starting distributed training");
    logger::log_info("- Model: " + training_config.model_name);
    logger::log_info("- Dataset: " + training_config.dataset_path);
    logger::log_info("- Expected nodes: " + std::to_string(training_config.expected_nodes));
    
    // Start distributed tracing for training
    std::string trace_id = "";
    if (debugging_monitor_) {
        trace_id = debugging_monitor_->start_trace("distributed_training", config_.node_id);
        debugging_monitor_->add_trace_tag(trace_id, "model_name", training_config.model_name);
        debugging_monitor_->add_trace_tag(trace_id, "dataset", training_config.dataset_path);
    }
    
    try {
        // Check network partition status
        if (partition_handler_) {
            auto training_decision = partition_handler_->make_training_decision(config_.node_id);
            
            if (!training_decision.should_continue_training) {
                logger::log_warning("Training blocked by partition handler: " + training_decision.reasoning);
                if (debugging_monitor_ && !trace_id.empty()) {
                    debugging_monitor_->finish_trace(trace_id, true, "Training blocked by network partition");
                }
                return false;
            }
            
            logger::log_info("Training approved with quality factor: " + 
                           std::to_string(training_decision.quality_degradation_factor));
        }
        
        // Initialize model parameters for convergence tracking
        if (convergence_manager_) {
            // Initialize with dummy parameters for demonstration
            std::vector<float> initial_params(1000000, 0.1f); // 1M parameters
            convergence_manager_->update_global_model(initial_params, 0);
        }
        
        // Initialize control variates for SCAFFOLD if we have participating nodes
        std::vector<std::string> participating_nodes = get_available_nodes();
        if (convergence_manager_ && !participating_nodes.empty()) {
            convergence_manager_->initialize_control_variates(participating_nodes, 1000000);
        }
        
        training_active_ = true;
        current_training_config_ = training_config;
        
        logger::log_info("Distributed training started successfully");
        
        if (debugging_monitor_ && !trace_id.empty()) {
            debugging_monitor_->log_trace_event(trace_id, "training_started", 
                                               {{"nodes", std::to_string(participating_nodes.size())}});
        }
        
        return true;
        
    } catch (const std::exception& e) {
        logger::log_error("Failed to start distributed training: " + std::string(e.what()));
        
        if (debugging_monitor_ && !trace_id.empty()) {
            debugging_monitor_->finish_trace(trace_id, true, "Training start failed: " + std::string(e.what()));
        }
        
        return false;
    }
}

void AdvancedIntegratedPlatform::stop_distributed_training() {
    if (!training_active_.load()) {
        logger::log_warning("Training is not active");
        return;
    }
    
    logger::log_info("Stopping distributed training");
    
    training_active_ = false;
    
    // Export training report
    if (convergence_manager_) {
        std::string report_path = "./training_report_" + 
                                 std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) + 
                                 ".txt";
        convergence_manager_->export_convergence_report(report_path);
    }
    
    logger::log_info("Distributed training stopped");
}

TrainingStepResult AdvancedIntegratedPlatform::execute_training_step(const std::vector<float>& input_data,
                                                                    const std::vector<float>& labels) {
    if (!training_active_.load()) {
        TrainingStepResult error_result;
        error_result.success = false;
        error_result.error_message = "Training not active";
        return error_result;
    }
    
    // Start trace for this training step
    std::string trace_id = "";
    if (debugging_monitor_) {
        trace_id = debugging_monitor_->start_trace("training_step", config_.node_id);
    }
    
    TrainingStepResult result;
    result.step_number = current_training_step_++;
    result.timestamp = std::chrono::steady_clock::now();
    
    try {
        // 1. Check if we should participate in this training step
        auto available_nodes = get_available_nodes();
        
        // 2. Simulate gradient computation (in real implementation, this would be actual training)
        std::vector<float> gradients = simulate_gradient_computation(input_data, labels);
        
        // 3. Create gradient fingerprint for Byzantine detection
        byzantine::GradientFingerprint gradient_fingerprint;
        gradient_fingerprint.node_id = config_.node_id;
        gradient_fingerprint.gradients = gradients;
        gradient_fingerprint.l2_norm = calculate_l2_norm(gradients);
        gradient_fingerprint.step_number = result.step_number;
        gradient_fingerprint.local_steps = 1;
        gradient_fingerprint.learning_rate = current_training_config_.learning_rate;
        gradient_fingerprint.timestamp = result.timestamp;
        
        // Sample gradients for clustering analysis
        gradient_fingerprint.gradient_sample = sample_gradients(gradients, 1000);
        
        // 4. Byzantine detection check
        if (byzantine_detector_) {
            std::vector<byzantine::GradientFingerprint> gradients_to_check = {gradient_fingerprint};
            auto detection_result = byzantine_detector_->run_full_detection_pipeline(gradients_to_check);
            
            // Check if this node is flagged as Byzantine
            bool is_byzantine = std::find(detection_result.byzantine_nodes.begin(), 
                                        detection_result.byzantine_nodes.end(), 
                                        config_.node_id) != detection_result.byzantine_nodes.end();
            
            if (is_byzantine) {
                result.success = false;
                result.error_message = "Node flagged as Byzantine";
                logger::log_warning("Training step blocked: Node flagged as Byzantine");
                
                if (debugging_monitor_ && !trace_id.empty()) {
                    debugging_monitor_->finish_trace(trace_id, true, "Byzantine detection failed");
                }
                
                return result;
            }
        }
        
        // 5. Process gradients with advanced algorithms
        std::vector<float> processed_gradients;
        if (convergence_manager_) {
            // Use SCAFFOLD or FedProx based on configuration
            convergence::GradientState gradient_state;
            gradient_state.node_id = config_.node_id;
            gradient_state.gradients = gradients;
            gradient_state.step_number = result.step_number;
            gradient_state.local_steps = 1;
            gradient_state.learning_rate = current_training_config_.learning_rate;
            gradient_state.timestamp = result.timestamp;
            
            std::vector<convergence::GradientState> gradient_states = {gradient_state};
            std::vector<float> global_model = convergence_manager_->get_global_model();
            
            processed_gradients = convergence_manager_->process_gradients_scaffold(gradient_states, global_model);
        } else {
            processed_gradients = gradients;
        }
        
        // 6. Simulate model update
        result.loss_value = simulate_loss_computation();
        result.gradient_norm = calculate_l2_norm(processed_gradients);
        result.processing_time_ms = 100.0f; // Simulated
        
        // 7. Update convergence tracking
        if (convergence_manager_) {
            std::vector<float> loss_history = {result.loss_value};
            auto convergence_metrics = convergence_manager_->analyze_convergence(loss_history, processed_gradients);
            result.convergence_score = convergence_metrics.convergence_score;
            result.is_converged = convergence_metrics.is_converged;
        }
        
        // 8. Log training metrics
        if (debugging_monitor_) {
            debugging_monitor_->submit_custom_metric(config_.node_id, "loss_value", result.loss_value);
            debugging_monitor_->submit_custom_metric(config_.node_id, "gradient_norm", result.gradient_norm);
            debugging_monitor_->submit_custom_metric(config_.node_id, "convergence_score", result.convergence_score);
        }
        
        result.success = true;
        
        if (debugging_monitor_ && !trace_id.empty()) {
            debugging_monitor_->log_trace_event(trace_id, "step_completed",
                                               {{"loss", std::to_string(result.loss_value)},
                                                {"gradient_norm", std::to_string(result.gradient_norm)}});
            debugging_monitor_->finish_trace(trace_id);
        }
        
    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
        
        logger::log_error("Training step failed: " + std::string(e.what()));
        
        if (debugging_monitor_ && !trace_id.empty()) {
            debugging_monitor_->finish_trace(trace_id, true, "Training step failed: " + std::string(e.what()));
        }
    }
    
    return result;
}

PlatformStatus AdvancedIntegratedPlatform::get_platform_status() const {
    PlatformStatus status;
    status.is_running = is_running_.load();
    status.training_active = training_active_.load();
    status.current_step = current_training_step_.load();
    
    // Get network status
    if (partition_handler_) {
        auto network_status = partition_handler_->get_network_status();
        status.total_nodes = network_status.total_nodes;
        status.active_nodes = network_status.reachable_nodes;
        status.is_partitioned = network_status.is_partitioned;
    }
    
    // Get health status
    if (debugging_monitor_) {
        status.system_healthy = debugging_monitor_->is_system_healthy();
        auto aggregated_metrics = debugging_monitor_->get_aggregated_metrics();
        status.average_cpu_usage = aggregated_metrics.average_cpu_usage;
        status.average_memory_usage = aggregated_metrics.average_memory_usage;
    }
    
    // Get convergence status
    if (convergence_manager_) {
        auto convergence_stats = convergence_manager_->get_convergence_stats();
        status.convergence_quality = static_cast<float>(convergence_stats.total_global_steps) / 1000.0f; // Normalized
    }
    
    status.last_updated = std::chrono::steady_clock::now();
    
    return status;
}

// Event handlers
void AdvancedIntegratedPlatform::handle_partition_detected(const network_partitions::PartitionInfo& partition) {
    logger::log_warning("Network partition detected! Partition ID: " + partition.partition_id +
                        ", Nodes: " + std::to_string(partition.partition_size));
    
    // Log to monitoring system
    if (debugging_monitor_) {
        debugging_monitor_->log_warning(config_.node_id, "network_partition", 
                                       "Network partition detected with " + 
                                       std::to_string(partition.partition_size) + " nodes");
    }
    
    // Create checkpoint if training is active
    if (training_active_.load() && partition_handler_) {
        partition_handler_->force_checkpoint_creation("partition_detected");
    }
}

void AdvancedIntegratedPlatform::handle_partition_healed(const std::vector<std::string>& healed_nodes) {
    logger::log_info("Network partition healed! Reconnected nodes: " + std::to_string(healed_nodes.size()));
    
    if (debugging_monitor_) {
        debugging_monitor_->log_info(config_.node_id, "network_partition", 
                                    "Network partition healed, " + std::to_string(healed_nodes.size()) + 
                                    " nodes reconnected");
    }
}

void AdvancedIntegratedPlatform::run_platform_loop() {
    logger::log_info("Started platform main loop");
    
    while (is_running_.load()) {
        try {
            // Update performance metrics
            if (performance_manager_) {
                // Collect current system performance
                heterogeneous::NodeCapabilities capabilities;
                capabilities.node_id = config_.node_id;
                capabilities.compute_score = 1.0f; // Would be measured dynamically
                capabilities.memory_gb = 8.0f; // Example
                capabilities.has_gpu = false;
                capabilities.last_benchmark = std::chrono::steady_clock::now();
                
                performance_manager_->update_node_capabilities(config_.node_id, capabilities);
            }
            
            // Update contribution tracking
            if (incentive_system_ && training_active_.load()) {
                // Simulate contribution
                incentive_system_->update_contribution(config_.node_id, 0.1f, 0.0f, 1);
            }
            
            // Collect system metrics
            if (debugging_monitor_) {
                debugging_monitor_->collect_system_metrics(config_.node_id);
            }
            
            // Sleep for main loop interval
            std::this_thread::sleep_for(std::chrono::seconds(10));
            
        } catch (const std::exception& e) {
            logger::log_error("Error in platform main loop: " + std::string(e.what()));
        }
    }
    
    logger::log_info("Platform main loop stopped");
}

// Helper methods
std::vector<std::string> AdvancedIntegratedPlatform::get_available_nodes() const {
    if (partition_handler_) {
        return partition_handler_->get_registered_nodes();
    }
    return {config_.node_id}; // At minimum, we have ourselves
}

std::vector<float> AdvancedIntegratedPlatform::simulate_gradient_computation(const std::vector<float>& input_data,
                                                                           const std::vector<float>& labels) {
    // Simulate gradient computation
    std::vector<float> gradients(1000000); // 1M parameters
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0f, 0.01f);
    
    for (auto& grad : gradients) {
        grad = dis(gen);
    }
    
    return gradients;
}

float AdvancedIntegratedPlatform::simulate_loss_computation() {
    // Simulate decreasing loss over time
    static float base_loss = 2.0f;
    static uint64_t steps = 0;
    
    steps++;
    base_loss *= 0.9999f; // Gradual decrease
    
    // Add some noise
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> noise(0.0f, 0.01f);
    
    return base_loss + noise(gen);
}

float AdvancedIntegratedPlatform::calculate_l2_norm(const std::vector<float>& vector) {
    float sum = 0.0f;
    for (float value : vector) {
        sum += value * value;
    }
    return std::sqrt(sum);
}

std::vector<float> AdvancedIntegratedPlatform::sample_gradients(const std::vector<float>& gradients, size_t sample_size) {
    if (gradients.size() <= sample_size) {
        return gradients;
    }
    
    std::vector<float> sampled(sample_size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dis(0, gradients.size() - 1);
    
    for (size_t i = 0; i < sample_size; ++i) {
        sampled[i] = gradients[dis(gen)];
    }
    
    return sampled;
}

} // namespace advanced_distributed
