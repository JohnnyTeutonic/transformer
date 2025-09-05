#include "../../include/training/training_state_manager.hpp"
#include "../../include/adaptive_batch_scheduler.hpp"

void TrainingStateManager::update_state(const TrainingMetrics& metrics) {
    // Update loss tracking
    loss_tracker->add_loss(metrics.loss);
    
    // Process gradients
    Matrix gradients_copy = metrics.gradients;  // Make a copy
    gradient_manager->process_gradients(gradients_copy);
    
    // Update learning rate if we have enough samples
    if (loss_tracker->should_adjust_lr()) {
        TrainingMetrics updated_metrics(
            metrics.loss,
            metrics.gradients,
            metrics.epoch,
            metrics.step,
            loss_tracker->get_trend(),
            metrics.grad_stats
        );
        lr_scheduler->get_learning_rate(updated_metrics);
    }
    
    // Handle instability if detected
    if (detect_instability()) {
        recover_from_instability();
    }
}

bool TrainingStateManager::detect_instability() const {
    return loss_tracker->get_trend() > INSTABILITY_THRESHOLD ||
           gradient_manager->explosion_detected();
}

void TrainingStateManager::recover_from_instability() {
    // Scale down learning rate significantly
    RunningStatistics stats = gradient_manager->get_statistics();
    
    // Create recovery metrics using constructor
    TrainingMetrics recovery_metrics(
        0.0f,                  // loss
        Matrix(0, 0),         // gradients
        0,                    // epoch
        0,                    // step
        2.0f,                // loss_trend
        stats                // grad_stats
    );
    
    // Get a reduced learning rate
    lr_scheduler->get_learning_rate(recovery_metrics);
}

// Adaptive batch scheduling methods
size_t TrainingStateManager::get_optimal_batch_size(const TransformerConfig& config) {
    if (!batch_scheduler_) {
        batch_scheduler_ = std::make_unique<AdaptiveBatchScheduler>(
            config.min_batch_size,
            config.max_batch_size
        );
    }
    
    return batch_scheduler_->calculate_optimal_batch_size(config);
}

void TrainingStateManager::update_batch_performance(const BatchPerformanceMetrics& metrics) {
    if (!batch_scheduler_) return;
    
    // Convert to AdaptiveBatchScheduler format
    AdaptiveBatchScheduler::PerformanceMetrics perf_metrics;
    perf_metrics.batch_size = metrics.batch_size;
    perf_metrics.samples_per_second = metrics.samples_per_second;
    perf_metrics.memory_utilization = metrics.memory_utilization;
    perf_metrics.gpu_utilization = metrics.gpu_utilization;
    perf_metrics.network_latency_ms = metrics.network_latency_ms;
    perf_metrics.loss_value = metrics.loss_value;
    
    batch_scheduler_->update_performance_metrics(perf_metrics);
}

void TrainingStateManager::print_batch_statistics() const {
    if (batch_scheduler_) {
        batch_scheduler_->print_statistics();
    }
} 