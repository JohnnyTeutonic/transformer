#include "../../include/training/training_monitor.hpp"

void TrainingMonitor::log_metrics(const TrainingMetrics& metrics) {
    update_running_statistics(metrics);
    detect_anomalies(metrics);
    
    // Calculate gradient norm
    float grad_norm = 0.0f;
    if (metrics.gradients.size() > 0) {
        grad_norm = 0.0f;
        for (size_t i = 0; i < metrics.gradients.rows(); ++i) {
            for (size_t j = 0; j < metrics.gradients.cols(); ++j) {
                grad_norm += metrics.gradients.at(i, j) * metrics.gradients.at(i, j);
            }
        }
        grad_norm = std::sqrt(grad_norm);
    }
    
    // Update loss tracker and visualizer
    loss_tracker.add_loss(metrics.loss);
    visualizer->add_loss(
        metrics.loss,               // Raw loss
        loss_tracker.get_recent_average(),  // Smoothed loss
        loss_tracker.get_trend(),   // Trend
        grad_norm,                  // Gradient norm
        metrics.learning_rate       // Add current learning rate
    );
    
    log_to_tensorboard(metrics);
}

bool TrainingMonitor::should_stop_training() {
    return detect_divergence() || 
           reached_convergence() || 
           exceeded_max_epochs();
}

bool TrainingMonitor::detect_divergence() {
    return loss_tracker.get_trend() > DIVERGENCE_THRESHOLD ||
           gradient_manager.explosion_detected() ||
           nan_counter > MAX_NAN_OCCURRENCES;
}

bool TrainingMonitor::reached_convergence() {
    // Check if loss has plateaued
    if (loss_tracker.get_sample_count() < LossTracker::MIN_SAMPLES) {
        return false;
    }
    
    float trend = loss_tracker.get_trend();
    return trend > 0.99f && trend < 1.01f;
}

bool TrainingMonitor::exceeded_max_epochs() {
    return current_epoch >= MAX_EPOCHS;
}

void TrainingMonitor::update_running_statistics(const TrainingMetrics& metrics) {
    if (!std::isfinite(metrics.loss)) {
        nan_counter++;
    } else {
        nan_counter = std::max(0, static_cast<int>(nan_counter) - 1);
    }
    current_epoch = metrics.epoch;
}

void TrainingMonitor::detect_anomalies(const TrainingMetrics& metrics) {
    // Check for unusual patterns in loss or gradients
    if (metrics.loss_trend > DIVERGENCE_THRESHOLD) {
        std::cout << "Warning: Unusual loss trend detected: " << metrics.loss_trend << std::endl;
    }
    
    if (gradient_manager.explosion_detected()) {
        std::cout << "Warning: Gradient explosion detected" << std::endl;
    }
}

void TrainingMonitor::log_to_tensorboard(const TrainingMetrics& metrics) {
    // TODO: Implement actual tensorboard logging
    std::cout << "Epoch " << metrics.epoch 
              << ", Step " << metrics.step 
              << ", Loss " << metrics.loss 
              << ", LR " << metrics.loss_trend << std::endl;
} 