#pragma once
#include "loss_tracker.hpp"
#include "gradient_manager.hpp"
#include "learning_rate_scheduler.hpp"
#include "training_metrics.hpp"
#include "../adaptive_batch_scheduler.hpp"
#include "../config.hpp"
#include <memory>

// Batch performance metrics for adaptive scheduling
struct BatchPerformanceMetrics {
    size_t batch_size;
    float samples_per_second;
    float memory_utilization;
    float gpu_utilization;
    float network_latency_ms;
    float loss_value;
};

class TrainingStateManager {
public:
    static constexpr float INSTABILITY_THRESHOLD = 2.0f;

    TrainingStateManager(float initial_lr = 0.001f)
        : loss_tracker(std::make_unique<LossTracker>()),
          gradient_manager(std::make_unique<GradientManager>()),
          lr_scheduler(std::make_unique<LearningRateScheduler>(initial_lr)) {}

    void update_state(const TrainingMetrics& metrics);
    float get_learning_rate() const { return lr_scheduler->get_current_lr(); }
    bool is_stable() const { return !detect_instability(); }

    // Adaptive batch scheduling methods
    size_t get_optimal_batch_size(const TransformerConfig& config);
    void update_batch_performance(const BatchPerformanceMetrics& metrics);
    void print_batch_statistics() const;

    // Access to gradient manager for P2P compression
    GradientManager* get_gradient_manager() { return gradient_manager.get(); }

private:
    std::unique_ptr<LossTracker> loss_tracker;
    std::unique_ptr<GradientManager> gradient_manager;
    std::unique_ptr<LearningRateScheduler> lr_scheduler;
    std::unique_ptr<AdaptiveBatchScheduler> batch_scheduler_;

    bool detect_instability() const;
    void recover_from_instability();
}; 