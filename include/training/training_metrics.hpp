#pragma once
#include "../matrix.hpp"
#include "gradient_manager.hpp"

struct TrainingMetrics {
    float loss;
    Matrix gradients;
    size_t epoch;
    size_t step;
    float loss_trend;
    const RunningStatistics& grad_stats;
    float learning_rate;

    // Constructor
    TrainingMetrics(
        float loss_,
        const Matrix& gradients_,
        size_t epoch_,
        size_t step_,
        float loss_trend_,
        const RunningStatistics& grad_stats_,
        float learning_rate_ = 0.0f
    ) : loss(loss_),
        gradients(gradients_),
        epoch(epoch_),
        step(step_),
        loss_trend(loss_trend_),
        grad_stats(grad_stats_),
        learning_rate(learning_rate_) {}
}; 