#pragma once
#include "tensor.hpp"
#include "loss_visualizer.hpp"
#include <deque>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <memory>

class LossTracker {
public:
    static constexpr size_t WINDOW_SIZE = 100;
    static constexpr size_t MIN_SAMPLES = 10;

    LossTracker() : visualizer(std::make_unique<LossVisualizer>("loss.log")) {}

    void add_loss(float loss);
    bool should_adjust_lr() const;
    float get_trend() const;
    float get_recent_average() const { return recent_average; }
    float get_overall_average() const { return overall_average; }
    float compute_loss(const Tensor& predictions, const Tensor& targets);
    size_t get_sample_count() const { return loss_history.size(); }

private:
    std::deque<float> loss_history;
    float recent_average = 0.0f;
    float overall_average = 0.0f;
    std::unique_ptr<LossVisualizer> visualizer;

    void update_statistics();
}; 