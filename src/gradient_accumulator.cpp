#include "gradient_accumulator.hpp"
#include "logger.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>

namespace gradient_accumulation {

GradientAccumulator::GradientAccumulator(const AccumulationConfig& config)
    : config_(config)
    , current_loss_scale_(config.initial_loss_scale)
    , steps_since_overflow_(0)
    , accumulation_count_(0) {
    
    gradient_norm_history_.reserve(NORM_HISTORY_SIZE);
    
    // Initialize statistics
    stats_.current_loss_scale = config.initial_loss_scale;
    stats_.total_accumulations = 0;
    stats_.overflow_count = 0;
    stats_.underflow_count = 0;
    stats_.average_gradient_norm = 0.0f;
    stats_.steps_since_overflow = 0;
    
    logger::log_info("Initialized GradientAccumulator with " + std::to_string(config.accumulation_steps) + 
                     " accumulation steps, loss scale: " + std::to_string(config.initial_loss_scale));
}

void GradientAccumulator::initialize(const std::vector<std::vector<size_t>>& parameter_shapes) {
    accumulated_gradients_.clear();
    parameter_sizes_.clear();
    
    accumulated_gradients_.resize(parameter_shapes.size());
    parameter_sizes_.resize(parameter_shapes.size());
    
    for (size_t i = 0; i < parameter_shapes.size(); ++i) {
        size_t total_size = 1;
        for (size_t dim : parameter_shapes[i]) {
            total_size *= dim;
        }
        
        parameter_sizes_[i] = total_size;
        accumulated_gradients_[i].resize(total_size, 0.0f);
    }
    
    logger::log_info("Initialized gradient accumulation buffers for " + std::to_string(parameter_shapes.size()) + " parameter tensors");
}

GradientStatus GradientAccumulator::accumulate_gradients_fp16(
    const std::vector<std::vector<half_precision::half>>& gradients, 
    float loss_value) {
    
    if (gradients.size() != accumulated_gradients_.size()) {
        logger::log_error("Gradient tensor count mismatch: expected " + std::to_string(accumulated_gradients_.size()) + 
                         ", got " + std::to_string(gradients.size()));
        return GradientStatus::OVERFLOW;
    }
    
    // Check for overflow/underflow in FP16 gradients
    if (check_gradient_overflow(gradients)) {
        logger::log_warning("Gradient overflow detected in FP16 computation");
        update_loss_scale(GradientStatus::OVERFLOW);
        return GradientStatus::OVERFLOW;
    }
    
    // Convert FP16 to FP32 with loss scaling
    std::vector<std::vector<float>> fp32_gradients(gradients.size());
    float current_scale = get_current_loss_scale();
    
    convert_and_scale_gradients(gradients, fp32_gradients, 1.0f / current_scale);
    
    // Add to accumulation buffer
    add_to_accumulation_buffer(fp32_gradients);
    
    // Update statistics
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.total_accumulations++;
        stats_.current_loss_scale = current_scale;
        stats_.steps_since_overflow = steps_since_overflow_.load();
    }
    
    return GradientStatus::VALID;
}

GradientStatus GradientAccumulator::accumulate_gradients_fp32(const std::vector<std::vector<float>>& gradients) {
    if (gradients.size() != accumulated_gradients_.size()) {
        logger::log_error("Gradient tensor count mismatch: expected " + std::to_string(accumulated_gradients_.size()) + 
                         ", got " + std::to_string(gradients.size()));
        return GradientStatus::OVERFLOW;
    }
    
    // Add to accumulation buffer
    add_to_accumulation_buffer(gradients);
    
    // Update statistics
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.total_accumulations++;
    }
    
    return GradientStatus::VALID;
}

bool GradientAccumulator::is_accumulation_complete() const {
    return accumulation_count_.load() >= config_.accumulation_steps;
}

std::vector<std::vector<float>> GradientAccumulator::get_and_reset_gradients() {
    if (!is_accumulation_complete()) {
        logger::log_warning("Attempting to get gradients before accumulation is complete");
        return {};
    }
    
    // Average the accumulated gradients
    std::vector<std::vector<float>> result = accumulated_gradients_;
    float scale_factor = 1.0f / static_cast<float>(config_.accumulation_steps);
    
    for (auto& tensor : result) {
        for (float& grad : tensor) {
            grad *= scale_factor;
        }
    }
    
    // Apply gradient clipping if enabled
    if (config_.enable_gradient_clipping) {
        float clip_threshold = config_.gradient_clip_threshold;
        
        // Use adaptive clipping if we have history
        if (!gradient_norm_history_.empty()) {
            clip_threshold = get_adaptive_clip_threshold();
        }
        
        float grad_norm = apply_gradient_clipping(result, clip_threshold);
        update_gradient_norm_history(grad_norm);
        
        // Update statistics
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.average_gradient_norm = (stats_.average_gradient_norm * 0.9f) + (grad_norm * 0.1f);
        }
    }
    
    // Reset accumulation state
    for (auto& tensor : accumulated_gradients_) {
        std::fill(tensor.begin(), tensor.end(), 0.0f);
    }
    accumulation_count_.store(0);
    
    logger::log_debug("Retrieved and reset accumulated gradients");
    
    return result;
}

float GradientAccumulator::apply_gradient_clipping(std::vector<std::vector<float>>& gradients, float clip_threshold) {
    float total_norm = calculate_gradient_norm(gradients);
    
    if (total_norm > clip_threshold) {
        float scale_factor = clip_threshold / total_norm;
        
        for (auto& tensor : gradients) {
            for (float& grad : tensor) {
                grad *= scale_factor;
            }
        }
        
        logger::log_debug("Applied gradient clipping: norm " + std::to_string(total_norm) + 
                         " -> " + std::to_string(clip_threshold));
    }
    
    return total_norm;
}

void GradientAccumulator::update_loss_scale(GradientStatus status) {
    if (!config_.enable_mixed_precision) {
        return;
    }
    
    if (status == GradientStatus::OVERFLOW) {
        // Reduce loss scale
        float new_scale = current_loss_scale_.load() * config_.loss_scale_backoff_factor;
        current_loss_scale_.store(std::max(1.0f, new_scale));
        steps_since_overflow_.store(0);
        
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.overflow_count++;
            stats_.current_loss_scale = current_loss_scale_.load();
        }
        
        logger::log_warning("Reduced loss scale to " + std::to_string(current_loss_scale_.load()) + " due to overflow");
    } else if (status == GradientStatus::VALID) {
        size_t steps = steps_since_overflow_.fetch_add(1) + 1;
        
        // Increase loss scale if we've been stable for a while
        if (steps >= config_.loss_scale_growth_interval) {
            float new_scale = current_loss_scale_.load() * config_.loss_scale_growth_factor;
            current_loss_scale_.store(std::min(65536.0f, new_scale)); // Cap at reasonable maximum
            steps_since_overflow_.store(0);
            
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                stats_.current_loss_scale = current_loss_scale_.load();
            }
            
            logger::log_info("Increased loss scale to " + std::to_string(current_loss_scale_.load()) + 
                           " after " + std::to_string(steps) + " stable steps");
        }
        
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.steps_since_overflow = steps;
        }
    }
}

float GradientAccumulator::get_current_loss_scale() const {
    return current_loss_scale_.load();
}

AccumulationStats GradientAccumulator::get_statistics() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void GradientAccumulator::reset() {
    // Reset accumulation buffers
    for (auto& tensor : accumulated_gradients_) {
        std::fill(tensor.begin(), tensor.end(), 0.0f);
    }
    
    accumulation_count_.store(0);
    current_loss_scale_.store(config_.initial_loss_scale);
    steps_since_overflow_.store(0);
    
    // Reset statistics
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.total_accumulations = 0;
        stats_.overflow_count = 0;
        stats_.underflow_count = 0;
        stats_.current_loss_scale = config_.initial_loss_scale;
        stats_.average_gradient_norm = 0.0f;
        stats_.steps_since_overflow = 0;
    }
    
    gradient_norm_history_.clear();
    
    logger::log_info("Reset gradient accumulation state");
}

void GradientAccumulator::update_config(const AccumulationConfig& config) {
    config_ = config;
    current_loss_scale_.store(config.initial_loss_scale);
    
    logger::log_info("Updated gradient accumulation configuration");
}

bool GradientAccumulator::check_gradient_overflow(const std::vector<std::vector<half_precision::half>>& gradients) const {
    for (const auto& tensor : gradients) {
        for (const auto& grad : tensor) {
            float grad_f32 = half_precision::half_to_float(grad);
            if (!std::isfinite(grad_f32) || std::abs(grad_f32) > 65504.0f) { // FP16 max value
                return true;
            }
        }
    }
    return false;
}

void GradientAccumulator::scale_gradients(std::vector<std::vector<float>>& gradients, float scale) const {
    for (auto& tensor : gradients) {
        for (float& grad : tensor) {
            grad *= scale;
        }
    }
}

float GradientAccumulator::calculate_gradient_norm(const std::vector<std::vector<float>>& gradients) const {
    float total_norm_squared = 0.0f;
    
    for (const auto& tensor : gradients) {
        for (float grad : tensor) {
            total_norm_squared += grad * grad;
        }
    }
    
    return std::sqrt(total_norm_squared);
}

void GradientAccumulator::update_gradient_norm_history(float norm) {
    gradient_norm_history_.push_back(norm);
    
    // Keep only recent history
    if (gradient_norm_history_.size() > NORM_HISTORY_SIZE) {
        gradient_norm_history_.erase(gradient_norm_history_.begin());
    }
}

float GradientAccumulator::get_adaptive_clip_threshold() const {
    if (gradient_norm_history_.empty()) {
        return config_.gradient_clip_threshold;
    }
    
    // Calculate percentile-based adaptive threshold
    std::vector<float> sorted_norms = gradient_norm_history_;
    std::sort(sorted_norms.begin(), sorted_norms.end());
    
    // Use 95th percentile as adaptive threshold
    size_t percentile_idx = static_cast<size_t>(sorted_norms.size() * 0.95f);
    percentile_idx = std::min(percentile_idx, sorted_norms.size() - 1);
    
    float adaptive_threshold = sorted_norms[percentile_idx];
    
    // Don't let adaptive threshold be too different from configured threshold
    float min_threshold = config_.gradient_clip_threshold * 0.5f;
    float max_threshold = config_.gradient_clip_threshold * 2.0f;
    
    return std::clamp(adaptive_threshold, min_threshold, max_threshold);
}

void GradientAccumulator::convert_and_scale_gradients(
    const std::vector<std::vector<half_precision::half>>& fp16_gradients,
    std::vector<std::vector<float>>& fp32_gradients,
    float loss_scale) const {
    
    for (size_t i = 0; i < fp16_gradients.size(); ++i) {
        fp32_gradients[i].resize(fp16_gradients[i].size());
        
        for (size_t j = 0; j < fp16_gradients[i].size(); ++j) {
            float grad_f32 = half_precision::half_to_float(fp16_gradients[i][j]);
            fp32_gradients[i][j] = grad_f32 * loss_scale;
        }
    }
}

void GradientAccumulator::add_to_accumulation_buffer(const std::vector<std::vector<float>>& gradients) {
    for (size_t i = 0; i < gradients.size(); ++i) {
        if (gradients[i].size() != accumulated_gradients_[i].size()) {
            logger::log_error("Gradient tensor size mismatch at index " + std::to_string(i) + 
                             ": expected " + std::to_string(accumulated_gradients_[i].size()) + 
                             ", got " + std::to_string(gradients[i].size()));
            continue;
        }
        
        for (size_t j = 0; j < gradients[i].size(); ++j) {
            accumulated_gradients_[i][j] += gradients[i][j];
        }
    }
    
    accumulation_count_.fetch_add(1);
}

bool GradientAccumulator::should_update_loss_scale() const {
    return config_.enable_mixed_precision && 
           (steps_since_overflow_.load() >= config_.loss_scale_growth_interval);
}

} // namespace gradient_accumulation
