#pragma once

#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include "types.hpp"
#include "half_precision.hpp"

namespace gradient_accumulation {

struct AccumulationConfig {
    size_t accumulation_steps = 4;
    bool enable_mixed_precision = true;
    float initial_loss_scale = 65536.0f;
    float loss_scale_growth_factor = 2.0f;
    float loss_scale_backoff_factor = 0.5f;
    size_t loss_scale_growth_interval = 2000;
    size_t max_loss_scale_reduction_attempts = 10;
    float gradient_clip_threshold = 1.0f;
    bool enable_gradient_clipping = true;
};

struct AccumulationStats {
    size_t total_accumulations = 0;
    size_t overflow_count = 0;
    size_t underflow_count = 0;
    float current_loss_scale = 0.0f;
    float average_gradient_norm = 0.0f;
    size_t steps_since_overflow = 0;
};

enum class GradientStatus {
    VALID,
    OVERFLOW,
    UNDERFLOW,
    CLIPPED
};

class GradientAccumulator {
private:
    AccumulationConfig config_;
    
    // Accumulation buffers (stored in FP32 for numerical stability)
    std::vector<std::vector<float>> accumulated_gradients_;
    std::vector<size_t> parameter_sizes_;
    
    // Mixed precision state
    std::atomic<float> current_loss_scale_;
    std::atomic<size_t> steps_since_overflow_;
    std::atomic<size_t> accumulation_count_;
    
    // Statistics
    AccumulationStats stats_;
    std::mutex stats_mutex_;
    
    // Gradient norm history for adaptive clipping
    std::vector<float> gradient_norm_history_;
    static constexpr size_t NORM_HISTORY_SIZE = 100;
    
public:
    explicit GradientAccumulator(const AccumulationConfig& config);
    ~GradientAccumulator() = default;
    
    /**
     * Initialize accumulation buffers for given parameter shapes
     * @param parameter_shapes Vector of parameter tensor shapes
     */
    void initialize(const std::vector<std::vector<size_t>>& parameter_shapes);
    
    /**
     * Accumulate gradients from FP16 computation
     * @param gradients Vector of gradient tensors (FP16)
     * @param loss_value Current loss value for scaling
     * @return Status indicating if gradients are valid
     */
    GradientStatus accumulate_gradients_fp16(const std::vector<std::vector<half_precision::half>>& gradients, float loss_value);
    
    /**
     * Accumulate gradients from FP32 computation
     * @param gradients Vector of gradient tensors (FP32)
     * @return Status indicating if gradients are valid
     */
    GradientStatus accumulate_gradients_fp32(const std::vector<std::vector<float>>& gradients);
    
    /**
     * Check if accumulation is complete and gradients are ready
     * @return True if ready to apply gradients
     */
    bool is_accumulation_complete() const;
    
    /**
     * Get accumulated gradients and reset accumulation
     * @return Vector of accumulated gradient tensors (FP32)
     */
    std::vector<std::vector<float>> get_and_reset_gradients();
    
    /**
     * Apply gradient clipping to accumulated gradients
     * @param gradients Gradient tensors to clip
     * @param clip_threshold Maximum gradient norm
     * @return Actual gradient norm before clipping
     */
    float apply_gradient_clipping(std::vector<std::vector<float>>& gradients, float clip_threshold);
    
    /**
     * Update loss scale based on gradient status
     * @param status Current gradient status
     */
    void update_loss_scale(GradientStatus status);
    
    /**
     * Get current loss scale for FP16 training
     * @return Current loss scale value
     */
    float get_current_loss_scale() const;
    
    /**
     * Get accumulation statistics
     * @return Current accumulation statistics
     */
    AccumulationStats get_statistics() const;
    
    /**
     * Reset accumulation state (useful for model changes)
     */
    void reset();
    
    /**
     * Set new accumulation configuration
     * @param config New configuration
     */
    void update_config(const AccumulationConfig& config);
    
    /**
     * Get current configuration
     * @return Current accumulation configuration
     */
    const AccumulationConfig& get_config() const { return config_; }

private:
    /**
     * Check for gradient overflow/underflow in FP16 gradients
     * @param gradients FP16 gradient tensors
     * @return True if overflow/underflow detected
     */
    bool check_gradient_overflow(const std::vector<std::vector<half_precision::half>>& gradients) const;
    
    /**
     * Scale gradients by loss scale factor
     * @param gradients Gradient tensors to scale
     * @param scale Scale factor
     */
    void scale_gradients(std::vector<std::vector<float>>& gradients, float scale) const;
    
    /**
     * Calculate L2 norm of gradient tensors
     * @param gradients Gradient tensors
     * @return L2 norm of all gradients
     */
    float calculate_gradient_norm(const std::vector<std::vector<float>>& gradients) const;
    
    /**
     * Update gradient norm history for adaptive clipping
     * @param norm Current gradient norm
     */
    void update_gradient_norm_history(float norm);
    
    /**
     * Get adaptive gradient clipping threshold
     * @return Adaptive clipping threshold based on history
     */
    float get_adaptive_clip_threshold() const;
    
    /**
     * Convert FP16 gradients to FP32 with loss scaling
     * @param fp16_gradients Input FP16 gradients
     * @param fp32_gradients Output FP32 gradients
     * @param loss_scale Loss scale factor
     */
    void convert_and_scale_gradients(
        const std::vector<std::vector<half_precision::half>>& fp16_gradients,
        std::vector<std::vector<float>>& fp32_gradients,
        float loss_scale
    ) const;
    
    /**
     * Add gradients to accumulation buffer
     * @param gradients Gradients to add (FP32)
     */
    void add_to_accumulation_buffer(const std::vector<std::vector<float>>& gradients);
    
    /**
     * Check if loss scale needs adjustment
     * @return True if loss scale should be updated
     */
    bool should_update_loss_scale() const;
};

/**
 * Utility class for automatic gradient accumulation management
 */
class ScopedGradientAccumulation {
private:
    GradientAccumulator& accumulator_;
    bool accumulation_complete_;
    
public:
    explicit ScopedGradientAccumulation(GradientAccumulator& accumulator)
        : accumulator_(accumulator), accumulation_complete_(false) {}
    
    ~ScopedGradientAccumulation() {
        if (accumulation_complete_) {
            // Gradients were retrieved, accumulation is reset
        }
    }
    
    /**
     * Add gradients to accumulation
     * @param gradients Gradient tensors
     * @return Status of gradient accumulation
     */
    template<typename T>
    GradientStatus add_gradients(const std::vector<std::vector<T>>& gradients) {
        if constexpr (std::is_same_v<T, half_precision::half>) {
            return accumulator_.accumulate_gradients_fp16(gradients, 1.0f);
        } else {
            return accumulator_.accumulate_gradients_fp32(gradients);
        }
    }
    
    /**
     * Check if accumulation is complete
     * @return True if ready to apply gradients
     */
    bool is_complete() const {
        return accumulator_.is_accumulation_complete();
    }
    
    /**
     * Get accumulated gradients (only if complete)
     * @return Accumulated gradients or empty vector
     */
    std::vector<std::vector<float>> get_gradients() {
        if (accumulator_.is_accumulation_complete()) {
            accumulation_complete_ = true;
            return accumulator_.get_and_reset_gradients();
        }
        return {};
    }
};

} // namespace gradient_accumulation
