#ifndef QUANTIZATION_CONFIG_HPP
#define QUANTIZATION_CONFIG_HPP

#include <cstddef>
#include <string>
#include "quantization.hpp"  // For QuantizationMode enum

/**
 * @file quantization_config.hpp
 * @brief Unified quantization configuration for training and inference
 * 
 * This module implements the Manifold Nyquist Criterion (Reich, 2025):
 * Per-channel quantization is optimal when heterogeneity H > 2^(b-1).
 */

// Use QuantizationMode from quantization.hpp
// enum class QuantizationMode {
//     PerTensor,   ///< Global quantization scale (legacy)
//     PerChannel,  ///< Per-channel quantization (optimal for transformers)
//     Auto         ///< Auto-select based on heterogeneity
// };

enum class QuantizationTarget {
    NONE           = 0x00,
    WEIGHTS        = 0x01,  ///< Quantize model weights (INT8/INT4)
    ACTIVATIONS    = 0x02,  ///< Quantize activations/hidden states
    GRADIENTS      = 0x04,  ///< Quantize gradients (for distributed training)
    KV_CACHE       = 0x08,  ///< Quantize key-value cache (inference only)
    ALL            = 0xFF
};

inline QuantizationTarget operator|(QuantizationTarget a, QuantizationTarget b) {
    return static_cast<QuantizationTarget>(static_cast<int>(a) | static_cast<int>(b));
}

inline bool operator&(QuantizationTarget a, QuantizationTarget b) {
    return (static_cast<int>(a) & static_cast<int>(b)) != 0;
}

/**
 * @brief Global quantization configuration
 * 
 * Designed to be compatible with:
 * - tinyllama.cpp (inference engine with GGUF weights)
 * - transformer_cpp (BFT training platform)
 */
struct QuantizationConfig {
    // Core settings
    QuantizationMode mode = QuantizationMode::Disabled;
    QuantizationTarget targets = QuantizationTarget::NONE;
    size_t bits = 8;  ///< Quantization bit width (4, 6, 8 supported)
    
    // Manifold Nyquist Criterion threshold
    float heterogeneity_threshold = 127.0f;  ///< H_max for per-tensor (=2^b-1 for INT8)
    bool auto_measure_heterogeneity = true;  ///< Measure H at runtime
    
    // Symmetric vs asymmetric quantization
    bool symmetric = false;  ///< Use symmetric quantization (scale only, no zero-point)
    
    // Calibration (for activation quantization)
    size_t calibration_samples = 100;  ///< Number of samples for calibration
    bool dynamic_quantization = true;   ///< Recompute scales per forward pass
    
    // Diagnostics (for paper validation)
    bool log_heterogeneity = false;     ///< Log H per layer for analysis
    std::string heterogeneity_log_path = "heterogeneity.csv";
    
    // GGUF weight loading (for tinyllama.cpp compatibility)
    std::string weight_format = "none";  ///< "Q4_K_M", "Q6_K", "Q8_0", "none"
    
    // Helper methods
    bool should_quantize_activations() const {
        return mode != QuantizationMode::Disabled && 
               (targets & QuantizationTarget::ACTIVATIONS);
    }
    
    bool should_quantize_weights() const {
        return mode != QuantizationMode::Disabled && 
               (targets & QuantizationTarget::WEIGHTS);
    }
    
    bool should_quantize_gradients() const {
        return mode != QuantizationMode::Disabled && 
               (targets & QuantizationTarget::GRADIENTS);
    }
    
    bool should_quantize_kv_cache() const {
        return mode != QuantizationMode::Disabled && 
               (targets & QuantizationTarget::KV_CACHE);
    }
    
    size_t get_quant_max() const {
        if (symmetric) {
            return (1 << (bits - 1)) - 1;  // INT8: 127
        } else {
            return (1 << bits) - 1;  // INT8: 255
        }
    }
    
    // Factory methods
    static QuantizationConfig disabled() {
        return QuantizationConfig{};
    }
    
    static QuantizationConfig int8_activations() {
        QuantizationConfig config;
        config.mode = QuantizationMode::Auto;
        config.targets = QuantizationTarget::ACTIVATIONS;
        config.bits = 8;
        config.symmetric = false;
        return config;
    }
    
    static QuantizationConfig int8_weights_and_activations() {
        QuantizationConfig config;
        config.mode = QuantizationMode::Auto;
        config.targets = QuantizationTarget::WEIGHTS | QuantizationTarget::ACTIVATIONS;
        config.bits = 8;
        config.symmetric = false;
        return config;
    }
    
    static QuantizationConfig full_int8() {
        QuantizationConfig config;
        config.mode = QuantizationMode::Auto;
        config.targets = QuantizationTarget::ALL;
        config.bits = 8;
        config.symmetric = false;
        return config;
    }
};

#endif // QUANTIZATION_CONFIG_HPP

