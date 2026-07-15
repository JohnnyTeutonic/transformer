#include "../include/quantization_api.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <chrono>

#ifdef USE_CUDA
#include "../include/cuda/quantization_kernels.cuh"
#endif

namespace quant {

// ============================================================================
// QuantizationState implementation
// ============================================================================

QuantizationState::QuantizationState(const QuantizationConfig& cfg)
    : config(cfg) {
    
    // Map QuantizationConfig::QuantizationMode to Quantizer's QuantizationMode
    ::QuantizationMode quant_mode;
    switch (config.mode) {
        case QuantizationMode::PerTensor:
            quant_mode = ::QuantizationMode::PerTensor;
            break;
        case QuantizationMode::PerChannel:
            quant_mode = ::QuantizationMode::PerChannel;
            break;
        case QuantizationMode::Auto:
            quant_mode = ::QuantizationMode::Auto;
            break;
        default:
            quant_mode = ::QuantizationMode::Auto;
            break;
    }
    
    // Create underlying quantizer
    quantizer = std::make_unique<Quantizer>(config.bits, quant_mode);
    
    // Open heterogeneity log if requested
    if (config.log_heterogeneity) {
        heterogeneity_log.open(config.heterogeneity_log_path, std::ios::out);
        if (heterogeneity_log.is_open()) {
            // Write CSV header
            heterogeneity_log << "forward_pass,layer_name,H,max_std,median_std,num_channels,requires_per_channel\n";
        } else {
            std::cerr << "Warning: Could not open heterogeneity log file: "
                      << config.heterogeneity_log_path << std::endl;
        }
    }
}

QuantizationState::~QuantizationState() {
    if (heterogeneity_log.is_open()) {
        heterogeneity_log.close();
    }
}

HeterogeneityMetrics QuantizationState::measure_heterogeneity(
    const Matrix& activations,
    const std::string& layer_name) {
    
    HeterogeneityMetrics metrics;
    metrics.layer_name = layer_name;
    metrics.num_channels = activations.cols();
    
    size_t rows = activations.rows();
    size_t cols = activations.cols();
    
    // Compute std dev per channel
    std::vector<float> channel_stds(cols, 0.0f);
    
    // MSVC: loop vars must be signed int
    #pragma omp parallel for if(cols > 100)
    for (int c = 0; c < static_cast<int>(cols); ++c) {
        // Compute mean
        float mean = 0.0f;
        for (size_t r = 0; r < rows; ++r) {
            mean += activations(r, c);
        }
        mean /= rows;
        
        // Compute variance
        float variance = 0.0f;
        for (size_t r = 0; r < rows; ++r) {
            float diff = activations(r, c) - mean;
            variance += diff * diff;
        }
        variance /= rows;
        
        channel_stds[c] = std::sqrt(variance);
    }
    
    // Compute max and median std dev
    std::vector<float> sorted_stds = channel_stds;
    std::sort(sorted_stds.begin(), sorted_stds.end());
    
    metrics.max_std = sorted_stds.back();
    metrics.median_std = sorted_stds[cols / 2];
    
    // Compute heterogeneity H = max / median
    if (metrics.median_std > 1e-8f) {
        metrics.H = metrics.max_std / metrics.median_std;
    } else {
        metrics.H = 1.0f;  // Uniform variance (no heterogeneity)
    }
    
    // Log if requested
    if (config.log_heterogeneity && heterogeneity_log.is_open()) {
        heterogeneity_log << forward_pass_count << ","
                          << layer_name << ","
                          << metrics.H << ","
                          << metrics.max_std << ","
                          << metrics.median_std << ","
                          << metrics.num_channels << ","
                          << (metrics.requires_per_channel(config.bits) ? "yes" : "no")
                          << "\n";
        heterogeneity_log.flush();
    }
    
    return metrics;
}

Matrix QuantizationState::quantize(const Matrix& activations, const std::string& layer_name) {
    if (config.mode == QuantizationMode::Disabled) {
        return activations;  // Pass-through
    }
    
    // Measure heterogeneity if requested
    if (config.auto_measure_heterogeneity && !layer_name.empty()) {
        measure_heterogeneity(activations, layer_name);
    }
    
    // Quantize using underlying quantizer
    return quantizer->quantize(activations);
}

Matrix QuantizationState::dequantize(const Matrix& quantized) {
    if (config.mode == QuantizationMode::Disabled) {
        return quantized;  // Pass-through
    }
    
    return quantizer->dequantize(quantized);
}

// ============================================================================
// Functional API implementation
// ============================================================================

void compute_per_channel_scales(
    const float* input,
    size_t rows,
    size_t cols,
    size_t bits,
    bool symmetric,
    float* scales,
    float* zero_points) {
    
    float quant_max = symmetric ? ((1 << (bits - 1)) - 1) : ((1 << bits) - 1);
    
    // MSVC: loop vars must be signed int
    #pragma omp parallel for if(cols > 100)
    for (int c = 0; c < static_cast<int>(cols); ++c) {
        // Find min/max for this channel
        float min_val = input[c];
        float max_val = input[c];
        
        for (size_t r = 0; r < rows; ++r) {
            float val = input[r * cols + c];
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }
        
        // Compute scale and zero point
        if (symmetric) {
            float abs_max = std::max(std::abs(min_val), std::abs(max_val));
            if (abs_max < 1e-8f) {
                scales[c] = 1.0f;
                zero_points[c] = 0.0f;
            } else {
                scales[c] = abs_max / quant_max;
                zero_points[c] = 0.0f;
            }
        } else {
            float range = max_val - min_val;
            if (range < 1e-8f) {
                scales[c] = 1.0f;
                zero_points[c] = 0.0f;
            } else {
                scales[c] = range / quant_max;
                zero_points[c] = -min_val / scales[c];
            }
        }
    }
}

void quantize_per_channel(
    const float* input,
    float* output,
    size_t rows,
    size_t cols,
    const float* scales,
    const float* zero_points) {
    
    // MSVC: loop vars must be signed int
    #pragma omp parallel for collapse(2) if(rows * cols > 10000)
    for (int r = 0; r < static_cast<int>(rows); ++r) {
        for (int c = 0; c < static_cast<int>(cols); ++c) {
            size_t idx = r * cols + c;
            float val = input[idx];
            float scale = scales[c];
            float zp = zero_points[c];
            output[idx] = std::nearbyint(val / scale + zp);
        }
    }
}

void dequantize_per_channel(
    const float* input,
    float* output,
    size_t rows,
    size_t cols,
    const float* scales,
    const float* zero_points) {
    
    // MSVC: loop vars must be signed int
    #pragma omp parallel for collapse(2) if(rows * cols > 10000)
    for (int r = 0; r < static_cast<int>(rows); ++r) {
        for (int c = 0; c < static_cast<int>(cols); ++c) {
            size_t idx = r * cols + c;
            float val = input[idx];
            float scale = scales[c];
            float zp = zero_points[c];
            output[idx] = (val - zp) * scale;
        }
    }
}

HeterogeneityMetrics measure_heterogeneity(
    const float* activations,
    size_t rows,
    size_t cols,
    const std::string& layer_name) {
    
    HeterogeneityMetrics metrics;
    metrics.layer_name = layer_name;
    metrics.num_channels = cols;
    
    // Compute std dev per channel
    std::vector<float> channel_stds(cols, 0.0f);
    
    // MSVC: loop vars must be signed int
    #pragma omp parallel for if(cols > 100)
    for (int c = 0; c < static_cast<int>(cols); ++c) {
        // Compute mean
        float mean = 0.0f;
        for (size_t r = 0; r < rows; ++r) {
            mean += activations[r * cols + c];
        }
        mean /= rows;
        
        // Compute variance
        float variance = 0.0f;
        for (size_t r = 0; r < rows; ++r) {
            float diff = activations[r * cols + c] - mean;
            variance += diff * diff;
        }
        variance /= rows;
        
        channel_stds[c] = std::sqrt(variance);
    }
    
    // Compute max and median std dev
    std::vector<float> sorted_stds = channel_stds;
    std::sort(sorted_stds.begin(), sorted_stds.end());
    
    metrics.max_std = sorted_stds.back();
    metrics.median_std = sorted_stds[cols / 2];
    
    // Compute heterogeneity H = max / median
    if (metrics.median_std > 1e-8f) {
        metrics.H = metrics.max_std / metrics.median_std;
    } else {
        metrics.H = 1.0f;  // Uniform variance (no heterogeneity)
    }
    
    return metrics;
}

} // namespace quant

