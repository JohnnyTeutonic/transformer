#pragma once

#include "quantization.hpp"
#include "quantization_config.hpp"
#include "matrix.hpp"
#include <memory>
#include <fstream>
#include <string>

/**
 * @file quantization_api.hpp
 * @brief Clean quantization API following tinyllama.cpp design patterns
 * 
 * This provides a unified interface for:
 * - transformer_cpp (BFT distributed training)
 * - tinyllama.cpp (lightweight inference)
 * 
 * Standardized layout: [rows=batch*seq, cols=hidden_dim]
 */

namespace quant {

/**
 * @brief Heterogeneity metrics for a tensor
 * 
 * Used to validate the Manifold Nyquist Criterion:
 * Per-channel quantization is optimal when H > 2^(b-1)
 */
struct HeterogeneityMetrics {
    float H;                  ///< Heterogeneity: max(σᵢ) / median(σᵢ)
    float max_std;            ///< Maximum std dev across channels
    float median_std;         ///< Median std dev across channels
    size_t num_channels;      ///< Number of channels (hidden dim)
    std::string layer_name;   ///< Layer identifier
    
    bool requires_per_channel(size_t bits) const {
        float threshold = (1 << (bits - 1)) - 1;  // 127 for INT8
        return H > threshold;
    }
};

/**
 * @brief Quantization state manager
 * 
 * Wraps the existing Quantizer class with a cleaner API
 */
class QuantizationState {
private:
    QuantizationConfig config;
    std::unique_ptr<Quantizer> quantizer;
    std::ofstream heterogeneity_log;
    size_t forward_pass_count = 0;
    
public:
    explicit QuantizationState(const QuantizationConfig& cfg);
    ~QuantizationState();
    
    /**
     * @brief Measure heterogeneity of activation tensor
     * 
     * @param activations Input tensor [rows=B*T, cols=D]
     * @param layer_name Layer identifier for logging
     * @return Heterogeneity metrics
     */
    HeterogeneityMetrics measure_heterogeneity(
        const Matrix& activations,
        const std::string& layer_name = "");
    
    /**
     * @brief Quantize activation tensor
     * 
     * Standard layout: [rows = batch*seq, cols = hidden_dim]
     * 
     * @param activations Input FP32 tensor
     * @param layer_name Layer identifier (for heterogeneity logging)
     * @return Quantized tensor
     */
    Matrix quantize(const Matrix& activations, const std::string& layer_name = "");
    
    /**
     * @brief Dequantize activation tensor
     * 
     * @param quantized Quantized tensor
     * @return FP32 tensor
     */
    Matrix dequantize(const Matrix& quantized);
    
    /**
     * @brief Get configuration
     * @return Current quantization config
     */
    const QuantizationConfig& get_config() const { return config; }
    
    /**
     * @brief Get underlying quantizer (for advanced use)
     * @return Pointer to quantizer
     */
    Quantizer* get_quantizer() { return quantizer.get(); }
};

/**
 * @brief Functional API for quantization (tinyllama.cpp style)
 * 
 * These are thin wrappers for one-off quantization without state management.
 * For training loops, prefer QuantizationState.
 */

/**
 * @brief Compute per-channel scales for a tensor
 * 
 * @param input Input tensor [rows, cols]
 * @param bits Quantization bits
 * @param symmetric Use symmetric quantization
 * @param[out] scales Output scales (size = cols)
 * @param[out] zero_points Output zero points (size = cols)
 */
void compute_per_channel_scales(
    const float* input,
    size_t rows,
    size_t cols,
    size_t bits,
    bool symmetric,
    float* scales,
    float* zero_points);

/**
 * @brief Quantize tensor with per-channel scales
 * 
 * @param input Input FP32 data
 * @param output Output quantized data (stored as FP32 for compatibility)
 * @param rows Number of rows
 * @param cols Number of columns (channels)
 * @param scales Per-channel scales
 * @param zero_points Per-channel zero points
 */
void quantize_per_channel(
    const float* input,
    float* output,
    size_t rows,
    size_t cols,
    const float* scales,
    const float* zero_points);

/**
 * @brief Dequantize tensor with per-channel scales
 * 
 * @param input Quantized data
 * @param output Output FP32 data
 * @param rows Number of rows
 * @param cols Number of columns (channels)
 * @param scales Per-channel scales
 * @param zero_points Per-channel zero points
 */
void dequantize_per_channel(
    const float* input,
    float* output,
    size_t rows,
    size_t cols,
    const float* scales,
    const float* zero_points);

/**
 * @brief Measure activation heterogeneity (standalone)
 * 
 * @param activations Input tensor [rows, cols]
 * @param rows Number of rows
 * @param cols Number of columns (hidden dim)
 * @param layer_name Layer identifier
 * @return Heterogeneity metrics
 */
HeterogeneityMetrics measure_heterogeneity(
    const float* activations,
    size_t rows,
    size_t cols,
    const std::string& layer_name = "");

} // namespace quant

