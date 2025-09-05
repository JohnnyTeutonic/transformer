#pragma once

#include "matrix.hpp"
#include <vector>
#include <chrono>
#include <random>
#include <memory>
#include <atomic>

namespace compression {

enum class CompressionType {
    TOP_K_SPARSIFICATION,
    RANDOM_SPARSIFICATION,
    THRESHOLD_SPARSIFICATION
};

struct NetworkConditions {
    float bandwidth_mbps = 100.0f;
    float latency_ms = 50.0f;
    size_t num_peers = 4;
    float packet_loss_rate = 0.0f;
};

struct LayerMetadata {
    size_t layer_index = 0;
    size_t original_rows = 0;
    size_t original_cols = 0;
    size_t num_selected = 0;
    size_t compressed_size = 0;
    CompressionType compression_type = CompressionType::TOP_K_SPARSIFICATION;
    float quantization_scale = 1.0f;
    uint8_t quantization_zero_point = 0;
    uint8_t quantization_bits = 8;
    float min_value = 0.0f;
};

struct CompressedGradients {
    std::vector<uint8_t> compressed_data;
    std::vector<LayerMetadata> layer_metadata;
    size_t original_size = 0;
    size_t compressed_size = 0;
    float compression_ratio = 1.0f;
    uint64_t compression_time_us = 0;
};

struct QuantizationResult {
    std::vector<uint8_t> quantized_values;
    float scale = 1.0f;
    uint8_t zero_point = 0;
    float min_val = 0.0f;
};

struct LayerCompressionResult {
    std::vector<uint8_t> compressed_data;
    LayerMetadata metadata;
    Matrix reconstructed;
};

struct CompressionConfig {
    float initial_compression_ratio = 10.0f;
    float sparsity_ratio = 0.1f;  // Keep top 10% of gradients
    uint8_t quantization_bits = 8;
    bool enable_error_feedback = true;
    bool enable_adaptive_ratios = true;
    float error_feedback_momentum = 0.9f;
    float adaptation_learning_rate = 0.01f;
};

struct CompressionStats {
    std::chrono::steady_clock::time_point timestamp;
    float compression_ratio = 1.0f;
    uint64_t compression_time_us = 0;
    NetworkConditions network_conditions;
};

class AdaptiveCompressor {
public:
    explicit AdaptiveCompressor(const CompressionConfig& config = CompressionConfig{});
    
    // Main compression/decompression interface
    CompressedGradients compress_gradients(
        const std::vector<Matrix>& gradients,
        const NetworkConditions& network_conditions = NetworkConditions{});
    
    std::vector<Matrix> decompress_gradients(const CompressedGradients& compressed);
    
    // Configuration and statistics
    void update_config(const CompressionConfig& config) { config_ = config; }
    CompressionConfig get_config() const { return config_; }
    CompressionStats get_compression_stats() const;
    
    // Reset error buffers (useful for new training runs)
    void reset_error_buffers() { 
        for (auto& buffer : error_buffers_) {
            buffer.zero();
        }
    }

private:
    // Core compression algorithms
    LayerCompressionResult compress_single_layer(
        const Matrix& gradient, size_t layer_idx, float compression_ratio);
    
    Matrix decompress_single_layer(
        const std::vector<uint8_t>& compressed_data, size_t offset,
        const LayerMetadata& metadata);
    
    // Error feedback mechanism
    Matrix apply_error_feedback(const Matrix& gradient, size_t layer_idx);
    void update_error_buffer(const Matrix& original, const Matrix& reconstructed, size_t layer_idx);
    
    // Adaptive compression ratio
    float adapt_compression_ratio(const NetworkConditions& conditions);
    
    // Quantization utilities
    QuantizationResult quantize_values(const std::vector<float>& values);
    float dequantize_value(uint8_t quantized_val, float scale, uint8_t zero_point, float min_val = 0.0f);
    
    // Serialization utilities
    void serialize_compressed_layer(
        std::vector<uint8_t>& output,
        const std::vector<uint32_t>& indices,
        const std::vector<uint8_t>& quantized_values,
        const LayerMetadata& metadata);
    
    void deserialize_compressed_layer(
        const std::vector<uint8_t>& compressed_data, size_t offset,
        const LayerMetadata& metadata,
        std::vector<uint32_t>& indices,
        std::vector<uint8_t>& quantized_values);
    
    // Utility functions
    size_t calculate_total_size(const std::vector<Matrix>& gradients);
    void update_compression_history(const CompressedGradients& result, const NetworkConditions& conditions);
    
    // Configuration and state
    CompressionConfig config_;
    std::vector<Matrix> error_buffers_;  // One per layer for error feedback
    std::vector<CompressionStats> compression_history_;
    std::mt19937 rng_;
};

// Factory function for easy integration
std::unique_ptr<AdaptiveCompressor> create_adaptive_compressor(
    const CompressionConfig& config = CompressionConfig{});

} // namespace compression
