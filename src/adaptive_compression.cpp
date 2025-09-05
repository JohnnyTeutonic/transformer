#include "../include/adaptive_compression.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <iostream>
#include <numeric>

namespace compression {

AdaptiveCompressor::AdaptiveCompressor(const CompressionConfig& config)
    : config_(config), rng_(std::random_device{}()) {
    
    // Initialize error accumulation buffers
    error_buffers_.clear();
    compression_history_.reserve(1000); // Keep last 1000 compression stats
    
    std::cout << "AdaptiveCompressor initialized with:" << std::endl;
    std::cout << "- Initial compression ratio: " << config_.initial_compression_ratio << std::endl;
    std::cout << "- Top-K sparsity: " << config_.sparsity_ratio << std::endl;
    std::cout << "- Error feedback: " << (config_.enable_error_feedback ? "enabled" : "disabled") << std::endl;
    std::cout << "- Adaptive ratios: " << (config_.enable_adaptive_ratios ? "enabled" : "disabled") << std::endl;
}

CompressedGradients AdaptiveCompressor::compress_gradients(
    const std::vector<Matrix>& gradients,
    const NetworkConditions& network_conditions) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    CompressedGradients result;
    result.original_size = calculate_total_size(gradients);
    result.layer_metadata.reserve(gradients.size());
    
    // Adapt compression ratio based on network conditions
    float current_ratio = adapt_compression_ratio(network_conditions);
    
    // Process each gradient matrix
    for (size_t layer_idx = 0; layer_idx < gradients.size(); ++layer_idx) {
        const Matrix& grad = gradients[layer_idx];
        
        // Ensure error buffer exists for this layer
        if (error_buffers_.size() <= layer_idx) {
            error_buffers_.resize(layer_idx + 1);
        }
        if (error_buffers_[layer_idx].rows() != grad.rows() || 
            error_buffers_[layer_idx].cols() != grad.cols()) {
            error_buffers_[layer_idx] = Matrix(grad.rows(), grad.cols());
            error_buffers_[layer_idx].zero();
        }
        
        // Apply error feedback if enabled
        Matrix adjusted_grad = grad;
        if (config_.enable_error_feedback) {
            adjusted_grad = apply_error_feedback(grad, layer_idx);
        }
        
        // Compress the gradient
        LayerCompressionResult layer_result = compress_single_layer(
            adjusted_grad, layer_idx, current_ratio);
        
        result.layer_metadata.push_back(layer_result.metadata);
        
        // Append compressed data
        size_t current_size = result.compressed_data.size();
        result.compressed_data.resize(current_size + layer_result.compressed_data.size());
        std::copy(layer_result.compressed_data.begin(), 
                 layer_result.compressed_data.end(),
                 result.compressed_data.begin() + current_size);
        
        // Update error buffer if using error feedback
        if (config_.enable_error_feedback) {
            update_error_buffer(adjusted_grad, layer_result.reconstructed, layer_idx);
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto compression_time = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time).count();
    
    result.compressed_size = result.compressed_data.size();
    result.compression_ratio = static_cast<float>(result.original_size) / result.compressed_size;
    result.compression_time_us = compression_time;
    
    // Update compression history for adaptation
    update_compression_history(result, network_conditions);
    
    std::cout << "Compressed " << gradients.size() << " layers: "
              << result.original_size << " -> " << result.compressed_size 
              << " bytes (ratio: " << result.compression_ratio << "x, "
              << "time: " << compression_time << "μs)" << std::endl;
    
    return result;
}

std::vector<Matrix> AdaptiveCompressor::decompress_gradients(
    const CompressedGradients& compressed) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<Matrix> gradients;
    gradients.reserve(compressed.layer_metadata.size());
    
    size_t data_offset = 0;
    
    for (const auto& metadata : compressed.layer_metadata) {
        Matrix grad = decompress_single_layer(compressed.compressed_data, data_offset, metadata);
        gradients.push_back(std::move(grad));
        data_offset += metadata.compressed_size;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto decompression_time = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time).count();
    
    std::cout << "Decompressed " << gradients.size() << " layers in " 
              << decompression_time << "μs" << std::endl;
    
    return gradients;
}

LayerCompressionResult AdaptiveCompressor::compress_single_layer(
    const Matrix& gradient, size_t layer_idx, float compression_ratio) {
    
    LayerCompressionResult result;
    result.metadata.layer_index = layer_idx;
    result.metadata.original_rows = gradient.rows();
    result.metadata.original_cols = gradient.cols();
    result.metadata.compression_type = CompressionType::TOP_K_SPARSIFICATION;
    
    // Calculate number of elements to keep based on sparsity ratio
    size_t total_elements = gradient.size();
    size_t k = static_cast<size_t>(total_elements * config_.sparsity_ratio / compression_ratio);
    k = std::max(k, static_cast<size_t>(1)); // Keep at least 1 element
    k = std::min(k, total_elements); // Don't exceed total elements
    
    // Create magnitude-index pairs for Top-K selection
    std::vector<std::pair<float, size_t>> magnitude_indices;
    magnitude_indices.reserve(total_elements);
    
    for (size_t i = 0; i < total_elements; ++i) {
        float value = gradient.data()[i];
        magnitude_indices.emplace_back(std::abs(value), i);
    }
    
    // Partial sort to get Top-K elements
    std::nth_element(magnitude_indices.begin(), 
                    magnitude_indices.begin() + k,
                    magnitude_indices.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Sort the Top-K elements by index for better compression
    std::sort(magnitude_indices.begin(), magnitude_indices.begin() + k,
             [](const auto& a, const auto& b) { return a.second < b.second; });
    
    // Quantize the selected values
    std::vector<float> selected_values;
    std::vector<uint32_t> selected_indices;
    selected_values.reserve(k);
    selected_indices.reserve(k);
    
    for (size_t i = 0; i < k; ++i) {
        size_t idx = magnitude_indices[i].second;
        selected_values.push_back(gradient.data()[idx]);
        selected_indices.push_back(static_cast<uint32_t>(idx));
    }
    
    // Quantize values
    QuantizationResult quant_result = quantize_values(selected_values);
    
    // Store metadata
    result.metadata.num_selected = k;
    result.metadata.quantization_scale = quant_result.scale;
    result.metadata.quantization_zero_point = quant_result.zero_point;
    result.metadata.quantization_bits = config_.quantization_bits;
    
    // Serialize compressed data
    serialize_compressed_layer(result.compressed_data, selected_indices, 
                              quant_result.quantized_values, result.metadata);
    
    result.metadata.compressed_size = result.compressed_data.size();
    
    // Create reconstructed gradient for error feedback
    result.reconstructed = Matrix(gradient.rows(), gradient.cols());
    result.reconstructed.zero();
    
    for (size_t i = 0; i < k; ++i) {
        size_t idx = selected_indices[i];
        float dequantized = dequantize_value(quant_result.quantized_values[i], 
                                           quant_result.scale, quant_result.zero_point);
        result.reconstructed.data()[idx] = dequantized;
    }
    
    return result;
}

Matrix AdaptiveCompressor::decompress_single_layer(
    const std::vector<uint8_t>& compressed_data, size_t offset,
    const LayerMetadata& metadata) {
    
    Matrix gradient(metadata.original_rows, metadata.original_cols);
    gradient.zero();
    
    // Deserialize compressed data
    std::vector<uint32_t> indices;
    std::vector<uint8_t> quantized_values;
    deserialize_compressed_layer(compressed_data, offset, metadata, indices, quantized_values);
    
    // Dequantize and restore values
    for (size_t i = 0; i < indices.size(); ++i) {
        float value = dequantize_value(quantized_values[i], 
                                     metadata.quantization_scale,
                                     metadata.quantization_zero_point);
        gradient.data()[indices[i]] = value;
    }
    
    return gradient;
}

Matrix AdaptiveCompressor::apply_error_feedback(const Matrix& gradient, size_t layer_idx) {
    Matrix& error_buffer = error_buffers_[layer_idx];
    
    // Add accumulated error to current gradient
    Matrix adjusted_gradient(gradient.rows(), gradient.cols());
    
    #pragma omp parallel for
    for (size_t i = 0; i < gradient.size(); ++i) {
        adjusted_gradient.data()[i] = gradient.data()[i] + 
                                    config_.error_feedback_momentum * error_buffer.data()[i];
    }
    
    return adjusted_gradient;
}

void AdaptiveCompressor::update_error_buffer(const Matrix& original, 
                                           const Matrix& reconstructed, 
                                           size_t layer_idx) {
    Matrix& error_buffer = error_buffers_[layer_idx];
    
    // Update error buffer with compression error
    #pragma omp parallel for
    for (size_t i = 0; i < original.size(); ++i) {
        float error = original.data()[i] - reconstructed.data()[i];
        error_buffer.data()[i] = config_.error_feedback_momentum * error_buffer.data()[i] + error;
    }
}

float AdaptiveCompressor::adapt_compression_ratio(const NetworkConditions& conditions) {
    if (!config_.enable_adaptive_ratios) {
        return config_.initial_compression_ratio;
    }
    
    float base_ratio = config_.initial_compression_ratio;
    
    // Adapt based on bandwidth (lower bandwidth = higher compression)
    float bandwidth_factor = std::max(0.1f, std::min(2.0f, 
        100.0f / std::max(1.0f, conditions.bandwidth_mbps)));
    
    // Adapt based on latency (higher latency = higher compression to reduce round trips)
    float latency_factor = std::max(0.5f, std::min(3.0f, 
        conditions.latency_ms / 50.0f));
    
    // Adapt based on peer count (more peers = higher compression to reduce total traffic)
    float peer_factor = std::max(1.0f, std::min(2.0f, 
        static_cast<float>(conditions.num_peers) / 4.0f));
    
    float adapted_ratio = base_ratio * bandwidth_factor * latency_factor * peer_factor;
    
    // Clamp to reasonable bounds
    adapted_ratio = std::max(1.1f, std::min(50.0f, adapted_ratio));
    
    return adapted_ratio;
}

QuantizationResult AdaptiveCompressor::quantize_values(const std::vector<float>& values) {
    if (values.empty()) {
        return {std::vector<uint8_t>(), 1.0f, 0};
    }
    
    // Find min/max for quantization range
    auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
    float min_val = *min_it;
    float max_val = *max_it;
    
    // Handle edge case where all values are the same
    if (std::abs(max_val - min_val) < 1e-8f) {
        std::vector<uint8_t> quantized(values.size(), 0);
        return {quantized, 1.0f, static_cast<uint8_t>(min_val)};
    }
    
    // Calculate quantization parameters
    uint32_t num_levels = (1u << config_.quantization_bits) - 1;
    float scale = (max_val - min_val) / num_levels;
    uint8_t zero_point = 0;
    
    // Quantize values
    std::vector<uint8_t> quantized;
    quantized.reserve(values.size());
    
    for (float value : values) {
        uint32_t quantized_val = static_cast<uint32_t>(
            std::round((value - min_val) / scale));
        quantized_val = std::min(quantized_val, num_levels);
        quantized.push_back(static_cast<uint8_t>(quantized_val));
    }
    
    return {quantized, scale, zero_point, min_val};
}

float AdaptiveCompressor::dequantize_value(uint8_t quantized_val, float scale, 
                                         uint8_t zero_point, float min_val) {
    return min_val + static_cast<float>(quantized_val) * scale;
}

void AdaptiveCompressor::serialize_compressed_layer(
    std::vector<uint8_t>& output,
    const std::vector<uint32_t>& indices,
    const std::vector<uint8_t>& quantized_values,
    const LayerMetadata& metadata) {
    
    // Reserve space (rough estimate)
    size_t estimated_size = sizeof(uint32_t) * indices.size() + quantized_values.size() + 64;
    output.reserve(estimated_size);
    
    // Serialize indices using delta encoding for better compression
    uint32_t prev_index = 0;
    for (uint32_t index : indices) {
        uint32_t delta = index - prev_index;
        
        // Variable-length encoding for deltas
        while (delta >= 128) {
            output.push_back(static_cast<uint8_t>((delta & 0x7F) | 0x80));
            delta >>= 7;
        }
        output.push_back(static_cast<uint8_t>(delta));
        prev_index = index;
    }
    
    // Add separator
    output.push_back(0xFF);
    
    // Serialize quantized values
    output.insert(output.end(), quantized_values.begin(), quantized_values.end());
}

void AdaptiveCompressor::deserialize_compressed_layer(
    const std::vector<uint8_t>& compressed_data, size_t offset,
    const LayerMetadata& metadata,
    std::vector<uint32_t>& indices,
    std::vector<uint8_t>& quantized_values) {
    
    indices.clear();
    quantized_values.clear();
    indices.reserve(metadata.num_selected);
    quantized_values.reserve(metadata.num_selected);
    
    size_t pos = offset;
    uint32_t current_index = 0;
    
    // Deserialize indices with delta decoding
    while (pos < compressed_data.size() && compressed_data[pos] != 0xFF) {
        uint32_t delta = 0;
        uint32_t shift = 0;
        
        while (pos < compressed_data.size()) {
            uint8_t byte = compressed_data[pos++];
            delta |= static_cast<uint32_t>(byte & 0x7F) << shift;
            
            if ((byte & 0x80) == 0) break;
            shift += 7;
        }
        
        current_index += delta;
        indices.push_back(current_index);
    }
    
    // Skip separator
    if (pos < compressed_data.size() && compressed_data[pos] == 0xFF) {
        pos++;
    }
    
    // Deserialize quantized values
    size_t values_size = std::min(metadata.num_selected, 
                                 compressed_data.size() - pos);
    quantized_values.assign(compressed_data.begin() + pos,
                           compressed_data.begin() + pos + values_size);
}

size_t AdaptiveCompressor::calculate_total_size(const std::vector<Matrix>& gradients) {
    size_t total = 0;
    for (const auto& grad : gradients) {
        total += grad.size() * sizeof(float);
    }
    return total;
}

void AdaptiveCompressor::update_compression_history(
    const CompressedGradients& result,
    const NetworkConditions& conditions) {
    
    CompressionStats stats;
    stats.timestamp = std::chrono::steady_clock::now();
    stats.compression_ratio = result.compression_ratio;
    stats.compression_time_us = result.compression_time_us;
    stats.network_conditions = conditions;
    
    compression_history_.push_back(stats);
    
    // Keep only recent history
    if (compression_history_.size() > 1000) {
        compression_history_.erase(compression_history_.begin(), 
                                 compression_history_.begin() + 100);
    }
}

CompressionStats AdaptiveCompressor::get_compression_stats() const {
    if (compression_history_.empty()) {
        return CompressionStats{};
    }
    
    // Calculate average stats from recent history
    size_t recent_count = std::min(compression_history_.size(), static_cast<size_t>(100));
    auto recent_start = compression_history_.end() - recent_count;
    
    float avg_ratio = 0.0f;
    float avg_time = 0.0f;
    
    for (auto it = recent_start; it != compression_history_.end(); ++it) {
        avg_ratio += it->compression_ratio;
        avg_time += it->compression_time_us;
    }
    
    CompressionStats stats = compression_history_.back();
    stats.compression_ratio = avg_ratio / recent_count;
    stats.compression_time_us = static_cast<uint64_t>(avg_time / recent_count);
    
    return stats;
}

} // namespace compression
