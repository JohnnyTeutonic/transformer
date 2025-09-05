#include "../include/gradient_compression.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <iostream>
#include <omp.h>

GradientCompressor::GradientCompressor(float sparsity_ratio, int quantization_bits)
    : sparsity_ratio_(sparsity_ratio), quantization_bits_(quantization_bits) {
    
    // Calculate quantization levels
    quantization_levels_ = (1 << quantization_bits_) - 1;
    
    std::cout << "GradientCompressor initialized:" << std::endl;
    std::cout << "- Sparsity ratio: " << sparsity_ratio_ << std::endl;
    std::cout << "- Quantization bits: " << quantization_bits_ << std::endl;
    std::cout << "- Quantization levels: " << quantization_levels_ << std::endl;
}

CompressedGradients GradientCompressor::compress(const std::vector<Matrix>& gradients) {
    CompressedGradients compressed;
    compressed.original_shapes.reserve(gradients.size());
    compressed.compressed_data.reserve(gradients.size());
    compressed.metadata.reserve(gradients.size());
    
    size_t total_original_size = 0;
    size_t total_compressed_size = 0;
    
    for (const auto& gradient : gradients) {
        total_original_size += gradient.size();
        
        // Store original shape
        compressed.original_shapes.push_back({gradient.rows(), gradient.cols()});
        
        // Compress this gradient matrix
        auto compressed_matrix = compress_matrix(gradient);
        compressed.compressed_data.push_back(compressed_matrix.data);
        compressed.metadata.push_back(compressed_matrix.metadata);
        
        total_compressed_size += compressed_matrix.data.size();
    }
    
    // Calculate compression ratio
    compressed.compression_ratio = static_cast<float>(total_original_size) / total_compressed_size;
    
    std::cout << "Gradient compression complete:" << std::endl;
    std::cout << "- Original size: " << total_original_size * sizeof(float) << " bytes" << std::endl;
    std::cout << "- Compressed size: " << total_compressed_size << " bytes" << std::endl;
    std::cout << "- Compression ratio: " << compressed.compression_ratio << "x" << std::endl;
    
    return compressed;
}

std::vector<Matrix> GradientCompressor::decompress(const CompressedGradients& compressed) {
    std::vector<Matrix> gradients;
    gradients.reserve(compressed.original_shapes.size());
    
    for (size_t i = 0; i < compressed.original_shapes.size(); ++i) {
        const auto& shape = compressed.original_shapes[i];
        const auto& data = compressed.compressed_data[i];
        const auto& metadata = compressed.metadata[i];
        
        Matrix gradient = decompress_matrix(data, metadata, shape.first, shape.second);
        gradients.push_back(std::move(gradient));
    }
    
    return gradients;
}

GradientCompressor::CompressedMatrix GradientCompressor::compress_matrix(const Matrix& matrix) {
    CompressedMatrix result;
    
    // Step 1: Top-K sparsification
    std::vector<std::pair<float, size_t>> magnitude_indices;
    magnitude_indices.reserve(matrix.size());
    
    for (size_t i = 0; i < matrix.size(); ++i) {
        magnitude_indices.emplace_back(std::abs(matrix.data()[i]), i);
    }
    
    // Sort by magnitude (descending)
    std::sort(magnitude_indices.begin(), magnitude_indices.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Keep top-k elements
    size_t k = static_cast<size_t>((1.0f - sparsity_ratio_) * matrix.size());
    k = std::max(k, size_t(1)); // Keep at least one element
    
    // Store metadata
    result.metadata.scale_factor = 0.0f;
    result.metadata.zero_point = 0.0f;
    result.metadata.num_nonzero = k;
    
    // Find min/max of top-k elements for quantization
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    
    for (size_t i = 0; i < k; ++i) {
        float val = matrix.data()[magnitude_indices[i].second];
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
    }
    
    // Calculate quantization parameters
    if (max_val != min_val) {
        result.metadata.scale_factor = (max_val - min_val) / quantization_levels_;
        result.metadata.zero_point = min_val;
    } else {
        result.metadata.scale_factor = 1.0f;
        result.metadata.zero_point = min_val;
    }
    
    // Step 2: Quantize and store top-k elements
    result.data.reserve(k * (sizeof(uint32_t) + 1)); // index + quantized value
    
    for (size_t i = 0; i < k; ++i) {
        size_t index = magnitude_indices[i].second;
        float value = matrix.data()[index];
        
        // Quantize value
        uint8_t quantized_value;
        if (result.metadata.scale_factor > 0) {
            float normalized = (value - result.metadata.zero_point) / result.metadata.scale_factor;
            quantized_value = static_cast<uint8_t>(std::clamp(normalized, 0.0f, static_cast<float>(quantization_levels_)));
        } else {
            quantized_value = 0;
        }
        
        // Store index (4 bytes) + quantized value (1 byte)
        uint32_t idx32 = static_cast<uint32_t>(index);
        result.data.insert(result.data.end(), 
                          reinterpret_cast<uint8_t*>(&idx32),
                          reinterpret_cast<uint8_t*>(&idx32) + sizeof(uint32_t));
        result.data.push_back(quantized_value);
    }
    
    return result;
}

Matrix GradientCompressor::decompress_matrix(const std::vector<uint8_t>& data,
                                           const CompressionMetadata& metadata,
                                           size_t rows, size_t cols) {
    Matrix result(rows, cols, 0.0f); // Initialize with zeros
    
    // Decompress sparse quantized data
    size_t pos = 0;
    for (size_t i = 0; i < metadata.num_nonzero && pos + 4 < data.size(); ++i) {
        // Read index (4 bytes)
        uint32_t index;
        std::memcpy(&index, &data[pos], sizeof(uint32_t));
        pos += sizeof(uint32_t);
        
        // Read quantized value (1 byte)
        uint8_t quantized_value = data[pos];
        pos += 1;
        
        // Dequantize
        float value = metadata.zero_point + quantized_value * metadata.scale_factor;
        
        // Store in result matrix
        if (index < result.size()) {
            result.data()[index] = value;
        }
    }
    
    return result;
}

float GradientCompressor::estimate_compression_ratio(const std::vector<Matrix>& gradients) const {
    size_t total_original_size = 0;
    size_t total_compressed_size = 0;
    
    for (const auto& gradient : gradients) {
        total_original_size += gradient.size() * sizeof(float);
        
        // Estimate compressed size
        size_t k = static_cast<size_t>((1.0f - sparsity_ratio_) * gradient.size());
        k = std::max(k, size_t(1));
        
        // Each non-zero element: 4 bytes (index) + 1 byte (quantized value)
        size_t compressed_size = k * 5 + sizeof(CompressionMetadata);
        total_compressed_size += compressed_size;
    }
    
    return static_cast<float>(total_original_size) / total_compressed_size;
}

void GradientCompressor::set_sparsity_ratio(float ratio) {
    sparsity_ratio_ = std::clamp(ratio, 0.0f, 0.99f);
    std::cout << "Updated sparsity ratio to: " << sparsity_ratio_ << std::endl;
}

void GradientCompressor::set_quantization_bits(int bits) {
    quantization_bits_ = std::clamp(bits, 1, 8);
    quantization_levels_ = (1 << quantization_bits_) - 1;
    std::cout << "Updated quantization bits to: " << quantization_bits_ 
              << " (levels: " << quantization_levels_ << ")" << std::endl;
}
