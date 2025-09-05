#pragma once

#include "matrix.hpp"
#include <vector>
#include <cstdint>

/**
 * High-performance gradient compression for P2P distributed training.
 * Combines top-k sparsification with quantization to achieve 90%+ bandwidth reduction
 * while maintaining training accuracy.
 */
class GradientCompressor {
public:
    struct CompressionMetadata {
        float scale_factor;     // Quantization scale
        float zero_point;       // Quantization zero point
        size_t num_nonzero;     // Number of non-zero elements
    };
    
    struct CompressedMatrix {
        std::vector<uint8_t> data;      // Compressed data (indices + quantized values)
        CompressionMetadata metadata;   // Compression parameters
    };
    
    struct CompressedGradients {
        std::vector<std::pair<size_t, size_t>> original_shapes;  // Original matrix dimensions
        std::vector<std::vector<uint8_t>> compressed_data;       // Compressed data for each matrix
        std::vector<CompressionMetadata> metadata;               // Metadata for each matrix
        float compression_ratio;                                 // Achieved compression ratio
    };
    
    /**
     * Constructor
     * @param sparsity_ratio Fraction of gradients to zero out (0.0 = no sparsity, 0.9 = 90% sparse)
     * @param quantization_bits Number of bits for quantization (1-8)
     */
    explicit GradientCompressor(float sparsity_ratio = 0.9f, int quantization_bits = 8);
    
    /**
     * Compress a vector of gradient matrices
     * @param gradients Vector of gradient matrices to compress
     * @return Compressed gradients with metadata
     */
    CompressedGradients compress(const std::vector<Matrix>& gradients);
    
    /**
     * Decompress gradients back to original format
     * @param compressed Compressed gradients
     * @return Vector of decompressed gradient matrices
     */
    std::vector<Matrix> decompress(const CompressedGradients& compressed);
    
    /**
     * Estimate compression ratio without actually compressing
     * @param gradients Gradients to estimate compression for
     * @return Estimated compression ratio
     */
    float estimate_compression_ratio(const std::vector<Matrix>& gradients) const;
    
    /**
     * Update sparsity ratio dynamically
     * @param ratio New sparsity ratio (0.0 - 0.99)
     */
    void set_sparsity_ratio(float ratio);
    
    /**
     * Update quantization bits dynamically
     * @param bits New quantization bits (1-8)
     */
    void set_quantization_bits(int bits);
    
    // Getters
    float get_sparsity_ratio() const { return sparsity_ratio_; }
    int get_quantization_bits() const { return quantization_bits_; }

private:
    float sparsity_ratio_;      // Fraction of gradients to zero out
    int quantization_bits_;     // Number of bits for quantization
    int quantization_levels_;   // Number of quantization levels
    
    /**
     * Compress a single matrix using top-k sparsification + quantization
     * @param matrix Matrix to compress
     * @return Compressed matrix data
     */
    CompressedMatrix compress_matrix(const Matrix& matrix);
    
    /**
     * Decompress a single matrix
     * @param data Compressed data
     * @param metadata Compression metadata
     * @param rows Original number of rows
     * @param cols Original number of columns
     * @return Decompressed matrix
     */
    Matrix decompress_matrix(const std::vector<uint8_t>& data,
                           const CompressionMetadata& metadata,
                           size_t rows, size_t cols);
};
