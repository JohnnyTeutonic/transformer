#include "../include/quantization.hpp"
#include <omp.h>
#include <algorithm>
#include <cmath>
#ifdef USE_CUDA
#include "../include/cuda/cuda_check.cuh"
#include "../include/cuda/cuda_launch.cuh"
#include "../include/cuda/quantization_kernels.cuh"
#endif

Quantizer::Quantizer(size_t num_bits, QuantizationMode quant_mode) 
    : bits(num_bits), mode(quant_mode), scale(1.0f), zero_point(0.0f), num_channels(0) {}

float Quantizer::measure_heterogeneity(const Matrix& input) {
    // Input shape: [batch*seq, hidden_dim]
    // Compute per-channel (per hidden dimension) standard deviations
    
    size_t rows = input.rows();
    size_t cols = input.cols();  // This is the hidden_dim / num_channels
    
    std::vector<float> channel_stds(cols, 0.0f);
    
    // Compute per-channel statistics (MSVC: loop vars must be signed int)
#pragma omp parallel for
    for (int c = 0; c < static_cast<int>(cols); ++c) {
        // Compute mean
        float mean = 0.0f;
        for (size_t r = 0; r < rows; ++r) {
            mean += input(r, c);
        }
        mean /= static_cast<float>(rows);
        
        // Compute variance
        float variance = 0.0f;
        for (size_t r = 0; r < rows; ++r) {
            float diff = input(r, c) - mean;
            variance += diff * diff;
        }
        variance /= static_cast<float>(rows);
        
        // Standard deviation
        channel_stds[c] = std::sqrt(variance);
    }
    
    // Find max and median standard deviations
    float max_std = *std::max_element(channel_stds.begin(), channel_stds.end());
    
    // Compute median
    std::vector<float> sorted_stds = channel_stds;
    std::sort(sorted_stds.begin(), sorted_stds.end());
    float median_std = sorted_stds[sorted_stds.size() / 2];
    
    // Avoid division by zero
    if (median_std < 1e-8f) {
        return 1.0f;  // Homogeneous case
    }
    
    // Heterogeneity: H = max(σ_i) / median(σ_i)
    float H = max_std / median_std;
    
    return H;
}

Matrix Quantizer::quantize(const Matrix& input) {
    // Auto-select quantization mode based on heterogeneity
    if (mode == QuantizationMode::Auto) {
        last_heterogeneity = measure_heterogeneity(input);
        
        // Manifold Nyquist Criterion: H > 127 for INT8 requires per-channel
        float threshold = (1 << bits) - 1;  // 127 for 8-bit, 255 for 9-bit, etc.
        
        if (last_heterogeneity > threshold) {
            return quantize_per_channel(input);
        }
        // else: fall through to per-tensor
    } else if (mode == QuantizationMode::PerChannel) {
        return quantize_per_channel(input);
    }
    
    // Per-tensor quantization (original implementation)
    // Find min and max values using OpenMP reduction
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();

#pragma omp parallel
    {
        float local_min = std::numeric_limits<float>::max();
        float local_max = std::numeric_limits<float>::lowest();

// MSVC: loop vars must be signed int
#pragma omp for collapse(2) nowait
        for (int i = 0; i < static_cast<int>(input.rows()); ++i) {
            for (int j = 0; j < static_cast<int>(input.cols()); ++j) {
                local_min = std::min(local_min, input(i, j));
                local_max = std::max(local_max, input(i, j));
            }
        }

#pragma omp critical
        {
            min_val = std::min(min_val, local_min);
            max_val = std::max(max_val, local_max);
        }
    }

    // Calculate scale and zero point
    float range = max_val - min_val;
    scale = range / ((1 << bits) - 1);
    zero_point = -min_val / scale;

    // Quantize values using OpenMP (MSVC: loop vars must be signed int)
    Matrix quantized(input.rows(), input.cols());
#pragma omp parallel for collapse(2)
    for (int i = 0; i < static_cast<int>(input.rows()); ++i) {
        for (int j = 0; j < static_cast<int>(input.cols()); ++j) {
            float val = input(i, j);
            quantized(i, j) = std::round(val / scale + zero_point);
        }
    }

    return quantized;
}

Matrix Quantizer::dequantize(const Matrix& quantized) {
    Matrix result(quantized.rows(), quantized.cols());

// MSVC: loop vars must be signed int
#pragma omp parallel for collapse(2)
    for (int i = 0; i < static_cast<int>(quantized.rows()); ++i) {
        for (int j = 0; j < static_cast<int>(quantized.cols()); ++j) {
            float val = quantized(i, j);
            result(i, j) = (val - zero_point) * scale;
        }
    }

    return result;
}

void Quantizer::save(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(&bits), sizeof(bits));
    os.write(reinterpret_cast<const char*>(&scale), sizeof(scale));
    os.write(reinterpret_cast<const char*>(&zero_point), sizeof(zero_point));
}

std::unique_ptr<Quantizer> Quantizer::load(std::istream& is) {
    size_t bits;
    float scale, zero_point;

    is.read(reinterpret_cast<char*>(&bits), sizeof(bits));
    is.read(reinterpret_cast<char*>(&scale), sizeof(scale));
    is.read(reinterpret_cast<char*>(&zero_point), sizeof(zero_point));

    auto quantizer = std::make_unique<Quantizer>(bits);
    quantizer->scale = scale;
    quantizer->zero_point = zero_point;

    return quantizer;
}

Matrix Quantizer::quantize_cuda(const Matrix& input) {
#ifdef USE_CUDA
    // Find min/max using CUDA reduction kernel
    float min_val, max_val;
    find_minmax_cuda(input.data(), input.rows() * input.cols(), &min_val, &max_val);

    // Calculate scale and zero point
    float range = max_val - min_val;
    scale = range / ((1 << bits) - 1);
    zero_point = -min_val / scale;

    // Setup CUDA grid dimensions
    const int block_size = 256;
    const int grid_size = (input.rows() * input.cols() + block_size - 1) / block_size;

    // Quantize using CUDA kernel
    Matrix quantized(input.rows(), input.cols());
    CUDA_LAUNCH(quantize_kernel, grid_size, block_size, 0, 0, input.data(), quantized.data(),
                input.rows() * input.cols(), scale, zero_point);

    return quantized;
#else
    return quantize(input);
#endif
}

Matrix Quantizer::dequantize_cuda(const Matrix& quantized) {
#ifdef USE_CUDA
    // Setup CUDA grid dimensions
    const int block_size = 256;
    const int grid_size = (quantized.rows() * quantized.cols() + block_size - 1) / block_size;

    Matrix result(quantized.rows(), quantized.cols());
    CUDA_LAUNCH(dequantize_kernel, grid_size, block_size, 0, 0, quantized.data(), result.data(),
                quantized.rows() * quantized.cols(), scale, zero_point);
    return result;
#else
    return dequantize(quantized);
#endif
}

Matrix Quantizer::quantize_per_channel(const Matrix& input) {
    // Input shape: [batch*seq, hidden_dim]
    size_t rows = input.rows();
    size_t cols = input.cols();  // hidden_dim
    
    num_channels = cols;
    channel_scales.resize(num_channels);
    channel_zero_points.resize(num_channels);
    
    // Compute per-channel scales and zero points (MSVC: loop vars must be signed int)
#pragma omp parallel for
    for (int c = 0; c < static_cast<int>(cols); ++c) {
        // Find min/max for this channel (column)
        float min_val = std::numeric_limits<float>::max();
        float max_val = std::numeric_limits<float>::lowest();
        
        for (size_t r = 0; r < rows; ++r) {
            float val = input(r, c);
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }
        
        // Calculate per-channel scale and zero point
        float range = max_val - min_val;
        if (range < 1e-8f) {
            // Constant channel - avoid division by zero
            channel_scales[c] = 1.0f;
            channel_zero_points[c] = 0.0f;
        } else {
            channel_scales[c] = range / ((1 << bits) - 1);
            channel_zero_points[c] = -min_val / channel_scales[c];
        }
    }
    
    // Quantize using per-channel scales (MSVC: loop vars must be signed int)
    Matrix quantized(rows, cols);
#pragma omp parallel for collapse(2)
    for (int r = 0; r < static_cast<int>(rows); ++r) {
        for (int c = 0; c < static_cast<int>(cols); ++c) {
            float val = input(r, c);
            quantized(r, c) = std::round(val / channel_scales[c] + channel_zero_points[c]);
        }
    }
    
    return quantized;
}

Matrix Quantizer::dequantize_per_channel(const Matrix& quantized) {
    if (channel_scales.empty() || channel_zero_points.empty()) {
        throw std::runtime_error("Per-channel dequantization called without calibration");
    }
    
    size_t rows = quantized.rows();
    size_t cols = quantized.cols();
    
    Matrix result(rows, cols);
// MSVC: loop vars must be signed int
#pragma omp parallel for collapse(2)
    for (int r = 0; r < static_cast<int>(rows); ++r) {
        for (int c = 0; c < static_cast<int>(cols); ++c) {
            float val = quantized(r, c);
            result(r, c) = (val - channel_zero_points[c]) * channel_scales[c];
        }
    }
    
    return result;
}

Matrix Quantizer::quantize_per_channel_cuda(const Matrix& input) {
#ifdef USE_CUDA
    // Input shape: [batch*seq, hidden_dim]
    size_t rows = input.rows();
    size_t cols = input.cols();  // hidden_dim
    
    num_channels = cols;
    channel_scales.resize(num_channels);
    channel_zero_points.resize(num_channels);
    
    // Call CUDA kernel to compute per-channel min/max and scales ON GPU
    // symmetric=false for now (asymmetric quantization)
    // TODO: Add config option to enable symmetric (better for sampling theory)
    compute_per_channel_scales_cuda(input.data(), rows, cols, bits,
                                   channel_scales.data(), channel_zero_points.data(),
                                   false);  // symmetric
    
    // Quantize using CUDA kernel with grid-stride loop
    Matrix quantized(rows, cols);
    const int block_size = 256;
    // Use reasonable grid size for multi-SM utilization (not necessarily total/block)
    const int grid_size = std::min(static_cast<int>((rows * cols + block_size - 1) / block_size), 2048);
    
    CUDA_LAUNCH(quantize_per_channel_kernel, grid_size, block_size, 0, 0,
                input.data(), quantized.data(), rows, cols,
                channel_scales.data(), channel_zero_points.data());
    
    return quantized;
#else
    return quantize_per_channel(input);
#endif
}

Matrix Quantizer::dequantize_per_channel_cuda(const Matrix& quantized) {
#ifdef USE_CUDA
    if (channel_scales.empty() || channel_zero_points.empty()) {
        throw std::runtime_error("Per-channel dequantization called without calibration");
    }
    
    size_t rows = quantized.rows();
    size_t cols = quantized.cols();
    
    Matrix result(rows, cols);
    const int block_size = 256;
    // Use reasonable grid size for multi-SM utilization (grid-stride handles the rest)
    const int grid_size = std::min(static_cast<int>((rows * cols + block_size - 1) / block_size), 2048);
    
    CUDA_LAUNCH(dequantize_per_channel_kernel, grid_size, block_size, 0, 0,
                quantized.data(), result.data(), rows, cols,
                channel_scales.data(), channel_zero_points.data());
    
    return result;
#else
    return dequantize_per_channel(quantized);
#endif
}