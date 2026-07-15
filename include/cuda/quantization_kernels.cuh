#pragma once
#include <cuda_runtime.h>

// Per-tensor quantization (legacy)
void find_minmax_cuda(const float* input, size_t size, float* min_val, float* max_val);

__global__ void quantize_kernel(const float* input, float* output, size_t size, float scale,
                                float zero_point);

__global__ void dequantize_kernel(const float* input, float* output, size_t size, float scale,
                                  float zero_point);

// Per-channel quantization (Manifold Nyquist optimal)
// symmetric=true uses max(|min|,|max|)/127, better for sampling theory
void compute_per_channel_scales_cuda(const float* input, size_t rows, size_t cols, size_t bits,
                                     float* channel_scales, float* channel_zero_points,
                                     bool symmetric = false);

__global__ void quantize_per_channel_kernel(const float* input, float* output, 
                                           size_t rows, size_t cols,
                                           const float* channel_scales,
                                           const float* channel_zero_points);

__global__ void dequantize_per_channel_kernel(const float* input, float* output,
                                              size_t rows, size_t cols,
                                              const float* channel_scales,
                                              const float* channel_zero_points);