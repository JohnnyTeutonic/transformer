#include "../../include/training/gradient_manager.hpp"
#include "../../include/gradient_compression.hpp"
#include <numeric>

void GradientManager::process_gradients(Matrix& gradients) {
    update_statistics(gradients);
    
    if (detect_explosion(gradients)) {
        recover_from_explosion(gradients);
        explosion_count++;
    } else {
        explosion_count = std::max(0, static_cast<int>(explosion_count) - 1);
    }
    
    clip_gradients(gradients);
}

void GradientManager::update_statistics(const Matrix& gradients) {
    float mean = 0.0f, variance = 0.0f;
    compute_running_statistics(gradients, mean, variance);
    grad_stats.update(mean, variance);
}

void GradientManager::clip_gradients(Matrix& gradients) {
    float clip_threshold = grad_stats.mean + 3 * std::sqrt(grad_stats.variance);
    clip_threshold = std::min(clip_threshold, MAX_GRAD_VALUE);
    
    for (size_t i = 0; i < gradients.size(); i++) {
        gradients.data()[i] = std::clamp(
            gradients.data()[i], 
            -clip_threshold, 
            clip_threshold
        );
    }
}

bool GradientManager::detect_explosion(const Matrix& gradients) {
    float max_abs_grad = 0.0f;
    for (size_t i = 0; i < gradients.size(); i++) {
        max_abs_grad = std::max(max_abs_grad, std::abs(gradients.data()[i]));
    }
    
    return max_abs_grad > MAX_GRAD_VALUE || 
           !std::isfinite(max_abs_grad) ||
           grad_stats.variance > MAX_GRAD_VALUE * MAX_GRAD_VALUE;
}

void GradientManager::recover_from_explosion(Matrix& gradients) {
    // Scale down gradients significantly
    float scale_factor = 0.1f;
    for (size_t i = 0; i < gradients.size(); i++) {
        gradients.data()[i] *= scale_factor;
    }
}

void GradientManager::compute_running_statistics(const Matrix& gradients, float& mean, float& variance) {
    mean = 0.0f;
    variance = 0.0f;
    size_t n = gradients.size();
    
    // Compute mean
    for (size_t i = 0; i < n; i++) {
        mean += gradients.data()[i];
    }
    mean /= n;
    
    // Compute variance
    for (size_t i = 0; i < n; i++) {
        float diff = gradients.data()[i] - mean;
        variance += diff * diff;
    }
    variance /= n;
}

// Gradient compression methods for P2P training
GradientCompressor::CompressedGradients GradientManager::compress_gradients_for_p2p(
    const std::vector<Matrix>& gradients) {
    
    if (!gradient_compressor_) {
        // Initialize compressor with adaptive settings
        gradient_compressor_ = std::make_unique<GradientCompressor>(
            0.9f,  // 90% sparsity ratio
            8      // 8-bit quantization
        );
    }
    
    return gradient_compressor_->compress(gradients);
}

std::vector<Matrix> GradientManager::decompress_gradients_from_p2p(
    const GradientCompressor::CompressedGradients& compressed) {
    
    if (!gradient_compressor_) {
        throw std::runtime_error("Gradient compressor not initialized");
    }
    
    return gradient_compressor_->decompress(compressed);
}

void GradientManager::update_compression_settings(float network_bandwidth_mbps, 
                                                 float network_latency_ms) {
    if (!gradient_compressor_) return;
    
    // Adapt compression based on network conditions
    float network_efficiency = network_bandwidth_mbps / (network_latency_ms + 1.0f);
    
    if (network_efficiency < 10.0f) {
        // Slow network: increase compression
        gradient_compressor_->set_sparsity_ratio(0.95f);
        gradient_compressor_->set_quantization_bits(4);
    } else if (network_efficiency > 100.0f) {
        // Fast network: reduce compression for better accuracy
        gradient_compressor_->set_sparsity_ratio(0.8f);
        gradient_compressor_->set_quantization_bits(8);
    } else {
        // Balanced settings
        gradient_compressor_->set_sparsity_ratio(0.9f);
        gradient_compressor_->set_quantization_bits(6);
    }
} 