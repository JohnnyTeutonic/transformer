#pragma once

#include <vector>
#include <memory>
#include <unordered_map>
#include <torch/torch.h>
#include <cuda_runtime.h>
#include <nccl.h>

namespace compression {

// Compression configuration
struct CompressionConfig {
    enum CompressionType {
        TOP_K,
        RANDOM_K,
        THRESHOLD,
        QUANTIZATION,
        ADAPTIVE,
        HYBRID
    };
    
    CompressionType type = TOP_K;
    float compression_ratio = 0.01f;  // Keep 1% of gradients
    float threshold = 1e-3f;
    int quantization_bits = 8;
    bool use_error_feedback = true;
    bool use_momentum = true;
    float momentum_beta = 0.9f;
    bool use_warmup = true;
    int warmup_steps = 1000;
};

// Compressed gradient
struct CompressedGradient {
    torch::Tensor indices;  // Indices of non-zero elements
    torch::Tensor values;   // Values of non-zero elements
    torch::Size original_shape;
    float compression_ratio;
    std::string compression_method;
};

// Error feedback compression
class ErrorFeedbackCompressor {
public:
    ErrorFeedbackCompressor(CompressionConfig config);
    ~ErrorFeedbackCompressor();
    
    // Compression methods
    CompressedGradient compress(const torch::Tensor& gradient);
    torch::Tensor decompress(const CompressedGradient& compressed);
    
    // Error feedback
    void update_error_buffer(const std::string& param_name,
                            const torch::Tensor& error);
    torch::Tensor get_error_corrected_gradient(const std::string& param_name,
                                              const torch::Tensor& gradient);
    
    // Statistics
    struct CompressionStats {
        float average_compression_ratio;
        float average_reconstruction_error;
        size_t bytes_saved;
        float speedup_factor;
        std::unordered_map<std::string, float> layer_compression_ratios;
    };
    CompressionStats get_stats() const { return stats_; }
    
private:
    CompressionConfig config_;
    CompressionStats stats_;
    
    // Error feedback buffers
    std::unordered_map<std::string, torch::Tensor> error_buffers_;
    std::unordered_map<std::string, torch::Tensor> momentum_buffers_;
    
    // Compression implementations
    CompressedGradient top_k_compression(const torch::Tensor& gradient);
    CompressedGradient random_k_compression(const torch::Tensor& gradient);
    CompressedGradient threshold_compression(const torch::Tensor& gradient);
    CompressedGradient quantization_compression(const torch::Tensor& gradient);
    CompressedGradient adaptive_compression(const torch::Tensor& gradient);
    
    // Helper functions
    torch::Tensor apply_momentum(const std::string& param_name,
                                const torch::Tensor& gradient);
    float compute_adaptive_threshold(const torch::Tensor& gradient);
    int compute_adaptive_k(const torch::Tensor& gradient);
};

// Distributed gradient compression
class DistributedGradientCompressor {
public:
    DistributedGradientCompressor(CompressionConfig config,
                                 int world_size,
                                 int rank);
    
    // All-reduce with compression
    torch::Tensor compressed_allreduce(const torch::Tensor& gradient);
    
    // Asynchronous compression
    void async_compress_and_send(const torch::Tensor& gradient,
                                const std::string& param_name);
    torch::Tensor async_receive_and_decompress(const std::string& param_name);
    
    // Layer-wise adaptive compression
    void set_layer_compression_ratio(const std::string& layer_name,
                                    float compression_ratio);
    
    // Communication optimization
    void optimize_communication_schedule(const std::vector<std::string>& param_names);
    
private:
    CompressionConfig config_;
    int world_size_;
    int rank_;
    ncclComm_t nccl_comm_;
    
    std::unique_ptr<ErrorFeedbackCompressor> compressor_;
    
    // Asynchronous buffers
    std::unordered_map<std::string, CompressedGradient> send_buffers_;
    std::unordered_map<std::string, CompressedGradient> recv_buffers_;
    std::unordered_map<std::string, cudaStream_t> param_streams_;
    
    // Communication schedule
    std::vector<std::vector<std::string>> communication_rounds_;
    
    void schedule_round_robin(const std::vector<std::string>& param_names);
    void schedule_priority_based(const std::vector<std::string>& param_names);
};

// Gradient accumulation with compression
class CompressedGradientAccumulator {
public:
    CompressedGradientAccumulator(int accumulation_steps);
    
    // Accumulate compressed gradients
    void accumulate(const std::string& param_name,
                   const CompressedGradient& compressed);
    
    // Get accumulated gradient
    torch::Tensor get_accumulated(const std::string& param_name);
    
    // Reset accumulator
    void reset();
    void reset_param(const std::string& param_name);
    
    // Adaptive accumulation
    void set_adaptive_steps(const std::string& param_name, int steps);
    
private:
    int accumulation_steps_;
    std::unordered_map<std::string, std::vector<CompressedGradient>> accumulated_;
    std::unordered_map<std::string, int> accumulation_counters_;
    std::unordered_map<std::string, int> adaptive_steps_;
    
    torch::Tensor merge_compressed_gradients(const std::vector<CompressedGradient>& compressed);
};

// PowerSGD compression
class PowerSGDCompressor {
public:
    PowerSGDCompressor(int rank = 2, bool use_error_feedback = true);
    
    // PowerSGD compression
    CompressedGradient compress_powersgd(const torch::Tensor& gradient);
    torch::Tensor decompress_powersgd(const CompressedGradient& compressed);
    
    // Warm-up phase
    void set_warmup_steps(int steps) { warmup_steps_ = steps; }
    bool is_warmup_complete() const { return current_step_ >= warmup_steps_; }
    
private:
    int rank_;
    bool use_error_feedback_;
    int warmup_steps_ = 1000;
    int current_step_ = 0;
    
    // Power iteration matrices
    std::unordered_map<torch::Size, torch::Tensor> P_matrices_;
    std::unordered_map<torch::Size, torch::Tensor> Q_matrices_;
    
    // Error feedback
    std::unordered_map<torch::Size, torch::Tensor> error_matrices_;
    
    void power_iteration(torch::Tensor& P, torch::Tensor& Q,
                        const torch::Tensor& matrix, int num_iters = 2);
};

// 1-bit quantization (SignSGD)
class OneBitCompressor {
public:
    OneBitCompressor(bool use_momentum = true, float momentum_beta = 0.9f);
    
    // 1-bit compression
    struct OneBitGradient {
        torch::Tensor signs;  // Sign bits
        float scale;          // Magnitude scale
        torch::Size shape;
    };
    
    OneBitGradient compress_1bit(const torch::Tensor& gradient);
    torch::Tensor decompress_1bit(const OneBitGradient& compressed);
    
    // Momentum correction
    torch::Tensor apply_momentum_correction(const std::string& param_name,
                                          const torch::Tensor& gradient);
    
private:
    bool use_momentum_;
    float momentum_beta_;
    
    std::unordered_map<std::string, torch::Tensor> momentum_buffers_;
};

// Adaptive quantization
class AdaptiveQuantizer {
public:
    AdaptiveQuantizer();
    
    // Adaptive bit allocation
    CompressedGradient quantize_adaptive(const torch::Tensor& gradient,
                                        float target_mse = 1e-6f);
    
    // Layer-wise bit allocation
    void allocate_bits_by_sensitivity(const std::unordered_map<std::string, float>& sensitivities);
    
    // Dynamic range adjustment
    void update_quantization_range(const torch::Tensor& gradient);
    
private:
    std::unordered_map<std::string, int> layer_bits_;
    std::unordered_map<std::string, std::pair<float, float>> quantization_ranges_;
    
    int compute_optimal_bits(const torch::Tensor& gradient, float target_mse);
    torch::Tensor quantize_uniform(const torch::Tensor& gradient, int bits);
    torch::Tensor quantize_logarithmic(const torch::Tensor& gradient, int bits);
};

} // namespace compression