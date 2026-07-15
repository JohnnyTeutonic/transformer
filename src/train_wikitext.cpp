/**
 * @file train_wikitext.cpp
 * @brief Training script for transformer on WikiText dataset
 * 
 * Features:
 * - Loads WikiText-103 dataset
 * - Trains transformer with per-channel quantization
 * - Validates Manifold Nyquist Criterion
 * - Saves checkpoints
 */

#include "../include/transformer.hpp"
#include "../include/config.hpp"
#include "../include/tiktoken_tokenizer.hpp"
#include "../include/utils.hpp"
#include "../include/quantization.hpp"
#include "../include/logger.hpp"
#include "../include/gradient_diagnostics.hpp"
#include "../include/gradient_checkpoint.hpp"
#include "../include/architecture.hpp"
#include "../include/gguf_export.hpp"
#include "../include/lora.hpp"
#include "../include/safetensors_export.hpp"
#include "../include/model_saver.hpp"
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include "../include/cuda/matrix_ops.cuh"
#include "../include/cuda/loss_kernels.cuh"
#endif
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <limits>
#include <cstdlib>

#ifdef USE_CUDA
/**
 * @brief Print CUDA device information at startup
 */
void print_cuda_info() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    
    if (device_count == 0) {
        std::cerr << "WARNING: No CUDA devices found!" << std::endl;
        return;
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "CUDA Device Information" << std::endl;
    std::cout << "========================================" << std::endl;
    
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        std::cout << "Device " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total Memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
        std::cout << "  SM Count: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Max Threads/Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Memory Clock: " << (prop.memoryClockRate / 1000) << " MHz" << std::endl;
        std::cout << "  Memory Bus Width: " << prop.memoryBusWidth << "-bit" << std::endl;
        
        // Calculate theoretical bandwidth
        float bandwidth_gb = 2.0f * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6f;
        std::cout << "  Memory Bandwidth: ~" << std::fixed << std::setprecision(1) << bandwidth_gb << " GB/s" << std::endl;
    }
    std::cout << "========================================\n" << std::endl;
}
#endif

// Training configuration
struct TrainingConfig {
    std::string data_path = "../data/wikitext-103-txt";
    std::string output_dir = "checkpoints";
    std::string resume_checkpoint = "";  // Path to checkpoint to resume from
#ifdef USE_CUDA
    size_t batch_size = 64;  // Using fused CUDA attention kernel
    bool use_fp16 = true;    // GPU: use FP16 mixed precision for 2x throughput
#else
    size_t batch_size = 4;   // CPU: small batch to avoid OOM
    bool use_fp16 = false;   // CPU: FP16 not beneficial without tensor cores
#endif
    size_t max_seq_len = 128;  // Reduced for faster iteration (512 later)
    size_t num_epochs = 10;
    float learning_rate = 2e-4f;  // Tuned for batch 32
    size_t warmup_steps = 100;     // Shorter warmup to reach full LR faster
    size_t log_interval = 10;   // Log every 10 steps for progress visibility
    size_t eval_interval = 1000;
    size_t save_interval = 250;  // Frequent checkpoints after TDR fix
    bool use_quantization = true;
    std::string quant_mode = "auto";  // "per_tensor", "per_channel", or "auto"
    size_t max_steps = 0;        // 0 = unlimited; otherwise stop after this many optimizer steps
    size_t max_train_seqs = 0;   // 0 = use all; otherwise cap number of training sequences
    bool use_small = false;      // Use the truncated *-small.txt training files (fast smoke tests)
    bool tiny = false;           // Tiny model dims (128/2 layers) for round-trip smoke tests
    std::string export_gguf = ""; // After training, export the model to this GGUF path
    std::string export_safetensors = ""; // After training, export the model to this safetensors path
    std::string config_json = ""; // Optional transformer_config.json overriding model defaults
    std::string family = "llama"; // Architecture family preset (llama | vanilla)
    bool cosine_decay = false;   // Cosine LR decay to 10% over max_steps (after
                                 // warmup). Constant-LR runs destabilized late
                                 // (2026-07-13: loss 3.4 -> 5.3 in epoch 2).
};

// Warmup + optional cosine decay. Peak = config LR; floor = 10% of peak.
static float effective_lr_at(const TrainingConfig& config, size_t global_step) {
    float lr = config.learning_rate;
    if (global_step < config.warmup_steps) {
        return lr * (float(global_step + 1) / float(config.warmup_steps));
    }
    if (config.cosine_decay && config.max_steps > config.warmup_steps) {
        float prog = float(global_step - config.warmup_steps)
                     / float(config.max_steps - config.warmup_steps);
        if (prog > 1.0f) prog = 1.0f;
        lr *= 0.1f + 0.45f * (1.0f + std::cos(3.14159265f * prog));
    }
    return lr;
}

/**
 * @brief Load text file and tokenize
 */
std::vector<std::vector<int>> load_and_tokenize(
    const std::string& filepath,
    TiktokenTokenizer& tokenizer,
    size_t max_seq_len
) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }

    std::vector<std::vector<int>> batches;
    std::string line;
    std::vector<int> current_batch;
    
    std::cout << "Loading " << filepath << "..." << std::endl;
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        // Tokenize line
        auto tokens = tokenizer.encode(line);
        
        // Add to current batch
        current_batch.insert(current_batch.end(), tokens.begin(), tokens.end());
        
        // If batch is full, save it
        while (current_batch.size() >= max_seq_len) {
            std::vector<int> sequence(
                current_batch.begin(), 
                current_batch.begin() + max_seq_len
            );
            batches.push_back(sequence);
            
            // Overlap by half for better context
            current_batch.erase(
                current_batch.begin(), 
                current_batch.begin() + (max_seq_len / 2)
            );
        }
    }
    
    std::cout << "Loaded " << batches.size() << " sequences" << std::endl;
    return batches;
}

/**
 * @brief Convert token sequences to Matrix format
 */
Matrix prepare_batch(
    const std::vector<std::vector<int>>& sequences,
    size_t start_idx,
    size_t batch_size
) {
    size_t actual_batch = std::min(batch_size, sequences.size() - start_idx);
    size_t seq_len = sequences[0].size();
    
    Matrix batch(actual_batch, seq_len);
    
    for (size_t i = 0; i < actual_batch; ++i) {
        for (size_t j = 0; j < seq_len; ++j) {
            batch(i, j) = static_cast<float>(sequences[start_idx + i][j]);
        }
    }
    
    return batch;
}

/**
 * @brief Calculate perplexity from loss
 */
float calculate_perplexity(float loss) {
    return std::exp(loss);
}

/**
 * @brief Save training summary to file
 */
void save_training_summary(const std::string& summary_file, 
                          const TrainingConfig& config,
                          float final_train_loss,
                          float best_val_loss,
                          float training_time_seconds) {
    std::ofstream summary(summary_file);
    if (!summary.is_open()) {
        std::cerr << "Warning: Could not save training summary" << std::endl;
        return;
    }
    
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    
    summary << "========================================" << std::endl;
    summary << "Training Summary" << std::endl;
    summary << "========================================" << std::endl;
    summary << "Date: " << std::put_time(std::localtime(&time_t_now), "%Y-%m-%d %H:%M:%S") << std::endl;
    summary << "\nConfiguration:" << std::endl;
    summary << "  Data path: " << config.data_path << std::endl;
    summary << "  Batch size: " << config.batch_size << std::endl;
    summary << "  Max sequence length: " << config.max_seq_len << std::endl;
    summary << "  Num epochs: " << config.num_epochs << std::endl;
    summary << "  Learning rate: " << config.learning_rate << std::endl;
    summary << "  Warmup steps: " << config.warmup_steps << std::endl;
    summary << "\nResults:" << std::endl;
    summary << "  Final training loss: " << std::fixed << std::setprecision(4) << final_train_loss << std::endl;
    summary << "  Final training perplexity: " << std::setprecision(2) << calculate_perplexity(final_train_loss) << std::endl;
    summary << "  Best validation loss: " << std::setprecision(4) << best_val_loss << std::endl;
    summary << "  Best validation perplexity: " << std::setprecision(2) << calculate_perplexity(best_val_loss) << std::endl;
    summary << "  Training time: " << std::setprecision(2) << training_time_seconds << " seconds" << std::endl;
    summary << "========================================" << std::endl;
    
    summary.close();
    std::cout << "Training summary saved to: " << summary_file << std::endl;
}

/**
 * @brief Training loop
 */
struct TrainingMetrics {
    float final_train_loss;
    float best_val_loss;
    float training_time_seconds;
};

TrainingMetrics train(
    Transformer& transformer,
    const std::vector<std::vector<int>>& train_data,
    const std::vector<std::vector<int>>& val_data,
    const TrainingConfig& config,
    TiktokenTokenizer& tokenizer,
    size_t start_step = 0,
    size_t start_epoch = 0
) {
    size_t global_step = start_step;
    float best_val_loss = std::numeric_limits<float>::max();
    float final_train_loss = 0.0f;
    
    // Open training log file (separate from Logger to keep console output)
    std::ofstream train_log("training_metrics.log", std::ios::out | std::ios::trunc);
    if (train_log.is_open()) {
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        train_log << "Training started at: " << std::ctime(&time_t_now);
        train_log << "Train sequences: " << train_data.size() << std::endl;
        train_log << "Val sequences: " << val_data.size() << std::endl;
        train_log << "Batch size: " << config.batch_size << std::endl;
        train_log << "Learning rate: " << config.learning_rate << std::endl;
        train_log << "Epochs: " << config.num_epochs << std::endl;
        train_log << "========================================" << std::endl;
        train_log.flush();
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Starting Training" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Train sequences: " << train_data.size() << std::endl;
    std::cout << "Val sequences: " << val_data.size() << std::endl;
    std::cout << "Batch size: " << config.batch_size << std::endl;
    std::cout << "Sequence length: " << config.max_seq_len << std::endl;
    std::cout << "Epochs: " << config.num_epochs << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Enable gradient diagnostics for debugging training instability
    GRAD_DIAG.enable("gradient_diagnostics.log");
    std::cout << "[DIAG] Gradient diagnostics enabled - logging to gradient_diagnostics.log" << std::endl;
    
    auto start_time = std::chrono::steady_clock::now();
    bool stop_training = false;
    
    for (size_t epoch = start_epoch; epoch < config.num_epochs; ++epoch) {
        std::cout << "\n--- Epoch " << (epoch + 1) << "/" << config.num_epochs 
                  << (start_step > 0 ? " (resumed)" : "") << " ---" << std::endl;
        
        // Calculate total batches for progress tracking
        size_t total_batches = (train_data.size() + config.batch_size - 1) / config.batch_size;
        std::cout << "Total batches in epoch: " << total_batches << std::endl;
        
        float epoch_loss = 0.0f;
        size_t num_batches = 0;
        
        // Training
        transformer.set_training(true);
        
        for (size_t i = 0; i < train_data.size(); i += config.batch_size) {
            auto batch_start = std::chrono::steady_clock::now();
            
            size_t current_batch_num = (i / config.batch_size) + 1;
            
            // Collect batch sequences
            std::vector<std::vector<int>> batch_seqs;
            for (size_t j = i; j < std::min(i + config.batch_size, train_data.size()); ++j) {
                if (!train_data[j].empty() && train_data[j].size() >= 2) {
                    // Truncate to max_seq_len
                    std::vector<int> seq(train_data[j].begin(), 
                        train_data[j].begin() + std::min(train_data[j].size(), config.max_seq_len));
                    batch_seqs.push_back(seq);
                }
            }
            
            if (batch_seqs.empty()) continue;
            
            size_t actual_batch_size = batch_seqs.size();
            size_t seq_len = config.max_seq_len;
            
            // BATCHED forward pass - all sequences at once!
            // NOTE: in the CUDA build the logits are kept device-resident and
            // output.logits is intentionally empty (see LanguageModelHead::forward).
            const bool phase_timing = (std::getenv("TCPP_PHASE_TIMING") != nullptr);
            auto pt0 = std::chrono::high_resolution_clock::now();
            TransformerOutput output = transformer.forward_batch(batch_seqs, seq_len);
            auto pt1 = std::chrono::high_resolution_clock::now();
            
            // Logits shape: [batch_size * seq_len x vocab_size]
            // Derive dimensions from the model config so this works whether or
            // not the logits were materialized on the host.
            size_t vocab_size = transformer.get_output_vocab_size();
            size_t total_positions = actual_batch_size * seq_len;
            
            // Build target indices for CUDA loss computation
            std::vector<int> target_indices(total_positions, 0);
            size_t loss_count = 0;
            
            for (size_t b = 0; b < actual_batch_size; ++b) {
                const auto& tokens = batch_seqs[b];
                for (size_t t = 1; t < tokens.size(); ++t) {
                    int target_token = tokens[t];
                    size_t pos = b * seq_len + (t - 1);
                    if (static_cast<size_t>(target_token) < vocab_size && pos < total_positions) {
                        target_indices[pos] = target_token;
                        loss_count++;
                    }
                }
            }
            
            // ========== CUDA FUSED SOFTMAX + CROSS-ENTROPY + GRADIENT ==========
            float avg_loss = 0.0f;
            
            // Effective learning rate: warmup + optional cosine decay
            float effective_lr = effective_lr_at(config, global_step);

#ifdef USE_CUDA
            // CUDA path: logits are already on the device (filled by the LM head's
            // device-resident projection). Compute loss/grad directly from them,
            // keeping the gradient on the GPU for backward.
            avg_loss = cuda::compute_loss_from_device_logits(
                target_indices.data(),
                static_cast<int>(total_positions),
                static_cast<int>(vocab_size)
            );
            
            GRAD_DIAG.set_step(global_step);
            GRAD_LOG_SCALAR("loss_before_backward", avg_loss);
            GRAD_LOG_SCALAR("effective_lr", effective_lr);
            
            auto pt2 = std::chrono::high_resolution_clock::now();
            // Backward with grad_hidden computed on GPU, weight update uses Adam
            transformer.backward_from_grad_cuda(
                cuda::get_device_grad_logits(),
                static_cast<int>(total_positions),
                static_cast<int>(vocab_size),
                effective_lr
            );
            if (phase_timing) {
                auto pt3 = std::chrono::high_resolution_clock::now();
                auto ms = [](auto a, auto b) {
                    return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
                };
                std::cout << "[PHASE_TIMING] forward=" << ms(pt0, pt1)
                          << "ms loss=" << ms(pt1, pt2)
                          << "ms backward+update=" << ms(pt2, pt3) << "ms" << std::endl;
            }
#else
            // CPU fallback: compute loss and gradient
            static Matrix grad_logits;
            if (grad_logits.rows() != total_positions || grad_logits.cols() != vocab_size) {
                grad_logits = Matrix(total_positions, vocab_size);
            }
            
            float batch_loss = 0.0f;
            #pragma omp parallel for reduction(+:batch_loss)
            for (int pos = 0; pos < static_cast<int>(total_positions); ++pos) {
                int target_token = target_indices[pos];
                float max_logit = -1e9f;
                for (size_t v = 0; v < vocab_size; ++v) {
                    max_logit = std::max(max_logit, output.logits(pos, v));
                }
                float sum_exp = 0.0f;
                for (size_t v = 0; v < vocab_size; ++v) {
                    float exp_v = std::exp(output.logits(pos, v) - max_logit);
                    sum_exp += exp_v;
                }
                for (size_t v = 0; v < vocab_size; ++v) {
                    float softmax_v = std::exp(output.logits(pos, v) - max_logit) / sum_exp;
                    float target_v = (static_cast<int>(v) == target_token) ? 1.0f : 0.0f;
                    grad_logits(pos, v) = softmax_v - target_v;
                }
                float log_prob = output.logits(pos, target_token) - max_logit - std::log(sum_exp + 1e-10f);
                batch_loss += -log_prob;
            }
            avg_loss = loss_count > 0 ? batch_loss / loss_count : 0.0f;
            
            GRAD_DIAG.set_step(global_step);
            GRAD_LOG_SCALAR("loss_before_backward", avg_loss);
            GRAD_LOG_SCALAR("effective_lr", effective_lr);
            
            transformer.backward_from_grad(grad_logits, effective_lr);
#endif
            
            // Clear cache periodically
#ifdef USE_CUDA
            if (global_step % 100 == 0) {
                GradientCheckpoint::clear_cache();
            }
#endif
            
            GRAD_DIAG.flush();
            
            // NaN/Inf detection
            if (std::isnan(avg_loss) || std::isinf(avg_loss)) {
                std::cerr << "\n!!! NUMERICAL INSTABILITY at step " << global_step << " !!!" << std::endl;
                continue;
            }
            
            if (avg_loss > 20.0f) {
                std::cerr << "\n!!! WARNING: Loss explosion (loss=" << avg_loss << ") !!!" << std::endl;
            }
            
            epoch_loss += avg_loss;
            num_batches++;
            global_step++;
            
            // Early stop for smoke tests / bounded runs
            if (config.max_steps > 0 && global_step >= config.max_steps) {
                std::cout << "\n[max-steps] Reached " << config.max_steps
                          << " steps; stopping early." << std::endl;
                stop_training = true;
            }
            
            // Logging every log_interval steps
            if (global_step % config.log_interval == 0) {
                auto batch_end = std::chrono::steady_clock::now();
                auto batch_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    batch_end - batch_start
                ).count();
                
                float perplexity = calculate_perplexity(avg_loss);
                
                // Calculate progress
                size_t current_batch_in_epoch = num_batches + 1;
                float progress_pct = (float)current_batch_in_epoch / total_batches * 100.0f;
                
                // Estimate time remaining for epoch
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    batch_end - start_time
                ).count();
                float batches_per_sec = (float)global_step / std::max<long long>(elapsed, 1LL);
                size_t remaining_batches = total_batches - current_batch_in_epoch;
                float eta_seconds = remaining_batches / std::max(batches_per_sec, 0.001f);
                
                // Calculate current effective LR for logging
                float current_effective_lr = effective_lr_at(config, global_step);
                
                // Get CUDA memory stats
                size_t cuda_free = 0, cuda_total = 0;
                float cuda_used_gb = 0, cuda_total_gb = 0;
#ifdef USE_CUDA
                cudaMemGetInfo(&cuda_free, &cuda_total);
                cuda_used_gb = (cuda_total - cuda_free) / (1024.0f * 1024.0f * 1024.0f);
                cuda_total_gb = cuda_total / (1024.0f * 1024.0f * 1024.0f);
#endif
                
                std::cout << "Step " << global_step 
                          << " | Batch [" << current_batch_in_epoch << "/" << total_batches 
                          << " = " << std::fixed << std::setprecision(1) << progress_pct << "%]"
                          << " | Loss: " << std::setprecision(4) << avg_loss
                          << " | PPL: " << std::setprecision(2) << perplexity
                          << " | GPU: " << std::setprecision(2) << cuda_used_gb << "/" << cuda_total_gb << "GB"
                          << " | Time: " << std::fixed << batch_time << "ms"
                          << " | ETA: " << (int)(eta_seconds / 60) << "m" << ((int)eta_seconds % 60) << "s"
                          << std::endl;
                std::cout.flush();  // Force immediate output
                
                // Write to simple progress file (always works, unbuffered)
                {
                    std::ofstream prog("progress.txt", std::ios::trunc);
                    prog << "Step " << global_step << "/" << (total_batches * config.num_epochs)
                         << " | Epoch " << epoch << "/" << config.num_epochs
                         << " | Loss: " << avg_loss
                         << " | GPU: " << cuda_used_gb << "/" << cuda_total_gb << "GB"
                         << " | ETA: " << (int)(eta_seconds / 60) << "m" << std::endl;
                }
                
                // Write to log file IMMEDIATELY
                if (train_log.is_open()) {
                    train_log << "Step " << global_step 
                              << " | Batch [" << current_batch_in_epoch << "/" << total_batches 
                              << " = " << std::setprecision(1) << progress_pct << "%]"
                              << " | Loss: " << std::setprecision(4) << avg_loss
                              << " | PPL: " << std::setprecision(2) << perplexity
                              << " | GPU: " << std::setprecision(2) << cuda_used_gb << "/" << cuda_total_gb << "GB"
                              << " | Time: " << batch_time << "ms"
                              << " | Sequences: " << actual_batch_size
                              << " | ETA: " << (int)(eta_seconds / 60) << "m" << ((int)eta_seconds % 60) << "s"
                              << std::endl;
                    train_log.flush();  // FLUSH IMMEDIATELY FOR REAL-TIME MONITORING
                }
            }
            
            // Validation
            if (global_step % config.eval_interval == 0 && !val_data.empty()) {
                transformer.set_training(false);
                
                float val_loss = 0.0f;
                size_t val_count = 0;
                
                // Validate on first 100 sequences
                for (size_t j = 0; j < std::min(size_t(100), val_data.size()); ++j) {
                    const auto& val_tokens = val_data[j];
                    if (val_tokens.empty() || val_tokens.size() < 2) continue;
                    
                    std::string val_query = "validate";
                    TransformerOutput val_output = transformer.forward(val_tokens, val_query, tokenizer);
                    
                    // Logits shape: [seq_len x vocab_size]
                    size_t val_seq_len = val_output.logits.rows();
                    size_t val_vocab_size = val_output.logits.cols();
                    
                    // Calculate validation loss with proper log-softmax
                    float seq_loss = 0.0f;
                    size_t loss_count = 0;
                    for (size_t t = 1; t < val_tokens.size() && t-1 < val_seq_len; ++t) {
                        int target = val_tokens[t];
                        size_t pos = t - 1;
                        
                        if (static_cast<size_t>(target) < val_vocab_size) {
                            // Compute log-softmax over vocabulary for this position
                            float max_logit = -std::numeric_limits<float>::infinity();
                            for (size_t v = 0; v < val_vocab_size; ++v) {
                                max_logit = std::max(max_logit, val_output.logits(pos, v));
                            }
                            
                            float sum_exp = 0.0f;
                            for (size_t v = 0; v < val_vocab_size; ++v) {
                                sum_exp += std::exp(val_output.logits(pos, v) - max_logit);
                            }
                            float log_sum_exp = max_logit + std::log(sum_exp + 1e-10f);
                            
                            float log_prob = val_output.logits(pos, target) - log_sum_exp;
                            seq_loss += -log_prob;
                            loss_count++;
                        }
                    }
                    
                    if (loss_count > 0) {
                        val_loss += seq_loss / loss_count;
                        val_count++;
                    }
                }
                
                if (val_count > 0) {
                    val_loss /= val_count;
                } else {
                    val_loss = 0.0f;
                }
                float val_perplexity = calculate_perplexity(val_loss);
                
                std::cout << "\n*** Validation ***" << std::endl;
                std::cout << "Val Loss: " << std::fixed << std::setprecision(4) << val_loss
                          << " | Val Perplexity: " << std::setprecision(2) << val_perplexity;
                
                if (val_loss < best_val_loss) {
                    best_val_loss = val_loss;
                    std::cout << " [NEW BEST!]";
                    
                    // Log to file
                    if (train_log.is_open()) {
                        train_log << "\n*** Validation at step " << global_step << " ***" << std::endl;
                        train_log << "Val Loss: " << std::fixed << std::setprecision(4) << val_loss
                                  << " | Val Perplexity: " << std::setprecision(2) << val_perplexity
                                  << " [NEW BEST!]" << std::endl;
                        train_log.flush();
                    }
                    
                    // Save best checkpoint
                    static ModelSaver best_saver;
                    if (best_saver.saveCheckpoint(transformer, config.output_dir, "best_model", 
                                                   epoch, val_loss, global_step)) {
                        std::cout << " (Best model saved!)";
                    }
                }
                std::cout << "\n" << std::endl;
                
                transformer.set_training(true);
            }
            
            // Save checkpoint periodically
            if (global_step % config.save_interval == 0 && global_step > 0) {
                std::cout << "\n[CHECKPOINT] Attempting save at step " << global_step 
                          << " to " << config.output_dir << std::endl;
                std::cout.flush();
                // Pull device-resident LM head weights to host before saving
                // (this save happens in training mode where the device copy is
                // authoritative).
                if (transformer.get_lm_head()) {
                    transformer.get_lm_head()->sync_weights_from_device();
                }
                static ModelSaver saver;
                if (saver.saveCheckpoint(transformer, config.output_dir, "wikitext_model", 
                                         static_cast<int>(epoch), epoch_loss / (num_batches + 1), global_step)) {
                    std::cout << "[CHECKPOINT] SUCCESS at step " << global_step << std::endl;
                } else {
                    std::cerr << "[CHECKPOINT] FAILED at step " << global_step << std::endl;
                }
                std::cout.flush();
            }
            if (stop_training) break;
        }
        
        if (stop_training) {
            // Early-stop summary
            float avg_loss = num_batches > 0 ? epoch_loss / num_batches : 0.0f;
            final_train_loss = avg_loss;
            std::cout << "\nStopped after " << global_step << " steps. Avg Loss: "
                      << std::fixed << std::setprecision(4) << avg_loss
                      << " | PPL: " << std::setprecision(2) << calculate_perplexity(avg_loss) << std::endl;
            break;
        }
        
        // Epoch summary
        float avg_loss = epoch_loss / num_batches;
        final_train_loss = avg_loss;  // Track final training loss
        float avg_perplexity = calculate_perplexity(avg_loss);
        
        auto epoch_end = std::chrono::steady_clock::now();
        auto epoch_time = std::chrono::duration_cast<std::chrono::seconds>(
            epoch_end - start_time
        ).count();
        
        std::cout << "\nEpoch " << (epoch + 1) << " Complete" << std::endl;
        std::cout << "Avg Loss: " << std::fixed << std::setprecision(4) << avg_loss
                  << " | Avg Perplexity: " << std::setprecision(2) << avg_perplexity
                  << " | Time: " << epoch_time << "s" << std::endl;
    }
    
    auto end_time = std::chrono::steady_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::seconds>(
        end_time - start_time
    ).count();
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Training Complete!" << std::endl;
    std::cout << "Best Validation Loss: " << std::fixed << std::setprecision(4) << best_val_loss << std::endl;
    std::cout << "Best Validation Perplexity: " << std::setprecision(2) 
              << calculate_perplexity(best_val_loss) << std::endl;
    std::cout << "Total Training Time: " << total_time << "s" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Write final summary to log
    if (train_log.is_open()) {
        train_log << "\n========================================" << std::endl;
        train_log << "Training Complete!" << std::endl;
        train_log << "Final Training Loss: " << std::fixed << std::setprecision(4) << final_train_loss << std::endl;
        train_log << "Best Validation Loss: " << std::setprecision(4) << best_val_loss << std::endl;
        train_log << "Best Validation Perplexity: " << std::setprecision(2) 
                  << calculate_perplexity(best_val_loss) << std::endl;
        train_log << "Total Training Time: " << total_time << "s" << std::endl;
        train_log << "========================================" << std::endl;
        train_log.close();
    }
    
    return {final_train_loss, best_val_loss, static_cast<float>(total_time)};
}

/**
 * @brief Main training function
 */
int main(int argc, char** argv) {
    // Logging disabled for performance - uncomment to enable
    // Logger::getInstance().startLogging("training.log");
    
    try {
#ifdef USE_CUDA
        // Initialize CUDA for cuBLAS matmul (attention kernels have bugs, so we use CPU attention)
        try {
            cuda::initialize_cuda();
            print_cuda_info();
            std::cout << "CUDA initialized - using cuBLAS for matrix operations" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "CUDA initialization failed: " << e.what() << std::endl;
            std::cerr << "Falling back to CPU-only mode" << std::endl;
        }
#endif
        
        // Parse arguments
        TrainingConfig train_config;
        
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--resume" && i + 1 < argc) {
                train_config.resume_checkpoint = argv[++i];
            } else if (arg == "--small") {
                train_config.use_small = true;
            } else if (arg == "--tiny") {
                train_config.tiny = true;
            } else if (arg == "--export-gguf" && i + 1 < argc) {
                train_config.export_gguf = argv[++i];
            } else if (arg == "--export-safetensors" && i + 1 < argc) {
                train_config.export_safetensors = argv[++i];
            } else if (arg == "--config" && i + 1 < argc) {
                train_config.config_json = argv[++i];
            } else if (arg == "--family" && i + 1 < argc) {
                train_config.family = argv[++i];
            } else if (arg == "--lora") {
                lora::settings().enabled = true;
                if (i + 1 < argc && argv[i + 1][0] != '-') {
                    lora::settings().rank = std::stoul(argv[++i]);
                }
            } else if (arg == "--max-steps" && i + 1 < argc) {
                train_config.max_steps = std::stoul(argv[++i]);
            } else if (arg == "--max-seqs" && i + 1 < argc) {
                train_config.max_train_seqs = std::stoul(argv[++i]);
            } else if (arg == "--epochs" && i + 1 < argc) {
                train_config.num_epochs = std::stoul(argv[++i]);
            } else if (arg == "--batch-size" && i + 1 < argc) {
                train_config.batch_size = std::stoul(argv[++i]);
            } else if (arg == "--lr" && i + 1 < argc) {
                train_config.learning_rate = std::stof(argv[++i]);
            } else if (arg == "--cosine-decay") {
                train_config.cosine_decay = true;
            } else if (arg == "--help" || arg == "-h") {
                std::cout << "Usage: train_wikitext [data_path] [options]\n"
                          << "  --small             use truncated *-small.txt train files\n"
                          << "  --tiny              tiny model dims (fast round-trip smoke tests)\n"
                          << "  --max-steps N       stop after N optimizer steps\n"
                          << "  --max-seqs N        cap number of training sequences\n"
                          << "  --epochs N          number of epochs\n"
                          << "  --batch-size N      batch size\n"
                          << "  --lr F              learning rate (default 2e-4; SGD-momentum wants\n"
                          << "                      ~10-50x more than an Adam-tuned value)\n"
                          << "  --resume PATH       resume from checkpoint\n"
                          << "  --export-gguf PATH  export model to GGUF after training\n"
                          << "  --export-safetensors PATH  export model to safetensors after training\n"
                          << "  --config PATH       transformer_config.json (incl. export block:\n"
                          << "                      {\"export\": {\"format\": \"gguf|safetensors|both\", \"path\": ...}}\n"
                          << "                      and architecture block:\n"
                          << "                      {\"architecture\": {\"family\": ..., \"overrides\": {...}}})\n"
                          << "  --family NAME       architecture family preset: llama (default) | vanilla\n"
                          << "  --lora [RANK]       LoRA fine-tuning: freeze base, train rank-r adapters\n"
                          << "                      (default rank 8; pair with --resume CKPT)\n";
                return 0;
            } else if (arg.find("--") != 0) {
                train_config.data_path = arg;
            }
        }
        
        std::cout << "========================================" << std::endl;
        std::cout << "Transformer Training on WikiText" << std::endl;
#ifdef USE_CUDA
        std::cout << "Mode: GPU (CUDA)" << std::endl;
#else
        std::cout << "Mode: CPU" << std::endl;
#endif
        std::cout << "========================================" << std::endl;
        std::cout << "Data path: " << train_config.data_path << std::endl;
        std::cout << "Output dir: " << train_config.output_dir << std::endl;
        
        // Create output directory (cross-platform)
        try {
            std::filesystem::create_directories(train_config.output_dir);
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to create output directory: " << e.what() << std::endl;
        }
        
        // Initialize tokenizer FIRST
        std::cout << "\nInitializing tokenizer..." << std::endl;
        auto tokenizer = std::make_shared<TiktokenTokenizer>();
        
        // Build vocabulary from WikiText training data itself
        std::cout << "Building vocabulary from WikiText data..." << std::endl;
        // Select training files: small files for CPU builds or when --small is given,
        // full files for the GPU build otherwise.
        bool use_small_files = train_config.use_small;
#ifndef USE_CUDA
        use_small_files = true;  // CPU: truncated datasets for tractable runtime
#endif
        std::string train_file_0, train_file_1;
        if (use_small_files) {
            train_file_0 = train_config.data_path + "/train-0-small.txt";
            train_file_1 = train_config.data_path + "/train-1-small.txt";
        } else {
            train_file_0 = train_config.data_path + "/train-0.txt";
            train_file_1 = train_config.data_path + "/train-1.txt";
        }
        std::cout << "Training files: " << train_file_0 << ", " << train_file_1 << std::endl;
        
        // Build vocab from both training files using plain text method
        try {
            std::cout << "Processing train-0.txt..." << std::endl;
            tokenizer->build_vocabulary_from_plain_text(train_file_0);
            
            std::cout << "Processing train-1.txt..." << std::endl;
            tokenizer->build_vocabulary_from_plain_text(train_file_1);
            
            std::cout << "Final vocabulary size: " << tokenizer->vocab_size() << " tokens" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Could not build vocabulary from WikiText data: " << e.what() << std::endl;
            std::cerr << "Falling back to training_pairs.txt..." << std::endl;
            tokenizer->build_vocabulary_from_file("../data/training_pairs.txt");
            std::cout << "Vocabulary size: " << tokenizer->vocab_size() << " tokens" << std::endl;
        }
        
        // Now initialize model config with vocabulary size from tokenizer
        TransformerConfig model_config;
        if (train_config.tiny) {
            // Tiny model for fast train -> export -> tinyllama.cpp round-trip tests
            model_config.vocab_size = std::min(tokenizer->vocab_size(), static_cast<size_t>(5000));
            model_config.hidden_size = 128;
            model_config.num_heads = 4;
            model_config.num_layers = 2;
            model_config.intermediate_size = 512;
        } else {
#ifdef USE_CUDA
            // GPU: Larger model to utilize RTX 4060's 8GB VRAM
            model_config.vocab_size = std::min(tokenizer->vocab_size(), static_cast<size_t>(20000));
            model_config.hidden_size = 512;
            model_config.num_heads = 8;
            model_config.num_layers = 6;
            model_config.intermediate_size = 2048;
#else
            // CPU: Smaller model for reasonable training time
            model_config.vocab_size = std::min(tokenizer->vocab_size(), static_cast<size_t>(5000));
            model_config.hidden_size = 128;
            model_config.num_heads = 4;
            model_config.num_layers = 2;
            model_config.intermediate_size = 512;
#endif
        }
        model_config.head_dim = model_config.hidden_size / model_config.num_heads;
        model_config.max_seq_length = train_config.max_seq_len;
        model_config.dropout_rate = 0.1f;
        model_config.layer_norm_epsilon = 1e-5f;
        std::cout << std::scientific << "[DEBUG] layer_norm_epsilon = " << model_config.layer_norm_epsilon << std::defaultfloat << std::endl;
        model_config.gradient_clip_threshold = 1.0f;  // Tightened from 5.0 for stability
        model_config.use_flash_attention = false;  // Disable buggy fused attention kernels

        // Keep attention config self-consistent: the batched path uses full multi-head
        // attention (num_heads), so disable GQA and align num_kv_heads with num_heads.
        // (Previously num_kv_heads kept a stale default of 12 while num_heads was 8.)
        model_config.use_gqa = false;
        model_config.num_kv_heads = model_config.num_heads;

        // Architecture selection: family preset (--family) applied first, then
        // an optional JSON config overlay (--config) which may carry its own
        // "architecture" block (plus export/model/training keys). The default
        // "llama" preset reproduces the old hardcoded llama_mode setup exactly:
        // RMSNorm + RoPE + SwiGLU + no biases, bit-exact with tinyllama.cpp.
        arch::ArchitectureSpec spec = arch::ArchitectureSpec::from_family(train_config.family);
        spec.validate();
        spec.apply(model_config);
        if (!train_config.config_json.empty()) {
            std::cout << "Loading config overlay: " << train_config.config_json << std::endl;
            model_config.load_from_json(train_config.config_json);
        }
        // Bias freezing must be set BEFORE model construction.
        transformer_runtime::llama_no_bias = !model_config.use_biases;
        std::cout << "Architecture: " << spec.describe()
                  << (train_config.config_json.empty() ? "" : " (+ config overlay)")
                  << std::endl;

        // LoRA: config-file block enables it too (CLI --lora wins on rank if given)
        if (model_config.lora_enabled && !lora::settings().enabled) {
            lora::settings().enabled = true;
            lora::settings().rank = model_config.lora_rank;
            lora::settings().alpha = model_config.lora_alpha;
        }
        if (lora::settings().enabled) {
            std::cout << "LoRA fine-tuning ENABLED: rank=" << lora::settings().rank
                      << " alpha=" << lora::settings().alpha
                      << " (base weights, embeddings, norms, lm_head frozen)" << std::endl;
        }
        
        std::cout << "\nModel Configuration:" << std::endl;
        std::cout << "Vocab size: " << model_config.vocab_size << " (from tokenizer)" << std::endl;
        std::cout << "Hidden size: " << model_config.hidden_size << std::endl;
        std::cout << "Num heads: " << model_config.num_heads << std::endl;
        std::cout << "Num layers: " << model_config.num_layers << std::endl;
        std::cout << "Max sequence: " << model_config.max_seq_length << std::endl;
        std::cout << "Layer norm epsilon: " << model_config.layer_norm_epsilon << std::endl;
        
#ifdef USE_CUDA
        std::cout << "\nLoading training data (FULL DATASET - GPU MODE)..." << std::endl;
#else
        std::cout << "\nLoading training data (SMALL DATASET - CPU MODE)..." << std::endl;
#endif
        auto train_data = std::vector<std::vector<int>>();
        
        // Load train-0-small.txt (using same path as vocab building)
        auto train_data_0 = load_and_tokenize(train_file_0, *tokenizer, train_config.max_seq_len);
        train_data.insert(train_data.end(), train_data_0.begin(), train_data_0.end());
        
        // Load train-1-small.txt (using same path as vocab building)
        auto train_data_1 = load_and_tokenize(train_file_1, *tokenizer, train_config.max_seq_len);
        train_data.insert(train_data.end(), train_data_1.begin(), train_data_1.end());
        
        // Optionally cap number of training sequences (fast smoke tests)
        if (train_config.max_train_seqs > 0 && train_data.size() > train_config.max_train_seqs) {
            train_data.resize(train_config.max_train_seqs);
            std::cout << "Capped training set to " << train_data.size() << " sequences" << std::endl;
        }
        
        // Load validation data
        std::string val_file = train_config.data_path + "/validation.txt";
        auto val_data = load_and_tokenize(val_file, *tokenizer, train_config.max_seq_len);
        
        // Initialize model
        std::cout << "\nInitializing transformer..." << std::endl;
        Transformer transformer(model_config, tokenizer);
        
        // Resume from checkpoint if specified
        size_t start_step = 0;
        size_t start_epoch = 0;
        if (!train_config.resume_checkpoint.empty()) {
            std::cout << "\n========================================" << std::endl;
            std::cout << "Resuming from checkpoint: " << train_config.resume_checkpoint << std::endl;
            std::cout << "========================================" << std::endl;
            
            // First read checkpoint metadata to get step/epoch
            std::ifstream meta_file(train_config.resume_checkpoint, std::ios::binary);
            if (meta_file) {
                size_t meta_size;
                meta_file.read(reinterpret_cast<char*>(&meta_size), sizeof(meta_size));
                if (meta_size > 0 && meta_size < 10000) {
                    std::string meta_str(meta_size, '\0');
                    meta_file.read(&meta_str[0], meta_size);
                    try {
                        auto checkpoint_meta = nlohmann::json::parse(meta_str);
                        if (checkpoint_meta.contains("step")) {
                            start_step = checkpoint_meta["step"].get<size_t>();
                        }
                        if (checkpoint_meta.contains("epoch")) {
                            start_epoch = checkpoint_meta["epoch"].get<size_t>();
                        }
                        std::cout << "Checkpoint metadata: epoch=" << start_epoch << " step=" << start_step << std::endl;
                    } catch (...) {
                        std::cerr << "Warning: Could not parse checkpoint metadata" << std::endl;
                    }
                }
            }
            meta_file.close();
            
            ModelSaver saver;
            if (saver.loadCheckpoint(transformer, train_config.resume_checkpoint)) {
                std::cout << "Loaded checkpoint from epoch " << start_epoch << ", step " << start_step << std::endl;
                // loadCheckpoint rebuilds layer components, dropping their
                // tokenizer wiring; without this, the first VALIDATION after a
                // resume crashes with "Tokenizer not set in MultiHeadAttention"
                // (found 2026-07-12 when reclaim-resume first exercised this path).
                transformer.set_tokenizer(tokenizer);
            } else {
                std::cerr << "WARNING: Failed to load checkpoint, starting from scratch" << std::endl;
                start_step = 0;
                start_epoch = 0;
            }
        }
        
        // Train! (transformer handles its own parameter updates internally)
        train(transformer, train_data, val_data, train_config, *tokenizer, start_step, start_epoch);

        // Resolve export requests: explicit CLI flags win; otherwise the
        // config file's export block ("gguf" | "safetensors" | "both") applies.
        if (train_config.export_gguf.empty() && train_config.export_safetensors.empty()
            && model_config.export_format != "none" && !model_config.export_path.empty()) {
            auto with_ext = [](std::string base, const std::string& ext) {
                if (base.size() < ext.size()
                    || base.compare(base.size() - ext.size(), ext.size(), ext) != 0) {
                    base += ext;
                }
                return base;
            };
            if (model_config.export_format == "gguf" || model_config.export_format == "both") {
                train_config.export_gguf = with_ext(model_config.export_path, ".gguf");
            }
            if (model_config.export_format == "safetensors" || model_config.export_format == "both") {
                train_config.export_safetensors = with_ext(model_config.export_path, ".safetensors");
            }
        }

        // Export to GGUF for tinyllama.cpp inference
        if (!train_config.export_gguf.empty()) {
            if (!model_config.use_rms_norm || !model_config.use_rope || model_config.use_biases) {
                std::cerr << "WARNING: GGUF export requested for a non-llama architecture "
                          << "(tinyllama.cpp executes RMSNorm + RoPE + bias-free models only); "
                          << "the file will be written but will not run correctly there. "
                          << "Use --export-safetensors for non-llama architectures." << std::endl;
            }
            std::cout << "\nExporting model to GGUF: " << train_config.export_gguf << std::endl;
#ifdef USE_CUDA
            // LM head weights live on the GPU during training; pull them back
            transformer.get_lm_head()->sync_weights_from_device();
#endif
            gguf_export::GGUFExportConfig export_cfg;
            export_cfg.model_name = "transformer_cpp_wikitext";
            if (gguf_export::export_to_gguf(transformer, *tokenizer,
                                            train_config.export_gguf, export_cfg)) {
                std::cout << "GGUF export complete: " << train_config.export_gguf << std::endl;
            } else {
                std::cerr << "GGUF export FAILED" << std::endl;
            }
        }

        // Export to safetensors (same tensor enumeration/layout as GGUF)
        if (!train_config.export_safetensors.empty()) {
            std::cout << "\nExporting model to safetensors: "
                      << train_config.export_safetensors << std::endl;
#ifdef USE_CUDA
            // LM head weights live on the GPU during training; pull them back
            // (idempotent if the GGUF export already synced).
            transformer.get_lm_head()->sync_weights_from_device();
#endif
            safetensors_export::SafetensorsExportConfig st_cfg;
            st_cfg.model_name = "transformer_cpp_lm";
            if (safetensors_export::export_to_safetensors(
                    transformer, *tokenizer,
                    train_config.export_safetensors, st_cfg)) {
                std::cout << "Safetensors export complete: "
                          << train_config.export_safetensors << std::endl;
            } else {
                std::cerr << "Safetensors export FAILED" << std::endl;
            }
        }

        // Stop logging and flush everything
        Logger::getInstance().stopLogging();

#ifdef USE_CUDA
        // Ensure all in-flight GPU work has completed, then free OUR CUDA
        // resources (loss pools + matmul pool/cuBLAS/streams) while the context
        // is alive.
        cudaDeviceSynchronize();
        cuda::cleanup_loss_resources();
        cuda::cleanup_cuda();

        // All meaningful state (checkpoints, logs) is persisted. Running the
        // normal shutdown path is unsafe here: the transformer's component
        // destructors (CudaMatrix cudaFree) and the cudart/cuBLAS atexit handlers
        // intermittently hit an access violation (0xC0000005) during the CUDA
        // runtime's teardown on Windows. Fast-exit BEFORE the transformer's
        // destructor runs; the OS reclaims the leaked device memory at exit.
        std::cout.flush();
        std::cerr.flush();
        std::_Exit(0);
#endif
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        Logger::getInstance().stopLogging();
        return 1;
    }
}


