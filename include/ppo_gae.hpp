#pragma once

#include <vector>
#include <memory>
#include <torch/torch.h>
#include <cuda_runtime.h>
#include "distributed_transformer.hpp"

namespace rlhf {

// Trajectory buffer for PPO
struct Trajectory {
    torch::Tensor states;           // [batch_size, seq_len, hidden_dim]
    torch::Tensor actions;          // [batch_size, seq_len]
    torch::Tensor action_log_probs; // [batch_size, seq_len]
    torch::Tensor rewards;          // [batch_size]
    torch::Tensor values;           // [batch_size]
    torch::Tensor advantages;       // [batch_size]
    torch::Tensor returns;          // [batch_size]
    torch::Tensor masks;            // [batch_size] - for episode boundaries
    
    void to_device(torch::Device device);
    torch::Tensor compute_gae(float gamma, float lambda);
};

// PPO configuration
struct PPOConfig {
    // Hyperparameters
    float learning_rate = 2e-5f;
    float clip_epsilon = 0.2f;
    float value_loss_coef = 0.5f;
    float entropy_coef = 0.01f;
    float max_grad_norm = 0.5f;
    
    // GAE parameters
    float gamma = 0.99f;
    float lambda = 0.95f;
    
    // Training parameters
    int ppo_epochs = 4;
    int mini_batch_size = 32;
    int trajectory_batch_size = 512;
    
    // Advantage normalization
    bool normalize_advantages = true;
    float advantage_epsilon = 1e-8f;
    
    // KL divergence control
    float target_kl = 0.01f;
    bool early_stop_on_kl = true;
    
    // Learning rate schedule
    bool use_lr_schedule = true;
    float lr_schedule_power = 1.0f;
    int total_timesteps = 1000000;
};

// Value head for PPO
class ValueHead : public torch::nn::Module {
public:
    ValueHead(int hidden_dim, int intermediate_dim = 1024);
    
    torch::Tensor forward(torch::Tensor hidden_states);
    
private:
    torch::nn::Linear fc1_{nullptr};
    torch::nn::Linear fc2_{nullptr};
    torch::nn::Linear fc3_{nullptr};
    torch::nn::LayerNorm ln_{nullptr};
    torch::nn::Dropout dropout_{nullptr};
};

// PPO trainer with GAE
class PPOTrainer {
public:
    PPOTrainer(std::shared_ptr<model::DistributedTransformer> policy_model,
              PPOConfig config = PPOConfig());
    ~PPOTrainer();
    
    // Training loop
    struct TrainingMetrics {
        float policy_loss;
        float value_loss;
        float entropy_loss;
        float total_loss;
        float kl_divergence;
        float explained_variance;
        float advantage_mean;
        float advantage_std;
        float value_pred_mean;
        float return_mean;
        float reward_mean;
        int gradient_steps;
    };
    
    TrainingMetrics train_step(Trajectory& trajectory);
    
    // Compute advantages using GAE
    void compute_advantages(Trajectory& trajectory);
    
    // Policy evaluation
    struct PolicyOutput {
        torch::Tensor action_log_probs;
        torch::Tensor values;
        torch::Tensor entropy;
        torch::Tensor actions;
    };
    
    PolicyOutput evaluate_actions(torch::Tensor states, torch::Tensor actions);
    PolicyOutput sample_actions(torch::Tensor states);
    
    // Learning rate scheduling
    void update_learning_rate(int current_timestep);
    float get_current_lr() const { return current_lr_; }
    
    // Checkpointing
    void save_checkpoint(const std::string& path);
    void load_checkpoint(const std::string& path);
    
    // Advanced features
    void enable_mixed_precision();
    void enable_gradient_checkpointing();
    void set_distributed_mode(bool enable);
    
private:
    std::shared_ptr<model::DistributedTransformer> policy_model_;
    std::unique_ptr<ValueHead> value_head_;
    PPOConfig config_;
    
    // Optimizers
    std::unique_ptr<torch::optim::AdamW> policy_optimizer_;
    std::unique_ptr<torch::optim::AdamW> value_optimizer_;
    
    // Learning rate
    float current_lr_;
    std::unique_ptr<torch::optim::lr_scheduler::LambdaLR> lr_scheduler_;
    
    // Statistics tracking
    struct RunningStat {
        float mean = 0.0f;
        float var = 1.0f;
        int count = 0;
        
        void update(float value);
        float normalize(float value) const;
    };
    
    RunningStat reward_stats_;
    RunningStat advantage_stats_;
    
    // Mixed precision training
    bool use_mixed_precision_ = false;
    std::unique_ptr<torch::cuda::amp::GradScaler> scaler_;
    
    // Helper functions
    torch::Tensor compute_policy_loss(torch::Tensor old_log_probs,
                                      torch::Tensor new_log_probs,
                                      torch::Tensor advantages);
    
    torch::Tensor compute_value_loss(torch::Tensor values,
                                     torch::Tensor returns);
    
    float compute_kl_divergence(torch::Tensor old_log_probs,
                               torch::Tensor new_log_probs);
    
    float compute_explained_variance(torch::Tensor values,
                                    torch::Tensor returns);
};

// Distributed PPO for multi-GPU training
class DistributedPPO : public PPOTrainer {
public:
    DistributedPPO(std::shared_ptr<model::DistributedTransformer> policy_model,
                  PPOConfig config,
                  int world_size,
                  int rank);
    
    // Distributed training
    TrainingMetrics distributed_train_step(std::vector<Trajectory>& trajectories);
    
    // All-reduce operations
    void all_reduce_gradients();
    void all_reduce_statistics(TrainingMetrics& metrics);
    
    // Trajectory gathering
    std::vector<Trajectory> gather_trajectories();
    
private:
    int world_size_;
    int rank_;
    ncclComm_t nccl_comm_;
    
    // Distributed buffers
    std::vector<torch::Tensor> gradient_buffers_;
    std::vector<torch::Tensor> statistics_buffers_;
};

// Advanced GAE features
class GeneralizedAdvantageEstimator {
public:
    GeneralizedAdvantageEstimator(float gamma = 0.99f, float lambda = 0.95f);
    
    // Compute GAE
    torch::Tensor compute(torch::Tensor rewards,
                         torch::Tensor values,
                         torch::Tensor next_values,
                         torch::Tensor dones);
    
    // Compute returns
    torch::Tensor compute_returns(torch::Tensor rewards,
                                 torch::Tensor dones);
    
    // V-trace for off-policy correction
    torch::Tensor compute_vtrace(torch::Tensor rewards,
                                torch::Tensor values,
                                torch::Tensor next_values,
                                torch::Tensor log_rhos,
                                torch::Tensor dones);
    
    // TD(λ) returns
    torch::Tensor compute_td_lambda(torch::Tensor rewards,
                                   torch::Tensor values,
                                   torch::Tensor next_values,
                                   torch::Tensor dones);
    
private:
    float gamma_;
    float lambda_;
    
    // Helper functions
    torch::Tensor discount_cumsum(torch::Tensor x, float discount);
};

// Adaptive PPO with automatic hyperparameter tuning
class AdaptivePPO : public PPOTrainer {
public:
    AdaptivePPO(std::shared_ptr<model::DistributedTransformer> policy_model,
               PPOConfig initial_config);
    
    // Adaptive training
    TrainingMetrics adaptive_train_step(Trajectory& trajectory);
    
    // Automatic hyperparameter adjustment
    void adjust_clip_epsilon(float kl_divergence);
    void adjust_learning_rate(float policy_loss);
    void adjust_entropy_coefficient(float entropy);
    
    // Performance tracking
    struct PerformanceHistory {
        std::vector<float> rewards;
        std::vector<float> kl_divergences;
        std::vector<float> policy_losses;
        std::vector<float> value_losses;
        
        void update(const TrainingMetrics& metrics, float reward);
        float get_trend() const;
    };
    
    PerformanceHistory get_history() const { return history_; }
    
private:
    PerformanceHistory history_;
    
    // Adaptation parameters
    float clip_epsilon_min_ = 0.1f;
    float clip_epsilon_max_ = 0.3f;
    float lr_min_ = 1e-6f;
    float lr_max_ = 1e-3f;
    float entropy_coef_min_ = 0.001f;
    float entropy_coef_max_ = 0.1f;
    
    // Adaptation rates
    float clip_adapt_rate_ = 0.01f;
    float lr_adapt_rate_ = 0.001f;
    float entropy_adapt_rate_ = 0.005f;
};

// Importance sampling for off-policy PPO
class ImportanceSamplingPPO : public PPOTrainer {
public:
    ImportanceSamplingPPO(std::shared_ptr<model::DistributedTransformer> policy_model,
                         PPOConfig config);
    
    // Off-policy training with importance sampling
    TrainingMetrics train_with_importance_sampling(Trajectory& trajectory,
                                                  torch::Tensor behavior_log_probs);
    
    // Compute importance weights
    torch::Tensor compute_importance_weights(torch::Tensor old_log_probs,
                                            torch::Tensor new_log_probs,
                                            float clip_threshold = 10.0f);
    
    // Truncated importance sampling
    torch::Tensor compute_truncated_weights(torch::Tensor weights,
                                          float threshold = 2.0f);
    
private:
    // Weight normalization
    torch::Tensor normalize_weights(torch::Tensor weights);
    
    // Effective sample size
    float compute_effective_sample_size(torch::Tensor weights);
};

} // namespace rlhf
