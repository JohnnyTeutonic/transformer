#include "ppo_gae.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>

namespace rlhf {

// Trajectory implementation
void Trajectory::to_device(torch::Device device) {
    states = states.to(device);
    actions = actions.to(device);
    action_log_probs = action_log_probs.to(device);
    rewards = rewards.to(device);
    values = values.to(device);
    advantages = advantages.to(device);
    returns = returns.to(device);
    masks = masks.to(device);
}

torch::Tensor Trajectory::compute_gae(float gamma, float lambda) {
    int batch_size = rewards.size(0);
    torch::Tensor advantages = torch::zeros_like(rewards);
    
    float gae = 0.0f;
    for (int t = batch_size - 1; t >= 0; --t) {
        float delta = rewards[t].item<float>();
        if (t < batch_size - 1) {
            delta += gamma * values[t + 1].item<float>() * masks[t + 1].item<float>();
        }
        delta -= values[t].item<float>();
        
        gae = delta + gamma * lambda * masks[t].item<float>() * gae;
        advantages[t] = gae;
    }
    
    return advantages;
}

// ValueHead implementation
ValueHead::ValueHead(int hidden_dim, int intermediate_dim) {
    fc1_ = register_module("fc1", torch::nn::Linear(hidden_dim, intermediate_dim));
    fc2_ = register_module("fc2", torch::nn::Linear(intermediate_dim, intermediate_dim));
    fc3_ = register_module("fc3", torch::nn::Linear(intermediate_dim, 1));
    ln_ = register_module("ln", torch::nn::LayerNorm(torch::nn::LayerNormOptions({intermediate_dim})));
    dropout_ = register_module("dropout", torch::nn::Dropout(0.1));
}

torch::Tensor ValueHead::forward(torch::Tensor hidden_states) {
    // Pool hidden states (use last token)
    auto pooled = hidden_states.select(1, -1);
    
    // Forward through layers
    auto x = torch::gelu(fc1_->forward(pooled));
    x = ln_->forward(x);
    x = dropout_->forward(x);
    x = torch::gelu(fc2_->forward(x));
    x = dropout_->forward(x);
    x = fc3_->forward(x);
    
    return x.squeeze(-1);
}

// PPOTrainer implementation
PPOTrainer::PPOTrainer(std::shared_ptr<model::DistributedTransformer> policy_model,
                      PPOConfig config)
    : policy_model_(policy_model), config_(config), current_lr_(config.learning_rate) {
    
    // Initialize value head
    value_head_ = std::make_unique<ValueHead>(policy_model_->get_hidden_dim());
    
    // Initialize optimizers
    std::vector<torch::Tensor> policy_params;
    for (const auto& p : policy_model_->parameters()) {
        policy_params.push_back(p);
    }
    
    policy_optimizer_ = std::make_unique<torch::optim::AdamW>(
        policy_params,
        torch::optim::AdamWOptions(config_.learning_rate)
            .betas({0.9, 0.999})
            .weight_decay(0.01)
    );
    
    value_optimizer_ = std::make_unique<torch::optim::AdamW>(
        value_head_->parameters(),
        torch::optim::AdamWOptions(config_.learning_rate)
            .betas({0.9, 0.999})
            .weight_decay(0.01)
    );
    
    // Initialize mixed precision if available
    if (torch::cuda::is_available() && use_mixed_precision_) {
        scaler_ = std::make_unique<torch::cuda::amp::GradScaler>();
    }
}

PPOTrainer::~PPOTrainer() = default;

PPOTrainer::TrainingMetrics PPOTrainer::train_step(Trajectory& trajectory) {
    TrainingMetrics metrics = {};
    
    // Compute advantages
    compute_advantages(trajectory);
    
    // Normalize advantages if configured
    if (config_.normalize_advantages) {
        auto adv_mean = trajectory.advantages.mean();
        auto adv_std = trajectory.advantages.std();
        trajectory.advantages = (trajectory.advantages - adv_mean) / (adv_std + config_.advantage_epsilon);
    }
    
    // Store old action log probs
    auto old_action_log_probs = trajectory.action_log_probs.detach();
    
    int batch_size = trajectory.states.size(0);
    int mini_batch_size = config_.mini_batch_size;
    
    // PPO epochs
    for (int epoch = 0; epoch < config_.ppo_epochs; ++epoch) {
        // Create random indices for mini-batch sampling
        std::vector<int> indices(batch_size);
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(indices.begin(), indices.end(), gen);
        
        // Mini-batch training
        for (int start_idx = 0; start_idx < batch_size; start_idx += mini_batch_size) {
            int end_idx = std::min(start_idx + mini_batch_size, batch_size);
            
            // Get mini-batch indices
            torch::Tensor batch_indices = torch::zeros({end_idx - start_idx}, torch::kLong);
            for (int i = 0; i < end_idx - start_idx; ++i) {
                batch_indices[i] = indices[start_idx + i];
            }
            
            // Get mini-batch data
            auto mb_states = trajectory.states.index_select(0, batch_indices);
            auto mb_actions = trajectory.actions.index_select(0, batch_indices);
            auto mb_old_log_probs = old_action_log_probs.index_select(0, batch_indices);
            auto mb_advantages = trajectory.advantages.index_select(0, batch_indices);
            auto mb_returns = trajectory.returns.index_select(0, batch_indices);
            
            // Forward pass
            auto output = evaluate_actions(mb_states, mb_actions);
            
            // Compute policy loss
            auto ratio = torch::exp(output.action_log_probs - mb_old_log_probs);
            auto surr1 = ratio * mb_advantages;
            auto surr2 = torch::clamp(ratio, 1.0f - config_.clip_epsilon, 1.0f + config_.clip_epsilon) * mb_advantages;
            auto policy_loss = -torch::min(surr1, surr2).mean();
            
            // Compute value loss
            auto value_loss = torch::mse_loss(output.values, mb_returns);
            
            // Compute entropy loss
            auto entropy_loss = -output.entropy.mean();
            
            // Total loss
            auto total_loss = policy_loss + 
                             config_.value_loss_coef * value_loss + 
                             config_.entropy_coef * entropy_loss;
            
            // Backward pass
            policy_optimizer_->zero_grad();
            value_optimizer_->zero_grad();
            
            if (use_mixed_precision_ && scaler_) {
                scaler_->scale(total_loss)->backward();
                scaler_->unscale_(*policy_optimizer_);
                scaler_->unscale_(*value_optimizer_);
                
                // Gradient clipping
                torch::nn::utils::clip_grad_norm_(policy_model_->parameters(), config_.max_grad_norm);
                torch::nn::utils::clip_grad_norm_(value_head_->parameters(), config_.max_grad_norm);
                
                scaler_->step(*policy_optimizer_);
                scaler_->step(*value_optimizer_);
                scaler_->update();
            } else {
                total_loss.backward();
                
                // Gradient clipping
                torch::nn::utils::clip_grad_norm_(policy_model_->parameters(), config_.max_grad_norm);
                torch::nn::utils::clip_grad_norm_(value_head_->parameters(), config_.max_grad_norm);
                
                policy_optimizer_->step();
                value_optimizer_->step();
            }
            
            // Update metrics
            metrics.policy_loss += policy_loss.item<float>();
            metrics.value_loss += value_loss.item<float>();
            metrics.entropy_loss += entropy_loss.item<float>();
            metrics.total_loss += total_loss.item<float>();
            metrics.gradient_steps++;
            
            // Compute KL divergence for early stopping
            float kl = compute_kl_divergence(mb_old_log_probs, output.action_log_probs);
            metrics.kl_divergence += kl;
            
            // Early stopping based on KL divergence
            if (config_.early_stop_on_kl && kl > config_.target_kl) {
                break;
            }
        }
        
        // Check if we should stop PPO epochs
        if (config_.early_stop_on_kl && 
            metrics.kl_divergence / metrics.gradient_steps > config_.target_kl) {
            break;
        }
    }
    
    // Average metrics
    if (metrics.gradient_steps > 0) {
        metrics.policy_loss /= metrics.gradient_steps;
        metrics.value_loss /= metrics.gradient_steps;
        metrics.entropy_loss /= metrics.gradient_steps;
        metrics.total_loss /= metrics.gradient_steps;
        metrics.kl_divergence /= metrics.gradient_steps;
    }
    
    // Additional metrics
    metrics.advantage_mean = trajectory.advantages.mean().item<float>();
    metrics.advantage_std = trajectory.advantages.std().item<float>();
    metrics.value_pred_mean = trajectory.values.mean().item<float>();
    metrics.return_mean = trajectory.returns.mean().item<float>();
    metrics.reward_mean = trajectory.rewards.mean().item<float>();
    metrics.explained_variance = compute_explained_variance(trajectory.values, trajectory.returns);
    
    return metrics;
}

void PPOTrainer::compute_advantages(Trajectory& trajectory) {
    GeneralizedAdvantageEstimator gae(config_.gamma, config_.lambda);
    
    // Compute next values (shift values by 1 and pad with 0)
    auto next_values = torch::cat({
        trajectory.values.slice(0, 1),
        torch::zeros({1}, trajectory.values.options())
    }, 0);
    
    // Compute GAE
    trajectory.advantages = gae.compute(
        trajectory.rewards,
        trajectory.values,
        next_values,
        1.0f - trajectory.masks  // Convert masks to dones
    );
    
    // Compute returns
    trajectory.returns = trajectory.advantages + trajectory.values;
}

PPOTrainer::PolicyOutput PPOTrainer::evaluate_actions(torch::Tensor states, 
                                                      torch::Tensor actions) {
    PolicyOutput output;
    
    // Get policy logits from the model
    auto model_output = policy_model_->forward(states);
    auto logits = model_output["logits"];
    
    // Compute action probabilities
    auto probs = torch::softmax(logits, -1);
    
    // Compute log probabilities for taken actions
    auto log_probs = torch::log_softmax(logits, -1);
    output.action_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1);
    
    // Compute entropy
    output.entropy = -(probs * log_probs).sum(-1);
    
    // Compute values
    auto hidden_states = model_output["hidden_states"];
    output.values = value_head_->forward(hidden_states);
    
    output.actions = actions;
    
    return output;
}

PPOTrainer::PolicyOutput PPOTrainer::sample_actions(torch::Tensor states) {
    PolicyOutput output;
    
    // Get policy logits from the model
    auto model_output = policy_model_->forward(states);
    auto logits = model_output["logits"];
    
    // Sample actions from categorical distribution
    auto probs = torch::softmax(logits, -1);
    auto dist = torch::distributions::Categorical(probs);
    output.actions = dist.sample();
    
    // Compute log probabilities
    auto log_probs = torch::log_softmax(logits, -1);
    output.action_log_probs = log_probs.gather(-1, output.actions.unsqueeze(-1)).squeeze(-1);
    
    // Compute entropy
    output.entropy = dist.entropy();
    
    // Compute values
    auto hidden_states = model_output["hidden_states"];
    output.values = value_head_->forward(hidden_states);
    
    return output;
}

void PPOTrainer::update_learning_rate(int current_timestep) {
    if (!config_.use_lr_schedule) return;
    
    float progress = static_cast<float>(current_timestep) / config_.total_timesteps;
    float lr_multiplier = std::pow(1.0f - progress, config_.lr_schedule_power);
    
    current_lr_ = config_.learning_rate * lr_multiplier;
    
    // Update optimizer learning rates
    for (auto& param_group : policy_optimizer_->param_groups()) {
        param_group.options().set_lr(current_lr_);
    }
    
    for (auto& param_group : value_optimizer_->param_groups()) {
        param_group.options().set_lr(current_lr_);
    }
}

void PPOTrainer::save_checkpoint(const std::string& path) {
    torch::save(policy_model_, path + "_policy.pt");
    torch::save(value_head_, path + "_value.pt");
    
    // Save optimizer states
    torch::serialize::OutputArchive policy_archive;
    policy_optimizer_->save(policy_archive);
    policy_archive.save_to(path + "_policy_optim.pt");
    
    torch::serialize::OutputArchive value_archive;
    value_optimizer_->save(value_archive);
    value_archive.save_to(path + "_value_optim.pt");
}

void PPOTrainer::load_checkpoint(const std::string& path) {
    torch::load(policy_model_, path + "_policy.pt");
    torch::load(value_head_, path + "_value.pt");
    
    // Load optimizer states
    torch::serialize::InputArchive policy_archive;
    policy_archive.load_from(path + "_policy_optim.pt");
    policy_optimizer_->load(policy_archive);
    
    torch::serialize::InputArchive value_archive;
    value_archive.load_from(path + "_value_optim.pt");
    value_optimizer_->load(value_archive);
}

void PPOTrainer::enable_mixed_precision() {
    use_mixed_precision_ = true;
    if (torch::cuda::is_available()) {
        scaler_ = std::make_unique<torch::cuda::amp::GradScaler>();
    }
}

void PPOTrainer::enable_gradient_checkpointing() {
    // Enable gradient checkpointing in the model
    policy_model_->enable_gradient_checkpointing();
}

float PPOTrainer::compute_kl_divergence(torch::Tensor old_log_probs, 
                                        torch::Tensor new_log_probs) {
    auto kl = (torch::exp(old_log_probs) * (old_log_probs - new_log_probs)).mean();
    return kl.item<float>();
}

float PPOTrainer::compute_explained_variance(torch::Tensor values, torch::Tensor returns) {
    auto var_returns = returns.var();
    auto var_error = (returns - values).var();
    
    if (var_returns.item<float>() == 0) {
        return 0.0f;
    }
    
    return 1.0f - var_error.item<float>() / var_returns.item<float>();
}

// GeneralizedAdvantageEstimator implementation
GeneralizedAdvantageEstimator::GeneralizedAdvantageEstimator(float gamma, float lambda)
    : gamma_(gamma), lambda_(lambda) {}

torch::Tensor GeneralizedAdvantageEstimator::compute(torch::Tensor rewards,
                                                    torch::Tensor values,
                                                    torch::Tensor next_values,
                                                    torch::Tensor dones) {
    auto deltas = rewards + gamma_ * next_values * (1.0f - dones) - values;
    
    // Compute GAE using discount_cumsum
    auto advantages = discount_cumsum(deltas, gamma_ * lambda_);
    
    return advantages;
}

torch::Tensor GeneralizedAdvantageEstimator::compute_returns(torch::Tensor rewards,
                                                            torch::Tensor dones) {
    return discount_cumsum(rewards, gamma_);
}

torch::Tensor GeneralizedAdvantageEstimator::compute_vtrace(torch::Tensor rewards,
                                                           torch::Tensor values,
                                                           torch::Tensor next_values,
                                                           torch::Tensor log_rhos,
                                                           torch::Tensor dones) {
    // V-trace implementation
    auto rhos = torch::exp(log_rhos);
    auto clipped_rhos = torch::minimum(rhos, torch::ones_like(rhos));
    auto cs = torch::minimum(rhos, torch::ones_like(rhos));
    
    auto deltas = clipped_rhos * (rewards + gamma_ * next_values * (1.0f - dones) - values);
    
    int T = deltas.size(0);
    auto vs_minus_v_xs = torch::zeros_like(deltas);
    
    float acc = 0.0f;
    for (int t = T - 1; t >= 0; --t) {
        acc = deltas[t].item<float>() + gamma_ * cs[t].item<float>() * acc;
        vs_minus_v_xs[t] = acc;
    }
    
    return values + vs_minus_v_xs;
}

torch::Tensor GeneralizedAdvantageEstimator::compute_td_lambda(torch::Tensor rewards,
                                                              torch::Tensor values,
                                                              torch::Tensor next_values,
                                                              torch::Tensor dones) {
    auto td_errors = rewards + gamma_ * next_values * (1.0f - dones) - values;
    return discount_cumsum(td_errors, gamma_ * lambda_);
}

torch::Tensor GeneralizedAdvantageEstimator::discount_cumsum(torch::Tensor x, float discount) {
    int T = x.size(0);
    auto result = torch::zeros_like(x);
    
    float acc = 0.0f;
    for (int t = T - 1; t >= 0; --t) {
        acc = x[t].item<float>() + discount * acc;
        result[t] = acc;
    }
    
    return result;
}

// AdaptivePPO implementation
AdaptivePPO::AdaptivePPO(std::shared_ptr<model::DistributedTransformer> policy_model,
                        PPOConfig initial_config)
    : PPOTrainer(policy_model, initial_config) {}

PPOTrainer::TrainingMetrics AdaptivePPO::adaptive_train_step(Trajectory& trajectory) {
    // Regular training step
    auto metrics = train_step(trajectory);
    
    // Update history
    history_.update(metrics, trajectory.rewards.mean().item<float>());
    
    // Adaptive adjustments based on performance
    adjust_clip_epsilon(metrics.kl_divergence);
    adjust_learning_rate(metrics.policy_loss);
    adjust_entropy_coefficient(metrics.entropy_loss);
    
    return metrics;
}

void AdaptivePPO::adjust_clip_epsilon(float kl_divergence) {
    // Increase clip if KL is too high, decrease if too low
    if (kl_divergence > config_.target_kl * 1.5f) {
        config_.clip_epsilon = std::max(clip_epsilon_min_, 
                                       config_.clip_epsilon - clip_adapt_rate_);
    } else if (kl_divergence < config_.target_kl * 0.5f) {
        config_.clip_epsilon = std::min(clip_epsilon_max_, 
                                       config_.clip_epsilon + clip_adapt_rate_);
    }
}

void AdaptivePPO::adjust_learning_rate(float policy_loss) {
    // Adjust learning rate based on policy loss trend
    float trend = history_.get_trend();
    
    if (trend > 0 && policy_loss > 0.1f) {
        // Loss is increasing, reduce learning rate
        current_lr_ = std::max(lr_min_, current_lr_ * (1.0f - lr_adapt_rate_));
        update_learning_rate(0);  // Apply the new rate
    } else if (trend < -0.01f && policy_loss < 0.01f) {
        // Loss is decreasing well, can increase learning rate slightly
        current_lr_ = std::min(lr_max_, current_lr_ * (1.0f + lr_adapt_rate_ * 0.5f));
        update_learning_rate(0);
    }
}

void AdaptivePPO::adjust_entropy_coefficient(float entropy) {
    // Adjust entropy coefficient to maintain exploration
    float target_entropy = -std::log(1.0f / 1000.0f);  // Assuming vocab size of 1000
    
    if (entropy < target_entropy * 0.5f) {
        // Too little entropy, increase coefficient
        config_.entropy_coef = std::min(entropy_coef_max_, 
                                       config_.entropy_coef * (1.0f + entropy_adapt_rate_));
    } else if (entropy > target_entropy * 2.0f) {
        // Too much entropy, decrease coefficient
        config_.entropy_coef = std::max(entropy_coef_min_, 
                                       config_.entropy_coef * (1.0f - entropy_adapt_rate_));
    }
}

void AdaptivePPO::PerformanceHistory::update(const TrainingMetrics& metrics, float reward) {
    rewards.push_back(reward);
    kl_divergences.push_back(metrics.kl_divergence);
    policy_losses.push_back(metrics.policy_loss);
    value_losses.push_back(metrics.value_loss);
    
    // Keep only last 100 entries
    if (rewards.size() > 100) {
        rewards.erase(rewards.begin());
        kl_divergences.erase(kl_divergences.begin());
        policy_losses.erase(policy_losses.begin());
        value_losses.erase(value_losses.begin());
    }
}

float AdaptivePPO::PerformanceHistory::get_trend() const {
    if (policy_losses.size() < 10) return 0.0f;
    
    // Simple linear regression for trend
    float sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
    int n = policy_losses.size();
    
    for (int i = 0; i < n; ++i) {
        sum_x += i;
        sum_y += policy_losses[i];
        sum_xy += i * policy_losses[i];
        sum_xx += i * i;
    }
    
    float slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    return slope;
}

// RunningStat implementation
void PPOTrainer::RunningStat::update(float value) {
    count++;
    float delta = value - mean;
    mean += delta / count;
    float delta2 = value - mean;
    var += (delta * delta2 - var) / count;
}

float PPOTrainer::RunningStat::normalize(float value) const {
    return (value - mean) / (std::sqrt(var) + 1e-8f);
}

} // namespace rlhf
