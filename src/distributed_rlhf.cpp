#include "../include/distributed_rlhf.hpp"
#include "../include/utils.hpp"
#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>
#include <fstream>
#include <cmath>

namespace rlhf {

// DistributedRewardModel implementation
DistributedRewardModel::DistributedRewardModel(std::shared_ptr<DistributedTransformer> base_model,
                                             const RLHFConfig& config)
    : base_model_(base_model), config_(config) {
    
    std::cout << "Initializing Distributed Reward Model" << std::endl;
    std::cout << "- Reward model epochs: " << config_.reward_model_epochs << std::endl;
    std::cout << "- Reward model learning rate: " << config_.reward_model_lr << std::endl;
    std::cout << "- Batch size: " << config_.reward_model_batch_size << std::endl;

    // Initialize reward head (single linear layer on top of transformer)
    size_t hidden_size = base_model_->get_config().hidden_size;
    reward_head_weights_ = std::make_unique<Matrix>(hidden_size, 1);
    reward_head_bias_ = std::make_unique<Vector>(1);
    
    // Xavier initialization for reward head
    float scale = std::sqrt(2.0f / (hidden_size + 1));
    reward_head_weights_->initialize_random(scale);
    reward_head_bias_->initialize_constant(0.0f);
    
    // Initialize metrics
    metrics_ = RewardModelMetrics{};
    metrics_.last_updated = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

DistributedRewardModel::~DistributedRewardModel() = default;

bool DistributedRewardModel::train_on_preferences(const std::vector<PreferenceData>& preferences) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    
    if (preferences.empty()) {
        std::cerr << "No preference data provided for training" << std::endl;
        return false;
    }
    
    std::cout << "Training reward model on " << preferences.size() << " preference pairs" << std::endl;
    
    float total_loss = 0.0f;
    uint32_t correct_predictions = 0;
    
    // Process preferences in batches
    for (size_t i = 0; i < preferences.size(); i += config_.reward_model_batch_size) {
        size_t batch_end = std::min(i + config_.reward_model_batch_size, preferences.size());
        std::vector<PreferenceData> batch(preferences.begin() + i, preferences.begin() + batch_end);
        
        // Compute batch loss and gradients
        float batch_loss = 0.0f;
        std::vector<Matrix> batch_gradients;
        
        for (const auto& preference : batch) {
            // Compute rewards for both responses
            float reward_a = compute_reward(preference.prompt, preference.response_a);
            float reward_b = compute_reward(preference.prompt, preference.response_b);
            
            // Compute preference loss (Bradley-Terry model)
            float preference_diff = reward_b - reward_a;
            float sigmoid_diff = 1.0f / (1.0f + std::exp(-preference_diff));
            
            // Target is based on preference score
            float target = (preference.preference_score + 1.0f) / 2.0f; // Convert from [-1,1] to [0,1]
            
            // Binary cross-entropy loss
            float loss = -target * std::log(sigmoid_diff + 1e-8f) - 
                        (1.0f - target) * std::log(1.0f - sigmoid_diff + 1e-8f);
            
            batch_loss += loss;
            total_loss += loss;
            
            // Check if prediction is correct
            bool predicted_b_better = sigmoid_diff > 0.5f;
            bool actual_b_better = preference.preference_score > 0.0f;
            if (predicted_b_better == actual_b_better) {
                correct_predictions++;
            }
            
            // Compute gradients (simplified - would need full backprop in practice)
            float grad_scale = sigmoid_diff - target;
            
            // Update reward head weights (gradient descent step)
            Matrix grad_weights(reward_head_weights_->rows(), reward_head_weights_->cols());
            Vector grad_bias(reward_head_bias_->size());
            
            // Simplified gradient computation
            for (size_t j = 0; j < reward_head_weights_->rows(); ++j) {
                grad_weights(j, 0) = grad_scale * 0.01f; // Simplified
            }
            grad_bias[0] = grad_scale;
            
            // Apply gradients
            for (size_t j = 0; j < reward_head_weights_->size(); ++j) {
                reward_head_weights_->data()[j] -= config_.reward_model_lr * grad_weights.data()[j];
            }
            (*reward_head_bias_)[0] -= config_.reward_model_lr * grad_bias[0];
        }
        
        std::cout << "Batch " << (i / config_.reward_model_batch_size + 1) 
                  << " loss: " << (batch_loss / batch.size()) << std::endl;
    }
    
    // Update metrics
    metrics_.loss = total_loss / preferences.size();
    metrics_.accuracy = static_cast<float>(correct_predictions) / preferences.size();
    metrics_.training_samples += preferences.size();
    metrics_.last_updated = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    std::cout << "Reward model training completed:" << std::endl;
    std::cout << "- Loss: " << metrics_.loss << std::endl;
    std::cout << "- Accuracy: " << metrics_.accuracy << std::endl;
    
    return true;
}

float DistributedRewardModel::compute_reward(const std::string& prompt, const std::string& response) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    
    // Tokenize input
    auto tokenizer = base_model_->get_tokenizer();
    if (!tokenizer) {
        std::cerr << "No tokenizer available for reward computation" << std::endl;
        return 0.0f;
    }
    
    std::string full_input = prompt + " " + response;
    std::vector<int> tokens = tokenizer->encode(full_input);
    
    // Get hidden states from base model
    base_model_->set_training(false); // Inference mode
    auto output = base_model_->forward(tokens, full_input, *tokenizer);
    
    // Use the last hidden state for reward computation
    Matrix hidden_states = output.logits; // This would be hidden states in practice
    
    // Take the last token's hidden state
    Vector last_hidden(hidden_states.cols());
    for (size_t i = 0; i < hidden_states.cols(); ++i) {
        last_hidden[i] = hidden_states(hidden_states.rows() - 1, i);
    }
    
    // Compute reward using reward head
    float reward = 0.0f;
    for (size_t i = 0; i < reward_head_weights_->rows(); ++i) {
        reward += last_hidden[i] * (*reward_head_weights_)(i, 0);
    }
    reward += (*reward_head_bias_)[0];
    
    return reward;
}

std::vector<float> DistributedRewardModel::compute_batch_rewards(const std::vector<std::string>& prompts,
                                                               const std::vector<std::string>& responses) {
    if (prompts.size() != responses.size()) {
        std::cerr << "Prompts and responses must have the same size" << std::endl;
        return {};
    }
    
    std::vector<float> rewards;
    rewards.reserve(prompts.size());
    
    for (size_t i = 0; i < prompts.size(); ++i) {
        rewards.push_back(compute_reward(prompts[i], responses[i]));
    }
    
    return rewards;
}

RewardModelMetrics DistributedRewardModel::get_metrics() const {
    std::lock_guard<std::mutex> lock(model_mutex_);
    return metrics_;
}

// DistributedPPOTrainer implementation
DistributedPPOTrainer::DistributedPPOTrainer(std::shared_ptr<DistributedTransformer> policy_model,
                                           std::shared_ptr<DistributedTransformer> value_model,
                                           std::shared_ptr<DistributedRewardModel> reward_model,
                                           const RLHFConfig& config)
    : policy_model_(policy_model), value_model_(value_model), reward_model_(reward_model), config_(config) {
    
    std::cout << "Initializing Distributed PPO Trainer" << std::endl;
    std::cout << "- PPO epochs: " << config_.ppo_epochs << std::endl;
    std::cout << "- PPO learning rate: " << config_.ppo_lr << std::endl;
    std::cout << "- Clip ratio: " << config_.ppo_clip_ratio << std::endl;
    std::cout << "- KL target: " << config_.kl_target << std::endl;
    
    // Initialize metrics
    metrics_ = PPOMetrics{};
    current_kl_coeff_ = config_.kl_coeff;
}

DistributedPPOTrainer::~DistributedPPOTrainer() = default;

bool DistributedPPOTrainer::run_ppo_step(const std::vector<std::string>& prompts) {
    std::lock_guard<std::mutex> lock(trainer_mutex_);
    
    if (prompts.empty()) {
        std::cerr << "No prompts provided for PPO step" << std::endl;
        return false;
    }
    
    std::cout << "Running PPO step with " << prompts.size() << " prompts" << std::endl;
    
    // Clear experience buffer
    experience_buffer_.clear();
    experience_buffer_.reserve(prompts.size());
    
    // Generate responses and collect experience
    policy_model_->set_training(false); // Inference mode for generation
    value_model_->set_training(false);
    
    for (const auto& prompt : prompts) {
        // Generate response using policy model
        auto tokenizer = policy_model_->get_tokenizer();
        std::vector<int> prompt_tokens = tokenizer->encode(prompt);
        
        // Simple greedy generation (would use sampling in practice)
        auto output = policy_model_->forward(prompt_tokens, prompt, *tokenizer);
        
        // Convert logits to response (simplified)
        std::string response = "generated_response"; // Placeholder
        
        // Compute reward
        float reward = reward_model_->compute_reward(prompt, response);
        
        // Compute value estimate
        std::string full_input = prompt + " " + response;
        std::vector<int> full_tokens = tokenizer->encode(full_input);
        auto value_output = value_model_->forward(full_tokens, full_input, *tokenizer);
        float value = value_output.logits(value_output.logits.rows() - 1, 0); // Last token value
        
        // Compute log probability (simplified)
        float log_prob = -1.0f; // Placeholder
        
        // Store experience
        Experience exp;
        exp.prompt = prompt;
        exp.response = response;
        exp.reward = reward;
        exp.value = value;
        exp.log_prob = log_prob;
        exp.advantage = 0.0f; // Will be computed later
        
        experience_buffer_.push_back(exp);
    }
    
    // Compute advantages using GAE (Generalized Advantage Estimation)
    std::vector<float> rewards, values;
    for (const auto& exp : experience_buffer_) {
        rewards.push_back(exp.reward);
        values.push_back(exp.value);
    }
    
    auto advantages = compute_advantages(rewards, values);
    for (size_t i = 0; i < experience_buffer_.size(); ++i) {
        experience_buffer_[i].advantage = advantages[i];
    }
    
    // PPO training epochs
    policy_model_->set_training(true);
    value_model_->set_training(true);
    
    float total_policy_loss = 0.0f;
    float total_value_loss = 0.0f;
    float total_entropy_loss = 0.0f;
    float total_kl_div = 0.0f;
    
    for (uint32_t epoch = 0; epoch < config_.ppo_epochs; ++epoch) {
        // Shuffle experience buffer
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(experience_buffer_.begin(), experience_buffer_.end(), g);
        
        // Process in mini-batches
        for (size_t i = 0; i < experience_buffer_.size(); i += config_.ppo_batch_size) {
            size_t batch_end = std::min(i + config_.ppo_batch_size, experience_buffer_.size());
            
            float batch_policy_loss = 0.0f;
            float batch_value_loss = 0.0f;
            float batch_entropy_loss = 0.0f;
            float batch_kl_div = 0.0f;
            
            for (size_t j = i; j < batch_end; ++j) {
                const auto& exp = experience_buffer_[j];
                
                // Recompute log probabilities and values with current model
                auto tokenizer = policy_model_->get_tokenizer();
                std::string full_input = exp.prompt + " " + exp.response;
                std::vector<int> tokens = tokenizer->encode(full_input);
                
                auto policy_output = policy_model_->forward(tokens, full_input, *tokenizer);
                auto value_output = value_model_->forward(tokens, full_input, *tokenizer);
                
                float new_log_prob = -1.0f; // Simplified
                float new_value = value_output.logits(value_output.logits.rows() - 1, 0);
                
                // Compute losses
                float policy_loss = compute_policy_loss(exp, new_log_prob);
                float value_loss = compute_value_loss(exp, new_value);
                
                // Simplified entropy computation
                std::vector<float> log_probs = {new_log_prob};
                float entropy_loss = compute_entropy_loss(log_probs);
                
                // KL divergence
                float kl_div = std::abs(new_log_prob - exp.log_prob);
                
                batch_policy_loss += policy_loss;
                batch_value_loss += value_loss;
                batch_entropy_loss += entropy_loss;
                batch_kl_div += kl_div;
            }
            
            // Average batch losses
            size_t batch_size = batch_end - i;
            batch_policy_loss /= batch_size;
            batch_value_loss /= batch_size;
            batch_entropy_loss /= batch_size;
            batch_kl_div /= batch_size;
            
            total_policy_loss += batch_policy_loss;
            total_value_loss += batch_value_loss;
            total_entropy_loss += batch_entropy_loss;
            total_kl_div += batch_kl_div;
            
            // Apply gradients (simplified - would need actual backprop)
            // This is where you'd call the distributed gradient coordination
        }
        
        // Check for early stopping based on KL divergence
        float avg_kl = total_kl_div / (experience_buffer_.size() / config_.ppo_batch_size);
        if (should_early_stop(avg_kl)) {
            std::cout << "Early stopping PPO training due to high KL divergence: " << avg_kl << std::endl;
            break;
        }
    }
    
    // Update metrics
    size_t num_batches = (experience_buffer_.size() + config_.ppo_batch_size - 1) / config_.ppo_batch_size;
    metrics_.policy_loss = total_policy_loss / (config_.ppo_epochs * num_batches);
    metrics_.value_loss = total_value_loss / (config_.ppo_epochs * num_batches);
    metrics_.entropy_loss = total_entropy_loss / (config_.ppo_epochs * num_batches);
    metrics_.kl_divergence = total_kl_div / (config_.ppo_epochs * num_batches);
    
    // Compute reward statistics
    float reward_sum = 0.0f;
    float reward_sq_sum = 0.0f;
    for (const auto& exp : experience_buffer_) {
        reward_sum += exp.reward;
        reward_sq_sum += exp.reward * exp.reward;
    }
    metrics_.reward_mean = reward_sum / experience_buffer_.size();
    metrics_.reward_std = std::sqrt(reward_sq_sum / experience_buffer_.size() - 
                                   metrics_.reward_mean * metrics_.reward_mean);
    
    metrics_.iteration++;
    
    // Update KL coefficient
    update_kl_coefficient(metrics_.kl_divergence);
    
    std::cout << "PPO step completed:" << std::endl;
    std::cout << "- Policy loss: " << metrics_.policy_loss << std::endl;
    std::cout << "- Value loss: " << metrics_.value_loss << std::endl;
    std::cout << "- KL divergence: " << metrics_.kl_divergence << std::endl;
    std::cout << "- Reward mean: " << metrics_.reward_mean << std::endl;
    
    return true;
}

std::vector<float> DistributedPPOTrainer::compute_advantages(const std::vector<float>& rewards,
                                                           const std::vector<float>& values) {
    if (rewards.size() != values.size()) {
        std::cerr << "Rewards and values must have the same size" << std::endl;
        return {};
    }
    
    std::vector<float> advantages(rewards.size());
    
    // GAE (Generalized Advantage Estimation)
    const float gamma = 0.99f; // Discount factor
    const float lambda = 0.95f; // GAE parameter
    
    float gae = 0.0f;
    for (int i = static_cast<int>(rewards.size()) - 1; i >= 0; --i) {
        float delta = rewards[i] - values[i];
        if (i < static_cast<int>(rewards.size()) - 1) {
            delta += gamma * values[i + 1];
        }
        
        gae = delta + gamma * lambda * gae;
        advantages[i] = gae;
    }
    
    // Normalize advantages
    float mean = 0.0f;
    for (float adv : advantages) {
        mean += adv;
    }
    mean /= advantages.size();
    
    float std_dev = 0.0f;
    for (float adv : advantages) {
        std_dev += (adv - mean) * (adv - mean);
    }
    std_dev = std::sqrt(std_dev / advantages.size());
    
    if (std_dev > 1e-8f) {
        for (float& adv : advantages) {
            adv = (adv - mean) / std_dev;
        }
    }
    
    return advantages;
}

float DistributedPPOTrainer::compute_policy_loss(const Experience& exp, float new_log_prob) {
    // PPO clipped objective
    float ratio = std::exp(new_log_prob - exp.log_prob);
    float clipped_ratio = std::max(1.0f - config_.ppo_clip_ratio, 
                                  std::min(1.0f + config_.ppo_clip_ratio, ratio));
    
    float loss1 = ratio * exp.advantage;
    float loss2 = clipped_ratio * exp.advantage;
    
    return -std::min(loss1, loss2); // Negative because we want to maximize
}

float DistributedPPOTrainer::compute_value_loss(const Experience& exp, float new_value) {
    // MSE loss between predicted value and target (reward + discounted future value)
    float target = exp.reward; // Simplified - would include future rewards
    float diff = new_value - target;
    return 0.5f * diff * diff;
}

float DistributedPPOTrainer::compute_entropy_loss(const std::vector<float>& log_probs) {
    // Entropy regularization to encourage exploration
    float entropy = 0.0f;
    for (float log_prob : log_probs) {
        float prob = std::exp(log_prob);
        entropy -= prob * log_prob;
    }
    return -config_.ppo_entropy_coeff * entropy; // Negative to encourage high entropy
}

void DistributedPPOTrainer::update_kl_coefficient(float kl_divergence) {
    if (!config_.adaptive_kl) {
        return;
    }
    
    // Adaptive KL penalty coefficient
    if (kl_divergence > 2.0f * config_.kl_target) {
        current_kl_coeff_ *= 1.5f;
    } else if (kl_divergence < 0.5f * config_.kl_target) {
        current_kl_coeff_ *= 0.5f;
    }
    
    // Clamp to reasonable range
    current_kl_coeff_ = std::max(0.001f, std::min(1.0f, current_kl_coeff_));
}

bool DistributedPPOTrainer::should_early_stop(float kl_divergence) {
    return kl_divergence > 4.0f * config_.kl_target;
}

PPOMetrics DistributedPPOTrainer::get_metrics() const {
    std::lock_guard<std::mutex> lock(trainer_mutex_);
    return metrics_;
}

// DistributedRLHFCoordinator implementation
DistributedRLHFCoordinator::DistributedRLHFCoordinator(
    std::shared_ptr<p2p::P2PNetwork> network,
    std::shared_ptr<curation::DistributedCurationPlatform> curation,
    std::shared_ptr<DistributedTransformer> model,
    const RLHFConfig& config)
    : network_(network), curation_(curation), base_model_(model), config_(config) {
    
    std::cout << "Initializing Distributed RLHF Coordinator" << std::endl;
    std::cout << "- Consensus threshold: " << config_.consensus_threshold_percent << "%" << std::endl;
    std::cout << "- Gradient compression: " << (config_.enable_gradient_compression ? "enabled" : "disabled") << std::endl;
    
    // Create reward model
    reward_model_ = std::make_unique<DistributedRewardModel>(base_model_, config_);
    
    // Create value model (copy of base model for value estimation)
    value_model_ = std::make_shared<DistributedTransformer>(base_model_->get_config(), base_model_->get_tokenizer());
    
    // Create PPO trainer
    ppo_trainer_ = std::make_unique<DistributedPPOTrainer>(base_model_, value_model_, reward_model_.get(), config_);
    
    // Initialize stats
    current_stats_ = RLHFStats{};
    current_stats_.last_update_timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

DistributedRLHFCoordinator::~DistributedRLHFCoordinator() {
    stop();
}

bool DistributedRLHFCoordinator::start() {
    if (running_.load()) {
        std::cout << "RLHF coordinator already running" << std::endl;
        return true;
    }
    
    std::cout << "Starting distributed RLHF coordinator..." << std::endl;
    
    // Register P2P message handlers
    network_->register_message_handler(p2p::MessageType::RLHF_REWARD_MODEL_GRADIENT,
        [this](const p2p::NetworkMessage& msg) { handle_reward_model_gradient(msg); });
    
    network_->register_message_handler(p2p::MessageType::RLHF_PPO_GRADIENT,
        [this](const p2p::NetworkMessage& msg) { handle_ppo_gradient(msg); });
    
    network_->register_message_handler(p2p::MessageType::RLHF_PREFERENCE_DATA_SHARE,
        [this](const p2p::NetworkMessage& msg) { handle_preference_data_share(msg); });
    
    network_->register_message_handler(p2p::MessageType::RLHF_TRAINING_METRICS_UPDATE,
        [this](const p2p::NetworkMessage& msg) { handle_training_metrics_update(msg); });
    
    // Start background threads
    running_.store(true);
    worker_threads_.emplace_back(&DistributedRLHFCoordinator::consensus_coordination_thread, this);
    worker_threads_.emplace_back(&DistributedRLHFCoordinator::metrics_collection_thread, this);
    
    std::cout << "Distributed RLHF coordinator started successfully" << std::endl;
    return true;
}

void DistributedRLHFCoordinator::stop() {
    if (!running_.load()) {
        return;
    }
    
    std::cout << "Stopping distributed RLHF coordinator..." << std::endl;
    running_.store(false);
    
    // Wait for worker threads to finish
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();
    
    std::cout << "Distributed RLHF coordinator stopped" << std::endl;
}

bool DistributedRLHFCoordinator::run_reward_model_training(uint32_t num_epochs) {
    std::cout << "Starting distributed reward model training for " << num_epochs << " epochs" << std::endl;
    
    // Collect preference data from curation platform
    collect_preference_data_from_curation();
    
    if (preference_dataset_.empty()) {
        std::cerr << "No preference data available for training" << std::endl;
        return false;
    }
    
    std::cout << "Training on " << preference_dataset_.size() << " preference pairs" << std::endl;
    
    for (uint32_t epoch = 0; epoch < num_epochs; ++epoch) {
        std::cout << "Reward model training epoch " << (epoch + 1) << "/" << num_epochs << std::endl;
        
        // Train local reward model
        bool success = reward_model_->train_on_preferences(preference_dataset_);
        if (!success) {
            std::cerr << "Local reward model training failed" << std::endl;
            return false;
        }
        
        // Coordinate with other nodes for consensus
        if (!coordinate_reward_model_training()) {
            std::cerr << "Reward model consensus coordination failed" << std::endl;
            return false;
        }
        
        // Update metrics
        current_stats_.reward_model_metrics = reward_model_->get_metrics();
        
        // Trigger callback
        if (reward_model_callback_) {
            reward_model_callback_(current_stats_.reward_model_metrics);
        }
    }
    
    std::cout << "Reward model training completed successfully" << std::endl;
    return true;
}

bool DistributedRLHFCoordinator::run_ppo_training(uint32_t num_iterations) {
    std::cout << "Starting distributed PPO training for " << num_iterations << " iterations" << std::endl;
    
    // Generate prompts for PPO training
    std::vector<std::string> prompts = {
        "Explain the concept of artificial intelligence",
        "What are the benefits of renewable energy?",
        "How can we improve education systems?",
        "Describe the importance of biodiversity",
        "What makes a good leader?"
    };
    
    for (uint32_t iteration = 0; iteration < num_iterations; ++iteration) {
        std::cout << "PPO training iteration " << (iteration + 1) << "/" << num_iterations << std::endl;
        
        // Run local PPO step
        bool success = ppo_trainer_->run_ppo_step(prompts);
        if (!success) {
            std::cerr << "Local PPO training failed" << std::endl;
            return false;
        }
        
        // Coordinate with other nodes for consensus
        if (!coordinate_ppo_training()) {
            std::cerr << "PPO consensus coordination failed" << std::endl;
            return false;
        }
        
        // Update metrics
        current_stats_.ppo_metrics = ppo_trainer_->get_metrics();
        
        // Trigger callback
        if (ppo_step_callback_) {
            ppo_step_callback_(current_stats_.ppo_metrics);
        }
    }
    
    std::cout << "PPO training completed successfully" << std::endl;
    return true;
}

bool DistributedRLHFCoordinator::run_full_rlhf_pipeline(uint32_t reward_epochs, uint32_t ppo_iterations) {
    std::cout << "Starting full RLHF pipeline:" << std::endl;
    std::cout << "- Reward model epochs: " << reward_epochs << std::endl;
    std::cout << "- PPO iterations: " << ppo_iterations << std::endl;
    
    // Phase 1: Reward model training
    if (!run_reward_model_training(reward_epochs)) {
        std::cerr << "Reward model training phase failed" << std::endl;
        return false;
    }
    
    // Phase 2: PPO training
    if (!run_ppo_training(ppo_iterations)) {
        std::cerr << "PPO training phase failed" << std::endl;
        return false;
    }
    
    std::cout << "Full RLHF pipeline completed successfully" << std::endl;
    return true;
}

bool DistributedRLHFCoordinator::collect_preference_data(uint32_t target_samples) {
    std::cout << "Collecting " << target_samples << " preference data samples" << std::endl;
    
    // This would integrate with the curation platform to collect human preferences
    collect_preference_data_from_curation();
    
    if (preference_dataset_.size() >= target_samples) {
        std::cout << "Successfully collected " << preference_dataset_.size() << " preference samples" << std::endl;
        return true;
    } else {
        std::cout << "Only collected " << preference_dataset_.size() << " out of " << target_samples << " samples" << std::endl;
        return false;
    }
}

std::vector<PreferenceData> DistributedRLHFCoordinator::get_preference_dataset() {
    std::lock_guard<std::mutex> lock(dataset_mutex_);
    return preference_dataset_;
}

DistributedRLHFCoordinator::RLHFStats DistributedRLHFCoordinator::get_training_stats() {
    std::lock_guard<std::mutex> lock(coordinator_mutex_);
    return current_stats_;
}

// Private methods
void DistributedRLHFCoordinator::collect_preference_data_from_curation() {
    if (!curation_) {
        std::cerr << "No curation platform available" << std::endl;
        return;
    }
    
    // Get completed annotations from curation platform
    auto completed_annotations = curation_->get_completed_annotations(1000);
    
    std::lock_guard<std::mutex> lock(dataset_mutex_);
    
    for (const auto& annotation : completed_annotations) {
        // Convert annotations to preference data
        // This is a simplified conversion - in practice, you'd need more sophisticated logic
        
        PreferenceData preference;
        preference.prompt = "Evaluate this content"; // Simplified
        preference.response_a = annotation.task_id + "_response_a";
        preference.response_b = annotation.task_id + "_response_b";
        
        // Convert consensus labels to preference score
        float quality_score = 0.0f;
        for (const auto& label : annotation.consensus_labels) {
            if (label.label_type == "quality") {
                quality_score = label.score;
                break;
            }
        }
        
        preference.preference_score = (quality_score - 0.5f) * 2.0f; // Convert [0,1] to [-1,1]
        preference.annotator_id = "consensus";
        preference.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        preference.signature = "consensus_signature";
        
        preference_dataset_.push_back(preference);
    }
    
    current_stats_.total_preference_samples = preference_dataset_.size();
    
    std::cout << "Collected " << preference_dataset_.size() << " preference data samples from curation platform" << std::endl;
}

bool DistributedRLHFCoordinator::coordinate_reward_model_training() {
    // Get local gradients from reward model
    auto local_gradients = reward_model_->get_gradients();
    
    // Reach consensus with other nodes
    std::vector<Matrix> consensus_gradients;
    bool success = reach_gradient_consensus(local_gradients, consensus_gradients);
    
    if (success) {
        // Apply consensus gradients
        reward_model_->apply_gradients(consensus_gradients);
        current_stats_.consensus_success_rate = 1.0f; // Simplified
        return true;
    } else {
        current_stats_.consensus_success_rate = 0.0f;
        if (consensus_failed_callback_) {
            consensus_failed_callback_("Reward model gradient consensus failed");
        }
        return false;
    }
}

bool DistributedRLHFCoordinator::coordinate_ppo_training() {
    // Get local gradients from PPO trainer
    auto [policy_gradients, value_gradients] = ppo_trainer_->get_gradients();
    
    // Reach consensus for both policy and value gradients
    std::vector<Matrix> consensus_policy_gradients, consensus_value_gradients;
    
    bool policy_success = reach_gradient_consensus(policy_gradients, consensus_policy_gradients);
    bool value_success = reach_gradient_consensus(value_gradients, consensus_value_gradients);
    
    if (policy_success && value_success) {
        // Apply consensus gradients
        ppo_trainer_->apply_gradients(consensus_policy_gradients, consensus_value_gradients);
        current_stats_.consensus_success_rate = 1.0f;
        return true;
    } else {
        current_stats_.consensus_success_rate = 0.0f;
        if (consensus_failed_callback_) {
            consensus_failed_callback_("PPO gradient consensus failed");
        }
        return false;
    }
}

bool DistributedRLHFCoordinator::reach_gradient_consensus(const std::vector<Matrix>& local_gradients,
                                                        std::vector<Matrix>& consensus_gradients) {
    // This would implement the actual BFT consensus protocol for gradients
    // For now, just return the local gradients as consensus
    consensus_gradients = local_gradients;
    return true;
}

// Background threads
void DistributedRLHFCoordinator::consensus_coordination_thread() {
    std::cout << "Started consensus coordination thread" << std::endl;
    
    while (running_.load()) {
        // Periodically check for consensus opportunities
        // This would handle incoming gradient proposals and coordinate responses
        
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    
    std::cout << "Consensus coordination thread stopped" << std::endl;
}

void DistributedRLHFCoordinator::metrics_collection_thread() {
    std::cout << "Started metrics collection thread" << std::endl;
    
    while (running_.load()) {
        // Update training statistics
        {
            std::lock_guard<std::mutex> lock(coordinator_mutex_);
            
            current_stats_.reward_model_metrics = reward_model_->get_metrics();
            current_stats_.ppo_metrics = ppo_trainer_->get_metrics();
            current_stats_.active_training_nodes = network_->get_peer_count();
            current_stats_.last_update_timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
        }
        
        std::this_thread::sleep_for(std::chrono::seconds(10));
    }
    
    std::cout << "Metrics collection thread stopped" << std::endl;
}

// P2P message handlers (simplified implementations)
void DistributedRLHFCoordinator::handle_reward_model_gradient(const p2p::NetworkMessage& message) {
    std::cout << "Received reward model gradient from " << message.sender_id << std::endl;
    // Handle incoming gradient proposals for reward model consensus
}

void DistributedRLHFCoordinator::handle_ppo_gradient(const p2p::NetworkMessage& message) {
    std::cout << "Received PPO gradient from " << message.sender_id << std::endl;
    // Handle incoming gradient proposals for PPO consensus
}

void DistributedRLHFCoordinator::handle_preference_data_share(const p2p::NetworkMessage& message) {
    std::cout << "Received preference data share from " << message.sender_id << std::endl;
    // Handle shared preference data from other nodes
}

void DistributedRLHFCoordinator::handle_training_metrics_update(const p2p::NetworkMessage& message) {
    std::cout << "Received training metrics update from " << message.sender_id << std::endl;
    // Handle training metrics updates from other nodes
}

// Callback setters
void DistributedRLHFCoordinator::set_reward_model_updated_callback(RewardModelUpdatedCallback callback) {
    reward_model_callback_ = callback;
}

void DistributedRLHFCoordinator::set_ppo_step_completed_callback(PPOStepCompletedCallback callback) {
    ppo_step_callback_ = callback;
}

void DistributedRLHFCoordinator::set_consensus_failed_callback(ConsensusFailedCallback callback) {
    consensus_failed_callback_ = callback;
}

// Utility function implementations
namespace utils {

std::vector<PreferenceData> load_preference_data(const std::string& file_path) {
    std::vector<PreferenceData> data;
    std::ifstream file(file_path);
    
    if (!file.is_open()) {
        std::cerr << "Cannot open preference data file: " << file_path << std::endl;
        return data;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // Parse preference data from line (simplified format)
        // Format: prompt|response_a|response_b|preference_score|annotator_id
        
        std::istringstream iss(line);
        std::string prompt, response_a, response_b, score_str, annotator_id;
        
        if (std::getline(iss, prompt, '|') &&
            std::getline(iss, response_a, '|') &&
            std::getline(iss, response_b, '|') &&
            std::getline(iss, score_str, '|') &&
            std::getline(iss, annotator_id)) {
            
            PreferenceData preference;
            preference.prompt = prompt;
            preference.response_a = response_a;
            preference.response_b = response_b;
            preference.preference_score = std::stof(score_str);
            preference.annotator_id = annotator_id;
            preference.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            preference.signature = "file_signature";
            
            data.push_back(preference);
        }
    }
    
    std::cout << "Loaded " << data.size() << " preference data samples from " << file_path << std::endl;
    return data;
}

bool save_preference_data(const std::vector<PreferenceData>& data, const std::string& file_path) {
    std::ofstream file(file_path);
    
    if (!file.is_open()) {
        std::cerr << "Cannot open file for writing: " << file_path << std::endl;
        return false;
    }
    
    for (const auto& preference : data) {
        file << preference.prompt << "|"
             << preference.response_a << "|"
             << preference.response_b << "|"
             << preference.preference_score << "|"
             << preference.annotator_id << "\n";
    }
    
    std::cout << "Saved " << data.size() << " preference data samples to " << file_path << std::endl;
    return true;
}

bool contains_unsafe_content(const std::string& text, const std::vector<std::string>& safety_keywords) {
    std::string lower_text = text;
    std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);
    
    for (const auto& keyword : safety_keywords) {
        std::string lower_keyword = keyword;
        std::transform(lower_keyword.begin(), lower_keyword.end(), lower_keyword.begin(), ::tolower);
        
        if (lower_text.find(lower_keyword) != std::string::npos) {
            return true;
        }
    }
    
    return false;
}

std::string sanitize_response(const std::string& response) {
    // Basic sanitization - remove potentially harmful content
    std::string sanitized = response;
    
    // Remove excessive whitespace
    sanitized = std::regex_replace(sanitized, std::regex("\\s+"), " ");
    
    // Trim
    sanitized.erase(0, sanitized.find_first_not_of(" \t\n\r"));
    sanitized.erase(sanitized.find_last_not_of(" \t\n\r") + 1);
    
    return sanitized;
}

} // namespace utils

} // namespace rlhf
