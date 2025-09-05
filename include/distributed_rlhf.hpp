#pragma once

#include "distributed_transformer.hpp"
#include "distributed_curation.hpp"
#include "p2p_network.hpp"
#include "matrix.hpp"
#include <memory>
#include <vector>
#include <functional>
#include <atomic>
#include <mutex>

namespace rlhf {

// RLHF-specific data structures
struct PreferenceData {
    std::string prompt;
    std::string response_a;
    std::string response_b;
    float preference_score;  // -1.0 (strongly prefer A) to 1.0 (strongly prefer B)
    std::string annotator_id;
    uint64_t timestamp;
    std::string signature;
};

struct RLHFTrainingBatch {
    std::vector<std::string> prompts;
    std::vector<std::string> responses;
    std::vector<float> rewards;
    std::vector<float> advantages;
    std::vector<float> old_log_probs;
    uint32_t batch_id;
    uint64_t timestamp;
};

struct RewardModelMetrics {
    float accuracy;
    float loss;
    float preference_agreement;
    uint32_t training_samples;
    uint64_t last_updated;
};

struct PPOMetrics {
    float policy_loss;
    float value_loss;
    float entropy_loss;
    float kl_divergence;
    float reward_mean;
    float reward_std;
    uint32_t iteration;
};

// Configuration for RLHF training
struct RLHFConfig {
    // Reward model training
    uint32_t reward_model_epochs = 3;
    float reward_model_lr = 1e-5f;
    uint32_t reward_model_batch_size = 32;
    float reward_model_weight_decay = 0.01f;

    // PPO training
    uint32_t ppo_epochs = 4;
    float ppo_lr = 1e-6f;
    uint32_t ppo_batch_size = 16;
    float ppo_clip_ratio = 0.2f;
    float ppo_entropy_coeff = 0.01f;
    float ppo_value_coeff = 0.5f;
    float ppo_max_grad_norm = 1.0f;

    // KL penalty
    float kl_target = 0.01f;
    float kl_coeff = 0.1f;
    bool adaptive_kl = true;

    // Distributed training
    uint32_t consensus_threshold_percent = 67;
    uint32_t max_gradient_staleness = 3;
    bool enable_gradient_compression = true;
    float compression_ratio = 0.1f;  // Top 10% of gradients

    // Safety and quality
    float max_response_length = 512;
    float min_reward_model_confidence = 0.7f;
    bool enable_safety_filtering = true;
    std::vector<std::string> safety_keywords;
};

// Reward model for preference learning
class DistributedRewardModel {
public:
    DistributedRewardModel(std::shared_ptr<DistributedTransformer> base_model,
                          const RLHFConfig& config);
    ~DistributedRewardModel();

    // Training
    bool train_on_preferences(const std::vector<PreferenceData>& preferences);
    bool update_with_consensus_gradients(const std::vector<Matrix>& gradients);

    // Inference
    float compute_reward(const std::string& prompt, const std::string& response);
    std::vector<float> compute_batch_rewards(const std::vector<std::string>& prompts,
                                           const std::vector<std::string>& responses);

    // Model management
    bool save_model(const std::string& path);
    bool load_model(const std::string& path);
    RewardModelMetrics get_metrics() const;

    // Distributed coordination
    std::vector<Matrix> get_gradients() const;
    void apply_gradients(const std::vector<Matrix>& gradients);
    bool synchronize_with_peers();

private:
    std::shared_ptr<DistributedTransformer> base_model_;
    RLHFConfig config_;
    
    // Reward head (additional layers on top of transformer)
    std::unique_ptr<Matrix> reward_head_weights_;
    std::unique_ptr<Vector> reward_head_bias_;
    
    // Training state
    RewardModelMetrics metrics_;
    std::vector<Matrix> cached_gradients_;
    
    // Thread safety
    mutable std::mutex model_mutex_;
    
    // Internal methods
    Matrix compute_reward_logits(const Matrix& hidden_states);
    float compute_preference_loss(const PreferenceData& preference);
    void update_metrics(const std::vector<PreferenceData>& batch);
};

// PPO trainer for policy optimization
class DistributedPPOTrainer {
public:
    DistributedPPOTrainer(std::shared_ptr<DistributedTransformer> policy_model,
                         std::shared_ptr<DistributedTransformer> value_model,
                         std::shared_ptr<DistributedRewardModel> reward_model,
                         const RLHFConfig& config);
    ~DistributedPPOTrainer();

    // Training step
    bool run_ppo_step(const std::vector<std::string>& prompts);
    bool update_with_consensus_gradients(const std::vector<Matrix>& policy_gradients,
                                        const std::vector<Matrix>& value_gradients);

    // Generation and evaluation
    std::vector<std::string> generate_responses(const std::vector<std::string>& prompts);
    std::vector<float> compute_advantages(const std::vector<float>& rewards,
                                        const std::vector<float>& values);

    // Metrics and monitoring
    PPOMetrics get_metrics() const;
    void reset_metrics();

    // Distributed coordination
    std::pair<std::vector<Matrix>, std::vector<Matrix>> get_gradients() const;
    void apply_gradients(const std::vector<Matrix>& policy_gradients,
                        const std::vector<Matrix>& value_gradients);

private:
    std::shared_ptr<DistributedTransformer> policy_model_;
    std::shared_ptr<DistributedTransformer> value_model_;
    std::shared_ptr<DistributedRewardModel> reward_model_;
    RLHFConfig config_;

    // Training state
    PPOMetrics metrics_;
    float current_kl_coeff_;
    
    // Experience buffer
    struct Experience {
        std::string prompt;
        std::string response;
        float reward;
        float value;
        float log_prob;
        float advantage;
    };
    std::vector<Experience> experience_buffer_;

    // Thread safety
    mutable std::mutex trainer_mutex_;

    // Internal methods
    float compute_policy_loss(const Experience& exp, float new_log_prob);
    float compute_value_loss(const Experience& exp, float new_value);
    float compute_entropy_loss(const std::vector<float>& log_probs);
    void update_kl_coefficient(float kl_divergence);
    bool should_early_stop(float kl_divergence);
};

// Main RLHF coordinator that integrates everything
class DistributedRLHFCoordinator {
public:
    DistributedRLHFCoordinator(std::shared_ptr<p2p::P2PNetwork> network,
                              std::shared_ptr<curation::DistributedCurationPlatform> curation,
                              std::shared_ptr<DistributedTransformer> model,
                              const RLHFConfig& config = RLHFConfig{});
    ~DistributedRLHFCoordinator();

    // Lifecycle
    bool start();
    void stop();
    bool is_running() const { return running_.load(); }

    // Training phases
    bool run_reward_model_training(uint32_t num_epochs);
    bool run_ppo_training(uint32_t num_iterations);
    bool run_full_rlhf_pipeline(uint32_t reward_epochs, uint32_t ppo_iterations);

    // Data management
    bool collect_preference_data(uint32_t target_samples);
    std::vector<PreferenceData> get_preference_dataset();
    bool validate_preference_data(const PreferenceData& data);

    // Monitoring and evaluation
    struct RLHFStats {
        RewardModelMetrics reward_model_metrics;
        PPOMetrics ppo_metrics;
        uint32_t total_preference_samples;
        uint32_t active_training_nodes;
        float consensus_success_rate;
        uint64_t last_update_timestamp;
    };
    RLHFStats get_training_stats();

    // Event callbacks
    using RewardModelUpdatedCallback = std::function<void(const RewardModelMetrics&)>;
    using PPOStepCompletedCallback = std::function<void(const PPOMetrics&)>;
    using ConsensusFailedCallback = std::function<void(const std::string&)>;

    void set_reward_model_updated_callback(RewardModelUpdatedCallback callback);
    void set_ppo_step_completed_callback(PPOStepCompletedCallback callback);
    void set_consensus_failed_callback(ConsensusFailedCallback callback);

private:
    std::shared_ptr<p2p::P2PNetwork> network_;
    std::shared_ptr<curation::DistributedCurationPlatform> curation_;
    std::shared_ptr<DistributedTransformer> base_model_;
    RLHFConfig config_;

    // RLHF components
    std::unique_ptr<DistributedRewardModel> reward_model_;
    std::unique_ptr<DistributedPPOTrainer> ppo_trainer_;
    std::shared_ptr<DistributedTransformer> value_model_;

    // Training state
    std::atomic<bool> running_{false};
    std::vector<PreferenceData> preference_dataset_;
    RLHFStats current_stats_;

    // Thread management
    std::vector<std::thread> worker_threads_;
    void consensus_coordination_thread();
    void metrics_collection_thread();

    // Thread safety
    mutable std::mutex coordinator_mutex_;
    mutable std::mutex dataset_mutex_;

    // P2P message handlers
    void handle_reward_model_gradient(const p2p::NetworkMessage& message);
    void handle_ppo_gradient(const p2p::NetworkMessage& message);
    void handle_preference_data_share(const p2p::NetworkMessage& message);
    void handle_training_metrics_update(const p2p::NetworkMessage& message);

    // Consensus coordination
    bool coordinate_reward_model_training();
    bool coordinate_ppo_training();
    bool reach_gradient_consensus(const std::vector<Matrix>& local_gradients,
                                std::vector<Matrix>& consensus_gradients);

    // Data collection and validation
    void collect_preference_data_from_curation();
    bool validate_and_filter_preferences();
    void share_preference_data_with_peers();

    // Callbacks
    RewardModelUpdatedCallback reward_model_callback_;
    PPOStepCompletedCallback ppo_step_callback_;
    ConsensusFailedCallback consensus_failed_callback_;
};

// Utility functions for RLHF
namespace utils {
    // Data processing
    std::vector<PreferenceData> load_preference_data(const std::string& file_path);
    bool save_preference_data(const std::vector<PreferenceData>& data, const std::string& file_path);
    
    // Evaluation metrics
    float compute_reward_model_accuracy(const DistributedRewardModel& model,
                                      const std::vector<PreferenceData>& test_data);
    float compute_policy_improvement(const std::vector<float>& old_rewards,
                                   const std::vector<float>& new_rewards);

    // Safety and filtering
    bool contains_unsafe_content(const std::string& text, 
                               const std::vector<std::string>& safety_keywords);
    std::string sanitize_response(const std::string& response);

    // Serialization
    std::vector<uint8_t> serialize_preference_data(const PreferenceData& data);
    PreferenceData deserialize_preference_data(const std::vector<uint8_t>& bytes);
    
    std::vector<uint8_t> serialize_rlhf_batch(const RLHFTrainingBatch& batch);
    RLHFTrainingBatch deserialize_rlhf_batch(const std::vector<uint8_t>& bytes);
}

} // namespace rlhf
