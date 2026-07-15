#pragma once

#include <vector>
#include <memory>
#include <unordered_map>
#include <torch/torch.h>
#include <nccl.h>
#include "distributed_transformer.hpp"
#include "kademlia_dht.hpp"

namespace rlhf {

// Reward model configuration
struct RewardModelConfig {
    int hidden_dim = 768;
    int num_layers = 12;
    int num_heads = 12;
    int intermediate_dim = 3072;
    int max_seq_length = 512;
    
    // Sharding configuration
    int num_shards = 4;
    bool enable_tensor_parallelism = true;
    bool enable_pipeline_parallelism = false;
    int pipeline_stages = 2;
    
    // Optimization
    bool use_gradient_checkpointing = false;
    bool use_mixed_precision = false;
    float gradient_accumulation_steps = 1;
};

// Shard information
struct ShardInfo {
    int shard_id;
    int start_layer;
    int end_layer;
    std::vector<int> parameter_indices;
    size_t memory_usage_bytes;
    torch::Device device;
    std::string node_id;  // P2P network node ID
};

// Reward model shard
class RewardModelShard : public torch::nn::Module {
public:
    RewardModelShard(RewardModelConfig config, ShardInfo shard_info);
    
    // Forward pass for this shard
    torch::Tensor forward(torch::Tensor input_embeddings,
                         torch::Tensor attention_mask = {});
    
    // Get shard information
    ShardInfo get_shard_info() const { return shard_info_; }
    size_t get_memory_usage() const;
    
    // Gradient accumulation
    void accumulate_gradients(const torch::Tensor& gradients);
    torch::Tensor get_accumulated_gradients();
    void reset_accumulated_gradients();
    
private:
    RewardModelConfig config_;
    ShardInfo shard_info_;
    
    // Transformer layers for this shard
    std::vector<std::shared_ptr<model::TransformerLayer>> layers_;
    
    // Output projection (only for last shard)
    torch::nn::Linear output_projection_{nullptr};
    torch::nn::LayerNorm final_norm_{nullptr};
    
    // Gradient accumulation
    std::vector<torch::Tensor> accumulated_gradients_;
    int accumulation_steps_ = 0;
};

// Distributed reward model with sharding
class DistributedRewardModel {
public:
    DistributedRewardModel(RewardModelConfig config,
                          std::shared_ptr<p2p::KademliaDHT> dht,
                          int world_size,
                          int rank);
    ~DistributedRewardModel();
    
    // Training
    struct RewardPrediction {
        torch::Tensor rewards;        // [batch_size]
        torch::Tensor confidence;     // [batch_size]
        torch::Tensor hidden_states;  // [batch_size, hidden_dim]
    };
    
    RewardPrediction forward(torch::Tensor input_ids,
                            torch::Tensor attention_mask = {});
    
    // Backward pass with gradient sharding
    void backward(torch::Tensor loss);
    
    // Optimizer step with distributed updates
    void step(float learning_rate);
    
    // Sharding management
    void redistribute_shards();
    std::vector<ShardInfo> get_shard_distribution() const;
    
    // Load balancing
    void balance_shards_by_memory();
    void balance_shards_by_compute();
    
    // Checkpointing
    void save_shard(int shard_id, const std::string& path);
    void load_shard(int shard_id, const std::string& path);
    void save_full_model(const std::string& path);
    void load_full_model(const std::string& path);
    
    // Ensemble methods
    RewardPrediction ensemble_forward(torch::Tensor input_ids,
                                     std::vector<int> model_indices);
    
private:
    RewardModelConfig config_;
    std::shared_ptr<p2p::KademliaDHT> dht_;
    int world_size_;
    int rank_;
    
    // Local shards
    std::vector<std::unique_ptr<RewardModelShard>> local_shards_;
    std::unordered_map<int, ShardInfo> shard_registry_;
    
    // Communication
    ncclComm_t nccl_comm_;
    std::vector<cudaStream_t> cuda_streams_;
    
    // Pipeline parallelism
    struct PipelineStage {
        int stage_id;
        torch::Tensor activation_buffer;
        torch::Tensor gradient_buffer;
        cudaEvent_t forward_event;
        cudaEvent_t backward_event;
    };
    std::vector<PipelineStage> pipeline_stages_;
    
    // Helper functions
    torch::Tensor all_gather_tensor(torch::Tensor local_tensor);
    void all_reduce_gradients();
    torch::Tensor route_to_shard(torch::Tensor input, int target_shard);
    
    // Load balancing algorithms
    float compute_shard_load(int shard_id);
    void migrate_shard(int shard_id, int target_rank);
};

// Ensemble reward model
class EnsembleRewardModel {
public:
    EnsembleRewardModel(std::vector<std::shared_ptr<DistributedRewardModel>> models);
    
    // Ensemble prediction
    struct EnsemblePrediction {
        torch::Tensor mean_reward;
        torch::Tensor std_reward;
        torch::Tensor min_reward;
        torch::Tensor max_reward;
        std::vector<torch::Tensor> individual_rewards;
    };
    
    EnsemblePrediction predict(torch::Tensor input_ids,
                              torch::Tensor attention_mask = {});
    
    // Uncertainty estimation
    torch::Tensor compute_epistemic_uncertainty(torch::Tensor input_ids);
    torch::Tensor compute_aleatoric_uncertainty(torch::Tensor input_ids);
    
    // Active learning
    std::vector<int> select_uncertain_samples(torch::Tensor input_ids,
                                             int num_samples);
    
    // Model selection
    void prune_weak_models(float threshold);
    void add_model(std::shared_ptr<DistributedRewardModel> model);
    
private:
    std::vector<std::shared_ptr<DistributedRewardModel>> models_;
    
    // Weighting for ensemble
    std::vector<float> model_weights_;
    
    // Statistics
    std::vector<float> model_performance_;
    
    void update_model_weights();
};

// Hierarchical reward model
class HierarchicalRewardModel {
public:
    HierarchicalRewardModel(RewardModelConfig base_config);
    
    // Multi-level rewards
    struct HierarchicalReward {
        torch::Tensor token_level;     // [batch_size, seq_len]
        torch::Tensor sentence_level;  // [batch_size, num_sentences]
        torch::Tensor document_level;  // [batch_size]
        torch::Tensor aspect_scores;   // [batch_size, num_aspects]
    };
    
    HierarchicalReward compute_hierarchical_rewards(torch::Tensor input_ids,
                                                    torch::Tensor attention_mask = {});
    
    // Aspect-based rewards
    void register_aspect(const std::string& aspect_name,
                        std::shared_ptr<RewardModelShard> aspect_model);
    
    torch::Tensor compute_aspect_reward(const std::string& aspect_name,
                                       torch::Tensor input_ids);
    
private:
    RewardModelConfig base_config_;
    
    // Hierarchical models
    std::unique_ptr<DistributedRewardModel> token_model_;
    std::unique_ptr<DistributedRewardModel> sentence_model_;
    std::unique_ptr<DistributedRewardModel> document_model_;
    
    // Aspect models
    std::unordered_map<std::string, std::shared_ptr<RewardModelShard>> aspect_models_;
    
    // Aggregation functions
    torch::Tensor aggregate_token_to_sentence(torch::Tensor token_rewards,
                                             torch::Tensor sentence_boundaries);
    torch::Tensor aggregate_sentence_to_document(torch::Tensor sentence_rewards);
};

// Dynamic reward model with online learning
class DynamicRewardModel : public DistributedRewardModel {
public:
    DynamicRewardModel(RewardModelConfig config,
                      std::shared_ptr<p2p::KademliaDHT> dht,
                      int world_size,
                      int rank);
    
    // Online learning
    void update_with_feedback(torch::Tensor input_ids,
                             torch::Tensor human_rewards,
                             float learning_rate = 1e-4f);
    
    // Adaptive sharding
    void adapt_sharding_to_workload(const std::vector<torch::Tensor>& recent_inputs);
    
    // Meta-learning
    void meta_update(const std::vector<std::pair<torch::Tensor, torch::Tensor>>& support_set,
                     const std::vector<std::pair<torch::Tensor, torch::Tensor>>& query_set);
    
    // Continual learning with elastic weight consolidation
    void consolidate_weights();
    void update_with_ewc_penalty(torch::Tensor loss, float ewc_lambda = 0.5f);
    
private:
    // Fisher information for EWC
    std::unordered_map<std::string, torch::Tensor> fisher_information_;
    std::unordered_map<std::string, torch::Tensor> optimal_weights_;
    
    // Meta-learning parameters
    std::unique_ptr<torch::optim::Adam> meta_optimizer_;
    float meta_learning_rate_ = 1e-3f;
    
    // Workload statistics
    struct WorkloadStats {
        std::vector<float> shard_utilization;
        std::vector<float> communication_overhead;
        float average_batch_size;
        float average_sequence_length;
    };
    WorkloadStats workload_stats_;
    
    void update_workload_stats(const torch::Tensor& input);
    void optimize_sharding_plan();
};

// Federated reward model
class FederatedRewardModel {
public:
    FederatedRewardModel(RewardModelConfig config,
                        int num_clients);
    
    // Federated learning round
    void federated_round(const std::vector<torch::Tensor>& client_gradients,
                        const std::vector<float>& client_weights);
    
    // Secure aggregation
    torch::Tensor secure_aggregate(const std::vector<torch::Tensor>& encrypted_gradients);
    
    // Differential privacy
    void add_differential_privacy_noise(torch::Tensor& gradients,
                                       float epsilon = 1.0f,
                                       float delta = 1e-5f);
    
    // Client selection
    std::vector<int> select_clients(int num_clients_to_select,
                                   const std::vector<float>& client_scores);
    
private:
    RewardModelConfig config_;
    int num_clients_;
    
    std::unique_ptr<DistributedRewardModel> global_model_;
    
    // Privacy budget
    float privacy_budget_epsilon_ = 10.0f;
    float privacy_budget_delta_ = 1e-5f;
    float consumed_privacy_budget_ = 0.0f;
    
    // Secure multi-party computation
    std::vector<torch::Tensor> secret_share_gradient(torch::Tensor gradient,
                                                     int num_shares);
    torch::Tensor reconstruct_from_shares(const std::vector<torch::Tensor>& shares);
};

} // namespace rlhf
