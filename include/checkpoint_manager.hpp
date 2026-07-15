#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <chrono>
#include <filesystem>
#include <torch/torch.h>
#include "distributed_transformer.hpp"

namespace checkpointing {

// Checkpoint metadata
struct CheckpointMetadata {
    std::string checkpoint_id;
    std::string model_name;
    uint64_t timestamp;
    size_t epoch;
    size_t global_step;
    float loss;
    float learning_rate;
    std::unordered_map<std::string, float> metrics;
    std::unordered_map<std::string, std::string> hyperparameters;
    size_t file_size_bytes;
    std::string compression_type;
    std::vector<std::string> tags;
};

// Checkpoint manager configuration
struct CheckpointConfig {
    std::string base_directory = "./checkpoints";
    int max_checkpoints = 5;
    bool save_optimizer_state = true;
    bool save_training_state = true;
    bool use_compression = true;
    std::string compression_type = "zstd";  // "none", "gzip", "zstd", "lz4"
    bool async_save = true;
    int save_interval_steps = 1000;
    float save_interval_hours = 1.0f;
    bool save_on_improvement = true;
    std::string metric_to_monitor = "loss";
    bool minimize_metric = true;
};

// Checkpoint manager
class CheckpointManager {
public:
    CheckpointManager(CheckpointConfig config);
    ~CheckpointManager();
    
    // Save operations
    std::string save_checkpoint(std::shared_ptr<model::DistributedTransformer> model,
                               const CheckpointMetadata& metadata);
    
    void save_async(std::shared_ptr<model::DistributedTransformer> model,
                   const CheckpointMetadata& metadata);
    
    // Load operations
    void load_checkpoint(std::shared_ptr<model::DistributedTransformer> model,
                        const std::string& checkpoint_id);
    
    void load_latest(std::shared_ptr<model::DistributedTransformer> model);
    
    void load_best(std::shared_ptr<model::DistributedTransformer> model,
                  const std::string& metric_name = "");
    
    // Checkpoint management
    std::vector<CheckpointMetadata> list_checkpoints();
    void delete_checkpoint(const std::string& checkpoint_id);
    void cleanup_old_checkpoints();
    
    // Checkpoint selection
    CheckpointMetadata get_best_checkpoint(const std::string& metric_name);
    CheckpointMetadata get_latest_checkpoint();
    std::vector<CheckpointMetadata> get_checkpoints_by_tag(const std::string& tag);
    
    // Automatic checkpointing
    bool should_save(size_t current_step, float current_metric);
    void update_save_schedule(size_t current_step, float current_metric);
    
    // Export/Import
    void export_checkpoint(const std::string& checkpoint_id,
                         const std::string& export_path,
                         const std::string& format = "pytorch");  // "pytorch", "onnx", "tensorrt"
    
    void import_checkpoint(const std::string& import_path,
                         const std::string& format = "pytorch");
    
private:
    CheckpointConfig config_;
    std::vector<CheckpointMetadata> checkpoint_history_;
    
    // Best metric tracking
    float best_metric_value_;
    std::string best_checkpoint_id_;
    
    // Save scheduling
    size_t last_save_step_ = 0;
    std::chrono::steady_clock::time_point last_save_time_;
    
    // Async saving
    std::thread async_save_thread_;
    std::queue<std::pair<torch::serialize::OutputArchive, std::string>> save_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> stop_async_thread_{false};
    
    // Helper functions
    std::string generate_checkpoint_id();
    std::filesystem::path get_checkpoint_path(const std::string& checkpoint_id);
    void compress_checkpoint(const std::filesystem::path& path);
    void decompress_checkpoint(const std::filesystem::path& path);
    void save_metadata(const CheckpointMetadata& metadata, const std::filesystem::path& path);
    CheckpointMetadata load_metadata(const std::filesystem::path& path);
    
    // Async save worker
    void async_save_worker();
};

// Distributed checkpointing
class DistributedCheckpointManager : public CheckpointManager {
public:
    DistributedCheckpointManager(CheckpointConfig config,
                                int world_size,
                                int rank);
    
    // Distributed save/load
    std::string save_distributed(std::shared_ptr<model::DistributedTransformer> model,
                                const CheckpointMetadata& metadata);
    
    void load_distributed(std::shared_ptr<model::DistributedTransformer> model,
                         const std::string& checkpoint_id);
    
    // Sharded checkpointing
    void save_shard(std::shared_ptr<model::DistributedTransformer> model,
                   int shard_id);
    
    void load_shard(std::shared_ptr<model::DistributedTransformer> model,
                   int shard_id);
    
    // Consistency checking
    bool verify_checkpoint_consistency(const std::string& checkpoint_id);
    
private:
    int world_size_;
    int rank_;
    
    // Synchronization
    void synchronize_save();
    void synchronize_load();
    
    // Shard management
    std::string get_shard_path(const std::string& checkpoint_id, int shard_id);
    void merge_shards(const std::string& checkpoint_id);
};

// Incremental checkpointing
class IncrementalCheckpointManager : public CheckpointManager {
public:
    IncrementalCheckpointManager(CheckpointConfig config);
    
    // Incremental saves
    std::string save_incremental(std::shared_ptr<model::DistributedTransformer> model,
                                const CheckpointMetadata& metadata);
    
    void load_incremental(std::shared_ptr<model::DistributedTransformer> model,
                         const std::string& checkpoint_id);
    
    // Delta computation
    torch::serialize::OutputArchive compute_delta(
        std::shared_ptr<model::DistributedTransformer> model);
    
    void apply_delta(std::shared_ptr<model::DistributedTransformer> model,
                    const torch::serialize::InputArchive& delta);
    
private:
    std::string base_checkpoint_id_;
    std::shared_ptr<model::DistributedTransformer> base_model_state_;
    
    std::vector<std::string> delta_chain_;
    
    void update_base_checkpoint(std::shared_ptr<model::DistributedTransformer> model);
    void consolidate_deltas();
};

// Cloud checkpoint storage
class CloudCheckpointStorage {
public:
    CloudCheckpointStorage(const std::string& cloud_provider,
                          const std::string& bucket_name,
                          const std::string& credentials_path);
    
    // Cloud operations
    void upload_checkpoint(const std::string& local_path,
                         const std::string& cloud_path);
    
    void download_checkpoint(const std::string& cloud_path,
                           const std::string& local_path);
    
    std::vector<std::string> list_cloud_checkpoints();
    
    void delete_cloud_checkpoint(const std::string& cloud_path);
    
    // Sync operations
    void sync_to_cloud(const std::string& local_directory);
    void sync_from_cloud(const std::string& local_directory);
    
    // Multi-region replication
    void replicate_to_region(const std::string& checkpoint_path,
                           const std::string& target_region);
    
private:
    std::string cloud_provider_;  // "aws", "gcp", "azure"
    std::string bucket_name_;
    std::string credentials_path_;
    
    // Cloud provider interfaces
    void upload_to_s3(const std::string& local_path, const std::string& s3_path);
    void upload_to_gcs(const std::string& local_path, const std::string& gcs_path);
    void upload_to_azure(const std::string& local_path, const std::string& azure_path);
};

// Model versioning
class ModelVersionManager {
public:
    ModelVersionManager(const std::string& model_name);
    
    // Version management
    struct ModelVersion {
        std::string version_id;
        std::string version_tag;  // e.g., "v1.0.0"
        std::string checkpoint_id;
        std::string description;
        uint64_t timestamp;
        std::unordered_map<std::string, float> performance_metrics;
        std::vector<std::string> compatible_versions;
        bool is_production;
    };
    
    std::string create_version(const std::string& checkpoint_id,
                              const std::string& version_tag,
                              const std::string& description);
    
    void promote_to_production(const std::string& version_id);
    
    void rollback_production(const std::string& version_id);
    
    // Version queries
    ModelVersion get_production_version();
    std::vector<ModelVersion> list_versions();
    ModelVersion get_version(const std::string& version_id);
    
    // A/B testing support
    std::vector<ModelVersion> get_ab_test_versions();
    void set_ab_test_versions(const std::vector<std::string>& version_ids);
    
private:
    std::string model_name_;
    std::vector<ModelVersion> versions_;
    std::string production_version_id_;
    std::vector<std::string> ab_test_version_ids_;
    
    void save_version_metadata();
    void load_version_metadata();
};

} // namespace checkpointing
