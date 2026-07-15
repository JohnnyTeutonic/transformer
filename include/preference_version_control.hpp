#pragma once

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <chrono>
#include <sqlite3.h>
#include <torch/torch.h>
#include "git2.h"

namespace rlhf {

// Preference data entry
struct PreferenceEntry {
    std::string id;
    std::string prompt;
    std::string chosen_response;
    std::string rejected_response;
    float preference_strength;  // 0.0 to 1.0
    std::string annotator_id;
    uint64_t timestamp;
    std::unordered_map<std::string, std::string> metadata;
};

// Version information
struct Version {
    std::string version_id;
    std::string parent_version_id;
    std::string commit_message;
    std::string author;
    uint64_t timestamp;
    size_t num_entries;
    std::vector<std::string> tags;
};

// Diff between versions
struct VersionDiff {
    std::vector<PreferenceEntry> added;
    std::vector<PreferenceEntry> removed;
    std::vector<std::pair<PreferenceEntry, PreferenceEntry>> modified;
    std::unordered_map<std::string, std::pair<std::string, std::string>> metadata_changes;
};

// Preference data version control system
class PreferenceVersionControl {
public:
    PreferenceVersionControl(const std::string& repository_path);
    ~PreferenceVersionControl();
    
    // Version management
    std::string commit(const std::vector<PreferenceEntry>& entries,
                      const std::string& message,
                      const std::string& author);
    
    void checkout(const std::string& version_id);
    std::string create_branch(const std::string& branch_name);
    void merge_branch(const std::string& source_branch,
                     const std::string& target_branch = "main");
    
    // Data operations
    void add_entries(const std::vector<PreferenceEntry>& entries);
    void remove_entries(const std::vector<std::string>& entry_ids);
    void update_entry(const PreferenceEntry& entry);
    
    // Query operations
    std::vector<PreferenceEntry> get_entries(const std::string& version_id = "HEAD");
    PreferenceEntry get_entry(const std::string& entry_id,
                            const std::string& version_id = "HEAD");
    
    // Version history
    std::vector<Version> get_history(int limit = 100);
    VersionDiff diff(const std::string& version1, const std::string& version2);
    
    // Filtering and searching
    std::vector<PreferenceEntry> filter_by_annotator(const std::string& annotator_id);
    std::vector<PreferenceEntry> filter_by_date_range(uint64_t start_time, uint64_t end_time);
    std::vector<PreferenceEntry> search_text(const std::string& query);
    
    // Statistics
    struct DatasetStatistics {
        size_t total_entries;
        size_t unique_prompts;
        std::unordered_map<std::string, size_t> annotator_contributions;
        float average_preference_strength;
        std::vector<std::pair<uint64_t, size_t>> entry_timeline;
    };
    DatasetStatistics get_statistics(const std::string& version_id = "HEAD");
    
    // Export/Import
    void export_to_json(const std::string& path, const std::string& version_id = "HEAD");
    void import_from_json(const std::string& path);
    void export_to_parquet(const std::string& path, const std::string& version_id = "HEAD");
    
private:
    std::string repository_path_;
    git_repository* git_repo_;
    sqlite3* db_;
    
    // Current working version
    std::string current_version_;
    std::vector<PreferenceEntry> working_entries_;
    
    // Git operations
    void init_repository();
    std::string compute_tree_hash(const std::vector<PreferenceEntry>& entries);
    void write_tree(const std::vector<PreferenceEntry>& entries);
    
    // Database operations
    void init_database();
    void insert_entries_to_db(const std::vector<PreferenceEntry>& entries,
                             const std::string& version_id);
    std::vector<PreferenceEntry> load_entries_from_db(const std::string& version_id);
};

// Distributed preference data management
class DistributedPreferenceData {
public:
    DistributedPreferenceData(const std::vector<std::string>& node_addresses);
    
    // Distributed operations
    void replicate_to_nodes(const std::vector<PreferenceEntry>& entries);
    std::vector<PreferenceEntry> gather_from_nodes();
    
    // Consensus
    PreferenceEntry resolve_conflicts(const std::vector<PreferenceEntry>& conflicting_entries);
    
    // Sharding
    void shard_by_hash(const std::vector<PreferenceEntry>& entries);
    std::vector<PreferenceEntry> retrieve_shard(int shard_id);
    
private:
    std::vector<std::string> node_addresses_;
    std::unordered_map<int, std::vector<PreferenceEntry>> shards_;
    
    int compute_shard_id(const std::string& entry_id);
};

// Data validation and quality control
class PreferenceDataValidator {
public:
    PreferenceDataValidator();
    
    // Validation rules
    struct ValidationRule {
        std::string name;
        std::function<bool(const PreferenceEntry&)> validator;
        std::string error_message;
    };
    
    void add_rule(const ValidationRule& rule);
    
    // Validation
    struct ValidationResult {
        bool is_valid;
        std::vector<std::string> errors;
        std::vector<std::string> warnings;
        std::unordered_map<std::string, float> quality_scores;
    };
    
    ValidationResult validate_entry(const PreferenceEntry& entry);
    ValidationResult validate_batch(const std::vector<PreferenceEntry>& entries);
    
    // Quality metrics
    float compute_inter_annotator_agreement(const std::vector<PreferenceEntry>& entries);
    float compute_preference_consistency(const std::vector<PreferenceEntry>& entries);
    
private:
    std::vector<ValidationRule> rules_;
    
    // Built-in validators
    bool validate_text_length(const PreferenceEntry& entry);
    bool validate_preference_strength(const PreferenceEntry& entry);
    bool validate_response_quality(const PreferenceEntry& entry);
};

// Active learning for preference data collection
class ActivePreferenceLearning {
public:
    ActivePreferenceLearning(std::shared_ptr<model::DistributedTransformer> model);
    
    // Sample selection strategies
    std::vector<std::pair<std::string, std::string>> select_uncertain_pairs(
        const std::vector<std::string>& prompts,
        int num_samples);
    
    std::vector<std::pair<std::string, std::string>> select_diverse_pairs(
        const std::vector<std::string>& prompts,
        int num_samples);
    
    std::vector<std::pair<std::string, std::string>> select_representative_pairs(
        const std::vector<std::string>& prompts,
        int num_samples);
    
    // Uncertainty estimation
    torch::Tensor compute_uncertainty(const std::string& prompt,
                                     const std::string& response1,
                                     const std::string& response2);
    
    // Batch active learning
    std::vector<PreferenceEntry> generate_queries(int batch_size,
                                                 const std::string& selection_strategy = "uncertainty");
    
private:
    std::shared_ptr<model::DistributedTransformer> model_;
    
    // Embeddings cache
    std::unordered_map<std::string, torch::Tensor> embedding_cache_;
    
    torch::Tensor get_embedding(const std::string& text);
    float compute_diversity_score(const std::vector<torch::Tensor>& embeddings);
};

// Preference data augmentation
class PreferenceDataAugmenter {
public:
    PreferenceDataAugmenter();
    
    // Augmentation strategies
    std::vector<PreferenceEntry> paraphrase_augmentation(const PreferenceEntry& entry);
    std::vector<PreferenceEntry> backtranslation_augmentation(const PreferenceEntry& entry);
    std::vector<PreferenceEntry> synonym_replacement(const PreferenceEntry& entry);
    std::vector<PreferenceEntry> preference_interpolation(const PreferenceEntry& entry1,
                                                         const PreferenceEntry& entry2);
    
    // Synthetic data generation
    std::vector<PreferenceEntry> generate_synthetic_preferences(
        const std::vector<std::string>& prompts,
        std::shared_ptr<model::DistributedTransformer> generator_model);
    
    // Consistency preservation
    void ensure_transitivity(std::vector<PreferenceEntry>& entries);
    void remove_contradictions(std::vector<PreferenceEntry>& entries);
    
private:
    // Language models for augmentation
    std::unique_ptr<model::DistributedTransformer> paraphrase_model_;
    std::unique_ptr<model::DistributedTransformer> translation_model_;
    
    // Augmentation parameters
    float temperature_ = 0.8f;
    int max_augmentations_per_sample_ = 5;
};

} // namespace rlhf
