#pragma once

#include <vector>
#include <memory>
#include <torch/torch.h>
#include "distributed_transformer.hpp"

namespace model_ops {

// Model merging strategies
enum class MergeStrategy {
    AVERAGE,           // Simple averaging
    WEIGHTED_AVERAGE,  // Weighted averaging
    FISHER_WEIGHTED,   // Fisher information weighted
    TASK_ARITHMETIC,   // Task vector arithmetic
    TIES_MERGING,      // Trim, Elect, and Merge
    DARE,              // Drop and Rescale
    LINEAR_MODE_CONNECTIVITY,  // Find linear path
    SLERP              // Spherical linear interpolation
};

// Model merger
class ModelMerger {
public:
    ModelMerger();
    
    // Basic merging
    std::shared_ptr<model::DistributedTransformer> merge(
        const std::vector<std::shared_ptr<model::DistributedTransformer>>& models,
        MergeStrategy strategy,
        const std::vector<float>& weights = {});
    
    // Task arithmetic
    std::shared_ptr<model::DistributedTransformer> task_arithmetic_merge(
        std::shared_ptr<model::DistributedTransformer> base_model,
        const std::vector<std::shared_ptr<model::DistributedTransformer>>& task_models,
        const std::vector<float>& task_weights);
    
    // TIES merging
    std::shared_ptr<model::DistributedTransformer> ties_merge(
        const std::vector<std::shared_ptr<model::DistributedTransformer>>& models,
        float density_threshold = 0.2f,
        float majority_sign_threshold = 0.5f);
    
    // DARE merging
    std::shared_ptr<model::DistributedTransformer> dare_merge(
        const std::vector<std::shared_ptr<model::DistributedTransformer>>& models,
        float drop_rate = 0.9f);
    
    // Evolutionary merging
    std::shared_ptr<model::DistributedTransformer> evolutionary_merge(
        const std::vector<std::shared_ptr<model::DistributedTransformer>>& models,
        const std::vector<std::string>& eval_tasks,
        int num_generations = 10);
    
private:
    // Helper functions
    torch::Tensor compute_fisher_information(
        std::shared_ptr<model::DistributedTransformer> model,
        const std::vector<torch::Tensor>& data);
    
    torch::Tensor compute_task_vector(
        std::shared_ptr<model::DistributedTransformer> task_model,
        std::shared_ptr<model::DistributedTransformer> base_model);
    
    torch::Tensor slerp_tensors(const torch::Tensor& t1,
                               const torch::Tensor& t2,
                               float alpha);
};

} // namespace model_ops
