#pragma once

#include <string>
#include <vector>
#include <memory>
#include <torch/torch.h>
#include "distributed_transformer.hpp"

namespace optimization {

// Prompt optimization configuration
struct PromptOptConfig {
    int max_prompt_length = 100;
    int num_soft_tokens = 20;
    float learning_rate = 0.1f;
    int optimization_steps = 100;
    std::string optimization_method = "gradient";  // "gradient", "evolutionary", "bayesian"
    bool use_continuous_prompts = true;
    int beam_size = 5;
    float temperature = 1.0f;
};

// Prompt optimizer
class PromptOptimizer {
public:
    PromptOptimizer(std::shared_ptr<model::DistributedTransformer> model,
                   PromptOptConfig config);
    
    // Optimize prompts
    std::string optimize_discrete_prompt(const std::string& task_description,
                                        const std::vector<std::pair<std::string, std::string>>& examples);
    
    torch::Tensor optimize_continuous_prompt(const std::string& task_description,
                                           const std::vector<std::pair<std::string, std::string>>& examples);
    
    // Automatic prompt engineering
    std::vector<std::string> generate_prompt_variants(const std::string& base_prompt,
                                                     int num_variants = 10);
    
    std::string select_best_prompt(const std::vector<std::string>& prompts,
                                  const std::vector<std::pair<std::string, std::string>>& validation_set);
    
private:
    std::shared_ptr<model::DistributedTransformer> model_;
    PromptOptConfig config_;
    
    // Optimization methods
    torch::Tensor gradient_based_optimization(const torch::Tensor& initial_prompt,
                                             const std::vector<torch::Tensor>& targets);
    
    std::string evolutionary_optimization(const std::string& initial_prompt,
                                        const std::vector<std::pair<std::string, std::string>>& examples);
    
    std::string bayesian_optimization(const std::string& initial_prompt,
                                     const std::vector<std::pair<std::string, std::string>>& examples);
};

} // namespace optimization
