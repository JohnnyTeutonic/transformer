#include "../include/convergence_quality.hpp"
#include "../include/logger.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>

namespace convergence {

ConvergenceQualityManager::ConvergenceQualityManager(const ConvergenceConfig& config)
    : config_(config), current_global_step_(0), current_proximal_mu_(config.fedprox_config.proximal_mu),
      current_server_lr_(config.scaffold_config.server_learning_rate) {
    
    logger::log_info("Initializing Convergence Quality Manager");
    logger::log_info("- FedProx: " + (config_.enable_fedprox ? "enabled" : "disabled"));
    logger::log_info("- SCAFFOLD: " + (config_.enable_scaffold ? "enabled" : "disabled"));
    logger::log_info("- Max staleness: " + std::to_string(config_.max_staleness_steps));
    logger::log_info("- Validation frequency: " + std::to_string(config_.validation_frequency));
    
    // Initialize convergence metrics
    current_metrics_ = {};
    current_metrics_.convergence_status = "initializing";
    
    // Initialize statistics
    stats_ = {};
    stats_.last_checkpoint = std::chrono::steady_clock::now();
    stats_.current_algorithm = config_.enable_scaffold ? "SCAFFOLD" : 
                              (config_.enable_fedprox ? "FedProx" : "Vanilla");
}

ConvergenceQualityManager::~ConvergenceQualityManager() {
    // Cleanup if needed
}

std::vector<float> ConvergenceQualityManager::process_gradients_fedprox(
    const std::vector<GradientState>& node_gradients, const std::vector<float>& global_model) {
    
    if (!config_.enable_fedprox) {
        logger::log_warning("FedProx processing requested but FedProx is disabled");
        return combine_gradients_with_staleness_handling(node_gradients);
    }
    
    logger::log_debug("Processing gradients with FedProx algorithm");
    
    if (node_gradients.empty()) {
        return {};
    }
    
    size_t gradient_size = node_gradients[0].gradients.size();
    std::vector<float> aggregated_gradients(gradient_size, 0.0f);
    
    // Adapt proximal term coefficient based on gradient heterogeneity
    float adaptive_mu = config_.fedprox_config.enable_adaptive_mu ? 
                       adapt_proximal_mu(node_gradients) : current_proximal_mu_;
    
    float total_weight = 0.0f;
    
    for (const auto& gradient_state : node_gradients) {
        // Calculate staleness weight
        float staleness_weight = calculate_staleness_weight(current_global_step_, gradient_state.step_number);
        
        // Apply FedProx regularization
        // The proximal term encourages local updates to stay close to global model
        std::vector<float> regularized_gradients = gradient_state.gradients;
        
        // Add proximal term: mu * (local_params - global_params)
        // Note: In practice, this would require access to local parameters
        // For now, we approximate by scaling gradients based on their magnitude
        float gradient_norm = 0.0f;
        for (float g : gradient_state.gradients) {
            gradient_norm += g * g;
        }
        gradient_norm = std::sqrt(gradient_norm);
        
        float proximal_scale = 1.0f / (1.0f + adaptive_mu * gradient_norm);
        
        // Apply regularization and accumulate
        for (size_t i = 0; i < gradient_size; ++i) {
            float regularized_grad = regularized_gradients[i] * proximal_scale;
            aggregated_gradients[i] += regularized_grad * staleness_weight;
        }
        
        total_weight += staleness_weight;
    }
    
    // Normalize by total weight
    if (total_weight > 0) {
        for (float& grad : aggregated_gradients) {
            grad /= total_weight;
        }
    }
    
    logger::log_debug("FedProx processing completed with mu=" + std::to_string(adaptive_mu) + 
                     ", nodes=" + std::to_string(node_gradients.size()));
    
    return aggregated_gradients;
}

std::vector<float> ConvergenceQualityManager::process_gradients_scaffold(
    const std::vector<GradientState>& node_gradients, const std::vector<float>& global_model) {
    
    if (!config_.enable_scaffold) {
        logger::log_warning("SCAFFOLD processing requested but SCAFFOLD is disabled");
        return combine_gradients_with_staleness_handling(node_gradients);
    }
    
    logger::log_debug("Processing gradients with SCAFFOLD algorithm");
    
    if (node_gradients.empty()) {
        return {};
    }
    
    size_t gradient_size = node_gradients[0].gradients.size();
    std::vector<float> aggregated_gradients(gradient_size, 0.0f);
    
    // Ensure server control variates are initialized
    {
        std::lock_guard<std::mutex> lock(control_variates_mutex_);
        if (server_control_variates_.empty()) {
            server_control_variates_.resize(gradient_size, 0.0f);
        }
    }
    
    float total_weight = 0.0f;
    std::vector<std::vector<float>> client_control_updates;
    
    for (const auto& gradient_state : node_gradients) {
        // Get client control variates
        std::vector<float> client_variates = get_control_variates(gradient_state.node_id);
        
        // Calculate staleness weight
        float staleness_weight = calculate_staleness_weight(current_global_step_, gradient_state.step_number);
        
        // Apply SCAFFOLD correction
        for (size_t i = 0; i < gradient_size; ++i) {
            float corrected_gradient = gradient_state.gradients[i];
            
            if (i < client_variates.size() && i < server_control_variates_.size()) {
                // SCAFFOLD correction: g_i - c_i + c_server
                corrected_gradient = gradient_state.gradients[i] - client_variates[i] + server_control_variates_[i];
            }
            
            aggregated_gradients[i] += corrected_gradient * staleness_weight;
        }
        
        total_weight += staleness_weight;
        
        // Store client control variate updates for later server update
        client_control_updates.push_back(client_variates);
    }
    
    // Normalize by total weight
    if (total_weight > 0) {
        for (float& grad : aggregated_gradients) {
            grad /= total_weight;
        }
    }
    
    // Update server control variates
    {
        std::lock_guard<std::mutex> lock(control_variates_mutex_);
        for (size_t i = 0; i < server_control_variates_.size(); ++i) {
            float client_avg = 0.0f;
            for (const auto& client_variates : client_control_updates) {
                if (i < client_variates.size()) {
                    client_avg += client_variates[i];
                }
            }
            if (!client_control_updates.empty()) {
                client_avg /= client_control_updates.size();
            }
            
            // Server control variate update with momentum
            if (config_.scaffold_config.enable_momentum && !server_momentum_.empty() && i < server_momentum_.size()) {
                float momentum_term = config_.scaffold_config.momentum_beta * server_momentum_[i];
                float gradient_term = (1.0f - config_.scaffold_config.momentum_beta) * client_avg;
                server_momentum_[i] = momentum_term + gradient_term;
                server_control_variates_[i] += current_server_lr_ * server_momentum_[i];
            } else {
                server_control_variates_[i] += current_server_lr_ * client_avg;
            }
        }
        
        // Initialize momentum if needed
        if (config_.scaffold_config.enable_momentum && server_momentum_.empty()) {
            server_momentum_.resize(gradient_size, 0.0f);
        }
    }
    
    logger::log_debug("SCAFFOLD processing completed with server_lr=" + std::to_string(current_server_lr_) + 
                     ", nodes=" + std::to_string(node_gradients.size()));
    
    return aggregated_gradients;
}

std::vector<float> ConvergenceQualityManager::combine_gradients_with_staleness_handling(
    const std::vector<GradientState>& node_gradients) {
    
    if (node_gradients.empty()) {
        return {};
    }
    
    logger::log_debug("Combining gradients with staleness handling");
    
    size_t gradient_size = node_gradients[0].gradients.size();
    std::vector<float> combined_gradients(gradient_size, 0.0f);
    
    float total_weight = 0.0f;
    uint32_t filtered_count = 0;
    
    for (const auto& gradient_state : node_gradients) {
        // Check staleness bounds
        uint64_t staleness = current_global_step_ - gradient_state.step_number;
        if (staleness > config_.max_staleness_steps) {
            logger::log_debug("Filtering out stale gradient from " + gradient_state.node_id + 
                             " (staleness: " + std::to_string(staleness) + ")");
            continue;
        }
        
        // Calculate weight based on staleness
        float staleness_weight = calculate_staleness_weight(current_global_step_, gradient_state.step_number);
        
        // Additional weighting based on local steps (more local steps = potentially more stale)
        float local_step_weight = 1.0f / (1.0f + gradient_state.local_steps * 0.1f);
        
        float combined_weight = staleness_weight * local_step_weight;
        
        // Accumulate weighted gradients
        for (size_t i = 0; i < gradient_size; ++i) {
            combined_gradients[i] += gradient_state.gradients[i] * combined_weight;
        }
        
        total_weight += combined_weight;
        filtered_count++;
    }
    
    // Normalize by total weight
    if (total_weight > 0) {
        for (float& grad : combined_gradients) {
            grad /= total_weight;
        }
    }
    
    logger::log_debug("Combined " + std::to_string(filtered_count) + "/" + 
                     std::to_string(node_gradients.size()) + " gradients");
    
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.active_nodes = filtered_count;
        
        // Calculate average staleness
        float total_staleness = 0.0f;
        for (const auto& gradient_state : node_gradients) {
            total_staleness += current_global_step_ - gradient_state.step_number;
        }
        stats_.average_gradient_staleness = total_staleness / node_gradients.size();
    }
    
    return combined_gradients;
}

ConvergenceMetrics ConvergenceQualityManager::analyze_convergence(
    const std::vector<float>& current_loss_history, const std::vector<float>& current_gradients) {
    
    std::lock_guard<std::mutex> lock(convergence_mutex_);
    
    ConvergenceMetrics metrics;
    
    // Update loss history
    if (!current_loss_history.empty()) {
        loss_history_.insert(loss_history_.end(), current_loss_history.begin(), current_loss_history.end());
        
        // Keep only recent history
        if (loss_history_.size() > config_.convergence_window * 2) {
            loss_history_.erase(loss_history_.begin(), 
                               loss_history_.begin() + (loss_history_.size() - config_.convergence_window * 2));
        }
    }
    
    // Calculate gradient norm
    if (!current_gradients.empty()) {
        float grad_norm_sq = 0.0f;
        for (float g : current_gradients) {
            grad_norm_sq += g * g;
        }
        metrics.gradient_norm = std::sqrt(grad_norm_sq);
    }
    
    // Analyze loss trend
    if (loss_history_.size() >= config_.convergence_window) {
        std::vector<float> recent_losses(loss_history_.end() - config_.convergence_window, loss_history_.end());
        
        // Calculate loss reduction rate (negative means loss is increasing)
        float avg_recent = std::accumulate(recent_losses.begin(), recent_losses.end(), 0.0f) / recent_losses.size();
        
        if (loss_history_.size() >= config_.convergence_window * 2) {
            std::vector<float> older_losses(loss_history_.end() - 2 * config_.convergence_window, 
                                          loss_history_.end() - config_.convergence_window);
            float avg_older = std::accumulate(older_losses.begin(), older_losses.end(), 0.0f) / older_losses.size();
            
            metrics.loss_reduction_rate = (avg_older - avg_recent) / avg_older;
        }
        
        // Check for convergence
        if (recent_losses.size() >= 2) {
            float loss_variance = 0.0f;
            for (float loss : recent_losses) {
                float diff = loss - avg_recent;
                loss_variance += diff * diff;
            }
            loss_variance /= recent_losses.size();
            
            // Convergence criteria
            bool small_gradient = metrics.gradient_norm < config_.convergence_threshold;
            bool stable_loss = loss_variance < config_.convergence_threshold * config_.convergence_threshold;
            bool improving = metrics.loss_reduction_rate > 0;
            
            if (small_gradient && stable_loss) {
                metrics.is_converged = true;
                metrics.convergence_status = "converged";
                metrics.convergence_score = 0.9f + 0.1f * std::min(1.0f, metrics.loss_reduction_rate);
            } else if (improving) {
                metrics.convergence_status = "converging";
                metrics.convergence_score = 0.5f + 0.4f * std::min(1.0f, metrics.loss_reduction_rate);
            } else {
                // Check for divergence
                float recent_max = *std::max_element(recent_losses.begin(), recent_losses.end());
                float recent_min = *std::min_element(recent_losses.begin(), recent_losses.end());
                
                if ((recent_max - recent_min) / recent_min > config_.divergence_threshold) {
                    metrics.is_diverging = true;
                    metrics.convergence_status = "diverging";
                    metrics.convergence_score = 0.1f;
                } else {
                    metrics.convergence_status = "stalled";
                    metrics.convergence_score = 0.3f;
                }
            }
        }
    } else {
        metrics.convergence_status = "insufficient_data";
        metrics.convergence_score = 0.5f; // Neutral score
    }
    
    // Count steps since improvement
    if (!loss_history_.empty() && loss_history_.size() >= 2) {
        float current_loss = loss_history_.back();
        bool found_improvement = false;
        
        for (int i = static_cast<int>(loss_history_.size()) - 2; i >= 0; --i) {
            if (loss_history_[i] > current_loss + config_.convergence_threshold) {
                found_improvement = true;
                metrics.steps_since_improvement = static_cast<uint32_t>(loss_history_.size() - 1 - i);
                break;
            }
        }
        
        if (!found_improvement) {
            metrics.steps_since_improvement = static_cast<uint32_t>(loss_history_.size());
        }
    }
    
    current_metrics_ = metrics;
    
    logger::log_debug("Convergence analysis: " + metrics.convergence_status + 
                     ", score=" + std::to_string(metrics.convergence_score) + 
                     ", grad_norm=" + std::to_string(metrics.gradient_norm));
    
    return metrics;
}

float ConvergenceQualityManager::calculate_staleness_weight(uint64_t current_step, uint64_t gradient_step) {
    if (current_step <= gradient_step) {
        return 1.0f; // Not stale
    }
    
    uint64_t staleness = current_step - gradient_step;
    
    if (staleness > config_.max_staleness_steps) {
        return 0.0f; // Too stale, reject
    }
    
    // Exponential decay weight
    return std::pow(config_.staleness_penalty_factor, static_cast<float>(staleness));
}

bool ConvergenceQualityManager::initialize_control_variates(const std::vector<std::string>& node_ids,
                                                           size_t parameter_count) {
    std::lock_guard<std::mutex> lock(control_variates_mutex_);
    
    logger::log_info("Initializing SCAFFOLD control variates for " + std::to_string(node_ids.size()) + 
                     " nodes, " + std::to_string(parameter_count) + " parameters");
    
    // Initialize client control variates to zero
    for (const auto& node_id : node_ids) {
        node_control_variates_[node_id] = std::vector<float>(parameter_count, 0.0f);
    }
    
    // Initialize server control variates to zero
    server_control_variates_ = std::vector<float>(parameter_count, 0.0f);
    
    // Initialize server momentum if enabled
    if (config_.scaffold_config.enable_momentum) {
        server_momentum_ = std::vector<float>(parameter_count, 0.0f);
    }
    
    return true;
}

bool ConvergenceQualityManager::update_control_variates(const std::string& node_id,
                                                       const std::vector<float>& local_gradients,
                                                       const std::vector<float>& global_gradients) {
    std::lock_guard<std::mutex> lock(control_variates_mutex_);
    
    auto it = node_control_variates_.find(node_id);
    if (it == node_control_variates_.end()) {
        logger::log_error("Node " + node_id + " not found in control variates");
        return false;
    }
    
    auto& client_variates = it->second;
    
    if (client_variates.size() != local_gradients.size() || 
        local_gradients.size() != global_gradients.size()) {
        logger::log_error("Gradient size mismatch in control variate update");
        return false;
    }
    
    // Update client control variates: c_i^{t+1} = c_i^t - c_server + (1/K) * sum(g_local - g_global)
    for (size_t i = 0; i < client_variates.size(); ++i) {
        float server_variate = (i < server_control_variates_.size()) ? server_control_variates_[i] : 0.0f;
        float gradient_diff = local_gradients[i] - global_gradients[i];
        
        if (config_.scaffold_config.enable_momentum && !server_momentum_.empty() && i < server_momentum_.size()) {
            // Apply momentum to the update
            float momentum_term = config_.scaffold_config.momentum_beta * server_momentum_[i];
            float gradient_term = (1.0f - config_.scaffold_config.momentum_beta) * gradient_diff;
            client_variates[i] = client_variates[i] - server_variate + momentum_term + gradient_term;
        } else {
            client_variates[i] = client_variates[i] - server_variate + gradient_diff;
        }
    }
    
    return true;
}

std::vector<float> ConvergenceQualityManager::get_control_variates(const std::string& node_id) const {
    std::lock_guard<std::mutex> lock(control_variates_mutex_);
    
    auto it = node_control_variates_.find(node_id);
    if (it != node_control_variates_.end()) {
        return it->second;
    }
    
    return {}; // Return empty vector if not found
}

float ConvergenceQualityManager::adapt_proximal_mu(const std::vector<GradientState>& gradients) {
    if (!config_.fedprox_config.enable_adaptive_mu) {
        return current_proximal_mu_;
    }
    
    // Calculate gradient heterogeneity
    float heterogeneity = compute_gradient_heterogeneity(gradients);
    gradient_heterogeneity_history_.push_back(heterogeneity);
    
    // Keep only recent history
    if (gradient_heterogeneity_history_.size() > 50) {
        gradient_heterogeneity_history_.erase(gradient_heterogeneity_history_.begin());
    }
    
    // Adapt mu based on heterogeneity (more heterogeneous = higher mu)
    float avg_heterogeneity = std::accumulate(gradient_heterogeneity_history_.begin(), 
                                            gradient_heterogeneity_history_.end(), 0.0f) / 
                             gradient_heterogeneity_history_.size();
    
    float target_mu = config_.fedprox_config.proximal_mu * (1.0f + avg_heterogeneity);
    target_mu = std::max(config_.fedprox_config.min_mu, 
                        std::min(config_.fedprox_config.max_mu, target_mu));
    
    // Smooth adaptation
    current_proximal_mu_ = current_proximal_mu_ + 
                          config_.fedprox_config.mu_adaptation_rate * (target_mu - current_proximal_mu_);
    
    logger::log_debug("Adapted proximal mu: " + std::to_string(current_proximal_mu_) + 
                     " (heterogeneity: " + std::to_string(heterogeneity) + ")");
    
    return current_proximal_mu_;
}

float ConvergenceQualityManager::compute_gradient_heterogeneity(const std::vector<GradientState>& gradients) {
    if (gradients.size() < 2) {
        return 0.0f;
    }
    
    // Compute mean gradient
    std::vector<float> mean_gradient = compute_gradient_mean(gradients);
    
    // Compute variance (measure of heterogeneity)
    float total_variance = 0.0f;
    
    for (const auto& gradient_state : gradients) {
        for (size_t i = 0; i < gradient_state.gradients.size() && i < mean_gradient.size(); ++i) {
            float diff = gradient_state.gradients[i] - mean_gradient[i];
            total_variance += diff * diff;
        }
    }
    
    if (!mean_gradient.empty()) {
        total_variance /= (gradients.size() * mean_gradient.size());
    }
    
    return std::sqrt(total_variance);
}

std::vector<float> ConvergenceQualityManager::compute_gradient_mean(const std::vector<GradientState>& gradients) {
    if (gradients.empty()) {
        return {};
    }
    
    size_t gradient_size = gradients[0].gradients.size();
    std::vector<float> mean_gradient(gradient_size, 0.0f);
    
    for (const auto& gradient_state : gradients) {
        for (size_t i = 0; i < gradient_size && i < gradient_state.gradients.size(); ++i) {
            mean_gradient[i] += gradient_state.gradients[i];
        }
    }
    
    for (float& grad : mean_gradient) {
        grad /= gradients.size();
    }
    
    return mean_gradient;
}

ConvergenceQualityManager::QualityAssessment ConvergenceQualityManager::assess_training_quality(
    const std::vector<GradientState>& recent_gradients, const std::vector<float>& recent_losses) {
    
    QualityAssessment assessment;
    assessment.overall_quality_score = 0.5f; // Default neutral score
    
    // 1. Gradient consistency score
    if (recent_gradients.size() >= 2) {
        float heterogeneity = compute_gradient_heterogeneity(recent_gradients);
        // Lower heterogeneity = higher consistency (invert and normalize)
        assessment.gradient_consistency_score = 1.0f / (1.0f + heterogeneity);
        
        if (heterogeneity > 2.0f) {
            assessment.quality_issues.push_back("High gradient heterogeneity detected");
        }
    } else {
        assessment.gradient_consistency_score = 0.5f;
        assessment.quality_issues.push_back("Insufficient gradient data for consistency analysis");
    }
    
    // 2. Convergence stability score
    if (recent_losses.size() >= 10) {
        float loss_variance = 0.0f;
        float loss_mean = std::accumulate(recent_losses.begin(), recent_losses.end(), 0.0f) / recent_losses.size();
        
        for (float loss : recent_losses) {
            float diff = loss - loss_mean;
            loss_variance += diff * diff;
        }
        loss_variance /= recent_losses.size();
        
        // Lower variance = higher stability
        assessment.convergence_stability_score = 1.0f / (1.0f + loss_variance);
        
        // Check for oscillations
        int direction_changes = 0;
        for (size_t i = 2; i < recent_losses.size(); ++i) {
            bool curr_increasing = recent_losses[i] > recent_losses[i-1];
            bool prev_increasing = recent_losses[i-1] > recent_losses[i-2];
            if (curr_increasing != prev_increasing) {
                direction_changes++;
            }
        }
        
        if (direction_changes > recent_losses.size() / 3) {
            assessment.quality_issues.push_back("Loss oscillations detected");
        }
    } else {
        assessment.convergence_stability_score = 0.5f;
        assessment.quality_issues.push_back("Insufficient loss history for stability analysis");
    }
    
    // 3. Parameter quality score (based on gradient norms)
    if (!recent_gradients.empty()) {
        std::vector<float> gradient_norms;
        for (const auto& gradient_state : recent_gradients) {
            float norm = 0.0f;
            for (float g : gradient_state.gradients) {
                norm += g * g;
            }
            gradient_norms.push_back(std::sqrt(norm));
        }
        
        float avg_norm = std::accumulate(gradient_norms.begin(), gradient_norms.end(), 0.0f) / gradient_norms.size();
        
        // Healthy gradient norms should be neither too large nor too small
        if (avg_norm > 0.001f && avg_norm < 10.0f) {
            assessment.parameter_quality_score = 0.8f + 0.2f / (1.0f + std::abs(std::log10(avg_norm)));
        } else {
            assessment.parameter_quality_score = 0.3f;
            if (avg_norm <= 0.001f) {
                assessment.quality_issues.push_back("Very small gradient norms - possible vanishing gradients");
            } else {
                assessment.quality_issues.push_back("Very large gradient norms - possible exploding gradients");
            }
        }
    } else {
        assessment.parameter_quality_score = 0.5f;
    }
    
    // 4. Overall quality score (weighted average)
    assessment.overall_quality_score = (assessment.gradient_consistency_score * 0.3f +
                                       assessment.convergence_stability_score * 0.4f +
                                       assessment.parameter_quality_score * 0.3f);
    
    // Add overall quality issues
    if (assessment.overall_quality_score < 0.3f) {
        assessment.quality_issues.push_back("Overall training quality is poor");
    } else if (assessment.overall_quality_score < 0.6f) {
        assessment.quality_issues.push_back("Training quality needs improvement");
    }
    
    logger::log_info("Training quality assessment: overall=" + std::to_string(assessment.overall_quality_score) + 
                     ", issues=" + std::to_string(assessment.quality_issues.size()));
    
    return assessment;
}

bool ConvergenceQualityManager::update_global_model(const std::vector<float>& new_parameters, uint64_t global_step) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    
    global_model_ = new_parameters;
    current_global_step_ = global_step;
    
    // Create model snapshot
    ModelSnapshot snapshot;
    snapshot.parameters = new_parameters;
    snapshot.global_step = global_step;
    snapshot.timestamp = std::chrono::steady_clock::now();
    
    model_history_.push_back(snapshot);
    
    // Keep only recent history
    if (model_history_.size() > 100) {
        model_history_.erase(model_history_.begin());
    }
    
    {
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        stats_.total_global_steps = global_step;
    }
    
    return true;
}

ConvergenceQualityManager::ConvergenceStats ConvergenceQualityManager::get_convergence_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

bool ConvergenceQualityManager::export_convergence_report(const std::string& output_path) const {
    try {
        std::ofstream file(output_path);
        if (!file.is_open()) {
            logger::log_error("Failed to open convergence report file: " + output_path);
            return false;
        }
        
        auto stats = get_convergence_stats();
        auto metrics = current_metrics_;
        
        file << "Convergence Quality Report\n";
        file << "========================\n\n";
        
        file << "Algorithm: " << stats.current_algorithm << "\n";
        file << "Global Steps: " << stats.total_global_steps << "\n";
        file << "Active Nodes: " << stats.active_nodes << "\n";
        file << "Average Gradient Staleness: " << stats.average_gradient_staleness << "\n";
        file << "Current Learning Rate: " << stats.current_learning_rate << "\n\n";
        
        file << "Convergence Metrics:\n";
        file << "- Status: " << metrics.convergence_status << "\n";
        file << "- Score: " << std::fixed << std::setprecision(4) << metrics.convergence_score << "\n";
        file << "- Gradient Norm: " << metrics.gradient_norm << "\n";
        file << "- Loss Reduction Rate: " << metrics.loss_reduction_rate << "\n";
        file << "- Steps Since Improvement: " << metrics.steps_since_improvement << "\n";
        file << "- Is Converged: " << (metrics.is_converged ? "Yes" : "No") << "\n";
        file << "- Is Diverging: " << (metrics.is_diverging ? "Yes" : "No") << "\n\n";
        
        // Loss history
        if (!loss_history_.empty()) {
            file << "Recent Loss History:\n";
            size_t start = loss_history_.size() > 20 ? loss_history_.size() - 20 : 0;
            for (size_t i = start; i < loss_history_.size(); ++i) {
                file << "Step " << (i + 1) << ": " << loss_history_[i] << "\n";
            }
        }
        
        file.close();
        logger::log_info("Convergence report exported to: " + output_path);
        return true;
        
    } catch (const std::exception& e) {
        logger::log_error("Failed to export convergence report: " + std::string(e.what()));
        return false;
    }
}

} // namespace convergence
