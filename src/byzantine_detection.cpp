#include "../include/byzantine_detection.hpp"
#include "../include/logger.hpp"
#include <algorithm>
#include <random>
#include <cmath>
#include <numeric>
#include <set>
#include <fstream>
#include <sstream>

namespace byzantine {

ByzantineDetectionEngine::ByzantineDetectionEngine(const ByzantineDetectionConfig& config)
    : config_(config), next_task_id_(1) {
    
    logger::log_info("Initializing Byzantine Detection Engine");
    logger::log_info("- Outlier threshold: " + std::to_string(config_.outlier_threshold));
    logger::log_info("- Cross-validation frequency: " + std::to_string(config_.cross_val_frequency));
    logger::log_info("- Gradient clustering: " + (config_.enable_gradient_clustering ? "enabled" : "disabled"));
    logger::log_info("- Cross-validation: " + (config_.enable_cross_validation ? "enabled" : "disabled"));
    logger::log_info("- Magnitude bounds: " + (config_.enable_magnitude_bounds ? "enabled" : "disabled"));
    
    // Initialize empty bounds
    current_bounds_ = {};
    current_bounds_.last_updated = std::chrono::steady_clock::now();
    
    // Initialize stats
    stats_ = {};
    stats_.last_detection_run = std::chrono::steady_clock::now();
}

ByzantineDetectionEngine::~ByzantineDetectionEngine() {
    // Any cleanup needed
}

ClusterAnalysis ByzantineDetectionEngine::analyze_gradient_similarity(
    const std::vector<GradientFingerprint>& gradients) {
    
    if (!config_.enable_gradient_clustering) {
        return ClusterAnalysis{};
    }
    
    logger::log_info("Analyzing gradient similarity for " + std::to_string(gradients.size()) + " gradients");
    
    ClusterAnalysis analysis;
    
    if (gradients.size() < config_.min_cluster_size) {
        logger::log_warning("Insufficient gradients for clustering analysis");
        analysis.outlier_nodes.reserve(gradients.size());
        for (const auto& grad : gradients) {
            analysis.outlier_nodes.push_back(grad.node_id);
        }
        return analysis;
    }
    
    // Extract gradient samples for clustering
    std::vector<std::vector<float>> gradient_samples;
    std::vector<std::string> node_ids;
    
    for (const auto& gradient : gradients) {
        gradient_samples.push_back(gradient.gradient_sample);
        node_ids.push_back(gradient.node_id);
    }
    
    // Determine optimal number of clusters (simple heuristic)
    size_t k = std::max(2UL, std::min(gradients.size() / 3, 5UL));
    
    try {
        // Perform K-means clustering
        auto cluster_indices = perform_kmeans_clustering(gradient_samples, k);
        
        // Convert cluster indices to node IDs
        analysis.clusters.resize(k);
        std::vector<size_t> cluster_sizes(k, 0);
        
        for (size_t i = 0; i < node_ids.size(); ++i) {
            for (size_t j = 0; j < cluster_indices.size(); ++j) {
                auto& cluster = cluster_indices[j];
                if (std::find(cluster.begin(), cluster.end(), i) != cluster.end()) {
                    analysis.clusters[j].push_back(node_ids[i]);
                    cluster_sizes[j]++;
                    break;
                }
            }
        }
        
        // Find dominant cluster
        auto max_cluster_it = std::max_element(cluster_sizes.begin(), cluster_sizes.end());
        analysis.dominant_cluster_size = *max_cluster_it;
        
        // Identify outliers (nodes in small clusters)
        for (size_t i = 0; i < analysis.clusters.size(); ++i) {
            if (cluster_sizes[i] < config_.min_cluster_size) {
                // Nodes in small clusters are considered outliers
                for (const auto& node_id : analysis.clusters[i]) {
                    analysis.outlier_nodes.push_back(node_id);
                }
            }
        }
        
        // Calculate clustering quality
        analysis.silhouette_score = calculate_silhouette_score(gradient_samples, cluster_indices);
        analysis.outlier_threshold = config_.outlier_threshold;
        
        logger::log_info("Clustering analysis completed:");
        logger::log_info("- " + std::to_string(analysis.clusters.size()) + " clusters found");
        logger::log_info("- " + std::to_string(analysis.outlier_nodes.size()) + " outlier nodes");
        logger::log_info("- Silhouette score: " + std::to_string(analysis.silhouette_score));
        
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.average_cluster_quality = (stats_.average_cluster_quality + analysis.silhouette_score) / 2.0f;
        }
        
    } catch (const std::exception& e) {
        logger::log_error("Clustering analysis failed: " + std::string(e.what()));
    }
    
    return analysis;
}

std::vector<std::string> ByzantineDetectionEngine::detect_outlier_nodes(
    const std::vector<GradientFingerprint>& gradients) {
    
    if (!config_.enable_gradient_clustering) {
        return {};
    }
    
    auto analysis = analyze_gradient_similarity(gradients);
    
    // Also check for statistical outliers in L2 norms
    std::vector<float> l2_norms;
    for (const auto& grad : gradients) {
        l2_norms.push_back(grad.l2_norm);
    }
    
    if (l2_norms.size() >= 3) {
        // Calculate mean and standard deviation
        float mean = std::accumulate(l2_norms.begin(), l2_norms.end(), 0.0f) / l2_norms.size();
        float sq_sum = std::inner_product(l2_norms.begin(), l2_norms.end(), l2_norms.begin(), 0.0f);
        float stdev = std::sqrt(sq_sum / l2_norms.size() - mean * mean);
        
        // Find statistical outliers
        for (size_t i = 0; i < gradients.size(); ++i) {
            float z_score = std::abs(gradients[i].l2_norm - mean) / stdev;
            if (z_score > config_.outlier_threshold) {
                const std::string& node_id = gradients[i].node_id;
                // Add to outliers if not already present
                if (std::find(analysis.outlier_nodes.begin(), analysis.outlier_nodes.end(), node_id) 
                    == analysis.outlier_nodes.end()) {
                    analysis.outlier_nodes.push_back(node_id);
                }
                
                // Report suspicious activity
                SuspiciousActivity activity;
                activity.node_id = node_id;
                activity.activity_type = "statistical_outlier";
                activity.description = "L2 norm z-score: " + std::to_string(z_score);
                activity.severity_score = std::min(1.0f, z_score / 5.0f);
                activity.timestamp = static_cast<uint32_t>(
                    std::chrono::system_clock::now().time_since_epoch().count());
                activity.metadata["z_score"] = z_score;
                activity.metadata["l2_norm"] = gradients[i].l2_norm;
                
                report_suspicious_activity(activity);
            }
        }
    }
    
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.outliers_detected += analysis.outlier_nodes.size();
    }
    
    return analysis.outlier_nodes;
}

bool ByzantineDetectionEngine::validate_gradient_magnitude(const std::string& node_id, 
                                                          const GradientFingerprint& gradient) {
    
    if (!config_.enable_magnitude_bounds) {
        return true; // Skip validation if disabled
    }
    
    std::lock_guard<std::mutex> lock(bounds_mutex_);
    
    if (current_bounds_.samples_count == 0) {
        // No historical data yet, accept all gradients
        return true;
    }
    
    // Check overall L2 norm
    float z_score = std::abs(gradient.l2_norm - current_bounds_.mean_l2_norm) / 
                   std::max(0.01f, current_bounds_.std_l2_norm);
    
    if (z_score > config_.magnitude_violation_multiplier) {
        SuspiciousActivity activity;
        activity.node_id = node_id;
        activity.activity_type = "magnitude_violation";
        activity.description = "Gradient L2 norm outside acceptable bounds";
        activity.severity_score = std::min(1.0f, z_score / config_.magnitude_violation_multiplier);
        activity.timestamp = static_cast<uint32_t>(
            std::chrono::system_clock::now().time_since_epoch().count());
        activity.metadata["z_score"] = z_score;
        activity.metadata["l2_norm"] = gradient.l2_norm;
        activity.metadata["expected_mean"] = current_bounds_.mean_l2_norm;
        activity.metadata["expected_std"] = current_bounds_.std_l2_norm;
        
        report_suspicious_activity(activity);
        return false;
    }
    
    // Check layer-wise norms if available
    if (!gradient.layer_norms.empty() && !current_bounds_.layer_mean_norms.empty()) {
        size_t min_layers = std::min(gradient.layer_norms.size(), current_bounds_.layer_mean_norms.size());
        
        for (size_t i = 0; i < min_layers; ++i) {
            float layer_z_score = std::abs(gradient.layer_norms[i] - current_bounds_.layer_mean_norms[i]) /
                                  std::max(0.01f, current_bounds_.layer_std_norms[i]);
            
            if (layer_z_score > config_.magnitude_violation_multiplier) {
                SuspiciousActivity activity;
                activity.node_id = node_id;
                activity.activity_type = "layer_magnitude_violation";
                activity.description = "Layer " + std::to_string(i) + " gradient norm outside bounds";
                activity.severity_score = std::min(1.0f, layer_z_score / config_.magnitude_violation_multiplier);
                activity.timestamp = static_cast<uint32_t>(
                    std::chrono::system_clock::now().time_since_epoch().count());
                activity.metadata["layer_index"] = static_cast<float>(i);
                activity.metadata["layer_z_score"] = layer_z_score;
                activity.metadata["layer_norm"] = gradient.layer_norms[i];
                
                report_suspicious_activity(activity);
                return false;
            }
        }
    }
    
    return true;
}

std::vector<CrossValidationTask> ByzantineDetectionEngine::generate_cross_validation_tasks(
    const std::vector<std::string>& participating_nodes) {
    
    if (!config_.enable_cross_validation) {
        return {};
    }
    
    logger::log_info("Generating cross-validation tasks for " + std::to_string(participating_nodes.size()) + " nodes");
    
    std::vector<CrossValidationTask> tasks;
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (const auto& node_id : participating_nodes) {
        for (size_t i = 0; i < config_.cross_val_tasks_per_node; ++i) {
            CrossValidationTask task = create_synthetic_validation_task({node_id});
            tasks.push_back(task);
            
            // Store in pending tasks
            std::lock_guard<std::mutex> lock(cross_val_mutex_);
            pending_cross_val_tasks_[task.task_id] = task;
        }
    }
    
    logger::log_info("Generated " + std::to_string(tasks.size()) + " cross-validation tasks");
    return tasks;
}

void ByzantineDetectionEngine::submit_cross_validation_result(const CrossValidationResult& result) {
    std::lock_guard<std::mutex> lock(cross_val_mutex_);
    
    cross_val_results_[result.node_id].push_back(result);
    
    // Check if this is a failure
    if (!result.validation_passed) {
        SuspiciousActivity activity;
        activity.node_id = result.node_id;
        activity.activity_type = "cross_validation_failure";
        activity.description = "Failed cross-validation task " + result.task_id;
        activity.severity_score = std::min(1.0f, result.error_magnitude / config_.cross_val_error_threshold);
        activity.timestamp = static_cast<uint32_t>(
            std::chrono::system_clock::now().time_since_epoch().count());
        activity.metadata["task_id"] = std::hash<std::string>{}(result.task_id); // Convert to float
        activity.metadata["error_magnitude"] = result.error_magnitude;
        activity.metadata["completion_time"] = static_cast<float>(result.completion_time_ms);
        
        report_suspicious_activity(activity);
        
        // Update reputation
        update_node_reputation(result.node_id, -0.1f);
        
        {
            std::lock_guard<std::mutex> stats_lock(stats_mutex_);
            stats_.cross_validation_failures++;
        }
    } else {
        // Successful validation improves reputation slightly
        update_node_reputation(result.node_id, 0.02f);
    }
    
    {
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        stats_.cross_validation_tasks_completed++;
    }
    
    logger::log_debug("Cross-validation result submitted for " + result.node_id + 
                     " (passed: " + (result.validation_passed ? "yes" : "no") + ")");
}

std::vector<std::string> ByzantineDetectionEngine::get_cross_validation_failures() const {
    std::lock_guard<std::mutex> lock(cross_val_mutex_);
    
    std::set<std::string> failed_nodes;
    
    for (const auto& [node_id, results] : cross_val_results_) {
        // Check recent failure rate
        size_t recent_failures = 0;
        size_t recent_total = 0;
        
        auto now = std::chrono::system_clock::now().time_since_epoch().count();
        
        for (const auto& result : results) {
            if (now - result.completion_time_ms < 300000) { // Last 5 minutes
                recent_total++;
                if (!result.validation_passed) {
                    recent_failures++;
                }
            }
        }
        
        if (recent_total >= 3 && static_cast<float>(recent_failures) / recent_total > 0.5f) {
            failed_nodes.insert(node_id);
        }
    }
    
    return std::vector<std::string>(failed_nodes.begin(), failed_nodes.end());
}

void ByzantineDetectionEngine::update_historical_bounds(const std::vector<GradientFingerprint>& gradients) {
    std::lock_guard<std::mutex> lock(bounds_mutex_);
    
    for (const auto& gradient : gradients) {
        gradient_history_.push(gradient);
        
        // Maintain window size
        if (gradient_history_.size() > config_.historical_window_size) {
            gradient_history_.pop();
        }
    }
    
    // Recalculate bounds from history
    if (!gradient_history_.empty()) {
        std::vector<float> l2_norms;
        std::vector<std::vector<float>> layer_norms_by_layer;
        
        // Convert queue to vector for easier processing
        std::queue<GradientFingerprint> temp_queue = gradient_history_;
        while (!temp_queue.empty()) {
            const auto& grad = temp_queue.front();
            l2_norms.push_back(grad.l2_norm);
            
            // Initialize layer vectors if needed
            if (layer_norms_by_layer.size() < grad.layer_norms.size()) {
                layer_norms_by_layer.resize(grad.layer_norms.size());
            }
            
            for (size_t i = 0; i < grad.layer_norms.size(); ++i) {
                layer_norms_by_layer[i].push_back(grad.layer_norms[i]);
            }
            
            temp_queue.pop();
        }
        
        // Calculate statistics for L2 norms
        if (!l2_norms.empty()) {
            current_bounds_.mean_l2_norm = std::accumulate(l2_norms.begin(), l2_norms.end(), 0.0f) / l2_norms.size();
            
            float sq_sum = 0.0f;
            for (float norm : l2_norms) {
                sq_sum += (norm - current_bounds_.mean_l2_norm) * (norm - current_bounds_.mean_l2_norm);
            }
            current_bounds_.std_l2_norm = std::sqrt(sq_sum / l2_norms.size());
            
            // Set acceptable bounds
            float multiplier = config_.magnitude_violation_multiplier;
            current_bounds_.max_acceptable_norm = current_bounds_.mean_l2_norm + multiplier * current_bounds_.std_l2_norm;
            current_bounds_.min_acceptable_norm = std::max(0.0f, current_bounds_.mean_l2_norm - multiplier * current_bounds_.std_l2_norm);
        }
        
        // Calculate statistics for layer norms
        current_bounds_.layer_mean_norms.clear();
        current_bounds_.layer_std_norms.clear();
        
        for (const auto& layer_norms : layer_norms_by_layer) {
            if (!layer_norms.empty()) {
                float mean = std::accumulate(layer_norms.begin(), layer_norms.end(), 0.0f) / layer_norms.size();
                
                float sq_sum = 0.0f;
                for (float norm : layer_norms) {
                    sq_sum += (norm - mean) * (norm - mean);
                }
                float std_dev = std::sqrt(sq_sum / layer_norms.size());
                
                current_bounds_.layer_mean_norms.push_back(mean);
                current_bounds_.layer_std_norms.push_back(std_dev);
            }
        }
        
        current_bounds_.samples_count = gradient_history_.size();
        current_bounds_.last_updated = std::chrono::steady_clock::now();
    }
}

ByzantineDetectionEngine::DetectionResult ByzantineDetectionEngine::run_full_detection_pipeline(
    const std::vector<GradientFingerprint>& gradients) {
    
    logger::log_info("Running full Byzantine detection pipeline on " + 
                     std::to_string(gradients.size()) + " gradients");
    
    DetectionResult result;
    result.detection_confidence = 0.0f;
    
    try {
        // Update historical bounds
        update_historical_bounds(gradients);
        
        // 1. Gradient clustering analysis
        if (config_.enable_gradient_clustering) {
            result.cluster_analysis = analyze_gradient_similarity(gradients);
        }
        
        // 2. Outlier detection
        auto outliers = detect_outlier_nodes(gradients);
        
        // 3. Magnitude validation
        for (const auto& gradient : gradients) {
            if (!validate_gradient_magnitude(gradient.node_id, gradient)) {
                if (std::find(outliers.begin(), outliers.end(), gradient.node_id) == outliers.end()) {
                    outliers.push_back(gradient.node_id);
                }
            }
        }
        
        // 4. Check cross-validation failures
        auto cross_val_failures = get_cross_validation_failures();
        
        // 5. Get quarantined nodes
        result.quarantined_nodes = get_quarantined_nodes();
        
        // 6. Reputation-based classification
        std::set<std::string> byzantine_set, suspicious_set;
        
        for (const auto& node_id : outliers) {
            float reputation = get_node_reputation(node_id);
            
            if (reputation < config_.quarantine_threshold) {
                byzantine_set.insert(node_id);
            } else if (reputation < 0.7f) {
                suspicious_set.insert(node_id);
            }
            
            // Reduce reputation for being flagged as outlier
            update_node_reputation(node_id, -0.05f);
        }
        
        for (const auto& node_id : cross_val_failures) {
            float reputation = get_node_reputation(node_id);
            
            if (reputation < config_.quarantine_threshold) {
                byzantine_set.insert(node_id);
            } else {
                suspicious_set.insert(node_id);
            }
        }
        
        result.byzantine_nodes = std::vector<std::string>(byzantine_set.begin(), byzantine_set.end());
        result.suspicious_nodes = std::vector<std::string>(suspicious_set.begin(), suspicious_set.end());
        
        // 7. Calculate detection confidence
        float clustering_confidence = result.cluster_analysis.silhouette_score;
        float statistical_confidence = std::min(1.0f, static_cast<float>(current_bounds_.samples_count) / 50.0f);
        float cross_val_confidence = stats_.cross_validation_tasks_completed > 10 ? 0.8f : 
                                    static_cast<float>(stats_.cross_validation_tasks_completed) / 10.0f;
        
        result.detection_confidence = (clustering_confidence + statistical_confidence + cross_val_confidence) / 3.0f;
        
        // 8. Generate summary
        std::stringstream summary;
        summary << "Detection completed: ";
        summary << result.byzantine_nodes.size() << " Byzantine nodes, ";
        summary << result.suspicious_nodes.size() << " suspicious nodes, ";
        summary << result.quarantined_nodes.size() << " quarantined nodes. ";
        summary << "Confidence: " << std::fixed << std::setprecision(2) << (result.detection_confidence * 100) << "%";
        result.detection_summary = summary.str();
        
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.total_gradients_analyzed += gradients.size();
            stats_.last_detection_run = std::chrono::steady_clock::now();
        }
        
        logger::log_info(result.detection_summary);
        
        // Process reputation decay and quarantine expiries
        decay_reputations();
        process_quarantine_expiries();
        
    } catch (const std::exception& e) {
        logger::log_error("Detection pipeline failed: " + std::string(e.what()));
        result.detection_confidence = 0.0f;
        result.detection_summary = "Detection failed: " + std::string(e.what());
    }
    
    return result;
}

float ByzantineDetectionEngine::get_node_reputation(const std::string& node_id) const {
    std::lock_guard<std::mutex> lock(reputation_mutex_);
    
    auto it = node_reputations_.find(node_id);
    if (it != node_reputations_.end()) {
        return it->second;
    }
    
    return 1.0f; // Default reputation for new nodes
}

void ByzantineDetectionEngine::update_node_reputation(const std::string& node_id, float reputation_delta) {
    std::lock_guard<std::mutex> lock(reputation_mutex_);
    
    float current_reputation = node_reputations_[node_id];
    if (node_reputations_.find(node_id) == node_reputations_.end()) {
        current_reputation = 1.0f; // Default for new nodes
    }
    
    float new_reputation = std::max(0.0f, std::min(1.0f, current_reputation + reputation_delta));
    node_reputations_[node_id] = new_reputation;
    
    // Check for quarantine
    if (new_reputation < config_.quarantine_threshold && 
        quarantine_expiry_.find(node_id) == quarantine_expiry_.end()) {
        
        auto expiry_time = std::chrono::steady_clock::now() + 
                          std::chrono::minutes(config_.quarantine_duration_minutes);
        quarantine_expiry_[node_id] = expiry_time;
        
        logger::log_warning("Node " + node_id + " quarantined (reputation: " + 
                           std::to_string(new_reputation) + ")");
        
        {
            std::lock_guard<std::mutex> stats_lock(stats_mutex_);
            stats_.nodes_quarantined++;
        }
    }
    
    logger::log_debug("Updated reputation for " + node_id + ": " + 
                     std::to_string(current_reputation) + " -> " + std::to_string(new_reputation));
}

std::vector<std::string> ByzantineDetectionEngine::get_quarantined_nodes() const {
    std::lock_guard<std::mutex> lock(reputation_mutex_);
    
    std::vector<std::string> quarantined;
    auto now = std::chrono::steady_clock::now();
    
    for (const auto& [node_id, expiry_time] : quarantine_expiry_) {
        if (now < expiry_time) {
            quarantined.push_back(node_id);
        }
    }
    
    return quarantined;
}

bool ByzantineDetectionEngine::is_node_quarantined(const std::string& node_id) const {
    std::lock_guard<std::mutex> lock(reputation_mutex_);
    
    auto it = quarantine_expiry_.find(node_id);
    if (it != quarantine_expiry_.end()) {
        return std::chrono::steady_clock::now() < it->second;
    }
    
    return false;
}

void ByzantineDetectionEngine::report_suspicious_activity(const SuspiciousActivity& activity) {
    std::lock_guard<std::mutex> lock(activity_mutex_);
    
    suspicious_activities_.push_back(activity);
    
    // Keep only recent activities (last 24 hours)
    auto cutoff_time = std::chrono::system_clock::now().time_since_epoch().count() - (24 * 60 * 60 * 1000);
    
    suspicious_activities_.erase(
        std::remove_if(suspicious_activities_.begin(), suspicious_activities_.end(),
                      [cutoff_time](const SuspiciousActivity& act) {
                          return act.timestamp < cutoff_time;
                      }),
        suspicious_activities_.end());
    
    logger::log_warning("Suspicious activity reported: " + activity.node_id + " - " + 
                       activity.activity_type + " (" + activity.description + ")");
}

// Helper method implementations

std::vector<std::vector<size_t>> ByzantineDetectionEngine::perform_kmeans_clustering(
    const std::vector<std::vector<float>>& data, size_t k) const {
    
    if (data.empty() || k == 0) {
        return {};
    }
    
    size_t n = data.size();
    size_t dim = data[0].size();
    
    // Initialize centroids randomly
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dis(0, n - 1);
    
    std::vector<std::vector<float>> centroids(k);
    for (size_t i = 0; i < k; ++i) {
        centroids[i] = data[dis(gen)];
    }
    
    std::vector<std::vector<size_t>> clusters(k);
    
    // K-means iterations
    for (int iter = 0; iter < 10; ++iter) {
        // Clear previous assignments
        for (auto& cluster : clusters) {
            cluster.clear();
        }
        
        // Assign points to closest centroids
        for (size_t i = 0; i < n; ++i) {
            float min_dist = std::numeric_limits<float>::max();
            size_t best_centroid = 0;
            
            for (size_t j = 0; j < k; ++j) {
                float dist = 0.0f;
                for (size_t d = 0; d < dim; ++d) {
                    float diff = data[i][d] - centroids[j][d];
                    dist += diff * diff;
                }
                
                if (dist < min_dist) {
                    min_dist = dist;
                    best_centroid = j;
                }
            }
            
            clusters[best_centroid].push_back(i);
        }
        
        // Update centroids
        for (size_t j = 0; j < k; ++j) {
            if (!clusters[j].empty()) {
                std::fill(centroids[j].begin(), centroids[j].end(), 0.0f);
                
                for (size_t idx : clusters[j]) {
                    for (size_t d = 0; d < dim; ++d) {
                        centroids[j][d] += data[idx][d];
                    }
                }
                
                for (size_t d = 0; d < dim; ++d) {
                    centroids[j][d] /= clusters[j].size();
                }
            }
        }
    }
    
    return clusters;
}

float ByzantineDetectionEngine::calculate_silhouette_score(
    const std::vector<std::vector<float>>& data,
    const std::vector<std::vector<size_t>>& clusters) const {
    
    if (data.empty() || clusters.empty()) {
        return 0.0f;
    }
    
    auto distance = [](const std::vector<float>& a, const std::vector<float>& b) {
        float dist = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            float diff = a[i] - b[i];
            dist += diff * diff;
        }
        return std::sqrt(dist);
    };
    
    float total_silhouette = 0.0f;
    size_t total_points = 0;
    
    for (size_t cluster_idx = 0; cluster_idx < clusters.size(); ++cluster_idx) {
        const auto& cluster = clusters[cluster_idx];
        
        if (cluster.size() <= 1) continue;
        
        for (size_t point_idx : cluster) {
            // Calculate a(i): average distance to other points in same cluster
            float a_i = 0.0f;
            for (size_t other_idx : cluster) {
                if (other_idx != point_idx) {
                    a_i += distance(data[point_idx], data[other_idx]);
                }
            }
            if (cluster.size() > 1) {
                a_i /= (cluster.size() - 1);
            }
            
            // Calculate b(i): minimum average distance to points in other clusters
            float b_i = std::numeric_limits<float>::max();
            for (size_t other_cluster_idx = 0; other_cluster_idx < clusters.size(); ++other_cluster_idx) {
                if (other_cluster_idx == cluster_idx) continue;
                
                const auto& other_cluster = clusters[other_cluster_idx];
                if (other_cluster.empty()) continue;
                
                float avg_dist = 0.0f;
                for (size_t other_point_idx : other_cluster) {
                    avg_dist += distance(data[point_idx], data[other_point_idx]);
                }
                avg_dist /= other_cluster.size();
                
                b_i = std::min(b_i, avg_dist);
            }
            
            // Calculate silhouette coefficient
            if (b_i < std::numeric_limits<float>::max()) {
                float s_i = (b_i - a_i) / std::max(a_i, b_i);
                total_silhouette += s_i;
                total_points++;
            }
        }
    }
    
    return total_points > 0 ? total_silhouette / total_points : 0.0f;
}

CrossValidationTask ByzantineDetectionEngine::create_synthetic_validation_task(
    const std::vector<std::string>& nodes) {
    
    CrossValidationTask task;
    task.task_id = generate_cross_validation_task_id();
    task.assigned_nodes = nodes;
    task.created_timestamp = static_cast<uint32_t>(
        std::chrono::system_clock::now().time_since_epoch().count());
    
    // Create a simple mathematical task: compute gradient of quadratic function
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    // Generate test input: [a, b, c, x] where f(x) = ax^2 + bx + c
    task.input_data = {dis(gen), dis(gen), dis(gen), dis(gen)};
    
    // Expected output: gradient df/dx = 2ax + b
    float a = task.input_data[0];
    float b = task.input_data[1];
    float x = task.input_data[3];
    float expected_gradient = 2 * a * x + b;
    
    task.expected_output = {expected_gradient};
    task.tolerance = 0.01f; // 1% tolerance
    
    return task;
}

std::string ByzantineDetectionEngine::generate_cross_validation_task_id() {
    return "cv_task_" + std::to_string(next_task_id_++);
}

void ByzantineDetectionEngine::decay_reputations() {
    std::lock_guard<std::mutex> lock(reputation_mutex_);
    
    for (auto& [node_id, reputation] : node_reputations_) {
        // Slowly recover reputation over time (if above quarantine threshold)
        if (reputation >= config_.quarantine_threshold) {
            reputation = std::min(1.0f, reputation + (1.0f - reputation) * (1.0f - config_.reputation_decay_rate));
        }
    }
}

void ByzantineDetectionEngine::process_quarantine_expiries() {
    std::lock_guard<std::mutex> lock(reputation_mutex_);
    
    auto now = std::chrono::steady_clock::now();
    auto it = quarantine_expiry_.begin();
    
    while (it != quarantine_expiry_.end()) {
        if (now >= it->second) {
            logger::log_info("Quarantine expired for node " + it->first);
            it = quarantine_expiry_.erase(it);
        } else {
            ++it;
        }
    }
}

ByzantineDetectionEngine::DetectionStats ByzantineDetectionEngine::get_detection_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

} // namespace byzantine
