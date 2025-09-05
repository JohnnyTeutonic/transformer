#include "../include/distributed_curation.hpp"
#include "../include/utils.hpp"
#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>

namespace curation {

DistributedCurationPlatform::DistributedCurationPlatform(
    std::shared_ptr<p2p::P2PNetwork> network,
    const CurationConfig& config)
    : network_(network), config_(config) {
    
    std::cout << "Initializing Distributed Curation Platform" << std::endl;
    std::cout << "- Min annotators per task: " << config_.min_annotators_per_task << std::endl;
    std::cout << "- Consensus threshold: " << config_.consensus_threshold << std::endl;
    std::cout << "- Base annotation reward: " << config_.base_annotation_reward << std::endl;
}

DistributedCurationPlatform::~DistributedCurationPlatform() {
    stop();
}

bool DistributedCurationPlatform::start() {
    if (running_.load()) {
        std::cout << "Curation platform already running" << std::endl;
        return true;
    }

    std::cout << "Starting distributed curation platform..." << std::endl;
    
    // Register P2P message handlers
    network_->register_message_handler(p2p::MessageType::CURATION_TASK_SUBMISSION,
        [this](const p2p::NetworkMessage& msg) { handle_task_submission(msg); });
    
    network_->register_message_handler(p2p::MessageType::CURATION_ANNOTATION_SUBMISSION,
        [this](const p2p::NetworkMessage& msg) { handle_annotation_submission(msg); });
    
    network_->register_message_handler(p2p::MessageType::CURATION_CONSENSUS_PROPOSAL,
        [this](const p2p::NetworkMessage& msg) { handle_consensus_proposal(msg); });
    
    network_->register_message_handler(p2p::MessageType::CURATION_REPUTATION_UPDATE,
        [this](const p2p::NetworkMessage& msg) { handle_reputation_update(msg); });

    // Start background processing threads
    running_.store(true);
    worker_threads_.emplace_back(&DistributedCurationPlatform::consensus_processing_thread, this);
    worker_threads_.emplace_back(&DistributedCurationPlatform::reputation_update_thread, this);
    worker_threads_.emplace_back(&DistributedCurationPlatform::quality_monitoring_thread, this);

    std::cout << "Distributed curation platform started successfully" << std::endl;
    return true;
}

void DistributedCurationPlatform::stop() {
    if (!running_.load()) {
        return;
    }

    std::cout << "Stopping distributed curation platform..." << std::endl;
    running_.store(false);

    // Wait for worker threads to finish
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();

    std::cout << "Distributed curation platform stopped" << std::endl;
}

std::string DistributedCurationPlatform::submit_annotation_task(const AnnotationTask& task) {
    std::lock_guard<std::mutex> lock(tasks_mutex_);
    
    // Generate unique task ID
    std::string task_id = "task_" + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count()) + "_" + 
        std::to_string(std::hash<std::string>{}(task.content));

    // Create task with ID
    AnnotationTask new_task = task;
    new_task.task_id = task_id;
    new_task.created_timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    // Store task locally
    active_tasks_[task_id] = new_task;
    task_submissions_[task_id] = std::vector<AnnotationSubmission>();

    // Broadcast task to network
    p2p::NetworkMessage message;
    message.type = p2p::MessageType::CURATION_TASK_SUBMISSION;
    message.sender_id = network_->get_node_id();
    message.payload = utils::serialize_annotation_task(new_task);
    
    network_->broadcast_message(message);

    std::cout << "Submitted annotation task: " << task_id << std::endl;
    return task_id;
}

bool DistributedCurationPlatform::cancel_task(const std::string& task_id) {
    std::lock_guard<std::mutex> lock(tasks_mutex_);
    
    auto it = active_tasks_.find(task_id);
    if (it == active_tasks_.end()) {
        return false;
    }

    // Remove from active tasks
    active_tasks_.erase(it);
    task_submissions_.erase(task_id);

    // Broadcast cancellation
    p2p::NetworkMessage message;
    message.type = p2p::MessageType::CURATION_TASK_CANCELLATION;
    message.sender_id = network_->get_node_id();
    message.payload = std::vector<uint8_t>(task_id.begin(), task_id.end());
    
    network_->broadcast_message(message);

    std::cout << "Cancelled annotation task: " << task_id << std::endl;
    return true;
}

std::vector<AnnotationTask> DistributedCurationPlatform::get_available_tasks(const std::string& annotator_id) {
    std::lock_guard<std::mutex> lock(tasks_mutex_);
    std::lock_guard<std::mutex> annotator_lock(annotators_mutex_);
    
    std::vector<AnnotationTask> available_tasks;
    
    // Get annotator profile to check reputation and specializations
    auto annotator_it = annotator_profiles_.find(annotator_id);
    if (annotator_it == annotator_profiles_.end()) {
        return available_tasks; // Annotator not registered
    }
    
    const AnnotatorProfile& profile = annotator_it->second;
    
    // Check minimum reputation requirement
    if (profile.reputation_score < config_.min_reputation_for_tasks) {
        return available_tasks;
    }

    // Find suitable tasks
    for (const auto& [task_id, task] : active_tasks_) {
        // Check if task needs more annotators
        auto submissions_it = task_submissions_.find(task_id);
        uint32_t current_annotators = (submissions_it != task_submissions_.end()) ? 
            submissions_it->second.size() : 0;
        
        if (current_annotators >= task.required_annotators) {
            continue; // Task already has enough annotators
        }

        // Check if annotator already submitted for this task
        bool already_submitted = false;
        if (submissions_it != task_submissions_.end()) {
            for (const auto& submission : submissions_it->second) {
                if (submission.annotator_id == annotator_id) {
                    already_submitted = true;
                    break;
                }
            }
        }
        
        if (already_submitted) {
            continue;
        }

        // Check task timeout
        uint64_t current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        uint64_t task_age_hours = (current_time - task.created_timestamp) / (1000 * 60 * 60);
        
        if (task_age_hours > config_.task_timeout_hours) {
            continue; // Task expired
        }

        available_tasks.push_back(task);
    }

    // Sort by difficulty and reward potential
    std::sort(available_tasks.begin(), available_tasks.end(),
        [](const AnnotationTask& a, const AnnotationTask& b) {
            return a.difficulty_score > b.difficulty_score; // Higher difficulty first
        });

    // Limit to max concurrent tasks
    if (available_tasks.size() > config_.max_concurrent_tasks_per_annotator) {
        available_tasks.resize(config_.max_concurrent_tasks_per_annotator);
    }

    return available_tasks;
}

bool DistributedCurationPlatform::submit_annotation(const AnnotationSubmission& submission) {
    std::lock_guard<std::mutex> lock(tasks_mutex_);
    
    // Verify task exists
    auto task_it = active_tasks_.find(submission.task_id);
    if (task_it == active_tasks_.end()) {
        std::cerr << "Cannot submit annotation for non-existent task: " << submission.task_id << std::endl;
        return false;
    }

    // Verify annotator is registered
    {
        std::lock_guard<std::mutex> annotator_lock(annotators_mutex_);
        auto annotator_it = annotator_profiles_.find(submission.annotator_id);
        if (annotator_it == annotator_profiles_.end()) {
            std::cerr << "Annotation from unregistered annotator: " << submission.annotator_id << std::endl;
            return false;
        }

        // Verify signature
        if (!utils::verify_annotation_signature(submission, annotator_it->second.public_key)) {
            std::cerr << "Invalid annotation signature from: " << submission.annotator_id << std::endl;
            return false;
        }
    }

    // Check if annotator already submitted for this task
    auto& submissions = task_submissions_[submission.task_id];
    for (const auto& existing : submissions) {
        if (existing.annotator_id == submission.annotator_id) {
            std::cerr << "Duplicate annotation from: " << submission.annotator_id << std::endl;
            return false;
        }
    }

    // Add submission
    submissions.push_back(submission);

    // Broadcast submission to network
    p2p::NetworkMessage message;
    message.type = p2p::MessageType::CURATION_ANNOTATION_SUBMISSION;
    message.sender_id = network_->get_node_id();
    message.payload = utils::serialize_annotation_submission(submission);
    
    network_->broadcast_message(message);

    // Check if we have enough submissions for consensus
    if (submissions.size() >= task_it->second.required_annotators) {
        // Trigger consensus computation
        compute_annotation_consensus(submission.task_id);
    }

    std::cout << "Received annotation for task " << submission.task_id 
              << " from " << submission.annotator_id 
              << " (" << submissions.size() << "/" << task_it->second.required_annotators << ")" << std::endl;

    return true;
}

std::optional<AnnotationConsensus> DistributedCurationPlatform::get_task_consensus(const std::string& task_id) {
    std::lock_guard<std::mutex> lock(consensus_mutex_);
    
    auto it = completed_tasks_.find(task_id);
    if (it != completed_tasks_.end()) {
        return it->second;
    }
    
    return std::nullopt;
}

std::vector<AnnotationConsensus> DistributedCurationPlatform::get_completed_annotations(uint32_t limit) {
    std::lock_guard<std::mutex> lock(consensus_mutex_);
    
    std::vector<AnnotationConsensus> results;
    results.reserve(std::min(limit, static_cast<uint32_t>(completed_tasks_.size())));
    
    for (const auto& [task_id, consensus] : completed_tasks_) {
        if (results.size() >= limit) break;
        results.push_back(consensus);
    }
    
    return results;
}

bool DistributedCurationPlatform::register_annotator(const AnnotatorProfile& profile) {
    std::lock_guard<std::mutex> lock(annotators_mutex_);
    
    // Verify the annotator doesn't already exist
    if (annotator_profiles_.find(profile.annotator_id) != annotator_profiles_.end()) {
        std::cerr << "Annotator already registered: " << profile.annotator_id << std::endl;
        return false;
    }

    // Add annotator profile
    annotator_profiles_[profile.annotator_id] = profile;

    // Broadcast registration to network
    p2p::NetworkMessage message;
    message.type = p2p::MessageType::CURATION_ANNOTATOR_REGISTRATION;
    message.sender_id = network_->get_node_id();
    
    // Serialize annotator profile
    serialization::Serializer s;
    s.write_string(profile.annotator_id);
    s.write_string(profile.public_key);
    s.write_trivial(profile.reputation_score);
    s.write_uint32(profile.total_annotations);
    s.write_uint32(profile.consensus_agreements);
    s.write_trivial(profile.average_confidence);
    s.write_uint32(static_cast<uint32_t>(profile.specializations.size()));
    for (const auto& spec : profile.specializations) {
        s.write_string(spec);
    }
    s.write_uint64(profile.last_active);
    s.write_uint8(profile.is_verified ? 1 : 0);
    
    message.payload = s.take_buffer();
    network_->broadcast_message(message);

    // Trigger callback
    if (annotator_joined_callback_) {
        annotator_joined_callback_(profile);
    }

    std::cout << "Registered new annotator: " << profile.annotator_id 
              << " (reputation: " << profile.reputation_score << ")" << std::endl;
    
    return true;
}

bool DistributedCurationPlatform::update_annotator_profile(const AnnotatorProfile& profile) {
    std::lock_guard<std::mutex> lock(annotators_mutex_);
    
    auto it = annotator_profiles_.find(profile.annotator_id);
    if (it == annotator_profiles_.end()) {
        return false;
    }

    // Update profile
    it->second = profile;
    
    std::cout << "Updated annotator profile: " << profile.annotator_id << std::endl;
    return true;
}

std::optional<AnnotatorProfile> DistributedCurationPlatform::get_annotator_profile(const std::string& annotator_id) {
    std::lock_guard<std::mutex> lock(annotators_mutex_);
    
    auto it = annotator_profiles_.find(annotator_id);
    if (it != annotator_profiles_.end()) {
        return it->second;
    }
    
    return std::nullopt;
}

std::vector<AnnotatorProfile> DistributedCurationPlatform::get_top_annotators(uint32_t limit) {
    std::lock_guard<std::mutex> lock(annotators_mutex_);
    
    std::vector<AnnotatorProfile> annotators;
    annotators.reserve(annotator_profiles_.size());
    
    for (const auto& [id, profile] : annotator_profiles_) {
        annotators.push_back(profile);
    }
    
    // Sort by reputation score
    std::sort(annotators.begin(), annotators.end(),
        [](const AnnotatorProfile& a, const AnnotatorProfile& b) {
            return a.reputation_score > b.reputation_score;
        });
    
    if (annotators.size() > limit) {
        annotators.resize(limit);
    }
    
    return annotators;
}

QualityMetrics DistributedCurationPlatform::compute_quality_metrics(const std::string& task_id) {
    std::lock_guard<std::mutex> lock(tasks_mutex_);
    
    QualityMetrics metrics{};
    
    auto submissions_it = task_submissions_.find(task_id);
    if (submissions_it == task_submissions_.end() || submissions_it->second.empty()) {
        return metrics;
    }
    
    const auto& submissions = submissions_it->second;
    
    // Compute inter-annotator agreement
    metrics.inter_annotator_agreement = utils::compute_inter_annotator_agreement(submissions);
    
    // Compute consensus strength
    if (!submissions.empty() && !submissions[0].labels.empty()) {
        std::vector<AnnotationLabel> all_labels;
        for (const auto& submission : submissions) {
            all_labels.insert(all_labels.end(), submission.labels.begin(), submission.labels.end());
        }
        metrics.consensus_strength = utils::compute_consensus_strength(all_labels);
    }
    
    // Compute time efficiency (average time to complete)
    if (submissions.size() > 1) {
        uint64_t total_time = 0;
        uint64_t min_time = submissions[0].submission_timestamp;
        uint64_t max_time = submissions[0].submission_timestamp;
        
        for (const auto& submission : submissions) {
            min_time = std::min(min_time, submission.submission_timestamp);
            max_time = std::max(max_time, submission.submission_timestamp);
        }
        
        metrics.time_efficiency = 1.0f / std::max(1.0f, static_cast<float>(max_time - min_time) / (1000.0f * 60.0f * 60.0f)); // Inverse of hours
    }
    
    return metrics;
}

void DistributedCurationPlatform::update_annotator_reputation(const std::string& annotator_id, float delta) {
    std::lock_guard<std::mutex> lock(annotators_mutex_);
    
    auto it = annotator_profiles_.find(annotator_id);
    if (it != annotator_profiles_.end()) {
        it->second.reputation_score = std::max(0.0f, std::min(1.0f, it->second.reputation_score + delta));
        
        // Broadcast reputation update
        p2p::NetworkMessage message;
        message.type = p2p::MessageType::CURATION_REPUTATION_UPDATE;
        message.sender_id = network_->get_node_id();
        
        serialization::Serializer s;
        s.write_string(annotator_id);
        s.write_trivial(it->second.reputation_score);
        message.payload = s.take_buffer();
        
        network_->broadcast_message(message);
    }
}

DistributedCurationPlatform::PlatformStats DistributedCurationPlatform::get_platform_stats() {
    std::lock_guard<std::mutex> tasks_lock(tasks_mutex_);
    std::lock_guard<std::mutex> consensus_lock(consensus_mutex_);
    std::lock_guard<std::mutex> annotators_lock(annotators_mutex_);
    
    PlatformStats stats{};
    
    stats.total_tasks = active_tasks_.size() + completed_tasks_.size();
    stats.completed_tasks = completed_tasks_.size();
    stats.active_annotators = annotator_profiles_.size();
    
    // Compute average consensus time
    if (!completed_tasks_.empty()) {
        float total_time = 0.0f;
        uint32_t count = 0;
        
        for (const auto& [task_id, consensus] : completed_tasks_) {
            auto task_it = active_tasks_.find(task_id);
            if (task_it != active_tasks_.end()) {
                // Find the latest submission time
                auto submissions_it = task_submissions_.find(task_id);
                if (submissions_it != task_submissions_.end() && !submissions_it->second.empty()) {
                    uint64_t latest_submission = 0;
                    for (const auto& submission : submissions_it->second) {
                        latest_submission = std::max(latest_submission, submission.submission_timestamp);
                    }
                    
                    float task_time = static_cast<float>(latest_submission - task_it->second.created_timestamp) / (1000.0f * 60.0f * 60.0f);
                    total_time += task_time;
                    count++;
                }
            }
        }
        
        if (count > 0) {
            stats.average_consensus_time_hours = total_time / count;
        }
    }
    
    // Compute platform quality score (average of completed task consensus strengths)
    if (!completed_tasks_.empty()) {
        float total_quality = 0.0f;
        for (const auto& [task_id, consensus] : completed_tasks_) {
            total_quality += consensus.consensus_confidence;
        }
        stats.platform_quality_score = total_quality / completed_tasks_.size();
    }
    
    // Count tasks by type
    for (const auto& [task_id, task] : active_tasks_) {
        stats.tasks_by_type[task.data_type]++;
    }
    
    return stats;
}

// Background processing threads
void DistributedCurationPlatform::consensus_processing_thread() {
    std::cout << "Started consensus processing thread" << std::endl;
    
    while (running_.load()) {
        std::vector<std::string> ready_tasks;
        
        // Find tasks ready for consensus
        {
            std::lock_guard<std::mutex> lock(tasks_mutex_);
            for (const auto& [task_id, task] : active_tasks_) {
                auto submissions_it = task_submissions_.find(task_id);
                if (submissions_it != task_submissions_.end() && 
                    submissions_it->second.size() >= task.required_annotators) {
                    ready_tasks.push_back(task_id);
                }
            }
        }
        
        // Process consensus for ready tasks
        for (const auto& task_id : ready_tasks) {
            compute_annotation_consensus(task_id);
        }
        
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
    
    std::cout << "Consensus processing thread stopped" << std::endl;
}

void DistributedCurationPlatform::reputation_update_thread() {
    std::cout << "Started reputation update thread" << std::endl;
    
    while (running_.load()) {
        // Apply daily reputation decay
        {
            std::lock_guard<std::mutex> lock(annotators_mutex_);
            for (auto& [id, profile] : annotator_profiles_) {
                profile.reputation_score *= (1.0f - config_.reputation_decay_rate);
                profile.reputation_score = std::max(0.0f, profile.reputation_score);
            }
        }
        
        std::this_thread::sleep_for(std::chrono::hours(24)); // Daily update
    }
    
    std::cout << "Reputation update thread stopped" << std::endl;
}

void DistributedCurationPlatform::quality_monitoring_thread() {
    std::cout << "Started quality monitoring thread" << std::endl;
    
    while (running_.load()) {
        // Monitor quality metrics and detect issues
        std::vector<std::string> completed_task_ids;
        
        {
            std::lock_guard<std::mutex> lock(consensus_mutex_);
            for (const auto& [task_id, consensus] : completed_tasks_) {
                completed_task_ids.push_back(task_id);
            }
        }
        
        // Check quality metrics for recent tasks
        for (const auto& task_id : completed_task_ids) {
            QualityMetrics metrics = compute_quality_metrics(task_id);
            
            // Trigger quality alert if metrics are poor
            if (metrics.inter_annotator_agreement < 0.5f || metrics.consensus_strength < 0.6f) {
                if (quality_alert_callback_) {
                    quality_alert_callback_(task_id, metrics);
                }
            }
        }
        
        std::this_thread::sleep_for(std::chrono::minutes(30));
    }
    
    std::cout << "Quality monitoring thread stopped" << std::endl;
}

// P2P message handlers
void DistributedCurationPlatform::handle_task_submission(const p2p::NetworkMessage& message) {
    try {
        AnnotationTask task = utils::deserialize_annotation_task(message.payload);
        
        std::lock_guard<std::mutex> lock(tasks_mutex_);
        
        // Add task if not already present
        if (active_tasks_.find(task.task_id) == active_tasks_.end()) {
            active_tasks_[task.task_id] = task;
            task_submissions_[task.task_id] = std::vector<AnnotationSubmission>();
            
            std::cout << "Received new annotation task: " << task.task_id << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error handling task submission: " << e.what() << std::endl;
    }
}

void DistributedCurationPlatform::handle_annotation_submission(const p2p::NetworkMessage& message) {
    try {
        AnnotationSubmission submission = utils::deserialize_annotation_submission(message.payload);
        
        // Process the submission (this will handle validation and storage)
        submit_annotation(submission);
    } catch (const std::exception& e) {
        std::cerr << "Error handling annotation submission: " << e.what() << std::endl;
    }
}

void DistributedCurationPlatform::handle_consensus_proposal(const p2p::NetworkMessage& message) {
    // Handle consensus proposals from other nodes
    // This would implement the BFT consensus protocol for annotation results
    std::cout << "Received consensus proposal from " << message.sender_id << std::endl;
}

void DistributedCurationPlatform::handle_reputation_update(const p2p::NetworkMessage& message) {
    try {
        serialization::Deserializer d(message.payload);
        std::string annotator_id = d.read_string();
        float new_reputation = d.read_trivial<float>();
        
        std::lock_guard<std::mutex> lock(annotators_mutex_);
        auto it = annotator_profiles_.find(annotator_id);
        if (it != annotator_profiles_.end()) {
            it->second.reputation_score = new_reputation;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error handling reputation update: " << e.what() << std::endl;
    }
}

bool DistributedCurationPlatform::compute_annotation_consensus(const std::string& task_id) {
    std::lock_guard<std::mutex> tasks_lock(tasks_mutex_);
    std::lock_guard<std::mutex> consensus_lock(consensus_mutex_);
    
    auto task_it = active_tasks_.find(task_id);
    auto submissions_it = task_submissions_.find(task_id);
    
    if (task_it == active_tasks_.end() || submissions_it == task_submissions_.end()) {
        return false;
    }
    
    const auto& task = task_it->second;
    const auto& submissions = submissions_it->second;
    
    if (submissions.size() < config_.min_annotators_per_task) {
        return false;
    }
    
    AnnotationConsensus consensus;
    consensus.task_id = task_id;
    consensus.total_submissions = submissions.size();
    
    // Compute consensus for each label type
    std::map<std::string, std::vector<float>> label_scores;
    
    for (const auto& submission : submissions) {
        for (const auto& label : submission.labels) {
            label_scores[label.label_type].push_back(label.score);
        }
    }
    
    // Create consensus labels
    for (const auto& [label_type, scores] : label_scores) {
        if (scores.empty()) continue;
        
        AnnotationLabel consensus_label;
        consensus_label.label_type = label_type;
        
        // Use median as consensus score
        std::vector<float> sorted_scores = scores;
        std::sort(sorted_scores.begin(), sorted_scores.end());
        
        if (sorted_scores.size() % 2 == 0) {
            consensus_label.score = (sorted_scores[sorted_scores.size()/2 - 1] + 
                                   sorted_scores[sorted_scores.size()/2]) / 2.0f;
        } else {
            consensus_label.score = sorted_scores[sorted_scores.size()/2];
        }
        
        consensus_label.annotator_id = "consensus";
        consensus_label.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        consensus.consensus_labels.push_back(consensus_label);
    }
    
    // Compute consensus confidence
    consensus.consensus_confidence = utils::compute_consensus_strength(consensus.consensus_labels);
    
    // Compute annotator agreements
    for (const auto& submission : submissions) {
        float agreement = compute_annotator_agreement(submission.annotator_id, consensus);
        consensus.annotator_agreements[submission.annotator_id] = agreement;
        
        // Update annotator reputation based on agreement
        float reputation_delta = (agreement - 0.5f) * 0.1f; // Scale agreement to reputation change
        update_annotator_reputation(submission.annotator_id, reputation_delta);
    }
    
    consensus.is_finalized = true;
    
    // Store consensus result
    completed_tasks_[task_id] = consensus;
    
    // Remove from active tasks
    active_tasks_.erase(task_it);
    task_submissions_.erase(submissions_it);
    
    // Broadcast consensus result
    broadcast_consensus_result(consensus);
    
    // Trigger callback
    if (task_completed_callback_) {
        task_completed_callback_(consensus);
    }
    
    std::cout << "Computed consensus for task " << task_id 
              << " (confidence: " << consensus.consensus_confidence << ")" << std::endl;
    
    return true;
}

float DistributedCurationPlatform::compute_label_consensus(const std::vector<AnnotationLabel>& labels) {
    if (labels.empty()) return 0.0f;
    
    // Compute variance of scores as inverse of consensus
    float mean = 0.0f;
    for (const auto& label : labels) {
        mean += label.score;
    }
    mean /= labels.size();
    
    float variance = 0.0f;
    for (const auto& label : labels) {
        float diff = label.score - mean;
        variance += diff * diff;
    }
    variance /= labels.size();
    
    // Convert variance to consensus score (lower variance = higher consensus)
    return 1.0f / (1.0f + variance);
}

void DistributedCurationPlatform::broadcast_consensus_result(const AnnotationConsensus& consensus) {
    p2p::NetworkMessage message;
    message.type = p2p::MessageType::CURATION_CONSENSUS_RESULT;
    message.sender_id = network_->get_node_id();
    
    // Serialize consensus result
    serialization::Serializer s;
    s.write_string(consensus.task_id);
    s.write_uint32(static_cast<uint32_t>(consensus.consensus_labels.size()));
    
    for (const auto& label : consensus.consensus_labels) {
        s.write_string(label.label_type);
        s.write_trivial(label.score);
        s.write_string(label.text_feedback);
        s.write_uint64(label.timestamp);
    }
    
    s.write_trivial(consensus.consensus_confidence);
    s.write_uint32(consensus.total_submissions);
    s.write_uint8(consensus.is_finalized ? 1 : 0);
    
    message.payload = s.take_buffer();
    network_->broadcast_message(message);
}

float DistributedCurationPlatform::compute_annotator_agreement(const std::string& annotator_id, 
                                                             const AnnotationConsensus& consensus) {
    auto submissions_it = task_submissions_.find(consensus.task_id);
    if (submissions_it == task_submissions_.end()) {
        return 0.0f;
    }
    
    // Find annotator's submission
    const AnnotationSubmission* annotator_submission = nullptr;
    for (const auto& submission : submissions_it->second) {
        if (submission.annotator_id == annotator_id) {
            annotator_submission = &submission;
            break;
        }
    }
    
    if (!annotator_submission) {
        return 0.0f;
    }
    
    // Compute agreement between annotator's labels and consensus
    float total_agreement = 0.0f;
    uint32_t matched_labels = 0;
    
    for (const auto& consensus_label : consensus.consensus_labels) {
        for (const auto& annotator_label : annotator_submission->labels) {
            if (consensus_label.label_type == annotator_label.label_type) {
                float score_diff = std::abs(consensus_label.score - annotator_label.score);
                float agreement = 1.0f - score_diff; // Agreement is inverse of difference
                total_agreement += std::max(0.0f, agreement);
                matched_labels++;
                break;
            }
        }
    }
    
    return matched_labels > 0 ? total_agreement / matched_labels : 0.0f;
}

// Callback setters
void DistributedCurationPlatform::set_task_completed_callback(TaskCompletedCallback callback) {
    task_completed_callback_ = callback;
}

void DistributedCurationPlatform::set_annotator_joined_callback(AnnotatorJoinedCallback callback) {
    annotator_joined_callback_ = callback;
}

void DistributedCurationPlatform::set_quality_alert_callback(QualityAlertCallback callback) {
    quality_alert_callback_ = callback;
}

// Utility function implementations
namespace utils {

std::vector<uint8_t> serialize_annotation_task(const AnnotationTask& task) {
    serialization::Serializer s;
    
    s.write_string(task.task_id);
    s.write_string(task.data_type);
    s.write_string(task.content);
    s.write_string(task.context);
    
    s.write_uint32(static_cast<uint32_t>(task.label_schema.size()));
    for (const auto& schema : task.label_schema) {
        s.write_string(schema);
    }
    
    s.write_uint64(task.created_timestamp);
    s.write_uint32(task.required_annotators);
    s.write_trivial(task.difficulty_score);
    
    return s.take_buffer();
}

AnnotationTask deserialize_annotation_task(const std::vector<uint8_t>& data) {
    serialization::Deserializer d(data);
    
    AnnotationTask task;
    task.task_id = d.read_string();
    task.data_type = d.read_string();
    task.content = d.read_string();
    task.context = d.read_string();
    
    uint32_t schema_count = d.read_uint32();
    task.label_schema.reserve(schema_count);
    for (uint32_t i = 0; i < schema_count; ++i) {
        task.label_schema.push_back(d.read_string());
    }
    
    task.created_timestamp = d.read_uint64();
    task.required_annotators = d.read_uint32();
    task.difficulty_score = d.read_trivial<float>();
    
    return task;
}

std::vector<uint8_t> serialize_annotation_submission(const AnnotationSubmission& submission) {
    serialization::Serializer s;
    
    s.write_string(submission.task_id);
    s.write_string(submission.annotator_id);
    
    s.write_uint32(static_cast<uint32_t>(submission.labels.size()));
    for (const auto& label : submission.labels) {
        s.write_string(label.label_type);
        s.write_trivial(label.score);
        s.write_string(label.text_feedback);
        s.write_string(label.annotator_id);
        s.write_uint64(label.timestamp);
    }
    
    s.write_uint64(submission.submission_timestamp);
    s.write_trivial(submission.confidence_score);
    s.write_string(submission.signature);
    
    return s.take_buffer();
}

AnnotationSubmission deserialize_annotation_submission(const std::vector<uint8_t>& data) {
    serialization::Deserializer d(data);
    
    AnnotationSubmission submission;
    submission.task_id = d.read_string();
    submission.annotator_id = d.read_string();
    
    uint32_t label_count = d.read_uint32();
    submission.labels.reserve(label_count);
    for (uint32_t i = 0; i < label_count; ++i) {
        AnnotationLabel label;
        label.label_type = d.read_string();
        label.score = d.read_trivial<float>();
        label.text_feedback = d.read_string();
        label.annotator_id = d.read_string();
        label.timestamp = d.read_uint64();
        submission.labels.push_back(label);
    }
    
    submission.submission_timestamp = d.read_uint64();
    submission.confidence_score = d.read_trivial<float>();
    submission.signature = d.read_string();
    
    return submission;
}

float compute_inter_annotator_agreement(const std::vector<AnnotationSubmission>& submissions) {
    if (submissions.size() < 2) return 1.0f;
    
    // Compute pairwise agreements and average them
    float total_agreement = 0.0f;
    uint32_t pairs = 0;
    
    for (size_t i = 0; i < submissions.size(); ++i) {
        for (size_t j = i + 1; j < submissions.size(); ++j) {
            const auto& sub_i = submissions[i];
            const auto& sub_j = submissions[j];
            
            // Find matching labels
            float pair_agreement = 0.0f;
            uint32_t matched_labels = 0;
            
            for (const auto& label_i : sub_i.labels) {
                for (const auto& label_j : sub_j.labels) {
                    if (label_i.label_type == label_j.label_type) {
                        float score_diff = std::abs(label_i.score - label_j.score);
                        pair_agreement += 1.0f - score_diff;
                        matched_labels++;
                        break;
                    }
                }
            }
            
            if (matched_labels > 0) {
                total_agreement += pair_agreement / matched_labels;
                pairs++;
            }
        }
    }
    
    return pairs > 0 ? total_agreement / pairs : 0.0f;
}

float compute_consensus_strength(const std::vector<AnnotationLabel>& labels) {
    if (labels.empty()) return 0.0f;
    
    // Group labels by type and compute variance
    std::map<std::string, std::vector<float>> label_groups;
    for (const auto& label : labels) {
        label_groups[label.label_type].push_back(label.score);
    }
    
    float total_strength = 0.0f;
    uint32_t label_types = 0;
    
    for (const auto& [type, scores] : label_groups) {
        if (scores.size() < 2) {
            total_strength += 1.0f; // Perfect consensus for single score
        } else {
            // Compute variance
            float mean = 0.0f;
            for (float score : scores) {
                mean += score;
            }
            mean /= scores.size();
            
            float variance = 0.0f;
            for (float score : scores) {
                float diff = score - mean;
                variance += diff * diff;
            }
            variance /= scores.size();
            
            // Convert to consensus strength
            total_strength += 1.0f / (1.0f + variance);
        }
        label_types++;
    }
    
    return label_types > 0 ? total_strength / label_types : 0.0f;
}

std::string sign_annotation(const AnnotationSubmission& submission, const std::string& private_key) {
    // This would implement actual cryptographic signing
    // For now, return a placeholder
    return "signature_" + submission.task_id + "_" + submission.annotator_id;
}

bool verify_annotation_signature(const AnnotationSubmission& submission, const std::string& public_key) {
    // This would implement actual signature verification
    // For now, just check if signature follows expected format
    std::string expected_prefix = "signature_" + submission.task_id + "_" + submission.annotator_id;
    return submission.signature.find(expected_prefix) == 0;
}

} // namespace utils

} // namespace curation
