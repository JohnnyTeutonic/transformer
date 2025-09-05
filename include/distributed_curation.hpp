#pragma once

#include "p2p_network.hpp"
#include "matrix.hpp"
#include "serialization.hpp"
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include <chrono>
#include <atomic>
#include <mutex>

namespace curation {

// Data structures for annotation tasks
struct AnnotationLabel {
    std::string label_type;     // "quality", "safety", "helpfulness", etc.
    float score;                // 0.0 to 1.0
    std::string text_feedback;  // Optional text explanation
    std::string annotator_id;
    uint64_t timestamp;
};

struct AnnotationTask {
    std::string task_id;
    std::string data_type;      // "text", "conversation", "code", etc.
    std::string content;        // The actual data to annotate
    std::string context;        // Additional context if needed
    std::vector<std::string> label_schema;  // What to annotate for
    uint64_t created_timestamp;
    uint32_t required_annotators;  // How many annotations needed
    float difficulty_score;     // 0.0 to 1.0, affects reward
};

struct AnnotationSubmission {
    std::string task_id;
    std::string annotator_id;
    std::vector<AnnotationLabel> labels;
    uint64_t submission_timestamp;
    float confidence_score;     // Annotator's confidence in their labels
    std::string signature;      // Cryptographic signature
};

struct AnnotationConsensus {
    std::string task_id;
    std::vector<AnnotationLabel> consensus_labels;
    float consensus_confidence;
    std::map<std::string, float> annotator_agreements;  // Agreement with consensus
    uint32_t total_submissions;
    bool is_finalized;
};

// Annotator reputation and quality metrics
struct AnnotatorProfile {
    std::string annotator_id;
    std::string public_key;
    float reputation_score;     // 0.0 to 1.0
    uint32_t total_annotations;
    uint32_t consensus_agreements;
    float average_confidence;
    std::vector<std::string> specializations;  // Domain expertise
    uint64_t last_active;
    bool is_verified;
};

// Quality assurance metrics
struct QualityMetrics {
    float inter_annotator_agreement;
    float consensus_strength;
    float label_consistency;
    float time_efficiency;
    std::map<std::string, float> domain_specific_metrics;
};

// Configuration for the curation system
struct CurationConfig {
    uint32_t min_annotators_per_task = 3;
    uint32_t max_annotators_per_task = 7;
    float consensus_threshold = 0.67f;          // BFT threshold
    float reputation_decay_rate = 0.01f;        // Daily decay
    float min_reputation_for_tasks = 0.3f;
    uint32_t max_concurrent_tasks_per_annotator = 10;
    uint32_t task_timeout_hours = 24;
    bool enable_quality_bonuses = true;
    float base_annotation_reward = 0.1f;        // Base reward per annotation
};

// Main distributed curation platform
class DistributedCurationPlatform {
public:
    DistributedCurationPlatform(std::shared_ptr<p2p::P2PNetwork> network,
                               const CurationConfig& config = CurationConfig{});
    ~DistributedCurationPlatform();

    // Platform lifecycle
    bool start();
    void stop();
    bool is_running() const { return running_.load(); }

    // Task management
    std::string submit_annotation_task(const AnnotationTask& task);
    bool cancel_task(const std::string& task_id);
    std::vector<AnnotationTask> get_available_tasks(const std::string& annotator_id);
    
    // Annotation submission and retrieval
    bool submit_annotation(const AnnotationSubmission& submission);
    std::optional<AnnotationConsensus> get_task_consensus(const std::string& task_id);
    std::vector<AnnotationConsensus> get_completed_annotations(uint32_t limit = 100);

    // Annotator management
    bool register_annotator(const AnnotatorProfile& profile);
    bool update_annotator_profile(const AnnotatorProfile& profile);
    std::optional<AnnotatorProfile> get_annotator_profile(const std::string& annotator_id);
    std::vector<AnnotatorProfile> get_top_annotators(uint32_t limit = 50);

    // Quality assurance
    QualityMetrics compute_quality_metrics(const std::string& task_id);
    void update_annotator_reputation(const std::string& annotator_id, float delta);
    std::vector<std::string> detect_low_quality_annotators();

    // Statistics and monitoring
    struct PlatformStats {
        uint32_t total_tasks;
        uint32_t completed_tasks;
        uint32_t active_annotators;
        float average_consensus_time_hours;
        float platform_quality_score;
        std::map<std::string, uint32_t> tasks_by_type;
    };
    PlatformStats get_platform_stats();

    // Event callbacks
    using TaskCompletedCallback = std::function<void(const AnnotationConsensus&)>;
    using AnnotatorJoinedCallback = std::function<void(const AnnotatorProfile&)>;
    using QualityAlertCallback = std::function<void(const std::string&, const QualityMetrics&)>;

    void set_task_completed_callback(TaskCompletedCallback callback);
    void set_annotator_joined_callback(AnnotatorJoinedCallback callback);
    void set_quality_alert_callback(QualityAlertCallback callback);

private:
    std::shared_ptr<p2p::P2PNetwork> network_;
    CurationConfig config_;
    std::atomic<bool> running_{false};

    // Internal data structures
    std::map<std::string, AnnotationTask> active_tasks_;
    std::map<std::string, std::vector<AnnotationSubmission>> task_submissions_;
    std::map<std::string, AnnotationConsensus> completed_tasks_;
    std::map<std::string, AnnotatorProfile> annotator_profiles_;

    // Thread safety
    mutable std::mutex tasks_mutex_;
    mutable std::mutex annotators_mutex_;
    mutable std::mutex consensus_mutex_;

    // Background processing
    std::vector<std::thread> worker_threads_;
    void consensus_processing_thread();
    void reputation_update_thread();
    void quality_monitoring_thread();

    // P2P message handlers
    void handle_task_submission(const p2p::NetworkMessage& message);
    void handle_annotation_submission(const p2p::NetworkMessage& message);
    void handle_consensus_proposal(const p2p::NetworkMessage& message);
    void handle_reputation_update(const p2p::NetworkMessage& message);

    // Consensus mechanisms
    bool compute_annotation_consensus(const std::string& task_id);
    float compute_label_consensus(const std::vector<AnnotationLabel>& labels);
    void broadcast_consensus_result(const AnnotationConsensus& consensus);

    // Quality assurance
    void validate_annotation_quality(const AnnotationSubmission& submission);
    float compute_annotator_agreement(const std::string& annotator_id, 
                                    const AnnotationConsensus& consensus);
    void detect_and_handle_spam(const std::string& annotator_id);

    // Callbacks
    TaskCompletedCallback task_completed_callback_;
    AnnotatorJoinedCallback annotator_joined_callback_;
    QualityAlertCallback quality_alert_callback_;
};

// Utility functions for data curation
namespace utils {
    // Serialization helpers
    std::vector<uint8_t> serialize_annotation_task(const AnnotationTask& task);
    AnnotationTask deserialize_annotation_task(const std::vector<uint8_t>& data);
    
    std::vector<uint8_t> serialize_annotation_submission(const AnnotationSubmission& submission);
    AnnotationSubmission deserialize_annotation_submission(const std::vector<uint8_t>& data);

    // Cryptographic helpers
    std::string sign_annotation(const AnnotationSubmission& submission, 
                               const std::string& private_key);
    bool verify_annotation_signature(const AnnotationSubmission& submission,
                                   const std::string& public_key);

    // Quality metrics
    float compute_inter_annotator_agreement(const std::vector<AnnotationSubmission>& submissions);
    float compute_consensus_strength(const std::vector<AnnotationLabel>& labels);
    
    // Task distribution
    std::vector<std::string> select_annotators_for_task(const AnnotationTask& task,
                                                       const std::vector<AnnotatorProfile>& annotators,
                                                       uint32_t count);
}

} // namespace curation
