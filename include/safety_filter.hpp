#pragma once

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <torch/torch.h>
#include <regex>
#include "distributed_transformer.hpp"

namespace safety {

// Safety violation types
enum class ViolationType {
    NONE = 0,
    TOXICITY = 1 << 0,
    BIAS = 1 << 1,
    VIOLENCE = 1 << 2,
    SEXUAL_CONTENT = 1 << 3,
    HATE_SPEECH = 1 << 4,
    SELF_HARM = 1 << 5,
    ILLEGAL_ACTIVITY = 1 << 6,
    PII_EXPOSURE = 1 << 7,
    MISINFORMATION = 1 << 8,
    MANIPULATION = 1 << 9,
    COPYRIGHT = 1 << 10,
    CUSTOM = 1 << 11
};

// Safety check result
struct SafetyCheckResult {
    bool is_safe;
    float safety_score;  // 0.0 (unsafe) to 1.0 (safe)
    uint32_t violation_flags;  // Bitwise OR of ViolationType
    std::vector<std::string> violation_details;
    std::unordered_map<std::string, float> category_scores;
    std::string remediation_suggestion;
    torch::Tensor attention_weights;  // For interpretability
};

// Safety configuration
struct SafetyConfig {
    // Thresholds
    float toxicity_threshold = 0.7f;
    float bias_threshold = 0.6f;
    float violence_threshold = 0.8f;
    float sexual_content_threshold = 0.9f;
    float hate_speech_threshold = 0.85f;
    float self_harm_threshold = 0.9f;
    float illegal_activity_threshold = 0.9f;
    float pii_threshold = 0.5f;
    float misinformation_threshold = 0.7f;
    float manipulation_threshold = 0.75f;
    float copyright_threshold = 0.8f;
    
    // Behavior
    bool block_unsafe_content = true;
    bool log_violations = true;
    bool enable_remediation = true;
    bool use_ensemble_filtering = false;
    
    // Performance
    int max_sequence_length = 512;
    int batch_size = 32;
    bool use_gpu = true;
};

// Base safety filter
class SafetyFilter {
public:
    SafetyFilter(SafetyConfig config);
    virtual ~SafetyFilter() = default;
    
    // Check content safety
    virtual SafetyCheckResult check(const std::string& text) = 0;
    virtual SafetyCheckResult check_batch(const std::vector<std::string>& texts);
    
    // Configuration
    void set_threshold(ViolationType type, float threshold);
    float get_threshold(ViolationType type) const;
    
    // Statistics
    struct FilterStatistics {
        uint64_t total_checks;
        uint64_t violations_detected;
        std::unordered_map<ViolationType, uint64_t> violation_counts;
        float average_safety_score;
        float false_positive_rate;
        float false_negative_rate;
    };
    FilterStatistics get_statistics() const { return stats_; }
    
protected:
    SafetyConfig config_;
    FilterStatistics stats_;
    
    // Helper functions
    uint32_t combine_violations(const std::vector<ViolationType>& violations);
    std::vector<ViolationType> extract_violations(uint32_t flags);
};

// Neural safety filter using transformer models
class NeuralSafetyFilter : public SafetyFilter {
public:
    NeuralSafetyFilter(SafetyConfig config,
                      std::shared_ptr<model::DistributedTransformer> model);
    
    SafetyCheckResult check(const std::string& text) override;
    
    // Fine-tuning
    void fine_tune(const std::vector<std::pair<std::string, SafetyCheckResult>>& labeled_data,
                  float learning_rate = 1e-5f,
                  int epochs = 3);
    
    // Model management
    void save_model(const std::string& path);
    void load_model(const std::string& path);
    
private:
    std::shared_ptr<model::DistributedTransformer> safety_model_;
    std::unique_ptr<torch::nn::Linear> classification_head_;
    
    // Tokenization
    torch::Tensor tokenize(const std::string& text);
    
    // Inference
    torch::Tensor forward(torch::Tensor input_ids);
    SafetyCheckResult interpret_output(torch::Tensor output, const std::string& text);
};

// Rule-based safety filter
class RuleBasedSafetyFilter : public SafetyFilter {
public:
    RuleBasedSafetyFilter(SafetyConfig config);
    
    SafetyCheckResult check(const std::string& text) override;
    
    // Rule management
    void add_rule(ViolationType type, const std::regex& pattern, float severity = 1.0f);
    void add_keyword_list(ViolationType type, const std::vector<std::string>& keywords);
    void load_rules_from_file(const std::string& path);
    
private:
    struct Rule {
        ViolationType type;
        std::regex pattern;
        float severity;
    };
    
    std::vector<Rule> rules_;
    std::unordered_map<ViolationType, std::vector<std::string>> keyword_lists_;
    
    // Pattern matching
    std::vector<std::pair<ViolationType, float>> match_patterns(const std::string& text);
    bool contains_keywords(const std::string& text, ViolationType type);
};

// Ensemble safety filter combining multiple approaches
class EnsembleSafetyFilter : public SafetyFilter {
public:
    EnsembleSafetyFilter(SafetyConfig config);
    
    // Add component filters
    void add_filter(std::shared_ptr<SafetyFilter> filter, float weight = 1.0f);
    
    SafetyCheckResult check(const std::string& text) override;
    
    // Voting strategies
    enum class VotingStrategy {
        UNANIMOUS,      // All filters must agree
        MAJORITY,       // Majority vote
        WEIGHTED,       // Weighted average
        CONSERVATIVE,   // Most restrictive
        ADAPTIVE        // Learn optimal weights
    };
    
    void set_voting_strategy(VotingStrategy strategy) { voting_strategy_ = strategy; }
    
private:
    std::vector<std::pair<std::shared_ptr<SafetyFilter>, float>> filters_;
    VotingStrategy voting_strategy_ = VotingStrategy::WEIGHTED;
    
    // Combine results
    SafetyCheckResult combine_results(const std::vector<std::pair<SafetyCheckResult, float>>& results);
    
    // Adaptive weight learning
    void update_weights(const SafetyCheckResult& ensemble_result,
                       const SafetyCheckResult& ground_truth);
};

// Context-aware safety filter
class ContextAwareSafetyFilter : public NeuralSafetyFilter {
public:
    ContextAwareSafetyFilter(SafetyConfig config,
                            std::shared_ptr<model::DistributedTransformer> model);
    
    // Check with context
    SafetyCheckResult check_with_context(const std::string& text,
                                        const std::vector<std::string>& context);
    
    // Conversation safety
    SafetyCheckResult check_conversation(const std::vector<std::string>& messages);
    
    // Intent detection
    struct IntentAnalysis {
        std::string primary_intent;
        std::vector<std::string> secondary_intents;
        float malicious_probability;
        std::unordered_map<std::string, float> intent_scores;
    };
    IntentAnalysis analyze_intent(const std::string& text);
    
private:
    // Context encoding
    torch::Tensor encode_context(const std::vector<std::string>& context);
    
    // Multi-turn analysis
    torch::Tensor analyze_conversation_flow(const std::vector<torch::Tensor>& message_encodings);
};

// Adversarial robustness for safety filters
class AdversarialSafetyFilter : public NeuralSafetyFilter {
public:
    AdversarialSafetyFilter(SafetyConfig config,
                           std::shared_ptr<model::DistributedTransformer> model);
    
    // Adversarial training
    void adversarial_training(const std::vector<std::string>& texts,
                             const std::vector<SafetyCheckResult>& labels,
                             float epsilon = 0.01f);
    
    // Generate adversarial examples
    std::vector<std::string> generate_adversarial_examples(const std::string& text);
    
    // Robustness testing
    struct RobustnessScore {
        float adversarial_accuracy;
        float perturbation_resistance;
        std::vector<std::string> successful_attacks;
        std::unordered_map<std::string, float> attack_type_success_rates;
    };
    RobustnessScore evaluate_robustness(const std::vector<std::string>& test_texts);
    
private:
    // Attack methods
    std::string character_substitution_attack(const std::string& text);
    std::string word_substitution_attack(const std::string& text);
    std::string insertion_attack(const std::string& text);
    std::string deletion_attack(const std::string& text);
    
    // Defense methods
    torch::Tensor adversarial_perturbation(torch::Tensor embeddings, float epsilon);
    torch::Tensor gradient_masking(torch::Tensor gradients);
};

// Real-time safety monitoring
class SafetyMonitor {
public:
    SafetyMonitor(std::shared_ptr<SafetyFilter> filter);
    
    // Stream monitoring
    void start_monitoring();
    void stop_monitoring();
    void process_stream(const std::string& text);
    
    // Alerts
    struct SafetyAlert {
        std::string alert_id;
        ViolationType violation_type;
        float severity;
        std::string content_sample;
        uint64_t timestamp;
        std::string source;
    };
    
    void register_alert_handler(std::function<void(const SafetyAlert&)> handler);
    std::vector<SafetyAlert> get_recent_alerts(int count = 100);
    
    // Metrics
    struct MonitoringMetrics {
        uint64_t total_content_processed;
        uint64_t violations_detected;
        float average_processing_time_ms;
        std::unordered_map<ViolationType, uint64_t> violation_distribution;
        std::vector<std::pair<uint64_t, float>> safety_score_timeline;
    };
    MonitoringMetrics get_metrics() const { return metrics_; }
    
private:
    std::shared_ptr<SafetyFilter> filter_;
    std::atomic<bool> monitoring_{false};
    
    // Alert system
    std::vector<std::function<void(const SafetyAlert&)>> alert_handlers_;
    std::deque<SafetyAlert> recent_alerts_;
    std::mutex alerts_mutex_;
    
    // Metrics
    MonitoringMetrics metrics_;
    std::mutex metrics_mutex_;
    
    // Processing
    std::thread monitoring_thread_;
    std::queue<std::string> processing_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    void monitoring_loop();
    void process_content(const std::string& text);
};

// Content remediation system
class ContentRemediator {
public:
    ContentRemediator();
    
    // Remediation strategies
    enum class RemediationStrategy {
        BLOCK,           // Block entire content
        REDACT,          // Redact problematic parts
        REPHRASE,        // Suggest alternative phrasing
        WARN,            // Add warning labels
        EDUCATE,         // Provide educational context
        FILTER_TOKENS    // Remove specific tokens
    };
    
    // Apply remediation
    struct RemediationResult {
        std::string remediated_content;
        RemediationStrategy strategy_used;
        std::vector<std::string> modifications;
        float content_preservation_ratio;
        std::string explanation;
    };
    
    RemediationResult remediate(const std::string& content,
                               const SafetyCheckResult& safety_result,
                               RemediationStrategy preferred_strategy = RemediationStrategy::REDACT);
    
    // Custom remediation rules
    void add_remediation_rule(ViolationType type,
                             RemediationStrategy strategy,
                             std::function<std::string(const std::string&)> transformer);
    
private:
    std::unordered_map<ViolationType, RemediationStrategy> default_strategies_;
    std::unordered_map<ViolationType, std::function<std::string(const std::string&)>> custom_transformers_;
    
    // Remediation methods
    std::string redact_content(const std::string& content, const SafetyCheckResult& result);
    std::string rephrase_content(const std::string& content, const SafetyCheckResult& result);
    std::string add_warnings(const std::string& content, const SafetyCheckResult& result);
    std::string filter_tokens(const std::string& content, const SafetyCheckResult& result);
};

// Safety filter factory
class SafetyFilterFactory {
public:
    // Create filters
    static std::unique_ptr<SafetyFilter> create_neural_filter(SafetyConfig config,
                                                             const std::string& model_path);
    static std::unique_ptr<SafetyFilter> create_rule_filter(SafetyConfig config,
                                                           const std::string& rules_path);
    static std::unique_ptr<SafetyFilter> create_ensemble_filter(SafetyConfig config,
                                                               const std::vector<std::string>& component_configs);
    
    // Presets
    static std::unique_ptr<SafetyFilter> create_strict_filter();
    static std::unique_ptr<SafetyFilter> create_moderate_filter();
    static std::unique_ptr<SafetyFilter> create_permissive_filter();
    static std::unique_ptr<SafetyFilter> create_child_safe_filter();
};

} // namespace safety
