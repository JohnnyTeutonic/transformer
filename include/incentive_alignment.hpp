#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <mutex>
#include <chrono>
#include <memory>
#include <queue>
#include <atomic>

namespace incentive {

struct ComputeContribution {
    std::string contributor_id;
    std::string organization_type;    // "university", "research_lab", "individual", "commercial"
    float compute_hours_contributed;  // Total compute hours donated
    float gpu_hours_contributed;     // GPU hours specifically
    uint64_t operations_completed;   // Number of training operations completed
    float uptime_percentage;         // Historical uptime
    uint32_t contribution_start_date; // When they started contributing
    std::chrono::steady_clock::time_point last_contribution;
};

struct ResourceRequest {
    std::string requestor_id;
    std::string request_id;
    std::string request_type;        // "training", "inference", "data_processing"
    float estimated_compute_hours;   // Estimated resource usage
    uint32_t priority_level;         // 1-10, higher = more priority
    std::string project_description; // What this request is for
    uint32_t requested_nodes;        // How many nodes requested
    uint32_t max_wait_time_hours;    // Maximum acceptable wait time
    std::chrono::steady_clock::time_point submission_time;
    bool is_academic;                // Academic research gets priority
};

struct PriorityTier {
    std::string tier_name;
    float min_contribution_ratio;    // Minimum contribution/usage ratio for this tier
    uint32_t base_priority_score;    // Base priority for this tier
    float resource_allocation_share; // Share of total resources (0.0-1.0)
    uint32_t max_concurrent_requests; // Max requests this tier can have running
    float preemption_protection;     // Protection against being preempted (0.0-1.0)
};

struct IncentiveConfig {
    // BOINC-style credit system
    float cpu_hour_credit_rate = 100.0f;        // Credits per CPU hour
    float gpu_hour_credit_rate = 1000.0f;       // Credits per GPU hour
    float credit_decay_rate = 0.99f;            // Monthly credit decay to encourage ongoing contribution
    
    // Priority system
    float academic_priority_boost = 1.5f;       // Boost for academic institutions
    float contribution_ratio_weight = 2.0f;     // Weight of contribution ratio in priority calculation
    float reputation_weight = 1.0f;             // Weight of reputation in priority calculation
    
    // Resource allocation
    float academic_reserved_share = 0.4f;       // 40% reserved for academic use
    float contributor_reserved_share = 0.3f;    // 30% reserved for active contributors
    float general_pool_share = 0.3f;           // 30% general availability
    
    // Time-based incentives
    uint32_t new_contributor_grace_days = 30;   // Grace period for new contributors
    float peak_hours_multiplier = 1.2f;        // Extra credits during peak usage
    float off_peak_multiplier = 0.8f;          // Reduced credits during off-peak
    
    // Preemption policies
    bool allow_preemption = true;
    float preemption_notice_hours = 1.0f;      // Notice before preempting a job
    float preemption_threshold_ratio = 0.1f;   // Contribution ratio below which preemption is allowed
};

struct ContributorStats {
    std::string contributor_id;
    float total_credits_earned;
    float total_credits_spent;
    float current_credit_balance;
    float contribution_ratio;        // credits_earned / credits_spent
    std::string current_tier;
    uint32_t jobs_completed;
    uint32_t jobs_preempted;
    float average_job_completion_time_hours;
    std::chrono::steady_clock::time_point member_since;
};

struct ResourceAllocationResult {
    std::string request_id;
    std::vector<std::string> allocated_nodes;
    uint32_t estimated_wait_time_hours;
    float estimated_cost_credits;
    std::string allocation_tier;
    bool allocation_successful;
    std::string allocation_message;
};

class IncentiveAlignmentSystem {
public:
    explicit IncentiveAlignmentSystem(const IncentiveConfig& config = {});
    ~IncentiveAlignmentSystem();
    
    // Contributor management
    bool register_contributor(const std::string& contributor_id, 
                             const std::string& organization_type,
                             const std::string& contact_info = "");
    bool update_contribution(const std::string& contributor_id, 
                           float compute_hours, 
                           float gpu_hours,
                           uint64_t operations);
    
    // Credit system (BOINC-style)
    float calculate_credits_earned(float compute_hours, float gpu_hours, bool is_peak_time = false);
    float get_contributor_credits(const std::string& contributor_id) const;
    bool spend_credits(const std::string& contributor_id, float credits);
    void apply_credit_decay();  // Called periodically to decay old credits
    
    // Priority system
    uint32_t calculate_priority_score(const ResourceRequest& request) const;
    std::string determine_priority_tier(const std::string& contributor_id) const;
    std::vector<PriorityTier> get_priority_tiers() const;
    
    // Resource allocation
    ResourceAllocationResult allocate_resources(const ResourceRequest& request,
                                              const std::vector<std::string>& available_nodes);
    bool can_preempt_job(const std::string& victim_id, const std::string& requester_id) const;
    std::vector<std::string> find_preemptible_jobs(const ResourceRequest& request) const;
    
    // Request queue management
    bool submit_resource_request(const ResourceRequest& request);
    std::vector<ResourceRequest> get_pending_requests_by_priority() const;
    bool cancel_resource_request(const std::string& request_id, const std::string& requestor_id);
    
    // Academic priority system
    bool verify_academic_status(const std::string& contributor_id, 
                               const std::string& institution,
                               const std::string& verification_token);
    float get_academic_priority_boost(const std::string& contributor_id) const;
    
    // Statistics and monitoring
    ContributorStats get_contributor_stats(const std::string& contributor_id) const;
    std::vector<ContributorStats> get_top_contributors(size_t count = 10) const;
    
    struct SystemStats {
        uint32_t total_registered_contributors;
        uint32_t active_contributors_last_30_days;
        float total_compute_hours_contributed;
        float total_gpu_hours_contributed;
        uint32_t pending_requests;
        float average_wait_time_hours;
        float academic_usage_percentage;
        float system_utilization;
    };
    
    SystemStats get_system_stats() const;
    
    // Fairness and anti-abuse
    bool detect_contribution_abuse(const std::string& contributor_id) const;
    bool validate_contribution_claim(const std::string& contributor_id, 
                                   float claimed_hours,
                                   const std::string& verification_proof);
    
    // Configuration management
    void update_config(const IncentiveConfig& new_config);
    IncentiveConfig get_config() const;
    
    // University/Research Lab Integration
    struct OrganizationRegistration {
        std::string organization_id;
        std::string organization_name;
        std::string organization_type;  // "university", "research_lab", "ngo"
        std::string verification_status; // "pending", "verified", "rejected"
        std::vector<std::string> authorized_users;
        float organization_contribution_bonus; // Extra credits for org members
        std::string contact_email;
        std::string verification_documents_hash;
    };
    
    bool register_organization(const OrganizationRegistration& org);
    bool add_user_to_organization(const std::string& org_id, const std::string& user_id);
    float get_organization_contribution_bonus(const std::string& contributor_id) const;
    
    // Export/Import for persistence
    bool export_contributor_data(const std::string& file_path) const;
    bool import_contributor_data(const std::string& file_path);
    
private:
    IncentiveConfig config_;
    mutable std::mutex config_mutex_;
    
    // Contributor data
    mutable std::mutex contributors_mutex_;
    std::unordered_map<std::string, ComputeContribution> contributors_;
    std::unordered_map<std::string, float> contributor_credits_;
    std::unordered_map<std::string, ContributorStats> contributor_stats_;
    
    // Priority tiers
    std::vector<PriorityTier> priority_tiers_;
    mutable std::mutex tiers_mutex_;
    
    // Request queue
    mutable std::mutex requests_mutex_;
    std::priority_queue<ResourceRequest, std::vector<ResourceRequest>, 
                       std::function<bool(const ResourceRequest&, const ResourceRequest&)>> request_queue_;
    std::unordered_map<std::string, ResourceRequest> active_requests_;
    
    // Academic verification
    mutable std::mutex academic_mutex_;
    std::unordered_map<std::string, bool> verified_academic_status_;
    std::unordered_map<std::string, std::string> academic_institutions_;
    
    // Organization management
    mutable std::mutex org_mutex_;
    std::unordered_map<std::string, OrganizationRegistration> organizations_;
    std::unordered_map<std::string, std::string> user_to_organization_;
    
    // System statistics
    mutable std::mutex stats_mutex_;
    SystemStats system_stats_;
    
    // Anti-abuse tracking
    struct AbuseTracker {
        uint32_t suspicious_contribution_patterns;
        std::chrono::steady_clock::time_point last_verification;
        std::vector<std::string> flagged_activities;
    };
    std::unordered_map<std::string, AbuseTracker> abuse_tracking_;
    mutable std::mutex abuse_mutex_;
    
    // Helper methods
    void initialize_priority_tiers();
    float calculate_contribution_ratio(const std::string& contributor_id) const;
    bool is_peak_time() const;
    void update_system_statistics();
    std::string generate_request_id();
    
    // Priority queue comparator
    struct RequestPriorityComparator {
        const IncentiveAlignmentSystem* system;
        
        RequestPriorityComparator(const IncentiveAlignmentSystem* sys) : system(sys) {}
        
        bool operator()(const ResourceRequest& a, const ResourceRequest& b) const {
            uint32_t priority_a = system->calculate_priority_score(a);
            uint32_t priority_b = system->calculate_priority_score(b);
            
            if (priority_a == priority_b) {
                // Secondary sort by submission time (older first)
                return a.submission_time > b.submission_time;
            }
            
            return priority_a < priority_b; // Higher priority comes first
        }
    };
    
    // Background maintenance
    std::atomic<bool> maintenance_running_{false};
    std::thread maintenance_thread_;
    void run_maintenance_loop();
};

// Factory for creating incentive systems with different policies
class IncentiveSystemFactory {
public:
    enum class PolicyType {
        ACADEMIC_PRIORITY,      // Prioritizes academic research
        CONTRIBUTOR_FIRST,      // Prioritizes heavy contributors
        FAIR_SHARE,            // Equal access for all
        COMMERCIAL_HYBRID      // Mixed academic/commercial model
    };
    
    static std::unique_ptr<IncentiveAlignmentSystem> 
    create_system(PolicyType policy, const IncentiveConfig& base_config = {});
    
    static IncentiveConfig create_academic_priority_config();
    static IncentiveConfig create_contributor_first_config();
    static IncentiveConfig create_fair_share_config();
};

} // namespace incentive
