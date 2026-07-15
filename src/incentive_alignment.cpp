#include "../include/incentive_alignment.hpp"
#include "../include/logger.hpp"
#include <algorithm>
#include <random>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iomanip>

namespace incentive {

IncentiveAlignmentSystem::IncentiveAlignmentSystem(const IncentiveConfig& config)
    : config_(config), request_queue_(RequestPriorityComparator(this)) {
    
    logger::log_info("Initializing Incentive Alignment System");
    logger::log_info("- CPU credit rate: " + std::to_string(config_.cpu_hour_credit_rate) + " credits/hour");
    logger::log_info("- GPU credit rate: " + std::to_string(config_.gpu_hour_credit_rate) + " credits/hour");
    logger::log_info("- Academic reserved share: " + std::to_string(config_.academic_reserved_share * 100) + "%");
    logger::log_info("- Contributor reserved share: " + std::to_string(config_.contributor_reserved_share * 100) + "%");
    
    initialize_priority_tiers();
    
    // Initialize system statistics
    system_stats_ = {};
    
    // Start maintenance thread
    maintenance_running_ = true;
    maintenance_thread_ = std::thread(&IncentiveAlignmentSystem::run_maintenance_loop, this);
}

IncentiveAlignmentSystem::~IncentiveAlignmentSystem() {
    maintenance_running_ = false;
    if (maintenance_thread_.joinable()) {
        maintenance_thread_.join();
    }
}

void IncentiveAlignmentSystem::initialize_priority_tiers() {
    std::lock_guard<std::mutex> lock(tiers_mutex_);
    
    priority_tiers_.clear();
    
    // Diamond tier: Major contributors (1000+ hours contributed)
    PriorityTier diamond;
    diamond.tier_name = "Diamond";
    diamond.min_contribution_ratio = 10.0f;  // Contributed 10x more than used
    diamond.base_priority_score = 1000;
    diamond.resource_allocation_share = 0.25f;  // 25% of resources
    diamond.max_concurrent_requests = 10;
    diamond.preemption_protection = 0.95f;
    priority_tiers_.push_back(diamond);
    
    // Platinum tier: Significant contributors (500+ hours)
    PriorityTier platinum;
    platinum.tier_name = "Platinum";
    platinum.min_contribution_ratio = 5.0f;
    platinum.base_priority_score = 800;
    platinum.resource_allocation_share = 0.20f;
    platinum.max_concurrent_requests = 8;
    platinum.preemption_protection = 0.85f;
    priority_tiers_.push_back(platinum);
    
    // Gold tier: Regular contributors (100+ hours)
    PriorityTier gold;
    gold.tier_name = "Gold";
    gold.min_contribution_ratio = 2.0f;
    gold.base_priority_score = 600;
    gold.resource_allocation_share = 0.20f;
    gold.max_concurrent_requests = 6;
    gold.preemption_protection = 0.70f;
    priority_tiers_.push_back(gold);
    
    // Silver tier: Modest contributors (20+ hours)
    PriorityTier silver;
    silver.tier_name = "Silver";
    silver.min_contribution_ratio = 1.0f;
    silver.base_priority_score = 400;
    silver.resource_allocation_share = 0.15f;
    silver.max_concurrent_requests = 4;
    silver.preemption_protection = 0.50f;
    priority_tiers_.push_back(silver);
    
    // Bronze tier: New/minimal contributors
    PriorityTier bronze;
    bronze.tier_name = "Bronze";
    bronze.min_contribution_ratio = 0.1f;
    bronze.base_priority_score = 200;
    bronze.resource_allocation_share = 0.20f;
    bronze.max_concurrent_requests = 2;
    bronze.preemption_protection = 0.20f;
    priority_tiers_.push_back(bronze);
    
    logger::log_info("Initialized " + std::to_string(priority_tiers_.size()) + " priority tiers");
}

bool IncentiveAlignmentSystem::register_contributor(const std::string& contributor_id, 
                                                   const std::string& organization_type,
                                                   const std::string& contact_info) {
    std::lock_guard<std::mutex> lock(contributors_mutex_);
    
    if (contributors_.find(contributor_id) != contributors_.end()) {
        logger::log_warning("Contributor " + contributor_id + " already registered");
        return false;
    }
    
    ComputeContribution contribution;
    contribution.contributor_id = contributor_id;
    contribution.organization_type = organization_type;
    contribution.compute_hours_contributed = 0.0f;
    contribution.gpu_hours_contributed = 0.0f;
    contribution.operations_completed = 0;
    contribution.uptime_percentage = 0.0f;
    contribution.contribution_start_date = static_cast<uint32_t>(
        std::chrono::system_clock::now().time_since_epoch().count());
    contribution.last_contribution = std::chrono::steady_clock::now();
    
    contributors_[contributor_id] = contribution;
    contributor_credits_[contributor_id] = 0.0f;
    
    // Initialize statistics
    ContributorStats stats;
    stats.contributor_id = contributor_id;
    stats.total_credits_earned = 0.0f;
    stats.total_credits_spent = 0.0f;
    stats.current_credit_balance = 0.0f;
    stats.contribution_ratio = 0.0f;
    stats.current_tier = "Bronze";
    stats.jobs_completed = 0;
    stats.jobs_preempted = 0;
    stats.average_job_completion_time_hours = 0.0f;
    stats.member_since = std::chrono::steady_clock::now();
    
    contributor_stats_[contributor_id] = stats;
    
    {
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        system_stats_.total_registered_contributors++;
    }
    
    logger::log_info("Registered new contributor: " + contributor_id + " (" + organization_type + ")");
    
    // Give new contributors some initial credits as a welcome bonus
    float welcome_bonus = (organization_type == "university" || organization_type == "research_lab") ? 
                         config_.cpu_hour_credit_rate * 10.0f :  // 10 hours worth of credits
                         config_.cpu_hour_credit_rate * 5.0f;    // 5 hours worth of credits
    
    contributor_credits_[contributor_id] = welcome_bonus;
    contributor_stats_[contributor_id].current_credit_balance = welcome_bonus;
    
    return true;
}

bool IncentiveAlignmentSystem::update_contribution(const std::string& contributor_id, 
                                                  float compute_hours, 
                                                  float gpu_hours,
                                                  uint64_t operations) {
    std::lock_guard<std::mutex> lock(contributors_mutex_);
    
    auto it = contributors_.find(contributor_id);
    if (it == contributors_.end()) {
        logger::log_error("Unknown contributor: " + contributor_id);
        return false;
    }
    
    // Validate contribution claim (basic checks)
    if (compute_hours < 0 || gpu_hours < 0 || compute_hours > 24 * 7) { // Max 1 week of compute
        logger::log_warning("Invalid contribution claim from " + contributor_id);
        return false;
    }
    
    // Update contribution data
    it->second.compute_hours_contributed += compute_hours;
    it->second.gpu_hours_contributed += gpu_hours;
    it->second.operations_completed += operations;
    it->second.last_contribution = std::chrono::steady_clock::now();
    
    // Calculate and award credits
    float credits_earned = calculate_credits_earned(compute_hours, gpu_hours, is_peak_time());
    
    // Apply organization bonus if applicable
    float org_bonus = get_organization_contribution_bonus(contributor_id);
    credits_earned *= (1.0f + org_bonus);
    
    contributor_credits_[contributor_id] += credits_earned;
    
    // Update statistics
    auto& stats = contributor_stats_[contributor_id];
    stats.total_credits_earned += credits_earned;
    stats.current_credit_balance = contributor_credits_[contributor_id];
    stats.contribution_ratio = calculate_contribution_ratio(contributor_id);
    stats.current_tier = determine_priority_tier(contributor_id);
    
    {
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        system_stats_.total_compute_hours_contributed += compute_hours;
        system_stats_.total_gpu_hours_contributed += gpu_hours;
    }
    
    logger::log_info("Updated contribution for " + contributor_id + ": +" + 
                     std::to_string(compute_hours) + " CPU hours, +" + 
                     std::to_string(gpu_hours) + " GPU hours, +" +
                     std::to_string(credits_earned) + " credits");
    
    return true;
}

float IncentiveAlignmentSystem::calculate_credits_earned(float compute_hours, float gpu_hours, bool is_peak_time) {
    float multiplier = is_peak_time ? config_.peak_hours_multiplier : config_.off_peak_multiplier;
    
    float cpu_credits = compute_hours * config_.cpu_hour_credit_rate;
    float gpu_credits = gpu_hours * config_.gpu_hour_credit_rate;
    
    return (cpu_credits + gpu_credits) * multiplier;
}

float IncentiveAlignmentSystem::get_contributor_credits(const std::string& contributor_id) const {
    std::lock_guard<std::mutex> lock(contributors_mutex_);
    
    auto it = contributor_credits_.find(contributor_id);
    if (it != contributor_credits_.end()) {
        return it->second;
    }
    
    return 0.0f;
}

bool IncentiveAlignmentSystem::spend_credits(const std::string& contributor_id, float credits) {
    std::lock_guard<std::mutex> lock(contributors_mutex_);
    
    auto it = contributor_credits_.find(contributor_id);
    if (it == contributor_credits_.end() || it->second < credits) {
        return false;
    }
    
    it->second -= credits;
    
    // Update statistics
    auto& stats = contributor_stats_[contributor_id];
    stats.total_credits_spent += credits;
    stats.current_credit_balance = it->second;
    stats.contribution_ratio = calculate_contribution_ratio(contributor_id);
    
    return true;
}

uint32_t IncentiveAlignmentSystem::calculate_priority_score(const ResourceRequest& request) const {
    std::lock_guard<std::mutex> lock(contributors_mutex_);
    
    uint32_t base_score = 100; // Minimum priority
    
    // Tier-based priority
    std::string tier = determine_priority_tier(request.requestor_id);
    std::lock_guard<std::mutex> tier_lock(tiers_mutex_);
    
    for (const auto& priority_tier : priority_tiers_) {
        if (priority_tier.tier_name == tier) {
            base_score = priority_tier.base_priority_score;
            break;
        }
    }
    
    // Academic boost
    if (request.is_academic) {
        base_score = static_cast<uint32_t>(base_score * config_.academic_priority_boost);
    }
    
    // Contribution ratio boost
    float contribution_ratio = calculate_contribution_ratio(request.requestor_id);
    uint32_t contribution_boost = static_cast<uint32_t>(contribution_ratio * config_.contribution_ratio_weight * 50);
    
    // Request priority level
    uint32_t request_boost = request.priority_level * 20;
    
    // Time-based boost (older requests get slight priority)
    auto now = std::chrono::steady_clock::now();
    auto wait_time = std::chrono::duration_cast<std::chrono::hours>(now - request.submission_time);
    uint32_t wait_boost = std::min(100u, static_cast<uint32_t>(wait_time.count() * 5));
    
    uint32_t final_score = base_score + contribution_boost + request_boost + wait_boost;
    
    return final_score;
}

std::string IncentiveAlignmentSystem::determine_priority_tier(const std::string& contributor_id) const {
    float contribution_ratio = calculate_contribution_ratio(contributor_id);
    
    std::lock_guard<std::mutex> lock(tiers_mutex_);
    
    // Find the highest tier the contributor qualifies for
    for (const auto& tier : priority_tiers_) {
        if (contribution_ratio >= tier.min_contribution_ratio) {
            return tier.tier_name;
        }
    }
    
    return "Bronze"; // Default tier
}

ResourceAllocationResult IncentiveAlignmentSystem::allocate_resources(
    const ResourceRequest& request, const std::vector<std::string>& available_nodes) {
    
    ResourceAllocationResult result;
    result.request_id = request.request_id;
    result.allocation_successful = false;
    
    // Check if requestor has sufficient credits
    float estimated_cost = request.estimated_compute_hours * config_.cpu_hour_credit_rate;
    float available_credits = get_contributor_credits(request.requestor_id);
    
    if (available_credits < estimated_cost) {
        result.allocation_message = "Insufficient credits. Required: " + std::to_string(estimated_cost) + 
                                   ", Available: " + std::to_string(available_credits);
        return result;
    }
    
    // Determine allocation tier and resource share
    std::string tier = determine_priority_tier(request.requestor_id);
    result.allocation_tier = tier;
    
    std::lock_guard<std::mutex> tier_lock(tiers_mutex_);
    float allocation_share = 0.1f; // Default fallback
    
    for (const auto& priority_tier : priority_tiers_) {
        if (priority_tier.tier_name == tier) {
            allocation_share = priority_tier.resource_allocation_share;
            break;
        }
    }
    
    // Calculate how many nodes this tier can use
    size_t max_nodes_for_tier = static_cast<size_t>(available_nodes.size() * allocation_share);
    size_t nodes_to_allocate = std::min(static_cast<size_t>(request.requested_nodes), max_nodes_for_tier);
    
    if (nodes_to_allocate == 0) {
        result.allocation_message = "No nodes available for tier " + tier;
        return result;
    }
    
    // Allocate best nodes for higher tiers
    std::vector<std::string> selected_nodes;
    if (tier == "Diamond" || tier == "Platinum") {
        // Give priority access to the best nodes
        for (size_t i = 0; i < nodes_to_allocate && i < available_nodes.size(); ++i) {
            selected_nodes.push_back(available_nodes[i]);
        }
    } else {
        // Random allocation for lower tiers
        std::vector<std::string> shuffled_nodes = available_nodes;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(shuffled_nodes.begin(), shuffled_nodes.end(), gen);
        
        for (size_t i = 0; i < nodes_to_allocate && i < shuffled_nodes.size(); ++i) {
            selected_nodes.push_back(shuffled_nodes[i]);
        }
    }
    
    if (!selected_nodes.empty()) {
        result.allocated_nodes = selected_nodes;
        result.estimated_cost_credits = estimated_cost;
        result.estimated_wait_time_hours = 0; // Immediate allocation
        result.allocation_successful = true;
        result.allocation_message = "Allocated " + std::to_string(selected_nodes.size()) + 
                                   " nodes for " + std::to_string(estimated_cost) + " credits";
        
        // Reserve the credits (will be spent when job starts)
        // In a real implementation, you'd have a more sophisticated reservation system
        
        logger::log_info("Resource allocation successful for " + request.requestor_id + 
                        " (" + tier + " tier): " + std::to_string(selected_nodes.size()) + " nodes");
    } else {
        result.allocation_message = "Resource allocation failed: No suitable nodes available";
    }
    
    return result;
}

bool IncentiveAlignmentSystem::submit_resource_request(const ResourceRequest& request) {
    std::lock_guard<std::mutex> lock(requests_mutex_);
    
    // Check if requestor exists
    {
        std::lock_guard<std::mutex> contrib_lock(contributors_mutex_);
        if (contributors_.find(request.requestor_id) == contributors_.end()) {
            logger::log_error("Unknown requestor: " + request.requestor_id);
            return false;
        }
    }
    
    // Add to priority queue
    request_queue_.push(request);
    active_requests_[request.request_id] = request;
    
    {
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        system_stats_.pending_requests++;
    }
    
    logger::log_info("Submitted resource request " + request.request_id + " for " + request.requestor_id);
    
    return true;
}

std::vector<ResourceRequest> IncentiveAlignmentSystem::get_pending_requests_by_priority() const {
    std::lock_guard<std::mutex> lock(requests_mutex_);
    
    std::vector<ResourceRequest> requests;
    
    // Copy queue to vector (since priority_queue doesn't allow iteration)
    auto temp_queue = request_queue_;
    while (!temp_queue.empty()) {
        requests.push_back(temp_queue.top());
        temp_queue.pop();
    }
    
    return requests;
}

bool IncentiveAlignmentSystem::verify_academic_status(const std::string& contributor_id, 
                                                     const std::string& institution,
                                                     const std::string& verification_token) {
    std::lock_guard<std::mutex> lock(academic_mutex_);
    
    // In a real implementation, this would verify against a database of academic institutions
    // For now, we'll do basic validation
    
    std::vector<std::string> known_universities = {
        "mit.edu", "stanford.edu", "berkeley.edu", "cmu.edu", "caltech.edu",
        "harvard.edu", "princeton.edu", "yale.edu", "oxford.ac.uk", "cambridge.ac.uk",
        "ethz.ch", "toronto.edu", "ubc.ca", "anu.edu.au", "unimelb.edu.au"
    };
    
    bool is_valid_institution = false;
    for (const auto& uni : known_universities) {
        if (institution.find(uni) != std::string::npos) {
            is_valid_institution = true;
            break;
        }
    }
    
    if (is_valid_institution && !verification_token.empty()) {
        verified_academic_status_[contributor_id] = true;
        academic_institutions_[contributor_id] = institution;
        
        logger::log_info("Verified academic status for " + contributor_id + " at " + institution);
        return true;
    }
    
    logger::log_warning("Failed to verify academic status for " + contributor_id);
    return false;
}

float IncentiveAlignmentSystem::get_academic_priority_boost(const std::string& contributor_id) const {
    std::lock_guard<std::mutex> lock(academic_mutex_);
    
    auto it = verified_academic_status_.find(contributor_id);
    if (it != verified_academic_status_.end() && it->second) {
        return config_.academic_priority_boost;
    }
    
    return 1.0f;
}

ContributorStats IncentiveAlignmentSystem::get_contributor_stats(const std::string& contributor_id) const {
    std::lock_guard<std::mutex> lock(contributors_mutex_);
    
    auto it = contributor_stats_.find(contributor_id);
    if (it != contributor_stats_.end()) {
        return it->second;
    }
    
    return ContributorStats{}; // Return empty stats for unknown contributors
}

IncentiveAlignmentSystem::SystemStats IncentiveAlignmentSystem::get_system_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return system_stats_;
}

float IncentiveAlignmentSystem::calculate_contribution_ratio(const std::string& contributor_id) const {
    auto stats_it = contributor_stats_.find(contributor_id);
    if (stats_it == contributor_stats_.end()) {
        return 0.0f;
    }
    
    const auto& stats = stats_it->second;
    
    if (stats.total_credits_spent <= 0.0f) {
        // If they haven't spent any credits yet, their ratio is based on earned credits
        return stats.total_credits_earned / std::max(1.0f, config_.cpu_hour_credit_rate);
    }
    
    return stats.total_credits_earned / stats.total_credits_spent;
}

bool IncentiveAlignmentSystem::is_peak_time() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto tm = *std::localtime(&time_t);
    
    // Peak hours: 9 AM to 5 PM local time, Monday to Friday
    bool is_weekday = (tm.tm_wday >= 1 && tm.tm_wday <= 5);
    bool is_business_hours = (tm.tm_hour >= 9 && tm.tm_hour < 17);
    
    return is_weekday && is_business_hours;
}

void IncentiveAlignmentSystem::run_maintenance_loop() {
    logger::log_info("Started incentive system maintenance loop");
    
    while (maintenance_running_.load()) {
        try {
            // Apply credit decay monthly
            static auto last_decay = std::chrono::steady_clock::now();
            auto now = std::chrono::steady_clock::now();
            
            if (std::chrono::duration_cast<std::chrono::hours>(now - last_decay).count() >= 24 * 30) {
                apply_credit_decay();
                last_decay = now;
            }
            
            // Update system statistics
            update_system_statistics();
            
            // Clean up expired requests
            // TODO: Implement request cleanup
            
        } catch (const std::exception& e) {
            logger::log_error("Error in incentive maintenance loop: " + std::string(e.what()));
        }
        
        // Sleep for 1 hour
        std::this_thread::sleep_for(std::chrono::hours(1));
    }
    
    logger::log_info("Incentive system maintenance loop stopped");
}

void IncentiveAlignmentSystem::apply_credit_decay() {
    std::lock_guard<std::mutex> lock(contributors_mutex_);
    
    logger::log_info("Applying monthly credit decay (rate: " + std::to_string(config_.credit_decay_rate) + ")");
    
    for (auto& [contributor_id, credits] : contributor_credits_) {
        float old_credits = credits;
        credits *= config_.credit_decay_rate;
        
        // Update stats
        auto& stats = contributor_stats_[contributor_id];
        stats.current_credit_balance = credits;
        
        logger::log_debug("Credit decay for " + contributor_id + ": " + 
                         std::to_string(old_credits) + " -> " + std::to_string(credits));
    }
}

void IncentiveAlignmentSystem::update_system_statistics() {
    std::lock_guard<std::mutex> stats_lock(stats_mutex_);
    std::lock_guard<std::mutex> contrib_lock(contributors_mutex_);
    std::lock_guard<std::mutex> requests_lock(requests_mutex_);
    
    // Count active contributors (contributed in last 30 days)
    auto now = std::chrono::steady_clock::now();
    auto cutoff = now - std::chrono::hours(24 * 30);
    
    uint32_t active_contributors = 0;
    float academic_usage = 0.0f;
    
    for (const auto& [contributor_id, contribution] : contributors_) {
        if (contribution.last_contribution > cutoff) {
            active_contributors++;
        }
        
        // Calculate academic usage
        auto stats_it = contributor_stats_.find(contributor_id);
        if (stats_it != contributor_stats_.end()) {
            auto academic_it = verified_academic_status_.find(contributor_id);
            if (academic_it != verified_academic_status_.end() && academic_it->second) {
                academic_usage += stats_it->second.total_credits_spent;
            }
        }
    }
    
    system_stats_.active_contributors_last_30_days = active_contributors;
    
    // Calculate academic usage percentage
    float total_usage = 0.0f;
    for (const auto& [contributor_id, stats] : contributor_stats_) {
        total_usage += stats.total_credits_spent;
    }
    
    if (total_usage > 0) {
        system_stats_.academic_usage_percentage = academic_usage / total_usage;
    }
    
    system_stats_.pending_requests = static_cast<uint32_t>(active_requests_.size());
    
    // TODO: Calculate average wait time and system utilization
}

// Factory implementation
std::unique_ptr<IncentiveAlignmentSystem> 
IncentiveSystemFactory::create_system(PolicyType policy, const IncentiveConfig& base_config) {
    
    IncentiveConfig config = base_config;
    
    switch (policy) {
        case PolicyType::ACADEMIC_PRIORITY:
            config = create_academic_priority_config();
            break;
        case PolicyType::CONTRIBUTOR_FIRST:
            config = create_contributor_first_config();
            break;
        case PolicyType::FAIR_SHARE:
            config = create_fair_share_config();
            break;
        case PolicyType::COMMERCIAL_HYBRID:
            // Use base config as-is for commercial hybrid
            break;
    }
    
    return std::make_unique<IncentiveAlignmentSystem>(config);
}

IncentiveConfig IncentiveSystemFactory::create_academic_priority_config() {
    IncentiveConfig config;
    config.academic_priority_boost = 2.0f;     // Double priority for academic users
    config.academic_reserved_share = 0.6f;     // 60% reserved for academic use
    config.contributor_reserved_share = 0.2f;  // 20% for contributors
    config.general_pool_share = 0.2f;         // 20% general
    config.new_contributor_grace_days = 60;    // Longer grace period
    return config;
}

IncentiveConfig IncentiveSystemFactory::create_contributor_first_config() {
    IncentiveConfig config;
    config.contribution_ratio_weight = 3.0f;   // Heavy weight on contribution
    config.academic_reserved_share = 0.2f;     // Only 20% for academic
    config.contributor_reserved_share = 0.6f;  // 60% for contributors
    config.general_pool_share = 0.2f;
    config.preemption_threshold_ratio = 0.5f;  // Higher threshold for preemption
    return config;
}

IncentiveConfig IncentiveSystemFactory::create_fair_share_config() {
    IncentiveConfig config;
    config.academic_priority_boost = 1.0f;     // No academic boost
    config.contribution_ratio_weight = 1.0f;   // Minimal contribution weight
    config.academic_reserved_share = 0.33f;    // Equal shares
    config.contributor_reserved_share = 0.33f;
    config.general_pool_share = 0.34f;
    config.allow_preemption = false;          // No preemption in fair share
    return config;
}

} // namespace incentive
