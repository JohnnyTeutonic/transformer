#pragma once

#include <cstdint>

namespace p2p {

// Extended message types for distributed curation and RLHF
enum class MessageType : uint32_t {
    // Existing P2P message types
    NODE_DISCOVERY = 1,
    NODE_ANNOUNCEMENT = 2,
    NODE_LEAVE = 3,
    GRADIENT_PROPOSAL = 4,
    GRADIENT_VOTE = 5,
    GRADIENT_COMMIT = 6,
    HEARTBEAT = 7,
    PEER_LIST_REQUEST = 8,
    PEER_LIST_RESPONSE = 9,
    STATE_SYNC_REQUEST = 10,
    STATE_SYNC_RESPONSE = 11,
    
    // Distributed Curation message types
    CURATION_TASK_SUBMISSION = 100,
    CURATION_TASK_CANCELLATION = 101,
    CURATION_ANNOTATION_SUBMISSION = 102,
    CURATION_CONSENSUS_PROPOSAL = 103,
    CURATION_CONSENSUS_RESULT = 104,
    CURATION_REPUTATION_UPDATE = 105,
    CURATION_ANNOTATOR_REGISTRATION = 106,
    CURATION_QUALITY_ALERT = 107,
    
    // RLHF message types
    RLHF_REWARD_MODEL_GRADIENT = 200,
    RLHF_PPO_GRADIENT = 201,
    RLHF_PREFERENCE_DATA_SHARE = 202,
    RLHF_TRAINING_METRICS_UPDATE = 203,
    RLHF_MODEL_STATE_SYNC = 204,
    RLHF_CONSENSUS_PROPOSAL = 205,
    RLHF_CONSENSUS_VOTE = 206,
    RLHF_CONSENSUS_COMMIT = 207,
    
    // Integration and coordination
    SYSTEM_STATUS_UPDATE = 300,
    CAPABILITY_ANNOUNCEMENT = 301,
    RESOURCE_ALLOCATION_REQUEST = 302,
    RESOURCE_ALLOCATION_RESPONSE = 303,
    PERFORMANCE_METRICS_SHARE = 304,
    ERROR_REPORT = 305
};

// Message priority levels
enum class MessagePriority : uint8_t {
    LOW = 0,
    NORMAL = 1,
    HIGH = 2,
    CRITICAL = 3
};

// Message flags
enum class MessageFlags : uint32_t {
    NONE = 0,
    REQUIRES_ACK = 1 << 0,
    ENCRYPTED = 1 << 1,
    COMPRESSED = 1 << 2,
    BROADCAST = 1 << 3,
    URGENT = 1 << 4
};

// Helper functions
inline MessagePriority get_message_priority(MessageType type) {
    switch (type) {
        case MessageType::HEARTBEAT:
        case MessageType::NODE_DISCOVERY:
        case MessageType::NODE_ANNOUNCEMENT:
            return MessagePriority::HIGH;
            
        case MessageType::GRADIENT_PROPOSAL:
        case MessageType::GRADIENT_VOTE:
        case MessageType::GRADIENT_COMMIT:
        case MessageType::RLHF_CONSENSUS_PROPOSAL:
        case MessageType::RLHF_CONSENSUS_VOTE:
        case MessageType::RLHF_CONSENSUS_COMMIT:
            return MessagePriority::HIGH;
            
        case MessageType::ERROR_REPORT:
        case MessageType::CURATION_QUALITY_ALERT:
            return MessagePriority::CRITICAL;
            
        default:
            return MessagePriority::NORMAL;
    }
}

inline bool requires_consensus(MessageType type) {
    switch (type) {
        case MessageType::GRADIENT_PROPOSAL:
        case MessageType::CURATION_CONSENSUS_PROPOSAL:
        case MessageType::RLHF_CONSENSUS_PROPOSAL:
            return true;
        default:
            return false;
    }
}

inline const char* message_type_to_string(MessageType type) {
    switch (type) {
        case MessageType::NODE_DISCOVERY: return "NODE_DISCOVERY";
        case MessageType::NODE_ANNOUNCEMENT: return "NODE_ANNOUNCEMENT";
        case MessageType::NODE_LEAVE: return "NODE_LEAVE";
        case MessageType::GRADIENT_PROPOSAL: return "GRADIENT_PROPOSAL";
        case MessageType::GRADIENT_VOTE: return "GRADIENT_VOTE";
        case MessageType::GRADIENT_COMMIT: return "GRADIENT_COMMIT";
        case MessageType::HEARTBEAT: return "HEARTBEAT";
        case MessageType::PEER_LIST_REQUEST: return "PEER_LIST_REQUEST";
        case MessageType::PEER_LIST_RESPONSE: return "PEER_LIST_RESPONSE";
        case MessageType::STATE_SYNC_REQUEST: return "STATE_SYNC_REQUEST";
        case MessageType::STATE_SYNC_RESPONSE: return "STATE_SYNC_RESPONSE";
        
        case MessageType::CURATION_TASK_SUBMISSION: return "CURATION_TASK_SUBMISSION";
        case MessageType::CURATION_TASK_CANCELLATION: return "CURATION_TASK_CANCELLATION";
        case MessageType::CURATION_ANNOTATION_SUBMISSION: return "CURATION_ANNOTATION_SUBMISSION";
        case MessageType::CURATION_CONSENSUS_PROPOSAL: return "CURATION_CONSENSUS_PROPOSAL";
        case MessageType::CURATION_CONSENSUS_RESULT: return "CURATION_CONSENSUS_RESULT";
        case MessageType::CURATION_REPUTATION_UPDATE: return "CURATION_REPUTATION_UPDATE";
        case MessageType::CURATION_ANNOTATOR_REGISTRATION: return "CURATION_ANNOTATOR_REGISTRATION";
        case MessageType::CURATION_QUALITY_ALERT: return "CURATION_QUALITY_ALERT";
        
        case MessageType::RLHF_REWARD_MODEL_GRADIENT: return "RLHF_REWARD_MODEL_GRADIENT";
        case MessageType::RLHF_PPO_GRADIENT: return "RLHF_PPO_GRADIENT";
        case MessageType::RLHF_PREFERENCE_DATA_SHARE: return "RLHF_PREFERENCE_DATA_SHARE";
        case MessageType::RLHF_TRAINING_METRICS_UPDATE: return "RLHF_TRAINING_METRICS_UPDATE";
        case MessageType::RLHF_MODEL_STATE_SYNC: return "RLHF_MODEL_STATE_SYNC";
        case MessageType::RLHF_CONSENSUS_PROPOSAL: return "RLHF_CONSENSUS_PROPOSAL";
        case MessageType::RLHF_CONSENSUS_VOTE: return "RLHF_CONSENSUS_VOTE";
        case MessageType::RLHF_CONSENSUS_COMMIT: return "RLHF_CONSENSUS_COMMIT";
        
        case MessageType::SYSTEM_STATUS_UPDATE: return "SYSTEM_STATUS_UPDATE";
        case MessageType::CAPABILITY_ANNOUNCEMENT: return "CAPABILITY_ANNOUNCEMENT";
        case MessageType::RESOURCE_ALLOCATION_REQUEST: return "RESOURCE_ALLOCATION_REQUEST";
        case MessageType::RESOURCE_ALLOCATION_RESPONSE: return "RESOURCE_ALLOCATION_RESPONSE";
        case MessageType::PERFORMANCE_METRICS_SHARE: return "PERFORMANCE_METRICS_SHARE";
        case MessageType::ERROR_REPORT: return "ERROR_REPORT";
        
        default: return "UNKNOWN";
    }
}

} // namespace p2p
