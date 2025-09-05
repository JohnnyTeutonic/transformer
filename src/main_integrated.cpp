#include "../include/integrated_platform.hpp"
#include <iostream>
#include <signal.h>
#include <thread>
#include <chrono>

// Global platform instance for signal handling
std::unique_ptr<integrated::IntegratedPlatform> g_platform;

void signal_handler(int signal) {
    std::cout << "\nReceived signal " << signal << ", shutting down gracefully..." << std::endl;
    if (g_platform) {
        g_platform->stop();
    }
    exit(0);
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --node-type <type>     Node type: full, training, annotation, lightweight (default: full)\n";
    std::cout << "  --node-id <id>         Unique node identifier (default: auto-generated)\n";
    std::cout << "  --p2p-port <port>      P2P network port (default: 7777)\n";
    std::cout << "  --web-port <port>      Web interface port (default: 8080)\n";
    std::cout << "  --bootstrap <peers>    Comma-separated list of bootstrap peers\n";
    std::cout << "  --model-path <path>    Path to model configuration file\n";
    std::cout << "  --checkpoint <path>    Path to model checkpoint file\n";
    std::cout << "  --help                 Show this help message\n";
    std::cout << "\nExample:\n";
    std::cout << "  " << program_name << " --node-type full --node-id node1 --bootstrap 192.168.1.100:7777,192.168.1.101:7777\n";
}

std::vector<std::string> split_string(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    
    while (std::getline(ss, token, delimiter)) {
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    
    return tokens;
}

std::string generate_node_id() {
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    return "node_" + std::to_string(timestamp);
}

int main(int argc, char* argv[]) {
    std::cout << "Distributed AI Training Platform - Integrated Node" << std::endl;
    std::cout << "=================================================" << std::endl;

    // Parse command line arguments
    std::string node_type = "full";
    std::string node_id = generate_node_id();
    uint16_t p2p_port = 7777;
    uint16_t web_port = 8080;
    std::vector<std::string> bootstrap_peers;
    std::string model_path;
    std::string checkpoint_path;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--node-type" && i + 1 < argc) {
            node_type = argv[++i];
        } else if (arg == "--node-id" && i + 1 < argc) {
            node_id = argv[++i];
        } else if (arg == "--p2p-port" && i + 1 < argc) {
            p2p_port = static_cast<uint16_t>(std::stoi(argv[++i]));
        } else if (arg == "--web-port" && i + 1 < argc) {
            web_port = static_cast<uint16_t>(std::stoi(argv[++i]));
        } else if (arg == "--bootstrap" && i + 1 < argc) {
            bootstrap_peers = split_string(argv[++i], ',');
        } else if (arg == "--model-path" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--checkpoint" && i + 1 < argc) {
            checkpoint_path = argv[++i];
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    // Setup signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    try {
        // Create platform configuration based on node type
        integrated::PlatformConfig config;
        
        if (node_type == "full") {
            config = integrated::PlatformFactory::create_full_node_config(node_id);
        } else if (node_type == "training") {
            config = integrated::PlatformFactory::create_training_node_config(node_id);
        } else if (node_type == "annotation") {
            config = integrated::PlatformFactory::create_annotation_node_config(node_id);
        } else if (node_type == "lightweight") {
            config = integrated::PlatformFactory::create_lightweight_node_config(node_id);
        } else {
            std::cerr << "Invalid node type: " << node_type << std::endl;
            std::cerr << "Valid types: full, training, annotation, lightweight" << std::endl;
            return 1;
        }

        // Apply command line overrides
        config.p2p_port = p2p_port;
        config.web_port = web_port;
        config.bootstrap_peers = bootstrap_peers;
        
        if (!model_path.empty()) {
            config.model_config_path = model_path;
        }
        
        if (!checkpoint_path.empty()) {
            config.checkpoint_path = checkpoint_path;
        }

        std::cout << "\nStarting " << node_type << " node with configuration:" << std::endl;
        std::cout << "- Node ID: " << config.node_id << std::endl;
        std::cout << "- P2P Port: " << config.p2p_port << std::endl;
        std::cout << "- Web Port: " << config.web_port << std::endl;
        std::cout << "- Bootstrap Peers: " << bootstrap_peers.size() << std::endl;
        std::cout << "- Curation: " << (config.enable_curation ? "enabled" : "disabled") << std::endl;
        std::cout << "- RLHF: " << (config.enable_rlhf ? "enabled" : "disabled") << std::endl;
        std::cout << "- Web Interface: " << (config.enable_web_interface ? "enabled" : "disabled") << std::endl;

        // Create and initialize platform
        g_platform = integrated::PlatformFactory::create_platform(config);
        
        // Set up event callback
        g_platform->set_event_callback([](integrated::PlatformEvent event, const std::string& details) {
            std::cout << "[EVENT] " << static_cast<int>(event) << ": " << details << std::endl;
        });

        // Initialize platform
        if (!g_platform->initialize()) {
            std::cerr << "Failed to initialize platform" << std::endl;
            return 1;
        }

        // Start platform
        if (!g_platform->start()) {
            std::cerr << "Failed to start platform" << std::endl;
            return 1;
        }

        std::cout << "\nPlatform started successfully!" << std::endl;
        
        if (config.enable_web_interface) {
            std::cout << "Web interface available at: http://localhost:" << config.web_port << std::endl;
        }
        
        std::cout << "Press Ctrl+C to stop the platform" << std::endl;

        // Main loop - monitor platform status
        while (g_platform->is_running()) {
            std::this_thread::sleep_for(std::chrono::seconds(30));
            
            // Print periodic status updates
            auto stats = g_platform->get_stats();
            auto health = g_platform->check_health();
            
            std::cout << "\n--- Platform Status ---" << std::endl;
            std::cout << "Connected Peers: " << stats.connected_peers << std::endl;
            std::cout << "Annotation Tasks: " << stats.completed_annotation_tasks << "/" << stats.total_annotation_tasks << std::endl;
            std::cout << "Active Annotators: " << stats.active_annotators << std::endl;
            std::cout << "RLHF Iterations: " << stats.rlhf_iterations_completed << std::endl;
            std::cout << "Web Sessions: " << stats.active_web_sessions << std::endl;
            std::cout << "CPU Usage: " << stats.cpu_usage_percent << "%" << std::endl;
            std::cout << "Memory Usage: " << stats.memory_usage_percent << "%" << std::endl;
            std::cout << "Health Score: " << health.overall_score << " (" << (health.is_healthy ? "healthy" : "issues detected") << ")" << std::endl;
            
            if (!health.is_healthy) {
                std::cout << "Health Issues:" << std::endl;
                for (const auto& issue : health.issues) {
                    std::cout << "  - " << issue << std::endl;
                }
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Platform error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Platform shutdown complete" << std::endl;
    return 0;
}

// Additional utility functions for demonstration
namespace demo {

void run_annotation_demo(integrated::IntegratedPlatform& platform) {
    std::cout << "\n=== Running Annotation Demo ===" << std::endl;
    
    // Create sample annotation tasks
    curation::AnnotationTask task1;
    task1.data_type = "text";
    task1.content = "The quick brown fox jumps over the lazy dog. This is a sample text for annotation.";
    task1.context = "Please evaluate the quality and helpfulness of this text.";
    task1.label_schema = {"quality", "helpfulness", "clarity"};
    task1.required_annotators = 3;
    task1.difficulty_score = 0.3f;
    
    curation::AnnotationTask task2;
    task2.data_type = "conversation";
    task2.content = "Human: What is the capital of France?\nAI: The capital of France is Paris.";
    task2.context = "Please evaluate this AI response for accuracy and helpfulness.";
    task2.label_schema = {"accuracy", "helpfulness", "completeness"};
    task2.required_annotators = 3;
    task2.difficulty_score = 0.2f;
    
    // Submit tasks
    if (platform.submit_annotation_task(task1)) {
        std::cout << "Submitted text annotation task" << std::endl;
    }
    
    if (platform.submit_annotation_task(task2)) {
        std::cout << "Submitted conversation annotation task" << std::endl;
    }
}

void run_training_demo(integrated::IntegratedPlatform& platform) {
    std::cout << "\n=== Running Training Demo ===" << std::endl;
    
    // Sample training data
    std::vector<std::string> training_data = {
        "The weather is nice today.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
        "The Internet has revolutionized how we communicate.",
        "Renewable energy sources are important for sustainability."
    };
    
    if (platform.start_training_session(training_data)) {
        std::cout << "Started training session with " << training_data.size() << " samples" << std::endl;
    }
}

void run_rlhf_demo(integrated::IntegratedPlatform& platform) {
    std::cout << "\n=== Running RLHF Demo ===" << std::endl;
    
    // Start RLHF training with sample parameters
    uint32_t reward_epochs = 2;
    uint32_t ppo_iterations = 5;
    
    if (platform.start_rlhf_training(reward_epochs, ppo_iterations)) {
        std::cout << "Started RLHF training: " << reward_epochs << " reward epochs, " << ppo_iterations << " PPO iterations" << std::endl;
    }
}

void print_platform_info(const integrated::IntegratedPlatform& platform) {
    std::cout << "\n=== Platform Information ===" << std::endl;
    
    auto config = platform.get_config();
    auto capabilities = platform.get_capabilities();
    auto peers = platform.get_connected_peers();
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Node ID: " << config.node_id << std::endl;
    std::cout << "  P2P Port: " << config.p2p_port << std::endl;
    std::cout << "  Web Port: " << config.web_port << std::endl;
    std::cout << "  Max Memory: " << config.max_memory_mb << " MB" << std::endl;
    std::cout << "  Max CPU Threads: " << config.max_cpu_threads << std::endl;
    
    std::cout << "\nCapabilities:" << std::endl;
    std::cout << "  Can Train Models: " << (capabilities.can_train_models ? "yes" : "no") << std::endl;
    std::cout << "  Can Annotate Data: " << (capabilities.can_annotate_data ? "yes" : "no") << std::endl;
    std::cout << "  Can Serve Web: " << (capabilities.can_serve_web_interface ? "yes" : "no") << std::endl;
    std::cout << "  Available Memory: " << capabilities.available_memory_mb << " MB" << std::endl;
    std::cout << "  CPU Cores: " << capabilities.cpu_cores << std::endl;
    std::cout << "  Has GPU: " << (capabilities.has_gpu ? "yes" : "no") << std::endl;
    
    std::cout << "\nConnected Peers: " << peers.size() << std::endl;
    for (const auto& peer : peers) {
        std::cout << "  - " << peer << std::endl;
    }
}

} // namespace demo
