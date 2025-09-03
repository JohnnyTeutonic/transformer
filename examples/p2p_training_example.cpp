#include "../include/p2p_network.hpp"
#include "../include/distributed_transformer.hpp"
#include "../include/tiktoken_tokenizer.hpp"
#include "../include/config.hpp"
#include "../include/utils.hpp"
#include <iostream>
#include <memory>
#include <thread>
#include <chrono>
#include <vector>
#include <string>
#include <atomic> // Added for atomic bool

using namespace p2p;

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --port PORT           Bind to specific port (default: 8888)" << std::endl;
    std::cout << "  --bootstrap HOST:PORT  Connect to bootstrap node" << std::endl;
    std::cout << "  --node-id ID          Use specific node ID" << std::endl;
    std::cout << "  --help                Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  # Start first node (bootstrap)" << std::endl;
    std::cout << "  " << program_name << " --port 8888" << std::endl;
    std::cout << std::endl;
    std::cout << "  # Start second node, connect to first" << std::endl;
    std::cout << "  " << program_name << " --port 8889 --bootstrap 127.0.0.1:8888" << std::endl;
    std::cout << std::endl;
    std::cout << "  # Start third node, connect to network" << std::endl;
    std::cout << "  " << program_name << " --port 8890 --bootstrap 127.0.0.1:8888" << std::endl;
}

P2PConfig parse_arguments(int argc, char* argv[]) {
    P2PConfig config;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else if (arg == "--port" && i + 1 < argc) {
            config.bind_port = static_cast<uint16_t>(std::stoi(argv[++i]));
        } else if (arg == "--bootstrap" && i + 1 < argc) {
            config.bootstrap_nodes.push_back(argv[++i]);
        } else if (arg == "--node-id" && i + 1 < argc) {
            config.node_id = argv[++i];
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            exit(1);
        }
    }
    
    return config;
}

void run_interactive_mode(std::shared_ptr<P2PNetwork> network, 
                         std::shared_ptr<P2PTrainingCoordinator> coordinator,
                         std::atomic<bool>& training_running) {
    std::cout << "\n=== P2P Training Interactive Mode ===" << std::endl;
    std::cout << "Commands:" << std::endl;
    std::cout << "  stats    - Show network statistics" << std::endl;
    std::cout << "  peers    - List active peers" << std::endl;
    std::cout << "  train    - Start training coordination" << std::endl;
    std::cout << "  stop     - Stop training coordination" << std::endl;
    std::cout << "  test     - Run test gradient consensus" << std::endl;
    std::cout << "  quit     - Exit program" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    std::string command;
    while (std::cout << "p2p> " && std::getline(std::cin, command)) {
        if (command == "quit" || command == "exit") {
            break;
        } else if (command == "stats") {
            network->get_stats().print_stats();
            
            auto training_stats = coordinator->get_training_stats();
            std::cout << "\n=== Training Statistics ===" << std::endl;
            std::cout << "Training steps: " << training_stats.training_steps << std::endl;
            std::cout << "Consensus failures: " << training_stats.consensus_failures << std::endl;
            std::cout << "Average step time: " << training_stats.average_step_time_ms << " ms" << std::endl;
            std::cout << "Active training nodes: " << training_stats.active_training_nodes << std::endl;
            std::cout << "===========================\n" << std::endl;
            
        } else if (command == "peers") {
            auto peers = network->get_active_peers();
            std::cout << "\nActive peers (" << peers.size() << "):" << std::endl;
            for (const auto& peer : peers) {
                std::cout << "  " << peer.node_id 
                          << " (" << peer.ip_address << ":" << peer.port << ")"
                          << " - Reputation: " << peer.reputation_score
                          << " - Trusted: " << (peer.is_trusted ? "Yes" : "No") << std::endl;
            }
            std::cout << std::endl;
            
        } else if (command == "train") {
            if (!training_running.load()) {
                training_running.store(true);
                coordinator->start();
                std::cout << "Asynchronous training started. Submitting dummy gradients..." << std::endl;
                
                // Example of a non-blocking training loop
                std::thread([&]() {
                    int batch_id = 0;
                    while(training_running.load()) {
                        std::vector<Matrix> dummy_gradients;
                        dummy_gradients.emplace_back(10, 10);
                        dummy_gradients.back().initialize_random(0.1f);
                        
                        coordinator->submit_gradients(dummy_gradients);
                        std::cout << "Submitted gradients for batch " << batch_id++ << std::endl;
                        
                        std::this_thread::sleep_for(std::chrono::seconds(5)); // Simulate work
                    }
                }).detach();

            } else {
                std::cout << "Training is already running." << std::endl;
            }
            
        } else if (command == "stop") {
            if (training_running.load()) {
                training_running.store(false);
                coordinator->stop();
                std::cout << "Training stopped." << std::endl;
            } else {
                std::cout << "Training is not running." << std::endl;
            }
            
        } else if (command == "test") {
            std::cout << "Submitting a single test gradient for asynchronous consensus..." << std::endl;
            std::vector<Matrix> test_gradients;
            test_gradients.emplace_back(10, 10);
            test_gradients.back().initialize_random(0.1f);
            coordinator->submit_gradients(test_gradients);
            
        } else if (command.empty()) {
            // Ignore empty commands
            continue;
        } else {
            std::cout << "Unknown command: " << command << std::endl;
            std::cout << "Type 'quit' to exit or use one of the available commands." << std::endl;
        }
    }
}

int main(int argc, char* argv[]) {
    try {
        std::cout << "=== P2P Distributed Transformer Training ===" << std::endl;
        std::cout << "Building peer-to-peer training network..." << std::endl;
        
        // Parse command line arguments
        P2PConfig p2p_config = parse_arguments(argc, argv);
        
        std::cout << "Node configuration:" << std::endl;
        std::cout << "  Node ID: " << p2p_config.node_id << std::endl;
        std::cout << "  Bind port: " << p2p_config.bind_port << std::endl;
        std::cout << "  Bootstrap nodes: ";
        if (p2p_config.bootstrap_nodes.empty()) {
            std::cout << "None (this is a bootstrap node)" << std::endl;
        } else {
            for (const auto& node : p2p_config.bootstrap_nodes) {
                std::cout << node << " ";
            }
            std::cout << std::endl;
        }
        
        // Create transformer configuration
        TransformerConfig transformer_config;
        transformer_config.vocab_size = 1000;  // Small vocab for demo
        transformer_config.hidden_size = 256;
        transformer_config.num_layers = 4;
        transformer_config.num_heads = 8;
        transformer_config.head_dim = 32;
        transformer_config.intermediate_size = 1024;
        transformer_config.max_seq_length = 128;
        transformer_config.dropout_rate = 0.1f;
        
        std::cout << "\nTransformer configuration:" << std::endl;
        std::cout << "  Vocabulary size: " << transformer_config.vocab_size << std::endl;
        std::cout << "  Hidden size: " << transformer_config.hidden_size << std::endl;
        std::cout << "  Number of layers: " << transformer_config.num_layers << std::endl;
        std::cout << "  Number of heads: " << transformer_config.num_heads << std::endl;
        
        // Initialize tokenizer
        auto tokenizer = std::make_shared<TiktokenTokenizer>();
        
        // Create some dummy training data for the tokenizer
        std::vector<std::string> dummy_vocab = {
            "hello", "world", "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
            "artificial", "intelligence", "machine", "learning", "neural", "network", "transformer",
            "attention", "gradient", "training", "distributed", "peer", "consensus", "blockchain"
        };
        
        // Build a simple vocabulary (in production, would load from file)
        std::cout << "\nInitializing tokenizer with dummy vocabulary..." << std::endl;
        // Note: This is a simplified initialization - in production you'd load from training data
        
        // Create P2P network
        std::cout << "\nCreating P2P network..." << std::endl;
        auto p2p_network = std::make_shared<P2PNetwork>(p2p_config);
        
        // Create distributed transformer
        std::cout << "Creating distributed transformer..." << std::endl;
        auto distributed_transformer = std::make_shared<DistributedTransformer>(
            transformer_config, tokenizer, &argc, &argv);
        
        // Create P2P training coordinator
        std::cout << "Creating P2P training coordinator..." << std::endl;
        auto training_coordinator = std::make_shared<P2PTrainingCoordinator>(
            distributed_transformer, p2p_network);
        
        // Link the network and coordinator
        p2p_network->set_coordinator(training_coordinator);

        // Start P2P network
        std::cout << "\nStarting P2P network..." << std::endl;
        if (!p2p_network->start()) {
            std::cerr << "Failed to start P2P network" << std::endl;
            return 1;
        }
        
        // Wait a bit for network to establish connections
        std::cout << "Waiting for network connections..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(3));
        
        // Show initial network status
        auto peers = p2p_network->get_active_peers();
        std::cout << "\nNetwork established with " << peers.size() << " peers" << std::endl;
        
        if (peers.empty()) {
            std::cout << "No peers connected. This node is running in standalone mode." << std::endl;
            std::cout << "Other nodes can connect to this bootstrap node at: " 
                      << utils::get_local_ip_address() << ":" << p2p_config.bind_port << std::endl;
        } else {
            std::cout << "Connected to P2P network with the following peers:" << std::endl;
            for (const auto& peer : peers) {
                std::cout << "  - " << peer.node_id << " (" << peer.ip_address << ":" << peer.port << ")" << std::endl;
            }
        }
        
        // Demonstrate basic P2P functionality
        std::cout << "\n=== Testing P2P Functionality ===" << std::endl;
        
        // Test 1: Network statistics
        std::cout << "\nTest 1: Network Statistics" << std::endl;
        p2p_network->get_stats().print_stats();
        
        // Test 2: Gradient consensus (if we have peers)
        if (!peers.empty()) {
            std::cout << "\nTest 2: Gradient Consensus" << std::endl;
            
            // Create dummy gradients
            std::vector<Matrix> test_gradients;
            test_gradients.emplace_back(5, 5);
            test_gradients[0].initialize_random(0.1f);
            
            std::string proposal_id = p2p_network->propose_gradient(test_gradients, 0, 0);
            if (!proposal_id.empty()) {
                std::cout << "Proposed test gradients: " << proposal_id << std::endl;
                
                bool consensus = p2p_network->wait_for_consensus(proposal_id, 5000);
                if (consensus) {
                    std::cout << "✓ Gradient consensus test passed!" << std::endl;
                } else {
                    std::cout << "✗ Gradient consensus test failed (timeout)" << std::endl;
                }
            }
        }
        
        // Test 3: Training coordination
        std::cout << "\nTest 3: Training Coordination" << std::endl;
        if (training_coordinator->start_distributed_training()) {
            std::cout << "✓ Training coordination started successfully" << std::endl;
            
            // Let it run for a few seconds
            std::this_thread::sleep_for(std::chrono::seconds(2));
            
            auto training_stats = training_coordinator->get_training_stats();
            std::cout << "Training stats - Steps: " << training_stats.training_steps 
                      << ", Active nodes: " << training_stats.active_training_nodes << std::endl;
            
            training_coordinator->stop_distributed_training();
            std::cout << "✓ Training coordination stopped successfully" << std::endl;
        } else {
            std::cout << "✗ Failed to start training coordination" << std::endl;
        }
        
        // Enter interactive mode
        std::cout << "\n=== Entering Interactive Mode ===" << std::endl;
        std::cout << "The P2P network is now running. You can:" << std::endl;
        std::cout << "- Monitor network statistics" << std::endl;
        std::cout << "- Test gradient consensus" << std::endl;
        std::cout << "- Start/stop distributed training" << std::endl;
        std::cout << "- View connected peers" << std::endl;
        
        std::atomic<bool> training_running(false);
        run_interactive_mode(p2p_network, training_coordinator, training_running);
        
        // Cleanup
        std::cout << "\nShutting down..." << std::endl;
        if (training_running.load()) {
            coordinator->stop();
        }
        p2p_network->stop();
        
        std::cout << "P2P training example completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
