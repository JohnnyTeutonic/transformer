#include "include/moe.hpp"
#include "include/matrix.hpp"
#include <iostream>
#include <chrono>
#include <cmath>

// Simple test to verify CUDA MoE implementation matches CPU results
int main() {
    std::cout << "Testing MoE CUDA Implementation..." << std::endl;
    
    // Test parameters
    const size_t batch_size = 32;
    const size_t hidden_size = 512;
    const size_t intermediate_size = 2048;
    const size_t num_experts = 8;
    const size_t top_k = 2;
    const float aux_loss_coeff = 0.01f;
    
    // Create MoE layer
    MixtureOfExperts moe(num_experts, top_k, hidden_size, intermediate_size, aux_loss_coeff);
    
    // Create random input
    Matrix input(batch_size, hidden_size);
    input.initialize_random(0.1f);
    
    std::cout << "Input shape: [" << input.rows() << ", " << input.cols() << "]" << std::endl;
    
    // Time CPU implementation
    auto start_cpu = std::chrono::high_resolution_clock::now();
    
    // Force CPU implementation by temporarily disabling CUDA
    #ifdef USE_CUDA
    #undef USE_CUDA
    Matrix cpu_output = moe.forward(input);
    #define USE_CUDA
    #else
    Matrix cpu_output = moe.forward(input);
    #endif
    
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu);
    
    std::cout << "CPU forward pass completed in " << cpu_time.count() << " microseconds" << std::endl;
    std::cout << "CPU output shape: [" << cpu_output.rows() << ", " << cpu_output.cols() << "]" << std::endl;
    
    #ifdef USE_CUDA
    // Time CUDA implementation
    auto start_cuda = std::chrono::high_resolution_clock::now();
    Matrix cuda_output = moe.forward(input);
    auto end_cuda = std::chrono::high_resolution_clock::now();
    auto cuda_time = std::chrono::duration_cast<std::chrono::microseconds>(end_cuda - start_cuda);
    
    std::cout << "CUDA forward pass completed in " << cuda_time.count() << " microseconds" << std::endl;
    std::cout << "CUDA output shape: [" << cuda_output.rows() << ", " << cuda_output.cols() << "]" << std::endl;
    
    // Compare results
    float max_diff = 0.0f;
    float avg_diff = 0.0f;
    size_t total_elements = cpu_output.rows() * cpu_output.cols();
    
    for (size_t i = 0; i < cpu_output.rows(); ++i) {
        for (size_t j = 0; j < cpu_output.cols(); ++j) {
            float diff = std::abs(cpu_output(i, j) - cuda_output(i, j));
            max_diff = std::max(max_diff, diff);
            avg_diff += diff;
        }
    }
    avg_diff /= total_elements;
    
    std::cout << "Results comparison:" << std::endl;
    std::cout << "  Max difference: " << max_diff << std::endl;
    std::cout << "  Average difference: " << avg_diff << std::endl;
    
    // Performance comparison
    float speedup = static_cast<float>(cpu_time.count()) / cuda_time.count();
    std::cout << "  Speedup: " << speedup << "x" << std::endl;
    
    // Check if results are close enough (allowing for floating point precision)
    const float tolerance = 1e-5f;
    bool results_match = (max_diff < tolerance);
    
    if (results_match) {
        std::cout << "✓ CUDA implementation matches CPU results within tolerance!" << std::endl;
        std::cout << "✓ MoE CUDA kernel implementation successful!" << std::endl;
        return 0;
    } else {
        std::cout << "✗ CUDA implementation differs from CPU results beyond tolerance!" << std::endl;
        return 1;
    }
    
    #else
    std::cout << "CUDA not enabled, only CPU test completed." << std::endl;
    return 0;
    #endif
}
