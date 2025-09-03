#include "../include/feed_forward.hpp"
#ifdef USE_CUDA
#include "../include/cuda/cuda_check.cuh"
#include "../include/cuda/cuda_launch.cuh"
#include "../include/cuda/feed_forward_kernels.cuh"
#include "../include/cuda/backward_ops.cuh"
#include "../include/cuda/matrix_ops.cuh"
#include "../include/cuda/memory_manager.cuh"
#endif
#include "../include/scope_logger.hpp"
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

FeedForward::FeedForward(size_t hidden_size, size_t intermediate_size, float dropout)
    : dropout_prob(dropout) {
    SCOPE_LOG();
    // Initialize matrices with correct dimensions
    params_.gate_proj_weights = Matrix(hidden_size, intermediate_size);
    params_.up_proj_weights = Matrix(hidden_size, intermediate_size);
    params_.down_proj_weights = Matrix(intermediate_size, hidden_size);

    params_.gate_proj_bias = FloatVector(intermediate_size);
    params_.up_proj_bias = FloatVector(intermediate_size);
    params_.down_proj_bias = FloatVector(hidden_size);
    
    // Initialize gradients with same dimensions
    grads_.gate_proj_grad = Matrix(hidden_size, intermediate_size);
    grads_.up_proj_grad = Matrix(hidden_size, intermediate_size);
    grads_.down_proj_grad = Matrix(intermediate_size, hidden_size);

    grads_.gate_proj_bias_grad = FloatVector(intermediate_size);
    grads_.up_proj_bias_grad = FloatVector(intermediate_size);
    grads_.down_proj_bias_grad = FloatVector(hidden_size);

    // Initialize weights with Xavier/Glorot initialization
    initialize_weights();

    // Initialize gradients to zero
    grads_.gate_proj_grad.initialize_constant(0.0f);
    grads_.up_proj_grad.initialize_constant(0.0f);
    grads_.down_proj_grad.initialize_constant(0.0f);
    grads_.gate_proj_bias_grad.initialize_constant(0.0f);
    grads_.up_proj_bias_grad.initialize_constant(0.0f);
    grads_.down_proj_bias_grad.initialize_constant(0.0f);
}

// Helper for Swish activation
inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

inline float swish(float x) {
    return x * sigmoid(x);
}

inline float swish_derivative(float x) {
    float sig_x = sigmoid(x);
    return sig_x + x * sig_x * (1.0f - sig_x);
}


Matrix FeedForward::forward(const Matrix& input) {
    SCOPE_LOG();
    try {
#ifdef USE_CUDA
        // Convert input to CudaMatrix
        cuda::CudaMatrix d_input(input);

        // Ensure model parameters are on the GPU
        if (!params_gpu_) {
            params_gpu_ = std::make_unique<CudaParameters>();
            params_gpu_->gate_proj_weights = cuda::CudaMatrix(params_.gate_proj_weights);
            params_gpu_->up_proj_weights = cuda::CudaMatrix(params_.up_proj_weights);
            params_gpu_->down_proj_weights = cuda::CudaMatrix(params_.down_proj_weights);
        }
        
        // Allocate intermediate and output buffers on GPU
        cuda::CudaMatrix d_gated_linear_output(input.rows(), params_gpu_->gate_proj_weights.cols());
        cuda::CudaMatrix d_up_proj_output(input.rows(), params_gpu_->up_proj_weights.cols());
        cuda::CudaMatrix d_output(input.rows(), params_gpu_->down_proj_weights.cols());

        // Launch SwiGLU forward kernel
        cuda::kernels::swiglu_forward_kernel_launcher(
            d_input,
            params_gpu_->gate_proj_weights,
            params_gpu_->up_proj_weights,
            params_gpu_->down_proj_weights,
            d_gated_linear_output,
            d_up_proj_output,
            d_output
        );

        // Cache intermediate results for backward pass
        gated_linear_output_cache_gpu_ = std::move(d_gated_linear_output);
        up_proj_output_cache_gpu_ = std::move(d_up_proj_output);
        input_cache_gpu_ = std::move(d_input);
        
        // Copy result back to CPU
        return d_output.to_matrix();
#else
        // CPU implementation
        input_cache_ = input;

        // Gate and Up projections
        Matrix gate = matmul(input, params_.gate_proj_weights);
        gate.add_bias(params_.gate_proj_bias);

        Matrix up = matmul(input, params_.up_proj_weights);
        up.add_bias(params_.up_proj_bias);

        // Swish activation on the gate
        Matrix gate_swish = gate;
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < gate_swish.rows(); ++i) {
            for (size_t j = 0; j < gate_swish.cols(); ++j) {
                gate_swish(i, j) = swish(gate_swish(i, j));
            }
        }
        
        // Element-wise multiplication
        Matrix gated = gate_swish; // Re-use memory
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < gated.rows(); ++i) {
            for (size_t j = 0; j < gated.cols(); ++j) {
                gated(i, j) *= up(i, j);
            }
        }
        
        // Cache for backward pass
        intermediate_cache_ = gated;
        gate_cache_ = gate;
        up_cache_ = up;

        // Down projection
        Matrix output = matmul(gated, params_.down_proj_weights);
        output.add_bias(params_.down_proj_bias);

        return output;
#endif

    } catch (const std::exception& e) {
        throw std::runtime_error("FeedForward forward failed: " + std::string(e.what()));
    }
}

void FeedForward::save(std::ostream& os) const {
    size_t hidden_size = params_.down_proj_weights.cols();
    size_t intermediate_size = params_.gate_proj_weights.cols();
    
    os.write(reinterpret_cast<const char*>(&hidden_size), sizeof(hidden_size));
    os.write(reinterpret_cast<const char*>(&intermediate_size), sizeof(intermediate_size));
    
    // Save weights
    params_.gate_proj_weights.save(os);
    params_.up_proj_weights.save(os);
    params_.down_proj_weights.save(os);
    
    // Save biases
    os.write(reinterpret_cast<const char*>(params_.gate_proj_bias.data()), params_.gate_proj_bias.size() * sizeof(float));
    os.write(reinterpret_cast<const char*>(params_.up_proj_bias.data()), params_.up_proj_bias.size() * sizeof(float));
    os.write(reinterpret_cast<const char*>(params_.down_proj_bias.data()), params_.down_proj_bias.size() * sizeof(float));
}

std::unique_ptr<FeedForward> FeedForward::load(std::istream& is) {
    size_t hidden_size, intermediate_size;
    is.read(reinterpret_cast<char*>(&hidden_size), sizeof(hidden_size));
    is.read(reinterpret_cast<char*>(&intermediate_size), sizeof(intermediate_size));
    
    auto ffn = std::make_unique<FeedForward>(hidden_size, intermediate_size);
    
    // Load weights
    ffn->params_.gate_proj_weights = Matrix::load(is);
    ffn->params_.up_proj_weights = Matrix::load(is);
    ffn->params_.down_proj_weights = Matrix::load(is);
    
    // Load biases
    is.read(reinterpret_cast<char*>(ffn->params_.gate_proj_bias.data()), ffn->params_.gate_proj_bias.size() * sizeof(float));
    is.read(reinterpret_cast<char*>(ffn->params_.up_proj_bias.data()), ffn->params_.up_proj_bias.size() * sizeof(float));
    is.read(reinterpret_cast<char*>(ffn->params_.down_proj_bias.data()), ffn->params_.down_proj_bias.size() * sizeof(float));
    
    return ffn;
}

Matrix FeedForward::backward(const Matrix& grad_output, const Matrix& original_input) {
    SCOPE_LOG();
    try {
#ifdef USE_CUDA
        // Convert grad_output to CudaMatrix
        cuda::CudaMatrix d_grad_output(grad_output);

        // Allocate gradient buffers on GPU
        cuda::CudaMatrix d_grad_input(original_input.rows(), original_input.cols());
        if (!grads_gpu_) {
            grads_gpu_ = std::make_unique<CudaGradients>();
            grads_gpu_->gate_proj_weights_grad = cuda::CudaMatrix(grads_.gate_proj_weights_grad.rows(), grads_.gate_proj_weights_grad.cols());
            grads_gpu_->up_proj_weights_grad = cuda::CudaMatrix(grads_.up_proj_weights_grad.rows(), grads_.up_proj_weights_grad.cols());
            grads_gpu_->down_proj_weights_grad = cuda::CudaMatrix(grads_.down_proj_weights_grad.rows(), grads_.down_proj_weights_grad.cols());
        }


        // Launch SwiGLU backward kernel
        cuda::kernels::swiglu_backward_kernel_launcher(
            d_grad_output,
            input_cache_gpu_,
            params_gpu_->gate_proj_weights,
            params_gpu_->up_proj_weights,
            params_gpu_->down_proj_weights,
            gated_linear_output_cache_gpu_,
            up_proj_output_cache_gpu_,
            d_grad_input,
            grads_gpu_->gate_proj_weights_grad,
            grads_gpu_->up_proj_weights_grad,
            grads_gpu_->down_proj_weights_grad
        );

        // Copy gradients back to CPU
        grads_.gate_proj_weights_grad = grads_gpu_->gate_proj_weights_grad.to_matrix();
        grads_.up_proj_weights_grad = grads_gpu_->up_proj_weights_grad.to_matrix();
        grads_.down_proj_weights_grad = grads_gpu_->down_proj_weights_grad.to_matrix();
        
        return d_grad_input.to_matrix();
#else
        // CPU implementation
        // Step 1: Gradient w.r.t. the down projection
        grads_.down_proj_grad = matmul(intermediate_cache_.transpose(), grad_output);
        grads_.down_proj_bias_grad = grad_output.column_sum();

        // Step 2: Backpropagate through the down projection
        Matrix d_gated = matmul(grad_output, params_.down_proj_weights.transpose());

        // Step 3: Backpropagate through the element-wise multiplication
        // d_up = d_gated * swish(gate)
        // d_gate_swish = d_gated * up
        Matrix d_up = d_gated;
        Matrix gate_swish = gate_cache_;
        gate_swish.apply_swish(); // swish(gate)

        #pragma omp parallel for collapse(2)
        for(size_t i=0; i<d_up.rows(); ++i){
            for(size_t j=0; j<d_up.cols(); ++j){
                d_up(i,j) *= gate_swish(i,j);
            }
        }

        Matrix d_gate_swish = d_gated;
        #pragma omp parallel for collapse(2)
        for(size_t i=0; i<d_gate_swish.rows(); ++i){
            for(size_t j=0; j<d_gate_swish.cols(); ++j){
                d_gate_swish(i,j) *= up_cache_(i,j);
            }
        }

        // Step 4: Backpropagate through the Swish activation
        Matrix d_gate = d_gate_swish;
        #pragma omp parallel for collapse(2)
        for(size_t i=0; i<d_gate.rows(); ++i){
            for(size_t j=0; j<d_gate.cols(); ++j){
                d_gate(i,j) *= swish_derivative(gate_cache_(i,j));
            }
        }

        // Step 5: Gradients for up and gate projections
        grads_.up_proj_grad = matmul(input_cache_.transpose(), d_up);
        grads_.up_proj_bias_grad = d_up.column_sum();

        grads_.gate_proj_grad = matmul(input_cache_.transpose(), d_gate);
        grads_.gate_proj_bias_grad = d_gate.column_sum();

        // Step 6: Gradient with respect to the input
        Matrix d_input = matmul(d_gate, params_.gate_proj_weights.transpose());
        d_input += matmul(d_up, params_.up_proj_weights.transpose());

        return d_input;
#endif

    } catch (const std::exception& e) {
        std::cerr << "\nError in FeedForward::backward: " << e.what() << std::endl;
        throw;
    }
}

void FeedForward::update_parameters(float learning_rate) {
    SCOPE_LOG();
    // Update gate projection
    params_.gate_proj_weights -= learning_rate * grads_.gate_proj_grad;
    for(size_t i=0; i<params_.gate_proj_bias.size(); ++i) params_.gate_proj_bias[i] -= learning_rate * grads_.gate_proj_bias_grad[i];
    
    // Update up projection
    params_.up_proj_weights -= learning_rate * grads_.up_proj_grad;
    for(size_t i=0; i<params_.up_proj_bias.size(); ++i) params_.up_proj_bias[i] -= learning_rate * grads_.up_proj_bias_grad[i];

    // Update down projection
    params_.down_proj_weights -= learning_rate * grads_.down_proj_grad;
    for(size_t i=0; i<params_.down_proj_bias.size(); ++i) params_.down_proj_bias[i] -= learning_rate * grads_.down_proj_bias_grad[i];
    
    // Reset gradients
    grads_.gate_proj_grad.initialize_constant(0.0f);
    grads_.gate_proj_bias_grad.initialize_constant(0.0f);
    grads_.up_proj_grad.initialize_constant(0.0f);
    grads_.up_proj_bias_grad.initialize_constant(0.0f);
    grads_.down_proj_grad.initialize_constant(0.0f);
    grads_.down_proj_bias_grad.initialize_constant(0.0f);
}

void FeedForward::initialize_weights() {
    // Get dimensions
    size_t hidden_size = params_.gate_proj_weights.rows();
    size_t intermediate_size = params_.gate_proj_weights.cols();

    // Initialize with Xavier/Glorot initialization
    float gate_limit = std::sqrt(6.0f / (hidden_size + intermediate_size));
    params_.gate_proj_weights.initialize_random(gate_limit);
    
    float up_limit = std::sqrt(6.0f / (hidden_size + intermediate_size));
    params_.up_proj_weights.initialize_random(up_limit);

    float down_limit = std::sqrt(6.0f / (intermediate_size + hidden_size));
    params_.down_proj_weights.initialize_random(down_limit);

    // Initialize biases to small positive values
    params_.gate_proj_bias.initialize_constant(0.01f);
    params_.up_proj_bias.initialize_constant(0.01f);
    params_.down_proj_bias.initialize_constant(0.01f);
}