#include "../include/layer_norm.hpp"
#include "../include/cuda/matrix_ops.cuh"
#include <cmath>
#include <omp.h>
#include "../include/cuda/backward_ops.cuh"
#include <random>

LayerNorm::LayerNorm(size_t hidden_size, float eps)
    : hidden_size_(hidden_size), 
      eps_(eps),
      gamma_(Matrix(1, hidden_size)),
      beta_(Matrix(1, hidden_size)),
      input_cache_(Matrix(1, hidden_size)),
      output_cache_(Matrix(1, hidden_size)),
      grad_gamma_(Matrix(1, hidden_size)),
      grad_beta_(Matrix(1, hidden_size)) {
    
    // Initialize gamma with small random values around 1.0
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> gamma_dist(1.0f, 0.02f);
    
    // Initialize beta with small random values around 0
    std::normal_distribution<float> beta_dist(0.0f, 0.02f);
    
    for (size_t i = 0; i < hidden_size; ++i) {
        gamma_(0, i) = gamma_dist(gen);
        beta_(0, i) = beta_dist(gen);
    }
    
    std::cout << "LayerNorm initialized with hidden_size=" << hidden_size 
              << ", eps=" << eps << std::endl;
}

Matrix LayerNorm::forward(const Matrix& input) {
    std::cout << "\n=== LayerNorm::forward START ===" << std::endl;
    std::cout << "Input dimensions: " << input.rows() << "x" << input.cols() << std::endl;
    
    Matrix output(input.rows(), input.cols());
    
    // Track statistics for debugging
    float min_mean = std::numeric_limits<float>::infinity();
    float max_mean = -std::numeric_limits<float>::infinity();
    float min_var = std::numeric_limits<float>::infinity();
    float max_var = -std::numeric_limits<float>::infinity();
    float min_norm = std::numeric_limits<float>::infinity();
    float max_norm = -std::numeric_limits<float>::infinity();
    
    // Normalize each row independently
    for (size_t i = 0; i < input.rows(); ++i) {
        // Calculate mean
        float mean = 0.0f;
        for (size_t j = 0; j < input.cols(); ++j) {
            mean += input(i, j);
        }
        mean /= input.cols();
        min_mean = std::min(min_mean, mean);
        max_mean = std::max(max_mean, mean);
        
        // Calculate variance
        float var = 0.0f;
        for (size_t j = 0; j < input.cols(); ++j) {
            float diff = input(i, j) - mean;
            var += diff * diff;
        }
        var /= input.cols();
        min_var = std::min(min_var, var);
        max_var = std::max(max_var, var);
        
        // Normalize and apply scale and shift
        float std_dev = std::sqrt(var + eps_);
        for (size_t j = 0; j < input.cols(); ++j) {
            float normalized = (input(i, j) - mean) / std_dev;
            output(i, j) = gamma_(0, j) * normalized + beta_(0, j);
            min_norm = std::min(min_norm, output(i, j));
            max_norm = std::max(max_norm, output(i, j));
        }
    }
    
    // Print statistics
    std::cout << "Layer Norm Statistics:" << std::endl;
    std::cout << "Mean range: [" << min_mean << ", " << max_mean << "]" << std::endl;
    std::cout << "Variance range: [" << min_var << ", " << max_var << "]" << std::endl;
    std::cout << "Output range: [" << min_norm << ", " << max_norm << "]" << std::endl;
    
    // Count near-zero values
    size_t near_zero = 0;
    for (size_t i = 0; i < output.rows(); ++i) {
        for (size_t j = 0; j < output.cols(); ++j) {
            if (std::abs(output(i, j)) < 1e-6) {
                near_zero++;
            }
        }
    }
    std::cout << "Near-zero values: " << near_zero << "/" << (output.rows() * output.cols()) << std::endl;
    
    // Check gamma and beta values
    float min_gamma = std::numeric_limits<float>::infinity();
    float max_gamma = -std::numeric_limits<float>::infinity();
    float min_beta = std::numeric_limits<float>::infinity();
    float max_beta = -std::numeric_limits<float>::infinity();
    
    for (size_t j = 0; j < gamma_.size(); ++j) {
        min_gamma = std::min(min_gamma, gamma_(0, j));
        max_gamma = std::max(max_gamma, gamma_(0, j));
        min_beta = std::min(min_beta, beta_(0, j));
        max_beta = std::max(max_beta, beta_(0, j));
    }
    
    std::cout << "Gamma range: [" << min_gamma << ", " << max_gamma << "]" << std::endl;
    std::cout << "Beta range: [" << min_beta << ", " << max_beta << "]" << std::endl;
    std::cout << "=== LayerNorm::forward END ===\n" << std::endl;
    
    return output;
}

Matrix LayerNorm::compute_gradients(const Matrix& grad_output) {
    try {
        Matrix grad_input(input_cache_.rows(), input_cache_.cols());
#ifdef USE_CUDA
        try {
            cuda::layer_norm_backward(grad_output, input_cache_, gamma_, 
                                    grad_input, eps_);
            return grad_input;
        } catch (const std::runtime_error& e) {
            std::cerr << "CUDA layer norm backward failed, falling back to CPU: " << e.what() << std::endl;
#endif
            // CPU implementation
            grad_gamma_ = Matrix(1, hidden_size_);
            grad_beta_ = Matrix(1, hidden_size_);

            for (size_t i = 0; i < input_cache_.rows(); ++i) {
                float mean = 0.0f;
                float var = 0.0f;
                
                // Compute mean and variance (cached from forward pass)
                for (size_t j = 0; j < input_cache_.cols(); ++j) {
                    mean += input_cache_(i, j);
                }
                mean /= input_cache_.cols();
                
                for (size_t j = 0; j < input_cache_.cols(); ++j) {
                    float diff = input_cache_(i, j) - mean;
                    var += diff * diff;
                }
                var /= input_cache_.cols();
                
                float std = std::sqrt(var + eps_);
                float inv_std = 1.0f / std;
                
                // Compute gradients
                for (size_t j = 0; j < input_cache_.cols(); ++j) {
                    float x_norm = (input_cache_(i, j) - mean) * inv_std;
                    grad_gamma_(0, j) += grad_output(i, j) * x_norm;
                    grad_beta_(0, j) += grad_output(i, j);
                    
                    // Gradient with respect to input
                    grad_input(i, j) = gamma_(0, j) * grad_output(i, j) * inv_std;
                }
            }
#ifdef USE_CUDA
        }
#endif
        return grad_input;
    } catch (const std::exception& e) {
        throw std::runtime_error("LayerNorm backward failed: " + std::string(e.what()));
    }
}

void LayerNorm::save(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(&eps_), sizeof(eps_));
    os.write(reinterpret_cast<const char*>(&hidden_size_), sizeof(hidden_size_));
    // Save gamma and beta as raw data
    os.write(reinterpret_cast<const char*>(gamma_.get_data()), hidden_size_ * sizeof(float));
    os.write(reinterpret_cast<const char*>(beta_.get_data()), hidden_size_ * sizeof(float));
}

std::unique_ptr<LayerNorm> LayerNorm::load(std::istream& is) {
    float eps;
    is.read(reinterpret_cast<char*>(&eps), sizeof(eps));

    // Read hidden size from stream
    size_t hidden_size;
    is.read(reinterpret_cast<char*>(&hidden_size), sizeof(hidden_size));

    auto ln = std::make_unique<LayerNorm>(hidden_size, eps);

    // Load gamma and beta data directly
    is.read(reinterpret_cast<char*>(ln->gamma_.get_data()), hidden_size * sizeof(float));
    is.read(reinterpret_cast<char*>(ln->beta_.get_data()), hidden_size * sizeof(float));

    return ln;
}