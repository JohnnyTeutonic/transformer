#include "../include/layer_norm.hpp"
#ifdef USE_CUDA
#include "../include/cuda/matrix_ops.cuh"
#endif
#include "../include/gradient_diagnostics.hpp"
#include <cmath>
#include <omp.h>
#include "../include/cuda/backward_ops.cuh"

LayerNorm::LayerNorm(size_t hidden_size_, float eps_, bool rms_)
    : hidden_size_(hidden_size_),
      eps_(eps_),
      rms_mode_(rms_) {
    // Initialize parameters
    params_.gamma = Matrix(1, hidden_size_, 1.0f);  // Initialize to ones
    params_.beta = Matrix(1, hidden_size_, 0.0f);   // Initialize to zeros

    // Initialize gradients
    grads_.gamma_grad = Matrix(1, hidden_size_);
    grads_.beta_grad = Matrix(1, hidden_size_);
}

Matrix LayerNorm::forward(const Matrix& input) {
    try {
        if (input.cols() != hidden_size_) {
            throw std::runtime_error("Input dimension mismatch in LayerNorm");
        }

        input_cache_ = input;
        Matrix output(input.rows(), input.cols());

        if (rms_mode_) {
            // RMSNorm (LLaMA): y = gamma * x / sqrt(mean(x^2) + eps)
            #pragma omp parallel for
            for (int i = 0; i < static_cast<int>(input.rows()); i++) {
                float sum_sq = 0.0f;
                for (int j = 0; j < static_cast<int>(input.cols()); j++) {
                    sum_sq += input(i, j) * input(i, j);
                }
                float inv_rms = 1.0f / std::sqrt(sum_sq / input.cols() + eps_);
                for (int j = 0; j < static_cast<int>(input.cols()); j++) {
                    output(i, j) = params_.gamma(0, j) * input(i, j) * inv_rms;
                }
            }
            return output;
        }

// CUDA layer norm disabled - memory transfer overhead dominates for small operations
// Would need persistent GPU buffers for parameters to be efficient
#if 0  // USE_CUDA - disabled for now
        cuda::layer_norm_forward(input, params_.gamma, params_.beta, output, eps_);
#else
        // CPU implementation
        Matrix mean(input.rows(), 1, 0.0f);
        Matrix var(input.rows(), 1, 0.0f);

        // Compute mean for each row (MSVC: loop vars must be signed int)
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(input.rows()); i++) {
            float sum = 0.0f;
            for (int j = 0; j < static_cast<int>(input.cols()); j++) {
                sum += input(i, j);
            }
            mean(i, 0) = sum / input.cols();
        }

        // Compute variance for each row (MSVC: loop vars must be signed int)
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(input.rows()); i++) {
            float sum_sq = 0.0f;
            for (int j = 0; j < static_cast<int>(input.cols()); j++) {
                float diff = input(i, j) - mean(i, 0);
                sum_sq += diff * diff;
            }
            var(i, 0) = sum_sq / input.cols();
        }

        // Normalize and apply scale/shift (MSVC: collapse ignored, loop vars must be signed int)
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < static_cast<int>(input.rows()); i++) {
            for (int j = 0; j < static_cast<int>(input.cols()); j++) {
                float normalized = (input(i, j) - mean(i, 0)) / std::sqrt(var(i, 0) + eps_);
                output(i, j) = params_.gamma(0, j) * normalized + params_.beta(0, j);
            }
        }
#endif

        return output;

    } catch (const std::exception& e) {
        throw std::runtime_error("Error in LayerNorm forward: " + std::string(e.what()));
    }
}

Matrix LayerNorm::compute_gradients(const Matrix& grad_output) {
    try {
        Matrix grad_input(grad_output.rows(), grad_output.cols());
        
        // Batch size for gradient normalization
        const float batch_size = static_cast<float>(grad_output.rows());
        const float grad_scale = 1.0f / batch_size;
        
        // CRITICAL: Reset gradients to zero before accumulating!
        // Without this, gradients accumulate across batches causing divergence.
        for (size_t j = 0; j < grads_.gamma_grad.cols(); ++j) {
            grads_.gamma_grad(0, j) = 0.0f;
            grads_.beta_grad(0, j) = 0.0f;
        }

        if (rms_mode_) {
            // Exact RMSNorm backward.
            // y_j = gamma_j * x_j * r, r = 1/sqrt(mean(x^2) + eps)
            // dx_j = r * (gamma_j * g_j - x_j * r^2 / d * sum_k(g_k * gamma_k * x_k))
            // dgamma_j = sum_rows(g_j * x_j * r) / batch
            const int d = static_cast<int>(input_cache_.cols());
            #pragma omp parallel for
            for (int i = 0; i < static_cast<int>(input_cache_.rows()); i++) {
                float sum_sq = 0.0f;
                for (int j = 0; j < d; j++) {
                    sum_sq += input_cache_(i, j) * input_cache_(i, j);
                }
                float inv_rms = 1.0f / std::sqrt(sum_sq / d + eps_);

                float dot = 0.0f;  // sum_k g_k * gamma_k * x_k
                for (int j = 0; j < d; j++) {
                    dot += grad_output(i, j) * params_.gamma(0, j) * input_cache_(i, j);
                }
                float coef = dot * inv_rms * inv_rms * inv_rms / d;

                for (int j = 0; j < d; j++) {
                    grad_input(i, j) =
                        inv_rms * params_.gamma(0, j) * grad_output(i, j)
                        - input_cache_(i, j) * coef;
                    #pragma omp atomic
                    grads_.gamma_grad(0, j) +=
                        grad_output(i, j) * input_cache_(i, j) * inv_rms * grad_scale;
                }
            }
            return grad_input;
        }

        // Compute mean and variance for backward pass
        Matrix mean(input_cache_.rows(), 1, 0.0f);
        Matrix var(input_cache_.rows(), 1, 0.0f);
        
        // MSVC: loop vars must be signed int
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(input_cache_.rows()); i++) {
            float sum = 0.0f;
            for (int j = 0; j < static_cast<int>(input_cache_.cols()); j++) {
                sum += input_cache_(i, j);
            }
            mean(i, 0) = sum / input_cache_.cols();
        }
        
        // MSVC: loop vars must be signed int
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(input_cache_.rows()); i++) {
            float sum_sq = 0.0f;
            for (int j = 0; j < static_cast<int>(input_cache_.cols()); j++) {
                float diff = input_cache_(i, j) - mean(i, 0);
                sum_sq += diff * diff;
            }
            var(i, 0) = sum_sq / input_cache_.cols();
        }

        // Compute gradients (MSVC: collapse ignored, loop vars must be signed int)
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < static_cast<int>(input_cache_.rows()); i++) {
            for (int j = 0; j < static_cast<int>(input_cache_.cols()); j++) {
                float inv_std = 1.0f / std::sqrt(var(i, 0) + eps_);
                float normalized = (input_cache_(i, j) - mean(i, 0)) * inv_std;
                
                // Gradient with respect to input
                grad_input(i, j) = params_.gamma(0, j) * grad_output(i, j) * inv_std;
                
                // Accumulate gradients for gamma and beta (normalized by batch size)
                #pragma omp atomic
                grads_.gamma_grad(0, j) += grad_output(i, j) * normalized * grad_scale;
                #pragma omp atomic
                grads_.beta_grad(0, j) += grad_output(i, j) * grad_scale;
            }
        }

        // Log gradient stats (only to file, not stdout)
        GRAD_LOG_MATRIX("layer_norm_gamma_grad", grads_.gamma_grad);
        GRAD_LOG_MATRIX("layer_norm_grad_input", grad_input);

        return grad_input;

    } catch (const std::exception& e) {
        throw std::runtime_error("Error in LayerNorm backward: " + std::string(e.what()));
    }
}

void LayerNorm::save(std::ostream& os) const {
    // Write hidden_size first (load() expects this)
    os.write(reinterpret_cast<const char*>(&hidden_size_), sizeof(hidden_size_));
    os.write(reinterpret_cast<const char*>(params_.gamma.data()), hidden_size_ * sizeof(float));
    os.write(reinterpret_cast<const char*>(params_.beta.data()), hidden_size_ * sizeof(float));
}

std::unique_ptr<LayerNorm> LayerNorm::load(std::istream& is) {
    size_t hidden_size;
    is.read(reinterpret_cast<char*>(&hidden_size), sizeof(hidden_size));
    
    auto ln = std::make_unique<LayerNorm>(hidden_size);
    
    // Read parameters
    is.read(reinterpret_cast<char*>(ln->params_.gamma.data()), hidden_size * sizeof(float));
    is.read(reinterpret_cast<char*>(ln->params_.beta.data()), hidden_size * sizeof(float));
    
    return ln;
}