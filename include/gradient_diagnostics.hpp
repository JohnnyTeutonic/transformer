#pragma once
#include <fstream>
#include <string>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <limits>
#include <algorithm>
#include "matrix.hpp"
#include "vector.hpp"
#include "types.hpp"

/**
 * Gradient flow diagnostic system for debugging training instability.
 * Logs gradient norms, parameter norms, and update magnitudes at each step.
 */
class GradientDiagnostics {
public:
    static GradientDiagnostics& instance() {
        static GradientDiagnostics inst;
        return inst;
    }
    
    void set_step(size_t step) {
        current_step_ = step;
    }
    
    void enable(const std::string& log_path = "gradient_diagnostics.log") {
        enabled_ = true;
        log_file_.open(log_path, std::ios::out | std::ios::trunc);
        log_file_ << "=== GRADIENT FLOW DIAGNOSTICS ===" << std::endl;
        log_file_ << "Format: [step] component | grad_norm | param_norm | update_mag | max_grad | min_grad" << std::endl;
        log_file_ << std::string(80, '=') << std::endl;
    }
    
    void disable() {
        enabled_ = false;
        if (log_file_.is_open()) {
            log_file_.close();
        }
    }
    
    // Log gradient stats for a matrix
    void log_gradient(const std::string& component, const Matrix& grad, 
                      const Matrix* param = nullptr, const Matrix* update = nullptr) {
        if (!enabled_) return;
        
        float grad_norm = compute_norm(grad);
        float max_grad = compute_max(grad);
        float min_grad = compute_min(grad);
        float mean_grad = compute_mean(grad);
        
        log_file_ << "[" << std::setw(4) << current_step_ << "] " 
                  << std::setw(30) << std::left << component << " | "
                  << "grad_norm=" << std::scientific << std::setprecision(3) << grad_norm << " | "
                  << "max=" << std::setprecision(3) << max_grad << " | "
                  << "min=" << std::setprecision(3) << min_grad << " | "
                  << "mean=" << std::setprecision(3) << mean_grad;
        
        if (param) {
            float param_norm = compute_norm(*param);
            log_file_ << " | param_norm=" << std::setprecision(3) << param_norm;
        }
        
        if (update) {
            float update_norm = compute_norm(*update);
            log_file_ << " | update_norm=" << std::setprecision(3) << update_norm;
        }
        
        // Flag potential issues
        if (std::isnan(grad_norm) || std::isinf(grad_norm)) {
            log_file_ << " [!!!NaN/Inf!!!]";
        } else if (grad_norm > 100.0f) {
            log_file_ << " [!EXPLODING!]";
        } else if (grad_norm < 1e-10f && grad_norm > 0) {
            log_file_ << " [!VANISHING!]";
        }
        
        log_file_ << std::endl;
    }
    
    // Log gradient stats for a vector
    void log_gradient(const std::string& component, const FloatVector& grad) {
        if (!enabled_) return;
        
        float grad_norm = 0.0f;
        float max_grad = -std::numeric_limits<float>::infinity();
        float min_grad = std::numeric_limits<float>::infinity();
        float sum = 0.0f;
        
        for (size_t i = 0; i < grad.size(); ++i) {
            grad_norm += grad[i] * grad[i];
            max_grad = std::max(max_grad, grad[i]);
            min_grad = std::min(min_grad, grad[i]);
            sum += grad[i];
        }
        grad_norm = std::sqrt(grad_norm);
        float mean_grad = grad.size() > 0 ? sum / grad.size() : 0.0f;
        
        log_file_ << "[" << std::setw(4) << current_step_ << "] " 
                  << std::setw(30) << std::left << component << " | "
                  << "grad_norm=" << std::scientific << std::setprecision(3) << grad_norm << " | "
                  << "max=" << std::setprecision(3) << max_grad << " | "
                  << "min=" << std::setprecision(3) << min_grad << " | "
                  << "mean=" << std::setprecision(3) << mean_grad << std::endl;
    }
    
    // Log a scalar value (loss, learning rate, etc.)
    void log_scalar(const std::string& name, float value) {
        if (!enabled_) return;
        
        log_file_ << "[" << std::setw(4) << current_step_ << "] " 
                  << std::setw(30) << std::left << name << " = "
                  << std::scientific << std::setprecision(6) << value;
        
        if (std::isnan(value) || std::isinf(value)) {
            log_file_ << " [!!!NaN/Inf!!!]";
        }
        log_file_ << std::endl;
    }
    
    // Log a stage marker for easier reading
    void log_stage(const std::string& stage) {
        if (!enabled_) return;
        log_file_ << "[" << std::setw(4) << current_step_ << "] === " << stage << " ===" << std::endl;
    }
    
    // Flush to ensure all data is written
    void flush() {
        if (log_file_.is_open()) {
            log_file_.flush();
        }
    }
    
private:
    GradientDiagnostics() : enabled_(false), current_step_(0) {}
    
    float compute_norm(const Matrix& m) {
        float norm = 0.0f;
        for (size_t i = 0; i < m.rows(); ++i) {
            for (size_t j = 0; j < m.cols(); ++j) {
                norm += m(i, j) * m(i, j);
            }
        }
        return std::sqrt(norm);
    }
    
    float compute_max(const Matrix& m) {
        float max_val = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < m.rows(); ++i) {
            for (size_t j = 0; j < m.cols(); ++j) {
                max_val = std::max(max_val, m(i, j));
            }
        }
        return max_val;
    }
    
    float compute_min(const Matrix& m) {
        float min_val = std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < m.rows(); ++i) {
            for (size_t j = 0; j < m.cols(); ++j) {
                min_val = std::min(min_val, m(i, j));
            }
        }
        return min_val;
    }
    
    float compute_mean(const Matrix& m) {
        float sum = 0.0f;
        size_t count = m.rows() * m.cols();
        for (size_t i = 0; i < m.rows(); ++i) {
            for (size_t j = 0; j < m.cols(); ++j) {
                sum += m(i, j);
            }
        }
        return count > 0 ? sum / count : 0.0f;
    }
    
    bool enabled_;
    size_t current_step_;
    std::ofstream log_file_;
};

// Convenience macros for easy logging
// DISABLED for performance - enable GRAD_DIAG_ENABLED to turn on
#ifdef GRAD_DIAG_ENABLED
#define GRAD_DIAG GradientDiagnostics::instance()
#define GRAD_LOG_MATRIX(name, grad) GRAD_DIAG.log_gradient(name, grad)
#define GRAD_LOG_MATRIX_FULL(name, grad, param, update) GRAD_DIAG.log_gradient(name, grad, param, update)
#define GRAD_LOG_VECTOR(name, grad) GRAD_DIAG.log_gradient(name, grad)
#define GRAD_LOG_SCALAR(name, value) GRAD_DIAG.log_scalar(name, value)
#define GRAD_LOG_STAGE(stage) GRAD_DIAG.log_stage(stage)
#else
#define GRAD_DIAG GradientDiagnostics::instance()
#define GRAD_LOG_MATRIX(name, grad) ((void)0)
#define GRAD_LOG_MATRIX_FULL(name, grad, param, update) ((void)0)
#define GRAD_LOG_VECTOR(name, grad) ((void)0)
#define GRAD_LOG_SCALAR(name, value) ((void)0)
#define GRAD_LOG_STAGE(stage) ((void)0)
#endif
