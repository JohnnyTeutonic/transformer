#include "../include/gradient_checkpoint.hpp"
#include <string>

std::unordered_map<size_t, Matrix> GradientCheckpoint::checkpoints;
std::unordered_map<std::string, Matrix> GradientCheckpoint::activation_cache;

void GradientCheckpoint::save_activation(const Matrix& activation, size_t layer) {
    try {
        // Use memory pool for efficient allocation
        Matrix& checkpoint = checkpoints[layer];
        checkpoint = Matrix(activation.rows(), activation.cols());

        if (checkpoint.size() == 0) {
            throw std::runtime_error("Failed to allocate checkpoint matrix for layer " +
                                     std::to_string(layer));
        }

// MSVC: loop vars must be signed int
#pragma omp parallel for collapse(2)
        for (int i = 0; i < static_cast<int>(activation.rows()); ++i) {
            for (int j = 0; j < static_cast<int>(activation.cols()); ++j) {
                checkpoint(i, j) = activation(i, j);
            }
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Error saving activation checkpoint: " + std::string(e.what()));
    }
}

Matrix GradientCheckpoint::get_activation(size_t layer) {
    auto it = checkpoints.find(layer);
    if (it == checkpoints.end()) {
        throw std::runtime_error("No checkpoint found for layer " + std::to_string(layer));
    }
    return it->second;
}

void GradientCheckpoint::cache_activation(const std::string& key, const Matrix& activation) {
    try {
        // Check if we have too many cached activations to prevent memory issues
        if (activation_cache.size() > 1000) { // Arbitrary limit, adjust as needed
            clear_cache();
        }

        // EXPLICIT: Create a temporary Matrix, then move it in
        // This avoids any potential issues with map's [] operator
        Matrix temp(activation.rows(), activation.cols());
        for (size_t i = 0; i < activation.size(); i++) {
            temp.data()[i] = activation.data()[i];
        }
        
        activation_cache[key] = std::move(temp);

        if (activation_cache[key].size() == 0) {
            throw std::runtime_error("Failed to allocate activation cache for key: " + key);
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Error caching activation: " + std::string(e.what()));
    }
}

Matrix GradientCheckpoint::get_activation(const std::string& key) {
    if (!has_activation(key)) {
        throw std::runtime_error("No activation found for key: " + key);
    }
    return activation_cache[key];
}

bool GradientCheckpoint::has_activation(const std::string& key) {
    return activation_cache.find(key) != activation_cache.end();
}

void GradientCheckpoint::clear_cache() {
    activation_cache.clear();
    checkpoints.clear();
}