#include "../include/matrix.hpp"
#include "../include/rng_seed.hpp"
#include <random>
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <cmath>

void Matrix::initialize_random(float scale) {
    if (!owns_data_) {
        throw std::runtime_error("Cannot initialize external data buffer");
    }
    
    std::mt19937 gen(tcpp_seed_source());
    std::uniform_real_distribution<float> dis(-scale, scale);
    
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            data_[i * cols_ + j] = dis(gen);
        }
    }
}

void Matrix::initialize_constant(float value) {
    if (!owns_data_) {
        throw std::runtime_error("Cannot initialize external data buffer");
    }
    
    std::fill(data_.begin(), data_.end(), value);
}

// apply_swish() and column_sum() are implemented in components.cpp to avoid duplicates 