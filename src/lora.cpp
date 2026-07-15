/**
 * @file lora.cpp
 * @brief LoRA adapters via gradient projection (see lora.hpp for the design).
 */

#include "../include/lora.hpp"

#include <cmath>
#include <random>

namespace lora {

LoRASettings& settings() {
    static LoRASettings s;
    return s;
}

void LoRAAdapter::attach(const Matrix& W, size_t rank, float alpha, unsigned seed) {
    const size_t in = W.rows(), out = W.cols();
    rank_ = rank;
    scale_ = alpha / static_cast<float>(rank);
    W0_ = W;                                  // frozen base snapshot
    A_ = Matrix(in, rank, 0.0f);
    B_ = Matrix(rank, out, 0.0f);             // B = 0 -> W_eff == W0 at start
    mA_ = Matrix(in, rank, 0.0f);
    mB_ = Matrix(rank, out, 0.0f);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(static_cast<float>(in)));
    for (size_t i = 0; i < in; ++i) {
        for (size_t r = 0; r < rank; ++r) {
            A_(i, r) = dist(gen);
        }
    }
    attached_ = true;
}

namespace {

// Per-matrix global-norm clip + SGD-momentum step, matching the trainer's
// existing update rule so LoRA and full fine-tuning share dynamics.
void momentum_step(Matrix& param, const Matrix& grad, Matrix& m, float lr) {
    const float beta = 0.9f, clip_threshold = 1.0f;
    float norm_sq = 0.0f;
    for (size_t i = 0; i < grad.rows(); ++i)
        for (size_t j = 0; j < grad.cols(); ++j)
            norm_sq += grad(i, j) * grad(i, j);
    const float norm = std::sqrt(norm_sq);
    const float clip = (norm > clip_threshold) ? (clip_threshold / (norm + 1e-8f)) : 1.0f;
    for (size_t i = 0; i < param.rows(); ++i) {
        for (size_t j = 0; j < param.cols(); ++j) {
            const float g = grad(i, j) * clip;
            m(i, j) = beta * m(i, j) + (1.0f - beta) * g;
            param(i, j) -= lr * m(i, j);
        }
    }
}

}  // namespace

void LoRAAdapter::update(Matrix& W, const Matrix& dW, float learning_rate) {
    if (!attached_) return;
    const size_t in = W.rows(), out = W.cols(), r = rank_;

    // dA = scale * dW @ B^T   [in, out] x [out, r] -> [in, r]
    Matrix dA(in, r, 0.0f);
    for (size_t i = 0; i < in; ++i) {
        for (size_t k = 0; k < r; ++k) {
            float acc = 0.0f;
            for (size_t j = 0; j < out; ++j) acc += dW(i, j) * B_(k, j);
            dA(i, k) = scale_ * acc;
        }
    }
    // dB = scale * A^T @ dW   [r, in] x [in, out] -> [r, out]
    Matrix dB(r, out, 0.0f);
    for (size_t k = 0; k < r; ++k) {
        for (size_t j = 0; j < out; ++j) {
            float acc = 0.0f;
            for (size_t i = 0; i < in; ++i) acc += A_(i, k) * dW(i, j);
            dB(k, j) = scale_ * acc;
        }
    }
    momentum_step(A_, dA, mA_, learning_rate);
    momentum_step(B_, dB, mB_, learning_rate);

    // Recompose: W = W0 + scale * A @ B
    for (size_t i = 0; i < in; ++i) {
        for (size_t j = 0; j < out; ++j) {
            float acc = 0.0f;
            for (size_t k = 0; k < r; ++k) acc += A_(i, k) * B_(k, j);
            W(i, j) = W0_(i, j) + scale_ * acc;
        }
    }
}

}  // namespace lora
