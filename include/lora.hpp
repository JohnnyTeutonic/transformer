#pragma once

/**
 * @file lora.hpp
 * @brief LoRA fine-tuning support (rank-decomposed adapters on the
 *        projection weights), implemented by GRADIENT PROJECTION.
 *
 * Design: for a projection weight W (stored row-major [in, out]) the adapted
 * weight is W_eff = W0 + (alpha/r) * A @ B with A:[in, r], B:[r, out]. The
 * existing forward/backward paths are UNTOUCHED — they keep using the
 * composed matrix. At update time the already-computed full gradient dW is
 * projected onto the factors (exact chain rule through the factorization):
 *
 *      dA = (alpha/r) * dW @ B^T          dB = (alpha/r) * A^T @ dW
 *
 * only A and B receive momentum/updates, and the composed weight is
 * refreshed as W = W0 + (alpha/r) * A @ B. The base W0 (snapshotted when the
 * adapter attaches, i.e. after any --resume checkpoint load) stays frozen.
 *
 * Properties: mathematically exact LoRA gradients; optimizer state only for
 * the factors; forward/CUDA kernels untouched. Known limitation (stated, not
 * hidden): the full dW is still materialized by the existing backward pass,
 * so activation/gradient memory is NOT reduced — this buys parameter-
 * efficient, checkpoint-friendly fine-tuning, not QLoRA-style memory savings.
 * QLoRA (4-bit frozen base + adapters) is the natural phase two: the
 * quantization kernels exist in src/quantization.cpp and tinyllama.cpp
 * already dequantizes q4_k/q6_k/q8_0.
 *
 * Exports: the composed W always carries W0 + (alpha/r)AB, so GGUF and
 * safetensors exports produce MERGED weights with no exporter changes.
 */

#include <string>

#include "components.hpp"   // Matrix

namespace lora {

struct LoRASettings {
    bool enabled = false;
    size_t rank = 8;
    float alpha = 16.0f;
};

/** Global runtime settings (set once in the trainer before construction;
 *  consulted by component update paths to freeze non-adapter parameters). */
LoRASettings& settings();

/** One adapter pair for a single projection weight. */
class LoRAAdapter {
public:
    /** Snapshot W as the frozen base and initialize A (kaiming-ish) and
     *  B (zeros), so W_eff == W0 at attach time (standard LoRA init). */
    void attach(const Matrix& W, size_t rank, float alpha, unsigned seed);

    /** Project dW onto the factors, momentum-update them, and recompose the
     *  the composed weight in place: W = W0 + (alpha/rank) * A @ B.
     *  Gradient clipping matches the trainer's per-matrix global-norm clip. */
    void update(Matrix& W, const Matrix& dW, float learning_rate);

    bool attached() const { return attached_; }

private:
    Matrix W0_, A_, B_, mA_, mB_;
    size_t rank_ = 0;
    float scale_ = 1.0f;
    bool attached_ = false;
};

}  // namespace lora
