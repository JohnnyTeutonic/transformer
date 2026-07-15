#pragma once

/**
 * @file architecture.hpp
 * @brief Declarative architecture specification ("IaC for the transformer").
 *
 * A user selects a FAMILY ("llama", "vanilla") or declares a CUSTOM
 * combination of the primitives this codebase natively implements:
 *
 *     norm       : rmsnorm | layernorm
 *     positions  : rope | sinusoidal        (additive, applied at embedding)
 *     ffn        : swiglu                   (gelu: planned, not yet native)
 *     attention  : mha | gqa
 *     biases     : true | false
 *
 * The spec VALIDATES the combination (unsupported primitives fail fast with
 * the list of natives) and APPLIES it onto TransformerConfig, replacing the
 * old monolithic `llama_mode` flag, which coupled norm choice and bias
 * freezing. JSON form (config file "architecture" block or a standalone file):
 *
 *     "architecture": {
 *         "family": "llama",                       // or "vanilla" | "custom"
 *         "overrides": { "norm": "layernorm", "attention": "gqa",
 *                        "num_kv_heads": 4, "biases": true }
 *     }
 *
 * Inference-side compatibility: tinyllama.cpp (the companion GGUF engine)
 * natively executes rmsnorm + rope + no-bias models with swiglu or
 * gelu/geglu activations. tinyllama_compatible() reports whether a spec's
 * GGUF export will run there, and why not otherwise.
 */

#include <string>

#include <nlohmann/json_fwd.hpp>

struct TransformerConfig;

namespace arch {

enum class Norm { RMSNorm, LayerNorm };
enum class Positions { RoPE, Sinusoidal };
enum class FFN { SwiGLU };
enum class Attention { MHA, GQA };

struct ArchitectureSpec {
    std::string family = "llama";
    Norm norm = Norm::RMSNorm;
    Positions positions = Positions::RoPE;
    FFN ffn = FFN::SwiGLU;
    Attention attention = Attention::MHA;
    bool biases = false;
    size_t num_kv_heads = 0;   // GQA only; 0 = same as num_heads

    // Sparse attention pattern. "sliding_window" (Mistral-style local
    // attention) is implemented and selectable here, but OFF in every preset:
    // at short training contexts it buys nothing, so it is config-available
    // rather than default-on. window_size is in tokens.
    bool sliding_window = false;
    size_t window_size = 0;

    /** Preset for a named family ("llama" | "vanilla"). Throws on unknown. */
    static ArchitectureSpec from_family(const std::string& family);

    /** Parse {"family": ..., "overrides": {...}} (or flat fields). */
    static ArchitectureSpec from_json(const nlohmann::json& j);

    /** Throws std::runtime_error listing supported natives on bad combos. */
    void validate() const;

    /** Map onto TransformerConfig (use_rms_norm / use_rope / use_gqa /
     *  use_biases and the derived legacy llama_mode). */
    void apply(TransformerConfig& cfg) const;

    /** Can tinyllama.cpp execute a GGUF export of this spec? If not,
     *  `reason` explains which primitive its runtime lacks. */
    bool tinyllama_compatible(std::string* reason = nullptr) const;

    std::string describe() const;
};

}  // namespace arch
