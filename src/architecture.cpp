/**
 * @file architecture.cpp
 * @brief Architecture spec: family presets, JSON parsing, validation, and
 *        application onto TransformerConfig.
 */

#include "../include/architecture.hpp"

#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "../include/config.hpp"

namespace arch {

namespace {

const char* SUPPORTED =
    "supported natives: norm={rmsnorm, layernorm}, positions={rope, sinusoidal}, "
    "ffn={swiglu} (gelu planned), attention={mha, gqa}, biases={true, false}";

Norm parse_norm(const std::string& s) {
    if (s == "rmsnorm") return Norm::RMSNorm;
    if (s == "layernorm") return Norm::LayerNorm;
    throw std::runtime_error("architecture: unknown norm '" + s + "'; " + SUPPORTED);
}

Positions parse_positions(const std::string& s) {
    if (s == "rope") return Positions::RoPE;
    if (s == "sinusoidal") return Positions::Sinusoidal;
    throw std::runtime_error("architecture: unknown positions '" + s + "'; " + SUPPORTED);
}

FFN parse_ffn(const std::string& s) {
    if (s == "swiglu") return FFN::SwiGLU;
    if (s == "gelu" || s == "geglu" || s == "relu") {
        throw std::runtime_error(
            "architecture: ffn '" + s + "' is not yet implemented in the trainer "
            "(the tinyllama.cpp inference side already dispatches on hidden_act, "
            "so this is a trainer-side gap); " + std::string(SUPPORTED));
    }
    throw std::runtime_error("architecture: unknown ffn '" + s + "'; " + SUPPORTED);
}

Attention parse_attention(const std::string& s) {
    if (s == "mha") return Attention::MHA;
    if (s == "gqa") return Attention::GQA;
    throw std::runtime_error("architecture: unknown attention '" + s + "'; " + SUPPORTED);
}

}  // namespace

ArchitectureSpec ArchitectureSpec::from_family(const std::string& family) {
    ArchitectureSpec s;
    s.family = family;
    if (family == "llama") {
        s.norm = Norm::RMSNorm;
        s.positions = Positions::RoPE;
        s.ffn = FFN::SwiGLU;
        s.attention = Attention::MHA;
        s.biases = false;
    } else if (family == "vanilla") {
        // Classic encoder-decoder-era decoder: LayerNorm, additive sinusoidal
        // positions, biases on. (FFN remains SwiGLU until a GELU MLP lands in
        // the trainer; this is stated rather than silently substituted.)
        s.norm = Norm::LayerNorm;
        s.positions = Positions::Sinusoidal;
        s.ffn = FFN::SwiGLU;
        s.attention = Attention::MHA;
        s.biases = true;
    } else if (family == "custom") {
        // caller sets fields explicitly (from_json overrides)
    } else {
        throw std::runtime_error("architecture: unknown family '" + family +
                                 "' (known: llama, vanilla, custom)");
    }
    return s;
}

ArchitectureSpec ArchitectureSpec::from_json(const nlohmann::json& j) {
    ArchitectureSpec s = from_family(j.value("family", std::string("llama")));
    const nlohmann::json& o = j.contains("overrides") ? j["overrides"] : j;
    if (o.contains("norm"))       s.norm = parse_norm(o["norm"].get<std::string>());
    if (o.contains("positions"))  s.positions = parse_positions(o["positions"].get<std::string>());
    if (o.contains("ffn"))        s.ffn = parse_ffn(o["ffn"].get<std::string>());
    if (o.contains("attention"))  s.attention = parse_attention(o["attention"].get<std::string>());
    if (o.contains("biases"))     s.biases = o["biases"].get<bool>();
    if (o.contains("num_kv_heads")) s.num_kv_heads = o["num_kv_heads"].get<size_t>();
    if (o.contains("sliding_window")) s.sliding_window = o["sliding_window"].get<bool>();
    if (o.contains("window_size"))    s.window_size = o["window_size"].get<size_t>();
    s.validate();
    return s;
}

void ArchitectureSpec::validate() const {
    if (attention == Attention::GQA && num_kv_heads == 0) {
        throw std::runtime_error(
            "architecture: attention=gqa requires overrides.num_kv_heads; " +
            std::string(SUPPORTED));
    }
    if (sliding_window && window_size == 0) {
        throw std::runtime_error(
            "architecture: sliding_window requires overrides.window_size (tokens)");
    }
    // Enum fields are validated at parse time; combinations are all trainable.
}

void ArchitectureSpec::apply(TransformerConfig& cfg) const {
    cfg.use_rms_norm = (norm == Norm::RMSNorm);
    cfg.use_rope = (positions == Positions::RoPE);
    cfg.use_biases = biases;
    cfg.use_gqa = (attention == Attention::GQA);
    cfg.num_kv_heads = (attention == Attention::GQA && num_kv_heads > 0)
                           ? num_kv_heads : cfg.num_heads;
    cfg.use_sliding_window = sliding_window;
    if (sliding_window) cfg.window_size = window_size;
    // Legacy coupled flag, kept consistent for any remaining consumers: the
    // exact LLaMA math is rmsnorm + rope + no biases.
    cfg.llama_mode = (norm == Norm::RMSNorm && positions == Positions::RoPE && !biases);
}

bool ArchitectureSpec::tinyllama_compatible(std::string* reason) const {
    std::ostringstream why;
    if (norm != Norm::RMSNorm) why << "tinyllama.cpp implements RMSNorm only; ";
    if (positions != Positions::RoPE) why << "tinyllama.cpp implements RoPE only; ";
    if (biases) why << "tinyllama.cpp runs bias-free (llama-family) weights; ";
    const std::string s = why.str();
    if (reason) *reason = s;
    return s.empty();
}

std::string ArchitectureSpec::describe() const {
    std::ostringstream os;
    os << "family=" << family
       << " norm=" << (norm == Norm::RMSNorm ? "rmsnorm" : "layernorm")
       << " positions=" << (positions == Positions::RoPE ? "rope" : "sinusoidal")
       << " ffn=swiglu"
       << " attention=" << (attention == Attention::MHA ? "mha" : "gqa")
       << " biases=" << (biases ? "true" : "false");
    if (attention == Attention::GQA) os << " num_kv_heads=" << num_kv_heads;
    if (sliding_window) os << " sliding_window=" << window_size;
    return os.str();
}

}  // namespace arch
