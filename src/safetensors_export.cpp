/**
 * @file safetensors_export.cpp
 * @brief Safetensors exporter. Mirrors the tensor enumeration of
 *        gguf_export.cpp exactly (same names, same [out,in] transposition),
 *        so a model exports identically under both formats.
 */

#include "../include/safetensors_export.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include <nlohmann/json.hpp>

#include "../include/transformer.hpp"
#include "../include/tiktoken_tokenizer.hpp"
#include "../include/layer_norm.hpp"

namespace safetensors_export {

namespace {

struct STTensor {
    std::string name;
    std::vector<uint64_t> shape;
    const float* data = nullptr;                     // non-owning view, or...
    std::unique_ptr<std::vector<float>> owned;       // ...owning (transposed copies)
    size_t num_elements = 0;

    const float* ptr() const { return owned ? owned->data() : data; }
};

}  // namespace

bool export_to_safetensors(
    const Transformer& transformer,
    const TiktokenTokenizer& tokenizer,
    const std::string& output_path,
    const SafetensorsExportConfig& config
) {
    std::cout << "Safetensors Export: Starting export to " << output_path << std::endl;

    const auto& model_config = transformer.getConfig();
    std::vector<STTensor> tensors;

    // Projection weights are stored row-major [in, out] (y = x @ W); export
    // transposed to [out, in], matching the GGUF exporter and torch's
    // nn.Linear convention.
    auto add_transposed = [&](const Matrix& w, const std::string& name) {
        STTensor t;
        t.name = name;
        const size_t R = w.rows(), C = w.cols();
        t.owned = std::make_unique<std::vector<float>>(R * C);
        for (size_t i = 0; i < R; ++i) {
            for (size_t j = 0; j < C; ++j) {
                (*t.owned)[j * R + i] = w(i, j);     // [out=C][in=R]
            }
        }
        t.shape = {static_cast<uint64_t>(C), static_cast<uint64_t>(R)};
        t.num_elements = R * C;
        tensors.push_back(std::move(t));
    };

    auto add_norm = [&](const LayerNorm* norm, const std::string& name) {
        if (!norm) return;
        const Matrix& gamma = norm->get_gamma();     // [1 x hidden]
        STTensor t;
        t.name = name;
        t.shape = {static_cast<uint64_t>(gamma.cols())};
        t.data = gamma.data();
        t.num_elements = gamma.cols();
        tensors.push_back(std::move(t));
    };

    // Token embeddings: stored [vocab, hidden] row-major; export the first
    // model vocab_size rows only (ids beyond the model vocab were <unk> in
    // training), exactly as the GGUF exporter does.
    const auto* embedding = transformer.getTokenEmbedding();
    if (embedding) {
        const Matrix& emb = embedding->getWeights();
        STTensor t;
        t.name = "token_embd.weight";
        const size_t vocab = std::min(static_cast<size_t>(emb.rows()),
                                      model_config.vocab_size);
        t.shape = {static_cast<uint64_t>(vocab), static_cast<uint64_t>(emb.cols())};
        t.data = emb.data();
        t.num_elements = vocab * emb.cols();
        tensors.push_back(std::move(t));
    }

    const auto& layers = transformer.getLayers();
    for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
        const auto& layer = layers[layer_idx];
        std::string prefix = "blk." + std::to_string(layer_idx) + ".";

        const auto* attention = layer->getAttention();
        if (attention) {
            const auto& ap = attention->parameters();
            add_transposed(ap.query_weights, prefix + "attn_q.weight");
            add_transposed(ap.key_weights, prefix + "attn_k.weight");
            add_transposed(ap.value_weights, prefix + "attn_v.weight");
            add_transposed(ap.output_weights, prefix + "attn_output.weight");
        }
        add_norm(layer->getLayerNorm(), prefix + "attn_norm.weight");

        const auto* ffn = layer->getFeedForward();
        if (ffn) {
            const auto& fp = ffn->parameters();
            add_transposed(fp.gate_proj_weights, prefix + "ffn_gate.weight");
            add_transposed(fp.up_proj_weights, prefix + "ffn_up.weight");
            add_transposed(fp.down_proj_weights, prefix + "ffn_down.weight");
        }
        add_norm(layer->getFfnLayerNorm(), prefix + "ffn_norm.weight");
    }

    add_norm(transformer.getFinalLayerNorm(), "output_norm.weight");

    // LM head: stored [hidden, vocab] -> [vocab, hidden]
    const auto* lm_head = transformer.getLMHead();
    if (lm_head && !config.tie_word_embeddings) {
        add_transposed(lm_head->getWeights(), "output.weight");
    }

    // ---- Build the JSON header (offsets are into the data section) ----
    nlohmann::json header;
    header["__metadata__"] = {
        {"format", "transformer_cpp"},
        {"model_name", config.model_name},
        {"architecture", "llama"},
        {"vocab_size", std::to_string(model_config.vocab_size)},
        {"hidden_size", std::to_string(model_config.hidden_size)},
        {"num_layers", std::to_string(model_config.num_layers)},
        {"num_heads", std::to_string(model_config.num_heads)},
        {"intermediate_size", std::to_string(model_config.intermediate_size)},
        {"max_seq_length", std::to_string(model_config.max_seq_length)},
        {"layer_norm_epsilon", std::to_string(model_config.layer_norm_epsilon)},
        {"use_rope", model_config.use_rope ? "true" : "false"},
        {"tokenizer", "word_level"},
    };
    uint64_t offset = 0;
    for (const auto& t : tensors) {
        const uint64_t nbytes = static_cast<uint64_t>(t.num_elements) * sizeof(float);
        header[t.name] = {
            {"dtype", "F32"},
            {"shape", t.shape},
            {"data_offsets", {offset, offset + nbytes}},
        };
        offset += nbytes;
    }
    std::string header_str = header.dump();
    // Pad the header with spaces to 8-byte alignment (spec recommendation).
    while (header_str.size() % 8 != 0) header_str.push_back(' ');

    std::ofstream out(output_path, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Safetensors Export: cannot open " << output_path << std::endl;
        return false;
    }
    const uint64_t header_len = header_str.size();
    out.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
    out.write(header_str.data(), static_cast<std::streamsize>(header_str.size()));
    for (const auto& t : tensors) {
        out.write(reinterpret_cast<const char*>(t.ptr()),
                  static_cast<std::streamsize>(t.num_elements * sizeof(float)));
    }
    out.close();
    if (!out) {
        std::cerr << "Safetensors Export: write failed for " << output_path << std::endl;
        return false;
    }
    std::cout << "Safetensors Export: wrote " << tensors.size() << " tensors ("
              << (offset / (1024.0 * 1024.0)) << " MB data) to "
              << output_path << std::endl;

    // ---- Vocabulary sidecar (safetensors carries no tokenizer) ----
    if (config.write_vocab_sidecar) {
        const size_t vocab_size = std::min(tokenizer.vocab_size(),
                                           model_config.vocab_size);
        nlohmann::json vocab;
        vocab["tokenizer"] = "word_level";
        vocab["special_tokens"] = {{"pad", 0}, {"unk", 1}, {"bos", 2}, {"eos", 3}};
        nlohmann::json id_to_token = nlohmann::json::array();
        for (size_t i = 0; i < vocab_size; ++i) {
            id_to_token.push_back(tokenizer.decode({static_cast<int>(i)}));
        }
        vocab["id_to_token"] = std::move(id_to_token);
        std::ofstream vf(output_path + ".vocab.json");
        vf << vocab.dump();
        std::cout << "Safetensors Export: vocab sidecar (" << vocab_size
                  << " tokens) -> " << output_path << ".vocab.json" << std::endl;
    }
    return true;
}

}  // namespace safetensors_export
