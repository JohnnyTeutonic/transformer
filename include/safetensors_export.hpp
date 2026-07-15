#pragma once

/**
 * @file safetensors_export.hpp
 * @brief Export a trained Transformer to the safetensors format.
 *
 * Safetensors layout: 8-byte little-endian u64 header length, then a JSON
 * header mapping tensor names to {dtype, shape, data_offsets}, then the raw
 * tensor bytes (offsets relative to the start of the data section).
 *
 * Tensor naming and layout follow the GGUF exporter exactly (llama.cpp-style
 * names; projection weights transposed to [out, in]) so the two exporters
 * describe the same model identically. Model hyperparameters are recorded in
 * the header's __metadata__ block, and the word-level vocabulary is written
 * to a sidecar <path>.vocab.json so the export is loadable end to end.
 */

#include <string>

class Transformer;
class TiktokenTokenizer;

namespace safetensors_export {

struct SafetensorsExportConfig {
    std::string model_name = "transformer_cpp";
    bool tie_word_embeddings = false;  // matches GGUFExportConfig default
    bool write_vocab_sidecar = true;
};

bool export_to_safetensors(
    const Transformer& transformer,
    const TiktokenTokenizer& tokenizer,
    const std::string& output_path,
    const SafetensorsExportConfig& config = SafetensorsExportConfig());

}  // namespace safetensors_export
