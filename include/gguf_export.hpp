#pragma once

/**
 * @file gguf_export.hpp
 * @brief GGUF (GPT-Generated Unified Format) exporter for transformer_cpp models
 * 
 * Exports trained models to GGUF format for inference with tinyllama.cpp
 * and other GGUF-compatible inference engines.
 */

#include <string>
#include <vector>
#include <cstdint>
#include <fstream>
#include <map>

// Forward declarations
class Transformer;
class TiktokenTokenizer;

namespace gguf_export {

// GGUF constants
constexpr uint32_t GGUF_MAGIC = 0x46554747;  // "GGUF" in little-endian
constexpr uint32_t GGUF_VERSION = 3;
constexpr uint64_t GGUF_DEFAULT_ALIGNMENT = 32;

// GGML tensor types
enum class GGMLType : uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    BF16 = 30
};

// GGUF value types for metadata
enum class GGUFValueType : uint32_t {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12
};

/**
 * @brief Configuration for GGUF export
 */
struct GGUFExportConfig {
    std::string model_name = "transformer_cpp_model";
    std::string architecture = "llama";  // For compatibility with llama.cpp ecosystem
    GGMLType weight_type = GGMLType::F32;  // F32 for full precision, F16 for half
    bool include_tokenizer = true;
    bool tie_word_embeddings = false;  // Whether lm_head shares weights with embeddings
};

/**
 * @brief Tensor information for GGUF export
 */
struct TensorExportInfo {
    std::string name;
    std::vector<uint64_t> shape;
    GGMLType type;
    const float* data;
    size_t num_elements;
};

/**
 * @brief GGUF file writer
 */
class GGUFWriter {
public:
    explicit GGUFWriter(const std::string& filename);
    ~GGUFWriter();
    
    // Metadata writing
    void write_metadata_string(const std::string& key, const std::string& value);
    void write_metadata_uint32(const std::string& key, uint32_t value);
    void write_metadata_int32(const std::string& key, int32_t value);
    void write_metadata_float32(const std::string& key, float value);
    void write_metadata_bool(const std::string& key, bool value);
    void write_metadata_string_array(const std::string& key, const std::vector<std::string>& values);
    void write_metadata_float_array(const std::string& key, const std::vector<float>& values);
    
    // Tensor writing
    void add_tensor(const TensorExportInfo& tensor);
    
    // Finalize and write to disk
    bool finalize();
    
private:
    std::string filename_;
    std::ofstream file_;
    
    // Buffered data
    std::vector<std::pair<std::string, std::vector<uint8_t>>> metadata_;
    std::vector<TensorExportInfo> tensors_;
    
    void write_string(const std::string& str);
    void write_header();
    void write_all_metadata();
    void write_tensor_infos();
    void write_tensor_data();
    
    template<typename T>
    void write_raw(const T& value) {
        file_.write(reinterpret_cast<const char*>(&value), sizeof(T));
    }
};

/**
 * @brief Export a Transformer model to GGUF format
 * 
 * @param transformer The trained transformer model
 * @param tokenizer The tokenizer (for vocabulary export)
 * @param output_path Path for the output .gguf file
 * @param config Export configuration
 * @return true if export succeeded
 */
bool export_to_gguf(
    const Transformer& transformer,
    const TiktokenTokenizer& tokenizer,
    const std::string& output_path,
    const GGUFExportConfig& config = GGUFExportConfig()
);

/**
 * @brief Get size in bytes for a GGML type
 */
size_t ggml_type_size(GGMLType type);

/**
 * @brief Convert tensor name from transformer_cpp convention to GGUF/llama convention
 * 
 * transformer_cpp: layer_0_attention_q_proj
 * GGUF:           blk.0.attn_q.weight
 */
std::string convert_tensor_name(const std::string& internal_name, int layer_idx = -1);

} // namespace gguf_export
