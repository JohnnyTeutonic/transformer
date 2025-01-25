#include "../include/config.hpp"
#include <iostream>
#include <stdexcept>

TransformerConfig::TransformerConfig(size_t vocab_size, size_t max_seq_length, size_t hidden_size,
                                     size_t num_layers, size_t num_heads, size_t batch_size,
                                     size_t num_epochs)
    : vocab_size(vocab_size), max_seq_length(max_seq_length), hidden_size(hidden_size),
      num_layers(num_layers), num_heads(num_heads), head_dim(hidden_size / num_heads),
      intermediate_size(4 * hidden_size), dropout_prob(0.1f), use_flash_attention(true),
      use_rope(true), use_sliding_window(false), window_size(512), use_gqa(false),
      num_kv_heads(num_heads / 2), use_fp16(false), use_gradient_checkpointing(true),
      memory_pool_size(1024), batch_size(batch_size), num_epochs(num_epochs), dropout_rate(0.1f),
      weight_decay(0.01f), debug_mode(false), log_frequency(100), load_from_checkpoint(false),
      checkpoint_to_load(""),
      paths{
          "models",            // save_directory
          "transformer_model", // model_name
          2                    // checkpoint_frequency
      },
      beam_search{
          true,   // use_beam_search
          5,      // beam_size
          4,      // beams_per_group
          3,      // num_groups
          1.5f,   // length_penalty
          1.0f,   // temperature
          0.9f,   // top_p
          20,     // max_length
          3.0f,   // initial_temperature
          0.8f,   // initial_noise_scale
          4.0f,   // diversity_strength
          100,    // top_k
          0.1f    // token_noise_scale
      },
      tokenizer{
          false,  // use_subword
          vocab_size,  // vocab_size
          "model/tokenizer.model",  // model_path
          {"<unk>", "<s>", "</s>", "<pad>", "<mask>"}  // special_tokens
      } {
    std::cout << "entering TransformerConfig constructor" << std::endl;
    if (hidden_size % num_heads != 0) {
        throw std::invalid_argument("Hidden size must be divisible by number of heads");
    }
    std::cout << "exiting TransformerConfig constructor" << std::endl;
}

// Define as a member function
bool TransformerConfig::operator!=(const TransformerConfig& other) const {
    return vocab_size != other.vocab_size || 
           max_seq_length != other.max_seq_length ||
           hidden_size != other.hidden_size ||
           num_layers != other.num_layers ||
           num_heads != other.num_heads ||
           head_dim != other.head_dim ||
           intermediate_size != other.intermediate_size ||
           dropout_prob != other.dropout_prob ||
           use_flash_attention != other.use_flash_attention ||
           use_rope != other.use_rope ||
           use_sliding_window != other.use_sliding_window ||
           window_size != other.window_size ||
           use_gqa != other.use_gqa ||
           num_kv_heads != other.num_kv_heads ||
           use_fp16 != other.use_fp16 ||
           use_gradient_checkpointing != other.use_gradient_checkpointing ||
           memory_pool_size != other.memory_pool_size ||
           batch_size != other.batch_size ||
           num_epochs != other.num_epochs ||
           dropout_rate != other.dropout_rate ||
           weight_decay != other.weight_decay ||
           debug_mode != other.debug_mode ||
           log_frequency != other.log_frequency;
}

void TransformerConfig::from_json(const nlohmann::json& j) {
    // Model parameters
    vocab_size = j["model"]["vocab_size"];
    hidden_size = j["model"]["hidden_size"];
    num_heads = j["model"]["num_heads"];
    num_layers = j["model"]["num_layers"];
    head_dim = j["model"]["head_dim"];
    intermediate_size = j["model"]["intermediate_size"];
    max_seq_length = j["model"]["max_seq_length"];

    // Training parameters
    batch_size = j["training"]["batch_size"];
    num_epochs = j["training"]["num_epochs"];
    dropout_rate = j["training"]["dropout_rate"];
    weight_decay = j["training"]["weight_decay"];

    // Attention parameters
    use_flash_attention = j["attention"]["use_flash_attention"];
    use_rope = j["attention"]["use_rope"];
    use_sliding_window = j["attention"]["use_sliding_window"];
    window_size = j["attention"]["window_size"];
    use_gqa = j["attention"]["use_gqa"];
    num_kv_heads = j["attention"]["num_kv_heads"];

    // Optimization parameters
    use_fp16 = j["optimization"]["use_fp16"];
    use_gradient_checkpointing = j["optimization"]["use_gradient_checkpointing"];
    memory_pool_size = j["optimization"]["memory_pool_size"];

    // Paths
    paths.save_directory = j["paths"]["save_directory"];
    paths.model_name = j["paths"]["model_name"];
    paths.checkpoint_frequency = j["paths"]["checkpoint_frequency"];

    // Beam search
    beam_search.use_beam_search = j["beam_search"]["use_beam_search"];
    beam_search.beam_size = j["beam_search"]["beam_size"];
    beam_search.beams_per_group = j["beam_search"]["beams_per_group"];
    beam_search.num_groups = j["beam_search"]["num_groups"];
    beam_search.length_penalty = j["beam_search"]["length_penalty"];
    beam_search.temperature = j["beam_search"]["temperature"];
    beam_search.initial_temperature = j["beam_search"]["initial_temperature"];
    beam_search.diversity_strength = j["beam_search"]["diversity_strength"];
    beam_search.top_k = j["beam_search"]["top_k"];
    beam_search.top_p = j["beam_search"]["top_p"];
    beam_search.max_length = j["beam_search"]["max_length"];
    beam_search.initial_noise_scale = j["beam_search"]["initial_noise_scale"];
    beam_search.token_noise_scale = j["beam_search"]["token_noise_scale"];

    // Tokenizer
    tokenizer.use_subword = j["tokenizer"]["use_subword"];
    tokenizer.vocab_size = j["tokenizer"]["vocab_size"];
    tokenizer.model_path = j["tokenizer"]["model_path"];
    tokenizer.special_tokens = j["tokenizer"]["special_tokens"].get<std::vector<std::string>>();

    // Debug settings
    if (j.contains("debug")) {
        const auto& debug = j["debug"];
        debug_mode = debug.value("verbose_logging", false);
        log_frequency = debug.value("log_frequency", 100);
    }

    // Checkpoint settings
    load_from_checkpoint = j.value("load_from_checkpoint", false);
    checkpoint_to_load = j.value("checkpoint_to_load", "");
}