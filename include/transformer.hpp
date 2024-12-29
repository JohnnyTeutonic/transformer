#pragma once
#include "components.hpp"
#include "attention.hpp"
#include "layernorm.hpp"
#include "tokenizer.hpp"
#include "cache.hpp"
#include "embeddings.hpp"
#include "feed_forward.hpp"
#include "cuda_manager.hpp"
#include <memory>
#include <vector>
#include <optional>
#include <cereal/access.hpp>

class TransformerConfig {
public:
    size_t vocab_size;
    size_t max_seq_length;
    size_t hidden_size;
    size_t num_layers;
    size_t num_heads;
    size_t head_dim;
    size_t intermediate_size;
    float dropout_prob;
    bool use_flash_attention;
    bool use_rope;
    bool use_sliding_window;
    size_t window_size;
    bool use_gqa;
    size_t num_kv_heads;
    bool use_cuda;
    
    TransformerConfig(size_t vocab_size = 50000,
                     size_t max_seq_length = 2048,
                     size_t hidden_size = 768,
                     size_t num_layers = 12,
                     size_t num_heads = 12);
    
    template<class Archive>
    void serialize(Archive & ar) {
        ar(vocab_size, max_seq_length, hidden_size, num_layers, num_heads,
           head_dim, intermediate_size, dropout_prob, use_flash_attention,
           use_rope, use_sliding_window, window_size, use_gqa, num_kv_heads,
           use_cuda);
    }
};

class TransformerLayer {
private:
    std::unique_ptr<MultiHeadAttention> self_attention;
    std::unique_ptr<LayerNorm> attention_ln;
    std::unique_ptr<FeedForward> feed_forward;
    std::unique_ptr<LayerNorm> ffn_ln;
    KVCache kv_cache;
    TransformerConfig config;
    
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive & ar) {
        ar(self_attention, attention_ln, feed_forward, ffn_ln, config);
    }

public:
    TransformerLayer() = default;
    
    explicit TransformerLayer(const TransformerConfig& config);
    Matrix forward(const Matrix& x, const AttentionMask& mask = {});
    void clear_cache();
    void save(std::ostream& os) const;
    static std::unique_ptr<TransformerLayer> load(std::istream& is);
    Matrix backward(const Matrix& grad, const Matrix& input) const;
    Matrix backward_cuda(const Matrix& grad, const Matrix& input) const;
    friend class Transformer;
    virtual ~TransformerLayer() = default;
};

class Transformer {
private:
    std::vector<std::unique_ptr<TransformerLayer>> layers;
    std::unique_ptr<TokenEmbedding> token_embedding;
    std::unique_ptr<PositionalEncoding> pos_encoding;
    std::unique_ptr<LayerNorm> final_ln;
    TransformerConfig config;
    
#ifdef USE_CUDA
    std::unique_ptr<CudaManager> cuda_manager;
#endif

    // Add back the helper methods
    Matrix compute_loss_gradients(const Matrix& logits, const std::vector<int>& targets);
    void backward_pass(const std::vector<Matrix>& activations, const Matrix& loss_grad);
    void update_parameters(float learning_rate);

    friend class cereal::access;
    friend void save_model(const std::string& path, const Transformer& model);
    friend void load_model(const std::string& path, Transformer& model);
    
    template<class Archive>
    void serialize(Archive & ar) {
        ar(layers, token_embedding, pos_encoding, final_ln, config);
    }

public:
    Transformer() = default;
    
    explicit Transformer(const TransformerConfig& config);
    
    // Forward pass
    Matrix forward(const std::vector<int>& input_tokens, bool use_cache = false);
    Matrix forward_cuda(const std::vector<int>& input_tokens, bool use_cache = false);
    
    // Training
    void train(const std::vector<std::vector<int>>& input_tokens,
              const std::vector<std::vector<int>>& target_tokens,
              size_t num_epochs,
              float learning_rate);
    
    // Serialization
    void save_model(const std::string& path) const;
    static Transformer load_model(const std::string& path);
    
    // Cache management
    void clear_kv_cache();
    
    // Backward pass
    Matrix backward(const Matrix& grad, const Matrix& activation, size_t layer_idx);
    Matrix backward_cuda(const Matrix& grad, const Matrix& activation, size_t layer_idx);
    
    // Parameter access
    std::vector<Matrix>& parameters();
    
    // Serialization helpers
    void save(std::ostream& os) const;
    void load(std::istream& is);
}; 