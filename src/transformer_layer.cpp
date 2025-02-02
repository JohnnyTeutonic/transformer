TransformerLayer::TransformerLayer(const TransformerConfig& config_, size_t idx)
    : config(config_), layer_idx(idx) {
    // Initialize components
    std::cout << "Initializing TransformerLayer " << idx << " with config:" << std::endl;
    std::cout << "- use_gqa: " << (config.use_gqa ? "true" : "false") << std::endl;
    std::cout << "- num_kv_heads: " << config.num_kv_heads << std::endl;
    std::cout << "- hidden_size: " << config.hidden_size << std::endl;
    std::cout << "- intermediate_size: " << config.intermediate_size << std::endl;
    std::cout << "- layer_norm_epsilon: " << config.layer_norm_epsilon << std::endl;
    std::cout << "- dropout_rate: " << config.dropout_rate << std::endl;

    self_attention = std::make_unique<MultiHeadAttention>(
        config.hidden_size, config.num_heads, config.head_dim, config.dropout_rate,
        config.use_flash_attention, config.use_rope, config.use_sliding_window, config.window_size,
        config.use_gqa, config.num_kv_heads, config.max_seq_length, config.use_fp16);

    attention_ln = std::make_unique<LayerNorm>(config.hidden_size, config.layer_norm_epsilon);
    feed_forward = std::make_unique<FeedForward>(config.hidden_size, config.intermediate_size);
    ffn_ln = std::make_unique<LayerNorm>(config.hidden_size, config.layer_norm_epsilon);

    // Initialize dropout layers
    attention_dropout = std::make_unique<Dropout>(config.dropout_rate);
    ffn_dropout = std::make_unique<Dropout>(config.dropout_rate);
}

Matrix TransformerLayer::forward(const Matrix& input, const AttentionMask& mask,
                                 const std::optional<KVCache>& kv_cache) {
    std::cout << "=== TransformerLayer::forward START ===" << std::endl;
    std::cout << "Input shape: [" << input.rows() << " x " << input.cols() << "]" << std::endl;

    // Layer norm before attention
    std::cout << "\nPre-attention layer normalization..." << std::endl;
    Matrix normalized = attention_ln->forward(input);
    std::cout << "Normalized shape: [" << normalized.rows() << " x " << normalized.cols() << "]" << std::endl;
    
    // Cache the normalized input for attention backward pass
    std::string attn_key = "attn_norm_" + std::to_string(layer_idx);
    GradientCheckpoint::cache_activation(attn_key, normalized);

    // Self attention with dropout
    std::cout << "\nSelf attention..." << std::endl;
    Matrix attention_output = self_attention->forward(normalized, mask, kv_cache);
    std::cout << "Attention output shape: [" << attention_output.rows() << " x " << attention_output.cols() << "]" << std::endl;
    
    if (training) {
        attention_output = attention_dropout->forward(attention_output);
    }
    
    // First residual connection
    Matrix residual = attention_output + input;
    std::cout << "\nPost-attention residual shape: [" << residual.rows() << " x " << residual.cols() << "]" << std::endl;
    
    // Feed forward network with layer norm and dropout
    std::cout << "\nPre-FFN layer normalization..." << std::endl;
    Matrix ffn_normalized = ffn_ln->forward(residual);
    std::cout << "FFN input shape: [" << ffn_normalized.rows() << " x " << ffn_normalized.cols() << "]" << std::endl;
    
    Matrix ffn_output = feed_forward->forward(ffn_normalized);
    std::cout << "FFN output shape: [" << ffn_output.rows() << " x " << ffn_output.cols() << "]" << std::endl;
    
    if (training) {
        ffn_output = ffn_dropout->forward(ffn_output);
    }
    
    // Second residual connection
    Matrix output = ffn_output + residual;
    std::cout << "\nFinal layer output shape: [" << output.rows() << " x " << output.cols() << "]" << std::endl;
    
    std::cout << "=== TransformerLayer::forward END ===" << std::endl;
    return output;
} 