#pragma once
#include "matrix.hpp"

/**
 * @brief Key-Value cache for efficient autoregressive generation.
 * 
 * The KVCache struct implements caching of key and value tensors in transformer
 * attention layers, enabling efficient autoregressive generation by avoiding
 * redundant computation of keys and values for previously processed tokens.
 * Features include:
 * - Efficient memory management
 * - Dynamic cache updates
 * - Automatic size handling
 * - Cache invalidation
 */
class KVCache {
public:
    // Constructors
    KVCache() = default;
    explicit KVCache(size_t max_len);  // Add declaration for max_len constructor
    KVCache(const Matrix& key_matrix, const Matrix& value_matrix);
    
    // Accessors
    const Matrix& get_key() const;
    const Matrix& get_value() const;
    
    // Modifiers
    void update(const Matrix& new_key, const Matrix& new_value);
    void clear();
    
    // State checks
    bool empty() const;
    std::pair<Matrix, Matrix> get_cached_kv() const;

private:
    Matrix key_cache;    ///< Cached key tensors from previous positions
    Matrix value_cache;  ///< Cached value tensors from previous positions
};