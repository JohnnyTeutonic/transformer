#include "../include/cache.hpp"

KVCache::KVCache(size_t max_len) {
    key_cache = Matrix();
    value_cache = Matrix();
}

KVCache::KVCache(const Matrix& key_matrix, const Matrix& value_matrix)
    : key_cache(key_matrix), value_cache(value_matrix) {}

const Matrix& KVCache::get_key() const {
    return key_cache;
}

const Matrix& KVCache::get_value() const {
    return value_cache;
}

void KVCache::update(const Matrix& new_key, const Matrix& new_value) {
    key_cache = new_key;
    value_cache = new_value;
}

void KVCache::clear() {
    key_cache = Matrix();
    value_cache = Matrix();
}

bool KVCache::empty() const {
    return key_cache.empty() || value_cache.empty();
}

std::pair<Matrix, Matrix> KVCache::get_cached_kv() const {
    return std::make_pair(key_cache, value_cache);
}