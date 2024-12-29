#pragma once
#include "components.hpp"

class TokenEmbedding {
private:
    Matrix weights;
    size_t vocab_size_;
    size_t hidden_size_;

    friend class cereal::access;
    template<class Archive>
    void serialize(Archive & ar) {
        ar(weights, vocab_size_, hidden_size_);
    }

public:
    TokenEmbedding() = default;
    TokenEmbedding(size_t vocab_size, size_t hidden_size);
    Matrix forward(const std::vector<int>& tokens);
    Matrix project_to_vocab(const Matrix& hidden_states);
    void save(std::ostream& os) const;
    static std::unique_ptr<TokenEmbedding> load(std::istream& is);
    friend class Transformer;
};

class PositionalEncoding {
private:
    Matrix encoding_matrix;

    friend class cereal::access;
    template<class Archive>
    void serialize(Archive & ar) {
        ar(encoding_matrix);
    }

public:
    PositionalEncoding() = default;
    PositionalEncoding(size_t max_seq_length, size_t hidden_size);
    Matrix forward(const Matrix& position_ids);
    void save(std::ostream& os) const;
    static std::unique_ptr<PositionalEncoding> load(std::istream& is);
}; 