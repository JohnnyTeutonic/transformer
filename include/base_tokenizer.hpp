#pragma once
#include <string>
#include <vector>

class BaseTokenizer {
public:
    virtual ~BaseTokenizer() = default;
    virtual std::vector<int> encode(const std::string& text) const = 0;
    virtual std::string decode(const std::vector<int>& tokens) const = 0;
    virtual size_t vocab_size() const = 0;
    virtual void initialize(const std::string& encoding_name) = 0;
    virtual bool is_initialized() const = 0;
    
    // Special token accessors
    virtual int get_pad_token_id() const = 0;
    virtual int get_unk_token_id() const = 0;
    virtual int get_bos_token_id() const = 0;
    virtual int get_eos_token_id() const = 0;
    virtual int get_mask_token_id() const = 0;

    // Configuration
    virtual void set_vocab_size(size_t size) { vocab_size_ = size; }
    size_t get_vocab_size() const { return vocab_size_; }

protected:
    size_t vocab_size_ = 0;
}; 