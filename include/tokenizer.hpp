#pragma once
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

class Tokenizer {
public:
    // Default constructor
    Tokenizer() = default;
    virtual ~Tokenizer() = default;

    // Pure virtual functions
    virtual std::vector<int> encode(const std::string& text) const = 0;
    virtual std::string decode(const std::vector<int>& tokens) const = 0;
    virtual void preprocess_text(std::string& text) const = 0;
    virtual bool is_special_token(int token_id) const = 0;

    // Virtual functions with default implementation
    virtual int get_pad_token_id() const = 0;
    virtual int get_unk_token_id() const = 0;
    virtual int get_bos_token_id() const = 0;
    virtual int get_eos_token_id() const = 0;
    virtual int get_mask_token_id() const = 0;
    virtual size_t vocab_size() const = 0;
    virtual bool is_initialized() const = 0;
    virtual void initialize() = 0;
    virtual void print_vocabulary_mappings() const = 0;

    // Non-virtual functions
    void save(std::ostream& os) const;
    static std::unique_ptr<Tokenizer> load(std::istream& is);

    // Special character mapping for preprocessing
    static const std::unordered_map<char, std::string> SPECIAL_CHAR_MAP;

    /**
     * @brief Checks if a token is a noun
     * @param token Token string to check
     * @return True if the token is a noun
     */
    bool is_noun(const std::string& token) const;

    /**
     * @brief Checks if a token is an adjective
     * @param token Token string to check
     * @return True if the token is an adjective
     */
    bool is_adjective(const std::string& token) const;

    /**
     * @brief Checks if a token is a determiner
     * @param token Token string to check
     * @return True if the token is a determiner
     */
    bool is_determiner(const std::string& token) const;

    std::vector<std::string> get_vocabulary_vector() const;

protected:
    virtual void save_vocabulary(std::ostream& os) const = 0;
};