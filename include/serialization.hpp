#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

// Forward declaration to avoid circular dependency
namespace p2p {
    struct NetworkMessage;
    struct ModelStateChunk;
}
class Matrix;

namespace serialization {

// Helper to check for trivial copyability
template<typename T>
concept TriviallyCopyable = std::is_trivially_copyable_v<T>;

class Serializer {
public:
    Serializer() = default;

    void write_uint8(uint8_t value);
    void write_uint16(uint16_t value);
    void write_uint32(uint32_t value);
    void write_uint64(uint64_t value);
    void write_string(const std::string& str);
    void write_bytes(const std::vector<uint8_t>& bytes);

    template<TriviallyCopyable T>
    void write_trivial(const T& value) {
        const uint8_t* begin = reinterpret_cast<const uint8_t*>(&value);
        buffer_.insert(buffer_.end(), begin, begin + sizeof(T));
    }

    const std::vector<uint8_t>& get_buffer() const;
    std::vector<uint8_t>&& take_buffer();

private:
    std::vector<uint8_t> buffer_;
};

class Deserializer {
public:
    Deserializer(const std::vector<uint8_t>& buffer);
    Deserializer(std::vector<uint8_t>&& buffer);

    uint8_t read_uint8();
    uint16_t read_uint16();
    uint32_t read_uint32();
    uint64_t read_uint64();
    std::string read_string();
    std::vector<uint8_t> read_bytes();
    
    template<TriviallyCopyable T>
    T read_trivial() {
        check_bounds(sizeof(T));
        T value;
        std::memcpy(&value, buffer_.data() + offset_, sizeof(T));
        offset_ += sizeof(T);
        return value;
    }

    bool has_more() const;

private:
    std::vector<uint8_t> buffer_data_;
    const std::vector<uint8_t>& buffer_;
    size_t offset_ = 0;
    
    void check_bounds(size_t size);
};

// Functions to serialize/deserialize the NetworkMessage
std::vector<uint8_t> serialize(const p2p::NetworkMessage& message);
p2p::NetworkMessage deserialize(const std::vector<uint8_t>& data);

// Serialization for state synchronization
std::vector<uint8_t> serialize_chunk(const p2p::ModelStateChunk& chunk);
p2p::ModelStateChunk deserialize_chunk(const std::vector<uint8_t>& data);
std::vector<uint8_t> serialize_matrix(const Matrix& matrix);
Matrix deserialize_matrix(const std::vector<uint8_t>& data);

} // namespace serialization
