#include "../include/serialization.hpp"
#include "../include/p2p_network.hpp"
#include <cstring> // For memcpy
#include <stdexcept>

#ifdef _WIN32
#include <winsock2.h>
#else
#include <arpa/inet.h>
#endif


namespace serialization {

// --- Serializer Implementation ---

void Serializer::write_uint8(uint8_t value) {
    buffer_.push_back(value);
}

void Serializer::write_uint16(uint16_t value) {
    uint16_t net_value = htons(value);
    const uint8_t* begin = reinterpret_cast<const uint8_t*>(&net_value);
    buffer_.insert(buffer_.end(), begin, begin + sizeof(net_value));
}

void Serializer::write_uint32(uint32_t value) {
    uint32_t net_value = htonl(value);
    const uint8_t* begin = reinterpret_cast<const uint8_t*>(&net_value);
    buffer_.insert(buffer_.end(), begin, begin + sizeof(net_value));
}

void Serializer::write_uint64(uint64_t value) {
    // htobe64 might not be available on all platforms, do manual conversion
    uint64_t net_value;
    uint32_t high = htonl(static_cast<uint32_t>(value >> 32));
    uint32_t low = htonl(static_cast<uint32_t>(value & 0xFFFFFFFFLL));
    net_value = (static_cast<uint64_t>(low) << 32) | high;
    const uint8_t* begin = reinterpret_cast<const uint8_t*>(&net_value);
    buffer_.insert(buffer_.end(), begin, begin + sizeof(net_value));
}

void Serializer::write_string(const std::string& str) {
    if (str.length() > UINT32_MAX) {
        throw std::runtime_error("String size exceeds maximum limit for serialization.");
    }
    write_uint32(static_cast<uint32_t>(str.length()));
    buffer_.insert(buffer_.end(), str.begin(), str.end());
}

void Serializer::write_bytes(const std::vector<uint8_t>& bytes) {
    if (bytes.size() > UINT32_MAX) {
        throw std::runtime_error("Byte vector size exceeds maximum limit for serialization.");
    }
    write_uint32(static_cast<uint32_t>(bytes.size()));
    buffer_.insert(buffer_.end(), bytes.begin(), bytes.end());
}

const std::vector<uint8_t>& Serializer::get_buffer() const {
    return buffer_;
}

std::vector<uint8_t>&& Serializer::take_buffer() {
    return std::move(buffer_);
}


// --- Deserializer Implementation ---

Deserializer::Deserializer(const std::vector<uint8_t>& buffer) 
    : buffer_(buffer) {}

Deserializer::Deserializer(std::vector<uint8_t>&& buffer)
    : buffer_data_(std::move(buffer)), buffer_(buffer_data_) {}


void Deserializer::check_bounds(size_t size) {
    if (offset_ + size > buffer_.size()) {
        throw std::runtime_error("Deserialization error: read out of bounds.");
    }
}

uint8_t Deserializer::read_uint8() {
    check_bounds(1);
    return buffer_[offset_++];
}

uint16_t Deserializer::read_uint16() {
    check_bounds(sizeof(uint16_t));
    uint16_t net_value;
    std::memcpy(&net_value, buffer_.data() + offset_, sizeof(net_value));
    offset_ += sizeof(net_value);
    return ntohs(net_value);
}

uint32_t Deserializer::read_uint32() {
    check_bounds(sizeof(uint32_t));
    uint32_t net_value;
    std::memcpy(&net_value, buffer_.data() + offset_, sizeof(net_value));
    offset_ += sizeof(net_value);
    return ntohl(net_value);
}

uint64_t Deserializer::read_uint64() {
    check_bounds(sizeof(uint64_t));
    uint64_t net_value;
    std::memcpy(&net_value, buffer_.data() + offset_, sizeof(net_value));
    offset_ += sizeof(net_value);
    
    // Manual byte swap for 64-bit
    uint32_t high = static_cast<uint32_t>(net_value >> 32);
    uint32_t low = static_cast<uint32_t>(net_value & 0xFFFFFFFFLL);
    high = ntohl(high);
    low = ntohl(low);
    return (static_cast<uint64_t>(high) << 32) | low;
}

std::string Deserializer::read_string() {
    uint32_t len = read_uint32();
    check_bounds(len);
    std::string str(buffer_.begin() + offset_, buffer_.begin() + offset_ + len);
    offset_ += len;
    return str;
}

std::vector<uint8_t> Deserializer::read_bytes() {
    uint32_t len = read_uint32();
    check_bounds(len);
    std::vector<uint8_t> bytes(buffer_.begin() + offset_, buffer_.begin() + offset_ + len);
    offset_ += len;
    return bytes;
}

bool Deserializer::has_more() const {
    return offset_ < buffer_.size();
}

// --- NetworkMessage Serialization ---

std::vector<uint8_t> serialize(const p2p::NetworkMessage& message) {
    Serializer s;
    s.write_uint8(static_cast<uint8_t>(message.type));
    s.write_string(message.sender_id);
    s.write_string(message.recipient_id);
    s.write_uint64(message.sequence_number);
    s.write_uint64(message.timestamp);
    s.write_bytes(message.payload);
    s.write_string(message.signature);
    return s.take_buffer();
}

p2p::NetworkMessage deserialize(const std::vector<uint8_t>& data) {
    Deserializer d(data);
    p2p::NetworkMessage message;
    message.type = static_cast<p2p::MessageType>(d.read_uint8());
    message.sender_id = d.read_string();
    message.recipient_id = d.read_string();
    message.sequence_number = d.read_uint64();
    message.timestamp = d.read_uint64();
    message.payload = d.read_bytes();
    message.signature = d.read_string();
    return message;
}

std::vector<uint8_t> serialize_chunk(const p2p::ModelStateChunk& chunk) {
    Serializer s;
    s.write_string(chunk.parameter_name);
    s.write_uint32(chunk.chunk_index);
    s.write_uint32(chunk.total_chunks);
    s.write_bytes(chunk.data);
    return s.take_buffer();
}

p2p::ModelStateChunk deserialize_chunk(const std::vector<uint8_t>& data) {
    Deserializer d(data);
    p2p::ModelStateChunk chunk;
    chunk.parameter_name = d.read_string();
    chunk.chunk_index = d.read_uint32();
    chunk.total_chunks = d.read_uint32();
    chunk.data = d.read_bytes(d.bytes_remaining());
    return chunk;
}

std::vector<uint8_t> serialize_matrix(const Matrix& matrix) {
    Serializer s;
    s.write_uint64(matrix.rows());
    s.write_uint64(matrix.cols());
    for (size_t i = 0; i < matrix.size(); ++i) {
        s.write_trivial(matrix.data()[i]);
    }
    return s.take_buffer();
}

Matrix deserialize_matrix(const std::vector<uint8_t>& data) {
    Deserializer d(data);
    uint64_t rows = d.read_uint64();
    uint64_t cols = d.read_uint64();
    Matrix matrix(rows, cols);
    for (size_t i = 0; i < matrix.size(); ++i) {
        matrix.data()[i] = d.read_trivial<float>();
    }
    return matrix;
}

} // namespace serialization
