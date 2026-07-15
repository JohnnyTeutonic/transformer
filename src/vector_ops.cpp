#include "../include/vector.hpp"
#include "../include/matrix.hpp"
#include <random>

// Constructor implementations
Vector::Vector() : size_(0) {}

Vector::Vector(size_t size, float default_value) : data_(size, default_value), size_(size) {}

Vector::Vector(const std::initializer_list<float>& list) : data_(list), size_(list.size()) {}

Vector::Vector(const Matrix& matrix) : data_(matrix.data(), matrix.data() + matrix.size()), size_(matrix.size()) {
    // MSVC: loop vars must be signed int
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(matrix.size()); ++i) {
        data_[i] = matrix.at(i / matrix.cols(), i % matrix.cols());
    }
}

// Member function implementations
void Vector::resize(size_t new_size) {
    data_.resize(new_size);
    size_ = new_size;
}

void Vector::fill(float value) {
    // MSVC: loop vars must be signed int
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(size_); ++i) {
        data_[i] = value;
    }
}

void Vector::randomize(float min_val, float max_val) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    // MSVC: loop vars must be signed int
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(size_); ++i) {
        float val = dis(gen);  // Note: Each thread gets its own random value
        data_[i] = val;
    }
}

void Vector::initialize_random(float scale) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0f, scale);
    // MSVC: loop vars must be signed int
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(size_); ++i) {
        float val = dis(gen);  // Note: Each thread gets its own random value
        data_[i] = val;
    }
}

void Vector::initialize_constant(float value) {
    // MSVC: loop vars must be signed int
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(size_); ++i) {
        data_[i] = value;
    }
}

Vector& Vector::operator+=(const Vector& other) {
    if (size_ != other.size()) {
        throw std::invalid_argument("Vector dimensions must match for addition");
    }
    // MSVC: loop vars must be signed int
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(size_); ++i) {
        data_[i] += other.data_[i];
    }
    return *this;
}

void Vector::save(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(&size_), sizeof(size_));
    os.write(reinterpret_cast<const char*>(data_.data()), size_ * sizeof(float));
}

Vector Vector::load(std::istream& is) {
    size_t size;
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    Vector vec(size);
    is.read(reinterpret_cast<char*>(vec.data_.data()), size * sizeof(float));
    return vec;
}

// Non-member operator implementations
Vector operator+(const Vector& a, const Vector& b) {
    Vector result = a;
    result += b;
    return result;
}

Vector operator-(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector dimensions must match for subtraction");
    }
    Vector result(a.size());
    // MSVC: loop vars must be signed int
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(a.size()); ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}

Vector operator*(const Vector& v, float scalar) {
    Vector result(v.size());
    // MSVC: loop vars must be signed int
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(v.size()); ++i) {
        result[i] = v[i] * scalar;
    }
    return result;
}

Vector operator*(float scalar, const Vector& v) {
    return v * scalar;
}

