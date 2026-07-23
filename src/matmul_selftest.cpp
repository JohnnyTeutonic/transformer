// Definitive test: does the optimized matmul() the trainer uses in every
// projection agree with a naive triple loop? A disagreement here would
// corrupt Q/K/V/O and every FFN matmul identically — the long-suspected
// CPU/GPU-forward divergence (see CHAT_EXPERIMENTS.md Finding 6). Tests the
// exact shapes the attention path uses (T x H) @ (H x H).
#include <cmath>
#include <cstdio>
#include <random>
#include <array>
#include <vector>
#include "../include/matrix.hpp"

static Matrix naive(const Matrix& A, const Matrix& B) {
    Matrix C(A.rows(), B.cols(), 0.0f);
    for (size_t i = 0; i < A.rows(); ++i)
        for (size_t j = 0; j < B.cols(); ++j) {
            float s = 0.f;
            for (size_t k = 0; k < A.cols(); ++k) s += A(i, k) * B(k, j);
            C(i, j) = s;
        }
    return C;
}

static float worst(const Matrix& X, const Matrix& Y) {
    float m = 0.f;
    for (size_t i = 0; i < X.rows(); ++i)
        for (size_t j = 0; j < X.cols(); ++j)
            m = std::max(m, std::abs(X(i, j) - Y(i, j)));
    return m;
}

int main() {
    std::mt19937 rng(0);
    std::normal_distribution<float> nd(0.f, 1.f);
    // Shapes the attention/FFN paths actually exercise at the mid preset.
    int fails = 0;
    for (auto dims : std::vector<std::array<int,3>>{
            {1, 256, 256},    // single-token projection (the golden-batch case)
            {5, 256, 256},    // few-token prompt
            {12, 256, 256},   // seq
            {256, 256, 1024}, // FFN up
            {256, 1024, 256}, // FFN down
            {3, 130, 130}}) { // straddles RECURSIVE_THRESHOLD=128
        int M = dims[0], K = dims[1], N = dims[2];
        Matrix A(M, K), B(K, N);
        for (size_t i = 0; i < A.size(); ++i) A.data()[i] = nd(rng);
        for (size_t i = 0; i < B.size(); ++i) B.data()[i] = nd(rng);
        float d = worst(matmul(A, B), naive(A, B));
        bool ok = d < 1e-3f;
        fails += !ok;
        printf("[MMTEST] %dx%d @ %dx%d  max|delta|=%g  %s\n", M, K, K, N, d,
               ok ? "OK" : "*** MISMATCH ***");
    }
    printf("%s\n", fails ? "MATMUL BROKEN" : "matmul agrees with naive");
    return fails ? 1 : 0;
}
