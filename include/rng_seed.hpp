#pragma once
// Deterministic-training seed source for proof-of-learning / reproducibility.
//
// Every RNG in the training path (dropout, weight init) normally self-seeds
// from std::random_device, which makes a training run non-reproducible — two
// identical runs from the same checkpoint diverge (measured 2026-07-24). For
// re-execution-based verification (a verifier re-runs a challenged segment and
// checks it reproduces the prover's committed state) the step must be
// deterministic. Set TCPP_SEED=<int> to route every self-seeding RNG through
// one committed master seed: each caller still gets a DISTINCT sub-seed (so
// dropout layers stay decorrelated), but the whole sequence is reproducible
// because model construction order is fixed. Unset TCPP_SEED => random_device
// (normal stochastic training).
#include <random>
#include <cstdlib>
#include <cstdint>
#include <mutex>

inline std::uint32_t tcpp_seed_source() {
    static const char* env = std::getenv("TCPP_SEED");
    if (!env) return std::random_device{}();
    static std::mutex m;
    static std::mt19937 master(static_cast<std::uint32_t>(std::strtoul(env, nullptr, 10)));
    std::lock_guard<std::mutex> lk(m);
    return master();
}
