#pragma once
// Deterministic (thread-count-invariant) reductions for gate-2 reproducibility.
//
// Gradient-norm-for-clipping was computed with `#pragma omp parallel for
// reduction(+:grad_norm)`, whose combine order depends on the thread count, so
// the clip scaling factor — and therefore the updated weights — differed
// between a 1-thread and a 4-thread run of the same seed (measured 2026-07-24:
// state commitments 1dd123..b vs 356e09..6). For proof-of-learning by
// re-execution the step must be reproducible regardless of how a machine
// splits the work.
//
// repro_sumsq is a single-pass Neumaier-compensated sum of squares in double:
// no parallelism, so it is trivially thread-invariant, and it uses only
// correctly-rounded IEEE ops so it is cross-machine reproducible with FMA
// contraction off. grad_norm is O(params), negligible next to the matmuls, so
// making it serial-compensated costs ~nothing and buys determinism.
#include <cmath>
#include <cstddef>

inline float repro_sumsq(const float* p, std::size_t n) {
    double s = 0.0, c = 0.0;                 // c carries the rounding residual
    for (std::size_t i = 0; i < n; ++i) {
        double x = static_cast<double>(p[i]) * static_cast<double>(p[i]);
        double t = s + x;
        c += (std::fabs(s) >= std::fabs(x)) ? (s - t) + x : (x - t) + s;
        s = t;
    }
    return static_cast<float>(s + c);
}
