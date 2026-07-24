// Gate-2 brick 2: thread-count-invariant (reproducible) parallel reduction.
//
// Measured on the real trainer: with a committed seed, 1-thread and 4-thread
// runs produce DIFFERENT state commitments -- the parallel reduction combines
// partials in a thread-count-dependent order, and float add is non-associative.
// That is the cross-machine determinism problem (different hardware = different
// split) in its most common form.
//
// Fix: split into FIXED-size chunks (independent of thread count), sum each
// chunk with Neumaier compensation (the residual-in-an-extra-float idea), then
// combine the chunk totals in FIXED chunk order. The result is then identical
// for any thread count -- and, with FMA contraction off, across machines, since
// it uses only correctly-rounded IEEE +,-,*.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include <omp.h>

static uint64_t bits(double d) { uint64_t u; std::memcpy(&u, &d, 8); return u; }

// Naive parallel reduction: combination order depends on the thread count.
static double naive_parallel_sum(const std::vector<double>& v) {
    double s = 0.0;
    #pragma omp parallel for reduction(+ : s)
    for (long i = 0; i < (long)v.size(); ++i) s += v[i];
    return s;
}

// Neumaier compensated chunk sum.
static double comp_chunk(const double* p, size_t n) {
    double s = 0.0, c = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double t = s + p[i];
        c += (std::fabs(s) >= std::fabs(p[i])) ? (s - t) + p[i] : (p[i] - t) + s;
        s = t;
    }
    return s + c;
}

// Reproducible reduction: fixed chunks (thread-count independent), each summed
// with compensation in parallel, combined in fixed chunk-index order.
static double repro_reduce(const std::vector<double>& v) {
    const size_t CHUNK = 1024;
    const size_t nch = (v.size() + CHUNK - 1) / CHUNK;
    std::vector<double> part(nch);
    #pragma omp parallel for
    for (long c = 0; c < (long)nch; ++c) {
        size_t b = c * CHUNK, e = std::min(b + CHUNK, v.size());
        part[c] = comp_chunk(v.data() + b, e - b);        // written by index c: no race
    }
    return comp_chunk(part.data(), nch);                  // fixed order
}

int main() {
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> big(-1e9, 1e9), tiny(-1e-3, 1e-3);
    std::vector<double> v;
    for (int i = 0; i < 1000000; ++i) v.push_back((i % 2) ? big(rng) : tiny(rng));

    printf("threads |            naive (bits)  |       reproducible (bits)\n");
    uint64_t naive0 = 0, repro0 = 0;
    bool naive_varies = false, repro_const = true;
    for (int t : {1, 2, 4, 8}) {
        omp_set_num_threads(t);
        uint64_t n = bits(naive_parallel_sum(v)), r = bits(repro_reduce(v));
        if (t == 1) { naive0 = n; repro0 = r; }
        naive_varies |= (n != naive0);
        repro_const  &= (r == repro0);
        printf("   %2d   |  %016llx        |  %016llx\n", t,
               (unsigned long long)n, (unsigned long long)r);
    }
    bool ok = naive_varies && repro_const;
    printf("\nnaive: %s across thread counts; reproducible: %s\n",
           naive_varies ? "VARIES" : "constant",
           repro_const ? "IDENTICAL" : "varies");
    printf("RESULT: %s\n", ok
        ? "reproducible reduction is thread-count-invariant -> gate-2 reduction primitive works"
        : "unexpected (naive may not have varied on this box; try a larger array)");
    return ok ? 0 : 1;
}
