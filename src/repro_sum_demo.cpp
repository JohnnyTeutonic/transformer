// Gate-2 brick 1: reproducible summation.
//
// Cross-machine bit-fidelity needs the reduction to give the SAME bits
// regardless of the order it's summed in (different hardware = different
// SIMD/thread order). Float addition is non-associative, so a naive sum is
// order-dependent — that is the cross-machine determinism problem in
// miniature. This demonstrates the fix and measures it:
//   * Neumaier compensation carries the rounding residual in an extra
//     accumulator `c` (Jonathan's "residual precision in an extra float").
//   * A canonical summation order (sort) removes the order-dependence, so any
//     permutation of the inputs yields identical bits.
// Together: order-invariant summation. (Sorting is the pedagogical form;
// ReproBLAS / Demmel-Nguyen get the same invariance at BLAS speed via
// error-free transforms + exponent binning, no sort.)
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static double naive_sum(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x;                 // order-dependent
    return s;
}

// Neumaier compensated sum: `c` accumulates the exact rounding error lost at
// each add — the residual carried in an extra float.
static double neumaier_sum(const std::vector<double>& v) {
    double s = 0.0, c = 0.0;
    for (double x : v) {
        double t = s + x;
        c += (std::fabs(s) >= std::fabs(x)) ? (s - t) + x : (x - t) + s;
        s = t;
    }
    return s + c;
}

// Reproducible: impose a canonical order, then compensate. Any permutation of
// the same multiset sums in the same order -> identical bits on any machine.
static double reproducible_sum(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return neumaier_sum(v);
}

static uint64_t bits(double d) { uint64_t u; std::memcpy(&u, &d, 8); return u; }

int main() {
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> big(-1e9, 1e9), tiny(-1e-3, 1e-3);
    std::vector<double> v;
    for (int i = 0; i < 200000; ++i) v.push_back((i % 2) ? big(rng) : tiny(rng));

    std::vector<double> fwd = v, rev = v, shuf = v;
    std::reverse(rev.begin(), rev.end());
    std::shuffle(shuf.begin(), shuf.end(), std::mt19937(7));

    auto report = [&](const char* name, double (*f)(std::vector<double>),
                      double (*fc)(const std::vector<double>&)) {
        double a = f ? f(fwd) : fc(fwd);
        double b = f ? f(rev) : fc(rev);
        double c = f ? f(shuf) : fc(shuf);
        bool inv = bits(a) == bits(b) && bits(a) == bits(c);
        printf("%-18s fwd=%.17g\n%-18s rev=%.17g\n%-18s shuf=%.17g\n",
               name, a, "", b, "", c);
        printf("   bits: fwd=%016llx rev=%016llx shuf=%016llx  -> order-invariant? %s\n\n",
               (unsigned long long)bits(a), (unsigned long long)bits(b),
               (unsigned long long)bits(c), inv ? "YES" : "NO");
        return inv;
    };

    bool naive_inv = report("naive_sum", nullptr, naive_sum);
    bool repro_inv = report("reproducible_sum", reproducible_sum, nullptr);

    bool ok = !naive_inv && repro_inv;
    printf("RESULT: %s\n", ok
        ? "naive is order-dependent, reproducible is order-INVARIANT -> gate-2 primitive works"
        : "unexpected");
    return ok ? 0 : 1;
}
