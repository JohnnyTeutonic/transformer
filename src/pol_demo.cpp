// Proof-of-learning challenge-response: the trust mechanism, end to end.
//
// A prover claims it ran N training steps. It commits a chain of per-step
// state hashes H_0..H_N. A verifier re-executes a FEW randomly chosen steps
// from the prover's committed prior state and checks each reproduces the
// committed hash. The verifier does O(challenges) work, not O(N) — that's the
// point: cheap to verify, and the prover can't predict which steps get
// checked, so it has to actually do all of them.
//
// This rests entirely on gate-1 determinism (rng_seed.hpp / TCPP_SEED): a step
// is reproducible only because its randomness comes from a COMMITTED per-step
// seed, not std::random_device. Without that, an honest verifier couldn't
// reproduce an honest step and the scheme is impossible. With it, a node that
// shortcut the work is caught with probability (faked fraction) per challenge.
//
// The training loop is a small real SGD (linear model, MSE, seeded minibatch)
// standing in for the full transformer step; wiring it to the real step is
// engineering, not concept.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using State = std::vector<double>;
static const int DIM = 16;

static uint64_t hash_state(const State& w) {          // FNV-1a = the commitment
    uint64_t h = 1469598103934665603ull;
    const auto* p = reinterpret_cast<const uint8_t*>(w.data());
    for (size_t i = 0; i < w.size() * sizeof(double); ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

static const State& target() {
    static State t = [] { State t(DIM); for (int i = 0; i < DIM; ++i) t[i] = std::sin(0.3 * i); return t; }();
    return t;
}

// One honest step: fully determined by (w, step_seed). The seed is the
// committed per-step randomness (gate-1). Seeded minibatch -> gradient -> SGD.
static State honest_step(const State& w, uint64_t step_seed) {
    std::mt19937_64 rng(step_seed);
    std::normal_distribution<double> nd(0, 1);
    State g(DIM, 0.0);
    const State& t = target();
    for (int b = 0; b < 8; ++b) {
        State x(DIM); double pred = 0, tgt = 0;
        for (int i = 0; i < DIM; ++i) { x[i] = nd(rng); pred += w[i] * x[i]; tgt += t[i] * x[i]; }
        double err = pred - tgt;
        for (int i = 0; i < DIM; ++i) g[i] += err * x[i];
    }
    State wn(DIM);
    for (int i = 0; i < DIM; ++i) wn[i] = w[i] - 0.01 * g[i] / 8.0;
    return wn;
}

static const int N = 20;
static const uint64_t BASE = 0xC0FFEEull;

// Verifier: challenge C random steps; return the step where a lie is caught, else 0.
static int verify(const std::vector<State>& chain, const std::vector<uint64_t>& commit,
                  int challenges, std::mt19937& vr) {
    std::uniform_int_distribution<int> pick(1, N);
    for (int c = 0; c < challenges; ++c) {
        int k = pick(vr);
        if (hash_state(honest_step(chain[k - 1], BASE + k)) != commit[k]) return k;
    }
    return 0;
}

int main() {
    State w0(DIM, 0.0);

    // Honest prover builds the real chain + commitments.
    std::vector<State> chain = {w0};
    std::vector<uint64_t> commit = {hash_state(w0)};
    for (int k = 1; k <= N; ++k) {
        chain.push_back(honest_step(chain[k - 1], BASE + k));
        commit.push_back(hash_state(chain.back()));
    }

    const int C = 5, TRIALS = 20000;

    // Honest prover must pass with EVERY verifier draw (it always reproduces).
    int hp = 0;
    for (int t = 0; t < TRIALS; ++t) { std::mt19937 vr(t); hp += (verify(chain, commit, C, vr) != 0); }
    printf("honest prover: false positives over %d verifier draws (%d challenges each) = %d\n\n",
           TRIALS, C, hp);

    // Lying provers: claim N steps but shortcut `fake` of them (state unchanged,
    // committed as if worked). Caught iff a challenge hits a faked step:
    // P(catch) = 1 - (1 - fake/N)^C. Measure empirically over many draws.
    printf("lying provers (shortcut the work), %d challenges, %d verifier draws each:\n", C, TRIALS);
    for (int fake : {N, N / 2, 3, 1}) {
        std::vector<State> lc = {w0};
        std::vector<uint64_t> lcm = {hash_state(w0)};
        std::mt19937 which(99); std::uniform_int_distribution<int> pk(1, N);
        std::vector<char> faked(N + 1, 0);
        for (int f = 0; f < fake;) { int s = pk(which); if (!faked[s]) { faked[s] = 1; ++f; } }
        for (int k = 1; k <= N; ++k) {
            State s = faked[k] ? lc[k - 1] : honest_step(lc[k - 1], BASE + k);
            lc.push_back(s); lcm.push_back(hash_state(s));
        }
        int caught = 0;
        for (int t = 0; t < TRIALS; ++t) { std::mt19937 vr2(t); caught += (verify(lc, lcm, C, vr2) != 0); }
        double p = 1.0 - std::pow(1.0 - double(fake) / N, C);
        printf("  faked %2d/%d  caught %5.1f%% empirical  (theory %5.1f%%)\n",
               fake, N, 100.0 * caught / TRIALS, 100 * p);
    }
    printf("\nHonest work always verifies; shortcut work is caught at the predicted rate.\n"
           "Small cheats need more challenges (P->1 exponentially in C). Rests on gate-1\n"
           "determinism: without a committed per-step seed, even honest steps wouldn't reproduce.\n");
    return 0;
}
