# Determinism / reproducibility notes (verifiable-training work, 2026-07-24)

Findings from making the trainer reproducible enough for proof-of-learning by
re-execution. Measured with `--pol-save` (commitment = `payload_fnv64` of the
full model state) on the chat8b checkpoint, 3 steps.

## Gate 1 — single-thread determinism: DONE
The RNGs (dropout, weight init) self-seeded from `std::random_device`, so runs
were non-reproducible. `include/rng_seed.hpp` + `TCPP_SEED` route them through
one committed master seed. With `TCPP_SEED` set and `OMP_NUM_THREADS=1`, two
runs are **bit-identical** (verified repeatedly). This is what the PoL protocol
uses (`scripts/pol_transformer.py`, `src/pol_demo.cpp`), and PoL on the real
transformer works: honest verifies, shortcut work is caught.

## grad_norm clipping reduction: fixed
`grad_norm` for gradient clipping used `#pragma omp parallel for
reduction(+:...)` (thread-order dependent); the clip scale multiplies every
weight update, so different thread counts gave different states. Replaced 4
sites with `repro_sumsq` (`include/repro_reduce.hpp`, serial compensated). This
removed the *thread-order* component of the divergence.

## KNOWN BUG — multi-thread CPU training has a DATA RACE (open)
After the grad_norm fix, `OMP_NUM_THREADS=4` still produces **different results
on repeated runs of the identical command** (measured: `35e174..`, `2ce6e1..`,
`e9d820..` across three runs), while `OMP_NUM_THREADS=1` is rock-stable. Three
distinct outcomes run-to-run at fixed thread count is a **race**, not
float-reduction order (which would be stable per thread count). It affects the
committed model state, so multi-threaded CPU training is non-reproducible and
possibly subtly wrong.

Narrowed (ruled out): `matmul_optimized_parallel` (per-block, race-free),
`Matrix::column_sum` (serial), the four grad_norm sites (now serial), FFN
backward (matmul + serial bias sum). Prime remaining suspect: a shared
accumulation in the attention backward weight/bias-gradient path, or a shared
cache written by parallel threads. NOT YET PINPOINTED.

Impact: the deployed chat models were trained on CUDA, not this CPU path, so
model quality is not implicated. But CPU multi-thread training is
non-deterministic and this is a real correctness bug to fix (find the unguarded
shared write; add `reduction`/atomic or per-thread partials + fixed combine).

## Resolution for PoL, and the real remaining frontier
The PoL protocol does not need thread-invariance: mandate **single-threaded
verification** of the (short) challenged segment — cheap, and deterministic by
gate 1. The genuine remaining frontier is **cross-machine** single-thread
bit-fidelity: FMA contraction, platform `libm` transcendentals, SIMD. The fix
is reproducible arithmetic (error-free-transform summation — `repro_sum_demo`,
`repro_reduce_demo`), pinned `exp`/`gelu` (correctly-rounded or fixed
polynomial), and FMA contraction off. Then measure the same segment on two
different machines (this laptop vs Colab).
