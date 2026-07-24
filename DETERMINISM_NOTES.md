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

## Multi-thread CPU training DATA RACE — FOUND and FIXED (2026-07-24)
After the grad_norm fix, `OMP_NUM_THREADS=4` still gave **different results on
repeated runs of the identical command** (`35e174`, `2ce6e1`, `e9d820`), while
1-thread was rock-stable — the signature of a race, not reduction order.

Root cause: **LayerNorm/RMSNorm backward** (`layer_norm.cpp`) accumulated the
gamma/beta gradients with `#pragma omp atomic grads_.gamma_grad(0,j) += ...`
inside a rows-parallel loop. The atomic prevents corruption but NOT ordering:
concurrent adds to the same column landed in thread-scheduling order, and float
addition is non-associative, so the summed gradient varied run-to-run. Since
norm gradients update gamma/beta on every layer, this made the whole committed
state non-deterministic under multithreading.

Fix: compute gamma/beta gradients by parallelizing over COLUMNS and summing
over rows in a fixed order (one thread per column, no atomics), in double.
grad_input stays rows-parallel (per-element, race-free). Both branches (RMSNorm
+ classic LayerNorm) fixed.

Verified: three `OMP_NUM_THREADS=4` runs now all give `35e174..`, identical to
the 1-thread result — **fully thread-count-invariant and deterministic
run-to-run**, loss unchanged (9.57). The trainer is now reproducible at any
thread count, not just single-threaded.

## Resolution for PoL, and the real remaining frontier
The PoL protocol does not need thread-invariance: mandate **single-threaded
verification** of the (short) challenged segment — cheap, and deterministic by
gate 1. The genuine remaining frontier is **cross-machine** single-thread
bit-fidelity: FMA contraction, platform `libm` transcendentals, SIMD. The fix
is reproducible arithmetic (error-free-transform summation — `repro_sum_demo`,
`repro_reduce_demo`), pinned `exp`/`gelu` (correctly-rounded or fixed
polynomial), and FMA contraction off. Then measure the same segment on two
different machines (this laptop vs Colab).
