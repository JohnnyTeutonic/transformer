# Training performance notes (2026-07-12 profiling session)

## Symptom
GPU-config training (512 hidden / 6 layers / batch 64 / seq 128) runs at
**~20-25 s/step** (RTX 4060) and ~56 s/step (Colab T4, 2 vCPUs). The July 3
baseline that appeared fast (~0.5 s/step) predates the gradient-correctness
commits and its loss *rose* — it was fast because backward was approximate/
near-free. Correct training exposed the real cost structure.

## Measured breakdown (batch 64, seq 128, RTX 4060; TCPP_PHASE_TIMING=1)
```
per step:  forward 7-20s | loss 0.1-0.4s (device-resident, fine) | backward+update 14s
forward:   per layer  attn ~0.5-1.4s, ffn ~0.3-0.4s, ln+caches ~0.15s
backward:  per layer  attn_bwd ~1.0s, ffn_bwd ~0.2-0.4s, ln ~0.2s; updates 0.9s total
matmul:    MATMUL_PROFILE=1 -> avg 31.4ms/call "launch" (alloc 7us, sync 23us);
           ~34.6MB avg traffic/call; ~100 calls/step ≈ 3s/step in copies alone
```

## Root cause
**Per-op host<->device round-trips on pageable memory.** Every cuda::matmul /
CudaMatrix / batched-attention call uploads its inputs and downloads its
outputs; `cudaMemcpyAsync` from pageable buffers degrades to a synchronous
~1GB/s copy. The cost repeats ~100x/step across matmuls, the attention
forward/backward pools (~200-400MB/layer/step incl. the 134MB score matrices),
FFN CudaMatrix wrappers, plus CPU-side RoPE rotation (~0.1s x24/step) and
LayerNorm forward/backward.

## What was already fixed (this session)
* Exact attention backward ported to cuBLAS (`cuda::batched_attention_backward`,
  attention_ops.cu), gradient parity vs the scalar CPU path = 3e-9.
  The scalar loops remain as the non-CUDA fallback and as the
  TCPP_BACKWARD_PARITY=1 verification path.
* Phase/layer/matmul profilers (env-gated: TCPP_PHASE_TIMING, MATMUL_PROFILE).

## Roadmap to fast training (in order of payoff)
1. **Resident device tensors (the real fix).** Keep activations on the GPU
   across ops within a step: upload the batch once, run
   LN -> QKV -> attention -> O -> FFN on device, download only the loss path
   needs. The device-resident LM head (lm_head backward_pass_cuda) is the
   in-repo template for the pattern. Expected: 20s -> <1s/step.
2. Pinned staging buffers in cuda::matmul + the attention pools (~40 lines):
   31ms -> ~8ms per call; saves ~4-5s/step. Worth doing only if (1) is
   deferred.
3. Move RoPE rotation and LayerNorm into kernels once (1) is in (they're
   memory-bound CPU passes today).

## Interim guidance
* Train the `--tiny` config (2L/128h) for real models now — it steps quickly
  even with the current architecture, and TinyStories-class data produces
  fluent tiny models.
* Colab: prefer A100 over T4 for the 512 config (the bottleneck includes CPU
  work; T4 VMs have 2 vCPUs).
