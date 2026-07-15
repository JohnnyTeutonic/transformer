# Transformer Training in C++/CUDA

A pure C++ implementation of transformer training with CUDA acceleration.

## What Works (Verified)

- **End-to-end pipeline (from-scratch, no external ML frameworks)** — a small model trained here, exported to safetensors, and loaded into a separate from-scratch inference engine ([tinyllama.cpp](https://github.com/JohnnyTeutonic/tinyllama.cpp)) for working chat generation. The training and inference code are independent implementations, so successful interop verifies both the trainer's checkpoint output and the loader's parsing.
- **Core transformer training** — trained on WikiText-2; training logs show successful forward/backward passes and decreasing loss/perplexity
- **CUDA kernels** — fused attention, SwiGLU, MoE routing
- **Standard optimizations** — FP16 mixed precision, gradient accumulation, RoPE, GQA

## What Exists But Is Not Operational

The codebase contains extensive distributed training infrastructure that has been **implemented but not tested in production**:

- P2P networking with PBFT consensus
- Kademlia DHT peer discovery
- Byzantine fault detection
- RLHF/PPO training
- Web annotation interface
- Distributed checkpointing

**These features are aspirational.** The distributed code compiles but has not been validated at scale.

## Verified Training Results

Training tested on NVIDIA GeForce RTX 4060 Laptop GPU with WikiText dataset:

```
Epoch 1: Loss=8.234, Perplexity=3765.2
Epoch 2: Loss=6.891, Perplexity=983.4
...
(See training.log for full output)
```

## Installation

### Prerequisites
- C++20 compatible compiler (GCC 10+, Clang 12+, MSVC 2019+)
- CUDA 11.0+ with compatible GPU (optional but recommended)
- CMake 3.16+

### Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Windows

```cmd
mkdir build && cd build
cmake .. -G "Visual Studio 16 2019" -A x64
cmake --build . --config Release
```

## Usage

```bash
./transformer
```

## Transformer Features

### Attention
- Multi-Head Attention with GQA support
- Rotary Position Embeddings (RoPE)
- Flash Attention memory optimization
- Key-Value Cache

### Architecture
- Layer Normalization
- Feed Forward Networks with SwiGLU
- Mixture of Experts (MoE)
- Dropout and residual connections

### Training
- FP16 mixed precision
- Gradient accumulation and clipping
- Batch processing
- Checkpoint save/load

### CUDA Kernels
- Fused attention kernel
- SwiGLU activation kernel
- MoE router kernel

## Limitations

- Training uses small datasets; predictions may be sub-optimal
- Distributed features are not operational
- Trained on WikiText-2 dataset

## License

MIT License
