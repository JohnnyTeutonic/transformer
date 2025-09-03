# Project Roadmap & Future Work

This document outlines potential future enhancements and areas for improvement for the C++ P2P Transformer project. These represent the next frontiers in performance, security, and decentralization.

## 1. Performance Optimizations

### 1.1. High-Performance MoE Dispatch/Combine CUDA Kernel

- **Current Status:** The Mixture of Experts (MoE) layer currently relies on a CPU-based fallback for the complex task of dispatching tokens to the correct expert GPUs and combining the results. While functional, this introduces a significant performance bottleneck by moving data between the CPU and GPU during the forward pass.
- **Next Step:** Implement a highly-optimized CUDA kernel for this "scatter/gather" operation.
- **Challenges:** This is a non-trivial task that involves advanced CUDA programming. It requires efficiently managing memory and execution for irregular, ragged groupings of data in parallel, which is a classic hard problem in high-performance computing.
- **Expected Impact:** A native CUDA implementation would provide a massive performance uplift, allowing the MoE layer to operate at full GPU speed and unlocking the model's true potential for large-scale, low-latency inference and training.

## 2. Consensus & Security Enhancements

### 2.1. Formal BFT Consensus Algorithm

- **Current Status:** The network currently uses a practical, two-phase (Prepare/Commit) BFT consensus mechanism. This provides strong guarantees against simple attacks and network splits and is a major step up from a basic majority vote.
- **Next Step:** Evolve the current mechanism into a more formal, academic-grade BFT protocol (e.g., implementing key aspects of PBFT or a more modern variant).
- **Benefits:** A formal protocol would provide mathematically provable guarantees of safety and liveness under a wider and more strictly-defined set of adversarial conditions. This would harden the network to a state where it could be considered for mission-critical or high-value decentralized applications.

### 2.2. Advanced Peer Discovery (DHT)

- **Current Status:** The network uses a robust Peer Exchange (PEX) protocol for decentralized peer discovery. This is effective and has no external dependencies.
- **Alternative Future Step:** For even greater scalability and resilience, especially in very large or partitioned networks, a Kademlia-based Distributed Hash Table (DHT) could be implemented.
- **Trade-offs:** While a DHT offers theoretical advantages in network topology management, it would likely require integrating a new, complex third-party library, increasing the build complexity for end-users. The current PEX implementation is a strong and simple alternative.
