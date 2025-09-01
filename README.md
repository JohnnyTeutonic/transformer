# A decoder-style Transformer in C++

A pure C++ implementation of a decoder-only transformer model with CUDA support. It is based on the paper "Attention is All You Need" by Vaswani et al and has been trained on an example dataset found in the `data` directory called 'training_pairs.txt'. It performs a single token prediction for each input.

## Transformer Implementation Features

## Core Attention Mechanisms

- Standard Multi-Head Attention
- Grouped Query Attention (GQA)
- Flash Attention optimization
- Rotary Position Embeddings (RoPE)
- Sliding Window Attention
- Key-Value Cache support

## Architecture Components

- Layer Normalization
- Feed Forward Networks
- Dropout layers
- Residual connections
- Language Model Head
- Tokenizer with vocabulary management

## Training Features

- Batch processing
- Beam Search
- Dynamic learning rate adjustment
- Gradient clipping
- Loss computation and backpropagation
- Training/Evaluation modes
- Gradient checkpointing
- Performance metrics tracking

## Optimization Features

- CUDA support for GPU acceleration
- OpenMP parallelisation
- Half-precision (FP16) support
- Memory pooling
- Gradient accumulation
- SAM (Sharpness-Aware Minimization) optimizer

## Distributed Training 🚀

**Scale your transformer training across multiple GPUs and nodes with zero-dependency MPI integration!**

### Features
- **Multi-node distributed training** with MPI
- **Multi-GPU communication** with NCCL
- **Automatic gradient synchronization** across all processes
- **Zero-dependency building** with automatic MPI/NCCL stubs
- **Non-breaking integration** - existing code works unchanged

### Quick Start

```cpp
#include "distributed_transformer.hpp"

int main(int argc, char** argv) {
    // Configure your transformer
    TransformerConfig config;
    config.vocab_size = 50257;
    config.hidden_size = 768;
    config.num_layers = 12;
    config.num_heads = 12;
    config.max_seq_length = 1024;
    config.initial_lr = 0.0001f;
    
    // Initialize tokenizer
    auto tokenizer = std::make_shared<TiktokenTokenizer>();
    
    // Create distributed transformer (MPI_Init called automatically)
    DistributedTransformer model(config, tokenizer, argc, argv);
    
    // Train across all nodes - gradients sync automatically!
    std::vector<std::vector<int>> input_batches = {{1, 2, 3, 4}};
    std::vector<std::vector<int>> target_batches = {{2, 3, 4, 5}};
    
    model.train(input_batches, target_batches, /*epochs=*/10, /*lr=*/0.001f);
    
    return 0;
}
```

### Building with Distributed Support

```bash
# Build with system MPI/NCCL (if available)
mkdir build && cd build
cmake ..
make distributed_training_example -j$(nproc)

# Run on single node (uses stubs)
./distributed_training_example

# Run on multiple nodes with MPI
mpirun -np 4 ./distributed_training_example

# Run on multiple nodes across machines
mpirun -np 8 -hostfile hosts.txt ./distributed_training_example
```

### Architecture

- **`DistributedTransformer`**: Wrapper around your existing `Transformer`
- **Gradient Synchronization**: AllReduce operations after each backward pass
- **CUDA Streams**: Separate compute and communication streams for efficiency
- **Automatic Cleanup**: MPI_Finalize called in destructor

### Performance Benefits

- **Linear scaling** across multiple GPUs/nodes
- **Reduced training time** from hours to minutes
- **Larger effective batch sizes** through data parallelism
- **Memory efficiency** by distributing model replicas

See `examples/distributed_training_example.cpp` for a complete working example.

## Advanced Features

- Quantization-Aware Training
- Adaptive cache replacement policies
- Token embedding with positional encoding
- Advanced attention mechanisms (block-sparse)
- Configurable model architecture

## Utility Features

- JSON configuration loading
- Model checkpointing and saving
- Performance metrics logging
- Validation data evaluation
- Token prediction and probability calculation
- Text preprocessing and tokenization

## Memory Management

- Memory pooling
- Cache management
- Gradient checkpointing
- Efficient matrix operations

## Development Features

- Comprehensive logging
- Error handling
- Configuration validation
- Performance profiling
- Debug output options

## Dependencies

### Required
- OpenMP: <https://github.com/OpenMP/openmp-api>
- CUDA: <https://developer.nvidia.com/cuda-downloads>
- nlohmann/json: <https://github.com/nlohmann/json> (auto-downloaded via CMake)

### Optional (for Distributed Training)
- MPI: <https://www.open-mpi.org/> or <https://www.mpich.org/>
- NCCL: <https://github.com/NVIDIA/nccl> (for multi-GPU communication)

**Note**: If MPI/NCCL are not installed, the build system automatically creates lightweight stubs for single-node development.

## Building

To build the project, you can use the following commands:

```bash
mkdir build
cd build
cmake ..
make
```

## Training the model

After building the project, running `main.cpp` will train the model and save the model hyperparamters to whatever directory is specified in the `config/transformer_config.json` file. To execute the training on the sample dataset, run the following command
from the build directory:

```bash
./transformer
```

## Logging

The logging is done to a file called `transformer.log` in the `build` directory.

## Configuration

The configuration is done in the `config/transformer_config.json` file.

## Limitations

- The model training is performed on a very small dataset, so its predictions are certainly sub-optimal, given its constraints.
- It only works on a format that follows the training data i.e I like to cook in the |kitchen (where `|` is the delimiter).
