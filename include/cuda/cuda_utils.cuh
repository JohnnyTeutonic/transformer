#pragma once
#include <cuda_runtime.h>

// Forward declare Matrix class
class Matrix;

namespace cuda {
    // CUDA initialization and cleanup
    bool is_initialized();
    void initialize_cuda();
    void cleanup_cuda();

    // Matrix operations
    __host__ bool customMatrixMultiply(const Matrix& A, const Matrix& B, Matrix& C);
    __host__ bool customMatrixTranspose(const Matrix& input, Matrix& output);
    __host__ bool customMatrixReshape(const Matrix& input, Matrix& output);
    __host__ void launch_softmax_kernel(float* scores, int seq_len, cudaStream_t stream);

    // Constants for optimized matrix operations
    constexpr int TILE_SIZE = 32;      // Tile size for matrix operations
    constexpr int BLOCK_ROWS = 8;      // Number of rows per thread block
    constexpr int WARP_SIZE = 32;      // Size of a warp
}

// Include Matrix definition after forward declarations
#include "../matrix.hpp"