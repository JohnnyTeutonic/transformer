/**
 * @file matmul_optimized.hpp
 * @brief Optimized matrix multiplication interface
 * 
 * Provides drop-in replacement for naive matmul with 12-18x speedup (single-threaded)
 * or 24-72x speedup (multi-threaded with OpenMP).
 * 
 * Usage:
 * ------
 * 
 * // Option 1: Single-threaded (deterministic, no dependencies)
 * Matrix C = matmul_optimized(A, B);
 * 
 * // Option 2: Multi-threaded (requires OpenMP)
 * Matrix C = matmul_optimized_parallel(A, B);
 * 
 * Compilation:
 * ------------
 * 
 * Single-threaded (no flags needed):
 *   g++ -O3 -march=native matmul_optimized.cpp
 * 
 * Multi-threaded (add -fopenmp):
 *   g++ -O3 -march=native -fopenmp matmul_optimized.cpp
 * 
 * AVX2 explicit (if -march=native doesn't work):
 *   g++ -O3 -mavx2 -mfma matmul_optimized.cpp
 * 
 * Architecture Support:
 * ---------------------
 * - AVX2 (2013+): Best performance (8 floats/instruction + FMA)
 * - AVX (2011+): Good performance (8 floats/instruction)
 * - SSE2 (2001+): Moderate performance (4 floats/instruction)
 * - Scalar: Fallback (still cache-optimized, just no SIMD)
 * 
 * The code auto-detects available instructions at compile time.
 */

#pragma once

#include "matrix.hpp"

/**
 * @brief Optimized single-threaded matrix multiplication
 * @param A Input matrix (M×K)
 * @param B Input matrix (K×N)
 * @return Result matrix C (M×N) where C = A * B
 * @throws std::runtime_error if dimensions don't match
 * 
 * Expected speedup over naive implementation:
 * - i3/i5/i7 (AVX2): 12-18x
 * - Older CPUs (SSE2): 6-10x
 * 
 * Memory usage: O(M*N) for result, no temporary buffers
 * Deterministic: Always produces same result
 */
Matrix matmul_optimized(const Matrix& A, const Matrix& B);

/**
 * @brief Optimized multi-threaded matrix multiplication (OpenMP)
 * @param A Input matrix (M×K)
 * @param B Input matrix (K×N)
 * @return Result matrix C (M×N) where C = A * B
 * @throws std::runtime_error if dimensions don't match
 * 
 * Expected speedup over naive implementation:
 * - 4-core i3 (AVX2): 24-72x (depends on thread scaling)
 * - 2-core i3 (AVX2): 18-36x
 * 
 * Requires: Compile with -fopenmp
 * Thread count: Controlled by OMP_NUM_THREADS environment variable
 * 
 * Example:
 *   export OMP_NUM_THREADS=4
 *   ./train_wikitext
 */
Matrix matmul_optimized_parallel(const Matrix& A, const Matrix& B);

