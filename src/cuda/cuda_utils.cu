#include "../../include/cuda/cuda_utils.cuh"
#include "../../include/cuda/cuda_check.cuh"
#include "../../include/cuda/cuda_init.cuh"
#include "../../include/matrix.hpp"
#include <cuda_runtime.h>

namespace cuda {
    // Remove cuBLAS-related constants
    static bool cuda_initialized = false;
    
    // Static storage for GPU matrices
    static Matrix stored_A_gpu;
    static Matrix stored_B_gpu;
    static Matrix stored_C_gpu;
    static bool matrices_initialized = false;

    // Optimize tile size for modern GPUs
    #define TILE_SIZE 32
    #define BLOCK_ROWS 8
    
    // Add warp size constant
    #define WARP_SIZE 32

    __global__ void matrixMulKernel(const float* __restrict__ A, 
                                   const float* __restrict__ B, 
                                   float* __restrict__ C,
                                   int M, int K, int N) {
        // Shared memory for tiles
        __shared__ float As[TILE_SIZE][TILE_SIZE];
        __shared__ float Bs[TILE_SIZE][TILE_SIZE];
        
        // Block row and column
        int blockRow = blockIdx.y;
        int blockCol = blockIdx.x;
        
        // Thread row and column within tile
        int row = threadIdx.y;
        int col = threadIdx.x;
        
        // Global row and column
        int globalRow = blockRow * TILE_SIZE + row;
        int globalCol = blockCol * TILE_SIZE + col;
        
        float sum = 0.0f;
        
        // Loop over tiles
        for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
            // Load tiles into shared memory
            if (globalRow < M && t * TILE_SIZE + col < K) {
                As[row][col] = A[globalRow * K + t * TILE_SIZE + col];
            } else {
                As[row][col] = 0.0f;
            }
            
            if (t * TILE_SIZE + row < K && globalCol < N) {
                Bs[row][col] = B[(t * TILE_SIZE + row) * N + globalCol];
            } else {
                Bs[row][col] = 0.0f;
            }
            
            __syncthreads();
            
            // Compute partial dot product for this tile
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; k++) {
                sum += As[row][k] * Bs[k][col];
            }
            
            __syncthreads();
        }
        
        // Write result
        if (globalRow < M && globalCol < N) {
            C[globalRow * N + globalCol] = sum;
        }
    }

    bool customMatrixMultiply(const Matrix& A, const Matrix& B, Matrix& C) {
        printf("\n=== Starting Custom Matrix Multiply ===\n");
        const int M = A.rows();
        const int K = A.cols();
        const int N = B.cols();
        
        printf("Matrix dimensions: A(%d,%d) * B(%d,%d) = C(%d,%d)\n", 
               M, K, K, N, M, N);
        
        // Configure kernel launch parameters
        dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
        dim3 numBlocks(
            (N + TILE_SIZE - 1) / TILE_SIZE,
            (M + TILE_SIZE - 1) / TILE_SIZE
        );
        
        printf("Launch configuration:\n");
        printf("Block dimensions: %dx%d\n", TILE_SIZE, TILE_SIZE);
        printf("Grid dimensions: %dx%d\n", numBlocks.x, numBlocks.y);
        
        // Launch kernel
        matrixMulKernel<<<numBlocks, threadsPerBlock>>>(
            A.get_data(),
            B.get_data(),
            C.get_data(),
            M, K, N
        );
        
        // Check for errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
            return false;
        }
        
        // Synchronize and check for execution errors
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
            return false;
        }
        
        printf("=== Matrix Multiplication Complete ===\n\n");
        return true;
    }

    __global__ void softmax_kernel(float* scores, int seq_len) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        printf("Row: %d, seq_len: %d\n", row, seq_len);
        if (row < seq_len) {
            // Find max value for numerical stability
            float max_val = scores[row * seq_len];
            for (int i = 1; i < seq_len; i++) {
                max_val = max(max_val, scores[row * seq_len + i]);
            }

            // Compute exp and sum
            float sum = 0.0f;
            for (int i = 0; i < seq_len; i++) {
                scores[row * seq_len + i] = expf(scores[row * seq_len + i] - max_val);
                sum += scores[row * seq_len + i];
            }

            // Normalize
            for (int i = 0; i < seq_len; i++) {
                scores[row * seq_len + i] /= sum;
            }
        }
        printf("Softmax kernel completed\n");
    }

    void launch_softmax_kernel(float* scores, int seq_len, cudaStream_t stream) {
        printf("Launching softmax kernel\n");
        if (!is_initialized()) {
            initialize_cuda();
        }

        dim3 block_dim(256);
        dim3 grid_dim((seq_len + block_dim.x - 1) / block_dim.x);

        softmax_kernel<<<grid_dim, block_dim, 0, stream>>>(scores, seq_len);
        CUDA_CHECK(cudaGetLastError());
    }

    Matrix matmul(const Matrix& A, const Matrix& B, Matrix* output) {
        printf("Starting matmul\n");
        if (!is_initialized()) {
            try {
                printf("\n=== Starting CUDA Matrix Multiplication ===\n");
                initialize_cuda();
            } catch (const std::exception& e) {
                printf("CUDA initialization failed: %s\n", e.what());
                return Matrix(0, 0);
            }
        }
        
        const int M = A.rows();
        const int N = B.cols();
        const int K = A.cols();
        
        printf("Matrix dimensions:\n");
        printf("A: %dx%d\n", M, K);
        printf("B: %dx%d\n", B.rows(), N);
        
        if (K != B.rows()) {
            printf("Dimension mismatch! A: %dx%d, B: %dx%d\n", M, K, B.rows(), N);
            throw std::runtime_error("Matrix dimension mismatch");
        }
        
        if (!A.get_data() || !B.get_data()) {
            printf("Input matrix data is null!\n");
            return Matrix(0, 0);
        }

        Matrix& C = output ? *output : *(new Matrix(M, N));
        
        printf("\nTransferring matrices to GPU:\n");
        printf("Matrix A location: %s\n", A.is_cuda() ? "GPU" : "CPU");
        printf("Matrix B location: %s\n", B.is_cuda() ? "GPU" : "CPU");
        
        Matrix A_gpu = A.is_cuda() ? A : A.to_gpu();
        printf("A transferred to GPU, pointer: %p\n", static_cast<const void*>(A_gpu.gpu_data()));
        
        Matrix B_gpu = B.is_cuda() ? B : B.to_gpu();
        printf("B transferred to GPU, pointer: %p\n", static_cast<const void*>(B_gpu.gpu_data()));
        
        Matrix C_gpu = C.is_cuda() ? C : Matrix(M, N, true);
        printf("C created on GPU, pointer: %p\n", static_cast<const void*>(C_gpu.gpu_data()));

        printf("\nStarting matrix multiplication...\n");
        if (!customMatrixMultiply(A_gpu, B_gpu, C_gpu)) {
            printf("Matrix multiplication failed\n");
            return Matrix(0, 0);
        }

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("Synchronization failed: %s\n", cudaGetErrorString(err));
            return Matrix(0, 0);
        }

        printf("\nTransferring result:\n");
        if (!output || !output->is_cuda()) {
            printf("Converting result to CPU\n");
            C = C_gpu.to_cpu();
        } else {
            printf("Keeping result on GPU\n");
            C = C_gpu;
        }
        printf("=== Matrix Multiplication Complete ===\n\n");

        return C;
    }

    // Add transpose kernel
    __global__ void transposeKernel(const float* input, float* output,
                                   int rows, int cols) {
        printf("Starting transpose kernel\n");
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int idy = blockIdx.y * blockDim.y + threadIdx.y;
        
        if (idx < cols && idy < rows) {
            output[idx * rows + idy] = input[idy * cols + idx];
        }
        printf("Transpose kernel completed\n");
    }

    bool customMatrixTranspose(const Matrix& input, Matrix& output) {
        printf("Starting customMatrixTranspose\n");
        const int rows = input.rows();
        const int cols = input.cols();
        
        // Verify dimensions
        if (output.rows() != cols || output.cols() != rows) {
            printf("Invalid output dimensions for transpose\n");
            return false;
        }
        
        // Configure kernel launch parameters
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks(
            (cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (rows + threadsPerBlock.y - 1) / threadsPerBlock.y
        );
        
        // Launch transpose kernel
        transposeKernel<<<numBlocks, threadsPerBlock>>>(
            input.get_data(),
            output.get_data(),
            rows, cols
        );
        
        // Check for errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Transpose kernel launch failed: %s\n", cudaGetErrorString(err));
            return false;
        }
        printf("Transpose kernel completed\n");
        return true;
    }

    __global__ void reshapeKernel(const float* input, float* output, int input_rows, int input_cols, int output_cols) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_elements = input_rows * output_cols;
        
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("Reshape Kernel Debug:\n");
            printf("Input dimensions: %dx%d\n", input_rows, input_cols);
            printf("Output cols: %d\n", output_cols);
            printf("Total elements: %d\n", total_elements);
        }
        
        if (idx < total_elements) {
            int row = idx / output_cols;
            int col = idx % output_cols;
            
            if (col < input_cols) {
                output[idx] = input[row * input_cols + col];
                if (idx % 100 == 0) { // Print every 100th element to avoid flooding
                    printf("Thread %d: Copying element from (%d,%d) to (%d,%d), value: %f\n", 
                           idx, row, col, row, col, output[idx]);
                }
            } else {
                output[idx] = 0.0f;
                if (idx % 100 == 0) { // Print every 100th element to avoid flooding
                    printf("Thread %d: Padding zero at (%d,%d)\n", idx, row, col);
                }
            }
        }
    }

    bool customMatrixReshape(const Matrix& input, Matrix& output) {
        printf("\n=== Starting Matrix Reshape ===\n");
        printf("Input matrix: %zux%zu\n", input.rows(), input.cols());
        printf("Output matrix: %zux%zu\n", output.rows(), output.cols());
        
        // Verify dimensions
        if (input.rows() != output.rows()) {
            printf("Error: Row dimensions must match! Input: %zu, Output: %zu\n", 
                   input.rows(), output.rows());
            return false;
        }
        
        // Get GPU pointers
        const float* d_input = input.gpu_data();
        float* d_output = output.gpu_data();
        
        if (!d_input || !d_output) {
            printf("Error: Invalid GPU pointers! Input: %p, Output: %p\n", 
                   static_cast<const void*>(d_input), static_cast<const void*>(d_output));
            return false;
        }
        
        // Calculate grid and block dimensions
        int total_elements = input.rows() * output.cols();
        int block_size = 256;
        int grid_size = (total_elements + block_size - 1) / block_size;
        
        printf("Launch configuration:\n");
        printf("Total elements: %d\n", total_elements);
        printf("Block size: %d\n", block_size);
        printf("Grid size: %d\n", grid_size);
        
        // Launch kernel
        printf("Launching reshape kernel...\n");
        reshapeKernel<<<grid_size, block_size>>>(
            d_input, d_output, 
            static_cast<int>(input.rows()), 
            static_cast<int>(input.cols()), 
            static_cast<int>(output.cols())
        );
        
        // Check for kernel errors
        cudaError_t cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(cuda_status));
            return false;
        }
        
        // Synchronize and check for errors
        cuda_status = cudaDeviceSynchronize();
        if (cuda_status != cudaSuccess) {
            printf("Kernel execution failed: %s\n", cudaGetErrorString(cuda_status));
            return false;
        }
        
        printf("=== Matrix Reshape Completed Successfully ===\n\n");
        return true;
    }
}