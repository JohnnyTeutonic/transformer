#include "../include/cuda/cuda_utils.cuh"

// Global cuBLAS handle definition
cublasHandle_t cublas_handle;

void initialize_cuda() {
  // Get the device with the highest compute capability
  int device_count;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  
  int max_compute = 0;
  int selected_device = 0;
  for (int i = 0; i < device_count; i++) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
    int compute = prop.major * 10 + prop.minor;
    if (compute > max_compute) {
      max_compute = compute;
      selected_device = i;
    }
  }
  
  CUDA_CHECK(cudaSetDevice(selected_device));
  CUBLAS_CHECK(cublasCreate(&cublas_handle));
  
  // Set optimal flags for maximum performance
  CUBLAS_CHECK(cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH));
  CUDA_CHECK(cudaFuncSetCacheConfig(matrix_multiply_kernel, cudaFuncCachePreferShared));
}

void cleanup_cuda() { 
  CUBLAS_CHECK(cublasDestroy(cublas_handle));
}

// Optimized matrix multiplication kernel
__global__ void matrix_multiply_kernel(const float *A, const float *B, float *C,
                                     int M, int N, int K) {
  // Use shared memory for better performance
  __shared__ float shared_A[32][32];
  __shared__ float shared_B[32][32];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int row = blockIdx.y * blockDim.y + ty;
  const int col = blockIdx.x * blockDim.x + tx;
  
  // Register for accumulating results
  float sum = 0.0f;

  // Process the matrix in 32x32 tiles
  const int num_tiles = (K + 31) / 32;
  
  for (int tile = 0; tile < num_tiles; ++tile) {
    // Collaborative loading of tiles into shared memory
    const int tile_idx = tile * 32;
    
    if (row < M && tile_idx + tx < K) {
      shared_A[ty][tx] = A[row * K + tile_idx + tx];
    } else {
      shared_A[ty][tx] = 0.0f;
    }
    
    if (col < N && tile_idx + ty < K) {
      shared_B[ty][tx] = B[(tile_idx + ty) * N + col];
    } else {
      shared_B[ty][tx] = 0.0f;
    }
    
    __syncthreads();

    // Compute partial dot product for this tile
    if (row < M && col < N) {
      #pragma unroll
      for (int k = 0; k < 32; ++k) {
        sum = __fmaf_rn(shared_A[ty][k], shared_B[k][tx], sum);  // Use FMA for better performance
      }
    }
    
    __syncthreads();
  }

  // Write result
  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}