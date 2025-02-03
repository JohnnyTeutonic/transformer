#include <cuda_runtime.h>

__global__ void matrix_multiply_transpose_kernel(
    const float* __restrict__ projection,    // [vocab_size x hidden_size]
    const float* __restrict__ grad_output,   // [batch_size x vocab_size]
    float* __restrict__ grad_proj,          // [batch_size x hidden_size]
    int batch_size,
    int vocab_size,
    int hidden_size
) {
    // Each thread computes one element of grad_proj
    int row = blockIdx.y * blockDim.y + threadIdx.y; // batch dimension
    int col = blockIdx.x * blockDim.x + threadIdx.x; // hidden dimension
    
    if (row < batch_size && col < hidden_size) {
        float sum = 0.0f;
        for (int k = 0; k < vocab_size; k++) {
            // projection[k][col] * grad_output[row][k]
            sum += projection[k * hidden_size + col] * grad_output[row * vocab_size + k];
        }
        grad_proj[row * hidden_size + col] = sum;
    }
}

extern "C" {

cudaError_t launch_matrix_multiply(
    const float* projection,
    const float* grad_output,
    float* grad_proj,
    int batch_size,
    int vocab_size,
    int hidden_size,
    cudaStream_t stream = nullptr
) {
    dim3 block(16, 16);  // 256 threads per block
    dim3 grid(
        (hidden_size + block.x - 1) / block.x,
        (batch_size + block.y - 1) / block.y
    );
    
    matrix_multiply_transpose_kernel<<<grid, block, 0, stream>>>(
        projection,
        grad_output,
        grad_proj,
        batch_size,
        vocab_size,
        hidden_size
    );
    
    return cudaGetLastError();
}

} 