#ifdef USE_CUDA

#include "../../include/cuda/swiglu_kernels.cuh"
#include "../../include/cuda/cuda_check.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cmath>

namespace cuda {
namespace kernels {

// Global cuBLAS handle (should be initialized elsewhere, but for now...)
static cublasHandle_t cublas_handle = nullptr;

void ensure_cublas_initialized() {
    if (!cublas_handle) {
        cublasCreate(&cublas_handle);
    }
}

// Simple swish activation kernel: out = x * sigmoid(x)
__global__ void swish_kernel(const float* __restrict__ input, 
                             float* __restrict__ output, 
                             size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float sigmoid = 1.0f / (1.0f + expf(-x));
        output[idx] = x * sigmoid;
    }
}

// Element-wise multiplication kernel: out = a * b
__global__ void elementwise_mul_kernel(const float* __restrict__ a,
                                       const float* __restrict__ b,
                                       float* __restrict__ out,
                                       size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}

// Helper: matrix multiplication using cuBLAS
void cublas_gemm(const CudaMatrix& A, const CudaMatrix& B, CudaMatrix& C,
                 bool transA = false, bool transB = false) {
    ensure_cublas_initialized();
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    int m = transA ? A.cols() : A.rows();
    int k = transA ? A.rows() : A.cols();
    int n = transB ? B.rows() : B.cols();
    
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    
    cublasSgemm(cublas_handle, opB, opA,
                n, m, k,
                &alpha,
                B.data(), transB ? k : n,
                A.data(), transA ? m : k,
                &beta,
                C.data(), n);
}

void swiglu_forward_kernel_launcher(
    const CudaMatrix& input,
    const CudaMatrix& gate_proj_weights,
    const CudaMatrix& up_proj_weights,
    const CudaMatrix& down_proj_weights,
    CudaMatrix& gated_output,
    CudaMatrix& up_output,
    CudaMatrix& output)
{
    const int block_size = 256;
    const size_t total_elements = gated_output.size();
    const int grid_size = (total_elements + block_size - 1) / block_size;
    
    // Step 1: gate_proj_output = input @ gate_proj_weights
    CudaMatrix gate_linear(input.rows(), gate_proj_weights.cols());
    cublas_gemm(input, gate_proj_weights, gate_linear, false, false);
    
    // Step 2: gated_output = swish(gate_linear)
    swish_kernel<<<grid_size, block_size>>>(
        gate_linear.data(), gated_output.data(), total_elements);
    CUDA_CHECK(cudaGetLastError());
    
    // Step 3: up_output = input @ up_proj_weights
    cublas_gemm(input, up_proj_weights, up_output, false, false);
    
    // Step 4: intermediate = gated_output * up_output
    CudaMatrix intermediate(gated_output.rows(), gated_output.cols());
    elementwise_mul_kernel<<<grid_size, block_size>>>(
        gated_output.data(), up_output.data(), intermediate.data(), total_elements);
    CUDA_CHECK(cudaGetLastError());
    
    // Step 5: output = intermediate @ down_proj_weights
    cublas_gemm(intermediate, down_proj_weights, output, false, false);
    
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Swish derivative: d_swish/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
__global__ void swish_backward_kernel(const float* __restrict__ input,
                                      const float* __restrict__ grad_output,
                                      float* __restrict__ grad_input,
                                      size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float sigmoid = 1.0f / (1.0f + expf(-x));
        float swish_grad = sigmoid * (1.0f + x * (1.0f - sigmoid));
        grad_input[idx] = grad_output[idx] * swish_grad;
    }
}

void swiglu_backward_kernel_launcher(
    const CudaMatrix& grad_output,
    const CudaMatrix& input,
    const CudaMatrix& gate_proj_weights,
    const CudaMatrix& up_proj_weights,
    const CudaMatrix& down_proj_weights,
    const CudaMatrix& gated_cache,
    const CudaMatrix& up_cache,
    CudaMatrix& grad_input,
    CudaMatrix& grad_gate_proj,
    CudaMatrix& grad_up_proj,
    CudaMatrix& grad_down_proj)
{
    const int block_size = 256;
    
    // Backward through down_proj: grad_intermediate = grad_output @ down_proj^T
    CudaMatrix grad_intermediate(grad_output.rows(), down_proj_weights.rows());
    cublas_gemm(grad_output, down_proj_weights, grad_intermediate, false, true);
    
    // grad_down_proj = intermediate^T @ grad_output
    // intermediate = gated_cache * up_cache, so need to reconstruct or use cache
    CudaMatrix intermediate(gated_cache.rows(), gated_cache.cols());
    size_t total_elem = intermediate.size();
    int grid_size = (total_elem + block_size - 1) / block_size;
    elementwise_mul_kernel<<<grid_size, block_size>>>(
        gated_cache.data(), up_cache.data(), intermediate.data(), total_elem);
    cublas_gemm(intermediate, grad_output, grad_down_proj, true, false);
    
    // Backward through element-wise multiply
    CudaMatrix grad_gated(gated_cache.rows(), gated_cache.cols());
    CudaMatrix grad_up(up_cache.rows(), up_cache.cols());
    elementwise_mul_kernel<<<grid_size, block_size>>>(
        grad_intermediate.data(), up_cache.data(), grad_gated.data(), total_elem);
    elementwise_mul_kernel<<<grid_size, block_size>>>(
        grad_intermediate.data(), gated_cache.data(), grad_up.data(), total_elem);
    
    // Backward through swish
    CudaMatrix gate_linear(input.rows(), gate_proj_weights.cols());
    cublas_gemm(input, gate_proj_weights, gate_linear, false, false);
    
    CudaMatrix grad_gate_linear(grad_gated.rows(), grad_gated.cols());
    swish_backward_kernel<<<grid_size, block_size>>>(
        gate_linear.data(), grad_gated.data(), grad_gate_linear.data(), total_elem);
    
    // grad_gate_proj = input^T @ grad_gate_linear
    cublas_gemm(input, grad_gate_linear, grad_gate_proj, true, false);
    
    // grad_up_proj = input^T @ grad_up
    cublas_gemm(input, grad_up, grad_up_proj, true, false);
    
    // grad_input = grad_gate_linear @ gate_proj^T + grad_up @ up_proj^T
    CudaMatrix grad_input_gate(input.rows(), input.cols());
    CudaMatrix grad_input_up(input.rows(), input.cols());
    cublas_gemm(grad_gate_linear, gate_proj_weights, grad_input_gate, false, true);
    cublas_gemm(grad_up, up_proj_weights, grad_input_up, false, true);
    
    // Sum the two gradients
    int input_grid = (grad_input.size() + block_size - 1) / block_size;
    elementwise_mul_kernel<<<input_grid, block_size>>>(
        grad_input_gate.data(), grad_input_up.data(), grad_input.data(), grad_input.size());
    
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace kernels
} // namespace cuda

#endif // USE_CUDA

