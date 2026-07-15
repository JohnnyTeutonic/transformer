#pragma once

#ifdef USE_CUDA

#include "cuda_matrix.hpp"

namespace cuda {
namespace kernels {

/**
 * @brief Launch SwiGLU forward pass kernel
 * 
 * Computes: output = down_proj(swish(gate_proj(x)) * up_proj(x))
 * where swish(x) = x * sigmoid(x)
 */
void swiglu_forward_kernel_launcher(
    const CudaMatrix& input,              // [batch*seq_len, hidden_size]
    const CudaMatrix& gate_proj_weights,  // [hidden_size, intermediate_size]
    const CudaMatrix& up_proj_weights,    // [hidden_size, intermediate_size]
    const CudaMatrix& down_proj_weights,  // [intermediate_size, hidden_size]
    CudaMatrix& gated_output,             // [batch*seq_len, intermediate_size] (cache)
    CudaMatrix& up_output,                // [batch*seq_len, intermediate_size] (cache)
    CudaMatrix& output                    // [batch*seq_len, hidden_size]
);

/**
 * @brief Launch SwiGLU backward pass kernel
 * 
 * Computes gradients for all parameters and input
 */
void swiglu_backward_kernel_launcher(
    const CudaMatrix& grad_output,        // [batch*seq_len, hidden_size]
    const CudaMatrix& input,              // [batch*seq_len, hidden_size]
    const CudaMatrix& gate_proj_weights,  // [hidden_size, intermediate_size]
    const CudaMatrix& up_proj_weights,    // [hidden_size, intermediate_size]
    const CudaMatrix& down_proj_weights,  // [intermediate_size, hidden_size]
    const CudaMatrix& gated_cache,        // [batch*seq_len, intermediate_size]
    const CudaMatrix& up_cache,           // [batch*seq_len, intermediate_size]
    CudaMatrix& grad_input,               // [batch*seq_len, hidden_size]
    CudaMatrix& grad_gate_proj,           // [hidden_size, intermediate_size]
    CudaMatrix& grad_up_proj,             // [hidden_size, intermediate_size]
    CudaMatrix& grad_down_proj            // [intermediate_size, hidden_size]
);

} // namespace kernels
} // namespace cuda

#endif // USE_CUDA

