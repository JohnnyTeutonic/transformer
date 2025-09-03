#pragma once

#include "matrix_ops.cuh"

namespace cuda {
    namespace kernels {

        /**
         * @brief Performs the forward pass for the SwiGLU activation function on the GPU.
         *
         * Computes: output = (Swish(input @ W_gate) * (input @ W_up)) @ W_down
         * Swish(x) = x * sigmoid(x)
         *
         * @param input The input matrix [batch_size, hidden_size].
         * @param gate_proj_weights The gate projection weights [hidden_size, intermediate_size].
         * @param up_proj_weights The up projection weights [hidden_size, intermediate_size].
         * @param down_proj_weights The down projection weights [intermediate_size, hidden_size].
         * @param gated_linear_output A buffer to store the intermediate result of Swish(input @ W_gate) [batch_size, intermediate_size].
         * @param up_proj_output A buffer to store the intermediate result of (input @ W_up) [batch_size, intermediate_size].
         * @param output The final output matrix [batch_size, hidden_size].
         */
        void swiglu_forward_kernel_launcher(
            const cuda::CudaMatrix& input,
            const cuda::CudaMatrix& gate_proj_weights,
            const cuda::CudaMatrix& up_proj_weights,
            const cuda::CudaMatrix& down_proj_weights,
            cuda::CudaMatrix& gated_linear_output,
            cuda::CudaMatrix& up_proj_output,
            cuda::CudaMatrix& output
        );

        /**
         * @brief Performs the backward pass for the SwiGLU activation function on the GPU.
         *
         * @param grad_output The gradient of the loss with respect to the SwiGLU output [batch_size, hidden_size].
         * @param input The original input to the SwiGLU layer [batch_size, hidden_size].
         * @param gate_proj_weights The gate projection weights [hidden_size, intermediate_size].
         * @param up_proj_weights The up projection weights [hidden_size, intermediate_size].
         * @param down_proj_weights The down projection weights [intermediate_size, hidden_size].
         * @param gated_linear_output The cached intermediate result from the forward pass [batch_size, intermediate_size].
         * @param up_proj_output The cached intermediate result from the forward pass [batch_size, intermediate_size].
         * @param grad_input The gradient of the loss with respect to the SwiGLU input [batch_size, hidden_size].
         * @param grad_gate_proj_weights The gradient for the gate projection weights [hidden_size, intermediate_size].
         * @param grad_up_proj_weights The gradient for the up projection weights [hidden_size, intermediate_size].
         * @param grad_down_proj_weights The gradient for the down projection weights [intermediate_size, hidden_size].
         */
        void swiglu_backward_kernel_launcher(
            const cuda::CudaMatrix& grad_output,
            const cuda::CudaMatrix& input,
            const cuda::CudaMatrix& gate_proj_weights,
            const cuda::CudaMatrix& up_proj_weights,
            const cuda::CudaMatrix& down_proj_weights,
            const cuda::CudaMatrix& gated_linear_output,
            const cuda::CudaMatrix& up_proj_output,
            cuda::CudaMatrix& grad_input,
            cuda::CudaMatrix& grad_gate_proj_weights,
            cuda::CudaMatrix& grad_up_proj_weights,
            cuda::CudaMatrix& grad_down_proj_weights
        );

    } // namespace kernels
} // namespace cuda
