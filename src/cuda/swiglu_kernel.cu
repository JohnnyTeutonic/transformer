#include "../../include/cuda/swiglu_kernel.cuh"
#include <cuda_runtime.h>

namespace cuda {
    namespace kernels {

        // Sigmoid and Swish device functions
        __device__ float sigmoidf(float x) {
            return 1.0f / (1.0f + expf(-x));
        }

        __device__ float swishf(float x) {
            return x * sigmoidf(x);
        }

        // Swish derivative for backward pass
        __device__ float swish_gradf(float x) {
            float sig_x = sigmoidf(x);
            return sig_x * (1.0f + x * (1.0f - sig_x));
        }
        
        // Kernel for element-wise Swish activation
        __global__ void apply_swish_kernel(float* gated_linear_output, const float* gate_proj_output, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                gated_linear_output[idx] = swishf(gate_proj_output[idx]);
            }
        }

        // Kernel for element-wise Swish gradient
        __global__ void apply_swish_grad_kernel(float* grad_swish, const float* gate_proj_output, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                grad_swish[idx] = swish_gradf(gate_proj_output[idx]);
            }
        }

        void swiglu_forward_kernel_launcher(
            const cuda::CudaMatrix& input,
            const cuda::CudaMatrix& gate_proj_weights,
            const cuda::CudaMatrix& up_proj_weights,
            const cuda::CudaMatrix& down_proj_weights,
            cuda::CudaMatrix& gated_linear_output,
            cuda::CudaMatrix& up_proj_output,
            cuda::CudaMatrix& output
        ) {
            // 1. Gate projection: gate_proj_output = input @ gate_proj_weights
            cuda::CudaMatrix gate_proj_output(input.rows(), gate_proj_weights.cols());
            cuda::kernels::matrix_multiply(input, gate_proj_weights, gate_proj_output);

            // 2. Apply Swish activation element-wise
            int size = gate_proj_output.rows() * gate_proj_output.cols();
            dim3 block_size(256);
            dim3 grid_size((size + block_size.x - 1) / block_size.x);
            apply_swish_kernel<<<grid_size, block_size>>>(gated_linear_output.data(), gate_proj_output.data(), size);
            
            // 3. Up projection: up_proj_output = input @ up_proj_weights
            cuda::kernels::matrix_multiply(input, up_proj_weights, up_proj_output);

            // 4. Element-wise multiplication: hadamard_product = gated_linear_output * up_proj_output
            cuda::CudaMatrix hadamard_product(input.rows(), gate_proj_weights.cols());
            cuda::kernels::element_wise_multiply(gated_linear_output, up_proj_output, hadamard_product);
            
            // 5. Down projection: output = hadamard_product @ down_proj_weights
            cuda::kernels::matrix_multiply(hadamard_product, down_proj_weights, output);
        }

        void swiglu_backward_kernel_launcher(
            const cuda::CudaMatrix& grad_output,
            const cuda::CudaMatrix& input,
            const cuda::CudaMatrix& gate_proj_weights,
            const cuda::CudaMatrix& up_proj_weights,
            const cuda::CudaMatrix& down_proj_weights,
            const cuda::CudaMatrix& gated_linear_output, // Swish(input @ W_gate)
            const cuda::CudaMatrix& up_proj_output,      // input @ W_up
            cuda::CudaMatrix& grad_input,
            cuda::CudaMatrix& grad_gate_proj_weights,
            cuda::CudaMatrix& grad_up_proj_weights,
            cuda::CudaMatrix& grad_down_proj_weights
        ) {
            // Let F = (Swish(X @ Wg) * (X @ Wu)) @ Wd
            // Let A = X @ Wg, S = Swish(A)
            // Let B = X @ Wu
            // Let H = S * B (Hadamard)
            // F = H @ Wd
            
            // 1. Gradient w.r.t H (dL/dH)
            // dL/dH = dL/dF @ Wd.T
            cuda::CudaMatrix grad_hadamard(grad_output.rows(), down_proj_weights.rows());
            cuda::kernels::matrix_multiply_transpose_rhs(grad_output, down_proj_weights, grad_hadamard);

            // 2. Gradient w.r.t Wd (dL/dWd)
            // dL/dWd = H.T @ dL/dF
            cuda::CudaMatrix hadamard_product(gated_linear_output.rows(), gated_linear_output.cols());
            cuda::kernels::element_wise_multiply(gated_linear_output, up_proj_output, hadamard_product);
            cuda::kernels::matrix_multiply_transpose_lhs(hadamard_product, grad_output, grad_down_proj_weights);

            // 3. Gradient w.r.t S (dL/dS) and B (dL/dB)
            // dL/dS = dL/dH * B
            // dL/dB = dL/dH * S
            cuda::CudaMatrix grad_gated_linear(grad_hadamard.rows(), grad_hadamard.cols());
            cuda::CudaMatrix grad_up_proj(grad_hadamard.rows(), grad_hadamard.cols());
            cuda::kernels::element_wise_multiply(grad_hadamard, up_proj_output, grad_gated_linear);
            cuda::kernels::element_wise_multiply(grad_hadamard, gated_linear_output, grad_up_proj);

            // 4. Gradient w.r.t Wu (dL/dWu)
            // dL/dWu = X.T @ dL/dB
            cuda::kernels::matrix_multiply_transpose_lhs(input, grad_up_proj, grad_up_proj_weights);
            
            // 5. Gradient w.r.t A (dL/dA)
            // dL/dA = dL/dS * dS/dA
            // dS/dA = swish_grad(A)
            cuda::CudaMatrix gate_proj_output(input.rows(), gate_proj_weights.cols());
            cuda::kernels::matrix_multiply(input, gate_proj_weights, gate_proj_output); // Recalculate A = X @ Wg
            
            cuda::CudaMatrix grad_swish(gate_proj_output.rows(), gate_proj_output.cols());
            int size = gate_proj_output.rows() * gate_proj_output.cols();
            dim3 block_size(256);
            dim3 grid_size((size + block_size.x - 1) / block_size.x);
            apply_swish_grad_kernel<<<grid_size, block_size>>>(grad_swish.data(), gate_proj_output.data(), size);

            cuda::CudaMatrix grad_gate_proj_logits(grad_gated_linear.rows(), grad_gated_linear.cols());
            cuda::kernels::element_wise_multiply(grad_gated_linear, grad_swish, grad_gate_proj_logits);
            
            // 6. Gradient w.r.t Wg (dL/dWg)
            // dL/dWg = X.T @ dL/dA
            cuda::kernels::matrix_multiply_transpose_lhs(input, grad_gate_proj_logits, grad_gate_proj_weights);

            // 7. Gradient w.r.t X (dL/dX)
            // dL/dX = (dL/dA @ Wg.T) + (dL/dB @ Wu.T)
            cuda::CudaMatrix grad_input_from_gate(grad_gate_proj_logits.rows(), gate_proj_weights.rows());
            cuda::CudaMatrix grad_input_from_up(grad_up_proj.rows(), up_proj_weights.rows());
            cuda::kernels::matrix_multiply_transpose_rhs(grad_gate_proj_logits, gate_proj_weights, grad_input_from_gate);
            cuda::kernels::matrix_multiply_transpose_rhs(grad_up_proj, up_proj_weights, grad_input_from_up);
            cuda::kernels::matrix_add(grad_input_from_gate, grad_input_from_up, grad_input);
        }


    } // namespace kernels
} // namespace cuda
