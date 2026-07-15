#include "../../include/cuda/router_kernel.cuh"
#include "../../include/cuda/matrix_ops.cuh"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>

namespace cuda {
    namespace kernels {

        // Simple softmax kernel
        __global__ void softmax_kernel(const float* input, float* output, int rows, int cols) {
            int row = blockIdx.x;
            if (row >= rows) return;
            
            const float* row_input = input + row * cols;
            float* row_output = output + row * cols;
            
            // Find max for numerical stability
            float max_val = row_input[0];
            for (int i = 1; i < cols; ++i) {
                max_val = fmaxf(max_val, row_input[i]);
            }
            
            // Compute exp and sum
            float sum = 0.0f;
            for (int i = 0; i < cols; ++i) {
                float exp_val = expf(row_input[i] - max_val);
                row_output[i] = exp_val;
                sum += exp_val;
            }
            
            // Normalize
            for (int i = 0; i < cols; ++i) {
                row_output[i] /= sum;
            }
        }

        // Simple top-k selection kernel
        __global__ void select_top_k_kernel(
            const float* probabilities,
            float* top_k_indices,
            float* top_k_weights,
            int rows,
            int cols,
            int k
        ) {
            int row = blockIdx.x;
            if (row >= rows) return;
            
            const float* row_probs = probabilities + row * cols;
            float* row_indices = top_k_indices + row * k;
            float* row_weights = top_k_weights + row * k;
            
            // Simple selection: find k largest values
            // For each k position, find the max not already selected
            bool selected[1024]; // Assuming max 1024 experts
            for (int i = 0; i < cols && i < 1024; ++i) selected[i] = false;
            
            float sum = 0.0f;
            for (int ki = 0; ki < k; ++ki) {
                int best_idx = -1;
                float best_val = -1e9f;
                
                for (int i = 0; i < cols; ++i) {
                    if (!selected[i] && row_probs[i] > best_val) {
                        best_val = row_probs[i];
                        best_idx = i;
                    }
                }
                
                if (best_idx >= 0) {
                    selected[best_idx] = true;
                    row_indices[ki] = static_cast<float>(best_idx);
                    row_weights[ki] = best_val;
                    sum += best_val;
                }
            }
            
            // Renormalize weights
            if (sum > 1e-9f) {
                for (int ki = 0; ki < k; ++ki) {
                    row_weights[ki] /= sum;
                }
            }
        }

        void router_forward_kernel_launcher(
            const cuda::CudaMatrix& hidden_states,
            const cuda::CudaMatrix& router_weights,
            int top_k,
            cuda::CudaMatrix& logits,
            cuda::CudaMatrix& probabilities,
            cuda::CudaMatrix& top_k_indices,
            cuda::CudaMatrix& top_k_weights
        ) {
            int num_rows = hidden_states.rows();
            int hidden_size = hidden_states.cols();
            int num_experts = router_weights.cols();
            
            // 1. Compute logits: hidden_states @ router_weights
            // Use the actual matmul function from matrix_ops.cuh
            Matrix h_host = hidden_states.to_matrix();
            Matrix w_host = router_weights.to_matrix();
            Matrix l_host(num_rows, num_experts);
            
            // This is inefficient but works for now - proper implementation would keep on GPU
            cuda::matmul(h_host, w_host, l_host);
            logits.from_host(l_host);
            
            // 2. Compute softmax
            dim3 softmax_grid(num_rows);
            dim3 softmax_block(1);
            softmax_kernel<<<softmax_grid, softmax_block>>>(
                logits.data(), probabilities.data(), num_rows, num_experts
            );
            cudaDeviceSynchronize();
            
            // 3. Select top-k and renormalize
            dim3 topk_grid(num_rows);
            dim3 topk_block(1);
            select_top_k_kernel<<<topk_grid, topk_block>>>(
                probabilities.data(),
                top_k_indices.data(),
                top_k_weights.data(),
                num_rows,
                num_experts,
                top_k
            );
            cudaDeviceSynchronize();
        }

    } // namespace kernels
} // namespace cuda
