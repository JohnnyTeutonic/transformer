#include "../../include/cuda/router_kernel.cuh"
#include "../../include/cuda/matrix_ops.cuh"
#include <cub/cub.cuh>

namespace cuda {
    namespace kernels {

        // Kernel to renormalize the top-k weights and store indices
        __global__ void renormalize_top_k_kernel(
            const float* all_probs,
            const int* top_k_indices_all,
            float* top_k_weights,
            float* top_k_indices_float, // CudaMatrix uses float
            int rows,
            int cols,
            int top_k
        ) {
            int row = blockIdx.x;
            int tid = threadIdx.x;

            if (row >= rows) return;

            // Step 1: Sum the probabilities of the top-k experts for the current row
            extern __shared__ float s_top_k_probs[];
            if (tid < top_k) {
                int expert_idx = top_k_indices_all[row * cols + tid]; // These are original indices
                s_top_k_probs[tid] = all_probs[row * cols + expert_idx];
            }
            __syncthreads();

            // Parallel reduction to find the sum
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s && (tid + s) < top_k) {
                    s_top_k_probs[tid] += s_top_k_probs[tid + s];
                }
                __syncthreads();
            }

            // The sum is in s_top_k_probs[0]
            float sum_top_k_probs = s_top_k_probs[0];
            
            // Avoid division by zero
            if (sum_top_k_probs < 1e-9) {
                sum_top_k_probs = 1.0f;
            }

            // Step 2: Renormalize and store weights and indices
            if (tid < top_k) {
                int original_expert_idx = top_k_indices_all[row * cols + tid];
                float prob = all_probs[row * cols + original_expert_idx];
                top_k_weights[row * top_k + tid] = prob / sum_top_k_probs;
                top_k_indices_float[row * top_k + tid] = static_cast<float>(original_expert_idx);
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
            // 1. Compute logits: hidden_states @ router_weights
            cuda::kernels::matrix_multiply(hidden_states, router_weights, logits);

            // 2. Compute softmax over logits to get probabilities
            cuda::kernels::softmax(logits, probabilities);

            // 3. Select top-k experts for each token using CUB
            size_t num_rows = probabilities.rows();
            size_t num_cols = probabilities.cols();

            // CUB requires temporary storage
            void* d_temp_storage = nullptr;
            size_t temp_storage_bytes = 0;
            
            cuda::CudaMatrix top_k_vals_temp(num_rows, top_k); // To store the top-k probabilities
            cuda::CudaMatrix top_k_indices_temp(num_rows, top_k); // CUB needs int*, we'll use a float CudaMatrix and cast

            // CUB works on each row independently. We can wrap this in a loop or a custom kernel if needed,
            // but for simplicity, we'll process row by row on the stream. A batch version would be more optimal.
            // For now, a simple loop:
            for(size_t i = 0; i < num_rows; ++i) {
                // Get temporary storage size
                cub::DeviceSelect::TopK(d_temp_storage, temp_storage_bytes, probabilities.data() + i * num_cols, top_k_vals_temp.data() + i * top_k, reinterpret_cast<int*>(top_k_indices.data()) + i * top_k, num_cols, top_k);
                
                // Allocate temporary storage
                cuda::CudaVector<char> temp_storage(temp_storage_bytes);

                // Run TopK
                cub::DeviceSelect::TopK(temp_storage.data(), temp_storage_bytes, probabilities.data() + i * num_cols, top_k_vals_temp.data() + i * top_k, reinterpret_cast<int*>(top_k_indices.data()) + i * top_k, num_cols, top_k);
            }


            // 4. Renormalize the weights of the top-k experts
            dim3 grid_dim(num_rows);
            dim3 block_dim(top_k);
            size_t shared_mem_size = top_k * sizeof(float);
            renormalize_top_k_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
                probabilities.data(),
                reinterpret_cast<int*>(top_k_indices.data()),
                top_k_weights.data(),
                top_k_indices.data(), // Write float indices here
                num_rows,
                num_cols,
                top_k
            );
        }

    } // namespace kernels
} // namespace cuda
