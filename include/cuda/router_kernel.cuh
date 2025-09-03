#pragma once

#include "matrix.cuh"

namespace cuda {
    namespace kernels {

        /**
         * @brief Performs the forward pass for the MoE router on the GPU.
         *
         * This kernel computes the router logits, applies softmax, and performs an efficient
         * top-k selection to determine which experts should process each token.
         *
         * @param hidden_states The input hidden states from the transformer layer [batch_size, hidden_size].
         * @param router_weights The router's weight matrix [hidden_size, num_experts].
         * @param top_k The number of experts to select for each token.
         * @param logits A buffer to store the computed logits [batch_size, num_experts].
         * @param probabilities A buffer to store the softmax probabilities [batch_size, num_experts].
         * @param top_k_indices The output matrix of selected expert indices [batch_size, top_k].
         * @param top_k_weights The output matrix of renormalized weights for the selected experts [batch_size, top_k].
         */
        void router_forward_kernel_launcher(
            const cuda::CudaMatrix& hidden_states,
            const cuda::CudaMatrix& router_weights,
            int top_k,
            cuda::CudaMatrix& logits,
            cuda::CudaMatrix& probabilities,
            cuda::CudaMatrix& top_k_indices,
            cuda::CudaMatrix& top_k_weights
        );

    } // namespace kernels
} // namespace cuda
