#pragma once

#include "matrix_ops.cuh"
#include "cuda_matrix.hpp"
#include "../feed_forward.hpp"
#include <vector>

namespace cuda {
    namespace kernels {

        /**
         * @brief Performs the forward pass for the Mixture of Experts layer on the GPU.
         *
         * This kernel orchestrates the entire MoE forward pass on the GPU. It takes the
         * routing decisions (top-k indices and weights) from the router, dispatches
         * the hidden states for each token to the appropriate expert (FeedForward) kernels,
         * weights the outputs of the experts, and combines them to produce the final output.
         *
         * @param hidden_states The input hidden states [batch_size, hidden_size].
         * @param top_k_indices The indices of the selected experts for each token [batch_size, top_k].
         * @param top_k_weights The renormalized weights for the selected experts [batch_size, top_k].
         * @param expert_params An array of CudaParameters for each FeedForward expert.
         * @param num_experts The total number of experts.
         * @param final_output The final output matrix [batch_size, hidden_size].
         */
        void moe_forward_kernel_launcher(
            const cuda::CudaMatrix& hidden_states,
            const cuda::CudaMatrix& top_k_indices,
            const cuda::CudaMatrix& top_k_weights,
            const ::std::vector<FeedForward::CudaParameters>& expert_params,
            int num_experts,
            cuda::CudaMatrix& final_output
        );

        /**
         * @brief Performs the backward pass for the Mixture of Experts layer on the GPU.
         *
         * @param grad_output The gradient of the loss with respect to the MoE output [batch_size, hidden_size].
         * @param hidden_states The original input hidden states [batch_size, hidden_size].
         * @param top_k_indices The indices of the selected experts for each token [batch_size, top_k].
         * @param top_k_weights The renormalized weights for the selected experts [batch_size, top_k].
         * @param expert_params An array of CudaParameters for each FeedForward expert.
         * @param num_experts The total number of experts.
         * @param grad_input The gradient with respect to the input [batch_size, hidden_size].
         * @param expert_grads An array of CudaGradients for each FeedForward expert.
         */
        void moe_backward_kernel_launcher(
            const cuda::CudaMatrix& grad_output,
            const cuda::CudaMatrix& hidden_states,
            const cuda::CudaMatrix& top_k_indices,
            const cuda::CudaMatrix& top_k_weights,
            const ::std::vector<FeedForward::CudaParameters>& expert_params,
            int num_experts,
            cuda::CudaMatrix& grad_input,
            ::std::vector<FeedForward::CudaGradients>& expert_grads
        );

    } // namespace kernels
} // namespace cuda
