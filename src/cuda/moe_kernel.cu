#include "../../include/cuda/moe_kernel.cuh"
#include "../../include/cuda/swiglu_kernel.cuh"
#include "../../include/feed_forward.hpp" // For CudaParameters struct
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>

namespace cuda {
    namespace kernels {
        
        // Structure to hold dispatch information for each expert
        struct ExpertDispatchInfo {
            int expert_id;
            int token_count;
            int start_offset;
        };

        // Kernel to count tokens assigned to each expert
        __global__ void count_tokens_per_expert_kernel(
            const float* top_k_indices,
            int* expert_token_counts,
            int batch_size,
            int top_k,
            int num_experts
        ) {
            int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (token_idx < batch_size) {
                for (int k = 0; k < top_k; ++k) {
                    int expert_idx = static_cast<int>(top_k_indices[token_idx * top_k + k]);
                    if (expert_idx >= 0 && expert_idx < num_experts) {
                        atomicAdd(&expert_token_counts[expert_idx], 1);
                    }
                }
            }
        }

        // Kernel to create dispatch indices for token-expert pairs
        __global__ void create_dispatch_indices_kernel(
            const float* top_k_indices,
            const float* top_k_weights,
            int* dispatch_token_indices,
            int* dispatch_expert_indices,
            float* dispatch_weights,
            int* expert_offsets,
            int* expert_counters,
            int batch_size,
            int top_k
        ) {
            int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (token_idx < batch_size) {
                for (int k = 0; k < top_k; ++k) {
                    int expert_idx = static_cast<int>(top_k_indices[token_idx * top_k + k]);
                    float weight = top_k_weights[token_idx * top_k + k];
                    
                    // Get position for this expert using atomic increment
                    int pos = atomicAdd(&expert_counters[expert_idx], 1);
                    int global_pos = expert_offsets[expert_idx] + pos;
                    
                    dispatch_token_indices[global_pos] = token_idx;
                    dispatch_expert_indices[global_pos] = expert_idx;
                    dispatch_weights[global_pos] = weight;
                }
            }
        }

        // Kernel to gather tokens for a specific expert
        __global__ void gather_tokens_for_expert_kernel(
            const float* hidden_states,
            float* expert_input_batch,
            const int* token_indices,
            int start_idx,
            int token_count,
            int hidden_size
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int total_elements = token_count * hidden_size;
            
            if (idx < total_elements) {
                int token_pos = idx / hidden_size;
                int feature_idx = idx % hidden_size;
                int source_token_idx = token_indices[start_idx + token_pos];
                
                expert_input_batch[idx] = hidden_states[source_token_idx * hidden_size + feature_idx];
            }
        }

        // Kernel to scatter expert outputs back to final output
        __global__ void scatter_expert_outputs_kernel(
            const float* expert_output_batch,
            float* final_output,
            const int* token_indices,
            const float* weights,
            int start_idx,
            int token_count,
            int hidden_size
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int total_elements = token_count * hidden_size;
            
            if (idx < total_elements) {
                int token_pos = idx / hidden_size;
                int feature_idx = idx % hidden_size;
                int target_token_idx = token_indices[start_idx + token_pos];
                float weight = weights[start_idx + token_pos];
                
                // Atomic add to handle multiple experts contributing to same token
                atomicAdd(&final_output[target_token_idx * hidden_size + feature_idx], 
                         expert_output_batch[idx] * weight);
            }
        }

        void moe_forward_kernel_launcher(
            const cuda::CudaMatrix& hidden_states,
            const cuda::CudaMatrix& top_k_indices,
            const cuda::CudaMatrix& top_k_weights,
            const ::std::vector<FeedForward::CudaParameters>& expert_params,
            int num_experts,
            cuda::CudaMatrix& final_output
        ) {
            int batch_size = hidden_states.rows();
            int hidden_size = hidden_states.cols();
            int top_k = top_k_indices.cols();
            int total_assignments = batch_size * top_k;

            // Initialize output to zeros
            final_output.fill(0.0f);

            // Step 1: Count tokens per expert
            thrust::device_vector<int> expert_token_counts(num_experts, 0);
            
            dim3 block_size(256);
            dim3 grid_size((batch_size + block_size.x - 1) / block_size.x);
            
            count_tokens_per_expert_kernel<<<grid_size, block_size>>>(
                top_k_indices.data(),
                thrust::raw_pointer_cast(expert_token_counts.data()),
                batch_size,
                top_k,
                num_experts
            );
            cudaDeviceSynchronize();

            // Step 2: Calculate cumulative offsets for each expert
            thrust::device_vector<int> expert_offsets(num_experts + 1);
            thrust::exclusive_scan(expert_token_counts.begin(), expert_token_counts.end(), 
                                 expert_offsets.begin());

            // Step 3: Create dispatch arrays
            thrust::device_vector<int> dispatch_token_indices(total_assignments);
            thrust::device_vector<int> dispatch_expert_indices(total_assignments);
            thrust::device_vector<float> dispatch_weights(total_assignments);
            thrust::device_vector<int> expert_counters(num_experts, 0);

            create_dispatch_indices_kernel<<<grid_size, block_size>>>(
                top_k_indices.data(),
                top_k_weights.data(),
                thrust::raw_pointer_cast(dispatch_token_indices.data()),
                thrust::raw_pointer_cast(dispatch_expert_indices.data()),
                thrust::raw_pointer_cast(dispatch_weights.data()),
                thrust::raw_pointer_cast(expert_offsets.data()),
                thrust::raw_pointer_cast(expert_counters.data()),
                batch_size,
                top_k
            );
            cudaDeviceSynchronize();

            // Step 4: Process each expert that has tokens assigned
            for (int expert_id = 0; expert_id < num_experts; ++expert_id) {
                int token_count = expert_token_counts[expert_id];
                if (token_count == 0) continue;

                int start_offset = expert_offsets[expert_id];
                
                // Allocate memory for this expert's batch
                cuda::CudaMatrix expert_input_batch(token_count, hidden_size);
                cuda::CudaMatrix expert_output_batch(token_count, hidden_size);
                
                // Gather tokens for this expert
                int total_elements = token_count * hidden_size;
                dim3 gather_grid((total_elements + 255) / 256);
                dim3 gather_block(256);
                
                gather_tokens_for_expert_kernel<<<gather_grid, gather_block>>>(
                    hidden_states.data(),
                    expert_input_batch.data(),
                    thrust::raw_pointer_cast(dispatch_token_indices.data()),
                    start_offset,
                    token_count,
                    hidden_size
                );
                cudaDeviceSynchronize();

                // Run expert forward pass using existing SwiGLU kernel
                const auto& expert_param = expert_params[expert_id];
                
                // Create intermediate buffers for SwiGLU
                int intermediate_size = expert_param.gate_proj_weights.cols();
                cuda::CudaMatrix gated_linear_output(token_count, intermediate_size);
                cuda::CudaMatrix up_proj_output(token_count, intermediate_size);
                
                swiglu_forward_kernel_launcher(
                    expert_input_batch,
                    expert_param.gate_proj_weights,
                    expert_param.up_proj_weights,
                    expert_param.down_proj_weights,
                    gated_linear_output,
                    up_proj_output,
                    expert_output_batch
                );

                // Scatter expert outputs back to final output
                scatter_expert_outputs_kernel<<<gather_grid, gather_block>>>(
                    expert_output_batch.data(),
                    final_output.data(),
                    thrust::raw_pointer_cast(dispatch_token_indices.data()),
                    thrust::raw_pointer_cast(dispatch_weights.data()),
                    start_offset,
                    token_count,
                    hidden_size
                );
                cudaDeviceSynchronize();
            }
        }

    } // namespace kernels
} // namespace cuda
