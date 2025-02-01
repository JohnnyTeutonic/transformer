#include "layernorm_kernels.cuh"

namespace cuda {

__global__ void layer_norm_stats_kernel(const float* input, float* mean, float* variance,
                                        const int hidden_size, const int batch_size) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const float MIN_VAR = 1e-6f;

    extern __shared__ float shared_mem[];
    float* shared_sum = shared_mem;
    float* shared_sq_sum = &shared_mem[blockDim.x];

    if (batch_idx >= batch_size) return;

    // Initialize shared memory
    shared_sum[tid] = 0.0f;
    shared_sq_sum[tid] = 0.0f;

    // Each thread processes multiple elements
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = input[batch_idx * hidden_size + i];
        shared_sum[tid] += val;
        shared_sq_sum[tid] += val * val;
    }
    __syncthreads();

    // Parallel reduction in shared memory
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_sq_sum[tid] += shared_sq_sum[tid + stride];
        }
        __syncthreads();
    }

    // Write results to global memory
    if (tid == 0) {
        mean[batch_idx] = shared_sum[0] / hidden_size;
        float var = (shared_sq_sum[0] / hidden_size) - (mean[batch_idx] * mean[batch_idx]);
        variance[batch_idx] = max(var, MIN_VAR);
    }
}

__global__ void layer_norm_kernel(const float* input, const float* mean, const float* variance,
                                  const float* gamma, const float* beta, float* output,
                                  const int hidden_size, const int batch_size, const float eps) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    const float mean_val = mean[batch_idx];
    const float var_val = variance[batch_idx];
    const float std_dev = sqrt(var_val + eps);

    // Each thread processes multiple elements
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        const int idx = batch_idx * hidden_size + i;
        const float val = input[idx];
        const float normalized = (val - mean_val) / std_dev;
        output[idx] = gamma[i] * normalized + beta[i];
    }
}

} // namespace cuda