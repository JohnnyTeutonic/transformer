#include "../../include/cuda/loss_kernels.cuh"
#include "../../include/cuda/cuda_check.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <stdexcept>
#include <string>

namespace cuda {

// ============================================================================
// PERSISTENT MEMORY POOL - Avoid cudaMalloc/cudaFree in hot path
// ============================================================================
struct LossKernelMemory {
    float* d_logits = nullptr;
    int* d_targets = nullptr;
    float* d_losses = nullptr;
    float* d_loss = nullptr;
    float* d_grad_output = nullptr;
    size_t allocated_positions = 0;
    size_t allocated_vocab = 0;
    
    void ensure_allocated(int num_positions, int vocab_size) {
        size_t needed_positions = static_cast<size_t>(num_positions);
        size_t needed_vocab = static_cast<size_t>(vocab_size);
        
        if (needed_positions > allocated_positions || needed_vocab > allocated_vocab) {
            free_all();
            
            CUDA_CHECK(cudaMalloc(&d_logits, needed_positions * needed_vocab * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_targets, needed_positions * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_losses, needed_positions * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_grad_output, needed_positions * needed_vocab * sizeof(float)));
            
            allocated_positions = needed_positions;
            allocated_vocab = needed_vocab;
        }
    }
    
    void free_all() {
        if (d_logits) { cudaFree(d_logits); d_logits = nullptr; }
        if (d_targets) { cudaFree(d_targets); d_targets = nullptr; }
        if (d_losses) { cudaFree(d_losses); d_losses = nullptr; }
        if (d_loss) { cudaFree(d_loss); d_loss = nullptr; }
        if (d_grad_output) { cudaFree(d_grad_output); d_grad_output = nullptr; }
        allocated_positions = 0;
        allocated_vocab = 0;
    }
    
    ~LossKernelMemory() { free_all(); }
};

static LossKernelMemory g_loss_memory;

// ============================================================================
// KERNEL: Fused softmax + cross-entropy + gradient computation
// Each block handles one position, threads cooperate on vocab dimension
// ============================================================================
__global__ void softmax_cross_entropy_grad_kernel(
    const float* logits,      // [num_positions x vocab_size]
    const int* targets,       // [num_positions]
    float* losses,            // [num_positions] - per-position losses
    float* grad_output,       // [num_positions x vocab_size] - gradient output
    int num_positions,
    int vocab_size
) {
    int pos = blockIdx.x;
    if (pos >= num_positions) return;
    
    // Shared memory for reduction
    extern __shared__ float shared[];
    float* shared_max = shared;
    float* shared_sum = shared + blockDim.x;
    
    int tid = threadIdx.x;
    int target = targets[pos];
    
    // Base pointers
    const float* pos_logits = logits + pos * vocab_size;
    float* pos_grad = grad_output + pos * vocab_size;
    
    // ========== STEP 1: Find max logit (for numerical stability) ==========
    float local_max = -1e30f;
    for (int v = tid; v < vocab_size; v += blockDim.x) {
        local_max = fmaxf(local_max, pos_logits[v]);
    }
    shared_max[tid] = local_max;
    __syncthreads();
    
    // Reduce to find global max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }
    float max_logit = shared_max[0];
    
    // ========== STEP 2: Compute sum of exp(logit - max) ==========
    float local_sum = 0.0f;
    for (int v = tid; v < vocab_size; v += blockDim.x) {
        local_sum += expf(pos_logits[v] - max_logit);
    }
    shared_sum[tid] = local_sum;
    __syncthreads();
    
    // Reduce sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    float sum_exp = shared_sum[0];
    float inv_sum = 1.0f / (sum_exp + 1e-10f);
    
    // ========== STEP 3: Compute loss AND gradient in one pass ==========
    // grad = softmax(logits) - one_hot(target)
    for (int v = tid; v < vocab_size; v += blockDim.x) {
        float softmax_v = expf(pos_logits[v] - max_logit) * inv_sum;
        float target_v = (v == target) ? 1.0f : 0.0f;
        pos_grad[v] = softmax_v - target_v;
    }
    
    // Compute cross-entropy loss (only thread 0)
    if (tid == 0) {
        float log_prob = pos_logits[target] - max_logit - logf(sum_exp + 1e-10f);
        losses[pos] = -log_prob;
    }
}

// ============================================================================
// KERNEL: Reduce per-position losses to total loss
// ============================================================================
__global__ void reduce_losses_kernel(
    const float* losses,
    float* total_loss,
    int num_positions
) {
    extern __shared__ float shared[];
    
    int tid = threadIdx.x;
    float local_sum = 0.0f;
    
    for (int i = tid; i < num_positions; i += blockDim.x) {
        local_sum += losses[i];
    }
    shared[tid] = local_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        *total_loss = shared[0] / num_positions;
    }
}

// ============================================================================
// HOST FUNCTION: Fused softmax + cross-entropy (no gradient)
// ============================================================================
void fused_softmax_cross_entropy(
    const float* d_logits,
    const int* d_targets,
    float* d_loss,
    int num_positions,
    int vocab_size,
    cudaStream_t stream
) {
    g_loss_memory.ensure_allocated(num_positions, vocab_size);
    
    int block_size = 256;
    int shared_mem = 2 * block_size * sizeof(float);
    
    softmax_cross_entropy_grad_kernel<<<num_positions, block_size, shared_mem, stream>>>(
        d_logits, d_targets, g_loss_memory.d_losses, g_loss_memory.d_grad_output,
        num_positions, vocab_size
    );
    CUDA_CHECK(cudaGetLastError());
    
    reduce_losses_kernel<<<1, 256, 256 * sizeof(float), stream>>>(
        g_loss_memory.d_losses, d_loss, num_positions
    );
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// HOST WRAPPER: Computes loss only (uses memory pool)
// ============================================================================
float compute_cross_entropy_loss_cuda(
    const float* h_logits,
    const int* h_targets,
    int num_positions,
    int vocab_size
) {
    g_loss_memory.ensure_allocated(num_positions, vocab_size);
    
    size_t logits_size = num_positions * vocab_size * sizeof(float);
    size_t targets_size = num_positions * sizeof(int);
    
    // Copy to pre-allocated device memory
    CUDA_CHECK(cudaMemcpy(g_loss_memory.d_logits, h_logits, logits_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_loss_memory.d_targets, h_targets, targets_size, cudaMemcpyHostToDevice));
    
    // Compute loss + gradient (gradient stored for later retrieval)
    int block_size = 256;
    int shared_mem = 2 * block_size * sizeof(float);
    
    softmax_cross_entropy_grad_kernel<<<num_positions, block_size, shared_mem, 0>>>(
        g_loss_memory.d_logits, g_loss_memory.d_targets,
        g_loss_memory.d_losses, g_loss_memory.d_grad_output,
        num_positions, vocab_size
    );
    CUDA_CHECK(cudaGetLastError());
    
    reduce_losses_kernel<<<1, 256, 256 * sizeof(float), 0>>>(
        g_loss_memory.d_losses, g_loss_memory.d_loss, num_positions
    );
    CUDA_CHECK(cudaGetLastError());
    
    // Copy loss back
    float h_loss;
    CUDA_CHECK(cudaMemcpy(&h_loss, g_loss_memory.d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    
    return h_loss;
}

// ============================================================================
// HOST WRAPPER: Computes loss AND returns gradient (avoids 655MB target_dist)
// ============================================================================
float compute_cross_entropy_loss_and_grad_cuda(
    const float* h_logits,
    const int* h_targets,
    float* h_grad_output,     // OUTPUT: [num_positions x vocab_size]
    int num_positions,
    int vocab_size
) {
    g_loss_memory.ensure_allocated(num_positions, vocab_size);
    
    size_t logits_size = num_positions * vocab_size * sizeof(float);
    size_t targets_size = num_positions * sizeof(int);
    
    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(g_loss_memory.d_logits, h_logits, logits_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_loss_memory.d_targets, h_targets, targets_size, cudaMemcpyHostToDevice));
    
    // Compute loss + gradient in one kernel
    int block_size = 256;
    int shared_mem = 2 * block_size * sizeof(float);
    
    softmax_cross_entropy_grad_kernel<<<num_positions, block_size, shared_mem, 0>>>(
        g_loss_memory.d_logits, g_loss_memory.d_targets,
        g_loss_memory.d_losses, g_loss_memory.d_grad_output,
        num_positions, vocab_size
    );
    CUDA_CHECK(cudaGetLastError());
    
    reduce_losses_kernel<<<1, 256, 256 * sizeof(float), 0>>>(
        g_loss_memory.d_losses, g_loss_memory.d_loss, num_positions
    );
    CUDA_CHECK(cudaGetLastError());
    
    // Copy results back
    float h_loss;
    CUDA_CHECK(cudaMemcpy(&h_loss, g_loss_memory.d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_grad_output, g_loss_memory.d_grad_output, logits_size, cudaMemcpyDeviceToHost));
    
    return h_loss;
}

// ============================================================================
// ZERO-COPY VERSION: Keeps gradient on device (avoids 655MB D2H transfer!)
// ============================================================================
float compute_cross_entropy_loss_keep_grad_on_device(
    const float* h_logits,
    const int* h_targets,
    int num_positions,
    int vocab_size
) {
    g_loss_memory.ensure_allocated(num_positions, vocab_size);
    
    size_t logits_size = num_positions * vocab_size * sizeof(float);
    size_t targets_size = num_positions * sizeof(int);
    
    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(g_loss_memory.d_logits, h_logits, logits_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_loss_memory.d_targets, h_targets, targets_size, cudaMemcpyHostToDevice));
    
    // Compute loss + gradient in one kernel
    int block_size = 256;
    int shared_mem = 2 * block_size * sizeof(float);
    
    softmax_cross_entropy_grad_kernel<<<num_positions, block_size, shared_mem, 0>>>(
        g_loss_memory.d_logits, g_loss_memory.d_targets,
        g_loss_memory.d_losses, g_loss_memory.d_grad_output,
        num_positions, vocab_size
    );
    CUDA_CHECK(cudaGetLastError());
    
    reduce_losses_kernel<<<1, 256, 256 * sizeof(float), 0>>>(
        g_loss_memory.d_losses, g_loss_memory.d_loss, num_positions
    );
    CUDA_CHECK(cudaGetLastError());
    
    // Only copy loss back (4 bytes), NOT gradient (655MB)!
    float h_loss;
    CUDA_CHECK(cudaMemcpy(&h_loss, g_loss_memory.d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    
    // Gradient stays on device at g_loss_memory.d_grad_output
    return h_loss;
}

// Get device pointer to gradient (valid until next loss computation)
float* get_device_grad_logits() {
    return g_loss_memory.d_grad_output;
}

// Get device pointer to logits
float* get_device_logits() {
    return g_loss_memory.d_logits;
}

// ============================================================================
// DEVICE-RESIDENT PROJECTION: logits = hidden @ projection + bias
// Keeps logits on the device, feeding directly into the loss kernel.
// ============================================================================

// Add per-row bias: logits[pos, v] += bias[v]
__global__ void add_row_bias_kernel(
    float* logits,        // [num_positions x vocab_size]
    const float* bias,    // [vocab_size]
    int num_positions,
    int vocab_size
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)num_positions * vocab_size;
    if (idx < total) {
        int v = (int)(idx % vocab_size);
        logits[idx] += bias[v];
    }
}

// Reuse the shared cuBLAS handle owned by matrix_ops.cu (creating a second
// handle here caused a teardown crash). Declared in the cuda namespace there.
cublasHandle_t get_cublas_handle();

// Faithful GPU port of the CPU Adam update in LanguageModelHead::backward_pass.
// Element-wise; bias correction factors (1 - beta^t) are precomputed on host.
__global__ void lmhead_adam_update_kernel(
    float* w, float* m, float* v, const float* g, long long n,
    float beta1, float beta2, float eps, float lr,
    float bias_corr1, float bias_corr2,
    float clip_threshold, float max_update, float max_allowed
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float grad = g[idx];
    if (!isfinite(grad)) return;  // CPU 'continue' leaves m/v/w unchanged

    if (fabsf(grad) > clip_threshold) {
        grad *= clip_threshold / fabsf(grad);
    }

    float new_m = beta1 * m[idx] + (1.0f - beta1) * grad;
    float new_v = beta2 * v[idx] + (1.0f - beta2) * grad * grad;

    float m_hat = new_m / bias_corr1;
    float v_hat = new_v / bias_corr2;

    m[idx] = new_m;
    v[idx] = new_v;

    float update = lr * m_hat / (sqrtf(v_hat) + eps);
    if (fabsf(update) > max_update) {
        update *= max_update / fabsf(update);
    }

    float new_value = w[idx] - update;
    if (fabsf(new_value) > max_allowed) {
        new_value = copysignf(max_allowed, new_value);
    }
    w[idx] = new_value;
}

// Per-step hidden buffer + device-resident LM head weights and Adam state.
namespace {
    float* g_d_hidden = nullptr;       // [num_positions x hidden]
    size_t g_hidden_capacity = 0;      // elements

    // Resident, authoritative-during-training LM head weights + Adam state.
    float* g_d_proj = nullptr;         // [hidden x vocab]
    float* g_d_bias = nullptr;         // [vocab]
    float* g_d_m_proj = nullptr;       // [hidden x vocab]
    float* g_d_v_proj = nullptr;       // [hidden x vocab]
    size_t g_proj_capacity = 0;        // elements (hidden*vocab)
    size_t g_bias_capacity = 0;        // elements (vocab)

    // Backward scratch.
    float* g_d_grad_proj = nullptr;    // [hidden x vocab]
    float* g_d_grad_hidden = nullptr;  // [num_positions x hidden]
    size_t g_grad_hidden_capacity = 0; // elements (num_positions*hidden)

    bool g_lmhead_initialized = false;

    void ensure_hidden_buffer(size_t hidden_elems) {
        if (hidden_elems > g_hidden_capacity) {
            if (g_d_hidden) cudaFree(g_d_hidden);
            CUDA_CHECK(cudaMalloc(&g_d_hidden, hidden_elems * sizeof(float)));
            g_hidden_capacity = hidden_elems;
        }
    }
}

void lmhead_invalidate_weights() {
    g_lmhead_initialized = false;
}

void lmhead_ensure_weights(
    const float* h_projection,
    const float* h_bias,
    int hidden_size,
    int vocab_size
) {
    if (g_lmhead_initialized) return;

    size_t proj_elems = (size_t)hidden_size * vocab_size;
    size_t bias_elems = (size_t)vocab_size;

    if (proj_elems > g_proj_capacity) {
        if (g_d_proj)   cudaFree(g_d_proj);
        if (g_d_m_proj) cudaFree(g_d_m_proj);
        if (g_d_v_proj) cudaFree(g_d_v_proj);
        CUDA_CHECK(cudaMalloc(&g_d_proj,   proj_elems * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&g_d_m_proj, proj_elems * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&g_d_v_proj, proj_elems * sizeof(float)));
        g_proj_capacity = proj_elems;
    }
    if (bias_elems > g_bias_capacity) {
        if (g_d_bias) cudaFree(g_d_bias);
        CUDA_CHECK(cudaMalloc(&g_d_bias, bias_elems * sizeof(float)));
        g_bias_capacity = bias_elems;
    }

    CUDA_CHECK(cudaMemcpy(g_d_proj, h_projection, proj_elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_d_bias, h_bias, bias_elems * sizeof(float), cudaMemcpyHostToDevice));
    // Zero Adam state.
    CUDA_CHECK(cudaMemset(g_d_m_proj, 0, proj_elems * sizeof(float)));
    CUDA_CHECK(cudaMemset(g_d_v_proj, 0, proj_elems * sizeof(float)));

    g_lmhead_initialized = true;
}

void project_logits_to_device(
    const float* h_hidden,
    int num_positions,
    int hidden_size,
    int vocab_size
) {
    g_loss_memory.ensure_allocated(num_positions, vocab_size);

    size_t hidden_elems = (size_t)num_positions * hidden_size;
    ensure_hidden_buffer(hidden_elems);

    // Upload only the (small) hidden states; the projection/bias are resident.
    CUDA_CHECK(cudaMemcpy(g_d_hidden, h_hidden, hidden_elems * sizeof(float), cudaMemcpyHostToDevice));

    // logits = hidden @ projection. Match cuda::matmul's column-major convention:
    //   cublasSgemm(N, N, C.cols(), C.rows(), A.cols(), alpha, B, B.cols(), A, A.cols(), beta, C, C.cols())
    // with A=hidden[num_positions x hidden], B=projection[hidden x vocab], C=logits[num_positions x vocab].
    float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t handle = get_cublas_handle();
    // Pin to the default stream so the subsequent bias-add and loss kernels
    // (which run on stream 0) observe the gemm result.
    cublasSetStream(handle, 0);
    cublasStatus_t status = cublasSgemm(handle,
                                        CUBLAS_OP_N, CUBLAS_OP_N,
                                        vocab_size, num_positions, hidden_size,
                                        &alpha,
                                        g_d_proj, vocab_size,
                                        g_d_hidden, hidden_size,
                                        &beta,
                                        g_loss_memory.d_logits, vocab_size);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("project_logits_to_device cuBLAS sgemm failed: " + std::to_string(status));
    }

    // Add resident bias on device.
    long long total = (long long)num_positions * vocab_size;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    add_row_bias_kernel<<<grid, block>>>(g_loss_memory.d_logits, g_d_bias, num_positions, vocab_size);
    CUDA_CHECK(cudaGetLastError());
}

void lmhead_backward_device(
    float* d_grad_logits,
    int num_positions,
    int hidden_size,
    int vocab_size,
    float beta1,
    float beta2,
    float eps,
    float learning_rate,
    int t,
    float* h_grad_hidden_out
) {
    size_t proj_elems = (size_t)hidden_size * vocab_size;
    size_t grad_hidden_elems = (size_t)num_positions * hidden_size;

    if (grad_hidden_elems > g_grad_hidden_capacity) {
        if (g_d_grad_hidden) cudaFree(g_d_grad_hidden);
        CUDA_CHECK(cudaMalloc(&g_d_grad_hidden, grad_hidden_elems * sizeof(float)));
        g_grad_hidden_capacity = grad_hidden_elems;
    }
    if (g_d_grad_proj == nullptr) {
        CUDA_CHECK(cudaMalloc(&g_d_grad_proj, proj_elems * sizeof(float)));
    }

    cublasHandle_t handle = get_cublas_handle();
    cublasSetStream(handle, 0);
    const float one = 1.0f, zero = 0.0f;

    // grad_hidden = grad_logits @ projection^T   (unscaled, matches CPU grad_input)
    // Column-major: GH_cm[H x N] = projection_cm^T @ grad_logits_cm.
    cublasStatus_t s1 = cublasSgemm(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        hidden_size, num_positions, vocab_size,
        &one,
        g_d_proj, vocab_size,
        d_grad_logits, vocab_size,
        &zero,
        g_d_grad_hidden, hidden_size);
    if (s1 != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("lmhead_backward_device grad_hidden sgemm failed: " + std::to_string(s1));
    }

    // Copy grad_hidden back to host (~16MB) for the CPU final-LN / layer backward.
    CUDA_CHECK(cudaMemcpy(h_grad_hidden_out, g_d_grad_hidden,
                          grad_hidden_elems * sizeof(float), cudaMemcpyDeviceToHost));

    // grad_proj = (1/num_positions) * hidden^T @ grad_logits   (averaged over batch)
    // Column-major: GP_cm[V x H] = grad_logits_cm @ hidden_cm^T, stored as row-major [H x V].
    const float grad_scale = 1.0f / static_cast<float>(num_positions);
    cublasStatus_t s2 = cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        vocab_size, hidden_size, num_positions,
        &grad_scale,
        d_grad_logits, vocab_size,
        g_d_hidden, hidden_size,
        &zero,
        g_d_grad_proj, vocab_size);
    if (s2 != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("lmhead_backward_device grad_proj sgemm failed: " + std::to_string(s2));
    }

    // Adam update of the resident projection weights (faithful to CPU path).
    const float clip_threshold = 5.0f;
    const float max_allowed_value = 100.0f;
    const float scale_factor = sqrtf(1.0f / static_cast<float>(hidden_size));
    const float max_update = 0.05f * scale_factor;
    const float bias_corr1 = 1.0f - powf(beta1, static_cast<float>(t));
    const float bias_corr2 = 1.0f - powf(beta2, static_cast<float>(t));

    long long n = (long long)proj_elems;
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    lmhead_adam_update_kernel<<<grid, block, 0, 0>>>(
        g_d_proj, g_d_m_proj, g_d_v_proj, g_d_grad_proj, n,
        beta1, beta2, eps, learning_rate,
        bias_corr1, bias_corr2,
        clip_threshold, max_update, max_allowed_value);
    CUDA_CHECK(cudaGetLastError());
}

void lmhead_sync_weights_to_host(float* h_projection, float* h_bias) {
    if (!g_lmhead_initialized) return;
    cudaDeviceSynchronize();
    if (h_projection && g_d_proj) {
        CUDA_CHECK(cudaMemcpy(h_projection, g_d_proj, g_proj_capacity * sizeof(float), cudaMemcpyDeviceToHost));
    }
    if (h_bias && g_d_bias) {
        CUDA_CHECK(cudaMemcpy(h_bias, g_d_bias, g_bias_capacity * sizeof(float), cudaMemcpyDeviceToHost));
    }
}

float compute_loss_from_device_logits(
    const int* h_targets,
    int num_positions,
    int vocab_size
) {
    g_loss_memory.ensure_allocated(num_positions, vocab_size);

    size_t targets_size = num_positions * sizeof(int);
    CUDA_CHECK(cudaMemcpy(g_loss_memory.d_targets, h_targets, targets_size, cudaMemcpyHostToDevice));

    int block_size = 256;
    int shared_mem = 2 * block_size * sizeof(float);

    softmax_cross_entropy_grad_kernel<<<num_positions, block_size, shared_mem, 0>>>(
        g_loss_memory.d_logits, g_loss_memory.d_targets,
        g_loss_memory.d_losses, g_loss_memory.d_grad_output,
        num_positions, vocab_size
    );
    CUDA_CHECK(cudaGetLastError());

    reduce_losses_kernel<<<1, 256, 256 * sizeof(float), 0>>>(
        g_loss_memory.d_losses, g_loss_memory.d_loss, num_positions
    );
    CUDA_CHECK(cudaGetLastError());

    float h_loss;
    CUDA_CHECK(cudaMemcpy(&h_loss, g_loss_memory.d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    return h_loss;
}

void cleanup_loss_resources() {
    // Make sure no kernels/copies are still using these buffers.
    cudaDeviceSynchronize();

    // Free the persistent loss pools (logits/grad/targets/losses).
    g_loss_memory.free_all();

    // Free the device-resident LM head buffers + Adam state + backward scratch.
    if (g_d_hidden)      { cudaFree(g_d_hidden);      g_d_hidden      = nullptr; g_hidden_capacity      = 0; }
    if (g_d_proj)        { cudaFree(g_d_proj);        g_d_proj        = nullptr; g_proj_capacity        = 0; }
    if (g_d_bias)        { cudaFree(g_d_bias);        g_d_bias        = nullptr; g_bias_capacity        = 0; }
    if (g_d_m_proj)      { cudaFree(g_d_m_proj);      g_d_m_proj      = nullptr; }
    if (g_d_v_proj)      { cudaFree(g_d_v_proj);      g_d_v_proj      = nullptr; }
    if (g_d_grad_proj)   { cudaFree(g_d_grad_proj);   g_d_grad_proj   = nullptr; }
    if (g_d_grad_hidden) { cudaFree(g_d_grad_hidden); g_d_grad_hidden = nullptr; g_grad_hidden_capacity = 0; }
    g_lmhead_initialized = false;
}

} // namespace cuda
