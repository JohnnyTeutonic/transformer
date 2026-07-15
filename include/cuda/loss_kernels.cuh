#pragma once

#include <cuda_runtime.h>

namespace cuda {

/**
 * Fused softmax + cross-entropy loss computation on GPU.
 * Computes: loss = -sum(log(softmax(logits)[target])) / num_positions
 * 
 * @param logits Input logits [num_positions x vocab_size] (device memory)
 * @param targets Target token indices [num_positions] (device memory)
 * @param loss Output scalar loss (device memory, single float)
 * @param num_positions Number of positions (batch_size * seq_len)
 * @param vocab_size Vocabulary size
 * @param stream CUDA stream
 */
void fused_softmax_cross_entropy(
    const float* logits,
    const int* targets,
    float* loss,
    int num_positions,
    int vocab_size,
    cudaStream_t stream = 0
);

/**
 * Host-side wrapper that handles memory transfers.
 * Takes host memory, computes on GPU, returns loss.
 */
float compute_cross_entropy_loss_cuda(
    const float* h_logits,      // Host: [num_positions x vocab_size]
    const int* h_targets,       // Host: [num_positions]
    int num_positions,
    int vocab_size
);

/**
 * Computes loss AND gradient in one fused kernel.
 * Returns loss, writes gradient to h_grad_output.
 * Eliminates need for 655MB target_dist matrix.
 */
float compute_cross_entropy_loss_and_grad_cuda(
    const float* h_logits,      // Host: [num_positions x vocab_size]
    const int* h_targets,       // Host: [num_positions]
    float* h_grad_output,       // Host OUTPUT: [num_positions x vocab_size]
    int num_positions,
    int vocab_size
);

/**
 * Computes loss AND keeps gradient on device (avoids 655MB D2H transfer).
 * Returns loss scalar, gradient stays in device memory for use by backward_cuda.
 */
float compute_cross_entropy_loss_keep_grad_on_device(
    const float* h_logits,      // Host: [num_positions x vocab_size]
    const int* h_targets,       // Host: [num_positions]
    int num_positions,
    int vocab_size
);

/**
 * Uploads the LM head projection + bias to the device ONCE and allocates the
 * device-resident Adam state (zeroed). Subsequent calls are no-ops unless the
 * weights have been invalidated (see lmhead_invalidate_weights). The device
 * copy is authoritative during training; sync it back with
 * lmhead_sync_weights_to_host before reading the host weights (eval/save).
 *
 * @param h_projection Host: [hidden_size x vocab_size] projection weights
 * @param h_bias       Host: [vocab_size] bias
 */
void lmhead_ensure_weights(
    const float* h_projection,
    const float* h_bias,
    int hidden_size,
    int vocab_size
);

/**
 * Marks the device-resident LM head weights as stale so the next
 * lmhead_ensure_weights re-uploads them from host (e.g., after loading a
 * checkpoint into the host weights).
 */
void lmhead_invalidate_weights();

/**
 * Device-resident projection: computes logits = hidden @ projection + bias
 * directly into the persistent device logits buffer (g_loss_memory.d_logits),
 * using the resident device weights (see lmhead_ensure_weights). Matches the
 * column-major cuBLAS convention used by cuda::matmul. Avoids the 655MB
 * device->host->device round trip of the logits.
 *
 * @param h_hidden Host: [num_positions x hidden_size] hidden states
 */
void project_logits_to_device(
    const float* h_hidden,
    int num_positions,
    int hidden_size,
    int vocab_size
);

/**
 * Device-side LM head backward + Adam weight update. Consumes the gradient that
 * is already on the device (d_grad_logits), so the 655MB grad never leaves the
 * GPU. Computes:
 *   - grad_hidden = grad_logits @ projection^T  (returned to host, ~16MB)
 *   - grad_proj   = (1/num_positions) * hidden^T @ grad_logits
 *   - Adam update of the resident projection weights (faithful to the CPU path:
 *     per-element clip, bias correction with t, update clamp, value clamp).
 * The bias is intentionally NOT updated (matches the existing CPU behavior).
 *
 * @param d_grad_logits     Device: [num_positions x vocab_size] gradient
 * @param h_grad_hidden_out Host OUTPUT: [num_positions x hidden_size]
 */
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
);

/**
 * Copies the resident device LM head weights back to host (projection always;
 * bias is unchanged during training but copied for completeness). No-op if the
 * device weights have not been initialized.
 */
void lmhead_sync_weights_to_host(float* h_projection, float* h_bias);

/**
 * Computes loss + gradient using the logits already resident on the device
 * (as filled by project_logits_to_device). No host logits transfer.
 * Gradient stays in device memory (see get_device_grad_logits()).
 */
float compute_loss_from_device_logits(
    const int* h_targets,
    int num_positions,
    int vocab_size
);

/**
 * Frees all device memory owned by the loss kernels (logits/grad pools and the
 * device-resident projection buffers) while the CUDA context is still alive.
 * Must be called during normal shutdown BEFORE the context is torn down to avoid
 * an access violation from cudaFree running in static destructors at exit.
 * Safe to call multiple times.
 */
void cleanup_loss_resources();

/**
 * Get the device pointer to the gradient computed by the last loss call.
 * Valid until next loss computation.
 */
float* get_device_grad_logits();

/**
 * Get the device pointer to the logits uploaded during loss computation.
 */
float* get_device_logits();

} // namespace cuda
