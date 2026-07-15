#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "../../include/cuda/attention_ops.cuh"
#include "../../include/cuda/cuda_check.cuh"
#include "../../include/cuda/cuda_utils.cuh"
#include "../../include/cuda/kernel_declarations.cuh"

namespace cuda {
    // Host functions only in namespace
    void compute_attention_scores(const Matrix& Q, const Matrix& K, Matrix& scores, float scale, int num_heads) {
        // Synchronize before starting
        CUDA_CHECK(cudaDeviceSynchronize());
        
        int batch_size = Q.rows();
        int hidden_dim = Q.cols();
        int head_dim = hidden_dim / num_heads;
        int seq_len = batch_size;

        // Verify all dimensions are valid
        if (batch_size <= 0 || hidden_dim <= 0 || head_dim <= 0 || seq_len <= 0) {
            throw std::runtime_error("Invalid dimensions detected");
        }

        // Memory allocation with error checking
        float* d_Q = nullptr;
        float* d_K = nullptr;
        float* d_scores = nullptr;
        
        try {
            CUDA_CHECK(cudaMalloc(&d_Q, Q.size() * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_K, K.size() * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_scores, scores.size() * sizeof(float)));
            
            // Zero initialize the scores buffer
            CUDA_CHECK(cudaMemset(d_scores, 0, scores.size() * sizeof(float)));

            CUDA_CHECK(cudaMemcpy(d_Q, Q.data(), Q.size() * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_K, K.data(), K.size() * sizeof(float), cudaMemcpyHostToDevice));
            
            // Synchronize to ensure memory transfers are complete
            CUDA_CHECK(cudaDeviceSynchronize());

            dim3 block(16, 16);
            dim3 grid((seq_len + block.x - 1) / block.x, (seq_len + block.y - 1) / block.y);
            
            attention_scores_kernel<<<grid, block>>>(d_Q, d_K, d_scores,
                scale, seq_len, head_dim);
                
            // Check for kernel launch errors
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("Kernel launch failed: ") + 
                                       cudaGetErrorString(err));
            }

            // Synchronize after kernel
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy(scores.data(), d_scores, scores.size() * sizeof(float), 
                                cudaMemcpyDeviceToHost));

        } catch (const std::exception& e) {
            printf("CUDA error caught: %s\n", e.what());
            // Clean up on error
            if (d_Q) cudaFree(d_Q);
            if (d_K) cudaFree(d_K);
            if (d_scores) cudaFree(d_scores);
            throw;  // Re-throw the exception
        }

        // Clean up
        CUDA_CHECK(cudaFree(d_Q));
        CUDA_CHECK(cudaFree(d_K));
        CUDA_CHECK(cudaFree(d_scores));
        
        // Final synchronize
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void apply_softmax(Matrix& matrix) {
        float* d_matrix;
        size_t size = matrix.size() * sizeof(float);

        CUDA_CHECK(cudaMalloc(&d_matrix, size));
        CUDA_CHECK(cudaMemcpy(d_matrix, matrix.data(), size, cudaMemcpyHostToDevice));

        softmax_kernel<<<matrix.rows(), 1>>>(d_matrix, matrix.rows(), matrix.cols());

        CUDA_CHECK(cudaMemcpy(matrix.data(), d_matrix, size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_matrix));
    }

    void attention_forward(const Matrix& Q, const Matrix& K, const Matrix& V, 
                         Matrix& output, int batch_size, int num_heads, int seq_len) {
        // Configure grid for batch and head parallelism
        dim3 block(32, 1);
        dim3 grid((batch_size + block.x - 1) / block.x, num_heads);
        
        int head_dim = Q.cols() / num_heads;
        int hidden_dim = Q.cols();  // Store the full hidden dimension

        float *d_Q, *d_K, *d_V, *d_output;
        size_t QKV_size = Q.size() * sizeof(float);
        size_t output_size = output.size() * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_Q, QKV_size));
        CUDA_CHECK(cudaMalloc(&d_K, QKV_size));
        CUDA_CHECK(cudaMalloc(&d_V, QKV_size));
        CUDA_CHECK(cudaMalloc(&d_output, output_size));

        CUDA_CHECK(cudaMemcpy(d_Q, Q.data(), QKV_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K, K.data(), QKV_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V, V.data(), QKV_size, cudaMemcpyHostToDevice));

        // Allocate shared memory for scores
        size_t shared_mem_size = seq_len * sizeof(float);
        attention_kernel<<<grid, block, shared_mem_size>>>(d_Q, d_K, d_V, d_output,
                                                         batch_size, seq_len, head_dim, hidden_dim);

        CUDA_CHECK(cudaMemcpy(output.data(), d_output, output_size, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_Q));
        CUDA_CHECK(cudaFree(d_K));
        CUDA_CHECK(cudaFree(d_V));
        CUDA_CHECK(cudaFree(d_output));
    }

    void launch_attention_scores_kernel(const float* Q, const float* K, float* scores, float scale,
                                      int seq_len, int head_dim, cudaStream_t stream) {
        dim3 block_dim(16, 16);
        dim3 grid_dim((seq_len + block_dim.x - 1) / block_dim.x,
                      (seq_len + block_dim.y - 1) / block_dim.y);

        attention_scores_kernel<<<grid_dim, block_dim, 0, stream>>>(Q, K, scores, scale, seq_len,
                                                                 head_dim);
    }

    // ============================================================================
    // BATCHED ATTENTION: Uses cuBLAS strided batched GEMM for all heads at once
    // Replaces 512 small CPU matmuls with batched GPU calls
    // ============================================================================
    
    // Persistent memory pool for batched attention
    static struct BatchedAttnMemory {
        float* d_Q = nullptr;
        float* d_K = nullptr;
        float* d_V = nullptr;
        float* d_scores = nullptr;
        float* d_output = nullptr;
        // Backward-pass buffers (allocated lazily by ensure_backward_allocated)
        float* d_dOut = nullptr;
        float* d_dP = nullptr;          // reused as dS in place
        float* d_dQ = nullptr;
        float* d_dK = nullptr;
        float* d_dV = nullptr;
        size_t allocated_size = 0;
        size_t bwd_allocated_size = 0;
        cublasHandle_t cublas_handle = nullptr;

        void ensure_allocated(size_t qkv_size, size_t scores_size) {
            size_t max_size = std::max(qkv_size, scores_size);
            if (max_size > allocated_size) {
                free_all();
                CUDA_CHECK(cudaMalloc(&d_Q, qkv_size));
                CUDA_CHECK(cudaMalloc(&d_K, qkv_size));
                CUDA_CHECK(cudaMalloc(&d_V, qkv_size));
                CUDA_CHECK(cudaMalloc(&d_scores, scores_size));
                CUDA_CHECK(cudaMalloc(&d_output, qkv_size));
                allocated_size = max_size;
            }
            if (!cublas_handle) {
                cublasCreate(&cublas_handle);
            }
        }

        void ensure_backward_allocated(size_t qkv_size, size_t scores_size) {
            size_t max_size = std::max(qkv_size, scores_size);
            if (max_size > bwd_allocated_size) {
                free_backward();
                CUDA_CHECK(cudaMalloc(&d_dOut, qkv_size));
                CUDA_CHECK(cudaMalloc(&d_dP, scores_size));
                CUDA_CHECK(cudaMalloc(&d_dQ, qkv_size));
                CUDA_CHECK(cudaMalloc(&d_dK, qkv_size));
                CUDA_CHECK(cudaMalloc(&d_dV, qkv_size));
                bwd_allocated_size = max_size;
            }
        }

        void free_backward() {
            if (d_dOut) { cudaFree(d_dOut); d_dOut = nullptr; }
            if (d_dP) { cudaFree(d_dP); d_dP = nullptr; }
            if (d_dQ) { cudaFree(d_dQ); d_dQ = nullptr; }
            if (d_dK) { cudaFree(d_dK); d_dK = nullptr; }
            if (d_dV) { cudaFree(d_dV); d_dV = nullptr; }
            bwd_allocated_size = 0;
        }

        void free_all() {
            if (d_Q) { cudaFree(d_Q); d_Q = nullptr; }
            if (d_K) { cudaFree(d_K); d_K = nullptr; }
            if (d_V) { cudaFree(d_V); d_V = nullptr; }
            if (d_scores) { cudaFree(d_scores); d_scores = nullptr; }
            if (d_output) { cudaFree(d_output); d_output = nullptr; }
            free_backward();
            if (cublas_handle) { cublasDestroy(cublas_handle); cublas_handle = nullptr; }
            allocated_size = 0;
        }

        ~BatchedAttnMemory() { free_all(); }
    } g_batched_attn;

    // Kernel for applying causal mask and softmax
    __global__ void batched_softmax_causal_kernel(
        float* scores,      // [num_batches, seq_len, seq_len]
        int num_batches,
        int seq_len
    ) {
        int batch_idx = blockIdx.x;
        int row = blockIdx.y;
        
        if (batch_idx >= num_batches || row >= seq_len) return;
        
        float* row_scores = scores + batch_idx * seq_len * seq_len + row * seq_len;
        
        // Apply causal mask
        for (int j = row + 1; j < seq_len; j++) {
            row_scores[j] = -1e30f;
        }
        
        // Find max for numerical stability
        float max_val = -1e30f;
        for (int j = 0; j < seq_len; j++) {
            max_val = fmaxf(max_val, row_scores[j]);
        }
        
        // Exp and sum
        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            row_scores[j] = expf(row_scores[j] - max_val);
            sum += row_scores[j];
        }
        
        // Normalize
        float inv_sum = 1.0f / (sum + 1e-10f);
        for (int j = 0; j < seq_len; j++) {
            row_scores[j] *= inv_sum;
        }
    }

    void batched_attention_forward(
        const float* h_Q,       // [batch_size * seq_len, hidden_size]
        const float* h_K, 
        const float* h_V,
        float* h_output,
        float* h_attn_weights,  // [batch_size * num_heads * seq_len, seq_len]
        int batch_size,
        int seq_len,
        int num_heads,
        int head_dim,
        float scale
    ) {
        int total_positions = batch_size * seq_len;
        int hidden_size = num_heads * head_dim;
        int num_batches = batch_size * num_heads;  // Total number of attention matrices
        
        size_t qkv_size = total_positions * hidden_size * sizeof(float);
        size_t scores_size = num_batches * seq_len * seq_len * sizeof(float);
        
        g_batched_attn.ensure_allocated(qkv_size, scores_size);
        
        // Copy Q, K, V to device
        CUDA_CHECK(cudaMemcpy(g_batched_attn.d_Q, h_Q, qkv_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(g_batched_attn.d_K, h_K, qkv_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(g_batched_attn.d_V, h_V, qkv_size, cudaMemcpyHostToDevice));
        
        // ========== STEP 1: Compute attention scores using batched GEMM ==========
        // For each (batch, head), compute: scores[b,h] = Q[b,h] @ K[b,h]^T
        // Q[b,h] is [seq_len, head_dim], K[b,h]^T is [head_dim, seq_len]
        // Result: scores[b,h] is [seq_len, seq_len]
        //
        // Data layout: Q, K, V are [batch_size * seq_len, hidden_size]
        // where hidden_size = num_heads * head_dim
        // We treat each head slice as a separate batch for strided GEMM
        
        const float alpha = scale;  // Include scaling in GEMM
        const float beta = 0.0f;
        
        // Use cublasSgemmStridedBatched for all attention score computations
        // For each batch item b and head h:
        //   Q_bh starts at: b * seq_len * hidden_size + h * head_dim
        //   Stride between consecutive batch items for same head: seq_len * hidden_size
        //   Stride between consecutive heads for same batch: head_dim
        
        // Actually, let's compute this per (batch, head) pair
        // Total batches = batch_size * num_heads
        for (int b = 0; b < batch_size; b++) {
            for (int h = 0; h < num_heads; h++) {
                int batch_offset = b * seq_len * hidden_size;
                int head_offset = h * head_dim;
                int score_offset = (b * num_heads + h) * seq_len * seq_len;
                
                // Q_bh: [seq_len, head_dim] at offset batch_offset, column head_offset
                // K_bh: [seq_len, head_dim] at offset batch_offset, column head_offset
                // scores_bh: [seq_len, seq_len] at score_offset
                
                // Q_bh @ K_bh^T = [seq_len, head_dim] @ [head_dim, seq_len] = [seq_len, seq_len]
                // cuBLAS: C = alpha * A @ B + beta * C
                // In column-major: we compute C^T = alpha * B^T @ A^T
                // A = Q_bh (row-major), B = K_bh^T (row-major)
                // For row-major A @ B = C, use cuBLAS: C^T = B^T @ A^T
                // cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, ...)
                
                const float* Q_ptr = g_batched_attn.d_Q + batch_offset + head_offset;
                const float* K_ptr = g_batched_attn.d_K + batch_offset + head_offset;
                float* scores_ptr = g_batched_attn.d_scores + score_offset;
                
                // Compute Q @ K^T
                // Row-major: C[m,n] = A[m,k] @ B^T[k,n] where B^T means B transposed
                // For cuBLAS column-major: use CUBLAS_OP_T on K
                cublasSgemm(
                    g_batched_attn.cublas_handle,
                    CUBLAS_OP_T,    // K transposed
                    CUBLAS_OP_N,    // Q not transposed
                    seq_len,        // n = columns of result (and K^T)
                    seq_len,        // m = rows of result (and Q)
                    head_dim,       // k = inner dimension
                    &alpha,
                    K_ptr, hidden_size,      // K: [seq_len, head_dim] with stride hidden_size
                    Q_ptr, hidden_size,      // Q: [seq_len, head_dim] with stride hidden_size
                    &beta,
                    scores_ptr, seq_len      // scores: [seq_len, seq_len]
                );
            }
        }
        
        // ========== STEP 2: Apply causal mask and softmax ==========
        dim3 softmax_grid(num_batches, seq_len);
        batched_softmax_causal_kernel<<<softmax_grid, 1>>>(
            g_batched_attn.d_scores, num_batches, seq_len
        );
        CUDA_CHECK(cudaGetLastError());
        
        // ========== STEP 3: Compute attention output: scores @ V ==========
        const float alpha1 = 1.0f;
        CUDA_CHECK(cudaMemset(g_batched_attn.d_output, 0, qkv_size));
        
        for (int b = 0; b < batch_size; b++) {
            for (int h = 0; h < num_heads; h++) {
                int batch_offset = b * seq_len * hidden_size;
                int head_offset = h * head_dim;
                int score_offset = (b * num_heads + h) * seq_len * seq_len;
                
                const float* scores_ptr = g_batched_attn.d_scores + score_offset;
                const float* V_ptr = g_batched_attn.d_V + batch_offset + head_offset;
                float* out_ptr = g_batched_attn.d_output + batch_offset + head_offset;
                
                // scores @ V = [seq_len, seq_len] @ [seq_len, head_dim] = [seq_len, head_dim]
                cublasSgemm(
                    g_batched_attn.cublas_handle,
                    CUBLAS_OP_N,    // V not transposed
                    CUBLAS_OP_N,    // scores not transposed
                    head_dim,       // n = columns of result (head_dim)
                    seq_len,        // m = rows of result (seq_len)
                    seq_len,        // k = inner dimension
                    &alpha1,
                    V_ptr, hidden_size,         // V: [seq_len, head_dim]
                    scores_ptr, seq_len,        // scores: [seq_len, seq_len]
                    &beta,
                    out_ptr, hidden_size        // output: [seq_len, head_dim]
                );
            }
        }
        
        // Copy results back
        CUDA_CHECK(cudaMemcpy(h_output, g_batched_attn.d_output, qkv_size, cudaMemcpyDeviceToHost));
        if (h_attn_weights) {
            CUDA_CHECK(cudaMemcpy(h_attn_weights, g_batched_attn.d_scores, scores_size, cudaMemcpyDeviceToHost));
        }
    }

    // Softmax backward, row-wise: dS = P .* (dP - rowsum(dP .* P)).
    // Same launch geometry as batched_softmax_causal_kernel; operates in place
    // on dP. Masked entries have P == 0, so dS is 0 there automatically.
    __global__ void batched_softmax_backward_kernel(
        const float* P,     // [num_batches, seq_len, seq_len] post-softmax weights
        float* dP,          // in: dP; out: dS
        int num_batches,
        int seq_len
    ) {
        int batch_idx = blockIdx.x;
        int row = blockIdx.y;
        if (batch_idx >= num_batches || row >= seq_len) return;

        const size_t off = (static_cast<size_t>(batch_idx) * seq_len + row) * seq_len;
        const float* p = P + off;
        float* dp = dP + off;

        float dot = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            dot += p[j] * dp[j];
        }
        for (int j = 0; j < seq_len; j++) {
            dp[j] = p[j] * (dp[j] - dot);
        }
    }

    void batched_attention_backward(
        const float* h_dOut,          // [batch*seq_len, hidden] grad wrt attention output
        const float* h_attn_weights,  // [batch*heads*seq_len, seq_len] cached softmax P
        const float* h_Q,             // [batch*seq_len, hidden] cached (post-RoPE)
        const float* h_K,
        const float* h_V,
        float* h_dQ,                  // outputs, same layout as Q/K/V
        float* h_dK,
        float* h_dV,
        int batch_size,
        int seq_len,
        int num_heads,
        int head_dim,
        float scale
    ) {
        const int total_positions = batch_size * seq_len;
        const int hidden_size = num_heads * head_dim;
        const int num_batches = batch_size * num_heads;

        const size_t qkv_size = static_cast<size_t>(total_positions) * hidden_size * sizeof(float);
        const size_t scores_size = static_cast<size_t>(num_batches) * seq_len * seq_len * sizeof(float);

        g_batched_attn.ensure_allocated(qkv_size, scores_size);
        g_batched_attn.ensure_backward_allocated(qkv_size, scores_size);

        CUDA_CHECK(cudaMemcpy(g_batched_attn.d_dOut, h_dOut, qkv_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(g_batched_attn.d_scores, h_attn_weights, scores_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(g_batched_attn.d_Q, h_Q, qkv_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(g_batched_attn.d_K, h_K, qkv_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(g_batched_attn.d_V, h_V, qkv_size, cudaMemcpyHostToDevice));

        const float one = 1.0f;
        const float zero = 0.0f;

        // ===== STEP 1: dP[b,h] = dOut_h @ V_h^T  (same GEMM shape as Q @ K^T) =====
        for (int b = 0; b < batch_size; b++) {
            for (int h = 0; h < num_heads; h++) {
                const int batch_offset = b * seq_len * hidden_size;
                const int head_offset = h * head_dim;
                const int score_offset = (b * num_heads + h) * seq_len * seq_len;
                cublasSgemm(
                    g_batched_attn.cublas_handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    seq_len, seq_len, head_dim,
                    &one,
                    g_batched_attn.d_V + batch_offset + head_offset, hidden_size,
                    g_batched_attn.d_dOut + batch_offset + head_offset, hidden_size,
                    &zero,
                    g_batched_attn.d_dP + score_offset, seq_len);
            }
        }

        // ===== STEP 2: softmax backward in place (dP -> dS) =====
        dim3 grid(num_batches, seq_len);
        batched_softmax_backward_kernel<<<grid, 1>>>(
            g_batched_attn.d_scores, g_batched_attn.d_dP, num_batches, seq_len);
        CUDA_CHECK(cudaGetLastError());

        // ===== STEP 3: dQ_h = scale * dS @ K_h ; dK_h = scale * dS^T @ Q_h ;
        //               dV_h = P^T @ dOut_h =====
        // Row-major C[m,n] = A[m,k] @ B[k,n] maps to
        //   cublasSgemm(handle, opB, opA, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc)
        // exactly as in batched_attention_forward; a transpose on A lands as
        // CUBLAS_OP_T in A's (second) slot.
        for (int b = 0; b < batch_size; b++) {
            for (int h = 0; h < num_heads; h++) {
                const int batch_offset = b * seq_len * hidden_size;
                const int head_offset = h * head_dim;
                const int score_offset = (b * num_heads + h) * seq_len * seq_len;

                // dQ_h = scale * dS @ K_h            [S,S] @ [S,hd] -> [S,hd]
                cublasSgemm(
                    g_batched_attn.cublas_handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    head_dim, seq_len, seq_len,
                    &scale,
                    g_batched_attn.d_K + batch_offset + head_offset, hidden_size,
                    g_batched_attn.d_dP + score_offset, seq_len,
                    &zero,
                    g_batched_attn.d_dQ + batch_offset + head_offset, hidden_size);

                // dK_h = scale * dS^T @ Q_h          [S,S]^T @ [S,hd] -> [S,hd]
                cublasSgemm(
                    g_batched_attn.cublas_handle,
                    CUBLAS_OP_N, CUBLAS_OP_T,
                    head_dim, seq_len, seq_len,
                    &scale,
                    g_batched_attn.d_Q + batch_offset + head_offset, hidden_size,
                    g_batched_attn.d_dP + score_offset, seq_len,
                    &zero,
                    g_batched_attn.d_dK + batch_offset + head_offset, hidden_size);

                // dV_h = P^T @ dOut_h                [S,S]^T @ [S,hd] -> [S,hd]
                cublasSgemm(
                    g_batched_attn.cublas_handle,
                    CUBLAS_OP_N, CUBLAS_OP_T,
                    head_dim, seq_len, seq_len,
                    &one,
                    g_batched_attn.d_dOut + batch_offset + head_offset, hidden_size,
                    g_batched_attn.d_scores + score_offset, seq_len,
                    &zero,
                    g_batched_attn.d_dV + batch_offset + head_offset, hidden_size);
            }
        }

        CUDA_CHECK(cudaMemcpy(h_dQ, g_batched_attn.d_dQ, qkv_size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_dK, g_batched_attn.d_dK, qkv_size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_dV, g_batched_attn.d_dV, qkv_size, cudaMemcpyDeviceToHost));
    }
}

// Kernel implementations outside namespace
extern "C" {
    CUDA_KERNEL void attention_scores_kernel(const float* Q, const float* K, float* scores,
                                           float scale, int seq_len, int head_dim) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        int col = blockIdx.y * blockDim.y + threadIdx.y;


        if (row < seq_len && col < seq_len) {
            float sum = 0.0f;
            for (int i = 0; i < head_dim; i++) {
                // Q and K are [batch_size x hidden_dim], need to index correctly
                int q_idx = row * head_dim + i;
                int k_idx = col * head_dim + i;
                
                // Bounds check
                if (q_idx < seq_len * head_dim && k_idx < seq_len * head_dim) {
                    sum += Q[q_idx] * K[k_idx];
                }
            }
            // Ensure we write to the correct location in scores matrix
            int score_idx = row * seq_len + col;
            if (score_idx < seq_len * seq_len) {
                scores[score_idx] = sum * scale;
            }
        }
    }

    CUDA_KERNEL void softmax_kernel(float* matrix, int rows, int cols) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < rows) {
            // Find max for numerical stability
            float max_val = matrix[row * cols];
            for (int i = 1; i < cols; i++) {
                max_val = max(max_val, matrix[row * cols + i]);
            }

            // Compute exp and sum
            float sum = 0.0f;
            for (int i = 0; i < cols; i++) {
                matrix[row * cols + i] = expf(matrix[row * cols + i] - max_val);
                sum += matrix[row * cols + i];
            }

            // Normalize
            for (int i = 0; i < cols; i++) {
                matrix[row * cols + i] /= sum;
            }
        }
    }

    CUDA_KERNEL void attention_kernel(const float* Q, const float* K, const float* V,
                                    float* output, int batch_size, int seq_len, 
                                    int head_dim, int hidden_dim) {
        int b = blockIdx.x * blockDim.x + threadIdx.x;  // batch index
        int h = blockIdx.y;  // head index

        if (b < batch_size) {
            // Process this batch element for the current head
            int head_offset = h * head_dim;
            int batch_offset = b * hidden_dim;
            
            // Allocate scores in shared memory
            extern __shared__ float scores[];
            
            // Compute attention scores for this head
            for (int j = 0; j < seq_len; j++) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    int q_idx = batch_offset + head_offset + d;
                    int k_idx = j * hidden_dim + head_offset + d;  // Fixed K indexing
                    
                    if (q_idx < batch_size * hidden_dim && k_idx < seq_len * hidden_dim) {
                        score += Q[q_idx] * K[k_idx];
                    }
                }
                scores[j] = score / sqrtf(float(head_dim));
            }

            // Apply softmax
            float max_score = scores[0];
            for (int j = 1; j < seq_len; j++) {
                max_score = max(max_score, scores[j]);
            }

            float sum = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                scores[j] = expf(scores[j] - max_score);
                sum += scores[j];
            }

            for (int j = 0; j < seq_len; j++) {
                scores[j] /= sum;
            }

            // Compute weighted sum with correct output indexing
            for (int d = 0; d < head_dim; d++) {
                float weighted_sum = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    int v_idx = j * hidden_dim + head_offset + d;  // Fixed V indexing
                    if (v_idx < seq_len * hidden_dim) {
                        weighted_sum += scores[j] * V[v_idx];
                    }
                }
                // Write to the correct output position
                int out_idx = batch_offset + head_offset + d;
                if (out_idx < batch_size * hidden_dim) {
                    output[out_idx] = weighted_sum;
                }
            }
        }
    }
} 