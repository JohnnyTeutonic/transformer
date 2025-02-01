#pragma once
#include "components.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>

class CudaMatrix {
  private:
    float* device_ptr;
    size_t rows_;
    size_t cols_;
    cublasHandle_t handle;

  public:
    CudaMatrix(const Matrix& host_matrix);
    ~CudaMatrix();

    // CUDA operations
    static CudaMatrix matmul(const CudaMatrix& a, const CudaMatrix& b);
    void apply_softmax();
    void apply_relu();
    void scale(float factor);

    // Data transfer
    Matrix to_host() const;
    void to_device(const Matrix& host_matrix);

    // Getters
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
};

namespace cuda {

// CUDA kernel declarations
__global__ void softmax_kernel(float* matrix, int rows, int cols);
__global__ void relu_kernel(float* matrix, int size);
__global__ void attention_scores_kernel(const float* Q, const float* K, float* scores,
                                      float scale, int seq_len, int head_dim);
__global__ void attention_kernel(float* Q, float* K, float* V, float* output, 
                               int batch_size, int seq_len, int head_dim);
__global__ void add_bias_kernel(float* output, const float* bias, int rows, int cols);
__global__ void row_sum_kernel(const float* input, float* output, int rows, int cols);
__global__ void adam_update_kernel(float* param, const float* grad,
                                 float* m, float* v,
                                 float beta1, float beta2,
                                 float eps, float lr,
                                 int step, int size);

// Launch function declarations
void launch_softmax(float* matrix, int rows, int cols,
                   cudaStream_t stream = nullptr);

void launch_attention(float* Q, float* K, float* V, float* output,
                     int batch_size, int seq_len, int head_dim,
                     cudaStream_t stream);

void launch_add_bias(float* output, const float* bias,
                    int rows, int cols,
                    cudaStream_t stream);

void launch_row_sum(const float* input, float* output,
                   int rows, int cols,
                   cudaStream_t stream);

void launch_adam_update(float* param, const float* grad,
                       float* m, float* v,
                       float beta1, float beta2,
                       float eps, float lr,
                       int step, int size,
                       cudaStream_t stream);

} // namespace cuda