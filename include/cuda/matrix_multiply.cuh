#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t launch_matrix_multiply(
    const float* projection,
    const float* grad_output,
    float* grad_proj,
    int batch_size,
    int vocab_size,
    int hidden_size,
    cudaStream_t stream = nullptr
);

#ifdef __cplusplus
}
#endif 