#pragma once
#include <cuda_runtime.h>
#include "matrix.hpp"

namespace cuda {
    // Kernel declarations
    void launch_softmax_kernel(float* input, float* output, int rows, int cols);
    void launch_gelu_kernel(float* input, float* output, int size);
    void launch_gelu_backward_kernel(float* grad_output, float* input, float* grad_input, int size);
}