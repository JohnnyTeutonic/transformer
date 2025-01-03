#pragma once

#include "cuda_utils.cuh"

#ifdef __cplusplus
extern "C" {
#endif

void launch_tensor_mul(const float* a, const float* b, float* result,
                      int d1, int d2, int d3, int d4, int b_d4);

#ifdef __cplusplus
}
#endif 