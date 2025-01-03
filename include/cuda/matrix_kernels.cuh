#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void launch_matrix_add(const float* a, const float* b, float* result,
                      int rows, int cols);

void launch_matrix_sub(const float* a, const float* b, float* result,
                      int rows, int cols);

void launch_matrix_scalar_mul(const float* a, float scalar, float* result,
                            int total_elements);

void launch_matrix_mul(const float* a, const float* b, float* result,
                      int m, int n, int k);

void launch_matrix_transpose(const float* a, float* result,
                           int rows, int cols);

#ifdef __cplusplus
}
#endif 