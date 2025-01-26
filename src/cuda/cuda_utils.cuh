#pragma once
#include "../include/matrix.hpp"

extern "C" {
    bool customMatrixMultiply(const Matrix& A, const Matrix& B, Matrix& C);
} 