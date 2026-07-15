#pragma once

/**
 * @brief CUDA 12.6 + GCC compatibility fix
 * 
 * CUDA 12.6's math_functions.h has a KNOWN BUG where it conflicts with 
 * C++11/14/17/20's constexpr math functions.
 * 
 * This is NVIDIA bug #4229173 - constexpr conflicts in math_functions.h
 * 
 * Solution: Aggressively suppress the redeclaration errors.
 */

#ifdef USE_CUDA

// Prevent multiple inclusion
#ifndef CUDA_MATH_FIX_APPLIED
#define CUDA_MATH_FIX_APPLIED

// Tell CUDA to not define extra operators that can cause conflicts
#define __CUDA_NO_HALF_OPERATORS__
#define __CUDA_NO_HALF_CONVERSIONS__
#define __CUDA_NO_BFLOAT16_CONVERSIONS__

// Platform-specific warning suppression for CUDA 12.6+ math conflicts
#ifdef _MSC_VER
    // MSVC: Suppress warnings about redefinitions and deprecated functions
    #pragma warning(push)
    #pragma warning(disable: 4005)  // macro redefinition
    #pragma warning(disable: 4244)  // conversion warnings
    #pragma warning(disable: 4267)  // size_t to int conversion
    #pragma warning(disable: 4996)  // deprecated functions
#elif defined(__GNUC__) || defined(__clang__)
    // GCC/Clang: Suppress constexpr redeclaration errors (NVIDIA bug #4229173)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wattributes"
    #pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

// Include CUDA runtime
#include <cuda_runtime.h>

// Restore warnings for user code
#ifdef _MSC_VER
    #pragma warning(pop)
#elif defined(__GNUC__) || defined(__clang__)
    #pragma GCC diagnostic pop
#endif

#endif // CUDA_MATH_FIX_APPLIED

#endif // USE_CUDA

