/**
 * @file cuda_compat.h
 * @brief CUDA compatibility definitions for newer Visual Studio versions
 * 
 * This header is force-included before all other headers to allow
 * CUDA 13.0 to work with Visual Studio 2026.
 */

#ifndef CUDA_COMPAT_H
#define CUDA_COMPAT_H

// Allow CUDA 13.0 to work with Visual Studio 2026
// This disables the version check in CUDA's host_config.h
#ifdef _MSC_VER
    // Disable CUDA host compiler version check
    #ifndef __NV_NO_HOST_COMPILER_CHECK
        #define __NV_NO_HOST_COMPILER_CHECK
    #endif
    
    // Also allow STL version mismatch (for good measure)
    #ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
        #define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
    #endif
    
    // Suppress MSVC warnings that can appear with CUDA headers
    #pragma warning(disable: 4005)  // macro redefinition
    #pragma warning(disable: 4244)  // possible loss of data
    #pragma warning(disable: 4267)  // size_t conversion
    #pragma warning(disable: 4996)  // deprecated functions
#endif

// Tell CUDA to skip problematic operator overloads that conflict with std::
#ifndef __CUDA_NO_HALF_OPERATORS__
    #define __CUDA_NO_HALF_OPERATORS__
#endif
#ifndef __CUDA_NO_HALF_CONVERSIONS__
    #define __CUDA_NO_HALF_CONVERSIONS__
#endif
#ifndef __CUDA_NO_BFLOAT16_CONVERSIONS__
    #define __CUDA_NO_BFLOAT16_CONVERSIONS__
#endif

#endif // CUDA_COMPAT_H

