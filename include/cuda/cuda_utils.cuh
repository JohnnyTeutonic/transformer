#pragma once

#include <cstdio>   // for fprintf, stderr
#include <cstdlib>  // for exit

#ifdef _WIN32
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#else
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

// Global cuBLAS handle
extern cublasHandle_t cublas_handle;

// CUDA error checking macro
#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); \
    } \
  } while(0)

// cuBLAS error checking macro
#define CUBLAS_CHECK(call) \
  do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
      fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, \
              status); \
      exit(EXIT_FAILURE); \
    } \
  } while(0)

// CUDA initialization and cleanup functions
void initialize_cuda();
void cleanup_cuda();