
#pragma once
#include <cuda_runtime.h>

// NCCL Types
typedef struct ncclComm* ncclComm_t;
typedef enum { ncclSuccess = 0 } ncclResult_t;
typedef enum { ncclSum = 0 } ncclRedOp_t;
typedef enum { ncclFloat = 0 } ncclDataType_t;
typedef struct { char internal[128]; } ncclUniqueId;

// NCCL Functions (stubs that fall back to CUDA memcpy)
inline ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId) { return ncclSuccess; }
inline ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank) { 
    *comm = (ncclComm_t)1; // Dummy handle
    return ncclSuccess; 
}
inline ncclResult_t ncclCommDestroy(ncclComm_t comm) { return ncclSuccess; }
inline ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, 
                                 ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
    // Single GPU: just copy data
    if (sendbuff != recvbuff) {
        cudaMemcpyAsync(recvbuff, sendbuff, count * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }
    return ncclSuccess;
}
inline const char* ncclGetErrorString(ncclResult_t result) { return "ncclSuccess"; }
