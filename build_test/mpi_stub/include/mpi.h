
#pragma once
#include <cstddef>
#include <cstring>

// MPI Constants
#define MPI_COMM_WORLD 0
#define MPI_SUCCESS 0
#define MPI_THREAD_MULTIPLE 3
#define MPI_FLOAT 1
#define MPI_INT 2
#define MPI_UNSIGNED_LONG 3
#define MPI_BYTE 4
#define MPI_SUM 1

// MPI Types
typedef int MPI_Comm;
typedef int MPI_Datatype;
typedef int MPI_Op;

// MPI Functions (single-node stubs)
inline int MPI_Init(int* argc, char*** argv) { return MPI_SUCCESS; }
inline int MPI_Init_thread(int* argc, char*** argv, int required, int* provided) { 
    if (provided) *provided = MPI_THREAD_MULTIPLE; 
    return MPI_SUCCESS; 
}
inline int MPI_Finalize() { return MPI_SUCCESS; }
inline int MPI_Initialized(int* flag) { *flag = 1; return MPI_SUCCESS; }
inline int MPI_Finalized(int* flag) { *flag = 0; return MPI_SUCCESS; }
inline int MPI_Comm_rank(MPI_Comm comm, int* rank) { *rank = 0; return MPI_SUCCESS; }
inline int MPI_Comm_size(MPI_Comm comm, int* size) { *size = 1; return MPI_SUCCESS; }
inline int MPI_Barrier(MPI_Comm comm) { return MPI_SUCCESS; }
inline int MPI_Bcast(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) { return MPI_SUCCESS; }
inline int MPI_Allreduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
    // Single rank: just copy data
    if (sendbuf != recvbuf) {
        size_t bytes = count * sizeof(float); // Assume float for simplicity
        memcpy(recvbuf, sendbuf, bytes);
    }
    return MPI_SUCCESS;
}
