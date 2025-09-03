cmake_minimum_required(VERSION 3.15)
project(transformer_cpp CUDA CXX)

# Set CUDA architecture policy
cmake_policy(SET CMP0104 NEW)

# Add Doxygen support
find_package(Doxygen)
if(DOXYGEN_FOUND)
    add_custom_target(docs
        ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set default CUDA architectures if not specified
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# Include FetchContent for downloading dependencies
include(FetchContent)

# Find required packages
find_package(CUDAToolkit)

# Find or fetch MPI for distributed training
find_package(MPI QUIET)

if(NOT MPI_FOUND)
    message(STATUS "MPI not found locally - creating minimal MPI stub for single-node training")
    
    # Create minimal MPI stub for development/testing
    set(MPI_STUB_DIR ${CMAKE_BINARY_DIR}/mpi_stub)
    file(MAKE_DIRECTORY ${MPI_STUB_DIR}/include)
    
    # Create minimal mpi.h stub
    file(WRITE ${MPI_STUB_DIR}/include/mpi.h "
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
")
    
    # Set MPI variables for our stub
    set(MPI_CXX_FOUND TRUE)
    set(MPI_CXX_INCLUDE_DIRS ${MPI_STUB_DIR}/include)
    set(MPI_CXX_LIBRARIES "")  # No libraries needed for stub
    
    message(STATUS "MPI stub created for single-node development")
    message(WARNING "Using MPI stub - install real MPI for multi-node distributed training")
endif()

if(MPI_FOUND OR MPI_CXX_FOUND)
    message(STATUS "MPI available - enabling distributed training support")
    add_definitions(-DMPI_AVAILABLE)
    
    # Find or fetch NCCL for GPU communication (if CUDA available)
    if(CUDAToolkit_FOUND)
        find_path(NCCL_INCLUDE_DIR nccl.h HINTS ${CUDA_TOOLKIT_ROOT_DIR}/include)
        find_library(NCCL_LIBRARY nccl HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib)
        
        if(NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
            message(STATUS "NCCL not found locally - creating NCCL stub for development")
            
            # Create NCCL stub directory
            set(NCCL_STUB_DIR ${CMAKE_BINARY_DIR}/nccl_stub)
            file(MAKE_DIRECTORY ${NCCL_STUB_DIR}/include)
            
            # Create minimal nccl.h stub
            file(WRITE ${NCCL_STUB_DIR}/include/nccl.h "
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
inline const char* ncclGetErrorString(ncclResult_t result) { return \"ncclSuccess\"; }
")
            
            # Set NCCL variables for stub
            set(NCCL_INCLUDE_DIR ${NCCL_STUB_DIR}/include)
            set(NCCL_LIBRARY "")  # No library needed for stub
            
            message(STATUS "NCCL stub created for single-GPU development")
            message(WARNING "Using NCCL stub - install real NCCL for multi-GPU distributed training")
        endif()
        
        if(NCCL_INCLUDE_DIR)
            message(STATUS "NCCL available - enabling GPU-to-GPU communication")
            add_definitions(-DNCCL_AVAILABLE)
        endif()
    endif()
else()
    message(STATUS "MPI not available - building without distributed training support")
endif()

# Set CUDA availability flag
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA found - enabling CUDA support")
    add_definitions(-DCUDA_AVAILABLE)
else()
    message(STATUS "CUDA not found - building without CUDA support")
endif()

# Optional: Find OpenMP
if(MSVC)
    find_package(OpenMP QUIET)
else()
    find_package(OpenMP)
endif()

# Store OpenMP status for later use
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found - enabling parallel processing support")
    if(MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")
    endif()
else()
    message(WARNING "OpenMP not found - parallel processing will be disabled")
endif()

# Download and build nlohmann/json
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(FETCHCONTENT_QUIET OFF)
FetchContent_MakeAvailable(json)

# Include directories
if(CUDAToolkit_FOUND)
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

# Base include directories - single source of truth for all includes
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
)

# Add tiktoken include directory globally
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tiktoken/cpp/include
)

# Core source files
set(CORE_SOURCES
    src/components.cpp
    src/vector_ops.cpp
    src/logger.cpp
    src/model_saver.cpp
    src/half_precision.cpp
    src/gradient_checkpoint.cpp
    src/utils.cpp
    src/performance_metrics.cpp
    src/gqa.cpp
    src/beam_search.cpp
    src/config.cpp
    src/tiktoken_tokenizer.cpp
    src/matrix.cpp
    src/data_augmentation.cpp
    src/lm_head.cpp
    src/hyperparameter_tuner.cpp
    src/count_vocabulary.cpp
    src/scope_logger.cpp
    # Add training components
    src/training/loss_tracker.cpp
    src/training/gradient_manager.cpp
    src/training/learning_rate_scheduler.cpp
    src/training/training_state_manager.cpp
    src/training/training_monitor.cpp
    src/phrase_analysis.cpp
    src/serialization.cpp
    src/tensor.cpp
    src/text_analysis.cpp
    src/quantization.cpp
    src/router.cpp
    src/moe.cpp
    src/optimizer/adam.cpp
    src/optimizer.cpp
    src/p2p_network.cpp
)

# Neural network components
set(NN_SOURCES
    src/attention.cpp
    src/transformer.cpp
    src/layer_norm.cpp
    src/feed_forward.cpp
    src/embeddings.cpp
)

# Distributed training components (conditional)
if(MPI_FOUND)
    set(DISTRIBUTED_SOURCES
        src/distributed_transformer.cpp
    )
endif()

# Training and utilities
set(UTIL_SOURCES
    src/cache.cpp
    src/optimizer.cpp
)

# Language model components
set(LM_SOURCES
    src/vocabulary.cpp
)

# Optimization components
set(OPTIMIZER_SOURCES
    src/optimizer/sam.cpp
)

# Add the tensor source file to the sources list
set(SOURCES
    ${SOURCES}
    src/tensor.cpp
)

# Add the tensor header to the includes
set(HEADERS
    ${HEADERS}
    include/tensor.hpp
)

# Add the debug header to the includes
set(HEADERS
    ${HEADERS}
    include/debug.hpp
)

# Add CUDA compilation flags
if(CUDAToolkit_FOUND)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 -arch=sm_60)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
endif()

# Create library with all components
add_library(transformer_lib
    ${CORE_SOURCES}
    ${NN_SOURCES}
    ${UTIL_SOURCES}
    ${LM_SOURCES}
    ${OPTIMIZER_SOURCES}
    ${DISTRIBUTED_SOURCES}
    ${SOURCES}
)

# Set CUDA specific properties for the library
set_target_properties(transformer_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Add OpenMP configuration if found
if(OpenMP_CXX_FOUND)
    target_compile_definitions(transformer_lib PUBLIC USE_OPENMP)
    target_compile_options(transformer_lib PRIVATE ${OpenMP_CXX_FLAGS})
    if(NOT MSVC)
        target_compile_options(transformer_lib PRIVATE -fopenmp)
    endif()
    target_link_libraries(transformer_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

# Create single executable
add_executable(transformer src/main.cpp)

# Link everything to the executable
target_link_libraries(transformer
    PRIVATE
    transformer_lib
    nlohmann_json::nlohmann_json
)

# Create distributed training example if MPI is available
if(MPI_FOUND OR MPI_CXX_FOUND)
    add_executable(distributed_training_example examples/distributed_training_example.cpp)
    target_link_libraries(distributed_training_example
        PRIVATE
        transformer_lib
        nlohmann_json::nlohmann_json
    )
    
    # Set properties for MPI executable (only if system MPI)
    if(MPI_FOUND)
        set_target_properties(distributed_training_example PROPERTIES
            COMPILE_FLAGS "${MPI_COMPILE_FLAGS}"
            LINK_FLAGS "${MPI_LINK_FLAGS}"
        )
    endif()
    
    message(STATUS "Distributed training example will be built")
endif()

if(CUDAToolkit_FOUND)
    target_link_libraries(transformer_lib
        PUBLIC
        CUDA::cudart
        CUDA::cublas
        CUDA::curand
    )
endif()

# Compiler options
target_compile_definitions(transformer_lib
    PUBLIC
    USE_CUDA
)

if(MSVC)
    target_compile_options(transformer_lib
        PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler="/W3">
    )
else()
    target_compile_options(transformer_lib
        PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
    )
endif()

# Installation
install(TARGETS transformer transformer_lib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.cuh"
)

# Optional dependencies
find_library(TCMALLOC_LIB tcmalloc)
if(TCMALLOC_LIB)
    message(STATUS "tcmalloc found: ${TCMALLOC_LIB}")
    target_link_libraries(transformer_lib PUBLIC ${TCMALLOC_LIB})
else()
    message(WARNING "tcmalloc not found, continuing without it")
endif()

# Link OpenMP if found
if(OpenMP_CXX_FOUND)
    target_link_libraries(transformer_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

# Link MPI if available
if(MPI_FOUND)
    # System MPI
    target_link_libraries(transformer_lib PUBLIC MPI::MPI_CXX)
    target_include_directories(transformer_lib PUBLIC ${MPI_INCLUDE_PATH})
elseif(MPI_CXX_FOUND)
    # Fetched MPI
    target_include_directories(transformer_lib PUBLIC ${MPI_CXX_INCLUDE_DIRS})
    target_link_libraries(transformer_lib PUBLIC ${MPI_CXX_LIBRARIES})
endif()

# Link NCCL if available
if(NCCL_INCLUDE_DIR)
    target_include_directories(transformer_lib PUBLIC ${NCCL_INCLUDE_DIR})
    if(NCCL_LIBRARY)
        target_link_libraries(transformer_lib PUBLIC ${NCCL_LIBRARY})
    endif()
endif()

# Add CUDA kernels library only if CUDA is found
if(CUDAToolkit_FOUND)
    add_library(cuda_kernels STATIC
        src/cuda/matrix_ops.cu
        src/cuda/attention_ops.cu
        src/cuda/backward_ops.cu
        src/cuda/feed_forward_kernels.cu
        src/cuda/half_precision_kernels.cu
        src/cuda/cuda_utils.cu
        src/cuda/cuda_init.cu
        src/cuda/token_embedding_cuda.cu
        src/cuda/layernorm_cuda.cu
        src/cuda/gqa_kernels.cu
        src/cuda/beam_search_ops.cu
        src/cuda/tokenizer_kernels.cu
        src/cuda/swiglu_kernel.cu
        src/cuda/router_kernel.cu
        src/cuda/moe_kernel.cu
    )

    set_target_properties(cuda_kernels PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        LANGUAGE CUDA
    )

    # Add CUDA include directories
    target_include_directories(cuda_kernels
        PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/third_party
    )
    
    # Add nlohmann_json include directory for CUDA kernels
    target_include_directories(cuda_kernels
        PRIVATE
        $<TARGET_PROPERTY:nlohmann_json::nlohmann_json,INTERFACE_INCLUDE_DIRECTORIES>
    )

    # Link CUDA libraries to cuda_kernels
    target_link_libraries(cuda_kernels
        PUBLIC
        CUDA::cudart
        CUDA::cublas
        CUDA::curand
        nlohmann_json::nlohmann_json
    )

    # Link cuda_kernels to transformer_lib
    target_link_libraries(transformer_lib
        PUBLIC
        cuda_kernels
    )
endif()

# Add SOURCES to the library instead
target_sources(transformer_lib PRIVATE
    src/transformer.cpp
    src/attention.cpp
    src/feed_forward.cpp
    src/layer_norm.cpp
    src/dropout.cpp
    src/p2p_network.cpp
    src/serialization.cpp
    src/router.cpp
    src/moe.cpp
    src/components.cpp
    src/utils.cpp
    src/config.cpp
    src/main.cpp
    src/multi_head_attention.cpp
    src/embedding.cpp
    src/positional_encoding.cpp
    src/logger.cpp
    src/tiktoken_tokenizer.cpp
    src/cuda/matrix_ops.cu
    src/cuda/swiglu_kernel.cu
)

find_package(nlohmann_json 3.2.0 REQUIRED)
target_link_libraries(transformer_lib PRIVATE nlohmann_json::nlohmann_json)

# Make sure the include directory is in the include path
target_include_directories(transformer_lib 
    PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
)

add_definitions(
    -DPAD_TOKEN_ID=0
    -DUNK_TOKEN_ID=1
    -DBOS_TOKEN_ID=2
    -DEOS_TOKEN_ID=3
    -DMASK_TOKEN_ID=4
)

# Simplified tiktoken data copying
add_custom_command(TARGET transformer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/tiktoken_data"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/third_party/tiktoken/tiktoken/tiktoken_data"
        "${CMAKE_BINARY_DIR}/tiktoken_data"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/third_party/tiktoken/tiktoken/cl100k_base.vocab.json"
        "${CMAKE_SOURCE_DIR}/third_party/tiktoken/tiktoken/cl100k_base.merges.json"
        "${CMAKE_BINARY_DIR}/tiktoken_data/"
    COMMENT "Copying tiktoken data files to build directory"
)

if(CUDAToolkit_FOUND)
    # Set source file properties to compile with CUDA
    set_source_files_properties(
        src/lm_head.cpp
        PROPERTIES
        LANGUAGE CUDA
    )
endif()

find_package(OpenSSL REQUIRED)
target_link_libraries(transformer_lib PRIVATE ${OpenSSL_LIBRARIES})