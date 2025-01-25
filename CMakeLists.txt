cmake_minimum_required(VERSION 3.10)
project(transformer CUDA CXX)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

# Find required packages
find_package(CUDA REQUIRED)
enable_language(CUDA)

# Set CUDA paths more explicitly
set(CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_PATH} CACHE PATH "CUDA Toolkit location")
set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/include CACHE PATH "CUDA Include Directory")

# Set minimal CUDA flags needed for compilation
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    -O3
    --expt-relaxed-constexpr
    -gencode arch=compute_60,code=sm_60
)

# Include FetchContent for downloading dependencies
include(FetchContent)

# Download and build nlohmann/json
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(json)

# Get nlohmann/json include directory
FetchContent_GetProperties(json)
get_target_property(JSON_INC_DIR nlohmann_json::nlohmann_json INTERFACE_INCLUDE_DIRECTORIES)

# Set CUDA architectures if not specified
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 60)
endif()

# Add CUDA-specific compile definitions
add_definitions(-DUSE_CUDA)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${JSON_INC_DIR}
)

# Add CUDA source files
set(CUDA_SOURCES
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
)

# Set CUDA specific compile options
foreach(cuda_source ${CUDA_SOURCES})
    set_source_files_properties(${cuda_source} PROPERTIES
        CUDA_SOURCE_PROPERTY_FORMAT OBJ
        CUDA_SEPARABLE_COMPILATION ON
        LANGUAGE CUDA
    )
endforeach()

# Create CUDA library
cuda_add_library(cuda_kernels STATIC ${CUDA_SOURCES})

# Set include directories for CUDA kernels
target_include_directories(cuda_kernels
    PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/cuda
    ${CUDA_INCLUDE_DIRS}
    ${CUDA_TOOLKIT_ROOT_DIR}/include
)

# Link CUDA libraries (using plain syntax)
target_link_libraries(cuda_kernels
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_curand_LIBRARY}
)

# Set CUDA specific properties
set_target_properties(cuda_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)

# Add source files
set(SOURCES
    src/main.cpp
    src/transformer.cpp
    src/matrix.cpp
    src/feed_forward.cpp
    src/attention.cpp
    src/logger.cpp
    src/utils.cpp
    src/tokenizer.cpp
    src/model_saver.cpp
    src/language_model_head.cpp
    src/components.cpp
    src/vector_ops.cpp
    src/half_precision.cpp
    src/gradient_checkpoint.cpp
    src/performance_metrics.cpp
    src/gqa.cpp
    src/beam_search.cpp
    src/config.cpp
    src/tiktoken_tokenizer.cpp
    src/cache.cpp
    src/quantization.cpp
    src/optimizer.cpp
    src/lm_head.cpp
    src/vocabulary.cpp
    src/tensor.cpp
    src/optimizer/sam.cpp
)

# Create main library
add_library(transformer_lib STATIC ${SOURCES})

# Set include directories for transformer_lib
target_include_directories(transformer_lib
    PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
    ${JSON_INC_DIR}  # Add json include directory
)

# Link libraries for transformer_lib
target_link_libraries(transformer_lib 
    PUBLIC
    cuda_kernels
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    nlohmann_json::nlohmann_json
)

# Create executable
add_executable(transformer src/main.cpp)
target_link_libraries(transformer 
    PRIVATE
    transformer_lib
)

# Optional: Find and link OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(transformer_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

# Add definitions
add_definitions(
    -DPAD_TOKEN_ID=0
    -DUNK_TOKEN_ID=1
    -DBOS_TOKEN_ID=2
    -DEOS_TOKEN_ID=3
    -DMASK_TOKEN_ID=4
    -DCUDA_AVAILABLE
)

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

if(CUDA_FOUND)
    # Set source file properties to compile with CUDA
    set_source_files_properties(
        src/lm_head.cpp
        PROPERTIES
        LANGUAGE CUDA
    )
endif()