cmake_minimum_required(VERSION 3.18)
project(transformer_cpp CUDA CXX)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA packages first
find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

# Set CUDA architecture based on your GPU
set(CMAKE_CUDA_ARCHITECTURES 60 70 75)  # Support multiple architectures

# Find required packages
find_package(CUDA REQUIRED)

# Set CUDA paths
if(DEFINED ENV{CUDA_PATH})
    set(CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_PATH})
else()
    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
endif()

# Set CUDA include paths
set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/include)

# Set basic CUDA flags with C++17 support
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    "-O3"
    "-std=c++17"
    "-Xcompiler=-fPIC"
    "-arch=sm_60"
    "--use_fast_math"
    "--expt-relaxed-constexpr"
    "--extended-lambda"
)

# Set C++17 for both host and device code
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++17")

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

# Add CUDA-specific compile definitions
add_definitions(-DUSE_CUDA)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/cuda
    ${CMAKE_SOURCE_DIR}/third_party
    ${CUDAToolkit_INCLUDE_DIRS}
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
        LANGUAGE CUDA
        CUDA_SOURCE_PROPERTY_FORMAT OBJ
        CUDA_SEPARABLE_COMPILATION ON
    )
endforeach()

# Create CUDA library
add_library(cuda_kernels STATIC ${CUDA_SOURCES})

# Set language for cuda_kernels target
set_target_properties(cuda_kernels PROPERTIES
    LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Set include directories for CUDA kernels
target_include_directories(cuda_kernels
    PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/cuda
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Link CUDA libraries using modern CMake
target_link_libraries(cuda_kernels
    PUBLIC
    CUDA::cudart
    CUDA::cublas
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
    src/layer_norm.cpp
    src/embeddings.cpp
    src/data_augmentation.cpp
    src/vector.cpp
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
    CUDA::cudart
    CUDA::cublas
    nlohmann_json::nlohmann_json
)

# Create executable
add_executable(transformer_cpp ${SOURCES})
target_link_libraries(transformer_cpp 
    PRIVATE
    transformer_lib
    CUDA::cudart
    CUDA::cublas
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

target_compile_definitions(transformer_cpp PRIVATE
    USE_CUDA
    CUDA_AVAILABLE
)

target_compile_definitions(cuda_kernels PRIVATE
    USE_CUDA
    CUDA_AVAILABLE
)

# Installation
install(TARGETS transformer_cpp transformer_lib
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
add_custom_command(TARGET transformer_cpp POST_BUILD
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