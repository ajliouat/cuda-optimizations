cmake_minimum_required(VERSION 3.14)
project(CUDAOptimizations)

set(CMAKE_CUDA_ARCHITECTURES 75)  # Adjust for your GPU architecture
set(CMAKE_CXX_STANDARD 17)

find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)

# Add CUDA kernels
cuda_add_library(cuda_ops SHARED
    src/cuda/attention.cu
    src/cuda/reduction.cu
    src/cuda/memory_optimization.cu
)

target_include_directories(cuda_ops PUBLIC ${CMAKE_SOURCE_DIR}/src/include)
target_link_libraries(cuda_ops PUBLIC ${TORCH_LIBRARIES})