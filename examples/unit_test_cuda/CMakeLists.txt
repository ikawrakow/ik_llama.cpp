# unit_test_cuda - byte-for-byte Q8_0 CUDA quantizer verification
#
# Links the real ggml/llama libs (like llama-cuda-pinmem) so the test exercises
# the exact production paths llama-quantize uses:
#   - GPU: ggml_cuda_quantize_q8_0  (ggml/src/ggml-cuda/quantize_gguf.cu)
#   - CPU: ggml_quantize_chunk      (quantize_q8_0 -> quantize_row_q8_0_ref)
# CUDA::cudart is only needed for the device-count/properties diagnostics.

cmake_minimum_required(VERSION 3.14)
project(unit_test_cuda C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(unit_test_cuda
    ../unit_test_cuda.cpp
)

target_include_directories(unit_test_cuda PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/include
)

target_compile_definitions(unit_test_cuda PRIVATE
    GGML_USE_CUDA
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(unit_test_cuda PRIVATE
    llama
    CUDA::cudart
    ${CMAKE_THREAD_LIBS_INIT}
)
