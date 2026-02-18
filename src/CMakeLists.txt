# TODO: should not use this
if (WIN32)
    if (BUILD_SHARED_LIBS)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
endif()

#
# libraries
#

# llama

if (LLAMA_ALL_WARNINGS)
    if (NOT MSVC)
        list(APPEND WARNING_FLAGS -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wno-sign-compare)
        if (APPLE)
            # shut up c99 extensions warning I get on my system due to arm_neon.h
            list(APPEND WARNING_FLAGS -Wno-c99-extensions)
        endif()
        list(APPEND C_FLAGS       -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes
                                  -Werror=implicit-int -Werror=implicit-function-declaration)
        list(APPEND CXX_FLAGS     -Wmissing-declarations -Wmissing-noreturn)

        list(APPEND C_FLAGS   ${WARNING_FLAGS})
        list(APPEND CXX_FLAGS ${WARNING_FLAGS})

        get_flags(${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})

        add_compile_options("$<$<COMPILE_LANGUAGE:C>:${C_FLAGS};${GF_C_FLAGS}>"
                            "$<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS};${GF_CXX_FLAGS}>")
    else()
        # todo : msvc
        set(C_FLAGS   "")
        set(CXX_FLAGS "")
    endif()
endif()


add_library(llama
            ../include/llama.h
            llama.cpp
            llama-vocab.cpp
            llama-grammar.cpp
            llama-sampling.cpp
            llama-mmap.cpp
            llama-model-loader.cpp
            llama-load-tensors.cpp
            llama-build-context.h
            llama-build-context.cpp
            llama-model.h
            llama-model.cpp
            llama-quantize.cpp
            llama-arch.h
            llama-arch.cpp
            llama-hparams.h
            llama-hparams.cpp
            llama-delta-net.h
            llama-delta-net.cpp
            unicode.h
            unicode.cpp
            unicode-data.cpp
            )

target_include_directories(llama PUBLIC . ../include)
target_include_directories(llama PRIVATE ../ggml/src)
target_compile_features   (llama PUBLIC cxx_std_17)

target_link_libraries(llama PUBLIC ggml)

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_SHARED LLAMA_BUILD)
endif()
