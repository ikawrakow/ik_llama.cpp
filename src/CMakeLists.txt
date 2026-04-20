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
            unicode-script-data.cpp
            # graphs
            graphs/build_llama.cpp
            graphs/build_mistral3.cpp
            graphs/build_deci.cpp
            graphs/build_baichuan.cpp
            graphs/build_xverse.cpp
            graphs/build_falcon.cpp
            graphs/build_grok.cpp
            graphs/build_dbrx.cpp
            graphs/build_starcoder.cpp
            graphs/build_starcoder2.cpp
            graphs/build_refact.cpp
            graphs/build_bert.cpp
            graphs/build_bloom.cpp
            graphs/build_mpt.cpp
            graphs/build_stablelm.cpp
            graphs/build_seedoss.cpp
            graphs/build_step35.cpp
            graphs/build_qwen.cpp
            graphs/build_qwen2.cpp
            graphs/build_qwen3.cpp
            graphs/build_qwen3next.cpp
            graphs/build_qwen35.cpp
            graphs/build_phi2.cpp
            graphs/build_phi3.cpp
            graphs/build_plamo.cpp
            graphs/build_gpt2.cpp
            graphs/build_codeshell.cpp
            graphs/build_orion.cpp
            graphs/build_internlm2.cpp
            graphs/build_minicpm.cpp
            graphs/build_gemma.cpp
            graphs/build_gemma2.cpp
            graphs/build_gemma3.cpp
            graphs/build_gemma4.cpp
            graphs/build_mamba.cpp
            graphs/build_command_r.cpp
            graphs/build_olmo.cpp
            graphs/build_openelm.cpp
            graphs/build_gptneox.cpp
            graphs/build_arctic.cpp
            graphs/build_deepseek2.cpp
            graphs/build_glm4.cpp
            graphs/build_bitnet.cpp
            graphs/build_cohere2.cpp
            graphs/build_t5.cpp
            graphs/build_jais.cpp
            graphs/build_chatglm.cpp
            graphs/build_dots1.cpp
            graphs/build_ernie45.cpp
            graphs/build_hunyuan.cpp
            graphs/build_mimo2.cpp
            graphs/build_openai.cpp
            graphs/build_bailingmoe2.cpp
            graphs/build_minimaxm2.cpp
            graphs/build_smollm3.cpp
            )

target_include_directories(llama PUBLIC . ../include)
target_include_directories(llama PRIVATE ../ggml/src)
target_compile_features   (llama PUBLIC cxx_std_17)

target_link_libraries(llama PUBLIC ggml)

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_SHARED LLAMA_BUILD)
endif()

if (GGML_MAX_CONTEXTS)
    add_compile_definitions(GGML_MAX_CONTEXTS=${GGML_MAX_CONTEXTS})
endif()
