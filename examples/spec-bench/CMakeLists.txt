set(TARGET llama-spec-bench)
set(SPEC_BENCH_PROMPT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/prompts")
set(SPEC_BENCH_PROMPT_FILES
    "${SPEC_BENCH_PROMPT_DIR}/code.txt"
    "${SPEC_BENCH_PROMPT_DIR}/extract.txt"
    "${SPEC_BENCH_PROMPT_DIR}/story.txt"
)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${SPEC_BENCH_PROMPT_FILES})

function(spec_bench_escape_cpp_string output_var input_value)
    string(REPLACE "\\" "\\\\" value "${input_value}")
    string(REPLACE [=["]=] [=[\"]=] value "${value}")
    string(REPLACE "\r" "\\r" value "${value}")
    string(REPLACE "\n" "\\n" value "${value}")
    set(${output_var} "${value}" PARENT_SCOPE)
endfunction()

file(READ "${SPEC_BENCH_PROMPT_DIR}/code.txt" SPEC_BENCH_PROMPT_CODE_RAW)
file(READ "${SPEC_BENCH_PROMPT_DIR}/extract.txt" SPEC_BENCH_PROMPT_EXTRACT_RAW)
file(READ "${SPEC_BENCH_PROMPT_DIR}/story.txt" SPEC_BENCH_PROMPT_STORY_RAW)
string(REGEX REPLACE "\n$" "" SPEC_BENCH_PROMPT_CODE_RAW "${SPEC_BENCH_PROMPT_CODE_RAW}")
string(REGEX REPLACE "\n$" "" SPEC_BENCH_PROMPT_EXTRACT_RAW "${SPEC_BENCH_PROMPT_EXTRACT_RAW}")
string(REGEX REPLACE "\n$" "" SPEC_BENCH_PROMPT_STORY_RAW "${SPEC_BENCH_PROMPT_STORY_RAW}")
string(REGEX REPLACE "\r$" "" SPEC_BENCH_PROMPT_CODE_RAW "${SPEC_BENCH_PROMPT_CODE_RAW}")
string(REGEX REPLACE "\r$" "" SPEC_BENCH_PROMPT_EXTRACT_RAW "${SPEC_BENCH_PROMPT_EXTRACT_RAW}")
string(REGEX REPLACE "\r$" "" SPEC_BENCH_PROMPT_STORY_RAW "${SPEC_BENCH_PROMPT_STORY_RAW}")
spec_bench_escape_cpp_string(SPEC_BENCH_PROMPT_CODE "${SPEC_BENCH_PROMPT_CODE_RAW}")
spec_bench_escape_cpp_string(SPEC_BENCH_PROMPT_EXTRACT "${SPEC_BENCH_PROMPT_EXTRACT_RAW}")
spec_bench_escape_cpp_string(SPEC_BENCH_PROMPT_STORY "${SPEC_BENCH_PROMPT_STORY_RAW}")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/spec-bench-prompts.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/spec-bench-prompts.h"
    @ONLY
)


add_executable(${TARGET} spec-bench.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_include_directories(${TARGET} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
