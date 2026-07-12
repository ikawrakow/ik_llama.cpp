set(TARGET llama-spec-bench)

set(SPEC_BENCH_FIXTURE_DIR "${CMAKE_CURRENT_BINARY_DIR}/fixtures")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/youtube-extract.txt"
    "${SPEC_BENCH_FIXTURE_DIR}/youtube-extract.txt"
    COPYONLY
)

add_executable(${TARGET} spec-bench.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_compile_definitions(${TARGET} PRIVATE SPEC_BENCH_FIXTURE_DIR=\"${SPEC_BENCH_FIXTURE_DIR}\")
