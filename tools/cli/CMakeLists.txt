# llama-cli-official: official llama.cpp CLI design running on the ik engine
# the CLI is a client that spawns the ik llama-server (llama_server symbol)
# or connects to a running one via --server-base

set(TARGET llama-cli-official)

add_executable(${TARGET}
    main.cpp
    cli.cpp
    cli-client.cpp
    cli-context.cpp
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../server
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE
    llama-server-impl
    common
    cpp-httplib
    ${CMAKE_THREAD_LIBS_INIT}
)
install(TARGETS ${TARGET} RUNTIME)
