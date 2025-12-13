set(TARGET llama-server)
option(LLAMA_SERVER_VERBOSE "Build verbose logging option for Server" ON)
option(LLAMA_SERVER_SSL     "Build SSL support for the server"        OFF)
option(LLAMA_SERVER_SQLITE3 "Build SQlite3 support for the server"    OFF)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

if (MINGW)
    # fix: https://github.com/ggerganov/llama.cpp/actions/runs/9651004652/job/26617901362?pr=8006
    add_compile_definitions(_WIN32_WINNT=${GGML_WIN_VER})
endif()

set(TARGET_SRCS
    server.cpp
    httplib.h
    server-task.cpp
    server-task.h
    server-queue.cpp
    server-queue.h
    server-common.cpp
    server-common.h
    server-context.cpp
    server-context.h
)
set(PUBLIC_ASSETS
    index.html.gz

)

foreach(asset ${PUBLIC_ASSETS})
    set(input "${CMAKE_CURRENT_SOURCE_DIR}/public/${asset}")
    set(output "${CMAKE_CURRENT_BINARY_DIR}/${asset}.hpp")
    list(APPEND TARGET_SRCS ${output})
    add_custom_command(
        DEPENDS "${input}"
        OUTPUT "${output}"
        COMMAND "${CMAKE_COMMAND}" "-DINPUT=${input}" "-DOUTPUT=${output}" -P "${PROJECT_SOURCE_DIR}/scripts/xxd.cmake"
    )
	message("TARGET_SRCS contains: ${input}")
	set_source_files_properties(${output} PROPERTIES GENERATED TRUE)
																
endforeach()

# include new llamacpp webui
set(ALT_PUBLIC_ASSETS
    index_llamacpp.html.gz
	loading.html
)

foreach(asset ${ALT_PUBLIC_ASSETS})
    set(input "${CMAKE_CURRENT_SOURCE_DIR}/public_llamacpp/${asset}")
    set(output "${CMAKE_CURRENT_BINARY_DIR}/${asset}.hpp")
    list(APPEND TARGET_SRCS ${output})
    add_custom_command(
        DEPENDS "${input}"
        OUTPUT "${output}"
        COMMAND "${CMAKE_COMMAND}" "-DINPUT=${input}" "-DOUTPUT=${output}" -P "${PROJECT_SOURCE_DIR}/scripts/xxd.cmake"
    )
	message("TARGET_SRCS contains: ${input}")
	set_source_files_properties(${output} PROPERTIES GENERATED TRUE)
																
endforeach()


add_executable(${TARGET} ${TARGET_SRCS})
install(TARGETS ${TARGET} RUNTIME)
target_compile_definitions(${TARGET} PRIVATE
    SERVER_VERBOSE=$<BOOL:${LLAMA_SERVER_VERBOSE}>
)
if (MSVC)
	target_link_options(${TARGET} PRIVATE
	 $<$<CONFIG:DEBUG>:/STACK:20971520,1048576 >
	  $<$<CONFIG:RELEASE>:/STACK:20971520,1048576>
	  )
endif()
# target_link_libraries(${TARGET} PRIVATE "/STACK:104857600")
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PRIVATE common ${CMAKE_THREAD_LIBS_INIT})

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_link_libraries(${TARGET} PRIVATE common mtmd ${CMAKE_THREAD_LIBS_INIT})

if (LLAMA_SERVER_SSL)
    find_package(OpenSSL REQUIRED)
    target_link_libraries(${TARGET} PRIVATE OpenSSL::SSL OpenSSL::Crypto)
    target_compile_definitions(${TARGET} PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
endif()

if (LLAMA_SERVER_SQLITE3)
    find_package(SQLite3  REQUIRED)
    target_link_libraries(${TARGET} PRIVATE SQLite::SQLite3)
    target_include_directories(${TARGET} PUBLIC ./sqlite_modern_cpp/hdr)
    target_compile_definitions(${TARGET} PRIVATE SQLITE3_MODERN_CPP_SUPPORT)
endif()

if (WIN32)
    TARGET_LINK_LIBRARIES(${TARGET} PRIVATE ws2_32)
endif()

target_compile_features(${TARGET} PRIVATE cxx_std_17)
