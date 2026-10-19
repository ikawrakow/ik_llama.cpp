set(TARGET llama-cpu-power)
add_executable(${TARGET}
    cpu-power.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/cpu_power.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../common)
target_link_libraries(${TARGET} PRIVATE Threads::Threads)
target_compile_features(${TARGET} PRIVATE cxx_std_11)
