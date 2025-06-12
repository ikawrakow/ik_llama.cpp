set(TARGET rpc-server)
add_executable(${TARGET} rpc-server.cpp)
target_link_libraries(${TARGET} PRIVATE ggml)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
if (MSVC)
	target_link_options(${TARGET} PRIVATE
	 $<$<CONFIG:DEBUG>:/STACK:20971520,1048576 >
	  $<$<CONFIG:RELEASE>:/STACK:20971520,1048576>
	  )
endif()