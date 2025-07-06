# Tell CMake we’re cross‑compiling (even though it’s Windows→Windows, 
# we’re “crossing” into Clang’s GNU‑like driver)
set( CMAKE_SYSTEM_NAME        Windows )
set( CMAKE_SYSTEM_PROCESSOR   x86_64 )

# Use Clang from VS’s LLVM toolchain
set( CMAKE_C_COMPILER    clang )
set( CMAKE_CXX_COMPILER  clang++ )

# Target triple
set( target x86_64-pc-windows-msvc )
set( CMAKE_C_COMPILER_TARGET   ${target} )
set( CMAKE_CXX_COMPILER_TARGET ${target} )

# Architecture flags: enable the instruction sets we need
set( arch_c_flags
  "-march=x86-64+f16c+avx2+fma+popcnt \
   -fvectorize \
   -ffp-model=fast \
   -fno-finite-math-only"
)

# Suppress some warnings
set( warn_c_flags
  "-Wno-format   \
   -Wno-unused-variable \
   -Wno-unused-function \
   -Wno-gnu-zero-variadic-macro-arguments"
)

# Apply to both C and C++
set( CMAKE_C_FLAGS_INIT   "${arch_c_flags} ${warn_c_flags}" )
set( CMAKE_CXX_FLAGS_INIT "${arch_c_flags} ${warn_c_flags}" )
