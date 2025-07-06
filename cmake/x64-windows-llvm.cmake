# Base Clang→MSVC toolchain for x86_64-windows
# This file sets up the compiler and target triple, plus common tuning flags.

# Cross‑compile to Windows x86_64
set(CMAKE_SYSTEM_NAME      Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

# Use Clang from Visual Studio's LLVM toolchain
set(CMAKE_C_COMPILER   clang)
set(CMAKE_CXX_COMPILER clang++)

# Target triple for MSVC ABI compatibility
set(target x86_64-pc-windows-msvc)
set(CMAKE_C_COMPILER_TARGET   ${target})
set(CMAKE_CXX_COMPILER_TARGET ${target})

# Common architecture tuning flags (no ISA extensions here)
# - Base x86_64 instruction set
# - Fast vectorization and FP model
# - Disable finite-math-only
set(arch_c_flags
  "-march=x86-64 \
   -fvectorize \
   -ffp-model=fast \
   -fno-finite-math-only"
)

# Warning suppression for this codebase
set(warn_c_flags
  "-Wno-format \
   -Wno-unused-variable \
   -Wno-unused-function \
   -Wno-gnu-zero-variadic-macro-arguments"
)

# Instruction set extensions based on GGML options
if(DEFINED GGML_AVX2)
  set(arch_c_flags "${arch_c_flags} -mavx2 -mfma -mf16c -mpopcnt")
endif()

if(DEFINED GGML_AVX512)
  set(arch_c_flags "${arch_c_flags} -mavx512f -mavx512vl -mavx512bw -mavx512dq")
endif()

if(DEFINED GGML_BF16)
  set(arch_c_flags "${arch_c_flags} -mavx512bf16")
endif()

# Initialize flags for C and C++
set(CMAKE_C_FLAGS_INIT   "${arch_c_flags} ${warn_c_flags}")
set(CMAKE_CXX_FLAGS_INIT "${arch_c_flags} ${warn_c_flags}")
