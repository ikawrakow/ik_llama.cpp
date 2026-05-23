# Build ik_llama.cpp

## Supported backends

> **Important:** `ik_llama.cpp` focuses on two fully functional and performant compute backends:
>
> - **CPU** — AVX2 required; AVX-512-capable CPUs (AMD Zen4 / Intel Sapphire Rapids+) are supported and unlock additional performance for quantized prompt processing.
> - **CUDA** — Turing architecture (RTX 20xx / GTX 16xx) or newer required
> - **ARM** (armv8.2-a+ / NEON)
>
> Other backends (Metal, ROCm/hipBLAS, Vulkan, SYCL, OpenBLAS, MUSA) are inherited from upstream
> `llama.cpp` and are **not actively maintained or tested** in this fork. Their build instructions
> have been removed to avoid misleading users. If you need one of these, refer to the upstream
> [llama.cpp build documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).

> **Note on `make`:** The Makefile-based build is **obsolete** and has been removed.  
> CMake is the only supported build system.

---

## Prerequisites

| Requirement | Details |
|---|---|
| **Git** | [git-scm.com](https://git-scm.com/download/) or system package manager |
| **CMake 3.21+** | [cmake.org](https://cmake.org/download/) or system package manager |
| **C++17 compiler** | GCC 10+, Clang 13+, MSVC 2022, or clang-cl |
| **CUDA Toolkit 12.x** or higher | Required for GPU builds only — [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) |

**Note**: Avoid using `CUDA Toolkit 13.2` due to a bug in CUDA that might produce garbage in certain combinations of quantizations and models.

Platform-specific installation details are covered in each build section below.

---

## Getting the source

```bash
git clone https://github.com/ikawrakow/ik_llama.cpp
cd ik_llama.cpp
```

---

## Building on Linux

Ensure all prerequisites are installed. On Debian/Ubuntu, this can be done in one step:

```bash
apt-get update && apt-get install -y build-essential git cmake libcurl4-openssl-dev curl libgomp1
```

For CUDA support, additionally install the NVIDIA drivers and CUDA Toolkit via the
[NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive).

### CPU only

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
cmake --build build --config Release -j $(nproc)
```

### CPU + CUDA

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_CUDA=ON
cmake --build build --config Release -j $(nproc)
```

`GGML_NATIVE=ON` enables `-march=native`, which is critical for performance. On AVX-512-capable
hardware (AMD Zen4, Intel Sapphire Rapids+) additional flags may be needed to unlock the full IQK
quantized GEMM path — see the [AVX-512 section](#cpu-build-flags-for-avx-512-zen4--sapphire-rapids).

The environment variable [`CUDA_VISIBLE_DEVICES`](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars)
selects which GPU(s) to use at runtime. To allow swapping to system RAM when VRAM is exhausted,
set `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` before running.

---

## Building on macOS

Ensure all prerequisites are installed, e.g. via Homebrew:

```bash
brew install cmake git
```

### CPU only

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
cmake --build build --config Release -j $(sysctl -n hw.logicalcpu)
```
> **Note**: `-DGGML_NATIVE=ON` will actually enable `Metal` support, which may or may not work, depending on model. Be aware that Metal is not officially supported in ik_llama.  
(Feedback from Mac users on this topic is welcome to help improve this documentation)

---

## Building on Windows (CPU + CUDA)

Windows requires a few extra setup steps because no compiler is included with the OS. The
sections below walk through prerequisites and then the build itself.

### Prerequisites

#### 1. C++ Build Environment (Visual Studio 2022 family)

You need the Microsoft C++ toolchain. This is available via either:

- **Visual Studio Build Tools 2022** (standalone, lightweight):
  [aka.ms/vs/17/release/vs_buildtools.exe](https://aka.ms/vs/17/release/vs_buildtools.exe)
- **Visual Studio 2022** IDE (Community, Professional, or Enterprise)

> **Important:** `Visual Studio` and `Visual Studio Code` (VS Code) are completely different
> products. VS Code is a source code editor, **not** a compiler suite. Having VS Code installed
> is **not sufficient**. You must install one of the Visual Studio 2022 options above.

Regardless of which installer you use, open the **Visual Studio Installer**, select *Modify*,
and ensure the following are checked:

- `Desktop development with C++` workload, **or** individually:
  - `MSVC v143 – VS 2022 C++ x64/x86 build tools`
  - `C++ CMake tools for Windows`
  - `Clang Compiler for Windows` (see next section) *(optional, but recommended for AVX-512 builds)*

#### Compiler Choice: MSVC vs. Clang-CL
You can install and use two different compilers in Visual Studio:
1. MSVC (cl.exe): The default. Simplest setup, but lacks automatic CPU feature detection (-march=native is not supported). You must manually enable AVX2/AVX-512.
2. Clang-CL (clang-cl.exe): Included with VS ("Clang Compiler for Windows" component). Supports -march=native, enabling GGML_NATIVE=ON to auto-detect your CPU's best features (AVX2, AVX-512, etc.). Recommended for maximum performance.

To use the `clang-cl` compiler, you have to add the following lines to your cmake args:
```
    -DCMAKE_C_COMPILER=clang-cl.exe ^
    -DCMAKE_CXX_COMPILER=clang-cl.exe
```
This will be covered later in the detailled configuration examples.

#### A note on Visual Studio 2026

At the time of writing, CUDA Toolkit 12.x and 13.1 are not compatible with Visual Studio 2026. CUDA Toolkit 13.2 should be compatible, but due to bugs in this version, you should avoid using 13.2. You may be able to use CUDA Toolkit 13.3 successfully with Visual Studio Developer Command Prompt 2026 once it is released.

#### 2. CUDA Toolkit 12.x or later

**Download** from the [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive).
During installation, choose *Custom Installation* and uncheck *Driver components* if building
inside a VM without a physical NVIDIA GPU.  
**Avoid** using Version 13.2 of the Toolkit.

#### 3. CMake and Ninja

CMake is included with Visual Studio. If you prefer a standalone installation, or need a newer
version:

```
winget install Kitware.CMake Ninja-build.Ninja
```

Or download from [cmake.org](https://cmake.org/download/).

#### 4. Use the Developer Command Prompt/PowerShell

Always execute build commands from inside the
**Developer Command Prompt for VS 2022** or **Developer PowerShell for VS 2022**, accessible
via the Windows Start Menu under *Visual Studio 2022* or from the "new tab" dropdown in Windows Terminal.

These terminals automatically configure all required compiler paths, environment variables, and
SDK settings. Running `cmake` from a plain `cmd.exe` or PowerShell will likely fail with
"command not found" or wrong-compiler errors due to missing environment initialisation.

**Verification step**  
To confirm your environment is correctly set up, run:
```cmd
cl.exe
clang-cl --version
nvcc --version
```
- `cl.exe` should print a version header and indicate an **x64** target (e.g., `for x64`, `für x64`, or similar depending on system language).
- `clang-cl` should show `Target: x86_64-pc-windows-msvc`.
- `nvcc` should display the installed CUDA version without errors.

If any command is not found or reports a 32-bit target, check if you called the correct Developer shell.

---

The following sections provide explanations about optimizations of a cmake command.  
You may skip them and go directly to the [full recommended CUDA build commands](#full-recommended-cuda-build-commands) for both compilers.

---

### Minimal configuration

The following minimal commands are sufficient to get a working build. Run them inside the
Developer Command Prompt.

**CPU only:**

```bat
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

**CPU + CUDA:**

```bat
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build --config Release
```

When invoked from the Developer Command Prompt, CMake automatically detects MSVC as the host
compiler and nvcc uses it accordingly. No additional compiler flags are required for a basic
build.

---

### Recommended additional CMake options

The minimal configuration above produces a correct build but leaves significant performance on
the table. The following options are strongly recommended and can be appended to either of the
`cmake -B build ...` commands above.

**Enable native CPU optimisations:**

* **If using Clang-CL (Recommended):**  
Add `-DGGML_NATIVE=ON`. This enables -march=native, automatically tuning the binary to your exact CPU (AVX2, AVX-512, etc.).  
*Required Flags:* `-DCMAKE_C_COMPILER=clang-cl.exe -DCMAKE_CXX_COMPILER=clang-cl.exe`  
* **If using MSVC (Default):**  
`GGML_NATIVE=ON` has no effect. You must explicitly enable instruction sets:  
  * For most modern CPUs: Add `-DGGML_AVX2=ON`
  * For Zen4/Sapphire Rapids+: Add `-DGGML_AVX512=ON` (plus VNNI/VBMI flags, see AVX-512 Section)

Since unused cmake args will just give you an informational warning and will otherwise be ignored, you can safely add both to your setup:
```
-DGGML_NATIVE=ON
-DGGML_AVX2=ON
```

> ⚠️ **VM Users:** Be careful with `-DGGML_NATIVE=ON` inside Virtual Machines. The hypervisor may not expose all host CPU features correctly. If you experience crashes, specify your CPU architecture explicitly (e.g., `-DGGML_AVX2=ON`) instead of using `NATIVE`.


**Target a specific CUDA architecture** (avoids compiling for all architectures, which is slow):

```
-DCMAKE_CUDA_ARCHITECTURES="86-real"
```

Replace `86` with the compute capability of your GPU. See the
[CUDA architecture reference](#cuda-architecture-reference) below for a full list.

**Use Ninja** for faster parallel compilation (requires Ninja to be installed):

```
-G Ninja
```


### Full recommended CUDA build commands

putting everything together, here are two recomended Example cmake setups for both compiler chices:

#### Example A: High Performance Build (Clang-CL + RTX 3060)
```
cmake -B build ^
    -G Ninja ^
    -DCMAKE_C_COMPILER=clang-cl.exe ^
    -DCMAKE_CXX_COMPILER=clang-cl.exe ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DGGML_NATIVE=ON ^
    -DGGML_CUDA=ON ^
    -DCMAKE_CUDA_ARCHITECTURES="86-real" ^

cmake --build build --config Release
```
#### Example B: Simple Build (MSVC + RTX 3060)
```
cmake -B build ^
    -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DGGML_AVX2=ON ^
    -DGGML_CUDA=ON ^
    -DCMAKE_CUDA_ARCHITECTURES="86-real" ^

cmake --build build --config Release
```

---

### PATH handling, DLLs and Runtime dependencies: 

After building, the executables in `build\bin` require several runtime DLLs. The recommended
approach is to ensure the relevant directories are on `PATH` — do not copy DLLs into the build
directory, as this complicates maintenance and breaks on clean rebuilds.

**CUDA DLLs** (`cublas64_12.dll`, `cublasLt64_12.dll`, `cudart64_12.dll`):  
  These are located in `%CUDA_PATH%\bin`, which the CUDA Toolkit installer adds to `PATH`
  automatically. No further action is needed.

**OpenMP runtime** (`libomp140.x86_64.dll`):  
  When building with the Clang/LLVM toolchain (using the `clang-cl` compiler), this DLL is located in the LLVM `bin` directory
  (e.g. `%VS_DIR%\VC\Tools\Llvm\x64\bin`). Add this directory to your user or system `PATH`.
  Do not source this file from `C:\Windows\System32`, that copy may be incompatible with
LLVM-compiled binaries.

**Hint:** Always use forward slashes (`/`) in CMake paths (e.g., `-B build`).  
  This document avoids using absolute or relative paths as much as possible. Cmake settings inside ik_llama should handle everything correct. But if necessary, please note:
  If you must use backslashes, escape them (`\\`). CMake on Windows parses `/` correctly and avoids escape-character issues.

---

### Windows — ARM64 (WoA)

```bat
cmake --preset arm64-windows-llvm-release -DGGML_OPENMP=OFF
cmake --build build-arm64-windows-llvm-release
```

MSVC does not support the inline ARM assembly used by the accelerated Q4_0_4_8 CPU kernels;
LLVM/clang-cl is therefore the preferred compiler for ARM64 Windows targets.

---

## Debug builds

**Single-config generators** (default on Linux/macOS — `Unix Makefiles`, `Ninja`):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

**Multi-config generators** (Visual Studio, Xcode):

```bash
cmake -B build -G "Visual Studio 17 2022"
cmake --build build --config Debug
```

---

## General build tips

- **Clean builds:** When your cmake build comand stops working at some time after any kind of misbehaviour, if you encounter strange linker errors or outdated behavior, delete the `build` directory and re-run the CMake configuration steps.
- **Parallel compilation:** pass `-j <N>` to the build step, e.g.
  `cmake --build build --config Release -j 8`.
- **Faster incremental rebuilds:** install [ccache](https://ccache.dev/) — CMake picks it up
  automatically when it is on `PATH`.

---

## CPU build flags for AVX-512 (Zen4 / Sapphire Rapids+)

The IQK quantized GEMM kernels in `ggml/src/iqk/iqk_gemm_*.cpp` (the dominant hot path for
quantized prompt processing) are gated by the `HAVE_FANCY_SIMD` macro defined in
[`ggml/src/iqk/iqk_config.h`](../ggml/src/iqk/iqk_config.h):

```c
#if defined(__AVX512F__)  && defined(__AVX512VNNI__) && \
    defined(__AVX512VL__) && defined(__AVX512BW__)   && defined(__AVX512DQ__)
    #define HAVE_FANCY_SIMD
#endif
```

If these five macros are not defined at compile time, the AVX-512 quantized matmul path is
skipped and the build falls back to AVX2. There is **no warning at build time and no obvious
symptom at runtime** — performance is simply lower than what the hardware can deliver.

A few related gates worth knowing:

- `f16`/`f32` GEMM is gated only by `__AVX512F__`.
- Native `bf16` GEMM and use of a `bf16` KV cache in flash attention is gated by
  `__AVX512BF16__`.
- A separate `HAVE_VNNI256` path (`iqk_config.h:52-54`) is gated by `__AVXVNNI__` *or*
  (`__AVX512VNNI__ && __AVX512VL__`). This gives a meaningful speedup on AVX2-only CPUs that
  have the VNNI extension (e.g. some Alder Lake / Raptor Lake parts), even without full
  AVX-512.

### Recommended: high-level CMake options

The standard `GGML_AVX512_*` options work on both MSVC and GCC/Clang and are the shortest
path to activating `HAVE_FANCY_SIMD`:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=ON \
    -DGGML_AVX512=ON \
    -DGGML_AVX512_VBMI=ON \
    -DGGML_AVX512_VNNI=ON \
    -DGGML_AVX512_BF16=ON
cmake --build build --config Release
```

Mechanics:

- On **MSVC**, `GGML_AVX512=ON` adds `/arch:AVX512` (which defines `__AVX512F__`,
  `__AVX512VL__`, `__AVX512BW__`, `__AVX512DQ__`, `__AVX512CD__`), and the
  `GGML_AVX512_VNNI=ON` / `_VBMI=ON` / `_BF16=ON` options add the corresponding
  `__AVX512VNNI__` / `__AVX512VBMI__` / `__AVX512BF16__` definitions.
  See [`ggml/src/CMakeLists.txt`](../ggml/src/CMakeLists.txt).
- On **GCC / Clang**, `GGML_NATIVE=ON` resolves to `-march=native` (Zen4 → `znver4`;
  Sapphire Rapids → `sapphirerapids`), and the `GGML_AVX512_*=ON` options add explicit
  `-mavx512vnni` / `-mavx512vbmi` / `-mavx512bf16` flags as a belt-and-braces measure.

**Verification** — confirm the quantized path compiled in:

```bash
objdump -d build/bin/llama-cli | grep -c vpdpbusd
# A non-trivial count (hundreds+) means VNNI is compiled in.
# Zero means the IQK kernels fell back to AVX2.
```

You can also check the runtime banner: a successful AVX-512 build prints
`HAVE_FANCY_SIMD is defined` and `system_info: AVX512_VNNI = 1`.

### Fallback: explicit `GGML_ARCH_FLAGS`

If the options above do not produce `HAVE_FANCY_SIMD is defined` on your toolchain (older
MSVC, exotic compilers, or cross-compiles where `-march=native` does not propagate the
required macros), the defines can be supplied explicitly via `GGML_ARCH_FLAGS`, which is
forwarded verbatim to the compiler:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DGGML_ARCH_FLAGS="-D__AVX512F__ -D__AVX512VNNI__ -D__AVX512VL__ -D__AVX512BW__ -D__AVX512DQ__ -D__AVX512BF16__"
cmake --build build --config Release
```

For AVX2 CPUs with VNNI but without AVX-512:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DGGML_ARCH_FLAGS="-D__AVXVNNI__"
cmake --build build --config Release
```

### Note on Zen4 throughput

On Zen4 the AVX-512 implementation is 256-bit double-pumped: each `_mm512_*` op issues two
micro-ops with a throughput of roughly one AVX-512 op per two cycles. The wider register width
and reduced loop overhead still produce measurable gains over AVX2 on prompt processing for
IQK kernels.

---
## Verifying your Build

After successfully building with cmake, you will find the executables in subdirectory `build/bin`, `build/bin/release` or something like that, depending on yourenvironment and the exact cmake commandline you used.
### Simple execution Test
To ensure your build was successful, run the following command from your build directory:

```bash
llama-cli --help
```

If this prints the help menu without errors, your binary is functional.  

### Quick Inference Test
Use a small model (e.g., Gemma4-E2B) and run a minimal test:
```bash
llama-cli -m <path-to-model> -p "Hello!" -n 20
```
This will prompt your model to answer with 20 tokens only.  
If you see text output, your setup is working correctly.

---

## CUDA architecture reference

Use `-DCMAKE_CUDA_ARCHITECTURES="<cc>-real"` to compile only for your GPU's compute capability.
Compiling for all architectures (the default when this flag is omitted) significantly increases
build time and binary size.

| Compute Capability | Architecture | Example GPUs |
|---|---|---|
| 7.5 | Turing | GeForce RTX 2060/2070/2080, GTX 1650/1660 series |
| 8.0 | Ampere | NVIDIA A100, A30, A10 (data centre) |
| 8.6 | Ampere | GeForce RTX 3060/3070/3080/3090, RTX A2000–A6000 |
| 8.7 | Ampere | Jetson Orin (embedded) |
| 8.9 | Ada Lovelace | GeForce RTX 4060/4070/4080/4090, RTX 6000 Ada |
| 9.0 | Hopper | NVIDIA H100, H200 (data centre) |
| 10.0 | Blackwell | GeForce RTX 5080/5090, GB200 (data centre) |

Use `real` (e.g. `86-real`) to embed only PTX/SASS for that architecture; omit `real` to also
embed PTX for forward compatibility. For a system with mixed GPU generations, a comma-separated
list is accepted: `-DCMAKE_CUDA_ARCHITECTURES="75-real;86-real"`.

---

## CUDA compilation options reference

The following CMake options are available for fine-tuning CUDA performance:

| Option | Values | Default | Description |
|---|---|---|---|
| `GGML_CUDA_FORCE_DMMV` | Boolean | `false` | Force dequantization + matrix-vector multiplication kernels instead of quantized MMVQ. By default the choice is made based on compute capability. Does not affect k-quants. |
| `GGML_CUDA_DMMV_X` | Integer ≥ 32 | `32` | Values in x direction per DMMV kernel iteration. Increasing improves performance on fast GPUs. Power of 2 strongly recommended. |
| `GGML_CUDA_MMV_Y` | Integer ≥ 1 | `1` | Block size in y direction for mul-mat-vec kernels. Power of 2 recommended. |
| `GGML_CUDA_FORCE_MMQ` | Boolean | `false` | Force custom quantized matrix multiplication kernels instead of FP16 cuBLAS even when no int8 tensor core implementation is available. Reduces VRAM at the cost of large-batch throughput. |
| `GGML_CUDA_FORCE_CUBLAS` | Boolean | `false` | Force FP16 cuBLAS instead of custom quantized matmul kernels. |
| `GGML_CUDA_F16` | Boolean | `false` | Use half-precision arithmetic in DMMV and q4_1/q5_1 matmul kernels. Can improve performance on recent GPUs. |
| `GGML_CUDA_KQUANTS_ITER` | `1` or `2` | `2` | Values per iteration per thread for Q2_K and Q6_K quantisation. Set to `1` for slower GPUs. |
| `GGML_CUDA_USE_GRAPHS` | Boolean | `false` | Use CUDA graphs to reduce kernel launch overhead. Recommended on modern drivers. |
| `GGML_CUDA_PEER_MAX_BATCH_SIZE` | Integer ≥ 1 | `128` | Maximum batch size for which peer access is enabled between multiple GPUs. Requires Linux or NVLink. |
| `GGML_CUDA_FA_ALL_QUANTS` | Boolean | `false` | Compile support for all KV cache quantisation type combinations in the FlashAttention CUDA kernels. Increases compile time significantly. |
## Debug Builds

- **Linux/macOS/Ninja:** `cmake -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build`
- **Visual Studio:** `cmake -B build -G "Visual Studio 17 2022" && cmake --build build --config Debug`

---

## Other Platforms

- **FreeBSD:** Use CMake. `pkg install cmake ninja git llvm`, then standard CMake commands.
- **Android:** See [android.md](https://chat.qwen.ai/c/android.md).
- **Experimental Backends:** Refer to upstream [llama.cpp docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).  
  These backends are **not supported** in `ik_llama.cpp`. They may serve as a starting point for contributors who are willing to actively maintain and fix them for this fork.