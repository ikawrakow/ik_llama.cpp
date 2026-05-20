# Build ik_llama.cpp

## Supported backends

> **Important:** `ik_llama.cpp` focuses on two fully functional and performant compute backends:
>
> - **CPU** — AVX2 required; AVX-512 (AMD Zen4 / Intel Sapphire Rapids+) strongly recommended
> - **CUDA** — Turing architecture (RTX 20xx / GTX 16xx) or newer required
>
> Other backends (Metal, ROCm/hipBLAS, Vulkan, SYCL, OpenBLAS, MUSA) are inherited from upstream
> `llama.cpp` and are **not actively maintained or tested** in this fork. Their build instructions
> have been removed to avoid misleading users. If you need one of these, refer to the upstream
> [llama.cpp build documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).

> **Note on `make`:** The Makefile-based build is **obsolete** and has been removed from this
> documentation. CMake is the only supported build system.

---

## Prerequisites

| Requirement | Details |
|---|---|
| **Git** | [git-scm.com](https://git-scm.com/download/) or system package manager |
| **CMake 3.21+** | [cmake.org](https://cmake.org/download/) or system package manager |
| **C++17 compiler** | GCC 10+, Clang 13+, MSVC 2022, or clang-cl |
| **CUDA Toolkit 12.x** | Required for GPU builds only — [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) |

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
hardware (AMD Zen4, Intel Sapphire Rapids+) additional flags are needed to unlock the full IQK
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
  - `Clang Compiler for Windows` *(optional, but recommended for AVX-512 builds)*

#### 2. CUDA Toolkit 12.x

Download from the [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive).
During installation, choose *Custom Installation* and uncheck *Driver components* if building
inside a VM without a physical NVIDIA GPU.

#### 3. CMake and Ninja

CMake is included with Visual Studio. If you prefer a standalone installation, or need a newer
version:

```
winget install Kitware.CMake Ninja-build.Ninja
```

Or download from [cmake.org](https://cmake.org/download/).

#### 4. Use the Developer Command Prompt

Always execute build commands from inside the
**Developer Command Prompt for VS 2022** or **Developer PowerShell for VS 2022**, accessible
via the Windows Start Menu under *Visual Studio 2022*.

These terminals automatically configure all required compiler paths, environment variables, and
SDK settings. Running `cmake` from a plain `cmd.exe` or PowerShell will likely fail with
"command not found" or wrong-compiler errors due to missing environment initialisation.

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

```
-DGGML_NATIVE=ON
```

This adds `-march=native` (GCC/Clang) or the equivalent MSVC flags, tuning the binary to the
exact CPU it is compiled on. On AVX-512 hardware (Zen4, Sapphire Rapids+) you additionally need
the flags described in the [AVX-512 section](#cpu-build-flags-for-avx-512-zen4--sapphire-rapids).

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

**Further performance and feature options:**

```
-DGGML_CUDA_USE_GRAPHS=ON
-DGGML_SCHED_MAX_COPIES=1
-DGGML_OPENMP=ON
-DLLAMA_CURL=OFF
```

**Example: a full recommended CUDA build command for an RTX 3060 (Ampere / cc86):**

```bat
cmake -B build ^
    -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DGGML_NATIVE=ON ^
    -DGGML_CUDA=ON ^
    -DCMAKE_CUDA_ARCHITECTURES="86-real" ^
    -DGGML_CUDA_USE_GRAPHS=ON ^
    -DGGML_OPENMP=ON ^
    -DLLAMA_CURL=OFF
    
cmake --build build --config Release
```

---

### Runtime dependencies: DLLs and PATH handling

After building, the executables in `build\bin` require several runtime DLLs. The recommended
approach is to ensure the relevant directories are on `PATH` — do not copy DLLs into the build
directory, as this complicates maintenance and breaks on clean rebuilds.

**CUDA DLLs** (`cublas64_12.dll`, `cublasLt64_12.dll`, `cudart64_12.dll`):
These are located in `%CUDA_PATH%\bin`, which the CUDA Toolkit installer adds to `PATH`
automatically. No further action is needed.

**OpenMP runtime** (`libomp140.x86_64.dll`):
When building with the Clang/LLVM toolchain, this DLL is located in the LLVM `bin` directory
(e.g. `%VS_DIR%\VC\Tools\Llvm\x64\bin`). Add this directory to your user or system `PATH`.
Do not source this file from `C:\Windows\System32` — that copy may be incompatible with
LLVM-compiled binaries.

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