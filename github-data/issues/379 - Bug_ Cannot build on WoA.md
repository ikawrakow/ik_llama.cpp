### 🐛 [#379](https://github.com/ikawrakow/ik_llama.cpp/issues/379) - Bug: Cannot build on WoA

| **Author** | `jdluzen` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2025-05-04 |
| **Updated** | 2025-05-05 |

---

#### Description

### What happened?

I am unable to build on Windows arm64, works out of the box on x64. The binaries do not work on arm64 using the translation layer either, my guess is some AVX instructions that are missing, but that's not related to this issue.
`cmake -B build` works.
`cmake --build build --config Release` fails with a number of errors:
`iqk_mul_mat.cpp(10643,42): error C2440: 'initializing': cannot convert from 'initializ
er list' to 'const uint32x4_t'`
`iqk_mul_mat.cpp(17283,81): error C1075: '{': no matching token found`
`C:\Program Files\Microsoft Visual Studio\2022\Preview\VC\Tools\MSVC\14.44.34918\include\ammintrin.h(35,1): error C1189:
 #error:  This header is specific to X86, X64, ARM64, and ARM64EC targets`

`cmake --preset arm64-windows-llvm-release -D GGML_OPENMP=OFF` fails with link errors for the standard Windows .libs like kernel32, etc.:
```
    lld-link: error: could not open 'kernel32.lib': no such file or directory
    lld-link: error: could not open 'user32.lib': no such file or directory
    lld-link: error: could not open 'gdi32.lib': no such file or directory
    lld-link: error: could not open 'winspool.lib': no such file or directory
    lld-link: error: could not open 'shell32.lib': no such file or directory
    lld-link: error: could not open 'ole32.lib': no such file or directory
    lld-link: error: could not open 'oleaut32.lib': no such file or directory
    lld-link: error: could not open 'uuid.lib': no such file or directory
    lld-link: error: could not open 'comdlg32.lib': no such file or directory
    lld-link: error: could not open 'advapi32.lib': no such file or directory
    clang: error: linker command failed with exit code 1 (use -v to see invocation)
    ninja: build stopped: subcommand failed.
```
I can see with `procmon` that the linker is not looking in the proper directory, mine is: `C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\um\arm64` but adding that directory using `target_link_directories` to the `CMakeLists.txt` or the `%PATH%` did not have any effect. 

### Name and Version

Tip of main f7c9a0f036951fecab32e056df954ebc54f8688f.

### What operating system are you seeing the problem on?

Windows

### Relevant log output

```shell

```

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-05-05** at **05:08:38**:<br>

The `ik_llama.cpp` build is less automated than mainline. I think you are the first to try building on Windows for ARM. You may need to manually specify the compiler options to make it work like this
```
cmake -B build -DGGML_ARCH_FLAGS="put the necessary flags here" etc.
```
To get rid of the `cannot convert from 'initializer list' to 'const uint32x4_t` and similar errors, one needs `-flax-vector-conversions` with GCC/clang. Don't know what is the corresponding MSVC compiler option. If MSVC does not automatically set the flags necessary to enable `ARM_NEON` SIMD instructions, you may need to set those manually as well. 

Concerning the `--preset arm64-windows-llvm-release`: this is something provided by `cmake`, so not sure why it doesn't work correctly in your case.