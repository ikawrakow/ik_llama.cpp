### 🔀 [#563](https://github.com/ikawrakow/ik_llama.cpp/pull/563) - Merge vulkan code from mainline up to commit of 6/28/2025

| **Author** | `firecoperana` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-29 |
| **Updated** | 2025-07-02 |

---

#### Description

* Vulkan Optimizations and Fixes (#8959)

* Optimize Vulkan REPEAT performance

.....................................................................................

vulkan: lock accesses of pinned_memory vector (#14333)

vulkan: handle noncontig in the final case of ggml_vk_get_cpy_pipeline (#14378)

Fix cuda build error




- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [ ] Low
  - [ ] Medium
  - [x] High

---

#### 💬 Conversation

👤 **firecoperana** commented the **2025-06-29** at **19:21:51**:<br>

Test Qwen 2.5 7B Q4_K_S and it runs fine, but for deepseek model, I was getting "GGGGGGG" output with -mla 1 -amb 512. Probably related to deepseek related optimization.

---

👤 **ubergarm** commented the **2025-06-29** at **19:51:08**:<br>

For deepseek often one wants to compile with `-DGGML_CUDA_IQK_FORCE_BF16=1` to avoid overflowing fp16 accumulator which manifests as gibberish, nans, or `GGG` typically I believe.

I just tried to compile but got an error, might be because I just updated my rig and now seem to have `gcc version 15.1.1 20250425 (GCC)`... I'll fuss with it a bit but put it here in the meantime.

Details inside:
<details>

<summary>👈 build command and logs</summary>

```bash
# attempt to build clean despite it seems to still be using cmake cache? hah...
$ rm -rf ./build
$ cmake -B build -DGGML_VULKAN=ON -DGGML_CUDA=OFF -DGGML_RPC=OFF -DGGML_BLAS=OFF GGML_CCACHE=OFF
$ cmake --build build --config Release -j $(nproc)

CMake Warning:
  Ignoring extra path from command line:

   "GGML_CCACHE=OFF"


-- The C compiler identification is GNU 15.1.1
-- The CXX compiler identification is GNU 15.1.1
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Found Git: /usr/bin/git (found version "2.50.0")
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Success
-- Found Threads: TRUE
-- Found OpenMP_C: -fopenmp (found version "4.5")
-- Found OpenMP_CXX: -fopenmp (found version "4.5")
-- Found OpenMP: TRUE (found version "4.5")
-- OpenMP found
-- Using optimized iqk matrix multiplications
-- Enabling IQK Flash Attention kernels
-- Using llamafile
-- Found Vulkan: /lib/libvulkan.so (found version "1.4.313") found components: glslc glslangValidator
-- Vulkan found
-- ccache found, compilation results will be cached. Disable with GGML_CCACHE=OFF.
-- CMAKE_SYSTEM_PROCESSOR: x86_64
-- x86 detected
-- ARCH_FLAGS = -march=native
-- Configuring done (0.5s)
-- Generating done (0.0s)
-- Build files have been written to: /mnt/astrodata/llm/ik_llama.cpp/build
[  0%] Generating build details from Git
[  0%] Building CXX object ggml/src/vulkan-shaders/CMakeFiles/vulkan-shaders-gen.dir/vulkan-shaders-gen.cpp.o
[  1%] Building C object examples/gguf-hash/CMakeFiles/sha256.dir/deps/sha256/sha256.c.o
[  3%] Building C object examples/gguf-hash/CMakeFiles/xxhash.dir/deps/xxhash/xxhash.c.o
[  3%] Building C object examples/gguf-hash/CMakeFiles/sha1.dir/deps/sha1/sha1.c.o
-- Found Git: /usr/bin/git (found version "2.50.0")
In function ‘SHA1Update’,
    inlined from ‘SHA1Final’ at /mnt/astrodata/llm/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c:265:5:
/mnt/astrodata/llm/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c:219:13: warning: ‘SHA1Transform’ reading 64 bytes from a region of size 0 [-Wstringop-overread]
  219 |             SHA1Transform(context->state, &data[i]);
      |             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/mnt/astrodata/llm/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c:219:13: note: referencing argument 2 of type ‘const unsigned char[64]’
/mnt/astrodata/llm/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c: In function ‘SHA1Final’:
/mnt/astrodata/llm/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c:54:6: note: in a call to function ‘SHA1Transform’
   54 | void SHA1Transform(
      |      ^~~~~~~~~~~~~
In function ‘SHA1Update’,
    inlined from ‘SHA1Final’ at /mnt/astrodata/llm/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c:269:9:
/mnt/astrodata/llm/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c:219:13: warning: ‘SHA1Transform’ reading 64 bytes from a region of size 0 [-Wstringop-overread]
  219 |             SHA1Transform(context->state, &data[i]);
      |             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/mnt/astrodata/llm/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c:219:13: note: referencing argument 2 of type ‘const unsigned char[64]’
/mnt/astrodata/llm/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c: In function ‘SHA1Final’:
/mnt/astrodata/llm/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c:54:6: note: in a call to function ‘SHA1Transform’
   54 | void SHA1Transform(
      |      ^~~~~~~~~~~~~
[  3%] Built target sha256
[  3%] Built target sha1
[  3%] Built target xxhash
[  3%] Generating build details from Git
-- Found Git: /usr/bin/git (found version "2.50.0")
[  4%] Building CXX object common/CMakeFiles/build_info.dir/build-info.cpp.o
[  5%] Linking CXX executable ../../../bin/vulkan-shaders-gen
[  5%] Built target build_info
[  5%] Built target vulkan-shaders-gen
[  6%] Generate vulkan shaders
ggml_vulkan: Generating and compiling shaders to SPIR-V
[  6%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml.c.o
[  7%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml-quants.c.o
[  8%] Building CXX object ggml/src/CMakeFiles/ggml.dir/ggml-vulkan.cpp.o
[  8%] Building CXX object ggml/src/CMakeFiles/ggml.dir/ggml-vulkan-shaders.cpp.o
[  9%] Building CXX object ggml/src/CMakeFiles/ggml.dir/llamafile/sgemm.cpp.o
[  9%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_gemm_kquants.cpp.o
[ 10%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml-alloc.c.o
[ 10%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_mul_mat.cpp.o
[ 11%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_flash_attn.cpp.o
[ 11%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/fa/iqk_fa_576_512.cpp.o
[ 11%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_gemm_iquants.cpp.o
[ 11%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/fa/iqk_fa_256_256.cpp.o
[ 12%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/fa/iqk_fa_192_128.cpp.o
[ 12%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml-backend.c.o
[ 14%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_gemm_ktquants.cpp.o
[ 14%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/fa/iqk_fa_128_128.cpp.o
[ 15%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/fa/iqk_fa_64_64.cpp.o
[ 16%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_gemm_legacy_quants.cpp.o
[ 16%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/fa/iqk_fa_96_96.cpp.o
[ 17%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_gemm_floats.cpp.o
[ 17%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_gemm_1bit.cpp.o
[ 18%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_gemm_iqk_quants.cpp.o
[ 18%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_quantize.cpp.o
[ 19%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml-aarch64.c.o
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/ggml.c: In function ‘ggml_compute_forward’:
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/ggml.c:19814:5: warning: enumeration value ‘GGML_OP_SIN’ not handled in switch [-Wswitch]
19814 |     switch (tensor->op) {
      |     ^~~~~~
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/ggml.c:19814:5: warning: enumeration value ‘GGML_OP_COS’ not handled in switch [-Wswitch]
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/ggml.c:19814:5: warning: enumeration value ‘GGML_OP_COUNT_EQUAL’ not handled in switch [-Wswitch]
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/ggml.c:19814:5: warning: enumeration value ‘GGML_OP_CONV_2D_DW’ not handled in switch [-Wswitch]
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/ggml.c:19814:5: warning: enumeration value ‘GGML_OP_RWKV_WKV6’ not handled in switch [-Wswitch]
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/ggml.c:19814:5: warning: enumeration value ‘GGML_OP_OPT_STEP_ADAMW’ not handled in switch [-Wswitch]
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/ggml.c: In function ‘ggml_compute_backward’:
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/ggml.c:20395:5: warning: enumeration value ‘GGML_OP_SIN’ not handled in switch [-Wswitch]
20395 |     switch (tensor->op) {
      |     ^~~~~~
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/ggml.c:20395:5: warning: enumeration value ‘GGML_OP_COS’ not handled in switch [-Wswitch]
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/ggml.c:20395:5: warning: enumeration value ‘GGML_OP_COUNT_EQUAL’ not handled in switch [-Wswitch]
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/ggml.c:20395:5: warning: enumeration value ‘GGML_OP_CONV_2D_DW’ not handled in switch [-Wswitch]
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/ggml.c:20395:5: warning: enumeration value ‘GGML_OP_RWKV_WKV6’ not handled in switch [-Wswitch]
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/ggml.c:20395:5: warning: enumeration value ‘GGML_OP_OPT_STEP_ADAMW’ not handled in switch [-Wswitch]
In file included from /usr/include/vulkan/vulkan_hpp_macros.hpp:35,
                 from /usr/include/vulkan/vulkan.hpp:11,
                 from /mnt/astrodata/llm/ik_llama.cpp/ggml/src/ggml-vulkan.cpp:8:
/usr/include/c++/15.1.1/ciso646:46:4: warning: #warning "<ciso646> is deprecated in C++17, use <version> to detect implementation-specific macros" [-Wcpp]
   46 | #  warning "<ciso646> is deprecated in C++17, use <version> to detect implementation-specific macros"
      |    ^~~~~~~
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/ggml-vulkan.cpp: In function ‘void ggml_vk_print_gpu_info(size_t)’:
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/ggml-vulkan.cpp:3541:18: warning: unused variable ‘subgroup_size’ [-Wunused-variable]
 3541 |     const size_t subgroup_size = (default_subgroup_size != 0) ? default_subgroup_size : subgroup_props.subgroupSize;
      |                  ^~~~~~~~~~~~~
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/ggml-vulkan.cpp:3542:16: warning: unused variable ‘uma’ [-Wunused-variable]
 3542 |     const bool uma = props2.properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu;
      |                ^~~
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/ggml-vulkan.cpp: In function ‘void ggml_vk_instance_init()’:
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/ggml-vulkan.cpp:3644:12: warning: unused variable ‘num_available_devices’ [-Wunused-variable]
 3644 |     size_t num_available_devices = vk_instance.instance.enumeratePhysicalDevices().size();
      |            ^~~~~~~~~~~~~~~~~~~~~
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/ggml-backend.c:269:16: warning: no previous prototype for ‘ggml_backend_tensor_memset’ [-Wmissing-prototypes]
  269 | GGML_CALL void ggml_backend_tensor_memset(struct ggml_tensor* tensor, uint8_t value, size_t offset, size_t size) {
      |                ^~~~~~~~~~~~~~~~~~~~~~~~~~
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/ggml-backend.c: In function ‘ggml_backend_multi_buffer_context_interface’:
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/ggml-backend.c:1022:34: error: initialization of ‘_Bool (*)(struct ggml_backend_buffer *, const struct ggml_tensor *, struct ggml_tensor *)’ from incompatible pointer type ‘void (*)(struct ggml_backend_buffer *, uint8_t)’ {aka ‘void (*)(struct ggml_backend_buffer *, unsigned char)’} [-Wincompatible-pointer-types]
 1022 |         /* .clear           = */ ggml_backend_multi_buffer_clear,
      |                                  ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/ggml-backend.c:1022:34: note: (near initialization for ‘multi_backend_buffer_i.cpy_tensor’)
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/ggml-backend.c:1006:23: note: ‘ggml_backend_multi_buffer_clear’ declared here
 1006 | GGML_CALL static void ggml_backend_multi_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
      |                       ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/ggml-backend.c:1024:5: warning: missing initializer for field ‘reset’ of ‘struct ggml_backend_buffer_i’ [-Wmissing-field-initializers]
 1024 |     };
      |     ^
In file included from /mnt/astrodata/llm/ik_llama.cpp/ggml/src/ggml-backend.c:1:
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/ggml-backend-impl.h:50:34: note: ‘reset’ declared here
   50 |         void         (*GGML_CALL reset)      (ggml_backend_buffer_t buffer); // reset any internal state due to tensor initialization, such as tensor extras
      |                                  ^~~~~
make[2]: *** [ggml/src/CMakeFiles/ggml.dir/build.make:222: ggml/src/CMakeFiles/ggml.dir/ggml-backend.c.o] Error 1
make[2]: *** Waiting for unfinished jobs....
make[1]: *** [CMakeFiles/Makefile2:2044: ggml/src/CMakeFiles/ggml.dir/all] Error 2
make: *** [Makefile:146: all] Error 2
```

</details>

---

👤 **ikawrakow** submitted a review the **2025-06-30** at **07:12:08**: 🔄 `CHANGES_REQUESTED`<br>

Please no new ops, new enum values, and no refactoring of the CPU backend. I think the Vulkan back-end can be updated to the latest without using the new back-end formalism in mainline.

---

👤 **ubergarm** commented the **2025-07-01** at **02:59:51**:<br>

@firecoperana 

Heya thanks again for digging into this! I have two different rigs on which I'm testing. It does now build on the AMD RX 7900 XTX Ubuntu 24.04 box now!

So good news I was able to compile and run `firecoperana/Merge_mainline_vulkan@495103bd` with vulkan backend! However, only seemed to run without `-fa`. If I try to use `-fa` it segfaults after its mostly loaded and right before llama-server would start listening for inputs.

Seems like something is still off as the speeds are off from mainline. Could be I'm using the AMDVLK driver as installed from `apt-get install libvulkan-dev` `1.4.313.0~rc1-1lunarg24.04-1` or that I'm compiling it wrong? Details in the fold:
<details>

<summary>👈 sweep-bench comparisons Qwen3-14B-Q4_0 dense no FA</summary>

![sweep-bench-pr-vs-mainline-vulkan](https://github.com/user-attachments/assets/57863083-8144-457c-81cc-ff9b9b395fed)


```bash
# checkout Merge_mainline_vulkan
$ git rev-parse --short HEAD
495103bd

# build
cmake -B build -DGGML_HIP=OFF -DGGML_HIPBLAS=OFF -DGGML_VULKAN=ON -DGGML_RPC=OFF -DGGML_CCACHE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j $(nproc)

# test
model=/home/w/projects/models/ubergarm/Qwen3-14B-GGUF/Qwen3-14B-Q4_0.gguf
sudo ./build/bin/llama-sweep-bench \
  --model "$model" \
  -ctk f16 -ctv f16 \
  -c 16896 \
  -ngl 99 \
  --warmup-batch \
  --threads 1
```

## ik_llama.cpp firecoperana/Merge_mainline_vulkan@495103bd FA=0
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    1.363 |   375.67 |    3.786 |    33.81 |
|   512 |    128 |    512 |    1.365 |   375.16 |    3.817 |    33.53 |
|   512 |    128 |   1024 |    1.414 |   362.06 |    3.844 |    33.30 |
|   512 |    128 |   1536 |    1.444 |   354.69 |    3.971 |    32.23 |
|   512 |    128 |   2048 |    1.429 |   358.21 |    3.965 |    32.28 |
|   512 |    128 |   2560 |    1.447 |   353.93 |    4.036 |    31.71 |
|   512 |    128 |   3072 |    1.462 |   350.17 |    4.099 |    31.23 |
|   512 |    128 |   3584 |    1.492 |   343.12 |    4.137 |    30.94 |
|   512 |    128 |   4096 |    1.499 |   341.62 |    4.233 |    30.24 |
|   512 |    128 |   4608 |    1.518 |   337.27 |    4.311 |    29.69 |
|   512 |    128 |   5120 |    1.525 |   335.71 |    4.355 |    29.39 |
|   512 |    128 |   5632 |    1.567 |   326.74 |    4.440 |    28.83 |
|   512 |    128 |   6144 |    1.556 |   329.11 |    4.508 |    28.39 |
|   512 |    128 |   6656 |    1.579 |   324.18 |    4.534 |    28.23 |
|   512 |    128 |   7168 |    1.596 |   320.79 |    4.600 |    27.83 |
|   512 |    128 |   7680 |    1.623 |   315.45 |    4.685 |    27.32 |
|   512 |    128 |   8192 |    1.640 |   312.19 |    4.775 |    26.80 |

## llama.cpp@27208bf6 FA=0
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    0.323 |  1585.78 |    1.822 |    70.27 |
|   512 |    128 |    512 |    0.334 |  1533.43 |    1.859 |    68.86 |
|   512 |    128 |   1024 |    0.369 |  1386.13 |    1.907 |    67.11 |
|   512 |    128 |   1536 |    0.382 |  1338.94 |    1.956 |    65.43 |
|   512 |    128 |   2048 |    0.374 |  1369.21 |    1.995 |    64.15 |
|   512 |    128 |   2560 |    0.391 |  1308.08 |    2.081 |    61.50 |
|   512 |    128 |   3072 |    0.396 |  1293.44 |    2.148 |    59.58 |
|   512 |    128 |   3584 |    0.422 |  1214.46 |    2.202 |    58.12 |
|   512 |    128 |   4096 |    0.422 |  1214.09 |    2.278 |    56.20 |
|   512 |    128 |   4608 |    0.435 |  1176.88 |    2.344 |    54.61 |
|   512 |    128 |   5120 |    0.441 |  1159.87 |    2.407 |    53.17 |
|   512 |    128 |   5632 |    0.482 |  1061.18 |    2.472 |    51.77 |
|   512 |    128 |   6144 |    0.465 |  1100.88 |    2.549 |    50.21 |
|   512 |    128 |   6656 |    0.483 |  1060.17 |    2.602 |    49.20 |
|   512 |    128 |   7168 |    0.494 |  1037.17 |    2.661 |    48.10 |
|   512 |    128 |   7680 |    0.523 |   979.25 |    2.724 |    46.99 |
|   512 |    128 |   8192 |    0.538 |   951.01 |    2.820 |    45.39 |

</details>

On my local rig with a CUDA and ARCH linux installing `extra/vulkan-utility-libraries 1.4.313.0-1 (vulkan-devel)` was having a compiling issue still complaining about RPC during linking. It might be because that super new gcc 15.1.1 though given I just updated everything...

```bash
$ cmake -B build -DGGML_VULKAN=ON -DGGML_CUDA=OFF -DGGML_RPC=OFF -DGGML_BLAS=OFF -DGGML_CCACHE=ON -DCMAKE_BUILD_TYPE=Debug
$ cmake --build build --config Debug -j $(nproc)

[ 24%] Building CXX object src/CMakeFiles/llama.dir/llama.cpp.o
[ 24%] Building CXX object src/CMakeFiles/llama.dir/unicode.cpp.o
[ 25%] Linking CXX executable ../../bin/llama-gguf
/mnt/astrodata/llm/ik_llama.cpp/src/unicode.cpp: In function ‘std::wstring unicode_wstring_from_utf8(const std::string&)’:
/mnt/astrodata/llm/ik_llama.cpp/src/unicode.cpp:232:10: warning: ‘template<class _Codecvt, class _Elem, class _Wide_alloc, class _Byte_alloc> class std::__cxx11::wstring_convert’ is deprecated [-Wdeprecated-declarations]
  232 |     std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
      |          ^~~~~~~~~~~~~~~
In file included from /usr/include/c++/15.1.1/locale:47,
                 from /usr/include/c++/15.1.1/regex:43,
                 from /mnt/astrodata/llm/ik_llama.cpp/src/unicode.cpp:12:
/usr/include/c++/15.1.1/bits/locale_conv.h:262:33: note: declared here
  262 |     class _GLIBCXX17_DEPRECATED wstring_convert
      |                                 ^~~~~~~~~~~~~~~
[ 25%] Linking CXX executable ../../bin/llama-gguf-hash
[ 26%] Linking CXX shared library libllama.so
/usr/bin/ld: ../../ggml/src/libggml.so: undefined reference to `ggml_backend_rpc_init'
collect2: error: ld returned 1 exit status
make[2]: *** [examples/gguf/CMakeFiles/llama-gguf.dir/build.make:102: bin/llama-gguf] Error 1
make[1]: *** [CMakeFiles/Makefile2:3314: examples/gguf/CMakeFiles/llama-gguf.dir/all] Error 2
make[1]: *** Waiting for unfinished jobs....
/usr/bin/ld: ../../ggml/src/libggml.so: undefined reference to `ggml_backend_rpc_init'
collect2: error: ld returned 1 exit status
make[2]: *** [examples/gguf-hash/CMakeFiles/llama-gguf-hash.dir/build.make:108: bin/llama-gguf-hash] Error 1
make[1]: *** [CMakeFiles/Makefile2:3151: examples/gguf-hash/CMakeFiles/llama-gguf-hash.dir/all] Error 2
[ 26%] Built target llama
make: *** [Makefile:146: all] Error 2
```

However, if I enable the RPC backend with `-DGGML_RPC=ON` it compiles now! Though starting up it throws some errors and isn't working yet
```bash
model=/mnt/astrodata/llm/models/ubergarm/Qwen3-14B-GGUF/Qwen3-14B-Q4_0.gguf

./build/bin/llama-sweep-bench \
  --model "$model" \
  -c 16896 \
  -ngl 99 \
  --warmup-batch \
  --threads 1

llm_load_tensors: ggml ctx size =    0.40 MiB
llm_load_tensors: offloading 40 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 41/41 layers to GPU
llm_load_tensors:    Vulkan0 buffer size =  7697.69 MiB
llm_load_tensors:        CPU buffer size =   417.30 MiB
.........................................................................................
llama_new_context_with_model: n_ctx      = 16896
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 0
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:    Vulkan0 KV buffer size =  2640.00 MiB
llama_new_context_with_model: KV self size  = 2640.00 MiB, K (f16): 1320.00 MiB, V (f16): 1320.00 MiB
llama_new_context_with_model: Vulkan_Host  output buffer size =     0.58 MiB
ggml_backend_sched_backend_from_buffer: warning: no backend supports op NONE with a weight with buffer type Vulkan0 used in tensor blk.0.attn_norm.weight, the weight will need to be copied
ggml_backend_sched_backend_from_buffer: warning: no backend supports op NONE with a weight with buffer type Vulkan0 used in tensor blk.0.attn_q_norm.weight, the weight will need to be copied
ggml_backend_sched_backend_from_buffer: warning: no backend supports op NONE with a weight with buffer type Vulkan0 used in tensor blk.0.attn_k_norm.weight, the weight will need to be copied
ggml_backend_sched_backend_from_buffer: warning: no backend supports op NONE with a weight with buffer type Vulkan0 used in tensor blk.0.ffn_norm.weight, the weight will need to be copied
ggml_backend_sched_backend_from_buffer: warning: no backend supports op NONE with a weight with buffer type Vulkan0 used in tensor blk.1.attn_norm.weight, the weight will need to be copied
```

Lemme know if there is a certain version of the vulkan backend that might work better or happy to try more iterations! Thanks!

---

👤 **firecoperana** commented the **2025-07-01** at **15:00:17**:<br>

I noticed something odd too and suspect it's related to vulkan shader. When I run llama server in visual studio, I can match the performance of the mainline, but if I run in command line, I was only getting 1/3 to 1/2 of the speed for token generation. If you have time, you can do some troubleshooting, as I'm not familiar with vulkan at all.

"warning: no backend supports op NONE with a weight with buffer type Vulkan0 used in tensor blk.0.attn_norm.weight" happens because vulkan does not support fused rms norm. It only shows in debug version.

---

👤 **ikawrakow** commented the **2025-07-01** at **16:38:42**:<br>

Tested on my RTX-4080. If I remove the fused ops (`GGML_OP_FUSED_RMS_NORM` and `GGML_OP_FUSED_MUL_UNARY`) and don't use flash attention, I get this for LlaMA-3.1-8B

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  1024 |    256 |      0 |    2.074 |   493.73 |    2.602 |    98.37 |
|  1024 |    256 |   1024 |    1.074 |   953.71 |    3.198 |    80.05 |
|  1024 |    256 |   2048 |    0.968 |  1058.33 |    3.069 |    83.41 |
|  1024 |    256 |   3072 |    0.907 |  1128.89 |    3.187 |    80.32 |
|  1024 |    256 |   4096 |    0.941 |  1088.54 |    3.368 |    76.00 |
|  1024 |    256 |   5120 |    0.962 |  1064.06 |    3.531 |    72.51 |
|  1024 |    256 |   6144 |    0.993 |  1030.96 |    3.742 |    68.42 |
|  1024 |    256 |   7168 |    1.037 |   987.64 |    3.963 |    64.60 |
|  1024 |    256 |   8192 |    1.098 |   932.90 |    4.223 |    60.62 |
|  1024 |    256 |   9216 |    1.156 |   885.58 |    4.474 |    57.22 |
|  1024 |    256 |  10240 |    1.216 |   842.27 |    4.711 |    54.34 |
|  1024 |    256 |  11264 |    1.271 |   805.53 |    4.949 |    51.73 |
|  1024 |    256 |  12288 |    1.323 |   774.28 |    5.201 |    49.22 |
|  1024 |    256 |  13312 |    1.381 |   741.70 |    5.457 |    46.92 |
|  1024 |    256 |  14336 |    1.440 |   711.14 |    5.709 |    44.84 |
|  1024 |    256 |  15360 |    1.469 |   696.92 |    5.962 |    42.94 |

Flash attention seems to be running on the CPU, so performance drops further with that. TG is on par with mainline for short context, but PP is ~3X lower.

---

👤 **ikawrakow** commented the **2025-07-01** at **16:48:33**:<br>

If I change the `LOG_DEBUG` to `LOG_INFO` in `ggml_vk_print_gpu_info`, I see this line:
```
ggml_vulkan: 0 = NVIDIA GeForce RTX 4080 (NVIDIA) | uma: 0 | fp16: 1 | warp size: 32 | shared memory: 49152 | int dot: 0 | matrix cores: none
```

On mainline I see this:
```
ggml_vulkan: 0 = NVIDIA GeForce RTX 4080 (NVIDIA) | uma: 0 | fp16: 1 | warp size: 32 | shared memory: 49152 | int dot: 1 | matrix cores: KHR_coopmat
```
So, for some reason int dot products and cooperative matrix are not enabled. I guess, this may explain the lower performance.

---

👤 **ikawrakow** submitted a review the **2025-07-01** at **18:07:18**: 💬 `COMMENTED`

---

👤 **firecoperana** submitted a review the **2025-07-02** at **01:10:01**: 💬 `COMMENTED`

---

👤 **firecoperana** commented during a code review the **2025-07-02** at **01:10:01** on `ggml/src/ggml-vulkan.cpp`:<br>

Removed.

---

👤 **ubergarm** commented the **2025-07-02** at **04:42:36**:<br>

> The new commit should remove the need to add these in cmake command. Also disable the fused ops for now.

Thanks I was having trouble getting it setup. First the amazing news, check this out on the AMD RX 7900 XTX it is up to snuff in early testing:

![sweep-bench-llama-cpp-vulkan-amd](https://github.com/user-attachments/assets/6877f569-5539-4d99-89a6-097755a9fbe7)

Very nice! I want to try some more models tomorrow but this is getting exciting!

I also got it to build and detect things properly on my local ARCH linux NVIDIA 3090TI FE rig, however when it starts up it throws an error:
```bash
ggml_vulkan: 0 = NVIDIA GeForce RTX 3090 Ti (NVIDIA) | uma: 0 | fp16: 1 | warp size: 32 | shared memory: 49152 | int dot: 1 | matrix cores: NV_coopmat2
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/ggml-vulkan.cpp:2031: GGML_ASSERT((GGML_KQ_MASK_PAD % rows_cols[0]) == 0) failed
```

Amazing progress in a short time!

---

👤 **ikawrakow** submitted a review the **2025-07-02** at **06:49:33**: ✅ `APPROVED`