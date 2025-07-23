### 🐛 [#308](https://github.com/ikawrakow/ik_llama.cpp/issues/308) - Bug: Compiling for arm64, error: cannot convert ‘const uint32x4_t’ to ‘uint8x16_t’ and similar errors

| **Author** | `smpurkis` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-04-03 |
| **Updated** | 2025-04-04 |

---

#### Description

### What happened?

I'm compiling on free tier Oracle Ampere A1 server, arm64 architecture.
Running `make`, I'm getting a very long set of compiler errors, almost all are some form of can't convert uint to int, e.g.
- ggml/src/iqk/iqk_mul_mat.cpp:10303:36: error: cannot convert ‘const uint32x4_t’ to ‘uint8x16_t’
- ggml/src/iqk/iqk_mul_mat.cpp:10305:54: error: cannot convert ‘uint8x16_t’ to ‘int8x16_t’
- ggml/src/iqk/iqk_mul_mat.cpp:10334:40: error: cannot convert ‘int8x16_t’ to ‘__Uint8x16_t’ in assignment
- ggml/src/iqk/iqk_mul_mat.cpp:10954:41: error: cannot convert ‘const __Int16x8_t’ to ‘const int8x16_t&’
- ggml/src/iqk/iqk_mul_mat.cpp:11271:37: error: cannot convert ‘int8x16_t’ to ‘uint8x16_t’
... and many more in the full log below
[logs.txt](https://github.com/user-attachments/files/19582579/logs.txt)

I would expect it to compile with no errors.

### Name and Version

Git commit id
```
❯ git rev-parse HEAD
07dbc1aa06d761634419759431ebb215baf698bb
```

### What operating system are you seeing the problem on?

Linux
```
❯ uname -a
Linux instance-20240214-1712 6.8.0-1018-oracle #19~22.04.1-Ubuntu SMP Mon Dec  9 23:49:53 UTC 2024 aarch64 aarch64 aarch64 GNU/Linux
```

### Relevant log output

[logs.txt](https://github.com/user-attachments/files/19582579/logs.txt)

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-04-03** at **08:29:58**:<br>

I'm not sure I want to fix those (I perceive them as useless noise from a compiler trying too hard to protect me). Can you try adding
```
-flax-vector-conversions
```
to the compilation options?

---

👤 **smpurkis** commented the **2025-04-03** at **08:41:16**:<br>

Still errors with int and float conversions, e.g. 
```
ggml/src/iqk/iqk_mul_mat.cpp:12791:44: error: cannot convert ‘int32x4_t’ to ‘float32x4_t’
```
[logs.txt](https://github.com/user-attachments/files/19582935/logs.txt)

I also tried adding `-fpermissive`, errors with the same

---

👤 **smpurkis** commented the **2025-04-03** at **08:43:58**:<br>

Not sure if it makes any difference my `gcc` and `g++` versions are both `12.3.0`

---

👤 **ikawrakow** commented the **2025-04-03** at **08:45:17**:<br>

I'll try to fix those. Give me a few minutes.

---

👤 **ikawrakow** commented the **2025-04-03** at **09:04:19**:<br>

Does #309 work?

---

👤 **smpurkis** commented the **2025-04-03** at **11:04:07**:<br>

Unfortunately not, it fails on only a few things now though
```
> make CXXFLAGS="-fpermissive -flax-vector-conversions" CFLAGS="-fpermissive -flax-vector-conversions" &> logs
I ccache not found. Consider installing it for faster compilation.
make: jetson_release: No such file or directory
I llama.cpp build info: 
I UNAME_S:   Linux
I UNAME_P:   aarch64
I UNAME_M:   aarch64
I CFLAGS:    -Iggml/include -Iggml/src -Iinclude -Isrc -Icommon -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_OPENMP -DGGML_USE_IQK_MULMAT -DGGML_USE_LLAMAFILE  -std=c11   -fPIC -O3 -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes -Werror=implicit-int -Werror=implicit-function-declaration -pthread -mcpu=native -fopenmp -DGGML_USE_IQK_MULMAT -Wdouble-promotion -fpermissive -flax-vector-conversions
I CXXFLAGS:  -std=c++17 -fPIC -O3 -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wmissing-declarations -Wmissing-noreturn -pthread -mcpu=native -fopenmp -fpermissive -flax-vector-conversions  -Wno-array-bounds -Wno-format-truncation -Wextra-semi -Iggml/include -Iggml/src -Iinclude -Isrc -Icommon -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_OPENMP -DGGML_USE_IQK_MULMAT -DGGML_USE_LLAMAFILE 
I NVCCFLAGS: -std=c++17 -O3 
I LDFLAGS:    
I CC:        cc (Ubuntu 12.3.0-1ubuntu1~22.04) 12.3.0
I CXX:       c++ (Ubuntu 12.3.0-1ubuntu1~22.04) 12.3.0

c++ -std=c++17 -fPIC -O3 -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wmissing-declarations -Wmissing-noreturn -pthread -mcpu=native -fopenmp -fpermissive -flax-vector-conversions  -Wno-array-bounds -Wno-format-truncation -Wextra-semi -Iggml/include -Iggml/src -Iinclude -Isrc -Icommon -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_OPENMP -DGGML_USE_IQK_MULMAT -DGGML_USE_LLAMAFILE  -c ggml/src/iqk/iqk_mul_mat.cpp -o ggml/src/iqk/iqk_mul_mat.o
ggml/src/iqk/iqk_mul_mat.cpp: In member function ‘float {anonymous}::FlashMS<q_step, k_step>::load_apply_mask_and_scale(int, float32x4_t*, const char*)’:
ggml/src/iqk/iqk_mul_mat.cpp:15832:67: error: cannot convert ‘__Float32x4_t’ to ‘uint32x4_t’
15832 |                                                         vbicq_u32(vinf, vm1)));
      |                                                                   ^~~~
      |                                                                   |
      |                                                                   __Float32x4_t
In file included from ggml/src/ggml-impl.h:158,
                 from ggml/src/iqk/iqk_mul_mat.cpp:18:
/usr/lib/gcc/aarch64-linux-gnu/12/include/arm_neon.h:1470:23: note:   initializing argument 1 of ‘uint32x4_t vbicq_u32(uint32x4_t, uint32x4_t)’
 1470 | vbicq_u32 (uint32x4_t __a, uint32x4_t __b)
      |            ~~~~~~~~~~~^~~
ggml/src/iqk/iqk_mul_mat.cpp:15834:67: error: cannot convert ‘__Float32x4_t’ to ‘uint32x4_t’
15834 |                                                         vbicq_u32(vinf, vm2)));
      |                                                                   ^~~~
      |                                                                   |
      |                                                                   __Float32x4_t
/usr/lib/gcc/aarch64-linux-gnu/12/include/arm_neon.h:1470:23: note:   initializing argument 1 of ‘uint32x4_t vbicq_u32(uint32x4_t, uint32x4_t)’
 1470 | vbicq_u32 (uint32x4_t __a, uint32x4_t __b)
      |            ~~~~~~~~~~~^~~
ggml/src/iqk/iqk_mul_mat.cpp: At global scope:
ggml/src/iqk/iqk_mul_mat.cpp:13625:24: warning: ‘always_inline’ function might not be inlinable [-Wattributes]
13625 | IQK_ALWAYS_INLINE void prepare_q4_k_quants(const uint8x16_t& m4, const uint8x16x4_t& bits, int8x16_t * qx) {
      |                        ^~~~~~~~~~~~~~~~~~~
ggml/src/iqk/iqk_mul_mat.cpp:12370:24: warning: ‘always_inline’ function might not be inlinable [-Wattributes]
12370 | IQK_ALWAYS_INLINE void prepare_iq4_nl_quants_r8(const int8x16_t& values, const uint8x16_t& m4, const uint8x16x2_t& bits, int8x16_t * qx) {
      |                        ^~~~~~~~~~~~~~~~~~~~~~~~
ggml/src/iqk/iqk_mul_mat.cpp:12359:24: warning: ‘always_inline’ function might not be inlinable [-Wattributes]
12359 | IQK_ALWAYS_INLINE void prepare_iq4_nl_quants(const int8x16_t& values, const uint8x16_t& m4, const uint8x16x4_t& bits, int8x16_t * qx) {
      |                        ^~~~~~~~~~~~~~~~~~~~~
ggml/src/iqk/iqk_mul_mat.cpp:12350:29: warning: ‘always_inline’ function might not be inlinable [-Wattributes]
12350 | IQK_ALWAYS_INLINE int32x4_t interleaved_dotq(const int8x16_t * qx, const int8x16_t& y) {
      |                             ^~~~~~~~~~~~~~~~
ggml/src/iqk/iqk_mul_mat.cpp:12337:31: warning: ‘always_inline’ function might not be inlinable [-Wattributes]
12337 | IQK_ALWAYS_INLINE int32x4x2_t interleaved_dotq_b16(const int8x16_t * qx, const int8x16x2_t& y) {
      |                               ^~~~~~~~~~~~~~~~~~~~
ggml/src/iqk/iqk_mul_mat.cpp:12324:29: warning: ‘always_inline’ function might not be inlinable [-Wattributes]
12324 | IQK_ALWAYS_INLINE int32x4_t interleaved_dotq(const int8x16_t * qx, const int8x16x2_t& y) {
      |                             ^~~~~~~~~~~~~~~~
make: *** [Makefile:1083: ggml/src/iqk/iqk_mul_mat.o] Error 1
```

---

👤 **ikawrakow** commented the **2025-04-03** at **11:12:17**:<br>

Thanks for testing. I have missed this one. The new version should compile now. The warnings are harmless.

---

👤 **smpurkis** commented the **2025-04-03** at **12:07:50**:<br>

Not sure if this is an issue with just my setup
I'm getting 
```
/usr/bin/ld: ggml/src/ggml.o: in function `ggml_compute_forward_flash_attn_ext_f16':
ggml.c:(.text+0x83f0): undefined reference to `iqk_flash_attn_noalibi'
```
full log
```
❯ make CXXFLAGS="-fpermissive -flax-vector-conversions" CFLAGS="-flax-vector-conversions"
I ccache not found. Consider installing it for faster compilation.
make: jetson_release: No such file or directory
I llama.cpp build info: 
I UNAME_S:   Linux
I UNAME_P:   aarch64
I UNAME_M:   aarch64
I CFLAGS:    -Iggml/include -Iggml/src -Iinclude -Isrc -Icommon -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_OPENMP -DGGML_USE_IQK_MULMAT -DGGML_USE_LLAMAFILE  -std=c11   -fPIC -O3 -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes -Werror=implicit-int -Werror=implicit-function-declaration -pthread -mcpu=native -fopenmp -DGGML_USE_IQK_MULMAT -Wdouble-promotion -flax-vector-conversions
I CXXFLAGS:  -std=c++17 -fPIC -O3 -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wmissing-declarations -Wmissing-noreturn -pthread -mcpu=native -fopenmp -fpermissive -flax-vector-conversions  -Wno-array-bounds -Wno-format-truncation -Wextra-semi -Iggml/include -Iggml/src -Iinclude -Isrc -Icommon -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_OPENMP -DGGML_USE_IQK_MULMAT -DGGML_USE_LLAMAFILE 
I NVCCFLAGS: -std=c++17 -O3 
I LDFLAGS:    
I CC:        cc (Ubuntu 12.3.0-1ubuntu1~22.04) 12.3.0
I CXX:       c++ (Ubuntu 12.3.0-1ubuntu1~22.04) 12.3.0

c++ -std=c++17 -fPIC -O3 -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wmissing-declarations -Wmissing-noreturn -pthread -mcpu=native -fopenmp -fpermissive -flax-vector-conversions  -Wno-array-bounds -Wno-format-truncation -Wextra-semi -Iggml/include -Iggml/src -Iinclude -Isrc -Icommon -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_OPENMP -DGGML_USE_IQK_MULMAT -DGGML_USE_LLAMAFILE  -c examples/baby-llama/baby-llama.cpp -o examples/baby-llama/baby-llama.o
c++ -std=c++17 -fPIC -O3 -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wmissing-declarations -Wmissing-noreturn -pthread -mcpu=native -fopenmp -fpermissive -flax-vector-conversions  -Wno-array-bounds -Wno-format-truncation -Wextra-semi -Iggml/include -Iggml/src -Iinclude -Isrc -Icommon -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_OPENMP -DGGML_USE_IQK_MULMAT -DGGML_USE_LLAMAFILE  ggml/src/iqk/iqk_quantize.o ggml/src/iqk/iqk_mul_mat.o ggml/src/llamafile/sgemm.o ggml/src/ggml.o ggml/src/ggml-alloc.o ggml/src/ggml-backend.o ggml/src/ggml-quants.o ggml/src/ggml-aarch64.o src/llama.o src/llama-vocab.o src/llama-grammar.o src/llama-sampling.o src/unicode.o src/unicode-data.o common/common.o common/console.o common/ngram-cache.o common/sampling.o common/train.o common/grammar-parser.o common/build-info.o common/json-schema-to-grammar.o examples/baby-llama/baby-llama.o -o llama-baby-llama  
/usr/bin/ld: ggml/src/ggml.o: in function `ggml_compute_forward_flash_attn_ext_f16':
ggml.c:(.text+0x83f0): undefined reference to `iqk_flash_attn_noalibi'
collect2: error: ld returned 1 exit status
make: *** [Makefile:1376: llama-baby-llama] Error 1
```

---

👤 **ikawrakow** commented the **2025-04-03** at **12:21:19**:<br>

Is `baby-llama` something that you have modified yourself? 
The link command lists all these object files, but normally it should just link against the `common` and `llama` libs:
```
set(TARGET llama-baby-llama)
add_executable(${TARGET} baby-llama.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)
```

Oh, you are using the Makefile? I think it only works with `cmake`. They have deprecated the Makefile also in mainline.

---

👤 **smpurkis** commented the **2025-04-03** at **12:26:37**:<br>

Coolio, will cmake a go
```
cmake -B build -DCMAKE_CXX_FLAGS="-fpermissive -flax-vector-conversions" -DCMAKE_C_FLAGS="-flax-vector-conversions" && cmake --build build --config Release -j 4
```
I have made no modifications to any files.

---

👤 **smpurkis** commented the **2025-04-03** at **12:29:40**:<br>

Hmm, getting other unresolved references with `cmake`
```
❯ cmake -B build -DCMAKE_CXX_FLAGS="-fpermissive -flax-vector-conversions" -DCMAKE_C_FLAGS="-flax-vector-conversions" && cmake --b
uild build --config Release 
-- OpenMP found
-- Using optimized iqk matrix multiplications
-- Using llamafile
-- Warning: ccache not found - consider installing it for faster compilation or disable this warning with GGML_CCACHE=OFF
-- CMAKE_SYSTEM_PROCESSOR: aarch64
-- ARM detected
-- Configuring done (0.2s)
-- Generating done (0.2s)
-- Build files have been written to: /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build
[  6%] Built target ggml
[  7%] Building CXX object src/CMakeFiles/llama.dir/llama.cpp.o
[  7%] Building CXX object src/CMakeFiles/llama.dir/unicode-data.cpp.o
[  8%] Linking CXX shared library libllama.so
[ 10%] Built target llama
[ 11%] Built target build_info
[ 11%] Building CXX object common/CMakeFiles/common.dir/common.cpp.o
[ 12%] Building CXX object common/CMakeFiles/common.dir/sampling.cpp.o
[ 12%] Building CXX object common/CMakeFiles/common.dir/console.cpp.o
[ 13%] Building CXX object common/CMakeFiles/common.dir/grammar-parser.cpp.o
[ 14%] Building CXX object common/CMakeFiles/common.dir/json-schema-to-grammar.cpp.o
[ 14%] Building CXX object common/CMakeFiles/common.dir/train.cpp.o
[ 15%] Building CXX object common/CMakeFiles/common.dir/ngram-cache.cpp.o
[ 15%] Linking CXX static library libcommon.a
[ 15%] Built target common
[ 16%] Building CXX object tests/CMakeFiles/test-tokenizer-0.dir/test-tokenizer-0.cpp.o
[ 17%] Linking CXX executable ../bin/test-tokenizer-0
/usr/bin/ld: ../ggml/src/libggml.so: undefined reference to `iqk_moe_fused_up_gate'
/usr/bin/ld: ../ggml/src/libggml.so: undefined reference to `iqk_mul_mat_4d'
/usr/bin/ld: ../ggml/src/libggml.so: undefined reference to `iqk_mul_mat'
/usr/bin/ld: ../ggml/src/libggml.so: undefined reference to `iqk_flash_attn_noalibi'
/usr/bin/ld: ../ggml/src/libggml.so: undefined reference to `iqk_mul_mat_moe'
collect2: error: ld returned 1 exit status
gmake[2]: *** [tests/CMakeFiles/test-tokenizer-0.dir/build.make:103: bin/test-tokenizer-0] Error 1
gmake[1]: *** [CMakeFiles/Makefile2:2123: tests/CMakeFiles/test-tokenizer-0.dir/all] Error 2
gmake: *** [Makefile:146: all] Error 2
```

---

👤 **ikawrakow** commented the **2025-04-03** at **12:42:54**:<br>

Can we take a look at the `compile_commands.json` in the `build` folder?

---

👤 **smpurkis** commented the **2025-04-03** at **12:44:33**:<br>

Sure here is it
```[
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/ggml/src",
  "command": "/usr/bin/cc -DGGML_BUILD -DGGML_SCHED_MAX_COPIES=4 -DGGML_SHARED -DGGML_USE_IQK_MULMAT -DGGML_USE_LLAMAFILE -DGGML_USE_OPENMP -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -Dggml_EXPORTS -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/. -flax-vector-conversions -O3 -DNDEBUG -std=gnu11 -fPIC -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes -Werror=implicit-int -Werror=implicit-function-declaration -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wdouble-promotion -fopenmp -o CMakeFiles/ggml.dir/ggml.c.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/ggml.c",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/ggml.c",
  "output": "ggml/src/CMakeFiles/ggml.dir/ggml.c.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/ggml/src",
  "command": "/usr/bin/cc -DGGML_BUILD -DGGML_SCHED_MAX_COPIES=4 -DGGML_SHARED -DGGML_USE_IQK_MULMAT -DGGML_USE_LLAMAFILE -DGGML_USE_OPENMP -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -Dggml_EXPORTS -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/. -flax-vector-conversions -O3 -DNDEBUG -std=gnu11 -fPIC -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes -Werror=implicit-int -Werror=implicit-function-declaration -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wdouble-promotion -fopenmp -o CMakeFiles/ggml.dir/ggml-alloc.c.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/ggml-alloc.c",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/ggml-alloc.c",
  "output": "ggml/src/CMakeFiles/ggml.dir/ggml-alloc.c.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/ggml/src",
  "command": "/usr/bin/cc -DGGML_BUILD -DGGML_SCHED_MAX_COPIES=4 -DGGML_SHARED -DGGML_USE_IQK_MULMAT -DGGML_USE_LLAMAFILE -DGGML_USE_OPENMP -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -Dggml_EXPORTS -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/. -flax-vector-conversions -O3 -DNDEBUG -std=gnu11 -fPIC -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes -Werror=implicit-int -Werror=implicit-function-declaration -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wdouble-promotion -fopenmp -o CMakeFiles/ggml.dir/ggml-backend.c.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/ggml-backend.c",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/ggml-backend.c",
  "output": "ggml/src/CMakeFiles/ggml.dir/ggml-backend.c.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/ggml/src",
  "command": "/usr/bin/cc -DGGML_BUILD -DGGML_SCHED_MAX_COPIES=4 -DGGML_SHARED -DGGML_USE_IQK_MULMAT -DGGML_USE_LLAMAFILE -DGGML_USE_OPENMP -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -Dggml_EXPORTS -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/. -flax-vector-conversions -O3 -DNDEBUG -std=gnu11 -fPIC -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes -Werror=implicit-int -Werror=implicit-function-declaration -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wdouble-promotion -fopenmp -o CMakeFiles/ggml.dir/ggml-quants.c.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/ggml-quants.c",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/ggml-quants.c",
  "output": "ggml/src/CMakeFiles/ggml.dir/ggml-quants.c.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/ggml/src",
  "command": "/usr/bin/c++ -DGGML_BUILD -DGGML_SCHED_MAX_COPIES=4 -DGGML_SHARED -DGGML_USE_IQK_MULMAT -DGGML_USE_LLAMAFILE -DGGML_USE_OPENMP -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -Dggml_EXPORTS -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/. -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -fPIC -Wmissing-declarations -Wmissing-noreturn -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wno-array-bounds -Wno-format-truncation -Wextra-semi -fopenmp -o CMakeFiles/ggml.dir/llamafile/sgemm.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/llamafile/sgemm.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/llamafile/sgemm.cpp",
  "output": "ggml/src/CMakeFiles/ggml.dir/llamafile/sgemm.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/ggml/src",
  "command": "/usr/bin/c++ -DGGML_BUILD -DGGML_SCHED_MAX_COPIES=4 -DGGML_SHARED -DGGML_USE_IQK_MULMAT -DGGML_USE_LLAMAFILE -DGGML_USE_OPENMP -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -Dggml_EXPORTS -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/. -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -fPIC -Wmissing-declarations -Wmissing-noreturn -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wno-array-bounds -Wno-format-truncation -Wextra-semi -fopenmp -o CMakeFiles/ggml.dir/iqk/iqk_mul_mat.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp",
  "output": "ggml/src/CMakeFiles/ggml.dir/iqk/iqk_mul_mat.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/ggml/src",
  "command": "/usr/bin/c++ -DGGML_BUILD -DGGML_SCHED_MAX_COPIES=4 -DGGML_SHARED -DGGML_USE_IQK_MULMAT -DGGML_USE_LLAMAFILE -DGGML_USE_OPENMP -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -Dggml_EXPORTS -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/. -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -fPIC -Wmissing-declarations -Wmissing-noreturn -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wno-array-bounds -Wno-format-truncation -Wextra-semi -fopenmp -o CMakeFiles/ggml.dir/iqk/iqk_flash_attn.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/iqk/iqk_flash_attn.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/iqk/iqk_flash_attn.cpp",
  "output": "ggml/src/CMakeFiles/ggml.dir/iqk/iqk_flash_attn.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/ggml/src",
  "command": "/usr/bin/c++ -DGGML_BUILD -DGGML_SCHED_MAX_COPIES=4 -DGGML_SHARED -DGGML_USE_IQK_MULMAT -DGGML_USE_LLAMAFILE -DGGML_USE_OPENMP -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -Dggml_EXPORTS -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/. -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -fPIC -Wmissing-declarations -Wmissing-noreturn -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wno-array-bounds -Wno-format-truncation -Wextra-semi -fopenmp -o CMakeFiles/ggml.dir/iqk/iqk_quantize.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/iqk/iqk_quantize.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/iqk/iqk_quantize.cpp",
  "output": "ggml/src/CMakeFiles/ggml.dir/iqk/iqk_quantize.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/ggml/src",
  "command": "/usr/bin/cc -DGGML_BUILD -DGGML_SCHED_MAX_COPIES=4 -DGGML_SHARED -DGGML_USE_IQK_MULMAT -DGGML_USE_LLAMAFILE -DGGML_USE_OPENMP -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -Dggml_EXPORTS -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/. -flax-vector-conversions -O3 -DNDEBUG -std=gnu11 -fPIC -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes -Werror=implicit-int -Werror=implicit-function-declaration -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wdouble-promotion -fopenmp -o CMakeFiles/ggml.dir/ggml-aarch64.c.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/ggml-aarch64.c",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/ggml-aarch64.c",
  "output": "ggml/src/CMakeFiles/ggml.dir/ggml-aarch64.c.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/src",
  "command": "/usr/bin/c++ -DLLAMA_BUILD -DLLAMA_SHARED -Dllama_EXPORTS -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../ggml/src -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -fPIC -o CMakeFiles/llama.dir/llama.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/llama.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/llama.cpp",
  "output": "src/CMakeFiles/llama.dir/llama.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/src",
  "command": "/usr/bin/c++ -DLLAMA_BUILD -DLLAMA_SHARED -Dllama_EXPORTS -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../ggml/src -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -fPIC -o CMakeFiles/llama.dir/llama-vocab.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/llama-vocab.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/llama-vocab.cpp",
  "output": "src/CMakeFiles/llama.dir/llama-vocab.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/src",
  "command": "/usr/bin/c++ -DLLAMA_BUILD -DLLAMA_SHARED -Dllama_EXPORTS -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../ggml/src -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -fPIC -o CMakeFiles/llama.dir/llama-grammar.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/llama-grammar.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/llama-grammar.cpp",
  "output": "src/CMakeFiles/llama.dir/llama-grammar.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/src",
  "command": "/usr/bin/c++ -DLLAMA_BUILD -DLLAMA_SHARED -Dllama_EXPORTS -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../ggml/src -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -fPIC -o CMakeFiles/llama.dir/llama-sampling.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/llama-sampling.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/llama-sampling.cpp",
  "output": "src/CMakeFiles/llama.dir/llama-sampling.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/src",
  "command": "/usr/bin/c++ -DLLAMA_BUILD -DLLAMA_SHARED -Dllama_EXPORTS -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../ggml/src -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -fPIC -o CMakeFiles/llama.dir/unicode.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/unicode.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/unicode.cpp",
  "output": "src/CMakeFiles/llama.dir/unicode.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/src",
  "command": "/usr/bin/c++ -DLLAMA_BUILD -DLLAMA_SHARED -Dllama_EXPORTS -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../ggml/src -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -fPIC -o CMakeFiles/llama.dir/unicode-data.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/unicode-data.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/unicode-data.cpp",
  "output": "src/CMakeFiles/llama.dir/unicode-data.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/common",
  "command": "/usr/bin/c++   -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -fPIC -o CMakeFiles/build_info.dir/build-info.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/build-info.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/build-info.cpp",
  "output": "common/CMakeFiles/build_info.dir/build-info.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/common",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -fPIC -o CMakeFiles/common.dir/common.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/common.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/common.cpp",
  "output": "common/CMakeFiles/common.dir/common.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/common",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -fPIC -o CMakeFiles/common.dir/sampling.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/sampling.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/sampling.cpp",
  "output": "common/CMakeFiles/common.dir/sampling.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/common",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -fPIC -o CMakeFiles/common.dir/console.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/console.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/console.cpp",
  "output": "common/CMakeFiles/common.dir/console.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/common",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -fPIC -o CMakeFiles/common.dir/grammar-parser.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/grammar-parser.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/grammar-parser.cpp",
  "output": "common/CMakeFiles/common.dir/grammar-parser.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/common",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -fPIC -o CMakeFiles/common.dir/json-schema-to-grammar.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/json-schema-to-grammar.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/json-schema-to-grammar.cpp",
  "output": "common/CMakeFiles/common.dir/json-schema-to-grammar.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/common",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -fPIC -o CMakeFiles/common.dir/train.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/train.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/train.cpp",
  "output": "common/CMakeFiles/common.dir/train.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/common",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -fPIC -o CMakeFiles/common.dir/ngram-cache.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/ngram-cache.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/ngram-cache.cpp",
  "output": "common/CMakeFiles/common.dir/ngram-cache.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/tests",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/test-tokenizer-0.dir/test-tokenizer-0.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/test-tokenizer-0.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/test-tokenizer-0.cpp",
  "output": "tests/CMakeFiles/test-tokenizer-0.dir/test-tokenizer-0.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/tests",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/test-tokenizer-1-bpe.dir/test-tokenizer-1-bpe.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/test-tokenizer-1-bpe.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/test-tokenizer-1-bpe.cpp",
  "output": "tests/CMakeFiles/test-tokenizer-1-bpe.dir/test-tokenizer-1-bpe.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/tests",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/test-tokenizer-1-spm.dir/test-tokenizer-1-spm.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/test-tokenizer-1-spm.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/test-tokenizer-1-spm.cpp",
  "output": "tests/CMakeFiles/test-tokenizer-1-spm.dir/test-tokenizer-1-spm.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/tests",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/test-quantize-fns.dir/test-quantize-fns.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/test-quantize-fns.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/test-quantize-fns.cpp",
  "output": "tests/CMakeFiles/test-quantize-fns.dir/test-quantize-fns.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/tests",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/test-quantize-fns.dir/get-model.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/get-model.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/get-model.cpp",
  "output": "tests/CMakeFiles/test-quantize-fns.dir/get-model.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/tests",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/test-quantize-perf.dir/test-quantize-perf.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/test-quantize-perf.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/test-quantize-perf.cpp",
  "output": "tests/CMakeFiles/test-quantize-perf.dir/test-quantize-perf.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/tests",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/test-quantize-perf.dir/get-model.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/get-model.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/get-model.cpp",
  "output": "tests/CMakeFiles/test-quantize-perf.dir/get-model.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/tests",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/test-sampling.dir/test-sampling.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/test-sampling.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/test-sampling.cpp",
  "output": "tests/CMakeFiles/test-sampling.dir/test-sampling.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/tests",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/test-sampling.dir/get-model.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/get-model.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/get-model.cpp",
  "output": "tests/CMakeFiles/test-sampling.dir/get-model.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/tests",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/test-chat-template.dir/test-chat-template.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/test-chat-template.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/test-chat-template.cpp",
  "output": "tests/CMakeFiles/test-chat-template.dir/test-chat-template.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/tests",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/test-chat-template.dir/get-model.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/get-model.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/get-model.cpp",
  "output": "tests/CMakeFiles/test-chat-template.dir/get-model.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/tests",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/test-grammar-parser.dir/test-grammar-parser.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/test-grammar-parser.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/test-grammar-parser.cpp",
  "output": "tests/CMakeFiles/test-grammar-parser.dir/test-grammar-parser.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/tests",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/test-grammar-parser.dir/get-model.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/get-model.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/get-model.cpp",
  "output": "tests/CMakeFiles/test-grammar-parser.dir/get-model.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/tests",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/test-llama-grammar.dir/test-llama-grammar.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/test-llama-grammar.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/test-llama-grammar.cpp",
  "output": "tests/CMakeFiles/test-llama-grammar.dir/test-llama-grammar.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/tests",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/test-llama-grammar.dir/get-model.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/get-model.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/get-model.cpp",
  "output": "tests/CMakeFiles/test-llama-grammar.dir/get-model.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/tests",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/test-grammar-integration.dir/test-grammar-integration.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/test-grammar-integration.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/test-grammar-integration.cpp",
  "output": "tests/CMakeFiles/test-grammar-integration.dir/test-grammar-integration.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/tests",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/test-grammar-integration.dir/get-model.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/get-model.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/get-model.cpp",
  "output": "tests/CMakeFiles/test-grammar-integration.dir/get-model.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/tests",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/test-grad0.dir/test-grad0.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/test-grad0.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/test-grad0.cpp",
  "output": "tests/CMakeFiles/test-grad0.dir/test-grad0.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/tests",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/test-grad0.dir/get-model.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/get-model.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/get-model.cpp",
  "output": "tests/CMakeFiles/test-grad0.dir/get-model.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/tests",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/test-backend-ops.dir/test-backend-ops.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/test-backend-ops.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/test-backend-ops.cpp",
  "output": "tests/CMakeFiles/test-backend-ops.dir/test-backend-ops.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/tests",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/test-backend-ops.dir/get-model.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/get-model.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/get-model.cpp",
  "output": "tests/CMakeFiles/test-backend-ops.dir/get-model.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/tests",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/test-rope.dir/test-rope.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/test-rope.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/test-rope.cpp",
  "output": "tests/CMakeFiles/test-rope.dir/test-rope.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/tests",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/test-rope.dir/get-model.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/get-model.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/get-model.cpp",
  "output": "tests/CMakeFiles/test-rope.dir/get-model.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/tests",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/test-model-load-cancel.dir/test-model-load-cancel.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/test-model-load-cancel.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/test-model-load-cancel.cpp",
  "output": "tests/CMakeFiles/test-model-load-cancel.dir/test-model-load-cancel.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/tests",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/test-model-load-cancel.dir/get-model.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/get-model.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/get-model.cpp",
  "output": "tests/CMakeFiles/test-model-load-cancel.dir/get-model.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/tests",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/test-autorelease.dir/test-autorelease.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/test-autorelease.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/test-autorelease.cpp",
  "output": "tests/CMakeFiles/test-autorelease.dir/test-autorelease.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/tests",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/test-autorelease.dir/get-model.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/get-model.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/get-model.cpp",
  "output": "tests/CMakeFiles/test-autorelease.dir/get-model.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/tests",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/../examples/server -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/test-json-schema-to-grammar.dir/test-json-schema-to-grammar.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/test-json-schema-to-grammar.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/test-json-schema-to-grammar.cpp",
  "output": "tests/CMakeFiles/test-json-schema-to-grammar.dir/test-json-schema-to-grammar.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/tests",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/../examples/server -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/test-json-schema-to-grammar.dir/get-model.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/get-model.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/get-model.cpp",
  "output": "tests/CMakeFiles/test-json-schema-to-grammar.dir/get-model.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/tests",
  "command": "/usr/bin/cc  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -flax-vector-conversions -O3 -DNDEBUG -o CMakeFiles/test-c.dir/test-c.c.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/test-c.c",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/tests/test-c.c",
  "output": "tests/CMakeFiles/test-c.dir/test-c.c.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/examples/cvector-generator",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/llama-cvector-generator.dir/cvector-generator.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/cvector-generator/cvector-generator.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/cvector-generator/cvector-generator.cpp",
  "output": "examples/cvector-generator/CMakeFiles/llama-cvector-generator.dir/cvector-generator.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/examples/baby-llama",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/llama-baby-llama.dir/baby-llama.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/baby-llama/baby-llama.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/baby-llama/baby-llama.cpp",
  "output": "examples/baby-llama/CMakeFiles/llama-baby-llama.dir/baby-llama.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/examples/batched-bench",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/llama-batched-bench.dir/batched-bench.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/batched-bench/batched-bench.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/batched-bench/batched-bench.cpp",
  "output": "examples/batched-bench/CMakeFiles/llama-batched-bench.dir/batched-bench.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/examples/batched",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/llama-batched.dir/batched.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/batched/batched.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/batched/batched.cpp",
  "output": "examples/batched/CMakeFiles/llama-batched.dir/batched.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/examples/benchmark",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/benchmark/../../common -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/llama-bench-matmult.dir/benchmark-matmult.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/benchmark/benchmark-matmult.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/benchmark/benchmark-matmult.cpp",
  "output": "examples/benchmark/CMakeFiles/llama-bench-matmult.dir/benchmark-matmult.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/examples/convert-llama2c-to-ggml",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/llama-convert-llama2c-to-ggml.dir/convert-llama2c-to-ggml.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/convert-llama2c-to-ggml/convert-llama2c-to-ggml.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/convert-llama2c-to-ggml/convert-llama2c-to-ggml.cpp",
  "output": "examples/convert-llama2c-to-ggml/CMakeFiles/llama-convert-llama2c-to-ggml.dir/convert-llama2c-to-ggml.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/examples/embedding",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/llama-embedding.dir/embedding.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/embedding/embedding.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/embedding/embedding.cpp",
  "output": "examples/embedding/CMakeFiles/llama-embedding.dir/embedding.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/examples/eval-callback",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/llama-eval-callback.dir/eval-callback.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/eval-callback/eval-callback.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/eval-callback/eval-callback.cpp",
  "output": "examples/eval-callback/CMakeFiles/llama-eval-callback.dir/eval-callback.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/examples/export-lora",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/llama-export-lora.dir/export-lora.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/export-lora/export-lora.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/export-lora/export-lora.cpp",
  "output": "examples/export-lora/CMakeFiles/llama-export-lora.dir/export-lora.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/examples/gbnf-validator",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/llama-gbnf-validator.dir/gbnf-validator.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/gbnf-validator/gbnf-validator.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/gbnf-validator/gbnf-validator.cpp",
  "output": "examples/gbnf-validator/CMakeFiles/llama-gbnf-validator.dir/gbnf-validator.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/examples/gguf-hash",
  "command": "/usr/bin/cc  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/gguf-hash/deps -flax-vector-conversions -O3 -DNDEBUG -o CMakeFiles/xxhash.dir/deps/xxhash/xxhash.c.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/gguf-hash/deps/xxhash/xxhash.c",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/gguf-hash/deps/xxhash/xxhash.c",
  "output": "examples/gguf-hash/CMakeFiles/xxhash.dir/deps/xxhash/xxhash.c.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/examples/gguf-hash",
  "command": "/usr/bin/cc  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/gguf-hash/deps -flax-vector-conversions -O3 -DNDEBUG -o CMakeFiles/sha1.dir/deps/sha1/sha1.c.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c",
  "output": "examples/gguf-hash/CMakeFiles/sha1.dir/deps/sha1/sha1.c.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/examples/gguf-hash",
  "command": "/usr/bin/cc  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/gguf-hash/deps -flax-vector-conversions -O3 -DNDEBUG -o CMakeFiles/sha256.dir/deps/sha256/sha256.c.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/gguf-hash/deps/sha256/sha256.c",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/gguf-hash/deps/sha256/sha256.c",
  "output": "examples/gguf-hash/CMakeFiles/sha256.dir/deps/sha256/sha256.c.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/examples/gguf-hash",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/gguf-hash/deps -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/llama-gguf-hash.dir/gguf-hash.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/gguf-hash/gguf-hash.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/gguf-hash/gguf-hash.cpp",
  "output": "examples/gguf-hash/CMakeFiles/llama-gguf-hash.dir/gguf-hash.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/examples/gguf-split",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/llama-gguf-split.dir/gguf-split.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/gguf-split/gguf-split.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/gguf-split/gguf-split.cpp",
  "output": "examples/gguf-split/CMakeFiles/llama-gguf-split.dir/gguf-split.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/examples/gguf",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/llama-gguf.dir/gguf.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/gguf/gguf.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/gguf/gguf.cpp",
  "output": "examples/gguf/CMakeFiles/llama-gguf.dir/gguf.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/examples/gritlm",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/llama-gritlm.dir/gritlm.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/gritlm/gritlm.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/gritlm/gritlm.cpp",
  "output": "examples/gritlm/CMakeFiles/llama-gritlm.dir/gritlm.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/examples/imatrix",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/llama-imatrix.dir/imatrix.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/imatrix/imatrix.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/imatrix/imatrix.cpp",
  "output": "examples/imatrix/CMakeFiles/llama-imatrix.dir/imatrix.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/examples/infill",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/llama-infill.dir/infill.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/infill/infill.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/infill/infill.cpp",
  "output": "examples/infill/CMakeFiles/llama-infill.dir/infill.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/examples/llama-bench",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/llama-bench.dir/llama-bench.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/llama-bench/llama-bench.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/llama-bench/llama-bench.cpp",
  "output": "examples/llama-bench/CMakeFiles/llama-bench.dir/llama-bench.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/examples/llava",
  "command": "/usr/bin/c++ -DLLAMA_BUILD -DLLAMA_SHARED -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/llava/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/llava/../.. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/llava/../../common -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -fPIC -Wno-cast-qual -o CMakeFiles/llava.dir/llava.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/llava/llava.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/llava/llava.cpp",
  "output": "examples/llava/CMakeFiles/llava.dir/llava.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/examples/llava",
  "command": "/usr/bin/c++ -DLLAMA_BUILD -DLLAMA_SHARED -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/llava/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/llava/../.. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/llava/../../common -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -fPIC -Wno-cast-qual -o CMakeFiles/llava.dir/clip.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/llava/clip.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/llava/clip.cpp",
  "output": "examples/llava/CMakeFiles/llava.dir/clip.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/examples/llava",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/llava/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/llava/../.. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/llava/../../common -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/llama-llava-cli.dir/llava-cli.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/llava/llava-cli.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/llava/llava-cli.cpp",
  "output": "examples/llava/CMakeFiles/llama-llava-cli.dir/llava-cli.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/examples/llava",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/llava/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/llava/../.. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/llava/../../common -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/llama-minicpmv-cli.dir/minicpmv-cli.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/llava/minicpmv-cli.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/llava/minicpmv-cli.cpp",
  "output": "examples/llava/CMakeFiles/llama-minicpmv-cli.dir/minicpmv-cli.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/examples/lookahead",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/llama-lookahead.dir/lookahead.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/lookahead/lookahead.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/lookahead/lookahead.cpp",
  "output": "examples/lookahead/CMakeFiles/llama-lookahead.dir/lookahead.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/examples/lookup",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/llama-lookup.dir/lookup.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/lookup/lookup.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/lookup/lookup.cpp",
  "output": "examples/lookup/CMakeFiles/llama-lookup.dir/lookup.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/examples/lookup",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/llama-lookup-create.dir/lookup-create.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/lookup/lookup-create.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/lookup/lookup-create.cpp",
  "output": "examples/lookup/CMakeFiles/llama-lookup-create.dir/lookup-create.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/examples/lookup",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/llama-lookup-merge.dir/lookup-merge.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/lookup/lookup-merge.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/lookup/lookup-merge.cpp",
  "output": "examples/lookup/CMakeFiles/llama-lookup-merge.dir/lookup-merge.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/examples/lookup",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/llama-lookup-stats.dir/lookup-stats.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/lookup/lookup-stats.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/lookup/lookup-stats.cpp",
  "output": "examples/lookup/CMakeFiles/llama-lookup-stats.dir/lookup-stats.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/examples/main",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/llama-cli.dir/main.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/main/main.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/main/main.cpp",
  "output": "examples/main/CMakeFiles/llama-cli.dir/main.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/examples/parallel",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/llama-parallel.dir/parallel.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/parallel/parallel.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/parallel/parallel.cpp",
  "output": "examples/parallel/CMakeFiles/llama-parallel.dir/parallel.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/examples/passkey",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/llama-passkey.dir/passkey.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/passkey/passkey.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/passkey/passkey.cpp",
  "output": "examples/passkey/CMakeFiles/llama-passkey.dir/passkey.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/examples/perplexity",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/llama-perplexity.dir/perplexity.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/perplexity/perplexity.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/perplexity/perplexity.cpp",
  "output": "examples/perplexity/CMakeFiles/llama-perplexity.dir/perplexity.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/examples/quantize-stats",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/quantize-stats/../../common -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/llama-quantize-stats.dir/quantize-stats.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/quantize-stats/quantize-stats.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/quantize-stats/quantize-stats.cpp",
  "output": "examples/quantize-stats/CMakeFiles/llama-quantize-stats.dir/quantize-stats.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/examples/quantize",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/quantize/../../common -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/llama-quantize.dir/quantize.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/quantize/quantize.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/quantize/quantize.cpp",
  "output": "examples/quantize/CMakeFiles/llama-quantize.dir/quantize.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/examples/retrieval",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/llama-retrieval.dir/retrieval.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/retrieval/retrieval.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/retrieval/retrieval.cpp",
  "output": "examples/retrieval/CMakeFiles/llama-retrieval.dir/retrieval.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/examples/server",
  "command": "/usr/bin/c++ -DSERVER_VERBOSE=1 -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/server -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/examples/server -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/llama-server.dir/server.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/server/server.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/server/server.cpp",
  "output": "examples/server/CMakeFiles/llama-server.dir/server.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/examples/save-load-state",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/llama-save-load-state.dir/save-load-state.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/save-load-state/save-load-state.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/save-load-state/save-load-state.cpp",
  "output": "examples/save-load-state/CMakeFiles/llama-save-load-state.dir/save-load-state.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/examples/simple",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/llama-simple.dir/simple.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/simple/simple.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/simple/simple.cpp",
  "output": "examples/simple/CMakeFiles/llama-simple.dir/simple.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/examples/speculative",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/llama-speculative.dir/speculative.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/speculative/speculative.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/speculative/speculative.cpp",
  "output": "examples/speculative/CMakeFiles/llama-speculative.dir/speculative.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/examples/sweep-bench",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/llama-sweep-bench.dir/sweep-bench.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/sweep-bench/sweep-bench.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/sweep-bench/sweep-bench.cpp",
  "output": "examples/sweep-bench/CMakeFiles/llama-sweep-bench.dir/sweep-bench.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/examples/tokenize",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/llama-tokenize.dir/tokenize.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/tokenize/tokenize.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/tokenize/tokenize.cpp",
  "output": "examples/tokenize/CMakeFiles/llama-tokenize.dir/tokenize.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/pocs/vdot",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/pocs -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/llama-vdot.dir/vdot.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/pocs/vdot/vdot.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/pocs/vdot/vdot.cpp",
  "output": "pocs/vdot/CMakeFiles/llama-vdot.dir/vdot.cpp.o"
},
{
  "directory": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build/pocs/vdot",
  "command": "/usr/bin/c++  -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/pocs -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/common/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/. -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/src/../include -I/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/../include -fpermissive -flax-vector-conversions -O3 -DNDEBUG -std=gnu++17 -o CMakeFiles/llama-q8dot.dir/q8dot.cpp.o -c /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/pocs/vdot/q8dot.cpp",
  "file": "/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/pocs/vdot/q8dot.cpp",
  "output": "pocs/vdot/CMakeFiles/llama-q8dot.dir/q8dot.cpp.o"
}
]```

---

👤 **ikawrakow** commented the **2025-04-03** at **12:55:13**:<br>

Are you cross-compiling? The above is missing the native flag, which should be ON by default unless cross-compiling. Can you try adding `-DGGML_NATIVE=1` to the `cmake` command?

Also not sure about OpenMP on this system (it is better to use it on `x86_64` Linux, but don't know about `ARM` Linux as it using OpenMP is worse on my M2-Max laptop).

---

👤 **smpurkis** commented the **2025-04-03** at **13:09:04**:<br>

I'm using whatever the default settings are.
Adding `-DGGML_NATIVE=1`, running the following unfortunately still errors
```
❯ cmake -B build -DGGML_NATIVE=1 -DCMAKE_CXX_FLAGS="-fpermissive -flax-vector-conversions" -DCMAKE_C_FLAGS="-flax-vector-conversions" && cmake --build build --config Release -j 4
-- The C compiler identification is GNU 12.3.0
-- The CXX compiler identification is GNU 12.3.0
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
-- Found Git: /usr/bin/git (found version "2.34.1")
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Success
-- Found Threads: TRUE
-- Found OpenMP_C: -fopenmp (found version "4.5")
-- Found OpenMP_CXX: -fopenmp (found version "4.5")
-- Found OpenMP: TRUE (found version "4.5")
-- OpenMP found
-- Using optimized iqk matrix multiplications
-- Using llamafile
-- Warning: ccache not found - consider installing it for faster compilation or disable this warning with GGML_CCACHE=OFF
-- CMAKE_SYSTEM_PROCESSOR: aarch64
-- ARM detected
-- Performing Test COMPILER_SUPPORTS_FP16_FORMAT_I3E
-- Performing Test COMPILER_SUPPORTS_FP16_FORMAT_I3E - Failed
-- Configuring done (1.9s)
-- Generating done (0.2s)
-- Build files have been written to: /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/build
[  1%] Building C object examples/gguf-hash/CMakeFiles/sha256.dir/deps/sha256/sha256.c.o
[  2%] Building C object examples/gguf-hash/CMakeFiles/xxhash.dir/deps/xxhash/xxhash.c.o
[  3%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml.c.o
[  4%] Building CXX object common/CMakeFiles/build_info.dir/build-info.cpp.o
[  4%] Built target build_info
[  4%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml-alloc.c.o
[  4%] Built target sha256
[  5%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml-backend.c.o
[  6%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml-quants.c.o
[  6%] Building CXX object ggml/src/CMakeFiles/ggml.dir/llamafile/sgemm.cpp.o
[  6%] Built target xxhash
[  6%] Building C object examples/gguf-hash/CMakeFiles/sha1.dir/deps/sha1/sha1.c.o
In function ‘SHA1Update’,
    inlined from ‘SHA1Final’ at /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c:265:5:
/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c:219:13: warning: ‘SHA1Transform’ reading 64 bytes from a region of size 0 [-Wstringop-overread]
  219 |             SHA1Transform(context->state, &data[i]);
      |             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c:219:13: note: referencing argument 2 of type ‘const unsigned char[64]’
/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c: In function ‘SHA1Final’:
/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c:54:6: note: in a call to function ‘SHA1Transform’
   54 | void SHA1Transform(
      |      ^~~~~~~~~~~~~
In function ‘SHA1Update’,
    inlined from ‘SHA1Final’ at /home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c:269:9:
/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c:219:13: warning: ‘SHA1Transform’ reading 64 bytes from a region of size 0 [-Wstringop-overread]
  219 |             SHA1Transform(context->state, &data[i]);
      |             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c:219:13: note: referencing argument 2 of type ‘const unsigned char[64]’
/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c: In function ‘SHA1Final’:
/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c:54:6: note: in a call to function ‘SHA1Transform’
   54 | void SHA1Transform(
      |      ^~~~~~~~~~~~~
[  6%] Built target sha1
[  7%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_mul_mat.cpp.o
/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:17501:6: warning: no previous declaration for ‘bool iqk_mul_mat(int, long int, long int, long int, int, const void*, long int, int, const void*, long int, float*, long int, int, int)’ [-Wmissing-declarations]
17501 | bool iqk_mul_mat(int, long, long, long, int, const void *, long, int, const void *, long, float *, long, int, int) {
      |      ^~~~~~~~~~~
/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:17505:6: warning: no previous declaration for ‘bool iqk_mul_mat_4d(long int, long int, long int, long int, long int, long int, long int, long int, long int, long int, long int, long int, long int, int, const void*, long int, int, const void*, long int, float*, long int, int, int)’ [-Wmissing-declarations]
17505 | bool iqk_mul_mat_4d(long /*Nx*/, long /*Ny*/, long /*ne00*/,
      |      ^~~~~~~~~~~~~~
/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:17514:6: warning: no previous declaration for ‘bool iqk_mul_mat_moe(long int, long int, long int, int, int, const void*, long int, int, const void*, long int, float*, long int, long int, const void*, int, int)’ [-Wmissing-declarations]
17514 | bool iqk_mul_mat_moe(long, long, long, int, int, const void *, long, int, const void *, long, float *, long, long,
      |      ^~~~~~~~~~~~~~~
/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:17519:6: warning: no previous declaration for ‘bool iqk_moe_fused_up_gate(long int, long int, long int, int, int, int, const void*, const void*, long int, int, const void*, long int, float*, long int, long int, const void*, int, int)’ [-Wmissing-declarations]
17519 | bool iqk_moe_fused_up_gate(long /*Nx*/, long /*Ny*/, long /*ne00*/, int /*ne11*/, int /*unary_op*/,
      |      ^~~~~~~~~~~~~~~~~~~~~
[  7%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_flash_attn.cpp.o
/home/ubuntu/projects/oobabooga_linux/ik_llama.cpp/ggml/src/iqk/iqk_flash_attn.cpp:189:6: warning: no previous declaration for ‘bool iqk_flash_attn_noalibi(int, int, float, int, int, long int, long int, int, int, long int, long int, int, int, long int, long int, int, int, long int, int, int, int, int, int, int, int, int, int, const void*, const void*, const void*, const void*, float, float, float*, void*, barrier_t, void*, int, int)’ [-Wmissing-declarations]
  189 | bool iqk_flash_attn_noalibi([[maybe_unused]] int type_q, [[maybe_unused]] int type_mask, [[maybe_unused]] float max_bias,
      |      ^~~~~~~~~~~~~~~~~~~~~~
[  8%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_quantize.cpp.o
[  8%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml-aarch64.c.o
[  9%] Linking CXX shared library libggml.so
[  9%] Built target ggml
[ 10%] Building CXX object examples/gguf-hash/CMakeFiles/llama-gguf-hash.dir/gguf-hash.cpp.o
[ 10%] Building CXX object src/CMakeFiles/llama.dir/llama-vocab.cpp.o
[ 11%] Building CXX object src/CMakeFiles/llama.dir/llama.cpp.o
[ 11%] Building CXX object examples/gguf/CMakeFiles/llama-gguf.dir/gguf.cpp.o
[ 12%] Linking CXX executable ../../bin/llama-gguf
[ 12%] Linking CXX executable ../../bin/llama-gguf-hash
/usr/bin/ld: ../../ggml/src/libggml.so: undefined reference to `iqk_moe_fused_up_gate'
/usr/bin/ld: ../../ggml/src/libggml.so: undefined reference to `iqk_mul_mat_4d'
/usr/bin/ld: ../../ggml/src/libggml.so: undefined reference to `iqk_mul_mat'
/usr/bin/ld: ../../ggml/src/libggml.so: undefined reference to `iqk_flash_attn_noalibi'
/usr/bin/ld: ../../ggml/src/libggml.so: undefined reference to `iqk_mul_mat_moe'
collect2: error: ld returned 1 exit status
gmake[2]: *** [examples/gguf/CMakeFiles/llama-gguf.dir/build.make:101: bin/llama-gguf] Error 1
gmake[1]: *** [CMakeFiles/Makefile2:3196: examples/gguf/CMakeFiles/llama-gguf.dir/all] Error 2
gmake[1]: *** Waiting for unfinished jobs....
[ 13%] Building CXX object src/CMakeFiles/llama.dir/llama-grammar.cpp.o
/usr/bin/ld: ../../ggml/src/libggml.so: undefined reference to `iqk_moe_fused_up_gate'
/usr/bin/ld: ../../ggml/src/libggml.so: undefined reference to `iqk_mul_mat_4d'
/usr/bin/ld: ../../ggml/src/libggml.so: undefined reference to `iqk_mul_mat'
/usr/bin/ld: ../../ggml/src/libggml.so: undefined reference to `iqk_flash_attn_noalibi'
/usr/bin/ld: ../../ggml/src/libggml.so: undefined reference to `iqk_mul_mat_moe'
collect2: error: ld returned 1 exit status
gmake[2]: *** [examples/gguf-hash/CMakeFiles/llama-gguf-hash.dir/build.make:107: bin/llama-gguf-hash] Error 1
gmake[1]: *** [CMakeFiles/Makefile2:3038: examples/gguf-hash/CMakeFiles/llama-gguf-hash.dir/all] Error 2
[ 13%] Building CXX object src/CMakeFiles/llama.dir/llama-sampling.cpp.o
[ 14%] Building CXX object src/CMakeFiles/llama.dir/unicode.cpp.o
[ 14%] Building CXX object src/CMakeFiles/llama.dir/unicode-data.cpp.o
[ 15%] Linking CXX shared library libllama.so
[ 15%] Built target llama
gmake: *** [Makefile:146: all] Error 2
```

---

👤 **smpurkis** commented the **2025-04-03** at **13:10:10**:<br>

Happy to close this issue if it is too much trouble. I believe this is a similar environment to an android phone running termux, I can try it on that as well.

---

👤 **ikawrakow** commented the **2025-04-03** at **13:25:32**:<br>

No, it would be useful to resolve it (if you have the time to test). I'm curious about performance on a Graviton CPU.

Somehow `cmake` (or the compiler?) doesn't like the manually overwritten flags, and as a result `-march=native` (or whatever is needed on this system) doesn't get added to the compilation. This disables the SIMD instructions, which leads to the needed functions not being compiled (and it is useless to run LLM inference without SIMD enabled).

I guess, if `DGGML_NATIVE` didn't help, then the next thing to try is to add `-march=native` to `-DCMAKE_CXX_FLAGS` and `-DCMAKE_C_FLAGS`.
I don't know if the correct flag is `-march=native` or perhaps `-mcpu=native`, or perhaps even `-Xaarch64-march=armv8.5-a+dotprod+fp16`

---

👤 **smpurkis** commented the **2025-04-03** at **13:40:16**:<br>

Adding `-march=native` to `-DCMAKE_CXX_FLAGS` and `-DCMAKE_C_FLAGS` worked. In full
```
cmake -B build -DCMAKE_CXX_FLAGS="-fpermissive -flax-vector-conversions -march=native" -DCMAKE_C_FLAGS="-flax-vector-conversions -march=native" && cmake --build build --config Release
```

---

👤 **ikawrakow** commented the **2025-04-03** at **13:50:21**:<br>

Great! Thank you for the patience. If you come around to test, I would be interested in the results.

---

👤 **smpurkis** commented the **2025-04-03** at **13:58:00**:<br>

Happy to test/benchmark, is there a script to run similar benchmarks to in the readme?

---

👤 **ikawrakow** commented the **2025-04-03** at **14:14:50**:<br>

The benchmarks were done using `llama-bench`.

To test prompt processing (PP) performance,
```
./bin/llama-bench -m $model -p 512 -n 0 -t $num_threads
```
where `$model` is some GGUF file that you have downloaded/prepared.

For token generation (TG) performance, same but `-p 0 -n 128`. TG performance is often better for few threads than the maximum available on the system. To investigate this, you can use a comma-separated list after `-t` (e.g., `-t 4,8,16,32`) with the `llama-bench` command.

One can also look into TG performance with some amount of tokens in the KV cache (more realistic for an actual interaction with the model). For that use `-p 0 -n 0 -gp Np,Nt`, where `Np` is the prompt (context) in tokens, and `Nt` is how many tokens to generate and measure (but this test takes longer).

All this (and other usage) is basically the same as mainline `llama.cpp` (expect for `-gp`, which is missing in mainline).

When running on the CPU one can gain quite a bit of prompt processing performance by using run-time-repacking. This is enabled with `-rtr 1` in the `llama-bench` command. `-rtr` makes model loading longer, so you can repack offline and then use the repacked model without `-rtr` like this
```
./bin/llama-quantize --repack $model_file_name $repacked_model_file_name $quantization_type
```
`quantization_type` is not really required but must be provided on the command line, and is one of `q8_0, q6_0, etc` (any of the available quantization types).

Let me know if you have more questions.

---

👤 **smpurkis** commented the **2025-04-04** at **15:43:05**:<br>

Here is what I got running the bench script over a variety of qwen 2.5 3b quants from https://huggingface.co/bartowski

```
llama.cpp, commit id 74d4f5b041ad837153b0e90fc864b8290e01d8d5
| model                          |       size |     params | backend    | threads |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
| qwen2 3B IQ3_S mix - 3.66 bpw  |   1.38 GiB |     3.09 B | CPU        |       1 |          pp64 |          1.62 ± 0.00 |
| qwen2 3B IQ3_S mix - 3.66 bpw  |   1.38 GiB |     3.09 B | CPU        |       1 |          tg32 |          1.41 ± 0.00 |
| qwen2 3B IQ3_S mix - 3.66 bpw  |   1.38 GiB |     3.09 B | CPU        |       2 |          pp64 |          3.23 ± 0.01 |
| qwen2 3B IQ3_S mix - 3.66 bpw  |   1.38 GiB |     3.09 B | CPU        |       2 |          tg32 |          2.75 ± 0.00 |
| qwen2 3B IQ3_S mix - 3.66 bpw  |   1.38 GiB |     3.09 B | CPU        |       3 |          pp64 |          4.76 ± 0.01 |
| qwen2 3B IQ3_S mix - 3.66 bpw  |   1.38 GiB |     3.09 B | CPU        |       3 |          tg32 |          3.78 ± 0.28 |
| qwen2 3B IQ4_XS - 4.25 bpw     |   1.61 GiB |     3.09 B | CPU        |       1 |          pp64 |          5.90 ± 0.00 |
| qwen2 3B IQ4_XS - 4.25 bpw     |   1.61 GiB |     3.09 B | CPU        |       1 |          tg32 |          3.83 ± 0.01 |
| qwen2 3B IQ4_XS - 4.25 bpw     |   1.61 GiB |     3.09 B | CPU        |       2 |          pp64 |         11.65 ± 0.04 |
| qwen2 3B IQ4_XS - 4.25 bpw     |   1.61 GiB |     3.09 B | CPU        |       2 |          tg32 |          6.93 ± 0.05 |
| qwen2 3B IQ4_XS - 4.25 bpw     |   1.61 GiB |     3.09 B | CPU        |       3 |          pp64 |         17.01 ± 0.16 |
| qwen2 3B IQ4_XS - 4.25 bpw     |   1.61 GiB |     3.09 B | CPU        |       3 |          tg32 |          9.37 ± 0.41 |
| qwen2 3B Q3_K - Large          |   1.58 GiB |     3.09 B | CPU        |       1 |          pp64 |          3.46 ± 0.00 |
| qwen2 3B Q3_K - Large          |   1.58 GiB |     3.09 B | CPU        |       1 |          tg32 |          2.77 ± 0.01 |
| qwen2 3B Q3_K - Large          |   1.58 GiB |     3.09 B | CPU        |       2 |          pp64 |          6.89 ± 0.01 |
| qwen2 3B Q3_K - Large          |   1.58 GiB |     3.09 B | CPU        |       2 |          tg32 |          5.29 ± 0.01 |
| qwen2 3B Q3_K - Large          |   1.58 GiB |     3.09 B | CPU        |       3 |          pp64 |          9.82 ± 0.57 |
| qwen2 3B Q3_K - Large          |   1.58 GiB |     3.09 B | CPU        |       3 |          tg32 |          7.24 ± 0.31 |
| qwen2 3B Q4_0                  |   1.70 GiB |     3.09 B | CPU        |       1 |          pp64 |         16.01 ± 0.02 |
| qwen2 3B Q4_0                  |   1.70 GiB |     3.09 B | CPU        |       1 |          tg32 |          4.73 ± 0.04 |
| qwen2 3B Q4_0                  |   1.70 GiB |     3.09 B | CPU        |       2 |          pp64 |         31.59 ± 0.16 |
| qwen2 3B Q4_0                  |   1.70 GiB |     3.09 B | CPU        |       2 |          tg32 |          8.91 ± 0.15 |
| qwen2 3B Q4_0                  |   1.70 GiB |     3.09 B | CPU        |       3 |          pp64 |         45.77 ± 0.56 |
| qwen2 3B Q4_0                  |   1.70 GiB |     3.09 B | CPU        |       3 |          tg32 |         11.86 ± 0.88 |
| qwen2 3B Q4_K - Medium         |   1.79 GiB |     3.09 B | CPU        |       1 |          pp64 |          5.03 ± 0.01 |
| qwen2 3B Q4_K - Medium         |   1.79 GiB |     3.09 B | CPU        |       1 |          tg32 |          3.41 ± 0.01 |
| qwen2 3B Q4_K - Medium         |   1.79 GiB |     3.09 B | CPU        |       2 |          pp64 |          9.95 ± 0.03 |
| qwen2 3B Q4_K - Medium         |   1.79 GiB |     3.09 B | CPU        |       2 |          tg32 |          6.37 ± 0.04 |
| qwen2 3B Q4_K - Medium         |   1.79 GiB |     3.09 B | CPU        |       3 |          pp64 |         14.68 ± 0.20 |
| qwen2 3B Q4_K - Medium         |   1.79 GiB |     3.09 B | CPU        |       3 |          tg32 |          9.06 ± 0.19 |
| qwen2 3B Q5_K - Medium         |   2.14 GiB |     3.09 B | CPU        |       1 |          pp64 |          3.44 ± 0.01 |
| qwen2 3B Q5_K - Medium         |   2.14 GiB |     3.09 B | CPU        |       1 |          tg32 |          2.67 ± 0.02 |
| qwen2 3B Q5_K - Medium         |   2.14 GiB |     3.09 B | CPU        |       2 |          pp64 |          6.87 ± 0.02 |
| qwen2 3B Q5_K - Medium         |   2.14 GiB |     3.09 B | CPU        |       2 |          tg32 |          5.06 ± 0.03 |
| qwen2 3B Q5_K - Medium         |   2.14 GiB |     3.09 B | CPU        |       3 |          pp64 |         10.09 ± 0.07 |
| qwen2 3B Q5_K - Medium         |   2.14 GiB |     3.09 B | CPU        |       3 |          tg32 |          7.10 ± 0.31 |
| qwen2 3B Q6_K                  |   2.36 GiB |     3.09 B | CPU        |       1 |          pp64 |          2.90 ± 0.00 |
| qwen2 3B Q6_K                  |   2.36 GiB |     3.09 B | CPU        |       1 |          tg32 |          2.23 ± 0.01 |
| qwen2 3B Q6_K                  |   2.36 GiB |     3.09 B | CPU        |       2 |          pp64 |          5.75 ± 0.04 |
| qwen2 3B Q6_K                  |   2.36 GiB |     3.09 B | CPU        |       2 |          tg32 |          4.20 ± 0.03 |
| qwen2 3B Q6_K                  |   2.36 GiB |     3.09 B | CPU        |       3 |          pp64 |          8.46 ± 0.09 |
| qwen2 3B Q6_K                  |   2.36 GiB |     3.09 B | CPU        |       3 |          tg32 |          5.83 ± 0.31 |
| qwen2 3B Q8_0                  |   3.05 GiB |     3.09 B | CPU        |       1 |          pp64 |          6.37 ± 0.02 |
| qwen2 3B Q8_0                  |   3.05 GiB |     3.09 B | CPU        |       1 |          tg32 |          2.78 ± 0.05 |
| qwen2 3B Q8_0                  |   3.05 GiB |     3.09 B | CPU        |       2 |          pp64 |         12.60 ± 0.08 |
| qwen2 3B Q8_0                  |   3.05 GiB |     3.09 B | CPU        |       2 |          tg32 |          5.00 ± 0.27 |
| qwen2 3B Q8_0                  |   3.05 GiB |     3.09 B | CPU        |       3 |          pp64 |         17.58 ± 0.78 |
| qwen2 3B Q8_0                  |   3.05 GiB |     3.09 B | CPU        |       3 |          tg32 |          7.12 ± 0.10 |


ik_llama.cpp, commit id 310bce3c1db882c2e057582c546a8bc3c04478e1
| model                          |       size |     params | backend    | threads |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
| qwen2 ?B IQ3_S mix - 3.66 bpw  |   1.62 GiB |     3.40 B | CPU        |       1 |          pp64 |      6.13 ± 0.02 |
| qwen2 ?B IQ3_S mix - 3.66 bpw  |   1.62 GiB |     3.40 B | CPU        |       1 |          tg32 |      1.42 ± 0.00 |
| qwen2 ?B IQ3_S mix - 3.66 bpw  |   1.62 GiB |     3.40 B | CPU        |       2 |          pp64 |     12.14 ± 0.06 |
| qwen2 ?B IQ3_S mix - 3.66 bpw  |   1.62 GiB |     3.40 B | CPU        |       2 |          tg32 |      2.79 ± 0.01 |
| qwen2 ?B IQ3_S mix - 3.66 bpw  |   1.62 GiB |     3.40 B | CPU        |       3 |          pp64 |     17.73 ± 0.26 |
| qwen2 ?B IQ3_S mix - 3.66 bpw  |   1.62 GiB |     3.40 B | CPU        |       3 |          tg32 |      3.93 ± 0.10 |
| qwen2 ?B IQ4_XS - 4.25 bpw     |   1.85 GiB |     3.40 B | CPU        |       1 |          pp64 |      8.40 ± 0.04 |
| qwen2 ?B IQ4_XS - 4.25 bpw     |   1.85 GiB |     3.40 B | CPU        |       1 |          tg32 |      3.74 ± 0.01 |
| qwen2 ?B IQ4_XS - 4.25 bpw     |   1.85 GiB |     3.40 B | CPU        |       2 |          pp64 |     16.66 ± 0.03 |
| qwen2 ?B IQ4_XS - 4.25 bpw     |   1.85 GiB |     3.40 B | CPU        |       2 |          tg32 |      7.20 ± 0.10 |
| qwen2 ?B IQ4_XS - 4.25 bpw     |   1.85 GiB |     3.40 B | CPU        |       3 |          pp64 |     24.33 ± 0.15 |
| qwen2 ?B IQ4_XS - 4.25 bpw     |   1.85 GiB |     3.40 B | CPU        |       3 |          tg32 |     10.10 ± 0.35 |
| qwen2 ?B Q3_K - Large          |   1.82 GiB |     3.40 B | CPU        |       1 |          pp64 |      5.75 ± 0.02 |
| qwen2 ?B Q3_K - Large          |   1.82 GiB |     3.40 B | CPU        |       1 |          tg32 |      2.60 ± 0.01 |
| qwen2 ?B Q3_K - Large          |   1.82 GiB |     3.40 B | CPU        |       2 |          pp64 |     11.45 ± 0.07 |
| qwen2 ?B Q3_K - Large          |   1.82 GiB |     3.40 B | CPU        |       2 |          tg32 |      5.07 ± 0.02 |
| qwen2 ?B Q3_K - Large          |   1.82 GiB |     3.40 B | CPU        |       3 |          pp64 |     16.80 ± 0.19 |
| qwen2 ?B Q3_K - Large          |   1.82 GiB |     3.40 B | CPU        |       3 |          tg32 |      7.11 ± 0.30 |
| qwen2 ?B Q4_0                  |   1.94 GiB |     3.40 B | CPU        |       1 |          pp64 |      8.29 ± 0.02 |
| qwen2 ?B Q4_0                  |   1.94 GiB |     3.40 B | CPU        |       1 |          tg32 |      3.81 ± 0.03 |
| qwen2 ?B Q4_0                  |   1.94 GiB |     3.40 B | CPU        |       2 |          pp64 |     16.43 ± 0.13 |
| qwen2 ?B Q4_0                  |   1.94 GiB |     3.40 B | CPU        |       2 |          tg32 |      7.34 ± 0.07 |
| qwen2 ?B Q4_0                  |   1.94 GiB |     3.40 B | CPU        |       3 |          pp64 |     23.86 ± 0.37 |
| qwen2 ?B Q4_0                  |   1.94 GiB |     3.40 B | CPU        |       3 |          tg32 |     10.39 ± 0.37 |
| qwen2 ?B Q4_K - Medium         |   2.03 GiB |     3.40 B | CPU        |       1 |          pp64 |      7.55 ± 0.02 |
| qwen2 ?B Q4_K - Medium         |   2.03 GiB |     3.40 B | CPU        |       1 |          tg32 |      3.43 ± 0.01 |
| qwen2 ?B Q4_K - Medium         |   2.03 GiB |     3.40 B | CPU        |       2 |          pp64 |     15.56 ± 0.06 |
| qwen2 ?B Q4_K - Medium         |   2.03 GiB |     3.40 B | CPU        |       2 |          tg32 |      6.63 ± 0.06 |
| qwen2 ?B Q4_K - Medium         |   2.03 GiB |     3.40 B | CPU        |       3 |          pp64 |     22.73 ± 0.58 |
| qwen2 ?B Q4_K - Medium         |   2.03 GiB |     3.40 B | CPU        |       3 |          tg32 |      8.94 ± 0.56 |
| qwen2 ?B Q5_K - Medium         |   2.30 GiB |     3.40 B | CPU        |       1 |          pp64 |      7.09 ± 0.02 |
| qwen2 ?B Q5_K - Medium         |   2.30 GiB |     3.40 B | CPU        |       1 |          tg32 |      2.60 ± 0.01 |
| qwen2 ?B Q5_K - Medium         |   2.30 GiB |     3.40 B | CPU        |       2 |          pp64 |     13.99 ± 0.07 |
| qwen2 ?B Q5_K - Medium         |   2.30 GiB |     3.40 B | CPU        |       2 |          tg32 |      5.02 ± 0.04 |
| qwen2 ?B Q5_K - Medium         |   2.30 GiB |     3.40 B | CPU        |       3 |          pp64 |     20.50 ± 0.21 |
| qwen2 ?B Q5_K - Medium         |   2.30 GiB |     3.40 B | CPU        |       3 |          tg32 |      7.12 ± 0.21 |
| qwen2 ?B Q6_K                  |   2.60 GiB |     3.40 B | CPU        |       1 |          pp64 |      5.35 ± 0.02 |
| qwen2 ?B Q6_K                  |   2.60 GiB |     3.40 B | CPU        |       1 |          tg32 |      2.64 ± 0.01 |
| qwen2 ?B Q6_K                  |   2.60 GiB |     3.40 B | CPU        |       2 |          pp64 |     10.61 ± 0.07 |
| qwen2 ?B Q6_K                  |   2.60 GiB |     3.40 B | CPU        |       2 |          tg32 |      5.14 ± 0.03 |
| qwen2 ?B Q6_K                  |   2.60 GiB |     3.40 B | CPU        |       3 |          pp64 |     15.33 ± 0.61 |
| qwen2 ?B Q6_K                  |   2.60 GiB |     3.40 B | CPU        |       3 |          tg32 |      7.26 ± 0.16 |
| qwen2 ?B Q8_0                  |   3.36 GiB |     3.40 B | CPU        |       1 |          pp64 |      7.34 ± 0.13 |
| qwen2 ?B Q8_0                  |   3.36 GiB |     3.40 B | CPU        |       1 |          tg32 |      3.11 ± 0.02 |
| qwen2 ?B Q8_0                  |   3.36 GiB |     3.40 B | CPU        |       2 |          pp64 |     14.25 ± 0.51 |
| qwen2 ?B Q8_0                  |   3.36 GiB |     3.40 B | CPU        |       2 |          tg32 |      5.86 ± 0.08 |
| qwen2 ?B Q8_0                  |   3.36 GiB |     3.40 B | CPU        |       3 |          pp64 |     21.18 ± 0.39 |
| qwen2 ?B Q8_0                  |   3.36 GiB |     3.40 B | CPU        |       3 |          tg32 |      8.17 ± 0.31 |
```
ik_llama.cpp is faster on all except q4_0 format.

---

👤 **smpurkis** commented the **2025-04-04** at **15:43:05**:<br>

Here is what I got running the bench script over a variety of qwen 2.5 3b quants from https://huggingface.co/bartowski

```
llama.cpp, commit id 74d4f5b041ad837153b0e90fc864b8290e01d8d5
| model                          |       size |     params | backend    | threads |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
| qwen2 3B IQ3_S mix - 3.66 bpw  |   1.38 GiB |     3.09 B | CPU        |       1 |          pp64 |          1.62 ± 0.00 |
| qwen2 3B IQ3_S mix - 3.66 bpw  |   1.38 GiB |     3.09 B | CPU        |       1 |          tg32 |          1.41 ± 0.00 |
| qwen2 3B IQ3_S mix - 3.66 bpw  |   1.38 GiB |     3.09 B | CPU        |       2 |          pp64 |          3.23 ± 0.01 |
| qwen2 3B IQ3_S mix - 3.66 bpw  |   1.38 GiB |     3.09 B | CPU        |       2 |          tg32 |          2.75 ± 0.00 |
| qwen2 3B IQ3_S mix - 3.66 bpw  |   1.38 GiB |     3.09 B | CPU        |       3 |          pp64 |          4.76 ± 0.01 |
| qwen2 3B IQ3_S mix - 3.66 bpw  |   1.38 GiB |     3.09 B | CPU        |       3 |          tg32 |          3.78 ± 0.28 |
| qwen2 3B IQ4_XS - 4.25 bpw     |   1.61 GiB |     3.09 B | CPU        |       1 |          pp64 |          5.90 ± 0.00 |
| qwen2 3B IQ4_XS - 4.25 bpw     |   1.61 GiB |     3.09 B | CPU        |       1 |          tg32 |          3.83 ± 0.01 |
| qwen2 3B IQ4_XS - 4.25 bpw     |   1.61 GiB |     3.09 B | CPU        |       2 |          pp64 |         11.65 ± 0.04 |
| qwen2 3B IQ4_XS - 4.25 bpw     |   1.61 GiB |     3.09 B | CPU        |       2 |          tg32 |          6.93 ± 0.05 |
| qwen2 3B IQ4_XS - 4.25 bpw     |   1.61 GiB |     3.09 B | CPU        |       3 |          pp64 |         17.01 ± 0.16 |
| qwen2 3B IQ4_XS - 4.25 bpw     |   1.61 GiB |     3.09 B | CPU        |       3 |          tg32 |          9.37 ± 0.41 |
| qwen2 3B Q3_K - Large          |   1.58 GiB |     3.09 B | CPU        |       1 |          pp64 |          3.46 ± 0.00 |
| qwen2 3B Q3_K - Large          |   1.58 GiB |     3.09 B | CPU        |       1 |          tg32 |          2.77 ± 0.01 |
| qwen2 3B Q3_K - Large          |   1.58 GiB |     3.09 B | CPU        |       2 |          pp64 |          6.89 ± 0.01 |
| qwen2 3B Q3_K - Large          |   1.58 GiB |     3.09 B | CPU        |       2 |          tg32 |          5.29 ± 0.01 |
| qwen2 3B Q3_K - Large          |   1.58 GiB |     3.09 B | CPU        |       3 |          pp64 |          9.82 ± 0.57 |
| qwen2 3B Q3_K - Large          |   1.58 GiB |     3.09 B | CPU        |       3 |          tg32 |          7.24 ± 0.31 |
| qwen2 3B Q4_0                  |   1.70 GiB |     3.09 B | CPU        |       1 |          pp64 |         16.01 ± 0.02 |
| qwen2 3B Q4_0                  |   1.70 GiB |     3.09 B | CPU        |       1 |          tg32 |          4.73 ± 0.04 |
| qwen2 3B Q4_0                  |   1.70 GiB |     3.09 B | CPU        |       2 |          pp64 |         31.59 ± 0.16 |
| qwen2 3B Q4_0                  |   1.70 GiB |     3.09 B | CPU        |       2 |          tg32 |          8.91 ± 0.15 |
| qwen2 3B Q4_0                  |   1.70 GiB |     3.09 B | CPU        |       3 |          pp64 |         45.77 ± 0.56 |
| qwen2 3B Q4_0                  |   1.70 GiB |     3.09 B | CPU        |       3 |          tg32 |         11.86 ± 0.88 |
| qwen2 3B Q4_K - Medium         |   1.79 GiB |     3.09 B | CPU        |       1 |          pp64 |          5.03 ± 0.01 |
| qwen2 3B Q4_K - Medium         |   1.79 GiB |     3.09 B | CPU        |       1 |          tg32 |          3.41 ± 0.01 |
| qwen2 3B Q4_K - Medium         |   1.79 GiB |     3.09 B | CPU        |       2 |          pp64 |          9.95 ± 0.03 |
| qwen2 3B Q4_K - Medium         |   1.79 GiB |     3.09 B | CPU        |       2 |          tg32 |          6.37 ± 0.04 |
| qwen2 3B Q4_K - Medium         |   1.79 GiB |     3.09 B | CPU        |       3 |          pp64 |         14.68 ± 0.20 |
| qwen2 3B Q4_K - Medium         |   1.79 GiB |     3.09 B | CPU        |       3 |          tg32 |          9.06 ± 0.19 |
| qwen2 3B Q5_K - Medium         |   2.14 GiB |     3.09 B | CPU        |       1 |          pp64 |          3.44 ± 0.01 |
| qwen2 3B Q5_K - Medium         |   2.14 GiB |     3.09 B | CPU        |       1 |          tg32 |          2.67 ± 0.02 |
| qwen2 3B Q5_K - Medium         |   2.14 GiB |     3.09 B | CPU        |       2 |          pp64 |          6.87 ± 0.02 |
| qwen2 3B Q5_K - Medium         |   2.14 GiB |     3.09 B | CPU        |       2 |          tg32 |          5.06 ± 0.03 |
| qwen2 3B Q5_K - Medium         |   2.14 GiB |     3.09 B | CPU        |       3 |          pp64 |         10.09 ± 0.07 |
| qwen2 3B Q5_K - Medium         |   2.14 GiB |     3.09 B | CPU        |       3 |          tg32 |          7.10 ± 0.31 |
| qwen2 3B Q6_K                  |   2.36 GiB |     3.09 B | CPU        |       1 |          pp64 |          2.90 ± 0.00 |
| qwen2 3B Q6_K                  |   2.36 GiB |     3.09 B | CPU        |       1 |          tg32 |          2.23 ± 0.01 |
| qwen2 3B Q6_K                  |   2.36 GiB |     3.09 B | CPU        |       2 |          pp64 |          5.75 ± 0.04 |
| qwen2 3B Q6_K                  |   2.36 GiB |     3.09 B | CPU        |       2 |          tg32 |          4.20 ± 0.03 |
| qwen2 3B Q6_K                  |   2.36 GiB |     3.09 B | CPU        |       3 |          pp64 |          8.46 ± 0.09 |
| qwen2 3B Q6_K                  |   2.36 GiB |     3.09 B | CPU        |       3 |          tg32 |          5.83 ± 0.31 |
| qwen2 3B Q8_0                  |   3.05 GiB |     3.09 B | CPU        |       1 |          pp64 |          6.37 ± 0.02 |
| qwen2 3B Q8_0                  |   3.05 GiB |     3.09 B | CPU        |       1 |          tg32 |          2.78 ± 0.05 |
| qwen2 3B Q8_0                  |   3.05 GiB |     3.09 B | CPU        |       2 |          pp64 |         12.60 ± 0.08 |
| qwen2 3B Q8_0                  |   3.05 GiB |     3.09 B | CPU        |       2 |          tg32 |          5.00 ± 0.27 |
| qwen2 3B Q8_0                  |   3.05 GiB |     3.09 B | CPU        |       3 |          pp64 |         17.58 ± 0.78 |
| qwen2 3B Q8_0                  |   3.05 GiB |     3.09 B | CPU        |       3 |          tg32 |          7.12 ± 0.10 |


ik_llama.cpp, commit id 310bce3c1db882c2e057582c546a8bc3c04478e1
| model                          |       size |     params | backend    | threads |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
| qwen2 ?B IQ3_S mix - 3.66 bpw  |   1.62 GiB |     3.40 B | CPU        |       1 |          pp64 |      6.13 ± 0.02 |
| qwen2 ?B IQ3_S mix - 3.66 bpw  |   1.62 GiB |     3.40 B | CPU        |       1 |          tg32 |      1.42 ± 0.00 |
| qwen2 ?B IQ3_S mix - 3.66 bpw  |   1.62 GiB |     3.40 B | CPU        |       2 |          pp64 |     12.14 ± 0.06 |
| qwen2 ?B IQ3_S mix - 3.66 bpw  |   1.62 GiB |     3.40 B | CPU        |       2 |          tg32 |      2.79 ± 0.01 |
| qwen2 ?B IQ3_S mix - 3.66 bpw  |   1.62 GiB |     3.40 B | CPU        |       3 |          pp64 |     17.73 ± 0.26 |
| qwen2 ?B IQ3_S mix - 3.66 bpw  |   1.62 GiB |     3.40 B | CPU        |       3 |          tg32 |      3.93 ± 0.10 |
| qwen2 ?B IQ4_XS - 4.25 bpw     |   1.85 GiB |     3.40 B | CPU        |       1 |          pp64 |      8.40 ± 0.04 |
| qwen2 ?B IQ4_XS - 4.25 bpw     |   1.85 GiB |     3.40 B | CPU        |       1 |          tg32 |      3.74 ± 0.01 |
| qwen2 ?B IQ4_XS - 4.25 bpw     |   1.85 GiB |     3.40 B | CPU        |       2 |          pp64 |     16.66 ± 0.03 |
| qwen2 ?B IQ4_XS - 4.25 bpw     |   1.85 GiB |     3.40 B | CPU        |       2 |          tg32 |      7.20 ± 0.10 |
| qwen2 ?B IQ4_XS - 4.25 bpw     |   1.85 GiB |     3.40 B | CPU        |       3 |          pp64 |     24.33 ± 0.15 |
| qwen2 ?B IQ4_XS - 4.25 bpw     |   1.85 GiB |     3.40 B | CPU        |       3 |          tg32 |     10.10 ± 0.35 |
| qwen2 ?B Q3_K - Large          |   1.82 GiB |     3.40 B | CPU        |       1 |          pp64 |      5.75 ± 0.02 |
| qwen2 ?B Q3_K - Large          |   1.82 GiB |     3.40 B | CPU        |       1 |          tg32 |      2.60 ± 0.01 |
| qwen2 ?B Q3_K - Large          |   1.82 GiB |     3.40 B | CPU        |       2 |          pp64 |     11.45 ± 0.07 |
| qwen2 ?B Q3_K - Large          |   1.82 GiB |     3.40 B | CPU        |       2 |          tg32 |      5.07 ± 0.02 |
| qwen2 ?B Q3_K - Large          |   1.82 GiB |     3.40 B | CPU        |       3 |          pp64 |     16.80 ± 0.19 |
| qwen2 ?B Q3_K - Large          |   1.82 GiB |     3.40 B | CPU        |       3 |          tg32 |      7.11 ± 0.30 |
| qwen2 ?B Q4_0                  |   1.94 GiB |     3.40 B | CPU        |       1 |          pp64 |      8.29 ± 0.02 |
| qwen2 ?B Q4_0                  |   1.94 GiB |     3.40 B | CPU        |       1 |          tg32 |      3.81 ± 0.03 |
| qwen2 ?B Q4_0                  |   1.94 GiB |     3.40 B | CPU        |       2 |          pp64 |     16.43 ± 0.13 |
| qwen2 ?B Q4_0                  |   1.94 GiB |     3.40 B | CPU        |       2 |          tg32 |      7.34 ± 0.07 |
| qwen2 ?B Q4_0                  |   1.94 GiB |     3.40 B | CPU        |       3 |          pp64 |     23.86 ± 0.37 |
| qwen2 ?B Q4_0                  |   1.94 GiB |     3.40 B | CPU        |       3 |          tg32 |     10.39 ± 0.37 |
| qwen2 ?B Q4_K - Medium         |   2.03 GiB |     3.40 B | CPU        |       1 |          pp64 |      7.55 ± 0.02 |
| qwen2 ?B Q4_K - Medium         |   2.03 GiB |     3.40 B | CPU        |       1 |          tg32 |      3.43 ± 0.01 |
| qwen2 ?B Q4_K - Medium         |   2.03 GiB |     3.40 B | CPU        |       2 |          pp64 |     15.56 ± 0.06 |
| qwen2 ?B Q4_K - Medium         |   2.03 GiB |     3.40 B | CPU        |       2 |          tg32 |      6.63 ± 0.06 |
| qwen2 ?B Q4_K - Medium         |   2.03 GiB |     3.40 B | CPU        |       3 |          pp64 |     22.73 ± 0.58 |
| qwen2 ?B Q4_K - Medium         |   2.03 GiB |     3.40 B | CPU        |       3 |          tg32 |      8.94 ± 0.56 |
| qwen2 ?B Q5_K - Medium         |   2.30 GiB |     3.40 B | CPU        |       1 |          pp64 |      7.09 ± 0.02 |
| qwen2 ?B Q5_K - Medium         |   2.30 GiB |     3.40 B | CPU        |       1 |          tg32 |      2.60 ± 0.01 |
| qwen2 ?B Q5_K - Medium         |   2.30 GiB |     3.40 B | CPU        |       2 |          pp64 |     13.99 ± 0.07 |
| qwen2 ?B Q5_K - Medium         |   2.30 GiB |     3.40 B | CPU        |       2 |          tg32 |      5.02 ± 0.04 |
| qwen2 ?B Q5_K - Medium         |   2.30 GiB |     3.40 B | CPU        |       3 |          pp64 |     20.50 ± 0.21 |
| qwen2 ?B Q5_K - Medium         |   2.30 GiB |     3.40 B | CPU        |       3 |          tg32 |      7.12 ± 0.21 |
| qwen2 ?B Q6_K                  |   2.60 GiB |     3.40 B | CPU        |       1 |          pp64 |      5.35 ± 0.02 |
| qwen2 ?B Q6_K                  |   2.60 GiB |     3.40 B | CPU        |       1 |          tg32 |      2.64 ± 0.01 |
| qwen2 ?B Q6_K                  |   2.60 GiB |     3.40 B | CPU        |       2 |          pp64 |     10.61 ± 0.07 |
| qwen2 ?B Q6_K                  |   2.60 GiB |     3.40 B | CPU        |       2 |          tg32 |      5.14 ± 0.03 |
| qwen2 ?B Q6_K                  |   2.60 GiB |     3.40 B | CPU        |       3 |          pp64 |     15.33 ± 0.61 |
| qwen2 ?B Q6_K                  |   2.60 GiB |     3.40 B | CPU        |       3 |          tg32 |      7.26 ± 0.16 |
| qwen2 ?B Q8_0                  |   3.36 GiB |     3.40 B | CPU        |       1 |          pp64 |      7.34 ± 0.13 |
| qwen2 ?B Q8_0                  |   3.36 GiB |     3.40 B | CPU        |       1 |          tg32 |      3.11 ± 0.02 |
| qwen2 ?B Q8_0                  |   3.36 GiB |     3.40 B | CPU        |       2 |          pp64 |     14.25 ± 0.51 |
| qwen2 ?B Q8_0                  |   3.36 GiB |     3.40 B | CPU        |       2 |          tg32 |      5.86 ± 0.08 |
| qwen2 ?B Q8_0                  |   3.36 GiB |     3.40 B | CPU        |       3 |          pp64 |     21.18 ± 0.39 |
| qwen2 ?B Q8_0                  |   3.36 GiB |     3.40 B | CPU        |       3 |          tg32 |      8.17 ± 0.31 |
```

---

👤 **ikawrakow** commented the **2025-04-04** at **15:49:16**:<br>

Thank you for these.

The CPU has only 3 corse?

To beat `llama.cpp` also for `Q4_0` quants, you need to use `-rtr 1`.

---

👤 **smpurkis** commented the **2025-04-04** at **15:55:08**:<br>

Ah, my mistake, will try again with `-rtr 1`. It has 4 cores, but lags badly when using all 4, so generally use 3 as other services are running on the server.

---

👤 **smpurkis** commented the **2025-04-04** at **16:04:17**:<br>

This is the results with `-rtr 1`, a bit slower than llama.cpp, about 30% slower on pp, same speed on tg though
```
| model                          |       size |     params | backend    | threads | rtr |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --: | ------------: | ---------------: |
| qwen2 ?B Q4_0                  |   1.94 GiB |     3.40 B | CPU        |       1 |   1 |          pp64 |     12.00 ± 0.22 |
| qwen2 ?B Q4_0                  |   1.94 GiB |     3.40 B | CPU        |       1 |   1 |          tg32 |      4.77 ± 0.02 |
| qwen2 ?B Q4_0                  |   1.94 GiB |     3.40 B | CPU        |       2 |   1 |          pp64 |     23.98 ± 0.17 |
| qwen2 ?B Q4_0                  |   1.94 GiB |     3.40 B | CPU        |       2 |   1 |          tg32 |      8.91 ± 0.13 |
| qwen2 ?B Q4_0                  |   1.94 GiB |     3.40 B | CPU        |       3 |   1 |          pp64 |     32.36 ± 3.46 |
| qwen2 ?B Q4_0                  |   1.94 GiB |     3.40 B | CPU        |       3 |   1 |          tg32 |     12.25 ± 0.74 |
```

---

👤 **ikawrakow** commented the **2025-04-04** at **16:11:38**:<br>

Interesting. On the M2-Max and any `x86_64` my `Q4_0` implementation beats mainline.