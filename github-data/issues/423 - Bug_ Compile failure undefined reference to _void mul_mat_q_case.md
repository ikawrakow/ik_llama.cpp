### 🐛 [#423](https://github.com/ikawrakow/ik_llama.cpp/issues/423) - Bug: Compile failure undefined reference to `void mul_mat_q_case

| **Author** | `nux` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-15 |
| **Updated** | 2025-05-15 |

---

#### Description

### What happened?

[ 62%] Building CXX object common/CMakeFiles/common.dir/grammar-parser.cpp.o
[ 63%] Building CXX object common/CMakeFiles/common.dir/train.cpp.o
[ 63%] Building CXX object common/CMakeFiles/common.dir/json-schema-to-grammar.cpp.o
[ 64%] Linking CXX executable ../bin/test-c
[ 64%] Built target test-c
[ 64%] Linking CXX executable ../../bin/llama-bench-matmult
[ 64%] Built target llama-bench-matmult
[ 65%] Linking CXX executable ../../bin/llama-quantize-stats
/usr/bin/ld: ../../ggml/src/libggml.a(mmq.cu.o): in function `ggml_cuda_op_mul_mat_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*)':
tmpxft_0000f476_00000000-7_mmq.cudafe1.cpp:(.text+0x120): undefined reference to `void mul_mat_q_case<(ggml_type)152>(ggml_backend_cuda_context&, mmq_args const&, CUstream_st*)'
collect2: error: ld returned 1 exit status
gmake[2]: *** [examples/quantize-stats/CMakeFiles/llama-quantize-stats.dir/build.make:113: bin/llama-quantize-stats] Error 1
gmake[1]: *** [CMakeFiles/Makefile2:3883: examples/quantize-stats/CMakeFiles/llama-quantize-stats.dir/all] Error 2
gmake[1]: *** Waiting for unfinished jobs....
[ 65%] Built target llava
[ 65%] Linking CXX static library libcommon.a
[ 65%] Built target common
gmake: *** [Makefile:146: all] Error 2


Did a git pull before attempting to build
git rev-parse --short HEAD 3d92d7f8

Building with:
cmake -B build -DGGML_CUDA_FA_ALL_QUANTS=ON -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON
cmake --build build --config Release -j --clean-first


### Name and Version

3d92d7f8

Debian latest: Linux red 6.1.0-34-amd64 #1 SMP PREEMPT_DYNAMIC Debian 6.1.135-1 (2025-04-25) x86_64 GNU/Linux


### What operating system are you seeing the problem on?

_No response_

### Relevant log output

```shell

```

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-05-15** at **13:50:36**:<br>

Sorry, forgot to add a file. It should work now.

---

👤 **nux** commented the **2025-05-15** at **13:50:54**:<br>

Thanks! Committed fix before my attempt to build just llama-server completed!