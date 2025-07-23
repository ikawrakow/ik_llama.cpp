### 🔀 [#405](https://github.com/ikawrakow/ik_llama.cpp/pull/405) - GPU offload policy

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-10 |
| **Updated** | 2025-05-12 |

---

#### Description

When part of the tensors are stored in RAM but there are faster back-ends available (GPU), the scheduler needs to decide if to offload the data for a given op to a faster back-end or to compute the op on the CPU. This is currently done via a simple heuristics where only matrix multiplications (`GGML_MUL_MAT` and `GGML_MUL_MAT_ID`) are offloaded if the batch size is larger than some threshold (currently 32). When `fmoe` is enabled, the fused `(ffn_up*X)*unary(ffn_gate*X))` op is never uploaded. In contrast, in mainline `llama.cpp` matrix multiplications are always offloaded when the batch size is `>= 32`. The result of this is that when the batch size becomes large enough, `llama.cpp` will outperform `ik_llama.cpp` in prompt processing speed. As "large enough" depends on many factors (size of tensors that need to be uploaded, speed of the PCI-E bus to the GPU, relative speed of the GPU vs the CPU), it is hard to devise a better offload policy that automatically takes the best decision.

Hence, this PR adds the ability to manually define the offload policy via a command line argument that can be used for all examples that use `common` (`llama-cli, llama-server, llama-sweep-bench, llama-perplexity`, etc.). The argument is
```
-op or --offload-policy a,b
``` 
where `a` and `b` are integers. One can have multiple pairs following the `-op` or `--offload-policy` argument (i.e., `-op a1,b1,a2,b2,a3,b3...`). The first integer defines the op (see below). The second integer is `0` or `1` and defines if the op should be offloaded (`1`) or not offloaded (`0`) to the GPU. The first integer is simply the enum value in the `ggml_op` enum. I know this is clunky, but I also didn't want to go with just allowing or disallowing offload for all ops. If the op is set to `-1`, then all op offloads are set to enabled or disabled. 

<details>
<summary>Current list of ops</summary>

```GGML_OP_NONE = 0 
GGML_OP_DUP = 1 
GGML_OP_ADD = 2 
GGML_OP_ADD1 = 3 
GGML_OP_ACC = 4 
GGML_OP_SUB = 5 
GGML_OP_MUL = 6 
GGML_OP_DIV = 7 
GGML_OP_SQR = 8 
GGML_OP_SQRT = 9 
GGML_OP_LOG = 10
GGML_OP_SUM = 11
GGML_OP_SUM_ROWS = 12
GGML_OP_MEAN = 13
GGML_OP_ARGMAX = 14
GGML_OP_REPEAT = 15
GGML_OP_REPEAT_BACK = 16
GGML_OP_CONCAT = 17
GGML_OP_SILU_BACK = 18
GGML_OP_NORM = 19
GGML_OP_RMS_NORM = 20
GGML_OP_RMS_NORM_BACK = 21
GGML_OP_GROUP_NORM = 22
GGML_OP_FUSED_RMS_NORM = 23
GGML_OP_FUSED_MUL_UNARY = 24
GGML_OP_MULTI_ADD = 25
GGML_OP_MUL_MAT = 26
GGML_OP_MUL_MAT_ID = 27
GGML_OP_OUT_PROD = 28
GGML_OP_MOE_FUSED_UP_GATE = 29
GGML_OP_SCALE = 30
GGML_OP_SET = 31
GGML_OP_CPY = 32
GGML_OP_CONT = 33
GGML_OP_RESHAPE = 34
GGML_OP_VIEW = 35
GGML_OP_PERMUTE = 36
GGML_OP_TRANSPOSE = 37
GGML_OP_GET_ROWS = 38
GGML_OP_GET_ROWS_BACK = 39
GGML_OP_DIAG = 40
GGML_OP_DIAG_MASK_INF = 41
GGML_OP_DIAG_MASK_ZERO = 42
GGML_OP_SOFT_MAX = 43
GGML_OP_SOFT_MAX_BACK = 44
GGML_OP_ROPE = 45
GGML_OP_ROPE_BACK = 46
GGML_OP_CLAMP = 47
GGML_OP_CONV_TRANSPOSE_1D = 48
GGML_OP_IM2COL = 49
GGML_OP_CONV_TRANSPOSE_2D = 50
GGML_OP_POOL_1D = 51
GGML_OP_POOL_2D = 52
GGML_OP_UPSCALE = 53
GGML_OP_PAD = 54
GGML_OP_ARANGE = 55
GGML_OP_TIMESTEP_EMBEDDING = 56
GGML_OP_ARGSORT = 57
GGML_OP_ARGSORT_THRESH = 58
GGML_OP_LEAKY_RELU = 59
GGML_OP_SOFTCAP = 60
GGML_OP_SOFT_CAP_MAX = 61
GGML_OP_FLASH_ATTN_EXT = 62
GGML_OP_FLASH_ATTN_BACK = 63
GGML_OP_SSM_CONV = 64
GGML_OP_SSM_SCAN = 65
GGML_OP_WIN_PART = 66
GGML_OP_WIN_UNPART = 67
GGML_OP_GET_REL_POS = 68
GGML_OP_ADD_REL_POS = 69
GGML_OP_UNARY = 70
GGML_OP_MAP_UNARY = 71
GGML_OP_MAP_BINARY = 72
GGML_OP_MAP_CUSTOM1_F32 = 73
GGML_OP_MAP_CUSTOM2_F32 = 74
GGML_OP_MAP_CUSTOM3_F32 = 75
GGML_OP_MAP_CUSTOM1 = 76
GGML_OP_MAP_CUSTOM2 = 77
GGML_OP_MAP_CUSTOM3 = 78
GGML_OP_CROSS_ENTROPY_LOSS = 79
GGML_OP_CROSS_ENTROPY_LOSS_BACK = 80
GGML_OP_COUNT = 81
```
</details>

Examples:
* `-op -1,0`: disable all offload to the GPU
* `-op 26,0`: disable offload of matrix multiplications to the GPU
* `-op 27,0`: disable offload of indirect  matrix multiplications to the GPU (used for the experts in a MoE model)
* `-op 29,0`: disable fused up-gate-unary op offload to the GPU (applied to MoE models with `-fmoe`)


>[!NOTE]
>Even if offload for an op is enabled, it may still not be offloaded based on the existing heuristics. This is important for, e.g., token generation where batch size is 1 and the offload will take much longer than just computing on the CPU.

>[!IMPORTANT]
>The PR also changes `ik_llama.cpp` to offload fused up-gate-unary ops for batch sizes `>= 32`. If you observe PP performance degradation compared to the main branch, the behavior prior to this PR can be recovered using `-op 29,0`

>[!NOTE]
>Row-interleaved quants (`IQ4_K_R4, IQ4_K_R4, Q4_0_R8`, etc.) are never offloaded because there is no CUDA GEMM/GEMV for these quantization types. Hence, using `-rtr` is equivalent to `-op 26,0,27,0,29,0`

---

#### 💬 Conversation

👤 **Panchovix** commented the **2025-05-10** at **18:12:44**:<br>

Many thanks for the PR! Sorry as I think I didn't understand correctly, for the case we were talking on https://github.com/ikawrakow/ik_llama.cpp/pull/394#issuecomment-2868723515, if we want to do the matrix multiplications on MoE models, we should specify

`-op 26,1,27,1` so the matrix multiplications are done on the GPU, or viceversa?

---

👤 **ikawrakow** commented the **2025-05-10** at **18:22:29**:<br>

This PR sets `ik_llama.cpp` GPU offload behavior to be the same as `llama.cpp`, so you don't need to use the `-op` argument. You would want to use it if you were running for instance Maverick, and then you would use `-op 27,0,29,0`.

---

👤 **Panchovix** commented the **2025-05-10** at **18:33:15**:<br>

Amazing, thanks! Now I'm trying to build from source but I'm getting some compilation issues, not sure if it is the PR or an update (I was on https://github.com/ikawrakow/ik_llama.cpp/commit/43a154d8b8b0e9217114577442cecb224a488d45 before)

```
[ 59%] Building CXX object src/CMakeFiles/llama.dir/unicode-data.cpp.o
/usr/bin/ld: ../../ggml/src/libggml.so: undefined reference to `x000fe200080f0eff'
collect2: error: ld returned 1 exit status
/usr/bin/ld: ../../ggml/src/libggml.so: undefined reference to `x000fe200080f0eff'
gmake[2]: *** [examples/gguf/CMakeFiles/llama-gguf.dir/build.make:103: bin/llama-gguf] Error 1
gmake[1]: *** [CMakeFiles/Makefile2:3260: examples/gguf/CMakeFiles/llama-gguf.dir/all] Error 2
gmake[1]: *** Waiting for unfinished jobs....
collect2: error: ld returned 1 exit status
gmake[2]: *** [examples/gguf-hash/CMakeFiles/llama-gguf-hash.dir/build.make:109: bin/llama-gguf-hash] Error 1
gmake[1]: *** [CMakeFiles/Makefile2:3097: examples/gguf-hash/CMakeFiles/llama-gguf-hash.dir/all] Error 2
[ 59%] Linking CXX shared library libllama.so
[ 59%] Built target llama
gmake: *** [Makefile:146: all] Error 2
```

```
make --build gpupol --config Release -j 7
[  0%] Built target build_info
[  0%] Built target sha1
[  0%] Built target sha256
[  1%] Built target xxhash
[ 56%] Built target ggml
[ 56%] Linking CXX executable ../../bin/llama-gguf
[ 57%] Linking CXX executable ../../bin/llama-gguf-hash
[ 59%] Built target llama
/usr/bin/ld: ../../ggml/src/libggml.so: undefined reference to `x000fe200080f0eff'
collect2: error: ld returned 1 exit status
/usr/bin/ld: ../../ggml/src/libggml.so: undefined reference to `x000fe200080f0eff'
gmake[2]: *** [examples/gguf/CMakeFiles/llama-gguf.dir/build.make:103: bin/llama-gguf] Error 1
gmake[1]: *** [CMakeFiles/Makefile2:3260: examples/gguf/CMakeFiles/llama-gguf.dir/all] Error 2
gmake[1]: *** Waiting for unfinished jobs....
collect2: error: ld returned 1 exit status
gmake[2]: *** [examples/gguf-hash/CMakeFiles/llama-gguf-hash.dir/build.make:109: bin/llama-gguf-hash] Error 1
gmake[1]: *** [CMakeFiles/Makefile2:3097: examples/gguf-hash/CMakeFiles/llama-gguf-hash.dir/all] Error 2
[ 59%] Building CXX object examples/llava/CMakeFiles/llava.dir/clip.cpp.o
[ 59%] Building CXX object common/CMakeFiles/common.dir/common.cpp.o
[ 60%] Building CXX object examples/benchmark/CMakeFiles/llama-bench-matmult.dir/benchmark-matmult.cpp.o
[ 60%] Building C object tests/CMakeFiles/test-c.dir/test-c.c.o
[ 60%] Building CXX object common/CMakeFiles/common.dir/sampling.cpp.o
[ 61%] Building CXX object examples/quantize-stats/CMakeFiles/llama-quantize-stats.dir/quantize-stats.cpp.o
[ 61%] Building CXX object examples/llava/CMakeFiles/llava.dir/llava.cpp.o
[ 61%] Linking C executable ../bin/test-c
/usr/bin/ld: ../ggml/src/libggml.so: undefined reference to `x000fe200080f0eff'
collect2: error: ld returned 1 exit status
gmake[2]: *** [tests/CMakeFiles/test-c.dir/build.make:104: bin/test-c] Error 1
gmake[1]: *** [CMakeFiles/Makefile2:2713: tests/CMakeFiles/test-c.dir/all] Error 2
[ 61%] Building CXX object common/CMakeFiles/common.dir/console.cpp.o
[ 61%] Building CXX object common/CMakeFiles/common.dir/grammar-parser.cpp.o
[ 62%] Linking CXX executable ../../bin/llama-bench-matmult
/usr/bin/ld: ../../ggml/src/libggml.so: undefined reference to `x000fe200080f0eff'
collect2: error: ld returned 1 exit status
gmake[2]: *** [examples/benchmark/CMakeFiles/llama-bench-matmult.dir/build.make:106: bin/llama-bench-matmult] Error 1
gmake[1]: *** [CMakeFiles/Makefile2:2887: examples/benchmark/CMakeFiles/llama-bench-matmult.dir/all] Error 2
[ 62%] Building CXX object common/CMakeFiles/common.dir/json-schema-to-grammar.cpp.o
[ 63%] Building CXX object common/CMakeFiles/common.dir/train.cpp.o
[ 63%] Building CXX object common/CMakeFiles/common.dir/ngram-cache.cpp.o
[ 63%] Linking CXX executable ../../bin/llama-quantize-stats
/usr/bin/ld: ../../ggml/src/libggml.so: undefined reference to `x000fe200080f0eff'
collect2: error: ld returned 1 exit status
gmake[2]: *** [examples/quantize-stats/CMakeFiles/llama-quantize-stats.dir/build.make:106: bin/llama-quantize-stats] Error 1
gmake[1]: *** [CMakeFiles/Makefile2:3920: examples/quantize-stats/CMakeFiles/llama-quantize-stats.dir/all] Error 2
In file included from /run/media/pancho/4C4643C74643B10E/ChatIAs/ik_llama.cpp/examples/llava/clip.cpp:24:
/run/media/pancho/4C4643C74643B10E/ChatIAs/ik_llama.cpp/examples/llava/../../common/stb_image.h: In function ‘int stbi__parse_png_file(stbi__png*, int, int)’:
/run/media/pancho/4C4643C74643B10E/ChatIAs/ik_llama.cpp/examples/llava/../../common/stb_image.h:5450:31: warning: writing 1 byte into a region of size 0 [-Wstringop-overflow=]
 5450 |                         tc[k] = (stbi_uc)(stbi__get16be(s) & 255) *
      |                         ~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 5451 |                                 stbi__depth_scale_table[z->depth]; // non 8-bit images will be larger
      |                                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/run/media/pancho/4C4643C74643B10E/ChatIAs/ik_llama.cpp/examples/llava/../../common/stb_image.h:5326:28: note: at offset 3 into destination object ‘tc’ of size 3
 5326 |     stbi_uc has_trans = 0, tc[3] = {0};
      |                            ^~
[ 63%] Built target llava
[ 63%] Linking CXX static library libcommon.a
[ 63%] Built target common
gmake: *** [Makefile:146: all] Error 2
```

It seems CUDA parts worked fine.

I'm building with

```
  CC=gcc-14 CXX=g++-14 CUDAHOSTCXX=g++-14 cmake -B build \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DGGML_BLAS=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="86;89;120" \
    -DGGML_IQK_FA_ALL_QUANTS=1 \
    -DGGML_SCHED_MAX_COPIES=1 \
    -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler -ccbin=g++-14"

  cmake --build build --config Release -j 7
```

---

👤 **ikawrakow** commented the **2025-05-10** at **18:45:34**:<br>

Not sure. `grep` on the source tree for `000fe200080f0eff` returns no results.

---

👤 **Panchovix** commented the **2025-05-10** at **19:39:27**:<br>

Okay restarting didn't work either. But cloning the PR itself in a new folder worked, so I guess there is an issue with my main folder after pulling the PR separately.

Now testing the PR itself, it works! Running with

```
./llama-server -m '/GGUFs/DeepSeek-V3-0324-UD-Q2_K_XL-merged.gguf' -c 16384 --no-mmap -v -ngl 999 -ot "blk.(0|1|2|3|4|5|6|7).ffn.=CUDA0" -ot "blk.(8|9|10|11).ffn.=CUDA1" -ot "blk.(12|13|14|15|16).ffn.=CUDA2" -ot "blk.(17|18|19|20|21|22|23|24|25|26).ffn.=CUDA3" -ot "ffn.*=CPU" -fa -mg 0 -ub 1024 -fmoe
```

Speeds are

```
INFO [           print_timings] prompt eval time     =   32736.15 ms /  3596 tokens (    9.10 ms per token,   109.85 tokens per second) | tid="140176171094016" timestamp=1746905794 id_slot=0 id_task=0 t_prompt_processing=32736.147 n_prompt_tokens_processed=3596 t_token=9.103489154616241 n_tokens_second=109.84799157946107
INFO [           print_timings] generation eval time =   57112.32 ms /   454 runs   (  125.80 ms per token,     7.95 tokens per second) | tid="140176171094016" timestamp=1746905794 id_slot=0 id_task=0 t_token_generation=57112.318 n_decoded=454 t_token=125.79805726872246 n_tokens_second=7.94924835654543
INFO [           print_timings]           total time =   89848.46 ms | tid="140176171094016" timestamp=1746905794 id_slot=0 id_task=0 t_prompt_processing=32736.147 t_token_generation=57112.318 t_total=89848.465
```

This is about 10% faster than main llamacpp with the same ubatch size, and GPU 0 running at X8 5.0 saturates at the absolute limit (28-29 GiB/s, 1-2GiB/s higher vs main llamacpp), so maybe there could be a benefit on X16 5.0, but that is yet to test.

---

👤 **Panchovix** commented the **2025-05-10** at **23:37:03**:<br>

Just an update, tested other deepseek models (v30324, chimera, r1) at q2_k_xl, iq3_xxs, q3_k_s and q3_k_xl, all working fine! So really nice work.

---

👤 **ikawrakow** commented the **2025-05-11** at **04:42:09**:<br>

Thanks for testing, I appreciate it!

Johannes has improved the performance `llama.cpp` for MoE models quite a bit in the last few weeks, so the performance differential is no longer so big as it used to be. But for larger batches (e.g., `-b 4096 -ub 4096`) and long prompts it is still quite significant. For example, with DeepSeek-Lite and a prompt of 65k tokens `ik_llama.cpp` is about 2X faster than `llama.cpp` for PP, and about 15% faster for TG.

---

👤 **Panchovix** commented the **2025-05-11** at **04:52:17**:<br>

I see! I think I would have to remove some layers from some experts from GPU to use -b and -ub 4096, which I think it would increase PP but maybe decrease TG a bit? At least I have noticed that with -b 2560 and -ub 2048 with less layers on GPU but more ctx (128K)

---

👤 **ikawrakow** commented the **2025-05-11** at **04:59:57**:<br>

> I think I would have to remove some layers from some experts from GPU to use -b and -ub 4096, which I think it would increase PP but maybe decrease TG a bit? 

Yes, so it depends what is more important to you. TG performance decrease will be quite modest, about 1/61 per extra not offloaded layer for DeepSeek-R1/V3.

> At least I have noticed that with -b 2560 and -ub 2048

What is the use case for `-b 2560 -ub 2048`? The computation will run one u-batch of 2048 and then another one of 512. I think it is always better to use a batch size that is a multiple of the u-batch size, so I would have used `-b 2048 -ub 2048`.

---

👤 **Panchovix** commented the **2025-05-11** at **05:12:45**:<br>

> > I think I would have to remove some layers from some experts from GPU to use -b and -ub 4096, which I think it would increase PP but maybe decrease TG a bit?
> 
> Yes, so it depends what is more important to you. TG performance decrease will be quite modest, about 1/61 per extra not offloaded layer for DeepSeek-R1/V3.
> 
> > At least I have noticed that with -b 2560 and -ub 2048
> 
> What is the use case for `-b 2560 -ub 2048`? The computation will run one u-batch of 2048 and then another one of 512. I think it is always better to use a batch size that is a multiple of the u-batch size, so I would have used `-b 2048 -ub 2048`.

Oh just when I was testing on main llamacpp, I had more memory usage with -b and -ub 2048 than 2560/2048 respectively, but maybe it was because something else.

Also just 1/61 the speed, pretty worth probably. I get 7 t/s on Q3_K_XL TG but ~80-90 t/s PP. I would trade 2 layers for ~6.3 t/s for more PP speed.

---

👤 **Panchovix** commented the **2025-05-11** at **22:34:17**:<br>

Okay testing Q2_K_XL with -b 4096 and -ub 4096, PP t/s are insane

```
INFO [           print_timings] prompt eval time     =   13435.86 ms /  3003 tokens (    4.47 ms per token,   223.51 tokens per second) | tid="140099605647360" timestamp=1747002757 id_slot=0 id_task=385 t_prompt_processing=13435.857 n_prompt_tokens_processed=3003 t_token=4.474144855144855 n_tokens_second=223.50639784272786
```

---

👤 **cosystudio** commented the **2025-05-12** at **21:52:32**:<br>

I want to say thank you as well as provide a datapoint. PP hit 301 tk/s vs about 230 tk/s vs commit ab7f694b. x2 3090 AMD Epyc 9654P + 12 channels of DDR5 4800 MT/s ram

./llama-server  --alias /Qwen3-235B-A22B-128K-UD-Q4_K_XL -m /home/dev/models/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-128K-UD-Q4_K_XL-00001-of-00003.gguf -c 92160 -t 96 -fa -amb 512 -mla 3 -rtr -fmoe -ctk q8_0  -ctv q8_0 --parallel 1 -ngl 99 -ot "blk\.(0|1|2|3|4|5|6|14|15|16)\.ffn.*=CUDA0" -ot "blk\.(7|8|9|10|11|12|13|17|18|19)\.ffn.*=CUDA1" -ot "blk\.2[0-9]\.ffn.*=CPU" -ot "blk\.[3-9][0-9]\.ffn.*=CPU" --host 0.0.0.0 --port 8080  --temp 0.6 --top-k 20 --top-p 0.95 --min-p 0  -np 8  -ub 1024 --metrics -dt 0.05 --threads-http 16 --prompt-cache-all --predict 38912 -b 4096 -ub 4096


INFO [           print_timings] prompt eval time     =   23946.86 ms /  7221 tokens (    3.32 ms per token,   301.54 tokens per second) | tid="130418296737792" timestamp=1747086263 id_slot=0 id_task=17 t_prompt_processing=23946.864 n_prompt_tokens_processed=7221 t_token=3.316280847528043 n_tokens_second=301.54261535038574
INFO [           print_timings] generation eval time =    3061.63 ms /    55 runs   (   55.67 ms per token,    17.96 tokens per second) | tid="130418296737792" timestamp=1747086263 id_slot=0 id_task=17 t_token_generation=3061.629 n_decoded=55 t_token=55.66598181818182 n_tokens_second=17.964292865007486
INFO [           print_timings]           total time =   27008.49 ms | tid="130418296737792" timestamp=1747086263 id_slot=0 id_task=17 t_prompt_processing=23946.864 t_token_generation=3061.629 t_total=27008.493000000002