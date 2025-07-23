### 🔀 [#386](https://github.com/ikawrakow/ik_llama.cpp/pull/386) - FlashMLA-3 for DeepSeek models on CUDA

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-06 |
| **Updated** | 2025-05-10 |

---

#### Description

[This PR](https://github.com/ggml-org/llama.cpp/pull/13306) in mainline `llama.cpp` is a CUDA flash attention (FA) implementation that also works with K head size of 576 and V head size of 512 as required for DeepSeek models with MLA enabled. **Caveat: it only works on Ampere or newer Nvidia GPUs**.

I have taken it and adapted it to the `ik_llama.cpp` environment, but only using it for the 576,512 case (for other head sizes it is slower than the existing implementation). This allows to finally have `-mla 3 -fa`, which is the recommended option for DeepSeek models when running on the CPU, also work on CUDA.

This results in massive DeepSeek TG performance gains for long contexts when self attention is computed on the GPU. The graph below shows an example for `Q4_0` quantized DeepSeek-Lite model that I can fully load on my RTX-4080 GPU with 16 GB VRAM. I have used u-batches of 4096 with `sweep-bench` to more quickly cover the context range of up to 65k tokens. The main branch results use `-mla 2 -fa`, the PR uses `-mla 3 -fa`. No FA (which is what happens for TG when `mla = 2`) is slightly faster with zero context. I have considered special-casing `N_KV <= 256` , but than decided against it. Less than 256 tokens in the KV cache has no relevance in actual usage other than bragging about the great performance one got on Reddit and elsewhere after saying 'Hello' to the LLM and checking the performance stats. At 60k tokens the PR is 3X faster than the main branch for TG!

I'm not including a comparison to the mainline PR as it has not been merged, so things can change (but for the curious, mainline TG is slightly faster for small `N_KV` and slower for large `N_KV`, PP continues to be far behind `ik_llama.cpp`). 
   
![dsl_cuda_mla3](https://github.com/user-attachments/assets/11c50e9a-813a-4384-b6e5-e3696ea772f9)

<details>
<summary>Main branch, mla = 2, fa</summary>

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |    0.423 |  9674.04 |    6.048 |   169.30 |
|  4096 |   1024 |   4096 |    0.535 |  7652.53 |    7.336 |   139.58 |
|  4096 |   1024 |   8192 |    0.647 |  6333.36 |    8.951 |   114.40 |
|  4096 |   1024 |  12288 |    0.758 |  5405.78 |   10.380 |    98.65 |
|  4096 |   1024 |  16384 |    0.873 |  4693.14 |   11.818 |    86.65 |
|  4096 |   1024 |  20480 |    0.988 |  4145.95 |   13.356 |    76.67 |
|  4096 |   1024 |  24576 |    1.099 |  3725.52 |   14.959 |    68.45 |
|  4096 |   1024 |  28672 |    1.222 |  3351.10 |   16.969 |    60.35 |
|  4096 |   1024 |  32768 |    1.357 |  3018.24 |   19.110 |    53.58 |
|  4096 |   1024 |  36864 |    1.453 |  2818.23 |   21.476 |    47.68 |
|  4096 |   1024 |  40960 |    1.583 |  2587.84 |   23.564 |    43.46 |
|  4096 |   1024 |  45056 |    1.695 |  2416.39 |   25.841 |    39.63 |
|  4096 |   1024 |  49152 |    1.836 |  2230.87 |   27.698 |    36.97 |
|  4096 |   1024 |  53248 |    1.942 |  2109.35 |   29.606 |    34.59 |
|  4096 |   1024 |  57344 |    2.044 |  2004.15 |   31.450 |    32.56 |
|  4096 |   1024 |  61440 |    2.163 |  1893.91 |   33.598 |    30.48 |
</details>

<details>
<summary>PR, mla = 3, fa</summary>

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |    0.447 |  9165.46 |    6.428 |   159.31 |
|  4096 |   1024 |   4096 |    0.535 |  7663.18 |    6.763 |   151.42 |
|  4096 |   1024 |   8192 |    0.646 |  6342.94 |    6.962 |   147.08 |
|  4096 |   1024 |  12288 |    0.760 |  5388.45 |    7.253 |   141.19 |
|  4096 |   1024 |  16384 |    0.877 |  4669.16 |    7.557 |   135.51 |
|  4096 |   1024 |  20480 |    0.991 |  4131.89 |    7.882 |   129.92 |
|  4096 |   1024 |  24576 |    1.108 |  3696.64 |    8.244 |   124.22 |
|  4096 |   1024 |  28672 |    1.226 |  3339.64 |    8.655 |   118.31 |
|  4096 |   1024 |  32768 |    1.344 |  3047.53 |    9.046 |   113.20 |
|  4096 |   1024 |  36864 |    1.457 |  2812.08 |    9.423 |   108.66 |
|  4096 |   1024 |  40960 |    1.575 |  2601.20 |   10.377 |    98.68 |
|  4096 |   1024 |  45056 |    1.691 |  2421.84 |   10.453 |    97.96 |
|  4096 |   1024 |  49152 |    1.807 |  2266.31 |   10.545 |    97.11 |
|  4096 |   1024 |  53248 |    1.923 |  2129.68 |   10.620 |    96.42 |
|  4096 |   1024 |  57344 |    2.044 |  2004.06 |   10.730 |    95.43 |
|  4096 |   1024 |  61440 |    2.158 |  1897.89 |   10.944 |    93.57 |

</details>

The PR also adds a tweak to matrix-vector multiplications that leads to minor TG performance gains for MoE models other than DeepSeek. As an example, the next graph shows TG performance for `IQ2_XS` quantized Qwen3-30B-A3B (so it fully loads on my 16 GB GPU) using  `-fmoe -fa -ub 2048`.

![q3_cuda](https://github.com/user-attachments/assets/82011208-ef05-4a98-bb48-c1b72964696b)

Testing with DeepSeek-V3/R1 will be greatly appreciated. Very few can run these models fully offloaded to the GPU, but I do expect non-negligible performance gains for long context also for hybrid GPU/CPU inference (where self attention is computed on the GPU). Checking that it works correctly is of course most important.

---

#### 💬 Conversation

👤 **infy-infy** commented the **2025-05-07** at **14:31:37**:<br>

Will `-mla 3 -fa` work in mixed cpu+multigpu setup with Amperes and Pascals? Or it would be better to continue use `-mla 2 -fa`? I mean, maybe `-mla 3 -fa` will use some fallback for old cards and it would still be better than `-mla 2 -fa`

---

👤 **ikawrakow** commented the **2025-05-07** at **14:36:26**:<br>

> Will -mla 3 -fa work in mixed cpu+multigpu setup with Amperes and Pascals?

If all attention calculations are done on the Ampere cards, it could work.
There is no fallback, and I'm not sure if I have put enough checks to prevent eventually hitting an assert if it attempts to run the 576,512 head size combination on a Pascal card.

---

👤 **Ph0rk0z** commented the **2025-05-07** at **18:38:22**:<br>

MLA 3 has faster sweep bench speeds for me but unfortunately deepseek 2.5 goes aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa

MLA 2 works.

---

👤 **ubergarm** commented the **2025-05-08** at **02:09:57**:<br>

I gave this a very quick try though the model doesn't fit in VRAM+RAM so pulls almost 6GB/s paging of a Gen5 PCIe NVME drive. This is a 3090TI FE 24GB VRAM GPU.

## tl;dr;
Something seems off with the response using `-mla 3` but still works how I'd expect for `-mla 2`. I didn't do the sweep's far enough at as it takes too long on this rig.

## Details
```
git rev-parse --short HEAD
4084ca
# i merged the batch warmup PR too and recompiled for CUDA like normal

./build/bin/llama-server \
    --model /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-IQ2_K_R4/DeepSeek-V3-0324-IQ2_K_R4-bartowski-imat.gguf \
    --alias ubergarm/DeepSeek-V3-0324-IQ2_K_R4 \
    --ctx-size 32768 \
    -ctk q8_0 \
    -mla 2 -fa \
    -amb 512 \
    -fmoe \
    -ser 6,1 \
    --n-gpu-layers 63 \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 16 \
    --host 127.0.0.1 \
    --port 8080
```

## `-mla 2`
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |   25.314 |    20.23 |   30.434 |     4.21 |
|   512 |    128 |    512 |   29.349 |    17.45 |   13.466 |     9.51 |
|   512 |    128 |   1024 |   29.555 |    17.32 |   13.596 |     9.41 |
|   512 |    128 |   1536 |   29.643 |    17.27 |   13.747 |     9.31 |
|   512 |    128 |   2048 |   29.202 |    17.53 |   13.819 |     9.26 |

```
>>> User:

Count from 1 to 10 in French.

>>> Assistant:

Here’s how to count from 1 to 10 in French:

1. **Un** (uhn)
2. **Deux** (du^C
```

## `-mla 3`
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |   26.291 |    19.47 |   32.849 |     3.90 |
|   512 |    128 |    512 |   29.949 |    17.10 |   13.085 |     9.78 |
|   512 |    128 |   1024 |   30.523 |    16.77 |   13.026 |     9.83 |
|   512 |    128 |   1536 |   29.763 |    17.20 |   13.095 |     9.77 |
|   512 |    128 |   2048 |   30.382 |    16.85 |   13.171 |     9.72 |

```
>>> User:

Count from 1 to 10 in French.

>>> Assistant:

Here are the numbers from 1 to 10 in     Please see the 1, 2, 3, 2, 8, 7, 3, 6, 1, 8, 3,   
We can see that the string is 8, 8, 2, , 0, 0, 0, 0, 8, 7, 1, 1, 0,0, 0, ^C
```

---

👤 **ikawrakow** commented the **2025-05-08** at **04:34:09**:<br>

OK, thanks for testing. Here is what I get with DeepSeek-Lite for @ubergarm's quantions:
```
<think>
Okay, so I need to count from 1 to 10 in French. Let me start by recalling what I know. I remember that French numbers are different from English, so I need to be careful. Let me think... I think "1" is "un" in French. Then "2" is "deux". Wait, is that right? I'm not sure. Maybe I should check. But since I can't look it up right now, I have to rely on my memory.

For "3", I think it's "trois". Yes, that sounds familiar. "4" is "quatre", right? Then "5" is "cinq". Hmm, I'm not sure about "cinq" but I think that's correct. "6" is "six", "7" is "sept", "8" is "huit", "9" is "neuf", and "10" is "dix". Wait, is that all? Let me go through each one again to make sure.

Starting over: 1 - un, 2 - deux, 3 - trois, 4 - quatre, 5 - cinq, 6 - six, 7 - sept, 8 - huit, 9 - neuf, 10 - dix. That seems right. I think that's correct. I don't remember any exceptions here. But maybe I should double-check the spelling. For example, "quatre" is spelled with a "c" and an "a", right? Yes, "quatre". "Cinq" is spelled with a "c" and "i" and "n" and "q"? Wait, no. "Cinq" is spelled c-i-n-q? Wait, no, it's c-i-n-q? Wait, no, "cinq" is c-i-n-q? Wait, no, it's c-i-n-q? Wait, no, it's "cinq" which is c-i-n-q. Wait, no, that's not right. Let me think. The word is "cinq", so c-i-n-q? Wait, no, it's spelled c-i-n-q? Wait, no, that's not correct. The correct spelling is c-i-n-q? No, it's "cinq" which is c-i-n-q? No, that's not right. Wait, maybe I'm confusing it with "cinq" which is spelled c-i-n-q. Wait, no, the correct spelling is c-i-n-q? No, "cinq" is spelled with a "c", then "i", "n", "q"? No, that's not right. Wait, "cinq" is actually spelled with a "c", "i", "n", "q"? No, that's not correct. Wait, "cinq" is spelled c-i-n-q? No, I think it's c-i-n-q. Wait, no. Let me think. The word "cinq" is spelled c-i-n-q? No, that's not correct. I think it's spelled c-i-n-q. Wait, I'm getting confused. Maybe I should just go with what I remember. I think "cinq" is correct. So I'll stick with that. 

Another thing: in French, numbers after 16 are sometimes different, but for 1-10, it's straightforward. So I think the list is correct. Let me just write them out again: 1 un, 2 deux, 3 trois, 4 quatre, 5 cinq, 6 six, 7 sept, 8 huit, 9 neuf, 10 dix. Yes, that seems right. I think that's accurate. I don't recall any mistakes here. So I'll present that as the answer.
</think>

Here is the count from 1 to 10 in French:

1. **un**  
2. **deux**  
3. **trois**  
4. **quatre**  
5. **cinq**  
6. **six**  
7. **sept**  
8. **huit**  
9. **neuf**  
10. **dix**  

Let me know if you'd like to practice further! 😊 
```

The difference is that Lite has 16 heads, while the big models have 128. So, I guess, something is not quite right with more than 16 heads.

---

👤 **ikawrakow** commented the **2025-05-08** at **07:29:24**:<br>

To be honest, I don't understand the failure.

**Recap `mla` options**
* `mla = 1`
   - Head sizes 576,512
   - This is what is done in mainline `llama.cpp`
   - FA did not work on CUDA prior to this PR and PR 13306 in mainline
* `mla = 2`
  - For prompt processing uses `attn_wkv_b` to convert the cache to head sizes 192,128 -> FA on CUDA works
  - For TG FA is disabled
* `mla = 3`
  - Prompt processing as in `mla = 2`
  - TG as `mla = 1`. FA on CUDA is possible after this PR
 
**Observations**
* According to @Panchovix who has tested mainline PR 13306 with a large DeepSeek model it works. Hence, the 576,512 kernel should be working
* According to @Ph0rk0z and @ubergarm `mla = 2` works. Hence we can conclude that the 192,128 kernel used for prompt processing also works
* When running CPU-only, `mla = 3, fa = 1` works. Hence, data handling and such should be fine.

So, based on observations, when we use 192,128 CUDA kernel for PP and 576,512 CUDA kernel for TG, it should be working. But it doesn't.

---

👤 **Ph0rk0z** commented the **2025-05-08** at **12:05:17**:<br>

How many heads does 2.5 have? Maybe there is some difference. It's easier to run and more like qwen in size. I will have to check the MLA 1 output, could be bug in FA. Also had some crash in MLA 2 after using it a while but haven't reproduced yet.

---

👤 **Ph0rk0z** commented the **2025-05-08** at **14:22:22**:<br>

Looks like my theory was correct. On my system MLA 1 also produces issues, probably as soon as FA kicks in. May start out coherent for the first bit of tokens and then descends intooooooooooooooooooosddkkkkkkkkasd

---

👤 **Panchovix** commented the **2025-05-08** at **14:39:38**:<br>

I can test on ikllamacpp in some hours if I can replicate on deepseek v3 0324 (I'm not home right now)

On main llamacpp I tested up to 64K CTX and it was working fine with the PR. If I understand correctly I have to use the latest quants and then use -mla 3 -fa? Main llamacpp uses -mla 2 -fa equivalent?

---

👤 **ikawrakow** commented the **2025-05-08** at **14:50:33**:<br>

> On main llamacpp I tested up to 64K CTX and it was working fine with the PR. If I understand correctly I have to use the latest quants and then use -mla 3 -fa? Main llamacpp uses -mla 2 -fa equivalent?

The mainline `llama.cpp` MLA implementation corresponds to `-mla 1` here. With this it wasn't possible to use flash attention on CUDA in the past, and it became possible with this PR and PR 13306 in mainline. If you use the latest quants that enable MLA in mainline, you require the not yet merged PR #394 that enables support for these incompatible models. Otherwise, you need to use an older model that does not allow MLA in mainline `llama.cpp`.

---

👤 **ikawrakow** commented the **2025-05-08** at **14:54:39**:<br>

> Looks like my theory was correct. On my system MLA 1 also produces issues, probably as soon as FA kicks in. May start out coherent for the first bit of tokens and then descends intooooooooooooooooooosddkkkkkkkkasd

`-mla 1 -fa` uses the same 576,512 CUDA kernel for prompt processing and for token generation. If the issue is with this kernel, then yes, `mla = 1` will also not work.

Then the conclusion would be that I introduced a bug when porting the mainline PR adding the 576,512 kernel. But it does work with DeepSeek-Lite, so I'm not sure how to debug.

---

👤 **ikawrakow** commented the **2025-05-08** at **15:11:44**:<br>

That would work as a test.

---

👤 **Panchovix** commented the **2025-05-08** at **18:19:57**:<br>

I just tried to load DeepSeek V3 Q2_K_XL but I get an issue on latest commit. This happens with both -mla 2 -fa and -mla 3 -fa. Not sure if I'm setting a parameter wrongly.

```
/llama-server --version
version: 3673 (4084ca73)
built with gcc-14 (GCC) 14.2.1 20250210 (Red Hat 14.2.1-8) for x86_64-redhat-linux
```

```
./llama-server -m '/models_llm/DeepSeek-V3-0324-UD-Q2_K_XL-00001-of-00006.gguf' -c 16384 --no-mmap --no-warmup -v -ngl 99 --override-tensor 'blk\.(2[5-9]|[3-6][0-9])\..*_exps\.=CPU' --override-tensor 'blk\.([1-6])\..*_exps\.=CUDA0' --override-tensor 'blk\.([7-9]|1[0])\..*_exps\.=CUDA1' --override-tensor 'blk\.(1[1-5])\..*_exps\.=CUDA2' --override-tensor 'blk\.(1[6-9]|2[0-4])\..*_exps\.=CUDA3' -fa -mla 3 -fmoe
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 4 CUDA devices:
  Device 0: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
  Device 1: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
  Device 2: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
  Device 3: NVIDIA RTX A6000, compute capability 8.6, VMM: yes
INFO [                    main] build info | tid="140686994853888" timestamp=1746728281 build=3673 commit="4084ca73"
INFO [                    main] system info | tid="140686994853888" timestamp=1746728281 n_threads=8 n_threads_batch=-1 total_threads=16 system_info="AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: additional 5 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 64 key-value pairs and 1086 tensors from /models_llm/DeepSeek-V3-0324-UD-Q2_K_XL-00001-of-00006.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Deepseek-V3-0324
llama_model_loader: - kv   3:                            general.version str              = V3-0324
llama_model_loader: - kv   4:                           general.basename str              = Deepseek-V3-0324
llama_model_loader: - kv   5:                       general.quantized_by str              = Unsloth
llama_model_loader: - kv   6:                         general.size_label str              = 256x20B
llama_model_loader: - kv   7:                            general.license str              = mit
llama_model_loader: - kv   8:                           general.repo_url str              = https://huggingface.co/unsloth
llama_model_loader: - kv   9:                   general.base_model.count u32              = 1
llama_model_loader: - kv  10:                  general.base_model.0.name str              = DeepSeek V3 0324
llama_model_loader: - kv  11:               general.base_model.0.version str              = V3-0324
llama_model_loader: - kv  12:          general.base_model.0.organization str              = Deepseek Ai
llama_model_loader: - kv  13:              general.base_model.0.repo_url str              = https://huggingface.co/deepseek-ai/De...
llama_model_loader: - kv  14:                               general.tags arr[str,4]       = ["deepseek_v3", "deepseek", "unsloth"...
llama_model_loader: - kv  15:                          general.languages arr[str,1]       = ["en"]
llama_model_loader: - kv  16:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv  17:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv  18:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv  19:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  20:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv  21:          deepseek2.attention.head_count_kv u32              = 1
llama_model_loader: - kv  22:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  23: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  24:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  25:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  26:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  27:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  28:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  29:             deepseek2.attention.key_length u32              = 576
llama_model_loader: - kv  30:           deepseek2.attention.value_length u32              = 512
llama_model_loader: - kv  31:         deepseek2.attention.key_length_mla u32              = 192
llama_model_loader: - kv  32:       deepseek2.attention.value_length_mla u32              = 128
llama_model_loader: - kv  33:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  34:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  35:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  36:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  37:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  38:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  39:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  40:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  41:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  42: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  43: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  44:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  45:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  46:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<�...
llama_model_loader: - kv  47:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  48:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
llama_model_loader: - kv  49:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  50:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  51:            tokenizer.ggml.padding_token_id u32              = 2
llama_model_loader: - kv  52:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  53:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  54:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  55:               general.quantization_version u32              = 2
llama_model_loader: - kv  56:                          general.file_type u32              = 10
llama_model_loader: - kv  57:                      quantize.imatrix.file str              = DeepSeek-V3-0324-GGUF/imatrix_unsloth...
llama_model_loader: - kv  58:                   quantize.imatrix.dataset str              = unsloth_calibration_DeepSeek-V3-0324.txt
llama_model_loader: - kv  59:             quantize.imatrix.entries_count i32              = 720
llama_model_loader: - kv  60:              quantize.imatrix.chunks_count i32              = 60
llama_model_loader: - kv  61:                                   split.no u16              = 0
llama_model_loader: - kv  62:                        split.tensors.count i32              = 1086
llama_model_loader: - kv  63:                                split.count u16              = 6
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  122 tensors
llama_model_loader: - type q2_K:  122 tensors
llama_model_loader: - type q3_K:   54 tensors
llama_model_loader: - type q4_K:  389 tensors
llama_model_loader: - type q5_K:   23 tensors
llama_model_loader: - type q6_K:   15 tensors
llm_load_vocab: special tokens cache size = 818
llm_load_vocab: token to piece cache size = 0.8223 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = deepseek2
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 129280
llm_load_print_meta: n_merges         = 127741
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 163840
llm_load_print_meta: n_embd           = 7168
llm_load_print_meta: n_layer          = 61
llm_load_print_meta: n_head           = 128
llm_load_print_meta: n_head_kv        = 1
llm_load_print_meta: n_rot            = 64
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 576
llm_load_print_meta: n_embd_head_v    = 512
llm_load_print_meta: n_gqa            = 128
llm_load_print_meta: n_embd_k_gqa     = 576
llm_load_print_meta: n_embd_v_gqa     = 512
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 18432
llm_load_print_meta: n_expert         = 256
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = yarn
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 0.025
llm_load_print_meta: n_ctx_orig_yarn  = 4096
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = 671B
llm_load_print_meta: model ftype      = Q2_K - Medium
llm_load_print_meta: model params     = 671.026 B
llm_load_print_meta: model size       = 233.180 GiB (2.985 BPW) 
llm_load_print_meta: repeating layers = 231.986 GiB (2.978 BPW, 669.173 B parameters)
llm_load_print_meta: general.name     = Deepseek-V3-0324
llm_load_print_meta: BOS token        = 0 '<｜begin▁of▁sentence｜>'
llm_load_print_meta: EOS token        = 1 '<｜end▁of▁sentence｜>'
llm_load_print_meta: PAD token        = 2 '<｜▁pad▁｜>'
llm_load_print_meta: LF token         = 131 'Ä'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_layer_dense_lead   = 3
llm_load_print_meta: n_lora_q             = 1536
llm_load_print_meta: n_lora_kv            = 512
llm_load_print_meta: n_ff_exp             = 2048
llm_load_print_meta: n_expert_shared      = 1
llm_load_print_meta: expert_weights_scale = 2.5
llm_load_print_meta: expert_weights_norm  = 1
llm_load_print_meta: expert_gating_func   = sigmoid
llm_load_print_meta: rope_yarn_log_mul    = 0.1000
llm_load_tensors: ggml ctx size =    2.23 MiB
llama_model_load: error loading model: check_tensor_dims: tensor 'blk.0.attn_q_b.weight' has wrong shape; expected  1536, 73728, got  1536, 24576,     1,     1
llama_load_model_from_file: failed to load model
llama_init_from_gpt_params: error: failed to load model '/run/media/pancho/5C8E54388E540D40/models_llm/DeepSeek-V3-0324-UD-Q2_K_XL-00001-of-00006.gguf'
 ERR [              load_model] unable to load model | tid="140686994853888" timestamp=1746728281 model="/run/media/pancho/5C8E54388E540D40/models_llm/DeepSeek-V3-0324-UD-Q2_K_XL-00001-of-00006.gguf"
Segmentation fault (core dumped)
```

---

👤 **ikawrakow** commented the **2025-05-08** at **19:08:29**:<br>

@Panchovix You are using a GGUF made for mainline llama.cpp MLA. As I wrote above, you need PR #394, which is an attempt to fix the incompatibility.

---

👤 **Panchovix** commented the **2025-05-08** at **19:09:38**:<br>

@ikawrakow ah I'm dumb, thanks! Haha gonna try the PR.

---

👤 **Ph0rk0z** commented the **2025-05-08** at **23:53:02**:<br>

Ok.. baby deepseek v2.0-chat, the ~16b one, right? Sort of inconclusive results.

MLA 1 - oooooooooooooooooooooooooo
MLA 2/3 crash with 8bit cache https://pastebin.com/0mkrcZwE

MLA 2/3 + FP16 cache do not exhibit too many issues from a quick test.

These quants are months and months old so I'm not sure if anything is wrong with them, I also used IQ4_XS

---

👤 **ikawrakow** commented the **2025-05-09** at **05:41:54**:<br>

I just tested [this model](https://huggingface.co/bartowski/DeepSeek-V2.5-1210-GGUF/tree/main/DeepSeek-V2.5-1210-IQ3_XXS), which is near the maximum size I can go. Seems to work perfectly fine with `fp16` KV cache:
```
./bin/llama-cli -m ./ds2.5/DeepSeek-V2.5-1210-IQ3_XXS-00001-of-00003.gguf -t 32 -ngl 100 -mla 3 -fa -c 32768 -s 1234 -ot exps=CPU  -cnv
```

<details>
<code>
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4080, compute capability 8.9, VMM: yes
Log start
main: build = 3673 (4084ca73)
main: built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu
main: seed  = 1234
llama_model_loader: additional 2 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 53 key-value pairs and 959 tensors from ./ds2.5/DeepSeek-V2.5-1210-IQ3_XXS-00001-of-00003.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = DeepSeek V2.5 1210
llama_model_loader: - kv   3:                            general.version str              = V2.5-1210
llama_model_loader: - kv   4:                           general.basename str              = DeepSeek
llama_model_loader: - kv   5:                         general.size_label str              = 160x14B
llama_model_loader: - kv   6:                            general.license str              = other
llama_model_loader: - kv   7:                       general.license.name str              = deepseek
llama_model_loader: - kv   8:                       general.license.link str              = https://github.com/deepseek-ai/DeepSe...
llama_model_loader: - kv   9:                      deepseek2.block_count u32              = 60
llama_model_loader: - kv  10:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv  11:                 deepseek2.embedding_length u32              = 5120
llama_model_loader: - kv  12:              deepseek2.feed_forward_length u32              = 12288
llama_model_loader: - kv  13:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv  14:          deepseek2.attention.head_count_kv u32              = 128
llama_model_loader: - kv  15:                   deepseek2.rope.freq_base f32              = 10000,000000
llama_model_loader: - kv  16: deepseek2.attention.layer_norm_rms_epsilon f32              = 0,000001
llama_model_loader: - kv  17:                deepseek2.expert_used_count u32              = 6
llama_model_loader: - kv  18:                          general.file_type u32              = 23
llama_model_loader: - kv  19:        deepseek2.leading_dense_block_count u32              = 1
llama_model_loader: - kv  20:                       deepseek2.vocab_size u32              = 102400
llama_model_loader: - kv  21:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  22:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  23:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  24:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  25:       deepseek2.expert_feed_forward_length u32              = 1536
llama_model_loader: - kv  26:                     deepseek2.expert_count u32              = 160
llama_model_loader: - kv  27:              deepseek2.expert_shared_count u32              = 2
llama_model_loader: - kv  28:             deepseek2.expert_weights_scale f32              = 16,000000
llama_model_loader: - kv  29:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  30:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  31:              deepseek2.rope.scaling.factor f32              = 40,000000
llama_model_loader: - kv  32: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  33: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0,100000
llama_model_loader: - kv  34:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  35:                         tokenizer.ggml.pre str              = deepseek-llm
llama_model_loader: - kv  36:                      tokenizer.ggml.tokens arr[str,102400]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  37:                  tokenizer.ggml.token_type arr[i32,102400]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  38:                      tokenizer.ggml.merges arr[str,99757]   = ["Ġ Ġ", "Ġ t", "Ġ a", "i n", "h e...
llama_model_loader: - kv  39:                tokenizer.ggml.bos_token_id u32              = 100000
llama_model_loader: - kv  40:                tokenizer.ggml.eos_token_id u32              = 100001
llama_model_loader: - kv  41:            tokenizer.ggml.padding_token_id u32              = 100001
llama_model_loader: - kv  42:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  43:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  44:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  45:               general.quantization_version u32              = 2
llama_model_loader: - kv  46:                      quantize.imatrix.file str              = /models_out/DeepSeek-V2.5-1210-GGUF/D...
llama_model_loader: - kv  47:                   quantize.imatrix.dataset str              = /training_dir/calibration_datav3.txt
llama_model_loader: - kv  48:             quantize.imatrix.entries_count i32              = 716
llama_model_loader: - kv  49:              quantize.imatrix.chunks_count i32              = 139
llama_model_loader: - kv  50:                                   split.no u16              = 0
llama_model_loader: - kv  51:                                split.count u16              = 3
llama_model_loader: - kv  52:                        split.tensors.count i32              = 959
llama_model_loader: - type  f32:  300 tensors
llama_model_loader: - type q5_K:    1 tensors
llama_model_loader: - type iq3_xxs:  597 tensors
llama_model_loader: - type iq3_s:   61 tensors
llm_load_vocab: special tokens cache size = 18
llm_load_vocab: token to piece cache size = 0,6411 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = deepseek2
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 102400
llm_load_print_meta: n_merges         = 99757
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 163840
llm_load_print_meta: n_embd           = 5120
llm_load_print_meta: n_layer          = 60
llm_load_print_meta: n_head           = 128
llm_load_print_meta: n_head_kv        = 128
llm_load_print_meta: n_rot            = 64
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 192
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: n_embd_k_gqa     = 24576
llm_load_print_meta: n_embd_v_gqa     = 16384
llm_load_print_meta: f_norm_eps       = 0,0e+00
llm_load_print_meta: f_norm_rms_eps   = 1,0e-06
llm_load_print_meta: f_clamp_kqv      = 0,0e+00
llm_load_print_meta: f_max_alibi_bias = 0,0e+00
llm_load_print_meta: f_logit_scale    = 0,0e+00
llm_load_print_meta: n_ff             = 12288
llm_load_print_meta: n_expert         = 160
llm_load_print_meta: n_expert_used    = 6
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = yarn
llm_load_print_meta: freq_base_train  = 10000,0
llm_load_print_meta: freq_scale_train = 0,025
llm_load_print_meta: n_ctx_orig_yarn  = 4096
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = 236B
llm_load_print_meta: model ftype      = IQ3_XXS - 3.0625 bpw
llm_load_print_meta: model params     = 235,741 B
llm_load_print_meta: model size       = 84,604 GiB (3,083 BPW) 
llm_load_print_meta: repeating layers = 84,058 GiB (3,077 BPW, 234,693 B parameters)
llm_load_print_meta: general.name     = DeepSeek V2.5 1210
llm_load_print_meta: BOS token        = 100000 '<｜begin▁of▁sentence｜>'
llm_load_print_meta: EOS token        = 100001 '<｜end▁of▁sentence｜>'
llm_load_print_meta: PAD token        = 100001 '<｜end▁of▁sentence｜>'
llm_load_print_meta: LF token         = 126 'Ä'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_layer_dense_lead   = 1
llm_load_print_meta: n_lora_q             = 1536
llm_load_print_meta: n_lora_kv            = 512
llm_load_print_meta: n_ff_exp             = 1536
llm_load_print_meta: n_expert_shared      = 2
llm_load_print_meta: expert_weights_scale = 16,0
llm_load_print_meta: expert_weights_norm  = 0
llm_load_print_meta: expert_gating_func   = softmax
llm_load_print_meta: rope_yarn_log_mul    = 0,1000
llm_load_tensors: ggml ctx size =    0,80 MiB
Tensor blk.1.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.1.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.1.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.2.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.2.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.2.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 60 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 61/61 layers to GPU
llm_load_tensors:        CPU buffer size = 37343,30 MiB
llm_load_tensors:        CPU buffer size = 37866,68 MiB
llm_load_tensors:        CPU buffer size = 10656,64 MiB
llm_load_tensors:        CPU buffer size =   214,84 MiB
llm_load_tensors:      CUDA0 buffer size =  5109,97 MiB
....................................................................................................
============ llm_load_tensors: need to compute 60 wk_b tensors
Computed blk.0.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.1.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.2.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.3.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.4.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.5.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.6.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.7.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.8.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.9.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.10.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.11.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.12.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.13.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.14.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.15.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.16.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.17.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.18.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.19.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.20.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.21.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.22.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.23.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.24.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.25.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.26.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.27.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.28.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.29.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.30.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.31.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.32.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.33.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.34.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.35.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.36.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.37.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.38.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.39.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.40.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.41.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.42.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.43.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.44.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.45.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.46.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.47.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.48.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.49.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.50.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.51.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.52.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.53.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.54.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.55.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.56.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.57.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.58.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.59.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 3
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 0
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000,0
llama_new_context_with_model: freq_scale = 0,025
llama_kv_cache_init: layer 0: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 1: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 2: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 3: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 4: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 5: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 6: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 7: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 8: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 9: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 10: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 11: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 12: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 13: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 14: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 15: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 16: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 17: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 18: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 19: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 20: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 21: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 22: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 23: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 24: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 25: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 26: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 27: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 28: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 29: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 30: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 31: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 32: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 33: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 34: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 35: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 36: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 37: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 38: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 39: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 40: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 41: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 42: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 43: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 44: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 45: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 46: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 47: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 48: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 49: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 50: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 51: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 52: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 53: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 54: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 55: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 56: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 57: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 58: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 59: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init:      CUDA0 KV buffer size =  2160,00 MiB
llama_new_context_with_model: KV self size  = 2160,00 MiB, c^KV (f16): 2160,00 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0,39 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =  6346,00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =    74,01 MiB
llama_new_context_with_model: graph nodes  = 3290
llama_new_context_with_model: graph splits = 179
main: chat template example: You are a helpful assistant

<｜User｜>Hello<｜Assistant｜>Hi there<｜end▁of▁sentence｜><｜User｜>How are you?<｜Assistant｜>

system_info: n_threads = 32 / 64 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | 
main: interactive mode on.
sampling: 
	repeat_last_n = 64, repeat_penalty = 1,000, frequency_penalty = 0,000, presence_penalty = 0,000
	top_k = 40, tfs_z = 1,000, top_p = 0,950, min_p = 0,050, typical_p = 1,000, temp = 0,800
	mirostat = 0, mirostat_lr = 0,100, mirostat_ent = 5,000
sampling order: 
CFG -> Penalties -> top_k -> tfs_z -> typical_p -> top_p -> min_p -> temperature 
generate: n_ctx = 32768, n_batch = 2048, n_predict = -1, n_keep = 1


== Running in interactive mode. ==
 - Press Ctrl+C to interject at any time.
 - Press Return to return control to the AI.
 - To return control without starting a new line, end your input with '/'.
 - If you want to submit another line, end your input with '\'.

Hi there


> Count from 1 to 10 in French
Sure! Here's how you count from 1 to 10 in French:

1. Un
2. Deux
3. Trois
4. Quatre
5. Cinq
6. Six
7. Sept
8. Huit
9. Neuf
10. Dix

> Tell me everything you know about radiation therapy of cancer
Radiation therapy, also known as radiotherapy, is a medical treatment that uses high doses of radiation to kill cancer cells and shrink tumors. Here's an overview of everything you need to know about radiation therapy for cancer:

### Types of Radiation Therapy

1. **External Beam Radiation Therapy (EBRT)**
   - **Conventional Radiation Therapy**: Delivered in daily fractions over several weeks.
   - **Intensity-Modulated Radiation Therapy (IMRT)**: Uses advanced technology to deliver precise radiation doses to the tumor while minimizing damage to surrounding healthy tissue.
   - **3D Conformal Radiation Therapy**: Delivers radiation in 3D to match the shape of the tumor.
   - **Proton Therapy**: Uses protons instead of X-rays to deliver radiation, which can be more precise and reduce side effects.

2. **Internal Radiation Therapy (Brachytherapy)**
   - **Permanent Seed Implantation**: Radioactive seeds are placed directly into or near the tumor.
   - **High-Dose-Rate (HDR) Brachytherapy**: Temporary implants that deliver a high dose of radiation over a short period.

3. **Systemic Radiation Therapy**
   - Involves giving radioactive materials (such as radioactive iodine) that travel throughout the body to target cancer cells.

### Indications

Radiation therapy is used to treat a wide range of cancers, including:
- Brain tumors
- Breast cancer
- Cervical cancer
- Lung cancer
- Prostate cancer
- Lymphomas
- Head and neck cancers

### Purpose

- **Curative Intent**: To eliminate cancer cells and potentially cure the patient.
- **Palliative**: To relieve symptoms and improve quality of life by shrinking tumors that cause pain, pressure, or other issues.
- **Adjuvant**: Given after surgery to kill any remaining cancer cells.
- **Neoadjuvant**: Given before surgery to shrink tumors, making surgery easier and potentially reducing the need for extensive procedures.

### Side Effects

- **Acute Side Effects**: Temporary and usually resolve after treatment. Examples include skin reactions (like a sunburn), fatigue, and temporary hair loss.
- **Late Side Effects**: Can occur months or years after treatment. Examples include radiation fibrosis, second cancers, and damage to nearby organs.

### Planning and Simulation

- **CT Scan**: Used to create a detailed map of the tumor and surrounding tissues.
- **Simulation**: The patient is placed in the exact position they will be in during treatment to ensure accuracy.
- **Immobilization Devices**: Used to keep the patient still during treatment.
- **Planning**: A team of specialists, including radiation oncologists, medical physicists, and dosimetrists, plan the treatment to deliver the most effective dose with minimal side effects.

### Delivery

- **Linear Accelerator**: The machine used to deliver external beam radiation.
- **Treatment Sessions**: Typically delivered 5 days a week for several weeks, allowing normal cells time to recover.

### Advantages

- Highly effective for many types of cancer.
- Non-invasive (for external beam therapy).
- Can be combined with other treatments like surgery and chemotherapy.

### Disadvantages

- Can cause side effects, both acute and long-term.
- May not be suitable for all types of cancer.
- Requires a team of specialists and specialized equipment.

### Future Directions

- **Imaging Technologies**: Advances in imaging are improving the accuracy of targeting tumors.
- **Immunotherapy Integration**: Combining radiation with immunotherapy to enhance the body's immune response against cancer cells.
- **Precision Medicine**: Personalized treatment plans based on genetic and molecular characteristics of the tumor.

### Conclusion

Radiation therapy is a cornerstone of cancer treatment, offering effective ways to target and destroy cancer cells while minimizing damage to healthy tissue. Its continued evolution through technological advancements and research promises even more effective and less toxic treatments in the future.

> How many r's are there in strawberry
The word "strawberry" contains **3 r's**.

> Can you solve the following system of equations for me? x + y = 2 x - y = 3
Certainly! Let's solve the system of equations step by step.

We have:
1. \( x + y = 2 \)
2. \( x - y = 3 \)

### Step 1: Add the two equations to eliminate \( y \)
\[
(x + y) + (x - y) = 2 + 3
\]
\[
2x = 5
\]

### Step 2: Solve for \( x \)
\[
x = \frac{5}{2}
\]

### Step 3: Substitute \( x = \frac{5}{2} \) into one of the original equations to find \( y \)
Using the first equation:
\[
\frac{5}{2} + y = 2
\]
\[
y = 2 - \frac{5}{2}
\]
\[
y = \frac{4}{2} - \frac{5}{2}
\]
\[
y = -\frac{1}{2}
\]

### Final Answer:
\[
x = \frac{5}{2}, \quad y = -\frac{1}{2}
\]

> Can you explain the German electoral system?
Certainly! The German electoral system is a mixed-member proportional representation (MMP) system, which combines elements of both direct representation and proportional representation. Here's a detailed explanation:

### Overview

1. **Federal Elections in Germany**:
   - Elect the members of the Bundestag (Federal Parliament).
   - There are two types of seats in the Bundestag: direct mandates (first-past-the-post) and party list seats (proportional representation).

2. **Direct Mandates**:
   - Voters cast two votes: one for a candidate in their constituency (Wahlkreis) and one for a party list.
   - The first vote determines the direct mandate. The candidate with the most votes in each constituency wins a seat directly.

3. **Party List Seats**:
   - The second vote determines the overall share of seats each party gets in the Bundestag.
   - If a party wins more direct mandates than its share of the second votes would allow, it keeps those extra seats (overhang seats).

### Key Features

1. **Majority Bonus (Überhangmandate)**:
   - If a party wins more direct mandates than its share of the party list votes, it keeps those extra seats. This can lead to more seats than the total number of second votes.

2. **Compensatory Seats**:
   - To ensure proportionality, compensatory seats are added to the Bundestag to balance the overhang seats.

3. **Threshold for Representation**:
   - A party must receive at least 5% of the second votes or win at least three direct mandates to enter the Bundestag. This rule prevents very small parties from gaining representation.

### Election Process

1. **Constituency Candidates**:
   - Each voter casts a vote for a candidate in their constituency. The candidate with the most votes wins a direct mandate.

2. **Party Lists**:
   - Each party prepares a list of candidates for the entire country. Voters cast a second vote for a party list.

3. **Seat Allocation**:
   - After the votes are counted, the Bundestag determines the total number of seats each party gets based on the second votes.
   - Direct mandates are allocated first. If a party has more direct mandates than its share of the second votes, it keeps those extra seats (overhang seats).
   - Compensatory seats are then added to ensure proportional representation.

### Example

Suppose there are 100 seats in the Bundestag and the following results:
- Party A wins 40 direct mandates and receives 35% of the second votes.
- Party B wins 25 direct mandates and receives 30% of the second votes.
- Party C wins 10 direct mandates and receives 15% of the second votes.
- Party D wins 5 direct mandates and receives 20% of the second votes.

After allocating direct mandates, Party A has 40 seats, Party B has 25 seats, Party C has 10 seats, and Party D has 5 seats. However, Party A should only have 35 seats based on its share of the second votes. To compensate, compensatory seats are added, bringing the total number of seats to 100, ensuring proportional representation.

### Conclusion

The German electoral system ensures a balance between the direct representation of candidates and the proportional representation of parties. It allows for a more representative parliament while maintaining a connection between voters and their representatives at the local level.

llama_print_timings:        load time =   11491,65 ms
llama_print_timings:      sample time =      71,37 ms /  1965 runs   (    0,04 ms per token, 27532,19 tokens per second)
llama_print_timings: prompt eval time =   16115,80 ms /    85 tokens (  189,60 ms per token,     5,27 tokens per second)
llama_print_timings:        eval time =  155304,51 ms /  1960 runs   (   79,24 ms per token,    12,62 tokens per second)
llama_print_timings:       total time =  326345,55 ms /  2045 tokens
</ code>
</details>

But yes, `q8_0` KV cache is broken. I'll investigate.

---

👤 **ikawrakow** commented the **2025-05-09** at **07:05:34**:<br>

OK, PR #400 should fix quantized KV cache.

---

👤 **ubergarm** commented the **2025-05-09** at **16:11:48**:<br>

> OK, PR #400 should fix quantized KV cache.

Yes this seems to work in my quick testing of big DeepSeek-R1-IQ2_K_R4 hybrid CPU+GPU on my local rig for both `-mla 2` and `-mla 3` e.g.
```
      -ctk q8_0 \
      -mla 3 -fa \
      -amb 512 \
      -fmoe \
      --n-gpu-layers 63 \
      --override-tensor exps=CPU \
```

However, I noticed for both `-mla 2` and `-mla 3` in combination with `-ser 6,1`, it seems to work okay for short prompts like `Count from 1 to 10 in French`, but for longer ~600 token prompts it will throw `DDDDDDDD` again. Not a priority, I only use `-ser` if I'm desperate and can't access a remote rig!

Thanks for working through all the combinations!

---

👤 **ikawrakow** commented the **2025-05-09** at **16:19:29**:<br>

Thanks for testing.

I'm not sure if the `DDDDDD` is an actual bug. It is a low bit quantization, and then on top 6 instead of 8 experts. I would be worried that it is a bug if you hadn't used `-ser`.

---

👤 **saood06** commented the **2025-05-09** at **19:28:45**:<br>

> However, I noticed for both `-mla 2` and `-mla 3` in combination with `-ser 6,1`, it seems to work okay for short prompts like `Count from 1 to 10 in French`, but for longer ~600 token prompts it will throw `DDDDDDDD` again. Not a priority, I only use `-ser` if I'm desperate and can't access a remote rig!

I've never gotten `-ser` to work for me when loading a long context session (but I haven't really tried it in any other situation). I've never opened an issue as I've never taken the time to produce a minimally reproducible example.

---

👤 **ikawrakow** commented the **2025-05-10** at **09:13:51**:<br>

> > However, I noticed for both `-mla 2` and `-mla 3` in combination with `-ser 6,1`, it seems to work okay for short prompts like `Count from 1 to 10 in French`, but for longer ~600 token prompts it will throw `DDDDDDDD` again. Not a priority, I only use `-ser` if I'm desperate and can't access a remote rig!
> 
> I've never gotten `-ser` to work for me when loading a long context session (but I haven't really tried it in any other situation). I've never opened an issue as I've never taken the time to produce a minimally reproducible example.

SER should hopefully work correctly now, see PR #404

---

👤 **ubergarm** commented the **2025-05-10** at **16:19:20**:<br>

> > > However, I noticed for both `-mla 2` and `-mla 3` in combination with `-ser 6,1`, it seems to work okay for short prompts like `Count from 1 to 10 in French`, but for longer ~600 token prompts it will throw `DDDDDDDD` again. Not a priority, I only use `-ser` if I'm desperate and can't access a remote rig!
> > 
> > 
> > I've never gotten `-ser` to work for me when loading a long context session (but I haven't really tried it in any other situation). I've never opened an issue as I've never taken the time to produce a minimally reproducible example.
> 
> SER should hopefully work correctly now, see PR #404

I just tried out PR404 which is now `main@a2d24c97`, but still seeing it reply `DDDDD` for longer contexts when using `-ser 6,1`.

Also got a segfault when hitting `control+c` on my client and canceling which may give a clue if something is up:

<details>

<summary>👈 Full command logs</summary>

```
model=/mnt/ai/models/ubergarm/DeepSeek-R1-GGUF/DeepSeek-R1-GGUF-IQ2_K_R4.gguf
name=ubergarm/DeepSeek-R1-IQ2_K_R4
./build/bin/llama-server \
    --model "$model" \
    --alias "$name" \
    --ctx-size 32768 \
    -ctk q8_0 \
    -mla 3 -fa \
    -amb 512 \
    -fmoe \
    -ser 6,1 \
    --n-gpu-layers 63 \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 16 \
    --host 127.0.0.1 \
    --port 8080

ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090 Ti, compute capability 8.6, VMM: yes
INFO [                    main] build info | tid="134627362045952" timestamp=1746893205 build=3680 commit="a2d24c97"
INFO [                    main] system info | tid="134627362045952" timestamp=1746893205 n_threads=16 n_threads_batch=-1 total_threads=32 system_info="AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
.
.
.
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  612 tensors
llama_model_loader: - type q2_k_r4:  116 tensors
llama_model_loader: - type q3_k_r4:   58 tensors
.
.
.
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size = 241396.85 MiB
llm_load_tensors:        CPU buffer size =   938.98 MiB
llm_load_tensors:      CUDA0 buffer size = 17744.02 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 3
llama_new_context_with_model: attn_max_b = 512
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = 6, 1
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init:      CUDA0 KV buffer size =  1166.65 MiB
llama_new_context_with_model: KV self size  = 1166.62 MiB, c^KV (q8_0): 1166.62 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.99 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =  3425.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   176.01 MiB
llama_new_context_with_model: graph nodes  = 8245
llama_new_context_with_model: graph splits = 118
.
.
.
INFO [            update_slots] kv cache rm [p0, end) | tid="134627362045952" timestamp=1746893303 id_slot=0 id_task=18 p0=451

CUDA error: an illegal memory access was encountered
  current device: 0, in function ggml_backend_cuda_synchronize at /mnt/astrodata/llm/ik_llama.cpp/ggml/src/ggml-cuda.cu:3049
  cudaStreamSynchronize(cuda_ctx->stream())
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/ggml-cuda.cu:110: CUDA error
ptrace: Operation not permitted.
No stack.
The program is not being run.
```

<details>

It could be this model is too small, all attention layers are Q8_0` for GPU, and for CPU ffn_down is IQ3_K_R4, ffn_(gate|up) are IQ2_K_R4.

Still works okay without `-ser 6,1`. I also tried removing `-fa` when testing ser and also threw DDDD.

---

👤 **Ph0rk0z** commented the **2025-05-10** at **16:43:00**:<br>

Deepseek 2.5 seems to work with q_8, tg/pp is slightly faster than F16. Unfortunately sometimes a GPU gets stuck at 100% in task manager and the bench or server halts then sits. GPU power draw not consistent with 100% usage of course. It could be due to my undervolting or something else? F16 sweep completes successfully and is definitely "heavier" on resources so I'm not sure anymore.

---

👤 **ikawrakow** commented the **2025-05-10** at **18:26:05**:<br>

> Still works okay without -ser 6,1. I also tried removing -fa when testing ser and also threw DDDD.

OK, thanks. The PR fixes things for me, but it seems there is still a bug lurking somewhere. I'll keep looking.

---

👤 **ikawrakow** commented the **2025-05-10** at **18:31:16**:<br>

> Unfortunately sometimes a GPU gets stuck at 100% in task manager and the bench or server halts then sits.

There have been reports about problems with FA also in mainline. As I took the DeepSeek implementation from there, I guess `ik_llama.cpp` has the same issues. Your observation of the calculation being stuck indicates a synchronization problem, likely with the async copies that are now being used in the middle of the kernel.

---

👤 **Ph0rk0z** commented the **2025-05-10** at **21:14:52**:<br>

Now that you mention it, that's the kind of error I'd get on llama-server. It would eventually fail and segfault there with synchronization listed as the fault. I assumed it was due to me undervolting. Setting lower max gpu clock along with the clock offset (only way to do it on linux) caused it to happen less often. 

Perhaps it's only coincidental. Haven't yet tested EXL2 tensor parallel and it's much higher GPU load on the same settings. If it dumps on me again, I'll try to grab the error.