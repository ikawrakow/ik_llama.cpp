## 🔀 [Pull Request #405](https://github.com/ikawrakow/ik_llama.cpp/pull/405) - GPU offload policy

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/offload_policy` |
| **Target Branch** | `main` |
| **Created** | 2025-05-10 |
| **Updated** | 2025-05-12 |
| **Merged** | 2025-05-12 |

---

## 📄 Description

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

## 💬 Conversation

👤 **Panchovix** commented on **2025-05-10** at **18:12:44**

Many thanks for the PR! Sorry as I think I didn't understand correctly, for the case we were talking on https://github.com/ikawrakow/ik_llama.cpp/pull/394#issuecomment-2868723515, if we want to do the matrix multiplications on MoE models, we should specify

`-op 26,1,27,1` so the matrix multiplications are done on the GPU, or viceversa?

---

👤 **ikawrakow** commented on **2025-05-10** at **18:22:29**

This PR sets `ik_llama.cpp` GPU offload behavior to be the same as `llama.cpp`, so you don't need to use the `-op` argument. You would want to use it if you were running for instance Maverick, and then you would use `-op 27,0,29,0`.

---

👤 **Panchovix** commented on **2025-05-10** at **18:33:15**

Amazing, thanks! EDIT: Compilation solved by doing a new git clone.

---

👤 **ikawrakow** commented on **2025-05-10** at **18:45:34**

Not sure. `grep` on the source tree for `000fe200080f0eff` returns no results.

---

👤 **Panchovix** commented on **2025-05-10** at **19:39:27**

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

👤 **Panchovix** commented on **2025-05-10** at **23:37:03**

Just an update, tested other deepseek models (v30324, chimera, r1) at q2_k_xl, iq3_xxs, q3_k_s and q3_k_xl, all working fine! So really nice work.

---

👤 **ikawrakow** commented on **2025-05-11** at **04:42:09**

Thanks for testing, I appreciate it!

Johannes has improved the performance `llama.cpp` for MoE models quite a bit in the last few weeks, so the performance differential is no longer so big as it used to be. But for larger batches (e.g., `-b 4096 -ub 4096`) and long prompts it is still quite significant. For example, with DeepSeek-Lite and a prompt of 65k tokens `ik_llama.cpp` is about 2X faster than `llama.cpp` for PP, and about 15% faster for TG.

---

👤 **Panchovix** commented on **2025-05-11** at **04:52:17**

I see! I think I would have to remove some layers from some experts from GPU to use -b and -ub 4096, which I think it would increase PP but maybe decrease TG a bit? At least I have noticed that with -b 2560 and -ub 2048 with less layers on GPU but more ctx (128K)

---

👤 **ikawrakow** commented on **2025-05-11** at **04:59:57**

> I think I would have to remove some layers from some experts from GPU to use -b and -ub 4096, which I think it would increase PP but maybe decrease TG a bit? 

Yes, so it depends what is more important to you. TG performance decrease will be quite modest, about 1/61 per extra not offloaded layer for DeepSeek-R1/V3.

> At least I have noticed that with -b 2560 and -ub 2048

What is the use case for `-b 2560 -ub 2048`? The computation will run one u-batch of 2048 and then another one of 512. I think it is always better to use a batch size that is a multiple of the u-batch size, so I would have used `-b 2048 -ub 2048`.

---

👤 **Panchovix** commented on **2025-05-11** at **05:12:45**

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

👤 **Panchovix** commented on **2025-05-11** at **22:34:17**

Okay testing Q2_K_XL with -b 4096 and -ub 4096, PP t/s are insane

```
INFO [           print_timings] prompt eval time     =   13435.86 ms /  3003 tokens (    4.47 ms per token,   223.51 tokens per second) | tid="140099605647360" timestamp=1747002757 id_slot=0 id_task=385 t_prompt_processing=13435.857 n_prompt_tokens_processed=3003 t_token=4.474144855144855 n_tokens_second=223.50639784272786
```

EDIT: After some gens it just gets faster

```
INFO [           print_timings] prompt eval time     =   14636.06 ms /  3595 tokens (    4.07 ms per token,   245.63 tokens per second) | tid="140099605647360" timestamp=1747003592 id_slot=0 id_task=2032 t_prompt_processing=14636.063 n_prompt_tokens_processed=3595 t_token=4.071227538247566 n_tokens_second=245.62616326535354
```