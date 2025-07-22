### 🐛 [#60](https://github.com/ikawrakow/ik_llama.cpp/issues/60) - Bug: Illegal instruction on NEON and Q4_0_4_4

| **Author** | `whoreson` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-09-19 |
| **Updated** | 2024-09-19 |

---

#### Description

### What happened?

It crashes. Okay, it also happens with mainstream llama.cpp but georgi banned me for making too many CUDA 10 bugreports, so I'm just gonna leave this here in case it's interesting - close it if not. Model is gemma-2-2b-it-Q4_0_4_4.gguf

### Name and Version

latest, on a Redmi Note 7

### What operating system are you seeing the problem on?

Other? (Please let us know in description)

### Relevant log output

```shell
llm_load_print_meta: UNK token        = 3 '<unk>'
llm_load_print_meta: PAD token        = 0 '<pad>'
llm_load_print_meta: LF token         = 227 '<0x0A>'
llm_load_print_meta: EOT token        = 107 '<end_of_turn>'
llm_load_print_meta: max token length = 48
llm_load_tensors: ggml ctx size =    0.13 MiB
llm_load_tensors:        CPU buffer size =  1548.25 MiB
.........................................................
llama_new_context_with_model: n_ctx      = 2048
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:        CPU KV buffer size =   208.00 MiB
llama_new_context_with_model: KV self size  =  208.00 MiB, K (f16):  104.00 MiB,
 V (f16):  104.00 MiB
llama_new_context_with_model:        CPU  output buffer size =     0.98 MiB
llama_new_context_with_model:        CPU compute buffer size =  1004.50 MiB
llama_new_context_with_model: graph nodes  = 865
llama_new_context_with_model: graph splits = 1
[New Thread 0x1aa7 (LWP 6823)]
[New Thread 0x1aa8 (LWP 6824)]
[New Thread 0x1aa9 (LWP 6825)]
[New Thread 0x1aaa (LWP 6826)]
[New Thread 0x1aab (LWP 6827)]
[New Thread 0x1aac (LWP 6828)]
[New Thread 0x1aad (LWP 6829)]

Thread 7 "llama-cli" received signal SIGILL, Illegal instruction.
[Switching to Thread 0x1aac (LWP 6828)]
0x0000005555749b38 in ggml_gemv_q4_0_4x4_q8_0 (n=2304, s=0x7ef2b03300, 
    bs=2048, vx=0x7f7161e260, vy=0x7fb63f11c0, nr=1, nc=256)
    at ggml/src/ggml-aarch64.c:402
402         __asm__ __volatile__(
(gdb) bt
#0  0x0000005555749b38 in ggml_gemv_q4_0_4x4_q8_0 (n=2304, s=0x7ef2b03300, 
    bs=2048, vx=0x7f7161e260, vy=0x7fb63f11c0, nr=1, nc=256)
    at ggml/src/ggml-aarch64.c:402
#1  0x00000055556e53ec in ggml_compute_forward_mul_mat (params=0x7eea3e96e0, 
    dst=0x7fb584e180) at ggml/src/ggml.c:13214
#2  0x00000055556e304c in ggml_compute_forward (params=0x7eea3e96e0, 
    tensor=0x7fb584e180) at ggml/src/ggml.c:17880
#3  0x00000055556e2d0c in ggml_graph_compute_thread (data=0x7eea3e9758)
    at ggml/src/ggml.c:19961
#4  0x00000055556e2c18 in .omp_outlined._debug__ (.global_tid.=0x7eea3e97ec, 
    .bound_tid.=0x7eea3e97e8, n_threads=@0x7fffff949c: 8, state_shared=...)
    at ggml/src/ggml.c:20012
#5  0x00000055556e2df0 in .omp_outlined. (.global_tid.=0x7eea3e97ec, 
    .bound_tid.=0x7eea3e97e8, n_threads=@0x7fffff949c: 8, state_shared=...)
    at ggml/src/ggml.c:19998
#6  0x0000005555a4d6fc in __kmp_invoke_microtask ()
```

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2024-09-19** at **08:45:58**:<br>

I never use or check `Q4_0_4_4` or `Q4_0_8_8`. Also, I will definitely not try to debug several hundred lines of ARM assembly written by someone else - closing.