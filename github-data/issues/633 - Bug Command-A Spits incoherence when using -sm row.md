## 📌 [Issue #633](https://github.com/ikawrakow/ik_llama.cpp/issues/633) - Bug: Command-A Spits incoherence when using -sm row

| **Author** | `Ph0rk0z` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2025-07-20 |
| **Updated** | 2025-07-23 |
| **Labels** | `help wanted` |

---

## 📄 Description

### What happened?

Command-A speeds for IK are higher than mainline, especially since they broke something recently to knock it back from 15t/s to 12t/s. I still see 15s in YALS which uses an older commit. Prompt processing drops from 350 to 140, but I assume that's a facet of using SM row and can't be fixed. Mainline does it too.

Here, I get 17, almost 18t/s, unfortunately the result is as follows:

<img width="812" height="844" alt="Image" src="https://github.com/user-attachments/assets/0e77def1-6aa2-47f5-b687-d2f9f23e8be2" />


Is it KVcache related? SM row puts the cache all on GPU0. 

SM layer works correctly but T/G speeds suffer.

### Name and Version

Git latest.

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 ./bin/llama-server \
    -c 32768 \
    --host 192.168.1.211 \
    -ngl 99 \
    -ctk q8_0 \
    -ctv q8_0 \
    -fa \
    -sm row \
    -b 2048 \
    -ub 2048 
INFO [                    main] build info | tid="139717304852480" timestamp=1753018161 build=3829 commit="f1323339"
INFO [                    main] system info | tid="139717304852480" timestamp=1753018161 n_threads=48 n_threads_batch=-1 total_threads=96 system_info="AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: additional 1 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 47 key-value pairs and 514 tensors from Agatha-111B-v1-Q4_K_L/Agatha-111B-v1-Q4_K_L-00001-of-00002.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = cohere2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Agatha 111B v1
llama_model_loader: - kv   3:                            general.version str              = v1
llama_model_loader: - kv   4:                           general.basename str              = Agatha
llama_model_loader: - kv   5:                         general.size_label str              = 111B
llama_model_loader: - kv   6:                   general.base_model.count u32              = 1
llama_model_loader: - kv   7:                  general.base_model.0.name str              = C4Ai Command A 03 2025
llama_model_loader: - kv   8:               general.base_model.0.version str              = 03-2025
llama_model_loader: - kv   9:          general.base_model.0.organization str              = CohereLabs
llama_model_loader: - kv  10:              general.base_model.0.repo_url str              = https://huggingface.co/CohereLabs/c4a...
llama_model_loader: - kv  11:                        cohere2.block_count u32              = 64
llama_model_loader: - kv  12:                     cohere2.context_length u32              = 262144
llama_model_loader: - kv  13:                   cohere2.embedding_length u32              = 12288
llama_model_loader: - kv  14:                cohere2.feed_forward_length u32              = 36864
llama_model_loader: - kv  15:               cohere2.attention.head_count u32              = 96
llama_model_loader: - kv  16:            cohere2.attention.head_count_kv u32              = 8
llama_model_loader: - kv  17:                     cohere2.rope.freq_base f32              = 50000.000000
llama_model_loader: - kv  18:       cohere2.attention.layer_norm_epsilon f32              = 0.000010
llama_model_loader: - kv  19:               cohere2.attention.key_length u32              = 128
llama_model_loader: - kv  20:             cohere2.attention.value_length u32              = 128
llama_model_loader: - kv  21:                        cohere2.logit_scale f32              = 0.250000
llama_model_loader: - kv  22:           cohere2.attention.sliding_window u32              = 4096
llama_model_loader: - kv  23:                         cohere2.vocab_size u32              = 256000
llama_model_loader: - kv  24:               cohere2.rope.dimension_count u32              = 128
llama_model_loader: - kv  25:                  cohere2.rope.scaling.type str              = none
llama_model_loader: - kv  26:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  27:                         tokenizer.ggml.pre str              = command-r
llama_model_loader: - kv  28:                      tokenizer.ggml.tokens arr[str,256000]  = ["<PAD>", "<UNK>", "<CLS>", "<SEP>", ...
llama_model_loader: - kv  29:                  tokenizer.ggml.token_type arr[i32,256000]  = [3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, ...
llama_model_loader: - kv  30:                      tokenizer.ggml.merges arr[str,253333]  = ["Ġ Ġ", "Ġ t", "e r", "i n", "Ġ a...
llama_model_loader: - kv  31:                tokenizer.ggml.bos_token_id u32              = 5
llama_model_loader: - kv  32:                tokenizer.ggml.eos_token_id u32              = 255001
llama_model_loader: - kv  33:            tokenizer.ggml.unknown_token_id u32              = 1
llama_model_loader: - kv  34:            tokenizer.ggml.padding_token_id u32              = 0
llama_model_loader: - kv  35:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  36:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  37:                    tokenizer.chat_template str              = {{ bos_token }}{% if documents %}\n{% ...
llama_model_loader: - kv  38:               general.quantization_version u32              = 2
llama_model_loader: - kv  39:                          general.file_type u32              = 15
llama_model_loader: - kv  40:                      quantize.imatrix.file str              = /models_out/Agatha-111B-v1-GGUF/TheDr...
llama_model_loader: - kv  41:                   quantize.imatrix.dataset str              = /training_dir/calibration_datav3.txt
llama_model_loader: - kv  42:             quantize.imatrix.entries_count i32              = 448
llama_model_loader: - kv  43:              quantize.imatrix.chunks_count i32              = 509
llama_model_loader: - kv  44:                                   split.no u16              = 0
llama_model_loader: - kv  45:                        split.tensors.count i32              = 514
llama_model_loader: - kv  46:                                split.count u16              = 2
llama_model_loader: - type  f32:   65 tensors
llama_model_loader: - type q8_0:    1 tensors
llama_model_loader: - type q4_K:  384 tensors
llama_model_loader: - type q6_K:   64 tensors
llm_load_vocab: special tokens cache size = 41
llm_load_vocab: token to piece cache size = 1.8428 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = cohere2
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 256000
llm_load_print_meta: n_merges         = 253333
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 262144
llm_load_print_meta: n_embd           = 12288
llm_load_print_meta: n_layer          = 64
llm_load_print_meta: n_head           = 96
llm_load_print_meta: n_head_kv        = 8
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 4096
llm_load_print_meta: n_swa_pattern    = 4
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 12
llm_load_print_meta: n_embd_k_gqa     = 1024
llm_load_print_meta: n_embd_v_gqa     = 1024
llm_load_print_meta: f_norm_eps       = 1.0e-05
llm_load_print_meta: f_norm_rms_eps   = 0.0e+00
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 2.5e-01
llm_load_print_meta: n_ff             = 36864
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = none
llm_load_print_meta: freq_base_train  = 50000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 262144
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = ?B
llm_load_print_meta: model ftype      = Q4_K - Medium
llm_load_print_meta: model params     = 111.058 B
llm_load_print_meta: model size       = 63.224 GiB (4.890 BPW) 
llm_load_print_meta: general.name     = Agatha 111B v1
llm_load_print_meta: BOS token        = 5 '<BOS_TOKEN>'
llm_load_print_meta: EOS token        = 255001 '<|END_OF_TURN_TOKEN|>'
llm_load_print_meta: UNK token        = 1 '<UNK>'
llm_load_print_meta: PAD token        = 0 '<PAD>'
llm_load_print_meta: LF token         = 136 'Ä'
llm_load_print_meta: max token length = 1024
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 4 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 1: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 2: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 3: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
llm_load_tensors: ggml ctx size =    0.74 MiB
llm_load_tensors: offloading 64 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 65/65 layers to GPU
llm_load_tensors: CUDA_Split buffer size = 64738.50 MiB
llm_load_tensors:        CPU buffer size =  3187.50 MiB
llm_load_tensors:      CUDA0 buffer size =     3.05 MiB
..............................................................................................
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 2048
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 0
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 50000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:      CUDA0 KV buffer size =  4352.03 MiB
llama_new_context_with_model: KV self size  = 4352.00 MiB, K (q8_0): 2176.00 MiB, V (q8_0): 2176.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     1.95 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =  2096.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   608.02 MiB
llama_new_context_with_model: graph nodes  = 1578
llama_new_context_with_model: graph splits = 2
```

---

## 💬 Conversation

👤 **ikawrakow** commented on **2025-07-20** at **13:54:00**

So, I have tried to preserve the row split mode while making changes to the CUDA back-end, but not having a multi-GPU system to test, I may have broken something. Also, the row split mode is in whatever state it was in mainline when I forked the project a year ago, so who knows if it worked there back then.

If this is broken, I would need help.

---

👤 **Ph0rk0z** commented on **2025-07-20** at **14:02:34**

It was working at the time. They since refactored it to split the cache evenly among the GPUs. Where would be a good place to start?

To me this looks as if you just ran the model with empty CTX because it's coherent-ish. It processes the context and never feeds it back in when generating.

---

👤 **ikawrakow** commented on **2025-07-20** at **14:17:46**

> Where would be a good place to start?

```
git blame ggml-cuda.cu | grep -i kawrakow
```
and then try to see if I have made a change that does not consider the split row mode (or considers it but incorrectly).

For sure split row mode does not work for MoE models (and it didn't work in mainline either). It would be a much bigger undertaking to fix that.

---

👤 **Ph0rk0z** commented on **2025-07-20** at **14:31:30**

I went through all the commits in mainline regarding row and so far nothing matched up. Besides ggml-cuda, I think some changes went into mmq.cuh, they hard a race condition at one point.

I'm going to try some other models as well to see if it's only cohere related. Quantized cache didn't make a difference, disabling FA still produced incoherent output but it was at least tangentially related to the CTX. 

L2-70b is working perfectly with same settings. Sounds like command-A related issue or some facet of it's architecture.

---

👤 **Ph0rk0z** commented on **2025-07-20** at **15:52:00**

Think we getting somewhere...

Cohere2 and Gemma3 both use SWA. The output from gemma3 is actually *worse*, but it exhibits similar behavior. I posit that the bug lies there.

Is there a way to disable SWA when loading the models for testing short of commenting out lines in the code?

---

👤 **ikawrakow** commented on **2025-07-20** at **15:56:28**

Before getting into debugging SWA, can you try a model that does not use SWA? So we know that split row mode is not broken?

---

👤 **Ph0rk0z** commented on **2025-07-20** at **16:05:50**

Yes. I used DarkMiqu which is L2 70b and it works fine.

Also loading a qwen2 right now to double check.

---

👤 **Ph0rk0z** commented on **2025-07-20** at **16:35:01**

Maybe spoke too soon. 

<details>
<summary> Qwen2 Arch </summary>

```
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 0
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:      CUDA0 KV buffer size =  5440.04 MiB
llama_new_context_with_model: KV self size  = 5440.00 MiB, K (q8_0): 2720.00 MiB, V (q8_0): 2720.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     1.16 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =   313.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =    80.01 MiB
llama_new_context_with_model: graph nodes  = 2246
llama_new_context_with_model: graph splits = 2
CUDA error: an illegal memory access was encountered
  current device: 1, in function ggml_cuda_op_mul_mat at /home/supermicro/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu:1752
  cudaGetLastError()
/home/supermicro/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu:110: CUDA error
```

</details>

GLM-Z1 also has problems. Have to take a look at what that arch is doing.

Am out of different GGUF archs to try, most of the rest are EXL2. Wish I had original CR+ in gguf.

---

👤 **ikawrakow** commented on **2025-07-20** at **16:44:20**

Can you run `cuda-gdb --args put_your_qwen2_command_here`. Then when you see `(cuda-gdb)`, just type `run`.

Then post the output when it crashes.

---

👤 **Ph0rk0z** commented on **2025-07-20** at **17:12:10**

I found that it loads if I recompile without GGML_CUDA_F16.. sadly only outputs *********. I will give you a stack trace when I come back from the grocery store.

---

👤 **Ph0rk0z** commented on **2025-07-22** at **19:29:49**

Welcome back to the land of the living. Here is where it crashes with F16 enabled.

```
CUDA Exception: Warp Illegal Address
The exception was triggered at PC 0x7ff35529cff0

Thread 1 "llama-server" received signal CUDA_EXCEPTION_14, Warp Illegal Address.
[Switching focus to CUDA kernel 0, grid 561, block (0,0,0), thread (192,0,0), device 1, sm 0, warp 5, lane 0]
0x00007ff35529d000 in void convert_unary<float, __half>(void const*, __half*, long)<<<(32,1,1),(256,1,1)>>> ()
```

Not sure if it's all the same bug but could be related.

---

👤 **ikawrakow** commented on **2025-07-23** at **10:49:02**

@Ph0rk0z Thank you. This narrows it down somewhat, but it would be useful to have the backtrace to see from where this was triggered. If you have time to recreate the issue, do `bt` or `thread apply all bt` when it crashes, and post the result.

---

👤 **Ph0rk0z** commented on **2025-07-23** at **11:51:58**

I'm not very good with GDB so wasn't sure what to do after.

<details>
<summary> Backtrace </summary>


```
CUDA Exception: Warp Illegal Address
The exception was triggered at PC 0x7ff35529cff0

Thread 1 "llama-server" received signal CUDA_EXCEPTION_14, Warp Illegal Address.
[Switching focus to CUDA kernel 0, grid 561, block (0,0,0), thread (192,0,0), device 1, sm 0, warp 7, lane 0]
0x00007ff35529d000 in void convert_unary<float, __half>(void const*, __half*, long)<<<(32,1,1),(256,1,1)>>> ()
(cuda-gdb) thread apply all bt

Thread 11 (LWP 1716794 "cuda-EvtHandlr"):
#0  0x00007fffd1518bcf in poll () from /lib/x86_64-linux-gnu/libc.so.6
#1  0x00007fffc6708517 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#2  0x00007fffc67cb17f in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#3  0x00007fffc66f8b23 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#4  0x00007fffd1494ac3 in ?? () from /lib/x86_64-linux-gnu/libc.so.6
#5  0x00007fffd1526850 in ?? () from /lib/x86_64-linux-gnu/libc.so.6

Thread 10 (LWP 1716793 "llama-server"):
#0  0x00007fffd1491117 in ?? () from /lib/x86_64-linux-gnu/libc.so.6
#1  0x00007fffd1493e9b in pthread_cond_timedwait () from /lib/x86_64-linux-gnu/libc.so.6
#2  0x00007fffc6673eea in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#3  0x00007fffc66f8b23 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#4  0x00007fffd1494ac3 in ?? () from /lib/x86_64-linux-gnu/libc.so.6
#5  0x00007fffd1526850 in ?? () from /lib/x86_64-linux-gnu/libc.so.6

Thread 9 (LWP 1716773 "cuda-EvtHandlr"):
#0  0x00007fffd1518bcf in poll () from /lib/x86_64-linux-gnu/libc.so.6
#1  0x00007fffc6708517 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#2  0x00007fffc67cb17f in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#3  0x00007fffc66f8b23 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#4  0x00007fffd1494ac3 in ?? () from /lib/x86_64-linux-gnu/libc.so.6
#5  0x00007fffd1526850 in ?? () from /lib/x86_64-linux-gnu/libc.so.6

Thread 8 (LWP 1716772 "llama-server"):
#0  0x00007fffd1491117 in ?? () from /lib/x86_64-linux-gnu/libc.so.6
#1  0x00007fffd1493e9b in pthread_cond_timedwait () from /lib/x86_64-linux-gnu/libc.so.6
#2  0x00007fffc6673eea in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#3  0x00007fffc66f8b23 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#4  0x00007fffd1494ac3 in ?? () from /lib/x86_64-linux-gnu/libc.so.6
#5  0x00007fffd1526850 in ?? () from /lib/x86_64-linux-gnu/libc.so.6

Thread 7 (LWP 1716743 "cuda-EvtHandlr"):
#0  0x00007fffd1518bcf in poll () from /lib/x86_64-linux-gnu/libc.so.6
#1  0x00007fffc6708517 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#2  0x00007fffc67cb17f in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#3  0x00007fffc66f8b23 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#4  0x00007fffd1494ac3 in ?? () from /lib/x86_64-linux-gnu/libc.so.6
#5  0x00007fffd1526850 in ?? () from /lib/x86_64-linux-gnu/libc.so.6

--Type <RET> for more, q to quit, c to continue without paging--
Thread 6 (LWP 1716742 "llama-server"):
#0  0x00007fffd1491117 in ?? () from /lib/x86_64-linux-gnu/libc.so.6
#1  0x00007fffd1493e9b in pthread_cond_timedwait () from /lib/x86_64-linux-gnu/libc.so.6
#2  0x00007fffc6673eea in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#3  0x00007fffc66f8b23 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#4  0x00007fffd1494ac3 in ?? () from /lib/x86_64-linux-gnu/libc.so.6
#5  0x00007fffd1526850 in ?? () from /lib/x86_64-linux-gnu/libc.so.6

Thread 5 (LWP 1716697 "cuda-EvtHandlr"):
#0  0x00007fffd1518bcf in poll () from /lib/x86_64-linux-gnu/libc.so.6
#1  0x00007fffc6708517 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#2  0x00007fffc67cb17f in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#3  0x00007fffc66f8b23 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#4  0x00007fffd1494ac3 in ?? () from /lib/x86_64-linux-gnu/libc.so.6
#5  0x00007fffd1526850 in ?? () from /lib/x86_64-linux-gnu/libc.so.6

Thread 4 (LWP 1716696 "llama-server"):
#0  0x00007fffd1491117 in ?? () from /lib/x86_64-linux-gnu/libc.so.6
#1  0x00007fffd1493e9b in pthread_cond_timedwait () from /lib/x86_64-linux-gnu/libc.so.6
#2  0x00007fffc6673eea in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#3  0x00007fffc66f8b23 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#4  0x00007fffd1494ac3 in ?? () from /lib/x86_64-linux-gnu/libc.so.6
#5  0x00007fffd1526850 in ?? () from /lib/x86_64-linux-gnu/libc.so.6

Thread 3 (LWP 1716686 "llama-server"):
#0  0x00007fffd1525e2e in epoll_wait () from /lib/x86_64-linux-gnu/libc.so.6
#1  0x00007fffa517f6e6 in ?? () from /lib/x86_64-linux-gnu/libcudadebugger.so.1
#2  0x00007fffa517c939 in ?? () from /lib/x86_64-linux-gnu/libcudadebugger.so.1
#3  0x00007fffa517ef3b in ?? () from /lib/x86_64-linux-gnu/libcudadebugger.so.1
#4  0x00007fffa51894ba in ?? () from /lib/x86_64-linux-gnu/libcudadebugger.so.1
#5  0x00007fffa51899bb in ?? () from /lib/x86_64-linux-gnu/libcudadebugger.so.1
#6  0x00007fffa505dc15 in ?? () from /lib/x86_64-linux-gnu/libcudadebugger.so.1
#7  0x00007fffa4ec7837 in ?? () from /lib/x86_64-linux-gnu/libcudadebugger.so.1
#8  0x00007fffd1494ac3 in ?? () from /lib/x86_64-linux-gnu/libc.so.6
#9  0x00007fffd1526850 in ?? () from /lib/x86_64-linux-gnu/libc.so.6

Thread 2 (LWP 1716685 "cuda00002000009"):
#0  0x00007fffd1518bcf in poll () from /lib/x86_64-linux-gnu/libc.so.6
#1  0x00007fffc6708517 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#2  0x00007fffc67cb17f in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#3  0x00007fffc66f8b23 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
--Type <RET> for more, q to quit, c to continue without paging--
#4  0x00007fffd1494ac3 in ?? () from /lib/x86_64-linux-gnu/libc.so.6
#5  0x00007fffd1526850 in ?? () from /lib/x86_64-linux-gnu/libc.so.6

Thread 1 (LWP 1716680 "llama-server"):
#0  0x00007fffd1491117 in ?? () from /lib/x86_64-linux-gnu/libc.so.6
#1  0x00007fffd1493a41 in pthread_cond_wait () from /lib/x86_64-linux-gnu/libc.so.6
#2  0x00007fffa517ca1b in ?? () from /lib/x86_64-linux-gnu/libcudadebugger.so.1
#3  0x00007fffa517ef3b in ?? () from /lib/x86_64-linux-gnu/libcudadebugger.so.1
#4  0x00007fffa4ee080d in ?? () from /lib/x86_64-linux-gnu/libcudadebugger.so.1
#5  0x00007fffa4f4a049 in ?? () from /lib/x86_64-linux-gnu/libcudadebugger.so.1
#6  0x00007fffa4d12718 in ?? () from /lib/x86_64-linux-gnu/libcudadebugger.so.1
#7  0x00007fffa4fdc5b6 in ?? () from /lib/x86_64-linux-gnu/libcudadebugger.so.1
#8  0x00007fffc65ace1a in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#9  0x00007fffc672238c in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#10 0x00007fffd1012f91 in ?? () from /home/supermicro/miniconda3/envs/cuda12/lib/libcudart.so.12
#11 0x00007fffd104f5e4 in cudaSetDevice () from /home/supermicro/miniconda3/envs/cuda12/lib/libcudart.so.12
#12 0x00007fffd2706552 in ggml_cuda_set_device(int) () from /home/supermicro/ai/ik_llama.cpp/ggml/src/libggml.so
#13 0x00007fffd2710fe2 in ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [clone .constprop.0] () from /home/supermicro/ai/ik_llama.cpp/ggml/src/libggml.so
#14 0x00007fffd271cbc6 in ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) () from /home/supermicro/ai/ik_llama.cpp/ggml/src/libggml.so
#15 0x00007fffd1917716 in ggml_backend_sched_graph_compute_async () from /home/supermicro/ai/ik_llama.cpp/ggml/src/libggml.so
#16 0x00007ffff7ea2832 in llama_decode () from /home/supermicro/ai/ik_llama.cpp/src/libllama.so
#17 0x000055555562f6b1 in llama_init_from_gpt_params(gpt_params&) ()
#18 0x00005555555c48a6 in server_context::load_model(gpt_params const&) ()
#19 0x0000555555579418 in main ()
```


</summary>

---

👤 **ikawrakow** commented on **2025-07-23** at **12:08:51**

It is a release build, so there isn't much you can do. I don't think I understand what the bug is from the backtrace. I guess, the only way to resolve this is for me to get a multi-GPU system and debug it myself.

But thanks for helping.

---

👤 **Ph0rk0z** commented on **2025-07-23** at **12:42:55**

Not worth it for me to compile it a different way? I saw in the code that on certain arch it upcasts to FP32 for several calculations and I have enabled FP16 cuda. Command-A and I think qwen2 among them. But cohere2 doesn't crash with FP16, just ignores the prompt and qwen2 loads but outputs only one or two token ids. 

BTW, some MoE can also load in row split, it just t/s tanks. IIRC, either qwen-235b or deepseek-v2.5 did.