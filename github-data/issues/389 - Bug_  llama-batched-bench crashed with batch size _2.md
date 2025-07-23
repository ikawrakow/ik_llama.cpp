### 🐛 [#389](https://github.com/ikawrakow/ik_llama.cpp/issues/389) - Bug:  llama-batched-bench crashed with batch size >2

| **Author** | `QuPengfei` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-07 |
| **Updated** | 2025-05-23 |

---

#### Description

### What happened?


failed with command when bs >2
numactl -m 0 -C 0-127 ./llama-batched-bench -m /models/unsloth/Qwen3-235B-A22B-GGUF/Q4_K_M/*00001*.gguf -c 8192  -b 2048 -ub 512 -ngl 0 -npp 128 -ntg 128 -npl 1,2,4 --cache-type-k q8_0 --numa numactl --threads 64 --threads-batch 128 -fa -fmoe -amb 1 -ser 7,1 -mla 1  --no-mmap


### Name and Version

build: e3fec173 (3667)

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
warning: not compiled with GPU offload support, --gpu-layers option will be ignored
warning: see main README.md for information on enabling GPU BLAS support
WARNING: /proc/sys/kernel/numa_balancing is enabled, this has been observed to impair performance
llama_model_loader: additional 2 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 46 key-value pairs and 1131 tensors from /models/unsloth/Qwen3-235B-A22B-GGUF/Q4_K_M/Qwen3-235B-A22B-Q4_K_M-00001-of-00003.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3-235B-A22B
llama_model_loader: - kv   3:                           general.basename str              = Qwen3-235B-A22B
llama_model_loader: - kv   4:                       general.quantized_by str              = Unsloth
llama_model_loader: - kv   5:                         general.size_label str              = 235B-A22B
llama_model_loader: - kv   6:                            general.license str              = apache-2.0
llama_model_loader: - kv   7:                       general.license.link str              = https://huggingface.co/Qwen/Qwen3-235...
llama_model_loader: - kv   8:                           general.repo_url str              = https://huggingface.co/unsloth
llama_model_loader: - kv   9:                   general.base_model.count u32              = 1
llama_model_loader: - kv  10:                  general.base_model.0.name str              = Qwen3 235B A22B
llama_model_loader: - kv  11:          general.base_model.0.organization str              = Qwen
llama_model_loader: - kv  12:              general.base_model.0.repo_url str              = https://huggingface.co/Qwen/Qwen3-235...
llama_model_loader: - kv  13:                               general.tags arr[str,2]       = ["unsloth", "text-generation"]
llama_model_loader: - kv  14:                       qwen3moe.block_count u32              = 94
llama_model_loader: - kv  15:                    qwen3moe.context_length u32              = 40960
llama_model_loader: - kv  16:                  qwen3moe.embedding_length u32              = 4096
llama_model_loader: - kv  17:               qwen3moe.feed_forward_length u32              = 12288
llama_model_loader: - kv  18:              qwen3moe.attention.head_count u32              = 64
llama_model_loader: - kv  19:           qwen3moe.attention.head_count_kv u32              = 4
llama_model_loader: - kv  20:                    qwen3moe.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  21:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  22:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  23:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  24:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  25:                      qwen3moe.expert_count u32              = 128
llama_model_loader: - kv  26:        qwen3moe.expert_feed_forward_length u32              = 1536
llama_model_loader: - kv  27:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  28:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  29:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  30:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  31:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv  32:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  33:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  34:                tokenizer.ggml.bos_token_id u32              = 151643
llama_model_loader: - kv  35:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  36:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - kv  37:               general.quantization_version u32              = 2
llama_model_loader: - kv  38:                          general.file_type u32              = 15
llama_model_loader: - kv  39:                      quantize.imatrix.file str              = Qwen3-235B-A22B-GGUF/imatrix_unsloth.dat
llama_model_loader: - kv  40:                   quantize.imatrix.dataset str              = unsloth_calibration_Qwen3-235B-A22B.txt
llama_model_loader: - kv  41:             quantize.imatrix.entries_count i32              = 752
llama_model_loader: - kv  42:              quantize.imatrix.chunks_count i32              = 32
llama_model_loader: - kv  43:                                   split.no u16              = 0
llama_model_loader: - kv  44:                        split.tensors.count i32              = 1131
llama_model_loader: - kv  45:                                split.count u16              = 3
llama_model_loader: - type  f32:  471 tensors
llama_model_loader: - type q4_K:  567 tensors
llama_model_loader: - type q6_K:   93 tensors
llm_load_vocab: special tokens cache size = 26
llm_load_vocab: token to piece cache size = 0.9311 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = qwen3moe
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 151936
llm_load_print_meta: n_merges         = 151387
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 40960
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_layer          = 94
llm_load_print_meta: n_head           = 64
llm_load_print_meta: n_head_kv        = 4
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 16
llm_load_print_meta: n_embd_k_gqa     = 512
llm_load_print_meta: n_embd_v_gqa     = 512
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 12288
llm_load_print_meta: n_expert         = 128
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 2
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 1000000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 40960
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = ?B
llm_load_print_meta: model ftype      = Q4_K - Medium
llm_load_print_meta: model params     = 235.094 B
llm_load_print_meta: model size       = 132.386 GiB (4.837 BPW)
llm_load_print_meta: repeating layers = 131.584 GiB (4.833 BPW, 233.849 B parameters)
llm_load_print_meta: general.name     = Qwen3-235B-A22B
llm_load_print_meta: BOS token        = 151643 '<|endoftext|>'
llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
llm_load_print_meta: PAD token        = 151643 '<|endoftext|>'
llm_load_print_meta: LF token         = 148848 'ÄĬ'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp         = 1536
llm_load_tensors: ggml ctx size =    0.50 MiB
llm_load_tensors:        CPU buffer size = 135562.96 MiB
....................................................................................................
=====================================================================
 MLA is only available for LLM_ARCH_DEEPSEEK2 -> turning off MLA
=====================================================================
llama_new_context_with_model: n_ctx      = 8192
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 1
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = 7, 1
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:        CPU KV buffer size =  1151.50 MiB
llama_new_context_with_model: KV self size  = 1151.50 MiB, K (q8_0):  399.50 MiB, V (f16):  752.00 MiB
llama_new_context_with_model:        CPU  output buffer size =     2.32 MiB
llama_new_context_with_model:        CPU compute buffer size =   304.75 MiB
llama_new_context_with_model: graph nodes  = 3672
llama_new_context_with_model: graph splits = 942
Unable to find TSan function AnnotateHappensAfter.
Unable to find TSan function AnnotateHappensBefore.
Unable to find TSan function AnnotateIgnoreWritesBegin.
Unable to find TSan function AnnotateIgnoreWritesEnd.
Unable to find TSan function AnnotateNewMemory.
Unable to find TSan function __tsan_func_entry.
Unable to find TSan function __tsan_func_exit.
Warning: please export TSAN_OPTIONS='ignore_noninstrumented_modules=1' to avoid false positive reports from the OpenMP runtime!

main: n_kv_max = 8192, n_batch = 2048, n_ubatch = 512, flash_attn = 1, is_pp_shared = 0, n_gpu_layers = 0, n_threads = 64, n_threads_batch = 128

|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
|   128 |    128 |    1 |    256 |    1.778 |    71.99 |    5.578 |    22.95 |    7.357 |    34.80 |
|   128 |    128 |    2 |    512 |    2.265 |   113.01 |    7.968 |    32.13 |   10.233 |    50.03 |
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: /app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: /app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: /app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
OMP: Warning #191: Forking a process while a parallel region is active is potentially unsafe.
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: /app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failedGGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed

/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed

GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed


OMP: Warning #191: Forking a process while a parallel region is active is potentially unsafe.
OMP: Warning #191: Forking a process while a parallel region is active is potentially unsafe.
libggml.so(+0x134d7)[0x725d77a3e4d7]
libggml.so(ggml_abort+0xd8)[0x725d77a3e468]
libggml.so(+0xcbf7da)[0x725d786ea7da]
OMP: Warning #191: Forking a process while a parallel region is active is potentially unsafe.
libggml.so(+0x468f0a)[0x725d77e93f0a]
libggml.so(_Z19iqk_flash_attn_impliiiiiiiiiiiPKfPKvS2_S2_ffPfS3_S3_+0x405)[0x725d77d0a175]
libggml.so(iqk_flash_attn_noalibi+0x1419)[0x725d79cc7e29]
libggml.so(+0x3a347)[0x725d77a65347]
/usr/local/lib/libiomp5.so(__kmp_invoke_microtask+0x93)[0x725d7a145603]
/usr/local/lib/libiomp5.so(+0xca633)[0x725d7a0ca633]
/usr/local/lib/libiomp5.so(+0xc90ae)[0x725d7a0c90ae]
/usr/local/lib/libiomp5.so(+0x146c21)[0x725d7a146c21]
/lib/x86_64-linux-gnu/libc.so.6(+0x94ac3)[0x725d7766aac3]
/lib/x86_64-linux-gnu/libc.so.6(+0x126850)[0x725d776fc850]
Aborted (core dumped)
```

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-05-07** at **05:21:57**:<br>

This assert almost always indicates a NaN somewhere in the calculation. What happens if you remove `-amb 1 -ser 7,1 -mla 1`

---

👤 **QuPengfei** commented the **2025-05-07** at **06:58:07**:<br>

Just confirmed,  this happened with -ser 7,1.

BTW, 
- i compiled the binary with OneAPI and icx.  If without OneAPI and icx, it worked well even with -ser 7,1.
- with OneAPI, S_PP t/s become worse. 

here is the options:
cmake -B build -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Intel10_64lp -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_NATIVE=ON 
thanks,
Pengfei

---

👤 **ikawrakow** commented the **2025-05-07** at **07:04:50**:<br>

Try building with BLAS disabled. I expect this to improve performance quite a bit.

I'll have to investigate why `-ser 7,1` leads to a problem. Normally it should work.

---

👤 **QuPengfei** commented the **2025-05-07** at **13:04:45**:<br>

@ikawrakow 

i see the similar issue on the DeepSeek-R1-Q4_K_M

here are observation with different runs:
- if run with --cache-type-k q4_0, bs1 got lower performance and bs2 performance is back.

![Image](https://github.com/user-attachments/assets/ab27d85f-e459-433b-8aea-2b4257dc770f)

- if run with --cache-type-k q8_0, bs1 performance is normal but failed when bs > 2
- if remove -ser 7,1 , performance will be very low.

here is command and log:
====
numactl -m 1 -C 128-255 ./llama-batched-bench -m /models1/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00001-of-00009.gguf -c 8192  -b 2048 -ub 512 -ngl 0 -npp 128 -ntg 128 -npl 1,2,4,8 --cache-type-k q8_0 --numa numactl --threads 64 --threads-batch 128 -fa -fmoe -amb 1 -ser 7,1 -mla 0  --no-mmap
warning: not compiled with GPU offload support, --gpu-layers option will be ignored
warning: see main README.md for information on enabling GPU BLAS support
WARNING: /proc/sys/kernel/numa_balancing is enabled, this has been observed to impair performance
llama_model_loader: additional 8 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 48 key-value pairs and 1025 tensors from /models1/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00001-of-00009.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = DeepSeek R1 BF16
llama_model_loader: - kv   3:                       general.quantized_by str              = Unsloth
llama_model_loader: - kv   4:                         general.size_label str              = 256x20B
llama_model_loader: - kv   5:                           general.repo_url str              = https://huggingface.co/unsloth
llama_model_loader: - kv   6:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv   7:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv   8:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv   9:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  10:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv  11:          deepseek2.attention.head_count_kv u32              = 128
llama_model_loader: - kv  12:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  13: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  14:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  15:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  16:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  17:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  18:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  19:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  20:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  21:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  22:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  23:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  24:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  25:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  26:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  27:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  28:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  29:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  30: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  31: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  32:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  33:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  34:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<▒...
llama_model_loader: - kv  35:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  36:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
llama_model_loader: - kv  37:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  38:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  39:            tokenizer.ggml.padding_token_id u32              = 128815
llama_model_loader: - kv  40:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  41:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  42:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  43:               general.quantization_version u32              = 2
llama_model_loader: - kv  44:                          general.file_type u32              = 15
llama_model_loader: - kv  45:                                   split.no u16              = 0
llama_model_loader: - kv  46:                        split.tensors.count i32              = 1025
llama_model_loader: - kv  47:                                split.count u16              = 9
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q4_K:  606 tensors
llama_model_loader: - type q6_K:   58 tensors
llm_load_vocab: special tokens cache size = 819
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
llm_load_print_meta: n_head_kv        = 128
llm_load_print_meta: n_rot            = 64
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 192
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: n_embd_k_gqa     = 24576
llm_load_print_meta: n_embd_v_gqa     = 16384
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
llm_load_print_meta: model ftype      = Q4_K - Medium
llm_load_print_meta: model params     = 671.026 B
llm_load_print_meta: model size       = 376.650 GiB (4.822 BPW)
llm_load_print_meta: repeating layers = 375.457 GiB (4.820 BPW, 669.173 B parameters)
llm_load_print_meta: general.name     = DeepSeek R1 BF16
llm_load_print_meta: BOS token        = 0 '<｜begin▁of▁sentence｜>'
llm_load_print_meta: EOS token        = 1 '<｜end▁of▁sentence｜>'
llm_load_print_meta: PAD token        = 128815 '<｜PAD▁TOKEN｜>'
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
llm_load_tensors: ggml ctx size =    0.42 MiB
llm_load_tensors:        CPU buffer size = 385689.63 MiB
....................................................................................................
============ llm_load_tensors: need to compute 61 wk_b tensors
Computed blk.0.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.1.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.2.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.3.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.4.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.5.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.6.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.7.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.8.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.9.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.10.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.11.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.12.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.13.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.14.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.15.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.16.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.17.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.18.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.19.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.20.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.21.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.22.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.23.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.24.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.25.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.26.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.27.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.28.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.29.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.30.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.31.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.32.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.33.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.34.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.35.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.36.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.37.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.38.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.39.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.40.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.41.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.42.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.43.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.44.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.45.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.46.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.47.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.48.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.49.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.50.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.51.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.52.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.53.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.54.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.55.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.56.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.57.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.58.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.59.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.60.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
llama_new_context_with_model: n_ctx      = 8192
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 1
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = 7, 1
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init:        CPU KV buffer size = 28060.00 MiB
llama_new_context_with_model: KV self size  = 28060.00 MiB, K (q8_0): 12444.00 MiB, V (f16): 15616.00 MiB
llama_new_context_with_model:        CPU  output buffer size =     3.95 MiB
llama_new_context_with_model:        CPU compute buffer size =   266.50 MiB
llama_new_context_with_model: graph nodes  = 3365
llama_new_context_with_model: graph splits = 1

main: n_kv_max = 8192, n_batch = 2048, n_ubatch = 512, flash_attn = 1, is_pp_shared = 0, n_gpu_layers = 0, n_threads = 64, n_threads_batch = 128

|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
|   128 |    128 |    1 |    256 |    1.560 |    82.05 |   10.533 |    12.15 |   12.094 |    21.17 |
|   128 |    128 |    2 |    512 |    2.663 |    96.14 |    9.856 |    25.97 |   12.519 |    40.90 |


/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: /app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: /app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: /app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: /app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: /app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: /app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: /app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: OMP: Warning #191: Forking a process while a parallel region is active is potentially unsafe.
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: /app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: /app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failedGGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
GGML_ASSERT(fms.S[j] > 0) failed
GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
GGML_ASSERT(fms.S[j] > 0) failed
GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed

/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed

GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed

/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: /app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed

/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
GGML_ASSERT(fms.S[j] > 0) failed



/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed




/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
OMP: Warning #191: Forking a process while a parallel region is active is potentially unsafe.
OMP: Warning #191: Forking a process while a parallel region is active is potentially unsafe.
libggml.so(+0x221ab)[0x77d53049d1ab]
libggml.so(ggml_abort+0x15e)[0x77d53049f76e]
libggml.so(+0x1c1217)[0x77d53063c217]
OMP: Warning #191: Forking a process while a parallel region is active is potentially unsafe.
libggml.so(+0x1caef9)[0x77d530645ef9]
libggml.so(+0x96ff2f)[0x77d530deaf2f]
libggml.so(+0xc4787f)[0x77d5310c287f]
libggml.so(_Z19iqk_flash_attn_impliiiiiiiiiiiPKfPKvS2_S2_ffPfS3_S3_+0x74b)[0x77d5310d275b]
libggml.so(iqk_flash_attn_noalibi+0xa70)[0x77d5310d3760]
libggml.so(+0x2dee0)[0x77d5304a8ee0]
libggml.so(+0x61f52)[0x77d5304dcf52]
libggml.so(+0x636bc)[0x77d5304de6bc]
libggml.so(+0x638a9)[0x77d5304de8a9]
/usr/local/lib/libiomp5.so(+0xa942b)[0x77d5314a942b]
/usr/local/lib/libiomp5.so(__kmp_invoke_microtask+0x93)[0x77d531545603]
/usr/local/lib/libiomp5.so(+0xca633)[0x77d5314ca633]
/usr/local/lib/libiomp5.so(+0xc90ae)[0x77d5314c90ae]
/usr/local/lib/libiomp5.so(+0x146c21)[0x77d531546c21]
/lib/x86_64-linux-gnu/libc.so.6(+0x94ac3)[0x77d5300baac3]
/lib/x86_64-linux-gnu/libc.so.6(+0x126850)[0x77d53014c850]
Aborted (core dumped)

---

👤 **QuPengfei** commented the **2025-05-07** at **13:04:45**:<br>

@ikawrakow 

i see the similar issue on the DeepSeek-R1-Q4_K_M

here are observation with different runs:
- if run with --cache-type-k q4_0, bs1 got lower performance and bs2 performance is back.

![Image](https://github.com/user-attachments/assets/ab27d85f-e459-433b-8aea-2b4257dc770f)

- if run with --cache-type-k q8_0, bs1 performance is normal but failed when bs > 2
- if remove -ser 7,1 , performance will be very low.

here is command and log:
====
numactl -m 1 -C 128-255 ./llama-batched-bench -m /models1/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00001-of-00009.gguf -c 8192  -b 2048 -ub 512 -ngl 0 -npp 128 -ntg 128 -npl 1,2,4,8 --cache-type-k q8_0 --numa numactl --threads 64 --threads-batch 128 -fa -fmoe -amb 1 -ser 7,1 -mla 0  --no-mmap
warning: not compiled with GPU offload support, --gpu-layers option will be ignored
warning: see main README.md for information on enabling GPU BLAS support
WARNING: /proc/sys/kernel/numa_balancing is enabled, this has been observed to impair performance
llama_model_loader: additional 8 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 48 key-value pairs and 1025 tensors from /models1/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00001-of-00009.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = DeepSeek R1 BF16
llama_model_loader: - kv   3:                       general.quantized_by str              = Unsloth
llama_model_loader: - kv   4:                         general.size_label str              = 256x20B
llama_model_loader: - kv   5:                           general.repo_url str              = https://huggingface.co/unsloth
llama_model_loader: - kv   6:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv   7:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv   8:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv   9:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  10:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv  11:          deepseek2.attention.head_count_kv u32              = 128
llama_model_loader: - kv  12:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  13: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  14:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  15:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  16:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  17:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  18:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  19:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  20:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  21:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  22:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  23:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  24:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  25:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  26:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  27:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  28:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  29:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  30: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  31: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  32:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  33:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  34:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<▒...
llama_model_loader: - kv  35:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  36:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
llama_model_loader: - kv  37:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  38:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  39:            tokenizer.ggml.padding_token_id u32              = 128815
llama_model_loader: - kv  40:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  41:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  42:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  43:               general.quantization_version u32              = 2
llama_model_loader: - kv  44:                          general.file_type u32              = 15
llama_model_loader: - kv  45:                                   split.no u16              = 0
llama_model_loader: - kv  46:                        split.tensors.count i32              = 1025
llama_model_loader: - kv  47:                                split.count u16              = 9
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q4_K:  606 tensors
llama_model_loader: - type q6_K:   58 tensors
llm_load_vocab: special tokens cache size = 819
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
llm_load_print_meta: n_head_kv        = 128
llm_load_print_meta: n_rot            = 64
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 192
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: n_embd_k_gqa     = 24576
llm_load_print_meta: n_embd_v_gqa     = 16384
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
llm_load_print_meta: model ftype      = Q4_K - Medium
llm_load_print_meta: model params     = 671.026 B
llm_load_print_meta: model size       = 376.650 GiB (4.822 BPW)
llm_load_print_meta: repeating layers = 375.457 GiB (4.820 BPW, 669.173 B parameters)
llm_load_print_meta: general.name     = DeepSeek R1 BF16
llm_load_print_meta: BOS token        = 0 '<｜begin▁of▁sentence｜>'
llm_load_print_meta: EOS token        = 1 '<｜end▁of▁sentence｜>'
llm_load_print_meta: PAD token        = 128815 '<｜PAD▁TOKEN｜>'
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
llm_load_tensors: ggml ctx size =    0.42 MiB
llm_load_tensors:        CPU buffer size = 385689.63 MiB
....................................................................................................
============ llm_load_tensors: need to compute 61 wk_b tensors
Computed blk.0.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.1.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.2.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.3.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.4.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.5.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.6.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.7.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.8.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.9.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.10.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.11.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.12.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.13.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.14.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.15.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.16.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.17.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.18.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.19.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.20.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.21.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.22.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.23.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.24.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.25.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.26.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.27.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.28.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.29.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.30.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.31.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.32.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.33.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.34.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.35.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.36.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.37.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.38.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.39.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.40.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.41.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.42.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.43.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.44.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.45.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.46.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.47.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.48.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.49.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.50.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.51.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.52.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.53.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.54.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.55.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.56.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.57.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.58.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.59.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.60.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
llama_new_context_with_model: n_ctx      = 8192
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 1
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = 7, 1
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init:        CPU KV buffer size = 28060.00 MiB
llama_new_context_with_model: KV self size  = 28060.00 MiB, K (q8_0): 12444.00 MiB, V (f16): 15616.00 MiB
llama_new_context_with_model:        CPU  output buffer size =     3.95 MiB
llama_new_context_with_model:        CPU compute buffer size =   266.50 MiB
llama_new_context_with_model: graph nodes  = 3365
llama_new_context_with_model: graph splits = 1

main: n_kv_max = 8192, n_batch = 2048, n_ubatch = 512, flash_attn = 1, is_pp_shared = 0, n_gpu_layers = 0, n_threads = 64, n_threads_batch = 128

|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
|   128 |    128 |    1 |    256 |    1.560 |    82.05 |   10.533 |    12.15 |   12.094 |    21.17 |
|   128 |    128 |    2 |    512 |    2.663 |    96.14 |    9.856 |    25.97 |   12.519 |    40.90 |
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: /app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: /app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: /app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: /app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: /app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: /app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: /app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: OMP: Warning #191: Forking a process while a parallel region is active is potentially unsafe.
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: /app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: /app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failedGGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
GGML_ASSERT(fms.S[j] > 0) failed
GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
GGML_ASSERT(fms.S[j] > 0) failed
GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed

/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed

GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed

/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: /app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed

/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
GGML_ASSERT(fms.S[j] > 0) failed



/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed




/app/llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
OMP: Warning #191: Forking a process while a parallel region is active is potentially unsafe.
OMP: Warning #191: Forking a process while a parallel region is active is potentially unsafe.
libggml.so(+0x221ab)[0x77d53049d1ab]
libggml.so(ggml_abort+0x15e)[0x77d53049f76e]
libggml.so(+0x1c1217)[0x77d53063c217]
OMP: Warning #191: Forking a process while a parallel region is active is potentially unsafe.
libggml.so(+0x1caef9)[0x77d530645ef9]
libggml.so(+0x96ff2f)[0x77d530deaf2f]
libggml.so(+0xc4787f)[0x77d5310c287f]
libggml.so(_Z19iqk_flash_attn_impliiiiiiiiiiiPKfPKvS2_S2_ffPfS3_S3_+0x74b)[0x77d5310d275b]
libggml.so(iqk_flash_attn_noalibi+0xa70)[0x77d5310d3760]
libggml.so(+0x2dee0)[0x77d5304a8ee0]
libggml.so(+0x61f52)[0x77d5304dcf52]
libggml.so(+0x636bc)[0x77d5304de6bc]
libggml.so(+0x638a9)[0x77d5304de8a9]
/usr/local/lib/libiomp5.so(+0xa942b)[0x77d5314a942b]
/usr/local/lib/libiomp5.so(__kmp_invoke_microtask+0x93)[0x77d531545603]
/usr/local/lib/libiomp5.so(+0xca633)[0x77d5314ca633]
/usr/local/lib/libiomp5.so(+0xc90ae)[0x77d5314c90ae]
/usr/local/lib/libiomp5.so(+0x146c21)[0x77d531546c21]
/lib/x86_64-linux-gnu/libc.so.6(+0x94ac3)[0x77d5300baac3]
/lib/x86_64-linux-gnu/libc.so.6(+0x126850)[0x77d53014c850]
Aborted (core dumped)

---

👤 **saood06** commented the **2025-05-16** at **11:09:52**:<br>

Now that SER has been fixed (#404 #415 #416) can you try again?

---

👤 **QuPengfei** commented the **2025-05-21** at **01:20:24**:<br>

thanks. it worked now.

BTW, I found there is performance regression for S_TG when bs1. (12 tokens/s vs 10 tokens/s)

here is the data for fixed version.
![Image](https://github.com/user-attachments/assets/b040fdb6-f4a6-48f2-88b5-e60a91011cc3)