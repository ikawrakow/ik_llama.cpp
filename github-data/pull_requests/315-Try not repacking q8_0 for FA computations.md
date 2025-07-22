### 🔀 [#315](https://github.com/ikawrakow/ik_llama.cpp/pull/315) - Try not repacking q8_0 for FA computations

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-04-06 |
| **Updated** | 2025-05-04 |

---

#### Description

On the master branch if the K-cache is `Q8_0` it is repacked to `Q8_0_R8` before performing the Flash Attention computation. This is only done for PP (number of tokens in the batch $\ge$ 8), and tends to improve PP performance when the K-cache size is not too large. But for large K-cache, performance may suffer due to the additional allocation of a fairly significant amount of memory.

This PR disables K-cache repacking to `Q8_0_R8` in the Flash Attention CPU implementation.
I'm throwing it out for testing with `Q8_0` KV cache and large context lengths.
    
I cannot test DeepSeek-V3/R1, but for DeepSeek-Lite I get inconclusive results:
* On my Ryzen-5975WX, PP performance remains about the same, while we get ~15% better TG performance with a context of 32k tokens
* On my Ryzen-7950X, TG performance remains about the same, but we get ~15% **lower** PP performance with a context of 32k tokens.  

Worth noting that the repacking is not done for TG. The effects on TG performance are merely due to the additional largish memory allocation that occurs during PP. Hence, it is hard to predict what happens with a very large model such as DeepSeek-V3/R1.

Another interesting observation is that there is no difference between offline and run-time repacking of the model weights on the Ryzen-7950X. But on the Ryzen-5975WX the offline repacked model results in ~10% better TG and PP performance with a context of 32k tokens.

---

#### 💬 Conversation

👤 **ubergarm** commented the **2025-04-06** at **15:43:02**:<br>

Picking up the conversation from [296](https://github.com/ikawrakow/ik_llama.cpp/issues/296#issuecomment-2781293572), I've run a comparison with the only variable being this PR (no repacking q8_0 for kcache).

#### PP
![performance_comparison_pp-2](https://github.com/user-attachments/assets/322560ef-5c7a-4365-afcb-2b71c26affbb)

#### TG
![performance_comparison_tg-2](https://github.com/user-attachments/assets/6783cd85-38ba-45b9-bc22-1160b7c200cd)

#### Observations
So at least on this Intel Xeon 6980P rig using a single socket, it seems like repacking is generally better for both PP and TG out to 32k context on this V3-0324 quant.

Still have some slight peaks in tg at the same places, I may try a run with say 80 tg threads to see if it shifts the peaks...

Thanks!

<details>

<summary>logs of this PR's sweep-bench run</summary>

```bash
$ git branch | grep try
* ik/try_fa_no_q80_repack

$ git rev-parse --short HEAD
0dbcd572

$ ./build/bin/llama-server --version
version: 3623 (0dbcd572)
built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu

$ numactl -N 0 -m 0 \
./build/bin/llama-sweep-bench \
    --model /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-CPU-IQ3_K_R4.gguf \
    --no-mmap \
    -ctk q8_0 \
    -mla 3 -fa \
    -amb 1024 \
    -fmoe \
    -c 32768 \
    -ub 512 \
    --threads 88 \
    --threads-batch 128 \
    --numa numactl

Current power profile is: performance
Current THP enabled and defrag configs are:
[always] madvise never
[always] defer defer+madvise madvise never
Set numa balancing to be:
0
llama_model_loader: loaded meta data with 50 key-value pairs and 1147 tensors from /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-CPU-IQ3_K_R4.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = DeepSeek V3 0324
llama_model_loader: - kv   3:                            general.version str              = V3-0324
llama_model_loader: - kv   4:                           general.basename str              = DeepSeek
llama_model_loader: - kv   5:                         general.size_label str              = 256x21B
llama_model_loader: - kv   6:                            general.license str              = mit
llama_model_loader: - kv   7:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv   8:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv   9:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv  10:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  11:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv  12:          deepseek2.attention.head_count_kv u32              = 128
llama_model_loader: - kv  13:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  14: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  15:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  16:                          general.file_type u32              = 139
llama_model_loader: - kv  17:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  18:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  19:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  20:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  21:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  22:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  23:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  24:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  25:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  26:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  27:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  28:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  29:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  30:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  31:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  32: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  33: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  34:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  35:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  36:                      tokenizer.ggml.tokens arr[str,129280]  = ["
llama_model_loader: - kv  37:                  tokenizer.ggml.token_type arr[i32,129280]  = [3
llama_model_loader: - kv  38:                      tokenizer.ggml.merges arr[str,127741]  = ["
llama_model_loader: - kv  39:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  40:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  41:            tokenizer.ggml.padding_token_id u32              = 1
llama_model_loader: - kv  42:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  43:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  44:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  45:               general.quantization_version u32              = 2
llama_model_loader: - kv  46:                      quantize.imatrix.file str              = /mnt/raid/models/ubergarm/DeepSeek-V3...
llama_model_loader: - kv  47:                   quantize.imatrix.dataset str              = calibration_data_v5_rc.txt
llama_model_loader: - kv  48:             quantize.imatrix.entries_count i32              = 720
llama_model_loader: - kv  49:              quantize.imatrix.chunks_count i32              = 213
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type iq6_k:    1 tensors
llama_model_loader: - type q6_0_r4:   61 tensors
llama_model_loader: - type iq3_k_r4:   82 tensors
llama_model_loader: - type iq4_k_r4:   75 tensors
llama_model_loader: - type iq5_k_r4:  567 tensors
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
llm_load_print_meta: n_head_kv        = 128
llm_load_print_meta: n_rot            = 64
llm_load_print_meta: n_swa            = 0
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
llm_load_print_meta: model ftype      = IQ3_K - 3.4325 bpw
llm_load_print_meta: model params     = 672.050 B
llm_load_print_meta: model size       = 324.011 GiB (4.141 BPW) 
llm_load_print_meta: repeating layers = 322.703 GiB (4.136 BPW, 670.196 B parameters)
llm_load_print_meta: general.name     = DeepSeek V3 0324
llm_load_print_meta: BOS token        = 0 '<｜begin▁of▁sentence｜>'
llm_load_print_meta: EOS token        = 1 '<｜end▁of▁sentence｜>'
llm_load_print_meta: PAD token        = 1 '<｜end▁of▁sentence｜>'
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
llm_load_tensors: ggml ctx size =    0.47 MiB
llm_load_tensors:        CPU buffer size = 331786.93 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 3
llama_new_context_with_model: attn_max_b = 1024
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
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
llama_kv_cache_init: layer 60: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init:        CPU KV buffer size =  1166.63 MiB
llama_new_context_with_model: KV self size  = 1166.62 MiB, c^KV (q8_0): 1166.62 MiB, kv^T: not used
llama_new_context_with_model:        CPU  output buffer size =     0.49 MiB
llama_new_context_with_model:        CPU compute buffer size =  2662.01 MiB
llama_new_context_with_model: graph nodes  = 5500
llama_new_context_with_model: graph splits = 1

main: n_kv_max = 32768, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = -1, n_threads = 88, n_threads_batch = 128
```

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    4.785 |   107.01 |   12.241 |    10.46 |
|   512 |    128 |    512 |    4.790 |   106.88 |   12.809 |     9.99 |
|   512 |    128 |   1024 |    5.579 |    91.78 |   13.024 |     9.83 |
|   512 |    128 |   1536 |    5.493 |    93.20 |   13.235 |     9.67 |
|   512 |    128 |   2048 |    6.139 |    83.40 |   13.448 |     9.52 |
|   512 |    128 |   2560 |    6.585 |    77.76 |   15.234 |     8.40 |
|   512 |    128 |   3072 |    7.277 |    70.36 |   14.106 |     9.07 |
|   512 |    128 |   3584 |    7.303 |    70.11 |   14.145 |     9.05 |
|   512 |    128 |   4096 |    7.973 |    64.22 |   14.403 |     8.89 |
|   512 |    128 |   4608 |    7.829 |    65.40 |   14.331 |     8.93 |
|   512 |    128 |   5120 |    8.461 |    60.51 |   15.514 |     8.25 |
|   512 |    128 |   5632 |    8.645 |    59.22 |   15.758 |     8.12 |
|   512 |    128 |   6144 |   11.206 |    45.69 |   15.757 |     8.12 |
|   512 |    128 |   6656 |   11.238 |    45.56 |   15.882 |     8.06 |
|   512 |    128 |   7168 |   10.073 |    50.83 |   15.638 |     8.19 |
|   512 |    128 |   7680 |   10.771 |    47.53 |   16.014 |     7.99 |
|   512 |    128 |   8192 |   10.546 |    48.55 |   17.639 |     7.26 |
|   512 |    128 |   8704 |   10.799 |    47.41 |   16.658 |     7.68 |
|   512 |    128 |   9216 |   11.152 |    45.91 |   16.381 |     7.81 |
|   512 |    128 |   9728 |   11.619 |    44.07 |   16.524 |     7.75 |
|   512 |    128 |  10240 |   11.792 |    43.42 |   17.213 |     7.44 |
|   512 |    128 |  10752 |   12.311 |    41.59 |   17.267 |     7.41 |
|   512 |    128 |  11264 |   13.110 |    39.05 |   18.420 |     6.95 |
|   512 |    128 |  11776 |   13.198 |    38.79 |   18.808 |     6.81 |
|   512 |    128 |  12288 |   13.695 |    37.39 |   19.295 |     6.63 |
|   512 |    128 |  12800 |   14.077 |    36.37 |   18.512 |     6.91 |
|   512 |    128 |  13312 |   14.542 |    35.21 |   18.896 |     6.77 |
|   512 |    128 |  13824 |   14.826 |    34.53 |   19.688 |     6.50 |
|   512 |    128 |  14336 |   14.957 |    34.23 |   19.614 |     6.53 |
|   512 |    128 |  14848 |   15.359 |    33.33 |   20.175 |     6.34 |
|   512 |    128 |  15360 |   15.671 |    32.67 |   21.683 |     5.90 |
|   512 |    128 |  15872 |   16.131 |    31.74 |   21.967 |     5.83 |
|   512 |    128 |  16384 |   16.073 |    31.85 |   22.157 |     5.78 |
|   512 |    128 |  16896 |   17.251 |    29.68 |   22.368 |     5.72 |
|   512 |    128 |  17408 |   17.549 |    29.17 |   22.054 |     5.80 |
|   512 |    128 |  17920 |   17.088 |    29.96 |   22.151 |     5.78 |
|   512 |    128 |  18432 |   17.419 |    29.39 |   21.529 |     5.95 |
|   512 |    128 |  18944 |   17.825 |    28.72 |   22.387 |     5.72 |
|   512 |    128 |  19456 |   18.189 |    28.15 |   21.878 |     5.85 |
|   512 |    128 |  19968 |   19.256 |    26.59 |   21.790 |     5.87 |
|   512 |    128 |  20480 |   19.052 |    26.87 |   23.344 |     5.48 |
|   512 |    128 |  20992 |   19.282 |    26.55 |   22.052 |     5.80 |
|   512 |    128 |  21504 |   19.819 |    25.83 |   24.614 |     5.20 |
|   512 |    128 |  22016 |   19.986 |    25.62 |   24.630 |     5.20 |
|   512 |    128 |  22528 |   20.422 |    25.07 |   25.011 |     5.12 |
|   512 |    128 |  23040 |   20.641 |    24.81 |   25.628 |     4.99 |
|   512 |    128 |  23552 |   20.650 |    24.79 |   26.092 |     4.91 |
|   512 |    128 |  24064 |   21.313 |    24.02 |   26.216 |     4.88 |
|   512 |    128 |  24576 |   21.688 |    23.61 |   26.284 |     4.87 |
|   512 |    128 |  25088 |   21.881 |    23.40 |   24.090 |     5.31 |
|   512 |    128 |  25600 |   22.037 |    23.23 |   26.860 |     4.77 |
|   512 |    128 |  26112 |   22.366 |    22.89 |   26.609 |     4.81 |
|   512 |    128 |  26624 |   23.119 |    22.15 |   26.998 |     4.74 |
|   512 |    128 |  27136 |   23.189 |    22.08 |   26.720 |     4.79 |
|   512 |    128 |  27648 |   23.747 |    21.56 |   27.567 |     4.64 |
|   512 |    128 |  28160 |   24.516 |    20.88 |   27.943 |     4.58 |
|   512 |    128 |  28672 |   24.567 |    20.84 |   28.062 |     4.56 |
|   512 |    128 |  29184 |   25.295 |    20.24 |   28.517 |     4.49 |
|   512 |    128 |  29696 |   25.251 |    20.28 |   28.897 |     4.43 |
|   512 |    128 |  30208 |   25.564 |    20.03 |   28.628 |     4.47 |
|   512 |    128 |  30720 |   26.003 |    19.69 |   27.277 |     4.69 |
|   512 |    128 |  31232 |   26.974 |    18.98 |   29.181 |     4.39 |
|   512 |    128 |  31744 |   26.174 |    19.56 |   28.908 |     4.43 |
|   512 |    128 |  32256 |   26.579 |    19.26 |   29.200 |     4.38 |


</details>

---

👤 **ikawrakow** commented the **2025-04-06** at **16:30:25**:<br>

Thank you for this.

So this does not explain it either.

It is hard to make progress without me being able to experiment on the actual big iron machine. I was promised some funding to rent a big iron cloud instance to sort out performance issues and look into the NUMA situation, but the funding hasn't materialized yet. Btw, are you renting the Xeon 6980P, or did you buy it? If you are renting, where did you rent it?

---

👤 **ubergarm** commented the **2025-04-06** at **17:53:17**:<br>

> Thank you for this.
> 
> So this does not explain it either.

Yeah, I ran a couple more tests against `main@ec84855c` (not this PR) reducing the number of threads which *does* move the peaks. This 6980P is a strange beast, given a single CPU socket has three physical compute dies with 43+43+42 physical cores currently configured into a single NUMA node (BIOS `SNC=Disable`). SMT is currently enabled fwiw...

#### pp
![performance_comparison_pp-03](https://github.com/user-attachments/assets/30f1693b-6fba-4c7a-9293-deeb4fc9c75b)

#### tg
![performance_comparison_tg-03](https://github.com/user-attachments/assets/7d63ebd0-a4a3-4d87-9603-e1ad3d20cb80)
 
> If you are renting, where did you rent it?

No, I've been "fun-employed" for over a year now just hacking around on whatever projects interest me, so trying to minimize costs. I used to work for a cloud provider on the east coast USA, and randomly lucked into access on the remote hardware. I believe [this YT video may be discussing the machine on which I'm testing](https://youtu.be/_uKxEkgGu9g?t=105) (or at least something similar).

Kind of a long story, but just under a year ago I built a new local rig for ai around the Intel i9-14900k, however within a couple months it had fried itself. I learned more about that CPUs issues hanging out at level1techs.com forum. I did a lot of benchmarks on the replacement 9950x rig I built and met folks on the forum. This eventually led to me getting more involved and having some limited access to hardware for testing and benchmarking.

It would be pretty amazing to actually make some money to do this stuff in which I'm interested haha... My impression is at least in the USA folks with money are tending towards:

1. using one of the big API providers
2. building out racks of servers configured with 8x H/B 100/200 CUDA GPU nodes and probably looking at SGLang, vLLM, or whatever big VRAM optimized solutions

Though my guess is in other countries like China and India, there are more use cases for hybrid CPU+GPU systems that can serve smaller numbers of users with more modest hardware. Amusingly this situation more closely matches many American "home lab enthusiasts" as well who are watching llama.cpp, ktransformers, and your ik_llama.cpp

Anyway, just rambling, I'm happy to run some tests on various PRs as time allows just at me with the details.

Thanks!

---

👤 **ikawrakow** commented the **2025-05-04** at **06:18:51**:<br>

Doesn't look like it is useful, closing.