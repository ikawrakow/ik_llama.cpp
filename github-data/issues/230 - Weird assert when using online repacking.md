### 📝 [#230](https://github.com/ikawrakow/ik_llama.cpp/issues/230) - Weird assert when using online repacking

| **Author** | `pt13762104` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-02-24 |
| **Updated** | 2025-02-24 |

---

#### Description

### What happened?

A weird error happened when I tried to use runtime repacking: `GGML_ASSERT(nrc_x%8 == 0) failed`.

### Name and Version

version: 3571 (ac1d259b)

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
| model                          |       size |     params | backend    | threads | rtr |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --: | ------------: | ---------------: |
llama_model_loader: loaded meta data with 42 key-value pairs and 377 tensors from /dev/shm/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.name str              = DeepSeek-Coder-V2-Lite-Instruct
llama_model_loader: - kv   2:                      deepseek2.block_count u32              = 27
llama_model_loader: - kv   3:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv   4:                 deepseek2.embedding_length u32              = 2048
llama_model_loader: - kv   5:              deepseek2.feed_forward_length u32              = 10944
llama_model_loader: - kv   6:             deepseek2.attention.head_count u32              = 16
llama_model_loader: - kv   7:          deepseek2.attention.head_count_kv u32              = 16
llama_model_loader: - kv   8:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv   9: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  10:                deepseek2.expert_used_count u32              = 6
llama_model_loader: - kv  11:                          general.file_type u32              = 15
llama_model_loader: - kv  12:        deepseek2.leading_dense_block_count u32              = 1
llama_model_loader: - kv  13:                       deepseek2.vocab_size u32              = 102400
llama_model_loader: - kv  14:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  15:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  16:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  17:       deepseek2.expert_feed_forward_length u32              = 1408
llama_model_loader: - kv  18:                     deepseek2.expert_count u32              = 64
llama_model_loader: - kv  19:              deepseek2.expert_shared_count u32              = 2
llama_model_loader: - kv  20:             deepseek2.expert_weights_scale f32              = 1.000000
llama_model_loader: - kv  21:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  22:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  23:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  24: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  25: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.070700
llama_model_loader: - kv  26:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  27:                         tokenizer.ggml.pre str              = deepseek-llm
llama_model_loader: - kv  28:                      tokenizer.ggml.tokens arr[str,102400]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  29:                  tokenizer.ggml.token_type arr[i32,102400]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  30:                      tokenizer.ggml.merges arr[str,99757]   = ["Ġ Ġ", "Ġ t", "Ġ a", "i n", "h e...
llama_model_loader: - kv  31:                tokenizer.ggml.bos_token_id u32              = 100000
llama_model_loader: - kv  32:                tokenizer.ggml.eos_token_id u32              = 100001
llama_model_loader: - kv  33:            tokenizer.ggml.padding_token_id u32              = 100001
llama_model_loader: - kv  34:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  35:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  36:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  37:               general.quantization_version u32              = 2
llama_model_loader: - kv  38:                      quantize.imatrix.file str              = /models/DeepSeek-Coder-V2-Lite-Instru...
llama_model_loader: - kv  39:                   quantize.imatrix.dataset str              = /training_data/calibration_datav3.txt
llama_model_loader: - kv  40:             quantize.imatrix.entries_count i32              = 293
llama_model_loader: - kv  41:              quantize.imatrix.chunks_count i32              = 139
llama_model_loader: - type  f32:  108 tensors
llama_model_loader: - type q5_0:   14 tensors
llama_model_loader: - type q8_0:   13 tensors
llama_model_loader: - type q4_K:  229 tensors
llama_model_loader: - type q6_K:   13 tensors
llm_load_vocab: special tokens cache size = 2400
llm_load_vocab: token to piece cache size = 0.6661 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = deepseek2
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 102400
llm_load_print_meta: n_merges         = 99757
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 163840
llm_load_print_meta: n_embd           = 2048
llm_load_print_meta: n_layer          = 27
llm_load_print_meta: n_head           = 16
llm_load_print_meta: n_head_kv        = 16
llm_load_print_meta: n_rot            = 64
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_embd_head_k    = 192
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: n_embd_k_gqa     = 3072
llm_load_print_meta: n_embd_v_gqa     = 2048
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 10944
llm_load_print_meta: n_expert         = 64
llm_load_print_meta: n_expert_used    = 6
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
llm_load_print_meta: model type       = 16B
llm_load_print_meta: model ftype      = Q4_K - Medium
llm_load_print_meta: model params     = 15.706 B
llm_load_print_meta: model size       = 9.649 GiB (5.277 BPW) 
llm_load_print_meta: repeating layers = 9.379 GiB (5.270 BPW, 15.287 B parameters)
llm_load_print_meta: general.name     = DeepSeek-Coder-V2-Lite-Instruct
llm_load_print_meta: BOS token        = 100000 '<｜begin▁of▁sentence｜>'
llm_load_print_meta: EOS token        = 100001 '<｜end▁of▁sentence｜>'
llm_load_print_meta: PAD token        = 100001 '<｜end▁of▁sentence｜>'
llm_load_print_meta: LF token         = 126 'Ä'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_layer_dense_lead   = 1
llm_load_print_meta: n_lora_q             = 0
llm_load_print_meta: n_lora_kv            = 512
llm_load_print_meta: n_ff_exp             = 1408
llm_load_print_meta: n_expert_shared      = 2
llm_load_print_meta: expert_weights_scale = 1.0
llm_load_print_meta: expert_weights_norm  = 0
llm_load_print_meta: expert_gating_func   = softmax
llm_load_print_meta: rope_yarn_log_mul    = 0.0707
llm_load_tensors: ggml ctx size =    0.16 MiB
llm_load_tensors:        CPU buffer size =  9880.47 MiB
.....................................................................................
============ Repacked 268 tensors
llama_new_context_with_model: n_ctx      = 512
llama_new_context_with_model: n_batch    = 512
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: fused_moe  = 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init:        CPU KV buffer size =   135.00 MiB
llama_new_context_with_model: KV self size  =  135.00 MiB, K (f16):   81.00 MiB, V (f16):   54.00 MiB
llama_new_context_with_model:        CPU  output buffer size =     0.39 MiB
llama_new_context_with_model:        CPU compute buffer size =   204.00 MiB
llama_new_context_with_model: graph nodes  = 1474
llama_new_context_with_model: graph splits = 1
/root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: GGML_ASSERT(nrc_x%8 == 0) failed/root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: /root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: /root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: /root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: /root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: /root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: /root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: GGML_ASSERT(nrc_x%8 == 0) failed/root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: GGML_ASSERT(nrc_x%8 == 0) failed
/root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: GGML_ASSERT(nrc_x%8 == 0) failed
/root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: GGML_ASSERT(nrc_x%8 == 0) failed
/root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: GGML_ASSERT(nrc_x%8 == 0) failed
/root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: GGML_ASSERT(nrc_x%8 == 0) failed
/root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: GGML_ASSERT(nrc_x%8 == 0) failed
/root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: GGML_ASSERT(nrc_x%8 == 0) failed
/root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: GGML_ASSERT(nrc_x%8 == 0) failed
/root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: GGML_ASSERT(nrc_x%8 == 0) failed
/root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: GGML_ASSERT(nrc_x%8 == 0) failed
/root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: GGML_ASSERT(nrc_x%8 == 0) failed
/root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: GGML_ASSERT(nrc_x%8 == 0) failed
/root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: GGML_ASSERT(nrc_x%8 == 0) failed
/root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: GGML_ASSERT(nrc_x%8 == 0) failed
/root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: GGML_ASSERT(nrc_x%8 == 0) failed
/root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: GGML_ASSERT(nrc_x%8 == 0) failed
/root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: GGML_ASSERT(nrc_x%8 == 0) failed
/root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: GGML_ASSERT(nrc_x%8 == 0) failed
/root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: GGML_ASSERT(nrc_x%8 == 0) failed
/root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: GGML_ASSERT(nrc_x%8 == 0) failed
/root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: GGML_ASSERT(nrc_x%8 == 0) failed
/root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: GGML_ASSERT(nrc_x%8 == 0) failed
/root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: GGML_ASSERT(nrc_x%8 == 0) failed
/root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: GGML_ASSERT(nrc_x%8 == 0) failed
/root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: GGML_ASSERT(nrc_x%8 == 0) failed
/root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: GGML_ASSERT(nrc_x%8 == 0) failed
/root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: GGML_ASSERT(nrc_x%8 == 0) failed
/root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: GGML_ASSERT(nrc_x%8 == 0) failed
/root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: GGML_ASSERT(nrc_x%8 == 0) failed
/root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: GGML_ASSERT(nrc_x%8 == 0) failed
/root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: GGML_ASSERT(nrc_x%8 == 0) failed
/root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: GGML_ASSERT(nrc_x%8 == 0) failed
/root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: GGML_ASSERT(nrc_x%8 == 0) failed
/root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: GGML_ASSERT(nrc_x%8 == 0) failed
GGML_ASSERT(nrc_x%8 == 0) failed
GGML_ASSERT(nrc_x%8 == 0) failed
/root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: GGML_ASSERT(nrc_x%8 == 0) failed
GGML_ASSERT(nrc_x%8 == 0) failed
GGML_ASSERT(nrc_x%8 == 0) failed
GGML_ASSERT(nrc_x%8 == 0) failed
GGML_ASSERT(nrc_x%8 == 0) failed
/root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: GGML_ASSERT(nrc_x%8 == 0) failed
/root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: GGML_ASSERT(nrc_x%8 == 0) failed

/root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: GGML_ASSERT(nrc_x%8 == 0) failed

/root/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:4065: GGML_ASSERT(nrc_x%8 == 0) failed
/root/ik_llama.cpp/build/ggml/src/libggml.so(+0x1b3b5)[0x7e716c2143b5]
/root/ik_llama.cpp/build/ggml/src/libggml.so(ggml_abort+0x136)[0x7e716c216266]
/root/ik_llama.cpp/build/ggml/src/libggml.so(+0x1a1cfd)[0x7e716c39acfd]
/root/ik_llama.cpp/build/ggml/src/libggml.so(iqk_mul_mat_moe+0x55a)[0x7e716c5afd3a]
/root/ik_llama.cpp/build/ggml/src/libggml.so(+0x32b98)[0x7e716c22bb98]
/root/ik_llama.cpp/build/ggml/src/libggml.so(+0x588b9)[0x7e716c2518b9]
/root/ik_llama.cpp/build/ggml/src/libggml.so(+0x58a55)[0x7e716c251a55]
/home/linuxbrew/.linuxbrew/lib/gcc/current/libgomp.so.1(+0x227ce)[0x7e716bc027ce]
/lib/x86_64-linux-gnu/libc.so.6(+0x891c4)[0x7e716bcc11c4]
/lib/x86_64-linux-gnu/libc.so.6(__clone+0x40)[0x7e716bd40ac0]
Aborted (core dumped)
```

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-02-24** at **06:16:16**:<br>

Dose #231 fix it?

---

👤 **pt13762104** commented the **2025-02-24** at **07:20:49**:<br>

It's working now, thank you!
```
| model                          |       size |     params | backend    | threads |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | ---------------: |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | CPU        |      48 |         pp512 |   303.36 ± 29.58 |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | CPU        |      48 |         tg128 |     19.92 ± 0.07 |

build: 4f2cfd6e (3572)
| model                          |       size |     params | backend    | threads | rtr |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --: | ------------: | ---------------: |
============ Repacked 268 tensors
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | CPU        |      48 |   1 |         pp512 |   393.53 ± 52.69 |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | CPU        |      48 |   1 |         tg128 |     21.71 ± 0.16 |

build: 4f2cfd6e (3572)
```

---

👤 **pt13762104** commented the **2025-02-24** at **07:20:49**:<br>

It's working now, thank you!
```
| model                          |       size |     params | backend    | threads |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | ---------------: |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | CPU        |      48 |         pp512 |   303.36 ± 29.58 |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | CPU        |      48 |         tg128 |     19.92 ± 0.07 |

build: 4f2cfd6e (3572)
| model                          |       size |     params | backend    | threads | rtr |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --: | ------------: | ---------------: |
============ Repacked 268 tensors
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | CPU        |      48 |   1 |         pp512 |   393.53 ± 52.69 |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | CPU        |      48 |   1 |         tg128 |     21.71 ± 0.16 |

build: 4f2cfd6e (3572)```

---

👤 **ikawrakow** commented the **2025-02-24** at **07:29:39**:<br>

What is the CPU for these benchmarks? Have you tried running TG with fewer threads?

---

👤 **pt13762104** commented the **2025-02-24** at **08:15:13**:<br>

No, I didn't try. Also it's 2x Xeon 24-core (unknown model name) from Kaggle.

---

👤 **pt13762104** commented the **2025-02-24** at **08:15:13**:<br>

No, I didn't try. Also it's 2x Xeon (unknown model name) from Kaggle.