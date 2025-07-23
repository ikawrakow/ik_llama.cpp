### 🐛 [#539](https://github.com/ikawrakow/ik_llama.cpp/issues/539) - Bug: garbage output

| **Author** | `jagusztinl` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-19 |
| **Updated** | 2025-06-26 |

---

#### Description

### What happened?

Please help, tried several models but there is no meaningful outut (cli and server is the same, with or w/o -rtr is the same):

@gpt:~/models$ ../ik_llama.cpp//build/bin/llama-cli  -m gemma-3-27b-it-Q4_0.gguf   --prompt "What is the meaning of life?"
Log start
main: build = 3751 (8b3002bb)
main: built with cc (Ubuntu 14.2.0-4ubuntu2~24.04) 14.2.0 for aarch64-linux-gnu
main: seed  = 1750314253
llama_model_loader: loaded meta data with 40 key-value pairs and 808 tensors from gemma-3-27b-it-Q4_0.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = gemma3
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Gemma-3-27B-It
llama_model_loader: - kv   3:                           general.finetune str              = it
llama_model_loader: - kv   4:                           general.basename str              = Gemma-3-27B-It
llama_model_loader: - kv   5:                       general.quantized_by str              = Unsloth
llama_model_loader: - kv   6:                         general.size_label str              = 27B
llama_model_loader: - kv   7:                           general.repo_url str              = https://huggingface.co/unsloth
llama_model_loader: - kv   8:                      gemma3.context_length u32              = 131072
llama_model_loader: - kv   9:                    gemma3.embedding_length u32              = 5376
llama_model_loader: - kv  10:                         gemma3.block_count u32              = 62
llama_model_loader: - kv  11:                 gemma3.feed_forward_length u32              = 21504
llama_model_loader: - kv  12:                gemma3.attention.head_count u32              = 32
llama_model_loader: - kv  13:    gemma3.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  14:                gemma3.attention.key_length u32              = 128
llama_model_loader: - kv  15:              gemma3.attention.value_length u32              = 128
llama_model_loader: - kv  16:                      gemma3.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  17:            gemma3.attention.sliding_window u32              = 1024
llama_model_loader: - kv  18:             gemma3.attention.head_count_kv u32              = 16
llama_model_loader: - kv  19:                   gemma3.rope.scaling.type str              = linear
llama_model_loader: - kv  20:                 gemma3.rope.scaling.factor f32              = 8.000000
llama_model_loader: - kv  21:                       tokenizer.ggml.model str              = llama
llama_model_loader: - kv  22:                         tokenizer.ggml.pre str              = default
llama_model_loader: - kv  23:                      tokenizer.ggml.tokens arr[str,262208]  = ["<pad>", "<eos>", "<bos>", "<unk>", ...
llama_model_loader: - kv  24:                      tokenizer.ggml.scores arr[f32,262208]  = [-1000.000000, -1000.000000, -1000.00...
llama_model_loader: - kv  25:                  tokenizer.ggml.token_type arr[i32,262208]  = [3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3, ...
llama_model_loader: - kv  26:                tokenizer.ggml.bos_token_id u32              = 2
llama_model_loader: - kv  27:                tokenizer.ggml.eos_token_id u32              = 106
llama_model_loader: - kv  28:            tokenizer.ggml.unknown_token_id u32              = 3
llama_model_loader: - kv  29:            tokenizer.ggml.padding_token_id u32              = 0
llama_model_loader: - kv  30:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  31:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  32:                    tokenizer.chat_template str              = {{ bos_token }}\n{%- if messages[0]['r...
llama_model_loader: - kv  33:            tokenizer.ggml.add_space_prefix bool             = false
llama_model_loader: - kv  34:               general.quantization_version u32              = 2
llama_model_loader: - kv  35:                          general.file_type u32              = 2
llama_model_loader: - kv  36:                      quantize.imatrix.file str              = gemma-3-27b-it-GGUF/imatrix_unsloth.dat
llama_model_loader: - kv  37:                   quantize.imatrix.dataset str              = unsloth_calibration_gemma-3-27b-it.txt
llama_model_loader: - kv  38:             quantize.imatrix.entries_count i32              = 434
llama_model_loader: - kv  39:              quantize.imatrix.chunks_count i32              = 663
llama_model_loader: - type  f32:  373 tensors
llama_model_loader: - type q4_0:  427 tensors
llama_model_loader: - type q4_1:    7 tensors
llama_model_loader: - type q6_K:    1 tensors
llm_load_vocab: special tokens cache size = 6415
llm_load_vocab: token to piece cache size = 1.9446 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = gemma3
llm_load_print_meta: vocab type       = SPM
llm_load_print_meta: n_vocab          = 262208
llm_load_print_meta: n_merges         = 0
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 131072
llm_load_print_meta: n_embd           = 5376
llm_load_print_meta: n_layer          = 62
llm_load_print_meta: n_head           = 32
llm_load_print_meta: n_head_kv        = 16
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 1024
llm_load_print_meta: n_swa_pattern    = 6
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 2
llm_load_print_meta: n_embd_k_gqa     = 2048
llm_load_print_meta: n_embd_v_gqa     = 2048
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 21504
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 2
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 1000000.0
llm_load_print_meta: freq_scale_train = 0.125
llm_load_print_meta: n_ctx_orig_yarn  = 131072
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = 27B
llm_load_print_meta: model ftype      = Q4_0
llm_load_print_meta: model params     = 27.009 B
llm_load_print_meta: model size       = 14.539 GiB (4.624 BPW)
llm_load_print_meta: general.name     = Gemma-3-27B-It
llm_load_print_meta: BOS token        = 2 '<bos>'
llm_load_print_meta: EOS token        = 106 '<end_of_turn>'
llm_load_print_meta: UNK token        = 3 '<unk>'
llm_load_print_meta: PAD token        = 0 '<pad>'
llm_load_print_meta: LF token         = 248 '<0x0A>'
llm_load_print_meta: EOT token        = 106 '<end_of_turn>'
llm_load_print_meta: max token length = 48
llm_load_tensors: ggml ctx size =    0.35 MiB
llm_load_tensors:        CPU buffer size = 14888.20 MiB
.........................................................................................
llama_new_context_with_model: n_ctx      = 131072
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 0
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 0.125
llama_kv_cache_init:        CPU KV buffer size = 63488.00 MiB
llama_new_context_with_model: KV self size  = 63488.00 MiB, K (f16): 31744.00 MiB, V (f16): 31744.00 MiB
llama_new_context_with_model:        CPU  output buffer size =     1.00 MiB
llama_new_context_with_model:        CPU compute buffer size =  8743.51 MiB
llama_new_context_with_model: graph nodes  = 2052
llama_new_context_with_model: graph splits = 1

system_info: n_threads = 64 / 64 | AVX = 0 | AVX_VNNI = 0 | AVX2 = 0 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 0 | NEON = 1 | SVE = 0 | ARM_FMA = 1 | F16C = 0 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 0 | SSSE3 = 0 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 |
sampling:
        repeat_last_n = 64, repeat_penalty = 1.000, frequency_penalty = 0.000, presence_penalty = 0.000
        top_k = 40, tfs_z = 1.000, top_p = 0.950, min_p = 0.050, typical_p = 1.000, temp = 0.800
        mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000
        xtc_probability = 0.000, xtc_threshold = 1.000, top_n_sigma = 0.000
sampling order:
CFG -> Penalties -> top_k -> tfs_z -> typical_p -> top_p -> min_p -> top_n_sigma -> temperature
generate: n_ctx = 131072, n_batch = 2048, n_predict = -1, n_keep = 1


What is the meaning of life?[multimodal][multimodal][multimodal][multimodal][multimodal]


OR

alerant@gpt:~/models$ ../ik_llama.cpp//build/bin/llama-cli  -m Qwen   --prompt "What is the meaning of life?"
Qwen2.5-Coder-32B-Instruct-Q4_0.gguf  Qwen3-32B-Q4_0.gguf
alerant@gpt:~/models$ ../ik_llama.cpp//build/bin/llama-cli  -m Qwen3-32B-Q4_0.gguf   --prompt "What is the meaning of life?"
Log start
main: build = 3751 (8b3002bb)
main: built with cc (Ubuntu 14.2.0-4ubuntu2~24.04) 14.2.0 for aarch64-linux-gnu
main: seed  = 1750314509
llama_model_loader: loaded meta data with 32 key-value pairs and 707 tensors from Qwen3-32B-Q4_0.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3-32B
llama_model_loader: - kv   3:                           general.basename str              = Qwen3-32B
llama_model_loader: - kv   4:                       general.quantized_by str              = Unsloth
llama_model_loader: - kv   5:                         general.size_label str              = 32B
llama_model_loader: - kv   6:                           general.repo_url str              = https://huggingface.co/unsloth
llama_model_loader: - kv   7:                          qwen3.block_count u32              = 64
llama_model_loader: - kv   8:                       qwen3.context_length u32              = 40960
llama_model_loader: - kv   9:                     qwen3.embedding_length u32              = 5120
llama_model_loader: - kv  10:                  qwen3.feed_forward_length u32              = 25600
llama_model_loader: - kv  11:                 qwen3.attention.head_count u32              = 64
llama_model_loader: - kv  12:              qwen3.attention.head_count_kv u32              = 8
llama_model_loader: - kv  13:                       qwen3.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  14:     qwen3.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  15:                 qwen3.attention.key_length u32              = 128
llama_model_loader: - kv  16:               qwen3.attention.value_length u32              = 128
llama_model_loader: - kv  17:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  18:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  19:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  20:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  21:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv  22:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  23:            tokenizer.ggml.padding_token_id u32              = 151654
llama_model_loader: - kv  24:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  25:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - kv  26:               general.quantization_version u32              = 2
llama_model_loader: - kv  27:                          general.file_type u32              = 2
llama_model_loader: - kv  28:                      quantize.imatrix.file str              = Qwen3-32B-GGUF/imatrix_unsloth.dat
llama_model_loader: - kv  29:                   quantize.imatrix.dataset str              = unsloth_calibration_Qwen3-32B.txt
llama_model_loader: - kv  30:             quantize.imatrix.entries_count i32              = 448
llama_model_loader: - kv  31:              quantize.imatrix.chunks_count i32              = 685
llama_model_loader: - type  f32:  257 tensors
llama_model_loader: - type q4_0:  441 tensors
llama_model_loader: - type q4_1:    8 tensors
llama_model_loader: - type q6_K:    1 tensors
llm_load_vocab: special tokens cache size = 26
llm_load_vocab: token to piece cache size = 0.9311 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = qwen3
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 151936
llm_load_print_meta: n_merges         = 151387
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 40960
llm_load_print_meta: n_embd           = 5120
llm_load_print_meta: n_layer          = 64
llm_load_print_meta: n_head           = 64
llm_load_print_meta: n_head_kv        = 8
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 8
llm_load_print_meta: n_embd_k_gqa     = 1024
llm_load_print_meta: n_embd_v_gqa     = 1024
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 25600
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
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
llm_load_print_meta: model ftype      = Q4_0
llm_load_print_meta: model params     = 32.762 B
llm_load_print_meta: model size       = 17.413 GiB (4.566 BPW)
llm_load_print_meta: repeating layers = 16.411 GiB (4.517 BPW, 31.206 B parameters)
llm_load_print_meta: general.name     = Qwen3-32B
llm_load_print_meta: BOS token        = 11 ','
llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
llm_load_print_meta: PAD token        = 151654 '<|vision_pad|>'
llm_load_print_meta: LF token         = 148848 'ÄĬ'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_tensors: ggml ctx size =    0.32 MiB
llm_load_tensors:        CPU buffer size = 17830.96 MiB
.................................................................................................
llama_new_context_with_model: n_ctx      = 40960
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 0
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:        CPU KV buffer size = 10240.00 MiB
llama_new_context_with_model: KV self size  = 10240.00 MiB, K (f16): 5120.00 MiB, V (f16): 5120.00 MiB
llama_new_context_with_model:        CPU  output buffer size =     0.58 MiB
llama_new_context_with_model:        CPU compute buffer size =  5252.01 MiB
llama_new_context_with_model: graph nodes  = 1989
llama_new_context_with_model: graph splits = 1

system_info: n_threads = 64 / 64 | AVX = 0 | AVX_VNNI = 0 | AVX2 = 0 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 0 | NEON = 1 | SVE = 0 | ARM_FMA = 1 | F16C = 0 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 0 | SSSE3 = 0 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 |
sampling:
        repeat_last_n = 64, repeat_penalty = 1.000, frequency_penalty = 0.000, presence_penalty = 0.000
        top_k = 40, tfs_z = 1.000, top_p = 0.950, min_p = 0.050, typical_p = 1.000, temp = 0.800
        mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000
        xtc_probability = 0.000, xtc_threshold = 1.000, top_n_sigma = 0.000
sampling order:
CFG -> Penalties -> top_k -> tfs_z -> typical_p -> top_p -> min_p -> top_n_sigma -> temperature
generate: n_ctx = 40960, n_batch = 2048, n_predict = -1, n_keep = 0


What is the meaning of life?*:F+=@*GB&-4%G0'B$4HF;@E(H(C6;()@:%'8"4<-HC.&$G>)$2)536.).C5346=D=6;C41AD@BD&6D';-.:G1+;=;C!+7;A>!+:8DG466)+9#:<99)3



### Name and Version

version: 3751 (8b3002bb)
built with cc (Ubuntu 14.2.0-4ubuntu2~24.04) 14.2.0 for aarch64-linux-gnu


### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell

```

---

#### 💬 Conversation

👤 **jagusztinl** commented the **2025-06-19** at **08:40:53**:<br>

I tried with IQ4_XS models (gemma) it works perfectly, maybe Q4_0 is bad. But with IQ4_XS and -rtr garbage again. What I miss?

(venv) alerant@gpt:~/models$ ../ik_llama.cpp//build/bin/llama-cli  -m   gemma-3-27b-it-IQ4_XS.gguf  -rtr  --prompt "What is the meaning of life? In english please"
Log start
main: build = 3751 (8b3002bb)
main: built with cc (Ubuntu 14.2.0-4ubuntu2~24.04) 14.2.0 for aarch64-linux-gnu
main: seed  = 1750322313
llama_model_loader: loaded meta data with 40 key-value pairs and 808 tensors from gemma-3-27b-it-IQ4_XS.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = gemma3
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Gemma-3-27B-It
llama_model_loader: - kv   3:                           general.finetune str              = it
llama_model_loader: - kv   4:                           general.basename str              = Gemma-3-27B-It
llama_model_loader: - kv   5:                       general.quantized_by str              = Unsloth
llama_model_loader: - kv   6:                         general.size_label str              = 27B
llama_model_loader: - kv   7:                           general.repo_url str              = https://huggingface.co/unsloth
llama_model_loader: - kv   8:                      gemma3.context_length u32              = 131072
llama_model_loader: - kv   9:                    gemma3.embedding_length u32              = 5376
llama_model_loader: - kv  10:                         gemma3.block_count u32              = 62
llama_model_loader: - kv  11:                 gemma3.feed_forward_length u32              = 21504
llama_model_loader: - kv  12:                gemma3.attention.head_count u32              = 32
llama_model_loader: - kv  13:    gemma3.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  14:                gemma3.attention.key_length u32              = 128
llama_model_loader: - kv  15:              gemma3.attention.value_length u32              = 128
llama_model_loader: - kv  16:                      gemma3.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  17:            gemma3.attention.sliding_window u32              = 1024
llama_model_loader: - kv  18:             gemma3.attention.head_count_kv u32              = 16
llama_model_loader: - kv  19:                   gemma3.rope.scaling.type str              = linear
llama_model_loader: - kv  20:                 gemma3.rope.scaling.factor f32              = 8.000000
llama_model_loader: - kv  21:                       tokenizer.ggml.model str              = llama
llama_model_loader: - kv  22:                         tokenizer.ggml.pre str              = default
llama_model_loader: - kv  23:                      tokenizer.ggml.tokens arr[str,262208]  = ["<pad>", "<eos>", "<bos>", "<unk>", ...
llama_model_loader: - kv  24:                      tokenizer.ggml.scores arr[f32,262208]  = [-1000.000000, -1000.000000, -1000.00...
llama_model_loader: - kv  25:                  tokenizer.ggml.token_type arr[i32,262208]  = [3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3, ...
llama_model_loader: - kv  26:                tokenizer.ggml.bos_token_id u32              = 2
llama_model_loader: - kv  27:                tokenizer.ggml.eos_token_id u32              = 106
llama_model_loader: - kv  28:            tokenizer.ggml.unknown_token_id u32              = 3
llama_model_loader: - kv  29:            tokenizer.ggml.padding_token_id u32              = 0
llama_model_loader: - kv  30:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  31:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  32:                    tokenizer.chat_template str              = {{ bos_token }}\n{%- if messages[0]['r...
llama_model_loader: - kv  33:            tokenizer.ggml.add_space_prefix bool             = false
llama_model_loader: - kv  34:               general.quantization_version u32              = 2
llama_model_loader: - kv  35:                          general.file_type u32              = 30
llama_model_loader: - kv  36:                      quantize.imatrix.file str              = gemma-3-27b-it-GGUF/imatrix_unsloth.dat
llama_model_loader: - kv  37:                   quantize.imatrix.dataset str              = unsloth_calibration_gemma-3-27b-it.txt
llama_model_loader: - kv  38:             quantize.imatrix.entries_count i32              = 434
llama_model_loader: - kv  39:              quantize.imatrix.chunks_count i32              = 663
llama_model_loader: - type  f32:  373 tensors
llama_model_loader: - type q6_K:    1 tensors
llama_model_loader: - type iq4_xs:  434 tensors
llm_load_vocab: special tokens cache size = 6415
llm_load_vocab: token to piece cache size = 1.9446 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = gemma3
llm_load_print_meta: vocab type       = SPM
llm_load_print_meta: n_vocab          = 262208
llm_load_print_meta: n_merges         = 0
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 131072
llm_load_print_meta: n_embd           = 5376
llm_load_print_meta: n_layer          = 62
llm_load_print_meta: n_head           = 32
llm_load_print_meta: n_head_kv        = 16
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 1024
llm_load_print_meta: n_swa_pattern    = 6
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 2
llm_load_print_meta: n_embd_k_gqa     = 2048
llm_load_print_meta: n_embd_v_gqa     = 2048
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 21504
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 2
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 1000000.0
llm_load_print_meta: freq_scale_train = 0.125
llm_load_print_meta: n_ctx_orig_yarn  = 131072
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = 27B
llm_load_print_meta: model ftype      = IQ4_XS - 4.25 bpw
llm_load_print_meta: model params     = 27.009 B
llm_load_print_meta: model size       = 13.747 GiB (4.372 BPW)
llm_load_print_meta: general.name     = Gemma-3-27B-It
llm_load_print_meta: BOS token        = 2 '<bos>'
llm_load_print_meta: EOS token        = 106 '<end_of_turn>'
llm_load_print_meta: UNK token        = 3 '<unk>'
llm_load_print_meta: PAD token        = 0 '<pad>'
llm_load_print_meta: LF token         = 248 '<0x0A>'
llm_load_print_meta: EOT token        = 106 '<end_of_turn>'
llm_load_print_meta: max token length = 48
llm_load_tensors: ggml ctx size =    0.35 MiB
llm_load_tensors:        CPU buffer size = 15179.85 MiB
........................................................................................
============ Repacked 434 tensors
llama_new_context_with_model: n_ctx      = 131072
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 0
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 0.125
llama_kv_cache_init:        CPU KV buffer size = 63488.00 MiB
llama_new_context_with_model: KV self size  = 63488.00 MiB, K (f16): 31744.00 MiB, V (f16): 31744.00 MiB
llama_new_context_with_model:        CPU  output buffer size =     1.00 MiB
llama_new_context_with_model:        CPU compute buffer size =  8743.51 MiB
llama_new_context_with_model: graph nodes  = 2052
llama_new_context_with_model: graph splits = 1

system_info: n_threads = 64 / 64 | AVX = 0 | AVX_VNNI = 0 | AVX2 = 0 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 0 | NEON = 1 | SVE = 0 | ARM_FMA = 1 | F16C = 0 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 0 | SSSE3 = 0 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 |
sampling:
        repeat_last_n = 64, repeat_penalty = 1.000, frequency_penalty = 0.000, presence_penalty = 0.000
        top_k = 40, tfs_z = 1.000, top_p = 0.950, min_p = 0.050, typical_p = 1.000, temp = 0.800
        mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000
        xtc_probability = 0.000, xtc_threshold = 1.000, top_n_sigma = 0.000
sampling order:
CFG -> Penalties -> top_k -> tfs_z -> typical_p -> top_p -> min_p -> top_n_sigma -> temperature
generate: n_ctx = 131072, n_batch = 2048, n_predict = -1, n_keep = 1


What is the meaning of life? In english please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please please

---

👤 **ikawrakow** commented the **2025-06-19** at **08:53:16**:<br>

Can you try the latest build?

---

👤 **jagusztinl** commented the **2025-06-20** at **08:01:04**:<br>

Same, please help:
:~/models$ uname -a
Linux gpt 6.11.0-1015-azure #15~24.04.1-Ubuntu SMP Thu May  1 03:01:44 UTC 2025 aarch64 aarch64 aarch64 GNU/Linux

:~/models$ gcc --version
gcc (Ubuntu 14.2.0-4ubuntu2~24.04) 14.2.0

git clone https://github.com/ikawrakow/ik_llama.cpp.git
cmake -B ./build -DGGML_CUDA=OFF -DGGML_BLAS=OFF
cmake --build ./build --config Release -j $(nproc)

~/models$ ../ik_llama.cpp//build/bin/llama-cli  -m  Qwen3-32B-Q4_0.gguf  --prompt "What is the meaning of life? In english please"
Log start
main: build = 3762 (1843ed22)
main: built with cc (Ubuntu 14.2.0-4ubuntu2~24.04) 14.2.0 for aarch64-linux-gnu
main: seed  = 1750406253
llama_model_loader: loaded meta data with 32 key-value pairs and 707 tensors from Qwen3-32B-Q4_0.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3-32B
llama_model_loader: - kv   3:                           general.basename str              = Qwen3-32B
llama_model_loader: - kv   4:                       general.quantized_by str              = Unsloth
llama_model_loader: - kv   5:                         general.size_label str              = 32B
llama_model_loader: - kv   6:                           general.repo_url str              = https://huggingface.co/unsloth
llama_model_loader: - kv   7:                          qwen3.block_count u32              = 64
llama_model_loader: - kv   8:                       qwen3.context_length u32              = 40960
llama_model_loader: - kv   9:                     qwen3.embedding_length u32              = 5120
llama_model_loader: - kv  10:                  qwen3.feed_forward_length u32              = 25600
llama_model_loader: - kv  11:                 qwen3.attention.head_count u32              = 64
llama_model_loader: - kv  12:              qwen3.attention.head_count_kv u32              = 8
llama_model_loader: - kv  13:                       qwen3.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  14:     qwen3.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  15:                 qwen3.attention.key_length u32              = 128
llama_model_loader: - kv  16:               qwen3.attention.value_length u32              = 128
llama_model_loader: - kv  17:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  18:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  19:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  20:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  21:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv  22:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  23:            tokenizer.ggml.padding_token_id u32              = 151654
llama_model_loader: - kv  24:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  25:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - kv  26:               general.quantization_version u32              = 2
llama_model_loader: - kv  27:                          general.file_type u32              = 2
llama_model_loader: - kv  28:                      quantize.imatrix.file str              = Qwen3-32B-GGUF/imatrix_unsloth.dat
llama_model_loader: - kv  29:                   quantize.imatrix.dataset str              = unsloth_calibration_Qwen3-32B.txt
llama_model_loader: - kv  30:             quantize.imatrix.entries_count i32              = 448
llama_model_loader: - kv  31:              quantize.imatrix.chunks_count i32              = 685
llama_model_loader: - type  f32:  257 tensors
llama_model_loader: - type q4_0:  441 tensors
llama_model_loader: - type q4_1:    8 tensors
llama_model_loader: - type q6_K:    1 tensors
llm_load_vocab: special tokens cache size = 26
llm_load_vocab: token to piece cache size = 0.9311 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = qwen3
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 151936
llm_load_print_meta: n_merges         = 151387
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 40960
llm_load_print_meta: n_embd           = 5120
llm_load_print_meta: n_layer          = 64
llm_load_print_meta: n_head           = 64
llm_load_print_meta: n_head_kv        = 8
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 8
llm_load_print_meta: n_embd_k_gqa     = 1024
llm_load_print_meta: n_embd_v_gqa     = 1024
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 25600
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
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
llm_load_print_meta: model ftype      = Q4_0
llm_load_print_meta: model params     = 32.762 B
llm_load_print_meta: model size       = 17.413 GiB (4.566 BPW)
llm_load_print_meta: repeating layers = 16.411 GiB (4.517 BPW, 31.206 B parameters)
llm_load_print_meta: general.name     = Qwen3-32B
llm_load_print_meta: BOS token        = 11 ','
llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
llm_load_print_meta: PAD token        = 151654 '<|vision_pad|>'
llm_load_print_meta: LF token         = 148848 'ÄĬ'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_tensors: ggml ctx size =    0.32 MiB
llm_load_tensors:        CPU buffer size = 17830.96 MiB
.................................................................................................
llama_new_context_with_model: n_ctx      = 40960
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 0
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:        CPU KV buffer size = 10240.00 MiB
llama_new_context_with_model: KV self size  = 10240.00 MiB, K (f16): 5120.00 MiB, V (f16): 5120.00 MiB
llama_new_context_with_model:        CPU  output buffer size =     0.58 MiB
llama_new_context_with_model:        CPU compute buffer size =  5252.01 MiB
llama_new_context_with_model: graph nodes  = 1989
llama_new_context_with_model: graph splits = 1

system_info: n_threads = 64 / 64 | AVX = 0 | AVX_VNNI = 0 | AVX2 = 0 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 0 | NEON = 1 | SVE = 0 | ARM_FMA = 1 | F16C = 0 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 0 | SSSE3 = 0 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 |
sampling:
        repeat_last_n = 64, repeat_penalty = 1.000, frequency_penalty = 0.000, presence_penalty = 0.000
        top_k = 40, tfs_z = 1.000, top_p = 0.950, min_p = 0.050, typical_p = 1.000, temp = 0.800
        mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000
        xtc_probability = 0.000, xtc_threshold = 1.000, top_n_sigma = 0.000
sampling order:
CFG -> Penalties -> dry -> top_k -> tfs_z -> typical_p -> top_p -> min_p -> xtc -> top_n_sigma -> temperature
generate: n_ctx = 40960, n_batch = 2048, n_predict = -1, n_keep = 0


What is the meaning of life? In english please-E4>6'236,(=+G7(@G>H$8,<F*("-D#'6:FC6.!+;1CF(B%D!-1@;8)((2+/5=>$,",E0CC*"B"61(F6<'8-,B9&

---

👤 **jagusztinl** commented the **2025-06-20** at **12:54:53**:<br>

FYI, I had this warnings during compilation:

[ 16%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml-aarch64.c.o
[ 16%] Built target build_info
In function ‘SHA1Update’,
    inlined from ‘SHA1Final’ at /home/alerant/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c:265:5:
/home/alerant/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c:219:13: warning: ‘SHA1Transform’ reading 64 bytes from a region of size 0 [-Wstringop-overread]
  219 |             SHA1Transform(context->state, &data[i]);
      |             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c:219:13: note: referencing argument 2 of type ‘const unsigned char[64]’
/home/alerant/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c: In function ‘SHA1Final’:
/home/alerant/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c:54:6: note: in a call to function ‘SHA1Transform’
   54 | void SHA1Transform(
      |      ^~~~~~~~~~~~~
In function ‘SHA1Update’,
    inlined from ‘SHA1Final’ at /home/alerant/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c:269:9:
/home/alerant/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c:219:13: warning: ‘SHA1Transform’ reading 64 bytes from a region of size 0 [-Wstringop-overread]
  219 |             SHA1Transform(context->state, &data[i]);
      |             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c:219:13: note: referencing argument 2 of type ‘const unsigned char[64]’
/home/alerant/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c: In function ‘SHA1Final’:
/home/alerant/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c:54:6: note: in a call to function ‘SHA1Transform’
   54 | void SHA1Transform(
      |      ^~~~~~~~~~~~~
[ 16%] Built target sha1
[ 16%] Built target sha256
In file included from /home/alerant/ik_llama.cpp/ggml/src/iqk/fa/iqk_fa_128_128.cpp:5:
/home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h: In member function ‘void {anonymous}::HelperQ40::load(int, int, {anonymous}::F16::Data&, {anonymous}::F16::Data&) const’:
/home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:534:30: warning: dereferencing type-punned pointer will break strict-aliasing rules [-Wstrict-aliasing]
  534 |         auto vd = F16::set1(*(const float16_t *)&dl->d);
      |                              ^~~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h: In member function ‘void {anonymous}::HelperQ41::load(int, int, {anonymous}::F16::Data&, {anonymous}::F16::Data&) const’:
/home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:578:30: warning: dereferencing type-punned pointer will break strict-aliasing rules [-Wstrict-aliasing]
  578 |         auto vd = F16::set1(*(const float16_t *)&dl->d);
      |                              ^~~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:579:30: warning: dereferencing type-punned pointer will break strict-aliasing rules [-Wstrict-aliasing]
  579 |         auto vm = F16::set1(*(const float16_t *)&dl->m);
      |                              ^~~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h: In member function ‘void {anonymous}::HelperIQ4nl::load(int, int, {anonymous}::F16::Data&, {anonymous}::F16::Data&) const’:
/home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:632:30: warning: dereferencing type-punned pointer will break strict-aliasing rules [-Wstrict-aliasing]
  632 |         auto vd = F16::set1(*(const float16_t *)&dl->d);
      |                              ^~~~~~~~~~~~~~~~~~~~~~~~~
In file included from /home/alerant/ik_llama.cpp/ggml/src/iqk/fa/iqk_fa_96_96.cpp:5:
/home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h: In member function ‘void {anonymous}::HelperQ40::load(int, int, {anonymous}::F16::Data&, {anonymous}::F16::Data&) const’:
/home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:534:30: warning: dereferencing type-punned pointer will break strict-aliasing rules [-Wstrict-aliasing]
  534 |         auto vd = F16::set1(*(const float16_t *)&dl->d);
      |                              ^~~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h: In member function ‘void {anonymous}::HelperQ41::load(int, int, {anonymous}::F16::Data&, {anonymous}::F16::Data&) const’:
/home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:578:30: warning: dereferencing type-punned pointer will break strict-aliasing rules [-Wstrict-aliasing]
  578 |         auto vd = F16::set1(*(const float16_t *)&dl->d);
      |                              ^~~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:579:30: warning: dereferencing type-punned pointer will break strict-aliasing rules [-Wstrict-aliasing]
  579 |         auto vm = F16::set1(*(const float16_t *)&dl->m);
      |                              ^~~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h: In member function ‘void {anonymous}::HelperIQ4nl::load(int, int, {anonymous}::F16::Data&, {anonymous}::F16::Data&) const’:
/home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:632:30: warning: dereferencing type-punned pointer will break strict-aliasing rules [-Wstrict-aliasing]
  632 |         auto vd = F16::set1(*(const float16_t *)&dl->d);
      |                              ^~~~~~~~~~~~~~~~~~~~~~~~~
In file included from /home/alerant/ik_llama.cpp/ggml/src/iqk/fa/iqk_fa_256_256.cpp:5:
/home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h: In member function ‘void {anonymous}::HelperQ40::load(int, int, {anonymous}::F16::Data&, {anonymous}::F16::Data&) const’:
/home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:534:30: warning: dereferencing type-punned pointer will break strict-aliasing rules [-Wstrict-aliasing]
  534 |         auto vd = F16::set1(*(const float16_t *)&dl->d);
      |                              ^~~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h: In member function ‘void {anonymous}::HelperQ41::load(int, int, {anonymous}::F16::Data&, {anonymous}::F16::Data&) const’:
/home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:578:30: warning: dereferencing type-punned pointer will break strict-aliasing rules [-Wstrict-aliasing]
  578 |         auto vd = F16::set1(*(const float16_t *)&dl->d);
      |                              ^~~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:579:30: warning: dereferencing type-punned pointer will break strict-aliasing rules [-Wstrict-aliasing]
  579 |         auto vm = F16::set1(*(const float16_t *)&dl->m);
      |                              ^~~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h: In member function ‘void {anonymous}::HelperIQ4nl::load(int, int, {anonymous}::F16::Data&, {anonymous}::F16::Data&) const’:
/home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:632:30: warning: dereferencing type-punned pointer will break strict-aliasing rules [-Wstrict-aliasing]
  632 |         auto vd = F16::set1(*(const float16_t *)&dl->d);
      |                              ^~~~~~~~~~~~~~~~~~~~~~~~~
In file included from /home/alerant/ik_llama.cpp/ggml/src/iqk/fa/iqk_fa_64_64.cpp:5:
/home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h: In member function ‘void {anonymous}::HelperQ40::load(int, int, {anonymous}::F16::Data&, {anonymous}::F16::Data&) const’:
/home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:534:30: warning: dereferencing type-punned pointer will break strict-aliasing rules [-Wstrict-aliasing]
  534 |         auto vd = F16::set1(*(const float16_t *)&dl->d);
      |                              ^~~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h: In member function ‘void {anonymous}::HelperQ41::load(int, int, {anonymous}::F16::Data&, {anonymous}::F16::Data&) const’:
/home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:578:30: warning: dereferencing type-punned pointer will break strict-aliasing rules [-Wstrict-aliasing]
  578 |         auto vd = F16::set1(*(const float16_t *)&dl->d);
      |                              ^~~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:579:30: warning: dereferencing type-punned pointer will break strict-aliasing rules [-Wstrict-aliasing]
  579 |         auto vm = F16::set1(*(const float16_t *)&dl->m);
      |                              ^~~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h: In member function ‘void {anonymous}::HelperIQ4nl::load(int, int, {anonymous}::F16::Data&, {anonymous}::F16::Data&) const’:
/home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:632:30: warning: dereferencing type-punned pointer will break strict-aliasing rules [-Wstrict-aliasing]
  632 |         auto vd = F16::set1(*(const float16_t *)&dl->d);
      |                              ^~~~~~~~~~~~~~~~~~~~~~~~~
In file included from /home/alerant/ik_llama.cpp/ggml/src/iqk/fa/iqk_fa_192_128.cpp:5:
/home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h: In member function ‘void {anonymous}::HelperQ40::load(int, int, {anonymous}::F16::Data&, {anonymous}::F16::Data&) const’:
/home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:534:30: warning: dereferencing type-punned pointer will break strict-aliasing rules [-Wstrict-aliasing]
  534 |         auto vd = F16::set1(*(const float16_t *)&dl->d);
      |                              ^~~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h: In member function ‘void {anonymous}::HelperQ41::load(int, int, {anonymous}::F16::Data&, {anonymous}::F16::Data&) const’:
/home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:578:30: warning: dereferencing type-punned pointer will break strict-aliasing rules [-Wstrict-aliasing]
  578 |         auto vd = F16::set1(*(const float16_t *)&dl->d);
      |                              ^~~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:579:30: warning: dereferencing type-punned pointer will break strict-aliasing rules [-Wstrict-aliasing]
  579 |         auto vm = F16::set1(*(const float16_t *)&dl->m);
      |                              ^~~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h: In member function ‘void {anonymous}::HelperIQ4nl::load(int, int, {anonymous}::F16::Data&, {anonymous}::F16::Data&) const’:
/home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:632:30: warning: dereferencing type-punned pointer will break strict-aliasing rules [-Wstrict-aliasing]
  632 |         auto vd = F16::set1(*(const float16_t *)&dl->d);
      |                              ^~~~~~~~~~~~~~~~~~~~~~~~~
In file included from /home/alerant/ik_llama.cpp/ggml/src/iqk/fa/iqk_fa_576_512.cpp:5:
/home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h: In member function ‘void {anonymous}::HelperQ40::load(int, int, {anonymous}::F16::Data&, {anonymous}::F16::Data&) const’:
/home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:534:30: warning: dereferencing type-punned pointer will break strict-aliasing rules [-Wstrict-aliasing]
  534 |         auto vd = F16::set1(*(const float16_t *)&dl->d);
      |                              ^~~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h: In member function ‘void {anonymous}::HelperQ41::load(int, int, {anonymous}::F16::Data&, {anonymous}::F16::Data&) const’:
/home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:578:30: warning: dereferencing type-punned pointer will break strict-aliasing rules [-Wstrict-aliasing]
  578 |         auto vd = F16::set1(*(const float16_t *)&dl->d);
      |                              ^~~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:579:30: warning: dereferencing type-punned pointer will break strict-aliasing rules [-Wstrict-aliasing]
  579 |         auto vm = F16::set1(*(const float16_t *)&dl->m);
      |                              ^~~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h: In member function ‘void {anonymous}::HelperIQ4nl::load(int, int, {anonymous}::F16::Data&, {anonymous}::F16::Data&) const’:
/home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:632:30: warning: dereferencing type-punned pointer will break strict-aliasing rules [-Wstrict-aliasing]
  632 |         auto vd = F16::set1(*(const float16_t *)&dl->d);
      |                              ^~~~~~~~~~~~~~~~~~~~~~~~~
In file included from /home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:1119:
/home/alerant/ik_llama.cpp/ggml/src/iqk/fa/iqk_fa_templates.h: In member function ‘void {anonymous}::HelperQ40::load(int, int, {anonymous}::F16::Data&, {anonymous}::F16::Data&) const’:
/home/alerant/ik_llama.cpp/ggml/src/iqk/fa/iqk_fa_templates.h:534:30: warning: dereferencing type-punned pointer will break strict-aliasing rules [-Wstrict-aliasing]
  534 |         auto vd = F16::set1(*(const float16_t *)&dl->d);
      |                              ^~~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/fa/iqk_fa_templates.h: In member function ‘void {anonymous}::HelperQ41::load(int, int, {anonymous}::F16::Data&, {anonymous}::F16::Data&) const’:
/home/alerant/ik_llama.cpp/ggml/src/iqk/fa/iqk_fa_templates.h:578:30: warning: dereferencing type-punned pointer will break strict-aliasing rules [-Wstrict-aliasing]
  578 |         auto vd = F16::set1(*(const float16_t *)&dl->d);
      |                              ^~~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/fa/iqk_fa_templates.h:579:30: warning: dereferencing type-punned pointer will break strict-aliasing rules [-Wstrict-aliasing]
  579 |         auto vm = F16::set1(*(const float16_t *)&dl->m);
      |                              ^~~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/fa/iqk_fa_templates.h: In member function ‘void {anonymous}::HelperIQ4nl::load(int, int, {anonymous}::F16::Data&, {anonymous}::F16::Data&) const’:
/home/alerant/ik_llama.cpp/ggml/src/iqk/fa/iqk_fa_templates.h:632:30: warning: dereferencing type-punned pointer will break strict-aliasing rules [-Wstrict-aliasing]
  632 |         auto vd = F16::set1(*(const float16_t *)&dl->d);
      |                              ^~~~~~~~~~~~~~~~~~~~~~~~~
In file included from /home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_gemm_floats.h:3,
                 from /home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:23:
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h: At global scope:
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:851:31: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  851 | static IQK_ALWAYS_INLINE void prepare_iq4_nl_quants_r8(const int8x16_t& values, const uint8x16_t& m4, const uint8x16x2_t& bits, int8x16_t * qx) {
      |                               ^~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:840:31: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  840 | static IQK_ALWAYS_INLINE void prepare_iq4_nl_quants(const int8x16_t& values, const uint8x16_t& m4, const uint8x16x4_t& bits, int8x16_t * qx) {
      |                               ^~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:831:36: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  831 | static IQK_ALWAYS_INLINE int32x4_t interleaved_dotq(const int8x16_t * qx, const int8x16_t& y) {
      |                                    ^~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:818:38: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  818 | static IQK_ALWAYS_INLINE int32x4x2_t interleaved_dotq_b16(const int8x16_t * qx, const int8x16x2_t& y) {
      |                                      ^~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:805:36: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  805 | static IQK_ALWAYS_INLINE int32x4_t interleaved_dotq(const int8x16_t * qx, const int8x16x2_t& y) {
      |                                    ^~~~~~~~~~~~~~~~
In file included from /home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_gemm_floats.h:3,
                 from /home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_gemm_floats.cpp:1:
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:851:31: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  851 | static IQK_ALWAYS_INLINE void prepare_iq4_nl_quants_r8(const int8x16_t& values, const uint8x16_t& m4, const uint8x16x2_t& bits, int8x16_t * qx) {
      |                               ^~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:840:31: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  840 | static IQK_ALWAYS_INLINE void prepare_iq4_nl_quants(const int8x16_t& values, const uint8x16_t& m4, const uint8x16x4_t& bits, int8x16_t * qx) {
      |                               ^~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:831:36: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  831 | static IQK_ALWAYS_INLINE int32x4_t interleaved_dotq(const int8x16_t * qx, const int8x16_t& y) {
      |                                    ^~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:818:38: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  818 | static IQK_ALWAYS_INLINE int32x4x2_t interleaved_dotq_b16(const int8x16_t * qx, const int8x16x2_t& y) {
      |                                      ^~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:805:36: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  805 | static IQK_ALWAYS_INLINE int32x4_t interleaved_dotq(const int8x16_t * qx, const int8x16x2_t& y) {
      |                                    ^~~~~~~~~~~~~~~~
In file included from /home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_gemm_1bit.h:3,
                 from /home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_gemm_1bit.cpp:1:
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:851:31: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  851 | static IQK_ALWAYS_INLINE void prepare_iq4_nl_quants_r8(const int8x16_t& values, const uint8x16_t& m4, const uint8x16x2_t& bits, int8x16_t * qx) {
      |                               ^~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:840:31: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  840 | static IQK_ALWAYS_INLINE void prepare_iq4_nl_quants(const int8x16_t& values, const uint8x16_t& m4, const uint8x16x4_t& bits, int8x16_t * qx) {
      |                               ^~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:831:36: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  831 | static IQK_ALWAYS_INLINE int32x4_t interleaved_dotq(const int8x16_t * qx, const int8x16_t& y) {
      |                                    ^~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:818:38: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  818 | static IQK_ALWAYS_INLINE int32x4x2_t interleaved_dotq_b16(const int8x16_t * qx, const int8x16x2_t& y) {
      |                                      ^~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:805:36: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  805 | static IQK_ALWAYS_INLINE int32x4_t interleaved_dotq(const int8x16_t * qx, const int8x16x2_t& y) {
      |                                    ^~~~~~~~~~~~~~~~
In file included from /home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_gemm_ktquants.cpp:1:
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:851:31: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  851 | static IQK_ALWAYS_INLINE void prepare_iq4_nl_quants_r8(const int8x16_t& values, const uint8x16_t& m4, const uint8x16x2_t& bits, int8x16_t * qx) {
      |                               ^~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:840:31: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  840 | static IQK_ALWAYS_INLINE void prepare_iq4_nl_quants(const int8x16_t& values, const uint8x16_t& m4, const uint8x16x4_t& bits, int8x16_t * qx) {
      |                               ^~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:831:36: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  831 | static IQK_ALWAYS_INLINE int32x4_t interleaved_dotq(const int8x16_t * qx, const int8x16_t& y) {
      |                                    ^~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:818:38: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  818 | static IQK_ALWAYS_INLINE int32x4x2_t interleaved_dotq_b16(const int8x16_t * qx, const int8x16x2_t& y) {
      |                                      ^~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:805:36: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  805 | static IQK_ALWAYS_INLINE int32x4_t interleaved_dotq(const int8x16_t * qx, const int8x16x2_t& y) {
      |                                    ^~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_gemm_floats.cpp: In function ‘void iqk_gemm_default_floats(int, int, const char*, size_t, DataInfo&, int)’:
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_gemm_floats.cpp:1039:34: warning: this statement may fall through [-Wimplicit-fallthrough=]
 1039 |             case  1: mm_helper<1>(D, nq, cx, bx, info, k_step);
      |                      ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_gemm_floats.cpp:1040:13: note: here
 1040 |             case  2: mm_helper<2>(D, nq, cx, bx, info, k_step);
      |             ^~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_gemm_floats.cpp:1040:34: warning: this statement may fall through [-Wimplicit-fallthrough=]
 1040 |             case  2: mm_helper<2>(D, nq, cx, bx, info, k_step);
      |                      ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_gemm_floats.cpp:1041:13: note: here
 1041 |             default: mm_helper<3>(D, nq, cx, bx, info, k_step);
      |             ^~~~~~~
In file included from /home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_gemm_iquants.h:3,
                 from /home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_gemm_iquants.cpp:1:
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:851:31: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  851 | static IQK_ALWAYS_INLINE void prepare_iq4_nl_quants_r8(const int8x16_t& values, const uint8x16_t& m4, const uint8x16x2_t& bits, int8x16_t * qx) {
      |                               ^~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:840:31: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  840 | static IQK_ALWAYS_INLINE void prepare_iq4_nl_quants(const int8x16_t& values, const uint8x16_t& m4, const uint8x16x4_t& bits, int8x16_t * qx) {
      |                               ^~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:831:36: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  831 | static IQK_ALWAYS_INLINE int32x4_t interleaved_dotq(const int8x16_t * qx, const int8x16_t& y) {
      |                                    ^~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:818:38: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  818 | static IQK_ALWAYS_INLINE int32x4x2_t interleaved_dotq_b16(const int8x16_t * qx, const int8x16x2_t& y) {
      |                                      ^~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:805:36: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  805 | static IQK_ALWAYS_INLINE int32x4_t interleaved_dotq(const int8x16_t * qx, const int8x16x2_t& y) {
      |                                    ^~~~~~~~~~~~~~~~
In file included from /home/alerant/ik_llama.cpp/ggml/src/./iqk/iqk_gemm_floats.h:3,
                 from /home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:23:
/home/alerant/ik_llama.cpp/ggml/src/./iqk/iqk_common.h: At global scope:
/home/alerant/ik_llama.cpp/ggml/src/./iqk/iqk_common.h:851:31: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  851 | static IQK_ALWAYS_INLINE void prepare_iq4_nl_quants_r8(const int8x16_t& values, const uint8x16_t& m4, const uint8x16x2_t& bits, int8x16_t * qx) {
      |                               ^~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/./iqk/iqk_common.h:840:31: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  840 | static IQK_ALWAYS_INLINE void prepare_iq4_nl_quants(const int8x16_t& values, const uint8x16_t& m4, const uint8x16x4_t& bits, int8x16_t * qx) {
      |                               ^~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/./iqk/iqk_common.h:831:36: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  831 | static IQK_ALWAYS_INLINE int32x4_t interleaved_dotq(const int8x16_t * qx, const int8x16_t& y) {
      |                                    ^~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/./iqk/iqk_common.h:818:38: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  818 | static IQK_ALWAYS_INLINE int32x4x2_t interleaved_dotq_b16(const int8x16_t * qx, const int8x16x2_t& y) {
      |                                      ^~~~~~~~~~~~~~~~~~~~
In file included from /home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_gemm_legacy_quants.h:3,
                 from /home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_gemm_legacy_quants.cpp:1:
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:851:31: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  851 | static IQK_ALWAYS_INLINE void prepare_iq4_nl_quants_r8(const int8x16_t& values, const uint8x16_t& m4, const uint8x16x2_t& bits, int8x16_t * qx) {
      |                               ^~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/./iqk/iqk_common.h:805:36: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  805 | static IQK_ALWAYS_INLINE int32x4_t interleaved_dotq(const int8x16_t * qx, const int8x16x2_t& y) {
      |                                    ^~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:840:31: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  840 | static IQK_ALWAYS_INLINE void prepare_iq4_nl_quants(const int8x16_t& values, const uint8x16_t& m4, const uint8x16x4_t& bits, int8x16_t * qx) {
      |                               ^~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:831:36: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  831 | static IQK_ALWAYS_INLINE int32x4_t interleaved_dotq(const int8x16_t * qx, const int8x16_t& y) {
      |                                    ^~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:818:38: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  818 | static IQK_ALWAYS_INLINE int32x4x2_t interleaved_dotq_b16(const int8x16_t * qx, const int8x16x2_t& y) {
      |                                      ^~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:805:36: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  805 | static IQK_ALWAYS_INLINE int32x4_t interleaved_dotq(const int8x16_t * qx, const int8x16x2_t& y) {
      |                                    ^~~~~~~~~~~~~~~~
In file included from /home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_gemm_iqk_quants.h:3,
                 from /home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_gemm_iqk_quants.cpp:1:
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:851:31: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  851 | static IQK_ALWAYS_INLINE void prepare_iq4_nl_quants_r8(const int8x16_t& values, const uint8x16_t& m4, const uint8x16x2_t& bits, int8x16_t * qx) {
      |                               ^~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:840:31: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  840 | static IQK_ALWAYS_INLINE void prepare_iq4_nl_quants(const int8x16_t& values, const uint8x16_t& m4, const uint8x16x4_t& bits, int8x16_t * qx) {
      |                               ^~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:831:36: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  831 | static IQK_ALWAYS_INLINE int32x4_t interleaved_dotq(const int8x16_t * qx, const int8x16_t& y) {
      |                                    ^~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:818:38: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  818 | static IQK_ALWAYS_INLINE int32x4x2_t interleaved_dotq_b16(const int8x16_t * qx, const int8x16x2_t& y) {
      |                                      ^~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:805:36: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  805 | static IQK_ALWAYS_INLINE int32x4_t interleaved_dotq(const int8x16_t * qx, const int8x16x2_t& y) {
      |                                    ^~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_gemm_kquants.cpp:3082:24: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
 3082 | IQK_ALWAYS_INLINE void prepare_q4_k_quants(const uint8x16_t& m4, const uint8x16x4_t& bits, int8x16_t * qx) {
      |                        ^~~~~~~~~~~~~~~~~~~
In file included from /home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_gemm_kquants.h:3,
                 from /home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_gemm_kquants.cpp:1:
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:851:31: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  851 | static IQK_ALWAYS_INLINE void prepare_iq4_nl_quants_r8(const int8x16_t& values, const uint8x16_t& m4, const uint8x16x2_t& bits, int8x16_t * qx) {
      |                               ^~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:840:31: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  840 | static IQK_ALWAYS_INLINE void prepare_iq4_nl_quants(const int8x16_t& values, const uint8x16_t& m4, const uint8x16x4_t& bits, int8x16_t * qx) {
      |                               ^~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:831:36: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  831 | static IQK_ALWAYS_INLINE int32x4_t interleaved_dotq(const int8x16_t * qx, const int8x16_t& y) {
      |                                    ^~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:818:38: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  818 | static IQK_ALWAYS_INLINE int32x4x2_t interleaved_dotq_b16(const int8x16_t * qx, const int8x16x2_t& y) {
      |                                      ^~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:805:36: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  805 | static IQK_ALWAYS_INLINE int32x4_t interleaved_dotq(const int8x16_t * qx, const int8x16x2_t& y) {
      |                                    ^~~~~~~~~~~~~~~~
In file included from /home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:21:
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_gemm_1bit.cpp: In function ‘void {anonymous}::mul_mat_iq1bn_q8_K64(int, const void*, size_t, const DataInfo&, int) [with int nrc_y = 1]’:
/home/alerant/ik_llama.cpp/ggml/src/./ggml-impl.h:408:42: warning: iteration 2 invokes undefined behavior [-Waggressive-loop-optimizations]
  408 | #define ggml_vdotq_s32(a, b, c) vdotq_s32(a, b, c)
      |                                 ~~~~~~~~~^~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_gemm_1bit.cpp:2015:31: note: in expansion of macro ‘ggml_vdotq_s32’
 2015 |                     accd[0] = ggml_vdotq_s32(accd[0], q.val[j], v1.val[j]);
      |                               ^~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_gemm_1bit.cpp:2014:35: note: within this loop
 2014 |                 for (int j = 0; j < 4; ++j) {
      |                                 ~~^~~
In file included from /home/alerant/ik_llama.cpp/ggml/src/./iqk/iqk_gemm_floats.h:3,
                 from /home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:23:
/home/alerant/ik_llama.cpp/ggml/src/./iqk/iqk_common.h: At global scope:
/home/alerant/ik_llama.cpp/ggml/src/./iqk/iqk_common.h:851:31: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  851 | static IQK_ALWAYS_INLINE void prepare_iq4_nl_quants_r8(const int8x16_t& values, const uint8x16_t& m4, const uint8x16x2_t& bits, int8x16_t * qx) {
      |                               ^~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/./iqk/iqk_common.h:840:31: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  840 | static IQK_ALWAYS_INLINE void prepare_iq4_nl_quants(const int8x16_t& values, const uint8x16_t& m4, const uint8x16x4_t& bits, int8x16_t * qx) {
      |                               ^~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/./iqk/iqk_common.h:831:36: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  831 | static IQK_ALWAYS_INLINE int32x4_t interleaved_dotq(const int8x16_t * qx, const int8x16_t& y) {
      |                                    ^~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/./iqk/iqk_common.h:818:38: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  818 | static IQK_ALWAYS_INLINE int32x4x2_t interleaved_dotq_b16(const int8x16_t * qx, const int8x16x2_t& y) {
      |                                      ^~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/./iqk/iqk_common.h:805:36: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  805 | static IQK_ALWAYS_INLINE int32x4_t interleaved_dotq(const int8x16_t * qx, const int8x16x2_t& y) {
      |                                    ^~~~~~~~~~~~~~~~
In file included from /home/alerant/ik_llama.cpp/ggml/src/./iqk/iqk_gemm_floats.h:3,
                 from /home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:23:
/home/alerant/ik_llama.cpp/ggml/src/./iqk/iqk_common.h: At global scope:
/home/alerant/ik_llama.cpp/ggml/src/./iqk/iqk_common.h:851:31: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  851 | static IQK_ALWAYS_INLINE void prepare_iq4_nl_quants_r8(const int8x16_t& values, const uint8x16_t& m4, const uint8x16x2_t& bits, int8x16_t * qx) {
      |                               ^~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/./iqk/iqk_common.h:840:31: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  840 | static IQK_ALWAYS_INLINE void prepare_iq4_nl_quants(const int8x16_t& values, const uint8x16_t& m4, const uint8x16x4_t& bits, int8x16_t * qx) {
      |                               ^~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/./iqk/iqk_common.h:831:36: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  831 | static IQK_ALWAYS_INLINE int32x4_t interleaved_dotq(const int8x16_t * qx, const int8x16_t& y) {
      |                                    ^~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/./iqk/iqk_common.h:818:38: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  818 | static IQK_ALWAYS_INLINE int32x4x2_t interleaved_dotq_b16(const int8x16_t * qx, const int8x16x2_t& y) {
      |                                      ^~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/./iqk/iqk_common.h:805:36: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  805 | static IQK_ALWAYS_INLINE int32x4_t interleaved_dotq(const int8x16_t * qx, const int8x16x2_t& y) {
      |                                    ^~~~~~~~~~~~~~~~
In file included from /home/alerant/ik_llama.cpp/ggml/src/./iqk/iqk_gemm_floats.h:3,
                 from /home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:23:
/home/alerant/ik_llama.cpp/ggml/src/./iqk/iqk_common.h: At global scope:
/home/alerant/ik_llama.cpp/ggml/src/./iqk/iqk_common.h:851:31: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  851 | static IQK_ALWAYS_INLINE void prepare_iq4_nl_quants_r8(const int8x16_t& values, const uint8x16_t& m4, const uint8x16x2_t& bits, int8x16_t * qx) {
      |                               ^~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/./iqk/iqk_common.h:840:31: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  840 | static IQK_ALWAYS_INLINE void prepare_iq4_nl_quants(const int8x16_t& values, const uint8x16_t& m4, const uint8x16x4_t& bits, int8x16_t * qx) {
      |                               ^~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/./iqk/iqk_common.h:831:36: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  831 | static IQK_ALWAYS_INLINE int32x4_t interleaved_dotq(const int8x16_t * qx, const int8x16_t& y) {
      |                                    ^~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/./iqk/iqk_common.h:818:38: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  818 | static IQK_ALWAYS_INLINE int32x4x2_t interleaved_dotq_b16(const int8x16_t * qx, const int8x16x2_t& y) {
      |                                      ^~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/./iqk/iqk_common.h:805:36: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  805 | static IQK_ALWAYS_INLINE int32x4_t interleaved_dotq(const int8x16_t * qx, const int8x16x2_t& y) {
      |                                    ^~~~~~~~~~~~~~~~
In file included from /home/alerant/ik_llama.cpp/ggml/src/./iqk/iqk_gemm_floats.h:3,
                 from /home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:23:
/home/alerant/ik_llama.cpp/ggml/src/./iqk/iqk_common.h: At global scope:
/home/alerant/ik_llama.cpp/ggml/src/./iqk/iqk_common.h:851:31: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  851 | static IQK_ALWAYS_INLINE void prepare_iq4_nl_quants_r8(const int8x16_t& values, const uint8x16_t& m4, const uint8x16x2_t& bits, int8x16_t * qx) {
      |                               ^~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/./iqk/iqk_common.h:840:31: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  840 | static IQK_ALWAYS_INLINE void prepare_iq4_nl_quants(const int8x16_t& values, const uint8x16_t& m4, const uint8x16x4_t& bits, int8x16_t * qx) {
      |                               ^~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/./iqk/iqk_common.h:831:36: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  831 | static IQK_ALWAYS_INLINE int32x4_t interleaved_dotq(const int8x16_t * qx, const int8x16_t& y) {
      |                                    ^~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/./iqk/iqk_common.h:818:38: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  818 | static IQK_ALWAYS_INLINE int32x4x2_t interleaved_dotq_b16(const int8x16_t * qx, const int8x16x2_t& y) {
      |                                      ^~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/./iqk/iqk_common.h:805:36: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  805 | static IQK_ALWAYS_INLINE int32x4_t interleaved_dotq(const int8x16_t * qx, const int8x16x2_t& y) {
      |                                    ^~~~~~~~~~~~~~~~
In file included from /home/alerant/ik_llama.cpp/ggml/src/./iqk/iqk_gemm_floats.h:3,
                 from /home/alerant/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:23:
/home/alerant/ik_llama.cpp/ggml/src/./iqk/iqk_common.h: At global scope:
/home/alerant/ik_llama.cpp/ggml/src/./iqk/iqk_common.h:851:31: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  851 | static IQK_ALWAYS_INLINE void prepare_iq4_nl_quants_r8(const int8x16_t& values, const uint8x16_t& m4, const uint8x16x2_t& bits, int8x16_t * qx) {
      |                               ^~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/./iqk/iqk_common.h:840:31: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  840 | static IQK_ALWAYS_INLINE void prepare_iq4_nl_quants(const int8x16_t& values, const uint8x16_t& m4, const uint8x16x4_t& bits, int8x16_t * qx) {
      |                               ^~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/./iqk/iqk_common.h:831:36: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  831 | static IQK_ALWAYS_INLINE int32x4_t interleaved_dotq(const int8x16_t * qx, const int8x16_t& y) {
      |                                    ^~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/./iqk/iqk_common.h:818:38: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  818 | static IQK_ALWAYS_INLINE int32x4x2_t interleaved_dotq_b16(const int8x16_t * qx, const int8x16x2_t& y) {
      |                                      ^~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/./iqk/iqk_common.h:805:36: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  805 | static IQK_ALWAYS_INLINE int32x4_t interleaved_dotq(const int8x16_t * qx, const int8x16x2_t& y) {
      |                                    ^~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_quantize.cpp: In instantiation of ‘void {anonymous}::QuantizerIQKT<block_size, group_size, num_bits, is_abs, is_int>::find_best_match(float, const float*, const float*, int*) const [with int block_size = 32; int group_size = 8; int num_bits = 16; bool is_abs = false; bool is_int = true]’:
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_quantize.cpp:8067:38:   required from here
 8067 |             quantizer.find_best_match( amax/scale_0, xb, weight, best_idx);
      |             ~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_quantize.cpp:7585:9: warning: unused variable ‘ncluster’ [-Wunused-variable]
 7585 |     int ncluster = m_clusters.size()/kGroupSize;
      |         ^~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_quantize.cpp:7586:11: warning: unused variable ‘id’ [-Wunused-variable]
 7586 |     float id = 1/d;
      |           ^~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_quantize.cpp:7580:110: warning: unused parameter ‘xb’ [-Wunused-parameter]
 7580 | void QuantizerIQKT<block_size, group_size, num_bits, is_abs, is_int>::find_best_match(float d, const float * xb, const float * weight, int * best_idx) const {
      |                                                                                                ~~~~~~~~~~~~~~^~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_quantize.cpp:7580:128: warning: unused parameter ‘weight’ [-Wunused-parameter]
 7580 | void QuantizerIQKT<block_size, group_size, num_bits, is_abs, is_int>::find_best_match(float d, const float * xb, const float * weight, int * best_idx) const {
      |                                                                                                                  ~~~~~~~~~~~~~~^~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_quantize.cpp: In instantiation of ‘std::pair<float, float> {anonymous}::QuantizerIQKT<block_size, group_size, num_bits, is_abs, is_int>::find_best_scale(const float*, const float*, const int*) const [with int block_size = 32; int group_size = 8; int num_bits = 16; bool is_abs = false; bool is_int = true]’:
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_quantize.cpp:8068:59:   required from here
 8068 |             auto [dp, score_p] = quantizer.find_best_scale(xb, weight, best_idx);
      |                                  ~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_quantize.cpp:7514:25: note: parameter passing for argument of type ‘std::pair<float, float>’ when C++17 is enabled changed to match C++14 in GCC 10.1
 7514 | std::pair<float, float> QuantizerIQKT<block_size, group_size, num_bits, is_abs, is_int>::find_best_scale(
      |                         ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_quantize.cpp: In instantiation of ‘void {anonymous}::QuantizerIQKT<block_size, group_size, num_bits, is_abs, is_int>::find_best_match(float, const float*, const float*, int*) const [with int block_size = 32; int group_size = 8; int num_bits = 16; bool is_abs = true; bool is_int = true]’:
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_quantize.cpp:8367:42:   required from here
 8367 |                 quantizer.find_best_match(amax/(scale_0 + kStep*itry), xaux, weight, best_idx);
      |                 ~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_quantize.cpp:7585:9: warning: unused variable ‘ncluster’ [-Wunused-variable]
 7585 |     int ncluster = m_clusters.size()/kGroupSize;
      |         ^~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_quantize.cpp:7586:11: warning: unused variable ‘id’ [-Wunused-variable]
 7586 |     float id = 1/d;
      |           ^~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_quantize.cpp:7580:110: warning: unused parameter ‘xb’ [-Wunused-parameter]
 7580 | void QuantizerIQKT<block_size, group_size, num_bits, is_abs, is_int>::find_best_match(float d, const float * xb, const float * weight, int * best_idx) const {
      |                                                                                                ~~~~~~~~~~~~~~^~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_quantize.cpp:7580:128: warning: unused parameter ‘weight’ [-Wunused-parameter]
 7580 | void QuantizerIQKT<block_size, group_size, num_bits, is_abs, is_int>::find_best_match(float d, const float * xb, const float * weight, int * best_idx) const {
      |                                                                                                                  ~~~~~~~~~~~~~~^~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_quantize.cpp: In instantiation of ‘void {anonymous}::QuantizerIQKT<block_size, group_size, num_bits, is_abs, is_int>::find_best_match(float, const float*, const float*, int*) const [with int block_size = 32; int group_size = 4; int num_bits = 15; bool is_abs = false; bool is_int = true]’:
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_quantize.cpp:8642:43:   required from here
 8642 |                 quantizer1.find_best_match( amax/(8.f*itry + scale_0), xaux, weight, best_idx);
      |                 ~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_quantize.cpp:7585:9: warning: unused variable ‘ncluster’ [-Wunused-variable]
 7585 |     int ncluster = m_clusters.size()/kGroupSize;
      |         ^~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_quantize.cpp:7586:11: warning: unused variable ‘id’ [-Wunused-variable]
 7586 |     float id = 1/d;
      |           ^~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_quantize.cpp:7580:110: warning: unused parameter ‘xb’ [-Wunused-parameter]
 7580 | void QuantizerIQKT<block_size, group_size, num_bits, is_abs, is_int>::find_best_match(float d, const float * xb, const float * weight, int * best_idx) const {
      |                                                                                                ~~~~~~~~~~~~~~^~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_quantize.cpp:7580:128: warning: unused parameter ‘weight’ [-Wunused-parameter]
 7580 | void QuantizerIQKT<block_size, group_size, num_bits, is_abs, is_int>::find_best_match(float d, const float * xb, const float * weight, int * best_idx) const {
      |                                                                                                                  ~~~~~~~~~~~~~~^~~~~~
In file included from /home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_gemm_ktquants.h:3,
                 from /home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_quantize.cpp:17:
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:851:31: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  851 | static IQK_ALWAYS_INLINE void prepare_iq4_nl_quants_r8(const int8x16_t& values, const uint8x16_t& m4, const uint8x16x2_t& bits, int8x16_t * qx) {
      |                               ^~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:840:31: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  840 | static IQK_ALWAYS_INLINE void prepare_iq4_nl_quants(const int8x16_t& values, const uint8x16_t& m4, const uint8x16x4_t& bits, int8x16_t * qx) {
      |                               ^~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:831:36: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  831 | static IQK_ALWAYS_INLINE int32x4_t interleaved_dotq(const int8x16_t * qx, const int8x16_t& y) {
      |                                    ^~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:818:38: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  818 | static IQK_ALWAYS_INLINE int32x4x2_t interleaved_dotq_b16(const int8x16_t * qx, const int8x16x2_t& y) {
      |                                      ^~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/ggml/src/iqk/iqk_common.h:805:36: warning: ‘always_inline’ function might not be inlinable unless also declared ‘inline’ [-Wattributes]
  805 | static IQK_ALWAYS_INLINE int32x4_t interleaved_dotq(const int8x16_t * qx, const int8x16x2_t& y) {
      |                                    ^~~~~~~~~~~~~~~~
[ 16%] Built target xxhash
[ 16%] Linking CXX shared library libggml.so
[ 16%] Built target ggml
[ 17%] Building CXX object src/CMakeFiles/llama.dir/llama.cpp.o
[ 18%] Building CXX object examples/gguf-hash/CMakeFiles/llama-gguf-hash.dir/gguf-hash.cpp.o
[ 19%] Building CXX object examples/gguf/CMakeFiles/llama-gguf.dir/gguf.cpp.o
[ 20%] Building CXX object src/CMakeFiles/llama.dir/llama-vocab.cpp.o
[ 20%] Building CXX object src/CMakeFiles/llama.dir/llama-grammar.cpp.o
[ 21%] Building CXX object src/CMakeFiles/llama.dir/llama-sampling.cpp.o
[ 21%] Building CXX object src/CMakeFiles/llama.dir/unicode.cpp.o
[ 22%] Building CXX object src/CMakeFiles/llama.dir/unicode-data.cpp.o
[ 22%] Linking CXX executable ../../bin/llama-gguf
[ 22%] Built target llama-gguf
[ 23%] Linking CXX executable ../../bin/llama-gguf-hash
[ 23%] Built target llama-gguf-hash
^Cgmake[2]: *** [src/CMakeFiles/llama.dir/build.make:76: src/CMakeFiles/llama.dir/llama.cpp.o] Interrupt
gmake[1]: *** [CMakeFiles/Makefile2:1647: src/CMakeFiles/llama.dir/all] Interrupt
gmake: *** [Makefile:146: all] Interrupt

alerant@gpt:~/ik_llama.cpp$ cmake --build ./build --config Release -j $(nproc)
[  1%] Built target build_info
[  2%] Built target sha256
[  3%] Built target xxhash
[  3%] Built target sha1
[ 16%] Built target ggml
[ 18%] Built target llama-gguf-hash
[ 19%] Built target llama-gguf
[ 20%] Building CXX object src/CMakeFiles/llama.dir/llama.cpp.o
[ 20%] Linking CXX shared library libllama.so
[ 23%] Built target llama
[ 24%] Building C object tests/CMakeFiles/test-c.dir/test-c.c.o
[ 24%] Building CXX object common/CMakeFiles/common.dir/common.cpp.o
[ 25%] Building CXX object examples/benchmark/CMakeFiles/llama-bench-matmult.dir/benchmark-matmult.cpp.o
[ 26%] Building CXX object common/CMakeFiles/common.dir/sampling.cpp.o
[ 26%] Building CXX object common/CMakeFiles/common.dir/console.cpp.o
[ 27%] Building CXX object examples/llava/CMakeFiles/llava.dir/llava.cpp.o
[ 28%] Building CXX object examples/quantize-stats/CMakeFiles/llama-quantize-stats.dir/quantize-stats.cpp.o
[ 29%] Building CXX object examples/llava/CMakeFiles/llava.dir/clip.cpp.o
[ 30%] Building CXX object common/CMakeFiles/common.dir/grammar-parser.cpp.o
[ 31%] Building CXX object common/CMakeFiles/common.dir/json-schema-to-grammar.cpp.o
[ 31%] Building CXX object common/CMakeFiles/common.dir/train.cpp.o
[ 32%] Building CXX object common/CMakeFiles/common.dir/ngram-cache.cpp.o
[ 33%] Linking C executable ../bin/test-c
[ 33%] Built target test-c
In file included from /usr/include/c++/14/bits/stl_algobase.h:64,
                 from /usr/include/c++/14/bits/specfun.h:43,
                 from /usr/include/c++/14/cmath:3898,
                 from /usr/include/c++/14/random:40,
                 from /home/alerant/ik_llama.cpp/src/../include/llama.h:1326,
                 from /home/alerant/ik_llama.cpp/examples/quantize-stats/../../common/common.h:12,
                 from /home/alerant/ik_llama.cpp/examples/quantize-stats/quantize-stats.cpp:9:
/usr/include/c++/14/bits/stl_pair.h: In instantiation of ‘constexpr std::pair<typename std::__strip_reference_wrapper<typename std::decay<_Tp>::type>::__type, typename std::__strip_reference_wrapper<typename std::decay<_Tp2>::type>::__type> std::make_pair(_T1&&, _T2&&) [with _T1 = float; _T2 = float; typename __strip_reference_wrapper<typename decay<_Tp>::type>::__type = float; typename decay<_Tp>::type = float; typename __strip_reference_wrapper<typename decay<_Tp2>::type>::__type = float; typename decay<_Tp2>::type = float]’:
/home/alerant/ik_llama.cpp/examples/quantize-stats/quantize-stats.cpp:392:68:   required from here
  392 |     std::vector<std::pair<float, float>> range(ndim, std::make_pair(INFINITY, -INFINITY));
      |                                                      ~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~
/usr/include/c++/14/bits/stl_pair.h:1132:5: note: parameter passing for argument of type ‘std::pair<float, float>’ when C++17 is enabled changed to match C++14 in GCC 10.1
 1132 |     make_pair(_T1&& __x, _T2&& __y)
      |     ^~~~~~~~~
[ 33%] Linking CXX executable ../../bin/llama-bench-matmult
[ 33%] Built target llama-bench-matmult
[ 33%] Linking CXX executable ../../bin/llama-quantize-stats
[ 33%] Built target llama-quantize-stats
In file included from /home/alerant/ik_llama.cpp/examples/llava/clip.cpp:24:
/home/alerant/ik_llama.cpp/examples/llava/../../common/stb_image.h: In function ‘int stbi__parse_png_file(stbi__png*, int, int)’:
/home/alerant/ik_llama.cpp/examples/llava/../../common/stb_image.h:5450:31: warning: writing 1 byte into a region of size 0 [-Wstringop-overflow=]
 5450 |                         tc[k] = (stbi_uc)(stbi__get16be(s) & 255) *
      |                         ~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 5451 |                                 stbi__depth_scale_table[z->depth]; // non 8-bit images will be larger
      |                                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/home/alerant/ik_llama.cpp/examples/llava/../../common/stb_image.h:5326:28: note: at offset 3 into destination object ‘tc’ of size 3
 5326 |     stbi_uc has_trans = 0, tc[3] = {0};
      |                            ^~
[ 33%] Built target llava

---

👤 **jagusztinl** commented the **2025-06-20** at **14:04:07**:<br>

Fixed:  build with -DGGML_SVE=ON solved it

But not faster inference for any model than the current llama.cpp build on ARM CPU (pp better):

For example, on the same server:

llama.cpp:
 deepseek2 671B Q4_0            | 353.47 GiB |   671.03 B | CPU        |  99 |  1 |           pp512 |         43.27 ± 0.16 |
 deepseek2 671B Q4_0            | 353.47 GiB |   671.03 B | CPU        |  99 |  1 |           tg128 |         10.97 ± 0.07 |

ik_llama.cpp:
| model                          |       size |     params | backend    | threads | type_k | type_v | fa | mla |   amb | rtr | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -----: | -----: | -: | --: | ----: | --: | ---: | ------------: | ---------------: |
============ Repacked 611 tensors
| deepseek2 671B Q4_K_R4         | 413.14 GiB |   672.05 B | CPU        |      64 |   q8_0 |   q8_0 |  1 |   3 |  2048 |   1 |    1 |         pp512 |     70.30 ± 0.08 |
| deepseek2 671B Q4_K_R4         | 413.14 GiB |   672.05 B | CPU        |      64 |   q8_0 |   q8_0 |  1 |   3 |  2048 |   1 |    1 |         tg128 |      9.59 ± 0.02 |

---

👤 **jagusztinl** commented the **2025-06-20** at **14:04:07**:<br>

Fixed:  build with -DGGML_SVE=ON solved it

---

👤 **jagusztinl** commented the **2025-06-20** at **14:06:39**:<br>

But not faster for any model than the current llama.cpp build on ARM CPU

---

👤 **ikawrakow** commented the **2025-06-20** at **15:50:59**:<br>

You never mentioned your are using an ARM CPU. Unlike llama.cpp, nothing is automatically set for you on ARM. It is likely you need to set arch options manually. `-DGGML_SVE=ON` solving your issues sounds strange to me as no usage is made of SVE anywhere in `ik_llama.cpp`. The only ARM implementation that exists is NEON.

A 60% difference in PP-performance is not faster on your book? And that is for the quant receiving the most love in mainline `llamas.cpp`, with a special purpose GEMM and GEMV implementations for ARM CPUs.

Also, `PP-512` and `TG-128` are very misleading measures of performance. When is it in real usage that I have zero tokens in the KV cache? Try running with something more significant in the KV cache (8k-18k tokens) and see how that goes. You may also want to try some of the i-quants.

But overall, yes, ARM CPUs are not a big focus of this project. I maintain it in a functional state, but haven't updated the ARM implementation for quite some time. It is missing the massive PP performance gains that I got on `AVX2` during the last 2-3 weeks.

---

👤 **ikawrakow** commented the **2025-06-20** at **15:59:12**:<br>

Oh, what is the CPU you are using?

---

👤 **jagusztinl** commented the **2025-06-21** at **08:39:04**:<br>

Thank you for your answer, a bit detailed explanation of the project:
-We are using Azure Cobalt ARM CPUs on spot VMs, (64 real core, 512Gb 12 channel very fast RAM) for 0.5USD/hour (!) instead of expensive GPU setups. The price/perforance ratio is unbeatable: our collegues can use DeepSeek privately for  80USD/month continuously without limits.
-We experimented with llama.cpp as the fastest inference engine, with this setup (optimized for Cobalt and linked with ARM performance libs): cmake -DCMAKE_CXX_FLAGS="-mcpu=cobalt-100 -mtune=cobalt-100 -flto -Ofast  -DINTEGER64 -I${ARMPL_DIR}/include -larmpl_ilp64_mp  -lamath -lastring -lm " -DCMAKE_C_FLAGS="-mcpu=cobalt-100 -mtune=cobalt-100 -flto -Ofast  -DINTEGER64 -I${ARMPL_DIR}/include -larmpl_ilp64_mp  -lamath -lastring -lm " and ggml detection results:
Adding CPU backend variant ggml-cpu: -mcpu=neoverse-n2+crc+sve2-aes+sve2-sha3+sve2-sm4+norng+nossbs+dotprod+i8mm+sve+nosme

The best result was this with llama.cpp, usable but we are looking for better performance, this is why we turned to your project:
| deepseek2 671B Q4_0            | 353.47 GiB |   671.03 B | RPC        |  99 |  1 |           pp512 |         43.27 ± 0.16 |
| deepseek2 671B Q4_0            | 353.47 GiB |   671.03 B | RPC        |  99 |  1 |           tg128 |         10.97 ± 0.07 |

Please advise how can we further optimize Deepseek inference with your solution.

---

👤 **jagusztinl** commented the **2025-06-21** at **08:39:04**:<br>

Thank you for your answer, a bit detail explanation of the project:
-We are using Azure Cobalt ARM CPUs on spot VMs, (64 real core, 512Gb 12 channel very fast RAM) for 0.5USD/hour (!) instead of expensive GPU setups. The price/perforance ratio is unbeatable: our collegues can use DeepSeek privately for  80USD/month continuosly. without limits.
-We experimented with llama.cpp as the fastest inference engine, with this setup (optimized for Cobalt and linked with ARM performance libs): cmake  -DGGML_CPU_KLEIDIAI=ON -DCMAKE_CXX_FLAGS="-mcpu=cobalt-100 -mtune=cobalt-100 -flto -Ofast  -DINTEGER64 -I${ARMPL_DIR}/include -larmpl_ilp64_mp  -lamath -lastring -lm " -DCMAKE_C_FLAGS="-mcpu=cobalt-100 -mtune=cobalt-100 -flto -Ofast  -DINTEGER64 -I${ARMPL_DIR}/include -larmpl_ilp64_mp  -lamath -lastring -lm " and ggml detection results:
Adding CPU backend variant ggml-cpu: -mcpu=neoverse-n2+crc+sve2-aes+sve2-sha3+sve2-sm4+norng+nossbs+dotprod+i8mm+sve+nosme

The best result was this with llama.cpp, usable but we are looking for better performance, this is why we turned to your project:
| deepseek2 671B Q4_0            | 353.47 GiB |   671.03 B | RPC        |  99 |  1 |           pp512 |         43.27 ± 0.16 |
| deepseek2 671B Q4_0            | 353.47 GiB |   671.03 B | RPC        |  99 |  1 |           tg128 |         10.97 ± 0.07 |

Please advise how can we further optimize Deepseek inference with your solution.

---

👤 **jagusztinl** commented the **2025-06-21** at **08:47:17**:<br>

About the garbage problem:
If I do not use -DGGML_SVE=ON during compilation, it is not detected:
use system_info: n_threads = 64 / 64 | AVX = 0 | AVX_VNNI = 0 | AVX2 = 0 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 0 | NEON = 1 | SVE = 0 | ARM_FMA = 1 | F16C = 0 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 0 | SSSE3 = 0 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 |
instead of:
system_info: n_threads = 64 / 64 | AVX = 0 | AVX_VNNI = 0 | AVX2 = 0 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 0 | NEON = 1 | SVE = 1 | ARM_FMA = 1 | F16C = 0 | FP16_VA = 1 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 0 | SSSE3 = 0 | VSX = 0 | MATMUL_INT8 = 1 | LLAMAFILE = 0 |
this is the root cause of the garbage output on this server.

---

👤 **ikawrakow** commented the **2025-06-21** at **09:22:44**:<br>

I'm open to working on optimizing this project for SVE, but it is a hobby project of mine without commercial backing, so I develop/test on the CPU platforms I have access to (`AVX2`, `Zen4`, `ARM_NEON` on an M2-Max CPU).

What are you looking to optimize? I read somewhere that the "typical enterprise" workflow (whatever that means) involves processing `N` token prompts and then generating a response with `N/10` tokens. Or are the prompts of your customers really short, but they are looking for long answers, so TG speed is all that matters?  What about context? Your customers never have a longer exchange with the LLM but always just ask a single short question, get the answer, and close the session?

---

👤 **saood06** commented the **2025-06-21** at **16:16:04**:<br>

So can you try experimenting with `-DGGML_ARCH_FLAGS=` added by #347. Some users have had some success with it see: https://github.com/ikawrakow/ik_llama.cpp/issues/345#issuecomment-2831460138. It looks like you have done similar experimenting with llama.cpp, in optimizing it.

---

👤 **jagusztinl** commented the **2025-06-23** at **15:34:50**:<br>

Using this:
cmake -B ./build -DGGML_LTO=ON -DCMAKE_CXX_FLAGS=" -flto -Ofast  -DINTEGER64 -I${ARMPL_DIR}/include -larmpl_ilp64_mp  -lamath -lastring -lm " -DCMAKE_C_FLAGS=" -flto -Ofast  -DINTEGER64 -I${ARMPL_DIR}/include -larmpl_ilp64_mp  -lamath -lastring -lm " -DGGML_ARCH_FLAGS="-mcpu=neoverse-n2+crc+sve2-aes+sve2-sha3+sve2-sm4+norng+nossbs+dotprod+i8mm+sve+nosme"

ik_llama.cpp is winner :-)
| deepseek2 671B Q4_0            | 354.49 GiB |   672.05 B | CPU        |      64 |   q8_0 |   q8_0 |  1 |   2 |  2048 |   1 |    1 |         pp512 |     68.19 ± 0.16 |
| deepseek2 671B Q4_0            | 354.49 GiB |   672.05 B | CPU        |      64 |   q8_0 |   q8_0 |  1 |   2 |  2048 |   1 |    1 |         tg128 |     11.54 ± 0.07 |

---

👤 **saood06** commented the **2025-06-23** at **20:40:31**:<br>

>ik_llama.cpp is winner :-)

Glad you found some settings that made it perform well for you.

Why are you using MLA 2 now instead of 3 like you were previously (assuming headers stayed the same)? Also two tips, using a high ubatch size can boost PP (assuming you can make use of those larger batch sizes) and you can use [sweep-bench](https://github.com/ikawrakow/ik_llama.cpp/tree/main/examples/sweep-bench) for benchmarking and seeing how much your performance drops with context (it even comes with it's own plotting tool).

>We are using Azure Cobalt ARM CPUs on spot VMs, (64 real core, 512Gb 12 channel very fast RAM) for 0.5USD/hour (!)

I was going to suggest going to the 48 core 384GB version since Deepseek would still fit, but looking at the spot price the 64 core is cheaper. (I did find certain regions where it goes down to $0.413).

By my math that does seem a bit cheaper than most inference providers (even using your cost), but I think your cost advantage goes away as performance will drop as context climbs.

>our collegues can use DeepSeek privately for 80USD/month continuously without limits

If your use case allows for it, you may be able to get better performance with batching, that way multiple people can be served by a single instance. Performance of that can be seen with [batched-bench](https://github.com/ikawrakow/ik_llama.cpp/tree/main/examples/batched-bench).

---

👤 **ikawrakow** commented the **2025-06-26** at **06:49:28**:<br>

No need to keep this open.