### 🐛 [#455](https://github.com/ikawrakow/ik_llama.cpp/issues/455) - Bug: KV cache is never reused in OpenAI compatible Chat Completion api

| **Author** | `luzamm` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-24 |
| **Updated** | 2025-05-28 |

---

#### Description

### What happened?

I use OpenAI compatible API Chat Completion on both Open WebUI and SillyTavern, the whole prompt will **always** re-evaluate from position p0 when I just regenerate the last message.
The log shows I generated 1 time and retried 2 times, totally 3 time to generate the answer. Ideally, it should use kv cache on last 2 retries because nothing changed but it didn't use the cache.

model: unsloth/DeepSeek-V3-0324-GGUF-UD
system prompt: You are a helpful assistant.
message1: Introduce AMD.
message2: Just tell me who is the CEO?
I regenerated message2's reply

Text Completion API and llama-server's built-in web server seems work well, cache was used.

I tried llama.cpp and it work well both in Chat Completion API and Text Completion API.

llama.cpp info (**not** ik_llama.cpp)
cmake -B ./build -DGGML_CUDA=ON -DGGML_BLAS=OFF -DGGML_CUDA_FA_ALL_QUANTS=ON

root@pve:~/llm/llama.cpp# ./build/bin/llama-server --version
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3080, compute capability 8.6, VMM: yes
version: 5474 (259469c4)
built with cc (Debian 12.2.0-14) 12.2.0 for x86_64-linux-gnu


### Name and Version

ik_llama.cpp build command:
cmake -B ./build -DGGML_CUDA=ON -DGGML_BLAS=OFF

version: 3712 (c7ecd4e2)
built with cc (Debian 12.2.0-14) 12.2.0 for x86_64-linux-gnu

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
root@pve:~/llm/ik_llama.cpp# ./build/bin/llama-server     --alias unsloth/DeepSeek-R1-Q4_K_XL     --model /mnt/pve/PE8110/llm/models/DeepSeek-V3-0324-UD-Q4_K_XL/DeepSeek-V3-0324-UD-Q4_K_XL-00001-of-00008.gguf    -rtr     --ctx-size 32768     -ctk q8_0     -mla 3 -fa     -amb 512     -fmoe     --n-gpu-layers 999     --override-tensor exps=CPU     --parallel 1     --threads 60     --host 0.0.0.0     --port 5001
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3080, compute capability 8.6, VMM: yes
INFO [                    main] build info | tid="137281198051328" timestamp=1748126804 build=3712 commit="c7ecd4e2"
INFO [                    main] system info | tid="137281198051328" timestamp=1748126804 n_threads=60 n_threads_batch=-1 total_threads=128 system_info="AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: additional 7 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 64 key-value pairs and 1086 tensors from /mnt/pve/PE8110/llm/models/DeepSeek-V3-0324-UD-Q4_K_XL/DeepSeek-V3-0324-UD-Q4_K_XL-00001-of-00008.gguf (version GGUF V3 (latest))
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
llama_model_loader: - kv  56:                          general.file_type u32              = 15
llama_model_loader: - kv  57:                      quantize.imatrix.file str              = DeepSeek-V3-0324-GGUF/imatrix_unsloth...
llama_model_loader: - kv  58:                   quantize.imatrix.dataset str              = unsloth_calibration_DeepSeek-V3-0324.txt
llama_model_loader: - kv  59:             quantize.imatrix.entries_count i32              = 720
llama_model_loader: - kv  60:              quantize.imatrix.chunks_count i32              = 60
llama_model_loader: - kv  61:                                   split.no u16              = 0
llama_model_loader: - kv  62:                        split.tensors.count i32              = 1086
llama_model_loader: - kv  63:                                split.count u16              = 8
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  122 tensors
llama_model_loader: - type q4_K:  485 tensors
llama_model_loader: - type q5_K:   95 tensors
llama_model_loader: - type q6_K:   23 tensors
==========================================================================
Detected incompatible DeepSeek model.
Will try to fix, but there are no guarantees

*** Your prompt processing speed will be crippled ***

Consider making your own ik_llama.cpp compatible model or
ask the model provider to make one for you,
==========================================================================
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
llm_load_print_meta: model size       = 357.623 GiB (4.578 BPW) 
llm_load_print_meta: repeating layers = 356.429 GiB (4.575 BPW, 669.173 B parameters)
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
llm_load_tensors: ggml ctx size =    0.89 MiB
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
Tensor blk.60.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size = 355712.00 MiB
llm_load_tensors:  CUDA_Host buffer size =   497.11 MiB
llm_load_tensors:      CUDA0 buffer size =  9996.68 MiB
....................................................................................................
============ llm_prepare_mla: need to compute 61 wkv_b tensors
Computed blk.0.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.1.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.2.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.3.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.4.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.5.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.6.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.7.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.8.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.9.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.10.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.11.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.12.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.13.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.14.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.15.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.16.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.17.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.18.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.19.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.20.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.21.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.22.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.23.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.24.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.25.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.26.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.27.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.28.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.29.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.30.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.31.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.32.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.33.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.34.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.35.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.36.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.37.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.38.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.39.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.40.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.41.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.42.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.43.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.44.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.45.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.46.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.47.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.48.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.49.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.50.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.51.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.52.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.53.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.54.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.55.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.56.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.57.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.58.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.59.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.60.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
============ Repacked 174 tensors
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 3
llama_new_context_with_model: attn_max_b = 512
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init:      CUDA0 KV buffer size =  1166.65 MiB
llama_new_context_with_model: KV self size  = 1166.62 MiB, c^KV (q8_0): 1166.62 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.99 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =  3425.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   176.01 MiB
llama_new_context_with_model: graph nodes  = 8245
llama_new_context_with_model: graph splits = 118
INFO [                    init] initializing slots | tid="137281198051328" timestamp=1748127054 n_slots=1
INFO [                    init] new slot | tid="137281198051328" timestamp=1748127054 id_slot=0 n_ctx_slot=32768
INFO [                    main] model loaded | tid="137281198051328" timestamp=1748127054
INFO [                    main] chat template | tid="137281198051328" timestamp=1748127054 chat_example="You are a helpful assistant\n\n<｜User｜>Hello<｜Assistant｜>Hi there<｜end▁of▁sentence｜><｜User｜>How are you?<｜Assistant｜>" built_in=true
INFO [                    main] HTTP server listening | tid="137281198051328" timestamp=1748127054 n_threads_http="127" port="5001" hostname="0.0.0.0"
INFO [            update_slots] all slots are idle | tid="137281198051328" timestamp=1748127054
INFO [      log_server_request] request | tid="136894792617984" timestamp=1748127109 remote_addr="192.168.123.99" remote_port=39142 status=200 method="GET" path="/v1/models" params={}
INFO [      log_server_request] request | tid="136894775832576" timestamp=1748127145 remote_addr="192.168.123.99" remote_port=33258 status=200 method="GET" path="/v1/models" params={}
INFO [      log_server_request] request | tid="136894801010688" timestamp=1748127169 remote_addr="192.168.123.99" remote_port=57604 status=200 method="GET" path="/v1/models" params={}
INFO [      log_server_request] request | tid="137279920132096" timestamp=1748127207 remote_addr="192.168.123.99" remote_port=39902 status=200 method="GET" path="/v1/models" params={}
INFO [   launch_slot_with_task] slot is processing task | tid="137281198051328" timestamp=1748127207 id_slot=0 id_task=0
INFO [            update_slots] kv cache rm [p0, end) | tid="137281198051328" timestamp=1748127207 id_slot=0 id_task=0 p0=0
INFO [           print_timings] prompt eval time     =    1170.90 ms /    13 tokens (   90.07 ms per token,    11.10 tokens per second) | tid="137281198051328" timestamp=1748127268 id_slot=0 id_task=0 t_prompt_processing=1170.897 n_prompt_tokens_processed=13 t_token=90.06899999999999 n_tokens_second=11.10259911845363
INFO [           print_timings] generation eval time =   59250.24 ms /   514 runs   (  115.27 ms per token,     8.68 tokens per second) | tid="137281198051328" timestamp=1748127268 id_slot=0 id_task=0 t_token_generation=59250.237 n_decoded=514 t_token=115.27283463035019 n_tokens_second=8.675070784948927
INFO [           print_timings]           total time =   60421.13 ms | tid="137281198051328" timestamp=1748127268 id_slot=0 id_task=0 t_prompt_processing=1170.897 t_token_generation=59250.237 t_total=60421.134
INFO [            update_slots] slot released | tid="137281198051328" timestamp=1748127268 id_slot=0 id_task=0 n_ctx=32768 n_past=526 n_system_tokens=0 n_cache_tokens=0 truncated=false
INFO [            update_slots] all slots are idle | tid="137281198051328" timestamp=1748127268
INFO [      log_server_request] request | tid="137279819341824" timestamp=1748127268 remote_addr="192.168.123.99" remote_port=39910 status=200 method="POST" path="/v1/chat/completions" params={}
INFO [            update_slots] all slots are idle | tid="137281198051328" timestamp=1748127268
INFO [      log_server_request] request | tid="137279737688064" timestamp=1748127286 remote_addr="192.168.123.99" remote_port=43354 status=200 method="GET" path="/v1/models" params={}
INFO [   launch_slot_with_task] slot is processing task | tid="137281198051328" timestamp=1748127286 id_slot=0 id_task=516
INFO [            update_slots] kv cache rm [p0, end) | tid="137281198051328" timestamp=1748127286 id_slot=0 id_task=516 p0=0
INFO [           print_timings] prompt eval time     =    6383.32 ms /   536 tokens (   11.91 ms per token,    83.97 tokens per second) | tid="137281198051328" timestamp=1748127305 id_slot=0 id_task=516 t_prompt_processing=6383.325 n_prompt_tokens_processed=536 t_token=11.90918843283582 n_tokens_second=83.96877802712537
INFO [           print_timings] generation eval time =   12977.77 ms /   113 runs   (  114.85 ms per token,     8.71 tokens per second) | tid="137281198051328" timestamp=1748127305 id_slot=0 id_task=516 t_token_generation=12977.773 n_decoded=113 t_token=114.84754867256636 n_tokens_second=8.707194986381717
INFO [           print_timings]           total time =   19361.10 ms | tid="137281198051328" timestamp=1748127305 id_slot=0 id_task=516 t_prompt_processing=6383.325 t_token_generation=12977.773 t_total=19361.097999999998
INFO [            update_slots] slot released | tid="137281198051328" timestamp=1748127305 id_slot=0 id_task=516 n_ctx=32768 n_past=648 n_system_tokens=0 n_cache_tokens=0 truncated=false
INFO [            update_slots] all slots are idle | tid="137281198051328" timestamp=1748127305
INFO [      log_server_request] request | tid="137279729295360" timestamp=1748127305 remote_addr="192.168.123.99" remote_port=43366 status=200 method="POST" path="/v1/chat/completions" params={}
INFO [            update_slots] all slots are idle | tid="137281198051328" timestamp=1748127305
INFO [      log_server_request] request | tid="137279720902656" timestamp=1748127309 remote_addr="192.168.123.99" remote_port=51502 status=200 method="GET" path="/v1/models" params={}
INFO [   launch_slot_with_task] slot is processing task | tid="137281198051328" timestamp=1748127309 id_slot=0 id_task=631
INFO [            update_slots] kv cache rm [p0, end) | tid="137281198051328" timestamp=1748127309 id_slot=0 id_task=631 p0=0
INFO [           print_timings] prompt eval time     =    6326.97 ms /   536 tokens (   11.80 ms per token,    84.72 tokens per second) | tid="137281198051328" timestamp=1748127329 id_slot=0 id_task=631 t_prompt_processing=6326.966 n_prompt_tokens_processed=536 t_token=11.80404104477612 n_tokens_second=84.71675049304832
INFO [           print_timings] generation eval time =   12948.27 ms /   113 runs   (  114.59 ms per token,     8.73 tokens per second) | tid="137281198051328" timestamp=1748127329 id_slot=0 id_task=631 t_token_generation=12948.269 n_decoded=113 t_token=114.58645132743364 n_tokens_second=8.727035250812289
INFO [           print_timings]           total time =   19275.24 ms | tid="137281198051328" timestamp=1748127329 id_slot=0 id_task=631 t_prompt_processing=6326.966 t_token_generation=12948.269 t_total=19275.235
INFO [            update_slots] slot released | tid="137281198051328" timestamp=1748127329 id_slot=0 id_task=631 n_ctx=32768 n_past=648 n_system_tokens=0 n_cache_tokens=0 truncated=false
INFO [            update_slots] all slots are idle | tid="137281198051328" timestamp=1748127329
INFO [      log_server_request] request | tid="137279712509952" timestamp=1748127329 remote_addr="192.168.123.99" remote_port=51508 status=200 method="POST" path="/v1/chat/completions" params={}
INFO [            update_slots] all slots are idle | tid="137281198051328" timestamp=1748127329
INFO [      log_server_request] request | tid="137279704117248" timestamp=1748127337 remote_addr="192.168.123.99" remote_port=55810 status=200 method="GET" path="/v1/models" params={}
INFO [   launch_slot_with_task] slot is processing task | tid="137281198051328" timestamp=1748127337 id_slot=0 id_task=746
INFO [            update_slots] kv cache rm [p0, end) | tid="137281198051328" timestamp=1748127337 id_slot=0 id_task=746 p0=0
INFO [           print_timings] prompt eval time     =    6375.81 ms /   536 tokens (   11.90 ms per token,    84.07 tokens per second) | tid="137281198051328" timestamp=1748127356 id_slot=0 id_task=746 t_prompt_processing=6375.806 n_prompt_tokens_processed=536 t_token=11.895160447761194 n_tokens_second=84.06780256488356
INFO [           print_timings] generation eval time =   12939.86 ms /   113 runs   (  114.51 ms per token,     8.73 tokens per second) | tid="137281198051328" timestamp=1748127356 id_slot=0 id_task=746 t_token_generation=12939.857 n_decoded=113 t_token=114.51200884955752 n_tokens_second=8.73270856084422
INFO [           print_timings]           total time =   19315.66 ms | tid="137281198051328" timestamp=1748127356 id_slot=0 id_task=746 t_prompt_processing=6375.806 t_token_generation=12939.857 t_total=19315.663
INFO [            update_slots] slot released | tid="137281198051328" timestamp=1748127356 id_slot=0 id_task=746 n_ctx=32768 n_past=648 n_system_tokens=0 n_cache_tokens=0 truncated=false
INFO [            update_slots] all slots are idle | tid="137281198051328" timestamp=1748127356
INFO [      log_server_request] request | tid="137279695724544" timestamp=1748127356 remote_addr="192.168.123.99" remote_port=55822 status=200 method="POST" path="/v1/chat/completions" params={}
INFO [            update_slots] all slots are idle | tid="137281198051328" timestamp=1748127356
```

---

#### 💬 Conversation

👤 **saood06** commented the **2025-05-24** at **23:39:01**:<br>

Are you passing in `cache_prompt: true` in your request?

I know llama.cpp now defaults to it being on, but we do not do that here (would be trivial to change), so as it stands it will not reuse the cache unless you pass that.

Edit: Just want to add I use the server and I can get KV cache to be reused between prompts where the prefix is shared, so it does work for me with that passed in my requests.

---

👤 **saood06** commented the **2025-05-24** at **23:39:01**:<br>

Are you passing in `cache_prompt: true` in your request?

I know llama.cpp now defaults to it being on, but we do not do that here (would be trivial to change), so as it stands it will not reuse the cache unless you pass that.

---

👤 **ikawrakow** commented the **2025-05-25** at **04:32:30**:<br>

@saood06 Maybe we should change the default?

---

👤 **saood06** commented the **2025-05-25** at **04:49:04**:<br>

> [@saood06](https://github.com/saood06) Maybe we should change the default?

I agree, it's a trivial change and with the implementation of caching that we have here there is almost no reason to turn it off.

I've been tinkering with an alternative caching mechanism as I don't fully like the new one mainline has with chunking since I'm fairly certain there are quality losses with especially if done excessively with small chunks. My alternative is more involved and has other benefits but it's still nowhere close to being done or even draft PR ready.

---

👤 **luzamm** commented the **2025-05-25** at **08:55:07**:<br>

After passing cache_prompt:true , it worked well. But there are many webui do not pass this field and nowhere to add easily. Is it better to turn it on by default?

---

👤 **saood06** commented the **2025-05-25** at **09:17:43**:<br>

> After passing cache_prompt:true , it worked well. 

I am glad to hear that.

>But there are many webui do not pass this field and nowhere to add easily. Is it better to turn it on by default?

Yes, I will do that. I looked into it enough to deem it trivial, just haven't gotten around to it yet, but I will get to it. I'll mark this closed once the default is set.

---

👤 **Ph0rk0z** commented the **2025-05-25** at **16:28:04**:<br>

It never reprocess my cache because I used text completion with sillytavern. What happens when you reach the context limit? I know that mainline has some mechanism for that. Does it just reprocess context with every message post limit?

---

👤 **saood06** commented the **2025-05-28** at **01:00:43**:<br>

@luzamm 
Sorry for the delay, but the PR has been made that changes the default, and I have linked it to this issue to automatically close once it gets merged in.

@Ph0rk0z 
>It never reprocess my cache because I used text completion with sillytavern. What happens when you reach the context limit? I know that mainline has some mechanism for that. Does it just reprocess context with every message post limit?

There is a feature called context shifting that will shift the entire context window (by I think half?) while keeping the system_prompt (if used). This feature does not work for all models and in my own personal experience leads to a noticeable and often severe degradation in output quality, but for some of my use-cases it was fine. 

I have not used context shifting in a long time but as far as I can tell the implementation here is the same as the one I have experienced.

---

👤 **Ph0rk0z** commented the **2025-05-28** at **15:12:09**:<br>

>I have not used context shifting in a long time but as far as I can tell the implementation here is the same as the one I have experienced.

I thought it doesn't work here because it was forked before the implementation in main. There is no --cache-reuse flag and I see nothing about context shift. Only ever tried the implementation in ooba.

---

👤 **saood06** commented the **2025-05-28** at **22:04:21**:<br>

> I thought it doesn't work here because it was forked before the implementation in main. There is no --cache-reuse flag and I see nothing about context shift. Only ever tried the implementation in ooba.

You are talking about two different things. Context shifting (which allows for an "infinite" amount of chatting) is supported see the code [here](https://github.com/ikawrakow/ik_llama.cpp/blob/ccd6d9cdf6851f7042c48d682daf47bc0e2eca27/examples/server/server.cpp#L1946) but there is no documentation for it.

I do not plan to port over the `--cache-reuse` flag from mainline which allows for you to reuse chunks of the prompt since it results in quality losses (although when used reasonably those quality losses may be acceptable or even imperceptible). I am working on an alternative that will have different tradeoffs (it will actually be better for some situations, but worse in others since it won't chunk the cache).