### 🔀 [#409](https://github.com/ikawrakow/ik_llama.cpp/pull/409) - Enable faster prompt processing with mainline llama.cpp GGUFs

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-11 |
| **Updated** | 2025-05-12 |

---

#### Description

Mainline llama.cpp [PR 12901](https://github.com/ggml-org/llama.cpp/pull/12801), which added MLA support for DeepSeek models 2.5 months after MLA was available here, broke backwards compatibility. As a result,
the new DeepSeek GGUFs that started appearing on HF became compatible with `ik_llama.cpp`, so I added support for the incompatible GGUFs in #394. But using such crippled DeepSeek GGUF results in a much lower prompt processing performance. This is because the `attn_wkv_b` tensor is missing, so one cannot use `mla = 3`.

This PR removes this limitation. When `-mla 0 or 2 or 3` is specified on the command line, missing `attn_wkv_b` tensors are created on-the-fly while loading the model. This is basically the reverse of #259, where the `attn_wk_b` and `attn_wv_b`tensors necessary for MLA were computed from the `attn_wkv_b` tensors in the original DeepSeek GGUFs.

To show why this is useful, the following graph compares PP performance between the main branch and this PR. The `sweep-bench` command is
```
./bin/llama-sweep-bench -m $model -c 65536 -t 1 -ngl 100 -mla 3 -fa -fmoe -b 4096 -ub 4096
```
The model is a mainline `llama.cpp` DeepSeek-Lite GGUF with the `attn_wkv_b` tensors missing. In that case the `mla = 3` parameter will be converted to `mla = 1` on the main branch, but trigger the generation of the `attn_wkv_b` tensors in this PR (so `mla = 3` can be used). The model is quantized with `Q4_0`, the GPU is RTX-4080. The x-axis is `N_KV/1000`, where `N_KV` is the number of tokens in the KV cache. I have used a logarithmic scale for the y axis to better show the growing difference in performance with increasing `N_KV`.

![z11](https://github.com/user-attachments/assets/aa0ef1a0-459c-4caa-9b05-9d3395e3e83b)

---

#### 💬 Conversation

👤 **Panchovix** commented the **2025-05-11** at **19:03:47**:<br>

Testing this PR (on top of https://github.com/ikawrakow/ik_llama.cpp/pull/405 and https://github.com/ikawrakow/ik_llama.cpp/pull/408 PRs), here's a complete log when loading DeepSeek V3 0324 Q2_K_XL. Notably, I had to reduce 1 layer on CUDA 2 (compared to https://github.com/ikawrakow/ik_llama.cpp/pull/405#issuecomment-2869126831), as now CUDA 2 was getting OOM. I noticed the compute buffers are ~3.3GB each instead of 2GB and 400MB respectively for each despite using the -fa flag with -mla 3.

```
./llama-server -m '/DeepSeek-V3-0324-UD-Q2_K_XL-00001-of-00006.gguf' -c 16384 --no-mmap -v -ngl 999 -ot "blk.(0|1|2|3|4|5|6|7).ffn.=CUDA0" -ot "blk.(8|9|10|11).ffn.=CUDA1" -ot "blk.(12|13|14|15).ffn.=CUDA2" -ot "blk.(16|17|18|19|20|21|22|23|24|25).ffn.=CUDA3" -ot "ffn.*=CPU" -fa -mla 3 -mg 0 -ub 1024 -fmoe
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 4 CUDA devices:
  Device 0: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
  Device 1: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
  Device 2: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
  Device 3: NVIDIA RTX A6000, compute capability 8.6, VMM: yes
INFO [                    main] build info | tid="140558519128064" timestamp=1746988793 build=3682 commit="154a195f"
INFO [                    main] system info | tid="140558519128064" timestamp=1746988793 n_threads=8 n_threads_batch=-1 total_threads=16 system_info="AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
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
Tensor blk.0.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_gate.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_down.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_up.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_gate.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_down.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_up.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_gate.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_down.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_up.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_up_shexp.weight buffer type overriden to CUDA1
Tensor blk.9.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.9.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.9.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.9.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.9.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.9.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.9.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.9.ffn_up_shexp.weight buffer type overriden to CUDA1
Tensor blk.10.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.10.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.10.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.10.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.10.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.10.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.10.ffn_up_shexp.weight buffer type overriden to CUDA1
Tensor blk.11.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.11.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.11.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.11.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.11.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.11.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.11.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.11.ffn_up_shexp.weight buffer type overriden to CUDA1
Tensor blk.12.ffn_norm.weight buffer type overriden to CUDA2
Tensor blk.12.ffn_gate_inp.weight buffer type overriden to CUDA2
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CUDA2
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CUDA2
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CUDA2
Tensor blk.12.ffn_gate_shexp.weight buffer type overriden to CUDA2
Tensor blk.12.ffn_down_shexp.weight buffer type overriden to CUDA2
Tensor blk.12.ffn_up_shexp.weight buffer type overriden to CUDA2
Tensor blk.13.ffn_norm.weight buffer type overriden to CUDA2
Tensor blk.13.ffn_gate_inp.weight buffer type overriden to CUDA2
Tensor blk.13.ffn_gate_exps.weight buffer type overriden to CUDA2
Tensor blk.13.ffn_down_exps.weight buffer type overriden to CUDA2
Tensor blk.13.ffn_up_exps.weight buffer type overriden to CUDA2
Tensor blk.13.ffn_gate_shexp.weight buffer type overriden to CUDA2
Tensor blk.13.ffn_down_shexp.weight buffer type overriden to CUDA2
Tensor blk.13.ffn_up_shexp.weight buffer type overriden to CUDA2
Tensor blk.14.ffn_norm.weight buffer type overriden to CUDA2
Tensor blk.14.ffn_gate_inp.weight buffer type overriden to CUDA2
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CUDA2
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CUDA2
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CUDA2
Tensor blk.14.ffn_gate_shexp.weight buffer type overriden to CUDA2
Tensor blk.14.ffn_down_shexp.weight buffer type overriden to CUDA2
Tensor blk.14.ffn_up_shexp.weight buffer type overriden to CUDA2
Tensor blk.15.ffn_norm.weight buffer type overriden to CUDA2
Tensor blk.15.ffn_gate_inp.weight buffer type overriden to CUDA2
Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CUDA2
Tensor blk.15.ffn_down_exps.weight buffer type overriden to CUDA2
Tensor blk.15.ffn_up_exps.weight buffer type overriden to CUDA2
Tensor blk.15.ffn_gate_shexp.weight buffer type overriden to CUDA2
Tensor blk.15.ffn_down_shexp.weight buffer type overriden to CUDA2
Tensor blk.15.ffn_up_shexp.weight buffer type overriden to CUDA2
Tensor blk.16.ffn_norm.weight buffer type overriden to CUDA3
Tensor blk.16.ffn_gate_inp.weight buffer type overriden to CUDA3
Tensor blk.16.ffn_gate_exps.weight buffer type overriden to CUDA3
Tensor blk.16.ffn_down_exps.weight buffer type overriden to CUDA3
Tensor blk.16.ffn_up_exps.weight buffer type overriden to CUDA3
Tensor blk.16.ffn_gate_shexp.weight buffer type overriden to CUDA3
Tensor blk.16.ffn_down_shexp.weight buffer type overriden to CUDA3
Tensor blk.16.ffn_up_shexp.weight buffer type overriden to CUDA3
Tensor blk.17.ffn_norm.weight buffer type overriden to CUDA3
Tensor blk.17.ffn_gate_inp.weight buffer type overriden to CUDA3
Tensor blk.17.ffn_gate_exps.weight buffer type overriden to CUDA3
Tensor blk.17.ffn_down_exps.weight buffer type overriden to CUDA3
Tensor blk.17.ffn_up_exps.weight buffer type overriden to CUDA3
Tensor blk.17.ffn_gate_shexp.weight buffer type overriden to CUDA3
Tensor blk.17.ffn_down_shexp.weight buffer type overriden to CUDA3
Tensor blk.17.ffn_up_shexp.weight buffer type overriden to CUDA3
Tensor blk.18.ffn_norm.weight buffer type overriden to CUDA3
Tensor blk.18.ffn_gate_inp.weight buffer type overriden to CUDA3
Tensor blk.18.ffn_gate_exps.weight buffer type overriden to CUDA3
Tensor blk.18.ffn_down_exps.weight buffer type overriden to CUDA3
Tensor blk.18.ffn_up_exps.weight buffer type overriden to CUDA3
Tensor blk.18.ffn_gate_shexp.weight buffer type overriden to CUDA3
Tensor blk.18.ffn_down_shexp.weight buffer type overriden to CUDA3
Tensor blk.18.ffn_up_shexp.weight buffer type overriden to CUDA3
Tensor blk.19.ffn_norm.weight buffer type overriden to CUDA3
Tensor blk.19.ffn_gate_inp.weight buffer type overriden to CUDA3
Tensor blk.19.ffn_gate_exps.weight buffer type overriden to CUDA3
Tensor blk.19.ffn_down_exps.weight buffer type overriden to CUDA3
Tensor blk.19.ffn_up_exps.weight buffer type overriden to CUDA3
Tensor blk.19.ffn_gate_shexp.weight buffer type overriden to CUDA3
Tensor blk.19.ffn_down_shexp.weight buffer type overriden to CUDA3
Tensor blk.19.ffn_up_shexp.weight buffer type overriden to CUDA3
Tensor blk.20.ffn_norm.weight buffer type overriden to CUDA3
Tensor blk.20.ffn_gate_inp.weight buffer type overriden to CUDA3
Tensor blk.20.ffn_gate_exps.weight buffer type overriden to CUDA3
Tensor blk.20.ffn_down_exps.weight buffer type overriden to CUDA3
Tensor blk.20.ffn_up_exps.weight buffer type overriden to CUDA3
Tensor blk.20.ffn_gate_shexp.weight buffer type overriden to CUDA3
Tensor blk.20.ffn_down_shexp.weight buffer type overriden to CUDA3
Tensor blk.20.ffn_up_shexp.weight buffer type overriden to CUDA3
Tensor blk.21.ffn_norm.weight buffer type overriden to CUDA3
Tensor blk.21.ffn_gate_inp.weight buffer type overriden to CUDA3
Tensor blk.21.ffn_gate_exps.weight buffer type overriden to CUDA3
Tensor blk.21.ffn_down_exps.weight buffer type overriden to CUDA3
Tensor blk.21.ffn_up_exps.weight buffer type overriden to CUDA3
Tensor blk.21.ffn_gate_shexp.weight buffer type overriden to CUDA3
Tensor blk.21.ffn_down_shexp.weight buffer type overriden to CUDA3
Tensor blk.21.ffn_up_shexp.weight buffer type overriden to CUDA3
Tensor blk.22.ffn_norm.weight buffer type overriden to CUDA3
Tensor blk.22.ffn_gate_inp.weight buffer type overriden to CUDA3
Tensor blk.22.ffn_gate_exps.weight buffer type overriden to CUDA3
Tensor blk.22.ffn_down_exps.weight buffer type overriden to CUDA3
Tensor blk.22.ffn_up_exps.weight buffer type overriden to CUDA3
Tensor blk.22.ffn_gate_shexp.weight buffer type overriden to CUDA3
Tensor blk.22.ffn_down_shexp.weight buffer type overriden to CUDA3
Tensor blk.22.ffn_up_shexp.weight buffer type overriden to CUDA3
Tensor blk.23.ffn_norm.weight buffer type overriden to CUDA3
Tensor blk.23.ffn_gate_inp.weight buffer type overriden to CUDA3
Tensor blk.23.ffn_gate_exps.weight buffer type overriden to CUDA3
Tensor blk.23.ffn_down_exps.weight buffer type overriden to CUDA3
Tensor blk.23.ffn_up_exps.weight buffer type overriden to CUDA3
Tensor blk.23.ffn_gate_shexp.weight buffer type overriden to CUDA3
Tensor blk.23.ffn_down_shexp.weight buffer type overriden to CUDA3
Tensor blk.23.ffn_up_shexp.weight buffer type overriden to CUDA3
Tensor blk.24.ffn_norm.weight buffer type overriden to CUDA3
Tensor blk.24.ffn_gate_inp.weight buffer type overriden to CUDA3
Tensor blk.24.ffn_gate_exps.weight buffer type overriden to CUDA3
Tensor blk.24.ffn_down_exps.weight buffer type overriden to CUDA3
Tensor blk.24.ffn_up_exps.weight buffer type overriden to CUDA3
Tensor blk.24.ffn_gate_shexp.weight buffer type overriden to CUDA3
Tensor blk.24.ffn_down_shexp.weight buffer type overriden to CUDA3
Tensor blk.24.ffn_up_shexp.weight buffer type overriden to CUDA3
Tensor blk.25.ffn_norm.weight buffer type overriden to CUDA3
Tensor blk.25.ffn_gate_inp.weight buffer type overriden to CUDA3
Tensor blk.25.ffn_gate_exps.weight buffer type overriden to CUDA3
Tensor blk.25.ffn_down_exps.weight buffer type overriden to CUDA3
Tensor blk.25.ffn_up_exps.weight buffer type overriden to CUDA3
Tensor blk.25.ffn_gate_shexp.weight buffer type overriden to CUDA3
Tensor blk.25.ffn_down_shexp.weight buffer type overriden to CUDA3
Tensor blk.25.ffn_up_shexp.weight buffer type overriden to CUDA3
Tensor blk.26.ffn_norm.weight buffer type overriden to CPU
Tensor blk.26.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.26.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_gate_shexp.weight buffer type overriden to CPU
Tensor blk.26.ffn_down_shexp.weight buffer type overriden to CPU
Tensor blk.26.ffn_up_shexp.weight buffer type overriden to CPU
Tensor blk.27.ffn_norm.weight buffer type overriden to CPU
Tensor blk.27.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.27.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_gate_shexp.weight buffer type overriden to CPU
Tensor blk.27.ffn_down_shexp.weight buffer type overriden to CPU
Tensor blk.27.ffn_up_shexp.weight buffer type overriden to CPU
Tensor blk.28.ffn_norm.weight buffer type overriden to CPU
Tensor blk.28.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.28.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_gate_shexp.weight buffer type overriden to CPU
Tensor blk.28.ffn_down_shexp.weight buffer type overriden to CPU
Tensor blk.28.ffn_up_shexp.weight buffer type overriden to CPU
Tensor blk.29.ffn_norm.weight buffer type overriden to CPU
Tensor blk.29.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.29.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_gate_shexp.weight buffer type overriden to CPU
Tensor blk.29.ffn_down_shexp.weight buffer type overriden to CPU
Tensor blk.29.ffn_up_shexp.weight buffer type overriden to CPU
Tensor blk.30.ffn_norm.weight buffer type overriden to CPU
Tensor blk.30.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.30.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_gate_shexp.weight buffer type overriden to CPU
Tensor blk.30.ffn_down_shexp.weight buffer type overriden to CPU
Tensor blk.30.ffn_up_shexp.weight buffer type overriden to CPU
Tensor blk.31.ffn_norm.weight buffer type overriden to CPU
Tensor blk.31.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.31.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_gate_shexp.weight buffer type overriden to CPU
Tensor blk.31.ffn_down_shexp.weight buffer type overriden to CPU
Tensor blk.31.ffn_up_shexp.weight buffer type overriden to CPU
Tensor blk.32.ffn_norm.weight buffer type overriden to CPU
Tensor blk.32.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.32.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_gate_shexp.weight buffer type overriden to CPU
Tensor blk.32.ffn_down_shexp.weight buffer type overriden to CPU
Tensor blk.32.ffn_up_shexp.weight buffer type overriden to CPU
Tensor blk.33.ffn_norm.weight buffer type overriden to CPU
Tensor blk.33.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.33.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_gate_shexp.weight buffer type overriden to CPU
Tensor blk.33.ffn_down_shexp.weight buffer type overriden to CPU
Tensor blk.33.ffn_up_shexp.weight buffer type overriden to CPU
Tensor blk.34.ffn_norm.weight buffer type overriden to CPU
Tensor blk.34.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.34.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_gate_shexp.weight buffer type overriden to CPU
Tensor blk.34.ffn_down_shexp.weight buffer type overriden to CPU
Tensor blk.34.ffn_up_shexp.weight buffer type overriden to CPU
Tensor blk.35.ffn_norm.weight buffer type overriden to CPU
Tensor blk.35.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.35.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_gate_shexp.weight buffer type overriden to CPU
Tensor blk.35.ffn_down_shexp.weight buffer type overriden to CPU
Tensor blk.35.ffn_up_shexp.weight buffer type overriden to CPU
Tensor blk.36.ffn_norm.weight buffer type overriden to CPU
Tensor blk.36.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.36.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_gate_shexp.weight buffer type overriden to CPU
Tensor blk.36.ffn_down_shexp.weight buffer type overriden to CPU
Tensor blk.36.ffn_up_shexp.weight buffer type overriden to CPU
Tensor blk.37.ffn_norm.weight buffer type overriden to CPU
Tensor blk.37.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.37.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_gate_shexp.weight buffer type overriden to CPU
Tensor blk.37.ffn_down_shexp.weight buffer type overriden to CPU
Tensor blk.37.ffn_up_shexp.weight buffer type overriden to CPU
Tensor blk.38.ffn_norm.weight buffer type overriden to CPU
Tensor blk.38.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.38.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_gate_shexp.weight buffer type overriden to CPU
Tensor blk.38.ffn_down_shexp.weight buffer type overriden to CPU
Tensor blk.38.ffn_up_shexp.weight buffer type overriden to CPU
Tensor blk.39.ffn_norm.weight buffer type overriden to CPU
Tensor blk.39.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.39.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_gate_shexp.weight buffer type overriden to CPU
Tensor blk.39.ffn_down_shexp.weight buffer type overriden to CPU
Tensor blk.39.ffn_up_shexp.weight buffer type overriden to CPU
Tensor blk.40.ffn_norm.weight buffer type overriden to CPU
Tensor blk.40.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.40.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_gate_shexp.weight buffer type overriden to CPU
Tensor blk.40.ffn_down_shexp.weight buffer type overriden to CPU
Tensor blk.40.ffn_up_shexp.weight buffer type overriden to CPU
Tensor blk.41.ffn_norm.weight buffer type overriden to CPU
Tensor blk.41.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.41.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_gate_shexp.weight buffer type overriden to CPU
Tensor blk.41.ffn_down_shexp.weight buffer type overriden to CPU
Tensor blk.41.ffn_up_shexp.weight buffer type overriden to CPU
Tensor blk.42.ffn_norm.weight buffer type overriden to CPU
Tensor blk.42.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.42.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_gate_shexp.weight buffer type overriden to CPU
Tensor blk.42.ffn_down_shexp.weight buffer type overriden to CPU
Tensor blk.42.ffn_up_shexp.weight buffer type overriden to CPU
Tensor blk.43.ffn_norm.weight buffer type overriden to CPU
Tensor blk.43.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.43.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_gate_shexp.weight buffer type overriden to CPU
Tensor blk.43.ffn_down_shexp.weight buffer type overriden to CPU
Tensor blk.43.ffn_up_shexp.weight buffer type overriden to CPU
Tensor blk.44.ffn_norm.weight buffer type overriden to CPU
Tensor blk.44.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.44.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_gate_shexp.weight buffer type overriden to CPU
Tensor blk.44.ffn_down_shexp.weight buffer type overriden to CPU
Tensor blk.44.ffn_up_shexp.weight buffer type overriden to CPU
Tensor blk.45.ffn_norm.weight buffer type overriden to CPU
Tensor blk.45.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.45.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_gate_shexp.weight buffer type overriden to CPU
Tensor blk.45.ffn_down_shexp.weight buffer type overriden to CPU
Tensor blk.45.ffn_up_shexp.weight buffer type overriden to CPU
Tensor blk.46.ffn_norm.weight buffer type overriden to CPU
Tensor blk.46.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.46.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_gate_shexp.weight buffer type overriden to CPU
Tensor blk.46.ffn_down_shexp.weight buffer type overriden to CPU
Tensor blk.46.ffn_up_shexp.weight buffer type overriden to CPU
Tensor blk.47.ffn_norm.weight buffer type overriden to CPU
Tensor blk.47.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.47.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_gate_shexp.weight buffer type overriden to CPU
Tensor blk.47.ffn_down_shexp.weight buffer type overriden to CPU
Tensor blk.47.ffn_up_shexp.weight buffer type overriden to CPU
Tensor blk.48.ffn_norm.weight buffer type overriden to CPU
Tensor blk.48.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.48.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_gate_shexp.weight buffer type overriden to CPU
Tensor blk.48.ffn_down_shexp.weight buffer type overriden to CPU
Tensor blk.48.ffn_up_shexp.weight buffer type overriden to CPU
Tensor blk.49.ffn_norm.weight buffer type overriden to CPU
Tensor blk.49.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.49.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_gate_shexp.weight buffer type overriden to CPU
Tensor blk.49.ffn_down_shexp.weight buffer type overriden to CPU
Tensor blk.49.ffn_up_shexp.weight buffer type overriden to CPU
Tensor blk.50.ffn_norm.weight buffer type overriden to CPU
Tensor blk.50.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.50.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_gate_shexp.weight buffer type overriden to CPU
Tensor blk.50.ffn_down_shexp.weight buffer type overriden to CPU
Tensor blk.50.ffn_up_shexp.weight buffer type overriden to CPU
Tensor blk.51.ffn_norm.weight buffer type overriden to CPU
Tensor blk.51.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.51.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_gate_shexp.weight buffer type overriden to CPU
Tensor blk.51.ffn_down_shexp.weight buffer type overriden to CPU
Tensor blk.51.ffn_up_shexp.weight buffer type overriden to CPU
Tensor blk.52.ffn_norm.weight buffer type overriden to CPU
Tensor blk.52.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.52.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_gate_shexp.weight buffer type overriden to CPU
Tensor blk.52.ffn_down_shexp.weight buffer type overriden to CPU
Tensor blk.52.ffn_up_shexp.weight buffer type overriden to CPU
Tensor blk.53.ffn_norm.weight buffer type overriden to CPU
Tensor blk.53.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.53.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_gate_shexp.weight buffer type overriden to CPU
Tensor blk.53.ffn_down_shexp.weight buffer type overriden to CPU
Tensor blk.53.ffn_up_shexp.weight buffer type overriden to CPU
Tensor blk.54.ffn_norm.weight buffer type overriden to CPU
Tensor blk.54.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.54.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_gate_shexp.weight buffer type overriden to CPU
Tensor blk.54.ffn_down_shexp.weight buffer type overriden to CPU
Tensor blk.54.ffn_up_shexp.weight buffer type overriden to CPU
Tensor blk.55.ffn_norm.weight buffer type overriden to CPU
Tensor blk.55.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.55.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_gate_shexp.weight buffer type overriden to CPU
Tensor blk.55.ffn_down_shexp.weight buffer type overriden to CPU
Tensor blk.55.ffn_up_shexp.weight buffer type overriden to CPU
Tensor blk.56.ffn_norm.weight buffer type overriden to CPU
Tensor blk.56.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.56.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_gate_shexp.weight buffer type overriden to CPU
Tensor blk.56.ffn_down_shexp.weight buffer type overriden to CPU
Tensor blk.56.ffn_up_shexp.weight buffer type overriden to CPU
Tensor blk.57.ffn_norm.weight buffer type overriden to CPU
Tensor blk.57.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.57.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_gate_shexp.weight buffer type overriden to CPU
Tensor blk.57.ffn_down_shexp.weight buffer type overriden to CPU
Tensor blk.57.ffn_up_shexp.weight buffer type overriden to CPU
Tensor blk.58.ffn_norm.weight buffer type overriden to CPU
Tensor blk.58.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.58.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_gate_shexp.weight buffer type overriden to CPU
Tensor blk.58.ffn_down_shexp.weight buffer type overriden to CPU
Tensor blk.58.ffn_up_shexp.weight buffer type overriden to CPU
Tensor blk.59.ffn_norm.weight buffer type overriden to CPU
Tensor blk.59.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.59.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_gate_shexp.weight buffer type overriden to CPU
Tensor blk.59.ffn_down_shexp.weight buffer type overriden to CPU
Tensor blk.59.ffn_up_shexp.weight buffer type overriden to CPU
Tensor blk.60.ffn_norm.weight buffer type overriden to CPU
Tensor blk.60.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.60.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_gate_shexp.weight buffer type overriden to CPU
Tensor blk.60.ffn_down_shexp.weight buffer type overriden to CPU
Tensor blk.60.ffn_up_shexp.weight buffer type overriden to CPU
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size = 138767.64 MiB
llm_load_tensors:  CUDA_Host buffer size =   497.11 MiB
llm_load_tensors:      CUDA0 buffer size = 22188.53 MiB
llm_load_tensors:      CUDA1 buffer size = 17471.11 MiB
llm_load_tensors:      CUDA2 buffer size = 17472.86 MiB
llm_load_tensors:      CUDA3 buffer size = 42378.83 MiB
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
Computed blk.15.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.16.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.17.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.18.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.19.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.20.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.21.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.22.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.23.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.24.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.25.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.26.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.27.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
Computed blk.28.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
Computed blk.29.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
Computed blk.30.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
Computed blk.31.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
Computed blk.32.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
Computed blk.33.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
Computed blk.34.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
Computed blk.35.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
Computed blk.36.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
Computed blk.37.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
Computed blk.38.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
Computed blk.39.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA3
Computed blk.40.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA3
Computed blk.41.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA3
Computed blk.42.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA3
Computed blk.43.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA3
Computed blk.44.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA3
Computed blk.45.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA3
Computed blk.46.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA3
Computed blk.47.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA3
Computed blk.48.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA3
Computed blk.49.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA3
Computed blk.50.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA3
Computed blk.51.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA3
Computed blk.52.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA3
Computed blk.53.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA3
Computed blk.54.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA3
Computed blk.55.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA3
Computed blk.56.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA3
Computed blk.57.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA3
Computed blk.58.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA3
Computed blk.59.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA3
Computed blk.60.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA3
llama_new_context_with_model: n_ctx      = 16384
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 1024
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 3
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init:      CUDA0 KV buffer size =   270.00 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =   216.00 MiB
llama_kv_cache_init:      CUDA2 KV buffer size =   216.00 MiB
llama_kv_cache_init:      CUDA3 KV buffer size =   396.00 MiB
llama_new_context_with_model: KV self size  = 1098.00 MiB, c^KV (f16): 1098.00 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.99 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      CUDA0 compute buffer size =  3444.00 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =  3362.00 MiB
llama_new_context_with_model:      CUDA2 compute buffer size =  3362.00 MiB
llama_new_context_with_model:      CUDA3 compute buffer size =  3362.01 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =    92.01 MiB
llama_new_context_with_model: graph nodes  = 3487
llama_new_context_with_model: graph splits = 389
```

I noticed about at 15% improvement on PP t/s over https://github.com/ikawrakow/ik_llama.cpp/pull/405 PR, so then that means about 21% faster PP vs main llamacpp (and like 400% improvement (no joke lol) without the https://github.com/ikawrakow/ik_llama.cpp/pull/405 PR on ik llamacpp)

```
INFO [           print_timings] prompt eval time     =   24764.06 ms /  3003 tokens (    8.25 ms per token,   121.26 tokens per second) | tid="140558519128064" timestamp=1746989499 id_slot=0 id_task=464 t_prompt_processing=24764.059 n_prompt_tokens_processed=3003 t_token=8.246439893439893 n_tokens_second=121.2644502260312
INFO [           print_timings] generation eval time =   57949.04 ms /   456 runs   (  127.08 ms per token,     7.87 tokens per second) | tid="140558519128064" timestamp=1746989499 id_slot=0 id_task=464 t_token_generation=57949.044 n_decoded=456 t_token=127.08123684210527 n_tokens_second=7.868982273460801
INFO [           print_timings]           total time =   82713.10 ms | tid="140558519128064" timestamp=1746989499 id_slot=0 id_task=464 t_prompt_processing=24764.059 t_token_generation=57949.044 t_total=82713.103
```



Testing with -mla 2, compute buffers are 3.4GB as well vs -mla 1 with -fa. Here it got a small perf improvement (109 t/s PP vs 106 t/s PP).