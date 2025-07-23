### 🐛 [#572](https://github.com/ikawrakow/ik_llama.cpp/issues/572) - Bug: Oops(ggml_compute_forward_sum_rows_f32, ffn_moe_weights_sum-60): found nan, on DeepSeek V3/R1 on CUDA + CPU

| **Author** | `Panchovix` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-07-02 |
| **Updated** | 2025-07-05 |

---

#### Description

### What happened?

Hi there, thanks for all your work.

Sometimes, but not always, I get the issue mentioned in the title when running normally or when running some benchmarks.

**I'm not sure how to replicate it as it happens randomly.**

I can't managed to replicate it on main llamacpp at the moment.

This happens with either V3 0324 or R1 0528.

ikllamacpp was built as:

```
cmake -B build \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DGGML_BLAS=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="86;89;120" \
    -DGGML_IQK_FA_ALL_QUANTS=1 \
    -DGGML_SCHED_MAX_COPIES=1 \
    -DGGML_CUDA_IQK_FORCE_BF16=1 \
```

This happen when running for example, V3 with

```
./llama-server -m '/models_llm/DeepSeek-V3-0324-UD-Q3_K_XL-merged.gguf' -c 16384 --no-mmap -ngl 999 \
-ot "blk.(0|1|2|3|4|5|6).ffn.=CUDA0" \
-ot "blk.(7|8|9).ffn.=CUDA1" \
-ot "blk.(10|11|12).ffn.=CUDA2" \
-ot "blk.(13|14|15|16).ffn.=CUDA3" \
-ot "blk.(17|18|19).ffn.=CUDA4" \
-ot "blk.(20|21|22).ffn.=CUDA5" \
-ot "blk.(23|24|25|26|27|28|29|30|31).ffn.=CUDA6" \
-ot "blk.32.ffn_(norm|gate_inp|gate_exps|down_exps|up_exps|gate_shexp|down_shexp|up_shexp).weight=CUDA1" \
-ot "blk.33.ffn_(norm|gate_inp|gate_exps|down_exps|up_exps|gate_shexp|down_shexp|up_shexp).weight=CUDA2" \
-ot "blk.34.ffn_(norm|gate_inp|gate_exps|down_exps|up_exps|gate_shexp|down_shexp|up_shexp).weight=CUDA4" \
-ot "blk.35.ffn_(norm|gate_inp|gate_exps|down_exps|up_exps|gate_shexp|down_shexp|up_shexp).weight=CUDA5" \
-ot "ffn.*=CPU" \
-fa -mg 0 -ub 2048 -mla 3 -amb 512
```

Or R1 with

```
./llama-server -m '/models_llm/DeepSeek-R1-0528-IQ3_K_R4-merged.gguf' -c 32768 --no-mmap -ngl 999 \
-ot "blk.(0|1|2|3|4|5|6).ffn.=CUDA0" \
-ot "blk.(7|8|9).ffn.=CUDA1" \
-ot "blk.(10|11|12).ffn.=CUDA2" \
-ot "blk.(13|14|15|16).ffn.=CUDA3" \
-ot "blk.(17|18|19).ffn.=CUDA4" \
-ot "blk.(21|22|23).ffn.=CUDA5" \
-ot "blk.(24|25|26|27|28|29|30).ffn.=CUDA6" \
-ot "blk.31.ffn_(norm|gate_inp|gate_shexp|down_shexp|up_shexp).weight=CUDA1" \
-ot "blk.31.ffn_gate_exps.weight=CUDA1" \
-ot "blk.31.ffn_down_exps.weight=CUDA2" \
-ot "blk.32.ffn_(norm|gate_inp|gate_shexp|down_shexp|up_shexp).weight=CUDA0" \
-ot "blk.32.ffn_gate_exps.weight=CUDA0" \
-ot "blk.32.ffn_down_exps.weight=CUDA3" \
-ot "blk.32.ffn_up_exps.weight=CUDA1" \
-ot "blk.33.ffn_gate_exps.weight=CUDA2" \
-ot "ffn.*=CPU" \
-fa -mg 0 -ub 2048 -mla 1
```

### Name and Version

./llama-cli --version
version: 3779 (c9148ba0)
built with cc (GCC) 14.3.1 20250523 (Red Hat 14.3.1-1) for x86_64-redhat-linux

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
./llama-sweep-bench -m '/models_llm/DeepSeek-V3-0324-UD-Q3_K_XL-merged.gguf' -c 16384 --no-mmap -ngl 999 \
-ot "blk.(0|1|2|3|4|5|6).ffn.=CUDA0" \
-ot "blk.(7|8|9).ffn.=CUDA1" \
-ot "blk.(10|11|12).ffn.=CUDA2" \
-ot "blk.(13|14|15|16).ffn.=CUDA3" \
-ot "blk.(17|18|19).ffn.=CUDA4" \
-ot "blk.(20|21|22).ffn.=CUDA5" \
-ot "blk.(23|24|25|26|27|28|29|30|31).ffn.=CUDA6" \
-ot "blk.32.ffn_(norm|gate_inp|gate_exps|down_exps|up_exps|gate_shexp|down_shexp|up_shexp).weight=CUDA1" \
-ot "blk.33.ffn_(norm|gate_inp|gate_exps|down_exps|up_exps|gate_shexp|down_shexp|up_shexp).weight=CUDA2" \
-ot "blk.34.ffn_(norm|gate_inp|gate_exps|down_exps|up_exps|gate_shexp|down_shexp|up_shexp).weight=CUDA4" \
-ot "blk.35.ffn_(norm|gate_inp|gate_exps|down_exps|up_exps|gate_shexp|down_shexp|up_shexp).weight=CUDA5" \
-ot "ffn.*=CPU" \
-fa -mg 0 -ub 2048 -mla 3 -amb 512
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 7 CUDA devices:
  Device 0: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
  Device 1: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
  Device 2: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
  Device 3: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
  Device 4: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 5: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 6: NVIDIA RTX A6000, compute capability 8.6, VMM: yes
llama_model_loader: loaded meta data with 64 key-value pairs and 1086 tensors from /models_llm/DeepSeek-V3-0324-UD-Q3_K_XL-merged.gguf (version GGUF V3 (latest))
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
llama_model_loader: - kv  56:                          general.file_type u32              = 12
llama_model_loader: - kv  57:                      quantize.imatrix.file str              = DeepSeek-V3-0324-GGUF/imatrix_unsloth...
llama_model_loader: - kv  58:                   quantize.imatrix.dataset str              = unsloth_calibration_DeepSeek-V3-0324.txt
llama_model_loader: - kv  59:             quantize.imatrix.entries_count i32              = 720
llama_model_loader: - kv  60:              quantize.imatrix.chunks_count i32              = 60
llama_model_loader: - kv  61:                                   split.no u16              = 0
llama_model_loader: - kv  62:                        split.tensors.count i32              = 1086
llama_model_loader: - kv  63:                                split.count u16              = 0
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  122 tensors
llama_model_loader: - type q3_K:  173 tensors
llama_model_loader: - type q4_K:  385 tensors
llama_model_loader: - type q5_K:   29 tensors
llama_model_loader: - type q6_K:   16 tensors
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
llm_load_print_meta: model ftype      = Q3_K - Medium
llm_load_print_meta: model params     = 671.026 B
llm_load_print_meta: model size       = 275.910 GiB (3.532 BPW) 
llm_load_print_meta: repeating layers = 274.717 GiB (3.526 BPW, 669.173 B parameters)
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
llm_load_tensors: ggml ctx size =    3.57 MiB
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
Tensor blk.7.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.7.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.7.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.7.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.7.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.7.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.7.ffn_up_shexp.weight buffer type overriden to CUDA1
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
Tensor blk.10.ffn_norm.weight buffer type overriden to CUDA2
Tensor blk.10.ffn_gate_inp.weight buffer type overriden to CUDA2
Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CUDA2
Tensor blk.10.ffn_down_exps.weight buffer type overriden to CUDA2
Tensor blk.10.ffn_up_exps.weight buffer type overriden to CUDA2
Tensor blk.10.ffn_gate_shexp.weight buffer type overriden to CUDA2
Tensor blk.10.ffn_down_shexp.weight buffer type overriden to CUDA2
Tensor blk.10.ffn_up_shexp.weight buffer type overriden to CUDA2
Tensor blk.11.ffn_norm.weight buffer type overriden to CUDA2
Tensor blk.11.ffn_gate_inp.weight buffer type overriden to CUDA2
Tensor blk.11.ffn_gate_exps.weight buffer type overriden to CUDA2
Tensor blk.11.ffn_down_exps.weight buffer type overriden to CUDA2
Tensor blk.11.ffn_up_exps.weight buffer type overriden to CUDA2
Tensor blk.11.ffn_gate_shexp.weight buffer type overriden to CUDA2
Tensor blk.11.ffn_down_shexp.weight buffer type overriden to CUDA2
Tensor blk.11.ffn_up_shexp.weight buffer type overriden to CUDA2
Tensor blk.12.ffn_norm.weight buffer type overriden to CUDA2
Tensor blk.12.ffn_gate_inp.weight buffer type overriden to CUDA2
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CUDA2
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CUDA2
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CUDA2
Tensor blk.12.ffn_gate_shexp.weight buffer type overriden to CUDA2
Tensor blk.12.ffn_down_shexp.weight buffer type overriden to CUDA2
Tensor blk.12.ffn_up_shexp.weight buffer type overriden to CUDA2
Tensor blk.13.ffn_norm.weight buffer type overriden to CUDA3
Tensor blk.13.ffn_gate_inp.weight buffer type overriden to CUDA3
Tensor blk.13.ffn_gate_exps.weight buffer type overriden to CUDA3
Tensor blk.13.ffn_down_exps.weight buffer type overriden to CUDA3
Tensor blk.13.ffn_up_exps.weight buffer type overriden to CUDA3
Tensor blk.13.ffn_gate_shexp.weight buffer type overriden to CUDA3
Tensor blk.13.ffn_down_shexp.weight buffer type overriden to CUDA3
Tensor blk.13.ffn_up_shexp.weight buffer type overriden to CUDA3
Tensor blk.14.ffn_norm.weight buffer type overriden to CUDA3
Tensor blk.14.ffn_gate_inp.weight buffer type overriden to CUDA3
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CUDA3
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CUDA3
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CUDA3
Tensor blk.14.ffn_gate_shexp.weight buffer type overriden to CUDA3
Tensor blk.14.ffn_down_shexp.weight buffer type overriden to CUDA3
Tensor blk.14.ffn_up_shexp.weight buffer type overriden to CUDA3
Tensor blk.15.ffn_norm.weight buffer type overriden to CUDA3
Tensor blk.15.ffn_gate_inp.weight buffer type overriden to CUDA3
Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CUDA3
Tensor blk.15.ffn_down_exps.weight buffer type overriden to CUDA3
Tensor blk.15.ffn_up_exps.weight buffer type overriden to CUDA3
Tensor blk.15.ffn_gate_shexp.weight buffer type overriden to CUDA3
Tensor blk.15.ffn_down_shexp.weight buffer type overriden to CUDA3
Tensor blk.15.ffn_up_shexp.weight buffer type overriden to CUDA3
Tensor blk.16.ffn_norm.weight buffer type overriden to CUDA3
Tensor blk.16.ffn_gate_inp.weight buffer type overriden to CUDA3
Tensor blk.16.ffn_gate_exps.weight buffer type overriden to CUDA3
Tensor blk.16.ffn_down_exps.weight buffer type overriden to CUDA3
Tensor blk.16.ffn_up_exps.weight buffer type overriden to CUDA3
Tensor blk.16.ffn_gate_shexp.weight buffer type overriden to CUDA3
Tensor blk.16.ffn_down_shexp.weight buffer type overriden to CUDA3
Tensor blk.16.ffn_up_shexp.weight buffer type overriden to CUDA3
Tensor blk.17.ffn_norm.weight buffer type overriden to CUDA4
Tensor blk.17.ffn_gate_inp.weight buffer type overriden to CUDA4
Tensor blk.17.ffn_gate_exps.weight buffer type overriden to CUDA4
Tensor blk.17.ffn_down_exps.weight buffer type overriden to CUDA4
Tensor blk.17.ffn_up_exps.weight buffer type overriden to CUDA4
Tensor blk.17.ffn_gate_shexp.weight buffer type overriden to CUDA4
Tensor blk.17.ffn_down_shexp.weight buffer type overriden to CUDA4
Tensor blk.17.ffn_up_shexp.weight buffer type overriden to CUDA4
Tensor blk.18.ffn_norm.weight buffer type overriden to CUDA4
Tensor blk.18.ffn_gate_inp.weight buffer type overriden to CUDA4
Tensor blk.18.ffn_gate_exps.weight buffer type overriden to CUDA4
Tensor blk.18.ffn_down_exps.weight buffer type overriden to CUDA4
Tensor blk.18.ffn_up_exps.weight buffer type overriden to CUDA4
Tensor blk.18.ffn_gate_shexp.weight buffer type overriden to CUDA4
Tensor blk.18.ffn_down_shexp.weight buffer type overriden to CUDA4
Tensor blk.18.ffn_up_shexp.weight buffer type overriden to CUDA4
Tensor blk.19.ffn_norm.weight buffer type overriden to CUDA4
Tensor blk.19.ffn_gate_inp.weight buffer type overriden to CUDA4
Tensor blk.19.ffn_gate_exps.weight buffer type overriden to CUDA4
Tensor blk.19.ffn_down_exps.weight buffer type overriden to CUDA4
Tensor blk.19.ffn_up_exps.weight buffer type overriden to CUDA4
Tensor blk.19.ffn_gate_shexp.weight buffer type overriden to CUDA4
Tensor blk.19.ffn_down_shexp.weight buffer type overriden to CUDA4
Tensor blk.19.ffn_up_shexp.weight buffer type overriden to CUDA4
Tensor blk.20.ffn_norm.weight buffer type overriden to CUDA5
Tensor blk.20.ffn_gate_inp.weight buffer type overriden to CUDA5
Tensor blk.20.ffn_gate_exps.weight buffer type overriden to CUDA5
Tensor blk.20.ffn_down_exps.weight buffer type overriden to CUDA5
Tensor blk.20.ffn_up_exps.weight buffer type overriden to CUDA5
Tensor blk.20.ffn_gate_shexp.weight buffer type overriden to CUDA5
Tensor blk.20.ffn_down_shexp.weight buffer type overriden to CUDA5
Tensor blk.20.ffn_up_shexp.weight buffer type overriden to CUDA5
Tensor blk.21.ffn_norm.weight buffer type overriden to CUDA5
Tensor blk.21.ffn_gate_inp.weight buffer type overriden to CUDA5
Tensor blk.21.ffn_gate_exps.weight buffer type overriden to CUDA5
Tensor blk.21.ffn_down_exps.weight buffer type overriden to CUDA5
Tensor blk.21.ffn_up_exps.weight buffer type overriden to CUDA5
Tensor blk.21.ffn_gate_shexp.weight buffer type overriden to CUDA5
Tensor blk.21.ffn_down_shexp.weight buffer type overriden to CUDA5
Tensor blk.21.ffn_up_shexp.weight buffer type overriden to CUDA5
Tensor blk.22.ffn_norm.weight buffer type overriden to CUDA5
Tensor blk.22.ffn_gate_inp.weight buffer type overriden to CUDA5
Tensor blk.22.ffn_gate_exps.weight buffer type overriden to CUDA5
Tensor blk.22.ffn_down_exps.weight buffer type overriden to CUDA5
Tensor blk.22.ffn_up_exps.weight buffer type overriden to CUDA5
Tensor blk.22.ffn_gate_shexp.weight buffer type overriden to CUDA5
Tensor blk.22.ffn_down_shexp.weight buffer type overriden to CUDA5
Tensor blk.22.ffn_up_shexp.weight buffer type overriden to CUDA5
Tensor blk.23.ffn_norm.weight buffer type overriden to CUDA6
Tensor blk.23.ffn_gate_inp.weight buffer type overriden to CUDA6
Tensor blk.23.ffn_gate_exps.weight buffer type overriden to CUDA6
Tensor blk.23.ffn_down_exps.weight buffer type overriden to CUDA6
Tensor blk.23.ffn_up_exps.weight buffer type overriden to CUDA6
Tensor blk.23.ffn_gate_shexp.weight buffer type overriden to CUDA6
Tensor blk.23.ffn_down_shexp.weight buffer type overriden to CUDA6
Tensor blk.23.ffn_up_shexp.weight buffer type overriden to CUDA6
Tensor blk.24.ffn_norm.weight buffer type overriden to CUDA6
Tensor blk.24.ffn_gate_inp.weight buffer type overriden to CUDA6
Tensor blk.24.ffn_gate_exps.weight buffer type overriden to CUDA6
Tensor blk.24.ffn_down_exps.weight buffer type overriden to CUDA6
Tensor blk.24.ffn_up_exps.weight buffer type overriden to CUDA6
Tensor blk.24.ffn_gate_shexp.weight buffer type overriden to CUDA6
Tensor blk.24.ffn_down_shexp.weight buffer type overriden to CUDA6
Tensor blk.24.ffn_up_shexp.weight buffer type overriden to CUDA6
Tensor blk.25.ffn_norm.weight buffer type overriden to CUDA6
Tensor blk.25.ffn_gate_inp.weight buffer type overriden to CUDA6
Tensor blk.25.ffn_gate_exps.weight buffer type overriden to CUDA6
Tensor blk.25.ffn_down_exps.weight buffer type overriden to CUDA6
Tensor blk.25.ffn_up_exps.weight buffer type overriden to CUDA6
Tensor blk.25.ffn_gate_shexp.weight buffer type overriden to CUDA6
Tensor blk.25.ffn_down_shexp.weight buffer type overriden to CUDA6
Tensor blk.25.ffn_up_shexp.weight buffer type overriden to CUDA6
Tensor blk.26.ffn_norm.weight buffer type overriden to CUDA6
Tensor blk.26.ffn_gate_inp.weight buffer type overriden to CUDA6
Tensor blk.26.ffn_gate_exps.weight buffer type overriden to CUDA6
Tensor blk.26.ffn_down_exps.weight buffer type overriden to CUDA6
Tensor blk.26.ffn_up_exps.weight buffer type overriden to CUDA6
Tensor blk.26.ffn_gate_shexp.weight buffer type overriden to CUDA6
Tensor blk.26.ffn_down_shexp.weight buffer type overriden to CUDA6
Tensor blk.26.ffn_up_shexp.weight buffer type overriden to CUDA6
Tensor blk.27.ffn_norm.weight buffer type overriden to CUDA6
Tensor blk.27.ffn_gate_inp.weight buffer type overriden to CUDA6
Tensor blk.27.ffn_gate_exps.weight buffer type overriden to CUDA6
Tensor blk.27.ffn_down_exps.weight buffer type overriden to CUDA6
Tensor blk.27.ffn_up_exps.weight buffer type overriden to CUDA6
Tensor blk.27.ffn_gate_shexp.weight buffer type overriden to CUDA6
Tensor blk.27.ffn_down_shexp.weight buffer type overriden to CUDA6
Tensor blk.27.ffn_up_shexp.weight buffer type overriden to CUDA6
Tensor blk.28.ffn_norm.weight buffer type overriden to CUDA6
Tensor blk.28.ffn_gate_inp.weight buffer type overriden to CUDA6
Tensor blk.28.ffn_gate_exps.weight buffer type overriden to CUDA6
Tensor blk.28.ffn_down_exps.weight buffer type overriden to CUDA6
Tensor blk.28.ffn_up_exps.weight buffer type overriden to CUDA6
Tensor blk.28.ffn_gate_shexp.weight buffer type overriden to CUDA6
Tensor blk.28.ffn_down_shexp.weight buffer type overriden to CUDA6
Tensor blk.28.ffn_up_shexp.weight buffer type overriden to CUDA6
Tensor blk.29.ffn_norm.weight buffer type overriden to CUDA6
Tensor blk.29.ffn_gate_inp.weight buffer type overriden to CUDA6
Tensor blk.29.ffn_gate_exps.weight buffer type overriden to CUDA6
Tensor blk.29.ffn_down_exps.weight buffer type overriden to CUDA6
Tensor blk.29.ffn_up_exps.weight buffer type overriden to CUDA6
Tensor blk.29.ffn_gate_shexp.weight buffer type overriden to CUDA6
Tensor blk.29.ffn_down_shexp.weight buffer type overriden to CUDA6
Tensor blk.29.ffn_up_shexp.weight buffer type overriden to CUDA6
Tensor blk.30.ffn_norm.weight buffer type overriden to CUDA6
Tensor blk.30.ffn_gate_inp.weight buffer type overriden to CUDA6
Tensor blk.30.ffn_gate_exps.weight buffer type overriden to CUDA6
Tensor blk.30.ffn_down_exps.weight buffer type overriden to CUDA6
Tensor blk.30.ffn_up_exps.weight buffer type overriden to CUDA6
Tensor blk.30.ffn_gate_shexp.weight buffer type overriden to CUDA6
Tensor blk.30.ffn_down_shexp.weight buffer type overriden to CUDA6
Tensor blk.30.ffn_up_shexp.weight buffer type overriden to CUDA6
Tensor blk.31.ffn_norm.weight buffer type overriden to CUDA6
Tensor blk.31.ffn_gate_inp.weight buffer type overriden to CUDA6
Tensor blk.31.ffn_gate_exps.weight buffer type overriden to CUDA6
Tensor blk.31.ffn_down_exps.weight buffer type overriden to CUDA6
Tensor blk.31.ffn_up_exps.weight buffer type overriden to CUDA6
Tensor blk.31.ffn_gate_shexp.weight buffer type overriden to CUDA6
Tensor blk.31.ffn_down_shexp.weight buffer type overriden to CUDA6
Tensor blk.31.ffn_up_shexp.weight buffer type overriden to CUDA6
Tensor blk.32.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.32.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.32.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.32.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.32.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.32.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.32.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.32.ffn_up_shexp.weight buffer type overriden to CUDA1
Tensor blk.33.ffn_norm.weight buffer type overriden to CUDA2
Tensor blk.33.ffn_gate_inp.weight buffer type overriden to CUDA2
Tensor blk.33.ffn_gate_exps.weight buffer type overriden to CUDA2
Tensor blk.33.ffn_down_exps.weight buffer type overriden to CUDA2
Tensor blk.33.ffn_up_exps.weight buffer type overriden to CUDA2
Tensor blk.33.ffn_gate_shexp.weight buffer type overriden to CUDA2
Tensor blk.33.ffn_down_shexp.weight buffer type overriden to CUDA2
Tensor blk.33.ffn_up_shexp.weight buffer type overriden to CUDA2
Tensor blk.34.ffn_norm.weight buffer type overriden to CUDA4
Tensor blk.34.ffn_gate_inp.weight buffer type overriden to CUDA4
Tensor blk.34.ffn_gate_exps.weight buffer type overriden to CUDA4
Tensor blk.34.ffn_down_exps.weight buffer type overriden to CUDA4
Tensor blk.34.ffn_up_exps.weight buffer type overriden to CUDA4
Tensor blk.34.ffn_gate_shexp.weight buffer type overriden to CUDA4
Tensor blk.34.ffn_down_shexp.weight buffer type overriden to CUDA4
Tensor blk.34.ffn_up_shexp.weight buffer type overriden to CUDA4
Tensor blk.35.ffn_norm.weight buffer type overriden to CUDA5
Tensor blk.35.ffn_gate_inp.weight buffer type overriden to CUDA5
Tensor blk.35.ffn_gate_exps.weight buffer type overriden to CUDA5
Tensor blk.35.ffn_down_exps.weight buffer type overriden to CUDA5
Tensor blk.35.ffn_up_exps.weight buffer type overriden to CUDA5
Tensor blk.35.ffn_gate_shexp.weight buffer type overriden to CUDA5
Tensor blk.35.ffn_down_shexp.weight buffer type overriden to CUDA5
Tensor blk.35.ffn_up_shexp.weight buffer type overriden to CUDA5
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
llm_load_tensors:        CPU buffer size = 118694.87 MiB
llm_load_tensors:  CUDA_Host buffer size =   497.11 MiB
llm_load_tensors:      CUDA0 buffer size = 20712.62 MiB
llm_load_tensors:      CUDA1 buffer size = 19841.07 MiB
llm_load_tensors:      CUDA2 buffer size = 20320.68 MiB
llm_load_tensors:      CUDA3 buffer size = 19580.03 MiB
llm_load_tensors:      CUDA4 buffer size = 19490.18 MiB
llm_load_tensors:      CUDA5 buffer size = 19364.96 MiB
llm_load_tensors:      CUDA6 buffer size = 44030.76 MiB
....
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  2048 |    512 |      0 |    9.166 |   223.43 |   56.876 |     9.00 |
|  2048 |    512 |   2048 |    9.549 |   214.48 |   57.088 |     8.97 |
|  2048 |    512 |   4096 |   10.041 |   203.96 |   57.929 |     8.84 |
|  2048 |    512 |   6144 |   10.534 |   194.42 |   58.584 |     8.74 |
Oops(ggml_compute_forward_sum_rows_f32, ffn_moe_weights_sum-60): found nan for i1 = 0, i2 = 0, i3 = 0. ne00 = 8
```

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-07-03** at **13:09:02**:<br>

So, nobody else has reported an issue such as this. But you are leaving the shared experts on the CPU. This is your intent?

---

👤 **Panchovix** commented the **2025-07-03** at **14:05:13**:<br>

Hi there, yes this is like a new issue that I have noticed just recently but not sure since when. You mean the shexps? Basically I leave an entire layer when I can on a GPU, or 1 layer on 2 GPUs if it's too big when increasing ubatch size.

---

👤 **ikawrakow** commented the **2025-07-03** at **14:40:44**:<br>

Can you try if you can reproduce on 8e5106b20f694c84811b073b3a4f86ca9d871441 ?

Thanks.

---

👤 **Panchovix** commented the **2025-07-03** at **16:05:29**:<br>

Was testing on that commit but got it again sadly

```
main: n_kv_max = 16384, n_batch = 2048, n_ubatch = 2048, flash_attn = 1, n_gpu_layers = 999, n_threads = 8, n_threads_batch = 8

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  2048 |    512 |      0 |   10.727 |   190.91 |   78.384 |     6.53 |
|  2048 |    512 |   2048 |   10.969 |   186.71 |   71.271 |     7.18 |
|  2048 |    512 |   4096 |   11.553 |   177.27 |   70.445 |     7.27 |
|  2048 |    512 |   6144 |   12.099 |   169.27 |   71.958 |     7.12 |
|  2048 |    512 |   8192 |   12.719 |   161.01 |   72.710 |     7.04 |
|  2048 |    512 |  10240 |   13.011 |   157.40 |   73.517 |     6.96 |
Oops(ggml_compute_forward_sum_rows_f32, ffn_moe_weights_sum-60): found nan for i1 = 0, i2 = 0, i3 = 0. ne00 = 8
```

```
./llama-cli --version
version: 3771 (8e5106b2)
built with cc (GCC) 14.3.1 20250523 (Red Hat 14.3.1-1) for x86_64-redhat-linux
```

EDIT: Just wondering, would for example a unstable RAM or CPU cause this? I have been using my RAM at 6000Mhz for about a year without issues, but maybe is not stable for this?

---

👤 **Panchovix** commented the **2025-07-05** at **16:58:48**:<br>

Okay for now I have reduced my VRAM overclocks on some 4090s I was using and it seems I haven't seen the error again. So I guess it was related to that. Closing!