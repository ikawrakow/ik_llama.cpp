### 🐛 [#450](https://github.com/ikawrakow/ik_llama.cpp/issues/450) - Bug: Performance regression

| **Author** | `cmoncure` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-23 |
| **Updated** | 2025-05-30 |

---

#### Description

### What happened?

After this PR: Refactor iqk_mul_mat.cpp (#435)

This commit results in a significant performance regression for me, established by git bisect.
My TG drops by about 30% on DeepSeek. (12.5 t/s => 9.5 t/s)

https://github.com/ikawrakow/ik_llama.cpp/commit/b94cd3b632a78dfb46b18d52b84be66bcf26166a is the first bad commit
commit https://github.com/ikawrakow/ik_llama.cpp/commit/b94cd3b632a78dfb46b18d52b84be66bcf26166a (HEAD)
Author: Kawrakow [iwankawrakow@gmail.com](mailto:iwankawrakow@gmail.com)
Date: Thu May 22 10:05:51 2025 +0300

Refactor iqk_mul_mat.cpp (#435)



### Name and Version

$ ./llama-cli --version
version: 3705 (ec456322)
built with cc (Ubuntu 14.2.0-4ubuntu2) 14.2.0 for x86_64-linux-gnu

~/ik_llama.cpp/build/bin/llama-server \
-mla 3 -fa \
-ctk q8_0 \
-ctv q8_0 \
--ctx-size 32768 \
-fmoe \
-amb 512 \
-b 1024 \
-ub 1024 \
-sm none \
--numa isolate \
--threads 16 \
--threads-batch 32 \
--n-gpu-layers 99 \
--override-tensor exps=CPU \
--override-tensor attn=CUDA0 \
--override-tensor exp=CUDA0 \
--override-tensor blk.*.ffn_gate_inp.weight=CUDA0 \
--override-tensor blk.*.ffn_down.weight=CUDA0 \
--override-tensor blk.*.ffn_gate.weight=CUDA0 \
--override-tensor blk.*.ffn_norm.weight=CUDA0 \
--override-tensor blk.*.ffn_up_shexp.weight=CUDA0 \
--override-tensor blk.*.ffn_down_shexp.weight=CUDA0 \
--override-tensor blk.*.ffn_gate_shexp.weight=CUDA0 \
--override-tensor blk.*.ffn_gate_inp.weight=CUDA0 \
--host 0.0.0.0 \
--port 7862 \
--alias DeepSeek/DeepSeek-V3-0324-IQ4_K_R4 \
-m ~/AIModels/textgen/DeepSeek-V3-0324-IQ4_K_R4.gguf

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell

```

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-05-23** at **12:49:28**:<br>

What is the CPU being used and how was the performance regression determined?
Log output (including when the server starts) could help.

---

👤 **cmoncure** commented the **2025-05-23** at **13:53:03**:<br>

CPU is EPYC 9175F
I used `git bisect` from HEAD~14 and ran the same prompt against each one.  Performance is good on every commit prior to this one.

GOOD log:

$ ./build/bin/llama-cli --version
version: 3703 (a2b5057a)
built with cc (Ubuntu 14.2.0-4ubuntu2) 14.2.0 for x86_64-linux-gnu


```ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA RTX 6000 Ada Generation, compute capability 8.9, VMM: yes
  Device 1: NVIDIA RTX 6000 Ada Generation, compute capability 8.9, VMM: yes
INFO [                    main] build info | tid="136521606795264" timestamp=1748008001 build=3703 commit="a2b5057a"
INFO [                    main] system info | tid="136521606795264" timestamp=1748008001 n_threads=16 n_threads_batch=32 total_threads=32 system_info="AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: loaded meta data with 53 key-value pairs and 1147 tensors from /home/corey/AIModels/textgen/DeepSeek-V3-0324-IQ4_K_R4.gguf (version GGUF V3 (latest))
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
llama_model_loader: - kv  16:                          general.file_type u32              = 340
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
llama_model_loader: - kv  36:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<�...
llama_model_loader: - kv  37:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  38:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
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
llama_model_loader: - kv  50:                                   split.no u16              = 0
llama_model_loader: - kv  51:                                split.count u16              = 0
llama_model_loader: - kv  52:                        split.tensors.count i32              = 1147
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  612 tensors
llama_model_loader: - type iq4_k_r4:  116 tensors
llama_model_loader: - type iq5_k_r4:   58 tensors
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
llm_load_print_meta: model ftype      = IQ4_K_R4 - 4.5 bpw
llm_load_print_meta: model params     = 672.050 B
llm_load_print_meta: model size       = 386.183 GiB (4.936 BPW) 
llm_load_print_meta: repeating layers = 384.349 GiB (4.926 BPW, 670.196 B parameters)
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
llm_load_tensors: ggml ctx size =    0.93 MiB
Tensor blk.0.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.0.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.0.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.0.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.0.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.0.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.0.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.0.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.0.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.0.attn_output.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_gate.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_down.weight buffer type overriden to CUDA0
Tensor blk.1.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.1.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.1.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.1.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.1.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.1.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.1.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.1.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.1.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.1.attn_output.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_gate.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_down.weight buffer type overriden to CUDA0
Tensor blk.2.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.2.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.2.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.2.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.2.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.2.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.2.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.2.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.2.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.2.attn_output.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_gate.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_down.weight buffer type overriden to CUDA0
Tensor blk.3.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.3.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.3.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.3.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.3.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.3.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.3.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.3.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.3.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.3.attn_output.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.3.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.4.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.4.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.4.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.4.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.4.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.4.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.4.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.4.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.4.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.4.attn_output.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.4.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.4.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.5.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.5.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.5.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.5.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.5.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.5.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.5.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.5.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.5.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.5.attn_output.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.5.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.5.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.6.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.6.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.6.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.6.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.6.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.6.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.6.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.6.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.6.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.6.attn_output.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.6.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.6.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.7.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.7.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.7.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.7.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.7.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.7.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.7.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.7.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.7.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.7.attn_output.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.7.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.8.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.8.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.8.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.8.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.8.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.8.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.8.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.8.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.8.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.8.attn_output.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.8.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.9.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.9.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.9.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.9.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.9.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.9.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.9.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.9.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.9.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.9.attn_output.weight buffer type overriden to CUDA0
Tensor blk.9.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.9.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.9.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.9.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.9.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.9.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.10.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.10.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.10.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.10.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.10.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.10.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.10.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.10.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.10.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.10.attn_output.weight buffer type overriden to CUDA0
Tensor blk.10.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.10.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.10.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.10.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.10.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.11.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.11.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.11.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.11.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.11.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.11.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.11.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.11.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.11.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.11.attn_output.weight buffer type overriden to CUDA0
Tensor blk.11.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.11.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.11.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.11.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.11.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.11.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.12.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.12.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.12.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.12.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.12.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.12.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.12.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.12.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.12.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.12.attn_output.weight buffer type overriden to CUDA0
Tensor blk.12.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.12.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.12.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.12.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.12.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.13.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.13.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.13.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.13.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.13.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.13.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.13.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.13.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.13.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.13.attn_output.weight buffer type overriden to CUDA0
Tensor blk.13.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.13.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.13.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.13.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.13.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.13.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.14.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.14.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.14.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.14.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.14.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.14.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.14.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.14.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.14.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.14.attn_output.weight buffer type overriden to CUDA0
Tensor blk.14.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.14.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.14.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.14.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.14.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.15.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.15.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.15.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.15.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.15.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.15.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.15.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.15.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.15.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.15.attn_output.weight buffer type overriden to CUDA0
Tensor blk.15.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.15.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.15.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.15.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.15.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.16.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.16.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.16.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.16.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.16.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.16.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.16.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.16.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.16.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.16.attn_output.weight buffer type overriden to CUDA0
Tensor blk.16.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.16.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.16.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.16.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.16.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.16.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.17.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.17.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.17.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.17.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.17.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.17.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.17.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.17.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.17.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.17.attn_output.weight buffer type overriden to CUDA0
Tensor blk.17.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.17.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.17.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.17.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.17.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.17.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.18.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.18.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.18.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.18.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.18.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.18.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.18.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.18.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.18.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.18.attn_output.weight buffer type overriden to CUDA0
Tensor blk.18.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.18.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.18.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.18.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.18.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.18.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.19.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.19.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.19.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.19.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.19.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.19.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.19.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.19.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.19.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.19.attn_output.weight buffer type overriden to CUDA0
Tensor blk.19.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.19.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.19.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.19.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.19.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.19.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.20.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.20.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.20.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.20.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.20.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.20.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.20.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.20.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.20.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.20.attn_output.weight buffer type overriden to CUDA0
Tensor blk.20.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.20.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.20.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.20.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.20.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.20.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.21.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.21.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.21.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.21.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.21.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.21.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.21.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.21.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.21.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.21.attn_output.weight buffer type overriden to CUDA0
Tensor blk.21.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.21.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.21.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.21.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.21.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.21.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.22.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.22.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.22.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.22.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.22.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.22.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.22.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.22.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.22.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.22.attn_output.weight buffer type overriden to CUDA0
Tensor blk.22.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.22.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.22.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.22.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.22.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.22.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.23.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.23.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.23.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.23.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.23.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.23.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.23.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.23.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.23.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.23.attn_output.weight buffer type overriden to CUDA0
Tensor blk.23.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.23.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.23.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.23.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.23.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.23.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.24.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.24.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.24.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.24.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.24.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.24.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.24.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.24.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.24.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.24.attn_output.weight buffer type overriden to CUDA0
Tensor blk.24.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.24.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.24.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.24.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.24.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.24.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.25.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.25.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.25.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.25.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.25.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.25.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.25.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.25.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.25.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.25.attn_output.weight buffer type overriden to CUDA0
Tensor blk.25.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.25.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.25.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.25.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.25.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.25.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.26.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.26.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.26.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.26.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.26.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.26.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.26.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.26.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.26.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.26.attn_output.weight buffer type overriden to CUDA0
Tensor blk.26.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.26.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.26.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.26.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.26.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.26.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.27.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.27.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.27.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.27.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.27.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.27.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.27.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.27.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.27.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.27.attn_output.weight buffer type overriden to CUDA0
Tensor blk.27.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.27.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.27.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.27.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.27.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.27.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.28.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.28.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.28.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.28.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.28.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.28.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.28.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.28.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.28.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.28.attn_output.weight buffer type overriden to CUDA0
Tensor blk.28.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.28.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.28.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.28.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.28.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.28.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.29.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.29.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.29.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.29.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.29.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.29.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.29.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.29.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.29.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.29.attn_output.weight buffer type overriden to CUDA0
Tensor blk.29.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.29.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.29.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.29.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.29.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.29.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.30.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.30.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.30.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.30.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.30.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.30.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.30.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.30.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.30.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.30.attn_output.weight buffer type overriden to CUDA0
Tensor blk.30.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.30.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.30.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.30.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.30.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.30.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.31.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.31.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.31.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.31.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.31.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.31.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.31.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.31.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.31.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.31.attn_output.weight buffer type overriden to CUDA0
Tensor blk.31.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.31.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.31.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.31.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.31.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.31.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.32.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.32.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.32.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.32.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.32.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.32.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.32.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.32.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.32.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.32.attn_output.weight buffer type overriden to CUDA0
Tensor blk.32.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.32.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.32.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.32.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.32.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.32.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.33.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.33.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.33.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.33.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.33.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.33.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.33.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.33.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.33.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.33.attn_output.weight buffer type overriden to CUDA0
Tensor blk.33.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.33.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.33.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.33.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.33.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.33.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.34.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.34.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.34.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.34.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.34.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.34.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.34.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.34.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.34.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.34.attn_output.weight buffer type overriden to CUDA0
Tensor blk.34.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.34.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.34.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.34.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.34.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.34.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.35.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.35.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.35.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.35.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.35.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.35.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.35.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.35.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.35.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.35.attn_output.weight buffer type overriden to CUDA0
Tensor blk.35.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.35.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.35.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.35.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.35.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.35.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.36.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.36.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.36.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.36.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.36.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.36.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.36.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.36.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.36.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.36.attn_output.weight buffer type overriden to CUDA0
Tensor blk.36.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.36.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.36.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.36.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.36.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.36.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.37.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.37.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.37.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.37.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.37.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.37.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.37.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.37.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.37.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.37.attn_output.weight buffer type overriden to CUDA0
Tensor blk.37.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.37.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.37.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.37.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.37.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.37.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.38.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.38.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.38.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.38.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.38.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.38.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.38.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.38.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.38.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.38.attn_output.weight buffer type overriden to CUDA0
Tensor blk.38.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.38.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.38.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.38.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.38.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.38.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.39.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.39.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.39.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.39.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.39.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.39.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.39.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.39.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.39.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.39.attn_output.weight buffer type overriden to CUDA0
Tensor blk.39.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.39.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.39.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.39.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.39.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.39.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.40.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.40.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.40.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.40.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.40.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.40.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.40.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.40.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.40.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.40.attn_output.weight buffer type overriden to CUDA0
Tensor blk.40.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.40.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.40.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.40.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.40.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.40.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.41.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.41.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.41.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.41.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.41.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.41.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.41.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.41.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.41.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.41.attn_output.weight buffer type overriden to CUDA0
Tensor blk.41.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.41.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.41.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.41.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.41.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.41.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.42.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.42.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.42.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.42.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.42.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.42.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.42.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.42.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.42.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.42.attn_output.weight buffer type overriden to CUDA0
Tensor blk.42.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.42.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.42.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.42.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.42.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.42.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.43.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.43.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.43.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.43.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.43.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.43.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.43.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.43.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.43.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.43.attn_output.weight buffer type overriden to CUDA0
Tensor blk.43.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.43.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.43.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.43.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.43.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.43.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.44.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.44.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.44.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.44.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.44.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.44.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.44.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.44.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.44.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.44.attn_output.weight buffer type overriden to CUDA0
Tensor blk.44.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.44.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.44.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.44.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.44.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.44.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.45.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.45.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.45.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.45.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.45.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.45.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.45.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.45.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.45.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.45.attn_output.weight buffer type overriden to CUDA0
Tensor blk.45.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.45.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.45.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.45.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.45.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.45.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.46.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.46.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.46.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.46.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.46.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.46.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.46.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.46.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.46.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.46.attn_output.weight buffer type overriden to CUDA0
Tensor blk.46.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.46.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.46.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.46.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.46.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.46.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.47.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.47.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.47.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.47.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.47.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.47.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.47.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.47.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.47.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.47.attn_output.weight buffer type overriden to CUDA0
Tensor blk.47.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.47.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.47.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.47.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.47.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.47.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.48.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.48.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.48.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.48.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.48.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.48.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.48.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.48.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.48.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.48.attn_output.weight buffer type overriden to CUDA0
Tensor blk.48.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.48.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.48.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.48.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.48.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.48.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.49.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.49.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.49.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.49.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.49.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.49.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.49.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.49.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.49.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.49.attn_output.weight buffer type overriden to CUDA0
Tensor blk.49.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.49.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.49.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.49.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.49.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.49.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.50.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.50.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.50.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.50.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.50.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.50.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.50.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.50.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.50.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.50.attn_output.weight buffer type overriden to CUDA0
Tensor blk.50.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.50.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.50.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.50.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.50.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.50.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.51.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.51.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.51.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.51.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.51.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.51.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.51.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.51.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.51.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.51.attn_output.weight buffer type overriden to CUDA0
Tensor blk.51.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.51.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.51.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.51.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.51.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.51.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.52.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.52.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.52.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.52.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.52.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.52.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.52.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.52.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.52.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.52.attn_output.weight buffer type overriden to CUDA0
Tensor blk.52.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.52.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.52.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.52.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.52.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.52.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.53.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.53.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.53.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.53.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.53.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.53.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.53.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.53.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.53.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.53.attn_output.weight buffer type overriden to CUDA0
Tensor blk.53.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.53.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.53.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.53.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.53.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.53.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.54.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.54.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.54.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.54.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.54.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.54.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.54.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.54.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.54.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.54.attn_output.weight buffer type overriden to CUDA0
Tensor blk.54.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.54.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.54.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.54.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.54.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.54.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.55.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.55.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.55.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.55.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.55.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.55.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.55.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.55.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.55.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.55.attn_output.weight buffer type overriden to CUDA0
Tensor blk.55.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.55.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.55.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.55.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.55.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.55.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.56.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.56.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.56.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.56.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.56.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.56.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.56.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.56.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.56.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.56.attn_output.weight buffer type overriden to CUDA0
Tensor blk.56.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.56.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.56.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.56.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.56.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.56.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.57.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.57.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.57.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.57.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.57.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.57.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.57.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.57.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.57.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.57.attn_output.weight buffer type overriden to CUDA0
Tensor blk.57.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.57.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.57.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.57.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.57.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.57.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.58.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.58.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.58.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.58.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.58.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.58.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.58.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.58.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.58.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.58.attn_output.weight buffer type overriden to CUDA0
Tensor blk.58.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.58.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.58.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.58.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.58.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.58.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.59.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.59.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.59.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.59.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.59.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.59.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.59.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.59.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.59.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.59.attn_output.weight buffer type overriden to CUDA0
Tensor blk.59.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.59.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.59.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.59.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.59.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.59.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.60.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.60.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.60.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.60.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.60.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.60.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.60.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.60.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.60.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.60.attn_output.weight buffer type overriden to CUDA0
Tensor blk.60.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.60.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.60.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.60.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.60.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.60.ffn_up_shexp.weight buffer type overriden to CUDA0
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size = 392428.85 MiB
llm_load_tensors:        CPU buffer size =   938.98 MiB
llm_load_tensors:      CUDA0 buffer size = 17744.02 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 1024
llama_new_context_with_model: n_ubatch   = 1024
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
llama_new_context_with_model:      CUDA0 compute buffer size =  3650.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   352.01 MiB
llama_new_context_with_model: graph nodes  = 8245
llama_new_context_with_model: graph splits = 118
INFO [                    init] initializing slots | tid="136521606795264" timestamp=1748008022 n_slots=1
INFO [                    init] new slot | tid="136521606795264" timestamp=1748008022 id_slot=0 n_ctx_slot=32768
INFO [                    main] model loaded | tid="136521606795264" timestamp=1748008022
INFO [                    main] chat template | tid="136521606795264" timestamp=1748008022 chat_example="You are a helpful assistant\n\n<｜User｜>Hello<｜Assistant｜>Hi there<｜end▁of▁sentence｜><｜User｜>How are you?<｜Assistant｜>" built_in=true
INFO [                    main] HTTP server listening | tid="136521606795264" timestamp=1748008022 n_threads_http="31" port="7862" hostname="0.0.0.0"
INFO [            update_slots] all slots are idle | tid="136521606795264" timestamp=1748008022
INFO [   launch_slot_with_task] slot is processing task | tid="136521606795264" timestamp=1748008040 id_slot=0 id_task=0
INFO [            update_slots] kv cache rm [p0, end) | tid="136521606795264" timestamp=1748008040 id_slot=0 id_task=0 p0=0
INFO [            update_slots] kv cache rm [p0, end) | tid="136521606795264" timestamp=1748008051 id_slot=0 id_task=0 p0=1024
INFO [            update_slots] kv cache rm [p0, end) | tid="136521606795264" timestamp=1748008063 id_slot=0 id_task=0 p0=2048
INFO [           print_timings] prompt eval time     =   25767.00 ms /  2190 tokens (   11.77 ms per token,    84.99 tokens per second) | tid="136521606795264" timestamp=1748008081 id_slot=0 id_task=0 t_prompt_processing=25767.002 n_prompt_tokens_processed=2190 t_token=11.765754337899544 n_tokens_second=84.9924255836981
INFO [           print_timings] generation eval time =   15701.68 ms /   222 runs   (   70.73 ms per token,    14.14 tokens per second) | tid="136521606795264" timestamp=1748008081 id_slot=0 id_task=0 t_token_generation=15701.681 n_decoded=222 t_token=70.7282927927928 n_tokens_second=14.138613566279941
INFO [           print_timings]           total time =   41468.68 ms | tid="136521606795264" timestamp=1748008081 id_slot=0 id_task=0 t_prompt_processing=25767.002 t_token_generation=15701.681 t_total=41468.683000000005
INFO [            update_slots] slot released | tid="136521606795264" timestamp=1748008081 id_slot=0 id_task=0 n_ctx=32768 n_past=2411 n_system_tokens=0 n_cache_tokens=2411 truncated=false
INFO [            update_slots] all slots are idle | tid="136521606795264" timestamp=1748008081
INFO [      log_server_request] request | tid="136105332502528" timestamp=1748008081 remote_addr="10.254.1.2" remote_port=51316 status=200 method="POST" path="/completion" params={}
INFO [            update_slots] all slots are idle | tid="136521606795264" timestamp=1748008081
```

BAD log:

$ ./build-bad/bin/llama-cli --version
version: 3705 (ec456322)
built with cc (Ubuntu 14.2.0-4ubuntu2) 14.2.0 for x86_64-linux-gnu

(by way of `diff`)
```
$ diff goodlog badlog
5,6c5,6
< INFO [                    main] build info | tid="136521606795264" timestamp=1748008001 build=3703 commit="a2b5057a"
< INFO [                    main] system info | tid="136521606795264" timestamp=1748008001 n_threads=16 n_threads_batch=32 total_threads=32 system_info="AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
---
> INFO [                    main] build info | tid="127511205212160" timestamp=1748008231 build=3705 commit="ec456322"
> INFO [                    main] system info | tid="127511205212160" timestamp=1748008231 n_threads=16 n_threads_batch=32 total_threads=32 system_info="AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
1293,1309c1293,1309
< INFO [                    init] initializing slots | tid="136521606795264" timestamp=1748008022 n_slots=1
< INFO [                    init] new slot | tid="136521606795264" timestamp=1748008022 id_slot=0 n_ctx_slot=32768
< INFO [                    main] model loaded | tid="136521606795264" timestamp=1748008022
< INFO [                    main] chat template | tid="136521606795264" timestamp=1748008022 chat_example="You are a helpful assistant\n\n<｜User｜>Hello<｜Assistant｜>Hi there<｜end▁of▁sentence｜><｜User｜>How are you?<｜Assistant｜>" built_in=true
< INFO [                    main] HTTP server listening | tid="136521606795264" timestamp=1748008022 n_threads_http="31" port="7862" hostname="0.0.0.0"
< INFO [            update_slots] all slots are idle | tid="136521606795264" timestamp=1748008022
< INFO [   launch_slot_with_task] slot is processing task | tid="136521606795264" timestamp=1748008040 id_slot=0 id_task=0
< INFO [            update_slots] kv cache rm [p0, end) | tid="136521606795264" timestamp=1748008040 id_slot=0 id_task=0 p0=0
< INFO [            update_slots] kv cache rm [p0, end) | tid="136521606795264" timestamp=1748008051 id_slot=0 id_task=0 p0=1024
< INFO [            update_slots] kv cache rm [p0, end) | tid="136521606795264" timestamp=1748008063 id_slot=0 id_task=0 p0=2048
< INFO [           print_timings] prompt eval time     =   25767.00 ms /  2190 tokens (   11.77 ms per token,    84.99 tokens per second) | tid="136521606795264" timestamp=1748008081 id_slot=0 id_task=0 t_prompt_processing=25767.002 n_prompt_tokens_processed=2190 t_token=11.765754337899544 n_tokens_second=84.9924255836981
< INFO [           print_timings] generation eval time =   15701.68 ms /   222 runs   (   70.73 ms per token,    14.14 tokens per second) | tid="136521606795264" timestamp=1748008081 id_slot=0 id_task=0 t_token_generation=15701.681 n_decoded=222 t_token=70.7282927927928 n_tokens_second=14.138613566279941
< INFO [           print_timings]           total time =   41468.68 ms | tid="136521606795264" timestamp=1748008081 id_slot=0 id_task=0 t_prompt_processing=25767.002 t_token_generation=15701.681 t_total=41468.683000000005
< INFO [            update_slots] slot released | tid="136521606795264" timestamp=1748008081 id_slot=0 id_task=0 n_ctx=32768 n_past=2411 n_system_tokens=0 n_cache_tokens=2411 truncated=false
< INFO [            update_slots] all slots are idle | tid="136521606795264" timestamp=1748008081
< INFO [      log_server_request] request | tid="136105332502528" timestamp=1748008081 remote_addr="10.254.1.2" remote_port=51316 status=200 method="POST" path="/completion" params={}
< INFO [            update_slots] all slots are idle | tid="136521606795264" timestamp=1748008081
---
> INFO [                    init] initializing slots | tid="127511205212160" timestamp=1748008241 n_slots=1
> INFO [                    init] new slot | tid="127511205212160" timestamp=1748008241 id_slot=0 n_ctx_slot=32768
> INFO [                    main] model loaded | tid="127511205212160" timestamp=1748008241
> INFO [                    main] chat template | tid="127511205212160" timestamp=1748008241 chat_example="You are a helpful assistant\n\n<｜User｜>Hello<｜Assistant｜>Hi there<｜end▁of▁sentence｜><｜User｜>How are you?<｜Assistant｜>" built_in=true
> INFO [                    main] HTTP server listening | tid="127511205212160" timestamp=1748008241 n_threads_http="31" port="7862" hostname="0.0.0.0"
> INFO [            update_slots] all slots are idle | tid="127511205212160" timestamp=1748008241
> INFO [   launch_slot_with_task] slot is processing task | tid="127511205212160" timestamp=1748008291 id_slot=0 id_task=0
> INFO [            update_slots] kv cache rm [p0, end) | tid="127511205212160" timestamp=1748008291 id_slot=0 id_task=0 p0=0
> INFO [            update_slots] kv cache rm [p0, end) | tid="127511205212160" timestamp=1748008303 id_slot=0 id_task=0 p0=1024
> INFO [            update_slots] kv cache rm [p0, end) | tid="127511205212160" timestamp=1748008315 id_slot=0 id_task=0 p0=2048
> INFO [           print_timings] prompt eval time     =   25845.83 ms /  2190 tokens (   11.80 ms per token,    84.73 tokens per second) | tid="127511205212160" timestamp=1748008339 id_slot=0 id_task=0 t_prompt_processing=25845.833 n_prompt_tokens_processed=2190 t_token=11.801750228310501 n_tokens_second=84.73319470879504
> INFO [           print_timings] generation eval time =   21665.24 ms /   222 runs   (   97.59 ms per token,    10.25 tokens per second) | tid="127511205212160" timestamp=1748008339 id_slot=0 id_task=0 t_token_generation=21665.244 n_decoded=222 t_token=97.59118918918918 n_tokens_second=10.246826668557253
> INFO [           print_timings]           total time =   47511.08 ms | tid="127511205212160" timestamp=1748008339 id_slot=0 id_task=0 t_prompt_processing=25845.833 t_token_generation=21665.244 t_total=47511.077
> INFO [            update_slots] slot released | tid="127511205212160" timestamp=1748008339 id_slot=0 id_task=0 n_ctx=32768 n_past=2411 n_system_tokens=0 n_cache_tokens=2411 truncated=false
> INFO [            update_slots] all slots are idle | tid="127511205212160" timestamp=1748008339
> INFO [      log_server_request] request | tid="127095162204160" timestamp=1748008339 remote_addr="10.254.1.2" remote_port=43794 status=200 method="POST" path="/completion" params={}
> INFO [            update_slots] all slots are idle | tid="127511205212160" timestamp=1748008339
```

---

👤 **cmoncure** commented the **2025-05-23** at **13:53:03**:<br>

CPU is EPYC 9175F
I used `git bisect` from HEAD~14 and ran the same prompt against each one.  Performance is good on every commit prior to this one.

GOOD log:

$ ./build/bin/llama-cli --version
version: 3703 (a2b5057a)
built with cc (Ubuntu 14.2.0-4ubuntu2) 14.2.0 for x86_64-linux-gnu


`ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA RTX 6000 Ada Generation, compute capability 8.9, VMM: yes
  Device 1: NVIDIA RTX 6000 Ada Generation, compute capability 8.9, VMM: yes
INFO [                    main] build info | tid="136521606795264" timestamp=1748008001 build=3703 commit="a2b5057a"
INFO [                    main] system info | tid="136521606795264" timestamp=1748008001 n_threads=16 n_threads_batch=32 total_threads=32 system_info="AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: loaded meta data with 53 key-value pairs and 1147 tensors from /home/corey/AIModels/textgen/DeepSeek-V3-0324-IQ4_K_R4.gguf (version GGUF V3 (latest))
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
llama_model_loader: - kv  16:                          general.file_type u32              = 340
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
llama_model_loader: - kv  36:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<�...
llama_model_loader: - kv  37:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  38:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
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
llama_model_loader: - kv  50:                                   split.no u16              = 0
llama_model_loader: - kv  51:                                split.count u16              = 0
llama_model_loader: - kv  52:                        split.tensors.count i32              = 1147
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  612 tensors
llama_model_loader: - type iq4_k_r4:  116 tensors
llama_model_loader: - type iq5_k_r4:   58 tensors
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
llm_load_print_meta: model ftype      = IQ4_K_R4 - 4.5 bpw
llm_load_print_meta: model params     = 672.050 B
llm_load_print_meta: model size       = 386.183 GiB (4.936 BPW) 
llm_load_print_meta: repeating layers = 384.349 GiB (4.926 BPW, 670.196 B parameters)
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
llm_load_tensors: ggml ctx size =    0.93 MiB
Tensor blk.0.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.0.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.0.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.0.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.0.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.0.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.0.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.0.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.0.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.0.attn_output.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_gate.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_down.weight buffer type overriden to CUDA0
Tensor blk.1.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.1.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.1.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.1.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.1.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.1.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.1.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.1.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.1.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.1.attn_output.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_gate.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_down.weight buffer type overriden to CUDA0
Tensor blk.2.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.2.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.2.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.2.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.2.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.2.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.2.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.2.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.2.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.2.attn_output.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_gate.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_down.weight buffer type overriden to CUDA0
Tensor blk.3.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.3.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.3.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.3.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.3.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.3.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.3.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.3.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.3.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.3.attn_output.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.3.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.4.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.4.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.4.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.4.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.4.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.4.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.4.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.4.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.4.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.4.attn_output.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.4.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.4.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.5.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.5.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.5.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.5.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.5.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.5.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.5.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.5.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.5.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.5.attn_output.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.5.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.5.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.6.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.6.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.6.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.6.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.6.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.6.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.6.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.6.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.6.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.6.attn_output.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.6.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.6.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.7.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.7.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.7.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.7.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.7.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.7.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.7.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.7.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.7.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.7.attn_output.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.7.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.8.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.8.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.8.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.8.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.8.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.8.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.8.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.8.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.8.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.8.attn_output.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.8.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.9.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.9.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.9.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.9.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.9.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.9.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.9.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.9.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.9.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.9.attn_output.weight buffer type overriden to CUDA0
Tensor blk.9.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.9.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.9.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.9.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.9.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.9.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.10.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.10.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.10.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.10.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.10.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.10.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.10.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.10.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.10.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.10.attn_output.weight buffer type overriden to CUDA0
Tensor blk.10.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.10.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.10.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.10.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.10.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.11.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.11.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.11.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.11.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.11.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.11.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.11.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.11.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.11.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.11.attn_output.weight buffer type overriden to CUDA0
Tensor blk.11.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.11.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.11.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.11.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.11.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.11.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.12.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.12.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.12.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.12.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.12.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.12.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.12.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.12.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.12.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.12.attn_output.weight buffer type overriden to CUDA0
Tensor blk.12.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.12.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.12.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.12.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.12.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.13.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.13.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.13.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.13.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.13.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.13.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.13.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.13.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.13.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.13.attn_output.weight buffer type overriden to CUDA0
Tensor blk.13.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.13.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.13.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.13.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.13.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.13.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.14.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.14.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.14.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.14.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.14.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.14.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.14.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.14.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.14.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.14.attn_output.weight buffer type overriden to CUDA0
Tensor blk.14.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.14.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.14.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.14.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.14.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.15.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.15.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.15.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.15.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.15.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.15.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.15.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.15.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.15.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.15.attn_output.weight buffer type overriden to CUDA0
Tensor blk.15.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.15.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.15.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.15.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.15.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.16.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.16.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.16.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.16.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.16.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.16.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.16.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.16.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.16.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.16.attn_output.weight buffer type overriden to CUDA0
Tensor blk.16.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.16.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.16.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.16.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.16.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.16.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.17.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.17.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.17.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.17.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.17.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.17.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.17.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.17.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.17.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.17.attn_output.weight buffer type overriden to CUDA0
Tensor blk.17.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.17.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.17.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.17.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.17.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.17.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.18.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.18.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.18.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.18.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.18.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.18.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.18.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.18.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.18.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.18.attn_output.weight buffer type overriden to CUDA0
Tensor blk.18.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.18.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.18.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.18.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.18.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.18.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.19.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.19.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.19.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.19.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.19.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.19.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.19.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.19.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.19.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.19.attn_output.weight buffer type overriden to CUDA0
Tensor blk.19.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.19.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.19.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.19.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.19.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.19.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.20.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.20.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.20.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.20.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.20.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.20.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.20.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.20.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.20.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.20.attn_output.weight buffer type overriden to CUDA0
Tensor blk.20.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.20.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.20.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.20.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.20.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.20.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.21.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.21.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.21.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.21.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.21.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.21.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.21.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.21.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.21.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.21.attn_output.weight buffer type overriden to CUDA0
Tensor blk.21.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.21.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.21.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.21.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.21.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.21.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.22.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.22.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.22.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.22.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.22.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.22.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.22.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.22.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.22.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.22.attn_output.weight buffer type overriden to CUDA0
Tensor blk.22.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.22.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.22.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.22.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.22.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.22.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.23.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.23.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.23.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.23.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.23.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.23.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.23.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.23.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.23.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.23.attn_output.weight buffer type overriden to CUDA0
Tensor blk.23.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.23.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.23.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.23.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.23.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.23.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.24.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.24.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.24.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.24.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.24.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.24.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.24.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.24.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.24.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.24.attn_output.weight buffer type overriden to CUDA0
Tensor blk.24.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.24.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.24.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.24.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.24.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.24.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.25.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.25.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.25.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.25.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.25.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.25.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.25.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.25.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.25.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.25.attn_output.weight buffer type overriden to CUDA0
Tensor blk.25.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.25.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.25.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.25.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.25.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.25.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.26.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.26.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.26.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.26.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.26.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.26.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.26.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.26.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.26.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.26.attn_output.weight buffer type overriden to CUDA0
Tensor blk.26.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.26.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.26.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.26.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.26.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.26.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.27.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.27.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.27.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.27.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.27.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.27.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.27.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.27.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.27.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.27.attn_output.weight buffer type overriden to CUDA0
Tensor blk.27.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.27.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.27.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.27.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.27.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.27.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.28.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.28.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.28.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.28.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.28.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.28.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.28.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.28.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.28.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.28.attn_output.weight buffer type overriden to CUDA0
Tensor blk.28.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.28.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.28.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.28.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.28.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.28.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.29.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.29.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.29.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.29.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.29.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.29.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.29.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.29.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.29.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.29.attn_output.weight buffer type overriden to CUDA0
Tensor blk.29.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.29.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.29.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.29.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.29.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.29.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.30.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.30.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.30.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.30.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.30.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.30.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.30.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.30.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.30.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.30.attn_output.weight buffer type overriden to CUDA0
Tensor blk.30.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.30.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.30.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.30.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.30.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.30.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.31.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.31.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.31.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.31.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.31.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.31.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.31.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.31.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.31.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.31.attn_output.weight buffer type overriden to CUDA0
Tensor blk.31.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.31.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.31.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.31.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.31.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.31.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.32.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.32.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.32.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.32.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.32.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.32.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.32.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.32.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.32.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.32.attn_output.weight buffer type overriden to CUDA0
Tensor blk.32.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.32.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.32.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.32.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.32.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.32.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.33.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.33.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.33.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.33.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.33.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.33.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.33.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.33.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.33.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.33.attn_output.weight buffer type overriden to CUDA0
Tensor blk.33.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.33.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.33.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.33.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.33.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.33.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.34.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.34.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.34.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.34.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.34.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.34.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.34.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.34.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.34.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.34.attn_output.weight buffer type overriden to CUDA0
Tensor blk.34.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.34.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.34.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.34.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.34.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.34.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.35.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.35.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.35.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.35.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.35.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.35.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.35.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.35.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.35.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.35.attn_output.weight buffer type overriden to CUDA0
Tensor blk.35.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.35.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.35.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.35.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.35.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.35.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.36.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.36.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.36.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.36.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.36.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.36.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.36.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.36.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.36.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.36.attn_output.weight buffer type overriden to CUDA0
Tensor blk.36.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.36.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.36.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.36.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.36.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.36.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.37.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.37.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.37.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.37.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.37.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.37.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.37.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.37.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.37.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.37.attn_output.weight buffer type overriden to CUDA0
Tensor blk.37.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.37.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.37.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.37.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.37.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.37.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.38.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.38.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.38.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.38.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.38.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.38.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.38.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.38.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.38.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.38.attn_output.weight buffer type overriden to CUDA0
Tensor blk.38.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.38.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.38.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.38.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.38.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.38.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.39.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.39.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.39.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.39.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.39.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.39.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.39.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.39.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.39.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.39.attn_output.weight buffer type overriden to CUDA0
Tensor blk.39.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.39.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.39.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.39.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.39.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.39.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.40.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.40.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.40.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.40.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.40.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.40.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.40.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.40.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.40.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.40.attn_output.weight buffer type overriden to CUDA0
Tensor blk.40.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.40.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.40.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.40.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.40.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.40.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.41.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.41.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.41.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.41.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.41.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.41.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.41.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.41.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.41.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.41.attn_output.weight buffer type overriden to CUDA0
Tensor blk.41.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.41.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.41.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.41.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.41.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.41.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.42.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.42.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.42.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.42.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.42.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.42.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.42.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.42.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.42.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.42.attn_output.weight buffer type overriden to CUDA0
Tensor blk.42.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.42.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.42.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.42.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.42.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.42.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.43.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.43.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.43.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.43.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.43.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.43.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.43.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.43.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.43.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.43.attn_output.weight buffer type overriden to CUDA0
Tensor blk.43.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.43.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.43.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.43.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.43.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.43.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.44.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.44.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.44.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.44.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.44.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.44.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.44.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.44.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.44.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.44.attn_output.weight buffer type overriden to CUDA0
Tensor blk.44.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.44.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.44.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.44.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.44.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.44.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.45.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.45.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.45.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.45.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.45.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.45.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.45.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.45.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.45.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.45.attn_output.weight buffer type overriden to CUDA0
Tensor blk.45.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.45.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.45.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.45.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.45.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.45.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.46.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.46.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.46.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.46.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.46.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.46.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.46.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.46.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.46.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.46.attn_output.weight buffer type overriden to CUDA0
Tensor blk.46.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.46.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.46.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.46.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.46.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.46.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.47.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.47.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.47.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.47.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.47.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.47.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.47.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.47.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.47.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.47.attn_output.weight buffer type overriden to CUDA0
Tensor blk.47.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.47.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.47.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.47.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.47.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.47.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.48.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.48.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.48.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.48.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.48.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.48.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.48.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.48.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.48.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.48.attn_output.weight buffer type overriden to CUDA0
Tensor blk.48.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.48.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.48.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.48.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.48.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.48.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.49.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.49.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.49.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.49.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.49.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.49.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.49.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.49.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.49.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.49.attn_output.weight buffer type overriden to CUDA0
Tensor blk.49.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.49.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.49.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.49.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.49.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.49.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.50.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.50.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.50.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.50.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.50.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.50.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.50.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.50.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.50.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.50.attn_output.weight buffer type overriden to CUDA0
Tensor blk.50.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.50.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.50.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.50.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.50.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.50.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.51.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.51.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.51.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.51.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.51.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.51.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.51.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.51.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.51.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.51.attn_output.weight buffer type overriden to CUDA0
Tensor blk.51.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.51.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.51.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.51.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.51.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.51.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.52.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.52.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.52.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.52.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.52.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.52.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.52.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.52.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.52.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.52.attn_output.weight buffer type overriden to CUDA0
Tensor blk.52.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.52.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.52.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.52.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.52.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.52.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.53.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.53.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.53.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.53.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.53.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.53.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.53.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.53.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.53.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.53.attn_output.weight buffer type overriden to CUDA0
Tensor blk.53.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.53.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.53.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.53.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.53.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.53.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.54.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.54.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.54.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.54.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.54.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.54.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.54.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.54.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.54.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.54.attn_output.weight buffer type overriden to CUDA0
Tensor blk.54.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.54.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.54.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.54.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.54.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.54.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.55.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.55.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.55.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.55.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.55.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.55.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.55.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.55.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.55.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.55.attn_output.weight buffer type overriden to CUDA0
Tensor blk.55.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.55.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.55.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.55.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.55.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.55.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.56.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.56.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.56.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.56.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.56.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.56.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.56.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.56.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.56.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.56.attn_output.weight buffer type overriden to CUDA0
Tensor blk.56.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.56.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.56.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.56.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.56.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.56.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.57.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.57.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.57.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.57.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.57.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.57.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.57.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.57.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.57.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.57.attn_output.weight buffer type overriden to CUDA0
Tensor blk.57.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.57.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.57.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.57.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.57.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.57.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.58.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.58.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.58.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.58.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.58.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.58.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.58.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.58.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.58.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.58.attn_output.weight buffer type overriden to CUDA0
Tensor blk.58.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.58.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.58.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.58.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.58.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.58.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.59.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.59.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.59.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.59.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.59.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.59.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.59.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.59.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.59.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.59.attn_output.weight buffer type overriden to CUDA0
Tensor blk.59.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.59.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.59.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.59.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.59.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.59.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.60.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.60.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.60.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.60.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.60.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.60.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.60.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.60.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.60.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.60.attn_output.weight buffer type overriden to CUDA0
Tensor blk.60.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.60.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.60.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.60.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.60.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.60.ffn_up_shexp.weight buffer type overriden to CUDA0
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size = 392428.85 MiB
llm_load_tensors:        CPU buffer size =   938.98 MiB
llm_load_tensors:      CUDA0 buffer size = 17744.02 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 1024
llama_new_context_with_model: n_ubatch   = 1024
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
llama_new_context_with_model:      CUDA0 compute buffer size =  3650.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   352.01 MiB
llama_new_context_with_model: graph nodes  = 8245
llama_new_context_with_model: graph splits = 118
INFO [                    init] initializing slots | tid="136521606795264" timestamp=1748008022 n_slots=1
INFO [                    init] new slot | tid="136521606795264" timestamp=1748008022 id_slot=0 n_ctx_slot=32768
INFO [                    main] model loaded | tid="136521606795264" timestamp=1748008022
INFO [                    main] chat template | tid="136521606795264" timestamp=1748008022 chat_example="You are a helpful assistant\n\n<｜User｜>Hello<｜Assistant｜>Hi there<｜end▁of▁sentence｜><｜User｜>How are you?<｜Assistant｜>" built_in=true
INFO [                    main] HTTP server listening | tid="136521606795264" timestamp=1748008022 n_threads_http="31" port="7862" hostname="0.0.0.0"
INFO [            update_slots] all slots are idle | tid="136521606795264" timestamp=1748008022
INFO [   launch_slot_with_task] slot is processing task | tid="136521606795264" timestamp=1748008040 id_slot=0 id_task=0
INFO [            update_slots] kv cache rm [p0, end) | tid="136521606795264" timestamp=1748008040 id_slot=0 id_task=0 p0=0
INFO [            update_slots] kv cache rm [p0, end) | tid="136521606795264" timestamp=1748008051 id_slot=0 id_task=0 p0=1024
INFO [            update_slots] kv cache rm [p0, end) | tid="136521606795264" timestamp=1748008063 id_slot=0 id_task=0 p0=2048
INFO [           print_timings] prompt eval time     =   25767.00 ms /  2190 tokens (   11.77 ms per token,    84.99 tokens per second) | tid="136521606795264" timestamp=1748008081 id_slot=0 id_task=0 t_prompt_processing=25767.002 n_prompt_tokens_processed=2190 t_token=11.765754337899544 n_tokens_second=84.9924255836981
INFO [           print_timings] generation eval time =   15701.68 ms /   222 runs   (   70.73 ms per token,    14.14 tokens per second) | tid="136521606795264" timestamp=1748008081 id_slot=0 id_task=0 t_token_generation=15701.681 n_decoded=222 t_token=70.7282927927928 n_tokens_second=14.138613566279941
INFO [           print_timings]           total time =   41468.68 ms | tid="136521606795264" timestamp=1748008081 id_slot=0 id_task=0 t_prompt_processing=25767.002 t_token_generation=15701.681 t_total=41468.683000000005
INFO [            update_slots] slot released | tid="136521606795264" timestamp=1748008081 id_slot=0 id_task=0 n_ctx=32768 n_past=2411 n_system_tokens=0 n_cache_tokens=2411 truncated=false
INFO [            update_slots] all slots are idle | tid="136521606795264" timestamp=1748008081
INFO [      log_server_request] request | tid="136105332502528" timestamp=1748008081 remote_addr="10.254.1.2" remote_port=51316 status=200 method="POST" path="/completion" params={}
INFO [            update_slots] all slots are idle | tid="136521606795264" timestamp=1748008081
`

BAD log:

$ ./build-bad/bin/llama-cli --version
version: 3705 (ec456322)
built with cc (Ubuntu 14.2.0-4ubuntu2) 14.2.0 for x86_64-linux-gnu

`ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA RTX 6000 Ada Generation, compute capability 8.9, VMM: yes
  Device 1: NVIDIA RTX 6000 Ada Generation, compute capability 8.9, VMM: yes
INFO [                    main] build info | tid="127511205212160" timestamp=1748008231 build=3705 commit="ec456322"
INFO [                    main] system info | tid="127511205212160" timestamp=1748008231 n_threads=16 n_threads_batch=32 total_threads=32 system_info="AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: loaded meta data with 53 key-value pairs and 1147 tensors from /home/corey/AIModels/textgen/DeepSeek-V3-0324-IQ4_K_R4.gguf (version GGUF V3 (latest))
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
llama_model_loader: - kv  16:                          general.file_type u32              = 340
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
llama_model_loader: - kv  36:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<�...
llama_model_loader: - kv  37:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  38:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
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
llama_model_loader: - kv  50:                                   split.no u16              = 0
llama_model_loader: - kv  51:                                split.count u16              = 0
llama_model_loader: - kv  52:                        split.tensors.count i32              = 1147
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  612 tensors
llama_model_loader: - type iq4_k_r4:  116 tensors
llama_model_loader: - type iq5_k_r4:   58 tensors
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
llm_load_print_meta: model ftype      = IQ4_K_R4 - 4.5 bpw
llm_load_print_meta: model params     = 672.050 B
llm_load_print_meta: model size       = 386.183 GiB (4.936 BPW) 
llm_load_print_meta: repeating layers = 384.349 GiB (4.926 BPW, 670.196 B parameters)
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
llm_load_tensors: ggml ctx size =    0.93 MiB
Tensor blk.0.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.0.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.0.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.0.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.0.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.0.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.0.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.0.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.0.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.0.attn_output.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_gate.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_down.weight buffer type overriden to CUDA0
Tensor blk.1.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.1.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.1.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.1.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.1.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.1.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.1.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.1.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.1.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.1.attn_output.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_gate.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_down.weight buffer type overriden to CUDA0
Tensor blk.2.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.2.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.2.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.2.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.2.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.2.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.2.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.2.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.2.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.2.attn_output.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_gate.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_down.weight buffer type overriden to CUDA0
Tensor blk.3.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.3.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.3.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.3.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.3.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.3.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.3.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.3.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.3.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.3.attn_output.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.3.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.4.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.4.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.4.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.4.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.4.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.4.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.4.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.4.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.4.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.4.attn_output.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.4.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.4.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.5.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.5.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.5.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.5.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.5.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.5.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.5.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.5.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.5.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.5.attn_output.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.5.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.5.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.6.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.6.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.6.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.6.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.6.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.6.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.6.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.6.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.6.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.6.attn_output.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.6.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.6.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.7.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.7.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.7.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.7.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.7.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.7.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.7.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.7.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.7.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.7.attn_output.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.7.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.8.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.8.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.8.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.8.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.8.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.8.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.8.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.8.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.8.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.8.attn_output.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.8.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.9.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.9.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.9.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.9.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.9.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.9.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.9.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.9.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.9.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.9.attn_output.weight buffer type overriden to CUDA0
Tensor blk.9.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.9.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.9.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.9.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.9.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.9.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.10.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.10.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.10.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.10.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.10.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.10.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.10.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.10.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.10.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.10.attn_output.weight buffer type overriden to CUDA0
Tensor blk.10.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.10.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.10.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.10.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.10.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.11.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.11.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.11.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.11.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.11.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.11.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.11.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.11.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.11.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.11.attn_output.weight buffer type overriden to CUDA0
Tensor blk.11.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.11.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.11.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.11.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.11.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.11.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.12.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.12.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.12.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.12.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.12.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.12.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.12.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.12.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.12.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.12.attn_output.weight buffer type overriden to CUDA0
Tensor blk.12.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.12.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.12.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.12.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.12.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.13.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.13.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.13.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.13.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.13.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.13.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.13.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.13.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.13.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.13.attn_output.weight buffer type overriden to CUDA0
Tensor blk.13.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.13.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.13.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.13.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.13.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.13.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.14.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.14.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.14.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.14.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.14.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.14.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.14.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.14.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.14.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.14.attn_output.weight buffer type overriden to CUDA0
Tensor blk.14.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.14.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.14.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.14.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.14.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.15.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.15.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.15.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.15.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.15.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.15.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.15.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.15.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.15.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.15.attn_output.weight buffer type overriden to CUDA0
Tensor blk.15.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.15.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.15.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.15.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.15.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.16.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.16.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.16.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.16.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.16.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.16.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.16.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.16.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.16.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.16.attn_output.weight buffer type overriden to CUDA0
Tensor blk.16.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.16.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.16.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.16.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.16.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.16.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.17.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.17.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.17.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.17.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.17.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.17.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.17.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.17.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.17.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.17.attn_output.weight buffer type overriden to CUDA0
Tensor blk.17.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.17.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.17.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.17.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.17.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.17.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.18.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.18.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.18.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.18.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.18.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.18.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.18.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.18.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.18.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.18.attn_output.weight buffer type overriden to CUDA0
Tensor blk.18.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.18.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.18.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.18.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.18.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.18.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.19.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.19.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.19.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.19.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.19.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.19.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.19.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.19.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.19.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.19.attn_output.weight buffer type overriden to CUDA0
Tensor blk.19.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.19.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.19.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.19.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.19.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.19.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.20.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.20.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.20.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.20.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.20.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.20.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.20.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.20.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.20.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.20.attn_output.weight buffer type overriden to CUDA0
Tensor blk.20.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.20.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.20.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.20.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.20.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.20.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.21.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.21.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.21.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.21.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.21.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.21.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.21.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.21.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.21.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.21.attn_output.weight buffer type overriden to CUDA0
Tensor blk.21.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.21.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.21.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.21.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.21.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.21.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.22.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.22.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.22.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.22.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.22.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.22.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.22.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.22.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.22.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.22.attn_output.weight buffer type overriden to CUDA0
Tensor blk.22.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.22.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.22.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.22.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.22.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.22.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.23.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.23.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.23.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.23.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.23.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.23.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.23.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.23.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.23.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.23.attn_output.weight buffer type overriden to CUDA0
Tensor blk.23.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.23.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.23.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.23.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.23.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.23.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.24.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.24.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.24.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.24.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.24.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.24.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.24.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.24.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.24.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.24.attn_output.weight buffer type overriden to CUDA0
Tensor blk.24.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.24.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.24.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.24.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.24.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.24.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.25.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.25.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.25.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.25.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.25.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.25.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.25.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.25.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.25.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.25.attn_output.weight buffer type overriden to CUDA0
Tensor blk.25.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.25.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.25.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.25.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.25.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.25.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.26.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.26.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.26.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.26.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.26.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.26.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.26.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.26.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.26.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.26.attn_output.weight buffer type overriden to CUDA0
Tensor blk.26.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.26.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.26.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.26.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.26.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.26.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.27.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.27.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.27.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.27.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.27.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.27.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.27.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.27.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.27.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.27.attn_output.weight buffer type overriden to CUDA0
Tensor blk.27.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.27.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.27.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.27.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.27.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.27.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.28.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.28.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.28.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.28.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.28.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.28.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.28.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.28.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.28.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.28.attn_output.weight buffer type overriden to CUDA0
Tensor blk.28.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.28.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.28.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.28.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.28.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.28.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.29.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.29.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.29.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.29.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.29.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.29.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.29.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.29.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.29.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.29.attn_output.weight buffer type overriden to CUDA0
Tensor blk.29.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.29.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.29.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.29.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.29.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.29.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.30.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.30.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.30.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.30.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.30.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.30.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.30.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.30.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.30.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.30.attn_output.weight buffer type overriden to CUDA0
Tensor blk.30.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.30.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.30.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.30.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.30.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.30.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.31.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.31.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.31.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.31.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.31.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.31.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.31.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.31.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.31.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.31.attn_output.weight buffer type overriden to CUDA0
Tensor blk.31.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.31.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.31.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.31.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.31.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.31.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.32.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.32.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.32.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.32.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.32.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.32.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.32.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.32.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.32.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.32.attn_output.weight buffer type overriden to CUDA0
Tensor blk.32.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.32.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.32.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.32.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.32.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.32.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.33.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.33.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.33.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.33.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.33.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.33.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.33.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.33.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.33.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.33.attn_output.weight buffer type overriden to CUDA0
Tensor blk.33.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.33.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.33.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.33.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.33.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.33.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.34.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.34.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.34.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.34.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.34.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.34.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.34.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.34.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.34.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.34.attn_output.weight buffer type overriden to CUDA0
Tensor blk.34.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.34.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.34.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.34.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.34.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.34.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.35.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.35.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.35.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.35.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.35.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.35.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.35.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.35.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.35.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.35.attn_output.weight buffer type overriden to CUDA0
Tensor blk.35.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.35.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.35.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.35.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.35.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.35.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.36.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.36.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.36.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.36.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.36.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.36.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.36.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.36.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.36.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.36.attn_output.weight buffer type overriden to CUDA0
Tensor blk.36.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.36.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.36.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.36.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.36.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.36.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.37.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.37.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.37.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.37.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.37.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.37.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.37.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.37.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.37.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.37.attn_output.weight buffer type overriden to CUDA0
Tensor blk.37.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.37.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.37.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.37.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.37.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.37.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.38.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.38.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.38.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.38.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.38.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.38.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.38.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.38.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.38.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.38.attn_output.weight buffer type overriden to CUDA0
Tensor blk.38.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.38.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.38.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.38.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.38.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.38.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.39.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.39.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.39.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.39.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.39.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.39.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.39.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.39.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.39.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.39.attn_output.weight buffer type overriden to CUDA0
Tensor blk.39.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.39.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.39.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.39.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.39.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.39.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.40.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.40.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.40.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.40.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.40.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.40.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.40.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.40.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.40.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.40.attn_output.weight buffer type overriden to CUDA0
Tensor blk.40.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.40.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.40.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.40.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.40.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.40.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.41.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.41.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.41.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.41.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.41.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.41.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.41.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.41.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.41.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.41.attn_output.weight buffer type overriden to CUDA0
Tensor blk.41.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.41.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.41.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.41.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.41.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.41.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.42.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.42.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.42.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.42.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.42.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.42.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.42.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.42.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.42.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.42.attn_output.weight buffer type overriden to CUDA0
Tensor blk.42.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.42.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.42.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.42.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.42.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.42.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.43.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.43.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.43.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.43.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.43.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.43.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.43.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.43.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.43.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.43.attn_output.weight buffer type overriden to CUDA0
Tensor blk.43.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.43.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.43.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.43.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.43.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.43.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.44.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.44.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.44.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.44.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.44.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.44.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.44.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.44.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.44.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.44.attn_output.weight buffer type overriden to CUDA0
Tensor blk.44.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.44.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.44.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.44.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.44.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.44.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.45.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.45.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.45.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.45.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.45.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.45.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.45.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.45.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.45.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.45.attn_output.weight buffer type overriden to CUDA0
Tensor blk.45.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.45.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.45.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.45.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.45.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.45.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.46.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.46.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.46.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.46.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.46.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.46.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.46.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.46.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.46.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.46.attn_output.weight buffer type overriden to CUDA0
Tensor blk.46.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.46.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.46.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.46.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.46.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.46.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.47.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.47.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.47.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.47.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.47.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.47.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.47.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.47.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.47.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.47.attn_output.weight buffer type overriden to CUDA0
Tensor blk.47.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.47.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.47.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.47.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.47.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.47.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.48.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.48.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.48.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.48.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.48.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.48.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.48.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.48.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.48.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.48.attn_output.weight buffer type overriden to CUDA0
Tensor blk.48.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.48.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.48.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.48.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.48.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.48.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.49.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.49.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.49.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.49.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.49.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.49.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.49.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.49.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.49.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.49.attn_output.weight buffer type overriden to CUDA0
Tensor blk.49.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.49.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.49.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.49.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.49.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.49.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.50.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.50.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.50.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.50.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.50.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.50.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.50.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.50.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.50.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.50.attn_output.weight buffer type overriden to CUDA0
Tensor blk.50.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.50.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.50.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.50.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.50.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.50.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.51.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.51.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.51.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.51.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.51.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.51.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.51.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.51.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.51.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.51.attn_output.weight buffer type overriden to CUDA0
Tensor blk.51.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.51.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.51.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.51.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.51.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.51.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.52.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.52.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.52.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.52.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.52.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.52.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.52.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.52.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.52.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.52.attn_output.weight buffer type overriden to CUDA0
Tensor blk.52.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.52.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.52.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.52.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.52.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.52.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.53.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.53.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.53.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.53.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.53.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.53.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.53.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.53.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.53.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.53.attn_output.weight buffer type overriden to CUDA0
Tensor blk.53.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.53.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.53.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.53.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.53.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.53.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.54.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.54.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.54.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.54.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.54.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.54.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.54.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.54.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.54.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.54.attn_output.weight buffer type overriden to CUDA0
Tensor blk.54.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.54.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.54.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.54.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.54.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.54.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.55.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.55.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.55.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.55.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.55.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.55.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.55.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.55.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.55.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.55.attn_output.weight buffer type overriden to CUDA0
Tensor blk.55.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.55.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.55.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.55.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.55.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.55.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.56.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.56.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.56.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.56.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.56.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.56.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.56.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.56.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.56.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.56.attn_output.weight buffer type overriden to CUDA0
Tensor blk.56.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.56.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.56.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.56.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.56.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.56.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.57.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.57.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.57.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.57.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.57.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.57.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.57.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.57.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.57.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.57.attn_output.weight buffer type overriden to CUDA0
Tensor blk.57.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.57.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.57.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.57.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.57.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.57.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.58.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.58.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.58.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.58.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.58.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.58.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.58.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.58.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.58.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.58.attn_output.weight buffer type overriden to CUDA0
Tensor blk.58.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.58.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.58.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.58.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.58.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.58.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.59.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.59.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.59.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.59.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.59.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.59.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.59.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.59.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.59.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.59.attn_output.weight buffer type overriden to CUDA0
Tensor blk.59.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.59.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.59.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.59.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.59.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.59.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.60.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.60.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.60.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.60.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.60.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.60.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.60.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.60.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.60.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.60.attn_output.weight buffer type overriden to CUDA0
Tensor blk.60.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.60.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.60.exp_probs_b.bias buffer type overriden to CUDA0
Tensor blk.60.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.60.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.60.ffn_up_shexp.weight buffer type overriden to CUDA0
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size = 392428.85 MiB
llm_load_tensors:        CPU buffer size =   938.98 MiB
llm_load_tensors:      CUDA0 buffer size = 17744.02 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 1024
llama_new_context_with_model: n_ubatch   = 1024
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
llama_new_context_with_model:      CUDA0 compute buffer size =  3650.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   352.01 MiB
llama_new_context_with_model: graph nodes  = 8245
llama_new_context_with_model: graph splits = 118
INFO [                    init] initializing slots | tid="127511205212160" timestamp=1748008241 n_slots=1
INFO [                    init] new slot | tid="127511205212160" timestamp=1748008241 id_slot=0 n_ctx_slot=32768
INFO [                    main] model loaded | tid="127511205212160" timestamp=1748008241
INFO [                    main] chat template | tid="127511205212160" timestamp=1748008241 chat_example="You are a helpful assistant\n\n<｜User｜>Hello<｜Assistant｜>Hi there<｜end▁of▁sentence｜><｜User｜>How are you?<｜Assistant｜>" built_in=true
INFO [                    main] HTTP server listening | tid="127511205212160" timestamp=1748008241 n_threads_http="31" port="7862" hostname="0.0.0.0"
INFO [            update_slots] all slots are idle | tid="127511205212160" timestamp=1748008241
INFO [   launch_slot_with_task] slot is processing task | tid="127511205212160" timestamp=1748008291 id_slot=0 id_task=0
INFO [            update_slots] kv cache rm [p0, end) | tid="127511205212160" timestamp=1748008291 id_slot=0 id_task=0 p0=0
INFO [            update_slots] kv cache rm [p0, end) | tid="127511205212160" timestamp=1748008303 id_slot=0 id_task=0 p0=1024
INFO [            update_slots] kv cache rm [p0, end) | tid="127511205212160" timestamp=1748008315 id_slot=0 id_task=0 p0=2048
INFO [           print_timings] prompt eval time     =   25845.83 ms /  2190 tokens (   11.80 ms per token,    84.73 tokens per second) | tid="127511205212160" timestamp=1748008339 id_slot=0 id_task=0 t_prompt_processing=25845.833 n_prompt_tokens_processed=2190 t_token=11.801750228310501 n_tokens_second=84.73319470879504
INFO [           print_timings] generation eval time =   21665.24 ms /   222 runs   (   97.59 ms per token,    10.25 tokens per second) | tid="127511205212160" timestamp=1748008339 id_slot=0 id_task=0 t_token_generation=21665.244 n_decoded=222 t_token=97.59118918918918 n_tokens_second=10.246826668557253
INFO [           print_timings]           total time =   47511.08 ms | tid="127511205212160" timestamp=1748008339 id_slot=0 id_task=0 t_prompt_processing=25845.833 t_token_generation=21665.244 t_total=47511.077
INFO [            update_slots] slot released | tid="127511205212160" timestamp=1748008339 id_slot=0 id_task=0 n_ctx=32768 n_past=2411 n_system_tokens=0 n_cache_tokens=2411 truncated=false
INFO [            update_slots] all slots are idle | tid="127511205212160" timestamp=1748008339
INFO [      log_server_request] request | tid="127095162204160" timestamp=1748008339 remote_addr="10.254.1.2" remote_port=43794 status=200 method="POST" path="/completion" params={}
INFO [            update_slots] all slots are idle | tid="127511205212160" timestamp=1748008339
`

---

👤 **ikawrakow** commented the **2025-05-23** at **15:09:25**:<br>

In my case I see zero difference between current main branch and a2b5057a0c9a2758830b6f841bb22150d2511bb1. Tested with DeepSeek-Lite (the 16B little sibling of DeepSeek-V3/R1) and Qwen3-30B-A3B using the exact same custom quantization as yours.

My CPU is Ryzen-7950X, so Zen4 core. Yours is Zen5, so both use the exact same implementation.
 
I wouldn't know why the performance would change. The 18k LOC `iqk_mul_mat.cpp` got refactored into multiple files for faster build times. There was zero change done in #435. 

I would try `echo 3 | sudo tee /proc/sys/vm/drop_caches`, and then load the model with the **main branch first** to see what happens.

---

👤 **cmoncure** commented the **2025-05-23** at **16:01:17**:<br>

Dropped cache.

Main (bad) build first "ec456322"
```
[           print_timings] prompt eval time     =   34619.60 ms /  2190 tokens (   15.81 ms per token,    63.26 tokens per second) | tid="138682949877760" timestamp=1748014236 id_slot=0 id_task=0 t_prompt_processing=34619.603 n_prompt_tokens_processed=2190 t_token=15.80803789954338 n_tokens_second=63.25895764893664
INFO [           print_timings] generation eval time =   22553.81 ms /   222 runs   (  101.59 ms per token,     9.84 tokens per second) | tid="138682949877760" timestamp=1748014236 id_slot=0 id_task=0 t_token_generation=22553.805 n_decoded=222 t_token=101.59371621621622 n_tokens_second=9.843128465462923
```

Switch to good build "a2b5057a"
```
INFO [           print_timings] prompt eval time     =   48430.56 ms /  2190 tokens (   22.11 ms per token,    45.22 tokens per second) | tid="128418970439680" timestamp=1748014922 id_slot=0 id_task=0 t_prompt_processing=48430.56 n_prompt_tokens_processed=2190 t_token=22.11441095890411 n_tokens_second=45.21938214218461
INFO [           print_timings] generation eval time =   24928.21 ms /   222 runs   (  112.29 ms per token,     8.91 tokens per second) | tid="128418970439680" timestamp=1748014922 id_slot=0 id_task=0 t_token_generation=24928.211 n_decoded=222 t_token=112.28923873873873 n_tokens_second=8.905572886879046
```

Well now both are bad.

Switch back to version: 3692 (b90d6ede)
```
INFO [           print_timings] prompt eval time     =   25607.00 ms /  2190 tokens (   11.69 ms per token,    85.52 tokens per second) | tid="132738167939072" timestamp=1748015946 id_slot=0 id_task=0 t_prompt_processing=25606.997 n_prompt_tokens_processed=2190 t_token=11.692692694063927 n_tokens_second=85.52349969033854
INFO [           print_timings] generation eval time =   15771.66 ms /   222 runs   (   71.04 ms per token,    14.08 tokens per second) | tid="132738167939072" timestamp=1748015946 id_slot=0 id_task=0 t_token_generation=15771.659 n_decoded=222 t_token=71.04350900900901 n_tokens_second=14.075881300755997
```
Alright, we're in business again.  I'll re-bisect dropping the cache each time.

---

👤 **ikawrakow** commented the **2025-05-23** at **16:28:30**:<br>

So, you cannot base your measurement on just a single load and one run with 2000 prompt tokens and 200 generated tokens. These giant models take some time to "warm up".

Your CPU has 16 cores, does `--threads-batch 32` help? In my case it always decreases performance compared to just using 16 threads on my 16-core CPU.

You could try a much simpler tensor override rule. Just `-exps=CPU -ngl 100`.

---

👤 **cmoncure** commented the **2025-05-23** at **18:33:25**:<br>

>  These giant models take some time to "warm up".

This differs from my observations, but I'll take it under advisement and post average results from 4 runs with 4 separate prompts, circling back to reuse one prompt at the end, and dropping cache with each build.

methodology:
1. echo 3 | sudo tee /proc/sys/vm/drop_caches
2. git checkout
3. cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
4. cmake --build build --config Release -j16
5. (my llama-server command)
6. prompt A
7. prompt B
8. prompt C
9. prompt A (repeated)

Runs:
1. version: 3698 (134d548) => 12.59 t/s (avg)
2. version: 3701 (b3036a8) => 12.50 t/s (avg)
3. version: 3703 (a2b5057) => 12.58 t/s (avg)
4. version: 3704 (b94cd3b) => 9.78 t/s (avg) !
5. version: 3703 (a2b5057) => 12.68 t/s (avg)
6. version: 3704 (b94cd3b) => 9.85 t/s (avg) !

(variance <= 0.14s in all runs)

Sure looks like version 3704 is bad. Maybe some compiler optimizations aren't applying?

---

👤 **cmoncure** commented the **2025-05-23** at **18:33:25**:<br>

>  These giant models take some time to "warm up".

This differs from my observations, but I'll take it under advisement and post average results from 4 runs with 4 separate prompts, circling back to reuse one prompt at the end, and dropping cache with each build.

methodology:
1. echo 3 | sudo tee /proc/sys/vm/drop_caches
2. git checkout
3. cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
4. cmake --build build --config Release -j16
5. (my llama-server command)
6. prompt A
7. prompt B
8. prompt C
9. prompt A (repeated)

Runs:
1. version: 3698 (134d548) => 12.59 t/s (avg)
2. version: 3701 (b3036a8) => 12.50 t/s (avg)
3. version: 3703 (a2b5057) => 12.58 t/s (avg)
4. version: 3704 (b94cd3b) => 9.78 t/s (avg) !
5. version: 3703 (a2b5057) => 12.68 t/s (avg)
6. version: 3704 (b94cd3b) => 9.85 t/s (avg) !

(variance <= 0.14s in all runs)

Sure looks like version 3703 is bad. Maybe some compiler optimizations aren't applying?

---

👤 **Ph0rk0z** commented the **2025-05-23** at **19:34:30**:<br>

Try with llama sweep bench to get a better average. I didn't notice anything either but I was just using qwen.

---

👤 **saood06** commented the **2025-05-24** at **23:53:08**:<br>

@cmoncure 

Do you mind trying if setting GGML_LTO on when building it helps?

---

👤 **cmoncure** commented the **2025-05-30** at **23:32:18**:<br>

Newer versions seem to have improved (to within 10% of a2b5057) so I'm closing this.