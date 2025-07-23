### 📝 [#530](https://github.com/ikawrakow/ik_llama.cpp/issues/530) - Getting crash on second prompt.

| **Author** | `mtcl` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-14 |
| **Updated** | 2025-06-15 |

---

#### Description

Getting crash on second prompt. Would there be any reason why? 


```
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ CUDA_VISIBLE_DEVICES="0,1" ./build/bin/llama-server \
  --model /home/mukul/dev-ai/models/unsloth/Qwen3-235B-A22B-128K-GGUF/Q4_K_M/Qwen3-235B-A22B-128K-Q4_K_M-00001-of-00003.gguf \
  --alias unsloth/Qwen3-235B-A22B-128K-Q4_K_M \
  --ctx-size 65536 \
  -ctk q8_0 -ctv q8_0 \
  -fa \
  -b 4096 -ub 4096 \
  -fmoe \
  --n-gpu-layers 100 \
  -ot "blk\.[0-9]\.ffn=CUDA0,blk\.1[0-4]\.ffn=CUDA0" \
  -ot "blk\.1[5-9]\.ffn=CUDA1,blk\.2[0-9]\.ffn=CUDA1" \
  --override-tensor exps=CPU \
  --parallel 1 \
  --threads 56 \
  --host 0.0.0.0 \
  --port 10002
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
  Device 1: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
INFO [                    main] build info | tid="136074680586240" timestamp=1749937648 build=3748 commit="066ed4fd"
INFO [                    main] system info | tid="136074680586240" timestamp=1749937648 n_threads=56 n_threads_batch=-1 total_threads=112 system_info="AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: additional 2 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 50 key-value pairs and 1131 tensors from /home/mukul/dev-ai/models/unsloth/Qwen3-235B-A22B-128K-GGUF/Q4_K_M/Qwen3-235B-A22B-128K-Q4_K_M-00001-of-00003.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3-235B-A22B-128K
llama_model_loader: - kv   3:                           general.finetune str              = 128k
llama_model_loader: - kv   4:                           general.basename str              = Qwen3-235B-A22B-128K
llama_model_loader: - kv   5:                       general.quantized_by str              = Unsloth
llama_model_loader: - kv   6:                         general.size_label str              = 235B-A22B
llama_model_loader: - kv   7:                            general.license str              = apache-2.0
llama_model_loader: - kv   8:                       general.license.link str              = https://huggingface.co/Qwen/Qwen3-235...
llama_model_loader: - kv   9:                           general.repo_url str              = https://huggingface.co/unsloth
llama_model_loader: - kv  10:                   general.base_model.count u32              = 1
llama_model_loader: - kv  11:                  general.base_model.0.name str              = Qwen3 235B A22B
llama_model_loader: - kv  12:          general.base_model.0.organization str              = Qwen
llama_model_loader: - kv  13:              general.base_model.0.repo_url str              = https://huggingface.co/Qwen/Qwen3-235...
llama_model_loader: - kv  14:                               general.tags arr[str,2]       = ["unsloth", "text-generation"]
llama_model_loader: - kv  15:                       qwen3moe.block_count u32              = 94
llama_model_loader: - kv  16:                    qwen3moe.context_length u32              = 131072
llama_model_loader: - kv  17:                  qwen3moe.embedding_length u32              = 4096
llama_model_loader: - kv  18:               qwen3moe.feed_forward_length u32              = 12288
llama_model_loader: - kv  19:              qwen3moe.attention.head_count u32              = 64
llama_model_loader: - kv  20:           qwen3moe.attention.head_count_kv u32              = 4
llama_model_loader: - kv  21:                    qwen3moe.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  22:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  23:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  24:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  25:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  26:                      qwen3moe.expert_count u32              = 128
llama_model_loader: - kv  27:        qwen3moe.expert_feed_forward_length u32              = 1536
llama_model_loader: - kv  28:                 qwen3moe.rope.scaling.type str              = yarn
llama_model_loader: - kv  29:               qwen3moe.rope.scaling.factor f32              = 4.000000
llama_model_loader: - kv  30: qwen3moe.rope.scaling.original_context_length u32              = 32768
llama_model_loader: - kv  31:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  32:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  33:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  34:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  35:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv  36:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  37:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  38:                tokenizer.ggml.bos_token_id u32              = 151643
llama_model_loader: - kv  39:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  40:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - kv  41:               general.quantization_version u32              = 2
llama_model_loader: - kv  42:                          general.file_type u32              = 15
llama_model_loader: - kv  43:                      quantize.imatrix.file str              = Qwen3-235B-A22B-128K-GGUF/imatrix_uns...
llama_model_loader: - kv  44:                   quantize.imatrix.dataset str              = unsloth_calibration_Qwen3-235B-A22B-1...
llama_model_loader: - kv  45:             quantize.imatrix.entries_count i32              = 752
llama_model_loader: - kv  46:              quantize.imatrix.chunks_count i32              = 46
llama_model_loader: - kv  47:                                   split.no u16              = 0
llama_model_loader: - kv  48:                        split.tensors.count i32              = 1131
llama_model_loader: - kv  49:                                split.count u16              = 3
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
llm_load_print_meta: n_ctx_train      = 131072
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
llm_load_print_meta: rope scaling     = yarn
llm_load_print_meta: freq_base_train  = 1000000.0
llm_load_print_meta: freq_scale_train = 0.25
llm_load_print_meta: n_ctx_orig_yarn  = 32768
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
llm_load_print_meta: general.name     = Qwen3-235B-A22B-128K
llm_load_print_meta: BOS token        = 151643 '<|endoftext|>'
llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
llm_load_print_meta: PAD token        = 151643 '<|endoftext|>'
llm_load_print_meta: LF token         = 148848 'ÄĬ'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp         = 1536
llm_load_tensors: ggml ctx size =    1.49 MiB
Tensor blk.0.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.9.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.9.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.9.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.9.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.9.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.10.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.10.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.10.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.10.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.11.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.11.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.11.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.11.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.11.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.12.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.12.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.13.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.13.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.13.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.13.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.13.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.14.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.14.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.15.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.15.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.15.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.15.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.16.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.16.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.16.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.16.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.16.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.17.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.17.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.17.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.17.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.17.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.18.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.18.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.18.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.18.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.18.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.19.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.19.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.19.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.19.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.19.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.20.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.20.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.20.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.20.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.20.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.21.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.21.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.21.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.21.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.21.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.22.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.22.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.22.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.22.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.22.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.23.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.23.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.23.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.23.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.23.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.24.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.24.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.24.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.24.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.24.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.25.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.25.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.25.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.25.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.25.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.26.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.26.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.26.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.26.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.26.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.27.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.27.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.27.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.27.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.27.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.28.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.28.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.28.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.28.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.28.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.29.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.29.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.29.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.29.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.29.ffn_up_exps.weight buffer type overriden to CUDA1
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
Tensor blk.61.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 94 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 95/95 layers to GPU
llm_load_tensors:        CPU buffer size =  3303.08 MiB
llm_load_tensors:        CPU buffer size = 47617.27 MiB
llm_load_tensors:        CPU buffer size = 40320.40 MiB
llm_load_tensors:        CPU buffer size =   333.84 MiB
llm_load_tensors:      CUDA0 buffer size = 23731.17 MiB
llm_load_tensors:      CUDA1 buffer size = 22811.95 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 65536
llama_new_context_with_model: n_batch    = 4096
llama_new_context_with_model: n_ubatch   = 4096
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 0.25
llama_kv_cache_init:      CUDA0 KV buffer size =  3264.02 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =  3128.02 MiB
llama_new_context_with_model: KV self size  = 6392.00 MiB, K (q8_0): 3196.00 MiB, V (q8_0): 3196.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     1.16 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      CUDA0 compute buffer size =  2432.02 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =  2502.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =  1088.05 MiB
llama_new_context_with_model: graph nodes  = 3672
llama_new_context_with_model: graph splits = 207
INFO [                    init] initializing slots | tid="136074680586240" timestamp=1749937723 n_slots=1
INFO [                    init] new slot | tid="136074680586240" timestamp=1749937723 id_slot=0 n_ctx_slot=65536
INFO [                    main] model loaded | tid="136074680586240" timestamp=1749937723
INFO [                    main] chat template | tid="136074680586240" timestamp=1749937723 chat_example="<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there<|im_end|>\n<|im_start|>user\nHow are you?<|im_end|>\n<|im_start|>assistant\n" built_in=true
INFO [                    main] HTTP server listening | tid="136074680586240" timestamp=1749937723 n_threads_http="111" port="10002" hostname="0.0.0.0"
INFO [            update_slots] all slots are idle | tid="136074680586240" timestamp=1749937723
INFO [      log_server_request] request | tid="136063907516416" timestamp=1749937803 remote_addr="172.17.0.3" remote_port=48272 status=200 method="GET" path="/v1/models" params={}
INFO [      log_server_request] request | tid="136063825735680" timestamp=1749937805 remote_addr="172.17.0.3" remote_port=33746 status=200 method="GET" path="/v1/models" params={}
INFO [      log_server_request] request | tid="136063817342976" timestamp=1749937806 remote_addr="172.17.0.3" remote_port=33748 status=200 method="GET" path="/v1/models" params={}
INFO [      log_server_request] request | tid="136063800557568" timestamp=1749937814 remote_addr="172.17.0.3" remote_port=33760 status=200 method="GET" path="/v1/models" params={}
INFO [   launch_slot_with_task] slot is processing task | tid="136074680586240" timestamp=1749937814 id_slot=0 id_task=0
INFO [            update_slots] kv cache rm [p0, end) | tid="136074680586240" timestamp=1749937814 id_slot=0 id_task=0 p0=0
INFO [            update_slots] kv cache rm [p0, end) | tid="136074680586240" timestamp=1749937826 id_slot=0 id_task=0 p0=4096
INFO [            update_slots] kv cache rm [p0, end) | tid="136074680586240" timestamp=1749937838 id_slot=0 id_task=0 p0=8192
INFO [            update_slots] kv cache rm [p0, end) | tid="136074680586240" timestamp=1749937850 id_slot=0 id_task=0 p0=12288
INFO [            update_slots] kv cache rm [p0, end) | tid="136074680586240" timestamp=1749937862 id_slot=0 id_task=0 p0=16384
INFO [           print_timings] prompt eval time     =   59784.01 ms / 19060 tokens (    3.14 ms per token,   318.81 tokens per second) | tid="136074680586240" timestamp=1749938015 id_slot=0 id_task=0 t_prompt_processing=59784.01 n_prompt_tokens_processed=19060 t_token=3.1366217208814273 n_tokens_second=318.8143451735673
INFO [           print_timings] generation eval time =  141528.25 ms /  2272 runs   (   62.29 ms per token,    16.05 tokens per second) | tid="136074680586240" timestamp=1749938015 id_slot=0 id_task=0 t_token_generation=141528.252 n_decoded=2272 t_token=62.29236443661972 n_tokens_second=16.053331881750363
INFO [           print_timings]           total time =  201312.26 ms | tid="136074680586240" timestamp=1749938015 id_slot=0 id_task=0 t_prompt_processing=59784.01 t_token_generation=141528.252 t_total=201312.26200000002
INFO [            update_slots] slot released | tid="136074680586240" timestamp=1749938015 id_slot=0 id_task=0 n_ctx=65536 n_past=21331 n_system_tokens=0 n_cache_tokens=21331 truncated=false
INFO [            update_slots] all slots are idle | tid="136074680586240" timestamp=1749938015
INFO [      log_server_request] request | tid="136063808950272" timestamp=1749938015 remote_addr="172.17.0.3" remote_port=33772 status=200 method="POST" path="/v1/chat/completions" params={}
INFO [            update_slots] all slots are idle | tid="136074680586240" timestamp=1749938015
INFO [      log_server_request] request | tid="136063775379456" timestamp=1749938035 remote_addr="172.17.0.3" remote_port=57224 status=200 method="GET" path="/v1/models" params={}
INFO [      log_server_request] request | tid="136063783772160" timestamp=1749938065 remote_addr="172.17.0.3" remote_port=42160 status=200 method="GET" path="/v1/models" params={}
INFO [   launch_slot_with_task] slot is processing task | tid="136074680586240" timestamp=1749938065 id_slot=0 id_task=2278
INFO [            update_slots] kv cache rm [p0, end) | tid="136074680586240" timestamp=1749938065 id_slot=0 id_task=2278 p0=1
INFO [            update_slots] kv cache rm [p0, end) | tid="136074680586240" timestamp=1749938077 id_slot=0 id_task=2278 p0=4097
INFO [            update_slots] kv cache rm [p0, end) | tid="136074680586240" timestamp=1749938089 id_slot=0 id_task=2278 p0=8193
INFO [            update_slots] kv cache rm [p0, end) | tid="136074680586240" timestamp=1749938101 id_slot=0 id_task=2278 p0=12289
INFO [            update_slots] kv cache rm [p0, end) | tid="136074680586240" timestamp=1749938113 id_slot=0 id_task=2278 p0=16385
INFO [            update_slots] kv cache rm [p0, end) | tid="136074680586240" timestamp=1749938125 id_slot=0 id_task=2278 p0=20481
/home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: Fatal error
/home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: Fatal error
/home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: Fatal error
/home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: Fatal error
/home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: Fatal error/home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: Fatal error

/home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: Fatal error
/home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: Fatal error
/home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: Fatal error
/home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: Fatal error
/home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: Fatal error
/home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: Fatal error
/home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: Fatal error
/home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: Fatal error/home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: Fatal error
/home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: /home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: Fatal error
/home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: Fatal error
/home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: Fatal error
/home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: Fatal error
/home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: Fatal error
/home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: /home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: Fatal error
/home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: /home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: /home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: /home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: /home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: Fatal error
/home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: Fatal error/home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: Fatal errorFatal error/home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: /home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: Fatal error
/home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: Fatal error
Fatal error
Fatal error/home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: Fatal error
/home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: Fatal error
/home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: Fatal error

/home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: Fatal error
/home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: Fatal error

/home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: Fatal error
Fatal error
Fatal error

/home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: 
/home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: /home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: Fatal error
/home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: Fatal error
/home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: Fatal error
Fatal error
/home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: Fatal error
/home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: Fatal error
Fatal error
Fatal error
/home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: Fatal error
/home/mukul/dev-ai/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:700: Fatal error

Fatal error
Could not attach to process.  If your uid matches the uid of the target
Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
Could not attach to process.  If your uid matches the uid of the target
Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
ptrace: Operation not permitted.ptrace: Operation not permitted.
ptrace: Operation not permitted.ptrace: Operation not permitted.


No stack.No stack.

No stack.No stack.

The program is not being run.The program is not being run.

The program is not being run.The program is not being run.

Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
ptrace: Operation not permitted.
No stack.
The program is not being run.
ptrace: Operation not permitted.
No stack.
The program is not being run.
Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
ptrace: Operation not permitted.
No stack.
The program is not being run.
Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
ptrace: Operation not permitted.
No stack.
The program is not being run.
Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
ptrace: Operation not permitted.
No stack.
The program is not being run.
ptrace: Operation not permitted.
No stack.
The program is not being run.
Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
ptrace: Operation not permitted.
No stack.
The program is not being run.
Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
ptrace: Operation not permitted.
No stack.
The program is not being run.
Aborted (core dumped)
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ 
```

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-06-15** at **04:44:17**:<br>

You are 1 commit behind current main branch, and that commit fixes exactly this problem.

---

👤 **mtcl** commented the **2025-06-15** at **05:03:14**:<br>

Alright, pulling latest, building and trying out again :) thank you so much!

---

👤 **mtcl** commented the **2025-06-15** at **05:38:58**:<br>

so i deleted, recloned, rebuilt, it loaded and then crashed when tried to process prompt. Is there a previous version that was stable that I can revert to?

```
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ CUDA_VISIBLE_DEVICES="0,1" ./build/bin/llama-server \
  --model /home/mukul/dev-ai/models/unsloth/Qwen3-235B-A22B-GGUF/Q4_K_M/Qwen3-235B-A22B-Q4_K_M-00001-of-00003.gguf \
  --alias unsloth/Qwen3-235B-A22B-Q4_K_M \
  --ctx-size 4096 \
  -ctk q8_0 -ctv q8_0 \
  -fa \
  -b 4096 -ub 4096 \
  -fmoe \
  --n-gpu-layers 100 \
  -ot "blk\.[0-9]\.ffn=CUDA0,blk\.1[0-4]\.ffn=CUDA0" \
  -ot "blk\.1[5-9]\.ffn=CUDA1,blk\.2[0-9]\.ffn=CUDA1" \
  --override-tensor exps=CPU \
  --parallel 1 \
  --threads 56 \
  --host 0.0.0.0 \
  --port 10002
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
  Device 1: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
INFO [                    main] build info | tid="125154541236224" timestamp=1749965548 build=3749 commit="6fc5bbb6"
INFO [                    main] system info | tid="125154541236224" timestamp=1749965548 n_threads=56 n_threads_batch=-1 total_threads=112 system_info="AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: additional 2 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 45 key-value pairs and 1131 tensors from /home/mukul/dev-ai/models/unsloth/Qwen3-235B-A22B-GGUF/Q4_K_M/Qwen3-235B-A22B-Q4_K_M-00001-of-00003.gguf (version GGUF V3 (latest))
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
llama_model_loader: - kv  33:            tokenizer.ggml.padding_token_id u32              = 151654
llama_model_loader: - kv  34:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  35:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - kv  36:               general.quantization_version u32              = 2
llama_model_loader: - kv  37:                          general.file_type u32              = 15
llama_model_loader: - kv  38:                      quantize.imatrix.file str              = Qwen3-235B-A22B-GGUF/imatrix_unsloth.dat
llama_model_loader: - kv  39:                   quantize.imatrix.dataset str              = unsloth_calibration_Qwen3-235B-A22B.txt
llama_model_loader: - kv  40:             quantize.imatrix.entries_count i32              = 744
llama_model_loader: - kv  41:              quantize.imatrix.chunks_count i32              = 685
llama_model_loader: - kv  42:                                   split.no u16              = 0
llama_model_loader: - kv  43:                        split.tensors.count i32              = 1131
llama_model_loader: - kv  44:                                split.count u16              = 3
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
llm_load_print_meta: BOS token        = 11 ','
llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
llm_load_print_meta: PAD token        = 151654 '<|vision_pad|>'
llm_load_print_meta: LF token         = 148848 'ÄĬ'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp         = 1536
llm_load_tensors: ggml ctx size =    1.49 MiB
Tensor blk.0.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.9.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.9.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.9.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.9.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.9.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.10.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.10.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.10.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.10.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.11.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.11.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.11.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.11.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.11.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.12.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.12.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.13.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.13.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.13.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.13.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.13.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.14.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.14.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.15.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.15.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.15.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.15.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.16.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.16.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.16.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.16.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.16.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.17.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.17.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.17.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.17.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.17.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.18.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.18.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.18.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.18.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.18.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.19.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.19.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.19.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.19.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.19.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.20.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.20.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.20.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.20.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.20.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.21.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.21.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.21.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.21.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.21.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.22.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.22.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.22.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.22.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.22.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.23.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.23.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.23.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.23.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.23.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.24.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.24.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.24.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.24.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.24.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.25.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.25.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.25.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.25.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.25.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.26.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.26.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.26.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.26.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.26.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.27.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.27.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.27.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.27.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.27.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.28.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.28.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.28.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.28.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.28.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.29.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.29.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.29.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.29.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.29.ffn_up_exps.weight buffer type overriden to CUDA1
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
Tensor blk.61.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 94 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 95/95 layers to GPU
llm_load_tensors:        CPU buffer size =  3303.08 MiB
llm_load_tensors:        CPU buffer size = 47617.27 MiB
llm_load_tensors:        CPU buffer size = 40320.40 MiB
llm_load_tensors:        CPU buffer size =   333.84 MiB
llm_load_tensors:      CUDA0 buffer size = 23731.17 MiB
llm_load_tensors:      CUDA1 buffer size = 22811.95 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 4096
llama_new_context_with_model: n_batch    = 4096
llama_new_context_with_model: n_ubatch   = 4096
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:      CUDA0 KV buffer size =   204.02 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =   195.52 MiB
llama_new_context_with_model: KV self size  =  399.50 MiB, K (q8_0):  199.75 MiB, V (q8_0):  199.75 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     1.16 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      CUDA0 compute buffer size =  1732.02 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =  2502.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   128.05 MiB
llama_new_context_with_model: graph nodes  = 3672
llama_new_context_with_model: graph splits = 207
INFO [                    init] initializing slots | tid="125154541236224" timestamp=1749965629 n_slots=1
INFO [                    init] new slot | tid="125154541236224" timestamp=1749965629 id_slot=0 n_ctx_slot=4096
INFO [                    main] model loaded | tid="125154541236224" timestamp=1749965629
INFO [                    main] chat template | tid="125154541236224" timestamp=1749965629 chat_example="<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there<|im_end|>\n<|im_start|>user\nHow are you?<|im_end|>\n<|im_start|>assistant\n" built_in=true
INFO [                    main] HTTP server listening | tid="125154541236224" timestamp=1749965629 n_threads_http="111" port="10002" hostname="0.0.0.0"
INFO [            update_slots] all slots are idle | tid="125154541236224" timestamp=1749965629
INFO [      log_server_request] request | tid="125142946533376" timestamp=1749965675 remote_addr="172.17.0.3" remote_port=48454 status=200 method="GET" path="/v1/models" params={}
INFO [      log_server_request] request | tid="125142864752640" timestamp=1749965676 remote_addr="172.17.0.3" remote_port=48460 status=200 method="GET" path="/v1/models" params={}
INFO [      log_server_request] request | tid="125142856359936" timestamp=1749965681 remote_addr="172.17.0.3" remote_port=48466 status=200 method="GET" path="/v1/models" params={}
INFO [   launch_slot_with_task] slot is processing task | tid="125154541236224" timestamp=1749965681 id_slot=0 id_task=0
INFO [            update_slots] kv cache rm [p0, end) | tid="125154541236224" timestamp=1749965681 id_slot=0 id_task=0 p0=0
INFO [            update_slots] slot context shift | tid="125154541236224" timestamp=1749965731 id_slot=0 id_task=0 n_keep=0 n_left=4095 n_discard=2047 n_ctx=4096 n_past=4095 n_system_tokens=0 n_cache_tokens=4095
/home/mukul/dev-ai/ik_llama.cpp/ggml/src/ggml-cuda/rope.cu:370: GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) failed
Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
ptrace: Operation not permitted.
No stack.
The program is not being run.
Aborted (core dumped)
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ 
```

---

👤 **ikawrakow** commented the **2025-06-15** at **05:43:55**:<br>

You are running with a context of 4096. That's what you wanted, or was it just a type missing a zero?

---

👤 **mtcl** commented the **2025-06-15** at **05:45:44**:<br>

> You are running with a context of 4096. That's what you wanted, or was it just a type missing a zero?

Wow, you know me better than I know myself! It indeed was a typo in a hurry! I wanted to try an easier context instead of 64K and missed a zero!

---

👤 **ikawrakow** commented the **2025-06-15** at **05:47:29**:<br>

So, what happened is that the context became full, it tried to shift it, and that may not work with q8_0 for KV cache.

---

👤 **mtcl** commented the **2025-06-15** at **05:50:44**:<br>

Ah I see, that makes sense. I will close this in that case! Thanks again.