### 🔀 [#260](https://github.com/ikawrakow/ik_llama.cpp/pull/260) - FlashMLA-2: reduce compute buffer size (CUDA and CPU)

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-03-17 |
| **Updated** | 2025-03-18 |

---

#### Description

This PR
* Implements the same compute buffer size reduction approach as PR #253 on CUDA
* Adds the ability to control the compute buffer size for FlashMLA-2  (`-mla 2 -fa`) via the `-amb` command line option. 
* Fixes a bunch of integer overflows that show up when one starts using very long contexts (in the `perplexity` tool, and in the CUDA implementation of `GGML_OP_CONCAT`)

For FlashMLA-2 one computes $X = W_{kv} K$, where $K$ is the K-cache and $W_{kv}$ is the `blk.*.attk_kv_b.weight` tensor. $X$ has the shape `(n_embd_k_nope + n_embd_v) x n_kv x n_head`, where `n_kv` is the number of tokens currently in the cache, `n_head` is the number of heads, and `n_embd_k_nope, n_embd_v` are the head dimensions. For DeepSeekV3/R1/Lite `n_embd_k_nope =  n_embd_v = 128`. As I don't have the ability to run DeepSeekV3/R1, I'm experimenting with DeepSeek-Lite, where `n_head = 16`, so I had not noticed how large $X$ can become (it is "just" 1 GiB for a context of 65k tokens). But `n_head = 128` for DeepSeekV3/R1, so for a context of 65k tokens $X$ becomes 8 GiB ($X$ is computed as `fp32`).  When attention is computed on the GPU the cache is `fp16` (quantized cache still does not work for FlashMLA-2 on CUDA), so $X$ gets converted to `fp16` tensors $V$ and $K_{\rm nope}$, both having half the elements of $X$. As all 3 tensors need to exist simultaneously before the memory used for $X$ can be reused for other data, we end up requiring 16 GiB for these 3 tensors for a context of 65k tokens. This severely limits the maximum context length that can be processed on a GPU with limited VRAM. This PR solves the problem by slitting the attention computation into chunks. The number of chunks used is determined by the size of $X$ and the maximum attention buffer size $B_{\rm max}$ specified on the command-line via the `-amb` option (the argument following `-amb` is maximum buffer size in MiB). We have $N_{\rm step} = {\rm sizeof}(X)/B_{\rm max}$. In each step, $1/N_{\rm srep}$ of the $W_{kv}$ matrix are used, and the entire FlashMLA-2 series of operations is processed with this reduced dataset (effectively using `n_head`/ $N_{\rm step}$ attention heads). The final attention result is obtained by concatenating the results of the individual steps along the head dimension.

For DeepSeek-Lite I need to use a quite low `-amb` threshold of 256 MiB to even trigger the multi-step attention calculation at 65k tokens (attention is computed with 4 steps at 65k tokens, 2 steps at 32k tokens, and 1 step for 16k tokens or less). I observe a 2-3% drop in performance on the CPU and on CUDA for context of 32k tokens computed in 2 steps. I would really appreciate if someone tested this PR with DeepSeekV3/R1 and reported
* Compute buffer size at 16k, 32k, 65k tokens using, e.g., `-mla 2 -fa -amb 1024 -fmoe`
* Performance relative to not using `-amb 1024` (only PP performance is required, TG in FlashMLA-2 is done the same way as no FA, so does not go through this memory optimization).

---

#### 💬 Conversation

👤 **davidsyoung** commented the **2025-03-17** at **15:00:38**:<br>

First model load:

```
./llama-server -m /models/DeepSeek-R1-GGUF/DeepSeek-R1-GGUF-iq4_xs__iq3_s_q8.gguf -amb 1024 -fmoe -mla 2 -fa -ts 24/24/24/24/24/24/24/24/24/24/24/24/24/24/24/24 -c 16384 -ub 1024 --n-gpu-layers 100 -ot "blk\.3\.ffn_(down|gate|up)_exps\.weight|blk\.4\.ffn_(down|gate|up)_exps\.weight|blk\.5\.ffn_(down|gate|up)_exps\.weight=CUDA0" -ot "blk\.6\.ffn_(down|gate|up)_exps\.weight|blk\.7\.ffn_(down|gate|up)_exps\.weight|blk\.8\.ffn_(down|gate|up)_exps\.weight=CUDA1" -ot "blk\.9\.ffn_(down|gate|up)_exps\.weight|blk\.10\.ffn_(down|gate|up)_exps\.weight|blk\.11\.ffn_(down|gate|up)_exps\.weight|blk\.12\.ffn_(down|gate|up)_exps\.weight=CUDA2" -ot "blk\.13\.ffn_(down|gate|up)_exps\.weight|blk\.14\.ffn_(down|gate|up)_exps\.weight|blk\.15\.ffn_(down|gate|up)_exps\.weight|blk\.16\.ffn_(down|gate|up)_exps\.weight=CUDA3" -ot "blk\.17\.ffn_(down|gate|up)_exps\.weight|blk\.18\.ffn_(down|gate|up)_exps\.weight|blk\.19\.ffn_(down|gate|up)_exps\.weight|blk\.20\.ffn_(down|gate|up)_exps\.weight=CUDA4" -ot "blk\.21\.ffn_(down|gate|up)_exps\.weight|blk\.22\.ffn_(down|gate|up)_exps\.weight|blk\.23\.ffn_(down|gate|up)_exps\.weight|blk\.24\.ffn_(down|gate|up)_exps\.weight=CUDA5" -ot "blk\.25\.ffn_(down|gate|up)_exps\.weight|blk\.26\.ffn_(down|gate|up)_exps\.weight|blk\.27\.ffn_(down|gate|up)_exps\.weight|blk\.28\.ffn_(down|gate|up)_exps\.weight=CUDA6" -ot "blk\.29\.ffn_(down|gate|up)_exps\.weight|blk\.30\.ffn_(down|gate|up)_exps\.weight|blk\.31\.ffn_(down|gate|up)_exps\.weight|blk\.32\.ffn_(down|gate|up)_exps\.weight=CUDA7" -ot "blk\.33\.ffn_(down|gate|up)_exps\.weight|blk\.34\.ffn_(down|gate|up)_exps\.weight|blk\.35\.ffn_(down|gate|up)_exps\.weight|blk\.36\.ffn_(down|gate|up)_exps\.weight=CUDA8" -ot "blk\.37\.ffn_(down|gate|up)_exps\.weight|blk\.38\.ffn_(down|gate|up)_exps\.weight|blk\.39\.ffn_(down|gate|up)_exps\.weight|blk\.40\.ffn_(down|gate|up)_exps\.weight=CUDA9" -ot "blk\.41\.ffn_(down|gate|up)_exps\.weight|blk\.42\.ffn_(down|gate|up)_exps\.weight|blk\.43\.ffn_(down|gate|up)_exps\.weight|blk\.44\.ffn_(down|gate|up)_exps\.weight=CUDA10" -ot "blk\.45\.ffn_(down|gate|up)_exps\.weight|blk\.46\.ffn_(down|gate|up)_exps\.weight|blk\.47\.ffn_(down|gate|up)_exps\.weight|blk\.48\.ffn_(down|gate|up)_exps\.weight=CUDA11" -ot "blk\.49\.ffn_(down|gate|up)_exps\.weight|blk\.50\.ffn_(down|gate|up)_exps\.weight|blk\.51\.ffn_(down|gate|up)_exps\.weight|blk\.52\.ffn_(down|gate|up)_exps\.weight=CUDA12" -ot "blk\.53\.ffn_(down|gate|up)_exps\.weight|blk\.54\.ffn_(down|gate|up)_exps\.weight|blk\.55\.ffn_(down|gate|up)_exps\.weight|blk\.56\.ffn_(down|gate|up)_exps\.weight=CUDA13" -ot "blk\.57\.ffn_(down|gate|up)_exps\.weight|blk\.58\.ffn_(down|gate|up)_exps\.weight|blk\.59\.ffn_(down|gate|up)_exps\.weight|blk\.60\.ffn_(down|gate|up)_exps\.weight=CUDA14" --seed 3704 --temp 0.5 --temp 0.5 --host 0.0.0.0 --port 8080
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 16 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 1: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 2: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 3: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 4: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 5: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 6: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 7: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 8: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 9: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 10: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 11: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 12: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 13: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 14: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 15: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
INFO [                    main] build info | tid="22757404872704" timestamp=1742222860 build=0 commit="unknown"
INFO [                    main] system info | tid="22757404872704" timestamp=1742222860 n_threads=64 n_threads_batch=-1 total_threads=128 system_info="AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: loaded meta data with 54 key-value pairs and 1147 tensors from /models/DeepSeek-R1-GGUF/DeepSeek-R1-GGUF-iq4_xs__iq3_s_q8.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = unsloth_DeepSeek R1 BF16
llama_model_loader: - kv   3:                         general.size_label str              = 256x21B
llama_model_loader: - kv   4:                            general.license str              = mit
llama_model_loader: - kv   5:                   general.base_model.count u32              = 1
llama_model_loader: - kv   6:                  general.base_model.0.name str              = DeepSeek R1
llama_model_loader: - kv   7:          general.base_model.0.organization str              = Deepseek Ai
llama_model_loader: - kv   8:              general.base_model.0.repo_url str              = https://huggingface.co/deepseek-ai/De...
llama_model_loader: - kv   9:                               general.tags arr[str,3]       = ["deepseek", "unsloth", "transformers"]
llama_model_loader: - kv  10:                          general.languages arr[str,1]       = ["en"]
llama_model_loader: - kv  11:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv  12:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv  13:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv  14:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  15:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv  16:          deepseek2.attention.head_count_kv u32              = 128
llama_model_loader: - kv  17:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  18: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  19:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  20:                          general.file_type u32              = 7
llama_model_loader: - kv  21:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  22:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  23:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  24:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  25:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  26:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  27:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  28:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  29:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  30:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  31:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  32:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  33:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  34:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  35:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  36: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  37: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  38:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  39:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  40:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<?...
llama_model_loader: - kv  41:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  42:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
llama_model_loader: - kv  43:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  44:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  45:            tokenizer.ggml.padding_token_id u32              = 128815
llama_model_loader: - kv  46:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  47:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  48:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  49:               general.quantization_version u32              = 2
llama_model_loader: - kv  50:                      quantize.imatrix.file str              = /models/deepseek-config/imatrix.dat
llama_model_loader: - kv  51:                   quantize.imatrix.dataset str              = imatrix-training-full-3
llama_model_loader: - kv  52:             quantize.imatrix.entries_count i32              = 720
llama_model_loader: - kv  53:              quantize.imatrix.chunks_count i32              = 315
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  438 tensors
llama_model_loader: - type q5_K:  180 tensors
llama_model_loader: - type iq3_s:  104 tensors
llama_model_loader: - type iq4_xs:   64 tensors
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
llm_load_print_meta: model ftype      = Q8_0
llm_load_print_meta: model params     = 672.050 B
llm_load_print_meta: model size       = 307.197 GiB (3.926 BPW) 
llm_load_print_meta: repeating layers = 305.363 GiB (3.914 BPW, 670.196 B parameters)
llm_load_print_meta: general.name     = unsloth_DeepSeek R1 BF16
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
llm_load_tensors: ggml ctx size =    7.94 MiB
Tensor blk.3.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.6.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.6.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.7.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.7.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.9.ffn_gate_exps.weight buffer type overriden to CUDA2
Tensor blk.9.ffn_down_exps.weight buffer type overriden to CUDA2
Tensor blk.9.ffn_up_exps.weight buffer type overriden to CUDA2
Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CUDA2
Tensor blk.10.ffn_down_exps.weight buffer type overriden to CUDA2
Tensor blk.10.ffn_up_exps.weight buffer type overriden to CUDA2
Tensor blk.11.ffn_gate_exps.weight buffer type overriden to CUDA2
Tensor blk.11.ffn_down_exps.weight buffer type overriden to CUDA2
Tensor blk.11.ffn_up_exps.weight buffer type overriden to CUDA2
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CUDA2
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CUDA2
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CUDA2
Tensor blk.13.ffn_gate_exps.weight buffer type overriden to CUDA3
Tensor blk.13.ffn_down_exps.weight buffer type overriden to CUDA3
Tensor blk.13.ffn_up_exps.weight buffer type overriden to CUDA3
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CUDA3
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CUDA3
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CUDA3
Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CUDA3
Tensor blk.15.ffn_down_exps.weight buffer type overriden to CUDA3
Tensor blk.15.ffn_up_exps.weight buffer type overriden to CUDA3
Tensor blk.16.ffn_gate_exps.weight buffer type overriden to CUDA3
Tensor blk.16.ffn_down_exps.weight buffer type overriden to CUDA3
Tensor blk.16.ffn_up_exps.weight buffer type overriden to CUDA3
Tensor blk.17.ffn_gate_exps.weight buffer type overriden to CUDA4
Tensor blk.17.ffn_down_exps.weight buffer type overriden to CUDA4
Tensor blk.17.ffn_up_exps.weight buffer type overriden to CUDA4
Tensor blk.18.ffn_gate_exps.weight buffer type overriden to CUDA4
Tensor blk.18.ffn_down_exps.weight buffer type overriden to CUDA4
Tensor blk.18.ffn_up_exps.weight buffer type overriden to CUDA4
Tensor blk.19.ffn_gate_exps.weight buffer type overriden to CUDA4
Tensor blk.19.ffn_down_exps.weight buffer type overriden to CUDA4
Tensor blk.19.ffn_up_exps.weight buffer type overriden to CUDA4
Tensor blk.20.ffn_gate_exps.weight buffer type overriden to CUDA4
Tensor blk.20.ffn_down_exps.weight buffer type overriden to CUDA4
Tensor blk.20.ffn_up_exps.weight buffer type overriden to CUDA4
Tensor blk.21.ffn_gate_exps.weight buffer type overriden to CUDA5
Tensor blk.21.ffn_down_exps.weight buffer type overriden to CUDA5
Tensor blk.21.ffn_up_exps.weight buffer type overriden to CUDA5
Tensor blk.22.ffn_gate_exps.weight buffer type overriden to CUDA5
Tensor blk.22.ffn_down_exps.weight buffer type overriden to CUDA5
Tensor blk.22.ffn_up_exps.weight buffer type overriden to CUDA5
Tensor blk.23.ffn_gate_exps.weight buffer type overriden to CUDA5
Tensor blk.23.ffn_down_exps.weight buffer type overriden to CUDA5
Tensor blk.23.ffn_up_exps.weight buffer type overriden to CUDA5
Tensor blk.24.ffn_gate_exps.weight buffer type overriden to CUDA5
Tensor blk.24.ffn_down_exps.weight buffer type overriden to CUDA5
Tensor blk.24.ffn_up_exps.weight buffer type overriden to CUDA5
Tensor blk.25.ffn_gate_exps.weight buffer type overriden to CUDA6
Tensor blk.25.ffn_down_exps.weight buffer type overriden to CUDA6
Tensor blk.25.ffn_up_exps.weight buffer type overriden to CUDA6
Tensor blk.26.ffn_gate_exps.weight buffer type overriden to CUDA6
Tensor blk.26.ffn_down_exps.weight buffer type overriden to CUDA6
Tensor blk.26.ffn_up_exps.weight buffer type overriden to CUDA6
Tensor blk.27.ffn_gate_exps.weight buffer type overriden to CUDA6
Tensor blk.27.ffn_down_exps.weight buffer type overriden to CUDA6
Tensor blk.27.ffn_up_exps.weight buffer type overriden to CUDA6
Tensor blk.28.ffn_gate_exps.weight buffer type overriden to CUDA6
Tensor blk.28.ffn_down_exps.weight buffer type overriden to CUDA6
Tensor blk.28.ffn_up_exps.weight buffer type overriden to CUDA6
Tensor blk.29.ffn_gate_exps.weight buffer type overriden to CUDA7
Tensor blk.29.ffn_down_exps.weight buffer type overriden to CUDA7
Tensor blk.29.ffn_up_exps.weight buffer type overriden to CUDA7
Tensor blk.30.ffn_gate_exps.weight buffer type overriden to CUDA7
Tensor blk.30.ffn_down_exps.weight buffer type overriden to CUDA7
Tensor blk.30.ffn_up_exps.weight buffer type overriden to CUDA7
Tensor blk.31.ffn_gate_exps.weight buffer type overriden to CUDA7
Tensor blk.31.ffn_down_exps.weight buffer type overriden to CUDA7
Tensor blk.31.ffn_up_exps.weight buffer type overriden to CUDA7
Tensor blk.32.ffn_gate_exps.weight buffer type overriden to CUDA7
Tensor blk.32.ffn_down_exps.weight buffer type overriden to CUDA7
Tensor blk.32.ffn_up_exps.weight buffer type overriden to CUDA7
Tensor blk.33.ffn_gate_exps.weight buffer type overriden to CUDA8
Tensor blk.33.ffn_down_exps.weight buffer type overriden to CUDA8
Tensor blk.33.ffn_up_exps.weight buffer type overriden to CUDA8
Tensor blk.34.ffn_gate_exps.weight buffer type overriden to CUDA8
Tensor blk.34.ffn_down_exps.weight buffer type overriden to CUDA8
Tensor blk.34.ffn_up_exps.weight buffer type overriden to CUDA8
Tensor blk.35.ffn_gate_exps.weight buffer type overriden to CUDA8
Tensor blk.35.ffn_down_exps.weight buffer type overriden to CUDA8
Tensor blk.35.ffn_up_exps.weight buffer type overriden to CUDA8
Tensor blk.36.ffn_gate_exps.weight buffer type overriden to CUDA8
Tensor blk.36.ffn_down_exps.weight buffer type overriden to CUDA8
Tensor blk.36.ffn_up_exps.weight buffer type overriden to CUDA8
Tensor blk.37.ffn_gate_exps.weight buffer type overriden to CUDA9
Tensor blk.37.ffn_down_exps.weight buffer type overriden to CUDA9
Tensor blk.37.ffn_up_exps.weight buffer type overriden to CUDA9
Tensor blk.38.ffn_gate_exps.weight buffer type overriden to CUDA9
Tensor blk.38.ffn_down_exps.weight buffer type overriden to CUDA9
Tensor blk.38.ffn_up_exps.weight buffer type overriden to CUDA9
Tensor blk.39.ffn_gate_exps.weight buffer type overriden to CUDA9
Tensor blk.39.ffn_down_exps.weight buffer type overriden to CUDA9
Tensor blk.39.ffn_up_exps.weight buffer type overriden to CUDA9
Tensor blk.40.ffn_gate_exps.weight buffer type overriden to CUDA9
Tensor blk.40.ffn_down_exps.weight buffer type overriden to CUDA9
Tensor blk.40.ffn_up_exps.weight buffer type overriden to CUDA9
Tensor blk.41.ffn_gate_exps.weight buffer type overriden to CUDA10
Tensor blk.41.ffn_down_exps.weight buffer type overriden to CUDA10
Tensor blk.41.ffn_up_exps.weight buffer type overriden to CUDA10
Tensor blk.42.ffn_gate_exps.weight buffer type overriden to CUDA10
Tensor blk.42.ffn_down_exps.weight buffer type overriden to CUDA10
Tensor blk.42.ffn_up_exps.weight buffer type overriden to CUDA10
Tensor blk.43.ffn_gate_exps.weight buffer type overriden to CUDA10
Tensor blk.43.ffn_down_exps.weight buffer type overriden to CUDA10
Tensor blk.43.ffn_up_exps.weight buffer type overriden to CUDA10
Tensor blk.44.ffn_gate_exps.weight buffer type overriden to CUDA10
Tensor blk.44.ffn_down_exps.weight buffer type overriden to CUDA10
Tensor blk.44.ffn_up_exps.weight buffer type overriden to CUDA10
Tensor blk.45.ffn_gate_exps.weight buffer type overriden to CUDA11
Tensor blk.45.ffn_down_exps.weight buffer type overriden to CUDA11
Tensor blk.45.ffn_up_exps.weight buffer type overriden to CUDA11
Tensor blk.46.ffn_gate_exps.weight buffer type overriden to CUDA11
Tensor blk.46.ffn_down_exps.weight buffer type overriden to CUDA11
Tensor blk.46.ffn_up_exps.weight buffer type overriden to CUDA11
Tensor blk.47.ffn_gate_exps.weight buffer type overriden to CUDA11
Tensor blk.47.ffn_down_exps.weight buffer type overriden to CUDA11
Tensor blk.47.ffn_up_exps.weight buffer type overriden to CUDA11
Tensor blk.48.ffn_gate_exps.weight buffer type overriden to CUDA11
Tensor blk.48.ffn_down_exps.weight buffer type overriden to CUDA11
Tensor blk.48.ffn_up_exps.weight buffer type overriden to CUDA11
Tensor blk.49.ffn_gate_exps.weight buffer type overriden to CUDA12
Tensor blk.49.ffn_down_exps.weight buffer type overriden to CUDA12
Tensor blk.49.ffn_up_exps.weight buffer type overriden to CUDA12
Tensor blk.50.ffn_gate_exps.weight buffer type overriden to CUDA12
Tensor blk.50.ffn_down_exps.weight buffer type overriden to CUDA12
Tensor blk.50.ffn_up_exps.weight buffer type overriden to CUDA12
Tensor blk.51.ffn_gate_exps.weight buffer type overriden to CUDA12
Tensor blk.51.ffn_down_exps.weight buffer type overriden to CUDA12
Tensor blk.51.ffn_up_exps.weight buffer type overriden to CUDA12
Tensor blk.52.ffn_gate_exps.weight buffer type overriden to CUDA12
Tensor blk.52.ffn_down_exps.weight buffer type overriden to CUDA12
Tensor blk.52.ffn_up_exps.weight buffer type overriden to CUDA12
Tensor blk.53.ffn_gate_exps.weight buffer type overriden to CUDA13
Tensor blk.53.ffn_down_exps.weight buffer type overriden to CUDA13
Tensor blk.53.ffn_up_exps.weight buffer type overriden to CUDA13
Tensor blk.54.ffn_gate_exps.weight buffer type overriden to CUDA13
Tensor blk.54.ffn_down_exps.weight buffer type overriden to CUDA13
Tensor blk.54.ffn_up_exps.weight buffer type overriden to CUDA13
Tensor blk.55.ffn_gate_exps.weight buffer type overriden to CUDA13
Tensor blk.55.ffn_down_exps.weight buffer type overriden to CUDA13
Tensor blk.55.ffn_up_exps.weight buffer type overriden to CUDA13
Tensor blk.56.ffn_gate_exps.weight buffer type overriden to CUDA13
Tensor blk.56.ffn_down_exps.weight buffer type overriden to CUDA13
Tensor blk.56.ffn_up_exps.weight buffer type overriden to CUDA13
Tensor blk.57.ffn_gate_exps.weight buffer type overriden to CUDA14
Tensor blk.57.ffn_down_exps.weight buffer type overriden to CUDA14
Tensor blk.57.ffn_up_exps.weight buffer type overriden to CUDA14
Tensor blk.58.ffn_gate_exps.weight buffer type overriden to CUDA14
Tensor blk.58.ffn_down_exps.weight buffer type overriden to CUDA14
Tensor blk.58.ffn_up_exps.weight buffer type overriden to CUDA14
Tensor blk.59.ffn_gate_exps.weight buffer type overriden to CUDA14
Tensor blk.59.ffn_down_exps.weight buffer type overriden to CUDA14
Tensor blk.59.ffn_up_exps.weight buffer type overriden to CUDA14
Tensor blk.60.ffn_gate_exps.weight buffer type overriden to CUDA14
Tensor blk.60.ffn_down_exps.weight buffer type overriden to CUDA14
Tensor blk.60.ffn_up_exps.weight buffer type overriden to CUDA14
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size =   938.98 MiB
llm_load_tensors:      CUDA0 buffer size = 20883.36 MiB
llm_load_tensors:      CUDA1 buffer size = 19786.12 MiB
llm_load_tensors:      CUDA2 buffer size = 20906.12 MiB
llm_load_tensors:      CUDA3 buffer size = 20906.12 MiB
llm_load_tensors:      CUDA4 buffer size = 20906.12 MiB
llm_load_tensors:      CUDA5 buffer size = 20906.12 MiB
llm_load_tensors:      CUDA6 buffer size = 20906.12 MiB
llm_load_tensors:      CUDA7 buffer size = 20663.59 MiB
llm_load_tensors:      CUDA8 buffer size = 20906.12 MiB
llm_load_tensors:      CUDA9 buffer size = 20906.12 MiB
llm_load_tensors:     CUDA10 buffer size = 20906.12 MiB
llm_load_tensors:     CUDA11 buffer size = 20906.12 MiB
llm_load_tensors:     CUDA12 buffer size = 20906.12 MiB
llm_load_tensors:     CUDA13 buffer size = 20906.12 MiB
llm_load_tensors:     CUDA14 buffer size = 20906.12 MiB
llm_load_tensors:     CUDA15 buffer size =  1424.07 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 16384
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 1024
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 2
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
llama_kv_cache_init:      CUDA0 KV buffer size =    72.00 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =    72.00 MiB
llama_kv_cache_init:      CUDA2 KV buffer size =    72.00 MiB
llama_kv_cache_init:      CUDA3 KV buffer size =    72.00 MiB
llama_kv_cache_init:      CUDA4 KV buffer size =    72.00 MiB
llama_kv_cache_init:      CUDA5 KV buffer size =    72.00 MiB
llama_kv_cache_init:      CUDA6 KV buffer size =    72.00 MiB
llama_kv_cache_init:      CUDA7 KV buffer size =    54.00 MiB
llama_kv_cache_init:      CUDA8 KV buffer size =    72.00 MiB
llama_kv_cache_init:      CUDA9 KV buffer size =    72.00 MiB
llama_kv_cache_init:     CUDA10 KV buffer size =    72.00 MiB
llama_kv_cache_init:     CUDA11 KV buffer size =    72.00 MiB
llama_kv_cache_init:     CUDA12 KV buffer size =    72.00 MiB
llama_kv_cache_init:     CUDA13 KV buffer size =    72.00 MiB
llama_kv_cache_init:     CUDA14 KV buffer size =    72.00 MiB
llama_kv_cache_init:     CUDA15 KV buffer size =    36.00 MiB
llama_new_context_with_model: KV self size  = 1098.00 MiB, c^KV (f16): 1098.00 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.99 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=4)
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 3480.01 MiB on device 0: cudaMalloc failed: out of memory
ggml_gallocr_reserve_n: failed to allocate CUDA0 buffer of size 3649053696
llama_new_context_with_model: failed to allocate compute buffers
llama_init_from_gpt_params: error: failed to create context with model '/models/DeepSeek-R1-GGUF/DeepSeek-R1-GGUF-iq4_xs__iq3_s_q8.gguf'
 ERR [              load_model] unable to load model | tid="22757404872704" timestamp=1742223553 model="/models/DeepSeek-R1-GGUF/DeepSeek-R1-GGUF-iq4_xs__iq3_s_q8.gguf"
Segmentation fault
root@7e406a084738:/app/build/bin# 
```

---

👤 **ikawrakow** commented the **2025-03-17** at **15:06:08**:<br>

> Those fixes for perplexity, do you believe that was related to NaN's in IX_K quants?

No. It is an integer overflow. The logic location in the array of logits was computed with 32-bit integers. As there are ~128k entries in the vocabulary, the integer multiplication `i * n_vocab` overflows for `i >= 16384`. You were computing PPL for contexts of 2048 or 512, so no issue there (`i < 2048`). The NaNs really are due to `fp16` arithmetic for the MoE matrix multiplications when using `IQ4_K` or `IQ4_KSS`. Apparently in the `llama.cpp` world it is well known that one cannot use the `fp16` DeepSeek models because one gets NaNs.

---

👤 **ikawrakow** commented the **2025-03-17** at **15:10:29**:<br>

> Segfault with `-c 16384 -amb 1024 -fmoe -mla 2 -fa`

It fails to allocate `3480 MiB`, so I guess there isn't enough VRAM? Try with `-amb 512` then.

---

👤 **ubergarm** commented the **2025-03-17** at **15:40:37**:<br>

I'll take a quick stab at it too given using a simple 1x RTX A6000 48GB GPU configuration.


#### Update
```bash
$ git checkout ik/flash_mla2_cuda_no_f32
$ git rev-parse --short HEAD
b147e31f
$ cmake -B ./build -DGGML_CUDA=ON -DGGML_BLAS=OFF
$ cmake --build ./build --config Release -j $(nproc)
$ ./build/bin/llama-server --version
version: 3601 (b147e31f)
built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu
```

> Compute buffer size at 16k, 32k, 65k tokens using, e.g., -mla 2 -fa -amb 1024 -fmoe

#### Basic Command
```bash
CUDA_VISIBLE_DEVICES="0," \
./build/bin/llama-server \
    --alias ubergarm/DeepSeek-R1-Q2_K_R4 \
    --model /mnt/raid/models/ubergarm/DeepSeek-R1-GGUF/DeepSeek-R1-GGUF-Q2_K_R4.gguf \
    --ctx-size 16384 \
    -ctk f16 -ctv f16 \
    -mla 2 -fa \
    -amb 1024 \
    -fmoe \
    --n-gpu-layers 63 \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 24 \
    --host 127.0.0.1 \
    --port 8080
```

#### Results

* 16k = TODO
* 32k = TODO
* 64k = TODO

> Performance relative to not using -amb 1024 (only PP performance is required, TG in FlashMLA-2 is done the same way as no FA, so does not go through this memory optimization).

#### llama-bench
```bash
echo TODO
```

---

👤 **ikawrakow** commented the **2025-03-17** at **16:04:33**:<br>

So, this looks quite a bit better than the main branch. It would seem that a single 24 GB GPU could handle the non-expert tensors and up to 32k context?

---

👤 **ikawrakow** commented the **2025-03-17** at **16:33:42**:<br>

> Oh, I thought I noticed it was reporting less with -ctk q8_0 -ctv q8_0, I'll do a quick check and update this TODO here and confirm.

I haven't put a guard against using quantized cache for `mla = 2`, so it will happily initialize (and report buffer sizes), but then it will terminate when it arrives at the op that is not supported on CUDA for quantized data.

Based on the performance values @ubergarm posted, there doesn't seem to be any major performance impact, even with `-amb 128`. What is the compute buffer size for `-amb 128`?

---

👤 **ubergarm** commented the **2025-03-17** at **16:47:48**:<br>

> What is the compute buffer size for -amb 128

The relevant part of the above table for this specific question:

| commit | ctx-size | amb | ctk/ctv | CUDA0 KV buffer | CUDA0 compute buffer | nvidia-smi |
| --- | --- | --- | --- | --- | --- | --- |
| branch/sha | size | quant | MiB | MiB | MiB |
| `flash_mla2_@b147e31f` | 32768 | 1024 | f16 | 2196 | 3790 | 24010 |
| `flash_mla2_@b147e31f` | 32768 |  128 | f16 | 2196 | 2817 | 23036 |

---

👤 **davidsyoung** commented the **2025-03-17** at **16:59:59**:<br>

Sorry for delay here. As model loading takes quite a long amount of time on 16 GPUs, and I'm near to the limit there's been some OOMs (my own fault nothing to do with PR), I've been quite slow to come back.

From what I can see so far, there is zero notable difference with performance of `-amb 256` vs `-amb 512`. I would imagine that this will continue to a point. I'll test lower and create a comparison here when done.

TODO

---

👤 **ubergarm** commented the **2025-03-17** at **17:25:23**:<br>

@davidsyoung 

>  I would imagine that this will continue to a point. I'll test lower and create a comparison here when done.

Yeah, please double check me, but I updated my chart and command above which suggests going down to `-amb 2` is about the limit with `-amb 1` having only slightly slower pp!????

Curious if you have similar outcome across all your GPUs!

---

👤 **davidsyoung** commented the **2025-03-17** at **17:39:21**:<br>

> @davidsyoung
> 
> > I would imagine that this will continue to a point. I'll test lower and create a comparison here when done.
> 
> Yeah, please double check me, but I updated my chart and command above which suggests going down to `-amb 2` is about the limit with `-amb 1` having only slightly slower pp!????
> 
> Curious if you have similar outcome across all your GPUs!

Interestingly I got an error for `-amb 32` when trying to maximise context length:
```
llama_new_context_with_model: KV self size  = 1647.00 MiB, c^KV (f16): 1647.00 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.99 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=4)
ggml_new_object: not enough space in the context's memory pool (needed 26231472, available 26231136)
Segmentation fault
```

Haven't seen that error before!

---

👤 **ikawrakow** commented the **2025-03-17** at **17:55:57**:<br>

Sorry, I wasn't clear enough with my request. The PP test should be done with `-p 16384` (or whatever context we are looking at). With `-p 512`, `llama-bench` will set the context to 512, so the required buffer to compute FlashMLA-2 will be quite small - `256 x 128 x 512 x 4 = 64 MiB`, so there will be more than one step only for `-amb 32` or lower. With `-amb 2` it will take 32 steps, so it will be processing 4 heads at a time. At `-amb 1` it will be 64 steps, so 2 heads per step. I find it quite surprising that we do not see performance degradation down to so many steps. 

> Also, you should test setting -ub 1024, you should see a big difference in PP performance compared to default of -ub 512 I believe.

This is only relevant of the MoE experts are computed on CUDA. When the MoE part runs on the CPU the default `u_batch` size of 512 tends to give the best PP performance.

---

👤 **ubergarm** commented the **2025-03-17** at **18:00:55**:<br>

@davidsyoung 

> pipeline parallelism enabled (n_copies=4)
Hrmm, I've seen some chatter about `-DGGML_SCHED_MAX_COPIES=4` before (default). Some folks were setting it to 1. Not sure why (maybe CUDA graphs?) and that was on vanilla llama.cpp so may not apply anymore.

I was kinda surprised that you were offloading shared experts onto GPUs with your config given that doesn't work on ktransformers yet in my own testing an in their documentation:

> Note：Currently, executing experts on the GPU will conflict with CUDA Graph. Without CUDA Graph, there will be a significant slowdown. Therefore, unless you have a substantial amount of VRAM (placing a single layer of experts for DeepSeek-V3/R1 on the GPU requires at least 5.6GB of VRAM), we do not recommend enabling this feature. We are actively working on optimization. Note KExpertsTorch is untested.

@ikawrakow 

> The PP test should be done with -p 16384

I'll set that up and post the results here soon.

---

👤 **ikawrakow** commented the **2025-03-17** at **18:09:14**:<br>

> I was kinda surprised that you were offloading shared experts onto GPUs with your config given that doesn't work on ktransformers yet in my own testing an in their documentation:

@davidsyoung has 16 x 3090's, so the entire model is run on the GPU's. CUDA graphs get disabled for MoE models (also true on mainline `llama.cpp`). Disabled CUDA graphs leading to a significant hit in performance is a myth. There is no effect for PP, and at most a few percent (< 5%, IIRC) for TG. The 16 x 3090 configuration gives ~350 t/s for PP and ~17 t/s for TG.

---

👤 **ikawrakow** commented the **2025-03-17** at **18:25:34**:<br>

> Interestingly I got an error for -amb 32 when trying to maximise context length:
> Haven't seen that error before!

Neither have I. It means that the back-end is miscalculating the required compute buffer size somehow. Not sure what to do about that.

---

👤 **ubergarm** commented the **2025-03-17** at **19:29:15**:<br>

I increased `-p 16384` and set `-r 2` repetitions down from default of 5 for a quick check but it crashed before finishing with error shown below.

```bash
CUDA_VISIBLE_DEVICES="0," \
./build/bin/llama-bench \
    --model /mnt/raid/models/ubergarm/DeepSeek-R1-GGUF/DeepSeek-R1-GGUF-Q2_K_R4.gguf \
    -ctk f16 -ctv f16 \
    -mla 2 -fa 1 \
    -amb 1024,128,16,8,4,2,1 \
    -p 16384,8192 \
    -n 0 \
    -fmoe 1 \
    -r 2 \
    --n-gpu-layers 63 \
    --override-tensor exps=CPU \
    --threads 24

ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA RTX A6000, compute capability 8.6, VMM: yes
```

| model                          |       size |     params | backend    | ngl | fa | mla |   amb | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | --: | ----: | ---: | ------------: | ---------------: |
| deepseek2 671B Q2_K_R4         | 238.69 GiB |   672.05 B | CUDA       |  63 |  1 |   2 |  1024 |    1 |       pp16384 |     84.16 ± 2.14 |
| deepseek2 671B Q2_K_R4         | 238.69 GiB |   672.05 B | CUDA       |  63 |  1 |   2 |  1024 |    1 |        pp8192 |     97.67 ± 1.21 |
| deepseek2 671B Q2_K_R4         | 238.69 GiB |   672.05 B | CUDA       |  63 |  1 |   2 |   128 |    1 |       pp16384 |     82.59 ± 2.70 |
| deepseek2 671B Q2_K_R4         | 238.69 GiB |   672.05 B | CUDA       |  63 |  1 |   2 |   128 |    1 |        pp8192 |     96.21 ± 1.67 |

```
ggml_new_object: not enough space in the context's memory pool (needed 26231472, available 26231136)
./myscripts/benchmark.sh: line 24: 2286044 Segmentation fault      (core dumped) CUDA_VISIBLE_DEVICES="0," ./build/bin/llama-bench --model /mnt/raid/models/ubergarm/DeepSeek-R1-GGUF/DeepSeek-R1-GGUF-Q2_K_R4.gguf -ctk f16 -ctv f16 -mla 2 -fa 1 -amb 1024,128,16,8,4,2,1 -p 16384,8192 -n 0 -fmoe 1 -r 2 --n-gpu-layers 63 --override-tensor exps=CPU --threads 24
```

---

👤 **davidsyoung** commented the **2025-03-17** at **23:37:51**:<br>

So compute buffers are massively improved. I don't have apples for apples comparison as I went down a rabbit hole after realising I could turn off pipeline parallel and it would also give me more VRAM back (thanks @ubergarm!). But it is massively improved.

Had some issues going below `-amb 32` as well with bench, but got some data:

| model                          |       size |     params | backend    | ngl | n_ubatch | fa | mla |   amb | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -: | --: | ----: | ---: | ------------: | ---------------: |
| deepseek2 671B Q8_0            | 307.20 GiB |   672.05 B | CUDA       |  63 |     2048 |  1 |   2 |   512 |    1 |       pp16384 |    235.62 ± 3.94 |
| deepseek2 671B Q8_0            | 307.20 GiB |   672.05 B | CUDA       |  63 |     2048 |  1 |   2 |   512 |    1 |        pp8192 |    293.09 ± 0.42 |
| deepseek2 671B Q8_0            | 307.20 GiB |   672.05 B | CUDA       |  63 |     2048 |  1 |   2 |   128 |    1 |       pp16384 |    231.66 ± 0.22 |
| deepseek2 671B Q8_0            | 307.20 GiB |   672.05 B | CUDA       |  63 |     2048 |  1 |   2 |   128 |    1 |        pp8192 |    289.73 ± 0.71 |
| deepseek2 671B Q8_0            | 307.20 GiB |   672.05 B | CUDA       |  63 |     2048 |  1 |   2 |    64 |    1 |       pp16384 |    224.48 ± 0.07 |
| deepseek2 671B Q8_0            | 307.20 GiB |   672.05 B | CUDA       |  63 |     2048 |  1 |   2 |    64 |    1 |        pp8192 |    283.72 ± 0.43 |
| deepseek2 671B Q8_0            | 307.20 GiB |   672.05 B | CUDA       |  63 |     2048 |  1 |   2 |    32 |    1 |       pp16384 |    215.12 ± 0.05 |
| deepseek2 671B Q8_0            | 307.20 GiB |   672.05 B | CUDA       |  63 |     2048 |  1 |   2 |    32 |    1 |        pp8192 |    274.51 ± 0.29 |

Same error:
```
ggml_new_object: not enough space in the context's memory pool (needed 26231472, available 26231136)
Segmentation fault
```

---

👤 **saood06** commented the **2025-03-17** at **23:48:58**:<br>

> I don't have apples for apples comparison as I went down a rabbit hole after realising I could turn off pipeline parallel and it would also give me more VRAM back (thanks @ubergarm!). But it is massively improved.

Even without the direct comparison, I'm curious what your at now. Also you probably have fixed it by now but CUDA15 was very unused in here:

> First model load:
>[...]
>CUDA15 buffer size =  1424.07 MiB

---

👤 **davidsyoung** commented the **2025-03-18** at **00:00:30**:<br>

Damn, I don’t have it right on me as I closed the laptop (night time here). I do have some data in notes from very early run. 

I was able to get to 24k context, with `-ub 2048`. I believe I could get to 32k, but I was getting some errors when playing with `-amb` lower than 32. 

Here are some very initial runs (this is without disabling pipeline parallelism). This is already quite improved from what I can remember. 

Also, for gpu 16, unfortunately I can’t really use it. I can’t split the layers any bit more evenly (at least with what I’ve tried - it’s a bit of a limitation unfortunately without being able to split by row).

# Compute Buffer Configuration Comparison

| Parameter/Variable          | Run 1 (`-c 8192 -amb 512`) | Run 2 (`-c 16843 -amb 256`) | Notes/Observations                                                                 |
|-----------------------------|----------------------------|-----------------------------|-----------------------------------------------------------------------------------|
| **Context Size (`-c`)**      | 8,192                      | 16,843                      | Context doubled (+106%), directly impacts KV cache size.                          |
| **Attention Mask Buffer (`-amb`)** | 512                        | 256                         | Reduced by 50%, but total compute buffer still increased.                         |
| **Total Compute Buffer**     | 31,178.17 MiB              | 38,430 MiB                  | +23% total memory usage despite smaller `-amb`, driven by larger context.         |
| **KV Self Size**             | 549.00 MiB                 | 1,098.00 MiB                | Doubled due to larger context (KV cache scales with sequence length).             |
| **CUDA_Host Compute Buffer** | 156.05 MiB                 | 284.05 MiB                  | +82% increase, likely due to larger context requiring more host-device transfers. |
| **Pipeline Copies (`n_copies`)** | 4                          | 4                           | Pipeline parallelism unchanged.                                                   |

---

### Example Device Buffer Changes (MiB):
| Device   | Run 1    | Run 2    | Change   |
|----------|----------|----------|----------|
| CUDA0    | 1,974.01 | 2,342.01 | +19%     |
| CUDA8    | 2,196.01 | 2,516.01 | +15%     |
| CUDA15   | 1,936.03 | 2,256.03 | +16%     |

---

### Key Findings:
1. **Context Size Dominates Memory**: Doubling the context size led to a 23% increase in total compute buffer usage despite halving `-amb`.
2. **KV Cache Impact**: The KV self size doubled exactly with context length, confirming linear scaling.

---

👤 **ikawrakow** commented the **2025-03-18** at **06:36:37**:<br>

@ubergarm @davidsyoung 

Thank you for testing! It looks like a winner, so merging it.