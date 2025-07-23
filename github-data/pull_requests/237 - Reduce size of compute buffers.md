### 🔀 [#237](https://github.com/ikawrakow/ik_llama.cpp/pull/237) - Reduce size of compute buffers

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-02-28 |
| **Updated** | 2025-03-01 |

---

#### Description

I have been focusing on reducing the KV cache size, but as per the lengthy exchange in #235 the actual issue for using a very long context is the size of the compute buffers. E.g., if one attempted to run DeepSeekV3/R1 with the claimed 163k tokens maximum context length, one would need over 40 GB of CUDA compute buffer **per GPU**. But even if running on the CPU, 40 GB is nothing to sneeze at.

This PR solves the problem. For GPU and CPU inference.

Where is the issue? The `K*Q` tensor, computed in the attention portion of the network, is of size `n_ctx x n_ubatch x n_head x sizeof(float)`. One also needs `softmax(K*Q)` (of the same size), but the back-end is fortunately clever enough to reuse the same buffer. DeepSeekV3/R1 has `n_head = 128`, so with the default u-batch size of 512 tokens, this works out to 256 kB per token in the KV cache. During model load, a tets compute graph is run where the KV cache has the maximum context length (specified by the model or set on the command line) to determine the size of the compute buffer. For very long context lengths, the determined size is dominated by the size of the `K*Q` tensor. For 163k tokens it is `163,000 x 256 kB = 42.7 GB. One can of course reduce the compute buffer size by using a smaller u-batch. But this comes at a heavy performance hit for prompt processing speed. E.g., to reduce the 42.7 GB compute buffer size to, say, 5 GB to have enough VRAM left for KV cache and at least the attention tensors of DeepSeekV3/R1, one needs to lower u-batch to 64, and this comes at the price of 3X slower prefill. 

How do we solve it?

We add a command line parameter that specifies the maximum `K*Q` size we want to tolerate.
```
-amb size_in_MiB  or --attn-max-batch MiB
```
Let's call this $M_{\rm max}$.
During inference, before performing the `K*Q` multiplication the size $M$ required by `K*Q` is computed. If $M \le M_{\rm max}$, the computation proceeds as usual. If $M > M_{\rm max}$, the `V*softmax(K*Q)` is performed in $n = (M + M_{\rm max} - 1) / M_{\rm max}$ steps ($M$ and $M_{\rm max}$ are integers rounded to the nearest MiB). If the number of heads is $K$, each step computes $K/n$ heads. In each step the `K*Q` tensor is $n$ times smaller. After multiplication with `V`, the resulting tensor contains only `n_embd * n_token` entries, which is negligible compared to the size of `K*Q` for such a long context. The final `V*softmax(K*Q)` result is assembled by concatenating the results of the $n$ steps. 

Let's look at some examples for DeepSeekV3/R1 using the full 163k context and `amb = 2048` (so, 2 GiB)
* For TG (`u_batch = 1`), the `K*Q` size is `163,000 x 128 x 1 x 4 = 79 MiB`, so the computation will proceed as usual
* When the test graph is run during mode load, `K*Q` for `u_batch = 512` will be `163,000 x 128 x 512 x 4 = 40750 MiB`. Hence, the computation will be done in 20 steps, each step processing 6 or 7 heads. The back-end will record 2 GiB as the size of the `K*Q` tensor, so the compute buffer will be only slightly larger than that (to accommodate other intermediate results).
* When processing a prompt, the 2 GiB set as maximum for `K*Q` will not be exceeded before there are 8k tokens in the KV cache. After that and up to 16k tokens the `V*softmax(K*Q)` calculation will be done in 2 steps, from 16k to 24k in 3 steps, etc. For such large `K` and `Q` tensors, the cost of the matrix multiplication is many times higher than the cost of launching 2, 3. etc. matrix multiplications and soft-max computations. Hence, there will be negligible impact on performance.

As a side note: I wasted at least 2 hours trying to figure out why my implementation wasn't working. At the end it turned out to be a bug in the CUDA implementation of `GGML_OP_CONCAT` used to concatenate the step results. This PR fixes the issue for the use case required by the PR (contiguous tensors, second tensor simply appended at the end of the first).

As another side note: I wasted at least another two hours fighting with the `ggml` back-end. I was trying to avoid the $2 n$ copies needed to concatenate the intermediate results by first allocating the final result, and then simply storing the step results at the appropriate offset. The back-end did not like this idea at all, and was crashing on a null pojnter access.

---

#### 💬 Conversation

👤 **davidsyoung** commented the **2025-03-01** at **00:26:54**:<br>

This has been an incredible PR. Hugely beneficial in multiple ways. The compute buffer is drastically lower, and now can run context at max context, no issues.

It has also allowed to increasse `-ub`, which has dramatically improved prefill time.

For reference, on 15x3090 (360GB total), Q2_K R1 (230GB~), I'm able to run full context context with the following:
```
-m /models/gghfez_DeepSeek-R1-11446-Q2_K/DeepSeek-R1-11446-Q2_K-00001-of-00030.gguf
      -mla 2
      -fmoe
      -b 2048
      -ub 1024
      -amb 1024
      --tensor-split 37,25,25,25,24.5,24,24,24,24,25,24,25,24,24.5,31
      --temp 0.5
      --ctx-size 163840
      --seed 3407
      --n-gpu-layers 100
      --host 0.0.0.0
      --port 8080
```

Here is how it's loaded:

```
INFO [                    main] build info | tid="22442514837504" timestamp=1740782032 build=0 commit="unknown"
INFO [                    main] system info | tid="22442514837504" timestamp=1740782032 n_threads=64 n_threads_batch=-1 total_threads=128 system_info="AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: additional 29 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 47 key-value pairs and 1147 tensors from /models/gghfez_DeepSeek-R1-11446-Q2_K/DeepSeek-R1-11446-Q2_K-00001-of-00030.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = DeepSeek R1
llama_model_loader: - kv   3:                         general.size_label str              = 256x21B
llama_model_loader: - kv   4:                            general.license str              = mit
llama_model_loader: - kv   5:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv   6:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv   7:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv   8:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv   9:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv  10:          deepseek2.attention.head_count_kv u32              = 128
llama_model_loader: - kv  11:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  12: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  13:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  14:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  15:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  16:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  17:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  18:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  19:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  20:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  21:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  22:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  23:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  24:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  25:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  26:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  27:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  28:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  29: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  30: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  31:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  32:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  33:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<�...
llama_model_loader: - kv  34:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  35:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
llama_model_loader: - kv  36:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  37:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  38:            tokenizer.ggml.padding_token_id u32              = 1
llama_model_loader: - kv  39:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  40:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  41:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  42:               general.quantization_version u32              = 2
llama_model_loader: - kv  43:                          general.file_type u32              = 10
llama_model_loader: - kv  44:                                   split.no u16              = 0
llama_model_loader: - kv  45:                                split.count u16              = 30
llama_model_loader: - kv  46:                        split.tensors.count i32              = 1147
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q2_K:  544 tensors
llama_model_loader: - type q3_K:  180 tensors
llama_model_loader: - type q6_K:    1 tensors
llama_model_loader: - type iq4_nl:   61 tensors
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
llm_load_print_meta: model ftype      = Q2_K - Medium
llm_load_print_meta: model params     = 672.050 B
llm_load_print_meta: model size       = 227.689 GiB (2.910 BPW) 
llm_load_print_meta: repeating layers = 226.697 GiB (2.906 BPW, 670.196 B parameters)
llm_load_print_meta: general.name     = DeepSeek R1
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
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 15 CUDA devices:
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
llm_load_tensors: ggml ctx size =    7.47 MiB
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size =   289.98 MiB
llm_load_tensors:      CUDA0 buffer size = 12615.77 MiB
llm_load_tensors:      CUDA1 buffer size = 15965.27 MiB
llm_load_tensors:      CUDA2 buffer size = 15965.27 MiB
llm_load_tensors:      CUDA3 buffer size = 15965.27 MiB
llm_load_tensors:      CUDA4 buffer size = 15965.27 MiB
llm_load_tensors:      CUDA5 buffer size = 15965.27 MiB
llm_load_tensors:      CUDA6 buffer size = 15965.27 MiB
llm_load_tensors:      CUDA7 buffer size = 15965.27 MiB
llm_load_tensors:      CUDA8 buffer size = 15965.27 MiB
llm_load_tensors:      CUDA9 buffer size = 15965.27 MiB
llm_load_tensors:     CUDA10 buffer size = 15965.27 MiB
llm_load_tensors:     CUDA11 buffer size = 15965.27 MiB
llm_load_tensors:     CUDA12 buffer size = 15965.27 MiB
llm_load_tensors:     CUDA13 buffer size = 15965.27 MiB
llm_load_tensors:     CUDA14 buffer size = 12698.93 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 163840
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 1024
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: mla_attn   = 2
llama_new_context_with_model: attn_max_b = 1024
llama_new_context_with_model: fused_moe  = 1
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
llama_kv_cache_init:      CUDA0 KV buffer size =  1080.00 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =   720.00 MiB
llama_kv_cache_init:      CUDA2 KV buffer size =   720.00 MiB
llama_kv_cache_init:      CUDA3 KV buffer size =   720.00 MiB
llama_kv_cache_init:      CUDA4 KV buffer size =   720.00 MiB
llama_kv_cache_init:      CUDA5 KV buffer size =   720.00 MiB
llama_kv_cache_init:      CUDA6 KV buffer size =   720.00 MiB
llama_kv_cache_init:      CUDA7 KV buffer size =   720.00 MiB
llama_kv_cache_init:      CUDA8 KV buffer size =   720.00 MiB
llama_kv_cache_init:      CUDA9 KV buffer size =   720.00 MiB
llama_kv_cache_init:     CUDA10 KV buffer size =   720.00 MiB
llama_kv_cache_init:     CUDA11 KV buffer size =   720.00 MiB
llama_kv_cache_init:     CUDA12 KV buffer size =   720.00 MiB
llama_kv_cache_init:     CUDA13 KV buffer size =   720.00 MiB
llama_kv_cache_init:     CUDA14 KV buffer size =   540.00 MiB
llama_new_context_with_model: KV self size  = 10980.00 MiB, c^KV (f16): 10980.00 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.99 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=4)
llama_new_context_with_model:      CUDA0 compute buffer size =  5088.02 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =  5088.02 MiB
llama_new_context_with_model:      CUDA2 compute buffer size =  5088.02 MiB
llama_new_context_with_model:      CUDA3 compute buffer size =  5088.02 MiB
llama_new_context_with_model:      CUDA4 compute buffer size =  5088.02 MiB
llama_new_context_with_model:      CUDA5 compute buffer size =  5088.02 MiB
llama_new_context_with_model:      CUDA6 compute buffer size =  5088.02 MiB
llama_new_context_with_model:      CUDA7 compute buffer size =  5088.02 MiB
llama_new_context_with_model:      CUDA8 compute buffer size =  5088.02 MiB
llama_new_context_with_model:      CUDA9 compute buffer size =  5088.02 MiB
llama_new_context_with_model:     CUDA10 compute buffer size =  5088.02 MiB
llama_new_context_with_model:     CUDA11 compute buffer size =  5088.02 MiB
llama_new_context_with_model:     CUDA12 compute buffer size =  5088.02 MiB
llama_new_context_with_model:     CUDA13 compute buffer size =  5088.02 MiB
llama_new_context_with_model:     CUDA14 compute buffer size =  5088.03 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =  2588.05 MiB
```

Huge improvement across the board! 

From a speed perspective, I've seen over 200t/s.

```
prompt eval time     =    3086.56 ms /   645 tokens (    4.79 ms per token,   208.97 tokens per second)
generation eval time =   91155.54 ms /  1587 runs   (   57.44 ms per token,    17.41 tokens per second)
```

```
prompt eval time     =    7659.83 ms /  1624 tokens (    4.72 ms per token,   212.02 tokens per second)
generation eval time =   54213.84 ms /   912 runs   (   59.44 ms per token,    16.82 tokens per second)
```

```
prompt eval time     =   23483.40 ms /  4748 tokens (    4.95 ms per token,   202.19 tokens per second)
generation eval time =  132673.47 ms /  2048 runs   (   64.78 ms per token,    15.44 tokens per second)
```

```
prompt eval time     =   40631.98 ms /  7324 tokens (    5.55 ms per token,   180.25 tokens per second)
generation eval time =   58970.74 ms /   864 runs   (   68.25 ms per token,    14.65 tokens per second)
```

```
prompt eval time     =  105435.60 ms / 14645 tokens (    7.20 ms per token,   138.90 tokens per second)
generation eval time =   86701.60 ms /  1041 runs   (   83.29 ms per token,    12.01 tokens per second)
```

I still want to experiment with lower `-amb` values and see how that impacts the compute buffer, but having `-ub 1024` most definitely speeds up prefill time. I believe I drop to around 120-140t/s with `-ub 512`. 

I cannot express how much this starts to make the model usable. I wonder what could either: a) reduce the compute buffer to even smaller, because if so, can run a much higher quant, or b) speed up the PP or TG further. Even during inference the gpu's are really only using like 5-10% usage!

I am picking up another 3090 tomorrow, so I'll have 16 in total, and provided I can get it loaded, I'll have more VRAM to play with and potentially a higher quant.

Excellent work on this.

---

👤 **davidsyoung** commented the **2025-03-01** at **00:55:02**:<br>

Also, gave the [84853b9](https://github.com/ikawrakow/ik_llama.cpp/pull/237/commits/84853b9a9bb2c71b80c704d2b0d0675cb132a539) commit a test run and it seems to be producing different outcomes each time on regeneration with a fixed seed. 

Not sure if it’s something I’m doing wrong on my end.

---

👤 **ikawrakow** commented the **2025-03-01** at **06:25:19**:<br>

> Also, gave the [84853b9](https://github.com/ikawrakow/ik_llama.cpp/pull/237/commits/84853b9a9bb2c71b80c704d2b0d0675cb132a539) commit a test run and it seems to be producing different outcomes each time on regeneration with a fixed seed.
> 
> Not sure if it’s something I’m doing wrong on my end.

I wouldn't know why that could affect your results. The change in 84853b9a9bb2c71b80c704d2b0d0675cb132a539 only runs on the CPU, so never gets executed in your case.

---

👤 **davidsyoung** commented the **2025-03-01** at **07:57:12**:<br>

> > Also, gave the [84853b9](https://github.com/ikawrakow/ik_llama.cpp/pull/237/commits/84853b9a9bb2c71b80c704d2b0d0675cb132a539) commit a test run and it seems to be producing different outcomes each time on regeneration with a fixed seed.
> > Not sure if it’s something I’m doing wrong on my end.
> 
> I wouldn't know why that could affect your results. The change in [84853b9](https://github.com/ikawrakow/ik_llama.cpp/commit/84853b9a9bb2c71b80c704d2b0d0675cb132a539) only runs on the CPU, so never gets executed in your case.

Ah weird. Maybe I’m going insane. Was late last night! 

Thank you again 👌🏽