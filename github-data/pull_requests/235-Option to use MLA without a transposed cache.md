### 🔀 [#235](https://github.com/ikawrakow/ik_llama.cpp/pull/235) - Option to use MLA without a transposed cache

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-02-27 |
| **Updated** | 2025-02-28 |

---

#### Description

The `-mla` (or `--mla-use`) command line option turns from previously a boolean value to an integer:
* `mla = 0`: use standard attention
* `mla = 1`: use MLA with transposed cache - this is the existing MLA implementation
* `mla = 2`: use MLA without transposed cache - this is the option added by this PR

Why do we need this? Apparently many people are interested in using the maximum context length of long context models. For DeepSeekV3/R1, the rage of the day, it is 163k tokens. This requires a lot of RAM/VRAM. Let's take a look:
  
* **Standard attention (mla = 0):** memory required per token is `n_layer * (3072 * sizeof(K cache element) + 2048 * sizeof(V cache element))`. For DeepSeekV3/R1 this works out to **610 kB** per token when using `fp16` cache. For `Q8_0` K and V cache it is **324 kB** per token, but this requires FA, so CPU-only inference (CUDA does not support FA with different K and V head sizes as found in the DeepSeek models). So, for GPU or mixed CPU/GPU inference the best one can do is `Q8_0` for K cache and `f16` for V cache, so  **438.4 kB** per token.
* **MLA (mla = 1):** memory required per token is `n_layer * (576 * sizeof(K cache element) + 512 * sizeof(V cache element))`. For DeepSeekV3/R1 this works out to **129.6 kB** per token for `fp16` cache. When using MLA the V cache is transposed, so quantization is not possible at all, so the best one can do is `Q8_0` for K cache and `fp16` for V cache. This results in **97.5 kB** per token
* **MLA(mla = 2):** memory required per token is `n_layer * 576 * sizeof(K cache element)`, so **68.6 kB** per token with `fp16` cache and **36.5 kB** per token with `Q8_0` cache.

I.e., for GPU-only or hybrid GPU/CPU inference, where VRAM is the limiting factor (unless one keeps the cache on the host and copies it to the GPU as needed, but this would make performance much lower), the new option added by the PR uses **12X** less KV cache memory than standards attention and **2.7X** less than the existing MLA implementation. For a context of 163k tokens the memory required will be **5.67 GiB**.

The down side of this is that one has to transpose the K cache during inference (`ggml`, despite representing itself as a general purpose ML library, lacks the ability to perform transposed matrix multiplications, and I haven't come around to add this ability to my fork). This adds an additional computation and requires an extra compute buffer (to hold the contiguous transposed copy of the entire K cache for one layer). The size of this extra buffer can be computed as `n_token * 512 * sizeof(float) = 318 MiB` for 163k tokens, so this should not be a serious limitation.  But the additional operation that copies the transposed K cache into contiguous memory may result in a significant performance penalty, so let's look at that. As I don't have the ability to run DeepSeekV3/R1, I'm using  for the performance comparisons below.  DeepSeek-Lite  has the same architecture as DeepSeekV3/R1 with fewer parameters (16B, MoE, 64 experts, 6 used experts, exat same attention tensor sizes as DeepSeekV3/R1).  


**Note**: at this point `ggml` does not support transposing quantized data, so for `mla = 2` the K cache must be `fp16` or `bf16`. Hence, the above analysis for quantized cache with `mla = 2` will only apply when I have come around to implement transposing a quantized cache. 

### Hybrid GPU/CPU inference

The GPU is RTX-4080, the CPU is Ryzen-7950X. Experts are kept on the CPU, all other tensors are offloaded to the GPU.

| model                 | rtr | fmoe |      test |   t/s (mla = 0)  |    t/s (mla = 1) |   t/s (mla = 2)  |
| --------------------- | --: | ---: | --------: | ---------------: | ---------------: | ---------------: |
| deepseek2 16B IQ4_NL  |   1 |    1 |     pp512 |  1161.48 ± 43.70 |  1121.07 ± 39.10 |  1116.03 ± 43.37 |
| deepseek2 16B IQ4_NL  |   1 |    1 |    pp1024 |   1170.21 ± 4.98 |  1113.50 ± 20.14 |   1124.49 ± 4.31 |
| deepseek2 16B IQ4_NL  |   1 |    1 |    pp2048 |   1149.21 ± 2.62 |   1104.81 ± 7.31 |  1099.57 ± 27.33 |
| deepseek2 16B IQ4_NL  |   1 |    1 |    pp4096 |  1117.39 ± 11.04 |   1081.31 ± 2.93 |   1087.32 ± 2.91 |
| deepseek2 16B IQ4_NL  |   1 |    1 |    pp8192 |  1064.98 ± 12.98 |   1026.89 ± 7.58 |  1022.51 ± 20.84 |
| deepseek2 16B IQ4_NL  |   1 |    1 |   pp16384 |   965.42 ± 11.44 |   924.85 ± 10.69 |    921.28 ± 4.84 |

 I.e., for prompt processing (a.k.a. "prefill") MLA is very slightly slower than standard attention, but there is not real difference between `mla = 1` and `mla = 2` added by this PR.

For token generation (TG) I use the `-gp` option in `llama-bench` to evaluate TG performance as a function of the number of tokens in the KV cache. Here are the results:
 
| model          | rtr | fmoe |          test |   t/s (mla = 1)  |   t/s (mla = 1)  |   t/s (mla = 2)  |
| -------------- | --: | ---: | ------------: | ---------------: | ---------------: | ---------------: |
| deepseek2 16B  |   1 |    1 |    tg64@pp128 |     52.37 ± 0.11 |     52.32 ± 0.04 |     52.63 ± 0.07 |
| deepseek2 16B  |   1 |    1 |    tg64@pp256 |     51.65 ± 1.38 |     52.25 ± 0.10 |     52.60 ± 0.04 |
| deepseek2 16B  |   1 |    1 |    tg64@pp512 |     51.47 ± 0.39 |     51.70 ± 0.34 |     52.20 ± 0.06 |
| deepseek2 16B  |   1 |    1 |   tg64@pp1024 |     48.61 ± 0.67 |     51.45 ± 0.41 |     51.58 ± 0.11 |
| deepseek2 16B  |   1 |    1 |   tg64@pp2048 |     50.10 ± 0.26 |     50.89 ± 0.52 |     50.10 ± 0.98 |
| deepseek2 16B  |   1 |    1 |   tg64@pp4096 |     47.75 ± 0.13 |     49.98 ± 0.44 |     48.78 ± 0.05 |
| deepseek2 16B  |   1 |    1 |   tg64@pp8192 |     43.22 ± 0.47 |     48.07 ± 0.14 |     45.42 ± 0.40 |

I.e., for short contexts `mla = 2` is about on par with `mla = 1`. As the context grows it becomes slower due to the added cost of transposing the K cache, but it is still better than standard attention (`mla = 0`) at 8k tokens. 

### CPU only inference

| model                | rtr | fmoe |      test |    t/s (mla = 0) |    t/s (mla = 1) |   t/s (mla = 2)  |
| -------------------- | --: | ---: | --------: | ---------------: | ---------------: | ---------------: |
| deepseek2 16B IQ4_NL |   1 |    1 |     pp512 |    638.34 ± 2.78 |    581.79 ± 0.82 |    588.73 ± 1.93 |
| deepseek2 16B IQ4_NL |   1 |    1 |    pp1024 |    613.98 ± 1.95 |    539.69 ± 2.67 |    541.44 ± 9.46 |
| deepseek2 16B IQ4_NL |   1 |    1 |    pp2048 |    571.96 ± 0.87 |    471.74 ± 4.37 |    477.14 ± 2.42 |
| deepseek2 16B IQ4_NL |   1 |    1 |    pp4096 |    495.86 ± 1.11 |    368.75 ± 2.62 |    372.69 ± 1.31 |
| deepseek2 16B IQ4_NL |   1 |    1 |    pp8192 |    390.40 ± 4.78 |    254.44 ± 0.06 |    255.92 ± 1.49 |
| deepseek2 16B IQ4_NL |   1 |    1 |   pp16384 |    272.56 ± 1.29 |    156.00 ± 0.75 |    154.40 ± 0.12 |

I.e., when running only on the CPU MLA is significantly slower than standard attention for prompt processing, but there is no real difference between `mla = 1` and `mla = 2`.

| model                | rtr | fmoe |          test |   t/s (mla = 0)  |   t/s (mla = 1)  |   t/s (mla = 2)  |
| ---------------------| --: | ---: | ------------: | ---------------: | ---------------: | ---------------: |
| deepseek2 16B IQ4_NL |   1 |    1 |    tg64@pp128 |     32.55 ± 0.01 |     33.30 ± 0.02 |     32.41 ± 0.05 |
| deepseek2 16B IQ4_NL |   1 |    1 |    tg64@pp256 |     31.74 ± 0.07 |     32.67 ± 0.01 |     31.22 ± 0.02 |
| deepseek2 16B IQ4_NL |   1 |    1 |    tg64@pp512 |     29.98 ± 0.01 |     32.06 ± 0.03 |     30.16 ± 0.01 |
| deepseek2 16B IQ4_NL |   1 |    1 |   tg64@pp1024 |     28.37 ± 0.02 |     31.68 ± 0.01 |     28.48 ± 0.09 |
| deepseek2 16B IQ4_NL |   1 |    1 |   tg64@pp2048 |     25.15 ± 0.02 |     29.98 ± 0.03 |     25.18 ± 0.04 |
| deepseek2 16B IQ4_NL |   1 |    1 |   tg64@pp4096 |     20.22 ± 0.02 |     27.22 ± 0.13 |     20.36 ± 0.01 |
| deepseek2 16B IQ4_NL |   1 |    1 |   tg64@pp8192 |     14.56 ± 0.01 |     22.98 ± 0.11 |     14.18 ± 0.01 |

Here `mla = 2` is much slower than `mla = 1` for long contexts, and about on par with standard attention (`mla = 0`). Looking at the code in `ggml_compute_forward_dup_bytes`, which gets invoked to copy the transposed K cache data to contiguous memory, it is pretty much as inefficient as it gets. But I leave this for a follow up PR.

---

#### 💬 Conversation

👤 **davidsyoung** commented the **2025-02-27** at **15:08:55**:<br>

Hey, thank you for your work on this. Trying to run with -mla 2, but still getting a 8900MB allocation per card. I'm not sure if this is correct, or am I doing something wrong with my run commands (I'm aware the layers are poorly balanced atm, but just wondering if this is as expected:

Command:
```
      -m /models/gghfez_DeepSeek-R1-11446-Q2_K/DeepSeek-R1-11446-Q2_K-00001-of-00030.gguf
      -ub 512
      -mla 2
      --cache-type-k q8_0
      --main-gpu 0
      --tensor-split 42,25,25,25,25,25,25,25,25,25,25,25,25,25,40
      --threads 64
      --temp 0.6
      --ctx-size 32768
      --seed 3407
      --n-gpu-layers 62
      --host 0.0.0.0
      --port 8080
```

Log:
```
INFO [                    main] build info | tid="22457510539264" timestamp=1740668223 build=0 commit="unknown"
INFO [                    main] system info | tid="22457510539264" timestamp=1740668223 n_threads=64 n_threads_batch=-1 total_threads=128 system_info="AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
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
llm_load_tensors:      CUDA0 buffer size = 16607.09 MiB
llm_load_tensors:      CUDA1 buffer size = 15965.27 MiB
llm_load_tensors:      CUDA2 buffer size = 15965.27 MiB
llm_load_tensors:      CUDA3 buffer size = 11973.95 MiB
llm_load_tensors:      CUDA4 buffer size = 15965.27 MiB
llm_load_tensors:      CUDA5 buffer size = 15965.27 MiB
llm_load_tensors:      CUDA6 buffer size = 15965.27 MiB
llm_load_tensors:      CUDA7 buffer size = 15965.27 MiB
llm_load_tensors:      CUDA8 buffer size = 11973.95 MiB
llm_load_tensors:      CUDA9 buffer size = 15965.27 MiB
llm_load_tensors:     CUDA10 buffer size = 15965.27 MiB
llm_load_tensors:     CUDA11 buffer size = 15965.27 MiB
llm_load_tensors:     CUDA12 buffer size = 15965.27 MiB
llm_load_tensors:     CUDA13 buffer size = 11973.95 MiB
llm_load_tensors:     CUDA14 buffer size = 20681.56 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: mla_attn   = 2
llama_new_context_with_model: fused_moe  = 0
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
llama_kv_cache_init:      CUDA0 KV buffer size =   252.00 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =   144.00 MiB
llama_kv_cache_init:      CUDA2 KV buffer size =   144.00 MiB
llama_kv_cache_init:      CUDA3 KV buffer size =   108.00 MiB
llama_kv_cache_init:      CUDA4 KV buffer size =   144.00 MiB
llama_kv_cache_init:      CUDA5 KV buffer size =   144.00 MiB
llama_kv_cache_init:      CUDA6 KV buffer size =   144.00 MiB
llama_kv_cache_init:      CUDA7 KV buffer size =   144.00 MiB
llama_kv_cache_init:      CUDA8 KV buffer size =   108.00 MiB
llama_kv_cache_init:      CUDA9 KV buffer size =   144.00 MiB
llama_kv_cache_init:     CUDA10 KV buffer size =   144.00 MiB
llama_kv_cache_init:     CUDA11 KV buffer size =   144.00 MiB
llama_kv_cache_init:     CUDA12 KV buffer size =   144.00 MiB
llama_kv_cache_init:     CUDA13 KV buffer size =   108.00 MiB
llama_kv_cache_init:     CUDA14 KV buffer size =   180.00 MiB
llama_new_context_with_model: KV self size  = 2196.00 MiB, c^KV (f16): 2196.00 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.99 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=4)
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 8900.01 MiB on device 0: cudaMalloc failed: out of memory
ggml_gallocr_reserve_n: failed to allocate CUDA0 buffer of size 9332334592
llama_new_context_with_model: failed to allocate compute buffers
llama_init_from_gpt_params: error: failed to create context with model '/models/gghfez_DeepSeek-R1-11446-Q2_K/DeepSeek-R1-11446-Q2_K-00001-of-00030.gguf'
 ERR [              load_model] unable to load model | tid="22457510539264" timestamp=1740668683 model="/models/gghfez_DeepSeek-R1-11446-Q2_K/DeepSeek-R1-11446-Q2_K-00001-of-00030.gguf"
free(): invalid pointer

```

Would really appreciate your help to see if I'm doing something wrong. Thank you!

---

👤 **davidsyoung** commented the **2025-02-27** at **16:35:08**:<br>

@ikawrakow 

Was able to run this with 24K ctx, but not sure if this amount of compute buffer is still correct:

```
INFO [                    main] build info | tid="22970858381312" timestamp=1740670359 build=0 commit="unknown"
INFO [                    main] system info | tid="22970858381312" timestamp=1740670359 n_threads=64 n_threads_batch=-1 total_threads=128 system_info="AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
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
llm_load_tensors:     CUDA10 buffer size = 11973.95 MiB
llm_load_tensors:     CUDA11 buffer size = 15965.27 MiB
llm_load_tensors:     CUDA12 buffer size = 15965.27 MiB
llm_load_tensors:     CUDA13 buffer size = 15965.27 MiB
llm_load_tensors:     CUDA14 buffer size = 16690.25 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 20480
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: mla_attn   = 2
llama_new_context_with_model: fused_moe  = 0
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
llama_kv_cache_init:      CUDA0 KV buffer size =   135.00 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =    90.00 MiB
llama_kv_cache_init:      CUDA2 KV buffer size =    90.00 MiB
llama_kv_cache_init:      CUDA3 KV buffer size =    90.00 MiB
llama_kv_cache_init:      CUDA4 KV buffer size =    90.00 MiB
llama_kv_cache_init:      CUDA5 KV buffer size =    90.00 MiB
llama_kv_cache_init:      CUDA6 KV buffer size =    90.00 MiB
llama_kv_cache_init:      CUDA7 KV buffer size =    90.00 MiB
llama_kv_cache_init:      CUDA8 KV buffer size =    90.00 MiB
llama_kv_cache_init:      CUDA9 KV buffer size =    90.00 MiB
llama_kv_cache_init:     CUDA10 KV buffer size =    67.50 MiB
llama_kv_cache_init:     CUDA11 KV buffer size =    90.00 MiB
llama_kv_cache_init:     CUDA12 KV buffer size =    90.00 MiB
llama_kv_cache_init:     CUDA13 KV buffer size =    90.00 MiB
llama_kv_cache_init:     CUDA14 KV buffer size =    90.00 MiB
llama_new_context_with_model: KV self size  = 1372.50 MiB, c^KV (f16): 1372.50 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.99 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=4)
llama_new_context_with_model:      CUDA0 compute buffer size =  5720.01 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =  5718.01 MiB
llama_new_context_with_model:      CUDA2 compute buffer size =  5718.01 MiB
llama_new_context_with_model:      CUDA3 compute buffer size =  5718.01 MiB
llama_new_context_with_model:      CUDA4 compute buffer size =  5718.01 MiB
llama_new_context_with_model:      CUDA5 compute buffer size =  5718.01 MiB
llama_new_context_with_model:      CUDA6 compute buffer size =  5718.01 MiB
llama_new_context_with_model:      CUDA7 compute buffer size =  5718.01 MiB
llama_new_context_with_model:      CUDA8 compute buffer size =  5718.01 MiB
llama_new_context_with_model:      CUDA9 compute buffer size =  5718.01 MiB
llama_new_context_with_model:     CUDA10 compute buffer size =  5718.01 MiB
llama_new_context_with_model:     CUDA11 compute buffer size =  5718.01 MiB
llama_new_context_with_model:     CUDA12 compute buffer size =  5718.01 MiB
llama_new_context_with_model:     CUDA13 compute buffer size =  5718.01 MiB
llama_new_context_with_model:     CUDA14 compute buffer size =  5718.02 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   174.02 MiB
llama_new_context_with_model: graph nodes  = 3724
llama_new_context_with_model: graph splits = 16
INFO [                    init] initializing slots | tid="22970858381312" timestamp=1740670875 n_slots=1
INFO [                    init] new slot | tid="22970858381312" timestamp=1740670875 id_slot=0 n_ctx_slot=20480
INFO [                    main] model loaded | tid="22970858381312" timestamp=1740670875
INFO [                    main] chat template | tid="22970858381312" timestamp=1740670875 chat_example="You are a helpful assistant\n\n<｜User｜>Hello<｜Assistant｜>Hi there<｜end▁of▁sentence｜><｜User｜>How are you?<｜Assistant｜>" built_in=true
INFO [                    main] HTTP server listening | tid="22970858381312" timestamp=1740670875 n_threads_http="127" port="8080" hostname="0.0.0.0"
INFO [            update_slots] all slots are idle | tid="22970858381312" timestamp=1740670875```

---

👤 **ikawrakow** commented the **2025-02-27** at **16:50:48**:<br>

So, when I wrote the PR description I had forgotten that it is not yet possible to transpose quantized cache, which would be needed if we wanted to use `mla = 2` with quantized cache. I realized my mistake and added a comment, but I guess it is easy to miss. So, at this point `mla = 2` uses `fp16` for the cache, which means about 69 kB per token for DeepSeek-R1, so 1.58 GiB for 24k context, so about 100 MiB per card in your 15 GPU setup (wow!). This is also what we see reported. 

I haven't looked in detail into the compute buffers on CUDA. I wouldn't have expected 5.7 GiB per GPU, this seems way too much. But I also don't have access to a multi-GPU box, so have never played with that. It looks like each GPU is allocating the same compute buffer as if the computation was running on a single GPU.

---

👤 **davidsyoung** commented the **2025-02-27** at **17:12:01**:<br>

Incredible, that makes sense. The cache using fp16 isn't a huge problem, to be honest. Also, yes, the 15 gpu build (trying to find a 16th for TP!) has been a lot of pain, so to see the speed increase on this, and longer context, is really promising. So thank you for all of your hard work.

For these compute buffers, is there anything I can do to reduce it to the expected amount?

---

👤 **ikawrakow** commented the **2025-02-27** at **17:14:38**:<br>

@davidsyoung Have you tried using `-fmoe` (`--fused-moe` from PR #229? This fuses several MoE operations. In my testing withDeepSeek-Lite it resulted in a significant boost in prefill performance (~30%) and a small gain in TG as well.

---

👤 **ikawrakow** commented the **2025-02-27** at **17:21:10**:<br>

> For these compute buffers, is there anything I can do to reduce it to the expected amount?

I need to look into this. Have you tried  `--split-mode row` and if yes, does it work?

---

👤 **davidsyoung** commented the **2025-02-27** at **17:27:39**:<br>

So I tried to change the following:

before: 
```
-ub 512
-ctx-size 20480
--cache-type-k q8_0
```
to
```
-ub 1024
-ctx-size 32768
//removed the cache type
```

and the kv size seem right at
```
KV self size  = 2196.00 MiB, c^KV (f16): 2196.00 MiB, kv^T: not used
```

However, the compute buffer is now trying to allocate  `17796.02 MiB` up from `5718.01 MiB` per card.

I believe this is `-ub` related possibly?

I will try `--split-mode row` now.

---

👤 **ikawrakow** commented the **2025-02-27** at **17:35:18**:<br>

Yes, the compute buffer size is proportional to the micro batch (ub) size. Typically performance first increases with increasing `ub` and then starts declining as `ub` increases. The default size is set based on the experience with much smaller models. I haven't seen people reporting performance values as a function of batch size or u-batch size for DeepSeek-R1.  You can try using `-b 512 -ub 256` and see what happens. This should decrease compute buffer size, but the question is how much performance penalty (if any) one gets from that.

---

👤 **ikawrakow** commented the **2025-02-27** at **17:48:10**:<br>

Just tried with DeepSeek-Lite. For a context of 32k tokens the CUDA compute buffer size is 1172 MiB with default batch/u-batch size. If I use `-b 512 -ub 256` it goes down to 972 MiB. With `-b 256 -ub 256` it becomes 603 MiB.

---

👤 **davidsyoung** commented the **2025-02-27** at **17:50:52**:<br>

> Just tried with DeepSeek-Lite. For a context of 32k tokens the CUDA compute buffer size is 1172 MiB with default batch/u-batch size. If I use `-b 512 -ub 256` it goes down to 972 MiB. With `-b 256 -ub 256` it becomes 603 MiB.

Is that behaving as expected for you when you see that? I can't tell if I should see similar amounts, or is what I'm seeing correct for the model size.

---

👤 **davidsyoung** commented the **2025-02-27** at **17:53:21**:<br>

``--split-mode row`` run:

```
INFO [                    main] build info | tid="23335418978304" timestamp=1740678236 build=0 commit="unknown"
INFO [                    main] system info | tid="23335418978304" timestamp=1740678236 n_threads=64 n_threads_batch=-1 total_threads=128 system_info="AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
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
llm_load_tensors: ggml ctx size =    1.40 MiB
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size =   289.98 MiB
llm_load_tensors:      CUDA0 buffer size =   409.90 MiB
llm_load_tensors: CUDA_Split buffer size = 232453.45 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 512
llama_new_context_with_model: n_ubatch   = 256
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: mla_attn   = 2
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
llama_kv_cache_init:      CUDA0 KV buffer size =  2196.00 MiB
llama_new_context_with_model: KV self size  = 2196.00 MiB, c^KV (f16): 2196.00 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.99 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =  4349.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =    39.00 MiB
llama_new_context_with_model: graph nodes  = 3608
llama_new_context_with_model: graph splits = 2
ggml/src/ggml-cuda.cu:731: GGML_ASSERT(tensor->view_src == nullptr) failed
```

---

👤 **ikawrakow** commented the **2025-02-27** at **17:58:24**:<br>

> do you think there's a sweeet spot for type of quant to use for R1 in terms of quality etc.

Do you quantize the model yourself or do you download a quantized model from somewhere? For DeepSeek it seems it is important to use more bits for the attention tensors and the shared experts. As most of the size is in the MoE experts this does not lead to a very significant increase in model size. After that you go with the highest bpw for the MoE experts that you can fit into VRAM (after deducting KV cache and compute buffers). `Q2_K` is not a very high quality quantization for the 2.625 bpw that it uses. The i-quants are better. If you have to stay below 3 bpw to fit the model in VRAM, `IQ2_M` is a better option for the experts than `Q2_K`. `IQ2_K` will likely give you similar quality as `Q2_K` while being ~10% smaller. If you can go to 3 bpw, then `IQ3_XXS` is a significant quality improvement compared to `Q2_K`. 

But all of this are just guesses as I have never tried DeepSeekV3/R1 myself.

---

👤 **davidsyoung** commented the **2025-02-27** at **18:01:58**:<br>

> > do you think there's a sweeet spot for type of quant to use for R1 in terms of quality etc.
> 
> Do you quantize the model yourself or do you download a quantized model from somewhere? For DeepSeek it seems it is important to use more bits for the attention tensors and the shared experts. As most of the size is in the MoE experts this does not lead to a very significant increase in model size. After that you go with the highest bpw for the MoE experts that you can fit into VRAM (after deducting KV cache and compute buffers). `Q2_K` is not a very high quality quantization for the 2.625 bpw that it uses. The i-quants are better. If you have to stay below 3 bpw to fit the model in VRAM, `IQ2_M` is a better option for the experts than `Q2_K`. `IQ2_K` will likely give you similar quality as `Q2_K` while being ~10% smaller. If you can go to 3 bpw, then `IQ3_XXS` is a significant quality improvement compared to `Q2_K`.
> 
> But all of this are just guesses as I have never tried DeepSeekV3/R1 myself.

Makes sense. Thank you. I am currently using https://huggingface.co/gghfez/DeepSeek-R1-11446-Q2_K, but now that it seems I'll be able to unlock a good bit of VRAM with your implementation (thank you), I may venture into trying to trying to quantize the model myself with a IQ3_XXS. It really depends on finding a sweet spot with this compute buffer!

Thank you for all of your help/work, it's massively appreciated.

---

👤 **davidsyoung** commented the **2025-02-27** at **19:13:10**:<br>

Doing some testing with different batch sizes, micro-batch sizes and context.

Test 1:

At `-b 512 -ub 256 --ctx-size 32768 (280w power limit - each card uses 120-160w~ during inference)`

`
llama_new_context_with_model: KV self size  = 2196.00 MiB, c^KV (f16): 2196.00 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.99 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=4)
llama_new_context_with_model:      CUDA0 compute buffer size =  4466.00 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =  4465.00 MiB
llama_new_context_with_model:      CUDA2 compute buffer size =  4465.00 MiB
llama_new_context_with_model:      CUDA3 compute buffer size =  4465.00 MiB
llama_new_context_with_model:      CUDA4 compute buffer size =  4465.00 MiB
llama_new_context_with_model:      CUDA5 compute buffer size =  4465.00 MiB
llama_new_context_with_model:      CUDA6 compute buffer size =  4465.00 MiB
llama_new_context_with_model:      CUDA7 compute buffer size =  4465.00 MiB
llama_new_context_with_model:      CUDA8 compute buffer size =  4465.00 MiB
llama_new_context_with_model:      CUDA9 compute buffer size =  4465.00 MiB
llama_new_context_with_model:     CUDA10 compute buffer size =  4465.00 MiB
llama_new_context_with_model:     CUDA11 compute buffer size =  4465.00 MiB
llama_new_context_with_model:     CUDA12 compute buffer size =  4465.00 MiB
llama_new_context_with_model:     CUDA13 compute buffer size =  4465.00 MiB
llama_new_context_with_model:     CUDA14 compute buffer size =  4465.01 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   135.01 MiB
`

I see pretty good performance overall. I have seen 140~ prefill before but I believe that was without MLA.

```
266.36 ms /     4 tokens (   66.59 ms per token,    15.02 tokens per second)
8889.48 ms /   161 runs   (   55.21 ms per token,    18.11 tokens per second)

3065.82 ms /   272 tokens (   11.27 ms per token,    88.72 tokens per second)
83350.14 ms /  1464 runs   (   56.93 ms per token,    17.56 tokens per second)

8095.82 ms /   940 tokens (    8.61 ms per token,   116.11 tokens per second)
115329.06 ms /  1965 runs   (   58.69 ms per token,    17.04 tokens per second)

41304.65 ms /  4748 tokens (    8.70 ms per token,   114.95 tokens per second)
79665.28 ms /  1247 runs   (   63.89 ms per token,    15.65 tokens per second)

189065.31 ms / 16613 tokens (   11.38 ms per token,    87.87 tokens per second)
84121.32 ms /   980 runs   (   85.84 ms per token,    11.65 tokens per second)
```

Test 2:

`-b 2048 -ub 512 --ctx-size 32768` gave out of memory: 
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 8898.01 MiB on device 1: cudaMalloc failed: out of memory

Test 3:

`-b 1024 -ub 512 --ctx-size 163840`

While the KV cache at max context of 163k is reasonable ` KV self size  = 10980.00 MiB, c^KV (f16): 10980.00 MiB, kv^T: not used`

The compute buffer goes pretty insane per GPU: `allocating 42820.01 MiB on device 0: cudaMalloc failed: out of memory`, that's with `-b 1024 -ub 512`. 

---

So I'm not too sure what's up with the compute buffer. Maybe this is just the size of it given the size of the model. But allocating 42.8GB per gpu, across 15 gpu's would be 642GB VRAM just for compute buffer.

Definitely seems a magnitude out, but I'm also really not sure what I'm taking about!

---

👤 **saood06** commented the **2025-02-27** at **20:52:40**:<br>

>For DeepSeek it seems it is important to use more bits for the attention tensors and the shared experts. As most of the size is in the MoE experts this does not lead to a very significant increase in model size.

The model size might not go up significantly but the performance does noticeably go down if you do that strategy as those weights are always used unlike the expert weights, this may not matter as much with them being on CUDA but from another user's reports on llama.cpp who was offloading those to CUDA they still had a performance hit. For me IQ4_K_R4 (V2) is slower than V1 with 2.63 t/s for V2 vs 3.22 t/s V1.

Here's a table of early perplexity values I've collected for various quants of Deepseek.

Quant      | [1] |	[2] 	|[3] 	|[4] |	[5] 	|[6]| 	[7]| 	[8]	|[9]	|[10]	|[11]	|[12]|[13]|[14]|[15]|[16]|SUM
--- 	               | --- 	| --- 	| --- 	|--- 	|--- 	|--- 	|--- 	|--- 	|--- 	|--- 	|--- | ---|---|---|---|---|---
| My IQ1_S_R4 | 3.7099 | 4.6162 | 3.5438 | 3.4199 | 3.5375 | 3.5710 | 3.5428 | 3.6748 | 3.7417 | 3.6724 | 3.7879 | 3.9602 | 4.0477 | 4.1439 | 4.2809 | 4.1981 | 61.4487 |
| My IQ4_K_R4 V1 (4.519 BPW) | 2.5944 | 3.3242 | 2.4001 | 1.9949 | 1.8067 | 1.6666 | 1.5704 | 1.5055 | 1.4559 | 1.4154 | 1.3999 | 1.4404 | 1.4500 | 1.5786 | 1.7101 | 1.7729 | 29.0860 |
| My IQ4_K_R4 V2 (4.589 BPW) | 2.5474 | 3.3247 | 2.4001 | 2.0029 | 1.8181 | 1.6716 | 1.5734 | 1.5084 | 1.4592 | 1.4194 | 1.4035 | 1.4376 | 1.4476 | 1.5734 | 1.7047 | 1.7654 | 29.0574 |
| My IQ4_K_R4 V3 (4.621 BPW) | 2.5551 | 3.3239 | 2.3980 | 1.9980 | 1.8057 | 1.6631 | 1.5676 | 1.5029 | 1.4525 | 1.4122 | 1.3963 | 1.4421 | 1.4516 | 1.5784 | 1.7089 | 1.7692 | 29.0255 |
| BF16/Q4_0/Q4_0** | 2.5160 | 3.3227 | 2.4058 | 2.0030 | 1.8059 | 1.6632 | 1.5704 | 1.5020 | 1.4516 | 1.4119 | 1.3972 | 1.4372 | 1.4479 | 1.5764 | 1.7091 | 1.7684 | 28.9887 |
| BF16/Q4_0/Q4_0 + `imatrix`** | 2.4996 | 3.3182 | 2.3944 | 1.9934 | 1.8041 | 1.6605 | 1.5667 | 1.4976 | 1.4491 | 1.4110 | 1.3963 | 1.4279 | 1.4390 | 1.5674 | 1.6989 | 1.7584 | 28.8825 |
| BF16/Q4_0/Q8_0** | 2.5046 | 3.2991 | 2.3829 | 1.9872 | 1.7991 | 1.6562 | 1.5628 | 1.4979 | 1.4485 | 1.4099 | 1.3955 | 1.4280 | 1.4409 | 1.5679 | 1.6980 | 1.7582 | 28.8367 |
| BF16/Q5_K/Q5_K** | 2.5143 | 3.3036 | 2.3746 | 1.9854 | 1.7920 | 1.6478 | 1.5561 | 1.4888 | 1.4393 | 1.4002 | 1.3845 | 1.4178 | 1.4293 | 1.5569 | 1.6882 | 1.7480 | 28.7268 |
| BF16/Q4_K/Q6_K** | 2.5266 | 3.3006 | 2.3780 | 1.9832 | 1.7932 | 1.6461 | 1.5550 | 1.4902 | 1.4404 | 1.3994 | 1.3840 | 1.4207 | 1.4321 | 1.5584 | 1.6898 | 1.7498 | 28.7475 |
| BF16/Q5_K/Q6_K** | 2.5030 | 3.2798 | 2.3704 | 1.9793 | 1.7866 | 1.6453 | 1.5536 | 1.4883 | 1.4388 | 1.3993 | 1.3838 | 1.4188 | 1.4298 | 1.5565 | 1.6874 | 1.7464 | 28.6671 |
IQ2_XXS |	3.39| 	4.56|	3.44| 	3.27| 	3.27| 	3.20| 	3.12 |	3.12|
IQ3_XXS  |	2.69 |	3.53|	2.51 |	2.11 |	1.91 |	1.78 |	1.69 |	1.62|
UD-IQ1_M |	3.4155  |4.2311 | 3.0817 | 2.8601 | 2.6933 | 2.5792 | 2.5123 | 2.5239  
UD-IQ1_S  | 3.8939  |4.7189 | 3.7812 | 3.6799 | 3.6215 | 3.6922 | 3.6442|  3.7472|  3.8353|	3.7663|	3.8983|	4.0621


**For these quants in the format A/B/C (also imatrix is Bartowski imatrix for experts only)

    // ###
    if (ftype == LLAMA_FTYPE_MOSTLY_Q_XXX) {
        if (name.find("_exps") != std::string::npos) {
            if (name.find("ffn_down") != std::string::npos) {
                new_type = GGML_TYPE_C;
            }
            else {
                new_type = GGML_TYPE_B;
            }
        }
        else {
            new_type = GGML_TYPE_A;
        }
    }
    else
    // ###

My V1/V2/V3, I employ the strategy described above, slightly increasing the size of the model but IMO the performance difference was not worth it (that might change with hybrid/full offload). All tensors for mine were imatrixed with mradermacher imatrix except for the new split tensor.

Also for reference here is some compute buffer sizes I've seen:

n_ctx = 128000
CPU compute buffer size = 64468.01 MiB
n_ctx = 64000
CPU compute buffer size = 32343.01 MiB

---

👤 **davidsyoung** commented the **2025-02-27** at **22:29:43**:<br>

I may have to start experimenting with quants myself, this is really useful. 

For the compute buffers, would you happen to know what batch/micro batch sizes were set to?

I’m getting a total of 67GB for 32k context. It would be nice if I could claw back some some how…

---

👤 **saood06** commented the **2025-02-27** at **23:08:14**:<br>

> I may have to start experimenting with quants myself, this is really useful.

Let me know if you do, as you can tell I'm collecting info on that. Also if you do want to easily benchmark and plot performance across your full context window for both TG and PP you can use the sweep-bench example I recently ported over to ik_llama.cpp

> For the compute buffers, would you happen to know what batch/micro batch sizes were set to?

n_batch    = 2048
n_ubatch   = 512

> I’m getting a total of 67GB for 32k context. It would be nice if I could claw back some some how…

I agree, that would be nice, I'm also curious as to why the split-mode row doesn't work. I've never run a setup with it but I've seen other it giving nice performance gains. 

For now I'm still stuck on CPU only, I did work a bit on porting the RPC updates to support it (and other models and cache quantization for models that were already supported) so that I can run hybrid CPU+GPU over RPC but I'm running into issues that I don't really understand.

---

👤 **davidsyoung** commented the **2025-02-28** at **09:32:23**:<br>

> So, based on this discussion, reducing compute buffer size is by far more important than reducing KV cache size. I'll see if I can do something about that.
> 
> > So I'm not too sure what's up with the compute buffer. Maybe this is just the size of it given the size of the model. But allocating 42.8GB per gpu, across 15 gpu's would be 642GB VRAM just for compute buffer.
> 
> Don't think about the fact that there are 15 GPUs. With per layer model split, each GPU needs to compute a full layer, so each GPU needs the exact same compute buffer as if the entire model was running on it (I.e., if you had a single GPU with enough VRAM to fit the entire model, the compute buffer will still be 42.8 GB and not 15 x 42.8 GB).
> 
> Why does the compute buffer become 42.8 GB for 160k context? There is the `K*Q` tensor that needs to materialize. It is of size `n_ctx x n_head x n_ubatch x sizeof(float)` (all compute buffers are `fp32` in `llama.cpp/ggml`). DeepSeek-R1 has 128 heads, so for 160k tokens this tensor alone is 41.9 GB for the default `u_batch` size of 512. It is needed on each GPU because each GPU needs to compute it for the layers stored on it. Your best bet for reducing compute buffer size is to use a smaller `n_ubatch`, but even with `-ub 128` you will not be able to run the full 163k token context. Still, I would be very curious to know how performance with `-ub 128` compares to the default (for a context length that fits in VRAM).
> 
> If you use flash attention (`-fa`), the `K*Q` tensor never materializes, so compute buffers are much smaller. But then the KV cache is much larger. I have been trying to make flash attention work with MLA, but have not been successful so far. Oops, CUDA flash attention does not work for DeepSeek, so that's only useful on the CPU.

That makes way more sense. Thank you. Would split mode row, if it worked, be a solution/help with this?

I tried to look into the assert that came up, but wasn't able to understand to resolve myself.

I however, have tested `-ub 128`, and was able to fit in about 51200 ctx:

```
-b 2048 -ub 256 —ctx-size 51200

llama_kv_cache_init:      CUDA0 KV buffer size =   393.75 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =   225.00 MiB
llama_kv_cache_init:      CUDA2 KV buffer size =   225.00 MiB
llama_kv_cache_init:      CUDA3 KV buffer size =   168.75 MiB
llama_kv_cache_init:      CUDA4 KV buffer size =   225.00 MiB
llama_kv_cache_init:      CUDA5 KV buffer size =   225.00 MiB
llama_kv_cache_init:      CUDA6 KV buffer size =   225.00 MiB
llama_kv_cache_init:      CUDA7 KV buffer size =   225.00 MiB
llama_kv_cache_init:      CUDA8 KV buffer size =   225.00 MiB
llama_kv_cache_init:      CUDA9 KV buffer size =   225.00 MiB
llama_kv_cache_init:     CUDA10 KV buffer size =   225.00 MiB
llama_kv_cache_init:     CUDA11 KV buffer size =   168.75 MiB
llama_kv_cache_init:     CUDA12 KV buffer size =   225.00 MiB
llama_kv_cache_init:     CUDA13 KV buffer size =   225.00 MiB
llama_kv_cache_init:     CUDA14 KV buffer size =   225.00 MiB
llama_new_context_with_model: KV self size  = 3431.25 MiB, c^KV (f16): 3431.25 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.99 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=4)
llama_new_context_with_model:      CUDA0 compute buffer size =  6655.00 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =  6654.50 MiB
llama_new_context_with_model:      CUDA2 compute buffer size =  6654.50 MiB
llama_new_context_with_model:      CUDA3 compute buffer size =  6654.50 MiB
llama_new_context_with_model:      CUDA4 compute buffer size =  6654.50 MiB
llama_new_context_with_model:      CUDA5 compute buffer size =  6654.50 MiB
llama_new_context_with_model:      CUDA6 compute buffer size =  6654.50 MiB
llama_new_context_with_model:      CUDA7 compute buffer size =  6654.50 MiB
llama_new_context_with_model:      CUDA8 compute buffer size =  6654.50 MiB
llama_new_context_with_model:      CUDA9 compute buffer size =  6654.50 MiB
llama_new_context_with_model:     CUDA10 compute buffer size =  6654.50 MiB
llama_new_context_with_model:     CUDA11 compute buffer size =  6654.50 MiB
llama_new_context_with_model:     CUDA12 compute buffer size =  6654.50 MiB
llama_new_context_with_model:     CUDA13 compute buffer size =  6654.50 MiB
llama_new_context_with_model:     CUDA14 compute buffer size =  6654.50 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   103.51 MiB
```

The performance mirrors the previous share above with tokens (with `-ub 256`), the only difference is prefill time has gone down from a max of `116.11 tokens per second` to `73.39 tokens per second`.

So there's a relatively decent drop that does have an impact on usability, but it does unlock 19k~ new max tokens.

Would there be any other optimisation that I could use that would improve the prefill time? Increasing pipeline parallelism, or anything like that? I don't fully understand that to know myself. It doesn't seem to be affected by batch size either.

---

👤 **ikawrakow** commented the **2025-02-28** at **10:12:51**:<br>

> Would there be any other optimisation that I could use that would improve the prefill time? 

Use `-fmoe`. Splitting by row should normally give a decent boost, but that does not work.

If you  change
```
#define IK_PRINT_TIMING 0
```
to
```
#define IK_PRINT_TIMING 1
```
in `ggml-cuda.cu`, rebuild, and run `llama-bench -m model -n 0 -p 512 -t 1 -w 0 -r 2 -fmoe 1 your_tensor_splits >log.out` and send me the output, perhaps I can see where are the major bottlenecks.

---

👤 **davidsyoung** commented the **2025-02-28** at **14:25:11**:<br>

I'm attempting to run llama-bench but it's trying to allocate the full model to device zero, even though I've set tensor splits.

```
-m /models/gghfez_DeepSeek-R1-11446-Q2_K/DeepSeek-R1-11446-Q2_K-00001-of-00030.gguf
      -n 0
      -p 512
      -t 1
      -w 0
      -r 2
      -fmoe 1
      -mla 2
      -ngl 99
      -ts 38,26,24,24,24,24,24,24,24,25,24,24,24,24,33
      -o md
      -v
```

```
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
llm_load_tensors: ggml ctx size =    0.93 MiB
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 232863.17 MiB on device 0: cudaMalloc failed: out of memory
llama_model_load: error loading model: unable to allocate backend buffer
llama_load_model_from_file: failed to load model
main: error: failed to load model '/models/gghfez_DeepSeek-R1-11446-Q2_K/DeepSeek-R1-11446-Q2_K-00001-of-00030.gguf'
| model                          |       size |     params | backend    | ngl | threads | mla | ts           | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --: | ------------ | ---: | ------------: | ---------------: |
```

---

👤 **ikawrakow** commented the **2025-02-28** at **15:51:58**:<br>

Well, not sure why `llama-bench` doesn't do the right thing.

But I think you will like PR #237 very much. Simply add
```
-amb 2048
```
to your command line, and the compute buffers should be no more than 3 GiB even for a context of 163k tokens!

---

👤 **davidsyoung** commented the **2025-02-28** at **16:33:15**:<br>

Holy shit. Will report back!