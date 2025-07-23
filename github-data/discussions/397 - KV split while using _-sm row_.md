### 🗣️ [#397](https://github.com/ikawrakow/ik_llama.cpp/discussions/397) - KV split while using `-sm row`

| **Author** | `pt13762104` |
| :--- | :--- |
| **Created** | 2025-05-08 |
| **Updated** | 2025-05-08 |

---

#### Description

I have found that ik_llama.cpp does NOT support kv-split while using `-sm row`, which is a limitation compared to llama.cpp. Is there any way to do this or it's just not implemented yet?
Example output:
```
INFO [                    main] build info | tid="137884088823808" timestamp=1746690385 build=3673 commit="4084ca73"
INFO [                    main] system info | tid="137884088823808" timestamp=1746690385 n_threads=2 n_threads_batch=-1 total_threads=4 system_info="AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: loaded meta data with 32 key-value pairs and 707 tensors from /root/Qwen3-32B-UD-Q5_K_XL.gguf (version GGUF V3 (latest))
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
llama_model_loader: - kv  27:                          general.file_type u32              = 17
llama_model_loader: - kv  28:                      quantize.imatrix.file str              = Qwen3-32B-GGUF/imatrix_unsloth.dat
llama_model_loader: - kv  29:                   quantize.imatrix.dataset str              = unsloth_calibration_Qwen3-32B.txt
llama_model_loader: - kv  30:             quantize.imatrix.entries_count i32              = 448
llama_model_loader: - kv  31:              quantize.imatrix.chunks_count i32              = 32
llama_model_loader: - type  f32:  257 tensors
llama_model_loader: - type q4_K:   28 tensors
llama_model_loader: - type q5_K:  300 tensors
llama_model_loader: - type q6_K:  122 tensors
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
llm_load_print_meta: model ftype      = Q5_K - Medium
llm_load_print_meta: model params     = 32.762 B
llm_load_print_meta: model size       = 21.603 GiB (5.664 BPW) 
llm_load_print_meta: repeating layers = 20.510 GiB (5.646 BPW, 31.206 B parameters)
llm_load_print_meta: general.name     = Qwen3-32B
llm_load_print_meta: BOS token        = 11 ','
llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
llm_load_print_meta: PAD token        = 151654 '<|vision_pad|>'
llm_load_print_meta: LF token         = 148848 'ÄĬ'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: Tesla T4, compute capability 7.5, VMM: yes
  Device 1: Tesla T4, compute capability 7.5, VMM: yes
llm_load_tensors: ggml ctx size =    0.95 MiB
llm_load_tensors: offloading 64 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 65/65 layers to GPU
llm_load_tensors: CUDA_Split buffer size = 21608.65 MiB
llm_load_tensors:        CPU buffer size =   510.04 MiB
llm_load_tensors:      CUDA0 buffer size =     2.58 MiB
..................................................................................................
llama_new_context_with_model: n_ctx      = 8192
llama_new_context_with_model: n_batch    = 4096
llama_new_context_with_model: n_ubatch   = 1024
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 0
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:      CUDA0 KV buffer size =  2048.00 MiB # where is CUDA1?
llama_new_context_with_model: KV self size  = 2048.00 MiB, K (f16): 1024.00 MiB, V (f16): 1024.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     1.16 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =   633.50 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =    52.01 MiB
llama_new_context_with_model: graph nodes  = 1734
llama_new_context_with_model: graph splits = 2
INFO [                    init] initializing slots | tid="137884088823808" timestamp=1746690394 n_slots=1
INFO [                    init] new slot | tid="137884088823808" timestamp=1746690394 id_slot=0 n_ctx_slot=8192
INFO [                    main] model loaded | tid="137884088823808" timestamp=1746690394
INFO [                    main] chat template | tid="137884088823808" timestamp=1746690394 chat_example="<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there<|im_end|>\n<|im_start|>user\nHow are you?<|im_end|>\n<|im_start|>assistant\n" built_in=true
INFO [                    main] HTTP server listening | tid="137884088823808" timestamp=1746690394 n_threads_http="3" port="8080" hostname="127.0.0.1"
INFO [            update_slots] all slots are idle | tid="137884088823808" timestamp=1746690394
^C
INFO [            update_slots] all slots are idle | tid="137884088823808" timestamp=1746690402
```

---

#### 🗣️ Discussion

👤 **ikawrakow** replied the **2025-05-08** at **08:08:16**:<br>

I have never looked into splitting the KV cache when using `-sm row`, so the behavior is whatever the behavior of `llama.cpp` was when I forked last year.

Out of curiosity: does `-sm row` give you a better performance compared to `-sm layer` ?

> 👤 **pt13762104** replied the **2025-05-08** at **08:36:42**:<br>
> Yes. About 1.5x better