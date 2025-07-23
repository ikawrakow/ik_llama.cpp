### 📝 [#305](https://github.com/ikawrakow/ik_llama.cpp/issues/305) - Gibberish output when using DeepSeek-V3-0324-IQ2_K_R4 on mixed CPU + 4 GPUs with -mla (1 or 2)

| **Author** | `Panchovix` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-04-01 |
| **Updated** | 2025-04-29 |

---

#### Description

HI there, thanks for your work!

I have found, from this reddit post https://www.reddit.com/r/LocalLLaMA/comments/1joyl9t/new_gguf_quants_of_v30324/, about some new quants of ik_llamacpp

My system consits of a AMD Ryzen 7 7800X3D, 192GB RAM, RTX 5090, RTX 4090x2 and an RTX A6000. OS is Fedora 41.

The model used is https://huggingface.co/ubergarm/DeepSeek-V3-0324-GGUF/tree/main/DeepSeek-V3-0324-IQ2_K_R4

I'm running it with

`/llama-server -m '/DeepSeek-V3-0324-IQ2_K_R4-00001-of-00005.gguf' -c 8192 -ngl 27 -ts 17,20,21,45 --no-warmup -mla 2` (or -mla 1)

I did build ik_llama.cpp with

`cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_CUDA_F16=ON -DGGML_IQK_FA_ALL_QUANTS=1`

The issue seems to be that, when trying to generate with any prompt, the output is gibberish (just DDDDDD)

![Image](https://github.com/user-attachments/assets/960ebb27-9f8b-472a-b8d7-14ad179f1b3d)

Log is this one

```
/build/bin$ ./llama-server -m '/GGUFs/DeepSeek-V3-0324-IQ2_K_R4-00001-of-00005.gguf' -c 8192 -ngl 27 -ts 17,20,21,45 --no-warmup -mla 2
INFO [                    main] build info | tid="140255828869120" timestamp=1743549988 build=3618 commit="6d405d1f"
INFO [                    main] system info | tid="140255828869120" timestamp=1743549988 n_threads=8 n_threads_batch=-1 total_threads=16 system_info="AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: additional 4 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 53 key-value pairs and 1147 tensors from /DeepSeek-V3-0324-IQ2_K_R4-00001-of-00005.gguf (version GGUF V3 (latest))
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
llama_model_loader: - kv  16:                          general.file_type u32              = 338
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
llama_model_loader: - kv  51:                                split.count u16              = 5
llama_model_loader: - kv  52:                        split.tensors.count i32              = 1147
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  612 tensors
llama_model_loader: - type iq2_k_r4:  116 tensors
llama_model_loader: - type iq3_k_r4:   58 tensors
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
llm_load_print_meta: model ftype      = IQ2_K_R4 - 2.375 bpw
llm_load_print_meta: model params     = 672.050 B
llm_load_print_meta: model size       = 226.003 GiB (2.889 BPW) 
llm_load_print_meta: repeating layers = 224.169 GiB (2.873 BPW, 670.196 B parameters)
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
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 4 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
  Device 1: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
  Device 2: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
  Device 3: NVIDIA RTX A6000, compute capability 8.6, VMM: yes
llm_load_tensors: ggml ctx size =    2.34 MiB
llm_load_tensors: offloading 27 repeating layers to GPU
llm_load_tensors: offloaded 27/62 layers to GPU
llm_load_tensors:        CPU buffer size = 46211.13 MiB
llm_load_tensors:        CPU buffer size = 47115.34 MiB
llm_load_tensors:        CPU buffer size = 31151.98 MiB
llm_load_tensors:        CPU buffer size =  4607.07 MiB
llm_load_tensors:      CUDA0 buffer size = 19631.39 MiB
llm_load_tensors:      CUDA1 buffer size = 19631.39 MiB
llm_load_tensors:      CUDA2 buffer size = 23557.67 MiB
llm_load_tensors:      CUDA3 buffer size = 43189.07 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 8192
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: mla_attn   = 2
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 0
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
llama_kv_cache_init:  CUDA_Host KV buffer size =   306.00 MiB
llama_kv_cache_init:      CUDA0 KV buffer size =    45.00 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =    45.00 MiB
llama_kv_cache_init:      CUDA2 KV buffer size =    54.00 MiB
llama_kv_cache_init:      CUDA3 KV buffer size =    99.00 MiB
llama_new_context_with_model: KV self size  =  549.00 MiB, c^KV (f16):  549.00 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.99 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =  2484.78 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =  2491.50 MiB
llama_new_context_with_model:      CUDA2 compute buffer size =  2491.50 MiB
llama_new_context_with_model:      CUDA3 compute buffer size =  2491.50 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =  2634.50 MiB
llama_new_context_with_model: graph nodes  = 3724
llama_new_context_with_model: graph splits = 707
INFO [                    init] initializing slots | tid="140255828869120" timestamp=1743550245 n_slots=1
INFO [                    init] new slot | tid="140255828869120" timestamp=1743550245 id_slot=0 n_ctx_slot=8192
INFO [                    main] model loaded | tid="140255828869120" timestamp=1743550245
INFO [                    main] chat template | tid="140255828869120" timestamp=1743550245 chat_example="You are a helpful assistant\n\n<｜User｜>Hello<｜Assistant｜>Hi there<｜end▁of▁sentence｜><｜User｜>How are you?<｜Assistant｜>" built_in=true
INFO [                    main] HTTP server listening | tid="140255828869120" timestamp=1743550245 n_threads_http="15" port="8080" hostname="127.0.0.1"
INFO [            update_slots] all slots are idle | tid="140255828869120" timestamp=1743550245
INFO [      log_server_request] request | tid="140133399519232" timestamp=1743550253 remote_addr="127.0.0.1" remote_port=51170 status=200 method="GET" path="/" params={}
INFO [      log_server_request] request | tid="140133399519232" timestamp=1743550253 remote_addr="127.0.0.1" remote_port=51170 status=200 method="GET" path="/index.js" params={}
INFO [      log_server_request] request | tid="140133391126528" timestamp=1743550253 remote_addr="127.0.0.1" remote_port=51186 status=200 method="GET" path="/completion.js" params={}
INFO [      log_server_request] request | tid="140133399519232" timestamp=1743550253 remote_addr="127.0.0.1" remote_port=51170 status=200 method="GET" path="/json-schema-to-grammar.mjs" params={}
INFO [      log_server_request] request | tid="140133399519232" timestamp=1743550254 remote_addr="127.0.0.1" remote_port=51170 status=404 method="GET" path="/favicon.ico" params={}
INFO [      log_server_request] request | tid="140133307248640" timestamp=1743550263 remote_addr="127.0.0.1" remote_port=33660 status=200 method="GET" path="/index-new.html" params={}
INFO [      log_server_request] request | tid="140133307248640" timestamp=1743550263 remote_addr="127.0.0.1" remote_port=33660 status=200 method="GET" path="/style.css" params={}
INFO [      log_server_request] request | tid="140133298855936" timestamp=1743550263 remote_addr="127.0.0.1" remote_port=33670 status=200 method="GET" path="/index.js" params={}
INFO [      log_server_request] request | tid="140133290463232" timestamp=1743550263 remote_addr="127.0.0.1" remote_port=33686 status=200 method="GET" path="/completion.js" params={}
INFO [      log_server_request] request | tid="140133282070528" timestamp=1743550263 remote_addr="127.0.0.1" remote_port=33696 status=200 method="GET" path="/json-schema-to-grammar.mjs" params={}
INFO [      log_server_request] request | tid="140133273677824" timestamp=1743550263 remote_addr="127.0.0.1" remote_port=33704 status=200 method="GET" path="/prompt-formats.js" params={}
INFO [      log_server_request] request | tid="140133265285120" timestamp=1743550263 remote_addr="127.0.0.1" remote_port=33718 status=200 method="GET" path="/system-prompts.js" params={}
INFO [      log_server_request] request | tid="140133307248640" timestamp=1743550263 remote_addr="127.0.0.1" remote_port=33660 status=200 method="GET" path="/colorthemes.css" params={}
INFO [      log_server_request] request | tid="140133307248640" timestamp=1743550263 remote_addr="127.0.0.1" remote_port=33660 status=200 method="GET" path="/theme-snowstorm.css" params={}
INFO [      log_server_request] request | tid="140133273677824" timestamp=1743550263 remote_addr="127.0.0.1" remote_port=33704 status=200 method="GET" path="/theme-polarnight.css" params={}
INFO [      log_server_request] request | tid="140133290463232" timestamp=1743550263 remote_addr="127.0.0.1" remote_port=33686 status=200 method="GET" path="/theme-ketivah.css" params={}
INFO [      log_server_request] request | tid="140133298855936" timestamp=1743550263 remote_addr="127.0.0.1" remote_port=33670 status=200 method="GET" path="/theme-mangotango.css" params={}
INFO [      log_server_request] request | tid="140133265285120" timestamp=1743550263 remote_addr="127.0.0.1" remote_port=33718 status=200 method="GET" path="/theme-playground.css" params={}
INFO [      log_server_request] request | tid="140133282070528" timestamp=1743550263 remote_addr="127.0.0.1" remote_port=33696 status=200 method="GET" path="/theme-beeninorder.css" params={}
INFO [      log_server_request] request | tid="140133282070528" timestamp=1743550267 remote_addr="127.0.0.1" remote_port=33696 status=200 method="GET" path="/" params={}
INFO [      log_server_request] request | tid="140133282070528" timestamp=1743550267 remote_addr="127.0.0.1" remote_port=33696 status=200 method="GET" path="/index.js" params={}
INFO [      log_server_request] request | tid="140133307248640" timestamp=1743550267 remote_addr="127.0.0.1" remote_port=33660 status=200 method="GET" path="/completion.js" params={}
INFO [      log_server_request] request | tid="140133273677824" timestamp=1743550267 remote_addr="127.0.0.1" remote_port=33704 status=200 method="GET" path="/json-schema-to-grammar.mjs" params={}
INFO [   launch_slot_with_task] slot is processing task | tid="140255828869120" timestamp=1743550272 id_slot=0 id_task=0
INFO [            update_slots] kv cache rm [p0, end) | tid="140255828869120" timestamp=1743550273 id_slot=0 id_task=0 p0=0
```

Maybe I'm using the flag incorrectly, or I didn't build ik_llama.cpp correctly?

When not using -mla, model seems to work normally, abeit slower than UD_Q2_K_XL (https://huggingface.co/unsloth/DeepSeek-V3-0324-GGUF/tree/main/UD-Q2_K_XL)

EDIT: To note that other models have the same issue (like the mentioned above), but those probably aren't expected to work since they aren't quanted with ik_llama.cpp

---

#### 💬 Conversation

👤 **saood06** commented the **2025-04-01** at **23:49:47**:<br>

I'm not sure why your getting bad output, but you might want to look into https://github.com/ikawrakow/ik_llama.cpp/pull/232 instead of just setting `-ngl` this is more tested and offers much higher performance.

More info about using it here: https://github.com/ikawrakow/ik_llama.cpp/discussions/258

---

👤 **Panchovix** commented the **2025-04-02** at **00:02:11**:<br>

@saood06 Thanks for the suggestion! I did see the post but not sure how to exactly use it, because it seems to use it on a single GPU for all the layers, but on my case I'm using 27 layers of 61 and multiGPU, not sure how to adapt it.

I also did try with -mla 2 and -fa but same issue.

I will try to rebuild with `cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_IQK_FA_ALL_QUANTS=1 -DGGML_BLAS=OFF` to see if it helps.

---

👤 **Panchovix** commented the **2025-04-02** at **00:02:11**:<br>

@saood06 Thanks for the suggestion! I did see the post but not sure how to exactly use it, because it seems to use it on a single GPU, it is a bit easier, but not sure how to adapt it to multiGPU.

I also did try with -mla 2 and -fa but same issue.

I will try to rebuild with `cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_IQK_FA_ALL_QUANTS=1 -DGGML_BLAS=OFF` to see if it helps.

---

👤 **saood06** commented the **2025-04-02** at **00:14:35**:<br>

> [@saood06](https://github.com/saood06) Thanks for the suggestion! I did see the post but not sure how to exactly use it, because it seems to use it on a single GPU for all the layers, but on my case I'm using 27 layers of 61 and multiGPU, not sure how to adapt it.

The goal is to use your CPU for the large pool of experts, and use the GPU for the rest. I'm not sure if the code as it is currently benefits from using more than one GPU though.

See https://github.com/ikawrakow/ik_llama.cpp/discussions/242#discussioncomment-12452986 for someone else using `-ot` and spreading those tensors across multiple GPU's (and his edit showing only one GPU was active).

---

👤 **Panchovix** commented the **2025-04-02** at **00:30:55**:<br>

@saood06 

Oh just saw it, seems interesting! Probably will take a while to understand it, since I read it a bit lightly and didn't understand much.

Now, still no luck with `cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_IQK_FA_ALL_QUANTS=1 -DGGML_BLAS=OFF`, so not sure if I'm missing something else :(

---

👤 **saood06** commented the **2025-04-02** at **00:34:10**:<br>

> [@saood06](https://github.com/saood06)
> 
> Oh just saw it, seems interesting! Probably will take a while to understand it, since I read it a bit lightly and didn't understand much.
> 

Let me know if you have any more questions, my Deepseek machine doesn't have a GPU (let alone multiple) so I can't test, but I have seen others have good success with `-ot`

---

👤 **ikawrakow** commented the **2025-04-02** at **05:21:58**:<br>

This model is not ideal for your multi-GPU setup. The row-interleaved quants (`X_R4, X_R8`) are best for CPU-only inference. They do not have CUDA matrix multiplication implementation, so all matrix multiplications involving them will run on the CPU, so yes, it will be slower (and your GPU's will be acting as very expensive RAM modules for your CPU).

But apart from this it still should work, and I don't know why t doesn't. I ended up in a situation where all users coming here want to use `ik_llama.cpp` for DeepSeek-V3/R1, but I don't have the hardware to run these models, so not able to debug when issues arise.

> EDIT: To note that other models have the same issue (like the mentioned above), but those probably aren't expected to work since they aren't quanted with ik_llama.cpp

All supported models working with mainline `llama.cpp` are supposed to work also here.

---

👤 **saood06** commented the **2025-04-02** at **05:30:56**:<br>

> This model is not ideal for your multi-GPU setup. The row-interleaved quants (`X_R4, X_R8`) are best for CPU-only inference. They do not have CUDA matrix multiplication implementation, so all matrix multiplications involving them will run on the CPU, so yes, it will be slower (and your GPU's will be acting as very expensive RAM modules for your CPU).
> 

It has `llama_model_loader: - type q8_0:  612 tensors`, this is ubergarm's mix where those are on the tensors that are better suited for GPU.

So if he uses -ot then he will be able to offload all those to GPU(s), leaving just the row-interleaved quants to the CPU

---

👤 **saood06** commented the **2025-04-02** at **05:30:56**:<br>

> This model is not ideal for your multi-GPU setup. The row-interleaved quants (`X_R4, X_R8`) are best for CPU-only inference. They do not have CUDA matrix multiplication implementation, so all matrix multiplications involving them will run on the CPU, so yes, it will be slower (and your GPU's will be acting as very expensive RAM modules for your CPU).
> 

It has `llama_model_loader: - type q8_0:  612 tensors`, this is ubergarm's mix where those are on the tensors that are better suited for GPU.

So if he uses -ot then he will be able to offload all those to GPU(s).

---

👤 **ikawrakow** commented the **2025-04-02** at **05:36:41**:<br>

> So if he uses -ot then he will be able to offload all those to GPU(s), leaving just the row-interleaved quants to the CPU

Yes, that's true. But that way they will be using a small fraction of the 120 GB VRAM available.

---

👤 **saood06** commented the **2025-04-02** at **05:53:21**:<br>

> Yes, that's true. But that way they will be using a small fraction of the 120 GB VRAM available.

In the linked discussion the commenter was never able to get more than one GPU to be active, has that been fixed? "Main GPU usage is at 25% and other cards are at 0% when generating." and "When removing -fmoe, the GPU usage is still centralized on the main GPU, with 20-25% usage at 130-140w, while the other cards stay at 0% at ~100w."

---

👤 **ikawrakow** commented the **2025-04-02** at **05:59:32**:<br>

If you have been using UD_Q2_K_XL, try running it with this fork the same way you have in mainline, but add
```
-fmoe -rtr -ub 2048
```
to your server command line. Loading the model will take longer, but then inference will be hopefully faster. The `-ub 2048` option will only have impact on prompt processing speed, so if TG is your main use case, you may leave it out. 

As @saood06 suggested, for best performance you should experiment with tensor overrides (`-ot`). Ideally, all attention and all shared experts should run on the GPUs. Then make use of the remaining VRAM to offload as many MoE tensors as will fit to the GPUs. It may be better to have all `ffn_down_exps` tensors left on the CPU, and instead have more of `ffn_up_exps` and `ffn_gate_exps` offloaded to the GPU. Example:
```
/llama-server -m '/DeepSeek-V3-0324-IQ2_K_R4-00001-of-00005.gguf' -c 8192 -ngl 100 -ts 17,20,21,45 \
-ot "ffn_down_exps=CPU,blk\.4[0-9]\.ffn_.*_exps=CPU,blk\.5[0-9]\.ffn_.*_exps=CPU,blk\.6[0-9]\.ffn_.*_exps=CPU"
```
will have all `ffn_down_exps` tensors and the `ffn_up/gate_exps` for layers 40-end on the CPU, everything else on the GPUs. If that does not fit, or if you have unused VRAM left, you can modify the regex to keep a different number of `ffn_up/gate_exps` tensors on the CPU.

---

👤 **ikawrakow** commented the **2025-04-02** at **06:06:28**:<br>

> In the linked discussion the commenter was never able to get more than one GPU to be active, has that been fixed? 

I remember #242, but I don't have multiple GPUs to understand why the issue occurs. Apart from this, @davidsyoung has been using it with 16 x 3090, and I do not recall him reporting that only one GPU is being used.

---

👤 **ikawrakow** commented the **2025-04-02** at **06:06:28**:<br>

> In the linked discussion the commenter was never able to get more than one GPU to be active, has that been fixed? 

I remember #242, but I don't have multiple GPUs to understand why the issue occurs. Apart from this, @davidsyoung has bee using it with 16 x 3090, and I do not recall him reporting that only one GPU is being used.

---

👤 **saood06** commented the **2025-04-02** at **06:14:41**:<br>

> I remember [#242](https://github.com/ikawrakow/ik_llama.cpp/discussions/242), but I don't have multiple GPUs to understand why the issue occurs. Apart from this, [@davidsyoung](https://github.com/davidsyoung) has been using it with 16 x 3090, and I do not recall him reporting that only one GPU is being used.

Yes but maybe it is different if it offloaded fully to CUDA, because ThomasBaruzier's who had the issue his comments are at a time when davidsyoung was using ik_llama.cpp. Maybe @Panchovix you can tell us if all GPU's are being used  when putting tensors on all of them with -ot.

---

👤 **saood06** commented the **2025-04-02** at **06:14:41**:<br>

> I remember [#242](https://github.com/ikawrakow/ik_llama.cpp/discussions/242), but I don't have multiple GPUs to understand why the issue occurs. Apart from this, [@davidsyoung](https://github.com/davidsyoung) has been using it with 16 x 3090, and I do not recall him reporting that only one GPU is being used.

Yes but maybe it is different if it offloaded fully to CUDA, because ThomasBaruzier's who had the issue his comments are at a time when davidsyoung was using ik_llama.cpp.

---

👤 **davidsyoung** commented the **2025-04-02** at **06:43:15**:<br>

Hey just wanted to jump in as tagged above. 

I never had an issue personally while using with all GPUs being used, but it’s going to be dependent on how tensors/attention is being balanced across GPUs. 

I didn’t have a mixed workflow of CPU/GPU offload like this, but if I was debugging I would go the route of what @ikawrakow is suggesting. 

I would also likely just to start, use a less exotic quantisation to rule that out. As you’re doing a mixed offload of GPU/CPU, I would use a standard Q4 quant. 

Then from there, I would use -ot commands like suggested above. 

Lower down the list of possibilities could be the -mla option you’re using, as it’s possible that combination of mixed offload, quant format, and those commands possibly haven’t been tested too heavily. 

It may also just simply be the model with Q2 quant.

Process of elimination!

---

👤 **davidsyoung** commented the **2025-04-02** at **06:43:15**:<br>

Hey just wanted to jump in as tagged above. 

I never had an issue personally while using with all GPUs being used, but it’s going to be dependent on how GPUs are being balanced across GPUs. 

I didn’t have a mixed workflow of CPU/GPU offload like this, but if I was debugging I would go the route of what @ikawrakow is suggesting. 

I would also likely just to start, use a less exotic quantisation to rule that out. As you’re doing a mixed offload of GPU/CPU, I would use a standard Q4 quant. 

Then from there, I would use -ot commands like suggested above. 

Lower down the list of possibilities could be the -mla option you’re using, as it’s possible that combination of mixed offload, quant format, and those commands possibly haven’t been tested too heavily. 

It may also just simply be the model with Q2 quant.

Process of elimination!

---

👤 **Panchovix** commented the **2025-04-02** at **11:40:17**:<br>

Hi there guys, just woke up and saw all the new information, many thanks! I will try the suggestions when I come home after work (in about 11 hours).

I will try some normal quants (123B at Q4_M_K or Q6_K). If those aren't quanted with ik_llamacpp, would they work with -mla 2?
I will try these with both full GPU and CPU + GPU.

From my understanding -ot may result in better performance but not address the gibberish output when using MLA 2 right? But even then, if not using MLA it will probably help (I can take 3-4 t/s while generating, but pre processing is really slow, it takes some minutes to start to generate. Probably because slow RAM (5600Mhz) and PCI-E speeds (X16/X4/X4/X4))

---

👤 **Panchovix** commented the **2025-04-02** at **16:35:44**:<br>

I did try a little via RDP (on Windows though, as I haven't managed to get a RDP client working unattended on Linux)

With `-fmoe -rtr -ub 2048` I get CUDA OOM (with stock -c 8192 -ngl 22 -ts 17,20,21,45 --no-warmup.) Without -ub it loads but it seems to use shared RAM into the GPUs, so it never gens. With just -fmoe it seems to work normally.

With `-ot "ffn_down_exps=CPU,blk\.4[0-9]\.ffn_.*_exps=CPU,blk\.5[0-9]\.ffn_.*_exps=CPU,blk\.6[0-9]\.ffn_.*_exps=CPU"` I also get CUDA OOM (with `-c 8192 -ngl 100 -ts 17,20,21,45`)

With the mix of the 2, I also get OOM.

I will try later on Linux to see how it behaves.

Probably I really don't know how to set up the -ot values and/or what does rtr with ub do. I'm new on llamacpp as previously I mostly used other backends with only GPU support and not shared CPU+GPU, so pardon me for my ignorance.

---

👤 **ubergarm** commented the **2025-04-02** at **22:07:04**:<br>

> I will try later on Linux to see how it behaves.

Heya @Panchovix , glad you found my reddit post and tried the model. I updated the model card to hopefully explain better that those quants are specifically designed for 24-48GB VRAM systems.

You, however, have a more complex setup with multiple GPUs. 

I would suggest that you start small and try to get some success using just a single GPU for example. Then as you learn, move up to bigger models and offload layers as you desire learning how to use `-ot`. I have [a rough guide](https://github.com/ikawrakow/ik_llama.cpp/discussions/258) with links to discussions for some of the features of this fork, this might be a good place to start digging in and learning.

> I will try some normal quants (123B at Q4_M_K or Q6_K). If those aren't quanted with ik_llamacpp, would they work with -mla 2?

As mentioned: "All supported models working with mainline llama.cpp are supposed to work also here."  So `ik_llama.cpp` can generate MLA tensors on the fly models that support MLA e.g. R1 and V3 models at least. I'm not sure what 123B model you are talking about though, if you mean Mistral-Large-Instruct, I don't think that default model architecture supports MLA.

Its possible I saw you over on level1techs forum too, feel free to reach out to me there if you want some help getting started. Cheers!

---

👤 **Panchovix** commented the **2025-04-02** at **22:56:11**:<br>

@ubergarm Thanks! But I think the model won't fit on 192GB RAM + 48GB RAM? Correct me if I'm wrong though. I will checkout the guide!

I will install Debian for a bit more stability than Fedora (though, Debian 13) and will try.

And ah I see, I was thinking of Mistral yeah, but it makes sense only Deepseek supports MLA.

I think I went some time ago on level1techs, but never went much anymore because it is too advanced for me haha.

---

👤 **ubergarm** commented the **2025-04-03** at **00:42:00**:<br>

>  192GB RAM + 48GB RAM

So the `IQ2_K_R4` is 226 GiB, of which about 17.33 GiB are layers designed for offload to GPU so that leaves ~210GiB for RAM. So technically it would work okay and probably get you 2-3 tok/sec because `mmap()` can leave some weights on disk and still run out of page cache. You could use the command I provided on the huggingface model card just for a test.

> I think I went some time ago on level1techs

Oops nope, there is a different person over there asking about using multiple GPUs like this! Thanks!

---

👤 **whatever1983** commented the **2025-04-15** at **19:43:49**:<br>

@ubergarm and @ikawrakow 
embedding needs to be iq3_k to emulate IQ2_M for way better coding performance.  ikawrakow, can you make that into the IQ2_K_M, IQ2_K_M_R4 standard?

---

👤 **ubergarm** commented the **2025-04-15** at **20:33:00**:<br>

@whatever1983 

> embedding needs to be iq3_k to emulate IQ2_M for way better coding performance.

Hey bud, which `embedding` are you talking about? If you check the model card side-bar on hf for the [DeepSeek-V3-0324-IQ2_K_R4](https://huggingface.co/ubergarm/DeepSeek-V3-0324-GGUF?show_file_info=DeepSeek-V3-0324-IQ2_K_R4%2FDeepSeek-V3-0324-IQ2_K_R4-00001-of-00005.gguf) (about which I assume you are referring?), the `token_embd.weight` is `q8_0`? 

> can you make that into the IQ2_K_M, IQ2_K_M_R4 standard?

This fork allows the user to cook up whatever combinations they want with `llama-quantize --quantize-q` ... (and my recipe is shown on the hf model card too). I'm not sure where you're talking about `IQ2_K_M` or `IQ2_K_M_R4` those are not quants with which I'm familiar. You can see the [quants available listed in the `quantize` code here](https://github.com/ikawrakow/ik_llama.cpp/blob/main/examples/quantize/quantize.cpp#L26).

Sorry I'm confused, if you have a specific reference to the exact quant in question I'll be back in office later this week. Cheers!

---

👤 **ubergarm** commented the **2025-04-15** at **20:33:00**:<br>

> embedding needs to be iq3_k to emulate IQ2_M for way better coding performance.

Hey bud, which `embedding` are you talking about? If you check the model card side-bar on hf for the [DeepSeek-V3-0324-IQ2_K_R4](https://huggingface.co/ubergarm/DeepSeek-V3-0324-GGUF?show_file_info=DeepSeek-V3-0324-IQ2_K_R4%2FDeepSeek-V3-0324-IQ2_K_R4-00001-of-00005.gguf) (about which I assume you are referring?), the `token_embd.weight` is `q8_0`? 

> can you make that into the IQ2_K_M, IQ2_K_M_R4 standard?

This fork allows the user to cook up whatever combinations they want with `llama-quantize --quantize-q` ... (and my recipe is shown on the hf model card too). I'm not sure where you're talking about `IQ2_K_M` or `IQ2_K_M_R4` those are not quants with which I'm familiar. You can see the [quants available listed in the `quantize` code here](https://github.com/ikawrakow/ik_llama.cpp/blob/main/examples/quantize/quantize.cpp#L26).

Sorry I'm confused, if you have a specific reference to the exact quant in question I'll be back in office later this week. Cheers!

---

👤 **Panchovix** commented the **2025-04-24** at **05:24:37**:<br>

HI there! Closing as MLA was recently merged into main llamacpp, and it seems to work with CUDA as for now, with newer quants (https://huggingface.co/unsloth/DeepSeek-V3-0324-GGUF-UD)

Many thanks for all the info!

EDIT: Re-opening as no luck for now either on main llamacpp

---

👤 **ubergarm** commented the **2025-04-26** at **19:13:31**:<br>

@Panchovix 

I see you over on https://huggingface.co/unsloth/DeepSeek-V3-0324-GGUF-UD/discussions/2#680d23a4806b446ebce4d723

Did you ever try running this fork with my one of [my quants](https://huggingface.co/ubergarm/DeepSeek-V3-0324-GGUF) using the provided commands on the model card?

Just to start out don't use all your GPUs at once, just use one of them and use `-ot exps=CPU` to see if you get it working.

Once you have something working, then you can figure out how to to optimize for all your GPUs. 

I have not tried that new "Unsloth Dynamic v2.0" quant with MLA, and am not sure how they even generated the [imatrix given bartowski was having issues with that](https://github.com/ggml-org/llama.cpp/pull/12801#issuecomment-2824767949) which commented on further down.

---

👤 **Panchovix** commented the **2025-04-26** at **20:47:26**:<br>

Hi there @ubergarm, I did try IQ2_K_R4, but with multiple GPUs. The issue is that with just one GPU, I tried but the model didn't fit with RAM + VRAM (In theory it should but it gave me OOM anywayas).

As mentioned there, on llamacpp the error seems a bit different, outputing gibberish at a bit larger contexts but starts fine, while with R4 I get just "DDDDD" on any ctx.

---

👤 **ubergarm** commented the **2025-04-27** at **01:38:32**:<br>

@Panchovix 

Right I think you were trying to run `_R4` quants on a GPU (because you were trying to use `-ngl` without `-ot`) when they are designed only for CPU inference psure.

Give this a try:
```
# Install build dependencies and cuda toolkit as needed
git clone https://github.com/ikawrakow/ik_llama.cpp
cd ik_llama.cpp

# Configure CUDA+CPU Backend
cmake -B ./build -DGGML_CUDA=ON -DGGML_BLAS=OFF
# Build
cmake --build ./build --config Release -j $(nproc)

# Confirm
./build/bin/llama-server --version
version: 3640 (xxxxxxxx)
built with cc (GCC) 14.2.1 20250128 for x86_64-pc-linux-gnu

# API Server using single GPU running out of mmap() only needs >~64GB RAM
CUDA_VISIBLE_DEVICES="0" \
./build/bin/llama-server \
    --model ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-IQ2_K_R4.gguf \
    --alias ubergarm/DeepSeek-R1-V3-0324-IQ2_K_R4 \
    --ctx-size 16384 \
    -ctk f16 \
    -mla 2 -fa \
    -amb 512 \
    -fmoe \
    --temp 0.3 \
    --min-p 0.05 \
    --n-gpu-layers 63 \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 16 \
    --host 127.0.0.1 \
    --port 8080
```

You can also try the various unsloth/bartowski/mradermacher quants (though I've not tested unsloth's new MLA quant on mainline `llama.cpp` nor this `ik_llama.cpp` fork... You just can't use `-rtr` with those as that would disable `mmap` and likely OOM you.

Let me know what errors you get if any trying it this way. If you are still OOMing what is the output of `sudo dmesg -T | grep -i oom` or similar... Thanks!

---

👤 **ubergarm** commented the **2025-04-27** at **01:38:32**:<br>

@Panchovix 

Give this a try:
```
# Install build dependencies and cuda toolkit as needed
git clone https://github.com/ikawrakow/ik_llama.cpp
cd ik_llama.cpp

# Configure CUDA+CPU Backend
cmake -B ./build -DGGML_CUDA=ON -DGGML_BLAS=OFF
# Build
cmake --build ./build --config Release -j $(nproc)

# Confirm
./build/bin/llama-server --version
version: 3640 (xxxxxxxx)
built with cc (GCC) 14.2.1 20250128 for x86_64-pc-linux-gnu

# API Server using single GPU running out of mmap() only needs >~64GB RAM
CUDA_VISIBLE_DEVICES="0" \
./build/bin/llama-server \
    --model ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-IQ2_K_R4.gguf \
    --alias ubergarm/DeepSeek-R1-V3-0324-IQ2_K_R4 \
    --ctx-size 16384 \
    -ctk f16 \
    -mla 2 -fa \
    -amb 512 \
    -fmoe \
    --temp 0.3 \
    --min-p 0.05 \
    --n-gpu-layers 63 \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 16 \
    --host 127.0.0.1 \
    --port 8080
```

You can also try the various unsloth quants (though I've not tested their new MLA quant on mainline `llama.cpp` nor this `ik_llama.cpp` fork... You just can't use `-rtr` as that would disable `mmap` and likely OOM you.

Let me know what errors you get if any trying it this way. If you are still OOMing what is the output of `sudo dmesg -T | grep -i oom` or similar... Thanks!

---

👤 **Panchovix** commented the **2025-04-28** at **19:48:30**:<br>

Sorry for the delay, haven't tested yet as I was trying with normal llamacpp to see how it behaves.

I have a question, though, as -ot seems to be pretty poluar now. Sorry if it's a too novice question.

How can I know the layers, the experts, the size of the experts and such to try to use -ot? For example, Since DeepSeek V3 0324 is 685B, I "assume" active params are 38B. Then, it is each expert 38B as well? Then, for example, IQ2_K_XL/IQ2_K_R4, the size of each of those is about is 2.71 bpw. So, each expert would be 2.71bpw size of each expert?

---

👤 **Panchovix** commented the **2025-04-29** at **04:34:52**:<br>

Just an small update, found https://huggingface.co/unsloth/DeepSeek-V3-0324-GGUF-UD/discussions/2#680fad80e3c723c4b1f20c63, then I tested https://huggingface.co/unsloth/DeepSeek-V3-0324-GGUF-UD/discussions/2#681047075bb07c42d7e44256

The suspicious that -ot would work, it does.

If you load the active parameters all on GPU and then offload the experts to CPU and some CUDA devices, it works fine.

The moment you load the active parameters with mixed CPU + CUDA, it outputs gibberish.

Same seems to happen here with IQ2_K_R4.

So it is maybe resolved? But the issue seems to come when MLA is mixed with active parameters + mixed CPU/GPU.

---

👤 **ubergarm** commented the **2025-04-29** at **16:14:26**:<br>

> How can I know the layers, the experts, the size of the experts and such to try to use -ot?

The easiest way is to look at the hugging face model card sidebar e.g. for [bartowski/THUDM_GLM-4-32B-0414-GGUF/THUDM_GLM-4-32B-0414-Q4_K_M.gguf](https://huggingface.co/bartowski/THUDM_GLM-4-32B-0414-GGUF?show_file_info=THUDM_GLM-4-32B-0414-Q4_K_M.gguf)

This does not show everything for ik_llama.cpp exclusive quants e.g. `iq4_k` as hugging face doesn't fully support those. 

The longer answer is that this is the output you get from `./gguf-py/scripts/gguf_dump.py`

---

👤 **ubergarm** commented the **2025-04-29** at **16:15:40**:<br>

> Same seems to happen here with IQ2_K_R4.

Don't run any `_R4` quants on GPU. Those are repacked for CPU use.

---

👤 **Panchovix** commented the **2025-04-29** at **16:31:00**:<br>

Noted, many thanks for all the help! Closing the issue.

---

👤 **ubergarm** commented the **2025-04-29** at **19:01:45**:<br>

> Noted, many thanks for all the help! Closing the issue.

You have a unique rig, 4x GPUs and 4x DIMMs in what I understand to be a gamer class AM5 MoBo. You can get good performance out of that, but it will require more complex consideration.

Keeps us posted on your progress and benchmarks as you progress in your journey!

---

👤 **Panchovix** commented the **2025-04-29** at **19:18:05**:<br>

Thanks! Yeah, I have 2 motherboards, a X670E Aorus Master and a X670 MSI Carbon, but using the latter now as it lets me use 4x48GB at 6000Mhz.

At some point I want to change to a threadripper/epyc processor.