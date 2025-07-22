### 🐛 [#596](https://github.com/ikawrakow/ik_llama.cpp/issues/596) - Bug: Lastest commit broke llama-cli on Windows - mmq.cuh:107: fatal error

| **Author** | `Thireus` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-07-10 |
| **Updated** | 2025-07-13 |

---

#### Description

### What happened?

Some changes made in commit 283753cabcabd30eb2cfb93739d9c1679200bf1f are causing llama-cli to crash. Which wasn't happening before this commit.

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2,1 llama-cli -m DeepSeek-R1-0528-THIREUS.gguf  -mla 3 -fa   -amb 1024   -fmoe   -ctk f16   -c 16384   -ngl 99   -ot "blk\.(3|4|5|6)\.ffn_.*=CUDA0" -ot "blk\.(7|8|9)\.ffn_.*=CUDA1" -ot "blk\.(10|11|12)\.ffn_.*=CUDA2"   -ot exps=CPU   -b 4096 -ub 4096   --warmup-batch   --no-mmap   --threads 36   --main-gpu 0   -p '<｜begin▁of▁sentence｜><｜User｜>What is the solution of x+5=-2?<｜Assistant｜><think>\n'
```

### Name and Version

https://github.com/ikawrakow/ik_llama.cpp/commit/283753cabcabd30eb2cfb93739d9c1679200bf1f#diff-f591a6af9587b282030c7387e32a880973e68370ee6ee3918bd5cd008d1fb89d

### What operating system are you seeing the problem on?

Windows

### Relevant log output

```shell
...
Tensor blk.60.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size = 210434.50 MiB
llm_load_tensors:  CUDA_Host buffer size =   938.98 MiB
llm_load_tensors:      CUDA0 buffer size = 17242.99 MiB
llm_load_tensors:      CUDA1 buffer size = 12195.88 MiB
llm_load_tensors:      CUDA2 buffer size = 14471.99 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 16384
llama_new_context_with_model: n_batch    = 4096
llama_new_context_with_model: n_ubatch   = 4096
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 3
llama_new_context_with_model: attn_max_b = 1024
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init:      CUDA0 KV buffer size =   450.00 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =   342.00 MiB
llama_kv_cache_init:      CUDA2 KV buffer size =   306.00 MiB
llama_new_context_with_model: KV self size  = 1098.00 MiB, c^KV (f16): 1098.00 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.49 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      CUDA0 compute buffer size =  4496.02 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =  3152.02 MiB
llama_new_context_with_model:      CUDA2 compute buffer size =  3152.03 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   368.05 MiB
llama_new_context_with_model: graph nodes  = 4202
llama_new_context_with_model: graph splits = 178

system_info: n_threads = 36 / 36 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 |
sampling:
        repeat_last_n = 64, repeat_penalty = 1.000, frequency_penalty = 0.000, presence_penalty = 0.000
        top_k = 40, tfs_z = 1.000, top_p = 0.950, min_p = 0.050, typical_p = 1.000, temp = 0.800
        mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000
        xtc_probability = 0.000, xtc_threshold = 1.000, top_n_sigma = 0.000
sampling order:
CFG -> Penalties -> dry -> top_k -> tfs_z -> typical_p -> top_p -> min_p -> xtc -> top_n_sigma -> temperature
generate: n_ctx = 16384, n_batch = 4096, n_predict = -1, n_keep = 1


<|begin?of?sentence|><|User|>What is the solution of x+5=-2?<|Assistant|><think>
D:\a\ik_llama.cpp\ik_llama.cpp\ggml\src\ggml-cuda\mmq.cuh:107: fatal error
```

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-07-10** at **07:52:57**:<br>

What is the quantization mix being used?

---

👤 **Thireus** commented the **2025-07-11** at **18:24:49**:<br>

This one: https://github.com/Thireus/GGUF-Tool-Suite/blob/main/recipe_examples/DeepSeek-R1-0528.THIREUS-3.1027bpw-3.3372ppl.242GB-GGUF_11GB-GPU_231GB-CPU.3c88ec6_adc8101.recipe

Just compiled the latest commit and still happening:
`D:\a\ik_llama.cpp\ik_llama.cpp\ggml\src\ggml-cuda\mmq.cuh:107: fatal error`

Edit: Link edited.

---

👤 **ikawrakow** commented the **2025-07-12** at **06:47:34**:<br>

The link you posted gives 404. But even if it worked, we know that the HF tensor viewer does not work when the model contains `ik_llama.cpp` specific quantization types.

How hard is it to to post the portion of the log that tells us how many tensors there are from what type?

---

👤 **Thireus** commented the **2025-07-12** at **07:19:00**:<br>

I'm not sure what you mean by "HF tensor viewer", I'm not using it.

Sorry, didn't realise I had missed that portion of the logs, here is another one:
```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2,1 ~/ik_llama-main-b3912-18b0375-bin-win-cuda-12.8-x64-avx512/llama-cli -m model.gguf  -mla 3 -fa \
  -amb 1024 \
  -fmoe \
  -ctk f16 \
  -c 16384 \
  -ngl 99 \
  -ot "blk\.(3|4|5|6)\.ffn_.*=CUDA0" -ot "blk\.(7|8|9)\.ffn_.*=CUDA1" -ot "blk\.(10|11|12)\.ffn_.*=CUDA2" \
  -ot exps=CPU \
  -b 4096 -ub 4096 \
  --warmup-batch \
  --no-mmap \
  --threads 36 \
  --main-gpu 0 \
  -p '<｜begin▁of▁sentence｜><｜User｜>What is the solution of x+5=-2?<｜Assistant｜><think>\n'
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 3 CUDA devices:
  Device 0: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
  Device 1: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 2: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
Log start
main: build = 1 (18b0375)
main: built with MSVC 19.44.35209.0 for
main: seed  = 1752263940
llama_model_loader: Max stdio successfully set to 2048
llama_model_loader: additional 1147 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 49 key-value pairs and 1147 tensors from DeepSeek-R1-0528-THIREUS-BF16-SPECIAL_TENSOR-00001-of-01148.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = DeepSeek R1 0528
llama_model_loader: - kv   3:                            general.version str              = 0528
llama_model_loader: - kv   4:                           general.basename str              = DeepSeek-R1
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
llama_model_loader: - kv  16:                          general.file_type u32              = 32
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
llama_model_loader: - kv  46:                                   split.no u16              = 0
llama_model_loader: - kv  47:                                split.count u16              = 1148
llama_model_loader: - kv  48:                        split.tensors.count i32              = 1147
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  193 tensors
llama_model_loader: - type iq4_xs:  305 tensors
llama_model_loader: - type iq2_k:   40 tensors
llama_model_loader: - type iq3_k:   88 tensors
llama_model_loader: - type iq6_k:  101 tensors
llama_model_loader: - type iq4_ks:   20 tensors
llama_model_loader: - type iq1_m_r4:   26 tensors
llama_model_loader: - type iq5_k_r4:   13 tensors
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
llm_load_print_meta: model ftype      = BF16
llm_load_print_meta: model params     = 672.050 B
llm_load_print_meta: model size       = 246.446 GiB (3.150 BPW)
llm_load_print_meta: repeating layers = 244.612 GiB (3.135 BPW, 670.196 B parameters)
llm_load_print_meta: general.name     = DeepSeek R1 0528
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
llm_load_tensors: ggml ctx size =    1.87 MiB
Tensor blk.3.ffn_norm.weight buffer type overriden to CUDA0
...
Tensor blk.60.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size = 209592.50 MiB
llm_load_tensors:  CUDA_Host buffer size =   938.98 MiB
llm_load_tensors:      CUDA0 buffer size = 16412.52 MiB
llm_load_tensors:      CUDA1 buffer size = 11530.49 MiB
llm_load_tensors:      CUDA2 buffer size = 13886.35 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 16384
llama_new_context_with_model: n_batch    = 4096
llama_new_context_with_model: n_ubatch   = 4096
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 3
llama_new_context_with_model: attn_max_b = 1024
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init:      CUDA0 KV buffer size =   450.00 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =   342.00 MiB
llama_kv_cache_init:      CUDA2 KV buffer size =   306.00 MiB
llama_new_context_with_model: KV self size  = 1098.00 MiB, c^KV (f16): 1098.00 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.49 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      CUDA0 compute buffer size =  4496.02 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =  3152.02 MiB
llama_new_context_with_model:      CUDA2 compute buffer size =  3152.03 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   368.05 MiB
llama_new_context_with_model: graph nodes  = 4200
llama_new_context_with_model: graph splits = 177

system_info: n_threads = 36 / 36 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 |
sampling: 
        repeat_last_n = 64, repeat_penalty = 1.000, frequency_penalty = 0.000, presence_penalty = 0.000
        top_k = 40, tfs_z = 1.000, top_p = 0.950, min_p = 0.050, typical_p = 1.000, temp = 0.800
        mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000
        xtc_probability = 0.000, xtc_threshold = 1.000, top_n_sigma = 0.000
sampling order:
CFG -> Penalties -> dry -> top_k -> tfs_z -> typical_p -> top_p -> min_p -> xtc -> top_n_sigma -> temperature
generate: n_ctx = 16384, n_batch = 4096, n_predict = -1, n_keep = 1


<|begin?of?sentence|><|User|>What is the solution of x+5=-2?<|Assistant|><think>
D:\a\ik_llama.cpp\ik_llama.cpp\ggml\src\ggml-cuda\mmq.cuh:107: fatal error
```

---

👤 **Thireus** commented the **2025-07-12** at **07:19:00**:<br>

Any of these won't work:
https://github.com/Thireus/GGUF-Tool-Suite/tree/main/recipe_examples

---

👤 **ikawrakow** commented the **2025-07-12** at **09:27:32**:<br>

Does #603 fix it for you?

There were two more commits after the commit that actually breaks it for your mix that uses `IQ1_M`, a typically not used quantization type.

---

👤 **Thireus** commented the **2025-07-12** at **11:08:41**:<br>

Thanks, I'll take a look now and will report back. It'll take a few hours.

---

👤 **Thireus** commented the **2025-07-12** at **18:35:40**:<br>

@ikawrakow, the fix is working! Thank you so much.

---

👤 **saood06** commented the **2025-07-13** at **07:56:18**:<br>

>The link you posted gives 404. But even if it worked, we know that the HF tensor viewer does not work when the model contains ik_llama.cpp specific quantization types.
>
>How hard is it to to post the portion of the log that tells us how many tensors there are from what type?

It no longer gives a 404 (I didn't see one). It is better than HF tensor viewer, it is a documented custom regex string.

---

👤 **saood06** commented the **2025-07-13** at **07:56:18**:<br>

>The link you posted gives 404. But even if it worked, we know that the HF tensor viewer does not work when the model contains ik_llama.cpp specific quantization types.
>
>How hard is it to to post the portion of the log that tells us how many tensors there are from what type?

It no longer gives a 404. It is better than HF tensor viewer, it is a documented custom regex string.

---

👤 **ikawrakow** commented the **2025-07-13** at **09:37:13**:<br>

> It no longer gives a 404 (I didn't see one). It is better than HF tensor viewer, it is a documented custom regex string.

Yes, I saw it after the link became accessible. That's how I knew what the issue was, and fixed it in #603.