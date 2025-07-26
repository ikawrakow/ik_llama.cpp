### [Pull Request #645](https://github.com/ikawrakow/ik_llama.cpp/pull/645) - Port speculative decoding from upstream to llama-server

| **Author** | `g2mt` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Source Branch** | `speculative-port` |
| **Target Branch** | `main` |
| **Created** | 2025-07-25 |
| **Updated** | 2025-07-26 |
| **Assignees** | `saood06` |

---

#### Description

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [ ] Low
  - [x] Medium
  - [ ] High

Related to [#322](https://github.com/ikawrakow/ik_llama.cpp/issues/322) 

This is a port of the speculative decoding function for llama-server from the upstream code base.

Changes:

- Updated llama-server source code
- Added several functions needed for speculative decoding.
- Add prefixes to KV cache tensors to  support loading of multiple models

I used Qwen3-235B in this PR.

---

#### 🔀 Conversation

👤 **saood06** commented on **2025-07-25** at **05:15:48**

Thank you for doing this. I can test/review/assist if you need.

---

👤 **saood06** commented on **2025-07-25** at **05:18:58**

Also are you aware this: https://github.com/ikawrakow/ik_llama.cpp/blob/main/examples/speculative/speculative.cpp exists.

---

👤 **g2mt** commented on **2025-07-25** at **05:26:10**

I got the server to compile, but when loading Qwen 2.5 1.5b with the 0.5b version as the draft, I get this error:

```
ggml_backend_alloc_ctx_tensors_from_buft: all tensors in the context are already allocated
llama_kv_cache_init: failed to allocate buffer for kv cache
llama_new_context_with_model: llama_kv_cache_init() failed for self-attention cache
llama_init_from_gpt_params: error: failed to create context with model 'Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf'
 ERR [              load_model] failed to load draft model | tid="140650859190528" timestamp=1753420591 model="Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf"
```

GDB says it occurred in this llama_init_from_gpt_params call:

```cpp
            llama_init_result llama_init_dft = llama_init_from_gpt_params(params_dft);
```

I wonder if llama_kv_cache_init is unable to load tensors with the same name. I'll try and fix the code later.

---

👤 **g2mt** commented on **2025-07-25** at **05:27:44**

> Also are you aware this: https://github.com/ikawrakow/ik_llama.cpp/blob/main/examples/speculative/speculative.cpp exists.

I am aware of the example. I'll check it later.

---

👤 **saood06** commented on **2025-07-25** at **05:34:38**

>I am aware of the example. I'll check it later.

Sorry. I forgot my history. The common one (introduced here: https://github.com/ggml-org/llama.cpp/pull/10362) was done before server: https://github.com/ggml-org/llama.cpp/pull/10455. The common implementation was made to be simpler to understand and work with which is why it came bundled with https://github.com/ggml-org/llama.cpp/tree/8f419181d1c20d8195148680df15b6f093cb1512/examples/speculative-simple

---

👤 **g2mt** commented on **2025-07-25** at **07:09:50**

I'm now able to load the draft model. It seems that the kv-cache tensor names were reused for both models. Prefixing them with the model name fixes it.

---

👤 **saood06** commented on **2025-07-25** at **07:47:27**

>I'm now able to load the draft model. It seems that the kv-cache tensor names were reused for both models. Prefixing them with the model name fixes it.

Nice. Did you get any accepted tokens?

---

👤 **g2mt** commented on **2025-07-25** at **09:02:33**

I think I got it working. For some reason ik_llama's slot.id is offset by 1, which tripped me off a bit.

A simple test of repeating a string shows it working:

```
curl -s http://localhost:9001/v1/chat/completions \
          -H "Content-Type: application/json" \
          -H "Authorization: Bearer no-key" \
          -d '{"model": "test","messages": [{"role": "user","content": "Repeat the following sentence, as is: The quick brown fox jumped over the lazy dog."}]}'
{"choices":[{"finish_reason":"stop","index":0,"message":{"role":"assistant","content":"The quick brown fox jumped over the lazy dog."}}],"created":1753433480,"model":"test","object":"chat.completion","usage":{"completion_tokens":14,"prompt_tokens":26,"total_tokens":40},"id":"chatcmpl-QK3CBenhWiSBeeuIs6UGs2yXCV5YpqRO","__verbose":{"content":"The quick brown fox jumped over the lazy dog.","generated_text":"The quick brown fox jumped over the lazy dog.",
```

Server logs do show the speculative decoding results being accepted:

```
VERB [            update_slots] speculative decoding result | tid="140737350637888" timestamp=1753433480 id_slot=0 accepted=12 total=13 new_n_past=39
```

It looks like it's working, but I think more testing is needed. If someone else could post more test results that would be great. I'll open the PR up for review now.

---

👤 **saood06** commented on **2025-07-25** at **09:12:46**

>If someone else could post more test results that would be great. I'll open the PR up for review now.

I'll try to do some tests within a day.

---

👤 **ikawrakow** commented on **2025-07-25** at **09:21:28**

@saood06 I'll be not able to review before August 7, so I have assigned you as a reviewer.

Hopefully more people will test.

---

👤 **saood06** commented on **2025-07-25** at **09:47:41**

> @saood06 I'll be not able to review before August 7, so I have assigned you as a reviewer.

I'll review and test it.

---

👤 **ChicoPinto70** commented on **2025-07-26** at **12:30:23**

Hi, Guys. I've tested this branch in a Dual Xeon E5 2699 v3, 256GB DDR4, 3xRTX 3090 in Ubuntu 24.04.2 LTS.

I compiled the project with these parameters:

cmake -B build -DGGML_CUDA=ON -DGGML_BLAS=OFF -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DGGML_SCHED_MAX_COPIES=1 DGGML_CUDA_DISABLE_GRAPHS=1 -DGGML_CUDA_IQK_FORCE_BF16=ON -DGGML_CUDA_MIN_BATCH_OFFLOAD=64

And I ran it with this command:

CUDA_VISIBLE_DEVICES="1,2,0" ./build/bin/llama-server  --alias unsloth/DeepSeek-R1-0528-UD-Q3_K_XL  -m /home/chico/.lmstudio/models/unsloth/DeepSeek-R1-0528-GGUF/DeepSeek-R1-0528-UD-Q3_K_XL-00001-of-00007.gguf  -ngl 64 -c 16384 -mla 3 -fa -amb 1024 -fmoe -t 32 -ctk q8_0 -ot "blk\.[0-6]\..*_exps\.=CUDA1,blk\.(7|8|9|10)\..*_exps\.=CUDA2,exps=CPU"  --parallel 1  --numa distribute -b 4096 -ub 4096 --no-mmap -ts 1,0,0 -ser 7,1 --host 192.168.0.9 --port 1235 -md /home/chico/.lmstudio/models/jukofyork/DeepSeek-R1-DRAFT-0.6B-v2.0-GGUF/DeepSeek-R1-DRAFT-0.6B-128k-Q4_0.gguf -ngld 64 

It failed with this message: /home/chico/ik_llama.cpp/ggml/src/ggml-cuda/mmvq.cu:604: GGML_ASSERT(src0->ne[2] == src1->ne[2] && src0->ne[2] == dst->ne[2]) failed

Bellow, the complete output:

ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 3 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 1: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 2: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
INFO [                    main] build info | tid="130964736602112" timestamp=1753532241 build=3841 commit="e938d9f6"
INFO [                    main] system info | tid="130964736602112" timestamp=1753532241 n_threads=32 n_threads_batch=-1 total_threads=36 system_info="AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: additional 6 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 63 key-value pairs and 1086 tensors from /home/chico/.lmstudio/models/unsloth/DeepSeek-R1-0528-GGUF/DeepSeek-R1-0528-UD-Q3_K_XL-00001-of-00007.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Deepseek-R1-0528
llama_model_loader: - kv   3:                           general.basename str              = Deepseek-R1-0528
llama_model_loader: - kv   4:                       general.quantized_by str              = Unsloth
llama_model_loader: - kv   5:                         general.size_label str              = 256x20B
llama_model_loader: - kv   6:                            general.license str              = mit
llama_model_loader: - kv   7:                           general.repo_url str              = https://huggingface.co/unsloth
llama_model_loader: - kv   8:                   general.base_model.count u32              = 1
llama_model_loader: - kv   9:                  general.base_model.0.name str              = DeepSeek R1 0528
llama_model_loader: - kv  10:               general.base_model.0.version str              = 0528
llama_model_loader: - kv  11:          general.base_model.0.organization str              = Deepseek Ai
llama_model_loader: - kv  12:              general.base_model.0.repo_url str              = https://huggingface.co/deepseek-ai/De...
llama_model_loader: - kv  13:                               general.tags arr[str,3]       = ["deepseek", "unsloth", "transformers"]
llama_model_loader: - kv  14:                          general.languages arr[str,1]       = ["en"]
llama_model_loader: - kv  15:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv  16:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv  17:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv  18:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  19:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv  20:          deepseek2.attention.head_count_kv u32              = 1
llama_model_loader: - kv  21:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  22: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  23:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  24:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  25:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  26:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  27:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  28:             deepseek2.attention.key_length u32              = 576
llama_model_loader: - kv  29:           deepseek2.attention.value_length u32              = 512
llama_model_loader: - kv  30:         deepseek2.attention.key_length_mla u32              = 192
llama_model_loader: - kv  31:       deepseek2.attention.value_length_mla u32              = 128
llama_model_loader: - kv  32:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  33:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  34:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  35:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  36:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  37:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  38:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  39:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  40:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  41: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  42: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  43:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  44:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  45:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<�...
llama_model_loader: - kv  46:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  47:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
llama_model_loader: - kv  48:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  49:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  50:            tokenizer.ggml.padding_token_id u32              = 2
llama_model_loader: - kv  51:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  52:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  53:                    tokenizer.chat_template str              = {%- if not add_generation_prompt is d...
llama_model_loader: - kv  54:               general.quantization_version u32              = 2
llama_model_loader: - kv  55:                          general.file_type u32              = 12
llama_model_loader: - kv  56:                      quantize.imatrix.file str              = DeepSeek-R1-0528-GGUF/imatrix_unsloth...
llama_model_loader: - kv  57:                   quantize.imatrix.dataset str              = unsloth_calibration_DeepSeek-R1-0528-...
llama_model_loader: - kv  58:             quantize.imatrix.entries_count i32              = 659
llama_model_loader: - kv  59:              quantize.imatrix.chunks_count i32              = 720
llama_model_loader: - kv  60:                                   split.no u16              = 0
llama_model_loader: - kv  61:                        split.tensors.count i32              = 1086
llama_model_loader: - kv  62:                                split.count u16              = 7
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  122 tensors
llama_model_loader: - type q3_K:  166 tensors
llama_model_loader: - type q4_K:  392 tensors
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
llm_load_print_meta: model size       = 275.576 GiB (3.528 BPW) 
llm_load_print_meta: repeating layers = 274.383 GiB (3.522 BPW, 669.173 B parameters)
llm_load_print_meta: general.name     = Deepseek-R1-0528
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
llm_load_tensors: ggml ctx size =    0.89 MiB
Tensor blk.3.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.3.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.3.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.4.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.4.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.4.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.6.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.6.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.6.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CUDA2
Tensor blk.7.ffn_down_exps.weight buffer type overriden to CUDA2
Tensor blk.7.ffn_up_exps.weight buffer type overriden to CUDA2
Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CUDA2
Tensor blk.8.ffn_down_exps.weight buffer type overriden to CUDA2
Tensor blk.8.ffn_up_exps.weight buffer type overriden to CUDA2
Tensor blk.9.ffn_gate_exps.weight buffer type overriden to CUDA2
Tensor blk.9.ffn_down_exps.weight buffer type overriden to CUDA2
Tensor blk.9.ffn_up_exps.weight buffer type overriden to CUDA2
Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CUDA2
Tensor blk.10.ffn_down_exps.weight buffer type overriden to CUDA2
Tensor blk.10.ffn_up_exps.weight buffer type overriden to CUDA2
Tensor blk.11.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_up_exps.weight buffer type overriden to CPU
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
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size = 233856.00 MiB
llm_load_tensors:  CUDA_Host buffer size =   497.11 MiB
llm_load_tensors:      CUDA0 buffer size =  9925.05 MiB
llm_load_tensors:      CUDA1 buffer size = 18956.00 MiB
llm_load_tensors:      CUDA2 buffer size = 18956.00 MiB
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
Computed blk.15.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.16.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.17.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.18.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.19.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.20.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.21.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.22.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.23.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.24.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.25.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.26.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.27.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.28.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.29.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.30.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.31.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.32.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.33.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.34.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.35.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.36.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.37.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.38.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.39.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.40.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.41.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.42.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.43.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.44.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.45.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.46.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.47.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.48.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.49.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.50.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.51.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.52.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.53.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.54.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.55.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.56.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.57.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.58.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.59.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.60.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
llama_new_context_with_model: n_ctx      = 16384
llama_new_context_with_model: n_batch    = 4096
llama_new_context_with_model: n_ubatch   = 4096
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 3
llama_new_context_with_model: attn_max_b = 1024
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = 7, 1
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init:      CUDA0 KV buffer size =   583.34 MiB
llama_new_context_with_model: KV self size  =  583.31 MiB, c^KV (q8_0):  583.31 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.99 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      CUDA0 compute buffer size =  4496.02 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =  1272.00 MiB
llama_new_context_with_model:      CUDA2 compute buffer size =  1272.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   368.05 MiB
llama_new_context_with_model: graph nodes  = 4219
llama_new_context_with_model: graph splits = 118
INFO [              load_model] loading draft model | tid="130964736602112" timestamp=1753532513 model="/home/chico/.lmstudio/models/jukofyork/DeepSeek-R1-DRAFT-0.6B-v2.0-GGUF/DeepSeek-R1-DRAFT-0.6B-128k-Q4_0.gguf"
llama_model_loader: loaded meta data with 30 key-value pairs and 291 tensors from /home/chico/.lmstudio/models/jukofyork/DeepSeek-R1-DRAFT-0.6B-v2.0-GGUF/DeepSeek-R1-DRAFT-0.6B-128k-Q4_0.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = DeepSeek R1 0528 DRAFT 0.6B
llama_model_loader: - kv   3:                           general.basename str              = DeepSeek-R1-0528-DRAFT
llama_model_loader: - kv   4:                         general.size_label str              = 0.6B
llama_model_loader: - kv   5:                          qwen2.block_count u32              = 24
llama_model_loader: - kv   6:                       qwen2.context_length u32              = 131072
llama_model_loader: - kv   7:                     qwen2.embedding_length u32              = 896
llama_model_loader: - kv   8:                  qwen2.feed_forward_length u32              = 4864
llama_model_loader: - kv   9:                 qwen2.attention.head_count u32              = 14
llama_model_loader: - kv  10:              qwen2.attention.head_count_kv u32              = 2
llama_model_loader: - kv  11:                       qwen2.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  12:     qwen2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  13:                    qwen2.rope.scaling.type str              = yarn
llama_model_loader: - kv  14:                  qwen2.rope.scaling.factor f32              = 4.000000
llama_model_loader: - kv  15: qwen2.rope.scaling.original_context_length u32              = 32768
llama_model_loader: - kv  16:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  17:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  18:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<�...
llama_model_loader: - kv  19:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  20:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
llama_model_loader: - kv  21:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  22:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  23:            tokenizer.ggml.padding_token_id u32              = 1
llama_model_loader: - kv  24:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  25:               tokenizer.ggml.add_sep_token bool             = false
llama_model_loader: - kv  26:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  27:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  28:               general.quantization_version u32              = 2
llama_model_loader: - kv  29:                          general.file_type u32              = 2
llama_model_loader: - type  f32:  121 tensors
llama_model_loader: - type q4_0:  169 tensors
llama_model_loader: - type q8_0:    1 tensors
llm_load_vocab: special tokens cache size = 818
llm_load_vocab: token to piece cache size = 0.8223 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = qwen2
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 129280
llm_load_print_meta: n_merges         = 127741
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 131072
llm_load_print_meta: n_embd           = 896
llm_load_print_meta: n_layer          = 24
llm_load_print_meta: n_head           = 14
llm_load_print_meta: n_head_kv        = 2
llm_load_print_meta: n_rot            = 64
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 64
llm_load_print_meta: n_embd_head_v    = 64
llm_load_print_meta: n_gqa            = 7
llm_load_print_meta: n_embd_k_gqa     = 128
llm_load_print_meta: n_embd_v_gqa     = 128
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 4864
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
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
llm_load_print_meta: model type       = 1B
llm_load_print_meta: model ftype      = Q4_0
llm_load_print_meta: model params     = 589.568 M
llm_load_print_meta: model size       = 371.738 MiB (5.289 BPW) 
llm_load_print_meta: repeating layers = 192.226 MiB (4.505 BPW, 357.898 M parameters)
llm_load_print_meta: general.name     = DeepSeek R1 0528 DRAFT 0.6B
llm_load_print_meta: BOS token        = 0 '<｜begin▁of▁sentence｜>'
llm_load_print_meta: EOS token        = 1 '<｜end▁of▁sentence｜>'
llm_load_print_meta: PAD token        = 1 '<｜end▁of▁sentence｜>'
llm_load_print_meta: LF token         = 131 'Ä'
llm_load_print_meta: max token length = 256
llm_load_tensors: ggml ctx size =    0.51 MiB
llm_load_tensors: offloading 24 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 25/25 layers to GPU
llm_load_tensors:        CPU buffer size =    62.14 MiB
llm_load_tensors:      CUDA0 buffer size =   120.15 MiB
llm_load_tensors:      CUDA1 buffer size =    48.06 MiB
llm_load_tensors:      CUDA2 buffer size =   141.41 MiB
......................................................
llama_new_context_with_model: n_ctx      = 16384
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 0
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 0.25
llama_kv_cache_init:      CUDA0 KV buffer size =    91.88 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =    36.75 MiB
llama_kv_cache_init:      CUDA2 KV buffer size =    18.38 MiB
llama_new_context_with_model: KV self size  =  147.00 MiB, K (q8_0):   51.00 MiB, V (f16):   96.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.49 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      CUDA0 compute buffer size =   487.00 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =   487.00 MiB
llama_new_context_with_model:      CUDA2 compute buffer size =   487.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =    33.76 MiB
llama_new_context_with_model: graph nodes  = 773
llama_new_context_with_model: graph splits = 4
/home/chico/ik_llama.cpp/ggml/src/ggml-cuda/mmvq.cu:604: GGML_ASSERT(src0->ne[2] == src1->ne[2] && src0->ne[2] == dst->ne[2]) failed
Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
ptrace: Operation not permitted.
No stack.
The program is not being run.
Aborted (core dumped)