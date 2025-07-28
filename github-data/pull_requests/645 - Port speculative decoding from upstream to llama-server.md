## 🔀 [Pull Request #645](https://github.com/ikawrakow/ik_llama.cpp/pull/645) - Port speculative decoding from upstream to llama-server

| **Author** | `g2mt` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Source Branch** | `speculative-port` |
| **Target Branch** | `main` |
| **Created** | 2025-07-25 |
| **Updated** | 2025-07-27 |
| **Assignees** | `saood06` |

---

## 📄 Description

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

## 💬 Conversation

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

---

👤 **usrlocalben** commented on **2025-07-26** at **18:35:07**

tl;dr: I have a working draft config for mainline llama. It works fine with this branch without changes.

I see the same relative perf improvement as mainline. (substantial, about +30-40% TG for codegen or other outputs with recurring patterns)

It isn't clear how to interpret the log output wrt. draft acceptance:

These blobs pop up from time to time during TG.  _reuse_i_ is always zero which seems suspicious. However, if I run the same config/prompt w and w/o the draft model, I do see expected change in perf.

```
llama_speculative_gen_draft: reuse_i = 0, reuse_n = 4693, prompt = 4693
llama_speculative_gen_draft: n_past = 4695
 - draft candidate   0, pos   0:  29938 (   1.000) ' "__'
 - draft candidate   1, pos   0:  37038 (   0.000) ' '__'
 - draft candidate   2, pos   0:    414 (   0.000) ' "'
 - draft candidate   0, pos   1:   9885 (   1.000) 'main'
 - draft candidate   1, pos   1:   6593 (   0.000) 'build'
 - draft candidate   2, pos   1:  13098 (   0.000) 'async'
 - draft candidate   0, pos   2:  71703 (   0.998) '__":
'
 - draft candidate   1, pos   2:   1025 (   0.002) '__'
 - draft candidate   2, pos   2:  59325 (   0.000) '__':
'
 - draft candidate   0, pos   3:    274 (   0.999) '   '
 - draft candidate   1, pos   3:    337 (   0.001) '       '
 - draft candidate   2, pos   3:    220 (   0.000) ' '
 - draft candidate   0, pos   4:  85632 (   0.422) ' asyncio'
 - draft candidate   1, pos   4:   7443 (   0.223) ' async'
 - draft candidate   2, pos   4:   5276 (   0.107) ' await'
 ```
 
 my config:
 ```
-fa -mla 2 -fmoe
-b 4096 -ub 4096
--n-gpu-layers 99
-c 32000
-ot exps=CPU
--top-k 1 --samplers "top_k"
-m /path/to/k2/DevQuasar/Q4_K_M/moonshotai.Kimi-K2-Instruct_updated.Q4_K_M-00001-of-00053.gguf
-md /path/to/k2/draft/Kimi-K2-Instruct-DRAFT-0.6B-32k-Q4_0.gguf
-ngld 99
```

model is Kimi-K2
target is [DevQuasar Q4_K_M](https://huggingface.co/DevQuasar/moonshotai.Kimi-K2-Instruct-GGUF)
draft is [jukofyork](https://huggingface.co/jukofyork/Kimi-K2-Instruct-DRAFT-0.6B-v2.0-GGUF)

some perf results, although it's all content dependent so grain of salt etc.
```
w/draft
prompt eval time     =   53826.90 ms /  2379 tokens (   22.63 ms per token,    44.20 tokens per second)
generation eval time =  162801.04 ms /  2328 runs   (   69.93 ms per token,    14.30 tokens per second)
          total time =  216627.94 ms


w/o draft
prompt eval time     =   53792.43 ms /  2379 tokens (   22.61 ms per token,    44.23 tokens per second)
generation eval time =  208580.89 ms /  2358 runs   (   88.46 ms per token,    11.30 tokens per second)
          total time =  262373.32 ms
```

and another K2 run, same content
target is [ubergarm IQ3_KS](https://huggingface.co/ubergarm/Kimi-K2-Instruct-GGUF)
draft is [jukofyork](https://huggingface.co/jukofyork/Kimi-K2-Instruct-DRAFT-0.6B-v2.0-GGUF)

```
w/draft
prompt eval time     =   24146.63 ms /  2379 tokens (   10.15 ms per token,    98.52 tokens per second)
generation eval time =  141704.33 ms /  2312 runs   (   61.29 ms per token,    16.32 tokens per second)
          total time =  165850.96 ms

w/o draft
prompt eval time     =   23862.52 ms /  2379 tokens (   10.03 ms per token,    99.70 tokens per second)
generation eval time =  174326.72 ms /  2260 runs   (   77.14 ms per token,    12.96 tokens per second)
          total time =  198189.24 ms
```

```
-fa -mla 2 -fmoe
-b 4096 -ub 4096
--n-gpu-layers 99
-c 32000
-ot "blk\.(1|2|3|4|5|6)\.ffn_up_exps=CUDA0,blk\.(1|2|3|4|5|6)\.ffn_gate_exps=CUDA0"
-ot exps=CPU
-op 26,0,27,0,29,0
--top-k 1 --samplers "top_k"
-m /path/to/k2/ubergarm/IQ3_KS/Kimi-K2-Instruct-IQ3_KS-00001-of-00010.gguf
-md /path/to/k2/draft/Kimi-K2-Instruct-DRAFT-0.6B-32k-Q4_0.gguf
-ngld 99
```

hardware is 2S EPYC 9115 NPS0, 24x DDR5, RTX 8000 (Turing)

---

👤 **g2mt** commented on **2025-07-26** at **18:51:50**

@ChicoPinto70 

I wonder if this is another tensor name conflict error. I don't have a GPU, so I can't really test this. Could you run the fork in gdb and paste the stack trace here?

> These blobs pop up from time to time during TG. _reuse_i_ is always zero which seems suspicious. However, if I run the same config/prompt w and w/o the draft model, I do see expected change in perf.

@usrlocalben 
reuse_i should be zero most of the time if I understood the original common/speculative.cpp code correctly. It represents the index of the first token in the prompt of the draft model that can be reused (basically the first index of the same token). If the prompt isn't changed across generation then it should stay at 0.

---

👤 **ChicoPinto70** commented on **2025-07-26** at **20:23:16**

> @ChicoPinto70
> 
> I wonder if this is another tensor name conflict error. I don't have a GPU, so I can't really test this. Could you run the fork in gdb and paste the stack trace here?
> 
> > These blobs pop up from time to time during TG. _reuse_i_ is always zero which seems suspicious. However, if I run the same config/prompt w and w/o the draft model, I do see expected change in perf.
> 
> @usrlocalben reuse_i should be zero most of the time if I understood the original common/speculative.cpp code correctly. It represents the index of the first token in the prompt of the draft model that can be reused (basically the first index of the same token). If the prompt isn't changed across generation then it should stay at 0.

Sure! Did I make it right?

(gdb) backtrace
[#0](https://github.com/ikawrakow/ik_llama.cpp/issues/0)  __pthread_kill_implementation (no_tid=0, signo=6, threadid=<optimized out>) at ./nptl/pthread_kill.c:44
[#1](https://github.com/ikawrakow/ik_llama.cpp/issues/1)  __pthread_kill_internal (signo=6, threadid=<optimized out>) at ./nptl/pthread_kill.c:78
[#2](https://github.com/ikawrakow/ik_llama.cpp/issues/2)  __GI___pthread_kill (threadid=<optimized out>, signo=signo@entry=6) at ./nptl/pthread_kill.c:89
[#3](https://github.com/ikawrakow/ik_llama.cpp/issues/3)  0x00007fffe444527e in __GI_raise (sig=sig@entry=6) at ../sysdeps/posix/raise.c:26
[#4](https://github.com/ikawrakow/ik_llama.cpp/issues/4)  0x00007fffe44288ff in __GI_abort () at ./stdlib/abort.c:79
[#5](https://github.com/ikawrakow/ik_llama.cpp/issues/5)  0x00007fffe4cb588c in ggml_abort (file=0x7fffe61cf590 "/home/chico/ik_llama.cpp/ggml/src/ggml-cuda/mmvq.cu", line=604, 
    fmt=0x7fffe61cfb5a "GGML_ASSERT(%s) failed") at /home/chico/ik_llama.cpp/ggml/src/ggml.c:270
[#6](https://github.com/ikawrakow/ik_llama.cpp/issues/6)  0x00007fffe4e4713f in ggml_cuda_op_mul_mat_vec_q_3D (ctx=..., src0=0x7fba52900fc0, src1=0x7fba52900e50, 
    dst=0x7fba529012a0, src0_dd_i=0x1d02000000 "", src1_ddf_i=0x7fb590540800, src1_ddq_i=0x1d02001380 "", 
    dst_dd_i=0x7fb590700800, row_low=0, row_high=32, src1_ncols=1, src1_padded_row_size=512, stream=0x5555672e3690)
    at /home/chico/ik_llama.cpp/ggml/src/ggml-cuda/mmvq.cu:604
[#7](https://github.com/ikawrakow/ik_llama.cpp/issues/7)  0x00007fffe4efcc6e in ggml_cuda_op_mul_mat (ctx=..., src0=0x7fba52900fc0, src1=0x7fba52900e50, dst=0x7fba529012a0, 
    op=0x7fffe4e471c3 <ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*)>, 
    quantize_src1=0x7fffe4ede4af <quantize_row_q8_1_cuda(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)>) at /home/chico/ik_llama.cpp/ggml/src/ggml-cuda.cu:1658
[#8](https://github.com/ikawrakow/ik_llama.cpp/issues/8)  0x00007fffe4eff807 in ggml_cuda_mul_mat (ctx=..., src0=0x7fba52900fc0, src1=0x7fba52900e50, dst=0x7fba529012a0)
    at /home/chico/ik_llama.cpp/ggml/src/ggml-cuda.cu:2176
[#9](https://github.com/ikawrakow/ik_llama.cpp/issues/9)  0x00007fffe4f05314 in ggml_cuda_compute_forward (ctx=..., dst=0x7fba529012a0, next=0x7fba52901410, 
    skip_next=@0x7fffffffa5e0: false) at /home/chico/ik_llama.cpp/ggml/src/ggml-cuda.cu:2937
[#10](https://github.com/ikawrakow/ik_llama.cpp/issues/10) 0x00007fffe4f069f8 in ggml_backend_cuda_graph_compute (backend=0x555565795070, cgraph=0x555564d3bcc8)
    at /home/chico/ik_llama.cpp/ggml/src/ggml-cuda.cu:3327
[#11](https://github.com/ikawrakow/ik_llama.cpp/issues/11) 0x00007fffe4d0e0d3 in ggml_backend_graph_compute_async (backend=0x555565795070, cgraph=0x555564d3bcc8)
    at /home/chico/ik_llama.cpp/ggml/src/ggml-backend.c:317
[#12](https://github.com/ikawrakow/ik_llama.cpp/issues/12) 0x00007fffe4d12c1f in ggml_backend_sched_compute_splits (sched=0x555564d398d0)
    at /home/chico/ik_llama.cpp/ggml/src/ggml-backend.c:1887
[#13](https://github.com/ikawrakow/ik_llama.cpp/issues/13) 0x00007fffe4d13831 in ggml_backend_sched_graph_compute_async (sched=0x555564d398d0, graph=0x7fba526fb030)
    at /home/chico/ik_llama.cpp/ggml/src/ggml-backend.c:2081
[#14](https://github.com/ikawrakow/ik_llama.cpp/issues/14) 0x00007ffff7cea0de in llama_graph_compute (lctx=..., gf=0x7fba526fb030, n_threads=36)
    at /home/chico/ik_llama.cpp/src/llama.cpp:18241
[#15](https://github.com/ikawrakow/ik_llama.cpp/issues/15) 0x00007ffff7cead49 in llama_decode_internal (lctx=..., batch_all=...) at /home/chico/ik_llama.cpp/src/llama.cpp:18457
[#16](https://github.com/ikawrakow/ik_llama.cpp/issues/16) 0x00007ffff7cfd6f7 in llama_decode (ctx=0x55556777b1f0, batch=...) at /home/chico/ik_llama.cpp/src/llama.cpp:22945
[#17](https://github.com/ikawrakow/ik_llama.cpp/issues/17) 0x000055555575aec4 in llama_init_from_gpt_params (params=...) at /home/chico/ik_llama.cpp/common/common.cpp:2414
[#18](https://github.com/ikawrakow/ik_llama.cpp/issues/18) 0x0000555555639ffd in server_context::load_model (this=0x7fffffffc9b0, params_=...)
    at /home/chico/ik_llama.cpp/examples/server/server.cpp:919
[#19](https://github.com/ikawrakow/ik_llama.cpp/issues/19) 0x00005555556063e4 in main (argc=42, argv=0x7fffffffd948) at /home/chico/ik_llama.cpp/examples/server/server.cpp:3386
(gdb) down
Bottom (innermost) frame selected; you cannot go down.
(gdb) up
[#1](https://github.com/ikawrakow/ik_llama.cpp/issues/1)  __pthread_kill_internal (signo=6, threadid=<optimized out>) at ./nptl/pthread_kill.c:78
78	in ./nptl/pthread_kill.c
(gdb)

---

👤 **saood06** commented on **2025-07-26** at **20:45:22**

> > If someone else could post more test results that would be great. I'll open the PR up for review now.
> 
> I'll try to do some tests within a day.

So far I've compiled and ran the new R1 with a draft model (using [this](https://huggingface.co/jukofyork/DeepSeek-R1-DRAFT-0.6B-v2.0-GGUF/blob/main/DeepSeek-R1-DRAFT-0.6B-64k-Q4_0.gguf) one). With both the Draft and main model on my CPU (this server has no GPU).

(I'll report performance numbers later, with tests at temp 0 for the best potential draft rate).

I'll do some more tests on Windows where I have a GPU, and RPC (I didn't see code that would allow you to offload your draft model via RPC but I might try it anyway), and with other models (can't run Deepseek on my Windows machine) and more thoroughly review (and submit potential comments) the code next.

Edit: For now I'm not sure if I agree with what the logging levels are "speculative decoding result" seems a lot more useful to print by default than the spam of  `reuse_i [...]` and `draft candidate`

Edit 2: I know performance with drafting has a lot of variables and factors so take that into consideration before the numbers below.

Without draft model:
`generation eval time =  423525.84 ms /  1155 runs   (  366.69 ms per token,     2.73 tokens per second)`

With draft model:
`generation eval time =  416970.47 ms /  1083 runs   (  385.01 ms per token,     2.60 tokens per second)`

So for these conditions (temperature 0, the specific models and hardware used) this resulted in worse performance.

The response also did change in the second paragraph of the thinking process which I'm fairly certain should not happen at temperature 0. (There is a chance this could be my error given I was testing this using the new WebUI that I am far less familiar with but looking at the other artifacts '/slots' endpoint the second test (without draft model) the same prompt was sent with 0 temp so the testing seems valid to me).

---

👤 **g2mt** commented on **2025-07-26** at **21:30:50**

I think I see the problem. It seems that building the computation graph creates new tensors with conflicting names:

https://github.com/ikawrakow/ik_llama.cpp/blob/7093a35869670cf954bd1ba843df8ccf0c2867f2/src/llama.cpp#L17445-L17461

The code that builds the computation graph (and maybe the rest of src/llama.cpp) uses a lot of hardcoded names. This would not work if there's more than one model being loaded. I think a bigger PR is needed for refactoring the file. My guess is that the code sections that search for hard coded tensor names aren't triggered during normal CPU-only inference. I'm not really familiar with this code base to make that assessment though.

Side note, wow the file is huge. clangd doesn't even show me autocompletions.

---

👤 **saood06** commented on **2025-07-26** at **21:42:06**

>My guess is that the code sections that search for hard coded tensor names aren't triggered during normal CPU-only inference. I'm not really familiar with this code base to make that assessment though.

But given that it worked for usrlocalben who was using pure GPU inference `--n-gpu-layers 99` and `-ngld 99` could it be the issue is when more than one backend type is used?

I could try and confirm that theory later on my Windows system.

>Side note, wow the file is huge. clangd doesn't even show me autocompletions.

Github doesn't index it for search or allow blame, or syntax highlight it. As mentioned [here](https://github.com/ikawrakow/ik_llama.cpp/issues/472#issuecomment-2924324079), it is something that would be nice to refactor but no one has done it yet.

---

👤 **usrlocalben** commented on **2025-07-26** at **23:49:10**

@saood06 my setup is mixed GPU/CPU. The tensor offload pattern rules have precedence, but -ngl is still needed to make them available for GPU.

In case anyone in this thread is unaware, speculative generation perf tends to be dependent on the content. Code happens to have a good outcome, and repetitive code even better.

[This thread](https://github.com/ggml-org/llama.cpp/discussions/10466) from mainline has some discussion on it.

here's one target quant:
```
IQ3_KS (ubergarm)
  TG  8.44 t/s speculative, general: "summarize this EULA"
  TG 11.32 t/s normal
  TG 13.87 t/s speculative, code: "rewrite using async/await"
```

My GPU (full offload of the draft model) is older/slower (Turing). Maybe newer tech would perform better in the face of low draft hit-rate.

---

👤 **saood06** commented on **2025-07-27** at **00:09:12**

> @saood06 my setup is mixed GPU/CPU. The tensor offload pattern rules have precedence, but -ngl is still needed to make them available for GPU.

Sorry I noticed that when I first read your post, but missed it when I came back to it.


> In case anyone in this thread is unaware, speculative generation perf tends to be dependent on the content. Code happens to have a good outcome, and repetitive code even better.
> 
> [This thread](https://github.com/ggml-org/llama.cpp/discussions/10466) from mainline has some discussion on it.

That is a good thread, thanks for the link.

> here's one target quant:
> 
> ```
> IQ3_KS (ubergarm)
>   TG  8.44 t/s speculative, general: "summarize this EULA"
>   TG 11.32 t/s normal
>   TG 13.87 t/s speculative, code: "rewrite using async/await"
> ```
> 

Nice. I do see you run `--top-k 1 --samplers "top_k` but I wonder how much using a non zero temperature would impact this.

> My GPU (full offload of the draft model) is older/slower (Turing). Maybe newer tech would perform better in the face of low draft hit-rate.

Well the faster your draft model the more tokens it produces so that may lower hit-rate even more, but thinking about performance when the draft and target on the same hardware seems complicated.

---

👤 **ikawrakow** commented on **2025-07-27** at **04:45:49**

Why is the presence of tensors with the same name in the two models a problem? And how does it work in mainline, where tensor names are given in the same way?

---

👤 **saood06** commented on **2025-07-27** at **05:05:38**

>Why is the presence of tensors with the same name in the two models a problem? And how does it work in mainline, where tensor names are given in the same way?

I'm not sure I agree with that conclusion of the issue.

It fails in the ggml_cuda_op_mul_mat_vec_q_3D of a llama_decode_internal call in the llama_graph_compute call. I'm going to test on my Windows machine with my 3090 shortly and see how things work there with different configs.

I'm trying to see if I can get RPC working now.

---

👤 **saood06** commented on **2025-07-27** at **05:08:54**

I see you just pushed a revert. Was that not needed?

---

👤 **g2mt** commented on **2025-07-27** at **05:09:07**

> Why is the presence of tensors with the same name in the two models a problem? And how does it work in mainline, where tensor names are given in the same way?

Turns out I did something weird with my environment which caused an error when loading the llama-server binary. I reverted the KV cache change. Sorry for any misunderstandings.

It should stiil work, not sure about the CUDA problem though.

---

👤 **saood06** commented on **2025-07-27** at **11:00:51**

I modified the code (inspired by the `-ot PR`: https://github.com/ikawrakow/ik_llama.cpp/pull/232) and got it to allocate all the buffers of the draft model to my RPC node.


```
Tensor blk.0.attn_k.weight buffer type overriden to RPC[10.0.0.250:50052]
Tensor blk.0.attn_v.weight buffer type overriden to RPC[10.0.0.250:50052]
Tensor blk.0.attn_output.weight buffer type overriden to RPC[10.0.0.250:50052]
[...]
Tensor blk.23.ffn_gate.weight buffer type overriden to RPC[10.0.0.250:50052]
Tensor blk.23.ffn_down.weight buffer type overriden to RPC[10.0.0.250:50052]
Tensor blk.23.ffn_up.weight buffer type overriden to RPC[10.0.0.250:50052]
```
but then it crashes with 

```
llama_model_load: error loading model: failed to allocate buffer
llama_load_model_from_file: failed to load model
llama_init_from_gpt_params: error: failed to load model '/mnt/sda/draft_models/R1/DeepSeek-R1-DRAFT-0.6B-64k-Q4_0.gguf'
```

Got a backtrace:

```
#0  __GI___libc_free () at malloc.c:3378
#1  0x0000555555417250 in llama_batch_free (batch=...) at /home/saood06/ik_temp/ik_llama.cpp/src/llama.cpp:22920
#2  0x00005555555cde00 in server_context::~server_context (this=0x7fffffffb290, __in_chrg=<optimized out>) at /home/saood06/ik_temp/ik_llama.cpp/examples/server/server.cpp:882
#3  0x000055555559129f in main (argc=<optimized out>, argv=<optimized out>) at /home/saood06/ik_temp/ik_llama.cpp/examples/server/server.cpp:4344
```
Not sure why right now. Stepping off for now, will test my 3090 and Windows tomorrow.

---

👤 **ChicoPinto70** commented on **2025-07-27** at **12:17:36**

> > Why is the presence of tensors with the same name in the two models a problem? And how does it work in mainline, where tensor names are given in the same way?
> 
> Turns out I did something weird with my environment which caused an error when loading the llama-server binary. I reverted the KV cache change. Sorry for any misunderstandings.
> 
> It should stiil work, not sure about the CUDA problem though.

Hi, I tested the current branch and the CUDA  problem persists.... 

I also tested run the draft model without offloading it to gpu (-ngld 0), but still keeping the full model in gpu,  and it worked, but the TG speed plummed and it was much slower than without the draft (from ~7 T/s to 0.7 T/s).