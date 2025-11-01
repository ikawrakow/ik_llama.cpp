## 📌 [Issue #641](https://github.com/ikawrakow/ik_llama.cpp/issues/641) - Bug: Vulkan issues with Qwen3-30B-A3B

| **Author** | `samteezy` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2025-07-23 |
| **Updated** | 2025-07-23 |

---

## 📄 Description

### What happened?

@ikawrakow your comment earlier [here](https://github.com/ikawrakow/ik_llama.cpp/issues/464#issuecomment-3105956547) reminded me of why I had `-ngl 0` in the settings I'd been playing with.

In my experimentation, using Vulkan with `Qwen3-30B-A3B` and having _any_ layers on the GPU returns a lot of endless repetition like:

```
Prompt: Tell me how the game mancala is played.

Response:
Thought Process:
Okay, so I need to figure out how to play Mancalas. Wait no, actually, I need to tell the user how the game mancala is played. I mean, I need to, uh... well, I need to think through how to explain it. Let me, let me try to recall what I know. So the game is... Mancala, right? That's a game where, you know, it's a game where the players, uh, the objective is to, well, I mean, I think that it's a game where the players, you know, the players, uh, the player is, well, the player is to, the player is to... I mean, I think that it's a game where you the players, you know, the player, you know, the player, you know, the player is to, well, the player is to, the player is to... I mean, I think that it's a game where you, the player, you know, the player, you know, the player is to, the player is to, the player is to... I mean, I think that the players are to, well, the players are to, the players are to, the players are to, the player is to... I mean, I think that it's a game where you, the player, you know, the player, you know, the player is to, the
```
(I put it out of its misery at that point).

Running with the nearly identical settings in mainline `llama.cpp`, I don't have this issue. (In mainline, because I have both ROCm and Vulkan built, I use `--device Vulkan1`, whereas in `ik_llama.cpp` I use `-mg 1` because I get the error `error: unknown argument: --device`.)

Where Vulkan in `ik_llama.cpp` has been working for me is with non-MoE models like `Qwen3-0.6B` or `Devstral-Small-2507`.

FYI, I'm using unsloth quants mainly across all of these models.

Oh, and for clarity, system specs:
- Xeon 2150B CPU
- Radeon V620 32GB (shows as device 0 with ROCm, device 1 in Vulkan, annoyingly)
- Radeon Pro WX3200 4GB (reverse of above)
- Running in Ubuntu 24.04 container within Proxmox-hosted LXC

### Name and Version

root@llama:~# llama-builds/ik_llama.cpp/bin/llama-server --version
version: 3816 (7093a358)

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell

```

---

## 💬 Conversation

👤 **ikawrakow** commented on **2025-07-23** at **13:38:01**

Is this with or without flash attention? Does changing the flash attention setting change the result? And what does it say about coopmat when initializing the Vulkan back-end?

---

👤 **samteezy** commented on **2025-07-23** at **14:28:34**

Prior runs were with -fa on and quantized cache, still encountering same issue without flash attention.

Last run without flash attention:

```bash
root@llama:~# /root/llama-builds/ik_llama.cpp/bin/llama-cli --threads 10 --n-gpu-layers 99 -mg 1 -ot exps=CPU -m /mnt/models/unsloth/Qwen3-30B-A3B-128K-UD-Q5_K_XL.gguf --temp 0.7 --min-p 0 --top-p 0.8 --top-k 20 --ctx-size 32000 --presence-penalty 0.1 -v --prompt "Tell me how the game mancala is played."
ggml_vulkan: 0 = AMD Radeon (TM) Pro WX 3200 Series (RADV POLARIS12) (radv) | uma: 0 | fp16: 0 | warp size: 64 | shared memory: 65536 | int dot: 0 | matrix cores: none
ggml_vulkan: 1 = AMD Radeon PRO V620 (RADV NAVI21) (radv) | uma: 0 | fp16: 1 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
Log start
main: build = 3816 (7093a358)
main: built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu
main: seed  = 1753280863
llama_model_loader: loaded meta data with 39 key-value pairs and 579 tensors from /mnt/models/unsloth/Qwen3-30B-A3B-128K-UD-Q5_K_XL.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3-30B-A3B-128K
llama_model_loader: - kv   3:                           general.finetune str              = 128k
llama_model_loader: - kv   4:                           general.basename str              = Qwen3-30B-A3B-128K
llama_model_loader: - kv   5:                       general.quantized_by str              = Unsloth
llama_model_loader: - kv   6:                         general.size_label str              = 30B-A3B
llama_model_loader: - kv   7:                           general.repo_url str              = https://huggingface.co/unsloth
llama_model_loader: - kv   8:                       qwen3moe.block_count u32              = 48
llama_model_loader: - kv   9:                    qwen3moe.context_length u32              = 131072
llama_model_loader: - kv  10:                  qwen3moe.embedding_length u32              = 2048
llama_model_loader: - kv  11:               qwen3moe.feed_forward_length u32              = 6144
llama_model_loader: - kv  12:              qwen3moe.attention.head_count u32              = 32
llama_model_loader: - kv  13:           qwen3moe.attention.head_count_kv u32              = 4
llama_model_loader: - kv  14:                    qwen3moe.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  15:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  16:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  17:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  18:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  19:                      qwen3moe.expert_count u32              = 128
llama_model_loader: - kv  20:        qwen3moe.expert_feed_forward_length u32              = 768
llama_model_loader: - kv  21:                 qwen3moe.rope.scaling.type str              = yarn
llama_model_loader: - kv  22:               qwen3moe.rope.scaling.factor f32              = 4.000000
llama_model_loader: - kv  23: qwen3moe.rope.scaling.original_context_length u32              = 32768
llama_model_loader: - kv  24:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  25:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  26:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  27:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  28:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv  29:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  30:            tokenizer.ggml.padding_token_id u32              = 151654
llama_model_loader: - kv  31:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  32:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - kv  33:               general.quantization_version u32              = 2
llama_model_loader: - kv  34:                          general.file_type u32              = 17
llama_model_loader: - kv  35:                      quantize.imatrix.file str              = Qwen3-30B-A3B-128K-GGUF/imatrix_unslo...
llama_model_loader: - kv  36:                   quantize.imatrix.dataset str              = unsloth_calibration_Qwen3-30B-A3B-128...
llama_model_loader: - kv  37:             quantize.imatrix.entries_count i32              = 384
llama_model_loader: - kv  38:              quantize.imatrix.chunks_count i32              = 685
llama_model_loader: - type  f32:  241 tensors
llama_model_loader: - type q8_0:    1 tensors
llama_model_loader: - type q4_K:   20 tensors
llama_model_loader: - type q5_K:  227 tensors
llama_model_loader: - type q6_K:   90 tensors
llm_load_vocab: special tokens cache size = 26
llm_load_vocab: token to piece cache size = 0.9311 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = qwen3moe
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 151936
llm_load_print_meta: n_merges         = 151387
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 131072
llm_load_print_meta: n_embd           = 2048
llm_load_print_meta: n_layer          = 48
llm_load_print_meta: n_head           = 32
llm_load_print_meta: n_head_kv        = 4
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 8
llm_load_print_meta: n_embd_k_gqa     = 512
llm_load_print_meta: n_embd_v_gqa     = 512
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 6144
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
llm_load_print_meta: model ftype      = Q5_K - Medium
llm_load_print_meta: model params     = 30.532 B
llm_load_print_meta: model size       = 20.242 GiB (5.695 BPW) 
llm_load_print_meta: repeating layers = 19.805 GiB (5.688 BPW, 29.910 B parameters)
llm_load_print_meta: general.name     = Qwen3-30B-A3B-128K
llm_load_print_meta: BOS token        = 11 ','
llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
llm_load_print_meta: PAD token        = 151654 '<|vision_pad|>'
llm_load_print_meta: LF token         = 148848 'ÄĬ'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp         = 768
llm_load_tensors: ggml ctx size =    0.76 MiB
Tensor blk.0.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.0.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.0.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.1.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.1.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.1.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.2.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.2.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.2.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_up_exps.weight buffer type overriden to CPU
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
llm_load_tensors: offloading 48 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 49/49 layers to GPU
llm_load_tensors:        CPU buffer size = 20266.31 MiB
llm_load_tensors:        CPU buffer size =   204.02 MiB
llm_load_tensors:    Vulkan1 buffer size =   810.48 MiB
llm_load_tensors:    Vulkan0 buffer size =    82.48 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 32000
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 0
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 0.25
llama_kv_cache_init:    Vulkan1 KV buffer size =  2625.00 MiB
llama_kv_cache_init:    Vulkan0 KV buffer size =   375.00 MiB
llama_new_context_with_model: KV self size  = 3000.00 MiB, K (f16): 1500.00 MiB, V (f16): 1500.00 MiB
llama_new_context_with_model:        CPU  output buffer size =     0.58 MiB
llama_new_context_with_model:    Vulkan0 compute buffer size =  2086.50 MiB
llama_new_context_with_model:    Vulkan1 compute buffer size =  2086.50 MiB
llama_new_context_with_model:        CPU compute buffer size =     0.00 MiB
llama_new_context_with_model: Vulkan_Host compute buffer size =    66.51 MiB
llama_new_context_with_model: graph nodes  = 2165
llama_new_context_with_model: graph splits = 189

system_info: n_threads = 10 / 18 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | 
sampling: 
        repeat_last_n = 64, repeat_penalty = 1.000, frequency_penalty = 0.000, presence_penalty = 0.100
        top_k = 20, tfs_z = 1.000, top_p = 0.800, min_p = 0.000, typical_p = 1.000, temp = 0.700
        mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000
        xtc_probability = 0.000, xtc_threshold = 1.000, top_n_sigma = 0.000
sampling order: 
CFG -> Penalties -> dry -> top_k -> tfs_z -> typical_p -> top_p -> min_p -> xtc -> top_n_sigma -> temperature 
generate: n_ctx = 32000, n_batch = 2048, n_predict = -1, n_keep = 0


Tell me how the game mancala is played. What is the rule of the game? What are the rules of the game? What is the rule of the game? What is the rule of the game? What is the rule of the game? What is the rule of the game? What is the rule of

llama_print_timings:        load time =    6099.71 ms
llama_print_timings:      sample time =     109.50 ms /    53 runs   (    2.07 ms per token,   484.01 tokens per second)
llama_print_timings: prompt eval time =     347.74 ms /    10 tokens (   34.77 ms per token,    28.76 tokens per second)
llama_print_timings:        eval time =    3571.80 ms /    53 runs   (   67.39 ms per token,    14.84 tokens per second)
llama_print_timings:       total time =    4110.00 ms /    63 tokens
```

I don't see any mention of coopmat in the console output, even with -v. What should I be searching for?

---

👤 **samteezy** commented on **2025-07-23** at **14:35:07**

And if it helps, my current build params:

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" -DBUILD_SHARED_LIBS=OFF \
    $CCACHE_FLAG \ #This is part of a script, so I set this dynamically, defaults to "on"
    -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=FLAME \
    -DGGML_VULKAN=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON #allowing all combinations of KV cache
```

---

👤 **ikawrakow** commented on **2025-07-23** at **14:40:38**

I see this when I run with the Vulkan back-end:
```
ggml_vulkan: 0 = NVIDIA GeForce RTX 4080 (NVIDIA) | uma: 0 | fp16: 1 | warp size: 32 | shared memory: 49152 | int dot: 1 | matrix cores: KHR_coopmat
```
If I go to another machine with the same GPU where I have updated the Nvidia driver to the latest and greatest, I see
```
ggml_vulkan: 0 = NVIDIA GeForce RTX 4080 (NVIDIA) | uma: 0 | fp16: 1 | warp size: 32 | shared memory: 49152 | int dot: 1 | matrix cores: NV_coopmat2
```
instead.

With the initial port of Vulkan back-end, some pre-processor macros were not set, and as a result the build was without coop mat enabled. This leads to a horrible performance. 

I'm missing the `ggml_vulkan: ...` output in your log, so not sure when and how your Vulkan back-end gets initialized.

---

👤 **samteezy** commented on **2025-07-23** at **14:51:10**

Well, that explains it... if you look at the top of the logs, you'll see that neither GPU has any matrix cores (hence why coopmat doesn't show in the first place)

---

👤 **ikawrakow** commented on **2025-07-23** at **14:57:44**

Oh, that's the log entry I was looking for, but I missed it because in my case it shows up somewhere else.

OK, the Vulkan port here was never tested without coopmat, so something is likely broken.

---

👤 **ikawrakow** commented on **2025-07-23** at **15:07:41**

OK, so this means that the scalar implementation of one of the non-linear self-attention ops is broken here. If you don't upload anything to the GPU, these ops will run on the CPU, and it works.

I'll try to debug when I come back from vacation in 2 weeks.