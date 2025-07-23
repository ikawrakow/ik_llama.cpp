### 🐛 [#503](https://github.com/ikawrakow/ik_llama.cpp/issues/503) - Bug: server/cli fails with segmentation fault

| **Author** | `OneOfOne` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-07 |
| **Updated** | 2025-06-28 |

---

#### Description

### What happened?

Trying to run: `./build/bin/llama-cli --model /nas/llm/unsloth/Qwen3-32B-GGUF/Qwen3-32B-UD-Q4_K_XL.gguf --alias qwen3-32b-q4_k_xl.gguf --ctx-size 16768 -ctk q8_0 -ctv q8_0 -fa -amb 512 --parallel 1 --n-gpu-layers 65 --threads 12 --override-tensor exps=CPU --port 12345 -p 'whats your name'`

### Name and Version

```ggml_vulkan: Found 1 Vulkan devices:
Vulkan0: AMD Radeon RX 7900 XTX (RADV NAVI31) (radv) | uma: 0 | fp16: 1 | warp size: 64
version: 3732 (9e567e38)
built with cc (GCC) 15.1.1 20250425 for x86_64-pc-linux-gnu```

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
❯ ./build/bin/llama-cli --model /nas/llm/unsloth/Qwen3-32B-GGUF/Qwen3-32B-UD-Q4_K_XL.gguf --alias qwen3-32b-q4_k_xl.gguf --ctx-size 16768 -ctk q8_0 -ctv q8_0 -fa -amb 512 --parallel 1 --n-gpu-layers 65 --threads 16 --override-tensor exps=CPU  --port 12345 -p "what's your name?"
ggml_vulkan: Found 1 Vulkan devices:
Vulkan0: AMD Radeon RX 7900 XTX (RADV NAVI31) (radv) | uma: 0 | fp16: 1 | warp size: 64
Log start
main: build = 3732 (9e567e38)
main: built with cc (GCC) 15.1.1 20250425 for x86_64-pc-linux-gnu
main: seed  = 1749325503
llama_model_loader: loaded meta data with 32 key-value pairs and 707 tensors from /nas/llm/unsloth/Qwen3-32B-GGUF/Qwen3-32B-UD-Q4_K_XL.gguf (version GGUF V3 (latest))
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
llama_model_loader: - kv  27:                          general.file_type u32              = 15
llama_model_loader: - kv  28:                      quantize.imatrix.file str              = Qwen3-32B-GGUF/imatrix_unsloth.dat
llama_model_loader: - kv  29:                   quantize.imatrix.dataset str              = unsloth_calibration_Qwen3-32B.txt
llama_model_loader: - kv  30:             quantize.imatrix.entries_count i32              = 448
llama_model_loader: - kv  31:              quantize.imatrix.chunks_count i32              = 685
llama_model_loader: - type  f32:  257 tensors
llama_model_loader: - type q4_K:  293 tensors
llama_model_loader: - type q5_K:   35 tensors
llama_model_loader: - type q6_K:   94 tensors
llama_model_loader: - type iq4_xs:   28 tensors
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
llm_load_print_meta: model ftype      = Q4_K - Medium
llm_load_print_meta: model params     = 32.762 B
llm_load_print_meta: model size       = 18.641 GiB (4.888 BPW) 
llm_load_print_meta: repeating layers = 17.639 GiB (4.855 BPW, 31.206 B parameters)
llm_load_print_meta: general.name     = Qwen3-32B
llm_load_print_meta: BOS token        = 11 ','
llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
llm_load_print_meta: PAD token        = 151654 '<|vision_pad|>'
llm_load_print_meta: LF token         = 148848 'ÄĬ'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_tensors: ggml ctx size =    0.63 MiB
llm_load_tensors: offloading 64 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 65/65 layers to GPU
llm_load_tensors: AMD Radeon RX 7900 XTX (RADV NAVI31) buffer size = 18671.19 MiB
llm_load_tensors:        CPU buffer size =   417.30 MiB
................................................................................................
llama_new_context_with_model: n_ctx      = 16896
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 512
llama_new_context_with_model: fused_moe  = 0
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init: AMD Radeon RX 7900 XTX (RADV NAVI31) KV buffer size =  2244.00 MiB
llama_new_context_with_model: KV self size  = 2244.00 MiB, K (q8_0): 1122.00 MiB, V (q8_0): 1122.00 MiB
llama_new_context_with_model: Vulkan_Host  output buffer size =     0.58 MiB
llama_new_context_with_model: AMD Radeon RX 7900 XTX (RADV NAVI31) compute buffer size =   306.75 MiB
llama_new_context_with_model: Vulkan_Host compute buffer size =   209.42 MiB
llama_new_context_with_model: graph nodes  = 1734
llama_new_context_with_model: graph splits = 779
[1]    3804384 segmentation fault (core dumped)
```

---

#### 💬 Conversation

👤 **OneOfOne** commented the **2025-06-07** at **20:05:07**:<br>

this only happens with the vulkan backend, I haven't figured out how to use rocm or if it's even supported.

---

👤 **OneOfOne** commented the **2025-06-07** at **20:36:11**:<br>

Narrowed it down to `-ctv / -ctk`, removing them makes the model load, however even with full offloading to the GPU, it's extremely slow.
2 tps vs 35tps on lm studio (vulkan backend).

---

👤 **Ph0rk0z** commented the **2025-06-07** at **22:35:28**:<br>

Since its not a large MOE but a dense model, not sure if there is a reason to use IK for it instead of mainline.

---

👤 **OneOfOne** commented the **2025-06-08** at **02:12:36**:<br>

I wanted to play with the some of the ggufs optimized for ik_llama, so I figured I'd give it a try, doesn't explain why those options don't work and why it's extremely slow with full gpu offload.

---

👤 **saood06** commented the **2025-06-08** at **04:56:55**:<br>

> Since its not a large MOE but a dense model, not sure if there is a reason to use IK for it instead of mainline.

That is not true at all. See this (https://github.com/ikawrakow/ik_llama.cpp/discussions/256#discussioncomment-12496828) for a list of reasons on top of the new quant types and there are so many examples of performance gains over mainline, such as for batched performance see the graph in #171.

Going back to the actual issue, vulkan and rocm may be functioning well in  this repo as they receive very little testing (this is the first I'm hearing of someone trying to use it) and as far as I'm aware have no development here.

---

👤 **ikawrakow** commented the **2025-06-08** at **05:04:08**:<br>

Yes, mainline is a much better place for Vulkan users. There has been zero development or updates to the Vulkan back-end since I forked the project. At that time the `llama.cpp` Vulkan back-end was quite immature. There has been a very active Vulkan development in mainline since then with many performance improvements. ROCm is also never tested, so unclear if it still works.

 > I wanted to play with the some of the ggufs optimized for ik_llama

These quantization types are not implemented in the Vulkan back-end, so it will run on the CPU. That's why you see the very low performance (and if the tensors are loaded on the GPU, it is even slower than just running CPU-only).

---

👤 **OneOfOne** commented the **2025-06-08** at **16:22:15**:<br>

Thanks for the replies and explanation, I'll close this issue for now until I get an nvidia card I guess

---

👤 **ubergarm** commented the **2025-06-28** at **22:48:25**:<br>

@OneOfOne 

Thanks for giving this a try and reporting your findings. Your experience lines up with my own brief exploration which I've documented in this discussion if you have any interest: https://github.com/ikawrakow/ik_llama.cpp/discussions/562

Thanks!