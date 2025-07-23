### 🐛 [#500](https://github.com/ikawrakow/ik_llama.cpp/issues/500) - Bug: Insane cudaMalloc OOM Error on Dual 3090 GPUs

| **Author** | `simple6502` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-06 |
| **Updated** | 2025-06-06 |

---

#### Description

### What happened?

When starting up Qwen3-235B-A22B-mix-IQ3_K on my dual 3090 setup with 128GBs DDR4 RAM with cli flag `--split-mode none`, it is able to work just fine on one GPU, but as soon as I remove that flag to use both GPUs, I get an extremely large cudaMalloc OOM error that is trying to allocate hundreds of gigabytes all at once, causing an abort.

Disabling fused MoE, turning off mmap, turning on mlock, and combinations of them does not resolve this issue.

Command used to generate the following logs below:
`./build/bin/llama-server --model /media/nix/Extra/Qwen3-235B-A22B-IQ3_K/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf --alias ubergarm/Qwen3-235B-A22B-mix-IQ3_K -fa -ctk q8_0 -ctv q8_0 -c 32768 -fmoe -amb 512 -rtr -ot blk.1[2-9].ffn.=CPU -ot blk.[2-8][0-9].ffn.=CPU -ot blk.9[0-3].ffn.*=CPU -ngl 99 --threads 16 --host 127.0.0.1 --port 5000`

### Name and Version

```
$./build/bin/llama-server --version
version: 3728 (8ffad187)
built with cc (Debian 12.2.0-14) 12.2.0 for x86_64-linux-gnu
```


### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
ggml_cuda_init: GGML_CUDA_FORCE_MMQ: no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
Device 1: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
INFO [ main] build info | tid="140309264326656" timestamp=1749118021 build=3728 commit="8ffad187"
INFO [ main] system info | tid="140309264326656" timestamp=1749118021 n_threads=16 n_threads_batch=-1 total_threads=32 system_info="AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: additional 2 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 40 key-value pairs and 1131 tensors from /media/nix/Extra/Qwen3-235B-A22B-IQ3_K/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv 0: general.architecture str = qwen3moe
llama_model_loader: - kv 1: general.type str = model
llama_model_loader: - kv 2: general.name str = Qwen3 235B A22B
llama_model_loader: - kv 3: general.basename str = Qwen3
llama_model_loader: - kv 4: general.size_label str = 235B-A22B
llama_model_loader: - kv 5: general.license str = apache-2.0
llama_model_loader: - kv 6: general.license.link str = https://huggingface.co/Qwen/Qwen3-235...
llama_model_loader: - kv 7: general.tags arr[str,1] = ["text-generation"]
llama_model_loader: - kv 8: qwen3moe.block_count u32 = 94
llama_model_loader: - kv 9: qwen3moe.context_length u32 = 40960
llama_model_loader: - kv 10: qwen3moe.embedding_length u32 = 4096
llama_model_loader: - kv 11: qwen3moe.feed_forward_length u32 = 12288
llama_model_loader: - kv 12: qwen3moe.attention.head_count u32 = 64
llama_model_loader: - kv 13: qwen3moe.attention.head_count_kv u32 = 4
llama_model_loader: - kv 14: qwen3moe.rope.freq_base f32 = 1000000.000000
llama_model_loader: - kv 15: qwen3moe.attention.layer_norm_rms_epsilon f32 = 0.000001
llama_model_loader: - kv 16: qwen3moe.expert_used_count u32 = 8
llama_model_loader: - kv 17: qwen3moe.attention.key_length u32 = 128
llama_model_loader: - kv 18: qwen3moe.attention.value_length u32 = 128
llama_model_loader: - kv 19: general.file_type u32 = 139
llama_model_loader: - kv 20: qwen3moe.expert_count u32 = 128
llama_model_loader: - kv 21: qwen3moe.expert_feed_forward_length u32 = 1536
llama_model_loader: - kv 22: general.quantization_version u32 = 2
llama_model_loader: - kv 23: tokenizer.ggml.model str = gpt2
llama_model_loader: - kv 24: tokenizer.ggml.pre str = qwen2
llama_model_loader: - kv 25: tokenizer.ggml.tokens arr[str,151936] = ["!", """, "#", "$", "%", "&", "'", ...
llama_model_loader: - kv 26: tokenizer.ggml.token_type arr[i32,151936] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv 27: tokenizer.ggml.merges arr[str,151387] = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv 28: tokenizer.ggml.eos_token_id u32 = 151645
llama_model_loader: - kv 29: tokenizer.ggml.padding_token_id u32 = 151643
llama_model_loader: - kv 30: tokenizer.ggml.bos_token_id u32 = 151643
llama_model_loader: - kv 31: tokenizer.ggml.add_bos_token bool = false
llama_model_loader: - kv 32: tokenizer.chat_template str = {%- if tools %}\n {{- '<|im_start|>...
llama_model_loader: - kv 33: quantize.imatrix.file str = /mnt/raid/models/ubergarm/Qwen3-235B-...
llama_model_loader: - kv 34: quantize.imatrix.dataset str = calibration_data_v5_rc.txt
llama_model_loader: - kv 35: quantize.imatrix.entries_count i32 = 753
llama_model_loader: - kv 36: quantize.imatrix.chunks_count i32 = 225
llama_model_loader: - kv 37: split.no u16 = 0
llama_model_loader: - kv 38: split.count u16 = 3
llama_model_loader: - kv 39: split.tensors.count i32 = 1131
llama_model_loader: - type f32: 471 tensors
llama_model_loader: - type q8_0: 2 tensors
llama_model_loader: - type iq3_k: 188 tensors
llama_model_loader: - type iq4_k: 94 tensors
llama_model_loader: - type iq6_k: 376 tensors
llm_load_vocab: special tokens cache size = 26
llm_load_vocab: token to piece cache size = 0.9311 MB
llm_load_print_meta: format = GGUF V3 (latest)
llm_load_print_meta: arch = qwen3moe
llm_load_print_meta: vocab type = BPE
llm_load_print_meta: n_vocab = 151936
llm_load_print_meta: n_merges = 151387
llm_load_print_meta: vocab_only = 0
llm_load_print_meta: n_ctx_train = 40960
llm_load_print_meta: n_embd = 4096
llm_load_print_meta: n_layer = 94
llm_load_print_meta: n_head = 64
llm_load_print_meta: n_head_kv = 4
llm_load_print_meta: n_rot = 128
llm_load_print_meta: n_swa = 0
llm_load_print_meta: n_swa_pattern = 1
llm_load_print_meta: n_embd_head_k = 128
llm_load_print_meta: n_embd_head_v = 128
llm_load_print_meta: n_gqa = 16
llm_load_print_meta: n_embd_k_gqa = 512
llm_load_print_meta: n_embd_v_gqa = 512
llm_load_print_meta: f_norm_eps = 0.0e+00
llm_load_print_meta: f_norm_rms_eps = 1.0e-06
llm_load_print_meta: f_clamp_kqv = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale = 0.0e+00
llm_load_print_meta: n_ff = 12288
llm_load_print_meta: n_expert = 128
llm_load_print_meta: n_expert_used = 8
llm_load_print_meta: causal attn = 1
llm_load_print_meta: pooling type = 0
llm_load_print_meta: rope type = 2
llm_load_print_meta: rope scaling = linear
llm_load_print_meta: freq_base_train = 1000000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn = 40960
llm_load_print_meta: rope_finetuned = unknown
llm_load_print_meta: ssm_d_conv = 0
llm_load_print_meta: ssm_d_inner = 0
llm_load_print_meta: ssm_d_state = 0
llm_load_print_meta: ssm_dt_rank = 0
llm_load_print_meta: model type = ?B
llm_load_print_meta: model ftype = IQ3_K - 3.4325 bpw
llm_load_print_meta: model params = 235.094 B
llm_load_print_meta: model size = 106.830 GiB (3.903 BPW)
llm_load_print_meta: repeating layers = 105.598 GiB (3.879 BPW, 233.849 B parameters)
llm_load_print_meta: general.name = Qwen3 235B A22B
llm_load_print_meta: BOS token = 151643 '<|endoftext|>'
llm_load_print_meta: EOS token = 151645 '<|im_end|>'
llm_load_print_meta: PAD token = 151643 '<|endoftext|>'
llm_load_print_meta: LF token = 148848 'ÄĬ'
llm_load_print_meta: EOT token = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp = 1536
llm_load_tensors: ggml ctx size = 1.49 MiB
Tensor blk.12.ffn_norm.weight buffer type overriden to CPU
Tensor blk.12.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_norm.weight buffer type overriden to CPU
Tensor blk.13.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.13.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_norm.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_norm.weight buffer type overriden to CPU

... Cut down to size ...

Tensor blk.92.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_norm.weight buffer type overriden to CPU
Tensor blk.93.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.93.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 94 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 95/95 layers to GPU
llm_load_tensors: CPU buffer size = 89709.28 MiB
llm_load_tensors: CUDA_Host buffer size = 630.59 MiB
llm_load_tensors: CUDA0 buffer size = 15831.98 MiB
llm_load_tensors: CUDA1 buffer size = 3221.75 MiB
....................................................................................................
============ Repacked 246 tensors
llama_new_context_with_model: n_ctx = 32768
llama_new_context_with_model: n_batch = 2048
llama_new_context_with_model: n_ubatch = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn = 0
llama_new_context_with_model: attn_max_b = 512
llama_new_context_with_model: fused_moe = 1
llama_new_context_with_model: ser = -1, 0
llama_new_context_with_model: freq_base = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init: CUDA0 KV buffer size = 1632.02 MiB
llama_kv_cache_init: CUDA1 KV buffer size = 1564.02 MiB
llama_new_context_with_model: KV self size = 3196.00 MiB, K (q8_0): 1598.00 MiB, V (q8_0): 1598.00 MiB
llama_new_context_with_model: CUDA_Host output buffer size = 1.16 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=4)
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 360757.13 MiB on device 0: cudaMalloc failed: out of memory
ggml_gallocr_reserve_n: failed to allocate CUDA0 buffer of size 378281271296
llama_new_context_with_model: failed to allocate compute buffers
llama_init_from_gpt_params: error: failed to create context with model '/media/nix/Extra/Qwen3-235B-A22B-IQ3_K/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf'
ERR [ load_model] unable to load model | tid="140309264326656" timestamp=1749118318 model="/media/nix/Extra/Qwen3-235B-A22B-IQ3_K/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf"
free(): invalid pointer
Aborted
```

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-06-06** at **15:58:12**:<br>

Try `cmake -DGGML_SCHED_MAX_COPIES=1 ...`

I guess, I need to change the default, which is 4, and for some reasons leads to insane memory allocations. Several people have run into the same issue.
 
Also add `--parallel 1` to your command line when starting the server.

---

👤 **simple6502** commented the **2025-06-06** at **16:23:11**:<br>

Perfect! It works fine now and I don't get any more of those issues. Now I can just fine-tune my settings to work best on my system.