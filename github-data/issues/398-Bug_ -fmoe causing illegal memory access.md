### 🐛 [#398](https://github.com/ikawrakow/ik_llama.cpp/issues/398) - Bug: -fmoe causing illegal memory access

| **Author** | `pt13762104` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-08 |
| **Updated** | 2025-05-23 |

---

#### Description

### What happened?

It seems like when I used Qwen3-30B-A3B with `-fmoe`, an "illegal memory access" always occur after a short period of time. Without `-fmoe`, it works fine.
I'm not sure if this is GPU-related.

### Name and Version

version: 3673 (4084ca73)
built with gcc-14 (Homebrew GCC 14.2.0_1) 14.2.0 for x86_64-pc-linux-gnu


### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
INFO [                    main] build info | tid="133287468544000" timestamp=1746695902 build=3673 commit="4084ca73"
INFO [                    main] system info | tid="133287468544000" timestamp=1746695902 n_threads=2 n_threads_batch=-1 total_threads=4 system_info="AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: loaded meta data with 35 key-value pairs and 579 tensors from /root/Qwen3-30B-A3B-UD-Q4_K_XL.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3-30B-A3B
llama_model_loader: - kv   3:                           general.basename str              = Qwen3-30B-A3B
llama_model_loader: - kv   4:                       general.quantized_by str              = Unsloth
llama_model_loader: - kv   5:                         general.size_label str              = 30B-A3B
llama_model_loader: - kv   6:                           general.repo_url str              = https://huggingface.co/unsloth
llama_model_loader: - kv   7:                       qwen3moe.block_count u32              = 48
llama_model_loader: - kv   8:                    qwen3moe.context_length u32              = 40960
llama_model_loader: - kv   9:                  qwen3moe.embedding_length u32              = 2048
llama_model_loader: - kv  10:               qwen3moe.feed_forward_length u32              = 6144
llama_model_loader: - kv  11:              qwen3moe.attention.head_count u32              = 32
llama_model_loader: - kv  12:           qwen3moe.attention.head_count_kv u32              = 4
llama_model_loader: - kv  13:                    qwen3moe.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  14:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  15:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  16:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  17:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  18:                      qwen3moe.expert_count u32              = 128
llama_model_loader: - kv  19:        qwen3moe.expert_feed_forward_length u32              = 768
llama_model_loader: - kv  20:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  21:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  22:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  23:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  24:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv  25:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  26:            tokenizer.ggml.padding_token_id u32              = 151654
llama_model_loader: - kv  27:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  28:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - kv  29:               general.quantization_version u32              = 2
llama_model_loader: - kv  30:                          general.file_type u32              = 15
llama_model_loader: - kv  31:                      quantize.imatrix.file str              = Qwen3-30B-A3B-GGUF/imatrix_unsloth.dat
llama_model_loader: - kv  32:                   quantize.imatrix.dataset str              = unsloth_calibration_Qwen3-30B-A3B.txt
llama_model_loader: - kv  33:             quantize.imatrix.entries_count i32              = 384
llama_model_loader: - kv  34:              quantize.imatrix.chunks_count i32              = 32
llama_model_loader: - type  f32:  241 tensors
llama_model_loader: - type q4_K:  290 tensors
llama_model_loader: - type q5_K:   37 tensors
llama_model_loader: - type q6_K:   11 tensors
llm_load_vocab: special tokens cache size = 26
llm_load_vocab: token to piece cache size = 0.9311 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = qwen3moe
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 151936
llm_load_print_meta: n_merges         = 151387
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 40960
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
llm_load_print_meta: model params     = 30.532 B
llm_load_print_meta: model size       = 16.493 GiB (4.640 BPW) 
llm_load_print_meta: repeating layers = 16.093 GiB (4.622 BPW, 29.910 B parameters)
llm_load_print_meta: general.name     = Qwen3-30B-A3B
llm_load_print_meta: BOS token        = 11 ','
llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
llm_load_print_meta: PAD token        = 151654 '<|vision_pad|>'
llm_load_print_meta: LF token         = 148848 'ÄĬ'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp         = 768
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: Tesla T4, compute capability 7.5, VMM: yes
  Device 1: Tesla T4, compute capability 7.5, VMM: yes
llm_load_tensors: ggml ctx size =    0.76 MiB
llm_load_tensors: offloading 48 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 49/49 layers to GPU
llm_load_tensors:        CPU buffer size =   166.92 MiB
llm_load_tensors:      CUDA0 buffer size =  8509.23 MiB
llm_load_tensors:      CUDA1 buffer size =  8213.14 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:      CUDA0 KV buffer size =  1600.00 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =  1472.00 MiB
llama_new_context_with_model: KV self size  = 3072.00 MiB, K (f16): 1536.00 MiB, V (f16): 1536.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     1.16 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=4)
llama_new_context_with_model:      CUDA0 compute buffer size =   368.01 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =   444.77 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   260.02 MiB
llama_new_context_with_model: graph nodes  = 1878
llama_new_context_with_model: graph splits = 3
INFO [                    init] initializing slots | tid="133287468544000" timestamp=1746695910 n_slots=1
INFO [                    init] new slot | tid="133287468544000" timestamp=1746695910 id_slot=0 n_ctx_slot=32768
INFO [                    main] model loaded | tid="133287468544000" timestamp=1746695910
INFO [                    main] chat template | tid="133287468544000" timestamp=1746695910 chat_example="<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there<|im_end|>\n<|im_start|>user\nHow are you?<|im_end|>\n<|im_start|>assistant\n" built_in=true
INFO [                    main] HTTP server listening | tid="133287468544000" timestamp=1746695910 n_threads_http="3" port="8080" hostname="127.0.0.1"
INFO [            update_slots] all slots are idle | tid="133287468544000" timestamp=1746695910
INFO [   launch_slot_with_task] slot is processing task | tid="133287468544000" timestamp=1746695926 id_slot=0 id_task=0
INFO [            update_slots] kv cache rm [p0, end) | tid="133287468544000" timestamp=1746695926 id_slot=0 id_task=0 p0=0
INFO [           print_timings] prompt eval time     =    1428.08 ms /   756 tokens (    1.89 ms per token,   529.38 tokens per second) | tid="133287468544000" timestamp=1746695972 id_slot=0 id_task=0 t_prompt_processing=1428.075 n_prompt_tokens_processed=756 t_token=1.8889880952380953 n_tokens_second=529.383960926422
INFO [           print_timings] generation eval time =   44081.50 ms /  2038 runs   (   21.63 ms per token,    46.23 tokens per second) | tid="133287468544000" timestamp=1746695972 id_slot=0 id_task=0 t_token_generation=44081.501 n_decoded=2038 t_token=21.629784592737977 n_tokens_second=46.23254548432914
INFO [           print_timings]           total time =   45509.58 ms | tid="133287468544000" timestamp=1746695972 id_slot=0 id_task=0 t_prompt_processing=1428.075 t_token_generation=44081.501 t_total=45509.575999999994
INFO [            update_slots] slot released | tid="133287468544000" timestamp=1746695972 id_slot=0 id_task=0 n_ctx=32768 n_past=2793 n_system_tokens=0 n_cache_tokens=0 truncated=false
INFO [            update_slots] all slots are idle | tid="133287468544000" timestamp=1746695972
INFO [      log_server_request] request | tid="133286382788608" timestamp=1746695972 remote_addr="127.0.0.1" remote_port=51948 status=200 method="POST" path="/chat/completions" params={}
INFO [            update_slots] all slots are idle | tid="133287468544000" timestamp=1746695972
INFO [   launch_slot_with_task] slot is processing task | tid="133287468544000" timestamp=1746695989 id_slot=0 id_task=2040
INFO [            update_slots] kv cache rm [p0, end) | tid="133287468544000" timestamp=1746695989 id_slot=0 id_task=2040 p0=0
INFO [           print_timings] prompt eval time     =    2259.97 ms /  1480 tokens (    1.53 ms per token,   654.88 tokens per second) | tid="133287468544000" timestamp=1746696002 id_slot=0 id_task=2040 t_prompt_processing=2259.965 n_prompt_tokens_processed=1480 t_token=1.5270033783783785 n_tokens_second=654.8773985437828
INFO [           print_timings] generation eval time =   10276.92 ms /   407 runs   (   25.25 ms per token,    39.60 tokens per second) | tid="133287468544000" timestamp=1746696002 id_slot=0 id_task=2040 t_token_generation=10276.922 n_decoded=407 t_token=25.250422604422607 n_tokens_second=39.603297563219805
INFO [           print_timings]           total time =   12536.89 ms | tid="133287468544000" timestamp=1746696002 id_slot=0 id_task=2040 t_prompt_processing=2259.965 t_token_generation=10276.922 t_total=12536.887
INFO [            update_slots] slot released | tid="133287468544000" timestamp=1746696002 id_slot=0 id_task=2040 n_ctx=32768 n_past=1886 n_system_tokens=0 n_cache_tokens=0 truncated=false
INFO [            update_slots] all slots are idle | tid="133287468544000" timestamp=1746696002
INFO [      log_server_request] request | tid="133286374395904" timestamp=1746696002 remote_addr="127.0.0.1" remote_port=36728 status=200 method="POST" path="/chat/completions" params={}
INFO [            update_slots] all slots are idle | tid="133287468544000" timestamp=1746696002
INFO [   launch_slot_with_task] slot is processing task | tid="133287468544000" timestamp=1746696077 id_slot=0 id_task=2449
INFO [            update_slots] kv cache rm [p0, end) | tid="133287468544000" timestamp=1746696077 id_slot=0 id_task=2449 p0=0
CUDA error: an illegal memory access was encountered
  current device: 1, in function ggml_cuda_up_gate_unary at /kaggle/working/ik_llama.cpp/ggml/src/ggml-cuda.cu:2555
  cudaMemcpyAsync(ids_host.data(), ids_dev, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream)
/kaggle/working/ik_llama.cpp/ggml/src/ggml-cuda.cu:110: CUDA error
```

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-05-08** at **11:11:23**:<br>

Can you add the command line you used? Thanks.

---

👤 **pt13762104** commented the **2025-05-08** at **14:15:50**:<br>

`ik_llama.cpp/build/bin/llama-server -m /root/Qwen3-30B-A3B-UD-Q4_K_XL.gguf -c 32768 -fmoe -fa -ngl 99`
It starts to do this in 2-3 prompts. Maybe it's related to the fact that the T4 doesn't have BF16 capability?

---

👤 **ikawrakow** commented the **2025-05-08** at **14:42:29**:<br>

It is more likely due to a bug that shows up in a multi-GPU setup that I cannot debug because I only have a single GPU.

I have a single 16 GB GPU and run Qwen3-30B-A3B with a pretty good performance using tensor overrides to keep part of the layers on the CPU. For instance,
```
./bin/llama-server -m model -t 16 -ngl 100 -fa -fmoe -rtr -c 32768 -rtr -ot "blk\.[3-4][0-9]\.ffn=CPU"
```
With my Ryzen-7950X CPU the above gives me better performance (~60 t/s) than uploading 35 layers to the GPU (~40 t/s).

If you are up to experimenting, you could try something like the above to run on a single GPU. If that works, it would confirm an issue with `fmoe` with multiple GPUs. You need to use
```
 -ot "blk\.[3-4][0-9]\.ffn=CPU,.*=CUDA0"
```
to put the first 30 layers on the first GPU and everything else on the CPU.

---

👤 **pt13762104** commented the **2025-05-09** at **01:35:39**:<br>

I can't even try this:
```
llm_load_vocab: special tokens cache size = 26
llm_load_vocab: token to piece cache size = 0.9311 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = qwen3moe
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 151936
llm_load_print_meta: n_merges         = 151387
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 40960
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
llm_load_print_meta: model params     = 30.532 B
llm_load_print_meta: model size       = 16.493 GiB (4.640 BPW) 
llm_load_print_meta: repeating layers = 16.093 GiB (4.622 BPW, 29.910 B parameters)
llm_load_print_meta: general.name     = Qwen3-30B-A3B
llm_load_print_meta: BOS token        = 11 ','
llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
llm_load_print_meta: PAD token        = 151654 '<|vision_pad|>'
llm_load_print_meta: LF token         = 148848 'ÄĬ'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp         = 768
llm_load_tensors: ggml ctx size =    0.76 MiB
Tensor token_embd.weight buffer type overriden to CUDA0
Tensor output_norm.weight buffer type overriden to CUDA0
Tensor output.weight buffer type overriden to CUDA0
Tensor blk.0.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.0.attn_q.weight buffer type overriden to CUDA0
Tensor blk.0.attn_k.weight buffer type overriden to CUDA0
Tensor blk.0.attn_v.weight buffer type overriden to CUDA0
Tensor blk.0.attn_output.weight buffer type overriden to CUDA0
Tensor blk.0.attn_k_norm.weight buffer type overriden to CUDA0
Tensor blk.0.attn_q_norm.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.1.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.1.attn_q.weight buffer type overriden to CUDA0
Tensor blk.1.attn_k.weight buffer type overriden to CUDA0
Tensor blk.1.attn_v.weight buffer type overriden to CUDA0
Tensor blk.1.attn_output.weight buffer type overriden to CUDA0
Tensor blk.1.attn_k_norm.weight buffer type overriden to CUDA0
Tensor blk.1.attn_q_norm.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.2.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.2.attn_q.weight buffer type overriden to CUDA0
Tensor blk.2.attn_k.weight buffer type overriden to CUDA0
Tensor blk.2.attn_v.weight buffer type overriden to CUDA0
Tensor blk.2.attn_output.weight buffer type overriden to CUDA0
Tensor blk.2.attn_k_norm.weight buffer type overriden to CUDA0
Tensor blk.2.attn_q_norm.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.3.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.3.attn_q.weight buffer type overriden to CUDA0
Tensor blk.3.attn_k.weight buffer type overriden to CUDA0
Tensor blk.3.attn_v.weight buffer type overriden to CUDA0
Tensor blk.3.attn_output.weight buffer type overriden to CUDA0
Tensor blk.3.attn_k_norm.weight buffer type overriden to CUDA0
Tensor blk.3.attn_q_norm.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.4.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.4.attn_q.weight buffer type overriden to CUDA0
Tensor blk.4.attn_k.weight buffer type overriden to CUDA0
Tensor blk.4.attn_v.weight buffer type overriden to CUDA0
Tensor blk.4.attn_output.weight buffer type overriden to CUDA0
Tensor blk.4.attn_k_norm.weight buffer type overriden to CUDA0
Tensor blk.4.attn_q_norm.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.5.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.5.attn_q.weight buffer type overriden to CUDA0
Tensor blk.5.attn_k.weight buffer type overriden to CUDA0
Tensor blk.5.attn_v.weight buffer type overriden to CUDA0
Tensor blk.5.attn_output.weight buffer type overriden to CUDA0
Tensor blk.5.attn_k_norm.weight buffer type overriden to CUDA0
Tensor blk.5.attn_q_norm.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.6.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.6.attn_q.weight buffer type overriden to CUDA0
Tensor blk.6.attn_k.weight buffer type overriden to CUDA0
Tensor blk.6.attn_v.weight buffer type overriden to CUDA0
Tensor blk.6.attn_output.weight buffer type overriden to CUDA0
Tensor blk.6.attn_k_norm.weight buffer type overriden to CUDA0
Tensor blk.6.attn_q_norm.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.7.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.7.attn_q.weight buffer type overriden to CUDA0
Tensor blk.7.attn_k.weight buffer type overriden to CUDA0
Tensor blk.7.attn_v.weight buffer type overriden to CUDA0
Tensor blk.7.attn_output.weight buffer type overriden to CUDA0
Tensor blk.7.attn_k_norm.weight buffer type overriden to CUDA0
Tensor blk.7.attn_q_norm.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.8.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.8.attn_q.weight buffer type overriden to CUDA0
Tensor blk.8.attn_k.weight buffer type overriden to CUDA0
Tensor blk.8.attn_v.weight buffer type overriden to CUDA0
Tensor blk.8.attn_output.weight buffer type overriden to CUDA0
Tensor blk.8.attn_k_norm.weight buffer type overriden to CUDA0
Tensor blk.8.attn_q_norm.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.9.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.9.attn_q.weight buffer type overriden to CUDA0
Tensor blk.9.attn_k.weight buffer type overriden to CUDA0
Tensor blk.9.attn_v.weight buffer type overriden to CUDA0
Tensor blk.9.attn_output.weight buffer type overriden to CUDA0
Tensor blk.9.attn_k_norm.weight buffer type overriden to CUDA0
Tensor blk.9.attn_q_norm.weight buffer type overriden to CUDA0
Tensor blk.9.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.9.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.9.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.9.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.9.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.10.attn_norm.weight buffer type overriden to CPU
Tensor blk.10.attn_q.weight buffer type overriden to CPU
Tensor blk.10.attn_k.weight buffer type overriden to CPU
Tensor blk.10.attn_v.weight buffer type overriden to CPU
Tensor blk.10.attn_output.weight buffer type overriden to CPU
Tensor blk.10.attn_k_norm.weight buffer type overriden to CPU
Tensor blk.10.attn_q_norm.weight buffer type overriden to CPU
Tensor blk.10.ffn_norm.weight buffer type overriden to CPU
Tensor blk.10.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.11.attn_norm.weight buffer type overriden to CPU
Tensor blk.11.attn_q.weight buffer type overriden to CPU
Tensor blk.11.attn_k.weight buffer type overriden to CPU
Tensor blk.11.attn_v.weight buffer type overriden to CPU
Tensor blk.11.attn_output.weight buffer type overriden to CPU
Tensor blk.11.attn_k_norm.weight buffer type overriden to CPU
Tensor blk.11.attn_q_norm.weight buffer type overriden to CPU
Tensor blk.11.ffn_norm.weight buffer type overriden to CPU
Tensor blk.11.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.11.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.12.attn_norm.weight buffer type overriden to CPU
Tensor blk.12.attn_q.weight buffer type overriden to CPU
Tensor blk.12.attn_k.weight buffer type overriden to CPU
Tensor blk.12.attn_v.weight buffer type overriden to CPU
Tensor blk.12.attn_output.weight buffer type overriden to CPU
Tensor blk.12.attn_k_norm.weight buffer type overriden to CPU
Tensor blk.12.attn_q_norm.weight buffer type overriden to CPU
Tensor blk.12.ffn_norm.weight buffer type overriden to CPU
Tensor blk.12.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.13.attn_norm.weight buffer type overriden to CPU
Tensor blk.13.attn_q.weight buffer type overriden to CPU
Tensor blk.13.attn_k.weight buffer type overriden to CPU
Tensor blk.13.attn_v.weight buffer type overriden to CPU
Tensor blk.13.attn_output.weight buffer type overriden to CPU
Tensor blk.13.attn_k_norm.weight buffer type overriden to CPU
Tensor blk.13.attn_q_norm.weight buffer type overriden to CPU
Tensor blk.13.ffn_norm.weight buffer type overriden to CPU
Tensor blk.13.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.13.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.14.attn_norm.weight buffer type overriden to CPU
Tensor blk.14.attn_q.weight buffer type overriden to CPU
Tensor blk.14.attn_k.weight buffer type overriden to CPU
Tensor blk.14.attn_v.weight buffer type overriden to CPU
Tensor blk.14.attn_output.weight buffer type overriden to CPU
Tensor blk.14.attn_k_norm.weight buffer type overriden to CPU
Tensor blk.14.attn_q_norm.weight buffer type overriden to CPU
Tensor blk.14.ffn_norm.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.15.attn_norm.weight buffer type overriden to CPU
Tensor blk.15.attn_q.weight buffer type overriden to CPU
Tensor blk.15.attn_k.weight buffer type overriden to CPU
Tensor blk.15.attn_v.weight buffer type overriden to CPU
Tensor blk.15.attn_output.weight buffer type overriden to CPU
Tensor blk.15.attn_k_norm.weight buffer type overriden to CPU
Tensor blk.15.attn_q_norm.weight buffer type overriden to CPU
Tensor blk.15.ffn_norm.weight buffer type overriden to CPU
Tensor blk.15.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.16.attn_norm.weight buffer type overriden to CPU
Tensor blk.16.attn_q.weight buffer type overriden to CPU
Tensor blk.16.attn_k.weight buffer type overriden to CPU
Tensor blk.16.attn_v.weight buffer type overriden to CPU
Tensor blk.16.attn_output.weight buffer type overriden to CPU
Tensor blk.16.attn_k_norm.weight buffer type overriden to CPU
Tensor blk.16.attn_q_norm.weight buffer type overriden to CPU
Tensor blk.16.ffn_norm.weight buffer type overriden to CPU
Tensor blk.16.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.16.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.17.attn_norm.weight buffer type overriden to CPU
Tensor blk.17.attn_q.weight buffer type overriden to CPU
Tensor blk.17.attn_k.weight buffer type overriden to CPU
Tensor blk.17.attn_v.weight buffer type overriden to CPU
Tensor blk.17.attn_output.weight buffer type overriden to CPU
Tensor blk.17.attn_k_norm.weight buffer type overriden to CPU
Tensor blk.17.attn_q_norm.weight buffer type overriden to CPU
Tensor blk.17.ffn_norm.weight buffer type overriden to CPU
Tensor blk.17.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.17.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.18.attn_norm.weight buffer type overriden to CPU
Tensor blk.18.attn_q.weight buffer type overriden to CPU
Tensor blk.18.attn_k.weight buffer type overriden to CPU
Tensor blk.18.attn_v.weight buffer type overriden to CPU
Tensor blk.18.attn_output.weight buffer type overriden to CPU
Tensor blk.18.attn_k_norm.weight buffer type overriden to CPU
Tensor blk.18.attn_q_norm.weight buffer type overriden to CPU
Tensor blk.18.ffn_norm.weight buffer type overriden to CPU
Tensor blk.18.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.18.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.19.attn_norm.weight buffer type overriden to CPU
Tensor blk.19.attn_q.weight buffer type overriden to CPU
Tensor blk.19.attn_k.weight buffer type overriden to CPU
Tensor blk.19.attn_v.weight buffer type overriden to CPU
Tensor blk.19.attn_output.weight buffer type overriden to CPU
Tensor blk.19.attn_k_norm.weight buffer type overriden to CPU
Tensor blk.19.attn_q_norm.weight buffer type overriden to CPU
Tensor blk.19.ffn_norm.weight buffer type overriden to CPU
Tensor blk.19.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.19.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.20.attn_norm.weight buffer type overriden to CPU
Tensor blk.20.attn_q.weight buffer type overriden to CPU
Tensor blk.20.attn_k.weight buffer type overriden to CPU
Tensor blk.20.attn_v.weight buffer type overriden to CPU
Tensor blk.20.attn_output.weight buffer type overriden to CPU
Tensor blk.20.attn_k_norm.weight buffer type overriden to CPU
Tensor blk.20.attn_q_norm.weight buffer type overriden to CPU
Tensor blk.20.ffn_norm.weight buffer type overriden to CPU
Tensor blk.20.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.20.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.21.attn_norm.weight buffer type overriden to CPU
Tensor blk.21.attn_q.weight buffer type overriden to CPU
Tensor blk.21.attn_k.weight buffer type overriden to CPU
Tensor blk.21.attn_v.weight buffer type overriden to CPU
Tensor blk.21.attn_output.weight buffer type overriden to CPU
Tensor blk.21.attn_k_norm.weight buffer type overriden to CPU
Tensor blk.21.attn_q_norm.weight buffer type overriden to CPU
Tensor blk.21.ffn_norm.weight buffer type overriden to CPU
Tensor blk.21.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.21.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.22.attn_norm.weight buffer type overriden to CPU
Tensor blk.22.attn_q.weight buffer type overriden to CPU
Tensor blk.22.attn_k.weight buffer type overriden to CPU
Tensor blk.22.attn_v.weight buffer type overriden to CPU
Tensor blk.22.attn_output.weight buffer type overriden to CPU
Tensor blk.22.attn_k_norm.weight buffer type overriden to CPU
Tensor blk.22.attn_q_norm.weight buffer type overriden to CPU
Tensor blk.22.ffn_norm.weight buffer type overriden to CPU
Tensor blk.22.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.22.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.23.attn_norm.weight buffer type overriden to CPU
Tensor blk.23.attn_q.weight buffer type overriden to CPU
Tensor blk.23.attn_k.weight buffer type overriden to CPU
Tensor blk.23.attn_v.weight buffer type overriden to CPU
Tensor blk.23.attn_output.weight buffer type overriden to CPU
Tensor blk.23.attn_k_norm.weight buffer type overriden to CPU
Tensor blk.23.attn_q_norm.weight buffer type overriden to CPU
Tensor blk.23.ffn_norm.weight buffer type overriden to CPU
Tensor blk.23.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.23.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.24.attn_norm.weight buffer type overriden to CPU
Tensor blk.24.attn_q.weight buffer type overriden to CPU
Tensor blk.24.attn_k.weight buffer type overriden to CPU
Tensor blk.24.attn_v.weight buffer type overriden to CPU
Tensor blk.24.attn_output.weight buffer type overriden to CPU
Tensor blk.24.attn_k_norm.weight buffer type overriden to CPU
Tensor blk.24.attn_q_norm.weight buffer type overriden to CPU
Tensor blk.24.ffn_norm.weight buffer type overriden to CPU
Tensor blk.24.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.24.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.25.attn_norm.weight buffer type overriden to CPU
Tensor blk.25.attn_q.weight buffer type overriden to CPU
Tensor blk.25.attn_k.weight buffer type overriden to CPU
Tensor blk.25.attn_v.weight buffer type overriden to CPU
Tensor blk.25.attn_output.weight buffer type overriden to CPU
Tensor blk.25.attn_k_norm.weight buffer type overriden to CPU
Tensor blk.25.attn_q_norm.weight buffer type overriden to CPU
Tensor blk.25.ffn_norm.weight buffer type overriden to CPU
Tensor blk.25.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.25.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.26.attn_norm.weight buffer type overriden to CPU
Tensor blk.26.attn_q.weight buffer type overriden to CPU
Tensor blk.26.attn_k.weight buffer type overriden to CPU
Tensor blk.26.attn_v.weight buffer type overriden to CPU
Tensor blk.26.attn_output.weight buffer type overriden to CPU
Tensor blk.26.attn_k_norm.weight buffer type overriden to CPU
Tensor blk.26.attn_q_norm.weight buffer type overriden to CPU
Tensor blk.26.ffn_norm.weight buffer type overriden to CPU
Tensor blk.26.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.26.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.27.attn_norm.weight buffer type overriden to CPU
Tensor blk.27.attn_q.weight buffer type overriden to CPU
Tensor blk.27.attn_k.weight buffer type overriden to CPU
Tensor blk.27.attn_v.weight buffer type overriden to CPU
Tensor blk.27.attn_output.weight buffer type overriden to CPU
Tensor blk.27.attn_k_norm.weight buffer type overriden to CPU
Tensor blk.27.attn_q_norm.weight buffer type overriden to CPU
Tensor blk.27.ffn_norm.weight buffer type overriden to CPU
Tensor blk.27.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.27.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.28.attn_norm.weight buffer type overriden to CPU
Tensor blk.28.attn_q.weight buffer type overriden to CPU
Tensor blk.28.attn_k.weight buffer type overriden to CPU
Tensor blk.28.attn_v.weight buffer type overriden to CPU
Tensor blk.28.attn_output.weight buffer type overriden to CPU
Tensor blk.28.attn_k_norm.weight buffer type overriden to CPU
Tensor blk.28.attn_q_norm.weight buffer type overriden to CPU
Tensor blk.28.ffn_norm.weight buffer type overriden to CPU
Tensor blk.28.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.28.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.29.attn_norm.weight buffer type overriden to CPU
Tensor blk.29.attn_q.weight buffer type overriden to CPU
Tensor blk.29.attn_k.weight buffer type overriden to CPU
Tensor blk.29.attn_v.weight buffer type overriden to CPU
Tensor blk.29.attn_output.weight buffer type overriden to CPU
Tensor blk.29.attn_k_norm.weight buffer type overriden to CPU
Tensor blk.29.attn_q_norm.weight buffer type overriden to CPU
Tensor blk.29.ffn_norm.weight buffer type overriden to CPU
Tensor blk.29.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.29.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.30.attn_norm.weight buffer type overriden to CPU
Tensor blk.30.attn_q.weight buffer type overriden to CPU
Tensor blk.30.attn_k.weight buffer type overriden to CPU
Tensor blk.30.attn_v.weight buffer type overriden to CPU
Tensor blk.30.attn_output.weight buffer type overriden to CPU
Tensor blk.30.attn_k_norm.weight buffer type overriden to CPU
Tensor blk.30.attn_q_norm.weight buffer type overriden to CPU
Tensor blk.30.ffn_norm.weight buffer type overriden to CPU
Tensor blk.30.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.30.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.31.attn_norm.weight buffer type overriden to CPU
Tensor blk.31.attn_q.weight buffer type overriden to CPU
Tensor blk.31.attn_k.weight buffer type overriden to CPU
Tensor blk.31.attn_v.weight buffer type overriden to CPU
Tensor blk.31.attn_output.weight buffer type overriden to CPU
Tensor blk.31.attn_k_norm.weight buffer type overriden to CPU
Tensor blk.31.attn_q_norm.weight buffer type overriden to CPU
Tensor blk.31.ffn_norm.weight buffer type overriden to CPU
Tensor blk.31.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.31.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.32.attn_norm.weight buffer type overriden to CPU
Tensor blk.32.attn_q.weight buffer type overriden to CPU
Tensor blk.32.attn_k.weight buffer type overriden to CPU
Tensor blk.32.attn_v.weight buffer type overriden to CPU
Tensor blk.32.attn_output.weight buffer type overriden to CPU
Tensor blk.32.attn_k_norm.weight buffer type overriden to CPU
Tensor blk.32.attn_q_norm.weight buffer type overriden to CPU
Tensor blk.32.ffn_norm.weight buffer type overriden to CPU
Tensor blk.32.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.32.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.33.attn_norm.weight buffer type overriden to CPU
Tensor blk.33.attn_q.weight buffer type overriden to CPU
Tensor blk.33.attn_k.weight buffer type overriden to CPU
Tensor blk.33.attn_v.weight buffer type overriden to CPU
Tensor blk.33.attn_output.weight buffer type overriden to CPU
Tensor blk.33.attn_k_norm.weight buffer type overriden to CPU
Tensor blk.33.attn_q_norm.weight buffer type overriden to CPU
Tensor blk.33.ffn_norm.weight buffer type overriden to CPU
Tensor blk.33.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.33.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.34.attn_norm.weight buffer type overriden to CPU
Tensor blk.34.attn_q.weight buffer type overriden to CPU
Tensor blk.34.attn_k.weight buffer type overriden to CPU
Tensor blk.34.attn_v.weight buffer type overriden to CPU
Tensor blk.34.attn_output.weight buffer type overriden to CPU
Tensor blk.34.attn_k_norm.weight buffer type overriden to CPU
Tensor blk.34.attn_q_norm.weight buffer type overriden to CPU
Tensor blk.34.ffn_norm.weight buffer type overriden to CPU
Tensor blk.34.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.34.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.35.attn_norm.weight buffer type overriden to CPU
Tensor blk.35.attn_q.weight buffer type overriden to CPU
Tensor blk.35.attn_k.weight buffer type overriden to CPU
Tensor blk.35.attn_v.weight buffer type overriden to CPU
Tensor blk.35.attn_output.weight buffer type overriden to CPU
Tensor blk.35.attn_k_norm.weight buffer type overriden to CPU
Tensor blk.35.attn_q_norm.weight buffer type overriden to CPU
Tensor blk.35.ffn_norm.weight buffer type overriden to CPU
Tensor blk.35.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.35.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.36.attn_norm.weight buffer type overriden to CPU
Tensor blk.36.attn_q.weight buffer type overriden to CPU
Tensor blk.36.attn_k.weight buffer type overriden to CPU
Tensor blk.36.attn_v.weight buffer type overriden to CPU
Tensor blk.36.attn_output.weight buffer type overriden to CPU
Tensor blk.36.attn_k_norm.weight buffer type overriden to CPU
Tensor blk.36.attn_q_norm.weight buffer type overriden to CPU
Tensor blk.36.ffn_norm.weight buffer type overriden to CPU
Tensor blk.36.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.36.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.37.attn_norm.weight buffer type overriden to CPU
Tensor blk.37.attn_q.weight buffer type overriden to CPU
Tensor blk.37.attn_k.weight buffer type overriden to CPU
Tensor blk.37.attn_v.weight buffer type overriden to CPU
Tensor blk.37.attn_output.weight buffer type overriden to CPU
Tensor blk.37.attn_k_norm.weight buffer type overriden to CPU
Tensor blk.37.attn_q_norm.weight buffer type overriden to CPU
Tensor blk.37.ffn_norm.weight buffer type overriden to CPU
Tensor blk.37.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.37.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.38.attn_norm.weight buffer type overriden to CPU
Tensor blk.38.attn_q.weight buffer type overriden to CPU
Tensor blk.38.attn_k.weight buffer type overriden to CPU
Tensor blk.38.attn_v.weight buffer type overriden to CPU
Tensor blk.38.attn_output.weight buffer type overriden to CPU
Tensor blk.38.attn_k_norm.weight buffer type overriden to CPU
Tensor blk.38.attn_q_norm.weight buffer type overriden to CPU
Tensor blk.38.ffn_norm.weight buffer type overriden to CPU
Tensor blk.38.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.38.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.39.attn_norm.weight buffer type overriden to CPU
Tensor blk.39.attn_q.weight buffer type overriden to CPU
Tensor blk.39.attn_k.weight buffer type overriden to CPU
Tensor blk.39.attn_v.weight buffer type overriden to CPU
Tensor blk.39.attn_output.weight buffer type overriden to CPU
Tensor blk.39.attn_k_norm.weight buffer type overriden to CPU
Tensor blk.39.attn_q_norm.weight buffer type overriden to CPU
Tensor blk.39.ffn_norm.weight buffer type overriden to CPU
Tensor blk.39.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.39.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.40.attn_norm.weight buffer type overriden to CPU
Tensor blk.40.attn_q.weight buffer type overriden to CPU
Tensor blk.40.attn_k.weight buffer type overriden to CPU
Tensor blk.40.attn_v.weight buffer type overriden to CPU
Tensor blk.40.attn_output.weight buffer type overriden to CPU
Tensor blk.40.attn_k_norm.weight buffer type overriden to CPU
Tensor blk.40.attn_q_norm.weight buffer type overriden to CPU
Tensor blk.40.ffn_norm.weight buffer type overriden to CPU
Tensor blk.40.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.40.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.41.attn_norm.weight buffer type overriden to CPU
Tensor blk.41.attn_q.weight buffer type overriden to CPU
Tensor blk.41.attn_k.weight buffer type overriden to CPU
Tensor blk.41.attn_v.weight buffer type overriden to CPU
Tensor blk.41.attn_output.weight buffer type overriden to CPU
Tensor blk.41.attn_k_norm.weight buffer type overriden to CPU
Tensor blk.41.attn_q_norm.weight buffer type overriden to CPU
Tensor blk.41.ffn_norm.weight buffer type overriden to CPU
Tensor blk.41.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.41.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.42.attn_norm.weight buffer type overriden to CPU
Tensor blk.42.attn_q.weight buffer type overriden to CPU
Tensor blk.42.attn_k.weight buffer type overriden to CPU
Tensor blk.42.attn_v.weight buffer type overriden to CPU
Tensor blk.42.attn_output.weight buffer type overriden to CPU
Tensor blk.42.attn_k_norm.weight buffer type overriden to CPU
Tensor blk.42.attn_q_norm.weight buffer type overriden to CPU
Tensor blk.42.ffn_norm.weight buffer type overriden to CPU
Tensor blk.42.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.42.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.43.attn_norm.weight buffer type overriden to CPU
Tensor blk.43.attn_q.weight buffer type overriden to CPU
Tensor blk.43.attn_k.weight buffer type overriden to CPU
Tensor blk.43.attn_v.weight buffer type overriden to CPU
Tensor blk.43.attn_output.weight buffer type overriden to CPU
Tensor blk.43.attn_k_norm.weight buffer type overriden to CPU
Tensor blk.43.attn_q_norm.weight buffer type overriden to CPU
Tensor blk.43.ffn_norm.weight buffer type overriden to CPU
Tensor blk.43.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.43.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.44.attn_norm.weight buffer type overriden to CPU
Tensor blk.44.attn_q.weight buffer type overriden to CPU
Tensor blk.44.attn_k.weight buffer type overriden to CPU
Tensor blk.44.attn_v.weight buffer type overriden to CPU
Tensor blk.44.attn_output.weight buffer type overriden to CPU
Tensor blk.44.attn_k_norm.weight buffer type overriden to CPU
Tensor blk.44.attn_q_norm.weight buffer type overriden to CPU
Tensor blk.44.ffn_norm.weight buffer type overriden to CPU
Tensor blk.44.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.44.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.45.attn_norm.weight buffer type overriden to CPU
Tensor blk.45.attn_q.weight buffer type overriden to CPU
Tensor blk.45.attn_k.weight buffer type overriden to CPU
Tensor blk.45.attn_v.weight buffer type overriden to CPU
Tensor blk.45.attn_output.weight buffer type overriden to CPU
Tensor blk.45.attn_k_norm.weight buffer type overriden to CPU
Tensor blk.45.attn_q_norm.weight buffer type overriden to CPU
Tensor blk.45.ffn_norm.weight buffer type overriden to CPU
Tensor blk.45.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.45.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.46.attn_norm.weight buffer type overriden to CPU
Tensor blk.46.attn_q.weight buffer type overriden to CPU
Tensor blk.46.attn_k.weight buffer type overriden to CPU
Tensor blk.46.attn_v.weight buffer type overriden to CPU
Tensor blk.46.attn_output.weight buffer type overriden to CPU
Tensor blk.46.attn_k_norm.weight buffer type overriden to CPU
Tensor blk.46.attn_q_norm.weight buffer type overriden to CPU
Tensor blk.46.ffn_norm.weight buffer type overriden to CPU
Tensor blk.46.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.46.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.47.attn_norm.weight buffer type overriden to CPU
Tensor blk.47.attn_q.weight buffer type overriden to CPU
Tensor blk.47.attn_k.weight buffer type overriden to CPU
Tensor blk.47.attn_v.weight buffer type overriden to CPU
Tensor blk.47.attn_output.weight buffer type overriden to CPU
Tensor blk.47.attn_k_norm.weight buffer type overriden to CPU
Tensor blk.47.attn_q_norm.weight buffer type overriden to CPU
Tensor blk.47.ffn_norm.weight buffer type overriden to CPU
Tensor blk.47.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.47.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_up_exps.weight buffer type overriden to CPU
llama_model_load: error loading model: failed to allocate buffer
llama_load_model_from_file: failed to load model
llama_init_from_gpt_params: error: failed to load model '/root/Qwen3-30B-A3B-UD-Q4_K_XL.gguf'
 ERR [              load_model] unable to load model | tid="135803250569216" timestamp=1746754485 model="/root/Qwen3-30B-A3B-UD-Q4_K_XL.gguf"
munmap_chunk(): invalid pointer # could be free() or it just disappears
```

---

👤 **pt13762104** commented the **2025-05-09** at **01:36:06**:<br>

Removing `.*=CUDA0` fixed that

---

👤 **pt13762104** commented the **2025-05-09** at **01:36:06**:<br>

Let me try IQ4_K model instead.

---

👤 **pt13762104** commented the **2025-05-09** at **01:59:34**:<br>

@ikawrakow I haven't found issues while using -fmoe on 1 GPU. It seems like a multi-GPU issue, given that the error always occur on device 1. The IQ4_K model doesn't seem to run into this bug.

---

👤 **Ph0rk0z** commented the **2025-05-09** at **11:52:43**:<br>

I'm not sure how it is done here but afaik, real cudaMemcpyAsync is not supported on SM75.

---

👤 **schynce** commented the **2025-05-12** at **18:47:03**:<br>

Hey @ikawrakow and @pt13762104,

I've been running into the exact same "illegal memory access" crash with 3x3090, but not with a specific quant.

I compiled ik_llama.cpp (4ba6bbb) like this:
```
git clone https://github.com/ikawrakow/ik_llama.cpp
cd ik_llama.cpp
cmake -B ./build -DGGML_CUDA=ON -DGGML_BLAS=OFF
cmake --build ./build --config Release -j $(nproc)
```

I have tested different quantizations from HuggingFace:

- IQ4_XS (unsloth/Qwen3-235B-A22B-GGUF)
- i1-Q4_K_S (mradermacher/Qwen3-235B-A22B-i1-GGUF)
- "mix-IQ3_K" (ubergarm/Qwen3-235B-A22B-GGUF)

Only the mix-IQ3_K seems to be working without crashing (and it is a ik_llama.cpp specific). The crash happens regardless of -fmoe. I can run the mix-IQ3_K quant with -fmoe without problems, like this:

```
./llama-server --model /mnt/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf --alias Qwen3-235B-A22B-mix-IQ3_K \
-fa -fmoe -rtr -c 40960 -ctk q8_0 -ctv q8_0 --threads 7 --no-kv-offload \
-ot "blk\.\d+\.attn=CUDA2" \
-ot "blk\.(0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20)\.=CUDA0" \
-ot "blk\.(21|22|23|24|25|26|27|28|29|30|31|32|33|34|35|36|37|38|39|40|41)\.=CUDA1" \
-ot "blk\.(42|43|44|45|46|47|48|49|50|51|52|53|54|55|56|57)\.=CUDA2"
```

On the other hand, this crashes (even if I remove -fmoe):

```
./llama-server --model /mnt/Qwen3-235B-A22B-IQ4_XS-00001-of-00003.gguf --alias Qwen3-235B-A22B-IQ4_XS \
-fa -fmoe -rtr -c 40960 -ctk q8_0 -ctv q8_0 --threads 7 --no-kv-offload \
-ot "blk\.\d+\.attn=CUDA2" \
-ot "blk\.(0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17)\.=CUDA0" \
-ot "blk\.(18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35)\.=CUDA1" \
-ot "blk\.(36|37|38|39|40|41|42|43|44|45|46|47|48|49|50)\.=CUDA2"
```

This is the crash:

```
INFO [      log_server_request] request | tid="140045957632000" timestamp=1746960702 remote_addr="127.0.0.1" remote_port=60492 status=200 method="GET" path="/v1/models" params={}
INFO [   launch_slot_with_task] slot is processing task | tid="140048404189184" timestamp=1746960702 id_slot=0 id_task=373
INFO [            update_slots] kv cache rm [p0, end) | tid="140048404189184" timestamp=1746960702 id_slot=0 id_task=373 p0=3
INFO [      log_server_request] request | tid="140045940846592" timestamp=1746960722 remote_addr="127.0.0.1" remote_port=44428 status=200 method="GET" path="/v1/models" params={}
INFO [            update_slots] kv cache rm [p0, end) | tid="140048404189184" timestamp=1746960741 id_slot=0 id_task=373 p0=2051
INFO [            update_slots] kv cache rm [p0, end) | tid="140048404189184" timestamp=1746960774 id_slot=0 id_task=373 p0=4099
INFO [            update_slots] kv cache rm [p0, end) | tid="140048404189184" timestamp=1746960808 id_slot=0 id_task=373 p0=6147
CUDA error: an illegal memory access was encountered
  current device: 2, in function ggml_backend_cuda_synchronize at /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu:3049
  cudaStreamSynchronize(cuda_ctx->stream())
/home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu:110: CUDA error
Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
ptrace: Operation not permitted.
No stack.
The program is not being run.
Aborted (core dumped)
```

For me, the crashing device is 2. It seems to be changing depending on the offloaded layers?

I would be happy to provide logs or test specific configurations to help debug this.

---

👤 **Ph0rk0z** commented the **2025-05-13** at **11:51:23**:<br>

Oh snap.. that's the FA error?! Try without flash attention and see if it still crashes.

---

👤 **ikawrakow** commented the **2025-05-13** at **12:33:36**:<br>

> Only the mix-IQ3_K seems to be working without crashing (and it is a ik_llama.cpp specific). The crash happens regardless of -fmoe. I can run the mix-IQ3_K quant with -fmoe without problems, like this:

This is useful info. The `IQX_K` quants do not have quantized matrix multiplication implementation, so matrix multiplications are computed via `dequantize -> cuBLAS`. If the illegal memory access does not occur in that case, it would indicate a problem in the quantized matrix multiplication implementation.

The problem is that I cannot trigger the bug on my single-GPU system. I need to get access to a multi-GPU system to be able to debug.

---

👤 **schynce** commented the **2025-05-13** at **22:33:11**:<br>

> Oh snap.. that's the FA error?! Try without flash attention and see if it still crashes.

I tested without -fa with the crashing IQ4_XS quant, like this:

```
./llama-server --model /mnt/Qwen3-235B-A22B-IQ4_XS-00001-of-00003.gguf --alias Qwen3-235B-A22B-IQ4_XS \
-fmoe -rtr -c 40960 --threads 7 --no-kv-offload \
-ot "blk\.\d+\.attn=CUDA2" \
-ot "blk\.(0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17)\.=CUDA0" \
-ot "blk\.(18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35)\.=CUDA1" \
-ot "blk\.(36|37|38|39|40|41|42|43|44|45|46|47|48|49|50)\.=CUDA2"
```

The prompt processing speed is absolutely glacial, but it does not seem to be crashing.

Long prompts seemed to reliably crash it before with flash attention. So, I ran the same 32K token prompt I used to test earlier through it like this. It took almost an hour to complete, but did so without incident. I also chatted with it a bit.

---

👤 **Panchovix** commented the **2025-05-14** at **16:32:23**:<br>

Just chiming in, I get a CUDA illegal memory access when using -fmoe on DeepSeekV3 0324

```
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init:      CUDA0 KV buffer size =   468.00 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =   360.00 MiB
llama_kv_cache_init:      CUDA2 KV buffer size =   360.00 MiB
llama_kv_cache_init:      CUDA3 KV buffer size =   360.00 MiB
llama_kv_cache_init:      CUDA4 KV buffer size =   648.00 MiB
llama_new_context_with_model: KV self size  = 2196.00 MiB, c^KV (f16): 2196.00 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.99 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      CUDA0 compute buffer size =  3520.01 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =  1540.01 MiB
llama_new_context_with_model:      CUDA2 compute buffer size =  1540.01 MiB
llama_new_context_with_model:      CUDA3 compute buffer size =  1540.01 MiB
llama_new_context_with_model:      CUDA4 compute buffer size =  1540.02 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   312.02 MiB
llama_new_context_with_model: graph nodes  = 3304
llama_new_context_with_model: graph splits = 393
INFO [                    init] initializing slots | tid="140562497785856" timestamp=1747239254 n_slots=1
INFO [                    init] new slot | tid="140562497785856" timestamp=1747239254 id_slot=0 n_ctx_slot=32768
INFO [                    main] model loaded | tid="140562497785856" timestamp=1747239254
INFO [                    main] chat template | tid="140562497785856" timestamp=1747239254 chat_example="You are a helpful assistant\n\n<｜User｜>Hello<｜Assistant｜>Hi there<｜end▁of▁sentence｜><｜User｜>How are you?<｜Assistant｜>" built_in=true
INFO [                    main] HTTP server listening | tid="140562497785856" timestamp=1747239254 n_threads_http="15" port="8080" hostname="127.0.0.1"
INFO [            update_slots] all slots are idle | tid="140562497785856" timestamp=1747239254
INFO [   launch_slot_with_task] slot is processing task | tid="140562497785856" timestamp=1747239313 id_slot=0 id_task=0
INFO [            update_slots] kv cache rm [p0, end) | tid="140562497785856" timestamp=1747239313 id_slot=0 id_task=0 p0=0
CUDA error: an illegal memory access was encountered
  current device: 0, in function ggml_cuda_op_mul_mat at /run/media/pancho/4C4643C74643B10E/ChatIAs/ik_llama.cpp/ggml/src/ggml-cuda.cu:1743
  cudaGetLastError()
/run/media/pancho/4C4643C74643B10E/ChatIAs/ik_llama.cpp/ggml/src/ggml-cuda.cu:110: CUDA error
[New LWP 25355]
[New LWP 25354]
[New LWP 25353]
[New LWP 25352]
[New LWP 25351]
[New LWP 25350]
[New LWP 25349]
[New LWP 25348]
[New LWP 25347]
[New LWP 25346]
[New LWP 25345]
[New LWP 25344]
[New LWP 25343]
[New LWP 25342]
[New LWP 25341]
[New LWP 25340]
[New LWP 24655]
[New LWP 24654]
[New LWP 24653]
[New LWP 24652]
[New LWP 24651]
[New LWP 24650]
[New LWP 24649]
[New LWP 23954]
[New LWP 23953]
[New LWP 23952]
[New LWP 23951]
[New LWP 23950]
[New LWP 23949]
[New LWP 23948]
[New LWP 23947]
[New LWP 23942]
[New LWP 23941]
[New LWP 23940]

This GDB supports auto-downloading debuginfo from the following URLs:
  <https://debuginfod.fedoraproject.org/>
Enable debuginfod for this session? (y or [n]) [answered N; input not from terminal]
Debuginfod has been disabled.
To make this setting permanent, add 'set debuginfod enabled off' to .gdbinit.
Function(s) ^std::(move|forward|as_const|(__)?addressof) will be skipped when stepping.
Function(s) ^std::(shared|unique)_ptr<.*>::(get|operator) will be skipped when stepping.
Function(s) ^std::(basic_string|vector|array|deque|(forward_)?list|(unordered_|flat_)?(multi)?(map|set)|span)<.*>::(c?r?(begin|end)|front|back|data|size|empty) will be skipped when stepping.
Function(s) ^std::(basic_string|vector|array|deque|span)<.*>::operator.] will be skipped when stepping.
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib64/libthread_db.so.1".
0x00007fd73d0876c2 in __syscall_cancel_arch () from /lib64/libc.so.6
#0  0x00007fd73d0876c2 in __syscall_cancel_arch () from /lib64/libc.so.6
#1  0x00007fd73d07b9da in __internal_syscall_cancel () from /lib64/libc.so.6
#2  0x00007fd73d07ba24 in __syscall_cancel () from /lib64/libc.so.6
#3  0x00007fd73d0eb5af in wait4 () from /lib64/libc.so.6
#4  0x00007fd741c58908 in ggml_abort () from /run/media/pancho/4C4643C74643B10E/ChatIAs/ik_llama.cpp/lenux/ggml/src/libggml.so
#5  0x00007fd741dded43 in ggml_cuda_error(char const*, char const*, char const*, int, char const*) () from /run/media/pancho/4C4643C74643B10E/ChatIAs/ik_llama.cpp/lenux/ggml/src/libggml.so
#6  0x00007fd741decb09 in ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [clone .constprop.1] () from /run/media/pancho/4C4643C74643B10E/ChatIAs/ik_llama.cpp/lenux/ggml/src/libggml.so
#7  0x00007fd741df42dd in ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) () from /run/media/pancho/4C4643C74643B10E/ChatIAs/ik_llama.cpp/lenux/ggml/src/libggml.so
#8  0x00007fd741caf9b3 in ggml_backend_sched_graph_compute_async () from /run/media/pancho/4C4643C74643B10E/ChatIAs/ik_llama.cpp/lenux/ggml/src/libggml.so
#9  0x00007fd79656af1a in llama_decode () from /run/media/pancho/4C4643C74643B10E/ChatIAs/ik_llama.cpp/lenux/src/libllama.so
#10 0x000000000049a2d4 in server_context::update_slots() ()
#11 0x000000000046cafc in server_queue::start_loop() ()
#12 0x0000000000416977 in main ()
[Inferior 1 (process 23939) detached]
```

Ran it with

```
./llama-server -m '/models_llm/DeepSeek-V3-0324-UD-Q3_K_XL-00001-of-00007.gguf' -c 32768 --no-mmap -ngl 999 -ot "blk.(0|1|2|3|4|5|6).ffn.=CUDA0" -ot "blk.(7|8|9|10).ffn.=CUDA1" -ot "blk.(11|12|13|14).ffn.=CUDA2" -ot "blk.(15|16|17).ffn.=CUDA3"  -ot "blk.(18|19|20|21|22|23|24|25).ffn.=CUDA4" -ot "ffn.*=CPU" -fa -mg 0 -ub 2048 -mla 1 -fmoe
```

Not using -fmoe makes it work without issues.

---

👤 **Panchovix** commented the **2025-05-14** at **16:32:23**:<br>

Just chiming in, I get a CUDA illegal memory access when using -fmoe on DeepSeekV3 0324

```
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init:      CUDA0 KV buffer size =   468.00 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =   360.00 MiB
llama_kv_cache_init:      CUDA2 KV buffer size =   360.00 MiB
llama_kv_cache_init:      CUDA3 KV buffer size =   360.00 MiB
llama_kv_cache_init:      CUDA4 KV buffer size =   648.00 MiB
llama_new_context_with_model: KV self size  = 2196.00 MiB, c^KV (f16): 2196.00 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.99 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      CUDA0 compute buffer size =  3520.01 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =  1540.01 MiB
llama_new_context_with_model:      CUDA2 compute buffer size =  1540.01 MiB
llama_new_context_with_model:      CUDA3 compute buffer size =  1540.01 MiB
llama_new_context_with_model:      CUDA4 compute buffer size =  1540.02 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   312.02 MiB
llama_new_context_with_model: graph nodes  = 3304
llama_new_context_with_model: graph splits = 393
INFO [                    init] initializing slots | tid="140562497785856" timestamp=1747239254 n_slots=1
INFO [                    init] new slot | tid="140562497785856" timestamp=1747239254 id_slot=0 n_ctx_slot=32768
INFO [                    main] model loaded | tid="140562497785856" timestamp=1747239254
INFO [                    main] chat template | tid="140562497785856" timestamp=1747239254 chat_example="You are a helpful assistant\n\n<｜User｜>Hello<｜Assistant｜>Hi there<｜end▁of▁sentence｜><｜User｜>How are you?<｜Assistant｜>" built_in=true
INFO [                    main] HTTP server listening | tid="140562497785856" timestamp=1747239254 n_threads_http="15" port="8080" hostname="127.0.0.1"
INFO [            update_slots] all slots are idle | tid="140562497785856" timestamp=1747239254
INFO [   launch_slot_with_task] slot is processing task | tid="140562497785856" timestamp=1747239313 id_slot=0 id_task=0
INFO [            update_slots] kv cache rm [p0, end) | tid="140562497785856" timestamp=1747239313 id_slot=0 id_task=0 p0=0
CUDA error: an illegal memory access was encountered
  current device: 0, in function ggml_cuda_op_mul_mat at /run/media/pancho/4C4643C74643B10E/ChatIAs/ik_llama.cpp/ggml/src/ggml-cuda.cu:1743
  cudaGetLastError()
/run/media/pancho/4C4643C74643B10E/ChatIAs/ik_llama.cpp/ggml/src/ggml-cuda.cu:110: CUDA error
[New LWP 25355]
[New LWP 25354]
[New LWP 25353]
[New LWP 25352]
[New LWP 25351]
[New LWP 25350]
[New LWP 25349]
[New LWP 25348]
[New LWP 25347]
[New LWP 25346]
[New LWP 25345]
[New LWP 25344]
[New LWP 25343]
[New LWP 25342]
[New LWP 25341]
[New LWP 25340]
[New LWP 24655]
[New LWP 24654]
[New LWP 24653]
[New LWP 24652]
[New LWP 24651]
[New LWP 24650]
[New LWP 24649]
[New LWP 23954]
[New LWP 23953]
[New LWP 23952]
[New LWP 23951]
[New LWP 23950]
[New LWP 23949]
[New LWP 23948]
[New LWP 23947]
[New LWP 23942]
[New LWP 23941]
[New LWP 23940]

This GDB supports auto-downloading debuginfo from the following URLs:
  <https://debuginfod.fedoraproject.org/>
Enable debuginfod for this session? (y or [n]) [answered N; input not from terminal]
Debuginfod has been disabled.
To make this setting permanent, add 'set debuginfod enabled off' to .gdbinit.
Function(s) ^std::(move|forward|as_const|(__)?addressof) will be skipped when stepping.
Function(s) ^std::(shared|unique)_ptr<.*>::(get|operator) will be skipped when stepping.
Function(s) ^std::(basic_string|vector|array|deque|(forward_)?list|(unordered_|flat_)?(multi)?(map|set)|span)<.*>::(c?r?(begin|end)|front|back|data|size|empty) will be skipped when stepping.
Function(s) ^std::(basic_string|vector|array|deque|span)<.*>::operator.] will be skipped when stepping.
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib64/libthread_db.so.1".
0x00007fd73d0876c2 in __syscall_cancel_arch () from /lib64/libc.so.6
#0  0x00007fd73d0876c2 in __syscall_cancel_arch () from /lib64/libc.so.6
#1  0x00007fd73d07b9da in __internal_syscall_cancel () from /lib64/libc.so.6
#2  0x00007fd73d07ba24 in __syscall_cancel () from /lib64/libc.so.6
#3  0x00007fd73d0eb5af in wait4 () from /lib64/libc.so.6
#4  0x00007fd741c58908 in ggml_abort () from /run/media/pancho/4C4643C74643B10E/ChatIAs/ik_llama.cpp/lenux/ggml/src/libggml.so
#5  0x00007fd741dded43 in ggml_cuda_error(char const*, char const*, char const*, int, char const*) () from /run/media/pancho/4C4643C74643B10E/ChatIAs/ik_llama.cpp/lenux/ggml/src/libggml.so
#6  0x00007fd741decb09 in ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [clone .constprop.1] () from /run/media/pancho/4C4643C74643B10E/ChatIAs/ik_llama.cpp/lenux/ggml/src/libggml.so
#7  0x00007fd741df42dd in ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) () from /run/media/pancho/4C4643C74643B10E/ChatIAs/ik_llama.cpp/lenux/ggml/src/libggml.so
#8  0x00007fd741caf9b3 in ggml_backend_sched_graph_compute_async () from /run/media/pancho/4C4643C74643B10E/ChatIAs/ik_llama.cpp/lenux/ggml/src/libggml.so
#9  0x00007fd79656af1a in llama_decode () from /run/media/pancho/4C4643C74643B10E/ChatIAs/ik_llama.cpp/lenux/src/libllama.so
#10 0x000000000049a2d4 in server_context::update_slots() ()
#11 0x000000000046cafc in server_queue::start_loop() ()
#12 0x0000000000416977 in main ()
[Inferior 1 (process 23939) detached]
```

Ran it with

```
./llama-server -m '/models_llm/DeepSeek-V3-0324-UD-Q3_K_XL-00001-of-00007.gguf' -c 32768 --no-mmap -ngl 999 -ot "blk.(0|1|2|3|4|5|6).ffn.=CUDA0" -ot "blk.(7|8|9|10).ffn.=CUDA1" -ot "blk.(11|12|13|14).ffn.=CUDA2" -ot "blk.(15|16|17).ffn.=CUDA3"  -ot "blk.(18|19|20|21|22|23|24|25).ffn.=CUDA4" -ot "ffn.*=CPU" -fa -mg 0 -ub 2048 -mla 1
```

Not using -fmoe makes it work without issues.

---

👤 **p4s2wd** commented the **2025-05-15** at **00:13:20**:<br>

> 顺便说一下，我在 DeepSeekV3 0324 上使用 -fmoe 时遇到了 CUDA 非法内存访问
> 
> ```
> llama_new_context_with_model: freq_scale = 0.025
> llama_kv_cache_init:      CUDA0 KV buffer size =   468.00 MiB
> llama_kv_cache_init:      CUDA1 KV buffer size =   360.00 MiB
> llama_kv_cache_init:      CUDA2 KV buffer size =   360.00 MiB
> llama_kv_cache_init:      CUDA3 KV buffer size =   360.00 MiB
> llama_kv_cache_init:      CUDA4 KV buffer size =   648.00 MiB
> llama_new_context_with_model: KV self size  = 2196.00 MiB, c^KV (f16): 2196.00 MiB, kv^T: not used
> llama_new_context_with_model:  CUDA_Host  output buffer size =     0.99 MiB
> llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
> llama_new_context_with_model:      CUDA0 compute buffer size =  3520.01 MiB
> llama_new_context_with_model:      CUDA1 compute buffer size =  1540.01 MiB
> llama_new_context_with_model:      CUDA2 compute buffer size =  1540.01 MiB
> llama_new_context_with_model:      CUDA3 compute buffer size =  1540.01 MiB
> llama_new_context_with_model:      CUDA4 compute buffer size =  1540.02 MiB
> llama_new_context_with_model:  CUDA_Host compute buffer size =   312.02 MiB
> llama_new_context_with_model: graph nodes  = 3304
> llama_new_context_with_model: graph splits = 393
> INFO [                    init] initializing slots | tid="140562497785856" timestamp=1747239254 n_slots=1
> INFO [                    init] new slot | tid="140562497785856" timestamp=1747239254 id_slot=0 n_ctx_slot=32768
> INFO [                    main] model loaded | tid="140562497785856" timestamp=1747239254
> INFO [                    main] chat template | tid="140562497785856" timestamp=1747239254 chat_example="You are a helpful assistant\n\n<｜User｜>Hello<｜Assistant｜>Hi there<｜end▁of▁sentence｜><｜User｜>How are you?<｜Assistant｜>" built_in=true
> INFO [                    main] HTTP server listening | tid="140562497785856" timestamp=1747239254 n_threads_http="15" port="8080" hostname="127.0.0.1"
> INFO [            update_slots] all slots are idle | tid="140562497785856" timestamp=1747239254
> INFO [   launch_slot_with_task] slot is processing task | tid="140562497785856" timestamp=1747239313 id_slot=0 id_task=0
> INFO [            update_slots] kv cache rm [p0, end) | tid="140562497785856" timestamp=1747239313 id_slot=0 id_task=0 p0=0
> CUDA error: an illegal memory access was encountered
>   current device: 0, in function ggml_cuda_op_mul_mat at /run/media/pancho/4C4643C74643B10E/ChatIAs/ik_llama.cpp/ggml/src/ggml-cuda.cu:1743
>   cudaGetLastError()
> /run/media/pancho/4C4643C74643B10E/ChatIAs/ik_llama.cpp/ggml/src/ggml-cuda.cu:110: CUDA error
> [New LWP 25355]
> [New LWP 25354]
> [New LWP 25353]
> [New LWP 25352]
> [New LWP 25351]
> [New LWP 25350]
> [New LWP 25349]
> [New LWP 25348]
> [New LWP 25347]
> [New LWP 25346]
> [New LWP 25345]
> [New LWP 25344]
> [New LWP 25343]
> [New LWP 25342]
> [New LWP 25341]
> [New LWP 25340]
> [New LWP 24655]
> [New LWP 24654]
> [New LWP 24653]
> [New LWP 24652]
> [New LWP 24651]
> [New LWP 24650]
> [New LWP 24649]
> [New LWP 23954]
> [New LWP 23953]
> [New LWP 23952]
> [New LWP 23951]
> [New LWP 23950]
> [New LWP 23949]
> [New LWP 23948]
> [New LWP 23947]
> [New LWP 23942]
> [New LWP 23941]
> [New LWP 23940]
> 
> This GDB supports auto-downloading debuginfo from the following URLs:
>   <https://debuginfod.fedoraproject.org/>
> Enable debuginfod for this session? (y or [n]) [answered N; input not from terminal]
> Debuginfod has been disabled.
> To make this setting permanent, add 'set debuginfod enabled off' to .gdbinit.
> Function(s) ^std::(move|forward|as_const|(__)?addressof) will be skipped when stepping.
> Function(s) ^std::(shared|unique)_ptr<.*>::(get|operator) will be skipped when stepping.
> Function(s) ^std::(basic_string|vector|array|deque|(forward_)?list|(unordered_|flat_)?(multi)?(map|set)|span)<.*>::(c?r?(begin|end)|front|back|data|size|empty) will be skipped when stepping.
> Function(s) ^std::(basic_string|vector|array|deque|span)<.*>::operator.] will be skipped when stepping.
> [Thread debugging using libthread_db enabled]
> Using host libthread_db library "/lib64/libthread_db.so.1".
> 0x00007fd73d0876c2 in __syscall_cancel_arch () from /lib64/libc.so.6
> #0  0x00007fd73d0876c2 in __syscall_cancel_arch () from /lib64/libc.so.6
> #1  0x00007fd73d07b9da in __internal_syscall_cancel () from /lib64/libc.so.6
> #2  0x00007fd73d07ba24 in __syscall_cancel () from /lib64/libc.so.6
> #3  0x00007fd73d0eb5af in wait4 () from /lib64/libc.so.6
> #4  0x00007fd741c58908 in ggml_abort () from /run/media/pancho/4C4643C74643B10E/ChatIAs/ik_llama.cpp/lenux/ggml/src/libggml.so
> #5  0x00007fd741dded43 in ggml_cuda_error(char const*, char const*, char const*, int, char const*) () from /run/media/pancho/4C4643C74643B10E/ChatIAs/ik_llama.cpp/lenux/ggml/src/libggml.so
> #6  0x00007fd741decb09 in ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [clone .constprop.1] () from /run/media/pancho/4C4643C74643B10E/ChatIAs/ik_llama.cpp/lenux/ggml/src/libggml.so
> #7  0x00007fd741df42dd in ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) () from /run/media/pancho/4C4643C74643B10E/ChatIAs/ik_llama.cpp/lenux/ggml/src/libggml.so
> #8  0x00007fd741caf9b3 in ggml_backend_sched_graph_compute_async () from /run/media/pancho/4C4643C74643B10E/ChatIAs/ik_llama.cpp/lenux/ggml/src/libggml.so
> #9  0x00007fd79656af1a in llama_decode () from /run/media/pancho/4C4643C74643B10E/ChatIAs/ik_llama.cpp/lenux/src/libllama.so
> #10 0x000000000049a2d4 in server_context::update_slots() ()
> #11 0x000000000046cafc in server_queue::start_loop() ()
> #12 0x0000000000416977 in main ()
> [Inferior 1 (process 23939) detached]
> ```
> 
> 运行它
> 
> ```
> ./llama-server -m '/models_llm/DeepSeek-V3-0324-UD-Q3_K_XL-00001-of-00007.gguf' -c 32768 --no-mmap -ngl 999 -ot "blk.(0|1|2|3|4|5|6).ffn.=CUDA0" -ot "blk.(7|8|9|10).ffn.=CUDA1" -ot "blk.(11|12|13|14).ffn.=CUDA2" -ot "blk.(15|16|17).ffn.=CUDA3"  -ot "blk.(18|19|20|21|22|23|24|25).ffn.=CUDA4" -ot "ffn.*=CPU" -fa -mg 0 -ub 2048 -mla 1 -fmoe
> ```
> 
> 不使用 -fm

---

👤 **p4s2wd** commented the **2025-05-15** at **00:21:27**:<br>

> Just chiming in, I get a CUDA illegal memory access when using -fmoe on DeepSeekV3 0324
> 
> ```
> llama_new_context_with_model: freq_scale = 0.025
> llama_kv_cache_init:      CUDA0 KV buffer size =   468.00 MiB
> llama_kv_cache_init:      CUDA1 KV buffer size =   360.00 MiB
> llama_kv_cache_init:      CUDA2 KV buffer size =   360.00 MiB
> llama_kv_cache_init:      CUDA3 KV buffer size =   360.00 MiB
> llama_kv_cache_init:      CUDA4 KV buffer size =   648.00 MiB
> llama_new_context_with_model: KV self size  = 2196.00 MiB, c^KV (f16): 2196.00 MiB, kv^T: not used
> llama_new_context_with_model:  CUDA_Host  output buffer size =     0.99 MiB
> llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
> llama_new_context_with_model:      CUDA0 compute buffer size =  3520.01 MiB
> llama_new_context_with_model:      CUDA1 compute buffer size =  1540.01 MiB
> llama_new_context_with_model:      CUDA2 compute buffer size =  1540.01 MiB
> llama_new_context_with_model:      CUDA3 compute buffer size =  1540.01 MiB
> llama_new_context_with_model:      CUDA4 compute buffer size =  1540.02 MiB
> llama_new_context_with_model:  CUDA_Host compute buffer size =   312.02 MiB
> llama_new_context_with_model: graph nodes  = 3304
> llama_new_context_with_model: graph splits = 393
> INFO [                    init] initializing slots | tid="140562497785856" timestamp=1747239254 n_slots=1
> INFO [                    init] new slot | tid="140562497785856" timestamp=1747239254 id_slot=0 n_ctx_slot=32768
> INFO [                    main] model loaded | tid="140562497785856" timestamp=1747239254
> INFO [                    main] chat template | tid="140562497785856" timestamp=1747239254 chat_example="You are a helpful assistant\n\n<｜User｜>Hello<｜Assistant｜>Hi there<｜end▁of▁sentence｜><｜User｜>How are you?<｜Assistant｜>" built_in=true
> INFO [                    main] HTTP server listening | tid="140562497785856" timestamp=1747239254 n_threads_http="15" port="8080" hostname="127.0.0.1"
> INFO [            update_slots] all slots are idle | tid="140562497785856" timestamp=1747239254
> INFO [   launch_slot_with_task] slot is processing task | tid="140562497785856" timestamp=1747239313 id_slot=0 id_task=0
> INFO [            update_slots] kv cache rm [p0, end) | tid="140562497785856" timestamp=1747239313 id_slot=0 id_task=0 p0=0
> CUDA error: an illegal memory access was encountered
>   current device: 0, in function ggml_cuda_op_mul_mat at /run/media/pancho/4C4643C74643B10E/ChatIAs/ik_llama.cpp/ggml/src/ggml-cuda.cu:1743
>   cudaGetLastError()
> /run/media/pancho/4C4643C74643B10E/ChatIAs/ik_llama.cpp/ggml/src/ggml-cuda.cu:110: CUDA error
> [New LWP 25355]
> [New LWP 25354]
> [New LWP 25353]
> [New LWP 25352]
> [New LWP 25351]
> [New LWP 25350]
> [New LWP 25349]
> [New LWP 25348]
> [New LWP 25347]
> [New LWP 25346]
> [New LWP 25345]
> [New LWP 25344]
> [New LWP 25343]
> [New LWP 25342]
> [New LWP 25341]
> [New LWP 25340]
> [New LWP 24655]
> [New LWP 24654]
> [New LWP 24653]
> [New LWP 24652]
> [New LWP 24651]
> [New LWP 24650]
> [New LWP 24649]
> [New LWP 23954]
> [New LWP 23953]
> [New LWP 23952]
> [New LWP 23951]
> [New LWP 23950]
> [New LWP 23949]
> [New LWP 23948]
> [New LWP 23947]
> [New LWP 23942]
> [New LWP 23941]
> [New LWP 23940]
> 
> This GDB supports auto-downloading debuginfo from the following URLs:
>   <https://debuginfod.fedoraproject.org/>
> Enable debuginfod for this session? (y or [n]) [answered N; input not from terminal]
> Debuginfod has been disabled.
> To make this setting permanent, add 'set debuginfod enabled off' to .gdbinit.
> Function(s) ^std::(move|forward|as_const|(__)?addressof) will be skipped when stepping.
> Function(s) ^std::(shared|unique)_ptr<.*>::(get|operator) will be skipped when stepping.
> Function(s) ^std::(basic_string|vector|array|deque|(forward_)?list|(unordered_|flat_)?(multi)?(map|set)|span)<.*>::(c?r?(begin|end)|front|back|data|size|empty) will be skipped when stepping.
> Function(s) ^std::(basic_string|vector|array|deque|span)<.*>::operator.] will be skipped when stepping.
> [Thread debugging using libthread_db enabled]
> Using host libthread_db library "/lib64/libthread_db.so.1".
> 0x00007fd73d0876c2 in __syscall_cancel_arch () from /lib64/libc.so.6
> #0  0x00007fd73d0876c2 in __syscall_cancel_arch () from /lib64/libc.so.6
> #1  0x00007fd73d07b9da in __internal_syscall_cancel () from /lib64/libc.so.6
> #2  0x00007fd73d07ba24 in __syscall_cancel () from /lib64/libc.so.6
> #3  0x00007fd73d0eb5af in wait4 () from /lib64/libc.so.6
> #4  0x00007fd741c58908 in ggml_abort () from /run/media/pancho/4C4643C74643B10E/ChatIAs/ik_llama.cpp/lenux/ggml/src/libggml.so
> #5  0x00007fd741dded43 in ggml_cuda_error(char const*, char const*, char const*, int, char const*) () from /run/media/pancho/4C4643C74643B10E/ChatIAs/ik_llama.cpp/lenux/ggml/src/libggml.so
> #6  0x00007fd741decb09 in ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [clone .constprop.1] () from /run/media/pancho/4C4643C74643B10E/ChatIAs/ik_llama.cpp/lenux/ggml/src/libggml.so
> #7  0x00007fd741df42dd in ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) () from /run/media/pancho/4C4643C74643B10E/ChatIAs/ik_llama.cpp/lenux/ggml/src/libggml.so
> #8  0x00007fd741caf9b3 in ggml_backend_sched_graph_compute_async () from /run/media/pancho/4C4643C74643B10E/ChatIAs/ik_llama.cpp/lenux/ggml/src/libggml.so
> #9  0x00007fd79656af1a in llama_decode () from /run/media/pancho/4C4643C74643B10E/ChatIAs/ik_llama.cpp/lenux/src/libllama.so
> #10 0x000000000049a2d4 in server_context::update_slots() ()
> #11 0x000000000046cafc in server_queue::start_loop() ()
> #12 0x0000000000416977 in main ()
> [Inferior 1 (process 23939) detached]
> ```
> 
> Ran it with
> 
> ```
> ./llama-server -m '/models_llm/DeepSeek-V3-0324-UD-Q3_K_XL-00001-of-00007.gguf' -c 32768 --no-mmap -ngl 999 -ot "blk.(0|1|2|3|4|5|6).ffn.=CUDA0" -ot "blk.(7|8|9|10).ffn.=CUDA1" -ot "blk.(11|12|13|14).ffn.=CUDA2" -ot "blk.(15|16|17).ffn.=CUDA3"  -ot "blk.(18|19|20|21|22|23|24|25).ffn.=CUDA4" -ot "ffn.*=CPU" -fa -mg 0 -ub 2048 -mla 1 -fmoe
> ```
> 
> Not using -fmoe makes it work without issues.

As you're using GPU+CPU, please try to replace "-mla 1" with "-mla 2".

---

👤 **ikawrakow** commented the **2025-05-15** at **04:35:23**:<br>

> As you're using GPU+CPU, please try to replace "-mla 1" with "-mla 2".

`-mla 3` work now on CPU+GPU and is the best option. 

Concerning the error, it is not triggered in a function related to `-fmoe`, so I wonder if it is a pre-existing bug (a bunch of those got fixed in mainline lately).

---

👤 **Panchovix** commented the **2025-05-15** at **22:22:06**:<br>

Okay tested again, after updating and rebooting Fedora and now -fmoe works fine with MLA 1 + FA on CUDA+CPU (I use it like to save vram on compute buffers)

Not sure exactly what would have causes the issue.

---

👤 **schynce** commented the **2025-05-15** at **22:32:20**:<br>

> Okay tested again, after updating and rebooting Fedora and now -fmoe works fine with MLA 1 + FA on CUDA+CPU (I use it like to save vram on compute buffers)
> 
> Not sure exactly what would have causes the issue.

Are you sure that it is actually fixed? I am asking because I had some commands that I thought "worked" and started happily using them only for them to crash 15 messages and >30K tokens later. Some would crash instantly or with long prompts.

---

👤 **Panchovix** commented the **2025-05-15** at **22:45:52**:<br>

@schynce you're correct, tried a few more and it got the illegal memory access again.

---

👤 **Panchovix** commented the **2025-05-15** at **22:45:52**:<br>

@schynce you're correct, tried a few more it got the illegal memory access.

---

👤 **divine-taco** commented the **2025-05-19** at **23:10:44**:<br>

Another data point. I'm not entirely sure `-fmoe` is the problem here. This is running multi gpu (3090) with cpu offload.

I can also report that it is rare for the crash to occur immediately. It's usually after a handful of turns.

Note this seems this a recently introduced bug:
`-fmoe -mla 2` does not crash on 6c23618ca5d680bd00f06a143dc4a1b386c827e3
`-fmoe -mla 3` does not crash on 6c23618ca5d680bd00f06a143dc4a1b386c827e3     (much slower than mla 2 on this commit)

It stopped working somewhen after this.
`-fmoe -mla 2` crashes for 2ec2229f2e9847d4e96bd7f163201810c8f8299a
`-fmoe -mla 3` crashes for 2ec2229f2e9847d4e96bd7f163201810c8f8299a

`-mla 2` without fmoe is also crashing for 2ec2229f2e9847d4e96bd7f163201810c8f8299a

If I get some time this week I'll try to isolate when the bug was introduced.
Probably worth someone else trying `6c23618ca5d680bd00f06a143dc4a1b386c827e3` to confirm this is the same issue everyone seems to be running into with multi gpu.

Suspect https://github.com/ikawrakow/ik_llama.cpp/issues/425 may be the same issue.

---

👤 **divine-taco** commented the **2025-05-19** at **23:10:44**:<br>

Another data point. I'm not entirely sure `-fmoe` is the problem here. This is running multi gpu (3090) with cpu offload.

I can also report that it is rare for the crash to occur immediately. It's usually after a handful of turns.

Note this seems this a recently introduced bug:
`-fmoe -mla 2` does not crash on 6c23618ca5d680bd00f06a143dc4a1b386c827e3

It stopped working somewhen after this.
`-fmoe -mla 2` is broken for 2ec2229f2e9847d4e96bd7f163201810c8f8299a

`-mla 2` without fmoe is also broken for 2ec2229f2e9847d4e96bd7f163201810c8f8299a

If I get some time this week I'll try to isolate when the bug was introduced.
Probably worth someone else trying `6c23618ca5d680bd00f06a143dc4a1b386c827e3` to confirm this is the same issue everyone seems to be running into with multi gpu.

---

👤 **ikawrakow** commented the **2025-05-20** at **04:34:00**:<br>

@divine-taco It would be useful to share your command line when reporting a problem.

The most significant change between  https://github.com/ikawrakow/ik_llama.cpp/commit/6c23618ca5d680bd00f06a143dc4a1b386c827e3 and https://github.com/ikawrakow/ik_llama.cpp/commit/2ec2229f2e9847d4e96bd7f163201810c8f8299a is PR #405. Prior to this PR the fused `ffn_up/ffn_gate` operation was not offloaded to the GPU if the tensors were on the CPU. After #405 the op is offloaded. You can disable that and restore the behavior prior to #405 using `-op 29,0`. Can you try that? Thanks.

---

👤 **divine-taco** commented the **2025-05-20** at **05:56:42**:<br>

~~@ikawrakow `-op 29,0` seems to fix the issues running with the latest commit - 2ec2229f2e9847d4e96bd7f163201810c8f8299a~~

Full command:

```
llama-server \
  --parallel 1 \
  -ctk f16 -ctv f16 \
  -ts 17,17,17,17,17,17,17,17,17 \
  --model /home/mx01/DeepSeek-V3-0324-GGUF-Q8_0 --host 0.0.0.0 --port 8080 \
  --ctx-size 44000 \
  -fmoe -rtr -mla 3 -fa \
  -b 2048 -ub 2048 -amb 512 \
  -op 29,0 \
  --no-mmap \
  --threads 64 --threads-batch 64 \
  -ngl 99 \
  -ot exps=CPU
```

Update:

2ec2229f2e9847d4e96bd7f163201810c8f8299a did eventually crash with `-op 29,0` in the same manner as before. It took quite a few turns to observe the behavior (~15).

```
CUDA error: an illegal memory access was encountered
  current device: 0, in function ggml_backend_cuda_synchronize at /app/ggml/src/ggml-cuda.cu:3067
  cudaStreamSynchronize(cuda_ctx->stream())
/app/ggml/src/ggml-cuda.cu:110: CUDA error
```

---

👤 **divine-taco** commented the **2025-05-20** at **05:56:42**:<br>

@ikawrakow `-op 29,0` seems to fix the issues running with the latest commit - 2ec2229f2e9847d4e96bd7f163201810c8f8299a

Full command:

```
llama-server \
  --parallel 1 \
  -ctk f16 -ctv f16 \
  -ts 17,17,17,17,17,17,17,17,17 \
  --model /home/mx01/DeepSeek-V3-0324-GGUF-Q8_0 --host 0.0.0.0 --port 8080 \
  --ctx-size 44000 \
  -fmoe -rtr -mla 3 -fa \
  -b 2048 -ub 2048 -amb 512 \
  -op 29,0 \
  --no-mmap \
  --threads 64 --threads-batch 64 \
  -ngl 99 \
  -ot exps=CPU
```

---

👤 **schynce** commented the **2025-05-20** at **13:44:34**:<br>

For me, the best way to trigger the bug quickly is to dump in a 30K token prompt. It seems to crash during the prompt processing or before generating a single token.

---

👤 **schynce** commented the **2025-05-20** at **13:44:34**:<br>

For me, the best way to trigger the bug quickly is to dump in a 30K token prompt. It seems to crash during the prompt processing.

---

👤 **ikawrakow** commented the **2025-05-20** at **14:23:18**:<br>

Does PR #438 help?

---

👤 **schynce** commented the **2025-05-20** at **15:58:47**:<br>

> Does PR [#438](https://github.com/ikawrakow/ik_llama.cpp/pull/438) help?

I tested #438 (branch ik/desperate_bug_fix_attempt) but unfortunately, it crashed almost straight away:

```
./llama-server --model /mnt/Qwen3-235B-A22B-IQ4_XS-00001-of-00003.gguf --alias Qwen3-235B-A22B-IQ4_XS \
-fa -rtr -c 40960 -ctk q8_0 -ctv q8_0 --threads 7 --no-kv-offload \
-ot "blk\.\d+\.attn=CUDA2" \
-ot "blk\.(0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17)\.=CUDA0" \
-ot "blk\.(18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35)\.=CUDA1" \
-ot "blk\.(36|37|38|39|40|41|42|43|44|45|46|47|48|49|50|51)\.=CUDA2"
```

```
INFO [            update_slots] kv cache rm [p0, end) | tid="139707044622336" timestamp=1747756441 id_slot=0 id_task=27 p0=4097
CUDA error: an illegal memory access was encountered
  current device: 2, in function ggml_backend_cuda_synchronize at /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu:3075
  cudaStreamSynchronize(cuda_ctx->stream())
/home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu:110: CUDA error
Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
ptrace: Operation not permitted.
No stack.
The program is not being run.
Aborted (core dumped)
```

---

👤 **divine-taco** commented the **2025-05-20** at **21:36:55**:<br>

~~PR #438 - 82871cc2a3366dfdeff758f04fdfcf5ae5859829 - looks to fix the issue for me. Tried 30 turn completions at long context and saw no issues.~~

Command used:
```
llama-server \
  --parallel 1 \
  -ctk f16 -ctv f16 \
  -ts 17,17,17,17,17,17,17,17,17 \
  --model /home/mx01/DeepSeek-V3-0324-GGUF-Q8_0 --host 0.0.0.0 --port 8080 \
  --ctx-size 44000 \
  -fmoe -rtr -mla 3 -fa \
  -b 2048 -ub 2048 -amb 512 \
  --no-mmap \
  --threads 64 --threads-batch 64 \
  -ngl 99 \
  -ot exps=CPU
```

@schynce - Have a link to the Qwen3-235B-A22B quant you used? I can try that as well.

Update: Failed with illegal memory access again on PR #438 with deepseek 0324 after I ran some automated completions tests. I don't have enough data yet to be confident, but it does seem to fail less frequently. I'll try running `--mla 2` on PR #438 to see if this makes any difference.

---

👤 **divine-taco** commented the **2025-05-20** at **21:36:55**:<br>

PR #438 - 82871cc2a3366dfdeff758f04fdfcf5ae5859829 - looks to fix the issue for me. Tried 30 turn completions at long context and saw no issues.

Command used:
```
llama-server \
  --parallel 1 \
  -ctk f16 -ctv f16 \
  -ts 17,17,17,17,17,17,17,17,17 \
  --model /home/mx01/DeepSeek-V3-0324-GGUF-Q8_0 --host 0.0.0.0 --port 8080 \
  --ctx-size 44000 \
  -fmoe -rtr -mla 3 -fa \
  -b 2048 -ub 2048 -amb 512 \
  --no-mmap \
  --threads 64 --threads-batch 64 \
  -ngl 99 \
  -ot exps=CPU
```

---

👤 **schynce** commented the **2025-05-20** at **21:49:54**:<br>

@divine-taco 

I used this:

https://huggingface.co/unsloth/Qwen3-235B-A22B-GGUF/tree/main/IQ4_XS

However, I notice that there have been some updates in the first split file since I downloaded it.

---

👤 **ikawrakow** commented the **2025-05-21** at **06:02:41**:<br>

Please use branch in PR #442 and post the CUDA call trace that will be printed when the application crashes.

---

👤 **schynce** commented the **2025-05-21** at **12:11:08**:<br>

> Please use branch in PR [#442](https://github.com/ikawrakow/ik_llama.cpp/pull/442) and post the CUDA call trace that will be printed when the application crashes.

```
llm_load_tensors:  CUDA_Host buffer size = 52313.37 MiB
llm_load_tensors:      CUDA0 buffer size = 22068.28 MiB
llm_load_tensors:      CUDA1 buffer size = 22068.28 MiB
llm_load_tensors:      CUDA2 buffer size = 23042.94 MiB
....................................................................................................
============ Repacked 127 tensors
llama_new_context_with_model: n_ctx      = 40960
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:  CUDA_Host KV buffer size =  3995.00 MiB
llama_new_context_with_model: KV self size  = 3995.00 MiB, K (q8_0): 1997.50 MiB, V (q8_0): 1997.50 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     1.16 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =   104.50 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =   104.50 MiB
llama_new_context_with_model:      CUDA2 compute buffer size =   189.25 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   304.75 MiB
llama_new_context_with_model: graph nodes  = 3672
llama_new_context_with_model: graph splits = 432
INFO [                    init] initializing slots | tid="140363884277760" timestamp=1747829175 n_slots=1
INFO [                    init] new slot | tid="140363884277760" timestamp=1747829175 id_slot=0 n_ctx_slot=40960
INFO [                    main] model loaded | tid="140363884277760" timestamp=1747829175
INFO [                    main] chat template | tid="140363884277760" timestamp=1747829175 chat_example="<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there<|im_end|>\n<|im_start|>user\nHow are you?<|im_end|>\n<|im_start|>assistant\n" built_in=true
INFO [                    main] HTTP server listening | tid="140363884277760" timestamp=1747829175 n_threads_http="15" port="5000" hostname="0.0.0.0"
INFO [            update_slots] all slots are idle | tid="140363884277760" timestamp=1747829175
INFO [      log_server_request] request | tid="140361486192640" timestamp=1747829175 remote_addr="127.0.0.1" remote_port=55754 status=200 method="GET" path="/v1/models" params={}
INFO [      log_server_request] request | tid="140361494585344" timestamp=1747829175 remote_addr="127.0.0.1" remote_port=57094 status=200 method="GET" path="/v1/models" params={}
INFO [      log_server_request] request | tid="140361477799936" timestamp=1747829182 remote_addr="127.0.0.1" remote_port=43408 status=200 method="GET" path="/v1/models" params={}
INFO [      log_server_request] request | tid="140361469407232" timestamp=1747829191 remote_addr="127.0.0.1" remote_port=49880 status=200 method="GET" path="/v1/models" params={}
INFO [   launch_slot_with_task] slot is processing task | tid="140363884277760" timestamp=1747829191 id_slot=0 id_task=0
INFO [            update_slots] kv cache rm [p0, end) | tid="140363884277760" timestamp=1747829191 id_slot=0 id_task=0 p0=0
CUDA error: an illegal memory access was encountered
  current device: 2, in function ggml_backend_cuda_synchronize at /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu:3085
  cudaStreamSynchronize(cuda_ctx->stream())
========================== CUDA trace: 315944 previous calls
      315943: function ggml_cuda_op_mul_mat, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1755
      315942: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      315941: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      315940: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      315939: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      315938: function ggml_cuda_op_mul_mat, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1632
      315937: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      315936: function ggml_cuda_op_mul_mat, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1755
      315935: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      315934: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      315933: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      315932: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      315931: function ggml_cuda_op_mul_mat, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1632
      315930: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      315929: function ggml_cuda_op_mul_mat, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1755
      315928: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      315927: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      315926: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      315925: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      315924: function ggml_cuda_op_mul_mat, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1632
      315923: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      315922: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 135
      315921: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      315920: function ggml_backend_cuda_cpy_tensor_async, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 3074
      315919: function ggml_backend_cuda_cpy_tensor_async, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 3071
      315918: function ggml_backend_cuda_cpy_tensor_async, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 3061
      315917: function ggml_backend_cuda_synchronize, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 3085
      315916: function ggml_cuda_up_gate_unary, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2773
      315915: function ggml_cuda_up_gate_unary, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2764
      315914: function ggml_cuda_op_mul_mat, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1755
      315913: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      315912: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      315911: function ggml_cuda_op_mul_mat, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1755
/home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu:122: CUDA error
Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
ptrace: Operation not permitted.
No stack.
The program is not being run.
```

---

👤 **ikawrakow** commented the **2025-05-21** at **12:37:17**:<br>

Thank you!

So, it crashes in a matrix multiplication. I have pushed another commit on the branch that will help narrow it down further if you rerun with that.

---

👤 **schynce** commented the **2025-05-21** at **13:29:25**:<br>

> Thank you!
> 
> So, it crashes in a matrix multiplication. I have pushed another commit on the branch that will help narrow it down further if you rerun with that.

Thanks for looking into the issue! Here you go:

```
CUDA error: an illegal memory access was encountered
  current device: 2, in function ggml_backend_cuda_synchronize at /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu:3085
  cudaStreamSynchronize(cuda_ctx->stream())
========================== CUDA trace: 335439 previous calls
      335438: function ggml_cuda_op_mul_mat, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1755
      335437: function launch_mul_mat_q, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda/template-instances/../mmq.cuh, line 3529
      335436: function launch_mul_mat_q, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda/template-instances/../mmq.cuh, line 3525
      335435: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      335434: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      335433: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      335432: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      335431: function ggml_cuda_op_mul_mat, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1632
      335430: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      335429: function ggml_cuda_op_mul_mat, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1755
      335428: function launch_mul_mat_q, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda/template-instances/../mmq.cuh, line 3529
      335427: function launch_mul_mat_q, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda/template-instances/../mmq.cuh, line 3525
      335426: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      335425: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      335424: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      335423: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      335422: function ggml_cuda_op_mul_mat, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1632
      335421: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      335420: function ggml_cuda_op_mul_mat, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1755
      335419: function launch_mul_mat_q, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda/template-instances/../mmq.cuh, line 3529
      335418: function launch_mul_mat_q, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda/template-instances/../mmq.cuh, line 3525
      335417: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      335416: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      335415: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      335414: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      335413: function ggml_cuda_op_mul_mat, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1632
      335412: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      335411: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 135
      335410: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      335409: function ggml_backend_cuda_cpy_tensor_async, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 3074
      335408: function ggml_backend_cuda_cpy_tensor_async, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 3071
      335407: function ggml_backend_cuda_cpy_tensor_async, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 3061
      335406: function ggml_cuda_op_mul_mat, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1755
/home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu:122: CUDA error
```

---

👤 **ikawrakow** commented the **2025-05-21** at **13:55:41**:<br>

I was confused. If there was something wrong with the matrix multiplications, it would have aborted there. The computations succeed, but then something goes wrong in the back-end. I have now added 2 additional asserts in the back-end at the place where the back-trace was when we did the debugging session.

---

👤 **schynce** commented the **2025-05-21** at **14:10:05**:<br>

> I was confused. If there was something wrong with the matrix multiplications, it would have aborted there. The computations succeed, but then something goes wrong in the back-end. I have now added 2 additional asserts in the back-end at the place where the back-trace was when we did the debugging session.

I tried the newest commit, but the backtrace is practically identical as far as I can tell:

```
CUDA error: an illegal memory access was encountered
  current device: 2, in function ggml_backend_cuda_synchronize at /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu:3089
  cudaStreamSynchronize(stream)
========================== CUDA trace: 335439 previous calls
      335438: function ggml_cuda_op_mul_mat, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1755
      335437: function launch_mul_mat_q, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda/template-instances/../mmq.cuh, line 3529
      335436: function launch_mul_mat_q, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda/template-instances/../mmq.cuh, line 3525
      335435: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      335434: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      335433: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      335432: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      335431: function ggml_cuda_op_mul_mat, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1632
      335430: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      335429: function ggml_cuda_op_mul_mat, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1755
      335428: function launch_mul_mat_q, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda/template-instances/../mmq.cuh, line 3529
      335427: function launch_mul_mat_q, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda/template-instances/../mmq.cuh, line 3525
      335426: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      335425: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      335424: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      335423: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      335422: function ggml_cuda_op_mul_mat, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1632
      335421: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      335420: function ggml_cuda_op_mul_mat, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1755
      335419: function launch_mul_mat_q, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda/template-instances/../mmq.cuh, line 3529
      335418: function launch_mul_mat_q, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda/template-instances/../mmq.cuh, line 3525
      335417: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      335416: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      335415: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      335414: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      335413: function ggml_cuda_op_mul_mat, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1632
      335412: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      335411: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 135
      335410: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      335409: function ggml_backend_cuda_cpy_tensor_async, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 3074
      335408: function ggml_backend_cuda_cpy_tensor_async, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 3071
      335407: function ggml_backend_cuda_cpy_tensor_async, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 3061
      335406: function ggml_cuda_op_mul_mat, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1755
/home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu:122: CUDA error
```

---

👤 **ikawrakow** commented the **2025-05-21** at **14:27:12**:<br>

Thanks! I'll keep digging.

---

👤 **ikawrakow** commented the **2025-05-21** at **15:26:00**:<br>

I have now added a trace to the back-end, so when the crash occurs it will print from where `ggml_backend_cuda_synchronize` was called. Can you try another time? Thanks!

---

👤 **schynce** commented the **2025-05-21** at **16:31:48**:<br>

> I have now added a trace to the back-end, so when the crash occurs it will print from where `ggml_backend_cuda_synchronize` was called. Can you try another time? Thanks!

```
CUDA error: an illegal memory access was encountered
  current device: 2, in function ggml_backend_sched_compute_splits at /home/user/ik_llama.cpp/ggml/src/ggml-backend.c:1835
  cudaStreamSynchronize
========================== CUDA trace: 335439 previous calls
      335438: function ggml_cuda_op_mul_mat, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1755
      335437: function launch_mul_mat_q, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda/template-instances/../mmq.cuh, line 3529
      335436: function launch_mul_mat_q, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda/template-instances/../mmq.cuh, line 3525
      335435: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      335434: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      335433: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      335432: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      335431: function ggml_cuda_op_mul_mat, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1632
      335430: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      335429: function ggml_cuda_op_mul_mat, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1755
      335428: function launch_mul_mat_q, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda/template-instances/../mmq.cuh, line 3529
      335427: function launch_mul_mat_q, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda/template-instances/../mmq.cuh, line 3525
      335426: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      335425: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      335424: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      335423: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      335422: function ggml_cuda_op_mul_mat, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1632
      335421: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      335420: function ggml_cuda_op_mul_mat, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1755
      335419: function launch_mul_mat_q, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda/template-instances/../mmq.cuh, line 3529
      335418: function launch_mul_mat_q, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda/template-instances/../mmq.cuh, line 3525
      335417: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      335416: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      335415: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      335414: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      335413: function ggml_cuda_op_mul_mat, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1632
      335412: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      335411: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 135
      335410: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      335409: function ggml_backend_cuda_cpy_tensor_async, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 3074
      335408: function ggml_backend_cuda_cpy_tensor_async, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 3071
      335407: function ggml_backend_cuda_cpy_tensor_async, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 3061
      335406: function ggml_cuda_op_mul_mat, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1755
/home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu:122: CUDA error
```

---

👤 **ikawrakow** commented the **2025-05-21** at **16:43:24**:<br>

@schynce You are running with `--no-kv-offload`, right? Your error is different. What happens if you don't use `--no-kv-offload`?

---

👤 **schynce** commented the **2025-05-21** at **16:55:42**:<br>

> [@schynce](https://github.com/schynce) You are running with `--no-kv-offload`, right? Your error is different. What happens if you don't use `--no-kv-offload`?

Yes, those logs were with this launch command:

```
./llama-server --model /mnt/Qwen3-235B-A22B-IQ4_XS-00001-of-00003.gguf --alias Qwen3-235B-A22B-IQ4_XS \
-fa -fmoe -rtr -c 40960 -ctk q8_0 -ctv q8_0 --threads 7 --no-kv-offload \
-ot "blk\.\d+\.attn=CUDA2" \
-ot "blk\.(0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17)\.=CUDA0" \
-ot "blk\.(18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35)\.=CUDA1" \
-ot "blk\.(36|37|38|39|40|41|42|43|44|45|46|47|48|49|50|51)\.=CUDA2"

```
---

I ran without --no-kv-offload and modified the layers to fit the KV cache:

```
./llama-server --model /mnt/Qwen3-235B-A22B-IQ4_XS-00001-of-00003.gguf --alias Qwen3-235B-A22B-IQ4_XS \
-fa -fmoe -rtr -c 40960 -ctk q8_0 -ctv q8_0 --threads 7 \
-ot "blk\.\d+\.attn=CUDA2" \
-ot "blk\.(0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16)\.=CUDA0" \
-ot "blk\.(17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33)\.=CUDA1" \
-ot "blk\.(34|35|36|37|38|39|40|41|42|43|44|45|46|47)\.=CUDA2"
```

It took considerably longer for the crash to appear this time:

```
INFO [   launch_slot_with_task] slot is processing task | tid="139770035781632" timestamp=1747846205 id_slot=0 id_task=0
INFO [            update_slots] kv cache rm [p0, end) | tid="139770035781632" timestamp=1747846205 id_slot=0 id_task=0 p0=0
INFO [            update_slots] kv cache rm [p0, end) | tid="139770035781632" timestamp=1747846249 id_slot=0 id_task=0 p0=2048
INFO [            update_slots] kv cache rm [p0, end) | tid="139770035781632" timestamp=1747846293 id_slot=0 id_task=0 p0=4096
INFO [            update_slots] kv cache rm [p0, end) | tid="139770035781632" timestamp=1747846338 id_slot=0 id_task=0 p0=6144
CUDA error: an illegal memory access was encountered
  current device: 2, in function ggml_backend_sched_compute_splits at /home/user/ik_llama.cpp/ggml/src/ggml-backend.c:1835
  cudaStreamSynchronize
========================== CUDA trace: 2460820 previous calls
     2460819: function ggml_cuda_op_mul_mat, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1755
     2460818: function launch_mul_mat_q, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda/template-instances/../mmq.cuh, line 3529
     2460817: function launch_mul_mat_q, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda/template-instances/../mmq.cuh, line 3525
     2460816: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
     2460815: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
     2460814: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
     2460813: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
     2460812: function ggml_cuda_op_mul_mat, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1632
     2460811: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
     2460810: function ggml_cuda_op_mul_mat, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1755
     2460809: function launch_mul_mat_q, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda/template-instances/../mmq.cuh, line 3529
     2460808: function launch_mul_mat_q, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda/template-instances/../mmq.cuh, line 3525
     2460807: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
     2460806: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
     2460805: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
     2460804: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
     2460803: function ggml_cuda_op_mul_mat, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1632
     2460802: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
     2460801: function ggml_cuda_op_mul_mat, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1755
     2460800: function launch_mul_mat_q, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda/template-instances/../mmq.cuh, line 3529
     2460799: function launch_mul_mat_q, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda/template-instances/../mmq.cuh, line 3525
     2460798: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
     2460797: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
     2460796: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
     2460795: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
     2460794: function ggml_cuda_op_mul_mat, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1632
     2460793: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
     2460792: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 135
     2460791: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
     2460790: function ggml_backend_cuda_cpy_tensor_async, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 3074
     2460789: function ggml_backend_cuda_cpy_tensor_async, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 3071
     2460788: function ggml_backend_cuda_cpy_tensor_async, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 3061
     2460787: function ggml_cuda_op_mul_mat, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1755
/home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu:122: CUDA error
```

---

👤 **ikawrakow** commented the **2025-05-22** at **06:44:46**:<br>

If you are not tired of testing, there are new changes on #442

---

👤 **schynce** commented the **2025-05-22** at **07:43:25**:<br>

> If you are not tired of testing, there are new changes on [#442](https://github.com/ikawrakow/ik_llama.cpp/pull/442)

Not even close to being tired yet, thank you for taking the time to look into this :)

I ran this command:
```

./llama-server --model /mnt/Qwen3-235B-A22B-IQ4_XS-00001-of-00003.gguf --alias Qwen3-235B-A22B-IQ4_XS \
-fa -fmoe -rtr -c 40960 -ctk q8_0 -ctv q8_0 --threads 7 \
-ot "blk\.\d+\.attn=CUDA2" \
-ot "blk\.(0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16)\.=CUDA0" \
-ot "blk\.(17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33)\.=CUDA1" \
-ot "blk\.(34|35|36|37|38|39|40|41|42|43|44|45|46|47)\.=CUDA2"
```

During context processing, the console was getting spammed with the `ggml_backend_cuda_synchronize` and `ggml_backend_cuda_cpy_tensor_async` lines. At the end of prompt processing (I assume), it crashed like before:

```
ggml_backend_cuda_cpy_tensor_async: attempt to copy from device 0 to device 2 without access enabled
ggml_backend_cuda_synchronize: curent device is 0, context device is 2
ggml_backend_cuda_synchronize: reverting device to 0
ggml_backend_cuda_synchronize: curent device is 0, context device is 2
ggml_backend_cuda_synchronize: reverting device to 0
ggml_backend_cuda_cpy_tensor_async: attempt to copy from device 0 to device 2 without access enabled
ggml_backend_cuda_synchronize: curent device is 0, context device is 2
ggml_backend_cuda_synchronize: reverting device to 0
ggml_backend_cuda_synchronize: curent device is 0, context device is 2
ggml_backend_cuda_synchronize: reverting device to 0
ggml_backend_cuda_synchronize: curent device is 0, context device is 2
ggml_backend_cuda_synchronize: reverting device to 0
ggml_backend_cuda_synchronize: curent device is 2, context device is 0
ggml_backend_cuda_synchronize: reverting device to 2
ggml_backend_cuda_cpy_tensor_async: attempt to copy from device 2 to device 0 without access enabled
ggml_backend_cuda_synchronize: curent device is 2, context device is 0
ggml_backend_cuda_synchronize: reverting device to 2
ggml_backend_cuda_synchronize: curent device is 2, context device is 0
ggml_backend_cuda_synchronize: reverting device to 2
ggml_backend_cuda_synchronize: curent device is 2, context device is 0
ggml_backend_cuda_synchronize: reverting device to 2
ggml_backend_cuda_synchronize: curent device is 0, context device is 2
ggml_backend_cuda_synchronize: reverting device to 0
ggml_backend_cuda_cpy_tensor_async: attempt to copy from device 0 to device 2 without access enabled
ggml_backend_cuda_synchronize: curent device is 0, context device is 2
ggml_backend_cuda_synchronize: reverting device to 0
ggml_backend_cuda_synchronize: curent device is 2, context device is 0
ggml_backend_cuda_synchronize: reverting device to 2
ggml_backend_cuda_cpy_tensor_async: attempt to copy from device 2 to device 0 without access enabled
ggml_backend_cuda_synchronize: curent device is 2, context device is 0
ggml_backend_cuda_synchronize: reverting device to 2
ggml_backend_cuda_synchronize: curent device is 0, context device is 2
ggml_backend_cuda_synchronize: reverting device to 0
ggml_backend_cuda_cpy_tensor_async: attempt to copy from device 0 to device 2 without access enabled
ggml_backend_cuda_synchronize: curent device is 0, context device is 2
ggml_backend_cuda_synchronize: reverting device to 0
ggml_backend_cuda_synchronize: curent device is 2, context device is 0
ggml_backend_cuda_synchronize: reverting device to 2
ggml_backend_cuda_cpy_tensor_async: attempt to copy from device 2 to device 0 without access enabled
ggml_backend_cuda_synchronize: curent device is 2, context device is 0
ggml_backend_cuda_synchronize: reverting device to 2
ggml_backend_cuda_synchronize: curent device is 0, context device is 2
ggml_backend_cuda_synchronize: reverting device to 0
ggml_backend_cuda_cpy_tensor_async: attempt to copy from device 0 to device 2 without access enabled
ggml_backend_cuda_synchronize: curent device is 0, context device is 2
ggml_backend_cuda_synchronize: reverting device to 0
ggml_backend_cuda_synchronize: curent device is 2, context device is 0
ggml_backend_cuda_synchronize: reverting device to 2
ggml_backend_cuda_cpy_tensor_async: attempt to copy from device 2 to device 0 without access enabled
ggml_backend_cuda_synchronize: curent device is 2, context device is 0
ggml_backend_cuda_synchronize: reverting device to 2
ggml_backend_cuda_synchronize: curent device is 0, context device is 2
ggml_backend_cuda_synchronize: reverting device to 0
ggml_backend_cuda_cpy_tensor_async: attempt to copy from device 0 to device 2 without access enabled
ggml_backend_cuda_synchronize: curent device is 0, context device is 2
ggml_backend_cuda_synchronize: reverting device to 0
ggml_backend_cuda_synchronize: curent device is 2, context device is 0
ggml_backend_cuda_synchronize: reverting device to 2
ggml_backend_cuda_cpy_tensor_async: attempt to copy from device 2 to device 0 without access enabled
ggml_backend_cuda_synchronize: curent device is 2, context device is 0
ggml_backend_cuda_synchronize: reverting device to 2
ggml_backend_cuda_synchronize: curent device is 0, context device is 2
ggml_backend_cuda_synchronize: reverting device to 0
ggml_backend_cuda_cpy_tensor_async: attempt to copy from device 0 to device 2 without access enabled
CUDA error: an illegal memory access was encountered
  current device: 0, in function ggml_backend_sched_compute_splits at /home/user/ik_llama.cpp/ggml/src/ggml-backend.c:1835
  cudaStreamSynchronize
========================== CUDA trace: 2486495 previous calls
     2486494: function ggml_backend_cuda_cpy_tensor_async, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 3070
     2486493: function ggml_backend_cuda_cpy_tensor_async, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 3055
     2486492: function ggml_backend_cuda_synchronize, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 3120
     2486491: function ggml_backend_sched_compute_splits, file /home/user/ik_llama.cpp/ggml/src/ggml-backend.c, line 1828
     2486490: function ggml_backend_cuda_synchronize, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 3107
     2486489: function ggml_cuda_up_gate_unary, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2774
     2486488: function ggml_cuda_up_gate_unary, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2765
     2486487: function ggml_cuda_op_mul_mat, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1756
     2486486: function ggml_cuda_op_mul_mat_vec_q, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda/mmvq.cu, line 593
     2486485: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
     2486484: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
     2486483: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
     2486482: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
     2486481: function ggml_cuda_op_mul_mat, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1632
     2486480: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
     2486479: function ggml_cuda_up_gate_unary, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2744
     2486478: function ggml_cuda_up_gate_unary, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2740
     2486477: function ggml_cuda_op_mul_mat, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1756
     2486476: function ggml_cuda_op_mul_mat_vec_q, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda/mmvq.cu, line 593
     2486475: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
     2486474: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
     2486473: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
     2486472: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
     2486471: function ggml_cuda_op_mul_mat, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1632
     2486470: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
     2486469: function ggml_cuda_up_gate_unary, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2736
     2486468: function ggml_cuda_op_mul_mat, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1756
     2486467: function ggml_cuda_op_mul_mat_vec_q, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda/mmvq.cu, line 593
     2486466: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
     2486465: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
     2486464: function ggml_cuda_get_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
     2486463: function ggml_cuda_set_device, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
     2486462: function ggml_backend_cuda_cpy_tensor_async, file /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 3070
/home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu:122: CUDA error
```