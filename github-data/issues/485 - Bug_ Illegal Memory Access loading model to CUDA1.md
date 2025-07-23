### 🐛 [#485](https://github.com/ikawrakow/ik_llama.cpp/issues/485) - Bug: Illegal Memory Access loading model to CUDA1

| **Author** | `cmoncure` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-02 |
| **Updated** | 2025-06-02 |

---

#### Description

### What happened?

I have two identical GPUs (Rtx 6000 Ada Generation 48 GB VRAM).  I have a llama-server commandline that works with device CUDA0, but fails with device CUDA1.  I have successfully tested device CUDA1 with:

- mainline llama.cpp
- oobabooga text-generation-webui

My script to run `llama-server` is as follows:
```
GPU1=CUDA0

args=(
  -mla 3
  -fa
  -ctk q8_0
  -ctv q8_0
  --ctx-size 131072
  -fmoe
  -amb 512
  -b 1024
  -ub 1024
  -sm none
  --numa isolate
  --threads 16
  --threads-batch 32
  --n-gpu-layers 99
  --override-tensor exps=CPU
  --override-tensor attn=$GPU1
  --override-tensor exp=$GPU1
  --override-tensor blk.*.ffn_gate_inp.weight=$GPU1
  --override-tensor blk.*.ffn_down.weight=$GPU1
  --override-tensor blk.*.ffn_gate.weight=$GPU1
  --override-tensor blk.*.ffn_norm.weight=$GPU1
  --override-tensor blk.*.ffn_up_shexp.weight=$GPU1
  --override-tensor blk.*.ffn_down_shexp.weight=$GPU1
  --override-tensor blk.*.ffn_gate_shexp.weight=$GPU1
  --override-tensor blk.*.ffn_gate_inp.weight=$GPU1
  --host 0.0.0.0
  --port 7862
  --alias DeepSeek/Deepseek-V3-0324
  -m "$model"
)

~/ik_llama.cpp/build/bin/llama-server "${args[@]}"
```

This runs with GPU1=CUDA0, but fails with GPU1 set to the identical CUDA1.

```
[ 5022.696822] Cannot map memory with base addr 0x7d523e000000 and size of 0x8700c pages
[ 5022.899731] NVRM: Xid (PCI:0000:07:00): 31, pid=16952, name=llama-server, Ch 00000008, intr 00000000. MMU Fault: ENGINE GRAPHICS GPC1 GPCCLIENT_T1_0 faulted @ 0x7d58_a0000000. Fault is of type FAULT_PDE ACCESS_TYPE_VIRT_READ
[ 5022.930157] llama-server[16980]: segfault at 20d803fdc ip 00007dbe270a3e47 sp 00007ffff184bf00 error 4 in libcuda.so.570.133.20[4a3e47,7dbe26d79000+d1c000] likely on CPU 29 (core 14, socket 0)
[ 5022.930169] Code: ef e8 2d 55 cd ff 83 3d ae f2 f6 03 01 49 8b 1c 24 76 0a 8b 05 b6 f2 f6 03 85 c0 74 56 49 8b 44 24 10 41 8b 4c 24 24 48 8b 13 <8b> 00 41 39 c6 74 52 8b b3 40 40 00 00 48 89 f0 89 8c b3 44 40 00
```

That base address don't look right.

### Name and Version

$./llama-cli --version
version: 3722 (7a8abe29)
built with cc (Ubuntu 14.2.0-4ubuntu2) 14.2.0 for x86_64-linux-gnu

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
$ ./run_deepseek_ik
Selected model: /home/corey/AIModels/textgen/DeepSeek-V3-0324-Q4_K_M-V2.gguf
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA RTX 6000 Ada Generation, compute capability 8.9, VMM: yes
  Device 1: NVIDIA RTX 6000 Ada Generation, compute capability 8.9, VMM: yes
INFO [                    main] build info | tid="132058923773952" timestamp=1748889508 build=3722 commit="7a8abe29"
INFO [                    main] system info | tid="132058923773952" timestamp=1748889508 n_threads=16 n_threads_batch=32 total_threads=32 system_info="AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: loaded meta data with 53 key-value pairs and 1025 tensors from /home/corey/AIModels/textgen/DeepSeek-V3-0324-Q4_K_M-V2.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = DeepSeek V3 0324
llama_model_loader: - kv   3:                            general.version str              = V3-0324
llama_model_loader: - kv   4:                           general.basename str              = DeepSeek
llama_model_loader: - kv   5:                         general.size_label str              = 256x20B
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
llama_model_loader: - kv  16:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  17:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  18:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  19:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  20:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  21:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  22:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  23:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  24:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  25:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  26:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  27:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  28:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  29:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  30:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  31: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  32: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  33:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  34:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  35:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<�...
llama_model_loader: - kv  36:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  37:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
llama_model_loader: - kv  38:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  39:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  40:            tokenizer.ggml.padding_token_id u32              = 1
llama_model_loader: - kv  41:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  42:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  43:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  44:               general.quantization_version u32              = 2
llama_model_loader: - kv  45:                          general.file_type u32              = 15
llama_model_loader: - kv  46:                      quantize.imatrix.file str              = /models/DeepSeek-V3-0324-GGUF/DeepSee...
llama_model_loader: - kv  47:                   quantize.imatrix.dataset str              = /workspace/calibration_datav3.txt
llama_model_loader: - kv  48:             quantize.imatrix.entries_count i32              = 720
llama_model_loader: - kv  49:              quantize.imatrix.chunks_count i32              = 124
llama_model_loader: - kv  50:                                   split.no u16              = 0
llama_model_loader: - kv  51:                        split.tensors.count i32              = 1025
llama_model_loader: - kv  52:                                split.count u16              = 0
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  151 tensors
llama_model_loader: - type q4_K:  154 tensors
llama_model_loader: - type q5_K:  153 tensors
llama_model_loader: - type q6_K:  206 tensors
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
llm_load_print_meta: model ftype      = Q4_K - Medium
llm_load_print_meta: model params     = 671.026 B
llm_load_print_meta: model size       = 379.030 GiB (4.852 BPW) 
llm_load_print_meta: repeating layers = 377.836 GiB (4.850 BPW, 669.173 B parameters)
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
llm_load_tensors: ggml ctx size =    0.85 MiB
Tensor blk.0.attn_norm.weight buffer type overriden to CUDA1
Tensor blk.0.attn_q_a_norm.weight buffer type overriden to CUDA1
Tensor blk.0.attn_kv_a_norm.weight buffer type overriden to CUDA1
Tensor blk.0.attn_q_a.weight buffer type overriden to CUDA1
Tensor blk.0.attn_q_b.weight buffer type overriden to CUDA1
Tensor blk.0.attn_kv_a_mqa.weight buffer type overriden to CUDA1
Tensor blk.0.attn_kv_b.weight buffer type overriden to CUDA1
Tensor blk.0.attn_k_b.weight buffer type overriden to CUDA1
Tensor blk.0.attn_v_b.weight buffer type overriden to CUDA1
Tensor blk.0.attn_output.weight buffer type overriden to CUDA1
Tensor blk.0.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.0.ffn_gate.weight buffer type overriden to CUDA1
Tensor blk.0.ffn_down.weight buffer type overriden to CUDA1
Tensor blk.1.attn_norm.weight buffer type overriden to CUDA1
Tensor blk.1.attn_q_a_norm.weight buffer type overriden to CUDA1
Tensor blk.1.attn_kv_a_norm.weight buffer type overriden to CUDA1
Tensor blk.1.attn_q_a.weight buffer type overriden to CUDA1
Tensor blk.1.attn_q_b.weight buffer type overriden to CUDA1
Tensor blk.1.attn_kv_a_mqa.weight buffer type overriden to CUDA1
Tensor blk.1.attn_kv_b.weight buffer type overriden to CUDA1
Tensor blk.1.attn_k_b.weight buffer type overriden to CUDA1
Tensor blk.1.attn_v_b.weight buffer type overriden to CUDA1
Tensor blk.1.attn_output.weight buffer type overriden to CUDA1
Tensor blk.1.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.1.ffn_gate.weight buffer type overriden to CUDA1
Tensor blk.1.ffn_down.weight buffer type overriden to CUDA1
Tensor blk.2.attn_norm.weight buffer type overriden to CUDA1
Tensor blk.2.attn_q_a_norm.weight buffer type overriden to CUDA1
Tensor blk.2.attn_kv_a_norm.weight buffer type overriden to CUDA1
Tensor blk.2.attn_q_a.weight buffer type overriden to CUDA1
Tensor blk.2.attn_q_b.weight buffer type overriden to CUDA1
Tensor blk.2.attn_kv_a_mqa.weight buffer type overriden to CUDA1
Tensor blk.2.attn_kv_b.weight buffer type overriden to CUDA1
Tensor blk.2.attn_k_b.weight buffer type overriden to CUDA1
Tensor blk.2.attn_v_b.weight buffer type overriden to CUDA1
Tensor blk.2.attn_output.weight buffer type overriden to CUDA1
Tensor blk.2.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.2.ffn_gate.weight buffer type overriden to CUDA1
Tensor blk.2.ffn_down.weight buffer type overriden to CUDA1
Tensor blk.3.attn_norm.weight buffer type overriden to CUDA1
Tensor blk.3.attn_q_a_norm.weight buffer type overriden to CUDA1
Tensor blk.3.attn_kv_a_norm.weight buffer type overriden to CUDA1
Tensor blk.3.attn_q_a.weight buffer type overriden to CUDA1
Tensor blk.3.attn_q_b.weight buffer type overriden to CUDA1
Tensor blk.3.attn_kv_a_mqa.weight buffer type overriden to CUDA1
Tensor blk.3.attn_kv_b.weight buffer type overriden to CUDA1
Tensor blk.3.attn_k_b.weight buffer type overriden to CUDA1
Tensor blk.3.attn_v_b.weight buffer type overriden to CUDA1
Tensor blk.3.attn_output.weight buffer type overriden to CUDA1
Tensor blk.3.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.3.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.3.exp_probs_b.bias buffer type overriden to CUDA1
Tensor blk.3.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.3.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.3.ffn_up_shexp.weight buffer type overriden to CUDA1
Tensor blk.4.attn_norm.weight buffer type overriden to CUDA1
Tensor blk.4.attn_q_a_norm.weight buffer type overriden to CUDA1
Tensor blk.4.attn_kv_a_norm.weight buffer type overriden to CUDA1
Tensor blk.4.attn_q_a.weight buffer type overriden to CUDA1
Tensor blk.4.attn_q_b.weight buffer type overriden to CUDA1
Tensor blk.4.attn_kv_a_mqa.weight buffer type overriden to CUDA1
Tensor blk.4.attn_kv_b.weight buffer type overriden to CUDA1
Tensor blk.4.attn_k_b.weight buffer type overriden to CUDA1
Tensor blk.4.attn_v_b.weight buffer type overriden to CUDA1
Tensor blk.4.attn_output.weight buffer type overriden to CUDA1
Tensor blk.4.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.4.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.4.exp_probs_b.bias buffer type overriden to CUDA1
Tensor blk.4.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.4.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.4.ffn_up_shexp.weight buffer type overriden to CUDA1

... log is too long, abbreviating ...

Tensor blk.57.attn_norm.weight buffer type overriden to CUDA1
Tensor blk.57.attn_q_a_norm.weight buffer type overriden to CUDA1
Tensor blk.57.attn_kv_a_norm.weight buffer type overriden to CUDA1
Tensor blk.57.attn_q_a.weight buffer type overriden to CUDA1
Tensor blk.57.attn_q_b.weight buffer type overriden to CUDA1
Tensor blk.57.attn_kv_a_mqa.weight buffer type overriden to CUDA1
Tensor blk.57.attn_kv_b.weight buffer type overriden to CUDA1
Tensor blk.57.attn_k_b.weight buffer type overriden to CUDA1
Tensor blk.57.attn_v_b.weight buffer type overriden to CUDA1
Tensor blk.57.attn_output.weight buffer type overriden to CUDA1
Tensor blk.57.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.57.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.57.exp_probs_b.bias buffer type overriden to CUDA1
Tensor blk.57.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.57.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.57.ffn_up_shexp.weight buffer type overriden to CUDA1
Tensor blk.58.attn_norm.weight buffer type overriden to CUDA1
Tensor blk.58.attn_q_a_norm.weight buffer type overriden to CUDA1
Tensor blk.58.attn_kv_a_norm.weight buffer type overriden to CUDA1
Tensor blk.58.attn_q_a.weight buffer type overriden to CUDA1
Tensor blk.58.attn_q_b.weight buffer type overriden to CUDA1
Tensor blk.58.attn_kv_a_mqa.weight buffer type overriden to CUDA1
Tensor blk.58.attn_kv_b.weight buffer type overriden to CUDA1
Tensor blk.58.attn_k_b.weight buffer type overriden to CUDA1
Tensor blk.58.attn_v_b.weight buffer type overriden to CUDA1
Tensor blk.58.attn_output.weight buffer type overriden to CUDA1
Tensor blk.58.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.58.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.58.exp_probs_b.bias buffer type overriden to CUDA1
Tensor blk.58.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.58.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.58.ffn_up_shexp.weight buffer type overriden to CUDA1
Tensor blk.59.attn_norm.weight buffer type overriden to CUDA1
Tensor blk.59.attn_q_a_norm.weight buffer type overriden to CUDA1
Tensor blk.59.attn_kv_a_norm.weight buffer type overriden to CUDA1
Tensor blk.59.attn_q_a.weight buffer type overriden to CUDA1
Tensor blk.59.attn_q_b.weight buffer type overriden to CUDA1
Tensor blk.59.attn_kv_a_mqa.weight buffer type overriden to CUDA1
Tensor blk.59.attn_kv_b.weight buffer type overriden to CUDA1
Tensor blk.59.attn_k_b.weight buffer type overriden to CUDA1
Tensor blk.59.attn_v_b.weight buffer type overriden to CUDA1
Tensor blk.59.attn_output.weight buffer type overriden to CUDA1
Tensor blk.59.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.59.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.59.exp_probs_b.bias buffer type overriden to CUDA1
Tensor blk.59.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.59.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.59.ffn_up_shexp.weight buffer type overriden to CUDA1
Tensor blk.60.attn_norm.weight buffer type overriden to CUDA1
Tensor blk.60.attn_q_a_norm.weight buffer type overriden to CUDA1
Tensor blk.60.attn_kv_a_norm.weight buffer type overriden to CUDA1
Tensor blk.60.attn_q_a.weight buffer type overriden to CUDA1
Tensor blk.60.attn_q_b.weight buffer type overriden to CUDA1
Tensor blk.60.attn_kv_a_mqa.weight buffer type overriden to CUDA1
Tensor blk.60.attn_kv_b.weight buffer type overriden to CUDA1
Tensor blk.60.attn_k_b.weight buffer type overriden to CUDA1
Tensor blk.60.attn_v_b.weight buffer type overriden to CUDA1
Tensor blk.60.attn_output.weight buffer type overriden to CUDA1
Tensor blk.60.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.60.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.60.exp_probs_b.bias buffer type overriden to CUDA1
Tensor blk.60.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.60.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.60.ffn_up_shexp.weight buffer type overriden to CUDA1
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size = 385631.46 MiB
llm_load_tensors:        CPU buffer size =   497.11 MiB
llm_load_tensors:      CUDA0 buffer size =   937.60 MiB
llm_load_tensors:      CUDA1 buffer size = 10959.57 MiB
....................................................................................................
============ llm_prepare_mla: need to compute 61 wk_b/wv_b tensors
Computed blk.0.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.1.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.2.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.3.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.4.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.5.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.6.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.7.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.8.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.9.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.10.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.11.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.12.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.13.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.14.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.15.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.16.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.17.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.18.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.19.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.20.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.21.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.22.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.23.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.24.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.25.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.26.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.27.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.28.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.29.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.30.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.31.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.32.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.33.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.34.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.35.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.36.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.37.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.38.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.39.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.40.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.41.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.42.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.43.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.44.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.45.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.46.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.47.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.48.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.49.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.50.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.51.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.52.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.53.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.54.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.55.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.56.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.57.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.58.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.59.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
Computed blk.60.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA1
llama_new_context_with_model: n_ctx      = 131072
llama_new_context_with_model: n_batch    = 1024
llama_new_context_with_model: n_ubatch   = 1024
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 3
llama_new_context_with_model: attn_max_b = 512
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init:      CUDA0 KV buffer size =  4666.53 MiB
llama_new_context_with_model: KV self size  = 4666.50 MiB, c^KV (q8_0): 4666.50 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.99 MiB
llama_new_context_with_model:      CUDA0 compute buffer size = 11718.25 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   540.01 MiB
llama_new_context_with_model: graph nodes  = 24349
llama_new_context_with_model: graph splits = 302
CUDA error: an illegal memory access was encountered
  current device: 0, in function ggml_backend_cuda_synchronize at /home/corey/ik_llama.cpp/ggml/src/ggml-cuda.cu:3073
  cudaStreamSynchronize(cuda_ctx->stream())
/home/corey/ik_llama.cpp/ggml/src/ggml-cuda.cu:110: CUDA error
./run_deepseek_ik: line 71: 55704 Aborted                 (core dumped) ~/ik_llama.cpp/build/bin/llama-server "${args[@]}"
```

---

#### 💬 Conversation

👤 **cmoncure** commented the **2025-06-02** at **21:15:21**:<br>

This is down to the ergonomics of the configuration options.
Adding -mg 1 solves it.  I don't think this should result in a segfault though. Alas, you're just one guy. 
Closing