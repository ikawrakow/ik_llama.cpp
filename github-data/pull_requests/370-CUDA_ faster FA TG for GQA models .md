### 🔀 [#370](https://github.com/ikawrakow/ik_llama.cpp/pull/370) - CUDA: faster FA TG for GQA models 

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-03 |
| **Updated** | 2025-05-04 |

---

#### Description

This PR improves CUDA FA performance for token generation by a significant margin.

It is derived from [mainline PR 12014](https://github.com/ggml-org/llama.cpp/pull/12014), but as the two code bases have diverged, significant adaptation was required.

The following graph shows a TG performance comparison for Qwen3-30B-A3B between the main branch (blue symbols) and this PR (shown in red) quantized with `Q4_0` (so we can also include mainline `llama.cpp` results shown in black). The x-xis is `N_KV`, the number of tokens in the KV cache. My GPU is RTX-4080, so the model cannot be fully offloaded. But to simulate the situation of someone running Qwen3-235B-A22B on a 24GB GPU, I have left all but the first 8 experts layers (1/6 of layers) on the CPU (Ryzen-5975WX).  

![qwen3_hybrid](https://github.com/user-attachments/assets/cecc7d5e-057f-4b3a-9043-6c86603aa896)

@ubergarm It would be great if you could test this PR with the models where you saw mainline outperforming `ik_llama.cpp` for TG with large contexts.

**Of note**: in mainline, the condition to invoke the MMA kernel for TG is this:
```c++
    const bool gqa_opt_applies = ((Q->ne[2] / K->ne[2]) % 2 == 0) && mask; // The mma-based kernels have GQA-specific optimizations
    const bool mma_needs_data_conversion = K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16;
    const bool mma_faster_for_bs1 = new_mma_available(cc) && gqa_opt_applies && cc < GGML_CUDA_CC_ADA_LOVELACE && !mma_needs_data_conversion;
```

My GPU is `ADA_LOVELACE`, so the MMA kernel does not get invoked for TG. But based on my testing, it is much faster to use the new MMA kernel also for TG, in addition to being also slightly faster when data conversion is required (i.e., quantized KV cache). So, I'm not really sure why it was done that way in mainline, but I have decided to invoke the new kernel if the GPU supports MMA and `Q->ne[2] / K->ne[2]) % 2 == 0`.

---

#### 💬 Conversation

👤 **ubergarm** commented the **2025-05-03** at **20:24:02**:<br>

Wow, I'll let the benchmarks speak for themselves.

---

## bartowski/THUDM_GLM-Z1-32B-0414-IQ4_XS

Just ran this efficient GQA model on home rig given it offloads fully fitting 32k context easily on <24GB VRAM without quantizing kv-cache.

I'll run some more benchmarks with the new Qwen3 MoEs and add below.

![thud-sweep-pr370](https://github.com/user-attachments/assets/2d075d46-94e0-4c41-9d68-d2aa06b44a1c)

<details>

<summary>👈Logs</summary>

## `llama.cpp/master@36667c8e` + `ug/port-sweep-bench@d541533a`
```
cmake -B build -DGGML_CUDA=ON -DGGML_RPC=OFF -DGGML_BLAS=OFF
cmake --build build --config Release -j $(nproc)

CUDA_VISIBLE_DEVICE=0 \
./build/bin/llama-sweep-bench \
    --model /mnt/astrodata/llm/models/bartowski/THUDM_GLM-Z1-32B-0414-GGUF/THUDM_GLM-Z1-32B-0414-IQ4_XS.gguf \
    -fa \
    -ctk f16 -ctv f16 \
    -c 32768 \
    -ngl 99 \
    --threads 1

ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090 Ti, compute capability 8.6, VMM: yes
build: 5274 (d541533a) with cc (GCC) 14.2.1 20250128 for x86_64-pc-linux-gnu
llama_model_load_from_file_impl: using device CUDA0 (NVIDIA GeForce RTX 3090 Ti) - 23041 MiB free
llama_model_loader: loaded meta data with 37 key-value pairs and 613 tensors from /mnt/astrodata/llm/models/bartowski/THUDM_GLM-Z1-32B-0414-GGUF/THUDM_GLM-Z1-32B-0414-IQ4_XS.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = glm4
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = GLM Z1 32B 0414
llama_model_loader: - kv   3:                            general.version str              = 0414
llama_model_loader: - kv   4:                           general.basename str              = GLM-Z1
llama_model_loader: - kv   5:                         general.size_label str              = 32B
llama_model_loader: - kv   6:                            general.license str              = mit
llama_model_loader: - kv   7:                               general.tags arr[str,1]       = ["text-generation"]
llama_model_loader: - kv   8:                          general.languages arr[str,2]       = ["zh", "en"]
llama_model_loader: - kv   9:                           glm4.block_count u32              = 61
llama_model_loader: - kv  10:                        glm4.context_length u32              = 32768
llama_model_loader: - kv  11:                      glm4.embedding_length u32              = 6144
llama_model_loader: - kv  12:                   glm4.feed_forward_length u32              = 23040
llama_model_loader: - kv  13:                  glm4.attention.head_count u32              = 48
llama_model_loader: - kv  14:               glm4.attention.head_count_kv u32              = 2
llama_model_loader: - kv  15:                        glm4.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  16:      glm4.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  17:                  glm4.attention.key_length u32              = 128
llama_model_loader: - kv  18:                glm4.attention.value_length u32              = 128
llama_model_loader: - kv  19:                  glm4.rope.dimension_count u32              = 64
llama_model_loader: - kv  20:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  21:                         tokenizer.ggml.pre str              = glm4
llama_model_loader: - kv  22:                      tokenizer.ggml.tokens arr[str,151552]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  23:                  tokenizer.ggml.token_type arr[i32,151552]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  24:                      tokenizer.ggml.merges arr[str,318088]  = ["Ġ Ġ", "Ġ ĠĠĠ", "ĠĠ ĠĠ", "...
llama_model_loader: - kv  25:                tokenizer.ggml.eos_token_id u32              = 151329
llama_model_loader: - kv  26:            tokenizer.ggml.padding_token_id u32              = 151329
llama_model_loader: - kv  27:                tokenizer.ggml.eot_token_id u32              = 151336
llama_model_loader: - kv  28:            tokenizer.ggml.unknown_token_id u32              = 151329
llama_model_loader: - kv  29:                tokenizer.ggml.bos_token_id u32              = 151331
llama_model_loader: - kv  30:                    tokenizer.chat_template str              = [gMASK]<sop>{%- if tools -%}<|system|...
llama_model_loader: - kv  31:               general.quantization_version u32              = 2
llama_model_loader: - kv  32:                          general.file_type u32              = 30
llama_model_loader: - kv  33:                      quantize.imatrix.file str              = /models_out/GLM-Z1-32B-0414-GGUF/THUD...
llama_model_loader: - kv  34:                   quantize.imatrix.dataset str              = /training_dir/calibration_datav3.txt
llama_model_loader: - kv  35:             quantize.imatrix.entries_count i32              = 366
llama_model_loader: - kv  36:              quantize.imatrix.chunks_count i32              = 125
llama_model_loader: - type  f32:  245 tensors
llama_model_loader: - type q5_K:   61 tensors
llama_model_loader: - type q6_K:    1 tensors
llama_model_loader: - type iq4_xs:  306 tensors
print_info: file format = GGUF V3 (latest)
print_info: file type   = IQ4_XS - 4.25 bpw
print_info: file size   = 16.38 GiB (4.32 BPW)
load: special_eot_id is not in special_eog_ids - the tokenizer config may be incorrect
load: special tokens cache size = 14
load: token to piece cache size = 0.9710 MB
print_info: arch             = glm4
print_info: vocab_only       = 0
print_info: n_ctx_train      = 32768
print_info: n_embd           = 6144
print_info: n_layer          = 61
print_info: n_head           = 48
print_info: n_head_kv        = 2
print_info: n_rot            = 64
print_info: n_swa            = 0
print_info: n_swa_pattern    = 1
print_info: n_embd_head_k    = 128
print_info: n_embd_head_v    = 128
print_info: n_gqa            = 24
print_info: n_embd_k_gqa     = 256
print_info: n_embd_v_gqa     = 256
print_info: f_norm_eps       = 0.0e+00
print_info: f_norm_rms_eps   = 1.0e-05
print_info: f_clamp_kqv      = 0.0e+00
print_info: f_max_alibi_bias = 0.0e+00
print_info: f_logit_scale    = 0.0e+00
print_info: f_attn_scale     = 0.0e+00
print_info: n_ff             = 23040
print_info: n_expert         = 0
print_info: n_expert_used    = 0
print_info: causal attn      = 1
print_info: pooling type     = 0
print_info: rope type        = 0
print_info: rope scaling     = linear
print_info: freq_base_train  = 10000.0
print_info: freq_scale_train = 1
print_info: n_ctx_orig_yarn  = 32768
print_info: rope_finetuned   = unknown
print_info: ssm_d_conv       = 0
print_info: ssm_d_inner      = 0
print_info: ssm_d_state      = 0
print_info: ssm_dt_rank      = 0
print_info: ssm_dt_b_c_rms   = 0
print_info: model type       = 32B
print_info: model params     = 32.57 B
print_info: general.name     = GLM Z1 32B 0414
print_info: vocab type       = BPE
print_info: n_vocab          = 151552
print_info: n_merges         = 318088
print_info: BOS token        = 151331 '[gMASK]'
print_info: EOS token        = 151329 '<|endoftext|>'
print_info: EOT token        = 151336 '<|user|>'
print_info: UNK token        = 151329 '<|endoftext|>'
print_info: PAD token        = 151329 '<|endoftext|>'
print_info: LF token         = 198 'Ċ'
print_info: EOG token        = 151329 '<|endoftext|>'
print_info: EOG token        = 151336 '<|user|>'
print_info: max token length = 1024
load_tensors: loading model tensors, this can take a while... (mmap = true)
load_tensors: offloading 61 repeating layers to GPU
load_tensors: offloading output layer to GPU
load_tensors: offloaded 62/62 layers to GPU
load_tensors:        CUDA0 model buffer size = 16303.48 MiB
load_tensors:   CPU_Mapped model buffer size =   471.75 MiB
...............................................................................................
llama_context: constructing llama_context
llama_context: n_seq_max     = 1
llama_context: n_ctx         = 32768
llama_context: n_ctx_per_seq = 32768
llama_context: n_batch       = 2048
llama_context: n_ubatch      = 512
llama_context: causal_attn   = 1
llama_context: flash_attn    = 1
llama_context: freq_base     = 10000.0
llama_context: freq_scale    = 1
llama_context:  CUDA_Host  output buffer size =     0.58 MiB
llama_kv_cache_unified: kv_size = 32768, type_k = 'f16', type_v = 'f16', n_layer = 61, can_shift = 1, padding = 256
llama_kv_cache_unified:      CUDA0 KV buffer size =  1952.00 MiB
llama_kv_cache_unified: KV self size  = 1952.00 MiB, K (f16):  976.00 MiB, V (f16):  976.00 MiB
llama_context:      CUDA0 compute buffer size =   353.00 MiB
llama_context:  CUDA_Host compute buffer size =    76.01 MiB
llama_context: graph nodes  = 2264
llama_context: graph splits = 2

main: n_kv_max = 32768, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = 99, n_threads = 1, n_threads_batch = 1
```

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    0.389 |  1315.53 |    3.340 |    38.32 |
|   512 |    128 |    512 |    0.392 |  1307.60 |    3.375 |    37.93 |
|   512 |    128 |   1024 |    0.397 |  1289.34 |    3.389 |    37.77 |
|   512 |    128 |   1536 |    0.402 |  1274.98 |    3.404 |    37.61 |
|   512 |    128 |   2048 |    0.408 |  1255.85 |    3.432 |    37.30 |
|   512 |    128 |   2560 |    0.413 |  1240.58 |    3.446 |    37.15 |
|   512 |    128 |   3072 |    0.418 |  1225.45 |    3.462 |    36.98 |
|   512 |    128 |   3584 |    0.425 |  1206.09 |    3.481 |    36.77 |
|   512 |    128 |   4096 |    0.429 |  1194.00 |    3.494 |    36.63 |
|   512 |    128 |   4608 |    0.436 |  1174.09 |    3.520 |    36.36 |
|   512 |    128 |   5120 |    0.441 |  1160.13 |    3.535 |    36.21 |
|   512 |    128 |   5632 |    0.447 |  1144.45 |    3.551 |    36.05 |
|   512 |    128 |   6144 |    0.452 |  1131.78 |    3.563 |    35.93 |
|   512 |    128 |   6656 |    0.458 |  1118.88 |    3.576 |    35.79 |
|   512 |    128 |   7168 |    0.463 |  1106.17 |    3.630 |    35.26 |
|   512 |    128 |   7680 |    0.469 |  1092.81 |    3.635 |    35.21 |
|   512 |    128 |   8192 |    0.472 |  1084.46 |    3.642 |    35.15 |
|   512 |    128 |   8704 |    0.477 |  1073.18 |    3.647 |    35.09 |
|   512 |    128 |   9216 |    0.483 |  1059.62 |    3.668 |    34.90 |
|   512 |    128 |   9728 |    0.490 |  1044.46 |    3.672 |    34.86 |
|   512 |    128 |  10240 |    0.496 |  1032.17 |    3.677 |    34.82 |
|   512 |    128 |  10752 |    0.500 |  1024.54 |    3.685 |    34.73 |
|   512 |    128 |  11264 |    0.506 |  1011.07 |    3.693 |    34.66 |
|   512 |    128 |  11776 |    0.510 |  1004.38 |    3.701 |    34.59 |
|   512 |    128 |  12288 |    0.515 |   994.60 |    3.707 |    34.53 |
|   512 |    128 |  12800 |    0.521 |   981.83 |    3.718 |    34.43 |
|   512 |    128 |  13312 |    0.525 |   975.25 |    3.724 |    34.37 |
|   512 |    128 |  13824 |    0.531 |   964.33 |    3.730 |    34.31 |
|   512 |    128 |  14336 |    0.534 |   959.03 |    3.781 |    33.85 |
|   512 |    128 |  14848 |    0.542 |   944.73 |    3.786 |    33.81 |
|   512 |    128 |  15360 |    0.545 |   939.97 |    3.790 |    33.77 |
|   512 |    128 |  15872 |    0.550 |   930.56 |    3.797 |    33.71 |
|   512 |    128 |  16384 |    0.557 |   919.93 |    3.806 |    33.63 |
|   512 |    128 |  16896 |    0.560 |   913.74 |    3.811 |    33.59 |
|   512 |    128 |  17408 |    0.565 |   906.08 |    3.816 |    33.54 |
|   512 |    128 |  17920 |    0.571 |   896.21 |    3.824 |    33.47 |
|   512 |    128 |  18432 |    0.577 |   888.05 |    3.831 |    33.41 |
|   512 |    128 |  18944 |    0.581 |   881.73 |    3.837 |    33.36 |
|   512 |    128 |  19456 |    0.587 |   872.23 |    3.843 |    33.31 |
|   512 |    128 |  19968 |    0.591 |   865.79 |    3.851 |    33.24 |
|   512 |    128 |  20480 |    0.596 |   858.70 |    3.858 |    33.18 |
|   512 |    128 |  20992 |    0.601 |   852.12 |    3.865 |    33.12 |
|   512 |    128 |  21504 |    0.607 |   844.03 |    3.911 |    32.73 |
|   512 |    128 |  22016 |    0.611 |   838.24 |    3.916 |    32.69 |
|   512 |    128 |  22528 |    0.617 |   830.13 |    3.919 |    32.66 |
|   512 |    128 |  23040 |    0.622 |   823.01 |    3.926 |    32.61 |
|   512 |    128 |  23552 |    0.626 |   817.25 |    3.934 |    32.54 |
|   512 |    128 |  24064 |    0.632 |   810.53 |    3.940 |    32.49 |
|   512 |    128 |  24576 |    0.637 |   803.70 |    3.944 |    32.45 |
|   512 |    128 |  25088 |    0.642 |   797.69 |    3.953 |    32.38 |
|   512 |    128 |  25600 |    0.647 |   791.88 |    3.959 |    32.33 |
|   512 |    128 |  26112 |    0.654 |   782.78 |    3.967 |    32.27 |
|   512 |    128 |  26624 |    0.660 |   776.28 |    3.984 |    32.13 |
|   512 |    128 |  27136 |    0.664 |   771.40 |    3.992 |    32.06 |
|   512 |    128 |  27648 |    0.670 |   764.30 |    3.998 |    32.02 |
|   512 |    128 |  28160 |    0.674 |   759.80 |    4.003 |    31.98 |
|   512 |    128 |  28672 |    0.679 |   754.23 |    4.047 |    31.63 |
|   512 |    128 |  29184 |    0.685 |   747.87 |    4.054 |    31.57 |
|   512 |    128 |  29696 |    0.689 |   742.64 |    4.063 |    31.51 |
|   512 |    128 |  30208 |    0.696 |   735.78 |    4.066 |    31.48 |
|   512 |    128 |  30720 |    0.699 |   732.51 |    4.072 |    31.43 |
|   512 |    128 |  31232 |    0.706 |   725.56 |    4.079 |    31.38 |
|   512 |    128 |  31744 |    0.711 |   720.48 |    4.082 |    31.36 |
|   512 |    128 |  32256 |    0.715 |   715.85 |    4.090 |    31.30 |

## `ik_llama.cpp/main@ab7f694b`
```
cmake -B build -DGGML_CUDA=ON -DGGML_RPC=OFF -DGGML_BLAS=OFF
cmake --build build --config Release -j $(nproc)

CUDA_VISIBLE_DEVICE=0 \
./build/bin/llama-sweep-bench \
    --model /mnt/astrodata/llm/models/bartowski/THUDM_GLM-Z1-32B-0414-GGUF/THUDM_GLM-Z1-32B-0414-IQ4_XS.gguf \
    -fa \
    -ctk f16 -ctv f16 \
    -c 32768 \
    -ngl 99 \
    --threads 1

llama_model_loader: loaded meta data with 37 key-value pairs and 613 tensors from /mnt/astrodata/llm/models/bartowski/THUDM_GLM-Z1-32B-0414-GGUF/THUDM_GLM-Z1-32B-0414-IQ4_XS.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = glm4
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = GLM Z1 32B 0414
llama_model_loader: - kv   3:                            general.version str              = 0414
llama_model_loader: - kv   4:                           general.basename str              = GLM-Z1
llama_model_loader: - kv   5:                         general.size_label str              = 32B
llama_model_loader: - kv   6:                            general.license str              = mit
llama_model_loader: - kv   7:                               general.tags arr[str,1]       = ["text-generation"]
llama_model_loader: - kv   8:                          general.languages arr[str,2]       = ["zh", "en"]
llama_model_loader: - kv   9:                           glm4.block_count u32              = 61
llama_model_loader: - kv  10:                        glm4.context_length u32              = 32768
llama_model_loader: - kv  11:                      glm4.embedding_length u32              = 6144
llama_model_loader: - kv  12:                   glm4.feed_forward_length u32              = 23040
llama_model_loader: - kv  13:                  glm4.attention.head_count u32              = 48
llama_model_loader: - kv  14:               glm4.attention.head_count_kv u32              = 2
llama_model_loader: - kv  15:                        glm4.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  16:      glm4.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  17:                  glm4.attention.key_length u32              = 128
llama_model_loader: - kv  18:                glm4.attention.value_length u32              = 128
llama_model_loader: - kv  19:                  glm4.rope.dimension_count u32              = 64
llama_model_loader: - kv  20:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  21:                         tokenizer.ggml.pre str              = glm4
llama_model_loader: - kv  22:                      tokenizer.ggml.tokens arr[str,151552]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  23:                  tokenizer.ggml.token_type arr[i32,151552]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  24:                      tokenizer.ggml.merges arr[str,318088]  = ["Ġ Ġ", "Ġ ĠĠĠ", "ĠĠ ĠĠ", "...
llama_model_loader: - kv  25:                tokenizer.ggml.eos_token_id u32              = 151329
llama_model_loader: - kv  26:            tokenizer.ggml.padding_token_id u32              = 151329
llama_model_loader: - kv  27:                tokenizer.ggml.eot_token_id u32              = 151336
llama_model_loader: - kv  28:            tokenizer.ggml.unknown_token_id u32              = 151329
llama_model_loader: - kv  29:                tokenizer.ggml.bos_token_id u32              = 151331
llama_model_loader: - kv  30:                    tokenizer.chat_template str              = [gMASK]<sop>{%- if tools -%}<|system|...
llama_model_loader: - kv  31:               general.quantization_version u32              = 2
llama_model_loader: - kv  32:                          general.file_type u32              = 30
llama_model_loader: - kv  33:                      quantize.imatrix.file str              = /models_out/GLM-Z1-32B-0414-GGUF/THUD...
llama_model_loader: - kv  34:                   quantize.imatrix.dataset str              = /training_dir/calibration_datav3.txt
llama_model_loader: - kv  35:             quantize.imatrix.entries_count i32              = 366
llama_model_loader: - kv  36:              quantize.imatrix.chunks_count i32              = 125
llama_model_loader: - type  f32:  245 tensors
llama_model_loader: - type q5_K:   61 tensors
llama_model_loader: - type q6_K:    1 tensors
llama_model_loader: - type iq4_xs:  306 tensors
llm_load_vocab: special tokens cache size = 14
llm_load_vocab: token to piece cache size = 0.9710 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = glm4
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 151552
llm_load_print_meta: n_merges         = 318088
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 32768
llm_load_print_meta: n_embd           = 6144
llm_load_print_meta: n_layer          = 61
llm_load_print_meta: n_head           = 48
llm_load_print_meta: n_head_kv        = 2
llm_load_print_meta: n_rot            = 64
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 24
llm_load_print_meta: n_embd_k_gqa     = 256
llm_load_print_meta: n_embd_v_gqa     = 256
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 23040
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 32768
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = 32B
llm_load_print_meta: model ftype      = IQ4_XS - 4.25 bpw
llm_load_print_meta: model params     = 32.566 B
llm_load_print_meta: model size       = 16.382 GiB (4.321 BPW)
llm_load_print_meta: repeating layers = 15.210 GiB (4.255 BPW, 30.704 B parameters)
llm_load_print_meta: general.name     = GLM Z1 32B 0414
llm_load_print_meta: BOS token        = 151331 '[gMASK]'
llm_load_print_meta: EOS token        = 151329 '<|endoftext|>'
llm_load_print_meta: UNK token        = 151329 '<|endoftext|>'
llm_load_print_meta: PAD token        = 151329 '<|endoftext|>'
llm_load_print_meta: LF token         = 128 'Ä'
llm_load_print_meta: EOT token        = 151336 '<|user|>'
llm_load_print_meta: max token length = 1024
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090 Ti, compute capability 8.6, VMM: yes
llm_load_tensors: ggml ctx size =    0.56 MiB
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size =   471.75 MiB
llm_load_tensors:      CUDA0 buffer size = 16303.48 MiB
...............................................................................................
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 0
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:      CUDA0 KV buffer size =  1952.00 MiB
llama_new_context_with_model: KV self size  = 1952.00 MiB, K (f16):  976.00 MiB, V (f16):  976.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.58 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =   308.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =    76.01 MiB
llama_new_context_with_model: graph nodes  = 1592
llama_new_context_with_model: graph splits = 2

main: n_kv_max = 32768, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = 99, n_threads = 1, n_threads_batch = 1
```

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    0.332 |  1543.09 |    3.197 |    40.04 |
|   512 |    128 |    512 |    0.341 |  1499.79 |    3.255 |    39.33 |
|   512 |    128 |   1024 |    0.351 |  1460.32 |    3.322 |    38.53 |
|   512 |    128 |   1536 |    0.362 |  1415.64 |    3.378 |    37.89 |
|   512 |    128 |   2048 |    0.370 |  1382.04 |    3.437 |    37.24 |
|   512 |    128 |   2560 |    0.382 |  1341.54 |    3.497 |    36.61 |
|   512 |    128 |   3072 |    0.392 |  1306.79 |    3.563 |    35.93 |
|   512 |    128 |   3584 |    0.402 |  1273.12 |    3.619 |    35.37 |
|   512 |    128 |   4096 |    0.412 |  1241.79 |    3.676 |    34.82 |
|   512 |    128 |   4608 |    0.424 |  1207.80 |    3.736 |    34.26 |
|   512 |    128 |   5120 |    0.434 |  1179.37 |    3.799 |    33.69 |
|   512 |    128 |   5632 |    0.444 |  1152.20 |    3.856 |    33.20 |
|   512 |    128 |   6144 |    0.455 |  1125.49 |    3.910 |    32.74 |
|   512 |    128 |   6656 |    0.467 |  1097.18 |    3.967 |    32.27 |
|   512 |    128 |   7168 |    0.477 |  1073.14 |    4.036 |    31.71 |
|   512 |    128 |   7680 |    0.488 |  1049.85 |    4.093 |    31.28 |
|   512 |    128 |   8192 |    0.497 |  1029.15 |    4.149 |    30.85 |
|   512 |    128 |   8704 |    0.508 |  1008.35 |    4.207 |    30.43 |
|   512 |    128 |   9216 |    0.519 |   987.41 |    4.263 |    30.03 |
|   512 |    128 |   9728 |    0.529 |   968.23 |    4.317 |    29.65 |
|   512 |    128 |  10240 |    0.539 |   949.58 |    4.371 |    29.28 |
|   512 |    128 |  10752 |    0.549 |   933.10 |    4.427 |    28.92 |
|   512 |    128 |  11264 |    0.561 |   913.08 |    4.483 |    28.55 |
|   512 |    128 |  11776 |    0.571 |   896.76 |    4.553 |    28.12 |
|   512 |    128 |  12288 |    0.581 |   881.75 |    4.610 |    27.76 |
|   512 |    128 |  12800 |    0.590 |   867.17 |    4.664 |    27.45 |
|   512 |    128 |  13312 |    0.602 |   849.99 |    4.720 |    27.12 |
|   512 |    128 |  13824 |    0.613 |   835.39 |    4.771 |    26.83 |
|   512 |    128 |  14336 |    0.622 |   822.59 |    4.827 |    26.52 |
|   512 |    128 |  14848 |    0.633 |   808.34 |    4.883 |    26.21 |
|   512 |    128 |  15360 |    0.641 |   798.21 |    4.939 |    25.92 |
|   512 |    128 |  15872 |    0.654 |   783.33 |    4.995 |    25.63 |
|   512 |    128 |  16384 |    0.663 |   771.99 |    5.047 |    25.36 |
|   512 |    128 |  16896 |    0.674 |   759.74 |    5.102 |    25.09 |
|   512 |    128 |  17408 |    0.682 |   750.42 |    5.158 |    24.81 |
|   512 |    128 |  17920 |    0.692 |   740.16 |    5.216 |    24.54 |
|   512 |    128 |  18432 |    0.702 |   729.00 |    5.272 |    24.28 |
|   512 |    128 |  18944 |    0.712 |   719.48 |    5.325 |    24.04 |
|   512 |    128 |  19456 |    0.722 |   709.41 |    5.380 |    23.79 |
|   512 |    128 |  19968 |    0.732 |   699.34 |    5.437 |    23.54 |
|   512 |    128 |  20480 |    0.742 |   689.87 |    5.491 |    23.31 |
|   512 |    128 |  20992 |    0.752 |   680.47 |    5.542 |    23.10 |
|   512 |    128 |  21504 |    0.761 |   672.51 |    5.598 |    22.86 |
|   512 |    128 |  22016 |    0.773 |   662.26 |    5.650 |    22.65 |
|   512 |    128 |  22528 |    0.783 |   653.79 |    5.704 |    22.44 |
|   512 |    128 |  23040 |    0.793 |   645.83 |    5.758 |    22.23 |
|   512 |    128 |  23552 |    0.802 |   638.69 |    5.815 |    22.01 |
|   512 |    128 |  24064 |    0.813 |   629.53 |    5.869 |    21.81 |
|   512 |    128 |  24576 |    0.822 |   622.92 |    5.923 |    21.61 |
|   512 |    128 |  25088 |    0.833 |   614.68 |    5.982 |    21.40 |
|   512 |    128 |  25600 |    0.841 |   608.59 |    6.034 |    21.21 |
|   512 |    128 |  26112 |    0.852 |   600.84 |    6.092 |    21.01 |
|   512 |    128 |  26624 |    0.862 |   594.04 |    6.148 |    20.82 |
|   512 |    128 |  27136 |    0.872 |   587.23 |    6.203 |    20.63 |
|   512 |    128 |  27648 |    0.882 |   580.33 |    6.255 |    20.46 |
|   512 |    128 |  28160 |    0.893 |   573.62 |    6.312 |    20.28 |
|   512 |    128 |  28672 |    0.903 |   567.14 |    6.367 |    20.10 |
|   512 |    128 |  29184 |    0.913 |   560.69 |    6.424 |    19.92 |
|   512 |    128 |  29696 |    0.924 |   554.36 |    6.479 |    19.75 |
|   512 |    128 |  30208 |    0.934 |   548.27 |    6.535 |    19.59 |
|   512 |    128 |  30720 |    0.944 |   542.16 |    6.592 |    19.42 |
|   512 |    128 |  31232 |    0.955 |   536.23 |    6.648 |    19.25 |
|   512 |    128 |  31744 |    0.965 |   530.46 |    6.701 |    19.10 |
|   512 |    128 |  32256 |    0.976 |   524.59 |    6.776 |    18.89 |

## `ik_llama.cpp/ik/fattn_mma@056f0818` PR370
```
cmake -B build -DGGML_CUDA=ON -DGGML_RPC=OFF -DGGML_BLAS=OFF
cmake --build build --config Release -j $(nproc)

CUDA_VISIBLE_DEVICE=0 \
./build/bin/llama-sweep-bench \
    --model /mnt/astrodata/llm/models/bartowski/THUDM_GLM-Z1-32B-0414-GGUF/THUDM_GLM-Z1-32B-0414-IQ4_XS.gguf \
    -fa \
    -ctk f16 -ctv f16 \
    -c 32768 \
    -ngl 99 \
    --threads 1

llama_model_loader: loaded meta data with 37 key-value pairs and 613 tensors from /mnt/astrodata/llm/models/bartowski/THUDM_GLM-Z1-32B-0414-GGUF/THUDM_GLM-Z1-32B-0414-IQ4_XS.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = glm4
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = GLM Z1 32B 0414
llama_model_loader: - kv   3:                            general.version str              = 0414
llama_model_loader: - kv   4:                           general.basename str              = GLM-Z1
llama_model_loader: - kv   5:                         general.size_label str              = 32B
llama_model_loader: - kv   6:                            general.license str              = mit
llama_model_loader: - kv   7:                               general.tags arr[str,1]       = ["text-generation"]
llama_model_loader: - kv   8:                          general.languages arr[str,2]       = ["zh", "en"]
llama_model_loader: - kv   9:                           glm4.block_count u32              = 61
llama_model_loader: - kv  10:                        glm4.context_length u32              = 32768
llama_model_loader: - kv  11:                      glm4.embedding_length u32              = 6144
llama_model_loader: - kv  12:                   glm4.feed_forward_length u32              = 23040
llama_model_loader: - kv  13:                  glm4.attention.head_count u32              = 48
llama_model_loader: - kv  14:               glm4.attention.head_count_kv u32              = 2
llama_model_loader: - kv  15:                        glm4.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  16:      glm4.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  17:                  glm4.attention.key_length u32              = 128
llama_model_loader: - kv  18:                glm4.attention.value_length u32              = 128
llama_model_loader: - kv  19:                  glm4.rope.dimension_count u32              = 64
llama_model_loader: - kv  20:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  21:                         tokenizer.ggml.pre str              = glm4
llama_model_loader: - kv  22:                      tokenizer.ggml.tokens arr[str,151552]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  23:                  tokenizer.ggml.token_type arr[i32,151552]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  24:                      tokenizer.ggml.merges arr[str,318088]  = ["Ġ Ġ", "Ġ ĠĠĠ", "ĠĠ ĠĠ", "...
llama_model_loader: - kv  25:                tokenizer.ggml.eos_token_id u32              = 151329
llama_model_loader: - kv  26:            tokenizer.ggml.padding_token_id u32              = 151329
llama_model_loader: - kv  27:                tokenizer.ggml.eot_token_id u32              = 151336
llama_model_loader: - kv  28:            tokenizer.ggml.unknown_token_id u32              = 151329
llama_model_loader: - kv  29:                tokenizer.ggml.bos_token_id u32              = 151331
llama_model_loader: - kv  30:                    tokenizer.chat_template str              = [gMASK]<sop>{%- if tools -%}<|system|...
llama_model_loader: - kv  31:               general.quantization_version u32              = 2
llama_model_loader: - kv  32:                          general.file_type u32              = 30
llama_model_loader: - kv  33:                      quantize.imatrix.file str              = /models_out/GLM-Z1-32B-0414-GGUF/THUD...
llama_model_loader: - kv  34:                   quantize.imatrix.dataset str              = /training_dir/calibration_datav3.txt
llama_model_loader: - kv  35:             quantize.imatrix.entries_count i32              = 366
llama_model_loader: - kv  36:              quantize.imatrix.chunks_count i32              = 125
llama_model_loader: - type  f32:  245 tensors
llama_model_loader: - type q5_K:   61 tensors
llama_model_loader: - type q6_K:    1 tensors
llama_model_loader: - type iq4_xs:  306 tensors
llm_load_vocab: special tokens cache size = 14
llm_load_vocab: token to piece cache size = 0.9710 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = glm4
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 151552
llm_load_print_meta: n_merges         = 318088
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 32768
llm_load_print_meta: n_embd           = 6144
llm_load_print_meta: n_layer          = 61
llm_load_print_meta: n_head           = 48
llm_load_print_meta: n_head_kv        = 2
llm_load_print_meta: n_rot            = 64
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 24
llm_load_print_meta: n_embd_k_gqa     = 256
llm_load_print_meta: n_embd_v_gqa     = 256
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 23040
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 32768
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = 32B
llm_load_print_meta: model ftype      = IQ4_XS - 4.25 bpw
llm_load_print_meta: model params     = 32.566 B
llm_load_print_meta: model size       = 16.382 GiB (4.321 BPW)
llm_load_print_meta: repeating layers = 15.210 GiB (4.255 BPW, 30.704 B parameters)
llm_load_print_meta: general.name     = GLM Z1 32B 0414
llm_load_print_meta: BOS token        = 151331 '[gMASK]'
llm_load_print_meta: EOS token        = 151329 '<|endoftext|>'
llm_load_print_meta: UNK token        = 151329 '<|endoftext|>'
llm_load_print_meta: PAD token        = 151329 '<|endoftext|>'
llm_load_print_meta: LF token         = 128 'Ä'
llm_load_print_meta: EOT token        = 151336 '<|user|>'
llm_load_print_meta: max token length = 1024
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090 Ti, compute capability 8.6, VMM: yes
llm_load_tensors: ggml ctx size =    0.56 MiB
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size =   471.75 MiB
llm_load_tensors:      CUDA0 buffer size = 16303.48 MiB
...............................................................................................
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 0
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:      CUDA0 KV buffer size =  1952.00 MiB
llama_new_context_with_model: KV self size  = 1952.00 MiB, K (f16):  976.00 MiB, V (f16):  976.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.58 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =   308.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =    76.01 MiB
llama_new_context_with_model: graph nodes  = 1592
llama_new_context_with_model: graph splits = 2

main: n_kv_max = 32768, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = 99, n_threads = 1, n_threads_batch = 1
```

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    0.328 |  1561.71 |    3.223 |    39.72 |
|   512 |    128 |    512 |    0.334 |  1535.12 |    3.235 |    39.57 |
|   512 |    128 |   1024 |    0.339 |  1511.39 |    3.253 |    39.35 |
|   512 |    128 |   1536 |    0.345 |  1485.88 |    3.273 |    39.11 |
|   512 |    128 |   2048 |    0.350 |  1462.82 |    3.297 |    38.83 |
|   512 |    128 |   2560 |    0.355 |  1440.97 |    3.312 |    38.64 |
|   512 |    128 |   3072 |    0.361 |  1416.81 |    3.333 |    38.40 |
|   512 |    128 |   3584 |    0.367 |  1395.76 |    3.352 |    38.19 |
|   512 |    128 |   4096 |    0.372 |  1375.23 |    3.367 |    38.02 |
|   512 |    128 |   4608 |    0.378 |  1353.27 |    3.382 |    37.85 |
|   512 |    128 |   5120 |    0.384 |  1333.01 |    3.403 |    37.62 |
|   512 |    128 |   5632 |    0.390 |  1311.91 |    3.419 |    37.44 |
|   512 |    128 |   6144 |    0.396 |  1294.48 |    3.432 |    37.30 |
|   512 |    128 |   6656 |    0.401 |  1277.80 |    3.446 |    37.14 |
|   512 |    128 |   7168 |    0.405 |  1262.77 |    3.499 |    36.59 |
|   512 |    128 |   7680 |    0.410 |  1247.29 |    3.507 |    36.50 |
|   512 |    128 |   8192 |    0.417 |  1227.94 |    3.525 |    36.31 |
|   512 |    128 |   8704 |    0.423 |  1211.30 |    3.531 |    36.25 |
|   512 |    128 |   9216 |    0.428 |  1195.33 |    3.543 |    36.13 |
|   512 |    128 |   9728 |    0.433 |  1182.45 |    3.551 |    36.05 |
|   512 |    128 |  10240 |    0.438 |  1167.71 |    3.557 |    35.99 |
|   512 |    128 |  10752 |    0.444 |  1153.72 |    3.566 |    35.90 |
|   512 |    128 |  11264 |    0.449 |  1141.40 |    3.575 |    35.81 |
|   512 |    128 |  11776 |    0.454 |  1127.10 |    3.582 |    35.73 |
|   512 |    128 |  12288 |    0.459 |  1115.10 |    3.588 |    35.68 |
|   512 |    128 |  12800 |    0.465 |  1102.12 |    3.599 |    35.56 |
|   512 |    128 |  13312 |    0.470 |  1089.86 |    3.605 |    35.51 |
|   512 |    128 |  13824 |    0.476 |  1076.57 |    3.612 |    35.44 |
|   512 |    128 |  14336 |    0.481 |  1065.35 |    3.666 |    34.91 |
|   512 |    128 |  14848 |    0.486 |  1053.61 |    3.672 |    34.86 |
|   512 |    128 |  15360 |    0.491 |  1043.09 |    3.677 |    34.81 |
|   512 |    128 |  15872 |    0.496 |  1031.87 |    3.683 |    34.75 |
|   512 |    128 |  16384 |    0.502 |  1020.64 |    3.692 |    34.67 |
|   512 |    128 |  16896 |    0.507 |  1010.75 |    3.696 |    34.63 |
|   512 |    128 |  17408 |    0.512 |   999.96 |    3.701 |    34.59 |
|   512 |    128 |  17920 |    0.517 |   989.76 |    3.711 |    34.49 |
|   512 |    128 |  18432 |    0.523 |   979.80 |    3.716 |    34.45 |
|   512 |    128 |  18944 |    0.528 |   970.51 |    3.722 |    34.39 |
|   512 |    128 |  19456 |    0.533 |   960.35 |    3.726 |    34.35 |
|   512 |    128 |  19968 |    0.538 |   951.88 |    3.738 |    34.25 |
|   512 |    128 |  20480 |    0.544 |   941.54 |    3.745 |    34.18 |
|   512 |    128 |  20992 |    0.548 |   934.39 |    3.749 |    34.14 |
|   512 |    128 |  21504 |    0.553 |   925.82 |    3.796 |    33.72 |
|   512 |    128 |  22016 |    0.558 |   917.11 |    3.802 |    33.67 |
|   512 |    128 |  22528 |    0.564 |   908.05 |    3.805 |    33.64 |
|   512 |    128 |  23040 |    0.569 |   900.42 |    3.810 |    33.59 |
|   512 |    128 |  23552 |    0.574 |   892.28 |    3.819 |    33.52 |
|   512 |    128 |  24064 |    0.579 |   883.61 |    3.824 |    33.47 |
|   512 |    128 |  24576 |    0.584 |   876.66 |    3.828 |    33.44 |
|   512 |    128 |  25088 |    0.589 |   869.39 |    3.834 |    33.39 |
|   512 |    128 |  25600 |    0.593 |   863.38 |    3.839 |    33.35 |
|   512 |    128 |  26112 |    0.599 |   855.39 |    3.846 |    33.28 |
|   512 |    128 |  26624 |    0.604 |   847.30 |    3.849 |    33.25 |
|   512 |    128 |  27136 |    0.608 |   841.55 |    3.861 |    33.16 |
|   512 |    128 |  27648 |    0.614 |   833.60 |    3.865 |    33.12 |
|   512 |    128 |  28160 |    0.619 |   826.85 |    3.872 |    33.06 |
|   512 |    128 |  28672 |    0.624 |   819.95 |    3.916 |    32.68 |
|   512 |    128 |  29184 |    0.630 |   812.95 |    3.922 |    32.64 |
|   512 |    128 |  29696 |    0.634 |   807.09 |    3.928 |    32.59 |
|   512 |    128 |  30208 |    0.640 |   799.91 |    3.930 |    32.57 |
|   512 |    128 |  30720 |    0.645 |   793.95 |    3.939 |    32.49 |
|   512 |    128 |  31232 |    0.650 |   787.49 |    3.944 |    32.45 |
|   512 |    128 |  31744 |    0.655 |   781.47 |    3.947 |    32.43 |
|   512 |    128 |  32256 |    0.662 |   773.87 |    3.954 |    32.37 |

</details>

---

👤 **ubergarm** commented the **2025-05-03** at **21:39:11**:<br>

I suppose I must let this benchmark speak for itself as well.

---

## bartowski/Qwen3-30B-A3B-Q4_K_M

![qwen3-30b-sweep-pr370](https://github.com/user-attachments/assets/240bcdbb-a2ec-40c7-a401-90a21466853e)

I had not yet run Qwen3-30B-A3B fully offloaded on my local 3090TI 24GB VRAM rig on mainline before, so this is data I have not seen. I have a couple more benchmarks to repeat including my `mix-IQ3_K` quants as well as the hybrid CPU+GPU setup too on the remote thread ripper RTX A6000 to confirm given this PR is largely about TG performance.

A couple observations about this test case:

- I used `-fmoe` with both ik cases as it seems to improve performance over removing it still.
- I noticed the power draw on my GPU was higher for mainline than this PR.

#### Mainline btop
![mainline-btop-gpu](https://github.com/user-attachments/assets/0d42ded5-c083-4ca2-b0a5-da62f3b4eddf)

#### ik PR370 btop
![ik-btop-gpu](https://github.com/user-attachments/assets/e235efb9-abba-4af7-a5da-7ad91162854f)

---

👤 **ubergarm** commented the **2025-05-03** at **22:05:33**:<br>

## [ubergarm/Qwen3-30B-A3B-mix-IQ4_K](https://huggingface.co/ubergarm/Qwen3-30B-A3B-GGUF)

![qwen3-mix-iq4_k-sweep-pr370](https://github.com/user-attachments/assets/1d0b20cf-024f-4d4a-a4fb-4bcfc3115a66)

This is comparing a mix of mostly IQ5_K/IQ4_K layers between ik@main baseline and this ik@PR370 showing improved performance of *both* PP and TG for full GPU offload case.

<details>

<summary></summary>

## `ik_llama.cpp/main@ab7f694b`
```
cmake -B build -DGGML_CUDA=ON -DGGML_RPC=OFF -DGGML_BLAS=OFF
cmake --build build --config Release -j $(nproc)

CUDA_VISIBLE_DEVICE=0 \
./build/bin/llama-sweep-bench \
    --model /mnt/astrodata/llm/models/ubergarm/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-mix-IQ4_K.gguf \
    -fmoe \
    -fa \
    -ctk f16 -ctv f16 \
    -c 32768 \
    -ngl 99 \
    --threads 1

llama_model_loader: loaded meta data with 41 key-value pairs and 579 tensors from /mnt/astrodata/llm/models/ubergarm/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-mix-IQ4_K.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3 30B A3B
llama_model_loader: - kv   3:                           general.basename str              = Qwen3
llama_model_loader: - kv   4:                         general.size_label str              = 30B-A3B
llama_model_loader: - kv   5:                            general.license str              = apache-2.0
llama_model_loader: - kv   6:                       general.license.link str              = https://huggingface.co/Qwen/Qwen3-30B...
llama_model_loader: - kv   7:                   general.base_model.count u32              = 1
llama_model_loader: - kv   8:                  general.base_model.0.name str              = Qwen3 30B A3B Base
llama_model_loader: - kv   9:          general.base_model.0.organization str              = Qwen
llama_model_loader: - kv  10:              general.base_model.0.repo_url str              = https://huggingface.co/Qwen/Qwen3-30B...
llama_model_loader: - kv  11:                               general.tags arr[str,1]       = ["text-generation"]
llama_model_loader: - kv  12:                       qwen3moe.block_count u32              = 48
llama_model_loader: - kv  13:                    qwen3moe.context_length u32              = 40960
llama_model_loader: - kv  14:                  qwen3moe.embedding_length u32              = 2048
llama_model_loader: - kv  15:               qwen3moe.feed_forward_length u32              = 6144
llama_model_loader: - kv  16:              qwen3moe.attention.head_count u32              = 32
llama_model_loader: - kv  17:           qwen3moe.attention.head_count_kv u32              = 4
llama_model_loader: - kv  18:                    qwen3moe.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  19:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  20:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  21:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  22:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  23:                          general.file_type u32              = 140
llama_model_loader: - kv  24:                      qwen3moe.expert_count u32              = 128
llama_model_loader: - kv  25:        qwen3moe.expert_feed_forward_length u32              = 768
llama_model_loader: - kv  26:               general.quantization_version u32              = 2
llama_model_loader: - kv  27:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  28:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  29:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  30:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  31:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv  32:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  33:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  34:                tokenizer.ggml.bos_token_id u32              = 151643
llama_model_loader: - kv  35:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  36:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - kv  37:                      quantize.imatrix.file str              = /mnt/raid/models/ubergarm/Qwen3-30B-A...
llama_model_loader: - kv  38:                   quantize.imatrix.dataset str              = calibration_data_v5_rc.txt
llama_model_loader: - kv  39:             quantize.imatrix.entries_count i32              = 385
llama_model_loader: - kv  40:              quantize.imatrix.chunks_count i32              = 225
llama_model_loader: - type  f32:  241 tensors
llama_model_loader: - type q8_0:    6 tensors
llama_model_loader: - type iq4_k:   96 tensors
llama_model_loader: - type iq5_k:   48 tensors
llama_model_loader: - type iq6_k:  188 tensors
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
llm_load_print_meta: model ftype      = IQ4_K - 4.5 bpw
llm_load_print_meta: model params     = 30.532 B
llm_load_print_meta: model size       = 17.679 GiB (4.974 BPW)
llm_load_print_meta: repeating layers = 17.063 GiB (4.900 BPW, 29.910 B parameters)
llm_load_print_meta: general.name     = Qwen3 30B A3B
llm_load_print_meta: BOS token        = 151643 '<|endoftext|>'
llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
llm_load_print_meta: PAD token        = 151643 '<|endoftext|>'
llm_load_print_meta: LF token         = 148848 'ÄĬ'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp         = 768
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090 Ti, compute capability 8.6, VMM: yes
llm_load_tensors: ggml ctx size =    0.51 MiB
llm_load_tensors: offloading 48 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 49/49 layers to GPU
llm_load_tensors:        CPU buffer size =   315.30 MiB
llm_load_tensors:      CUDA0 buffer size = 17787.83 MiB
...................................................................................................
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
llama_kv_cache_init:      CUDA0 KV buffer size =  3072.00 MiB
llama_new_context_with_model: KV self size  = 3072.00 MiB, K (f16): 1536.00 MiB, V (f16): 1536.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.58 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =   304.75 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =    68.01 MiB
llama_new_context_with_model: graph nodes  = 1878
llama_new_context_with_model: graph splits = 2

main: n_kv_max = 32768, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = 99, n_threads = 1, n_threads_batch = 1
```

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    0.375 |  1364.01 |    1.205 |   106.26 |
|   512 |    128 |    512 |    0.310 |  1649.13 |    1.243 |   103.02 |
|   512 |    128 |   1024 |    0.315 |  1625.12 |    1.260 |   101.55 |
|   512 |    128 |   1536 |    0.317 |  1614.48 |    1.292 |    99.09 |
|   512 |    128 |   2048 |    0.331 |  1545.86 |    1.309 |    97.81 |
|   512 |    128 |   2560 |    0.324 |  1581.21 |    1.347 |    95.04 |
|   512 |    128 |   3072 |    0.334 |  1532.67 |    1.375 |    93.07 |
|   512 |    128 |   3584 |    0.334 |  1533.76 |    1.401 |    91.36 |
|   512 |    128 |   4096 |    0.344 |  1486.87 |    1.435 |    89.22 |
|   512 |    128 |   4608 |    0.346 |  1479.26 |    1.455 |    87.98 |
|   512 |    128 |   5120 |    0.351 |  1460.15 |    1.495 |    85.60 |
|   512 |    128 |   5632 |    0.353 |  1449.31 |    1.509 |    84.85 |
|   512 |    128 |   6144 |    0.359 |  1427.70 |    1.549 |    82.65 |
|   512 |    128 |   6656 |    0.365 |  1402.56 |    1.560 |    82.03 |
|   512 |    128 |   7168 |    0.375 |  1364.59 |    1.602 |    79.88 |
|   512 |    128 |   7680 |    0.374 |  1369.04 |    1.618 |    79.12 |
|   512 |    128 |   8192 |    0.386 |  1325.57 |    1.656 |    77.30 |
|   512 |    128 |   8704 |    0.387 |  1323.28 |    1.691 |    75.71 |
|   512 |    128 |   9216 |    0.393 |  1301.43 |    1.714 |    74.69 |
|   512 |    128 |   9728 |    0.397 |  1288.16 |    1.750 |    73.16 |
|   512 |    128 |  10240 |    0.399 |  1284.26 |    1.765 |    72.53 |
|   512 |    128 |  10752 |    0.411 |  1245.77 |    1.805 |    70.90 |
|   512 |    128 |  11264 |    0.411 |  1244.98 |    1.822 |    70.25 |
|   512 |    128 |  11776 |    0.419 |  1223.34 |    1.858 |    68.89 |
|   512 |    128 |  12288 |    0.419 |  1220.72 |    1.874 |    68.29 |
|   512 |    128 |  12800 |    0.427 |  1198.57 |    1.913 |    66.91 |
|   512 |    128 |  13312 |    0.432 |  1185.28 |    1.935 |    66.14 |
|   512 |    128 |  13824 |    0.437 |  1171.84 |    1.968 |    65.03 |
|   512 |    128 |  14336 |    0.438 |  1168.37 |    1.990 |    64.31 |
|   512 |    128 |  14848 |    0.448 |  1142.44 |    2.018 |    63.43 |
|   512 |    128 |  15360 |    0.451 |  1134.54 |    2.045 |    62.60 |
|   512 |    128 |  15872 |    0.457 |  1120.54 |    2.071 |    61.79 |
|   512 |    128 |  16384 |    0.461 |  1110.40 |    2.101 |    60.93 |
|   512 |    128 |  16896 |    0.467 |  1097.51 |    2.128 |    60.16 |
|   512 |    128 |  17408 |    0.475 |  1078.83 |    2.157 |    59.33 |
|   512 |    128 |  17920 |    0.479 |  1067.95 |    2.182 |    58.65 |
|   512 |    128 |  18432 |    0.488 |  1049.35 |    2.223 |    57.57 |
|   512 |    128 |  18944 |    0.487 |  1050.46 |    2.242 |    57.10 |
|   512 |    128 |  19456 |    0.497 |  1029.72 |    2.274 |    56.29 |
|   512 |    128 |  19968 |    0.501 |  1022.44 |    2.297 |    55.73 |
|   512 |    128 |  20480 |    0.499 |  1025.29 |    2.327 |    55.00 |
|   512 |    128 |  20992 |    0.506 |  1011.09 |    2.355 |    54.34 |
|   512 |    128 |  21504 |    0.517 |   990.59 |    2.382 |    53.74 |
|   512 |    128 |  22016 |    0.519 |   986.43 |    2.414 |    53.02 |
|   512 |    128 |  22528 |    0.528 |   968.85 |    2.440 |    52.45 |
|   512 |    128 |  23040 |    0.529 |   966.97 |    2.471 |    51.81 |
|   512 |    128 |  23552 |    0.534 |   958.13 |    2.495 |    51.30 |
|   512 |    128 |  24064 |    0.540 |   947.95 |    2.526 |    50.67 |
|   512 |    128 |  24576 |    0.549 |   933.39 |    2.569 |    49.83 |
|   512 |    128 |  25088 |    0.554 |   924.20 |    2.598 |    49.28 |
|   512 |    128 |  25600 |    0.556 |   920.25 |    2.628 |    48.71 |
|   512 |    128 |  26112 |    0.562 |   911.64 |    2.650 |    48.30 |
|   512 |    128 |  26624 |    0.566 |   904.68 |    2.682 |    47.72 |
|   512 |    128 |  27136 |    0.575 |   891.13 |    2.707 |    47.28 |
|   512 |    128 |  27648 |    0.577 |   887.14 |    2.737 |    46.77 |
|   512 |    128 |  28160 |    0.584 |   876.50 |    2.764 |    46.31 |
|   512 |    128 |  28672 |    0.593 |   863.88 |    2.796 |    45.79 |
|   512 |    128 |  29184 |    0.597 |   858.11 |    2.822 |    45.36 |
|   512 |    128 |  29696 |    0.599 |   855.36 |    2.847 |    44.96 |
|   512 |    128 |  30208 |    0.603 |   848.96 |    2.879 |    44.47 |
|   512 |    128 |  30720 |    0.609 |   840.31 |    2.906 |    44.05 |
|   512 |    128 |  31232 |    0.614 |   833.43 |    2.937 |    43.59 |
|   512 |    128 |  31744 |    0.617 |   830.35 |    2.964 |    43.19 |
|   512 |    128 |  32256 |    0.625 |   819.73 |    2.993 |    42.76 |

## `ik_llama.cpp/ik/fattn_mma@056f0818` PR370
```
cmake -B build -DGGML_CUDA=ON -DGGML_RPC=OFF -DGGML_BLAS=OFF
cmake --build build --config Release -j $(nproc)

CUDA_VISIBLE_DEVICE=0 \
./build/bin/llama-sweep-bench \
    --model /mnt/astrodata/llm/models/ubergarm/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-mix-IQ4_K.gguf \
    -fmoe \
    -fa \
    -ctk f16 -ctv f16 \
    -c 32768 \
    -ngl 99 \
    --threads 1

llama_model_loader: loaded meta data with 41 key-value pairs and 579 tensors from /mnt/astrodata/llm/models/ubergarm/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-mix-IQ4_K.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3 30B A3B
llama_model_loader: - kv   3:                           general.basename str              = Qwen3
llama_model_loader: - kv   4:                         general.size_label str              = 30B-A3B
llama_model_loader: - kv   5:                            general.license str              = apache-2.0
llama_model_loader: - kv   6:                       general.license.link str              = https://huggingface.co/Qwen/Qwen3-30B...
llama_model_loader: - kv   7:                   general.base_model.count u32              = 1
llama_model_loader: - kv   8:                  general.base_model.0.name str              = Qwen3 30B A3B Base
llama_model_loader: - kv   9:          general.base_model.0.organization str              = Qwen
llama_model_loader: - kv  10:              general.base_model.0.repo_url str              = https://huggingface.co/Qwen/Qwen3-30B...
llama_model_loader: - kv  11:                               general.tags arr[str,1]       = ["text-generation"]
llama_model_loader: - kv  12:                       qwen3moe.block_count u32              = 48
llama_model_loader: - kv  13:                    qwen3moe.context_length u32              = 40960
llama_model_loader: - kv  14:                  qwen3moe.embedding_length u32              = 2048
llama_model_loader: - kv  15:               qwen3moe.feed_forward_length u32              = 6144
llama_model_loader: - kv  16:              qwen3moe.attention.head_count u32              = 32
llama_model_loader: - kv  17:           qwen3moe.attention.head_count_kv u32              = 4
llama_model_loader: - kv  18:                    qwen3moe.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  19:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  20:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  21:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  22:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  23:                          general.file_type u32              = 140
llama_model_loader: - kv  24:                      qwen3moe.expert_count u32              = 128
llama_model_loader: - kv  25:        qwen3moe.expert_feed_forward_length u32              = 768
llama_model_loader: - kv  26:               general.quantization_version u32              = 2
llama_model_loader: - kv  27:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  28:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  29:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  30:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  31:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv  32:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  33:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  34:                tokenizer.ggml.bos_token_id u32              = 151643
llama_model_loader: - kv  35:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  36:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - kv  37:                      quantize.imatrix.file str              = /mnt/raid/models/ubergarm/Qwen3-30B-A...
llama_model_loader: - kv  38:                   quantize.imatrix.dataset str              = calibration_data_v5_rc.txt
llama_model_loader: - kv  39:             quantize.imatrix.entries_count i32              = 385
llama_model_loader: - kv  40:              quantize.imatrix.chunks_count i32              = 225
llama_model_loader: - type  f32:  241 tensors
llama_model_loader: - type q8_0:    6 tensors
llama_model_loader: - type iq4_k:   96 tensors
llama_model_loader: - type iq5_k:   48 tensors
llama_model_loader: - type iq6_k:  188 tensors
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
llm_load_print_meta: model ftype      = IQ4_K - 4.5 bpw
llm_load_print_meta: model params     = 30.532 B
llm_load_print_meta: model size       = 17.679 GiB (4.974 BPW)
llm_load_print_meta: repeating layers = 17.063 GiB (4.900 BPW, 29.910 B parameters)
llm_load_print_meta: general.name     = Qwen3 30B A3B
llm_load_print_meta: BOS token        = 151643 '<|endoftext|>'
llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
llm_load_print_meta: PAD token        = 151643 '<|endoftext|>'
llm_load_print_meta: LF token         = 148848 'ÄĬ'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp         = 768
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090 Ti, compute capability 8.6, VMM: yes
llm_load_tensors: ggml ctx size =    0.51 MiB
llm_load_tensors: offloading 48 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 49/49 layers to GPU
llm_load_tensors:        CPU buffer size =   315.30 MiB
llm_load_tensors:      CUDA0 buffer size = 17787.83 MiB
...................................................................................................
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
llama_kv_cache_init:      CUDA0 KV buffer size =  3072.00 MiB
llama_new_context_with_model: KV self size  = 3072.00 MiB, K (f16): 1536.00 MiB, V (f16): 1536.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.58 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =   304.75 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =    68.01 MiB
llama_new_context_with_model: graph nodes  = 1878
llama_new_context_with_model: graph splits = 2

main: n_kv_max = 32768, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = 99, n_threads = 1, n_threads_batch = 1
```

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    0.334 |  1531.88 |    1.235 |   103.64 |
|   512 |    128 |    512 |    0.303 |  1687.36 |    1.250 |   102.36 |
|   512 |    128 |   1024 |    0.309 |  1655.96 |    1.250 |   102.43 |
|   512 |    128 |   1536 |    0.310 |  1652.94 |    1.274 |   100.46 |
|   512 |    128 |   2048 |    0.323 |  1587.41 |    1.297 |    98.72 |
|   512 |    128 |   2560 |    0.313 |  1634.32 |    1.297 |    98.68 |
|   512 |    128 |   3072 |    0.322 |  1591.20 |    1.299 |    98.51 |
|   512 |    128 |   3584 |    0.320 |  1600.29 |    1.305 |    98.12 |
|   512 |    128 |   4096 |    0.326 |  1568.57 |    1.320 |    97.00 |
|   512 |    128 |   4608 |    0.327 |  1567.24 |    1.339 |    95.61 |
|   512 |    128 |   5120 |    0.329 |  1555.33 |    1.349 |    94.92 |
|   512 |    128 |   5632 |    0.331 |  1547.75 |    1.352 |    94.66 |
|   512 |    128 |   6144 |    0.334 |  1533.00 |    1.359 |    94.21 |
|   512 |    128 |   6656 |    0.338 |  1514.30 |    1.371 |    93.34 |
|   512 |    128 |   7168 |    0.346 |  1478.59 |    1.390 |    92.06 |
|   512 |    128 |   7680 |    0.344 |  1489.07 |    1.409 |    90.83 |
|   512 |    128 |   8192 |    0.353 |  1452.44 |    1.415 |    90.45 |
|   512 |    128 |   8704 |    0.351 |  1459.20 |    1.419 |    90.23 |
|   512 |    128 |   9216 |    0.355 |  1442.70 |    1.435 |    89.17 |
|   512 |    128 |   9728 |    0.356 |  1436.19 |    1.454 |    88.03 |
|   512 |    128 |  10240 |    0.358 |  1431.70 |    1.468 |    87.18 |
|   512 |    128 |  10752 |    0.368 |  1391.52 |    1.514 |    84.54 |
|   512 |    128 |  11264 |    0.366 |  1400.46 |    1.524 |    83.98 |
|   512 |    128 |  11776 |    0.371 |  1381.80 |    1.520 |    84.21 |
|   512 |    128 |  12288 |    0.370 |  1384.65 |    1.522 |    84.11 |
|   512 |    128 |  12800 |    0.376 |  1363.08 |    1.527 |    83.84 |
|   512 |    128 |  13312 |    0.377 |  1356.85 |    1.526 |    83.90 |
|   512 |    128 |  13824 |    0.380 |  1345.77 |    1.528 |    83.77 |
|   512 |    128 |  14336 |    0.380 |  1348.43 |    1.530 |    83.64 |
|   512 |    128 |  14848 |    0.387 |  1323.19 |    1.534 |    83.47 |
|   512 |    128 |  15360 |    0.389 |  1317.18 |    1.537 |    83.27 |
|   512 |    128 |  15872 |    0.393 |  1301.82 |    1.545 |    82.83 |
|   512 |    128 |  16384 |    0.395 |  1297.74 |    1.554 |    82.36 |
|   512 |    128 |  16896 |    0.398 |  1287.50 |    1.567 |    81.67 |
|   512 |    128 |  17408 |    0.404 |  1265.79 |    1.577 |    81.17 |
|   512 |    128 |  17920 |    0.406 |  1260.26 |    1.585 |    80.75 |
|   512 |    128 |  18432 |    0.414 |  1235.55 |    1.592 |    80.42 |
|   512 |    128 |  18944 |    0.411 |  1245.21 |    1.595 |    80.26 |
|   512 |    128 |  19456 |    0.418 |  1224.55 |    1.600 |    80.02 |
|   512 |    128 |  19968 |    0.421 |  1217.49 |    1.607 |    79.64 |
|   512 |    128 |  20480 |    0.418 |  1224.76 |    1.614 |    79.29 |
|   512 |    128 |  20992 |    0.422 |  1213.36 |    1.629 |    78.59 |
|   512 |    128 |  21504 |    0.430 |  1190.89 |    1.660 |    77.13 |
|   512 |    128 |  22016 |    0.431 |  1189.12 |    1.689 |    75.78 |
|   512 |    128 |  22528 |    0.438 |  1168.70 |    1.672 |    76.54 |
|   512 |    128 |  23040 |    0.436 |  1173.08 |    1.675 |    76.43 |
|   512 |    128 |  23552 |    0.439 |  1164.98 |    1.689 |    75.78 |
|   512 |    128 |  24064 |    0.442 |  1157.12 |    1.691 |    75.69 |
|   512 |    128 |  24576 |    0.447 |  1145.15 |    1.693 |    75.60 |
|   512 |    128 |  25088 |    0.450 |  1138.86 |    1.699 |    75.32 |
|   512 |    128 |  25600 |    0.450 |  1139.02 |    1.701 |    75.24 |
|   512 |    128 |  26112 |    0.453 |  1130.26 |    1.704 |    75.13 |
|   512 |    128 |  26624 |    0.455 |  1125.05 |    1.709 |    74.89 |
|   512 |    128 |  27136 |    0.462 |  1109.35 |    1.714 |    74.67 |
|   512 |    128 |  27648 |    0.463 |  1106.15 |    1.724 |    74.26 |
|   512 |    128 |  28160 |    0.467 |  1096.92 |    1.728 |    74.06 |
|   512 |    128 |  28672 |    0.473 |  1083.01 |    1.742 |    73.46 |
|   512 |    128 |  29184 |    0.475 |  1078.34 |    1.752 |    73.05 |
|   512 |    128 |  29696 |    0.475 |  1077.81 |    1.760 |    72.73 |
|   512 |    128 |  30208 |    0.477 |  1072.64 |    1.766 |    72.50 |
|   512 |    128 |  30720 |    0.481 |  1064.37 |    1.769 |    72.36 |
|   512 |    128 |  31232 |    0.484 |  1058.83 |    1.774 |    72.16 |
|   512 |    128 |  31744 |    0.484 |  1057.39 |    1.778 |    71.99 |
|   512 |    128 |  32256 |    0.490 |  1044.28 |    1.822 |    70.24 |

</details>

---

👤 **AesSedai** commented the **2025-05-03** at **22:57:58**:<br>

I've run the tests for 235B-A22B Q6 as well to compare. I used the Unsloth Q6 quant for both ik_llama.cpp and llama.cpp, the only arg difference in the calls is for ik_llama.cpp's support of `-fmoe -rtr`. Same offload the the rest otherwise.

<details>
<summary>ik_llama.cpp ik/fattn_mma</summary>

```
SHA 056f08182ab82f4bc8862c293c977f0207c0f17a

./build/bin/llama-sweep-bench -m /mnt/srv/slush/gguf/Qwen3-235B-A22B-128K-GGUF/Q6_K/Qwen3-235B-A22B-128K-Q6_K-00001-of-00004.gguf -c 16384 -t 48 -fa -rtr -fmoe -ctk q8_0 -ctv q8_0 -ngl 99 -ot "blk\.(0|1|2|3|4|5|6)\.ffn.*=CUDA0" -ot "blk\.(7|8|9|10|11|12|13)\.ffn.*=CUDA1" -ot "blk\.1[4-9]\.ffn.*=CPU" -ot "blk\.[2-9][0-9]\.ffn.*=CPU"
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 1: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
llama_model_loader: additional 3 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 47 key-value pairs and 1131 tensors from /mnt/srv/slush/gguf/Qwen3-235B-A22B-128K-GGUF/Q6_K/Qwen3-235B-A22B-128K-Q6_K-00001-of-00004.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3-235B-A22B-128K
llama_model_loader: - kv   3:                           general.finetune str              = 128k
llama_model_loader: - kv   4:                           general.basename str              = Qwen3-235B-A22B-128K
llama_model_loader: - kv   5:                       general.quantized_by str              = Unsloth
llama_model_loader: - kv   6:                         general.size_label str              = 235B-A22B
llama_model_loader: - kv   7:                            general.license str              = apache-2.0
llama_model_loader: - kv   8:                       general.license.link str              = https://huggingface.co/Qwen/Qwen3-235...
llama_model_loader: - kv   9:                           general.repo_url str              = https://huggingface.co/unsloth
llama_model_loader: - kv  10:                   general.base_model.count u32              = 1
llama_model_loader: - kv  11:                  general.base_model.0.name str              = Qwen3 235B A22B
llama_model_loader: - kv  12:          general.base_model.0.organization str              = Qwen
llama_model_loader: - kv  13:              general.base_model.0.repo_url str              = https://huggingface.co/Qwen/Qwen3-235...
llama_model_loader: - kv  14:                               general.tags arr[str,2]       = ["unsloth", "text-generation"]
llama_model_loader: - kv  15:                       qwen3moe.block_count u32              = 94
llama_model_loader: - kv  16:                    qwen3moe.context_length u32              = 131072
llama_model_loader: - kv  17:                  qwen3moe.embedding_length u32              = 4096
llama_model_loader: - kv  18:               qwen3moe.feed_forward_length u32              = 12288
llama_model_loader: - kv  19:              qwen3moe.attention.head_count u32              = 64
llama_model_loader: - kv  20:           qwen3moe.attention.head_count_kv u32              = 4
llama_model_loader: - kv  21:                    qwen3moe.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  22:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  23:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  24:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  25:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  26:                      qwen3moe.expert_count u32              = 128
llama_model_loader: - kv  27:        qwen3moe.expert_feed_forward_length u32              = 1536
llama_model_loader: - kv  28:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  29:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  30:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  31:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  32:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv  33:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  34:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  35:                tokenizer.ggml.bos_token_id u32              = 151643
llama_model_loader: - kv  36:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  37:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - kv  38:               general.quantization_version u32              = 2
llama_model_loader: - kv  39:                          general.file_type u32              = 18
llama_model_loader: - kv  40:                      quantize.imatrix.file str              = Qwen3-235B-A22B-128K-GGUF/imatrix_uns...
llama_model_loader: - kv  41:                   quantize.imatrix.dataset str              = unsloth_calibration_Qwen3-235B-A22B-1...
llama_model_loader: - kv  42:             quantize.imatrix.entries_count i32              = 752
llama_model_loader: - kv  43:              quantize.imatrix.chunks_count i32              = 46
llama_model_loader: - kv  44:                                   split.no u16              = 0
llama_model_loader: - kv  45:                        split.tensors.count i32              = 1131
llama_model_loader: - kv  46:                                split.count u16              = 4
llama_model_loader: - type  f32:  471 tensors
llama_model_loader: - type q6_K:  660 tensors
llm_load_vocab: special tokens cache size = 26
llm_load_vocab: token to piece cache size = 0.9311 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = qwen3moe
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 151936
llm_load_print_meta: n_merges         = 151387
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 131072
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_layer          = 94
llm_load_print_meta: n_head           = 64
llm_load_print_meta: n_head_kv        = 4
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 16
llm_load_print_meta: n_embd_k_gqa     = 512
llm_load_print_meta: n_embd_v_gqa     = 512
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 12288
llm_load_print_meta: n_expert         = 128
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 2
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 1000000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 131072
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = ?B
llm_load_print_meta: model ftype      = Q6_K
llm_load_print_meta: model params     = 235.094 B
llm_load_print_meta: model size       = 179.754 GiB (6.568 BPW) 
llm_load_print_meta: repeating layers = 178.803 GiB (6.568 BPW, 233.849 B parameters)
llm_load_print_meta: general.name     = Qwen3-235B-A22B-128K
llm_load_print_meta: BOS token        = 151643 '<|endoftext|>'
llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
llm_load_print_meta: PAD token        = 151643 '<|endoftext|>'
llm_load_print_meta: LF token         = 148848 'ÄĬ'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp         = 1536
llm_load_tensors: ggml ctx size =    1.49 MiB
Tensor blk.0.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.7.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.7.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.7.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.9.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.9.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.9.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.9.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.9.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.10.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.10.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.10.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.10.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.11.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.11.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.11.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.11.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.11.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.12.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.12.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.13.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.13.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.13.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.13.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.13.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.14.ffn_norm.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_norm.weight buffer type overriden to CPU
Tensor blk.15.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_norm.weight buffer type overriden to CPU
Tensor blk.16.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.16.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_norm.weight buffer type overriden to CPU
Tensor blk.17.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.17.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_norm.weight buffer type overriden to CPU
Tensor blk.18.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.18.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_norm.weight buffer type overriden to CPU
Tensor blk.19.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.19.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_norm.weight buffer type overriden to CPU
Tensor blk.20.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.20.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_norm.weight buffer type overriden to CPU
Tensor blk.21.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.21.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_norm.weight buffer type overriden to CPU
Tensor blk.22.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.22.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_norm.weight buffer type overriden to CPU
Tensor blk.23.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.23.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_norm.weight buffer type overriden to CPU
Tensor blk.24.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.24.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_norm.weight buffer type overriden to CPU
Tensor blk.25.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.25.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_norm.weight buffer type overriden to CPU
Tensor blk.26.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.26.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_norm.weight buffer type overriden to CPU
Tensor blk.27.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.27.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_norm.weight buffer type overriden to CPU
Tensor blk.28.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.28.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_norm.weight buffer type overriden to CPU
Tensor blk.29.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.29.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_norm.weight buffer type overriden to CPU
Tensor blk.30.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.30.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_norm.weight buffer type overriden to CPU
Tensor blk.31.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.31.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_norm.weight buffer type overriden to CPU
Tensor blk.32.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.32.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_norm.weight buffer type overriden to CPU
Tensor blk.33.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.33.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_norm.weight buffer type overriden to CPU
Tensor blk.34.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.34.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_norm.weight buffer type overriden to CPU
Tensor blk.35.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.35.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_norm.weight buffer type overriden to CPU
Tensor blk.36.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.36.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_norm.weight buffer type overriden to CPU
Tensor blk.37.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.37.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_norm.weight buffer type overriden to CPU
Tensor blk.38.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.38.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_norm.weight buffer type overriden to CPU
Tensor blk.39.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.39.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_norm.weight buffer type overriden to CPU
Tensor blk.40.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.40.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_norm.weight buffer type overriden to CPU
Tensor blk.41.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.41.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_norm.weight buffer type overriden to CPU
Tensor blk.42.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.42.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_norm.weight buffer type overriden to CPU
Tensor blk.43.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.43.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_norm.weight buffer type overriden to CPU
Tensor blk.44.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.44.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_norm.weight buffer type overriden to CPU
Tensor blk.45.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.45.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_norm.weight buffer type overriden to CPU
Tensor blk.46.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.46.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_norm.weight buffer type overriden to CPU
Tensor blk.47.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.47.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_norm.weight buffer type overriden to CPU
Tensor blk.48.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.48.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_norm.weight buffer type overriden to CPU
Tensor blk.49.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.49.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_norm.weight buffer type overriden to CPU
Tensor blk.50.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.50.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_norm.weight buffer type overriden to CPU
Tensor blk.51.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.51.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_norm.weight buffer type overriden to CPU
Tensor blk.52.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.52.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_norm.weight buffer type overriden to CPU
Tensor blk.53.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.53.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_norm.weight buffer type overriden to CPU
Tensor blk.54.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.54.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_norm.weight buffer type overriden to CPU
Tensor blk.55.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.55.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_norm.weight buffer type overriden to CPU
Tensor blk.56.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.56.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_norm.weight buffer type overriden to CPU
Tensor blk.57.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.57.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_norm.weight buffer type overriden to CPU
Tensor blk.58.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.58.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_norm.weight buffer type overriden to CPU
Tensor blk.59.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.59.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_norm.weight buffer type overriden to CPU
Tensor blk.60.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.60.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_norm.weight buffer type overriden to CPU
Tensor blk.61.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.61.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_norm.weight buffer type overriden to CPU
Tensor blk.62.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.62.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_norm.weight buffer type overriden to CPU
Tensor blk.63.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.63.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_norm.weight buffer type overriden to CPU
Tensor blk.64.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.64.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_norm.weight buffer type overriden to CPU
Tensor blk.65.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.65.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_norm.weight buffer type overriden to CPU
Tensor blk.66.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.66.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_norm.weight buffer type overriden to CPU
Tensor blk.67.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.67.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_norm.weight buffer type overriden to CPU
Tensor blk.68.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.68.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_norm.weight buffer type overriden to CPU
Tensor blk.69.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.69.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_norm.weight buffer type overriden to CPU
Tensor blk.70.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.70.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_norm.weight buffer type overriden to CPU
Tensor blk.71.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.71.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_norm.weight buffer type overriden to CPU
Tensor blk.72.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.72.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_norm.weight buffer type overriden to CPU
Tensor blk.73.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.73.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_norm.weight buffer type overriden to CPU
Tensor blk.74.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.74.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_norm.weight buffer type overriden to CPU
Tensor blk.75.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.75.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_norm.weight buffer type overriden to CPU
Tensor blk.76.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.76.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_norm.weight buffer type overriden to CPU
Tensor blk.77.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.77.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_norm.weight buffer type overriden to CPU
Tensor blk.78.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.78.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_norm.weight buffer type overriden to CPU
Tensor blk.79.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.79.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_norm.weight buffer type overriden to CPU
Tensor blk.80.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.80.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_norm.weight buffer type overriden to CPU
Tensor blk.81.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.81.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_norm.weight buffer type overriden to CPU
Tensor blk.82.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.82.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_norm.weight buffer type overriden to CPU
Tensor blk.83.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.83.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_norm.weight buffer type overriden to CPU
Tensor blk.84.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.84.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_norm.weight buffer type overriden to CPU
Tensor blk.85.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.85.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_norm.weight buffer type overriden to CPU
Tensor blk.86.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.86.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_norm.weight buffer type overriden to CPU
Tensor blk.87.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.87.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_norm.weight buffer type overriden to CPU
Tensor blk.88.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.88.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_norm.weight buffer type overriden to CPU
Tensor blk.89.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.89.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_norm.weight buffer type overriden to CPU
Tensor blk.90.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.90.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_norm.weight buffer type overriden to CPU
Tensor blk.91.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.91.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_norm.weight buffer type overriden to CPU
Tensor blk.92.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.92.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_norm.weight buffer type overriden to CPU
Tensor blk.93.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.93.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 94 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 95/95 layers to GPU
llm_load_tensors:        CPU buffer size = 151361.25 MiB
llm_load_tensors:  CUDA_Host buffer size =   486.86 MiB
llm_load_tensors:      CUDA0 buffer size = 15922.41 MiB
llm_load_tensors:      CUDA1 buffer size = 16297.68 MiB
....................................................................................................
============ Repacked 240 tensors
llama_new_context_with_model: n_ctx      = 16384
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:      CUDA0 KV buffer size =   816.02 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =   782.02 MiB
llama_new_context_with_model: KV self size  = 1598.00 MiB, K (q8_0):  799.00 MiB, V (q8_0):  799.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.58 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      CUDA0 compute buffer size =   144.00 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =   312.75 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   120.01 MiB
llama_new_context_with_model: graph nodes  = 3672
llama_new_context_with_model: graph splits = 336

main: n_kv_max = 16384, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = 99, n_threads = 48, n_threads_batch = 48

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    2.919 |   175.38 |    6.754 |    18.95 |
|   512 |    128 |    512 |    2.907 |   176.13 |    7.031 |    18.20 |
|   512 |    128 |   1024 |    2.917 |   175.54 |    7.088 |    18.06 |
|   512 |    128 |   1536 |    2.917 |   175.52 |    6.872 |    18.63 |
|   512 |    128 |   2048 |    2.934 |   174.52 |    6.948 |    18.42 |
|   512 |    128 |   2560 |    2.942 |   174.01 |    6.998 |    18.29 |
|   512 |    128 |   3072 |    2.956 |   173.20 |    7.087 |    18.06 |
|   512 |    128 |   3584 |    2.954 |   173.33 |    9.249 |    13.84 |
|   512 |    128 |   4096 |    2.997 |   170.84 |    9.920 |    12.90 |
|   512 |    128 |   4608 |    2.992 |   171.10 |    9.857 |    12.99 |
|   512 |    128 |   5120 |    3.026 |   169.23 |   10.022 |    12.77 |
|   512 |    128 |   5632 |    3.035 |   168.72 |   10.151 |    12.61 |
|   512 |    128 |   6144 |    3.047 |   168.01 |   10.021 |    12.77 |
|   512 |    128 |   6656 |    3.082 |   166.10 |   10.181 |    12.57 |
|   512 |    128 |   7168 |    3.088 |   165.81 |   10.061 |    12.72 |
|   512 |    128 |   7680 |    3.105 |   164.89 |   10.145 |    12.62 |
|   512 |    128 |   8192 |    3.108 |   164.73 |   10.201 |    12.55 |
|   512 |    128 |   8704 |    3.128 |   163.66 |   10.300 |    12.43 |
|   512 |    128 |   9216 |    3.133 |   163.40 |   10.353 |    12.36 |
|   512 |    128 |   9728 |    3.162 |   161.93 |   10.382 |    12.33 |
|   512 |    128 |  10240 |    3.192 |   160.41 |   10.486 |    12.21 |
|   512 |    128 |  10752 |    3.177 |   161.18 |   10.598 |    12.08 |
|   512 |    128 |  11264 |    3.209 |   159.53 |   10.580 |    12.10 |
|   512 |    128 |  11776 |    3.232 |   158.40 |   10.826 |    11.82 |
|   512 |    128 |  12288 |    3.233 |   158.35 |   10.663 |    12.00 |
|   512 |    128 |  12800 |    3.277 |   156.25 |   10.735 |    11.92 |
|   512 |    128 |  13312 |    3.290 |   155.64 |   10.874 |    11.77 |
|   512 |    128 |  13824 |    3.295 |   155.41 |   10.899 |    11.74 |
|   512 |    128 |  14336 |    3.300 |   155.16 |   11.041 |    11.59 |
|   512 |    128 |  14848 |    3.338 |   153.41 |   10.984 |    11.65 |
|   512 |    128 |  15360 |    3.338 |   153.39 |   10.999 |    11.64 |
|   512 |    128 |  15872 |    3.352 |   152.74 |   11.685 |    10.95 |
```

</details>


<details>
<summary>ik_llama.cpp main</summary>

```
SHA ab7f694b71497d216e1e7bad50bb4471feee7652

./build/bin/llama-sweep-bench -m /mnt/srv/slush/gguf/Qwen3-235B-A22B-128K-GGUF/Q6_K/Qwen3-235B-A22B-128K-Q6_K-00001-of-00004.gguf -c 16384 -t 48 -fa -rtr -fmoe -ctk q8_0 -ctv q8_0 -ngl 99 -ot "blk\.(0|1|2|3|4|5|6)\.ffn.*=CUDA0" -ot "blk\.(7|8|9|10|11|12|13)\.ffn.*=CUDA1" -ot "blk\.1[4-9]\.ffn.*=CPU" -ot "blk\.[2-9][0-9]\.ffn.*=CPU"
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 1: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
llama_model_loader: additional 3 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 47 key-value pairs and 1131 tensors from /mnt/srv/slush/gguf/Qwen3-235B-A22B-128K-GGUF/Q6_K/Qwen3-235B-A22B-128K-Q6_K-00001-of-00004.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3-235B-A22B-128K
llama_model_loader: - kv   3:                           general.finetune str              = 128k
llama_model_loader: - kv   4:                           general.basename str              = Qwen3-235B-A22B-128K
llama_model_loader: - kv   5:                       general.quantized_by str              = Unsloth
llama_model_loader: - kv   6:                         general.size_label str              = 235B-A22B
llama_model_loader: - kv   7:                            general.license str              = apache-2.0
llama_model_loader: - kv   8:                       general.license.link str              = https://huggingface.co/Qwen/Qwen3-235...
llama_model_loader: - kv   9:                           general.repo_url str              = https://huggingface.co/unsloth
llama_model_loader: - kv  10:                   general.base_model.count u32              = 1
llama_model_loader: - kv  11:                  general.base_model.0.name str              = Qwen3 235B A22B
llama_model_loader: - kv  12:          general.base_model.0.organization str              = Qwen
llama_model_loader: - kv  13:              general.base_model.0.repo_url str              = https://huggingface.co/Qwen/Qwen3-235...
llama_model_loader: - kv  14:                               general.tags arr[str,2]       = ["unsloth", "text-generation"]
llama_model_loader: - kv  15:                       qwen3moe.block_count u32              = 94
llama_model_loader: - kv  16:                    qwen3moe.context_length u32              = 131072
llama_model_loader: - kv  17:                  qwen3moe.embedding_length u32              = 4096
llama_model_loader: - kv  18:               qwen3moe.feed_forward_length u32              = 12288
llama_model_loader: - kv  19:              qwen3moe.attention.head_count u32              = 64
llama_model_loader: - kv  20:           qwen3moe.attention.head_count_kv u32              = 4
llama_model_loader: - kv  21:                    qwen3moe.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  22:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  23:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  24:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  25:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  26:                      qwen3moe.expert_count u32              = 128
llama_model_loader: - kv  27:        qwen3moe.expert_feed_forward_length u32              = 1536
llama_model_loader: - kv  28:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  29:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  30:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  31:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  32:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv  33:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  34:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  35:                tokenizer.ggml.bos_token_id u32              = 151643
llama_model_loader: - kv  36:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  37:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - kv  38:               general.quantization_version u32              = 2
llama_model_loader: - kv  39:                          general.file_type u32              = 18
llama_model_loader: - kv  40:                      quantize.imatrix.file str              = Qwen3-235B-A22B-128K-GGUF/imatrix_uns...
llama_model_loader: - kv  41:                   quantize.imatrix.dataset str              = unsloth_calibration_Qwen3-235B-A22B-1...
llama_model_loader: - kv  42:             quantize.imatrix.entries_count i32              = 752
llama_model_loader: - kv  43:              quantize.imatrix.chunks_count i32              = 46
llama_model_loader: - kv  44:                                   split.no u16              = 0
llama_model_loader: - kv  45:                        split.tensors.count i32              = 1131
llama_model_loader: - kv  46:                                split.count u16              = 4
llama_model_loader: - type  f32:  471 tensors
llama_model_loader: - type q6_K:  660 tensors
llm_load_vocab: special tokens cache size = 26
llm_load_vocab: token to piece cache size = 0.9311 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = qwen3moe
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 151936
llm_load_print_meta: n_merges         = 151387
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 131072
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_layer          = 94
llm_load_print_meta: n_head           = 64
llm_load_print_meta: n_head_kv        = 4
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 16
llm_load_print_meta: n_embd_k_gqa     = 512
llm_load_print_meta: n_embd_v_gqa     = 512
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 12288
llm_load_print_meta: n_expert         = 128
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 2
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 1000000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 131072
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = ?B
llm_load_print_meta: model ftype      = Q6_K
llm_load_print_meta: model params     = 235.094 B
llm_load_print_meta: model size       = 179.754 GiB (6.568 BPW) 
llm_load_print_meta: repeating layers = 178.803 GiB (6.568 BPW, 233.849 B parameters)
llm_load_print_meta: general.name     = Qwen3-235B-A22B-128K
llm_load_print_meta: BOS token        = 151643 '<|endoftext|>'
llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
llm_load_print_meta: PAD token        = 151643 '<|endoftext|>'
llm_load_print_meta: LF token         = 148848 'ÄĬ'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp         = 1536
llm_load_tensors: ggml ctx size =    1.49 MiB
Tensor blk.0.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.7.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.7.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.7.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.9.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.9.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.9.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.9.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.9.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.10.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.10.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.10.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.10.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.11.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.11.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.11.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.11.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.11.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.12.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.12.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.13.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.13.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.13.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.13.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.13.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.14.ffn_norm.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_norm.weight buffer type overriden to CPU
Tensor blk.15.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_norm.weight buffer type overriden to CPU
Tensor blk.16.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.16.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_norm.weight buffer type overriden to CPU
Tensor blk.17.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.17.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_norm.weight buffer type overriden to CPU
Tensor blk.18.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.18.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_norm.weight buffer type overriden to CPU
Tensor blk.19.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.19.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_norm.weight buffer type overriden to CPU
Tensor blk.20.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.20.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_norm.weight buffer type overriden to CPU
Tensor blk.21.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.21.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_norm.weight buffer type overriden to CPU
Tensor blk.22.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.22.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_norm.weight buffer type overriden to CPU
Tensor blk.23.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.23.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_norm.weight buffer type overriden to CPU
Tensor blk.24.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.24.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_norm.weight buffer type overriden to CPU
Tensor blk.25.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.25.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_norm.weight buffer type overriden to CPU
Tensor blk.26.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.26.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_norm.weight buffer type overriden to CPU
Tensor blk.27.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.27.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_norm.weight buffer type overriden to CPU
Tensor blk.28.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.28.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_norm.weight buffer type overriden to CPU
Tensor blk.29.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.29.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_norm.weight buffer type overriden to CPU
Tensor blk.30.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.30.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_norm.weight buffer type overriden to CPU
Tensor blk.31.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.31.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_norm.weight buffer type overriden to CPU
Tensor blk.32.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.32.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_norm.weight buffer type overriden to CPU
Tensor blk.33.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.33.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_norm.weight buffer type overriden to CPU
Tensor blk.34.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.34.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_norm.weight buffer type overriden to CPU
Tensor blk.35.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.35.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_norm.weight buffer type overriden to CPU
Tensor blk.36.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.36.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_norm.weight buffer type overriden to CPU
Tensor blk.37.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.37.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_norm.weight buffer type overriden to CPU
Tensor blk.38.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.38.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_norm.weight buffer type overriden to CPU
Tensor blk.39.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.39.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_norm.weight buffer type overriden to CPU
Tensor blk.40.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.40.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_norm.weight buffer type overriden to CPU
Tensor blk.41.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.41.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_norm.weight buffer type overriden to CPU
Tensor blk.42.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.42.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_norm.weight buffer type overriden to CPU
Tensor blk.43.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.43.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_norm.weight buffer type overriden to CPU
Tensor blk.44.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.44.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_norm.weight buffer type overriden to CPU
Tensor blk.45.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.45.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_norm.weight buffer type overriden to CPU
Tensor blk.46.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.46.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_norm.weight buffer type overriden to CPU
Tensor blk.47.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.47.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_norm.weight buffer type overriden to CPU
Tensor blk.48.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.48.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_norm.weight buffer type overriden to CPU
Tensor blk.49.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.49.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_norm.weight buffer type overriden to CPU
Tensor blk.50.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.50.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_norm.weight buffer type overriden to CPU
Tensor blk.51.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.51.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_norm.weight buffer type overriden to CPU
Tensor blk.52.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.52.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_norm.weight buffer type overriden to CPU
Tensor blk.53.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.53.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_norm.weight buffer type overriden to CPU
Tensor blk.54.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.54.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_norm.weight buffer type overriden to CPU
Tensor blk.55.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.55.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_norm.weight buffer type overriden to CPU
Tensor blk.56.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.56.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_norm.weight buffer type overriden to CPU
Tensor blk.57.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.57.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_norm.weight buffer type overriden to CPU
Tensor blk.58.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.58.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_norm.weight buffer type overriden to CPU
Tensor blk.59.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.59.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_norm.weight buffer type overriden to CPU
Tensor blk.60.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.60.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_norm.weight buffer type overriden to CPU
Tensor blk.61.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.61.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_norm.weight buffer type overriden to CPU
Tensor blk.62.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.62.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_norm.weight buffer type overriden to CPU
Tensor blk.63.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.63.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_norm.weight buffer type overriden to CPU
Tensor blk.64.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.64.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_norm.weight buffer type overriden to CPU
Tensor blk.65.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.65.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_norm.weight buffer type overriden to CPU
Tensor blk.66.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.66.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_norm.weight buffer type overriden to CPU
Tensor blk.67.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.67.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_norm.weight buffer type overriden to CPU
Tensor blk.68.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.68.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_norm.weight buffer type overriden to CPU
Tensor blk.69.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.69.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_norm.weight buffer type overriden to CPU
Tensor blk.70.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.70.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_norm.weight buffer type overriden to CPU
Tensor blk.71.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.71.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_norm.weight buffer type overriden to CPU
Tensor blk.72.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.72.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_norm.weight buffer type overriden to CPU
Tensor blk.73.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.73.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_norm.weight buffer type overriden to CPU
Tensor blk.74.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.74.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_norm.weight buffer type overriden to CPU
Tensor blk.75.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.75.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_norm.weight buffer type overriden to CPU
Tensor blk.76.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.76.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_norm.weight buffer type overriden to CPU
Tensor blk.77.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.77.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_norm.weight buffer type overriden to CPU
Tensor blk.78.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.78.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_norm.weight buffer type overriden to CPU
Tensor blk.79.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.79.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_norm.weight buffer type overriden to CPU
Tensor blk.80.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.80.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_norm.weight buffer type overriden to CPU
Tensor blk.81.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.81.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_norm.weight buffer type overriden to CPU
Tensor blk.82.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.82.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_norm.weight buffer type overriden to CPU
Tensor blk.83.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.83.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_norm.weight buffer type overriden to CPU
Tensor blk.84.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.84.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_norm.weight buffer type overriden to CPU
Tensor blk.85.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.85.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_norm.weight buffer type overriden to CPU
Tensor blk.86.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.86.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_norm.weight buffer type overriden to CPU
Tensor blk.87.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.87.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_norm.weight buffer type overriden to CPU
Tensor blk.88.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.88.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_norm.weight buffer type overriden to CPU
Tensor blk.89.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.89.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_norm.weight buffer type overriden to CPU
Tensor blk.90.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.90.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_norm.weight buffer type overriden to CPU
Tensor blk.91.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.91.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_norm.weight buffer type overriden to CPU
Tensor blk.92.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.92.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_norm.weight buffer type overriden to CPU
Tensor blk.93.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.93.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 94 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 95/95 layers to GPU
llm_load_tensors:        CPU buffer size = 151361.25 MiB
llm_load_tensors:  CUDA_Host buffer size =   486.86 MiB
llm_load_tensors:      CUDA0 buffer size = 15922.41 MiB
llm_load_tensors:      CUDA1 buffer size = 16297.68 MiB
....................................................................................................
============ Repacked 240 tensors
llama_new_context_with_model: n_ctx      = 16384
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:      CUDA0 KV buffer size =   816.02 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =   782.02 MiB
llama_new_context_with_model: KV self size  = 1598.00 MiB, K (q8_0):  799.00 MiB, V (q8_0):  799.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.58 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      CUDA0 compute buffer size =   144.00 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =   312.75 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   120.01 MiB
llama_new_context_with_model: graph nodes  = 3672
llama_new_context_with_model: graph splits = 336

main: n_kv_max = 16384, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = 99, n_threads = 48, n_threads_batch = 48

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    2.945 |   173.84 |    9.407 |    13.61 |
|   512 |    128 |    512 |    2.942 |   174.04 |    9.773 |    13.10 |
|   512 |    128 |   1024 |    2.957 |   173.15 |    9.950 |    12.86 |
|   512 |    128 |   1536 |    2.986 |   171.48 |    9.968 |    12.84 |
|   512 |    128 |   2048 |    2.993 |   171.09 |   10.211 |    12.54 |
|   512 |    128 |   2560 |    3.029 |   169.05 |   10.325 |    12.40 |
|   512 |    128 |   3072 |    3.036 |   168.65 |   10.607 |    12.07 |
|   512 |    128 |   3584 |    3.054 |   167.67 |   10.799 |    11.85 |
|   512 |    128 |   4096 |    3.094 |   165.46 |   10.940 |    11.70 |
|   512 |    128 |   4608 |    3.117 |   164.25 |   11.128 |    11.50 |
|   512 |    128 |   5120 |    3.150 |   162.55 |   11.280 |    11.35 |
|   512 |    128 |   5632 |    3.172 |   161.40 |   11.531 |    11.10 |
|   512 |    128 |   6144 |    3.245 |   157.80 |   11.793 |    10.85 |
|   512 |    128 |   6656 |    3.233 |   158.38 |   11.908 |    10.75 |
|   512 |    128 |   7168 |    3.260 |   157.08 |   12.065 |    10.61 |
|   512 |    128 |   7680 |    3.291 |   155.56 |   12.248 |    10.45 |
|   512 |    128 |   8192 |    3.327 |   153.87 |   12.597 |    10.16 |
|   512 |    128 |   8704 |    3.365 |   152.15 |   12.555 |    10.19 |
|   512 |    128 |   9216 |    3.407 |   150.26 |   12.851 |     9.96 |
|   512 |    128 |   9728 |    3.427 |   149.39 |   12.987 |     9.86 |
|   512 |    128 |  10240 |    3.413 |   150.03 |   13.295 |     9.63 |
|   512 |    128 |  10752 |    3.460 |   147.99 |   13.415 |     9.54 |
|   512 |    128 |  11264 |    3.470 |   147.54 |   13.561 |     9.44 |
|   512 |    128 |  11776 |    3.517 |   145.57 |   13.824 |     9.26 |
|   512 |    128 |  12288 |    3.536 |   144.82 |   14.081 |     9.09 |
|   512 |    128 |  12800 |    3.558 |   143.91 |   14.130 |     9.06 |
|   512 |    128 |  13312 |    3.566 |   143.56 |   14.339 |     8.93 |
|   512 |    128 |  13824 |    3.551 |   144.19 |   14.535 |     8.81 |
|   512 |    128 |  14336 |    3.593 |   142.49 |   14.832 |     8.63 |
|   512 |    128 |  14848 |    3.602 |   142.12 |   14.890 |     8.60 |
|   512 |    128 |  15360 |    3.591 |   142.60 |   15.167 |     8.44 |
|   512 |    128 |  15872 |    3.632 |   140.98 |   15.235 |     8.40 |
```

</details>

<details>
<summary>llama.cpp master</summary>

```
SHA 36667c8edcded08063ed51c7d57e9e086bbfc903

./build/bin/llama-sweep-bench -m /mnt/srv/slush/gguf/Qwen3-235B-A22B-128K-GGUF/Q6_K/Qwen3-235B-A22B-128K-Q6_K-00001-of-00004.gguf -c 16384 -t 48 -fa -ctk q8_0 -ctv q8_0 -ngl 99 -ot "blk\.(0|1|2|3|4|5|6)\.ffn.*=CUDA0" -ot "blk\.(7|8|9|10|11|12|13)\.ffn.*=CUDA1" -ot "blk\.1[4-9]\.ffn.*=CPU" -ot "blk\.[2-9][0-9]\.ffn.*=CPU" 
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 1: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
build: 5271 (36667c8e) with cc (GCC) 14.2.1 20250110 (Red Hat 14.2.1-7) for x86_64-redhat-linux
llama_model_load_from_file_impl: using device CUDA0 (NVIDIA GeForce RTX 3090) - 23871 MiB free
llama_model_load_from_file_impl: using device CUDA1 (NVIDIA GeForce RTX 3090) - 23871 MiB free
llama_model_loader: additional 3 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 47 key-value pairs and 1131 tensors from /mnt/srv/slush/gguf/Qwen3-235B-A22B-128K-GGUF/Q6_K/Qwen3-235B-A22B-128K-Q6_K-00001-of-00004.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3-235B-A22B-128K
llama_model_loader: - kv   3:                           general.finetune str              = 128k
llama_model_loader: - kv   4:                           general.basename str              = Qwen3-235B-A22B-128K
llama_model_loader: - kv   5:                       general.quantized_by str              = Unsloth
llama_model_loader: - kv   6:                         general.size_label str              = 235B-A22B
llama_model_loader: - kv   7:                            general.license str              = apache-2.0
llama_model_loader: - kv   8:                       general.license.link str              = https://huggingface.co/Qwen/Qwen3-235...
llama_model_loader: - kv   9:                           general.repo_url str              = https://huggingface.co/unsloth
llama_model_loader: - kv  10:                   general.base_model.count u32              = 1
llama_model_loader: - kv  11:                  general.base_model.0.name str              = Qwen3 235B A22B
llama_model_loader: - kv  12:          general.base_model.0.organization str              = Qwen
llama_model_loader: - kv  13:              general.base_model.0.repo_url str              = https://huggingface.co/Qwen/Qwen3-235...
llama_model_loader: - kv  14:                               general.tags arr[str,2]       = ["unsloth", "text-generation"]
llama_model_loader: - kv  15:                       qwen3moe.block_count u32              = 94
llama_model_loader: - kv  16:                    qwen3moe.context_length u32              = 131072
llama_model_loader: - kv  17:                  qwen3moe.embedding_length u32              = 4096
llama_model_loader: - kv  18:               qwen3moe.feed_forward_length u32              = 12288
llama_model_loader: - kv  19:              qwen3moe.attention.head_count u32              = 64
llama_model_loader: - kv  20:           qwen3moe.attention.head_count_kv u32              = 4
llama_model_loader: - kv  21:                    qwen3moe.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  22:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  23:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  24:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  25:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  26:                      qwen3moe.expert_count u32              = 128
llama_model_loader: - kv  27:        qwen3moe.expert_feed_forward_length u32              = 1536
llama_model_loader: - kv  28:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  29:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  30:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  31:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  32:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv  33:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  34:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  35:                tokenizer.ggml.bos_token_id u32              = 151643
llama_model_loader: - kv  36:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  37:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - kv  38:               general.quantization_version u32              = 2
llama_model_loader: - kv  39:                          general.file_type u32              = 18
llama_model_loader: - kv  40:                      quantize.imatrix.file str              = Qwen3-235B-A22B-128K-GGUF/imatrix_uns...
llama_model_loader: - kv  41:                   quantize.imatrix.dataset str              = unsloth_calibration_Qwen3-235B-A22B-1...
llama_model_loader: - kv  42:             quantize.imatrix.entries_count i32              = 752
llama_model_loader: - kv  43:              quantize.imatrix.chunks_count i32              = 46
llama_model_loader: - kv  44:                                   split.no u16              = 0
llama_model_loader: - kv  45:                        split.tensors.count i32              = 1131
llama_model_loader: - kv  46:                                split.count u16              = 4
llama_model_loader: - type  f32:  471 tensors
llama_model_loader: - type q6_K:  660 tensors
print_info: file format = GGUF V3 (latest)
print_info: file type   = Q6_K
print_info: file size   = 179.75 GiB (6.57 BPW) 
load: special tokens cache size = 26
load: token to piece cache size = 0.9311 MB
print_info: arch             = qwen3moe
print_info: vocab_only       = 0
print_info: n_ctx_train      = 131072
print_info: n_embd           = 4096
print_info: n_layer          = 94
print_info: n_head           = 64
print_info: n_head_kv        = 4
print_info: n_rot            = 128
print_info: n_swa            = 0
print_info: n_swa_pattern    = 1
print_info: n_embd_head_k    = 128
print_info: n_embd_head_v    = 128
print_info: n_gqa            = 16
print_info: n_embd_k_gqa     = 512
print_info: n_embd_v_gqa     = 512
print_info: f_norm_eps       = 0.0e+00
print_info: f_norm_rms_eps   = 1.0e-06
print_info: f_clamp_kqv      = 0.0e+00
print_info: f_max_alibi_bias = 0.0e+00
print_info: f_logit_scale    = 0.0e+00
print_info: f_attn_scale     = 0.0e+00
print_info: n_ff             = 12288
print_info: n_expert         = 128
print_info: n_expert_used    = 8
print_info: causal attn      = 1
print_info: pooling type     = 0
print_info: rope type        = 2
print_info: rope scaling     = linear
print_info: freq_base_train  = 1000000.0
print_info: freq_scale_train = 1
print_info: n_ctx_orig_yarn  = 131072
print_info: rope_finetuned   = unknown
print_info: ssm_d_conv       = 0
print_info: ssm_d_inner      = 0
print_info: ssm_d_state      = 0
print_info: ssm_dt_rank      = 0
print_info: ssm_dt_b_c_rms   = 0
print_info: model type       = 235B.A22B
print_info: model params     = 235.09 B
print_info: general.name     = Qwen3-235B-A22B-128K
print_info: n_ff_exp         = 1536
print_info: vocab type       = BPE
print_info: n_vocab          = 151936
print_info: n_merges         = 151387
print_info: BOS token        = 151643 '<|endoftext|>'
print_info: EOS token        = 151645 '<|im_end|>'
print_info: EOT token        = 151645 '<|im_end|>'
print_info: PAD token        = 151643 '<|endoftext|>'
print_info: LF token         = 198 'Ċ'
print_info: FIM PRE token    = 151659 '<|fim_prefix|>'
print_info: FIM SUF token    = 151661 '<|fim_suffix|>'
print_info: FIM MID token    = 151660 '<|fim_middle|>'
print_info: FIM PAD token    = 151662 '<|fim_pad|>'
print_info: FIM REP token    = 151663 '<|repo_name|>'
print_info: FIM SEP token    = 151664 '<|file_sep|>'
print_info: EOG token        = 151643 '<|endoftext|>'
print_info: EOG token        = 151645 '<|im_end|>'
print_info: EOG token        = 151662 '<|fim_pad|>'
print_info: EOG token        = 151663 '<|repo_name|>'
print_info: EOG token        = 151664 '<|file_sep|>'
print_info: max token length = 256
load_tensors: loading model tensors, this can take a while... (mmap = true)
load_tensors: offloading 94 repeating layers to GPU
load_tensors: offloading output layer to GPU
load_tensors: offloaded 95/95 layers to GPU
load_tensors:   CPU_Mapped model buffer size = 46604.38 MiB
load_tensors:   CPU_Mapped model buffer size = 47377.52 MiB
load_tensors:   CPU_Mapped model buffer size = 47377.52 MiB
load_tensors:   CPU_Mapped model buffer size = 42166.10 MiB
load_tensors:        CUDA0 model buffer size = 15922.41 MiB
load_tensors:        CUDA1 model buffer size = 16297.68 MiB
....................................................................................................
llama_context: constructing llama_context
llama_context: n_seq_max     = 1
llama_context: n_ctx         = 16384
llama_context: n_ctx_per_seq = 16384
llama_context: n_batch       = 2048
llama_context: n_ubatch      = 512
llama_context: causal_attn   = 1
llama_context: flash_attn    = 1
llama_context: freq_base     = 1000000.0
llama_context: freq_scale    = 1
llama_context: n_ctx_per_seq (16384) < n_ctx_train (131072) -- the full capacity of the model will not be utilized
llama_context:  CUDA_Host  output buffer size =     0.58 MiB
llama_kv_cache_unified: kv_size = 16384, type_k = 'q8_0', type_v = 'q8_0', n_layer = 94, can_shift = 1, padding = 256
llama_kv_cache_unified:      CUDA0 KV buffer size =   816.00 MiB
llama_kv_cache_unified:      CUDA1 KV buffer size =   782.00 MiB
llama_kv_cache_unified: KV self size  = 1598.00 MiB, K (q8_0):  799.00 MiB, V (q8_0):  799.00 MiB
llama_context:      CUDA0 compute buffer size =   774.00 MiB
llama_context:      CUDA1 compute buffer size =   304.75 MiB
llama_context:  CUDA_Host compute buffer size =    40.01 MiB
llama_context: graph nodes  = 5741
llama_context: graph splits = 463 (with bs=512), 176 (with bs=1)

main: n_kv_max = 16384, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = 99, n_threads = 48, n_threads_batch = 48

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    7.331 |    69.84 |   10.382 |    12.33 |
|   512 |    128 |    512 |    7.256 |    70.57 |   10.500 |    12.19 |
|   512 |    128 |   1024 |    7.276 |    70.37 |   10.564 |    12.12 |
|   512 |    128 |   1536 |    7.289 |    70.24 |   10.582 |    12.10 |
|   512 |    128 |   2048 |    7.295 |    70.18 |   10.571 |    12.11 |
|   512 |    128 |   2560 |    7.305 |    70.09 |   10.724 |    11.94 |
|   512 |    128 |   3072 |    7.317 |    69.98 |   11.011 |    11.62 |
|   512 |    128 |   3584 |    7.321 |    69.94 |   10.878 |    11.77 |
|   512 |    128 |   4096 |    7.343 |    69.72 |   11.094 |    11.54 |
|   512 |    128 |   4608 |    7.347 |    69.69 |   11.332 |    11.30 |
|   512 |    128 |   5120 |    7.365 |    69.51 |   11.439 |    11.19 |
|   512 |    128 |   5632 |    7.379 |    69.38 |   11.833 |    10.82 |
|   512 |    128 |   6144 |    7.383 |    69.35 |   11.561 |    11.07 |
|   512 |    128 |   6656 |    7.397 |    69.22 |   11.750 |    10.89 |
|   512 |    128 |   7168 |    7.417 |    69.03 |   11.963 |    10.70 |
|   512 |    128 |   7680 |    7.422 |    68.99 |   11.992 |    10.67 |
|   512 |    128 |   8192 |    7.446 |    68.76 |   12.188 |    10.50 |
|   512 |    128 |   8704 |    7.448 |    68.75 |   12.335 |    10.38 |
|   512 |    128 |   9216 |    7.465 |    68.59 |   12.618 |    10.14 |
|   512 |    128 |   9728 |    7.470 |    68.54 |   12.410 |    10.31 |
|   512 |    128 |  10240 |    7.480 |    68.44 |   12.631 |    10.13 |
|   512 |    128 |  10752 |    7.499 |    68.27 |   12.799 |    10.00 |
|   512 |    128 |  11264 |    7.511 |    68.17 |   12.992 |     9.85 |
|   512 |    128 |  11776 |    7.525 |    68.04 |   13.076 |     9.79 |
|   512 |    128 |  12288 |    7.541 |    67.90 |   13.154 |     9.73 |
|   512 |    128 |  12800 |    7.538 |    67.93 |   13.472 |     9.50 |
|   512 |    128 |  13312 |    7.546 |    67.85 |   13.388 |     9.56 |
|   512 |    128 |  13824 |    7.579 |    67.55 |   13.573 |     9.43 |
|   512 |    128 |  14336 |    7.577 |    67.57 |   13.870 |     9.23 |
|   512 |    128 |  14848 |    7.587 |    67.49 |   13.735 |     9.32 |
|   512 |    128 |  15360 |    7.595 |    67.42 |   13.969 |     9.16 |
|   512 |    128 |  15872 |    7.606 |    67.32 |   14.183 |     9.02 |
```

</details>

![sweep](https://github.com/user-attachments/assets/a5d3a5b0-791e-415e-9dd0-77327a6d9e4d)

---

👤 **ubergarm** commented the **2025-05-03** at **23:17:07**:<br>

@AesSedai *very nice*! Cool to see you are getting some uplift in PP as well and more linear fall-off for TG. I'm running that quant's little brother on my local rig in hybrid CPU+GPU inference in this test for comparison, but no mainline comparison as its the `-mix-IQ3_K`. 

Hope to finally get three runs of the hybrid CPU+GPU of the full Q8_0 across both forks before the night it out! If i have any juice left in me I might revisit earlier runs to add in `-ctk q8_0 -ctv q8_0` to see if any uplift for fully offloaded quantized kv-cache.

## [ubergarm/Qwen3-235B-A22B-mix-IQ3_K](https://huggingface.co/ubergarm/Qwen3-235B-A22B-GGUF)

![qwen3-235b-mix-iq3_k-sweep-pr370](https://github.com/user-attachments/assets/5afcc1f7-5f52-4c7f-82f0-b9568122c148)

<details>

<summary>👈 Logs</summary>

## `ik_llama.cpp/main@ab7f694b`
```
cmake -B build -DGGML_CUDA=ON -DGGML_RPC=OFF -DGGML_BLAS=OFF
cmake --build build --config Release -j $(nproc)

CUDA_VISIBLE_DEVICES="0" \
./build/bin/llama-sweep-bench \
  --model /mnt/ai/models/ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf \
  -fa \
  -ctk q8_0 -ctv q8_0 \
  -c 32768 \
  -fmoe \
  -amb 512 \
  -rtr \
  -ot blk\.1[2-9]\.ffn.*=CPU \
  -ot blk\.[2-8][0-9]\.ffn.*=CPU \
  -ot blk\.9[0-3]\.ffn.*=CPU \
  -ngl 99 \
  --threads 16

ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090 Ti, compute capability 8.6, VMM: yes
llama_model_loader: additional 2 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 40 key-value pairs and 1131 tensors from /mnt/ai/models/ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3 235B A22B
llama_model_loader: - kv   3:                           general.basename str              = Qwen3
llama_model_loader: - kv   4:                         general.size_label str              = 235B-A22B
llama_model_loader: - kv   5:                            general.license str              = apache-2.0
llama_model_loader: - kv   6:                       general.license.link str              = https://huggingface.co/Qwen/Qwen3-235...
llama_model_loader: - kv   7:                               general.tags arr[str,1]       = ["text-generation"]
llama_model_loader: - kv   8:                       qwen3moe.block_count u32              = 94
llama_model_loader: - kv   9:                    qwen3moe.context_length u32              = 40960
llama_model_loader: - kv  10:                  qwen3moe.embedding_length u32              = 4096
llama_model_loader: - kv  11:               qwen3moe.feed_forward_length u32              = 12288
llama_model_loader: - kv  12:              qwen3moe.attention.head_count u32              = 64
llama_model_loader: - kv  13:           qwen3moe.attention.head_count_kv u32              = 4
llama_model_loader: - kv  14:                    qwen3moe.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  15:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  16:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  17:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  18:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  19:                          general.file_type u32              = 139
llama_model_loader: - kv  20:                      qwen3moe.expert_count u32              = 128
llama_model_loader: - kv  21:        qwen3moe.expert_feed_forward_length u32              = 1536
llama_model_loader: - kv  22:               general.quantization_version u32              = 2
llama_model_loader: - kv  23:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  24:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  25:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  26:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  27:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv  28:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  29:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  30:                tokenizer.ggml.bos_token_id u32              = 151643
llama_model_loader: - kv  31:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  32:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - kv  33:                      quantize.imatrix.file str              = /mnt/raid/models/ubergarm/Qwen3-235B-...
llama_model_loader: - kv  34:                   quantize.imatrix.dataset str              = calibration_data_v5_rc.txt
llama_model_loader: - kv  35:             quantize.imatrix.entries_count i32              = 753
llama_model_loader: - kv  36:              quantize.imatrix.chunks_count i32              = 225
llama_model_loader: - kv  37:                                   split.no u16              = 0
llama_model_loader: - kv  38:                                split.count u16              = 3
llama_model_loader: - kv  39:                        split.tensors.count i32              = 1131
llama_model_loader: - type  f32:  471 tensors
llama_model_loader: - type q8_0:    2 tensors
llama_model_loader: - type iq3_k:  188 tensors
llama_model_loader: - type iq4_k:   94 tensors
llama_model_loader: - type iq6_k:  376 tensors
llm_load_vocab: special tokens cache size = 26
llm_load_vocab: token to piece cache size = 0.9311 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = qwen3moe
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 151936
llm_load_print_meta: n_merges         = 151387
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 40960
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_layer          = 94
llm_load_print_meta: n_head           = 64
llm_load_print_meta: n_head_kv        = 4
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 16
llm_load_print_meta: n_embd_k_gqa     = 512
llm_load_print_meta: n_embd_v_gqa     = 512
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 12288
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
llm_load_print_meta: model ftype      = IQ3_K - 3.4325 bpw
llm_load_print_meta: model params     = 235.094 B
llm_load_print_meta: model size       = 106.830 GiB (3.903 BPW)
llm_load_print_meta: repeating layers = 105.598 GiB (3.879 BPW, 233.849 B parameters)
llm_load_print_meta: general.name     = Qwen3 235B A22B
llm_load_print_meta: BOS token        = 151643 '<|endoftext|>'
llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
llm_load_print_meta: PAD token        = 151643 '<|endoftext|>'
llm_load_print_meta: LF token         = 148848 'ÄĬ'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp         = 1536
llm_load_tensors: ggml ctx size =    0.99 MiB
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
.
.
.
Tensor blk.92.ffn_norm.weight buffer type overriden to CPU
Tensor blk.92.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.92.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_norm.weight buffer type overriden to CPU
Tensor blk.93.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.93.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 94 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 95/95 layers to GPU
llm_load_tensors:        CPU buffer size = 89709.28 MiB
llm_load_tensors:  CUDA_Host buffer size =   630.59 MiB
llm_load_tensors:      CUDA0 buffer size = 19053.73 MiB
....................................................................................................
============ Repacked 246 tensors
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 512
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:      CUDA0 KV buffer size =  3196.05 MiB
llama_new_context_with_model: KV self size  = 3196.00 MiB, K (q8_0): 1598.00 MiB, V (q8_0): 1598.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.58 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =   312.75 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   128.01 MiB
llama_new_context_with_model: graph nodes  = 3672
llama_new_context_with_model: graph splits = 330

main: n_kv_max = 32768, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = 99, n_threads = 16, n_threads_batch = 16
```

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    3.667 |   139.63 |   11.865 |    10.79 |
|   512 |    128 |    512 |    3.564 |   143.66 |   12.067 |    10.61 |
|   512 |    128 |   1024 |    3.594 |   142.45 |   12.239 |    10.46 |
|   512 |    128 |   1536 |    3.615 |   141.62 |   12.422 |    10.30 |
|   512 |    128 |   2048 |    3.638 |   140.75 |   12.606 |    10.15 |
|   512 |    128 |   2560 |    3.642 |   140.59 |   12.770 |    10.02 |
|   512 |    128 |   3072 |    3.672 |   139.44 |   12.954 |     9.88 |
|   512 |    128 |   3584 |    3.687 |   138.87 |   13.135 |     9.75 |
|   512 |    128 |   4096 |    3.708 |   138.08 |   13.311 |     9.62 |
|   512 |    128 |   4608 |    3.729 |   137.31 |   13.489 |     9.49 |
|   512 |    128 |   5120 |    3.746 |   136.68 |   13.674 |     9.36 |
|   512 |    128 |   5632 |    3.759 |   136.20 |   13.856 |     9.24 |
|   512 |    128 |   6144 |    3.786 |   135.24 |   14.030 |     9.12 |
|   512 |    128 |   6656 |    3.792 |   135.03 |   14.208 |     9.01 |
|   512 |    128 |   7168 |    3.817 |   134.15 |   14.403 |     8.89 |
|   512 |    128 |   7680 |    3.821 |   134.00 |   14.575 |     8.78 |
|   512 |    128 |   8192 |    3.855 |   132.83 |   14.750 |     8.68 |
|   512 |    128 |   8704 |    3.859 |   132.68 |   14.936 |     8.57 |
|   512 |    128 |   9216 |    3.884 |   131.81 |   15.119 |     8.47 |
|   512 |    128 |   9728 |    3.891 |   131.57 |   15.302 |     8.36 |
|   512 |    128 |  10240 |    3.916 |   130.74 |   15.423 |     8.30 |
|   512 |    128 |  10752 |    3.928 |   130.35 |   15.614 |     8.20 |
|   512 |    128 |  11264 |    3.962 |   129.23 |   15.784 |     8.11 |
|   512 |    128 |  11776 |    4.014 |   127.55 |   15.800 |     8.10 |
|   512 |    128 |  12288 |    3.987 |   128.42 |   15.812 |     8.10 |
|   512 |    128 |  12800 |    3.999 |   128.03 |   15.824 |     8.09 |
|   512 |    128 |  13312 |    4.007 |   127.78 |   16.001 |     8.00 |
|   512 |    128 |  13824 |    4.048 |   126.47 |   16.150 |     7.93 |
|   512 |    128 |  14336 |    4.051 |   126.38 |   16.322 |     7.84 |
|   512 |    128 |  14848 |    4.065 |   125.94 |   16.484 |     7.76 |
|   512 |    128 |  15360 |    4.082 |   125.43 |   16.642 |     7.69 |
|   512 |    128 |  15872 |    4.103 |   124.77 |   16.808 |     7.62 |
|   512 |    128 |  16384 |    4.121 |   124.23 |   16.962 |     7.55 |
|   512 |    128 |  16896 |    4.135 |   123.84 |   17.122 |     7.48 |
|   512 |    128 |  17408 |    4.167 |   122.88 |   17.291 |     7.40 |
|   512 |    128 |  17920 |    4.191 |   122.16 |   17.458 |     7.33 |
|   512 |    128 |  18432 |    4.192 |   122.13 |   17.627 |     7.26 |
|   512 |    128 |  18944 |    4.210 |   121.61 |   17.789 |     7.20 |
|   512 |    128 |  19456 |    4.231 |   121.03 |   17.946 |     7.13 |
|   512 |    128 |  19968 |    4.258 |   120.25 |   18.109 |     7.07 |
|   512 |    128 |  20480 |    4.263 |   120.12 |   18.267 |     7.01 |
|   512 |    128 |  20992 |    4.274 |   119.79 |   18.431 |     6.94 |
|   512 |    128 |  21504 |    4.300 |   119.07 |   18.586 |     6.89 |
|   512 |    128 |  22016 |    4.325 |   118.37 |   18.743 |     6.83 |
|   512 |    128 |  22528 |    4.349 |   117.74 |   18.906 |     6.77 |
|   512 |    128 |  23040 |    4.354 |   117.59 |   19.067 |     6.71 |
|   512 |    128 |  23552 |    4.373 |   117.08 |   19.282 |     6.64 |
|   512 |    128 |  24064 |    4.391 |   116.59 |   19.456 |     6.58 |
|   512 |    128 |  24576 |    4.412 |   116.06 |   19.616 |     6.53 |
|   512 |    128 |  25088 |    4.435 |   115.45 |   19.777 |     6.47 |
|   512 |    128 |  25600 |    4.442 |   115.26 |   19.947 |     6.42 |
|   512 |    128 |  26112 |    4.462 |   114.76 |   20.106 |     6.37 |
|   512 |    128 |  26624 |    4.481 |   114.25 |   20.274 |     6.31 |
|   512 |    128 |  27136 |    4.501 |   113.76 |   20.439 |     6.26 |
|   512 |    128 |  27648 |    4.521 |   113.24 |   20.597 |     6.21 |
|   512 |    128 |  28160 |    4.533 |   112.94 |   20.768 |     6.16 |
|   512 |    128 |  28672 |    4.547 |   112.60 |   20.927 |     6.12 |
|   512 |    128 |  29184 |    4.577 |   111.86 |   21.093 |     6.07 |
|   512 |    128 |  29696 |    4.587 |   111.63 |   21.252 |     6.02 |
|   512 |    128 |  30208 |    4.604 |   111.20 |   21.416 |     5.98 |
|   512 |    128 |  30720 |    4.630 |   110.57 |   21.584 |     5.93 |
|   512 |    128 |  31232 |    4.644 |   110.24 |   21.749 |     5.89 |
|   512 |    128 |  31744 |    4.661 |   109.84 |   21.920 |     5.84 |
|   512 |    128 |  32256 |    4.685 |   109.28 |   22.087 |     5.80 |

## `ik_llama.cpp/ik/fattn_mma@056f0818` PR370
```
cmake -B build -DGGML_CUDA=ON -DGGML_RPC=OFF -DGGML_BLAS=OFF
cmake --build build --config Release -j $(nproc)

CUDA_VISIBLE_DEVICES="0" \
./build/bin/llama-sweep-bench \
  --model /mnt/ai/models/ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf \
  -fa \
  -ctk q8_0 -ctv q8_0 \
  -c 32768 \
  -fmoe \
  -amb 512 \
  -rtr \
  -ot blk\.1[2-9]\.ffn.*=CPU \
  -ot blk\.[2-8][0-9]\.ffn.*=CPU \
  -ot blk\.9[0-3]\.ffn.*=CPU \
  -ngl 99 \
  --threads 16

ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090 Ti, compute capability 8.6, VMM: yes
llama_model_loader: additional 2 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 40 key-value pairs and 1131 tensors from /mnt/ai/models/ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3 235B A22B
llama_model_loader: - kv   3:                           general.basename str              = Qwen3
llama_model_loader: - kv   4:                         general.size_label str              = 235B-A22B
llama_model_loader: - kv   5:                            general.license str              = apache-2.0
llama_model_loader: - kv   6:                       general.license.link str              = https://huggingface.co/Qwen/Qwen3-235...
llama_model_loader: - kv   7:                               general.tags arr[str,1]       = ["text-generation"]
llama_model_loader: - kv   8:                       qwen3moe.block_count u32              = 94
llama_model_loader: - kv   9:                    qwen3moe.context_length u32              = 40960
llama_model_loader: - kv  10:                  qwen3moe.embedding_length u32              = 4096
llama_model_loader: - kv  11:               qwen3moe.feed_forward_length u32              = 12288
llama_model_loader: - kv  12:              qwen3moe.attention.head_count u32              = 64
llama_model_loader: - kv  13:           qwen3moe.attention.head_count_kv u32              = 4
llama_model_loader: - kv  14:                    qwen3moe.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  15:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  16:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  17:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  18:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  19:                          general.file_type u32              = 139
llama_model_loader: - kv  20:                      qwen3moe.expert_count u32              = 128
llama_model_loader: - kv  21:        qwen3moe.expert_feed_forward_length u32              = 1536
llama_model_loader: - kv  22:               general.quantization_version u32              = 2
llama_model_loader: - kv  23:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  24:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  25:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  26:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  27:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv  28:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  29:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  30:                tokenizer.ggml.bos_token_id u32              = 151643
llama_model_loader: - kv  31:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  32:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - kv  33:                      quantize.imatrix.file str              = /mnt/raid/models/ubergarm/Qwen3-235B-...
llama_model_loader: - kv  34:                   quantize.imatrix.dataset str              = calibration_data_v5_rc.txt
llama_model_loader: - kv  35:             quantize.imatrix.entries_count i32              = 753
llama_model_loader: - kv  36:              quantize.imatrix.chunks_count i32              = 225
llama_model_loader: - kv  37:                                   split.no u16              = 0
llama_model_loader: - kv  38:                                split.count u16              = 3
llama_model_loader: - kv  39:                        split.tensors.count i32              = 1131
llama_model_loader: - type  f32:  471 tensors
llama_model_loader: - type q8_0:    2 tensors
llama_model_loader: - type iq3_k:  188 tensors
llama_model_loader: - type iq4_k:   94 tensors
llama_model_loader: - type iq6_k:  376 tensors
llm_load_vocab: special tokens cache size = 26
llm_load_vocab: token to piece cache size = 0.9311 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = qwen3moe
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 151936
llm_load_print_meta: n_merges         = 151387
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 40960
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_layer          = 94
llm_load_print_meta: n_head           = 64
llm_load_print_meta: n_head_kv        = 4
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 16
llm_load_print_meta: n_embd_k_gqa     = 512
llm_load_print_meta: n_embd_v_gqa     = 512
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 12288
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
llm_load_print_meta: model ftype      = IQ3_K - 3.4325 bpw
llm_load_print_meta: model params     = 235.094 B
llm_load_print_meta: model size       = 106.830 GiB (3.903 BPW)
llm_load_print_meta: repeating layers = 105.598 GiB (3.879 BPW, 233.849 B parameters)
llm_load_print_meta: general.name     = Qwen3 235B A22B
llm_load_print_meta: BOS token        = 151643 '<|endoftext|>'
llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
llm_load_print_meta: PAD token        = 151643 '<|endoftext|>'
llm_load_print_meta: LF token         = 148848 'ÄĬ'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp         = 1536
llm_load_tensors: ggml ctx size =    0.99 MiB
Tensor blk.12.ffn_norm.weight buffer type overriden to CPU
Tensor blk.12.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CPU
.
.
.
Tensor blk.92.ffn_norm.weight buffer type overriden to CPU
Tensor blk.92.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.92.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_norm.weight buffer type overriden to CPU
Tensor blk.93.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.93.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 94 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 95/95 layers to GPU
llm_load_tensors:        CPU buffer size = 89709.28 MiB
llm_load_tensors:  CUDA_Host buffer size =   630.59 MiB
llm_load_tensors:      CUDA0 buffer size = 19053.73 MiB
....................................................................................................
============ Repacked 246 tensors
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 512
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:      CUDA0 KV buffer size =  3196.05 MiB
llama_new_context_with_model: KV self size  = 3196.00 MiB, K (q8_0): 1598.00 MiB, V (q8_0): 1598.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.58 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =   312.75 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   128.01 MiB
llama_new_context_with_model: graph nodes  = 3672
llama_new_context_with_model: graph splits = 330

main: n_kv_max = 32768, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = 99, n_threads = 16, n_threads_batch = 16
```

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    3.602 |   142.16 |   11.894 |    10.76 |
|   512 |    128 |    512 |    3.549 |   144.25 |   11.982 |    10.68 |
|   512 |    128 |   1024 |    3.563 |   143.72 |   11.981 |    10.68 |
|   512 |    128 |   1536 |    3.577 |   143.14 |   12.010 |    10.66 |
|   512 |    128 |   2048 |    3.606 |   142.00 |   12.061 |    10.61 |
|   512 |    128 |   2560 |    3.597 |   142.36 |   12.088 |    10.59 |
|   512 |    128 |   3072 |    3.626 |   141.22 |   12.122 |    10.56 |
|   512 |    128 |   3584 |    3.618 |   141.52 |   12.173 |    10.52 |
|   512 |    128 |   4096 |    3.639 |   140.70 |   12.196 |    10.50 |
|   512 |    128 |   4608 |    3.640 |   140.65 |   12.243 |    10.45 |
|   512 |    128 |   5120 |    3.653 |   140.16 |   12.270 |    10.43 |
|   512 |    128 |   5632 |    3.666 |   139.65 |   12.385 |    10.34 |
|   512 |    128 |   6144 |    3.669 |   139.55 |   12.415 |    10.31 |
|   512 |    128 |   6656 |    3.677 |   139.24 |   12.478 |    10.26 |
|   512 |    128 |   7168 |    3.701 |   138.34 |   12.474 |    10.26 |
|   512 |    128 |   7680 |    3.702 |   138.29 |   12.491 |    10.25 |
|   512 |    128 |   8192 |    3.716 |   137.77 |   12.543 |    10.20 |
|   512 |    128 |   8704 |    3.731 |   137.23 |   12.562 |    10.19 |
|   512 |    128 |   9216 |    3.731 |   137.21 |   12.598 |    10.16 |
|   512 |    128 |   9728 |    3.737 |   137.00 |   12.629 |    10.14 |
|   512 |    128 |  10240 |    3.773 |   135.71 |   12.667 |    10.11 |
|   512 |    128 |  10752 |    3.772 |   135.75 |   12.780 |    10.02 |
|   512 |    128 |  11264 |    3.785 |   135.28 |   12.838 |     9.97 |
|   512 |    128 |  11776 |    3.787 |   135.20 |   12.830 |     9.98 |
|   512 |    128 |  12288 |    3.810 |   134.40 |   12.852 |     9.96 |
|   512 |    128 |  12800 |    3.804 |   134.59 |   12.910 |     9.91 |
|   512 |    128 |  13312 |    3.815 |   134.21 |   12.923 |     9.90 |
|   512 |    128 |  13824 |    3.817 |   134.13 |   12.943 |     9.89 |
|   512 |    128 |  14336 |    3.824 |   133.90 |   12.985 |     9.86 |
|   512 |    128 |  14848 |    3.844 |   133.19 |   13.024 |     9.83 |
|   512 |    128 |  15360 |    3.848 |   133.05 |   13.051 |     9.81 |
|   512 |    128 |  15872 |    3.890 |   131.63 |   13.066 |     9.80 |
|   512 |    128 |  16384 |    3.892 |   131.55 |   13.182 |     9.71 |
|   512 |    128 |  16896 |    3.880 |   131.96 |   13.218 |     9.68 |
|   512 |    128 |  17408 |    3.901 |   131.26 |   13.277 |     9.64 |
|   512 |    128 |  17920 |    3.905 |   131.12 |   13.278 |     9.64 |
|   512 |    128 |  18432 |    3.943 |   129.85 |   13.313 |     9.61 |
|   512 |    128 |  18944 |    3.909 |   130.97 |   13.315 |     9.61 |
|   512 |    128 |  19456 |    3.927 |   130.39 |   13.315 |     9.61 |
|   512 |    128 |  19968 |    3.950 |   129.63 |   13.364 |     9.58 |
|   512 |    128 |  20480 |    3.934 |   130.16 |   13.404 |     9.55 |
|   512 |    128 |  20992 |    3.935 |   130.12 |   13.415 |     9.54 |
|   512 |    128 |  21504 |    3.973 |   128.86 |   13.522 |     9.47 |
|   512 |    128 |  22016 |    3.975 |   128.80 |   13.583 |     9.42 |
|   512 |    128 |  22528 |    4.004 |   127.88 |   13.580 |     9.43 |
|   512 |    128 |  23040 |    3.993 |   128.24 |   13.606 |     9.41 |
|   512 |    128 |  23552 |    3.996 |   128.13 |   13.660 |     9.37 |
|   512 |    128 |  24064 |    4.024 |   127.24 |   13.663 |     9.37 |
|   512 |    128 |  24576 |    4.024 |   127.25 |   13.692 |     9.35 |
|   512 |    128 |  25088 |    4.041 |   126.69 |   13.737 |     9.32 |
|   512 |    128 |  25600 |    4.040 |   126.75 |   13.763 |     9.30 |
|   512 |    128 |  26112 |    4.047 |   126.51 |   13.791 |     9.28 |
|   512 |    128 |  26624 |    4.070 |   125.81 |   13.828 |     9.26 |
|   512 |    128 |  27136 |    4.080 |   125.49 |   13.935 |     9.19 |
|   512 |    128 |  27648 |    4.087 |   125.27 |   13.960 |     9.17 |
|   512 |    128 |  28160 |    4.093 |   125.09 |   14.016 |     9.13 |
|   512 |    128 |  28672 |    4.095 |   125.02 |   14.016 |     9.13 |
|   512 |    128 |  29184 |    4.120 |   124.28 |   14.055 |     9.11 |
|   512 |    128 |  29696 |    4.121 |   124.23 |   14.097 |     9.08 |
|   512 |    128 |  30208 |    4.124 |   124.14 |   14.107 |     9.07 |
|   512 |    128 |  30720 |    4.152 |   123.31 |   14.150 |     9.05 |
|   512 |    128 |  31232 |    4.155 |   123.23 |   14.170 |     9.03 |
|   512 |    128 |  31744 |    4.160 |   123.07 |   14.208 |     9.01 |
|   512 |    128 |  32256 |    4.180 |   122.48 |   14.296 |     8.95 |

</details>

---

👤 **ubergarm** commented the **2025-05-04** at **01:36:41**:<br>

## ubergarm/Qwen3-235B-A22B-Q8_0

![qwen3-235b-Q8_0-sweep-pr370](https://github.com/user-attachments/assets/6d7dc116-898d-4c76-9a75-e74718dd1fe9)

Some uplift on PP even and wow on TG! fwiw I benched this rig at around 225-250GB/s RAM i/o 8x32GB DDR5 running at slower 4800MHz with Intel Memory Latency Checker `mlc`.

<details>

<summary>👈 Logs</summary>

## `llama.cpp/master@36667c8e` + `ug/port-sweep-bench@d541533a`
```
cmake -B build -DGGML_CUDA=ON -DGGML_RPC=OFF -DGGML_BLAS=OFF
cmake --build build --config Release -j $(nproc)

CUDA_VISIBLE_DEVICES="0" \
./build/bin/llama-sweep-bench \
  --no-mmap \
  --model /mnt/raid/models/ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-Q8_0.gguf \
  -fa \
  -ctk q8_0 -ctv q8_0 \
  -c 32768 \
  -ot blk\.1[4-9]\.ffn.*=CPU \
  -ot blk\.[2-9][0-9]\.ffn.*=CPU \
  -ngl 99 \
  --threads 24

ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA RTX A6000, compute capability 8.6, VMM: yes
build: 5274 (d541533a) with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu
llama_model_load_from_file_impl: using device CUDA0 (NVIDIA RTX A6000) - 48267 MiB free
llama_model_loader: loaded meta data with 33 key-value pairs and 1131 tensors from /mnt/raid/models/ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-Q8_0.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3 235B A22B
llama_model_loader: - kv   3:                           general.basename str              = Qwen3
llama_model_loader: - kv   4:                         general.size_label str              = 235B-A22B
llama_model_loader: - kv   5:                            general.license str              = apache-2.0
llama_model_loader: - kv   6:                       general.license.link str              = https://huggingface.co/Qwen/Qwen3-235...
llama_model_loader: - kv   7:                               general.tags arr[str,1]       = ["text-generation"]
llama_model_loader: - kv   8:                       qwen3moe.block_count u32              = 94
llama_model_loader: - kv   9:                    qwen3moe.context_length u32              = 40960
llama_model_loader: - kv  10:                  qwen3moe.embedding_length u32              = 4096
llama_model_loader: - kv  11:               qwen3moe.feed_forward_length u32              = 12288
llama_model_loader: - kv  12:              qwen3moe.attention.head_count u32              = 64
llama_model_loader: - kv  13:           qwen3moe.attention.head_count_kv u32              = 4
llama_model_loader: - kv  14:                    qwen3moe.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  15:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  16:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  17:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  18:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  19:                          general.file_type u32              = 7
llama_model_loader: - kv  20:                      qwen3moe.expert_count u32              = 128
llama_model_loader: - kv  21:        qwen3moe.expert_feed_forward_length u32              = 1536
llama_model_loader: - kv  22:               general.quantization_version u32              = 2
llama_model_loader: - kv  23:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  24:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  25:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  26:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  27:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv  28:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  29:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  30:                tokenizer.ggml.bos_token_id u32              = 151643
llama_model_loader: - kv  31:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  32:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - type  f32:  471 tensors
llama_model_loader: - type q8_0:  660 tensors
print_info: file format = GGUF V3 (latest)
print_info: file type   = Q8_0
print_info: file size   = 232.77 GiB (8.51 BPW) 
load: special tokens cache size = 26
load: token to piece cache size = 0.9311 MB
print_info: arch             = qwen3moe
print_info: vocab_only       = 0
print_info: n_ctx_train      = 40960
print_info: n_embd           = 4096
print_info: n_layer          = 94
print_info: n_head           = 64
print_info: n_head_kv        = 4
print_info: n_rot            = 128
print_info: n_swa            = 0
print_info: n_swa_pattern    = 1
print_info: n_embd_head_k    = 128
print_info: n_embd_head_v    = 128
print_info: n_gqa            = 16
print_info: n_embd_k_gqa     = 512
print_info: n_embd_v_gqa     = 512
print_info: f_norm_eps       = 0.0e+00
print_info: f_norm_rms_eps   = 1.0e-06
print_info: f_clamp_kqv      = 0.0e+00
print_info: f_max_alibi_bias = 0.0e+00
print_info: f_logit_scale    = 0.0e+00
print_info: f_attn_scale     = 0.0e+00
print_info: n_ff             = 12288
print_info: n_expert         = 128
print_info: n_expert_used    = 8
print_info: causal attn      = 1
print_info: pooling type     = 0
print_info: rope type        = 2
print_info: rope scaling     = linear
print_info: freq_base_train  = 1000000.0
print_info: freq_scale_train = 1
print_info: n_ctx_orig_yarn  = 40960
print_info: rope_finetuned   = unknown
print_info: ssm_d_conv       = 0
print_info: ssm_d_inner      = 0
print_info: ssm_d_state      = 0
print_info: ssm_dt_rank      = 0
print_info: ssm_dt_b_c_rms   = 0
print_info: model type       = 235B.A22B
print_info: model params     = 235.09 B
print_info: general.name     = Qwen3 235B A22B
print_info: n_ff_exp         = 1536
print_info: vocab type       = BPE
print_info: n_vocab          = 151936
print_info: n_merges         = 151387
print_info: BOS token        = 151643 '<|endoftext|>'
print_info: EOS token        = 151645 '<|im_end|>'
print_info: EOT token        = 151645 '<|im_end|>'
print_info: PAD token        = 151643 '<|endoftext|>'
print_info: LF token         = 198 'Ċ'
print_info: FIM PRE token    = 151659 '<|fim_prefix|>'
print_info: FIM SUF token    = 151661 '<|fim_suffix|>'
print_info: FIM MID token    = 151660 '<|fim_middle|>'
print_info: FIM PAD token    = 151662 '<|fim_pad|>'
print_info: FIM REP token    = 151663 '<|repo_name|>'
print_info: FIM SEP token    = 151664 '<|file_sep|>'
print_info: EOG token        = 151643 '<|endoftext|>'
print_info: EOG token        = 151645 '<|im_end|>'
print_info: EOG token        = 151662 '<|fim_pad|>'
print_info: EOG token        = 151663 '<|repo_name|>'
print_info: EOG token        = 151664 '<|file_sep|>'
print_info: max token length = 256
load_tensors: loading model tensors, this can take a while... (mmap = false)
load_tensors: offloading 94 repeating layers to GPU
load_tensors: offloading output layer to GPU
load_tensors: offloaded 95/95 layers to GPU
load_tensors:    CUDA_Host model buffer size =   630.59 MiB
load_tensors:        CUDA0 model buffer size = 41723.89 MiB
load_tensors:          CPU model buffer size = 196001.25 MiB
....................................................................................................
llama_context: constructing llama_context
llama_context: n_seq_max     = 1
llama_context: n_ctx         = 32768
llama_context: n_ctx_per_seq = 32768
llama_context: n_batch       = 2048
llama_context: n_ubatch      = 512
llama_context: causal_attn   = 1
llama_context: flash_attn    = 1
llama_context: freq_base     = 1000000.0
llama_context: freq_scale    = 1
llama_context: n_ctx_per_seq (32768) < n_ctx_train (40960) -- the full capacity of the model will not be utilized
llama_context:  CUDA_Host  output buffer size =     0.58 MiB
llama_kv_cache_unified: kv_size = 32768, type_k = 'q8_0', type_v = 'q8_0', n_layer = 94, can_shift = 1, padding = 256
llama_kv_cache_unified:      CUDA0 KV buffer size =  3196.00 MiB
llama_kv_cache_unified: KV self size  = 3196.00 MiB, K (q8_0): 1598.00 MiB, V (q8_0): 1598.00 MiB
llama_context:      CUDA0 compute buffer size =  1024.00 MiB
llama_context:  CUDA_Host compute buffer size =    72.01 MiB
llama_context: graph nodes  = 5741
llama_context: graph splits = 402 (with bs=512), 162 (with bs=1)

main: n_kv_max = 32768, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = 99, n_threads = 24, n_threads_batch = 24
```

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |   14.007 |    36.55 |   12.532 |    10.21 |
|   512 |    128 |    512 |    8.816 |    58.08 |   12.413 |    10.31 |
|   512 |    128 |   1024 |    8.829 |    57.99 |   12.261 |    10.44 |
|   512 |    128 |   1536 |    8.845 |    57.89 |   12.561 |    10.19 |
|   512 |    128 |   2048 |    8.945 |    57.24 |   12.554 |    10.20 |
|   512 |    128 |   2560 |    8.867 |    57.74 |   12.692 |    10.09 |
|   512 |    128 |   3072 |    8.885 |    57.63 |   13.042 |     9.81 |
|   512 |    128 |   3584 |    8.970 |    57.08 |   12.867 |     9.95 |
|   512 |    128 |   4096 |    8.905 |    57.50 |   13.031 |     9.82 |
|   512 |    128 |   4608 |    8.905 |    57.49 |   13.275 |     9.64 |
|   512 |    128 |   5120 |    8.970 |    57.08 |   13.348 |     9.59 |
|   512 |    128 |   5632 |    8.923 |    57.38 |   13.429 |     9.53 |
|   512 |    128 |   6144 |    8.937 |    57.29 |   13.767 |     9.30 |
|   512 |    128 |   6656 |    8.946 |    57.23 |   13.772 |     9.29 |
|   512 |    128 |   7168 |    9.008 |    56.84 |   13.779 |     9.29 |
|   512 |    128 |   7680 |    8.969 |    57.09 |   13.994 |     9.15 |
|   512 |    128 |   8192 |    8.987 |    56.97 |   14.149 |     9.05 |
|   512 |    128 |   8704 |    9.075 |    56.42 |   14.104 |     9.08 |
|   512 |    128 |   9216 |    9.012 |    56.81 |   14.282 |     8.96 |
|   512 |    128 |   9728 |    9.015 |    56.80 |   14.566 |     8.79 |
|   512 |    128 |  10240 |    9.106 |    56.23 |   14.534 |     8.81 |
|   512 |    128 |  10752 |    9.038 |    56.65 |   14.579 |     8.78 |
|   512 |    128 |  11264 |    9.047 |    56.59 |   14.862 |     8.61 |
|   512 |    128 |  11776 |    9.051 |    56.57 |   14.918 |     8.58 |
|   512 |    128 |  12288 |    9.147 |    55.97 |   14.928 |     8.57 |
|   512 |    128 |  12800 |    9.072 |    56.44 |   15.027 |     8.52 |
|   512 |    128 |  13312 |    9.076 |    56.41 |   15.275 |     8.38 |
|   512 |    128 |  13824 |    9.090 |    56.32 |   15.356 |     8.34 |
|   512 |    128 |  14336 |    9.177 |    55.79 |   15.364 |     8.33 |
|   512 |    128 |  14848 |    9.109 |    56.21 |   15.496 |     8.26 |
|   512 |    128 |  15360 |    9.114 |    56.18 |   15.733 |     8.14 |
|   512 |    128 |  15872 |    9.133 |    56.06 |   15.904 |     8.05 |
|   512 |    128 |  16384 |    9.222 |    55.52 |   15.832 |     8.09 |
|   512 |    128 |  16896 |    9.149 |    55.96 |   15.974 |     8.01 |
|   512 |    128 |  17408 |    9.173 |    55.82 |   16.203 |     7.90 |
|   512 |    128 |  17920 |    9.176 |    55.80 |   16.438 |     7.79 |
|   512 |    128 |  18432 |    9.264 |    55.27 |   16.402 |     7.80 |
|   512 |    128 |  18944 |    9.191 |    55.71 |   16.485 |     7.76 |
|   512 |    128 |  19456 |    9.203 |    55.63 |   16.812 |     7.61 |
|   512 |    128 |  19968 |    9.227 |    55.49 |   16.948 |     7.55 |
|   512 |    128 |  20480 |    9.227 |    55.49 |   17.059 |     7.50 |
|   512 |    128 |  20992 |    9.309 |    55.00 |   17.053 |     7.51 |
|   512 |    128 |  21504 |    9.241 |    55.40 |   17.064 |     7.50 |
|   512 |    128 |  22016 |    9.256 |    55.31 |   17.331 |     7.39 |
|   512 |    128 |  22528 |    9.260 |    55.29 |   17.527 |     7.30 |
|   512 |    128 |  23040 |    9.268 |    55.24 |   17.592 |     7.28 |
|   512 |    128 |  23552 |    9.361 |    54.69 |   17.661 |     7.25 |
|   512 |    128 |  24064 |    9.374 |    54.62 |   17.745 |     7.21 |
|   512 |    128 |  24576 |    9.301 |    55.05 |   17.900 |     7.15 |
|   512 |    128 |  25088 |    9.309 |    55.00 |   18.105 |     7.07 |
|   512 |    128 |  25600 |    9.319 |    54.94 |   18.279 |     7.00 |
|   512 |    128 |  26112 |    9.333 |    54.86 |   18.366 |     6.97 |
|   512 |    128 |  26624 |    9.425 |    54.32 |   18.404 |     6.95 |
|   512 |    128 |  27136 |    9.431 |    54.29 |   18.559 |     6.90 |
|   512 |    128 |  27648 |    9.364 |    54.68 |   18.721 |     6.84 |
|   512 |    128 |  28160 |    9.369 |    54.65 |   18.969 |     6.75 |
|   512 |    128 |  28672 |    9.379 |    54.59 |   19.154 |     6.68 |
|   512 |    128 |  29184 |    9.394 |    54.50 |   19.230 |     6.66 |
|   512 |    128 |  29696 |    9.398 |    54.48 |   19.305 |     6.63 |
|   512 |    128 |  30208 |    9.422 |    54.34 |   19.402 |     6.60 |
|   512 |    128 |  30720 |    9.498 |    53.90 |   19.485 |     6.57 |
|   512 |    128 |  31232 |    9.515 |    53.81 |   19.626 |     6.52 |
|   512 |    128 |  31744 |    9.436 |    54.26 |   19.686 |     6.50 |
|   512 |    128 |  32256 |    9.455 |    54.15 |   19.969 |     6.41 |

## `ik_llama.cpp/main@ab7f694b`
```
cmake -B build -DGGML_CUDA=ON -DGGML_RPC=OFF -DGGML_BLAS=OFF
cmake --build build --config Release -j $(nproc)

CUDA_VISIBLE_DEVICES="0" \
./build/bin/llama-sweep-bench \
  --no-mmap \
  --model /mnt/raid/models/ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-Q8_0.gguf \
  -fa \
  -rtr -fmoe \
  -ctk q8_0 -ctv q8_0 \
  -c 32768 \
  -ot blk\.1[4-9]\.ffn.*=CPU \
  -ot blk\.[2-9][0-9]\.ffn.*=CPU \
  -ngl 99 \
  --threads 24


ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA RTX A6000, compute capability 8.6, VMM: yes
llama_model_loader: loaded meta data with 33 key-value pairs and 1131 tensors from /mnt/raid/models/ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-Q8_0.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3 235B A22B
llama_model_loader: - kv   3:                           general.basename str              = Qwen3
llama_model_loader: - kv   4:                         general.size_label str              = 235B-A22B
llama_model_loader: - kv   5:                            general.license str              = apache-2.0
llama_model_loader: - kv   6:                       general.license.link str              = https://huggingface.co/Qwen/Qwen3-235...
llama_model_loader: - kv   7:                               general.tags arr[str,1]       = ["text-generation"]
llama_model_loader: - kv   8:                       qwen3moe.block_count u32              = 94
llama_model_loader: - kv   9:                    qwen3moe.context_length u32              = 40960
llama_model_loader: - kv  10:                  qwen3moe.embedding_length u32              = 4096
llama_model_loader: - kv  11:               qwen3moe.feed_forward_length u32              = 12288
llama_model_loader: - kv  12:              qwen3moe.attention.head_count u32              = 64
llama_model_loader: - kv  13:           qwen3moe.attention.head_count_kv u32              = 4
llama_model_loader: - kv  14:                    qwen3moe.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  15:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  16:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  17:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  18:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  19:                          general.file_type u32              = 7
llama_model_loader: - kv  20:                      qwen3moe.expert_count u32              = 128
llama_model_loader: - kv  21:        qwen3moe.expert_feed_forward_length u32              = 1536
llama_model_loader: - kv  22:               general.quantization_version u32              = 2
llama_model_loader: - kv  23:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  24:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  25:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  26:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  27:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv  28:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  29:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  30:                tokenizer.ggml.bos_token_id u32              = 151643
llama_model_loader: - kv  31:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  32:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - type  f32:  471 tensors
llama_model_loader: - type q8_0:  660 tensors
llm_load_vocab: special tokens cache size = 26
llm_load_vocab: token to piece cache size = 0.9311 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = qwen3moe
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 151936
llm_load_print_meta: n_merges         = 151387
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 40960
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_layer          = 94
llm_load_print_meta: n_head           = 64
llm_load_print_meta: n_head_kv        = 4
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 16
llm_load_print_meta: n_embd_k_gqa     = 512
llm_load_print_meta: n_embd_v_gqa     = 512
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 12288
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
llm_load_print_meta: model ftype      = Q8_0
llm_load_print_meta: model params     = 235.094 B
llm_load_print_meta: model size       = 232.769 GiB (8.505 BPW) 
llm_load_print_meta: repeating layers = 231.538 GiB (8.505 BPW, 233.849 B parameters)
llm_load_print_meta: general.name     = Qwen3 235B A22B
llm_load_print_meta: BOS token        = 151643 '<|endoftext|>'
llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
llm_load_print_meta: PAD token        = 151643 '<|endoftext|>'
llm_load_print_meta: LF token         = 148848 'ÄĬ'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp         = 1536
llm_load_tensors: ggml ctx size =    0.99 MiB
Tensor blk.14.ffn_norm.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CPU
.
.
.
Tensor blk.93.ffn_norm.weight buffer type overriden to CPU
Tensor blk.93.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.93.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 94 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 95/95 layers to GPU
llm_load_tensors:        CPU buffer size = 196001.25 MiB
llm_load_tensors:  CUDA_Host buffer size =   630.59 MiB
llm_load_tensors:      CUDA0 buffer size = 41723.89 MiB
....................................................................................................
============ Repacked 240 tensors
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
llama_kv_cache_init:      CUDA0 KV buffer size =  3196.05 MiB
llama_new_context_with_model: KV self size  = 3196.00 MiB, K (q8_0): 1598.00 MiB, V (q8_0): 1598.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.58 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =   312.75 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   128.01 MiB
llama_new_context_with_model: graph nodes  = 3672
llama_new_context_with_model: graph splits = 322

main: n_kv_max = 32768, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = 99, n_threads = 24, n_threads_batch = 24

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    3.130 |   163.60 |   12.075 |    10.60 |
|   512 |    128 |    512 |    3.077 |   166.38 |   11.996 |    10.67 |
|   512 |    128 |   1024 |    3.189 |   160.57 |   12.285 |    10.42 |
|   512 |    128 |   1536 |    3.107 |   164.80 |   12.322 |    10.39 |
|   512 |    128 |   2048 |    3.132 |   163.50 |   12.622 |    10.14 |
|   512 |    128 |   2560 |    3.260 |   157.06 |   12.659 |    10.11 |
|   512 |    128 |   3072 |    3.157 |   162.19 |   12.875 |     9.94 |
|   512 |    128 |   3584 |    3.255 |   157.32 |   12.953 |     9.88 |
|   512 |    128 |   4096 |    3.223 |   158.85 |   13.228 |     9.68 |
|   512 |    128 |   4608 |    3.231 |   158.46 |   13.312 |     9.62 |
|   512 |    128 |   5120 |    3.346 |   153.02 |   13.649 |     9.38 |
|   512 |    128 |   5632 |    3.301 |   155.10 |   13.704 |     9.34 |
|   512 |    128 |   6144 |    3.377 |   151.63 |   13.940 |     9.18 |
|   512 |    128 |   6656 |    3.316 |   154.40 |   14.032 |     9.12 |
|   512 |    128 |   7168 |    3.343 |   153.17 |   14.353 |     8.92 |
|   512 |    128 |   7680 |    3.426 |   149.45 |   14.372 |     8.91 |
|   512 |    128 |   8192 |    3.378 |   151.59 |   14.688 |     8.71 |
|   512 |    128 |   8704 |    3.458 |   148.07 |   14.630 |     8.75 |
|   512 |    128 |   9216 |    3.397 |   150.74 |   14.790 |     8.65 |
|   512 |    128 |   9728 |    3.673 |   139.41 |   14.919 |     8.58 |
|   512 |    128 |  10240 |    3.451 |   148.38 |   15.128 |     8.46 |
|   512 |    128 |  10752 |    3.538 |   144.70 |   15.245 |     8.40 |
|   512 |    128 |  11264 |    3.499 |   146.33 |   15.421 |     8.30 |
|   512 |    128 |  11776 |    3.518 |   145.52 |   15.652 |     8.18 |
|   512 |    128 |  12288 |    3.547 |   144.33 |   15.755 |     8.12 |
|   512 |    128 |  12800 |    3.555 |   144.02 |   15.985 |     8.01 |
|   512 |    128 |  13312 |    3.770 |   135.81 |   16.114 |     7.94 |
|   512 |    128 |  13824 |    3.564 |   143.67 |   16.239 |     7.88 |
|   512 |    128 |  14336 |    3.580 |   143.00 |   16.504 |     7.76 |
|   512 |    128 |  14848 |    3.604 |   142.05 |   16.563 |     7.73 |
|   512 |    128 |  15360 |    3.617 |   141.54 |   16.772 |     7.63 |
|   512 |    128 |  15872 |    3.909 |   130.97 |   16.899 |     7.57 |
|   512 |    128 |  16384 |    3.652 |   140.18 |   17.049 |     7.51 |
|   512 |    128 |  16896 |    3.674 |   139.36 |   17.253 |     7.42 |
|   512 |    128 |  17408 |    3.705 |   138.19 |   17.436 |     7.34 |
|   512 |    128 |  17920 |    3.754 |   136.40 |   17.676 |     7.24 |
|   512 |    128 |  18432 |    3.846 |   133.11 |   17.804 |     7.19 |
|   512 |    128 |  18944 |    3.811 |   134.36 |   17.920 |     7.14 |
|   512 |    128 |  19456 |    3.791 |   135.06 |   18.148 |     7.05 |
|   512 |    128 |  19968 |    3.816 |   134.15 |   18.329 |     6.98 |
|   512 |    128 |  20480 |    3.813 |   134.27 |   18.433 |     6.94 |
|   512 |    128 |  20992 |    3.864 |   132.52 |   18.645 |     6.87 |
|   512 |    128 |  21504 |    3.864 |   132.51 |   18.878 |     6.78 |
|   512 |    128 |  22016 |    3.961 |   129.26 |   18.987 |     6.74 |
|   512 |    128 |  22528 |    4.109 |   124.60 |   19.224 |     6.66 |
|   512 |    128 |  23040 |    3.916 |   130.75 |   19.421 |     6.59 |
|   512 |    128 |  23552 |    4.215 |   121.46 |   19.463 |     6.58 |
|   512 |    128 |  24064 |    3.952 |   129.57 |   19.637 |     6.52 |
|   512 |    128 |  24576 |    3.978 |   128.71 |   19.946 |     6.42 |
|   512 |    128 |  25088 |    4.003 |   127.92 |   20.090 |     6.37 |
|   512 |    128 |  25600 |    4.062 |   126.05 |   20.141 |     6.36 |
|   512 |    128 |  26112 |    4.062 |   126.05 |   20.327 |     6.30 |
|   512 |    128 |  26624 |    4.094 |   125.06 |   20.528 |     6.24 |
|   512 |    128 |  27136 |    4.150 |   123.38 |   20.700 |     6.18 |
|   512 |    128 |  27648 |    4.091 |   125.16 |   20.846 |     6.14 |
|   512 |    128 |  28160 |    4.102 |   124.81 |   21.089 |     6.07 |
|   512 |    128 |  28672 |    4.151 |   123.33 |   21.263 |     6.02 |
|   512 |    128 |  29184 |    4.210 |   121.62 |   21.369 |     5.99 |
|   512 |    128 |  29696 |    4.191 |   122.16 |   21.497 |     5.95 |
|   512 |    128 |  30208 |    4.252 |   120.41 |   21.699 |     5.90 |
|   512 |    128 |  30720 |    4.184 |   122.36 |   21.891 |     5.85 |
|   512 |    128 |  31232 |    4.260 |   120.19 |   22.087 |     5.80 |
|   512 |    128 |  31744 |    4.245 |   120.60 |   22.239 |     5.76 |
|   512 |    128 |  32256 |    4.262 |   120.13 |   22.378 |     5.72 |
```

## `ik_llama.cpp/ik/fattn_mma@056f0818` PR370
```
cmake -B build -DGGML_CUDA=ON -DGGML_RPC=OFF -DGGML_BLAS=OFF
cmake --build build --config Release -j $(nproc)

CUDA_VISIBLE_DEVICES="0" \
./build/bin/llama-sweep-bench \
  --no-mmap \
  --model /mnt/raid/models/ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-Q8_0.gguf \
  -fa \
  -rtr -fmoe \
  -ctk q8_0 -ctv q8_0 \
  -c 32768 \
  -ot blk\.1[4-9]\.ffn.*=CPU \
  -ot blk\.[2-9][0-9]\.ffn.*=CPU \
  -ngl 99 \
  --threads 24

ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA RTX A6000, compute capability 8.6, VMM: yes
llama_model_loader: loaded meta data with 33 key-value pairs and 1131 tensors from /mnt/raid/models/ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-Q8_0.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3 235B A22B
llama_model_loader: - kv   3:                           general.basename str              = Qwen3
llama_model_loader: - kv   4:                         general.size_label str              = 235B-A22B
llama_model_loader: - kv   5:                            general.license str              = apache-2.0
llama_model_loader: - kv   6:                       general.license.link str              = https://huggingface.co/Qwen/Qwen3-235...
llama_model_loader: - kv   7:                               general.tags arr[str,1]       = ["text-generation"]
llama_model_loader: - kv   8:                       qwen3moe.block_count u32              = 94
llama_model_loader: - kv   9:                    qwen3moe.context_length u32              = 40960
llama_model_loader: - kv  10:                  qwen3moe.embedding_length u32              = 4096
llama_model_loader: - kv  11:               qwen3moe.feed_forward_length u32              = 12288
llama_model_loader: - kv  12:              qwen3moe.attention.head_count u32              = 64
llama_model_loader: - kv  13:           qwen3moe.attention.head_count_kv u32              = 4
llama_model_loader: - kv  14:                    qwen3moe.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  15:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  16:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  17:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  18:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  19:                          general.file_type u32              = 7
llama_model_loader: - kv  20:                      qwen3moe.expert_count u32              = 128
llama_model_loader: - kv  21:        qwen3moe.expert_feed_forward_length u32              = 1536
llama_model_loader: - kv  22:               general.quantization_version u32              = 2
llama_model_loader: - kv  23:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  24:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  25:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  26:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  27:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv  28:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  29:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  30:                tokenizer.ggml.bos_token_id u32              = 151643
llama_model_loader: - kv  31:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  32:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - type  f32:  471 tensors
llama_model_loader: - type q8_0:  660 tensors
llm_load_vocab: special tokens cache size = 26
llm_load_vocab: token to piece cache size = 0.9311 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = qwen3moe
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 151936
llm_load_print_meta: n_merges         = 151387
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 40960
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_layer          = 94
llm_load_print_meta: n_head           = 64
llm_load_print_meta: n_head_kv        = 4
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 16
llm_load_print_meta: n_embd_k_gqa     = 512
llm_load_print_meta: n_embd_v_gqa     = 512
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 12288
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
llm_load_print_meta: model ftype      = Q8_0
llm_load_print_meta: model params     = 235.094 B
llm_load_print_meta: model size       = 232.769 GiB (8.505 BPW) 
llm_load_print_meta: repeating layers = 231.538 GiB (8.505 BPW, 233.849 B parameters)
llm_load_print_meta: general.name     = Qwen3 235B A22B
llm_load_print_meta: BOS token        = 151643 '<|endoftext|>'
llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
llm_load_print_meta: PAD token        = 151643 '<|endoftext|>'
llm_load_print_meta: LF token         = 148848 'ÄĬ'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp         = 1536
llm_load_tensors: ggml ctx size =    0.99 MiB
Tensor blk.14.ffn_norm.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CPU
.
.
.
Tensor blk.93.ffn_norm.weight buffer type overriden to CPU
Tensor blk.93.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.93.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 94 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 95/95 layers to GPU
llm_load_tensors:        CPU buffer size = 196001.25 MiB
llm_load_tensors:  CUDA_Host buffer size =   630.59 MiB
llm_load_tensors:      CUDA0 buffer size = 41723.89 MiB
....................................................................................................
============ Repacked 240 tensors
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
llama_kv_cache_init:      CUDA0 KV buffer size =  3196.05 MiB
llama_new_context_with_model: KV self size  = 3196.00 MiB, K (q8_0): 1598.00 MiB, V (q8_0): 1598.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.58 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =   312.75 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   128.01 MiB
llama_new_context_with_model: graph nodes  = 3672
llama_new_context_with_model: graph splits = 322

main: n_kv_max = 32768, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = 99, n_threads = 24, n_threads_batch = 24
```

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    3.120 |   164.13 |   12.045 |    10.63 |
|   512 |    128 |    512 |    3.232 |   158.43 |   12.124 |    10.56 |
|   512 |    128 |   1024 |    3.090 |   165.69 |   12.246 |    10.45 |
|   512 |    128 |   1536 |    3.367 |   152.06 |   12.028 |    10.64 |
|   512 |    128 |   2048 |    3.146 |   162.74 |   12.325 |    10.39 |
|   512 |    128 |   2560 |    3.160 |   162.01 |   12.146 |    10.54 |
|   512 |    128 |   3072 |    3.501 |   146.26 |   12.181 |    10.51 |
|   512 |    128 |   3584 |    3.115 |   164.36 |   12.193 |    10.50 |
|   512 |    128 |   4096 |    3.163 |   161.88 |   12.252 |    10.45 |
|   512 |    128 |   4608 |    3.151 |   162.50 |   12.467 |    10.27 |
|   512 |    128 |   5120 |    3.156 |   162.24 |   12.366 |    10.35 |
|   512 |    128 |   5632 |    3.220 |   159.01 |   12.665 |    10.11 |
|   512 |    128 |   6144 |    3.186 |   160.70 |   12.558 |    10.19 |
|   512 |    128 |   6656 |    3.198 |   160.11 |   12.734 |    10.05 |
|   512 |    128 |   7168 |    3.501 |   146.26 |   12.618 |    10.14 |
|   512 |    128 |   7680 |    3.267 |   156.74 |   12.704 |    10.08 |
|   512 |    128 |   8192 |    3.250 |   157.56 |   12.718 |    10.06 |
|   512 |    128 |   8704 |    3.258 |   157.15 |   12.887 |     9.93 |
|   512 |    128 |   9216 |    3.279 |   156.12 |   12.802 |    10.00 |
|   512 |    128 |   9728 |    3.427 |   149.38 |   12.825 |     9.98 |
|   512 |    128 |  10240 |    3.330 |   153.74 |   12.848 |     9.96 |
|   512 |    128 |  10752 |    3.639 |   140.70 |   12.982 |     9.86 |
|   512 |    128 |  11264 |    3.300 |   155.17 |   13.083 |     9.78 |
|   512 |    128 |  11776 |    3.543 |   144.51 |   13.104 |     9.77 |
|   512 |    128 |  12288 |    3.437 |   148.99 |   13.078 |     9.79 |
|   512 |    128 |  12800 |    3.473 |   147.42 |   13.164 |     9.72 |
|   512 |    128 |  13312 |    3.330 |   153.75 |   13.247 |     9.66 |
|   512 |    128 |  13824 |    3.347 |   152.98 |   13.190 |     9.70 |
|   512 |    128 |  14336 |    3.357 |   152.53 |   13.398 |     9.55 |
|   512 |    128 |  14848 |    3.357 |   152.52 |   13.296 |     9.63 |
|   512 |    128 |  15360 |    3.502 |   146.21 |   13.476 |     9.50 |
|   512 |    128 |  15872 |    3.475 |   147.33 |   13.364 |     9.58 |
|   512 |    128 |  16384 |    3.372 |   151.84 |   13.651 |     9.38 |
|   512 |    128 |  16896 |    3.372 |   151.84 |   13.507 |     9.48 |
|   512 |    128 |  17408 |    3.400 |   150.57 |   13.666 |     9.37 |
|   512 |    128 |  17920 |    3.419 |   149.77 |   13.615 |     9.40 |
|   512 |    128 |  18432 |    3.467 |   147.68 |   13.737 |     9.32 |
|   512 |    128 |  18944 |    3.432 |   149.19 |   13.663 |     9.37 |
|   512 |    128 |  19456 |    3.442 |   148.74 |   13.804 |     9.27 |
|   512 |    128 |  19968 |    3.462 |   147.88 |   13.756 |     9.31 |
|   512 |    128 |  20480 |    3.451 |   148.35 |   13.920 |     9.20 |
|   512 |    128 |  20992 |    3.469 |   147.59 |   13.851 |     9.24 |
|   512 |    128 |  21504 |    3.485 |   146.91 |   14.089 |     9.08 |
|   512 |    128 |  22016 |    3.497 |   146.41 |   14.044 |     9.11 |
|   512 |    128 |  22528 |    3.507 |   146.01 |   14.086 |     9.09 |
|   512 |    128 |  23040 |    3.511 |   145.84 |   14.040 |     9.12 |
|   512 |    128 |  23552 |    3.702 |   138.31 |   14.251 |     8.98 |
|   512 |    128 |  24064 |    3.919 |   130.66 |   14.129 |     9.06 |
|   512 |    128 |  24576 |    3.656 |   140.04 |   14.210 |     9.01 |
|   512 |    128 |  25088 |    4.069 |   125.84 |   14.330 |     8.93 |
|   512 |    128 |  25600 |    3.539 |   144.67 |   14.242 |     8.99 |
|   512 |    128 |  26112 |    3.579 |   143.07 |   14.357 |     8.92 |
|   512 |    128 |  26624 |    3.563 |   143.70 |   14.370 |     8.91 |
|   512 |    128 |  27136 |    3.619 |   141.48 |   14.677 |     8.72 |
|   512 |    128 |  27648 |    3.592 |   142.55 |   14.492 |     8.83 |
|   512 |    128 |  28160 |    3.589 |   142.66 |   14.715 |     8.70 |
|   512 |    128 |  28672 |    3.611 |   141.79 |   14.591 |     8.77 |
|   512 |    128 |  29184 |    3.612 |   141.73 |   14.741 |     8.68 |
|   512 |    128 |  29696 |    3.618 |   141.51 |   14.655 |     8.73 |
|   512 |    128 |  30208 |    3.716 |   137.80 |   14.820 |     8.64 |
|   512 |    128 |  30720 |    3.637 |   140.78 |   14.624 |     8.75 |
|   512 |    128 |  31232 |    3.729 |   137.31 |   14.793 |     8.65 |
|   512 |    128 |  31744 |    3.694 |   138.59 |   14.731 |     8.69 |
|   512 |    128 |  32256 |    3.732 |   137.20 |   14.901 |     8.59 |

</details>

---

👤 **ubergarm** commented the **2025-05-04** at **04:41:15**:<br>

Finally, I also tested this PR to ensure the models were still actually working in addition to being faster. I used this PR + my [ubergarm/Qwen3-30B-A3B-mix-IQ4_K](https://huggingface.co/ubergarm/Qwen3-30B-A3B-GGUF) to vibe code up the imatrix-statistics visualization scripts to parse and and plot data the stats: https://gist.github.com/ubergarm/2aa9327f7b98a9b16fef62b4941c7e76

So anecdotally the model still seems to work fine fwiw. Cheers and g'night!

---

👤 **ikawrakow** commented the **2025-05-04** at **06:17:33**:<br>

Thank you for these results and for testing!

Mainline has become faster for prompt processing with `bartowski/Qwen3-30B-A3B-Q4_K_M` fully offloaded to the GPU only after [this recent mainline PR](https://github.com/ggml-org/llama.cpp/pull/13199). The PR does a better job at implementing experts matrix multiplication than what I have done with `-fmoe`. But I think the `-fmoe` implementation may still be better when there is more than one expert.

In any case, this PR looks like a winner, so merging.

---

👤 **ubergarm** commented the **2025-05-04** at **17:08:14**:<br>

Amazing work y'all! I did a little post to let folks know its time to `git pull` and rebuild to take advantage of all the improvements!

https://www.reddit.com/r/LocalLLaMA/comments/1keoint/llama_gotta_go_fast_both_ik_and_mainline_llamacpp/