### 📝 [#629](https://github.com/ikawrakow/ik_llama.cpp/issues/629) - Multi-GPU performance (Windows) is significantly worse than single-GPU

| **Author** | `sousekd` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2025-07-18 |
| **Updated** | 2025-07-19 |

---

#### Description

Testing on a single NPS1 Epyc 9355 system equipped with an RTX 5090 and an RTX 4090, I observe slightly lower PP t/s and much lower TG t/s when both GPUs are enabled compared with using just one.

I suspect the problem is related either to the absence of GPU P2P or to some other Windows-specific factor. I'll soon switch to Linux and don't intend to use multiple GPUs for inference, so this doesn't affect me personally, but I'm curious about the cause - it may matter to other users.

Below are benchmarks for Qwen-235B, @ubergarm's IQ3_K, and bartowski's Q8_0, but I observed very similar results for DeepSeek models as well. In each case I offload as many layers as possible to each GPU; the exact command-line arguments are in the attached logs.

As the charts show, the multi-GPU setup delivers roughly the same PP t/s as the RTX 5090-only setup when running IQ3_K, and roughly the same PP t/s as the RTX 4090-only setup when running Q8_0, where the RTX 5090-only configuration actually performs better.

**For TG t/s, however, the multi-GPU setup is universally worse.**

<img width="2379" height="1180" alt="Image" src="https://github.com/user-attachments/assets/60963385-c90c-4ad7-93c0-daa8cc6f2a53" />

<img width="2379" height="1180" alt="Image" src="https://github.com/user-attachments/assets/887761ca-b701-46be-a0bf-9ce7cbee1543" />


<details>
<summary>ik_llama.cpp build command</summary>

```
$env:CC  = "clang-cl"
$env:CXX = "clang-cl"

cmake -B build -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_C_COMPILER="$env:CC" `
  -DCMAKE_CXX_COMPILER="$env:CXX" `
  -DCMAKE_CUDA_HOST_COMPILER="cl.exe" `
  -DGGML_CUDA=ON `
  -DGGML_AVX512=ON `
  -DGGML_AVX512_VNNI=ON `
  -DGGML_AVX512_VBMI=ON `
  -DGGML_AVX512_BF16=ON `
  -DGGML_SCHED_MAX_COPIES=1 `
  -DGGML_BLAS=OFF `
  -DGGML_CCACHE=OFF `
  -DCMAKE_C_FLAGS='/clang:-march=znver5' `
  -DCMAKE_CXX_FLAGS='/EHsc /clang:-march=znver5' `
  -DCMAKE_CUDA_ARCHITECTURES="89-real;120-real" `
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON `
  -DLLAMA_CURL=OFF `
  -DBUILD_SHARED_LIBS=OFF
```

</details>

<details>
<summary>ubergarm_Qwen3-235B-A22B-mix-IQ3_K-multi</summary>

```
PS> .\bin\llama-server --version
version: 3772 (5236c98b)
built with Clang 19.1.5 for
PS> .\bin\llama-sweep-bench.exe 
    --alias ubergarm/Qwen3-235B-A22B-mix-IQ3_K 
    --model C:\Users\Administrator\.lmstudio\models\ubergarm\Qwen3-235B-A22B-GGUF\Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf 
    --no-mmap -rtr -fa -fmoe  
    -c 32768 -amb 512 -b 4096 -ub 2048 
    -ngl 999  -ot "blk\.([0-9]|1[0-9])\.ffn_.*=CUDA0" -ot "blk\.(2[0-9]|3[0-3])\.ffn_.*=CUDA1" -ot "blk\.[0-9]+\.ffn.*=CPU" 
    --parallel 1 --threads 32 
    --main-gpu 0 
    --warmup-batch
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
  Device 1: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
llama_model_loader: additional 2 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 40 key-value pairs and 1131 tensors from C:\Users\Administrator\.lms
tudio\models\ubergarm\Qwen3-235B-A22B-GGUF\Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf (version GGUF V3 (lat
est))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3 235B A22B
llama_model_loader: - kv   3:                           general.basename str              = Qwen3
llama_model_loader: - kv   4:                         general.size_label str              = 235B-A22B
llama_model_loader: - kv   5:                            general.license str              = apache-2.0
llama_model_loader: - kv   6:                       general.license.link str              = https://huggingfac
e.co/Qwen/Qwen3-235...
llama_model_loader: - kv   7:                               general.tags arr[str,1]       = ["text-generation"
]
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
llama_model_loader: - kv  25:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "
$", "%", "&", "'", ...
llama_model_loader: - kv  26:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  27:                      tokenizer.ggml.merges arr[str,151387]  = ["─á ─á", "─á─á ─á
─á", "i n", "─á t",...
llama_model_loader: - kv  28:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  29:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  30:                tokenizer.ggml.bos_token_id u32              = 151643
llama_model_loader: - kv  31:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  32:                    tokenizer.chat_template str              = {%- if tools %}\n
   {{- '<|im_start|>...
llama_model_loader: - kv  33:                      quantize.imatrix.file str              = /mnt/raid/models/u
bergarm/Qwen3-235B-...
llama_model_loader: - kv  34:                   quantize.imatrix.dataset str              = calibration_data_v
5_rc.txt
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
llm_load_print_meta: LF token         = 148848 '├ä─¼'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp         = 1536
llm_load_tensors: ggml ctx size =    1.49 MiB
Tensor blk.0.ffn_norm.weight buffer type overriden to CUDA0
[...0-19 to CUDA0...]
Tensor blk.19.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.20.ffn_norm.weight buffer type overriden to CUDA1
[...20-33 to CUDA1...]
Tensor blk.33.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.34.ffn_norm.weight buffer type overriden to CPU
[...rest to CPU...]
llm_load_tensors: offloading 94 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 95/95 layers to GPU
llm_load_tensors:  CUDA_Host buffer size =   630.59 MiB
llm_load_tensors:        CPU buffer size = 65640.94 MiB
llm_load_tensors:      CUDA0 buffer size = 24978.41 MiB
llm_load_tensors:      CUDA1 buffer size = 18143.66 MiB
....................................................................................................
============ Repacked 180 tensors
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 4096
llama_new_context_with_model: n_ubatch   = 2048
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 512
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:      CUDA0 KV buffer size =  3520.00 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =  2496.00 MiB
llama_new_context_with_model: KV self size  = 6016.00 MiB, K (f16): 3008.00 MiB, V (f16): 3008.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.58 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      CUDA0 compute buffer size =  1264.01 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =  1251.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   288.02 MiB
llama_new_context_with_model: graph nodes  = 3672
llama_new_context_with_model: graph splits = 310

main: n_kv_max = 32768, n_batch = 4096, n_ubatch = 2048, flash_attn = 1, n_gpu_layers = 999, n_threads = 32, n
_threads_batch = 32

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  2048 |    512 |      0 |    5.522 |   370.86 |   31.733 |    16.13 |
|  2048 |    512 |   2048 |    5.827 |   351.45 |   32.446 |    15.78 |
|  2048 |    512 |   4096 |    5.786 |   353.96 |   32.979 |    15.52 |
|  2048 |    512 |   6144 |    5.819 |   351.96 |   31.916 |    16.04 |
|  2048 |    512 |   8192 |    6.014 |   340.54 |   33.722 |    15.18 |
|  2048 |    512 |  10240 |    6.084 |   336.61 |   34.216 |    14.96 |
|  2048 |    512 |  12288 |    6.260 |   327.15 |   34.437 |    14.87 |
|  2048 |    512 |  14336 |    6.311 |   324.53 |   35.951 |    14.24 |
|  2048 |    512 |  16384 |    6.503 |   314.94 |   35.322 |    14.50 |
|  2048 |    512 |  18432 |    6.494 |   315.37 |   35.579 |    14.39 |
|  2048 |    512 |  20480 |    6.647 |   308.11 |   35.238 |    14.53 |
|  2048 |    512 |  22528 |    6.745 |   303.65 |   35.927 |    14.25 |
|  2048 |    512 |  24576 |    6.712 |   305.13 |   36.214 |    14.14 |
|  2048 |    512 |  26624 |    6.845 |   299.20 |   36.340 |    14.09 |
|  2048 |    512 |  28672 |    6.704 |   305.51 |   36.698 |    13.95 |
|  2048 |    512 |  30720 |    6.960 |   294.24 |   36.758 |    13.93 |
```

</details>

<details>
<summary>ubergarm_Qwen3-235B-A22B-mix-IQ3_K-5090</summary>

```
PS> .\bin\llama-server --version
version: 3772 (5236c98b)
built with Clang 19.1.5 for
PS> .\bin\llama-sweep-bench.exe 
    --alias ubergarm/Qwen3-235B-A22B-mix-IQ3_K 
    --model C:\Users\Administrator\.lmstudio\models\ubergarm\Qwen3-235B-A22B-GGUF\Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf 
    --no-mmap -rtr -fa -fmoe  
    -c 32768 -amb 512 -b 4096 -ub 2048 
    -ngl 999  -ot "blk\.([0-9]|1[0-5])\.ffn_.*=CUDA0" -ot "blk\.[0-9]+\.ffn.*=CPU" 
    --parallel 1 --threads 32 
    --warmup-batch
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
llama_model_loader: additional 2 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 40 key-value pairs and 1131 tensors from C:\Users\Administrator\.lms
tudio\models\ubergarm\Qwen3-235B-A22B-GGUF\Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf (version GGUF V3 (lat
est))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3 235B A22B
llama_model_loader: - kv   3:                           general.basename str              = Qwen3
llama_model_loader: - kv   4:                         general.size_label str              = 235B-A22B
llama_model_loader: - kv   5:                            general.license str              = apache-2.0
llama_model_loader: - kv   6:                       general.license.link str              = https://huggingfac
e.co/Qwen/Qwen3-235...
llama_model_loader: - kv   7:                               general.tags arr[str,1]       = ["text-generation"
]
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
llama_model_loader: - kv  25:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "
$", "%", "&", "'", ...
llama_model_loader: - kv  26:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  27:                      tokenizer.ggml.merges arr[str,151387]  = ["─á ─á", "─á─á ─á
─á", "i n", "─á t",...
llama_model_loader: - kv  28:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  29:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  30:                tokenizer.ggml.bos_token_id u32              = 151643
llama_model_loader: - kv  31:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  32:                    tokenizer.chat_template str              = {%- if tools %}\n
   {{- '<|im_start|>...
llama_model_loader: - kv  33:                      quantize.imatrix.file str              = /mnt/raid/models/u
bergarm/Qwen3-235B-...
llama_model_loader: - kv  34:                   quantize.imatrix.dataset str              = calibration_data_v
5_rc.txt
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
llm_load_print_meta: LF token         = 148848 '├ä─¼'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp         = 1536
llm_load_tensors: ggml ctx size =    0.99 MiB
Tensor blk.0.ffn_norm.weight buffer type overriden to CUDA0
[...0-15 to CUDA0...]
Tensor blk.15.ffn_up_exps.weight buffer type overriden to CUDA0
[...rest to CPU...]
llm_load_tensors: offloading 94 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 95/95 layers to GPU
llm_load_tensors:  CUDA_Host buffer size =   630.59 MiB
llm_load_tensors:        CPU buffer size = 85333.22 MiB
llm_load_tensors:      CUDA0 buffer size = 23429.79 MiB
....................................................................................................
============ Repacked 234 tensors
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 4096
llama_new_context_with_model: n_ubatch   = 2048
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 512
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:      CUDA0 KV buffer size =  6016.00 MiB
llama_new_context_with_model: KV self size  = 6016.00 MiB, K (f16): 3008.00 MiB, V (f16): 3008.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.58 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =  1264.01 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   288.02 MiB
llama_new_context_with_model: graph nodes  = 3672
llama_new_context_with_model: graph splits = 314

main: n_kv_max = 32768, n_batch = 4096, n_ubatch = 2048, flash_attn = 1, n_gpu_layers = 999, n_threads = 32, n
_threads_batch = 32

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  2048 |    512 |      0 |    5.747 |   356.39 |   22.790 |    22.47 |
|  2048 |    512 |   2048 |    5.776 |   354.56 |   22.611 |    22.64 |
|  2048 |    512 |   4096 |    5.847 |   350.29 |   22.861 |    22.40 |
|  2048 |    512 |   6144 |    5.999 |   341.38 |   23.027 |    22.23 |
|  2048 |    512 |   8192 |    6.054 |   338.28 |   23.567 |    21.73 |
|  2048 |    512 |  10240 |    6.047 |   338.66 |   24.076 |    21.27 |
|  2048 |    512 |  12288 |    6.183 |   331.23 |   24.044 |    21.29 |
|  2048 |    512 |  14336 |    6.216 |   329.46 |   24.511 |    20.89 |
|  2048 |    512 |  16384 |    6.296 |   325.27 |   24.262 |    21.10 |
|  2048 |    512 |  18432 |    6.370 |   321.50 |   24.298 |    21.07 |
|  2048 |    512 |  20480 |    6.431 |   318.47 |   24.882 |    20.58 |
|  2048 |    512 |  22528 |    6.494 |   315.39 |   25.508 |    20.07 |
|  2048 |    512 |  24576 |    6.545 |   312.92 |   25.480 |    20.09 |
|  2048 |    512 |  26624 |    6.560 |   312.21 |   25.985 |    19.70 |
|  2048 |    512 |  28672 |    6.661 |   307.44 |   25.826 |    19.83 |
|  2048 |    512 |  30720 |    6.691 |   306.09 |   25.709 |    19.92 |
```

</details>

<details>
<summary>bartowski_Qwen3-235B-A22B-Q8_0-multi</summary>

```
PS> .\bin\llama-server --version
version: 3772 (5236c98b)
built with Clang 19.1.5 for
PS> .\bin\llama-sweep-bench.exe 
    --alias bartowski/Qwen3-235B-A22B-Q8_0 
    --model C:\Users\Administrator\.lmstudio\models\lmstudio-community\Qwen3-235B-A22B-GGUF\Qwen3-235B-A22B-Q8_0-00001-of-00007.gguf 
    --no-mmap -rtr -fa -fmoe  
    -c 32768 -amb 512 -b 4096 -ub 2048 
    -ngl 999  -ot "blk\.[0-8]\.ffn_.*=CUDA0" -ot "blk\.(9|1[0-4])\.ffn_.*=CUDA1" -ot "blk\.[0-9]+\.ffn.*=CPU" 
    --parallel 1 --threads 32 
    --main-gpu 0 
    --warmup-batch
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
  Device 1: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
llama_model_loader: additional 6 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 36 key-value pairs and 1131 tensors from C:\Users\Administrator\.lms
tudio\models\lmstudio-community\Qwen3-235B-A22B-GGUF\Qwen3-235B-A22B-Q8_0-00001-of-00007.gguf (version GGUF V3
 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3 235B A22B
llama_model_loader: - kv   3:                           general.basename str              = Qwen3
llama_model_loader: - kv   4:                         general.size_label str              = 235B-A22B
llama_model_loader: - kv   5:                            general.license str              = apache-2.0
llama_model_loader: - kv   6:                       general.license.link str              = https://huggingfac
e.co/Qwen/Qwen3-235...
llama_model_loader: - kv   7:                               general.tags arr[str,1]       = ["text-generation"
]
llama_model_loader: - kv   8:                       qwen3moe.block_count u32              = 94
llama_model_loader: - kv   9:                    qwen3moe.context_length u32              = 32768
llama_model_loader: - kv  10:                  qwen3moe.embedding_length u32              = 4096
llama_model_loader: - kv  11:               qwen3moe.feed_forward_length u32              = 12288
llama_model_loader: - kv  12:              qwen3moe.attention.head_count u32              = 64
llama_model_loader: - kv  13:           qwen3moe.attention.head_count_kv u32              = 4
llama_model_loader: - kv  14:                    qwen3moe.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  15:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  16:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  17:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  18:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  19:                      qwen3moe.expert_count u32              = 128
llama_model_loader: - kv  20:        qwen3moe.expert_feed_forward_length u32              = 1536
llama_model_loader: - kv  21:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  22:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  23:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "
$", "%", "&", "'", ...
llama_model_loader: - kv  24:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  25:                      tokenizer.ggml.merges arr[str,151387]  = ["─á ─á", "─á─á ─á
─á", "i n", "─á t",...
llama_model_loader: - kv  26:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  27:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  28:                tokenizer.ggml.bos_token_id u32              = 151643
llama_model_loader: - kv  29:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  30:                    tokenizer.chat_template str              = {%- if tools %}\n
   {{- '<|im_start|>...
llama_model_loader: - kv  31:               general.quantization_version u32              = 2
llama_model_loader: - kv  32:                          general.file_type u32              = 7
llama_model_loader: - kv  33:                                   split.no u16              = 0
llama_model_loader: - kv  34:                        split.tensors.count i32              = 1131
llama_model_loader: - kv  35:                                split.count u16              = 7
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
llm_load_print_meta: n_ctx_train      = 32768
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
llm_load_print_meta: n_ctx_orig_yarn  = 32768
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
llm_load_print_meta: LF token         = 148848 '├ä─¼'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp         = 1536
llm_load_tensors: ggml ctx size =    1.49 MiB
Tensor blk.0.ffn_norm.weight buffer type overriden to CUDA0
[...0-8 to CUDA0...]
Tensor blk.8.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.9.ffn_norm.weight buffer type overriden to CUDA1
[...9-14 to CUDA1...]
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.15.ffn_norm.weight buffer type overriden to CPU
[...rest to CPU...]
llm_load_tensors: offloading 94 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 95/95 layers to GPU
llm_load_tensors:  CUDA_Host buffer size =   630.59 MiB
llm_load_tensors:        CPU buffer size = 193551.23 MiB
llm_load_tensors:      CUDA0 buffer size = 26024.80 MiB
llm_load_tensors:      CUDA1 buffer size = 18149.10 MiB
....................................................................................................
============ Repacked 237 tensors
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 4096
llama_new_context_with_model: n_ubatch   = 2048
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 512
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:      CUDA0 KV buffer size =  3520.00 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =  2496.00 MiB
llama_new_context_with_model: KV self size  = 6016.00 MiB, K (f16): 3008.00 MiB, V (f16): 3008.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.58 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      CUDA0 compute buffer size =   832.01 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =  1251.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   512.02 MiB
llama_new_context_with_model: graph nodes  = 3672
llama_new_context_with_model: graph splits = 330

main: n_kv_max = 32768, n_batch = 4096, n_ubatch = 2048, flash_attn = 1, n_gpu_layers = 999, n_threads = 32, n
_threads_batch = 32

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  2048 |    512 |      0 |    6.596 |   310.51 |   43.222 |    11.85 |
|  2048 |    512 |   2048 |    6.517 |   314.24 |   44.251 |    11.57 |
|  2048 |    512 |   4096 |    6.724 |   304.58 |   44.600 |    11.48 |
|  2048 |    512 |   6144 |    6.794 |   301.44 |   45.582 |    11.23 |
|  2048 |    512 |   8192 |    6.935 |   295.30 |   46.255 |    11.07 |
|  2048 |    512 |  10240 |    6.857 |   298.67 |   46.837 |    10.93 |
|  2048 |    512 |  12288 |    7.092 |   288.78 |   47.158 |    10.86 |
|  2048 |    512 |  14336 |    7.346 |   278.78 |   47.718 |    10.73 |
|  2048 |    512 |  16384 |    7.487 |   273.56 |   47.775 |    10.72 |
|  2048 |    512 |  18432 |    7.267 |   281.81 |   48.049 |    10.66 |
|  2048 |    512 |  20480 |    7.133 |   287.12 |   48.458 |    10.57 |
|  2048 |    512 |  22528 |    7.163 |   285.90 |   49.036 |    10.44 |
|  2048 |    512 |  24576 |    7.243 |   282.77 |   49.195 |    10.41 |
|  2048 |    512 |  26624 |    7.053 |   290.37 |   48.996 |    10.45 |
|  2048 |    512 |  28672 |    7.591 |   269.78 |   49.566 |    10.33 |
|  2048 |    512 |  30720 |    8.018 |   255.42 |   49.734 |    10.29 |
```

</details>

<details>
<summary>bartowski_Qwen3-235B-A22B-Q8_0-5090</summary>

```
PS> .\bin\llama-server --version
version: 3772 (5236c98b)
built with Clang 19.1.5 for
PS> .\bin\llama-sweep-bench.exe 
    --alias bartowski/Qwen3-235B-A22B-Q8_0 
    --model C:\Users\Administrator\.lmstudio\models\lmstudio-community\Qwen3-235B-A22B-GGUF\Qwen3-235B-A22B-Q8_0-00001-of-00007.gguf 
    --no-mmap -rtr -fa -fmoe  
    -c 32768 -amb 512 -b 4096 -ub 2048 
    -ngl 999  -ot "blk\.[0-5]\.ffn_.*=CUDA0" -ot "blk\.[0-9]+\.ffn.*=CPU" 
    --parallel 1 --threads 32 
    --warmup-batch
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
llama_model_loader: additional 6 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 36 key-value pairs and 1131 tensors from C:\Users\Administrator\.lms
tudio\models\lmstudio-community\Qwen3-235B-A22B-GGUF\Qwen3-235B-A22B-Q8_0-00001-of-00007.gguf (version GGUF V3
 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3 235B A22B
llama_model_loader: - kv   3:                           general.basename str              = Qwen3
llama_model_loader: - kv   4:                         general.size_label str              = 235B-A22B
llama_model_loader: - kv   5:                            general.license str              = apache-2.0
llama_model_loader: - kv   6:                       general.license.link str              = https://huggingfac
e.co/Qwen/Qwen3-235...
llama_model_loader: - kv   7:                               general.tags arr[str,1]       = ["text-generation"
]
llama_model_loader: - kv   8:                       qwen3moe.block_count u32              = 94
llama_model_loader: - kv   9:                    qwen3moe.context_length u32              = 32768
llama_model_loader: - kv  10:                  qwen3moe.embedding_length u32              = 4096
llama_model_loader: - kv  11:               qwen3moe.feed_forward_length u32              = 12288
llama_model_loader: - kv  12:              qwen3moe.attention.head_count u32              = 64
llama_model_loader: - kv  13:           qwen3moe.attention.head_count_kv u32              = 4
llama_model_loader: - kv  14:                    qwen3moe.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  15:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  16:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  17:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  18:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  19:                      qwen3moe.expert_count u32              = 128
llama_model_loader: - kv  20:        qwen3moe.expert_feed_forward_length u32              = 1536
llama_model_loader: - kv  21:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  22:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  23:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "
$", "%", "&", "'", ...
llama_model_loader: - kv  24:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  25:                      tokenizer.ggml.merges arr[str,151387]  = ["─á ─á", "─á─á ─á
─á", "i n", "─á t",...
llama_model_loader: - kv  26:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  27:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  28:                tokenizer.ggml.bos_token_id u32              = 151643
llama_model_loader: - kv  29:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  30:                    tokenizer.chat_template str              = {%- if tools %}\n
   {{- '<|im_start|>...
llama_model_loader: - kv  31:               general.quantization_version u32              = 2
llama_model_loader: - kv  32:                          general.file_type u32              = 7
llama_model_loader: - kv  33:                                   split.no u16              = 0
llama_model_loader: - kv  34:                        split.tensors.count i32              = 1131
llama_model_loader: - kv  35:                                split.count u16              = 7
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
llm_load_print_meta: n_ctx_train      = 32768
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
llm_load_print_meta: n_ctx_orig_yarn  = 32768
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
llm_load_print_meta: LF token         = 148848 '├ä─¼'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp         = 1536
llm_load_tensors: ggml ctx size =    0.99 MiB
Tensor blk.0.ffn_norm.weight buffer type overriden to CUDA0
[...0-5 to CUDA0...]
Tensor blk.5.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_norm.weight buffer type overriden to CPU
[...rest to CPU...]
llm_load_tensors: offloading 94 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 95/95 layers to GPU
llm_load_tensors:  CUDA_Host buffer size =   630.59 MiB
llm_load_tensors:        CPU buffer size = 215601.38 MiB
llm_load_tensors:      CUDA0 buffer size = 22123.76 MiB
....................................................................................................
============ Repacked 264 tensors
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 4096
llama_new_context_with_model: n_ubatch   = 2048
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 512
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:      CUDA0 KV buffer size =  6016.00 MiB
llama_new_context_with_model: KV self size  = 6016.00 MiB, K (f16): 3008.00 MiB, V (f16): 3008.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.58 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =  1251.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   512.02 MiB
llama_new_context_with_model: graph nodes  = 3672
llama_new_context_with_model: graph splits = 354

main: n_kv_max = 32768, n_batch = 4096, n_ubatch = 2048, flash_attn = 1, n_gpu_layers = 999, n_threads = 32, n
_threads_batch = 32

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  2048 |    512 |      0 |    5.832 |   351.16 |   31.514 |    16.25 |
|  2048 |    512 |   2048 |    6.388 |   320.60 |   31.760 |    16.12 |
|  2048 |    512 |   4096 |    6.012 |   340.63 |   32.756 |    15.63 |
|  2048 |    512 |   6144 |    5.906 |   346.76 |   32.283 |    15.86 |
|  2048 |    512 |   8192 |    5.960 |   343.63 |   32.414 |    15.80 |
|  2048 |    512 |  10240 |    6.084 |   336.64 |   32.778 |    15.62 |
|  2048 |    512 |  12288 |    6.173 |   331.75 |   32.884 |    15.57 |
|  2048 |    512 |  14336 |    6.305 |   324.84 |   33.211 |    15.42 |
|  2048 |    512 |  16384 |    6.892 |   297.14 |   33.712 |    15.19 |
|  2048 |    512 |  18432 |    6.643 |   308.29 |   33.624 |    15.23 |
|  2048 |    512 |  20480 |    6.886 |   297.40 |   34.327 |    14.92 |
|  2048 |    512 |  22528 |    6.753 |   303.27 |   34.457 |    14.86 |
|  2048 |    512 |  24576 |    6.507 |   314.75 |   34.359 |    14.90 |
|  2048 |    512 |  26624 |    7.039 |   290.93 |   34.675 |    14.77 |
|  2048 |    512 |  28672 |    6.715 |   304.98 |   34.370 |    14.90 |
|  2048 |    512 |  30720 |    7.123 |   287.50 |   35.114 |    14.58 |
```

</details>

<details>
<summary>nvidia-smi -q</summary>

```
Driver Version : 576.80          CUDA Version : 12.9
Attached GPUs  : 2

GPU 0 – 00000000:01:00.0  (NVIDIA GeForce RTX 5090  – Blackwell)
  Driver Model       : WDDM
  PCIe Gen | Width   : Current 1 ×16   (Max 5 ×16  Host Max 5)
  BAR1 Memory Usage  : 32 768 MiB Total  • 32 740 MiB Used   28 MiB Free
  FB  Memory Usage   : 32 607 MiB Total  •    507 MiB Resvd   0 MiB Used
  Perf State         : P8
  Clocks (MHz)       : Gfx 24   SM 24   Mem 405   Vid 600
  Max   (MHz)        : Gfx 3090 SM 3090 Mem 14001
  Power Draw / Limit : 8 W  / 600 W  (Min 400  Max 600)

GPU 1 – 00000000:C1:00.0  (NVIDIA GeForce RTX 4090  – Ada Lovelace)
  Driver Model       : WDDM
  PCIe Gen | Width   : Current 4 ×16   (Max 4 ×16  Host Max 4)
  BAR1 Memory Usage  : 32 768 MiB Total  • 32 740 MiB Used   28 MiB Free
  FB  Memory Usage   : 24 564 MiB Total  •    422 MiB Resvd 217 MiB Used
  Perf State         : P0
  Clocks (MHz)       : Gfx 2520 SM 2520 Mem 10 501 Vid 1980
  Max   (MHz)        : Gfx 3105 SM 3105 Mem 10 501
  Power Draw / Limit : 54 W  / 450 W  (Min 150  Max 600)
```

</details>

<details>
<summary>p2pBandwidthLatencyTest.exe</summary>

```
[P2P (Peer-to-Peer) GPU Bandwidth Latency Test]
Device: 0, NVIDIA GeForce RTX 4090, pciBusID: c1, pciDeviceID: 0, pciDomainID:0
Device: 1, NVIDIA GeForce RTX 5090, pciBusID: 1, pciDeviceID: 0, pciDomainID:0
Device=0 CANNOT Access Peer Device=1
Device=1 CANNOT Access Peer Device=0

P2P Connectivity Matrix
     D\D     0     1
     0       1     0
     1       0     1

Unidirectional P2P=Disabled Bandwidth Matrix (GB/s)
   D\D     0      1
     0 909.49  20.65
     1  20.21 1545.69

Unidirectional P2P=Enabled Bandwidth (P2P Writes) Matrix (GB/s)
   D\D     0      1
     0 915.89  20.72
     1  20.28 1536.43

Bidirectional P2P=Disabled Bandwidth Matrix (GB/s)
   D\D     0      1
     0 920.23  31.70
     1  31.70 1541.09

Bidirectional P2P=Enabled Bandwidth Matrix (GB/s)
   D\D     0      1
     0 920.84  31.57
     1  31.83 1539.03

P2P=Disabled Latency Matrix (us)
   GPU     0      1
     0   2.62  46.09
     1  38.57   3.81

   CPU     0      1
     0   1.62   5.13
     1   3.72   1.66

P2P=Enabled Latency (P2P Writes) Matrix (us)
   GPU     0      1
     0   2.59  45.93
     1  38.46   3.60

   CPU     0      1
     0   1.58   3.43
     1   3.03   1.64
```

</details>

Any thoughts?

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-07-18** at **13:39:02**:<br>

I think there are too many graph splits.
Look for this line or similar
```
llama_new_context_with_model: graph splits = 310
```
For good TG performance you want to have as few graph splits as possible as each graph split requires synchronization and copying data from one device to another, which adds up to a measurable TG performance drop when there are many splits. 

For instance, for 1 GPU, I would try
```
-ngl 999 -ot "blk\.(1[6-9] | [2-9][0-9])\.ffn_.*_exps=CPU"
```
to keep **only the routed experts** in layers 16-93 on the CPU. I cannot run Qwen3-235-A22B with my hardware, but trying it on Qwen3-32B-A3B, I get a ~2% better TG performance with that compared to your override (58 vs 114 graph splits).

For two GPU's, I would try the following 2 approaches (not sure which will work better for TG)
1. Keep everything on `CUDA0` (which is hopefully the faster 5090), and only put as many routed experts as would fit on `CUDA1`. E.g.,
```
-ngl 999 -ot "blk\.(1[6-9] | 2[0-9])\.ffn_.*_exps=CUDA1,blk\.[3-9][0-9]\.ffn_.*_exps=CPU"
```
2. Only specify the routed experts that will stay on the CPU via `-ot "blk\.??\.ffn_.*_exps=CPU`, and distribute all remaining tensors between the two GPU's using, e.g.,`-ngl 999 -ts 60,40`

This will hopefully result in fewer graph splits and better TG performance.

I don't know if peer-to-peer copy works on your system, but if it doesn't, this is probably quite bad for TG performance because data copies from one GPU to another goes via `GPU1 -> CPU -> GPU2`, which adds quite a bit of extra latency. 

If one of the suggestions helps, please let us know as this would be useful to quite a few people.

---

👤 **ubergarm** commented the **2025-07-18** at **14:31:20**:<br>

fwiw someone was asking me about Qwen3-235B on ik fork with windows also saying they weren't getting the speed-ups they were expecting with multi-GPU


> I have a 12600k with 128GB DDR5 running at 4000mhz, along with a 24GB 3090, and a 16GB 3060ti
>
> I tried the Unsloth iq3 quant of Q3 235b, your version of it, hunyuan Q5, and Q3 30b a3b Q8. All of them have been notably slower in IK for me for some reason
>
> Here is an example of one of the commands I would run:
>
> llama-server.exe --port 12345 -m "E:\Tabby2\tabbyAPI\models\Qwen3-235B-A22B-UD-Q3_K_XL-00001-of-00003.gguf" -c 8192 -fa -t 16 -tb 16 -ngl 999 -ot "blk\.(18|19|20|21|22|23|24|25|26|27|28|29)\.ffn.*=CUDA1" -ot "blk\.(0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17)\.ffn.*=CUDA0" -ot "exps=CPU" --api-key cool_password --no-mmap -fmoe
>
> interesting fun fact, I get nearly the exact same PP/generation speed only loading attention to the GPU's, compared to all those extra layers as well
> 
> I am on CUDA 12.8
>
> I don't actually compile LCPP myself, But I did talk to a friend who does do his own builds, and I have rebuilt IK about four or five times with his recommended settings to make sure that flash attention is enabled on everything, it's using FP16 support on CPU, and some various other options. Through all of the different builds, I don't think I've really seen any change at all unfortunately 
> 
> As for the difference in naming, I'm sure there are more efficient ways that I could go about loading the models properly, but my main confusion is that I'm loading them identically in normal LCPP and IK, so I would expect the one that is currently faster to continue to be faster when optimizing and running a better 
>
> The reason that I'm not using the 4096 batch is because the prompt processing isn't as important as the generation speed for me on these tests, and in order for me to increase the batch size, I have to unload about six layers from the GPUs 
>
> GPU-1 is my slower 3060 TI, and GPU zero is my faster 3090. 
>
> -Sytan on BeaverAI Club Discord `#help` channel

They are going to test `-rtr` after some more discussion, but I'll point them here as well if they want to chime in or test as well.

---

👤 **sousekd** commented the **2025-07-18** at **14:40:14**:<br>

Hmmm, I see. Do I understand correctly, that with a typical layer of Qwen3 looking like this:

```
Tensor blk.#.ffn_norm.weight
Tensor blk.#.ffn_gate_inp.weight
Tensor blk.#.ffn_gate_exps.weight
Tensor blk.#.ffn_down_exps.weight
Tensor blk.#.ffn_up_exps.weight
```

You advice to keep **all** `norm` and `gate_inp` tensors on CUDA0 (fastest GPU), try to fit there as many `gate_exps`, `down_exps` and `up_exps` as possible, too, and then the rest `exps` to either CUDA1...X (from the fastest one to the slowest one) or CPU?

I'll try. I thought "splitting layers" was against the general advice and I haven't seen the `exps` mentioned in Qwen3 `-ot` regexps on Hugging Face and elsewhere.


> I don't know if peer-to-peer copy works on your system, but if it doesn't, this is probably quite bad for TG performance because data copies from one GPU to another goes via `GPU1 -> CPU -> GPU2`, which adds quite a bit of extra latency.

It doesn't. Nvidia blocks it for **consumer-level** cards in their Windows drivers. Not sure whether one needs to use an alternative drivers on Linux, too, or Nvidia's are fine, but there is no way to overcome this limit on Windows AFAIK.

---

👤 **Panchovix** commented the **2025-07-18** at **15:19:30**:<br>

As a user with 7 GPUs, I would say just use Linux (sadly or not, depending on what you like) for LLMs, as I feel there is something wrong on Windows related to threading and multiGPU.

I read some time ago on a main llamacpp issue something related to this.

For example https://github.com/ggml-org/llama.cpp/issues/6442#issuecomment-2035218406, it mentions that CUDA libraries on Windows are not the best and I tend to agree.

I haven't actually tested iklcpp on main windows, but on main llamacpp before moving mostly to Linux, I was getting:

I.e. on DeepSeek Q2_K_XL, offloading ~140GB to RAM and the rest to VRAM (I had 4 GPUs at that time):

- 5 t/s PP, 1.5 t/s TG on Windows
- 60 t/s PP, 7 t/s TG on Linux (Ubuntu at that time but moved to Fedora afterwards).

Nowadays I get about 3-5x times that PP and a bit more TG t/s.

Also another example with other backend (exllamav2):

Mistral 123B 6bpw, running fully on GPU, no Tensor Parallel

- 10-11 t/s on Windows
- 15-16 t/s on Linux

And with TP

- 11-12 t/s on Windows
- 22-24 t/s on Linux

I just found when running on a single GPU (for example a small model) or when using diffusion pipelines (txt2img, txt2vid, etc) speeds are pretty similar.

Also I know NVIDIA doesn't support nccl on Windows but I don't think affects lcpp/iklcpp, prob mostly distributed training and vllm.

---

👤 **sousekd** commented the **2025-07-18** at **19:51:54**:<br>

Your suggestion definitely helped, @ikawrakow.
I only experimented with @ubergarm's **Qwen3-235B-A22B-mix-IQ3_K**, as it is likely relevant to more users:

First, I tried manually overriding **only the exps tensors** (and a few others) to the CPU and CUDA1 while keeping everything else on CUDA0, using `-ot "blk\.[0-9]+\.ffn.*_exps=CPU" -ot .*=CUDA0` and similar. Unfortunately, it always failed with *"unable to allocate backend buffer"*. But at least I learned there are more tensors per layer than what I suggested above 😉.

So, I gave up and went the *tensor‑split* route. The `--main-gpu 0` option definitely has effect, as I needed `-ts 25,75` to fill both cards VRAM (32+24G), while adjusting `-ot "blk\.(???)\.ffn_.*_exps=CPU"` to find the sweet spot.

This resulted in better speeds than when offloading entire layers. TG is still slower in the multi-GPU setup than on a single GPU, but PP performance has improved.

<img width="1979" height="974" alt="Image" src="https://github.com/user-attachments/assets/8ea080b6-8bd0-4d37-aa53-0ebefd909f63" />

<img width="1979" height="974" alt="Image" src="https://github.com/user-attachments/assets/5ec4feed-8a48-4cf8-9d96-5796dfd7c887" />

I am not sure using a second GPU is worth it, though... at least on Windows, on a machine with fast-enough RAM.

<details>
<summary>RTX 5090 only, -ot "blk\.(1[5-9]|[2-9][0-9])\.ffn_.*_exps=CPU"</summary>

```
PS> .\bin\llama-server --version
version: 3772 (5236c98b)
built with Clang 19.1.5 for
PS> .\bin\llama-sweep-bench.exe 
    --alias ubergarm/Qwen3-235B-A22B-mix-IQ3_K 
    --model C:\Users\Administrator\.lmstudio\models\ubergarm\Qwen3-235B-A22B-GGUF\Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf 
    --no-mmap -rtr -fa -fmoe  
    -c 32768 -amb 512 -b 4096 -ub 2048 
    -ngl 999  -ot "blk\.(1[5-9]|[2-9][0-9])\.ffn_.*_exps=CPU" 
    --parallel 1 --threads 32 
    --main-gpu 0 
    --warmup-batch
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
llama_model_loader: additional 2 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 40 key-value pairs and 1131 tensors from C:\Users\Administrator\.lms
tudio\models\ubergarm\Qwen3-235B-A22B-GGUF\Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf (version GGUF V3 (lat
est))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3 235B A22B
llama_model_loader: - kv   3:                           general.basename str              = Qwen3
llama_model_loader: - kv   4:                         general.size_label str              = 235B-A22B
llama_model_loader: - kv   5:                            general.license str              = apache-2.0
llama_model_loader: - kv   6:                       general.license.link str              = https://huggingfac
e.co/Qwen/Qwen3-235...
llama_model_loader: - kv   7:                               general.tags arr[str,1]       = ["text-generation"
]
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
llama_model_loader: - kv  25:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "
$", "%", "&", "'", ...
llama_model_loader: - kv  26:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  27:                      tokenizer.ggml.merges arr[str,151387]  = ["─á ─á", "─á─á ─á
─á", "i n", "─á t",...
llama_model_loader: - kv  28:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  29:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  30:                tokenizer.ggml.bos_token_id u32              = 151643
llama_model_loader: - kv  31:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  32:                    tokenizer.chat_template str              = {%- if tools %}\n
   {{- '<|im_start|>...
llama_model_loader: - kv  33:                      quantize.imatrix.file str              = /mnt/raid/models/u
bergarm/Qwen3-235B-...
llama_model_loader: - kv  34:                   quantize.imatrix.dataset str              = calibration_data_v
5_rc.txt
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
llm_load_print_meta: LF token         = 148848 '├ä─¼'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp         = 1536
llm_load_tensors: ggml ctx size =    0.99 MiB
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
Tensor blk.61.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 94 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 95/95 layers to GPU
llm_load_tensors:  CUDA_Host buffer size =   630.59 MiB
llm_load_tensors:        CPU buffer size = 86268.00 MiB
llm_load_tensors:      CUDA0 buffer size = 22495.01 MiB
....................................................................................................
============ Repacked 237 tensors
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 4096
llama_new_context_with_model: n_ubatch   = 2048
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 512
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:      CUDA0 KV buffer size =  6016.00 MiB
llama_new_context_with_model: KV self size  = 6016.00 MiB, K (f16): 3008.00 MiB, V (f16): 3008.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.58 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =  1264.01 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   288.02 MiB
llama_new_context_with_model: graph nodes  = 3672
llama_new_context_with_model: graph splits = 160

main: n_kv_max = 32768, n_batch = 4096, n_ubatch = 2048, flash_attn = 1, n_gpu_layers = 999, n_threads = 32, n
_threads_batch = 32

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  2048 |    512 |      0 |    5.689 |   360.01 |   21.272 |    24.07 |
|  2048 |    512 |   2048 |    5.679 |   360.63 |   21.940 |    23.34 |
|  2048 |    512 |   4096 |    5.772 |   354.82 |   21.786 |    23.50 |
|  2048 |    512 |   6144 |    5.837 |   350.84 |   23.445 |    21.84 |
|  2048 |    512 |   8192 |    5.924 |   345.70 |   21.879 |    23.40 |
|  2048 |    512 |  10240 |    5.999 |   341.40 |   22.474 |    22.78 |
|  2048 |    512 |  12288 |    6.060 |   337.94 |   22.852 |    22.40 |
|  2048 |    512 |  14336 |    6.124 |   334.44 |   22.670 |    22.58 |
|  2048 |    512 |  16384 |    6.178 |   331.48 |   23.226 |    22.04 |
|  2048 |    512 |  18432 |    6.250 |   327.69 |   22.997 |    22.26 |
|  2048 |    512 |  20480 |    6.265 |   326.88 |   24.764 |    20.68 |
|  2048 |    512 |  22528 |    6.359 |   322.08 |   23.715 |    21.59 |
|  2048 |    512 |  24576 |    6.454 |   317.34 |   24.515 |    20.88 |
|  2048 |    512 |  26624 |    6.494 |   315.36 |   24.823 |    20.63 |
|  2048 |    512 |  28672 |    6.530 |   313.64 |   24.246 |    21.12 |
|  2048 |    512 |  30720 |    6.601 |   310.27 |   25.295 |    20.24 |
```

</details>

<details>
<summary>Both GPUs, -ts 25,75 -ot "blk\.(3[5-9]|[4-9][0-9])\.ffn_.*_exps=CPU"</summary>

```
PS> .\bin\llama-server --version
version: 3772 (5236c98b)
built with Clang 19.1.5 for
PS> .\bin\llama-sweep-bench.exe 
    --alias ubergarm/Qwen3-235B-A22B-mix-IQ3_K 
    --model C:\Users\Administrator\.lmstudio\models\ubergarm\Qwen3-235B-A22B-GGUF\Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf 
    --no-mmap -rtr -fa -fmoe  
    -c 32768 -amb 512 -b 4096 -ub 2048 
    -ngl 999  -ts 25,75 -ot "blk\.(3[5-9]|[4-9][0-9])\.ffn_.*_exps=CPU" 
    --parallel 1 --threads 32 
    --main-gpu 0 
    --warmup-batch
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
  Device 1: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
llama_model_loader: additional 2 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 40 key-value pairs and 1131 tensors from C:\Users\Administrator\.lms
tudio\models\ubergarm\Qwen3-235B-A22B-GGUF\Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf (version GGUF V3 (lat
est))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3 235B A22B
llama_model_loader: - kv   3:                           general.basename str              = Qwen3
llama_model_loader: - kv   4:                         general.size_label str              = 235B-A22B
llama_model_loader: - kv   5:                            general.license str              = apache-2.0
llama_model_loader: - kv   6:                       general.license.link str              = https://huggingfac
e.co/Qwen/Qwen3-235...
llama_model_loader: - kv   7:                               general.tags arr[str,1]       = ["text-generation"
]
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
llama_model_loader: - kv  25:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "
$", "%", "&", "'", ...
llama_model_loader: - kv  26:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  27:                      tokenizer.ggml.merges arr[str,151387]  = ["─á ─á", "─á─á ─á
─á", "i n", "─á t",...
llama_model_loader: - kv  28:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  29:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  30:                tokenizer.ggml.bos_token_id u32              = 151643
llama_model_loader: - kv  31:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  32:                    tokenizer.chat_template str              = {%- if tools %}\n
   {{- '<|im_start|>...
llama_model_loader: - kv  33:                      quantize.imatrix.file str              = /mnt/raid/models/u
bergarm/Qwen3-235B-...
llama_model_loader: - kv  34:                   quantize.imatrix.dataset str              = calibration_data_v
5_rc.txt
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
llm_load_print_meta: LF token         = 148848 '├ä─¼'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp         = 1536
llm_load_tensors: ggml ctx size =    1.49 MiB
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
Tensor blk.61.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 94 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 95/95 layers to GPU
llm_load_tensors:  CUDA_Host buffer size =   630.59 MiB
llm_load_tensors:        CPU buffer size = 64428.00 MiB
llm_load_tensors:      CUDA0 buffer size = 27608.27 MiB
llm_load_tensors:      CUDA1 buffer size = 16726.74 MiB
....................................................................................................
============ Repacked 177 tensors
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 4096
llama_new_context_with_model: n_ubatch   = 2048
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 512
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:      CUDA0 KV buffer size =  1536.00 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =  4480.00 MiB
llama_new_context_with_model: KV self size  = 6016.00 MiB, K (f16): 3008.00 MiB, V (f16): 3008.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.58 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      CUDA0 compute buffer size =  1045.00 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =  1251.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   288.02 MiB
llama_new_context_with_model: graph nodes  = 3672
llama_new_context_with_model: graph splits = 180

main: n_kv_max = 32768, n_batch = 4096, n_ubatch = 2048, flash_attn = 1, n_gpu_layers = 999, n_threads = 32, n
_threads_batch = 32

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  2048 |    512 |      0 |    4.910 |   417.10 |   23.396 |    21.88 |
|  2048 |    512 |   2048 |    4.905 |   417.55 |   22.153 |    23.11 |
|  2048 |    512 |   4096 |    5.039 |   406.41 |   22.911 |    22.35 |
|  2048 |    512 |   6144 |    5.071 |   403.88 |   22.953 |    22.31 |
|  2048 |    512 |   8192 |    5.064 |   404.39 |   23.478 |    21.81 |
|  2048 |    512 |  10240 |    5.038 |   406.52 |   23.530 |    21.76 |
|  2048 |    512 |  12288 |    5.073 |   403.69 |   23.760 |    21.55 |
|  2048 |    512 |  14336 |    5.148 |   397.79 |   23.533 |    21.76 |
|  2048 |    512 |  16384 |    5.193 |   394.41 |   23.955 |    21.37 |
|  2048 |    512 |  18432 |    5.156 |   397.21 |   23.782 |    21.53 |
|  2048 |    512 |  20480 |    5.226 |   391.89 |   24.045 |    21.29 |
|  2048 |    512 |  22528 |    5.287 |   387.38 |   24.333 |    21.04 |
|  2048 |    512 |  24576 |    5.283 |   387.65 |   24.508 |    20.89 |
|  2048 |    512 |  26624 |    5.354 |   382.50 |   24.832 |    20.62 |
|  2048 |    512 |  28672 |    5.347 |   383.05 |   24.696 |    20.73 |
|  2048 |    512 |  30720 |    5.347 |   383.01 |   25.192 |    20.32 |
```

</details>

---

👤 **sousekd** commented the **2025-07-18** at **22:28:37**:<br>

Results for bartowski's **Qwen3‑235B‑A22B‑Q8_0** are less encouraging: although they're a bit better than before, the multi-GPU setup improves neither PP t/s nor TG t/s when compared with a single GPU:

<details>
<summary>RTX 5090 only, -ot "blk\.([6-9]|[1-9][0-9])\.ffn_.*_exps=CPU"</summary>

```
PS> .\bin\llama-server --version
version: 3772 (5236c98b)
built with Clang 19.1.5 for
PS> .\bin\llama-sweep-bench.exe 
    --alias bartowski/Qwen3-235B-A22B-Q8_0 
    --model C:\Users\Administrator\.lmstudio\models\lmstudio-community\Qwen3-235B-A22B-GGUF\Qwen3-235B-A22B-Q8_0-00001-of-00007.gguf 
    --no-mmap -rtr -fa -fmoe  
    -c 32768 -amb 512 -b 4096 -ub 2048 
    -ngl 999  -ot "blk\.([6-9]|[1-9][0-9])\.ffn_.*_exps=CPU" 
    --parallel 1 --threads 32 
    --main-gpu 0 
    --warmup-batch
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
llama_model_loader: additional 6 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 36 key-value pairs and 1131 tensors from C:\Users\Administrator\.lms
tudio\models\lmstudio-community\Qwen3-235B-A22B-GGUF\Qwen3-235B-A22B-Q8_0-00001-of-00007.gguf (version GGUF V3
 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3 235B A22B
llama_model_loader: - kv   3:                           general.basename str              = Qwen3
llama_model_loader: - kv   4:                         general.size_label str              = 235B-A22B
llama_model_loader: - kv   5:                            general.license str              = apache-2.0
llama_model_loader: - kv   6:                       general.license.link str              = https://huggingfac
e.co/Qwen/Qwen3-235...
llama_model_loader: - kv   7:                               general.tags arr[str,1]       = ["text-generation"
]
llama_model_loader: - kv   8:                       qwen3moe.block_count u32              = 94
llama_model_loader: - kv   9:                    qwen3moe.context_length u32              = 32768
llama_model_loader: - kv  10:                  qwen3moe.embedding_length u32              = 4096
llama_model_loader: - kv  11:               qwen3moe.feed_forward_length u32              = 12288
llama_model_loader: - kv  12:              qwen3moe.attention.head_count u32              = 64
llama_model_loader: - kv  13:           qwen3moe.attention.head_count_kv u32              = 4
llama_model_loader: - kv  14:                    qwen3moe.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  15:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  16:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  17:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  18:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  19:                      qwen3moe.expert_count u32              = 128
llama_model_loader: - kv  20:        qwen3moe.expert_feed_forward_length u32              = 1536
llama_model_loader: - kv  21:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  22:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  23:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "
$", "%", "&", "'", ...
llama_model_loader: - kv  24:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  25:                      tokenizer.ggml.merges arr[str,151387]  = ["─á ─á", "─á─á ─á
─á", "i n", "─á t",...
llama_model_loader: - kv  26:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  27:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  28:                tokenizer.ggml.bos_token_id u32              = 151643
llama_model_loader: - kv  29:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  30:                    tokenizer.chat_template str              = {%- if tools %}\n
   {{- '<|im_start|>...
llama_model_loader: - kv  31:               general.quantization_version u32              = 2
llama_model_loader: - kv  32:                          general.file_type u32              = 7
llama_model_loader: - kv  33:                                   split.no u16              = 0
llama_model_loader: - kv  34:                        split.tensors.count i32              = 1131
llama_model_loader: - kv  35:                                split.count u16              = 7
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
llm_load_print_meta: n_ctx_train      = 32768
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
llm_load_print_meta: n_ctx_orig_yarn  = 32768
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
llm_load_print_meta: LF token         = 148848 '├ä─¼'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp         = 1536
llm_load_tensors: ggml ctx size =    0.99 MiB
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
Tensor blk.61.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 94 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 95/95 layers to GPU
llm_load_tensors:  CUDA_Host buffer size =   630.59 MiB
llm_load_tensors:        CPU buffer size = 215424.00 MiB
llm_load_tensors:      CUDA0 buffer size = 22301.14 MiB
....................................................................................................
============ Repacked 264 tensors
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 4096
llama_new_context_with_model: n_ubatch   = 2048
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 512
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:      CUDA0 KV buffer size =  6016.00 MiB
llama_new_context_with_model: KV self size  = 6016.00 MiB, K (f16): 3008.00 MiB, V (f16): 3008.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.58 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =  1219.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   512.02 MiB
llama_new_context_with_model: graph nodes  = 3672
llama_new_context_with_model: graph splits = 178

main: n_kv_max = 32768, n_batch = 4096, n_ubatch = 2048, flash_attn = 1, n_gpu_layers = 999, n_threads = 32, n
_threads_batch = 32

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  2048 |    512 |      0 |    6.428 |   318.60 |   30.099 |    17.01 |
|  2048 |    512 |   2048 |    5.890 |   347.68 |   30.214 |    16.95 |
|  2048 |    512 |   4096 |    6.133 |   333.92 |   30.408 |    16.84 |
|  2048 |    512 |   6144 |    6.621 |   309.33 |   31.142 |    16.44 |
|  2048 |    512 |   8192 |    6.440 |   318.01 |   30.985 |    16.52 |
|  2048 |    512 |  10240 |    6.548 |   312.78 |   31.281 |    16.37 |
|  2048 |    512 |  12288 |    6.770 |   302.51 |   31.928 |    16.04 |
|  2048 |    512 |  14336 |    8.115 |   252.37 |   31.983 |    16.01 |
|  2048 |    512 |  16384 |    7.641 |   268.02 |   32.442 |    15.78 |
|  2048 |    512 |  18432 |    7.978 |   256.69 |   32.626 |    15.69 |
|  2048 |    512 |  20480 |    8.510 |   240.66 |   32.577 |    15.72 |
|  2048 |    512 |  22528 |    8.480 |   241.52 |   33.178 |    15.43 |
|  2048 |    512 |  24576 |    9.111 |   224.78 |   33.144 |    15.45 |
|  2048 |    512 |  26624 |    6.628 |   308.98 |   33.405 |    15.33 |
|  2048 |    512 |  28672 |    7.182 |   285.14 |   33.316 |    15.37 |
```

</details>

<details>
<summary>Both GPUs, -ts 11,89 -ot "blk\.(1[5-9]|[2-9][0-9])\.ffn_.*_exps=CPU"</summary>

```
PS> .\bin\llama-server --version
version: 3772 (5236c98b)
built with Clang 19.1.5 for
PS> .\bin\llama-sweep-bench.exe 
    --alias bartowski/Qwen3-235B-A22B-Q8_0 
    --model C:\Users\Administrator\.lmstudio\models\lmstudio-community\Qwen3-235B-A22B-GGUF\Qwen3-235B-A22B-Q8_0-00001-of-00007.gguf 
    --no-mmap -rtr -fa -fmoe  
    -c 32768 -amb 512 -b 4096 -ub 2048 
    -ngl 999  -ts 11,89 -ot "blk\.(1[5-9]|[2-9][0-9])\.ffn_.*_exps=CPU" 
    --parallel 1 --threads 32 
    --main-gpu 0 
    --warmup-batch
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
  Device 1: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
llama_model_loader: additional 6 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 36 key-value pairs and 1131 tensors from C:\Users\Administrator\.lms
tudio\models\lmstudio-community\Qwen3-235B-A22B-GGUF\Qwen3-235B-A22B-Q8_0-00001-of-00007.gguf (version GGUF V3
 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3 235B A22B
llama_model_loader: - kv   3:                           general.basename str              = Qwen3
llama_model_loader: - kv   4:                         general.size_label str              = 235B-A22B
llama_model_loader: - kv   5:                            general.license str              = apache-2.0
llama_model_loader: - kv   6:                       general.license.link str              = https://huggingfac
e.co/Qwen/Qwen3-235...
llama_model_loader: - kv   7:                               general.tags arr[str,1]       = ["text-generation"
]
llama_model_loader: - kv   8:                       qwen3moe.block_count u32              = 94
llama_model_loader: - kv   9:                    qwen3moe.context_length u32              = 32768
llama_model_loader: - kv  10:                  qwen3moe.embedding_length u32              = 4096
llama_model_loader: - kv  11:               qwen3moe.feed_forward_length u32              = 12288
llama_model_loader: - kv  12:              qwen3moe.attention.head_count u32              = 64
llama_model_loader: - kv  13:           qwen3moe.attention.head_count_kv u32              = 4
llama_model_loader: - kv  14:                    qwen3moe.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  15:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  16:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  17:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  18:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  19:                      qwen3moe.expert_count u32              = 128
llama_model_loader: - kv  20:        qwen3moe.expert_feed_forward_length u32              = 1536
llama_model_loader: - kv  21:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  22:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  23:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "
$", "%", "&", "'", ...
llama_model_loader: - kv  24:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  25:                      tokenizer.ggml.merges arr[str,151387]  = ["─á ─á", "─á─á ─á
─á", "i n", "─á t",...
llama_model_loader: - kv  26:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  27:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  28:                tokenizer.ggml.bos_token_id u32              = 151643
llama_model_loader: - kv  29:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  30:                    tokenizer.chat_template str              = {%- if tools %}\n
   {{- '<|im_start|>...
llama_model_loader: - kv  31:               general.quantization_version u32              = 2
llama_model_loader: - kv  32:                          general.file_type u32              = 7
llama_model_loader: - kv  33:                                   split.no u16              = 0
llama_model_loader: - kv  34:                        split.tensors.count i32              = 1131
llama_model_loader: - kv  35:                                split.count u16              = 7
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
llm_load_print_meta: n_ctx_train      = 32768
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
llm_load_print_meta: n_ctx_orig_yarn  = 32768
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
llm_load_print_meta: LF token         = 148848 '├ä─¼'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp         = 1536
llm_load_tensors: ggml ctx size =    1.49 MiB
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
Tensor blk.61.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 94 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 95/95 layers to GPU
llm_load_tensors:  CUDA_Host buffer size =   630.59 MiB
llm_load_tensors:        CPU buffer size = 193392.00 MiB
llm_load_tensors:      CUDA0 buffer size = 27745.10 MiB
llm_load_tensors:      CUDA1 buffer size = 16588.03 MiB
....................................................................................................
============ Repacked 237 tensors
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 4096
llama_new_context_with_model: n_ubatch   = 2048
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 512
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:      CUDA0 KV buffer size =   704.00 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =  5312.00 MiB
llama_new_context_with_model: KV self size  = 6016.00 MiB, K (f16): 3008.00 MiB, V (f16): 3008.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.58 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      CUDA0 compute buffer size =   832.01 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =  1251.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   512.02 MiB
llama_new_context_with_model: graph nodes  = 3672
llama_new_context_with_model: graph splits = 161

main: n_kv_max = 32768, n_batch = 4096, n_ubatch = 2048, flash_attn = 1, n_gpu_layers = 999, n_threads = 32, n
_threads_batch = 32

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  2048 |    512 |      0 |    6.244 |   327.97 |   35.789 |    14.31 |
|  2048 |    512 |   2048 |    6.703 |   305.52 |   37.055 |    13.82 |
|  2048 |    512 |   4096 |    7.033 |   291.19 |   36.991 |    13.84 |
|  2048 |    512 |   6144 |    6.878 |   297.76 |   37.356 |    13.71 |
|  2048 |    512 |   8192 |    7.178 |   285.30 |   34.877 |    14.68 |
|  2048 |    512 |  10240 |    6.623 |   309.21 |   35.364 |    14.48 |
|  2048 |    512 |  12288 |    7.588 |   269.89 |   38.203 |    13.40 |
|  2048 |    512 |  14336 |    7.067 |   289.81 |   36.447 |    14.05 |
|  2048 |    512 |  16384 |    7.026 |   291.47 |   37.939 |    13.50 |
|  2048 |    512 |  18432 |    7.106 |   288.22 |   36.550 |    14.01 |
|  2048 |    512 |  20480 |    7.021 |   291.68 |   36.748 |    13.93 |
|  2048 |    512 |  22528 |    9.771 |   209.61 |   35.841 |    14.29 |
|  2048 |    512 |  24576 |    7.363 |   278.14 |   35.104 |    14.59 |
|  2048 |    512 |  26624 |    7.908 |   258.98 |   34.859 |    14.69 |
|  2048 |    512 |  28672 |    8.499 |   240.98 |   38.081 |    13.45 |
|  2048 |    512 |  30720 |    8.279 |   247.38 |   36.174 |    14.15 |
```

</details>

---

👤 **ikawrakow** commented the **2025-07-19** at **06:45:47**:<br>

Thank you for these results.

I guess, with me never having run these giant models myself, and all my experience coming from much smaller MoE models, I just don't have the intuition of where things are getting bottlenecked.

I'm still waiting for the day when someone will decide to build a system with a 7995WX CPU, instead of dropping the required 10 grant on buying multiple high-end GPUs. A 7995WX system with all memory banks populated with high speed RAM may not be able to compete with your system on PP performance, but I wouldn't be surprised if it beats it in TG speed.

---

👤 **sousekd** commented the **2025-07-19** at **08:14:42**:<br>

Yeah, I think these results are really proof of the great optimizations you did on the CPU side… and also proof of Nvidia’s policy of deliberately disabling hardware features to drive upsells.

> I'm still waiting for the day when someone will decide to build a system with a 7995WX CPU, instead of dropping the required 10 grant on buying multiple high-end GPUs. A 7995WX system with all memory banks populated with high speed RAM may not be able to compete with your system on PP performance, but I wouldn't be surprised if it beats it in TG speed.

I'd probably expect the opposite (?), beating Epyc on PP due to its performance, but not quite reaching the memory bandwidth of 12 channels. But it would be nice to see - my Epyc 9355 was less then third of 7995WX price!

Anyway, it is amazing to see these huge models running on a relatively affordable hardware.

---

👤 **sousekd** commented the **2025-07-19** at **08:14:42**:<br>

Yeah, I think these results are really proof of the great optimizations you did on the CPU side… and also proof of Nvidia’s policy of deliberately disabling hardware features to drive upsells.

> I'm still waiting for the day when someone will decide to build a system with a 7995WX CPU, instead of dropping the required 10 grant on buying multiple high-end GPUs. A 7995WX system with all memory banks populated with high speed RAM may not be able to compete with your system on PP performance, but I wouldn't be surprised if it beats it in TG speed.

I'd probably expect the opposite - beating Epyc on PP, but not quite reaching the memory bandwidth of Epyc. But it would be nice to see - my Epyc 9355 was less then third of 7995WX price!

---

👤 **ikawrakow** commented the **2025-07-19** at **09:18:10**:<br>

Maybe you have posted CPU-only performance results somewhere else, but it is becoming hard to find stuff in this repository, so do you mind re-posting here? Just so one has it side-by-side to see how much you gain from adding the 5090. Thanks.

---

👤 **sousekd** commented the **2025-07-19** at **20:12:42**:<br>

@ikawrakow Seems GPU is still quite handy for these larger models :)

<img width="1979" height="1180" alt="Image" src="https://github.com/user-attachments/assets/8f9f2f6d-e357-4a03-b7a9-9f2a851cb9ca" />

<img width="1979" height="1180" alt="Image" src="https://github.com/user-attachments/assets/5b3f4c0c-40b9-45d2-9ed0-b1d0df95fd2a" />

---

👤 **ikawrakow** commented the **2025-07-20** at **05:29:03**:<br>

> @ikawrakow Seems having at least some GPU is still quite handy for these larger models :)

Arghh, you are destroying my dream of a GPU-free world!

More seriously: thank you for the graphs, this is useful to have in one spot.

In defense of the CPU only scenario:
* If you ran a 4 bpw instead of the 2 bpw quant you used, CPU PP performance will stay about the same while GPU performance will drop significantly
* If you had decided to run a different MoE model (e.g., Qwen3-235B-A22B, Maverick), relative performance would improve in favor of the CPU quite a bit. For these models self-attention represents a much smaller fraction of the overall computation cost, so gains from having it run on a GPU are significantly less
* If you had decided to not buy a 5090 but spend the money on upgrading your EPYC 9355 to 9555, then you would a) have nearly double the PP performance, and b) will have a much lower drop in TG performance with increasing `N_KV`.
* Your CPU is already very competitive with any GPU that is not a high-end Nvidia GPU

---

👤 **sousekd** commented the **2025-07-20** at **10:31:10**:<br>

Everything you said makes perfect sense. I also haven’t really tuned the inference parameters for optimal performance here, unlike with the CPU+GPU setup.

That said, I think a solid CPU paired with a decent GPU offers the best overall value (especially with ik_llama): it’s powerful enough to run large models, and the ability to run smaller models fast is a plus when building agents.