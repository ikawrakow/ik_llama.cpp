## 🔀 [Pull Request #484](https://github.com/ikawrakow/ik_llama.cpp/pull/484) - BF16 Trellis implementation

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Source Branch** | `ik/trellis_bf16` |
| **Target Branch** | `main` |
| **Created** | 2025-06-02 |
| **Updated** | 2025-06-19 |

---

## 📄 Description

This PR adds a `bf16` CPU implementation for the trellis quants `IQ2_KT, IQ3_KT` and `IQ4_KT` for CPUs with native `bf16` support.

We get massive gains in prompt processing speeds, and a ~5-10% gain in TG performance. On my Ryzen-7950X CPU that supports `bf16`, all 3 types now have PP-512 in the range of 230-240 t/s for 8B LLaMA-3. This makes them comparable to row-interleaved quants (where PP-512 performance on this CPU is in the 240-300 t/s range).

TG-128 performance for 8B LlaMA-3 on the Ryzen-7950X changes as follows

| type | f32 t/s | bf16 t/s|
|---: | ---: | ---: |
| IQ2_KT | 12.17 | 12.65 |
| IQ3_KT | 10.54 | 11.22 |
| IQ4_KT | 8.39 | 9.45 |

PP-512 performance for 8B LlaMA-3 on the Ryzen-7950X changes as follows

| type | f32 t/s | bf16 t/s|
|---: | ---: | ---: |
| IQ2_KT | 132.47 | 233.96 |
| IQ3_KT | 127.80 | 233.37 |
| IQ4_KT | 126.31 | 243.17 |

A similar optimization can be done for CPUs with native `fp16` support, but as I don't have access to one of those, this is not implemented for now.

---

## 💬 Conversation

👤 **ubergarm** commented on **2025-06-02** at **21:16:37**

Just did some a/b testing with llama-sweep-bench on my home rig using that new Qwen3-8B dense model distillation of R1-0528.

1. The good news: The PR is definitely faster than main branch as my AMD 9950X has cpu flag `avx512_bf16`
2. The bad news: Not sure what happened, but the first try with this branch it crashed with `ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed`. Worked fine the second time I ran it. Could *possibly* be my overclocked RAM but didn't see any other system instability and ran clean and finished on the second try without changing anything. Details below. *EDIT*: Was able to crash it again with a few more tries.
3. *EDIT2*: It worked okay on the Intel Xeon 6980P doing perplexity and llama-sweep-bench on the Thread Ripper Pro mentioned below now.

## Full GPU Offload
![trellis-full-gpu-offload](https://github.com/user-attachments/assets/a01c1fdb-448b-4b51-977c-5552c63f50f0)

## CPU Only
![trellis-cpu-only](https://github.com/user-attachments/assets/e40217e0-5be1-41db-bdc0-02152e0f0f53)

<details>

<summary>👈 Details and Logs</summary>

#### Test Quants
```
## DeepSeek-R1-0528-Qwen3-8B-IQ3_K
llama_model_loader: - type  f32:  145 tensors
llama_model_loader: - type iq3_k:   72 tensors ffn_(gate|up)
llama_model_loader: - type iq4_ks:  182 tensors everything else
llm_load_print_meta: model size       = 3.714 GiB (3.895 BPW)
Final estimate: PPL = 11.7407 +/- 0.09382

## DeepSeek-R1-0528-Qwen3-8B-IQ3_KT.gguf
llama_model_loader: - type  f32:  145 tensors
llama_model_loader: - type iq3_kt:   72 tensors ffn_(gate|up)
llama_model_loader: - type iq4_kt:  182 tensors everything else
llm_load_print_meta: model size       = 3.455 GiB (3.624 BPW)
Final estimate: PPL = 12.2157 +/- 0.09915
```

#### llama-sweep-bench
#### Full GPU Offload
```bash
$ git checkout main
$ git rev-parse --short HEAD
7a8abe29

cmake -B build -DGGML_CUDA=ON -DGGML_RPC=OFF -DGGML_BLAS=OFF -DGGML_CUDA_IQK_FORCE_BF16=1 -DGGML_SCHED_MAX_COPIES=1
cmake --build build --config Release -j $(nproc)

#model=/mnt/astrodata/llm/models/ubergarm/DeepSeek-R1-0528-Qwen3-8B-GGUF/DeepSeek-R1-0528-Qwen3-8B-IQ3_K.gguf
model=/mnt/astrodata/llm/models/ubergarm/DeepSeek-R1-0528-Qwen3-8B-GGUF/DeepSeek-R1-0528-Qwen3-8B-IQ3_KT.gguf
CUDA_VISIBLE_DEVICES="0" \
./build/bin/llama-sweep-bench \
  --model "$model" \
  -fa \
  -c 32768 \
  -ngl 99 \
  --threads 1 \
  --warmup-batch
```

#### CPU Only
```bash
# main test case
$ git checkout main
$ git rev-parse --short HEAD
7a8abe29

# PR484 ik/trellis_bf16 test case
$ git checkout ik/trellis_bf16
$ git rev-parse --short HEAD
061d064b

cmake -B build -DGGML_CUDA=OFF -DGGML_BLAS=OFF
cmake --build build --config Release -j $(nproc)

# with and without -rtr test cases
./build/bin/llama-sweep-bench \
  --model "$model" \
  -fa \
  -c 8704 \
  --threads 16 \
  --warmup-batch
```

#### Full Crash Logs
```
model=/mnt/astrodata/llm/models/ubergarm/DeepSeek-R1-0528-Qwen3-8B-GGUF/DeepSeek-R1-0528-Qwen3-8B-IQ3_KT.gguf

./build/bin/llama-sweep-bench \
  --model "$model" \
  -fa \
  -c 8704 \
  --threads 16 \
  --warmup-batch

llama_model_loader: loaded meta data with 36 key-value pairs and 399 tensors from /mnt/astrodata/llm/models/ubergarm/DeepSeek-R1-0528-Qwen3-8B-GGUF/DeepSeek-R1-0528-Qwen3-8B-IQ3_KT.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = DeepSeek R1 0528 Qwen3 8B
llama_model_loader: - kv   3:                           general.basename str              = DeepSeek-R1-0528-Qwen3
llama_model_loader: - kv   4:                         general.size_label str              = 8B
llama_model_loader: - kv   5:                            general.license str              = mit
llama_model_loader: - kv   6:                          qwen3.block_count u32              = 36
llama_model_loader: - kv   7:                       qwen3.context_length u32              = 131072
llama_model_loader: - kv   8:                     qwen3.embedding_length u32              = 4096
llama_model_loader: - kv   9:                  qwen3.feed_forward_length u32              = 12288
llama_model_loader: - kv  10:                 qwen3.attention.head_count u32              = 32
llama_model_loader: - kv  11:              qwen3.attention.head_count_kv u32              = 8
llama_model_loader: - kv  12:                       qwen3.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  13:     qwen3.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  14:                 qwen3.attention.key_length u32              = 128
llama_model_loader: - kv  15:               qwen3.attention.value_length u32              = 128
llama_model_loader: - kv  16:                          general.file_type u32              = 152
llama_model_loader: - kv  17:                    qwen3.rope.scaling.type str              = yarn
llama_model_loader: - kv  18:                  qwen3.rope.scaling.factor f32              = 4.000000
llama_model_loader: - kv  19: qwen3.rope.scaling.original_context_length u32              = 32768
llama_model_loader: - kv  20:               general.quantization_version u32              = 2
llama_model_loader: - kv  21:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  22:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  23:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  24:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  25:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv  26:                tokenizer.ggml.bos_token_id u32              = 151643
llama_model_loader: - kv  27:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  28:            tokenizer.ggml.padding_token_id u32              = 151645
llama_model_loader: - kv  29:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  30:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  31:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  32:                      quantize.imatrix.file str              = /mnt/raid/models/ubergarm/DeepSeek-R1...
llama_model_loader: - kv  33:                   quantize.imatrix.dataset str              = ubergarm-imatrix-calibration-corpus-v...
llama_model_loader: - kv  34:             quantize.imatrix.entries_count i32              = 253
llama_model_loader: - kv  35:              quantize.imatrix.chunks_count i32              = 840
llama_model_loader: - type  f32:  145 tensors
llama_model_loader: - type iq3_kt:   72 tensors
llama_model_loader: - type iq4_kt:  182 tensors
llm_load_vocab: special tokens cache size = 28
llm_load_vocab: token to piece cache size = 0.9311 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = qwen3
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 151936
llm_load_print_meta: n_merges         = 151387
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 131072
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_layer          = 36
llm_load_print_meta: n_head           = 32
llm_load_print_meta: n_head_kv        = 8
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 4
llm_load_print_meta: n_embd_k_gqa     = 1024
llm_load_print_meta: n_embd_v_gqa     = 1024
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 12288
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
llm_load_print_meta: model type       = ?B
llm_load_print_meta: model ftype      = IQ3_KT - 3.125 bpw
llm_load_print_meta: model params     = 8.191 B
llm_load_print_meta: model size       = 3.455 GiB (3.624 BPW)
llm_load_print_meta: repeating layers = 2.874 GiB (3.554 BPW, 6.946 B parameters)
llm_load_print_meta: general.name     = DeepSeek R1 0528 Qwen3 8B
llm_load_print_meta: BOS token        = 151643 '<｜begin▁of▁sentence｜>'
llm_load_print_meta: EOS token        = 151645 '<｜end▁of▁sentence｜>'
llm_load_print_meta: PAD token        = 151645 '<｜end▁of▁sentence｜>'
llm_load_print_meta: LF token         = 148848 'ÄĬ'
llm_load_print_meta: max token length = 256
llm_load_tensors: ggml ctx size =    0.18 MiB
llm_load_tensors:        CPU buffer size =  3538.31 MiB
......................................................................................
llama_new_context_with_model: n_ctx      = 8704
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 0
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 0.25
llama_kv_cache_init:        CPU KV buffer size =  1224.00 MiB
llama_new_context_with_model: KV self size  = 1224.00 MiB, K (f16):  612.00 MiB, V (f16):  612.00 MiB
llama_new_context_with_model:        CPU  output buffer size =     0.58 MiB
llama_new_context_with_model:        CPU compute buffer size =   304.75 MiB
llama_new_context_with_model: graph nodes  = 978
llama_new_context_with_model: graph splits = 1

main: n_kv_max = 8704, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = -1, n_threads = 16, n_threads_batch = 16

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    1.610 |   318.01 |    8.114 |    15.77 |
|   512 |    128 |    512 |    1.672 |   306.24 |    8.222 |    15.57 |
|   512 |    128 |   1024 |    1.727 |   296.51 |    8.403 |    15.23 |
|   512 |    128 |   1536 |    1.787 |   286.52 |    8.455 |    15.14 |
|   512 |    128 |   2048 |    1.843 |   277.76 |    8.639 |    14.82 |
|   512 |    128 |   2560 |    1.897 |   269.93 |    8.709 |    14.70 |
|   512 |    128 |   3072 |    1.949 |   262.74 |    8.831 |    14.49 |
|   512 |    128 |   3584 |    1.999 |   256.17 |    8.952 |    14.30 |
|   512 |    128 |   4096 |    2.057 |   248.87 |    9.074 |    14.11 |
|   512 |    128 |   4608 |    2.175 |   235.36 |    9.384 |    13.64 |
|   512 |    128 |   5120 |    2.167 |   236.23 |    9.352 |    13.69 |
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
.
.
.
```

*EDIT* without rebooting it ran clean twice then the third time blew up again with:
```
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    1.632 |   313.64 |    8.088 |    15.83 |
|   512 |    128 |    512 |    1.683 |   304.24 |    8.214 |    15.58 |
|   512 |    128 |   1024 |    1.741 |   294.14 |    8.619 |    14.85 |
|   512 |    128 |   1536 |    1.798 |   284.73 |    8.462 |    15.13 |
|   512 |    128 |   2048 |    1.851 |   276.66 |    8.621 |    14.85 |
|   512 |    128 |   2560 |    1.909 |   268.16 |    8.725 |    14.67 |
|   512 |    128 |   3072 |    1.966 |   260.48 |    8.851 |    14.46 |
|   512 |    128 |   3584 |    2.022 |   253.27 |    8.981 |    14.25 |
|   512 |    128 |   4096 |    2.072 |   247.09 |    9.151 |    13.99 |
|   512 |    128 |   4608 |    2.157 |   237.39 |    9.218 |    13.89 |
|   512 |    128 |   5120 |    2.179 |   234.97 |    9.344 |    13.70 |
|   512 |    128 |   5632 |    2.248 |   227.72 |    9.499 |    13.48 |
|   512 |    128 |   6144 |    2.286 |   223.97 |    9.649 |    13.27 |
|   512 |    128 |   6656 |    2.339 |   218.94 |   10.081 |    12.70 |
|   512 |    128 |   7168 |    2.396 |   213.67 |    9.989 |    12.81 |
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed/mnt/astrodata/llm/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed

/mnt/astrodata/llm/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/mnt/astrodata/llm/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
ptrace: Operation not permitted.
```

</details>

## CPU Only 7965WX
It took just under 8 hours to slow cook `DeepSeek-R1-0528-IQ2_KT` 196.696 GiB (2.514 BPW) on this rig. It doesn't run with CUDA offload as possibly missing some mmvq stuff `ik_llama.cpp/ggml/src/ggml-cuda/mmvq.cu:564: fatal error`, but seems to work fine when compiling for CPU only. Didn't get the assert like above either in very limited testing. 

![trellis-sweep-iq2_kt-cpu-only](https://github.com/user-attachments/assets/d118842d-fb8a-48cb-8aa9-0ba338b81197)

```bash
./build/bin/llama-sweep-bench \
    --model ubergarm/DeepSeek-R1-0528-GGUF/DeepSeek-R1-0528-IQ2_KT.gguf \
    --ctx-size 4608 \
    -mla 3 -fa \
    -amb 512 \
    -fmoe \
    --threads 24 \
    --warmup-batch \
    --no-mmap

llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q5_0:   61 tensors attn_k_b (crashes if u try to quantize to iq4_kt)
llama_model_loader: - type iq2_kt:  116 tensors ffn_(up|gate)_exps
llama_model_loader: - type iq3_kt:   58 tensors ffn_down_exps
llama_model_loader: - type iq4_kt:  551 tensors attn/shexp/token_embd
```

Happy to try out anything to reproduce and hope it isn't a Heisenbug...

Also, I was considering cooking a hybrid iq4_kt attn/shexp with iq3_k/iq2_k down/(up|gate) R1-0528, but with this speed-up to CPU inferencing I'll go all in with iq3_kt/iq2_kt down/(gate|up) just to see what happens. Gonna take a while to cook though! Thanks!

---

👤 **saood06** commented on **2025-06-03** at **00:38:49**

>`iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed`

I'm fairly certain that means there is a NaN somewhere in the calculations.

---

👤 **ikawrakow** commented on **2025-06-03** at **04:22:51**

Thank for testing.

Yes, this assert is always associated with a NaN somewhere else. I ran into NaNs with the `fp16` implementation on NEON, and had to be extra careful with under- and overflows and what needs to be computed with `fp32`. But I wouldn't have thought there could be similar issues with `bf16`.

Looking at the low GPU TG performance, my guess is that you need to explicitly enable `F16` on CUDA (`cmake -DGGML_CUDA_F16=ON`).

---

👤 **ubergarm** commented on **2025-06-03** at **05:22:12**

I didn't run into that assert in limited testing a mixes of iqN_kt with DeepSeek-R1-0528 on two remote systems fwiw. This PR did speed up CPU only compiled inferencing but couldn't test CUDA offload as described. Accidently updated my above comment before realizing you'd already commented. Its past my bed time hah.

> -DGGML_CUDA_F16=ON

That did the trick for the `_kt` quant!

![trellis-full-gpu-offload-ggml_cuda_f16-on](https://github.com/user-attachments/assets/9bbf23ff-a76c-44fb-9699-8194dae76f07)

Thanks!

---

👤 **ikawrakow** commented on **2025-06-03** at **07:10:14**

I hadn't tested this PR with a DeepSeek model. Testing now I see DeepSeek-Lite breaks with `bf16` precision. I don't get NaNs but I get extremely high perplexity values and gibberish in TG.

---

👤 **ikawrakow** commented on **2025-06-03** at **07:24:15**

Something goes wrong on CUDA too with DeepSeek-Lite. So, it seems, trellis quants are not quite ready for prime time yet.

---

👤 **ikawrakow** commented on **2025-06-19** at **07:26:25**

Closing in favor of [#529](https://github.com/ikawrakow/ik_llama.cpp/issues/529)