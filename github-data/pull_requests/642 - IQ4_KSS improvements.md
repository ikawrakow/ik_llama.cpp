## 🔀 [Pull Request #642](https://github.com/ikawrakow/ik_llama.cpp/pull/642) - IQ4_KSS improvements

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/iq4_kss_improvements` |
| **Target Branch** | `main` |
| **Created** | 2025-07-23 |
| **Updated** | 2025-07-27 |
| **Merged** | 2025-07-23 |

---

## 📄 Description

Not much is known about `IQ4_KSS`, and nobody seems to be using it. So, I decided to give it some attention.

Quick reminder (for more, see [#89](https://github.com/ikawrakow/ik_llama.cpp/issues/89))
* `IQ4_KSS` uses exactly 4.0 bpw just like `IQ4_KT`
* Performance on CUDA is very similar to `IQ4_KT` (after this PR)
* PP CPU performance is similar to `IQ4_KT` (after this PR)
* TG CPU performance is quite a bit better than `IQ4_KT`
* PPL is only slightly worse than `IQ4_KT` 

This PR
* Adds CUDA quantized matrix multiplication kernel
* Adds repacking to `Q8_K_R8` for fast CPU GEMM
* Adds a small improvement in quantization accuracy

---

## 💬 Conversation

👤 **ubergarm** commented on **2025-07-23** at **16:05:19**

I had just made an unreleased Qwen3-235B-A22B-Instruct-2507-IQ4_KSS feeling around for the sweet spot near 4BPW for mostly CPU inference. It seemed pretty good for the size, but I was fiddling around juicing up some attn tensors and first few layers as well so too many variables.

If i get some time later this week, I might revisit that and do a proper a/b comparison of PPL for this PR.

Swamped by all the releases and slowly digging out what a wild ride this week lol...

Here is a lot of my raw data from testing with that model:

<details>

<summary>👈 Details</summary>

```bash
#!/usr/bin/env bash

# Repeating Layers [0-93]

custom="
# Attention
blk\..*\.attn_q.*=iq6_k
blk\..*\.attn_k.*=q8_0
blk\..*\.attn_v.*=q8_0
blk\..*\.attn_output.*=iq6_k

# Routed Experts
blk\.(0|1|2|3)\.ffn_down_exps\.weight=iq5_ks
blk\.(0|1|2|3)\.ffn_(gate|up)_exps\.weight=iq4_ks
blk\..*\.ffn_down_exps\.weight=iq4_ks
blk\..*\.ffn_(gate|up)_exps\.weight=iq4_kss

# Token Embedding
token_embd\.weight=iq4_k
output\.weight=iq6_k
"

custom=$(
  echo "$custom" | grep -v '^#' | \
  sed -Ez 's:\n+:,:g;s:,$::;s:^,::'
)

numactl -N 0 -m 0 \
./build/bin/llama-quantize \
    --custom-q "$custom" \
    --imatrix /mnt/raid/models/ubergarm/Qwen3-235B-A22B-Instruct-2507-GGUF/imatrix-Qwen3-235B-A22B-Instruct-2507-BF16.dat \
    /mnt/raid/models/ubergarm/Qwen3-235B-A22B-Instruct-2507-GGUF/Qwen3-235B-A22B-Instruct-2507-BF16-00001-of-00010.gguf \
    /mnt/raid/models/ubergarm/Qwen3-235B-A22B-Instruct-2507-GGUF/Qwen3-235B-A22B-Instruct-2507-smort-IQ4_KSS.gguf \
    IQ4_KSS \
    192
```

note data might have some copy paste errors in the comments, its been a busy week lol
```json
[
  {
    "name": "BF16",
    "ppl": "4.3079 +/- 0.02544",
    "size": 437.989,
    "bpw": 16.003,
    "legend": "pure",
    "comment": ""
  },
  {
    "name": "Q8_0",
    "ppl": "4.3139 +/- 0.02550",
    "size": 232.769,
    "bpw": 8.505,
    "legend": "pure"
  },
  {
    "name": "pure-IQ4_KS",
    "ppl": "4.4156 +/- 0.02624",
    "size": 116.994,
    "bpw": 4.275,
    "legend": "pure",
    "comment": "iq4_k token_embd, iq6_k output, ubergarm-imatrix-calibration-corpus-v02.txt"
  },
  {
    "name": "IQ2_KL",
    "ppl": "4.7912 +/- 0.02910",
    "size": 81.866,
    "bpw": 2.991,
    "legend": "ubergarm",
    "comment": "juiced q8_0 k|v, iq6_k q|o, iq3_ks down, iq2_kl gate|up"
  },
  {
    "name": "IQ3_KS",
    "ppl": "4.5275 +/- 0.02703",
    "size": 97.968,
    "bpw": 3.580,
    "legend": "ubergarm",
    "comment": "iq4_kt attn_.*, iq4_ks down, iq3_ks gate|up"
  },
  {
    "name": "mix-IQ3_KS",
    "ppl": "4.5078 +/- 0.02700",
    "size": 98.979,
    "bpw": 3.617,
    "legend": "ubergarm",
    "comment": "iq5_ks attn_.*, iq4_ks down, iq3_ks gate|up"
  },
  {
    "name": "smort-IQ3_KS",
    "ppl": "4.4915 +/- 0.02685",
    "size": 101.308,
    "bpw": 3.702,
    "legend": "ubergarm",
    "comment": "juiced q8_0 k|v, iq6_k q|o, iq4_ks down, iq3_ks gate|up"
  },
  {
    "name": "IQ3_K",
    "ppl": "4.4561 +/- 0.02657",
    "size": 106.644,
    "bpw": 3.897,
    "legend": "ubergarm",
    "comment": "juiced q8_0 k|v, iq6_k q|o, iq4_k down, iq3_k gate|up"
  },
  {
    "name": "smort-IQ4_KSS",
    "ppl": "4.4017 +/- 0.02614",
    "size": 115.085,
    "bpw": 4.205,
    "legend": "ubergarm",
    "comment": "juiced q8_0 k|v, iq6_k q|o, juiced first 4 routed exps layers, iq4_ks down, iq4_kss gate|up"
  },
  {
    "name": "IQ4_KS",
    "ppl": "4.3923 +/- 0.02618",
    "size": 126.587,
    "bpw": 4.625,
    "legend": "ubergarm",
    "comment": "iq5_ks attn_.*"
  },
  {
    "name": "IQ5_K",
    "ppl": "4.3351 +/- 0.02566",
    "size": 161.722,
    "bpw": 5.909,
    "legend": "ubergarm",
    "comment": "juiced q8_0 k|v, iq6_k q|o, iq6_k down, iq5_k gate|up"
  }
]
```

</details>

<img width="1788" height="1179" alt="ppl-Qwen3-235B-2507" src="https://github.com/user-attachments/assets/077d5875-0f6c-46b3-9c61-32d9defa9f1d" />

---

👤 **ikawrakow** commented on **2025-07-23** at **16:11:03**

@ubergarm  Btw, I'm not finding where you mentioned to be seeing pauses after a comma to ping you there in case you missed PR [#639](https://github.com/ikawrakow/ik_llama.cpp/issues/639) that fixes the issue.

---

👤 **ikawrakow** commented on **2025-07-23** at **17:48:16**

So, I'll disappear tomorrow for 2 weeks. Do I merge this before I go?

---

👤 **ubergarm** commented on **2025-07-23** at **18:43:07**

YOLO! (you only live once 🤣)

i have not tested yet, but it seems at quick glance the code changes don't
effect non-IQ4_KSS quants. as there aren't any of those quants released of
which i know — yeah merge it and we can sort it out later lol!

unrelated, i have not opened an issue, but was having a segfault in
llama-quantize with IQ3_KT trellis quant so have not released. Recipe here:

https://huggingface.co/ubergarm/Qwen3-Coder-480B-A35B-Instruct-GGUF#iq2_kt-todo

Finally, unrelated, when trying to run this IQ2_KL (it quantizes fine) but
crashes with asserts towards the end of starting up :
https://huggingface.co/ubergarm/Qwen3-Coder-480B-A35B-Instruct-GGUF#iq2_kl-169597-gib-3034-bpw

compiled CPU only, on that big dual socket epyc

Sorry, not at home today now for proper logs

finally finally, feel free to ignore all this and have a great couple
weeks!!! 😋 catch you later!

---

👤 **ikawrakow** commented on **2025-07-23** at **18:50:45**

When you get a chance, post the assert that the `IQ2_KL` model hits. The `IQ3_KT` segfault will be much more difficult to fix without a run in the debugger.

---

👤 **ThomasBaruzier** commented on **2025-07-23** at **18:52:06**

> So, I'll disappear tomorrow for 2 weeks

Noooooo

Not urgent, but did you have the chance to look into the issue where imatrix data for `attn_k_b` was missing when quantizing kimi?

---

👤 **ikawrakow** commented on **2025-07-23** at **19:10:20**

> Not urgent, but did you have the chance to look into the issue where imatrix data for attn_k_b was missing when quantizing kimi?

Ha, I looked into it, then searched for the thread where we were talking about it, didn't find it, and then forgot.

I'm actually not sure what happens in the Kimi runs. imatrix works fine when I test with a smaller model with the same attention architecture (DeepSeek-Lite). I tested with a GGUF created specifically for `llama.cpp` MLA (so `attn_k_b` and `attn_v_b` present, but not `attn_kv_b`), with a GGUF that precedes `ik_llama.cpp` MLA (so only `attn_kv_b` present), and with a version created from the safetensors with the `ik_llama.cpp` `convert_hf_to_gguf.py` script (so, all 3 present in the GGUF). In all 3 cases it worked fine with `-mla 1`. I didn't see tensor names with `(view of ...)` appended to the `attn_k_b` name, and `attn_v_b` calls were always triggered as expected. The only thing I was not sure if I was exercising was the split of the attention calculation using `-amb` (DeepSeek-Lite has 8 times fewer attention heads than the giant MLA models, so not easy to trigger the split). So, perhaps running the imatrix calculation without `-amb` would resolve it? The imatrix runs don't need such a big context, the `-mla 3` option that requires large work buffer without `-amb` is not being used, so it should be OK to run without `-amb`.

So, in short, just try running without `-amb`. First with `--verbosity 2` to see if the imatrix data collection function gets called with `attn_k_b` and `attn_v_b`.  If yes, rerun the imatrix calculation that way. If it still doesn't work, it will have to wait until I come back.

---

👤 **ubergarm** commented on **2025-07-23** at **20:41:55**

Hope you get some sleep before your travels! Besides we can just use Qwen3-Coder now to fix everything right? :rofl: 

I'll open proper issues for these if I can't figure it out. Zero rush or priority here as I've not released these two models giving me troubles.

Just got a laptop with some WiFi and can give a quick log:

> When you get a chance, post the assert that the IQ2_KL model hits.

*EDIT* Here is the Issue: [#649](https://github.com/ikawrakow/ik_llama.cpp/issues/649) 

<details>

<summary>IQ2_KL assert run and log</summary>

```bash
model=/mnt/raid/hf/Qwen3-Coder-480B-A35B-Instruct-GGUF/IQ2_KL/Qwen3-480B-A35B-Instruct-IQ2_KL-00001-of-00004.gguf

numactl -N 1 -m 1 \
./build/bin/llama-server \
    --model "$model"\
    --alias ubergarm/Qwen3-Coder-480B-A35B-Instruct \
    --ctx-size 196608 \
    -ctk q8_0 -ctv q8_0 \
    -fa -fmoe \
    -ub 4096 -b 4096 \
    --parallel 3 \
    --threads 128 \
    --threads-batch 192 \
    --numa numactl \
    --host 127.0.0.1 \
    --port 8080 \
    --no-mmap

INFO [                    main] build info | tid="127586578487488" timestamp=1753302334 build=3821 commit="1b052109"
INFO [                    main] system info | tid="127586578487488" timestamp=1753302334 n_threads=128 n_threads_batch=192 total_threads=768 system_info="AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: additional 3 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 41 key-value pairs and 747 tensors from /mnt/raid/hf/Qwen3-Coder-480B-A35B-Instruct-GGUF/IQ2_KL/Qwen3-480B-A35B-Instruct-IQ2_KL-00001-of-00004.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3 Coder 480B A35B Instruct
llama_model_loader: - kv   3:                           general.finetune str              = Instruct
llama_model_loader: - kv   4:                           general.basename str              = Qwen3-Coder
llama_model_loader: - kv   5:                         general.size_label str              = 480B-A35B
llama_model_loader: - kv   6:                            general.license str              = apache-2.0
llama_model_loader: - kv   7:                       general.license.link str              = https://huggingface.co/Qwen/Qwen3-Cod...
llama_model_loader: - kv   8:                               general.tags arr[str,1]       = ["text-generation"]
llama_model_loader: - kv   9:                       qwen3moe.block_count u32              = 62
llama_model_loader: - kv  10:                    qwen3moe.context_length u32              = 262144
llama_model_loader: - kv  11:                  qwen3moe.embedding_length u32              = 6144
llama_model_loader: - kv  12:               qwen3moe.feed_forward_length u32              = 8192
llama_model_loader: - kv  13:              qwen3moe.attention.head_count u32              = 96
llama_model_loader: - kv  14:           qwen3moe.attention.head_count_kv u32              = 8
llama_model_loader: - kv  15:                    qwen3moe.rope.freq_base f32              = 10000000.000000
llama_model_loader: - kv  16:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  17:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  18:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  19:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  20:                          general.file_type u32              = 155
llama_model_loader: - kv  21:                      qwen3moe.expert_count u32              = 160
llama_model_loader: - kv  22:        qwen3moe.expert_feed_forward_length u32              = 2560
llama_model_loader: - kv  23: qwen3moe.expert_shared_feed_forward_length u32              = 0
llama_model_loader: - kv  24:               general.quantization_version u32              = 2
llama_model_loader: - kv  25:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  26:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  27:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  28:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  29:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv  30:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  31:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  32:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  33:                    tokenizer.chat_template str              = {% macro render_item_list(item_list, ...
llama_model_loader: - kv  34:                      quantize.imatrix.file str              = /mnt/raid/models/ubergarm/Qwen3-Coder...
llama_model_loader: - kv  35:                   quantize.imatrix.dataset str              = ubergarm-imatrix-calibration-corpus-v...
llama_model_loader: - kv  36:             quantize.imatrix.entries_count i32              = 497
llama_model_loader: - kv  37:              quantize.imatrix.chunks_count i32              = 840
llama_model_loader: - kv  38:                                   split.no u16              = 0
llama_model_loader: - kv  39:                                split.count u16              = 4
llama_model_loader: - kv  40:                        split.tensors.count i32              = 747
llama_model_loader: - type  f32:  311 tensors
llama_model_loader: - type q8_0:  124 tensors
llama_model_loader: - type iq3_k:   62 tensors
llama_model_loader: - type iq4_k:    1 tensors
llama_model_loader: - type iq6_k:  125 tensors
llama_model_loader: - type iq2_kl:  124 tensors
llm_load_vocab: special tokens cache size = 26
llm_load_vocab: token to piece cache size = 0.9311 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = qwen3moe
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 151936
llm_load_print_meta: n_merges         = 151387
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 262144
llm_load_print_meta: n_embd           = 6144
llm_load_print_meta: n_layer          = 62
llm_load_print_meta: n_head           = 96
llm_load_print_meta: n_head_kv        = 8
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 12
llm_load_print_meta: n_embd_k_gqa     = 1024
llm_load_print_meta: n_embd_v_gqa     = 1024
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 8192
llm_load_print_meta: n_expert         = 160
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 2
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 10000000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 262144
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = ?B
llm_load_print_meta: model ftype      = IQ2_KL - 2.6875 bpw
llm_load_print_meta: model params     = 480.155 B
llm_load_print_meta: model size       = 169.597 GiB (3.034 BPW) 
llm_load_print_meta: repeating layers = 168.388 GiB (3.024 BPW, 478.288 B parameters)
llm_load_print_meta: general.name     = Qwen3 Coder 480B A35B Instruct
llm_load_print_meta: BOS token        = 11 ','
llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
llm_load_print_meta: PAD token        = 151643 '<|endoftext|>'
llm_load_print_meta: LF token         = 148848 'ÄĬ'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp         = 2560
llm_load_tensors: ggml ctx size =    0.33 MiB
llm_load_tensors:        CPU buffer size = 173666.87 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 196608
llama_new_context_with_model: n_batch    = 4096
llama_new_context_with_model: n_ubatch   = 4096
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:        CPU KV buffer size = 25296.00 MiB
llama_new_context_with_model: KV self size  = 25296.00 MiB, K (q8_0): 12648.00 MiB, V (q8_0): 12648.00 MiB
llama_new_context_with_model:        CPU  output buffer size =     2.32 MiB
llama_new_context_with_model:        CPU compute buffer size =  5184.05 MiB
llama_new_context_with_model: graph nodes  = 2424
llama_new_context_with_model: graph splits = 1
/home/w/projects/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: /home/w/projects/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: /home/w/projects/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: /home/w/projects/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed/home/w/projects/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: /home/w/projects/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/home/w/projects/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/home/w/projects/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/home/w/projects/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/home/w/projects/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/home/w/projects/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/home/w/projects/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/home/w/projects/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/home/w/projects/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/home/w/projects/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/home/w/projects/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/home/w/projects/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/home/w/projects/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/home/w/projects/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
GGML_ASSERT(fms.S[j] > 0) failed
/home/w/projects/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/home/w/projects/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/home/w/projects/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/home/w/projects/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/home/w/projects/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/home/w/projects/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/home/w/projects/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/home/w/projects/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/home/w/projects/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/home/w/projects/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/home/w/projects/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/home/w/projects/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
GGML_ASSERT(fms.S[j] > 0) failed
GGML_ASSERT(fms.S[j] > 0) failed

GGML_ASSERT(fms.S[j] > 0) failed
/home/w/projects/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
ptrace: Inappropriate ioctl for device.
No stack.
The program is not being run.
Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
ptrace: Inappropriate ioctl for device.
No stack.
The program is not being run.
Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
ptrace: Inappropriate ioctl for device.
No stack.
The program is not being run.
Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
warning: process 4140403 is a zombie - the process has already terminated
ptrace: Inappropriate ioctl for device.
No stack.
The program is not being run.
./myscripts/api-server-Qwen3-Coder-480B-A35B-Instruct.sh: line 34: 4140403 Aborted                 (core dumped) numactl -N 1 -m 1 ./build/bin/llama-server --model "$model" --alias ubergarm/Qwen3-Coder-480B-A35B-Instruct --ctx-size 196608 -ctk q8_0 -ctv q8_0 -fa -fmoe -ub 4096 -b 4096 --parallel 3 --threads 128 --threads-batch 192 --numa numactl --host 127.0.0.1 --port 8080 --no-mmap
```

</details>

> The IQ3_KT segfault will be much more difficult to fix without a run in the debugger.

*EDIT* here is that issue with debug logs: [#650](https://github.com/ikawrakow/ik_llama.cpp/issues/650) 

Yeah, I'll give full logs on its own issue later, it could just be this hardware possibly as it throws an error in `dmesg` as well. Here is the quick look

<details>

<summary>segfault quantizing iq3_kt</summary>

```bash

$ sudo dmest -T --follow

[Wed Jul 23 16:36:14 2025] llama-quantize[4140724]: segfault at 7dd4d780a9d0 ip 00007eb9b81c634f sp 00007fff3c7bfd40 error 4 in libggml.so[9c634f,7eb9b7815000+9be000] likely on CPU 195 (core 3, socket 1)
[Wed Jul 23 16:36:14 2025] Code: ca 0f 87 80 fe ff ff c5 e8 57 d2 c5 f8 28 c2 e9 7f fe ff ff 8b bd 20 ff ff ff 8b b5 24 ff ff ff 8d 14 fd 00 00 00 00 48 63 d2 <c5> fa 10 04 90 48 8d 14 95 04 00 00 00 c5 fa 11 03 c5 fa 10 04 10

$ #!/usr/bin/env bash

# Repeating Layers [0-61]

custom="
# Attention
blk\..*\.attn_q.*=iq4_kt
blk\..*\.attn_k.*=iq4_kt
blk\..*\.attn_v.*=iq4_kt
blk\..*\.attn_output.*=iq4_kt

# Routed Experts
blk\..*\.ffn_down_exps\.weight=iq3_kt
blk\..*\.ffn_(gate|up)_exps\.weight=iq2_kt

# Non-Repeating Layers
token_embd\.weight=iq4_kt
output\.weight=iq6_k
"

custom=$(
  echo "$custom" | grep -v '^#' | \
  sed -Ez 's:\n+:,:g;s:,$::;s:^,::'
)

numactl -N 1 -m 1 \
./build/bin/llama-quantize \
    --custom-q "$custom" \
    --imatrix /mnt/raid/models/ubergarm/Qwen3-Coder-480B-A35B-Instruct-GGUF/imatrix-Qwen3-Coder-480B-A35B-Instruct-Q8_0.dat \
    /mnt/raid/models/ubergarm/Qwen3-Coder-480B-A35B-Instruct-GGUF/Qwen3-Coder-480B-A35B-Instruct-BF16-00001-of-00021.gguf \
    /mnt/raid/models/ubergarm/Qwen3-Coder-480B-A35B-Instruct-GGUF/Qwen3-Coder-480B-A35B-Instruct-IQ2_KT.gguf \
    IQ2_KT \
    192


main: build = 3823 (fd711836)
main: built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu
main: quantizing '/mnt/raid/models/ubergarm/Qwen3-Coder-480B-A35B-Instruct-GGUF/Qwen3-Coder-480B-A35B-Instruct-BF16-00001-of-00021.gguf' to '/mnt/raid/models/ubergarm/Qwen3-Coder-480B-A35B-Instruct-GGUF/Qwen3-Coder-480B-A35B-Instruct-IQ2_KT.gguf' as IQ2_KT using 192 threads
llama_model_loader: additional 20 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 37 key-value pairs and 747 tensors from /mnt/raid/models/ubergarm/Qwen3-Coder-480B-A35B-Instruct-GGUF/Qwen3-Coder-480B-A35B-Instruct-BF16-00001-of-00021.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3 Coder 480B A35B Instruct
llama_model_loader: - kv   3:                           general.finetune str              = Instruct
llama_model_loader: - kv   4:                           general.basename str              = Qwen3-Coder
llama_model_loader: - kv   5:                         general.size_label str              = 480B-A35B
llama_model_loader: - kv   6:                            general.license str              = apache-2.0
llama_model_loader: - kv   7:                       general.license.link str              = https://huggingface.co/Qwen/Qwen3-Cod...
llama_model_loader: - kv   8:                               general.tags arr[str,1]       = ["text-generation"]
llama_model_loader: - kv   9:                       qwen3moe.block_count u32              = 62
llama_model_loader: - kv  10:                    qwen3moe.context_length u32              = 262144
llama_model_loader: - kv  11:                  qwen3moe.embedding_length u32              = 6144
llama_model_loader: - kv  12:               qwen3moe.feed_forward_length u32              = 8192
llama_model_loader: - kv  13:              qwen3moe.attention.head_count u32              = 96
llama_model_loader: - kv  14:           qwen3moe.attention.head_count_kv u32              = 8
llama_model_loader: - kv  15:                    qwen3moe.rope.freq_base f32              = 10000000.000000
llama_model_loader: - kv  16:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  17:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  18:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  19:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  20:                          general.file_type u32              = 32
llama_model_loader: - kv  21:                      qwen3moe.expert_count u32              = 160
llama_model_loader: - kv  22:        qwen3moe.expert_feed_forward_length u32              = 2560
llama_model_loader: - kv  23: qwen3moe.expert_shared_feed_forward_length u32              = 0
llama_model_loader: - kv  24:               general.quantization_version u32              = 2
llama_model_loader: - kv  25:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  26:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  27:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  28:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  29:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv  30:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  31:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  32:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  33:                    tokenizer.chat_template str              = {% macro render_item_list(item_list, ...
llama_model_loader: - kv  34:                                   split.no u16              = 0
llama_model_loader: - kv  35:                                split.count u16              = 21
llama_model_loader: - kv  36:                        split.tensors.count i32              = 747
llama_model_loader: - type  f32:  311 tensors
llama_model_loader: - type bf16:  436 tensors
================================ Have weights data with 497 entries
[   1/ 747]                    token_embd.weight - [ 6144, 151936,     1,     1], type =   bf16, Using custom type iq4_kt for tensor token_embd.weight

====== llama_model_quantize_internal: did not find weights for token_embd.weight
converting to iq4_kt .. Adding custom rule blk\..*\.attn_q.* -> iq4_kt
Adding custom rule blk\..*\.attn_k.* -> iq4_kt
Adding custom rule blk\..*\.attn_v.* -> iq4_kt
Adding custom rule blk\..*\.attn_output.* -> iq4_kt
Adding custom rule blk\..*\.ffn_down_exps\.weight -> iq3_kt
Adding custom rule blk\..*\.ffn_(gate|up)_exps\.weight -> iq2_kt
Adding custom rule token_embd\.weight -> iq4_kt
Adding custom rule output\.weight -> iq6_k
load_imatrix: imatrix dataset='ubergarm-imatrix-calibration-corpus-v02.txt'
load_imatrix: loaded 497 importance matrix entries from /mnt/raid/models/ubergarm/Qwen3-Coder-480B-A35B-Instruct-GGUF/imatrix-Qwen3-Coder-480B-A35B-Instruct-Q8_0.dat computed on 840 chunks
prepare_imatrix: have 497 importance matrix entries
size =  1780.50 MiB ->   445.70 MiB
[   2/ 747]             blk.0.attn_k_norm.weight - [  128,     1,     1,     1], type =    f32, size =    0.000 MB
[   3/ 747]                  blk.0.attn_k.weight - [ 6144,  1024,     1,     1], type =   bf16, Using custom type iq4_kt for tensor blk.0.attn_k.weight
converting to iq4_kt .. cluster_points: Oops. Cluster 4 has no points:  0 1 0 0
cluster_points: 1 out of 625 clusters dir not have any points
size =    12.00 MiB ->     3.00 MiB
[   4/ 747]             blk.0.attn_output.weight - [12288,  6144,     1,     1], type =   bf16, Using custom type iq4_kt for tensor blk.0.attn_output.weight
converting to iq4_kt .. size =   144.00 MiB ->    36.02 MiB
[   5/ 747]             blk.0.attn_q_norm.weight - [  128,     1,     1,     1], type =    f32, size =    0.000 MB
[   6/ 747]                  blk.0.attn_q.weight - [ 6144, 12288,     1,     1], type =   bf16, Using custom type iq4_kt for tensor blk.0.attn_q.weight
converting to iq4_kt .. size =   144.00 MiB ->    36.05 MiB
[   7/ 747]                  blk.0.attn_v.weight - [ 6144,  1024,     1,     1], type =   bf16, Using custom type iq4_kt for tensor blk.0.attn_v.weight
converting to iq4_kt .. size =    12.00 MiB ->     3.00 MiB
[   8/ 747]               blk.0.attn_norm.weight - [ 6144,     1,     1,     1], type =    f32, size =    0.023 MB
[   9/ 747]           blk.0.ffn_down_exps.weight - [ 2560,  6144,   160,     1], type =   bf16, Using custom type iq3_kt for tensor blk.0.ffn_down_exps.weight
converting to iq3_kt .. ./myscripts/quantize-Qwen3-Coder-480B-A35B-Instruct-v08.sh: line 33: 2323451 Segmentation fault      (core dumped) numactl -N 0 -m 0 ./build/bin/llama-quantize --custom-q "$custom" --imatrix /mnt/raid/models/ubergarm/Qwen3-Coder-480B-A35B-Instruct-GGUF/imatrix-Qwen3-Coder-480B-A35B-Instruct-Q8_0.dat /mnt/raid/models/ubergarm/Qwen3-Coder-480B-A35B-Instruct-GGUF/Qwen3-Coder-480B-A35B-Instruct-BF16-00001-of-00021.gguf /mnt/raid/models/ubergarm/Qwen3-Coder-480B-A35B-Instruct-GGUF/Qwen3-Coder-480B-A35B-Instruct-IQ2_KT.gguf IQ2_KT 192
```

</details>

@ThomasBaruzier 

I can open a 3rd issue for the mla stuff and put all the notes in one place along with ik's above comments and work together to figure out what is going on. thanks!

*EDIT* Here is that issue now: [#651](https://github.com/ikawrakow/ik_llama.cpp/issues/651)

---

👤 **Nexesenex** commented on **2025-07-23** at **21:12:23**

> Not much is known about IQ4_KSS, and nobody seems to be using it. So, I decided to give it some attention.

@Ikawrakow : And now that it has Cuda MMQ, I will use it! Thanks for completing it!

And have a great time off!

---

👤 **ThomasBaruzier** commented on **2025-07-23** at **22:55:11**

> So, in short, just try running without -amb. First with --verbosity 2 to see if the imatrix data collection function gets called with attn_k_b and attn_v_b. If yes, rerun the imatrix calculation that way. If it still doesn't work, it will have to wait until I come back.

Thank you for the detailed explanation! Since I rely on @ubergarm's imatrix due to hardware limitations (no pressure as well), I won't be able to verify this on my end right now. You'll be back in two weeks anyway (have a great time!).

> Just got a laptop with some WiFi

You seem like someone who would really appreciate [Termux](https://github.com/termux/termux-app). Apologies for the poor internet, seems we're all on vacation/away 😅

https://github.com/user-attachments/assets/9cde804a-b6bd-487f-b25e-f2b1848d9394

> I can open a 3rd issue for the mla stuff and put all the notes in one place along with ik's above comments and work together to figure out what is going on

That sounds really nice! Thanks

---

👤 **ubergarm** commented on **2025-07-25** at **22:22:49**

<img width="1785" height="1179" alt="ppl-Qwen3-235B-Thinking-2507" src="https://github.com/user-attachments/assets/62e270ab-2c07-4d2b-a270-4d7480e1fdd7" />

The IQ4_KSS is looking like a pretty good spot for [ubergarm/Qwen3-235B-A22B-Thinking-2507](https://huggingface.co/ubergarm/Qwen3-235B-A22B-Thinking-2507-GGUF)

---

👤 **ubergarm** commented on **2025-07-27** at **18:36:50**

I used Qwen3-Coder-480B-A35B-Instruct-IQ5_K to vibe code up some new matplot lib software and actually fixup my Y-axis log scale more similar to how I've seen some of ik's plots. The IQ4_KSS recipes seem quite strong. They differ *slightly* from each other, exact recipe in links below.

<img width="2067" height="1400" alt="ppl-Qwen3-235B-Instruct-2507" src="https://github.com/user-attachments/assets/bed95d03-af2a-4f78-8a99-bb8de5f1b9d2" />

<img width="2067" height="1400" alt="ppl-Qwen3-235B-Thinking-2507" src="https://github.com/user-attachments/assets/54e9cac5-da3a-4fc1-80b7-7ac5f290c7d8" />

* https://huggingface.co/ubergarm/Qwen3-235B-A22B-Instruct-2507-GGUF#iq4_kss-115085-gib-4205-bpw
* https://huggingface.co/ubergarm/Qwen3-235B-A22B-Thinking-2507-GGUF#iq4_kss-114093-gib-4169-bpw

*UPDATE*

And just finished up the bigger [Qwen3-Coder-480B-A35B-Instruct-GGUF IQ4_KSS](https://huggingface.co/ubergarm/Qwen3-Coder-480B-A35B-Instruct-GGUF#iq4_kss-233676-gib-4180-bpw)

<img width="2069" height="1400" alt="Qwen3-Coder-480B-ppl" src="https://github.com/user-attachments/assets/7cf0eebf-f8bc-488c-941a-8f6a0dc023e8" />

(*note that the IQ2_K here is using iq2_kl as ffn_down_exps instead of larger iq3_k so it is right in line with what the IQ2_KS would be for size and PPL).