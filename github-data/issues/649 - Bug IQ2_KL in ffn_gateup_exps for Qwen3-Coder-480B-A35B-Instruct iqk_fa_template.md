## 📌 [Issue #649](https://github.com/ikawrakow/ik_llama.cpp/issues/649) - Bug: IQ2_KL in ffn_(gate|up)_exps for Qwen3-Coder-480B-A35B-Instruct `iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed`

| **Author** | `ubergarm` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2025-07-26 |
| **Updated** | 2025-07-26 |

---

## 📄 Description

### What happened?

I cooked a quant using IQ2_KL in ffn_(gate|up)_exps tensors for Qwen3-Coder-480B-A35B-Instruct. However, when trying to run it throws  `iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed` and exits on startup. Compiled CPU-only backend.

Oddly though, I have a different quant using IQ2_KL in only the ffn_down tensors and it works fine. It is identical to the recipe except for slightly larger routed exps:
```
# Routed Experts
blk\..*\.ffn_down_exps\.weight=iq2_kl
blk\..*\.ffn_(gate|up)_exps\.weight=iq2_k
```

Finally just to test, I tried removing `-fmoe` but it failed with the same assert error. Removing `-fa` it exited with this error: `Oops(ggml_compute_forward_sum_rows_f32, ffn_moe_weights_sum-2): found nan for i1 = 0, i2 = 0, i3 = 0. ne00 = 160`

I noticed this issue before releasing the quant so no rush. If I have time I might try rolling back to earlier version to see if it was possibly a regression.

*EDIT*: Also my recent `Kimi-K2-Instruct-IQ2_KL` quants are working fine too:
```
Adding custom rule blk\..*\.ffn_down_exps\.weight -> iq3_ks
# Adding custom rule blk\..*\.ffn_down_exps\.weight -> iq2_kl # <--- i have one with all exps iq2_kl also
Adding custom rule blk\..*\.ffn_(gate|up)_exps\.weight -> iq2_kl
```

<details>

<summary>👈 IQ2_KL Quant Recipe</summary>

This is the quant that throws the error:
```bash
#!/usr/bin/env bash

# Repeating Layers [0-61]

custom="
# Attention
blk\..*\.attn_q.*=iq6_k
blk\..*\.attn_k.*=q8_0
blk\..*\.attn_v.*=q8_0
blk\..*\.attn_output.*=iq6_k

# Routed Experts
blk\..*\.ffn_down_exps\.weight=iq3_k
blk\..*\.ffn_(gate|up)_exps\.weight=iq2_kl

# Non-Repeating Layers
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
    --imatrix /mnt/raid/models/ubergarm/Qwen3-Coder-480B-A35B-Instruct-GGUF/imatrix-Qwen3-Coder-480B-A35B-Instruct-Q8_0.dat \
    /mnt/raid/models/ubergarm/Qwen3-Coder-480B-A35B-Instruct-GGUF/Qwen3-Coder-480B-A35B-Instruct-BF16-00001-of-00021.gguf \
    /mnt/raid/models/ubergarm/Qwen3-Coder-480B-A35B-Instruct-GGUF/Qwen3-Coder-480B-A35B-Instruct-IQ2_KL.gguf \
    IQ2_KL \
    192
```

</details>


<details>

<summary>👈 llama-server command, log and error</summary>

```bash
# compile CPU-only backend

model=/mnt/raid/hf/Qwen3-Coder-480B-A35B-Instruct-GGUF/IQ2_KL/Qwen3-480B-A35B-Instruct-IQ2_KL-00001-of-00004.gguf

numactl -N 1 -m 1 \
./build/bin/llama-server \
    --model "$model"\
    --alias ubergarm/Qwen3-Coder-480B-A35B-Instruct \
    --ctx-size 196608 \
    -ctk q8_0 -ctv q8_0 \
    -fa -fmoe \
    -ub 4096 -b 4096 \
    --parallel 1 \
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


### Name and Version

$ ./build/bin/llama-server --version
version: 3822 (4e9c78c0)
built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell

```