### 📝 [#296](https://github.com/ikawrakow/ik_llama.cpp/issues/296) - Possible numerical stability issue with experimental quant of DeepSeek-V3-0324?

| **Author** | `ubergarm` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-03-30 |
| **Updated** | 2025-04-06 |

---

#### Description

## tl;dr;
*UPDATE*: skip to the end, I probably shouldn't use `q8_0_r8` for `token_embd.weight` and just leave that `q8_0`.

I cooked up a `DeepSeek-V3-0324` quant specificly for CPU only inferencing on the xeon 6980P rig and am getting very large perplexity values and broken llama-server responses.

Not sure if user error, an invalid recipe, or if there is some issue with computing one of the quant types etc.

## Details

This was my intended recipe mix:

* `q8_0_r8` for all the embeddings, attention, norms, bias, and shared experts tensors
* `q5_k_r4` for all routed MoE down projection tensors
* `q4_k_r4` for all routed MoE gate/up tensors

This is what is reported when starting up with it:
```
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0_r8:  612 tensors
llama_model_loader: - type iq4_k_r4:  116 tensors
llama_model_loader: - type iq5_k_r4:   58 tensors
```

I'm not 100% sure if the issue could be with the `q5_k_r4` or `q4_k_r4` inferencing CPU computation possibly? Or maybe I messed up somewhere in my scripts.

Potentially relevent topics:
1. Recent PR292 seems to have fixed the previous issue with `q8_0` numerical stability.
2. I asked @saood06 as he has been experimenting with these quants in [our discussion here](https://github.com/ikawrakow/ik_llama.cpp/discussions/286#discussioncomment-12668598).

## Logs

I've provided logs of quantization, perplexity, and llama-server below for reference.

Everything rebuilt and run on updated `ik_llama.cpp/main@4819257c`.

<details>

<summary>Quantization Procedure</summary>

#### Quantization Recipe Script
```bash
#!/usr/bin/env bash

custom="
# Token embedding and output tensors
token_embd\.weight=q8_0_r8
output\.weight=q8_0_r8
output_norm\.weight=q8_0_r8

# First 3 dense layers (0-3)
blk\.[0-2]\..*=q8_0_r8

# All attention, norm weights, and bias tensors for MoE layers (3-60)
blk\.[3-9]\.attn_.*=q8_0_r8
blk\.[1-5][0-9]\.attn_.*=q8_0_r8
blk\.60\.attn_.*=q8_0_r8

blk\.[3-9]\.ffn_norm\.weight=q8_0_r8
blk\.[1-5][0-9]\.ffn_norm\.weight=q8_0_r8
blk\.60\.ffn_norm\.weight=q8_0_r8

blk\.[3-9]\.exp_probs_b\.bias=q8_0_r8
blk\.[1-5][0-9]\.exp_probs_b\.bias=q8_0_r8
blk\.60\.exp_probs_b\.bias=q8_0_r8

# Shared Experts (3-60)
blk\.[3-9]\.ffn_down_shexp\.weight=q8_0_r8
blk\.[1-5][0-9]\.ffn_down_shexp\.weight=q8_0_r8
blk\.60\.ffn_down_shexp\.weight=q8_0_r8

blk\.[3-9]\.ffn_(gate|up)_shexp\.weight=q8_0_r8
blk\.[1-5][0-9]\.ffn_(gate|up)_shexp\.weight=q8_0_r8
blk\.60\.ffn_(gate|up)_shexp\.weight=q8_0_r8

# MoE Experts (3-60)
blk\.[3-9]\.ffn_down_exps\.weight=iq5_k_r4
blk\.[1-5][0-9]\.ffn_down_exps\.weight=iq5_k_r4
blk\.60\.ffn_down_exps\.weight=iq5_k_r4

blk\.[3-9]\.ffn_(gate|up)_exps\.weight=iq4_k_r4
blk\.[1-5][0-9]\.ffn_(gate|up)_exps\.weight=iq4_k_r4
blk\.60\.ffn_(gate|up)_exps\.weight=iq4_k_r4
"

custom=$(
  echo "$custom" | grep -v '^#' | \
  sed -Ez 's:\n+:,:g;s:,$::;s:^,::'
)

./build/bin/llama-quantize \
    --imatrix /mnt/raid/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324.imatrix \
    --token-embedding-type q8_0_r8 \
    --output-tensor-type q8_0_r8 \
    --custom-q "$custom" \
    /mnt/raid/models/deepseek-ai/DeepSeek-V3-0324-bf16-GGUF/DeepSeek-256x21B-V3-0324-BF16-00001-of-00030.gguf \
    /mnt/raid/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-CPU-IQ4_K_R4.gguf \
    IQ4_K_R4 \
    24
```

#### Output Logs
```bash
main: build = 3613 (4819257c)
main: built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu
main: quantizing '/mnt/raid/models/deepseek-ai/DeepSeek-V3-0324-bf16-GGUF/DeepSeek-256x21B-V3-0324-BF16-00001-of-00030.gguf' to '/mnt/raid/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-CPU-IQ4_K_R4.gguf' as IQ4_K_R4 using 24 threads
llama_model_loader: additional 29 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 49 key-value pairs and 1147 tensors from /mnt/raid/models/deepseek-ai/DeepSeek-V3-0324-bf16-GGUF/DeepSeek-256x21B-V3-0324-BF16-00001-of-00030.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = DeepSeek V3 0324
llama_model_loader: - kv   3:                            general.version str              = V3-0324
llama_model_loader: - kv   4:                           general.basename str              = DeepSeek
llama_model_loader: - kv   5:                         general.size_label str              = 256x21B
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
llama_model_loader: - kv  16:                          general.file_type u32              = 32
llama_model_loader: - kv  17:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  18:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  19:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  20:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  21:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  22:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  23:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  24:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  25:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  26:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  27:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  28:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  29:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  30:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  31:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  32: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  33: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  34:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  35:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  36:                      tokenizer.ggml.tokens arr[str,129280]  = ["
llama_model_loader: - kv  37:                  tokenizer.ggml.token_type arr[i32,129280]  = [3
llama_model_loader: - kv  38:                      tokenizer.ggml.merges arr[str,127741]  = ["
llama_model_loader: - kv  39:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  40:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  41:            tokenizer.ggml.padding_token_id u32              = 1
llama_model_loader: - kv  42:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  43:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  44:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  45:               general.quantization_version u32              = 2
llama_model_loader: - kv  46:                                   split.no u16              = 0
llama_model_loader: - kv  47:                                split.count u16              = 30
llama_model_loader: - kv  48:                        split.tensors.count i32              = 1147
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type bf16:  786 tensors
================================ Have weights data with 720 entries
[   1/1147]                    token_embd.weight - [ 7168, 129280,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor token_embd.weight

====== llama_model_quantize_internal: did not find weights for token_embd.weight
converting to q8_0_r8 .. Adding custom rule token_embd\.weight -> q8_0_r8
Adding custom rule output\.weight -> q8_0_r8
Adding custom rule output_norm\.weight -> q8_0_r8
Adding custom rule blk\.[0-2]\..* -> q8_0_r8
Adding custom rule blk\.[3-9]\.attn_.* -> q8_0_r8
Adding custom rule blk\.[1-5][0-9]\.attn_.* -> q8_0_r8
Adding custom rule blk\.60\.attn_.* -> q8_0_r8
Adding custom rule blk\.[3-9]\.ffn_norm\.weight -> q8_0_r8
Adding custom rule blk\.[1-5][0-9]\.ffn_norm\.weight -> q8_0_r8
Adding custom rule blk\.60\.ffn_norm\.weight -> q8_0_r8
Adding custom rule blk\.[3-9]\.exp_probs_b\.bias -> q8_0_r8
Adding custom rule blk\.[1-5][0-9]\.exp_probs_b\.bias -> q8_0_r8
Adding custom rule blk\.60\.exp_probs_b\.bias -> q8_0_r8
Adding custom rule blk\.[3-9]\.ffn_down_shexp\.weight -> q8_0_r8
Adding custom rule blk\.[1-5][0-9]\.ffn_down_shexp\.weight -> q8_0_r8
Adding custom rule blk\.60\.ffn_down_shexp\.weight -> q8_0_r8
Adding custom rule blk\.[3-9]\.ffn_(gate|up)_shexp\.weight -> q8_0_r8
Adding custom rule blk\.[1-5][0-9]\.ffn_(gate|up)_shexp\.weight -> q8_0_r8
Adding custom rule blk\.60\.ffn_(gate|up)_shexp\.weight -> q8_0_r8
Adding custom rule blk\.[3-9]\.ffn_down_exps\.weight -> iq5_k_r4
Adding custom rule blk\.[1-5][0-9]\.ffn_down_exps\.weight -> iq5_k_r4
Adding custom rule blk\.60\.ffn_down_exps\.weight -> iq5_k_r4
Adding custom rule blk\.[3-9]\.ffn_(gate|up)_exps\.weight -> iq4_k_r4
Adding custom rule blk\.[1-5][0-9]\.ffn_(gate|up)_exps\.weight -> iq4_k_r4
Adding custom rule blk\.60\.ffn_(gate|up)_exps\.weight -> iq4_k_r4
load_imatrix: imatrix dataset='calibration_data_v5_rc.txt'
load_imatrix: loaded 720 importance matrix entries from /mnt/raid/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324.imatrix computed on 213 chunks
prepare_imatrix: have 720 importance matrix entries
size =  1767.50 MiB ->   938.98 MiB
[   2/1147]               blk.0.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[   3/1147]                blk.0.ffn_down.weight - [18432,  7168,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.0.ffn_down.weight
converting to q8_0_r8 .. size =   252.00 MiB ->   133.88 MiB
[   4/1147]                blk.0.ffn_gate.weight - [ 7168, 18432,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.0.ffn_gate.weight
converting to q8_0_r8 .. size =   252.00 MiB ->   133.88 MiB
[   5/1147]                  blk.0.ffn_up.weight - [ 7168, 18432,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.0.ffn_up.weight
converting to q8_0_r8 .. size =   252.00 MiB ->   133.88 MiB
[   6/1147]                blk.0.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[   7/1147]          blk.0.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[   8/1147]           blk.0.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.0.attn_kv_a_mqa.weight
converting to q8_0_r8 .. size =     7.88 MiB ->     4.18 MiB
[   9/1147]               blk.0.attn_kv_b.weight - [  512, 32768,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.0.attn_kv_b.weight
converting to q8_0_r8 .. size =    32.00 MiB ->    17.00 MiB
[  10/1147]                blk.0.attn_k_b.weight - [  128, 65536,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.0.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.0.attn_k_b.weight
converting to q8_0_r8 .. size =    16.00 MiB ->     8.50 MiB
[  11/1147]                blk.0.attn_v_b.weight - [  512, 16384,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.0.attn_v_b.weight
converting to q8_0_r8 .. size =    16.00 MiB ->     8.50 MiB
[  12/1147]             blk.0.attn_output.weight - [16384,  7168,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.0.attn_output.weight
converting to q8_0_r8 .. size =   224.00 MiB ->   119.00 MiB
[  13/1147]           blk.0.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  14/1147]                blk.0.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.0.attn_q_a.weight
converting to q8_0_r8 .. size =    21.00 MiB ->    11.16 MiB
[  15/1147]                blk.0.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.0.attn_q_b.weight
converting to q8_0_r8 .. size =    72.00 MiB ->    38.25 MiB
[  16/1147]               blk.1.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  17/1147]                blk.1.ffn_down.weight - [18432,  7168,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.1.ffn_down.weight
converting to q8_0_r8 .. size =   252.00 MiB ->   133.88 MiB
[  18/1147]                blk.1.ffn_gate.weight - [ 7168, 18432,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.1.ffn_gate.weight
converting to q8_0_r8 .. size =   252.00 MiB ->   133.88 MiB
[  19/1147]                  blk.1.ffn_up.weight - [ 7168, 18432,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.1.ffn_up.weight
converting to q8_0_r8 .. size =   252.00 MiB ->   133.88 MiB
[  20/1147]                blk.1.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  21/1147]          blk.1.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  22/1147]           blk.1.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.1.attn_kv_a_mqa.weight
converting to q8_0_r8 .. size =     7.88 MiB ->     4.18 MiB
[  23/1147]               blk.1.attn_kv_b.weight - [  512, 32768,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.1.attn_kv_b.weight
converting to q8_0_r8 .. size =    32.00 MiB ->    17.00 MiB
[  24/1147]                blk.1.attn_k_b.weight - [  128, 65536,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.1.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.1.attn_k_b.weight
converting to q8_0_r8 .. size =    16.00 MiB ->     8.50 MiB
[  25/1147]                blk.1.attn_v_b.weight - [  512, 16384,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.1.attn_v_b.weight
converting to q8_0_r8 .. size =    16.00 MiB ->     8.50 MiB
[  26/1147]             blk.1.attn_output.weight - [16384,  7168,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.1.attn_output.weight
converting to q8_0_r8 .. size =   224.00 MiB ->   119.00 MiB
[  27/1147]           blk.1.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  28/1147]                blk.1.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.1.attn_q_a.weight
converting to q8_0_r8 .. size =    21.00 MiB ->    11.16 MiB
[  29/1147]                blk.1.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.1.attn_q_b.weight
converting to q8_0_r8 .. size =    72.00 MiB ->    38.25 MiB
[  30/1147]               blk.2.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  31/1147]                blk.2.ffn_down.weight - [18432,  7168,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.2.ffn_down.weight
converting to q8_0_r8 .. size =   252.00 MiB ->   133.88 MiB
[  32/1147]                blk.2.ffn_gate.weight - [ 7168, 18432,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.2.ffn_gate.weight
converting to q8_0_r8 .. size =   252.00 MiB ->   133.88 MiB
[  33/1147]                  blk.2.ffn_up.weight - [ 7168, 18432,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.2.ffn_up.weight
converting to q8_0_r8 .. size =   252.00 MiB ->   133.88 MiB
[  34/1147]                blk.2.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  35/1147]          blk.2.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  36/1147]           blk.2.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.2.attn_kv_a_mqa.weight
converting to q8_0_r8 .. size =     7.88 MiB ->     4.18 MiB
[  37/1147]               blk.2.attn_kv_b.weight - [  512, 32768,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.2.attn_kv_b.weight
converting to q8_0_r8 .. size =    32.00 MiB ->    17.00 MiB
[  38/1147]                blk.2.attn_k_b.weight - [  128, 65536,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.2.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.2.attn_k_b.weight
converting to q8_0_r8 .. size =    16.00 MiB ->     8.50 MiB
[  39/1147]                blk.2.attn_v_b.weight - [  512, 16384,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.2.attn_v_b.weight
converting to q8_0_r8 .. size =    16.00 MiB ->     8.50 MiB
[  40/1147]             blk.2.attn_output.weight - [16384,  7168,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.2.attn_output.weight
converting to q8_0_r8 .. size =   224.00 MiB ->   119.00 MiB
[  41/1147]           blk.2.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  42/1147]                blk.2.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.2.attn_q_a.weight
converting to q8_0_r8 .. size =    21.00 MiB ->    11.16 MiB
[  43/1147]                blk.2.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.2.attn_q_b.weight
converting to q8_0_r8 .. size =    72.00 MiB ->    38.25 MiB
[  44/1147]               blk.3.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[  45/1147]            blk.3.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[  46/1147]          blk.3.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.3.ffn_down_shexp.weight
converting to q8_0_r8 .. size =    28.00 MiB ->    14.88 MiB
[  47/1147]          blk.3.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.3.ffn_gate_shexp.weight
converting to q8_0_r8 .. size =    28.00 MiB ->    14.88 MiB
[  48/1147]            blk.3.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.3.ffn_up_shexp.weight
converting to q8_0_r8 .. size =    28.00 MiB ->    14.88 MiB
[  49/1147]          blk.3.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  50/1147]           blk.3.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.3.attn_kv_a_mqa.weight
converting to q8_0_r8 .. size =     7.88 MiB ->     4.18 MiB
[  51/1147]               blk.3.attn_kv_b.weight - [  512, 32768,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.3.attn_kv_b.weight
converting to q8_0_r8 .. size =    32.00 MiB ->    17.00 MiB
[  52/1147]                blk.3.attn_k_b.weight - [  128, 65536,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.3.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.3.attn_k_b.weight
converting to q8_0_r8 .. size =    16.00 MiB ->     8.50 MiB
[  53/1147]                blk.3.attn_v_b.weight - [  512, 16384,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.3.attn_v_b.weight
converting to q8_0_r8 .. size =    16.00 MiB ->     8.50 MiB
[  54/1147]             blk.3.attn_output.weight - [16384,  7168,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.3.attn_output.weight
converting to q8_0_r8 .. size =   224.00 MiB ->   119.00 MiB
[  55/1147]           blk.3.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  56/1147]                blk.3.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.3.attn_q_a.weight
converting to q8_0_r8 .. size =    21.00 MiB ->    11.16 MiB
[  57/1147]                blk.3.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.3.attn_q_b.weight
converting to q8_0_r8 .. size =    72.00 MiB ->    38.25 MiB
[  58/1147]               blk.3.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  59/1147]           blk.3.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   bf16, Using custom type iq5_k_r4 for tensor blk.3.ffn_down_exps.weight
converting to iq5_k_r4 .. size =  7168.00 MiB ->  2464.00 MiB
[  60/1147]           blk.3.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   bf16, Using custom type iq4_k_r4 for tensor blk.3.ffn_gate_exps.weight
converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[  61/1147]             blk.3.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   bf16, Using custom type iq4_k_r4 for tensor blk.3.ffn_up_exps.weight
converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[  62/1147]                blk.3.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  63/1147]               blk.4.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[  64/1147]            blk.4.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[  65/1147]          blk.4.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.4.ffn_down_shexp.weight
converting to q8_0_r8 .. size =    28.00 MiB ->    14.88 MiB
[  66/1147]          blk.4.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.4.ffn_gate_shexp.weight
converting to q8_0_r8 .. size =    28.00 MiB ->    14.88 MiB
[  67/1147]            blk.4.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.4.ffn_up_shexp.weight
converting to q8_0_r8 .. size =    28.00 MiB ->    14.88 MiB
[  68/1147]          blk.4.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  69/1147]           blk.4.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.4.attn_kv_a_mqa.weight
converting to q8_0_r8 .. size =     7.88 MiB ->     4.18 MiB
[  70/1147]               blk.4.attn_kv_b.weight - [  512, 32768,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.4.attn_kv_b.weight
converting to q8_0_r8 .. size =    32.00 MiB ->    17.00 MiB
[  71/1147]                blk.4.attn_k_b.weight - [  128, 65536,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.4.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.4.attn_k_b.weight
converting to q8_0_r8 .. size =    16.00 MiB ->     8.50 MiB
[  72/1147]                blk.4.attn_v_b.weight - [  512, 16384,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.4.attn_v_b.weight
converting to q8_0_r8 .. size =    16.00 MiB ->     8.50 MiB
[  73/1147]             blk.4.attn_output.weight - [16384,  7168,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.4.attn_output.weight
converting to q8_0_r8 .. size =   224.00 MiB ->   119.00 MiB
[  74/1147]           blk.4.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  75/1147]                blk.4.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.4.attn_q_a.weight
converting to q8_0_r8 .. size =    21.00 MiB ->    11.16 MiB
[  76/1147]                blk.4.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.4.attn_q_b.weight
converting to q8_0_r8 .. size =    72.00 MiB ->    38.25 MiB
[  77/1147]               blk.4.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  78/1147]           blk.4.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   bf16, Using custom type iq5_k_r4 for tensor blk.4.ffn_down_exps.weight
converting to iq5_k_r4 .. size =  7168.00 MiB ->  2464.00 MiB
[  79/1147]           blk.4.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   bf16, Using custom type iq4_k_r4 for tensor blk.4.ffn_gate_exps.weight
converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[  80/1147]             blk.4.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   bf16, Using custom type iq4_k_r4 for tensor blk.4.ffn_up_exps.weight
converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[  81/1147]                blk.4.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  82/1147]          blk.5.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  83/1147]           blk.5.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.5.attn_kv_a_mqa.weight
converting to q8_0_r8 .. size =     7.88 MiB ->     4.18 MiB
[  84/1147]               blk.5.attn_kv_b.weight - [  512, 32768,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.5.attn_kv_b.weight
converting to q8_0_r8 .. size =    32.00 MiB ->    17.00 MiB
[  85/1147]                blk.5.attn_k_b.weight - [  128, 65536,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.5.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.5.attn_k_b.weight

# SNIP text was too long for github issues

====== llama_model_quantize_internal: did not find weights for blk.59.attn_k_b.weight
converting to q8_0_r8 .. size =    16.00 MiB ->     8.50 MiB
[1117/1147]               blk.59.attn_v_b.weight - [  512, 16384,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.59.attn_v_b.weight
converting to q8_0_r8 .. size =    16.00 MiB ->     8.50 MiB
[1118/1147]            blk.59.attn_output.weight - [16384,  7168,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.59.attn_output.weight
converting to q8_0_r8 .. size =   224.00 MiB ->   119.00 MiB
[1119/1147]          blk.59.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1120/1147]               blk.59.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.59.attn_q_a.weight
converting to q8_0_r8 .. size =    21.00 MiB ->    11.16 MiB
[1121/1147]               blk.59.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.59.attn_q_b.weight
converting to q8_0_r8 .. size =    72.00 MiB ->    38.25 MiB
[1122/1147]              blk.59.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1123/1147]          blk.59.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   bf16, Using custom type iq5_k_r4 for tensor blk.59.ffn_down_exps.weight
converting to iq5_k_r4 .. size =  7168.00 MiB ->  2464.00 MiB
[1124/1147]          blk.59.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   bf16, Using custom type iq4_k_r4 for tensor blk.59.ffn_gate_exps.weight
converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[1125/1147]            blk.59.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   bf16, Using custom type iq4_k_r4 for tensor blk.59.ffn_up_exps.weight
converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[1126/1147]               blk.59.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1127/1147]              blk.60.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1128/1147]           blk.60.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1129/1147]         blk.60.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.60.ffn_down_shexp.weight
converting to q8_0_r8 .. size =    28.00 MiB ->    14.88 MiB
[1130/1147]         blk.60.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.60.ffn_gate_shexp.weight
converting to q8_0_r8 .. size =    28.00 MiB ->    14.88 MiB
[1131/1147]           blk.60.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.60.ffn_up_shexp.weight
converting to q8_0_r8 .. size =    28.00 MiB ->    14.88 MiB
[1132/1147]         blk.60.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1133/1147]          blk.60.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.60.attn_kv_a_mqa.weight
converting to q8_0_r8 .. size =     7.88 MiB ->     4.18 MiB
[1134/1147]              blk.60.attn_kv_b.weight - [  512, 32768,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.60.attn_kv_b.weight
converting to q8_0_r8 .. size =    32.00 MiB ->    17.00 MiB
[1135/1147]               blk.60.attn_k_b.weight - [  128, 65536,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.60.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.60.attn_k_b.weight
converting to q8_0_r8 .. size =    16.00 MiB ->     8.50 MiB
[1136/1147]               blk.60.attn_v_b.weight - [  512, 16384,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.60.attn_v_b.weight
converting to q8_0_r8 .. size =    16.00 MiB ->     8.50 MiB
[1137/1147]            blk.60.attn_output.weight - [16384,  7168,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.60.attn_output.weight
converting to q8_0_r8 .. size =   224.00 MiB ->   119.00 MiB
[1138/1147]          blk.60.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1139/1147]               blk.60.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.60.attn_q_a.weight
converting to q8_0_r8 .. size =    21.00 MiB ->    11.16 MiB
[1140/1147]               blk.60.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.60.attn_q_b.weight
converting to q8_0_r8 .. size =    72.00 MiB ->    38.25 MiB
[1141/1147]                        output.weight - [ 7168, 129280,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor output.weight

====== llama_model_quantize_internal: did not find weights for output.weight
converting to q8_0_r8 .. size =  1767.50 MiB ->   938.98 MiB
[1142/1147]              blk.60.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1143/1147]          blk.60.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   bf16, Using custom type iq5_k_r4 for tensor blk.60.ffn_down_exps.weight
converting to iq5_k_r4 .. size =  7168.00 MiB ->  2464.00 MiB
[1144/1147]          blk.60.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   bf16, Using custom type iq4_k_r4 for tensor blk.60.ffn_gate_exps.weight
converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[1145/1147]            blk.60.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   bf16, Using custom type iq4_k_r4 for tensor blk.60.ffn_up_exps.weight
converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[1146/1147]               blk.60.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1147/1147]                   output_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
llama_model_quantize_internal: model size  = 1282038.27 MB
llama_model_quantize_internal: quant size  = 395450.97 MB

main: quantize time = 5308904.06 ms
main:    total time = 5308904.06 ms
```

</details>


<details>

<summary>Perplexity Procedure</summary>

#### Output Logs
```bash
$ numactl -N 1 -m 1 \
./build/bin/llama-perplexity \
    --model /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-CPU-IQ4_K_R4.gguf \
    -ctk q8_0 \
    -mla 3 -fa \
    -amb 512 \
    -fmoe \
    --ctx-size 512 \
    --ubatch-size 512 \
    -f wiki.test.raw \
    --seed 1337 \
    --numa numactl \
    --threads 128

main: build = 3613 (4819257c)
main: built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu
main: seed  = 1337
llama_model_loader: loaded meta data with 50 key-value pairs and 1147 tensors from /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-CPU-
IQ4_K_R4.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = DeepSeek V3 0324
llama_model_loader: - kv   3:                            general.version str              = V3-0324
llama_model_loader: - kv   4:                           general.basename str              = DeepSeek
llama_model_loader: - kv   5:                         general.size_label str              = 256x21B
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
llama_model_loader: - kv  16:                          general.file_type u32              = 340
llama_model_loader: - kv  17:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  18:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  19:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  20:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  21:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  22:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  23:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  24:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  25:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  26:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  27:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  28:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  29:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  30:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  31:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  32: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  33: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  34:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  35:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  36:                      tokenizer.ggml.tokens arr[str,129280]  = ["
llama_model_loader: - kv  37:                  tokenizer.ggml.token_type arr[i32,129280]  = [3
llama_model_loader: - kv  38:                      tokenizer.ggml.merges arr[str,127741]  = ["
llama_model_loader: - kv  39:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  40:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  41:            tokenizer.ggml.padding_token_id u32              = 1
llama_model_loader: - kv  42:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  43:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  44:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  45:               general.quantization_version u32              = 2
llama_model_loader: - kv  46:                      quantize.imatrix.file str              = /mnt/raid/models/ubergarm/DeepSeek-V3...
llama_model_loader: - kv  47:                   quantize.imatrix.dataset str              = calibration_data_v5_rc.txt
llama_model_loader: - kv  48:             quantize.imatrix.entries_count i32              = 720
llama_model_loader: - kv  49:              quantize.imatrix.chunks_count i32              = 213
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0_r8:  612 tensors
llama_model_loader: - type iq4_k_r4:  116 tensors
llama_model_loader: - type iq5_k_r4:   58 tensors
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
llm_load_print_meta: model ftype      = IQ4_K_R4 - 4.5 bpw
llm_load_print_meta: model params     = 672.050 B
llm_load_print_meta: model size       = 386.183 GiB (4.936 BPW)
llm_load_print_meta: repeating layers = 384.349 GiB (4.926 BPW, 670.196 B parameters)
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
llm_load_tensors: ggml ctx size =    0.47 MiB
llm_load_tensors:        CPU buffer size = 395450.97 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 2048
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 3
llama_new_context_with_model: attn_max_b = 512
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init: layer 0: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 1: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 2: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 3: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 4: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 5: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 6: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 7: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 8: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 9: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 10: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 11: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 12: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 13: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 14: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 15: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 16: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 17: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 18: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 19: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 20: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 21: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 22: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 23: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 24: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 25: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 26: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 27: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 28: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 29: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 30: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 31: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 32: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 33: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 34: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 35: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 36: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 37: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 38: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 39: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 40: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 41: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 42: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 43: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 44: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 45: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 46: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 47: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 48: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 49: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 50: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 51: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 52: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 53: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 54: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 55: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 56: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 57: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 58: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 59: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 60: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init:        CPU KV buffer size =    72.91 MiB
llama_new_context_with_model: KV self size  =   72.91 MiB, c^KV (q8_0):   72.91 MiB, kv^T: not used
llama_new_context_with_model:        CPU  output buffer size =     1.97 MiB
llama_new_context_with_model:        CPU compute buffer size =   450.01 MiB
llama_new_context_with_model: graph nodes  = 3487
llama_new_context_with_model: graph splits = 1

system_info: n_threads = 128 / 512 | AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 |
NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE =
1 |
perplexity: tokenizing the input ..
perplexity: tokenization took 928.692 ms
perplexity: calculating perplexity over 561 chunks, n_ctx=512, batch_size=2048, n_seq=4
perplexity: 15.08 seconds per pass - ETA 35.23 minutes
[1]621042.4845,[2]480288.4154,[3]384849.5504,[4]411291.6749,[5]342382.0527,[6]347496.7446,[7]338598.0612,[8]338938.1630,[9]343341.0863,[10]329407.7871
,[11]328794.0950,[12]349036.2429,[13]339812.6162,[14]327127.2843,[15]318294.3349,[16]320629.0762,[17]318911.2283,[18]306946.2653,[19]320742.9747,[20]3
20520.4166,[21]323369.9752,[22]321108.7583,[23]320950.8245,[24]323537.1597,[25]313530.9380,[26]307858.8254,[27]305584.6174,[28]304930.6946,[29]319325.
7633,[30]316463.6020,[31]318028.8556,[32]323730.7568,[33]336376.3859,[34]338644.6368,[35]341295.5596,[36]346582.1772,[37]343638.6921,[38]346920.0126,[
39]346553.2755,[40]339975.1907,[41]338080.6482,[42]341607.0511,[43]342165.4351,[44]343495.4481,[45]341683.0497,[46]341841.5203,[47]341968.2578,[48]341
018.8794,[49]337906.3680,[50]340880.4017,[51]343264.7780,[52]341172.5260,[53]341895.8030,[54]342362.6716,[55]339077.7577,[56]337472.3629,[57]338597.41
79,[58]338840.4233,[59]340391.7068,[60]341329.9617,[61]338907.2644,[62]338654.8390,[63]340597.6581,[64]341464.0272,[65]339761.5866,[66]337473.1508,[67
]334628.2254,[68]335027.1919,[69]336085.7135,[70]334748.0318,[71]334310.4754,[72]332610.8172,[73]331121.5117,[74]331604.3876,[75]331320.1529,[76]33491
0.8814,[77]336051.4006,[78]335753.6115,[79]337362.5269,[80]335564.3466,[81]332456.8750,[82]331609.4385,[83]333316.4520,[84]335084.6156,[85]334711.4110
,[86]334160.7888,[87]332126.7278,[88]331597.7024,[89]331461.8908,[90]330703.9912,[91]331143.7667,[92]328566.8218,[93]327220.3991,[94]327306.2202,[95]3
28760.6069,[96]331831.1512,[97]331100.4377,[98]331676.2039,[99]331115.3237,[100]332922.5225,[101]330521.2050,[102]330638.9063,[103]330508.2943,[104]33
3336.3249,[105]332252.4134,[106]331511.8882,[107]331478.9005,[108]330800.7499,[109]331643.0452,[110]332295.2747,[111]331716.4016,[112]333145.4543,[113
]332446.6042,[114]332605.4088,[115]334144.7878,[116]334062.6775,[117]334795.9300,[118]335185.6388,[119]336442.8975,[120]336288.3524,[121]337854.3067,[
122]342121.8593,[123]342443.4687,[124]343659.0524,[125]344785.3775,[126]345809.3526,[127]347207.6305,[128]348210.4479,[129]349672.3288,[130]350221.461
2,[131]350215.0059,[132]352167.2450,[133]351660.6672,[134]353361.5754,[135]354848.8108,[136]353175.7897,[137]353870.5511,[138]355061.4101,[139]355874.
4197,[140]356669.3123,[141]355293.1474,[142]354584.2063,[143]353505.6443,[144]354011.7258,[145]352950.0290,[146]352775.3758,[147]350332.0398,[148]3489
19.1460,[149]348589.1782,[150]348457.2881,[151]347884.5859,[152]347551.9711,[153]346394.1977,[154]345076.4034,[155]342799.4862,[156]342481.4941,[157]3
42472.8007,[158]341437.5809,[159]341069.4855,[160]340176.4801,[161]340547.0153,[162]341245.8648,[163]340449.0528,[164]339162.6069,[165]339049.6867,[16
6]340108.0202,[167]338993.8220,[168]338633.1774,[169]337653.7408,[170]337330.2507,[171]337964.2748,[172]336817.5461,[173]335656.4557,[174]335356.9395,
[175]335636.9791,[176]336962.6238,[177]336571.5140,[178]336611.6326,[179]336169.1428,[180]337152.8681,[181]336928.3568,[182]337374.7017,[183]336574.88
30,[184]336549.1612,[185]336890.1861,[186]336270.8240,[187]336033.7314,[188]336260.7362,[189]336337.6063,[190]335905.2686,[191]335671.5326,[192]336063
.9825,[193]336254.3945,[194]336390.3271,[195]336058.7223,[196]336123.5871,[197]336272.6905,[198]336581.7609,[199]336125.9311,[200]336175.1478,[201]335
261.2004,[202]335722.4991,[203]335732.0036,[204]336010.6380,[205]336554.9746,[206]336870.3485,[207]337512.5650,[208]337800.7907,[209]337957.8198,[210]
339006.8855,[211]339536.3558,[212]339771.6654,[213]339820.9878,[214]340649.4873,[215]340871.1208,[216]341088.6222,[217]340871.9526,[218]340944.1487,[2
19]341612.6012,[220]342518.8541,[221]342988.1971,[222]342574.7840,[223]343481.4894,[224]343029.3821,[225]343295.2932,[226]343032.9993,[227]343704.6932
,[228]345175.9576,[229]345567.2666,[230]346984.2971,[231]347891.9790,[232]348421.3554,[233]347906.3728,[234]348105.3882,[235]347709.6448,[236]347865.7
097,[237]347051.5113,[238]347476.0560,[239]348607.8464,[240]347950.9243,[241]348175.2049,[242]348260.1216,[243]348118.1121,[244]349105.7627,[245]35034
3.6532,[246]351018.4541,[247]349972.1138,[248]349626.9985,[249]349815.8200,[250]349784.0491,[251]349044.6743,[252]348851.4149,[253]347922.8042,[254]34
7737.7496,[255]347553.6986,[256]347998.6214,[257]348681.4274,[258]348605.3748,[259]347746.3318,[260]347249.1009,[261]347208.6900,[262]346804.7642,[263
]346325.7216,[264]345906.9311,[265]345908.3860,[266]345701.0113,[267]345709.4001,[268]345912.5002,[269]346098.0048,[270]345980.1661,[271]345810.4070,[
272]345554.0991,[273]345337.1543,[274]344923.7055,[275]344460.3920,[276]343342.6230,[277]343576.3771,[278]342718.8707,[279]342988.6333,[280]343045.420
5,[281]342954.1471,[282]343121.6664,[283]343447.0750,[284]343345.1687,[285]343518.5285,[286]343098.9947,[287]342822.1719,[288]342853.3967,[289]343641.
2162,[290]343374.6100,[291]343746.9794,[292]343718.3872,[293]343928.4375,[294]344298.2272,[295]344357.2789,[296]344897.7471,[297]343889.5777,[298]3443
89.0557,[299]345317.8505,[300]344843.8735,[301]345089.1796,[302]345391.7513,[303]344981.9309,[304]345274.1943,[305]345361.9946,[306]344615.1515,[307]3
44191.7641,[308]344244.3699,[309]343919.6349,[310]344199.1177,[311]344405.9163,[312]344450.0979,[313]344439.8224,[314]344141.4730,[315]342825.3627,[31
6]341433.4296,[317]340663.0907,[318]339582.1865,[319]338423.3959,[320]338431.9492,[321]338115.6464,[322]337707.7252,[323]337509.5115,[324]337143.1945,
[325]336863.2449,[326]336823.7532,[327]336944.8010,[328]336631.8671,[329]335992.6150,[330]335818.9447,[331]335230.9186,[332]335293.0504,[333]334905.10
22,[334]335016.8497,[335]334882.2233,[336]335010.3878,[337]334898.4524,[338]334669.4391,[339]334527.0858,[340]334121.5989,[341]333836.9861,[342]334106
.1635,[343]334063.7962,[344]334203.4633,[345]334543.9787,[346]334077.9966,[347]334284.0650,[348]334445.7269,[349]334827.9118,[350]334821.3506,[351]334
479.8770,[352]334176.5657,[353]334025.4542,[354]333939.9035,[355]333898.6704,[356]333624.9149,[357]333237.7507,[358]333661.4850,[359]334098.6600,[360]
334318.0128,[361]334045.3073,[362]333919.0924,[363]333648.6163,[364]334117.8579,[365]334137.6652,[366]334344.9832,[367]334292.8768,[368]334416.0816,[3
69]334236.0430,[370]334155.9937,[371]333734.8777,[372]334073.4287,[373]333972.2325,[374]333610.6319,[375]333627.4234,[376]333967.3869,[377]334455.1315
,[378]334648.7305,[379]334723.9790,[380]334915.8106,[381]334783.0520,[382]334792.9807,[383]334292.3066,[384]334761.0592,[385]334650.0049,[386]334250.9
363,[387]334130.7030,[388]334962.6261,[389]335103.6648,[390]334964.4796,[391]335155.0150,[392]335258.2591,[393]335715.2107,[394]336216.3549,[395]33678
4.9280,[396]336825.6375,[397]336514.6311,[398]336291.0403,[399]335938.5148,[400]335934.1942,[401]336392.6242,[402]335974.0197,[403]336289.9238,[404]33
6379.4946,[405]336555.6353,[406]336369.9217,[407]336264.4100,[408]336306.2972,[409]336062.0189,[410]336218.9131,[411]335872.2278,[412]335754.9736,[413
]335586.0973,[414]335124.5066,[415]335378.1566,[416]335487.5042,[417]335712.7851,[418]335428.0417,[419]335734.1041,[420]336284.5707,[421]336296.1309,[
422]335716.1559,[423]335819.8443,[424]335746.8833,[425]335446.8556,[426]335455.4698,[427]335421.7328,[428]335308.4573,[429]335308.3605,[430]335634.427
1,[431]335941.7238,[432]335805.4835,[433]335864.1890,[434]335795.2289,[435]335790.3390,[436]336183.7092,[437]336053.6280,[438]336412.7182,[439]336779.
1893,[440]336638.0088,[441]336696.3587,[442]336693.5864,[443]336947.3901,[444]337364.4074,[445]337188.6797,[446]336960.3097,[447]336982.3581,[448]3367
40.4896,[449]336800.7335,[450]337456.5018,[451]337628.6795,[452]338075.0179,[453]338217.9506,[454]338563.8328,[455]338449.4376,[456]338244.9696,[457]3
38254.9905,[458]337899.0490,[459]338065.0851,[460]338084.4375,[461]338013.8557,[462]337774.4167,[463]338030.2594,[464]337997.7621,[465]338313.0132,[46
6]338480.3486,[467]338553.1094,[468]338698.8431,[469]338961.8873,[470]339099.5448,[471]339529.5247,[472]339518.9106,[473]339533.8010,[474]339280.8227,
[475]339337.3000,[476]339614.2696,[477]339436.1779,[478]339499.3813,[479]339569.9636,[480]339304.3727,[481]339458.5688,[482]339531.7829,[483]339698.45
70,[484]339156.1393,[485]339477.7685,[486]340238.3424,[487]340379.7815,[488]340655.9210,[489]340516.3203,[490]340570.0327,[491]340506.7411,[492]340278
.8962,[493]340258.7227,[494]340450.1686,[495]339995.1085,[496]340057.2055,[497]340209.0422,[498]339943.5230,[499]339784.5338,[500]339990.5147,[501]339
970.8131,[502]340371.5679,[503]340059.3617,[504]339792.6366,[505]339453.2254,[506]339424.0224,[507]339627.8620,[508]339683.1626,[509]339688.5786,[510]
339971.3743,[511]340134.1403,[512]340558.5657,[513]340734.9633,[514]341007.3962,[515]341043.8739,[516]341339.0372,[517]341604.4826,[518]341228.6644,[5
19]340909.3084,[520]340917.5889,[521]340871.2405,[522]340629.4603,[523]340600.1478,[524]340494.6514,[525]339985.5894,[526]339798.1336,[527]339423.1168
,[528]339574.7999,[529]338999.3788,[530]338866.6454,[531]339064.2290,[532]338175.7611,[533]338193.8181,[534]338591.1751,[535]338794.1938,[536]338815.3
925,[537]338854.7276,[538]338997.8122,[539]339560.6960,[540]339563.1839,[541]339606.7486,[542]339558.3348,[543]339493.1708,[544]339729.4373,[545]34020
8.8763,[546]340231.7345,[547]340359.0196,[548]340906.6126,[549]341063.1162,[550]341158.9496,[551]341645.1513,[552]341690.2990,[553]341566.8309,[554]34
1969.4067,[555]341819.3313,[556]341737.7033,[557]341893.9760,[558]341486.6305,[559]341186.3327,[560]340936.6909,[561]340925.0560,
llama_print_timings:        load time =    2238.45 ms
llama_print_timings:      sample time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings: prompt eval time = 2214337.48 ms / 287232 tokens (    7.71 ms per token,   129.71 tokens per second)
llama_print_timings:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings:       total time = 2264778.41 ms / 287233 tokens

Final estimate: PPL = 340925.0560 +/- 2519.12041
```

</details>


<details>

<summary>llama-server response to chat client looks wrong</summary>

I tried various combinations of server configs and all yielded same wrong looking responses in client.

#### Start Server
```bash
#### First attempt
numactl -N 0 -m 0 \
./build/bin/llama-server \
    --model /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-CPU-IQ4_K_R4.gguf \
    --alias ubergarm/DeepSeek-V3-0324-CPU-IQ4_K_R4 \
    --ctx-size 8192 \
    -ctk q8_0 \
    -mla 3 -fa \ # also tried -mla 2
    -amb 2048 \
    -fmoe \
    --temp 0.3 \
    --parallel 1 \
    --threads 128 \
    --numa numactl \
    --host 127.0.0.1 \
    --port 8080

#### Second attempt
numactl -N 0 -m 0 \
./build/bin/llama-server \
    --model /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-CPU-IQ4_K_R4.gguf \
    --alias ubergarm/DeepSeek-V3-0324-CPU-IQ4_K_R4 \
    --ctx-size 8192 \
    --parallel 1 \
    --threads 128 \
    --numa numactl \
    --host 127.0.0.1 \
    --port 8080
```

#### Start Client
```bash
$ python dchat.py
Input prompt then press Ctrl+D twice (or once on empty line) to send.
Ctrl+C to cancel response or twice to exit.

>>> User:

Count from 1 to 10 in French.

>>> Assistant:

AlrightAlrightAlrightAlright
>>> User:

^C^C

Exiting...
```

</details>

---

#### 💬 Conversation

👤 **saood06** commented the **2025-03-30** at **20:23:35**:<br>

My working mix:
llama_model_loader: - type f32: 361 tensors
llama_model_loader: - type q8_0: 246 tensors
llama_model_loader: - type iq4_k_r4: 357 tensors
llama_model_loader: - type iq5_k_r4: 61 tensors

Full quant log below:

<details>

<summary>Log</summary>

```
./bin/llama-quantize --allow-requantize  --imatrix /mnt/sda/deepseek-ai_DeepSeek-V3-0324.imatrix --token-embedding-type q8_0 --output-tensor-type q8_0 /mnt/sda/deepseek-ai_DeepSeek-V3-0324-Q8_0/deepseek-ai_DeepSeek-V3-0324-Q8_0-00001-of-00020.gguf /mnt/sda/DeepSeek-V3-0324-IQ4_K_R4.gguf IQ4_K_R4 48
load_imatrix: imatrix dataset='/workspace/calibration_datav3.txt'
load_imatrix: loaded 720 importance matrix entries from /mnt/sda/deepseek-ai_DeepSeek-V3-0324.imatrix computed on 124 chunks
prepare_imatrix: have 720 importance matrix entries
main: build = 3617 (f31aca2d)
main: built with gcc (Clear Linux OS for Intel Architecture) 14.2.1 20241210 releases/gcc-14.2.0-551-g21a09f0507 for x86_64-generic-linux
main: quantizing '/mnt/sda/deepseek-ai_DeepSeek-V3-0324-Q8_0/deepseek-ai_DeepSeek-V3-0324-Q8_0-00001-of-00020.gguf' to '/mnt/sda/DeepSeek-V3-0324-IQ4_K_R4.gguf' as IQ4_K_R4 using 48 threads
llama_model_loader: additional 19 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 47 key-value pairs and 1025 tensors from /mnt/sda/deepseek-ai_DeepSeek-V3-0324-Q8_0/deepseek-ai_DeepSeek-V3-0324-Q8_0-00001-of-00020.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = DeepSeek V3 0324 Bf16
llama_model_loader: - kv   3:                         general.size_label str              = 256x20B
llama_model_loader: - kv   4:                            general.license str              = mit
llama_model_loader: - kv   5:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv   6:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv   7:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv   8:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv   9:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv  10:          deepseek2.attention.head_count_kv u32              = 128
llama_model_loader: - kv  11:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  12: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  13:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  14:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  15:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  16:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  17:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  18:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  19:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  20:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  21:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  22:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  23:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  24:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  25:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  26:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  27:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  28:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  29: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  30: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  31:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  32:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  33:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<▒...
llama_model_loader: - kv  34:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  35:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
llama_model_loader: - kv  36:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  37:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  38:            tokenizer.ggml.padding_token_id u32              = 1
llama_model_loader: - kv  39:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  40:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  41:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  42:               general.quantization_version u32              = 2
llama_model_loader: - kv  43:                          general.file_type u32              = 7
llama_model_loader: - kv  44:                                   split.no u16              = 0
llama_model_loader: - kv  45:                        split.tensors.count i32              = 1025
llama_model_loader: - kv  46:                                split.count u16              = 20
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  664 tensors
================================ Have weights data with 720 entries
[   1/1025]                        output.weight - [ 7168, 129280,     1,     1], type =   q8_0, size =  938.984 MB
[   2/1025]                   output_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[   3/1025]                    token_embd.weight - [ 7168, 129280,     1,     1], type =   q8_0, size =  938.984 MB
[   4/1025]           blk.0.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[   5/1025]          blk.0.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[   6/1025]               blk.0.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[   7/1025]               blk.0.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[   8/1025]             blk.0.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[   9/1025]                blk.0.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[  10/1025]           blk.0.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  11/1025]                blk.0.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[  12/1025]                blk.0.ffn_down.weight - [18432,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =   133.88 MiB ->    70.88 MiB
[  13/1025]                blk.0.ffn_gate.weight - [ 7168, 18432,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =   133.88 MiB ->    70.88 MiB
[  14/1025]                blk.0.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  15/1025]                  blk.0.ffn_up.weight - [ 7168, 18432,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =   133.88 MiB ->    70.88 MiB
[  16/1025]           blk.1.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[  17/1025]          blk.1.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  18/1025]               blk.1.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[  19/1025]               blk.1.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  20/1025]             blk.1.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[  21/1025]                blk.1.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[  22/1025]           blk.1.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  23/1025]                blk.1.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[  24/1025]                blk.1.ffn_down.weight - [18432,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =   133.88 MiB ->    70.88 MiB
[  25/1025]                blk.1.ffn_gate.weight - [ 7168, 18432,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =   133.88 MiB ->    70.88 MiB
[  26/1025]                blk.1.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  27/1025]                  blk.1.ffn_up.weight - [ 7168, 18432,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =   133.88 MiB ->    70.88 MiB
[  28/1025]           blk.2.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[  29/1025]          blk.2.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  30/1025]               blk.2.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[  31/1025]               blk.2.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  32/1025]             blk.2.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[  33/1025]                blk.2.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[  34/1025]           blk.2.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  35/1025]                blk.2.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[  36/1025]                blk.2.ffn_down.weight - [18432,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =   133.88 MiB ->    70.88 MiB
[  37/1025]                blk.2.ffn_gate.weight - [ 7168, 18432,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =   133.88 MiB ->    70.88 MiB
[  38/1025]                blk.2.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  39/1025]                  blk.2.ffn_up.weight - [ 7168, 18432,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =   133.88 MiB ->    70.88 MiB
[  40/1025]           blk.3.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[  41/1025]          blk.3.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  42/1025]               blk.3.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[  43/1025]               blk.3.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  44/1025]             blk.3.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[  45/1025]                blk.3.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[  46/1025]           blk.3.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  47/1025]                blk.3.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[  48/1025]               blk.3.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[  49/1025]           blk.3.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[  50/1025]          blk.3.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[  51/1025]           blk.3.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[  52/1025]            blk.3.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[  53/1025]          blk.3.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[  54/1025]                blk.3.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  55/1025]             blk.3.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[  56/1025]            blk.3.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[  57/1025]           blk.4.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[  58/1025]          blk.4.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  59/1025]               blk.4.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[  60/1025]               blk.4.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  61/1025]             blk.4.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[  62/1025]                blk.4.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[  63/1025]           blk.4.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  64/1025]                blk.4.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[  65/1025]               blk.4.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[  66/1025]           blk.4.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[  67/1025]          blk.4.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[  68/1025]           blk.4.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[  69/1025]            blk.4.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[  70/1025]          blk.4.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[  71/1025]                blk.4.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  72/1025]             blk.4.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[  73/1025]            blk.4.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[  74/1025]           blk.5.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[  75/1025]          blk.5.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  76/1025]               blk.5.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[  77/1025]               blk.5.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  78/1025]             blk.5.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[  79/1025]                blk.5.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[  80/1025]           blk.5.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  81/1025]                blk.5.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[  82/1025]               blk.5.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[  83/1025]           blk.5.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[  84/1025]          blk.5.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[  85/1025]           blk.5.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[  86/1025]            blk.5.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[  87/1025]          blk.5.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[  88/1025]                blk.5.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  89/1025]             blk.5.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[  90/1025]            blk.5.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[  91/1025]           blk.6.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[  92/1025]          blk.6.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  93/1025]               blk.6.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[  94/1025]               blk.6.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  95/1025]             blk.6.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[  96/1025]                blk.6.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[  97/1025]           blk.6.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  98/1025]                blk.6.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[  99/1025]               blk.6.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 100/1025]           blk.6.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 101/1025]          blk.6.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 102/1025]           blk.6.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 103/1025]            blk.6.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 104/1025]          blk.6.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 105/1025]                blk.6.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 106/1025]             blk.6.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 107/1025]            blk.6.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 108/1025]           blk.7.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 109/1025]          blk.7.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 110/1025]               blk.7.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 111/1025]               blk.7.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 112/1025]             blk.7.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 113/1025]                blk.7.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 114/1025]           blk.7.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 115/1025]                blk.7.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 116/1025]               blk.7.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 117/1025]           blk.7.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 118/1025]          blk.7.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 119/1025]           blk.7.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 120/1025]            blk.7.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 121/1025]          blk.7.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 122/1025]                blk.7.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 123/1025]             blk.7.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 124/1025]            blk.7.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 125/1025]           blk.8.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 126/1025]          blk.8.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 127/1025]               blk.8.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 128/1025]               blk.8.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 129/1025]             blk.8.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 130/1025]                blk.8.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 131/1025]           blk.8.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 132/1025]                blk.8.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 133/1025]               blk.8.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 134/1025]           blk.8.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 135/1025]          blk.8.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 136/1025]           blk.8.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 137/1025]            blk.8.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 138/1025]          blk.8.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 139/1025]                blk.8.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 140/1025]             blk.8.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 141/1025]            blk.8.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 142/1025]           blk.9.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 143/1025]          blk.9.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 144/1025]               blk.9.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 145/1025]               blk.9.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 146/1025]             blk.9.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 147/1025]                blk.9.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 148/1025]           blk.9.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 149/1025]                blk.9.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 150/1025]               blk.9.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 151/1025]           blk.9.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 152/1025]          blk.9.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 153/1025]           blk.9.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 154/1025]            blk.9.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 155/1025]          blk.9.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 156/1025]                blk.9.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 157/1025]             blk.9.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 158/1025]            blk.9.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 159/1025]          blk.10.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 160/1025]         blk.10.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 161/1025]              blk.10.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 162/1025]              blk.10.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 163/1025]            blk.10.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 164/1025]               blk.10.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 165/1025]          blk.10.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 166/1025]               blk.10.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 167/1025]              blk.10.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 168/1025]          blk.10.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 169/1025]         blk.10.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 170/1025]          blk.10.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 171/1025]           blk.10.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 172/1025]         blk.10.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 173/1025]               blk.10.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 174/1025]            blk.10.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 175/1025]           blk.10.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 176/1025]          blk.11.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 177/1025]         blk.11.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 178/1025]              blk.11.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 179/1025]              blk.11.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 180/1025]            blk.11.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 181/1025]               blk.11.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 182/1025]          blk.11.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 183/1025]               blk.11.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 184/1025]              blk.11.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 185/1025]          blk.11.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 186/1025]         blk.11.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 187/1025]          blk.11.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 188/1025]           blk.11.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 189/1025]         blk.11.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 190/1025]               blk.11.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 191/1025]            blk.11.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 192/1025]           blk.11.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 193/1025]          blk.12.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 194/1025]         blk.12.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 195/1025]              blk.12.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 196/1025]              blk.12.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 197/1025]            blk.12.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 198/1025]               blk.12.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 199/1025]          blk.12.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 200/1025]               blk.12.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 201/1025]              blk.12.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 202/1025]          blk.12.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 203/1025]         blk.12.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 204/1025]          blk.12.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 205/1025]           blk.12.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 206/1025]         blk.12.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 207/1025]               blk.12.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 208/1025]            blk.12.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 209/1025]           blk.12.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 210/1025]          blk.13.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 211/1025]         blk.13.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 212/1025]              blk.13.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 213/1025]              blk.13.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 214/1025]            blk.13.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 215/1025]               blk.13.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 216/1025]          blk.13.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 217/1025]               blk.13.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 218/1025]              blk.13.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 219/1025]          blk.13.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 220/1025]         blk.13.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 221/1025]          blk.13.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 222/1025]           blk.13.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 223/1025]         blk.13.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 224/1025]               blk.13.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 225/1025]            blk.13.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 226/1025]           blk.13.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 227/1025]          blk.14.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 228/1025]         blk.14.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 229/1025]              blk.14.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 230/1025]              blk.14.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 231/1025]            blk.14.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 232/1025]               blk.14.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 233/1025]          blk.14.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 234/1025]               blk.14.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 235/1025]              blk.14.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 236/1025]          blk.14.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 237/1025]         blk.14.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 238/1025]          blk.14.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 239/1025]           blk.14.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 240/1025]         blk.14.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 241/1025]               blk.14.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 242/1025]            blk.14.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 243/1025]           blk.14.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 244/1025]          blk.15.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 245/1025]         blk.15.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 246/1025]              blk.15.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 247/1025]              blk.15.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 248/1025]            blk.15.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 249/1025]               blk.15.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 250/1025]          blk.15.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 251/1025]               blk.15.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 252/1025]              blk.15.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 253/1025]          blk.15.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 254/1025]         blk.15.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 255/1025]          blk.15.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 256/1025]           blk.15.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 257/1025]         blk.15.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 258/1025]               blk.15.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 259/1025]            blk.15.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 260/1025]           blk.15.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 261/1025]          blk.16.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 262/1025]         blk.16.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 263/1025]              blk.16.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 264/1025]              blk.16.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 265/1025]            blk.16.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 266/1025]               blk.16.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 267/1025]          blk.16.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 268/1025]               blk.16.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 269/1025]              blk.16.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 270/1025]          blk.16.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 271/1025]         blk.16.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 272/1025]          blk.16.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 273/1025]           blk.16.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 274/1025]         blk.16.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 275/1025]               blk.16.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 276/1025]            blk.16.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 277/1025]           blk.16.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 278/1025]          blk.17.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 279/1025]         blk.17.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 280/1025]              blk.17.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 281/1025]              blk.17.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 282/1025]            blk.17.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 283/1025]               blk.17.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 284/1025]          blk.17.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 285/1025]               blk.17.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 286/1025]              blk.17.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 287/1025]          blk.17.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 288/1025]         blk.17.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 289/1025]          blk.17.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 290/1025]           blk.17.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 291/1025]         blk.17.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 292/1025]               blk.17.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 293/1025]            blk.17.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 294/1025]           blk.17.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 295/1025]          blk.18.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 296/1025]         blk.18.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 297/1025]              blk.18.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 298/1025]              blk.18.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 299/1025]            blk.18.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 300/1025]               blk.18.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 301/1025]          blk.18.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 302/1025]               blk.18.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 303/1025]              blk.18.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 304/1025]          blk.18.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 305/1025]         blk.18.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 306/1025]          blk.18.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 307/1025]           blk.18.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 308/1025]         blk.18.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 309/1025]               blk.18.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 310/1025]            blk.18.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 311/1025]           blk.18.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 312/1025]          blk.19.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 313/1025]         blk.19.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 314/1025]              blk.19.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 315/1025]              blk.19.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 316/1025]            blk.19.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 317/1025]               blk.19.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 318/1025]          blk.19.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 319/1025]               blk.19.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 320/1025]              blk.19.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 321/1025]          blk.19.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 322/1025]         blk.19.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 323/1025]          blk.19.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 324/1025]           blk.19.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 325/1025]         blk.19.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 326/1025]               blk.19.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 327/1025]            blk.19.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 328/1025]           blk.19.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 329/1025]          blk.20.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 330/1025]         blk.20.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 331/1025]              blk.20.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 332/1025]              blk.20.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 333/1025]            blk.20.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 334/1025]               blk.20.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 335/1025]          blk.20.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 336/1025]               blk.20.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 337/1025]              blk.20.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 338/1025]          blk.20.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 339/1025]         blk.20.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 340/1025]          blk.20.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 341/1025]           blk.20.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 342/1025]         blk.20.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 343/1025]               blk.20.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 344/1025]            blk.20.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 345/1025]           blk.20.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 346/1025]          blk.21.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 347/1025]         blk.21.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 348/1025]              blk.21.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 349/1025]              blk.21.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 350/1025]            blk.21.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 351/1025]               blk.21.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 352/1025]          blk.21.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 353/1025]               blk.21.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 354/1025]              blk.21.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 355/1025]          blk.21.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 356/1025]         blk.21.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 357/1025]          blk.21.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 358/1025]           blk.21.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 359/1025]         blk.21.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 360/1025]               blk.21.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 361/1025]            blk.21.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 362/1025]           blk.21.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 363/1025]          blk.22.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 364/1025]         blk.22.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 365/1025]              blk.22.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 366/1025]              blk.22.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 367/1025]            blk.22.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 368/1025]               blk.22.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 369/1025]          blk.22.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 370/1025]               blk.22.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 371/1025]              blk.22.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 372/1025]          blk.22.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 373/1025]         blk.22.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 374/1025]          blk.22.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 375/1025]           blk.22.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 376/1025]         blk.22.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 377/1025]               blk.22.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 378/1025]            blk.22.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 379/1025]           blk.22.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 380/1025]          blk.23.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 381/1025]         blk.23.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 382/1025]              blk.23.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 383/1025]              blk.23.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 384/1025]            blk.23.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 385/1025]               blk.23.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 386/1025]          blk.23.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 387/1025]               blk.23.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 388/1025]              blk.23.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 389/1025]          blk.23.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 390/1025]         blk.23.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 391/1025]          blk.23.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 392/1025]           blk.23.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 393/1025]         blk.23.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 394/1025]               blk.23.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 395/1025]            blk.23.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 396/1025]           blk.23.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 397/1025]          blk.24.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 398/1025]         blk.24.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 399/1025]              blk.24.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 400/1025]              blk.24.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 401/1025]            blk.24.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 402/1025]               blk.24.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 403/1025]          blk.24.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 404/1025]               blk.24.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 405/1025]              blk.24.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 406/1025]          blk.24.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 407/1025]         blk.24.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 408/1025]          blk.24.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 409/1025]           blk.24.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 410/1025]         blk.24.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 411/1025]               blk.24.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 412/1025]            blk.24.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 413/1025]           blk.24.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 414/1025]          blk.25.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 415/1025]         blk.25.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 416/1025]              blk.25.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 417/1025]              blk.25.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 418/1025]            blk.25.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 419/1025]               blk.25.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 420/1025]          blk.25.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 421/1025]               blk.25.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 422/1025]              blk.25.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 423/1025]          blk.25.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 424/1025]         blk.25.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 425/1025]          blk.25.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 426/1025]           blk.25.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 427/1025]         blk.25.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 428/1025]               blk.25.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 429/1025]            blk.25.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 430/1025]           blk.25.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 431/1025]          blk.26.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 432/1025]         blk.26.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 433/1025]              blk.26.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 434/1025]              blk.26.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 435/1025]            blk.26.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 436/1025]               blk.26.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 437/1025]          blk.26.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 438/1025]               blk.26.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 439/1025]              blk.26.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 440/1025]          blk.26.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 441/1025]         blk.26.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 442/1025]          blk.26.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 443/1025]           blk.26.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 444/1025]         blk.26.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 445/1025]               blk.26.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 446/1025]            blk.26.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 447/1025]           blk.26.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 448/1025]          blk.27.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 449/1025]         blk.27.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 450/1025]              blk.27.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 451/1025]              blk.27.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 452/1025]            blk.27.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 453/1025]               blk.27.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 454/1025]          blk.27.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 455/1025]               blk.27.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 456/1025]              blk.27.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 457/1025]          blk.27.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 458/1025]         blk.27.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 459/1025]          blk.27.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 460/1025]           blk.27.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 461/1025]         blk.27.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 462/1025]               blk.27.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 463/1025]            blk.27.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 464/1025]           blk.27.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 465/1025]          blk.28.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 466/1025]         blk.28.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 467/1025]              blk.28.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 468/1025]              blk.28.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 469/1025]            blk.28.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 470/1025]               blk.28.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 471/1025]          blk.28.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 472/1025]               blk.28.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 473/1025]              blk.28.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 474/1025]          blk.28.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 475/1025]         blk.28.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 476/1025]          blk.28.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 477/1025]           blk.28.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 478/1025]         blk.28.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 479/1025]               blk.28.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 480/1025]            blk.28.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 481/1025]           blk.28.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 482/1025]          blk.29.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 483/1025]         blk.29.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 484/1025]              blk.29.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 485/1025]              blk.29.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 486/1025]            blk.29.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 487/1025]               blk.29.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 488/1025]          blk.29.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 489/1025]               blk.29.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 490/1025]              blk.29.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 491/1025]          blk.29.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 492/1025]         blk.29.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 493/1025]          blk.29.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 494/1025]           blk.29.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 495/1025]         blk.29.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 496/1025]               blk.29.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 497/1025]            blk.29.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 498/1025]           blk.29.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 499/1025]          blk.30.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 500/1025]         blk.30.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 501/1025]              blk.30.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 502/1025]              blk.30.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 503/1025]            blk.30.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 504/1025]               blk.30.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 505/1025]          blk.30.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 506/1025]               blk.30.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 507/1025]              blk.30.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 508/1025]          blk.30.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 509/1025]         blk.30.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 510/1025]          blk.30.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 511/1025]           blk.30.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 512/1025]         blk.30.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 513/1025]               blk.30.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 514/1025]            blk.30.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 515/1025]           blk.30.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 516/1025]          blk.31.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 517/1025]         blk.31.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 518/1025]              blk.31.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 519/1025]              blk.31.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 520/1025]            blk.31.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 521/1025]               blk.31.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 522/1025]          blk.31.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 523/1025]               blk.31.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 524/1025]              blk.31.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 525/1025]          blk.31.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 526/1025]         blk.31.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 527/1025]          blk.31.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 528/1025]           blk.31.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 529/1025]         blk.31.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 530/1025]               blk.31.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 531/1025]            blk.31.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 532/1025]           blk.31.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 533/1025]          blk.32.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 534/1025]         blk.32.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 535/1025]              blk.32.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 536/1025]              blk.32.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 537/1025]            blk.32.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 538/1025]               blk.32.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 539/1025]          blk.32.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 540/1025]               blk.32.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 541/1025]              blk.32.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 542/1025]          blk.32.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 543/1025]         blk.32.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 544/1025]          blk.32.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 545/1025]           blk.32.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 546/1025]         blk.32.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 547/1025]               blk.32.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 548/1025]            blk.32.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 549/1025]           blk.32.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 550/1025]          blk.33.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 551/1025]         blk.33.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 552/1025]              blk.33.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 553/1025]              blk.33.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 554/1025]            blk.33.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 555/1025]               blk.33.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 556/1025]          blk.33.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 557/1025]               blk.33.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 558/1025]              blk.33.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 559/1025]          blk.33.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 560/1025]         blk.33.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 561/1025]          blk.33.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 562/1025]           blk.33.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 563/1025]         blk.33.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 564/1025]               blk.33.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 565/1025]            blk.33.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 566/1025]           blk.33.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 567/1025]          blk.34.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 568/1025]         blk.34.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 569/1025]              blk.34.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 570/1025]              blk.34.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 571/1025]            blk.34.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 572/1025]               blk.34.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 573/1025]          blk.34.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 574/1025]               blk.34.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 575/1025]              blk.34.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 576/1025]          blk.34.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 577/1025]         blk.34.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 578/1025]          blk.34.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 579/1025]           blk.34.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 580/1025]         blk.34.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 581/1025]               blk.34.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 582/1025]            blk.34.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 583/1025]           blk.34.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 584/1025]          blk.35.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 585/1025]         blk.35.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 586/1025]              blk.35.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 587/1025]              blk.35.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 588/1025]            blk.35.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 589/1025]               blk.35.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 590/1025]          blk.35.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 591/1025]               blk.35.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 592/1025]              blk.35.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 593/1025]          blk.35.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 594/1025]         blk.35.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 595/1025]          blk.35.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 596/1025]           blk.35.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 597/1025]         blk.35.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 598/1025]               blk.35.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 599/1025]            blk.35.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 600/1025]           blk.35.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 601/1025]          blk.36.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 602/1025]         blk.36.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 603/1025]              blk.36.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 604/1025]              blk.36.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 605/1025]            blk.36.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 606/1025]               blk.36.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 607/1025]          blk.36.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 608/1025]               blk.36.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 609/1025]              blk.36.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 610/1025]          blk.36.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 611/1025]         blk.36.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 612/1025]          blk.36.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 613/1025]           blk.36.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 614/1025]         blk.36.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 615/1025]               blk.36.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 616/1025]            blk.36.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 617/1025]           blk.36.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 618/1025]          blk.37.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 619/1025]         blk.37.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 620/1025]              blk.37.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 621/1025]              blk.37.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 622/1025]            blk.37.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 623/1025]               blk.37.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 624/1025]          blk.37.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 625/1025]               blk.37.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 626/1025]              blk.37.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 627/1025]          blk.37.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 628/1025]         blk.37.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 629/1025]          blk.37.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 630/1025]           blk.37.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 631/1025]         blk.37.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 632/1025]               blk.37.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 633/1025]            blk.37.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 634/1025]           blk.37.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 635/1025]          blk.38.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 636/1025]         blk.38.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 637/1025]              blk.38.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 638/1025]              blk.38.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 639/1025]            blk.38.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 640/1025]               blk.38.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 641/1025]          blk.38.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 642/1025]               blk.38.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 643/1025]              blk.38.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 644/1025]          blk.38.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 645/1025]         blk.38.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 646/1025]          blk.38.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 647/1025]           blk.38.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 648/1025]         blk.38.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 649/1025]               blk.38.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 650/1025]            blk.38.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 651/1025]           blk.38.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 652/1025]          blk.39.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 653/1025]         blk.39.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 654/1025]              blk.39.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 655/1025]              blk.39.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 656/1025]            blk.39.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 657/1025]               blk.39.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 658/1025]          blk.39.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 659/1025]               blk.39.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 660/1025]              blk.39.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 661/1025]          blk.39.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 662/1025]         blk.39.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 663/1025]          blk.39.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 664/1025]           blk.39.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 665/1025]         blk.39.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 666/1025]               blk.39.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 667/1025]            blk.39.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 668/1025]           blk.39.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 669/1025]          blk.40.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 670/1025]         blk.40.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 671/1025]              blk.40.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 672/1025]              blk.40.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 673/1025]            blk.40.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 674/1025]               blk.40.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 675/1025]          blk.40.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 676/1025]               blk.40.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 677/1025]              blk.40.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 678/1025]          blk.40.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 679/1025]         blk.40.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 680/1025]          blk.40.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 681/1025]           blk.40.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 682/1025]         blk.40.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 683/1025]               blk.40.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 684/1025]            blk.40.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 685/1025]           blk.40.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 686/1025]          blk.41.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 687/1025]         blk.41.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 688/1025]              blk.41.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 689/1025]              blk.41.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 690/1025]            blk.41.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 691/1025]               blk.41.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 692/1025]          blk.41.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 693/1025]               blk.41.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 694/1025]              blk.41.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 695/1025]          blk.41.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 696/1025]         blk.41.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 697/1025]          blk.41.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 698/1025]           blk.41.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 699/1025]         blk.41.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 700/1025]               blk.41.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 701/1025]            blk.41.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 702/1025]           blk.41.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 703/1025]          blk.42.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 704/1025]         blk.42.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 705/1025]              blk.42.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 706/1025]              blk.42.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 707/1025]            blk.42.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 708/1025]               blk.42.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 709/1025]          blk.42.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 710/1025]               blk.42.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 711/1025]              blk.42.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 712/1025]          blk.42.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 713/1025]         blk.42.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 714/1025]          blk.42.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 715/1025]           blk.42.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 716/1025]         blk.42.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 717/1025]               blk.42.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 718/1025]            blk.42.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 719/1025]           blk.42.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 720/1025]          blk.43.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 721/1025]         blk.43.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 722/1025]              blk.43.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 723/1025]              blk.43.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 724/1025]            blk.43.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 725/1025]               blk.43.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 726/1025]          blk.43.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 727/1025]               blk.43.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 728/1025]              blk.43.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 729/1025]          blk.43.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 730/1025]         blk.43.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 731/1025]          blk.43.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 732/1025]           blk.43.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 733/1025]         blk.43.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 734/1025]               blk.43.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 735/1025]            blk.43.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 736/1025]           blk.43.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 737/1025]          blk.44.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 738/1025]         blk.44.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 739/1025]              blk.44.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 740/1025]              blk.44.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 741/1025]            blk.44.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 742/1025]               blk.44.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 743/1025]          blk.44.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 744/1025]               blk.44.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 745/1025]              blk.44.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 746/1025]          blk.44.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 747/1025]         blk.44.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 748/1025]          blk.44.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 749/1025]           blk.44.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 750/1025]         blk.44.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 751/1025]               blk.44.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 752/1025]            blk.44.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 753/1025]           blk.44.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 754/1025]          blk.45.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 755/1025]         blk.45.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 756/1025]              blk.45.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 757/1025]              blk.45.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 758/1025]            blk.45.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 759/1025]               blk.45.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 760/1025]          blk.45.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 761/1025]               blk.45.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 762/1025]              blk.45.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 763/1025]          blk.45.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 764/1025]         blk.45.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 765/1025]          blk.45.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 766/1025]           blk.45.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 767/1025]         blk.45.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 768/1025]               blk.45.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 769/1025]            blk.45.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 770/1025]           blk.45.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 771/1025]          blk.46.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 772/1025]         blk.46.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 773/1025]              blk.46.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 774/1025]              blk.46.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 775/1025]            blk.46.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 776/1025]               blk.46.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 777/1025]          blk.46.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 778/1025]               blk.46.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 779/1025]              blk.46.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 780/1025]          blk.46.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 781/1025]         blk.46.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 782/1025]          blk.46.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 783/1025]           blk.46.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 784/1025]         blk.46.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 785/1025]               blk.46.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 786/1025]            blk.46.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 787/1025]           blk.46.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 788/1025]          blk.47.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 789/1025]         blk.47.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 790/1025]              blk.47.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 791/1025]              blk.47.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 792/1025]            blk.47.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 793/1025]               blk.47.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 794/1025]          blk.47.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 795/1025]               blk.47.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 796/1025]              blk.47.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 797/1025]          blk.47.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 798/1025]         blk.47.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 799/1025]          blk.47.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 800/1025]           blk.47.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 801/1025]         blk.47.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 802/1025]               blk.47.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 803/1025]            blk.47.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 804/1025]           blk.47.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 805/1025]          blk.48.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 806/1025]         blk.48.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 807/1025]              blk.48.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 808/1025]              blk.48.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 809/1025]            blk.48.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 810/1025]               blk.48.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 811/1025]          blk.48.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 812/1025]               blk.48.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 813/1025]              blk.48.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 814/1025]          blk.48.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 815/1025]         blk.48.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 816/1025]          blk.48.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 817/1025]           blk.48.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 818/1025]         blk.48.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 819/1025]               blk.48.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 820/1025]            blk.48.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 821/1025]           blk.48.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 822/1025]          blk.49.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 823/1025]         blk.49.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 824/1025]              blk.49.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 825/1025]              blk.49.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 826/1025]            blk.49.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 827/1025]               blk.49.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 828/1025]          blk.49.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 829/1025]               blk.49.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 830/1025]              blk.49.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 831/1025]          blk.49.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 832/1025]         blk.49.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 833/1025]          blk.49.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 834/1025]           blk.49.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 835/1025]         blk.49.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 836/1025]               blk.49.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 837/1025]            blk.49.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 838/1025]           blk.49.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 839/1025]          blk.50.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 840/1025]         blk.50.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 841/1025]              blk.50.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 842/1025]              blk.50.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 843/1025]            blk.50.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 844/1025]               blk.50.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 845/1025]          blk.50.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 846/1025]               blk.50.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 847/1025]              blk.50.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 848/1025]          blk.50.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 849/1025]         blk.50.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 850/1025]          blk.50.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 851/1025]           blk.50.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 852/1025]         blk.50.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 853/1025]               blk.50.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 854/1025]            blk.50.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 855/1025]           blk.50.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 856/1025]          blk.51.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 857/1025]         blk.51.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 858/1025]              blk.51.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 859/1025]              blk.51.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 860/1025]            blk.51.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 861/1025]               blk.51.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 862/1025]          blk.51.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 863/1025]               blk.51.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 864/1025]              blk.51.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 865/1025]          blk.51.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 866/1025]         blk.51.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 867/1025]          blk.51.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 868/1025]           blk.51.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 869/1025]         blk.51.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 870/1025]               blk.51.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 871/1025]            blk.51.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 872/1025]           blk.51.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 873/1025]          blk.52.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 874/1025]         blk.52.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 875/1025]              blk.52.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 876/1025]              blk.52.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 877/1025]            blk.52.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 878/1025]               blk.52.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 879/1025]          blk.52.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 880/1025]               blk.52.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 881/1025]              blk.52.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 882/1025]          blk.52.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 883/1025]         blk.52.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 884/1025]          blk.52.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 885/1025]           blk.52.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 886/1025]         blk.52.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 887/1025]               blk.52.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 888/1025]            blk.52.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 889/1025]           blk.52.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 890/1025]          blk.53.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 891/1025]         blk.53.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 892/1025]              blk.53.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 893/1025]              blk.53.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 894/1025]            blk.53.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 895/1025]               blk.53.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 896/1025]          blk.53.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 897/1025]               blk.53.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 898/1025]              blk.53.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 899/1025]          blk.53.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 900/1025]         blk.53.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 901/1025]          blk.53.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 902/1025]           blk.53.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 903/1025]         blk.53.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 904/1025]               blk.53.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 905/1025]            blk.53.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 906/1025]           blk.53.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 907/1025]          blk.54.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 908/1025]         blk.54.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 909/1025]              blk.54.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 910/1025]              blk.54.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 911/1025]            blk.54.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 912/1025]               blk.54.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 913/1025]          blk.54.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 914/1025]               blk.54.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 915/1025]              blk.54.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 916/1025]          blk.54.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 917/1025]         blk.54.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 918/1025]          blk.54.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 919/1025]           blk.54.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 920/1025]         blk.54.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 921/1025]               blk.54.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 922/1025]            blk.54.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 923/1025]           blk.54.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 924/1025]          blk.55.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 925/1025]         blk.55.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 926/1025]              blk.55.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 927/1025]              blk.55.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 928/1025]            blk.55.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 929/1025]               blk.55.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 930/1025]          blk.55.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 931/1025]               blk.55.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 932/1025]              blk.55.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 933/1025]          blk.55.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 934/1025]         blk.55.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 935/1025]          blk.55.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 936/1025]           blk.55.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 937/1025]         blk.55.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 938/1025]               blk.55.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 939/1025]            blk.55.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 940/1025]           blk.55.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 941/1025]          blk.56.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 942/1025]         blk.56.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 943/1025]              blk.56.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 944/1025]              blk.56.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 945/1025]            blk.56.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 946/1025]               blk.56.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 947/1025]          blk.56.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 948/1025]               blk.56.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 949/1025]              blk.56.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 950/1025]          blk.56.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 951/1025]         blk.56.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 952/1025]          blk.56.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 953/1025]           blk.56.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 954/1025]         blk.56.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 955/1025]               blk.56.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 956/1025]            blk.56.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 957/1025]           blk.56.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 958/1025]          blk.57.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 959/1025]         blk.57.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 960/1025]              blk.57.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 961/1025]              blk.57.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 962/1025]            blk.57.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 963/1025]               blk.57.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 964/1025]          blk.57.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 965/1025]               blk.57.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 966/1025]              blk.57.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 967/1025]          blk.57.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 968/1025]         blk.57.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 969/1025]          blk.57.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 970/1025]           blk.57.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 971/1025]         blk.57.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 972/1025]               blk.57.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 973/1025]            blk.57.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 974/1025]           blk.57.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 975/1025]          blk.58.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 976/1025]         blk.58.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 977/1025]              blk.58.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 978/1025]              blk.58.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 979/1025]            blk.58.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 980/1025]               blk.58.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 981/1025]          blk.58.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 982/1025]               blk.58.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 983/1025]              blk.58.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 984/1025]          blk.58.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 985/1025]         blk.58.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 986/1025]          blk.58.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 987/1025]           blk.58.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 988/1025]         blk.58.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 989/1025]               blk.58.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 990/1025]            blk.58.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 991/1025]           blk.58.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 992/1025]          blk.59.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 993/1025]         blk.59.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 994/1025]              blk.59.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 995/1025]              blk.59.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 996/1025]            blk.59.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 997/1025]               blk.59.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 998/1025]          blk.59.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 999/1025]               blk.59.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[1000/1025]              blk.59.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1001/1025]          blk.59.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[1002/1025]         blk.59.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[1003/1025]          blk.59.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[1004/1025]           blk.59.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1005/1025]         blk.59.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[1006/1025]               blk.59.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1007/1025]            blk.59.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[1008/1025]           blk.59.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[1009/1025]          blk.60.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[1010/1025]         blk.60.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1011/1025]              blk.60.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[1012/1025]              blk.60.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1013/1025]            blk.60.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[1014/1025]               blk.60.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[1015/1025]          blk.60.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1016/1025]               blk.60.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[1017/1025]              blk.60.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1018/1025]          blk.60.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[1019/1025]         blk.60.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[1020/1025]          blk.60.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[1021/1025]           blk.60.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1022/1025]         blk.60.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[1023/1025]               blk.60.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1024/1025]            blk.60.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[1025/1025]           blk.60.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
llama_model_quantize_internal: model size  = 680237.97 MB
llama_model_quantize_internal: quant size  = 364082.97 MB

main: quantize time = 13350534.07 ms
main:    total time = 13350534.07 ms
```

</details>

This mix functions (albeit a bit slow for my liking) and we know q8 functions as it was tested before closing #285.

I have had two non functional mixes so far as mentioned in https://github.com/ikawrakow/ik_llama.cpp/pull/295#issuecomment-2762814972 and the comments that follow.

Two things I didn't mention over there though

1) My functional DeepSeek-V3-0324 mix used bartowski's imatrix file and the two non functional one used the one from team mradermacher. 

2) The second broken mix (where I was going to test setting output.weight to iq6_k), I ended up realizing after I tested it I messed up the custom quant rule and it actually ended up being q6_k_r4 for both `blk.X.attn_output.weight` and `output.weight` so the fact that it didn't work is even more suprising when looking at it versus the working R1 mix, and why my next mix went back to the imatrix dataset I know worked for me.

I just finished testing a 4th mix going back to bartowski's and it is also not functional. It seems to babble vaguely related tokens to ones that make sense before it turns to `Alright` spam (although the probability of Alright is not actually 100% so it will deviate).

Command used to make this fourth quant 
```
./llama-quantize --imatrix /mnt/sda/deepseek-ai_DeepSeek-V3-0324.imatrix --custom-q ".*\.attn_output.weight=q5_k_r4,output\.weight=q6_k_r4,.*=iq4_k_r4" /mnt/sda/DeepseekV3_0324/DeepseekV3_0324-256x21B-BF16.gguf /mnt/sda/DeepSeek-V3-0324-IQ4_K_R4_ATT4.gguf IQ4_K_R4 48
```

---

👤 **saood06** commented the **2025-03-30** at **20:23:35**:<br>

My working mix:
llama_model_loader: - type f32: 361 tensors
llama_model_loader: - type q8_0: 246 tensors
llama_model_loader: - type iq4_k_r4: 357 tensors
llama_model_loader: - type iq5_k_r4: 61 tensors

Full quant log below:

<details>

<summary>Log</summary>

./bin/llama-quantize --allow-requantize  --imatrix /mnt/sda/deepseek-ai_DeepSeek-V3-0324.imatrix --token-embedding-type q8_0 --output-tensor-type q8_0 /mnt/sda/deepseek-ai_DeepSeek-V3-0324-Q8_0/deepseek-a                                     i_DeepSeek-V3-0324-Q8_0-00001-of-00020.gguf /mnt/sda/DeepSeek-V3-0324-IQ4_K_R4.gguf IQ4_K_R4 48
load_imatrix: imatrix dataset='/workspace/calibration_datav3.txt'
load_imatrix: loaded 720 importance matrix entries from /mnt/sda/deepseek-ai_DeepSeek-V3-0324.imatrix computed on 124 chunks
prepare_imatrix: have 720 importance matrix entries
main: build = 3617 (f31aca2d)
main: built with gcc (Clear Linux OS for Intel Architecture) 14.2.1 20241210 releases/gcc-14.2.0-551-g21a09f0507 for x86_64-generic-linux
main: quantizing '/mnt/sda/deepseek-ai_DeepSeek-V3-0324-Q8_0/deepseek-ai_DeepSeek-V3-0324-Q8_0-00001-of-00020.gguf' to '/mnt/sda/DeepSeek-V3-0324-IQ4_K_R4.gguf' as IQ4_K_R4 using 48 threads
llama_model_loader: additional 19 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 47 key-value pairs and 1025 tensors from /mnt/sda/deepseek-ai_DeepSeek-V3-0324-Q8_0/deepseek-ai_DeepSeek-V3-0324-Q8_0-00001-of-00020.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = DeepSeek V3 0324 Bf16
llama_model_loader: - kv   3:                         general.size_label str              = 256x20B
llama_model_loader: - kv   4:                            general.license str              = mit
llama_model_loader: - kv   5:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv   6:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv   7:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv   8:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv   9:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv  10:          deepseek2.attention.head_count_kv u32              = 128
llama_model_loader: - kv  11:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  12: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  13:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  14:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  15:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  16:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  17:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  18:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  19:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  20:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  21:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  22:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  23:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  24:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  25:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  26:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  27:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  28:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  29: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  30: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  31:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  32:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  33:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<▒...
llama_model_loader: - kv  34:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  35:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
llama_model_loader: - kv  36:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  37:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  38:            tokenizer.ggml.padding_token_id u32              = 1
llama_model_loader: - kv  39:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  40:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  41:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  42:               general.quantization_version u32              = 2
llama_model_loader: - kv  43:                          general.file_type u32              = 7
llama_model_loader: - kv  44:                                   split.no u16              = 0
llama_model_loader: - kv  45:                        split.tensors.count i32              = 1025
llama_model_loader: - kv  46:                                split.count u16              = 20
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  664 tensors
================================ Have weights data with 720 entries
[   1/1025]                        output.weight - [ 7168, 129280,     1,     1], type =   q8_0, size =  938.984 MB
[   2/1025]                   output_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[   3/1025]                    token_embd.weight - [ 7168, 129280,     1,     1], type =   q8_0, size =  938.984 MB
[   4/1025]           blk.0.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[   5/1025]          blk.0.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[   6/1025]               blk.0.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[   7/1025]               blk.0.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[   8/1025]             blk.0.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[   9/1025]                blk.0.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[  10/1025]           blk.0.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  11/1025]                blk.0.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[  12/1025]                blk.0.ffn_down.weight - [18432,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =   133.88 MiB ->    70.88 MiB
[  13/1025]                blk.0.ffn_gate.weight - [ 7168, 18432,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =   133.88 MiB ->    70.88 MiB
[  14/1025]                blk.0.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  15/1025]                  blk.0.ffn_up.weight - [ 7168, 18432,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =   133.88 MiB ->    70.88 MiB
[  16/1025]           blk.1.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[  17/1025]          blk.1.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  18/1025]               blk.1.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[  19/1025]               blk.1.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  20/1025]             blk.1.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[  21/1025]                blk.1.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[  22/1025]           blk.1.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  23/1025]                blk.1.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[  24/1025]                blk.1.ffn_down.weight - [18432,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =   133.88 MiB ->    70.88 MiB
[  25/1025]                blk.1.ffn_gate.weight - [ 7168, 18432,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =   133.88 MiB ->    70.88 MiB
[  26/1025]                blk.1.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  27/1025]                  blk.1.ffn_up.weight - [ 7168, 18432,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =   133.88 MiB ->    70.88 MiB
[  28/1025]           blk.2.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[  29/1025]          blk.2.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  30/1025]               blk.2.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[  31/1025]               blk.2.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  32/1025]             blk.2.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[  33/1025]                blk.2.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[  34/1025]           blk.2.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  35/1025]                blk.2.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[  36/1025]                blk.2.ffn_down.weight - [18432,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =   133.88 MiB ->    70.88 MiB
[  37/1025]                blk.2.ffn_gate.weight - [ 7168, 18432,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =   133.88 MiB ->    70.88 MiB
[  38/1025]                blk.2.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  39/1025]                  blk.2.ffn_up.weight - [ 7168, 18432,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =   133.88 MiB ->    70.88 MiB
[  40/1025]           blk.3.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[  41/1025]          blk.3.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  42/1025]               blk.3.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[  43/1025]               blk.3.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  44/1025]             blk.3.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[  45/1025]                blk.3.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[  46/1025]           blk.3.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  47/1025]                blk.3.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[  48/1025]               blk.3.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[  49/1025]           blk.3.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[  50/1025]          blk.3.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[  51/1025]           blk.3.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[  52/1025]            blk.3.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[  53/1025]          blk.3.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[  54/1025]                blk.3.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  55/1025]             blk.3.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[  56/1025]            blk.3.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[  57/1025]           blk.4.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[  58/1025]          blk.4.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  59/1025]               blk.4.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[  60/1025]               blk.4.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  61/1025]             blk.4.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[  62/1025]                blk.4.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[  63/1025]           blk.4.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  64/1025]                blk.4.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[  65/1025]               blk.4.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[  66/1025]           blk.4.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[  67/1025]          blk.4.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[  68/1025]           blk.4.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[  69/1025]            blk.4.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[  70/1025]          blk.4.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[  71/1025]                blk.4.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  72/1025]             blk.4.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[  73/1025]            blk.4.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[  74/1025]           blk.5.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[  75/1025]          blk.5.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  76/1025]               blk.5.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[  77/1025]               blk.5.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  78/1025]             blk.5.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[  79/1025]                blk.5.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[  80/1025]           blk.5.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  81/1025]                blk.5.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[  82/1025]               blk.5.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[  83/1025]           blk.5.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[  84/1025]          blk.5.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[  85/1025]           blk.5.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[  86/1025]            blk.5.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[  87/1025]          blk.5.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[  88/1025]                blk.5.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  89/1025]             blk.5.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[  90/1025]            blk.5.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[  91/1025]           blk.6.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[  92/1025]          blk.6.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  93/1025]               blk.6.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[  94/1025]               blk.6.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  95/1025]             blk.6.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[  96/1025]                blk.6.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[  97/1025]           blk.6.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  98/1025]                blk.6.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[  99/1025]               blk.6.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 100/1025]           blk.6.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 101/1025]          blk.6.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 102/1025]           blk.6.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 103/1025]            blk.6.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 104/1025]          blk.6.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 105/1025]                blk.6.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 106/1025]             blk.6.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 107/1025]            blk.6.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 108/1025]           blk.7.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 109/1025]          blk.7.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 110/1025]               blk.7.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 111/1025]               blk.7.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 112/1025]             blk.7.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 113/1025]                blk.7.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 114/1025]           blk.7.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 115/1025]                blk.7.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 116/1025]               blk.7.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 117/1025]           blk.7.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 118/1025]          blk.7.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 119/1025]           blk.7.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 120/1025]            blk.7.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 121/1025]          blk.7.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 122/1025]                blk.7.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 123/1025]             blk.7.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 124/1025]            blk.7.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 125/1025]           blk.8.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 126/1025]          blk.8.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 127/1025]               blk.8.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 128/1025]               blk.8.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 129/1025]             blk.8.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 130/1025]                blk.8.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 131/1025]           blk.8.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 132/1025]                blk.8.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 133/1025]               blk.8.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 134/1025]           blk.8.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 135/1025]          blk.8.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 136/1025]           blk.8.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 137/1025]            blk.8.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 138/1025]          blk.8.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 139/1025]                blk.8.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 140/1025]             blk.8.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 141/1025]            blk.8.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 142/1025]           blk.9.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 143/1025]          blk.9.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 144/1025]               blk.9.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 145/1025]               blk.9.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 146/1025]             blk.9.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 147/1025]                blk.9.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 148/1025]           blk.9.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 149/1025]                blk.9.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 150/1025]               blk.9.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 151/1025]           blk.9.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 152/1025]          blk.9.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 153/1025]           blk.9.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 154/1025]            blk.9.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 155/1025]          blk.9.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 156/1025]                blk.9.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 157/1025]             blk.9.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 158/1025]            blk.9.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 159/1025]          blk.10.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 160/1025]         blk.10.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 161/1025]              blk.10.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 162/1025]              blk.10.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 163/1025]            blk.10.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 164/1025]               blk.10.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 165/1025]          blk.10.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 166/1025]               blk.10.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 167/1025]              blk.10.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 168/1025]          blk.10.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 169/1025]         blk.10.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 170/1025]          blk.10.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 171/1025]           blk.10.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 172/1025]         blk.10.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 173/1025]               blk.10.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 174/1025]            blk.10.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 175/1025]           blk.10.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 176/1025]          blk.11.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 177/1025]         blk.11.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 178/1025]              blk.11.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 179/1025]              blk.11.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 180/1025]            blk.11.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 181/1025]               blk.11.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 182/1025]          blk.11.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 183/1025]               blk.11.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 184/1025]              blk.11.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 185/1025]          blk.11.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 186/1025]         blk.11.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 187/1025]          blk.11.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 188/1025]           blk.11.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 189/1025]         blk.11.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 190/1025]               blk.11.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 191/1025]            blk.11.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 192/1025]           blk.11.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 193/1025]          blk.12.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 194/1025]         blk.12.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 195/1025]              blk.12.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 196/1025]              blk.12.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 197/1025]            blk.12.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 198/1025]               blk.12.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 199/1025]          blk.12.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 200/1025]               blk.12.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 201/1025]              blk.12.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 202/1025]          blk.12.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 203/1025]         blk.12.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 204/1025]          blk.12.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 205/1025]           blk.12.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 206/1025]         blk.12.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 207/1025]               blk.12.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 208/1025]            blk.12.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 209/1025]           blk.12.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 210/1025]          blk.13.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 211/1025]         blk.13.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 212/1025]              blk.13.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 213/1025]              blk.13.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 214/1025]            blk.13.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 215/1025]               blk.13.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 216/1025]          blk.13.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 217/1025]               blk.13.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 218/1025]              blk.13.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 219/1025]          blk.13.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 220/1025]         blk.13.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 221/1025]          blk.13.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 222/1025]           blk.13.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 223/1025]         blk.13.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 224/1025]               blk.13.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 225/1025]            blk.13.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 226/1025]           blk.13.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 227/1025]          blk.14.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 228/1025]         blk.14.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 229/1025]              blk.14.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 230/1025]              blk.14.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 231/1025]            blk.14.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 232/1025]               blk.14.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 233/1025]          blk.14.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 234/1025]               blk.14.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 235/1025]              blk.14.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 236/1025]          blk.14.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 237/1025]         blk.14.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 238/1025]          blk.14.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 239/1025]           blk.14.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 240/1025]         blk.14.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 241/1025]               blk.14.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 242/1025]            blk.14.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 243/1025]           blk.14.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 244/1025]          blk.15.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 245/1025]         blk.15.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 246/1025]              blk.15.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 247/1025]              blk.15.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 248/1025]            blk.15.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 249/1025]               blk.15.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 250/1025]          blk.15.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 251/1025]               blk.15.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 252/1025]              blk.15.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 253/1025]          blk.15.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 254/1025]         blk.15.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 255/1025]          blk.15.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 256/1025]           blk.15.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 257/1025]         blk.15.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 258/1025]               blk.15.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 259/1025]            blk.15.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 260/1025]           blk.15.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 261/1025]          blk.16.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 262/1025]         blk.16.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 263/1025]              blk.16.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 264/1025]              blk.16.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 265/1025]            blk.16.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 266/1025]               blk.16.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 267/1025]          blk.16.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 268/1025]               blk.16.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 269/1025]              blk.16.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 270/1025]          blk.16.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 271/1025]         blk.16.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 272/1025]          blk.16.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 273/1025]           blk.16.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 274/1025]         blk.16.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 275/1025]               blk.16.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 276/1025]            blk.16.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 277/1025]           blk.16.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 278/1025]          blk.17.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 279/1025]         blk.17.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 280/1025]              blk.17.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 281/1025]              blk.17.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 282/1025]            blk.17.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 283/1025]               blk.17.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 284/1025]          blk.17.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 285/1025]               blk.17.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 286/1025]              blk.17.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 287/1025]          blk.17.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 288/1025]         blk.17.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 289/1025]          blk.17.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 290/1025]           blk.17.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 291/1025]         blk.17.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 292/1025]               blk.17.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 293/1025]            blk.17.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 294/1025]           blk.17.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 295/1025]          blk.18.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 296/1025]         blk.18.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 297/1025]              blk.18.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 298/1025]              blk.18.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 299/1025]            blk.18.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 300/1025]               blk.18.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 301/1025]          blk.18.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 302/1025]               blk.18.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 303/1025]              blk.18.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 304/1025]          blk.18.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 305/1025]         blk.18.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 306/1025]          blk.18.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 307/1025]           blk.18.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 308/1025]         blk.18.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 309/1025]               blk.18.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 310/1025]            blk.18.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 311/1025]           blk.18.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 312/1025]          blk.19.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 313/1025]         blk.19.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 314/1025]              blk.19.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 315/1025]              blk.19.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 316/1025]            blk.19.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 317/1025]               blk.19.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 318/1025]          blk.19.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 319/1025]               blk.19.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 320/1025]              blk.19.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 321/1025]          blk.19.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 322/1025]         blk.19.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 323/1025]          blk.19.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 324/1025]           blk.19.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 325/1025]         blk.19.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 326/1025]               blk.19.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 327/1025]            blk.19.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 328/1025]           blk.19.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 329/1025]          blk.20.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 330/1025]         blk.20.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 331/1025]              blk.20.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 332/1025]              blk.20.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 333/1025]            blk.20.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 334/1025]               blk.20.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 335/1025]          blk.20.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 336/1025]               blk.20.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 337/1025]              blk.20.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 338/1025]          blk.20.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 339/1025]         blk.20.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 340/1025]          blk.20.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 341/1025]           blk.20.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 342/1025]         blk.20.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 343/1025]               blk.20.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 344/1025]            blk.20.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 345/1025]           blk.20.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 346/1025]          blk.21.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 347/1025]         blk.21.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 348/1025]              blk.21.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 349/1025]              blk.21.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 350/1025]            blk.21.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 351/1025]               blk.21.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 352/1025]          blk.21.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 353/1025]               blk.21.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 354/1025]              blk.21.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 355/1025]          blk.21.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 356/1025]         blk.21.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 357/1025]          blk.21.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 358/1025]           blk.21.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 359/1025]         blk.21.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 360/1025]               blk.21.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 361/1025]            blk.21.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 362/1025]           blk.21.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 363/1025]          blk.22.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 364/1025]         blk.22.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 365/1025]              blk.22.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 366/1025]              blk.22.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 367/1025]            blk.22.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 368/1025]               blk.22.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 369/1025]          blk.22.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 370/1025]               blk.22.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 371/1025]              blk.22.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 372/1025]          blk.22.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 373/1025]         blk.22.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 374/1025]          blk.22.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 375/1025]           blk.22.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 376/1025]         blk.22.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 377/1025]               blk.22.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 378/1025]            blk.22.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 379/1025]           blk.22.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 380/1025]          blk.23.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 381/1025]         blk.23.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 382/1025]              blk.23.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 383/1025]              blk.23.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 384/1025]            blk.23.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 385/1025]               blk.23.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 386/1025]          blk.23.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 387/1025]               blk.23.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 388/1025]              blk.23.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 389/1025]          blk.23.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 390/1025]         blk.23.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 391/1025]          blk.23.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 392/1025]           blk.23.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 393/1025]         blk.23.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 394/1025]               blk.23.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 395/1025]            blk.23.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 396/1025]           blk.23.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 397/1025]          blk.24.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 398/1025]         blk.24.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 399/1025]              blk.24.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 400/1025]              blk.24.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 401/1025]            blk.24.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 402/1025]               blk.24.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 403/1025]          blk.24.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 404/1025]               blk.24.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 405/1025]              blk.24.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 406/1025]          blk.24.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 407/1025]         blk.24.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 408/1025]          blk.24.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 409/1025]           blk.24.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 410/1025]         blk.24.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 411/1025]               blk.24.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 412/1025]            blk.24.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 413/1025]           blk.24.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 414/1025]          blk.25.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 415/1025]         blk.25.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 416/1025]              blk.25.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 417/1025]              blk.25.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 418/1025]            blk.25.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 419/1025]               blk.25.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 420/1025]          blk.25.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 421/1025]               blk.25.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 422/1025]              blk.25.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 423/1025]          blk.25.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 424/1025]         blk.25.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 425/1025]          blk.25.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 426/1025]           blk.25.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 427/1025]         blk.25.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 428/1025]               blk.25.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 429/1025]            blk.25.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 430/1025]           blk.25.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 431/1025]          blk.26.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 432/1025]         blk.26.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 433/1025]              blk.26.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 434/1025]              blk.26.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 435/1025]            blk.26.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 436/1025]               blk.26.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 437/1025]          blk.26.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 438/1025]               blk.26.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 439/1025]              blk.26.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 440/1025]          blk.26.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 441/1025]         blk.26.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 442/1025]          blk.26.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 443/1025]           blk.26.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 444/1025]         blk.26.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 445/1025]               blk.26.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 446/1025]            blk.26.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 447/1025]           blk.26.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 448/1025]          blk.27.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 449/1025]         blk.27.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 450/1025]              blk.27.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 451/1025]              blk.27.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 452/1025]            blk.27.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 453/1025]               blk.27.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 454/1025]          blk.27.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 455/1025]               blk.27.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 456/1025]              blk.27.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 457/1025]          blk.27.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 458/1025]         blk.27.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 459/1025]          blk.27.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 460/1025]           blk.27.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 461/1025]         blk.27.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 462/1025]               blk.27.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 463/1025]            blk.27.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 464/1025]           blk.27.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 465/1025]          blk.28.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 466/1025]         blk.28.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 467/1025]              blk.28.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 468/1025]              blk.28.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 469/1025]            blk.28.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 470/1025]               blk.28.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 471/1025]          blk.28.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 472/1025]               blk.28.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 473/1025]              blk.28.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 474/1025]          blk.28.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 475/1025]         blk.28.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 476/1025]          blk.28.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 477/1025]           blk.28.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 478/1025]         blk.28.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 479/1025]               blk.28.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 480/1025]            blk.28.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 481/1025]           blk.28.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 482/1025]          blk.29.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 483/1025]         blk.29.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 484/1025]              blk.29.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 485/1025]              blk.29.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 486/1025]            blk.29.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 487/1025]               blk.29.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 488/1025]          blk.29.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 489/1025]               blk.29.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 490/1025]              blk.29.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 491/1025]          blk.29.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 492/1025]         blk.29.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 493/1025]          blk.29.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 494/1025]           blk.29.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 495/1025]         blk.29.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 496/1025]               blk.29.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 497/1025]            blk.29.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 498/1025]           blk.29.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 499/1025]          blk.30.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 500/1025]         blk.30.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 501/1025]              blk.30.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 502/1025]              blk.30.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 503/1025]            blk.30.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 504/1025]               blk.30.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 505/1025]          blk.30.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 506/1025]               blk.30.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 507/1025]              blk.30.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 508/1025]          blk.30.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 509/1025]         blk.30.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 510/1025]          blk.30.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 511/1025]           blk.30.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 512/1025]         blk.30.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 513/1025]               blk.30.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 514/1025]            blk.30.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 515/1025]           blk.30.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 516/1025]          blk.31.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 517/1025]         blk.31.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 518/1025]              blk.31.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 519/1025]              blk.31.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 520/1025]            blk.31.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 521/1025]               blk.31.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 522/1025]          blk.31.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 523/1025]               blk.31.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 524/1025]              blk.31.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 525/1025]          blk.31.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 526/1025]         blk.31.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 527/1025]          blk.31.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 528/1025]           blk.31.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 529/1025]         blk.31.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 530/1025]               blk.31.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 531/1025]            blk.31.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 532/1025]           blk.31.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 533/1025]          blk.32.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 534/1025]         blk.32.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 535/1025]              blk.32.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 536/1025]              blk.32.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 537/1025]            blk.32.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 538/1025]               blk.32.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 539/1025]          blk.32.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 540/1025]               blk.32.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 541/1025]              blk.32.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 542/1025]          blk.32.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 543/1025]         blk.32.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 544/1025]          blk.32.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 545/1025]           blk.32.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 546/1025]         blk.32.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 547/1025]               blk.32.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 548/1025]            blk.32.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 549/1025]           blk.32.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 550/1025]          blk.33.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 551/1025]         blk.33.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 552/1025]              blk.33.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 553/1025]              blk.33.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 554/1025]            blk.33.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 555/1025]               blk.33.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 556/1025]          blk.33.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 557/1025]               blk.33.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 558/1025]              blk.33.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 559/1025]          blk.33.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 560/1025]         blk.33.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 561/1025]          blk.33.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 562/1025]           blk.33.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 563/1025]         blk.33.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 564/1025]               blk.33.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 565/1025]            blk.33.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 566/1025]           blk.33.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 567/1025]          blk.34.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 568/1025]         blk.34.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 569/1025]              blk.34.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 570/1025]              blk.34.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 571/1025]            blk.34.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 572/1025]               blk.34.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 573/1025]          blk.34.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 574/1025]               blk.34.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 575/1025]              blk.34.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 576/1025]          blk.34.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 577/1025]         blk.34.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 578/1025]          blk.34.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 579/1025]           blk.34.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 580/1025]         blk.34.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 581/1025]               blk.34.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 582/1025]            blk.34.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 583/1025]           blk.34.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 584/1025]          blk.35.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 585/1025]         blk.35.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 586/1025]              blk.35.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 587/1025]              blk.35.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 588/1025]            blk.35.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 589/1025]               blk.35.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 590/1025]          blk.35.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 591/1025]               blk.35.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 592/1025]              blk.35.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 593/1025]          blk.35.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 594/1025]         blk.35.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 595/1025]          blk.35.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 596/1025]           blk.35.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 597/1025]         blk.35.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 598/1025]               blk.35.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 599/1025]            blk.35.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 600/1025]           blk.35.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 601/1025]          blk.36.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 602/1025]         blk.36.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 603/1025]              blk.36.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 604/1025]              blk.36.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 605/1025]            blk.36.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 606/1025]               blk.36.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 607/1025]          blk.36.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 608/1025]               blk.36.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 609/1025]              blk.36.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 610/1025]          blk.36.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 611/1025]         blk.36.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 612/1025]          blk.36.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 613/1025]           blk.36.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 614/1025]         blk.36.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 615/1025]               blk.36.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 616/1025]            blk.36.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 617/1025]           blk.36.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 618/1025]          blk.37.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 619/1025]         blk.37.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 620/1025]              blk.37.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 621/1025]              blk.37.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 622/1025]            blk.37.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 623/1025]               blk.37.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 624/1025]          blk.37.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 625/1025]               blk.37.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 626/1025]              blk.37.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 627/1025]          blk.37.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 628/1025]         blk.37.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 629/1025]          blk.37.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 630/1025]           blk.37.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 631/1025]         blk.37.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 632/1025]               blk.37.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 633/1025]            blk.37.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 634/1025]           blk.37.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 635/1025]          blk.38.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 636/1025]         blk.38.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 637/1025]              blk.38.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 638/1025]              blk.38.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 639/1025]            blk.38.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 640/1025]               blk.38.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 641/1025]          blk.38.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 642/1025]               blk.38.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 643/1025]              blk.38.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 644/1025]          blk.38.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 645/1025]         blk.38.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 646/1025]          blk.38.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 647/1025]           blk.38.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 648/1025]         blk.38.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 649/1025]               blk.38.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 650/1025]            blk.38.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 651/1025]           blk.38.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 652/1025]          blk.39.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 653/1025]         blk.39.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 654/1025]              blk.39.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 655/1025]              blk.39.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 656/1025]            blk.39.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 657/1025]               blk.39.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 658/1025]          blk.39.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 659/1025]               blk.39.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 660/1025]              blk.39.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 661/1025]          blk.39.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 662/1025]         blk.39.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 663/1025]          blk.39.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 664/1025]           blk.39.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 665/1025]         blk.39.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 666/1025]               blk.39.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 667/1025]            blk.39.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 668/1025]           blk.39.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 669/1025]          blk.40.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 670/1025]         blk.40.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 671/1025]              blk.40.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 672/1025]              blk.40.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 673/1025]            blk.40.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 674/1025]               blk.40.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 675/1025]          blk.40.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 676/1025]               blk.40.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 677/1025]              blk.40.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 678/1025]          blk.40.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 679/1025]         blk.40.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 680/1025]          blk.40.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 681/1025]           blk.40.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 682/1025]         blk.40.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 683/1025]               blk.40.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 684/1025]            blk.40.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 685/1025]           blk.40.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 686/1025]          blk.41.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 687/1025]         blk.41.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 688/1025]              blk.41.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 689/1025]              blk.41.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 690/1025]            blk.41.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 691/1025]               blk.41.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 692/1025]          blk.41.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 693/1025]               blk.41.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 694/1025]              blk.41.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 695/1025]          blk.41.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 696/1025]         blk.41.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 697/1025]          blk.41.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 698/1025]           blk.41.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 699/1025]         blk.41.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 700/1025]               blk.41.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 701/1025]            blk.41.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 702/1025]           blk.41.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 703/1025]          blk.42.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 704/1025]         blk.42.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 705/1025]              blk.42.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 706/1025]              blk.42.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 707/1025]            blk.42.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 708/1025]               blk.42.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 709/1025]          blk.42.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 710/1025]               blk.42.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 711/1025]              blk.42.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 712/1025]          blk.42.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 713/1025]         blk.42.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 714/1025]          blk.42.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 715/1025]           blk.42.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 716/1025]         blk.42.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 717/1025]               blk.42.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 718/1025]            blk.42.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 719/1025]           blk.42.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 720/1025]          blk.43.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 721/1025]         blk.43.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 722/1025]              blk.43.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 723/1025]              blk.43.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 724/1025]            blk.43.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 725/1025]               blk.43.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 726/1025]          blk.43.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 727/1025]               blk.43.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 728/1025]              blk.43.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 729/1025]          blk.43.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 730/1025]         blk.43.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 731/1025]          blk.43.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 732/1025]           blk.43.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 733/1025]         blk.43.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 734/1025]               blk.43.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 735/1025]            blk.43.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 736/1025]           blk.43.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 737/1025]          blk.44.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 738/1025]         blk.44.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 739/1025]              blk.44.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 740/1025]              blk.44.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 741/1025]            blk.44.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 742/1025]               blk.44.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 743/1025]          blk.44.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 744/1025]               blk.44.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 745/1025]              blk.44.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 746/1025]          blk.44.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 747/1025]         blk.44.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 748/1025]          blk.44.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 749/1025]           blk.44.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 750/1025]         blk.44.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 751/1025]               blk.44.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 752/1025]            blk.44.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 753/1025]           blk.44.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 754/1025]          blk.45.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 755/1025]         blk.45.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 756/1025]              blk.45.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 757/1025]              blk.45.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 758/1025]            blk.45.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 759/1025]               blk.45.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 760/1025]          blk.45.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 761/1025]               blk.45.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 762/1025]              blk.45.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 763/1025]          blk.45.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 764/1025]         blk.45.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 765/1025]          blk.45.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 766/1025]           blk.45.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 767/1025]         blk.45.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 768/1025]               blk.45.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 769/1025]            blk.45.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 770/1025]           blk.45.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 771/1025]          blk.46.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 772/1025]         blk.46.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 773/1025]              blk.46.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 774/1025]              blk.46.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 775/1025]            blk.46.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 776/1025]               blk.46.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 777/1025]          blk.46.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 778/1025]               blk.46.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 779/1025]              blk.46.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 780/1025]          blk.46.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 781/1025]         blk.46.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 782/1025]          blk.46.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 783/1025]           blk.46.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 784/1025]         blk.46.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 785/1025]               blk.46.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 786/1025]            blk.46.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 787/1025]           blk.46.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 788/1025]          blk.47.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 789/1025]         blk.47.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 790/1025]              blk.47.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 791/1025]              blk.47.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 792/1025]            blk.47.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 793/1025]               blk.47.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 794/1025]          blk.47.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 795/1025]               blk.47.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 796/1025]              blk.47.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 797/1025]          blk.47.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 798/1025]         blk.47.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 799/1025]          blk.47.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 800/1025]           blk.47.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 801/1025]         blk.47.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 802/1025]               blk.47.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 803/1025]            blk.47.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 804/1025]           blk.47.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 805/1025]          blk.48.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 806/1025]         blk.48.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 807/1025]              blk.48.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 808/1025]              blk.48.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 809/1025]            blk.48.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 810/1025]               blk.48.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 811/1025]          blk.48.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 812/1025]               blk.48.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 813/1025]              blk.48.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 814/1025]          blk.48.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 815/1025]         blk.48.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 816/1025]          blk.48.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 817/1025]           blk.48.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 818/1025]         blk.48.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 819/1025]               blk.48.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 820/1025]            blk.48.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 821/1025]           blk.48.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 822/1025]          blk.49.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 823/1025]         blk.49.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 824/1025]              blk.49.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 825/1025]              blk.49.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 826/1025]            blk.49.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 827/1025]               blk.49.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 828/1025]          blk.49.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 829/1025]               blk.49.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 830/1025]              blk.49.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 831/1025]          blk.49.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 832/1025]         blk.49.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 833/1025]          blk.49.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 834/1025]           blk.49.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 835/1025]         blk.49.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 836/1025]               blk.49.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 837/1025]            blk.49.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 838/1025]           blk.49.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 839/1025]          blk.50.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 840/1025]         blk.50.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 841/1025]              blk.50.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 842/1025]              blk.50.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 843/1025]            blk.50.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 844/1025]               blk.50.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 845/1025]          blk.50.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 846/1025]               blk.50.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 847/1025]              blk.50.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 848/1025]          blk.50.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 849/1025]         blk.50.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 850/1025]          blk.50.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 851/1025]           blk.50.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 852/1025]         blk.50.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 853/1025]               blk.50.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 854/1025]            blk.50.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 855/1025]           blk.50.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 856/1025]          blk.51.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 857/1025]         blk.51.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 858/1025]              blk.51.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 859/1025]              blk.51.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 860/1025]            blk.51.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 861/1025]               blk.51.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 862/1025]          blk.51.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 863/1025]               blk.51.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 864/1025]              blk.51.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 865/1025]          blk.51.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 866/1025]         blk.51.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 867/1025]          blk.51.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 868/1025]           blk.51.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 869/1025]         blk.51.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 870/1025]               blk.51.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 871/1025]            blk.51.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 872/1025]           blk.51.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 873/1025]          blk.52.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 874/1025]         blk.52.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 875/1025]              blk.52.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 876/1025]              blk.52.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 877/1025]            blk.52.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 878/1025]               blk.52.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 879/1025]          blk.52.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 880/1025]               blk.52.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 881/1025]              blk.52.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 882/1025]          blk.52.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 883/1025]         blk.52.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 884/1025]          blk.52.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 885/1025]           blk.52.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 886/1025]         blk.52.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 887/1025]               blk.52.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 888/1025]            blk.52.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 889/1025]           blk.52.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 890/1025]          blk.53.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 891/1025]         blk.53.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 892/1025]              blk.53.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 893/1025]              blk.53.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 894/1025]            blk.53.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 895/1025]               blk.53.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 896/1025]          blk.53.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 897/1025]               blk.53.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 898/1025]              blk.53.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 899/1025]          blk.53.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 900/1025]         blk.53.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 901/1025]          blk.53.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 902/1025]           blk.53.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 903/1025]         blk.53.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 904/1025]               blk.53.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 905/1025]            blk.53.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 906/1025]           blk.53.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 907/1025]          blk.54.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 908/1025]         blk.54.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 909/1025]              blk.54.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 910/1025]              blk.54.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 911/1025]            blk.54.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 912/1025]               blk.54.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 913/1025]          blk.54.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 914/1025]               blk.54.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 915/1025]              blk.54.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 916/1025]          blk.54.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 917/1025]         blk.54.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 918/1025]          blk.54.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 919/1025]           blk.54.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 920/1025]         blk.54.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 921/1025]               blk.54.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 922/1025]            blk.54.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 923/1025]           blk.54.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 924/1025]          blk.55.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 925/1025]         blk.55.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 926/1025]              blk.55.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 927/1025]              blk.55.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 928/1025]            blk.55.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 929/1025]               blk.55.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 930/1025]          blk.55.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 931/1025]               blk.55.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 932/1025]              blk.55.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 933/1025]          blk.55.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 934/1025]         blk.55.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 935/1025]          blk.55.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 936/1025]           blk.55.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 937/1025]         blk.55.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 938/1025]               blk.55.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 939/1025]            blk.55.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 940/1025]           blk.55.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 941/1025]          blk.56.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 942/1025]         blk.56.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 943/1025]              blk.56.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 944/1025]              blk.56.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 945/1025]            blk.56.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 946/1025]               blk.56.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 947/1025]          blk.56.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 948/1025]               blk.56.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 949/1025]              blk.56.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 950/1025]          blk.56.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 951/1025]         blk.56.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 952/1025]          blk.56.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 953/1025]           blk.56.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 954/1025]         blk.56.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 955/1025]               blk.56.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 956/1025]            blk.56.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 957/1025]           blk.56.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 958/1025]          blk.57.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 959/1025]         blk.57.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 960/1025]              blk.57.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 961/1025]              blk.57.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 962/1025]            blk.57.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 963/1025]               blk.57.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 964/1025]          blk.57.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 965/1025]               blk.57.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 966/1025]              blk.57.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 967/1025]          blk.57.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 968/1025]         blk.57.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 969/1025]          blk.57.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 970/1025]           blk.57.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 971/1025]         blk.57.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 972/1025]               blk.57.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 973/1025]            blk.57.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 974/1025]           blk.57.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 975/1025]          blk.58.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 976/1025]         blk.58.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 977/1025]              blk.58.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 978/1025]              blk.58.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 979/1025]            blk.58.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 980/1025]               blk.58.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 981/1025]          blk.58.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 982/1025]               blk.58.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[ 983/1025]              blk.58.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 984/1025]          blk.58.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 985/1025]         blk.58.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 986/1025]          blk.58.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 987/1025]           blk.58.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 988/1025]         blk.58.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 989/1025]               blk.58.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 990/1025]            blk.58.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[ 991/1025]           blk.58.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[ 992/1025]          blk.59.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[ 993/1025]         blk.59.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 994/1025]              blk.59.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[ 995/1025]              blk.59.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 996/1025]            blk.59.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[ 997/1025]               blk.59.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[ 998/1025]          blk.59.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 999/1025]               blk.59.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[1000/1025]              blk.59.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1001/1025]          blk.59.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[1002/1025]         blk.59.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[1003/1025]          blk.59.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[1004/1025]           blk.59.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1005/1025]         blk.59.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[1006/1025]               blk.59.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1007/1025]            blk.59.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[1008/1025]           blk.59.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[1009/1025]          blk.60.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, size =    4.184 MB
[1010/1025]         blk.60.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1011/1025]              blk.60.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q8_0, size =   17.000 MB
[1012/1025]              blk.60.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1013/1025]            blk.60.attn_output.weight - [16384,  7168,     1,     1], type =   q8_0, converting to iq5_k_r4 .. size =   119.00 MiB ->    77.00 MiB
[1014/1025]               blk.60.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, size =   11.156 MB
[1015/1025]          blk.60.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1016/1025]               blk.60.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q8_0, size =   38.250 MB
[1017/1025]              blk.60.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1018/1025]          blk.60.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[1019/1025]         blk.60.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[1020/1025]          blk.60.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[1021/1025]           blk.60.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1022/1025]         blk.60.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
[1023/1025]               blk.60.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1024/1025]            blk.60.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q8_0, converting to iq4_k_r4 .. size =  3808.00 MiB ->  2016.00 MiB
[1025/1025]           blk.60.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, converting to iq4_k_r4 .. size =    14.88 MiB ->     7.88 MiB
llama_model_quantize_internal: model size  = 680237.97 MB
llama_model_quantize_internal: quant size  = 364082.97 MB

main: quantize time = 13350534.07 ms
main:    total time = 13350534.07 ms

</details>

This mix functions (albeit a bit slow for my liking) and we know q8 functions as it was tested before closing #285.

I have had two non functional mixes so far as mentioned in https://github.com/ikawrakow/ik_llama.cpp/pull/295#issuecomment-2762814972 and the comments that follow.

Two things I didn't mention over there though

1) My functional DeepSeek-V3-0324 mix used bartowski's imatrix file and the two non functional one used the one from team mradermacher. 

2) The second broken mix (where I was going to test setting output.weight to iq6_k), I ended up realizing after I tested it I messed up the custom quant rule and it actually ended up being q6_k_r4 for both `blk.X.attn_output.weight` and `output.weight` so the fact that it didn't work is even more suprising when looking at it versus the working R1 mix, and why my next mix went back to the imatrix dataset I know worked for me.

I just finished testing a 4th mix going back to bartowski's and it is also not functional. It seems to babble vaguely related tokens to ones that make sense before it turns to `Alright` spam (although the probability of Alright is not actually 100% so it will deviate).

Command used to make this fourth quant 
```
./llama-quantize --imatrix /mnt/sda/deepseek-ai_DeepSeek-V3-0324.imatrix --custom-q ".*\.attn_output.weight=q5_k_r4,output\.weight=q6_k_r4,.*=iq4_k_r4" /mnt/sda/DeepseekV3_0324/DeepseekV3_0324-256x21B-BF16.gguf /mnt/sda/DeepSeek-V3-0324-IQ4_K_R4_ATT4.gguf IQ4_K_R4 48
```

---

👤 **ubergarm** commented the **2025-03-30** at **21:11:28**:<br>

> This mix functions (albeit a bit slow for my liking) and we know q8 functions as it was tested before closing https://github.com/ikawrakow/ik_llama.cpp/issues/285.

Okay, thanks for confirming success with those tensor types. I'll re-cooking again just changing `q8_0_r8` to `q8_0` to see if there is any effect. Plus it would allow use on GPU.

> The second broken mix (where I was going to test setting output.weight to iq6_k), I ended up realizing after I tested it I messed up the custom quant rule and it actually ended up being q6_k_r4 for both blk.X.attn_output.weight and output.weight

> ...4th mix going back to bartowski's and it is also not functional....

Hrmm, I'm wondering if this has something to do with setting `token_embd.weight` weight to repacked quant types? I'm speculating wildly, hopefully my above test will give another datapoint though.

I recall when I used the offline-repack tool with a `Q8_0` it converted everything to `q8_0_r8` except for one tensor, which stuck out to me but I didn't think much of it at the time:
```
[1/1025] token_embd.weight - [ 7168, 129280,1,1], type = q8_0, size =  938.984 MB, type = q8_0
```

> I just finished testing a 4th mix going back to bartowski's and it is also not functional. It seems to babble vaguely related tokens to ones that make sense before it turns to Alright spam (although the probability of Alright is not actually 100% so it will deviate).

I see, yeah a lot of variables in play with multiple imatrix files and all. Interesting it also babbles `Alright` sometimes.

Anyway, I'll keep you posted if I get one cooked up that seems to be working better and narrow down if it is anything odd going on or just not all quants play well together on this model.

---

👤 **ubergarm** commented the **2025-03-30** at **21:11:28**:<br>

> This mix functions (albeit a bit slow for my liking) and we know q8 functions as it was tested before closing https://github.com/ikawrakow/ik_llama.cpp/issues/285.

Okay, thanks for confirming success with those tensor types. I'll re-cooking again just changing `q8_0_r8` to `q8_0` to see if there is any effect. Plus it would allow use on GPU.

> The second broken mix (where I was going to test setting output.weight to iq6_k), I ended up realizing after I tested it I messed up the custom quant rule and it actually ended up being q6_k_r4 for both blk.X.attn_output.weight and output.weight

> ...4th mix going back to bartowski's and it is also not functional....

Hrmm, I'm wondering if this has something to do with setting `token_embd.weight` weight to repacked quant types? I'm speculating wildly, hopefully my above test will give another datapoint though.

I recall when I used the offline-repack tool with a `Q8_0` it converted everything to `q8_0_r8` except for one tensor, which stuck out to me but I didn't think much of it at the time:
```

[   1/1025]                    token_embd.weight - [ 7168, 129280,     1,     1], type =   q8_0, size =  938.984 MB, type = q8_0
```

> I just finished testing a 4th mix going back to bartowski's and it is also not functional. It seems to babble vaguely related tokens to ones that make sense before it turns to Alright spam (although the probability of Alright is not actually 100% so it will deviate).

I see, yeah a lot of variables in play with multiple imatrix files and all. Interesting it also babbles `Alright` sometimes.

Anyway, I'll keep you posted if I get one cooked up that seems to be working better and narrow down if it is anything odd going on or just not all quants play well together on this model.

---

👤 **saood06** commented the **2025-03-30** at **21:30:18**:<br>

> > This mix functions (albeit a bit slow for my liking) and we know q8 functions as it was tested before closing [#285](https://github.com/ikawrakow/ik_llama.cpp/issues/285).

> Okay, thanks for confirming success with those tensor types. I'll re-cooking again just changing `q8_0_r8` to `q8_0` to see if there is any effect. Plus it would allow use on GPU.

Thanks, I don't have any mix cooking right, but I could do one overnight to test another mix if that would be helpful.

> Hrmm, I'm wondering if this has something to do with setting `token_embd.weight` weight to repacked quant types? I'm speculating wildly, hopefully my above test will give another datapoint though.
> 
> I recall when I used the offline-repack tool with a `Q8_0` it converted everything to `q8_0_r8` except for one tensor, which stuck out to me but I didn't think much of it at the time:
> 
> ```
> [1/1025] token_embd.weight - [ 7168, 129280,1,1], type = q8_0, size =  938.984 MB, type = q8_0
> ```
> 

I don't think it is a wild speculation.

It might be the reason, see [this](https://github.com/ikawrakow/ik_llama.cpp/pull/272/files#diff-b74fdb6e796b36d230cafcbff50ebd34cf27bd55b6b4ca0ad5a2ff8191b1066bR6784-R6786) and [this](https://github.com/ikawrakow/ik_llama.cpp/blob/4819257ce66a680608cf9c7871156041d00eb7da/src/llama.cpp#L16920).

Also now that you do mention it I do think something about this was brought up at some point but I can't remember where (so no reference).

> > I just finished testing a 4th mix going back to bartowski's and it is also not functional. It seems to babble vaguely related tokens to ones that make sense before it turns to Alright spam (although the probability of Alright is not actually 100% so it will deviate).
> 
> I see, yeah a lot of variables in play with multiple imatrix files and all. Interesting it also babbles `Alright` sometimes.
> 
> Anyway, I'll keep you posted if I get one cooked up that seems to be working better and narrow down if it is anything odd going on or just not all quants play well together on this model.

Thanks, I'll do the same.

---

👤 **saood06** commented the **2025-03-30** at **21:30:18**:<br>

> > This mix functions (albeit a bit slow for my liking) and we know q8 functions as it was tested before closing [#285](https://github.com/ikawrakow/ik_llama.cpp/issues/285).

> Okay, thanks for confirming success with those tensor types. I'll re-cooking again just changing `q8_0_r8` to `q8_0` to see if there is any effect. Plus it would allow use on GPU.

Thanks, I don't have any mix cooking right, but I could do one overnight to test another mix if that would be helpful.

> Hrmm, I'm wondering if this has something to do with setting `token_embd.weight` weight to repacked quant types? I'm speculating wildly, hopefully my above test will give another datapoint though.
> 
> I recall when I used the offline-repack tool with a `Q8_0` it converted everything to `q8_0_r8` except for one tensor, which stuck out to me but I didn't think much of it at the time:
> 
> ```
> [1/1025] token_embd.weight - [ 7168, 129280,1,1], type = q8_0, size =  938.984 MB, type = q8_0
> ```
> 

I don't think it is a wild speculation.

This might be the reason, see [this](https://github.com/ikawrakow/ik_llama.cpp/pull/272/files#diff-b74fdb6e796b36d230cafcbff50ebd34cf27bd55b6b4ca0ad5a2ff8191b1066bR6784-R6786) and [this](https://github.com/ikawrakow/ik_llama.cpp/blob/4819257ce66a680608cf9c7871156041d00eb7da/src/llama.cpp#L16920).

Also now that you do mention it I do think something about this was brought up at some point but I can't remember where (so no reference).

> > I just finished testing a 4th mix going back to bartowski's and it is also not functional. It seems to babble vaguely related tokens to ones that make sense before it turns to Alright spam (although the probability of Alright is not actually 100% so it will deviate).
> 
> I see, yeah a lot of variables in play with multiple imatrix files and all. Interesting it also babbles `Alright` sometimes.
> 
> Anyway, I'll keep you posted if I get one cooked up that seems to be working better and narrow down if it is anything odd going on or just not all quants play well together on this model.

---

👤 **ubergarm** commented the **2025-03-31** at **00:42:54**:<br>

> I don't think it is a wild speculation.
>
> It might be the reason, see [this](https://github.com/ikawrakow/ik_llama.cpp/pull/272/files#diff-b74fdb6e796b36d230cafcbff50ebd34cf27bd55b6b4ca0ad5a2ff8191b1066bR6784-R6786) and [this](https://github.com/ikawrakow/ik_llama.cpp/blob/4819257ce66a680608cf9c7871156041d00eb7da/src/llama.cpp#L16920).
>
> Also now that you do mention it I do think something about this was brought up at some point but I can't remember where (so no reference).

Wow thanks, you are really good with keeping track of so much disperate information and links haha...

Seems like logic for `token_embd.weight` is `if (new_type == GGML_TYPE_Q8_0_R8) { new_type = GGML_TYPE_Q8_0; }`

And I am currently testing perplexity on my experiment above using `Q8_0` instead of `Q8_0_R8` quant and its looking just fine:

```
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  612 tensors
llama_model_loader: - type iq4_k_r4:  116 tensors
llama_model_loader: - type iq5_k_r4:   58 tensors
```

So probably yeah, the issue I'm seeing here is because I used `q8_0_r8` for `token_embd.weight` which seems like a known invalid combination.

Gonna let it finish up and curious how good the perplexity is relative to the full `Q8_0` hehe... its addictive...

---

*UPDATE* Wow!! `3.2596 +/- 0.01786` for this `DeepSeek-V3-0324-IQ4_K_R4.gguf` quant vs full `Q8_0` at `3.2454 +/- 0.01773` in almost half the size!

```bash
llm_load_print_meta: model size       = 386.183 GiB (4.936 BPW)

llama_print_timings:        load time =    2327.19 ms
llama_print_timings:      sample time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings: prompt eval time = 3249602.81 ms / 287232 tokens (   11.31 ms per token,    88.39 tokens per second)
llama_print_timings:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings:       total time = 3300377.65 ms / 287233 tokens

Final estimate: PPL = 3.2596 +/- 0.01786
```

</details>

---

👤 **ubergarm** commented the **2025-03-31** at **00:42:54**:<br>

> I don't think it is a wild speculation.
>
> It might be the reason, see [this](https://github.com/ikawrakow/ik_llama.cpp/pull/272/files#diff-b74fdb6e796b36d230cafcbff50ebd34cf27bd55b6b4ca0ad5a2ff8191b1066bR6784-R6786) and [this](https://github.com/ikawrakow/ik_llama.cpp/blob/4819257ce66a680608cf9c7871156041d00eb7da/src/llama.cpp#L16920).
>
> Also now that you do mention it I do think something about this was brought up at some point but I can't remember where (so no reference).

Wow thanks, you are really good with keeping track of so much disperate information and links haha...

Seems like logic for `token_embd.weight` is `if (new_type == GGML_TYPE_Q8_0_R8) { new_type = GGML_TYPE_Q8_0; }`

And I am currently testing perplexity on my experiment above using `Q8_0` instead of `Q8_0_R8` quant and its looking just fine:

```
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  612 tensors
llama_model_loader: - type iq4_k_r4:  116 tensors
llama_model_loader: - type iq5_k_r4:   58 tensors
```

So probably yeah, the issue I'm seeing here is because I used `q8_0_r8` for `token_embd.weight` which seems like a known invalid combination.

Gonna let it finish up and curious how good the perplexity is relative to the full `Q8_0` hehe... its addictive...

---

👤 **saood06** commented the **2025-03-31** at **01:46:10**:<br>

> Wow thanks, you are really good with keeping track of so much disperate information and links haha...

You say this right after I say I don't have a reference (I jest).

> 
> And I am currently testing perplexity on my experiment above using `Q8_0` instead of `Q8_0_R8` quant and its looking just fine:

Nice.

> Gonna let it finish up and curious how good the perplexity is relative to the full `Q8_0` hehe... its addictive...

I know, I want to test my pure IQ4_K_R4 (minus the token_embd.weight), I'm probably going to have that quant cook overnight and test it later. The 4th mix was fast in the preliminary performance screening I did before functionality testing it.

I thought about how the ratio of tokens used by me in sweep-bench vs server and I had an idea that I could tweak sweep-bench to do actually useful work instead of just decoding and prefilling random tokens.

>UPDATE Wow!! 3.2596 +/- 0.01786 for this DeepSeek-V3-0324-IQ4_K_R4.gguf quant vs full Q8_0 at 3.2454 +/- 0.01773 in almost half the size!

Ooh, nice. If you don't mind would you test pure IQ4_K_R4 with IQ4_K token_embd.weight and see how close that gets? I know `-ser` is designed to be used instead, but it would be interesting see it tested for IQ4_K/IQ4_K_R4.

>model size       = 386.183 GiB (4.936 BPW)

Just barely out of reach for my 384 GB RAM server, but I also think that using IQ6_K for some of the Q8_0 could get me there without affecting PPL much at all, but I did experiment with something similar with my third IQ4_K_R4 based mix of R1, which I barely used because I preferred the faster mixes.

---

👤 **ubergarm** commented the **2025-03-31** at **02:02:07**:<br>

> You say this right after I say I don't have a reference (I jest).

😂 

> If you don't mind would you test pure IQ4_K_R4 with IQ4_K token_embd.weight and see how close that gets?

I think I can clean up some disk space now that I know which of my previous gguf's experiments are junk. Do I need to use `--pure` ? Otherwise I'll just update my existing `--custom-q` with your requested types.

> Just barely out of reach for my 384 GB RAM server,

Is this server CPU only? Otherwise all the q8_0's will fit in under 24GB VRAM with 32k context which might barely work for you.

Interesting, yeah chopping the q8_0's could trim a little bit. It's pretty interesting how little of the weights are for attention relative to the MoEs. Psure GPT-3 was like 1/3rd attention weights. Deepseek seems like under 5% or something (didn't actually calculate it). I wonder if making say the last 10 routed experts slightly smaller would save more space while keeping attention maxxed out. Just spitballing, I really dunno what I'm doing haha...

---

👤 **saood06** commented the **2025-03-31** at **02:15:36**:<br>

> > If you don't mind would you test pure IQ4_K_R4 with IQ4_K token_embd.weight and see how close that gets?
> 
> I think I can clean up some disk space now that I know which of my previous gguf's experiments are junk. Do I need to use `--pure` ? Otherwise I'll just update my existing `--custom-q` with your requested types.

You can use whatever you find easier, I find `--custom-q` easier as well, what matters is the mix it produces.

> > Just barely out of reach for my 384 GB RAM server,
> 
> Is this server CPU only? Otherwise all the q8_0's will fit in under 24GB VRAM with 32k context which might barely work for you.

The server is CPU only, I have a 3090 but in another machine that could be used with RPC, but my RPC sync still hasn't progressed to test it here, and my initial testing on llama.cpp showed RPC didn't help with the tensor offload/MLA stuff.

> Interesting, yeah chopping the q8_0's could trim a little bit. It's pretty interesting how little of the weights are for attention relative to the MoEs. Psure GPT-3 was like 1/3rd attention weights. Deepseek seems like under 5% or something (didn't actually calculate it). I wonder if making say the last 10 routed experts slightly smaller would save more space while keeping attention maxxed out. Just spitballing, I really dunno what I'm doing haha...

I'm not sure what your trying to say. MoE's are different from dense models, but both have tensors that are more or less sensitive to being quantized.

---

👤 **ubergarm** commented the **2025-03-31** at **03:21:46**:<br>

> You can use whatever you find easier, I find --custom-q easier as well, what matters is the mix it produces.

Super, it is cooking now, however, I looks like one of the tensors is not happy with `iq4_k_r4` and falling back to `q5_0`. The log is a bit wonky, but it could just be that unused `attn_k_b.weight` so not an actual issue. I'll let it keep going and hopefully get your perplexity by tomorrow morning!

<details>

<summary>quantize snippet for `iq4_k_r4`</summary>

```bash
[ 793/1147]               blk.42.attn_k_b.weight - [  128, 65536,     1,     1], type =   bf16, Using custom type iq4_k_r4 for tensor blk
.42.attn_k_b.weight


change_type_if_necessar : tensor cols 128 x 65536 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.42.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 794/1147]               blk.42.attn_v_b.weight - [  512, 16384,     1,     1], type =   bf16, Using custom type iq4_k_r4 for tensor blk
.42.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 795/1147]            blk.42.attn_output.weight - [16384,  7168,     1,     1], type =   bf16, Using custom type iq4_k_r4 for tensor blk
.42.attn_output.weight
converting to iq4_k_r4 .. size =   224.00 MiB ->    63.00 MiB
[ 796/1147]          blk.42.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 797/1147]               blk.42.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   bf16, Using custom type iq4_k_r4 for tensor blk
.42.attn_q_a.weight
converting to iq4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 798/1147]               blk.42.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   bf16, Using custom type iq4_k_r4 for tensor blk
.42.attn_q_b.weight
converting to iq4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 799/1147]              blk.42.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 800/1147]          blk.42.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   bf16, Using custom type iq4_k_r4 for tensor blk
.42.ffn_down_exps.weight
converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[ 801/1147]          blk.42.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   bf16, Using custom type iq4_k_r4 for tensor blk
.42.ffn_gate_exps.weight
converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[ 802/1147]            blk.42.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   bf16, Using custom type iq4_k_r4 for tensor blk
.42.ffn_up_exps.weight
converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[ 803/1147]               blk.42.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 804/1147]              blk.43.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 805/1147]           blk.43.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 806/1147]         blk.43.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   bf16, Using custom type iq4_k_r4 for tensor blk
.43.ffn_down_shexp.weight
converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 807/1147]         blk.43.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   bf16, Using custom type iq4_k_r4 for tensor blk
.43.ffn_gate_shexp.weight
converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 808/1147]           blk.43.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   bf16, Using custom type iq4_k_r4 for tensor blk
.43.ffn_up_shexp.weight
converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 809/1147]         blk.43.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 810/1147]          blk.43.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   bf16, Using custom type iq4_k_r4 for tensor blk
.43.attn_kv_a_mqa.weight
converting to iq4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 811/1147]              blk.43.attn_kv_b.weight - [  512, 32768,     1,     1], type =   bf16, Using custom type iq4_k_r4 for tensor blk
.43.attn_kv_b.weight
converting to iq4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 812/1147]               blk.43.attn_k_b.weight - [  128, 65536,     1,     1], type =   bf16, Using custom type iq4_k_r4 for tensor blk
.43.attn_k_b.weight


change_type_if_necessar : tensor cols 128 x 65536 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.43.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 813/1147]               blk.43.attn_v_b.weight - [  512, 16384,     1,     1], type =   bf16, Using custom type iq4_k_r4 for tensor blk
.43.attn_v_b.weight
```

</details>

> I have a 3090 but in another machine that could be used with RPC

ooh right right, yeah so all CPU it is.

> I'm not sure what your trying to say. MoE's are different from dense models, but both have tensors that are more or less sensitive to being quantized.

Haha, I'm not sure either 💀 lol... I'm just wondering if trimming weight at say the last 10 layers of the *routed experts* (not MoEs) might drop overall size quicker than trimming it from the already fairly small embeddings/dense layers/attention/norms/bias/shared expert layers.

---

👤 **saood06** commented the **2025-03-31** at **03:37:38**:<br>

> > You can use whatever you find easier, I find --custom-q easier as well, what matters is the mix it produces.
> 
> Super, it is cooking now, however, I looks like one of the tensors is not happy with `iq4_k_r4` and falling back to `q5_0`. 

That is fine and expected for that tensor.

>I'll let it keep going and hopefully get your perplexity by tomorrow morning!

Thanks!

>ooh right right, yeah so all CPU it is.

There are still models (and configurations) where RPC on ik_llama.cpp would benefit performance such as Miqu based quants. Deepseek is just not one of those.

---

👤 **saood06** commented the **2025-03-31** at **03:37:38**:<br>

> > You can use whatever you find easier, I find --custom-q easier as well, what matters is the mix it produces.
> 
> Super, it is cooking now, however, I looks like one of the tensors is not happy with `iq4_k_r4` and falling back to `q5_0`. 

That is fine and expected for that tensor.

>I'll let it keep going and hopefully get your perplexity by tomorrow morning!

Thanks.

>ooh right right, yeah so all CPU it is.

There are still models (and configurations) where RPC on ik_llama.cpp would benefit performance such as Miqu based quants. Deepseek is just not one of those.

---

👤 **ikawrakow** commented the **2025-03-31** at **05:50:51**:<br>

So, `token_embd.weight` cannot be quantized with row-interleaved quants (one needs to be able to get individual single rows out if this tensor to fill the input state, but the row-interleaved quants pack 4 or 8 rows together, so this does not work). I have checks in place, but it looks like I'm not catching all possible paths to arrive at an interleaved quants. So, I guess, until I find and fix the issue it is best to just explicitly specify the type of the `token_embd.weight` tensor with a custom rule. 

`attn_k_b.weight` can't be k-, i-, or iqk-quant because its row size is 128, so not a multiple of 256 as needed by i-, k-, idk-quants. Normally this should be caught and a corresponding legacy quant with a block size of 32 should be used instead.

> UPDATE Wow!! 3.2596 +/- 0.01786 for this DeepSeek-V3-0324-IQ4_K_R4.gguf quant vs full Q8_0 at 3.2454 +/- 0.01773 in almost half the size!

Amazing! You should publish this model.

I second @saood06's request to explore how much quality degradation there will be from moving the attention tensors and the shared experts  to `iq6_k` and `iq5_k`, as this will make CPU-only TG quite a bit faster. For hybrid setups (with attention and shared experts being run on the GPU), one should look into `q6_K/q5_K` instead.

---

👤 **saood06** commented the **2025-03-31** at **06:55:11**:<br>

>So, token_embd.weight cannot be quantized with row-interleaved quants (one needs to be able to get individual single rows out if this tensor to fill the input state, but the row-interleaved quants pack 4 or 8 rows together, so this does not work). I have checks in place, but it looks like I'm not catching all possible paths to arrive at an interleaved quants.

Thanks for the explanation.

> `attn_k_b.weight` can't be k-, i-, or iqk-quant because its row size is 128, so not a multiple of 256 as needed by i-, k-, idk-quants. Normally this should be caught and a corresponding legacy quant with a block size of 32 should be used instead.

I've had situations where it doesn't and llama-quantize crashes.

command: `./llama-quantize --pure --imatrix /mnt/sda/imatrix_V30324_mrader.dat --output-tensor-type q6_k_r4  /mnt/sda/DeepseekV3_0324/DeepseekV3_0324-256x21B-BF16.gguf /mnt/sda/DeepSeek-V3-0324-IQ4_K_R4_ATT3.gguf  IQ4_K_R4 48`

The assert being triggered:
```
====== llama_model_quantize_internal: did not find weights for blk.0.attn_k_b.weight
converting to iq4_k_r4 .. /home/saood06/ik_main/ik_llama.cpp/ggml/src/iqk/iqk_quantize.cpp:5244: GGML_ASSERT(n_per_row%QK_K == 0) failed
```

> 
> > UPDATE Wow!! 3.2596 +/- 0.01786 for this DeepSeek-V3-0324-IQ4_K_R4.gguf quant vs full Q8_0 at 3.2454 +/- 0.01773 in almost half the size!
> 
> Amazing!

Yes. It is impressive how good the quants that can be made from this repo are.

> I second [@saood06](https://github.com/saood06)'s request to explore how much quality degradation there will be from moving the attention tensors and the shared experts to `iq6_k` and `iq5_k`, as this will make CPU-only TG quite a bit faster. 

Yes, and maybe also try to use iq5_k_r4 for less MoE down projection tensors, maybe just the first 3. That should shave off a good bit of size and hopefully maintain almost all of the benefit of the MoE down projection tensors with just the first 3. Writing the `--custom-q` command it should be possible to just specify it for blk 3, blk 4, blk 5, as the first three blocks are dense and don't have any MoE down projection tensors, so they start at blk 3.

>For hybrid setups (with attention and shared experts being run on the GPU), one should look into `q6_K/q5_K` instead.

I wonder how much extra context that would let you squeeze in. I've gone above 32k before and Deepseek docs say "Note that the CoT output can reach up to 32K tokens".

---

👤 **ikawrakow** commented the **2025-03-31** at **08:41:57**:<br>

> I've had situations where it doesn't and llama-quantize crashes.

This happened after PR #294? #294 should have fixed the `--pure` use case.

---

👤 **saood06** commented the **2025-03-31** at **08:58:01**:<br>

> > I've had situations where it doesn't and llama-quantize crashes.
> 
> This happened after PR [#294](https://github.com/ikawrakow/ik_llama.cpp/pull/294)? [#294](https://github.com/ikawrakow/ik_llama.cpp/pull/294) should have fixed the `--pure` use case.

This was before, that looks like it would fix it. Thanks.

---

👤 **ubergarm** commented the **2025-03-31** at **14:22:13**:<br>

> > I'll let it keep going and hopefully get your perplexity by tomorrow morning!

> Thanks!

Just grabbed the log, here is how your "pure" `iq4_k_r4` stacks up on full perplexity run , size, and duration:
| Model | Size (GiB) | PPL | Duration (minutes) |
| --- | --- | --- | --- |
| DeepSeek-V3-0324-IQ2_K_R4 | 227 | 3.5614 +/- 0.02001 | (different rig) | 
| DeepSeek-V3-0324-PURE-IQ4_K_R4 | 353 | 3.2942 +/- 0.01812 | 47.56 | 
| DeepSeek-V3-0324-IQ4_K_R4 | 387 | 3.2596 +/- 0.01786 | 55.01 |
| DeepSeek-V3-0324-Q8_0 | 666 | 3.2454 +/- 0.01773 | 68.87 |

![Image](https://github.com/user-attachments/assets/cf36b5ea-a1ec-4267-a25e-a0c52ccabaef)

In terms of speed to calculate perplexity, these three were similar setups more or less using a single socket of the Xeon 6980P

![Image](https://github.com/user-attachments/assets/b00a57c9-a242-4b07-b945-26de8eae89e7)

#### "PURE" `IQ4_K_R4` perplexity log details
```
main: build = 3613 (4819257c)
main: built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu
main: seed  = 1337

llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q5_0:   61 tensors
llama_model_loader: - type iq4_k:    1 tensors
llama_model_loader: - type iq4_k_r4:  724 tensors

llm_load_print_meta: model size       = 352.470 GiB (4.505 BPW) 

perplexity: calculating perplexity over 561 chunks, n_ctx=512, batch_size=2048, n_seq=4
perplexity: 19.63 seconds per pass - ETA 45.88 minutes
[1]2.4366,[2]3.1393,[3]2.3037,[4]1.9385,[5]1.7532,[6]1.6176,[7]1.5316,[8]1.4745,[9]1.4313,[10]1.3953,[11]1.3829,[12]1.4097,[13]1.4224,[14]1.5443,[15]1.6735,[16]1.7303,[17]1.8888,[18]2.0140,[19]1.9767,[20]1.9637,[21]2.0686,[22]2.0468,[23]2.0218,[24]2.0329,[25]2.0040,[26]1.9824,[27]2.0276,[28]2.0377,[29]2.0839,[30]2.1167,[31]2.1493,[32]2.1657,[33]2.2060,[34]2.2503,[35]2.2965,[36]2.3499,[37]2.3852,[38]2.4336,[39]2.4732,[40]2.5311,[41]2.5728,[42]2.5850,[43]2.6354,[44]2.6530,[45]2.7332,[46]2.7820,[47]2.7394,[48]2.6930,[49]2.6667,[50]2.6835,[51]2.7280,[52]2.7399,[53]2.7902,[54]2.8021,[55]2.8316,[56]2.8626,[57]2.8758,[58]2.9093,[59]2.9190,[60]2.9659,[61]3.0052,[62]3.0520,[63]3.0836,[64]3.1250,[65]3.1341,[66]3.1157,[67]3.0915,[68]3.1179,[69]3.1110,[70]3.1238,[71]3.1416,[72]3.1557,[73]3.1697,[74]3.1909,[75]3.1705,[76]3.1256,[77]3.0826,[78]3.0789,[79]3.0595,[80]3.0426,[81]3.0078,[82]3.0106,[83]2.9793,[84]2.9450,[85]2.9116,[86]2.8887,[87]2.8825,[88]2.8559,[89]2.8395,[90]2.8144,[91]2.7862,[92]2.7616,[93]2.7362,[94]2.7115,[95]2.6895,[96]2.6870,[97]2.6926,[98]2.6774,[99]2.6605,[100]2.6627,[101]2.6544,[102]2.6697,[103]2.6946,[104]2.7113,[105]2.7078,[106]2.7294,[107]2.7536,[108]2.7740,[109]2.8065,[110]2.8397,[111]2.8578,[112]2.8328,[113]2.8199,[114]2.7992,[115]2.7843,[116]2.7698,[117]2.7482,[118]2.7275,[119]2.7064,[120]2.6881,[121]2.6734,[122]2.6562,[123]2.6392,[124]2.6209,[125]2.6041,[126]2.5874,[127]2.5740,[128]2.5650,[129]2.5535,[130]2.5403,[131]2.5311,[132]2.5374,[133]2.5470,[134]2.5539,[135]2.5645,[136]2.5795,[137]2.5931,[138]2.6010,[139]2.6117,[140]2.6123,[141]2.6142,[142]2.6130,[143]2.6143,[144]2.6119,[145]2.6040,[146]2.6025,[147]2.6072,[148]2.6072,[149]2.6088,[150]2.6037,[151]2.6020,[152]2.5995,[153]2.5956,[154]2.5956,[155]2.5999,[156]2.6014,[157]2.6067,[158]2.6150,[159]2.6172,[160]2.6265,[161]2.6347,[162]2.6448,[163]2.6492,[164]2.6696,[165]2.6929,[166]2.7101,[167]2.7218,[168]2.7453,[169]2.7678,[170]2.7894,[171]2.8113,[172]2.7959,[173]2.7801,[174]2.7666,[175]2.7552,[176]2.7436,[177]2.7320,[178]2.7195,[179]2.7066,[180]2.7101,[181]2.7245,[182]2.7393,[183]2.7539,[184]2.7673,[185]2.7776,[186]2.7936,[187]2.8089,[188]2.8233,[189]2.8342,[190]2.8351,[191]2.8425,[192]2.8457,[193]2.8508,[194]2.8699,[195]2.8784,[196]2.8913,[197]2.9010,[198]2.9059,[199]2.9117,[200]2.9111,[201]2.9259,[202]2.9213,[203]2.9270,[204]2.9302,[205]2.9297,[206]2.9326,[207]2.9412,[208]2.9508,[209]2.9597,[210]2.9604,[211]2.9557,[212]2.9561,[213]2.9636,[214]2.9655,[215]2.9709,[216]2.9716,[217]2.9673,[218]2.9673,[219]2.9682,[220]2.9683,[221]2.9689,[222]2.9690,[223]2.9691,[224]2.9737,[225]2.9755,[226]2.9680,[227]2.9658,[228]2.9675,[229]2.9713,[230]2.9773,[231]2.9834,[232]2.9758,[233]2.9687,[234]2.9685,[235]2.9668,[236]2.9753,[237]2.9836,[238]2.9929,[239]3.0028,[240]3.0120,[241]3.0232,[242]3.0379,[243]3.0503,[244]3.0585,[245]3.0702,[246]3.0808,[247]3.0796,[248]3.0754,[249]3.0734,[250]3.0675,[251]3.0655,[252]3.0677,[253]3.0718,[254]3.0790,[255]3.0855,[256]3.0890,[257]3.0915,[258]3.0927,[259]3.0964,[260]3.0987,[261]3.1000,[262]3.0991,[263]3.1047,[264]3.1072,[265]3.1079,[266]3.1095,[267]3.1113,[268]3.1145,[269]3.1173,[270]3.1163,[271]3.1147,[272]3.1084,[273]3.1080,[274]3.1011,[275]3.0904,[276]3.0793,[277]3.0812,[278]3.0911,[279]3.0973,[280]3.1049,[281]3.1121,[282]3.1179,[283]3.1240,[284]3.1302,[285]3.1435,[286]3.1456,[287]3.1488,[288]3.1540,[289]3.1560,[290]3.1480,[291]3.1395,[292]3.1371,[293]3.1359,[294]3.1333,[295]3.1311,[296]3.1328,[297]3.1335,[298]3.1388,[299]3.1447,[300]3.1474,[301]3.1517,[302]3.1536,[303]3.1550,[304]3.1546,[305]3.1661,[306]3.1730,[307]3.1836,[308]3.1729,[309]3.1675,[310]3.1583,[311]3.1607,[312]3.1624,[313]3.1680,[314]3.1704,[315]3.1735,[316]3.1749,[317]3.1767,[318]3.1771,[319]3.1771,[320]3.1812,[321]3.1816,[322]3.1835,[323]3.1896,[324]3.1904,[325]3.1957,[326]3.1999,[327]3.2036,[328]3.2058,[329]3.2078,[330]3.2141,[331]3.2171,[332]3.2210,[333]3.2202,[334]3.2205,[335]3.2212,[336]3.2213,[337]3.2225,[338]3.2227,[339]3.2253,[340]3.2289,[341]3.2341,[342]3.2428,[343]3.2517,[344]3.2569,[345]3.2484,[346]3.2405,[347]3.2354,[348]3.2282,[349]3.2243,[350]3.2229,[351]3.2274,[352]3.2418,[353]3.2506,[354]3.2630,[355]3.2712,[356]3.2767,[357]3.2881,[358]3.2977,[359]3.3005,[360]3.3067,[361]3.3162,[362]3.3246,[363]3.3303,[364]3.3371,[365]3.3426,[366]3.3527,[367]3.3613,[368]3.3678,[369]3.3754,[370]3.3842,[371]3.3974,[372]3.4064,[373]3.4098,[374]3.4130,[375]3.4179,[376]3.4301,[377]3.4412,[378]3.4442,[379]3.4440,[380]3.4407,[381]3.4455,[382]3.4513,[383]3.4546,[384]3.4588,[385]3.4627,[386]3.4688,[387]3.4744,[388]3.4774,[389]3.4675,[390]3.4587,[391]3.4486,[392]3.4433,[393]3.4341,[394]3.4256,[395]3.4167,[396]3.4071,[397]3.3985,[398]3.3894,[399]3.3794,[400]3.3711,[401]3.3614,[402]3.3515,[403]3.3434,[404]3.3336,[405]3.3244,[406]3.3149,[407]3.3058,[408]3.2972,[409]3.2888,[410]3.2830,[411]3.2839,[412]3.2794,[413]3.2811,[414]3.2828,[415]3.2799,[416]3.2799,[417]3.2821,[418]3.2767,[419]3.2778,[420]3.2752,[421]3.2738,[422]3.2743,[423]3.2736,[424]3.2771,[425]3.2768,[426]3.2773,[427]3.2766,[428]3.2791,[429]3.2805,[430]3.2830,[431]3.2838,[432]3.2831,[433]3.2794,[434]3.2796,[435]3.2722,[436]3.2665,[437]3.2625,[438]3.2609,[439]3.2579,[440]3.2627,[441]3.2680,[442]3.2753,[443]3.2732,[444]3.2742,[445]3.2752,[446]3.2792,[447]3.2825,[448]3.2848,[449]3.2878,[450]3.2916,[451]3.2947,[452]3.2968,[453]3.2982,[454]3.2969,[455]3.2993,[456]3.2997,[457]3.3022,[458]3.3073,[459]3.3077,[460]3.3079,[461]3.3048,[462]3.3084,[463]3.3156,[464]3.3208,[465]3.3144,[466]3.3124,[467]3.3104,[468]3.3117,[469]3.3091,[470]3.3065,[471]3.3070,[472]3.3078,[473]3.3071,[474]3.3061,[475]3.3071,[476]3.3057,[477]3.3050,[478]3.3057,[479]3.3075,[480]3.3100,[481]3.3063,[482]3.3098,[483]3.3091,[484]3.3127,[485]3.3189,[486]3.3221,[487]3.3255,[488]3.3309,[489]3.3334,[490]3.3384,[491]3.3444,[492]3.3489,[493]3.3486,[494]3.3498,[495]3.3522,[496]3.3540,[497]3.3568,[498]3.3572,[499]3.3569,[500]3.3608,[501]3.3654,[502]3.3644,[503]3.3631,[504]3.3651,[505]3.3682,[506]3.3761,[507]3.3791,[508]3.3826,[509]3.3753,[510]3.3699,[511]3.3635,[512]3.3592,[513]3.3533,[514]3.3518,[515]3.3536,[516]3.3488,[517]3.3487,[518]3.3473,[519]3.3476,[520]3.3515,[521]3.3505,[522]3.3490,[523]3.3545,[524]3.3535,[525]3.3520,[526]3.3473,[527]3.3423,[528]3.3391,[529]3.3361,[530]3.3332,[531]3.3303,[532]3.3249,[533]3.3190,[534]3.3145,[535]3.3149,[536]3.3173,[537]3.3203,[538]3.3224,[539]3.3250,[540]3.3303,[541]3.3334,[542]3.3357,[543]3.3302,[544]3.3259,[545]3.3256,[546]3.3193,[547]3.3131,[548]3.3067,[549]3.3000,[550]3.2943,[551]3.2882,[552]3.2827,[553]3.2773,[554]3.2754,[555]3.2737,[556]3.2764,[557]3.2803,[558]3.2863,[559]3.2908,[560]3.2961,[561]3.2942,
llama_print_timings:        load time =    2197.28 ms
llama_print_timings:      sample time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings: prompt eval time = 2802141.29 ms / 287232 tokens (    9.76 ms per token,   102.50 tokens per second)
llama_print_timings:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings:       total time = 2853371.87 ms / 287233 tokens

Final estimate: PPL = 3.2942 +/- 0.01812
```

---

👤 **ubergarm** commented the **2025-03-31** at **14:22:13**:<br>

> > I'll let it keep going and hopefully get your perplexity by tomorrow morning!

> Thanks!

Just grabbed the log, here is how your "pure" `iq4_k_r4` stacks up on full perplexity run and size:
| Model | Size (GiB) | PPL |
| --- | --- | --- |
| DeepSeek-V3-0324-IQ2_K_R4 | 227 | 3.5614 +/- 0.02001 |
| DeepSeek-V3-0324-PURE-IQ4_K_R4 | 353 | 3.2942 +/- 0.01812 |
| DeepSeek-V3-0324-IQ4_K_R4 | 387 | 3.2596 +/- 0.01786 | 
| DeepSeek-V3-0324-Q8_0 | 666 | 3.2454 +/- 0.01773 |

![Image](https://github.com/user-attachments/assets/cf36b5ea-a1ec-4267-a25e-a0c52ccabaef)

#### "PURE" `IQ4_K_R4` perplexity log details
```
main: build = 3613 (4819257c)
main: built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu
main: seed  = 1337

llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q5_0:   61 tensors
llama_model_loader: - type iq4_k:    1 tensors
llama_model_loader: - type iq4_k_r4:  724 tensors

llm_load_print_meta: model size       = 352.470 GiB (4.505 BPW) 

perplexity: calculating perplexity over 561 chunks, n_ctx=512, batch_size=2048, n_seq=4
perplexity: 19.63 seconds per pass - ETA 45.88 minutes
[1]2.4366,[2]3.1393,[3]2.3037,[4]1.9385,[5]1.7532,[6]1.6176,[7]1.5316,[8]1.4745,[9]1.4313,[10]1.3953,[11]1.3829,[12]1.4097,[13]1.4224,[14]1.5443,[15]1.6735,[16]1.7303,[17]1.8888,[18]2.0140,[19]1.9767,[20]1.9637,[21]2.0686,[22]2.0468,[23]2.0218,[24]2.0329,[25]2.0040,[26]1.9824,[27]2.0276,[28]2.0377,[29]2.0839,[30]2.1167,[31]2.1493,[32]2.1657,[33]2.2060,[34]2.2503,[35]2.2965,[36]2.3499,[37]2.3852,[38]2.4336,[39]2.4732,[40]2.5311,[41]2.5728,[42]2.5850,[43]2.6354,[44]2.6530,[45]2.7332,[46]2.7820,[47]2.7394,[48]2.6930,[49]2.6667,[50]2.6835,[51]2.7280,[52]2.7399,[53]2.7902,[54]2.8021,[55]2.8316,[56]2.8626,[57]2.8758,[58]2.9093,[59]2.9190,[60]2.9659,[61]3.0052,[62]3.0520,[63]3.0836,[64]3.1250,[65]3.1341,[66]3.1157,[67]3.0915,[68]3.1179,[69]3.1110,[70]3.1238,[71]3.1416,[72]3.1557,[73]3.1697,[74]3.1909,[75]3.1705,[76]3.1256,[77]3.0826,[78]3.0789,[79]3.0595,[80]3.0426,[81]3.0078,[82]3.0106,[83]2.9793,[84]2.9450,[85]2.9116,[86]2.8887,[87]2.8825,[88]2.8559,[89]2.8395,[90]2.8144,[91]2.7862,[92]2.7616,[93]2.7362,[94]2.7115,[95]2.6895,[96]2.6870,[97]2.6926,[98]2.6774,[99]2.6605,[100]2.6627,[101]2.6544,[102]2.6697,[103]2.6946,[104]2.7113,[105]2.7078,[106]2.7294,[107]2.7536,[108]2.7740,[109]2.8065,[110]2.8397,[111]2.8578,[112]2.8328,[113]2.8199,[114]2.7992,[115]2.7843,[116]2.7698,[117]2.7482,[118]2.7275,[119]2.7064,[120]2.6881,[121]2.6734,[122]2.6562,[123]2.6392,[124]2.6209,[125]2.6041,[126]2.5874,[127]2.5740,[128]2.5650,[129]2.5535,[130]2.5403,[131]2.5311,[132]2.5374,[133]2.5470,[134]2.5539,[135]2.5645,[136]2.5795,[137]2.5931,[138]2.6010,[139]2.6117,[140]2.6123,[141]2.6142,[142]2.6130,[143]2.6143,[144]2.6119,[145]2.6040,[146]2.6025,[147]2.6072,[148]2.6072,[149]2.6088,[150]2.6037,[151]2.6020,[152]2.5995,[153]2.5956,[154]2.5956,[155]2.5999,[156]2.6014,[157]2.6067,[158]2.6150,[159]2.6172,[160]2.6265,[161]2.6347,[162]2.6448,[163]2.6492,[164]2.6696,[165]2.6929,[166]2.7101,[167]2.7218,[168]2.7453,[169]2.7678,[170]2.7894,[171]2.8113,[172]2.7959,[173]2.7801,[174]2.7666,[175]2.7552,[176]2.7436,[177]2.7320,[178]2.7195,[179]2.7066,[180]2.7101,[181]2.7245,[182]2.7393,[183]2.7539,[184]2.7673,[185]2.7776,[186]2.7936,[187]2.8089,[188]2.8233,[189]2.8342,[190]2.8351,[191]2.8425,[192]2.8457,[193]2.8508,[194]2.8699,[195]2.8784,[196]2.8913,[197]2.9010,[198]2.9059,[199]2.9117,[200]2.9111,[201]2.9259,[202]2.9213,[203]2.9270,[204]2.9302,[205]2.9297,[206]2.9326,[207]2.9412,[208]2.9508,[209]2.9597,[210]2.9604,[211]2.9557,[212]2.9561,[213]2.9636,[214]2.9655,[215]2.9709,[216]2.9716,[217]2.9673,[218]2.9673,[219]2.9682,[220]2.9683,[221]2.9689,[222]2.9690,[223]2.9691,[224]2.9737,[225]2.9755,[226]2.9680,[227]2.9658,[228]2.9675,[229]2.9713,[230]2.9773,[231]2.9834,[232]2.9758,[233]2.9687,[234]2.9685,[235]2.9668,[236]2.9753,[237]2.9836,[238]2.9929,[239]3.0028,[240]3.0120,[241]3.0232,[242]3.0379,[243]3.0503,[244]3.0585,[245]3.0702,[246]3.0808,[247]3.0796,[248]3.0754,[249]3.0734,[250]3.0675,[251]3.0655,[252]3.0677,[253]3.0718,[254]3.0790,[255]3.0855,[256]3.0890,[257]3.0915,[258]3.0927,[259]3.0964,[260]3.0987,[261]3.1000,[262]3.0991,[263]3.1047,[264]3.1072,[265]3.1079,[266]3.1095,[267]3.1113,[268]3.1145,[269]3.1173,[270]3.1163,[271]3.1147,[272]3.1084,[273]3.1080,[274]3.1011,[275]3.0904,[276]3.0793,[277]3.0812,[278]3.0911,[279]3.0973,[280]3.1049,[281]3.1121,[282]3.1179,[283]3.1240,[284]3.1302,[285]3.1435,[286]3.1456,[287]3.1488,[288]3.1540,[289]3.1560,[290]3.1480,[291]3.1395,[292]3.1371,[293]3.1359,[294]3.1333,[295]3.1311,[296]3.1328,[297]3.1335,[298]3.1388,[299]3.1447,[300]3.1474,[301]3.1517,[302]3.1536,[303]3.1550,[304]3.1546,[305]3.1661,[306]3.1730,[307]3.1836,[308]3.1729,[309]3.1675,[310]3.1583,[311]3.1607,[312]3.1624,[313]3.1680,[314]3.1704,[315]3.1735,[316]3.1749,[317]3.1767,[318]3.1771,[319]3.1771,[320]3.1812,[321]3.1816,[322]3.1835,[323]3.1896,[324]3.1904,[325]3.1957,[326]3.1999,[327]3.2036,[328]3.2058,[329]3.2078,[330]3.2141,[331]3.2171,[332]3.2210,[333]3.2202,[334]3.2205,[335]3.2212,[336]3.2213,[337]3.2225,[338]3.2227,[339]3.2253,[340]3.2289,[341]3.2341,[342]3.2428,[343]3.2517,[344]3.2569,[345]3.2484,[346]3.2405,[347]3.2354,[348]3.2282,[349]3.2243,[350]3.2229,[351]3.2274,[352]3.2418,[353]3.2506,[354]3.2630,[355]3.2712,[356]3.2767,[357]3.2881,[358]3.2977,[359]3.3005,[360]3.3067,[361]3.3162,[362]3.3246,[363]3.3303,[364]3.3371,[365]3.3426,[366]3.3527,[367]3.3613,[368]3.3678,[369]3.3754,[370]3.3842,[371]3.3974,[372]3.4064,[373]3.4098,[374]3.4130,[375]3.4179,[376]3.4301,[377]3.4412,[378]3.4442,[379]3.4440,[380]3.4407,[381]3.4455,[382]3.4513,[383]3.4546,[384]3.4588,[385]3.4627,[386]3.4688,[387]3.4744,[388]3.4774,[389]3.4675,[390]3.4587,[391]3.4486,[392]3.4433,[393]3.4341,[394]3.4256,[395]3.4167,[396]3.4071,[397]3.3985,[398]3.3894,[399]3.3794,[400]3.3711,[401]3.3614,[402]3.3515,[403]3.3434,[404]3.3336,[405]3.3244,[406]3.3149,[407]3.3058,[408]3.2972,[409]3.2888,[410]3.2830,[411]3.2839,[412]3.2794,[413]3.2811,[414]3.2828,[415]3.2799,[416]3.2799,[417]3.2821,[418]3.2767,[419]3.2778,[420]3.2752,[421]3.2738,[422]3.2743,[423]3.2736,[424]3.2771,[425]3.2768,[426]3.2773,[427]3.2766,[428]3.2791,[429]3.2805,[430]3.2830,[431]3.2838,[432]3.2831,[433]3.2794,[434]3.2796,[435]3.2722,[436]3.2665,[437]3.2625,[438]3.2609,[439]3.2579,[440]3.2627,[441]3.2680,[442]3.2753,[443]3.2732,[444]3.2742,[445]3.2752,[446]3.2792,[447]3.2825,[448]3.2848,[449]3.2878,[450]3.2916,[451]3.2947,[452]3.2968,[453]3.2982,[454]3.2969,[455]3.2993,[456]3.2997,[457]3.3022,[458]3.3073,[459]3.3077,[460]3.3079,[461]3.3048,[462]3.3084,[463]3.3156,[464]3.3208,[465]3.3144,[466]3.3124,[467]3.3104,[468]3.3117,[469]3.3091,[470]3.3065,[471]3.3070,[472]3.3078,[473]3.3071,[474]3.3061,[475]3.3071,[476]3.3057,[477]3.3050,[478]3.3057,[479]3.3075,[480]3.3100,[481]3.3063,[482]3.3098,[483]3.3091,[484]3.3127,[485]3.3189,[486]3.3221,[487]3.3255,[488]3.3309,[489]3.3334,[490]3.3384,[491]3.3444,[492]3.3489,[493]3.3486,[494]3.3498,[495]3.3522,[496]3.3540,[497]3.3568,[498]3.3572,[499]3.3569,[500]3.3608,[501]3.3654,[502]3.3644,[503]3.3631,[504]3.3651,[505]3.3682,[506]3.3761,[507]3.3791,[508]3.3826,[509]3.3753,[510]3.3699,[511]3.3635,[512]3.3592,[513]3.3533,[514]3.3518,[515]3.3536,[516]3.3488,[517]3.3487,[518]3.3473,[519]3.3476,[520]3.3515,[521]3.3505,[522]3.3490,[523]3.3545,[524]3.3535,[525]3.3520,[526]3.3473,[527]3.3423,[528]3.3391,[529]3.3361,[530]3.3332,[531]3.3303,[532]3.3249,[533]3.3190,[534]3.3145,[535]3.3149,[536]3.3173,[537]3.3203,[538]3.3224,[539]3.3250,[540]3.3303,[541]3.3334,[542]3.3357,[543]3.3302,[544]3.3259,[545]3.3256,[546]3.3193,[547]3.3131,[548]3.3067,[549]3.3000,[550]3.2943,[551]3.2882,[552]3.2827,[553]3.2773,[554]3.2754,[555]3.2737,[556]3.2764,[557]3.2803,[558]3.2863,[559]3.2908,[560]3.2961,[561]3.2942,
llama_print_timings:        load time =    2197.28 ms
llama_print_timings:      sample time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings: prompt eval time = 2802141.29 ms / 287232 tokens (    9.76 ms per token,   102.50 tokens per second)
llama_print_timings:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings:       total time = 2853371.87 ms / 287233 tokens

Final estimate: PPL = 3.2942 +/- 0.01812
```

---

👤 **ikawrakow** commented the **2025-03-31** at **14:52:10**:<br>

`3.2942` is 1.5% higher than `Q8_0`, so not too bad. I think with `IQ5_K` for the attention tensors and shared experts it should be (almost) on par with the result obtained with `Q8_0` for these. 

I'm somewhat surprised that the PP speed of the pure `IQ4_K` is better than the `IQ4_K` mix by almost 15%. Is it so that you used `Q8_0`, and not `Q8_0_R8` for the mix, because there was the issue with the NaN/very high PPL due to row-interleaved quants being used for token embeddings?

---

👤 **ubergarm** commented the **2025-03-31** at **15:56:26**:<br>

> 3.2942 is 1.5% higher than Q8_0, so not too bad. I think with IQ5_K for the attention tensors and shared experts it should be (almost) on par with the result obtained with Q8_0 for these.

Nice, getting it dialed in. I don't think @saood06 tried that exact combo in his mixes yet.

> I'm somewhat surprised that the PP speed of the pure IQ4_K is better than the IQ4_K mix by almost 15%. Is it so that you used Q8_0, and not Q8_0_R8 for the mix, because there was the issue with the NaN/very high PPL due to row-interleaved quants being used for token embeddings?

Right, the "non pure" `IQ4_K_R4` here has `Q8_0`s for attention/embeds/dense/shared expert layers as well as `IQ5_K_R4` for routed experted down projections. I just didn't specify `-rtr` on the perplexity script is all. That nan issue has been fixed in the branch I was using.

So the duration is not a fair comparison given the "pure" was using repacked quants while the "non pure" and full `q8_0` were *not* repacked.

Maybe I'll follow up later with proper llama-bench comparisons after getting the mixes dialed in for perplexity.

Can close this issue now then as the original question has been answered.

Thanks!

---

👤 **ubergarm** commented the **2025-03-31** at **19:52:27**:<br>

> Maybe I'll follow up later with proper llama-bench comparisons 

> I'm somewhat surprised that the PP speed of the pure IQ4_K is better than the IQ4_K mix by almost 15%

@ikawrakow

I did a quick llama-bench comparison between the `PURE-IQ4_K_R4` and the `q8_0`/mix `IQ4_K_R4` (using -rtr 1 for `q8_0_r8` this time) on the CPU only the Xeon 6980P with 88 threads and found the results interesting. The graph shows the "pure" version as baseline 100%.

I believe this is basically the same as @saood06 's pure version rolled last night vs his earlier working mix mentioned above.

![Image](https://github.com/user-attachments/assets/08dc4b2f-86be-43f5-8bc8-da16eacee582)

<details>

<summary>Command details and raw data</summary>

## Common Setup
```bash
echo Setting power profile to performance:
powerprofilesctl set performance

echo Set numa balancing to be off:
echo 0 | sudo tee /proc/sys/kernel/numa_balancing

echo Maximizing chances of loading model into THPs
echo always | sudo tee -a /sys/kernel/mm/transparent_hugepage/enabled
echo always | sudo tee -a /sys/kernel/mm/transparent_hugepage/defrag

echo Dropping all caches... (to hopefully use more THPs)
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
```

## `IQ4_K_R4`
```bash
numactl -N 0 -m 0 \
./build/bin/llama-bench \
    -rtr 1 \
    -thp 0 \
    --mmap 0 \
    --model /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-IQ4_K_R4.gguf \
    -ctk q8_0 \
    -mla 3 -fa 1 \
    -amb 1024 \
    -fmoe 1 \
    -p 512,8192,16384 -n 0 \
    -gp 512,64 \
    -gp 8192,64 \
    -gp 16384,64 \
    -r 2 \
    --numa numactl \
    --threads 88

## note all q8_0 get repacked with `-rtr 1` to be `q8_r_8` including `attn_k_b.weight` presumably
llama_model_loader: - type q8_0:  612 tensors
llama_model_loader: - type iq4_k_r4:  116 tensors
llama_model_loader: - type iq5_k_r4:   58 tensors

## Confirm fully loaded into THPs
$ grep Huge /proc/meminfo
AnonHugePages:  41615360 kB
ShmemHugePages:        0 kB
FileHugePages:         0 kB
HugePages_Total:       0
HugePages_Free:        0
HugePages_Rsvd:        0
HugePages_Surp:        0
Hugepagesize:       2048 kB
Hugetlb:               0 kB

$ du /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-IQ4_K_R4.gguf
404947028       /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-IQ4_K_R4.gguf
```

| model                          |       size |     params | backend    | threads | type_k | fa | mla |   amb | mmap | rtr | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -----: | -: | --: | ----: | ---: | --: | ---: | ------------: | ---------------: |
============ Repacked 611 tensors
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 386.18 GiB |   672.05 B | CPU        |      88 |   q8_0 |  1 |   3 |  1024 |    0 |   1 |    1 |         pp512 |    122.55 ± 3.11 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 386.18 GiB |   672.05 B | CPU        |      88 |   q8_0 |  1 |   3 |  1024 |    0 |   1 |    1 |        pp8192 |     74.34 ± 2.11 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 386.18 GiB |   672.05 B | CPU        |      88 |   q8_0 |  1 |   3 |  1024 |    0 |   1 |    1 |       pp16384 |     52.68 ± 0.21 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 386.18 GiB |   672.05 B | CPU        |      88 |   q8_0 |  1 |   3 |  1024 |    0 |   1 |    1 |    tg64@pp512 |      8.20 ± 0.00 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 386.18 GiB |   672.05 B | CPU        |      88 |   q8_0 |  1 |   3 |  1024 |    0 |   1 |    1 |   tg64@pp8192 |      6.70 ± 0.00 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 386.18 GiB |   672.05 B | CPU        |      88 |   q8_0 |  1 |   3 |  1024 |    0 |   1 |    1 |  tg64@pp16384 |      5.52 ± 0.00 |

`build: 4819257c (3613)`

## `PURE-IQ4_K_R4`
```bash
numactl -N 0 -m 0 \
./build/bin/llama-bench \
    -thp 0 \
    --mmap 0 \
    --model /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-PURE-IQ4_K_R4.gguf \
    -ctk q8_0 \
    -mla 3 -fa 1 \
    -amb 1024 \
    -fmoe 1 \
    -p 512,8192,16384 -n 0 \
    -gp 512,64 \
    -gp 8192,64 \
    -gp 16384,64 \
    -r 2 \
    --numa numactl \
    --threads 88

## note the q5_0 attn_k_b.weight so not totally "pure" hah...
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q5_0:   61 tensors
llama_model_loader: - type iq4_k:    1 tensors
llama_model_loader: - type iq4_k_r4:  724 tensors

## Confirm fully loaded into THPs
$ grep Huge /proc/meminfo
AnonHugePages:  372733952 kB
ShmemHugePages:        0 kB
FileHugePages:         0 kB
HugePages_Total:       0
HugePages_Free:        0
HugePages_Rsvd:        0
HugePages_Surp:        0
Hugepagesize:       2048 kB
Hugetlb:               0 kB

$ du /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-PURE-IQ4_K_R4.gguf
369596400       /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-PURE-IQ4_K_R4.gguf
```

| model                          |       size |     params | backend    | threads | type_k | fa | mla |   amb | mmap | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -----: | -: | --: | ----: | ---: | ---: | ------------: | ---------------: |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 352.47 GiB |   672.05 B | CPU        |      88 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |         pp512 |    112.83 ± 0.69 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 352.47 GiB |   672.05 B | CPU        |      88 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |        pp8192 |     63.66 ± 0.00 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 352.47 GiB |   672.05 B | CPU        |      88 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |       pp16384 |     47.50 ± 0.15 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 352.47 GiB |   672.05 B | CPU        |      88 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |    tg64@pp512 |      8.50 ± 0.00 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 352.47 GiB |   672.05 B | CPU        |      88 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |   tg64@pp8192 |      7.13 ± 0.02 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 352.47 GiB |   672.05 B | CPU        |      88 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |  tg64@pp16384 |      5.48 ± 0.02 |

`build: 4819257c (3613)`

</details>

> attn_k_b.weight can't be k-, i-, or iqk-quant because its row size is 128, so not a multiple of 256 as needed by i-, k-, idk-quants. Normally this should be caught and a corresponding legacy quant with a block size of 32 should be used instead.

I'm still wondering a bit about that `attn_k_b.weight` error `128 x 65536 are not divisible by 256` which falls back to `q4_0` or `q5_0` etc.  However it seems that `q8_0_r8` is okay?

```
[  52/1147]                blk.3.attn_k_b.weight - [  128, 65536,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk$
3.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.3.attn_k_b.weight
converting to q8_0_r8 .. size =    16.00 MiB ->     8.50 MiB
```

So wondering if I do a mostly `iq5_k_r4` attention/shared experts, should I let the `attn_k_b.weight` fall back to `q5_0` or set them up to `q8_0_r8` (assuming CPU inference).

Anyway, learning a lot as usual, gonna close this one as solved. Cheers!

---

👤 **ubergarm** commented the **2025-03-31** at **19:52:27**:<br>

> Maybe I'll follow up later with proper llama-bench comparisons 

> I'm somewhat surprised that the PP speed of the pure IQ4_K is better than the IQ4_K mix by almost 15%

@ikawrakow

I did a quick llama-bench comparison between the `PURE-IQ4_K_R4` and the `q8_0`/mix `IQ4_K_R4` (using -rtr 1 for `q8_0_r8` this time) on the CPU only the Xeon 6980P with 88 threads and found the results interesting. The graph shows the "pure" version as baseline 100%.

I believe this is basically the same as @saood06 's pure version rolled last night vs his earlier working mix mentioned above.

![Image](https://github.com/user-attachments/assets/08dc4b2f-86be-43f5-8bc8-da16eacee582)

<details>

<summary>Command details and raw data</summary>

## Common Setup
```bash
echo Setting power profile to performance:
powerprofilesctl set performance

echo Set numa balancing to be off:
echo 0 | sudo tee /proc/sys/kernel/numa_balancing

echo Maximizing chances of loading model into THPs
echo always | sudo tee -a /sys/kernel/mm/transparent_hugepage/enabled
echo always | sudo tee -a /sys/kernel/mm/transparent_hugepage/defrag

echo Dropping all caches... (to hopefully use more THPs)
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
```

## `IQ4_K_R4`
```bash
numactl -N 0 -m 0 \
./build/bin/llama-bench \
    -rtr 1 \
    -thp 0 \
    --mmap 0 \
    --model /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-IQ4_K_R4.gguf \
    -ctk q8_0 \
    -mla 3 -fa 1 \
    -amb 1024 \
    -fmoe 1 \
    -p 512,8192,16384 -n 0 \
    -gp 512,64 \
    -gp 8192,64 \
    -gp 16384,64 \
    -r 2 \
    --numa numactl \
    --threads 88

## Confirm fully loaded into THPs
$ grep Huge /proc/meminfo
AnonHugePages:  41615360 kB
ShmemHugePages:        0 kB
FileHugePages:         0 kB
HugePages_Total:       0
HugePages_Free:        0
HugePages_Rsvd:        0
HugePages_Surp:        0
Hugepagesize:       2048 kB
Hugetlb:               0 kB

$ du /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-IQ4_K_R4.gguf
404947028       /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-IQ4_K_R4.gguf
```

| model                          |       size |     params | backend    | threads | type_k | fa | mla |   amb | mmap | rtr | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -----: | -: | --: | ----: | ---: | --: | ---: | ------------: | ---------------: |
============ Repacked 611 tensors
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 386.18 GiB |   672.05 B | CPU        |      88 |   q8_0 |  1 |   3 |  1024 |    0 |   1 |    1 |         pp512 |    122.55 ± 3.11 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 386.18 GiB |   672.05 B | CPU        |      88 |   q8_0 |  1 |   3 |  1024 |    0 |   1 |    1 |        pp8192 |     74.34 ± 2.11 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 386.18 GiB |   672.05 B | CPU        |      88 |   q8_0 |  1 |   3 |  1024 |    0 |   1 |    1 |       pp16384 |     52.68 ± 0.21 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 386.18 GiB |   672.05 B | CPU        |      88 |   q8_0 |  1 |   3 |  1024 |    0 |   1 |    1 |    tg64@pp512 |      8.20 ± 0.00 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 386.18 GiB |   672.05 B | CPU        |      88 |   q8_0 |  1 |   3 |  1024 |    0 |   1 |    1 |   tg64@pp8192 |      6.70 ± 0.00 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 386.18 GiB |   672.05 B | CPU        |      88 |   q8_0 |  1 |   3 |  1024 |    0 |   1 |    1 |  tg64@pp16384 |      5.52 ± 0.00 |

`build: 4819257c (3613)`

## `PURE-IQ4_K_R4`
```bash
numactl -N 0 -m 0 \
./build/bin/llama-bench \
    -thp 0 \
    --mmap 0 \
    --model /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-PURE-IQ4_K_R4.gguf \
    -ctk q8_0 \
    -mla 3 -fa 1 \
    -amb 1024 \
    -fmoe 1 \
    -p 512,8192,16384 -n 0 \
    -gp 512,64 \
    -gp 8192,64 \
    -gp 16384,64 \
    -r 2 \
    --numa numactl \
    --threads 88

## Confirm fully loaded into THPs
$ grep Huge /proc/meminfo
AnonHugePages:  372733952 kB
ShmemHugePages:        0 kB
FileHugePages:         0 kB
HugePages_Total:       0
HugePages_Free:        0
HugePages_Rsvd:        0
HugePages_Surp:        0
Hugepagesize:       2048 kB
Hugetlb:               0 kB

$ du /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-PURE-IQ4_K_R4.gguf
369596400       /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-PURE-IQ4_K_R4.gguf
```

| model                          |       size |     params | backend    | threads | type_k | fa | mla |   amb | mmap | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -----: | -: | --: | ----: | ---: | ---: | ------------: | ---------------: |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 352.47 GiB |   672.05 B | CPU        |      88 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |         pp512 |    112.83 ± 0.69 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 352.47 GiB |   672.05 B | CPU        |      88 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |        pp8192 |     63.66 ± 0.00 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 352.47 GiB |   672.05 B | CPU        |      88 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |       pp16384 |     47.50 ± 0.15 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 352.47 GiB |   672.05 B | CPU        |      88 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |    tg64@pp512 |      8.50 ± 0.00 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 352.47 GiB |   672.05 B | CPU        |      88 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |   tg64@pp8192 |      7.13 ± 0.02 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 352.47 GiB |   672.05 B | CPU        |      88 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |  tg64@pp16384 |      5.48 ± 0.02 |

`build: 4819257c (3613)`

</details>

> attn_k_b.weight can't be k-, i-, or iqk-quant because its row size is 128, so not a multiple of 256 as needed by i-, k-, idk-quants. Normally this should be caught and a corresponding legacy quant with a block size of 32 should be used instead.

I'm still wondering a bit about that `attn_k_b.weight` error `128 x 65536 are not divisible by 256` which falls back to `q4_0` or `q5_0` etc.  However it seems that `q8_0_r8` is okay?

```
[  52/1147]                blk.3.attn_k_b.weight - [  128, 65536,     1,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk$
3.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.3.attn_k_b.weight
converting to q8_0_r8 .. size =    16.00 MiB ->     8.50 MiB
```

So wondering if I do a mostly `iq5_k_r4` attention/shared experts, should I let the `attn_k_b.weight` fall back to `q5_0` or set them up to `q8_0_r8` (assuming CPU inference).

Anyway, learning a lot as usual, gonna close this one as solved. Cheers!

---

👤 **saood06** commented the **2025-04-01** at **01:02:46**:<br>

> Just grabbed the log, here is how your "pure" `iq4_k_r4` stacks up on full perplexity run , size, and duration:
> Model 	Size (GiB) 	PPL 	Duration (minutes)
> DeepSeek-V3-0324-IQ2_K_R4 	227 	3.5614 +/- 0.02001 	(different rig)
> DeepSeek-V3-0324-PURE-IQ4_K_R4 	353 	3.2942 +/- 0.01812 	47.56
> DeepSeek-V3-0324-IQ4_K_R4 	387 	3.2596 +/- 0.01786 	55.01
> DeepSeek-V3-0324-Q8_0 	666 	3.2454 +/- 0.01773 	68.87
> 
> ![Image](https://github.com/user-attachments/assets/cf36b5ea-a1ec-4267-a25e-a0c52ccabaef)
> 
> In terms of speed to calculate perplexity, these three were similar setups more or less using a single socket of the Xeon 6980P

Thanks, it looks like an acceptable loss in quality for me if it performs fast (wasn't able to make the quant overnight, the quant is cooking now)


> `3.2942` is 1.5% higher than `Q8_0`, so not too bad. 

I agree.

>I think with `IQ5_K` for the attention tensors and shared experts it should be (almost) on par with the result obtained with `Q8_0` for these.

It might be, but I probably won't test it as doing full ppl runs takes me way too long, and I think I'll be happy with my "pure" IQ4_K_R4 as that should still be faster, even if it is a bit lower quality.


> I did a quick llama-bench comparison between the `PURE-IQ4_K_R4` and the `q8_0`/mix `IQ4_K_R4` (using -rtr 1 for `q8_0_r8` this time) on the CPU only the Xeon 6980P with 88 threads and found the results interesting. The graph shows the "pure" version as baseline 100%.
> 
> ![Image](https://github.com/user-attachments/assets/08dc4b2f-86be-43f5-8bc8-da16eacee582)

I'm really surprised that the PURE gains a bit more TG lead at a depth of 8K, but then ends up behind at 16K. This is different from what I've seen when testing. It would be interesting to see the sweep bench and when they actually intersect and how the curves actually look because on my system I've tested up to that depth and the pure still wins out in TG (and it seems like it will always stay ahead with the lead gaining like you saw initially), so I'm curious as to why it ends up losing at higher depths for you.


> > attn_k_b.weight can't be k-, i-, or iqk-quant because its row size is 128, so not a multiple of 256 as needed by i-, k-, idk-quants. Normally this should be caught and a corresponding legacy quant with a block size of 32 should be used instead.
> 
> I'm still wondering a bit about that `attn_k_b.weight` error `128 x 65536 are not divisible by 256` which falls back to `q4_0` or `q5_0` etc. However it seems that `q8_0_r8` is okay?

Yes. `q8_0_r8` is not an  i-, k-, or iqk-quants. 
 

> So wondering if I do a mostly `iq5_k_r4` attention/shared experts, should I let the `attn_k_b.weight` fall back to `q5_0` or set them up to `q8_0_r8` (assuming CPU inference).

Both work and will have tradeoffs. I think `q5_0` is fine, but other people think that tensor is more sensitive and should be set higher when you can.

---

👤 **ikawrakow** commented the **2025-04-01** at **08:20:43**:<br>

>> I'm still wondering a bit about that attn_k_b.weight error 128 x 65536 are not divisible by 256 which falls back to q4_0 or q5_0 etc. However it seems that q8_0_r8 is okay?
>
> Both work and will have tradeoffs. I think q5_0 is fine, but other people think that tensor is more sensitive and should be set higher when you can.

Note that `Q5_0` quantization was improved in #295, so it should be fine now. But if in doubt, you can use `Q6_0`, which is basically on par with `Q6_K` after PR #295. For CPU-only you can use `q5_0_r4` or `q6_0_r4`. 

> It might be, but I probably won't test it as doing full ppl runs takes me way too long, and I think I'll be happy with my "pure" IQ4_K_R4 as that should still be faster, even if it is a bit lower quality.

Fair enough.

But if you get the urge to experiment and you are content with slight accuracy loss, you may consider `IQ4_KS`. Here is a performance comparison between pure `IQ4_K` and pure `IQ4_KS` for DeepSeek-Lite on my Ryzen-7950X CPU:

| model                |       size | fa | mla | rtr | fmoe |          test |              t/s |
| -------------------- | ---------: | -: | --: | --: | ---: | ------------: | ---------------: |
| deepseek2 16B IQ4_KS |   8.15 GiB |  1 |   3 |   1 |    1 |         pp512 |    700.85 ± 2.43 |
| deepseek2 16B IQ4_KS |   8.15 GiB |  1 |   3 |   1 |    1 |   tg128@pp512 |     34.41 ± 0.00 |
| deepseek2 16B IQ4_KS |   8.15 GiB |  1 |   3 |   1 |    1 |  tg128@pp4096 |     31.93 ± 0.01 |
| deepseek2 16B IQ4_KS |   8.15 GiB |  1 |   3 |   1 |    1 | tg128@pp16384 |     25.78 ± 0.00 |
| deepseek2 16B IQ4_K  |   9.00 GiB |  1 |   3 |   1 |    1 |         pp512 |    659.06 ± 2.14 |
| deepseek2 16B IQ4_K  |   9.00 GiB |  1 |   3 |   1 |    1 |   tg128@pp512 |     32.04 ± 0.06 |
| deepseek2 16B IQ4_K  |   9.00 GiB |  1 |   3 |   1 |    1 |  tg128@pp4096 |     29.66 ± 0.02 |
| deepseek2 16B IQ4_K  |   9.00 GiB |  1 |   3 |   1 |    1 | tg128@pp16384 |     23.74 ± 0.00 |
 
For DeepSeek-Lite we have `PPL(bf16) = 6.767`, `PPL(pure IQ4_K) = 6.821` (so +0.80%), and `PPL(pure IQ4_KS) = 6.858` (so, +1.34%).

---

👤 **ubergarm** commented the **2025-04-01** at **15:22:03**:<br>

> > UPDATE Wow!! 3.2596 +/- 0.01786 for this DeepSeek-V3-0324-IQ4_K_R4.gguf quant vs full Q8_0 at 3.2454 +/- 0.01773 in almost half the size!
>
> Amazing! You should publish this model.

Okay, I have two published `ik_llama.cpp` exclusive quants up on [huggingface ubergarm/DeepSeek-V3-0324-GGUF](https://huggingface.co/ubergarm/DeepSeek-V3-0324-GGUF) repo with hopefully enough of a quick start to get people curious enough to try this fork!

> Note that Q5_0 quantization was improved in https://github.com/ikawrakow/ik_llama.cpp/pull/295, so it should be fine now. But if in doubt, you can use Q6_0, which is basically on par with Q6_K after PR https://github.com/ikawrakow/ik_llama.cpp/pull/295. For CPU-only you can use q5_0_r4 or q6_0_r4

Ahh great, I didn't realize there was a `q5_0_r4`/`q6_0_r4` which is exactly what I was looking for to keep that tensor optimized. So if I re-made the "pure" benchmarked above it could be optimized using the `_r4` for possibly a bit more speed which may be related to:

> I'm really surprised that the PURE gains a bit more TG lead at a depth of 8K, but then ends up behind at 16K. This is different from what I've seen when testing. It would be interesting to see the sweep bench and when they actually intersect and how the curves actually look...

Yeah I was surprised about that too, I still need to dial in how many threads for tg vs pp too as it pp scales up and actually seems to improve with more threads. I'm out tomorrow but would like to finally get a good llama-sweep-bench going, I should have enough info to run it and get a curve. Thanks!

---

👤 **saood06** commented the **2025-04-01** at **21:39:19**:<br>

> Fair enough.
> 
> But if you get the urge to experiment and you are content with slight accuracy loss, you may consider `IQ4_KS`. Here is a performance comparison between pure `IQ4_K` and pure `IQ4_KS` for DeepSeek-Lite on my Ryzen-7950X CPU:

| model                |       size | fa | mla | rtr | fmoe |          test |              t/s |
| -------------------- | ---------: | -: | --: | --: | ---: | ------------: | ---------------: |
| deepseek2 16B IQ4_KS |   8.15 GiB |  1 |   3 |   1 |    1 |         pp512 |    700.85 ± 2.43 |
| deepseek2 16B IQ4_KS |   8.15 GiB |  1 |   3 |   1 |    1 |   tg128@pp512 |     34.41 ± 0.00 |
| deepseek2 16B IQ4_KS |   8.15 GiB |  1 |   3 |   1 |    1 |  tg128@pp4096 |     31.93 ± 0.01 |
| deepseek2 16B IQ4_KS |   8.15 GiB |  1 |   3 |   1 |    1 | tg128@pp16384 |     25.78 ± 0.00 |
| deepseek2 16B IQ4_K  |   9.00 GiB |  1 |   3 |   1 |    1 |         pp512 |    659.06 ± 2.14 |
| deepseek2 16B IQ4_K  |   9.00 GiB |  1 |   3 |   1 |    1 |   tg128@pp512 |     32.04 ± 0.06 |
| deepseek2 16B IQ4_K  |   9.00 GiB |  1 |   3 |   1 |    1 |  tg128@pp4096 |     29.66 ± 0.02 |
| deepseek2 16B IQ4_K  |   9.00 GiB |  1 |   3 |   1 |    1 | tg128@pp16384 |     23.74 ± 0.00 |
> 
> For DeepSeek-Lite we have `PPL(bf16) = 6.767`, `PPL(pure IQ4_K) = 6.821` (so +0.80%), and `PPL(pure IQ4_KS) = 6.858` (so, +1.34%).

This on the other hand does tempt me. I like my IQ4_K_R4 but trading off more quality for speed is still tempting.



> Ahh great, I didn't realize there was a `q5_0_r4`/`q6_0_r4` which is exactly what I was looking for to keep that tensor optimized. So if I re-made the "pure" benchmarked above it could be optimized using the `_r4` for possibly a bit more speed

I forgot about it as well, since I just let the fallback handle that tensor.

> Yeah I was surprised about that too, I still need to dial in how many threads for tg vs pp too as it pp scales up and actually seems to improve with more threads. I'm out tomorrow but would like to finally get a good llama-sweep-bench going, I should have enough info to run it and get a curve. Thanks!

If you do it would be interesting to see (also I haven't tested it, but setting -tb in sweep-bench should work and allow you to run different thread counts for TG and PP just like you can for the other examples like server and main).

My "pure" IQ4_K_R4  finished and the preliminary sweep bench results were really good (didn't benchmark very far as I wanted to inference with it, and just wanted to confirm it was loaded in and fast). I'll post a sweep bench graph out to 16K comparing it to some of my old results later.

---

👤 **saood06** commented the **2025-04-01** at **21:39:19**:<br>

> Fair enough.
> 
> But if you get the urge to experiment and you are content with slight accuracy loss, you may consider `IQ4_KS`. Here is a performance comparison between pure `IQ4_K` and pure `IQ4_KS` for DeepSeek-Lite on my Ryzen-7950X CPU:
> model 	size 	fa 	mla 	rtr 	fmoe 	test 	t/s
> deepseek2 16B IQ4_KS 	8.15 GiB 	1 	3 	1 	1 	pp512 	700.85 ± 2.43
> deepseek2 16B IQ4_KS 	8.15 GiB 	1 	3 	1 	1 	tg128@pp512 	34.41 ± 0.00
> deepseek2 16B IQ4_KS 	8.15 GiB 	1 	3 	1 	1 	tg128@pp4096 	31.93 ± 0.01
> deepseek2 16B IQ4_KS 	8.15 GiB 	1 	3 	1 	1 	tg128@pp16384 	25.78 ± 0.00
> deepseek2 16B IQ4_K 	9.00 GiB 	1 	3 	1 	1 	pp512 	659.06 ± 2.14
> deepseek2 16B IQ4_K 	9.00 GiB 	1 	3 	1 	1 	tg128@pp512 	32.04 ± 0.06
> deepseek2 16B IQ4_K 	9.00 GiB 	1 	3 	1 	1 	tg128@pp4096 	29.66 ± 0.02
> deepseek2 16B IQ4_K 	9.00 GiB 	1 	3 	1 	1 	tg128@pp16384 	23.74 ± 0.00
> 
> For DeepSeek-Lite we have `PPL(bf16) = 6.767`, `PPL(pure IQ4_K) = 6.821` (so +0.80%), and `PPL(pure IQ4_KS) = 6.858` (so, +1.34%).

This on the other hand does tempt me. I like my IQ4_K_R4 but trading off more quality for speed is still tempting.



> Ahh great, I didn't realize there was a `q5_0_r4`/`q6_0_r4` which is exactly what I was looking for to keep that tensor optimized. So if I re-made the "pure" benchmarked above it could be optimized using the `_r4` for possibly a bit more speed

I forgot about it as well, since I just let the fallback handle that tensor.

> Yeah I was surprised about that too, I still need to dial in how many threads for tg vs pp too as it pp scales up and actually seems to improve with more threads. I'm out tomorrow but would like to finally get a good llama-sweep-bench going, I should have enough info to run it and get a curve. Thanks!

If you do it would be interesting to see.

My "pure" IQ4_K_R4  finished and the preliminary sweep bench results were really good (didn't benchmark very far as I wanted to inference with it, and just wanted to confirm it was loaded in and fast). I'll post a sweep bench graph out to 16K comparing it to some of my old results later.

---

👤 **saood06** commented the **2025-04-03** at **03:10:35**:<br>

Here's the full graph comparing my currently used fast quants for R1 and V3. The mixes for both are similar. I'm going to go back and test #287 next with more configurations to see if I can find one that gives me more performance.

![Image](https://github.com/user-attachments/assets/43f4fd30-8d4a-4a96-8ced-854c4f502bfb)

![Image](https://github.com/user-attachments/assets/139a26a5-56b2-4489-bb66-6a512c5bda53)

Not included in the graph, but looking at other tests I ran #259 does seem to have an impact on performance on my system since I had a very similar quant mix with and without those tensors and they performed slightly differently.

---

👤 **saood06** commented the **2025-04-03** at **03:10:35**:<br>

Here's the full graph comparing my fast quants for both R1 and V3. The mixes for both are similar. I'm going to go back and test #287 next with more configurations to see if I can find one that works for it.

![Image](https://github.com/user-attachments/assets/43f4fd30-8d4a-4a96-8ced-854c4f502bfb)

![Image](https://github.com/user-attachments/assets/139a26a5-56b2-4489-bb66-6a512c5bda53)

Not included in the graph, but looking at other tests I ran #259 does seem to have an impact on performance on my system since I had a very similar quant mix with and without those tensors and they performed slightly differently.

---

👤 **saood06** commented the **2025-04-04** at **13:59:03**:<br>

Finally tested batch performance but this is at depth of 0, I'll test deeper depths later.

![batch_throughput](https://github.com/user-attachments/assets/17432dc5-5d14-41a8-870f-00e3540c317d)

12 is the highest, but 6 gets most of the way there.

---

👤 **ubergarm** commented the **2025-04-04** at **15:43:41**:<br>

Currently cooking up a CPU only "speed mix" blend using some of the advice from above. Will keep you posted.

Otherwise, ran a CPU only `llama-sweep-bench` on the `IQ5_K_R4/IQ4_K_R4` routed experts /`q8_0` all else blend. Accidently left the Intel Xeon 6980P in `balanced` mode instead of `performance`, but the trends should be similar.

![Image](https://github.com/user-attachments/assets/935b2ea4-80af-4edb-ad89-c67721863804)

<details>

<summary>llama-sweep-bench DeepSeek-V3-0324-IQ4_K_R4 logs</summary>

```bash
numactl -N 0 -m 0 \
./build/bin/llama-sweep-bench \
    --model /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-IQ4_K_R4.gguf \
    --alias ubergarm/DeepSeek-V3-0324-IQ4_K_R4 \
    --run-time-repack \
    --no-mmap \
    -ctk q8_0 \
    -mla 3 -fa \
    -amb 1024 \
    -fmoe \
    -c 32768 \
    -ub 512 \
    --threads 88 \
    --threads-batch 128 \
    --numa numactl

Current power profile is: balanced
Current THP enabled and defrag configs are:
[always] madvise never
[always] defer defer+madvise madvise never
Set numa balancing to be:
0

llama_model_loader: loaded meta data with 50 key-value pairs and 1147 tensors from /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-IQ4_K_R4.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = DeepSeek V3 0324
llama_model_loader: - kv   3:                            general.version str              = V3-0324
llama_model_loader: - kv   4:                           general.basename str              = DeepSeek
llama_model_loader: - kv   5:                         general.size_label str              = 256x21B
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
llama_model_loader: - kv  16:                          general.file_type u32              = 340
llama_model_loader: - kv  17:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  18:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  19:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  20:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  21:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  22:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  23:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  24:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  25:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  26:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  27:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  28:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  29:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  30:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  31:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  32: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  33: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  34:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  35:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  36:                      tokenizer.ggml.tokens arr[str,129280]  = ["...
llama_model_loader: - kv  37:                  tokenizer.ggml.token_type arr[i32,129280]  = [3...
llama_model_loader: - kv  38:                      tokenizer.ggml.merges arr[str,127741]  = ["...
llama_model_loader: - kv  39:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  40:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  41:            tokenizer.ggml.padding_token_id u32              = 1
llama_model_loader: - kv  42:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  43:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  44:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  45:               general.quantization_version u32              = 2
llama_model_loader: - kv  46:                      quantize.imatrix.file str              = /mnt/raid/models/ubergarm/DeepSeek-V3...
llama_model_loader: - kv  47:                   quantize.imatrix.dataset str              = calibration_data_v5_rc.txt
llama_model_loader: - kv  48:             quantize.imatrix.entries_count i32              = 720
llama_model_loader: - kv  49:              quantize.imatrix.chunks_count i32              = 213
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  612 tensors
llama_model_loader: - type iq4_k_r4:  116 tensors
llama_model_loader: - type iq5_k_r4:   58 tensors
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
llm_load_print_meta: model ftype      = IQ4_K_R4 - 4.5 bpw
llm_load_print_meta: model params     = 672.050 B
llm_load_print_meta: model size       = 386.183 GiB (4.936 BPW) 
llm_load_print_meta: repeating layers = 384.349 GiB (4.926 BPW, 670.196 B parameters)
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
llm_load_tensors: ggml ctx size =    0.47 MiB
llm_load_tensors:        CPU buffer size = 395450.97 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 3
llama_new_context_with_model: attn_max_b = 1024
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init: layer 0: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 1: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 2: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 3: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 4: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 5: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 6: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 7: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 8: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 9: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 10: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 11: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 12: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 13: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 14: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 15: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 16: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 17: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 18: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 19: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 20: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 21: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 22: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 23: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 24: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 25: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 26: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 27: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 28: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 29: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 30: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 31: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 32: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 33: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 34: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 35: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 36: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 37: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 38: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 39: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 40: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 41: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 42: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 43: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 44: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 45: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 46: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 47: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 48: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 49: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 50: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 51: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 52: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 53: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 54: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 55: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 56: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 57: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 58: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 59: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 60: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init:        CPU KV buffer size =  1166.63 MiB
llama_new_context_with_model: KV self size  = 1166.62 MiB, c^KV (q8_0): 1166.62 MiB, kv^T: not used
llama_new_context_with_model:        CPU  output buffer size =     0.49 MiB
llama_new_context_with_model:        CPU compute buffer size =  2662.01 MiB
llama_new_context_with_model: graph nodes  = 5500
llama_new_context_with_model: graph splits = 1

main: n_kv_max = 32768, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = -1, n_threads = 88, n_threads_batch = 128

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    4.412 |   116.05 |   13.303 |     9.62 |
|   512 |    128 |    512 |    4.384 |   116.79 |   13.639 |     9.38 |
|   512 |    128 |   1024 |    4.711 |   108.69 |   14.823 |     8.64 |
|   512 |    128 |   1536 |    5.448 |    93.98 |   15.187 |     8.43 |
|   512 |    128 |   2048 |    5.361 |    95.51 |   15.282 |     8.38 |
|   512 |    128 |   2560 |    6.005 |    85.26 |   16.579 |     7.72 |
|   512 |    128 |   3072 |    6.276 |    81.58 |   15.304 |     8.36 |
|   512 |    128 |   3584 |    6.383 |    80.21 |   15.072 |     8.49 |
|   512 |    128 |   4096 |    6.548 |    78.19 |   15.006 |     8.53 |
|   512 |    128 |   4608 |    7.245 |    70.67 |   15.262 |     8.39 |
|   512 |    128 |   5120 |    7.498 |    68.29 |   15.404 |     8.31 |
|   512 |    128 |   5632 |    7.992 |    64.06 |   15.555 |     8.23 |
|   512 |    128 |   6144 |    7.825 |    65.43 |   16.026 |     7.99 |
|   512 |    128 |   6656 |    8.140 |    62.90 |   16.011 |     7.99 |
|   512 |    128 |   7168 |    9.216 |    55.55 |   16.322 |     7.84 |
|   512 |    128 |   7680 |    9.197 |    55.67 |   16.641 |     7.69 |
|   512 |    128 |   8192 |    9.601 |    53.33 |   17.393 |     7.36 |
|   512 |    128 |   8704 |    9.049 |    56.58 |   17.375 |     7.37 |
|   512 |    128 |   9216 |    9.669 |    52.95 |   17.475 |     7.32 |
|   512 |    128 |   9728 |    9.592 |    53.38 |   17.728 |     7.22 |
|   512 |    128 |  10240 |   10.385 |    49.30 |   18.297 |     7.00 |
|   512 |    128 |  10752 |   10.284 |    49.79 |   18.500 |     6.92 |
|   512 |    128 |  11264 |   10.422 |    49.13 |   18.387 |     6.96 |
|   512 |    128 |  11776 |   11.144 |    45.94 |   18.602 |     6.88 |
|   512 |    128 |  12288 |   11.066 |    46.27 |   19.002 |     6.74 |
|   512 |    128 |  12800 |   11.749 |    43.58 |   19.933 |     6.42 |
|   512 |    128 |  13312 |   11.813 |    43.34 |   19.790 |     6.47 |
|   512 |    128 |  13824 |   12.959 |    39.51 |   18.546 |     6.90 |
|   512 |    128 |  14336 |   12.402 |    41.28 |   20.914 |     6.12 |
|   512 |    128 |  14848 |   13.064 |    39.19 |   20.959 |     6.11 |
|   512 |    128 |  15360 |   13.137 |    38.97 |   21.331 |     6.00 |
|   512 |    128 |  15872 |   13.158 |    38.91 |   21.756 |     5.88 |
|   512 |    128 |  16384 |   13.227 |    38.71 |   21.625 |     5.92 |
|   512 |    128 |  16896 |   14.089 |    36.34 |   22.327 |     5.73 |
|   512 |    128 |  17408 |   14.251 |    35.93 |   22.982 |     5.57 |
|   512 |    128 |  17920 |   14.794 |    34.61 |   22.817 |     5.61 |
|   512 |    128 |  18432 |   14.544 |    35.20 |   23.187 |     5.52 |
|   512 |    128 |  18944 |   14.835 |    34.51 |   23.744 |     5.39 |
|   512 |    128 |  19456 |   15.538 |    32.95 |   20.042 |     6.39 |
|   512 |    128 |  19968 |   16.182 |    31.64 |   24.139 |     5.30 |
|   512 |    128 |  20480 |   16.972 |    30.17 |   24.933 |     5.13 |
|   512 |    128 |  20992 |   15.876 |    32.25 |   25.319 |     5.06 |
|   512 |    128 |  21504 |   16.150 |    31.70 |   25.309 |     5.06 |
|   512 |    128 |  22016 |   16.810 |    30.46 |   25.217 |     5.08 |
|   512 |    128 |  22528 |   17.180 |    29.80 |   25.202 |     5.08 |
|   512 |    128 |  23040 |   18.171 |    28.18 |   25.445 |     5.03 |
|   512 |    128 |  23552 |   17.318 |    29.56 |   26.029 |     4.92 |
|   512 |    128 |  24064 |   18.848 |    27.16 |   26.128 |     4.90 |
|   512 |    128 |  24576 |   18.282 |    28.01 |   26.675 |     4.80 |
|   512 |    128 |  25088 |   18.234 |    28.08 |   21.079 |     6.07 |
|   512 |    128 |  25600 |   18.584 |    27.55 |   27.583 |     4.64 |
|   512 |    128 |  26112 |   19.350 |    26.46 |   27.687 |     4.62 |
|   512 |    128 |  26624 |   19.053 |    26.87 |   27.982 |     4.57 |
|   512 |    128 |  27136 |   19.228 |    26.63 |   28.328 |     4.52 |
|   512 |    128 |  27648 |   20.705 |    24.73 |   28.819 |     4.44 |
|   512 |    128 |  28160 |   19.993 |    25.61 |   29.508 |     4.34 |
|   512 |    128 |  28672 |   20.698 |    24.74 |   29.902 |     4.28 |
|   512 |    128 |  29184 |   20.320 |    25.20 |   29.555 |     4.33 |
|   512 |    128 |  29696 |   21.366 |    23.96 |   30.114 |     4.25 |
|   512 |    128 |  30208 |   21.293 |    24.05 |   29.625 |     4.32 |
|   512 |    128 |  30720 |   21.417 |    23.91 |   22.628 |     5.66 |
|   512 |    128 |  31232 |   21.941 |    23.34 |   30.653 |     4.18 |
|   512 |    128 |  31744 |   22.326 |    22.93 |   31.921 |     4.01 |
|   512 |    128 |  32256 |   23.055 |    22.21 |   31.750 |     4.03 |
============ Repacked 611 tensors

```

</details>

> Finally tested batch performance

Oh nice, is that with `llama-batched-bench` ?

---

👤 **ikawrakow** commented the **2025-04-04** at **16:55:06**:<br>

Nearly a 6X decrease in PP performance is quite a bit more than I'm expecting. In my testing it has been more in the 2.5X range when going to 32k tokens. I wonder if this is due to the balanced performance setting or the huge model (or both).

---

👤 **ubergarm** commented the **2025-04-04** at **17:59:03**:<br>

> Nearly a 6X decrease in PP performance is quite a bit more than I'm expecting. In my testing it has been more in the 2.5X range when going to 32k tokens. I wonder if this is due to the balanced performance setting or the huge model (or both).

Yeah, a lot of little variables can effect performance. One other data point I got was from [fairydreaming on r/LocalLLama](https://www.reddit.com/r/LocalLLaMA/comments/1joyl9t/comment/ml1lgob/) which drops off more slowly on their CPU+GPU rig to ~1.5X decrease in PP performance across 32k context.

---

👤 **ikawrakow** commented the **2025-04-04** at **18:02:59**:<br>

The TG peaks are also quite interesting. If I could make the performance stay where the peaks are for any `N_KV`, it would be a ~40% improvement at 32k tokens! Here I wonder if it is related to the 88 threads (and the work not splitting very well between them), or somehow related to the `-amb` option. 

@ubergarm 

You always use `numactl`. I'm really curious to know what happens if you don't involve `numactl` at all. I.e.,
```
./build/bin/llama-sweep-bench \
    --model /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-IQ4_K_R4.gguf \
    --alias ubergarm/DeepSeek-V3-0324-IQ4_K_R4 \
    --run-time-repack \
    --no-mmap \
    -ctk q8_0 \
    -mla 3 -fa \
    -amb 1024 \
    -fmoe \
    -c 32768 \
    -ub 512 \
    --threads 88 \
    --threads-batch 128 
```

---

👤 **ikawrakow** commented the **2025-04-04** at **18:05:38**:<br>

The fairydreaming tests use a GPU for attention, the slower drop in performance is expected in that setup. But for pure CPU inference I'm expecting around 2.5X lower performance at 32k tokens.

---

👤 **ubergarm** commented the **2025-04-04** at **21:02:06**:<br>

> You always use numactl. I'm really curious to know what happens if you don't involve numactl at all. I.e.,

I had some time while waiting for my "speed blend" to rsync between servers and tried the command without any numactl stuff. Interestingly, it loaded mostly on node 1, then some of the weights went into node 0 just before loading finished. I included numastat to show that in the detailed log.

![Image](https://github.com/user-attachments/assets/e502d14b-02ae-4729-992e-363e1f238dc8)

<details>

<summary>llama-sweep-bench without `numactl` stuff</summary>

```bash
# drop caches
$ sync && echo 3 | sudo tee /proc/sys/vm/drop_caches

# set to performance this time
Current power profile is: performance

# always encourages it to use anonhugepages
# as testing suggets improves performance on this rig
Current THP enabled and defrag configs are:
[always]
[always]

# numa_balancing off
Set numa balancing to be: 0

$ ./build/bin/llama-sweep-bench \
    --model /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-IQ4_K_R4.gguf \
    --alias ubergarm/DeepSeek-V3-0324-IQ4_K_R4 \
    --run-time-repack \
    --no-mmap \
    -ctk q8_0 \
    -mla 3 -fa \
    -amb 1024 \
    -fmoe \
    -c 32768 \
    -ub 512 \
    --threads 88 \
    --threads-batch 128 2>&1 | tee -a output.log

llama_model_loader: loaded meta data with 50 key-value pairs and 1147 tensors from /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-IQ4_K_R4.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = DeepSeek V3 0324
llama_model_loader: - kv   3:                            general.version str              = V3-0324
llama_model_loader: - kv   4:                           general.basename str              = DeepSeek
llama_model_loader: - kv   5:                         general.size_label str              = 256x21B
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
llama_model_loader: - kv  16:                          general.file_type u32              = 340
llama_model_loader: - kv  17:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  18:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  19:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  20:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  21:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  22:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  23:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  24:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  25:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  26:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  27:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  28:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  29:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  30:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  31:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  32: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  33: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  34:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  35:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  36:                      tokenizer.ggml.tokens arr[str,129280]  = ["...
llama_model_loader: - kv  37:                  tokenizer.ggml.token_type arr[i32,129280]  = [3...
llama_model_loader: - kv  38:                      tokenizer.ggml.merges arr[str,127741]  = ["...
llama_model_loader: - kv  39:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  40:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  41:            tokenizer.ggml.padding_token_id u32              = 1
llama_model_loader: - kv  42:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  43:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  44:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  45:               general.quantization_version u32              = 2
llama_model_loader: - kv  46:                      quantize.imatrix.file str              = /mnt/raid/models/ubergarm/DeepSeek-V3...
llama_model_loader: - kv  47:                   quantize.imatrix.dataset str              = calibration_data_v5_rc.txt
llama_model_loader: - kv  48:             quantize.imatrix.entries_count i32              = 720
llama_model_loader: - kv  49:              quantize.imatrix.chunks_count i32              = 213
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  612 tensors
llama_model_loader: - type iq4_k_r4:  116 tensors
llama_model_loader: - type iq5_k_r4:   58 tensors
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
llm_load_print_meta: model ftype      = IQ4_K_R4 - 4.5 bpw
llm_load_print_meta: model params     = 672.050 B
llm_load_print_meta: model size       = 386.183 GiB (4.936 BPW) 
llm_load_print_meta: repeating layers = 384.349 GiB (4.926 BPW, 670.196 B parameters)
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
llm_load_tensors: ggml ctx size =    0.47 MiB
llm_load_tensors:        CPU buffer size = 395450.97 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 3
llama_new_context_with_model: attn_max_b = 1024
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init: layer 0: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 1: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 2: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 3: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 4: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 5: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 6: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 7: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 8: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 9: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 10: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 11: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 12: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 13: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 14: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 15: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 16: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 17: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 18: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 19: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 20: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 21: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 22: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 23: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 24: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 25: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 26: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 27: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 28: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 29: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 30: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 31: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 32: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 33: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 34: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 35: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 36: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 37: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 38: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 39: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 40: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 41: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 42: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 43: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 44: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 45: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 46: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 47: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 48: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 49: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 50: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 51: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 52: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 53: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 54: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 55: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 56: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 57: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 58: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 59: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 60: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init:        CPU KV buffer size =  1166.63 MiB
llama_new_context_with_model: KV self size  = 1166.62 MiB, c^KV (q8_0): 1166.62 MiB, kv^T: not used
llama_new_context_with_model:        CPU  output buffer size =     0.49 MiB
llama_new_context_with_model:        CPU compute buffer size =  2662.01 MiB
llama_new_context_with_model: graph nodes  = 5500
llama_new_context_with_model: graph splits = 1

main: n_kv_max = 32768, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = -1, n_threads = 88, n_threads_batch = 128
```

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    4.214 |   121.49 |   19.559 |     6.54 |
|   512 |    128 |    512 |    4.304 |   118.97 |   19.317 |     6.63 |
|   512 |    128 |   1024 |    4.539 |   112.79 |   19.692 |     6.50 |
|   512 |    128 |   1536 |    4.859 |   105.37 |   20.024 |     6.39 |
|   512 |    128 |   2048 |    5.429 |    94.31 |   21.110 |     6.06 |
|   512 |    128 |   2560 |    5.698 |    89.86 |   21.308 |     6.01 |
|   512 |    128 |   3072 |    5.948 |    86.08 |   21.940 |     5.83 |
|   512 |    128 |   3584 |    6.368 |    80.40 |   21.664 |     5.91 |
|   512 |    128 |   4096 |    6.665 |    76.82 |   21.375 |     5.99 |
|   512 |    128 |   4608 |    7.055 |    72.57 |   21.764 |     5.88 |
|   512 |    128 |   5120 |    7.397 |    69.22 |   21.929 |     5.84 |
|   512 |    128 |   5632 |    7.846 |    65.25 |   21.051 |     6.08 |
|   512 |    128 |   6144 |    8.496 |    60.27 |   23.048 |     5.55 |
|   512 |    128 |   6656 |    8.884 |    57.63 |   21.473 |     5.96 |
|   512 |    128 |   7168 |    9.241 |    55.41 |   22.841 |     5.60 |
|   512 |    128 |   7680 |    9.832 |    52.08 |   21.809 |     5.87 |
|   512 |    128 |   8192 |    9.957 |    51.42 |   22.837 |     5.60 |
|   512 |    128 |   8704 |   10.521 |    48.67 |   23.967 |     5.34 |
|   512 |    128 |   9216 |   10.787 |    47.46 |   23.475 |     5.45 |
|   512 |    128 |   9728 |   11.187 |    45.77 |   23.407 |     5.47 |
|   512 |    128 |  10240 |   11.988 |    42.71 |   25.122 |     5.10 |
|   512 |    128 |  10752 |   12.502 |    40.95 |   24.736 |     5.17 |
|   512 |    128 |  11264 |   12.874 |    39.77 |   24.705 |     5.18 |
|   512 |    128 |  11776 |   12.893 |    39.71 |   24.578 |     5.21 |
|   512 |    128 |  12288 |   13.309 |    38.47 |   25.649 |     4.99 |
|   512 |    128 |  12800 |   13.647 |    37.52 |   24.652 |     5.19 |
|   512 |    128 |  13312 |   14.318 |    35.76 |   25.035 |     5.11 |
|   512 |    128 |  13824 |   14.879 |    34.41 |   24.243 |     5.28 |
|   512 |    128 |  14336 |   15.221 |    33.64 |   25.826 |     4.96 |
|   512 |    128 |  14848 |   15.292 |    33.48 |   26.096 |     4.91 |
|   512 |    128 |  15360 |   15.592 |    32.84 |   25.744 |     4.97 |
|   512 |    128 |  15872 |   15.757 |    32.49 |   26.224 |     4.88 |
|   512 |    128 |  16384 |   14.834 |    34.51 |   26.616 |     4.81 |
|   512 |    128 |  16896 |   15.757 |    32.49 |   27.967 |     4.58 |
|   512 |    128 |  17408 |   16.378 |    31.26 |   27.682 |     4.62 |
|   512 |    128 |  17920 |   16.754 |    30.56 |   27.855 |     4.60 |
|   512 |    128 |  18432 |   17.300 |    29.59 |   27.905 |     4.59 |
|   512 |    128 |  18944 |   17.347 |    29.52 |   28.338 |     4.52 |
|   512 |    128 |  19456 |   17.895 |    28.61 |   24.992 |     5.12 |
|   512 |    128 |  19968 |   18.210 |    28.12 |   28.662 |     4.47 |
|   512 |    128 |  20480 |   18.579 |    27.56 |   28.880 |     4.43 |
|   512 |    128 |  20992 |   18.920 |    27.06 |   29.153 |     4.39 |
|   512 |    128 |  21504 |   19.537 |    26.21 |   29.282 |     4.37 |
|   512 |    128 |  22016 |   19.716 |    25.97 |   29.682 |     4.31 |
|   512 |    128 |  22528 |   20.576 |    24.88 |   30.040 |     4.26 |
|   512 |    128 |  23040 |   20.705 |    24.73 |   30.366 |     4.22 |
|   512 |    128 |  23552 |   21.201 |    24.15 |   30.501 |     4.20 |
|   512 |    128 |  24064 |   21.809 |    23.48 |   30.800 |     4.16 |
|   512 |    128 |  24576 |   22.042 |    23.23 |   30.988 |     4.13 |
|   512 |    128 |  25088 |   22.660 |    22.59 |   26.174 |     4.89 |
|   512 |    128 |  25600 |   23.038 |    22.22 |   31.451 |     4.07 |
|   512 |    128 |  26112 |   23.601 |    21.69 |   31.606 |     4.05 |
|   512 |    128 |  26624 |   23.744 |    21.56 |   31.454 |     4.07 |
|   512 |    128 |  27136 |   24.403 |    20.98 |   32.176 |     3.98 |
|   512 |    128 |  27648 |   24.954 |    20.52 |   31.961 |     4.00 |
|   512 |    128 |  28160 |   25.142 |    20.36 |   32.050 |     3.99 |
|   512 |    128 |  28672 |   25.774 |    19.87 |   32.425 |     3.95 |
|   512 |    128 |  29184 |   25.847 |    19.81 |   33.104 |     3.87 |
|   512 |    128 |  29696 |   26.218 |    19.53 |   32.757 |     3.91 |
|   512 |    128 |  30208 |   26.704 |    19.17 |   33.055 |     3.87 |
|   512 |    128 |  30720 |   27.111 |    18.89 |   27.009 |     4.74 |
|   512 |    128 |  31232 |   26.987 |    18.97 |   33.298 |     3.84 |
|   512 |    128 |  31744 |   26.712 |    19.17 |   33.334 |     3.84 |
|   512 |    128 |  32256 |   28.083 |    18.23 |   33.414 |     3.83 |

`============ Repacked 611 tensors`

```bash
$ grep Huge /proc/meminfo
AnonHugePages:  406736896 kB
ShmemHugePages:        0 kB
FileHugePages:         0 kB
HugePages_Total:       0
HugePages_Free:        0
HugePages_Rsvd:        0
HugePages_Surp:        0
Hugepagesize:       2048 kB
Hugetlb:               0 kB

$ numastat -m -p $(pidof llama-sweep-bench)
Per-node process memory usage (in MBs) for PID 659855 (llama-sweep-ben)
                           Node 0          Node 1           Total
                  --------------- --------------- ---------------
Huge                         0.00            0.00            0.00
Heap                         2.80           34.14           36.94
Stack                        0.04            0.05            0.08
Private                  13999.99       383083.54       397083.52
----------------  --------------- --------------- ---------------
Total                    14002.82       383117.72       397120.54

Per-node system memory usage (in MBs):
                          Node 0          Node 1           Total
                 --------------- --------------- ---------------
MemTotal               771710.76       773987.20      1545697.96
MemFree                743559.40         1745.54       745304.94
MemUsed                 28151.36       772241.67       800393.03
SwapCached                  0.21            0.69            0.90
Active                  14157.56       383159.96       397317.52
Inactive                 8662.71       383016.18       391678.89
Active(anon)            14076.79       383139.31       397216.09
Inactive(anon)              3.26           22.98           26.25
Active(file)               80.78           20.65          101.43
Inactive(file)           8659.45       382993.20       391652.64
Unevictable                29.86            5.50           35.36
Mlocked                    21.07            5.50           26.57
Dirty                      20.00            0.05           20.05
Writeback                   0.00            0.00            0.00
FilePages                8755.46       383025.92       391781.38
Mapped                     82.61           63.21          145.82
AnonPages               14097.36       383158.36       397255.73
Shmem                      11.92            5.88           17.80
KernelStack                39.69           38.11           77.80
PageTables                  6.78          775.85          782.62
SecPageTables               0.00            0.00            0.00
NFS_Unstable                0.00            0.00            0.00
Bounce                      0.00            0.00            0.00
WritebackTmp                0.00            0.00            0.00
Slab                     2489.91         2737.77         5227.68
SReclaimable              402.44         1022.84         1425.27
SUnreclaim               2087.47         1714.93         3802.40
AnonHugePages           14010.00       383100.00       397110.00
ShmemHugePages              0.00            0.00            0.00
ShmemPmdMapped              0.00            0.00            0.00
FileHugePages               0.00            0.00            0.00
FilePmdMapped               0.00            0.00            0.00
HugePages_Total             0.00            0.00            0.00
HugePages_Free              0.00            0.00            0.00
HugePages_Surp              0.00            0.00            0.00
KReclaimable              402.44         1022.84         1425.27
```

</details>

---

👤 **ubergarm** commented the **2025-04-04** at **21:02:06**:<br>

> You always use numactl. I'm really curious to know what happens if you don't involve numactl at all. I.e.,

I had some time while waiting for my "speed blend" to rsync between servers and tried the command without any numactl stuff. Interestingly, it loaded mostly on node 1, then some of the weights went into node 0 just before loading finished. I included numastat to show that in the detailed log.

![Image](https://github.com/user-attachments/assets/e502d14b-02ae-4729-992e-363e1f238dc8)

<details>

<summary>llama-sweep-bench without `numactl` stuff</summary>

```bash
# drop caches
$ sync && echo 3 | sudo tee /proc/sys/vm/drop_caches

# set to performance this time
Current power profile is: performance

# always encourages it to use anonhugepages
# as testing suggets improves performance on this rig
Current THP enabled and defrag configs are:
[always]
[always]

# numa_balancing off
Set numa balancing to be: 0

$ ./build/bin/llama-sweep-bench \
    --model /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-IQ4_K_R4.gguf \
    --alias ubergarm/DeepSeek-V3-0324-IQ4_K_R4 \
    --run-time-repack \
    --no-mmap \
    -ctk q8_0 \
    -mla 3 -fa \
    -amb 1024 \
    -fmoe \
    -c 32768 \
    -ub 512 \
    --threads 88 \
    --threads-batch 128 2>&1 | tee -a output.log

llama_model_loader: loaded meta data with 50 key-value pairs and 1147 tensors from /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-IQ4_K_R4.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = DeepSeek V3 0324
llama_model_loader: - kv   3:                            general.version str              = V3-0324
llama_model_loader: - kv   4:                           general.basename str              = DeepSeek
llama_model_loader: - kv   5:                         general.size_label str              = 256x21B
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
llama_model_loader: - kv  16:                          general.file_type u32              = 340
llama_model_loader: - kv  17:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  18:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  19:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  20:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  21:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  22:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  23:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  24:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  25:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  26:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  27:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  28:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  29:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  30:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  31:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  32: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  33: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  34:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  35:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  36:                      tokenizer.ggml.tokens arr[str,129280]  = ["...
llama_model_loader: - kv  37:                  tokenizer.ggml.token_type arr[i32,129280]  = [3...
llama_model_loader: - kv  38:                      tokenizer.ggml.merges arr[str,127741]  = ["...
llama_model_loader: - kv  39:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  40:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  41:            tokenizer.ggml.padding_token_id u32              = 1
llama_model_loader: - kv  42:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  43:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  44:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  45:               general.quantization_version u32              = 2
llama_model_loader: - kv  46:                      quantize.imatrix.file str              = /mnt/raid/models/ubergarm/DeepSeek-V3...
llama_model_loader: - kv  47:                   quantize.imatrix.dataset str              = calibration_data_v5_rc.txt
llama_model_loader: - kv  48:             quantize.imatrix.entries_count i32              = 720
llama_model_loader: - kv  49:              quantize.imatrix.chunks_count i32              = 213
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  612 tensors
llama_model_loader: - type iq4_k_r4:  116 tensors
llama_model_loader: - type iq5_k_r4:   58 tensors
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
llm_load_print_meta: model ftype      = IQ4_K_R4 - 4.5 bpw
llm_load_print_meta: model params     = 672.050 B
llm_load_print_meta: model size       = 386.183 GiB (4.936 BPW) 
llm_load_print_meta: repeating layers = 384.349 GiB (4.926 BPW, 670.196 B parameters)
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
llm_load_tensors: ggml ctx size =    0.47 MiB
llm_load_tensors:        CPU buffer size = 395450.97 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 3
llama_new_context_with_model: attn_max_b = 1024
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init: layer 0: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 1: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 2: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 3: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 4: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 5: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 6: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 7: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 8: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 9: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 10: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 11: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 12: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 13: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 14: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 15: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 16: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 17: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 18: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 19: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 20: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 21: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 22: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 23: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 24: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 25: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 26: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 27: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 28: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 29: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 30: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 31: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 32: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 33: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 34: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 35: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 36: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 37: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 38: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 39: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 40: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 41: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 42: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 43: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 44: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 45: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 46: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 47: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 48: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 49: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 50: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 51: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 52: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 53: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 54: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 55: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 56: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 57: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 58: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 59: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 60: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init:        CPU KV buffer size =  1166.63 MiB
llama_new_context_with_model: KV self size  = 1166.62 MiB, c^KV (q8_0): 1166.62 MiB, kv^T: not used
llama_new_context_with_model:        CPU  output buffer size =     0.49 MiB
llama_new_context_with_model:        CPU compute buffer size =  2662.01 MiB
llama_new_context_with_model: graph nodes  = 5500
llama_new_context_with_model: graph splits = 1

main: n_kv_max = 32768, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = -1, n_threads = 88, n_threads_batch = 128
```

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    4.214 |   121.49 |   19.559 |     6.54 |
|   512 |    128 |    512 |    4.304 |   118.97 |   19.317 |     6.63 |
|   512 |    128 |   1024 |    4.539 |   112.79 |   19.692 |     6.50 |
|   512 |    128 |   1536 |    4.859 |   105.37 |   20.024 |     6.39 |
|   512 |    128 |   2048 |    5.429 |    94.31 |   21.110 |     6.06 |
|   512 |    128 |   2560 |    5.698 |    89.86 |   21.308 |     6.01 |
|   512 |    128 |   3072 |    5.948 |    86.08 |   21.940 |     5.83 |
|   512 |    128 |   3584 |    6.368 |    80.40 |   21.664 |     5.91 |
|   512 |    128 |   4096 |    6.665 |    76.82 |   21.375 |     5.99 |
|   512 |    128 |   4608 |    7.055 |    72.57 |   21.764 |     5.88 |
|   512 |    128 |   5120 |    7.397 |    69.22 |   21.929 |     5.84 |
|   512 |    128 |   5632 |    7.846 |    65.25 |   21.051 |     6.08 |
|   512 |    128 |   6144 |    8.496 |    60.27 |   23.048 |     5.55 |
|   512 |    128 |   6656 |    8.884 |    57.63 |   21.473 |     5.96 |
|   512 |    128 |   7168 |    9.241 |    55.41 |   22.841 |     5.60 |
|   512 |    128 |   7680 |    9.832 |    52.08 |   21.809 |     5.87 |
|   512 |    128 |   8192 |    9.957 |    51.42 |   22.837 |     5.60 |
|   512 |    128 |   8704 |   10.521 |    48.67 |   23.967 |     5.34 |
|   512 |    128 |   9216 |   10.787 |    47.46 |   23.475 |     5.45 |
|   512 |    128 |   9728 |   11.187 |    45.77 |   23.407 |     5.47 |
|   512 |    128 |  10240 |   11.988 |    42.71 |   25.122 |     5.10 |
|   512 |    128 |  10752 |   12.502 |    40.95 |   24.736 |     5.17 |
|   512 |    128 |  11264 |   12.874 |    39.77 |   24.705 |     5.18 |
|   512 |    128 |  11776 |   12.893 |    39.71 |   24.578 |     5.21 |
|   512 |    128 |  12288 |   13.309 |    38.47 |   25.649 |     4.99 |
|   512 |    128 |  12800 |   13.647 |    37.52 |   24.652 |     5.19 |
|   512 |    128 |  13312 |   14.318 |    35.76 |   25.035 |     5.11 |
|   512 |    128 |  13824 |   14.879 |    34.41 |   24.243 |     5.28 |
|   512 |    128 |  14336 |   15.221 |    33.64 |   25.826 |     4.96 |
|   512 |    128 |  14848 |   15.292 |    33.48 |   26.096 |     4.91 |
|   512 |    128 |  15360 |   15.592 |    32.84 |   25.744 |     4.97 |
|   512 |    128 |  15872 |   15.757 |    32.49 |   26.224 |     4.88 |
|   512 |    128 |  16384 |   14.834 |    34.51 |   26.616 |     4.81 |
|   512 |    128 |  16896 |   15.757 |    32.49 |   27.967 |     4.58 |
|   512 |    128 |  17408 |   16.378 |    31.26 |   27.682 |     4.62 |
|   512 |    128 |  17920 |   16.754 |    30.56 |   27.855 |     4.60 |
|   512 |    128 |  18432 |   17.300 |    29.59 |   27.905 |     4.59 |
|   512 |    128 |  18944 |   17.347 |    29.52 |   28.338 |     4.52 |
|   512 |    128 |  19456 |   17.895 |    28.61 |   24.992 |     5.12 |
|   512 |    128 |  19968 |   18.210 |    28.12 |   28.662 |     4.47 |
|   512 |    128 |  20480 |   18.579 |    27.56 |   28.880 |     4.43 |
|   512 |    128 |  20992 |   18.920 |    27.06 |   29.153 |     4.39 |
|   512 |    128 |  21504 |   19.537 |    26.21 |   29.282 |     4.37 |
|   512 |    128 |  22016 |   19.716 |    25.97 |   29.682 |     4.31 |
|   512 |    128 |  22528 |   20.576 |    24.88 |   30.040 |     4.26 |
|   512 |    128 |  23040 |   20.705 |    24.73 |   30.366 |     4.22 |
|   512 |    128 |  23552 |   21.201 |    24.15 |   30.501 |     4.20 |
|   512 |    128 |  24064 |   21.809 |    23.48 |   30.800 |     4.16 |
|   512 |    128 |  24576 |   22.042 |    23.23 |   30.988 |     4.13 |
|   512 |    128 |  25088 |   22.660 |    22.59 |   26.174 |     4.89 |
|   512 |    128 |  25600 |   23.038 |    22.22 |   31.451 |     4.07 |
|   512 |    128 |  26112 |   23.601 |    21.69 |   31.606 |     4.05 |
|   512 |    128 |  26624 |   23.744 |    21.56 |   31.454 |     4.07 |
|   512 |    128 |  27136 |   24.403 |    20.98 |   32.176 |     3.98 |
|   512 |    128 |  27648 |   24.954 |    20.52 |   31.961 |     4.00 |
|   512 |    128 |  28160 |   25.142 |    20.36 |   32.050 |     3.99 |
|   512 |    128 |  28672 |   25.774 |    19.87 |   32.425 |     3.95 |
|   512 |    128 |  29184 |   25.847 |    19.81 |   33.104 |     3.87 |
|   512 |    128 |  29696 |   26.218 |    19.53 |   32.757 |     3.91 |
|   512 |    128 |  30208 |   26.704 |    19.17 |   33.055 |     3.87 |
|   512 |    128 |  30720 |   27.111 |    18.89 |   27.009 |     4.74 |
|   512 |    128 |  31232 |   26.987 |    18.97 |   33.298 |     3.84 |
|   512 |    128 |  31744 |   26.712 |    19.17 |   33.334 |     3.84 |
|   512 |    128 |  32256 |   28.083 |    18.23 |   33.414 |     3.83 |

`============ Repacked 611 tensors`

```bash
$ grep Huge /proc/meminfo
AnonHugePages:  406736896 kB
ShmemHugePages:        0 kB
FileHugePages:         0 kB
HugePages_Total:       0
HugePages_Free:        0
HugePages_Rsvd:        0
HugePages_Surp:        0
Hugepagesize:       2048 kB
Hugetlb:               0 kB

$ numastat -m -p $(pidof llama-sweep-bench)
Per-node process memory usage (in MBs) for PID 659855 (llama-sweep-ben)
                           Node 0          Node 1           Total
                  --------------- --------------- ---------------
Huge                         0.00            0.00            0.00
Heap                         2.80           34.14           36.94
Stack                        0.04            0.05            0.08
Private                  13999.99       383083.54       397083.52
----------------  --------------- --------------- ---------------
Total                    14002.82       383117.72       397120.54

Per-node system memory usage (in MBs):
                          Node 0          Node 1           Total
                 --------------- --------------- ---------------
MemTotal               771710.76       773987.20      1545697.96
MemFree                743559.40         1745.54       745304.94
MemUsed                 28151.36       772241.67       800393.03
SwapCached                  0.21            0.69            0.90
Active                  14157.56       383159.96       397317.52
Inactive                 8662.71       383016.18       391678.89
Active(anon)            14076.79       383139.31       397216.09
Inactive(anon)              3.26           22.98           26.25
Active(file)               80.78           20.65          101.43
Inactive(file)           8659.45       382993.20       391652.64
Unevictable                29.86            5.50           35.36
Mlocked                    21.07            5.50           26.57
Dirty                      20.00            0.05           20.05
Writeback                   0.00            0.00            0.00
FilePages                8755.46       383025.92       391781.38
Mapped                     82.61           63.21          145.82
AnonPages               14097.36       383158.36       397255.73
Shmem                      11.92            5.88           17.80
KernelStack                39.69           38.11           77.80
PageTables                  6.78          775.85          782.62
SecPageTables               0.00            0.00            0.00
NFS_Unstable                0.00            0.00            0.00
Bounce                      0.00            0.00            0.00
WritebackTmp                0.00            0.00            0.00
Slab                     2489.91         2737.77         5227.68
SReclaimable              402.44         1022.84         1425.27
SUnreclaim               2087.47         1714.93         3802.40
AnonHugePages           14010.00       383100.00       397110.00
ShmemHugePages              0.00            0.00            0.00
ShmemPmdMapped              0.00            0.00            0.00
FileHugePages               0.00            0.00            0.00
FilePmdMapped               0.00            0.00            0.00
HugePages_Total             0.00            0.00            0.00
HugePages_Free              0.00            0.00            0.00
HugePages_Surp              0.00            0.00            0.00
KReclaimable              402.44         1022.84         1425.27
```

<details>

---

👤 **saood06** commented the **2025-04-05** at **02:58:44**:<br>

@ubergarm 

You can use the script included to plot them together with the legend using the filenames.

I did it using your raw data. 

TG:
![Image](https://github.com/user-attachments/assets/9c58101b-1b64-4ec3-8668-dccbf06fcd5a)

PP:

![Image](https://github.com/user-attachments/assets/ca9f3ab7-6b00-4951-b870-be16e1e1caa9)

>Oh nice, is that with llama-batched-bench ?

It is but I just used a script to graph it. Raw results below, the result for B=1, sweep bench result was used.

|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
|     0 |    128 |    2 |    256 |    0.961 |     0.00 |   42.118 |     6.08 |   43.079 |     5.94 |
|     0 |    128 |    3 |    384 |    0.963 |     0.00 |   46.332 |     8.29 |   47.295 |     8.12 |
|     0 |    128 |    4 |    512 |    0.971 |     0.00 |   54.238 |     9.44 |   55.209 |     9.27 |
|     0 |    128 |    5 |    640 |    1.114 |     0.00 |   58.274 |    10.98 |   59.387 |    10.78 |
|     0 |    128 |    6 |    768 |    0.960 |     0.00 |   64.813 |    11.85 |   65.773 |    11.68 |
|     0 |    128 |    7 |    896 |    0.959 |     0.00 |   82.076 |    10.92 |   83.035 |    10.79 |
|     0 |    128 |    8 |   1024 |    0.961 |     0.00 |   88.326 |    11.59 |   89.287 |    11.47 |
|     0 |    128 |    9 |   1152 |    0.963 |     0.00 |  105.301 |    10.94 |  106.264 |    10.84 |
|     0 |    128 |   10 |   1280 |    0.960 |     0.00 |  103.148 |    12.41 |  104.108 |    12.29 |
|     0 |    128 |   11 |   1408 |    0.960 |     0.00 |  118.788 |    11.85 |  119.748 |    11.76 |
|     0 |    128 |   12 |   1536 |    0.962 |     0.00 |  118.974 |    12.91 |  119.936 |    12.81 |
|     0 |    128 |   13 |   1664 |    0.965 |     0.00 |  141.875 |    11.73 |  142.840 |    11.65 |
|     0 |    128 |   14 |   1792 |    0.972 |     0.00 |  150.249 |    11.93 |  151.221 |    11.85 |
|     0 |    128 |   15 |   1920 |    0.962 |     0.00 |  158.899 |    12.08 |  159.861 |    12.01 |
|     0 |    128 |   16 |   2048 |    0.965 |     0.00 |  197.818 |    10.35 |  198.783 |    10.30 |


@ikawrakow 

> The fairydreaming tests use a GPU for attention, the slower drop in performance is expected in that setup. But for pure CPU inference I'm expecting around 2.5X lower performance at 32k tokens.

My own results show ~3.5X lower PP performance at just 16k tokens.

---

👤 **ikawrakow** commented the **2025-04-05** at **06:07:18**:<br>

I'm almost sure the TG peaks are due to number of threads. If you try with 128 TG threads, performance will be slightly lower at zero context, but for large contexts it should match the peaks for all context lengths.

---

👤 **ubergarm** commented the **2025-04-05** at **15:58:02**:<br>

Okay, got my "CPU only speed blend" quant cooked, copied over, perplexity, and a few sweep-bench comparisons against itself with different threads and amb settings.

<details>

<summary>DeepSeek-V3-0324-CPU-IQ3_K_R4 "CPU only speed blend" mix</summary>

## tl;dr;

Mostly ~q6/iq5_k_r4 for embedding/attention/dense layers/shared experts. First 17 routed experts are down/(up|gate) iq5_k_r4/iq4_k_r4 and the remainder are iq4_k_r4/iq3_k_r4.

`PPL = 3.3193 +/- 0.01830`

```bash
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type iq6_k:    1 tensors
llama_model_loader: - type q6_0_r4:   61 tensors
llama_model_loader: - type iq3_k_r4:   82 tensors
llama_model_loader: - type iq4_k_r4:   75 tensors
llama_model_loader: - type iq5_k_r4:  567 tensors

llm_load_print_meta: model type       = 671B
llm_load_print_meta: model ftype      = IQ3_K - 3.4325 bpw
llm_load_print_meta: model params     = 672.050 B
llm_load_print_meta: model size       = 324.011 GiB (4.141 BPW)
llm_load_print_meta: repeating layers = 322.703 GiB (4.136 BPW, 670.196 B parameters)
llm_load_print_meta: general.name     = DeepSeek V3 0324
```

## Perplexity
```bash
numactl -N 1 -m 1 \
./build/bin/llama-perplexity \
    --model /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-CPU-IQ3_K_R4.gguf \
    -ctk q8_0 \
    -mla 3 -fa \
    -amb 512 \
    -fmoe \
    --ctx-size 512 \
    --ubatch-size 512 \
    -f wiki.test.raw \
    --seed 1337 \
    --numa numactl \
    --threads 128

main: build = 3622 (c616306a)
main: built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu
main: seed  = 1337

llama_kv_cache_init:        CPU KV buffer size =    72.91 MiB
llama_new_context_with_model: KV self size  =   72.91 MiB, c^KV (q8_0):   72.91 MiB, kv^T: not used
llama_new_context_with_model:        CPU  output buffer size =     1.97 MiB
llama_new_context_with_model:        CPU compute buffer size =   450.01 MiB
llama_new_context_with_model: graph nodes  = 3487
llama_new_context_with_model: graph splits = 1

system_info: n_threads = 128 / 512 | AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | 
perplexity: tokenizing the input ..
perplexity: tokenization took 885.253 ms
perplexity: calculating perplexity over 561 chunks, n_ctx=512, batch_size=2048, n_seq=4
perplexity: 18.52 seconds per pass - ETA 43.28 minutes
[1]2.5128,[2]3.1998,[3]2.3365,[4]1.9572,[5]1.7672,[6]1.6281,[7]1.5395,[8]1.4757,[9]1.4355,[10]1.3986,[11]1.3863,[12]1.4171,[13]1.4335,[14]1.5570,[15]1.6860,[16]1.7427,[17]1.9032,[18]2.0271,[19]1.9913,[20]1.9776,[21]2.0854,[22]2.0602,[23]2.0347,[24]2.0476,[25]2.0186,[26]1.9969,[27]2.0413,[28]2.0507,[29]2.0970,[30]2.1295,[31]2.1608,[32]2.1794,[33]2.2186,[34]2.2617,[35]2.3099,[36]2.3635,[37]2.3978,[38]2.4457,[39]2.4853,[40]2.5440,[41]2.5853,[42]2.5976,[43]2.6473,[44]2.6637,[45]2.7436,[46]2.7934,[47]2.7499,[48]2.7051,[49]2.6812,[50]2.6987,[51]2.7413,[52]2.7537,[53]2.8060,[54]2.8201,[55]2.8508,[56]2.8807,[57]2.8940,[58]2.9277,[59]2.9387,[60]2.9864,[61]3.0248,[62]3.0709,[63]3.1017,[64]3.1429,[65]3.1526,[66]3.1355,[67]3.1118,[68]3.1372,[69]3.1314,[70]3.1476,[71]3.1660,[72]3.1796,[73]3.1931,[74]3.2149,[75]3.1951,[76]3.1489,[77]3.1060,[78]3.1012,[79]3.0804,[80]3.0632,[81]3.0289,[82]3.0333,[83]3.0030,[84]2.9691,[85]2.9358,[86]2.9134,[87]2.9083,[88]2.8809,[89]2.8642,[90]2.8387,[91]2.8113,[92]2.7865,[93]2.7604,[94]2.7369,[95]2.7151,[96]2.7141,[97]2.7189,[98]2.7038,[99]2.6870,[100]2.6894,[101]2.6821,[102]2.6980,[103]2.7237,[104]2.7405,[105]2.7372,[106]2.7591,[107]2.7837,[108]2.8041,[109]2.8372,[110]2.8699,[111]2.8884,[112]2.8629,[113]2.8500,[114]2.8292,[115]2.8139,[116]2.8010,[117]2.7792,[118]2.7587,[119]2.7376,[120]2.7196,[121]2.7036,[122]2.6864,[123]2.6691,[124]2.6500,[125]2.6333,[126]2.6165,[127]2.6034,[128]2.5949,[129]2.5838,[130]2.5714,[131]2.5622,[132]2.5688,[133]2.5782,[134]2.5857,[135]2.5965,[136]2.6115,[137]2.6256,[138]2.6335,[139]2.6442,[140]2.6447,[141]2.6465,[142]2.6450,[143]2.6459,[144]2.6432,[145]2.6352,[146]2.6334,[147]2.6377,[148]2.6379,[149]2.6395,[150]2.6337,[151]2.6321,[152]2.6294,[153]2.6255,[154]2.6254,[155]2.6295,[156]2.6307,[157]2.6363,[158]2.6444,[159]2.6469,[160]2.6556,[161]2.6641,[162]2.6743,[163]2.6796,[164]2.6999,[165]2.7236,[166]2.7410,[167]2.7531,[168]2.7770,[169]2.7996,[170]2.8214,[171]2.8429,[172]2.8273,[173]2.8112,[174]2.7987,[175]2.7868,[176]2.7746,[177]2.7635,[178]2.7508,[179]2.7373,[180]2.7409,[181]2.7550,[182]2.7698,[183]2.7839,[184]2.7969,[185]2.8065,[186]2.8224,[187]2.8380,[188]2.8519,[189]2.8622,[190]2.8627,[191]2.8698,[192]2.8729,[193]2.8780,[194]2.8971,[195]2.9057,[196]2.9187,[197]2.9283,[198]2.9329,[199]2.9386,[200]2.9379,[201]2.9528,[202]2.9480,[203]2.9532,[204]2.9558,[205]2.9556,[206]2.9582,[207]2.9667,[208]2.9757,[209]2.9846,[210]2.9847,[211]2.9802,[212]2.9808,[213]2.9883,[214]2.9901,[215]2.9957,[216]2.9962,[217]2.9920,[218]2.9920,[219]2.9927,[220]2.9925,[221]2.9932,[222]2.9930,[223]2.9939,[224]2.9986,[225]3.0004,[226]2.9925,[227]2.9900,[228]2.9914,[229]2.9951,[230]3.0014,[231]3.0074,[232]2.9994,[233]2.9921,[234]2.9923,[235]2.9911,[236]2.9998,[237]3.0079,[238]3.0172,[239]3.0268,[240]3.0361,[241]3.0471,[242]3.0615,[243]3.0741,[244]3.0820,[245]3.0929,[246]3.1031,[247]3.1021,[248]3.0979,[249]3.0960,[250]3.0899,[251]3.0878,[252]3.0899,[253]3.0939,[254]3.1008,[255]3.1070,[256]3.1101,[257]3.1131,[258]3.1144,[259]3.1179,[260]3.1201,[261]3.1214,[262]3.1205,[263]3.1263,[264]3.1286,[265]3.1291,[266]3.1306,[267]3.1327,[268]3.1357,[269]3.1385,[270]3.1378,[271]3.1363,[272]3.1297,[273]3.1294,[274]3.1225,[275]3.1122,[276]3.1010,[277]3.1029,[278]3.1128,[279]3.1187,[280]3.1265,[281]3.1338,[282]3.1394,[283]3.1458,[284]3.1518,[285]3.1654,[286]3.1675,[287]3.1708,[288]3.1759,[289]3.1781,[290]3.1701,[291]3.1613,[292]3.1597,[293]3.1591,[294]3.1570,[295]3.1548,[296]3.1570,[297]3.1575,[298]3.1631,[299]3.1689,[300]3.1718,[301]3.1758,[302]3.1780,[303]3.1795,[304]3.1790,[305]3.1904,[306]3.1973,[307]3.2079,[308]3.1969,[309]3.1920,[310]3.1831,[311]3.1862,[312]3.1877,[313]3.1936,[314]3.1959,[315]3.1990,[316]3.2006,[317]3.2026,[318]3.2032,[319]3.2035,[320]3.2076,[321]3.2078,[322]3.2096,[323]3.2160,[324]3.2167,[325]3.2221,[326]3.2263,[327]3.2302,[328]3.2327,[329]3.2346,[330]3.2409,[331]3.2439,[332]3.2478,[333]3.2467,[334]3.2467,[335]3.2474,[336]3.2475,[337]3.2486,[338]3.2488,[339]3.2512,[340]3.2547,[341]3.2599,[342]3.2687,[343]3.2775,[344]3.2824,[345]3.2740,[346]3.2664,[347]3.2617,[348]3.2543,[349]3.2505,[350]3.2491,[351]3.2537,[352]3.2683,[353]3.2772,[354]3.2897,[355]3.2982,[356]3.3034,[357]3.3150,[358]3.3248,[359]3.3276,[360]3.3340,[361]3.3433,[362]3.3519,[363]3.3572,[364]3.3639,[365]3.3695,[366]3.3796,[367]3.3881,[368]3.3943,[369]3.4019,[370]3.4104,[371]3.4235,[372]3.4322,[373]3.4356,[374]3.4389,[375]3.4437,[376]3.4563,[377]3.4674,[378]3.4704,[379]3.4704,[380]3.4668,[381]3.4718,[382]3.4775,[383]3.4807,[384]3.4850,[385]3.4888,[386]3.4947,[387]3.5004,[388]3.5033,[389]3.4933,[390]3.4842,[391]3.4740,[392]3.4687,[393]3.4596,[394]3.4511,[395]3.4422,[396]3.4325,[397]3.4241,[398]3.4150,[399]3.4048,[400]3.3963,[401]3.3865,[402]3.3766,[403]3.3683,[404]3.3584,[405]3.3492,[406]3.3398,[407]3.3307,[408]3.3220,[409]3.3136,[410]3.3076,[411]3.3086,[412]3.3038,[413]3.3059,[414]3.3075,[415]3.3050,[416]3.3052,[417]3.3071,[418]3.3014,[419]3.3026,[420]3.3000,[421]3.2989,[422]3.2994,[423]3.2989,[424]3.3026,[425]3.3024,[426]3.3029,[427]3.3019,[428]3.3043,[429]3.3055,[430]3.3082,[431]3.3091,[432]3.3081,[433]3.3046,[434]3.3051,[435]3.2979,[436]3.2921,[437]3.2881,[438]3.2863,[439]3.2839,[440]3.2887,[441]3.2943,[442]3.3014,[443]3.2995,[444]3.3002,[445]3.3011,[446]3.3052,[447]3.3086,[448]3.3108,[449]3.3137,[450]3.3174,[451]3.3201,[452]3.3221,[453]3.3237,[454]3.3223,[455]3.3248,[456]3.3250,[457]3.3274,[458]3.3324,[459]3.3327,[460]3.3328,[461]3.3296,[462]3.3332,[463]3.3404,[464]3.3456,[465]3.3391,[466]3.3371,[467]3.3352,[468]3.3366,[469]3.3339,[470]3.3313,[471]3.3317,[472]3.3325,[473]3.3316,[474]3.3305,[475]3.3315,[476]3.3304,[477]3.3295,[478]3.3301,[479]3.3316,[480]3.3341,[481]3.3304,[482]3.3339,[483]3.3334,[484]3.3369,[485]3.3428,[486]3.3461,[487]3.3495,[488]3.3550,[489]3.3575,[490]3.3626,[491]3.3687,[492]3.3732,[493]3.3730,[494]3.3741,[495]3.3762,[496]3.3781,[497]3.3809,[498]3.3814,[499]3.3810,[500]3.3848,[501]3.3892,[502]3.3883,[503]3.3870,[504]3.3888,[505]3.3918,[506]3.3999,[507]3.4030,[508]3.4065,[509]3.3990,[510]3.3941,[511]3.3880,[512]3.3837,[513]3.3780,[514]3.3765,[515]3.3785,[516]3.3735,[517]3.3735,[518]3.3724,[519]3.3725,[520]3.3764,[521]3.3751,[522]3.3735,[523]3.3789,[524]3.3778,[525]3.3762,[526]3.3717,[527]3.3665,[528]3.3636,[529]3.3604,[530]3.3576,[531]3.3545,[532]3.3490,[533]3.3432,[534]3.3388,[535]3.3392,[536]3.3418,[537]3.3449,[538]3.3475,[539]3.3500,[540]3.3552,[541]3.3583,[542]3.3606,[543]3.3552,[544]3.3510,[545]3.3506,[546]3.3443,[547]3.3382,[548]3.3318,[549]3.3255,[550]3.3199,[551]3.3139,[552]3.3083,[553]3.3027,[554]3.3008,[555]3.2993,[556]3.3020,[557]3.3058,[558]3.3116,[559]3.3158,[560]3.3212,[561]3.3193,
llama_print_timings:        load time =  225352.00 ms
llama_print_timings:      sample time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings: prompt eval time = 2556352.12 ms / 287232 tokens (    8.90 ms per token,   112.36 tokens per second)
llama_print_timings:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings:       total time = 2599092.64 ms / 287233 tokens

Final estimate: PPL = 3.3193 +/- 0.01830
```

## Quantization
```bash
#!/usr/bin/env bash

# Notes:
# https://github.com/ikawrakow/ik_llama.cpp/issues/296#issuecomment-2765210993
# https://github.com/ikawrakow/ik_llama.cpp/issues/296#issuecomment-2768567062
custom="
# Token embedding and output tensors
# note token_embd cannot be repacked quant type e.g. `*_r4`
token_embd\.weight=iq6_k
output\.weight=iq5_k_r4
output_norm\.weight=iq5_k_r4

# First 3 dense layers (0-3)
blk\.[0-2]\.attn_k_b.*=q6_0_r4
blk\.[0-2]\.attn_.*=iq5_k_r4
blk\.[0-2]\..*=iq5_k_r4

# All attention, norm weights, and bias tensors for MoE layers (3-60)
# Except blk.*.attn_k_b.weight is not divisible by 256, so no iq6_k, so go with q6_0_r4
blk\.[3-9]\.attn_k_b.*=q6_0_r4
blk\.[1-5][0-9]\.attn_k_b.*=q6_0_r4
blk\.60\.attn_k_b.*=q6_0_r4

blk\.[3-9]\.attn_.*=iq5_k_r4
blk\.[1-5][0-9]\.attn_.*=iq5_k_r4
blk\.60\.attn_.*=iq5_k_r4

blk\.[3-9]\.ffn_norm\.weight=iq5_k_r4
blk\.[1-5][0-9]\.ffn_norm\.weight=iq5_k_r4
blk\.60\.ffn_norm\.weight=iq5_k_r4

blk\.[3-9]\.exp_probs_b\.bias=iq5_k_r4
blk\.[1-5][0-9]\.exp_probs_b\.bias=iq5_k_r4
blk\.60\.exp_probs_b\.bias=iq5_k_r4

# Shared Experts (3-60)
blk\.[3-9]\.ffn_down_shexp\.weight=iq5_k_r4
blk\.[1-5][0-9]\.ffn_down_shexp\.weight=iq5_k_r4
blk\.60\.ffn_down_shexp\.weight=iq5_k_r4

blk\.[3-9]\.ffn_(gate|up)_shexp\.weight=iq5_k_r4
blk\.[1-5][0-9]\.ffn_(gate|up)_shexp\.weight=iq5_k_r4
blk\.60\.ffn_(gate|up)_shexp\.weight=iq5_k_r4

# Routed Experts (3-60)
# First ~16 layers are more sensitive so keep larger
blk\.[3-9]\.ffn_down_exps\.weight=iq5_k_r4
blk\.[1][0-9]\.ffn_down_exps\.weight=iq5_k_r4
blk\.[2-5][0-9]\.ffn_down_exps\.weight=iq4_k_r4
blk\.60\.ffn_down_exps\.weight=iq4_k_r4

blk\.[3-9]\.ffn_(gate|up)_exps\.weight=iq4_k_r4
blk\.[1][0-9]\.ffn_(gate|up)_exps\.weight=iq4_k_r4
blk\.[2-5][0-9]\.ffn_(gate|up)_exps\.weight=iq3_k_r4
blk\.60\.ffn_(gate|up)_exps\.weight=iq3_k_r4
"
custom=$(
  echo "$custom" | grep -v '^#' | \
  sed -Ez 's:\n+:,:g;s:,$::;s:^,::'
)

./build/bin/llama-quantize \
    --imatrix /mnt/raid/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324.imatrix \
    --token-embedding-type iq6_k \
    --output-tensor-type iq5_k_r4 \
    --custom-q "$custom" \
    /mnt/raid/models/deepseek-ai/DeepSeek-V3-0324-bf16-GGUF/DeepSeek-256x21B-V3-0324-BF16-00001-of-00030.gguf \
    /mnt/raid/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-CPU-IQ3_K.gguf \
    IQ3_K \
    24
```

</details>

@saood06 

> You can use the script included to plot them together with the legend using the filenames.

Ahh yes, I see, got the script working like so:
```bash
$ uv venv ./venv --python 3.12 --python-preference=only-managed
$ source ./venv/bin/activate
$ uv pip install pandas matplotlib
$ python ./examples/sweep-bench/sweep-bench-plot.py \
    DeepSeek-V3-0324-CPU-IQ3_K_R4-tb128-t88-amb1024.md \
    DeepSeek-V3-0324-CPU-IQ3_K_R4-tb128-t128-amb1024.md \
    DeepSeek-V3-0324-CPU-IQ3_K_R4-tb128-t88-amb1536.md
```

---

@ikawrakow 

> I'm almost sure the TG peaks are due to number of threads. If you try with 128 TG threads, performance will be slightly lower at zero context, but for large contexts it should match the peaks for all context lengths.

I used saood06's script above to graph these three configurations. The variables between the runs are:
* `--threads` either 88 or 128
* `-amb` either 1024 or 1536

I left `--threads-batch` constant at 128 using single socket of Intel Xeon 6980P (with numactl).

#### pp

![Image](https://github.com/user-attachments/assets/8cce45da-7c64-4a20-b359-7308e58410a6)

#### tg

![Image](https://github.com/user-attachments/assets/5a81f755-8baa-4ab2-bb11-65df90943ba5)

## Observations

* With tg threads 88 the bumps in speed occur at the same place for both `-amb 1024` and `-amb 1536`.
* Raising tg threads to 128 seems slightly worse with no bumps in speed.
* Oddly pp had some variability between the runs despite keeping `--threads-batch 128` constant

I'm not sure what to try next. I could:
* play with `numactl --interleave=all llama-sweep-bench --numa distribute` and pump up threads to 256 (each CPU has 128 physical cores).
* try varying `--threads` to other multiples of 8 e.g. 64,72,80, ,96 to see if it effects the tg bump
* explore perplexity/speed trade-off using smaller quant vs `-ser 6,1`

That's all for now. Below are just the swee-bench logs for reference. Thanks!

## Logs

<details>

<summary>llama-sweep-bench logs and raw data</summary>

```bash
## pp 128 threads, tg 88 threads, amb 1024
numactl -N 0 -m 0 \
./build/bin/llama-sweep-bench \
    --model /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-CPU-IQ3_K_R4.gguf \
    --no-mmap \
    -ctk q8_0 \
    -mla 3 -fa \
    -amb 1024 \
    -fmoe \
    -c 32768 \
    -ub 512 \
    --threads 88 \
    --threads-batch 128 \
    --numa numactl

Current power profile is: performance
Current THP enabled and defrag configs are:
[always] madvise never
[always] defer defer+madvise madvise never
Set numa balancing to be:
0
llama_model_loader: loaded meta data with 50 key-value pairs and 1147 tensors from /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-CPU-
IQ3_K_R4.gguf (version GGUF V3 (latest))

llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type iq6_k:    1 tensors
llama_model_loader: - type q6_0_r4:   61 tensors
llama_model_loader: - type iq3_k_r4:   82 tensors
llama_model_loader: - type iq4_k_r4:   75 tensors
llama_model_loader: - type iq5_k_r4:  567 tensors

llm_load_print_meta: model type       = 671B
llm_load_print_meta: model ftype      = IQ3_K - 3.4325 bpw
llm_load_print_meta: model params     = 672.050 B
llm_load_print_meta: model size       = 324.011 GiB (4.141 BPW)
llm_load_print_meta: repeating layers = 322.703 GiB (4.136 BPW, 670.196 B parameters)
llm_load_print_meta: general.name     = DeepSeek V3 0324

llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 3
llama_new_context_with_model: attn_max_b = 1024
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025

llama_kv_cache_init:        CPU KV buffer size =  1166.63 MiB
llama_new_context_with_model: KV self size  = 1166.62 MiB, c^KV (q8_0): 1166.62 MiB, kv^T: not used
llama_new_context_with_model:        CPU  output buffer size =     0.49 MiB
llama_new_context_with_model:        CPU compute buffer size =  2662.01 MiB
llama_new_context_with_model: graph nodes  = 5500
llama_new_context_with_model: graph splits = 1

main: n_kv_max = 32768, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = -1, n_threads = 88, n_threads_batch = 128

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    4.705 |   108.82 |   11.986 |    10.68 |
|   512 |    128 |    512 |    4.756 |   107.65 |   12.792 |    10.01 |
|   512 |    128 |   1024 |    5.161 |    99.20 |   12.700 |    10.08 |
|   512 |    128 |   1536 |    5.728 |    89.39 |   12.775 |    10.02 |
|   512 |    128 |   2048 |    5.682 |    90.11 |   12.947 |     9.89 |
|   512 |    128 |   2560 |    6.333 |    80.84 |   14.947 |     8.56 |
|   512 |    128 |   3072 |    6.517 |    78.57 |   13.199 |     9.70 |
|   512 |    128 |   3584 |    6.776 |    75.56 |   13.677 |     9.36 |
|   512 |    128 |   4096 |    7.022 |    72.92 |   13.826 |     9.26 |
|   512 |    128 |   4608 |    7.585 |    67.51 |   13.937 |     9.18 |
|   512 |    128 |   5120 |    9.009 |    56.83 |   14.367 |     8.91 |
|   512 |    128 |   5632 |    8.190 |    62.51 |   14.409 |     8.88 |
|   512 |    128 |   6144 |    8.799 |    58.19 |   14.651 |     8.74 |
|   512 |    128 |   6656 |    9.711 |    52.72 |   14.788 |     8.66 |
|   512 |    128 |   7168 |    9.143 |    56.00 |   15.070 |     8.49 |
|   512 |    128 |   7680 |    9.905 |    51.69 |   15.394 |     8.31 |
|   512 |    128 |   8192 |    9.458 |    54.14 |   16.353 |     7.83 |
|   512 |    128 |   8704 |   10.134 |    50.52 |   15.867 |     8.07 |
|   512 |    128 |   9216 |   10.179 |    50.30 |   16.088 |     7.96 |
|   512 |    128 |   9728 |   10.385 |    49.30 |   16.817 |     7.61 |
|   512 |    128 |  10240 |   10.765 |    47.56 |   17.119 |     7.48 |
|   512 |    128 |  10752 |   10.896 |    46.99 |   17.115 |     7.48 |
|   512 |    128 |  11264 |   11.317 |    45.24 |   17.280 |     7.41 |
|   512 |    128 |  11776 |   11.461 |    44.67 |   17.702 |     7.23 |
|   512 |    128 |  12288 |   12.248 |    41.80 |   18.129 |     7.06 |
|   512 |    128 |  12800 |   12.176 |    42.05 |   18.294 |     7.00 |
|   512 |    128 |  13312 |   12.296 |    41.64 |   18.273 |     7.00 |
|   512 |    128 |  13824 |   13.446 |    38.08 |   17.938 |     7.14 |
|   512 |    128 |  14336 |   13.376 |    38.28 |   19.027 |     6.73 |
|   512 |    128 |  14848 |   13.901 |    36.83 |   19.547 |     6.55 |
|   512 |    128 |  15360 |   13.727 |    37.30 |   19.853 |     6.45 |
|   512 |    128 |  15872 |   14.168 |    36.14 |   20.259 |     6.32 |
|   512 |    128 |  16384 |   14.756 |    34.70 |   20.206 |     6.33 |
|   512 |    128 |  16896 |   15.237 |    33.60 |   20.719 |     6.18 |
|   512 |    128 |  17408 |   15.027 |    34.07 |   20.608 |     6.21 |
|   512 |    128 |  17920 |   15.585 |    32.85 |   21.305 |     6.01 |
|   512 |    128 |  18432 |   15.882 |    32.24 |   21.786 |     5.88 |
|   512 |    128 |  18944 |   16.613 |    30.82 |   22.082 |     5.80 |
|   512 |    128 |  19456 |   16.195 |    31.61 |   18.518 |     6.91 |
|   512 |    128 |  19968 |   17.213 |    29.75 |   22.846 |     5.60 |
|   512 |    128 |  20480 |   17.539 |    29.19 |   22.746 |     5.63 |
|   512 |    128 |  20992 |   17.368 |    29.48 |   23.104 |     5.54 |
|   512 |    128 |  21504 |   17.592 |    29.10 |   23.148 |     5.53 |
|   512 |    128 |  22016 |   17.977 |    28.48 |   23.651 |     5.41 |
|   512 |    128 |  22528 |   18.229 |    28.09 |   23.878 |     5.36 |
|   512 |    128 |  23040 |   18.590 |    27.54 |   24.244 |     5.28 |
|   512 |    128 |  23552 |   19.303 |    26.52 |   24.274 |     5.27 |
|   512 |    128 |  24064 |   19.662 |    26.04 |   25.586 |     5.00 |
|   512 |    128 |  24576 |   20.019 |    25.58 |   25.427 |     5.03 |
|   512 |    128 |  25088 |   20.519 |    24.95 |   19.775 |     6.47 |
|   512 |    128 |  25600 |   20.427 |    25.06 |   26.742 |     4.79 |
|   512 |    128 |  26112 |   20.727 |    24.70 |   26.280 |     4.87 |
|   512 |    128 |  26624 |   20.837 |    24.57 |   27.207 |     4.70 |
|   512 |    128 |  27136 |   21.536 |    23.77 |   27.221 |     4.70 |
|   512 |    128 |  27648 |   21.512 |    23.80 |   27.161 |     4.71 |
|   512 |    128 |  28160 |   21.916 |    23.36 |   27.883 |     4.59 |
|   512 |    128 |  28672 |   22.764 |    22.49 |   27.623 |     4.63 |
|   512 |    128 |  29184 |   22.665 |    22.59 |   28.389 |     4.51 |
|   512 |    128 |  29696 |   23.483 |    21.80 |   28.581 |     4.48 |
|   512 |    128 |  30208 |   23.785 |    21.53 |   28.538 |     4.49 |
|   512 |    128 |  30720 |   24.100 |    21.24 |   21.589 |     5.93 |
|   512 |    128 |  31232 |   24.275 |    21.09 |   29.526 |     4.34 |
|   512 |    128 |  31744 |   24.416 |    20.97 |   28.978 |     4.42 |
|   512 |    128 |  32256 |   25.127 |    20.38 |   28.427 |     4.50 |

---

## pp 128 threads, tg 128 threads, amb 1024
numactl -N 0 -m 0 \
./build/bin/llama-sweep-bench \
    --model /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-CPU-IQ3_K_R4.gguf \
    --no-mmap \
    -ctk q8_0 \
    -mla 3 -fa \
    -amb 1024 \
    -fmoe \
    -c 32768 \
    -ub 512 \
    --threads 128 \
    --threads-batch 128 \
    --numa numactl

llm_load_tensors:        CPU buffer size = 331786.93 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 3
llama_new_context_with_model: attn_max_b = 1024
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025

llama_kv_cache_init:        CPU KV buffer size =  1166.63 MiB
llama_new_context_with_model: KV self size  = 1166.62 MiB, c^KV (q8_0): 1166.62 MiB, kv^T: not used
llama_new_context_with_model:        CPU  output buffer size =     0.49 MiB
llama_new_context_with_model:        CPU compute buffer size =  2662.01 MiB
llama_new_context_with_model: graph nodes  = 5500
llama_new_context_with_model: graph splits = 1

main: n_kv_max = 32768, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = -1, n_threads = 128, n_threads_batch = 128

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    3.779 |   135.47 |   13.193 |     9.70 |
|   512 |    128 |    512 |    4.045 |   126.57 |   13.382 |     9.56 |
|   512 |    128 |   1024 |    4.369 |   117.19 |   13.530 |     9.46 |
|   512 |    128 |   1536 |    4.770 |   107.33 |   13.700 |     9.34 |
|   512 |    128 |   2048 |    5.170 |    99.04 |   13.834 |     9.25 |
|   512 |    128 |   2560 |    5.480 |    93.42 |   13.874 |     9.23 |
|   512 |    128 |   3072 |    5.845 |    87.59 |   14.029 |     9.12 |
|   512 |    128 |   3584 |    6.176 |    82.90 |   14.164 |     9.04 |
|   512 |    128 |   4096 |    6.658 |    76.90 |   14.341 |     8.93 |
|   512 |    128 |   4608 |    6.973 |    73.42 |   14.519 |     8.82 |
|   512 |    128 |   5120 |    7.357 |    69.59 |   14.709 |     8.70 |
|   512 |    128 |   5632 |    7.727 |    66.26 |   14.921 |     8.58 |
|   512 |    128 |   6144 |    8.305 |    61.65 |   15.091 |     8.48 |
|   512 |    128 |   6656 |    8.449 |    60.60 |   15.324 |     8.35 |
|   512 |    128 |   7168 |    9.073 |    56.43 |   15.551 |     8.23 |
|   512 |    128 |   7680 |    9.224 |    55.51 |   15.783 |     8.11 |
|   512 |    128 |   8192 |    9.140 |    56.02 |   16.039 |     7.98 |
|   512 |    128 |   8704 |    9.140 |    56.02 |   16.306 |     7.85 |
|   512 |    128 |   9216 |    9.465 |    54.09 |   16.553 |     7.73 |
|   512 |    128 |   9728 |   10.000 |    51.20 |   16.827 |     7.61 |
|   512 |    128 |  10240 |   10.120 |    50.59 |   17.263 |     7.41 |
|   512 |    128 |  10752 |   10.410 |    49.18 |   17.336 |     7.38 |
|   512 |    128 |  11264 |   11.062 |    46.29 |   17.599 |     7.27 |
|   512 |    128 |  11776 |   11.012 |    46.49 |   17.861 |     7.17 |
|   512 |    128 |  12288 |   11.309 |    45.27 |   18.129 |     7.06 |
|   512 |    128 |  12800 |   11.971 |    42.77 |   18.366 |     6.97 |
|   512 |    128 |  13312 |   12.554 |    40.79 |   18.661 |     6.86 |
|   512 |    128 |  13824 |   12.917 |    39.64 |   18.894 |     6.77 |
|   512 |    128 |  14336 |   12.615 |    40.59 |   19.122 |     6.69 |
|   512 |    128 |  14848 |   13.540 |    37.81 |   19.439 |     6.58 |
|   512 |    128 |  15360 |   13.878 |    36.89 |   19.695 |     6.50 |
|   512 |    128 |  15872 |   14.107 |    36.30 |   20.001 |     6.40 |
|   512 |    128 |  16384 |   13.998 |    36.58 |   20.294 |     6.31 |
|   512 |    128 |  16896 |   14.100 |    36.31 |   20.600 |     6.21 |
|   512 |    128 |  17408 |   14.413 |    35.52 |   21.126 |     6.06 |
|   512 |    128 |  17920 |   14.795 |    34.61 |   21.591 |     5.93 |
|   512 |    128 |  18432 |   15.112 |    33.88 |   22.046 |     5.81 |
|   512 |    128 |  18944 |   16.007 |    31.99 |   22.389 |     5.72 |
|   512 |    128 |  19456 |   16.391 |    31.24 |   22.861 |     5.60 |
|   512 |    128 |  19968 |   16.073 |    31.85 |   23.214 |     5.51 |
|   512 |    128 |  20480 |   16.437 |    31.15 |   23.621 |     5.42 |
|   512 |    128 |  20992 |   16.814 |    30.45 |   24.032 |     5.33 |
|   512 |    128 |  21504 |   17.145 |    29.86 |   24.297 |     5.27 |
|   512 |    128 |  22016 |   18.069 |    28.34 |   24.443 |     5.24 |
|   512 |    128 |  22528 |   17.998 |    28.45 |   24.715 |     5.18 |
|   512 |    128 |  23040 |   18.518 |    27.65 |   25.119 |     5.10 |
|   512 |    128 |  23552 |   18.645 |    27.46 |   25.608 |     5.00 |
|   512 |    128 |  24064 |   19.016 |    26.93 |   26.009 |     4.92 |
|   512 |    128 |  24576 |   19.271 |    26.57 |   26.465 |     4.84 |
|   512 |    128 |  25088 |   19.655 |    26.05 |   26.904 |     4.76 |
|   512 |    128 |  25600 |   19.987 |    25.62 |   27.073 |     4.73 |
|   512 |    128 |  26112 |   20.322 |    25.19 |   27.443 |     4.66 |
|   512 |    128 |  26624 |   20.694 |    24.74 |   27.875 |     4.59 |
|   512 |    128 |  27136 |   20.961 |    24.43 |   28.282 |     4.53 |
|   512 |    128 |  27648 |   21.311 |    24.02 |   28.494 |     4.49 |
|   512 |    128 |  28160 |   21.620 |    23.68 |   28.750 |     4.45 |
|   512 |    128 |  28672 |   22.491 |    22.76 |   28.979 |     4.42 |
|   512 |    128 |  29184 |   22.813 |    22.44 |   29.399 |     4.35 |
|   512 |    128 |  29696 |   22.584 |    22.67 |   29.749 |     4.30 |
|   512 |    128 |  30208 |   22.926 |    22.33 |   30.058 |     4.26 |
|   512 |    128 |  30720 |   23.372 |    21.91 |   30.385 |     4.21 |
|   512 |    128 |  31232 |   23.479 |    21.81 |   30.789 |     4.16 |
|   512 |    128 |  31744 |   23.455 |    21.83 |   31.089 |     4.12 |
|   512 |    128 |  32256 |   24.589 |    20.82 |   31.422 |     4.07 |

---

## pp 128 threads, tg 128 threads, amb 1536

numactl -N 0 -m 0 \
./build/bin/llama-sweep-bench \
    --model /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-CPU-IQ3_K_R4.gguf \
    --no-mmap \
    -ctk q8_0 \
    -mla 3 -fa \
    -amb 1536 \
    -fmoe \
    -c 32768 \
    -ub 512 \
    --threads 88 \
    --threads-batch 128 \
    --numa numactl

llm_load_tensors:        CPU buffer size = 331786.93 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 3
llama_new_context_with_model: attn_max_b = 1536
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025

llama_kv_cache_init:        CPU KV buffer size =  1166.63 MiB
llama_new_context_with_model: KV self size  = 1166.62 MiB, c^KV (q8_0): 1166.62 MiB, kv^T: not used
llama_new_context_with_model:        CPU  output buffer size =     0.49 MiB
llama_new_context_with_model:        CPU compute buffer size =  2662.01 MiB
llama_new_context_with_model: graph nodes  = 5500
llama_new_context_with_model: graph splits = 1

main: n_kv_max = 32768, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = -1, n_threads = 88, n_threads_batch = 128

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    4.455 |   114.93 |   12.232 |    10.46 |
|   512 |    128 |    512 |    4.597 |   111.38 |   12.618 |    10.14 |
|   512 |    128 |   1024 |    4.789 |   106.91 |   12.856 |     9.96 |
|   512 |    128 |   1536 |    5.212 |    98.24 |   12.819 |     9.99 |
|   512 |    128 |   2048 |    5.514 |    92.85 |   13.029 |     9.82 |
|   512 |    128 |   2560 |    5.848 |    87.56 |   14.833 |     8.63 |
|   512 |    128 |   3072 |    6.283 |    81.49 |   13.322 |     9.61 |
|   512 |    128 |   3584 |    6.673 |    76.73 |   13.870 |     9.23 |
|   512 |    128 |   4096 |    7.769 |    65.90 |   14.078 |     9.09 |
|   512 |    128 |   4608 |    8.379 |    61.11 |   14.311 |     8.94 |
|   512 |    128 |   5120 |    7.530 |    67.99 |   14.187 |     9.02 |
|   512 |    128 |   5632 |    8.165 |    62.70 |   14.485 |     8.84 |
|   512 |    128 |   6144 |    8.587 |    59.63 |   14.747 |     8.68 |
|   512 |    128 |   6656 |    9.117 |    56.16 |   15.042 |     8.51 |
|   512 |    128 |   7168 |    9.610 |    53.28 |   15.254 |     8.39 |
|   512 |    128 |   7680 |    9.586 |    53.41 |   15.127 |     8.46 |
|   512 |    128 |   8192 |    9.961 |    51.40 |   15.912 |     8.04 |
|   512 |    128 |   8704 |   10.993 |    46.58 |   15.844 |     8.08 |
|   512 |    128 |   9216 |   10.423 |    49.12 |   16.107 |     7.95 |
|   512 |    128 |   9728 |   10.673 |    47.97 |   16.464 |     7.77 |
|   512 |    128 |  10240 |   11.141 |    45.96 |   16.899 |     7.57 |
|   512 |    128 |  10752 |   11.421 |    44.83 |   16.458 |     7.78 |
|   512 |    128 |  11264 |   14.421 |    35.50 |   17.190 |     7.45 |
|   512 |    128 |  11776 |   12.696 |    40.33 |   17.436 |     7.34 |
|   512 |    128 |  12288 |   12.079 |    42.39 |   17.327 |     7.39 |
|   512 |    128 |  12800 |   12.304 |    41.61 |   17.591 |     7.28 |
|   512 |    128 |  13312 |   13.400 |    38.21 |   17.857 |     7.17 |
|   512 |    128 |  13824 |   12.764 |    40.11 |   17.791 |     7.19 |
|   512 |    128 |  14336 |   13.515 |    37.88 |   18.744 |     6.83 |
|   512 |    128 |  14848 |   13.556 |    37.77 |   18.888 |     6.78 |
|   512 |    128 |  15360 |   13.925 |    36.77 |   19.552 |     6.55 |
|   512 |    128 |  15872 |   14.119 |    36.26 |   20.393 |     6.28 |
|   512 |    128 |  16384 |   14.246 |    35.94 |   20.078 |     6.38 |
|   512 |    128 |  16896 |   14.739 |    34.74 |   20.428 |     6.27 |
|   512 |    128 |  17408 |   15.744 |    32.52 |   21.013 |     6.09 |
|   512 |    128 |  17920 |   15.983 |    32.03 |   21.100 |     6.07 |
|   512 |    128 |  18432 |   16.247 |    31.51 |   21.502 |     5.95 |
|   512 |    128 |  18944 |   16.554 |    30.93 |   21.797 |     5.87 |
|   512 |    128 |  19456 |   16.923 |    30.25 |   18.987 |     6.74 |
|   512 |    128 |  19968 |   17.313 |    29.57 |   22.714 |     5.64 |
|   512 |    128 |  20480 |   17.972 |    28.49 |   22.245 |     5.75 |
|   512 |    128 |  20992 |   17.986 |    28.47 |   22.409 |     5.71 |
|   512 |    128 |  21504 |   18.304 |    27.97 |   23.061 |     5.55 |
|   512 |    128 |  22016 |   19.044 |    26.88 |   23.934 |     5.35 |
|   512 |    128 |  22528 |   19.563 |    26.17 |   23.447 |     5.46 |
|   512 |    128 |  23040 |   20.054 |    25.53 |   23.932 |     5.35 |
|   512 |    128 |  23552 |   20.210 |    25.33 |   24.398 |     5.25 |
|   512 |    128 |  24064 |   21.129 |    24.23 |   25.225 |     5.07 |
|   512 |    128 |  24576 |   19.675 |    26.02 |   25.531 |     5.01 |
|   512 |    128 |  25088 |   20.162 |    25.39 |   19.989 |     6.40 |
|   512 |    128 |  25600 |   20.685 |    24.75 |   25.551 |     5.01 |
|   512 |    128 |  26112 |   20.721 |    24.71 |   26.588 |     4.81 |
|   512 |    128 |  26624 |   20.997 |    24.38 |   27.079 |     4.73 |
|   512 |    128 |  27136 |   21.587 |    23.72 |   27.030 |     4.74 |
|   512 |    128 |  27648 |   22.148 |    23.12 |   27.153 |     4.71 |
|   512 |    128 |  28160 |   22.081 |    23.19 |   27.515 |     4.65 |
|   512 |    128 |  28672 |   22.620 |    22.64 |   27.332 |     4.68 |
|   512 |    128 |  29184 |   22.811 |    22.45 |   27.864 |     4.59 |
|   512 |    128 |  29696 |   22.791 |    22.47 |   28.755 |     4.45 |
|   512 |    128 |  30208 |   23.195 |    22.07 |   28.234 |     4.53 |
|   512 |    128 |  30720 |   23.924 |    21.40 |   21.459 |     5.96 |
|   512 |    128 |  31232 |   23.809 |    21.50 |   29.165 |     4.39 |
|   512 |    128 |  31744 |   23.712 |    21.59 |   29.106 |     4.40 |
|   512 |    128 |  32256 |   24.421 |    20.97 |   29.634 |     4.32 |
```

</details>

---

👤 **ubergarm** commented the **2025-04-05** at **15:58:02**:<br>

Okay, got my "CPU only speed blend" quant cooked, copied over, perplexity, and a few sweep-bench comparisons against itself with different threads and amb settings.

<details>

<summary>DeepSeek-V3-0324-CPU-IQ3_K_R4 "CPU only speed blend" mix</summary>

## tl;dr;

Mostly ~q6/iq5_k_r4 for embedding/attention/dense layers/shared experts. First 17 routed experts are down/(up|gate) iq5_k_r4/iq4_k_r4 and the remainder are iq4_k_r4/iq3_k_r4.

`PPL = 3.3193 +/- 0.01830`

```bash
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type iq6_k:    1 tensors
llama_model_loader: - type q6_0_r4:   61 tensors
llama_model_loader: - type iq3_k_r4:   82 tensors
llama_model_loader: - type iq4_k_r4:   75 tensors
llama_model_loader: - type iq5_k_r4:  567 tensors

llm_load_print_meta: model type       = 671B
llm_load_print_meta: model ftype      = IQ3_K - 3.4325 bpw
llm_load_print_meta: model params     = 672.050 B
llm_load_print_meta: model size       = 324.011 GiB (4.141 BPW)
llm_load_print_meta: repeating layers = 322.703 GiB (4.136 BPW, 670.196 B parameters)
llm_load_print_meta: general.name     = DeepSeek V3 0324
```

## Perplexity
```bash
numactl -N 1 -m 1 \
./build/bin/llama-perplexity \
    --model /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-CPU-IQ3_K_R4.gguf \
    -ctk q8_0 \
    -mla 3 -fa \
    -amb 512 \
    -fmoe \
    --ctx-size 512 \
    --ubatch-size 512 \
    -f wiki.test.raw \
    --seed 1337 \
    --numa numactl \
    --threads 128

main: build = 3622 (c616306a)
main: built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu
main: seed  = 1337

llama_kv_cache_init:        CPU KV buffer size =    72.91 MiB
llama_new_context_with_model: KV self size  =   72.91 MiB, c^KV (q8_0):   72.91 MiB, kv^T: not used
llama_new_context_with_model:        CPU  output buffer size =     1.97 MiB
llama_new_context_with_model:        CPU compute buffer size =   450.01 MiB
llama_new_context_with_model: graph nodes  = 3487
llama_new_context_with_model: graph splits = 1

system_info: n_threads = 128 / 512 | AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | 
perplexity: tokenizing the input ..
perplexity: tokenization took 885.253 ms
perplexity: calculating perplexity over 561 chunks, n_ctx=512, batch_size=2048, n_seq=4
perplexity: 18.52 seconds per pass - ETA 43.28 minutes
[1]2.5128,[2]3.1998,[3]2.3365,[4]1.9572,[5]1.7672,[6]1.6281,[7]1.5395,[8]1.4757,[9]1.4355,[10]1.3986,[11]1.3863,[12]1.4171,[13]1.4335,[14]1.5570,[15]1.6860,[16]1.7427,[17]1.9032,[18]2.0271,[19]1.9913,[20]1.9776,[21]2.0854,[22]2.0602,[23]2.0347,[24]2.0476,[25]2.0186,[26]1.9969,[27]2.0413,[28]2.0507,[29]2.0970,[30]2.1295,[31]2.1608,[32]2.1794,[33]2.2186,[34]2.2617,[35]2.3099,[36]2.3635,[37]2.3978,[38]2.4457,[39]2.4853,[40]2.5440,[41]2.5853,[42]2.5976,[43]2.6473,[44]2.6637,[45]2.7436,[46]2.7934,[47]2.7499,[48]2.7051,[49]2.6812,[50]2.6987,[51]2.7413,[52]2.7537,[53]2.8060,[54]2.8201,[55]2.8508,[56]2.8807,[57]2.8940,[58]2.9277,[59]2.9387,[60]2.9864,[61]3.0248,[62]3.0709,[63]3.1017,[64]3.1429,[65]3.1526,[66]3.1355,[67]3.1118,[68]3.1372,[69]3.1314,[70]3.1476,[71]3.1660,[72]3.1796,[73]3.1931,[74]3.2149,[75]3.1951,[76]3.1489,[77]3.1060,[78]3.1012,[79]3.0804,[80]3.0632,[81]3.0289,[82]3.0333,[83]3.0030,[84]2.9691,[85]2.9358,[86]2.9134,[87]2.9083,[88]2.8809,[89]2.8642,[90]2.8387,[91]2.8113,[92]2.7865,[93]2.7604,[94]2.7369,[95]2.7151,[96]2.7141,[97]2.7189,[98]2.7038,[99]2.6870,[100]2.6894,[101]2.6821,[102]2.6980,[103]2.7237,[104]2.7405,[105]2.7372,[106]2.7591,[107]2.7837,[108]2.8041,[109]2.8372,[110]2.8699,[111]2.8884,[112]2.8629,[113]2.8500,[114]2.8292,[115]2.8139,[116]2.8010,[117]2.7792,[118]2.7587,[119]2.7376,[120]2.7196,[121]2.7036,[122]2.6864,[123]2.6691,[124]2.6500,[125]2.6333,[126]2.6165,[127]2.6034,[128]2.5949,[129]2.5838,[130]2.5714,[131]2.5622,[132]2.5688,[133]2.5782,[134]2.5857,[135]2.5965,[136]2.6115,[137]2.6256,[138]2.6335,[139]2.6442,[140]2.6447,[141]2.6465,[142]2.6450,[143]2.6459,[144]2.6432,[145]2.6352,[146]2.6334,[147]2.6377,[148]2.6379,[149]2.6395,[150]2.6337,[151]2.6321,[152]2.6294,[153]2.6255,[154]2.6254,[155]2.6295,[156]2.6307,[157]2.6363,[158]2.6444,[159]2.6469,[160]2.6556,[161]2.6641,[162]2.6743,[163]2.6796,[164]2.6999,[165]2.7236,[166]2.7410,[167]2.7531,[168]2.7770,[169]2.7996,[170]2.8214,[171]2.8429,[172]2.8273,[173]2.8112,[174]2.7987,[175]2.7868,[176]2.7746,[177]2.7635,[178]2.7508,[179]2.7373,[180]2.7409,[181]2.7550,[182]2.7698,[183]2.7839,[184]2.7969,[185]2.8065,[186]2.8224,[187]2.8380,[188]2.8519,[189]2.8622,[190]2.8627,[191]2.8698,[192]2.8729,[193]2.8780,[194]2.8971,[195]2.9057,[196]2.9187,[197]2.9283,[198]2.9329,[199]2.9386,[200]2.9379,[201]2.9528,[202]2.9480,[203]2.9532,[204]2.9558,[205]2.9556,[206]2.9582,[207]2.9667,[208]2.9757,[209]2.9846,[210]2.9847,[211]2.9802,[212]2.9808,[213]2.9883,[214]2.9901,[215]2.9957,[216]2.9962,[217]2.9920,[218]2.9920,[219]2.9927,[220]2.9925,[221]2.9932,[222]2.9930,[223]2.9939,[224]2.9986,[225]3.0004,[226]2.9925,[227]2.9900,[228]2.9914,[229]2.9951,[230]3.0014,[231]3.0074,[232]2.9994,[233]2.9921,[234]2.9923,[235]2.9911,[236]2.9998,[237]3.0079,[238]3.0172,[239]3.0268,[240]3.0361,[241]3.0471,[242]3.0615,[243]3.0741,[244]3.0820,[245]3.0929,[246]3.1031,[247]3.1021,[248]3.0979,[249]3.0960,[250]3.0899,[251]3.0878,[252]3.0899,[253]3.0939,[254]3.1008,[255]3.1070,[256]3.1101,[257]3.1131,[258]3.1144,[259]3.1179,[260]3.1201,[261]3.1214,[262]3.1205,[263]3.1263,[264]3.1286,[265]3.1291,[266]3.1306,[267]3.1327,[268]3.1357,[269]3.1385,[270]3.1378,[271]3.1363,[272]3.1297,[273]3.1294,[274]3.1225,[275]3.1122,[276]3.1010,[277]3.1029,[278]3.1128,[279]3.1187,[280]3.1265,[281]3.1338,[282]3.1394,[283]3.1458,[284]3.1518,[285]3.1654,[286]3.1675,[287]3.1708,[288]3.1759,[289]3.1781,[290]3.1701,[291]3.1613,[292]3.1597,[293]3.1591,[294]3.1570,[295]3.1548,[296]3.1570,[297]3.1575,[298]3.1631,[299]3.1689,[300]3.1718,[301]3.1758,[302]3.1780,[303]3.1795,[304]3.1790,[305]3.1904,[306]3.1973,[307]3.2079,[308]3.1969,[309]3.1920,[310]3.1831,[311]3.1862,[312]3.1877,[313]3.1936,[314]3.1959,[315]3.1990,[316]3.2006,[317]3.2026,[318]3.2032,[319]3.2035,[320]3.2076,[321]3.2078,[322]3.2096,[323]3.2160,[324]3.2167,[325]3.2221,[326]3.2263,[327]3.2302,[328]3.2327,[329]3.2346,[330]3.2409,[331]3.2439,[332]3.2478,[333]3.2467,[334]3.2467,[335]3.2474,[336]3.2475,[337]3.2486,[338]3.2488,[339]3.2512,[340]3.2547,[341]3.2599,[342]3.2687,[343]3.2775,[344]3.2824,[345]3.2740,[346]3.2664,[347]3.2617,[348]3.2543,[349]3.2505,[350]3.2491,[351]3.2537,[352]3.2683,[353]3.2772,[354]3.2897,[355]3.2982,[356]3.3034,[357]3.3150,[358]3.3248,[359]3.3276,[360]3.3340,[361]3.3433,[362]3.3519,[363]3.3572,[364]3.3639,[365]3.3695,[366]3.3796,[367]3.3881,[368]3.3943,[369]3.4019,[370]3.4104,[371]3.4235,[372]3.4322,[373]3.4356,[374]3.4389,[375]3.4437,[376]3.4563,[377]3.4674,[378]3.4704,[379]3.4704,[380]3.4668,[381]3.4718,[382]3.4775,[383]3.4807,[384]3.4850,[385]3.4888,[386]3.4947,[387]3.5004,[388]3.5033,[389]3.4933,[390]3.4842,[391]3.4740,[392]3.4687,[393]3.4596,[394]3.4511,[395]3.4422,[396]3.4325,[397]3.4241,[398]3.4150,[399]3.4048,[400]3.3963,[401]3.3865,[402]3.3766,[403]3.3683,[404]3.3584,[405]3.3492,[406]3.3398,[407]3.3307,[408]3.3220,[409]3.3136,[410]3.3076,[411]3.3086,[412]3.3038,[413]3.3059,[414]3.3075,[415]3.3050,[416]3.3052,[417]3.3071,[418]3.3014,[419]3.3026,[420]3.3000,[421]3.2989,[422]3.2994,[423]3.2989,[424]3.3026,[425]3.3024,[426]3.3029,[427]3.3019,[428]3.3043,[429]3.3055,[430]3.3082,[431]3.3091,[432]3.3081,[433]3.3046,[434]3.3051,[435]3.2979,[436]3.2921,[437]3.2881,[438]3.2863,[439]3.2839,[440]3.2887,[441]3.2943,[442]3.3014,[443]3.2995,[444]3.3002,[445]3.3011,[446]3.3052,[447]3.3086,[448]3.3108,[449]3.3137,[450]3.3174,[451]3.3201,[452]3.3221,[453]3.3237,[454]3.3223,[455]3.3248,[456]3.3250,[457]3.3274,[458]3.3324,[459]3.3327,[460]3.3328,[461]3.3296,[462]3.3332,[463]3.3404,[464]3.3456,[465]3.3391,[466]3.3371,[467]3.3352,[468]3.3366,[469]3.3339,[470]3.3313,[471]3.3317,[472]3.3325,[473]3.3316,[474]3.3305,[475]3.3315,[476]3.3304,[477]3.3295,[478]3.3301,[479]3.3316,[480]3.3341,[481]3.3304,[482]3.3339,[483]3.3334,[484]3.3369,[485]3.3428,[486]3.3461,[487]3.3495,[488]3.3550,[489]3.3575,[490]3.3626,[491]3.3687,[492]3.3732,[493]3.3730,[494]3.3741,[495]3.3762,[496]3.3781,[497]3.3809,[498]3.3814,[499]3.3810,[500]3.3848,[501]3.3892,[502]3.3883,[503]3.3870,[504]3.3888,[505]3.3918,[506]3.3999,[507]3.4030,[508]3.4065,[509]3.3990,[510]3.3941,[511]3.3880,[512]3.3837,[513]3.3780,[514]3.3765,[515]3.3785,[516]3.3735,[517]3.3735,[518]3.3724,[519]3.3725,[520]3.3764,[521]3.3751,[522]3.3735,[523]3.3789,[524]3.3778,[525]3.3762,[526]3.3717,[527]3.3665,[528]3.3636,[529]3.3604,[530]3.3576,[531]3.3545,[532]3.3490,[533]3.3432,[534]3.3388,[535]3.3392,[536]3.3418,[537]3.3449,[538]3.3475,[539]3.3500,[540]3.3552,[541]3.3583,[542]3.3606,[543]3.3552,[544]3.3510,[545]3.3506,[546]3.3443,[547]3.3382,[548]3.3318,[549]3.3255,[550]3.3199,[551]3.3139,[552]3.3083,[553]3.3027,[554]3.3008,[555]3.2993,[556]3.3020,[557]3.3058,[558]3.3116,[559]3.3158,[560]3.3212,[561]3.3193,
llama_print_timings:        load time =  225352.00 ms
llama_print_timings:      sample time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings: prompt eval time = 2556352.12 ms / 287232 tokens (    8.90 ms per token,   112.36 tokens per second)
llama_print_timings:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings:       total time = 2599092.64 ms / 287233 tokens

Final estimate: PPL = 3.3193 +/- 0.01830
```

## Quantization
```bash
#!/usr/bin/env bash

# Notes:
# https://github.com/ikawrakow/ik_llama.cpp/issues/296#issuecomment-2765210993
# https://github.com/ikawrakow/ik_llama.cpp/issues/296#issuecomment-2768567062
custom="
# Token embedding and output tensors
# note token_embd cannot be repacked quant type e.g. `*_r4`
token_embd\.weight=iq6_k
output\.weight=iq5_k_r4
output_norm\.weight=iq5_k_r4

# First 3 dense layers (0-3)
blk\.[0-2]\.attn_k_b.*=q6_0_r4
blk\.[0-2]\.attn_.*=iq5_k_r4
blk\.[0-2]\..*=iq5_k_r4

# All attention, norm weights, and bias tensors for MoE layers (3-60)
# Except blk.*.attn_k_b.weight is not divisible by 256, so no iq6_k, so go with q6_0_r4
blk\.[3-9]\.attn_k_b.*=q6_0_r4
blk\.[1-5][0-9]\.attn_k_b.*=q6_0_r4
blk\.60\.attn_k_b.*=q6_0_r4

blk\.[3-9]\.attn_.*=iq5_k_r4
blk\.[1-5][0-9]\.attn_.*=iq5_k_r4
blk\.60\.attn_.*=iq5_k_r4

blk\.[3-9]\.ffn_norm\.weight=iq5_k_r4
blk\.[1-5][0-9]\.ffn_norm\.weight=iq5_k_r4
blk\.60\.ffn_norm\.weight=iq5_k_r4

blk\.[3-9]\.exp_probs_b\.bias=iq5_k_r4
blk\.[1-5][0-9]\.exp_probs_b\.bias=iq5_k_r4
blk\.60\.exp_probs_b\.bias=iq5_k_r4

# Shared Experts (3-60)
blk\.[3-9]\.ffn_down_shexp\.weight=iq5_k_r4
blk\.[1-5][0-9]\.ffn_down_shexp\.weight=iq5_k_r4
blk\.60\.ffn_down_shexp\.weight=iq5_k_r4

blk\.[3-9]\.ffn_(gate|up)_shexp\.weight=iq5_k_r4
blk\.[1-5][0-9]\.ffn_(gate|up)_shexp\.weight=iq5_k_r4
blk\.60\.ffn_(gate|up)_shexp\.weight=iq5_k_r4

# Routed Experts (3-60)
# First ~16 layers are more sensitive so keep larger
blk\.[3-9]\.ffn_down_exps\.weight=iq5_k_r4
blk\.[1][0-9]\.ffn_down_exps\.weight=iq5_k_r4
blk\.[2-5][0-9]\.ffn_down_exps\.weight=iq4_k_r4
blk\.60\.ffn_down_exps\.weight=iq4_k_r4

blk\.[3-9]\.ffn_(gate|up)_exps\.weight=iq4_k_r4
blk\.[1][0-9]\.ffn_(gate|up)_exps\.weight=iq4_k_r4
blk\.[2-5][0-9]\.ffn_(gate|up)_exps\.weight=iq3_k_r4
blk\.60\.ffn_(gate|up)_exps\.weight=iq3_k_r4
"
custom=$(
  echo "$custom" | grep -v '^#' | \
  sed -Ez 's:\n+:,:g;s:,$::;s:^,::'
)

./build/bin/llama-quantize \
    --imatrix /mnt/raid/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324.imatrix \
    --token-embedding-type iq6_k \
    --output-tensor-type iq5_k_r4 \
    --custom-q "$custom" \
    /mnt/raid/models/deepseek-ai/DeepSeek-V3-0324-bf16-GGUF/DeepSeek-256x21B-V3-0324-BF16-00001-of-00030.gguf \
    /mnt/raid/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-CPU-IQ3_K.gguf \
    IQ3_K \
    24
```

</details>

@saood06 

> You can use the script included to plot them together with the legend using the filenames.

Ahh yes, I see, got the script working like so:
```bash
$ uv venv ./venv --python 3.12 --python-preference=only-managed
$ source ./venv/bin/activate
$ uv pip install pandas matplotlib
$ python ./examples/sweep-bench/sweep-bench-plot.py \
    DeepSeek-V3-0324-CPU-IQ3_K_R4-tb128-t88-amb1024.md \
    DeepSeek-V3-0324-CPU-IQ3_K_R4-tb128-t128-amb1024.md \
    DeepSeek-V3-0324-CPU-IQ3_K_R4-tb128-t88-amb1536.md
```

---

@ikawrakow 

> I'm almost sure the TG peaks are due to number of threads. If you try with 128 TG threads, performance will be slightly lower at zero context, but for large contexts it should match the peaks for all context lengths.

I used saood06's script above to graph these three configurations. The variables between the runs are:
* `--threads` either 88 or 128
* `-amb` either 1024 or 1536

I left `--threads-batch` constant at 128 using single socket of Intel Xeon 6980P (with numactl).

#### pp

![Image](https://github.com/user-attachments/assets/8cce45da-7c64-4a20-b359-7308e58410a6)

#### tg

![Image](https://github.com/user-attachments/assets/5a81f755-8baa-4ab2-bb11-65df90943ba5)

## Observations

* With tg threads 88 the bumps in speed occur at the same place for both `-amb 1024` and `-amb 1536`.
* Raising tg threads to 128 seems slightly worse with no bumps in speed.
* Oddly pp had some variability between the runs despite keeping `--threads-batch 128` constant

I'm not sure what to try next. I could:
* play with `numactl --interleave=all llama-sweep-bench --numa distribute` and pump up threads to 256 (each CPU has 128 physical cores).
* try varying `--threads` to other multiples of 8 e.g. 64,72,80, ,96 to see if it effects the tg bump

That's all for now. Below are just the swee-bench logs for reference. Thanks!

## Logs

<details>

<summary>llama-sweep-bench logs and raw data</summary>

```bash
## pp 128 threads, tg 88 threads, amb 1024
numactl -N 0 -m 0 \
./build/bin/llama-sweep-bench \
    --model /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-CPU-IQ3_K_R4.gguf \
    --no-mmap \
    -ctk q8_0 \
    -mla 3 -fa \
    -amb 1024 \
    -fmoe \
    -c 32768 \
    -ub 512 \
    --threads 88 \
    --threads-batch 128 \
    --numa numactl

Current power profile is: performance
Current THP enabled and defrag configs are:
[always] madvise never
[always] defer defer+madvise madvise never
Set numa balancing to be:
0
llama_model_loader: loaded meta data with 50 key-value pairs and 1147 tensors from /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-CPU-
IQ3_K_R4.gguf (version GGUF V3 (latest))

llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type iq6_k:    1 tensors
llama_model_loader: - type q6_0_r4:   61 tensors
llama_model_loader: - type iq3_k_r4:   82 tensors
llama_model_loader: - type iq4_k_r4:   75 tensors
llama_model_loader: - type iq5_k_r4:  567 tensors

llm_load_print_meta: model type       = 671B
llm_load_print_meta: model ftype      = IQ3_K - 3.4325 bpw
llm_load_print_meta: model params     = 672.050 B
llm_load_print_meta: model size       = 324.011 GiB (4.141 BPW)
llm_load_print_meta: repeating layers = 322.703 GiB (4.136 BPW, 670.196 B parameters)
llm_load_print_meta: general.name     = DeepSeek V3 0324

llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 3
llama_new_context_with_model: attn_max_b = 1024
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025

llama_kv_cache_init:        CPU KV buffer size =  1166.63 MiB
llama_new_context_with_model: KV self size  = 1166.62 MiB, c^KV (q8_0): 1166.62 MiB, kv^T: not used
llama_new_context_with_model:        CPU  output buffer size =     0.49 MiB
llama_new_context_with_model:        CPU compute buffer size =  2662.01 MiB
llama_new_context_with_model: graph nodes  = 5500
llama_new_context_with_model: graph splits = 1

main: n_kv_max = 32768, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = -1, n_threads = 88, n_threads_batch = 128

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    4.705 |   108.82 |   11.986 |    10.68 |
|   512 |    128 |    512 |    4.756 |   107.65 |   12.792 |    10.01 |
|   512 |    128 |   1024 |    5.161 |    99.20 |   12.700 |    10.08 |
|   512 |    128 |   1536 |    5.728 |    89.39 |   12.775 |    10.02 |
|   512 |    128 |   2048 |    5.682 |    90.11 |   12.947 |     9.89 |
|   512 |    128 |   2560 |    6.333 |    80.84 |   14.947 |     8.56 |
|   512 |    128 |   3072 |    6.517 |    78.57 |   13.199 |     9.70 |
|   512 |    128 |   3584 |    6.776 |    75.56 |   13.677 |     9.36 |
|   512 |    128 |   4096 |    7.022 |    72.92 |   13.826 |     9.26 |
|   512 |    128 |   4608 |    7.585 |    67.51 |   13.937 |     9.18 |
|   512 |    128 |   5120 |    9.009 |    56.83 |   14.367 |     8.91 |
|   512 |    128 |   5632 |    8.190 |    62.51 |   14.409 |     8.88 |
|   512 |    128 |   6144 |    8.799 |    58.19 |   14.651 |     8.74 |
|   512 |    128 |   6656 |    9.711 |    52.72 |   14.788 |     8.66 |
|   512 |    128 |   7168 |    9.143 |    56.00 |   15.070 |     8.49 |
|   512 |    128 |   7680 |    9.905 |    51.69 |   15.394 |     8.31 |
|   512 |    128 |   8192 |    9.458 |    54.14 |   16.353 |     7.83 |
|   512 |    128 |   8704 |   10.134 |    50.52 |   15.867 |     8.07 |
|   512 |    128 |   9216 |   10.179 |    50.30 |   16.088 |     7.96 |
|   512 |    128 |   9728 |   10.385 |    49.30 |   16.817 |     7.61 |
|   512 |    128 |  10240 |   10.765 |    47.56 |   17.119 |     7.48 |
|   512 |    128 |  10752 |   10.896 |    46.99 |   17.115 |     7.48 |
|   512 |    128 |  11264 |   11.317 |    45.24 |   17.280 |     7.41 |
|   512 |    128 |  11776 |   11.461 |    44.67 |   17.702 |     7.23 |
|   512 |    128 |  12288 |   12.248 |    41.80 |   18.129 |     7.06 |
|   512 |    128 |  12800 |   12.176 |    42.05 |   18.294 |     7.00 |
|   512 |    128 |  13312 |   12.296 |    41.64 |   18.273 |     7.00 |
|   512 |    128 |  13824 |   13.446 |    38.08 |   17.938 |     7.14 |
|   512 |    128 |  14336 |   13.376 |    38.28 |   19.027 |     6.73 |
|   512 |    128 |  14848 |   13.901 |    36.83 |   19.547 |     6.55 |
|   512 |    128 |  15360 |   13.727 |    37.30 |   19.853 |     6.45 |
|   512 |    128 |  15872 |   14.168 |    36.14 |   20.259 |     6.32 |
|   512 |    128 |  16384 |   14.756 |    34.70 |   20.206 |     6.33 |
|   512 |    128 |  16896 |   15.237 |    33.60 |   20.719 |     6.18 |
|   512 |    128 |  17408 |   15.027 |    34.07 |   20.608 |     6.21 |
|   512 |    128 |  17920 |   15.585 |    32.85 |   21.305 |     6.01 |
|   512 |    128 |  18432 |   15.882 |    32.24 |   21.786 |     5.88 |
|   512 |    128 |  18944 |   16.613 |    30.82 |   22.082 |     5.80 |
|   512 |    128 |  19456 |   16.195 |    31.61 |   18.518 |     6.91 |
|   512 |    128 |  19968 |   17.213 |    29.75 |   22.846 |     5.60 |
|   512 |    128 |  20480 |   17.539 |    29.19 |   22.746 |     5.63 |
|   512 |    128 |  20992 |   17.368 |    29.48 |   23.104 |     5.54 |
|   512 |    128 |  21504 |   17.592 |    29.10 |   23.148 |     5.53 |
|   512 |    128 |  22016 |   17.977 |    28.48 |   23.651 |     5.41 |
|   512 |    128 |  22528 |   18.229 |    28.09 |   23.878 |     5.36 |
|   512 |    128 |  23040 |   18.590 |    27.54 |   24.244 |     5.28 |
|   512 |    128 |  23552 |   19.303 |    26.52 |   24.274 |     5.27 |
|   512 |    128 |  24064 |   19.662 |    26.04 |   25.586 |     5.00 |
|   512 |    128 |  24576 |   20.019 |    25.58 |   25.427 |     5.03 |
|   512 |    128 |  25088 |   20.519 |    24.95 |   19.775 |     6.47 |
|   512 |    128 |  25600 |   20.427 |    25.06 |   26.742 |     4.79 |
|   512 |    128 |  26112 |   20.727 |    24.70 |   26.280 |     4.87 |
|   512 |    128 |  26624 |   20.837 |    24.57 |   27.207 |     4.70 |
|   512 |    128 |  27136 |   21.536 |    23.77 |   27.221 |     4.70 |
|   512 |    128 |  27648 |   21.512 |    23.80 |   27.161 |     4.71 |
|   512 |    128 |  28160 |   21.916 |    23.36 |   27.883 |     4.59 |
|   512 |    128 |  28672 |   22.764 |    22.49 |   27.623 |     4.63 |
|   512 |    128 |  29184 |   22.665 |    22.59 |   28.389 |     4.51 |
|   512 |    128 |  29696 |   23.483 |    21.80 |   28.581 |     4.48 |
|   512 |    128 |  30208 |   23.785 |    21.53 |   28.538 |     4.49 |
|   512 |    128 |  30720 |   24.100 |    21.24 |   21.589 |     5.93 |
|   512 |    128 |  31232 |   24.275 |    21.09 |   29.526 |     4.34 |
|   512 |    128 |  31744 |   24.416 |    20.97 |   28.978 |     4.42 |
|   512 |    128 |  32256 |   25.127 |    20.38 |   28.427 |     4.50 |

---

## pp 128 threads, tg 128 threads, amb 1024
numactl -N 0 -m 0 \
./build/bin/llama-sweep-bench \
    --model /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-CPU-IQ3_K_R4.gguf \
    --no-mmap \
    -ctk q8_0 \
    -mla 3 -fa \
    -amb 1024 \
    -fmoe \
    -c 32768 \
    -ub 512 \
    --threads 128 \
    --threads-batch 128 \
    --numa numactl

llm_load_tensors:        CPU buffer size = 331786.93 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 3
llama_new_context_with_model: attn_max_b = 1024
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025

llama_kv_cache_init:        CPU KV buffer size =  1166.63 MiB
llama_new_context_with_model: KV self size  = 1166.62 MiB, c^KV (q8_0): 1166.62 MiB, kv^T: not used
llama_new_context_with_model:        CPU  output buffer size =     0.49 MiB
llama_new_context_with_model:        CPU compute buffer size =  2662.01 MiB
llama_new_context_with_model: graph nodes  = 5500
llama_new_context_with_model: graph splits = 1

main: n_kv_max = 32768, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = -1, n_threads = 128, n_threads_batch = 128

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    3.779 |   135.47 |   13.193 |     9.70 |
|   512 |    128 |    512 |    4.045 |   126.57 |   13.382 |     9.56 |
|   512 |    128 |   1024 |    4.369 |   117.19 |   13.530 |     9.46 |
|   512 |    128 |   1536 |    4.770 |   107.33 |   13.700 |     9.34 |
|   512 |    128 |   2048 |    5.170 |    99.04 |   13.834 |     9.25 |
|   512 |    128 |   2560 |    5.480 |    93.42 |   13.874 |     9.23 |
|   512 |    128 |   3072 |    5.845 |    87.59 |   14.029 |     9.12 |
|   512 |    128 |   3584 |    6.176 |    82.90 |   14.164 |     9.04 |
|   512 |    128 |   4096 |    6.658 |    76.90 |   14.341 |     8.93 |
|   512 |    128 |   4608 |    6.973 |    73.42 |   14.519 |     8.82 |
|   512 |    128 |   5120 |    7.357 |    69.59 |   14.709 |     8.70 |
|   512 |    128 |   5632 |    7.727 |    66.26 |   14.921 |     8.58 |
|   512 |    128 |   6144 |    8.305 |    61.65 |   15.091 |     8.48 |
|   512 |    128 |   6656 |    8.449 |    60.60 |   15.324 |     8.35 |
|   512 |    128 |   7168 |    9.073 |    56.43 |   15.551 |     8.23 |
|   512 |    128 |   7680 |    9.224 |    55.51 |   15.783 |     8.11 |
|   512 |    128 |   8192 |    9.140 |    56.02 |   16.039 |     7.98 |
|   512 |    128 |   8704 |    9.140 |    56.02 |   16.306 |     7.85 |
|   512 |    128 |   9216 |    9.465 |    54.09 |   16.553 |     7.73 |
|   512 |    128 |   9728 |   10.000 |    51.20 |   16.827 |     7.61 |
|   512 |    128 |  10240 |   10.120 |    50.59 |   17.263 |     7.41 |
|   512 |    128 |  10752 |   10.410 |    49.18 |   17.336 |     7.38 |
|   512 |    128 |  11264 |   11.062 |    46.29 |   17.599 |     7.27 |
|   512 |    128 |  11776 |   11.012 |    46.49 |   17.861 |     7.17 |
|   512 |    128 |  12288 |   11.309 |    45.27 |   18.129 |     7.06 |
|   512 |    128 |  12800 |   11.971 |    42.77 |   18.366 |     6.97 |
|   512 |    128 |  13312 |   12.554 |    40.79 |   18.661 |     6.86 |
|   512 |    128 |  13824 |   12.917 |    39.64 |   18.894 |     6.77 |
|   512 |    128 |  14336 |   12.615 |    40.59 |   19.122 |     6.69 |
|   512 |    128 |  14848 |   13.540 |    37.81 |   19.439 |     6.58 |
|   512 |    128 |  15360 |   13.878 |    36.89 |   19.695 |     6.50 |
|   512 |    128 |  15872 |   14.107 |    36.30 |   20.001 |     6.40 |
|   512 |    128 |  16384 |   13.998 |    36.58 |   20.294 |     6.31 |
|   512 |    128 |  16896 |   14.100 |    36.31 |   20.600 |     6.21 |
|   512 |    128 |  17408 |   14.413 |    35.52 |   21.126 |     6.06 |
|   512 |    128 |  17920 |   14.795 |    34.61 |   21.591 |     5.93 |
|   512 |    128 |  18432 |   15.112 |    33.88 |   22.046 |     5.81 |
|   512 |    128 |  18944 |   16.007 |    31.99 |   22.389 |     5.72 |
|   512 |    128 |  19456 |   16.391 |    31.24 |   22.861 |     5.60 |
|   512 |    128 |  19968 |   16.073 |    31.85 |   23.214 |     5.51 |
|   512 |    128 |  20480 |   16.437 |    31.15 |   23.621 |     5.42 |
|   512 |    128 |  20992 |   16.814 |    30.45 |   24.032 |     5.33 |
|   512 |    128 |  21504 |   17.145 |    29.86 |   24.297 |     5.27 |
|   512 |    128 |  22016 |   18.069 |    28.34 |   24.443 |     5.24 |
|   512 |    128 |  22528 |   17.998 |    28.45 |   24.715 |     5.18 |
|   512 |    128 |  23040 |   18.518 |    27.65 |   25.119 |     5.10 |
|   512 |    128 |  23552 |   18.645 |    27.46 |   25.608 |     5.00 |
|   512 |    128 |  24064 |   19.016 |    26.93 |   26.009 |     4.92 |
|   512 |    128 |  24576 |   19.271 |    26.57 |   26.465 |     4.84 |
|   512 |    128 |  25088 |   19.655 |    26.05 |   26.904 |     4.76 |
|   512 |    128 |  25600 |   19.987 |    25.62 |   27.073 |     4.73 |
|   512 |    128 |  26112 |   20.322 |    25.19 |   27.443 |     4.66 |
|   512 |    128 |  26624 |   20.694 |    24.74 |   27.875 |     4.59 |
|   512 |    128 |  27136 |   20.961 |    24.43 |   28.282 |     4.53 |
|   512 |    128 |  27648 |   21.311 |    24.02 |   28.494 |     4.49 |
|   512 |    128 |  28160 |   21.620 |    23.68 |   28.750 |     4.45 |
|   512 |    128 |  28672 |   22.491 |    22.76 |   28.979 |     4.42 |
|   512 |    128 |  29184 |   22.813 |    22.44 |   29.399 |     4.35 |
|   512 |    128 |  29696 |   22.584 |    22.67 |   29.749 |     4.30 |
|   512 |    128 |  30208 |   22.926 |    22.33 |   30.058 |     4.26 |
|   512 |    128 |  30720 |   23.372 |    21.91 |   30.385 |     4.21 |
|   512 |    128 |  31232 |   23.479 |    21.81 |   30.789 |     4.16 |
|   512 |    128 |  31744 |   23.455 |    21.83 |   31.089 |     4.12 |
|   512 |    128 |  32256 |   24.589 |    20.82 |   31.422 |     4.07 |

---

## pp 128 threads, tg 128 threads, amb 1536

numactl -N 0 -m 0 \
./build/bin/llama-sweep-bench \
    --model /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-CPU-IQ3_K_R4.gguf \
    --no-mmap \
    -ctk q8_0 \
    -mla 3 -fa \
    -amb 1536 \
    -fmoe \
    -c 32768 \
    -ub 512 \
    --threads 88 \
    --threads-batch 128 \
    --numa numactl

llm_load_tensors:        CPU buffer size = 331786.93 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 3
llama_new_context_with_model: attn_max_b = 1536
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025

llama_kv_cache_init:        CPU KV buffer size =  1166.63 MiB
llama_new_context_with_model: KV self size  = 1166.62 MiB, c^KV (q8_0): 1166.62 MiB, kv^T: not used
llama_new_context_with_model:        CPU  output buffer size =     0.49 MiB
llama_new_context_with_model:        CPU compute buffer size =  2662.01 MiB
llama_new_context_with_model: graph nodes  = 5500
llama_new_context_with_model: graph splits = 1

main: n_kv_max = 32768, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = -1, n_threads = 88, n_threads_batch = 128

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    4.455 |   114.93 |   12.232 |    10.46 |
|   512 |    128 |    512 |    4.597 |   111.38 |   12.618 |    10.14 |
|   512 |    128 |   1024 |    4.789 |   106.91 |   12.856 |     9.96 |
|   512 |    128 |   1536 |    5.212 |    98.24 |   12.819 |     9.99 |
|   512 |    128 |   2048 |    5.514 |    92.85 |   13.029 |     9.82 |
|   512 |    128 |   2560 |    5.848 |    87.56 |   14.833 |     8.63 |
|   512 |    128 |   3072 |    6.283 |    81.49 |   13.322 |     9.61 |
|   512 |    128 |   3584 |    6.673 |    76.73 |   13.870 |     9.23 |
|   512 |    128 |   4096 |    7.769 |    65.90 |   14.078 |     9.09 |
|   512 |    128 |   4608 |    8.379 |    61.11 |   14.311 |     8.94 |
|   512 |    128 |   5120 |    7.530 |    67.99 |   14.187 |     9.02 |
|   512 |    128 |   5632 |    8.165 |    62.70 |   14.485 |     8.84 |
|   512 |    128 |   6144 |    8.587 |    59.63 |   14.747 |     8.68 |
|   512 |    128 |   6656 |    9.117 |    56.16 |   15.042 |     8.51 |
|   512 |    128 |   7168 |    9.610 |    53.28 |   15.254 |     8.39 |
|   512 |    128 |   7680 |    9.586 |    53.41 |   15.127 |     8.46 |
|   512 |    128 |   8192 |    9.961 |    51.40 |   15.912 |     8.04 |
|   512 |    128 |   8704 |   10.993 |    46.58 |   15.844 |     8.08 |
|   512 |    128 |   9216 |   10.423 |    49.12 |   16.107 |     7.95 |
|   512 |    128 |   9728 |   10.673 |    47.97 |   16.464 |     7.77 |
|   512 |    128 |  10240 |   11.141 |    45.96 |   16.899 |     7.57 |
|   512 |    128 |  10752 |   11.421 |    44.83 |   16.458 |     7.78 |
|   512 |    128 |  11264 |   14.421 |    35.50 |   17.190 |     7.45 |
|   512 |    128 |  11776 |   12.696 |    40.33 |   17.436 |     7.34 |
|   512 |    128 |  12288 |   12.079 |    42.39 |   17.327 |     7.39 |
|   512 |    128 |  12800 |   12.304 |    41.61 |   17.591 |     7.28 |
|   512 |    128 |  13312 |   13.400 |    38.21 |   17.857 |     7.17 |
|   512 |    128 |  13824 |   12.764 |    40.11 |   17.791 |     7.19 |
|   512 |    128 |  14336 |   13.515 |    37.88 |   18.744 |     6.83 |
|   512 |    128 |  14848 |   13.556 |    37.77 |   18.888 |     6.78 |
|   512 |    128 |  15360 |   13.925 |    36.77 |   19.552 |     6.55 |
|   512 |    128 |  15872 |   14.119 |    36.26 |   20.393 |     6.28 |
|   512 |    128 |  16384 |   14.246 |    35.94 |   20.078 |     6.38 |
|   512 |    128 |  16896 |   14.739 |    34.74 |   20.428 |     6.27 |
|   512 |    128 |  17408 |   15.744 |    32.52 |   21.013 |     6.09 |
|   512 |    128 |  17920 |   15.983 |    32.03 |   21.100 |     6.07 |
|   512 |    128 |  18432 |   16.247 |    31.51 |   21.502 |     5.95 |
|   512 |    128 |  18944 |   16.554 |    30.93 |   21.797 |     5.87 |
|   512 |    128 |  19456 |   16.923 |    30.25 |   18.987 |     6.74 |
|   512 |    128 |  19968 |   17.313 |    29.57 |   22.714 |     5.64 |
|   512 |    128 |  20480 |   17.972 |    28.49 |   22.245 |     5.75 |
|   512 |    128 |  20992 |   17.986 |    28.47 |   22.409 |     5.71 |
|   512 |    128 |  21504 |   18.304 |    27.97 |   23.061 |     5.55 |
|   512 |    128 |  22016 |   19.044 |    26.88 |   23.934 |     5.35 |
|   512 |    128 |  22528 |   19.563 |    26.17 |   23.447 |     5.46 |
|   512 |    128 |  23040 |   20.054 |    25.53 |   23.932 |     5.35 |
|   512 |    128 |  23552 |   20.210 |    25.33 |   24.398 |     5.25 |
|   512 |    128 |  24064 |   21.129 |    24.23 |   25.225 |     5.07 |
|   512 |    128 |  24576 |   19.675 |    26.02 |   25.531 |     5.01 |
|   512 |    128 |  25088 |   20.162 |    25.39 |   19.989 |     6.40 |
|   512 |    128 |  25600 |   20.685 |    24.75 |   25.551 |     5.01 |
|   512 |    128 |  26112 |   20.721 |    24.71 |   26.588 |     4.81 |
|   512 |    128 |  26624 |   20.997 |    24.38 |   27.079 |     4.73 |
|   512 |    128 |  27136 |   21.587 |    23.72 |   27.030 |     4.74 |
|   512 |    128 |  27648 |   22.148 |    23.12 |   27.153 |     4.71 |
|   512 |    128 |  28160 |   22.081 |    23.19 |   27.515 |     4.65 |
|   512 |    128 |  28672 |   22.620 |    22.64 |   27.332 |     4.68 |
|   512 |    128 |  29184 |   22.811 |    22.45 |   27.864 |     4.59 |
|   512 |    128 |  29696 |   22.791 |    22.47 |   28.755 |     4.45 |
|   512 |    128 |  30208 |   23.195 |    22.07 |   28.234 |     4.53 |
|   512 |    128 |  30720 |   23.924 |    21.40 |   21.459 |     5.96 |
|   512 |    128 |  31232 |   23.809 |    21.50 |   29.165 |     4.39 |
|   512 |    128 |  31744 |   23.712 |    21.59 |   29.106 |     4.40 |
|   512 |    128 |  32256 |   24.421 |    20.97 |   29.634 |     4.32 |
```

</details>

---

👤 **ikawrakow** commented the **2025-04-06** at **07:58:05**:<br>

@ubergarm 

Thank you for the testing. I have no working hypothesis at this point what is causing the TG performance spikes, and why we cannot have the performance at the peaks for all KV cache sizes. When I test with DeepSeek-Lite, TG-performance for context of 32k tokens is about 60% of the performance with zero context, so consistent with the performance at the spike peaks in your testing.

> I'm not sure what to try next

I added PR #315. It disables K-cache repacking. That has a non-negligible impact on performance for large contexts. Here is a graph that compares your TG results to 3 different runs with DeepSeek-Lite. I have scaled with TG performance at zero context length so we can have them on the same graph.  The red symbols are with PR #315. The blue and magenta symbols are with the main branch (one uses `-rtr`, the other uses the offline repacked version of the same model). Important to note that the K-cache repacking is done only for PP, and yet this additional memory allocation does affect TG performance! The effect for DeepSeek-R1/V3 may be bigger as the K-cache is larger. I did have runs where the TG performance drop happened earlier, and they ended with a lower performance at 32k tokens (but I didn't keep the logs for those).

![Image](https://github.com/user-attachments/assets/177a2b40-d4ff-4219-b35d-a024d3a94972)