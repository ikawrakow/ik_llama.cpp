### 🔀 [#299](https://github.com/ikawrakow/ik_llama.cpp/pull/299) - Additional guards for interleaved quants

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-03-31 |
| **Updated** | 2025-04-01 |

---

#### Description

Apparently not all use cases are covered when using interleaved quants, see #296.

Hopefully this PR handles all scenarios where one may arrive at using an interleaved quantization type where this is not possible.

---

#### 💬 Conversation

👤 **saood06** commented the **2025-03-31** at **12:05:48**:<br>

Decided to test this branch, using just pure with `./llama-quantize  --imatrix /mnt/sda/imatrix_V30324_mrader.dat --pure /mnt/sda/DeepseekV3_0324/DeepseekV3_0324-256x21B-BF16.gguf /mnt/sda/DeepSeek-V3-0324-IQ4_K_R4_ATT5.gguf IQ4_K_R4 48` and token embedding was still using the interleaved type.

```
[   1/1147]                    token_embd.weight - [ 7168, 129280,     1,     1], type =   bf16,
====== llama_model_quantize_internal: did not find weights for token_embd.weight
converting to iq4_k_r4 .. size =  1767.50 MiB ->   497.11 MiB
```

Then specifying token embedding type `./llama-quantize  --imatrix /mnt/sda/imatrix_V30324_mrader.dat --pure --token-embedding-type iq4_k /mnt/sda/DeepseekV3_0324/DeepseekV3_0324-256x21B-BF16.gguf /mnt/sda/DeepSeek-V3-0324-IQ4_K_R4_ATT5.gguf IQ4_K_R4 48`

It does result in it setting token embeddings quant type correctly but then it hits the assert.

```
[  10/1147]                blk.0.attn_k_b.weight - [  128, 65536,     1,     1], type =   bf16,
====== llama_model_quantize_internal: did not find weights for blk.0.attn_k_b.weight
converting to iq4_k_r4 .. /home/saood06/ik_main/ik_llama.cpp/ggml/src/iqk/iqk_quantize.cpp:5244: GGML_ASSERT(n_per_row%QK_K == 0) failed
```

Setting custom quant with ` --custom-q ".*=iq4_k_r4"` does not hit the assert but then token embeddings quant type is set to interleaved again.

```
[   1/1147]                    token_embd.weight - [ 7168, 129280,     1,     1], type =   bf16, Using custom type iq4_k_r4 for tensor token_embd.weight

====== llama_model_quantize_internal: did not find weights for token_embd.weight
converting to iq4_k_r4 .. size =  1767.50 MiB ->   497.11 MiB
```

(I ended up using ` --custom-q "token_embd.weight=iq4_k,.*=iq4_k_r4"` to make the mix I wanted)

---

👤 **ikawrakow** commented the **2025-03-31** at **12:46:26**:<br>

None of the above happens to me. Here the log of
```
./bin/llama-quantize --imatrix ../ncuda/dsl_imat_512.dat --pure ../models/deep2_lite/Deep-2-Lite-64x1.5B-F16-mla.gguf junk.bin iq4_k_r4
```
<details>
<code>
load_imatrix: imatrix dataset='../../llama.cpp/tests/wiki.train.raw'
load_imatrix: loaded 293 importance matrix entries from ../ncuda/dsl_imat_512.dat computed on 1000 chunks
prepare_imatrix: have 293 importance matrix entries
main: build = 3615 (7d55051f)
main: built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu
main: quantizing '../../iquants/models/deep2_lite/Deep-2-Lite-64x1.5B-F16-mla.gguf' to 'junk.bin' as IQ4_K_R4
llama_model_loader: loaded meta data with 45 key-value pairs and 431 tensors from ../../iquants/models/deep2_lite/Deep-2-Lite-64x1.5B-F16-mla.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Deep 2 Lite
llama_model_loader: - kv   3:                         general.size_label str              = 64x1.6B
llama_model_loader: - kv   4:                            general.license str              = other
llama_model_loader: - kv   5:                       general.license.name str              = deepseek
llama_model_loader: - kv   6:                       general.license.link str              = https://github.com/deepseek-ai/DeepSe...
llama_model_loader: - kv   7:                      deepseek2.block_count u32              = 27
llama_model_loader: - kv   8:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv   9:                 deepseek2.embedding_length u32              = 2048
llama_model_loader: - kv  10:              deepseek2.feed_forward_length u32              = 10944
llama_model_loader: - kv  11:             deepseek2.attention.head_count u32              = 16
llama_model_loader: - kv  12:          deepseek2.attention.head_count_kv u32              = 16
llama_model_loader: - kv  13:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  14: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  15:                deepseek2.expert_used_count u32              = 6
llama_model_loader: - kv  16:                          general.file_type u32              = 1
llama_model_loader: - kv  17:        deepseek2.leading_dense_block_count u32              = 1
llama_model_loader: - kv  18:                       deepseek2.vocab_size u32              = 102400
llama_model_loader: - kv  19:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  20:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  21:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  22:       deepseek2.expert_feed_forward_length u32              = 1408
llama_model_loader: - kv  23:                     deepseek2.expert_count u32              = 64
llama_model_loader: - kv  24:              deepseek2.expert_shared_count u32              = 2
llama_model_loader: - kv  25:             deepseek2.expert_weights_scale f32              = 1.000000
llama_model_loader: - kv  26:              deepseek2.expert_weights_norm bool             = false
llama_model_loader: - kv  27:               deepseek2.expert_gating_func u32              = 1
llama_model_loader: - kv  28:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  29:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  30:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  31: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  32: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.070700
llama_model_loader: - kv  33:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  34:                         tokenizer.ggml.pre str              = deepseek-llm
llama_model_loader: - kv  35:                      tokenizer.ggml.tokens arr[str,102400]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  36:                  tokenizer.ggml.token_type arr[i32,102400]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  37:                      tokenizer.ggml.merges arr[str,99757]   = ["Ġ Ġ", "Ġ t", "Ġ a", "i n", "h e...
llama_model_loader: - kv  38:                tokenizer.ggml.bos_token_id u32              = 100000
llama_model_loader: - kv  39:                tokenizer.ggml.eos_token_id u32              = 100001
llama_model_loader: - kv  40:            tokenizer.ggml.padding_token_id u32              = 100001
llama_model_loader: - kv  41:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  42:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  43:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  44:               general.quantization_version u32              = 2
llama_model_loader: - type  f32:  108 tensors
llama_model_loader: - type  f16:  323 tensors
================================ Have weights data with 293 entries
[   1/ 431]                        output.weight - [ 2048, 102400,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for output.weight
converting to iq4_k_r4 .. size =   400.00 MiB ->   112.50 MiB
[   2/ 431]                    token_embd.weight - [ 2048, 102400,     1,     1], type =    f16, 
============ Token embeddings cannot be quantized with row-interleaved quants
---> Changed iq4_k_r4 to iq4_k

====== llama_model_quantize_internal: did not find weights for token_embd.weight
converting to iq4_k .. size =   400.00 MiB ->   112.50 MiB
[   3/ 431]               blk.0.attn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[   4/ 431]                blk.0.ffn_down.weight - [10944,  2048,     1,     1], type =    f16, 

change_type_if_necessary : tensor cols 10944 x 2048 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0
converting to q5_0 .. size =    42.75 MiB ->    14.70 MiB
[   5/ 431]                blk.0.ffn_gate.weight - [ 2048, 10944,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    42.75 MiB ->    12.02 MiB
[   6/ 431]                  blk.0.ffn_up.weight - [ 2048, 10944,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    42.75 MiB ->    12.02 MiB
[   7/ 431]                blk.0.ffn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[   8/ 431]          blk.0.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[   9/ 431]           blk.0.attn_kv_a_mqa.weight - [ 2048,   576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.25 MiB ->     0.63 MiB
[  10/ 431]               blk.0.attn_kv_b.weight - [  512,  4096,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     4.00 MiB ->     1.12 MiB
[  11/ 431]                blk.0.attn_k_b.weight - [  128,  8192,     1,     1], type =    f16, 

change_type_if_necessary : tensor cols 128 x 8192 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.0.attn_k_b.weight
converting to q5_0 .. size =     2.00 MiB ->     0.69 MiB
[  12/ 431]                blk.0.attn_v_b.weight - [  512,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.00 MiB ->     0.56 MiB
[  13/ 431]             blk.0.attn_output.weight - [ 2048,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     8.00 MiB ->     2.25 MiB
[  14/ 431]                  blk.0.attn_q.weight - [ 2048,  3072,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    12.00 MiB ->     3.38 MiB
[  15/ 431]               blk.1.attn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[  16/ 431]           blk.1.ffn_down_exps.weight - [ 1408,  2048,    64,     1], type =    f16, 

change_type_if_necessary : tensor cols 1408 x 2048 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0
converting to q5_0 .. size =   352.00 MiB ->   121.00 MiB
[  17/ 431]           blk.1.ffn_gate_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[  18/ 431]             blk.1.ffn_up_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[  19/ 431]            blk.1.ffn_gate_inp.weight - [ 2048,    64,     1,     1], type =    f32, size =    0.500 MB
[  20/ 431]          blk.1.ffn_down_shexp.weight - [ 2816,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[  21/ 431]          blk.1.ffn_gate_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[  22/ 431]            blk.1.ffn_up_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[  23/ 431]                blk.1.ffn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[  24/ 431]          blk.1.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  25/ 431]           blk.1.attn_kv_a_mqa.weight - [ 2048,   576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.25 MiB ->     0.63 MiB
[  26/ 431]               blk.1.attn_kv_b.weight - [  512,  4096,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     4.00 MiB ->     1.12 MiB
[  27/ 431]                blk.1.attn_k_b.weight - [  128,  8192,     1,     1], type =    f16, 

change_type_if_necessary : tensor cols 128 x 8192 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.1.attn_k_b.weight
converting to q5_0 .. size =     2.00 MiB ->     0.69 MiB
[  28/ 431]                blk.1.attn_v_b.weight - [  512,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.00 MiB ->     0.56 MiB
[  29/ 431]             blk.1.attn_output.weight - [ 2048,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     8.00 MiB ->     2.25 MiB
[  30/ 431]                  blk.1.attn_q.weight - [ 2048,  3072,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    12.00 MiB ->     3.38 MiB
[  31/ 431]               blk.2.attn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[  32/ 431]           blk.2.ffn_down_exps.weight - [ 1408,  2048,    64,     1], type =    f16, 

change_type_if_necessary : tensor cols 1408 x 2048 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0
converting to q5_0 .. size =   352.00 MiB ->   121.00 MiB
[  33/ 431]           blk.2.ffn_gate_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[  34/ 431]             blk.2.ffn_up_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[  35/ 431]            blk.2.ffn_gate_inp.weight - [ 2048,    64,     1,     1], type =    f32, size =    0.500 MB
[  36/ 431]          blk.2.ffn_down_shexp.weight - [ 2816,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[  37/ 431]          blk.2.ffn_gate_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[  38/ 431]            blk.2.ffn_up_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[  39/ 431]                blk.2.ffn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[  40/ 431]          blk.2.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  41/ 431]           blk.2.attn_kv_a_mqa.weight - [ 2048,   576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.25 MiB ->     0.63 MiB
[  42/ 431]               blk.2.attn_kv_b.weight - [  512,  4096,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     4.00 MiB ->     1.12 MiB
[  43/ 431]                blk.2.attn_k_b.weight - [  128,  8192,     1,     1], type =    f16, 

change_type_if_necessary : tensor cols 128 x 8192 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.2.attn_k_b.weight
converting to q5_0 .. size =     2.00 MiB ->     0.69 MiB
[  44/ 431]                blk.2.attn_v_b.weight - [  512,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.00 MiB ->     0.56 MiB
[  45/ 431]             blk.2.attn_output.weight - [ 2048,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     8.00 MiB ->     2.25 MiB
[  46/ 431]                  blk.2.attn_q.weight - [ 2048,  3072,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    12.00 MiB ->     3.38 MiB
[  47/ 431]               blk.3.attn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[  48/ 431]           blk.3.ffn_down_exps.weight - [ 1408,  2048,    64,     1], type =    f16, 

change_type_if_necessary : tensor cols 1408 x 2048 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0
converting to q5_0 .. size =   352.00 MiB ->   121.00 MiB
[  49/ 431]           blk.3.ffn_gate_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[  50/ 431]             blk.3.ffn_up_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[  51/ 431]            blk.3.ffn_gate_inp.weight - [ 2048,    64,     1,     1], type =    f32, size =    0.500 MB
[  52/ 431]          blk.3.ffn_down_shexp.weight - [ 2816,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[  53/ 431]          blk.3.ffn_gate_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[  54/ 431]            blk.3.ffn_up_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[  55/ 431]                blk.3.ffn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[  56/ 431]          blk.3.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  57/ 431]           blk.3.attn_kv_a_mqa.weight - [ 2048,   576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.25 MiB ->     0.63 MiB
[  58/ 431]               blk.3.attn_kv_b.weight - [  512,  4096,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     4.00 MiB ->     1.12 MiB
[  59/ 431]                blk.3.attn_k_b.weight - [  128,  8192,     1,     1], type =    f16, 

change_type_if_necessary : tensor cols 128 x 8192 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.3.attn_k_b.weight
converting to q5_0 .. size =     2.00 MiB ->     0.69 MiB
[  60/ 431]                blk.3.attn_v_b.weight - [  512,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.00 MiB ->     0.56 MiB
[  61/ 431]             blk.3.attn_output.weight - [ 2048,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     8.00 MiB ->     2.25 MiB
[  62/ 431]                  blk.3.attn_q.weight - [ 2048,  3072,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    12.00 MiB ->     3.38 MiB
[  63/ 431]               blk.4.attn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[  64/ 431]           blk.4.ffn_down_exps.weight - [ 1408,  2048,    64,     1], type =    f16, 

change_type_if_necessary : tensor cols 1408 x 2048 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0
converting to q5_0 .. size =   352.00 MiB ->   121.00 MiB
[  65/ 431]           blk.4.ffn_gate_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[  66/ 431]             blk.4.ffn_up_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[  67/ 431]            blk.4.ffn_gate_inp.weight - [ 2048,    64,     1,     1], type =    f32, size =    0.500 MB
[  68/ 431]          blk.4.ffn_down_shexp.weight - [ 2816,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[  69/ 431]          blk.4.ffn_gate_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[  70/ 431]            blk.4.ffn_up_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[  71/ 431]                blk.4.ffn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[  72/ 431]          blk.4.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  73/ 431]           blk.4.attn_kv_a_mqa.weight - [ 2048,   576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.25 MiB ->     0.63 MiB
[  74/ 431]               blk.4.attn_kv_b.weight - [  512,  4096,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     4.00 MiB ->     1.12 MiB
[  75/ 431]                blk.4.attn_k_b.weight - [  128,  8192,     1,     1], type =    f16, 

change_type_if_necessary : tensor cols 128 x 8192 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.4.attn_k_b.weight
converting to q5_0 .. size =     2.00 MiB ->     0.69 MiB
[  76/ 431]                blk.4.attn_v_b.weight - [  512,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.00 MiB ->     0.56 MiB
[  77/ 431]             blk.4.attn_output.weight - [ 2048,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     8.00 MiB ->     2.25 MiB
[  78/ 431]                  blk.4.attn_q.weight - [ 2048,  3072,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    12.00 MiB ->     3.38 MiB
[  79/ 431]               blk.5.attn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[  80/ 431]           blk.5.ffn_down_exps.weight - [ 1408,  2048,    64,     1], type =    f16, 

change_type_if_necessary : tensor cols 1408 x 2048 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0
converting to q5_0 .. size =   352.00 MiB ->   121.00 MiB
[  81/ 431]           blk.5.ffn_gate_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[  82/ 431]             blk.5.ffn_up_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[  83/ 431]            blk.5.ffn_gate_inp.weight - [ 2048,    64,     1,     1], type =    f32, size =    0.500 MB
[  84/ 431]          blk.5.ffn_down_shexp.weight - [ 2816,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[  85/ 431]          blk.5.ffn_gate_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[  86/ 431]            blk.5.ffn_up_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[  87/ 431]                blk.5.ffn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[  88/ 431]          blk.5.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  89/ 431]           blk.5.attn_kv_a_mqa.weight - [ 2048,   576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.25 MiB ->     0.63 MiB
[  90/ 431]               blk.5.attn_kv_b.weight - [  512,  4096,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     4.00 MiB ->     1.12 MiB
[  91/ 431]                blk.5.attn_k_b.weight - [  128,  8192,     1,     1], type =    f16, 

change_type_if_necessary : tensor cols 128 x 8192 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.5.attn_k_b.weight
converting to q5_0 .. size =     2.00 MiB ->     0.69 MiB
[  92/ 431]                blk.5.attn_v_b.weight - [  512,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.00 MiB ->     0.56 MiB
[  93/ 431]             blk.5.attn_output.weight - [ 2048,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     8.00 MiB ->     2.25 MiB
[  94/ 431]                  blk.5.attn_q.weight - [ 2048,  3072,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    12.00 MiB ->     3.38 MiB
[  95/ 431]               blk.6.attn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[  96/ 431]           blk.6.ffn_down_exps.weight - [ 1408,  2048,    64,     1], type =    f16, 

change_type_if_necessary : tensor cols 1408 x 2048 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0
converting to q5_0 .. size =   352.00 MiB ->   121.00 MiB
[  97/ 431]           blk.6.ffn_gate_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[  98/ 431]             blk.6.ffn_up_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[  99/ 431]            blk.6.ffn_gate_inp.weight - [ 2048,    64,     1,     1], type =    f32, size =    0.500 MB
[ 100/ 431]          blk.6.ffn_down_shexp.weight - [ 2816,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 101/ 431]          blk.6.ffn_gate_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 102/ 431]            blk.6.ffn_up_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 103/ 431]                blk.6.ffn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[ 104/ 431]          blk.6.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 105/ 431]           blk.6.attn_kv_a_mqa.weight - [ 2048,   576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.25 MiB ->     0.63 MiB
[ 106/ 431]               blk.6.attn_kv_b.weight - [  512,  4096,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     4.00 MiB ->     1.12 MiB
[ 107/ 431]                blk.6.attn_k_b.weight - [  128,  8192,     1,     1], type =    f16, 

change_type_if_necessary : tensor cols 128 x 8192 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.6.attn_k_b.weight
converting to q5_0 .. size =     2.00 MiB ->     0.69 MiB
[ 108/ 431]                blk.6.attn_v_b.weight - [  512,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.00 MiB ->     0.56 MiB
[ 109/ 431]             blk.6.attn_output.weight - [ 2048,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     8.00 MiB ->     2.25 MiB
[ 110/ 431]                  blk.6.attn_q.weight - [ 2048,  3072,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    12.00 MiB ->     3.38 MiB
[ 111/ 431]            blk.7.ffn_gate_inp.weight - [ 2048,    64,     1,     1], type =    f32, size =    0.500 MB
[ 112/ 431]          blk.7.ffn_down_shexp.weight - [ 2816,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 113/ 431]          blk.7.ffn_gate_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 114/ 431]            blk.7.ffn_up_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 115/ 431]          blk.7.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 116/ 431]           blk.7.attn_kv_a_mqa.weight - [ 2048,   576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.25 MiB ->     0.63 MiB
[ 117/ 431]               blk.7.attn_kv_b.weight - [  512,  4096,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     4.00 MiB ->     1.12 MiB
[ 118/ 431]                blk.7.attn_k_b.weight - [  128,  8192,     1,     1], type =    f16, 

change_type_if_necessary : tensor cols 128 x 8192 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.7.attn_k_b.weight
converting to q5_0 .. size =     2.00 MiB ->     0.69 MiB
[ 119/ 431]                blk.7.attn_v_b.weight - [  512,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.00 MiB ->     0.56 MiB
[ 120/ 431]             blk.7.attn_output.weight - [ 2048,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     8.00 MiB ->     2.25 MiB
[ 121/ 431]                  blk.7.attn_q.weight - [ 2048,  3072,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    12.00 MiB ->     3.38 MiB
[ 122/ 431]                   output_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[ 123/ 431]              blk.10.attn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[ 124/ 431]          blk.10.ffn_down_exps.weight - [ 1408,  2048,    64,     1], type =    f16, 

change_type_if_necessary : tensor cols 1408 x 2048 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0
converting to q5_0 .. size =   352.00 MiB ->   121.00 MiB
[ 125/ 431]          blk.10.ffn_gate_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[ 126/ 431]            blk.10.ffn_up_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[ 127/ 431]           blk.10.ffn_gate_inp.weight - [ 2048,    64,     1,     1], type =    f32, size =    0.500 MB
[ 128/ 431]         blk.10.ffn_down_shexp.weight - [ 2816,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 129/ 431]         blk.10.ffn_gate_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 130/ 431]           blk.10.ffn_up_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 131/ 431]               blk.10.ffn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[ 132/ 431]         blk.10.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 133/ 431]          blk.10.attn_kv_a_mqa.weight - [ 2048,   576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.25 MiB ->     0.63 MiB
[ 134/ 431]              blk.10.attn_kv_b.weight - [  512,  4096,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     4.00 MiB ->     1.12 MiB
[ 135/ 431]               blk.10.attn_k_b.weight - [  128,  8192,     1,     1], type =    f16, 

change_type_if_necessary : tensor cols 128 x 8192 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.10.attn_k_b.weight
converting to q5_0 .. size =     2.00 MiB ->     0.69 MiB
[ 136/ 431]               blk.10.attn_v_b.weight - [  512,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.00 MiB ->     0.56 MiB
[ 137/ 431]            blk.10.attn_output.weight - [ 2048,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     8.00 MiB ->     2.25 MiB
[ 138/ 431]                 blk.10.attn_q.weight - [ 2048,  3072,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    12.00 MiB ->     3.38 MiB
[ 139/ 431]              blk.11.attn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[ 140/ 431]          blk.11.ffn_down_exps.weight - [ 1408,  2048,    64,     1], type =    f16, 

change_type_if_necessary : tensor cols 1408 x 2048 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0
converting to q5_0 .. size =   352.00 MiB ->   121.00 MiB
[ 141/ 431]          blk.11.ffn_gate_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[ 142/ 431]            blk.11.ffn_up_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[ 143/ 431]           blk.11.ffn_gate_inp.weight - [ 2048,    64,     1,     1], type =    f32, size =    0.500 MB
[ 144/ 431]         blk.11.ffn_down_shexp.weight - [ 2816,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 145/ 431]         blk.11.ffn_gate_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 146/ 431]           blk.11.ffn_up_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 147/ 431]               blk.11.ffn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[ 148/ 431]         blk.11.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 149/ 431]          blk.11.attn_kv_a_mqa.weight - [ 2048,   576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.25 MiB ->     0.63 MiB
[ 150/ 431]              blk.11.attn_kv_b.weight - [  512,  4096,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     4.00 MiB ->     1.12 MiB
[ 151/ 431]               blk.11.attn_k_b.weight - [  128,  8192,     1,     1], type =    f16, 

change_type_if_necessary : tensor cols 128 x 8192 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.11.attn_k_b.weight
converting to q5_0 .. size =     2.00 MiB ->     0.69 MiB
[ 152/ 431]               blk.11.attn_v_b.weight - [  512,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.00 MiB ->     0.56 MiB
[ 153/ 431]            blk.11.attn_output.weight - [ 2048,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     8.00 MiB ->     2.25 MiB
[ 154/ 431]                 blk.11.attn_q.weight - [ 2048,  3072,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    12.00 MiB ->     3.38 MiB
[ 155/ 431]              blk.12.attn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[ 156/ 431]          blk.12.ffn_down_exps.weight - [ 1408,  2048,    64,     1], type =    f16, 

change_type_if_necessary : tensor cols 1408 x 2048 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0
converting to q5_0 .. size =   352.00 MiB ->   121.00 MiB
[ 157/ 431]          blk.12.ffn_gate_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[ 158/ 431]            blk.12.ffn_up_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[ 159/ 431]           blk.12.ffn_gate_inp.weight - [ 2048,    64,     1,     1], type =    f32, size =    0.500 MB
[ 160/ 431]         blk.12.ffn_down_shexp.weight - [ 2816,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 161/ 431]         blk.12.ffn_gate_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 162/ 431]           blk.12.ffn_up_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 163/ 431]               blk.12.ffn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[ 164/ 431]         blk.12.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 165/ 431]          blk.12.attn_kv_a_mqa.weight - [ 2048,   576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.25 MiB ->     0.63 MiB
[ 166/ 431]              blk.12.attn_kv_b.weight - [  512,  4096,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     4.00 MiB ->     1.12 MiB
[ 167/ 431]               blk.12.attn_k_b.weight - [  128,  8192,     1,     1], type =    f16, 

change_type_if_necessary : tensor cols 128 x 8192 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.12.attn_k_b.weight
converting to q5_0 .. size =     2.00 MiB ->     0.69 MiB
[ 168/ 431]               blk.12.attn_v_b.weight - [  512,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.00 MiB ->     0.56 MiB
[ 169/ 431]            blk.12.attn_output.weight - [ 2048,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     8.00 MiB ->     2.25 MiB
[ 170/ 431]                 blk.12.attn_q.weight - [ 2048,  3072,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    12.00 MiB ->     3.38 MiB
[ 171/ 431]              blk.13.attn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[ 172/ 431]          blk.13.ffn_down_exps.weight - [ 1408,  2048,    64,     1], type =    f16, 

change_type_if_necessary : tensor cols 1408 x 2048 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0
converting to q5_0 .. size =   352.00 MiB ->   121.00 MiB
[ 173/ 431]          blk.13.ffn_gate_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[ 174/ 431]            blk.13.ffn_up_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[ 175/ 431]           blk.13.ffn_gate_inp.weight - [ 2048,    64,     1,     1], type =    f32, size =    0.500 MB
[ 176/ 431]         blk.13.ffn_down_shexp.weight - [ 2816,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 177/ 431]         blk.13.ffn_gate_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 178/ 431]           blk.13.ffn_up_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 179/ 431]               blk.13.ffn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[ 180/ 431]         blk.13.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 181/ 431]          blk.13.attn_kv_a_mqa.weight - [ 2048,   576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.25 MiB ->     0.63 MiB
[ 182/ 431]              blk.13.attn_kv_b.weight - [  512,  4096,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     4.00 MiB ->     1.12 MiB
[ 183/ 431]               blk.13.attn_k_b.weight - [  128,  8192,     1,     1], type =    f16, 

change_type_if_necessary : tensor cols 128 x 8192 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.13.attn_k_b.weight
converting to q5_0 .. size =     2.00 MiB ->     0.69 MiB
[ 184/ 431]               blk.13.attn_v_b.weight - [  512,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.00 MiB ->     0.56 MiB
[ 185/ 431]            blk.13.attn_output.weight - [ 2048,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     8.00 MiB ->     2.25 MiB
[ 186/ 431]                 blk.13.attn_q.weight - [ 2048,  3072,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    12.00 MiB ->     3.38 MiB
[ 187/ 431]           blk.14.ffn_gate_inp.weight - [ 2048,    64,     1,     1], type =    f32, size =    0.500 MB
[ 188/ 431]         blk.14.ffn_down_shexp.weight - [ 2816,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 189/ 431]         blk.14.ffn_gate_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 190/ 431]           blk.14.ffn_up_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 191/ 431]         blk.14.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 192/ 431]          blk.14.attn_kv_a_mqa.weight - [ 2048,   576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.25 MiB ->     0.63 MiB
[ 193/ 431]              blk.14.attn_kv_b.weight - [  512,  4096,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     4.00 MiB ->     1.12 MiB
[ 194/ 431]               blk.14.attn_k_b.weight - [  128,  8192,     1,     1], type =    f16, 

change_type_if_necessary : tensor cols 128 x 8192 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.14.attn_k_b.weight
converting to q5_0 .. size =     2.00 MiB ->     0.69 MiB
[ 195/ 431]               blk.14.attn_v_b.weight - [  512,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.00 MiB ->     0.56 MiB
[ 196/ 431]            blk.14.attn_output.weight - [ 2048,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     8.00 MiB ->     2.25 MiB
[ 197/ 431]                 blk.14.attn_q.weight - [ 2048,  3072,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    12.00 MiB ->     3.38 MiB
[ 198/ 431]               blk.7.attn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[ 199/ 431]           blk.7.ffn_down_exps.weight - [ 1408,  2048,    64,     1], type =    f16, 

change_type_if_necessary : tensor cols 1408 x 2048 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0
converting to q5_0 .. size =   352.00 MiB ->   121.00 MiB
[ 200/ 431]           blk.7.ffn_gate_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[ 201/ 431]             blk.7.ffn_up_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[ 202/ 431]                blk.7.ffn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[ 203/ 431]               blk.8.attn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[ 204/ 431]           blk.8.ffn_down_exps.weight - [ 1408,  2048,    64,     1], type =    f16, 

change_type_if_necessary : tensor cols 1408 x 2048 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0
converting to q5_0 .. size =   352.00 MiB ->   121.00 MiB
[ 205/ 431]           blk.8.ffn_gate_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[ 206/ 431]             blk.8.ffn_up_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[ 207/ 431]            blk.8.ffn_gate_inp.weight - [ 2048,    64,     1,     1], type =    f32, size =    0.500 MB
[ 208/ 431]          blk.8.ffn_down_shexp.weight - [ 2816,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 209/ 431]          blk.8.ffn_gate_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 210/ 431]            blk.8.ffn_up_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 211/ 431]                blk.8.ffn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[ 212/ 431]          blk.8.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 213/ 431]           blk.8.attn_kv_a_mqa.weight - [ 2048,   576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.25 MiB ->     0.63 MiB
[ 214/ 431]               blk.8.attn_kv_b.weight - [  512,  4096,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     4.00 MiB ->     1.12 MiB
[ 215/ 431]                blk.8.attn_k_b.weight - [  128,  8192,     1,     1], type =    f16, 

change_type_if_necessary : tensor cols 128 x 8192 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.8.attn_k_b.weight
converting to q5_0 .. size =     2.00 MiB ->     0.69 MiB
[ 216/ 431]                blk.8.attn_v_b.weight - [  512,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.00 MiB ->     0.56 MiB
[ 217/ 431]             blk.8.attn_output.weight - [ 2048,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     8.00 MiB ->     2.25 MiB
[ 218/ 431]                  blk.8.attn_q.weight - [ 2048,  3072,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    12.00 MiB ->     3.38 MiB
[ 219/ 431]               blk.9.attn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[ 220/ 431]           blk.9.ffn_down_exps.weight - [ 1408,  2048,    64,     1], type =    f16, 

change_type_if_necessary : tensor cols 1408 x 2048 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0
converting to q5_0 .. size =   352.00 MiB ->   121.00 MiB
[ 221/ 431]           blk.9.ffn_gate_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[ 222/ 431]             blk.9.ffn_up_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[ 223/ 431]            blk.9.ffn_gate_inp.weight - [ 2048,    64,     1,     1], type =    f32, size =    0.500 MB
[ 224/ 431]          blk.9.ffn_down_shexp.weight - [ 2816,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 225/ 431]          blk.9.ffn_gate_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 226/ 431]            blk.9.ffn_up_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 227/ 431]                blk.9.ffn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[ 228/ 431]          blk.9.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 229/ 431]           blk.9.attn_kv_a_mqa.weight - [ 2048,   576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.25 MiB ->     0.63 MiB
[ 230/ 431]               blk.9.attn_kv_b.weight - [  512,  4096,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     4.00 MiB ->     1.12 MiB
[ 231/ 431]                blk.9.attn_k_b.weight - [  128,  8192,     1,     1], type =    f16, 

change_type_if_necessary : tensor cols 128 x 8192 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.9.attn_k_b.weight
converting to q5_0 .. size =     2.00 MiB ->     0.69 MiB
[ 232/ 431]                blk.9.attn_v_b.weight - [  512,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.00 MiB ->     0.56 MiB
[ 233/ 431]             blk.9.attn_output.weight - [ 2048,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     8.00 MiB ->     2.25 MiB
[ 234/ 431]                  blk.9.attn_q.weight - [ 2048,  3072,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    12.00 MiB ->     3.38 MiB
[ 235/ 431]              blk.14.attn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[ 236/ 431]          blk.14.ffn_down_exps.weight - [ 1408,  2048,    64,     1], type =    f16, 

change_type_if_necessary : tensor cols 1408 x 2048 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0
converting to q5_0 .. size =   352.00 MiB ->   121.00 MiB
[ 237/ 431]          blk.14.ffn_gate_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[ 238/ 431]            blk.14.ffn_up_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[ 239/ 431]               blk.14.ffn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[ 240/ 431]              blk.15.attn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[ 241/ 431]          blk.15.ffn_down_exps.weight - [ 1408,  2048,    64,     1], type =    f16, 

change_type_if_necessary : tensor cols 1408 x 2048 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0
converting to q5_0 .. size =   352.00 MiB ->   121.00 MiB
[ 242/ 431]          blk.15.ffn_gate_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[ 243/ 431]            blk.15.ffn_up_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[ 244/ 431]           blk.15.ffn_gate_inp.weight - [ 2048,    64,     1,     1], type =    f32, size =    0.500 MB
[ 245/ 431]         blk.15.ffn_down_shexp.weight - [ 2816,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 246/ 431]         blk.15.ffn_gate_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 247/ 431]           blk.15.ffn_up_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 248/ 431]               blk.15.ffn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[ 249/ 431]         blk.15.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 250/ 431]          blk.15.attn_kv_a_mqa.weight - [ 2048,   576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.25 MiB ->     0.63 MiB
[ 251/ 431]              blk.15.attn_kv_b.weight - [  512,  4096,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     4.00 MiB ->     1.12 MiB
[ 252/ 431]               blk.15.attn_k_b.weight - [  128,  8192,     1,     1], type =    f16, 

change_type_if_necessary : tensor cols 128 x 8192 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.15.attn_k_b.weight
converting to q5_0 .. size =     2.00 MiB ->     0.69 MiB
[ 253/ 431]               blk.15.attn_v_b.weight - [  512,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.00 MiB ->     0.56 MiB
[ 254/ 431]            blk.15.attn_output.weight - [ 2048,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     8.00 MiB ->     2.25 MiB
[ 255/ 431]                 blk.15.attn_q.weight - [ 2048,  3072,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    12.00 MiB ->     3.38 MiB
[ 256/ 431]              blk.16.attn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[ 257/ 431]          blk.16.ffn_down_exps.weight - [ 1408,  2048,    64,     1], type =    f16, 

change_type_if_necessary : tensor cols 1408 x 2048 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0
converting to q5_0 .. size =   352.00 MiB ->   121.00 MiB
[ 258/ 431]          blk.16.ffn_gate_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[ 259/ 431]            blk.16.ffn_up_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[ 260/ 431]           blk.16.ffn_gate_inp.weight - [ 2048,    64,     1,     1], type =    f32, size =    0.500 MB
[ 261/ 431]         blk.16.ffn_down_shexp.weight - [ 2816,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 262/ 431]         blk.16.ffn_gate_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 263/ 431]           blk.16.ffn_up_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 264/ 431]               blk.16.ffn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[ 265/ 431]         blk.16.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 266/ 431]          blk.16.attn_kv_a_mqa.weight - [ 2048,   576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.25 MiB ->     0.63 MiB
[ 267/ 431]              blk.16.attn_kv_b.weight - [  512,  4096,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     4.00 MiB ->     1.12 MiB
[ 268/ 431]               blk.16.attn_k_b.weight - [  128,  8192,     1,     1], type =    f16, 

change_type_if_necessary : tensor cols 128 x 8192 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.16.attn_k_b.weight
converting to q5_0 .. size =     2.00 MiB ->     0.69 MiB
[ 269/ 431]               blk.16.attn_v_b.weight - [  512,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.00 MiB ->     0.56 MiB
[ 270/ 431]            blk.16.attn_output.weight - [ 2048,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     8.00 MiB ->     2.25 MiB
[ 271/ 431]                 blk.16.attn_q.weight - [ 2048,  3072,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    12.00 MiB ->     3.38 MiB
[ 272/ 431]              blk.17.attn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[ 273/ 431]          blk.17.ffn_down_exps.weight - [ 1408,  2048,    64,     1], type =    f16, 

change_type_if_necessary : tensor cols 1408 x 2048 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0
converting to q5_0 .. size =   352.00 MiB ->   121.00 MiB
[ 274/ 431]          blk.17.ffn_gate_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[ 275/ 431]            blk.17.ffn_up_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[ 276/ 431]           blk.17.ffn_gate_inp.weight - [ 2048,    64,     1,     1], type =    f32, size =    0.500 MB
[ 277/ 431]         blk.17.ffn_down_shexp.weight - [ 2816,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 278/ 431]         blk.17.ffn_gate_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 279/ 431]           blk.17.ffn_up_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 280/ 431]               blk.17.ffn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[ 281/ 431]         blk.17.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 282/ 431]          blk.17.attn_kv_a_mqa.weight - [ 2048,   576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.25 MiB ->     0.63 MiB
[ 283/ 431]              blk.17.attn_kv_b.weight - [  512,  4096,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     4.00 MiB ->     1.12 MiB
[ 284/ 431]               blk.17.attn_k_b.weight - [  128,  8192,     1,     1], type =    f16, 

change_type_if_necessary : tensor cols 128 x 8192 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.17.attn_k_b.weight
converting to q5_0 .. size =     2.00 MiB ->     0.69 MiB
[ 285/ 431]               blk.17.attn_v_b.weight - [  512,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.00 MiB ->     0.56 MiB
[ 286/ 431]            blk.17.attn_output.weight - [ 2048,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     8.00 MiB ->     2.25 MiB
[ 287/ 431]                 blk.17.attn_q.weight - [ 2048,  3072,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    12.00 MiB ->     3.38 MiB
[ 288/ 431]              blk.18.attn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[ 289/ 431]          blk.18.ffn_down_exps.weight - [ 1408,  2048,    64,     1], type =    f16, 

change_type_if_necessary : tensor cols 1408 x 2048 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0
converting to q5_0 .. size =   352.00 MiB ->   121.00 MiB
[ 290/ 431]          blk.18.ffn_gate_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[ 291/ 431]            blk.18.ffn_up_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[ 292/ 431]           blk.18.ffn_gate_inp.weight - [ 2048,    64,     1,     1], type =    f32, size =    0.500 MB
[ 293/ 431]         blk.18.ffn_down_shexp.weight - [ 2816,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 294/ 431]         blk.18.ffn_gate_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 295/ 431]           blk.18.ffn_up_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 296/ 431]               blk.18.ffn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[ 297/ 431]         blk.18.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 298/ 431]          blk.18.attn_kv_a_mqa.weight - [ 2048,   576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.25 MiB ->     0.63 MiB
[ 299/ 431]              blk.18.attn_kv_b.weight - [  512,  4096,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     4.00 MiB ->     1.12 MiB
[ 300/ 431]               blk.18.attn_k_b.weight - [  128,  8192,     1,     1], type =    f16, 

change_type_if_necessary : tensor cols 128 x 8192 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.18.attn_k_b.weight
converting to q5_0 .. size =     2.00 MiB ->     0.69 MiB
[ 301/ 431]               blk.18.attn_v_b.weight - [  512,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.00 MiB ->     0.56 MiB
[ 302/ 431]            blk.18.attn_output.weight - [ 2048,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     8.00 MiB ->     2.25 MiB
[ 303/ 431]                 blk.18.attn_q.weight - [ 2048,  3072,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    12.00 MiB ->     3.38 MiB
[ 304/ 431]              blk.19.attn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[ 305/ 431]          blk.19.ffn_down_exps.weight - [ 1408,  2048,    64,     1], type =    f16, 

change_type_if_necessary : tensor cols 1408 x 2048 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0
converting to q5_0 .. size =   352.00 MiB ->   121.00 MiB
[ 306/ 431]          blk.19.ffn_gate_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[ 307/ 431]            blk.19.ffn_up_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[ 308/ 431]           blk.19.ffn_gate_inp.weight - [ 2048,    64,     1,     1], type =    f32, size =    0.500 MB
[ 309/ 431]         blk.19.ffn_down_shexp.weight - [ 2816,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 310/ 431]         blk.19.ffn_gate_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 311/ 431]           blk.19.ffn_up_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 312/ 431]               blk.19.ffn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[ 313/ 431]         blk.19.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 314/ 431]          blk.19.attn_kv_a_mqa.weight - [ 2048,   576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.25 MiB ->     0.63 MiB
[ 315/ 431]              blk.19.attn_kv_b.weight - [  512,  4096,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     4.00 MiB ->     1.12 MiB
[ 316/ 431]               blk.19.attn_k_b.weight - [  128,  8192,     1,     1], type =    f16, 

change_type_if_necessary : tensor cols 128 x 8192 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.19.attn_k_b.weight
converting to q5_0 .. size =     2.00 MiB ->     0.69 MiB
[ 317/ 431]               blk.19.attn_v_b.weight - [  512,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.00 MiB ->     0.56 MiB
[ 318/ 431]            blk.19.attn_output.weight - [ 2048,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     8.00 MiB ->     2.25 MiB
[ 319/ 431]                 blk.19.attn_q.weight - [ 2048,  3072,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    12.00 MiB ->     3.38 MiB
[ 320/ 431]              blk.20.attn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[ 321/ 431]          blk.20.ffn_down_exps.weight - [ 1408,  2048,    64,     1], type =    f16, 

change_type_if_necessary : tensor cols 1408 x 2048 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0
converting to q5_0 .. size =   352.00 MiB ->   121.00 MiB
[ 322/ 431]          blk.20.ffn_gate_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[ 323/ 431]            blk.20.ffn_up_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[ 324/ 431]           blk.20.ffn_gate_inp.weight - [ 2048,    64,     1,     1], type =    f32, size =    0.500 MB
[ 325/ 431]         blk.20.ffn_down_shexp.weight - [ 2816,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 326/ 431]         blk.20.ffn_gate_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 327/ 431]           blk.20.ffn_up_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 328/ 431]               blk.20.ffn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[ 329/ 431]         blk.20.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 330/ 431]          blk.20.attn_kv_a_mqa.weight - [ 2048,   576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.25 MiB ->     0.63 MiB
[ 331/ 431]              blk.20.attn_kv_b.weight - [  512,  4096,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     4.00 MiB ->     1.12 MiB
[ 332/ 431]               blk.20.attn_k_b.weight - [  128,  8192,     1,     1], type =    f16, 

change_type_if_necessary : tensor cols 128 x 8192 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.20.attn_k_b.weight
converting to q5_0 .. size =     2.00 MiB ->     0.69 MiB
[ 333/ 431]               blk.20.attn_v_b.weight - [  512,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.00 MiB ->     0.56 MiB
[ 334/ 431]            blk.20.attn_output.weight - [ 2048,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     8.00 MiB ->     2.25 MiB
[ 335/ 431]                 blk.20.attn_q.weight - [ 2048,  3072,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    12.00 MiB ->     3.38 MiB
[ 336/ 431]              blk.21.attn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[ 337/ 431]          blk.21.ffn_down_exps.weight - [ 1408,  2048,    64,     1], type =    f16, 

change_type_if_necessary : tensor cols 1408 x 2048 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0
converting to q5_0 .. size =   352.00 MiB ->   121.00 MiB
[ 338/ 431]          blk.21.ffn_gate_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[ 339/ 431]            blk.21.ffn_up_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[ 340/ 431]           blk.21.ffn_gate_inp.weight - [ 2048,    64,     1,     1], type =    f32, size =    0.500 MB
[ 341/ 431]         blk.21.ffn_down_shexp.weight - [ 2816,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 342/ 431]         blk.21.ffn_gate_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 343/ 431]           blk.21.ffn_up_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 344/ 431]               blk.21.ffn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[ 345/ 431]         blk.21.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 346/ 431]          blk.21.attn_kv_a_mqa.weight - [ 2048,   576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.25 MiB ->     0.63 MiB
[ 347/ 431]              blk.21.attn_kv_b.weight - [  512,  4096,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     4.00 MiB ->     1.12 MiB
[ 348/ 431]               blk.21.attn_k_b.weight - [  128,  8192,     1,     1], type =    f16, 

change_type_if_necessary : tensor cols 128 x 8192 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.21.attn_k_b.weight
converting to q5_0 .. size =     2.00 MiB ->     0.69 MiB
[ 349/ 431]               blk.21.attn_v_b.weight - [  512,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.00 MiB ->     0.56 MiB
[ 350/ 431]            blk.21.attn_output.weight - [ 2048,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     8.00 MiB ->     2.25 MiB
[ 351/ 431]                 blk.21.attn_q.weight - [ 2048,  3072,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    12.00 MiB ->     3.38 MiB
[ 352/ 431]           blk.22.ffn_gate_inp.weight - [ 2048,    64,     1,     1], type =    f32, size =    0.500 MB
[ 353/ 431]         blk.22.ffn_down_shexp.weight - [ 2816,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 354/ 431]         blk.22.ffn_gate_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 355/ 431]           blk.22.ffn_up_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 356/ 431]         blk.22.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 357/ 431]          blk.22.attn_kv_a_mqa.weight - [ 2048,   576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.25 MiB ->     0.63 MiB
[ 358/ 431]              blk.22.attn_kv_b.weight - [  512,  4096,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     4.00 MiB ->     1.12 MiB
[ 359/ 431]               blk.22.attn_k_b.weight - [  128,  8192,     1,     1], type =    f16, 

change_type_if_necessary : tensor cols 128 x 8192 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.22.attn_k_b.weight
converting to q5_0 .. size =     2.00 MiB ->     0.69 MiB
[ 360/ 431]               blk.22.attn_v_b.weight - [  512,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.00 MiB ->     0.56 MiB
[ 361/ 431]            blk.22.attn_output.weight - [ 2048,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     8.00 MiB ->     2.25 MiB
[ 362/ 431]                 blk.22.attn_q.weight - [ 2048,  3072,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    12.00 MiB ->     3.38 MiB
[ 363/ 431]              blk.22.attn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[ 364/ 431]          blk.22.ffn_down_exps.weight - [ 1408,  2048,    64,     1], type =    f16, 

change_type_if_necessary : tensor cols 1408 x 2048 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0
converting to q5_0 .. size =   352.00 MiB ->   121.00 MiB
[ 365/ 431]          blk.22.ffn_gate_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[ 366/ 431]            blk.22.ffn_up_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[ 367/ 431]               blk.22.ffn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[ 368/ 431]              blk.23.attn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[ 369/ 431]          blk.23.ffn_down_exps.weight - [ 1408,  2048,    64,     1], type =    f16, 

change_type_if_necessary : tensor cols 1408 x 2048 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0
converting to q5_0 .. size =   352.00 MiB ->   121.00 MiB
[ 370/ 431]          blk.23.ffn_gate_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[ 371/ 431]            blk.23.ffn_up_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[ 372/ 431]           blk.23.ffn_gate_inp.weight - [ 2048,    64,     1,     1], type =    f32, size =    0.500 MB
[ 373/ 431]         blk.23.ffn_down_shexp.weight - [ 2816,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 374/ 431]         blk.23.ffn_gate_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 375/ 431]           blk.23.ffn_up_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 376/ 431]               blk.23.ffn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[ 377/ 431]         blk.23.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 378/ 431]          blk.23.attn_kv_a_mqa.weight - [ 2048,   576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.25 MiB ->     0.63 MiB
[ 379/ 431]              blk.23.attn_kv_b.weight - [  512,  4096,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     4.00 MiB ->     1.12 MiB
[ 380/ 431]               blk.23.attn_k_b.weight - [  128,  8192,     1,     1], type =    f16, 

change_type_if_necessary : tensor cols 128 x 8192 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.23.attn_k_b.weight
converting to q5_0 .. size =     2.00 MiB ->     0.69 MiB
[ 381/ 431]               blk.23.attn_v_b.weight - [  512,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.00 MiB ->     0.56 MiB
[ 382/ 431]            blk.23.attn_output.weight - [ 2048,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     8.00 MiB ->     2.25 MiB
[ 383/ 431]                 blk.23.attn_q.weight - [ 2048,  3072,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    12.00 MiB ->     3.38 MiB
[ 384/ 431]              blk.24.attn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[ 385/ 431]          blk.24.ffn_down_exps.weight - [ 1408,  2048,    64,     1], type =    f16, 

change_type_if_necessary : tensor cols 1408 x 2048 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0
converting to q5_0 .. size =   352.00 MiB ->   121.00 MiB
[ 386/ 431]          blk.24.ffn_gate_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[ 387/ 431]            blk.24.ffn_up_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[ 388/ 431]           blk.24.ffn_gate_inp.weight - [ 2048,    64,     1,     1], type =    f32, size =    0.500 MB
[ 389/ 431]         blk.24.ffn_down_shexp.weight - [ 2816,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 390/ 431]         blk.24.ffn_gate_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 391/ 431]           blk.24.ffn_up_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 392/ 431]               blk.24.ffn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[ 393/ 431]         blk.24.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 394/ 431]          blk.24.attn_kv_a_mqa.weight - [ 2048,   576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.25 MiB ->     0.63 MiB
[ 395/ 431]              blk.24.attn_kv_b.weight - [  512,  4096,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     4.00 MiB ->     1.12 MiB
[ 396/ 431]               blk.24.attn_k_b.weight - [  128,  8192,     1,     1], type =    f16, 

change_type_if_necessary : tensor cols 128 x 8192 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.24.attn_k_b.weight
converting to q5_0 .. size =     2.00 MiB ->     0.69 MiB
[ 397/ 431]               blk.24.attn_v_b.weight - [  512,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.00 MiB ->     0.56 MiB
[ 398/ 431]            blk.24.attn_output.weight - [ 2048,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     8.00 MiB ->     2.25 MiB
[ 399/ 431]                 blk.24.attn_q.weight - [ 2048,  3072,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    12.00 MiB ->     3.38 MiB
[ 400/ 431]              blk.25.attn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[ 401/ 431]          blk.25.ffn_down_exps.weight - [ 1408,  2048,    64,     1], type =    f16, 

change_type_if_necessary : tensor cols 1408 x 2048 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0
converting to q5_0 .. size =   352.00 MiB ->   121.00 MiB
[ 402/ 431]          blk.25.ffn_gate_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[ 403/ 431]            blk.25.ffn_up_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[ 404/ 431]           blk.25.ffn_gate_inp.weight - [ 2048,    64,     1,     1], type =    f32, size =    0.500 MB
[ 405/ 431]         blk.25.ffn_down_shexp.weight - [ 2816,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 406/ 431]         blk.25.ffn_gate_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 407/ 431]           blk.25.ffn_up_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 408/ 431]               blk.25.ffn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[ 409/ 431]         blk.25.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 410/ 431]          blk.25.attn_kv_a_mqa.weight - [ 2048,   576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.25 MiB ->     0.63 MiB
[ 411/ 431]              blk.25.attn_kv_b.weight - [  512,  4096,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     4.00 MiB ->     1.12 MiB
[ 412/ 431]               blk.25.attn_k_b.weight - [  128,  8192,     1,     1], type =    f16, 

change_type_if_necessary : tensor cols 128 x 8192 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.25.attn_k_b.weight
converting to q5_0 .. size =     2.00 MiB ->     0.69 MiB
[ 413/ 431]               blk.25.attn_v_b.weight - [  512,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.00 MiB ->     0.56 MiB
[ 414/ 431]            blk.25.attn_output.weight - [ 2048,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     8.00 MiB ->     2.25 MiB
[ 415/ 431]                 blk.25.attn_q.weight - [ 2048,  3072,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    12.00 MiB ->     3.38 MiB
[ 416/ 431]              blk.26.attn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[ 417/ 431]          blk.26.ffn_down_exps.weight - [ 1408,  2048,    64,     1], type =    f16, 

change_type_if_necessary : tensor cols 1408 x 2048 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0
converting to q5_0 .. size =   352.00 MiB ->   121.00 MiB
[ 418/ 431]          blk.26.ffn_gate_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[ 419/ 431]            blk.26.ffn_up_exps.weight - [ 2048,  1408,    64,     1], type =    f16, converting to iq4_k_r4 .. size =   352.00 MiB ->    99.00 MiB
[ 420/ 431]           blk.26.ffn_gate_inp.weight - [ 2048,    64,     1,     1], type =    f32, size =    0.500 MB
[ 421/ 431]         blk.26.ffn_down_shexp.weight - [ 2816,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 422/ 431]         blk.26.ffn_gate_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 423/ 431]           blk.26.ffn_up_shexp.weight - [ 2048,  2816,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    11.00 MiB ->     3.09 MiB
[ 424/ 431]               blk.26.ffn_norm.weight - [ 2048,     1,     1,     1], type =    f32, size =    0.008 MB
[ 425/ 431]         blk.26.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 426/ 431]          blk.26.attn_kv_a_mqa.weight - [ 2048,   576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.25 MiB ->     0.63 MiB
[ 427/ 431]              blk.26.attn_kv_b.weight - [  512,  4096,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     4.00 MiB ->     1.12 MiB
[ 428/ 431]               blk.26.attn_k_b.weight - [  128,  8192,     1,     1], type =    f16, 

change_type_if_necessary : tensor cols 128 x 8192 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.26.attn_k_b.weight
converting to q5_0 .. size =     2.00 MiB ->     0.69 MiB
[ 429/ 431]               blk.26.attn_v_b.weight - [  512,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     2.00 MiB ->     0.56 MiB
[ 430/ 431]            blk.26.attn_output.weight - [ 2048,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     8.00 MiB ->     2.25 MiB
[ 431/ 431]                 blk.26.attn_q.weight - [ 2048,  3072,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    12.00 MiB ->     3.38 MiB
llama_model_quantize_internal: model size  = 30072.48 MB
llama_model_quantize_internal: quant size  =  9045.62 MB
llama_model_quantize_internal: WARNING: 54 of 54 tensor(s) required fallback quantization

main: quantize time = 95227.57 ms
main:    total time = 95227.57 ms
</code>
</details>

Same outcome with `--custom-q ".*=iq4_k_r4"`.

---

👤 **saood06** commented the **2025-04-01** at **00:08:56**:<br>

> None of the above happens to me. Here the log of

Sorry I was running on the wrong branch. You can ignore my comment, as it all works on this branch.