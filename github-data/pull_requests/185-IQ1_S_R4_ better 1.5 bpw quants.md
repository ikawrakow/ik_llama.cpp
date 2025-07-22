### 🔀 [#185](https://github.com/ikawrakow/ik_llama.cpp/pull/185) - IQ1_S_R4: better 1.5 bpw quants

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-02-05 |
| **Updated** | 2025-02-08 |

---

#### Description

Given the hype around DeepSeek's models and [Unsloth's sub-2 bpw](https://huggingface.co/unsloth/DeepSeek-R1-GGUF) quantization of DeepSeek-R1 using `IQ1_S/IQ1_M`, I decided to give some love to sub-2 bpw quants. This PR adds `IQ1_S_R4`, a 4-row interleaved version of `IQ1_S`.

* `IQ1_S_R4` uses 1.5 bpw instead of the 1.5625 bpw needed by `IQ1_S`. The `f16` super-block scale is removed and is replaced by a `f16` scale per row
* `IQ1_S_R4` is implemented with a block size of 32. I wanted to have this because DeepSeek-Lite, the model I'm testing with, has a lot of tensors with row sizes not divisible by 256, so a significant fraction of tensors gets quantized to `IQ4_NL` when using `IQ1_S`
*  Quantization mixes for MoE models are adjusted. It is funny to observe how much credit Unsloth collected for their DeepSeek-R1 quantization. Their so called "dynamic" quantization has been in `llama.cpp` since the introduction of k-quants. The only reason it does not work well for DeepSeek's models is that the attention tensors have different names so that the heuristics used to assign a higher bpw quantization to the attention tensors fails. Case in point, today's mainline `llama.cpp` arrives at a context-512 perplexity (`PPL(512)` in what follows) of 36.8 for DeepSeek-Lite using 2.62 bpw. The `IQ1_S_R4` quantization in this PR gets `PPL-512 = 9.4` with 1.766 bpw for the repeating layers.
* `IQ1_S_R4` is **much faster** on the CPU compared to `IQ1_S` (see tables below). I never implemented iqk-style GEMM for `IQ1_S/IQ1_M`, so these quantization types run at the snail speed of mainline `llama.cpp`.
* Caveat: it is CPU only for now.

The following table compares prompt processing (pp512) and token generation (tg128) speed for LLaMA-3.1-8B on `AVX2` (Ryzen-5975WX), `Zen4` (Ryzen-7950X) and `ARM_NEON` (M2-Max CPU). I didn't use DeepSeek-Lite for this comparison to avoid the difference in quantization types one ends up with due to not all tensors having row sizes that are multiple of 256.

| platform   | threads |          test |  t/s (IQ1_S)     |   t/s (IQ1_S_R4) |  Speedup |
| ---------- | ------: | ------------: | ---------------: | ---------------: | -------: |
| AVX2       |      32 |         pp512 |     59.91 ± 0.07 |    218.78 ± 0.14 |  3.652   |   
| Zen4       |      16 |         pp512 |     35.78 ± 0.11 |    183.03 ± 1.09 |  5.115   |
| NEON       |       8 |         pp512 |     21.71 ± 0.24 |     78.37 ± 0.00 |  3.610   |
| AVX2       |       2 |         tg128 |      3.46 ± 0.00 |      5.05 ± 0.00 |  1.460   |   
|            |       4 |         tg128 |      6.89 ± 0.00 |      9.86 ± 0.00 |  1.431   |   
|            |       8 |         tg128 |     13.01 ± 0.08 |     17.54 ± 0.03 |  1.348   |   
|            |      16 |         tg128 |     21.99 ± 0.01 |     28.18 ± 0.00 |  1.281   |   
|            |      32 |         tg128 |     31.66 ± 0.02 |     33.22 ± 0.01 |  1.049   |   
| Zen4       |       2 |         tg128 |      4.41 ± 0.01 |      6.94 ± 0.01 |  1.574   |   
|            |       4 |         tg128 |      8.41 ± 0.00 |     12.97 ± 0.01 |  1.542   |   
|            |       8 |         tg128 |     14.04 ± 0.02 |     20.31 ± 0.00 |  1.447   |   
|            |      16 |         tg128 |     23.53 ± 0.02 |     29.15 ± 0.02 |  1.239   |   
| NEON       |       2 |         tg128 |      5.12 ± 0.00 |      6.86 ± 0.01 |  1.340   |
|            |       4 |         tg128 |      9.63 ± 0.00 |     13.01 ± 0.01 |  1.351   |
|            |       8 |         tg128 |     18.26 ± 0.14 |     24.30 ± 0.03 |  1.331   |

I don't have the disk space and RAM to play with DeepSeek-R1, so I would be really curious to hear from someone trying this PR for this model. It should be quite a bit faster than mainline, and I wouldn't be surprised if quality is better than Unsloth's `IQ1_S` quantization.

---

#### 💬 Conversation

👤 **saood06** commented the **2025-02-06** at **08:31:42**:<br>

>I don't have the disk space and RAM to play with DeepSeek-R1

I do.

>It should be quite a bit faster than mainline

It is.

>I wouldn't be surprised if quality is better than Unsloth's IQ1_S quantization.

Sadly, it doesn't really function. I haven't tried his IQ1_S, but yours might just be too small. You did a 127 GB. The unsloth creator said on reddit "I had a 127GB version, but it didn't go that good".

---

👤 **ikawrakow** commented the **2025-02-06** at **08:40:00**:<br>

@saood06 Do you have by any chance the quantization log? It would be useful to have it to verify that the intended tensors with higher bpw are correctly selected. It ends up being smaller than Unsloth's because `IQ1_S_R4` is 1.5 bpw vs 1.5625 bpw for `IQ1_S`. This 4% difference pretty much corresponds to the difference between 131 GiB and 127 GiB.

Oh, the other thing is that I did not change the default quantization for the token embeddings. It will use `Q2_K` by defualt for `IQ1_S/M/R4`, which did not work well for DeepSeek-Lite. I manually override this using `--token-embedding-type q8_0` when quantizing.

---

👤 **saood06** commented the **2025-02-06** at **08:48:25**:<br>

>Do you have by any chance the quantization log? 

Yes, I had to do some tweaks to it as well to work with the new tensor. It is in the log below. I want to say, I'm happy with my IQ4_K_R4, using this saood06/ik_llama.cpp/pull/1 I got all the way up to 30K context fitting on 384 GB of RAM without any cache quantization.

```
diff --git a/src/llama.cpp b/src/llama.cpp
index 02ad25ce..e23b4d5d 100644
--- a/src/llama.cpp
+++ b/src/llama.cpp
@@ -16215,7 +16215,7 @@ static ggml_type llama_tensor_get_type(quantize_state_internal & qs, ggml_type n
             }
         }
     } else if (ftype == LLAMA_FTYPE_MOSTLY_IQ1_S_R4) {
-        if (name.find("attn_v.weight") != std::string::npos) {
+        if (name.find("attn_v.weight") != std::string::npos || name.find("attn_v_b.weight") != std::string::npos) {
             if (qs.model.hparams.n_expert >= 4 || qs.model.hparams.n_gqa() >= 4) new_type = GGML_TYPE_IQ4_K_R4;
             else if (qs.model.hparams.n_gqa() >= 2) new_type = GGML_TYPE_IQ3_K_R4;
             else new_type = GGML_TYPE_Q2_K_R4;

```

<details open>
<summary>Log</summary>

```
load_imatrix: imatrix dataset='imatrix-training-full-3'
load_imatrix: loaded 720 importance matrix entries from /mnt/sda/mradermacher_DeepSeek-R1-GGUF/imatrix.dat computed on 315 chunks
prepare_imatrix: have 720 importance matrix entries
main: build = 3549 (ac732053)
main: built with gcc (Clear Linux OS for Intel Architecture) 14.2.1 20241210 releases/gcc-14.2.0-551-g21a09f0507 for x86_64-generic-linux
main: quantizing '/mnt/sda/opensourcerelease_DeepSeek-R1-bf16/opensourcerelease_DeepSeek-R1-Bf16-256x21B-F16.gguf' to '/mnt/sda/opensourcerelease_DeepSeek-R1-bf16/opensourcerelease_DeepSeek-R1-Bf16-256x21B-IQ1_S_R4.gguf' as IQ1_S_R4 using 48 threads
llama_model_loader: loaded meta data with 48 key-value pairs and 1147 tensors from /mnt/sda/opensourcerelease_DeepSeek-R1-bf16/opensourcerelease_DeepSeek-R1-Bf16-256x21B-F16.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = opensourcerelease_DeepSeek R1 Bf16
llama_model_loader: - kv   3:                         general.size_label str              = 256x21B
llama_model_loader: - kv   4:                            general.license str              = mit
llama_model_loader: - kv   5:                   general.base_model.count u32              = 1
llama_model_loader: - kv   6:                  general.base_model.0.name str              = DeepSeek R1
llama_model_loader: - kv   7:          general.base_model.0.organization str              = Deepseek Ai
llama_model_loader: - kv   8:              general.base_model.0.repo_url str              = https://huggingface.co/deepseek-ai/De...
llama_model_loader: - kv   9:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv  10:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv  11:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv  12:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  13:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv  14:          deepseek2.attention.head_count_kv u32              = 128
llama_model_loader: - kv  15:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  16: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  17:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  18:                          general.file_type u32              = 1
llama_model_loader: - kv  19:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  20:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  21:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  22:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  23:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  24:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  25:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  26:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  27:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  28:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  29:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  30:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  31:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  32:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  33:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  34: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  35: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  36:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  37:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  38:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<▒...
llama_model_loader: - kv  39:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  40:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
llama_model_loader: - kv  41:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  42:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  43:            tokenizer.ggml.padding_token_id u32              = 1
llama_model_loader: - kv  44:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  45:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  46:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  47:               general.quantization_version u32              = 2
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type  f16:  786 tensors
================================ Have weights data with 720 entries
[   1/1147]                    token_embd.weight - [ 7168, 129280,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for token_embd.weight
converting to q2_K .. size =  1767.50 MiB ->   289.98 MiB
[   2/1147]               blk.0.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[   3/1147]                blk.0.ffn_down.weight - [18432,  7168,     1,     1], type =    f16, converting to iq3_k_r4 .. size =   252.00 MiB ->    54.14 MiB
[   4/1147]                blk.0.ffn_gate.weight - [ 7168, 18432,     1,     1], type =    f16, converting to iq3_k_r4 .. size =   252.00 MiB ->    54.14 MiB
[   5/1147]                  blk.0.ffn_up.weight - [ 7168, 18432,     1,     1], type =    f16, converting to iq3_k_r4 .. size =   252.00 MiB ->    54.14 MiB
[   6/1147]                blk.0.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[   7/1147]          blk.0.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[   8/1147]           blk.0.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[   9/1147]               blk.0.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[  10/1147]                blk.0.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.0.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[  11/1147]                blk.0.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.0.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[  12/1147]             blk.0.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[  13/1147]           blk.0.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  14/1147]                blk.0.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[  15/1147]                blk.0.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[  16/1147]               blk.1.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  17/1147]                blk.1.ffn_down.weight - [18432,  7168,     1,     1], type =    f16, converting to q2_k_r4 .. size =   252.00 MiB ->    41.34 MiB
[  18/1147]                blk.1.ffn_gate.weight - [ 7168, 18432,     1,     1], type =    f16, converting to iq1_s_r4 .. size =   252.00 MiB ->    23.66 MiB
[  19/1147]                  blk.1.ffn_up.weight - [ 7168, 18432,     1,     1], type =    f16, converting to iq1_s_r4 .. size =   252.00 MiB ->    23.66 MiB
[  20/1147]                blk.1.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  21/1147]          blk.1.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  22/1147]           blk.1.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[  23/1147]               blk.1.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[  24/1147]                blk.1.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.1.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[  25/1147]                blk.1.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.1.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[  26/1147]             blk.1.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[  27/1147]           blk.1.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  28/1147]                blk.1.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[  29/1147]                blk.1.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[  30/1147]               blk.2.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  31/1147]                blk.2.ffn_down.weight - [18432,  7168,     1,     1], type =    f16, converting to q2_k_r4 .. size =   252.00 MiB ->    41.34 MiB
[  32/1147]                blk.2.ffn_gate.weight - [ 7168, 18432,     1,     1], type =    f16, converting to iq1_s_r4 .. size =   252.00 MiB ->    23.66 MiB
[  33/1147]                  blk.2.ffn_up.weight - [ 7168, 18432,     1,     1], type =    f16, converting to iq1_s_r4 .. size =   252.00 MiB ->    23.66 MiB
[  34/1147]                blk.2.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  35/1147]          blk.2.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  36/1147]           blk.2.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[  37/1147]               blk.2.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[  38/1147]                blk.2.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.2.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[  39/1147]                blk.2.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.2.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[  40/1147]             blk.2.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[  41/1147]           blk.2.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  42/1147]                blk.2.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[  43/1147]                blk.2.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[  44/1147]               blk.3.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[  45/1147]            blk.3.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[  46/1147]          blk.3.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[  47/1147]          blk.3.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[  48/1147]            blk.3.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[  49/1147]          blk.3.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  50/1147]           blk.3.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[  51/1147]               blk.3.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[  52/1147]                blk.3.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.3.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[  53/1147]                blk.3.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.3.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[  54/1147]             blk.3.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[  55/1147]           blk.3.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  56/1147]                blk.3.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[  57/1147]                blk.3.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[  58/1147]               blk.3.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  59/1147]           blk.3.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to q2_k_r4 .. size =  7168.00 MiB ->  1176.00 MiB
[  60/1147]           blk.3.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[  61/1147]             blk.3.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[  62/1147]                blk.3.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  63/1147]               blk.4.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[  64/1147]            blk.4.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[  65/1147]          blk.4.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[  66/1147]          blk.4.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[  67/1147]            blk.4.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[  68/1147]          blk.4.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  69/1147]           blk.4.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[  70/1147]               blk.4.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[  71/1147]                blk.4.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.4.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[  72/1147]                blk.4.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.4.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[  73/1147]             blk.4.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[  74/1147]           blk.4.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  75/1147]                blk.4.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[  76/1147]                blk.4.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[  77/1147]               blk.4.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  78/1147]           blk.4.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to q2_k_r4 .. size =  7168.00 MiB ->  1176.00 MiB
[  79/1147]           blk.4.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[  80/1147]             blk.4.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[  81/1147]                blk.4.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  82/1147]          blk.5.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  83/1147]           blk.5.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[  84/1147]               blk.5.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[  85/1147]                blk.5.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.5.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[  86/1147]                blk.5.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.5.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[  87/1147]             blk.5.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[  88/1147]           blk.5.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  89/1147]                blk.5.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[  90/1147]                blk.5.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[  91/1147]               blk.5.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[  92/1147]            blk.5.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[  93/1147]          blk.5.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[  94/1147]          blk.5.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[  95/1147]            blk.5.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[  96/1147]               blk.5.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  97/1147]           blk.5.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to q2_k_r4 .. size =  7168.00 MiB ->  1176.00 MiB
[  98/1147]           blk.5.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[  99/1147]             blk.5.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 100/1147]                blk.5.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 101/1147]               blk.6.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 102/1147]            blk.6.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 103/1147]          blk.6.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 104/1147]          blk.6.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 105/1147]            blk.6.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 106/1147]          blk.6.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 107/1147]           blk.6.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 108/1147]               blk.6.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 109/1147]                blk.6.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.6.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 110/1147]                blk.6.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.6.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 111/1147]             blk.6.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 112/1147]           blk.6.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 113/1147]                blk.6.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 114/1147]                blk.6.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 115/1147]               blk.6.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 116/1147]           blk.6.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to q2_k_r4 .. size =  7168.00 MiB ->  1176.00 MiB
[ 117/1147]           blk.6.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 118/1147]             blk.6.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 119/1147]                blk.6.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 120/1147]               blk.7.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 121/1147]            blk.7.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 122/1147]          blk.7.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 123/1147]          blk.7.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 124/1147]            blk.7.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 125/1147]          blk.7.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 126/1147]           blk.7.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 127/1147]               blk.7.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 128/1147]                blk.7.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.7.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 129/1147]                blk.7.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.7.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 130/1147]             blk.7.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 131/1147]           blk.7.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 132/1147]                blk.7.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 133/1147]                blk.7.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 134/1147]               blk.7.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 135/1147]           blk.7.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 136/1147]           blk.7.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 137/1147]             blk.7.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 138/1147]                blk.7.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 139/1147]               blk.8.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 140/1147]            blk.8.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 141/1147]          blk.8.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 142/1147]          blk.8.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 143/1147]            blk.8.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 144/1147]          blk.8.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 145/1147]           blk.8.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 146/1147]               blk.8.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 147/1147]                blk.8.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.8.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 148/1147]                blk.8.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.8.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 149/1147]             blk.8.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 150/1147]           blk.8.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 151/1147]                blk.8.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 152/1147]                blk.8.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 153/1147]               blk.8.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 154/1147]           blk.8.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 155/1147]           blk.8.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 156/1147]             blk.8.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 157/1147]                blk.8.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 158/1147]               blk.9.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 159/1147]            blk.9.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 160/1147]          blk.9.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 161/1147]          blk.9.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 162/1147]            blk.9.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 163/1147]          blk.9.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 164/1147]           blk.9.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 165/1147]               blk.9.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 166/1147]                blk.9.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.9.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 167/1147]                blk.9.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.9.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 168/1147]             blk.9.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 169/1147]           blk.9.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 170/1147]                blk.9.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 171/1147]                blk.9.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 172/1147]              blk.10.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 173/1147]           blk.10.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 174/1147]         blk.10.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 175/1147]         blk.10.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 176/1147]           blk.10.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 177/1147]         blk.10.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 178/1147]          blk.10.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 179/1147]              blk.10.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 180/1147]               blk.10.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.10.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 181/1147]               blk.10.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.10.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 182/1147]            blk.10.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 183/1147]          blk.10.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 184/1147]               blk.10.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 185/1147]               blk.10.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 186/1147]               blk.9.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 187/1147]           blk.9.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 188/1147]           blk.9.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 189/1147]             blk.9.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 190/1147]                blk.9.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 191/1147]              blk.10.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 192/1147]          blk.10.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 193/1147]          blk.10.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 194/1147]            blk.10.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 195/1147]               blk.10.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 196/1147]              blk.11.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 197/1147]           blk.11.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 198/1147]         blk.11.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 199/1147]         blk.11.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 200/1147]           blk.11.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 201/1147]         blk.11.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 202/1147]          blk.11.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 203/1147]              blk.11.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 204/1147]               blk.11.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.11.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 205/1147]               blk.11.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.11.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 206/1147]            blk.11.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 207/1147]          blk.11.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 208/1147]               blk.11.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 209/1147]               blk.11.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 210/1147]              blk.11.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 211/1147]          blk.11.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 212/1147]          blk.11.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 213/1147]            blk.11.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 214/1147]               blk.11.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 215/1147]              blk.12.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 216/1147]           blk.12.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 217/1147]         blk.12.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 218/1147]         blk.12.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 219/1147]           blk.12.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 220/1147]         blk.12.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 221/1147]          blk.12.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 222/1147]              blk.12.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 223/1147]               blk.12.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.12.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 224/1147]               blk.12.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.12.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 225/1147]            blk.12.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 226/1147]          blk.12.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 227/1147]               blk.12.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 228/1147]               blk.12.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 229/1147]              blk.12.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 230/1147]          blk.12.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 231/1147]          blk.12.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 232/1147]            blk.12.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 233/1147]               blk.12.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 234/1147]              blk.13.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 235/1147]           blk.13.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 236/1147]         blk.13.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 237/1147]         blk.13.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 238/1147]           blk.13.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 239/1147]         blk.13.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 240/1147]          blk.13.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 241/1147]              blk.13.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 242/1147]               blk.13.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.13.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 243/1147]               blk.13.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.13.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 244/1147]            blk.13.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 245/1147]          blk.13.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 246/1147]               blk.13.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 247/1147]               blk.13.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 248/1147]              blk.13.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 249/1147]          blk.13.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 250/1147]          blk.13.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 251/1147]            blk.13.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 252/1147]               blk.13.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 253/1147]              blk.14.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 254/1147]           blk.14.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 255/1147]         blk.14.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 256/1147]         blk.14.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 257/1147]           blk.14.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 258/1147]         blk.14.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 259/1147]          blk.14.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 260/1147]              blk.14.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 261/1147]               blk.14.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.14.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 262/1147]               blk.14.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.14.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 263/1147]            blk.14.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 264/1147]          blk.14.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 265/1147]               blk.14.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 266/1147]               blk.14.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 267/1147]              blk.14.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 268/1147]          blk.14.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 269/1147]          blk.14.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 270/1147]            blk.14.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 271/1147]               blk.14.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 272/1147]              blk.15.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 273/1147]           blk.15.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 274/1147]         blk.15.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 275/1147]         blk.15.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 276/1147]           blk.15.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 277/1147]         blk.15.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 278/1147]          blk.15.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 279/1147]              blk.15.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 280/1147]               blk.15.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.15.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 281/1147]               blk.15.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.15.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 282/1147]            blk.15.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 283/1147]          blk.15.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 284/1147]               blk.15.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 285/1147]               blk.15.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 286/1147]              blk.15.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 287/1147]          blk.15.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 288/1147]          blk.15.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 289/1147]            blk.15.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 290/1147]               blk.15.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 291/1147]              blk.16.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 292/1147]           blk.16.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 293/1147]         blk.16.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 294/1147]         blk.16.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 295/1147]           blk.16.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 296/1147]         blk.16.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 297/1147]          blk.16.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 298/1147]              blk.16.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 299/1147]               blk.16.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.16.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 300/1147]               blk.16.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.16.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 301/1147]            blk.16.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 302/1147]          blk.16.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 303/1147]               blk.16.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 304/1147]               blk.16.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 305/1147]              blk.16.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 306/1147]          blk.16.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 307/1147]          blk.16.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 308/1147]            blk.16.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 309/1147]               blk.16.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 310/1147]              blk.17.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 311/1147]           blk.17.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 312/1147]         blk.17.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 313/1147]         blk.17.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 314/1147]           blk.17.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 315/1147]         blk.17.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 316/1147]          blk.17.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 317/1147]              blk.17.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 318/1147]               blk.17.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.17.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 319/1147]               blk.17.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.17.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 320/1147]            blk.17.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 321/1147]          blk.17.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 322/1147]               blk.17.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 323/1147]               blk.17.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 324/1147]              blk.17.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 325/1147]          blk.17.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 326/1147]          blk.17.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 327/1147]            blk.17.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 328/1147]               blk.17.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 329/1147]              blk.18.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 330/1147]           blk.18.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 331/1147]         blk.18.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 332/1147]         blk.18.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 333/1147]           blk.18.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 334/1147]         blk.18.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 335/1147]          blk.18.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 336/1147]              blk.18.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 337/1147]               blk.18.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.18.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 338/1147]               blk.18.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.18.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 339/1147]            blk.18.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 340/1147]          blk.18.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 341/1147]               blk.18.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 342/1147]               blk.18.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 343/1147]              blk.18.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 344/1147]          blk.18.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 345/1147]          blk.18.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 346/1147]            blk.18.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 347/1147]               blk.18.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 348/1147]              blk.19.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 349/1147]           blk.19.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 350/1147]         blk.19.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 351/1147]         blk.19.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 352/1147]           blk.19.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 353/1147]         blk.19.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 354/1147]          blk.19.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 355/1147]              blk.19.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 356/1147]               blk.19.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.19.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 357/1147]               blk.19.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.19.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 358/1147]            blk.19.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 359/1147]          blk.19.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 360/1147]               blk.19.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 361/1147]               blk.19.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 362/1147]              blk.19.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 363/1147]          blk.19.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 364/1147]          blk.19.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 365/1147]            blk.19.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 366/1147]               blk.19.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 367/1147]              blk.20.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 368/1147]           blk.20.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 369/1147]         blk.20.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 370/1147]         blk.20.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 371/1147]           blk.20.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 372/1147]         blk.20.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 373/1147]          blk.20.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 374/1147]              blk.20.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 375/1147]               blk.20.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.20.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 376/1147]               blk.20.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.20.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 377/1147]            blk.20.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 378/1147]          blk.20.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 379/1147]               blk.20.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 380/1147]               blk.20.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 381/1147]              blk.20.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 382/1147]          blk.20.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 383/1147]          blk.20.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 384/1147]            blk.20.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 385/1147]               blk.20.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 386/1147]              blk.21.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 387/1147]           blk.21.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 388/1147]         blk.21.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 389/1147]         blk.21.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 390/1147]           blk.21.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 391/1147]         blk.21.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 392/1147]          blk.21.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 393/1147]              blk.21.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 394/1147]               blk.21.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.21.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 395/1147]               blk.21.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.21.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 396/1147]            blk.21.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 397/1147]          blk.21.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 398/1147]               blk.21.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 399/1147]               blk.21.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 400/1147]              blk.21.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 401/1147]          blk.21.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 402/1147]          blk.21.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 403/1147]            blk.21.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 404/1147]               blk.21.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 405/1147]              blk.22.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 406/1147]           blk.22.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 407/1147]         blk.22.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 408/1147]         blk.22.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 409/1147]           blk.22.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 410/1147]         blk.22.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 411/1147]          blk.22.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 412/1147]              blk.22.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 413/1147]               blk.22.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.22.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 414/1147]               blk.22.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.22.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 415/1147]            blk.22.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 416/1147]          blk.22.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 417/1147]               blk.22.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 418/1147]               blk.22.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 419/1147]              blk.22.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 420/1147]          blk.22.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 421/1147]          blk.22.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 422/1147]            blk.22.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 423/1147]               blk.22.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 424/1147]              blk.23.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 425/1147]           blk.23.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 426/1147]         blk.23.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 427/1147]         blk.23.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 428/1147]           blk.23.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 429/1147]         blk.23.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 430/1147]          blk.23.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 431/1147]              blk.23.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 432/1147]               blk.23.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.23.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 433/1147]               blk.23.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.23.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 434/1147]            blk.23.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 435/1147]          blk.23.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 436/1147]               blk.23.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 437/1147]               blk.23.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 438/1147]              blk.23.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 439/1147]          blk.23.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 440/1147]          blk.23.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 441/1147]            blk.23.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 442/1147]               blk.23.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 443/1147]              blk.24.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 444/1147]           blk.24.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 445/1147]         blk.24.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 446/1147]         blk.24.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 447/1147]           blk.24.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 448/1147]         blk.24.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 449/1147]          blk.24.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 450/1147]              blk.24.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 451/1147]               blk.24.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.24.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 452/1147]               blk.24.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.24.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 453/1147]            blk.24.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 454/1147]          blk.24.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 455/1147]               blk.24.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 456/1147]               blk.24.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 457/1147]              blk.24.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 458/1147]          blk.24.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 459/1147]          blk.24.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 460/1147]            blk.24.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 461/1147]               blk.24.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 462/1147]              blk.25.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 463/1147]           blk.25.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 464/1147]         blk.25.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 465/1147]         blk.25.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 466/1147]           blk.25.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 467/1147]         blk.25.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 468/1147]          blk.25.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 469/1147]              blk.25.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 470/1147]               blk.25.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.25.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 471/1147]               blk.25.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.25.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 472/1147]            blk.25.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 473/1147]          blk.25.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 474/1147]               blk.25.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 475/1147]               blk.25.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 476/1147]              blk.25.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 477/1147]          blk.25.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 478/1147]          blk.25.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 479/1147]            blk.25.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 480/1147]               blk.25.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 481/1147]              blk.26.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 482/1147]           blk.26.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 483/1147]         blk.26.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 484/1147]         blk.26.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 485/1147]           blk.26.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 486/1147]         blk.26.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 487/1147]          blk.26.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 488/1147]              blk.26.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 489/1147]               blk.26.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.26.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 490/1147]               blk.26.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.26.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 491/1147]            blk.26.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 492/1147]          blk.26.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 493/1147]               blk.26.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 494/1147]               blk.26.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 495/1147]              blk.26.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 496/1147]          blk.26.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 497/1147]          blk.26.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 498/1147]            blk.26.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 499/1147]               blk.26.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 500/1147]              blk.27.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 501/1147]           blk.27.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 502/1147]         blk.27.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 503/1147]         blk.27.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 504/1147]           blk.27.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 505/1147]         blk.27.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 506/1147]          blk.27.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 507/1147]              blk.27.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 508/1147]               blk.27.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.27.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 509/1147]               blk.27.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.27.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 510/1147]            blk.27.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 511/1147]          blk.27.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 512/1147]               blk.27.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 513/1147]               blk.27.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 514/1147]              blk.27.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 515/1147]          blk.27.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 516/1147]          blk.27.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 517/1147]            blk.27.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 518/1147]               blk.27.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 519/1147]              blk.28.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 520/1147]           blk.28.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 521/1147]         blk.28.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 522/1147]         blk.28.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 523/1147]           blk.28.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 524/1147]         blk.28.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 525/1147]          blk.28.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 526/1147]              blk.28.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 527/1147]               blk.28.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.28.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 528/1147]               blk.28.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.28.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 529/1147]            blk.28.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 530/1147]          blk.28.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 531/1147]               blk.28.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 532/1147]               blk.28.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 533/1147]              blk.28.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 534/1147]          blk.28.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 535/1147]          blk.28.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 536/1147]            blk.28.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 537/1147]               blk.28.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 538/1147]              blk.29.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 539/1147]           blk.29.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 540/1147]         blk.29.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 541/1147]         blk.29.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 542/1147]           blk.29.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 543/1147]         blk.29.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 544/1147]          blk.29.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 545/1147]              blk.29.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 546/1147]               blk.29.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.29.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 547/1147]               blk.29.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.29.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 548/1147]            blk.29.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 549/1147]          blk.29.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 550/1147]               blk.29.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 551/1147]               blk.29.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 552/1147]              blk.29.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 553/1147]          blk.29.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 554/1147]          blk.29.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 555/1147]            blk.29.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 556/1147]               blk.29.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 557/1147]              blk.30.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 558/1147]           blk.30.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 559/1147]         blk.30.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 560/1147]         blk.30.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 561/1147]           blk.30.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 562/1147]         blk.30.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 563/1147]          blk.30.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 564/1147]              blk.30.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 565/1147]               blk.30.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.30.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 566/1147]               blk.30.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.30.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 567/1147]            blk.30.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 568/1147]          blk.30.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 569/1147]               blk.30.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 570/1147]               blk.30.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 571/1147]              blk.30.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 572/1147]          blk.30.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 573/1147]          blk.30.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 574/1147]            blk.30.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 575/1147]               blk.30.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 576/1147]              blk.31.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 577/1147]           blk.31.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 578/1147]         blk.31.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 579/1147]         blk.31.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 580/1147]           blk.31.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 581/1147]         blk.31.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 582/1147]          blk.31.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 583/1147]              blk.31.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 584/1147]               blk.31.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.31.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 585/1147]               blk.31.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.31.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 586/1147]            blk.31.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 587/1147]          blk.31.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 588/1147]               blk.31.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 589/1147]               blk.31.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 590/1147]              blk.31.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 591/1147]          blk.31.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 592/1147]          blk.31.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 593/1147]            blk.31.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 594/1147]               blk.31.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 595/1147]              blk.32.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 596/1147]           blk.32.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 597/1147]         blk.32.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 598/1147]         blk.32.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 599/1147]           blk.32.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 600/1147]         blk.32.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 601/1147]          blk.32.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 602/1147]              blk.32.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 603/1147]               blk.32.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.32.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 604/1147]               blk.32.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.32.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 605/1147]            blk.32.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 606/1147]          blk.32.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 607/1147]               blk.32.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 608/1147]               blk.32.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 609/1147]              blk.32.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 610/1147]          blk.32.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 611/1147]          blk.32.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 612/1147]            blk.32.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 613/1147]               blk.32.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 614/1147]              blk.33.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 615/1147]           blk.33.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 616/1147]         blk.33.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 617/1147]         blk.33.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 618/1147]           blk.33.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 619/1147]         blk.33.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 620/1147]          blk.33.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 621/1147]              blk.33.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 622/1147]               blk.33.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.33.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 623/1147]               blk.33.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.33.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 624/1147]            blk.33.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 625/1147]          blk.33.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 626/1147]               blk.33.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 627/1147]               blk.33.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 628/1147]              blk.33.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 629/1147]          blk.33.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 630/1147]          blk.33.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 631/1147]            blk.33.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 632/1147]               blk.33.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 633/1147]              blk.34.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 634/1147]           blk.34.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 635/1147]         blk.34.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 636/1147]         blk.34.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 637/1147]           blk.34.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 638/1147]         blk.34.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 639/1147]          blk.34.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 640/1147]              blk.34.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 641/1147]               blk.34.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.34.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 642/1147]               blk.34.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.34.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 643/1147]            blk.34.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 644/1147]          blk.34.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 645/1147]               blk.34.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 646/1147]               blk.34.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 647/1147]              blk.34.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 648/1147]          blk.34.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 649/1147]          blk.34.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 650/1147]            blk.34.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 651/1147]               blk.34.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 652/1147]              blk.35.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 653/1147]           blk.35.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 654/1147]         blk.35.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 655/1147]         blk.35.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 656/1147]           blk.35.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 657/1147]         blk.35.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 658/1147]          blk.35.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 659/1147]              blk.35.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 660/1147]               blk.35.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.35.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 661/1147]               blk.35.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.35.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 662/1147]            blk.35.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 663/1147]          blk.35.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 664/1147]               blk.35.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 665/1147]               blk.35.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 666/1147]              blk.35.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 667/1147]          blk.35.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 668/1147]          blk.35.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 669/1147]            blk.35.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 670/1147]               blk.35.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 671/1147]              blk.36.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 672/1147]           blk.36.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 673/1147]         blk.36.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 674/1147]         blk.36.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 675/1147]           blk.36.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 676/1147]         blk.36.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 677/1147]          blk.36.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 678/1147]              blk.36.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 679/1147]               blk.36.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.36.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 680/1147]               blk.36.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.36.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 681/1147]            blk.36.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 682/1147]          blk.36.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 683/1147]               blk.36.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 684/1147]               blk.36.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 685/1147]              blk.36.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 686/1147]          blk.36.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 687/1147]          blk.36.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 688/1147]            blk.36.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 689/1147]               blk.36.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 690/1147]              blk.37.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 691/1147]           blk.37.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 692/1147]         blk.37.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 693/1147]         blk.37.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 694/1147]           blk.37.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 695/1147]         blk.37.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 696/1147]          blk.37.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 697/1147]              blk.37.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 698/1147]               blk.37.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.37.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 699/1147]               blk.37.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.37.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 700/1147]            blk.37.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 701/1147]          blk.37.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 702/1147]               blk.37.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 703/1147]               blk.37.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 704/1147]              blk.37.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 705/1147]          blk.37.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 706/1147]          blk.37.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 707/1147]            blk.37.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 708/1147]               blk.37.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 709/1147]              blk.38.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 710/1147]           blk.38.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 711/1147]         blk.38.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 712/1147]         blk.38.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 713/1147]           blk.38.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 714/1147]         blk.38.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 715/1147]          blk.38.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 716/1147]              blk.38.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 717/1147]               blk.38.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.38.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 718/1147]               blk.38.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.38.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 719/1147]            blk.38.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 720/1147]          blk.38.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 721/1147]               blk.38.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 722/1147]               blk.38.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 723/1147]              blk.38.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 724/1147]          blk.38.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 725/1147]          blk.38.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 726/1147]            blk.38.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 727/1147]               blk.38.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 728/1147]              blk.39.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 729/1147]           blk.39.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 730/1147]         blk.39.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 731/1147]         blk.39.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 732/1147]           blk.39.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 733/1147]         blk.39.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 734/1147]          blk.39.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 735/1147]              blk.39.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 736/1147]               blk.39.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.39.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 737/1147]               blk.39.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.39.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 738/1147]            blk.39.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 739/1147]          blk.39.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 740/1147]               blk.39.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 741/1147]               blk.39.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 742/1147]              blk.39.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 743/1147]          blk.39.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 744/1147]          blk.39.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 745/1147]            blk.39.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 746/1147]               blk.39.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 747/1147]              blk.40.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 748/1147]           blk.40.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 749/1147]         blk.40.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 750/1147]         blk.40.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 751/1147]           blk.40.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 752/1147]         blk.40.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 753/1147]          blk.40.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 754/1147]              blk.40.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 755/1147]               blk.40.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.40.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 756/1147]               blk.40.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.40.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 757/1147]            blk.40.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 758/1147]          blk.40.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 759/1147]               blk.40.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 760/1147]               blk.40.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 761/1147]              blk.40.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 762/1147]          blk.40.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 763/1147]          blk.40.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 764/1147]            blk.40.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 765/1147]               blk.40.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 766/1147]              blk.41.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 767/1147]           blk.41.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 768/1147]         blk.41.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 769/1147]         blk.41.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 770/1147]           blk.41.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 771/1147]         blk.41.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 772/1147]          blk.41.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 773/1147]              blk.41.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 774/1147]               blk.41.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.41.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 775/1147]               blk.41.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.41.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 776/1147]            blk.41.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 777/1147]          blk.41.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 778/1147]               blk.41.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 779/1147]               blk.41.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 780/1147]              blk.41.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 781/1147]          blk.41.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 782/1147]          blk.41.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 783/1147]            blk.41.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 784/1147]               blk.41.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 785/1147]              blk.42.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 786/1147]           blk.42.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 787/1147]         blk.42.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 788/1147]         blk.42.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 789/1147]           blk.42.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 790/1147]         blk.42.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 791/1147]          blk.42.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 792/1147]              blk.42.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 793/1147]               blk.42.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.42.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 794/1147]               blk.42.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.42.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 795/1147]            blk.42.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 796/1147]          blk.42.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 797/1147]               blk.42.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 798/1147]               blk.42.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 799/1147]              blk.42.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 800/1147]          blk.42.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 801/1147]          blk.42.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 802/1147]            blk.42.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 803/1147]               blk.42.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 804/1147]              blk.43.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 805/1147]           blk.43.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 806/1147]         blk.43.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 807/1147]         blk.43.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 808/1147]           blk.43.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 809/1147]         blk.43.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 810/1147]          blk.43.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 811/1147]              blk.43.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 812/1147]               blk.43.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.43.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 813/1147]               blk.43.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.43.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 814/1147]            blk.43.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 815/1147]          blk.43.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 816/1147]               blk.43.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 817/1147]               blk.43.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 818/1147]              blk.43.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 819/1147]          blk.43.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 820/1147]          blk.43.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 821/1147]            blk.43.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 822/1147]               blk.43.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 823/1147]              blk.44.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 824/1147]           blk.44.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 825/1147]         blk.44.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 826/1147]         blk.44.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 827/1147]           blk.44.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 828/1147]         blk.44.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 829/1147]          blk.44.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 830/1147]              blk.44.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 831/1147]               blk.44.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.44.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 832/1147]               blk.44.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.44.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 833/1147]            blk.44.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 834/1147]          blk.44.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 835/1147]               blk.44.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 836/1147]               blk.44.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 837/1147]              blk.44.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 838/1147]          blk.44.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 839/1147]          blk.44.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 840/1147]            blk.44.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 841/1147]               blk.44.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 842/1147]              blk.45.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 843/1147]           blk.45.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 844/1147]         blk.45.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 845/1147]         blk.45.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 846/1147]           blk.45.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 847/1147]         blk.45.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 848/1147]          blk.45.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 849/1147]              blk.45.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 850/1147]               blk.45.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.45.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 851/1147]               blk.45.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.45.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 852/1147]            blk.45.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 853/1147]          blk.45.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 854/1147]               blk.45.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 855/1147]               blk.45.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 856/1147]              blk.45.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 857/1147]          blk.45.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 858/1147]          blk.45.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 859/1147]            blk.45.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 860/1147]               blk.45.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 861/1147]              blk.46.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 862/1147]           blk.46.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 863/1147]         blk.46.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 864/1147]         blk.46.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 865/1147]           blk.46.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 866/1147]         blk.46.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 867/1147]          blk.46.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 868/1147]              blk.46.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 869/1147]               blk.46.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.46.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 870/1147]               blk.46.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.46.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 871/1147]            blk.46.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 872/1147]          blk.46.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 873/1147]               blk.46.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 874/1147]               blk.46.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 875/1147]              blk.46.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 876/1147]          blk.46.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 877/1147]          blk.46.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 878/1147]            blk.46.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 879/1147]               blk.46.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 880/1147]              blk.47.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 881/1147]           blk.47.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 882/1147]         blk.47.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 883/1147]         blk.47.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 884/1147]           blk.47.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 885/1147]         blk.47.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 886/1147]          blk.47.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 887/1147]              blk.47.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 888/1147]               blk.47.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.47.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 889/1147]               blk.47.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.47.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 890/1147]            blk.47.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 891/1147]          blk.47.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 892/1147]               blk.47.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 893/1147]               blk.47.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 894/1147]              blk.47.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 895/1147]          blk.47.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 896/1147]          blk.47.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 897/1147]            blk.47.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 898/1147]               blk.47.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 899/1147]              blk.48.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 900/1147]           blk.48.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 901/1147]         blk.48.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 902/1147]         blk.48.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 903/1147]           blk.48.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 904/1147]         blk.48.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 905/1147]          blk.48.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 906/1147]              blk.48.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 907/1147]               blk.48.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.48.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 908/1147]               blk.48.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.48.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 909/1147]            blk.48.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 910/1147]          blk.48.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 911/1147]               blk.48.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 912/1147]               blk.48.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 913/1147]              blk.48.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 914/1147]          blk.48.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 915/1147]          blk.48.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 916/1147]            blk.48.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 917/1147]               blk.48.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 918/1147]              blk.49.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 919/1147]           blk.49.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 920/1147]         blk.49.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 921/1147]         blk.49.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 922/1147]           blk.49.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 923/1147]         blk.49.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 924/1147]          blk.49.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 925/1147]              blk.49.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 926/1147]               blk.49.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.49.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 927/1147]               blk.49.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.49.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 928/1147]            blk.49.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 929/1147]          blk.49.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 930/1147]               blk.49.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 931/1147]               blk.49.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 932/1147]              blk.49.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 933/1147]          blk.49.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 934/1147]          blk.49.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 935/1147]            blk.49.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 936/1147]               blk.49.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 937/1147]              blk.50.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 938/1147]           blk.50.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 939/1147]         blk.50.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 940/1147]         blk.50.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 941/1147]           blk.50.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 942/1147]         blk.50.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 943/1147]          blk.50.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 944/1147]              blk.50.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 945/1147]               blk.50.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.50.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 946/1147]               blk.50.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.50.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 947/1147]            blk.50.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 948/1147]          blk.50.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 949/1147]               blk.50.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 950/1147]               blk.50.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 951/1147]              blk.50.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 952/1147]          blk.50.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 953/1147]          blk.50.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 954/1147]            blk.50.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 955/1147]               blk.50.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 956/1147]              blk.51.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 957/1147]           blk.51.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 958/1147]         blk.51.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 959/1147]         blk.51.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 960/1147]           blk.51.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 961/1147]         blk.51.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 962/1147]          blk.51.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 963/1147]              blk.51.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 964/1147]               blk.51.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.51.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 965/1147]               blk.51.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.51.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 966/1147]            blk.51.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 967/1147]          blk.51.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 968/1147]               blk.51.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 969/1147]               blk.51.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 970/1147]              blk.51.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 971/1147]          blk.51.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 972/1147]          blk.51.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 973/1147]            blk.51.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 974/1147]               blk.51.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 975/1147]              blk.52.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 976/1147]           blk.52.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 977/1147]         blk.52.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 978/1147]         blk.52.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 979/1147]           blk.52.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 980/1147]         blk.52.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 981/1147]          blk.52.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 982/1147]              blk.52.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 983/1147]               blk.52.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.52.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 984/1147]               blk.52.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.52.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 985/1147]            blk.52.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 986/1147]          blk.52.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 987/1147]               blk.52.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 988/1147]               blk.52.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 989/1147]              blk.52.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 990/1147]          blk.52.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 991/1147]          blk.52.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 992/1147]            blk.52.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 993/1147]               blk.52.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 994/1147]              blk.53.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 995/1147]           blk.53.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 996/1147]         blk.53.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 997/1147]         blk.53.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 998/1147]           blk.53.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 999/1147]         blk.53.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1000/1147]          blk.53.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[1001/1147]              blk.53.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[1002/1147]               blk.53.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.53.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[1003/1147]               blk.53.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.53.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[1004/1147]            blk.53.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[1005/1147]          blk.53.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1006/1147]               blk.53.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[1007/1147]               blk.53.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[1008/1147]              blk.53.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1009/1147]          blk.53.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[1010/1147]          blk.53.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[1011/1147]            blk.53.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[1012/1147]               blk.53.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1013/1147]              blk.54.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1014/1147]           blk.54.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1015/1147]         blk.54.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[1016/1147]         blk.54.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[1017/1147]           blk.54.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[1018/1147]         blk.54.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1019/1147]          blk.54.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[1020/1147]              blk.54.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[1021/1147]               blk.54.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.54.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[1022/1147]               blk.54.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.54.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[1023/1147]            blk.54.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[1024/1147]          blk.54.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1025/1147]               blk.54.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[1026/1147]               blk.54.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[1027/1147]              blk.54.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1028/1147]          blk.54.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[1029/1147]          blk.54.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[1030/1147]            blk.54.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[1031/1147]               blk.54.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1032/1147]              blk.55.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1033/1147]           blk.55.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1034/1147]         blk.55.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[1035/1147]         blk.55.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[1036/1147]           blk.55.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[1037/1147]         blk.55.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1038/1147]          blk.55.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[1039/1147]              blk.55.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[1040/1147]               blk.55.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.55.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[1041/1147]               blk.55.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.55.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[1042/1147]            blk.55.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[1043/1147]          blk.55.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1044/1147]               blk.55.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[1045/1147]               blk.55.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[1046/1147]              blk.55.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1047/1147]          blk.55.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[1048/1147]          blk.55.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[1049/1147]            blk.55.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[1050/1147]               blk.55.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1051/1147]              blk.56.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1052/1147]           blk.56.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1053/1147]         blk.56.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[1054/1147]         blk.56.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[1055/1147]           blk.56.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[1056/1147]         blk.56.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1057/1147]          blk.56.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[1058/1147]              blk.56.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[1059/1147]               blk.56.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.56.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[1060/1147]               blk.56.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.56.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[1061/1147]            blk.56.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[1062/1147]          blk.56.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1063/1147]               blk.56.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[1064/1147]               blk.56.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[1065/1147]              blk.56.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1066/1147]          blk.56.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[1067/1147]          blk.56.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[1068/1147]            blk.56.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[1069/1147]               blk.56.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1070/1147]              blk.57.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1071/1147]           blk.57.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1072/1147]         blk.57.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[1073/1147]         blk.57.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[1074/1147]           blk.57.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[1075/1147]         blk.57.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1076/1147]          blk.57.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[1077/1147]              blk.57.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[1078/1147]               blk.57.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.57.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[1079/1147]               blk.57.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.57.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[1080/1147]            blk.57.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[1081/1147]          blk.57.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1082/1147]               blk.57.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[1083/1147]               blk.57.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[1084/1147]              blk.57.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1085/1147]          blk.57.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[1086/1147]          blk.57.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[1087/1147]            blk.57.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[1088/1147]               blk.57.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1089/1147]              blk.58.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1090/1147]           blk.58.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1091/1147]         blk.58.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[1092/1147]         blk.58.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[1093/1147]           blk.58.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[1094/1147]         blk.58.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1095/1147]          blk.58.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[1096/1147]              blk.58.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[1097/1147]               blk.58.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.58.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[1098/1147]               blk.58.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.58.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[1099/1147]            blk.58.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[1100/1147]          blk.58.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1101/1147]               blk.58.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[1102/1147]               blk.58.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[1103/1147]              blk.58.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1104/1147]          blk.58.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[1105/1147]          blk.58.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[1106/1147]            blk.58.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[1107/1147]               blk.58.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1108/1147]              blk.59.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1109/1147]           blk.59.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1110/1147]         blk.59.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[1111/1147]         blk.59.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[1112/1147]           blk.59.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[1113/1147]         blk.59.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1114/1147]          blk.59.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[1115/1147]              blk.59.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[1116/1147]               blk.59.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.59.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[1117/1147]               blk.59.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.59.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[1118/1147]            blk.59.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[1119/1147]          blk.59.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1120/1147]               blk.59.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[1121/1147]               blk.59.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[1122/1147]              blk.59.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1123/1147]          blk.59.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[1124/1147]          blk.59.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[1125/1147]            blk.59.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[1126/1147]               blk.59.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1127/1147]              blk.60.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1128/1147]           blk.60.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1129/1147]         blk.60.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[1130/1147]         blk.60.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[1131/1147]           blk.60.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[1132/1147]         blk.60.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1133/1147]          blk.60.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[1134/1147]              blk.60.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[1135/1147]               blk.60.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.60.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[1136/1147]               blk.60.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.60.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[1137/1147]            blk.60.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[1138/1147]          blk.60.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1139/1147]               blk.60.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[1140/1147]               blk.60.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[1141/1147]                        output.weight - [ 7168, 129280,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for output.weight
converting to q5_K .. size =  1767.50 MiB ->   607.58 MiB
[1142/1147]              blk.60.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1143/1147]          blk.60.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[1144/1147]          blk.60.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[1145/1147]            blk.60.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[1146/1147]               blk.60.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1147/1147]                   output_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
llama_model_quantize_internal: model size  = 1282038.27 MB
llama_model_quantize_internal: quant size  = 129853.09 MB
llama_model_quantize_internal: WARNING: 61 of 612 tensor(s) required fallback quantization

main: quantize time = 9034503.69 ms
main:    total time = 9034503.69 ms


```



</details>

---

👤 **ikawrakow** commented the **2025-02-06** at **08:59:44**:<br>

I think `token_embedding.weight` is the issue. If you use `Q8_0` instead of `Q2_K`, model size will increase by 660 MiB but quality will be quite a bit better.

Do you have an imatrix with the changed attention tensors?

---

👤 **saood06** commented the **2025-02-06** at **09:08:55**:<br>

>I think token_embedding.weight is the issue. If you use Q8_0 instead of Q2_K, model size will increase by 660 MiB but quality will be quite a bit better.

I can try that, will let you know later as this quant takes a bit of time to make.

>Do you have an imatrix with the changed attention tensors?

No, and I don't have the dataset or the compute. The new tensors are split from an old one is there a chance they could be converted from the old one?

---

👤 **ikawrakow** commented the **2025-02-06** at **09:15:48**:<br>

In that case I would simply use `Q8_0` for `attn_k_b` and `attn_v_b`. They are quite small, so model size will increase by just ~0.5 GiB.

---

👤 **saood06** commented the **2025-02-06** at **09:35:01**:<br>

> In that case I would simply use `Q8_0` for `attn_k_b` and `attn_v_b`. They are quite small, so model size will increase by just ~0.5 GiB.

I'll do that. I'll probably remake my IQ4_K_R4 with these changes.

---

👤 **ikawrakow** commented the **2025-02-06** at **09:37:43**:<br>

You may also want to change
```c++
        else if (qs.model.hparams.n_expert >= 8 && (name.find("blk.0.ffn_down") != std::string::npos ||
                                                    name.find("blk.0.ffn_gate") != std::string::npos ||
                                                    name.find("blk.0.ffn_up") != std::string::npos)) {
            new_type = GGML_TYPE_IQ3_K_R4;
        }
```
to
```c++
        else if (qs.model.hparams.n_expert >= 8 && (name.find("ffn_down.weight") != std::string::npos ||
                                                    name.find("ffn_gate.weight") != std::string::npos ||
                                                    name.find("ffn_up.weight") != std::string::npos)) {
            new_type = GGML_TYPE_IQ4_K_R4;
        }
```
This will cost ~0.4 GiB in quantized model size increase. The check is like this because in DeepSeek-Lite there is a single layer without MoE, but in DeepSeek-R1 there are 3 such layers, and my guess is that those are important to get things on the right track before the experts get involved.

---

👤 **ikawrakow** commented the **2025-02-06** at **09:45:37**:<br>

> why for attn_q and attn_k do you use Q4_K_R4 and not IQ4_K_R4

Because of copy/paste. It can be changed to `IQ4_K_R4`.

---

👤 **saood06** commented the **2025-02-06** at **14:40:00**:<br>

I changed some things but it still didn't work.
<details>
<summary>Log</summary>

```
load_imatrix: imatrix dataset='imatrix-training-full-3'
load_imatrix: loaded 720 importance matrix entries from /mnt/sda/mradermacher_DeepSeek-R1-GGUF/imatrix.dat computed on 315 chunks
prepare_imatrix: have 720 importance matrix entries
main: build = 3549 (ac732053)
main: built with gcc (Clear Linux OS for Intel Architecture) 14.2.1 20241210 releases/gcc-14.2.0-551-g21a09f0507 for x86_64-generic-linux
main: quantizing '/mnt/sda/opensourcerelease_DeepSeek-R1-bf16/opensourcerelease_DeepSeek-R1-Bf16-256x21B-F16.gguf' to '/mnt/sda/opensourcerelease_DeepSeek-R1-bf16/opensourcerelease_DeepSeek-R1-Bf16-256x21B-IQ1_S_R4_ATT2.gguf' as IQ1_S_R4 using 48 threads
llama_model_loader: loaded meta data with 48 key-value pairs and 1147 tensors from /mnt/sda/opensourcerelease_DeepSeek-R1-bf16/opensourcerelease_DeepSeek-R1-Bf16-256x21B-F16.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = opensourcerelease_DeepSeek R1 Bf16
llama_model_loader: - kv   3:                         general.size_label str              = 256x21B
llama_model_loader: - kv   4:                            general.license str              = mit
llama_model_loader: - kv   5:                   general.base_model.count u32              = 1
llama_model_loader: - kv   6:                  general.base_model.0.name str              = DeepSeek R1
llama_model_loader: - kv   7:          general.base_model.0.organization str              = Deepseek Ai
llama_model_loader: - kv   8:              general.base_model.0.repo_url str              = https://huggingface.co/deepseek-ai/De...
llama_model_loader: - kv   9:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv  10:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv  11:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv  12:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  13:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv  14:          deepseek2.attention.head_count_kv u32              = 128
llama_model_loader: - kv  15:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  16: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  17:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  18:                          general.file_type u32              = 1
llama_model_loader: - kv  19:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  20:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  21:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  22:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  23:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  24:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  25:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  26:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  27:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  28:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  29:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  30:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  31:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  32:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  33:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  34: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  35: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  36:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  37:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  38:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<▒...
llama_model_loader: - kv  39:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  40:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
llama_model_loader: - kv  41:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  42:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  43:            tokenizer.ggml.padding_token_id u32              = 1
llama_model_loader: - kv  44:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  45:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  46:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  47:               general.quantization_version u32              = 2
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type  f16:  786 tensors
================================ Have weights data with 720 entries
[   1/1147]                    token_embd.weight - [ 7168, 129280,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for token_embd.weight
converting to q8_0 .. size =  1767.50 MiB ->   938.98 MiB
[   2/1147]               blk.0.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[   3/1147]                blk.0.ffn_down.weight - [18432,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   252.00 MiB ->    86.62 MiB
[   4/1147]                blk.0.ffn_gate.weight - [ 7168, 18432,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   252.00 MiB ->    86.62 MiB
[   5/1147]                  blk.0.ffn_up.weight - [ 7168, 18432,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   252.00 MiB ->    86.62 MiB
[   6/1147]                blk.0.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[   7/1147]          blk.0.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[   8/1147]           blk.0.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[   9/1147]               blk.0.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[  10/1147]                blk.0.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.0.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  11/1147]                blk.0.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.0.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  12/1147]             blk.0.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[  13/1147]           blk.0.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  14/1147]                blk.0.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[  15/1147]                blk.0.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[  16/1147]               blk.1.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  17/1147]                blk.1.ffn_down.weight - [18432,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   252.00 MiB ->    86.62 MiB
[  18/1147]                blk.1.ffn_gate.weight - [ 7168, 18432,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   252.00 MiB ->    86.62 MiB
[  19/1147]                  blk.1.ffn_up.weight - [ 7168, 18432,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   252.00 MiB ->    86.62 MiB
[  20/1147]                blk.1.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  21/1147]          blk.1.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  22/1147]           blk.1.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[  23/1147]               blk.1.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[  24/1147]                blk.1.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.1.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  25/1147]                blk.1.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.1.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  26/1147]             blk.1.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[  27/1147]           blk.1.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  28/1147]                blk.1.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[  29/1147]                blk.1.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[  30/1147]               blk.2.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  31/1147]                blk.2.ffn_down.weight - [18432,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   252.00 MiB ->    86.62 MiB
[  32/1147]                blk.2.ffn_gate.weight - [ 7168, 18432,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   252.00 MiB ->    86.62 MiB
[  33/1147]                  blk.2.ffn_up.weight - [ 7168, 18432,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   252.00 MiB ->    86.62 MiB
[  34/1147]                blk.2.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  35/1147]          blk.2.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  36/1147]           blk.2.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[  37/1147]               blk.2.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[  38/1147]                blk.2.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.2.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  39/1147]                blk.2.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.2.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  40/1147]             blk.2.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[  41/1147]           blk.2.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  42/1147]                blk.2.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[  43/1147]                blk.2.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[  44/1147]               blk.3.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[  45/1147]            blk.3.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[  46/1147]          blk.3.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[  47/1147]          blk.3.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[  48/1147]            blk.3.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[  49/1147]          blk.3.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  50/1147]           blk.3.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[  51/1147]               blk.3.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[  52/1147]                blk.3.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.3.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  53/1147]                blk.3.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.3.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  54/1147]             blk.3.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[  55/1147]           blk.3.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  56/1147]                blk.3.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[  57/1147]                blk.3.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[  58/1147]               blk.3.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  59/1147]           blk.3.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq2_k_r4 .. size =  7168.00 MiB ->  1064.00 MiB
[  60/1147]           blk.3.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[  61/1147]             blk.3.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[  62/1147]                blk.3.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  63/1147]               blk.4.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[  64/1147]            blk.4.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[  65/1147]          blk.4.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[  66/1147]          blk.4.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[  67/1147]            blk.4.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[  68/1147]          blk.4.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  69/1147]           blk.4.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[  70/1147]               blk.4.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[  71/1147]                blk.4.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.4.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  72/1147]                blk.4.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.4.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  73/1147]             blk.4.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[  74/1147]           blk.4.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  75/1147]                blk.4.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[  76/1147]                blk.4.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[  77/1147]               blk.4.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  78/1147]           blk.4.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq2_k_r4 .. size =  7168.00 MiB ->  1064.00 MiB
[  79/1147]           blk.4.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[  80/1147]             blk.4.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[  81/1147]                blk.4.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  82/1147]          blk.5.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  83/1147]           blk.5.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[  84/1147]               blk.5.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[  85/1147]                blk.5.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.5.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  86/1147]                blk.5.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.5.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  87/1147]             blk.5.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[  88/1147]           blk.5.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  89/1147]                blk.5.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[  90/1147]                blk.5.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[  91/1147]               blk.5.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[  92/1147]            blk.5.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[  93/1147]          blk.5.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[  94/1147]          blk.5.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[  95/1147]            blk.5.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[  96/1147]               blk.5.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  97/1147]           blk.5.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq2_k_r4 .. size =  7168.00 MiB ->  1064.00 MiB
[  98/1147]           blk.5.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[  99/1147]             blk.5.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 100/1147]                blk.5.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 101/1147]               blk.6.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 102/1147]            blk.6.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 103/1147]          blk.6.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 104/1147]          blk.6.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 105/1147]            blk.6.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 106/1147]          blk.6.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 107/1147]           blk.6.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 108/1147]               blk.6.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 109/1147]                blk.6.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.6.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 110/1147]                blk.6.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.6.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 111/1147]             blk.6.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 112/1147]           blk.6.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 113/1147]                blk.6.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 114/1147]                blk.6.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 115/1147]               blk.6.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 116/1147]           blk.6.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq2_k_r4 .. size =  7168.00 MiB ->  1064.00 MiB
[ 117/1147]           blk.6.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 118/1147]             blk.6.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 119/1147]                blk.6.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 120/1147]               blk.7.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 121/1147]            blk.7.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 122/1147]          blk.7.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 123/1147]          blk.7.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 124/1147]            blk.7.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 125/1147]          blk.7.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 126/1147]           blk.7.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 127/1147]               blk.7.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 128/1147]                blk.7.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.7.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 129/1147]                blk.7.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.7.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 130/1147]             blk.7.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 131/1147]           blk.7.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 132/1147]                blk.7.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 133/1147]                blk.7.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 134/1147]               blk.7.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 135/1147]           blk.7.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 136/1147]           blk.7.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 137/1147]             blk.7.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 138/1147]                blk.7.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 139/1147]               blk.8.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 140/1147]            blk.8.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 141/1147]          blk.8.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 142/1147]          blk.8.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 143/1147]            blk.8.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 144/1147]          blk.8.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 145/1147]           blk.8.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 146/1147]               blk.8.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 147/1147]                blk.8.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.8.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 148/1147]                blk.8.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.8.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 149/1147]             blk.8.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 150/1147]           blk.8.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 151/1147]                blk.8.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 152/1147]                blk.8.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 153/1147]               blk.8.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 154/1147]           blk.8.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 155/1147]           blk.8.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 156/1147]             blk.8.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 157/1147]                blk.8.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 158/1147]               blk.9.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 159/1147]            blk.9.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 160/1147]          blk.9.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 161/1147]          blk.9.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 162/1147]            blk.9.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 163/1147]          blk.9.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 164/1147]           blk.9.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 165/1147]               blk.9.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 166/1147]                blk.9.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.9.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 167/1147]                blk.9.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.9.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 168/1147]             blk.9.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 169/1147]           blk.9.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 170/1147]                blk.9.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 171/1147]                blk.9.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 172/1147]              blk.10.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 173/1147]           blk.10.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 174/1147]         blk.10.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 175/1147]         blk.10.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 176/1147]           blk.10.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 177/1147]         blk.10.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 178/1147]          blk.10.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 179/1147]              blk.10.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 180/1147]               blk.10.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.10.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 181/1147]               blk.10.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.10.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 182/1147]            blk.10.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 183/1147]          blk.10.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 184/1147]               blk.10.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 185/1147]               blk.10.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 186/1147]               blk.9.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 187/1147]           blk.9.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 188/1147]           blk.9.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 189/1147]             blk.9.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 190/1147]                blk.9.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 191/1147]              blk.10.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 192/1147]          blk.10.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 193/1147]          blk.10.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 194/1147]            blk.10.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 195/1147]               blk.10.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 196/1147]              blk.11.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 197/1147]           blk.11.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 198/1147]         blk.11.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 199/1147]         blk.11.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 200/1147]           blk.11.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 201/1147]         blk.11.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 202/1147]          blk.11.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 203/1147]              blk.11.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 204/1147]               blk.11.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.11.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 205/1147]               blk.11.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.11.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 206/1147]            blk.11.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 207/1147]          blk.11.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 208/1147]               blk.11.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 209/1147]               blk.11.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 210/1147]              blk.11.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 211/1147]          blk.11.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 212/1147]          blk.11.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 213/1147]            blk.11.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 214/1147]               blk.11.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 215/1147]              blk.12.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 216/1147]           blk.12.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 217/1147]         blk.12.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 218/1147]         blk.12.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 219/1147]           blk.12.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 220/1147]         blk.12.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 221/1147]          blk.12.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 222/1147]              blk.12.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 223/1147]               blk.12.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.12.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 224/1147]               blk.12.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.12.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 225/1147]            blk.12.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 226/1147]          blk.12.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 227/1147]               blk.12.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 228/1147]               blk.12.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 229/1147]              blk.12.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 230/1147]          blk.12.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 231/1147]          blk.12.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 232/1147]            blk.12.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 233/1147]               blk.12.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 234/1147]              blk.13.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 235/1147]           blk.13.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 236/1147]         blk.13.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 237/1147]         blk.13.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 238/1147]           blk.13.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 239/1147]         blk.13.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 240/1147]          blk.13.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 241/1147]              blk.13.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 242/1147]               blk.13.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.13.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 243/1147]               blk.13.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.13.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 244/1147]            blk.13.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 245/1147]          blk.13.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 246/1147]               blk.13.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 247/1147]               blk.13.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 248/1147]              blk.13.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 249/1147]          blk.13.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 250/1147]          blk.13.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 251/1147]            blk.13.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 252/1147]               blk.13.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 253/1147]              blk.14.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 254/1147]           blk.14.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 255/1147]         blk.14.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 256/1147]         blk.14.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 257/1147]           blk.14.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 258/1147]         blk.14.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 259/1147]          blk.14.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 260/1147]              blk.14.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 261/1147]               blk.14.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.14.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 262/1147]               blk.14.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.14.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 263/1147]            blk.14.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 264/1147]          blk.14.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 265/1147]               blk.14.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 266/1147]               blk.14.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 267/1147]              blk.14.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 268/1147]          blk.14.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 269/1147]          blk.14.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 270/1147]            blk.14.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 271/1147]               blk.14.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 272/1147]              blk.15.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 273/1147]           blk.15.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 274/1147]         blk.15.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 275/1147]         blk.15.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 276/1147]           blk.15.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 277/1147]         blk.15.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 278/1147]          blk.15.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 279/1147]              blk.15.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 280/1147]               blk.15.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.15.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 281/1147]               blk.15.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.15.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 282/1147]            blk.15.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 283/1147]          blk.15.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 284/1147]               blk.15.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 285/1147]               blk.15.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 286/1147]              blk.15.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 287/1147]          blk.15.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 288/1147]          blk.15.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 289/1147]            blk.15.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 290/1147]               blk.15.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 291/1147]              blk.16.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 292/1147]           blk.16.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 293/1147]         blk.16.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 294/1147]         blk.16.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 295/1147]           blk.16.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 296/1147]         blk.16.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 297/1147]          blk.16.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 298/1147]              blk.16.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 299/1147]               blk.16.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.16.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 300/1147]               blk.16.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.16.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 301/1147]            blk.16.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 302/1147]          blk.16.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 303/1147]               blk.16.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 304/1147]               blk.16.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 305/1147]              blk.16.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 306/1147]          blk.16.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 307/1147]          blk.16.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 308/1147]            blk.16.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 309/1147]               blk.16.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 310/1147]              blk.17.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 311/1147]           blk.17.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 312/1147]         blk.17.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 313/1147]         blk.17.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 314/1147]           blk.17.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 315/1147]         blk.17.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 316/1147]          blk.17.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 317/1147]              blk.17.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 318/1147]               blk.17.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.17.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 319/1147]               blk.17.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.17.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 320/1147]            blk.17.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 321/1147]          blk.17.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 322/1147]               blk.17.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 323/1147]               blk.17.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 324/1147]              blk.17.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 325/1147]          blk.17.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 326/1147]          blk.17.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 327/1147]            blk.17.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 328/1147]               blk.17.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 329/1147]              blk.18.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 330/1147]           blk.18.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 331/1147]         blk.18.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 332/1147]         blk.18.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 333/1147]           blk.18.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 334/1147]         blk.18.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 335/1147]          blk.18.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 336/1147]              blk.18.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 337/1147]               blk.18.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.18.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 338/1147]               blk.18.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.18.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 339/1147]            blk.18.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 340/1147]          blk.18.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 341/1147]               blk.18.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 342/1147]               blk.18.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 343/1147]              blk.18.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 344/1147]          blk.18.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 345/1147]          blk.18.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 346/1147]            blk.18.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 347/1147]               blk.18.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 348/1147]              blk.19.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 349/1147]           blk.19.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 350/1147]         blk.19.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 351/1147]         blk.19.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 352/1147]           blk.19.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 353/1147]         blk.19.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 354/1147]          blk.19.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 355/1147]              blk.19.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 356/1147]               blk.19.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.19.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 357/1147]               blk.19.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.19.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 358/1147]            blk.19.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 359/1147]          blk.19.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 360/1147]               blk.19.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 361/1147]               blk.19.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 362/1147]              blk.19.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 363/1147]          blk.19.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 364/1147]          blk.19.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 365/1147]            blk.19.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 366/1147]               blk.19.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 367/1147]              blk.20.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 368/1147]           blk.20.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 369/1147]         blk.20.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 370/1147]         blk.20.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 371/1147]           blk.20.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 372/1147]         blk.20.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 373/1147]          blk.20.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 374/1147]              blk.20.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 375/1147]               blk.20.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.20.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 376/1147]               blk.20.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.20.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 377/1147]            blk.20.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 378/1147]          blk.20.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 379/1147]               blk.20.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 380/1147]               blk.20.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 381/1147]              blk.20.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 382/1147]          blk.20.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 383/1147]          blk.20.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 384/1147]            blk.20.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 385/1147]               blk.20.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 386/1147]              blk.21.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 387/1147]           blk.21.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 388/1147]         blk.21.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 389/1147]         blk.21.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 390/1147]           blk.21.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 391/1147]         blk.21.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 392/1147]          blk.21.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 393/1147]              blk.21.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 394/1147]               blk.21.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.21.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 395/1147]               blk.21.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.21.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 396/1147]            blk.21.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 397/1147]          blk.21.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 398/1147]               blk.21.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 399/1147]               blk.21.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 400/1147]              blk.21.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 401/1147]          blk.21.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 402/1147]          blk.21.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 403/1147]            blk.21.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 404/1147]               blk.21.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 405/1147]              blk.22.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 406/1147]           blk.22.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 407/1147]         blk.22.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 408/1147]         blk.22.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 409/1147]           blk.22.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 410/1147]         blk.22.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 411/1147]          blk.22.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 412/1147]              blk.22.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 413/1147]               blk.22.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.22.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 414/1147]               blk.22.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.22.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 415/1147]            blk.22.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 416/1147]          blk.22.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 417/1147]               blk.22.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 418/1147]               blk.22.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 419/1147]              blk.22.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 420/1147]          blk.22.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 421/1147]          blk.22.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 422/1147]            blk.22.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 423/1147]               blk.22.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 424/1147]              blk.23.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 425/1147]           blk.23.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 426/1147]         blk.23.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 427/1147]         blk.23.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 428/1147]           blk.23.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 429/1147]         blk.23.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 430/1147]          blk.23.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 431/1147]              blk.23.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 432/1147]               blk.23.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.23.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 433/1147]               blk.23.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.23.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 434/1147]            blk.23.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 435/1147]          blk.23.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 436/1147]               blk.23.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 437/1147]               blk.23.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 438/1147]              blk.23.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 439/1147]          blk.23.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 440/1147]          blk.23.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 441/1147]            blk.23.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 442/1147]               blk.23.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 443/1147]              blk.24.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 444/1147]           blk.24.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 445/1147]         blk.24.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 446/1147]         blk.24.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 447/1147]           blk.24.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 448/1147]         blk.24.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 449/1147]          blk.24.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 450/1147]              blk.24.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 451/1147]               blk.24.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.24.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 452/1147]               blk.24.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.24.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 453/1147]            blk.24.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 454/1147]          blk.24.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 455/1147]               blk.24.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 456/1147]               blk.24.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 457/1147]              blk.24.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 458/1147]          blk.24.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 459/1147]          blk.24.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 460/1147]            blk.24.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 461/1147]               blk.24.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 462/1147]              blk.25.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 463/1147]           blk.25.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 464/1147]         blk.25.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 465/1147]         blk.25.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 466/1147]           blk.25.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 467/1147]         blk.25.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 468/1147]          blk.25.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 469/1147]              blk.25.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 470/1147]               blk.25.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.25.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 471/1147]               blk.25.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.25.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 472/1147]            blk.25.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 473/1147]          blk.25.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 474/1147]               blk.25.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 475/1147]               blk.25.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 476/1147]              blk.25.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 477/1147]          blk.25.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 478/1147]          blk.25.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 479/1147]            blk.25.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 480/1147]               blk.25.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 481/1147]              blk.26.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 482/1147]           blk.26.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 483/1147]         blk.26.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 484/1147]         blk.26.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 485/1147]           blk.26.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 486/1147]         blk.26.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 487/1147]          blk.26.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 488/1147]              blk.26.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 489/1147]               blk.26.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.26.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 490/1147]               blk.26.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.26.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 491/1147]            blk.26.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 492/1147]          blk.26.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 493/1147]               blk.26.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 494/1147]               blk.26.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 495/1147]              blk.26.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 496/1147]          blk.26.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 497/1147]          blk.26.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 498/1147]            blk.26.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 499/1147]               blk.26.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 500/1147]              blk.27.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 501/1147]           blk.27.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 502/1147]         blk.27.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 503/1147]         blk.27.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 504/1147]           blk.27.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 505/1147]         blk.27.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 506/1147]          blk.27.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 507/1147]              blk.27.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 508/1147]               blk.27.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.27.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 509/1147]               blk.27.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.27.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 510/1147]            blk.27.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 511/1147]          blk.27.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 512/1147]               blk.27.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 513/1147]               blk.27.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 514/1147]              blk.27.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 515/1147]          blk.27.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 516/1147]          blk.27.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 517/1147]            blk.27.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 518/1147]               blk.27.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 519/1147]              blk.28.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 520/1147]           blk.28.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 521/1147]         blk.28.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 522/1147]         blk.28.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 523/1147]           blk.28.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 524/1147]         blk.28.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 525/1147]          blk.28.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 526/1147]              blk.28.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 527/1147]               blk.28.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.28.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 528/1147]               blk.28.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.28.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 529/1147]            blk.28.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 530/1147]          blk.28.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 531/1147]               blk.28.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 532/1147]               blk.28.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 533/1147]              blk.28.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 534/1147]          blk.28.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 535/1147]          blk.28.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 536/1147]            blk.28.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 537/1147]               blk.28.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 538/1147]              blk.29.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 539/1147]           blk.29.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 540/1147]         blk.29.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 541/1147]         blk.29.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 542/1147]           blk.29.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 543/1147]         blk.29.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 544/1147]          blk.29.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 545/1147]              blk.29.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 546/1147]               blk.29.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.29.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 547/1147]               blk.29.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.29.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 548/1147]            blk.29.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 549/1147]          blk.29.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 550/1147]               blk.29.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 551/1147]               blk.29.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 552/1147]              blk.29.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 553/1147]          blk.29.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 554/1147]          blk.29.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 555/1147]            blk.29.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 556/1147]               blk.29.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 557/1147]              blk.30.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 558/1147]           blk.30.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 559/1147]         blk.30.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 560/1147]         blk.30.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 561/1147]           blk.30.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 562/1147]         blk.30.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 563/1147]          blk.30.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 564/1147]              blk.30.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 565/1147]               blk.30.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.30.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 566/1147]               blk.30.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.30.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 567/1147]            blk.30.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 568/1147]          blk.30.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 569/1147]               blk.30.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 570/1147]               blk.30.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 571/1147]              blk.30.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 572/1147]          blk.30.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 573/1147]          blk.30.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 574/1147]            blk.30.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 575/1147]               blk.30.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 576/1147]              blk.31.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 577/1147]           blk.31.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 578/1147]         blk.31.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 579/1147]         blk.31.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 580/1147]           blk.31.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 581/1147]         blk.31.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 582/1147]          blk.31.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 583/1147]              blk.31.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 584/1147]               blk.31.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.31.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 585/1147]               blk.31.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.31.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 586/1147]            blk.31.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 587/1147]          blk.31.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 588/1147]               blk.31.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 589/1147]               blk.31.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 590/1147]              blk.31.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 591/1147]          blk.31.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 592/1147]          blk.31.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 593/1147]            blk.31.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 594/1147]               blk.31.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 595/1147]              blk.32.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 596/1147]           blk.32.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 597/1147]         blk.32.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 598/1147]         blk.32.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 599/1147]           blk.32.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 600/1147]         blk.32.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 601/1147]          blk.32.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 602/1147]              blk.32.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 603/1147]               blk.32.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.32.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 604/1147]               blk.32.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.32.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 605/1147]            blk.32.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 606/1147]          blk.32.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 607/1147]               blk.32.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 608/1147]               blk.32.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 609/1147]              blk.32.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 610/1147]          blk.32.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 611/1147]          blk.32.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 612/1147]            blk.32.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 613/1147]               blk.32.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 614/1147]              blk.33.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 615/1147]           blk.33.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 616/1147]         blk.33.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 617/1147]         blk.33.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 618/1147]           blk.33.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 619/1147]         blk.33.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 620/1147]          blk.33.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 621/1147]              blk.33.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 622/1147]               blk.33.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.33.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 623/1147]               blk.33.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.33.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 624/1147]            blk.33.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 625/1147]          blk.33.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 626/1147]               blk.33.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 627/1147]               blk.33.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 628/1147]              blk.33.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 629/1147]          blk.33.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 630/1147]          blk.33.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 631/1147]            blk.33.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 632/1147]               blk.33.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 633/1147]              blk.34.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 634/1147]           blk.34.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 635/1147]         blk.34.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 636/1147]         blk.34.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 637/1147]           blk.34.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 638/1147]         blk.34.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 639/1147]          blk.34.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 640/1147]              blk.34.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 641/1147]               blk.34.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.34.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 642/1147]               blk.34.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.34.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 643/1147]            blk.34.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 644/1147]          blk.34.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 645/1147]               blk.34.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 646/1147]               blk.34.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 647/1147]              blk.34.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 648/1147]          blk.34.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 649/1147]          blk.34.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 650/1147]            blk.34.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 651/1147]               blk.34.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 652/1147]              blk.35.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 653/1147]           blk.35.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 654/1147]         blk.35.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 655/1147]         blk.35.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 656/1147]           blk.35.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 657/1147]         blk.35.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 658/1147]          blk.35.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 659/1147]              blk.35.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 660/1147]               blk.35.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.35.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 661/1147]               blk.35.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.35.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 662/1147]            blk.35.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 663/1147]          blk.35.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 664/1147]               blk.35.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 665/1147]               blk.35.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 666/1147]              blk.35.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 667/1147]          blk.35.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 668/1147]          blk.35.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 669/1147]            blk.35.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 670/1147]               blk.35.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 671/1147]              blk.36.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 672/1147]           blk.36.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 673/1147]         blk.36.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 674/1147]         blk.36.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 675/1147]           blk.36.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 676/1147]         blk.36.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 677/1147]          blk.36.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 678/1147]              blk.36.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 679/1147]               blk.36.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.36.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 680/1147]               blk.36.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.36.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 681/1147]            blk.36.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 682/1147]          blk.36.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 683/1147]               blk.36.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 684/1147]               blk.36.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 685/1147]              blk.36.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 686/1147]          blk.36.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 687/1147]          blk.36.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 688/1147]            blk.36.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 689/1147]               blk.36.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 690/1147]              blk.37.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 691/1147]           blk.37.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 692/1147]         blk.37.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 693/1147]         blk.37.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 694/1147]           blk.37.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 695/1147]         blk.37.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 696/1147]          blk.37.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 697/1147]              blk.37.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 698/1147]               blk.37.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.37.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 699/1147]               blk.37.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.37.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 700/1147]            blk.37.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 701/1147]          blk.37.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 702/1147]               blk.37.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 703/1147]               blk.37.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 704/1147]              blk.37.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 705/1147]          blk.37.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 706/1147]          blk.37.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 707/1147]            blk.37.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 708/1147]               blk.37.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 709/1147]              blk.38.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 710/1147]           blk.38.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 711/1147]         blk.38.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 712/1147]         blk.38.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 713/1147]           blk.38.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 714/1147]         blk.38.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 715/1147]          blk.38.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 716/1147]              blk.38.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 717/1147]               blk.38.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.38.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 718/1147]               blk.38.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.38.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 719/1147]            blk.38.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 720/1147]          blk.38.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 721/1147]               blk.38.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 722/1147]               blk.38.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 723/1147]              blk.38.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 724/1147]          blk.38.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 725/1147]          blk.38.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 726/1147]            blk.38.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 727/1147]               blk.38.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 728/1147]              blk.39.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 729/1147]           blk.39.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 730/1147]         blk.39.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 731/1147]         blk.39.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 732/1147]           blk.39.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 733/1147]         blk.39.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 734/1147]          blk.39.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 735/1147]              blk.39.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 736/1147]               blk.39.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.39.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 737/1147]               blk.39.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.39.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 738/1147]            blk.39.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 739/1147]          blk.39.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 740/1147]               blk.39.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 741/1147]               blk.39.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 742/1147]              blk.39.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 743/1147]          blk.39.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 744/1147]          blk.39.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 745/1147]            blk.39.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 746/1147]               blk.39.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 747/1147]              blk.40.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 748/1147]           blk.40.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 749/1147]         blk.40.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 750/1147]         blk.40.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 751/1147]           blk.40.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 752/1147]         blk.40.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 753/1147]          blk.40.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 754/1147]              blk.40.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 755/1147]               blk.40.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.40.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 756/1147]               blk.40.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.40.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 757/1147]            blk.40.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 758/1147]          blk.40.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 759/1147]               blk.40.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 760/1147]               blk.40.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 761/1147]              blk.40.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 762/1147]          blk.40.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 763/1147]          blk.40.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 764/1147]            blk.40.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 765/1147]               blk.40.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 766/1147]              blk.41.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 767/1147]           blk.41.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 768/1147]         blk.41.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 769/1147]         blk.41.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 770/1147]           blk.41.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 771/1147]         blk.41.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 772/1147]          blk.41.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 773/1147]              blk.41.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 774/1147]               blk.41.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.41.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 775/1147]               blk.41.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.41.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 776/1147]            blk.41.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 777/1147]          blk.41.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 778/1147]               blk.41.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 779/1147]               blk.41.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 780/1147]              blk.41.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 781/1147]          blk.41.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 782/1147]          blk.41.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 783/1147]            blk.41.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 784/1147]               blk.41.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 785/1147]              blk.42.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 786/1147]           blk.42.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 787/1147]         blk.42.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 788/1147]         blk.42.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 789/1147]           blk.42.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 790/1147]         blk.42.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 791/1147]          blk.42.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 792/1147]              blk.42.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 793/1147]               blk.42.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.42.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 794/1147]               blk.42.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.42.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 795/1147]            blk.42.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 796/1147]          blk.42.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 797/1147]               blk.42.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 798/1147]               blk.42.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 799/1147]              blk.42.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 800/1147]          blk.42.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 801/1147]          blk.42.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 802/1147]            blk.42.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 803/1147]               blk.42.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 804/1147]              blk.43.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 805/1147]           blk.43.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 806/1147]         blk.43.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 807/1147]         blk.43.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 808/1147]           blk.43.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 809/1147]         blk.43.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 810/1147]          blk.43.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 811/1147]              blk.43.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 812/1147]               blk.43.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.43.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 813/1147]               blk.43.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.43.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 814/1147]            blk.43.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 815/1147]          blk.43.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 816/1147]               blk.43.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 817/1147]               blk.43.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 818/1147]              blk.43.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 819/1147]          blk.43.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 820/1147]          blk.43.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 821/1147]            blk.43.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 822/1147]               blk.43.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 823/1147]              blk.44.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 824/1147]           blk.44.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 825/1147]         blk.44.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 826/1147]         blk.44.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 827/1147]           blk.44.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 828/1147]         blk.44.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 829/1147]          blk.44.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 830/1147]              blk.44.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 831/1147]               blk.44.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.44.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 832/1147]               blk.44.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.44.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 833/1147]            blk.44.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 834/1147]          blk.44.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 835/1147]               blk.44.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 836/1147]               blk.44.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 837/1147]              blk.44.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 838/1147]          blk.44.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 839/1147]          blk.44.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 840/1147]            blk.44.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 841/1147]               blk.44.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 842/1147]              blk.45.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 843/1147]           blk.45.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 844/1147]         blk.45.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 845/1147]         blk.45.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 846/1147]           blk.45.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 847/1147]         blk.45.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 848/1147]          blk.45.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 849/1147]              blk.45.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 850/1147]               blk.45.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.45.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 851/1147]               blk.45.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.45.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 852/1147]            blk.45.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 853/1147]          blk.45.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 854/1147]               blk.45.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 855/1147]               blk.45.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 856/1147]              blk.45.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 857/1147]          blk.45.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 858/1147]          blk.45.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 859/1147]            blk.45.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 860/1147]               blk.45.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 861/1147]              blk.46.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 862/1147]           blk.46.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 863/1147]         blk.46.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 864/1147]         blk.46.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 865/1147]           blk.46.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 866/1147]         blk.46.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 867/1147]          blk.46.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 868/1147]              blk.46.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 869/1147]               blk.46.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.46.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 870/1147]               blk.46.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.46.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 871/1147]            blk.46.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 872/1147]          blk.46.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 873/1147]               blk.46.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 874/1147]               blk.46.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 875/1147]              blk.46.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 876/1147]          blk.46.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 877/1147]          blk.46.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 878/1147]            blk.46.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 879/1147]               blk.46.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 880/1147]              blk.47.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 881/1147]           blk.47.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 882/1147]         blk.47.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 883/1147]         blk.47.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 884/1147]           blk.47.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 885/1147]         blk.47.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 886/1147]          blk.47.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 887/1147]              blk.47.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 888/1147]               blk.47.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.47.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 889/1147]               blk.47.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.47.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 890/1147]            blk.47.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 891/1147]          blk.47.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 892/1147]               blk.47.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 893/1147]               blk.47.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 894/1147]              blk.47.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 895/1147]          blk.47.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 896/1147]          blk.47.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 897/1147]            blk.47.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 898/1147]               blk.47.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 899/1147]              blk.48.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 900/1147]           blk.48.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 901/1147]         blk.48.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 902/1147]         blk.48.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 903/1147]           blk.48.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 904/1147]         blk.48.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 905/1147]          blk.48.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 906/1147]              blk.48.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 907/1147]               blk.48.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.48.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 908/1147]               blk.48.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.48.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 909/1147]            blk.48.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 910/1147]          blk.48.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 911/1147]               blk.48.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 912/1147]               blk.48.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 913/1147]              blk.48.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 914/1147]          blk.48.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 915/1147]          blk.48.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 916/1147]            blk.48.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 917/1147]               blk.48.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 918/1147]              blk.49.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 919/1147]           blk.49.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 920/1147]         blk.49.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 921/1147]         blk.49.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 922/1147]           blk.49.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 923/1147]         blk.49.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 924/1147]          blk.49.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 925/1147]              blk.49.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 926/1147]               blk.49.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.49.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 927/1147]               blk.49.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.49.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 928/1147]            blk.49.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 929/1147]          blk.49.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 930/1147]               blk.49.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 931/1147]               blk.49.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 932/1147]              blk.49.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 933/1147]          blk.49.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 934/1147]          blk.49.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 935/1147]            blk.49.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 936/1147]               blk.49.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 937/1147]              blk.50.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 938/1147]           blk.50.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 939/1147]         blk.50.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 940/1147]         blk.50.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 941/1147]           blk.50.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 942/1147]         blk.50.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 943/1147]          blk.50.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 944/1147]              blk.50.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 945/1147]               blk.50.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.50.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 946/1147]               blk.50.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.50.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 947/1147]            blk.50.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 948/1147]          blk.50.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 949/1147]               blk.50.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 950/1147]               blk.50.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 951/1147]              blk.50.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 952/1147]          blk.50.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 953/1147]          blk.50.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 954/1147]            blk.50.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 955/1147]               blk.50.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 956/1147]              blk.51.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 957/1147]           blk.51.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 958/1147]         blk.51.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 959/1147]         blk.51.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 960/1147]           blk.51.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 961/1147]         blk.51.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 962/1147]          blk.51.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 963/1147]              blk.51.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 964/1147]               blk.51.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.51.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 965/1147]               blk.51.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.51.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 966/1147]            blk.51.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 967/1147]          blk.51.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 968/1147]               blk.51.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 969/1147]               blk.51.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 970/1147]              blk.51.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 971/1147]          blk.51.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 972/1147]          blk.51.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 973/1147]            blk.51.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 974/1147]               blk.51.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 975/1147]              blk.52.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 976/1147]           blk.52.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 977/1147]         blk.52.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 978/1147]         blk.52.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 979/1147]           blk.52.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 980/1147]         blk.52.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 981/1147]          blk.52.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[ 982/1147]              blk.52.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[ 983/1147]               blk.52.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.52.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 984/1147]               blk.52.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.52.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 985/1147]            blk.52.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[ 986/1147]          blk.52.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 987/1147]               blk.52.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[ 988/1147]               blk.52.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[ 989/1147]              blk.52.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 990/1147]          blk.52.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[ 991/1147]          blk.52.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 992/1147]            blk.52.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[ 993/1147]               blk.52.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 994/1147]              blk.53.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 995/1147]           blk.53.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 996/1147]         blk.53.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 997/1147]         blk.53.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 998/1147]           blk.53.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[ 999/1147]         blk.53.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1000/1147]          blk.53.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[1001/1147]              blk.53.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[1002/1147]               blk.53.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.53.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1003/1147]               blk.53.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.53.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1004/1147]            blk.53.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[1005/1147]          blk.53.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1006/1147]               blk.53.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[1007/1147]               blk.53.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[1008/1147]              blk.53.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1009/1147]          blk.53.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[1010/1147]          blk.53.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[1011/1147]            blk.53.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[1012/1147]               blk.53.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1013/1147]              blk.54.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1014/1147]           blk.54.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1015/1147]         blk.54.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[1016/1147]         blk.54.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[1017/1147]           blk.54.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[1018/1147]         blk.54.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1019/1147]          blk.54.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[1020/1147]              blk.54.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[1021/1147]               blk.54.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.54.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1022/1147]               blk.54.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.54.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1023/1147]            blk.54.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[1024/1147]          blk.54.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1025/1147]               blk.54.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[1026/1147]               blk.54.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[1027/1147]              blk.54.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1028/1147]          blk.54.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[1029/1147]          blk.54.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[1030/1147]            blk.54.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[1031/1147]               blk.54.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1032/1147]              blk.55.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1033/1147]           blk.55.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1034/1147]         blk.55.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[1035/1147]         blk.55.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[1036/1147]           blk.55.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[1037/1147]         blk.55.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1038/1147]          blk.55.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[1039/1147]              blk.55.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[1040/1147]               blk.55.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.55.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1041/1147]               blk.55.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.55.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1042/1147]            blk.55.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[1043/1147]          blk.55.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1044/1147]               blk.55.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[1045/1147]               blk.55.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[1046/1147]              blk.55.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1047/1147]          blk.55.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[1048/1147]          blk.55.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[1049/1147]            blk.55.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[1050/1147]               blk.55.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1051/1147]              blk.56.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1052/1147]           blk.56.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1053/1147]         blk.56.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[1054/1147]         blk.56.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[1055/1147]           blk.56.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[1056/1147]         blk.56.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1057/1147]          blk.56.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[1058/1147]              blk.56.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[1059/1147]               blk.56.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.56.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1060/1147]               blk.56.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.56.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1061/1147]            blk.56.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[1062/1147]          blk.56.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1063/1147]               blk.56.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[1064/1147]               blk.56.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[1065/1147]              blk.56.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1066/1147]          blk.56.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[1067/1147]          blk.56.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[1068/1147]            blk.56.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[1069/1147]               blk.56.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1070/1147]              blk.57.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1071/1147]           blk.57.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1072/1147]         blk.57.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[1073/1147]         blk.57.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[1074/1147]           blk.57.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[1075/1147]         blk.57.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1076/1147]          blk.57.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[1077/1147]              blk.57.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[1078/1147]               blk.57.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.57.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1079/1147]               blk.57.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.57.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1080/1147]            blk.57.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[1081/1147]          blk.57.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1082/1147]               blk.57.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[1083/1147]               blk.57.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[1084/1147]              blk.57.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1085/1147]          blk.57.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[1086/1147]          blk.57.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[1087/1147]            blk.57.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[1088/1147]               blk.57.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1089/1147]              blk.58.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1090/1147]           blk.58.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1091/1147]         blk.58.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[1092/1147]         blk.58.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[1093/1147]           blk.58.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[1094/1147]         blk.58.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1095/1147]          blk.58.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[1096/1147]              blk.58.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[1097/1147]               blk.58.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.58.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1098/1147]               blk.58.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.58.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1099/1147]            blk.58.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[1100/1147]          blk.58.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1101/1147]               blk.58.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[1102/1147]               blk.58.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[1103/1147]              blk.58.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1104/1147]          blk.58.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[1105/1147]          blk.58.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[1106/1147]            blk.58.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[1107/1147]               blk.58.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1108/1147]              blk.59.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1109/1147]           blk.59.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1110/1147]         blk.59.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[1111/1147]         blk.59.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[1112/1147]           blk.59.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[1113/1147]         blk.59.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1114/1147]          blk.59.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[1115/1147]              blk.59.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[1116/1147]               blk.59.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.59.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1117/1147]               blk.59.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.59.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1118/1147]            blk.59.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[1119/1147]          blk.59.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1120/1147]               blk.59.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[1121/1147]               blk.59.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[1122/1147]              blk.59.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1123/1147]          blk.59.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[1124/1147]          blk.59.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[1125/1147]            blk.59.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[1126/1147]               blk.59.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1127/1147]              blk.60.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1128/1147]           blk.60.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1129/1147]         blk.60.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[1130/1147]         blk.60.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[1131/1147]           blk.60.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to q6_k_r4 .. size =    28.00 MiB ->    11.48 MiB
[1132/1147]         blk.60.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1133/1147]          blk.60.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =     7.88 MiB ->     2.71 MiB
[1134/1147]              blk.60.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    32.00 MiB ->    11.00 MiB
[1135/1147]               blk.60.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.60.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1136/1147]               blk.60.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.60.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1137/1147]            blk.60.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq5_k_r4 .. size =   224.00 MiB ->    77.00 MiB
[1138/1147]          blk.60.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1139/1147]               blk.60.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    21.00 MiB ->     7.22 MiB
[1140/1147]               blk.60.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq5_k_r4 .. size =    72.00 MiB ->    24.75 MiB
[1141/1147]                        output.weight - [ 7168, 129280,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for output.weight
converting to q5_K .. size =  1767.50 MiB ->   607.58 MiB
[1142/1147]              blk.60.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1143/1147]          blk.60.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   675.50 MiB
[1144/1147]          blk.60.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[1145/1147]            blk.60.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq1_s_r4 .. size =  7168.00 MiB ->   673.00 MiB
[1146/1147]               blk.60.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1147/1147]                   output_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
llama_model_quantize_internal: model size  = 1282038.27 MB
llama_model_quantize_internal: quant size  = 132055.59 MB

main: quantize time = 9295125.73 ms
main:    total time = 9295125.73 ms
```

---

👤 **ikawrakow** commented the **2025-02-06** at **14:46:28**:<br>

When you say "It didn't work", how did it not work? Produced NaNs? Produced gibberish? Produced something like human language but with no real meaning? It isn't as coherent as a higher bit quantization?

---

👤 **saood06** commented the **2025-02-06** at **15:00:37**:<br>

>When you say "It didn't work", how did it not work? Produced NaNs? Produced gibberish? Produced something like human language but with no real meaning? It isn't as coherent as a higher bit quantization?

Original one produced just NaNs
Second one produced one token before NaN and the token distribution of that one token compared to my highest quality working quant is only vaguely similar token distribution
IQ1_S_R4 single token
 Even : 0.4562944173812866
" But" : 0.16470757126808167
" It" : 0.08828949928283691
" I": 0.05235012248158455
" She": 0.04799338057637215
" Now": 0.0435505285859108
" The" : 0.025533469393849373
" Sometimes" : 0.018458260223269463
" \\n\\n" : 0.01704910397529602
" When" : 0.015356291085481644
IQ4_K_R4 single token
" But" : 0.6323568224906921
" Even" : 0.2135329246520996
" It" : 0.07232297211885452
" I" : 0.03508976474404335
" As" : 0.014349701814353466
" Now" : 0.008230382576584816
" However" : 0.007817259058356285
" \\n\\n" : 0.0060447207652032375
" And" : 0.005831697024405003
" For" : 0.004423711448907852

---

👤 **saood06** submitted a review the **2025-02-06** at **15:16:38**: 💬 `COMMENTED`

---

👤 **saood06** commented during a code review the **2025-02-06** at **15:16:38** on `src/llama.cpp`:<br>

Could this need to be higher for R1? The unsloth quant does this up to and including layer 8, my most recent attempt only did up to and including layer 6.

---

👤 **ikawrakow** commented the **2025-02-06** at **15:28:21**:<br>

Hmm, not sure. The token probabilities are not completely useless (same top-4 tokens). It is possible the imatrix is not adequate. 4+ bpw quants work even without an imatrix, so a bad imatrix is not immediately recognizable. I see in the log that 315 chunks were used. We have 8 out of 256 experts being active, so each expert got on average less than 10 chunks. That's not a lot of data to properly determine the relative importance of the tensor columns.

In case you have time and energy:
* Can you try without MLA? I took your PR #180 and made MLA optional (see #188). While testing I noticed that one gets different results and, without having done any meaningful evaluation, my impression was that MLA produced worse responses (tested with DeepSeek-Lite using `f16` to not worry about quantization effects).
* Have you tried running perplexity? Just a few chunks to compare to your best quantized model

It is of course also possible that removing the super-block scale in `IQ1_S_R4` was not a good move. It didn't have any impact on DeepSeek-Lite, but having 3-bit block scales with just a single row scale is risky, and may result in too much precision loss in case there are big magnitude variations in the model weights.

---

👤 **ikawrakow** submitted a review the **2025-02-06** at **15:30:35**: 💬 `COMMENTED`

---

👤 **ikawrakow** commented during a code review the **2025-02-06** at **15:30:35** on `src/llama.cpp`:<br>

Yes, the early layers tend to be more important, so increasing the number of layers and/or increasing the bpw of the quantization used will improve results. It is basically a matter of the balance between quantization quality and model size.

---

👤 **saood06** commented the **2025-02-06** at **16:06:00**:<br>

>It is possible the imatrix is not adequate. 4+ bpw quants work even without an imatrix, so a bad imatrix is not immediately recognizable. I see in the log that 315 chunks were used.

The one unsloth uses is significantly shorter, only 124. I also do believe the imatrix data is better. The Arctic MoE his imatrix activated all but one expert and they tried hard to get the last one to no avail. All other imatrix activated far less.

>Can you try without MLA? I took your PR https://github.com/ikawrakow/ik_llama.cpp/pull/180 and made MLA optional (see https://github.com/ikawrakow/ik_llama.cpp/pull/188). While testing I noticed that one gets different results and, without having done any meaningful evaluation, my impression was that MLA produced worse responses (tested with DeepSeek-Lite using f16 to not worry about quantization effects).

I think this is to be expected. It is a whole different attention mechanism. MLA uses less bits to represents the KV, it is far better at conserving information while compressing the KV cache compared to GQA, but it is still less bits than MHA. They claim it is better than MHA because redundancy in information between heads means you do have some effectively lossless compression. But I've seen enough people actually micro benchmark MHA and MLA and it does seem a bit worse.

The real benefit of MLA is that it uses less bits, and there was a branch I was working on which allowed me to make use of that (thanks to another one of fairydreaming's PR), which uses mmap to avoid allocating KV until used which means the old gigantic KV (full 128k is ~600 GB), does not allocate and start paging me out. I was able to request 64K of context ( CPU NUMA KV buffer size = 313101.56 MiB ) from server and I used 30K before ending that test, and it never paged to disk thanks to the mmap only allocating what was used.

I saw your PR #188 , there was some minor optimizations from fairydreaming that have that haven't made it to my PR ( #180 ) , along with some other stuff from fairydreaming that is experimental (mmap) and QoL stuff (MoE warmup actually loads in all experts).

Although the mmap allocator is working for me (and I might create a PR with it being toggled via a CLI argument) I think when MLA is toggled on the other KV cache should not allocate.

>Have you tried running perplexity? Just a few chunks to compare to your best quantized model
>Can you try without MLA?

When I have some more time I will.

---

👤 **saood06** submitted a review the **2025-02-06** at **16:18:14**: 💬 `COMMENTED`

---

👤 **saood06** commented during a code review the **2025-02-06** at **16:18:14** on `src/llama.cpp`:<br>

>in DeepSeek-Lite there is a single layer without MoE, but in DeepSeek-R1 there are 3 such layers

The additional 2 layers of dense, means you hit 2 less MoE layers with this then you on Lite, and this is still the only meaningful way I can see that the quant I just made is worse, basically everything else is better, or the same.

---

👤 **saood06** commented the **2025-02-06** at **20:26:59**:<br>

@ikawrakow 

>Have you tried running perplexity? Just a few chunks to compare to your best quantized model

Model 	       | [1] |	[2] 	|[3] 	|[4] |	[5] 	|[6]| 	[7]| 	[8]	|[9]	|[10]	|[11]	|[12]
--- 	       | --- 	| --- 	| --- 	|--- 	|--- 	|--- 	|--- 	|--- 	|--- 	|--- 	|--- | ---
IQ2_XXS **|	3.39| 	4.56|	3.44| 	3.27| 	3.27| 	3.20| 	3.12 |	3.12|
IQ3_XXS ** |	2.69 |	3.53|	2.51 |	2.11 |	1.91 |	1.78 |	1.69 |	1.62|
IQ4_K_R4 (V1) |  2.5954 | 3.3338|	2.3993	|1.9972	|1.8080	|1.6659	|1.5697|	1.5047|	1.4555|	1.4154|	1.4007|	1.4493
UD-IQ1_M **|	3.4155  |4.2311 | 3.0817 | 2.8601 | 2.6933 | 2.5792 | 2.5123 | 2.5239  
UD-IQ1_S **    | 3.8939  |4.7189 | 3.7812 | 3.6799 | 3.6215 | 3.6922 | 3.6442|  3.7472|  3.8353|	3.7663|	3.8983|	4.0621
IQ1_S_R4 (V2)|	3.7554	|4.6569	|3.5681	|3.4458|	nan|	nan|	nan|	nan|	nan|	nan|	nan|	nan|	nan|	nan|	nan|	nan|

** is data that was posted by other people online, not my tests.
UD refers to Unsloth quants.
(V2) for IQ1_S_R4 refers to the one that had the one token
(V1) for IQ4_K_R4 refers to the fact that I plan to requant this.

---

👤 **ikawrakow** commented the **2025-02-07** at **06:33:14**:<br>

@saood06  Thanks for these results.

So, it looks like `IQ1_S_R4` is better than Unsloth's until something goes wrong. There seems to be an issue in `ggml` itself as the result is supposed to be independent of batch size, but it isn't in the `IQ1_S_R4` runs where we get `NaN` in the 5th chunk with the default batch size and not `NaN` with a batch size of 4096. Something strange happens in the 5th chunk as `IQ1_S_R4` PPL with batch size 4096 is higher than the 4th chunk while it is lower for all other quants.

I have added some extra guards in #191, but they never trigger with DeepSeek-Lite or LLaMA-3.1-8B-Instruct, so not sure if this will help. It may be useful to try `IQ1_M_R4` and see how that goes.

---

👤 **ikawrakow** commented the **2025-02-07** at **10:05:20**:<br>

@saood06 I would appreciate if you tried running the `IQ1_S_R4` DeepSeek-R1 model with #192. There appears to be a race on the main branch that can cause the NaNs, and #192 hopefully fixes that.

---

👤 **saood06** commented the **2025-02-07** at **22:41:11**:<br>

@ikawrakow 

I have tested #192 by merging it into my WIP testing branch, saood06/ik_llama.cpp/pull/1. IQ1_S_R4 (V2) and in my single very basic test it now functions (produced coherent output), but it still produced `NaN` in the perplexity test from chunk 13 and on, and the perplexity values for it and other quants have changed slightly compared to previously. No results for IQ1_S_R4 (V1) as I deleted that and don't feel like recreating it.

Only including new results in the table below.

Quant 	       | [1] |	[2] 	|[3] 	|[4] |	[5] 	|[6]| 	[7]| 	[8]	|[9]	|[10]	|[11]	|[12]|[13]|[14]|[15]|[16]|[17]|[18]|[19]|[20]|[21]|[22]|[23]|[24]
--- 	               | --- 	| --- 	| --- 	|--- 	|--- 	|--- 	|--- 	|--- 	|--- 	|--- 	|--- | ---|---|---|---|---|---|---|---|---|---|---|---|---
IQ4_K_R4 (V1) |2.5944|3.3242|2.4001|1.9949|1.8067|1.6666|1.5704|1.5055|1.4559|1.4154|1.3999|1.4404|1.4500|1.5786|1.7101|1.7729|1.9347|2.0639|2.0260|2.0157|2.1257|2.0994|2.0710|2.0844,
IQ4_K_R4 (V2) |2.5474|3.3247|2.4001|2.0029|1.8181|1.6716|1.5734|1.5084|1.4592|1.4194|1.4035|1.4376|1.4476|1.5734|1.7047|1.7654|1.9276|2.0560|2.0189|2.0066|2.1138|2.0865|2.0588|2.0738
IQ1_S_R4 (V2) |3.7087|4.6034|3.5369|3.4023|3.5178|3.5631|3.5441|3.6670|3.7329|3.6657|3.7786|3.9536|nan|nan|nan|nan|nan|nan|nan|nan|nan|nan|nan|nan

IQ4_K_R4 (V2) is slower (2.63 t/s for V2 vs 3.22 t/s V1) for TG probably because it uses IQ6_K as IQ6_K_R4 does not exist, and thus for now I still think I prefer V1 even with its flaws.

Off topic but when should you use Q8_K_R8 vs Q8_0_R8?

Also there may be some MLA quality issues, there is some discussion happening over at https://github.com/ggerganov/llama.cpp/pull/11446 where setting GGML_TYPE_F32 for some tensors helped quality (GGML_TYPE_F16 for those tensors broke it, while Q8_0 worked but with noticeably degraded performance).

<details>
<summary>IQ4_K_R4 V1 quantization logs</summary>

load_imatrix: imatrix dataset='imatrix-training-full-3'
load_imatrix: loaded 720 importance matrix entries from /mnt/sda/mradermacher_DeepSeek-R1-GGUF/imatrix.dat computed on 315 chunks
prepare_imatrix: have 720 importance matrix entries
main: build = 3539 (31744dd4)
main: built with gcc (Clear Linux OS for Intel Architecture) 14.2.1 20241210 releases/gcc-14.2.0-551-g21a09f0507 for x86_64-generic-linux
main: quantizing '/mnt/sda/opensourcerelease_DeepSeek-R1-bf16/opensourcerelease_DeepSeek-R1-Bf16-256x21B-F16.gguf' to '/mnt/sda/opensourcerelease_DeepSeek-R1-bf16/opensourcerelease_DeepSeek-R1-Bf16-256x21B-IQ4_K_R4.gguf' as IQ4_K_R4 using 48 threads
llama_model_loader: loaded meta data with 48 key-value pairs and 1147 tensors from /mnt/sda/opensourcerelease_DeepSeek-R1-bf16/opensourcerelease_DeepSeek-R1-Bf16-256x21B-F16.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = opensourcerelease_DeepSeek R1 Bf16
llama_model_loader: - kv   3:                         general.size_label str              = 256x21B
llama_model_loader: - kv   4:                            general.license str              = mit
llama_model_loader: - kv   5:                   general.base_model.count u32              = 1
llama_model_loader: - kv   6:                  general.base_model.0.name str              = DeepSeek R1
llama_model_loader: - kv   7:          general.base_model.0.organization str              = Deepseek Ai
llama_model_loader: - kv   8:              general.base_model.0.repo_url str              = https://huggingface.co/deepseek-ai/De...
llama_model_loader: - kv   9:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv  10:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv  11:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv  12:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  13:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv  14:          deepseek2.attention.head_count_kv u32              = 128
llama_model_loader: - kv  15:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  16: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  17:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  18:                          general.file_type u32              = 1
llama_model_loader: - kv  19:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  20:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  21:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  22:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  23:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  24:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  25:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  26:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  27:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  28:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  29:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  30:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  31:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  32:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  33:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  34: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  35: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  36:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  37:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  38:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<▒...
llama_model_loader: - kv  39:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  40:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
llama_model_loader: - kv  41:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  42:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  43:            tokenizer.ggml.padding_token_id u32              = 1
llama_model_loader: - kv  44:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  45:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  46:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  47:               general.quantization_version u32              = 2
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type  f16:  786 tensors
================================ Have weights data with 720 entries
[   1/1147]                    token_embd.weight - [ 7168, 129280,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for token_embd.weight
converting to iq4_k .. size =  1767.50 MiB ->   497.11 MiB
[   2/1147]               blk.0.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[   3/1147]                blk.0.ffn_down.weight - [18432,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =   252.00 MiB ->    70.88 MiB
[   4/1147]                blk.0.ffn_gate.weight - [ 7168, 18432,     1,     1], type =    f16, converting to iq4_k_r4 .. size =   252.00 MiB ->    70.88 MiB
[   5/1147]                  blk.0.ffn_up.weight - [ 7168, 18432,     1,     1], type =    f16, converting to iq4_k_r4 .. size =   252.00 MiB ->    70.88 MiB
[   6/1147]                blk.0.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[   7/1147]          blk.0.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[   8/1147]           blk.0.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[   9/1147]               blk.0.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[  10/1147]                blk.0.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.0.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[  11/1147]                blk.0.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.0.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[  12/1147]             blk.0.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_K .. size =   224.00 MiB ->    77.00 MiB
[  13/1147]           blk.0.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  14/1147]                blk.0.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[  15/1147]                blk.0.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[  16/1147]               blk.1.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  17/1147]                blk.1.ffn_down.weight - [18432,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =   252.00 MiB ->    70.88 MiB
[  18/1147]                blk.1.ffn_gate.weight - [ 7168, 18432,     1,     1], type =    f16, converting to iq4_k_r4 .. size =   252.00 MiB ->    70.88 MiB
[  19/1147]                  blk.1.ffn_up.weight - [ 7168, 18432,     1,     1], type =    f16, converting to iq4_k_r4 .. size =   252.00 MiB ->    70.88 MiB
[  20/1147]                blk.1.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  21/1147]          blk.1.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  22/1147]           blk.1.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[  23/1147]               blk.1.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[  24/1147]                blk.1.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.1.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[  25/1147]                blk.1.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.1.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[  26/1147]             blk.1.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_K .. size =   224.00 MiB ->    77.00 MiB
[  27/1147]           blk.1.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  28/1147]                blk.1.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[  29/1147]                blk.1.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[  30/1147]               blk.2.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  31/1147]                blk.2.ffn_down.weight - [18432,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =   252.00 MiB ->    70.88 MiB
[  32/1147]                blk.2.ffn_gate.weight - [ 7168, 18432,     1,     1], type =    f16, converting to iq4_k_r4 .. size =   252.00 MiB ->    70.88 MiB
[  33/1147]                  blk.2.ffn_up.weight - [ 7168, 18432,     1,     1], type =    f16, converting to iq4_k_r4 .. size =   252.00 MiB ->    70.88 MiB
[  34/1147]                blk.2.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  35/1147]          blk.2.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  36/1147]           blk.2.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[  37/1147]               blk.2.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[  38/1147]                blk.2.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.2.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[  39/1147]                blk.2.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.2.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[  40/1147]             blk.2.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_K .. size =   224.00 MiB ->    77.00 MiB
[  41/1147]           blk.2.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  42/1147]                blk.2.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[  43/1147]                blk.2.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[  44/1147]               blk.3.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[  45/1147]            blk.3.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[  46/1147]          blk.3.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[  47/1147]          blk.3.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[  48/1147]            blk.3.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[  49/1147]          blk.3.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  50/1147]           blk.3.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[  51/1147]               blk.3.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[  52/1147]                blk.3.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.3.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[  53/1147]                blk.3.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.3.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[  54/1147]             blk.3.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_K .. size =   224.00 MiB ->    77.00 MiB
[  55/1147]           blk.3.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  56/1147]                blk.3.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[  57/1147]                blk.3.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[  58/1147]               blk.3.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  59/1147]           blk.3.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[  60/1147]           blk.3.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[  61/1147]             blk.3.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[  62/1147]                blk.3.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  63/1147]               blk.4.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[  64/1147]            blk.4.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[  65/1147]          blk.4.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[  66/1147]          blk.4.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[  67/1147]            blk.4.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[  68/1147]          blk.4.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  69/1147]           blk.4.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[  70/1147]               blk.4.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[  71/1147]                blk.4.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.4.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[  72/1147]                blk.4.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.4.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[  73/1147]             blk.4.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_K .. size =   224.00 MiB ->    77.00 MiB
[  74/1147]           blk.4.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  75/1147]                blk.4.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[  76/1147]                blk.4.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[  77/1147]               blk.4.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  78/1147]           blk.4.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[  79/1147]           blk.4.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[  80/1147]             blk.4.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[  81/1147]                blk.4.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  82/1147]          blk.5.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  83/1147]           blk.5.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[  84/1147]               blk.5.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[  85/1147]                blk.5.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.5.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[  86/1147]                blk.5.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.5.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[  87/1147]             blk.5.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_K .. size =   224.00 MiB ->    77.00 MiB
[  88/1147]           blk.5.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  89/1147]                blk.5.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[  90/1147]                blk.5.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[  91/1147]               blk.5.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[  92/1147]            blk.5.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[  93/1147]          blk.5.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[  94/1147]          blk.5.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[  95/1147]            blk.5.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[  96/1147]               blk.5.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  97/1147]           blk.5.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[  98/1147]           blk.5.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[  99/1147]             blk.5.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[ 100/1147]                blk.5.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 101/1147]               blk.6.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 102/1147]            blk.6.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 103/1147]          blk.6.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 104/1147]          blk.6.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 105/1147]            blk.6.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 106/1147]          blk.6.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 107/1147]           blk.6.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 108/1147]               blk.6.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 109/1147]                blk.6.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.6.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 110/1147]                blk.6.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.6.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 111/1147]             blk.6.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_K .. size =   224.00 MiB ->    77.00 MiB
[ 112/1147]           blk.6.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 113/1147]                blk.6.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 114/1147]                blk.6.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 115/1147]               blk.6.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 116/1147]           blk.6.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[ 117/1147]           blk.6.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[ 118/1147]             blk.6.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[ 119/1147]                blk.6.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 120/1147]               blk.7.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 121/1147]            blk.7.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 122/1147]          blk.7.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 123/1147]          blk.7.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 124/1147]            blk.7.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 125/1147]          blk.7.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 126/1147]           blk.7.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 127/1147]               blk.7.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 128/1147]                blk.7.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.7.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 129/1147]                blk.7.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.7.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 130/1147]             blk.7.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_K .. size =   224.00 MiB ->    77.00 MiB
[ 131/1147]           blk.7.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 132/1147]                blk.7.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 133/1147]                blk.7.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 134/1147]               blk.7.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 135/1147]           blk.7.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[ 136/1147]           blk.7.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[ 137/1147]             blk.7.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[ 138/1147]                blk.7.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 139/1147]               blk.8.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 140/1147]            blk.8.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 141/1147]          blk.8.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 142/1147]          blk.8.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 143/1147]            blk.8.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 144/1147]          blk.8.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 145/1147]           blk.8.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 146/1147]               blk.8.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 147/1147]                blk.8.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.8.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 148/1147]                blk.8.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.8.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 149/1147]             blk.8.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_K .. size =   224.00 MiB ->    77.00 MiB
[ 150/1147]           blk.8.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 151/1147]                blk.8.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 152/1147]                blk.8.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 153/1147]               blk.8.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 154/1147]           blk.8.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[ 155/1147]           blk.8.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[ 156/1147]             blk.8.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[ 157/1147]                blk.8.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 158/1147]               blk.9.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 159/1147]            blk.9.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 160/1147]          blk.9.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 161/1147]          blk.9.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 162/1147]            blk.9.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 163/1147]          blk.9.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 164/1147]           blk.9.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 165/1147]               blk.9.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 166/1147]                blk.9.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.9.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 167/1147]                blk.9.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.9.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 168/1147]             blk.9.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_K .. size =   224.00 MiB ->    77.00 MiB
[ 169/1147]           blk.9.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 170/1147]                blk.9.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 171/1147]                blk.9.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 172/1147]              blk.10.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 173/1147]           blk.10.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 174/1147]         blk.10.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 175/1147]         blk.10.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 176/1147]           blk.10.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 177/1147]         blk.10.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 178/1147]          blk.10.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 179/1147]              blk.10.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 180/1147]               blk.10.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.10.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 181/1147]               blk.10.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.10.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 182/1147]            blk.10.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_K .. size =   224.00 MiB ->    77.00 MiB
[ 183/1147]          blk.10.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 184/1147]               blk.10.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 185/1147]               blk.10.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 186/1147]               blk.9.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 187/1147]           blk.9.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[ 188/1147]           blk.9.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[ 189/1147]             blk.9.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[ 190/1147]                blk.9.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 191/1147]              blk.10.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 192/1147]          blk.10.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[ 193/1147]          blk.10.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[ 194/1147]            blk.10.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[ 195/1147]               blk.10.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 196/1147]              blk.11.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 197/1147]           blk.11.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 198/1147]         blk.11.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 199/1147]         blk.11.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 200/1147]           blk.11.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[ 201/1147]         blk.11.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 202/1147]          blk.11.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[ 203/1147]              blk.11.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[ 204/1147]               blk.11.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.11.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[ 205/1147]               blk.11.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.11.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[ 206/1147]            blk.11.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_K .. size =   224.00 MiB ->    77.00 MiB
[ 207/1147]          blk.11.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 208/1147]               blk.11.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[ 209/1147]               blk.11.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[ 210/1147]              blk.11.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 211/1147]          blk.11.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[ 212/1147]          blk.11.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[ 213/1147]            blk.11.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[ 214/1147]               blk.11.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[...]
[1089/1147]              blk.58.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1090/1147]           blk.58.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1091/1147]         blk.58.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[1092/1147]         blk.58.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[1093/1147]           blk.58.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[1094/1147]         blk.58.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1095/1147]          blk.58.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[1096/1147]              blk.58.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[1097/1147]               blk.58.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.58.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[1098/1147]               blk.58.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.58.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[1099/1147]            blk.58.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_K .. size =   224.00 MiB ->    77.00 MiB
[1100/1147]          blk.58.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1101/1147]               blk.58.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[1102/1147]               blk.58.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[1103/1147]              blk.58.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1104/1147]          blk.58.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[1105/1147]          blk.58.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[1106/1147]            blk.58.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[1107/1147]               blk.58.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1108/1147]              blk.59.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1109/1147]           blk.59.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1110/1147]         blk.59.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[1111/1147]         blk.59.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[1112/1147]           blk.59.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[1113/1147]         blk.59.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1114/1147]          blk.59.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[1115/1147]              blk.59.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[1116/1147]               blk.59.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.59.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[1117/1147]               blk.59.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.59.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[1118/1147]            blk.59.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_K .. size =   224.00 MiB ->    77.00 MiB
[1119/1147]          blk.59.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1120/1147]               blk.59.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[1121/1147]               blk.59.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[1122/1147]              blk.59.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1123/1147]          blk.59.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[1124/1147]          blk.59.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[1125/1147]            blk.59.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[1126/1147]               blk.59.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1127/1147]              blk.60.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1128/1147]           blk.60.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1129/1147]         blk.60.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[1130/1147]         blk.60.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[1131/1147]           blk.60.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    28.00 MiB ->     7.88 MiB
[1132/1147]         blk.60.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1133/1147]          blk.60.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
[1134/1147]              blk.60.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
[1135/1147]               blk.60.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0

====== llama_model_quantize_internal: did not find weights for blk.60.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
[1136/1147]               blk.60.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.60.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
[1137/1147]            blk.60.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to q5_K .. size =   224.00 MiB ->    77.00 MiB
[1138/1147]          blk.60.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1139/1147]               blk.60.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    21.00 MiB ->     5.91 MiB
[1140/1147]               blk.60.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq4_k_r4 .. size =    72.00 MiB ->    20.25 MiB
[1141/1147]                        output.weight - [ 7168, 129280,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for output.weight
converting to q6_K .. size =  1767.50 MiB ->   724.95 MiB
[1142/1147]              blk.60.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1143/1147]          blk.60.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[1144/1147]          blk.60.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[1145/1147]            blk.60.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[1146/1147]               blk.60.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1147/1147]                   output_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
llama_model_quantize_internal: model size  = 1282038.27 MB
llama_model_quantize_internal: quant size  = 362010.72 MB
llama_model_quantize_internal: WARNING: 61 of 786 tensor(s) required fallback quantization

main: quantize time = 13788349.37 ms
main:    total time = 13788349.37 ms
</details>

<details>
<summary>IQ4_K_R4 V2 quantization logs</summary>

load_imatrix: imatrix dataset='imatrix-training-full-3'
load_imatrix: loaded 720 importance matrix entries from /mnt/sda/mradermacher_DeepSeek-R1-GGUF/imatrix.dat computed on 315 chunks
prepare_imatrix: have 720 importance matrix entries
main: build = 3549 (ac732053)
main: built with gcc (Clear Linux OS for Intel Architecture) 14.2.1 20241210 releases/gcc-14.2.0-551-g21a09f0507 for x86_64-generic-linux
main: quantizing '/mnt/sda/opensourcerelease_DeepSeek-R1-bf16/opensourcerelease_DeepSeek-R1-Bf16-256x21B-F16.gguf' to '/mnt/sda/opensourcerelease_DeepSeek-R1-bf16/opensourcerelease_DeepSeek-R1-Bf16-256x21B-IQ4_K_R4_ATT2.gguf' as IQ4_K_R4 using 48 threads
llama_model_loader: loaded meta data with 48 key-value pairs and 1147 tensors from /mnt/sda/opensourcerelease_DeepSeek-R1-bf16/opensourcerelease_DeepSeek-R1-Bf16-256x21B-F16.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = opensourcerelease_DeepSeek R1 Bf16
llama_model_loader: - kv   3:                         general.size_label str              = 256x21B
llama_model_loader: - kv   4:                            general.license str              = mit
llama_model_loader: - kv   5:                   general.base_model.count u32              = 1
llama_model_loader: - kv   6:                  general.base_model.0.name str              = DeepSeek R1
llama_model_loader: - kv   7:          general.base_model.0.organization str              = Deepseek Ai
llama_model_loader: - kv   8:              general.base_model.0.repo_url str              = https://huggingface.co/deepseek-ai/De...
llama_model_loader: - kv   9:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv  10:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv  11:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv  12:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  13:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv  14:          deepseek2.attention.head_count_kv u32              = 128
llama_model_loader: - kv  15:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  16: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  17:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  18:                          general.file_type u32              = 1
llama_model_loader: - kv  19:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  20:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  21:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  22:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  23:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  24:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  25:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  26:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  27:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  28:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  29:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  30:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  31:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  32:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  33:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  34: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  35: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  36:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  37:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  38:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<▒...
llama_model_loader: - kv  39:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  40:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
llama_model_loader: - kv  41:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  42:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  43:            tokenizer.ggml.padding_token_id u32              = 1
llama_model_loader: - kv  44:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  45:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  46:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  47:               general.quantization_version u32              = 2
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type  f16:  786 tensors
================================ Have weights data with 720 entries
[   1/1147]                    token_embd.weight - [ 7168, 129280,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for token_embd.weight
converting to q8_0 .. size =  1767.50 MiB ->   938.98 MiB
[   2/1147]               blk.0.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[   3/1147]                blk.0.ffn_down.weight - [18432,  7168,     1,     1], type =    f16, converting to iq6_k .. size =   252.00 MiB ->   104.34 MiB
[   4/1147]                blk.0.ffn_gate.weight - [ 7168, 18432,     1,     1], type =    f16, converting to iq6_k .. size =   252.00 MiB ->   104.34 MiB
[   5/1147]                  blk.0.ffn_up.weight - [ 7168, 18432,     1,     1], type =    f16, converting to iq6_k .. size =   252.00 MiB ->   104.34 MiB
[   6/1147]                blk.0.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[   7/1147]          blk.0.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[   8/1147]           blk.0.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq6_k .. size =     7.88 MiB ->     3.26 MiB
[   9/1147]               blk.0.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq6_k .. size =    32.00 MiB ->    13.25 MiB
[  10/1147]                blk.0.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.0.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  11/1147]                blk.0.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.0.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  12/1147]             blk.0.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq6_k .. size =   224.00 MiB ->    92.75 MiB
[  13/1147]           blk.0.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  14/1147]                blk.0.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq6_k .. size =    21.00 MiB ->     8.70 MiB
[  15/1147]                blk.0.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq6_k .. size =    72.00 MiB ->    29.81 MiB
[  16/1147]               blk.1.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  17/1147]                blk.1.ffn_down.weight - [18432,  7168,     1,     1], type =    f16, converting to iq6_k .. size =   252.00 MiB ->   104.34 MiB
[  18/1147]                blk.1.ffn_gate.weight - [ 7168, 18432,     1,     1], type =    f16, converting to iq6_k .. size =   252.00 MiB ->   104.34 MiB
[  19/1147]                  blk.1.ffn_up.weight - [ 7168, 18432,     1,     1], type =    f16, converting to iq6_k .. size =   252.00 MiB ->   104.34 MiB
[  20/1147]                blk.1.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  21/1147]          blk.1.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  22/1147]           blk.1.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq6_k .. size =     7.88 MiB ->     3.26 MiB
[  23/1147]               blk.1.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq6_k .. size =    32.00 MiB ->    13.25 MiB
[  24/1147]                blk.1.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.1.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  25/1147]                blk.1.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.1.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  26/1147]             blk.1.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq6_k .. size =   224.00 MiB ->    92.75 MiB
[  27/1147]           blk.1.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  28/1147]                blk.1.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq6_k .. size =    21.00 MiB ->     8.70 MiB
[  29/1147]                blk.1.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq6_k .. size =    72.00 MiB ->    29.81 MiB
[  30/1147]               blk.2.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  31/1147]                blk.2.ffn_down.weight - [18432,  7168,     1,     1], type =    f16, converting to iq6_k .. size =   252.00 MiB ->   104.34 MiB
[  32/1147]                blk.2.ffn_gate.weight - [ 7168, 18432,     1,     1], type =    f16, converting to iq6_k .. size =   252.00 MiB ->   104.34 MiB
[  33/1147]                  blk.2.ffn_up.weight - [ 7168, 18432,     1,     1], type =    f16, converting to iq6_k .. size =   252.00 MiB ->   104.34 MiB
[  34/1147]                blk.2.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  35/1147]          blk.2.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  36/1147]           blk.2.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq6_k .. size =     7.88 MiB ->     3.26 MiB
[  37/1147]               blk.2.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq6_k .. size =    32.00 MiB ->    13.25 MiB
[  38/1147]                blk.2.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.2.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  39/1147]                blk.2.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.2.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  40/1147]             blk.2.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq6_k .. size =   224.00 MiB ->    92.75 MiB
[  41/1147]           blk.2.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  42/1147]                blk.2.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq6_k .. size =    21.00 MiB ->     8.70 MiB
[  43/1147]                blk.2.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq6_k .. size =    72.00 MiB ->    29.81 MiB
[  44/1147]               blk.3.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[  45/1147]            blk.3.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[  46/1147]          blk.3.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq6_k .. size =    28.00 MiB ->    11.59 MiB
[  47/1147]          blk.3.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq6_k .. size =    28.00 MiB ->    11.59 MiB
[  48/1147]            blk.3.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq6_k .. size =    28.00 MiB ->    11.59 MiB
[  49/1147]          blk.3.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  50/1147]           blk.3.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq6_k .. size =     7.88 MiB ->     3.26 MiB
[  51/1147]               blk.3.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq6_k .. size =    32.00 MiB ->    13.25 MiB
[  52/1147]                blk.3.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.3.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  53/1147]                blk.3.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.3.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  54/1147]             blk.3.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq6_k .. size =   224.00 MiB ->    92.75 MiB
[  55/1147]           blk.3.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  56/1147]                blk.3.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq6_k .. size =    21.00 MiB ->     8.70 MiB
[  57/1147]                blk.3.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq6_k .. size =    72.00 MiB ->    29.81 MiB
[  58/1147]               blk.3.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  59/1147]           blk.3.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq5_k_r4 .. size =  7168.00 MiB ->  2464.00 MiB
[  60/1147]           blk.3.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[  61/1147]             blk.3.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[  62/1147]                blk.3.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  63/1147]               blk.4.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[  64/1147]            blk.4.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[  65/1147]          blk.4.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq6_k .. size =    28.00 MiB ->    11.59 MiB
[  66/1147]          blk.4.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq6_k .. size =    28.00 MiB ->    11.59 MiB
[  67/1147]            blk.4.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq6_k .. size =    28.00 MiB ->    11.59 MiB
[  68/1147]          blk.4.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  69/1147]           blk.4.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq6_k .. size =     7.88 MiB ->     3.26 MiB
[  70/1147]               blk.4.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq6_k .. size =    32.00 MiB ->    13.25 MiB
[  71/1147]                blk.4.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.4.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  72/1147]                blk.4.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.4.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  73/1147]             blk.4.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq6_k .. size =   224.00 MiB ->    92.75 MiB
[  74/1147]           blk.4.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  75/1147]                blk.4.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq6_k .. size =    21.00 MiB ->     8.70 MiB
[  76/1147]                blk.4.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq6_k .. size =    72.00 MiB ->    29.81 MiB
[  77/1147]               blk.4.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  78/1147]           blk.4.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq5_k_r4 .. size =  7168.00 MiB ->  2464.00 MiB
[  79/1147]           blk.4.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[  80/1147]             blk.4.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[  81/1147]                blk.4.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  82/1147]          blk.5.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  83/1147]           blk.5.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq6_k .. size =     7.88 MiB ->     3.26 MiB
[  84/1147]               blk.5.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq6_k .. size =    32.00 MiB ->    13.25 MiB
[  85/1147]                blk.5.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.5.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  86/1147]                blk.5.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.5.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  87/1147]             blk.5.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq6_k .. size =   224.00 MiB ->    92.75 MiB
[  88/1147]           blk.5.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  89/1147]                blk.5.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq6_k .. size =    21.00 MiB ->     8.70 MiB
[  90/1147]                blk.5.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq6_k .. size =    72.00 MiB ->    29.81 MiB
[  91/1147]               blk.5.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[  92/1147]            blk.5.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[  93/1147]          blk.5.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq6_k .. size =    28.00 MiB ->    11.59 MiB
[  94/1147]          blk.5.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq6_k .. size =    28.00 MiB ->    11.59 MiB
[  95/1147]            blk.5.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq6_k .. size =    28.00 MiB ->    11.59 MiB
[  96/1147]               blk.5.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  97/1147]           blk.5.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq5_k_r4 .. size =  7168.00 MiB ->  2464.00 MiB
[  98/1147]           blk.5.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[  99/1147]             blk.5.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[ 100/1147]                blk.5.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 101/1147]               blk.6.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 102/1147]            blk.6.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 103/1147]          blk.6.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq6_k .. size =    28.00 MiB ->    11.59 MiB
[ 104/1147]          blk.6.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq6_k .. size =    28.00 MiB ->    11.59 MiB
[ 105/1147]            blk.6.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq6_k .. size =    28.00 MiB ->    11.59 MiB
[ 106/1147]          blk.6.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 107/1147]           blk.6.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq6_k .. size =     7.88 MiB ->     3.26 MiB
[ 108/1147]               blk.6.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq6_k .. size =    32.00 MiB ->    13.25 MiB
[ 109/1147]                blk.6.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.6.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 110/1147]                blk.6.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.6.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 111/1147]             blk.6.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq6_k .. size =   224.00 MiB ->    92.75 MiB
[ 112/1147]           blk.6.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 113/1147]                blk.6.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq6_k .. size =    21.00 MiB ->     8.70 MiB
[ 114/1147]                blk.6.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq6_k .. size =    72.00 MiB ->    29.81 MiB
[ 115/1147]               blk.6.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 116/1147]           blk.6.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq5_k_r4 .. size =  7168.00 MiB ->  2464.00 MiB
[ 117/1147]           blk.6.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[ 118/1147]             blk.6.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[ 119/1147]                blk.6.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 120/1147]               blk.7.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 121/1147]            blk.7.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 122/1147]          blk.7.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq6_k .. size =    28.00 MiB ->    11.59 MiB
[ 123/1147]          blk.7.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq6_k .. size =    28.00 MiB ->    11.59 MiB
[ 124/1147]            blk.7.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq6_k .. size =    28.00 MiB ->    11.59 MiB
[ 125/1147]          blk.7.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 126/1147]           blk.7.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq6_k .. size =     7.88 MiB ->     3.26 MiB
[ 127/1147]               blk.7.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq6_k .. size =    32.00 MiB ->    13.25 MiB
[ 128/1147]                blk.7.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.7.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 129/1147]                blk.7.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.7.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 130/1147]             blk.7.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq6_k .. size =   224.00 MiB ->    92.75 MiB
[ 131/1147]           blk.7.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 132/1147]                blk.7.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq6_k .. size =    21.00 MiB ->     8.70 MiB
[ 133/1147]                blk.7.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq6_k .. size =    72.00 MiB ->    29.81 MiB
[ 134/1147]               blk.7.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 135/1147]           blk.7.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[ 136/1147]           blk.7.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[ 137/1147]             blk.7.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[ 138/1147]                blk.7.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 139/1147]               blk.8.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 140/1147]            blk.8.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 141/1147]          blk.8.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq6_k .. size =    28.00 MiB ->    11.59 MiB
[ 142/1147]          blk.8.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq6_k .. size =    28.00 MiB ->    11.59 MiB
[ 143/1147]            blk.8.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq6_k .. size =    28.00 MiB ->    11.59 MiB
[ 144/1147]          blk.8.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 145/1147]           blk.8.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq6_k .. size =     7.88 MiB ->     3.26 MiB
[ 146/1147]               blk.8.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq6_k .. size =    32.00 MiB ->    13.25 MiB
[ 147/1147]                blk.8.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.8.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 148/1147]                blk.8.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.8.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 149/1147]             blk.8.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq6_k .. size =   224.00 MiB ->    92.75 MiB
[ 150/1147]           blk.8.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 151/1147]                blk.8.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq6_k .. size =    21.00 MiB ->     8.70 MiB
[ 152/1147]                blk.8.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq6_k .. size =    72.00 MiB ->    29.81 MiB
[ 153/1147]               blk.8.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 154/1147]           blk.8.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[ 155/1147]           blk.8.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[ 156/1147]             blk.8.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[ 157/1147]                blk.8.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 158/1147]               blk.9.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 159/1147]            blk.9.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 160/1147]          blk.9.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq6_k .. size =    28.00 MiB ->    11.59 MiB
[ 161/1147]          blk.9.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq6_k .. size =    28.00 MiB ->    11.59 MiB
[ 162/1147]            blk.9.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq6_k .. size =    28.00 MiB ->    11.59 MiB
[ 163/1147]          blk.9.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 164/1147]           blk.9.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq6_k .. size =     7.88 MiB ->     3.26 MiB
[ 165/1147]               blk.9.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq6_k .. size =    32.00 MiB ->    13.25 MiB
[ 166/1147]                blk.9.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.9.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 167/1147]                blk.9.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.9.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 168/1147]             blk.9.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq6_k .. size =   224.00 MiB ->    92.75 MiB
[ 169/1147]           blk.9.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 170/1147]                blk.9.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq6_k .. size =    21.00 MiB ->     8.70 MiB
[ 171/1147]                blk.9.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq6_k .. size =    72.00 MiB ->    29.81 MiB
[ 172/1147]              blk.10.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 173/1147]           blk.10.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 174/1147]         blk.10.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq6_k .. size =    28.00 MiB ->    11.59 MiB
[ 175/1147]         blk.10.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq6_k .. size =    28.00 MiB ->    11.59 MiB
[ 176/1147]           blk.10.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq6_k .. size =    28.00 MiB ->    11.59 MiB
[ 177/1147]         blk.10.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 178/1147]          blk.10.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq6_k .. size =     7.88 MiB ->     3.26 MiB
[ 179/1147]              blk.10.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq6_k .. size =    32.00 MiB ->    13.25 MiB
[ 180/1147]               blk.10.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.10.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 181/1147]               blk.10.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.10.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 182/1147]            blk.10.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq6_k .. size =   224.00 MiB ->    92.75 MiB
[ 183/1147]          blk.10.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 184/1147]               blk.10.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq6_k .. size =    21.00 MiB ->     8.70 MiB
[ 185/1147]               blk.10.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq6_k .. size =    72.00 MiB ->    29.81 MiB
[ 186/1147]               blk.9.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 187/1147]           blk.9.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[ 188/1147]           blk.9.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[ 189/1147]             blk.9.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[ 190/1147]                blk.9.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 191/1147]              blk.10.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 192/1147]          blk.10.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[ 193/1147]          blk.10.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[ 194/1147]            blk.10.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[ 195/1147]               blk.10.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 196/1147]              blk.11.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 197/1147]           blk.11.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 198/1147]         blk.11.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq6_k .. size =    28.00 MiB ->    11.59 MiB
[ 199/1147]         blk.11.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq6_k .. size =    28.00 MiB ->    11.59 MiB
[ 200/1147]           blk.11.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq6_k .. size =    28.00 MiB ->    11.59 MiB
[ 201/1147]         blk.11.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 202/1147]          blk.11.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq6_k .. size =     7.88 MiB ->     3.26 MiB
[ 203/1147]              blk.11.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq6_k .. size =    32.00 MiB ->    13.25 MiB
[ 204/1147]               blk.11.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.11.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 205/1147]               blk.11.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.11.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 206/1147]            blk.11.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq6_k .. size =   224.00 MiB ->    92.75 MiB
[ 207/1147]          blk.11.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 208/1147]               blk.11.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq6_k .. size =    21.00 MiB ->     8.70 MiB
[ 209/1147]               blk.11.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq6_k .. size =    72.00 MiB ->    29.81 MiB
[ 210/1147]              blk.11.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 211/1147]          blk.11.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[ 212/1147]          blk.11.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[ 213/1147]            blk.11.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[ 214/1147]               blk.11.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[...]
[1089/1147]              blk.58.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1090/1147]           blk.58.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1091/1147]         blk.58.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq6_k .. size =    28.00 MiB ->    11.59 MiB
[1092/1147]         blk.58.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq6_k .. size =    28.00 MiB ->    11.59 MiB
[1093/1147]           blk.58.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq6_k .. size =    28.00 MiB ->    11.59 MiB
[1094/1147]         blk.58.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1095/1147]          blk.58.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq6_k .. size =     7.88 MiB ->     3.26 MiB
[1096/1147]              blk.58.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq6_k .. size =    32.00 MiB ->    13.25 MiB
[1097/1147]               blk.58.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.58.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1098/1147]               blk.58.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.58.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1099/1147]            blk.58.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq6_k .. size =   224.00 MiB ->    92.75 MiB
[1100/1147]          blk.58.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1101/1147]               blk.58.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq6_k .. size =    21.00 MiB ->     8.70 MiB
[1102/1147]               blk.58.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq6_k .. size =    72.00 MiB ->    29.81 MiB
[1103/1147]              blk.58.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1104/1147]          blk.58.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[1105/1147]          blk.58.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[1106/1147]            blk.58.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[1107/1147]               blk.58.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1108/1147]              blk.59.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1109/1147]           blk.59.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1110/1147]         blk.59.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq6_k .. size =    28.00 MiB ->    11.59 MiB
[1111/1147]         blk.59.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq6_k .. size =    28.00 MiB ->    11.59 MiB
[1112/1147]           blk.59.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq6_k .. size =    28.00 MiB ->    11.59 MiB
[1113/1147]         blk.59.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1114/1147]          blk.59.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq6_k .. size =     7.88 MiB ->     3.26 MiB
[1115/1147]              blk.59.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq6_k .. size =    32.00 MiB ->    13.25 MiB
[1116/1147]               blk.59.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.59.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1117/1147]               blk.59.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.59.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1118/1147]            blk.59.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq6_k .. size =   224.00 MiB ->    92.75 MiB
[1119/1147]          blk.59.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1120/1147]               blk.59.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq6_k .. size =    21.00 MiB ->     8.70 MiB
[1121/1147]               blk.59.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq6_k .. size =    72.00 MiB ->    29.81 MiB
[1122/1147]              blk.59.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1123/1147]          blk.59.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[1124/1147]          blk.59.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[1125/1147]            blk.59.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[1126/1147]               blk.59.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1127/1147]              blk.60.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1128/1147]           blk.60.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1129/1147]         blk.60.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, converting to iq6_k .. size =    28.00 MiB ->    11.59 MiB
[1130/1147]         blk.60.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq6_k .. size =    28.00 MiB ->    11.59 MiB
[1131/1147]           blk.60.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, converting to iq6_k .. size =    28.00 MiB ->    11.59 MiB
[1132/1147]         blk.60.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1133/1147]          blk.60.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to iq6_k .. size =     7.88 MiB ->     3.26 MiB
[1134/1147]              blk.60.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to iq6_k .. size =    32.00 MiB ->    13.25 MiB
[1135/1147]               blk.60.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.60.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1136/1147]               blk.60.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for blk.60.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1137/1147]            blk.60.attn_output.weight - [16384,  7168,     1,     1], type =    f16, converting to iq6_k .. size =   224.00 MiB ->    92.75 MiB
[1138/1147]          blk.60.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1139/1147]               blk.60.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to iq6_k .. size =    21.00 MiB ->     8.70 MiB
[1140/1147]               blk.60.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to iq6_k .. size =    72.00 MiB ->    29.81 MiB
[1141/1147]                        output.weight - [ 7168, 129280,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for output.weight
converting to q6_K .. size =  1767.50 MiB ->   724.95 MiB
[1142/1147]              blk.60.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1143/1147]          blk.60.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[1144/1147]          blk.60.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[1145/1147]            blk.60.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, converting to iq4_k_r4 .. size =  7168.00 MiB ->  2016.00 MiB
[1146/1147]               blk.60.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1147/1147]                   output_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
llama_model_quantize_internal: model size  = 1282038.27 MB
llama_model_quantize_internal: quant size  = 367657.12 MB

main: quantize time = 10290932.85 ms
main:    total time = 10290932.85 ms

</details>

---

👤 **jukofyork** commented the **2025-02-08** at **02:53:50**:<br>

Just saw this thread linked from the main MLA PR:

- It's some or all of the `attn_k_b.weight` tensors that can't be quantised as `float16` (it will just repeat the same word over and over in the after outputting the opening `<thinking>` tag).
- The model is also very sensitive to `ffn_down_exps.weight` bitrate (`Q3_K` or less and it starts to get *really* dumb...).

This 128 token prompt:

```
> Varis adjusted the noose, its hemp fibers grinding beneath his calluses. “Last chance,” he said, voice like gravel dragged through mud. “Confess, and your soul stays your own.”
> Jurl laughed—a wet, gurgling sound. “You’re knee-deep in it, Coldwater. ” The thing inside him twisted the boy’s lips into a grin too wide for his face. “The Great Wolf’s howlin’ again. The Dead’s Gate’s rusted through… ”

Turn this into the opening chapter of a Grimdark trilogy.
```

seems to be a good test of the model getting dumber, eg:

- The number of tokens in the thinking section starts to drop off.
- The story it generates won't actually use the quoted strings.
- The "planning" in the thinking section goes way down and just write a few vague guidelines/paragraphs.
- It will just start to make up a vaguely "dark" story without using any of what you gave it for low `ffn_down_exps.weight` bitrate.

---

👤 **saood06** commented the **2025-02-08** at **03:16:25**:<br>

@jukofyork 

I was just about to edit my comment, and mention attn_k_b.weight. 

Since you found your way here, I want to tell you with a 4.52BPW (using quant types that are better than those that exist on mainline), on a dual socket  dual socket Xeon E5-2690 v3 without any offloading I get this performance ( I use batched-bench to test PP performance as context grows, and also spot test TG performance at various context depths).

|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
|   128 |     32 |    1 |    160 |   14.776 |     8.66 |    9.929 |     3.22 |   24.704 |     6.48 |
|   256 |     32 |    1 |    288 |   28.084 |     9.12 |   10.025 |     3.19 |   38.110 |     7.56 |
|   512 |     32 |    1 |    544 |   60.362 |     8.48 |   10.199 |     3.14 |   70.561 |     7.71 |
|  1024 |     32 |    1 |   1056 |  128.774 |     7.95 |   10.440 |     3.07 |  139.215 |     7.59 |
|  2048 |     32 |    1 |   2080 |  287.581 |     7.12 |   10.958 |     2.92 |  298.538 |     6.97 |

My initial tests with offloading ( on mainline llama.cpp with the PR that lets override tensor placement to keep non-shared experts on CPU) showed worse performance the more layers I offloaded. This fork currently is missing some RPC fixes that would support this model, and also some RPC performance tweaks, but I do plan to bring those over here.

---

👤 **ikawrakow** commented the **2025-02-08** at **07:18:55**:<br>

> Off topic but when should you use Q8_K_R8 vs Q8_0_R8?

Anytime the tiny difference in accuracy does not matter to you (and a block size of 256 is possible). It is faster than `Q8_0` and also slightly smaller (8.0625 bpw vs 8.5 bpw). On an `AVX2` system the performance difference is not as large as it is on `ARM` or `AVX512` (Zen4/5 cores, recent Intel CPU's where `AVX512` has not been disabled).

Here is a PP performance comparison between `Q8_0/Q8_0_R8` and `Q8_K_R8` for 8B LLaMA on a vanilla `AVX2` system (Ryzen-5975WX), this should be representative for your dual Xeon E5-2690 system:

| model                          |       size | threads | fa | rtr |          test |              t/s |
| ------------------------------ | ---------: | ------: | -: | --: | ------------: | ---------------: |
| llama 8B Q8_0                  |   7.95 GiB |      32 |  1 |   0 |         pp512 |    193.45 ± 0.32 |
| llama 8B Q8_0                  |   7.95 GiB |      32 |  1 |   1 |         pp512 |    254.21 ± 0.30 |
| llama 8B Q8_K_R8               |   7.56 GiB |      32 |  1 |   1 |         pp512 |    285.09 ± 0.35 |

And here the same comparison on Zen4 (Ryzen-7950X)

| model                          |       size | threads | fa | rtr |          test |              t/s |
| ------------------------------ | ---------: | ------: | -: | --: | ------------: | ---------------: |
| llama 8B Q8_0                  |   7.95 GiB |      16 |  1 |   0 |         pp512 |    165.26 ± 3.16 |
| llama 8B Q8_0                  |   7.95 GiB |      16 |  1 |   1 |         pp512 |    304.90 ± 0.12 |
| llama 8B Q8_K_R8               |   7.56 GiB |      16 |  1 |   1 |         pp512 |    387.23 ± 1.10 |

To put things in perspective, the best mainline `llama.cpp` can do on the Ryzen-7950X is 165 t/s for `Q4_0` (fastest quant in `llama.cpp`). On my M2-Max `Q8_K_R8` gets 172 t/s vs 125 t/s for `Q4_0`.

On the Ryzen-7950X memory bandwidth is fully saturated with just 2 threads with `Q8_K_R8` for TG. Which means that I can let the LLM run and generate tokens while I'm doing something else without the system feeling totally bogged down.

---

👤 **ikawrakow** commented the **2025-02-08** at **07:36:52**:<br>

Concerning `fp16` vs `bf16` for `attn_k_b`: In mainline `llama.cpp` when a model tensor is `fp16`, activations get converted from `fp32` (the result of the previous operation) to `fp16` before performing the matrix multiplication with the `fp16` model tensor. If the observation is that the model becomes "dumb" when `attn_k_b` is `fp16`, the conclusion is that there are activations that are outside of the `fp16` range, and they get truncated in the conversion. This is not the case in this repository, at least not on `x86_64`. I have matrix multiplication kernels for any `fpX x fpY` combination, so for model tensors in `fp16` the matrix multiplication is done directly on the `fp32` activations. Hence, there shouldn't be any accuracy loss (unless the model contains weights outside of the `fp16` range). On `ARM`, I still convert the activations to `fp16` as `fp16 x fp16` matrix multiplications are almost 2X faster on my M2-Max. 

If there are indeed activations that fall outside the `fp16` range, then `attn_k_b` as `Q8_0` might indeed work better. In this case activations get quantized to `Q8_0`. There may be some precision loss in that process, but there is no truncation, so I expect the outcome to be indeed better in mainline `llama.cpp`.