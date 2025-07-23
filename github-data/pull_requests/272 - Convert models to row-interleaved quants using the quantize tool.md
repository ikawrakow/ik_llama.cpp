### 🔀 [#272](https://github.com/ikawrakow/ik_llama.cpp/pull/272) - Convert models to row-interleaved quants using the quantize tool

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-03-20 |
| **Updated** | 2025-03-21 |

---

#### Description

The main purpose of this PR is to remove the need for run-time-repacking (command line argument `-rtr`) by having a tool to convert models to row-interleaved quantization types. The main motivation for providing this tool is to allow using `mmap` when loading a model and still having row-interleaved quants, so that one can combine the claimed performance gains from using 1 GiB huge pages (see #267) with the performance gains due to row-interleaved quants.

**Note:** this is only useful for **CPU-only** inference. The converted (repacked) model **will not work on a GPU** (or rather it will work but will be slow as all matrix multiplications with the repacked tensors will be done on the CPU).

To use it, simply
```
./bin/llama-quantize --repack some_model repacked_model some_quant
```
The `some_quant` argument is not actually used, but I didn't want to make modifications to the `llama-quantize` command line argument parsing, so the argument must be provided, but it is ignored.

Oh, `bf16` and `f16` models can be repacked too, one gets a `GGML_TYPE_BF16_R16` model as a result. On CPU's with native `bf16` support, `GGML_TYPE_BF16_R16 ` is about 15% faster than `GGML_TYPE_BF16`, and nearly 2X faster than `GGML_TYPE_F16` (for prompt processing, TG is memory bound, so not much difference there).

**Caveat:** Some of the quantization types had a relatively minor, platform-specific, optimization applied when run-time-repacking. But as there is no way to tell if the repacking was done online, or if we are dealing with an offline-repacked model, I had to remove this optimization. This affects `Q8_0_R8, Q8_K_R8, Q8_KV_R8` on Zen4 (127 was added to these quants during run-time-repacking to avoid doing this during inference), and `Q4_0_R8` on ARM (a mask of `0x88` was applied to the packed bits, which converts the otherwise unsigned `Q4_0` values to signed values multiplied with 16).  

Closes #228

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-03-20** at **14:53:05**:<br>

Does the last commit fix it? Strange that we can no longer compare `std::string` to a C-string, and a reference to `std::string` is no longer automatically instantiated from a C-string. Seriously? This will brake billions of LoC of C++.

---

👤 **ubergarm** commented the **2025-03-20** at **14:55:53**:<br>

Seems to be compiling now on `d27b7226`. I'll go back and check if simply adding `#include string` to `./ggml/src/iqk/iqk_quantize.cpp` would also fix it to confirm.

---

👤 **ubergarm** commented the **2025-03-20** at **14:58:43**:<br>

Yeah, just needs the include e.g.

```
$ git rev-parse --short HEAD
9fbe5bee
$ git diff
diff --git a/ggml/src/iqk/iqk_quantize.cpp b/ggml/src/iqk/iqk_quantize.cpp
index bc6f34eb..0375b878 100644
--- a/ggml/src/iqk/iqk_quantize.cpp
+++ b/ggml/src/iqk/iqk_quantize.cpp
@@ -21,6 +21,7 @@
 #include <array>
 #include <algorithm>
 #include <cstring>
+#include <string>
 #include <mutex>
 #include <thread>
 #include <atomic>

## builds good
```

---

👤 **ikawrakow** commented the **2025-03-20** at **15:36:25**:<br>

I think we can leave the two unnecessary changes. If we remove the explicit string construction, the compiler does it for us anyway.

---

👤 **ubergarm** commented the **2025-03-20** at **15:38:00**:<br>

Okay, repacking seems to be working. I'll try out the freshly generated repacked weights next.

<details>
<summary>Detailed Command Output Logs</summary>

```bash
$ git rev-parse --short HEAD
9fe6fc37

$ ./build/bin/llama-quantize \
    --repack /mnt/ai/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00001-of-00009.gguf \
    /mnt/ai/models/unsloth/repack/DeepSeek-R1-Q4_K_R4.gguf \
    Q4_K_R4 # <--- *NOTE*: this is unused, but must be any valid option

main: invalid ftype '/mnt/ai/models/unsloth/repack/DeepSeek-R1-Q4_K_R4.gguf'
main: build = 3604 (9fe6fc37)
main: built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu
main: quantizing '/mnt/ai/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00001-of-00009.gguf' to '/mnt/ai/models/unsloth/repack/DeepSeek-R1-Q4_K_R4.gguf' as Q4_K_R4
llama_model_loader: additional 8 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 48 key-value pairs and 1025 tensors from /mnt/ai/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00001-of-00009.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = DeepSeek R1 BF16
llama_model_loader: - kv   3:                       general.quantized_by str              = Unsloth
llama_model_loader: - kv   4:                         general.size_label str              = 256x20B
llama_model_loader: - kv   5:                           general.repo_url str              = https://huggingface.co/unsloth
llama_model_loader: - kv   6:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv   7:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv   8:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv   9:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  10:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv  11:          deepseek2.attention.head_count_kv u32              = 128
llama_model_loader: - kv  12:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  13: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  14:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  15:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  16:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  17:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  18:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  19:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  20:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  21:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  22:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  23:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  24:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  25:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  26:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  27:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  28:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  29:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  30: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  31: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  32:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  33:                         tokenizer.ggml.pre str              = deepseek-v3
.
.
.
[   1/1025]                        output.weight - [ 7168, 129280,     1,     1], type =   q6_K, size =  724.951 MB, type = q6_k_r4
[   2/1025]                   output_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[   3/1025]                    token_embd.weight - [ 7168, 129280,     1,     1], type =   q4_K, size =  497.109 MB, type = q4_K
[   4/1025]           blk.0.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[   5/1025]          blk.0.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[   6/1025]               blk.0.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[   7/1025]               blk.0.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[   8/1025]             blk.0.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[   9/1025]                blk.0.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[  10/1025]           blk.0.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[  11/1025]                blk.0.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[  12/1025]                blk.0.ffn_down.weight - [18432,  7168,     1,     1], type =   q6_K, size =  103.359 MB, type = q6_k_r4
[  13/1025]                blk.0.ffn_gate.weight - [ 7168, 18432,     1,     1], type =   q4_K, size =   70.875 MB, type = q4_k_r4
[  14/1025]                blk.0.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[  15/1025]                  blk.0.ffn_up.weight - [ 7168, 18432,     1,     1], type =   q4_K, size =   70.875 MB, type = q4_k_r4
[  16/1025]           blk.1.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[  17/1025]          blk.1.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[  18/1025]               blk.1.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[  19/1025]               blk.1.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[  20/1025]             blk.1.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[  21/1025]                blk.1.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[  22/1025]           blk.1.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[  23/1025]                blk.1.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[  24/1025]                blk.1.ffn_down.weight - [18432,  7168,     1,     1], type =   q6_K, size =  103.359 MB, type = q6_k_r4
[  25/1025]                blk.1.ffn_gate.weight - [ 7168, 18432,     1,     1], type =   q4_K, size =   70.875 MB, type = q4_k_r4
[  26/1025]                blk.1.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[  27/1025]                  blk.1.ffn_up.weight - [ 7168, 18432,     1,     1], type =   q4_K, size =   70.875 MB, type = q4_k_r4
[  28/1025]           blk.2.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[  29/1025]          blk.2.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[  30/1025]               blk.2.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[  31/1025]               blk.2.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[  32/1025]             blk.2.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[  33/1025]                blk.2.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[  34/1025]           blk.2.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[  35/1025]                blk.2.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[  36/1025]                blk.2.ffn_down.weight - [18432,  7168,     1,     1], type =   q6_K, size =  103.359 MB, type = q6_k_r4
[  37/1025]                blk.2.ffn_gate.weight - [ 7168, 18432,     1,     1], type =   q4_K, size =   70.875 MB, type = q4_k_r4
[  38/1025]                blk.2.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[  39/1025]                  blk.2.ffn_up.weight - [ 7168, 18432,     1,     1], type =   q4_K, size =   70.875 MB, type = q4_k_r4
[  40/1025]           blk.3.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[  41/1025]          blk.3.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[  42/1025]               blk.3.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[  43/1025]               blk.3.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[  44/1025]             blk.3.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[  45/1025]                blk.3.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[  46/1025]           blk.3.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[  47/1025]                blk.3.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[  48/1025]               blk.3.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[  49/1025]           blk.3.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q6_K, size = 2940.000 MB, type = q6_k_r4
[  50/1025]          blk.3.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q6_K, size =   11.484 MB, type = q6_k_r4
[  51/1025]           blk.3.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[  52/1025]            blk.3.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[  53/1025]          blk.3.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[  54/1025]                blk.3.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[  55/1025]             blk.3.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[  56/1025]            blk.3.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[  57/1025]           blk.4.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[  58/1025]          blk.4.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[  59/1025]               blk.4.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[  60/1025]               blk.4.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[  61/1025]             blk.4.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[  62/1025]                blk.4.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[  63/1025]           blk.4.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[  64/1025]                blk.4.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[  65/1025]               blk.4.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[  66/1025]           blk.4.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q6_K, size = 2940.000 MB, type = q6_k_r4
[  67/1025]          blk.4.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q6_K, size =   11.484 MB, type = q6_k_r4
[  68/1025]           blk.4.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[  69/1025]            blk.4.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[  70/1025]          blk.4.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[  71/1025]                blk.4.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[  72/1025]             blk.4.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[  73/1025]            blk.4.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[  74/1025]           blk.5.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[  75/1025]          blk.5.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[  76/1025]               blk.5.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[  77/1025]               blk.5.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[  78/1025]             blk.5.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[  79/1025]                blk.5.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[  80/1025]           blk.5.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[  81/1025]                blk.5.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[  82/1025]               blk.5.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[  83/1025]           blk.5.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q6_K, size = 2940.000 MB, type = q6_k_r4
[  84/1025]          blk.5.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q6_K, size =   11.484 MB, type = q6_k_r4
[  85/1025]           blk.5.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[  86/1025]            blk.5.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[  87/1025]          blk.5.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[  88/1025]                blk.5.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[  89/1025]             blk.5.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[  90/1025]            blk.5.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[  91/1025]           blk.6.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[  92/1025]          blk.6.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[  93/1025]               blk.6.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[  94/1025]               blk.6.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[  95/1025]             blk.6.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[  96/1025]                blk.6.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[  97/1025]           blk.6.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[  98/1025]                blk.6.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[  99/1025]               blk.6.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 100/1025]           blk.6.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q6_K, size = 2940.000 MB, type = q6_k_r4
[ 101/1025]          blk.6.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q6_K, size =   11.484 MB, type = q6_k_r4
[ 102/1025]           blk.6.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 103/1025]            blk.6.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 104/1025]          blk.6.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 105/1025]                blk.6.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 106/1025]             blk.6.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 107/1025]            blk.6.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 108/1025]           blk.7.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 109/1025]          blk.7.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 110/1025]               blk.7.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 111/1025]               blk.7.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 112/1025]             blk.7.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 113/1025]                blk.7.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 114/1025]           blk.7.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 115/1025]                blk.7.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 116/1025]               blk.7.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 117/1025]           blk.7.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 118/1025]          blk.7.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 119/1025]           blk.7.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 120/1025]            blk.7.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 121/1025]          blk.7.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 122/1025]                blk.7.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 123/1025]             blk.7.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 124/1025]            blk.7.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 125/1025]           blk.8.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 126/1025]          blk.8.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 127/1025]               blk.8.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 128/1025]               blk.8.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 129/1025]             blk.8.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 130/1025]                blk.8.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 131/1025]           blk.8.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 132/1025]                blk.8.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 133/1025]               blk.8.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 134/1025]           blk.8.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 135/1025]          blk.8.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 136/1025]           blk.8.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 137/1025]            blk.8.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 138/1025]          blk.8.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 139/1025]                blk.8.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 140/1025]             blk.8.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 141/1025]            blk.8.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 142/1025]           blk.9.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 143/1025]          blk.9.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 144/1025]               blk.9.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 145/1025]               blk.9.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 146/1025]             blk.9.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 147/1025]                blk.9.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 148/1025]           blk.9.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 149/1025]                blk.9.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 150/1025]               blk.9.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 151/1025]           blk.9.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q6_K, size = 2940.000 MB, type = q6_k_r4
[ 152/1025]          blk.9.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q6_K, size =   11.484 MB, type = q6_k_r4
[ 153/1025]           blk.9.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 154/1025]            blk.9.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 155/1025]          blk.9.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 156/1025]                blk.9.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 157/1025]             blk.9.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 158/1025]            blk.9.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 159/1025]          blk.10.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 160/1025]         blk.10.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 161/1025]              blk.10.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 162/1025]              blk.10.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 163/1025]            blk.10.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 164/1025]               blk.10.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 165/1025]          blk.10.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 166/1025]               blk.10.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 167/1025]              blk.10.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 168/1025]          blk.10.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 169/1025]         blk.10.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 170/1025]          blk.10.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 171/1025]           blk.10.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 172/1025]         blk.10.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 173/1025]               blk.10.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 174/1025]            blk.10.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 175/1025]           blk.10.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 176/1025]          blk.11.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 177/1025]         blk.11.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 178/1025]              blk.11.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 179/1025]              blk.11.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 180/1025]            blk.11.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 181/1025]               blk.11.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 182/1025]          blk.11.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 183/1025]               blk.11.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 184/1025]              blk.11.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 185/1025]          blk.11.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 186/1025]         blk.11.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 187/1025]          blk.11.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 188/1025]           blk.11.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 189/1025]         blk.11.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 190/1025]               blk.11.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 191/1025]            blk.11.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 192/1025]           blk.11.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 193/1025]          blk.12.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 194/1025]         blk.12.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 195/1025]              blk.12.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 196/1025]              blk.12.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 197/1025]            blk.12.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 198/1025]               blk.12.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 199/1025]          blk.12.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 200/1025]               blk.12.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 201/1025]              blk.12.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 202/1025]          blk.12.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q6_K, size = 2940.000 MB, type = q6_k_r4
[ 203/1025]         blk.12.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q6_K, size =   11.484 MB, type = q6_k_r4
[ 204/1025]          blk.12.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 205/1025]           blk.12.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 206/1025]         blk.12.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 207/1025]               blk.12.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 208/1025]            blk.12.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 209/1025]           blk.12.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 210/1025]          blk.13.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 211/1025]         blk.13.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 212/1025]              blk.13.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 213/1025]              blk.13.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 214/1025]            blk.13.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 215/1025]               blk.13.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 216/1025]          blk.13.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 217/1025]               blk.13.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 218/1025]              blk.13.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 219/1025]          blk.13.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 220/1025]         blk.13.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 221/1025]          blk.13.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 222/1025]           blk.13.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 223/1025]         blk.13.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 224/1025]               blk.13.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 225/1025]            blk.13.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 226/1025]           blk.13.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 227/1025]          blk.14.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 228/1025]         blk.14.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 229/1025]              blk.14.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 230/1025]              blk.14.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 231/1025]            blk.14.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 232/1025]               blk.14.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 233/1025]          blk.14.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 234/1025]               blk.14.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 235/1025]              blk.14.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 236/1025]          blk.14.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 237/1025]         blk.14.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 238/1025]          blk.14.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 239/1025]           blk.14.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 240/1025]         blk.14.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 241/1025]               blk.14.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 242/1025]            blk.14.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 243/1025]           blk.14.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 244/1025]          blk.15.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 245/1025]         blk.15.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 246/1025]              blk.15.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 247/1025]              blk.15.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 248/1025]            blk.15.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 249/1025]               blk.15.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 250/1025]          blk.15.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 251/1025]               blk.15.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 252/1025]              blk.15.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 253/1025]          blk.15.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q6_K, size = 2940.000 MB, type = q6_k_r4
[ 254/1025]         blk.15.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q6_K, size =   11.484 MB, type = q6_k_r4
[ 255/1025]          blk.15.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 256/1025]           blk.15.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 257/1025]         blk.15.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 258/1025]               blk.15.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 259/1025]            blk.15.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 260/1025]           blk.15.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 261/1025]          blk.16.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 262/1025]         blk.16.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 263/1025]              blk.16.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 264/1025]              blk.16.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 265/1025]            blk.16.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 266/1025]               blk.16.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 267/1025]          blk.16.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 268/1025]               blk.16.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 269/1025]              blk.16.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 270/1025]          blk.16.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 271/1025]         blk.16.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 272/1025]          blk.16.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 273/1025]           blk.16.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 274/1025]         blk.16.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 275/1025]               blk.16.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 276/1025]            blk.16.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 277/1025]           blk.16.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 278/1025]          blk.17.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 279/1025]         blk.17.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 280/1025]              blk.17.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 281/1025]              blk.17.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 282/1025]            blk.17.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 283/1025]               blk.17.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 284/1025]          blk.17.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 285/1025]               blk.17.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 286/1025]              blk.17.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 287/1025]          blk.17.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 288/1025]         blk.17.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 289/1025]          blk.17.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 290/1025]           blk.17.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 291/1025]         blk.17.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 292/1025]               blk.17.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 293/1025]            blk.17.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 294/1025]           blk.17.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 295/1025]          blk.18.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 296/1025]         blk.18.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 297/1025]              blk.18.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 298/1025]              blk.18.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 299/1025]            blk.18.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 300/1025]               blk.18.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 301/1025]          blk.18.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 302/1025]               blk.18.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 303/1025]              blk.18.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 304/1025]          blk.18.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q6_K, size = 2940.000 MB, type = q6_k_r4
[ 305/1025]         blk.18.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q6_K, size =   11.484 MB, type = q6_k_r4
[ 306/1025]          blk.18.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 307/1025]           blk.18.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 308/1025]         blk.18.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 309/1025]               blk.18.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 310/1025]            blk.18.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 311/1025]           blk.18.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 312/1025]          blk.19.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 313/1025]         blk.19.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 314/1025]              blk.19.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 315/1025]              blk.19.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 316/1025]            blk.19.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 317/1025]               blk.19.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 318/1025]          blk.19.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 319/1025]               blk.19.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 320/1025]              blk.19.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 321/1025]          blk.19.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 322/1025]         blk.19.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 323/1025]          blk.19.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 324/1025]           blk.19.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 325/1025]         blk.19.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 326/1025]               blk.19.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 327/1025]            blk.19.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 328/1025]           blk.19.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 329/1025]          blk.20.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 330/1025]         blk.20.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 331/1025]              blk.20.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 332/1025]              blk.20.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 333/1025]            blk.20.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 334/1025]               blk.20.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 335/1025]          blk.20.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 336/1025]               blk.20.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 337/1025]              blk.20.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 338/1025]          blk.20.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 339/1025]         blk.20.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 340/1025]          blk.20.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 341/1025]           blk.20.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 342/1025]         blk.20.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 343/1025]               blk.20.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 344/1025]            blk.20.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 345/1025]           blk.20.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 346/1025]          blk.21.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 347/1025]         blk.21.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 348/1025]              blk.21.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 349/1025]              blk.21.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 350/1025]            blk.21.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 351/1025]               blk.21.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 352/1025]          blk.21.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 353/1025]               blk.21.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 354/1025]              blk.21.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 355/1025]          blk.21.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q6_K, size = 2940.000 MB, type = q6_k_r4
[ 356/1025]         blk.21.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q6_K, size =   11.484 MB, type = q6_k_r4
[ 357/1025]          blk.21.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 358/1025]           blk.21.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 359/1025]         blk.21.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 360/1025]               blk.21.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 361/1025]            blk.21.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 362/1025]           blk.21.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 363/1025]          blk.22.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 364/1025]         blk.22.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 365/1025]              blk.22.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 366/1025]              blk.22.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 367/1025]            blk.22.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 368/1025]               blk.22.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 369/1025]          blk.22.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 370/1025]               blk.22.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 371/1025]              blk.22.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 372/1025]          blk.22.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 373/1025]         blk.22.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 374/1025]          blk.22.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 375/1025]           blk.22.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 376/1025]         blk.22.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 377/1025]               blk.22.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 378/1025]            blk.22.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 379/1025]           blk.22.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 380/1025]          blk.23.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 381/1025]         blk.23.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 382/1025]              blk.23.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 383/1025]              blk.23.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 384/1025]            blk.23.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 385/1025]               blk.23.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 386/1025]          blk.23.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 387/1025]               blk.23.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 388/1025]              blk.23.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 389/1025]          blk.23.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 390/1025]         blk.23.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 391/1025]          blk.23.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 392/1025]           blk.23.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 393/1025]         blk.23.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 394/1025]               blk.23.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 395/1025]            blk.23.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 396/1025]           blk.23.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 397/1025]          blk.24.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 398/1025]         blk.24.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 399/1025]              blk.24.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 400/1025]              blk.24.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 401/1025]            blk.24.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 402/1025]               blk.24.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 403/1025]          blk.24.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 404/1025]               blk.24.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 405/1025]              blk.24.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 406/1025]          blk.24.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q6_K, size = 2940.000 MB, type = q6_k_r4
[ 407/1025]         blk.24.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q6_K, size =   11.484 MB, type = q6_k_r4
[ 408/1025]          blk.24.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 409/1025]           blk.24.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 410/1025]         blk.24.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 411/1025]               blk.24.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 412/1025]            blk.24.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 413/1025]           blk.24.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 414/1025]          blk.25.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 415/1025]         blk.25.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 416/1025]              blk.25.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 417/1025]              blk.25.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 418/1025]            blk.25.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 419/1025]               blk.25.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 420/1025]          blk.25.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 421/1025]               blk.25.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 422/1025]              blk.25.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 423/1025]          blk.25.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 424/1025]         blk.25.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 425/1025]          blk.25.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 426/1025]           blk.25.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 427/1025]         blk.25.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 428/1025]               blk.25.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 429/1025]            blk.25.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 430/1025]           blk.25.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 431/1025]          blk.26.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 432/1025]         blk.26.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 433/1025]              blk.26.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 434/1025]              blk.26.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 435/1025]            blk.26.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 436/1025]               blk.26.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 437/1025]          blk.26.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 438/1025]               blk.26.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 439/1025]              blk.26.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 440/1025]          blk.26.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 441/1025]         blk.26.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 442/1025]          blk.26.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 443/1025]           blk.26.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 444/1025]         blk.26.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 445/1025]               blk.26.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 446/1025]            blk.26.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 447/1025]           blk.26.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 448/1025]          blk.27.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 449/1025]         blk.27.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 450/1025]              blk.27.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 451/1025]              blk.27.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 452/1025]            blk.27.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 453/1025]               blk.27.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 454/1025]          blk.27.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 455/1025]               blk.27.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 456/1025]              blk.27.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 457/1025]          blk.27.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q6_K, size = 2940.000 MB, type = q6_k_r4
[ 458/1025]         blk.27.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q6_K, size =   11.484 MB, type = q6_k_r4
[ 459/1025]          blk.27.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 460/1025]           blk.27.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 461/1025]         blk.27.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 462/1025]               blk.27.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 463/1025]            blk.27.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 464/1025]           blk.27.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 465/1025]          blk.28.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 466/1025]         blk.28.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 467/1025]              blk.28.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 468/1025]              blk.28.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 469/1025]            blk.28.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 470/1025]               blk.28.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 471/1025]          blk.28.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 472/1025]               blk.28.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 473/1025]              blk.28.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 474/1025]          blk.28.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 475/1025]         blk.28.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 476/1025]          blk.28.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 477/1025]           blk.28.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 478/1025]         blk.28.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 479/1025]               blk.28.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 480/1025]            blk.28.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 481/1025]           blk.28.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 482/1025]          blk.29.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 483/1025]         blk.29.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 484/1025]              blk.29.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 485/1025]              blk.29.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 486/1025]            blk.29.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 487/1025]               blk.29.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 488/1025]          blk.29.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 489/1025]               blk.29.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 490/1025]              blk.29.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 491/1025]          blk.29.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 492/1025]         blk.29.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 493/1025]          blk.29.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 494/1025]           blk.29.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 495/1025]         blk.29.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 496/1025]               blk.29.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 497/1025]            blk.29.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 498/1025]           blk.29.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 499/1025]          blk.30.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 500/1025]         blk.30.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 501/1025]              blk.30.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 502/1025]              blk.30.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 503/1025]            blk.30.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 504/1025]               blk.30.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 505/1025]          blk.30.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 506/1025]               blk.30.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 507/1025]              blk.30.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 508/1025]          blk.30.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q6_K, size = 2940.000 MB, type = q6_k_r4
[ 509/1025]         blk.30.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q6_K, size =   11.484 MB, type = q6_k_r4
[ 510/1025]          blk.30.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 511/1025]           blk.30.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 512/1025]         blk.30.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 513/1025]               blk.30.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 514/1025]            blk.30.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 515/1025]           blk.30.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 516/1025]          blk.31.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 517/1025]         blk.31.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 518/1025]              blk.31.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 519/1025]              blk.31.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 520/1025]            blk.31.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 521/1025]               blk.31.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 522/1025]          blk.31.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 523/1025]               blk.31.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 524/1025]              blk.31.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 525/1025]          blk.31.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 526/1025]         blk.31.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 527/1025]          blk.31.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 528/1025]           blk.31.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 529/1025]         blk.31.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 530/1025]               blk.31.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 531/1025]            blk.31.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 532/1025]           blk.31.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 533/1025]          blk.32.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 534/1025]         blk.32.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 535/1025]              blk.32.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 536/1025]              blk.32.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 537/1025]            blk.32.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 538/1025]               blk.32.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 539/1025]          blk.32.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 540/1025]               blk.32.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 541/1025]              blk.32.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 542/1025]          blk.32.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 543/1025]         blk.32.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 544/1025]          blk.32.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 545/1025]           blk.32.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 546/1025]         blk.32.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 547/1025]               blk.32.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 548/1025]            blk.32.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 549/1025]           blk.32.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 550/1025]          blk.33.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 551/1025]         blk.33.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 552/1025]              blk.33.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 553/1025]              blk.33.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 554/1025]            blk.33.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 555/1025]               blk.33.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 556/1025]          blk.33.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 557/1025]               blk.33.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 558/1025]              blk.33.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 559/1025]          blk.33.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q6_K, size = 2940.000 MB, type = q6_k_r4
[ 560/1025]         blk.33.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q6_K, size =   11.484 MB, type = q6_k_r4
[ 561/1025]          blk.33.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 562/1025]           blk.33.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 563/1025]         blk.33.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 564/1025]               blk.33.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 565/1025]            blk.33.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 566/1025]           blk.33.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 567/1025]          blk.34.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 568/1025]         blk.34.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 569/1025]              blk.34.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 570/1025]              blk.34.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 571/1025]            blk.34.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 572/1025]               blk.34.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 573/1025]          blk.34.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 574/1025]               blk.34.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 575/1025]              blk.34.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 576/1025]          blk.34.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 577/1025]         blk.34.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 578/1025]          blk.34.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 579/1025]           blk.34.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 580/1025]         blk.34.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 581/1025]               blk.34.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 582/1025]            blk.34.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 583/1025]           blk.34.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 584/1025]          blk.35.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 585/1025]         blk.35.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 586/1025]              blk.35.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 587/1025]              blk.35.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 588/1025]            blk.35.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 589/1025]               blk.35.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 590/1025]          blk.35.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 591/1025]               blk.35.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 592/1025]              blk.35.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 593/1025]          blk.35.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 594/1025]         blk.35.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 595/1025]          blk.35.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 596/1025]           blk.35.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 597/1025]         blk.35.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 598/1025]               blk.35.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 599/1025]            blk.35.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 600/1025]           blk.35.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 601/1025]          blk.36.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 602/1025]         blk.36.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 603/1025]              blk.36.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 604/1025]              blk.36.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 605/1025]            blk.36.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 606/1025]               blk.36.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 607/1025]          blk.36.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 608/1025]               blk.36.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 609/1025]              blk.36.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 610/1025]          blk.36.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q6_K, size = 2940.000 MB, type = q6_k_r4
[ 611/1025]         blk.36.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q6_K, size =   11.484 MB, type = q6_k_r4
[ 612/1025]          blk.36.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 613/1025]           blk.36.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 614/1025]         blk.36.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 615/1025]               blk.36.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 616/1025]            blk.36.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 617/1025]           blk.36.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 618/1025]          blk.37.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 619/1025]         blk.37.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 620/1025]              blk.37.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 621/1025]              blk.37.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 622/1025]            blk.37.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 623/1025]               blk.37.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 624/1025]          blk.37.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 625/1025]               blk.37.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 626/1025]              blk.37.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 627/1025]          blk.37.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 628/1025]         blk.37.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 629/1025]          blk.37.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 630/1025]           blk.37.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 631/1025]         blk.37.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 632/1025]               blk.37.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 633/1025]            blk.37.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 634/1025]           blk.37.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 635/1025]          blk.38.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 636/1025]         blk.38.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 637/1025]              blk.38.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 638/1025]              blk.38.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 639/1025]            blk.38.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 640/1025]               blk.38.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 641/1025]          blk.38.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 642/1025]               blk.38.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 643/1025]              blk.38.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 644/1025]          blk.38.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 645/1025]         blk.38.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 646/1025]          blk.38.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 647/1025]           blk.38.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 648/1025]         blk.38.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 649/1025]               blk.38.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 650/1025]            blk.38.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 651/1025]           blk.38.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 652/1025]          blk.39.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 653/1025]         blk.39.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 654/1025]              blk.39.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 655/1025]              blk.39.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 656/1025]            blk.39.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 657/1025]               blk.39.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 658/1025]          blk.39.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 659/1025]               blk.39.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 660/1025]              blk.39.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 661/1025]          blk.39.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q6_K, size = 2940.000 MB, type = q6_k_r4
[ 662/1025]         blk.39.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q6_K, size =   11.484 MB, type = q6_k_r4
[ 663/1025]          blk.39.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 664/1025]           blk.39.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 665/1025]         blk.39.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 666/1025]               blk.39.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 667/1025]            blk.39.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 668/1025]           blk.39.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 669/1025]          blk.40.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 670/1025]         blk.40.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 671/1025]              blk.40.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 672/1025]              blk.40.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 673/1025]            blk.40.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 674/1025]               blk.40.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 675/1025]          blk.40.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 676/1025]               blk.40.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 677/1025]              blk.40.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 678/1025]          blk.40.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 679/1025]         blk.40.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 680/1025]          blk.40.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 681/1025]           blk.40.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 682/1025]         blk.40.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 683/1025]               blk.40.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 684/1025]            blk.40.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 685/1025]           blk.40.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 686/1025]          blk.41.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 687/1025]         blk.41.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 688/1025]              blk.41.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 689/1025]              blk.41.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 690/1025]            blk.41.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 691/1025]               blk.41.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 692/1025]          blk.41.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 693/1025]               blk.41.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 694/1025]              blk.41.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 695/1025]          blk.41.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 696/1025]         blk.41.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 697/1025]          blk.41.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 698/1025]           blk.41.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 699/1025]         blk.41.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 700/1025]               blk.41.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 701/1025]            blk.41.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 702/1025]           blk.41.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 703/1025]          blk.42.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 704/1025]         blk.42.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 705/1025]              blk.42.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 706/1025]              blk.42.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 707/1025]            blk.42.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 708/1025]               blk.42.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 709/1025]          blk.42.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 710/1025]               blk.42.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 711/1025]              blk.42.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 712/1025]          blk.42.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q6_K, size = 2940.000 MB, type = q6_k_r4
[ 713/1025]         blk.42.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q6_K, size =   11.484 MB, type = q6_k_r4
[ 714/1025]          blk.42.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 715/1025]           blk.42.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 716/1025]         blk.42.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 717/1025]               blk.42.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 718/1025]            blk.42.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 719/1025]           blk.42.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 720/1025]          blk.43.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 721/1025]         blk.43.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 722/1025]              blk.43.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 723/1025]              blk.43.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 724/1025]            blk.43.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 725/1025]               blk.43.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 726/1025]          blk.43.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 727/1025]               blk.43.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 728/1025]              blk.43.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 729/1025]          blk.43.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 730/1025]         blk.43.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 731/1025]          blk.43.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 732/1025]           blk.43.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 733/1025]         blk.43.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 734/1025]               blk.43.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 735/1025]            blk.43.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 736/1025]           blk.43.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 737/1025]          blk.44.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 738/1025]         blk.44.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 739/1025]              blk.44.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 740/1025]              blk.44.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 741/1025]            blk.44.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 742/1025]               blk.44.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 743/1025]          blk.44.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 744/1025]               blk.44.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 745/1025]              blk.44.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 746/1025]          blk.44.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 747/1025]         blk.44.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 748/1025]          blk.44.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 749/1025]           blk.44.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 750/1025]         blk.44.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 751/1025]               blk.44.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 752/1025]            blk.44.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 753/1025]           blk.44.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 754/1025]          blk.45.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 755/1025]         blk.45.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 756/1025]              blk.45.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 757/1025]              blk.45.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 758/1025]            blk.45.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 759/1025]               blk.45.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 760/1025]          blk.45.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 761/1025]               blk.45.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 762/1025]              blk.45.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 763/1025]          blk.45.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q6_K, size = 2940.000 MB, type = q6_k_r4
[ 764/1025]         blk.45.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q6_K, size =   11.484 MB, type = q6_k_r4
[ 765/1025]          blk.45.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 766/1025]           blk.45.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 767/1025]         blk.45.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 768/1025]               blk.45.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 769/1025]            blk.45.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 770/1025]           blk.45.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 771/1025]          blk.46.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 772/1025]         blk.46.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 773/1025]              blk.46.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 774/1025]              blk.46.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 775/1025]            blk.46.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 776/1025]               blk.46.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 777/1025]          blk.46.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 778/1025]               blk.46.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 779/1025]              blk.46.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 780/1025]          blk.46.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 781/1025]         blk.46.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 782/1025]          blk.46.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 783/1025]           blk.46.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 784/1025]         blk.46.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 785/1025]               blk.46.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 786/1025]            blk.46.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 787/1025]           blk.46.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 788/1025]          blk.47.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 789/1025]         blk.47.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 790/1025]              blk.47.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 791/1025]              blk.47.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 792/1025]            blk.47.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 793/1025]               blk.47.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 794/1025]          blk.47.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 795/1025]               blk.47.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 796/1025]              blk.47.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 797/1025]          blk.47.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 798/1025]         blk.47.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 799/1025]          blk.47.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 800/1025]           blk.47.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 801/1025]         blk.47.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 802/1025]               blk.47.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 803/1025]            blk.47.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 804/1025]           blk.47.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 805/1025]          blk.48.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 806/1025]         blk.48.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 807/1025]              blk.48.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 808/1025]              blk.48.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 809/1025]            blk.48.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 810/1025]               blk.48.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 811/1025]          blk.48.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 812/1025]               blk.48.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 813/1025]              blk.48.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 814/1025]          blk.48.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q6_K, size = 2940.000 MB, type = q6_k_r4
[ 815/1025]         blk.48.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q6_K, size =   11.484 MB, type = q6_k_r4
[ 816/1025]          blk.48.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 817/1025]           blk.48.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 818/1025]         blk.48.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 819/1025]               blk.48.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 820/1025]            blk.48.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 821/1025]           blk.48.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 822/1025]          blk.49.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 823/1025]         blk.49.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 824/1025]              blk.49.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 825/1025]              blk.49.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 826/1025]            blk.49.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 827/1025]               blk.49.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 828/1025]          blk.49.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 829/1025]               blk.49.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 830/1025]              blk.49.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 831/1025]          blk.49.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 832/1025]         blk.49.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 833/1025]          blk.49.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 834/1025]           blk.49.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 835/1025]         blk.49.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 836/1025]               blk.49.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 837/1025]            blk.49.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 838/1025]           blk.49.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 839/1025]          blk.50.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 840/1025]         blk.50.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 841/1025]              blk.50.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 842/1025]              blk.50.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 843/1025]            blk.50.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 844/1025]               blk.50.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 845/1025]          blk.50.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 846/1025]               blk.50.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 847/1025]              blk.50.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 848/1025]          blk.50.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 849/1025]         blk.50.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 850/1025]          blk.50.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 851/1025]           blk.50.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 852/1025]         blk.50.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 853/1025]               blk.50.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 854/1025]            blk.50.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 855/1025]           blk.50.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 856/1025]          blk.51.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 857/1025]         blk.51.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 858/1025]              blk.51.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 859/1025]              blk.51.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 860/1025]            blk.51.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 861/1025]               blk.51.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 862/1025]          blk.51.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 863/1025]               blk.51.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 864/1025]              blk.51.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 865/1025]          blk.51.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q6_K, size = 2940.000 MB, type = q6_k_r4
[ 866/1025]         blk.51.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q6_K, size =   11.484 MB, type = q6_k_r4
[ 867/1025]          blk.51.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 868/1025]           blk.51.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 869/1025]         blk.51.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 870/1025]               blk.51.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 871/1025]            blk.51.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 872/1025]           blk.51.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 873/1025]          blk.52.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 874/1025]         blk.52.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 875/1025]              blk.52.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 876/1025]              blk.52.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 877/1025]            blk.52.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 878/1025]               blk.52.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 879/1025]          blk.52.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 880/1025]               blk.52.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 881/1025]              blk.52.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 882/1025]          blk.52.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 883/1025]         blk.52.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 884/1025]          blk.52.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 885/1025]           blk.52.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 886/1025]         blk.52.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 887/1025]               blk.52.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 888/1025]            blk.52.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 889/1025]           blk.52.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 890/1025]          blk.53.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 891/1025]         blk.53.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 892/1025]              blk.53.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 893/1025]              blk.53.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 894/1025]            blk.53.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 895/1025]               blk.53.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 896/1025]          blk.53.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 897/1025]               blk.53.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 898/1025]              blk.53.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 899/1025]          blk.53.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q6_K, size = 2940.000 MB, type = q6_k_r4
[ 900/1025]         blk.53.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q6_K, size =   11.484 MB, type = q6_k_r4
[ 901/1025]          blk.53.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 902/1025]           blk.53.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 903/1025]         blk.53.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 904/1025]               blk.53.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 905/1025]            blk.53.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 906/1025]           blk.53.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 907/1025]          blk.54.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 908/1025]         blk.54.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 909/1025]              blk.54.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 910/1025]              blk.54.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 911/1025]            blk.54.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 912/1025]               blk.54.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 913/1025]          blk.54.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 914/1025]               blk.54.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 915/1025]              blk.54.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 916/1025]          blk.54.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q6_K, size = 2940.000 MB, type = q6_k_r4
[ 917/1025]         blk.54.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q6_K, size =   11.484 MB, type = q6_k_r4
[ 918/1025]          blk.54.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 919/1025]           blk.54.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 920/1025]         blk.54.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 921/1025]               blk.54.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 922/1025]            blk.54.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 923/1025]           blk.54.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 924/1025]          blk.55.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 925/1025]         blk.55.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 926/1025]              blk.55.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 927/1025]              blk.55.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 928/1025]            blk.55.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 929/1025]               blk.55.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 930/1025]          blk.55.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 931/1025]               blk.55.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 932/1025]              blk.55.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 933/1025]          blk.55.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q6_K, size = 2940.000 MB, type = q6_k_r4
[ 934/1025]         blk.55.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q6_K, size =   11.484 MB, type = q6_k_r4
[ 935/1025]          blk.55.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 936/1025]           blk.55.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 937/1025]         blk.55.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 938/1025]               blk.55.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 939/1025]            blk.55.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 940/1025]           blk.55.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 941/1025]          blk.56.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 942/1025]         blk.56.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 943/1025]              blk.56.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 944/1025]              blk.56.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 945/1025]            blk.56.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 946/1025]               blk.56.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 947/1025]          blk.56.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 948/1025]               blk.56.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 949/1025]              blk.56.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 950/1025]          blk.56.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q6_K, size = 2940.000 MB, type = q6_k_r4
[ 951/1025]         blk.56.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q6_K, size =   11.484 MB, type = q6_k_r4
[ 952/1025]          blk.56.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 953/1025]           blk.56.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 954/1025]         blk.56.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 955/1025]               blk.56.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 956/1025]            blk.56.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 957/1025]           blk.56.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 958/1025]          blk.57.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 959/1025]         blk.57.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 960/1025]              blk.57.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 961/1025]              blk.57.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 962/1025]            blk.57.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 963/1025]               blk.57.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 964/1025]          blk.57.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 965/1025]               blk.57.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 966/1025]              blk.57.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 967/1025]          blk.57.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q6_K, size = 2940.000 MB, type = q6_k_r4
[ 968/1025]         blk.57.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q6_K, size =   11.484 MB, type = q6_k_r4
[ 969/1025]          blk.57.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 970/1025]           blk.57.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 971/1025]         blk.57.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 972/1025]               blk.57.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 973/1025]            blk.57.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 974/1025]           blk.57.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 975/1025]          blk.58.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 976/1025]         blk.58.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 977/1025]              blk.58.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 978/1025]              blk.58.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 979/1025]            blk.58.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 980/1025]               blk.58.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 981/1025]          blk.58.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 982/1025]               blk.58.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[ 983/1025]              blk.58.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[ 984/1025]          blk.58.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q6_K, size = 2940.000 MB, type = q6_k_r4
[ 985/1025]         blk.58.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q6_K, size =   11.484 MB, type = q6_k_r4
[ 986/1025]          blk.58.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 987/1025]           blk.58.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[ 988/1025]         blk.58.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 989/1025]               blk.58.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 990/1025]            blk.58.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[ 991/1025]           blk.58.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[ 992/1025]          blk.59.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[ 993/1025]         blk.59.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[ 994/1025]              blk.59.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[ 995/1025]              blk.59.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[ 996/1025]            blk.59.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[ 997/1025]               blk.59.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[ 998/1025]          blk.59.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[ 999/1025]               blk.59.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[1000/1025]              blk.59.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[1001/1025]          blk.59.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q6_K, size = 2940.000 MB, type = q6_k_r4
[1002/1025]         blk.59.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q6_K, size =   11.484 MB, type = q6_k_r4
[1003/1025]          blk.59.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[1004/1025]           blk.59.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[1005/1025]         blk.59.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[1006/1025]               blk.59.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[1007/1025]            blk.59.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[1008/1025]           blk.59.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[1009/1025]          blk.60.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q4_K, size =    2.215 MB, type = q4_k_r4
[1010/1025]         blk.60.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB, type = f32
[1011/1025]              blk.60.attn_kv_b.weight - [  512, 32768,     1,     1], type =   q4_K, size =    9.000 MB, type = q4_k_r4
[1012/1025]              blk.60.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[1013/1025]            blk.60.attn_output.weight - [16384,  7168,     1,     1], type =   q4_K, size =   63.000 MB, type = q4_k_r4
[1014/1025]               blk.60.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q4_K, size =    5.906 MB, type = q4_k_r4
[1015/1025]          blk.60.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB, type = f32
[1016/1025]               blk.60.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   q4_K, size =   20.250 MB, type = q4_k_r4
[1017/1025]              blk.60.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB, type = f32
[1018/1025]          blk.60.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   q6_K, size = 2940.000 MB, type = q6_k_r4
[1019/1025]         blk.60.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q6_K, size =   11.484 MB, type = q6_k_r4
[1020/1025]          blk.60.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[1021/1025]           blk.60.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB, type = f32
[1022/1025]         blk.60.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
[1023/1025]               blk.60.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB, type = f32
[1024/1025]            blk.60.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   q4_K, size = 2016.000 MB, type = q4_k_r4
[1025/1025]           blk.60.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q4_K, size =    7.875 MB, type = q4_k_r4
llama_model_quantize_internal: model size  = 385689.62 MB
llama_model_quantize_internal: quant size  = 385689.62 MB
===================== Model ftype: Q4_K - Medium: Repacked ftype: Q4_K_R4

main: quantize time = 724052.06 ms
main:    total time = 724052.06 ms

$ du -c /mnt/ai/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-Q4_K_M/*.gguf
47206828        /mnt/ai/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00001-of-00009.gguf
48270904        /mnt/ai/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00002-of-00009.gguf
48366528        /mnt/ai/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00003-of-00009.gguf
47141132        /mnt/ai/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00004-of-00009.gguf
48263708        /mnt/ai/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00005-of-00009.gguf
47141132        /mnt/ai/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00006-of-00009.gguf
48270904        /mnt/ai/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00007-of-00009.gguf
45838656        /mnt/ai/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00008-of-00009.gguf
14451648        /mnt/ai/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00009-of-00009.gguf
394951440       total

$ ls -la /mnt/ai/models/unsloth/repack/DeepSeek-R1-Q4_K_R4.gguf
-rw-rw-r-- 1 j j 404430186592 Mar 20 15:32 /mnt/ai/models/unsloth/repack/DeepSeek-R1-Q4_K_R4.gguf
```

<details>