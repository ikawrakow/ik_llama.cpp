## 🔀 [Pull Request #624](https://github.com/ikawrakow/ik_llama.cpp/pull/624) - Quantization tweaks

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Source Branch** | `ik/quantization_tweaks` |
| **Target Branch** | `main` |
| **Created** | 2025-07-17 |
| **Updated** | 2025-07-19 |

---

## 📄 Description

Minor tweaks in the quantization methods for `Q2_K, Q3_K, Q4_K, Q5_K, IQ2_KS, IQ3_KS, IQ3_K`.

Also changed the automatic recipes to use `IQ2_KL` instead of `Q2_K`.

---

## 💬 Conversation

👤 **Nexesenex** commented on **2025-07-17** at **16:18:26**

Hey IK.
You devised small gains on perplexity for all those ggml_types, I presume, besides the works on the ftypes/quant strategies?

---

👤 **ikawrakow** commented on **2025-07-17** at **16:32:39**

> You devised small gains on perplexity for all those ggml_types, I presume, besides the works on the ftypes/quant strategies?

Yes. But it is basically the same trick.

Most of the heavy-duty lifting during quantization is in determining the block scales. The block scales are floats and then get rounded to an integer in a way depending on how many bits we are spending for block scales. Typically this is just round-to-nearest from a super-block or tensor row scale. While working on `IQ2_KL` I decided to see what happens if I also check the nearest integer values for a block scale, and pick the integer value that minimizes RMSE (changing the block scales can change the quant values, which can sometimes result in a lower difference to the original model weights). This did give a small but non-negligible improvement for `IQ2_KL`. So, today I decided to see if the same trick can be applied  to other quantization types, and the PR includes changes to those types where it helped.

But as perplexity does not tell us anything, I did not post any PPL changes.

Just kidding. I felt lazy to do the usual evaluation with multiple models, so that's why I'm not posting PPL results. I expect people to try and will tell me if it became better.  But it is not a major improvement, just a relatively minor tweak.

---

👤 **ubergarm** commented on **2025-07-18** at **03:11:51**

I had just finished cooking a Kimi-K2-Instruct-IQ3_KS when I noticed this PR!

So I had to cook again to compare. I also went back and re-cooked my IQ2_KS recipe. Not perfect recipes to test this thoroughly given mixed types, but a couple data points at least coming together now.

The IQ2_KS looks slightly better, but the IQ3_KS seemed worse for this PR. Haven't tried others or any other tests.

Full recipe is [available on the hf model card secret recipe details](https://huggingface.co/ubergarm/Kimi-K2-Instruct-GGUF#-iq3_ks-427205-gib-3573-bpw)
* The IQ3_KS uses IQ3_KS for ffn_(gate|up)_exps and IQ4_KS for ffn_down_exps
* The IQ2_KS uses IQ2_KS for ffn_(gate|up)_exps and IQ2_KL for ffn_down_exps

Also getting around to checking some perplexities of various UD quants which are also on the chart for any that I've measured myself.

*EDIT* to avoid duplicating this graph everywhere [I'll keep it and data here for now.](https://github.com/ikawrakow/ik_llama.cpp/pull/616#issuecomment-3087170346).

I'll update this chart once I run one more and drop it with the data over on the IQ1_KT PR were we are discussing that more specifically.

---

👤 **ikawrakow** commented on **2025-07-18** at **05:05:47**

@ubergarm

Thank you for this plot. So, the pure `IQ1_KT` model is basically on par with Unsloth's `IQ1_S`, while being 22% smaller! 

Isn't the bpw for "badname-UD-TQ1_0" wrong? This model shows as just 245 GB on HF (or is HF also wrong about model sizes now?). 

I see `UD-IQ1_S` labeled as "nofmoe". Does this mean that `-fmoe` is not working? I saw elsewhere a report about models failing with `-fmoe`, but no-one would bother to post the model quant composition so I can try to understand what is wrong. If `UD-IQ1_S` is failing with `-fmoe`, can you open an issue for that? Thanks.

---

👤 **ikawrakow** commented on **2025-07-18** at **06:58:19**

> The IQ2_KS looks slightly better, but the IQ3_KS seemed worse for this PR. Haven't tried others or any other tests.

This is strange. Because of the worse result for `IQ3_KS` for Kimi-2, I now ran perplexity calculations for my usual set of 5 models: LlaMA-1-7B, LlaMA-2-7B, Mistral-7B<sup>1</sup>, LlaMA-3.1-Instruct-8B, DeepSeek-Lite, and also added Qwen3-22B-A3B. Here are the PPL results for Wikitext2 for 2 different context lengths using (almost) pure `IQ3_KS` quantization (only `attn_v` is `IQ4_KS`, token embeddings and output are left at `Q8_0` to not have irrelevant effects from these two tensors)

| Model | Context | PPL (main) | PPL (PR) |
| ---: | ---: | ---: | ---: |
| LlaMA-1-7B |    512 | 6.1930 | 6.1807 |
|                      | 2048 | 5.3355 | 5.3211 |
| LlaMA-2-7B |    512 | 6.1114 | 6.1001 |
|                       | 2048 | 5.3355 | 5.3211 |
| Mistral-7B.   |   512 | 5.9519 | 5.9330 |
|                       | 2048 | 5.0769 | 5.0603 |
| LlaMA-3-8B |  512 | 8.1346 | 8.1198 |
|                       | 2048 | 7.0888 | 7.0715 |
| DeepSeek    |  512 | 7.0893 | 7.0834 |
|                      | 2048 | 6.2253 | 6.2164 |
| Qwen3         | 512 | 9.5122 | 9.4694 |
|                     | 2048 | 8.1964 | 8.1604 |

We see a small but consistent improvement for all 12 cases.

How was the imatrix for Kimi-2 generated? 

___
<sup>1</sup> Why use such ancient models? The LLaMA-v1 models were the basis for k-quants development. i-quants were developed using LLaMA-v1, LLaMA-v2 and Mistral-7B. In my experience, if a quantization technique does well on all 3 of these, it is (almost) guaranteed to do well on any other model out there.

---

👤 **ubergarm** commented on **2025-07-18** at **14:09:23**

> Thank you for this plot. So, the pure IQ1_KT model is basically on par with Unsloth's IQ1_S, while being 22% smaller!

Yes the KT quants are looking strong in the low bpw ranges here!

> Isn't the bpw for "badname-UD-TQ1_0" wrong? This model shows as just 245 GB on HF (or is HF also wrong about model sizes now?).

Here is what it prints out when I start it up:
```bash
model=/mnt/data/models/unsloth/Kimi-K2-Instruct-GGUF/UD-TQ1_0/Kimi-K2-Instruct-UD-TQ1_0-00001-of-00005.gguf

llm_load_print_meta: model ftype      = IQ1_S - 1.5625 bpw
llm_load_print_meta: model params     = 1.026 T
llm_load_print_meta: model size       = 227.854 GiB (1.907 BPW)
llm_load_print_meta: repeating layers = 226.342 GiB (1.899 BPW, 1024.059 B parameters)
llm_load_print_meta: general.name     = Kimi-K2-Instruct
```

So yes, I see I made a copy paste error and kept the size/bpw from the UD-IQ1_S, I've updated my data file now!

> I see UD-IQ1_S labeled as "nofmoe". Does this mean that -fmoe is not working? I saw elsewhere a report about models failing with -fmoe, but no-one would bother to post the model quant composition so I can try to understand what is wrong. If UD-IQ1_S is failing with -fmoe, can you open an issue for that? Thanks.

Correct, that UD-IQ1_S is the only one failing with `-fmoe`. You can look inside it here and change the filename in the URL to see the other GGUF splits contents: https://huggingface.co/unsloth/Kimi-K2-Instruct-GGUF?show_file_info=UD-IQ1_S%2FKimi-K2-Instruct-UD-IQ1_S-00001-of-00006.gguf

I'm just catching up on the action and will open an issue and link to everything (if there is not already an issue open now).

> How was the imatrix for Kimi-2 generated?

Yeah I was surprised at the increase in PPL on my PR624-IQ3_KS as well and want to double check for any operator (me) error. Here is the imatrix command I ran and full logs. I've used the resulting imatrix.dat for all of my quants:

<details>

<summary>👈 kimi-k2-instruct imatrix command and logs</summary>

On earlier deepseek models I left out `-mla 1` but added it for this given recent discussions on attn_kv_b and such.

```bash
model=/mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/Kimi-K2-Instruct-Q8_0.gguf

numactl --interleave=all \
./build/bin/llama-imatrix \
    -m "$model" \
    -f ubergarm-imatrix-calibration-corpus-v02.txt \
    -o /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat \
    -mla 1 \
    --verbosity 1 \
    --ctx-size 512 \
    --layer-similarity \
    --numa distribute \
    --threads 384

llama_model_loader: loaded meta data with 42 key-value pairs and 1157 tensors from /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/Kimi-K2-Instruct-Q8_0.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Kimi K2 Instruct Bf16 Safetensors
llama_model_loader: - kv   3:                           general.finetune str              = Instruct-safetensors
llama_model_loader: - kv   4:                           general.basename str              = Kimi-K2
llama_model_loader: - kv   5:                         general.size_label str              = 384x15B
llama_model_loader: - kv   6:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv   7:                   deepseek2.context_length u32              = 131072
llama_model_loader: - kv   8:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv   9:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  10:             deepseek2.attention.head_count u32              = 64
llama_model_loader: - kv  11:          deepseek2.attention.head_count_kv u32              = 64
llama_model_loader: - kv  12:                   deepseek2.rope.freq_base f32              = 50000.000000
llama_model_loader: - kv  13: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  14:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  15:                          general.file_type u32              = 7
llama_model_loader: - kv  16:        deepseek2.leading_dense_block_count u32              = 1
llama_model_loader: - kv  17:                       deepseek2.vocab_size u32              = 163840
llama_model_loader: - kv  18:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  19:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  20:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  21:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  22:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  23:                     deepseek2.expert_count u32              = 384
llama_model_loader: - kv  24:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  25:             deepseek2.expert_weights_scale f32              = 2.827000
llama_model_loader: - kv  26:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  27:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  28:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  29:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  30:              deepseek2.rope.scaling.factor f32              = 32.000000
llama_model_loader: - kv  31: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  32: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  33:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  34:                         tokenizer.ggml.pre str              = kimi-k2
llama_model_loader: - kv  35:                      tokenizer.ggml.tokens arr[str,163840]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  36:                  tokenizer.ggml.token_type arr[i32,163840]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  37:                      tokenizer.ggml.merges arr[str,163328]  = ["Ġ Ġ", "ĠĠ ĠĠ", "Ġ t", "i n",...
llama_model_loader: - kv  38:                tokenizer.ggml.bos_token_id u32              = 163584
llama_model_loader: - kv  39:                tokenizer.ggml.eos_token_id u32              = 163585
llama_model_loader: - kv  40:                    tokenizer.chat_template str              = {% if tools -%}\n    {{ '<|im_system|>...
llama_model_loader: - kv  41:               general.quantization_version u32              = 2
llama_model_loader: - type  f32:  365 tensors
llama_model_loader: - type q8_0:  792 tensors
llm_load_vocab: special tokens cache size = 256
llm_load_vocab: token to piece cache size = 1.0607 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = deepseek2
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 163840
llm_load_print_meta: n_merges         = 163328
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 131072
llm_load_print_meta: n_embd           = 7168
llm_load_print_meta: n_layer          = 61
llm_load_print_meta: n_head           = 64
llm_load_print_meta: n_head_kv        = 64
llm_load_print_meta: n_rot            = 64
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 192
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: n_embd_k_gqa     = 12288
llm_load_print_meta: n_embd_v_gqa     = 8192
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 18432
llm_load_print_meta: n_expert         = 384
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = yarn
llm_load_print_meta: freq_base_train  = 50000.0
llm_load_print_meta: freq_scale_train = 0.03125
llm_load_print_meta: n_ctx_orig_yarn  = 4096
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = 671B
llm_load_print_meta: model ftype      = Q8_0
llm_load_print_meta: model params     = 1.027 T
llm_load_print_meta: model size       = 1016.623 GiB (8.504 BPW) 
llm_load_print_meta: repeating layers = 1014.299 GiB (8.504 BPW, 1024.571 B parameters)
llm_load_print_meta: general.name     = Kimi K2 Instruct Bf16 Safetensors
llm_load_print_meta: BOS token        = 163584 '[BOS]'
llm_load_print_meta: EOS token        = 163585 '[EOS]'
llm_load_print_meta: LF token         = 128 'Ä'
llm_load_print_meta: EOT token        = 163586 '<|im_end|>'
llm_load_print_meta: max token length = 512
llm_load_print_meta: n_layer_dense_lead   = 1
llm_load_print_meta: n_lora_q             = 1536
llm_load_print_meta: n_lora_kv            = 512
llm_load_print_meta: n_ff_exp             = 2048
llm_load_print_meta: n_expert_shared      = 1
llm_load_print_meta: expert_weights_scale = 2.8
llm_load_print_meta: expert_weights_norm  = 1
llm_load_print_meta: expert_gating_func   = sigmoid
llm_load_print_meta: rope_yarn_log_mul    = 0.1000
llm_load_tensors: ggml ctx size =    0.47 MiB
llm_load_tensors:        CPU buffer size = 1041021.91 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 512
llama_new_context_with_model: n_batch    = 512
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: mla_attn   = 1
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 0
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 50000.0
llama_new_context_with_model: freq_scale = 0.03125
llama_kv_cache_init:        CPU KV buffer size =    64.81 MiB
llama_new_context_with_model: KV self size  =   64.81 MiB, c^KV (f16):   34.31 MiB, kv^T (f16):   30.50 MiB
llama_new_context_with_model:        CPU  output buffer size =     0.63 MiB
llama_new_context_with_model:        CPU compute buffer size =   334.00 MiB
llama_new_context_with_model: graph nodes  = 3827
llama_new_context_with_model: graph splits = 1

system_info: n_threads = 384 / 768 | AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | 
compute_imatrix: tokenizing the input ..
compute_imatrix: tokenization took 835.506 ms
compute_imatrix: computing over 826 chunks with batch_size 512
compute_imatrix: 43.88 seconds per pass - ETA 10 hours 4.05 minutes
[1]75.3007,[2]13.9305,[3]6.7296,[4]4.1851,[5]3.2372,[6]2.6987,[7]2.3609,[8]2.1425,[9]2.0965,
save_imatrix: entry '             blk.59.ffn_down_exps.weight' has partial data (99.74%) 1 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.59.ffn_up_exps.weight' has partial data (99.74%) 1 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.59.ffn_gate_exps.weight' has partial data (99.74%) 1 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.58.ffn_down_exps.weight' has partial data (99.74%) 1 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.56.ffn_down_exps.weight' has partial data (99.22%) 3 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.56.ffn_gate_exps.weight' has partial data (99.22%) 3 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.54.ffn_down_exps.weight' has partial data (99.74%) 1 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.54.ffn_up_exps.weight' has partial data (99.74%) 1 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.52.ffn_down_exps.weight' has partial data (98.96%) 4 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.52.ffn_up_exps.weight' has partial data (98.96%) 4 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.52.ffn_gate_exps.weight' has partial data (98.96%) 4 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.51.ffn_down_exps.weight' has partial data (99.48%) 2 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.51.ffn_gate_exps.weight' has partial data (99.48%) 2 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.57.ffn_gate_exps.weight' has partial data (99.74%) 1 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.49.ffn_gate_exps.weight' has partial data (99.74%) 1 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.54.ffn_gate_exps.weight' has partial data (99.74%) 1 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.48.ffn_up_exps.weight' has partial data (99.74%) 1 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.47.ffn_down_exps.weight' has partial data (99.48%) 2 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.47.ffn_up_exps.weight' has partial data (99.48%) 2 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.46.ffn_down_exps.weight' has partial data (99.22%) 3 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.46.ffn_up_exps.weight' has partial data (99.22%) 3 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.46.ffn_gate_exps.weight' has partial data (99.22%) 3 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.45.ffn_down_exps.weight' has partial data (99.22%) 3 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.22.ffn_up_exps.weight' has partial data (99.48%) 2 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.33.ffn_down_exps.weight' has partial data (99.22%) 3 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.19.ffn_down_exps.weight' has partial data (99.74%) 1 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.57.ffn_down_exps.weight' has partial data (99.74%) 1 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.45.ffn_gate_exps.weight' has partial data (99.22%) 3 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.21.ffn_gate_exps.weight' has partial data (98.96%) 4 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.19.ffn_up_exps.weight' has partial data (99.74%) 1 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.18.ffn_down_exps.weight' has partial data (99.22%) 3 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.17.ffn_gate_exps.weight' has partial data (98.96%) 4 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.20.ffn_up_exps.weight' has partial data (98.70%) 5 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.48.ffn_down_exps.weight' has partial data (99.74%) 1 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '              blk.8.ffn_down_exps.weight' has partial data (98.18%) 7 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.14.ffn_down_exps.weight' has partial data (99.48%) 2 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '              blk.1.ffn_gate_exps.weight' has partial data (98.44%) 6 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '              blk.1.ffn_down_exps.weight' has partial data (98.44%) 6 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.47.ffn_gate_exps.weight' has partial data (99.48%) 2 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.11.ffn_gate_exps.weight' has partial data (99.48%) 2 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.11.ffn_down_exps.weight' has partial data (99.48%) 2 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.21.ffn_down_exps.weight' has partial data (98.96%) 4 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.34.ffn_down_exps.weight' has partial data (99.22%) 3 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '              blk.4.ffn_down_exps.weight' has partial data (97.40%) 10 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.51.ffn_up_exps.weight' has partial data (99.48%) 2 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.20.ffn_gate_exps.weight' has partial data (98.70%) 5 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.36.ffn_down_exps.weight' has partial data (99.48%) 2 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.16.ffn_gate_exps.weight' has partial data (99.48%) 2 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.13.ffn_down_exps.weight' has partial data (99.22%) 3 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '                blk.5.ffn_up_exps.weight' has partial data (97.66%) 9 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.38.ffn_up_exps.weight' has partial data (99.22%) 3 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '              blk.3.ffn_gate_exps.weight' has partial data (98.18%) 7 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '              blk.4.ffn_gate_exps.weight' has partial data (97.40%) 10 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.44.ffn_up_exps.weight' has partial data (99.48%) 2 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.58.ffn_gate_exps.weight' has partial data (99.74%) 1 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.48.ffn_gate_exps.weight' has partial data (99.74%) 1 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '              blk.6.ffn_gate_exps.weight' has partial data (99.22%) 3 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.34.ffn_gate_exps.weight' has partial data (99.22%) 3 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '              blk.5.ffn_down_exps.weight' has partial data (97.66%) 9 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.30.ffn_up_exps.weight' has partial data (99.48%) 2 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '                blk.7.ffn_up_exps.weight' has partial data (98.96%) 4 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.44.ffn_down_exps.weight' has partial data (99.48%) 2 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.17.ffn_down_exps.weight' has partial data (98.96%) 4 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.25.ffn_down_exps.weight' has partial data (99.74%) 1 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.25.ffn_up_exps.weight' has partial data (99.74%) 1 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.58.ffn_up_exps.weight' has partial data (99.74%) 1 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.57.ffn_up_exps.weight' has partial data (99.74%) 1 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.13.ffn_up_exps.weight' has partial data (99.22%) 3 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.39.ffn_gate_exps.weight' has partial data (98.70%) 5 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '                blk.1.ffn_up_exps.weight' has partial data (98.44%) 6 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.18.ffn_gate_exps.weight' has partial data (99.22%) 3 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '              blk.2.ffn_gate_exps.weight' has partial data (97.92%) 8 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.25.ffn_gate_exps.weight' has partial data (99.74%) 1 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.41.ffn_up_exps.weight' has partial data (99.74%) 1 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '              blk.3.ffn_down_exps.weight' has partial data (98.18%) 7 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.45.ffn_up_exps.weight' has partial data (99.22%) 3 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.40.ffn_up_exps.weight' has partial data (99.48%) 2 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.24.ffn_gate_exps.weight' has partial data (99.48%) 2 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.14.ffn_up_exps.weight' has partial data (99.48%) 2 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.31.ffn_gate_exps.weight' has partial data (99.22%) 3 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.21.ffn_up_exps.weight' has partial data (98.96%) 4 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.36.ffn_gate_exps.weight' has partial data (99.48%) 2 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.41.ffn_gate_exps.weight' has partial data (99.74%) 1 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.37.ffn_up_exps.weight' has partial data (98.70%) 5 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.37.ffn_down_exps.weight' has partial data (98.70%) 5 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.11.ffn_up_exps.weight' has partial data (99.48%) 2 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '                blk.6.ffn_up_exps.weight' has partial data (99.22%) 3 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.34.ffn_up_exps.weight' has partial data (99.22%) 3 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '              blk.6.ffn_down_exps.weight' has partial data (99.22%) 3 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '              blk.2.ffn_down_exps.weight' has partial data (97.92%) 8 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.17.ffn_up_exps.weight' has partial data (98.96%) 4 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.49.ffn_down_exps.weight' has partial data (99.74%) 1 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.33.ffn_up_exps.weight' has partial data (99.22%) 3 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '                blk.3.ffn_up_exps.weight' has partial data (98.18%) 7 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '              blk.7.ffn_down_exps.weight' has partial data (98.96%) 4 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.44.ffn_gate_exps.weight' has partial data (99.48%) 2 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.49.ffn_up_exps.weight' has partial data (99.74%) 1 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.22.ffn_gate_exps.weight' has partial data (99.48%) 2 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.22.ffn_down_exps.weight' has partial data (99.48%) 2 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.14.ffn_gate_exps.weight' has partial data (99.48%) 2 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.56.ffn_up_exps.weight' has partial data (99.22%) 3 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.23.ffn_gate_exps.weight' has partial data (98.44%) 6 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.23.ffn_up_exps.weight' has partial data (98.44%) 6 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.23.ffn_down_exps.weight' has partial data (98.44%) 6 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.16.ffn_down_exps.weight' has partial data (99.48%) 2 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.24.ffn_up_exps.weight' has partial data (99.48%) 2 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.24.ffn_down_exps.weight' has partial data (99.48%) 2 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.18.ffn_up_exps.weight' has partial data (99.22%) 3 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.26.ffn_gate_exps.weight' has partial data (97.66%) 9 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.26.ffn_up_exps.weight' has partial data (97.66%) 9 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.26.ffn_down_exps.weight' has partial data (97.66%) 9 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.27.ffn_down_exps.weight' has partial data (99.22%) 3 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '                blk.4.ffn_up_exps.weight' has partial data (97.40%) 10 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.27.ffn_gate_exps.weight' has partial data (99.22%) 3 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.27.ffn_up_exps.weight' has partial data (99.22%) 3 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.38.ffn_down_exps.weight' has partial data (99.22%) 3 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.31.ffn_up_exps.weight' has partial data (99.22%) 3 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.28.ffn_gate_exps.weight' has partial data (99.48%) 2 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.28.ffn_up_exps.weight' has partial data (99.48%) 2 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.28.ffn_down_exps.weight' has partial data (99.48%) 2 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.42.ffn_up_exps.weight' has partial data (99.48%) 2 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.29.ffn_gate_exps.weight' has partial data (98.96%) 4 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.29.ffn_up_exps.weight' has partial data (98.96%) 4 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.29.ffn_down_exps.weight' has partial data (98.96%) 4 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.30.ffn_gate_exps.weight' has partial data (99.48%) 2 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.30.ffn_down_exps.weight' has partial data (99.48%) 2 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '                blk.8.ffn_up_exps.weight' has partial data (98.18%) 7 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.43.ffn_gate_exps.weight' has partial data (99.48%) 2 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.36.ffn_up_exps.weight' has partial data (99.48%) 2 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.40.ffn_gate_exps.weight' has partial data (99.48%) 2 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.13.ffn_gate_exps.weight' has partial data (99.22%) 3 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.31.ffn_down_exps.weight' has partial data (99.22%) 3 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.16.ffn_up_exps.weight' has partial data (99.48%) 2 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.39.ffn_up_exps.weight' has partial data (98.70%) 5 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.33.ffn_gate_exps.weight' has partial data (99.22%) 3 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '              blk.5.ffn_gate_exps.weight' has partial data (97.66%) 9 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.20.ffn_down_exps.weight' has partial data (98.70%) 5 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.35.ffn_up_exps.weight' has partial data (97.92%) 8 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.35.ffn_down_exps.weight' has partial data (97.92%) 8 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.40.ffn_down_exps.weight' has partial data (99.48%) 2 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.37.ffn_gate_exps.weight' has partial data (98.70%) 5 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '              blk.8.ffn_gate_exps.weight' has partial data (98.18%) 7 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.39.ffn_down_exps.weight' has partial data (98.70%) 5 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.38.ffn_gate_exps.weight' has partial data (99.22%) 3 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '                blk.2.ffn_up_exps.weight' has partial data (97.92%) 8 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '              blk.7.ffn_gate_exps.weight' has partial data (98.96%) 4 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.41.ffn_down_exps.weight' has partial data (99.74%) 1 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.42.ffn_gate_exps.weight' has partial data (99.48%) 2 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.35.ffn_gate_exps.weight' has partial data (97.92%) 8 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.42.ffn_down_exps.weight' has partial data (99.48%) 2 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.43.ffn_up_exps.weight' has partial data (99.48%) 2 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.19.ffn_gate_exps.weight' has partial data (99.74%) 1 out of 384 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.43.ffn_down_exps.weight' has partial data (99.48%) 2 out of 384 experts are missing data Storing **but be aware**

save_imatrix: stored collected data after 10 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[10]2.0447,[11]2.0553,[12]2.2739,[13]2.3537,[14]2.3295,[15]2.2035,[16]2.1080,[17]2.0208,[18]1.9580,[19]1.8930,
save_imatrix: stored collected data after 20 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[20]1.8448,[21]1.7927,[22]1.7578,[23]1.7213,[24]1.6852,[25]1.6508,[26]1.7266,[27]1.8283,[28]1.8931,[29]1.8844,
save_imatrix: stored collected data after 30 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[30]1.8766,[31]1.8525,[32]1.8491,[33]1.8515,[34]1.8373,[35]1.8234,[36]1.8112,[37]1.8104,[38]1.8069,[39]1.7878,
save_imatrix: stored collected data after 40 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[40]1.7629,[41]1.7450,[42]1.7292,[43]1.7117,[44]1.6987,[45]1.6951,[46]1.6825,[47]1.6741,[48]1.6705,[49]1.6613,
save_imatrix: stored collected data after 50 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[50]1.6534,[51]1.6677,[52]1.6804,[53]1.6799,[54]1.6973,[55]1.7078,[56]1.7172,[57]1.7084,[58]1.7473,[59]1.7778,
save_imatrix: stored collected data after 60 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[60]1.8056,[61]1.8425,[62]1.8813,[63]1.9221,[64]1.9550,[65]2.0082,[66]2.0360,[67]2.0632,[68]2.1073,[69]2.1413,
save_imatrix: stored collected data after 70 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[70]2.1653,[71]2.1940,[72]2.2106,[73]2.2301,[74]2.2592,[75]2.2820,[76]2.2968,[77]2.3122,[78]2.3190,[79]2.3225,
save_imatrix: stored collected data after 80 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[80]2.3332,[81]2.3528,[82]2.3927,[83]2.4148,[84]2.4200,[85]2.4355,[86]2.4338,[87]2.4763,[88]2.5016,[89]2.5260,
save_imatrix: stored collected data after 90 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[90]2.5510,[91]2.5572,[92]2.5870,[93]2.5924,[94]2.5996,[95]2.6035,[96]2.6109,[97]2.6077,[98]2.6330,[99]2.6132,
save_imatrix: stored collected data after 100 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[100]2.6467,[101]2.6665,[102]2.6550,[103]2.6847,[104]2.7280,[105]2.7568,[106]2.7900,[107]2.8202,[108]2.8503,[109]2.8766,
save_imatrix: stored collected data after 110 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[110]2.8644,[111]2.8817,[112]2.8940,[113]2.9070,[114]2.9062,[115]2.9370,[116]2.9746,[117]2.9949,[118]2.9870,[119]2.9623,
save_imatrix: stored collected data after 120 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[120]2.9471,[121]2.9625,[122]2.9627,[123]2.9414,[124]2.9315,[125]2.9299,[126]2.9324,[127]2.9374,[128]2.9410,[129]2.9433,
save_imatrix: stored collected data after 130 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[130]2.9611,[131]2.9937,[132]3.0287,[133]3.0204,[134]2.9960,[135]2.9719,[136]2.9483,[137]2.9260,[138]2.9292,[139]2.9501,
save_imatrix: stored collected data after 140 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[140]2.9749,[141]3.0066,[142]3.0007,[143]3.0139,[144]3.0319,[145]3.0514,[146]3.0670,[147]3.0876,[148]3.1107,[149]3.1310,
save_imatrix: stored collected data after 150 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[150]3.1519,[151]3.1509,[152]3.1542,[153]3.1568,[154]3.1834,[155]3.1951,[156]3.2028,[157]3.2163,[158]3.2280,[159]3.2295,
save_imatrix: stored collected data after 160 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[160]3.2334,[161]3.2440,[162]3.2529,[163]3.2575,[164]3.2713,[165]3.2735,[166]3.2772,[167]3.2836,[168]3.2885,[169]3.2943,
save_imatrix: stored collected data after 170 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[170]3.2882,[171]3.3057,[172]3.3126,[173]3.3172,[174]3.3278,[175]3.3394,[176]3.3374,[177]3.3441,[178]3.3507,[179]3.3664,
save_imatrix: stored collected data after 180 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[180]3.3795,[181]3.3895,[182]3.3839,[183]3.3796,[184]3.3757,[185]3.3701,[186]3.3651,[187]3.3589,[188]3.3538,[189]3.3612,
save_imatrix: stored collected data after 190 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[190]3.3734,[191]3.4025,[192]3.4251,[193]3.4465,[194]3.4784,[195]3.5022,[196]3.5219,[197]3.5393,[198]3.5498,[199]3.5526,
save_imatrix: stored collected data after 200 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[200]3.5398,[201]3.5190,[202]3.4978,[203]3.5173,[204]3.5273,[205]3.5347,[206]3.5496,[207]3.5697,[208]3.5833,[209]3.5974,
save_imatrix: stored collected data after 210 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[210]3.6182,[211]3.6258,[212]3.6256,[213]3.6040,[214]3.5825,[215]3.5618,[216]3.5414,[217]3.5210,[218]3.5009,[219]3.4832,
save_imatrix: stored collected data after 220 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[220]3.4740,[221]3.4707,[222]3.4537,[223]3.4414,[224]3.4436,[225]3.4451,[226]3.4657,[227]3.4843,[228]3.4953,[229]3.5154,
save_imatrix: stored collected data after 230 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[230]3.5062,[231]3.5273,[232]3.5476,[233]3.5552,[234]3.5718,[235]3.5840,[236]3.6057,[237]3.6260,[238]3.6246,[239]3.6336,
save_imatrix: stored collected data after 240 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[240]3.6456,[241]3.6565,[242]3.6775,[243]3.6929,[244]3.7058,[245]3.7153,[246]3.7066,[247]3.7349,[248]3.7442,[249]3.7650,
save_imatrix: stored collected data after 250 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[250]3.7733,[251]3.7784,[252]3.7874,[253]3.7947,[254]3.8057,[255]3.8116,[256]3.8196,[257]3.8304,[258]3.8384,[259]3.8467,
save_imatrix: stored collected data after 260 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[260]3.8583,[261]3.8709,[262]3.8819,[263]3.8946,[264]3.8779,[265]3.8819,[266]3.8895,[267]3.8978,[268]3.9040,[269]3.9176,
save_imatrix: stored collected data after 270 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[270]3.9365,[271]3.9377,[272]3.9451,[273]3.9542,[274]3.9638,[275]3.9775,[276]3.9886,[277]3.9999,[278]4.0093,[279]4.0130,
save_imatrix: stored collected data after 280 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[280]4.0236,[281]4.0310,[282]4.0362,[283]4.0499,[284]4.0520,[285]4.0557,[286]4.0540,[287]4.0495,[288]4.0615,[289]4.0583,
save_imatrix: stored collected data after 290 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[290]4.0618,[291]4.0807,[292]4.0948,[293]4.1044,[294]4.1210,[295]4.1255,[296]4.1441,[297]4.1552,[298]4.1710,[299]4.1837,
save_imatrix: stored collected data after 300 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[300]4.1961,[301]4.2035,[302]4.2221,[303]4.2279,[304]4.2312,[305]4.2356,[306]4.2510,[307]4.2603,[308]4.2672,[309]4.2743,
save_imatrix: stored collected data after 310 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[310]4.2823,[311]4.2950,[312]4.3023,[313]4.3084,[314]4.3195,[315]4.3304,[316]4.3446,[317]4.3474,[318]4.3325,[319]4.3156,
save_imatrix: stored collected data after 320 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[320]4.3037,[321]4.2875,[322]4.2860,[323]4.2880,[324]4.2691,[325]4.2831,[326]4.2950,[327]4.2992,[328]4.3054,[329]4.3034,
save_imatrix: stored collected data after 330 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[330]4.3036,[331]4.3185,[332]4.3162,[333]4.3262,[334]4.3406,[335]4.3495,[336]4.3521,[337]4.3418,[338]4.3541,[339]4.3684,
save_imatrix: stored collected data after 340 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[340]4.3836,[341]4.3974,[342]4.4151,[343]4.4409,[344]4.4443,[345]4.4444,[346]4.4450,[347]4.4501,[348]4.4658,[349]4.4725,
save_imatrix: stored collected data after 350 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[350]4.4720,[351]4.4696,[352]4.4761,[353]4.4682,[354]4.4626,[355]4.4568,[356]4.4533,[357]4.4560,[358]4.4654,[359]4.4638,
save_imatrix: stored collected data after 360 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[360]4.4644,[361]4.4488,[362]4.4307,[363]4.4157,[364]4.4027,[365]4.3857,[366]4.3739,[367]4.3623,[368]4.3509,[369]4.3422,
save_imatrix: stored collected data after 370 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[370]4.3307,[371]4.3225,[372]4.3189,[373]4.3071,[374]4.2967,[375]4.2871,[376]4.2740,[377]4.2640,[378]4.2608,[379]4.2489,
save_imatrix: stored collected data after 380 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[380]4.2445,[381]4.2489,[382]4.2369,[383]4.2320,[384]4.2229,[385]4.2074,[386]4.1938,[387]4.1897,[388]4.1818,[389]4.1666,
save_imatrix: stored collected data after 390 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[390]4.1514,[391]4.1363,[392]4.1340,[393]4.1288,[394]4.1247,[395]4.1171,[396]4.1098,[397]4.1043,[398]4.0902,[399]4.0791,
save_imatrix: stored collected data after 400 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[400]4.0756,[401]4.0638,[402]4.0550,[403]4.0447,[404]4.0398,[405]4.0295,[406]4.0172,[407]4.0072,[408]3.9978,[409]3.9892,
save_imatrix: stored collected data after 410 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[410]3.9827,[411]3.9762,[412]3.9700,[413]3.9650,[414]3.9583,[415]3.9507,[416]3.9439,[417]3.9313,[418]3.9186,[419]3.9059,
save_imatrix: stored collected data after 420 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[420]3.8960,[421]3.8844,[422]3.8736,[423]3.8632,[424]3.8510,[425]3.8421,[426]3.8321,[427]3.8206,[428]3.8094,[429]3.8012,
save_imatrix: stored collected data after 430 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[430]3.7922,[431]3.7851,[432]3.7848,[433]3.7833,[434]3.7796,[435]3.7709,[436]3.7662,[437]3.7549,[438]3.7445,[439]3.7341,
save_imatrix: stored collected data after 440 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[440]3.7248,[441]3.7155,[442]3.7139,[443]3.7058,[444]3.7040,[445]3.6987,[446]3.6918,[447]3.6922,[448]3.6879,[449]3.6824,
save_imatrix: stored collected data after 450 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[450]3.6737,[451]3.6644,[452]3.6633,[453]3.6562,[454]3.6460,[455]3.6361,[456]3.6273,[457]3.6184,[458]3.6090,[459]3.6005,
save_imatrix: stored collected data after 460 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[460]3.5919,[461]3.5893,[462]3.5814,[463]3.5770,[464]3.5740,[465]3.5712,[466]3.5668,[467]3.5624,[468]3.5589,[469]3.5547,
save_imatrix: stored collected data after 470 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[470]3.5509,[471]3.5473,[472]3.5433,[473]3.5394,[474]3.5355,[475]3.5315,[476]3.5277,[477]3.5225,[478]3.5150,[479]3.5061,
save_imatrix: stored collected data after 480 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[480]3.5025,[481]3.4995,[482]3.4998,[483]3.4914,[484]3.4839,[485]3.4784,[486]3.4715,[487]3.4640,[488]3.4572,[489]3.4516,
save_imatrix: stored collected data after 490 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[490]3.4454,[491]3.4428,[492]3.4370,[493]3.4317,[494]3.4278,[495]3.4239,[496]3.4174,[497]3.4155,[498]3.4167,[499]3.4200,
save_imatrix: stored collected data after 500 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[500]3.4197,[501]3.4218,[502]3.4232,[503]3.4190,[504]3.4123,[505]3.4200,[506]3.4290,[507]3.4385,[508]3.4462,[509]3.4529,
save_imatrix: stored collected data after 510 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[510]3.4615,[511]3.4646,[512]3.4667,[513]3.4676,[514]3.4706,[515]3.4731,[516]3.4767,[517]3.4750,[518]3.4901,[519]3.5010,
save_imatrix: stored collected data after 520 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[520]3.5139,[521]3.5209,[522]3.5251,[523]3.5289,[524]3.5327,[525]3.5361,[526]3.5361,[527]3.5358,[528]3.5403,[529]3.5435,
save_imatrix: stored collected data after 530 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[530]3.5473,[531]3.5494,[532]3.5515,[533]3.5546,[534]3.5587,[535]3.5615,[536]3.5674,[537]3.5684,[538]3.5739,[539]3.5771,
save_imatrix: stored collected data after 540 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[540]3.5802,[541]3.5840,[542]3.5871,[543]3.5894,[544]3.5886,[545]3.5892,[546]3.5910,[547]3.5931,[548]3.5945,[549]3.5971,
save_imatrix: stored collected data after 550 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[550]3.6003,[551]3.6015,[552]3.6033,[553]3.6079,[554]3.6123,[555]3.6171,[556]3.6223,[557]3.6286,[558]3.6324,[559]3.6357,
save_imatrix: stored collected data after 560 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[560]3.6387,[561]3.6410,[562]3.6424,[563]3.6404,[564]3.6422,[565]3.6436,[566]3.6446,[567]3.6411,[568]3.6420,[569]3.6433,
save_imatrix: stored collected data after 570 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[570]3.6410,[571]3.6428,[572]3.6429,[573]3.6457,[574]3.6454,[575]3.6411,[576]3.6340,[577]3.6323,[578]3.6305,[579]3.6284,
save_imatrix: stored collected data after 580 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[580]3.6253,[581]3.6211,[582]3.6139,[583]3.6082,[584]3.6009,[585]3.5942,[586]3.5871,[587]3.5798,[588]3.5804,[589]3.5793,
save_imatrix: stored collected data after 590 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[590]3.5794,[591]3.5802,[592]3.5782,[593]3.5778,[594]3.5768,[595]3.5747,[596]3.5718,[597]3.5710,[598]3.5696,[599]3.5690,
save_imatrix: stored collected data after 600 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[600]3.5648,[601]3.5615,[602]3.5597,[603]3.5551,[604]3.5510,[605]3.5483,[606]3.5472,[607]3.5461,[608]3.5469,[609]3.5459,
save_imatrix: stored collected data after 610 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[610]3.5453,[611]3.5397,[612]3.5378,[613]3.5400,[614]3.5415,[615]3.5407,[616]3.5394,[617]3.5361,[618]3.5349,[619]3.5337,
save_imatrix: stored collected data after 620 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[620]3.5347,[621]3.5329,[622]3.5301,[623]3.5310,[624]3.5312,[625]3.5254,[626]3.5200,[627]3.5136,[628]3.5078,[629]3.5014,
save_imatrix: stored collected data after 630 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[630]3.4957,[631]3.4895,[632]3.4836,[633]3.4770,[634]3.4703,[635]3.4635,[636]3.4626,[637]3.4563,[638]3.4515,[639]3.4457,
save_imatrix: stored collected data after 640 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[640]3.4398,[641]3.4353,[642]3.4294,[643]3.4256,[644]3.4250,[645]3.4190,[646]3.4152,[647]3.4143,[648]3.4103,[649]3.4039,
save_imatrix: stored collected data after 650 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[650]3.3983,[651]3.3924,[652]3.3864,[653]3.3805,[654]3.3747,[655]3.3688,[656]3.3630,[657]3.3573,[658]3.3514,[659]3.3458,
save_imatrix: stored collected data after 660 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[660]3.3409,[661]3.3353,[662]3.3296,[663]3.3238,[664]3.3182,[665]3.3125,[666]3.3068,[667]3.3011,[668]3.2956,[669]3.2900,
save_imatrix: stored collected data after 670 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[670]3.2843,[671]3.2865,[672]3.2869,[673]3.2903,[674]3.2889,[675]3.2839,[676]3.2800,[677]3.2753,[678]3.2702,[679]3.2666,
save_imatrix: stored collected data after 680 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[680]3.2610,[681]3.2564,[682]3.2516,[683]3.2466,[684]3.2415,[685]3.2367,[686]3.2326,[687]3.2281,[688]3.2234,[689]3.2184,
save_imatrix: stored collected data after 690 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[690]3.2148,[691]3.2097,[692]3.2057,[693]3.2010,[694]3.1958,[695]3.1906,[696]3.1882,[697]3.1832,[698]3.1796,[699]3.1764,
save_imatrix: stored collected data after 700 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[700]3.1721,[701]3.1699,[702]3.1651,[703]3.1604,[704]3.1560,[705]3.1522,[706]3.1483,[707]3.1459,[708]3.1452,[709]3.1441,
save_imatrix: stored collected data after 710 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[710]3.1427,[711]3.1413,[712]3.1399,[713]3.1385,[714]3.1382,[715]3.1372,[716]3.1365,[717]3.1354,[718]3.1341,[719]3.1326,
save_imatrix: stored collected data after 720 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[720]3.1318,[721]3.1303,[722]3.1288,[723]3.1280,[724]3.1278,[725]3.1290,[726]3.1302,[727]3.1323,[728]3.1336,[729]3.1355,
save_imatrix: stored collected data after 730 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[730]3.1371,[731]3.1400,[732]3.1418,[733]3.1421,[734]3.1435,[735]3.1451,[736]3.1465,[737]3.1480,[738]3.1508,[739]3.1527,
save_imatrix: stored collected data after 740 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[740]3.1546,[741]3.1554,[742]3.1561,[743]3.1575,[744]3.1610,[745]3.1629,[746]3.1640,[747]3.1647,[748]3.1657,[749]3.1682,
save_imatrix: stored collected data after 750 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[750]3.1696,[751]3.1716,[752]3.1728,[753]3.1751,[754]3.1767,[755]3.1779,[756]3.1786,[757]3.1800,[758]3.1820,[759]3.1838,
save_imatrix: stored collected data after 760 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[760]3.1855,[761]3.1872,[762]3.1894,[763]3.1908,[764]3.1931,[765]3.1945,[766]3.1958,[767]3.1970,[768]3.1996,[769]3.2020,
save_imatrix: stored collected data after 770 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[770]3.2032,[771]3.2049,[772]3.2065,[773]3.2076,[774]3.2098,[775]3.2108,[776]3.2133,[777]3.2140,[778]3.2160,[779]3.2173,
save_imatrix: stored collected data after 780 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[780]3.2185,[781]3.2198,[782]3.2218,[783]3.2232,[784]3.2250,[785]3.2258,[786]3.2273,[787]3.2296,[788]3.2318,[789]3.2344,
save_imatrix: stored collected data after 790 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[790]3.2348,[791]3.2370,[792]3.2377,[793]3.2396,[794]3.2418,[795]3.2442,[796]3.2452,[797]3.2464,[798]3.2484,[799]3.2498,
save_imatrix: stored collected data after 800 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[800]3.2509,[801]3.2531,[802]3.2543,[803]3.2562,[804]3.2570,[805]3.2588,[806]3.2606,[807]3.2614,[808]3.2615,[809]3.2624,
save_imatrix: stored collected data after 810 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[810]3.2642,[811]3.2663,[812]3.2670,[813]3.2674,[814]3.2696,[815]3.2714,[816]3.2731,[817]3.2749,[818]3.2766,[819]3.2782,
save_imatrix: stored collected data after 820 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat
[820]3.2799,[821]3.2816,[822]3.2831,[823]3.2844,[824]3.2857,[825]3.2868,[826]3.2880,
save_imatrix: stored collected data after 826 chunks in /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat

Final estimate: PPL = 3.2880 +/- 0.01495

llama_print_timings:        load time =   44750.06 ms
llama_print_timings:      sample time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings: prompt eval time = 11519326.21 ms / 422912 tokens (   27.24 ms per token,    36.71 tokens per second)
llama_print_timings:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings:       total time = 11547591.52 ms / 422913 tokens

======================== sorted layer importances
  0: Layer   0, <cos_sim> = 0.54895
  1: Layer   3, <cos_sim> = 0.78673
  2: Layer  60, <cos_sim> = 0.801527
  3: Layer   1, <cos_sim> = 0.813399
  4: Layer   2, <cos_sim> = 0.818612
  5: Layer   6, <cos_sim> = 0.861989
  6: Layer   5, <cos_sim> = 0.867296
  7: Layer   4, <cos_sim> = 0.882697
  8: Layer  13, <cos_sim> = 0.903652
  9: Layer  14, <cos_sim> = 0.904641
 10: Layer  12, <cos_sim> = 0.908486
 11: Layer  10, <cos_sim> = 0.910599
 12: Layer  15, <cos_sim> = 0.912134
 13: Layer   8, <cos_sim> = 0.920342
 14: Layer  59, <cos_sim> = 0.920597
 15: Layer  16, <cos_sim> = 0.920976
 16: Layer  18, <cos_sim> = 0.921181
 17: Layer  17, <cos_sim> = 0.921184
 18: Layer   7, <cos_sim> = 0.921782
 19: Layer  11, <cos_sim> = 0.925572
 20: Layer   9, <cos_sim> = 0.928968
 21: Layer  23, <cos_sim> = 0.930959
 22: Layer  24, <cos_sim> = 0.931823
 23: Layer  20, <cos_sim> = 0.932656
 24: Layer  21, <cos_sim> = 0.933109
 25: Layer  22, <cos_sim> = 0.933117
 26: Layer  19, <cos_sim> = 0.9381
 27: Layer  27, <cos_sim> = 0.938829
 28: Layer  25, <cos_sim> = 0.93897
 29: Layer  29, <cos_sim> = 0.942702
 30: Layer  26, <cos_sim> = 0.944389
 31: Layer  30, <cos_sim> = 0.944426
 32: Layer  28, <cos_sim> = 0.944919
 33: Layer  58, <cos_sim> = 0.95032
 34: Layer  32, <cos_sim> = 0.950972
 35: Layer  31, <cos_sim> = 0.951155
 36: Layer  34, <cos_sim> = 0.955866
 37: Layer  57, <cos_sim> = 0.956497
 38: Layer  56, <cos_sim> = 0.956722
 39: Layer  35, <cos_sim> = 0.957403
 40: Layer  55, <cos_sim> = 0.959099
 41: Layer  33, <cos_sim> = 0.960765
 42: Layer  54, <cos_sim> = 0.964028
 43: Layer  36, <cos_sim> = 0.964487
 44: Layer  43, <cos_sim> = 0.965472
 45: Layer  37, <cos_sim> = 0.965515
 46: Layer  53, <cos_sim> = 0.967063
 47: Layer  42, <cos_sim> = 0.967151
 48: Layer  38, <cos_sim> = 0.967854
 49: Layer  52, <cos_sim> = 0.969066
 50: Layer  39, <cos_sim> = 0.969307
 51: Layer  40, <cos_sim> = 0.970395
 52: Layer  51, <cos_sim> = 0.971089
 53: Layer  50, <cos_sim> = 0.971141
 54: Layer  49, <cos_sim> = 0.972038
 55: Layer  41, <cos_sim> = 0.972043
 56: Layer  46, <cos_sim> = 0.972849
 57: Layer  45, <cos_sim> = 0.973011
 58: Layer  44, <cos_sim> = 0.973383
 59: Layer  47, <cos_sim> = 0.974174
 60: Layer  48, <cos_sim> = 0.975424

======================== sorted attention importances
  0: Layer   0, <cos_sim> = 0.528929
  1: Layer   3, <cos_sim> = 0.609517
  2: Layer   2, <cos_sim> = 0.737416
  3: Layer   1, <cos_sim> = 0.765176
  4: Layer   6, <cos_sim> = 0.822542
  5: Layer   4, <cos_sim> = 0.852033
  6: Layer   8, <cos_sim> = 0.869524
  7: Layer   5, <cos_sim> = 0.870499
  8: Layer  10, <cos_sim> = 0.872662
  9: Layer   9, <cos_sim> = 0.879495
 10: Layer   7, <cos_sim> = 0.883822
 11: Layer  14, <cos_sim> = 0.898271
 12: Layer  12, <cos_sim> = 0.899972
 13: Layer  13, <cos_sim> = 0.912961
 14: Layer  15, <cos_sim> = 0.918265
 15: Layer  11, <cos_sim> = 0.926531
 16: Layer  18, <cos_sim> = 0.934695
 17: Layer  16, <cos_sim> = 0.937328
 18: Layer  17, <cos_sim> = 0.941984
 19: Layer  23, <cos_sim> = 0.944046
 20: Layer  20, <cos_sim> = 0.945272
 21: Layer  28, <cos_sim> = 0.946108
 22: Layer  31, <cos_sim> = 0.946182
 23: Layer  43, <cos_sim> = 0.947348
 24: Layer  25, <cos_sim> = 0.948715
 25: Layer  32, <cos_sim> = 0.950976
 26: Layer  22, <cos_sim> = 0.953634
 27: Layer  21, <cos_sim> = 0.953882
 28: Layer  30, <cos_sim> = 0.953936
 29: Layer  24, <cos_sim> = 0.954201
 30: Layer  29, <cos_sim> = 0.95446
 31: Layer  19, <cos_sim> = 0.955263
 32: Layer  38, <cos_sim> = 0.956067
 33: Layer  27, <cos_sim> = 0.95718
 34: Layer  34, <cos_sim> = 0.957277
 35: Layer  26, <cos_sim> = 0.958979
 36: Layer  35, <cos_sim> = 0.961119
 37: Layer  39, <cos_sim> = 0.961912
 38: Layer  36, <cos_sim> = 0.962639
 39: Layer  33, <cos_sim> = 0.962902
 40: Layer  45, <cos_sim> = 0.963958
 41: Layer  54, <cos_sim> = 0.964774
 42: Layer  49, <cos_sim> = 0.966144
 43: Layer  37, <cos_sim> = 0.967254
 44: Layer  55, <cos_sim> = 0.967591
 45: Layer  42, <cos_sim> = 0.967956
 46: Layer  57, <cos_sim> = 0.968065
 47: Layer  59, <cos_sim> = 0.968123
 48: Layer  56, <cos_sim> = 0.968696
 49: Layer  60, <cos_sim> = 0.969505
 50: Layer  40, <cos_sim> = 0.969653
 51: Layer  58, <cos_sim> = 0.969745
 52: Layer  52, <cos_sim> = 0.970129
 53: Layer  41, <cos_sim> = 0.970522
 54: Layer  50, <cos_sim> = 0.972281
 55: Layer  47, <cos_sim> = 0.972728
 56: Layer  44, <cos_sim> = 0.974193
 57: Layer  48, <cos_sim> = 0.974345
 58: Layer  46, <cos_sim> = 0.978292
 59: Layer  51, <cos_sim> = 0.979166
 60: Layer  53, <cos_sim> = 0.979395

======================== sorted ffn importances
  0: Layer   2, <cos_sim> = 0.600363
  1: Layer   0, <cos_sim> = 0.78679
  2: Layer   1, <cos_sim> = 0.78881
  3: Layer  60, <cos_sim> = 0.804641
  4: Layer   3, <cos_sim> = 0.814239
  5: Layer   8, <cos_sim> = 0.846997
  6: Layer   5, <cos_sim> = 0.848068
  7: Layer   7, <cos_sim> = 0.850158
  8: Layer   6, <cos_sim> = 0.857595
  9: Layer   9, <cos_sim> = 0.862339
 10: Layer   4, <cos_sim> = 0.873048
 11: Layer  13, <cos_sim> = 0.882637
 12: Layer  11, <cos_sim> = 0.887424
 13: Layer  10, <cos_sim> = 0.899452
 14: Layer  12, <cos_sim> = 0.902722
 15: Layer  14, <cos_sim> = 0.910508
 16: Layer  15, <cos_sim> = 0.920924
 17: Layer  17, <cos_sim> = 0.922436
 18: Layer  16, <cos_sim> = 0.924198
 19: Layer  27, <cos_sim> = 0.927228
 20: Layer  22, <cos_sim> = 0.927292
 21: Layer  19, <cos_sim> = 0.930707
 22: Layer  18, <cos_sim> = 0.931487
 23: Layer  24, <cos_sim> = 0.932161
 24: Layer  42, <cos_sim> = 0.932389
 25: Layer  59, <cos_sim> = 0.932592
 26: Layer  30, <cos_sim> = 0.932863
 27: Layer  20, <cos_sim> = 0.936043
 28: Layer  31, <cos_sim> = 0.938531
 29: Layer  21, <cos_sim> = 0.938706
 30: Layer  25, <cos_sim> = 0.941162
 31: Layer  37, <cos_sim> = 0.941747
 32: Layer  26, <cos_sim> = 0.941901
 33: Layer  23, <cos_sim> = 0.942403
 34: Layer  29, <cos_sim> = 0.942694
 35: Layer  28, <cos_sim> = 0.944772
 36: Layer  32, <cos_sim> = 0.945089
 37: Layer  38, <cos_sim> = 0.94832
 38: Layer  35, <cos_sim> = 0.94834
 39: Layer  33, <cos_sim> = 0.949204
 40: Layer  44, <cos_sim> = 0.950097
 41: Layer  53, <cos_sim> = 0.951411
 42: Layer  58, <cos_sim> = 0.951411
 43: Layer  34, <cos_sim> = 0.951841
 44: Layer  36, <cos_sim> = 0.952884
 45: Layer  48, <cos_sim> = 0.953085
 46: Layer  55, <cos_sim> = 0.953457
 47: Layer  41, <cos_sim> = 0.954042
 48: Layer  51, <cos_sim> = 0.954259
 49: Layer  56, <cos_sim> = 0.954951
 50: Layer  39, <cos_sim> = 0.955321
 51: Layer  57, <cos_sim> = 0.955975
 52: Layer  40, <cos_sim> = 0.956567
 53: Layer  54, <cos_sim> = 0.956844
 54: Layer  49, <cos_sim> = 0.956899
 55: Layer  46, <cos_sim> = 0.957188
 56: Layer  43, <cos_sim> = 0.958947
 57: Layer  47, <cos_sim> = 0.959267
 58: Layer  52, <cos_sim> = 0.963017
 59: Layer  50, <cos_sim> = 0.963043
 60: Layer  45, <cos_sim> = 0.964393
```

</details>

---

👤 **ubergarm** commented on **2025-07-18** at **15:14:24**

The quickest way for me to test some more IQ3_KS tensors with this PR is to re-do my Kimi-K2-Instruct IQ2_KL which uses:

* llama_model_loader: - type iq3_ks:   60 tensors ffn_down_exps
* llama_model_loader: - type iq2_kl:  120 tensors ffn_(gate|up)_exps

Those ffn_down_exps are the only tensors in the recipe affected by this PR so can compare before/after. I'll update this comment with results soon:

*WIP* *TODO* add results for https://huggingface.co/ubergarm/Kimi-K2-Instruct-GGUF#-iq2_kl-345687-gib-2892-bpw

Well using this PR624 for iq3_ks ffn_down_exps increased PPl slightly for PR624-IQ2_KL as shown on graph and data over at: https://github.com/ikawrakow/ik_llama.cpp/pull/616#issuecomment-3087170346

* main-IQ2_KL 3.2741 +/- 0.01689
* PR-IQ2_KL 3.3055 +/- 0.01709

So far the two tests that show increasing/worse perplexity were specifically related to using IQ3_KS with routed experts tensors `ffn_*_exps` on kimi-k2... This model is a pain to work with given its size hah...

I'll try out some more small tests with my set of Qwen3-14B as that should be faster.

---

👤 **ubergarm** commented on **2025-07-18** at **18:37:43**

Well I took some of my old Qwen3-14B quants and added to the set to compare all the types involved here between main and this PR. This data might just muddy the waters even more hah..

I tested with my usual `perplexity: calculating perplexity over 584 chunks, n_ctx=512, batch_size=2048, n_seq=4` on my same `wiki.test.raw`. This imatrix corpus used for these imatrix quants is the same corpus I used on kimi-k2-instrict fwiw.

<details>

<summary>👈 JSON data</summary>

```json
[
  {
    "name": "bf16",
    "ppl": "9.0133 +/- 0.07115",
    "size": 27.509,
    "bpw": 16.000,
    "legend": "main"
  },
  {
    "name": "q4_K",
    "ppl": "9.1487 +/- 0.07232",
    "size": 7.925,
    "bpw": 4.609,
    "legend": "main"
  },
  {
    "name": "q2_K",
    "ppl": "10.6691 +/- 0.08376",
    "size": 5.041,
    "bpw": 2.932,
    "legend": "main"
  },
  {
    "name": "q3_K",
    "ppl": "9.4405 +/- 0.07422",
    "size": 6.291,
    "bpw": 3.659,
    "legend": "main"
  },
  {
    "name": "q5_K",
    "ppl": "9.0413 +/- 0.07128",
    "size": 9.463,
    "bpw": 5.504,
    "legend": "main"
  },
  {
    "name": "iq3_ks",
    "ppl": "9.6945 +/- 0.07826",
    "size": 5.910,
    "bpw": 3.438,
    "legend": "main"
  },
  {
    "name": "iq3_k",
    "ppl": "9.3296 +/- 0.07371",
    "size": 6.291,
    "bpw": 3.659,
    "legend": "main"
  },
  {
    "name": "iq2_ks",
    "ppl": "11.8117 +/- 0.09367",
    "size": 4.372,
    "bpw": 2.543,
    "legend": "main"
  },
  {
    "name": "PR624-q2_K",
    "ppl": "10.7015 +/- 0.08453",
    "size": 5.041,
    "bpw": 2.932,
    "legend": "PR624"
  },
  {
    "name": "PR624-q3_K",
    "ppl": "9.3747 +/- 0.07318",
    "size": 6.291,
    "bpw": 3.659,
    "legend": "PR624"
  },
  {
    "name": "PR624-q4_K",
    "ppl": "9.1210 +/- 0.07194",
    "size": 7.925,
    "bpw": 4.609,
    "legend": "PR624"
  },
  {
    "name": "PR624-q5_K",
    "ppl": "9.0391 +/- 0.07129",
    "size": 9.463,
    "bpw": 5.504,
    "legend": "PR624"
  },
  {
    "name": "PR624-iq2_ks",
    "ppl": "11.8160 +/- 0.09371",
    "size": 4.372,
    "bpw": 2.543,
    "legend": "PR624"
  },
  {
    "name": "PR624-iq3_ks",
    "ppl": "9.5529 +/- 0.07619",
    "size": 5.910,
    "bpw": 3.438,
    "legend": "PR624"
  },
  {
    "name": "PR624-iq3_k",
    "ppl": "9.3818 +/- 0.07445",
    "size": 6.291,
    "bpw": 3.659,
    "legend": "PR624"
  }
]
```

</details>

<img width="1776" height="1179" alt="ppl-Qwen3-14B-pr624" src="https://github.com/user-attachments/assets/3be09c9e-1a7d-4684-b2c0-f601b14cdb2e" />

* "better" on this PR624: iq3_ks q3_K q4_K q5_K
* "better" on main: iq2_ks q2_K iq3_k

These results are annoying because I'm seeing worse iq3_ks PPL on Kimi-K2-Instruct hah. I'm not sure how to read the tea leaves here, and I didn't check `n_ctx 2048` and I know some of these "pure" mixes are really over-quantized for a dense model.

---

👤 **ikawrakow** commented on **2025-07-19** at **06:49:33**

Can I see your imatrix calibration data?

---

👤 **ubergarm** commented on **2025-07-19** at **15:08:07**

@ikawrakow 

* [ubergarm-imatrix-calibration-corpus-v02.txt](https://gist.github.com/ubergarm/edfeb3ff9c6ec8b49e88cdf627b0711a)
* [Qwen3-14B imatrix dat with above corpus](https://huggingface.co/ubergarm/Qwen3-14B-GGUF/blob/main/imatrix-v02-Qwen3-14B-BF16.dat)
* [Kimi-K2-Instruct imatrix dat with above corpus](https://huggingface.co/ubergarm/Kimi-K2-Instruct-GGUF/blob/main/imatrix-Kimi-K2-Instruct-Q8_0.dat)

I'd like to spend some time improving my automation/scripts to remove the human error in making these graphs at some point. Thanks for rolling with what we have so far!