## 📌 [Issue #658](https://github.com/ikawrakow/ik_llama.cpp/issues/658) - Bug: Kimi K2 quantization fails with error "Unknown model"

| **Author** | `Lissanro` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2025-07-27 |
| **Updated** | 2025-07-27 |

---

## 📄 Description

### What happened?

I tried to quantize Kimi K2 model from BF16:

```
~/pkgs/ik_llama.cpp/build/bin/llama-quantize \
/mnt/neuro/models/Kimi-K2-Instruct/Kimi-K2-Instruct-BF16.gguf \
/mnt/neuro/models/Kimi-K2-Instruct/Kimi-K2-Instruct-Q6_K.gguf \
Q6_K
```

And get this error:

```
Sorry, uknown model => cannot fix it => bailing out
```

Maybe support for K2 wasn't added yet for quantization (I tested also with llama.cpp, and it finished quantizing it successfully, even though with a warning "61 of 731 tensor(s) required fallback quantization" due to "tensor cols 128 x 512 are not divisible by 256, required for q6_K" for some of the tensors, so I am assuming by BF16 files are fine).

Also, very minor thing - there is a typo in the error message ("uknown" instead of "unknown").

### Name and Version

> ~/pkgs/ik_llama.cpp/build/bin/llama-cli --version
version: 3826 (ae0ba31f)
built with cc (Ubuntu 14.2.0-19ubuntu2) 14.2.0 for x86_64-linux-gnu

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
> ~/pkgs/ik_llama.cpp/build/bin/llama-quantize /mnt/neuro/models/Kimi-K2-Instruct/Kimi-K2-Instruct-BF16.gguf /mnt/neuro/models/Kimi-K2-Instruct/Kimi-K2-Instruct-Q6_K.gguf Q6_K
main: build = 3826 (ae0ba31f)
main: built with cc (Ubuntu 14.2.0-19ubuntu2) 14.2.0 for x86_64-linux-gnu
main: quantizing '/mnt/neuro/models/Kimi-K2-Instruct/Kimi-K2-Instruct-BF16.gguf' to '/mnt/neuro/models/Kimi-K2-Instruct/Kimi-K2-Instruct-Q6_K.2.gguf' as Q6_K
llama_model_loader: loaded meta data with 43 key-value pairs and 1096 tensors from /mnt/neuro/models/Kimi-K2-Instruct/Kimi-K2-Instruct-BF16.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Kimi K2 Instruct
llama_model_loader: - kv   3:                           general.finetune str              = Instruct
llama_model_loader: - kv   4:                           general.basename str              = Kimi-K2
llama_model_loader: - kv   5:                         general.size_label str              = 384x14B
llama_model_loader: - kv   6:                            general.license str              = other
llama_model_loader: - kv   7:                       general.license.name str              = modified-mit
llama_model_loader: - kv   8:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv   9:                   deepseek2.context_length u32              = 131072
llama_model_loader: - kv  10:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv  11:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  12:             deepseek2.attention.head_count u32              = 64
llama_model_loader: - kv  13:          deepseek2.attention.head_count_kv u32              = 1
llama_model_loader: - kv  14:                   deepseek2.rope.freq_base f32              = 50000.000000
llama_model_loader: - kv  15: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  16:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  17:                          general.file_type u32              = 32
llama_model_loader: - kv  18:        deepseek2.leading_dense_block_count u32              = 1
llama_model_loader: - kv  19:                       deepseek2.vocab_size u32              = 163840
llama_model_loader: - kv  20:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  21:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  22:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  23:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  24:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  25:                     deepseek2.expert_count u32              = 384
llama_model_loader: - kv  26:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  27:             deepseek2.expert_weights_scale f32              = 2.827000
llama_model_loader: - kv  28:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  29:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  30:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  31:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  32:              deepseek2.rope.scaling.factor f32              = 32.000000
llama_model_loader: - kv  33: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  34: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  35:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  36:                         tokenizer.ggml.pre str              = kimi-k2
llama_model_loader: - kv  37:                      tokenizer.ggml.tokens arr[str,163842]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  38:                  tokenizer.ggml.token_type arr[i32,163842]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  39:                tokenizer.ggml.bos_token_id u32              = 163584
llama_model_loader: - kv  40:                tokenizer.ggml.eos_token_id u32              = 163585
llama_model_loader: - kv  41:                    tokenizer.chat_template str              = {%- if tools -%}\n  <|im_system|>tool_...
llama_model_loader: - kv  42:               general.quantization_version u32              = 2
llama_model_loader: - type  f32:  365 tensors
llama_model_loader: - type bf16:  731 tensors
==========================================================================
Detected incompatible DeepSeek model.
Will try to fix, but there are no guarantees

*** Your prompt processing speed will be crippled ***

Consider making your own ik_llama.cpp compatible model or
ask the model provider to make one for you,
Sorry, uknown model => cannot fix it => bailing out
```

---

## 💬 Conversation

👤 **ikawrakow** commented on **2025-07-27** at **15:55:14**

This is a GGUF that you made yourself, correct?

The self-attention related metadata needs to be the way it comes out of either the `ik_llama.cpp` or the `llama.cpp` `convert_hf_to_gguf.py` conversion script, else it will not work. I'm on mobile, so difficult to sort out exactly what is not correct in your GGUF.

---

👤 **Lissanro** commented on **2025-07-27** at **16:44:09**

Yes, correct, due to limits of my internet connection I could not download premade BF16, only the original FP8, so had to create my own BF16 GGUF.

I have used `convert_hf_to_gguf.py` from [llama.cpp evshiron fork](https://github.com/evshiron/llama.cpp), with the [patch](https://dragon.studio/2025/07/lama.cpp-fp8-to-bf16-patch.diff) applied (the patch is based on the differences between the evshiron fork and the upstream llama.cpp related to the conversion script and Kimi K2 updates).

I ran this command to convert from the original FP8 to BF16:

```
python3 ~/pkgs/llama.cpp-fp8-to-bf16/llama.cpp/convert_hf_to_gguf.py \
--outtype bf16 \
--outfile /mnt/neuro/Kimi-K2-Instruct/Kimi-K2-Instruct-BF16.gguf \
/mnt/neuro/models/Kimi-K2-Instruct/
```

Maybe I missed a step or some extra option required to add metadata? But given `convert_hf_to_gguf.py` conversion script in ik_llama.cpp or mainline llama.cpp does not support Kimi K2 original model that comes in FP8 format, could you please clarify what did you mean by "_self-attention related metadata needs to be the way it comes out of either the ik_llama.cpp or the llama.cpp convert_hf_to_gguf.py conversion script_" - do I need to extract this metadata somehow and add it manually, or is it something that needs to be patched in the evshiron's llama.cpp fork? Maybe lack of some metadata is also the reason why mainline's `convert_hf_to_gguf.py` prints warnings when processing my BF16.