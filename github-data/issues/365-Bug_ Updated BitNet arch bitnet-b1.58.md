### 🐛 [#365](https://github.com/ikawrakow/ik_llama.cpp/issues/365) - Bug: Updated BitNet arch bitnet-b1.58

| **Author** | `jdluzen` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-01 |
| **Updated** | 2025-05-03 |

---

#### Description

### What happened?

I'm very rusty at ggml, quants, etc. so please forgive my ignorance.
I've been attempting to get BitNet running, and by that I mean the _new_ BitNet as of April 23rd. MS uploaded a new version to HF, replacing the old one, and it seems to have breaking changes.
From what I gather, #337 add support for the original 2025 BitNet with arch `bitnet-25`, but now the new one is `bitnet-b1.58`. I've been trying to add the changes from https://github.com/microsoft/BitNet/pull/212 with limited success. I'm also guessing that I need https://github.com/ggml-org/llama.cpp/compare/gg/bitnet since I am crashing because `vec_dot` is null at https://github.com/ikawrakow/ik_llama.cpp/blob/main/ggml/src/ggml.c#L14311 when `type` is `GGML_TYPE_I2_S` 36. Will try to get that implementation going next. I'm also on Windows arm64 which makes things more fun 😅
Am I on the right track here?

### Name and Version

Tip of main 98d1626469879d35faba9cb7e9d0b1ddaf853eee.

### What operating system are you seeing the problem on?

Windows

### Relevant log output

```shell

```

---

#### 💬 Conversation

👤 **usatenko** commented the **2025-05-02** at **01:26:16**:<br>

looks like I faced the same problem on macos, new ms model
`./bin/llama-quantize --allow-requantize models/ggml-model-i2_s.gguf ggml-model-i2_s_bn.gguf iq2_bn`
```
main: build = 3657 (98d16264)
main: built with Apple clang version 17.0.0 (clang-1700.0.13.3) for arm64-apple-darwin24.4.0
main: quantizing 'models/ggml-model-i2_s.gguf' to 'ggml-model-i2_s_bn.gguf' as IQ2_BN
llama_model_loader: loaded meta data with 24 key-value pairs and 332 tensors from models/ggml-model-i2_s.gguf (version GGUF V3 (latest))
llama_model_loader: unknown type i2_s
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = bitnet-b1.58
llama_model_loader: - kv   1:                               general.name str              = bitnet2b
llama_model_loader: - kv   2:                    bitnet-b1.58.vocab_size u32              = 128256
llama_model_loader: - kv   3:                bitnet-b1.58.context_length u32              = 4096
llama_model_loader: - kv   4:              bitnet-b1.58.embedding_length u32              = 2560
llama_model_loader: - kv   5:                   bitnet-b1.58.block_count u32              = 30
llama_model_loader: - kv   6:           bitnet-b1.58.feed_forward_length u32              = 6912
llama_model_loader: - kv   7:          bitnet-b1.58.rope.dimension_count u32              = 128
llama_model_loader: - kv   8:          bitnet-b1.58.attention.head_count u32              = 20
llama_model_loader: - kv   9:       bitnet-b1.58.attention.head_count_kv u32              = 5
llama_model_loader: - kv  10:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  11: bitnet-b1.58.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  12:                bitnet-b1.58.rope.freq_base f32              = 500000.000000
llama_model_loader: - kv  13:                          general.file_type u32              = 40
llama_model_loader: - kv  14:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  15:                      tokenizer.ggml.tokens arr[str,128256]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  16:                      tokenizer.ggml.scores arr[f32,128256]  = [0.000000, 0.000000, 0.000000, 0.0000...
llama_model_loader: - kv  17:                  tokenizer.ggml.token_type arr[i32,128256]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  18:                      tokenizer.ggml.merges arr[str,280147]  = ["Ġ Ġ", "Ġ ĠĠĠ", "ĠĠ ĠĠ", "...
llama_model_loader: - kv  19:                tokenizer.ggml.bos_token_id u32              = 128000
llama_model_loader: - kv  20:                tokenizer.ggml.eos_token_id u32              = 128001
llama_model_loader: - kv  21:            tokenizer.ggml.padding_token_id u32              = 128001
llama_model_loader: - kv  22:                    tokenizer.chat_template str              = {% for message in messages %}{% if lo...
llama_model_loader: - kv  23:               general.quantization_version u32              = 2
llama_model_loader: - type  f32:  121 tensors
llama_model_loader: - type  f16:    1 tensors
llama_model_loader: - type i2_s:  210 tensors
llama_model_quantize: failed to quantize: unknown model architecture: 'bitnet-b1.58'
main: failed to quantize model from 'models/ggml-model-i2_s.gguf'
```
@ikawrakow can you help?

the model is from here: https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf

---

👤 **usatenko** commented the **2025-05-02** at **01:26:16**:<br>

looks like I faced the same problem on macos, new ms model
`./bin/llama-quantize --allow-requantize models/ggml-model-i2_s.gguf ggml-model-i2_s_bn.gguf iq2_bn`
```
main: build = 3657 (98d16264)
main: built with Apple clang version 17.0.0 (clang-1700.0.13.3) for arm64-apple-darwin24.4.0
main: quantizing 'models/ggml-model-i2_s.gguf' to 'ggml-model-i2_s_bn.gguf' as IQ2_BN
llama_model_loader: loaded meta data with 24 key-value pairs and 332 tensors from models/ggml-model-i2_s.gguf (version GGUF V3 (latest))
llama_model_loader: unknown type i2_s
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = bitnet-b1.58
llama_model_loader: - kv   1:                               general.name str              = bitnet2b
llama_model_loader: - kv   2:                    bitnet-b1.58.vocab_size u32              = 128256
llama_model_loader: - kv   3:                bitnet-b1.58.context_length u32              = 4096
llama_model_loader: - kv   4:              bitnet-b1.58.embedding_length u32              = 2560
llama_model_loader: - kv   5:                   bitnet-b1.58.block_count u32              = 30
llama_model_loader: - kv   6:           bitnet-b1.58.feed_forward_length u32              = 6912
llama_model_loader: - kv   7:          bitnet-b1.58.rope.dimension_count u32              = 128
llama_model_loader: - kv   8:          bitnet-b1.58.attention.head_count u32              = 20
llama_model_loader: - kv   9:       bitnet-b1.58.attention.head_count_kv u32              = 5
llama_model_loader: - kv  10:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  11: bitnet-b1.58.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  12:                bitnet-b1.58.rope.freq_base f32              = 500000.000000
llama_model_loader: - kv  13:                          general.file_type u32              = 40
llama_model_loader: - kv  14:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  15:                      tokenizer.ggml.tokens arr[str,128256]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  16:                      tokenizer.ggml.scores arr[f32,128256]  = [0.000000, 0.000000, 0.000000, 0.0000...
llama_model_loader: - kv  17:                  tokenizer.ggml.token_type arr[i32,128256]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  18:                      tokenizer.ggml.merges arr[str,280147]  = ["Ġ Ġ", "Ġ ĠĠĠ", "ĠĠ ĠĠ", "...
llama_model_loader: - kv  19:                tokenizer.ggml.bos_token_id u32              = 128000
llama_model_loader: - kv  20:                tokenizer.ggml.eos_token_id u32              = 128001
llama_model_loader: - kv  21:            tokenizer.ggml.padding_token_id u32              = 128001
llama_model_loader: - kv  22:                    tokenizer.chat_template str              = {% for message in messages %}{% if lo...
llama_model_loader: - kv  23:               general.quantization_version u32              = 2
llama_model_loader: - type  f32:  121 tensors
llama_model_loader: - type  f16:    1 tensors
llama_model_loader: - type i2_s:  210 tensors
llama_model_quantize: failed to quantize: unknown model architecture: 'bitnet-b1.58'
main: failed to quantize model from 'models/ggml-model-i2_s.gguf'
```
@ikawrakow can you help?

---

👤 **saood06** commented the **2025-05-02** at **03:29:42**:<br>

I looked into this, and was able to reproduce and then port the commit that fixes it.

I have made #366 that adds the new name.

I also confirmed that this is only a name change, as I ran gguf-hash.py on both the newly converted gguf based on the updated model and the one I had previously converted available [here](https://huggingface.co/tdh111/bitnet-b1.58-2B-4T-GGUF/tree/main) and the hashes are the same.

---

👤 **usatenko** commented the **2025-05-02** at **10:18:54**:<br>

thank you, it works now

---

👤 **jdluzen** commented the **2025-05-03** at **02:01:15**:<br>

Thanks, those were the changes that I was trying to implement. Glad to know it works for others.
I switched back to Winx64 for now, but it seems my problems could be more than just this. Is the original model supposed to just work out of the box? https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/tree/main
Using a debug build `llama-cli.exe -m ggml-model-i2_s.gguf -p "hi what are you"` I get:
`Assertion failed: ldb >= k, file A:\src\ik_llama.cpp\ggml\src\llamafile\sgemm.cpp, line 856`

---

👤 **ikawrakow** commented the **2025-05-03** at **06:41:56**:<br>

The Microsoft model uses their own quantization type `I2_S`. To use it with `ik_llama.cpp` you need to convert it like this
```
./bin/llama-quantize --allow-requantize $microsoft_model $converted_model iq2_bn
```
This will convert to `IQ2_BN`. If you are running CPU only, you can replace `iq2_bn` with `iq2_bn_r4` (`iq2_bn_r4` uses row-interleaved packing and will give you a better prompt processing performance). If you want to have a smaller model, you can use `iq1_bn` instead. This uses 1.625 bits per weight. PP performance will be lower than `iq2_bn/iq2_bn_4`, but depending on CPU you may get a slightly better token generation speed.