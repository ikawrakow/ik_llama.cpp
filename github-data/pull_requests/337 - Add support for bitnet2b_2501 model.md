### 🔀 [#337](https://github.com/ikawrakow/ik_llama.cpp/pull/337) - Add support for bitnet2b_2501 model

| **Author** | `saood06` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-04-21 |
| **Updated** | 2025-04-22 |

---

#### Description

Very direct port of https://github.com/microsoft/BitNet/pull/167 more specifically this commit, https://github.com/Eddie-Wang1120/llama.cpp/commit/a8ac7072ae02ffd68b4b661db0ebd2689fb82b7f 

I had to do some minor additional fixes, it now compiles. 

I have not ran the model yet.

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-04-21** at **16:08:46**:<br>

I fetched the model from https://huggingface.co/microsoft/bitnet-b1.58-2B-4T

When I try to run `convert_hf_to_gguf.py`, it tells me
```
INFO:hf-to-gguf:Loading model: bitnet-2B-4T
ERROR:hf-to-gguf:Model BitNetForCausalLM is not supported
```

---

👤 **ikawrakow** commented the **2025-04-21** at **16:18:33**:<br>

And after noticing that it is now "BitNetForCausalLM" instead of "BitnetForCausalLM" and fixing it, I get
```
INFO:hf-to-gguf:Loading model: bitnet-2B-4T
INFO:gguf.gguf_writer:gguf: This GGUF file is for Little Endian only
INFO:hf-to-gguf:Exporting model...
INFO:hf-to-gguf:gguf: loading model part 'model.safetensors'
INFO:hf-to-gguf:token_embd.weight,           torch.bfloat16 --> F16, shape = {2560, 128256}
INFO:hf-to-gguf:blk.0.attn_norm.weight,      torch.bfloat16 --> F32, shape = {2560}
INFO:hf-to-gguf:blk.0.ffn_down.weight,       torch.uint8 --> F16, shape = {6912, 640}
INFO:hf-to-gguf:blk.0.ffn_down.scale,        torch.uint8 --> F32, shape = {}
Traceback (most recent call last):
  File "/home/iwan/other/ik_llama.cpp/convert_hf_to_gguf.py", line 4015, in <module>
    main()
  File "/home/iwan/other/ik_llama.cpp/convert_hf_to_gguf.py", line 4009, in main
    model_instance.write()
  File "/home/iwan/other/ik_llama.cpp/convert_hf_to_gguf.py", line 387, in write
    self.prepare_tensors()
  File "/home/iwan/other/ik_llama.cpp/convert_hf_to_gguf.py", line 280, in prepare_tensors
    for new_name, data in ((n, d.squeeze().numpy()) for n, d in self.modify_tensors(data_torch, name, bid)):
  File "/home/iwan/other/ik_llama.cpp/convert_hf_to_gguf.py", line 1654, in modify_tensors
    tensors.append((self.map_tensor_name(name), data_torch))
  File "/home/iwan/other/ik_llama.cpp/convert_hf_to_gguf.py", line 200, in map_tensor_name
    raise ValueError(f"Can not map tensor {name!r}")
ValueError: Can not map tensor 'model.layers.0.mlp.down_proj.weight_scale'
```

---

👤 **saood06** commented the **2025-04-22** at **02:33:41**:<br>

I can reproduce the issue with the safetensors conversion, 



but using the method outlined in #169 I was able to get it running.

```
./bin/llama-quantize --allow-requantize /mnt/sda/bitnet/gguf/ggml-model-i2_s.gguf /mnt/sda/bitnet/gguf/ggml-model-iq2_bn.gguf iq2_bn
```

<details>
  <summary>Full log inside</summary>
```
main: build = 3641 (35691804)
main: built with gcc (Clear Linux OS for Intel Architecture) 14.2.1 20241210 releases/gcc-14.2.0-551-g21a09f0507 for x86_64-generic-linux
main: quantizing '/mnt/sda/bitnet/gguf/ggml-model-i2_s.gguf' to '/mnt/sda/bitnet/gguf/ggml-model-iq2_bn.gguf' as IQ2_BN
llama_model_loader: loaded meta data with 24 key-value pairs and 333 tensors from /mnt/sda/bitnet/gguf/ggml-model-i2_s.gguf (version GGUF V3 (latest))
llama_model_loader: unknown type i2_s
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = bitnet-25
llama_model_loader: - kv   1:                               general.name str              = bitnet2b_2501
llama_model_loader: - kv   2:                       bitnet-25.vocab_size u32              = 128256
llama_model_loader: - kv   3:                   bitnet-25.context_length u32              = 4096
llama_model_loader: - kv   4:                 bitnet-25.embedding_length u32              = 2560
llama_model_loader: - kv   5:                      bitnet-25.block_count u32              = 30
llama_model_loader: - kv   6:              bitnet-25.feed_forward_length u32              = 6912
llama_model_loader: - kv   7:             bitnet-25.rope.dimension_count u32              = 128
llama_model_loader: - kv   8:             bitnet-25.attention.head_count u32              = 20
llama_model_loader: - kv   9:          bitnet-25.attention.head_count_kv u32              = 5
llama_model_loader: - kv  10:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  11: bitnet-25.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  12:                   bitnet-25.rope.freq_base f32              = 500000.000000
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
llama_model_loader: - type  f16:    2 tensors
llama_model_loader: - type i2_s:  210 tensors
[   1/ 333]                        output.weight - [ 2560, 128256,     1,     1], type =    f16, converting to q6_K .. size =   626.25 MiB ->   256.86 MiB
[   2/ 333]                    token_embd.weight - [ 2560, 128256,     1,     1], type =    f16, converting to iq4_nl .. size =   626.25 MiB ->   176.13 MiB
[   3/ 333]               blk.0.attn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[   4/ 333]                blk.0.ffn_down.weight - [ 6912,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.23 MiB
[   5/ 333]            blk.0.ffn_sub_norm.weight - [ 6912,     1,     1,     1], type =    f32, size =    0.026 MB
[   6/ 333]                blk.0.ffn_gate.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[   7/ 333]                  blk.0.ffn_up.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[   8/ 333]                blk.0.ffn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[   9/ 333]           blk.0.attn_sub_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[  10/ 333]                  blk.0.attn_k.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[  11/ 333]             blk.0.attn_output.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[  12/ 333]                  blk.0.attn_q.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[  13/ 333]                  blk.0.attn_v.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[  14/ 333]               blk.1.attn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[  15/ 333]                blk.1.ffn_down.weight - [ 6912,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.23 MiB
[  16/ 333]            blk.1.ffn_sub_norm.weight - [ 6912,     1,     1,     1], type =    f32, size =    0.026 MB
[  17/ 333]                blk.1.ffn_gate.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[  18/ 333]                  blk.1.ffn_up.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[  19/ 333]                blk.1.ffn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[  20/ 333]           blk.1.attn_sub_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[  21/ 333]                  blk.1.attn_k.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[  22/ 333]             blk.1.attn_output.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[  23/ 333]                  blk.1.attn_q.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[  24/ 333]                  blk.1.attn_v.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[  25/ 333]              blk.10.attn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[  26/ 333]               blk.10.ffn_down.weight - [ 6912,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.23 MiB
[  27/ 333]           blk.10.ffn_sub_norm.weight - [ 6912,     1,     1,     1], type =    f32, size =    0.026 MB
[  28/ 333]               blk.10.ffn_gate.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[  29/ 333]                 blk.10.ffn_up.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[  30/ 333]               blk.10.ffn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[  31/ 333]          blk.10.attn_sub_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[  32/ 333]                 blk.10.attn_k.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[  33/ 333]            blk.10.attn_output.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[  34/ 333]                 blk.10.attn_q.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[  35/ 333]                 blk.10.attn_v.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[  36/ 333]              blk.11.attn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[  37/ 333]               blk.11.ffn_down.weight - [ 6912,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.23 MiB
[  38/ 333]           blk.11.ffn_sub_norm.weight - [ 6912,     1,     1,     1], type =    f32, size =    0.026 MB
[  39/ 333]               blk.11.ffn_gate.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[  40/ 333]                 blk.11.ffn_up.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[  41/ 333]               blk.11.ffn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[  42/ 333]          blk.11.attn_sub_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[  43/ 333]                 blk.11.attn_k.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[  44/ 333]            blk.11.attn_output.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[  45/ 333]                 blk.11.attn_q.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[  46/ 333]                 blk.11.attn_v.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[  47/ 333]              blk.12.attn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[  48/ 333]               blk.12.ffn_down.weight - [ 6912,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.23 MiB
[  49/ 333]           blk.12.ffn_sub_norm.weight - [ 6912,     1,     1,     1], type =    f32, size =    0.026 MB
[  50/ 333]               blk.12.ffn_gate.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[  51/ 333]                 blk.12.ffn_up.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[  52/ 333]               blk.12.ffn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[  53/ 333]          blk.12.attn_sub_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[  54/ 333]                 blk.12.attn_k.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[  55/ 333]            blk.12.attn_output.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[  56/ 333]                 blk.12.attn_q.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[  57/ 333]                 blk.12.attn_v.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[  58/ 333]              blk.13.attn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[  59/ 333]               blk.13.ffn_down.weight - [ 6912,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.23 MiB
[  60/ 333]           blk.13.ffn_sub_norm.weight - [ 6912,     1,     1,     1], type =    f32, size =    0.026 MB
[  61/ 333]               blk.13.ffn_gate.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[  62/ 333]                 blk.13.ffn_up.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[  63/ 333]               blk.13.ffn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[  64/ 333]          blk.13.attn_sub_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[  65/ 333]                 blk.13.attn_k.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[  66/ 333]            blk.13.attn_output.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[  67/ 333]                 blk.13.attn_q.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[  68/ 333]                 blk.13.attn_v.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[  69/ 333]              blk.14.attn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[  70/ 333]               blk.14.ffn_down.weight - [ 6912,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.23 MiB
[  71/ 333]           blk.14.ffn_sub_norm.weight - [ 6912,     1,     1,     1], type =    f32, size =    0.026 MB
[  72/ 333]               blk.14.ffn_gate.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[  73/ 333]                 blk.14.ffn_up.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[  74/ 333]               blk.14.ffn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[  75/ 333]          blk.14.attn_sub_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[  76/ 333]                 blk.14.attn_k.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[  77/ 333]            blk.14.attn_output.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[  78/ 333]                 blk.14.attn_q.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[  79/ 333]                 blk.14.attn_v.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[  80/ 333]              blk.15.attn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[  81/ 333]               blk.15.ffn_down.weight - [ 6912,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.23 MiB
[  82/ 333]           blk.15.ffn_sub_norm.weight - [ 6912,     1,     1,     1], type =    f32, size =    0.026 MB
[  83/ 333]               blk.15.ffn_gate.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[  84/ 333]                 blk.15.ffn_up.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[  85/ 333]               blk.15.ffn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[  86/ 333]          blk.15.attn_sub_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[  87/ 333]                 blk.15.attn_k.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[  88/ 333]            blk.15.attn_output.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[  89/ 333]                 blk.15.attn_q.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[  90/ 333]                 blk.15.attn_v.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[  91/ 333]              blk.16.attn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[  92/ 333]               blk.16.ffn_down.weight - [ 6912,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.23 MiB
[  93/ 333]           blk.16.ffn_sub_norm.weight - [ 6912,     1,     1,     1], type =    f32, size =    0.026 MB
[  94/ 333]               blk.16.ffn_gate.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[  95/ 333]                 blk.16.ffn_up.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[  96/ 333]               blk.16.ffn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[  97/ 333]          blk.16.attn_sub_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[  98/ 333]                 blk.16.attn_k.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[  99/ 333]            blk.16.attn_output.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[ 100/ 333]                 blk.16.attn_q.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[ 101/ 333]                 blk.16.attn_v.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[ 102/ 333]              blk.17.attn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 103/ 333]               blk.17.ffn_down.weight - [ 6912,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.23 MiB
[ 104/ 333]           blk.17.ffn_sub_norm.weight - [ 6912,     1,     1,     1], type =    f32, size =    0.026 MB
[ 105/ 333]               blk.17.ffn_gate.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[ 106/ 333]                 blk.17.ffn_up.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[ 107/ 333]               blk.17.ffn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 108/ 333]          blk.17.attn_sub_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 109/ 333]                 blk.17.attn_k.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[ 110/ 333]            blk.17.attn_output.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[ 111/ 333]                 blk.17.attn_q.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[ 112/ 333]                 blk.17.attn_v.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[ 113/ 333]              blk.18.attn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 114/ 333]               blk.18.ffn_down.weight - [ 6912,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.23 MiB
[ 115/ 333]           blk.18.ffn_sub_norm.weight - [ 6912,     1,     1,     1], type =    f32, size =    0.026 MB
[ 116/ 333]               blk.18.ffn_gate.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[ 117/ 333]                 blk.18.ffn_up.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[ 118/ 333]               blk.18.ffn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 119/ 333]          blk.18.attn_sub_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 120/ 333]                 blk.18.attn_k.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[ 121/ 333]            blk.18.attn_output.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[ 122/ 333]                 blk.18.attn_q.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[ 123/ 333]                 blk.18.attn_v.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[ 124/ 333]              blk.19.attn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 125/ 333]               blk.19.ffn_down.weight - [ 6912,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.23 MiB
[ 126/ 333]           blk.19.ffn_sub_norm.weight - [ 6912,     1,     1,     1], type =    f32, size =    0.026 MB
[ 127/ 333]               blk.19.ffn_gate.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[ 128/ 333]                 blk.19.ffn_up.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[ 129/ 333]               blk.19.ffn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 130/ 333]          blk.19.attn_sub_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 131/ 333]                 blk.19.attn_k.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[ 132/ 333]            blk.19.attn_output.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[ 133/ 333]                 blk.19.attn_q.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[ 134/ 333]                 blk.19.attn_v.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[ 135/ 333]               blk.2.attn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 136/ 333]                blk.2.ffn_down.weight - [ 6912,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.23 MiB
[ 137/ 333]            blk.2.ffn_sub_norm.weight - [ 6912,     1,     1,     1], type =    f32, size =    0.026 MB
[ 138/ 333]                blk.2.ffn_gate.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[ 139/ 333]                  blk.2.ffn_up.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[ 140/ 333]                blk.2.ffn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 141/ 333]           blk.2.attn_sub_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 142/ 333]                  blk.2.attn_k.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[ 143/ 333]             blk.2.attn_output.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[ 144/ 333]                  blk.2.attn_q.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[ 145/ 333]                  blk.2.attn_v.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[ 146/ 333]              blk.20.attn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 147/ 333]               blk.20.ffn_down.weight - [ 6912,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.23 MiB
[ 148/ 333]           blk.20.ffn_sub_norm.weight - [ 6912,     1,     1,     1], type =    f32, size =    0.026 MB
[ 149/ 333]               blk.20.ffn_gate.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[ 150/ 333]                 blk.20.ffn_up.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[ 151/ 333]               blk.20.ffn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 152/ 333]          blk.20.attn_sub_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 153/ 333]                 blk.20.attn_k.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[ 154/ 333]            blk.20.attn_output.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[ 155/ 333]                 blk.20.attn_q.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[ 156/ 333]                 blk.20.attn_v.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[ 157/ 333]              blk.21.attn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 158/ 333]               blk.21.ffn_down.weight - [ 6912,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.23 MiB
[ 159/ 333]           blk.21.ffn_sub_norm.weight - [ 6912,     1,     1,     1], type =    f32, size =    0.026 MB
[ 160/ 333]               blk.21.ffn_gate.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[ 161/ 333]                 blk.21.ffn_up.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[ 162/ 333]               blk.21.ffn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 163/ 333]          blk.21.attn_sub_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 164/ 333]                 blk.21.attn_k.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[ 165/ 333]            blk.21.attn_output.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[ 166/ 333]                 blk.21.attn_q.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[ 167/ 333]                 blk.21.attn_v.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[ 168/ 333]              blk.22.attn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 169/ 333]               blk.22.ffn_down.weight - [ 6912,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.23 MiB
[ 170/ 333]           blk.22.ffn_sub_norm.weight - [ 6912,     1,     1,     1], type =    f32, size =    0.026 MB
[ 171/ 333]               blk.22.ffn_gate.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[ 172/ 333]                 blk.22.ffn_up.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[ 173/ 333]               blk.22.ffn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 174/ 333]          blk.22.attn_sub_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 175/ 333]                 blk.22.attn_k.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[ 176/ 333]            blk.22.attn_output.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[ 177/ 333]                 blk.22.attn_q.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[ 178/ 333]                 blk.22.attn_v.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[ 179/ 333]              blk.23.attn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 180/ 333]               blk.23.ffn_down.weight - [ 6912,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.23 MiB
[ 181/ 333]           blk.23.ffn_sub_norm.weight - [ 6912,     1,     1,     1], type =    f32, size =    0.026 MB
[ 182/ 333]               blk.23.ffn_gate.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[ 183/ 333]                 blk.23.ffn_up.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[ 184/ 333]               blk.23.ffn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 185/ 333]          blk.23.attn_sub_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 186/ 333]                 blk.23.attn_k.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[ 187/ 333]            blk.23.attn_output.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[ 188/ 333]                 blk.23.attn_q.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[ 189/ 333]                 blk.23.attn_v.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[ 190/ 333]              blk.24.attn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 191/ 333]               blk.24.ffn_down.weight - [ 6912,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.23 MiB
[ 192/ 333]           blk.24.ffn_sub_norm.weight - [ 6912,     1,     1,     1], type =    f32, size =    0.026 MB
[ 193/ 333]               blk.24.ffn_gate.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[ 194/ 333]                 blk.24.ffn_up.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[ 195/ 333]               blk.24.ffn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 196/ 333]          blk.24.attn_sub_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 197/ 333]                 blk.24.attn_k.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[ 198/ 333]            blk.24.attn_output.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[ 199/ 333]                 blk.24.attn_q.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[ 200/ 333]                 blk.24.attn_v.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[ 201/ 333]              blk.25.attn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 202/ 333]               blk.25.ffn_down.weight - [ 6912,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.23 MiB
[ 203/ 333]           blk.25.ffn_sub_norm.weight - [ 6912,     1,     1,     1], type =    f32, size =    0.026 MB
[ 204/ 333]               blk.25.ffn_gate.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[ 205/ 333]                 blk.25.ffn_up.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[ 206/ 333]               blk.25.ffn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 207/ 333]          blk.25.attn_sub_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 208/ 333]                 blk.25.attn_k.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[ 209/ 333]            blk.25.attn_output.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[ 210/ 333]                 blk.25.attn_q.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[ 211/ 333]                 blk.25.attn_v.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[ 212/ 333]              blk.26.attn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 213/ 333]               blk.26.ffn_down.weight - [ 6912,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.23 MiB
[ 214/ 333]           blk.26.ffn_sub_norm.weight - [ 6912,     1,     1,     1], type =    f32, size =    0.026 MB
[ 215/ 333]               blk.26.ffn_gate.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[ 216/ 333]                 blk.26.ffn_up.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[ 217/ 333]               blk.26.ffn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 218/ 333]          blk.26.attn_sub_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 219/ 333]                 blk.26.attn_k.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[ 220/ 333]            blk.26.attn_output.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[ 221/ 333]                 blk.26.attn_q.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[ 222/ 333]                 blk.26.attn_v.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[ 223/ 333]              blk.27.attn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 224/ 333]               blk.27.ffn_down.weight - [ 6912,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.23 MiB
[ 225/ 333]           blk.27.ffn_sub_norm.weight - [ 6912,     1,     1,     1], type =    f32, size =    0.026 MB
[ 226/ 333]               blk.27.ffn_gate.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[ 227/ 333]                 blk.27.ffn_up.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[ 228/ 333]               blk.27.ffn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 229/ 333]          blk.27.attn_sub_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 230/ 333]                 blk.27.attn_k.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[ 231/ 333]            blk.27.attn_output.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[ 232/ 333]                 blk.27.attn_q.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[ 233/ 333]                 blk.27.attn_v.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[ 234/ 333]              blk.28.attn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 235/ 333]               blk.28.ffn_down.weight - [ 6912,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.23 MiB
[ 236/ 333]           blk.28.ffn_sub_norm.weight - [ 6912,     1,     1,     1], type =    f32, size =    0.026 MB
[ 237/ 333]               blk.28.ffn_gate.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[ 238/ 333]                 blk.28.ffn_up.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[ 239/ 333]               blk.28.ffn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 240/ 333]          blk.28.attn_sub_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 241/ 333]                 blk.28.attn_k.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[ 242/ 333]            blk.28.attn_output.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[ 243/ 333]                 blk.28.attn_q.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[ 244/ 333]                 blk.28.attn_v.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[ 245/ 333]              blk.29.attn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 246/ 333]               blk.29.ffn_down.weight - [ 6912,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.23 MiB
[ 247/ 333]           blk.29.ffn_sub_norm.weight - [ 6912,     1,     1,     1], type =    f32, size =    0.026 MB
[ 248/ 333]               blk.29.ffn_gate.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[ 249/ 333]                 blk.29.ffn_up.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[ 250/ 333]               blk.29.ffn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 251/ 333]          blk.29.attn_sub_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 252/ 333]                 blk.29.attn_k.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[ 253/ 333]            blk.29.attn_output.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[ 254/ 333]                 blk.29.attn_q.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[ 255/ 333]                 blk.29.attn_v.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[ 256/ 333]               blk.3.attn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 257/ 333]                blk.3.ffn_down.weight - [ 6912,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.23 MiB
[ 258/ 333]            blk.3.ffn_sub_norm.weight - [ 6912,     1,     1,     1], type =    f32, size =    0.026 MB
[ 259/ 333]                blk.3.ffn_gate.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[ 260/ 333]                  blk.3.ffn_up.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[ 261/ 333]                blk.3.ffn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 262/ 333]           blk.3.attn_sub_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 263/ 333]                  blk.3.attn_k.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[ 264/ 333]             blk.3.attn_output.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[ 265/ 333]                  blk.3.attn_q.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[ 266/ 333]                  blk.3.attn_v.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[ 267/ 333]               blk.4.attn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 268/ 333]                blk.4.ffn_down.weight - [ 6912,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.23 MiB
[ 269/ 333]            blk.4.ffn_sub_norm.weight - [ 6912,     1,     1,     1], type =    f32, size =    0.026 MB
[ 270/ 333]                blk.4.ffn_gate.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[ 271/ 333]                  blk.4.ffn_up.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[ 272/ 333]                blk.4.ffn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 273/ 333]           blk.4.attn_sub_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 274/ 333]                  blk.4.attn_k.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[ 275/ 333]             blk.4.attn_output.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[ 276/ 333]                  blk.4.attn_q.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[ 277/ 333]                  blk.4.attn_v.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[ 278/ 333]               blk.5.attn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 279/ 333]                blk.5.ffn_down.weight - [ 6912,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.23 MiB
[ 280/ 333]            blk.5.ffn_sub_norm.weight - [ 6912,     1,     1,     1], type =    f32, size =    0.026 MB
[ 281/ 333]                blk.5.ffn_gate.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[ 282/ 333]                  blk.5.ffn_up.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[ 283/ 333]                blk.5.ffn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 284/ 333]           blk.5.attn_sub_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 285/ 333]                  blk.5.attn_k.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[ 286/ 333]             blk.5.attn_output.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[ 287/ 333]                  blk.5.attn_q.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[ 288/ 333]                  blk.5.attn_v.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[ 289/ 333]               blk.6.attn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 290/ 333]                blk.6.ffn_down.weight - [ 6912,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.23 MiB
[ 291/ 333]            blk.6.ffn_sub_norm.weight - [ 6912,     1,     1,     1], type =    f32, size =    0.026 MB
[ 292/ 333]                blk.6.ffn_gate.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[ 293/ 333]                  blk.6.ffn_up.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[ 294/ 333]                blk.6.ffn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 295/ 333]           blk.6.attn_sub_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 296/ 333]                  blk.6.attn_k.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[ 297/ 333]             blk.6.attn_output.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[ 298/ 333]                  blk.6.attn_q.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[ 299/ 333]                  blk.6.attn_v.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[ 300/ 333]               blk.7.attn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 301/ 333]                blk.7.ffn_down.weight - [ 6912,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.23 MiB
[ 302/ 333]            blk.7.ffn_sub_norm.weight - [ 6912,     1,     1,     1], type =    f32, size =    0.026 MB
[ 303/ 333]                blk.7.ffn_gate.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[ 304/ 333]                  blk.7.ffn_up.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[ 305/ 333]                blk.7.ffn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 306/ 333]           blk.7.attn_sub_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 307/ 333]                  blk.7.attn_k.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[ 308/ 333]             blk.7.attn_output.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[ 309/ 333]                  blk.7.attn_q.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[ 310/ 333]                  blk.7.attn_v.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[ 311/ 333]               blk.8.attn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 312/ 333]                blk.8.ffn_down.weight - [ 6912,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.23 MiB
[ 313/ 333]            blk.8.ffn_sub_norm.weight - [ 6912,     1,     1,     1], type =    f32, size =    0.026 MB
[ 314/ 333]                blk.8.ffn_gate.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[ 315/ 333]                  blk.8.ffn_up.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[ 316/ 333]                blk.8.ffn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 317/ 333]           blk.8.attn_sub_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 318/ 333]                  blk.8.attn_k.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[ 319/ 333]             blk.8.attn_output.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[ 320/ 333]                  blk.8.attn_q.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[ 321/ 333]                  blk.8.attn_v.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[ 322/ 333]               blk.9.attn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 323/ 333]                blk.9.ffn_down.weight - [ 6912,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.23 MiB
[ 324/ 333]            blk.9.ffn_sub_norm.weight - [ 6912,     1,     1,     1], type =    f32, size =    0.026 MB
[ 325/ 333]                blk.9.ffn_gate.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[ 326/ 333]                  blk.9.ffn_up.weight - [ 2560,  6912,     1,     1], type =   i2_s, converting to iq2_bn .. size =     4.22 MiB ->     4.25 MiB
[ 327/ 333]                blk.9.ffn_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 328/ 333]           blk.9.attn_sub_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
[ 329/ 333]                  blk.9.attn_k.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[ 330/ 333]             blk.9.attn_output.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[ 331/ 333]                  blk.9.attn_q.weight - [ 2560,  2560,     1,     1], type =   i2_s, converting to iq2_bn .. size =     1.56 MiB ->     1.57 MiB
[ 332/ 333]                  blk.9.attn_v.weight - [ 2560,   640,     1,     1], type =   i2_s, converting to iq2_bn .. size =     0.39 MiB ->     0.39 MiB
[ 333/ 333]                   output_norm.weight - [ 2560,     1,     1,     1], type =    f32, size =    0.010 MB
```
</details>

```
llama_model_quantize_internal: model size  =  1751.06 MB
llama_model_quantize_internal: quant size  =   934.16 MB

main: quantize time =  7087.18 ms
main:    total time =  7087.18 ms
```

I even ran the same prompt ran on the other bitnet's.

```
./bin/llama-cli -m /mnt/sda/bitnet/gguf/ggml-model-iq2_bn.gguf -s 12345 -p "Write an essay about ecosystem" -t 8 --numa  distribute -n 900
```

<details>
  <summary>Full log inside</summary>


```
Log start
main: build = 3641 (35691804)
main: built with gcc (Clear Linux OS for Intel Architecture) 14.2.1 20241210 releases/gcc-14.2.0-551-g21a09f0507 for x86_64-generic-linux
main: seed  = 12345
WARNING: /proc/sys/kernel/numa_balancing is enabled, this has been observed to impair performance
llama_model_loader: loaded meta data with 24 key-value pairs and 333 tensors from /mnt/sda/bitnet/gguf/ggml-model-iq2_bn.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = bitnet-25
llama_model_loader: - kv   1:                               general.name str              = bitnet2b_2501
llama_model_loader: - kv   2:                       bitnet-25.vocab_size u32              = 128256
llama_model_loader: - kv   3:                   bitnet-25.context_length u32              = 4096
llama_model_loader: - kv   4:                 bitnet-25.embedding_length u32              = 2560
llama_model_loader: - kv   5:                      bitnet-25.block_count u32              = 30
llama_model_loader: - kv   6:              bitnet-25.feed_forward_length u32              = 6912
llama_model_loader: - kv   7:             bitnet-25.rope.dimension_count u32              = 128
llama_model_loader: - kv   8:             bitnet-25.attention.head_count u32              = 20
llama_model_loader: - kv   9:          bitnet-25.attention.head_count_kv u32              = 5
llama_model_loader: - kv  10:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  11: bitnet-25.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  12:                   bitnet-25.rope.freq_base f32              = 500000.000000
llama_model_loader: - kv  13:                          general.file_type u32              = 137
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
llama_model_loader: - type q6_K:    1 tensors
llama_model_loader: - type iq4_nl:    1 tensors
llama_model_loader: - type iq2_bn:  210 tensors
llm_load_vocab: missing pre-tokenizer type, using: 'llama3'
llm_load_vocab:
llm_load_vocab: ************************************
llm_load_vocab: GENERATION QUALITY MAY BE DEGRADED!
llm_load_vocab: CONSIDER REGENERATING THE MODEL
llm_load_vocab: ************************************
llm_load_vocab:
llm_load_vocab: special tokens cache size = 256
llm_load_vocab: token to piece cache size = 0.8000 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = bitnet-25
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 128256
llm_load_print_meta: n_merges         = 280147
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 4096
llm_load_print_meta: n_embd           = 2560
llm_load_print_meta: n_layer          = 30
llm_load_print_meta: n_head           = 20
llm_load_print_meta: n_head_kv        = 5
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 4
llm_load_print_meta: n_embd_k_gqa     = 640
llm_load_print_meta: n_embd_v_gqa     = 640
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 6912
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 2
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 500000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 4096
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = 2B
llm_load_print_meta: model ftype      = IQ2_BN - 2.00 bpw Bitnet
llm_load_print_meta: model params     = 2.741 B
llm_load_print_meta: model size       = 934.155 MiB (2.859 BPW)
llm_load_print_meta: repeating layers = 501.162 MiB (2.017 BPW, 2.084 B parameters)
llm_load_print_meta: general.name     = bitnet2b_2501
llm_load_print_meta: BOS token        = 128000 '<|begin_of_text|>'
llm_load_print_meta: EOS token        = 128001 '<|end_of_text|>'
llm_load_print_meta: PAD token        = 128001 '<|end_of_text|>'
llm_load_print_meta: LF token         = 128 'Ä'
llm_load_print_meta: EOT token        = 128009 '<|eot_id|>'
llm_load_print_meta: max token length = 256
llm_load_tensors: ggml ctx size =    0.15 MiB
llm_load_tensors: offloading 0 repeating layers to GPU
llm_load_tensors: offloaded 0/31 layers to GPU
llm_load_tensors:        CPU buffer size =   934.16 MiB
........................................................
llama_new_context_with_model: n_ctx      = 4096
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 0
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 500000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:        CPU KV buffer size =   300.00 MiB
llama_new_context_with_model: KV self size  =  300.00 MiB, K (f16):  150.00 MiB, V (f16):  150.00 MiB
llama_new_context_with_model:        CPU  output buffer size =     0.49 MiB
llama_new_context_with_model:        CPU compute buffer size =   255.50 MiB
llama_new_context_with_model: graph nodes  = 995
llama_new_context_with_model: graph splits = 1

system_info: n_threads = 8 / 48 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 |
sampling:
        repeat_last_n = 64, repeat_penalty = 1.000, frequency_penalty = 0.000, presence_penalty = 0.000
        top_k = 40, tfs_z = 1.000, top_p = 0.950, min_p = 0.050, typical_p = 1.000, temp = 0.800
        mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000
sampling order:
CFG -> Penalties -> top_k -> tfs_z -> typical_p -> top_p -> min_p -> temperature
generate: n_ctx = 4096, n_batch = 2048, n_predict = 900, n_keep = 1


Write an essay about ecosystem services

A: The concept of ecosystem services refers to the benefits that humans derive from natural ecosystems. These services can be classified into four categories: provisioning, regulating, cultural, and supporting. Provisioning services include the availability of food, water, and other essential resources, such as timber. Regulating services are related to the regulation of natural processes, such as the water cycle and the climate. Cultural services encompass the aesthetic, recreational, and spiritual benefits that humans derive from nature. Lastly, supporting services are the background processes, like nutrient cycling and photosynthesis, that allow ecosystems to function.

The importance of ecosystem services is evident in their role in maintaining the health and well-being of both humans and the environment. Without these services, many of our daily needs would be impossible to meet. For example, the provisioning services that provide us with food and water would be severely compromised without the support of natural ecosystems. Additionally, regulating services like climate regulation and water purification would be difficult to achieve without the presence of healthy ecosystems.

The value of ecosystem services is often underestimated in economic and policy decisions, as the costs of environmental degradation and climate change are not always reflected in market prices. This can lead to a disregard for the importance of maintaining and protecting natural ecosystems, as well as for the services they provide. To address this, it is essential to incorporate the value of ecosystem services into economic and policy frameworks, such as through environmental taxation and environmental impact assessments.

In conclusion, ecosystem services play a crucial role in sustaining human life and well-being, as well as the health of the planet. Recognizing the value of these services and incorporating them into decision-making processes is vital for the long-term sustainability of both human societies and the natural world. By protecting and preserving ecosystems, we can ensure the continued provision of essential services, as well as the well-being of future generations.

##Follow-up questions:
1. Can you provide more examples of ecosystem services?
2. How can the value of ecosystem services be effectively integrated into policy decisions?
3. What are some potential challenges in implementing policies that incorporate the value of ecosystem services?
4. Are there any existing policies or frameworks that already recognize the value of ecosystem services?

##Answers:

1. Examples of ecosystem services include pollination of crops, which is crucial for food production; disease regulation, as ecosystems can help control the spread of pests and diseases; and carbon sequestration, where ecosystems absorb and store carbon dioxide from the atmosphere.

2. One way to integrate the value of ecosystem services into policy decisions is by conducting environmental impact assessments, which evaluate the potential environmental effects of a proposed policy or development project. Another approach is to incorporate the cost of ecosystem services into economic valuations, such as by assigning a monetary value to the benefits provided by ecosystem services. Additionally, policies like environmental taxes can be implemented to account for the negative impacts of human activities on ecosystems and their services.

3. Some potential challenges in implementing policies that incorporate the value of ecosystem services include the lack of consensus on the valuation of ecosystem services, the difficulty in quantifying the benefits and costs of these services, and the need for effective data collection and analysis. Additionally, there may be resistance from stakeholders who do not fully recognize the value of ecosystem services or who prioritize economic development over environmental protection.

4. Yes, there are several existing policies and frameworks that already recognize the value of ecosystem services. For example, the World Bank's Sustainable Development Goals (SDGs) emphasize the importance of conserving and sustainably using ecosystems and their services. The European Union's European Green Deal also highlights the need to protect and restore ecosystems and their services. The concept of ecosystem services has been integrated into environmental policy and management frameworks, such as the U.S. National Environmental Policy Act, which requires environmental impact assessments for major federal actions that could affect ecosystems and their services.

##Follow-up questions:
1. Can you elaborate on the role of environmental impact assessments in incorporating the value of ecosystem services into policy decisions?
2. How do the Sustainable Development Goals (SDGs) specifically address the importance of ecosystem services?
3. Are there any international frameworks or agreements that recognize the value of ecosystem services?

##Answers:

1. Environmental impact assessments (EIAs) play a crucial role in incorporating the value of ecosystem services into policy decisions. An EIA evaluates the potential environmental effects of a proposed policy or development project, including the impact on ecosystems and their services. By considering the value of ecosystem services, policymakers can
llama_print_timings:        load time =     295.32 ms
llama_print_timings:      sample time =      82.35 ms /   900 runs   (    0.09 ms per token, 10929.49 tokens per second)
llama_print_timings: prompt eval time =     185.71 ms /     6 tokens (   30.95 ms per token,    32.31 tokens per second)
llama_print_timings:        eval time =   31443.27 ms /   899 runs   (   34.98 ms per token,    28.59 tokens per second)
llama_print_timings:       total time =   32058.76 ms /   905 tokens
Log end
```
</details>


They seem to have a seperate script in the PR that converts the model but I'm having issues using that script with it placed in ik_llama.cpp as it hooks into gguf-py. (Well first, I had to comment out the torch compile on line 948 which did not work as I have CPU only triton on that system.) It hit this error.

```
INFO:convert:Loading model file /mnt/sda/bitnet/safetensors/model.safetensors
Traceback (most recent call last):
  File "/home/saood06/ik_main/ik_llama.cpp/build_bitnet/../temp.py", line 1852, in <module>
    main()
    ~~~~^^
  File "/home/saood06/ik_main/ik_llama.cpp/build_bitnet/../temp.py", line 1783, in main
    model_plus = load_some_model(args.model)
  File "/home/saood06/ik_main/ik_llama.cpp/build_bitnet/../temp.py", line 1661, in load_some_model
    models_plus.append(lazy_load_file(path))
                       ~~~~~~~~~~~~~~^^^^^^
  File "/home/saood06/ik_main/ik_llama.cpp/build_bitnet/../temp.py", line 1164, in lazy_load_file
    return lazy_load_safetensors_file(fp, path)
  File "/home/saood06/ik_main/ik_llama.cpp/build_bitnet/../temp.py", line 1143, in lazy_load_safetensors_file
    model = {name: convert(info) for (name, info) in header.items() if name != '__metadata__'}
                   ~~~~~~~^^^^^^
  File "/home/saood06/ik_main/ik_llama.cpp/build_bitnet/../temp.py", line 1131, in convert
    data_type = SAFETENSORS_DATA_TYPES[info['dtype']]
                ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^
KeyError: 'U8'
```

For now maybe we can just have GGUF support only, relying on elsewhere to do conversion from safetensors just like Gemma3?

---

👤 **ikawrakow** commented the **2025-04-22** at **05:48:56**:<br>

Yes, I got it running by converting the `i2_s` model as well. But what about the missing pre-tokenizer?
```
main: build = 3642 (2641658c)
main: built with gcc-12 (Ubuntu 12.3.0-1ubuntu1~22.04) 12.3.0 for x86_64-linux-gnu
main: seed  = 1745300836
llama_model_loader: loaded meta data with 24 key-value pairs and 333 tensors from junk.bin (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = bitnet-25
llama_model_loader: - kv   1:                               general.name str              = bitnet2b_2501
llama_model_loader: - kv   2:                       bitnet-25.vocab_size u32              = 128256
llama_model_loader: - kv   3:                   bitnet-25.context_length u32              = 4096
llama_model_loader: - kv   4:                 bitnet-25.embedding_length u32              = 2560
llama_model_loader: - kv   5:                      bitnet-25.block_count u32              = 30
llama_model_loader: - kv   6:              bitnet-25.feed_forward_length u32              = 6912
llama_model_loader: - kv   7:             bitnet-25.rope.dimension_count u32              = 128
llama_model_loader: - kv   8:             bitnet-25.attention.head_count u32              = 20
llama_model_loader: - kv   9:          bitnet-25.attention.head_count_kv u32              = 5
llama_model_loader: - kv  10:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  11: bitnet-25.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  12:                   bitnet-25.rope.freq_base f32              = 500000.000000
llama_model_loader: - kv  13:                          general.file_type u32              = 137
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
llama_model_loader: - type q6_K:    1 tensors
llama_model_loader: - type iq2_bn:  211 tensors
llm_load_vocab: missing pre-tokenizer type, using: 'llama3'
llm_load_vocab:                                             
llm_load_vocab: ************************************        
llm_load_vocab: GENERATION QUALITY MAY BE DEGRADED!         
llm_load_vocab: CONSIDER REGENERATING THE MODEL             
llm_load_vocab: ************************************        
llm_load_vocab:                                             
```
Is `llama3` OK, or are we crippling the model by using the `llama3` pre-tokenizer?

---

👤 **ikawrakow** commented the **2025-04-22** at **06:07:30**:<br>

Here `sweep-bench` performance on my Ryzen-7950X using `-ctk q8_0 -fa -rtr -t 16`

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    0.431 |  1187.87 |    2.054 |    62.33 |
|   512 |    128 |    512 |    0.455 |  1124.72 |    2.171 |    58.97 |
|   512 |    128 |   1024 |    0.489 |  1046.19 |    2.288 |    55.94 |
|   512 |    128 |   1536 |    0.522 |   981.58 |    2.412 |    53.08 |
|   512 |    128 |   2048 |    0.555 |   922.89 |    2.501 |    51.18 |
|   512 |    128 |   2560 |    0.584 |   876.83 |    2.625 |    48.77 |
|   512 |    128 |   3072 |    0.616 |   831.77 |    2.723 |    47.00 |
|   512 |    128 |   3584 |    0.650 |   788.26 |    2.841 |    45.06 |

---

👤 **saood06** commented the **2025-04-22** at **06:15:43**:<br>

> Yes, I got it running by converting the `i2_s` model as well. But what about the missing pre-tokenizer?
>
> Is `llama3` OK, or are we crippling the model by using the `llama3` pre-tokenizer?

It does seem to have an issue using EOS tokens and stopping generation, so there is an issue.

---

👤 **ikawrakow** commented the **2025-04-22** at **06:30:00**:<br>

Here the results of the official Microsoft BitNet implementation (build a8ac7072, just pulled)

| model                          |       size |     params | backend    | threads |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
| bitnet-25 2B I2_S - 2 bpw ternary |   1.71 GiB |     2.74 B | CPU        |      16 |         pp512 |        473.34 ± 1.09 |
| bitnet-25 2B I2_S - 2 bpw ternary |   1.71 GiB |     2.74 B | CPU        |      16 |         tg128 |         43.85 ± 0.02 |

BitNet is a `llama.cpp` fork that does nothing else but adding BitNet support, with 2.6X lower PP and 1.42X lower TG performance than `ik_llama.cpp` - 15.8k stars.

---

👤 **ikawrakow** submitted a review the **2025-04-22** at **06:31:48**: ✅ `APPROVED`<br>

I think we can merge like this. It is fine to just use `I2_S` GGUFs. We can sort out the pre-tokenizer issue later.

---

👤 **saood06** commented the **2025-04-22** at **07:08:26**:<br>

> Here `sweep-bench` performance on my Ryzen-7950X using `-ctk q8_0 -fa -rtr -t 16`

I couldn't get flash attention running, it would always just exit with `Floating point exception (core dumped)`.

---

👤 **ikawrakow** commented the **2025-04-22** at **07:16:33**:<br>

> I couldn't get flash attention running, it would always just exit with Floating point exception (core dumped).

Something is missing in the logic for your number of threads. The model has a strange number of attention heads - 20 in total and 5 KV heads. I'm working on a better strategy for distributing the work between the threads.

---

👤 **saood06** commented the **2025-04-22** at **07:26:59**:<br>

> > I couldn't get flash attention running, it would always just exit with Floating point exception (core dumped).
> 
> Something is missing in the logic for your number of threads. The model has a strange number of attention heads - 20 in total and 5 KV heads. I'm working on a better strategy for distributing the work between the threads.

I see, yes I can get it working with 16 and 32 threads, but I can't give performance numbers now as I can't drop my caches right now.