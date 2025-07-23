### 🐛 [#376](https://github.com/ikawrakow/ik_llama.cpp/issues/376) - Bug: unknown model architecture: 'deci' (when loading Llama-3_1-Nemotron-Ultra-253B)

| **Author** | `Lissanro` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-04 |
| **Updated** | 2025-05-09 |

---

#### Description

### What happened?

Llama-3_1-Nemotron-Ultra-253B has special architecture called "deci", its support has been added to llama.cpp using this PR: https://github.com/ggml-org/llama.cpp/pull/12843 - perhaps adding support for this architecture could be considered for ik_llama.cpp?

### Name and Version

Latest git

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
~/pkgs/ik_llama.cpp/build/bin/llama-server --model /mnt/secondary/neuro/Llama-3_1-Nemotron-Ultra-253B-v1-GGUF-UD-Q4_K_XL-131072seq/Llama-3_1-Nemotron-Ultra-253B-v1-UD-Q4_K_XL-00001-of-00004.gguf --ctx-size 81920 --n-gpu-layers 12 --tensor-split 25,25,25,25 -fa -ctk q8_0 -ctv q8_0 --threads 64 --host 0.0.0.0 --port 5000 --split-mode row
INFO [                    main] build info | tid="136009399906304" timestamp=1746347014 build=3661 commit="ab7f694b"
INFO [                    main] system info | tid="136009399906304" timestamp=1746347014 n_threads=64 n_threads_batch=-1 total_threads=128 system_info="AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: additional 3 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 43 key-value pairs and 648 tensors from /mnt/secondary/neuro/Llama-3_1-Nemotron-Ultra-253B-v1-GGUF-UD-Q4_K_XL-131072seq/Llama-3_1-Nemotron-Ultra-253B-v1-UD-Q4_K_XL-00001-of-00004.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deci
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Llama_Nemotron_Ultra
llama_model_loader: - kv   3:                            general.version str              = v1
llama_model_loader: - kv   4:                           general.finetune str              = 3_1-Nemotron-Ultra
llama_model_loader: - kv   5:                           general.basename str              = Llama-3_1-Nemotron-Ultra-253B-V1
llama_model_loader: - kv   6:                       general.quantized_by str              = Unsloth
llama_model_loader: - kv   7:                         general.size_label str              = 253B
llama_model_loader: - kv   8:                            general.license str              = other
llama_model_loader: - kv   9:                       general.license.name str              = nvidia-open-model-license
llama_model_loader: - kv  10:                       general.license.link str              = https://www.nvidia.com/en-us/agreemen...
llama_model_loader: - kv  11:                           general.repo_url str              = https://huggingface.co/unsloth
llama_model_loader: - kv  12:                               general.tags arr[str,4]       = ["nvidia", "llama-3", "pytorch", "tex...
llama_model_loader: - kv  13:                          general.languages arr[str,1]       = ["en"]
llama_model_loader: - kv  14:                        deci.rope.freq_base f32              = 500000.000000
llama_model_loader: - kv  15:               deci.attention.head_count_kv arr[i32,162]     = [8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, ...
llama_model_loader: - kv  16:                  deci.attention.head_count arr[i32,162]     = [128, 128, 128, 128, 128, 128, 128, 1...
llama_model_loader: - kv  17:                   deci.feed_forward_length arr[i32,162]     = [5376, 10752, 16128, 16128, 16128, 16...
llama_model_loader: - kv  18:                           deci.block_count u32              = 162
llama_model_loader: - kv  19:                        deci.context_length u32              = 131072
llama_model_loader: - kv  20:                      deci.embedding_length u32              = 16384
llama_model_loader: - kv  21:      deci.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  22:                  deci.attention.key_length u32              = 128
llama_model_loader: - kv  23:                deci.attention.value_length u32              = 128
llama_model_loader: - kv  24:                            deci.vocab_size u32              = 128256
llama_model_loader: - kv  25:                  deci.rope.dimension_count u32              = 128
llama_model_loader: - kv  26:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  27:                         tokenizer.ggml.pre str              = llama-bpe
llama_model_loader: - kv  28:                      tokenizer.ggml.tokens arr[str,128256]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  29:                  tokenizer.ggml.token_type arr[i32,128256]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  30:                      tokenizer.ggml.merges arr[str,280147]  = ["Ġ Ġ", "Ġ ĠĠĠ", "ĠĠ ĠĠ", "...
llama_model_loader: - kv  31:                tokenizer.ggml.bos_token_id u32              = 128000
llama_model_loader: - kv  32:                tokenizer.ggml.eos_token_id u32              = 128009
llama_model_loader: - kv  33:                    tokenizer.chat_template str              = {{- bos_token }}{%- if messages[0]['r...
llama_model_loader: - kv  34:               general.quantization_version u32              = 2
llama_model_loader: - kv  35:                          general.file_type u32              = 15
llama_model_loader: - kv  36:                      quantize.imatrix.file str              = Llama-3_1-Nemotron-Ultra-253B-v1-GGUF...
llama_model_loader: - kv  37:                   quantize.imatrix.dataset str              = unsloth_calibration_Llama-3_1-Nemotro...
llama_model_loader: - kv  38:             quantize.imatrix.entries_count i32              = 499
llama_model_loader: - kv  39:              quantize.imatrix.chunks_count i32              = 544
llama_model_loader: - kv  40:                                   split.no u16              = 0
llama_model_loader: - kv  41:                        split.tensors.count i32              = 648
llama_model_loader: - kv  42:                                split.count u16              = 4
llama_model_loader: - type  f32:  147 tensors
llama_model_loader: - type q4_K:  428 tensors
llama_model_loader: - type q6_K:   73 tensors
llama_model_load: error loading model: error loading model architecture: unknown model architecture: 'deci'
llama_load_model_from_file: failed to load model
llama_init_from_gpt_params: error: failed to load model '/mnt/secondary/neuro/Llama-3_1-Nemotron-Ultra-253B-v1-GGUF-UD-Q4_K_XL-131072seq/Llama-3_1-Nemotron-Ultra-253B-v1-UD-Q4_K_XL-00001-of-00004.gguf'
 ERR [              load_model] unable to load model | tid="136009399906304" timestamp=1746347014 model="/mnt/secondary/neuro/Llama-3_1-Nemotron-Ultra-253B-v1-GGUF-UD-Q4_K_XL-131072seq/Llama-3_1-Nemotron-Ultra-253B-v1-UD-Q4_K_XL-00001-of-00004.gguf"
munmap_chunk(): invalid pointer
```

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-05-04** at **09:35:03**:<br>

I can take a look, but as with other giant models, I cannot test. Are you willing to test and provide benchmarks?

---

👤 **saood06** commented the **2025-05-04** at **09:38:35**:<br>

I'm already working on it.

---

👤 **Lissanro** commented the **2025-05-04** at **11:01:33**:<br>

> Are you willing to test and provide benchmarks?

Sure, I will be happy to test, at both short and log context lengths.

As of benchmarks, at very least I planned to test input processing and output generation speeds - but if something else will be needed please let me know and I will consider it if I can test it.

---

👤 **saood06** commented the **2025-05-04** at **11:07:21**:<br>

>Sure, I will be happy to test, at both short and log context lengths.

What about the small model as the initial architecture support was with that one: https://github.com/ggml-org/llama.cpp/pull/10669

>As of benchmarks, at very least I planned to test input processing and output generation speeds

You can use sweep-bench to do that.

---

👤 **Lissanro** commented the **2025-05-04** at **11:34:07**:<br>

I do not have the smaller model yet but I can try downloading it, for example from here https://huggingface.co/bartowski/Llama-3_1-Nemotron-51B-Instruct-GGUF (I only have 4G connection though and have some things still downloading, but I should be able to get the 51B within 2 days in case it will be needed for testing).

---

👤 **saood06** commented the **2025-05-04** at **11:46:19**:<br>

>but I should be able to get the 51B within 2 days in case it will be needed for testing

I'll try to test the smaller one then. I just created a draft PR: https://github.com/ikawrakow/ik_llama.cpp/pull/377