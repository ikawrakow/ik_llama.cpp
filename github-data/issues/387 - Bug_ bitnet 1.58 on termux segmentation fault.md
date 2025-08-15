### 🐛 [#387](https://github.com/ikawrakow/ik_llama.cpp/issues/387) - Bug: bitnet 1.58 on termux segmentation fault

| **Author** | `Benjamin-Wegener` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-06 |
| **Updated** | 2025-05-23 |

---

#### Description

### What happened?

trying original microsoft bitnet 1.58 gguf with ~/ik_llama.cpp $ wget https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf?download=true
creates segmentation fault using
$ ./build/bin/llama-server -mla 3 --model ./models/ggml-model-i2_s.gguf\?download\=true                              INFO [                    main] build info | tid="527362528504" timestamp=1746553079 build=3666 commit="f7c9a0f0"                   INFO [                    main] system info | tid="527362528504" timestamp=1746553079 n_threads=8 n_threads_batch=-1 total_threads=8 system_info="AVX = 0 | AVX_VNNI = 0 | AVX2 = 0 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 0 | NEON = 1 | SVE = 0 | ARM_FMA = 1 | F16C = 0 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 0 | SSSE3 = 0 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "                          llama_model_loader: loaded meta data with 24 key-value pairs and 332 tensors from ./models/ggml-model-i2_s.gguf?download=true (version GGUF V3 (latest))                        llama_model_loader: unknown type i2_s       llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.                                   llama_model_loader: - kv   0:                       general.architecture str              = bitnet-b1.58                            llama_model_loader: - kv   1:                               general.name str              = bitnet2b                                llama_model_loader: - kv   2:                    bitnet-b1.58.vocab_size u32              = 128256                                  llama_model_loader: - kv   3:                bitnet-b1.58.context_length u32              = 4096                                    llama_model_loader: - kv   4:              bitnet-b1.58.embedding_length u32              = 2560                                    llama_model_loader: - kv   5:                   bitnet-b1.58.block_count u32              = 30                                      llama_model_loader: - kv   6:           bitnet-b1.58.feed_forward_length u32              = 6912                                    llama_model_loader: - kv   7:          bitnet-b1.58.rope.dimension_count u32              = 128                                     llama_model_loader: - kv   8:          bitnet-b1.58.attention.head_count u32              = 20                                      llama_model_loader: - kv   9:       bitnet-b1.58.attention.head_count_kv u32              = 5                                       llama_model_loader: - kv  10:               tokenizer.ggml.add_bos_token bool             = true                                    llama_model_loader: - kv  11: bitnet-b1.58.attention.layer_norm_rms_epsilon f32              = 0.000010                             llama_model_loader: - kv  12:                bitnet-b1.58.rope.freq_base f32              = 500000.000000                           llama_model_loader: - kv  13:                          general.file_type u32              = 40                                      llama_model_loader: - kv  14:                       tokenizer.ggml.model str              = gpt2                                    llama_model_loader: - kv  15:                      tokenizer.ggml.tokens arr[str,128256]  = ["!", "\"", "#", "$", "%", "&", "'", ...llama_model_loader: - kv  16:                      tokenizer.ggml.scores arr[f32,128256]  = [0.000000, 0.000000, 0.000000, 0.0000...llama_model_loader: - kv  17:                  tokenizer.ggml.token_type arr[i32,128256]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...llama_model_loader: - kv  18:                      tokenizer.ggml.merges arr[str,280147]  = ["Ġ Ġ", "Ġ ĠĠĠ", "ĠĠ ĠĠ", "...          llama_model_loader: - kv  19:                tokenizer.ggml.bos_token_id u32              = 128000                                  llama_model_loader: - kv  20:                tokenizer.ggml.eos_token_id u32              = 128001                                  llama_model_loader: - kv  21:            tokenizer.ggml.padding_token_id u32              = 128001                                  llama_model_loader: - kv  22:                    tokenizer.chat_template str              = {% for message in messages %}{% if lo...llama_model_loader: - kv  23:               general.quantization_version u32              = 2                                       llama_model_loader: - type  f32:  121 tensors                                           llama_model_loader: - type  f16:    1 tensors                                           llama_model_loader: - type i2_s:  210 tensors                                           llm_load_vocab: missing pre-tokenizer type, using: 'llama3'                             llm_load_vocab:                                                                         llm_load_vocab: ************************************                                    llm_load_vocab: GENERATION QUALITY MAY BE DEGRADED!                                     llm_load_vocab: CONSIDER REGENERATING THE MODEL                                         llm_load_vocab: ************************************                                    llm_load_vocab:                                                                         llm_load_vocab: special tokens cache size = 256                                         llm_load_vocab: token to piece cache size = 0.8000 MB                                   llm_load_print_meta: format           = GGUF V3 (latest)                                llm_load_print_meta: arch             = bitnet-b1.58                                    llm_load_print_meta: vocab type       = BPE llm_load_print_meta: n_vocab          = 128256                                          llm_load_print_meta: n_merges         = 280147                                          llm_load_print_meta: vocab_only       = 0   llm_load_print_meta: n_ctx_train      = 4096llm_load_print_meta: n_embd           = 2560llm_load_print_meta: n_layer          = 30  llm_load_print_meta: n_head           = 20  llm_load_print_meta: n_head_kv        = 5   llm_load_print_meta: n_rot            = 128 llm_load_print_meta: n_swa            = 0   llm_load_print_meta: n_swa_pattern    = 1   llm_load_print_meta: n_embd_head_k    = 128 llm_load_print_meta: n_embd_head_v    = 128 llm_load_print_meta: n_gqa            = 4   llm_load_print_meta: n_embd_k_gqa     = 640 llm_load_print_meta: n_embd_v_gqa     = 640 llm_load_print_meta: f_norm_eps       = 0.0e+00                                         llm_load_print_meta: f_norm_rms_eps   = 1.0e-05                                         llm_load_print_meta: f_clamp_kqv      = 0.0e+00                                         llm_load_print_meta: f_max_alibi_bias = 0.0e+00                                         llm_load_print_meta: f_logit_scale    = 0.0e+00                                         llm_load_print_meta: n_ff             = 6912llm_load_print_meta: n_expert         = 0   llm_load_print_meta: n_expert_used    = 0   llm_load_print_meta: causal attn      = 1   llm_load_print_meta: pooling type     = 0   llm_load_print_meta: rope type        = 2   llm_load_print_meta: rope scaling     = linear                                          llm_load_print_meta: freq_base_train  = 500000.0                                        llm_load_print_meta: freq_scale_train = 1   llm_load_print_meta: n_ctx_orig_yarn  = 4096llm_load_print_meta: rope_finetuned   = unknown                                         llm_load_print_meta: ssm_d_conv       = 0   llm_load_print_meta: ssm_d_inner      = 0   llm_load_print_meta: ssm_d_state      = 0   llm_load_print_meta: ssm_dt_rank      = 0   llm_load_print_meta: model type       = 2B  llm_load_print_meta: model ftype      = unknown, may not work                           llm_load_print_meta: model params     = 2.413 B                                         llm_load_print_meta: model size       = 1.098 GiB (3.911 BPW)                           llm_load_print_meta: general.name     = bitnet2b                                        llm_load_print_meta: BOS token        = 128000 '<|begin_of_text|>'                      llm_load_print_meta: EOS token        = 128001 '<|end_of_text|>'                        llm_load_print_meta: PAD token        = 128001 '<|end_of_text|>'                        llm_load_print_meta: LF token         = 128 'Ä'                                         llm_load_print_meta: EOT token        = 128009 '<|eot_id|>'                             llm_load_print_meta: max token length = 256 llm_load_tensors: ggml ctx size =    0.15 MiB                                           llm_load_tensors:        CPU buffer size =  1124.81 MiB                                 ...............................             =====================================================================                    MLA is only available for LLM_ARCH_DEEPSEEK2 -> turning off MLA                        =====================================================================                   llama_new_context_with_model: n_ctx      = 4096                                         llama_new_context_with_model: n_batch    = 2048                                         llama_new_context_with_model: n_ubatch   = 512                                          llama_new_context_with_model: flash_attn = 0llama_new_context_with_model: mla_attn   = 0llama_new_context_with_model: attn_max_b = 0llama_new_context_with_model: fused_moe  = 0llama_new_context_with_model: ser        = -1, 0                                        llama_new_context_with_model: freq_base  = 500000.0                                     llama_new_context_with_model: freq_scale = 1llama_kv_cache_init:        CPU KV buffer size =   300.00 MiB                           llama_new_context_with_model: KV self size  =  300.00 MiB, K (f16):  150.00 MiB, V (f16):  150.00 MiB                               llama_new_context_with_model:        CPU  output buffer size =     0.98 MiB             llama_new_context_with_model:        CPU compute buffer size =   255.50 MiB             llama_new_context_with_model: graph nodes  = 995                                        llama_new_context_with_model: graph splits = 1                                          Segmentation fault

note: running the optimized version from https://huggingface.co/tdh111/bitnet-b1.58-2B-4T-GGUF/tree/main is starting, but creating gibberish answers
User: hello

Llama: [Nga92SK3#mK\^(K"9E(-l^*hg-,C'2!,

### Name and Version

~/ik_llama.cpp $ ./build/bin/llama-server --version                                     version: 3666 (f7c9a0f0)                    built with clang version 20.1.3 for aarch64-unknown-linux-android24

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell

```

---

#### 💬 Conversation

👤 **Benjamin-Wegener** commented the **2025-05-06** at **17:42:16**:<br>

used
cmake -B ./build -DGGML_CUDA=OFF -DGGML_BLAS=OFF
cmake --build ./build --config Release -j $(nproc)

---

👤 **ikawrakow** commented the **2025-05-06** at **17:45:58**:<br>

You need to convert the model. If you don't find how, I'll add the instructions when back at a computer.

---

👤 **Benjamin-Wegener** commented the **2025-05-06** at **18:09:09**:<br>

thanks, ill report back

---

👤 **Benjamin-Wegener** commented the **2025-05-06** at **19:04:56**:<br>

~/ik_llama.cpp $ ./build/bin/llama-quantize --allow-requantize ./models/bitnet1582b4t-iq2_bn_r4.gguf\?download\=true  ./models/bitnet.gguf iq2_bn_r4

now the model loads with llama-server using no extra args and standard config in browser but just produces User: hello

Llama: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

---

👤 **ikawrakow** commented the **2025-05-06** at **19:37:24**:<br>

You need to convert the `i2_s` model that you downloaded previously
```
./bin/llama-quantize --allow-requantize iq2_s_model new_model_name iq2_bn_r4
./bin/llama-cli -m new_model_name -n 128 -p "The meaning of life is"
```

---

👤 **saood06** commented the **2025-05-06** at **19:51:09**:<br>

I think the issue is #361 which can be worked around using #347 

One indicator of that is if the build process took a short amount of time.

Try adding `-DGGML_ARCH_FLAGS="-march=armv8.2-a+dotprod+fp16"` to your build. (Also do you mind telling us what device you are trying to run this on?)

The models in https://huggingface.co/tdh111/bitnet-b1.58-2B-4T-GGUF are already preconverted (and I ran into the same garbage output when using them on an Android device without building with the flags above)

To test in the server you can send the following request which is lifted straight from from their [transformers PR](https://github.com/huggingface/transformers/pull/37503/files) (the BOS token is ommited as ik_llama.cpp/llama.cpp automatically inserts one):

"User: Hey, are you conscious? Can you talk to me?<|eot_id|>Assistant: "

---

👤 **saood06** commented the **2025-05-06** at **19:51:09**:<br>

I think the issue is #361 which can be worked around using #347 

One indicator of that is if the build process took a short amount of time.

Try adding `-DGGML_ARCH_FLAGS="-march=armv8.2-a+dotprod+fp16"` to your build. 

To test in the server you can send the following request which is lifted straight from from their [transformers PR](https://github.com/huggingface/transformers/pull/37503/files) (the BOS token is ommited as ik_llama.cpp/llama.cpp automatically inserts one):

"User: Hey, are you conscious? Can you talk to me?<|eot_id|>Assistant:"

---

👤 **Benjamin-Wegener** commented the **2025-05-07** at **06:28:44**:<br>

> I think the issue is [#361](https://github.com/ikawrakow/ik_llama.cpp/issues/361) which can be worked around using [#347](https://github.com/ikawrakow/ik_llama.cpp/pull/347)
> 
> One indicator of that is if the build process took a short amount of time.
> 
> Try adding `-DGGML_ARCH_FLAGS="-march=armv8.2-a+dotprod+fp16"` to your build. (Also do you mind telling us what device you are trying to run this on?)
> 
> The models in https://huggingface.co/tdh111/bitnet-b1.58-2B-4T-GGUF are already preconverted (and I ran into the same garbage output when using them on an Android device without building with the flags above)
> 
> To test in the server you can send the following request which is lifted straight from from their [transformers PR](https://github.com/huggingface/transformers/pull/37503/files) (the BOS token is ommited as ik_llama.cpp/llama.cpp automatically inserts one):
> 
> "User: Hey, are you conscious? Can you talk to me?<|eot_id|>Assistant:"

that helps, now its working, thank you

---

👤 **Benjamin-Wegener** commented the **2025-05-09** at **04:30:45**:<br>

just for convenience all subsequential commands to install bitnet (or other cpu models) on a fresh termux aarch64:
```bash
apt update && apt install wget cmake git -y
git clone https://github.com/ikawrakow/ik_llama.cpp
cd ik_llama.cpp
cmake -B ./build -DGGML_CUDA=OFF -DGGML_BLAS=OFF -DGGML_ARCH_FLAGS="-march=armv8.2-a+dotprod+fp16"
cmake --build ./build --config Release -j $(nproc)
wget https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf?download=true -O ./models/ggml-model-i2_s.gguf
./build/bin/llama-quantize --allow-requantize ./models/ggml-model-is_s.gguf ./models/bitnet.gguf iq2_bn_r4
./build/bin/llama-server -mla 3 --model ./models/bitnet.gguf
```

---

👤 **Benjamin-Wegener** commented the **2025-05-09** at **04:30:45**:<br>

just for convenience all subsequential commands to install bitnet (or other cpu models) on a fresh termux aarch64:
`
apt update && apt install wget cmake git -y
git clone https://github.com/ikawrakow/ik_llama.cpp
cd ik_llama.cpp
cmake -B ./build -DGGML_CUDA=OFF -DGGML_BLAS=OFF -DGGML_ARCH_FLAGS="-march=armv8.2-a+dotprod+fp16"
cmake --build ./build --config Release -j $(nproc)
wget https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf?download=true -O ./models/ggml-model-i2_s.gguf
./build/bin/llama-quantize --allow-requantize ./models/ggml-model-is_s.gguf ./models/bitnet.gguf iq2_bn_r4
./build/bin/llama-server -mla 3 --model ./models/bitnet.gguf
`

---

👤 **ikawrakow** commented the **2025-05-09** at **08:19:12**:<br>

@Benjamin-Wegener Thank you for these instructions. Do you mind if I take them and make a Discussion for better visibility. Or, if you prefer, you can do it yourself. Let me know.

---

👤 **Benjamin-Wegener** commented the **2025-05-09** at **09:20:13**:<br>

sure, will do
EDIT: done https://github.com/ikawrakow/ik_llama.cpp/discussions/401

---

👤 **Benjamin-Wegener** commented the **2025-05-09** at **09:20:13**:<br>

sure, will do

---

👤 **Manamama** commented the **2025-05-23** at **08:50:18**:<br>

FYI, I have tested your https://github.com/ikawrakow/ik_llama.cpp/issues/387#issuecomment-2865065414 out of curiosity on my "somewhat contaminated" Termux. 

Both llama.cpp and yours used to compile fine, but at least today: 
1. llama.cpp still compiles fine (but then seg faults on some ggufs only, see https://github.com/ggml-org/llama.cpp/issues/13708#issuecomment-2902117306) 
2. Your one, when I do just that: https://github.com/ikawrakow/ik_llama.cpp/issues/387#issuecomment-2865065414, causes: 

```
Environment at system:
Linux localhost 4.14.186+ #1 SMP PREEMPT Thu Mar 17 16:28:22 CST 2022 aarch64 Android


PATH: /data/data/com.termux/files/usr/google-cloud-sdk/bin:/data/data/com.termux/files/home/.opam/default/bin:/data/data/com.termux/files/usr/bin:/system/bin/:/data/data/com.termux/files/usr/bin:/system/bin/:/data/data/com.termux/files/usr/bin:/data/data/com.termux/files/usr/bin/texlive:/data/data/com.termux/files/usr/bin/texlive:/data/data/com.termux/files/home/.local/bin:/build-tools/30.0.3

LD_PRELOAD: /data/data/com.termux/files/usr/lib/libtermux-exec-direct-ld-preload.so

LD_LIBRARY_PATH: 

CC: clang
CXX: clang++
C_INCLUDE_PATH: 
FC: lfortran
CFLAGS: 
CXXFLAGS: 
LDFLAGS: -llog -largp -lm
CPPFLAGS: 
CMAKE_PREFIX_PATH: :/data/data/com.termux/files/usr/lib/cmake/Qt6HostInfo

JAVA_HOME: /data/data/com.termux/files/usr/lib/jvm/java-17-openjdk
ANDROID_NDK: /storage/emulated/0/Download/android-ndk-r26b
ANDROID_SDK: /storage/sdcard1/Installs/Android_ndk_sdk/SDK

```
and then
```

~/downloads $ git clone https://github.com/ikawrakow/ik_llama.cpp
cd ik_llama.cpp
Cloning into 'ik_llama.cpp'...
remote: Enumerating objects: 29327, done.
remote: Counting objects: 100% (8480/8480), done.
remote: Compressing objects: 100% (788/788), done.
remote: Total 29327 (delta 8003), reused 7707 (delta 7692), pack-reused 20847 (from 2)
Receiving objects: 100% (29327/29327), 34.13 MiB | 98.00 KiB/s, done.
Resolving deltas: 100% (22227/22227), done.
Updating files: 100% (1027/1027), done.
~/downloads/ik_llama.cpp $ cd ik^C
~/downloads/ik_llama.cpp $ ls
 AUTHORS          CMakePresets.json       convert_hf_to_gguf_update.py    examples     gguf-py    Makefile   Package.swift   pyproject.toml      󰌠 requirements.txt  󰙨 tests
 ci               common                  convert_llama_ggml_to_gguf.py   flake.lock   grammars   media      pocs            pyrightconfig.json   scripts           
 cmake            CONTRIBUTING.md         convert_lora_to_gguf.py         flake.nix    include    models     poetry.lock     README.md            spm-headers       
 CMakeLists.txt   convert_hf_to_gguf.py   docs                            ggml         LICENSE    mypy.ini   prompts         requirements        󱧼 src               
~/downloads/ik_llama.cpp $ 
cmake -B ./build -DGGML_CUDA=OFF -DGGML_BLAS=OFF -DGGML_ARCH_FLAGS="-march=armv8.2-a+dotprod+fp16"
cmake --build ./build --config Release -j $(nproc)
-- The C compiler identification is Clang 20.1.5
-- The CXX compiler identification is Clang 20.1.5
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /data/data/com.termux/files/usr/bin/clang - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /data/data/com.termux/files/usr/bin/clang++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Found Git: /data/data/com.termux/files/usr/bin/git (found version "2.49.0")
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Failed
-- Check if compiler accepts -pthread
-- Check if compiler accepts -pthread - yes
-- Found Threads: TRUE
-- Found OpenMP_C: -fopenmp=libomp (found version "5.1")
-- Found OpenMP_CXX: -fopenmp=libomp (found version "5.1")
-- Found OpenMP: TRUE (found version "5.1")
-- OpenMP found
-- Using optimized iqk matrix multiplications
-- Enabling IQK Flash Attention kernels
-- Using llamafile
-- ccache found, compilation results will be cached. Disable with GGML_CCACHE=OFF.
-- CMAKE_SYSTEM_PROCESSOR: aarch64
-- ARM detected
-- Performing Test COMPILER_SUPPORTS_FP16_FORMAT_I3E
-- Performing Test COMPILER_SUPPORTS_FP16_FORMAT_I3E - Failed
-- Looking for pthread_create in pthreads
-- Looking for pthread_create in pthreads - not found
-- Looking for pthread_create in pthread
-- Looking for pthread_create in pthread - found
-- ARCH_FLAGS = -march=native
-- Configuring done (17.5s)
-- Generating done (1.4s)
-- Build files have been written to: /data/data/com.termux/files/home/downloads/ik_llama.cpp/build
[  0%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml.c.o
[  1%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml-alloc.c.o
... 
[ 79%] Building CXX object examples/perplexity/CMakeFiles/llama-perplexity.dir/perplexity.cpp.o
[ 80%] Linking CXX executable ../../bin/llama-perplexity
[ 80%] Built target llama-perplexity
[ 81%] Building CXX object examples/quantize-stats/CMakeFiles/llama-quantize-stats.dir/quantize-stats.cpp.o
/data/data/com.termux/files/home/downloads/ik_llama.cpp/examples/quantize-stats/quantize-stats.cpp:782:57: error: expected ')'
  782 |                         if (sumqx*sumqx*sumq2i[j] > best]) {
      |                                                         ^
/data/data/com.termux/files/home/downloads/ik_llama.cpp/examples/quantize-stats/quantize-stats.cpp:782:28: note: to match this '('
  782 |                         if (sumqx*sumqx*sumq2i[j] > best]) {
      |                            ^
/data/data/com.termux/files/home/downloads/ik_llama.cpp/examples/quantize-stats/quantize-stats.cpp:782:57: error: expected expression
  782 |                         if (sumqx*sumqx*sumq2i[j] > best]) {
      |                                                         ^
/data/data/com.termux/files/home/downloads/ik_llama.cpp/examples/quantize-stats/quantize-stats.cpp:782:58: error: expected expression
  782 |                         if (sumqx*sumqx*sumq2i[j] > best]) {
      |                                                          ^
3 errors generated.
make[2]: *** [examples/quantize-stats/CMakeFiles/llama-quantize-stats.dir/build.make:79: examples/quantize-stats/CMakeFiles/llama-quantize-stats.dir/quantize-stats.cpp.o] Error 1
make[1]: *** [CMakeFiles/Makefile2:3920: examples/quantize-stats/CMakeFiles/llama-quantize-stats.dir/all] Error 2
make: *** [Makefile:146: all] Error 2
```
 
I have taken a peek at this `quantize-stats.cpp` and these strings asre indeed there, but I am bad in counting the closing brackets vs the opening ones by hand ...

---

👤 **Manamama** commented the **2025-05-23** at **08:50:18**:<br>

FYI, I have tested your https://github.com/ikawrakow/ik_llama.cpp/issues/387#issuecomment-2865065414 out of curiosity on my "somewhat contaminated" Termux. 

Both llama.cpp and yours used to compile fine, but at least today: 
1. llama.cpp still compiles fine (but then seg faults on some ggufs only, see https://github.com/ggml-org/llama.cpp/issues/13708#issuecomment-2902117306) 
2. Your one, when I do just that: https://github.com/ikawrakow/ik_llama.cpp/issues/387#issuecomment-2865065414, causes: 

```
Environment at system:
Linux localhost 4.14.186+ #1 SMP PREEMPT Thu Mar 17 16:28:22 CST 2022 aarch64 Android


PATH: /data/data/com.termux/files/usr/google-cloud-sdk/bin:/data/data/com.termux/files/home/.opam/default/bin:/data/data/com.termux/files/usr/bin:/system/bin/:/data/data/com.termux/files/usr/bin:/system/bin/:/data/data/com.termux/files/usr/bin:/data/data/com.termux/files/usr/bin/texlive:/data/data/com.termux/files/usr/bin/texlive:/data/data/com.termux/files/home/.local/bin:/build-tools/30.0.3

LD_PRELOAD: /data/data/com.termux/files/usr/lib/libtermux-exec-direct-ld-preload.so

LD_LIBRARY_PATH: 

CC: clang
CXX: clang++
C_INCLUDE_PATH: 
FC: lfortran
CFLAGS: 
CXXFLAGS: 
LDFLAGS: -llog -largp -lm
CPPFLAGS: 
CMAKE_PREFIX_PATH: :/data/data/com.termux/files/usr/lib/cmake/Qt6HostInfo

JAVA_HOME: /data/data/com.termux/files/usr/lib/jvm/java-17-openjdk
ANDROID_NDK: /storage/emulated/0/Download/android-ndk-r26b
ANDROID_SDK: /storage/sdcard1/Installs/Android_ndk_sdk/SDK

```
~/downloads $ git clone https://github.com/ikawrakow/ik_llama.cpp
cd ik_llama.cpp
Cloning into 'ik_llama.cpp'...
remote: Enumerating objects: 29327, done.
remote: Counting objects: 100% (8480/8480), done.
remote: Compressing objects: 100% (788/788), done.
remote: Total 29327 (delta 8003), reused 7707 (delta 7692), pack-reused 20847 (from 2)
Receiving objects: 100% (29327/29327), 34.13 MiB | 98.00 KiB/s, done.
Resolving deltas: 100% (22227/22227), done.
Updating files: 100% (1027/1027), done.
~/downloads/ik_llama.cpp $ cd ik^C
~/downloads/ik_llama.cpp $ ls
 AUTHORS          CMakePresets.json       convert_hf_to_gguf_update.py    examples     gguf-py    Makefile   Package.swift   pyproject.toml      󰌠 requirements.txt  󰙨 tests
 ci               common                  convert_llama_ggml_to_gguf.py   flake.lock   grammars   media      pocs            pyrightconfig.json   scripts           
 cmake            CONTRIBUTING.md         convert_lora_to_gguf.py         flake.nix    include    models     poetry.lock     README.md            spm-headers       
 CMakeLists.txt   convert_hf_to_gguf.py   docs                            ggml         LICENSE    mypy.ini   prompts         requirements        󱧼 src               
~/downloads/ik_llama.cpp $ 
cmake -B ./build -DGGML_CUDA=OFF -DGGML_BLAS=OFF -DGGML_ARCH_FLAGS="-march=armv8.2-a+dotprod+fp16"
cmake --build ./build --config Release -j $(nproc)
-- The C compiler identification is Clang 20.1.5
-- The CXX compiler identification is Clang 20.1.5
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /data/data/com.termux/files/usr/bin/clang - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /data/data/com.termux/files/usr/bin/clang++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Found Git: /data/data/com.termux/files/usr/bin/git (found version "2.49.0")
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Failed
-- Check if compiler accepts -pthread
-- Check if compiler accepts -pthread - yes
-- Found Threads: TRUE
-- Found OpenMP_C: -fopenmp=libomp (found version "5.1")
-- Found OpenMP_CXX: -fopenmp=libomp (found version "5.1")
-- Found OpenMP: TRUE (found version "5.1")
-- OpenMP found
-- Using optimized iqk matrix multiplications
-- Enabling IQK Flash Attention kernels
-- Using llamafile
-- ccache found, compilation results will be cached. Disable with GGML_CCACHE=OFF.
-- CMAKE_SYSTEM_PROCESSOR: aarch64
-- ARM detected
-- Performing Test COMPILER_SUPPORTS_FP16_FORMAT_I3E
-- Performing Test COMPILER_SUPPORTS_FP16_FORMAT_I3E - Failed
-- Looking for pthread_create in pthreads
-- Looking for pthread_create in pthreads - not found
-- Looking for pthread_create in pthread
-- Looking for pthread_create in pthread - found
-- ARCH_FLAGS = -march=native
-- Configuring done (17.5s)
-- Generating done (1.4s)
-- Build files have been written to: /data/data/com.termux/files/home/downloads/ik_llama.cpp/build
[  0%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml.c.o
[  1%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml-alloc.c.o
... 
[ 79%] Building CXX object examples/perplexity/CMakeFiles/llama-perplexity.dir/perplexity.cpp.o
[ 80%] Linking CXX executable ../../bin/llama-perplexity
[ 80%] Built target llama-perplexity
[ 81%] Building CXX object examples/quantize-stats/CMakeFiles/llama-quantize-stats.dir/quantize-stats.cpp.o
/data/data/com.termux/files/home/downloads/ik_llama.cpp/examples/quantize-stats/quantize-stats.cpp:782:57: error: expected ')'
  782 |                         if (sumqx*sumqx*sumq2i[j] > best]) {
      |                                                         ^
/data/data/com.termux/files/home/downloads/ik_llama.cpp/examples/quantize-stats/quantize-stats.cpp:782:28: note: to match this '('
  782 |                         if (sumqx*sumqx*sumq2i[j] > best]) {
      |                            ^
/data/data/com.termux/files/home/downloads/ik_llama.cpp/examples/quantize-stats/quantize-stats.cpp:782:57: error: expected expression
  782 |                         if (sumqx*sumqx*sumq2i[j] > best]) {
      |                                                         ^
/data/data/com.termux/files/home/downloads/ik_llama.cpp/examples/quantize-stats/quantize-stats.cpp:782:58: error: expected expression
  782 |                         if (sumqx*sumqx*sumq2i[j] > best]) {
      |                                                          ^
3 errors generated.
make[2]: *** [examples/quantize-stats/CMakeFiles/llama-quantize-stats.dir/build.make:79: examples/quantize-stats/CMakeFiles/llama-quantize-stats.dir/quantize-stats.cpp.o] Error 1
make[1]: *** [CMakeFiles/Makefile2:3920: examples/quantize-stats/CMakeFiles/llama-quantize-stats.dir/all] Error 2
make: *** [Makefile:146: all] Error 2
```
 
I have taken a peek at this `quantize-stats.cpp` and these strings asre indeed there, but I am bad in counting the closing brackets vs the opening ones by hand ...
```

---

👤 **ikawrakow** commented the **2025-05-23** at **09:02:05**:<br>

Does #445 fix it?

---

👤 **Manamama** commented the **2025-05-23** at **18:34:02**:<br>

Yes, it compiles now. 
Testing:
```
wget https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf?download=true -O ./models/ggml-model-i2_s.gguf
./build/bin/llama-quantize --allow-requantize ./models/ggml-model-is_s.gguf ./models/bitnet.gguf iq2_bn_r4
./build/bin/llama-server -mla 3 --model ./models/bitnet.gguf
```
...

It fails now with:

```
Resolving cdn-lfs-us-1.hf.co (cdn-lfs-us-1.hf.co)... 18.164.52.87, 18.164.52.5, 18.164.52.44, ...         Connecting to cdn-lfs-us-1.hf.co (cdn-lfs-us-1.hf.co)|18.164.52.87|:443... connected.                     HTTP request sent, awaiting response... 200 OK       Length: 1187801280 (1.1G) [application/octet-stream] Saving to: ‘./models/ggml-model-i2_s.gguf’                                                                ./models/ggml 100%   1.11G   774KB/s    in 25m 14s                                                        2025-05-23 20:58:34 (766 KB/s) - ‘./models/ggml-model-i2_s.gguf’ saved [1187801280/1187801280]                                                                 CANNOT LINK EXECUTABLE "./build/bin/llama-quantize": cannot locate symbol "ggml_backend_reg_get_count" referenced by "/data/data/com.termux/files/home/downloads/ik_llama.cpp/build/bin/llama-quantize"...          CANNOT LINK EXECUTABLE "./build/bin/llama-server": cannot locate symbol "llama_get_kv_cache_token_count" referenced by "/data/data/com.termux/files/home/downloads/ik_llama.cpp/build/bin/llama-server"...          ~/downloads/ik_llama.cpp $

```


This may be needed, once again: https://github.com/ikawrakow/ik_llama.cpp/issues/388#issue-3043737093

Quick update: my trick does not help either. 

```
~/downloads/ik_llama.cpp $ ./build/bin/llama-quantize --allow-requantize ./models/ggml-model-is_s.gguf ./models/bitnet.gguf iq2_bn_r4                          CANNOT LINK EXECUTABLE "./build/bin/llama-quantize": cannot locate symbol "ggml_backend_reg_get_count" referenced by "/data/data/com.termux/files/home/downloads/ik_llama.cpp/build/bin/llama-quantize"...          ~/downloads/ik_llama.cpp $ ldd "/data/data/com.termux/files/home/downloads/ik_llama.cpp/build/bin/llama-quantize"                                                      liblog.so => /system/lib64/liblog.so                 libargp.so => /data/data/com.termux/files/usr/lib/libargp.so                                              libc.so => /system/lib64/libc.so                     libllama.so => /data/data/com.termux/files/usr/lib/libllama.so
        libggml.so => /data/data/com.termux/files/usr/lib/libggml.so                                              libc++_shared.so => /data/data/com.termux/files/usr/lib/libc++_shared.so
        libdl.so => /system/lib64/libdl.so                   libm.so => /system/lib64/libm.so
        libc++.so => /system/lib64/libc++.so                 ld-android.so => /system/lib64/ld-android.so         libclang_rt.asan-aarch64-android.so => /system/lib64/libclang_rt.asan-aarch64-android.so                  libggml-cpu.so => /data/data/com.termux/files/usr/lib/libggml-cpu.so                                      libggml-base.so => /data/data/com.termux/files/usr/lib/libggml-base.so                            ~/downloads/ik_llama.cpp $
```
after recompilation, too. 

Ver. 1.3

---

👤 **Manamama** commented the **2025-05-23** at **18:34:02**:<br>

Yes, it compiles now. 
Testing:
```
wget https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf?download=true -O ./models/ggml-model-i2_s.gguf
./build/bin/llama-quantize --allow-requantize ./models/ggml-model-is_s.gguf ./models/bitnet.gguf iq2_bn_r4
./build/bin/llama-server -mla 3 --model ./models/bitnet.gguf
```
...