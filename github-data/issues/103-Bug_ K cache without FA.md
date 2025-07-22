### 🐛 [#103](https://github.com/ikawrakow/ik_llama.cpp/issues/103) - Bug: K cache without FA

| **Author** | `Nexesenex` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-10-23 |
| **Updated** | 2024-10-24 |

---

#### Description

### What happened?

With the non-FA Quantum K cache, q6_0 works.

But q4_0, q4_1, q5_0, q5_1, q8_0 do not work anymore as K quant without FA, both on IK_L and mainline, and go NaN instead. As does iq4_nl K/no FA.

(I personally don't mind, K q6_0 is my new bff K cache quant).

Tested on Llama 3.1 8b Q5_K.

### Name and Version

b3962 on Mainline.
Pre granite merge on IK.

### What operating system are you seeing the problem on?

Windows

### Relevant log output

```shell
Q:\LLAMA_IK>llama-perplexity -m D:\text-generation-webui\models\Meta_Llama_3.1_8b_it-f16-iMat-Q5_K_S_q4_v6.gguf -f wiki.test.raw --parallel 1 -ngl 150 -b 1024 -ts 40,0 --no-mmap -ctk iq4_nl -c 512 --chunks 211
main: build = 3475 (ac156500)
main: built with MSVC 19.38.33141.0 for
main: seed  = 1729657101
llama_model_loader: loaded meta data with 31 key-value pairs and 292 tensors from D:\text-generation-webui\models\Meta_Llama_3.1_8b_it-f16-iMat-Q5_K_S_q4_v6.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Meta_Llama_3.1_8b_it
llama_model_loader: - kv   3:                         general.size_label str              = 8.0B
llama_model_loader: - kv   4:                            general.license str              = llama3.1
llama_model_loader: - kv   5:                               general.tags arr[str,6]       = ["facebook", "meta", "pytorch", "llam...
llama_model_loader: - kv   6:                          general.languages arr[str,8]       = ["en", "de", "fr", "it", "pt", "hi", ...
llama_model_loader: - kv   7:                          llama.block_count u32              = 32
llama_model_loader: - kv   8:                       llama.context_length u32              = 131072
llama_model_loader: - kv   9:                     llama.embedding_length u32              = 4096
llama_model_loader: - kv  10:                  llama.feed_forward_length u32              = 14336
llama_model_loader: - kv  11:                 llama.attention.head_count u32              = 32
llama_model_loader: - kv  12:              llama.attention.head_count_kv u32              = 8
llama_model_loader: - kv  13:                       llama.rope.freq_base f32              = 500000.000000
llama_model_loader: - kv  14:     llama.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  15:                          general.file_type u32              = 16
llama_model_loader: - kv  16:                           llama.vocab_size u32              = 128256
llama_model_loader: - kv  17:                 llama.rope.dimension_count u32              = 128
llama_model_loader: - kv  18:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  19:                         tokenizer.ggml.pre str              = llama-bpe
llama_model_loader: - kv  20:                      tokenizer.ggml.tokens arr[str,128256]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  21:                  tokenizer.ggml.token_type arr[i32,128256]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  22:                      tokenizer.ggml.merges arr[str,280147]  = ["─á ─á", "─á ─á─á─á", "─á─á ─á─á", "...
llama_model_loader: - kv  23:                tokenizer.ggml.bos_token_id u32              = 128000
llama_model_loader: - kv  24:                tokenizer.ggml.eos_token_id u32              = 128009
llama_model_loader: - kv  25:                    tokenizer.chat_template str              = {{- bos_token }}\n{%- if custom_tools ...
llama_model_loader: - kv  26:               general.quantization_version u32              = 2
llama_model_loader: - kv  27:                      quantize.imatrix.file str              = Q:\iMatrix\Meta_Llama_3.1_8b_it-f16.i...
llama_model_loader: - kv  28:                   quantize.imatrix.dataset str              = groups_merged-enhancedV3_FR_SRB_HR.txt
llama_model_loader: - kv  29:             quantize.imatrix.entries_count i32              = 224
llama_model_loader: - kv  30:              quantize.imatrix.chunks_count i32              = 145
llama_model_loader: - type  f32:   66 tensors
llama_model_loader: - type q4_K:   32 tensors
llama_model_loader: - type q5_K:  161 tensors
llama_model_loader: - type q6_K:   33 tensors
llm_load_vocab: special tokens cache size = 256
llm_load_vocab: token to piece cache size = 0.7999 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = llama
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 128256
llm_load_print_meta: n_merges         = 280147
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 131072
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_layer          = 32
llm_load_print_meta: n_head           = 32
llm_load_print_meta: n_head_kv        = 8
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 4
llm_load_print_meta: n_embd_k_gqa     = 1024
llm_load_print_meta: n_embd_v_gqa     = 1024
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 14336
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 500000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 131072
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = 8B
llm_load_print_meta: model ftype      = Q5_K - Small
llm_load_print_meta: model params     = 8.030 B
llm_load_print_meta: model size       = 5.162 GiB (5.521 BPW)
llm_load_print_meta: repeating layers = 4.424 GiB (5.445 BPW, 6.980 B parameters)
llm_load_print_meta: general.name     = Meta_Llama_3.1_8b_it
llm_load_print_meta: BOS token        = 128000 '<|begin_of_text|>'
llm_load_print_meta: EOS token        = 128009 '<|eot_id|>'
llm_load_print_meta: LF token         = 128 '├ä'
llm_load_print_meta: EOT token        = 128009 '<|eot_id|>'
llm_load_print_meta: max token length = 256
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 3 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 1: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 2: NVIDIA RTX A4000, compute capability 8.6, VMM: yes
llm_load_tensors: ggml ctx size =    0.27 MiB
llm_load_tensors: offloading 32 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 33/33 layers to GPU
llm_load_tensors:  CUDA_Host buffer size =   344.44 MiB
llm_load_tensors:      CUDA0 buffer size =  4941.00 MiB
........................................................................................
llama_new_context_with_model: n_ctx      = 1024
llama_new_context_with_model: n_batch    = 1024
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: freq_base  = 500000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:      CUDA0 KV buffer size =    82.00 MiB
llama_new_context_with_model: KV self size  =   82.00 MiB, K (iq4_nl):   18.00 MiB, V (f16):   64.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.98 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      CUDA0 compute buffer size =   266.50 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =    10.01 MiB
llama_new_context_with_model: graph nodes  = 933
llama_new_context_with_model: graph splits = 2

system_info: n_threads = 8 / 16 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 |
perplexity: tokenizing the input ..
perplexity: tokenization took 213.322 ms
perplexity: calculating perplexity over 211 chunks, n_ctx=512, batch_size=1024, n_seq=2
perplexity: 9.17 seconds per pass - ETA 16.12 minutes
[1]-nan,[2]-nan,[3]-nan,[4]-nan,[5]-nan,[6]-nan,[7]-nan,[8]-nan,[9]-nan,[10]-nan,[11]-nan,[12]-nan,[13]-nan,[14]-nan,[15]-nan,[16]-nan,[17]-nan,[18]-nan,[19]-nan,[20]-nan,[21]-nan,[22]-nan,[23]-nan,[24]-nan,[25]-nan,[26]-nan,[27]-nan,[28]-nan,[29]-nan,[30]-nan,[31]-nan,[32]-nan,[33]-nan,[34]-nan,[35]-nan,[36]-nan,[37]-nan,[38]-nan,[39]-nan,[40]-nan,[41]-nan,[42]-nan,[43]-nan,[44]-nan,[45]-nan,[46]-nan,[47]-nan,[48]-nan,[49]-nan,[50]-nan,[51]-nan,[52]-nan,[53]-nan,[54]-nan,[55]-nan,[56]-nan,[57]-nan,[58]-nan,[59]-nan,[60]-nan,[61]-nan,[62]-nan,[63]-nan,[64]-nan,[65]-nan,[66]-nan,[67]-nan,[68]-nan,[69]-nan,[70]-nan,[71]-nan,[72]-nan,[73]-nan,[74]-nan,[75]-nan,[76]-nan,[77]-nan,[78]-nan,[79]-nan,[80]-nan,[81]-nan,[82]-nan,[83]-nan,[84]-nan,[85]-nan,[86]-nan,[87]-nan,[88]-nan,[89]-nan,[90]-nan,[91]-nan,[92]-nan,[93]-nan,[94]-nan,[95]-nan,[96]-nan,[97]-nan,[98]-nan,[99]-nan,[100]-nan,[101]-nan,[102]-nan,[103]-nan,[104]-nan,[105]-nan,[106]-nan,[107]-nan,[108]-nan,[109]-nan,[110]-nan,[111]-nan,[112]-nan,[113]-nan,[114]-nan,[115]-nan,[116]-nan,[117]-nan,[118]-nan,[119]-nan,[120]-nan,[121]-nan,[122]-nan,[123]-nan,[124]-nan,[125]-nan,[126]-nan,[127]-nan,[128]-nan,[129]-nan,[130]-nan,[131]-nan,[132]-nan,[133]-nan,[134]-nan,[135]-nan,[136]-nan,[137]-nan,[138]-nan,[139]-nan,[140]-nan,[141]-nan,[142]-nan,[143]-nan,[144]-nan,[145]-nan,[146]-nan,[147]-nan,[148]-nan,[149]-nan,[150]-nan,[151]-nan,[152]-nan,[153]-nan,[154]-nan,[155]-nan,[156]-nan,[157]-nan,[158]-nan,[159]-nan,[160]-nan,[161]-nan,[162]-nan,[163]-nan,[164]-nan,[165]-nan,[166]-nan,[167]-nan,[168]-nan,[169]-nan,[170]-nan,[171]-nan,[172]-nan,[173]-nan,[174]-nan,[175]-nan,[176]-nan,[177]-nan,[178]-nan,[179]-nan,[180]-nan,[181]-nan,[182]-nan,[183]-nan,[184]-nan,[185]-nan,[186]-nan,[187]-nan,[188]-nan,[189]-nan,[190]-nan,[191]-nan,[192]-nan,[193]-nan,[194]-nan,[195]-nan,[196]-nan,[197]-nan,[198]-nan,[199]-nan,[200]-nan,[201]-nan,[202]-nan,[203]-nan,[204]-nan,[205]-nan,[206]-nan,[207]-nan,[208]-nan,[209]-nan,[210]-nan,[211]-nan,
Unexpected negative standard deviation of log(prob)

llama_print_timings:        load time =    1581.30 ms
llama_print_timings:      sample time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings: prompt eval time =   47678.04 ms / 108032 tokens (    0.44 ms per token,  2265.87 tokens per second)
llama_print_timings:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings:       total time =   52725.87 ms / 108033 tokens
```

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2024-10-23** at **06:27:44**:<br>

Thanks for the report. Happens for me too. I'll investigate.

---

👤 **ikawrakow** commented the **2024-10-23** at **07:09:28**:<br>

@Nexesenex 

This is also broken on mainline `llama.cpp`, no?
With the latest `llama.cpp` (`873279b1592e433c4d9eb5065091cc98473c7bee`) without FA I get NaNs for any of the supported K-cache quantization types.

---

👤 **ikawrakow** commented the **2024-10-23** at **07:43:52**:<br>

CUDA on mainline `llama.cpp` without FA is broken with quantized K-cache for all models I tried (LLaMA-3.1-8B, LLaMA-3.2-3B, LLaMA-2-7B). So, I guess, this issue is inherited. Perhaps you should file a bug report there?

---

👤 **Nexesenex** commented the **2024-10-23** at **07:48:47**:<br>

Indeed, it's on mainline also.
I'll holler them. ^^

---

👤 **ikawrakow** commented the **2024-10-23** at **08:00:33**:<br>

It was puzzling to me why `Q6_0` works here but none of the other types, neither here nor on mainline. But I think I know what is the issue. I haven't implemented a MMQ kernel for `Q6_0`, so the `K*Q` matrix multiplication is done via dequantize `K` -> cuBLAS gemm. While all other types go via @JohannesGaessler MMQ kernels. There have been all these reports about `llama.cpp` producing gibberish for some models, the latest being the Granite models, and the typical fix is to set the `K*Q` matrix multiplication precision to `F32`. Well, if `F16` is not precise enough for `K*Q`, then quantized precision is definitely not precise enough either. So, basically, the issue has existed in mainline `llama.cpp` since @JohannesGaessler switched the default for matrix multiplications to MMQ. Strange that nobody has noticed for so long.

---

👤 **ikawrakow** commented the **2024-10-23** at **08:00:33**:<br>

It was puzzling to me why `Q6_0` works here but none of the other types, neither here nor on mainline. But I think I know what is the issue. I haven't implemented a MMQ kernel for `Q6_0`, so the `K*Q` matrix multiplication is done via dequantize `K` -> cuBLAS gemm. While all other types go via Johannes' MMQ kernels. There have been all these reports about `llama.cpp` producing gibberish for some models, the latest being the Granite models, and the typical fix is to set the `K*Q` matrix multiplication precision to `F32`. Well, if `F16` is not precise enough for `K*Q`, then quantized precision is definitely not precise enough either. So, basically, the issue has existed in mainline `llama.cpp` since Johannes switched the default for matrix multiplications to MMQ. Strange that nobody has noticed for so long.

---

👤 **ikawrakow** commented the **2024-10-23** at **11:16:41**:<br>

Thinking more about this, it is kind of strange. It does work on the CPU, where `Q` gets quantized to `Q8_K` when `K` is quantized, and `Q8_K` is less accurate than `Q8_0` (one float scale per 256 weights for `Q8_K` vs 1 float scale per 32 for `Q8_0`). So, precision/range loss does not seem to be the likely cause. Instead, more likely, there is some other bug in the MMQ kernel that manifests itself only under specific conditions.

---

👤 **ikawrakow** commented the **2024-10-24** at **07:43:55**:<br>

@Nexesenex Does [this PR](https://github.com/ggerganov/llama.cpp/pull/10021) fix it for you? It is approved and all, but I still get NaN's with a quantized model. It does appear to work with the `f16` model, so there is at least some progress.

---

👤 **ikawrakow** commented the **2024-10-24** at **07:43:55**:<br>

@Nexesenex Does [this PR](https://github.com/ggerganov/llama.cpp/pull/10021) fix it for you? It is approved and all, but I still get NaN's.

---

👤 **JohannesGaessler** commented the **2024-10-24** at **09:08:55**:<br>

I also get NaN with a q8_0 model when using `-ctk q8_0`, there are probably multiple bugs.

---

👤 **ikawrakow** commented the **2024-10-24** at **09:16:10**:<br>

> I also get NaN with a q8_0 model when using `-ctk q8_0`, there are probably multiple bugs.

It is not just `q8_0`. Any quantized model with any quantized k-cache without FA produces NaNs on `perplexity` runs. If it helps you, TG appears to work. PP also works if I use `-ub 8` to force the `K*Q` matrix multiplication to go via `MMVQ`.

---

👤 **Nexesenex** commented the **2024-10-24** at **12:39:43**:<br>

@ikawrakow I just confirmed that all K quantum cache no-FA modes present on mainline are now working : https://github.com/ggerganov/llama.cpp/issues/10011#issuecomment-2435180867

I also used https://github.com/ggerganov/llama.cpp/pull/10015 while I was at it.

---

👤 **Nexesenex** commented the **2024-10-24** at **12:39:43**:<br>

@ikawrakow I confirmed it works on master here : https://github.com/ggerganov/llama.cpp/issues/10011#issuecomment-2435180867

I also used https://github.com/ggerganov/llama.cpp/pull/10015 while I was at it.