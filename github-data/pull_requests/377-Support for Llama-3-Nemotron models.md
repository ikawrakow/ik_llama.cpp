### 🔀 [#377](https://github.com/ikawrakow/ik_llama.cpp/pull/377) - Support for Llama-3-Nemotron models

| **Author** | `saood06` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-04 |
| **Updated** | 2025-05-09 |

---

#### Description

Port of https://github.com/ggml-org/llama.cpp/pull/10669

It compiles, have not tested yet. Testers welcome, but will try to test myself later.

---

#### 💬 Conversation

👤 **saood06** commented the **2025-05-04** at **12:31:11**:<br>

I downloaded the source model and was able to convert it with `convert_hf_to_gguf.py` but I hit an error when attempting to quantize it.

`llama.cpp:19268: GGML_ASSERT((qs.n_attention_wv == 0 || qs.n_attention_wv == (int)model.hparams.n_layer || qs.n_attention_wv == 3 * (int)model.hparams.n_layer) && "n_attention_wv is unexpected") failed`

---

👤 **ikawrakow** commented the **2025-05-04** at **12:38:47**:<br>

Well, you see what `n_attention_wv` is and add another rule for accepting it. This is because of the layers that don't have the usual attention mechanism, I guess.

---

👤 **saood06** commented the **2025-05-04** at **13:02:38**:<br>

It's quantizing now.

---

👤 **ikawrakow** commented the **2025-05-04** at **13:10:46**:<br>

Apart from the 253B version that is beyond my reach, this will add support for this model: https://huggingface.co/nvidia/Llama-3_1-Nemotron-51B-Instruct ?

What about https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1 which seems more recent?

---

👤 **saood06** commented the **2025-05-04** at **13:14:52**:<br>

> Apart from the 253B version that is beyond my reach

Support for that is not added yet, that one is missing https://github.com/ggml-org/llama.cpp/pull/12843

> this will add support for this model: https://huggingface.co/nvidia/Llama-3_1-Nemotron-51B-Instruct ?

That is the one I am testing with right now.

> What about https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1 which seems more recent?

That one should work (maybe the convert python might not?) but you may need to add the n_attention_wv value if it is different.

---

👤 **saood06** commented the **2025-05-04** at **13:18:16**:<br>

It is coherent in the cli.

Will sweep-bench it later.

---

👤 **ikawrakow** submitted a review the **2025-05-04** at **14:02:57**: 💬 `COMMENTED`

---

👤 **ikawrakow** commented the **2025-05-04** at **14:05:15**:<br>

I get this error when I try to run the [49B model](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1) (after adjusting the `n_attention_vw` check):
```
llama_model_load: error loading model: error loading model vocabulary: cannot find tokenizer merges in model file
```

---

👤 **ikawrakow** commented the **2025-05-04** at **14:16:43**:<br>

Works if I convert with mainline, so something is missing in the conversion script.

---

👤 **saood06** submitted a review the **2025-05-04** at **14:19:07**: 💬 `COMMENTED`

---

👤 **saood06** commented during a code review the **2025-05-04** at **14:19:07** on `src/llama.cpp`:<br>

Sorry I didn't notice these. They are in the original PR as well (which I cherry-picked as it was from when they hadn't diverged too much), I'll take them out. Right now I'm working on the larger model as that can't be cherry-picked

---

👤 **saood06** commented the **2025-05-04** at **14:19:52**:<br>

> Works if I convert with mainline, so something is missing in the conversion script.

Thanks for testing that, I'll look into the script.

---

👤 **saood06** commented the **2025-05-04** at **15:23:42**:<br>

@Lissanro

Can you try Llama-3_1-Nemotron-Ultra-253B now, the n_attention_vw check may be broken but everything else I think should be fine.

---

👤 **ikawrakow** commented the **2025-05-04** at **15:29:05**:<br>

> the n_attention_vw check may be broken but everything else I think should be fine.

Oh, I forgot to comment on that one. I solved it for the 49B model by simply accepting `n_attention_vw` if `model.arch == LLM_ARCH_DECI`. In that way we don't need to adjust that check for every variation they may come up with.

---

👤 **saood06** commented the **2025-05-04** at **15:41:08**:<br>

@ikawrakow 

Can you test the conversion again? This is good to review again, I'm done pushing changes.

---

👤 **ikawrakow** commented the **2025-05-04** at **15:48:16**:<br>

I'm running something on the computer where I downloaded the model. I'll test in a bit when the run finishes.

---

👤 **saood06** commented the **2025-05-04** at **15:52:22**:<br>

>I'll test in a bit when the run finishes.

Take your time, I'm heading off for now anyways.

---

👤 **Panchovix** commented the **2025-05-04** at **19:39:28**:<br>

Thanks for the work! I'm trying L3 Nemotron 253B Q3_K_XL from unsloth (https://huggingface.co/unsloth/Llama-3_1-Nemotron-Ultra-253B-v1-GGUF/tree/main/UD-Q3_K_XL), here how is the log looks

```
pancho@fedora:/run/media/pancho/6AE20D1AE20CEBDF/ChatIAs/ik_llama.cpp/build_linux/bin$ ./llama-server -m /run/media/pancho/08329F4A329F3B9E/models_llm/Llama-3_1-Nemotron-Ultra-253B-v1-UD-Q3_K_XL-00001-of-00003.gguf -c 12228 -ngl 163 -ts 6.5,6,10,4 --no-warmup -fa -ctk q8_0 -ctv q4_0 -mg 2
INFO [                    main] build info | tid="139738867924992" timestamp=1746386578 build=3671 commit="0e001215"
INFO [                    main] system info | tid="139738867924992" timestamp=1746386578 n_threads=8 n_threads_batch=-1 total_threads=16 system_info="AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: additional 2 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 43 key-value pairs and 648 tensors from /run/media/pancho/08329F4A329F3B9E/models_llm/Llama-3_1-Nemotron-Ultra-253B-v1-UD-Q3_K_XL-00001-of-00003.gguf (version GGUF V3 (latest))
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
llama_model_loader: - kv  35:                          general.file_type u32              = 12
llama_model_loader: - kv  36:                      quantize.imatrix.file str              = Llama-3_1-Nemotron-Ultra-253B-v1-GGUF...
llama_model_loader: - kv  37:                   quantize.imatrix.dataset str              = unsloth_calibration_Llama-3_1-Nemotro...
llama_model_loader: - kv  38:             quantize.imatrix.entries_count i32              = 499
llama_model_loader: - kv  39:              quantize.imatrix.chunks_count i32              = 544
llama_model_loader: - kv  40:                                   split.no u16              = 0
llama_model_loader: - kv  41:                        split.tensors.count i32              = 648
llama_model_loader: - kv  42:                                split.count u16              = 3
llama_model_loader: - type  f32:  147 tensors
llama_model_loader: - type q3_K:  162 tensors
llama_model_loader: - type q4_K:  326 tensors
llama_model_loader: - type q5_K:   13 tensors
llm_load_vocab: special tokens cache size = 256
llm_load_vocab: token to piece cache size = 0.7999 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = deci
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 128256
llm_load_print_meta: n_merges         = 280147
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 131072
llm_load_print_meta: n_embd           = 16384
llm_load_print_meta: n_layer          = 162
llm_load_print_meta: n_head           = [128, 128, 128, 128, 128, 128, 128, 128, 128, 0, 0, 0, 0, 128, 128, 128, 128, 128, 0, 0, 0, 0, 0, 0, 128, 128, 128, 0, 0, 0, 0, 0, 128, 128, 128, 128, 0, 0, 0, 128, 128, 128, 0, 128, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 0, 0, 0, 0, 0, 128, 128, 128, 128, 0, 0, 0, 0, 0, 128, 128, 128, 128, 0, 0, 0, 0, 0, 128, 128, 128, 128, 0, 0, 0, 0, 0, 128, 128, 128, 128, 0, 0, 128, 128, 128, 128, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 128, 128, 0, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 0, 128, 128, 128, 128, 128, 128, 128, 128]
llm_load_print_meta: n_head_kv        = [8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 8, 8, 8, 0, 0, 0, 0, 0, 8, 8, 8, 8, 0, 0, 0, 8, 8, 8, 0, 8, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 0, 0, 0, 0, 0, 8, 8, 8, 8, 0, 0, 0, 0, 0, 8, 8, 8, 8, 0, 0, 0, 0, 0, 8, 8, 8, 8, 0, 0, 0, 0, 0, 8, 8, 8, 8, 0, 0, 8, 8, 8, 8, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 8, 8, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 0, 8, 8, 8, 8, 8, 8, 8, 8]
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = [16, 16, 16, 16, 16, 16, 16, 16, 16, 0, 0, 0, 0, 16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 16, 16, 16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 0, 0, 0, 16, 16, 16, 0, 16, 0, 0, 0, 0, 0, 0, 16, 16, 16, 16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 0, 0, 16, 16, 16, 16, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 16, 16, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 16, 0, 16, 16, 16, 16, 16, 16, 16, 16]
llm_load_print_meta: n_embd_k_gqa     = [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 1024, 0, 0, 0, 0, 0, 0, 1024, 1024, 1024, 0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 0, 0, 0, 1024, 1024, 1024, 0, 1024, 0, 0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 0, 0, 1024, 1024, 1024, 1024, 0, 0, 1024, 0, 0, 0, 0, 0, 0, 0, 0, 1024, 0, 0, 0, 0, 0, 1024, 1024, 0, 1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1024, 1024, 0, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024]
llm_load_print_meta: n_embd_v_gqa     = [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 1024, 0, 0, 0, 0, 0, 0, 1024, 1024, 1024, 0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 0, 0, 0, 1024, 1024, 1024, 0, 1024, 0, 0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 0, 0, 1024, 1024, 1024, 1024, 0, 0, 1024, 0, 0, 0, 0, 0, 0, 0, 0, 1024, 0, 0, 0, 0, 0, 1024, 1024, 0, 1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1024, 1024, 0, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024]
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = [5376, 10752, 16128, 16128, 16128, 16128, 16128, 16128, 21504, 0, 0, 0, 0, 21504, 21504, 21504, 53248, 53248, 0, 0, 0, 0, 0, 0, 53248, 53248, 53248, 0, 0, 0, 0, 0, 53248, 53248, 53248, 26624, 0, 0, 0, 21504, 21504, 21504, 21504, 53248, 53248, 0, 0, 0, 0, 0, 53248, 53248, 53248, 53248, 0, 0, 0, 0, 0, 53248, 53248, 53248, 53248, 0, 0, 0, 0, 0, 53248, 53248, 53248, 53248, 0, 0, 0, 0, 0, 53248, 53248, 53248, 53248, 0, 0, 0, 0, 0, 53248, 37376, 37376, 37376, 0, 0, 32000, 26624, 26624, 26624, 26624, 26624, 26624, 0, 26624, 26624, 26624, 26624, 26624, 26624, 26624, 26624, 0, 0, 0, 0, 0, 32000, 53248, 53248, 53248, 0, 0, 0, 0, 0, 0, 0, 0, 399360, 0, 0, 0, 0, 0, 0, 0, 0, 425984, 0, 0, 0, 0, 0, 0, 0, 0, 343040, 0, 0, 0, 0, 0, 301056, 21504, 21504, 26624, 0, 26624, 26624, 37376, 53248, 53248, 53248, 53248, 26624]
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
llm_load_print_meta: model type       = 405B
llm_load_print_meta: model ftype      = Q3_K - Medium
llm_load_print_meta: model params     = 253.401 B
llm_load_print_meta: model size       = 115.764 GiB (3.924 BPW) 
llm_load_print_meta: repeating layers = 113.318 GiB (3.906 BPW, 249.199 B parameters)
llm_load_print_meta: general.name     = Llama_Nemotron_Ultra
llm_load_print_meta: BOS token        = 128000 '<|begin_of_text|>'
llm_load_print_meta: EOS token        = 128009 '<|eot_id|>'
llm_load_print_meta: LF token         = 128 'Ä'
llm_load_print_meta: EOT token        = 128009 '<|eot_id|>'
llm_load_print_meta: max token length = 256
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 4 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
  Device 1: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
  Device 2: NVIDIA RTX A6000, compute capability 8.6, VMM: yes
  Device 3: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
llm_load_tensors: ggml ctx size =    1.99 MiB
llm_load_tensors: offloading 162 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 163/163 layers to GPU
llm_load_tensors:        CPU buffer size =  1127.25 MiB
llm_load_tensors:      CUDA0 buffer size = 21995.70 MiB
llm_load_tensors:      CUDA1 buffer size = 22587.26 MiB
llm_load_tensors:      CUDA2 buffer size = 45199.39 MiB
llm_load_tensors:      CUDA3 buffer size = 27632.88 MiB
.....................................................................................
llama_new_context_with_model: n_ctx      = 12288
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 0
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 500000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:      CUDA0 KV buffer size =   429.02 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =   292.52 MiB
llama_kv_cache_init:      CUDA2 KV buffer size =   331.53 MiB
llama_kv_cache_init:      CUDA3 KV buffer size =   195.01 MiB
llama_new_context_with_model: KV self size  = 1248.00 MiB, K (q8_0):  816.00 MiB, V (q4_0):  432.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.98 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      CUDA0 compute buffer size =   412.00 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =   420.00 MiB
llama_new_context_with_model:      CUDA2 compute buffer size =  2560.00 MiB
llama_new_context_with_model:      CUDA3 compute buffer size =  2086.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =    56.01 MiB
llama_new_context_with_model: graph nodes  = 1708
llama_new_context_with_model: graph splits = 5
INFO [                    init] initializing slots | tid="139738867924992" timestamp=1746386887 n_slots=1
INFO [                    init] new slot | tid="139738867924992" timestamp=1746386887 id_slot=0 n_ctx_slot=12288
INFO [                    main] model loaded | tid="139738867924992" timestamp=1746386887
INFO [                    main] chat template | tid="139738867924992" timestamp=1746386887 chat_example="<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nHi there<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHow are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" built_in=true
INFO [                    main] HTTP server listening | tid="139738867924992" timestamp=1746386887 n_threads_http="15" port="8080" hostname="127.0.0.1"
INFO [            update_slots] all slots are idle | tid="139738867924992" timestamp=1746386887
```

And it seems to work without issues

![image](https://github.com/user-attachments/assets/7c8f4a1b-1b05-4af8-99c4-736238fe9bad)

Not sure if there's a flag that could improve things for dense models. Also not exactly sure how to enable thinking, but maybe that depends of the UI when using it via API.

---

👤 **ikawrakow** commented the **2025-05-05** at **06:58:29**:<br>

With the commit that I just pushed `convert_hf_to_gguf.py` now converts the [Nemotron-Super-49B](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1) model correctly.

But then I see a difference in PPL.

I didn't run the `bf16` model directly (comes dangerously close to the total RAM I have), but using `Q8_0` quantization. I arrive at a lower PPL using the HF->GGUF conversion script in this PR compared to using mainline conversion:
* `PPL = 7.0801` using mainline HF->GGUF
* `PPL = `7.0347`  using this PR HF->GGUF

Quantization is done in exactly the same way, I'm running with exact same parameters on the same hardware, so something else is different in the converted `bf16` models (and just simple `diff` tells me that the files differ).

---

👤 **ikawrakow** submitted a review the **2025-05-05** at **13:14:51**: ✅ `APPROVED`<br>

From my perspective this is ready to merge.
Just waiting for @Lissanro to confirm that it is working for them.

---

👤 **Lissanro** commented the **2025-05-05** at **15:18:43**:<br>

I tried at first using this command:

```
~/pkgs/ik_llama.cpp/build/bin/llama-server \
--model /mnt/secondary/neuro/Llama-3_1-Nemotron-Ultra-253B-v1-GGUF-UD-Q4_K_XL-131072seq/Llama-3_1-Nemotron-Ultra-253B-v1-UD-Q4_K_XL-00001-of-00004.gguf \
--ctx-size 81920 --n-gpu-layers 36 --tensor-split 25,25,25,25 \
-fa -ctk q8_0 -ctv q8_0 --threads 64 --host 0.0.0.0 --port 5000 --split-mode row
```

It loaded successfully, but when trying inference I got this error:

```
CUDA error: an illegal memory access was encountered
  current device: 0, in function ggml_backend_cuda_synchronize at /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu:3054
  cudaStreamSynchronize(cuda_ctx->stream())
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu:110: CUDA error
```

With 12 layers on GPU the error is the same (loads fine, but crashes when I try to use it). If I remove `--split-mode row`, it also results with the same error.

As a last resort, I tried to load only on CPU (`CUDA_VISIBLE_DEVICES="" is necessary otherwise it still tries to use CUDA):
`
```
CUDA_VISIBLE_DEVICES="" ~/pkgs/ik_llama.cpp/build/bin/llama-server \
--model /mnt/secondary/neuro/Llama-3_1-Nemotron-Ultra-253B-v1-GGUF-UD-Q4_K_XL-131072seq/Llama-3_1-Nemotron-Ultra-253B-v1-UD-Q4_K_XL-00001-of-00004.gguf \
--ctx-size 81920 -fa -ctk q8_0 -ctv q8_0 --threads 64 --host 0.0.0.0 --port 5000
```

...then at first I thought it worked. So it seems there is an issue specific to CUDA, but CPU-only mode works. Please let me know if additional debugging from my side could help, and if so what steps I need to follow.

---

👤 **ikawrakow** commented the **2025-05-05** at **15:23:43**:<br>

Can you try
```
~/pkgs/ik_llama.cpp/build/bin/llama-server \
--model /mnt/secondary/neuro/Llama-3_1-Nemotron-Ultra-253B-v1-GGUF-UD-Q4_K_XL-131072seq/Llama-3_1-Nemotron-Ultra-253B-v1-UD-Q4_K_XL-00001-of-00004.gguf \
--ctx-size 81920 --n-gpu-layers 36 --tensor-split 25,25,25,25 \
-fa -ctk q8_0 -ctv q8_0 --threads 64 --host 0.0.0.0 --port 5000 -fmoe
```
Thanks.

---

👤 **saood06** commented the **2025-05-05** at **22:20:08**:<br>

> With the commit that I just pushed `convert_hf_to_gguf.py` now converts the [Nemotron-Super-49B](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1) model correctly.

Nice, I see you grabbed the only changes to the vocab.py file that we were behind: https://github.com/ggml-org/llama.cpp/commit/8ba38584b2bf744814e1131f6f6aec97df5a57e1 and https://github.com/ggml-org/llama.cpp/commit/a686171ea71ed8cb8a324850d146cb65a001e141. I think you might have been able to cherry-pick those commits directly. 
> 
> But then I see a difference in PPL.
> 
> I didn't run the `bf16` model directly (comes dangerously close to the total RAM I have), but using `Q8_0` quantization. I arrive at a lower PPL using the HF->GGUF conversion script in this PR compared to using mainline conversion:
> 
>     * `PPL = 7.0801` using mainline HF->GGUF
> 
>     * `PPL = 7.0347`  using this PR HF->GGUF
> 
> 
> Quantization is done in exactly the same way, I'm running with exact same parameters on the same hardware, so something else is different in the converted `bf16` models (and just simple `diff` tells me that the files differ).
> 
> OK, doing `diff` on the logs, I see this difference:
> 
> ```
> llama_model_loader: - type  f32:  131 tensors   (mainline)
> vs
> llama_model_loader: - type  f32:  130 tensors   (this PR)
> ```

Interesting, do you mind checking with gguf-hash or some other tool if that one changed tensor is the only difference? I am curious to know why this PR does one tensor less of f32 than mainline.

---

👤 **Lissanro** commented the **2025-05-05** at **22:48:49**:<br>

> Can you try
> ~/pkgs/ik_llama.cpp/build/bin/llama-server \
> --model /mnt/secondary/neuro/Llama-3_1-Nemotron-Ultra-253B-v1-GGUF-UD-Q4_K_XL-131072seq/Llama-3_1-Nemotron-Ultra-253B-v1-UD-Q4_K_XL-00001-of-00004.gguf \
> --ctx-size 81920 --n-gpu-layers 36 --tensor-split 25,25,25,25 \
> -fa -ctk q8_0 -ctv q8_0 --threads 64 --host 0.0.0.0 --port 5000 -fmoe

Sure, here is the full log: https://pastebin.com/TjqnExDv - it loaded fine, then when I attempted inference it crashed with this error:

```
CUDA error: an illegal memory access was encountered
  current device: 0, in function ggml_backend_cuda_synchronize at /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu:3054
  cudaStreamSynchronize(cuda_ctx->stream())
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu:110: CUDA error
```

---

👤 **Panchovix** commented the **2025-05-06** at **15:26:40**:<br>

Correct me if I'm wrong but isn't nemotron 253B a dense model? So no experts and such

---

👤 **ikawrakow** commented the **2025-05-06** at **15:35:26**:<br>

> Correct me if I'm wrong but isn't nemotron 253B a dense model? So no experts and such

Oops, I'm getting confused. Doing too many things at a time. Not sure then why partial offload is not working.

---

👤 **saood06** commented the **2025-05-07** at **01:47:18**:<br>

> I used `gguf-dump.py`, and the missing tensor is `rope_freqs`.

I'm not sure why it is missing (and whether it would cause worse quality at long contexts), the conversion script looks like it handles it. 

I can see that tensor in gguf's that are on huggingface for these models, so it does seem like it should be there.
 
> The other difference is that ours is `general.file_type = 24`, while theirs is `general.file_type = 32`. I don't know what that means.

This one I understand, they both map to MOSTLY_BF16 ([ik_llama.cpp source](https://github.com/ikawrakow/ik_llama.cpp/blob/6c23618ca5d680bd00f06a143dc4a1b386c827e3/gguf-py/gguf/constants.py#L1327C5-L1327C28) and [llama.cpp source](https://github.com/ggml-org/llama.cpp/blob/141a908a59bbc68ceae3bf090b850e33322a2ca9/gguf-py/gguf/constants.py#L2117)).

---

👤 **Lissanro** commented the **2025-05-07** at **20:37:15**:<br>

If there is still something I need to test, please let me know (my understanding the last command was given under assumption it was MoE, but since it is a dense model, I assume I need some other command to test or maybe I already provided all debug info that is possible from my side). In any case, thank you very much for looking into this.

---

👤 **ikawrakow** commented the **2025-05-09** at **07:09:55**:<br>

I think I'll merge this one despite the missing `rope_freqs` tensors. We can try to sort out later why is it missing if people find performance degradation with long context.

---

👤 **saood06** commented the **2025-05-09** at **07:54:55**:<br>

> I think I'll merge this one despite the missing `rope_freqs` tensors. We can try to sort out later why is it missing if people find performance degradation with long context.

I think I figured it out (or at least I found one reason why it is missing if it turns out there is more), I'll make a PR later (heading off for a bit right now).