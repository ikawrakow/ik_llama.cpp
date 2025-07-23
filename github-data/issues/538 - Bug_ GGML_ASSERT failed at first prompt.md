### 🐛 [#538](https://github.com/ikawrakow/ik_llama.cpp/issues/538) - Bug: GGML_ASSERT failed at first prompt

| **Author** | `iehgit` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-18 |
| **Updated** | 2025-06-19 |

---

#### Description

### What happened?

Model seems to load fine, but GGML_ASSERT failed and crash at the first prompt. See log below.

### Name and Version

./build/bin/llama-server --version
version: 3756 (0ade5343)
built with cc (Debian 14.2.0-19) 14.2.0 for x86_64-linux-gnu


### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
./build/bin/llama-server -m /media/raid0/mla/DeepSeek-R1-0528-IQ2_K_R4-00001-of-00005.gguf --host :: -fa -c 16384 -t 16 -mla 3 -fmoe -ctk q8_0
INFO [                    main] build info | tid="140367990282560" timestamp=1750279261 build=3756 commit="0ade5343"
INFO [                    main] system info | tid="140367990282560" timestamp=1750279261 n_threads=16 n_threads_batch=-1 total_threads=32 system_info="AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: additional 4 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 52 key-value pairs and 1147 tensors from /media/raid0/mla/DeepSeek-R1-0528-IQ2_K_R4-00001-of-00005.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = DeepSeek R1 0528
llama_model_loader: - kv   3:                            general.version str              = 0528
llama_model_loader: - kv   4:                           general.basename str              = DeepSeek-R1
llama_model_loader: - kv   5:                         general.size_label str              = 256x21B
llama_model_loader: - kv   6:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv   7:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv   8:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv   9:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  10:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv  11:          deepseek2.attention.head_count_kv u32              = 128
llama_model_loader: - kv  12:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  13: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  14:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  15:                          general.file_type u32              = 338
llama_model_loader: - kv  16:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  17:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  18:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  19:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  20:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  21:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  22:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  23:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  24:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  25:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  26:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  27:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  28:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  29:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  30:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  31: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  32: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  33:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  34:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  35:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<�...
llama_model_loader: - kv  36:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  37:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
llama_model_loader: - kv  38:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  39:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  40:            tokenizer.ggml.padding_token_id u32              = 1
llama_model_loader: - kv  41:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  42:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  43:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  44:               general.quantization_version u32              = 2
llama_model_loader: - kv  45:                      quantize.imatrix.file str              = /mnt/raid/models/ubergarm/DeepSeek-R1...
llama_model_loader: - kv  46:                   quantize.imatrix.dataset str              = ubergarm-imatrix-calibration-corpus-v...
llama_model_loader: - kv  47:             quantize.imatrix.entries_count i32              = 721
llama_model_loader: - kv  48:              quantize.imatrix.chunks_count i32              = 812
llama_model_loader: - kv  49:                                   split.no u16              = 0
llama_model_loader: - kv  50:                                split.count u16              = 5
llama_model_loader: - kv  51:                        split.tensors.count i32              = 1147
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q5_0:   61 tensors
llama_model_loader: - type iq4_ks:  116 tensors
llama_model_loader: - type iq5_ks:  435 tensors
llama_model_loader: - type iq2_k_r4:  116 tensors
llama_model_loader: - type iq3_k_r4:   58 tensors
llm_load_vocab: special tokens cache size = 818
llm_load_vocab: token to piece cache size = 0.8223 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = deepseek2
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 129280
llm_load_print_meta: n_merges         = 127741
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 163840
llm_load_print_meta: n_embd           = 7168
llm_load_print_meta: n_layer          = 61
llm_load_print_meta: n_head           = 128
llm_load_print_meta: n_head_kv        = 128
llm_load_print_meta: n_rot            = 64
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 192
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: n_embd_k_gqa     = 24576
llm_load_print_meta: n_embd_v_gqa     = 16384
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 18432
llm_load_print_meta: n_expert         = 256
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = yarn
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 0.025
llm_load_print_meta: n_ctx_orig_yarn  = 4096
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = 671B
llm_load_print_meta: model ftype      = IQ2_K_R4 - 2.375 bpw
llm_load_print_meta: model params     = 672.050 B
llm_load_print_meta: model size       = 219.019 GiB (2.799 BPW) 
llm_load_print_meta: repeating layers = 217.886 GiB (2.793 BPW, 670.196 B parameters)
llm_load_print_meta: general.name     = DeepSeek R1 0528
llm_load_print_meta: BOS token        = 0 '<｜begin▁of▁sentence｜>'
llm_load_print_meta: EOS token        = 1 '<｜end▁of▁sentence｜>'
llm_load_print_meta: PAD token        = 1 '<｜end▁of▁sentence｜>'
llm_load_print_meta: LF token         = 131 'Ä'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_layer_dense_lead   = 3
llm_load_print_meta: n_lora_q             = 1536
llm_load_print_meta: n_lora_kv            = 512
llm_load_print_meta: n_ff_exp             = 2048
llm_load_print_meta: n_expert_shared      = 1
llm_load_print_meta: expert_weights_scale = 2.5
llm_load_print_meta: expert_weights_norm  = 1
llm_load_print_meta: expert_gating_func   = sigmoid
llm_load_print_meta: rope_yarn_log_mul    = 0.1000
llm_load_tensors: ggml ctx size =    0.47 MiB
llm_load_tensors:        CPU buffer size = 45509.83 MiB
llm_load_tensors:        CPU buffer size = 44388.02 MiB
llm_load_tensors:        CPU buffer size = 45775.72 MiB
llm_load_tensors:        CPU buffer size = 44856.99 MiB
llm_load_tensors:        CPU buffer size = 43745.20 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 16384
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 3
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init:        CPU KV buffer size =   583.31 MiB
llama_new_context_with_model: KV self size  =  583.31 MiB, c^KV (q8_0):  583.31 MiB, kv^T: not used
llama_new_context_with_model:        CPU  output buffer size =     0.99 MiB
llama_new_context_with_model:        CPU compute buffer size =  2778.01 MiB
llama_new_context_with_model: graph nodes  = 3487
llama_new_context_with_model: graph splits = 1
INFO [                    init] initializing slots | tid="140367990282560" timestamp=1750279344 n_slots=1
INFO [                    init] new slot | tid="140367990282560" timestamp=1750279344 id_slot=0 n_ctx_slot=16384
INFO [                    main] model loaded | tid="140367990282560" timestamp=1750279344
INFO [                    main] chat template | tid="140367990282560" timestamp=1750279344 chat_example="You are a helpful assistant\n\n<｜User｜>Hello<｜Assistant｜>Hi there<｜end▁of▁sentence｜><｜User｜>How are you?<｜Assistant｜>" built_in=true
INFO [                    main] HTTP server listening | tid="140367990282560" timestamp=1750279344 n_threads_http="31" port="8080" hostname="::"
INFO [            update_slots] all slots are idle | tid="140367990282560" timestamp=1750279344
INFO [   launch_slot_with_task] slot is processing task | tid="140367990282560" timestamp=1750279395 id_slot=0 id_task=0
INFO [            update_slots] kv cache rm [p0, end) | tid="140367990282560" timestamp=1750279395 id_slot=0 id_task=0 p0=0
/home/user/src/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: /home/user/src/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed/home/user/src/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/home/user/src/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/home/user/src/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
GGML_ASSERT(fms.S[j] > 0) failed

/home/user/src/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/home/user/src/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/home/user/src/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/home/user/src/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/home/user/src/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/home/user/src/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/home/user/src/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/home/user/src/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/home/user/src/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/home/user/src/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
/home/user/src/ik_llama.cpp/ggml/src/./iqk/fa/iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed
[New LWP 16723]
[New LWP 16722]
[New LWP 16721]
[New LWP 16720]
[New LWP 16719]
[New LWP 16718]
[New LWP 16717]
[New LWP 16716]
[New LWP 16715]
[New LWP 16714]
[New LWP 16713]
[New LWP 16712]
[New LWP 16711]
[New LWP 16710]
[New LWP 16709]
[New LWP 16708]
[New LWP 16707]
[New LWP 16706]
[New LWP 16705]
[New LWP 16704]
[New LWP 16703]
[New LWP 16702]
[New LWP 16701]
[New LWP 16700]
[New LWP 16699]
[New LWP 16698]
[New LWP 16697]
[New LWP 16696]
[New LWP 16695]
[New LWP 16694]
[New LWP 16693]
[New LWP 16692]
[New LWP 16691]
[New LWP 16690]
[New LWP 16689]
[New LWP 16688]
[New LWP 16687]
[New LWP 16686]
[New LWP 16685]
[New LWP 16684]
[New LWP 16683]
[New LWP 16682]
[New LWP 16681]
[New LWP 16680]
[New LWP 16679]
[New LWP 16678]
[New LWP 16677]
warning: process 16676 is already traced by process 16727
warning: process 16676 is already traced by process 16727
warning: process 16676 is already traced by process 16727
ptrace: Operation not permitted.ptrace: Operation not permitted.warning: process 16676 is already traced by process 16727
ptrace: Operation not permitted.warning: process 16676 is already traced by process 16727
warning: process 16676 is already traced by process 16727
warning: process 16676 is already traced by process 16727
warning: process 16676 is already traced by process 16727


ptrace: Operation not permitted.
ptrace: Operation not permitted.ptrace: Operation not permitted.ptrace: Operation not permitted.ptrace: Operation not permitted.




No stack.No stack.

No stack.No stack.No stack.

No stack.The program is not being run.The program is not being run.
No stack.No stack.


The program is not being run.

The program is not being run.
The program is not being run.
The program is not being run.

The program is not being run.The program is not being run.

warning: process 16676 is already traced by process 16727
warning: process 16676 is already traced by process 16727
ptrace: Operation not permitted.ptrace: Operation not permitted.

No stack.No stack.

The program is not being run.The program is not being run.

warning: process 16676 is already traced by process 16727
ptrace: Operation not permitted.
No stack.
The program is not being run.
warning: process 16676 is already traced by process 16727
ptrace: Operation not permitted.
No stack.
The program is not being run.
warning: process 16676 is already traced by process 16727
ptrace: Operation not permitted.
No stack.
The program is not being run.
warning: process 16676 is already traced by process 16727
ptrace: Operation not permitted.
No stack.
The program is not being run.
warning: process 16676 is already traced by process 16727
ptrace: Operation not permitted.
No stack.
The program is not being run.
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
0x00007fa9f72a49ee in ?? () from /lib/x86_64-linux-gnu/libc.so.6
#0  0x00007fa9f72a49ee in ?? () from /lib/x86_64-linux-gnu/libc.so.6
#1  0x00007fa9f7299668 in ?? () from /lib/x86_64-linux-gnu/libc.so.6
#2  0x00007fa9f72996ad in ?? () from /lib/x86_64-linux-gnu/libc.so.6
#3  0x00007fa9f7304787 in wait4 () from /lib/x86_64-linux-gnu/libc.so.6
#4  0x00007fa9f781a608 in ggml_abort () from /home/user/src/ik_llama.cpp/build/ggml/src/libggml.so
#5  0x00007fa9f78d4c05 in void (anonymous namespace)::FlashQKV<512, 8, 32>::normalize_and_store_1row<(anonymous namespace)::FlashMS<8, 32> >((anonymous namespace)::FlashMS<8, 32> const&, int, float const*, float*) const [clone .part.0] () from /home/user/src/ik_llama.cpp/build/ggml/src/libggml.so
#6  0x00007fa9f78e2f88 in void (anonymous namespace)::iqk_deepseek_helper<32, (anonymous namespace)::HelperQ80R8<576>, (anonymous namespace)::HelperQ80>((anonymous namespace)::HelperQ80R8<576>&, (anonymous namespace)::HelperQ80&, int, int, int, int, int, float const*, char const*, float, float, float*, float*, float*) [clone .constprop.0] () from /home/user/src/ik_llama.cpp/build/ggml/src/libggml.so
#7  0x00007fa9f78e6f64 in bool (anonymous namespace)::iqk_deepseek_helper<32>(ggml_type, int, int, int, int, int, int, int, float const*, char const*, char const*, char const*, float, float, float*, float*, float*) () from /home/user/src/ik_llama.cpp/build/ggml/src/libggml.so
#8  0x00007fa9f78ce9d2 in iqk_flash_attn_noalibi () from /home/user/src/ik_llama.cpp/build/ggml/src/libggml.so
#9  0x00007fa9f7824693 in ggml_compute_forward_flash_attn_ext_f16 () from /home/user/src/ik_llama.cpp/build/ggml/src/libggml.so
#10 0x00007fa9f785b1f9 in ggml_graph_compute_thread.constprop.0.isra () from /home/user/src/ik_llama.cpp/build/ggml/src/libggml.so
#11 0x00007fa9f785b395 in ggml_graph_compute._omp_fn () from /home/user/src/ik_llama.cpp/build/ggml/src/libggml.so
#12 0x00007fa9f8349fe6 in GOMP_parallel () from /lib/x86_64-linux-gnu/libgomp.so.1
#13 0x00007fa9f785ef30 in ggml_graph_compute () from /home/user/src/ik_llama.cpp/build/ggml/src/libggml.so
#14 0x00007fa9f786c352 in ggml_backend_cpu_graph_compute () from /home/user/src/ik_llama.cpp/build/ggml/src/libggml.so
#15 0x00007fa9f7871873 in ggml_backend_sched_graph_compute_async () from /home/user/src/ik_llama.cpp/build/ggml/src/libggml.so
#16 0x00007fa9f85498e1 in llama_decode () from /home/user/src/ik_llama.cpp/build/src/libllama.so
#17 0x0000559092821e65 in server_context::update_slots() ()
#18 0x00005590927f0fbc in server_queue::start_loop() ()
#19 0x00005590927913de in main ()
[Inferior 1 (process 16676) detached]
Aborted
```

---

#### 💬 Conversation

👤 **ubergarm** commented the **2025-06-19** at **01:26:22**:<br>

Hrmm... I'm getting something odd now too with my `DeepSeek-R1-0528-IQ4_KS_R4` as well as mostly pure models.

This commit is working fine for me: dc96820d

However, trying commit c410cc72 throws this on startup when compiled CPU only: 

`Oops(ggml_compute_forward_sum_rows_f32, ffn_moe_weights_sum-3): found -nan for i1 = 0, i2 = 0, i3 = 0. ne00 = 256`

@iehgit 

You might try `git checkout dc96820d` and re-build to see if that gets you for now, maybe?

---

👤 **ikawrakow** commented the **2025-06-19** at **06:36:55**:<br>

Is it fixed on the latest after #540?

---

👤 **ubergarm** commented the **2025-06-19** at **15:35:21**:<br>

I recompiled to tip of main 3f111ad7 which includes PR540.

Confirmed it is working again for me and no longer throwing the `Oops(ggml_compute_forward_sum_rows_f32` from before.

---

👤 **iehgit** commented the **2025-06-19** at **16:54:14**:<br>

Fixed indeed. Thanks!