### 🗣️ [#242](https://github.com/ikawrakow/ik_llama.cpp/discussions/242) - Switching from llama.cpp/ktransformers, seeking advice/guidance

| **Author** | `ThomasBaruzier` |
| :--- | :--- |
| **Created** | 2025-03-05 |
| **Updated** | 2025-03-15 |

---

#### Description

Hello,

I discovered this repo today, and I'm very excited to try all the new features and optimizations made here.

I am currently downloading R1 BF16 (can't convert using 3090, lack of fp8 support), and in the meantime, I am trying to learn as much as possible.

The goal is to run R1 with a reasonable PPL using 72GB VRAM and 128 GB RAM. Looking at the PRs and comments, the new IQ1_S_R4 (https://github.com/ikawrakow/ik_llama.cpp/pull/185) and IQ1_M_R4 (https://github.com/ikawrakow/ik_llama.cpp/pull/187) quants look really promising, as well as all the fancy stuff related to MLA and context cache (https://github.com/ikawrakow/ik_llama.cpp/pull/208, https://github.com/ikawrakow/ik_llama.cpp/pull/240, https://github.com/ikawrakow/ik_llama.cpp/pull/241, ...), but it's a bit overwhelming at first glance.

I guess that the best option right now is to run one of these R4 quants, writing rules that are equivalent to a Ktransformers config for partial offload of critical sections of the model (https://github.com/ikawrakow/ik_llama.cpp/pull/232), and try poking around with `--mla` values. For cache, I guess I can play with the new Q8_KV if applicable. Regarding CUDA, MLA and/or FA, I am sure what is compatible for CPU / GPU / multi GPU, what combinations of parameters could work.

Do you have any advice regarding this type of setup? Is there a way to use more VRAM by selectively offloading individual experts/layers? If I read it right, R4 quants do not support offloading yet. Are there other tweaks or resources I can learn from to try and use your work as efficiently as possible?

I'd be happy to share my benchmarks and params when I am done quanting the model.

Thank you very much

---

#### 🗣️ Discussion

👤 **ikawrakow** replied the **2025-03-06** at **06:01:05**:<br>

Is the 72 GB VRAM from 3 x 24 GB GPUs?

You setup is somewhat unusual as you "only" have 128 GB of RAM. If you want to use a ready model your only option would be the `IQ1_S` or `IQ1_M` models from Unsloth. The next step up is already too big for the 200 GB you have available.

If you are willing to do your custom quantization, it will require a manual setup as there isn't an out-of-the-box mix to best take advantage of your amount of RAM+VRAM. I guess, I should add a similar functionality as the tensor overrides from #232 also to `llama-quantize` so people don't need to go and change the code to get the quantization mix they want.

Once you have a model that you want to use, I think the best way to distribute the model weights between CPU RAM and GPU VRAM will be to use several `-ot` command line arguments. But to determine the regular expressions required one needs to know the quantization types (and hence sizes) of all tensors.

What is the CPU in this system?

> 👤 **ThomasBaruzier** replied the **2025-03-06** at **14:02:48**:<br>
> Yes, I have 3xRTX 3090 and a Ryzen 9 5950x.
> 
> > If you want to use a ready model
> 
> I don't mind making quants; that's why I wanted to try the 1bit R4 quants that are supposedly superior to unsloth's versions. Surprisingly, I got IQ2_XXS dynamic working with 4k context without mmap at around 3tok/s with llama.cpp thanks to efficient splitting and no GPU compute buffers by setting `-b 31` and `-ub 31`. This way, each GPU uses the exact same amount of VRAM, making use of 98-99% of the 24GB. So in theory, there is a bit of headroom to play with if I do custom quants.
> 
> > I guess, I should add a similar functionality as the tensor overrides from #232 also to llama-quantize so people don't need to go and change the code to get the quantization mix they want.
> 
> This would be very useful. There was a PR on llama.cpp that accomplished this purpose but never got merged: https://github.com/ggml-org/llama.cpp/pull/6844#issuecomment-2423363813
> 
> > I think the best way to distribute the model weights between CPU RAM and GPU VRAM will be to use several -ot command line arguments.
> 
> So a custom quant mixing offloadable and non offloadable quant types and using `-ot` select what is able to run on GPUs, as well as the other components offloaded by Ktransformers (it's only like 16 GB for 180 GB models)?

---

👤 **ikawrakow** replied the **2025-03-07** at **12:00:58**:<br>

PR #244 has been merged, so hopefully this will help you with making your custom DeepSeekR1 quantization.

The `-b 31 -ub 31` option is a clever hack, but I expect prompt processing performance to be unacceptably low. So will be TG with any significant context (more than a few hundred tokens). Or not?

> 👤 **ThomasBaruzier** replied the **2025-03-07** at **16:03:24**:<br>
> This is very cool, thank you for this.
> 
> I did not properly measure the performance impact of `-b 31 -ub 31`, it was a quick test. The logic was that the compute will be slower, but the model read access will be faster. Will report back.

---

👤 **ikawrakow** replied the **2025-03-07** at **15:16:11**:<br>

Could the following work in your 3x24 GiB VRAM + 128 GiB RAM:

* The first 3 dense layers + `output.weight` + all attention tensors + all shared experts on GPU0. If you quantize of of these with `Q6_K` or `Q5_K`, this will use 12.2 GiB or 10.2 GiB of VRAM. This will allow you to use longer contexts. If you don't need the longer context, you can add 2-3 MoE experts layers to GPU0.
* Let's assume you decide to put 2 extra layers on GPU0. The first MoE layers are very important, so I would use `IQ4_XS` for `ffn_down_exps` and `IQ2_XXS` for `ffn_up/gate_exps`. This uses 3.664 GiB per layer, so with the 10.24 GiB from above using `Q5_K` you have used up 17.57 GiB on GPU0. 6.5 remaining GiB is still plenty for KV cache and compute buffer if you use `mla = 2` for attention. 
* 7 MoE layers (layers 5-11) on GPU1 where `ffn_down_exps` is quantized with `IQ3_XXS`, and `ffn_gate_exps` and `ffn_up_exps` with `IQ2_XXS`.  This uses 22.3 GiB of VRAM, so ~1.5 GiB are left for compute buffers so you don't need `-b 31 -ub 31`
* Another 7 MoE layers (layers 12-18) done the same way on GPU2 (not 100% sure about that, it might be that it is better to put the last 7 layers on GPU2. From past experience using more bits on the last few layers improved some models).
* You are now left with 42 layers for the 128 GiB of RAM to be processed by the CPU. If you use `IQ2_K` for `ffn_down_exps` and `IQ2_XXS` for `ffn_up/gate_exps`, this is 2.844 GiB per layer, so 119.44 GiB in total. 

Oh, forgot. The tensors that go on the CPU should be quantized to the corresponding `_R4`  variant. You can decide to not quantize to `*_R4` and then use run time repacking (`-rtr`) to repack to `_R4`, but this adds quite a bit of extra loading time (2-3 minutes on a 32-core EPYC).

> 👤 **ThomasBaruzier** replied the **2025-03-07** at **17:26:56**:<br>
> I couldn't be more grateful. I will try this custom quant as soon as the imatrix is done.
> 
> Speaking of imatrix, I have some weird log outputs, am I doing something wrong?
> 
> `CMD | './ik_llama.cpp/llama-imatrix' -m '/home/user/nvme/gguf/DeepSeek-R1/DeepSeek-R1-F16.gguf' -o '/home/user/nvme/gguf/DeepSeek-R1/imatrix.dat' -f '/home/user/files/ai/quants/misc/calibration_datav3.txt' -ngl 3 -b 31 -ub 31`
> 
> For instance: `save_imatrix: entry ' blk.8.ffn_down_exps.weight' has partial data (96.09%) 10 out of 256 experts are missing data Storing **but be aware**`
> 
> Or a bit more concerning: `[14]-nan,`: PPL is logged until pass 9, then it is reported as `nan`.
> 
> <details>
> <summary>Full log</summary>
> 
> ```
> llama_model_loader: loaded meta data with 44 key-value pairs and 1147 tensors from /home/user/nvme/gguf/DeepSeek-R1/DeepSeek-R1-F16.gguf (version GGUF V3 (latest))
> llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
> llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
> llama_model_loader: - kv   1:                               general.type str              = model
> llama_model_loader: - kv   2:                               general.name str              = DeepSeek R1 Bf16
> llama_model_loader: - kv   3:                         general.size_label str              = 256x21B
> llama_model_loader: - kv   4:                               general.tags arr[str,1]       = ["text-generation"]
> llama_model_loader: - kv   5:                      deepseek2.block_count u32              = 61
> llama_model_loader: - kv   6:                   deepseek2.context_length u32              = 163840
> llama_model_loader: - kv   7:                 deepseek2.embedding_length u32              = 7168
> llama_model_loader: - kv   8:              deepseek2.feed_forward_length u32              = 18432
> llama_model_loader: - kv   9:             deepseek2.attention.head_count u32              = 128
> llama_model_loader: - kv  10:          deepseek2.attention.head_count_kv u32              = 128
> llama_model_loader: - kv  11:                   deepseek2.rope.freq_base f32              = 10000.000000
> llama_model_loader: - kv  12: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
> llama_model_loader: - kv  13:                deepseek2.expert_used_count u32              = 8
> llama_model_loader: - kv  14:                          general.file_type u32              = 1
> llama_model_loader: - kv  15:        deepseek2.leading_dense_block_count u32              = 3
> llama_model_loader: - kv  16:                       deepseek2.vocab_size u32              = 129280
> llama_model_loader: - kv  17:            deepseek2.attention.q_lora_rank u32              = 1536
> llama_model_loader: - kv  18:           deepseek2.attention.kv_lora_rank u32              = 512
> llama_model_loader: - kv  19:             deepseek2.attention.key_length u32              = 192
> llama_model_loader: - kv  20:           deepseek2.attention.value_length u32              = 128
> llama_model_loader: - kv  21:       deepseek2.expert_feed_forward_length u32              = 2048
> llama_model_loader: - kv  22:                     deepseek2.expert_count u32              = 256
> llama_model_loader: - kv  23:              deepseek2.expert_shared_count u32              = 1
> llama_model_loader: - kv  24:             deepseek2.expert_weights_scale f32              = 2.500000
> llama_model_loader: - kv  25:              deepseek2.expert_weights_norm bool             = true
> llama_model_loader: - kv  26:               deepseek2.expert_gating_func u32              = 2
> llama_model_loader: - kv  27:             deepseek2.rope.dimension_count u32              = 64
> llama_model_loader: - kv  28:                deepseek2.rope.scaling.type str              = yarn
> llama_model_loader: - kv  29:              deepseek2.rope.scaling.factor f32              = 40.000000
> llama_model_loader: - kv  30: deepseek2.rope.scaling.original_context_length u32              = 4096
> llama_model_loader: - kv  31: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
> llama_model_loader: - kv  32:                       tokenizer.ggml.model str              = gpt2
> llama_model_loader: - kv  33:                         tokenizer.ggml.pre str              = deepseek-v3
> llama_model_loader: - kv  34:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<�...
> llama_model_loader: - kv  35:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
> llama_model_loader: - kv  36:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
> llama_model_loader: - kv  37:                tokenizer.ggml.bos_token_id u32              = 0
> llama_model_loader: - kv  38:                tokenizer.ggml.eos_token_id u32              = 1
> llama_model_loader: - kv  39:            tokenizer.ggml.padding_token_id u32              = 1
> llama_model_loader: - kv  40:               tokenizer.ggml.add_bos_token bool             = true
> llama_model_loader: - kv  41:               tokenizer.ggml.add_eos_token bool             = false
> llama_model_loader: - kv  42:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
> llama_model_loader: - kv  43:               general.quantization_version u32              = 2
> llama_model_loader: - type  f32:  361 tensors
> llama_model_loader: - type  f16:  786 tensors
> llm_load_vocab: special tokens cache size = 818
> llm_load_vocab: token to piece cache size = 0.8223 MB
> llm_load_print_meta: format           = GGUF V3 (latest)
> llm_load_print_meta: arch             = deepseek2
> llm_load_print_meta: vocab type       = BPE
> llm_load_print_meta: n_vocab          = 129280
> llm_load_print_meta: n_merges         = 127741
> llm_load_print_meta: vocab_only       = 0
> llm_load_print_meta: n_ctx_train      = 163840
> llm_load_print_meta: n_embd           = 7168
> llm_load_print_meta: n_layer          = 61
> llm_load_print_meta: n_head           = 128
> llm_load_print_meta: n_head_kv        = 128
> llm_load_print_meta: n_rot            = 64
> llm_load_print_meta: n_swa            = 0
> llm_load_print_meta: n_embd_head_k    = 192
> llm_load_print_meta: n_embd_head_v    = 128
> llm_load_print_meta: n_gqa            = 1
> llm_load_print_meta: n_embd_k_gqa     = 24576
> llm_load_print_meta: n_embd_v_gqa     = 16384
> llm_load_print_meta: f_norm_eps       = 0.0e+00
> llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
> llm_load_print_meta: f_clamp_kqv      = 0.0e+00
> llm_load_print_meta: f_max_alibi_bias = 0.0e+00
> llm_load_print_meta: f_logit_scale    = 0.0e+00
> llm_load_print_meta: n_ff             = 18432
> llm_load_print_meta: n_expert         = 256
> llm_load_print_meta: n_expert_used    = 8
> llm_load_print_meta: causal attn      = 1
> llm_load_print_meta: pooling type     = 0
> llm_load_print_meta: rope type        = 0
> llm_load_print_meta: rope scaling     = yarn
> llm_load_print_meta: freq_base_train  = 10000.0
> llm_load_print_meta: freq_scale_train = 0.025
> llm_load_print_meta: n_ctx_orig_yarn  = 4096
> llm_load_print_meta: rope_finetuned   = unknown
> llm_load_print_meta: ssm_d_conv       = 0
> llm_load_print_meta: ssm_d_inner      = 0
> llm_load_print_meta: ssm_d_state      = 0
> llm_load_print_meta: ssm_dt_rank      = 0
> llm_load_print_meta: model type       = 671B
> llm_load_print_meta: model ftype      = F16
> llm_load_print_meta: model params     = 672.050 B
> llm_load_print_meta: model size       = 1251.990 GiB (16.003 BPW) 
> llm_load_print_meta: repeating layers = 1248.538 GiB (16.003 BPW, 670.196 B parameters)
> llm_load_print_meta: general.name     = DeepSeek R1 Bf16
> llm_load_print_meta: BOS token        = 0 '<｜begin▁of▁sentence｜>'
> llm_load_print_meta: EOS token        = 1 '<｜end▁of▁sentence｜>'
> llm_load_print_meta: PAD token        = 1 '<｜end▁of▁sentence｜>'
> llm_load_print_meta: LF token         = 131 'Ä'
> llm_load_print_meta: max token length = 256
> llm_load_print_meta: n_layer_dense_lead   = 3
> llm_load_print_meta: n_lora_q             = 1536
> llm_load_print_meta: n_lora_kv            = 512
> llm_load_print_meta: n_ff_exp             = 2048
> llm_load_print_meta: n_expert_shared      = 1
> llm_load_print_meta: expert_weights_scale = 2.5
> llm_load_print_meta: expert_weights_norm  = 1
> llm_load_print_meta: expert_gating_func   = sigmoid
> llm_load_print_meta: rope_yarn_log_mul    = 0.1000
> ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
> ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
> ggml_cuda_init: found 3 CUDA devices:
>   Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
>   Device 1: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
>   Device 2: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
> llm_load_tensors: ggml ctx size =    1.87 MiB
> llm_load_tensors: offloading 3 repeating layers to GPU
> llm_load_tensors: offloaded 3/62 layers to GPU
> llm_load_tensors:        CPU buffer size = 1282038.27 MiB
> llm_load_tensors:      CUDA0 buffer size = 21983.94 MiB
> llm_load_tensors:      CUDA1 buffer size = 21983.94 MiB
> llm_load_tensors:      CUDA2 buffer size = 21983.94 MiB
> ....................................................................................................
> llama_new_context_with_model: n_batch is less than GGML_KQ_MASK_PAD - increasing to 32
> llama_new_context_with_model: n_ctx      = 512
> llama_new_context_with_model: n_batch    = 32
> llama_new_context_with_model: n_ubatch   = 31
> llama_new_context_with_model: flash_attn = 0
> llama_new_context_with_model: mla_attn   = 0
> llama_new_context_with_model: attn_max_b = 0
> llama_new_context_with_model: fused_moe  = 0
> llama_new_context_with_model: ser        = -1, 0
> llama_new_context_with_model: freq_base  = 10000.0
> llama_new_context_with_model: freq_scale = 0.025
> llama_kv_cache_init:  CUDA_Host KV buffer size =  2320.00 MiB
> llama_kv_cache_init:      CUDA0 KV buffer size =    40.00 MiB
> llama_kv_cache_init:      CUDA1 KV buffer size =    40.00 MiB
> llama_kv_cache_init:      CUDA2 KV buffer size =    40.00 MiB
> llama_new_context_with_model: KV self size  = 2440.00 MiB, K (f16): 1464.00 MiB, V (f16):  976.00 MiB
> llama_new_context_with_model:  CUDA_Host  output buffer size =     0.49 MiB
> llama_new_context_with_model:      CUDA0 compute buffer size =    17.14 MiB
> llama_new_context_with_model:      CUDA1 compute buffer size =    16.65 MiB
> llama_new_context_with_model:      CUDA2 compute buffer size =    16.65 MiB
> llama_new_context_with_model:        CPU compute buffer size =     0.00 MiB
> llama_new_context_with_model:  CUDA_Host compute buffer size =    17.14 MiB
> llama_new_context_with_model: graph nodes  = 3724
> llama_new_context_with_model: graph splits = 5
> 
> system_info: n_threads = 16 / 32 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | 
> compute_imatrix: tokenizing the input ..
> compute_imatrix: tokenization took 217.036 ms
> compute_imatrix: computing over 124 chunks with batch_size 31
> 
> save_imatrix: entry '             blk.60.ffn_down_exps.weight' has partial data (90.23%) 25 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.59.ffn_down_exps.weight' has partial data (87.11%) 33 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.59.ffn_up_exps.weight' has partial data (87.11%) 33 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.59.ffn_gate_exps.weight' has partial data (87.11%) 33 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.60.ffn_gate_exps.weight' has partial data (90.23%) 25 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.58.ffn_down_exps.weight' has partial data (89.84%) 26 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.57.ffn_down_exps.weight' has partial data (87.50%) 32 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.56.ffn_down_exps.weight' has partial data (90.62%) 24 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.56.ffn_gate_exps.weight' has partial data (90.62%) 24 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.55.ffn_down_exps.weight' has partial data (87.89%) 31 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.55.ffn_gate_exps.weight' has partial data (87.89%) 31 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.54.ffn_down_exps.weight' has partial data (90.23%) 25 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.54.ffn_up_exps.weight' has partial data (90.23%) 25 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.53.ffn_gate_exps.weight' has partial data (88.28%) 30 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.52.ffn_down_exps.weight' has partial data (87.50%) 32 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.52.ffn_up_exps.weight' has partial data (87.50%) 32 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.52.ffn_gate_exps.weight' has partial data (87.50%) 32 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.51.ffn_down_exps.weight' has partial data (83.59%) 42 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.51.ffn_gate_exps.weight' has partial data (83.59%) 42 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.50.ffn_down_exps.weight' has partial data (85.94%) 36 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.50.ffn_gate_exps.weight' has partial data (85.94%) 36 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.57.ffn_gate_exps.weight' has partial data (87.50%) 32 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.49.ffn_gate_exps.weight' has partial data (86.72%) 34 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.54.ffn_gate_exps.weight' has partial data (90.23%) 25 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.48.ffn_up_exps.weight' has partial data (89.06%) 28 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.47.ffn_up_exps.weight' has partial data (88.67%) 29 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.46.ffn_down_exps.weight' has partial data (88.67%) 29 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.46.ffn_up_exps.weight' has partial data (88.67%) 29 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.46.ffn_gate_exps.weight' has partial data (88.67%) 29 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.49.ffn_up_exps.weight' has partial data (86.72%) 34 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.22.ffn_up_exps.weight' has partial data (88.28%) 30 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.33.ffn_down_exps.weight' has partial data (87.11%) 33 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.19.ffn_down_exps.weight' has partial data (92.19%) 20 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.58.ffn_gate_exps.weight' has partial data (89.84%) 26 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.48.ffn_gate_exps.weight' has partial data (89.06%) 28 out of 256 experts are missing data - skipping
> save_imatrix: entry '              blk.6.ffn_gate_exps.weight' has partial data (98.44%) 4 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.12.ffn_up_exps.weight' has partial data (96.88%) 8 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.21.ffn_gate_exps.weight' has partial data (89.84%) 26 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.19.ffn_up_exps.weight' has partial data (92.19%) 20 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.17.ffn_gate_exps.weight' has partial data (92.97%) 18 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.58.ffn_up_exps.weight' has partial data (89.84%) 26 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.16.ffn_down_exps.weight' has partial data (89.45%) 27 out of 256 experts are missing data - skipping
> save_imatrix: entry '                blk.9.ffn_up_exps.weight' has partial data (96.48%) 9 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.15.ffn_up_exps.weight' has partial data (89.45%) 27 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.10.ffn_up_exps.weight' has partial data (93.75%) 16 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.15.ffn_gate_exps.weight' has partial data (89.45%) 27 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.53.ffn_up_exps.weight' has partial data (88.28%) 30 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.20.ffn_up_exps.weight' has partial data (92.19%) 20 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.48.ffn_down_exps.weight' has partial data (89.06%) 28 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.45.ffn_down_exps.weight' has partial data (86.33%) 35 out of 256 experts are missing data - skipping
> save_imatrix: entry '              blk.8.ffn_down_exps.weight' has partial data (96.09%) 10 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.14.ffn_down_exps.weight' has partial data (89.84%) 26 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.10.ffn_down_exps.weight' has partial data (93.75%) 16 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.40.ffn_up_exps.weight' has partial data (85.94%) 36 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.47.ffn_gate_exps.weight' has partial data (88.67%) 29 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.60.ffn_up_exps.weight' has partial data (90.23%) 25 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.12.ffn_gate_exps.weight' has partial data (96.88%) 8 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.37.ffn_up_exps.weight' has partial data (89.06%) 28 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.25.ffn_down_exps.weight' has partial data (80.86%) 49 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.26.ffn_down_exps.weight' has partial data (81.64%) 47 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.27.ffn_down_exps.weight' has partial data (85.16%) 38 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.11.ffn_down_exps.weight' has partial data (95.31%) 12 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '              blk.4.ffn_down_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.51.ffn_up_exps.weight' has partial data (83.59%) 42 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.36.ffn_down_exps.weight' has partial data (89.06%) 28 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.12.ffn_down_exps.weight' has partial data (96.88%) 8 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '              blk.3.ffn_down_exps.weight' has partial data (98.83%) 3 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '              blk.3.ffn_gate_exps.weight' has partial data (98.83%) 3 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '              blk.5.ffn_down_exps.weight' has partial data (98.83%) 3 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.30.ffn_up_exps.weight' has partial data (87.11%) 33 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.17.ffn_up_exps.weight' has partial data (92.97%) 18 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.50.ffn_up_exps.weight' has partial data (85.94%) 36 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.41.ffn_up_exps.weight' has partial data (91.02%) 23 out of 256 experts are missing data - skipping
> save_imatrix: entry '              blk.4.ffn_gate_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.44.ffn_up_exps.weight' has partial data (89.45%) 27 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.39.ffn_up_exps.weight' has partial data (89.84%) 26 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.16.ffn_gate_exps.weight' has partial data (89.45%) 27 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.13.ffn_down_exps.weight' has partial data (94.92%) 13 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.55.ffn_up_exps.weight' has partial data (87.89%) 31 out of 256 experts are missing data - skipping
> save_imatrix: entry '                blk.5.ffn_up_exps.weight' has partial data (98.83%) 3 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.21.ffn_up_exps.weight' has partial data (89.84%) 26 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.18.ffn_down_exps.weight' has partial data (90.62%) 24 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.25.ffn_up_exps.weight' has partial data (80.86%) 49 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.14.ffn_up_exps.weight' has partial data (89.84%) 26 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.31.ffn_gate_exps.weight' has partial data (86.33%) 35 out of 256 experts are missing data - skipping
> save_imatrix: entry '                blk.7.ffn_up_exps.weight' has partial data (96.88%) 8 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.44.ffn_down_exps.weight' has partial data (89.45%) 27 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.17.ffn_down_exps.weight' has partial data (92.97%) 18 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.20.ffn_gate_exps.weight' has partial data (92.19%) 20 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.18.ffn_gate_exps.weight' has partial data (90.62%) 24 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.57.ffn_up_exps.weight' has partial data (87.50%) 32 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.49.ffn_down_exps.weight' has partial data (86.72%) 34 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.32.ffn_gate_exps.weight' has partial data (87.89%) 31 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.44.ffn_gate_exps.weight' has partial data (89.45%) 27 out of 256 experts are missing data - skipping
> save_imatrix: entry '              blk.9.ffn_down_exps.weight' has partial data (96.48%) 9 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '              blk.7.ffn_gate_exps.weight' has partial data (96.88%) 8 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.36.ffn_gate_exps.weight' has partial data (89.06%) 28 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.41.ffn_gate_exps.weight' has partial data (91.02%) 23 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.47.ffn_down_exps.weight' has partial data (88.67%) 29 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.21.ffn_down_exps.weight' has partial data (89.84%) 26 out of 256 experts are missing data - skipping
> save_imatrix: entry '              blk.9.ffn_gate_exps.weight' has partial data (96.48%) 9 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.37.ffn_down_exps.weight' has partial data (89.06%) 28 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.34.ffn_up_exps.weight' has partial data (83.59%) 42 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.32.ffn_up_exps.weight' has partial data (87.89%) 31 out of 256 experts are missing data - skipping
> save_imatrix: entry '              blk.6.ffn_down_exps.weight' has partial data (98.44%) 4 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.15.ffn_down_exps.weight' has partial data (89.45%) 27 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.13.ffn_up_exps.weight' has partial data (94.92%) 13 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.11.ffn_up_exps.weight' has partial data (95.31%) 12 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '                blk.6.ffn_up_exps.weight' has partial data (98.44%) 4 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '                blk.3.ffn_up_exps.weight' has partial data (98.83%) 3 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.10.ffn_gate_exps.weight' has partial data (93.75%) 16 out of 256 experts are missing data - skipping
> save_imatrix: entry '              blk.7.ffn_down_exps.weight' has partial data (96.88%) 8 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.11.ffn_gate_exps.weight' has partial data (95.31%) 12 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.22.ffn_gate_exps.weight' has partial data (88.28%) 30 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.22.ffn_down_exps.weight' has partial data (88.28%) 30 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.14.ffn_gate_exps.weight' has partial data (89.84%) 26 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.56.ffn_up_exps.weight' has partial data (90.62%) 24 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.23.ffn_gate_exps.weight' has partial data (82.81%) 44 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.23.ffn_down_exps.weight' has partial data (82.81%) 44 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.24.ffn_gate_exps.weight' has partial data (80.86%) 49 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.24.ffn_up_exps.weight' has partial data (80.86%) 49 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.24.ffn_down_exps.weight' has partial data (80.86%) 49 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.18.ffn_up_exps.weight' has partial data (90.62%) 24 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.25.ffn_gate_exps.weight' has partial data (80.86%) 49 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.26.ffn_gate_exps.weight' has partial data (81.64%) 47 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.26.ffn_up_exps.weight' has partial data (81.64%) 47 out of 256 experts are missing data - skipping
> save_imatrix: entry '                blk.4.ffn_up_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.27.ffn_gate_exps.weight' has partial data (85.16%) 38 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.27.ffn_up_exps.weight' has partial data (85.16%) 38 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.38.ffn_down_exps.weight' has partial data (88.28%) 30 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.31.ffn_up_exps.weight' has partial data (86.33%) 35 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.23.ffn_up_exps.weight' has partial data (82.81%) 44 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.28.ffn_gate_exps.weight' has partial data (85.94%) 36 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.28.ffn_up_exps.weight' has partial data (85.94%) 36 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.28.ffn_down_exps.weight' has partial data (85.94%) 36 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.42.ffn_up_exps.weight' has partial data (87.89%) 31 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.29.ffn_gate_exps.weight' has partial data (88.67%) 29 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.29.ffn_up_exps.weight' has partial data (88.67%) 29 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.29.ffn_down_exps.weight' has partial data (88.67%) 29 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.30.ffn_gate_exps.weight' has partial data (87.11%) 33 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.30.ffn_down_exps.weight' has partial data (87.11%) 33 out of 256 experts are missing data - skipping
> save_imatrix: entry '                blk.8.ffn_up_exps.weight' has partial data (96.09%) 10 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.43.ffn_gate_exps.weight' has partial data (88.28%) 30 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.36.ffn_up_exps.weight' has partial data (89.06%) 28 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.40.ffn_gate_exps.weight' has partial data (85.94%) 36 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.13.ffn_gate_exps.weight' has partial data (94.92%) 13 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.31.ffn_down_exps.weight' has partial data (86.33%) 35 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.32.ffn_down_exps.weight' has partial data (87.89%) 31 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.16.ffn_up_exps.weight' has partial data (89.45%) 27 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.33.ffn_gate_exps.weight' has partial data (87.11%) 33 out of 256 experts are missing data - skipping
> save_imatrix: entry '              blk.5.ffn_gate_exps.weight' has partial data (98.83%) 3 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.20.ffn_down_exps.weight' has partial data (92.19%) 20 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.34.ffn_gate_exps.weight' has partial data (83.59%) 42 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.34.ffn_down_exps.weight' has partial data (83.59%) 42 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.33.ffn_up_exps.weight' has partial data (87.11%) 33 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.35.ffn_up_exps.weight' has partial data (87.50%) 32 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.35.ffn_down_exps.weight' has partial data (87.50%) 32 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.40.ffn_down_exps.weight' has partial data (85.94%) 36 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.37.ffn_gate_exps.weight' has partial data (89.06%) 28 out of 256 experts are missing data - skipping
> save_imatrix: entry '              blk.8.ffn_gate_exps.weight' has partial data (96.09%) 10 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.39.ffn_down_exps.weight' has partial data (89.84%) 26 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.38.ffn_gate_exps.weight' has partial data (88.28%) 30 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.38.ffn_up_exps.weight' has partial data (88.28%) 30 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.39.ffn_gate_exps.weight' has partial data (89.84%) 26 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.41.ffn_down_exps.weight' has partial data (91.02%) 23 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.53.ffn_down_exps.weight' has partial data (88.28%) 30 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.42.ffn_gate_exps.weight' has partial data (87.89%) 31 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.35.ffn_gate_exps.weight' has partial data (87.50%) 32 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.42.ffn_down_exps.weight' has partial data (87.89%) 31 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.43.ffn_up_exps.weight' has partial data (88.28%) 30 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.19.ffn_gate_exps.weight' has partial data (92.19%) 20 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.43.ffn_down_exps.weight' has partial data (88.28%) 30 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.45.ffn_gate_exps.weight' has partial data (86.33%) 35 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.45.ffn_up_exps.weight' has partial data (86.33%) 35 out of 256 experts are missing data - skipping
> save_imatrix: warning: storing only 573 out of 720 entries
> 
> save_imatrix: stored collected data after 10 chunks in /home/user/nvme/gguf/DeepSeek-R1/imatrix.dat
> compute_imatrix: 2230.84 seconds per pass - ETA 76 hours 50.38 minutes
> [1]4.3392,
> save_imatrix: entry '             blk.60.ffn_down_exps.weight' has partial data (93.75%) 16 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.59.ffn_down_exps.weight' has partial data (94.53%) 14 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.59.ffn_up_exps.weight' has partial data (94.53%) 14 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.59.ffn_gate_exps.weight' has partial data (94.53%) 14 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.60.ffn_gate_exps.weight' has partial data (93.75%) 16 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.58.ffn_down_exps.weight' has partial data (99.22%) 2 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.57.ffn_down_exps.weight' has partial data (98.44%) 4 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.56.ffn_down_exps.weight' has partial data (97.27%) 7 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.56.ffn_gate_exps.weight' has partial data (97.27%) 7 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.55.ffn_down_exps.weight' has partial data (96.48%) 9 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.55.ffn_gate_exps.weight' has partial data (96.48%) 9 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.54.ffn_down_exps.weight' has partial data (98.44%) 4 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.54.ffn_up_exps.weight' has partial data (98.44%) 4 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.53.ffn_gate_exps.weight' has partial data (97.66%) 6 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.52.ffn_down_exps.weight' has partial data (96.48%) 9 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.52.ffn_up_exps.weight' has partial data (96.48%) 9 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.52.ffn_gate_exps.weight' has partial data (96.48%) 9 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.51.ffn_down_exps.weight' has partial data (96.88%) 8 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.51.ffn_gate_exps.weight' has partial data (96.88%) 8 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.50.ffn_down_exps.weight' has partial data (96.48%) 9 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.50.ffn_gate_exps.weight' has partial data (96.48%) 9 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.57.ffn_gate_exps.weight' has partial data (98.44%) 4 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.49.ffn_gate_exps.weight' has partial data (95.31%) 12 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.54.ffn_gate_exps.weight' has partial data (98.44%) 4 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.48.ffn_up_exps.weight' has partial data (94.92%) 13 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.47.ffn_up_exps.weight' has partial data (95.70%) 11 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.46.ffn_down_exps.weight' has partial data (95.70%) 11 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.46.ffn_up_exps.weight' has partial data (95.70%) 11 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.46.ffn_gate_exps.weight' has partial data (95.70%) 11 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.49.ffn_up_exps.weight' has partial data (95.31%) 12 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.22.ffn_up_exps.weight' has partial data (96.09%) 10 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.33.ffn_down_exps.weight' has partial data (95.70%) 11 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.19.ffn_down_exps.weight' has partial data (97.27%) 7 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.58.ffn_gate_exps.weight' has partial data (99.22%) 2 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.48.ffn_gate_exps.weight' has partial data (94.92%) 13 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.21.ffn_gate_exps.weight' has partial data (96.48%) 9 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.19.ffn_up_exps.weight' has partial data (97.27%) 7 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.17.ffn_gate_exps.weight' has partial data (98.44%) 4 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.58.ffn_up_exps.weight' has partial data (99.22%) 2 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.16.ffn_down_exps.weight' has partial data (98.44%) 4 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.15.ffn_up_exps.weight' has partial data (98.44%) 4 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.10.ffn_up_exps.weight' has partial data (98.05%) 5 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.15.ffn_gate_exps.weight' has partial data (98.44%) 4 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.53.ffn_up_exps.weight' has partial data (97.66%) 6 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.20.ffn_up_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.48.ffn_down_exps.weight' has partial data (94.92%) 13 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.45.ffn_down_exps.weight' has partial data (96.09%) 10 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.14.ffn_down_exps.weight' has partial data (98.05%) 5 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.10.ffn_down_exps.weight' has partial data (98.05%) 5 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.40.ffn_up_exps.weight' has partial data (97.66%) 6 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.47.ffn_gate_exps.weight' has partial data (95.70%) 11 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.60.ffn_up_exps.weight' has partial data (93.75%) 16 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.37.ffn_up_exps.weight' has partial data (95.70%) 11 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.25.ffn_down_exps.weight' has partial data (92.19%) 20 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.26.ffn_down_exps.weight' has partial data (94.53%) 14 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.27.ffn_down_exps.weight' has partial data (94.53%) 14 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.51.ffn_up_exps.weight' has partial data (96.88%) 8 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.36.ffn_down_exps.weight' has partial data (98.05%) 5 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.30.ffn_up_exps.weight' has partial data (97.27%) 7 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.17.ffn_up_exps.weight' has partial data (98.44%) 4 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.50.ffn_up_exps.weight' has partial data (96.48%) 9 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.41.ffn_up_exps.weight' has partial data (96.48%) 9 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.44.ffn_up_exps.weight' has partial data (96.48%) 9 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.39.ffn_up_exps.weight' has partial data (97.66%) 6 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.16.ffn_gate_exps.weight' has partial data (98.44%) 4 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.13.ffn_down_exps.weight' has partial data (98.44%) 4 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.55.ffn_up_exps.weight' has partial data (96.48%) 9 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.21.ffn_up_exps.weight' has partial data (96.48%) 9 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.18.ffn_down_exps.weight' has partial data (96.88%) 8 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.25.ffn_up_exps.weight' has partial data (92.19%) 20 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.14.ffn_up_exps.weight' has partial data (98.05%) 5 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.31.ffn_gate_exps.weight' has partial data (94.14%) 15 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.44.ffn_down_exps.weight' has partial data (96.48%) 9 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.17.ffn_down_exps.weight' has partial data (98.44%) 4 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.20.ffn_gate_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.18.ffn_gate_exps.weight' has partial data (96.88%) 8 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.57.ffn_up_exps.weight' has partial data (98.44%) 4 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.49.ffn_down_exps.weight' has partial data (95.31%) 12 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.32.ffn_gate_exps.weight' has partial data (97.27%) 7 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.44.ffn_gate_exps.weight' has partial data (96.48%) 9 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.36.ffn_gate_exps.weight' has partial data (98.05%) 5 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.41.ffn_gate_exps.weight' has partial data (96.48%) 9 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.47.ffn_down_exps.weight' has partial data (95.70%) 11 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.21.ffn_down_exps.weight' has partial data (96.48%) 9 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.37.ffn_down_exps.weight' has partial data (95.70%) 11 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.34.ffn_up_exps.weight' has partial data (94.92%) 13 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.32.ffn_up_exps.weight' has partial data (97.27%) 7 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.15.ffn_down_exps.weight' has partial data (98.44%) 4 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.13.ffn_up_exps.weight' has partial data (98.44%) 4 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.10.ffn_gate_exps.weight' has partial data (98.05%) 5 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.22.ffn_gate_exps.weight' has partial data (96.09%) 10 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.22.ffn_down_exps.weight' has partial data (96.09%) 10 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.14.ffn_gate_exps.weight' has partial data (98.05%) 5 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.56.ffn_up_exps.weight' has partial data (97.27%) 7 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.23.ffn_gate_exps.weight' has partial data (92.97%) 18 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.23.ffn_down_exps.weight' has partial data (92.97%) 18 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.24.ffn_gate_exps.weight' has partial data (94.14%) 15 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.24.ffn_up_exps.weight' has partial data (94.14%) 15 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.24.ffn_down_exps.weight' has partial data (94.14%) 15 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.18.ffn_up_exps.weight' has partial data (96.88%) 8 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.25.ffn_gate_exps.weight' has partial data (92.19%) 20 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.26.ffn_gate_exps.weight' has partial data (94.53%) 14 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.26.ffn_up_exps.weight' has partial data (94.53%) 14 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.27.ffn_gate_exps.weight' has partial data (94.53%) 14 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.27.ffn_up_exps.weight' has partial data (94.53%) 14 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.38.ffn_down_exps.weight' has partial data (95.70%) 11 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.31.ffn_up_exps.weight' has partial data (94.14%) 15 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.23.ffn_up_exps.weight' has partial data (92.97%) 18 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.28.ffn_gate_exps.weight' has partial data (95.31%) 12 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.28.ffn_up_exps.weight' has partial data (95.31%) 12 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.28.ffn_down_exps.weight' has partial data (95.31%) 12 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.42.ffn_up_exps.weight' has partial data (95.31%) 12 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.29.ffn_gate_exps.weight' has partial data (95.31%) 12 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.29.ffn_up_exps.weight' has partial data (95.31%) 12 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.29.ffn_down_exps.weight' has partial data (95.31%) 12 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.30.ffn_gate_exps.weight' has partial data (97.27%) 7 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.30.ffn_down_exps.weight' has partial data (97.27%) 7 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.43.ffn_gate_exps.weight' has partial data (96.48%) 9 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.36.ffn_up_exps.weight' has partial data (98.05%) 5 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.40.ffn_gate_exps.weight' has partial data (97.66%) 6 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.13.ffn_gate_exps.weight' has partial data (98.44%) 4 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.31.ffn_down_exps.weight' has partial data (94.14%) 15 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.32.ffn_down_exps.weight' has partial data (97.27%) 7 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.16.ffn_up_exps.weight' has partial data (98.44%) 4 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.33.ffn_gate_exps.weight' has partial data (95.70%) 11 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.20.ffn_down_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.34.ffn_gate_exps.weight' has partial data (94.92%) 13 out of 256 experts are missing data - skipping
> save_imatrix: entry '             blk.34.ffn_down_exps.weight' has partial data (94.92%) 13 out of 256 experts are missing data - skipping
> save_imatrix: entry '               blk.33.ffn_up_exps.weight' has partial data (95.70%) 11 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.35.ffn_up_exps.weight' has partial data (96.48%) 9 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.35.ffn_down_exps.weight' has partial data (96.48%) 9 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.40.ffn_down_exps.weight' has partial data (97.66%) 6 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.37.ffn_gate_exps.weight' has partial data (95.70%) 11 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.39.ffn_down_exps.weight' has partial data (97.66%) 6 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.38.ffn_gate_exps.weight' has partial data (95.70%) 11 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.38.ffn_up_exps.weight' has partial data (95.70%) 11 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.39.ffn_gate_exps.weight' has partial data (97.66%) 6 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.41.ffn_down_exps.weight' has partial data (96.48%) 9 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.53.ffn_down_exps.weight' has partial data (97.66%) 6 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.42.ffn_gate_exps.weight' has partial data (95.31%) 12 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.35.ffn_gate_exps.weight' has partial data (96.48%) 9 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.42.ffn_down_exps.weight' has partial data (95.31%) 12 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.43.ffn_up_exps.weight' has partial data (96.48%) 9 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.19.ffn_gate_exps.weight' has partial data (97.27%) 7 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.43.ffn_down_exps.weight' has partial data (96.48%) 9 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.45.ffn_gate_exps.weight' has partial data (96.09%) 10 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.45.ffn_up_exps.weight' has partial data (96.09%) 10 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: warning: storing only 690 out of 720 entries
> 
> save_imatrix: stored collected data after 20 chunks in /home/user/nvme/gguf/DeepSeek-R1/imatrix.dat
> 
> save_imatrix: entry '             blk.60.ffn_down_exps.weight' has partial data (97.27%) 7 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.59.ffn_down_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.59.ffn_up_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.59.ffn_gate_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.60.ffn_gate_exps.weight' has partial data (97.27%) 7 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.48.ffn_up_exps.weight' has partial data (96.88%) 8 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.48.ffn_gate_exps.weight' has partial data (96.88%) 8 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.48.ffn_down_exps.weight' has partial data (96.88%) 8 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.60.ffn_up_exps.weight' has partial data (97.27%) 7 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.25.ffn_down_exps.weight' has partial data (97.27%) 7 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.26.ffn_down_exps.weight' has partial data (98.83%) 3 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.27.ffn_down_exps.weight' has partial data (96.88%) 8 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.25.ffn_up_exps.weight' has partial data (97.27%) 7 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.31.ffn_gate_exps.weight' has partial data (97.66%) 6 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.34.ffn_up_exps.weight' has partial data (97.27%) 7 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.23.ffn_gate_exps.weight' has partial data (97.27%) 7 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.23.ffn_down_exps.weight' has partial data (97.27%) 7 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.24.ffn_gate_exps.weight' has partial data (96.88%) 8 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.24.ffn_up_exps.weight' has partial data (96.88%) 8 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.24.ffn_down_exps.weight' has partial data (96.88%) 8 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.25.ffn_gate_exps.weight' has partial data (97.27%) 7 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.26.ffn_gate_exps.weight' has partial data (98.83%) 3 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.26.ffn_up_exps.weight' has partial data (98.83%) 3 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.27.ffn_gate_exps.weight' has partial data (96.88%) 8 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.27.ffn_up_exps.weight' has partial data (96.88%) 8 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.31.ffn_up_exps.weight' has partial data (97.66%) 6 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.23.ffn_up_exps.weight' has partial data (97.27%) 7 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.31.ffn_down_exps.weight' has partial data (97.66%) 6 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.34.ffn_gate_exps.weight' has partial data (97.27%) 7 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.34.ffn_down_exps.weight' has partial data (97.27%) 7 out of 256 experts are missing data Storing **but be aware**
> 
> save_imatrix: stored collected data after 30 chunks in /home/user/nvme/gguf/DeepSeek-R1/imatrix.dat
> [2]3.3852,
> save_imatrix: stored collected data after 40 chunks in /home/user/nvme/gguf/DeepSeek-R1/imatrix.dat
> 
> save_imatrix: stored collected data after 50 chunks in /home/user/nvme/gguf/DeepSeek-R1/imatrix.dat
> [3]3.2894,
> save_imatrix: stored collected data after 60 chunks in /home/user/nvme/gguf/DeepSeek-R1/imatrix.dat
> [4]3.8763,
> save_imatrix: stored collected data after 70 chunks in /home/user/nvme/gguf/DeepSeek-R1/imatrix.dat
> 
> save_imatrix: stored collected data after 80 chunks in /home/user/nvme/gguf/DeepSeek-R1/imatrix.dat
> [5]3.9718,
> save_imatrix: stored collected data after 90 chunks in /home/user/nvme/gguf/DeepSeek-R1/imatrix.dat
> 
> save_imatrix: stored collected data after 100 chunks in /home/user/nvme/gguf/DeepSeek-R1/imatrix.dat
> [6]4.0138,
> save_imatrix: stored collected data after 110 chunks in /home/user/nvme/gguf/DeepSeek-R1/imatrix.dat
> [7]3.4810,
> save_imatrix: stored collected data after 120 chunks in /home/user/nvme/gguf/DeepSeek-R1/imatrix.dat
> 
> save_imatrix: stored collected data after 130 chunks in /home/user/nvme/gguf/DeepSeek-R1/imatrix.dat
> [8]4.0895,
> save_imatrix: stored collected data after 140 chunks in /home/user/nvme/gguf/DeepSeek-R1/imatrix.dat
> 
> save_imatrix: stored collected data after 150 chunks in /home/user/nvme/gguf/DeepSeek-R1/imatrix.dat
> [9]4.3512,
> save_imatrix: stored collected data after 160 chunks in /home/user/nvme/gguf/DeepSeek-R1/imatrix.dat
> 
> save_imatrix: stored collected data after 170 chunks in /home/user/nvme/gguf/DeepSeek-R1/imatrix.dat
> [10]4.0907,
> save_imatrix: stored collected data after 180 chunks in /home/user/nvme/gguf/DeepSeek-R1/imatrix.dat
> [11]4.4292,
> save_imatrix: stored collected data after 190 chunks in /home/user/nvme/gguf/DeepSeek-R1/imatrix.dat
> 
> save_imatrix: stored collected data after 200 chunks in /home/user/nvme/gguf/DeepSeek-R1/imatrix.dat
> [12]-nan,
> save_imatrix: stored collected data after 210 chunks in /home/user/nvme/gguf/DeepSeek-R1/imatrix.dat
> 
> save_imatrix: stored collected data after 220 chunks in /home/user/nvme/gguf/DeepSeek-R1/imatrix.dat
> [13]-nan,
> 
> save_imatrix: stored collected data after 230 chunks in /home/user/nvme/gguf/DeepSeek-R1/imatrix.dat
> [14]-nan,
> save_imatrix: stored collected data after 240 chunks in /home/user/nvme/gguf/DeepSeek-R1/imatrix.dat
> ```
> </details>
> 
> Finally, I have a question regarding the MoE layers: is each layer data split across all experts, or are they only linked to one or multiple specific experts? If so, would it be beneficial to log which combination of experts are used the most depending on use cases such as coding or agentic workflows, in order to offload the most used layers for improved efficiency?

---

👤 **ikawrakow** replied the **2025-03-07** at **17:57:23**:<br>

The NaNs are concerning. If we got NaN probabilities (logits) out of the forward pass, the imatrix will be useless (will likely have NaNs). Another way to get a NaN in the perplexity is if the predicted probability for the observed token is zero. You maybe better of getting an imatrix from somewhere else. Have you tried running the same calculation with mainline `llama.cpp`? Btw, if you want to create imatrix data yourself and have enough disk space, you can quantize to `Q8_0` (no imatrix required for that), and then use the quantized model for the imatrix calculation. You will fit 2X more layers on the GPUs, so it may be somewhat faster. 

The messages about partial data are to be expected. Only 8 out of 256 experts get activated per token, so if the batch was short, it is likely to have some experts that never were activated, so the imatrix for those contains just zeros. If one tries to use such an imatrix to quantize a model, this can lead to bad results (including NaNs in the model). That's why in mainline `llama.cpp` they wouldn't let you save the data for **the entire experts tensor**, even if just one expert is missing data. I have changed that to allow the imatrix to be saved (and fill the missing experts with 1s to avoid issues during quantization), but only if the number of missing experts is greater than some fraction of the total experts in the tensor. That's why initially you see for some tensors "storing but be aware", and for others you see "skipping". As you collect more data eventually all experts have seen at least one token, so the messages go away.

Concerning offloading specific experts: I haven't gathered statistics myself, so I don't know how useful that could be. I have seen claims around the Internet that one can gain that way (by offloading often used experts). On the other hand, this is such an obvious thing to do but has not become widely used, so my guess is that this may not be really true. The term "expert" is kind of misleading in the sense that it kind of implies that a given set of experts will be active when dealing with a given kind of context. But this is absolutely not true. If you process a paragraph of, say, 500 tokens on some specific topic, you will observe that basically all "experts" were active at least once.

> 👤 **saood06** replied the **2025-03-09** at **03:39:15**:<br>
> Slightly offtopic but, how does the imatrix command here handle the 3 attention tensors? Since there will always be one set of tensors not activated depending on how you set the mla argument and I'm not sure how the imatrix program would handle that without resorting to generating an imatrix with data for only one type of attention.
> 
> > Concerning offloading specific experts: I haven't gathered statistics myself, so I don't know how useful that could be. I have seen claims around the Internet that one can gain that way (by offloading often used experts). On the other hand, this is such an obvious thing to do but has not become widely used, so my guess is that this may not be really true.
> 
> There is some truth to that claim for Deepseek-R1 since it is helpful for the creators, quote from the Deepseek-V3 whitepaper :
> 
> >In addition, although the batch-wise load balancing methods show consistent performance advantages, they also face two potential challenges in efficiency: [...] (2) domain-shift-induced load imbalance during inference. [...] For the second challenge, we also design and implement an efficient inference framework with redundant expert deployment, as described in [ [this code](https://github.com/deepseek-ai/EPLB) ].")
> 
> Is there any chance this could be useful for hybrid inference?
> 
> > The term "expert" is kind of misleading in the sense that it kind of implies that a given set of experts will be active when dealing with a given kind of context. But this is absolutely not true. If you process a paragraph of, say, 500 tokens on some specific topic, you will observe that basically all "experts" were active at least once.
> 
> It really depends on how the MoE is designed and then trained/[merged](https://github.com/arcee-ai/mergekit/blob/main/docs/moe.md). For Deepseek-V3/R1 the paper states:
> 
> >The key distinction between auxiliary-loss-free balancing and sequence-wise auxiliary loss lies in their balancing scope: batch-wise versus sequence-wise. Compared with the sequence-wise auxiliary loss, batch-wise balancing imposes a more flexible constraint, as it does not enforce in-domain balance on each sequence. This flexibility allows experts to better specialize in different domains. To validate this, we record and analyze the expert load of a 16B auxiliary-loss-based baseline and a 16B auxiliary-loss-free model on different domains in the Pile test set. As illustrated in Figure 9, we observe that the auxiliary-loss-free model demonstrates greater expert specialization patterns as expected.
> >[...]
> >[...] compared with the purely auxiliary-loss-based method, the auxiliary-loss-free strategy consistently achieves better model performance on most of the evaluation benchmarks
> 
> 👤 **ThomasBaruzier** replied the **2025-03-09** at **14:28:25**:<br>
> >  You maybe better of getting an imatrix from somewhere else.
> 
> I tried using one from [Bartowski's repo](https://huggingface.co/bartowski/DeepSeek-R1-GGUF/blob/main/DeepSeek-R1.imatrix) and [mradermacher's repo](https://huggingface.co/mradermacher/DeepSeek-R1-i1-GGUF/blob/main/imatrix.dat)
> 
> Unfortunately, I get this error with the following command:
> 
> `CMD | '/home/user/files/ai/llama/ik_llama.cpp/llama-quantize' --imatrix '/home/user/nvme/gguf/DeepSeek-R1/imatrix.dat' '/home/user/nvme/gguf/DeepSeek-R1/DeepSeek-R1-F16.gguf' '/home/user/nvme/gguf/DeepSeek-R1/DeepSeek-R1-iq1_s_r4.gguf' 'iq1_s_r4' '32'`
> 
> ```
> Missing importance matrix for tensor blk.0.attn_v_b.weight in a very low-bit quantization
> ```
> 
> <details>
> <summary>Full logs</summary>
> 
> ```
> Skipping F16 as it already exists: /home/user/nvme/gguf/DeepSeek-R1/DeepSeek-R1-F16.gguf
> Skipping imatrix as it already exists: /home/user/nvme/gguf/DeepSeek-R1/imatrix.dat
> CMD | '/home/user/files/ai/llama/ik_llama.cpp/llama-quantize' --imatrix '/home/user/nvme/gguf/DeepSeek-R1/imatrix.dat' '/home/user/nvme/gguf/DeepSeek-R1/DeepSeek-R1-F16.gguf' '/home/user/nvme/gguf/DeepSeek-R1/DeepSeek-R1-iq1_s_r4.gguf' 'iq1_s_r4' '32'
> load_imatrix: imatrix dataset='/training_data/calibration_datav3.txt'
> load_imatrix: loaded 720 importance matrix entries from /home/user/nvme/gguf/DeepSeek-R1/imatrix.dat computed on 124 chunks
> prepare_imatrix: have 720 importance matrix entries
> main: build = 1 (7bdbf99)
> main: built with cc (GCC) 14.2.1 20250207 for x86_64-pc-linux-gnu
> main: quantizing '/home/user/nvme/gguf/DeepSeek-R1/DeepSeek-R1-F16.gguf' to '/home/user/nvme/gguf/DeepSeek-R1/DeepSeek-R1-iq1_s_r4.gguf' as IQ1_S_R4 using 32 threads
> llama_model_loader: loaded meta data with 44 key-value pairs and 1147 tensors from /home/user/nvme/gguf/DeepSeek-R1/DeepSeek-R1-F16.gguf (version GGUF V3 (latest))
> llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
> llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
> llama_model_loader: - kv   1:                               general.type str              = model
> llama_model_loader: - kv   2:                               general.name str              = DeepSeek R1 Bf16
> llama_model_loader: - kv   3:                         general.size_label str              = 256x21B
> llama_model_loader: - kv   4:                               general.tags arr[str,1]       = ["text-generation"]
> llama_model_loader: - kv   5:                      deepseek2.block_count u32              = 61
> llama_model_loader: - kv   6:                   deepseek2.context_length u32              = 163840
> llama_model_loader: - kv   7:                 deepseek2.embedding_length u32              = 7168
> llama_model_loader: - kv   8:              deepseek2.feed_forward_length u32              = 18432
> llama_model_loader: - kv   9:             deepseek2.attention.head_count u32              = 128
> llama_model_loader: - kv  10:          deepseek2.attention.head_count_kv u32              = 128
> llama_model_loader: - kv  11:                   deepseek2.rope.freq_base f32              = 10000.000000
> llama_model_loader: - kv  12: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
> llama_model_loader: - kv  13:                deepseek2.expert_used_count u32              = 8
> llama_model_loader: - kv  14:                          general.file_type u32              = 1
> llama_model_loader: - kv  15:        deepseek2.leading_dense_block_count u32              = 3
> llama_model_loader: - kv  16:                       deepseek2.vocab_size u32              = 129280
> llama_model_loader: - kv  17:            deepseek2.attention.q_lora_rank u32              = 1536
> llama_model_loader: - kv  18:           deepseek2.attention.kv_lora_rank u32              = 512
> llama_model_loader: - kv  19:             deepseek2.attention.key_length u32              = 192
> llama_model_loader: - kv  20:           deepseek2.attention.value_length u32              = 128
> llama_model_loader: - kv  21:       deepseek2.expert_feed_forward_length u32              = 2048
> llama_model_loader: - kv  22:                     deepseek2.expert_count u32              = 256
> llama_model_loader: - kv  23:              deepseek2.expert_shared_count u32              = 1
> llama_model_loader: - kv  24:             deepseek2.expert_weights_scale f32              = 2.500000
> llama_model_loader: - kv  25:              deepseek2.expert_weights_norm bool             = true
> llama_model_loader: - kv  26:               deepseek2.expert_gating_func u32              = 2
> llama_model_loader: - kv  27:             deepseek2.rope.dimension_count u32              = 64
> llama_model_loader: - kv  28:                deepseek2.rope.scaling.type str              = yarn
> llama_model_loader: - kv  29:              deepseek2.rope.scaling.factor f32              = 40.000000
> llama_model_loader: - kv  30: deepseek2.rope.scaling.original_context_length u32              = 4096
> llama_model_loader: - kv  31: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
> llama_model_loader: - kv  32:                       tokenizer.ggml.model str              = gpt2
> llama_model_loader: - kv  33:                         tokenizer.ggml.pre str              = deepseek-v3
> llama_model_loader: - kv  34:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<�...
> llama_model_loader: - kv  35:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
> llama_model_loader: - kv  36:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
> llama_model_loader: - kv  37:                tokenizer.ggml.bos_token_id u32              = 0
> llama_model_loader: - kv  38:                tokenizer.ggml.eos_token_id u32              = 1
> llama_model_loader: - kv  39:            tokenizer.ggml.padding_token_id u32              = 1
> llama_model_loader: - kv  40:               tokenizer.ggml.add_bos_token bool             = true
> llama_model_loader: - kv  41:               tokenizer.ggml.add_eos_token bool             = false
> llama_model_loader: - kv  42:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
> llama_model_loader: - kv  43:               general.quantization_version u32              = 2
> llama_model_loader: - type  f32:  361 tensors
> llama_model_loader: - type  f16:  786 tensors
> ================================ Have weights data with 720 entries
> [   1/1147]                    token_embd.weight - [ 7168, 129280,     1,     1], type =    f16, 
> ====== llama_model_quantize_internal: did not find weights for token_embd.weight
> converting to q8_0 .. size =  1767.50 MiB ->   938.98 MiB
> [   2/1147]               blk.0.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
> [   3/1147]                blk.0.ffn_down.weight - [18432,  7168,     1,     1], type =    f16, converting to iq3_k_r4 .. size =   252.00 MiB ->    54.14 MiB
> [   4/1147]                blk.0.ffn_gate.weight - [ 7168, 18432,     1,     1], type =    f16, converting to iq3_k_r4 .. size =   252.00 MiB ->    54.14 MiB
> [   5/1147]                  blk.0.ffn_up.weight - [ 7168, 18432,     1,     1], type =    f16, converting to iq3_k_r4 .. size =   252.00 MiB ->    54.14 MiB
> [   6/1147]                blk.0.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
> [   7/1147]          blk.0.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
> [   8/1147]           blk.0.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
> [   9/1147]               blk.0.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
> [  10/1147]                blk.0.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 
> 
> llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0
> 
> ====== llama_model_quantize_internal: did not find weights for blk.0.attn_k_b.weight
> converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
> [  11/1147]                blk.0.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
> ====== llama_model_quantize_internal: did not find weights for blk.0.attn_v_b.weight
> 
> 
> ============================================================
> Missing importance matrix for tensor blk.0.attn_v_b.weight in a very low-bit quantization
> The result will be garbage, so bailing out
> ============================================================
> 
> llama_model_quantize: failed to quantize: Missing importance matrix for tensor blk.0.attn_v_b.weight in a very low-bit quantization
> main: failed to quantize model from '/home/user/nvme/gguf/DeepSeek-R1/DeepSeek-R1-F16.gguf'
> ```
> </details>
> 
> But it's not your repo, llama.cpp faces the exact same issue for some reason, with the equivalent command:
> 
> `CMD | '/home/user/files/ai/llama/llama.cpp/llama-quantize' --imatrix '/home/user/nvme/gguf/DeepSeek-R1/imatrix.dat' '/home/user/nvme/gguf/DeepSeek-R1/DeepSeek-R1-F16.gguf' '/home/user/nvme/gguf/DeepSeek-R1/DeepSeek-R1-iq1_s.gguf' 'iq1_s' '32'`
> 
> For completeness, I used `arcee-ai/DeepSeek-R1-bf16` to create the F16 GGUF using the following command:
> 
> `CMD | python '/home/user/files/ai/llama/ik_llama.cpp/convert_hf_to_gguf.py' '/home/user/nvme/models/DeepSeek-R1-bf16' --outfile '/home/user/storage/quants/gguf/DeepSeek-R1-bf16/DeepSeek-R1-bf16-F16.gguf' --outtype f16`
> 
> ```
> INFO:hf-to-gguf:Model successfully exported to /home/user/storage/quants/gguf/DeepSeek-R1-bf16/DeepSeek-R1-bf16-F16.gguf
> ```
> 
> I'm having a hard time figuring out what I did wrong to end up having these issues. By any chance, would you have an idea about what is going on?
> 
> ---
> 
> > On the other hand, this is such an obvious thing to do but has not become widely used, so my guess is that this may not be really true.
> 
> I guess I could try making stats about experts usage and see what happens. Even so the distribution of tokens accross experts is supposed to be even, nothing said that some experts could be used a little bit more than others, just like what happens when creating an imatrix for the model?
> 
> ---
> 
> Finally, thanks for all the other precious explanations. I just started making the imatrix for R1 using mainline llama.cpp, brb.
> 
> 👤 **ikawrakow** replied the **2025-03-09** at **14:32:32**:<br>
> Try adding `--ignore-imatrix-rules` to your `quantize` command.
> 
> 👤 **ThomasBaruzier** replied the **2025-03-09** at **14:46:11**:<br>
> So far so good, but the errors `did not find weights for blk.0.attn_k_b.weight` and `did not find weights for blk.0.attn_v_b.weight` are persisting across every layer quantized so far (0 though 7 for now). I don't know enough to tell, but wouldn't that mean that this is going to be equal to a non-imatrix quant?
> 
> 👤 **ikawrakow** replied the **2025-03-09** at **14:47:20**:<br>
> Explanation: the imatrix you use has been computed with standard attention. For MLA one adds two additional tensors (` attn_v_b` and `attn_k_b`). As these were not present during the imatrix calculation, they never got data. In mainline you cannot quantize a low-bit model with such imatrix. Here you can do it by adding `--ignore-imatrix-rules` to the command.
> 
> 👤 **ikawrakow** replied the **2025-03-09** at **14:49:44**:<br>
> > but wouldn't that mean that this is going to be equal to a non-imatrix quant
> 
> Only these two tensors (in each layer) will be quantized without imatrix. I see in the log they are quantized with `Q5_0`. This is not ideal (`Q5_K` would have been better), but at 5 bits the gain from having an imatrix is quite modest.
> 
> 👤 **ikawrakow** replied the **2025-03-09** at **14:52:42**:<br>
> If you are using the latest `ik_llama.cpp`, you can overwrite the `Q5_0` choice for these tensors by using
> ```
> --custom-q "\.attn_k_b\.weight=Q5_K,\.attn_v_b\.weight=Q5_K"
> ```
> 
> 👤 **ThomasBaruzier** replied the **2025-03-09** at **14:53:50**:<br>
> Wouldn't that mean I should be better off trying again making the imatrix myself with this repo for a higher quality result? Or, maybe, do these tensors not having any imatrix data have a negligible impact on the conversion?
> 
> Edit: I guess negligible looking at your latest answers
> 
> 👤 **ThomasBaruzier** replied the **2025-03-09** at **15:27:39**:<br>
> There is an issue when adding the `custom-q` argument:
> 
> `'./ik_llama.cpp/llama-quantize' --imatrix 'imatrix.dat' --token-embedding-type q8_0 --custom-q '\.attn_k_b\.weight=Q5_K,\.attn_v_b\.weight=Q5_K' --ignore-imatrix-rules 'DeepSeek-R1-F16.gguf' 'DeepSeek-R1-IQ1_S_R4.gguf' 'IQ1_S_R4' '32'`
> 
> ```
> Invalid quantization type 'Q5_K' in custom quantization input \.attn_k_b\.weight=Q5_K
> ```
> 
> Simplifying to commands like `--custom-q "\.attn_v_b\.weight=17"` or `--custom-q "test=Q4_0"` does not help. The error is thrown in .04s, before the model had a chance to be read.
> 
> 👤 **ikawrakow** replied the **2025-03-09** at **16:15:56**:<br>
> Sorry, it is `q5_K`, to `Q5_K`. It needs to match the quantization name in `ggml.c`.
> 
> 👤 **ThomasBaruzier** replied the **2025-03-09** at **16:37:29**:<br>
> Seems to work, thanks!

---

👤 **ikawrakow** replied the **2025-03-09** at **08:05:31**:<br>

> Slightly offtopic but, how does the imatrix command here handle the 3 attention tensors?

You calculate the imatrix with MLA enabled (and no FA, because this skips one of the activations). This gives you imatrix data for `wk_b` and `wv_b`. As `wv_b` is just the low half of `wkv_b`, the imatrix data for these two is the same. It is very easy to add this to the quantization function. I haven't done that because I don't have the concept of many MLA imatrix data files to be floating around the Internet. But if I'm wrong, let me know, and I'll put that in.

For imatrix data computed with standard attention, imatrix data for `wkv_b` apply to `wv_b` (see above). So, the only tensor left that does not have imatrix data is `wk_b`, which is the transposed version of the upper half of `wkv_b`. I don't think this is a big issue because one shouldn't be using low-bit quantization for `wk_b`, and once you go to `Q5_K` or above, there is barely any difference between quantization quality with and without imatrix.

> 👤 **ikawrakow** replied the **2025-03-09** at **08:12:21**:<br>
> > It really depends on how the MoE is designed and then trained/[merged](https://github.com/arcee-ai/mergekit/blob/main/docs/moe.md). For Deepseek-V3/R1 the paper states:
> 
> The paper can say many things when the day is long, but the only thing that is important is what happens in practice. What we observe in practice is that basically all experts participate in the processing of a batch containing tokens of the same topic. If that weren't true, we wouldn't be observing such a massive increase in PP performance as we increase batch and u-batch size.

---

👤 **ThomasBaruzier** replied the **2025-03-10** at **18:19:24**:<br>

So here's what I came up with following your instructions:

`custom.sh`:
```sh
#!/bin/bash

cd /home/user/nvme/gguf/DeepSeek-R1
rm -f DeepSeek-R1-custom.gguf

custom="
# Token embedding and output tensors
token_embd\.weight=q8_0
output\.weight=q6_K
output_norm\.weight=q5_K

# First 3 dense layers (GPU0)
blk\.[0-2]\..*=q5_K

# Layers 3-4 (GPU0) - MoE experts
blk\.[3-4]\.ffn_down_exps\.weight=iq4_xs
blk\.[3-4]\.ffn_gate_exps\.weight=iq2_xxs
blk\.[3-4]\.ffn_up_exps\.weight=iq2_xxs

# Layers 5-11 (GPU1) - MoE experts
blk\.[5-9]\.ffn_down_exps\.weight=iq3_xxs
blk\.[5-9]\.ffn_gate_exps\.weight=iq2_xxs
blk\.[5-9]\.ffn_up_exps\.weight=iq2_xxs
blk\.1[0-1]\.ffn_down_exps\.weight=iq3_xxs
blk\.1[0-1]\.ffn_gate_exps\.weight=iq2_xxs
blk\.1[0-1]\.ffn_up_exps\.weight=iq2_xxs

# Layers 12-18 (GPU2) - MoE experts
blk\.1[2-8]\.ffn_down_exps\.weight=iq3_xxs
blk\.1[2-8]\.ffn_gate_exps\.weight=iq2_xxs
blk\.1[2-8]\.ffn_up_exps\.weight=iq2_xxs

# Layers 19-60 (CPU) - MoE experts
blk\.19\.ffn_down_exps\.weight=iq2_k_r4
blk\.[2-5][0-9]\.ffn_down_exps\.weight=iq2_k_r4
blk\.60\.ffn_down_exps\.weight=iq2_k_r4
blk\.19\.ffn_gate_exps\.weight=iq2_xxs_r4
blk\.[2-5][0-9]\.ffn_gate_exps\.weight=iq2_xxs_r4
blk\.60\.ffn_gate_exps\.weight=iq2_xxs_r4
blk\.19\.ffn_up_exps\.weight=iq2_xxs_r4
blk\.[2-5][0-9]\.ffn_up_exps\.weight=iq2_xxs_r4
blk\.60\.ffn_up_exps\.weight=iq2_xxs_r4

# All attention tensors for MoE layers (3-60)
blk\.[3-9]\.attn_.*=q5_K
blk\.[1-5][0-9]\.attn_.*=q5_K
blk\.60\.attn_.*=q5_K

# Norm weights and bias for MoE layers (3-60)
blk\.[3-9]\.ffn_norm\.weight=q5_K
blk\.[1-5][0-9]\.ffn_norm\.weight=q5_K
blk\.60\.ffn_norm\.weight=q5_K
blk\.[3-9]\.exp_probs_b\.bias=q5_K
blk\.[1-5][0-9]\.exp_probs_b\.bias=q5_K
blk\.60\.exp_probs_b\.bias=q5_K

# Shared experts weights for MoE layers (3-60)
blk\.3\.ffn_.*shexp\.weight=q5_K
blk\.[4-9]\.ffn_.*shexp\.weight=q5_K
blk\.[1-5][0-9]\.ffn_.*shexp\.weight=q5_K
blk\.60\.ffn_.*shexp\.weight=q5_K
"

custom=$(
  echo "$custom" | grep -v '^#' | \
  sed -Ez 's:\n+:,:g;s:,$::;s:^,::'
)

/home/user/files/ai/llama/ik_llama.cpp/llama-quantize \
  --imatrix imatrix.dat \
  --token-embedding-type q8_0 \
  --output-tensor-type q6_K \
  --ignore-imatrix-rules \
  --custom-q "$custom" \
  DeepSeek-R1-F16.gguf DeepSeek-R1-custom.gguf Q6_K 32
```

`server.sh` (CUDA0 and CUDA1 switched because of PCIe speeds):
```sh
#!/bin/bash

/home/user/files/ai/llama/ik_llama.cpp/llama-server \
  -m /home/user/nvme/gguf/DeepSeek-R1/DeepSeek-R1-custom.gguf \
  --api-key "$LOCAL_API_KEY" \
  --host 0.0.0.0 \
  --port 5000 \
  -c 8192 \
  -t 16 \
  -sm layer \
  -mg 1 \
  -mla 2 \
  -fmoe \
  -ot "output\.weight=CUDA1" \
  -ot "output_norm\.weight=CUDA1" \
  -ot "token_embd\.weight=CUDA1" \
  -ot "blk\.[0-4]\..*=CUDA1" \
  -ot "blk\.[3-9]\.attn_.*=CUDA1" \
  -ot "blk\.[1-5][0-9]\.attn_.*=CUDA1" \
  -ot "blk\.60\.attn_.*=CUDA1" \
  -ot "blk\.[3-9]\.ffn_norm\.weight=CUDA1" \
  -ot "blk\.[1-5][0-9]\.ffn_norm\.weight=CUDA1" \
  -ot "blk\.60\.ffn_norm\.weight=CUDA1" \
  -ot "blk\.[3-9]\.ffn_.*shexp\.weight=CUDA1" \
  -ot "blk\.[1-5][0-9]\.ffn_.*shexp\.weight=CUDA1" \
  -ot "blk\.60\.ffn_.*shexp\.weight=CUDA1" \
  -ot "blk\.[5-9]\.ffn_down_exps\.weight=CUDA0" \
  -ot "blk\.[5-9]\.ffn_gate_exps\.weight=CUDA0" \
  -ot "blk\.[5-9]\.ffn_up_exps\.weight=CUDA0" \
  -ot "blk\.1[0-1]\.ffn_down_exps\.weight=CUDA0" \
  -ot "blk\.1[0-1]\.ffn_gate_exps\.weight=CUDA0" \
  -ot "blk\.1[0-1]\.ffn_up_exps\.weight=CUDA0" \
  -ot "blk\.1[2-8]\.ffn_down_exps\.weight=CUDA2" \
  -ot "blk\.1[2-8]\.ffn_gate_exps\.weight=CUDA2" \
  -ot "blk\.1[2-8]\.ffn_up_exps\.weight=CUDA2" \
```

Even though I haven't spent much time playing with the settings, the speed is already at 7.1-7.3 tok/s with very short prompt and generation, 6.6-6.8tok/s with a few hundred tokens and 6.2-6.4tok/s for 1k. Also, a ~1k token ingestion goes at 35-40tok/s. I don't really know if those numbers make sense given the setup, but I am already very happy with these speeds.

VRAM use is 23.59GB on the main GPU and 23.00GB on the other two. So 2.3/2.4GB is free to play with for longer context.

Next steps:
- play with kv cache quants and optimizations (would you have any recommendations?)
- run `llama-bench` and `llama-perplexity`

Also, it seems that I can't use `-ot` with llama-perplexity (haven't tried with `llama-bench`)

Edit: Main GPU usage is at 25% and other cards are at 0% when generating. Is it because of the RAM speed limitations?

> 👤 **ikawrakow** replied the **2025-03-11** at **06:33:54**:<br>
> I think these are very nice results! 
> 
> > Also, it seems that I can't use -ot with llama-perplexity (haven't tried with llama-bench)
> 
> `-ot` is implemented in `common`, so all examples should support it, including `llama-bench` and `llama-perplexity`.
> 
> > Main GPU usage is at 25% and other cards are at 0% when generating. Is it because of the RAM speed limitations?
> 
> So, this is stuff inherited from upstream that I don't understand very well. Not sure why the back end decides to run everything on the main GPU. If that really is the case, your other 2 GPUs are acting as very expensive RAM, and there is potential for improvement if one could convince the system to use all 3 GPUs (less data will be copied back-and-fort between the GPUs).
> 
> >  play with kv cache quants and optimizations (would you have any recommendations?)
> 
> You are using `mla = 2`, so the only supported KV cache type is `fp16` when the computation is done on the GPU. I'm working on adding `Q8_0` to further reduce the KV cache size, but still having some issues with that. You can try adding `-fa` to see if this would increase your prompt processing speed (it shouldn't have major impact on token generation).
> 
> 👤 **ikawrakow** replied the **2025-03-11** at **06:43:37**:<br>
> If you remove the `-fmoe`, does it still run everything on the main GPU?
> 
> 👤 **ThomasBaruzier** replied the **2025-03-11** at **16:30:22**:<br>
> Great! Thank you for all the advice, once again.
> 
> It seems that I forgot a backslash, `llama-bench` and `llama-perplexity` correctly uses the `-ot` argument, oops.
> 
> `llama-perplexity` works well, but I still have some issues with llama-bench, and the error is not very descriptive:
> ```
> ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
> ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
> ggml_cuda_init: found 3 CUDA devices:
>   Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
>   Device 1: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
>   Device 2: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
> | model                          |       size |     params | backend    | ngl |   main_gpu | mla | fmoe |          test |              t/s |
> | ------------------------------ | ---------: | ---------: | ---------- | --: | ---------: | --: | ---: | ------------: | ---------------: |
> main: error: failed to load model '/home/user/nvme/gguf/DeepSeek-R1/DeepSeek-R1-custom.gguf'
> ```
> 
> <details>
> <summary>Full command</summary>
> 
> ```sh
> #!/bin/bash
> 
> /home/user/files/ai/llama/ik_llama.cpp/llama-bench \
>   -m /home/user/nvme/gguf/DeepSeek-R1/DeepSeek-R1-custom.gguf \
>   -p 1024 \
>   -n 128 \
>   -t 16 \
>   -sm layer \
>   -mg 1 \
>   -mla 2 \
>   -fmoe 1 \
>   -ot "output\.weight=CUDA1" \
>   -ot "output_norm\.weight=CUDA1" \
>   -ot "token_embd\.weight=CUDA1" \
>   -ot "blk\.[0-4]\..*=CUDA1" \
>   -ot "blk\.[3-9]\.attn_.*=CUDA1" \
>   -ot "blk\.[1-5][0-9]\.attn_.*=CUDA1" \
>   -ot "blk\.60\.attn_.*=CUDA1" \
>   -ot "blk\.[3-9]\.ffn_norm\.weight=CUDA1" \
>   -ot "blk\.[1-5][0-9]\.ffn_norm\.weight=CUDA1" \
>   -ot "blk\.60\.ffn_norm\.weight=CUDA1" \
>   -ot "blk\.[3-9]\.ffn_.*shexp\.weight=CUDA1" \
>   -ot "blk\.[1-5][0-9]\.ffn_.*shexp\.weight=CUDA1" \
>   -ot "blk\.60\.ffn_.*shexp\.weight=CUDA1" \
>   -ot "blk\.[5-9]\.ffn_down_exps\.weight=CUDA0" \
>   -ot "blk\.[5-9]\.ffn_gate_exps\.weight=CUDA0" \
>   -ot "blk\.[5-9]\.ffn_up_exps\.weight=CUDA0" \
>   -ot "blk\.1[0-1]\.ffn_down_exps\.weight=CUDA0" \
>   -ot "blk\.1[0-1]\.ffn_gate_exps\.weight=CUDA0" \
>   -ot "blk\.1[0-1]\.ffn_up_exps\.weight=CUDA0" \
>   -ot "blk\.1[2-8]\.ffn_down_exps\.weight=CUDA2" \
>   -ot "blk\.1[2-8]\.ffn_gate_exps\.weight=CUDA2" \
>   -ot "blk\.1[2-8]\.ffn_up_exps\.weight=CUDA2" \
> ```
> </details>
> 
> Edit: using `--verbose`, I get: `llama_model_load: error loading model: failed to allocate buffer`. Is it allocating more context than it should? There is no `-c` equivalent (other than values in `-p` and `-n`), it seems.
> 
> When removing `-fmoe`, the GPU usage is still centralized on the main GPU, with 20-25% usage at 130-140w, while the other cards stay at 0% at ~100w.
> 
> Finally, using `-fa` slows down the prompt ingestion speeds to 28tok/s. Generation seems to not be affected. I've already seen this behavior on mainline when using `fa` with CPU offloading.
> 
> 👤 **ikawrakow** replied the **2025-03-11** at **16:36:21**:<br>
> You can add `-v` to `llama-bench` to see why it fails to load the model.
> 
> 👤 **ThomasBaruzier** replied the **2025-03-11** at **16:57:45**:<br>
> I get: `llama_model_load: error loading model: failed to allocate buffer`. Is it trying to allocate the full 128k context? There is no `-c` equivalent (other than values in `-p` and `-n`), it seems.
> 
> 👤 **ikawrakow** replied the **2025-03-11** at **18:04:04**:<br>
> No, it should use a context given by the sum of `-p` and `-n`.

---

👤 **ThomasBaruzier** replied the **2025-03-13** at **14:22:08**:<br>

Here are some early results for wiki.test:
IQ1_S unsloth (1.67 BPW): 5.5749 +/- 0.03545
IQ1_M unsloth (2.01 BPW): 4.7238 +/- 0.02859
IQ2_XXS custom (2.34 BPW): 4.1059 +/- 0.02411

PPL for IQ2_XXS unsloth (size equivalent with your custom quant) and IQ1_S_R4/IQ1_M_R4 are still running.

In the meantime, is there any reason why you didn't recommend your new SOTA quant types like IQ2_K, or IQ4_KSS?
Or, are these not quant types but rather full quants consisting of an improved mixture of already existing quants types? (Edit: seems like new quant types that  are fast on CPU as well, wow https://github.com/ikawrakow/ik_llama.cpp/discussions/8)

I see you added Q8 KV cache for MLA2. Nice! I will test perfs after the PPL tests.

Finally, I stumbled upon this paper I thought you might find interesting: https://arxiv.org/pdf/2503.05840
TLDR no more V cache as it can be retrieved from K cache with full accuracy, supposedly compatible with quantization and FA, with nice speed benefits.
Edit: I don't think it could apply here: "Slim attention is somewhat similar to DeepSeek’s multi-head latent attention"

---

👤 **ikawrakow** replied the **2025-03-13** at **15:15:04**:<br>

> In the meantime, is there any reason why you didn't recommend your new SOTA quant types like IQ2_K, or IQ4_KSS?

Someone else was observing issues (NaNs) with `IQ4_KSS` and `IQ4_K` and I wasn't sure where the problem is. In the meantime I know that the problem is with using those on CUDA for the experts weights. These quants do not have quantized matrix multiplication kernels (a.k.a. MMQ), so for them on CUDA matrix multiplications are done by first dequantizing to `fp16` and then using cuBLAS `fp16` GEMM. It turns out, for DeepSeek-R1 this does not work, the `fp16` range is not sufficient to accommodate the result.  Hence, these quants cannot be used on CUDA for the DeepSeek models. But if you want to use them for experts that are computed on the CPU, this is perfectly fine. `IQ4_K` in particular is much better than any other 4-bit quantization type for the models I have tested (all LLaMA-3 models apart from the 405B one, Gemma2, Qwen-2.5, Mistral-Nemo, etc.). `IQ4_KSS` does not have an `_r4` variant. The bit packing is very awkward to achieve exactly 4 bpw, so implemnting the `_r4` version will be a bit of a nightmare, so  I keep postponing to do it). `IQ4_KS` (same size as `IQ4_XS`) is a bit of hit-or-miss. For some models it is quite a bit better than `IQ4_XS`, but for some models it is only on par (and it has a slightly lower inference performance than `IQ4_XS`). `IQ3_K` is slighty better than `IQ3_S` with the same bpw, but it is much faster on the CPU. `IQ2_K` is about in the middle between `IQ2_XS` and `IQ2_S` in terms of size and quality, but should also be much faster. If you feel like experimenting with these, I would be curious to learn about their performance for DeepSeekR1.

> Finally, I stumbled upon this paper I thought you might find interesting: https://arxiv.org/pdf/2503.05840

Yes, I know about this paper. MLA=2 does the same thing, there is only K cache and the `V` tensor gets computed from that (in different ways, depending on context). The only difference is that with MLA one does not need to compute $W_K^{-1}$ matrix, the equivalent is provided by the DeepSeek $W_{KV}$ tensor. It sounds nice in theory, but there is the theory and than there is the practice. In practice one needs to also consider compute buffers as intermediate results need to go somewhere, and the fact that counting multiply-adds is just a very rough estimate of actual performance, which also depends on memory access patterns, matrix shapes and sizes, etc. IIRC, the main factor that made me reluctant to spend the time implementing something along these lines is the fact that the benefit mostly goes away for GQA, which most models use these days.

> 👤 **ThomasBaruzier** replied the **2025-03-13** at **16:20:03**:<br>
> > If you feel like experimenting with these, I would be curious to learn about their performance for DeepSeekR1
> 
> I'd be happy to. I spend more time setting up my LLMs than using them anyway. Thanks for all the valuable info about the quants, this will save me hours.
> 
> > MLA=2 does the same thing
> > spend the time implementing something along these lines
> 
> So what's the difference between MLA=2 and "something along these lines"?
> 
> 👤 **ikawrakow** replied the **2025-03-13** at **17:17:46**:<br>
> > So what's the difference between MLA=2 and "something along these lines"?
> 
> MLA=2 is specific to the DeepSeek attention mechanism. "Something along these lines" would be a generic implementation for any MHA model.

---

👤 **ikawrakow** replied the **2025-03-15** at **09:31:42**:<br>

> PPL for IQ2_XXS unsloth (size equivalent with your custom quant) and IQ1_S_R4/IQ1_M_R4 are still running.

Do you have the results now? I'm curious to know.

> 👤 **ThomasBaruzier** replied the **2025-03-15** at **11:02:21**:<br>
> | Quant | Size (MB) | PPL |
> |------------|-----------|-----|
> | DeepSeek-R1-UD-IQ1_S | 133,736 | 5.5749 |
> | DeepSeek-R1-UD-IQ1_M | 161,092 | 4.7238 |
> | DeepSeek-R1-UD-IQ2_XXS | 187,076 | 4.0964 |
> | DeepSeek-R1-custom | 188,544 | 4.1059 |
> 
> I couldn't run more tests for now since I got some issues with my GPUs. The temporary PLA shroud started to melt for the first one (having a hard time printing ABS rn) and a fan broke for the second one. It shoudn't take too long since the replacement part is already here.