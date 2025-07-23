### 🔀 [#394](https://github.com/ikawrakow/ik_llama.cpp/pull/394) - Handle incompatible DeepSeek GGUFs

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-07 |
| **Updated** | 2025-05-10 |

---

#### Description

Mainline `llama.cpp` [PR 12801](https://github.com/ggml-org/llama.cpp/pull/12801), which added MLA support for DeepSeek models 2.5 months after MLA was available here, broke backwards compatibility. As a result, the new DeepSeek GGUFs that started appearing on HF are not compatible with `ik_llama.cpp`, resulting in issues #373 and #383.

My initial reaction was to not support the new DeepSeek GGUFs, as there was no real reason to introduce the backwards incompatibility (and have people re-download the giant DeepSeek-R1/V3 models). The two new tensors (per layer) required for MLA can be easily created on-the-fly when loading the model as it is done here.

But after some more thought I decided to handle the incompatible GGUFs, and this functionality is added with this PR.

I have tested with DeepSeek-Lite, which uses the exact same attention architecture as DeepSeek-R1/V3. As I don't have the ability to run the large DeepSeek models, I would really appreciate if someone confirmed that it works for them.

**Big caveat**:  Using an incompatible model will only allow the initial MLA implementation (`mla = 1`) in this repository, which corresponds to what is done in mainline `llama.cpp`. The consequences are
* Lower prompt processing performance compared to `mla = 3`. The performance degradation increases with increasing context length (number of tokens in the KV cache)
* GPU Flash Attention will only be available for Ampere or newer Nvidia GPUs

---

#### 💬 Conversation

👤 **whatever1983** commented the **2025-05-09** at **05:36:06**:<br>

python convert_hf_to_gguf.py --outfile /mydata/Downloads/DeepSeek-V3-0324-Pruned-Coder-411B-q8_0-ik.gguf --outtype q8_0 /mydata/Downloads/DeepSeek-V3-0324-Pruned-Coder-411B/

WARNING:gguf.vocab:Adding merges requested but no merges found, output may be non-functional.

using llama.cpp's convert_hf_to_gguf.py works, but if I requantize into IQ4K, tensor errors pop out:

llama_model_load: error loading model: check_tensor_dims: tensor 'blk.0.attn_q_b.weight' has wrong shape; expected  1536, 73728, got  1536, 24576,     1,     1
llama_load_model_from_file: failed to load model
llama_init_from_gpt_params: error: failed to load model '/mydata/Downloads/DeepSeek-V3-0324-Pruned-Coder-411B-IQ4K.gguf'
 ERR [              load_model] unable to load model | tid="140599261888512" timestamp=1746768164 model="/mydata/Downloads/DeepSeek-V3-0324-Pruned-Coder-411B-IQ4K.gguf"

I would rather have convert_hf_to_gguf.py from the ik_llama.cpp repo work.

---

👤 **ikawrakow** commented the **2025-05-09** at **05:47:47**:<br>

> WARNING:gguf.vocab:Adding merges requested but no merges found, output may be non-functional.

Yes, the `convert_hf_to_gguf.py` script currently on master does not handle merges well. There is a fix in PR #377, but I haven't merged because for some reason it misses the `rope_scaling` tensor, and we have not understood why.

---

👤 **Panchovix** commented the **2025-05-09** at **17:24:31**:<br>

I'm testing now! With DeepSeekV3 0324 Q2_K_XL latest quant, on 128GB VRAM (5090+4090x2+A6000) and 192GB RAM (6000Mhz 7800X3D). But first I just noticed this

```
llm_load_tensors:        CPU buffer size = 133756.00 MiB
llm_load_tensors:  CUDA_Host buffer size =   497.11 MiB
llm_load_tensors:      CUDA0 buffer size = 22412.07 MiB
llm_load_tensors:      CUDA1 buffer size = 17714.47 MiB
llm_load_tensors:      CUDA2 buffer size = 21610.08 MiB
llm_load_tensors:      CUDA3 buffer size = 42786.36 MiB
```

Is there a way to load on GPU first and then CPU? This explains why on ikllamacpp I get 5-20 t/s on PP vs 60-100 t/s on llamacpp (on the latter it looks like this)

```
load_tensors: offloading output layer to GPU
load_tensors: offloaded 62/62 layers to GPU
load_tensors:        CUDA0 model buffer size = 22412.07 MiB
load_tensors:        CUDA1 model buffer size = 17714.47 MiB
load_tensors:        CUDA2 model buffer size = 21610.08 MiB
load_tensors:        CUDA3 model buffer size = 42786.36 MiB
load_tensors:          CPU model buffer size = 134253.11 MiB
```

Okay now regarding the model itself, I have loaded it with (no fa since I think fa is merged on main but not on the PR), had to change the allocation a bit to make it work.

```
./llama-server -m '/GGUFs/DeepSeek-V3-0324-UD-Q2_K_XL-merged.gguf' -c 16384 --no-mmap --no-warmup -v -ngl 999 -ot "blk.(0|1|2|3|4|5|6).ffn.=CUDA0" -ot "blk.(7|8|9|10).ffn.=CUDA1" -ot "blk.(11|12|13|14).ffn.=CUDA2"  -ot "blk.(15|16|17|18|19|20|21|22|23|24).ffn.=CUDA3"  -ot "ffn.*=CPU" -fmoe -mla 1

```

And it loads without issues

```
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size = 142690.30 MiB
llm_load_tensors:  CUDA_Host buffer size =   497.11 MiB
llm_load_tensors:      CUDA0 buffer size = 18265.88 MiB
llm_load_tensors:      CUDA1 buffer size = 17471.11 MiB
llm_load_tensors:      CUDA2 buffer size = 17472.86 MiB
llm_load_tensors:      CUDA3 buffer size = 42378.83 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 16384
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: mla_attn   = 1
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init:      CUDA0 KV buffer size =   510.00 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =   408.00 MiB
llama_kv_cache_init:      CUDA2 KV buffer size =   408.00 MiB
llama_kv_cache_init:      CUDA3 KV buffer size =   748.00 MiB
llama_new_context_with_model: KV self size  = 2074.00 MiB, c^KV (f16): 1098.00 MiB, kv^T (f16):  976.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.99 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      CUDA0 compute buffer size =  4522.00 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =  4481.00 MiB
llama_new_context_with_model:      CUDA2 compute buffer size =  4481.00 MiB
llama_new_context_with_model:      CUDA3 compute buffer size =  4481.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =    78.50 MiB
llama_new_context_with_model: graph nodes  = 3547
llama_new_context_with_model: graph splits = 398
```

Then generating also works without issues! 

Speeds look like this

```
INFO [           print_timings] prompt eval time     =  246500.09 ms /  3218 tokens (   76.60 ms per token,    13.05 tokens per second) | tid="140049526018048" timestamp=1746811228 id_slot=0 id_task=0 t_prompt_processing=246500.088 n_prompt_tokens_processed=3218 t_token=76.60040024860162 n_tokens_second=13.05476207375634
INFO [           print_timings] generation eval time =   63970.82 ms /   428 runs   (  149.46 ms per token,     6.69 tokens per second) | tid="140049526018048" timestamp=1746811228 id_slot=0 id_task=0 t_token_generation=63970.815 n_decoded=428 t_token=149.46452102803738 n_tokens_second=6.690550995793941
INFO [           print_timings]           total time =  310470.90 ms | tid="140049526018048" timestamp=1746811228 id_slot=0 id_task=0 t_prompt_processing=2
```

For reference, as I mentioned above on llamacpp with the same command but having CUDA0 loading first instead of CPU, I get

```
prompt eval time = 51369.66 ms / 3252 tokens ( 15.80 ms per token, 63.31 tokens per second)
```

So I can confirm latest quants with MLA works on ik llamacpp.

---

👤 **ikawrakow** commented the **2025-05-09** at **19:00:28**:<br>

@Panchovix Thanks for testing!

Why don't you simply use the same tensor overrides that you use with mainline `llama.cpp`?

If you post your `llama.cpp` command here, perhaps we can give you suggestions how you can improve it for `ik_llama.cpp`.

---

👤 **Panchovix** commented the **2025-05-09** at **19:11:22**:<br>

> @Panchovix Thanks for testing!
> 
> Why don't you simply use the same tensor overrides that you use with mainline `llama.cpp`?
> 
> If you post your `llama.cpp` command here, perhaps we can give you suggestions how you can improve it for `ik_llama.cpp`.

Had to modify it as I use -fa on main llamacpp and I think this PR was done before fa + mla was possible on main. The compute buffers on FA were 3.7 GB and then 400mb each, while here it was 4.5GB each buffer (which is near 1 tensor per GPU)

My command on main is

```
./llama-server -m '/GGUFs/DeepSeek-V3-0324-UD-Q2_K_XL-merged.gguf' -c 16384 --no-mmap --no-warmup -v -ngl 99 --override-tensor 'blk\.([0-7])\..*_exps\.=CUDA0' --override-tensor 'blk\.([8-9]|1[0-1])\..*_exps\.=CUDA1' --override-tensor 'blk\.(1[2-6])\..*_exps\.=CUDA2' --override-tensor 'blk\.(1[7-9]|2[0-6])\..*_exps\.=CUDA3' -fa --override-tensor 'blk\..*_exps\.=CPU' -mg 0
```
Adding -ub 1024 increases PP from 66 t/s to 100 t/s and -ub 1536 to 126 t/s

Sometimes it tries to load on CPU first, but I cancel and start it again until it starts to load on CUDA0. That way PP T/s perform as it should. If it loads on CPU first it drops to 20 t/s or less, so same behaviour as ik llamacpp for example.

---

👤 **ikawrakow** commented the **2025-05-09** at **19:24:18**:<br>

I have merged this PR. If you take the current main main branch and try
```
./llama-server -m '/GGUFs/DeepSeek-V3-0324-UD-Q2_K_XL-merged.gguf' -c 16384 --no-mmap --no-warmup -v -ngl 99 
    --override-tensor 'blk\.([0-7])\..*_exps\.=CUDA0'
    --override-tensor 'blk\.([8-9]|1[0-1])\..*_exps\.=CUDA1'
    --override-tensor 'blk\.(1[2-6])\..*_exps\.=CUDA2'
    --override-tensor 'blk\.(1[7-9]|2[0-6])\..*_exps\.=CUDA3'
    --override-tensor 'exps=CPU' -mg 0 -fa -fmoe -ub 1536
```
it should give you a similar TG performance as current `llama.cpp`, but better PP performance. With many tokens in the KV cache, TG performance will also become better.

If you have the patience to wait for the longer loading time, adding `-rtr` to the above will give you even better PP performance. 

As `llama.cpp` still stores a V cache, your should have some extra space to perhaps increase u-batch size to 2048.

---

👤 **Panchovix** commented the **2025-05-09** at **19:47:23**:<br>

Thanks! I went ahead and test, this is the output

```
./llama-server -m '/GGUFs/DeepSeek-V3-0324-UD-Q2_K_XL-merged.gguf' -c 16384 --no-mmap --no-warmup -v -ngl 99 --override-tensor 'blk\.([0-7])\..*_exps\.=CUDA0' --override-tensor 'blk\.([8-9]|1[0-1])\..*_exps\.=CUDA1' --override-tensor 'blk\.(1[2-6])\..*_exps\.=CUDA2' --override-tensor 'blk\.(1[7-9]|2[0-6])\..*_exps\.=CUDA3' --override-tensor 'exps=CPU' -mg 0 -fa -fmoe
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 4 CUDA devices:
  Device 0: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
  Device 1: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
  Device 2: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
  Device 3: NVIDIA RTX A6000, compute capability 8.6, VMM: yes
INFO [                    main] build info | tid="139803965288448" timestamp=1746819238 build=3679 commit="43a154d8"
INFO [                    main] system info | tid="139803965288448" timestamp=1746819238 n_threads=8 n_threads_batch=-1 total_threads=16 system_info="AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: loaded meta data with 64 key-value pairs and 1086 tensors from /GGUFs/DeepSeek-V3-0324-UD-Q2_K_XL-merged.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Deepseek-V3-0324
llama_model_loader: - kv   3:                            general.version str              = V3-0324
llama_model_loader: - kv   4:                           general.basename str              = Deepseek-V3-0324
llama_model_loader: - kv   5:                       general.quantized_by str              = Unsloth
llama_model_loader: - kv   6:                         general.size_label str              = 256x20B
llama_model_loader: - kv   7:                            general.license str              = mit
llama_model_loader: - kv   8:                           general.repo_url str              = https://huggingface.co/unsloth
llama_model_loader: - kv   9:                   general.base_model.count u32              = 1
llama_model_loader: - kv  10:                  general.base_model.0.name str              = DeepSeek V3 0324
llama_model_loader: - kv  11:               general.base_model.0.version str              = V3-0324
llama_model_loader: - kv  12:          general.base_model.0.organization str              = Deepseek Ai
llama_model_loader: - kv  13:              general.base_model.0.repo_url str              = https://huggingface.co/deepseek-ai/De...
llama_model_loader: - kv  14:                               general.tags arr[str,4]       = ["deepseek_v3", "deepseek", "unsloth"...
llama_model_loader: - kv  15:                          general.languages arr[str,1]       = ["en"]
llama_model_loader: - kv  16:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv  17:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv  18:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv  19:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  20:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv  21:          deepseek2.attention.head_count_kv u32              = 1
llama_model_loader: - kv  22:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  23: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  24:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  25:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  26:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  27:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  28:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  29:             deepseek2.attention.key_length u32              = 576
llama_model_loader: - kv  30:           deepseek2.attention.value_length u32              = 512
llama_model_loader: - kv  31:         deepseek2.attention.key_length_mla u32              = 192
llama_model_loader: - kv  32:       deepseek2.attention.value_length_mla u32              = 128
llama_model_loader: - kv  33:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  34:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  35:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  36:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  37:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  38:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  39:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  40:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  41:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  42: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  43: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  44:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  45:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  46:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<�...
llama_model_loader: - kv  47:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  48:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
llama_model_loader: - kv  49:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  50:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  51:            tokenizer.ggml.padding_token_id u32              = 2
llama_model_loader: - kv  52:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  53:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  54:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  55:               general.quantization_version u32              = 2
llama_model_loader: - kv  56:                          general.file_type u32              = 10
llama_model_loader: - kv  57:                      quantize.imatrix.file str              = DeepSeek-V3-0324-GGUF/imatrix_unsloth...
llama_model_loader: - kv  58:                   quantize.imatrix.dataset str              = unsloth_calibration_DeepSeek-V3-0324.txt
llama_model_loader: - kv  59:             quantize.imatrix.entries_count i32              = 720
llama_model_loader: - kv  60:              quantize.imatrix.chunks_count i32              = 60
llama_model_loader: - kv  61:                                   split.no u16              = 0
llama_model_loader: - kv  62:                        split.tensors.count i32              = 1086
llama_model_loader: - kv  63:                                split.count u16              = 0
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  122 tensors
llama_model_loader: - type q2_K:  122 tensors
llama_model_loader: - type q3_K:   54 tensors
llama_model_loader: - type q4_K:  389 tensors
llama_model_loader: - type q5_K:   23 tensors
llama_model_loader: - type q6_K:   15 tensors
==========================================================================
Detected incompatible DeepSeek model.
Will try to fix, but there are no guarantees

*** Your prompt processing speed will be crippled ***

Consider making your own ik_llama.cpp compatible model or
ask the model provider to make one for you,
==========================================================================
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
llm_load_print_meta: model ftype      = Q2_K - Medium
llm_load_print_meta: model params     = 671.026 B
llm_load_print_meta: model size       = 233.180 GiB (2.985 BPW) 
llm_load_print_meta: repeating layers = 231.986 GiB (2.978 BPW, 669.173 B parameters)
llm_load_print_meta: general.name     = Deepseek-V3-0324
llm_load_print_meta: BOS token        = 0 '<｜begin▁of▁sentence｜>'
llm_load_print_meta: EOS token        = 1 '<｜end▁of▁sentence｜>'
llm_load_print_meta: PAD token        = 2 '<｜▁pad▁｜>'
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
llm_load_tensors: ggml ctx size =    2.23 MiB
Tensor blk.3.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.9.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.9.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.9.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.10.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.10.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.11.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.11.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.11.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CUDA2
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CUDA2
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CUDA2
Tensor blk.13.ffn_gate_exps.weight buffer type overriden to CUDA2
Tensor blk.13.ffn_down_exps.weight buffer type overriden to CUDA2
Tensor blk.13.ffn_up_exps.weight buffer type overriden to CUDA2
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CUDA2
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CUDA2
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CUDA2
Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CUDA2
Tensor blk.15.ffn_down_exps.weight buffer type overriden to CUDA2
Tensor blk.15.ffn_up_exps.weight buffer type overriden to CUDA2
Tensor blk.16.ffn_gate_exps.weight buffer type overriden to CUDA2
Tensor blk.16.ffn_down_exps.weight buffer type overriden to CUDA2
Tensor blk.16.ffn_up_exps.weight buffer type overriden to CUDA2
Tensor blk.17.ffn_gate_exps.weight buffer type overriden to CUDA3
Tensor blk.17.ffn_down_exps.weight buffer type overriden to CUDA3
Tensor blk.17.ffn_up_exps.weight buffer type overriden to CUDA3
Tensor blk.18.ffn_gate_exps.weight buffer type overriden to CUDA3
Tensor blk.18.ffn_down_exps.weight buffer type overriden to CUDA3
Tensor blk.18.ffn_up_exps.weight buffer type overriden to CUDA3
Tensor blk.19.ffn_gate_exps.weight buffer type overriden to CUDA3
Tensor blk.19.ffn_down_exps.weight buffer type overriden to CUDA3
Tensor blk.19.ffn_up_exps.weight buffer type overriden to CUDA3
Tensor blk.20.ffn_gate_exps.weight buffer type overriden to CUDA3
Tensor blk.20.ffn_down_exps.weight buffer type overriden to CUDA3
Tensor blk.20.ffn_up_exps.weight buffer type overriden to CUDA3
Tensor blk.21.ffn_gate_exps.weight buffer type overriden to CUDA3
Tensor blk.21.ffn_down_exps.weight buffer type overriden to CUDA3
Tensor blk.21.ffn_up_exps.weight buffer type overriden to CUDA3
Tensor blk.22.ffn_gate_exps.weight buffer type overriden to CUDA3
Tensor blk.22.ffn_down_exps.weight buffer type overriden to CUDA3
Tensor blk.22.ffn_up_exps.weight buffer type overriden to CUDA3
Tensor blk.23.ffn_gate_exps.weight buffer type overriden to CUDA3
Tensor blk.23.ffn_down_exps.weight buffer type overriden to CUDA3
Tensor blk.23.ffn_up_exps.weight buffer type overriden to CUDA3
Tensor blk.24.ffn_gate_exps.weight buffer type overriden to CUDA3
Tensor blk.24.ffn_down_exps.weight buffer type overriden to CUDA3
Tensor blk.24.ffn_up_exps.weight buffer type overriden to CUDA3
Tensor blk.25.ffn_gate_exps.weight buffer type overriden to CUDA3
Tensor blk.25.ffn_down_exps.weight buffer type overriden to CUDA3
Tensor blk.25.ffn_up_exps.weight buffer type overriden to CUDA3
Tensor blk.26.ffn_gate_exps.weight buffer type overriden to CUDA3
Tensor blk.26.ffn_down_exps.weight buffer type overriden to CUDA3
Tensor blk.26.ffn_up_exps.weight buffer type overriden to CUDA3
Tensor blk.27.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size = 133756.00 MiB
llm_load_tensors:  CUDA_Host buffer size =   497.11 MiB
llm_load_tensors:      CUDA0 buffer size = 22412.07 MiB
llm_load_tensors:      CUDA1 buffer size = 17714.47 MiB
llm_load_tensors:      CUDA2 buffer size = 21610.08 MiB
llm_load_tensors:      CUDA3 buffer size = 42786.36 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 16384
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
=========================================================
llama_kv_cache_init: missing wkv_b tensor(s)
llama_kv_cache_init: changing MLA from 0 to 1
=========================================================
llama_kv_cache_init:      CUDA0 KV buffer size =   270.00 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =   216.00 MiB
llama_kv_cache_init:      CUDA2 KV buffer size =   216.00 MiB
llama_kv_cache_init:      CUDA3 KV buffer size =   396.00 MiB
llama_new_context_with_model: KV self size  = 1098.00 MiB, c^KV (f16): 1098.00 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.99 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      CUDA0 compute buffer size =  2161.00 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =   394.00 MiB
llama_new_context_with_model:      CUDA2 compute buffer size =   394.00 MiB
llama_new_context_with_model:      CUDA3 compute buffer size =   394.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =    64.01 MiB
llama_new_context_with_model: graph nodes  = 3304
llama_new_context_with_model: graph splits = 145
```

I could probably add 1 layer to a 4090 (5 GB left) and one 4090 (4GB left)

PP is still slower than main llamcpp, but I think it's becuase the reason I mentioned before.

On ikllamacpp, it seems the main GPU doesn't get saturated when starting as on llamacpp, and that also happens on main llamacpp if it loads CPU first before CUDA 0

As can you see on

```
llm_load_tensors:        CPU buffer size = 133756.00 MiB
llm_load_tensors:  CUDA_Host buffer size =   497.11 MiB
llm_load_tensors:      CUDA0 buffer size = 22412.07 MiB
llm_load_tensors:      CUDA1 buffer size = 17714.47 MiB
llm_load_tensors:      CUDA2 buffer size = 21610.08 MiB
llm_load_tensors:      CUDA3 buffer size = 42786.36 MiB
```

It starts loading from CPU buffer size instead of CUDA 0. Also this seems to make the CPU to stutter a bit while loading. I haven't tested with mmap yet.

RX/TX looks like this on PP

![image](https://github.com/user-attachments/assets/1ac4afc1-4959-4dd6-843e-7035d3a63b64)

While on main llamacpp looks like this (5090 X8 5.0 is saturated) 

![image](https://github.com/user-attachments/assets/61865c63-e866-4287-94b6-34c1a625b420)

Tested now on both latest commit of llamacpp and ikllamacpp, and speeds look like this

llamacpp (with the command I mentioned earlier, ub 1024)

```
prompt eval time =   35950.29 ms /  3218 tokens (   11.17 ms per token,    89.51 tokens per second)
       eval time =   44338.15 ms /   380 tokens (  116.68 ms per token,     8.57 tokens per second)
```

ikllamacpp with the command above + rtr (ub 1536)

```
INFO [           print_timings] prompt eval time     =  104442.50 ms /  3218 tokens (   32.46 ms per token,    30.81 tokens per second) | tid="139803965288448" timestamp=1746819713 id_slot=0 id_task=0 t_prompt_processing=104442.501 n_prompt_tokens_processed=3218 t_token=32.45571814791796 n_tokens_second=30.811211615853587
INFO [           print_timings] generation eval time =   51656.22 ms /   435 runs   (  118.75 ms per token,     8.42 tokens per second) | tid="139803965288448" timestamp=1746819713 id_slot=0 id_task=0 t_token_generation=51656.225 n_decoded=435 t_token=118.74994252873563 n_tokens_second=8.421056707105484
INFO [           print_timings]           total time =  156098.73 ms | tid="139803965288448" timestamp=1746819713 id_slot=0 id_task=0 t_prompt_processing=104442.501 t_token_generation=51656.225 t_total=156098.726
```

30 t/s PP still is pretty fast to not saturate GPU 0.

This is the output as reference from llamacpp

```
./llama-server -m '/GGUFs/DeepSeek-V3-0324-UD-Q2_K_XL-merged.gguf' -c 16384 --no-mmap --no-warmup -ngl 99 --override-tensor 'blk\.([0-7])\..*_exps\.=CUDA0' --override-tensor 'blk\.([8-9]|1[0-1])\..*_exps\.=CUDA1' --override-tensor 'blk\.(1[2-6])\..*_exps\.=CUDA2' --override-tensor 'blk\.(1[7-9]|2[0-6])\..*_exps\.=CUDA3' -fa --override-tensor 'blk\..*_exps\.=CPU' -mg 0 --ubatch-size 1024
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 4 CUDA devices:
  Device 0: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
  Device 1: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
  Device 2: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
  Device 3: NVIDIA RTX A6000, compute capability 8.6, VMM: yes
build: 5331 (33eff402) with gcc-14 (GCC) 14.2.1 20250210 (Red Hat 14.2.1-8) for x86_64-redhat-linux
system info: n_threads = 8, n_threads_batch = 8, total_threads = 16

system_info: n_threads = 8 (n_threads_batch = 8) / 16 | CUDA : ARCHS = 860,890,1200 | USE_GRAPHS = 1 | PEER_MAX_BATCH_SIZE = 128 | FA_ALL_QUANTS = 1 | CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | BMI2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | LLAMAFILE = 1 | OPENMP = 1 | AARCH64_REPACK = 1 | 

main: binding port with default address family
main: HTTP server is listening, hostname: 127.0.0.1, port: 8080, http threads: 15
main: loading model
srv    load_model: loading model '/GGUFs/DeepSeek-V3-0324-UD-Q2_K_XL-merged.gguf'
llama_model_load_from_file_impl: using device CUDA0 (NVIDIA GeForce RTX 5090) - 29249 MiB free
llama_model_load_from_file_impl: using device CUDA1 (NVIDIA GeForce RTX 4090) - 23633 MiB free
llama_model_load_from_file_impl: using device CUDA2 (NVIDIA GeForce RTX 4090) - 23698 MiB free
llama_model_load_from_file_impl: using device CUDA3 (NVIDIA RTX A6000) - 48280 MiB free
llama_model_loader: loaded meta data with 64 key-value pairs and 1086 tensors from /run/media/pancho/DE1652041651DDD9/HuggingFaceModelDownloader/Storage/GGUFs/DeepSeek-V3-0324-UD-Q2_K_XL-merged.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Deepseek-V3-0324
llama_model_loader: - kv   3:                            general.version str              = V3-0324
llama_model_loader: - kv   4:                           general.basename str              = Deepseek-V3-0324
llama_model_loader: - kv   5:                       general.quantized_by str              = Unsloth
llama_model_loader: - kv   6:                         general.size_label str              = 256x20B
llama_model_loader: - kv   7:                            general.license str              = mit
llama_model_loader: - kv   8:                           general.repo_url str              = https://huggingface.co/unsloth
llama_model_loader: - kv   9:                   general.base_model.count u32              = 1
llama_model_loader: - kv  10:                  general.base_model.0.name str              = DeepSeek V3 0324
llama_model_loader: - kv  11:               general.base_model.0.version str              = V3-0324
llama_model_loader: - kv  12:          general.base_model.0.organization str              = Deepseek Ai
llama_model_loader: - kv  13:              general.base_model.0.repo_url str              = https://huggingface.co/deepseek-ai/De...
llama_model_loader: - kv  14:                               general.tags arr[str,4]       = ["deepseek_v3", "deepseek", "unsloth"...
llama_model_loader: - kv  15:                          general.languages arr[str,1]       = ["en"]
llama_model_loader: - kv  16:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv  17:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv  18:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv  19:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  20:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv  21:          deepseek2.attention.head_count_kv u32              = 1
llama_model_loader: - kv  22:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  23: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  24:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  25:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  26:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  27:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  28:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  29:             deepseek2.attention.key_length u32              = 576
llama_model_loader: - kv  30:           deepseek2.attention.value_length u32              = 512
llama_model_loader: - kv  31:         deepseek2.attention.key_length_mla u32              = 192
llama_model_loader: - kv  32:       deepseek2.attention.value_length_mla u32              = 128
llama_model_loader: - kv  33:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  34:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  35:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  36:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  37:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  38:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  39:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  40:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  41:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  42: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  43: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  44:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  45:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  46:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<�...
llama_model_loader: - kv  47:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  48:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
llama_model_loader: - kv  49:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  50:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  51:            tokenizer.ggml.padding_token_id u32              = 2
llama_model_loader: - kv  52:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  53:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  54:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  55:               general.quantization_version u32              = 2
llama_model_loader: - kv  56:                          general.file_type u32              = 10
llama_model_loader: - kv  57:                      quantize.imatrix.file str              = DeepSeek-V3-0324-GGUF/imatrix_unsloth...
llama_model_loader: - kv  58:                   quantize.imatrix.dataset str              = unsloth_calibration_DeepSeek-V3-0324.txt
llama_model_loader: - kv  59:             quantize.imatrix.entries_count i32              = 720
llama_model_loader: - kv  60:              quantize.imatrix.chunks_count i32              = 60
llama_model_loader: - kv  61:                                   split.no u16              = 0
llama_model_loader: - kv  62:                        split.tensors.count i32              = 1086
llama_model_loader: - kv  63:                                split.count u16              = 0
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  122 tensors
llama_model_loader: - type q2_K:  122 tensors
llama_model_loader: - type q3_K:   54 tensors
llama_model_loader: - type q4_K:  389 tensors
llama_model_loader: - type q5_K:   23 tensors
llama_model_loader: - type q6_K:   15 tensors
print_info: file format = GGUF V3 (latest)
print_info: file type   = Q2_K - Medium
print_info: file size   = 233.18 GiB (2.98 BPW) 
load: special_eos_id is not in special_eog_ids - the tokenizer config may be incorrect
load: special tokens cache size = 818
load: token to piece cache size = 0.8223 MB
print_info: arch             = deepseek2
print_info: vocab_only       = 0
print_info: n_ctx_train      = 163840
print_info: n_embd           = 7168
print_info: n_layer          = 61
print_info: n_head           = 128
print_info: n_head_kv        = 1
print_info: n_rot            = 64
print_info: n_swa            = 0
print_info: n_swa_pattern    = 1
print_info: n_embd_head_k    = 576
print_info: n_embd_head_v    = 512
print_info: n_gqa            = 128
print_info: n_embd_k_gqa     = 576
print_info: n_embd_v_gqa     = 512
print_info: f_norm_eps       = 0.0e+00
print_info: f_norm_rms_eps   = 1.0e-06
print_info: f_clamp_kqv      = 0.0e+00
print_info: f_max_alibi_bias = 0.0e+00
print_info: f_logit_scale    = 0.0e+00
print_info: f_attn_scale     = 0.0e+00
print_info: n_ff             = 18432
print_info: n_expert         = 256
print_info: n_expert_used    = 8
print_info: causal attn      = 1
print_info: pooling type     = 0
print_info: rope type        = 0
print_info: rope scaling     = yarn
print_info: freq_base_train  = 10000.0
print_info: freq_scale_train = 0.025
print_info: n_ctx_orig_yarn  = 4096
print_info: rope_finetuned   = unknown
print_info: ssm_d_conv       = 0
print_info: ssm_d_inner      = 0
print_info: ssm_d_state      = 0
print_info: ssm_dt_rank      = 0
print_info: ssm_dt_b_c_rms   = 0
print_info: model type       = 671B
print_info: model params     = 671.03 B
print_info: general.name     = Deepseek-V3-0324
print_info: n_layer_dense_lead   = 3
print_info: n_lora_q             = 1536
print_info: n_lora_kv            = 512
print_info: n_embd_head_k_mla    = 192
print_info: n_embd_head_v_mla    = 128
print_info: n_ff_exp             = 2048
print_info: n_expert_shared      = 1
print_info: expert_weights_scale = 2.5
print_info: expert_weights_norm  = 1
print_info: expert_gating_func   = sigmoid
print_info: rope_yarn_log_mul    = 0.1000
print_info: vocab type       = BPE
print_info: n_vocab          = 129280
print_info: n_merges         = 127741
print_info: BOS token        = 0 '<｜begin▁of▁sentence｜>'
print_info: EOS token        = 1 '<｜end▁of▁sentence｜>'
print_info: EOT token        = 1 '<｜end▁of▁sentence｜>'
print_info: PAD token        = 2 '<｜▁pad▁｜>'
print_info: LF token         = 201 'Ċ'
print_info: FIM PRE token    = 128801 '<｜fim▁begin｜>'
print_info: FIM SUF token    = 128800 '<｜fim▁hole｜>'
print_info: FIM MID token    = 128802 '<｜fim▁end｜>'
print_info: EOG token        = 1 '<｜end▁of▁sentence｜>'
print_info: max token length = 256
load_tensors: loading model tensors, this can take a while... (mmap = false)
load_tensors: offloading 61 repeating layers to GPU
load_tensors: offloading output layer to GPU
load_tensors: offloaded 62/62 layers to GPU
load_tensors:        CUDA0 model buffer size = 22412.07 MiB
load_tensors:        CUDA1 model buffer size = 17714.47 MiB
load_tensors:        CUDA2 model buffer size = 21610.08 MiB
load_tensors:        CUDA3 model buffer size = 42786.36 MiB
load_tensors:          CPU model buffer size = 134253.11 MiB
....................................................................................................
llama_context: constructing llama_context
llama_context: n_seq_max     = 1
llama_context: n_ctx         = 16384
llama_context: n_ctx_per_seq = 16384
llama_context: n_batch       = 2048
llama_context: n_ubatch      = 1024
llama_context: causal_attn   = 1
llama_context: flash_attn    = 1
llama_context: freq_base     = 10000.0
llama_context: freq_scale    = 0.025
llama_context: n_ctx_per_seq (16384) < n_ctx_train (163840) -- the full capacity of the model will not be utilized
llama_context:  CUDA_Host  output buffer size =     0.49 MiB
llama_kv_cache_unified: kv_size = 16384, type_k = 'f16', type_v = 'f16', n_layer = 61, can_shift = 1, padding = 256
llama_kv_cache_unified:      CUDA0 KV buffer size =   510.00 MiB
llama_kv_cache_unified:      CUDA1 KV buffer size =   408.00 MiB
llama_kv_cache_unified:      CUDA2 KV buffer size =   408.00 MiB
llama_kv_cache_unified:      CUDA3 KV buffer size =   748.00 MiB
llama_kv_cache_unified: KV self size  = 2074.00 MiB, K (f16): 1098.00 MiB, V (f16):  976.00 MiB
llama_context:      CUDA0 compute buffer size =  3285.00 MiB
llama_context:      CUDA1 compute buffer size =   788.00 MiB
llama_context:      CUDA2 compute buffer size =   788.00 MiB
llama_context:      CUDA3 compute buffer size =   788.01 MiB
llama_context:  CUDA_Host compute buffer size =    92.01 MiB
llama_context: graph nodes  = 4782
llama_context: graph splits = 179 (with bs=1024), 111 (with bs=1)
```

I can add more info if needed!

---

👤 **ikawrakow** commented the **2025-05-10** at **10:13:53**:<br>

@Panchovix 

Thanks for the above.

I now finally understand. The difference is that `llama.cpp` offloads the tensors stored in RAM to the GPU and will do the matrix multiplication there. `ik_llama.cpp` does not do that, the matrix multiplication is performed on the CPU. In your specific case (not very strong CPU, lots of VRAM, small model, fast PCI-E, large batches) `llama.cpp` approach turns out better.

But if it happens that you feel bored, try Maverick (e.g., [this model](https://huggingface.co/unsloth/Llama-4-Maverick-17B-128E-Instruct-GGUF/tree/main/UD-Q4_K_XL)) and see what happens. 

There is a PR in mainline `llama.cpp` to allow disabling offload to the GPU, see [this PR](https://github.com/ggml-org/llama.cpp/pull/13386), and it is there because many times not offloading experts stored in RAM to the GPU gives better PP performance. I guess, I could add the opposite feature here to allow users to force GPU offload for tensors stored in RAM.

---

👤 **Panchovix** commented the **2025-05-10** at **17:08:54**:<br>

@ikawrakow ohh I see! If it's possible to do add the reverse feature it would be great! As I think ik llamacpp with it's optimizations would be faster than llamacpp for PP t/s if we could do the matrix multiplication in the GPU.

---

👤 **ikawrakow** commented the **2025-05-10** at **17:15:44**:<br>

There is PR #405 now. You can try it with as high u-batch size as you can go. Don't use '-rtr' as this will disable the GPU offload of the experts.