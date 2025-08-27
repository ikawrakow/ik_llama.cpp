## 🔀 [Pull Request #292](https://github.com/ikawrakow/ik_llama.cpp/pull/292) - Use bf16 instead of fp16 block scales for q8_1

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/use_q8_2` |
| **Target Branch** | `main` |
| **Created** | 2025-03-26 |
| **Updated** | 2025-03-27 |
| **Merged** | 2025-03-27 |

---

## 📄 Description

DeepSeek-V3/R1 gives NaNs when inference is run on a computer with `AVX512_VNNI` and the model is quantized with `Q8_0/Q8_0_R8` (issue [#285](https://github.com/ikawrakow/ik_llama.cpp/issues/285)). The difference to vanilla `AVX2` is that in that case activations are quantized with `Q8_1/Q8_1_X4`. The block scale and sum in `Q8_1/Q8_1_X4` are `fp16`.

We did have similar issues with `IQ1_S`, which was solved in [#194](https://github.com/ikawrakow/ik_llama.cpp/issues/194) by going to a different quantization type for the activations. I did create issue [#196](https://github.com/ikawrakow/ik_llama.cpp/issues/196) because of that.

We also observed NaNs on CUDA for `IQ4_K` and `IQ4_KS`. These quantization types do not have MMQ kernels, so matrix multiplications were done via dequantization to `fp16` and cuBLAS GEMM. The NaNs were resolved via dequantizing to `bf16` instead (PR [#261](https://github.com/ikawrakow/ik_llama.cpp/issues/261))

So, it seems one can not use `fp16` arithmetic in DeepSeek-V3/R1.

This is further confirmed by [#291](https://github.com/ikawrakow/ik_llama.cpp/issues/291), where we observe no NaNs when switching `Q8_0/Q8_0_R8` to vanilla `AVX2` implementation.  

This PR introduces `Q8_2/Q8_2_X4` quantization types that use `bf16` block scale and sum. All quantization types that previously used `Q8_1/Q8_1_X4` to quantize activations for CPU GEMM/GEMV are switched to `Q8_2/Q8_2_X4`.

This should resolve all NaNs on the CPU. 

I wonder why we are not getting NaNs on CUDA for the quantization types that do use `Q8_1`. Or maybe we do, and it is just that nobody has reported.

Closes [#285](https://github.com/ikawrakow/ik_llama.cpp/issues/285) and [#196](https://github.com/ikawrakow/ik_llama.cpp/issues/196)

---

## 💬 Conversation

👤 **ubergarm** commented on **2025-03-26** at **19:37:47**

I'm mostly afk until Friday, but had a moment to rebuild with this PR and run another perplexity test on the `q8_0` with the CPU only Xeon 6980P rig. *FINISHED*, looks clean, No NaNs  :point_down: 

<details>

<summary>llama-perplexity run on `q8_0` with this `PR@918abd1`</summary>

```bash
$ git branch | grep q8_2
* ik/use_q8_2

$ git rev-parse --short HEAD
918abd16

$ numactl -N 1 -m 1 \
./build/bin/llama-perplexity \
    -m /mnt/ai/models/unsloth/repack/DeepSeek-R1-Q8_0_R8.gguf \
    -f wiki.test.raw \
    -t 128 \
    -b 512 \
    --numa numactl 2>&1 | tee -a output.log

main: build = 3619 (918abd16)
main: built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu
main: seed  = 1743024820
llama_model_loader: loaded meta data with 45 key-value pairs and 1025 tensors from /mnt/ai/models/unsloth/repack/DeepSeek-R1-Q8_0_R8.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = DeepSeek R1 BF16
llama_model_loader: - kv   3:                       general.quantized_by str              = Unsloth
llama_model_loader: - kv   4:                         general.size_label str              = 256x20B
llama_model_loader: - kv   5:                           general.repo_url str              = https://huggingface.co/unsloth
llama_model_loader: - kv   6:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv   7:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv   8:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv   9:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  10:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv  11:          deepseek2.attention.head_count_kv u32              = 128
llama_model_loader: - kv  12:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  13: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  14:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  15:                          general.file_type u32              = 207
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
llama_model_loader: - kv  35:                      tokenizer.ggml.tokens arr[str,129280]  = ["
llama_model_loader: - kv  36:                  tokenizer.ggml.token_type arr[i32,129280]  = [3
llama_model_loader: - kv  37:                      tokenizer.ggml.merges arr[str,127741]  = ["
llama_model_loader: - kv  38:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  39:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  40:            tokenizer.ggml.padding_token_id u32              = 128815
llama_model_loader: - kv  41:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  42:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  43:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  44:               general.quantization_version u32              = 2
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:    1 tensors
llama_model_loader: - type q8_0_r8:  663 tensors
llm_load_vocab: special tokens cache size = 819
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
llm_load_print_meta: model ftype      = Q8_0_R8 - 8.5 bpw
llm_load_print_meta: model params     = 671.026 B
llm_load_print_meta: model size       = 664.295 GiB (8.504 BPW) 
llm_load_print_meta: repeating layers = 662.461 GiB (8.504 BPW, 669.173 B parameters)
llm_load_print_meta: general.name     = DeepSeek R1 BF16
llm_load_print_meta: BOS token        = 0 '<｜begin▁of▁sentence｜>'
llm_load_print_meta: EOS token        = 1 '<｜end▁of▁sentence｜>'
llm_load_print_meta: PAD token        = 128815 '<｜PAD▁TOKEN｜>'
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
llm_load_tensors: ggml ctx size =    0.42 MiB
llm_load_tensors:        CPU buffer size = 680237.97 MiB
....................................................................................................
============ llm_load_tensors: need to compute 61 wk_b tensors
Computed blk.0.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.1.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.2.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.3.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.4.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.5.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.6.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.7.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.8.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.9.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.10.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.11.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.12.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.13.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.14.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.15.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.16.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.17.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.18.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.19.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.20.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.21.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.22.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.23.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.24.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.25.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.26.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.27.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.28.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.29.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.30.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.31.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.32.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.33.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.34.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.35.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.36.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.37.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.38.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.39.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.40.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.41.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.42.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.43.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.44.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.45.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.46.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.47.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.48.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.49.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.50.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.51.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.52.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.53.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Collama_new_context_with_model: n_ctx      = 512
llama_new_context_with_model: n_batch    = 512
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 0
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init:        CPU KV buffer size =  2440.00 MiB
llama_new_context_with_model: KV self size  = 2440.00 MiB, K (f16): 1464.00 MiB, V (f16):  976.00 MiB
llama_new_context_with_model:        CPU  output buffer size =     0.49 MiB
llama_new_context_with_model:        CPU compute buffer size =   283.01 MiB
llama_new_context_with_model: graph nodes  = 3724
llama_new_context_with_model: graph splits = 1

system_info: n_threads = 128 / 512 | AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | 
perplexity: tokenizing the input ..
perplexity: tokenization took 926.669 ms
perplexity: calculating perplexity over 561 chunks, n_ctx=512, batch_size=512, n_seq=1
perplexity: 4.79 seconds per pass - ETA 44.82 minutes
mputed blk.54.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.55.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.56.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.57.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.58.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.59.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.60.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
[1]2.5126,[2]3.2872,[3]2.3691,[4]1.9785,[5]1.7891,[6]1.6484,[7]1.5564,[8]1.4901,[9]1.4404,[10]1.4011,[11]1.3853,[12]1.4164,[13]1.4278,[14]1.5541,[15]1.6851,[16]1.7456,[17]1.9079,[18]2.0380,[19]2.0009,[20]1.9896,[21]2.0973,[22]2.0702,[23]2.0438,[24]2.0563,[25]2.0272,[26]2.0041,[27]2.0526,[28]2.0594,[29]2.1082,[30]2.1390,[31]2.1727,[32]2.1906,[33]2.2302,[34]2.2708,[35]2.3199,[36]2.3726,[37]2.4078,[38]2.4521,[39]2.4930,[40]2.5516,[41]2.5934,[42]2.6057,[43]2.6551,[44]2.6716,[45]2.7514,[46]2.8017,[47]2.7570,[48]2.7112,[49]2.6853,[50]2.7049,[51]2.7508,[52]2.7659,[53]2.8151,[54]2.8282,[55]2.8593,[56]2.8908,[57]2.9048,[58]2.9409,[59]2.9515,[60]2.9977,[61]3.0377,[62]3.0942,[63]3.1262,[64]3.1707,[65]3.1801,[66]3.1624,[67]3.1393,[68]3.1709,[69]3.1661,[70]3.1813,[71]3.2001,[72]3.2159,[73]3.2304,[74]3.2536,[75]3.2327,[76]3.1859,[77]3.1427,[78]3.1378,[79]3.1155,[80]3.0971,[81]3.0604,[82]3.0639,[83]3.0325,[84]2.9965,[85]2.9614,[86]2.9361,[87]2.9298,[88]2.9012,[89]2.8843,[90]2.8585,[91]2.8289,[92]2.8038,[93]2.7767,[94]2.7499,[95]2.7259,[96]2.7243,[97]2.7308,[98]2.7150,[99]2.6975,[100]2.6998,[101]2.6914,[102]2.7080,[103]2.7339,[104]2.7526,[105]2.7495,[106]2.7716,[107]2.7961,[108]2.8169,[109]2.8509,[110]2.8849,[111]2.9048,[112]2.8790,[113]2.8658,[114]2.8435,[115]2.8283,[116]2.8133,[117]2.7903,[118]2.7695,[119]2.7482,[120]2.7294,[121]2.7136,[122]2.6960,[123]2.6797,[124]2.6610,[125]2.6435,[126]2.6269,[127]2.6128,[128]2.6037,[129]2.5931,[130]2.5809,[131]2.5732,[132]2.5804,[133]2.5900,[134]2.5968,[135]2.6076,[136]2.6240,[137]2.6392,[138]2.6474,[139]2.6589,[140]2.6599,[141]2.6616,[142]2.6608,[143]2.6614,[144]2.6584,[145]2.6495,[146]2.6480,[147]2.6525,[148]2.6523,[149]2.6539,[150]2.6487,[151]2.6469,[152]2.6441,[153]2.6403,[154]2.6411,[155]2.6455,[156]2.6476,[157]2.6536,[158]2.6625,[159]2.6642,[160]2.6732,[161]2.6815,[162]2.6908,[163]2.6950,[164]2.7148,[165]2.7384,[166]2.7558,[167]2.7681,[168]2.7922,[169]2.8146,[170]2.8362,[171]2.8593,[172]2.8434,[173]2.8269,[174]2.8132,[175]2.8000,[176]2.7878,[177]2.7763,[178]2.7636,[179]2.7498,[180]2.7537,[181]2.7677,[182]2.7829,[183]2.7977,[184]2.8118,[185]2.8223,[186]2.8388,[187]2.8541,[188]2.8683,[189]2.8790,[190]2.8792,[191]2.8865,[192]2.8905,[193]2.8958,[194]2.9153,[195]2.9242,[196]2.9376,[197]2.9475,[198]2.9519,[199]2.9577,[200]2.9572,[201]2.9722,[202]2.9676,[203]2.9729,[204]2.9764,[205]2.9765,[206]2.9792,[207]2.9880,[208]2.9977,[209]3.0068,[210]3.0074,[211]3.0029,[212]3.0028,[213]3.0104,[214]3.0123,[215]3.0181,[216]3.0188,[217]3.0150,[218]3.0151,[219]3.0161,[220]3.0156,[221]3.0156,[222]3.0157,[223]3.0162,[224]3.0212,[225]3.0230,[226]3.0150,[227]3.0126,[228]3.0150,[229]3.0194,[230]3.0260,[231]3.0321,[232]3.0239,[233]3.0161,[234]3.0162,[235]3.0146,[236]3.0237,[237]3.0321,[238]3.0416,[239]3.0515,[240]3.0607,[241]3.0719,[242]3.0863,[243]3.0996,[244]3.1077,[245]3.1189,[246]3.1294,[247]3.1283,[248]3.1240,[249]3.1221,[250]3.1160,[251]3.1140,[252]3.1166,[253]3.1205,[254]3.1276,[255]3.1340,[256]3.1378,[257]3.1402,[258]3.1414,[259]3.1447,[260]3.1469,[261]3.1482,[262]3.1474,[263]3.1531,[264]3.1553,[265]3.1558,[266]3.1577,[267]3.1605,[268]3.1642,[269]3.1674,[270]3.1667,[271]3.1650,[272]3.1582,[273]3.1578,[274]3.1510,[275]3.1401,[276]3.1291,[277]3.1309,[278]3.1410,[279]3.1473,[280]3.1551,[281]3.1625,[282]3.1688,[283]3.1751,[284]3.1818,[285]3.1953,[286]3.1978,[287]3.2012,[288]3.2061,[289]3.2087,[290]3.2005,[291]3.1911,[292]3.1892,[293]3.1882,[294]3.1852,[295]3.1827,[296]3.1848,[297]3.1853,[298]3.1902,[299]3.1958,[300]3.1988,[301]3.2028,[302]3.2052,[303]3.2073,[304]3.2067,[305]3.2187,[306]3.2263,[307]3.2372,[308]3.2261,[309]3.2206,[310]3.2110,[311]3.2146,[312]3.2169,[313]3.2233,[314]3.2253,[315]3.2285,[316]3.2299,[317]3.2317,[318]3.2323,[319]3.2327,[320]3.2369,[321]3.2372,[322]3.2392,[323]3.2455,[324]3.2463,[325]3.2518,[326]3.2565,[327]3.2608,[328]3.2638,[329]3.2656,[330]3.2719,[331]3.2758,[332]3.2804,[333]3.2791,[334]3.2792,[335]3.2797,[336]3.2800,[337]3.2811,[338]3.2814,[339]3.2841,[340]3.2878,[341]3.2933,[342]3.3023,[343]3.3116,[344]3.3169,[345]3.3083,[346]3.3006,[347]3.2954,[348]3.2880,[349]3.2843,[350]3.2827,[351]3.2873,[352]3.3022,[353]3.3113,[354]3.3242,[355]3.3326,[356]3.3378,[357]3.3495,[358]3.3591,[359]3.3622,[360]3.3687,[361]3.3780,[362]3.3865,[363]3.3923,[364]3.3989,[365]3.4051,[366]3.4155,[367]3.4242,[368]3.4311,[369]3.4390,[370]3.4475,[371]3.4612,[372]3.4699,[373]3.4731,[374]3.4765,[375]3.4815,[376]3.4943,[377]3.5055,[378]3.5082,[379]3.5076,[380]3.5042,[381]3.5088,[382]3.5145,[383]3.5181,[384]3.5224,[385]3.5262,[386]3.5323,[387]3.5381,[388]3.5415,[389]3.5311,[390]3.5218,[391]3.5114,[392]3.5056,[393]3.4961,[394]3.4871,[395]3.4779,[396]3.4679,[397]3.4590,[398]3.4495,[399]3.4391,[400]3.4303,[401]3.4203,[402]3.4100,[403]3.4014,[404]3.3912,[405]3.3818,[406]3.3719,[407]3.3627,[408]3.3538,[409]3.3453,[410]3.3393,[411]3.3400,[412]3.3354,[413]3.3370,[414]3.3391,[415]3.3360,[416]3.3358,[417]3.3381,[418]3.3323,[419]3.3338,[420]3.3313,[421]3.3301,[422]3.3317,[423]3.3309,[424]3.3350,[425]3.3344,[426]3.3349,[427]3.3338,[428]3.3363,[429]3.3381,[430]3.3409,[431]3.3416,[432]3.3406,[433]3.3370,[434]3.3370,[435]3.3293,[436]3.3229,[437]3.3188,[438]3.3171,[439]3.3137,[440]3.3187,[441]3.3240,[442]3.3315,[443]3.3297,[444]3.3305,[445]3.3318,[446]3.3365,[447]3.3398,[448]3.3424,[449]3.3454,[450]3.3492,[451]3.3522,[452]3.3543,[453]3.3559,[454]3.3545,[455]3.3567,[456]3.3570,[457]3.3597,[458]3.3649,[459]3.3655,[460]3.3656,[461]3.3624,[462]3.3662,[463]3.3734,[464]3.3787,[465]3.3716,[466]3.3695,[467]3.3676,[468]3.3686,[469]3.3657,[470]3.3630,[471]3.3633,[472]3.3641,[473]3.3632,[474]3.3623,[475]3.3636,[476]3.3620,[477]3.3610,[478]3.3618,[479]3.3635,[480]3.3662,[481]3.3622,[482]3.3656,[483]3.3648,[484]3.3683,[485]3.3748,[486]3.3777,[487]3.3814,[488]3.3867,[489]3.3892,[490]3.3938,[491]3.3999,[492]3.4044,[493]3.4042,[494]3.4053,[495]3.4079,[496]3.4097,[497]3.4126,[498]3.4129,[499]3.4124,[500]3.4164,[501]3.4211,[502]3.4202,[503]3.4186,[504]3.4207,[505]3.4241,[506]3.4326,[507]3.4354,[508]3.4389,[509]3.4316,[510]3.4258,[511]3.4193,[512]3.4147,[513]3.4084,[514]3.4069,[515]3.4089,[516]3.4039,[517]3.4036,[518]3.4028,[519]3.4034,[520]3.4078,[521]3.4066,[522]3.4053,[523]3.4111,[524]3.4098,[525]3.4083,[526]3.4034,[527]3.3984,[528]3.3949,[529]3.3919,[530]3.3888,[531]3.3858,[532]3.3803,[533]3.3741,[534]3.3698,[535]3.3706,[536]3.3733,[537]3.3765,[538]3.3789,[539]3.3816,[540]3.3868,[541]3.3901,[542]3.3924,[543]3.3869,[544]3.3826,[545]3.3822,[546]3.3756,[547]3.3691,[548]3.3627,[549]3.3560,[550]3.3499,[551]3.3437,[552]3.3379,[553]3.3320,[554]3.3299,[555]3.3285,[556]3.3313,[557]3.3353,[558]3.3412,[559]3.3457,[560]3.3509,[561]3.3491,
llama_print_timings:        load time =    4673.16 ms
llama_print_timings:      sample time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings: prompt eval time = 2385511.51 ms / 287232 tokens (    8.31 ms per token,   120.41 tokens per second)
llama_print_timings:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings:       total time = 2597613.67 ms / 287233 tokens

Final estimate: PPL = 3.3491 +/- 0.01849
```

</details>

---

👤 **ikawrakow** commented on **2025-03-27** at **04:49:07**

Thank you for verifying that it works!

---

👤 **saood06** commented on **2025-03-27** at **08:14:07**

> Closes [#285](https://github.com/ikawrakow/ik_llama.cpp/issues/285) and [#196](https://github.com/ikawrakow/ik_llama.cpp/issues/196)

This only closed [#285](https://github.com/ikawrakow/ik_llama.cpp/issues/285), for multiple commands need to use a comma and repeat each command ([source](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/linking-a-pull-request-to-an-issue)).

---

👤 **saood06** commented on **2025-03-27** at **08:23:08**

>So, it seems one can not use fp16 arithmetic in DeepSeek-V3/R1.

Is this why https://github.com/ikawrakow/ik_llama.cpp/discussions/242#discussioncomment-12429240 the imatrix in that comment was failing?

---

👤 **ikawrakow** commented on **2025-03-27** at **08:27:17**

> Is this why https://github.com/ikawrakow/ik_llama.cpp/discussions/242#discussioncomment-12429240 the imatrix in that comment was failing?

With a very high degree of probability, yes. I get NaNs even for DeepSeek-Lite when I use the `fp16` model on the GPU.