## 🔀 [Pull Request #291](https://github.com/ikawrakow/ik_llama.cpp/pull/291) - Disable Zen4 optimizations for Q8_0/Q8_0_R8

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Source Branch** | `ik/test_q80_NaNs` |
| **Target Branch** | `main` |
| **Created** | 2025-03-26 |
| **Updated** | 2025-03-27 |

---

## 📄 Description

The purpose of this PR is to test if the NaNs observed for `Q8_0/Q8_0_R8` quantized DeepSeekV3/R1 will go away ([#285](https://github.com/ikawrakow/ik_llama.cpp/issues/285))

My hypothesis is that we get an overflow in the block sum of `Q8_1/Q8_1_X4`, which is stored as `fp16`. `Q8_1/Q8_1_X4` is used for activation quantization on Zen4 for `Q8_0/Q8_0_R8` quants. See also [#196](https://github.com/ikawrakow/ik_llama.cpp/issues/196) 
  
The PR disables the Zen4 optimization and reverts to the vanilla `AVX2` implementation, which uses `Q8_0` (just like mainline `llama.cpp`).

Performance goes down quite a bit, but if we confirm that the change eliminates the NaNs, I will make a better PR that keeps the performance while avoiding the NaNs.

---

## 💬 Conversation

👤 **ubergarm** commented on **2025-03-26** at **15:17:57**

*UPDATE* Finished successfully. Complete perplexity log shown now. Thanks!

---

I gotta head out for a night or two, but will bring my laptop and hope to check in.

I'm leaving a run going now, initial results are looking promising. Check full logs :point_down: 

<details>

<summary>repacked `q8_0_r8` `llama-perplexity` logs</summary>

I'm guessing the borked messages are because of how stderr is piped to stdout and not actually a race condition. I saw similar output with llama-sweep-bench last night but it was running okay psure.

```bash
$ git branch | grep NaN
* ik/test_q80_NaNs

$ git rev-parse --short HEAD
2089147a

$ numactl -N 1 -m 1 \
./build/bin/llama-perplexity \
    -m /mnt/ai/models/unsloth/repack/DeepSeek-R1-Q8_0_R8.gguf \
    -f wiki.test.raw \
    -t 128 \
    -b 512 \
    --numa numactl 2>&1 | tee -a output.log

main: build = 3611 (2089147a)
main: built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu
main: seed  = 1743000715
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
perplexity: tokenization took 911.451 ms
perplexity: calculating perplexity over 561 chunks, n_ctx=512, batch_size=512, n_seq=1
perplexity: 6.36 seconds per pass - ETA 59.47 minutes
mputed blk.54.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.55.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.56.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.57.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.58.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.59.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.60.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
[1]2.5068,[2]3.2747,[3]2.3640,[4]1.9767,[5]1.7836,[6]1.6435,[7]1.5510,[8]1.4847,[9]1.4354,[10]1.3962,[11]1.3807,[12]1.4146,[13]1.4261,[14]1.5535,[15]1.6850,[16]1.7449,[17]1.9072,[18]2.0355,[19]1.9972,[20]1.9858,[21]2.0933,[22]2.0657,[23]2.0388,[24]2.0524,[25]2.0237,[26]2.0005,[27]2.0483,[28]2.0558,[29]2.1043,[30]2.1353,[31]2.1692,[32]2.1872,[33]2.2279,[34]2.2686,[35]2.3178,[36]2.3707,[37]2.4059,[38]2.4519,[39]2.4927,[40]2.5518,[41]2.5946,[42]2.6070,[43]2.6557,[44]2.6716,[45]2.7512,[46]2.8016,[47]2.7570,[48]2.7106,[49]2.6842,[50]2.7037,[51]2.7502,[52]2.7650,[53]2.8155,[54]2.8279,[55]2.8581,[56]2.8891,[57]2.9030,[58]2.9395,[59]2.9508,[60]2.9974,[61]3.0373,[62]3.0910,[63]3.1225,[64]3.1665,[65]3.1766,[66]3.1596,[67]3.1367,[68]3.1680,[69]3.1627,[70]3.1774,[71]3.1957,[72]3.2116,[73]3.2261,[74]3.2490,[75]3.2284,[76]3.1815,[77]3.1386,[78]3.1336,[79]3.1111,[80]3.0923,[81]3.0557,[82]3.0591,[83]3.0273,[84]2.9912,[85]2.9563,[86]2.9312,[87]2.9238,[88]2.8955,[89]2.8785,[90]2.8526,[91]2.8228,[92]2.7979,[93]2.7711,[94]2.7445,[95]2.7210,[96]2.7197,[97]2.7267,[98]2.7113,[99]2.6942,[100]2.6965,[101]2.6881,[102]2.7047,[103]2.7309,[104]2.7490,[105]2.7459,[106]2.7686,[107]2.7931,[108]2.8142,[109]2.8481,[110]2.8820,[111]2.9021,[112]2.8764,[113]2.8634,[114]2.8412,[115]2.8259,[116]2.8103,[117]2.7873,[118]2.7665,[119]2.7453,[120]2.7265,[121]2.7110,[122]2.6934,[123]2.6770,[124]2.6583,[125]2.6407,[126]2.6241,[127]2.6098,[128]2.6007,[129]2.5902,[130]2.5779,[131]2.5704,[132]2.5777,[133]2.5873,[134]2.5939,[135]2.6045,[136]2.6209,[137]2.6363,[138]2.6443,[139]2.6560,[140]2.6570,[141]2.6588,[142]2.6581,[143]2.6587,[144]2.6557,[145]2.6469,[146]2.6456,[147]2.6501,[148]2.6500,[149]2.6515,[150]2.6464,[151]2.6445,[152]2.6419,[153]2.6382,[154]2.6390,[155]2.6434,[156]2.6455,[157]2.6517,[158]2.6607,[159]2.6625,[160]2.6715,[161]2.6798,[162]2.6891,[163]2.6934,[164]2.7133,[165]2.7368,[166]2.7541,[167]2.7663,[168]2.7908,[169]2.8131,[170]2.8348,[171]2.8579,[172]2.8420,[173]2.8255,[174]2.8118,[175]2.7986,[176]2.7863,[177]2.7747,[178]2.7621,[179]2.7483,[180]2.7522,[181]2.7662,[182]2.7813,[183]2.7960,[184]2.8102,[185]2.8208,[186]2.8373,[187]2.8526,[188]2.8666,[189]2.8772,[190]2.8775,[191]2.8849,[192]2.8888,[193]2.8940,[194]2.9137,[195]2.9224,[196]2.9359,[197]2.9459,[198]2.9504,[199]2.9561,[200]2.9556,[201]2.9707,[202]2.9660,[203]2.9715,[204]2.9751,[205]2.9749,[206]2.9776,[207]2.9864,[208]2.9961,[209]3.0055,[210]3.0059,[211]3.0013,[212]3.0012,[213]3.0088,[214]3.0108,[215]3.0165,[216]3.0171,[217]3.0132,[218]3.0133,[219]3.0144,[220]3.0137,[221]3.0138,[222]3.0139,[223]3.0144,[224]3.0196,[225]3.0216,[226]3.0137,[227]3.0114,[228]3.0138,[229]3.0183,[230]3.0247,[231]3.0309,[232]3.0227,[233]3.0149,[234]3.0150,[235]3.0132,[236]3.0221,[237]3.0305,[238]3.0400,[239]3.0500,[240]3.0591,[241]3.0703,[242]3.0849,[243]3.0983,[244]3.1064,[245]3.1178,[246]3.1281,[247]3.1269,[248]3.1227,[249]3.1208,[250]3.1146,[251]3.1126,[252]3.1152,[253]3.1190,[254]3.1261,[255]3.1326,[256]3.1363,[257]3.1387,[258]3.1399,[259]3.1432,[260]3.1453,[261]3.1467,[262]3.1459,[263]3.1516,[264]3.1539,[265]3.1544,[266]3.1562,[267]3.1591,[268]3.1629,[269]3.1661,[270]3.1654,[271]3.1637,[272]3.1571,[273]3.1570,[274]3.1502,[275]3.1395,[276]3.1286,[277]3.1304,[278]3.1404,[279]3.1468,[280]3.1547,[281]3.1621,[282]3.1684,[283]3.1749,[284]3.1816,[285]3.1951,[286]3.1976,[287]3.2010,[288]3.2058,[289]3.2084,[290]3.2002,[291]3.1907,[292]3.1886,[293]3.1876,[294]3.1847,[295]3.1821,[296]3.1841,[297]3.1845,[298]3.1894,[299]3.1954,[300]3.1984,[301]3.2022,[302]3.2044,[303]3.2064,[304]3.2058,[305]3.2177,[306]3.2253,[307]3.2362,[308]3.2251,[309]3.2196,[310]3.2101,[311]3.2136,[312]3.2159,[313]3.2223,[314]3.2244,[315]3.2275,[316]3.2290,[317]3.2309,[318]3.2315,[319]3.2319,[320]3.2361,[321]3.2365,[322]3.2386,[323]3.2451,[324]3.2458,[325]3.2513,[326]3.2560,[327]3.2601,[328]3.2631,[329]3.2649,[330]3.2712,[331]3.2750,[332]3.2798,[333]3.2784,[334]3.2784,[335]3.2790,[336]3.2791,[337]3.2801,[338]3.2804,[339]3.2831,[340]3.2868,[341]3.2923,[342]3.3013,[343]3.3107,[344]3.3160,[345]3.3076,[346]3.2998,[347]3.2948,[348]3.2874,[349]3.2838,[350]3.2820,[351]3.2866,[352]3.3015,[353]3.3106,[354]3.3234,[355]3.3319,[356]3.3371,[357]3.3487,[358]3.3583,[359]3.3615,[360]3.3680,[361]3.3772,[362]3.3858,[363]3.3914,[364]3.3979,[365]3.4040,[366]3.4144,[367]3.4231,[368]3.4298,[369]3.4376,[370]3.4460,[371]3.4597,[372]3.4685,[373]3.4718,[374]3.4752,[375]3.4801,[376]3.4930,[377]3.5043,[378]3.5070,[379]3.5064,[380]3.5030,[381]3.5076,[382]3.5133,[383]3.5169,[384]3.5213,[385]3.5252,[386]3.5313,[387]3.5371,[388]3.5404,[389]3.5301,[390]3.5207,[391]3.5102,[392]3.5046,[393]3.4950,[394]3.4861,[395]3.4769,[396]3.4669,[397]3.4580,[398]3.4485,[399]3.4383,[400]3.4295,[401]3.4195,[402]3.4092,[403]3.4005,[404]3.3904,[405]3.3809,[406]3.3710,[407]3.3617,[408]3.3529,[409]3.3444,[410]3.3384,[411]3.3391,[412]3.3343,[413]3.3361,[414]3.3381,[415]3.3350,[416]3.3347,[417]3.3370,[418]3.3312,[419]3.3326,[420]3.3302,[421]3.3291,[422]3.3306,[423]3.3298,[424]3.3339,[425]3.3334,[426]3.3339,[427]3.3328,[428]3.3352,[429]3.3371,[430]3.3400,[431]3.3407,[432]3.3397,[433]3.3361,[434]3.3361,[435]3.3284,[436]3.3221,[437]3.3181,[438]3.3163,[439]3.3130,[440]3.3179,[441]3.3233,[442]3.3308,[443]3.3291,[444]3.3299,[445]3.3312,[446]3.3360,[447]3.3392,[448]3.3418,[449]3.3449,[450]3.3487,[451]3.3516,[452]3.3538,[453]3.3553,[454]3.3540,[455]3.3562,[456]3.3564,[457]3.3592,[458]3.3644,[459]3.3651,[460]3.3653,[461]3.3621,[462]3.3658,[463]3.3730,[464]3.3783,[465]3.3712,[466]3.3692,[467]3.3672,[468]3.3681,[469]3.3651,[470]3.3624,[471]3.3628,[472]3.3635,[473]3.3627,[474]3.3617,[475]3.3629,[476]3.3613,[477]3.3603,[478]3.3612,[479]3.3628,[480]3.3655,[481]3.3616,[482]3.3650,[483]3.3641,[484]3.3677,[485]3.3741,[486]3.3769,[487]3.3805,[488]3.3858,[489]3.3882,[490]3.3929,[491]3.3991,[492]3.4036,[493]3.4034,[494]3.4046,[495]3.4071,[496]3.4090,[497]3.4120,[498]3.4123,[499]3.4117,[500]3.4158,[501]3.4204,[502]3.4196,[503]3.4181,[504]3.4202,[505]3.4235,[506]3.4319,[507]3.4347,[508]3.4380,[509]3.4307,[510]3.4250,[511]3.4184,[512]3.4138,[513]3.4075,[514]3.4059,[515]3.4078,[516]3.4029,[517]3.4027,[518]3.4018,[519]3.4023,[520]3.4067,[521]3.4055,[522]3.4042,[523]3.4099,[524]3.4087,[525]3.4071,[526]3.4022,[527]3.3972,[528]3.3937,[529]3.3908,[530]3.3878,[531]3.3848,[532]3.3793,[533]3.3731,[534]3.3688,[535]3.3697,[536]3.3724,[537]3.3755,[538]3.3780,[539]3.3806,[540]3.3858,[541]3.3891,[542]3.3914,[543]3.3857,[544]3.3815,[545]3.3811,[546]3.3745,[547]3.3680,[548]3.3616,[549]3.3549,[550]3.3489,[551]3.3428,[552]3.3370,[553]3.3311,[554]3.3290,[555]3.3275,[556]3.3303,[557]3.3344,[558]3.3402,[559]3.3447,[560]3.3499,[561]3.3482,
llama_print_timings:        load time = 1021262.28 ms
llama_print_timings:      sample time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings: prompt eval time = 3230181.16 ms / 287232 tokens (   11.25 ms per token,    88.92 tokens per second)
llama_print_timings:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings:       total time = 3434307.80 ms / 287233 tokens

Final estimate: PPL = 3.3482 +/- 0.01847
```

</details>

---

👤 **ubergarm** commented on **2025-03-26** at **18:52:28**

Finished successfully, just updated logs. Thanks!

---

👤 **ubergarm** commented on **2025-03-26** at **19:28:51**

Oh nice, seems like with this patch I'm also able to get an imatrix going with MLA tensors on the `V3-0324` `q8_0` gguf I recently made.  *Finished* cooking, logs look good :point_down: 

<details>

<summary>llama-imatrix run on q8_0</summary>

```bash
# download https://gist.github.com/tristandruyen/9e207a95c7d75ddf37525d353e00659c#file-calibration_data_v5_rc-txt

$ git rev-parse --short HEAD
2089147a

$ numactl -N 1 -m 1 \
./build/bin/llama-imatrix \
    --verbosity 1 \
    -m /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-Q8_0.gguf \
    -f calibration_data_v5_rc.txt \
    -o /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-$(git rev-parse --short HEAD).dat \
    --ctx-size 512 \
    --numa numactl \
    --threads 128 2>&1 | tee -a output.log

llama_model_loader: loaded meta data with 46 key-value pairs and 1147 tensors from /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-Q8_0.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = DeepSeek V3 0324
llama_model_loader: - kv   3:                            general.version str              = V3-0324
llama_model_loader: - kv   4:                           general.basename str              = DeepSeek
llama_model_loader: - kv   5:                         general.size_label str              = 256x21B
llama_model_loader: - kv   6:                            general.license str              = mit
llama_model_loader: - kv   7:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv   8:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv   9:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv  10:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  11:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv  12:          deepseek2.attention.head_count_kv u32              = 128
llama_model_loader: - kv  13:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  14: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  15:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  16:                          general.file_type u32              = 7
llama_model_loader: - kv  17:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  18:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  19:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  20:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  21:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  22:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  23:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  24:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  25:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  26:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  27:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  28:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  29:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  30:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  31:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  32: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  33: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  34:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  35:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  36:                      tokenizer.ggml.tokens arr[str,129280]  = ["
llama_model_loader: - kv  37:                  tokenizer.ggml.token_type arr[i32,129280]  = [3
llama_model_loader: - kv  38:                      tokenizer.ggml.merges arr[str,127741]  = ["
llama_model_loader: - kv  39:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  40:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  41:            tokenizer.ggml.padding_token_id u32              = 1
llama_model_loader: - kv  42:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  43:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  44:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  45:               general.quantization_version u32              = 2
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  786 tensors
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
llm_load_print_meta: model ftype      = Q8_0
llm_load_print_meta: model params     = 672.050 B
llm_load_print_meta: model size       = 665.308 GiB (8.504 BPW) 
llm_load_print_meta: repeating layers = 663.474 GiB (8.504 BPW, 670.196 B parameters)
llm_load_print_meta: general.name     = DeepSeek V3 0324
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
llm_load_tensors:        CPU buffer size = 681274.97 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 512
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
compute_imatrix: tokenizing the input ..
compute_imatrix: tokenization took 313.289 ms
compute_imatrix: computing over 213 chunks with batch_size 512
compute_imatrix: 41.77 seconds per pass - ETA 2 hours 28.28 minutes
[1]60.9029,[2]10.8011,[3]5.8709,[4]3.7872,[5]2.9688,[6]2.5088,[7]2.2214,[8]2.0224,[9]1.9110,
save_imatrix: entry '             blk.60.ffn_down_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.60.ffn_gate_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.60.ffn_up_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**

save_imatrix: stored collected data after 10 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-2089147a.dat
[10]1.8230,[11]2.0314,[12]2.0866,[13]2.1000,[14]2.1455,[15]2.0412,[16]1.9535,[17]1.8827,[18]1.8197,[19]1.7778,
save_imatrix: stored collected data after 20 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-2089147a.dat
[20]1.7349,[21]1.7018,[22]1.6640,[23]1.6347,[24]1.6222,[25]1.6104,[26]1.5849,[27]1.6838,[28]1.7577,[29]1.8237,
save_imatrix: stored collected data after 30 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-2089147a.dat
[30]1.8219,[31]1.8354,[32]1.8351,[33]1.8125,[34]1.8489,[35]1.8250,[36]1.8245,[37]1.8131,[38]1.8239,[39]1.8108,
save_imatrix: stored collected data after 40 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-2089147a.dat
[40]1.7876,[41]1.7643,[42]1.7444,[43]1.7325,[44]1.7193,[45]1.7059,[46]1.7016,[47]1.6954,[48]1.6846,[49]1.6741,
save_imatrix: stored collected data after 50 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-2089147a.dat
[50]1.6684,[51]1.6656,[52]1.6657,[53]1.6704,[54]1.6844,[55]1.6811,[56]1.6712,[57]1.6794,[58]1.6833,[59]1.6943,
save_imatrix: stored collected data after 60 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-2089147a.dat

.
.
.

[210]3.5371,[211]3.5164,[212]3.4959,[213]3.4755,
save_imatrix: stored collected data after 213 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-2089147a.dat

llama_print_timings:        load time =   42726.11 ms
llama_print_timings:      sample time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings: prompt eval time = 7125661.28 ms / 109056 tokens (   65.34 ms per token,    15.30 tokens per second)
llama_print_timings:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings:       total time = 7201368.59 ms / 109057 tokens

Final estimate: PPL = 3.4755 +/- 0.03305
```

</details>

---

👤 **ikawrakow** commented on **2025-03-27** at **04:49:39**

Close in favor of [#292](https://github.com/ikawrakow/ik_llama.cpp/issues/292)