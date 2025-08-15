### 🔀 [#326](https://github.com/ikawrakow/ik_llama.cpp/pull/326) - WIP Compute per layer LIM Scores during imatrix

| **Author** | `ubergarm` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-04-13 |
| **Updated** | 2025-04-16 |

---

#### Description

*WARNING*: This is mostly vibe code. Hope I'm not wasting y'alls time.

Compute Layer Importance Modification (LIM) Scores

The goal of this PR is to rank layers of a given tensor in order of sensitivity to quantization error. Given that it is now possible to use `llama-quantize --custom-q ...` regex, it may be possible to use these LIM Scores to decide which layers of a given tensor to quantize more or less in an attempt to preserve generation quality (e.g. low perplexity) while reducing memory footprint as compared to using same quant size across all layers of a given tensor.

This experimental PR was motivated by this comment and PR: https://github.com/ggml-org/llama.cpp/pull/12718#issuecomment-2781723233 (*EDIT* fixed link directly to comment)

I may force-push this after more testing and experimenting to see if it is actually doing the right thing and if the output is actually useful to improve quantization quality e.g. PPL per GiB... This may just be a big mistake, lol.

This is built on existing imatrix computation and assumes that values of `x[j]` are the "activations" coming right in/out of the given tensor layer. I don't know GGML and generally work in python or vanilla c not so much c++.  So a lot of this was vibe coded running [ubergarm/DeepSeek-V3-0324-GGUF IQ4_K_R4 quant](https://huggingface.co/ubergarm/DeepSeek-V3-0324-GGUF/tree/main/DeepSeek-V3-0324-IQ4_K_R4). So this is partially an experiment actually trying to use an LLM instead of just enjoying the meta of manual quantization min-maxing.

## TODO

- [x] test locally on `Qwen/CodeQwen1.5-7B-Chat-GGUF` `q8_0`
- [x] test on `ubergarm/DeepSeek-V3-0324-GGUF` `q8_0`
- [ ] Use LIM Scores to generate a `--custom-q` regex and compare PPL per GiB
- [ ] cleanup code and actually gate computation based on input param
- [ ] consider usability as it just dumps a lot of stuff when you may just want the imatrix PPL information

## Reference
```
@misc{dumitru2024layerwisequantizationpragmaticeffective,
      title={Layer-Wise Quantization: A Pragmatic and Effective Method for Quantizing LLMs Beyond Integer Bit-Levels},
      author={Razvan-Gabriel Dumitru and Vikas Yadav and Rishabh Maheshwary and Paul-Ioan Clotan and Sathwik Tejaswi Madhusudhan and Mihai Surdeanu},
      year={2024},
      eprint={2406.17415},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.17415},
      code={https://github.com/RazvanDu/LayerwiseQuant/},
}
```

## Logs

<details>

<summary>llama-imatrix run printing out what hopefully are actually LIM scores</summary>

```bash
numactl -N 1 -m 1 \
./build/bin/llama-imatrix \
    --verbosity 1 \
    -m /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-Q8_0.gguf \
    -f calibration_data_v5_rc.txt \
    -o imatrix.dat \
    --ctx-size 512 \
    --numa numactl \
    --threads 128

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
compute_imatrix: tokenization took 312.531 ms
compute_imatrix: computing over 213 chunks with batch_size 512
compute_imatrix: 53.45 seconds per pass - ETA 3 hours 9.73 minutes
[1]60.9619,[2]10.7701,[3]5.8724,[4]3.7883,[5]2.9691,[6]2.5089,[7]2.2199,[8]2.0199,[9]1.9095,
save_imatrix: entry '             blk.60.ffn_down_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.60.ffn_gate_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.60.ffn_up_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.25.ffn_down_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.26.ffn_down_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.25.ffn_up_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.25.ffn_gate_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.26.ffn_gate_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.26.ffn_up_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**

save_imatrix: stored collected data after 10 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-0e808309.dat
[10]1.8219,[11]2.0296,[12]2.0839,[13]2.0978,[14]2.1403,[15]2.0365,[16]1.9492,[17]1.8786,[18]1.8160,[19]1.7743,
save_imatrix: stored collected data after 20 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-0e808309.dat
[20]1.7315,[21]1.6986,[22]1.6609,[23]1.6319,[24]1.6201,[25]1.6080,[26]1.5822,[27]1.6812,[28]1.7547,[29]1.8204,
save_imatrix: stored collected data after 30 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-0e808309.dat
[30]1.8188,[31]1.8323,[32]1.8317,[33]1.8091,[34]1.8457,[35]1.8217,[36]1.8215,[37]1.8106,[38]1.8208,[39]1.8070,
save_imatrix: stored collected data after 40 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-0e808309.dat
[40]1.7838,[41]1.7606,[42]1.7410,[43]1.7291,[44]1.7157,[45]1.7023,[46]1.6981,[47]1.6919,[48]1.6811,[49]1.6707,
save_imatrix: stored collected data after 50 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-0e808309.dat
[50]1.6650,[51]1.6623,[52]1.6625,[53]1.6672,[54]1.6812,[55]1.6781,[56]1.6683,[57]1.6764,[58]1.6796,[59]1.6906,
save_imatrix: stored collected data after 60 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-0e808309.dat
[60]1.6855,[61]1.7243,[62]1.7565,[63]1.7884,[64]1.8197,[65]1.8677,[66]1.8802,[67]1.9148,[68]1.9442,[69]1.9996,
save_imatrix: stored collected data after 70 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-0e808309.dat
[70]2.0525,[71]2.0832,[72]2.1136,[73]2.1258,[74]2.1407,[75]2.1702,[76]2.2011,[77]2.2185,[78]2.2164,[79]2.2313,
save_imatrix: stored collected data after 80 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-0e808309.dat
[80]2.2543,[81]2.2904,[82]2.3238,[83]2.3342,[84]2.3650,[85]2.3733,[86]2.3730,[87]2.4024,[88]2.4344,[89]2.4899,
save_imatrix: stored collected data after 90 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-0e808309.dat
[90]2.5102,[91]2.5125,[92]2.5192,[93]2.5349,[94]2.5452,[95]2.5779,[96]2.5670,[97]2.6058,[98]2.6319,[99]2.6214,
save_imatrix: stored collected data after 100 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-0e808309.dat
[100]2.6537,[101]2.7008,[102]2.7326,[103]2.7740,[104]2.8020,[105]2.8310,[106]2.8682,[107]2.8605,[108]2.8789,[109]2.8849,
save_imatrix: stored collected data after 110 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-0e808309.dat
[110]2.8910,[111]2.8878,[112]2.9177,[113]2.9435,[114]2.9520,[115]2.9363,[116]2.9104,[117]2.9044,[118]2.9147,[119]2.9003,
save_imatrix: stored collected data after 120 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-0e808309.dat
[120]2.8773,[121]2.8737,[122]2.8738,[123]2.8819,[124]2.8872,[125]2.8942,[126]2.9018,[127]2.9043,[128]2.9343,[129]2.9484,
save_imatrix: stored collected data after 130 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-0e808309.dat
[130]2.9241,[131]2.9003,[132]2.8771,[133]2.8544,[134]2.8563,[135]2.8567,[136]2.8828,[137]2.9150,[138]2.9340,[139]2.9389,
save_imatrix: stored collected data after 140 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-0e808309.dat
[140]2.9637,[141]2.9866,[142]3.0151,[143]3.0354,[144]3.0569,[145]3.0766,[146]3.0972,[147]3.1154,[148]3.1266,[149]3.1351,
save_imatrix: stored collected data after 150 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-0e808309.dat
[150]3.1395,[151]3.1572,[152]3.1761,[153]3.1759,[154]3.1834,[155]3.1945,[156]3.2035,[157]3.2148,[158]3.2209,[159]3.2300,
save_imatrix: stored collected data after 160 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-0e808309.dat
[160]3.2442,[161]3.2498,[162]3.2525,[163]3.2595,[164]3.2704,[165]3.2724,[166]3.2737,[167]3.2912,[168]3.3010,[169]3.3082,
save_imatrix: stored collected data after 170 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-0e808309.dat
[170]3.3258,[171]3.3403,[172]3.3354,[173]3.3417,[174]3.3424,[175]3.3575,[176]3.3691,[177]3.3818,[178]3.3768,[179]3.3734,
save_imatrix: stored collected data after 180 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-0e808309.dat
[180]3.3682,[181]3.3635,[182]3.3578,[183]3.3531,[184]3.3472,[185]3.3600,[186]3.3887,[187]3.4121,[188]3.4336,[189]3.4550,
save_imatrix: stored collected data after 190 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-0e808309.dat
[190]3.4850,[191]3.4990,[192]3.5134,[193]3.5036,[194]3.5210,[195]3.5145,[196]3.4953,[197]3.4747,[198]3.4946,[199]3.5110,
save_imatrix: stored collected data after 200 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-0e808309.dat
[200]3.5207,[201]3.5290,[202]3.5447,[203]3.5621,[204]3.5748,[205]3.5874,[206]3.6021,[207]3.5989,[208]3.5771,[209]3.5556,
save_imatrix: stored collected data after 210 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-0e808309.dat
[210]3.5342,[211]3.5134,[212]3.4930,[213]3.4727,
save_imatrix: stored collected data after 213 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-0e808309.dat

llama_print_timings:        load time =   54390.61 ms
llama_print_timings:      sample time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings: prompt eval time = 10568880.33 ms / 109056 tokens (   96.91 ms per token,    10.32 tokens per second)
llama_print_timings:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings:       total time = 10644363.84 ms / 109057 tokens

Final estimate: PPL = 3.4727 +/- 0.03300

===
Computing Layer Importance Modification (LIM) Scores...

Tensor: ffn_down
Layer	LIM Score
-----	---------
0	-0.0005
1	0.0003

Tensor: ffn_gate
Layer	LIM Score
-----	---------
0	-0.9435
1	-0.9339

Tensor: attn_kv_b
Layer	LIM Score
-----	---------
0	0.0158
1	-0.0101
2	0.1035
3	0.0725
4	0.0570
5	-0.1063
6	-0.0104
7	-0.0682
8	0.0010
9	-0.0483
10	0.0071
11	-0.0183
12	0.0444
13	-0.0155
14	-0.0235
15	-0.0039
16	-0.0144
17	0.0431
18	0.1076
19	0.0789
20	-0.0668
21	-0.0136
22	-0.0317
23	0.0152
24	0.0210
25	-0.0111
26	0.0289
27	0.0192
28	-0.0513
29	0.0366
30	0.0046
31	-0.0151
32	-0.0159
33	0.0894
34	0.0484
35	0.0126
36	0.0168
37	-0.0292
38	0.0405
39	-0.0329
40	0.0770
41	0.0044
42	0.0064
43	0.0106
44	0.0041
45	0.0120
46	-0.0012
47	-0.0506
48	-0.0222
49	0.0434
50	0.0409
51	0.0133
52	0.0315
53	0.0141
54	0.0002
55	-0.0269
56	-0.0391
57	0.0213
58	0.0365
59	-0.0249

Tensor: attn_q_a
Layer	LIM Score
-----	---------
0	-0.4179
1	-0.8773
2	-0.9436
3	-0.9022
4	-0.9166
5	-0.9418
6	-0.9812
7	-0.9599
8	-0.9085
9	-0.9724
10	-0.9882
11	-0.9868
12	-0.9906
13	-0.9816
14	-0.9827
15	-0.9766
16	-0.9590
17	-0.9474
18	-0.9573
19	-0.9601
20	-0.9553
21	-0.9345
22	-0.9042
23	-0.9299
24	-0.9555
25	-0.9554
26	-0.9598
27	-0.9575
28	-0.9610
29	-0.9634
30	-0.9601
31	-0.9572
32	-0.9674
33	-0.9619
34	-0.9707
35	-0.9493
36	-0.9801
37	-0.9702
38	-0.9737
39	-0.9567
40	-0.9366
41	-0.9667
42	-0.9751
43	-0.9566
44	-0.9488
45	-0.9364
46	-0.9516
47	-0.9355
48	-0.9723
49	-0.9630
50	-0.9702
51	-0.9591
52	-0.9670
53	-0.8937
54	-0.9420
55	-0.9566
56	-0.9543
57	-0.8239
58	-0.8915
59	-0.9073

Tensor: ffn_up
Layer	LIM Score
-----	---------
0	-0.9435
1	-0.9339

Tensor: ffn_gate_shexp
Layer	LIM Score
-----	---------
3	-0.9355
4	-0.9365
5	-0.9068
6	-0.9485
7	-0.9117
8	-0.8524
9	-0.9458
10	-0.9404
11	-0.9593
12	-0.9458
13	-0.9364
14	-0.9494
15	-0.8997
16	-0.9017
17	-0.8748
18	-0.8369
19	-0.9108
20	-0.8583
21	-0.8067
22	-0.8093
23	-0.8568
24	-0.8719
25	-0.8983
26	-0.9103
27	-0.8789
28	-0.9135
29	-0.9107
30	-0.8975
31	-0.9346
32	-0.9335
33	-0.9334
34	-0.9343
35	-0.9524
36	-0.9404
37	-0.9573
38	-0.9487
39	-0.8949
40	-0.9070
41	-0.9669
42	-0.9815
43	-0.9481
44	-0.9233
45	-0.9606
46	-0.9472
47	-0.9145
48	-0.9580
49	-0.9672
50	-0.9689
51	-0.9570
52	-0.9670
53	-0.9735
54	-0.9553
55	-0.9542
56	-0.9671
57	-0.9526
58	-0.9285
59	-0.9185

Tensor: attn_output
Layer	LIM Score
-----	---------
0	-0.0085
1	-0.0031
2	-0.0161
3	0.0021
4	-0.0048
5	-0.0054
6	-0.0048
7	0.0039
8	0.0093
9	0.0012
10	0.0088
11	0.0053
12	-0.0081
13	-0.0059
14	-0.0070
15	0.0006
16	-0.0065
17	-0.0013
18	-0.0146
19	0.0130
20	0.0002
21	0.0036
22	0.0010
23	-0.0060
24	-0.0079
25	0.0084
26	0.0084
27	0.0064
28	0.0000
29	0.0105
30	-0.0013
31	-0.0003
32	-0.0054
33	0.0022
34	-0.0029
35	-0.0028
36	0.0048
37	0.0044
38	-0.0011
39	-0.0155
40	0.0008
41	-0.0222
42	0.0034
43	0.0029
44	0.0060
45	-0.0064
46	0.0054
47	-0.0042
48	0.0226
49	-0.0025
50	-0.0013
51	-0.0026
52	-0.0077
53	-0.0047
54	0.0012
55	-0.0097
56	-0.0060
57	-0.0017
58	-0.0126
59	-0.0006

Tensor: attn_q_b
Layer	LIM Score
-----	---------
0	-0.0019
1	0.0326
2	-0.0428
3	0.0138
4	-0.0080
5	0.0039
6	-0.0023
7	0.0048
8	-0.0020
9	-0.0183
10	-0.0130
11	0.0098
12	-0.0203
13	0.0459
14	-0.0151
15	0.0240
16	-0.0004
17	0.0102
18	0.0228
19	-0.0027
20	0.0248
21	-0.0085
22	-0.0558
23	0.0006
24	0.0064
25	0.0101
26	0.0460
27	-0.0457
28	0.0438
29	0.0190
30	0.0018
31	-0.0275
32	0.0409
33	-0.0184
34	0.0215
35	-0.0329
36	0.0059
37	-0.0366
38	-0.0044
39	0.0191
40	-0.0017
41	-0.0191
42	-0.0314
43	-0.0303
44	0.0249
45	0.0063
46	0.0204
47	-0.0585
48	-0.0175
49	0.0103
50	-0.0059
51	-0.0109
52	-0.0188
53	-0.0267
54	-0.0126
55	0.0192
56	-0.0573
57	-0.0073
58	0.0007
59	0.0150

Tensor: ffn_up_exps
Layer	LIM Score
-----	---------
3	-0.5456
4	-0.4082
5	-0.2537
6	-0.1726
7	-0.1470
8	-0.1202
9	-0.1336
10	-0.1300
11	-0.1028
12	-0.0907
13	-0.0846
14	-0.1017
15	-0.1079
16	-0.1087
17	-0.1140
18	-0.1238
19	-0.1185
20	-0.1048
21	-0.1017
22	-0.1183
23	-0.1191
24	-0.1308
25	-0.1321
26	-0.1296
27	-0.1313
28	-0.1243
29	-0.1219
30	-0.1115
31	-0.1232
32	-0.1394
33	-0.1531
34	-0.1637
35	-0.1862
36	-0.1986
37	-0.1989
38	-0.1842
39	-0.1887
40	-0.1801
41	-0.1856
42	-0.1775
43	-0.1715
44	-0.1735
45	-0.1763
46	-0.1583
47	-0.1574
48	-0.1662
49	-0.1617
50	-0.1480
51	-0.1449
52	-0.1454
53	-0.1490
54	-0.1414
55	-0.1439
56	-0.1482
57	-0.1503
58	-0.1510
59	-0.1676

Tensor: ffn_down_shexp
Layer	LIM Score
-----	---------
3	-0.0069
4	-0.0084
5	-0.0035
6	0.0161
7	-0.0323
8	0.0076
9	-0.0282
10	0.0427
11	0.0319
12	-0.0441
13	-0.0088
14	0.0075
15	0.0354
16	0.0322
17	0.0148
18	0.0170
19	0.0018
20	0.0105
21	-0.0051
22	0.0146
23	0.0331
24	-0.0011
25	0.0010
26	0.0267
27	-0.0100
28	0.0151
29	0.0055
30	-0.0155
31	-0.0191
32	-0.0075
33	-0.0136
34	-0.0237
35	-0.0251
36	-0.0276
37	0.0159
38	-0.0328
39	-0.0050
40	0.0141
41	-0.0140
42	-0.0111
43	0.0180
44	-0.0102
45	-0.0356
46	0.0016
47	0.0206
48	-0.0075
49	-0.0405
50	0.0422
51	-0.0146
52	-0.0320
53	0.0046
54	0.0311
55	0.0032
56	-0.0039
57	-0.0203
58	-0.0136
59	-0.0119

Tensor: ffn_up_shexp
Layer	LIM Score
-----	---------
3	-0.9355
4	-0.9365
5	-0.9068
6	-0.9485
7	-0.9117
8	-0.8524
9	-0.9458
10	-0.9404
11	-0.9593
12	-0.9458
13	-0.9364
14	-0.9494
15	-0.8997
16	-0.9017
17	-0.8748
18	-0.8369
19	-0.9108
20	-0.8583
21	-0.8067
22	-0.8093
23	-0.8568
24	-0.8719
25	-0.8983
26	-0.9103
27	-0.8789
28	-0.9135
29	-0.9107
30	-0.8975
31	-0.9346
32	-0.9335
33	-0.9334
34	-0.9343
35	-0.9524
36	-0.9404
37	-0.9573
38	-0.9487
39	-0.8949
40	-0.9070
41	-0.9669
42	-0.9815
43	-0.9481
44	-0.9233
45	-0.9606
46	-0.9472
47	-0.9145
48	-0.9580
49	-0.9672
50	-0.9689
51	-0.9570
52	-0.9670
53	-0.9735
54	-0.9553
55	-0.9542
56	-0.9671
57	-0.9526
58	-0.9285
59	-0.9185

Tensor: attn_kv_a_mqa
Layer	LIM Score
-----	---------
0	-0.4179
1	-0.8773
2	-0.9436
3	-0.9022
4	-0.9166
5	-0.9418
6	-0.9812
7	-0.9599
8	-0.9085
9	-0.9724
10	-0.9882
11	-0.9868
12	-0.9906
13	-0.9816
14	-0.9827
15	-0.9766
16	-0.9590
17	-0.9474
18	-0.9573
19	-0.9601
20	-0.9553
21	-0.9345
22	-0.9042
23	-0.9299
24	-0.9555
25	-0.9554
26	-0.9598
27	-0.9575
28	-0.9610
29	-0.9634
30	-0.9601
31	-0.9572
32	-0.9674
33	-0.9619
34	-0.9707
35	-0.9493
36	-0.9801
37	-0.9702
38	-0.9737
39	-0.9567
40	-0.9366
41	-0.9667
42	-0.9751
43	-0.9566
44	-0.9488
45	-0.9364
46	-0.9516
47	-0.9355
48	-0.9723
49	-0.9630
50	-0.9702
51	-0.9591
52	-0.9670
53	-0.8937
54	-0.9420
55	-0.9566
56	-0.9543
57	-0.8239
58	-0.8915
59	-0.9073

Tensor: ffn_gate_inp
Layer	LIM Score
-----	---------
3	-0.9355
4	-0.9365
5	-0.9068
6	-0.9485
7	-0.9117
8	-0.8524
9	-0.9458
10	-0.9404
11	-0.9593
12	-0.9458
13	-0.9364
14	-0.9494
15	-0.8997
16	-0.9017
17	-0.8748
18	-0.8369
19	-0.9108
20	-0.8583
21	-0.8067
22	-0.8093
23	-0.8568
24	-0.8719
25	-0.8983
26	-0.9103
27	-0.8789
28	-0.9135
29	-0.9107
30	-0.8975
31	-0.9346
32	-0.9335
33	-0.9334
34	-0.9343
35	-0.9524
36	-0.9404
37	-0.9573
38	-0.9487
39	-0.8949
40	-0.9070
41	-0.9669
42	-0.9815
43	-0.9481
44	-0.9233
45	-0.9606
46	-0.9472
47	-0.9145
48	-0.9580
49	-0.9672
50	-0.9689
51	-0.9570
52	-0.9670
53	-0.9735
54	-0.9553
55	-0.9542
56	-0.9671
57	-0.9526
58	-0.9285
59	-0.9185

Tensor: ffn_gate_exps
Layer	LIM Score
-----	---------
3	-0.5456
4	-0.4082
5	-0.2537
6	-0.1726
7	-0.1470
8	-0.1202
9	-0.1336
10	-0.1300
11	-0.1028
12	-0.0907
13	-0.0846
14	-0.1017
15	-0.1079
16	-0.1087
17	-0.1140
18	-0.1238
19	-0.1185
20	-0.1048
21	-0.1017
22	-0.1183
23	-0.1191
24	-0.1308
25	-0.1321
26	-0.1296
27	-0.1313
28	-0.1243
29	-0.1219
30	-0.1115
31	-0.1232
32	-0.1394
33	-0.1531
34	-0.1637
35	-0.1862
36	-0.1986
37	-0.1989
38	-0.1842
39	-0.1887
40	-0.1801
41	-0.1856
42	-0.1775
43	-0.1715
44	-0.1735
45	-0.1763
46	-0.1583
47	-0.1574
48	-0.1662
49	-0.1617
50	-0.1480
51	-0.1449
52	-0.1454
53	-0.1490
54	-0.1414
55	-0.1439
56	-0.1482
57	-0.1503
58	-0.1510
59	-0.1676

Tensor: ffn_down_exps
Layer	LIM Score
-----	---------
3	-0.0001
4	0.0004
5	-0.0014
6	0.0006
7	-0.0001
8	-0.0015
9	0.0008
10	0.0013
11	0.0021
12	-0.0015
13	0.0004
14	0.0010
15	0.0022
16	-0.0002
17	-0.0001
18	-0.0021
19	0.0021
20	-0.0013
21	0.0003
22	0.0013
23	-0.0014
24	0.0006
25	0.0001
26	-0.0002
27	-0.0016
28	0.0003
29	0.0004
30	-0.0011
31	-0.0014
32	0.0021
33	-0.0017
34	-0.0005
35	-0.0011
36	-0.0006
37	-0.0007
38	0.0010
39	-0.0037
40	0.0004
41	0.0012
42	-0.0012
43	0.0018
44	-0.0005
45	0.0028
46	0.0009
47	-0.0015
48	0.0000
49	0.0013
50	-0.0012
51	0.0011
52	0.0016
53	0.0005
54	0.0007
55	-0.0021
56	0.0001
57	0.0021
58	-0.0003
59	0.0001
```

</details>

<details>

<summary>Raw LIM Scores for all tensors and layers of `DeepSeek-V3-0324` `q8_0` GGUF</summary>

![DeepSeek-V3-0324-Q8_0-LIM-Scores](https://github.com/user-attachments/assets/e2f71cd3-db25-419d-84d1-2d54be31a590)

</details>

<details>

<summary>Normalized LIM Scores for all tensors and layers of `DeepSeek-V3-0324` `q8_0` GGUF</summary>

![DeepSeek-V3-0324-Q8_0-LIM-Scores_Normalized](https://github.com/user-attachments/assets/72881614-9b83-4f53-983e-31f27d8e0604)

</details>

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-04-13** at **06:30:24**:<br>

Do I understand the results in the quoted PR correctly? The `ffn_down` tensors are the least important? This would be really funny, because everybody knows that quantization errors in `ffn_down` have the highest impact on observed quantization quality. 

I didn't go to read the blog post, but why would cosine similarity between the inputs of two subsequent layers measure layer importance?

---

👤 **ikawrakow** submitted a review the **2025-04-13** at **07:05:04**: 💬 `COMMENTED`

---

👤 **ubergarm** commented the **2025-04-13** at **15:58:29**:<br>

> Do I understand the results in the quoted PR correctly? The `ffn_down` tensors are the least important? This would be really funny, because everybody knows that quantization errors in `ffn_down` have the highest impact on observed quantization quality.

Correct, the summary of the rest of that PR thread including the specific comment by @compilade point out issues with that initial experiment and suggest it may be possible to implement the cosine similarity estimate of relative layer importance in `llama-imatrix`.

> llama-imatrix technically has access to both the input and output activations of a layer, but only uses its input.

---

> I didn't go to read the blog post, but why would cosine similarity between the inputs of two subsequent layers measure layer importance?

The paper that suggests using cosine similarity says:

> The intuition behind LIM is that the more a layer changes its received input embeddings, the more important it must be. https://arxiv.org/pdf/2406.17415

I'll hack around some more to see if I can fix the implementation to possibly do a "running cosine similarity" given the naive first attempt is not properly doing a statistical evaluation across all the input tokens.

The paper suggests another possible method of measuring relative layer sensitivity that I didn't try. Maybe one could calculate the "condition numbers" or "max stretch" for each layer's tensor and rank them, just wildly spit-balling beyond my pay grade xD...

Really appreciate your time, thanks!

---

👤 **ikawrakow** commented the **2025-04-13** at **16:29:52**:<br>

> The paper that suggests using cosine similarity says:
>
>>The intuition behind LIM is that the more a layer changes its received input embeddings, the more important it must be. >>https://arxiv.org/pdf/2406.17415

Sure. But the activations did not change due to that tensor only, they changed due to all tensors in the preceding layer. Or more precisely, activations changed due to the tensor we are considering, plus all tensors with their linear and non-linear operations that followed, before arriving at the same tensor type in the next layer. If the changes in the activations were trivially predictable, people wouldn't be doing complicated networks, and wouldn't be experimenting around with GELU's, RELU's, SILU's, variations of RoPE, different combinations of activation normalizations, and all that jazz. I can see looking at the activation change between **whole layers** to derive an estimate of how important the **entire layer** was, but claiming that the difference in activation input to a specific tensor type between two consecutive layers is a measure of how important this **specific tensor type** is? That's pushing it.

---

👤 **compilade** commented the **2025-04-13** at **17:58:43**:<br>

I agree with @ikawrakow, comparing across layers for a particular tensor seems like it would have non-intuitive results which might not necessarily be linked to relative importance of the tensors.

I think what is calculated here is the cosine similarity across the *inputs* of between consecutive layers of each linear operations in the model(s). It's not particularly clear how this information can be used.

> llama-imatrix technically has access to both the input and output activations of a layer, but only uses its input.

@ubergarm What I meant by this was to calculate LIM scores with the input and output ***within*** each linear operations (i.e. what `llama-imatrix` already considers). The output would be from `t->data` while the input would still be from `src1->data`.
Each layer should be independent in this approach. I don't know what they used (in the paper) to combine the results across multiple tokens, though. Likely the average, but I'm not sure.

---

👤 **ikawrakow** commented the **2025-04-14** at **07:26:42**:<br>

@compilade 

Can you be more specific how you want to calculate the impact of a linear operation from the input activations and the result of the linear operation?

I have used this to derive corrections for a quantized model (have not published, it is in a private repository where I experiment with stuff). But I don't really see how one can derive tensor importance scores from that.

---

👤 **compilade** commented the **2025-04-15** at **22:13:03**:<br>

> Can you be more specific how you want to calculate the impact of a linear operation from the input activations and the result of the linear operation?

@ikawrakow I might not have thought this through properly.

I was thinking of directly calculating a dot product between the input and output of each matmul (and normalizing) to get LIM scores by negating that, but this would only work for square matrices (where the input and output have the same shape).

---

👤 **ubergarm** commented the **2025-04-16** at **15:06:47**:<br>

Closing this in favor of implementation in PR#328.

## Experiment

Still more experimentation to do, and sorry no visual graphs as I'm away from my desk, but did a quick A/B test comparing two `V3-0324` quants which have the same final size but vary only in which routed expert layers receive more or less quantization. For this discussion I'll refer to the baseline case of giving the first 17 routed expert layers more bpw as `FIRST-N` approach vs using the results of layer importance from PR#328 `COSSIM` to decide which 17 routed expert layers should receive more bpw.

Finally, I provide the `--show-statistics` of the computed imatrix used for these quantizations from [@EAddario's mainline llama.cpp PR#12718](https://github.com/ggml-org/llama.cpp/pull/12718) if anyone wants to compare the numbers themselves. (I haven't had a chance to compare myself yet).

## tl;dr;
Using PR#328 `llama-imatrix --layer-similarity [-lsim]` to decide which layers to prioritize quantization showed slightly better perplexity score than naively using the first 17 layers in a single experiment on `V3-0324`.

* `FIRST-N` Final estimate: PPL = 3.3193 +/- 0.01830
* `COSSIM` Final estimate: PPL = 3.3151 +/- 0.0182

While it is within the noise, there may be room for further improvement applying the scores to attention layer quantization as well which I didn't do for this experiment.

## Procedure

<details>

<summary>Compute imatrix and layer similarity scores using `V3-0324` `q8_0`</summary>

```bash
$ numactl -N 1 -m 1 \
./build/bin/llama-imatrix \
    --verbosity 1 \
    --layer-similarity \
    -m /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-Q8_0.gguf \
    -f calibration_data_v5_rc.txt \
    -o /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-$(git rev-parse --short HEAD).dat \
    --ctx-size 512 \
    --numa numactl \
    --threads 128

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
compute_imatrix: tokenization took 309.837 ms
compute_imatrix: computing over 213 chunks with batch_size 512
compute_imatrix: 37.90 seconds per pass - ETA 2 hours 14.55 minutes
[1]60.9619,[2]10.7701,[3]5.8724,[4]3.7883,[5]2.9691,[6]2.5089,[7]2.2199,[8]2.0199,[9]1.9095,
save_imatrix: entry '             blk.60.ffn_down_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.60.ffn_gate_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.60.ffn_up_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.25.ffn_down_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.26.ffn_down_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.25.ffn_up_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.25.ffn_gate_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.26.ffn_gate_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.26.ffn_up_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**

save_imatrix: stored collected data after 10 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-f7c5a94e.dat
[10]1.8219,[11]2.0296,[12]2.0839,[13]2.0978,[14]2.1403,[15]2.0365,[16]1.9492,[17]1.8786,[18]1.8160,[19]1.7743,
save_imatrix: stored collected data after 20 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-f7c5a94e.dat
[20]1.7315,[21]1.6986,[22]1.6609,[23]1.6319,[24]1.6201,[25]1.6080,[26]1.5822,[27]1.6812,[28]1.7547,[29]1.8204,
save_imatrix: stored collected data after 30 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-f7c5a94e.dat
[30]1.8188,[31]1.8323,[32]1.8317,[33]1.8091,[34]1.8457,[35]1.8217,[36]1.8215,[37]1.8106,[38]1.8208,[39]1.8070,
save_imatrix: stored collected data after 40 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-f7c5a94e.dat
[40]1.7838,[41]1.7606,[42]1.7410,[43]1.7291,[44]1.7157,[45]1.7023,[46]1.6981,[47]1.6919,[48]1.6811,[49]1.6707,
save_imatrix: stored collected data after 50 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-f7c5a94e.dat
[50]1.6650,[51]1.6623,[52]1.6625,[53]1.6672,[54]1.6812,[55]1.6781,[56]1.6683,[57]1.6764,[58]1.6796,[59]1.6906,
save_imatrix: stored collected data after 60 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-f7c5a94e.dat
[60]1.6855,[61]1.7243,[62]1.7565,[63]1.7884,[64]1.8197,[65]1.8677,[66]1.8802,[67]1.9148,[68]1.9442,[69]1.9996,
save_imatrix: stored collected data after 70 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-f7c5a94e.dat
[70]2.0525,[71]2.0832,[72]2.1136,[73]2.1258,[74]2.1407,[75]2.1702,[76]2.2011,[77]2.2185,[78]2.2164,[79]2.2313,
save_imatrix: stored collected data after 80 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-f7c5a94e.dat
[80]2.2543,[81]2.2904,[82]2.3238,[83]2.3342,[84]2.3650,[85]2.3733,[86]2.3730,[87]2.4024,[88]2.4344,[89]2.4899,
save_imatrix: stored collected data after 90 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-f7c5a94e.dat
[90]2.5102,[91]2.5125,[92]2.5192,[93]2.5349,[94]2.5452,[95]2.5779,[96]2.5670,[97]2.6058,[98]2.6319,[99]2.6214,
save_imatrix: stored collected data after 100 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-f7c5a94e.dat
[100]2.6537,[101]2.7008,[102]2.7326,[103]2.7740,[104]2.8020,[105]2.8310,[106]2.8682,[107]2.8605,[108]2.8789,[109]2.8849,
save_imatrix: stored collected data after 110 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-f7c5a94e.dat
[110]2.8910,[111]2.8878,[112]2.9177,[113]2.9435,[114]2.9520,[115]2.9363,[116]2.9104,[117]2.9044,[118]2.9147,[119]2.9003,
save_imatrix: stored collected data after 120 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-f7c5a94e.dat
[120]2.8773,[121]2.8737,[122]2.8738,[123]2.8819,[124]2.8872,[125]2.8942,[126]2.9018,[127]2.9043,[128]2.9343,[129]2.9484,
save_imatrix: stored collected data after 130 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-f7c5a94e.dat
[130]2.9241,[131]2.9003,[132]2.8771,[133]2.8544,[134]2.8563,[135]2.8567,[136]2.8828,[137]2.9150,[138]2.9340,[139]2.9389,
save_imatrix: stored collected data after 140 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-f7c5a94e.dat
[140]2.9637,[141]2.9866,[142]3.0151,[143]3.0354,[144]3.0569,[145]3.0766,[146]3.0972,[147]3.1154,[148]3.1266,[149]3.1351,
save_imatrix: stored collected data after 150 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-f7c5a94e.dat
[150]3.1395,[151]3.1572,[152]3.1761,[153]3.1759,[154]3.1834,[155]3.1945,[156]3.2035,[157]3.2148,[158]3.2209,[159]3.2300,
save_imatrix: stored collected data after 160 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-f7c5a94e.dat
[160]3.2442,[161]3.2498,[162]3.2525,[163]3.2595,[164]3.2704,[165]3.2724,[166]3.2737,[167]3.2912,[168]3.3010,[169]3.3082,
save_imatrix: stored collected data after 170 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-f7c5a94e.dat
[170]3.3258,[171]3.3403,[172]3.3354,[173]3.3417,[174]3.3424,[175]3.3575,[176]3.3691,[177]3.3818,[178]3.3768,[179]3.3734,
save_imatrix: stored collected data after 180 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-f7c5a94e.dat
[180]3.3682,[181]3.3635,[182]3.3578,[183]3.3531,[184]3.3472,[185]3.3600,[186]3.3887,[187]3.4121,[188]3.4336,[189]3.4550,
save_imatrix: stored collected data after 190 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-f7c5a94e.dat
[190]3.4850,[191]3.4990,[192]3.5134,[193]3.5036,[194]3.5210,[195]3.5145,[196]3.4953,[197]3.4747,[198]3.4946,[199]3.5110,
save_imatrix: stored collected data after 200 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-f7c5a94e.dat
[200]3.5207,[201]3.5290,[202]3.5447,[203]3.5621,[204]3.5748,[205]3.5874,[206]3.6021,[207]3.5989,[208]3.5771,[209]3.5556,
save_imatrix: stored collected data after 210 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-f7c5a94e.dat
[210]3.5342,[211]3.5134,[212]3.4930,[213]3.4727,
save_imatrix: stored collected data after 213 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-f7c5a94e.dat

Final estimate: PPL = 3.4727 +/- 0.03300

llama_print_timings:        load time =   38826.79 ms
llama_print_timings:      sample time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings: prompt eval time = 7699212.14 ms / 109056 tokens (   70.60 ms per token,    14.16 tokens per second)
llama_print_timings:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings:       total time = 7777812.63 ms / 109057 tokens


======================== sorted layer importances
  0: Layer   0, <cos_sim> = 0.517453
  1: Layer  60, <cos_sim> = 0.59436
  2: Layer   8, <cos_sim> = 0.857555
  3: Layer   3, <cos_sim> = 0.858137
  4: Layer   1, <cos_sim> = 0.869657
  5: Layer  59, <cos_sim> = 0.875667
  6: Layer  57, <cos_sim> = 0.888417
  7: Layer   5, <cos_sim> = 0.906457
  8: Layer  58, <cos_sim> = 0.911674
  9: Layer   7, <cos_sim> = 0.921961
 10: Layer  53, <cos_sim> = 0.926514
 11: Layer  22, <cos_sim> = 0.932632
 12: Layer  17, <cos_sim> = 0.936935
 13: Layer  24, <cos_sim> = 0.93742
 14: Layer  23, <cos_sim> = 0.939419
 15: Layer   4, <cos_sim> = 0.941044
 16: Layer  15, <cos_sim> = 0.945621
 17: Layer  25, <cos_sim> = 0.94563
 18: Layer   6, <cos_sim> = 0.946055
# NOTE: i prioritized the above 17 routed expert layers [3-60] for more bpw quantization (first 0-2 layers are dense)
 19: Layer  21, <cos_sim> = 0.946446
 20: Layer  16, <cos_sim> = 0.947423
 21: Layer  27, <cos_sim> = 0.947699
 22: Layer  18, <cos_sim> = 0.948201
 23: Layer  10, <cos_sim> = 0.949096
 24: Layer  54, <cos_sim> = 0.949141
 25: Layer   2, <cos_sim> = 0.949452
 26: Layer  20, <cos_sim> = 0.949668
 27: Layer  30, <cos_sim> = 0.949811
 28: Layer  26, <cos_sim> = 0.951796
 29: Layer  13, <cos_sim> = 0.951903
 30: Layer  14, <cos_sim> = 0.952166
 31: Layer   9, <cos_sim> = 0.952194
 32: Layer  44, <cos_sim> = 0.952973
 33: Layer  35, <cos_sim> = 0.953037
 34: Layer  45, <cos_sim> = 0.953128
 35: Layer  29, <cos_sim> = 0.954667
 36: Layer  28, <cos_sim> = 0.954742
 37: Layer  31, <cos_sim> = 0.954809
 38: Layer  56, <cos_sim> = 0.955925
 39: Layer  43, <cos_sim> = 0.956722
 40: Layer  50, <cos_sim> = 0.958269
 41: Layer  19, <cos_sim> = 0.959386
 42: Layer  33, <cos_sim> = 0.95975
 43: Layer  32, <cos_sim> = 0.960649
 44: Layer  55, <cos_sim> = 0.960837
 45: Layer  11, <cos_sim> = 0.961299
 46: Layer  34, <cos_sim> = 0.961852
 47: Layer  12, <cos_sim> = 0.962011
 48: Layer  46, <cos_sim> = 0.962943
 49: Layer  49, <cos_sim> = 0.965045
 50: Layer  39, <cos_sim> = 0.96526
 51: Layer  40, <cos_sim> = 0.96575
 52: Layer  37, <cos_sim> = 0.967049
 53: Layer  36, <cos_sim> = 0.96716
 54: Layer  52, <cos_sim> = 0.967574
 55: Layer  38, <cos_sim> = 0.968262
 56: Layer  41, <cos_sim> = 0.968457
 57: Layer  48, <cos_sim> = 0.968755
 58: Layer  51, <cos_sim> = 0.968768
 59: Layer  47, <cos_sim> = 0.968788
 60: Layer  42, <cos_sim> = 0.971662

======================== sorted attention importances
  0: Layer   0, <cos_sim> = 0.13174
  1: Layer   8, <cos_sim> = 0.516951
  2: Layer  11, <cos_sim> = 0.61188
  3: Layer  10, <cos_sim> = 0.612091
  4: Layer  12, <cos_sim> = 0.612348
  5: Layer  18, <cos_sim> = 0.616718
  6: Layer  16, <cos_sim> = 0.61912
  7: Layer   9, <cos_sim> = 0.655522
  8: Layer  13, <cos_sim> = 0.665296
  9: Layer  22, <cos_sim> = 0.672061
 10: Layer   6, <cos_sim> = 0.699289
 11: Layer  19, <cos_sim> = 0.700966
 12: Layer  20, <cos_sim> = 0.704575
 13: Layer   7, <cos_sim> = 0.71001
 14: Layer  14, <cos_sim> = 0.725971
 15: Layer  23, <cos_sim> = 0.740926
 16: Layer  25, <cos_sim> = 0.747222
 17: Layer  17, <cos_sim> = 0.749419
 18: Layer  15, <cos_sim> = 0.754558
 19: Layer  21, <cos_sim> = 0.761675
 20: Layer  24, <cos_sim> = 0.761882
 21: Layer   5, <cos_sim> = 0.766086
 22: Layer   2, <cos_sim> = 0.767046
 23: Layer  30, <cos_sim> = 0.772412
 24: Layer   1, <cos_sim> = 0.772533
 25: Layer  44, <cos_sim> = 0.777696
 26: Layer  29, <cos_sim> = 0.779458
 27: Layer  28, <cos_sim> = 0.779721
 28: Layer  37, <cos_sim> = 0.780809
 29: Layer  26, <cos_sim> = 0.781589
 30: Layer   4, <cos_sim> = 0.786884
 31: Layer  34, <cos_sim> = 0.787128
 32: Layer  36, <cos_sim> = 0.78846
 33: Layer  27, <cos_sim> = 0.791454
 34: Layer  31, <cos_sim> = 0.805225
 35: Layer  33, <cos_sim> = 0.806554
 36: Layer  57, <cos_sim> = 0.809911
 37: Layer  32, <cos_sim> = 0.811714
 38: Layer  38, <cos_sim> = 0.81192
 39: Layer  35, <cos_sim> = 0.816966
 40: Layer  41, <cos_sim> = 0.820029
 41: Layer  40, <cos_sim> = 0.833644
 42: Layer   3, <cos_sim> = 0.83367
 43: Layer  39, <cos_sim> = 0.835849
 44: Layer  42, <cos_sim> = 0.841079
 45: Layer  60, <cos_sim> = 0.853526
 46: Layer  45, <cos_sim> = 0.857364
 47: Layer  56, <cos_sim> = 0.859897
 48: Layer  59, <cos_sim> = 0.861441
 49: Layer  53, <cos_sim> = 0.864087
 50: Layer  46, <cos_sim> = 0.864727
 51: Layer  43, <cos_sim> = 0.864848
 52: Layer  51, <cos_sim> = 0.872346
 53: Layer  48, <cos_sim> = 0.87434
 54: Layer  52, <cos_sim> = 0.874649
 55: Layer  47, <cos_sim> = 0.878183
 56: Layer  58, <cos_sim> = 0.879985
 57: Layer  49, <cos_sim> = 0.880846
 58: Layer  55, <cos_sim> = 0.885206
 59: Layer  50, <cos_sim> = 0.897436
 60: Layer  54, <cos_sim> = 0.921917

======================== sorted ffn importances
  0: Layer   7, <cos_sim> = 0.571293
  1: Layer  10, <cos_sim> = 0.590428
  2: Layer  11, <cos_sim> = 0.591834
  3: Layer  17, <cos_sim> = 0.608386
  4: Layer  15, <cos_sim> = 0.620593
  5: Layer   0, <cos_sim> = 0.632572
  6: Layer   9, <cos_sim> = 0.643826
  7: Layer  12, <cos_sim> = 0.64739
  8: Layer   8, <cos_sim> = 0.649753
  9: Layer  21, <cos_sim> = 0.67168
 10: Layer  18, <cos_sim> = 0.679443
 11: Layer  19, <cos_sim> = 0.701283
 12: Layer  60, <cos_sim> = 0.701407
 13: Layer  13, <cos_sim> = 0.712941
 14: Layer  16, <cos_sim> = 0.722858
 15: Layer  24, <cos_sim> = 0.725591
 16: Layer  14, <cos_sim> = 0.727539
 17: Layer  22, <cos_sim> = 0.728219
 18: Layer  20, <cos_sim> = 0.736531
 19: Layer   6, <cos_sim> = 0.744335
 20: Layer  23, <cos_sim> = 0.749712
 21: Layer  29, <cos_sim> = 0.757133
 22: Layer  25, <cos_sim> = 0.758496
 23: Layer   5, <cos_sim> = 0.759015
 24: Layer  27, <cos_sim> = 0.759242
 25: Layer  28, <cos_sim> = 0.76237
 26: Layer  43, <cos_sim> = 0.764705
 27: Layer  36, <cos_sim> = 0.766839
 28: Layer  35, <cos_sim> = 0.773264
 29: Layer  26, <cos_sim> = 0.775702
 30: Layer  33, <cos_sim> = 0.778872
 31: Layer  32, <cos_sim> = 0.790364
 32: Layer   3, <cos_sim> = 0.790503
 33: Layer  30, <cos_sim> = 0.792984
 34: Layer  31, <cos_sim> = 0.79496
 35: Layer  37, <cos_sim> = 0.795521
 36: Layer  34, <cos_sim> = 0.796573
 37: Layer  56, <cos_sim> = 0.804781
 38: Layer  40, <cos_sim> = 0.806738
 39: Layer  59, <cos_sim> = 0.808235
 40: Layer   4, <cos_sim> = 0.809825
 41: Layer   1, <cos_sim> = 0.819665
 42: Layer  38, <cos_sim> = 0.820409
 43: Layer  39, <cos_sim> = 0.820894
 44: Layer  41, <cos_sim> = 0.824874
 45: Layer  44, <cos_sim> = 0.846473
 46: Layer  52, <cos_sim> = 0.849335
 47: Layer  42, <cos_sim> = 0.850524
 48: Layer  45, <cos_sim> = 0.851349
 49: Layer  55, <cos_sim> = 0.852943
 50: Layer  47, <cos_sim> = 0.85862
 51: Layer  50, <cos_sim> = 0.858953
 52: Layer  51, <cos_sim> = 0.861418
 53: Layer  58, <cos_sim> = 0.861473
 54: Layer   2, <cos_sim> = 0.862156
 55: Layer  57, <cos_sim> = 0.86361
 56: Layer  46, <cos_sim> = 0.864787
 57: Layer  48, <cos_sim> = 0.867249
 58: Layer  54, <cos_sim> = 0.876651
 59: Layer  49, <cos_sim> = 0.883354
 60: Layer  53, <cos_sim> = 0.90793
```

</details>

## `FIRST-N-IQ3_K_R4`
```
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type iq6_k:    1 tensors
llama_model_loader: - type q6_0_r4:   61 tensors
llama_model_loader: - type iq3_k_r4:   82 tensors
llama_model_loader: - type iq4_k_r4:   75 tensors
llama_model_loader: - type iq5_k_r4:  567 tensors
```

```
# Routed Experts (3-60) (CPU)
# Prioritize first 17 layers with larger quants
blk\.[3-9]\.ffn_down_exps\.weight=iq5_k_r4
blk\.[1][0-9]\.ffn_down_exps\.weight=iq5_k_r4
blk\.[2-5][0-9]\.ffn_down_exps\.weight=iq4_k_r4
blk\.60\.ffn_down_exps\.weight=iq4_k_r4

blk\.[3-9]\.ffn_(gate|up)_exps\.weight=iq4_k_r4
blk\.[1][0-9]\.ffn_(gate|up)_exps\.weight=iq4_k_r4
blk\.[2-5][0-9]\.ffn_(gate|up)_exps\.weight=iq3_k_r4
blk\.60\.ffn_(gate|up)_exps\.weight=iq3_k_r4
```

## `COSSIM-IQ3_K_R4`
```
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type iq6_k:    1 tensors
llama_model_loader: - type q6_0_r4:   61 tensors
llama_model_loader: - type iq3_k_r4:   82 tensors
llama_model_loader: - type iq4_k_r4:   75 tensors
llama_model_loader: - type iq5_k_r4:  567 tensors
```

```
# Routed Experts (3-60) (CPU)
# Prioritize quantizing 17 layers given by lowest cos similarity scores with larger bpw quant size
blk\.3\.ffn_down_exps\.weight=iq5_k_r4
blk\.4\.ffn_down_exps\.weight=iq5_k_r4
blk\.5\.ffn_down_exps\.weight=iq5_k_r4
blk\.6\.ffn_down_exps\.weight=iq5_k_r4
blk\.7\.ffn_down_exps\.weight=iq5_k_r4
blk\.8\.ffn_down_exps\.weight=iq5_k_r4
blk\.15\.ffn_down_exps\.weight=iq5_k_r4
blk\.17\.ffn_down_exps\.weight=iq5_k_r4
blk\.22\.ffn_down_exps\.weight=iq5_k_r4
blk\.23\.ffn_down_exps\.weight=iq5_k_r4
blk\.24\.ffn_down_exps\.weight=iq5_k_r4
blk\.25\.ffn_down_exps\.weight=iq5_k_r4
blk\.53\.ffn_down_exps\.weight=iq5_k_r4
blk\.57\.ffn_down_exps\.weight=iq5_k_r4
blk\.58\.ffn_down_exps\.weight=iq5_k_r4
blk\.59\.ffn_down_exps\.weight=iq5_k_r4
blk\.60\.ffn_down_exps\.weight=iq5_k_r4
## remainder
blk\.[3-9]\.ffn_down_exps\.weight=iq4_k_r4
blk\.[1-5][0-9]\.ffn_down_exps\.weight=iq4_k_r4

# Same for gate/up
blk\.3\.ffn_(gate|up)_exps\.weight=iq4_k_r4
blk\.4\.ffn_(gate|up)_exps\.weight=iq4_k_r4
blk\.5\.ffn_(gate|up)_exps\.weight=iq4_k_r4
blk\.6\.ffn_(gate|up)_exps\.weight=iq4_k_r4
blk\.7\.ffn_(gate|up)_exps\.weight=iq4_k_r4
blk\.8\.ffn_(gate|up)_exps\.weight=iq4_k_r4
blk\.15\.ffn_(gate|up)_exps\.weight=iq4_k_r4
blk\.17\.ffn_(gate|up)_exps\.weight=iq4_k_r4
blk\.22\.ffn_(gate|up)_exps\.weight=iq4_k_r4
blk\.23\.ffn_(gate|up)_exps\.weight=iq4_k_r4
blk\.24\.ffn_(gate|up)_exps\.weight=iq4_k_r4
blk\.25\.ffn_(gate|up)_exps\.weight=iq4_k_r4
blk\.53\.ffn_(gate|up)_exps\.weight=iq4_k_r4
blk\.57\.ffn_(gate|up)_exps\.weight=iq4_k_r4
blk\.58\.ffn_(gate|up)_exps\.weight=iq4_k_r4
blk\.59\.ffn_(gate|up)_exps\.weight=iq4_k_r4
blk\.60\.ffn_(gate|up)_exps\.weight=iq4_k_r4
## remainder
blk\.[3-9]\.ffn_(gate|up)_exps\.weight=iq3_k_r4
blk\.[1-5][0-9]\.ffn_(gate|up)_exps\.weight=iq3_k_r4
blk\.60\.ffn_(gate|up)_exps\.weight=iq3_k_r4
```

## Comparison with `--show-statistics`

To compare stats I also ran mainline's `--show-statistics` experimental PR against the imatrix.dat file and include it here for reference.

<details>

<summary>show imatrix stats</summary>

```
$ git rev-parse --short HEAD
52e86e2c

$ ./build/bin/llama-imatrix --version
version: 5149 (52e86e2c)
built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu

$ ./build/bin/llama-imatrix \
    --in-file /mnt/raid/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324.imatrix \
    --show-statistics

ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA RTX A6000, compute capability 8.6, VMM: yes

Computing statistics for /mnt/raid/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324.imatrix (720 tensors)

Layer	Tensor              	  Σ(Bias)	    Min	         Max	       μ	        σ	 % Active	     N	     Entropy	E (norm)	  ZD Score
==========================================================================================================================================================================
   59	attn_kv_a_mqa       	     90.49	 0.0030	     12.4869	  0.0126	   0.1507	  100.00%	  7168	     11.0850	  86.55%	     0.18%
   56	attn_kv_a_mqa       	     80.09	 0.0047	      8.0205	  0.0112	   0.1075	  100.00%	  7168	     10.9840	  85.76%	     0.31%
   53	attn_kv_a_mqa       	     70.07	 0.0044	      7.5596	  0.0098	   0.1005	  100.00%	  7168	     10.8180	  84.47%	     0.32%
   49	attn_kv_a_mqa       	     69.86	 0.0048	      3.3494	  0.0097	   0.0605	  100.00%	  7168	     11.2925	  88.17%	     0.40%
   46	attn_kv_a_mqa       	     66.83	 0.0042	      5.2714	  0.0093	   0.0802	  100.00%	  7168	     11.0102	  85.97%	     0.29%
    8	attn_kv_a_mqa       	     65.87	 0.0003	     30.7816	  0.0092	   0.3722	  100.00%	  7168	      5.6626	  44.21%	     0.18%
   45	attn_kv_a_mqa       	     65.12	 0.0041	      2.7374	  0.0091	   0.0630	  100.00%	  7168	     11.0425	  86.22%	     0.39%
   55	attn_kv_a_mqa       	     64.58	 0.0045	      4.1384	  0.0090	   0.0651	  100.00%	  7168	     11.3060	  88.28%	     0.38%
   52	attn_kv_a_mqa       	     63.81	 0.0040	      4.6357	  0.0089	   0.0695	  100.00%	  7168	     11.1023	  86.69%	     0.39%
   42	attn_kv_a_mqa       	     62.13	 0.0041	      3.5418	  0.0087	   0.0734	  100.00%	  7168	     10.6817	  83.40%	     0.35%
   40	attn_kv_a_mqa       	     60.16	 0.0037	      4.7100	  0.0084	   0.0803	  100.00%	  7168	     10.5976	  82.75%	     0.32%
   50	attn_kv_a_mqa       	     58.66	 0.0041	      3.0096	  0.0082	   0.0495	  100.00%	  7168	     11.5214	  89.96%	     0.46%
   43	attn_kv_a_mqa       	     57.84	 0.0041	      2.7142	  0.0081	   0.0581	  100.00%	  7168	     11.0605	  86.36%	     0.35%
   54	attn_kv_a_mqa       	     53.82	 0.0039	      2.6784	  0.0075	   0.0405	  100.00%	  7168	     11.8078	  92.20%	     0.29%
   36	attn_kv_a_mqa       	     53.42	 0.0030	      6.0237	  0.0075	   0.0951	  100.00%	  7168	      9.9056	  77.34%	     0.31%
   39	attn_kv_a_mqa       	     53.06	 0.0033	      2.9174	  0.0074	   0.0626	  100.00%	  7168	     10.6570	  83.21%	     0.39%
    6	attn_kv_a_mqa       	     52.69	 0.0002	     22.8025	  0.0074	   0.2735	  100.00%	  7168	      6.4878	  50.66%	     0.20%
    3	attn_kv_a_mqa       	     52.55	 0.0001	     31.5538	  0.0073	   0.3736	  100.00%	  7168	      4.8646	  37.98%	     0.13%
   48	attn_kv_a_mqa       	     52.29	 0.0035	      2.9375	  0.0073	   0.0513	  100.00%	  7168	     11.1767	  87.27%	     0.33%
   47	attn_kv_a_mqa       	     51.19	 0.0033	      3.1746	  0.0071	   0.0493	  100.00%	  7168	     11.2441	  87.79%	     0.47%
   31	attn_kv_a_mqa       	     47.25	 0.0028	      4.2665	  0.0066	   0.0696	  100.00%	  7168	     10.2530	  80.06%	     0.35%
   30	attn_kv_a_mqa       	     46.10	 0.0024	      3.8427	  0.0064	   0.0764	  100.00%	  7168	      9.8028	  76.54%	     0.36%
   57	attn_kv_a_mqa       	     43.52	 0.0022	      8.5336	  0.0061	   0.1032	  100.00%	  7168	     10.1204	  79.02%	     0.27%
   51	attn_kv_a_mqa       	     43.38	 0.0027	      3.0131	  0.0061	   0.0441	  100.00%	  7168	     11.1298	  86.90%	     0.42%
   44	attn_kv_a_mqa       	     43.34	 0.0020	      5.2019	  0.0060	   0.0773	  100.00%	  7168	      9.7626	  76.23%	     0.35%
    2	attn_kv_a_mqa       	     43.09	 0.0000	     18.1894	  0.0060	   0.2170	   99.99%	  7168	      6.6727	  52.10%	     0.18%
   35	attn_kv_a_mqa       	     43.04	 0.0026	      3.6656	  0.0060	   0.0589	  100.00%	  7168	     10.3826	  81.07%	     0.35%
   58	attn_kv_a_mqa       	     41.57	 0.0019	      1.4918	  0.0058	   0.0283	  100.00%	  7168	     11.7008	  91.36%	     0.54%
   34	attn_kv_a_mqa       	     40.83	 0.0023	      4.2025	  0.0057	   0.0654	  100.00%	  7168	     10.0369	  78.37%	     0.35%
   29	attn_kv_a_mqa       	     40.42	 0.0021	      4.0808	  0.0056	   0.0676	  100.00%	  7168	      9.8758	  77.11%	     0.38%
   37	attn_kv_a_mqa       	     40.14	 0.0019	      4.1508	  0.0056	   0.0705	  100.00%	  7168	      9.8134	  76.62%	     0.32%
   33	attn_kv_a_mqa       	     39.93	 0.0022	      3.4713	  0.0056	   0.0569	  100.00%	  7168	     10.2643	  80.14%	     0.39%
   32	attn_kv_a_mqa       	     39.70	 0.0024	      3.5055	  0.0055	   0.0567	  100.00%	  7168	     10.2928	  80.37%	     0.38%
   38	attn_kv_a_mqa       	     39.46	 0.0021	      3.5038	  0.0055	   0.0595	  100.00%	  7168	     10.2390	  79.95%	     0.33%
   41	attn_kv_a_mqa       	     39.27	 0.0023	      2.6274	  0.0055	   0.0536	  100.00%	  7168	     10.3751	  81.01%	     0.31%
    1	attn_kv_a_mqa       	     38.02	 0.0000	      9.3369	  0.0053	   0.1163	   99.97%	  7168	      7.6337	  59.60%	     0.40%
   27	attn_kv_a_mqa       	     37.55	 0.0021	      2.9428	  0.0052	   0.0576	  100.00%	  7168	     10.1568	  79.30%	     0.36%
    0	attn_kv_a_mqa       	     37.33	 0.0001	      4.3022	  0.0052	   0.0674	  100.00%	  7168	      8.3011	  64.81%	     1.12%
    5	attn_kv_a_mqa       	     36.35	 0.0000	      8.2527	  0.0051	   0.1102	  100.00%	  7168	      8.1113	  63.33%	     0.27%
   12	attn_kv_a_mqa       	     35.13	 0.0005	      9.7724	  0.0049	   0.1234	  100.00%	  7168	      7.7981	  60.89%	     0.36%
   28	attn_kv_a_mqa       	     35.01	 0.0018	      3.0860	  0.0049	   0.0548	  100.00%	  7168	      9.9199	  77.45%	     0.39%
    7	attn_kv_a_mqa       	     33.68	 0.0003	      9.6207	  0.0047	   0.1187	  100.00%	  7168	      8.1082	  63.31%	     0.28%
   60	attn_kv_a_mqa       	     32.02	 0.0000	      5.2868	  0.0045	   0.0634	   99.99%	  7168	     10.8390	  84.63%	     0.15%
   26	attn_kv_a_mqa       	     31.92	 0.0016	      3.4728	  0.0045	   0.0544	  100.00%	  7168	      9.9117	  77.39%	     0.35%
   25	attn_kv_a_mqa       	     30.18	 0.0014	      2.8025	  0.0042	   0.0548	  100.00%	  7168	      9.5139	  74.28%	     0.38%
   22	attn_kv_a_mqa       	     26.66	 0.0008	      3.7990	  0.0037	   0.0641	  100.00%	  7168	      8.3974	  65.57%	     0.35%
   24	attn_kv_a_mqa       	     25.26	 0.0012	      2.7091	  0.0035	   0.0441	  100.00%	  7168	      9.7836	  76.39%	     0.32%
   23	attn_kv_a_mqa       	     23.71	 0.0010	      2.4957	  0.0033	   0.0442	  100.00%	  7168	      9.3907	  73.32%	     0.33%
   13	attn_kv_a_mqa       	     22.19	 0.0004	      4.5967	  0.0031	   0.0604	  100.00%	  7168	      8.6560	  67.59%	     0.36%
   18	attn_kv_a_mqa       	     18.76	 0.0004	      4.7766	  0.0026	   0.0634	  100.00%	  7168	      7.4838	  58.43%	     0.29%
   20	attn_kv_a_mqa       	     18.39	 0.0006	      2.0356	  0.0026	   0.0364	  100.00%	  7168	      9.0449	  70.62%	     0.42%
   21	attn_kv_a_mqa       	     18.15	 0.0008	      1.4004	  0.0025	   0.0308	  100.00%	  7168	      9.5419	  74.50%	     0.38%
    4	attn_kv_a_mqa       	     17.48	 0.0000	      3.9561	  0.0024	   0.0508	  100.00%	  7168	      8.3132	  64.91%	     0.29%
   19	attn_kv_a_mqa       	     16.86	 0.0005	      2.3614	  0.0024	   0.0371	  100.00%	  7168	      8.7611	  68.41%	     0.40%
   14	attn_kv_a_mqa       	     16.72	 0.0005	      2.2532	  0.0023	   0.0319	  100.00%	  7168	      9.6589	  75.42%	     0.40%
   10	attn_kv_a_mqa       	     15.69	 0.0002	      3.4866	  0.0022	   0.0459	  100.00%	  7168	      8.2331	  64.28%	     0.33%
   16	attn_kv_a_mqa       	     14.88	 0.0003	      3.3163	  0.0021	   0.0443	  100.00%	  7168	      7.9409	  62.00%	     0.36%
   11	attn_kv_a_mqa       	     12.25	 0.0002	      2.8678	  0.0017	   0.0367	  100.00%	  7168	      8.1340	  63.51%	     0.40%
    9	attn_kv_a_mqa       	     11.66	 0.0001	      2.1372	  0.0016	   0.0296	  100.00%	  7168	      8.5938	  67.10%	     0.42%
   15	attn_kv_a_mqa       	     11.06	 0.0004	      1.3714	  0.0015	   0.0197	  100.00%	  7168	      9.8387	  76.82%	     0.45%
   17	attn_kv_a_mqa       	      9.08	 0.0002	      1.0626	  0.0013	   0.0159	  100.00%	  7168	      9.6649	  75.46%	     0.54%
   59	attn_kv_b           	   1494.94	 0.3075	     23.5223	  2.9198	   1.5359	  100.00%	   512	      8.8840	  98.71%	     4.69%
   55	attn_kv_b           	   1402.27	 0.0013	     31.8818	  2.7388	   1.4726	  100.00%	   512	      8.9138	  99.04%	     1.76%
   54	attn_kv_b           	   1238.10	 1.0519	     24.4297	  2.4182	   1.3123	  100.00%	   512	      8.9096	  99.00%	     1.56%
   58	attn_kv_b           	   1225.82	 0.0140	     12.6256	  2.3942	   1.0253	  100.00%	   512	      8.8844	  98.72%	     7.42%
   50	attn_kv_b           	    997.21	 0.3756	     27.5049	  1.9477	   1.2311	  100.00%	   512	      8.9022	  98.91%	     0.98%
   56	attn_kv_b           	    992.19	 0.7272	     37.7112	  1.9379	   1.7799	  100.00%	   512	      8.8176	  97.97%	     1.37%
   57	attn_kv_b           	    972.69	 0.0029	     31.7707	  1.8998	   1.5565	  100.00%	   512	      8.8230	  98.03%	     2.34%
   60	attn_kv_b           	    959.44	 0.1139	     10.0245	  1.8739	   0.8823	  100.00%	   512	      8.8704	  98.56%	     6.84%
   47	attn_kv_b           	    914.51	 0.4712	     19.7740	  1.7862	   1.1224	  100.00%	   512	      8.8865	  98.74%	     2.15%
   52	attn_kv_b           	    865.20	 0.0005	     23.7891	  1.6898	   1.1451	  100.00%	   512	      8.8781	  98.65%	     2.15%
   46	attn_kv_b           	    864.89	 1.1356	      7.0131	  1.6892	   0.5083	  100.00%	   512	      8.9572	  99.52%	     2.54%
   43	attn_kv_b           	    718.84	 0.9749	     11.9806	  1.4040	   0.6587	  100.00%	   512	      8.9202	  99.11%	     3.12%
   53	attn_kv_b           	    703.52	 0.2564	     39.0490	  1.3741	   1.7476	  100.00%	   512	      8.7467	  97.19%	     1.17%
   48	attn_kv_b           	    700.92	 0.8222	     14.0137	  1.3690	   0.7406	  100.00%	   512	      8.9101	  99.00%	     1.76%
   51	attn_kv_b           	    695.03	 0.0845	     23.6498	  1.3575	   1.0650	  100.00%	   512	      8.8613	  98.46%	     1.95%
   49	attn_kv_b           	    612.83	 0.0039	     24.0295	  1.1969	   1.0562	  100.00%	   512	      8.8483	  98.31%	     1.56%
   42	attn_kv_b           	    504.51	 0.1635	      5.2517	  0.9854	   0.3455	  100.00%	   512	      8.9460	  99.40%	     3.32%
   39	attn_kv_b           	    503.64	 0.6865	     12.0894	  0.9837	   0.6730	  100.00%	   512	      8.8509	  98.34%	     3.32%
   38	attn_kv_b           	    444.43	 0.1402	     10.3335	  0.8680	   0.5410	  100.00%	   512	      8.8793	  98.66%	     3.52%
   45	attn_kv_b           	    402.63	 0.1703	      5.7610	  0.7864	   0.4650	  100.00%	   512	      8.8696	  98.55%	     2.93%
   44	attn_kv_b           	    387.33	 0.0004	     16.0984	  0.7565	   0.7421	  100.00%	   512	      8.7984	  97.76%	     1.95%
   41	attn_kv_b           	    361.93	 0.0001	     12.1827	  0.7069	   0.5555	  100.00%	   512	      8.8518	  98.35%	     2.34%
   37	attn_kv_b           	    274.39	 0.3684	      5.1937	  0.5359	   0.3424	  100.00%	   512	      8.8541	  98.38%	     4.88%
   40	attn_kv_b           	    242.05	 0.3611	      2.1434	  0.4728	   0.1593	  100.00%	   512	      8.9484	  99.43%	     2.73%
   33	attn_kv_b           	    220.05	 0.0542	      8.7845	  0.4298	   0.4231	  100.00%	   512	      8.8099	  97.89%	     0.98%
   35	attn_kv_b           	    183.88	 0.2648	      7.3889	  0.3591	   0.3258	  100.00%	   512	      8.8431	  98.26%	     1.56%
   36	attn_kv_b           	    178.06	 0.2396	      4.3345	  0.3478	   0.2659	  100.00%	   512	      8.8125	  97.92%	     3.12%
   32	attn_kv_b           	    175.28	 0.0932	      5.5267	  0.3424	   0.2547	  100.00%	   512	      8.8629	  98.48%	     2.34%
   34	attn_kv_b           	    174.02	 0.2489	      3.9327	  0.3399	   0.2438	  100.00%	   512	      8.8384	  98.20%	     2.34%
   31	attn_kv_b           	    149.06	 0.2084	      3.9671	  0.2911	   0.2000	  100.00%	   512	      8.8630	  98.48%	     3.12%
   29	attn_kv_b           	    138.36	 0.1415	      3.2425	  0.2702	   0.1785	  100.00%	   512	      8.8653	  98.50%	     3.32%
   28	attn_kv_b           	    132.83	 0.1636	      4.4650	  0.2594	   0.2310	  100.00%	   512	      8.7947	  97.72%	     2.93%
   30	attn_kv_b           	    114.01	 0.1569	      2.5871	  0.2227	   0.1762	  100.00%	   512	      8.8213	  98.01%	     1.76%
   26	attn_kv_b           	     81.90	 0.0896	      1.4522	  0.1600	   0.0826	  100.00%	   512	      8.9017	  98.91%	     3.71%
   27	attn_kv_b           	     80.11	 0.1076	      1.3855	  0.1565	   0.0793	  100.00%	   512	      8.9065	  98.96%	     2.73%
   24	attn_kv_b           	     54.69	 0.0755	      0.9860	  0.1068	   0.0529	  100.00%	   512	      8.9115	  99.02%	     3.32%
   25	attn_kv_b           	     50.91	 0.0506	      1.0480	  0.0994	   0.0676	  100.00%	   512	      8.8460	  98.29%	     3.12%
   23	attn_kv_b           	     42.40	 0.0425	      1.0716	  0.0828	   0.0516	  100.00%	   512	      8.8893	  98.77%	     2.34%
   21	attn_kv_b           	     37.33	 0.0009	      0.6518	  0.0729	   0.0412	  100.00%	   512	      8.8853	  98.73%	     3.12%
   20	attn_kv_b           	     26.03	 0.0162	      0.5325	  0.0508	   0.0332	  100.00%	   512	      8.8736	  98.60%	     1.56%
   22	attn_kv_b           	     25.88	 0.0363	      0.6945	  0.0505	   0.0452	  100.00%	   512	      8.7836	  97.60%	     2.15%
   19	attn_kv_b           	     19.91	 0.0078	      0.6287	  0.0389	   0.0305	  100.00%	   512	      8.8407	  98.23%	     2.34%
   17	attn_kv_b           	     18.19	 0.0027	      0.6438	  0.0355	   0.0373	  100.00%	   512	      8.6515	  96.13%	     6.45%
   18	attn_kv_b           	      7.33	 0.0003	      0.1274	  0.0143	   0.0107	  100.00%	   512	      8.8154	  97.95%	     2.73%
   15	attn_kv_b           	      5.69	 0.0010	      0.1487	  0.0111	   0.0076	  100.00%	   512	      8.8790	  98.66%	     1.56%
   16	attn_kv_b           	      5.43	 0.0014	      0.0778	  0.0106	   0.0059	  100.00%	   512	      8.8783	  98.65%	     4.49%
   11	attn_kv_b           	      4.37	 0.0000	      0.1024	  0.0085	   0.0059	  100.00%	   512	      8.8589	  98.43%	     2.34%
    9	attn_kv_b           	      4.08	 0.0000	      0.0975	  0.0080	   0.0061	   99.80%	   512	      8.8329	  98.14%	     2.34%
   14	attn_kv_b           	      2.58	 0.0003	      0.0537	  0.0050	   0.0037	  100.00%	   512	      8.8494	  98.33%	     1.37%
   13	attn_kv_b           	      1.65	 0.0011	      0.0678	  0.0032	   0.0032	  100.00%	   512	      8.7962	  97.74%	     1.76%
   10	attn_kv_b           	      1.59	 0.0000	      0.0314	  0.0031	   0.0022	  100.00%	   512	      8.8398	  98.22%	     3.52%
    4	attn_kv_b           	      1.05	 0.0000	      0.1156	  0.0021	   0.0055	   99.22%	   512	      7.7967	  86.63%	     2.15%
   12	attn_kv_b           	      0.81	 0.0006	      0.0261	  0.0016	   0.0018	  100.00%	   512	      8.7079	  96.75%	     2.15%
    7	attn_kv_b           	      0.25	 0.0000	      0.0050	  0.0005	   0.0004	  100.00%	   512	      8.8049	  97.83%	     3.52%
    8	attn_kv_b           	      0.20	 0.0000	      0.0278	  0.0004	   0.0015	   99.80%	   512	      7.3417	  81.57%	     1.56%
    5	attn_kv_b           	      0.15	 0.0000	      0.0031	  0.0003	   0.0003	  100.00%	   512	      8.6747	  96.39%	     6.25%
    6	attn_kv_b           	      0.08	 0.0001	      0.0013	  0.0001	   0.0001	  100.00%	   512	      8.7243	  96.94%	     6.05%
    3	attn_kv_b           	      0.05	 0.0000	      0.0030	  0.0001	   0.0003	   85.16%	   512	      7.3249	  81.39%	     7.23%
    1	attn_kv_b           	      0.04	 0.0000	      0.0082	  0.0001	   0.0005	   48.83%	   512	      4.6483	  51.65%	     1.37%
    0	attn_kv_b           	      0.02	 0.0000	      0.0114	  0.0000	   0.0005	   76.56%	   512	      4.6186	  51.32%	     0.59%
    2	attn_kv_b           	      0.02	 0.0000	      0.0025	  0.0000	   0.0002	   52.34%	   512	      4.7599	  52.89%	     1.37%
   59	attn_output         	   2256.06	 0.0002	     20.1496	  0.1377	   0.3751	  100.00%	 16384	     12.9230	  92.31%	     1.78%
   60	attn_output         	   2223.60	 0.0000	     45.2379	  0.1357	   0.6299	   99.99%	 16384	     11.7820	  84.16%	     2.48%
   58	attn_output         	    916.07	 0.0000	     11.7246	  0.0559	   0.2273	   99.85%	 16384	     11.9157	  85.11%	     3.11%
   57	attn_output         	    737.59	 0.0005	      2.0145	  0.0450	   0.0730	  100.00%	 16384	     12.9948	  92.82%	    10.07%
   56	attn_output         	    732.92	 0.0000	      0.8182	  0.0447	   0.0509	  100.00%	 16384	     13.2646	  94.75%	    13.39%
   55	attn_output         	    649.83	 0.0001	      0.3707	  0.0397	   0.0524	  100.00%	 16384	     13.1331	  93.81%	    11.69%
   54	attn_output         	    518.38	 0.0002	      0.5761	  0.0316	   0.0606	  100.00%	 16384	     12.8354	  91.68%	     5.94%
   52	attn_output         	    379.53	 0.0000	      0.2733	  0.0232	   0.0317	  100.00%	 16384	     13.0202	  93.00%	    13.09%
   50	attn_output         	    350.30	 0.0001	      0.2044	  0.0214	   0.0247	  100.00%	 16384	     13.3942	  95.67%	     6.82%
   49	attn_output         	    327.73	 0.0000	      0.1923	  0.0200	   0.0197	  100.00%	 16384	     13.3652	  95.47%	    14.56%
   53	attn_output         	    322.94	 0.0001	      0.3084	  0.0197	   0.0266	  100.00%	 16384	     13.0837	  93.45%	    11.65%
   51	attn_output         	    307.16	 0.0001	      0.3191	  0.0187	   0.0234	  100.00%	 16384	     13.2167	  94.40%	    12.93%
   45	attn_output         	    258.54	 0.0000	      0.6566	  0.0158	   0.0171	  100.00%	 16384	     13.5446	  96.75%	     7.40%
   48	attn_output         	    246.87	 0.0004	      0.1836	  0.0151	   0.0226	  100.00%	 16384	     13.1545	  93.96%	     6.77%
   46	attn_output         	    221.22	 0.0009	      0.1359	  0.0135	   0.0108	  100.00%	 16384	     13.6601	  97.57%	    10.43%
   40	attn_output         	    176.53	 0.0011	      0.1423	  0.0108	   0.0070	  100.00%	 16384	     13.7565	  98.26%	    11.60%
   47	attn_output         	    169.71	 0.0004	      0.2394	  0.0104	   0.0103	  100.00%	 16384	     13.5015	  96.44%	     9.54%
   44	attn_output         	    166.33	 0.0001	      0.1025	  0.0102	   0.0088	  100.00%	 16384	     13.5438	  96.74%	    13.82%
   41	attn_output         	    151.62	 0.0001	      0.2025	  0.0093	   0.0120	  100.00%	 16384	     13.2275	  94.48%	     8.29%
   42	attn_output         	    145.63	 0.0000	      0.2178	  0.0089	   0.0080	  100.00%	 16384	     13.5394	  96.71%	    10.53%
   43	attn_output         	    130.97	 0.0000	      0.1820	  0.0080	   0.0056	  100.00%	 16384	     13.6839	  97.74%	    11.63%
   36	attn_output         	    111.38	 0.0004	      0.0755	  0.0068	   0.0048	  100.00%	 16384	     13.7184	  97.99%	     9.89%
   39	attn_output         	    106.88	 0.0002	      0.0873	  0.0065	   0.0061	  100.00%	 16384	     13.5046	  96.46%	     9.92%
   37	attn_output         	    103.40	 0.0000	      0.0977	  0.0063	   0.0072	  100.00%	 16384	     13.3763	  95.55%	     8.83%
   38	attn_output         	     88.38	 0.0002	      0.0638	  0.0054	   0.0062	  100.00%	 16384	     13.3962	  95.69%	     7.75%
   31	attn_output         	     85.47	 0.0001	      0.0668	  0.0052	   0.0038	  100.00%	 16384	     13.7217	  98.01%	     6.43%
   34	attn_output         	     79.43	 0.0003	      0.0379	  0.0048	   0.0034	  100.00%	 16384	     13.7335	  98.10%	     8.85%
   30	attn_output         	     71.85	 0.0002	      0.0503	  0.0044	   0.0045	  100.00%	 16384	     13.5299	  96.64%	     5.84%
   35	attn_output         	     67.57	 0.0002	      0.0511	  0.0041	   0.0031	  100.00%	 16384	     13.6574	  97.55%	    11.18%
   29	attn_output         	     63.43	 0.0004	      0.0535	  0.0039	   0.0027	  100.00%	 16384	     13.7382	  98.13%	     8.00%
   32	attn_output         	     60.09	 0.0001	      0.0887	  0.0037	   0.0030	  100.00%	 16384	     13.6176	  97.27%	    10.94%
   33	attn_output         	     51.98	 0.0001	      0.0353	  0.0032	   0.0033	  100.00%	 16384	     13.4085	  95.77%	    13.37%
   28	attn_output         	     50.97	 0.0001	      0.0510	  0.0031	   0.0024	  100.00%	 16384	     13.7028	  97.88%	     8.36%
   27	attn_output         	     49.07	 0.0010	      0.0674	  0.0030	   0.0021	  100.00%	 16384	     13.7798	  98.43%	     6.32%
   26	attn_output         	     35.64	 0.0002	      0.0451	  0.0022	   0.0015	  100.00%	 16384	     13.7605	  98.29%	     8.06%
   25	attn_output         	     30.20	 0.0001	      0.0211	  0.0018	   0.0014	  100.00%	 16384	     13.7249	  98.03%	     8.29%
   24	attn_output         	     26.62	 0.0000	      0.0162	  0.0016	   0.0013	  100.00%	 16384	     13.7012	  97.87%	     7.35%
   23	attn_output         	     18.72	 0.0000	      0.0179	  0.0011	   0.0009	  100.00%	 16384	     13.6784	  97.70%	     7.66%
   22	attn_output         	     14.94	 0.0000	      0.0147	  0.0009	   0.0011	  100.00%	 16384	     13.4394	  96.00%	     5.87%
   20	attn_output         	      9.40	 0.0000	      0.0087	  0.0006	   0.0007	   99.55%	 16384	     13.3127	  95.09%	    11.13%
   21	attn_output         	      7.85	 0.0000	      0.0315	  0.0005	   0.0007	  100.00%	 16384	     13.2632	  94.74%	     4.85%
   19	attn_output         	      4.22	 0.0000	      0.0064	  0.0003	   0.0004	   99.61%	 16384	     12.9946	  92.82%	     5.90%
   18	attn_output         	      3.97	 0.0000	      0.0066	  0.0002	   0.0004	   98.41%	 16384	     13.0795	  93.43%	     9.68%
   16	attn_output         	      2.44	 0.0000	      0.0071	  0.0001	   0.0002	   96.51%	 16384	     13.0842	  93.46%	     9.36%
   17	attn_output         	      1.97	 0.0000	      0.0157	  0.0001	   0.0004	   92.42%	 16384	     12.0951	  86.39%	     4.36%
   15	attn_output         	      1.72	 0.0000	      0.0039	  0.0001	   0.0001	   99.41%	 16384	     13.2541	  94.67%	     6.37%
   14	attn_output         	      1.34	 0.0000	      0.0019	  0.0001	   0.0001	   99.49%	 16384	     13.4760	  96.26%	     8.22%
   13	attn_output         	      0.91	 0.0000	      0.0019	  0.0001	   0.0001	   98.88%	 16384	     13.5013	  96.44%	    10.00%
   10	attn_output         	      0.73	 0.0000	      0.0007	  0.0000	   0.0000	   92.07%	 16384	     13.2821	  94.87%	    12.11%
   12	attn_output         	      0.67	 0.0000	      0.0016	  0.0000	   0.0000	   95.09%	 16384	     13.4088	  95.78%	     8.01%
    9	attn_output         	      0.57	 0.0000	      0.0024	  0.0000	   0.0001	   92.27%	 16384	     12.9745	  92.68%	     8.11%
   11	attn_output         	      0.49	 0.0000	      0.0022	  0.0000	   0.0001	   83.72%	 16384	     12.6325	  90.23%	     8.00%
    7	attn_output         	      0.16	 0.0000	      0.0003	  0.0000	   0.0000	   80.84%	 16384	     12.8689	  91.92%	     7.69%
    8	attn_output         	      0.15	 0.0000	      0.0005	  0.0000	   0.0000	   81.81%	 16384	     13.0056	  92.90%	     6.27%
    5	attn_output         	      0.15	 0.0000	      0.0002	  0.0000	   0.0000	   75.36%	 16384	     12.6737	  90.53%	     6.27%
    6	attn_output         	      0.14	 0.0000	      0.0002	  0.0000	   0.0000	   78.35%	 16384	     12.9705	  92.65%	     9.44%
    4	attn_output         	      0.10	 0.0000	      0.0018	  0.0000	   0.0000	   37.54%	 16384	     10.4958	  74.97%	     2.15%
    3	attn_output         	      0.09	 0.0000	      0.0001	  0.0000	   0.0000	   39.42%	 16384	     11.7575	  83.98%	     8.06%
    0	attn_output         	      0.07	 0.0000	      0.0014	  0.0000	   0.0000	    9.58%	 16384	      8.8406	  63.15%	     1.97%
    1	attn_output         	      0.03	 0.0000	      0.0001	  0.0000	   0.0000	   25.74%	 16384	     11.7785	  84.13%	     6.84%
    2	attn_output         	      0.02	 0.0000	      0.0001	  0.0000	   0.0000	   27.37%	 16384	     12.5260	  89.47%	     6.23%
   59	attn_q_a            	     90.49	 0.0030	     12.4869	  0.0126	   0.1507	  100.00%	  7168	     11.0850	  86.55%	     0.18%
   56	attn_q_a            	     80.09	 0.0047	      8.0205	  0.0112	   0.1075	  100.00%	  7168	     10.9840	  85.76%	     0.31%
   53	attn_q_a            	     70.07	 0.0044	      7.5596	  0.0098	   0.1005	  100.00%	  7168	     10.8180	  84.47%	     0.32%
   49	attn_q_a            	     69.86	 0.0048	      3.3494	  0.0097	   0.0605	  100.00%	  7168	     11.2925	  88.17%	     0.40%
   46	attn_q_a            	     66.83	 0.0042	      5.2714	  0.0093	   0.0802	  100.00%	  7168	     11.0102	  85.97%	     0.29%
    8	attn_q_a            	     65.87	 0.0003	     30.7816	  0.0092	   0.3722	  100.00%	  7168	      5.6626	  44.21%	     0.18%
   45	attn_q_a            	     65.12	 0.0041	      2.7374	  0.0091	   0.0630	  100.00%	  7168	     11.0425	  86.22%	     0.39%
   55	attn_q_a            	     64.58	 0.0045	      4.1384	  0.0090	   0.0651	  100.00%	  7168	     11.3060	  88.28%	     0.38%
   52	attn_q_a            	     63.81	 0.0040	      4.6357	  0.0089	   0.0695	  100.00%	  7168	     11.1023	  86.69%	     0.39%
   42	attn_q_a            	     62.13	 0.0041	      3.5418	  0.0087	   0.0734	  100.00%	  7168	     10.6817	  83.40%	     0.35%
   40	attn_q_a            	     60.16	 0.0037	      4.7100	  0.0084	   0.0803	  100.00%	  7168	     10.5976	  82.75%	     0.32%
   50	attn_q_a            	     58.66	 0.0041	      3.0096	  0.0082	   0.0495	  100.00%	  7168	     11.5214	  89.96%	     0.46%
   43	attn_q_a            	     57.84	 0.0041	      2.7142	  0.0081	   0.0581	  100.00%	  7168	     11.0605	  86.36%	     0.35%
   54	attn_q_a            	     53.82	 0.0039	      2.6784	  0.0075	   0.0405	  100.00%	  7168	     11.8078	  92.20%	     0.29%
   36	attn_q_a            	     53.42	 0.0030	      6.0237	  0.0075	   0.0951	  100.00%	  7168	      9.9056	  77.34%	     0.31%
   39	attn_q_a            	     53.06	 0.0033	      2.9174	  0.0074	   0.0626	  100.00%	  7168	     10.6570	  83.21%	     0.39%
    6	attn_q_a            	     52.69	 0.0002	     22.8025	  0.0074	   0.2735	  100.00%	  7168	      6.4878	  50.66%	     0.20%
    3	attn_q_a            	     52.55	 0.0001	     31.5538	  0.0073	   0.3736	  100.00%	  7168	      4.8646	  37.98%	     0.13%
   48	attn_q_a            	     52.29	 0.0035	      2.9375	  0.0073	   0.0513	  100.00%	  7168	     11.1767	  87.27%	     0.33%
   47	attn_q_a            	     51.19	 0.0033	      3.1746	  0.0071	   0.0493	  100.00%	  7168	     11.2441	  87.79%	     0.47%
   31	attn_q_a            	     47.25	 0.0028	      4.2665	  0.0066	   0.0696	  100.00%	  7168	     10.2530	  80.06%	     0.35%
   30	attn_q_a            	     46.10	 0.0024	      3.8427	  0.0064	   0.0764	  100.00%	  7168	      9.8028	  76.54%	     0.36%
   57	attn_q_a            	     43.52	 0.0022	      8.5336	  0.0061	   0.1032	  100.00%	  7168	     10.1204	  79.02%	     0.27%
   51	attn_q_a            	     43.38	 0.0027	      3.0131	  0.0061	   0.0441	  100.00%	  7168	     11.1298	  86.90%	     0.42%
   44	attn_q_a            	     43.34	 0.0020	      5.2019	  0.0060	   0.0773	  100.00%	  7168	      9.7626	  76.23%	     0.35%
    2	attn_q_a            	     43.09	 0.0000	     18.1894	  0.0060	   0.2170	   99.99%	  7168	      6.6727	  52.10%	     0.18%
   35	attn_q_a            	     43.04	 0.0026	      3.6656	  0.0060	   0.0589	  100.00%	  7168	     10.3826	  81.07%	     0.35%
   58	attn_q_a            	     41.57	 0.0019	      1.4918	  0.0058	   0.0283	  100.00%	  7168	     11.7008	  91.36%	     0.54%
   34	attn_q_a            	     40.83	 0.0023	      4.2025	  0.0057	   0.0654	  100.00%	  7168	     10.0369	  78.37%	     0.35%
   29	attn_q_a            	     40.42	 0.0021	      4.0808	  0.0056	   0.0676	  100.00%	  7168	      9.8758	  77.11%	     0.38%
   37	attn_q_a            	     40.14	 0.0019	      4.1508	  0.0056	   0.0705	  100.00%	  7168	      9.8134	  76.62%	     0.32%
   33	attn_q_a            	     39.93	 0.0022	      3.4713	  0.0056	   0.0569	  100.00%	  7168	     10.2643	  80.14%	     0.39%
   32	attn_q_a            	     39.70	 0.0024	      3.5055	  0.0055	   0.0567	  100.00%	  7168	     10.2928	  80.37%	     0.38%
   38	attn_q_a            	     39.46	 0.0021	      3.5038	  0.0055	   0.0595	  100.00%	  7168	     10.2390	  79.95%	     0.33%
   41	attn_q_a            	     39.27	 0.0023	      2.6274	  0.0055	   0.0536	  100.00%	  7168	     10.3751	  81.01%	     0.31%
    1	attn_q_a            	     38.02	 0.0000	      9.3369	  0.0053	   0.1163	   99.97%	  7168	      7.6337	  59.60%	     0.40%
   27	attn_q_a            	     37.55	 0.0021	      2.9428	  0.0052	   0.0576	  100.00%	  7168	     10.1568	  79.30%	     0.36%
    0	attn_q_a            	     37.33	 0.0001	      4.3022	  0.0052	   0.0674	  100.00%	  7168	      8.3011	  64.81%	     1.12%
    5	attn_q_a            	     36.35	 0.0000	      8.2527	  0.0051	   0.1102	  100.00%	  7168	      8.1113	  63.33%	     0.27%
   12	attn_q_a            	     35.13	 0.0005	      9.7724	  0.0049	   0.1234	  100.00%	  7168	      7.7981	  60.89%	     0.36%
   28	attn_q_a            	     35.01	 0.0018	      3.0860	  0.0049	   0.0548	  100.00%	  7168	      9.9199	  77.45%	     0.39%
    7	attn_q_a            	     33.68	 0.0003	      9.6207	  0.0047	   0.1187	  100.00%	  7168	      8.1082	  63.31%	     0.28%
   60	attn_q_a            	     32.02	 0.0000	      5.2868	  0.0045	   0.0634	   99.99%	  7168	     10.8390	  84.63%	     0.15%
   26	attn_q_a            	     31.92	 0.0016	      3.4728	  0.0045	   0.0544	  100.00%	  7168	      9.9117	  77.39%	     0.35%
   25	attn_q_a            	     30.18	 0.0014	      2.8025	  0.0042	   0.0548	  100.00%	  7168	      9.5139	  74.28%	     0.38%
   22	attn_q_a            	     26.66	 0.0008	      3.7990	  0.0037	   0.0641	  100.00%	  7168	      8.3974	  65.57%	     0.35%
   24	attn_q_a            	     25.26	 0.0012	      2.7091	  0.0035	   0.0441	  100.00%	  7168	      9.7836	  76.39%	     0.32%
   23	attn_q_a            	     23.71	 0.0010	      2.4957	  0.0033	   0.0442	  100.00%	  7168	      9.3907	  73.32%	     0.33%
   13	attn_q_a            	     22.19	 0.0004	      4.5967	  0.0031	   0.0604	  100.00%	  7168	      8.6560	  67.59%	     0.36%
   18	attn_q_a            	     18.76	 0.0004	      4.7766	  0.0026	   0.0634	  100.00%	  7168	      7.4838	  58.43%	     0.29%
   20	attn_q_a            	     18.39	 0.0006	      2.0356	  0.0026	   0.0364	  100.00%	  7168	      9.0449	  70.62%	     0.42%
   21	attn_q_a            	     18.15	 0.0008	      1.4004	  0.0025	   0.0308	  100.00%	  7168	      9.5419	  74.50%	     0.38%
    4	attn_q_a            	     17.48	 0.0000	      3.9561	  0.0024	   0.0508	  100.00%	  7168	      8.3132	  64.91%	     0.29%
   19	attn_q_a            	     16.86	 0.0005	      2.3614	  0.0024	   0.0371	  100.00%	  7168	      8.7611	  68.41%	     0.40%
   14	attn_q_a            	     16.72	 0.0005	      2.2532	  0.0023	   0.0319	  100.00%	  7168	      9.6589	  75.42%	     0.40%
   10	attn_q_a            	     15.69	 0.0002	      3.4866	  0.0022	   0.0459	  100.00%	  7168	      8.2331	  64.28%	     0.33%
   16	attn_q_a            	     14.88	 0.0003	      3.3163	  0.0021	   0.0443	  100.00%	  7168	      7.9409	  62.00%	     0.36%
   11	attn_q_a            	     12.25	 0.0002	      2.8678	  0.0017	   0.0367	  100.00%	  7168	      8.1340	  63.51%	     0.40%
    9	attn_q_a            	     11.66	 0.0001	      2.1372	  0.0016	   0.0296	  100.00%	  7168	      8.5938	  67.10%	     0.42%
   15	attn_q_a            	     11.06	 0.0004	      1.3714	  0.0015	   0.0197	  100.00%	  7168	      9.8387	  76.82%	     0.45%
   17	attn_q_a            	      9.08	 0.0002	      1.0626	  0.0013	   0.0159	  100.00%	  7168	      9.6649	  75.46%	     0.54%
   15	attn_q_b            	   4898.20	 0.0039	     13.3113	  3.1889	   2.0671	  100.00%	  1536	     10.2478	  96.81%	    13.87%
   17	attn_q_b            	   4308.99	 0.0015	     23.4383	  2.8053	   2.1873	  100.00%	  1536	     10.1596	  95.98%	    13.28%
   14	attn_q_b            	   3394.86	 0.0037	     13.0595	  2.2102	   1.7177	  100.00%	  1536	     10.1377	  95.77%	    13.67%
   20	attn_q_b            	   3074.27	 0.0009	      8.5872	  2.0015	   0.7373	  100.00%	  1536	     10.4916	  99.12%	    10.09%
   16	attn_q_b            	   3056.14	 0.0052	     12.2748	  1.9897	   1.3679	  100.00%	  1536	     10.2628	  96.96%	    11.78%
    9	attn_q_b            	   2959.18	 0.0029	     11.6299	  1.9265	   1.4102	  100.00%	  1536	     10.2074	  96.43%	    13.41%
   10	attn_q_b            	   2857.09	 0.0142	     14.3529	  1.8601	   1.3366	  100.00%	  1536	     10.2324	  96.67%	    11.59%
   24	attn_q_b            	   2853.79	 0.2388	      4.5031	  1.8579	   0.4692	  100.00%	  1536	     10.5427	  99.60%	    12.43%
   25	attn_q_b            	   2849.11	 0.5384	      9.0233	  1.8549	   0.7104	  100.00%	  1536	     10.5101	  99.29%	     8.85%
   11	attn_q_b            	   2803.07	 0.0013	     13.3497	  1.8249	   1.6738	  100.00%	  1536	     10.0154	  94.62%	    11.46%
   18	attn_q_b            	   2686.75	 0.0046	     25.2570	  1.7492	   1.2205	  100.00%	  1536	     10.2926	  97.24%	    11.07%
   19	attn_q_b            	   2645.55	 0.0070	     13.5765	  1.7224	   0.9523	  100.00%	  1536	     10.3828	  98.09%	     9.18%
   21	attn_q_b            	   2612.06	 0.0181	      9.7499	  1.7006	   0.6858	  100.00%	  1536	     10.4779	  98.99%	     8.46%
   13	attn_q_b            	   2594.99	 0.0011	     11.0899	  1.6894	   1.6663	  100.00%	  1536	      9.9292	  93.81%	    14.97%
   23	attn_q_b            	   2568.32	 0.2155	      7.2474	  1.6721	   0.6191	  100.00%	  1536	     10.5066	  99.26%	     9.70%
   26	attn_q_b            	   2552.49	 0.5804	      7.8258	  1.6618	   0.5362	  100.00%	  1536	     10.5292	  99.47%	     7.49%
   27	attn_q_b            	   2384.72	 0.2631	      5.1858	  1.5526	   0.4378	  100.00%	  1536	     10.5383	  99.56%	     9.24%
   22	attn_q_b            	   2338.36	 0.0583	      5.3827	  1.5224	   0.5864	  100.00%	  1536	     10.4829	  99.04%	    12.50%
   30	attn_q_b            	   2045.48	 0.1801	      5.0771	  1.3317	   0.3821	  100.00%	  1536	     10.5338	  99.52%	    11.13%
   28	attn_q_b            	   2010.25	 0.3732	      4.8973	  1.3088	   0.4051	  100.00%	  1536	     10.5289	  99.47%	     9.90%
   36	attn_q_b            	   2002.53	 0.1708	      3.9997	  1.3037	   0.3997	  100.00%	  1536	     10.5223	  99.41%	    12.89%
    4	attn_q_b            	   1909.60	 0.0008	     15.8039	  1.2432	   1.5481	  100.00%	  1536	      9.8492	  93.05%	     9.70%
   29	attn_q_b            	   1825.52	 0.6715	      6.3764	  1.1885	   0.3670	  100.00%	  1536	     10.5311	  99.49%	     9.90%
   34	attn_q_b            	   1655.94	 0.1306	      3.4338	  1.0781	   0.3488	  100.00%	  1536	     10.5203	  99.39%	    11.26%
   35	attn_q_b            	   1608.97	 0.1139	      4.9449	  1.0475	   0.3705	  100.00%	  1536	     10.5188	  99.37%	     7.55%
   32	attn_q_b            	   1584.73	 0.4995	      4.8153	  1.0317	   0.3620	  100.00%	  1536	     10.5164	  99.35%	     9.31%
   31	attn_q_b            	   1513.96	 0.3923	      5.3906	  0.9857	   0.3545	  100.00%	  1536	     10.5189	  99.38%	     7.03%
   12	attn_q_b            	   1513.57	 0.0025	      8.2049	  0.9854	   1.0615	  100.00%	  1536	      9.8949	  93.48%	    12.50%
   40	attn_q_b            	   1512.81	 0.0205	      3.2214	  0.9849	   0.3062	  100.00%	  1536	     10.5210	  99.40%	    11.13%
   37	attn_q_b            	   1437.20	 0.0171	      3.3046	  0.9357	   0.3550	  100.00%	  1536	     10.4867	  99.07%	    11.46%
    7	attn_q_b            	   1383.37	 0.0035	     23.5277	  0.9006	   1.1451	  100.00%	  1536	      9.8051	  92.63%	    10.94%
   38	attn_q_b            	   1240.76	 0.0441	      2.8901	  0.8078	   0.2619	  100.00%	  1536	     10.5141	  99.33%	    11.46%
   51	attn_q_b            	   1223.03	 0.0109	      3.9245	  0.7962	   0.3609	  100.00%	  1536	     10.4461	  98.69%	    11.33%
   41	attn_q_b            	   1202.66	 0.0398	      3.1615	  0.7830	   0.2857	  100.00%	  1536	     10.5017	  99.21%	    10.48%
   33	attn_q_b            	   1106.37	 0.0648	      3.0177	  0.7203	   0.2551	  100.00%	  1536	     10.5092	  99.28%	     9.64%
   44	attn_q_b            	   1097.14	 0.0027	      3.2862	  0.7143	   0.4210	  100.00%	  1536	     10.3422	  97.71%	    13.02%
   39	attn_q_b            	   1086.03	 0.2737	      3.4080	  0.7070	   0.2529	  100.00%	  1536	     10.5160	  99.35%	     8.33%
    5	attn_q_b            	   1074.41	 0.0030	     97.6718	  0.6995	   3.0029	  100.00%	  1536	      8.8704	  83.80%	     1.24%
   45	attn_q_b            	   1042.66	 0.0014	      4.8517	  0.6788	   0.3498	  100.00%	  1536	     10.4278	  98.52%	     9.31%
   42	attn_q_b            	    994.22	 0.0034	      1.8925	  0.6473	   0.1928	  100.00%	  1536	     10.5271	  99.45%	    12.89%
   57	attn_q_b            	    906.67	 0.0002	      4.9234	  0.5903	   0.4409	  100.00%	  1536	     10.2317	  96.66%	    10.55%
   49	attn_q_b            	    900.11	 0.0119	      2.0469	  0.5860	   0.2232	  100.00%	  1536	     10.4822	  99.03%	    13.15%
    6	attn_q_b            	    888.45	 0.0014	     12.2543	  0.5784	   0.7697	  100.00%	  1536	      9.7627	  92.23%	     9.77%
   60	attn_q_b            	    863.70	 0.0007	     11.7012	  0.5623	   0.8940	  100.00%	  1536	      9.4711	  89.48%	    10.16%
   47	attn_q_b            	    839.23	 0.0025	      4.3674	  0.5464	   0.2500	  100.00%	  1536	     10.4786	  99.00%	     6.12%
   43	attn_q_b            	    791.75	 0.0006	      3.5828	  0.5155	   0.2563	  100.00%	  1536	     10.4643	  98.86%	     7.16%
   48	attn_q_b            	    711.80	 0.0002	      2.4682	  0.4634	   0.2380	  100.00%	  1536	     10.4201	  98.44%	     9.90%
   52	attn_q_b            	    698.59	 0.0009	      2.7554	  0.4548	   0.2461	  100.00%	  1536	     10.3982	  98.24%	    10.94%
   58	attn_q_b            	    660.69	 0.0000	      7.2421	  0.4301	   0.5576	  100.00%	  1536	      9.7975	  92.56%	     9.18%
    8	attn_q_b            	    608.56	 0.0008	     14.3081	  0.3962	   0.7170	  100.00%	  1536	      9.3927	  88.74%	     8.72%
   56	attn_q_b            	    570.15	 0.0000	      3.4873	  0.3712	   0.3198	  100.00%	  1536	     10.1900	  96.27%	     9.90%
   53	attn_q_b            	    566.11	 0.0040	      1.5279	  0.3686	   0.1813	  100.00%	  1536	     10.4245	  98.48%	    13.41%
   59	attn_q_b            	    564.93	 0.0000	      5.6375	  0.3678	   0.3650	   99.87%	  1536	     10.0970	  95.39%	     9.31%
   55	attn_q_b            	    541.02	 0.0000	      2.6658	  0.3522	   0.1818	  100.00%	  1536	     10.4361	  98.59%	     8.53%
   50	attn_q_b            	    509.99	 0.0000	      2.3454	  0.3320	   0.1798	   99.93%	  1536	     10.4149	  98.39%	     8.66%
   54	attn_q_b            	    498.41	 0.0000	      1.8858	  0.3245	   0.1857	  100.00%	  1536	     10.3392	  97.68%	    12.30%
    1	attn_q_b            	    496.95	 0.0001	     14.0359	  0.3235	   0.7821	  100.00%	  1536	      8.8694	  83.79%	     6.84%
   46	attn_q_b            	    460.99	 0.0001	      3.1108	  0.3001	   0.1930	  100.00%	  1536	     10.3853	  98.11%	     6.58%
    2	attn_q_b            	    455.55	 0.0004	      5.3332	  0.2966	   0.5375	  100.00%	  1536	      9.2562	  87.45%	     9.38%
    3	attn_q_b            	    438.44	 0.0008	      6.0336	  0.2854	   0.5114	  100.00%	  1536	      9.3591	  88.42%	     8.33%
    0	attn_q_b            	    421.85	 0.0043	     75.1209	  0.2746	   2.2495	  100.00%	  1536	      7.6381	  72.16%	     0.85%
    0	ffn_down            	      0.10	 0.0000	      0.0620	  0.0000	   0.0005	    1.06%	 18432	      2.6024	  18.37%	     0.09%
    2	ffn_down            	      0.03	 0.0000	      0.0044	  0.0000	   0.0000	    1.25%	 18432	      6.4311	  45.39%	     0.60%
    1	ffn_down            	      0.01	 0.0000	      0.0013	  0.0000	   0.0000	    0.87%	 18432	      6.9409	  48.98%	     0.45%
   60	ffn_down_exps       	1427484160.00	 0.0000	468131808.0000	2722.7100	870953.0625	   88.36%	524288	      3.0095	  15.84%	     0.00%
   59	ffn_down_exps       	1584705.50	 0.0000	 177050.6094	  3.0226	 415.1663	   99.39%	524288	      8.4992	  44.73%	     0.04%
   58	ffn_down_exps       	 242964.50	 0.0000	   6859.1543	  0.4634	  15.0820	   99.91%	524288	     16.7247	  88.02%	     0.05%
   57	ffn_down_exps       	 201643.98	 0.0000	    656.0131	  0.3846	   1.8084	   99.94%	524288	     17.9736	  94.60%	     1.29%
   56	ffn_down_exps       	 179375.91	 0.0000	   1569.5106	  0.3421	   2.5400	   99.96%	524288	     18.0471	  94.98%	     0.50%
   55	ffn_down_exps       	 158350.44	 0.0000	    278.4516	  0.3020	   0.8650	   99.98%	524288	     18.2290	  95.94%	     2.37%
   54	ffn_down_exps       	 120926.02	 0.0000	    192.8161	  0.2306	   0.5291	   99.99%	524288	     18.2689	  96.15%	     3.35%
   53	ffn_down_exps       	 117281.12	 0.0000	     83.7105	  0.2237	   0.3874	   99.99%	524288	     18.3404	  96.53%	     5.17%
   52	ffn_down_exps       	 101822.54	 0.0000	    116.1872	  0.1942	   0.4036	   99.99%	524288	     18.3544	  96.60%	     3.60%
   51	ffn_down_exps       	  94081.48	 0.0000	    445.0449	  0.1794	   0.9121	  100.00%	524288	     18.2085	  95.83%	     0.85%
   50	ffn_down_exps       	  82177.88	 0.0000	     76.8421	  0.1567	   0.2628	  100.00%	524288	     18.3961	  96.82%	     5.09%
   49	ffn_down_exps       	  74394.23	 0.0000	    205.1828	  0.1419	   0.4407	  100.00%	524288	     18.3488	  96.57%	     1.92%
   48	ffn_down_exps       	  63786.91	 0.0000	     75.5943	  0.1217	   0.2597	  100.00%	524288	     18.3503	  96.58%	     3.52%
   47	ffn_down_exps       	  58732.44	 0.0000	     42.9317	  0.1120	   0.1934	  100.00%	524288	     18.4322	  97.01%	     4.44%
   46	ffn_down_exps       	  55001.15	 0.0000	    742.2943	  0.1049	   1.6408	  100.00%	524288	     17.8671	  94.04%	     0.08%
   45	ffn_down_exps       	  49853.35	 0.0000	    117.0673	  0.0951	   0.3184	  100.00%	524288	     18.3280	  96.46%	     1.37%
   44	ffn_down_exps       	  43965.39	 0.0000	     41.9712	  0.0839	   0.1498	  100.00%	524288	     18.4421	  97.06%	     4.28%
   43	ffn_down_exps       	  38034.37	 0.0000	     47.6111	  0.0725	   0.1218	  100.00%	524288	     18.4817	  97.27%	     4.64%
   42	ffn_down_exps       	  35822.17	 0.0000	     98.8058	  0.0683	   0.2288	   99.99%	524288	     18.4564	  97.14%	     1.22%
   41	ffn_down_exps       	  33698.05	 0.0000	    171.5939	  0.0643	   0.2891	  100.00%	524288	     18.3354	  96.50%	     0.96%
   40	ffn_down_exps       	  29231.90	 0.0000	     10.5563	  0.0558	   0.0762	  100.00%	524288	     18.5317	  97.54%	     5.53%
   39	ffn_down_exps       	  26981.38	 0.0000	    164.4935	  0.0515	   0.2585	  100.00%	524288	     18.4112	  96.90%	     0.55%
   38	ffn_down_exps       	  23507.75	 0.0000	     63.0665	  0.0448	   0.1181	  100.00%	524288	     18.5132	  97.44%	     1.79%
   37	ffn_down_exps       	  22260.31	 0.0000	     42.8334	  0.0425	   0.1101	   99.97%	524288	     18.4383	  97.04%	     2.07%
   36	ffn_down_exps       	  20084.83	 0.0000	     25.4857	  0.0383	   0.0741	  100.00%	524288	     18.5363	  97.56%	     2.83%
   33	ffn_down_exps       	  19850.38	 0.0000	    741.4769	  0.0379	   1.9280	  100.00%	524288	     15.6416	  82.32%	     0.02%
   35	ffn_down_exps       	  18202.50	 0.0000	     57.3977	  0.0347	   0.1362	   99.99%	524288	     18.4334	  97.02%	     0.88%
   34	ffn_down_exps       	  16816.51	 0.0000	     24.7398	  0.0321	   0.0627	   99.99%	524288	     18.5034	  97.39%	     2.89%
   32	ffn_down_exps       	  14768.93	 0.0000	     14.3600	  0.0282	   0.0457	  100.00%	524288	     18.5912	  97.85%	     3.34%
   31	ffn_down_exps       	  13125.16	 0.0000	     11.1927	  0.0250	   0.0388	   99.99%	524288	     18.5688	  97.73%	     3.94%
   30	ffn_down_exps       	  11744.80	 0.0000	     17.0473	  0.0224	   0.0400	  100.00%	524288	     18.5747	  97.76%	     2.98%
   29	ffn_down_exps       	  11107.87	 0.0000	      3.9050	  0.0212	   0.0260	   99.99%	524288	     18.6090	  97.94%	     5.37%
   28	ffn_down_exps       	   9513.78	 0.0000	     12.4004	  0.0181	   0.0392	  100.00%	524288	     18.5809	  97.79%	     1.86%
   27	ffn_down_exps       	   8284.32	 0.0000	     61.6065	  0.0158	   0.0895	   99.97%	524288	     18.5233	  97.49%	     0.27%
   26	ffn_down_exps       	   6924.30	 0.0000	      5.8146	  0.0132	   0.0165	  100.00%	524288	     18.6663	  98.24%	     4.42%
   25	ffn_down_exps       	   6157.18	 0.0000	     32.2405	  0.0117	   0.0496	   99.97%	524288	     18.5635	  97.70%	     0.56%
   24	ffn_down_exps       	   5432.28	 0.0000	     10.9044	  0.0104	   0.0249	   99.99%	524288	     18.5412	  97.59%	     1.72%
   23	ffn_down_exps       	   4419.98	 0.0000	     82.8847	  0.0084	   0.1189	   99.96%	524288	     18.2329	  95.96%	     0.10%
   22	ffn_down_exps       	   3255.14	 0.0000	      9.8661	  0.0062	   0.0194	   99.96%	524288	     18.5614	  97.69%	     0.95%
    8	ffn_down_exps       	   2717.52	 0.0000	   2514.7446	  0.0052	   3.4735	   98.88%	524288	      1.4308	   7.53%	     0.00%
   21	ffn_down_exps       	   2535.68	 0.0000	      9.7229	  0.0048	   0.0157	   99.97%	524288	     18.5886	  97.83%	     0.77%
   20	ffn_down_exps       	   1958.92	 0.0000	      7.4523	  0.0037	   0.0126	   99.92%	524288	     18.6065	  97.93%	     0.72%
   19	ffn_down_exps       	   1557.38	 0.0000	      5.8262	  0.0030	   0.0117	   99.86%	524288	     18.5550	  97.66%	     0.60%
   18	ffn_down_exps       	   1284.72	 0.0000	     14.9335	  0.0025	   0.0223	   99.75%	524288	     18.3895	  96.79%	     0.14%
   13	ffn_down_exps       	   1199.58	 0.0000	    275.7088	  0.0023	   0.4687	   99.84%	524288	      9.7130	  51.12%	     0.01%
   17	ffn_down_exps       	    973.16	 0.0000	      1.7178	  0.0019	   0.0047	   99.62%	524288	     18.5279	  97.52%	     1.44%
   16	ffn_down_exps       	    817.71	 0.0000	     22.4418	  0.0016	   0.0325	   99.45%	524288	     18.1084	  95.31%	     0.03%
   15	ffn_down_exps       	    713.93	 0.0000	      5.1272	  0.0014	   0.0107	   99.88%	524288	     18.3014	  96.32%	     0.23%
   14	ffn_down_exps       	    615.45	 0.0000	     20.1744	  0.0012	   0.0311	   99.54%	524288	     17.2862	  90.98%	     0.05%
   12	ffn_down_exps       	    396.81	 0.0000	      3.2651	  0.0008	   0.0074	   99.75%	524288	     18.0962	  95.24%	     0.20%
   11	ffn_down_exps       	    330.39	 0.0000	      1.2094	  0.0006	   0.0024	   99.95%	524288	     18.4213	  96.95%	     0.96%
   10	ffn_down_exps       	    285.10	 0.0000	      4.6264	  0.0005	   0.0071	   99.81%	524288	     18.2258	  95.93%	     0.14%
    9	ffn_down_exps       	    207.70	 0.0000	      0.7035	  0.0004	   0.0018	   99.41%	524288	     18.0912	  95.22%	     1.20%
    6	ffn_down_exps       	    143.44	 0.0000	     47.4681	  0.0003	   0.0656	   97.44%	524288	     12.7939	  67.34%	     0.00%
    7	ffn_down_exps       	    118.27	 0.0000	      0.3406	  0.0002	   0.0009	   99.15%	524288	     18.1776	  95.67%	     1.19%
    5	ffn_down_exps       	     56.35	 0.0000	      0.4644	  0.0001	   0.0008	   91.09%	524288	     17.7248	  93.29%	     0.78%
    4	ffn_down_exps       	     21.69	 0.0000	      0.0639	  0.0000	   0.0002	   66.67%	524288	     16.5410	  87.06%	     2.09%
    3	ffn_down_exps       	     16.73	 0.0000	      0.6279	  0.0000	   0.0009	   55.10%	524288	     15.6772	  82.51%	     0.27%
   60	ffn_down_shexp      	 291939.81	 0.0316	  16247.2402	142.5487	 726.0824	  100.00%	  2048	      7.4141	  67.40%	     3.86%
   59	ffn_down_shexp      	  11269.72	 0.0142	   1667.0308	  5.5028	  49.3786	  100.00%	  2048	      6.8448	  62.23%	     1.46%
   58	ffn_down_shexp      	   1567.28	 0.0037	    133.1941	  0.7653	   4.2163	  100.00%	  2048	      8.6688	  78.81%	     1.56%
   57	ffn_down_shexp      	    724.09	 0.0030	     38.5607	  0.3536	   1.1812	  100.00%	  2048	      9.6368	  87.61%	     1.71%
   56	ffn_down_shexp      	    532.24	 0.0027	     35.1167	  0.2599	   0.8470	  100.00%	  2048	      9.9061	  90.06%	     2.00%
   55	ffn_down_shexp      	    366.55	 0.0020	      5.0115	  0.1790	   0.2701	  100.00%	  2048	     10.2249	  92.95%	     6.84%
   54	ffn_down_shexp      	    296.03	 0.0028	      4.5417	  0.1445	   0.2145	  100.00%	  2048	     10.2937	  93.58%	     7.18%
   52	ffn_down_shexp      	    289.31	 0.0011	     38.7988	  0.1413	   1.2306	  100.00%	  2048	      8.3114	  75.56%	     0.29%
   53	ffn_down_shexp      	    262.95	 0.0022	     23.2386	  0.1284	   0.5976	  100.00%	  2048	      9.5241	  86.58%	     0.78%
   33	ffn_down_shexp      	    177.04	 0.0039	     58.6099	  0.0864	   1.8082	  100.00%	  2048	      3.8644	  35.13%	     0.24%
   51	ffn_down_shexp      	    170.69	 0.0014	      2.5751	  0.0833	   0.1210	  100.00%	  2048	     10.2683	  93.35%	     7.47%
   50	ffn_down_shexp      	    131.79	 0.0020	      0.9058	  0.0643	   0.0730	  100.00%	  2048	     10.4147	  94.68%	     8.94%
   49	ffn_down_shexp      	    125.67	 0.0017	      1.4481	  0.0614	   0.0712	  100.00%	  2048	     10.4174	  94.70%	     9.57%
   47	ffn_down_shexp      	    109.16	 0.0018	      2.4731	  0.0533	   0.0835	  100.00%	  2048	     10.3803	  94.37%	     4.79%
   48	ffn_down_shexp      	    106.67	 0.0021	      1.1842	  0.0521	   0.0557	  100.00%	  2048	     10.5051	  95.50%	     8.98%
   45	ffn_down_shexp      	     98.14	 0.0044	      2.1655	  0.0479	   0.0670	  100.00%	  2048	     10.5186	  95.62%	     4.44%
   46	ffn_down_shexp      	     95.77	 0.0019	      0.8243	  0.0468	   0.0464	  100.00%	  2048	     10.6050	  96.41%	     7.86%
   44	ffn_down_shexp      	     82.12	 0.0049	      2.9412	  0.0401	   0.0794	  100.00%	  2048	     10.4047	  94.59%	     2.39%
   43	ffn_down_shexp      	     69.88	 0.0052	      2.4087	  0.0341	   0.0656	  100.00%	  2048	     10.4463	  94.97%	     2.64%
   42	ffn_down_shexp      	     57.88	 0.0050	      0.4198	  0.0283	   0.0259	  100.00%	  2048	     10.6691	  96.99%	     6.84%
   36	ffn_down_shexp      	     55.00	 0.0049	     19.5248	  0.0269	   0.4323	  100.00%	  2048	      7.6343	  69.40%	     0.15%
   41	ffn_down_shexp      	     54.02	 0.0060	      0.3927	  0.0264	   0.0255	  100.00%	  2048	     10.6416	  96.74%	     6.64%
   40	ffn_down_shexp      	     48.19	 0.0047	      0.5253	  0.0235	   0.0232	  100.00%	  2048	     10.6536	  96.85%	     6.69%
   14	ffn_down_shexp      	     46.14	 0.0000	     24.5456	  0.0225	   0.7142	  100.00%	  2048	      1.1926	  10.84%	     0.10%
   39	ffn_down_shexp      	     44.26	 0.0055	      0.6898	  0.0216	   0.0250	  100.00%	  2048	     10.6033	  96.39%	     5.76%
    8	ffn_down_shexp      	     43.71	 0.0000	     43.5080	  0.0213	   0.9612	  100.00%	  2048	      0.0727	   0.66%	     0.05%
   35	ffn_down_shexp      	     42.71	 0.0036	      2.8710	  0.0209	   0.1124	  100.00%	  2048	      9.2517	  84.11%	     0.98%
   38	ffn_down_shexp      	     41.46	 0.0062	      0.8854	  0.0202	   0.0278	  100.00%	  2048	     10.5393	  95.81%	     4.44%
   37	ffn_down_shexp      	     40.12	 0.0051	      4.4147	  0.0196	   0.0996	  100.00%	  2048	      9.8689	  89.72%	     0.88%
   34	ffn_down_shexp      	     28.07	 0.0040	      1.7014	  0.0137	   0.0415	  100.00%	  2048	     10.2322	  93.02%	     1.66%
   32	ffn_down_shexp      	     24.72	 0.0042	      0.3472	  0.0121	   0.0176	  100.00%	  2048	     10.4665	  95.15%	     4.00%
   31	ffn_down_shexp      	     22.45	 0.0039	      0.4385	  0.0110	   0.0171	  100.00%	  2048	     10.4471	  94.97%	     3.37%
   30	ffn_down_shexp      	     19.51	 0.0032	      0.2624	  0.0095	   0.0125	  100.00%	  2048	     10.5594	  95.99%	     3.76%
   29	ffn_down_shexp      	     18.16	 0.0027	      0.2475	  0.0089	   0.0096	  100.00%	  2048	     10.6369	  96.70%	     5.37%
   28	ffn_down_shexp      	     15.29	 0.0026	      0.1510	  0.0075	   0.0069	  100.00%	  2048	     10.6981	  97.26%	     5.66%
   27	ffn_down_shexp      	     13.04	 0.0023	      0.1757	  0.0064	   0.0065	  100.00%	  2048	     10.6818	  97.11%	     7.03%
   26	ffn_down_shexp      	     12.73	 0.0020	      0.4903	  0.0062	   0.0147	  100.00%	  2048	     10.3839	  94.40%	     1.42%
   25	ffn_down_shexp      	     12.59	 0.0017	      1.0960	  0.0061	   0.0283	  100.00%	  2048	      9.8456	  89.51%	     0.44%
   24	ffn_down_shexp      	     12.34	 0.0014	      1.6588	  0.0060	   0.0435	  100.00%	  2048	      9.0506	  82.28%	     0.39%
   22	ffn_down_shexp      	     10.47	 0.0007	      3.0412	  0.0051	   0.0681	  100.00%	  2048	      8.0979	  73.62%	     0.24%
   23	ffn_down_shexp      	      7.94	 0.0004	      0.0807	  0.0039	   0.0040	  100.00%	  2048	     10.6597	  96.91%	     4.49%
   15	ffn_down_shexp      	      6.20	 0.0001	      5.3702	  0.0030	   0.1186	  100.00%	  2048	      1.9206	  17.46%	     0.05%
   21	ffn_down_shexp      	      4.78	 0.0002	      0.0332	  0.0023	   0.0019	  100.00%	  2048	     10.7048	  97.32%	     7.81%
   20	ffn_down_shexp      	      3.14	 0.0002	      0.0351	  0.0015	   0.0015	  100.00%	  2048	     10.6472	  96.79%	     6.25%
   19	ffn_down_shexp      	      2.54	 0.0001	      0.0348	  0.0012	   0.0016	  100.00%	  2048	     10.4813	  95.28%	     5.27%
   18	ffn_down_shexp      	      1.93	 0.0001	      0.0425	  0.0009	   0.0014	  100.00%	  2048	     10.3854	  94.41%	     5.08%
   17	ffn_down_shexp      	      1.43	 0.0001	      0.0141	  0.0007	   0.0008	  100.00%	  2048	     10.4364	  94.88%	     6.79%
   16	ffn_down_shexp      	      1.40	 0.0001	      0.5226	  0.0007	   0.0116	  100.00%	  2048	      7.3799	  67.09%	     0.05%
   13	ffn_down_shexp      	      0.38	 0.0000	      0.0071	  0.0002	   0.0003	  100.00%	  2048	     10.3175	  93.80%	     6.10%
   12	ffn_down_shexp      	      0.29	 0.0000	      0.0159	  0.0001	   0.0004	  100.00%	  2048	     10.2096	  92.81%	     2.34%
   11	ffn_down_shexp      	      0.23	 0.0000	      0.0025	  0.0001	   0.0001	  100.00%	  2048	     10.3600	  94.18%	     9.08%
    9	ffn_down_shexp      	      0.19	 0.0000	      0.0034	  0.0001	   0.0002	  100.00%	  2048	     10.0837	  91.67%	     6.45%
   10	ffn_down_shexp      	      0.18	 0.0000	      0.0022	  0.0001	   0.0001	  100.00%	  2048	     10.2756	  93.41%	     8.98%
    7	ffn_down_shexp      	      0.10	 0.0000	      0.0078	  0.0000	   0.0003	  100.00%	  2048	      8.7174	  79.25%	     1.22%
    6	ffn_down_shexp      	      0.06	 0.0000	      0.0076	  0.0000	   0.0002	  100.00%	  2048	      9.1243	  82.95%	     1.17%
    5	ffn_down_shexp      	      0.03	 0.0000	      0.0009	  0.0000	   0.0000	  100.00%	  2048	      9.4177	  85.62%	     4.59%
    4	ffn_down_shexp      	      0.03	 0.0000	      0.0029	  0.0000	   0.0001	  100.00%	  2048	      9.0306	  82.10%	     2.54%
    3	ffn_down_shexp      	      0.01	 0.0000	      0.0002	  0.0000	   0.0000	  100.00%	  2048	     10.5171	  95.61%	     2.44%
    2	ffn_gate            	    859.43	 0.0000	    802.1978	  0.1199	   9.4779	   99.83%	  7168	      0.6756	   5.27%	     0.03%
    1	ffn_gate            	    592.96	 0.0000	    429.3697	  0.0827	   5.0879	   99.89%	  7168	      2.4691	  19.28%	     0.13%
    0	ffn_gate            	    483.51	 0.0000	    450.5507	  0.0675	   5.3236	   97.56%	  7168	      0.6201	   4.84%	     0.06%
   57	ffn_gate_exps       	1108622.00	 0.0574	     18.0424	  0.6042	   0.1643	  100.00%	1835008	     20.7916	  99.92%	     1.51%
   56	ffn_gate_exps       	1098842.75	 0.1342	     21.3571	  0.5988	   0.1600	  100.00%	1835008	     20.7988	  99.96%	     1.60%
   58	ffn_gate_exps       	1059858.50	 0.0017	     20.6275	  0.5776	   0.1614	  100.00%	1835008	     20.7922	  99.93%	     1.73%
   55	ffn_gate_exps       	1029864.69	 0.1899	     24.0345	  0.5612	   0.1825	  100.00%	1835008	     20.7925	  99.93%	     1.18%
   54	ffn_gate_exps       	 950597.38	 0.2668	     28.8253	  0.5180	   0.1960	  100.00%	1835008	     20.7858	  99.90%	     0.96%
   53	ffn_gate_exps       	 919925.69	 0.2293	     31.0064	  0.5013	   0.1928	  100.00%	1835008	     20.7866	  99.90%	     0.93%
   52	ffn_gate_exps       	 839725.12	 0.1856	     23.6457	  0.4576	   0.1782	  100.00%	1835008	     20.7856	  99.90%	     0.85%
   59	ffn_gate_exps       	 788085.31	 0.0001	     32.1861	  0.4295	   0.1922	  100.00%	1835008	     20.7695	  99.82%	     0.71%
   51	ffn_gate_exps       	 783379.31	 0.1706	     24.2819	  0.4269	   0.1622	  100.00%	1835008	     20.7859	  99.90%	     0.88%
   50	ffn_gate_exps       	 749826.50	 0.1400	     21.6678	  0.4086	   0.1490	  100.00%	1835008	     20.7899	  99.92%	     0.90%
   49	ffn_gate_exps       	 712692.44	 0.1545	     23.0501	  0.3884	   0.1351	  100.00%	1835008	     20.7872	  99.90%	     1.04%
   48	ffn_gate_exps       	 652600.50	 0.1266	     17.2781	  0.3556	   0.1236	  100.00%	1835008	     20.7942	  99.94%	     1.03%
   47	ffn_gate_exps       	 624720.88	 0.1098	     30.8410	  0.3404	   0.1301	  100.00%	1835008	     20.8078	 100.00%	     0.78%
   46	ffn_gate_exps       	 583974.00	 0.1477	     26.1010	  0.3182	   0.1009	  100.00%	1835008	     20.7921	  99.93%	     1.10%
   45	ffn_gate_exps       	 547631.69	 0.1284	     14.7849	  0.2984	   0.0870	  100.00%	1835008	     20.7918	  99.93%	     1.44%
   44	ffn_gate_exps       	 517168.44	 0.1231	     22.0782	  0.2818	   0.0875	  100.00%	1835008	     20.8003	  99.97%	     1.32%
   43	ffn_gate_exps       	 486536.84	 0.1024	     32.9791	  0.2651	   0.0996	  100.00%	1835008	     20.8003	  99.97%	     0.81%
   42	ffn_gate_exps       	 459638.69	 0.1057	     18.4986	  0.2505	   0.0764	  100.00%	1835008	     20.7969	  99.95%	     1.27%
   41	ffn_gate_exps       	 435830.34	 0.0979	     14.5584	  0.2375	   0.0705	  100.00%	1835008	     20.7998	  99.96%	     1.46%
   40	ffn_gate_exps       	 417437.19	 0.1014	     11.7959	  0.2275	   0.0697	  100.00%	1835008	     20.7969	  99.95%	     1.38%
   39	ffn_gate_exps       	 399054.31	 0.1064	     19.4026	  0.2175	   0.0743	  100.00%	1835008	     20.7920	  99.93%	     1.13%
   38	ffn_gate_exps       	 368285.38	 0.0749	     15.0838	  0.2007	   0.0680	  100.00%	1835008	     20.8033	  99.98%	     1.26%
   37	ffn_gate_exps       	 346157.62	 0.0642	      8.4320	  0.1886	   0.0567	  100.00%	1835008	     20.7879	  99.91%	     1.51%
   36	ffn_gate_exps       	 333243.12	 0.0730	     11.6749	  0.1816	   0.0538	  100.00%	1835008	     20.7971	  99.95%	     1.51%
   35	ffn_gate_exps       	 315236.34	 0.0432	     16.8776	  0.1718	   0.0634	  100.00%	1835008	     20.8073	 100.00%	     0.98%
   34	ffn_gate_exps       	 308240.75	 0.0462	     11.0697	  0.1680	   0.0521	  100.00%	1835008	     20.8190	 100.06%	     1.21%
   33	ffn_gate_exps       	 292961.50	 0.0501	     17.4166	  0.1597	   0.0579	  100.00%	1835008	     20.8051	  99.99%	     0.82%
   32	ffn_gate_exps       	 281822.19	 0.0545	     16.2088	  0.1536	   0.0615	  100.00%	1835008	     20.7920	  99.93%	     0.77%
   60	ffn_gate_exps       	 275449.28	 0.0000	     53.8235	  0.1501	   0.2789	  100.00%	1835008	     20.6214	  99.11%	     0.09%
   31	ffn_gate_exps       	 264012.66	 0.0627	     23.6177	  0.1439	   0.0607	  100.00%	1835008	     20.8012	  99.97%	     0.73%
   30	ffn_gate_exps       	 242871.81	 0.0746	     11.3317	  0.1324	   0.0526	  100.00%	1835008	     20.7986	  99.96%	     0.83%
   29	ffn_gate_exps       	 236621.69	 0.0708	     12.5480	  0.1289	   0.0505	  100.00%	1835008	     20.7994	  99.96%	     0.84%
   28	ffn_gate_exps       	 219571.83	 0.0656	     16.1806	  0.1197	   0.0603	  100.00%	1835008	     20.7942	  99.94%	     0.61%
   27	ffn_gate_exps       	 203887.56	 0.0648	     16.1550	  0.1111	   0.0594	  100.00%	1835008	     20.7817	  99.88%	     0.50%
   26	ffn_gate_exps       	 188690.89	 0.0456	      9.7137	  0.1028	   0.0436	  100.00%	1835008	     20.8035	  99.98%	     0.69%
   25	ffn_gate_exps       	 171281.08	 0.0441	      9.9973	  0.0933	   0.0420	  100.00%	1835008	     20.7806	  99.87%	     0.64%
   24	ffn_gate_exps       	 158806.77	 0.0401	      7.9296	  0.0865	   0.0405	  100.00%	1835008	     20.7953	  99.94%	     0.60%
   23	ffn_gate_exps       	 140877.31	 0.0399	      4.9228	  0.0768	   0.0279	  100.00%	1835008	     20.7861	  99.90%	     0.90%
   22	ffn_gate_exps       	 121295.08	 0.0384	      3.9828	  0.0661	   0.0227	  100.00%	1835008	     20.7894	  99.91%	     1.04%
   21	ffn_gate_exps       	 109139.78	 0.0260	     16.0739	  0.0595	   0.0452	  100.00%	1835008	     20.7649	  99.80%	     0.40%
   20	ffn_gate_exps       	  95741.52	 0.0227	      6.8249	  0.0522	   0.0226	  100.00%	1835008	     20.7793	  99.86%	     0.66%
   19	ffn_gate_exps       	  83921.45	 0.0200	      2.8252	  0.0457	   0.0179	  100.00%	1835008	     20.7710	  99.83%	     0.95%
   18	ffn_gate_exps       	  74025.85	 0.0140	      2.6935	  0.0403	   0.0158	  100.00%	1835008	     20.7662	  99.80%	     0.99%
   17	ffn_gate_exps       	  67284.16	 0.0135	      2.3618	  0.0367	   0.0147	  100.00%	1835008	     20.7702	  99.82%	     0.81%
   16	ffn_gate_exps       	  61220.83	 0.0103	      1.9943	  0.0334	   0.0104	  100.00%	1835008	     20.7856	  99.90%	     1.41%
   15	ffn_gate_exps       	  58135.96	 0.0112	      3.1859	  0.0317	   0.0112	  100.00%	1835008	     20.7830	  99.88%	     0.99%
   14	ffn_gate_exps       	  53397.41	 0.0089	      1.1326	  0.0291	   0.0071	  100.00%	1835008	     20.7873	  99.90%	     3.18%
   13	ffn_gate_exps       	  49976.98	 0.0044	      1.7784	  0.0272	   0.0076	  100.00%	1835008	     20.7836	  99.89%	     2.73%
   12	ffn_gate_exps       	  45768.75	 0.0021	      3.0780	  0.0249	   0.0089	  100.00%	1835008	     20.7758	  99.85%	     1.62%
   11	ffn_gate_exps       	  39124.46	 0.0006	      1.5074	  0.0213	   0.0065	  100.00%	1835008	     20.7666	  99.80%	     4.91%
   10	ffn_gate_exps       	  34817.07	 0.0007	      1.1131	  0.0190	   0.0075	  100.00%	1835008	     20.7560	  99.75%	     2.79%
    9	ffn_gate_exps       	  29854.04	 0.0009	      1.6395	  0.0163	   0.0126	  100.00%	1835008	     20.7027	  99.50%	     0.58%
    8	ffn_gate_exps       	  26950.78	 0.0006	      1.6468	  0.0147	   0.0131	  100.00%	1835008	     20.6652	  99.32%	     0.52%
    3	ffn_gate_exps       	  24427.59	 0.0000	     66.7534	  0.0133	   0.5410	   99.98%	1835008	     14.9015	  71.62%	     0.04%
    7	ffn_gate_exps       	  21764.22	 0.0001	      3.2781	  0.0119	   0.0272	  100.00%	1835008	     20.4541	  98.30%	     0.11%
    6	ffn_gate_exps       	  21277.98	 0.0001	      6.6201	  0.0116	   0.0631	  100.00%	1835008	     20.0195	  96.21%	     0.07%
    4	ffn_gate_exps       	  18856.03	 0.0000	     38.3090	  0.0103	   0.3010	   99.98%	1835008	     16.2252	  77.98%	     0.04%
    5	ffn_gate_exps       	  18769.08	 0.0000	     16.2609	  0.0102	   0.1502	  100.00%	1835008	     18.4726	  88.78%	     0.04%
   57	ffn_gate_inp        	   4342.22	 0.1044	      7.2990	  0.6058	   0.1245	  100.00%	  7168	     12.7942	  99.90%	     0.84%
   56	ffn_gate_inp        	   4303.31	 0.1893	      7.3898	  0.6003	   0.1111	  100.00%	  7168	     12.7964	  99.91%	     1.38%
   58	ffn_gate_inp        	   4154.51	 0.0036	      9.2729	  0.5796	   0.1254	  100.00%	  7168	     12.7927	  99.89%	     0.78%
   55	ffn_gate_inp        	   4032.60	 0.3283	      9.3460	  0.5626	   0.1289	  100.00%	  7168	     12.7932	  99.89%	     1.23%
   54	ffn_gate_inp        	   3724.53	 0.3516	     10.7018	  0.5196	   0.1388	  100.00%	  7168	     12.7904	  99.87%	     1.12%
   53	ffn_gate_inp        	   3604.73	 0.3538	     11.3448	  0.5029	   0.1447	  100.00%	  7168	     12.7888	  99.86%	     1.09%
   52	ffn_gate_inp        	   3288.52	 0.3025	     10.1119	  0.4588	   0.1298	  100.00%	  7168	     12.7889	  99.86%	     1.06%
   59	ffn_gate_inp        	   3083.86	 0.0004	     13.7678	  0.4302	   0.1691	  100.00%	  7168	     12.7747	  99.74%	     0.25%
   51	ffn_gate_inp        	   3067.81	 0.2711	      8.4771	  0.4280	   0.1118	  100.00%	  7168	     12.7901	  99.87%	     1.05%
   50	ffn_gate_inp        	   2942.98	 0.2604	      7.4818	  0.4106	   0.1014	  100.00%	  7168	     12.7908	  99.87%	     1.05%
   49	ffn_gate_inp        	   2792.88	 0.2567	      5.7642	  0.3896	   0.0829	  100.00%	  7168	     12.7930	  99.89%	     1.30%
   48	ffn_gate_inp        	   2556.00	 0.2407	      4.8123	  0.3566	   0.0719	  100.00%	  7168	     12.7935	  99.89%	     1.40%
   47	ffn_gate_inp        	   2446.98	 0.2099	      3.5467	  0.3414	   0.0594	  100.00%	  7168	     12.7955	  99.91%	     1.67%
   46	ffn_gate_inp        	   2285.12	 0.2029	      2.6480	  0.3188	   0.0502	  100.00%	  7168	     12.7966	  99.92%	     1.80%
   45	ffn_gate_inp        	   2143.51	 0.2553	      2.0089	  0.2990	   0.0423	  100.00%	  7168	     12.7978	  99.93%	     2.30%
   44	ffn_gate_inp        	   2024.29	 0.2251	      1.8251	  0.2824	   0.0393	  100.00%	  7168	     12.7981	  99.93%	     2.59%
   43	ffn_gate_inp        	   1905.67	 0.1806	      1.5305	  0.2659	   0.0352	  100.00%	  7168	     12.7988	  99.93%	     2.37%
   42	ffn_gate_inp        	   1798.81	 0.2058	      1.4089	  0.2510	   0.0331	  100.00%	  7168	     12.7987	  99.93%	     2.50%
   41	ffn_gate_inp        	   1705.82	 0.1887	      1.5552	  0.2380	   0.0335	  100.00%	  7168	     12.7978	  99.93%	     2.37%
   40	ffn_gate_inp        	   1633.65	 0.1743	      1.4432	  0.2279	   0.0323	  100.00%	  7168	     12.7977	  99.92%	     2.32%
   39	ffn_gate_inp        	   1560.66	 0.1826	      1.3440	  0.2177	   0.0293	  100.00%	  7168	     12.7983	  99.93%	     2.58%
   38	ffn_gate_inp        	   1440.72	 0.1637	      1.1312	  0.2010	   0.0271	  100.00%	  7168	     12.7981	  99.93%	     2.58%
   37	ffn_gate_inp        	   1353.36	 0.1321	      1.0998	  0.1888	   0.0261	  100.00%	  7168	     12.7978	  99.93%	     2.41%
   36	ffn_gate_inp        	   1302.77	 0.1082	      0.8941	  0.1817	   0.0231	  100.00%	  7168	     12.7989	  99.93%	     2.62%
   35	ffn_gate_inp        	   1232.80	 0.0755	      0.8060	  0.1720	   0.0223	  100.00%	  7168	     12.7987	  99.93%	     2.16%
   34	ffn_gate_inp        	   1204.46	 0.0729	      0.7595	  0.1680	   0.0216	  100.00%	  7168	     12.7989	  99.93%	     2.33%
   33	ffn_gate_inp        	   1143.78	 0.0709	      0.9042	  0.1596	   0.0228	  100.00%	  7168	     12.7977	  99.92%	     1.93%
   32	ffn_gate_inp        	   1099.52	 0.0818	      0.8105	  0.1534	   0.0226	  100.00%	  7168	     12.7968	  99.92%	     1.84%
   60	ffn_gate_inp        	   1078.51	 0.0001	     20.6208	  0.1505	   0.2457	  100.00%	  7168	     12.6422	  98.71%	     0.10%
   31	ffn_gate_inp        	   1029.70	 0.0938	      0.8485	  0.1437	   0.0226	  100.00%	  7168	     12.7959	  99.91%	     1.67%
   30	ffn_gate_inp        	    948.08	 0.0994	      0.8589	  0.1323	   0.0224	  100.00%	  7168	     12.7944	  99.90%	     1.59%
   29	ffn_gate_inp        	    923.32	 0.1143	      0.7502	  0.1288	   0.0208	  100.00%	  7168	     12.7952	  99.91%	     1.55%
   28	ffn_gate_inp        	    857.50	 0.1050	      0.8266	  0.1196	   0.0197	  100.00%	  7168	     12.7951	  99.90%	     1.55%
   27	ffn_gate_inp        	    795.67	 0.0908	      0.7870	  0.1110	   0.0177	  100.00%	  7168	     12.7962	  99.91%	     1.46%
   26	ffn_gate_inp        	    736.90	 0.0784	      0.7393	  0.1028	   0.0169	  100.00%	  7168	     12.7955	  99.91%	     1.46%
   25	ffn_gate_inp        	    667.83	 0.0700	      0.8148	  0.0932	   0.0164	  100.00%	  7168	     12.7947	  99.90%	     1.33%
   24	ffn_gate_inp        	    619.78	 0.0657	      0.8708	  0.0865	   0.0164	  100.00%	  7168	     12.7936	  99.89%	     1.20%
   23	ffn_gate_inp        	    550.91	 0.0638	      0.9747	  0.0769	   0.0176	  100.00%	  7168	     12.7898	  99.86%	     1.13%
   22	ffn_gate_inp        	    473.30	 0.0550	      0.7791	  0.0660	   0.0160	  100.00%	  7168	     12.7880	  99.85%	     1.12%
   21	ffn_gate_inp        	    425.76	 0.0463	      0.6638	  0.0594	   0.0159	  100.00%	  7168	     12.7845	  99.82%	     0.98%
   20	ffn_gate_inp        	    373.53	 0.0377	      0.5380	  0.0521	   0.0109	  100.00%	  7168	     12.7912	  99.87%	     1.19%
   19	ffn_gate_inp        	    327.81	 0.0331	      0.5958	  0.0457	   0.0110	  100.00%	  7168	     12.7872	  99.84%	     1.09%
   18	ffn_gate_inp        	    288.33	 0.0259	      0.5437	  0.0402	   0.0093	  100.00%	  7168	     12.7885	  99.85%	     1.13%
   17	ffn_gate_inp        	    262.71	 0.0221	      0.6237	  0.0367	   0.0089	  100.00%	  7168	     12.7898	  99.86%	     1.05%
   16	ffn_gate_inp        	    239.14	 0.0150	      0.3143	  0.0334	   0.0052	  100.00%	  7168	     12.7968	  99.92%	     1.73%
   15	ffn_gate_inp        	    227.29	 0.0155	      0.4654	  0.0317	   0.0064	  100.00%	  7168	     12.7940	  99.90%	     1.12%
   14	ffn_gate_inp        	    208.76	 0.0130	      0.3669	  0.0291	   0.0049	  100.00%	  7168	     12.7971	  99.92%	     1.65%
   13	ffn_gate_inp        	    195.33	 0.0077	      0.3455	  0.0272	   0.0046	  100.00%	  7168	     12.7965	  99.92%	     1.69%
   12	ffn_gate_inp        	    179.43	 0.0035	      0.3448	  0.0250	   0.0047	  100.00%	  7168	     12.7938	  99.89%	     1.66%
   11	ffn_gate_inp        	    153.69	 0.0014	      0.3143	  0.0214	   0.0045	  100.00%	  7168	     12.7874	  99.84%	     2.26%
   10	ffn_gate_inp        	    136.43	 0.0012	      0.4756	  0.0190	   0.0062	  100.00%	  7168	     12.7744	  99.74%	     0.95%
    9	ffn_gate_inp        	    116.60	 0.0016	      0.9678	  0.0163	   0.0121	  100.00%	  7168	     12.7233	  99.34%	     0.28%
    8	ffn_gate_inp        	    105.89	 0.0009	      0.9859	  0.0148	   0.0127	  100.00%	  7168	     12.6870	  99.06%	     0.27%
    3	ffn_gate_inp        	     95.53	 0.0000	     44.2083	  0.0133	   0.5280	   99.97%	  7168	      6.9930	  54.60%	     0.04%
    7	ffn_gate_inp        	     85.46	 0.0005	      2.0256	  0.0119	   0.0266	  100.00%	  7168	     12.4812	  97.45%	     0.08%
    6	ffn_gate_inp        	     83.44	 0.0001	      4.7202	  0.0116	   0.0623	  100.00%	  7168	     12.0480	  94.07%	     0.07%
    4	ffn_gate_inp        	     73.80	 0.0000	     23.8029	  0.0103	   0.2955	   99.99%	  7168	      8.2841	  64.68%	     0.04%
    5	ffn_gate_inp        	     73.60	 0.0000	     11.3983	  0.0103	   0.1479	  100.00%	  7168	     10.5195	  82.14%	     0.04%
   57	ffn_gate_shexp      	   4342.22	 0.1044	      7.2990	  0.6058	   0.1245	  100.00%	  7168	     12.7942	  99.90%	     0.84%
   56	ffn_gate_shexp      	   4303.31	 0.1893	      7.3898	  0.6003	   0.1111	  100.00%	  7168	     12.7964	  99.91%	     1.38%
   58	ffn_gate_shexp      	   4154.51	 0.0036	      9.2729	  0.5796	   0.1254	  100.00%	  7168	     12.7927	  99.89%	     0.78%
   55	ffn_gate_shexp      	   4032.60	 0.3283	      9.3460	  0.5626	   0.1289	  100.00%	  7168	     12.7932	  99.89%	     1.23%
   54	ffn_gate_shexp      	   3724.53	 0.3516	     10.7018	  0.5196	   0.1388	  100.00%	  7168	     12.7904	  99.87%	     1.12%
   53	ffn_gate_shexp      	   3604.73	 0.3538	     11.3448	  0.5029	   0.1447	  100.00%	  7168	     12.7888	  99.86%	     1.09%
   52	ffn_gate_shexp      	   3288.52	 0.3025	     10.1119	  0.4588	   0.1298	  100.00%	  7168	     12.7889	  99.86%	     1.06%
   59	ffn_gate_shexp      	   3083.86	 0.0004	     13.7678	  0.4302	   0.1691	  100.00%	  7168	     12.7747	  99.74%	     0.25%
   51	ffn_gate_shexp      	   3067.81	 0.2711	      8.4771	  0.4280	   0.1118	  100.00%	  7168	     12.7901	  99.87%	     1.05%
   50	ffn_gate_shexp      	   2942.98	 0.2604	      7.4818	  0.4106	   0.1014	  100.00%	  7168	     12.7908	  99.87%	     1.05%
   49	ffn_gate_shexp      	   2792.88	 0.2567	      5.7642	  0.3896	   0.0829	  100.00%	  7168	     12.7930	  99.89%	     1.30%
   48	ffn_gate_shexp      	   2556.00	 0.2407	      4.8123	  0.3566	   0.0719	  100.00%	  7168	     12.7935	  99.89%	     1.40%
   47	ffn_gate_shexp      	   2446.98	 0.2099	      3.5467	  0.3414	   0.0594	  100.00%	  7168	     12.7955	  99.91%	     1.67%
   46	ffn_gate_shexp      	   2285.12	 0.2029	      2.6480	  0.3188	   0.0502	  100.00%	  7168	     12.7966	  99.92%	     1.80%
   45	ffn_gate_shexp      	   2143.51	 0.2553	      2.0089	  0.2990	   0.0423	  100.00%	  7168	     12.7978	  99.93%	     2.30%
   44	ffn_gate_shexp      	   2024.29	 0.2251	      1.8251	  0.2824	   0.0393	  100.00%	  7168	     12.7981	  99.93%	     2.59%
   43	ffn_gate_shexp      	   1905.67	 0.1806	      1.5305	  0.2659	   0.0352	  100.00%	  7168	     12.7988	  99.93%	     2.37%
   42	ffn_gate_shexp      	   1798.81	 0.2058	      1.4089	  0.2510	   0.0331	  100.00%	  7168	     12.7987	  99.93%	     2.50%
   41	ffn_gate_shexp      	   1705.82	 0.1887	      1.5552	  0.2380	   0.0335	  100.00%	  7168	     12.7978	  99.93%	     2.37%
   40	ffn_gate_shexp      	   1633.65	 0.1743	      1.4432	  0.2279	   0.0323	  100.00%	  7168	     12.7977	  99.92%	     2.32%
   39	ffn_gate_shexp      	   1560.66	 0.1826	      1.3440	  0.2177	   0.0293	  100.00%	  7168	     12.7983	  99.93%	     2.58%
   38	ffn_gate_shexp      	   1440.72	 0.1637	      1.1312	  0.2010	   0.0271	  100.00%	  7168	     12.7981	  99.93%	     2.58%
   37	ffn_gate_shexp      	   1353.36	 0.1321	      1.0998	  0.1888	   0.0261	  100.00%	  7168	     12.7978	  99.93%	     2.41%
   36	ffn_gate_shexp      	   1302.77	 0.1082	      0.8941	  0.1817	   0.0231	  100.00%	  7168	     12.7989	  99.93%	     2.62%
   35	ffn_gate_shexp      	   1232.80	 0.0755	      0.8060	  0.1720	   0.0223	  100.00%	  7168	     12.7987	  99.93%	     2.16%
   34	ffn_gate_shexp      	   1204.46	 0.0729	      0.7595	  0.1680	   0.0216	  100.00%	  7168	     12.7989	  99.93%	     2.33%
   33	ffn_gate_shexp      	   1143.78	 0.0709	      0.9042	  0.1596	   0.0228	  100.00%	  7168	     12.7977	  99.92%	     1.93%
   32	ffn_gate_shexp      	   1099.52	 0.0818	      0.8105	  0.1534	   0.0226	  100.00%	  7168	     12.7968	  99.92%	     1.84%
   60	ffn_gate_shexp      	   1078.51	 0.0001	     20.6208	  0.1505	   0.2457	  100.00%	  7168	     12.6422	  98.71%	     0.10%
   31	ffn_gate_shexp      	   1029.70	 0.0938	      0.8485	  0.1437	   0.0226	  100.00%	  7168	     12.7959	  99.91%	     1.67%
   30	ffn_gate_shexp      	    948.08	 0.0994	      0.8589	  0.1323	   0.0224	  100.00%	  7168	     12.7944	  99.90%	     1.59%
   29	ffn_gate_shexp      	    923.32	 0.1143	      0.7502	  0.1288	   0.0208	  100.00%	  7168	     12.7952	  99.91%	     1.55%
   28	ffn_gate_shexp      	    857.50	 0.1050	      0.8266	  0.1196	   0.0197	  100.00%	  7168	     12.7951	  99.90%	     1.55%
   27	ffn_gate_shexp      	    795.67	 0.0908	      0.7870	  0.1110	   0.0177	  100.00%	  7168	     12.7962	  99.91%	     1.46%
   26	ffn_gate_shexp      	    736.90	 0.0784	      0.7393	  0.1028	   0.0169	  100.00%	  7168	     12.7955	  99.91%	     1.46%
   25	ffn_gate_shexp      	    667.83	 0.0700	      0.8148	  0.0932	   0.0164	  100.00%	  7168	     12.7947	  99.90%	     1.33%
   24	ffn_gate_shexp      	    619.78	 0.0657	      0.8708	  0.0865	   0.0164	  100.00%	  7168	     12.7936	  99.89%	     1.20%
   23	ffn_gate_shexp      	    550.91	 0.0638	      0.9747	  0.0769	   0.0176	  100.00%	  7168	     12.7898	  99.86%	     1.13%
   22	ffn_gate_shexp      	    473.30	 0.0550	      0.7791	  0.0660	   0.0160	  100.00%	  7168	     12.7880	  99.85%	     1.12%
   21	ffn_gate_shexp      	    425.76	 0.0463	      0.6638	  0.0594	   0.0159	  100.00%	  7168	     12.7845	  99.82%	     0.98%
   20	ffn_gate_shexp      	    373.53	 0.0377	      0.5380	  0.0521	   0.0109	  100.00%	  7168	     12.7912	  99.87%	     1.19%
   19	ffn_gate_shexp      	    327.81	 0.0331	      0.5958	  0.0457	   0.0110	  100.00%	  7168	     12.7872	  99.84%	     1.09%
   18	ffn_gate_shexp      	    288.33	 0.0259	      0.5437	  0.0402	   0.0093	  100.00%	  7168	     12.7885	  99.85%	     1.13%
   17	ffn_gate_shexp      	    262.71	 0.0221	      0.6237	  0.0367	   0.0089	  100.00%	  7168	     12.7898	  99.86%	     1.05%
   16	ffn_gate_shexp      	    239.14	 0.0150	      0.3143	  0.0334	   0.0052	  100.00%	  7168	     12.7968	  99.92%	     1.73%
   15	ffn_gate_shexp      	    227.29	 0.0155	      0.4654	  0.0317	   0.0064	  100.00%	  7168	     12.7940	  99.90%	     1.12%
   14	ffn_gate_shexp      	    208.76	 0.0130	      0.3669	  0.0291	   0.0049	  100.00%	  7168	     12.7971	  99.92%	     1.65%
   13	ffn_gate_shexp      	    195.33	 0.0077	      0.3455	  0.0272	   0.0046	  100.00%	  7168	     12.7965	  99.92%	     1.69%
   12	ffn_gate_shexp      	    179.43	 0.0035	      0.3448	  0.0250	   0.0047	  100.00%	  7168	     12.7938	  99.89%	     1.66%
   11	ffn_gate_shexp      	    153.69	 0.0014	      0.3143	  0.0214	   0.0045	  100.00%	  7168	     12.7874	  99.84%	     2.26%
   10	ffn_gate_shexp      	    136.43	 0.0012	      0.4756	  0.0190	   0.0062	  100.00%	  7168	     12.7744	  99.74%	     0.95%
    9	ffn_gate_shexp      	    116.60	 0.0016	      0.9678	  0.0163	   0.0121	  100.00%	  7168	     12.7233	  99.34%	     0.28%
    8	ffn_gate_shexp      	    105.89	 0.0009	      0.9859	  0.0148	   0.0127	  100.00%	  7168	     12.6870	  99.06%	     0.27%
    3	ffn_gate_shexp      	     95.53	 0.0000	     44.2083	  0.0133	   0.5280	   99.97%	  7168	      6.9930	  54.60%	     0.04%
    7	ffn_gate_shexp      	     85.46	 0.0005	      2.0256	  0.0119	   0.0266	  100.00%	  7168	     12.4812	  97.45%	     0.08%
    6	ffn_gate_shexp      	     83.44	 0.0001	      4.7202	  0.0116	   0.0623	  100.00%	  7168	     12.0480	  94.07%	     0.07%
    4	ffn_gate_shexp      	     73.80	 0.0000	     23.8029	  0.0103	   0.2955	   99.99%	  7168	      8.2841	  64.68%	     0.04%
    5	ffn_gate_shexp      	     73.60	 0.0000	     11.3983	  0.0103	   0.1479	  100.00%	  7168	     10.5195	  82.14%	     0.04%
    2	ffn_up              	    859.43	 0.0000	    802.1978	  0.1199	   9.4779	   99.83%	  7168	      0.6756	   5.27%	     0.03%
    1	ffn_up              	    592.96	 0.0000	    429.3697	  0.0827	   5.0879	   99.89%	  7168	      2.4691	  19.28%	     0.13%
    0	ffn_up              	    483.51	 0.0000	    450.5507	  0.0675	   5.3236	   97.56%	  7168	      0.6201	   4.84%	     0.06%
   57	ffn_up_exps         	1108622.00	 0.0574	     18.0424	  0.6042	   0.1643	  100.00%	1835008	     20.7916	  99.92%	     1.51%
   56	ffn_up_exps         	1098842.75	 0.1342	     21.3571	  0.5988	   0.1600	  100.00%	1835008	     20.7988	  99.96%	     1.60%
   58	ffn_up_exps         	1059858.50	 0.0017	     20.6275	  0.5776	   0.1614	  100.00%	1835008	     20.7922	  99.93%	     1.73%
   55	ffn_up_exps         	1029864.69	 0.1899	     24.0345	  0.5612	   0.1825	  100.00%	1835008	     20.7925	  99.93%	     1.18%
   54	ffn_up_exps         	 950597.38	 0.2668	     28.8253	  0.5180	   0.1960	  100.00%	1835008	     20.7858	  99.90%	     0.96%
   53	ffn_up_exps         	 919925.69	 0.2293	     31.0064	  0.5013	   0.1928	  100.00%	1835008	     20.7866	  99.90%	     0.93%
   52	ffn_up_exps         	 839725.12	 0.1856	     23.6457	  0.4576	   0.1782	  100.00%	1835008	     20.7856	  99.90%	     0.85%
   59	ffn_up_exps         	 788085.31	 0.0001	     32.1861	  0.4295	   0.1922	  100.00%	1835008	     20.7695	  99.82%	     0.71%
   51	ffn_up_exps         	 783379.31	 0.1706	     24.2819	  0.4269	   0.1622	  100.00%	1835008	     20.7859	  99.90%	     0.88%
   50	ffn_up_exps         	 749826.50	 0.1400	     21.6678	  0.4086	   0.1490	  100.00%	1835008	     20.7899	  99.92%	     0.90%
   49	ffn_up_exps         	 712692.44	 0.1545	     23.0501	  0.3884	   0.1351	  100.00%	1835008	     20.7872	  99.90%	     1.04%
   48	ffn_up_exps         	 652600.50	 0.1266	     17.2781	  0.3556	   0.1236	  100.00%	1835008	     20.7942	  99.94%	     1.03%
   47	ffn_up_exps         	 624720.88	 0.1098	     30.8410	  0.3404	   0.1301	  100.00%	1835008	     20.8078	 100.00%	     0.78%
   46	ffn_up_exps         	 583974.00	 0.1477	     26.1010	  0.3182	   0.1009	  100.00%	1835008	     20.7921	  99.93%	     1.10%
   45	ffn_up_exps         	 547631.69	 0.1284	     14.7849	  0.2984	   0.0870	  100.00%	1835008	     20.7918	  99.93%	     1.44%
   44	ffn_up_exps         	 517168.44	 0.1231	     22.0782	  0.2818	   0.0875	  100.00%	1835008	     20.8003	  99.97%	     1.32%
   43	ffn_up_exps         	 486536.84	 0.1024	     32.9791	  0.2651	   0.0996	  100.00%	1835008	     20.8003	  99.97%	     0.81%
   42	ffn_up_exps         	 459638.69	 0.1057	     18.4986	  0.2505	   0.0764	  100.00%	1835008	     20.7969	  99.95%	     1.27%
   41	ffn_up_exps         	 435830.34	 0.0979	     14.5584	  0.2375	   0.0705	  100.00%	1835008	     20.7998	  99.96%	     1.46%
   40	ffn_up_exps         	 417437.19	 0.1014	     11.7959	  0.2275	   0.0697	  100.00%	1835008	     20.7969	  99.95%	     1.38%
   39	ffn_up_exps         	 399054.31	 0.1064	     19.4026	  0.2175	   0.0743	  100.00%	1835008	     20.7920	  99.93%	     1.13%
   38	ffn_up_exps         	 368285.38	 0.0749	     15.0838	  0.2007	   0.0680	  100.00%	1835008	     20.8033	  99.98%	     1.26%
   37	ffn_up_exps         	 346157.62	 0.0642	      8.4320	  0.1886	   0.0567	  100.00%	1835008	     20.7879	  99.91%	     1.51%
   36	ffn_up_exps         	 333243.12	 0.0730	     11.6749	  0.1816	   0.0538	  100.00%	1835008	     20.7971	  99.95%	     1.51%
   35	ffn_up_exps         	 315236.34	 0.0432	     16.8776	  0.1718	   0.0634	  100.00%	1835008	     20.8073	 100.00%	     0.98%
   34	ffn_up_exps         	 308240.75	 0.0462	     11.0697	  0.1680	   0.0521	  100.00%	1835008	     20.8190	 100.06%	     1.21%
   33	ffn_up_exps         	 292961.50	 0.0501	     17.4166	  0.1597	   0.0579	  100.00%	1835008	     20.8051	  99.99%	     0.82%
   32	ffn_up_exps         	 281822.19	 0.0545	     16.2088	  0.1536	   0.0615	  100.00%	1835008	     20.7920	  99.93%	     0.77%
   60	ffn_up_exps         	 275449.28	 0.0000	     53.8235	  0.1501	   0.2789	  100.00%	1835008	     20.6214	  99.11%	     0.09%
   31	ffn_up_exps         	 264012.66	 0.0627	     23.6177	  0.1439	   0.0607	  100.00%	1835008	     20.8012	  99.97%	     0.73%
   30	ffn_up_exps         	 242871.81	 0.0746	     11.3317	  0.1324	   0.0526	  100.00%	1835008	     20.7986	  99.96%	     0.83%
   29	ffn_up_exps         	 236621.69	 0.0708	     12.5480	  0.1289	   0.0505	  100.00%	1835008	     20.7994	  99.96%	     0.84%
   28	ffn_up_exps         	 219571.83	 0.0656	     16.1806	  0.1197	   0.0603	  100.00%	1835008	     20.7942	  99.94%	     0.61%
   27	ffn_up_exps         	 203887.56	 0.0648	     16.1550	  0.1111	   0.0594	  100.00%	1835008	     20.7817	  99.88%	     0.50%
   26	ffn_up_exps         	 188690.89	 0.0456	      9.7137	  0.1028	   0.0436	  100.00%	1835008	     20.8035	  99.98%	     0.69%
   25	ffn_up_exps         	 171281.08	 0.0441	      9.9973	  0.0933	   0.0420	  100.00%	1835008	     20.7806	  99.87%	     0.64%
   24	ffn_up_exps         	 158806.77	 0.0401	      7.9296	  0.0865	   0.0405	  100.00%	1835008	     20.7953	  99.94%	     0.60%
   23	ffn_up_exps         	 140877.31	 0.0399	      4.9228	  0.0768	   0.0279	  100.00%	1835008	     20.7861	  99.90%	     0.90%
   22	ffn_up_exps         	 121295.08	 0.0384	      3.9828	  0.0661	   0.0227	  100.00%	1835008	     20.7894	  99.91%	     1.04%
   21	ffn_up_exps         	 109139.78	 0.0260	     16.0739	  0.0595	   0.0452	  100.00%	1835008	     20.7649	  99.80%	     0.40%
   20	ffn_up_exps         	  95741.52	 0.0227	      6.8249	  0.0522	   0.0226	  100.00%	1835008	     20.7793	  99.86%	     0.66%
   19	ffn_up_exps         	  83921.45	 0.0200	      2.8252	  0.0457	   0.0179	  100.00%	1835008	     20.7710	  99.83%	     0.95%
   18	ffn_up_exps         	  74025.85	 0.0140	      2.6935	  0.0403	   0.0158	  100.00%	1835008	     20.7662	  99.80%	     0.99%
   17	ffn_up_exps         	  67284.16	 0.0135	      2.3618	  0.0367	   0.0147	  100.00%	1835008	     20.7702	  99.82%	     0.81%
   16	ffn_up_exps         	  61220.83	 0.0103	      1.9943	  0.0334	   0.0104	  100.00%	1835008	     20.7856	  99.90%	     1.41%
   15	ffn_up_exps         	  58135.96	 0.0112	      3.1859	  0.0317	   0.0112	  100.00%	1835008	     20.7830	  99.88%	     0.99%
   14	ffn_up_exps         	  53397.41	 0.0089	      1.1326	  0.0291	   0.0071	  100.00%	1835008	     20.7873	  99.90%	     3.18%
   13	ffn_up_exps         	  49976.98	 0.0044	      1.7784	  0.0272	   0.0076	  100.00%	1835008	     20.7836	  99.89%	     2.73%
   12	ffn_up_exps         	  45768.75	 0.0021	      3.0780	  0.0249	   0.0089	  100.00%	1835008	     20.7758	  99.85%	     1.62%
   11	ffn_up_exps         	  39124.46	 0.0006	      1.5074	  0.0213	   0.0065	  100.00%	1835008	     20.7666	  99.80%	     4.91%
   10	ffn_up_exps         	  34817.07	 0.0007	      1.1131	  0.0190	   0.0075	  100.00%	1835008	     20.7560	  99.75%	     2.79%
    9	ffn_up_exps         	  29854.04	 0.0009	      1.6395	  0.0163	   0.0126	  100.00%	1835008	     20.7027	  99.50%	     0.58%
    8	ffn_up_exps         	  26950.78	 0.0006	      1.6468	  0.0147	   0.0131	  100.00%	1835008	     20.6652	  99.32%	     0.52%
    3	ffn_up_exps         	  24427.59	 0.0000	     66.7534	  0.0133	   0.5410	   99.98%	1835008	     14.9015	  71.62%	     0.04%
    7	ffn_up_exps         	  21764.22	 0.0001	      3.2781	  0.0119	   0.0272	  100.00%	1835008	     20.4541	  98.30%	     0.11%
    6	ffn_up_exps         	  21277.98	 0.0001	      6.6201	  0.0116	   0.0631	  100.00%	1835008	     20.0195	  96.21%	     0.07%
    4	ffn_up_exps         	  18856.03	 0.0000	     38.3090	  0.0103	   0.3010	   99.98%	1835008	     16.2252	  77.98%	     0.04%
    5	ffn_up_exps         	  18769.08	 0.0000	     16.2609	  0.0102	   0.1502	  100.00%	1835008	     18.4726	  88.78%	     0.04%
   57	ffn_up_shexp        	   4342.22	 0.1044	      7.2990	  0.6058	   0.1245	  100.00%	  7168	     12.7942	  99.90%	     0.84%
   56	ffn_up_shexp        	   4303.31	 0.1893	      7.3898	  0.6003	   0.1111	  100.00%	  7168	     12.7964	  99.91%	     1.38%
   58	ffn_up_shexp        	   4154.51	 0.0036	      9.2729	  0.5796	   0.1254	  100.00%	  7168	     12.7927	  99.89%	     0.78%
   55	ffn_up_shexp        	   4032.60	 0.3283	      9.3460	  0.5626	   0.1289	  100.00%	  7168	     12.7932	  99.89%	     1.23%
   54	ffn_up_shexp        	   3724.53	 0.3516	     10.7018	  0.5196	   0.1388	  100.00%	  7168	     12.7904	  99.87%	     1.12%
   53	ffn_up_shexp        	   3604.73	 0.3538	     11.3448	  0.5029	   0.1447	  100.00%	  7168	     12.7888	  99.86%	     1.09%
   52	ffn_up_shexp        	   3288.52	 0.3025	     10.1119	  0.4588	   0.1298	  100.00%	  7168	     12.7889	  99.86%	     1.06%
   59	ffn_up_shexp        	   3083.86	 0.0004	     13.7678	  0.4302	   0.1691	  100.00%	  7168	     12.7747	  99.74%	     0.25%
   51	ffn_up_shexp        	   3067.81	 0.2711	      8.4771	  0.4280	   0.1118	  100.00%	  7168	     12.7901	  99.87%	     1.05%
   50	ffn_up_shexp        	   2942.98	 0.2604	      7.4818	  0.4106	   0.1014	  100.00%	  7168	     12.7908	  99.87%	     1.05%
   49	ffn_up_shexp        	   2792.88	 0.2567	      5.7642	  0.3896	   0.0829	  100.00%	  7168	     12.7930	  99.89%	     1.30%
   48	ffn_up_shexp        	   2556.00	 0.2407	      4.8123	  0.3566	   0.0719	  100.00%	  7168	     12.7935	  99.89%	     1.40%
   47	ffn_up_shexp        	   2446.98	 0.2099	      3.5467	  0.3414	   0.0594	  100.00%	  7168	     12.7955	  99.91%	     1.67%
   46	ffn_up_shexp        	   2285.12	 0.2029	      2.6480	  0.3188	   0.0502	  100.00%	  7168	     12.7966	  99.92%	     1.80%
   45	ffn_up_shexp        	   2143.51	 0.2553	      2.0089	  0.2990	   0.0423	  100.00%	  7168	     12.7978	  99.93%	     2.30%
   44	ffn_up_shexp        	   2024.29	 0.2251	      1.8251	  0.2824	   0.0393	  100.00%	  7168	     12.7981	  99.93%	     2.59%
   43	ffn_up_shexp        	   1905.67	 0.1806	      1.5305	  0.2659	   0.0352	  100.00%	  7168	     12.7988	  99.93%	     2.37%
   42	ffn_up_shexp        	   1798.81	 0.2058	      1.4089	  0.2510	   0.0331	  100.00%	  7168	     12.7987	  99.93%	     2.50%
   41	ffn_up_shexp        	   1705.82	 0.1887	      1.5552	  0.2380	   0.0335	  100.00%	  7168	     12.7978	  99.93%	     2.37%
   40	ffn_up_shexp        	   1633.65	 0.1743	      1.4432	  0.2279	   0.0323	  100.00%	  7168	     12.7977	  99.92%	     2.32%
   39	ffn_up_shexp        	   1560.66	 0.1826	      1.3440	  0.2177	   0.0293	  100.00%	  7168	     12.7983	  99.93%	     2.58%
   38	ffn_up_shexp        	   1440.72	 0.1637	      1.1312	  0.2010	   0.0271	  100.00%	  7168	     12.7981	  99.93%	     2.58%
   37	ffn_up_shexp        	   1353.36	 0.1321	      1.0998	  0.1888	   0.0261	  100.00%	  7168	     12.7978	  99.93%	     2.41%
   36	ffn_up_shexp        	   1302.77	 0.1082	      0.8941	  0.1817	   0.0231	  100.00%	  7168	     12.7989	  99.93%	     2.62%
   35	ffn_up_shexp        	   1232.80	 0.0755	      0.8060	  0.1720	   0.0223	  100.00%	  7168	     12.7987	  99.93%	     2.16%
   34	ffn_up_shexp        	   1204.46	 0.0729	      0.7595	  0.1680	   0.0216	  100.00%	  7168	     12.7989	  99.93%	     2.33%
   33	ffn_up_shexp        	   1143.78	 0.0709	      0.9042	  0.1596	   0.0228	  100.00%	  7168	     12.7977	  99.92%	     1.93%
   32	ffn_up_shexp        	   1099.52	 0.0818	      0.8105	  0.1534	   0.0226	  100.00%	  7168	     12.7968	  99.92%	     1.84%
   60	ffn_up_shexp        	   1078.51	 0.0001	     20.6208	  0.1505	   0.2457	  100.00%	  7168	     12.6422	  98.71%	     0.10%
   31	ffn_up_shexp        	   1029.70	 0.0938	      0.8485	  0.1437	   0.0226	  100.00%	  7168	     12.7959	  99.91%	     1.67%
   30	ffn_up_shexp        	    948.08	 0.0994	      0.8589	  0.1323	   0.0224	  100.00%	  7168	     12.7944	  99.90%	     1.59%
   29	ffn_up_shexp        	    923.32	 0.1143	      0.7502	  0.1288	   0.0208	  100.00%	  7168	     12.7952	  99.91%	     1.55%
   28	ffn_up_shexp        	    857.50	 0.1050	      0.8266	  0.1196	   0.0197	  100.00%	  7168	     12.7951	  99.90%	     1.55%
   27	ffn_up_shexp        	    795.67	 0.0908	      0.7870	  0.1110	   0.0177	  100.00%	  7168	     12.7962	  99.91%	     1.46%
   26	ffn_up_shexp        	    736.90	 0.0784	      0.7393	  0.1028	   0.0169	  100.00%	  7168	     12.7955	  99.91%	     1.46%
   25	ffn_up_shexp        	    667.83	 0.0700	      0.8148	  0.0932	   0.0164	  100.00%	  7168	     12.7947	  99.90%	     1.33%
   24	ffn_up_shexp        	    619.78	 0.0657	      0.8708	  0.0865	   0.0164	  100.00%	  7168	     12.7936	  99.89%	     1.20%
   23	ffn_up_shexp        	    550.91	 0.0638	      0.9747	  0.0769	   0.0176	  100.00%	  7168	     12.7898	  99.86%	     1.13%
   22	ffn_up_shexp        	    473.30	 0.0550	      0.7791	  0.0660	   0.0160	  100.00%	  7168	     12.7880	  99.85%	     1.12%
   21	ffn_up_shexp        	    425.76	 0.0463	      0.6638	  0.0594	   0.0159	  100.00%	  7168	     12.7845	  99.82%	     0.98%
   20	ffn_up_shexp        	    373.53	 0.0377	      0.5380	  0.0521	   0.0109	  100.00%	  7168	     12.7912	  99.87%	     1.19%
   19	ffn_up_shexp        	    327.81	 0.0331	      0.5958	  0.0457	   0.0110	  100.00%	  7168	     12.7872	  99.84%	     1.09%
   18	ffn_up_shexp        	    288.33	 0.0259	      0.5437	  0.0402	   0.0093	  100.00%	  7168	     12.7885	  99.85%	     1.13%
   17	ffn_up_shexp        	    262.71	 0.0221	      0.6237	  0.0367	   0.0089	  100.00%	  7168	     12.7898	  99.86%	     1.05%
   16	ffn_up_shexp        	    239.14	 0.0150	      0.3143	  0.0334	   0.0052	  100.00%	  7168	     12.7968	  99.92%	     1.73%
   15	ffn_up_shexp        	    227.29	 0.0155	      0.4654	  0.0317	   0.0064	  100.00%	  7168	     12.7940	  99.90%	     1.12%
   14	ffn_up_shexp        	    208.76	 0.0130	      0.3669	  0.0291	   0.0049	  100.00%	  7168	     12.7971	  99.92%	     1.65%
   13	ffn_up_shexp        	    195.33	 0.0077	      0.3455	  0.0272	   0.0046	  100.00%	  7168	     12.7965	  99.92%	     1.69%
   12	ffn_up_shexp        	    179.43	 0.0035	      0.3448	  0.0250	   0.0047	  100.00%	  7168	     12.7938	  99.89%	     1.66%
   11	ffn_up_shexp        	    153.69	 0.0014	      0.3143	  0.0214	   0.0045	  100.00%	  7168	     12.7874	  99.84%	     2.26%
   10	ffn_up_shexp        	    136.43	 0.0012	      0.4756	  0.0190	   0.0062	  100.00%	  7168	     12.7744	  99.74%	     0.95%
    9	ffn_up_shexp        	    116.60	 0.0016	      0.9678	  0.0163	   0.0121	  100.00%	  7168	     12.7233	  99.34%	     0.28%
    8	ffn_up_shexp        	    105.89	 0.0009	      0.9859	  0.0148	   0.0127	  100.00%	  7168	     12.6870	  99.06%	     0.27%
    3	ffn_up_shexp        	     95.53	 0.0000	     44.2083	  0.0133	   0.5280	   99.97%	  7168	      6.9930	  54.60%	     0.04%
    7	ffn_up_shexp        	     85.46	 0.0005	      2.0256	  0.0119	   0.0266	  100.00%	  7168	     12.4812	  97.45%	     0.08%
    6	ffn_up_shexp        	     83.44	 0.0001	      4.7202	  0.0116	   0.0623	  100.00%	  7168	     12.0480	  94.07%	     0.07%
    4	ffn_up_shexp        	     73.80	 0.0000	     23.8029	  0.0103	   0.2955	   99.99%	  7168	      8.2841	  64.68%	     0.04%
    5	ffn_up_shexp        	     73.60	 0.0000	     11.3983	  0.0103	   0.1479	  100.00%	  7168	     10.5195	  82.14%	     0.04%
```

</details>