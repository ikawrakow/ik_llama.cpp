### 🔀 [#328](https://github.com/ikawrakow/ik_llama.cpp/pull/328) - imatrix: collect layer influence statistics

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-04-14 |
| **Updated** | 2025-04-14 |

---

#### Description

@ubergarm

Here is how one can collect statistics about the activations change caused by a layer using cosine similarity.

---

#### 💬 Conversation

👤 **ubergarm** commented the **2025-04-14** at **14:39:20**:<br>

Holy smokes, amazing! I'm out for a couple nights, but going to pull this and try quick before leaving the house haha... Thanks!

---

👤 **ikawrakow** commented the **2025-04-14** at **16:02:02**:<br>

Does the last commit fix it? I had forgotten about having to strip the tensor name (and for whatever reason I didn't have the issue even though running on CUDA).

---

👤 **ubergarm** commented the **2025-04-14** at **16:10:14**:<br>

Yep, that did the trick! Thanks! I have a chart I just graphed, will put it here with logs before heading out.

---

👤 **ikawrakow** commented the **2025-04-14** at **16:13:51**:<br>

Using this on LLaMA-4-Scout, I get this as the layers sorted by importance (most important first):
```
======================== sorted layer importances
  0: Layer   0, <cos_sim> = 0.147234
  1: Layer   2, <cos_sim> = 0.338908
  2: Layer  47, <cos_sim> = 0.413196
  3: Layer   1, <cos_sim> = 0.626674
  4: Layer   7, <cos_sim> = 0.835974
  5: Layer   6, <cos_sim> = 0.841949
  6: Layer   4, <cos_sim> = 0.844908
  7: Layer   3, <cos_sim> = 0.849444
  8: Layer  10, <cos_sim> = 0.869448
  9: Layer  34, <cos_sim> = 0.875514
 10: Layer  22, <cos_sim> = 0.880165
 11: Layer  46, <cos_sim> = 0.881091
 12: Layer  11, <cos_sim> = 0.887115
 13: Layer  31, <cos_sim> = 0.889579
 14: Layer  35, <cos_sim> = 0.893048
 15: Layer  26, <cos_sim> = 0.897382
 16: Layer  18, <cos_sim> = 0.898017
 17: Layer  23, <cos_sim> = 0.898672
 18: Layer  21, <cos_sim> = 0.900372
 19: Layer  14, <cos_sim> = 0.902133
 20: Layer  43, <cos_sim> = 0.908545
 21: Layer  44, <cos_sim> = 0.908824
 22: Layer  38, <cos_sim> = 0.909535
 23: Layer  45, <cos_sim> = 0.909808
 24: Layer  19, <cos_sim> = 0.911718
 25: Layer   8, <cos_sim> = 0.911922
 26: Layer  30, <cos_sim> = 0.913816
 27: Layer  13, <cos_sim> = 0.916391
 28: Layer  39, <cos_sim> = 0.917897
 29: Layer  25, <cos_sim> = 0.917991
 30: Layer  24, <cos_sim> = 0.918002
 31: Layer  27, <cos_sim> = 0.918821
 32: Layer   5, <cos_sim> = 0.920709
 33: Layer  15, <cos_sim> = 0.921429
 34: Layer   9, <cos_sim> = 0.922202
 35: Layer  29, <cos_sim> = 0.923448
 36: Layer  16, <cos_sim> = 0.924396
 37: Layer  17, <cos_sim> = 0.925231
 38: Layer  42, <cos_sim> = 0.925237
 39: Layer  12, <cos_sim> = 0.926379
 40: Layer  37, <cos_sim> = 0.926797
 41: Layer  20, <cos_sim> = 0.92796
 42: Layer  28, <cos_sim> = 0.933169
 43: Layer  36, <cos_sim> = 0.936506
 44: Layer  32, <cos_sim> = 0.936671
 45: Layer  41, <cos_sim> = 0.939215
 46: Layer  33, <cos_sim> = 0.940524
 47: Layer  40, <cos_sim> = 0.948523
```

I had a pretty good L4-Scout recipe for `IQ2_K`
```
./bin/llama-quantize --imatrix l4_scout_imat_512.out --custom-q "ffn_gate_shexp=iq4_ks,ffn_up_shexp=iq4_ks,ffn_down_shexp=iq5_k,attn=iq4_ks,token_embd.weight=q4_K,output.weight=q6_K,blk\.[0-5]\.ffn_down_exps=iq4_ks,ffn_down_exps=iq3_k,ffn_up_exps=iq2_k,ffn_gate_exps=iq2_k" ../../iquants/models/l4_109B/Llama4-Scout-16x17B-BF16.gguf junk1.bin iq2_k
```

It arrived at a `PPL = 9.7545`, so nearly on par with Unsloth's `UD-Q2_K_XL`, despite being 2.6 GB smaller. The recipe uses `IQ4_KS` for the first 6 layers of `ffn_down_exps`. If instead I use layers `0,1,2,4,6,7`, PPL becomes `9.7066`, so we do get a small improvement from that (but using layer 47 instead of layer 4, which according to the metric would be the right thing to do, results in a worse outcome)

---

👤 **ubergarm** commented the **2025-04-14** at **16:28:24**:<br>

> (but using layer 47 instead of layer 4, which according to the metric would be the right thing to do, results in a worse outcome)

Very interesting. Yeah, I'm curious how much the input text for imatrix effects these cosine similarities as well. 

I did a quick run with `llama-2-13b-chat.Q8_0.gguf` and plotted the results to compare against that [Layer-wise Quantization](https://arxiv.org/pdf/2406.17415) paper which suggests for this model the three most important layers would be 1, 2, and 40 while the least important would be 32, 33, and 34. Though I'm not sure how they got that final layer 40 cosine similarity.

<details>

<summary>Results Graph and Log of modified llama-imatrix -lsim</summary>

![lsim](https://github.com/user-attachments/assets/757081ca-8251-4405-a0df-3b4299caef54)

```bash
$ git branch | grep '*'
* ik/imatrix_lsim

$ git rev-parse --short HEAD
8bff04c9

$ ./build/bin/llama-imatrix --version
version: 3638 (8bff04c9)
built with cc (GCC) 14.2.1 20250128 for x86_64-pc-linux-gnu

$ ./build/bin/llama-imatrix \
    --verbosity 1 \
    -m /mnt/astrodata/llm/models/TheBloke/Llama-2-13B-chat-GGUF/llama-2-13b-chat.Q8_0.gguf \
    -f calibration_data_v5_rc.txt \
    -o imatrix-calibration_data_v5_rc-llama-2-13b-chat.dat \
    --layer-similarity \
    --output-tensor-name ffn_down.weight \
    --ctx-size 512 \
    --threads 16

llama_model_loader: loaded meta data with 19 key-value pairs and 363 tensors from /mnt/astrodata/llm/models/TheBloke/Llama-2-13B-chat-GGUF/llama-2-13b-chat.Q8_0.gguf (version GGUF V2)
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.name str              = LLaMA v2
llama_model_loader: - kv   2:                       llama.context_length u32              = 4096
llama_model_loader: - kv   3:                     llama.embedding_length u32              = 5120
llama_model_loader: - kv   4:                          llama.block_count u32              = 40
llama_model_loader: - kv   5:                  llama.feed_forward_length u32              = 13824
llama_model_loader: - kv   6:                 llama.rope.dimension_count u32              = 128
llama_model_loader: - kv   7:                 llama.attention.head_count u32              = 40
llama_model_loader: - kv   8:              llama.attention.head_count_kv u32              = 40
llama_model_loader: - kv   9:     llama.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  10:                          general.file_type u32              = 7
llama_model_loader: - kv  11:                       tokenizer.ggml.model str              = llama
llama_model_loader: - kv  12:                      tokenizer.ggml.tokens arr[str,32000]   = [
llama_model_loader: - kv  13:                      tokenizer.ggml.scores arr[f32,32000]   = [
llama_model_loader: - kv  14:                  tokenizer.ggml.token_type arr[i32,32000]   = [
llama_model_loader: - kv  15:                tokenizer.ggml.bos_token_id u32              = 1
llama_model_loader: - kv  16:                tokenizer.ggml.eos_token_id u32              = 2
llama_model_loader: - kv  17:            tokenizer.ggml.unknown_token_id u32              = 0
llama_model_loader: - kv  18:               general.quantization_version u32              = 2
llama_model_loader: - type  f32:   81 tensors
llama_model_loader: - type q8_0:  282 tensors
llm_load_vocab: special tokens cache size = 3
llm_load_vocab: token to piece cache size = 0.1684 MB
llm_load_print_meta: format           = GGUF V2
llm_load_print_meta: arch             = llama
llm_load_print_meta: vocab type       = SPM
llm_load_print_meta: n_vocab          = 32000
llm_load_print_meta: n_merges         = 0
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 4096
llm_load_print_meta: n_embd           = 5120
llm_load_print_meta: n_layer          = 40
llm_load_print_meta: n_head           = 40
llm_load_print_meta: n_head_kv        = 40
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: n_embd_k_gqa     = 5120
llm_load_print_meta: n_embd_v_gqa     = 5120
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 13824
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 4096
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = 13B
llm_load_print_meta: model ftype      = Q8_0
llm_load_print_meta: model params     = 13.016 B
llm_load_print_meta: model size       = 12.881 GiB (8.501 BPW)
llm_load_print_meta: repeating layers = 12.556 GiB (8.501 BPW, 12.688 B parameters)
llm_load_print_meta: general.name     = LLaMA v2
llm_load_print_meta: BOS token        = 1 '<s>'
llm_load_print_meta: EOS token        = 2 '</s>'
llm_load_print_meta: UNK token        = 0 '<unk>'
llm_load_print_meta: LF token         = 13 '<0x0A>'
llm_load_print_meta: max token length = 48
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090 Ti, compute capability 8.6, VMM: yes
llm_load_tensors: ggml ctx size =    0.17 MiB
llm_load_tensors: offloading 0 repeating layers to GPU
llm_load_tensors: offloaded 0/41 layers to GPU
llm_load_tensors:        CPU buffer size = 13189.86 MiB
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
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:  CUDA_Host KV buffer size =   400.00 MiB
llama_new_context_with_model: KV self size  =  400.00 MiB, K (f16):  200.00 MiB, V (f16):  200.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.12 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =   248.54 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =    21.01 MiB
llama_new_context_with_model: graph nodes  = 1165
llama_new_context_with_model: graph splits = 443

system_info: n_threads = 16 / 32 | AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 |
compute_imatrix: tokenizing the input ..
compute_imatrix: tokenization took 102.865 ms
compute_imatrix: computing over 277 chunks with batch_size 512
compute_imatrix: 1.61 seconds per pass - ETA 7.40 minutes
[1]8.4429,[2]9.0054,[3]6.0236,[4]5.1203,[5]5.4399,[6]4.1193,[7]3.4893,[8]3.0374,[9]2.7789,
save_imatrix: stored collected data after 10 chunks in imatrix-calibration_data_v5_rc-llama-2-13b-chat.dat
[10]2.5790,[11]2.4134,[12]2.2756,[13]2.2770,[14]2.1808,[15]2.3420,[16]2.5229,[17]2.6719,[18]2.6744,[19]2.7924,
save_imatrix: stored collected data after 20 chunks in imatrix-calibration_data_v5_rc-llama-2-13b-chat.dat
[20]2.8546,[21]2.8265,[22]2.8108,[23]2.8379,[24]2.8256,[25]2.8160,[26]2.7995,[27]2.8185,[28]2.8211,[29]2.7253,
save_imatrix: stored collected data after 30 chunks in imatrix-calibration_data_v5_rc-llama-2-13b-chat.dat
[30]2.7067,[31]2.7767,[32]2.8058,[33]2.8023,[34]2.8008,[35]2.8470,[36]2.9396,[37]2.9690,[38]3.0215,[39]3.0308,
save_imatrix: stored collected data after 40 chunks in imatrix-calibration_data_v5_rc-llama-2-13b-chat.dat
[40]3.0702,[41]3.1383,[42]3.1813,[43]3.2972,[44]3.3731,[45]3.3880,[46]3.3992,[47]3.4017,[48]3.4401,[49]3.4639,
save_imatrix: stored collected data after 50 chunks in imatrix-calibration_data_v5_rc-llama-2-13b-chat.dat
[50]3.5158,[51]3.5482,[52]3.5576,[53]3.5790,[54]3.6064,[55]3.6440,[56]3.6855,[57]3.6976,[58]3.7151,[59]3.7365,
save_imatrix: stored collected data after 60 chunks in imatrix-calibration_data_v5_rc-llama-2-13b-chat.dat
[60]3.7198,[61]3.7105,[62]3.6868,[63]3.6517,[64]3.6643,[65]3.6569,[66]3.6335,[67]3.6463,[68]3.6364,[69]3.6098,
save_imatrix: stored collected data after 70 chunks in imatrix-calibration_data_v5_rc-llama-2-13b-chat.dat
[70]3.5796,[71]3.5663,[72]3.5423,[73]3.5180,[74]3.4853,[75]3.4602,[76]3.4389,[77]3.4079,[78]3.4590,[79]3.4885,
save_imatrix: stored collected data after 80 chunks in imatrix-calibration_data_v5_rc-llama-2-13b-chat.dat
[80]3.5384,[81]3.5655,[82]3.5703,[83]3.6146,[84]3.6383,[85]3.6433,[86]3.6712,[87]3.6529,[88]3.6616,[89]3.6659,
save_imatrix: stored collected data after 90 chunks in imatrix-calibration_data_v5_rc-llama-2-13b-chat.dat
[90]3.6578,[91]3.6563,[92]3.7242,[93]3.7772,[94]3.8348,[95]3.8650,[96]3.9093,[97]3.9215,[98]3.9316,[99]3.9614,
save_imatrix: stored collected data after 100 chunks in imatrix-calibration_data_v5_rc-llama-2-13b-chat.dat
[100]3.9870,[101]3.9926,[102]4.0022,[103]4.0078,[104]4.0241,[105]4.0021,[106]4.0216,[107]4.0284,[108]4.0321,[109]4.0764,
save_imatrix: stored collected data after 110 chunks in imatrix-calibration_data_v5_rc-llama-2-13b-chat.dat
[110]4.1078,[111]4.1195,[112]4.1347,[113]4.1305,[114]4.1078,[115]4.1262,[116]4.1317,[117]4.1305,[118]4.1626,[119]4.1574,
save_imatrix: stored collected data after 120 chunks in imatrix-calibration_data_v5_rc-llama-2-13b-chat.dat
[120]4.1461,[121]4.1457,[122]4.1450,[123]4.1398,[124]4.1433,[125]4.1565,[126]4.1668,[127]4.1812,[128]4.1865,[129]4.1768,
save_imatrix: stored collected data after 130 chunks in imatrix-calibration_data_v5_rc-llama-2-13b-chat.dat
[130]4.1734,[131]4.2002,[132]4.2067,[133]4.2000,[134]4.1810,[135]4.2081,[136]4.2197,[137]4.2454,[138]4.2620,[139]4.2528,
save_imatrix: stored collected data after 140 chunks in imatrix-calibration_data_v5_rc-llama-2-13b-chat.dat
[140]4.2720,[141]4.2953,[142]4.3222,[143]4.3403,[144]4.3690,[145]4.3883,[146]4.4270,[147]4.4502,[148]4.4468,[149]4.4334,
save_imatrix: stored collected data after 150 chunks in imatrix-calibration_data_v5_rc-llama-2-13b-chat.dat
[150]4.4587,[151]4.4729,[152]4.4900,[153]4.4847,[154]4.5262,[155]4.5341,[156]4.5600,[157]4.5479,[158]4.5551,[159]4.5649,
save_imatrix: stored collected data after 160 chunks in imatrix-calibration_data_v5_rc-llama-2-13b-chat.dat
[160]4.5906,[161]4.5990,[162]4.6071,[163]4.5763,[164]4.5561,[165]4.5295,[166]4.5200,[167]4.5107,[168]4.5148,[169]4.5286,
save_imatrix: stored collected data after 170 chunks in imatrix-calibration_data_v5_rc-llama-2-13b-chat.dat
[170]4.5453,[171]4.5400,[172]4.5458,[173]4.5576,[174]4.5648,[175]4.5852,[176]4.6067,[177]4.6488,[178]4.6855,[179]4.7140,
save_imatrix: stored collected data after 180 chunks in imatrix-calibration_data_v5_rc-llama-2-13b-chat.dat
[180]4.7434,[181]4.7628,[182]4.7820,[183]4.7710,[184]4.7853,[185]4.8189,[186]4.8460,[187]4.8477,[188]4.8348,[189]4.8479,
save_imatrix: stored collected data after 190 chunks in imatrix-calibration_data_v5_rc-llama-2-13b-chat.dat
[190]4.8627,[191]4.8802,[192]4.9172,[193]4.9458,[194]4.9610,[195]4.9765,[196]4.9902,[197]5.0011,[198]4.9910,[199]4.9894,
save_imatrix: stored collected data after 200 chunks in imatrix-calibration_data_v5_rc-llama-2-13b-chat.dat
[200]4.9818,[201]4.9788,[202]4.9866,[203]4.9945,[204]4.9993,[205]5.0029,[206]5.0112,[207]5.0217,[208]5.0205,[209]5.0324,
save_imatrix: stored collected data after 210 chunks in imatrix-calibration_data_v5_rc-llama-2-13b-chat.dat
[210]5.0529,[211]5.0635,[212]5.0756,[213]5.0723,[214]5.0873,[215]5.0975,[216]5.1073,[217]5.1171,[218]5.1213,[219]5.1426,
save_imatrix: stored collected data after 220 chunks in imatrix-calibration_data_v5_rc-llama-2-13b-chat.dat
[220]5.1445,[221]5.1374,[222]5.1562,[223]5.1764,[224]5.1933,[225]5.1982,[226]5.2087,[227]5.2195,[228]5.2394,[229]5.2261,
save_imatrix: stored collected data after 230 chunks in imatrix-calibration_data_v5_rc-llama-2-13b-chat.dat
[230]5.2269,[231]5.2197,[232]5.2361,[233]5.2403,[234]5.2375,[235]5.2346,[236]5.2321,[237]5.2252,[238]5.2216,[239]5.2124,
save_imatrix: stored collected data after 240 chunks in imatrix-calibration_data_v5_rc-llama-2-13b-chat.dat
[240]5.2077,[241]5.2022,[242]5.1967,[243]5.1920,[244]5.1865,[245]5.1891,[246]5.1968,[247]5.2214,[248]5.2460,[249]5.2682,
save_imatrix: stored collected data after 250 chunks in imatrix-calibration_data_v5_rc-llama-2-13b-chat.dat
[250]5.2993,[251]5.3283,[252]5.3306,[253]5.3429,[254]5.3461,[255]5.3590,[256]5.3653,[257]5.3726,[258]5.3645,[259]5.3569,
save_imatrix: stored collected data after 260 chunks in imatrix-calibration_data_v5_rc-llama-2-13b-chat.dat
[260]5.3674,[261]5.3848,[262]5.3862,[263]5.3887,[264]5.3941,[265]5.4030,[266]5.4143,[267]5.4201,[268]5.4234,[269]5.4354,
save_imatrix: stored collected data after 270 chunks in imatrix-calibration_data_v5_rc-llama-2-13b-chat.dat
[270]5.4386,[271]5.4424,[272]5.4521,[273]5.4564,[274]5.4639,[275]5.4791,[276]5.4830,[277]5.4973,
save_imatrix: stored collected data after 277 chunks in imatrix-calibration_data_v5_rc-llama-2-13b-chat.dat

Final estimate: PPL = 5.4973 +/- 0.05449

llama_print_timings:        load time =    2147.62 ms
llama_print_timings:      sample time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings: prompt eval time =  424676.05 ms / 141824 tokens (    2.99 ms per token,   333.96 tokens per second)
llama_print_timings:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings:       total time =  428646.74 ms / 141825 tokens

======================== sorted layer importances
  0: Layer   0, <cos_sim> = 0.0804587
  1: Layer   1, <cos_sim> = 0.816333
  2: Layer   3, <cos_sim> = 0.855579
  3: Layer   2, <cos_sim> = 0.870939
  4: Layer   5, <cos_sim> = 0.882884
  5: Layer   7, <cos_sim> = 0.886822
  6: Layer   6, <cos_sim> = 0.891157
  7: Layer   4, <cos_sim> = 0.897281
  8: Layer   8, <cos_sim> = 0.898462
  9: Layer   9, <cos_sim> = 0.900521
 10: Layer  10, <cos_sim> = 0.910075
 11: Layer  11, <cos_sim> = 0.912746
 12: Layer  12, <cos_sim> = 0.916058
 13: Layer  13, <cos_sim> = 0.918256
 14: Layer  15, <cos_sim> = 0.921156
 15: Layer  16, <cos_sim> = 0.922013
 16: Layer  17, <cos_sim> = 0.923089
 17: Layer  14, <cos_sim> = 0.923667
 18: Layer  18, <cos_sim> = 0.935129
 19: Layer  21, <cos_sim> = 0.935497
 20: Layer  38, <cos_sim> = 0.938946
 21: Layer  20, <cos_sim> = 0.939555
 22: Layer  19, <cos_sim> = 0.939993
 23: Layer  22, <cos_sim> = 0.949833
 24: Layer  37, <cos_sim> = 0.952011
 25: Layer  23, <cos_sim> = 0.955484
 26: Layer  36, <cos_sim> = 0.956569
 27: Layer  24, <cos_sim> = 0.96045
 28: Layer  25, <cos_sim> = 0.963482
 29: Layer  35, <cos_sim> = 0.96357
 30: Layer  26, <cos_sim> = 0.963717
 31: Layer  34, <cos_sim> = 0.966742
 32: Layer  27, <cos_sim> = 0.967312
 33: Layer  33, <cos_sim> = 0.967905
 34: Layer  28, <cos_sim> = 0.96873
 35: Layer  32, <cos_sim> = 0.969066
 36: Layer  30, <cos_sim> = 0.969155
 37: Layer  29, <cos_sim> = 0.969895
 38: Layer  31, <cos_sim> = 0.969988

======================== sorted attention importances
  0: Layer   0, <cos_sim> = 0.253426
  1: Layer   1, <cos_sim> = 0.38511
  2: Layer   2, <cos_sim> = 0.568119
  3: Layer   3, <cos_sim> = 0.70009
  4: Layer   4, <cos_sim> = 0.753275
  5: Layer   5, <cos_sim> = 0.783473
  6: Layer   7, <cos_sim> = 0.822807
  7: Layer   6, <cos_sim> = 0.833536
  8: Layer   8, <cos_sim> = 0.85773
  9: Layer   9, <cos_sim> = 0.869933
 10: Layer  10, <cos_sim> = 0.870238
 11: Layer  11, <cos_sim> = 0.876139
 12: Layer  12, <cos_sim> = 0.880516
 13: Layer  15, <cos_sim> = 0.883828
 14: Layer  14, <cos_sim> = 0.890839
 15: Layer  13, <cos_sim> = 0.891501
 16: Layer  17, <cos_sim> = 0.892781
 17: Layer  16, <cos_sim> = 0.897206
 18: Layer  20, <cos_sim> = 0.90434
 19: Layer  19, <cos_sim> = 0.905305
 20: Layer  21, <cos_sim> = 0.905376
 21: Layer  18, <cos_sim> = 0.910555
 22: Layer  23, <cos_sim> = 0.921951
 23: Layer  26, <cos_sim> = 0.926056
 24: Layer  25, <cos_sim> = 0.927626
 25: Layer  24, <cos_sim> = 0.928499
 26: Layer  28, <cos_sim> = 0.936632
 27: Layer  22, <cos_sim> = 0.936688
 28: Layer  27, <cos_sim> = 0.939766
 29: Layer  29, <cos_sim> = 0.946173
 30: Layer  31, <cos_sim> = 0.950643
 31: Layer  39, <cos_sim> = 0.951655
 32: Layer  30, <cos_sim> = 0.952739
 33: Layer  32, <cos_sim> = 0.955543
 34: Layer  36, <cos_sim> = 0.955873
 35: Layer  34, <cos_sim> = 0.957643
 36: Layer  33, <cos_sim> = 0.958336
 37: Layer  38, <cos_sim> = 0.960393
 38: Layer  37, <cos_sim> = 0.960471
 39: Layer  35, <cos_sim> = 0.962264

======================== sorted ffn importances
  0: Layer   0, <cos_sim> = 0.562579
  1: Layer   1, <cos_sim> = 0.580676
  2: Layer   2, <cos_sim> = 0.616983
  3: Layer   3, <cos_sim> = 0.706686
  4: Layer   4, <cos_sim> = 0.731208
  5: Layer   6, <cos_sim> = 0.756786
  6: Layer   5, <cos_sim> = 0.757354
  7: Layer   7, <cos_sim> = 0.796257
  8: Layer   8, <cos_sim> = 0.815461
  9: Layer  10, <cos_sim> = 0.824589
 10: Layer   9, <cos_sim> = 0.826519
 11: Layer  11, <cos_sim> = 0.846745
 12: Layer  13, <cos_sim> = 0.859737
 13: Layer  14, <cos_sim> = 0.86228
 14: Layer  12, <cos_sim> = 0.866246
 15: Layer  16, <cos_sim> = 0.866582
 16: Layer  15, <cos_sim> = 0.868753
 17: Layer  18, <cos_sim> = 0.870342
 18: Layer  19, <cos_sim> = 0.870973
 19: Layer  17, <cos_sim> = 0.874143
 20: Layer  20, <cos_sim> = 0.886187
 21: Layer  22, <cos_sim> = 0.892857
 22: Layer  21, <cos_sim> = 0.902702
 23: Layer  23, <cos_sim> = 0.902868
 24: Layer  24, <cos_sim> = 0.904163
 25: Layer  25, <cos_sim> = 0.904319
 26: Layer  27, <cos_sim> = 0.914438
 27: Layer  26, <cos_sim> = 0.917688
 28: Layer  28, <cos_sim> = 0.926051
 29: Layer  38, <cos_sim> = 0.927326
 30: Layer  29, <cos_sim> = 0.92942
 31: Layer  30, <cos_sim> = 0.932488
 32: Layer  35, <cos_sim> = 0.934298
 33: Layer  31, <cos_sim> = 0.934668
 34: Layer  37, <cos_sim> = 0.935018
 35: Layer  33, <cos_sim> = 0.936569
 36: Layer  32, <cos_sim> = 0.938647
 37: Layer  36, <cos_sim> = 0.938813
 38: Layer  34, <cos_sim> = 0.94036
```

</details>

Really appreciate your implementing this for further experimentation! Gotta run for now but will dig in more later this week! Thanks!