### 📝 [#285](https://github.com/ikawrakow/ik_llama.cpp/issues/285) - llama-perplexity giving all NaNs on unsloth Q8_0 quant

| **Author** | `ubergarm` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-03-24 |
| **Updated** | 2025-03-27 |

---

#### Description

Moving this into its own ticket from [#271](https://github.com/ikawrakow/ik_llama.cpp/issues/271#issuecomment-2740969252).

Basically, I was able to run a clean `llama-perplexity` on [unsloth/DeepSeek-R1-Q8_0](https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main/DeepSeek-R1-Q8_0) with mainline llama.cpp, but when I tried with this fork it was throwing all NaNs.

It might be a race condition or something given the logging messages seem to indicate perplexity calculations may be starting before the tensor buffers are fully computed (or just a logging fluke).

> Are the NaNs ik_llama.cpp specific, or does also mainline produce NaNs with the Unsloth Q8_0 model?

Yes, I got mainline `llama.cpp@b1b132ef` to give at a full clean `llama-perplexity` run with no NaNs with the same GGUF files:

<details>
<summary>mainline llama.cpp clean Q8_0 perplexity run</summary>

```bash
## llama.cpp mainline

$ git rev-parse --short head
b1b132ef

$ numactl -N 0 -m 0 \
./build/bin/llama-perplexity \
    --model /mnt/ai/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-Q8_0/DeepSeek-R1.Q8_0-00001-of-00015.gguf \
    -ctk f16 -ctv f16 \
    --ctx-size 512 \
    --ubatch-size 512 \
    -f wiki.test.raw \
    --numa numactl \
    --threads 80

build: 4905 (b1b132ef) with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu
/proc/sys/kernel/numa_balancing is enabled, this has been observed to impair performance
llama_model_loader: additional 14 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 48 key-value pairs and 1025 tensors from /mnt/ai/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-Q8_0/DeepSeek-R1.Q8_0-00001-of-00015.gguf (version GGUF V3 (latest))
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
llama_model_loader: - kv  15:                          general.file_type u32              = 7
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
# remove tokenizer as characters mess up my copy/paste clipboard
llama_model_loader: - kv  38:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  39:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  40:            tokenizer.ggml.padding_token_id u32              = 128815
llama_model_loader: - kv  41:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  42:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  43:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  44:               general.quantization_version u32              = 2
llama_model_loader: - kv  45:                                   split.no u16              = 0
llama_model_loader: - kv  46:                                split.count u16              = 15
llama_model_loader: - kv  47:                        split.tensors.count i32              = 1025
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  664 tensors
print_info: file format = GGUF V3 (latest)
print_info: file type   = Q8_0
print_info: file size   = 664.29 GiB (8.50 BPW) 
load: special_eos_id is not in special_eog_ids - the tokenizer config may be incorrect
load: special tokens cache size = 819
load: token to piece cache size = 0.8223 MB
print_info: arch             = deepseek2
print_info: vocab_only       = 0
print_info: n_ctx_train      = 163840
print_info: n_embd           = 7168
print_info: n_layer          = 61
print_info: n_head           = 128
print_info: n_head_kv        = 128
print_info: n_rot            = 64
print_info: n_swa            = 0
print_info: n_swa_pattern    = 1
print_info: n_embd_head_k    = 192
print_info: n_embd_head_v    = 128
print_info: n_gqa            = 1
print_info: n_embd_k_gqa     = 24576
print_info: n_embd_v_gqa     = 16384
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
print_info: general.name     = DeepSeek R1 BF16
print_info: n_layer_dense_lead   = 3
print_info: n_lora_q             = 1536
print_info: n_lora_kv            = 512
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
print_info: PAD token        = 128815 '<｜PAD▁TOKEN｜>'
print_info: LF token         = 201 'Ċ'
print_info: FIM PRE token    = 128801 '<｜fim▁begin｜>'
print_info: FIM SUF token    = 128800 '<｜fim▁hole｜>'
print_info: FIM MID token    = 128802 '<｜fim▁end｜>'
print_info: EOG token        = 1 '<｜end▁of▁sentence｜>'
print_info: max token length = 256
load_tensors: loading model tensors, this can take a while... (mmap = true)
load_tensors:          AMX model buffer size = 18214.39 MiB
load_tensors:   CPU_Mapped model buffer size = 45565.90 MiB
load_tensors:   CPU_Mapped model buffer size = 46661.11 MiB
load_tensors:   CPU_Mapped model buffer size = 46661.11 MiB
load_tensors:   CPU_Mapped model buffer size = 46661.11 MiB
load_tensors:   CPU_Mapped model buffer size = 46661.11 MiB
load_tensors:   CPU_Mapped model buffer size = 46661.11 MiB
load_tensors:   CPU_Mapped model buffer size = 46661.11 MiB
load_tensors:   CPU_Mapped model buffer size = 46661.11 MiB
load_tensors:   CPU_Mapped model buffer size = 46661.11 MiB
load_tensors:   CPU_Mapped model buffer size = 46661.11 MiB
load_tensors:   CPU_Mapped model buffer size = 46661.11 MiB
load_tensors:   CPU_Mapped model buffer size = 46661.11 MiB
load_tensors:   CPU_Mapped model buffer size = 46661.11 MiB
load_tensors:   CPU_Mapped model buffer size = 46661.11 MiB
load_tensors:   CPU_Mapped model buffer size = 28077.60 MiB
....................................................................................................
llama_context: constructing llama_context
llama_context: n_seq_max     = 4
llama_context: n_ctx         = 2048
llama_context: n_ctx_per_seq = 512
llama_context: n_batch       = 2048
llama_context: n_ubatch      = 512
llama_context: causal_attn   = 1
llama_context: flash_attn    = 0
llama_context: freq_base     = 10000.0
llama_context: freq_scale    = 0.025
llama_context: n_ctx_per_seq (512) < n_ctx_train (163840) -- the full capacity of the model will not be utilized
llama_context:        CPU  output buffer size =     1.97 MiB
init: kv_size = 2048, offload = 1, type_k = 'f16', type_v = 'f16', n_layer = 61, can_shift = 0
init:        CPU KV buffer size =  9760.00 MiB
llama_context: KV self size  = 9760.00 MiB, K (f16): 5856.00 MiB, V (f16): 3904.00 MiB
llama_context:        CPU compute buffer size =   670.01 MiB
llama_context: graph nodes  = 5025
llama_context: graph splits = 1
common_init_from_params: KV cache shifting is not supported for this context, disabling KV cache shifting
common_init_from_params: setting dry_penalty_last_n to ctx_size = 2048
common_init_from_params: warming up the model with an empty run - please wait ... (--no-warmup to disable)

system_info: n_threads = 80 (n_threads_batch = 80) / 512 | CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | BMI2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | AMX_INT8 = 1 | LLAMAFILE = 1 | OPENMP = 1 | AARCH64_REPACK = 1 | 
perplexity: tokenizing the input ..
perplexity: tokenization took 724.131 ms
perplexity: calculating perplexity over 561 chunks, n_ctx=512, batch_size=2048, n_seq=4
perplexity: 60.35 seconds per pass - ETA 2 hours 21.05 minutes
[1]2.5013,[2]3.2882,[3]2.3700,[4]1.9826,[5]1.7891,[6]1.6469,[7]1.5544,[8]1.4883,[9]1.4387,[10]1.3997,[11]1.3842,[12]1.4194,[13]1.4299,[14]1.5576,[15]1.6890,[16]1.7483,[17]1.9110,[18]2.0408,[19]2.0033,[20]1.9911,[21]2.0982,[22]2.0702,[23]2.0430,[24]2.0560,[25]2.0267,[26]2.0035,[27]2.0524,[28]2.0598,[29]2.1085,[30]2.1396,[31]2.1742,[32]2.1918,[33]2.2304,[34]2.2706,[35]2.3192,[36]2.3717,[37]2.4071,[38]2.4526,[39]2.4940,[40]2.5527,[41]2.5950,[42]2.6072,[43]2.6559,[44]2.6723,[45]2.7517,[46]2.8023,[47]2.7573,[48]2.7107,[49]2.6842,[50]2.7039,[51]2.7504,[52]2.7650,[53]2.8143,[54]2.8275,[55]2.8585,[56]2.8898,[57]2.9036,[58]2.9402,[59]2.9512,[60]2.9968,[61]3.0366,[62]3.0894,[63]3.1213,[64]3.1652,[65]3.1751,[66]3.1579,[67]3.1353,[68]3.1665,[69]3.1618,[70]3.1771,[71]3.1956,[72]3.2115,[73]3.2259,[74]3.2494,[75]3.2284,[76]3.1816,[77]3.1389,[78]3.1344,[79]3.1122,[80]3.0929,[81]3.0561,[82]3.0596,[83]3.0282,[84]2.9923,[85]2.9572,[86]2.9321,[87]2.9257,[88]2.8971,[89]2.8805,[90]2.8542,[91]2.8245,[92]2.7997,[93]2.7731,[94]2.7463,[95]2.7224,[96]2.7210,[97]2.7283,[98]2.7132,[99]2.6960,[100]2.6985,[101]2.6899,[102]2.7065,[103]2.7327,[104]2.7513,[105]2.7482,[106]2.7706,[107]2.7948,[108]2.8154,[109]2.8493,[110]2.8832,[111]2.9028,[112]2.8771,[113]2.8641,[114]2.8419,[115]2.8266,[116]2.8114,[117]2.7885,[118]2.7677,[119]2.7465,[120]2.7277,[121]2.7122,[122]2.6947,[123]2.6785,[124]2.6597,[125]2.6422,[126]2.6257,[127]2.6117,[128]2.6027,[129]2.5920,[130]2.5797,[131]2.5724,[132]2.5798,[133]2.5894,[134]2.5959,[135]2.6064,[136]2.6225,[137]2.6379,[138]2.6461,[139]2.6576,[140]2.6586,[141]2.6603,[142]2.6594,[143]2.6599,[144]2.6569,[145]2.6481,[146]2.6467,[147]2.6512,[148]2.6510,[149]2.6527,[150]2.6476,[151]2.6458,[152]2.6429,[153]2.6392,[154]2.6399,[155]2.6443,[156]2.6465,[157]2.6527,[158]2.6615,[159]2.6634,[160]2.6723,[161]2.6806,[162]2.6900,[163]2.6941,[164]2.7141,[165]2.7378,[166]2.7551,[167]2.7673,[168]2.7915,[169]2.8139,[170]2.8354,[171]2.8586,[172]2.8427,[173]2.8264,[174]2.8128,[175]2.7995,[176]2.7872,[177]2.7756,[178]2.7630,[179]2.7493,[180]2.7532,[181]2.7671,[182]2.7822,[183]2.7970,[184]2.8112,[185]2.8216,[186]2.8381,[187]2.8534,[188]2.8675,[189]2.8782,[190]2.8785,[191]2.8859,[192]2.8899,[193]2.8950,[194]2.9146,[195]2.9234,[196]2.9368,[197]2.9468,[198]2.9513,[199]2.9570,[200]2.9566,[201]2.9717,[202]2.9671,[203]2.9724,[204]2.9760,[205]2.9759,[206]2.9785,[207]2.9874,[208]2.9970,[209]3.0063,[210]3.0069,[211]3.0022,[212]3.0021,[213]3.0097,[214]3.0116,[215]3.0174,[216]3.0180,[217]3.0140,[218]3.0142,[219]3.0152,[220]3.0146,[221]3.0148,[222]3.0149,[223]3.0155,[224]3.0205,[225]3.0224,[226]3.0144,[227]3.0122,[228]3.0145,[229]3.0191,[230]3.0256,[231]3.0318,[232]3.0236,[233]3.0158,[234]3.0158,[235]3.0142,[236]3.0230,[237]3.0315,[238]3.0410,[239]3.0508,[240]3.0601,[241]3.0713,[242]3.0857,[243]3.0992,[244]3.1073,[245]3.1183,[246]3.1288,[247]3.1276,[248]3.1235,[249]3.1216,[250]3.1154,[251]3.1133,[252]3.1158,[253]3.1196,[254]3.1267,[255]3.1331,[256]3.1369,[257]3.1393,[258]3.1405,[259]3.1438,[260]3.1459,[261]3.1473,[262]3.1465,[263]3.1522,[264]3.1545,[265]3.1550,[266]3.1568,[267]3.1597,[268]3.1634,[269]3.1665,[270]3.1659,[271]3.1644,[272]3.1577,[273]3.1576,[274]3.1507,[275]3.1399,[276]3.1291,[277]3.1308,[278]3.1410,[279]3.1472,[280]3.1551,[281]3.1625,[282]3.1687,[283]3.1751,[284]3.1818,[285]3.1954,[286]3.1979,[287]3.2013,[288]3.2060,[289]3.2087,[290]3.2005,[291]3.1911,[292]3.1892,[293]3.1883,[294]3.1855,[295]3.1829,[296]3.1848,[297]3.1853,[298]3.1902,[299]3.1961,[300]3.1992,[301]3.2030,[302]3.2052,[303]3.2072,[304]3.2067,[305]3.2186,[306]3.2261,[307]3.2370,[308]3.2258,[309]3.2204,[310]3.2109,[311]3.2145,[312]3.2167,[313]3.2230,[314]3.2251,[315]3.2283,[316]3.2297,[317]3.2315,[318]3.2321,[319]3.2324,[320]3.2367,[321]3.2370,[322]3.2390,[323]3.2454,[324]3.2463,[325]3.2516,[326]3.2563,[327]3.2604,[328]3.2634,[329]3.2652,[330]3.2715,[331]3.2752,[332]3.2800,[333]3.2786,[334]3.2787,[335]3.2792,[336]3.2794,[337]3.2805,[338]3.2808,[339]3.2835,[340]3.2871,[341]3.2925,[342]3.3015,[343]3.3108,[344]3.3161,[345]3.3074,[346]3.2997,[347]3.2945,[348]3.2872,[349]3.2835,[350]3.2817,[351]3.2864,[352]3.3013,[353]3.3104,[354]3.3232,[355]3.3318,[356]3.3371,[357]3.3487,[358]3.3583,[359]3.3615,[360]3.3680,[361]3.3772,[362]3.3858,[363]3.3915,[364]3.3981,[365]3.4044,[366]3.4148,[367]3.4234,[368]3.4301,[369]3.4380,[370]3.4465,[371]3.4602,[372]3.4689,[373]3.4722,[374]3.4758,[375]3.4808,[376]3.4936,[377]3.5048,[378]3.5075,[379]3.5069,[380]3.5037,[381]3.5083,[382]3.5139,[383]3.5175,[384]3.5218,[385]3.5257,[386]3.5319,[387]3.5377,[388]3.5411,[389]3.5308,[390]3.5213,[391]3.5107,[392]3.5051,[393]3.4955,[394]3.4865,[395]3.4772,[396]3.4672,[397]3.4584,[398]3.4488,[399]3.4385,[400]3.4296,[401]3.4196,[402]3.4093,[403]3.4007,[404]3.3905,[405]3.3811,[406]3.3711,[407]3.3619,[408]3.3531,[409]3.3446,[410]3.3386,[411]3.3392,[412]3.3345,[413]3.3363,[414]3.3385,[415]3.3353,[416]3.3351,[417]3.3375,[418]3.3317,[419]3.3332,[420]3.3308,[421]3.3298,[422]3.3312,[423]3.3304,[424]3.3346,[425]3.3341,[426]3.3346,[427]3.3335,[428]3.3360,[429]3.3378,[430]3.3406,[431]3.3413,[432]3.3403,[433]3.3366,[434]3.3366,[435]3.3289,[436]3.3226,[437]3.3185,[438]3.3167,[439]3.3134,[440]3.3183,[441]3.3237,[442]3.3311,[443]3.3293,[444]3.3302,[445]3.3315,[446]3.3363,[447]3.3396,[448]3.3421,[449]3.3452,[450]3.3490,[451]3.3520,[452]3.3540,[453]3.3557,[454]3.3543,[455]3.3564,[456]3.3567,[457]3.3594,[458]3.3646,[459]3.3653,[460]3.3654,[461]3.3622,[462]3.3659,[463]3.3732,[464]3.3785,[465]3.3714,[466]3.3696,[467]3.3677,[468]3.3688,[469]3.3658,[470]3.3631,[471]3.3634,[472]3.3640,[473]3.3632,[474]3.3624,[475]3.3635,[476]3.3619,[477]3.3610,[478]3.3617,[479]3.3633,[480]3.3660,[481]3.3620,[482]3.3654,[483]3.3646,[484]3.3682,[485]3.3746,[486]3.3775,[487]3.3812,[488]3.3864,[489]3.3889,[490]3.3935,[491]3.3997,[492]3.4042,[493]3.4040,[494]3.4052,[495]3.4076,[496]3.4095,[497]3.4124,[498]3.4127,[499]3.4122,[500]3.4163,[501]3.4209,[502]3.4200,[503]3.4185,[504]3.4205,[505]3.4239,[506]3.4323,[507]3.4350,[508]3.4385,[509]3.4312,[510]3.4254,[511]3.4188,[512]3.4142,[513]3.4080,[514]3.4065,[515]3.4084,[516]3.4033,[517]3.4032,[518]3.4024,[519]3.4029,[520]3.4073,[521]3.4062,[522]3.4047,[523]3.4105,[524]3.4092,[525]3.4076,[526]3.4028,[527]3.3979,[528]3.3942,[529]3.3913,[530]3.3883,[531]3.3852,[532]3.3797,[533]3.3735,[534]3.3692,[535]3.3700,[536]3.3728,[537]3.3759,[538]3.3785,[539]3.3812,[540]3.3865,[541]3.3898,[542]3.3922,[543]3.3865,[544]3.3822,[545]3.3819,[546]3.3753,[547]3.3688,[548]3.3624,[549]3.3557,[550]3.3497,[551]3.3436,[552]3.3378,[553]3.3319,[554]3.3298,[555]3.3283,[556]3.3311,[557]3.3351,[558]3.3410,[559]3.3455,[560]3.3508,[561]3.3490,
Final estimate: PPL = 3.3490 +/- 0.01849

llama_perf_context_print:        load time =  226439.86 ms
llama_perf_context_print: prompt eval time = 8320298.42 ms / 287232 tokens (   28.97 ms per token,    34.52 tokens per second)
llama_perf_context_print:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_perf_context_print:       total time = 8511632.28 ms / 287233 tokens
```
</details>

I tried a few combinations of sha's with and without `-rtr`, `-mla 1`, exact same command as mainline llama.cpp above, etc, but always getting NaNs with `ik_llama.cpp` so far:

<details>
<summary>ik_llama.cpp NaNs on same quant</summary>

```bash
## ik_llama.cpp@f2fb15de

$ numactl -N 0 -m 0 \
./build/bin/llama-perplexity \
    --model /mnt/ai/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-Q8_0/DeepSeek-R1.Q8_0-00001-of-00015.gguf \
    -rtr \
    -ctk f16 -ctv f16 \
    -mla 2 -fa \
    -amb 2048 \
    -fmoe \
    --ctx-size 512 \
    --ubatch-size 512 \
    -f wiki.test.raw \
    --numa numactl \
    --threads 80

main: build = 3596 (f2fb15de)
main: built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu
main: seed  = 1742247516
llama_model_loader: additional 14 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 48 key-value pairs and 1025 tensors from /mnt/ai/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-Q8_0/DeepSeek-R1.Q8_0-00001-of-00015.gguf (version GGUF V3 (latest))
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
llama_model_loader: - kv  15:                          general.file_type u32              = 7
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
# comment out tokenzier stuff for my poor clipboard
llama_model_loader: - kv  38:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  39:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  40:            tokenizer.ggml.padding_token_id u32              = 128815
llama_model_loader: - kv  41:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  42:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  43:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  44:               general.quantization_version u32              = 2
llama_model_loader: - kv  45:                                   split.no u16              = 0
llama_model_loader: - kv  46:                                split.count u16              = 15
llama_model_loader: - kv  47:                        split.tensors.count i32              = 1025
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  664 tensors
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
llm_load_print_meta: model ftype      = Q8_0
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
WARNING: /proc/sys/kernel/numa_balancing is enabled, this has been observed to impair performance
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
Computed blk.52.attn_v_b.weight as 128 x 512 x 128 and sllama_new_context_with_model: n_ctx      = 2048
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 2
llama_new_context_with_model: attn_max_b = 2048
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init: layer 0: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 1: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 2: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 3: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 4: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 5: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 6: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 7: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 8: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 9: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 10: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 11: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 12: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 13: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 14: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 15: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 16: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 17: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 18: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 19: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 20: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 21: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 22: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 23: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 24: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 25: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 26: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 27: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 28: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 29: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 30: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 31: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 32: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 33: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 34: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 35: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 36: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 37: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 38: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 39: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 40: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 41: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 42: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 43: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 44: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 45: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 46: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 47: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 48: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 49: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 50: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 51: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 52: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 53: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 54: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 55: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 56: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 57: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 58: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 59: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 60: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init:        CPU KV buffer size =   137.25 MiB
llama_new_context_with_model: KV self size  =  137.25 MiB, c^KV (f16):  137.25 MiB, kv^T: not used
llama_new_context_with_model:        CPU  output buffer size =     1.97 MiB
llama_new_context_with_model:        CPU compute buffer size =   432.01 MiB
llama_new_context_with_model: graph nodes  = 3365
llama_new_context_with_model: graph splits = 1

system_info: n_threads = 80 / 512 | AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | 
perplexity: tokenizing the input ..
perplexity: tokenization took 912.853 ms
perplexity: calculating perplexity over 561 chunks, n_ctx=512, batch_size=2048, n_seq=4
perplexity: 21.11 seconds per pass - ETA 49.35 minutes
tored in buffer CPU
Computed blk.53.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.54.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.55.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.56.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.57.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.58.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.59.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.60.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
============ Repacked 663 tensors
[1]nan,[2]nan,[3]nan,[4]nan,[5]nan,[6]nan,[7]nan,[8]nan,[9]nan,[10]nan,[11]nan,[12]nan,[13]nan,[14]nan,[15]nan,[16]nan,[17]nan,[18]nan,[19]nan,[20]nan,[21]nan,[22]nan,[23]nan,[24]nan,[25]nan,[26]nan,[27]nan,[28]nan,[29]nan,[30]nan,[31]nan,[32]nan,[33]nan,[34]nan,[35]nan,[36]nan,[37]nan,[38]nan,[39]nan,[40]nan,[41]nan,[42]nan,[43]nan,[44]nan,[45]nan,[46]nan,[47]nan,[48]nan,[49]nan,[50]nan,[51]nan,[52]nan,[53]nan,[54]nan,[55]nan,[56]nan,[57]nan,[58]nan,[59]nan,[60]nan,[61]nan,[62]nan,[63]nan,[64]nan,
```
</details>

Trying one more time with todays updates and an offline repacked quant:

<details>

<summary>Trying `ik_llama.cpp@9fe6fc37` with offline repacked quant</summary>

```bash
$ git checkout ik/offline_repack

$ git rev-parse --short HEAD
9fe6fc37

$ numactl -N 0 -m 0 \
./build/bin/llama-perplexity \
    --model /mnt/ai/models/unsloth/repack/DeepSeek-R1-Q8_0_R8.gguf \
    -ctk q8_0 \
    -mla 2 -fa \
    -amb 512 \
    -fmoe \
    --ctx-size 512 \
    --ubatch-size 512 \
    -f wiki.test.raw \
    --seed 1337 \
    --numa numactl \
    --threads 128

main: build = 3604 (9fe6fc37)
main: built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu
main: seed  = 1337
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
Collama_new_context_with_model: n_ctx      = 2048
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 2
llama_new_context_with_model: attn_max_b = 512
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init: layer 0: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 1: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 2: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 3: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 4: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 5: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 6: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 7: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 8: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 9: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 10: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 11: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 12: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 13: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 14: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 15: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 16: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 17: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 18: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 19: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 20: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 21: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 22: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 23: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 24: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 25: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 26: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 27: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 28: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 29: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 30: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 31: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 32: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 33: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 34: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 35: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 36: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 37: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 38: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 39: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 40: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 41: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 42: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 43: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 44: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 45: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 46: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 47: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 48: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 49: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 50: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 51: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 52: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 53: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 54: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 55: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 56: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 57: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 58: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 59: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 60: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init:        CPU KV buffer size =    72.91 MiB
llama_new_context_with_model: KV self size  =   72.91 MiB, c^KV (q8_0):   72.91 MiB, kv^T: not used
llama_new_context_with_model:        CPU  output buffer size =     1.97 MiB
llama_new_context_with_model:        CPU compute buffer size =   450.01 MiB
llama_new_context_with_model: graph nodes  = 3487
llama_new_context_with_model: graph splits = 1

system_info: n_threads = 128 / 512 | AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | 
perplexity: tokenizing the input ..
perplexity: tokenization took 1752.8 ms
perplexity: calculating perplexity over 561 chunks, n_ctx=512, batch_size=2048, n_seq=4
perplexity: 15.91 seconds per pass - ETA 37.20 minutes
mputed blk.54.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.55.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.56.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.57.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.58.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.59.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.60.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
[1]nan,[2]nan,[3]nan,[4]nan,[5]nan,[6]nan,[7]nan,[8]nan,[9]nan,[10]nan,[11]nan,[12]nan,[13]nan,[14]nan,[15]nan,[16]nan,[17]nan,[18]nan,[19]nan,[20]nan,[21]nan,[22]nan,[23]nan,[24]nan,[25]nan,[26]nan,[27]nan,[28]nan,
```

</details>

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-03-25** at **07:28:41**:<br>

Not sure how to solve this one. As you are using `Q8_0` for the attention tensors in other models, and not getting NaNs, the issue must be somehow in the expert tensors. To help debug the problem, I would appreciate if you could do a run with the model producing NaNs with vanilla configuration, i.e.,
```
./bin/llama-perplexity -m /mnt/ai/models/unsloth/repack/DeepSeek-R1-Q8_0_R8.gguf -f wiki.test.raw -t 128 -b 512
```

---

👤 **ubergarm** commented the **2025-03-25** at **17:33:14**:<br>

Okay, coming back around to this one after seeing some NaNs running [llama-perplexity](https://github.com/ikawrakow/ik_llama.cpp/discussions/286#discussioncomment-12618097)

<details>

<summary>llama-perplexity `main@98a264a2` Logs</summary>

```bash
$ git rev-parse --short HEAD
98a264a2

# running on CPU 1 as CPU 0 is busy with benchmarks...
$ numactl -N 1 -m 1 \
./build/bin/llama-perplexity \
    -m /mnt/ai/models/unsloth/repack/DeepSeek-R1-Q8_0_R8.gguf \
    -f wiki.test.raw \
    -t 128 \
    -b 512 \
    --numa numactl

main: build = 3608 (98a264a2)
main: built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu
main: seed  = 1742922529
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
llama_model_loader: - kv  35:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<...
llama_model_loader: - kv  36:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  37:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
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
Computed blk.54.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.55.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.56.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.57.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.58.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.59.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.60.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
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
perplexity: tokenizing the input ..
perplexity: tokenization took 1055.81 ms
perplexity: calculating perplexity over 561 chunks, n_ctx=512, batch_size=512, n_seq=1
perplexity: 5.62 seconds per pass - ETA 52.57 minutes
[1]nan,[2]nan,[3]nan,[4]nan,[5]nan,[6]nan,[7]nan,[8]nan,[9]nan,[10]nan,[11]nan,[12]nan,[13]nan,[14]nan,[15]nan,[16]nan,[17]nan,[18]nan,[19]nan,[20]nan,[21]nan,[22]nan,[23]nan,[24]nan,[25]nan,[26]nan,[27]nan,[28]nan,[29]nan,[30]nan,[31]nan,[32]nan,[33]nan,[34]nan,[35]nan,[36]nan,[37]nan,[38]nan,[39]nan,[40]nan,[41]nan,[42]nan,[43]nan,[44]nan,[45]nan,[46]nan,[47]nan,[48]nan,[49]nan,[50]nan,[51]nan,[52]nan,[53]nan,[54]nan,[55]nan,[56]nan,[57]nan,[58]nan,[59]nan,[60]nan,[61]nan,[62]nan,[63]nan,[64]nan,[65]nan,[66]nan,[67]nan,[68]nan,[69]nan,[70]nan,[71]nan,[72]nan,[73]nan,[74]nan,[75]nan,[76]nan,[77]nan,[78]nan,[79]nan,[80]nan,[81]nan,[82]nan,[83]nan,[84]nan,[85]nan,[86]nan,^C^C
```

</details>

---

👤 **ubergarm** commented the **2025-03-25** at **17:33:14**:<br>

Okay, coming back around to this one after seeing some NaNs running [llama-perplexity]()

<details>

<summary>llama-perplexity `main@98a264a2` Logs</summary>

```bash
$ git rev-parse --short HEAD
98a264a2

# running on CPU 1 as CPU 0 is busy with benchmarks...
$ numactl -N 1 -m 1 \
./build/bin/llama-perplexity \
    -m /mnt/ai/models/unsloth/repack/DeepSeek-R1-Q8_0_R8.gguf \
    -f wiki.test.raw \
    -t 128 \
    -b 512 \
    --numa numactl

main: build = 3608 (98a264a2)
main: built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu
main: seed  = 1742922529
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
llama_model_loader: - kv  35:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<...
llama_model_loader: - kv  36:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  37:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
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
Computed blk.54.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.55.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.56.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.57.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.58.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.59.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.60.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
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
perplexity: tokenizing the input ..
perplexity: tokenization took 1055.81 ms
perplexity: calculating perplexity over 561 chunks, n_ctx=512, batch_size=512, n_seq=1
perplexity: 5.62 seconds per pass - ETA 52.57 minutes
[1]nan,[2]nan,[3]nan,[4]nan,[5]nan,[6]nan,[7]nan,[8]nan,[9]nan,[10]nan,[11]nan,[12]nan,[13]nan,[14]nan,[15]nan,[16]nan,[17]nan,[18]nan,[19]nan,[20]nan,[21]nan,[22]nan,[23]nan,[24]nan,[25]nan,[26]nan,[27]nan,[28]nan,[29]nan,[30]nan,[31]nan,[32]nan,[33]nan,[34]nan,[35]nan,[36]nan,[37]nan,[38]nan,[39]nan,[40]nan,[41]nan,[42]nan,[43]nan,[44]nan,[45]nan,[46]nan,[47]nan,[48]nan,[49]nan,[50]nan,[51]nan,[52]nan,[53]nan,[54]nan,[55]nan,[56]nan,[57]nan,[58]nan,[59]nan,[60]nan,[61]nan,[62]nan,[63]nan,[64]nan,[65]nan,[66]nan,[67]nan,[68]nan,[69]nan,[70]nan,[71]nan,[72]nan,[73]nan,[74]nan,[75]nan,[76]nan,[77]nan,[78]nan,[79]nan,[80]nan,[81]nan,[82]nan,[83]nan,[84]nan,[85]nan,[86]nan,^C^C
```

</details>

---

👤 **ubergarm** commented the **2025-03-25** at **20:24:54**:<br>

Ahh, tried another more simple `q8_0` everything quant for `llama-imatrix` and still got `nan` eventually:

https://github.com/ikawrakow/ik_llama.cpp/discussions/286#discussioncomment-12620133

---

👤 **ikawrakow** commented the **2025-03-26** at **06:39:01**:<br>

So, my current hypothesis is that the NaNs are caused by an overflow of the `Q8_1` block sum, which is stored as `fp16`.

@ubergarm 

Can you test if #291 eliminates the NaNs for `Q8_0` and/or `Q8_0_R8`? Thanks.