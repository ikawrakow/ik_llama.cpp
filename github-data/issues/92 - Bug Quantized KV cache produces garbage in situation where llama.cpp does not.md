## 📌 [Issue #92](https://github.com/ikawrakow/ik_llama.cpp/issues/92) - Bug: Quantized KV cache produces garbage in situation where llama.cpp does not

| **Author** | `saood06` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-10-17 |
| **Updated** | 2025-02-11 |

---

## 📄 Description

### What happened?

Was running Mistral Large 2 with partial offload with AMD 5600X + RTX 3090.
Provided the same ~28k prompt to each, llama.cpp produced output that was coherent and similar to non quantized KV cache, ik_llama.cpp produced garbage ( the top 10 token candidates stayed mostly the same as it was outputting).

Command used:
.\llama-server.exe -m "...\Mistral-Large-Instruct-2407.i1-IQ4_XS.gguf" -t 6 -ngl 29 -c 33333 --host 0.0.0.0 --no-mmap -fa -ctk q8_0 -ctv q8_0


### Name and Version

Working llama.cpp version:
version: 3658 (f1485161)
Not working ik_llama.cpp:
version: 3459 (baab1d9a)

### What operating system are you seeing the problem on?

Windows

### Relevant log output

_No response_

---

## 💬 Conversation

👤 **ikawrakow** commented on **2024-10-18** at **07:35:48**

What happens with `-ctv q4_0` ?

This model is too large for my compute capabilities, so I cannot try myself to see what happens. Is FA running on the GPU or on the CPU? The `llama.cpp` GPU FA implementation supports a limited set of head sizes, which is even more limited with quantized cache. If the GPU FA does not support this particular model and the FA kernel is running on the CPU, then you cannot use `Q8_0` for the V-cache. This is because I have changed the bit arrangement in `Q8_0` when quantization is done during inference, with the result that `Q8_0` cannot be used for V cache when FA is running on the CPU. From my experience, it is much more important to use better quantization accuracy for the K-cache than it is for the V-cache. `-ctk q8_0 -ctv iq4_nl` is basically the same as `-ctk q8_0 -ctv q8_0` while needing 25% less RAM/VRAM.

But if the FA kernel is running on the GPU then I don't know what is happening. I haven't made any changes there.

---

👤 **Nexesenex** commented on **2024-10-18** at **17:16:57**

@saood06 : you can compile with GGML_FA_ALL_QUANTS, and try -ctk q5_1 -ctv q5_0, retain a very decent quality, and check if the phenomena you mention in q8_0 still occurs there. That KV quant works on IK's LlamaCPP, on a Mistral Large (I use it too) quantized with in custom quant based mainly on IQ4_KSS.

![2024-10-18 19_13_38-KVQ ods — LibreOffice Calc](https://github.com/user-attachments/assets/f285b951-15d8-43d5-b1ff-f17c0934fb02)

Data are Johannes Gaessler's.

---

👤 **ikawrakow** commented on **2024-10-19** at **14:24:01**

Judging by PPL and KLD, `-ctk q8_0 -ctv iq4_nl` beats `ctk q5_1 -ctv q5_0` by quite some margin. It uses ~10% more memory for the cache,  but inference is slightly faster.

---

👤 **Nexesenex** commented on **2024-10-19** at **14:31:47**

As far as I know, IQ quants are not available for KVQ cache on Cuda.
ggml\src\ggml-cuda\fattn.cu doesn't contain any reference to them.

---

👤 **ikawrakow** commented on **2024-10-19** at **14:54:06**

> As far as I know, IQ quants are not available for KVQ cache on Cuda.

Have you tried with this repo? `IQ4_NL` is available for KV cache. 

<details>
<summary>./bin/llama-perplexity -m llama-3.1-instruct-iq4kss.gguf -f ../tests/wiki.test.raw -t 1 -ngl 100 -fa -ctk q8_0 -ctv iq4_nl</summary>
<code>
llama_kv_cache_init:      CUDA0 KV buffer size =   104.00 MiB 
llama_new_context_with_model: KV self size  =  104.00 MiB, K (q8_0):   68.00 MiB, V (iq4_nl):   36.00 MiB 
llama_new_context_with_model:  CUDA_Host  output buffer size =     1.96 MiB 
llama_new_context_with_model:      CUDA0 compute buffer size =   266.50 MiB 
llama_new_context_with_model:  CUDA_Host compute buffer size =    12.01 MiB 
llama_new_context_with_model: graph nodes  = 806 
llama_new_context_with_model: graph splits = 2 

system_info: n_threads = 1 / 32 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | 
perplexity: tokenizing the input ..
perplexity: tokenization took 142.301 ms
perplexity: calculating perplexity over 564 chunks, n_ctx=512, batch_size=2048, n_seq=4
perplexity: 0.64 seconds per pass - ETA 1.50 minutes
[1]4.2997,[2]5.5078,[3]6.0053,[4]6.3547,[5]6.7268,[6]7.0718,[7]7.4507,[8]7.9743,[9]8.6317,[10]8.8219,[11]8.9887,[12]9.0891,[13]9.4975,[14]9.0932,[15]9.0087,[16]8.7617,[17]8.6791,[18]8.8135,[19]8.5408,[20]8.3868,[21]8.3814,[22]8.0731,[23]7.7960,[24]7.6181,[25]7.3896,[26]7.2858,[27]7.1902,[28]7.1038,[29]7.1737,[30]7.1573,[31]7.1504,[32]7.0949,[33]7.1199,[34]7.1658,[35]7.2052,[36]7.3079,[37]7.2762,[38]7.3041,[39]7.2843,[40]7.2991,[41]7.2945,[42]7.2208,[43]7.2553,[44]7.2124,[45]7.3190,[46]7.3427,[47]7.3360,[48]7.3206,[49]7.2950,[50]7.3607,[51]7.4344,[52]7.4000,[53]7.5139,[54]7.5210,[55]7.5206,[56]7.5684,[57]7.5831,[58]7.5933,[59]7.5393,[60]7.5898,[61]7.6529,[62]7.7069,[63]7.7720,[64]7.8413,[65]7.8296,[66]7.8277,[67]7.8087,[68]7.8351,[69]7.8853,[70]7.9126,[71]7.8960,[72]7.8540,[73]7.8263,[74]7.8362,[75]7.7787,[76]7.7507,[77]7.6992,[78]7.7113,[79]7.7276,[80]7.7405,[81]7.7354,[82]7.7526,[83]7.7624,[84]7.7487,[85]7.7436,[86]7.7346,[87]7.8240,[88]7.8164,[89]7.8366,[90]7.8449,[91]7.8379,[92]7.8383,[93]7.8281,[94]7.8329,[95]7.8276,[96]7.8583,[97]7.8737,[98]7.8766,[99]7.8749,[100]7.8598,[101]7.8589,[102]7.8825,[103]7.9153,[104]7.9718,[105]7.9641,[106]8.0166,[107]8.0411,[108]8.0494,[109]8.0896,[110]8.1375,[111]8.1515,[112]8.1137,[113]8.1044,[114]8.0952,[115]8.0772,[116]8.0743,[117]8.0643,[118]8.0438,[119]8.0250,[120]7.9950,[121]7.9713,[122]7.9441,[123]7.9163,[124]7.8621,[125]7.8154,[126]7.7847,[127]7.7520,[128]7.7520,[129]7.7509,[130]7.7584,[131]7.7590,[132]7.7372,[133]7.7097,[134]7.7157,[135]7.7059,[136]7.7096,[137]7.7207,[138]7.7470,[139]7.7690,[140]7.7515,[141]7.7154,[142]7.6819,[143]7.6296,[144]7.5938,[145]7.5441,[146]7.5113,[147]7.4822,[148]7.4584,[149]7.4323,[150]7.4114,[151]7.3773,[152]7.3468,[153]7.3190,[154]7.2849,[155]7.2581,[156]7.2436,[157]7.2149,[158]7.2123,[159]7.1853,[160]7.1736,[161]7.1930,[162]7.1944,[163]7.2152,[164]7.2231,[165]7.2550,[166]7.2869,[167]7.3102,[168]7.3540,[169]7.3755,[170]7.4089,[171]7.4476,[172]7.4573,[173]7.4620,[174]7.4610,[175]7.4836,[176]7.4910,[177]7.4986,[178]7.5098,[179]7.5079,[180]7.5188,[181]7.5233,[182]7.5326,[183]7.5573,[184]7.5695,[185]7.5824,[186]7.5844,[187]7.6069,[188]7.6232,[189]7.6350,[190]7.6462,[191]7.6373,[192]7.6259,[193]7.6156,[194]7.6106,[195]7.6455,[196]7.6438,[197]7.6479,[198]7.6358,[199]7.6274,[200]7.6108,[201]7.5794,[202]7.5717,[203]7.5371,[204]7.5324,[205]7.5233,[206]7.5085,[207]7.4972,[208]7.5045,[209]7.5122,[210]7.5131,[211]7.4948,[212]7.4682,[213]7.4592,[214]7.4617,[215]7.4480,[216]7.4518,[217]7.4322,[218]7.4167,[219]7.4102,[220]7.4058,[221]7.3850,[222]7.3709,[223]7.3576,[224]7.3493,[225]7.3522,[226]7.3436,[227]7.3198,[228]7.3134,[229]7.3022,[230]7.2868,[231]7.2873,[232]7.2918,[233]7.3000,[234]7.3005,[235]7.3161,[236]7.3193,[237]7.3355,[238]7.3477,[239]7.3573,[240]7.3605,[241]7.3645,[242]7.3793,[243]7.3826,[244]7.4030,[245]7.4255,[246]7.4274,[247]7.4276,[248]7.4372,[249]7.4261,[250]7.3997,[251]7.3887,[252]7.3680,[253]7.3593,[254]7.3584,[255]7.3656,[256]7.3644,[257]7.3653,[258]7.3607,[259]7.3586,[260]7.3505,[261]7.3348,[262]7.3223,[263]7.3181,[264]7.3031,[265]7.3033,[266]7.2874,[267]7.2800,[268]7.2723,[269]7.2663,[270]7.2570,[271]7.2509,[272]7.2522,[273]7.2265,[274]7.2096,[275]7.2139,[276]7.2146,[277]7.2002,[278]7.1953,[279]7.1980,[280]7.2106,[281]7.2210,[282]7.2333,[283]7.2392,[284]7.2417,[285]7.2586,[286]7.2584,[287]7.2669,[288]7.2588,[289]7.2535,[290]7.2528,[291]7.2558,[292]7.2508,[293]7.2517,[294]7.2564,[295]7.2560,[296]7.2576,[297]7.2562,[298]7.2514,[299]7.2561,[300]7.2595,[301]7.2535,[302]7.2462,[303]7.2483,[304]7.2376,[305]7.2403,[306]7.2528,[307]7.2602,[308]7.2602,[309]7.2695,[310]7.2607,[311]7.2611,[312]7.2701,[313]7.2854,[314]7.3038,[315]7.3074,[316]7.3150,[317]7.3100,[318]7.3121,[319]7.3037,[320]7.2952,[321]7.2946,[322]7.2932,[323]7.2850,[324]7.2912,[325]7.2795,[326]7.2812,[327]7.2825,[328]7.2752,[329]7.2690,[330]7.2534,[331]7.2593,[332]7.2568,[333]7.2518,[334]7.2483,[335]7.2343,[336]7.2305,[337]7.2225,[338]7.2168,[339]7.2128,[340]7.2161,[341]7.2155,[342]7.2190,[343]7.2267,[344]7.2382,[345]7.2416,[346]7.2436,[347]7.2470,[348]7.2545,[349]7.2606,[350]7.2634,[351]7.2663,[352]7.2726,[353]7.2941,[354]7.3126,[355]7.3299,[356]7.3420,[357]7.3602,[358]7.3746,[359]7.3920,[360]7.4038,[361]7.4081,[362]7.4218,[363]7.4288,[364]7.4291,[365]7.4385,[366]7.4519,[367]7.4621,[368]7.4703,[369]7.4767,[370]7.4875,[371]7.5017,[372]7.5163,[373]7.5175,[374]7.5126,[375]7.5047,[376]7.5086,[377]7.5259,[378]7.5400,[379]7.5385,[380]7.5353,[381]7.5278,[382]7.5303,[383]7.5364,[384]7.5388,[385]7.5414,[386]7.5440,[387]7.5499,[388]7.5562,[389]7.5588,[390]7.5467,[391]7.5348,[392]7.5273,[393]7.5312,[394]7.5318,[395]7.5288,[396]7.5301,[397]7.5429,[398]7.5402,[399]7.5345,[400]7.5446,[401]7.5432,[402]7.5352,[403]7.5381,[404]7.5355,[405]7.5383,[406]7.5421,[407]7.5421,[408]7.5367,[409]7.5421,[410]7.5333,[411]7.5328,[412]7.5217,[413]7.5218,[414]7.5311,[415]7.5373,[416]7.5389,[417]7.5353,[418]7.5381,[419]7.5325,[420]7.5329,[421]7.5349,[422]7.5320,[423]7.5361,[424]7.5308,[425]7.5165,[426]7.5184,[427]7.5167,[428]7.5118,[429]7.5018,[430]7.5016,[431]7.4934,[432]7.4873,[433]7.4852,[434]7.4847,[435]7.4713,[436]7.4753,[437]7.4711,[438]7.4659,[439]7.4637,[440]7.4614,[441]7.4642,[442]7.4650,[443]7.4803,[444]7.4849,[445]7.4829,[446]7.4803,[447]7.4787,[448]7.4837,[449]7.4830,[450]7.4803,[451]7.4814,[452]7.4877,[453]7.4917,[454]7.4918,[455]7.4949,[456]7.4897,[457]7.4921,[458]7.4799,[459]7.4861,[460]7.4946,[461]7.4923,[462]7.4919,[463]7.4862,[464]7.4903,[465]7.5053,[466]7.5125,[467]7.5117,[468]7.5133,[469]7.5104,[470]7.5090,[471]7.5053,[472]7.4992,[473]7.4918,[474]7.4884,[475]7.4870,[476]7.4857,[477]7.4776,[478]7.4751,[479]7.4695,[480]7.4704,[481]7.4713,[482]7.4749,[483]7.4695,[484]7.4701,[485]7.4656,[486]7.4692,[487]7.4761,[488]7.4784,[489]7.4800,[490]7.4841,[491]7.4816,[492]7.4829,[493]7.4890,[494]7.4904,[495]7.4871,[496]7.4845,[497]7.4849,[498]7.4822,[499]7.4836,[500]7.4811,[501]7.4752,[502]7.4762,[503]7.4787,[504]7.4771,[505]7.4722,[506]7.4737,[507]7.4761,[508]7.4822,[509]7.4786,[510]7.4791,[511]7.4746,[512]7.4771,[513]7.4766,[514]7.4786,[515]7.4771,[516]7.4803,[517]7.4832,[518]7.4780,[519]7.4801,[520]7.4853,[521]7.4877,[522]7.4977,[523]7.4953,[524]7.4886,[525]7.4893,[526]7.4906,[527]7.4942,[528]7.4911,[529]7.4814,[530]7.4713,[531]7.4785,[532]7.4709,[533]7.4653,[534]7.4477,[535]7.4383,[536]7.4371,[537]7.4408,[538]7.4446,[539]7.4431,[540]7.4492,[541]7.4507,[542]7.4566,[543]7.4649,[544]7.4726,[545]7.4720,[546]7.4806,[547]7.4839,[548]7.4732,[549]7.4691,[550]7.4604,[551]7.4618,[552]7.4648,[553]7.4710,[554]7.4731,[555]7.4727,[556]7.4713,[557]7.4646,[558]7.4678,[559]7.4703,[560]7.4750,[561]7.4814,[562]7.4941,[563]7.4882,[564]7.4896,
Final estimate: PPL = 7.4896 +/- 0.04778

llama_print_timings:        load time =     893.21 ms
llama_print_timings:      sample time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings: prompt eval time =   58848.32 ms / 288768 tokens (    0.20 ms per token,  4906.99 tokens per second)
llama_print_timings:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings:       total time =   62023.33 ms / 288769 tokens
</code>
</details>

<details>
<summary>./bin/llama-perplexity -m llama-3.1-instruct-iq4kss.gguf -f ../tests/wiki.test.raw -t 1 -ngl 100 -fa -ctk q5_1 -ctv q5_0l</summary>
<code>
llama_kv_cache_init:      CUDA0 KV buffer size =    92.00 MiB
llama_new_context_with_model: KV self size  =   92.00 MiB, K (q5_1):   48.00 MiB, V (q5_0):   44.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     1.96 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =   266.50 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =    12.01 MiB
llama_new_context_with_model: graph nodes  = 806
llama_new_context_with_model: graph splits = 2

system_info: n_threads = 1 / 32 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | 
perplexity: tokenizing the input ..
perplexity: tokenization took 146.542 ms
perplexity: calculating perplexity over 564 chunks, n_ctx=512, batch_size=2048, n_seq=4
perplexity: 0.63 seconds per pass - ETA 1.47 minutes
[1]4.3405,[2]5.5099,[3]6.0314,[4]6.3570,[5]6.7371,[6]7.0748,[7]7.4564,[8]7.9839,[9]8.6380,[10]8.8331,[11]8.9924,[12]9.0971,[13]9.5138,[14]9.1123,[15]9.0236,[16]8.7697,[17]8.6805,[18]8.8077,[19]8.5344,[20]8.3874,[21]8.3766,[22]8.0658,[23]7.7916,[24]7.6164,[25]7.3842,[26]7.2792,[27]7.1844,[28]7.0971,[29]7.1678,[30]7.1542,[31]7.1464,[32]7.0900,[33]7.1167,[34]7.1636,[35]7.2032,[36]7.3050,[37]7.2715,[38]7.3009,[39]7.2807,[40]7.2959,[41]7.2919,[42]7.2182,[43]7.2519,[44]7.2063,[45]7.3137,[46]7.3355,[47]7.3302,[48]7.3152,[49]7.2899,[50]7.3547,[51]7.4284,[52]7.3953,[53]7.5116,[54]7.5166,[55]7.5167,[56]7.5652,[57]7.5822,[58]7.5914,[59]7.5385,[60]7.5890,[61]7.6513,[62]7.7063,[63]7.7700,[64]7.8389,[65]7.8279,[66]7.8246,[67]7.8051,[68]7.8313,[69]7.8818,[70]7.9101,[71]7.8918,[72]7.8497,[73]7.8221,[74]7.8332,[75]7.7752,[76]7.7461,[77]7.6960,[78]7.7099,[79]7.7259,[80]7.7391,[81]7.7337,[82]7.7512,[83]7.7609,[84]7.7474,[85]7.7426,[86]7.7338,[87]7.8233,[88]7.8163,[89]7.8370,[90]7.8458,[91]7.8388,[92]7.8383,[93]7.8300,[94]7.8353,[95]7.8301,[96]7.8611,[97]7.8772,[98]7.8809,[99]7.8802,[100]7.8651,[101]7.8648,[102]7.8886,[103]7.9214,[104]7.9777,[105]7.9696,[106]8.0222,[107]8.0466,[108]8.0554,[109]8.0949,[110]8.1419,[111]8.1558,[112]8.1176,[113]8.1082,[114]8.0990,[115]8.0806,[116]8.0780,[117]8.0684,[118]8.0474,[119]8.0282,[120]7.9980,[121]7.9749,[122]7.9479,[123]7.9206,[124]7.8673,[125]7.8196,[126]7.7894,[127]7.7561,[128]7.7576,[129]7.7565,[130]7.7639,[131]7.7649,[132]7.7434,[133]7.7153,[134]7.7212,[135]7.7118,[136]7.7156,[137]7.7265,[138]7.7522,[139]7.7743,[140]7.7560,[141]7.7191,[142]7.6855,[143]7.6338,[144]7.5982,[145]7.5495,[146]7.5164,[147]7.4873,[148]7.4638,[149]7.4379,[150]7.4171,[151]7.3830,[152]7.3527,[153]7.3248,[154]7.2907,[155]7.2646,[156]7.2502,[157]7.2215,[158]7.2191,[159]7.1921,[160]7.1803,[161]7.2005,[162]7.2022,[163]7.2226,[164]7.2300,[165]7.2621,[166]7.2937,[167]7.3171,[168]7.3609,[169]7.3827,[170]7.4161,[171]7.4551,[172]7.4647,[173]7.4693,[174]7.4683,[175]7.4908,[176]7.4983,[177]7.5060,[178]7.5173,[179]7.5156,[180]7.5265,[181]7.5308,[182]7.5402,[183]7.5648,[184]7.5771,[185]7.5904,[186]7.5931,[187]7.6155,[188]7.6323,[189]7.6442,[190]7.6555,[191]7.6467,[192]7.6355,[193]7.6245,[194]7.6199,[195]7.6545,[196]7.6530,[197]7.6571,[198]7.6448,[199]7.6367,[200]7.6199,[201]7.5885,[202]7.5809,[203]7.5463,[204]7.5411,[205]7.5316,[206]7.5170,[207]7.5058,[208]7.5129,[209]7.5204,[210]7.5212,[211]7.5031,[212]7.4767,[213]7.4675,[214]7.4698,[215]7.4562,[216]7.4600,[217]7.4404,[218]7.4250,[219]7.4179,[220]7.4133,[221]7.3923,[222]7.3785,[223]7.3648,[224]7.3571,[225]7.3594,[226]7.3510,[227]7.3275,[228]7.3214,[229]7.3098,[230]7.2946,[231]7.2951,[232]7.2995,[233]7.3076,[234]7.3078,[235]7.3233,[236]7.3268,[237]7.3430,[238]7.3548,[239]7.3643,[240]7.3674,[241]7.3717,[242]7.3864,[243]7.3901,[244]7.4106,[245]7.4330,[246]7.4352,[247]7.4355,[248]7.4452,[249]7.4339,[250]7.4073,[251]7.3962,[252]7.3755,[253]7.3671,[254]7.3663,[255]7.3735,[256]7.3726,[257]7.3739,[258]7.3696,[259]7.3674,[260]7.3594,[261]7.3435,[262]7.3307,[263]7.3267,[264]7.3116,[265]7.3115,[266]7.2958,[267]7.2883,[268]7.2805,[269]7.2747,[270]7.2653,[271]7.2595,[272]7.2615,[273]7.2360,[274]7.2190,[275]7.2233,[276]7.2242,[277]7.2101,[278]7.2050,[279]7.2078,[280]7.2205,[281]7.2307,[282]7.2432,[283]7.2493,[284]7.2518,[285]7.2689,[286]7.2690,[287]7.2770,[288]7.2688,[289]7.2638,[290]7.2630,[291]7.2657,[292]7.2608,[293]7.2616,[294]7.2664,[295]7.2660,[296]7.2676,[297]7.2660,[298]7.2611,[299]7.2658,[300]7.2691,[301]7.2631,[302]7.2555,[303]7.2575,[304]7.2465,[305]7.2488,[306]7.2614,[307]7.2686,[308]7.2687,[309]7.2781,[310]7.2692,[311]7.2695,[312]7.2790,[313]7.2944,[314]7.3129,[315]7.3165,[316]7.3243,[317]7.3194,[318]7.3214,[319]7.3129,[320]7.3043,[321]7.3035,[322]7.3021,[323]7.2939,[324]7.3000,[325]7.2885,[326]7.2903,[327]7.2916,[328]7.2844,[329]7.2783,[330]7.2623,[331]7.2681,[332]7.2655,[333]7.2606,[334]7.2570,[335]7.2431,[336]7.2394,[337]7.2314,[338]7.2256,[339]7.2216,[340]7.2245,[341]7.2238,[342]7.2271,[343]7.2348,[344]7.2463,[345]7.2496,[346]7.2520,[347]7.2554,[348]7.2628,[349]7.2688,[350]7.2713,[351]7.2740,[352]7.2803,[353]7.3016,[354]7.3198,[355]7.3373,[356]7.3493,[357]7.3675,[358]7.3819,[359]7.3994,[360]7.4108,[361]7.4151,[362]7.4286,[363]7.4356,[364]7.4360,[365]7.4456,[366]7.4591,[367]7.4695,[368]7.4774,[369]7.4839,[370]7.4945,[371]7.5087,[372]7.5233,[373]7.5243,[374]7.5193,[375]7.5113,[376]7.5153,[377]7.5326,[378]7.5468,[379]7.5454,[380]7.5421,[381]7.5349,[382]7.5374,[383]7.5436,[384]7.5462,[385]7.5489,[386]7.5514,[387]7.5573,[388]7.5636,[389]7.5661,[390]7.5540,[391]7.5419,[392]7.5342,[393]7.5382,[394]7.5388,[395]7.5359,[396]7.5373,[397]7.5501,[398]7.5472,[399]7.5416,[400]7.5516,[401]7.5504,[402]7.5425,[403]7.5453,[404]7.5426,[405]7.5454,[406]7.5492,[407]7.5495,[408]7.5442,[409]7.5494,[410]7.5408,[411]7.5404,[412]7.5293,[413]7.5293,[414]7.5384,[415]7.5448,[416]7.5464,[417]7.5428,[418]7.5455,[419]7.5398,[420]7.5403,[421]7.5426,[422]7.5397,[423]7.5441,[424]7.5387,[425]7.5245,[426]7.5265,[427]7.5247,[428]7.5198,[429]7.5097,[430]7.5091,[431]7.5010,[432]7.4949,[433]7.4928,[434]7.4924,[435]7.4790,[436]7.4831,[437]7.4789,[438]7.4740,[439]7.4718,[440]7.4698,[441]7.4727,[442]7.4735,[443]7.4887,[444]7.4934,[445]7.4915,[446]7.4888,[447]7.4874,[448]7.4926,[449]7.4919,[450]7.4893,[451]7.4907,[452]7.4969,[453]7.5009,[454]7.5010,[455]7.5042,[456]7.4990,[457]7.5014,[458]7.4892,[459]7.4954,[460]7.5038,[461]7.5016,[462]7.5014,[463]7.4957,[464]7.4998,[465]7.5148,[466]7.5224,[467]7.5217,[468]7.5232,[469]7.5204,[470]7.5190,[471]7.5152,[472]7.5089,[473]7.5016,[474]7.4983,[475]7.4969,[476]7.4956,[477]7.4874,[478]7.4849,[479]7.4793,[480]7.4800,[481]7.4809,[482]7.4844,[483]7.4791,[484]7.4798,[485]7.4751,[486]7.4786,[487]7.4855,[488]7.4877,[489]7.4894,[490]7.4936,[491]7.4910,[492]7.4924,[493]7.4982,[494]7.4994,[495]7.4962,[496]7.4936,[497]7.4939,[498]7.4913,[499]7.4926,[500]7.4901,[501]7.4841,[502]7.4853,[503]7.4876,[504]7.4860,[505]7.4811,[506]7.4824,[507]7.4848,[508]7.4912,[509]7.4876,[510]7.4882,[511]7.4836,[512]7.4860,[513]7.4854,[514]7.4873,[515]7.4861,[516]7.4892,[517]7.4920,[518]7.4865,[519]7.4887,[520]7.4941,[521]7.4963,[522]7.5060,[523]7.5035,[524]7.4966,[525]7.4972,[526]7.4984,[527]7.5018,[528]7.4988,[529]7.4892,[530]7.4790,[531]7.4862,[532]7.4787,[533]7.4731,[534]7.4555,[535]7.4464,[536]7.4453,[537]7.4493,[538]7.4530,[539]7.4515,[540]7.4573,[541]7.4590,[542]7.4648,[543]7.4733,[544]7.4810,[545]7.4805,[546]7.4891,[547]7.4924,[548]7.4816,[549]7.4777,[550]7.4689,[551]7.4703,[552]7.4733,[553]7.4794,[554]7.4811,[555]7.4806,[556]7.4792,[557]7.4725,[558]7.4756,[559]7.4781,[560]7.4830,[561]7.4896,[562]7.5023,[563]7.4964,[564]7.4978,
Final estimate: PPL = 7.4978 +/- 0.04775

llama_print_timings:        load time =     862.72 ms
llama_print_timings:      sample time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings: prompt eval time =   58481.55 ms / 288768 tokens (    0.20 ms per token,  4937.76 tokens per second)
llama_print_timings:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings:       total time =   61661.41 ms / 288769 tokens
</code>
</details>

It is a matter of having `GGML_COPY` available, which I implemented for `IQ4_NL` a while ago. It is also available in mainline `llama.cpp` CUDA code, except that there someone has disabled it for whatever reason. It is enabled here as you can see from the logs above.

I see now that performance on CUDA is pretty much the same:

| model                          |       size |     params | backend    | ngl | type_k | type_v | fa |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | -: | ------------: | ---------------: |
| llama 8B IQ4_KS - 4.25 bpw     |   4.14 GiB |     8.03 B | CUDA       |  99 |   q5_1 |   q5_0 |  1 |        pp8192 |   4777.42 ± 3.50 |
| llama 8B IQ4_KS - 4.25 bpw     |   4.14 GiB |     8.03 B | CUDA       |  99 |   q8_0 | iq4_nl |  1 |        pp8192 |   4757.62 ± 2.13 |

It is on the CPU where `-ctk q8_0 -ctv iq4_nl` is quite a bit faster.

---

👤 **Nexesenex** commented on **2024-10-19** at **19:44:00**

Well, I can execute PPL tests on both mainline and IK_Llama with V cache in iq4_nl, no problem with that.

But if I want to use Llama server, or integrate it into my KoboldCPP fork, here's what I get instead of a generation :

```
INFO [            update_slots] kv cache rm [p0, end) | tid="19596" timestamp=1729366649 id_slot=0 id_task=0 p0=0
INFO [            update_slots] kv cache rm [p0, end) | tid="19596" timestamp=1729366649 id_slot=0 id_task=0 p0=1024
INFO [            update_slots] kv cache rm [p0, end) | tid="19596" timestamp=1729366650 id_slot=0 id_task=0 p0=2048
INFO [            update_slots] kv cache rm [p0, end) | tid="19596" timestamp=1729366650 id_slot=0 id_task=0 p0=3072
Unsupported KV type combination for head_size 128.
Supported combinations:
  - K == q4_0, V == q4_0,  4.50 BPV
  - K == q8_0, V == q8_0,  8.50 BPV
  - K == f16,  V == f16,  16.00 BPV
Compile with GGML_CUDA_FA_ALL_QUANTS for all combinations of q4_0, q4_1, q5_0, q5_1, q8_0, and f16.
Q:\GitHub\ik_llama.cpp.fks\ggml\src\ggml-cuda\fattn-common.cuh:576: fatal error

Q:\LLAMA_IK>pause
Press any key to continue . . .
```

My fork of KoboldCPP is compiled with the tag FA_ALL_QUANTS, and the KVQ combos I use with the legacy KV quants are all working, iq4_nl is not.

```
Processing Prompt [BLAS] (13200 / 13200 tokens)
Generating (1 / 512 tokens)Unsupported KV type combination for head_size 128.
Supported combinations:
  - K == q4_0, V == q4_0,  4.50 BPV
  - K == q8_0, V == q8_0,  8.50 BPV
  - K == f16,  V == f16,  16.00 BPV
Compile with GGML_CUDA_FA_ALL_QUANTS for all combinations of q4_0, q4_1, q5_0, q5_1, q8_0, and f16.
Q:\GitHub\kobold.cpp\ggml\src\ggml-cuda\fattn-common.cuh:576: fatal error

Q:\Kob\KoboldNew\Dist>pause
Press any key to continue . . .
```

Which makes sense, no FA kernel being available, thus compiled, for such a KV cache.

---

👤 **saood06** commented on **2024-10-20** at **07:05:51**

> What happens with `-ctv q4_0` ?

`-fa -ctk q8_0 -ctv q4_0` produced the same garbage output.

>Is FA running on the GPU or on the CPU?

I don't know. Is it possible that it is running on both given that the KV is allocated per layer? I had to recompile with GGML_CUDA_FA_ALL_QUANTS because initially it gave me the issue of "Unsupported KV type combination for head_size 128. ... fattn-common.cuh:576: fatal error".

I also tested `-fa -ctk q8_0 -ctv q4_0 -nkvo`, because I thought maybe putting all of the KV cache on the CPU would fix it, but this resulted in an even worse output. Instead of something like " to, of for. for" as it did before for Q8_0/Q8_0 and Q8_0/Q4_0. It was spamming [control_36]. The ten 10 tokens in probs were [control_36],[control_20],[IMG],[control_32],[control_24],[control_16],[control_18],[/INST],[control_22],[MIDDLE], with them all showing a probability of null.

>This is because I have changed the bit arrangement in `Q8_0` when quantization is done during inference, with the result that `Q8_0` cannot be used for V cache when FA is running on the CPU.

You mentioned this in [#76](https://github.com/ikawrakow/ik_llama.cpp/issues/76) but as the error above says this is a head size of 128. If that's the case, shouldn't -ctk q8_0 -ctv q8_0 work for this model?

---

👤 **ikawrakow** commented on **2024-10-20** at **09:04:35**

> My fork of KoboldCPP is compiled with the tag FA_ALL_QUANTS, and the KVQ combos I use with the legacy KV quants are all working, iq4_nl is not.

Yes, sorry, it needed some extra things to also work for TG. See [#99](https://github.com/ikawrakow/ik_llama.cpp/issues/99) that enables `IQ4_NL` for V-cache when attention head size is 128.

---

👤 **ikawrakow** commented on **2024-10-20** at **09:11:14**

> You mentioned this in https://github.com/ikawrakow/ik_llama.cpp/pull/76 but as the error above says this is a head size of 128. If that's the case, shouldn't -ctk q8_0 -ctv q8_0 work for this model?

OK, then, it is something else. The question is why does it work for @Nexesenex with this model? The problem is that Mistral Large is just too large for the computers I have available. It would be useful to have a repro with a smaller model so I can debug the issue. Can you post the quantization types used in your model? `llama.cpp` outputs something like this when loading the model:
```
llama_model_loader: - type  f32:   66 tensors
llama_model_loader: - type q6_K:    1 tensors
llama_model_loader: - type iq5_k:   32 tensors
llama_model_loader: - type iq4_ks:  193 tensors
```

---

👤 **saood06** commented on **2024-10-20** at **18:41:36**

>The question is why does it work for @Nexesenex with this model? 

I don't think he is partially offloading it.
@Nexesenex How many layers are you offloading?

>Can you post the quantization types used in your model?
 ```
llama_model_loader: - type  f32:  177 tensors
llama_model_loader: - type q5_K:   88 tensors
llama_model_loader: - type q6_K:    1 tensors
llama_model_loader: - type iq4_xs:  529 tensors
 ```

Also the other thing I noted don't know if it is at all relevant was running Q8/Q8 on llama.cpp had CUDA0 compute buffer size =   436.05 MiB, while ik_llama.cpp had CUDA0 compute buffer size =   474.39 MiB.

---

👤 **Nexesenex** commented on **2024-10-20** at **23:37:25**

Well, I merged the IQ4_NL PRs (improve quant speed, and token generation) on my KCPP fork and tested with success V cache IQ4_NL in full offload on Mistral 123b IQ4_3S/IQ4_XS mix with K cache q8, 5.1, 5.0. K cache IQ4_NL doesn't seem to work, it produces gibberish.

Here's what works for me :
https://github.com/Nexesenex/croco.cpp/tree/qkv

On IK LLama, I didn't make it work, surprisingly, despite trying 2 different compiling (one with the PR, one with some edits on my branch nex_3).

As for my model, Mistral 123b IQ3/IQ4 mix, it's quantized with that tensor config :

llama_model_loader: - type  f32:  177 tensors
llama_model_loader: - type q6_K:   89 tensors
llama_model_loader: - type iq3_xxs:   88 tensors
llama_model_loader: - type iq3_s:  110 tensors
llama_model_loader: - type iq4_xs:  331 tensors

---

👤 **ikawrakow** commented on **2024-10-21** at **05:43:15**

@Nexesenex 

> K cache IQ4_NL doesn't seem to work, it produces gibberish.

`IQ4_NL` for K-cache is not supposed to work (and I'm surprised it doesn't crash or stop with error message). To have `IQ4_NL` for K-cache one needs to also implement a dot product, which I didn't feel like doing considering that < 5 bpw K-cache is not very useful (and I think it would be better to disable the `Q4_0 + Q4_0` KV-cache combination as it is way off the mark).

To make sure I understand correctly: you added the `IQ4_NL` V-cache related changes to your `KCPP`, and it work there. But it does not work with `ik_llama.cpp`?

---

👤 **Nexesenex** commented on **2024-10-21** at **06:06:12**

Hey @ikawrakow 

I agree with you, I'm always thinking "wtf" when people are using KV Q4_0 or Q4_1, my daily combo being q5_1/q5_0 when I lacked of VRAM (I don't have patience for less than full offload).

-> and I don't really lack of VRAM anymore - I just pushed to 64GB - except for 123b full context, thus the use of V iq4_nl if I want to hit 128k with a smaller quant with the best ratio between model loss and KVQuant loss, I guess I can now go to less than 3.20 PPL 512 for 128k context.

But you have a lot of folks running on such cache still because they have 6-8GB of VRAM and want to run Gemma v2 for example, and if that's not too much of hassle for you to make that dot product, simply switching them on IQ4_NL would grant them a whole 1.2% of perplexity reduction (on L3 8B) accordingly to what I tested on KVquant iq4_nk vs KVquant Q4_0, and even 1.1% compared to K q4_0 and V iq4_nl.

As for adding on KCPP, yes, i've been thorough so it would work. While on IK_L, I just compiled what you offered, failed, made a few edits which "made sense", failed again, and dropped it. I'm sure it works, but I'm missing something I didn't miss on KCPP. Now that I slept, I will try again.

---

👤 **Nexesenex** commented on **2024-10-21** at **06:19:14**

Edit:
Fresh as a flower, I recompiled, launched Llama_IK main, and it worked like a charm in generation (K q8_0, V iq4_nl). Dunno what I did different yesterday, but I was exhausted. So forget my report about it not working.

Also, I noticed something yesterday in ggml\src\ggml-cuda\fattn.cu

You have on IK_L,  ggml_tensor * Q = dst->src[1];

```
static void ggml_cuda_flash_attn_ext_vec_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[1];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];
```

But on mainline,  ggml_tensor * Q = dst->src[0];

```
static void ggml_cuda_flash_attn_ext_vec_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];
```

Is this normal? (I'm sorry if it sounds silly.. I'm no dev ^^)

---

👤 **ikawrakow** commented on **2024-10-21** at **10:12:07**

> You have on IK_L, ggml_tensor * Q = dst->src[1];
>
>static void ggml_cuda_flash_attn_ext_vec_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
>    ggml_tensor * Q = dst->src[1];
>    ggml_tensor * K = dst->src[1];
>    ggml_tensor * V = dst->src[2];

Yes, this is a silly typo. `git blame` tells me that this line comes from Johannes. It must have been changed on mainline after I last merged. But at the end it is just a thing of preventing confusion, as `Q->ne[0] = K->ne[0]` (otherwise one wouldn't be able to do the matrix multiplication), so it is irrelevant for the actual functioning of the code.  I'm still going to change it to avoid future second guessing.

---

👤 **saood06** commented on **2024-10-21** at **17:52:43**

> < 5 bpw K-cache is not very useful (and I think it would be better to disable the `Q4_0 + Q4_0` KV-cache combination as it is way off the mark).

This isn't always the case. For example the original Command-R (35b) had no GQA and from my experience had no degradation going down to Q4/Q4 cache. Different models have very different sensitivity to KV cache quantization, and based on my limited experience this is much more varied than different model's sensitivity to weight quantization.

Going back to my original issue, it seems like it is working for Nexesenex because he is fully offloading it to the GPU and thus only using the FA kernel for the GPU which is the same as llama.cpp. 

I would test the no offload case, but I do not have the system resources to do so ( it only fits on my system via partial offload).

---

👤 **ikawrakow** commented on **2024-10-21** at **18:24:22**

> Going back to my original issue, it seems like it is working for Nexesenex because he is fully offloading it to the GPU and thus only using the FA kernel for the GPU which is the same as llama.cpp.

But partial offload works for me just fine. I just cannot test with Mistral Large. I can go up to 70B with the RAM/VRAM I have available (and that's how I run LLaMA-3.1-70B).

---

👤 **saood06** commented on **2024-10-22** at **01:43:54**

Was able to reproduce the issue with smaller models, it also does not seem to be exclusive to partial offloading, but also affects CPU only inference. 

Tested Q8/Q8, Q8/Q4, Q4/Q4 partially offloaded, and Q4/Q4 with no offload at all on this [model](https://huggingface.co/mradermacher/Midnight-Miqu-70B-v1.5-i1-GGUF/tree/main?show_file_info=Midnight-Miqu-70B-v1.5.i1-Q4_K_S.gguf). All resulted in all probs being null. Tested llama.cpp Q8/Q8 and Q4/Q4 with partial offload and all output is coherent and similar to non quantized. 

Also CPU only FA, with no KV quant on ik_llama.cpp also resulted in correct output.

Tested a Gemma-2 27B based model as well a bit and resulted in the same null probs output with partial offload. I was unable to compare full offload case as for my system I can fully offload with llama.cpp but ik_llama.cpp has ~500MB larger CUDA0 compute buffer size when fully offloaded vs llama.cpp which prevented me from being able to fully offload.

Mistral Large 2 was the only model where a quantized KV cache resulted in output that was incoherent but still not completely broken, everything else I tested is like the Mistral Large 2 nkvo case where it is all null probs.

Edit: I have a theory on what may be the issue will test and report back later.

---

👤 **ikawrakow** commented on **2024-10-22** at **06:33:15**

Well, in my case Miqu and Gemma-27b-Instruct both work fine.

Here is Miqu you linked hosted on a CPU with `AVX2`
<img width="1376" alt="Screenshot 2024-10-22 at 8 29 35 AM" src="https://github.com/user-attachments/assets/6d477975-de5f-4209-8720-b19cc20af538">

And here is Gemma2-27b hosted on a Zen4 CPU:
<img width="1376" alt="Screenshot 2024-10-22 at 7 38 13 AM" src="https://github.com/user-attachments/assets/d27f3f3a-0461-4ede-b916-c8f40adb5c74">

Both with partial offload as my GPU has only 16 GB VRAM. The screenshots are for `Q8_0 + Q8_0` KV-cache, but the other variants work as well.

So, not really sure what happens in your case. Hopefully your theory what might be wrong will find the problem.

---

👤 **saood06** commented on **2024-10-22** at **20:35:43**

> Hopefully your theory what might be wrong will find the problem.

My theory is that it is a platform/compiler issue, but so far I still haven't resolved it.

Only change I tested so far was changing long to long long as long everywhere besides Windows on x86-64 is 8 bytes, but on Windows it is 4 bytes. Long Long is 8 bytes everywhere.

The other thing that came to mind was struct packing and alignment, but I have not made any progress on finding any issues there, and I'm not sure I will.

I'm going to attempt to build it on Clang ( and maybe GCC if that doesn't resolve it) later.  

[This](https://stackoverflow.com/a/45514409) shows Clang on Windows can produce structs laid out compatible with MSVC or GCC but still not really sure how to choose which it is doing

The change I tested below:

```diff
diff --git a/ggml/src/iqk/iqk_mul_mat.cpp b/ggml/src/iqk/iqk_mul_mat.cpp
index b77d08b6..128be7cf 100644
--- a/ggml/src/iqk/iqk_mul_mat.cpp
+++ b/ggml/src/iqk/iqk_mul_mat.cpp
@@ -156,9 +156,9 @@ private:

 }

-bool iqk_mul_mat(long Nx, long Ny, long ne00,
-        int typeA, const void * A, long strideA,
-        int typeB, const void * B, long strideB,
+bool iqk_mul_mat(long long Nx, long long Ny, long long ne00,
+        int typeA, const void * A, long long strideA,
+        int typeB, const void * B, long long strideB,
         float * C, long stride_C, int ith, int nth) {

     MulMat mm;
@@ -181,10 +181,10 @@ bool iqk_mul_mat(long Nx, long Ny, long ne00,
     return true;
 }

-bool iqk_mul_mat_moe(long Nx, long Ny, long ne00, int ne11,
-        int typeA, const void * A, long strideA,
-        int typeB, const void * B, long strideB,
-        float * C, long nb1, long nb2, const void * vrow_mapping, int ith, int nth) {
+bool iqk_mul_mat_moe(long long Nx, long long Ny, long long ne00, int ne11,
+        int typeA, const void * A, long long strideA,
+        int typeB, const void * B, long long strideB,
+        float * C, long long nb1, long long nb2, const void * vrow_mapping, int ith, int nth) {
     const mmid_row_mapping * row_mapping = (const mmid_row_mapping *)vrow_mapping;
     assert(row_mapping != nullptr);

@@ -9059,11 +9059,11 @@ bool iqk_flash_attn_noalibi(int int_type_k,         // type of k

 #else  // IQK_IMPLEMENT

-bool iqk_mul_mat(int, long, long, long, int, const void *, long, int, const void *, long, float *, long, int, int) {
+bool iqk_mul_mat(int, long long, long long, long long, int, const void *, long long, int, const void *, long long, float *, long long, int, int) {
     return false;
 }

-bool iqk_mul_mat_moe(long, long, long, int, int, const void *, long, int, const void *, long, float *, long, long,
+bool iqk_mul_mat_moe(long long, long long, long long, int, int, const void *, long long, int, const void *, long long, float *, long long, long long,
         const void *, int, int) {
     return false;
 }
diff --git a/ggml/src/iqk/iqk_mul_mat.h b/ggml/src/iqk/iqk_mul_mat.h
index 6e27c614..61db23ed 100644
--- a/ggml/src/iqk/iqk_mul_mat.h
+++ b/ggml/src/iqk/iqk_mul_mat.h
@@ -11,15 +11,15 @@
 extern "C" {
 #endif

-bool iqk_mul_mat(long Nx, long Ny, long ne00,
-        int typeA, const void * A, long strideA,
-        int typeB, const void * B, long strideB,
-        float * C, long stride_C, int ith, int nth);
+bool iqk_mul_mat(long long Nx, long long Ny, long long ne00,
+        int typeA, const void * A, long long strideA,
+        int typeB, const void * B, long long strideB,
+        float * C, long long stride_C, int ith, int nth);

-bool iqk_mul_mat_moe(long Nx, long Ny, long ne00, int ne11,
-        int typeA, const void * A, long strideA,
-        int typeB, const void * B, long strideB,
-        float * C, long nb1, long nb2, const void * vrow_mapping, int ith, int nth);
+bool iqk_mul_mat_moe(long long Nx, long long Ny, long long ne00, int ne11,
+        int typeA, const void * A, long long strideA,
+        int typeB, const void * B, long long strideB,
+        float * C, long long nb1, long long nb2, const void * vrow_mapping, int ith, int nth);

 bool iqk_flash_attn_noalibi(int type_k,             // type of k
                             int type_v,             // type of v
```

---

👤 **saood06** commented on **2024-10-23** at **04:14:35**

Update, built it with GCC without CUDA, ran FA Q4/Q4 with the long long changes above. Same null probs result. Just realized I forgot to set [this](https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html#index-mms-bitfields). Will try again setting that to disable the MS layout for structs later. If this is the issue, then this also should be an easy way for you to reproduce it without having access to a Windows machine.

---

👤 **saood06** commented on **2024-10-23** at **22:00:48**

Compiled with the flag on GCC and still same null probs result. 

The fact that FA with FP16 KV works, but not anything quantized does narrow the scope of the issue but I have no more ideas anymore of what the issue could be.

---

👤 **ikawrakow** commented on **2025-01-30** at **15:44:38**

@saood06 There have been quite a few changes (and fixes) in the CPU FA implementation since October. Are you still observing the problem?

---

👤 **saood06** commented on **2025-02-11** at **20:01:35**

> [@saood06](https://github.com/saood06) There have been quite a few changes (and fixes) in the CPU FA implementation since October. Are you still observing the problem?

The problem can no longer be reproduced. I'm too lazy to git-bisect what fixed it, but last I looked into it was December 2, where I had a debug build and was was trying to find the issue. So it was fixed sometime between then and now.