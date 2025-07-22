### 🔀 [#283](https://github.com/ikawrakow/ik_llama.cpp/pull/283) - CUDA: better MoE implementation

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-03-24 |
| **Updated** | 2025-04-05 |

---

#### Description

This PR makes "indirect" matrix multiplications as used for MoE models inference reproducible on CUDA, and closes #249 

As a bonus, we get a ~10% PP speedup as measured with DeepSeek-Lite. I wouldn't be surprised if the benefit is even larger for DeepSeek-R1 as it has 4X more experts than DeepSeek-Lite.

The culprit for non-reproducible results and sluggish performance was the `k_copy_src1_to_contiguous` kernel, where an atomic increment is used, which is slow, on top of making the order in which the `src1` rows are added to the contiguous copy random. This kernel is invoked `n_as` times, where `n_as` is the total number of experts, making the `mul_mat_id` implementation quite inefficient.

---

#### 💬 Conversation

👤 **ubergarm** commented the **2025-03-24** at **16:48:31**:<br>

I pulled and built this branch and benchmarked speed vs main branch as well as llama-perplexity runs.

Seems to be about the same performance on the Thread Ripper Pro 24 core with 256GB RAM using single RTX A6000 48GB VRAM GPU between this PR and main.

Also two back to back runs of `llama-perplexity` gave what looks like the same result. This is also the same result I got a few days ago without this PR.

Let me know if there is some other condition or way to test. Thanks!

<details>

<summary>Benchmark and Testing Logs</summary>

## Benchmark
```bash
CUDA_VISIBLE_DEVICES="0," \
./build/bin/llama-bench \
    --model /mnt/raid/models/ubergarm/DeepSeek-R1-GGUF/DeepSeek-R1-IQ2_K_R4.gguf \
    -ctk q8_0 \
    -mla 2 -fa 1 \
    -amb 512 \
    -fmoe 1 \
    -p 512,4096 -n 0 \
    -gp 512,64 \
    -gp 4096,64 \
    -r 2 \
    --n-gpu-layers 63 \
    --override-tensor exps=CPU \
    --threads 24
```

#### Baseline

Device 0: NVIDIA RTX A6000, compute capability 8.6, VMM: yes

| model                          |       size |     params | backend    | ngl | type_k | fa | mla |   amb | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -: | --: | ----: | ---: | ------------: | ---------------: |
| deepseek2 671B IQ2_K_R4 - 2.375 bpw | 226.00 GiB |   672.05 B | CUDA       |  63 |   q8_0 |  1 |   2 |   512 |    1 |         pp512 |    105.50 ± 0.62 |
| deepseek2 671B IQ2_K_R4 - 2.375 bpw | 226.00 GiB |   672.05 B | CUDA       |  63 |   q8_0 |  1 |   2 |   512 |    1 |        pp4096 |     99.93 ± 0.04 |
| deepseek2 671B IQ2_K_R4 - 2.375 bpw | 226.00 GiB |   672.05 B | CUDA       |  63 |   q8_0 |  1 |   2 |   512 |    1 |    tg64@pp512 |     10.31 ± 0.00 |
| deepseek2 671B IQ2_K_R4 - 2.375 bpw | 226.00 GiB |   672.05 B | CUDA       |  63 |   q8_0 |  1 |   2 |   512 |    1 |   tg64@pp4096 |      9.66 ± 0.04 |

build: f9307d79 (3607)

#### PR `283` Test Case

Device 0: NVIDIA RTX A6000, compute capability 8.6, VMM: yes

| model                          |       size |     params | backend    | ngl | type_k | fa | mla |   amb | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -: | --: | ----: | ---: | ------------: | ---------------: |
| deepseek2 671B IQ2_K_R4 - 2.375 bpw | 226.00 GiB |   672.05 B | CUDA       |  63 |   q8_0 |  1 |   2 |   512 |    1 |         pp512 |    105.85 ± 0.35 |
| deepseek2 671B IQ2_K_R4 - 2.375 bpw | 226.00 GiB |   672.05 B | CUDA       |  63 |   q8_0 |  1 |   2 |   512 |    1 |        pp4096 |     99.64 ± 0.02 |
| deepseek2 671B IQ2_K_R4 - 2.375 bpw | 226.00 GiB |   672.05 B | CUDA       |  63 |   q8_0 |  1 |   2 |   512 |    1 |    tg64@pp512 |     10.24 ± 0.08 |
| deepseek2 671B IQ2_K_R4 - 2.375 bpw | 226.00 GiB |   672.05 B | CUDA       |  63 |   q8_0 |  1 |   2 |   512 |    1 |   tg64@pp4096 |      9.62 ± 0.01 |

build: 7f6980fa (3610)

## Perplexity
```bash
CUDA_VISIBLE_DEVICES="0," \
./build/bin/llama-perplexity \
    --model /mnt/raid/models/ubergarm/DeepSeek-R1-GGUF/DeepSeek-R1-IQ2_K_R4.gguf \
    -ctk q8_0 \
    -mla 2 -fa \
    -amb 512 \
    -fmoe \
    --ctx-size 512 \
    --ubatch-size 512 \
    -f wiki.test.raw \
    --seed 1337 \
    --n-gpu-layers 63 \
    --override-tensor exps=CPU \
    --threads 24
```

#### Basline
```bash
## I had run this command a few days ago
main: build = 3601 (3d6e25c8)

perplexity: 20.37 seconds per pass - ETA 47.62 minutes
[1]2.8167,[2]3.5984,[3]2.5279,[4]2.1350,[5]1.9307,[6]1.8199,[7]1.7183,[8]1.6549,[9]1.6132,[10]1.5715,[11]1.5652,[12]1.6259,[13]1.6478,[14]1.7798,[15]1.9153,[16]1.9692,[17]2.1392,[18]2.2755,[19]2.2279,[20]2.2171,[21]2.3203,[22]2.2886,[23]2.2519,[24]2.2700,[25]2.2320,[26]2.2026,[27]2.2543,[28]2.2624,[29]2.3195,[30]2.3504,[31]2.3870,[32]2.4029,[33]2.4421,[34]2.4923,[35]2.5471,[36]2.6029,[37]2.6384,[38]2.6881,[39]2.7250,[40]2.7885,[41]2.8333,[42]2.8477,[43]2.9012,[44]2.9163,[45]3.0018,[46]3.0529,[47]3.0155,[48]2.9704,[49]2.9533,[50]2.9794,[51]3.0260,[52]3.0432,[53]3.1013,[54]3.1143,[55]3.1468,[56]3.1829,[57]3.2004,[58]3.2455,[59]3.2565,[60]3.3071,[61]3.3500,[62]3.4085,[63]3.4443,[64]3.4925,[65]3.5020,[66]3.4960,[67]3.4727,[68]3.5045,[69]3.5053,[70]3.5287,[71]3.5449,[72]3.5590,[73]3.5715,[74]3.5914,[75]3.5693,[76]3.5179,[77]3.4743,[78]3.4715,[79]3.4516,[80]3.4385,[81]3.4028,[82]3.4083,[83]3.3817,[84]3.3448,[85]3.3113,[86]3.2904,[87]3.2976,[88]3.2723,[89]3.2646,[90]3.2395,[91]3.2150,[92]3.1917,[93]3.1638,[94]3.1410,[95]3.1215,[96]3.1248,[97]3.1335,[98]3.1231,[99]3.1061,[100]3.1060,[101]3.0979,[102]3.1176,[103]3.1448,[104]3.1673,[105]3.1652,[106]3.1920,[107]3.2174,[108]3.2381,[109]3.2746,[110]3.3091,[111]3.3311,[112]3.3003,[113]3.2870,[114]3.2635,[115]3.2465,[116]3.2384,[117]3.2167,[118]3.1937,[119]3.1713,[120]3.1487,[121]3.1329,[122]3.1128,[123]3.0950,[124]3.0722,[125]3.0524,[126]3.0345,[127]3.0218,[128]3.0145,[129]3.0055,[130]2.9943,[131]2.9862,[132]2.9922,[133]2.9999,[134]3.0062,[135]3.0185,[136]3.0349,[137]3.0503,[138]3.0577,[139]3.0696,[140]3.0682,[141]3.0675,[142]3.0642,[143]3.0624,[144]3.0560,[145]3.0458,[146]3.0428,[147]3.0450,[148]3.0424,[149]3.0424,[150]3.0349,[151]3.0310,[152]3.0262,[153]3.0201,[154]3.0184,[155]3.0218,[156]3.0224,[157]3.0273,[158]3.0364,[159]3.0374,[160]3.0464,[161]3.0545,[162]3.0632,[163]3.0686,[164]3.0893,[165]3.1137,[166]3.1324,[167]3.1459,[168]3.1722,[169]3.1956,[170]3.2185,[171]3.2428,[172]3.2243,[173]3.2042,[174]3.1909,[175]3.1779,[176]3.1654,[177]3.1541,[178]3.1408,[179]3.1267,[180]3.1301,[181]3.1442,[182]3.1594,[183]3.1742,[184]3.1882,[185]3.1979,[186]3.2146,[187]3.2298,[188]3.2433,[189]3.2538,[190]3.2533,[191]3.2597,[192]3.2620,[193]3.2666,[194]3.2868,[195]3.2961,[196]3.3094,[197]3.3196,[198]3.3230,[199]3.3280,[200]3.3258,[201]3.3412,[202]3.3351,[203]3.3396,[204]3.3417,[205]3.3418,[206]3.3442,[207]3.3534,[208]3.3635,[209]3.3729,[210]3.3721,[211]3.3663,[212]3.3666,[213]3.3746,[214]3.3760,[215]3.3822,[216]3.3823,[217]3.3756,[218]3.3754,[219]3.3761,[220]3.3743,[221]3.3739,[222]3.3731,[223]3.3745,[224]3.3794,[225]3.3812,[226]3.3714,[227]3.3702,[228]3.3716,[229]3.3757,[230]3.3812,[231]3.3870,[232]3.3788,[233]3.3715,[234]3.3735,[235]3.3734,[236]3.3822,[237]3.3904,[238]3.4001,[239]3.4104,[240]3.4189,[241]3.4301,[242]3.4457,[243]3.4594,[244]3.4676,[245]3.4795,[246]3.4902,[247]3.4876,[248]3.4827,[249]3.4802,[250]3.4725,[251]3.4688,[252]3.4704,[253]3.4731,[254]3.4793,[255]3.4855,[256]3.4890,[257]3.4906,[258]3.4907,[259]3.4927,[260]3.4949,[261]3.4954,[262]3.4931,[263]3.4987,[264]3.5010,[265]3.5011,[266]3.5027,[267]3.5054,[268]3.5099,[269]3.5128,[270]3.5109,[271]3.5089,[272]3.5014,[273]3.5018,[274]3.4945,[275]3.4831,[276]3.4719,[277]3.4732,[278]3.4836,[279]3.4894,[280]3.4974,[281]3.5045,[282]3.5104,[283]3.5171,[284]3.5233,[285]3.5375,[286]3.5392,[287]3.5420,[288]3.5462,[289]3.5486,[290]3.5395,[291]3.5314,[292]3.5335,[293]3.5346,[294]3.5327,[295]3.5317,[296]3.5342,[297]3.5356,[298]3.5404,[299]3.5472,[300]3.5502,[301]3.5536,[302]3.5554,[303]3.5564,[304]3.5546,[305]3.5669,[306]3.5741,[307]3.5855,[308]3.5734,[309]3.5676,[310]3.5575,[311]3.5611,[312]3.5644,[313]3.5713,[314]3.5734,[315]3.5763,[316]3.5771,[317]3.5780,[318]3.5784,[319]3.5792,[320]3.5834,[321]3.5835,[322]3.5852,[323]3.5914,[324]3.5913,[325]3.5967,[326]3.6011,[327]3.6050,[328]3.6073,[329]3.6086,[330]3.6146,[331]3.6183,[332]3.6224,[333]3.6204,[334]3.6199,[335]3.6193,[336]3.6187,[337]3.6194,[338]3.6192,[339]3.6215,[340]3.6248,[341]3.6304,[342]3.6399,[343]3.6496,[344]3.6548,[345]3.6471,[346]3.6407,[347]3.6381,[348]3.6305,[349]3.6265,[350]3.6247,[351]3.6297,[352]3.6453,[353]3.6544,[354]3.6677,[355]3.6766,[356]3.6830,[357]3.6952,[358]3.7059,[359]3.7091,[360]3.7151,[361]3.7246,[362]3.7337,[363]3.7394,[364]3.7462,[365]3.7520,[366]3.7629,[367]3.7718,[368]3.7787,[369]3.7863,[370]3.7948,[371]3.8090,[372]3.8188,[373]3.8216,[374]3.8250,[375]3.8296,[376]3.8427,[377]3.8541,[378]3.8562,[379]3.8550,[380]3.8515,[381]3.8561,[382]3.8620,[383]3.8653,[384]3.8698,[385]3.8737,[386]3.8797,[387]3.8852,[388]3.8884,[389]3.8764,[390]3.8669,[391]3.8562,[392]3.8500,[393]3.8403,[394]3.8315,[395]3.8224,[396]3.8120,[397]3.8024,[398]3.7916,[399]3.7813,[400]3.7720,[401]3.7610,[402]3.7497,[403]3.7400,[404]3.7283,[405]3.7171,[406]3.7060,[407]3.6953,[408]3.6859,[409]3.6767,[410]3.6704,[411]3.6721,[412]3.6675,[413]3.6708,[414]3.6744,[415]3.6716,[416]3.6722,[417]3.6743,[418]3.6686,[419]3.6700,[420]3.6670,[421]3.6655,[422]3.6680,[423]3.6679,[424]3.6724,[425]3.6721,[426]3.6730,[427]3.6723,[428]3.6754,[429]3.6767,[430]3.6800,[431]3.6808,[432]3.6794,[433]3.6754,[434]3.6759,[435]3.6699,[436]3.6642,[437]3.6599,[438]3.6578,[439]3.6563,[440]3.6613,[441]3.6664,[442]3.6743,[443]3.6722,[444]3.6726,[445]3.6734,[446]3.6784,[447]3.6816,[448]3.6841,[449]3.6867,[450]3.6906,[451]3.6941,[452]3.6967,[453]3.6982,[454]3.6964,[455]3.6985,[456]3.6982,[457]3.7008,[458]3.7059,[459]3.7063,[460]3.7060,[461]3.7018,[462]3.7057,[463]3.7133,[464]3.7193,[465]3.7124,[466]3.7106,[467]3.7094,[468]3.7118,[469]3.7091,[470]3.7064,[471]3.7068,[472]3.7077,[473]3.7068,[474]3.7055,[475]3.7070,[476]3.7055,[477]3.7043,[478]3.7053,[479]3.7071,[480]3.7095,[481]3.7052,[482]3.7088,[483]3.7075,[484]3.7110,[485]3.7175,[486]3.7204,[487]3.7238,[488]3.7292,[489]3.7315,[490]3.7362,[491]3.7426,[492]3.7472,[493]3.7465,[494]3.7474,[495]3.7497,[496]3.7512,[497]3.7541,[498]3.7543,[499]3.7532,[500]3.7569,[501]3.7613,[502]3.7604,[503]3.7586,[504]3.7608,[505]3.7641,[506]3.7728,[507]3.7754,[508]3.7785,[509]3.7704,[510]3.7659,[511]3.7599,[512]3.7561,[513]3.7495,[514]3.7488,[515]3.7515,[516]3.7472,[517]3.7477,[518]3.7471,[519]3.7481,[520]3.7532,[521]3.7515,[522]3.7495,[523]3.7557,[524]3.7544,[525]3.7533,[526]3.7488,[527]3.7433,[528]3.7407,[529]3.7373,[530]3.7342,[531]3.7305,[532]3.7239,[533]3.7171,[534]3.7130,[535]3.7146,[536]3.7176,[537]3.7211,[538]3.7247,[539]3.7276,[540]3.7332,[541]3.7369,[542]3.7395,[543]3.7350,[544]3.7308,[545]3.7304,[546]3.7231,[547]3.7171,[548]3.7102,[549]3.7039,[550]3.6979,[551]3.6923,[552]3.6866,[553]3.6810,[554]3.6803,[555]3.6789,[556]3.6814,[557]3.6851,[558]3.6912,[559]3.6956,[560]3.7011,[561]3.6989,
Final estimate: PPL = 3.6989 +/- 0.02106
```

#### PR `283` Test Case
```bash
$ git rev-parse --short HEAD
7f6980fa

## Run 1

perplexity: tokenizing the input ..
perplexity: tokenization took 612.634 ms
perplexity: calculating perplexity over 561 chunks, n_ctx=512, batch_size=2048, n_seq=4
perplexity: 20.36 seconds per pass - ETA 47.58 minutes
[1]2.8167,[2]3.5984,[3]2.5279,[4]2.1350,[5]1.9307,[6]1.8199,[7]1.7183,[8]1.6549,[9]1.6132,[10]1.5715,[11]1.5652,[12]1.6259,[13]1.6478,[14]1.7798,[15]1.9153,[16]1.9692,[17]2.1392,[18]2.2755,[19]2.2279,[20]2.2171,[21]2.3203,[22]2.2886,[23]2.2519,[24]2.2700,[25]2.2320,[26]2.2026,[27]2.2543,[28]2.2624,[29]2.3195,[30]2.3504,[31]2.3870,[32]2.4029,[33]2.4421,[34]2.4923,[35]2.5471,[36]2.6029,[37]2.6384,[38]2.6881,[39]2.7250,[40]2.7885,[41]2.8333,[42]2.8477,[43]2.9012,[44]2.9163,[45]3.0018,[46]3.0529,[47]3.0155,[48]2.9704,[49]2.9533,[50]2.9794,[51]3.0260,[52]3.0432,[53]3.1013,[54]3.1143,[55]3.1468,[56]3.1829,[57]3.2004,[58]3.2455,[59]3.2565,[60]3.3071,[61]3.3500,[62]3.4085,[63]3.4443,[64]3.4925,[65]3.5020,[66]3.4960,[67]3.4727,[68]3.5045,[69]3.5053,[70]3.5287,[71]3.5449,[72]3.5590,[73]3.5715,[74]3.5914,[75]3.5693,[76]3.5179,[77]3.4743,[78]3.4715,[79]3.4516,[80]3.4385,[81]3.4028,[82]3.4083,[83]3.3817,[84]3.3448,[85]3.3113,[86]3.2904,[87]3.2976,[88]3.2723,[89]3.2646,[90]3.2395,[91]3.2150,[92]3.1917,[93]3.1638,[94]3.1410,[95]3.1215,[96]3.1248,[97]3.1335,[98]3.1231,[99]3.1061,[100]3.1060,[101]3.0979,[102]3.1176,[103]3.1448,[104]3.1673,[105]3.1652,[106]3.1920,[107]3.2174,[108]3.2381,[109]3.2746,[110]3.3091,[111]3.3311,[112]3.3003,[113]3.2870,[114]3.2635,[115]3.2465,[116]3.2384,[117]3.2167,[118]3.1937,[119]3.1713,[120]3.1487,[121]3.1329,[122]3.1128,[123]3.0950,[124]3.0722,[125]3.0524,[126]3.0345,[127]3.0218,[128]3.0145,[129]3.0055,[130]2.9943,[131]2.9862,[132]2.9922,[133]2.9999,[134]3.0062,[135]3.0185,[136]3.0349,[137]3.0503,[138]3.0577,[139]3.0696,[140]3.0682,[141]3.0675,[142]3.0642,[143]3.0624,[144]3.0560,[145]3.0458,[146]3.0428,[147]3.0450,[148]3.0424,[149]3.0424,[150]3.0349,[151]3.0310,[152]3.0262,[153]3.0201,[154]3.0184,[155]3.0218,[156]3.0224,[157]3.0273,[158]3.0364,[159]3.0374,[160]3.0464,[161]3.0545,[162]3.0632,[163]3.0686,[164]3.0893,[165]3.1137,[166]3.1324,[167]3.1459,[168]3.1722,[169]3.1956,[170]3.2185,[171]3.2428,[172]3.2243,[173]3.2042,[174]3.1909,[175]3.1779,[176]3.1654,[177]3.1541,[178]3.1408,[179]3.1267,[180]3.1301,[181]3.1442,[182]3.1594,[183]3.1742,[184]3.1882,[185]3.1979,[186]3.2146,[187]3.2298,[188]3.2433,[189]3.2538,[190]3.2533,[191]3.2597,[192]3.2620,[193]3.2666,[194]3.2868,[195]3.2961,[196]3.3094,[197]3.3196,[198]3.3230,[199]3.3280,[200]3.3258,[201]3.3412,[202]3.3351,[203]3.3396,[204]3.3417,[205]3.3418,[206]3.3442,[207]3.3534,[208]3.3635,[209]3.3729,[210]3.3721,[211]3.3663,[212]3.3666,[213]3.3746,[214]3.3760,[215]3.3822,[216]3.3823,[217]3.3756,[218]3.3754,[219]3.3761,[220]3.3743,[221]3.3739,[222]3.3731,[223]3.3745,[224]3.3794,[225]3.3812,[226]3.3714,[227]3.3702,[228]3.3716,[229]3.3757,[230]3.3812,[231]3.3870,[232]3.3788,[233]3.3715,[234]3.3735,[235]3.3734,[236]3.3822,[237]3.3904,[238]3.4001,[239]3.4104,[240]3.4189,[241]3.4301,[242]3.4457,[243]3.4594,[244]3.4676,[245]3.4795,[246]3.4902,[247]3.4876,[248]3.4827,[249]3.4802,[250]3.4725,[251]3.4688,[252]3.4704,[253]3.4731,[254]3.4793,[255]3.4855,[256]3.4890,[257]3.4906,[258]3.4907,[259]3.4927,[260]3.4949,[261]3.4954,[262]3.4931,[263]3.4987,[264]3.5010,[265]3.5011,[266]3.5027,[267]3.5054,[268]3.5099,[269]3.5128,[270]3.5109,[271]3.5089,[272]3.5014,[273]3.5018,[274]3.4945,[275]3.4831,[276]3.4719,[277]3.4732,[278]3.4836,[279]3.4894,[280]3.4974,[281]3.5045,[282]3.5104,[283]3.5171,[284]3.5233,[285]3.5375,[286]3.5392,[287]3.5420,[288]3.5462,[289]3.5486,[290]3.5395,[291]3.5314,[292]3.5335,[293]3.5346,[294]3.5327,[295]3.5317,[296]3.5342,[297]3.5356,[298]3.5404,[299]3.5472,[300]3.5502,[301]3.5536,[302]3.5554,[303]3.5564,[304]3.5546,[305]3.5669,[306]3.5741,[307]3.5855,[308]3.5734,[309]3.5676,[310]3.5575,[311]3.5611,[312]3.5644,[313]3.5713,[314]3.5734,[315]3.5763,[316]3.5771,[317]3.5780,[318]3.5784,[319]3.5792,[320]3.5834,[321]3.5835,[322]3.5852,[323]3.5914,[324]3.5913,[325]3.5967,[326]3.6011,[327]3.6050,[328]3.6073,[329]3.6086,[330]3.6146,[331]3.6183,[332]3.6224,[333]3.6204,[334]3.6199,[335]3.6193,[336]3.6187,[337]3.6194,[338]3.6192,[339]3.6215,[340]3.6248,[341]3.6304,[342]3.6399,[343]3.6496,[344]3.6548,[345]3.6471,[346]3.6407,[347]3.6381,[348]3.6305,[349]3.6265,[350]3.6247,[351]3.6297,[352]3.6453,[353]3.6544,[354]3.6677,[355]3.6766,[356]3.6830,[357]3.6952,[358]3.7059,[359]3.7091,[360]3.7151,[361]3.7246,[362]3.7337,[363]3.7394,[364]3.7462,[365]3.7520,[366]3.7629,[367]3.7718,[368]3.7787,[369]3.7863,[370]3.7948,[371]3.8090,[372]3.8188,[373]3.8216,[374]3.8250,[375]3.8296,[376]3.8427,[377]3.8541,[378]3.8562,[379]3.8550,[380]3.8515,[381]3.8561,[382]3.8620,[383]3.8653,[384]3.8698,[385]3.8737,[386]3.8797,[387]3.8852,[388]3.8884,[389]3.8764,[390]3.8669,[391]3.8562,[392]3.8500,[393]3.8403,[394]3.8315,[395]3.8224,[396]3.8120,[397]3.8024,[398]3.7916,[399]3.7813,[400]3.7720,[401]3.7610,[402]3.7497,[403]3.7400,[404]3.7283,[405]3.7171,[406]3.7060,[407]3.6953,[408]3.6859,[409]3.6767,[410]3.6704,[411]3.6721,[412]3.6675,[413]3.6708,[414]3.6744,[415]3.6716,[416]3.6722,[417]3.6743,[418]3.6686,[419]3.6700,[420]3.6670,[421]3.6655,[422]3.6680,[423]3.6679,[424]3.6724,[425]3.6721,[426]3.6730,[427]3.6723,[428]3.6754,[429]3.6767,[430]3.6800,[431]3.6808,[432]3.6794,[433]3.6754,[434]3.6759,[435]3.6699,[436]3.6642,[437]3.6599,[438]3.6578,[439]3.6563,[440]3.6613,[441]3.6664,[442]3.6743,[443]3.6722,[444]3.6726,[445]3.6734,[446]3.6784,[447]3.6816,[448]3.6841,[449]3.6867,[450]3.6906,[451]3.6941,[452]3.6967,[453]3.6982,[454]3.6964,[455]3.6985,[456]3.6982,[457]3.7008,[458]3.7059,[459]3.7063,[460]3.7060,[461]3.7018,[462]3.7057,[463]3.7133,[464]3.7193,[465]3.7124,[466]3.7106,[467]3.7094,[468]3.7118,[469]3.7091,[470]3.7064,[471]3.7068,[472]3.7077,[473]3.7068,[474]3.7055,[475]3.7070,[476]3.7055,[477]3.7043,[478]3.7053,[479]3.7071,[480]3.7095,[481]3.7052,[482]3.7088,[483]3.7075,[484]3.7110,[485]3.7175,[486]3.7204,[487]3.7238,[488]3.7292,[489]3.7315,[490]3.7362,[491]3.7426,[492]3.7472,[493]3.7465,[494]3.7474,[495]3.7497,[496]3.7512,[497]3.7541,[498]3.7543,[499]3.7532,[500]3.7569,[501]3.7613,[502]3.7604,[503]3.7586,[504]3.7608,[505]3.7641,[506]3.7728,[507]3.7754,[508]3.7785,[509]3.7704,[510]3.7659,[511]3.7599,[512]3.7561,[513]3.7495,[514]3.7488,[515]3.7515,[516]3.7472,[517]3.7477,[518]3.7471,[519]3.7481,[520]3.7532,[521]3.7515,[522]3.7495,[523]3.7557,[524]3.7544,[525]3.7533,[526]3.7488,[527]3.7433,[528]3.7407,[529]3.7373,[530]3.7342,[531]3.7305,[532]3.7239,[533]3.7171,[534]3.7130,[535]3.7146,[536]3.7176,[537]3.7211,[538]3.7247,[539]3.7276,[540]3.7332,[541]3.7369,[542]3.7395,[543]3.7350,[544]3.7308,[545]3.7304,[546]3.7231,[547]3.7171,[548]3.7102,[549]3.7039,[550]3.6979,[551]3.6923,[552]3.6866,[553]3.6810,[554]3.6803,[555]3.6789,[556]3.6814,[557]3.6851,[558]3.6912,[559]3.6956,[560]3.7011,[561]3.6989,
Final estimate: PPL = 3.6989 +/- 0.02106

llama_print_timings:        load time =   10381.37 ms
llama_print_timings:      sample time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings: prompt eval time = 2840034.02 ms / 287232 tokens (    9.89 ms per token,   101.14 tokens per second)
llama_print_timings:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings:       total time = 2843617.25 ms / 287233 tokens

## Run 2

perplexity: tokenizing the input ..
perplexity: tokenization took 581.663 ms
perplexity: calculating perplexity over 561 chunks, n_ctx=512, batch_size=2048, n_seq=4
perplexity: 20.35 seconds per pass - ETA 47.57 minutes
[1]2.8167,[2]3.5984,[3]2.5279,[4]2.1350,[5]1.9307,[6]1.8199,[7]1.7183,[8]1.6549,[9]1.6132,[10]1.5715,[11]1.5652,[12]1.6259,[13]1.6478,[14]1.7798,[15]1.9153,[16]1.9692,[17]2.1392,[18]2.2755,[19]2.2279,[20]2.2171,[21]2.3203,[22]2.2886,[23]2.2519,[24]2.2700,[25]2.2320,[26]2.2026,[27]2.2543,[28]2.2624,[29]2.3195,[30]2.3504,[31]2.3870,[32]2.4029,[33]2.4421,[34]2.4923,[35]2.5471,[36]2.6029,[37]2.6384,[38]2.6881,[39]2.7250,[40]2.7885,[41]2.8333,[42]2.8477,[43]2.9012,[44]2.9163,[45]3.0018,[46]3.0529,[47]3.0155,[48]2.9704,[49]2.9533,[50]2.9794,[51]3.0260,[52]3.0432,[53]3.1013,[54]3.1143,[55]3.1468,[56]3.1829,[57]3.2004,[58]3.2455,[59]3.2565,[60]3.3071,[61]3.3500,[62]3.4085,[63]3.4443,[64]3.4925,[65]3.5020,[66]3.4960,[67]3.4727,[68]3.5045,[69]3.5053,[70]3.5287,[71]3.5449,[72]3.5590,[73]3.5715,[74]3.5914,[75]3.5693,[76]3.5179,[77]3.4743,[78]3.4715,[79]3.4516,[80]3.4385,[81]3.4028,[82]3.4083,[83]3.3817,[84]3.3448,[85]3.3113,[86]3.2904,[87]3.2976,[88]3.2723,[89]3.2646,[90]3.2395,[91]3.2150,[92]3.1917,[93]3.1638,[94]3.1410,[95]3.1215,[96]3.1248,[97]3.1335,[98]3.1231,[99]3.1061,[100]3.1060,[101]3.0979,[102]3.1176,[103]3.1448,[104]3.1673,[105]3.1652,[106]3.1920,[107]3.2174,[108]3.2381,[109]3.2746,[110]3.3091,[111]3.3311,[112]3.3003,[113]3.2870,[114]3.2635,[115]3.2465,[116]3.2384,[117]3.2167,[118]3.1937,[119]3.1713,[120]3.1487,[121]3.1329,[122]3.1128,[123]3.0950,[124]3.0722,[125]3.0524,[126]3.0345,[127]3.0218,[128]3.0145,[129]3.0055,[130]2.9943,[131]2.9862,[132]2.9922,[133]2.9999,[134]3.0062,[135]3.0185,[136]3.0349,[137]3.0503,[138]3.0577,[139]3.0696,[140]3.0682,[141]3.0675,[142]3.0642,[143]3.0624,[144]3.0560,[145]3.0458,[146]3.0428,[147]3.0450,[148]3.0424,[149]3.0424,[150]3.0349,[151]3.0310,[152]3.0262,[153]3.0201,[154]3.0184,[155]3.0218,[156]3.0224,[157]3.0273,[158]3.0364,[159]3.0374,[160]3.0464,[161]3.0545,[162]3.0632,[163]3.0686,[164]3.0893,[165]3.1137,[166]3.1324,[167]3.1459,[168]3.1722,[169]3.1956,[170]3.2185,[171]3.2428,[172]3.2243,[173]3.2042,[174]3.1909,[175]3.1779,[176]3.1654,[177]3.1541,[178]3.1408,[179]3.1267,[180]3.1301,[181]3.1442,[182]3.1594,[183]3.1742,[184]3.1882,[185]3.1979,[186]3.2146,[187]3.2298,[188]3.2433,[189]3.2538,[190]3.2533,[191]3.2597,[192]3.2620,[193]3.2666,[194]3.2868,[195]3.2961,[196]3.3094,[197]3.3196,[198]3.3230,[199]3.3280,[200]3.3258,[201]3.3412,[202]3.3351,[203]3.3396,[204]3.3417,[205]3.3418,[206]3.3442,[207]3.3534,[208]3.3635,[209]3.3729,[210]3.3721,[211]3.3663,[212]3.3666,[213]3.3746,[214]3.3760,[215]3.3822,[216]3.3823,[217]3.3756,[218]3.3754,[219]3.3761,[220]3.3743,[221]3.3739,[222]3.3731,[223]3.3745,[224]3.3794,[225]3.3812,[226]3.3714,[227]3.3702,[228]3.3716,[229]3.3757,[230]3.3812,[231]3.3870,[232]3.3788,[233]3.3715,[234]3.3735,[235]3.3734,[236]3.3822,[237]3.3904,[238]3.4001,[239]3.4104,[240]3.4189,[241]3.4301,[242]3.4457,[243]3.4594,[244]3.4676,[245]3.4795,[246]3.4902,[247]3.4876,[248]3.4827,[249]3.4802,[250]3.4725,[251]3.4688,[252]3.4704,[253]3.4731,[254]3.4793,[255]3.4855,[256]3.4890,[257]3.4906,[258]3.4907,[259]3.4927,[260]3.4949,[261]3.4954,[262]3.4931,[263]3.4987,[264]3.5010,[265]3.5011,[266]3.5027,[267]3.5054,[268]3.5099,[269]3.5128,[270]3.5109,[271]3.5089,[272]3.5014,[273]3.5018,[274]3.4945,[275]3.4831,[276]3.4719,[277]3.4732,[278]3.4836,[279]3.4894,[280]3.4974,[281]3.5045,[282]3.5104,[283]3.5171,[284]3.5233,[285]3.5375,[286]3.5392,[287]3.5420,[288]3.5462,[289]3.5486,[290]3.5395,[291]3.5314,[292]3.5335,[293]3.5346,[294]3.5327,[295]3.5317,[296]3.5342,[297]3.5356,[298]3.5404,[299]3.5472,[300]3.5502,[301]3.5536,[302]3.5554,[303]3.5564,[304]3.5546,[305]3.5669,[306]3.5741,[307]3.5855,[308]3.5734,[309]3.5676,[310]3.5575,[311]3.5611,[312]3.5644,[313]3.5713,[314]3.5734,[315]3.5763,[316]3.5771,[317]3.5780,[318]3.5784,[319]3.5792,[320]3.5834,[321]3.5835,[322]3.5852,[323]3.5914,[324]3.5913,[325]3.5967,[326]3.6011,[327]3.6050,[328]3.6073,[329]3.6086,[330]3.6146,[331]3.6183,[332]3.6224,[333]3.6204,[334]3.6199,[335]3.6193,[336]3.6187,[337]3.6194,[338]3.6192,[339]3.6215,[340]3.6248,[341]3.6304,[342]3.6399,[343]3.6496,[344]3.6548,[345]3.6471,[346]3.6407,[347]3.6381,[348]3.6305,[349]3.6265,[350]3.6247,[351]3.6297,[352]3.6453,[353]3.6544,[354]3.6677,[355]3.6766,[356]3.6830,[357]3.6952,[358]3.7059,[359]3.7091,[360]3.7151,[361]3.7246,[362]3.7337,[363]3.7394,[364]3.7462,[365]3.7520,[366]3.7629,[367]3.7718,[368]3.7787,[369]3.7863,[370]3.7948,[371]3.8090,[372]3.8188,[373]3.8216,[374]3.8250,[375]3.8296,[376]3.8427,[377]3.8541,[378]3.8562,[379]3.8550,[380]3.8515,[381]3.8561,[382]3.8620,[383]3.8653,[384]3.8698,[385]3.8737,[386]3.8797,[387]3.8852,[388]3.8884,[389]3.8764,[390]3.8669,[391]3.8562,[392]3.8500,[393]3.8403,[394]3.8315,[395]3.8224,[396]3.8120,[397]3.8024,[398]3.7916,[399]3.7813,[400]3.7720,[401]3.7610,[402]3.7497,[403]3.7400,[404]3.7283,[405]3.7171,[406]3.7060,[407]3.6953,[408]3.6859,[409]3.6767,[410]3.6704,[411]3.6721,[412]3.6675,[413]3.6708,[414]3.6744,[415]3.6716,[416]3.6722,[417]3.6743,[418]3.6686,[419]3.6700,[420]3.6670,[421]3.6655,[422]3.6680,[423]3.6679,[424]3.6724,[425]3.6721,[426]3.6730,[427]3.6723,[428]3.6754,[429]3.6767,[430]3.6800,[431]3.6808,[432]3.6794,[433]3.6754,[434]3.6759,[435]3.6699,[436]3.6642,[437]3.6599,[438]3.6578,[439]3.6563,[440]3.6613,[441]3.6664,[442]3.6743,[443]3.6722,[444]3.6726,[445]3.6734,[446]3.6784,[447]3.6816,[448]3.6841,[449]3.6867,[450]3.6906,[451]3.6941,[452]3.6967,[453]3.6982,[454]3.6964,[455]3.6985,[456]3.6982,[457]3.7008,[458]3.7059,[459]3.7063,[460]3.7060,[461]3.7018,[462]3.7057,[463]3.7133,[464]3.7193,[465]3.7124,[466]3.7106,[467]3.7094,[468]3.7118,[469]3.7091,[470]3.7064,[471]3.7068,[472]3.7077,[473]3.7068,[474]3.7055,[475]3.7070,[476]3.7055,[477]3.7043,[478]3.7053,[479]3.7071,[480]3.7095,[481]3.7052,[482]3.7088,[483]3.7075,[484]3.7110,[485]3.7175,[486]3.7204,[487]3.7238,[488]3.7292,[489]3.7315,[490]3.7362,[491]3.7426,[492]3.7472,[493]3.7465,[494]3.7474,[495]3.7497,[496]3.7512,[497]3.7541,[498]3.7543,[499]3.7532,[500]3.7569,[501]3.7613,[502]3.7604,[503]3.7586,[504]3.7608,[505]3.7641,[506]3.7728,[507]3.7754,[508]3.7785,[509]3.7704,[510]3.7659,[511]3.7599,[512]3.7561,[513]3.7495,[514]3.7488,[515]3.7515,[516]3.7472,[517]3.7477,[518]3.7471,[519]3.7481,[520]3.7532,[521]3.7515,[522]3.7495,[523]3.7557,[524]3.7544,[525]3.7533,[526]3.7488,[527]3.7433,[528]3.7407,[529]3.7373,[530]3.7342,[531]3.7305,[532]3.7239,[533]3.7171,[534]3.7130,[535]3.7146,[536]3.7176,[537]3.7211,[538]3.7247,[539]3.7276,[540]3.7332,[541]3.7369,[542]3.7395,[543]3.7350,[544]3.7308,[545]3.7304,[546]3.7231,[547]3.7171,[548]3.7102,[549]3.7039,[550]3.6979,[551]3.6923,[552]3.6866,[553]3.6810,[554]3.6803,[555]3.6789,[556]3.6814,[557]3.6851,[558]3.6912,[559]3.6956,[560]3.7011,[561]3.6989,
Final estimate: PPL = 3.6989 +/- 0.02106

llama_print_timings:        load time =   10091.05 ms
llama_print_timings:      sample time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings: prompt eval time = 2834878.21 ms / 287232 tokens (    9.87 ms per token,   101.32 tokens per second)
llama_print_timings:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings:       total time = 2838434.18 ms / 287233 tokens
```

</details>

---

👤 **ikawrakow** commented the **2025-03-24** at **17:21:53**:<br>

Thanks for testing. You are running the MoE experts on the CPU, so you are not supposed to see a difference (and is good you confirm that you don't). At least part of the MoE experts need to run on the GPU to see a benefit (or at least a difference). I expect @davidsyoung with his 16 x 3090 configuration to see PP performance uplift.

---

👤 **davidsyoung** commented the **2025-03-24** at **18:24:25**:<br>

Awesome work!

I'm away at the moment, but I can possibly SSH in and run a `llama-bench` and we can compare to some data over at https://github.com/ikawrakow/ik_llama.cpp/discussions/266. 

Any particular `llama-bench` you'd like @ikawrakow?

---

👤 **davidsyoung** commented the **2025-03-24** at **18:36:17**:<br>

Will run both PP and TG for completeness, running:

`./llama-bench -m /models/DeepSeek-R1-GGUF/DeepSeek-R1-GGUF-iq4_xs__iq3_s_q8.gguf -b 2048 -ub 2048 -fa 1 -mla 2 -amb 128 -fmoe 1 -r 2 -p 512,1024,2048,4096,8192 -n 128,256,512,1024,2048 -n 0 -ngl 63 `

# Comparable data from #266:

| model                          |       size |     params | backend    | ngl | n_ubatch | fa | mla |   amb | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -: | --: | ----: | ---: | ------------: | ---------------: |
| deepseek2 671B Q8_0            | 307.20 GiB |   672.05 B | CUDA       |  63 |    2048 |     2048 |  1 |   2 |   128 |    1 |         pp512 |    238.52 ± 1.44 |
| deepseek2 671B Q8_0            | 307.20 GiB |   672.05 B | CUDA       |  63 |    2048 |     2048 |  1 |   2 |   128 |    1 |        pp1024 |    304.77 ± 0.07 |
| deepseek2 671B Q8_0            | 307.20 GiB |   672.05 B | CUDA       |  63 |    2048 |     2048 |  1 |   2 |   128 |    1 |        pp2048 |    348.11 ± 0.69 |
| deepseek2 671B Q8_0            | 307.20 GiB |   672.05 B | CUDA       |  63 |    2048 |     2048 |  1 |   2 |   128 |    1 |        pp4096 |    326.30 ± 0.69 |
| deepseek2 671B Q8_0            | 307.20 GiB |   672.05 B | CUDA       |  63 |    2048 |     2048 |  1 |   2 |   128 |    1 |        pp8192 |    288.35 ± 0.12 |
| deepseek2 671B Q8_0            | 307.20 GiB |   672.05 B | CUDA       |  63 |    2048 |     2048 |  1 |   2 |   128 |    1 |         tg128 |     17.24 ± 0.02 |
| deepseek2 671B Q8_0            | 307.20 GiB |   672.05 B | CUDA       |  63 |    2048 |     2048 |  1 |   2 |   128 |    1 |         tg256 |     17.88 ± 0.00 |
| deepseek2 671B Q8_0            | 307.20 GiB |   672.05 B | CUDA       |  63 |    2048 |     2048 |  1 |   2 |   128 |    1 |         tg512 |     18.07 ± 0.02 |
| deepseek2 671B Q8_0            | 307.20 GiB |   672.05 B | CUDA       |  63 |    2048 |     2048 |  1 |   2 |   128 |    1 |        tg1024 |     18.05 ± 0.00 |
| deepseek2 671B Q8_0            | 307.20 GiB |   672.05 B | CUDA       |  63 |    2048 |     2048 |  1 |   2 |   128 |    1 |        tg2048 |     17.77 ± 0.01 |

---

# ik/cuda_better_moe:

| model                          |       size |     params | backend    | ngl | n_ubatch | fa | mla |   amb | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -: | --: | ----: | ---: | ------------: | ---------------: |


_will edit as it completes_

---

👤 **davidsyoung** commented the **2025-03-24** at **19:30:14**:<br>

Awesome improvement! @ikawrakow 

![CleanShot 2025-03-24 at 19 29 26@2x](https://github.com/user-attachments/assets/f0244a4b-955c-43b8-8091-d4df17fe6b1e)

---

👤 **ikawrakow** commented the **2025-03-25** at **06:47:06**:<br>

This looks like a winner, merging.

---

👤 **ikawrakow** commented the **2025-03-25** at **18:37:34**:<br>

> Awesome work. Thank you. You are starting to get near to VLLM performance on PP.

How far am I from vLLM?

---

👤 **saood06** commented the **2025-03-25** at **18:45:41**:<br>

> > This looks like a winner, merging.
> 
> Awesome work. Thank you. You are starting to get near to VLLM performance on PP.

And what about sglang which is supposedly even better for Deepseek? Also what about for TG?

---

👤 **davidsyoung** commented the **2025-03-25** at **21:09:44**:<br>

vLLM currently has an overflow issue (for myself personally), with Q3. So it’s not usable (this is with gguf).

They have no support for imatrix quants either, so I’m stuck with q3. I can’t fit q4. 

Sglang has no gguf support. 

I have seen prompt processing of 4-500 t/s with vLLM, but again, I haven’t done a proper bench and I know it can batch requests well. Token generation is upwards towards 25-30 t/s. That’s with tensor parallelism 8 and pipeline parallelism 2. 

But again, it’s broken at the moment.

---

👤 **saood06** commented the **2025-03-25** at **21:17:08**:<br>

>Sglang has no gguf support.

As mentioned before, you might fit AWQ, and that quant has good support on sglang.

---

👤 **davidsyoung** commented the **2025-03-25** at **23:52:05**:<br>

> > Sglang has no gguf support.
> 
> As mentioned before, you might fit AWQ, and that quant has good support on sglang.

Unfortunately not, I’m a bit short of VRAM. If AWQ had 3 bit or 3.5bit possibly…

---

👤 **saood06** commented the **2025-03-26** at **00:59:31**:<br>

> > > Sglang has no gguf support.
> > 
> > 
> > As mentioned before, you might fit AWQ, and that quant has good support on sglang.
> 
> Unfortunately not, I’m a bit short of VRAM. If AWQ had 3 bit or 3.5bit possibly…

That is really unfortunate, as 16x 24GB cards would have probably been the cheapest AWQ capable setup if it had fit.

---

👤 **ikawrakow** commented the **2025-03-26** at **10:14:53**:<br>

> As mentioned before, you might fit AWQ, and that quant has good support on sglang.

@saood06 

You seem to be recommending AWQ quants. On my book AWQ quants are pretty low quality. At least this was the case last I checked. Has something changed since then?

---

👤 **saood06** commented the **2025-03-27** at **04:17:57**:<br>

> > As mentioned before, you might fit AWQ, and that quant has good support on sglang.
> 
> @saood06
> 
> You seem to be recommending AWQ quants. On my book AWQ quants are pretty low quality. At least this was the case last I checked. Has something changed since then?

I'm not sure, I haven't looked deeply into AWQ in a while, I was just curious about sglang's implementation of Deepseek compared to the one here. Normally you wouldn't be able to run sglang without far more expensive GPUs but I thought 16x3090's might be able to run it, but it turns out that is not true.

---

👤 **JohannesGaessler** submitted a review the **2025-04-05** at **10:04:54**: 💬 `COMMENTED`

---

👤 **JohannesGaessler** commented during a code review the **2025-04-05** at **10:04:54** on `ggml/src/ggml-cuda.cu`:<br>

This synchronization is not safe to remove. `ids_host` and `rmapping` are deallocated when they go out of scope and the source pointers for `cudaMemcpyAsync` become dangling pointers. As the name implies, the memcpy is asynchronous and without an explicit synchronization there is no guarantee that the data is still valid once it's being copied to the device.

---

👤 **JohannesGaessler** commented the **2025-04-05** at **10:14:21**:<br>

>Awesome work. Thank you. You are starting to get near to VLLM performance on PP.

If you are using GGUF models in both cases you should be aware that vLLM at some point transplanted quantization-specific CUDA code that I wrote for ggml. I have since improved this code but vLLM has to my knowledge not taken over these improvements.

---

👤 **ikawrakow** submitted a review the **2025-04-05** at **10:55:53**: 💬 `COMMENTED`

---

👤 **ikawrakow** commented during a code review the **2025-04-05** at **10:55:53** on `ggml/src/ggml-cuda.cu`:<br>

Yes, they are deallocated when the function completes. Neither `ids_host` nor `ids` (or `ids_dev`) is used after that. The only reason this forgotten to remove synchronization is there is because I did have a  bug while developing this function. The bug resulted in out of bounds access, so before finding the actual bug one hypothesis I had was that I needed to synchronize because the copy had not finished when I started using the row ids.

---

👤 **JohannesGaessler** submitted a review the **2025-04-05** at **11:11:58**: 💬 `COMMENTED`

---

👤 **JohannesGaessler** commented during a code review the **2025-04-05** at **11:11:58** on `ggml/src/ggml-cuda.cu`:<br>

The original code had synchronization directly after the memcpy so I had assumed that that is where this line comes from. But that is I think not relevant to the discussion.

When you call `cudaMemcpyAsync` you merely pass a pointer and queue a memcpy from that pointer to the device. As it is you don't have any guarantees that that memcpy will happen before the function returns and the memory is deallocated. Even if you are unable to provoke a bug in testing this is a defect which will result in sporadic segfaults or copying of garbage data.

---

👤 **ikawrakow** commented the **2025-04-05** at **11:17:43**:<br>

> I have since improved this code but vLLM has to my knowledge not taken over these improvements.

Based on the performance comparisons on my GPU (RTX-4080) against mainline that I ran after the improvements, they were too minor to offset the performance gains I have from other modifications. For MoE models with many experts such as DeepSeek-V3/R1/Lite, `ik_llama.cpp` is ~1.8X faster than mainline for PP after this PR. It is also ~80-90% of vLLM performance on a multi-GPU system such as the one davidsyoung has, where vLLM uses tensor parallelism and `ik_llama.cpp` does not (so all that will take to match or beat vLLM is to make row split work with MoE models). Given my very limited experience with GPU programming, and given my very rudimentary CUDA knowledge, I'm content with being at 90% of the performance of a repo with 900+ contributors (and the quantized matrix multiplications came from no-one less than you, @JohannesGaessler).

---

👤 **ikawrakow** submitted a review the **2025-04-05** at **11:48:50**: 💬 `COMMENTED`

---

👤 **ikawrakow** commented during a code review the **2025-04-05** at **11:48:50** on `ggml/src/ggml-cuda.cu`:<br>

That would be true if nothing happened after this call. But the row ids are being used in subsequent calls in the same function, so the memcpy must have completed before the function exits. Let's take a look at your original `mul_mat_id` implementation. At the end we have [this call](https://github.com/ggml-org/llama.cpp/blob/7a84777f42a9b3ba47db5d20b7662f8ddf92f652/ggml/src/ggml-cuda/ggml-cuda.cu#L2093). This copies the data from the contiguous memory pool-allocated in the function to its final destination. Now, if this call has not completed by the time the function returns, than we would obviously have "sporadic segfaults and copying of garbage data". So, even without knowing anything about CUDA, one needs to assume that a call such as this completes synchronously, else the entire `llama.cpp` CUDA stack would be a collection of "sporadic segfaults and copying of garbage data". Well, there are calls such as that one in my function as well before it returns. These kernel calls, as well as the preceding processing, they all use the row ids that you are claiming may go out of scope. But in order for them to execute, the queued memcpy must have completed, so no, no "sporadic segfaults and copying of garbage data" at this point.

But at the end of the day, if you are able to trigger the bug, using whatever it takes to trigger it, I'll be happy to uncomment the synchronization call.

---

👤 **JohannesGaessler** submitted a review the **2025-04-05** at **12:23:11**: 💬 `COMMENTED`

---

👤 **JohannesGaessler** commented during a code review the **2025-04-05** at **12:23:11** on `ggml/src/ggml-cuda.cu`:<br>

`k_copy_dst_from_contiguous` only uses device pointers. The point in time at which their data is valid is automatically synchronized with the execution of the kernel because CUDA streams guarantee an ordering in which device code is executed. `cudaMemcpyAsync` is fundamentally different because it uses a host pointer with memory that can become invalid under the control of host code.

>Let's take a look at your original mul_mat_id implementation. At the end we have [this call](https://github.com/ggml-org/llama.cpp/blob/7a84777f42a9b3ba47db5d20b7662f8ddf92f652/ggml/src/ggml-cuda/ggml-cuda.cu#L2093). This copies the data from the contiguous memory pool-allocated in the function to its final destination.

The way the CUDA memory pools work is that the memory is allocated in a single, large block that can grow dynamically. Assuming that you don't need to increase the size of the block an "allocation" `ggml_cuda_pool_alloc` does not actually allocate any new memory, it simply returns a pointer into the large block that is selected in such a way that there are no conflicts between the "allocated" memory regions while the "allocations" are in scope. The actual memory continues to be a valid allocation afterwards, though it will likely be overwritten by other kernels. This is very similar to how the ggml graph planner is giving each tensor a pointer to some data where at the time of the tensor being executed the data is guaranteed to be valid but the memory is re-used for other tensors as long as there are no conflicts.

---

👤 **JohannesGaessler** submitted a review the **2025-04-05** at **12:24:46**: 💬 `COMMENTED`

---

👤 **JohannesGaessler** commented during a code review the **2025-04-05** at **12:24:46** on `ggml/src/ggml-cuda.cu`:<br>

>This is very similar to how the ggml graph planner is giving each tensor a pointer to some data

Actually, `wdata` may be a better comparison.

---

👤 **ikawrakow** submitted a review the **2025-04-05** at **12:33:00**: 💬 `COMMENTED`

---

👤 **ikawrakow** commented during a code review the **2025-04-05** at **12:33:00** on `ggml/src/ggml-cuda.cu`:<br>

See #313. The issue is not that it will go out of scope, but that I'm using the data on the host before the copy may have completed.

---

👤 **JohannesGaessler** submitted a review the **2025-04-05** at **12:43:28**: 💬 `COMMENTED`

---

👤 **JohannesGaessler** commented during a code review the **2025-04-05** at **12:43:28** on `ggml/src/ggml-cuda.cu`:<br>

Sorry, I just noticed that I mixed up the copy directions for the two memcpys.