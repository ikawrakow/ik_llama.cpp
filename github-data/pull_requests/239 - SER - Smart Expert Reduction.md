### 🔀 [#239](https://github.com/ikawrakow/ik_llama.cpp/pull/239) - SER - Smart Expert Reduction

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-03-01 |
| **Updated** | 2025-03-18 |

---

#### Description

The idea behind this PR is very simple: we define new parameters (specified via the command line) $K_{\rm min}$ and $t$. During inference experts are normally selected by sorting their computed probabilities $p_i$ in descending order and picking the top $K$ experts. We modify this expert selection algorithm by always selecting the top $K_{\rm min}$ experts ($K_{\rm min} < K$), and using experts between $K_{\rm min}$ and $K$ only if $p_i > t\cdot p_0$ (i.e., only if their probability $p_i$ relative to the top expert probability $p_0$ is greater than the specified threshold $t$). If we set $t = 0$, this expert selection modification is never invoked, so we have the behavior of the original model. If we set $t = 1$, we use a fixed number of experts $K_{\rm min}$ (the same can be achieved by using `--override-kv deepseek2.expert_used_count=int:Kmin` on the command line, but using `-ser Kmin,1` is clearly much easier to type and remember).

What is the purpose of this? We are hoping to gain performance without a significant loss of precision. Let's take a look at some data. Model is DeepSeek-Lite quantized with `IQ4_NL`. We measure accuracy loss (or error) via `PPL(SER)/PPL(full)-1`. I know some people don't like using perplexity. To each their own. On my book perplexity is a perfectly fine way (to not say the best way) to measure accuracy loss due to some model approximation (quantization, or, as here, selectively using fewer experts) as we are comparing to the base model and not to some other model. The following graph shows quantization error (as defined above) as a function of threshold $t$ for $K_{\rm min}=$ 3, 4, and 5 (DeepSeek-Lite has 6 active experts specified). 

![ser_ppl](https://github.com/user-attachments/assets/a33cf048-027d-4b2e-96b0-6e212a82b892)

We observe kind of expected sigmoid change of the error between base at $t = 0$ (0.8% due to quantization) and the upper threshold defined by always using exactly $K_{\rm min}$ experts. For $K_{\rm min}$ there is barely any increase in the precision loss (1.36% at $t = 1$). For $K_{\rm min} = 3$ and 4 we see that we can keep the error to a more acceptable range if we use $t < \sim0.4$.

The best way to examine performance gains is to look at performance relative to base as a function of precision loss.  The following graph shows the results for CUDA (RTX-4080). Black symbols are for processing a prompt of 2048 tokens (`pp2048`), red symbols are for token generation (`tg128`). 

![ser_performance](https://github.com/user-attachments/assets/350bf6cc-ce69-4fd0-862d-0cc8a0fbf0a2)

What are the megenta symbols? These are for a model quantized with `--pure` (i.e., all tensors are `IQ4_NL` except for the output tensor and the token embeddings). Without this option `llama-quantize` will use a mix of 5-,6- and even 8-bit quants for the attention tensors and shared experts of MoE models such as DeepSeek-Lite/V3/R1. In [this discussion](https://github.com/ikawrakow/ik_llama.cpp/pull/235#issuecomment-2689086533) @saood06 wrote that doing that is not a good idea as this leads to a significant performance penalty. This is of course true, using more bits always comes with a price in TG performance due to TG being memory bound. But typically one wants to pick the best balance between precision loss and performance. Based in the above plot, at least on CUDA, it is much better to use fewer experts than to be stingy with bits for attention tensors. At the 1.6% quantization error of 4-bit attention tensors one can get a 12% TG performance boost with $K_{\rm min} = 4, t = 0.4$ using the default `IQ4_NL` quantization scheme, vs the 2.3% one gets with `--pure`.

But this is CUDA specific, so let's look at the same plot running on the CPU (Ryzen-7950X).
  
![ser_performance_cpu](https://github.com/user-attachments/assets/4e5836e6-0b76-4660-81d2-18ec0323e7ae)

Here magenta TG performance is more competitive with this PR, but still cannot compete with just using 5 instead of 6 experts. 

In summary: Based on these results, using $K_{min} = 4, t = 0.2$ or $K_{\rm min} = 5, t = 0.4$ looks to me as a very viable option. We get a noticeable TG performance  gain of 5-7% without much reduction in model quality. It would be great if somebody could study the behavior of DeepSeekV3/R1 with this PR. There we have slightly more room for expert reduction from 8 to 5, 6, or 7. 

I wonder if this (or something similar) is what they call "selectively using 6 experts" in the KTransformers repository. Does somebody know?

Almost forgot: to use this option, add
```
-ser Kmin,t  or --smart-expert-reduction Kmin,t
```
to the command line. 

**Caveat:** not implemented on Metal. The Metal back end has started to seriously fall behind, so at some point I need to take the time to add this and all other missing features.

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-03-01** at **15:49:06**:<br>

Here a graph for error versus performance gain for hybrid CPU/GPU inference (Ryzen-7950X/RTX-4080) for DeepSeek-Lite. Operation with MoE tensors are computed on the CPU, all others on the GPU.

![ser_performance_hybrid](https://github.com/user-attachments/assets/2f5ff74c-eddb-493e-b215-38bb070baaa8)

Here performance gains are much more significant. As attention and shared experts computation done on the GPU is much faster than the MoE calculation done on the CPU, we gain more by selectively reducing experts. If we just use 5 experts instead of 6, TG performance increases by nearly 20% while the associated error is significantly less than using 4 bits for the attention layers.

---

👤 **davidsyoung** commented the **2025-03-01** at **16:25:50**:<br>

This looks very interesting - what would you recommend is the best way to test this with full CUDA off-load with R1? If you have some harnesses to test PPL, that would be great

---

👤 **ikawrakow** commented the **2025-03-01** at **17:11:55**:<br>

I typically use Wikitext2 `PPL`. There are many people out there who believe that this is not good, but I have also compared to C4 `PPL` (English and French) and, once you look at the ratio of `PPL(approximate model)/PPL(full model)-1`, things do not depend that much on the specific test corpus. The same is also true for context length. Even though PPL can change a lot with the context window used for evaluation, the ratio `PPL(approximate model)/PPL(full model)` is nearly independent of context length. One can also compute KL divergence (and many people think this is better than `PPL`), but that is much less convenient (one must first run a calculation with the full model, generate a huge data file, to then run with the approximate model to get the KL divergence values), to only find out that the mean KL divergence correlates almost 100% with `log(PPL(approximate)/PPL(full))`. Same is true for HellaSwag, the other benchmark one can run with `llama.cpp`. The correlation coefficient between `HellaSwag(full) - HellaSwag(approximate)` with `PPL(approximate)/PPL(full)-1` tends to be over 90%, so this doesn't give much additional information (but takes way longer to compute than PPL). So, at then end, if you have settled on a model you want to use, comparing `PPL` with SER to `PPL` without will give good indication about performance degradation.

It is of course also important to just use it and see if you think the quality of the responses is degraded. This is very subjective, but it will be you using it, so you must like it.

But with the 150-200 t/s you are getting for R1 it will not be easy to get a detailed evaluation. Each point in the graphs above takes less than 2 minutes to compute, so with a simple script it was done in less than 1 hour. In your case, a full PPL calculation on Wikitext2 with optimistically 200 t/s will take close to 30 minutes. I have seen people looking at just the first 10 or 20 batches. This is by far not enough as results tend to change quite a bit after that. So, I think it is important to carefully select the few full runs you want to do. I would first check 6 and 7 experts using `-ser 6,1` / `-ser 7,1`, see how much performance one gains and how much quality degrades, and then decide how to proceed.

---

👤 **davidsyoung** commented the **2025-03-01** at **17:25:56**:<br>

Okay, cool! I am going to first create my own quant somewhere around `i1-IQ3_XXS`, `i1-IQ3_XS`, or `i1-IQ3_S`. I'm downloading the full BF16 model right now, and then when I have the best fit of quants, I'll figure out how to run a PPL test... :) Thank you.

---

👤 **davidsyoung** commented the **2025-03-03** at **21:35:39**:<br>

@ikawrakow a little bit off topic but didn't know where better to ask.

I have downloaded the BF16 version, converted to gguf, and then quantisizing to `IQ3_S` with an imatrix from https://huggingface.co/mradermacher/DeepSeek-R1-GGUF with the following command:

```
./llama-quantize --imatrix /models/deepseek-config/imatrix.dat  /storage/unsloth_DeepSeek-R1-BF16/unsloth_DeepSeek-R1-BF16-256x21B-F16-00001-of-00059.gguf /models/DeepSeek-R1-GGUF-IQ3_S.gguf IQ3_S
```

All seems to be going well, until I hit:

```
ggml_validate_row_data: found inf value at block 3405774848
llama_model_quantize: failed to quantize: tensor 'blk.40.ffn_down_exps.weight' has invalid data
main: failed to quantize model from '/storage/unsloth_DeepSeek-R1-BF16/unsloth_DeepSeek-R1-BF16-256x21B-F16-00001-of-00059.gguf'
```

Now I don't know if this is because of the imatrix, the changes for MLA with the quantize process, or a corrupted BF16 model file. I am currently re-checking the hash of the `BF16` model files to see if I downloaded a corrupt part.

Likely a corrupt part. But just wondering, is there anything I'm doing wrong here? I wasn't 100% sure if that's a correct quantize command, or something I'm missing.

TYVM

---

👤 **ikawrakow** commented the **2025-03-04** at **11:21:38**:<br>

Let me know if it works after you re-download the corrupt file. If it doesn't, the I would need to make the quantization more robust against missing imatrix data. DeepSeekV3/R1 is tricky because only 8 out of 256 experts are activated per token, so for an imatrix calculation with a given amount of calibration data there will be 32X less data collected for the experts compared to a dense model. This may lead to missing/insufficient imatrix data, which may not be handled gracefully by the quantization functions.

---

👤 **davidsyoung** commented the **2025-03-04** at **11:48:46**:<br>

I will! Reconverting to GGUF from BF16 takes a decent amount of time on HDDs compared to NVME. Should be done around 6pm tonight, and I’ll quantize soon after that! Thank you for all of the help and your work on improving inference with DS V3/R1 - its excellent!

---

👤 **davidsyoung** commented the **2025-03-04** at **20:16:54**:<br>

@ikawrakow 

Seemed to quantize fine, but got this on model load:

```
INFO [                    main] build info | tid="23133942390784" timestamp=1741119264 build=0 commit="unknown"
INFO [                    main] system info | tid="23133942390784" timestamp=1741119264 n_threads=64 n_threads_batch=-1 total_threads=128 system_info="AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: loaded meta data with 53 key-value pairs and 1147 tensors from /models/DeepSeek-R1-GGUF-IQ3_S.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = unsloth_DeepSeek R1 BF16
llama_model_loader: - kv   3:                         general.size_label str              = 256x21B
llama_model_loader: - kv   4:                            general.license str              = mit
llama_model_loader: - kv   5:                   general.base_model.count u32              = 1
llama_model_loader: - kv   6:                  general.base_model.0.name str              = DeepSeek R1
llama_model_loader: - kv   7:          general.base_model.0.organization str              = Deepseek Ai
llama_model_loader: - kv   8:              general.base_model.0.repo_url str              = https://huggingface.co/deepseek-ai/De...
llama_model_loader: - kv   9:                               general.tags arr[str,3]       = ["deepseek", "unsloth", "transformers"]
llama_model_loader: - kv  10:                          general.languages arr[str,1]       = ["en"]
llama_model_loader: - kv  11:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv  12:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv  13:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv  14:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  15:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv  16:          deepseek2.attention.head_count_kv u32              = 128
llama_model_loader: - kv  17:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  18: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  19:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  20:                          general.file_type u32              = 26
llama_model_loader: - kv  21:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  22:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  23:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  24:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  25:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  26:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  27:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  28:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  29:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  30:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  31:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  32:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  33:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  34:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  35:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  36: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  37: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  38:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  39:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  40:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<�...
llama_model_loader: - kv  41:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  42:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  43:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  44:            tokenizer.ggml.padding_token_id u32              = 128815
llama_model_loader: - kv  45:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  46:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  47:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  48:               general.quantization_version u32              = 2
llama_model_loader: - kv  49:                      quantize.imatrix.file str              = /models/deepseek-config/imatrix.dat
llama_model_loader: - kv  50:                   quantize.imatrix.dataset str              = imatrix-training-full-3
llama_model_loader: - kv  51:             quantize.imatrix.entries_count i32              = 720
llama_model_loader: - kv  52:              quantize.imatrix.chunks_count i32              = 315
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  305 tensors
llama_model_loader: - type q5_K:   61 tensors
llama_model_loader: - type q6_K:    1 tensors
llama_model_loader: - type iq3_s:  419 tensors
llama_model_load: error loading model: error loading model vocabulary: cannot find tokenizer merges in model file

llama_load_model_from_file: failed to load model
llama_init_from_gpt_params: error: failed to load model '/models/DeepSeek-R1-GGUF-IQ3_S.gguf'
 ERR [              load_model] unable to load model | tid="23133942390784" timestamp=1741119264 model="/models/DeepSeek-R1-GGUF-IQ3_S.gguf"
/app/.devops/tools_new.sh: line 47:    13 Segmentation fault      ./llama-server "$@"
```

---

👤 **davidsyoung** commented the **2025-03-05** at **12:36:10**:<br>

Preliminary results with `-ser 6,1` and `-ser 7,1` show no major difference to TG performance - it's -/+ 1 t/s. Likely that with 16x3090 it's not compute limited, as GPU's are only running at 5-10% during inference.

---

👤 **ikawrakow** commented the **2025-03-05** at **12:54:10**:<br>

> Likely that with 16x3090 it's not compute limited, as GPU's are only running at 5-10% during inference.

You observe 5-10% GPU utilization because each GPU is only processing 1/16th of the layers, so it is busy only 1/16th of the time (the other time it is just waiting for the next piece of data). You said you are getting ~ 17 t/s, so each token is taking about 60 ms, so each GPU is busy for about 4 ms out of the 60 ms. But while it is busy, the calculation is limited by something (else it would finish in zero time). If the computation is dominated by the MoE part of the model (it is on my RTX-4080), then using fewer experts will make it run faster, no matter if it is memory or compute bound. With 6 instead of 8 experts it should be spending 3 ms instead of 4 ms in each GPU, so you should see up to 20% speedup. It is less than that in practice due to MoE not being 100%, latencies, etc.  Say it is 10%. That's only 1.7 t/s faster. With the massive fluctuations in processing speed that I see in the logs you have posted before, it is probably hard to measure a 10% speedup.  You will need `llama-bench`, but you said that `llama-bench` is not doing the layer split correctly. Perhaps you could see it in prompt processing speed if you process a longer prompt. I think @saood06 was mentioning somewhere that one needs to "warm up" the model for quite some time before performance becomes more stable, perhaps this is also true for your system.

---

👤 **davidsyoung** commented the **2025-03-05** at **13:43:59**:<br>

This makes sense, thank you for taking the time to type it out! 

Do you have commands that you’d like to run to test SER / PPL for you? llama-bench wasn’t splitting over GPUs unfortunately.

I’m also quanting a IQ4_KSS which I feel will be a great sweet spot, so thank you!

---

👤 **davidsyoung** commented the **2025-03-05** at **14:02:55**:<br>

Super stuff. When some with quant I’ll do that! 

Also, just in terms of FA, when I tried to run FA earlier it tried to allocate 150GB to first GPU. So just went back to MLA. Not sure if I was doing something wrong on my side, I just swapped MLA for FA And ran with the same params otherwise.

---

👤 **ikawrakow** commented the **2025-03-05** at **16:26:50**:<br>

> Also, just in terms of FA, when I tried to run FA earlier it tried to allocate 150GB to first GPU.

That happened after PR #241 was merged and you updated to latest? I guess, you are trying to run with a context of 163k tokens. For the `perplexity` calculation with the above command (context of 2048 tokens) the KV cache will be 1.2 GiB and the compute buffer should not be more than 1-2 GiB. If you go to `Q8_0` KV cache (add `-ctk q8_0 -ctv q8_0` to the above command), than KV cache will be only 600 MiB.

---

👤 **davidsyoung** commented the **2025-03-05** at **21:21:02**:<br>

Ok got some PPL runs!

All perplexity evals were ran with:
`./llama-perplexity -m /models/DeepSeek-R1-GGUF/DeepSeek-R1-GGUF-IQ3_M.gguf -f /models/wiki.test.raw -fmoe -fa -c 2048 -ub 2048       --n-gpu-layers 100       -ts 41,23.5,26,24.5,23.5,25.5,24.4,23.5,25.5,24.5,23.5,25.5,24.5,23.5,25.5,30`.

@saood06 tagging you as I know you are collecting PPL
---

# No -SER

```
perplexity: calculating perplexity over 140 chunks, n_ctx=2048, batch_size=2048, n_seq=1
perplexity: 6.81 seconds per pass - ETA 15.88 minutes
[1]1.5219,[2]1.3062,[3]1.2690,[4]1.7245,[5]1.7816,[6]1.7482,[7]1.8539,[8]1.9760,[9]2.1651,[10]2.3565,[11]2.4824,[12]2.3622,[13]2.4881,[14]2.5845,[15]2.7140,[16]2.8380,[17]2.8269,[18]2.8864,[19]2.8244,[20]2.7444,[21]2.6764,[22]2.6061,[23]2.5171,[24]2.4632,[25]2.4308,[26]2.5115,[27]2.5878,[28]2.5893,[29]2.5345,[30]2.4758,[31]2.4206,[32]2.3780,[33]2.3617,[34]2.4010,[35]2.4370,[36]2.4366,[37]2.4430,[38]2.4384,[39]2.4483,[40]2.4779,[41]2.5326,[42]2.6112,[43]2.6407,[44]2.5960,[45]2.5671,[46]2.6201,[47]2.6735,[48]2.6953,[49]2.7431,[50]2.7610,[51]2.7831,[52]2.8062,[53]2.8094,[54]2.8229,[55]2.8223,[56]2.8345,[57]2.8370,[58]2.8563,[59]2.8702,[60]2.9023,[61]2.9445,[62]2.9476,[63]2.9493,[64]2.9675,[65]2.9752,[66]2.9866,[67]2.9954,[68]2.9791,[69]2.9405,[70]2.9686,[71]2.9976,[72]3.0062,[73]2.9826,[74]2.9864,[75]3.0042,[76]3.0098,[77]3.0103,[78]3.0153,[79]3.0243,[80]3.0311,[81]3.0345,[82]3.0403,[83]3.0541,[84]3.0555,[85]3.0685,[86]3.0931,[87]3.0703,[88]3.0997,[89]3.1293,[90]3.1523,[91]3.1733,[92]3.2027,[93]3.2350,[94]3.2659,[95]3.2670,[96]3.2850,[97]3.2967,[98]3.2653,[99]3.2293,[100]3.1937,[101]3.1593,[102]3.1260,[103]3.1185,[104]3.1088,[105]3.1104,[106]3.1116,[107]3.1140,[108]3.1163,[109]3.0945,[110]3.0931,[111]3.0901,[112]3.1006,[113]3.1141,[114]3.1198,[115]3.1294,[116]3.1480,[117]3.1476,[118]3.1467,[119]3.1469,[120]3.1499,[121]3.1513,[122]3.1640,[123]3.1804,[124]3.1842,[125]3.1914,[126]3.1909,[127]3.1993,[128]3.1818,[129]3.1758,[130]3.1812,[131]3.1899,[132]3.1730,[133]3.1593,[134]3.1662,[135]3.1793,[136]3.1690,[137]3.1456,[138]3.1233,[139]3.1267,[140]3.1464,
Final estimate: PPL = 3.1464 +/- 0.01620

llama_print_timings:        load time =  628119.80 ms
llama_print_timings:      sample time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings: prompt eval time =  802810.58 ms / 286720 tokens (    2.80 ms per token,   357.15 tokens per second)
llama_print_timings:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings:       total time =  808019.78 ms / 286721 tokens
```

---

# -SER 7,1
```
perplexity: calculating perplexity over 140 chunks, n_ctx=2048, batch_size=2048, n_seq=1
perplexity: 6.56 seconds per pass - ETA 15.30 minutes
[1]1.5243,[2]1.3114,[3]1.2793,[4]1.7346,[5]1.7938,[6]1.7580,[7]1.8619,[8]1.9872,[9]2.1788,[10]2.3730,[11]2.4995,[12]2.3798,[13]2.5058,[14]2.6012,[15]2.7314,[16]2.8552,[17]2.8424,[18]2.9014,[19]2.8407,[20]2.7598,[21]2.6919,[22]2.6221,[23]2.5355,[24]2.4823,[25]2.4508,[26]2.5318,[27]2.6078,[28]2.6092,[29]2.5539,[30]2.4951,[31]2.4399,[32]2.3969,[33]2.3812,[34]2.4205,[35]2.4569,[36]2.4560,[37]2.4621,[38]2.4572,[39]2.4670,[40]2.4963,[41]2.5511,[42]2.6304,[43]2.6604,[44]2.6159,[45]2.5879,[46]2.6411,[47]2.6950,[48]2.7167,[49]2.7647,[50]2.7826,[51]2.8042,[52]2.8270,[53]2.8299,[54]2.8433,[55]2.8428,[56]2.8545,[57]2.8570,[58]2.8762,[59]2.8898,[60]2.9221,[61]2.9649,[62]2.9680,[63]2.9692,[64]2.9871,[65]2.9949,[66]3.0067,[67]3.0156,[68]2.9993,[69]2.9606,[70]2.9886,[71]3.0181,[72]3.0269,[73]3.0015,[74]3.0055,[75]3.0227,[76]3.0287,[77]3.0292,[78]3.0346,[79]3.0435,[80]3.0503,[81]3.0535,[82]3.0589,[83]3.0728,[84]3.0740,[85]3.0867,[86]3.1115,[87]3.0887,[88]3.1186,[89]3.1484,[90]3.1719,[91]3.1927,[92]3.2225,[93]3.2546,[94]3.2860,[95]3.2870,[96]3.3051,[97]3.3167,[98]3.2852,[99]3.2492,[100]3.2133,[101]3.1788,[102]3.1452,[103]3.1376,[104]3.1281,[105]3.1295,[106]3.1305,[107]3.1330,[108]3.1351,[109]3.1130,[110]3.1119,[111]3.1090,[112]3.1195,[113]3.1332,[114]3.1389,[115]3.1485,[116]3.1672,[117]3.1667,[118]3.1657,[119]3.1659,[120]3.1689,[121]3.1700,[122]3.1829,[123]3.1993,[124]3.2029,[125]3.2102,[126]3.2093,[127]3.2175,[128]3.2004,[129]3.1942,[130]3.1997,[131]3.2087,[132]3.1916,[133]3.1780,[134]3.1851,[135]3.1985,[136]3.1883,[137]3.1647,[138]3.1426,[139]3.1461,[140]3.1658,
Final estimate: PPL = 3.1658 +/- 0.01626

llama_print_timings:        load time =  632730.77 ms
llama_print_timings:      sample time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings: prompt eval time =  773941.67 ms / 286720 tokens (    2.70 ms per token,   370.47 tokens per second)
llama_print_timings:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings:       total time =  779226.67 ms / 286721 tokens
```

---

# -SER 6,1
```
perplexity: calculating perplexity over 140 chunks, n_ctx=2048, batch_size=2048, n_seq=1
perplexity: 6.37 seconds per pass - ETA 14.87 minutes
[1]1.5452,[2]1.3293,[3]1.3015,[4]1.7622,[5]1.8204,[6]1.7847,[7]1.8853,[8]2.0146,[9]2.2088,[10]2.4046,[11]2.5308,[12]2.4110,[13]2.5382,[14]2.6335,[15]2.7661,[16]2.8906,[17]2.8784,[18]2.9380,[19]2.8757,[20]2.7967,[21]2.7298,[22]2.6618,[23]2.5762,[24]2.5232,[25]2.4925,[26]2.5753,[27]2.6532,[28]2.6540,[29]2.5986,[30]2.5397,[31]2.4828,[32]2.4393,[33]2.4252,[34]2.4649,[35]2.5011,[36]2.4996,[37]2.5052,[38]2.5001,[39]2.5092,[40]2.5383,[41]2.5936,[42]2.6741,[43]2.7048,[44]2.6597,[45]2.6318,[46]2.6853,[47]2.7405,[48]2.7627,[49]2.8111,[50]2.8289,[51]2.8503,[52]2.8733,[53]2.8756,[54]2.8886,[55]2.8874,[56]2.8990,[57]2.9006,[58]2.9199,[59]2.9338,[60]2.9663,[61]3.0093,[62]3.0122,[63]3.0135,[64]3.0319,[65]3.0400,[66]3.0522,[67]3.0613,[68]3.0439,[69]3.0049,[70]3.0340,[71]3.0641,[72]3.0732,[73]3.0484,[74]3.0525,[75]3.0697,[76]3.0754,[77]3.0758,[78]3.0811,[79]3.0895,[80]3.0964,[81]3.0994,[82]3.1045,[83]3.1183,[84]3.1194,[85]3.1321,[86]3.1567,[87]3.1336,[88]3.1640,[89]3.1943,[90]3.2180,[91]3.2392,[92]3.2691,[93]3.3017,[94]3.3336,[95]3.3346,[96]3.3528,[97]3.3644,[98]3.3328,[99]3.2966,[100]3.2602,[101]3.2252,[102]3.1912,[103]3.1836,[104]3.1742,[105]3.1753,[106]3.1759,[107]3.1787,[108]3.1809,[109]3.1586,[110]3.1576,[111]3.1544,[112]3.1650,[113]3.1789,[114]3.1846,[115]3.1943,[116]3.2133,[117]3.2133,[118]3.2125,[119]3.2123,[120]3.2153,[121]3.2162,[122]3.2291,[123]3.2455,[124]3.2489,[125]3.2561,[126]3.2548,[127]3.2632,[128]3.2459,[129]3.2400,[130]3.2456,[131]3.2550,[132]3.2378,[133]3.2239,[134]3.2312,[135]3.2448,[136]3.2351,[137]3.2113,[138]3.1893,[139]3.1928,[140]3.2128,
Final estimate: PPL = 3.2128 +/- 0.01647

llama_print_timings:        load time =  628991.99 ms
llama_print_timings:      sample time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings: prompt eval time =  751365.40 ms / 286720 tokens (    2.62 ms per token,   381.60 tokens per second)
llama_print_timings:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings:       total time =  756557.67 ms / 286721 tokens
```

Next I'm going to try to run `IQ4_KSS`, but splitting the layers over the GPU's always unevenly split and I'm not sure I can fit it in. If we could get `-split-mode row` working it'd be very helpful! But not sure if it's an easy fix (likely not), for example here's how it looks atm trying to balance over `-ts`:

```
llm_load_tensors: ggml ctx size =    7.94 MiB
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size =   938.98 MiB
llm_load_tensors:      CUDA0 buffer size = 16863.73 MiB
llm_load_tensors:      CUDA1 buffer size = 19696.41 MiB
llm_load_tensors:      CUDA2 buffer size = 19218.55 MiB
llm_load_tensors:      CUDA3 buffer size = 14413.91 MiB
llm_load_tensors:      CUDA4 buffer size = 19218.55 MiB
llm_load_tensors:      CUDA5 buffer size = 19218.55 MiB
llm_load_tensors:      CUDA6 buffer size = 14413.91 MiB
llm_load_tensors:      CUDA7 buffer size = 19218.55 MiB
llm_load_tensors:      CUDA8 buffer size = 19218.55 MiB
llm_load_tensors:      CUDA9 buffer size = 14413.91 MiB
llm_load_tensors:     CUDA10 buffer size = 19218.55 MiB
llm_load_tensors:     CUDA11 buffer size = 19218.55 MiB
llm_load_tensors:     CUDA12 buffer size = 14413.91 MiB
llm_load_tensors:     CUDA13 buffer size = 19218.55 MiB
llm_load_tensors:     CUDA14 buffer size = 19218.55 MiB
llm_load_tensors:     CUDA15 buffer size = 15138.89 MiB
```

It takes quite some time for the buffers to allocate so it's a slow feedback loop to try to balance.

---

👤 **davidsyoung** commented the **2025-03-05** at **21:31:59**:<br>

![perplexity_across_chunks](https://github.com/user-attachments/assets/ff289b56-7237-4288-9b70-9215f9ff959f)
![perplexity_vs_speed](https://github.com/user-attachments/assets/92a3622d-6b99-492c-903e-a00310cb8152)

---

👤 **ikawrakow** commented the **2025-03-06** at **06:11:44**:<br>

Great results, thank you for these.

357 t/s prompt processing speed is pretty good! (at least relative to what I have seen people reporting for consumer grade hardware).

Have you tried using `-ot` to distribute the model tensors between the GPUs? You will need 16 arguments `-ot "regexp_i=CUDA_i` to force a specific range of layers on specific GPUs. If that works out, perhaps you can also try forcing the non-MoE tensors be all on 1 or 2 GPUs, and use the remaining 14 or 15 to do the MoE tensors. That may increase the VRAM you have available as the MoE GPU's should not require VRAM for KV cache (at least this is my expectation, but `llama.cpp` and as a result `ik_llama.cpp` not always does what one expects).

---

👤 **davidsyoung** commented the **2025-03-06** at **09:47:08**:<br>

> Thanks for running the PPL, hoping you can fit IQ4_KSS as it will be higher quality.
> 
> > Next I'm going to try to run `IQ4_KSS`, but splitting the layers over the GPU's always unevenly split and I'm not sure I can fit it in. If we could get `-split-mode row` working it'd be very helpful! But not sure if it's an easy fix (likely not), for example here's how it looks atm trying to balance over `-ts`:
> 
> This comment has a method that might be worth trying and seeing if it helps you get split-mode row working: [ggml-org/llama.cpp#11446 (comment)](https://github.com/ggml-org/llama.cpp/pull/11446#issuecomment-2651659237)
> 
> > It takes quite some time for the buffers to allocate so it's a slow feedback loop to try to balance.
> 
> If the above doesn't work then you may try something similar to the code from this PR to save you time while searching for the right values https://github.com/nicoboss/llama.cpp/pull/3/files this basically just skips actually allocating the buffers but prints how much would be allocated. Obviously this won't work for actually running the model and may not handle every edge case ( also the code is for llama.cpp which has diverted in ways that will make you manually port over some of the changes, so not sure if you will find it worthwhile ).
> 
> > I think @saood06 was mentioning somewhere that one needs to "warm up" the model for quite some time before performance becomes more stable, perhaps this is also true for your system.
> 
> That problem should no longer occur anywhere unless you pass the --no-warmup argument. It occurred because the old warmup code only worked for dense models, MoEs were only being partially loaded in as it would only activate a single tokens worth of active experts. The code now activates all experts during the warmup phase. This was very noticeable if you looked at disk I/O and before I would only post performance numbers once disk I/O was no longer happening, and on my setup where the model was stored on a HDD with slow seek times it definitely mattered even when the amount of data being read was low but not zero.

This has been really helpful. I was able to use the dry run approach to get a faster feedback loop on allocating to GPUs, so thank you! 

I also tried to allocate those tensors to CUDA0 without any luck. I got a different error, but still an error. Can’t remember it offhand, but if it’s useful to solve the split mode issues @ikawrakow let me know and I’ll give it a go again!

---

👤 **ikawrakow** commented the **2025-03-06** at **09:58:44**:<br>

Do I understand correctly that the `IQ4_KSS` model works correctly with MLA but produces NaNs with FA? Or does it always produce NaNs?

---

👤 **davidsyoung** commented the **2025-03-06** at **10:00:34**:<br>

> Do I understand correctly that the `IQ4_KSS` model works correctly with MLA but produces NaNs with FA? Or does it always produce NaNs?

I haven’t ran perplexity with MLA, only FA - which produced NANs. I then loaded the model myself, for inference, using MLA (assuming I messed up the quant somehow), but it worked. 

I’m now loading the model with FA now for inference, to see if it’s an issue running with perplexity, or FA itself.

---

👤 **davidsyoung** commented the **2025-03-06** at **10:07:40**:<br>

OK, update. Model works with FA. Just doesn’t run under perplexity. Weird. Any idea?

---

👤 **ikawrakow** commented the **2025-03-06** at **13:23:40**:<br>

Not sure. It works with the models I have tested with.

---

👤 **davidsyoung** commented the **2025-03-06** at **13:58:27**:<br>

> Not sure. It works with the models I have tested with.

Strange. Ignore it for now, maybe it's something I did wrong with quant and merges.txt issue.

Anyway. I'm working on spreading the components of the experts over 14/15 GPUs, but the KV cache/compute buffer is still getting spread over all GPUs.

Would it be possible to get a parameter to decide what GPU's to split the KV Cache (and compute buffer if possible, but not sure?) over? Similar to `-ts`, or even a more crude implementation. It doesn't have to be perfect, but would definitely help to better spread KV cache/compute buffers and make better use of vram!

---

👤 **ikawrakow** commented the **2025-03-06** at **14:05:32**:<br>

>  I'm working on spreading the components of the experts over 14/15 GPUs, but the KV cache/compute buffer is still getting spread over all GPUs.

I was wondering about that myself. It is not code that I wrote, so I don't know (yet) why this happens. The behavior should be that if all tensors involved with attention calculations for a given layer are on a given GPU (or, more generally, given back-end), the associated KV cache should be all on that back-end and nowhere else. 

What happens if you try standard attention. Use a short context (`-c 512`) to not finish VRAM. Do we get the KV cache spread around the GPU's in that case, or is it still so that each GPU has the entire KV cache for all layers?

---

👤 **davidsyoung** commented the **2025-03-06** at **14:48:40**:<br>

> > I'm working on spreading the components of the experts over 14/15 GPUs, but the KV cache/compute buffer is still getting spread over all GPUs.
> 
> I was wondering about that myself. It is not code that I wrote, so I don't know (yet) why this happens. The behavior should be that if all tensors involved with attention calculations for a given layer are on a given GPU (or, more generally, given back-end), the associated KV cache should be all on that back-end and nowhere else.
> 
> What happens if you try standard attention. Use a short context (`-c 512`) to not finish VRAM. Do we get the KV cache spread around the GPU's in that case, or is it still so that each GPU has the entire KV cache for all layers?

Unfortunately I don’t have much time today to test this. But, tbh, I don’t think it’ll be as much of an issue when MLA FA is implemented. I may need to specify each attention tensor so that it doesn’t create a backend for each GPU.

---

👤 **davidsyoung** commented the **2025-03-07** at **00:22:47**:<br>

@ikawrakow 

I don't suppose you can see anything I'm doing (obviously) wrong here? Might just be tired.

I have successfully spread all tensors out equally across GPU's, and now I'm just trying to get it to load. However, compute buffers are being allocated only to CUDA0. All I can think of is that I'm moving around tensors in a way I shouldn't, and as a result, it's causing issues with compute buffer allocations. Latest repo version. It seems as though if the compute buffer was split across all 16 GPU's, instead of just one, it would work out around the right amount of expected compute buffer. TYVM in advance. 

```
-m /models/DeepSeek-R1-GGUF/DeepSeek-R1-GGUF-IQ4_KSS.gguf
      -mla 2
      -fmoe
      --split-mode layer
      -ot "blk\.59\.ffn_gate_exps\.weight|blk\.46\.ffn_up_exps\.weight|blk\.59\.ffn_up_exps\.weight|blk\.50\.ffn_up_exps\.weight|blk\.60\.ffn_down_exps\.weight|blk\.54\.ffn_up_exps\.weight|blk\.60\.ffn_gate_exps\.weight|blk\.58\.ffn_up_exps\.weight|blk\.60\.ffn_up_exps\.weight|blk\.42\.ffn_up_exps\.weight=CUDA0"
      -ot "blk\.3\.ffn_(down|gate|up)_exps\.weight|blk\.4\.ffn_(down|gate|up)_exps\.weight|blk\.5\.ffn_(down|gate|up)_exps\.weight|blk\.6\.ffn_(down|gate)_exps\.weight=CUDA1" 
      -ot "blk\.7\.ffn_(down|gate|up)_exps\.weight|blk\.8\.ffn_(down|gate|up)_exps\.weight|blk\.9\.ffn_(down|gate|up)_exps\.weight|blk\.10\.ffn_(down|gate)_exps\.weight=CUDA2" 
      -ot "blk\.11\.ffn_(down|gate|up)_exps\.weight|blk\.12\.ffn_(down|gate|up)_exps\.weight|blk\.13\.ffn_(down|gate|up)_exps\.weight|blk\.14\.ffn_(down|gate)_exps\.weight=CUDA3" 
      -ot "blk\.15\.ffn_(down|gate|up)_exps\.weight|blk\.16\.ffn_(down|gate|up)_exps\.weight|blk\.17\.ffn_(down|gate|up)_exps\.weight|blk\.18\.ffn_(down|gate)_exps\.weight=CUDA4" 
      -ot "blk\.19\.ffn_(down|gate|up)_exps\.weight|blk\.20\.ffn_(down|gate|up)_exps\.weight|blk\.21\.ffn_(down|gate|up)_exps\.weight|blk\.22\.ffn_(down|gate)_exps\.weight=CUDA5" 
      -ot "blk\.23\.ffn_(down|gate|up)_exps\.weight|blk\.24\.ffn_(down|gate|up)_exps\.weight|blk\.25\.ffn_(down|gate|up)_exps\.weight|blk\.26\.ffn_(down|gate)_exps\.weight=CUDA6" 
      -ot "blk\.27\.ffn_(down|gate|up)_exps\.weight|blk\.28\.ffn_(down|gate|up)_exps\.weight|blk\.29\.ffn_(down|gate|up)_exps\.weight|blk\.30\.ffn_(down|gate)_exps\.weight=CUDA7" 
      -ot "blk\.31\.ffn_(down|gate|up)_exps\.weight|blk\.32\.ffn_(down|gate|up)_exps\.weight|blk\.33\.ffn_(down|gate|up)_exps\.weight|blk\.34\.ffn_(down|gate)_exps\.weight=CUDA8" 
      -ot "blk\.35\.ffn_(down|gate|up)_exps\.weight|blk\.36\.ffn_(down|gate|up)_exps\.weight|blk\.37\.ffn_(down|gate|up)_exps\.weight|blk\.38\.ffn_(down|gate)_exps\.weight=CUDA9" 
      -ot "blk\.39\.ffn_(down|gate|up)_exps\.weight|blk\.40\.ffn_(down|gate|up)_exps\.weight|blk\.41\.ffn_(down|gate|up)_exps\.weight|blk\.42\.ffn_(down|gate)_exps\.weight=CUDA10" 
      -ot "blk\.43\.ffn_(down|gate|up)_exps\.weight|blk\.44\.ffn_(down|gate|up)_exps\.weight|blk\.45\.ffn_(down|gate|up)_exps\.weight|blk\.46\.ffn_(down|gate)_exps\.weight=CUDA11" 
      -ot "blk\.47\.ffn_(down|gate|up)_exps\.weight|blk\.48\.ffn_(down|gate|up)_exps\.weight|blk\.49\.ffn_(down|gate|up)_exps\.weight|blk\.50\.ffn_(down|gate)_exps\.weight=CUDA12" 
      -ot "blk\.51\.ffn_(down|gate|up)_exps\.weight|blk\.52\.ffn_(down|gate|up)_exps\.weight|blk\.53\.ffn_(down|gate|up)_exps\.weight|blk\.54\.ffn_(down|gate)_exps\.weight=CUDA13" 
      -ot "blk\.55\.ffn_(down|gate|up)_exps\.weight|blk\.56\.ffn_(down|gate|up)_exps\.weight|blk\.57\.ffn_(down|gate|up)_exps\.weight|blk\.58\.ffn_(down|gate)_exps\.weight=CUDA14" 
      -ot "blk\.6\.ffn_up_exps\.weight|blk\.10\.ffn_up_exps\.weight|blk\.14\.ffn_up_exps\.weight|blk\.18\.ffn_up_exps\.weight|blk\.22\.ffn_up_exps\.weight|blk\.26\.ffn_up_exps\.weight|blk\.30\.ffn_up_exps\.weight|blk\.34\.ffn_up_exps\.weight|blk\.38\.ffn_up_exps\.weight|blk\.59\.ffn_down_exps\.weight=CUDA15"
      -ts 24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24
      -b 2048
      -ub 1024
      -amb 1024
      --temp 0.5
      --ctx-size 8192
      --seed 3407
      --n-gpu-layers 100
      --host 0.0.0.0
      --port 8080
```

```
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 16 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 1: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 2: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 3: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 4: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 5: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 6: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 7: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 8: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 9: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 10: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 11: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 12: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 13: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 14: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 15: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
INFO [                    main] build info | tid="22486911213568" timestamp=1741305679 build=0 commit="unknown"
INFO [                    main] system info | tid="22486911213568" timestamp=1741305679 n_threads=64 n_threads_batch=-1 total_threads=128 system_info="AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: loaded meta data with 53 key-value pairs and 1147 tensors from /models/DeepSeek-R1-GGUF/DeepSeek-R1-GGUF-IQ4_KSS.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = unsloth_DeepSeek R1 BF16
llama_model_loader: - kv   3:                         general.size_label str              = 256x21B
llama_model_loader: - kv   4:                            general.license str              = mit
llama_model_loader: - kv   5:                   general.base_model.count u32              = 1
llama_model_loader: - kv   6:                  general.base_model.0.name str              = DeepSeek R1
llama_model_loader: - kv   7:          general.base_model.0.organization str              = Deepseek Ai
llama_model_loader: - kv   8:              general.base_model.0.repo_url str              = https://huggingface.co/deepseek-ai/De...
llama_model_loader: - kv   9:                               general.tags arr[str,3]       = ["deepseek", "unsloth", "transformers"]
llama_model_loader: - kv  10:                          general.languages arr[str,1]       = ["en"]
llama_model_loader: - kv  11:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv  12:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv  13:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv  14:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  15:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv  16:          deepseek2.attention.head_count_kv u32              = 128
llama_model_loader: - kv  17:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  18: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  19:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  20:                          general.file_type u32              = 148
llama_model_loader: - kv  21:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  22:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  23:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  24:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  25:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  26:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  27:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  28:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  29:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  30:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  31:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  32:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  33:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  34:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  35:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  36: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  37: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  38:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  39:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  40:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<�...
llama_model_loader: - kv  41:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  42:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  43:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  44:            tokenizer.ggml.padding_token_id u32              = 128815
llama_model_loader: - kv  45:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  46:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  47:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  48:               general.quantization_version u32              = 2
llama_model_loader: - kv  49:                      quantize.imatrix.file str              = /models/deepseek-config/imatrix.dat
llama_model_loader: - kv  50:                   quantize.imatrix.dataset str              = imatrix-training-full-3
llama_model_loader: - kv  51:             quantize.imatrix.entries_count i32              = 720
llama_model_loader: - kv  52:              quantize.imatrix.chunks_count i32              = 315
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  306 tensors
llama_model_loader: - type q6_K:    1 tensors
llama_model_loader: - type iq4_kss:  479 tensors
loaded 127741 merges from merges.txt
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
llm_load_print_meta: model ftype      = IQ4_KSS - 4.0 bpw
llm_load_print_meta: model params     = 672.050 B
llm_load_print_meta: model size       = 317.185 GiB (4.054 BPW) 
llm_load_print_meta: repeating layers = 315.560 GiB (4.045 BPW, 670.196 B parameters)
llm_load_print_meta: general.name     = unsloth_DeepSeek R1 BF16
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
llm_load_tensors: ggml ctx size =    7.94 MiB
Tensor blk.3.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.3.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.3.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.4.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.4.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.4.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.6.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.6.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.6.ffn_up_exps.weight buffer type overriden to CUDA15
Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CUDA2
Tensor blk.7.ffn_down_exps.weight buffer type overriden to CUDA2
Tensor blk.7.ffn_up_exps.weight buffer type overriden to CUDA2
Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CUDA2
Tensor blk.8.ffn_down_exps.weight buffer type overriden to CUDA2
Tensor blk.8.ffn_up_exps.weight buffer type overriden to CUDA2
Tensor blk.9.ffn_gate_exps.weight buffer type overriden to CUDA2
Tensor blk.9.ffn_down_exps.weight buffer type overriden to CUDA2
Tensor blk.9.ffn_up_exps.weight buffer type overriden to CUDA2
Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CUDA2
Tensor blk.10.ffn_down_exps.weight buffer type overriden to CUDA2
Tensor blk.10.ffn_up_exps.weight buffer type overriden to CUDA15
Tensor blk.11.ffn_gate_exps.weight buffer type overriden to CUDA3
Tensor blk.11.ffn_down_exps.weight buffer type overriden to CUDA3
Tensor blk.11.ffn_up_exps.weight buffer type overriden to CUDA3
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CUDA3
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CUDA3
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CUDA3
Tensor blk.13.ffn_gate_exps.weight buffer type overriden to CUDA3
Tensor blk.13.ffn_down_exps.weight buffer type overriden to CUDA3
Tensor blk.13.ffn_up_exps.weight buffer type overriden to CUDA3
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CUDA3
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CUDA3
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CUDA15
Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CUDA4
Tensor blk.15.ffn_down_exps.weight buffer type overriden to CUDA4
Tensor blk.15.ffn_up_exps.weight buffer type overriden to CUDA4
Tensor blk.16.ffn_gate_exps.weight buffer type overriden to CUDA4
Tensor blk.16.ffn_down_exps.weight buffer type overriden to CUDA4
Tensor blk.16.ffn_up_exps.weight buffer type overriden to CUDA4
Tensor blk.17.ffn_gate_exps.weight buffer type overriden to CUDA4
Tensor blk.17.ffn_down_exps.weight buffer type overriden to CUDA4
Tensor blk.17.ffn_up_exps.weight buffer type overriden to CUDA4
Tensor blk.18.ffn_gate_exps.weight buffer type overriden to CUDA4
Tensor blk.18.ffn_down_exps.weight buffer type overriden to CUDA4
Tensor blk.18.ffn_up_exps.weight buffer type overriden to CUDA15
Tensor blk.19.ffn_gate_exps.weight buffer type overriden to CUDA5
Tensor blk.19.ffn_down_exps.weight buffer type overriden to CUDA5
Tensor blk.19.ffn_up_exps.weight buffer type overriden to CUDA5
Tensor blk.20.ffn_gate_exps.weight buffer type overriden to CUDA5
Tensor blk.20.ffn_down_exps.weight buffer type overriden to CUDA5
Tensor blk.20.ffn_up_exps.weight buffer type overriden to CUDA5
Tensor blk.21.ffn_gate_exps.weight buffer type overriden to CUDA5
Tensor blk.21.ffn_down_exps.weight buffer type overriden to CUDA5
Tensor blk.21.ffn_up_exps.weight buffer type overriden to CUDA5
Tensor blk.22.ffn_gate_exps.weight buffer type overriden to CUDA5
Tensor blk.22.ffn_down_exps.weight buffer type overriden to CUDA5
Tensor blk.22.ffn_up_exps.weight buffer type overriden to CUDA15
Tensor blk.23.ffn_gate_exps.weight buffer type overriden to CUDA6
Tensor blk.23.ffn_down_exps.weight buffer type overriden to CUDA6
Tensor blk.23.ffn_up_exps.weight buffer type overriden to CUDA6
Tensor blk.24.ffn_gate_exps.weight buffer type overriden to CUDA6
Tensor blk.24.ffn_down_exps.weight buffer type overriden to CUDA6
Tensor blk.24.ffn_up_exps.weight buffer type overriden to CUDA6
Tensor blk.25.ffn_gate_exps.weight buffer type overriden to CUDA6
Tensor blk.25.ffn_down_exps.weight buffer type overriden to CUDA6
Tensor blk.25.ffn_up_exps.weight buffer type overriden to CUDA6
Tensor blk.26.ffn_gate_exps.weight buffer type overriden to CUDA6
Tensor blk.26.ffn_down_exps.weight buffer type overriden to CUDA6
Tensor blk.26.ffn_up_exps.weight buffer type overriden to CUDA15
Tensor blk.27.ffn_gate_exps.weight buffer type overriden to CUDA7
Tensor blk.27.ffn_down_exps.weight buffer type overriden to CUDA7
Tensor blk.27.ffn_up_exps.weight buffer type overriden to CUDA7
Tensor blk.28.ffn_gate_exps.weight buffer type overriden to CUDA7
Tensor blk.28.ffn_down_exps.weight buffer type overriden to CUDA7
Tensor blk.28.ffn_up_exps.weight buffer type overriden to CUDA7
Tensor blk.29.ffn_gate_exps.weight buffer type overriden to CUDA7
Tensor blk.29.ffn_down_exps.weight buffer type overriden to CUDA7
Tensor blk.29.ffn_up_exps.weight buffer type overriden to CUDA7
Tensor blk.30.ffn_gate_exps.weight buffer type overriden to CUDA7
Tensor blk.30.ffn_down_exps.weight buffer type overriden to CUDA7
Tensor blk.30.ffn_up_exps.weight buffer type overriden to CUDA15
Tensor blk.31.ffn_gate_exps.weight buffer type overriden to CUDA8
Tensor blk.31.ffn_down_exps.weight buffer type overriden to CUDA8
Tensor blk.31.ffn_up_exps.weight buffer type overriden to CUDA8
Tensor blk.32.ffn_gate_exps.weight buffer type overriden to CUDA8
Tensor blk.32.ffn_down_exps.weight buffer type overriden to CUDA8
Tensor blk.32.ffn_up_exps.weight buffer type overriden to CUDA8
Tensor blk.33.ffn_gate_exps.weight buffer type overriden to CUDA8
Tensor blk.33.ffn_down_exps.weight buffer type overriden to CUDA8
Tensor blk.33.ffn_up_exps.weight buffer type overriden to CUDA8
Tensor blk.34.ffn_gate_exps.weight buffer type overriden to CUDA8
Tensor blk.34.ffn_down_exps.weight buffer type overriden to CUDA8
Tensor blk.34.ffn_up_exps.weight buffer type overriden to CUDA15
Tensor blk.35.ffn_gate_exps.weight buffer type overriden to CUDA9
Tensor blk.35.ffn_down_exps.weight buffer type overriden to CUDA9
Tensor blk.35.ffn_up_exps.weight buffer type overriden to CUDA9
Tensor blk.36.ffn_gate_exps.weight buffer type overriden to CUDA9
Tensor blk.36.ffn_down_exps.weight buffer type overriden to CUDA9
Tensor blk.36.ffn_up_exps.weight buffer type overriden to CUDA9
Tensor blk.37.ffn_gate_exps.weight buffer type overriden to CUDA9
Tensor blk.37.ffn_down_exps.weight buffer type overriden to CUDA9
Tensor blk.37.ffn_up_exps.weight buffer type overriden to CUDA9
Tensor blk.38.ffn_gate_exps.weight buffer type overriden to CUDA9
Tensor blk.38.ffn_down_exps.weight buffer type overriden to CUDA9
Tensor blk.38.ffn_up_exps.weight buffer type overriden to CUDA15
Tensor blk.39.ffn_gate_exps.weight buffer type overriden to CUDA10
Tensor blk.39.ffn_down_exps.weight buffer type overriden to CUDA10
Tensor blk.39.ffn_up_exps.weight buffer type overriden to CUDA10
Tensor blk.40.ffn_gate_exps.weight buffer type overriden to CUDA10
Tensor blk.40.ffn_down_exps.weight buffer type overriden to CUDA10
Tensor blk.40.ffn_up_exps.weight buffer type overriden to CUDA10
Tensor blk.41.ffn_gate_exps.weight buffer type overriden to CUDA10
Tensor blk.41.ffn_down_exps.weight buffer type overriden to CUDA10
Tensor blk.41.ffn_up_exps.weight buffer type overriden to CUDA10
Tensor blk.42.ffn_gate_exps.weight buffer type overriden to CUDA10
Tensor blk.42.ffn_down_exps.weight buffer type overriden to CUDA10
Tensor blk.42.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.43.ffn_gate_exps.weight buffer type overriden to CUDA11
Tensor blk.43.ffn_down_exps.weight buffer type overriden to CUDA11
Tensor blk.43.ffn_up_exps.weight buffer type overriden to CUDA11
Tensor blk.44.ffn_gate_exps.weight buffer type overriden to CUDA11
Tensor blk.44.ffn_down_exps.weight buffer type overriden to CUDA11
Tensor blk.44.ffn_up_exps.weight buffer type overriden to CUDA11
Tensor blk.45.ffn_gate_exps.weight buffer type overriden to CUDA11
Tensor blk.45.ffn_down_exps.weight buffer type overriden to CUDA11
Tensor blk.45.ffn_up_exps.weight buffer type overriden to CUDA11
Tensor blk.46.ffn_gate_exps.weight buffer type overriden to CUDA11
Tensor blk.46.ffn_down_exps.weight buffer type overriden to CUDA11
Tensor blk.46.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.47.ffn_gate_exps.weight buffer type overriden to CUDA12
Tensor blk.47.ffn_down_exps.weight buffer type overriden to CUDA12
Tensor blk.47.ffn_up_exps.weight buffer type overriden to CUDA12
Tensor blk.48.ffn_gate_exps.weight buffer type overriden to CUDA12
Tensor blk.48.ffn_down_exps.weight buffer type overriden to CUDA12
Tensor blk.48.ffn_up_exps.weight buffer type overriden to CUDA12
Tensor blk.49.ffn_gate_exps.weight buffer type overriden to CUDA12
Tensor blk.49.ffn_down_exps.weight buffer type overriden to CUDA12
Tensor blk.49.ffn_up_exps.weight buffer type overriden to CUDA12
Tensor blk.50.ffn_gate_exps.weight buffer type overriden to CUDA12
Tensor blk.50.ffn_down_exps.weight buffer type overriden to CUDA12
Tensor blk.50.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.51.ffn_gate_exps.weight buffer type overriden to CUDA13
Tensor blk.51.ffn_down_exps.weight buffer type overriden to CUDA13
Tensor blk.51.ffn_up_exps.weight buffer type overriden to CUDA13
Tensor blk.52.ffn_gate_exps.weight buffer type overriden to CUDA13
Tensor blk.52.ffn_down_exps.weight buffer type overriden to CUDA13
Tensor blk.52.ffn_up_exps.weight buffer type overriden to CUDA13
Tensor blk.53.ffn_gate_exps.weight buffer type overriden to CUDA13
Tensor blk.53.ffn_down_exps.weight buffer type overriden to CUDA13
Tensor blk.53.ffn_up_exps.weight buffer type overriden to CUDA13
Tensor blk.54.ffn_gate_exps.weight buffer type overriden to CUDA13
Tensor blk.54.ffn_down_exps.weight buffer type overriden to CUDA13
Tensor blk.54.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.55.ffn_gate_exps.weight buffer type overriden to CUDA14
Tensor blk.55.ffn_down_exps.weight buffer type overriden to CUDA14
Tensor blk.55.ffn_up_exps.weight buffer type overriden to CUDA14
Tensor blk.56.ffn_gate_exps.weight buffer type overriden to CUDA14
Tensor blk.56.ffn_down_exps.weight buffer type overriden to CUDA14
Tensor blk.56.ffn_up_exps.weight buffer type overriden to CUDA14
Tensor blk.57.ffn_gate_exps.weight buffer type overriden to CUDA14
Tensor blk.57.ffn_down_exps.weight buffer type overriden to CUDA14
Tensor blk.57.ffn_up_exps.weight buffer type overriden to CUDA14
Tensor blk.58.ffn_gate_exps.weight buffer type overriden to CUDA14
Tensor blk.58.ffn_down_exps.weight buffer type overriden to CUDA14
Tensor blk.58.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.59.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.59.ffn_down_exps.weight buffer type overriden to CUDA15
Tensor blk.59.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.60.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.60.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.60.ffn_up_exps.weight buffer type overriden to CUDA0
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size =   938.98 MiB
llm_load_tensors:      CUDA0 buffer size = 19097.52 MiB
llm_load_tensors:      CUDA1 buffer size = 20423.15 MiB
llm_load_tensors:      CUDA2 buffer size = 20423.15 MiB
llm_load_tensors:      CUDA3 buffer size = 20423.15 MiB
llm_load_tensors:      CUDA4 buffer size = 20423.15 MiB
llm_load_tensors:      CUDA5 buffer size = 20423.15 MiB
llm_load_tensors:      CUDA6 buffer size = 20423.15 MiB
llm_load_tensors:      CUDA7 buffer size = 20255.86 MiB
llm_load_tensors:      CUDA8 buffer size = 20423.15 MiB
llm_load_tensors:      CUDA9 buffer size = 20423.15 MiB
llm_load_tensors:     CUDA10 buffer size = 20423.15 MiB
llm_load_tensors:     CUDA11 buffer size = 20423.15 MiB
llm_load_tensors:     CUDA12 buffer size = 20423.15 MiB
llm_load_tensors:     CUDA13 buffer size = 20423.15 MiB
llm_load_tensors:     CUDA14 buffer size = 20423.15 MiB
llm_load_tensors:     CUDA15 buffer size = 19004.55 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 8192
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 1024
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: mla_attn   = 2
llama_new_context_with_model: attn_max_b = 1024
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
llama_kv_cache_init:      CUDA0 KV buffer size =    36.00 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =    36.00 MiB
llama_kv_cache_init:      CUDA2 KV buffer size =    36.00 MiB
llama_kv_cache_init:      CUDA3 KV buffer size =    36.00 MiB
llama_kv_cache_init:      CUDA4 KV buffer size =    36.00 MiB
llama_kv_cache_init:      CUDA5 KV buffer size =    36.00 MiB
llama_kv_cache_init:      CUDA6 KV buffer size =    36.00 MiB
llama_kv_cache_init:      CUDA7 KV buffer size =    27.00 MiB
llama_kv_cache_init:      CUDA8 KV buffer size =    36.00 MiB
llama_kv_cache_init:      CUDA9 KV buffer size =    36.00 MiB
llama_kv_cache_init:     CUDA10 KV buffer size =    36.00 MiB
llama_kv_cache_init:     CUDA11 KV buffer size =    36.00 MiB
llama_kv_cache_init:     CUDA12 KV buffer size =    36.00 MiB
llama_kv_cache_init:     CUDA13 KV buffer size =    36.00 MiB
llama_kv_cache_init:     CUDA14 KV buffer size =    36.00 MiB
llama_kv_cache_init:     CUDA15 KV buffer size =    18.00 MiB
llama_new_context_with_model: KV self size  =  549.00 MiB, c^KV (f16):  549.00 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.99 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=4)
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 38999.99 MiB on device 0: cudaMalloc failed: out of memory
ggml_gallocr_reserve_n: failed to allocate CUDA0 buffer of size 40894455296
llama_new_context_with_model: failed to allocate compute buffers
llama_init_from_gpt_params: error: failed to create context with model '/models/DeepSeek-R1-GGUF/DeepSeek-R1-GGUF-IQ4_KSS.gguf'
 ERR [              load_model] unable to load model | tid="22486911213568" timestamp=1741306387 model="/models/DeepSeek-R1-GGUF/DeepSeek-R1-GGUF-IQ4_KSS.gguf"
/app/.devops/tools_new.sh: line 47:    13 Segmentation fault      ./llama-server "$@"
```

Again, I could be doing something v obviously wrong here, but my brain can't make sense of it. Thank you!

---

👤 **ikawrakow** commented the **2025-03-07** at **05:33:17**:<br>

Not sure. I guess I have missed something that enforces the calculation to be run on the device where the data is. Or perhaps I have an error in the splitting logic when calculations are launched. The split looks really nice, too bad it does not work. Can you try without `-fmoe`?

---

👤 **davidsyoung** commented the **2025-03-07** at **10:47:05**:<br>

> Not sure. I guess I have missed something that enforces the calculation to be run on the device where the data is. Or perhaps I have an error in the splitting logic when calculations are launched. The split looks really nice, too bad it does not work. Can you try without `-fmoe`? I have no access to a multi-GPU system, so not able to debug.

Of course, happy to debug as much as I can! 

So I realised that I had an unbalanced amount of up/gate/down tensors (I had many _up_ tensors on CUDA0/CUDA15 and that was allocating a high amount of compute buffer on that GPU).

So I balanced them best I could across the GPUs. I'm not 100% clear which tensors require the most compute yet, but preliminarily it seems the up and potentially down tensors too.

It also seems that when -fmoe is set, there's a higher amount of compute buffer allocated to certain GPU's. I've got some runs here for you to look at:

---

# Parameters
```
      -s
      -m /models/DeepSeek-R1-GGUF/DeepSeek-R1-GGUF-IQ4_KSS.gguf
      -mla 2
      -ts 24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24
      -ot "blk\.3\.ffn_down_exps\.weight|blk\.3\.ffn_gate_exps\.weight|blk\.3\.ffn_up_exps\.weight|blk\.4\.ffn_down_exps\.weight|blk\.4\.ffn_gate_exps\.weight|blk\.4\.ffn_up_exps\.weight|blk\.5\.ffn_down_exps\.weight|blk\.5\.ffn_gate_exps\.weight|blk\.5\.ffn_up_exps\.weight|blk\.6\.ffn_down_exps\.weight=CUDA0"
      -ot "blk\.6\.ffn_gate_exps\.weight|blk\.6\.ffn_up_exps\.weight|blk\.7\.ffn_down_exps\.weight|blk\.7\.ffn_gate_exps\.weight|blk\.7\.ffn_up_exps\.weight|blk\.8\.ffn_down_exps\.weight|blk\.8\.ffn_gate_exps\.weight|blk\.8\.ffn_up_exps\.weight|blk\.9\.ffn_down_exps\.weight|blk\.9\.ffn_gate_exps\.weight|blk\.9\.ffn_up_exps\.weight=CUDA1"
      -ot "blk\.10\.ffn_down_exps\.weight|blk\.10\.ffn_gate_exps\.weight|blk\.10\.ffn_up_exps\.weight|blk\.11\.ffn_down_exps\.weight|blk\.11\.ffn_gate_exps\.weight|blk\.11\.ffn_up_exps\.weight|blk\.12\.ffn_down_exps\.weight|blk\.12\.ffn_gate_exps\.weight|blk\.12\.ffn_up_exps\.weight|blk\.13\.ffn_down_exps\.weight|blk\.13\.ffn_gate_exps\.weight=CUDA2"
      -ot "blk\.13\.ffn_up_exps\.weight|blk\.14\.ffn_down_exps\.weight|blk\.14\.ffn_gate_exps\.weight|blk\.14\.ffn_up_exps\.weight|blk\.15\.ffn_down_exps\.weight|blk\.15\.ffn_gate_exps\.weight|blk\.15\.ffn_up_exps\.weight|blk\.16\.ffn_down_exps\.weight|blk\.16\.ffn_gate_exps\.weight|blk\.16\.ffn_up_exps\.weight|blk\.17\.ffn_down_exps\.weight=CUDA3"
      -ot "blk\.17\.ffn_gate_exps\.weight|blk\.17\.ffn_up_exps\.weight|blk\.18\.ffn_down_exps\.weight|blk\.18\.ffn_gate_exps\.weight|blk\.18\.ffn_up_exps\.weight|blk\.19\.ffn_down_exps\.weight|blk\.19\.ffn_gate_exps\.weight|blk\.19\.ffn_up_exps\.weight|blk\.20\.ffn_down_exps\.weight|blk\.20\.ffn_gate_exps\.weight|blk\.20\.ffn_up_exps\.weight=CUDA4"
      -ot "blk\.21\.ffn_down_exps\.weight|blk\.21\.ffn_gate_exps\.weight|blk\.21\.ffn_up_exps\.weight|blk\.22\.ffn_down_exps\.weight|blk\.22\.ffn_gate_exps\.weight|blk\.22\.ffn_up_exps\.weight|blk\.23\.ffn_down_exps\.weight|blk\.23\.ffn_gate_exps\.weight|blk\.23\.ffn_up_exps\.weight|blk\.24\.ffn_down_exps\.weight|blk\.24\.ffn_gate_exps\.weight=CUDA5"
      -ot "blk\.24\.ffn_up_exps\.weight|blk\.25\.ffn_down_exps\.weight|blk\.25\.ffn_gate_exps\.weight|blk\.25\.ffn_up_exps\.weight|blk\.26\.ffn_down_exps\.weight|blk\.26\.ffn_gate_exps\.weight|blk\.26\.ffn_up_exps\.weight|blk\.27\.ffn_down_exps\.weight|blk\.27\.ffn_gate_exps\.weight|blk\.27\.ffn_up_exps\.weight|blk\.28\.ffn_down_exps\.weight=CUDA6"
      -ot "blk\.28\.ffn_gate_exps\.weight|blk\.28\.ffn_up_exps\.weight|blk\.29\.ffn_down_exps\.weight|blk\.29\.ffn_gate_exps\.weight|blk\.29\.ffn_up_exps\.weight|blk\.30\.ffn_down_exps\.weight|blk\.30\.ffn_gate_exps\.weight|blk\.30\.ffn_up_exps\.weight|blk\.31\.ffn_down_exps\.weight|blk\.31\.ffn_gate_exps\.weight|blk\.31\.ffn_up_exps\.weight=CUDA7"
      -ot "blk\.32\.ffn_down_exps\.weight|blk\.32\.ffn_gate_exps\.weight|blk\.32\.ffn_up_exps\.weight|blk\.33\.ffn_down_exps\.weight|blk\.33\.ffn_gate_exps\.weight|blk\.33\.ffn_up_exps\.weight|blk\.34\.ffn_down_exps\.weight|blk\.34\.ffn_gate_exps\.weight|blk\.34\.ffn_up_exps\.weight|blk\.35\.ffn_down_exps\.weight|blk\.35\.ffn_gate_exps\.weight=CUDA8"
      -ot "blk\.35\.ffn_up_exps\.weight|blk\.36\.ffn_down_exps\.weight|blk\.36\.ffn_gate_exps\.weight|blk\.36\.ffn_up_exps\.weight|blk\.37\.ffn_down_exps\.weight|blk\.37\.ffn_gate_exps\.weight|blk\.37\.ffn_up_exps\.weight|blk\.38\.ffn_down_exps\.weight|blk\.38\.ffn_gate_exps\.weight|blk\.38\.ffn_up_exps\.weight|blk\.39\.ffn_down_exps\.weight=CUDA9"
      -ot "blk\.39\.ffn_gate_exps\.weight|blk\.39\.ffn_up_exps\.weight|blk\.40\.ffn_down_exps\.weight|blk\.40\.ffn_gate_exps\.weight|blk\.40\.ffn_up_exps\.weight|blk\.41\.ffn_down_exps\.weight|blk\.41\.ffn_gate_exps\.weight|blk\.41\.ffn_up_exps\.weight|blk\.42\.ffn_down_exps\.weight|blk\.42\.ffn_gate_exps\.weight|blk\.42\.ffn_up_exps\.weight=CUDA10"
      -ot "blk\.43\.ffn_down_exps\.weight|blk\.43\.ffn_gate_exps\.weight|blk\.43\.ffn_up_exps\.weight|blk\.44\.ffn_down_exps\.weight|blk\.44\.ffn_gate_exps\.weight|blk\.44\.ffn_up_exps\.weight|blk\.45\.ffn_down_exps\.weight|blk\.45\.ffn_gate_exps\.weight|blk\.45\.ffn_up_exps\.weight|blk\.46\.ffn_down_exps\.weight|blk\.46\.ffn_gate_exps\.weight=CUDA11"
      -ot "blk\.46\.ffn_up_exps\.weight|blk\.47\.ffn_down_exps\.weight|blk\.47\.ffn_gate_exps\.weight|blk\.47\.ffn_up_exps\.weight|blk\.48\.ffn_down_exps\.weight|blk\.48\.ffn_gate_exps\.weight|blk\.48\.ffn_up_exps\.weight|blk\.49\.ffn_down_exps\.weight|blk\.49\.ffn_gate_exps\.weight|blk\.49\.ffn_up_exps\.weight|blk\.50\.ffn_down_exps\.weight=CUDA12"
      -ot "blk\.50\.ffn_gate_exps\.weight|blk\.50\.ffn_up_exps\.weight|blk\.51\.ffn_down_exps\.weight|blk\.51\.ffn_gate_exps\.weight|blk\.51\.ffn_up_exps\.weight|blk\.52\.ffn_down_exps\.weight|blk\.52\.ffn_gate_exps\.weight|blk\.52\.ffn_up_exps\.weight|blk\.53\.ffn_down_exps\.weight|blk\.53\.ffn_gate_exps\.weight|blk\.53\.ffn_up_exps\.weight=CUDA13"
      -ot "blk\.54\.ffn_down_exps\.weight|blk\.54\.ffn_gate_exps\.weight|blk\.54\.ffn_up_exps\.weight|blk\.55\.ffn_down_exps\.weight|blk\.55\.ffn_gate_exps\.weight|blk\.55\.ffn_up_exps\.weight|blk\.56\.ffn_down_exps\.weight|blk\.56\.ffn_gate_exps\.weight|blk\.56\.ffn_up_exps\.weight|blk\.57\.ffn_down_exps\.weight|blk\.57\.ffn_gate_exps\.weight=CUDA14"
      -ot "blk\.57\.ffn_up_exps\.weight|blk\.58\.ffn_down_exps\.weight|blk\.58\.ffn_gate_exps\.weight|blk\.58\.ffn_up_exps\.weight|blk\.59\.ffn_down_exps\.weight|blk\.59\.ffn_gate_exps\.weight|blk\.59\.ffn_up_exps\.weight|blk\.60\.ffn_down_exps\.weight|blk\.60\.ffn_gate_exps\.weight|blk\.60\.ffn_up_exps\.weight=CUDA15"
      -b 2048
      -ub 1024
      -amb 64
      --temp 0.5
      --ctx-size 8192
      --seed 3407
      --n-gpu-layers 100
      --host 0.0.0.0
      --port 8080
```

---

# With `-fmoe` (no other changes):


```
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size =   938.98 MiB
llm_load_tensors:      CUDA0 buffer size = 19112.52 MiB
llm_load_tensors:      CUDA1 buffer size = 20418.15 MiB
llm_load_tensors:      CUDA2 buffer size = 20423.15 MiB
llm_load_tensors:      CUDA3 buffer size = 20423.15 MiB
llm_load_tensors:      CUDA4 buffer size = 20418.15 MiB
llm_load_tensors:      CUDA5 buffer size = 20423.15 MiB
llm_load_tensors:      CUDA6 buffer size = 20423.15 MiB
llm_load_tensors:      CUDA7 buffer size = 20250.86 MiB
llm_load_tensors:      CUDA8 buffer size = 20423.15 MiB
llm_load_tensors:      CUDA9 buffer size = 20423.15 MiB
llm_load_tensors:     CUDA10 buffer size = 20418.15 MiB
llm_load_tensors:     CUDA11 buffer size = 20423.15 MiB
llm_load_tensors:     CUDA12 buffer size = 20423.15 MiB
llm_load_tensors:     CUDA13 buffer size = 20418.15 MiB
llm_load_tensors:     CUDA14 buffer size = 20423.15 MiB
llm_load_tensors:     CUDA15 buffer size = 19014.55 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 8192
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 1024
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: mla_attn   = 2
llama_new_context_with_model: attn_max_b = 64
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
llama_kv_cache_init:      CUDA0 KV buffer size =    36.00 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =    36.00 MiB
llama_kv_cache_init:      CUDA2 KV buffer size =    36.00 MiB
llama_kv_cache_init:      CUDA3 KV buffer size =    36.00 MiB
llama_kv_cache_init:      CUDA4 KV buffer size =    36.00 MiB
llama_kv_cache_init:      CUDA5 KV buffer size =    36.00 MiB
llama_kv_cache_init:      CUDA6 KV buffer size =    36.00 MiB
llama_kv_cache_init:      CUDA7 KV buffer size =    27.00 MiB
llama_kv_cache_init:      CUDA8 KV buffer size =    36.00 MiB
llama_kv_cache_init:      CUDA9 KV buffer size =    36.00 MiB
llama_kv_cache_init:     CUDA10 KV buffer size =    36.00 MiB
llama_kv_cache_init:     CUDA11 KV buffer size =    36.00 MiB
llama_kv_cache_init:     CUDA12 KV buffer size =    36.00 MiB
llama_kv_cache_init:     CUDA13 KV buffer size =    36.00 MiB
llama_kv_cache_init:     CUDA14 KV buffer size =    36.00 MiB
llama_kv_cache_init:     CUDA15 KV buffer size =    18.00 MiB
llama_new_context_with_model: KV self size  =  549.00 MiB, c^KV (f16):  549.00 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.99 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=4)
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 9032.01 MiB on device 3: cudaMalloc failed: out of memory
ggml_gallocr_reserve_n: failed to allocate CUDA3 buffer of size 9470747648
llama_new_context_with_model: failed to allocate compute buffers

```

---

# Without `-fmoe`:

```
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size =   938.98 MiB
llm_load_tensors:      CUDA0 buffer size = 19112.52 MiB
llm_load_tensors:      CUDA1 buffer size = 20418.15 MiB
llm_load_tensors:      CUDA2 buffer size = 20423.15 MiB
llm_load_tensors:      CUDA3 buffer size = 20423.15 MiB
llm_load_tensors:      CUDA4 buffer size = 20418.15 MiB
llm_load_tensors:      CUDA5 buffer size = 20423.15 MiB
llm_load_tensors:      CUDA6 buffer size = 20423.15 MiB
llm_load_tensors:      CUDA7 buffer size = 20250.86 MiB
llm_load_tensors:      CUDA8 buffer size = 20423.15 MiB
llm_load_tensors:      CUDA9 buffer size = 20423.15 MiB
llm_load_tensors:     CUDA10 buffer size = 20418.15 MiB
llm_load_tensors:     CUDA11 buffer size = 20423.15 MiB
llm_load_tensors:     CUDA12 buffer size = 20423.15 MiB
llm_load_tensors:     CUDA13 buffer size = 20418.15 MiB
llm_load_tensors:     CUDA14 buffer size = 20423.15 MiB
llm_load_tensors:     CUDA15 buffer size = 19014.55 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 8192
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 1024
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: mla_attn   = 2
llama_new_context_with_model: attn_max_b = 64
llama_new_context_with_model: fused_moe  = 0
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
llama_kv_cache_init:      CUDA0 KV buffer size =    36.00 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =    36.00 MiB
llama_kv_cache_init:      CUDA2 KV buffer size =    36.00 MiB
llama_kv_cache_init:      CUDA3 KV buffer size =    36.00 MiB
llama_kv_cache_init:      CUDA4 KV buffer size =    36.00 MiB
llama_kv_cache_init:      CUDA5 KV buffer size =    36.00 MiB
llama_kv_cache_init:      CUDA6 KV buffer size =    36.00 MiB
llama_kv_cache_init:      CUDA7 KV buffer size =    27.00 MiB
llama_kv_cache_init:      CUDA8 KV buffer size =    36.00 MiB
llama_kv_cache_init:      CUDA9 KV buffer size =    36.00 MiB
llama_kv_cache_init:     CUDA10 KV buffer size =    36.00 MiB
llama_kv_cache_init:     CUDA11 KV buffer size =    36.00 MiB
llama_kv_cache_init:     CUDA12 KV buffer size =    36.00 MiB
llama_kv_cache_init:     CUDA13 KV buffer size =    36.00 MiB
llama_kv_cache_init:     CUDA14 KV buffer size =    36.00 MiB
llama_kv_cache_init:     CUDA15 KV buffer size =    18.00 MiB
llama_new_context_with_model: KV self size  =  549.00 MiB, c^KV (f16):  549.00 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.99 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=4)
llama_new_context_with_model:      CUDA0 compute buffer size =  1728.01 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =  1800.01 MiB
llama_new_context_with_model:      CUDA2 compute buffer size =  1968.01 MiB
llama_new_context_with_model:      CUDA3 compute buffer size =  2112.01 MiB
llama_new_context_with_model:      CUDA4 compute buffer size =  1592.01 MiB
llama_new_context_with_model:      CUDA5 compute buffer size =  1736.01 MiB
llama_new_context_with_model:      CUDA6 compute buffer size =  1880.01 MiB
llama_new_context_with_model:      CUDA7 compute buffer size =  1480.01 MiB
llama_new_context_with_model:      CUDA8 compute buffer size =  1736.01 MiB
llama_new_context_with_model:      CUDA9 compute buffer size =  1880.01 MiB
llama_new_context_with_model:     CUDA10 compute buffer size =  1360.02 MiB
llama_new_context_with_model:     CUDA11 compute buffer size =  1504.02 MiB
llama_new_context_with_model:     CUDA12 compute buffer size =  1876.01 MiB
llama_new_context_with_model:     CUDA13 compute buffer size =  1476.01 MiB
llama_new_context_with_model:     CUDA14 compute buffer size =  1835.01 MiB
llama_new_context_with_model:     CUDA15 compute buffer size =  1740.02 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   156.05 MiB
llama_new_context_with_model: graph nodes  = 23000
llama_new_context_with_model: graph splits = 63
INFO [                    init] initializing slots | tid="22527965712384" timestamp=1741342650 n_slots=1
INFO [                    init] new slot | tid="22527965712384" timestamp=1741342650 id_slot=0 n_ctx_slot=8192
INFO [                    main] model loaded | tid="22527965712384" timestamp=1741342650
INFO [                    main] chat template | tid="22527965712384" timestamp=1741342650 chat_example="You are a helpful assistant\n\n<｜User｜>Hello<｜Assistant｜>Hi there<｜end▁of▁sentence｜><｜User｜>How are you?<｜Assistant｜>" built_in=true
INFO [                    main] HTTP server listening | tid="22527965712384" timestamp=1741342650 n_threads_http="127" port="8080" hostname="0.0.0.0"
INFO [            update_slots] all slots are idle | tid="22527965712384" timestamp=1741342650
```

To make it easier for you to understand which layers are on which GPUs:

| GPU | Down | Gate | Up | Total Components | Compute Buffer (MiB) | Model Buffer (MiB) | KV Buffer (MiB) |
|--------|------|------|----|-----------------|--------------------|------------------|----------------|
| CUDA0 | 4 | 3 | 3 | 10 | 1728.01 | 19112.52 | 36.00 |
| CUDA1 | 3 | 4 | 4 | 11 | 1800.01 | 20418.15 | 36.00 |
| CUDA2 | 4 | 4 | 3 | 11 | 1968.01 | 20423.15 | 36.00 |
| CUDA3 | 4 | 3 | 4 | 11 | 2112.01 | 20423.15 | 36.00 |
| CUDA4 | 3 | 4 | 4 | 11 | 1592.01 | 20418.15 | 36.00 |
| CUDA5 | 4 | 4 | 3 | 11 | 1736.01 | 20423.15 | 36.00 |
| CUDA6 | 4 | 3 | 4 | 11 | 1880.01 | 20423.15 | 36.00 |
| CUDA7 | 3 | 4 | 4 | 11 | 1480.01 | 20250.86 | 27.00 |
| CUDA8 | 4 | 4 | 3 | 11 | 1736.01 | 20423.15 | 36.00 |
| CUDA9 | 4 | 3 | 4 | 11 | 1880.01 | 20423.15 | 36.00 |
| CUDA10 | 3 | 4 | 4 | 11 | 1360.02 | 20418.15 | 36.00 |
| CUDA11 | 4 | 4 | 3 | 11 | 1504.02 | 20423.15 | 36.00 |
| CUDA12 | 4 | 3 | 4 | 11 | 1876.01 | 20423.15 | 36.00 |
| CUDA13 | 3 | 4 | 4 | 11 | 1476.01 | 20418.15 | 36.00 |
| CUDA14 | 4 | 4 | 3 | 11 | 1835.01 | 20423.15 | 36.00 |
| CUDA15 | 3 | 3 | 4 | 10 | 1740.02 | 19014.55 | 18.00 |
| Total | 58 | 58 | 58 | 174 | 28704.19 | 324059.88 | 549.00 |

---

If you look at the regex, you'll see that `blk.X.` up/gate/down tensors are split across multiple GPUs. This may be a stupidly obvious thing _not_ to do, but to me I don't fully understand LLM architecture so I don't know if I shouldn't do this... 😂.

It also seems that compute buffer is higher than previously for this amount of `-ub`, but I could be just imagining that.

---

👤 **ikawrakow** commented the **2025-03-07** at **14:28:58**:<br>

So, without me having access to a multi-GPU device, I cannot really give a meaningful advice. Still, what about the following split:
* All attention tensors, plus all shared experts, plus the `ffn` tensors of the first 3 layers, plus the output tensor, all on GPU0. E.g.,  `-ot "\.attn_.*\.weight=CUDA0" -ot "\.ffn_.*_shexp\.=CUDA0" -ot blk\.[0-2]\.ffn=CUDA0" -ot "output\.weight=CUDA0"`
* Remaining tensors, which are just the MoE experts, split between the 15 GPUs. There are 58 such layers, so 13 GPUs will get 4 layers with experts and 2 will get 3. E.g. `-ot "blk\.[3-6]\.ffn_.*_exps\.=CUDA1"`, etc.

I count `16,083,517,440` parameters for the weights on GPU0. One wants to spend more for those, say we use `Q6_K`, which is 6.5 bpw. So, GPU0 will have  12.2 GiB full with model weights, and almost 12 GiB left for KV cache and compute buffer. The compute buffer on GPU0 needs to be larger to allow for a longer context.

The MoE experts are 7168 x 2048 x 256, and there are `ffn_up_exps, ffn_gate_exps` and `ffn_down_exps`. The `ffn_down_exps` are more important for preserving model quality than `ffn_up/gate`, so let's spend 4.5 bpw on those (e.g., `IQ4_K`), and 3.5 bpw on `ffn_up/gate` (e.g., `IQ3_K` or `IQ3_S`). This works out to  20.125 GiB for 4 layers. As there is no attention involved for GPU1...15, the compute buffer should be smaller, so almost 4 GiB is plenty. If it is smaller, and you want to max out the VRAM, one can consider using more bits for 1 of the 4 layers (this does improve model quality). There is also the observation that the first few experts layers are more important for model quality than the layers after that, so you may put layers 3,4,5 on GPU1, layers 6,7,8 on GPU2, and use more bits for those experts (e.g, `Q5_K` for `ffn_down` and `IQ4_K` for `ffn_up` and `ffn_gate`, this works out to 19 GiB for 3 layers). The remaining layers are then as discussed on the remaining 13 GPU's, with 4 layers per GPU.

---

👤 **davidsyoung** commented the **2025-03-07** at **16:00:40**:<br>

> So, without me having access to a multi-GPU device, I cannot really give a meaningful advice. Still, what about the following split:
> 
> * All attention tensors, plus all shared experts, plus the `ffn` tensors of the first 3 layers, plus the output tensor, all on GPU0. E.g.,  `-ot "\.attn_.*\.weight=CUDA0" -ot "\.ffn_.*_shexp\.=CUDA0" -ot blk\.[0-2]\.ffn=CUDA0" -ot "output\.weight=CUDA0"`
> * Remaining tensors, which are just the MoE experts, split between the 15 GPUs. There are 58 such layers, so 13 GPUs will get 4 layers with experts and 2 will get 3. E.g. `-ot "blk\.[3-6]\.ffn_.*_exps\.=CUDA1"`, etc.
> 
> I count `16,083,517,440` parameters for the weights on GPU0. One wants to spend more for those, say we use `Q6_K`, which is 6.5 bpw. So, GPU0 will have 12.2 GiB full with model weights, and almost 12 GiB left for KV cache and compute buffer. The compute buffer on GPU0 needs to be larger to allow for a longer context.
> 
> The MoE experts are 7168 x 2048 x 256, and there are `ffn_up_exps, ffn_gate_exps` and `ffn_down_exps`. The `ffn_down_exps` are more important for preserving model quality than `ffn_up/gate`, so let's spend 4.5 bpw on those (e.g., `IQ4_K`), and 3.5 bpw on `ffn_up/gate` (e.g., `IQ3_K` or `IQ3_S`). This works out to 20.125 GiB for 4 layers. As there is no attention involved for GPU1...15, the compute buffer should be smaller, so almost 4 GiB is plenty. If it is smaller, and you want to max out the VRAM, one can consider using more bits for 1 of the 4 layers (this does improve model quality). There is also the observation that the first few experts layers are more important for model quality than the layers after that, so you may put layers 3,4,5 on GPU1, layers 6,7,8 on GPU2, and use more bits for those experts (e.g, `Q5_K` for `ffn_down` and `IQ4_K` for `ffn_up` and `ffn_gate`, this works out to 19 GiB for 3 layers). The remaining layers are then as discussed on the remaining 13 GPU's, with 4 layers per GPU.

This is really helpful! I am going to try to find a way to get PPL working, and then look at quanting this config above :)

---

👤 **ikawrakow** commented the **2025-03-08** at **14:59:10**:<br>

Oops. Yes, of course. So this approach is limited to contexts of up to 8k or 16k tokens. OK, I'll try to think of something else.

---

👤 **davidsyoung** commented the **2025-03-08** at **16:06:11**:<br>

> Oops. Yes, of course. So this approach is limited to contexts of up to 8k or 16k tokens. OK, I'll try to think of something else.

Honestly, keep working away on that MLA FA ;) that'll be a better use of your time.

This quant came in a bit lower on perplexity too, `3.1464 +/- 0.01620` on `IQ4_KSS` vs  `3.0848 +/- 0.01608` on this blend you suggested above. I'm assuming I'm looking at the right figure to compare, right ("Final estimate")? Instead of adding together all numbers and summing them or anything like that.

---

👤 **ikawrakow** commented the **2025-03-08** at **16:22:29**:<br>

Yes, "Final estimate" is the thing to look at. This is about a 2% reduction in PPL. I don't know what the `f16` PPL is for DeepSeekR1, but for the models I can play with `IQ4_KSS` will typically have in the range of 2-3% higher PPL than the `fp16` model. If this is the case also for DeepSeekR1, then 2% is a very significant reduction and would make the quantization almost lossless.

---

👤 **saood06** commented the **2025-03-08** at **22:19:39**:<br>

> This quant came in a bit lower on perplexity too, `3.1464 +/- 0.01620` on `IQ4_KSS` vs `3.0848 +/- 0.01608` on this blend you suggested above. I'm assuming I'm looking at the right figure to compare, right ("Final estimate")? Instead of adding together all numbers and summing them or anything like that.

Can you post the exact code/command/quant log for that blend you use, the PPL looks really good. I only have one other data point for the full run of ppl: `Q5_K_XL : 479.64 GiB (6.13 BPW) | PPL = 3.3499 +/- 0.01849`. This was done on llama.cpp by jukofyork. It is interesting that both your results are considerably lower than this, especially considering that your using less BPW.

@jukofyork I think you might be interested in this, as this does provide even more evidence that llama.cpp was causing quality issues. The full perplexity number I'm referencing is old so maybe you already have addressed the issue as I know you've been working on it.

---

👤 **davidsyoung** commented the **2025-03-08** at **23:32:11**:<br>

@ikawrakow 
> Yes, "Final estimate" is the thing to look at. This is about a 2% reduction in PPL. I don't know what the f16 PPL is for DeepSeekR1, but for the models I can play with IQ4_KSS will typically have in the range of 2-3% higher PPL than the fp16 model. If this is the case also for DeepSeekR1, then 2% is a very significant reduction and would make the quantization almost lossless.

This is awesome, thank you. Really good to know.

@saood06 Of course:

```
./llama-quantize --imatrix /models/deepseek-config/imatrix.dat \
  --token-embedding-type q8_0 \
  --attn-q-type q6_K \
  --attn-k-type q6_K \
  --attn-v-type q6_K \
  --attn-qkv-type q6_K \
  --attn-output-type q6_K \
  --ffn-gate-type q6_K \
  --ffn-down-type q6_K \
  --ffn-up-type q6_K \
  --custom-q "\.ffn_.*_shexp\.weight=q6_K,output\.weight=q6_K" \
  --custom-q "blk\.3\.ffn_down_exps\.weight=q5_K,blk\.4\.ffn_down_exps\.weight=q5_K,blk\.5\.ffn_down_exps\.weight=q5_K,blk\.3\.ffn_up_exps\.weight=iq4_k,blk\.3\.ffn_gate_exps\.weight=iq4_k,blk\.4\.ffn_up_exps\.weight=iq4_k,blk\.4\.ffn_gate_exps\.weight=iq4_k,blk\.5\.ffn_up_exps\.weight=iq4_k,blk\.5\.ffn_gate_exps\.weight=iq4_k" \
  --custom-q "blk\.6\.ffn_down_exps\.weight=q5_K,blk\.7\.ffn_down_exps\.weight=q5_K,blk\.8\.ffn_down_exps\.weight=q5_K,blk\.6\.ffn_up_exps\.weight=iq4_k,blk\.6\.ffn_gate_exps\.weight=iq4_k,blk\.7\.ffn_up_exps\.weight=iq4_k,blk\.7\.ffn_gate_exps\.weight=iq4_k,blk\.8\.ffn_up_exps\.weight=iq4_k,blk\.8\.ffn_gate_exps\.weight=iq4_k" \
  --custom-q "blk\.9\.ffn_down_exps\.weight=iq4_k,blk\.10\.ffn_down_exps\.weight=iq4_k,blk\.11\.ffn_down_exps\.weight=iq4_k,blk\.12\.ffn_down_exps\.weight=iq4_k,blk\.9\.ffn_up_exps\.weight=iq3_k,blk\.9\.ffn_gate_exps\.weight=iq3_k,blk\.10\.ffn_up_exps\.weight=iq3_k,blk\.10\.ffn_gate_exps\.weight=iq3_k,blk\.11\.ffn_up_exps\.weight=iq3_k,blk\.11\.ffn_gate_exps\.weight=iq3_k,blk\.12\.ffn_up_exps\.weight=iq3_k,blk\.12\.ffn_gate_exps\.weight=iq3_k" \
  --custom-q "blk\.13\.ffn_down_exps\.weight=iq4_k,blk\.14\.ffn_down_exps\.weight=iq4_k,blk\.15\.ffn_down_exps\.weight=iq4_k,blk\.16\.ffn_down_exps\.weight=iq4_k,blk\.13\.ffn_up_exps\.weight=iq3_k,blk\.13\.ffn_gate_exps\.weight=iq3_k,blk\.14\.ffn_up_exps\.weight=iq3_k,blk\.14\.ffn_gate_exps\.weight=iq3_k,blk\.15\.ffn_up_exps\.weight=iq3_k,blk\.15\.ffn_gate_exps\.weight=iq3_k,blk\.16\.ffn_up_exps\.weight=iq3_k,blk\.16\.ffn_gate_exps\.weight=iq3_k" \
  --custom-q "blk\.17\.ffn_down_exps\.weight=iq4_k,blk\.18\.ffn_down_exps\.weight=iq4_k,blk\.19\.ffn_down_exps\.weight=iq4_k,blk\.20\.ffn_down_exps\.weight=iq4_k,blk\.17\.ffn_up_exps\.weight=iq3_k,blk\.17\.ffn_gate_exps\.weight=iq3_k,blk\.18\.ffn_up_exps\.weight=iq3_k,blk\.18\.ffn_gate_exps\.weight=iq3_k,blk\.19\.ffn_up_exps\.weight=iq3_k,blk\.19\.ffn_gate_exps\.weight=iq3_k,blk\.20\.ffn_up_exps\.weight=iq3_k,blk\.20\.ffn_gate_exps\.weight=iq3_k" \
  --custom-q "blk\.21\.ffn_down_exps\.weight=iq4_k,blk\.22\.ffn_down_exps\.weight=iq4_k,blk\.23\.ffn_down_exps\.weight=iq4_k,blk\.24\.ffn_down_exps\.weight=iq4_k,blk\.21\.ffn_up_exps\.weight=iq3_k,blk\.21\.ffn_gate_exps\.weight=iq3_k,blk\.22\.ffn_up_exps\.weight=iq3_k,blk\.22\.ffn_gate_exps\.weight=iq3_k,blk\.23\.ffn_up_exps\.weight=iq3_k,blk\.23\.ffn_gate_exps\.weight=iq3_k,blk\.24\.ffn_up_exps\.weight=iq3_k,blk\.24\.ffn_gate_exps\.weight=iq3_k" \
  --custom-q "blk\.25\.ffn_down_exps\.weight=iq4_k,blk\.26\.ffn_down_exps\.weight=iq4_k,blk\.27\.ffn_down_exps\.weight=iq4_k,blk\.28\.ffn_down_exps\.weight=iq4_k,blk\.25\.ffn_up_exps\.weight=iq3_k,blk\.25\.ffn_gate_exps\.weight=iq3_k,blk\.26\.ffn_up_exps\.weight=iq3_k,blk\.26\.ffn_gate_exps\.weight=iq3_k,blk\.27\.ffn_up_exps\.weight=iq3_k,blk\.27\.ffn_gate_exps\.weight=iq3_k,blk\.28\.ffn_up_exps\.weight=iq3_k,blk\.28\.ffn_gate_exps\.weight=iq3_k" \
  --custom-q "blk\.29\.ffn_down_exps\.weight=iq4_k,blk\.30\.ffn_down_exps\.weight=iq4_k,blk\.31\.ffn_down_exps\.weight=iq4_k,blk\.32\.ffn_down_exps\.weight=iq4_k,blk\.29\.ffn_up_exps\.weight=iq3_k,blk\.29\.ffn_gate_exps\.weight=iq3_k,blk\.30\.ffn_up_exps\.weight=iq3_k,blk\.30\.ffn_gate_exps\.weight=iq3_k,blk\.31\.ffn_up_exps\.weight=iq3_k,blk\.31\.ffn_gate_exps\.weight=iq3_k,blk\.32\.ffn_up_exps\.weight=iq3_k,blk\.32\.ffn_gate_exps\.weight=iq3_k" \
  --custom-q "blk\.33\.ffn_down_exps\.weight=iq4_k,blk\.34\.ffn_down_exps\.weight=iq4_k,blk\.35\.ffn_down_exps\.weight=iq4_k,blk\.36\.ffn_down_exps\.weight=iq4_k,blk\.33\.ffn_up_exps\.weight=iq3_k,blk\.33\.ffn_gate_exps\.weight=iq3_k,blk\.34\.ffn_up_exps\.weight=iq3_k,blk\.34\.ffn_gate_exps\.weight=iq3_k,blk\.35\.ffn_up_exps\.weight=iq3_k,blk\.35\.ffn_gate_exps\.weight=iq3_k,blk\.36\.ffn_up_exps\.weight=iq3_k,blk\.36\.ffn_gate_exps\.weight=iq3_k" \
  --custom-q "blk\.37\.ffn_down_exps\.weight=iq4_k,blk\.38\.ffn_down_exps\.weight=iq4_k,blk\.39\.ffn_down_exps\.weight=iq4_k,blk\.40\.ffn_down_exps\.weight=iq4_k,blk\.37\.ffn_up_exps\.weight=iq3_k,blk\.37\.ffn_gate_exps\.weight=iq3_k,blk\.38\.ffn_up_exps\.weight=iq3_k,blk\.38\.ffn_gate_exps\.weight=iq3_k,blk\.39\.ffn_up_exps\.weight=iq3_k,blk\.39\.ffn_gate_exps\.weight=iq3_k,blk\.40\.ffn_up_exps\.weight=iq3_k,blk\.40\.ffn_gate_exps\.weight=iq3_k" \
  --custom-q "blk\.41\.ffn_down_exps\.weight=iq4_k,blk\.42\.ffn_down_exps\.weight=iq4_k,blk\.43\.ffn_down_exps\.weight=iq4_k,blk\.44\.ffn_down_exps\.weight=iq4_k,blk\.41\.ffn_up_exps\.weight=iq3_k,blk\.41\.ffn_gate_exps\.weight=iq3_k,blk\.42\.ffn_up_exps\.weight=iq3_k,blk\.42\.ffn_gate_exps\.weight=iq3_k,blk\.43\.ffn_up_exps\.weight=iq3_k,blk\.43\.ffn_gate_exps\.weight=iq3_k,blk\.44\.ffn_up_exps\.weight=iq3_k,blk\.44\.ffn_gate_exps\.weight=iq3_k" \
  --custom-q "blk\.45\.ffn_down_exps\.weight=iq4_k,blk\.46\.ffn_down_exps\.weight=iq4_k,blk\.47\.ffn_down_exps\.weight=iq4_k,blk\.48\.ffn_down_exps\.weight=iq4_k,blk\.45\.ffn_up_exps\.weight=iq3_k,blk\.45\.ffn_gate_exps\.weight=iq3_k,blk\.46\.ffn_up_exps\.weight=iq3_k,blk\.46\.ffn_gate_exps\.weight=iq3_k,blk\.47\.ffn_up_exps\.weight=iq3_k,blk\.47\.ffn_gate_exps\.weight=iq3_k,blk\.48\.ffn_up_exps\.weight=iq3_k,blk\.48\.ffn_gate_exps\.weight=iq3_k" \
  --custom-q "blk\.49\.ffn_down_exps\.weight=iq4_k,blk\.50\.ffn_down_exps\.weight=iq4_k,blk\.51\.ffn_down_exps\.weight=iq4_k,blk\.52\.ffn_down_exps\.weight=iq4_k,blk\.49\.ffn_up_exps\.weight=iq3_k,blk\.49\.ffn_gate_exps\.weight=iq3_k,blk\.50\.ffn_up_exps\.weight=iq3_k,blk\.50\.ffn_gate_exps\.weight=iq3_k,blk\.51\.ffn_up_exps\.weight=iq3_k,blk\.51\.ffn_gate_exps\.weight=iq3_k,blk\.52\.ffn_up_exps\.weight=iq3_k,blk\.52\.ffn_gate_exps\.weight=iq3_k" \
  --custom-q "blk\.53\.ffn_down_exps\.weight=iq4_k,blk\.54\.ffn_down_exps\.weight=iq4_k,blk\.55\.ffn_down_exps\.weight=iq4_k,blk\.56\.ffn_down_exps\.weight=iq4_k,blk\.53\.ffn_up_exps\.weight=iq3_k,blk\.53\.ffn_gate_exps\.weight=iq3_k,blk\.54\.ffn_up_exps\.weight=iq3_k,blk\.54\.ffn_gate_exps\.weight=iq3_k,blk\.55\.ffn_up_exps\.weight=iq3_k,blk\.55\.ffn_gate_exps\.weight=iq3_k,blk\.56\.ffn_up_exps\.weight=iq3_k,blk\.56\.ffn_gate_exps\.weight=iq3_k" \
  --custom-q "blk\.57\.ffn_down_exps\.weight=iq4_k,blk\.58\.ffn_down_exps\.weight=iq4_k,blk\.59\.ffn_down_exps\.weight=iq4_k,blk\.60\.ffn_down_exps\.weight=iq4_k,blk\.57\.ffn_up_exps\.weight=iq3_k,blk\.57\.ffn_gate_exps\.weight=iq3_k,blk\.58\.ffn_up_exps\.weight=iq3_k,blk\.58\.ffn_gate_exps\.weight=iq3_k,blk\.59\.ffn_up_exps\.weight=iq3_k,blk\.59\.ffn_gate_exps\.weight=iq3_k,blk\.60\.ffn_up_exps\.weight=iq3_k,blk\.60\.ffn_gate_exps\.weight=iq3_k" \
  /storage/DeepSeek-R1-GGUF/unsloth_DeepSeek-R1-BF16-256x21B-F16-00001-of-00059.gguf \
  /models/DeepSeek-R1-GGUF/DeepSeek-R1-GGUF-IQ4_K__IQ3_K.gguf \
  q6_K 64
```

This is using the latest pull request that added custom quant rules: https://github.com/ikawrakow/ik_llama.cpp/pull/244.


Quant log:

```
Adding custom rule \.ffn_.*_shexp\.weight -> q6_K
Adding custom rule output\.weight -> q6_K
Adding custom rule blk\.3\.ffn_down_exps\.weight -> q5_K
Adding custom rule blk\.4\.ffn_down_exps\.weight -> q5_K
Adding custom rule blk\.5\.ffn_down_exps\.weight -> q5_K
Adding custom rule blk\.3\.ffn_up_exps\.weight -> iq4_k
Adding custom rule blk\.3\.ffn_gate_exps\.weight -> iq4_k
Adding custom rule blk\.4\.ffn_up_exps\.weight -> iq4_k
Adding custom rule blk\.4\.ffn_gate_exps\.weight -> iq4_k
Adding custom rule blk\.5\.ffn_up_exps\.weight -> iq4_k
Adding custom rule blk\.5\.ffn_gate_exps\.weight -> iq4_k
Adding custom rule blk\.6\.ffn_down_exps\.weight -> q5_K
Adding custom rule blk\.7\.ffn_down_exps\.weight -> q5_K
Adding custom rule blk\.8\.ffn_down_exps\.weight -> q5_K
Adding custom rule blk\.6\.ffn_up_exps\.weight -> iq4_k
Adding custom rule blk\.6\.ffn_gate_exps\.weight -> iq4_k
Adding custom rule blk\.7\.ffn_up_exps\.weight -> iq4_k
Adding custom rule blk\.7\.ffn_gate_exps\.weight -> iq4_k
Adding custom rule blk\.8\.ffn_up_exps\.weight -> iq4_k
Adding custom rule blk\.8\.ffn_gate_exps\.weight -> iq4_k
Adding custom rule blk\.9\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.10\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.11\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.12\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.9\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.9\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.10\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.10\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.11\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.11\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.12\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.12\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.13\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.14\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.15\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.16\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.13\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.13\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.14\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.14\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.15\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.15\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.16\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.16\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.17\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.18\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.19\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.20\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.17\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.17\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.18\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.18\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.19\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.19\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.20\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.20\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.21\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.22\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.23\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.24\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.21\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.21\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.22\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.22\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.23\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.23\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.24\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.24\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.25\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.26\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.27\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.28\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.25\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.25\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.26\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.26\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.27\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.27\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.28\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.28\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.29\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.30\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.31\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.32\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.29\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.29\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.30\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.30\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.31\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.31\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.32\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.32\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.33\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.34\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.35\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.36\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.33\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.33\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.34\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.34\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.35\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.35\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.36\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.36\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.37\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.38\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.39\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.40\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.37\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.37\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.38\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.38\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.39\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.39\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.40\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.40\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.41\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.42\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.43\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.44\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.41\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.41\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.42\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.42\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.43\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.43\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.44\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.44\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.45\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.46\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.47\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.48\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.45\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.45\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.46\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.46\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.47\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.47\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.48\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.48\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.49\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.50\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.51\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.52\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.49\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.49\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.50\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.50\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.51\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.51\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.52\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.52\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.53\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.54\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.55\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.56\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.53\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.53\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.54\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.54\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.55\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.55\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.56\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.56\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.57\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.58\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.59\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.60\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.57\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.57\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.58\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.58\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.59\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.59\.ffn_gate_exps\.weight -> iq3_k
Adding custom rule blk\.60\.ffn_up_exps\.weight -> iq3_k
Adding custom rule blk\.60\.ffn_gate_exps\.weight -> iq3_k
load_imatrix: imatrix dataset='imatrix-training-full-3'
load_imatrix: loaded 720 importance matrix entries from /models/deepseek-config/imatrix.dat computed on 315 chunks
prepare_imatrix: have 720 importance matrix entries
main: build = 0 (unknown)
main: built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu
main: quantizing '/storage/DeepSeek-R1-GGUF/unsloth_DeepSeek-R1-BF16-256x21B-F16-00001-of-00059.gguf' to '/models/DeepSeek-R1-GGUF/DeepSeek-R1-GGUF-IQ4_K__IQ3_K.gguf' as Q6_K using 64 threads
llama_model_loader: additional 58 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 53 key-value pairs and 1147 tensors from /storage/DeepSeek-R1-GGUF/unsloth_DeepSeek-R1-BF16-256x21B-F16-00001-of-00059.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = unsloth_DeepSeek R1 BF16
llama_model_loader: - kv   3:                         general.size_label str              = 256x21B
llama_model_loader: - kv   4:                            general.license str              = mit
llama_model_loader: - kv   5:                   general.base_model.count u32              = 1
llama_model_loader: - kv   6:                  general.base_model.0.name str              = DeepSeek R1
llama_model_loader: - kv   7:          general.base_model.0.organization str              = Deepseek Ai
llama_model_loader: - kv   8:              general.base_model.0.repo_url str              = https://huggingface.co/deepseek-ai/De...
llama_model_loader: - kv   9:                               general.tags arr[str,3]       = ["deepseek", "unsloth", "transformers"]
llama_model_loader: - kv  10:                          general.languages arr[str,1]       = ["en"]
llama_model_loader: - kv  11:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv  12:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv  13:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv  14:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  15:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv  16:          deepseek2.attention.head_count_kv u32              = 128
llama_model_loader: - kv  17:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  18: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  19:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  20:                          general.file_type u32              = 1
llama_model_loader: - kv  21:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  22:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  23:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  24:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  25:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  26:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  27:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  28:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  29:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  30:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  31:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  32:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  33:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  34:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  35:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  36: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  37: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  38:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  39:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  40:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<...
llama_model_loader: - kv  41:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  42:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
llama_model_loader: - kv  43:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  44:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  45:            tokenizer.ggml.padding_token_id u32              = 128815
llama_model_loader: - kv  46:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  47:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  48:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  49:               general.quantization_version u32              = 2
llama_model_loader: - kv  50:                                   split.no u16              = 0
llama_model_loader: - kv  51:                                split.count u16              = 59
llama_model_loader: - kv  52:                        split.tensors.count i32              = 1147
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type  f16:  786 tensors
================================ Have weights data with 720 entries
[   1/1147]                    token_embd.weight - [ 7168, 129280,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for token_embd.weight
converting to q8_0 .. size =  1767.50 MiB ->   938.98 MiB
[   2/1147]               blk.0.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[   3/1147]                blk.0.ffn_down.weight - [18432,  7168,     1,     1], type =    f16, converting to q6_K .. size =   252.00 MiB ->   103.36 MiB
[   4/1147]                blk.0.ffn_gate.weight - [ 7168, 18432,     1,     1], type =    f16, converting to q6_K .. size =   252.00 MiB ->   103.36 MiB
[   5/1147]                  blk.0.ffn_up.weight - [ 7168, 18432,     1,     1], type =    f16, converting to q6_K .. size =   252.00 MiB ->   103.36 MiB
[   6/1147]                blk.0.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[   7/1147]          blk.0.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[   8/1147]           blk.0.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[   9/1147]               blk.0.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[  10/1147]                blk.0.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.0.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  11/1147]                blk.0.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.0.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[  12/1147]             blk.0.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.0.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[  13/1147]           blk.0.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  14/1147]                blk.0.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[  15/1147]                blk.0.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[  16/1147]               blk.1.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  17/1147]                blk.1.ffn_down.weight - [18432,  7168,     1,     1], type =    f16, converting to q6_K .. size =   252.00 MiB ->   103.36 MiB
[  18/1147]                blk.1.ffn_gate.weight - [ 7168, 18432,     1,     1], type =    f16, converting to q6_K .. size =   252.00 MiB ->   103.36 MiB
[  19/1147]                  blk.1.ffn_up.weight - [ 7168, 18432,     1,     1], type =    f16, converting to q6_K .. size =   252.00 MiB ->   103.36 MiB
[  20/1147]                blk.1.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  21/1147]          blk.1.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  22/1147]           blk.1.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[  23/1147]               blk.1.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[  24/1147]                blk.1.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.1.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  25/1147]                blk.1.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.1.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[  26/1147]             blk.1.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.1.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[  27/1147]           blk.1.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  28/1147]                blk.1.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[  29/1147]                blk.1.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[  30/1147]               blk.2.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  31/1147]                blk.2.ffn_down.weight - [18432,  7168,     1,     1], type =    f16, converting to q6_K .. size =   252.00 MiB ->   103.36 MiB
[  32/1147]                blk.2.ffn_gate.weight - [ 7168, 18432,     1,     1], type =    f16, converting to q6_K .. size =   252.00 MiB ->   103.36 MiB
[  33/1147]                  blk.2.ffn_up.weight - [ 7168, 18432,     1,     1], type =    f16, converting to q6_K .. size =   252.00 MiB ->   103.36 MiB
[  34/1147]                blk.2.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  35/1147]          blk.2.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  36/1147]           blk.2.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[  37/1147]               blk.2.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[  38/1147]                blk.2.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.2.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  39/1147]                blk.2.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.2.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[  40/1147]             blk.2.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.2.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[  41/1147]           blk.2.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  42/1147]                blk.2.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[  43/1147]                blk.2.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[  44/1147]               blk.3.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[  45/1147]            blk.3.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[  46/1147]          blk.3.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.3.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[  47/1147]          blk.3.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.3.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[  48/1147]            blk.3.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.3.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[  49/1147]          blk.3.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  50/1147]           blk.3.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[  51/1147]               blk.3.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[  52/1147]                blk.3.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.3.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  53/1147]                blk.3.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.3.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[  54/1147]             blk.3.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.3.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[  55/1147]           blk.3.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  56/1147]                blk.3.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[  57/1147]                blk.3.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[  58/1147]               blk.3.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  59/1147]           blk.3.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type q5_K for tensor blk.3.ffn_down_exps.weight
converting to q5_K .. size =  7168.00 MiB ->  2464.00 MiB
[  60/1147]           blk.3.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.3.ffn_gate_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[  61/1147]             blk.3.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.3.ffn_up_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[  62/1147]                blk.3.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  63/1147]               blk.4.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[  64/1147]            blk.4.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[  65/1147]          blk.4.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.4.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[  66/1147]          blk.4.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.4.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[  67/1147]            blk.4.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.4.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[  68/1147]          blk.4.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  69/1147]           blk.4.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[  70/1147]               blk.4.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[  71/1147]                blk.4.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.4.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  72/1147]                blk.4.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.4.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[  73/1147]             blk.4.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.4.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[  74/1147]           blk.4.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  75/1147]                blk.4.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[  76/1147]                blk.4.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[  77/1147]               blk.4.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  78/1147]           blk.4.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type q5_K for tensor blk.4.ffn_down_exps.weight
converting to q5_K .. size =  7168.00 MiB ->  2464.00 MiB
[  79/1147]           blk.4.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.4.ffn_gate_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[  80/1147]             blk.4.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.4.ffn_up_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[  81/1147]                blk.4.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  82/1147]          blk.5.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  83/1147]           blk.5.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[  84/1147]               blk.5.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[  85/1147]                blk.5.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.5.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  86/1147]                blk.5.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.5.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[  87/1147]             blk.5.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.5.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[  88/1147]           blk.5.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  89/1147]                blk.5.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[  90/1147]                blk.5.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[  91/1147]               blk.5.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[  92/1147]            blk.5.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[  93/1147]          blk.5.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.5.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[  94/1147]          blk.5.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.5.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[  95/1147]            blk.5.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.5.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[  96/1147]               blk.5.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  97/1147]           blk.5.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type q5_K for tensor blk.5.ffn_down_exps.weight
converting to q5_K .. size =  7168.00 MiB ->  2464.00 MiB
[  98/1147]           blk.5.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.5.ffn_gate_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[  99/1147]             blk.5.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.5.ffn_up_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 100/1147]                blk.5.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 101/1147]               blk.6.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 102/1147]            blk.6.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 103/1147]          blk.6.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.6.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 104/1147]          blk.6.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.6.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 105/1147]            blk.6.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.6.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 106/1147]          blk.6.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 107/1147]           blk.6.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 108/1147]               blk.6.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 109/1147]                blk.6.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.6.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 110/1147]                blk.6.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.6.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 111/1147]             blk.6.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.6.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 112/1147]           blk.6.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 113/1147]                blk.6.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 114/1147]                blk.6.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 115/1147]               blk.6.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 116/1147]           blk.6.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type q5_K for tensor blk.6.ffn_down_exps.weight
converting to q5_K .. size =  7168.00 MiB ->  2464.00 MiB
[ 117/1147]           blk.6.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.6.ffn_gate_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 118/1147]             blk.6.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.6.ffn_up_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 119/1147]                blk.6.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 120/1147]               blk.7.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 121/1147]            blk.7.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 122/1147]          blk.7.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.7.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 123/1147]          blk.7.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.7.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 124/1147]            blk.7.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.7.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 125/1147]          blk.7.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 126/1147]           blk.7.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 127/1147]               blk.7.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 128/1147]                blk.7.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.7.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 129/1147]                blk.7.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.7.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 130/1147]             blk.7.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.7.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 131/1147]           blk.7.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 132/1147]                blk.7.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 133/1147]                blk.7.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 134/1147]               blk.7.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 135/1147]           blk.7.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type q5_K for tensor blk.7.ffn_down_exps.weight
converting to q5_K .. size =  7168.00 MiB ->  2464.00 MiB
[ 136/1147]           blk.7.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.7.ffn_gate_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 137/1147]             blk.7.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.7.ffn_up_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 138/1147]                blk.7.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 139/1147]               blk.8.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 140/1147]            blk.8.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 141/1147]          blk.8.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.8.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 142/1147]          blk.8.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.8.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 143/1147]            blk.8.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.8.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 144/1147]          blk.8.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 145/1147]           blk.8.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 146/1147]               blk.8.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 147/1147]                blk.8.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.8.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 148/1147]                blk.8.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.8.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 149/1147]             blk.8.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.8.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 150/1147]           blk.8.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 151/1147]                blk.8.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 152/1147]                blk.8.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 153/1147]               blk.8.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 154/1147]           blk.8.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type q5_K for tensor blk.8.ffn_down_exps.weight
converting to q5_K .. size =  7168.00 MiB ->  2464.00 MiB
[ 155/1147]           blk.8.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.8.ffn_gate_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 156/1147]             blk.8.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.8.ffn_up_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 157/1147]                blk.8.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 158/1147]               blk.9.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 159/1147]            blk.9.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 160/1147]          blk.9.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.9.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 161/1147]          blk.9.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.9.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 162/1147]            blk.9.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.9.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 163/1147]          blk.9.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 164/1147]           blk.9.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 165/1147]               blk.9.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 166/1147]                blk.9.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.9.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 167/1147]                blk.9.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.9.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 168/1147]             blk.9.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.9.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 169/1147]           blk.9.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 170/1147]                blk.9.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 171/1147]                blk.9.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 172/1147]              blk.10.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 173/1147]           blk.10.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 174/1147]         blk.10.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.10.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 175/1147]         blk.10.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.10.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 176/1147]           blk.10.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.10.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 177/1147]         blk.10.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 178/1147]          blk.10.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 179/1147]              blk.10.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 180/1147]               blk.10.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.10.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 181/1147]               blk.10.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.10.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 182/1147]            blk.10.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.10.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 183/1147]          blk.10.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 184/1147]               blk.10.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 185/1147]               blk.10.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 186/1147]               blk.9.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 187/1147]           blk.9.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.9.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 188/1147]           blk.9.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.9.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 189/1147]             blk.9.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.9.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 190/1147]                blk.9.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 191/1147]              blk.10.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 192/1147]          blk.10.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.10.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 193/1147]          blk.10.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.10.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 194/1147]            blk.10.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.10.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 195/1147]               blk.10.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 196/1147]              blk.11.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 197/1147]           blk.11.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 198/1147]         blk.11.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.11.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 199/1147]         blk.11.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.11.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 200/1147]           blk.11.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.11.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 201/1147]         blk.11.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 202/1147]          blk.11.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 203/1147]              blk.11.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 204/1147]               blk.11.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.11.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 205/1147]               blk.11.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.11.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 206/1147]            blk.11.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.11.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 207/1147]          blk.11.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 208/1147]               blk.11.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 209/1147]               blk.11.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 210/1147]              blk.11.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 211/1147]          blk.11.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.11.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 212/1147]          blk.11.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.11.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 213/1147]            blk.11.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.11.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 214/1147]               blk.11.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 215/1147]              blk.12.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 216/1147]           blk.12.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 217/1147]         blk.12.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.12.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 218/1147]         blk.12.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.12.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 219/1147]           blk.12.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.12.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 220/1147]         blk.12.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 221/1147]          blk.12.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 222/1147]              blk.12.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 223/1147]               blk.12.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.12.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 224/1147]               blk.12.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.12.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 225/1147]            blk.12.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.12.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 226/1147]          blk.12.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 227/1147]               blk.12.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 228/1147]               blk.12.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 229/1147]              blk.12.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 230/1147]          blk.12.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.12.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 231/1147]          blk.12.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.12.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 232/1147]            blk.12.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.12.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 233/1147]               blk.12.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 234/1147]              blk.13.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 235/1147]           blk.13.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 236/1147]         blk.13.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.13.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 237/1147]         blk.13.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.13.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 238/1147]           blk.13.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.13.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 239/1147]         blk.13.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 240/1147]          blk.13.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 241/1147]              blk.13.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 242/1147]               blk.13.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.13.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 243/1147]               blk.13.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.13.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 244/1147]            blk.13.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.13.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 245/1147]          blk.13.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 246/1147]               blk.13.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 247/1147]               blk.13.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 248/1147]              blk.13.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 249/1147]          blk.13.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.13.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 250/1147]          blk.13.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.13.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 251/1147]            blk.13.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.13.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 252/1147]               blk.13.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 253/1147]              blk.14.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 254/1147]           blk.14.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 255/1147]         blk.14.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.14.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 256/1147]         blk.14.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.14.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 257/1147]           blk.14.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.14.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 258/1147]         blk.14.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 259/1147]          blk.14.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 260/1147]              blk.14.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 261/1147]               blk.14.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.14.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 262/1147]               blk.14.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.14.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 263/1147]            blk.14.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.14.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 264/1147]          blk.14.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 265/1147]               blk.14.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 266/1147]               blk.14.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 267/1147]              blk.14.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 268/1147]          blk.14.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.14.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 269/1147]          blk.14.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.14.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 270/1147]            blk.14.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.14.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 271/1147]               blk.14.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 272/1147]              blk.15.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 273/1147]           blk.15.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 274/1147]         blk.15.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.15.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 275/1147]         blk.15.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.15.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 276/1147]           blk.15.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.15.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 277/1147]         blk.15.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 278/1147]          blk.15.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 279/1147]              blk.15.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 280/1147]               blk.15.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.15.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 281/1147]               blk.15.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.15.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 282/1147]            blk.15.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.15.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 283/1147]          blk.15.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 284/1147]               blk.15.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 285/1147]               blk.15.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 286/1147]              blk.15.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 287/1147]          blk.15.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.15.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 288/1147]          blk.15.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.15.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 289/1147]            blk.15.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.15.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 290/1147]               blk.15.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 291/1147]              blk.16.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 292/1147]           blk.16.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 293/1147]         blk.16.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.16.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 294/1147]         blk.16.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.16.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 295/1147]           blk.16.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.16.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 296/1147]         blk.16.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 297/1147]          blk.16.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 298/1147]              blk.16.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 299/1147]               blk.16.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.16.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 300/1147]               blk.16.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.16.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 301/1147]            blk.16.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.16.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 302/1147]          blk.16.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 303/1147]               blk.16.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 304/1147]               blk.16.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 305/1147]              blk.16.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 306/1147]          blk.16.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.16.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 307/1147]          blk.16.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.16.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 308/1147]            blk.16.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.16.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 309/1147]               blk.16.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 310/1147]              blk.17.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 311/1147]           blk.17.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 312/1147]         blk.17.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.17.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 313/1147]         blk.17.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.17.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 314/1147]           blk.17.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.17.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 315/1147]         blk.17.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 316/1147]          blk.17.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 317/1147]              blk.17.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 318/1147]               blk.17.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.17.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 319/1147]               blk.17.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.17.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 320/1147]            blk.17.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.17.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 321/1147]          blk.17.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 322/1147]               blk.17.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 323/1147]               blk.17.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 324/1147]              blk.17.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 325/1147]          blk.17.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.17.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 326/1147]          blk.17.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.17.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 327/1147]            blk.17.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.17.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 328/1147]               blk.17.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 329/1147]              blk.18.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 330/1147]           blk.18.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 331/1147]         blk.18.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.18.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 332/1147]         blk.18.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.18.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 333/1147]           blk.18.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.18.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 334/1147]         blk.18.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 335/1147]          blk.18.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 336/1147]              blk.18.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 337/1147]               blk.18.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.18.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 338/1147]               blk.18.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.18.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 339/1147]            blk.18.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.18.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 340/1147]          blk.18.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 341/1147]               blk.18.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 342/1147]               blk.18.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 343/1147]              blk.18.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 344/1147]          blk.18.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.18.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 345/1147]          blk.18.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.18.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 346/1147]            blk.18.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.18.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 347/1147]               blk.18.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 348/1147]              blk.19.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 349/1147]           blk.19.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 350/1147]         blk.19.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.19.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 351/1147]         blk.19.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.19.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 352/1147]           blk.19.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.19.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 353/1147]         blk.19.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 354/1147]          blk.19.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 355/1147]              blk.19.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 356/1147]               blk.19.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.19.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 357/1147]               blk.19.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.19.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 358/1147]            blk.19.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.19.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 359/1147]          blk.19.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 360/1147]               blk.19.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 361/1147]               blk.19.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 362/1147]              blk.19.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 363/1147]          blk.19.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.19.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 364/1147]          blk.19.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.19.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 365/1147]            blk.19.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.19.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 366/1147]               blk.19.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 367/1147]              blk.20.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 368/1147]           blk.20.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 369/1147]         blk.20.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.20.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 370/1147]         blk.20.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.20.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 371/1147]           blk.20.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.20.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 372/1147]         blk.20.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 373/1147]          blk.20.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 374/1147]              blk.20.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 375/1147]               blk.20.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.20.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 376/1147]               blk.20.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.20.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 377/1147]            blk.20.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.20.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 378/1147]          blk.20.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 379/1147]               blk.20.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 380/1147]               blk.20.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 381/1147]              blk.20.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 382/1147]          blk.20.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.20.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 383/1147]          blk.20.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.20.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 384/1147]            blk.20.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.20.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 385/1147]               blk.20.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 386/1147]              blk.21.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 387/1147]           blk.21.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 388/1147]         blk.21.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.21.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 389/1147]         blk.21.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.21.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 390/1147]           blk.21.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.21.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 391/1147]         blk.21.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 392/1147]          blk.21.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 393/1147]              blk.21.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 394/1147]               blk.21.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.21.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 395/1147]               blk.21.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.21.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 396/1147]            blk.21.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.21.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 397/1147]          blk.21.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 398/1147]               blk.21.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 399/1147]               blk.21.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 400/1147]              blk.21.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 401/1147]          blk.21.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.21.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 402/1147]          blk.21.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.21.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 403/1147]            blk.21.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.21.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 404/1147]               blk.21.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 405/1147]              blk.22.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 406/1147]           blk.22.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 407/1147]         blk.22.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.22.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 408/1147]         blk.22.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.22.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 409/1147]           blk.22.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.22.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 410/1147]         blk.22.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 411/1147]          blk.22.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 412/1147]              blk.22.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 413/1147]               blk.22.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.22.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 414/1147]               blk.22.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.22.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 415/1147]            blk.22.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.22.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 416/1147]          blk.22.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 417/1147]               blk.22.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 418/1147]               blk.22.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 419/1147]              blk.22.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 420/1147]          blk.22.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.22.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 421/1147]          blk.22.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.22.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 422/1147]            blk.22.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.22.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 423/1147]               blk.22.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 424/1147]              blk.23.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 425/1147]           blk.23.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 426/1147]         blk.23.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.23.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 427/1147]         blk.23.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.23.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 428/1147]           blk.23.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.23.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 429/1147]         blk.23.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 430/1147]          blk.23.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 431/1147]              blk.23.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 432/1147]               blk.23.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.23.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 433/1147]               blk.23.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.23.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 434/1147]            blk.23.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.23.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 435/1147]          blk.23.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 436/1147]               blk.23.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 437/1147]               blk.23.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 438/1147]              blk.23.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 439/1147]          blk.23.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.23.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 440/1147]          blk.23.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.23.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 441/1147]            blk.23.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.23.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 442/1147]               blk.23.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 443/1147]              blk.24.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 444/1147]           blk.24.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 445/1147]         blk.24.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.24.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 446/1147]         blk.24.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.24.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 447/1147]           blk.24.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.24.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 448/1147]         blk.24.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 449/1147]          blk.24.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 450/1147]              blk.24.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 451/1147]               blk.24.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.24.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 452/1147]               blk.24.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.24.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 453/1147]            blk.24.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.24.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 454/1147]          blk.24.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 455/1147]               blk.24.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 456/1147]               blk.24.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 457/1147]              blk.24.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 458/1147]          blk.24.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.24.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 459/1147]          blk.24.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.24.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 460/1147]            blk.24.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.24.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 461/1147]               blk.24.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 462/1147]              blk.25.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 463/1147]           blk.25.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 464/1147]         blk.25.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.25.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 465/1147]         blk.25.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.25.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 466/1147]           blk.25.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.25.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 467/1147]         blk.25.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 468/1147]          blk.25.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 469/1147]              blk.25.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 470/1147]               blk.25.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.25.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 471/1147]               blk.25.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.25.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 472/1147]            blk.25.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.25.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 473/1147]          blk.25.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 474/1147]               blk.25.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 475/1147]               blk.25.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 476/1147]              blk.25.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 477/1147]          blk.25.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.25.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 478/1147]          blk.25.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.25.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 479/1147]            blk.25.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.25.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 480/1147]               blk.25.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 481/1147]              blk.26.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 482/1147]           blk.26.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 483/1147]         blk.26.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.26.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 484/1147]         blk.26.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.26.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 485/1147]           blk.26.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.26.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 486/1147]         blk.26.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 487/1147]          blk.26.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 488/1147]              blk.26.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 489/1147]               blk.26.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.26.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 490/1147]               blk.26.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.26.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 491/1147]            blk.26.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.26.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 492/1147]          blk.26.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 493/1147]               blk.26.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 494/1147]               blk.26.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 495/1147]              blk.26.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 496/1147]          blk.26.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.26.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 497/1147]          blk.26.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.26.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 498/1147]            blk.26.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.26.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 499/1147]               blk.26.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 500/1147]              blk.27.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 501/1147]           blk.27.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 502/1147]         blk.27.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.27.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 503/1147]         blk.27.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.27.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 504/1147]           blk.27.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.27.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 505/1147]         blk.27.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 506/1147]          blk.27.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 507/1147]              blk.27.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 508/1147]               blk.27.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.27.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 509/1147]               blk.27.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.27.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 510/1147]            blk.27.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.27.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 511/1147]          blk.27.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 512/1147]               blk.27.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 513/1147]               blk.27.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 514/1147]              blk.27.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 515/1147]          blk.27.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.27.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 516/1147]          blk.27.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.27.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 517/1147]            blk.27.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.27.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 518/1147]               blk.27.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 519/1147]              blk.28.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 520/1147]           blk.28.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 521/1147]         blk.28.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.28.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 522/1147]         blk.28.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.28.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 523/1147]           blk.28.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.28.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 524/1147]         blk.28.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 525/1147]          blk.28.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 526/1147]              blk.28.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 527/1147]               blk.28.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.28.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 528/1147]               blk.28.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.28.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 529/1147]            blk.28.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.28.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 530/1147]          blk.28.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 531/1147]               blk.28.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 532/1147]               blk.28.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 533/1147]              blk.28.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 534/1147]          blk.28.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.28.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 535/1147]          blk.28.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.28.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 536/1147]            blk.28.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.28.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 537/1147]               blk.28.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 538/1147]              blk.29.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 539/1147]           blk.29.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 540/1147]         blk.29.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.29.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 541/1147]         blk.29.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.29.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 542/1147]           blk.29.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.29.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 543/1147]         blk.29.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 544/1147]          blk.29.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 545/1147]              blk.29.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 546/1147]               blk.29.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.29.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 547/1147]               blk.29.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.29.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 548/1147]            blk.29.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.29.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 549/1147]          blk.29.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 550/1147]               blk.29.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 551/1147]               blk.29.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 552/1147]              blk.29.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 553/1147]          blk.29.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.29.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 554/1147]          blk.29.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.29.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 555/1147]            blk.29.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.29.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 556/1147]               blk.29.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 557/1147]              blk.30.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 558/1147]           blk.30.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 559/1147]         blk.30.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.30.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 560/1147]         blk.30.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.30.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 561/1147]           blk.30.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.30.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 562/1147]         blk.30.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 563/1147]          blk.30.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 564/1147]              blk.30.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 565/1147]               blk.30.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.30.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 566/1147]               blk.30.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.30.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 567/1147]            blk.30.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.30.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 568/1147]          blk.30.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 569/1147]               blk.30.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 570/1147]               blk.30.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 571/1147]              blk.30.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 572/1147]          blk.30.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.30.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 573/1147]          blk.30.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.30.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 574/1147]            blk.30.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.30.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 575/1147]               blk.30.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 576/1147]              blk.31.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 577/1147]           blk.31.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 578/1147]         blk.31.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.31.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 579/1147]         blk.31.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.31.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 580/1147]           blk.31.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.31.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 581/1147]         blk.31.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 582/1147]          blk.31.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 583/1147]              blk.31.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 584/1147]               blk.31.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.31.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 585/1147]               blk.31.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.31.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 586/1147]            blk.31.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.31.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 587/1147]          blk.31.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 588/1147]               blk.31.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 589/1147]               blk.31.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 590/1147]              blk.31.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 591/1147]          blk.31.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.31.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 592/1147]          blk.31.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.31.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 593/1147]            blk.31.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.31.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 594/1147]               blk.31.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 595/1147]              blk.32.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 596/1147]           blk.32.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 597/1147]         blk.32.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.32.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 598/1147]         blk.32.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.32.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 599/1147]           blk.32.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.32.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 600/1147]         blk.32.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 601/1147]          blk.32.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 602/1147]              blk.32.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 603/1147]               blk.32.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.32.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 604/1147]               blk.32.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.32.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 605/1147]            blk.32.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.32.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 606/1147]          blk.32.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 607/1147]               blk.32.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 608/1147]               blk.32.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 609/1147]              blk.32.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 610/1147]          blk.32.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.32.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 611/1147]          blk.32.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.32.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 612/1147]            blk.32.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.32.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 613/1147]               blk.32.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 614/1147]              blk.33.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 615/1147]           blk.33.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 616/1147]         blk.33.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.33.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 617/1147]         blk.33.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.33.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 618/1147]           blk.33.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.33.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 619/1147]         blk.33.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 620/1147]          blk.33.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 621/1147]              blk.33.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 622/1147]               blk.33.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.33.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 623/1147]               blk.33.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.33.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 624/1147]            blk.33.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.33.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 625/1147]          blk.33.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 626/1147]               blk.33.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 627/1147]               blk.33.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 628/1147]              blk.33.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 629/1147]          blk.33.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.33.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 630/1147]          blk.33.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.33.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 631/1147]            blk.33.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.33.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 632/1147]               blk.33.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 633/1147]              blk.34.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 634/1147]           blk.34.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 635/1147]         blk.34.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.34.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 636/1147]         blk.34.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.34.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 637/1147]           blk.34.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.34.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 638/1147]         blk.34.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 639/1147]          blk.34.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 640/1147]              blk.34.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 641/1147]               blk.34.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.34.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 642/1147]               blk.34.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.34.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 643/1147]            blk.34.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.34.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 644/1147]          blk.34.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 645/1147]               blk.34.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 646/1147]               blk.34.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 647/1147]              blk.34.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 648/1147]          blk.34.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.34.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 649/1147]          blk.34.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.34.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 650/1147]            blk.34.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.34.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 651/1147]               blk.34.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 652/1147]              blk.35.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 653/1147]           blk.35.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 654/1147]         blk.35.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.35.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 655/1147]         blk.35.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.35.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 656/1147]           blk.35.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.35.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 657/1147]         blk.35.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 658/1147]          blk.35.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 659/1147]              blk.35.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 660/1147]               blk.35.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.35.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 661/1147]               blk.35.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.35.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 662/1147]            blk.35.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.35.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 663/1147]          blk.35.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 664/1147]               blk.35.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 665/1147]               blk.35.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 666/1147]              blk.35.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 667/1147]          blk.35.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.35.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 668/1147]          blk.35.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.35.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 669/1147]            blk.35.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.35.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 670/1147]               blk.35.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 671/1147]              blk.36.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 672/1147]           blk.36.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 673/1147]         blk.36.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.36.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 674/1147]         blk.36.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.36.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 675/1147]           blk.36.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.36.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 676/1147]         blk.36.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 677/1147]          blk.36.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 678/1147]              blk.36.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 679/1147]               blk.36.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.36.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 680/1147]               blk.36.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.36.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 681/1147]            blk.36.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.36.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 682/1147]          blk.36.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 683/1147]               blk.36.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 684/1147]               blk.36.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 685/1147]              blk.36.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 686/1147]          blk.36.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.36.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 687/1147]          blk.36.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.36.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 688/1147]            blk.36.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.36.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 689/1147]               blk.36.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 690/1147]              blk.37.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 691/1147]           blk.37.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 692/1147]         blk.37.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.37.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 693/1147]         blk.37.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.37.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 694/1147]           blk.37.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.37.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 695/1147]         blk.37.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 696/1147]          blk.37.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 697/1147]              blk.37.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 698/1147]               blk.37.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.37.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 699/1147]               blk.37.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.37.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 700/1147]            blk.37.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.37.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 701/1147]          blk.37.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 702/1147]               blk.37.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 703/1147]               blk.37.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 704/1147]              blk.37.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 705/1147]          blk.37.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.37.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 706/1147]          blk.37.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.37.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 707/1147]            blk.37.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.37.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 708/1147]               blk.37.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 709/1147]              blk.38.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 710/1147]           blk.38.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 711/1147]         blk.38.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.38.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 712/1147]         blk.38.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.38.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 713/1147]           blk.38.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.38.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 714/1147]         blk.38.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 715/1147]          blk.38.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 716/1147]              blk.38.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 717/1147]               blk.38.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.38.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 718/1147]               blk.38.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.38.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 719/1147]            blk.38.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.38.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 720/1147]          blk.38.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 721/1147]               blk.38.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 722/1147]               blk.38.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 723/1147]              blk.38.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 724/1147]          blk.38.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.38.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 725/1147]          blk.38.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.38.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 726/1147]            blk.38.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.38.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 727/1147]               blk.38.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 728/1147]              blk.39.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 729/1147]           blk.39.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 730/1147]         blk.39.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.39.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 731/1147]         blk.39.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.39.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 732/1147]           blk.39.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.39.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 733/1147]         blk.39.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 734/1147]          blk.39.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 735/1147]              blk.39.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 736/1147]               blk.39.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.39.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 737/1147]               blk.39.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.39.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 738/1147]            blk.39.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.39.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 739/1147]          blk.39.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 740/1147]               blk.39.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 741/1147]               blk.39.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 742/1147]              blk.39.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 743/1147]          blk.39.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.39.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 744/1147]          blk.39.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.39.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 745/1147]            blk.39.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.39.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 746/1147]               blk.39.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 747/1147]              blk.40.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 748/1147]           blk.40.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 749/1147]         blk.40.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.40.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 750/1147]         blk.40.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.40.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 751/1147]           blk.40.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.40.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 752/1147]         blk.40.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 753/1147]          blk.40.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 754/1147]              blk.40.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 755/1147]               blk.40.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.40.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 756/1147]               blk.40.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.40.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 757/1147]            blk.40.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.40.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 758/1147]          blk.40.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 759/1147]               blk.40.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 760/1147]               blk.40.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 761/1147]              blk.40.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 762/1147]          blk.40.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.40.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 763/1147]          blk.40.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.40.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 764/1147]            blk.40.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.40.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 765/1147]               blk.40.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 766/1147]              blk.41.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 767/1147]           blk.41.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 768/1147]         blk.41.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.41.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 769/1147]         blk.41.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.41.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 770/1147]           blk.41.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.41.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 771/1147]         blk.41.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 772/1147]          blk.41.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 773/1147]              blk.41.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 774/1147]               blk.41.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.41.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 775/1147]               blk.41.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.41.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 776/1147]            blk.41.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.41.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 777/1147]          blk.41.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 778/1147]               blk.41.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 779/1147]               blk.41.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 780/1147]              blk.41.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 781/1147]          blk.41.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.41.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 782/1147]          blk.41.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.41.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 783/1147]            blk.41.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.41.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 784/1147]               blk.41.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 785/1147]              blk.42.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 786/1147]           blk.42.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 787/1147]         blk.42.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.42.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 788/1147]         blk.42.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.42.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 789/1147]           blk.42.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.42.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 790/1147]         blk.42.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 791/1147]          blk.42.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 792/1147]              blk.42.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 793/1147]               blk.42.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.42.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 794/1147]               blk.42.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.42.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 795/1147]            blk.42.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.42.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 796/1147]          blk.42.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 797/1147]               blk.42.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 798/1147]               blk.42.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 799/1147]              blk.42.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 800/1147]          blk.42.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.42.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 801/1147]          blk.42.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.42.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 802/1147]            blk.42.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.42.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 803/1147]               blk.42.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 804/1147]              blk.43.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 805/1147]           blk.43.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 806/1147]         blk.43.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.43.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 807/1147]         blk.43.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.43.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 808/1147]           blk.43.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.43.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 809/1147]         blk.43.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 810/1147]          blk.43.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 811/1147]              blk.43.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 812/1147]               blk.43.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.43.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 813/1147]               blk.43.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.43.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 814/1147]            blk.43.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.43.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 815/1147]          blk.43.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 816/1147]               blk.43.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 817/1147]               blk.43.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 818/1147]              blk.43.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 819/1147]          blk.43.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.43.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 820/1147]          blk.43.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.43.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 821/1147]            blk.43.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.43.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 822/1147]               blk.43.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 823/1147]              blk.44.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 824/1147]           blk.44.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 825/1147]         blk.44.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.44.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 826/1147]         blk.44.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.44.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 827/1147]           blk.44.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.44.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 828/1147]         blk.44.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 829/1147]          blk.44.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 830/1147]              blk.44.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 831/1147]               blk.44.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.44.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 832/1147]               blk.44.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.44.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 833/1147]            blk.44.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.44.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 834/1147]          blk.44.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 835/1147]               blk.44.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 836/1147]               blk.44.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 837/1147]              blk.44.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 838/1147]          blk.44.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.44.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 839/1147]          blk.44.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.44.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 840/1147]            blk.44.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.44.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 841/1147]               blk.44.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 842/1147]              blk.45.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 843/1147]           blk.45.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 844/1147]         blk.45.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.45.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 845/1147]         blk.45.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.45.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 846/1147]           blk.45.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.45.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 847/1147]         blk.45.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 848/1147]          blk.45.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 849/1147]              blk.45.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 850/1147]               blk.45.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.45.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 851/1147]               blk.45.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.45.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 852/1147]            blk.45.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.45.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 853/1147]          blk.45.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 854/1147]               blk.45.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 855/1147]               blk.45.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 856/1147]              blk.45.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 857/1147]          blk.45.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.45.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 858/1147]          blk.45.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.45.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 859/1147]            blk.45.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.45.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 860/1147]               blk.45.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 861/1147]              blk.46.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 862/1147]           blk.46.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 863/1147]         blk.46.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.46.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 864/1147]         blk.46.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.46.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 865/1147]           blk.46.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.46.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 866/1147]         blk.46.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 867/1147]          blk.46.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 868/1147]              blk.46.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 869/1147]               blk.46.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.46.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 870/1147]               blk.46.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.46.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 871/1147]            blk.46.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.46.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 872/1147]          blk.46.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 873/1147]               blk.46.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 874/1147]               blk.46.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 875/1147]              blk.46.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 876/1147]          blk.46.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.46.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 877/1147]          blk.46.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.46.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 878/1147]            blk.46.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.46.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 879/1147]               blk.46.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 880/1147]              blk.47.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 881/1147]           blk.47.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 882/1147]         blk.47.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.47.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 883/1147]         blk.47.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.47.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 884/1147]           blk.47.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.47.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 885/1147]         blk.47.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 886/1147]          blk.47.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 887/1147]              blk.47.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 888/1147]               blk.47.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.47.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 889/1147]               blk.47.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.47.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 890/1147]            blk.47.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.47.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 891/1147]          blk.47.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 892/1147]               blk.47.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 893/1147]               blk.47.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 894/1147]              blk.47.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 895/1147]          blk.47.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.47.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 896/1147]          blk.47.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.47.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 897/1147]            blk.47.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.47.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 898/1147]               blk.47.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 899/1147]              blk.48.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 900/1147]           blk.48.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 901/1147]         blk.48.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.48.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 902/1147]         blk.48.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.48.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 903/1147]           blk.48.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.48.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 904/1147]         blk.48.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 905/1147]          blk.48.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 906/1147]              blk.48.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 907/1147]               blk.48.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.48.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 908/1147]               blk.48.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.48.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 909/1147]            blk.48.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.48.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 910/1147]          blk.48.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 911/1147]               blk.48.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 912/1147]               blk.48.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 913/1147]              blk.48.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 914/1147]          blk.48.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.48.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 915/1147]          blk.48.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.48.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 916/1147]            blk.48.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.48.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 917/1147]               blk.48.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 918/1147]              blk.49.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 919/1147]           blk.49.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 920/1147]         blk.49.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.49.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 921/1147]         blk.49.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.49.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 922/1147]           blk.49.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.49.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 923/1147]         blk.49.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 924/1147]          blk.49.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 925/1147]              blk.49.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 926/1147]               blk.49.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.49.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 927/1147]               blk.49.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.49.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 928/1147]            blk.49.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.49.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 929/1147]          blk.49.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 930/1147]               blk.49.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 931/1147]               blk.49.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 932/1147]              blk.49.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 933/1147]          blk.49.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.49.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 934/1147]          blk.49.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.49.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 935/1147]            blk.49.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.49.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 936/1147]               blk.49.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 937/1147]              blk.50.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 938/1147]           blk.50.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 939/1147]         blk.50.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.50.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 940/1147]         blk.50.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.50.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 941/1147]           blk.50.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.50.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 942/1147]         blk.50.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 943/1147]          blk.50.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 944/1147]              blk.50.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 945/1147]               blk.50.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.50.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 946/1147]               blk.50.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.50.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 947/1147]            blk.50.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.50.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 948/1147]          blk.50.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 949/1147]               blk.50.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 950/1147]               blk.50.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 951/1147]              blk.50.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 952/1147]          blk.50.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.50.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 953/1147]          blk.50.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.50.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 954/1147]            blk.50.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.50.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 955/1147]               blk.50.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 956/1147]              blk.51.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 957/1147]           blk.51.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 958/1147]         blk.51.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.51.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 959/1147]         blk.51.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.51.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 960/1147]           blk.51.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.51.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 961/1147]         blk.51.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 962/1147]          blk.51.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 963/1147]              blk.51.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 964/1147]               blk.51.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.51.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 965/1147]               blk.51.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.51.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 966/1147]            blk.51.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.51.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 967/1147]          blk.51.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 968/1147]               blk.51.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 969/1147]               blk.51.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 970/1147]              blk.51.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 971/1147]          blk.51.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.51.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 972/1147]          blk.51.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.51.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 973/1147]            blk.51.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.51.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 974/1147]               blk.51.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 975/1147]              blk.52.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 976/1147]           blk.52.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 977/1147]         blk.52.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.52.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 978/1147]         blk.52.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.52.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 979/1147]           blk.52.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.52.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 980/1147]         blk.52.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 981/1147]          blk.52.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[ 982/1147]              blk.52.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[ 983/1147]               blk.52.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.52.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 984/1147]               blk.52.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.52.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[ 985/1147]            blk.52.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.52.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[ 986/1147]          blk.52.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 987/1147]               blk.52.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[ 988/1147]               blk.52.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[ 989/1147]              blk.52.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 990/1147]          blk.52.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.52.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 991/1147]          blk.52.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.52.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 992/1147]            blk.52.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.52.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[ 993/1147]               blk.52.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 994/1147]              blk.53.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 995/1147]           blk.53.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 996/1147]         blk.53.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.53.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 997/1147]         blk.53.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.53.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 998/1147]           blk.53.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.53.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[ 999/1147]         blk.53.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1000/1147]          blk.53.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[1001/1147]              blk.53.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[1002/1147]               blk.53.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.53.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1003/1147]               blk.53.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.53.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[1004/1147]            blk.53.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.53.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[1005/1147]          blk.53.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1006/1147]               blk.53.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[1007/1147]               blk.53.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[1008/1147]              blk.53.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1009/1147]          blk.53.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.53.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[1010/1147]          blk.53.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.53.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[1011/1147]            blk.53.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.53.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[1012/1147]               blk.53.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1013/1147]              blk.54.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1014/1147]           blk.54.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1015/1147]         blk.54.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.54.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[1016/1147]         blk.54.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.54.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[1017/1147]           blk.54.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.54.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[1018/1147]         blk.54.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1019/1147]          blk.54.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[1020/1147]              blk.54.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[1021/1147]               blk.54.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.54.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1022/1147]               blk.54.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.54.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[1023/1147]            blk.54.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.54.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[1024/1147]          blk.54.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1025/1147]               blk.54.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[1026/1147]               blk.54.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[1027/1147]              blk.54.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1028/1147]          blk.54.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.54.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[1029/1147]          blk.54.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.54.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[1030/1147]            blk.54.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.54.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[1031/1147]               blk.54.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1032/1147]              blk.55.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1033/1147]           blk.55.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1034/1147]         blk.55.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.55.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[1035/1147]         blk.55.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.55.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[1036/1147]           blk.55.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.55.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[1037/1147]         blk.55.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1038/1147]          blk.55.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[1039/1147]              blk.55.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[1040/1147]               blk.55.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.55.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1041/1147]               blk.55.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.55.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[1042/1147]            blk.55.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.55.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[1043/1147]          blk.55.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1044/1147]               blk.55.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[1045/1147]               blk.55.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[1046/1147]              blk.55.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1047/1147]          blk.55.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.55.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[1048/1147]          blk.55.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.55.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[1049/1147]            blk.55.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.55.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[1050/1147]               blk.55.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1051/1147]              blk.56.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1052/1147]           blk.56.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1053/1147]         blk.56.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.56.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[1054/1147]         blk.56.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.56.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[1055/1147]           blk.56.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.56.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[1056/1147]         blk.56.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1057/1147]          blk.56.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[1058/1147]              blk.56.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[1059/1147]               blk.56.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.56.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1060/1147]               blk.56.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.56.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[1061/1147]            blk.56.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.56.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[1062/1147]          blk.56.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1063/1147]               blk.56.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[1064/1147]               blk.56.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[1065/1147]              blk.56.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1066/1147]          blk.56.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.56.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[1067/1147]          blk.56.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.56.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[1068/1147]            blk.56.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.56.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[1069/1147]               blk.56.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1070/1147]              blk.57.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1071/1147]           blk.57.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1072/1147]         blk.57.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.57.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[1073/1147]         blk.57.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.57.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[1074/1147]           blk.57.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.57.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[1075/1147]         blk.57.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1076/1147]          blk.57.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[1077/1147]              blk.57.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[1078/1147]               blk.57.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.57.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1079/1147]               blk.57.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.57.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[1080/1147]            blk.57.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.57.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[1081/1147]          blk.57.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1082/1147]               blk.57.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[1083/1147]               blk.57.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[1084/1147]              blk.57.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1085/1147]          blk.57.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.57.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[1086/1147]          blk.57.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.57.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[1087/1147]            blk.57.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.57.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[1088/1147]               blk.57.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1089/1147]              blk.58.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1090/1147]           blk.58.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1091/1147]         blk.58.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.58.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[1092/1147]         blk.58.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.58.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[1093/1147]           blk.58.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.58.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[1094/1147]         blk.58.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1095/1147]          blk.58.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[1096/1147]              blk.58.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[1097/1147]               blk.58.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.58.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1098/1147]               blk.58.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.58.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[1099/1147]            blk.58.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.58.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[1100/1147]          blk.58.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1101/1147]               blk.58.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[1102/1147]               blk.58.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[1103/1147]              blk.58.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1104/1147]          blk.58.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.58.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[1105/1147]          blk.58.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.58.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[1106/1147]            blk.58.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.58.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[1107/1147]               blk.58.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1108/1147]              blk.59.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1109/1147]           blk.59.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1110/1147]         blk.59.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.59.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[1111/1147]         blk.59.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.59.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[1112/1147]           blk.59.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.59.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[1113/1147]         blk.59.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1114/1147]          blk.59.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[1115/1147]              blk.59.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[1116/1147]               blk.59.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.59.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1117/1147]               blk.59.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.59.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[1118/1147]            blk.59.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.59.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[1119/1147]          blk.59.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1120/1147]               blk.59.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[1121/1147]               blk.59.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[1122/1147]              blk.59.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1123/1147]          blk.59.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.59.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[1124/1147]          blk.59.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.59.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[1125/1147]            blk.59.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.59.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[1126/1147]               blk.59.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1127/1147]              blk.60.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1128/1147]           blk.60.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1129/1147]         blk.60.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.60.ffn_down_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[1130/1147]         blk.60.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.60.ffn_gate_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[1131/1147]           blk.60.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q6_K for tensor blk.60.ffn_up_shexp.weight
converting to q6_K .. size =    28.00 MiB ->    11.48 MiB
[1132/1147]         blk.60.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1133/1147]          blk.60.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q6_K .. size =     7.88 MiB ->     3.23 MiB
[1134/1147]              blk.60.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q6_K .. size =    32.00 MiB ->    13.12 MiB
[1135/1147]               blk.60.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 

llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q6_K - using fallback quantization q8_0

====== llama_model_quantize_internal: did not find weights for blk.60.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1136/1147]               blk.60.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
====== llama_model_quantize_internal: did not find weights for blk.60.attn_v_b.weight
converting to q6_K .. size =    16.00 MiB ->     6.56 MiB
[1137/1147]            blk.60.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q6_K for tensor blk.60.attn_output.weight
converting to q6_K .. size =   224.00 MiB ->    91.88 MiB
[1138/1147]          blk.60.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1139/1147]               blk.60.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, converting to q6_K .. size =    21.00 MiB ->     8.61 MiB
[1140/1147]               blk.60.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, converting to q6_K .. size =    72.00 MiB ->    29.53 MiB
[1141/1147]                        output.weight - [ 7168, 129280,     1,     1], type =    f16, Using custom type q6_K for tensor output.weight

====== llama_model_quantize_internal: did not find weights for output.weight
converting to q6_K .. size =  1767.50 MiB ->   724.95 MiB
[1142/1147]              blk.60.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1143/1147]          blk.60.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.60.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[1144/1147]          blk.60.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.60.ffn_gate_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[1145/1147]            blk.60.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_k for tensor blk.60.ffn_up_exps.weight
converting to iq3_k .. size =  7168.00 MiB ->  1540.00 MiB
[1146/1147]               blk.60.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1147/1147]                   output_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
llama_model_quantize_internal: model size  = 1282038.27 MB
llama_model_quantize_internal: quant size  = 318818.01 MB
llama_model_quantize_internal: WARNING: 61 of 785 tensor(s) required fallback quantization

main: quantize time = 10582798.69 ms
main:    total time = 10582798.69 ms

```

---

👤 **davidsyoung** commented the **2025-03-08** at **23:32:16**:<br>

PPL run (I'm getting NaN's if `-ub` is set higher than 32, and finding it hard to balance layers across GPUs here, but it ran):
```
root@5d30ef8d3bb7:/app# ./llama-perplexity -m /models/DeepSeek-R1-GGUF/DeepSeek-R1-GGUF-IQ4_K__IQ3_K.gguf -f /models/wiki.test.raw -mla 2 -ts 24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24 -c 2028 -ot "\.attn_.*\.weight|\.ffn_.*_shexp\.|blk\.0\.ffn|blk\.1\.ffn|blk\.2\.ffn|output\.weight=CUDA0" -ot "blk\.3\.ffn_(down|gate|up)_exps\.weight|blk\.4\.ffn_(down|gate|up)_exps\.weight|blk\.5\.ffn_(down|gate|up)_exps\.weight=CUDA1" -ot "blk\.6\.ffn_(down|gate|up)_exps\.weight|blk\.7\.ffn_(down|gate|up)_exps\.weight|blk\.8\.ffn_(down|gate|up)_exps\.weight=CUDA2" -ot "blk\.9\.ffn_(down|gate|up)_exps\.weight|blk\.10\.ffn_(down|gate|up)_exps\.weight|blk\.11\.ffn_(down|gate|up)_exps\.weight|blk\.12\.ffn_(down|gate|up)_exps\.weight=CUDA3" -ot "blk\.13\.ffn_(down|gate|up)_exps\.weight|blk\.14\.ffn_(down|gate|up)_exps\.weight|blk\.15\.ffn_(down|gate|up)_exps\.weight|blk\.16\.ffn_(down|gate|up)_exps\.weight=CUDA4" -ot "blk\.17\.ffn_(down|gate|up)_exps\.weight|blk\.18\.ffn_(down|gate|up)_exps\.weight|blk\.19\.ffn_(down|gate|up)_exps\.weight|blk\.20\.ffn_(down|gate|up)_exps\.weight=CUDA5" -ot "blk\.21\.ffn_(down|gate|up)_exps\.weight|blk\.22\.ffn_(down|gate|up)_exps\.weight|blk\.23\.ffn_(down|gate|up)_exps\.weight|blk\.24\.ffn_(down|gate|up)_exps\.weight=CUDA6" -ot "blk\.25\.ffn_(down|gate|up)_exps\.weight|blk\.26\.ffn_(down|gate|up)_exps\.weight|blk\.27\.ffn_(down|gate|up)_exps\.weight|blk\.28\.ffn_(down|gate|up)_exps\.weight=CUDA7" -ot "blk\.29\.ffn_(down|gate|up)_exps\.weight|blk\.30\.ffn_(down|gate|up)_exps\.weight|blk\.31\.ffn_(down|gate|up)_exps\.weight|blk\.32\.ffn_(down|gate|up)_exps\.weight=CUDA8" -ot "blk\.33\.ffn_(down|gate|up)_exps\.weight|blk\.34\.ffn_(down|gate|up)_exps\.weight|blk\.35\.ffn_(down|gate|up)_exps\.weight|blk\.36\.ffn_(down|gate|up)_exps\.weight=CUDA9" -ot "blk\.37\.ffn_(down|gate|up)_exps\.weight|blk\.38\.ffn_(down|gate|up)_exps\.weight|blk\.39\.ffn_(down|gate|up)_exps\.weight|blk\.40\.ffn_(down|gate|up)_exps\.weight=CUDA10" -ot "blk\.41\.ffn_(down|gate|up)_exps\.weight|blk\.42\.ffn_(down|gate|up)_exps\.weight|blk\.43\.ffn_(down|gate|up)_exps\.weight|blk\.44\.ffn_(down|gate|up)_exps\.weight=CUDA11" -ot "blk\.45\.ffn_(down|gate|up)_exps\.weight|blk\.46\.ffn_(down|gate|up)_exps\.weight|blk\.47\.ffn_(down|gate|up)_exps\.weight|blk\.48\.ffn_(down|gate|up)_exps\.weight=CUDA12" -ot "blk\.49\.ffn_(down|gate|up)_exps\.weight|blk\.50\.ffn_(down|gate|up)_exps\.weight|blk\.51\.ffn_(down|gate|up)_exps\.weight|blk\.52\.ffn_(down|gate|up)_exps\.weight=CUDA13" -ot "blk\.53\.ffn_(down|gate|up)_exps\.weight|blk\.54\.ffn_(down|gate|up)_exps\.weight|blk\.55\.ffn_(down|gate|up)_exps\.weight|blk\.56\.ffn_(down|gate|up)_exps\.weight=CUDA14" -ot "blk\.57\.ffn_(down|gate|up)_exps\.weight|blk\.58\.ffn_(down|gate|up)_exps\.weight|blk\.59\.ffn_(down|gate|up)_exps\.weight|blk\.60\.ffn_(down|gate|up)_exps\.weight=CUDA15" -b 2048 -ub 32 -amb 64 -ngl 100 --seed 3407 --temp 0.5
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    yes
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 16 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 1: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 2: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 3: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 4: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 5: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 6: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 7: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 8: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 9: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 10: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 11: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 12: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 13: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 14: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 15: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
main: build = 0 (unknown)
main: built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu
main: seed  = 3407
llama_model_loader: loaded meta data with 54 key-value pairs and 1147 tensors from /models/DeepSeek-R1-GGUF/DeepSeek-R1-GGUF-IQ4_K__IQ3_K.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = unsloth_DeepSeek R1 BF16
llama_model_loader: - kv   3:                         general.size_label str              = 256x21B
llama_model_loader: - kv   4:                            general.license str              = mit
llama_model_loader: - kv   5:                   general.base_model.count u32              = 1
llama_model_loader: - kv   6:                  general.base_model.0.name str              = DeepSeek R1
llama_model_loader: - kv   7:          general.base_model.0.organization str              = Deepseek Ai
llama_model_loader: - kv   8:              general.base_model.0.repo_url str              = https://huggingface.co/deepseek-ai/De...
llama_model_loader: - kv   9:                               general.tags arr[str,3]       = ["deepseek", "unsloth", "transformers"]
llama_model_loader: - kv  10:                          general.languages arr[str,1]       = ["en"]
llama_model_loader: - kv  11:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv  12:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv  13:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv  14:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  15:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv  16:          deepseek2.attention.head_count_kv u32              = 128
llama_model_loader: - kv  17:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  18: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  19:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  20:                          general.file_type u32              = 18
llama_model_loader: - kv  21:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  22:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  23:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  24:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  25:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  26:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  27:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  28:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  29:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  30:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  31:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  32:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  33:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  34:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  35:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  36: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  37: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  38:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  39:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  40:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<?...
llama_model_loader: - kv  41:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  42:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
llama_model_loader: - kv  43:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  44:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  45:            tokenizer.ggml.padding_token_id u32              = 128815
llama_model_loader: - kv  46:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  47:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  48:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  49:               general.quantization_version u32              = 2
llama_model_loader: - kv  50:                      quantize.imatrix.file str              = /models/deepseek-config/imatrix.dat
llama_model_loader: - kv  51:                   quantize.imatrix.dataset str              = imatrix-training-full-3
llama_model_loader: - kv  52:             quantize.imatrix.entries_count i32              = 720
llama_model_loader: - kv  53:              quantize.imatrix.chunks_count i32              = 315
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:   62 tensors
llama_model_loader: - type q5_K:    6 tensors
llama_model_loader: - type q6_K:  550 tensors
llama_model_loader: - type iq3_k:  104 tensors
llama_model_loader: - type iq4_k:   64 tensors
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
llm_load_print_meta: model ftype      = Q6_K
llm_load_print_meta: model params     = 672.050 B
llm_load_print_meta: model size       = 311.346 GiB (3.980 BPW) 
llm_load_print_meta: repeating layers = 309.721 GiB (3.970 BPW, 670.196 B parameters)
llm_load_print_meta: general.name     = unsloth_DeepSeek R1 BF16
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
llm_load_tensors: ggml ctx size =    7.94 MiB
Tensor output.weight buffer type overriden to CUDA0
Tensor blk.0.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.0.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.0.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.0.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.0.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.0.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.0.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.0.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.0.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.0.attn_output.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_gate.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_down.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_up.weight buffer type overriden to CUDA0
Tensor blk.1.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.1.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.1.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.1.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.1.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.1.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.1.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.1.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.1.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.1.attn_output.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_gate.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_down.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_up.weight buffer type overriden to CUDA0
Tensor blk.2.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.2.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.2.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.2.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.2.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.2.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.2.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.2.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.2.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.2.attn_output.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_gate.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_down.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_up.weight buffer type overriden to CUDA0
Tensor blk.3.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.3.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.3.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.3.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.3.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.3.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.3.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.3.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.3.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.3.attn_output.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.3.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.3.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.3.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.4.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.4.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.4.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.4.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.4.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.4.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.4.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.4.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.4.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.4.attn_output.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.4.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.4.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.4.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.5.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.5.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.5.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.5.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.5.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.5.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.5.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.5.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.5.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.5.attn_output.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.6.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.6.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.6.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.6.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.6.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.6.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.6.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.6.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.6.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.6.attn_output.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_gate_exps.weight buffer type overriden to CUDA2
Tensor blk.6.ffn_down_exps.weight buffer type overriden to CUDA2
Tensor blk.6.ffn_up_exps.weight buffer type overriden to CUDA2
Tensor blk.6.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.7.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.7.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.7.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.7.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.7.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.7.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.7.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.7.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.7.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.7.attn_output.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CUDA2
Tensor blk.7.ffn_down_exps.weight buffer type overriden to CUDA2
Tensor blk.7.ffn_up_exps.weight buffer type overriden to CUDA2
Tensor blk.7.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.8.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.8.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.8.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.8.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.8.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.8.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.8.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.8.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.8.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.8.attn_output.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CUDA2
Tensor blk.8.ffn_down_exps.weight buffer type overriden to CUDA2
Tensor blk.8.ffn_up_exps.weight buffer type overriden to CUDA2
Tensor blk.8.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.9.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.9.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.9.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.9.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.9.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.9.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.9.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.9.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.9.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.9.attn_output.weight buffer type overriden to CUDA0
Tensor blk.9.ffn_gate_exps.weight buffer type overriden to CUDA3
Tensor blk.9.ffn_down_exps.weight buffer type overriden to CUDA3
Tensor blk.9.ffn_up_exps.weight buffer type overriden to CUDA3
Tensor blk.9.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.9.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.9.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.10.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.10.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.10.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.10.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.10.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.10.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.10.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.10.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.10.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.10.attn_output.weight buffer type overriden to CUDA0
Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CUDA3
Tensor blk.10.ffn_down_exps.weight buffer type overriden to CUDA3
Tensor blk.10.ffn_up_exps.weight buffer type overriden to CUDA3
Tensor blk.10.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.10.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.10.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.11.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.11.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.11.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.11.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.11.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.11.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.11.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.11.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.11.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.11.attn_output.weight buffer type overriden to CUDA0
Tensor blk.11.ffn_gate_exps.weight buffer type overriden to CUDA3
Tensor blk.11.ffn_down_exps.weight buffer type overriden to CUDA3
Tensor blk.11.ffn_up_exps.weight buffer type overriden to CUDA3
Tensor blk.11.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.11.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.11.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.12.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.12.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.12.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.12.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.12.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.12.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.12.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.12.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.12.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.12.attn_output.weight buffer type overriden to CUDA0
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CUDA3
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CUDA3
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CUDA3
Tensor blk.12.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.12.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.12.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.13.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.13.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.13.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.13.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.13.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.13.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.13.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.13.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.13.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.13.attn_output.weight buffer type overriden to CUDA0
Tensor blk.13.ffn_gate_exps.weight buffer type overriden to CUDA4
Tensor blk.13.ffn_down_exps.weight buffer type overriden to CUDA4
Tensor blk.13.ffn_up_exps.weight buffer type overriden to CUDA4
Tensor blk.13.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.13.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.13.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.14.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.14.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.14.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.14.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.14.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.14.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.14.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.14.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.14.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.14.attn_output.weight buffer type overriden to CUDA0
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CUDA4
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CUDA4
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CUDA4
Tensor blk.14.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.14.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.14.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.15.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.15.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.15.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.15.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.15.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.15.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.15.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.15.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.15.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.15.attn_output.weight buffer type overriden to CUDA0
Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CUDA4
Tensor blk.15.ffn_down_exps.weight buffer type overriden to CUDA4
Tensor blk.15.ffn_up_exps.weight buffer type overriden to CUDA4
Tensor blk.15.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.15.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.15.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.16.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.16.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.16.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.16.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.16.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.16.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.16.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.16.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.16.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.16.attn_output.weight buffer type overriden to CUDA0
Tensor blk.16.ffn_gate_exps.weight buffer type overriden to CUDA4
Tensor blk.16.ffn_down_exps.weight buffer type overriden to CUDA4
Tensor blk.16.ffn_up_exps.weight buffer type overriden to CUDA4
Tensor blk.16.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.16.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.16.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.17.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.17.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.17.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.17.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.17.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.17.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.17.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.17.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.17.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.17.attn_output.weight buffer type overriden to CUDA0
Tensor blk.17.ffn_gate_exps.weight buffer type overriden to CUDA5
Tensor blk.17.ffn_down_exps.weight buffer type overriden to CUDA5
Tensor blk.17.ffn_up_exps.weight buffer type overriden to CUDA5
Tensor blk.17.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.17.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.17.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.18.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.18.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.18.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.18.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.18.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.18.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.18.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.18.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.18.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.18.attn_output.weight buffer type overriden to CUDA0
Tensor blk.18.ffn_gate_exps.weight buffer type overriden to CUDA5
Tensor blk.18.ffn_down_exps.weight buffer type overriden to CUDA5
Tensor blk.18.ffn_up_exps.weight buffer type overriden to CUDA5
Tensor blk.18.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.18.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.18.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.19.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.19.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.19.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.19.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.19.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.19.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.19.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.19.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.19.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.19.attn_output.weight buffer type overriden to CUDA0
Tensor blk.19.ffn_gate_exps.weight buffer type overriden to CUDA5
Tensor blk.19.ffn_down_exps.weight buffer type overriden to CUDA5
Tensor blk.19.ffn_up_exps.weight buffer type overriden to CUDA5
Tensor blk.19.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.19.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.19.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.20.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.20.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.20.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.20.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.20.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.20.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.20.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.20.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.20.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.20.attn_output.weight buffer type overriden to CUDA0
Tensor blk.20.ffn_gate_exps.weight buffer type overriden to CUDA5
Tensor blk.20.ffn_down_exps.weight buffer type overriden to CUDA5
Tensor blk.20.ffn_up_exps.weight buffer type overriden to CUDA5
Tensor blk.20.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.20.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.20.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.21.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.21.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.21.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.21.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.21.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.21.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.21.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.21.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.21.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.21.attn_output.weight buffer type overriden to CUDA0
Tensor blk.21.ffn_gate_exps.weight buffer type overriden to CUDA6
Tensor blk.21.ffn_down_exps.weight buffer type overriden to CUDA6
Tensor blk.21.ffn_up_exps.weight buffer type overriden to CUDA6
Tensor blk.21.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.21.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.21.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.22.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.22.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.22.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.22.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.22.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.22.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.22.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.22.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.22.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.22.attn_output.weight buffer type overriden to CUDA0
Tensor blk.22.ffn_gate_exps.weight buffer type overriden to CUDA6
Tensor blk.22.ffn_down_exps.weight buffer type overriden to CUDA6
Tensor blk.22.ffn_up_exps.weight buffer type overriden to CUDA6
Tensor blk.22.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.22.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.22.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.23.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.23.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.23.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.23.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.23.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.23.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.23.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.23.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.23.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.23.attn_output.weight buffer type overriden to CUDA0
Tensor blk.23.ffn_gate_exps.weight buffer type overriden to CUDA6
Tensor blk.23.ffn_down_exps.weight buffer type overriden to CUDA6
Tensor blk.23.ffn_up_exps.weight buffer type overriden to CUDA6
Tensor blk.23.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.23.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.23.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.24.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.24.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.24.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.24.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.24.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.24.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.24.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.24.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.24.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.24.attn_output.weight buffer type overriden to CUDA0
Tensor blk.24.ffn_gate_exps.weight buffer type overriden to CUDA6
Tensor blk.24.ffn_down_exps.weight buffer type overriden to CUDA6
Tensor blk.24.ffn_up_exps.weight buffer type overriden to CUDA6
Tensor blk.24.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.24.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.24.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.25.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.25.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.25.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.25.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.25.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.25.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.25.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.25.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.25.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.25.attn_output.weight buffer type overriden to CUDA0
Tensor blk.25.ffn_gate_exps.weight buffer type overriden to CUDA7
Tensor blk.25.ffn_down_exps.weight buffer type overriden to CUDA7
Tensor blk.25.ffn_up_exps.weight buffer type overriden to CUDA7
Tensor blk.25.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.25.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.25.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.26.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.26.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.26.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.26.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.26.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.26.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.26.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.26.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.26.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.26.attn_output.weight buffer type overriden to CUDA0
Tensor blk.26.ffn_gate_exps.weight buffer type overriden to CUDA7
Tensor blk.26.ffn_down_exps.weight buffer type overriden to CUDA7
Tensor blk.26.ffn_up_exps.weight buffer type overriden to CUDA7
Tensor blk.26.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.26.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.26.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.27.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.27.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.27.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.27.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.27.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.27.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.27.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.27.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.27.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.27.attn_output.weight buffer type overriden to CUDA0
Tensor blk.27.ffn_gate_exps.weight buffer type overriden to CUDA7
Tensor blk.27.ffn_down_exps.weight buffer type overriden to CUDA7
Tensor blk.27.ffn_up_exps.weight buffer type overriden to CUDA7
Tensor blk.27.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.27.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.27.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.28.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.28.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.28.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.28.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.28.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.28.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.28.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.28.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.28.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.28.attn_output.weight buffer type overriden to CUDA0
Tensor blk.28.ffn_gate_exps.weight buffer type overriden to CUDA7
Tensor blk.28.ffn_down_exps.weight buffer type overriden to CUDA7
Tensor blk.28.ffn_up_exps.weight buffer type overriden to CUDA7
Tensor blk.28.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.28.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.28.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.29.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.29.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.29.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.29.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.29.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.29.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.29.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.29.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.29.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.29.attn_output.weight buffer type overriden to CUDA0
Tensor blk.29.ffn_gate_exps.weight buffer type overriden to CUDA8
Tensor blk.29.ffn_down_exps.weight buffer type overriden to CUDA8
Tensor blk.29.ffn_up_exps.weight buffer type overriden to CUDA8
Tensor blk.29.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.29.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.29.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.30.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.30.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.30.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.30.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.30.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.30.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.30.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.30.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.30.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.30.attn_output.weight buffer type overriden to CUDA0
Tensor blk.30.ffn_gate_exps.weight buffer type overriden to CUDA8
Tensor blk.30.ffn_down_exps.weight buffer type overriden to CUDA8
Tensor blk.30.ffn_up_exps.weight buffer type overriden to CUDA8
Tensor blk.30.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.30.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.30.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.31.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.31.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.31.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.31.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.31.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.31.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.31.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.31.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.31.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.31.attn_output.weight buffer type overriden to CUDA0
Tensor blk.31.ffn_gate_exps.weight buffer type overriden to CUDA8
Tensor blk.31.ffn_down_exps.weight buffer type overriden to CUDA8
Tensor blk.31.ffn_up_exps.weight buffer type overriden to CUDA8
Tensor blk.31.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.31.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.31.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.32.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.32.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.32.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.32.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.32.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.32.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.32.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.32.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.32.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.32.attn_output.weight buffer type overriden to CUDA0
Tensor blk.32.ffn_gate_exps.weight buffer type overriden to CUDA8
Tensor blk.32.ffn_down_exps.weight buffer type overriden to CUDA8
Tensor blk.32.ffn_up_exps.weight buffer type overriden to CUDA8
Tensor blk.32.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.32.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.32.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.33.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.33.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.33.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.33.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.33.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.33.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.33.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.33.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.33.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.33.attn_output.weight buffer type overriden to CUDA0
Tensor blk.33.ffn_gate_exps.weight buffer type overriden to CUDA9
Tensor blk.33.ffn_down_exps.weight buffer type overriden to CUDA9
Tensor blk.33.ffn_up_exps.weight buffer type overriden to CUDA9
Tensor blk.33.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.33.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.33.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.34.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.34.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.34.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.34.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.34.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.34.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.34.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.34.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.34.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.34.attn_output.weight buffer type overriden to CUDA0
Tensor blk.34.ffn_gate_exps.weight buffer type overriden to CUDA9
Tensor blk.34.ffn_down_exps.weight buffer type overriden to CUDA9
Tensor blk.34.ffn_up_exps.weight buffer type overriden to CUDA9
Tensor blk.34.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.34.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.34.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.35.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.35.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.35.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.35.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.35.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.35.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.35.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.35.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.35.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.35.attn_output.weight buffer type overriden to CUDA0
Tensor blk.35.ffn_gate_exps.weight buffer type overriden to CUDA9
Tensor blk.35.ffn_down_exps.weight buffer type overriden to CUDA9
Tensor blk.35.ffn_up_exps.weight buffer type overriden to CUDA9
Tensor blk.35.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.35.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.35.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.36.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.36.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.36.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.36.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.36.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.36.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.36.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.36.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.36.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.36.attn_output.weight buffer type overriden to CUDA0
Tensor blk.36.ffn_gate_exps.weight buffer type overriden to CUDA9
Tensor blk.36.ffn_down_exps.weight buffer type overriden to CUDA9
Tensor blk.36.ffn_up_exps.weight buffer type overriden to CUDA9
Tensor blk.36.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.36.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.36.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.37.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.37.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.37.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.37.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.37.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.37.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.37.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.37.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.37.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.37.attn_output.weight buffer type overriden to CUDA0
Tensor blk.37.ffn_gate_exps.weight buffer type overriden to CUDA10
Tensor blk.37.ffn_down_exps.weight buffer type overriden to CUDA10
Tensor blk.37.ffn_up_exps.weight buffer type overriden to CUDA10
Tensor blk.37.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.37.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.37.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.38.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.38.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.38.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.38.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.38.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.38.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.38.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.38.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.38.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.38.attn_output.weight buffer type overriden to CUDA0
Tensor blk.38.ffn_gate_exps.weight buffer type overriden to CUDA10
Tensor blk.38.ffn_down_exps.weight buffer type overriden to CUDA10
Tensor blk.38.ffn_up_exps.weight buffer type overriden to CUDA10
Tensor blk.38.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.38.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.38.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.39.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.39.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.39.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.39.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.39.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.39.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.39.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.39.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.39.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.39.attn_output.weight buffer type overriden to CUDA0
Tensor blk.39.ffn_gate_exps.weight buffer type overriden to CUDA10
Tensor blk.39.ffn_down_exps.weight buffer type overriden to CUDA10
Tensor blk.39.ffn_up_exps.weight buffer type overriden to CUDA10
Tensor blk.39.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.39.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.39.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.40.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.40.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.40.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.40.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.40.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.40.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.40.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.40.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.40.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.40.attn_output.weight buffer type overriden to CUDA0
Tensor blk.40.ffn_gate_exps.weight buffer type overriden to CUDA10
Tensor blk.40.ffn_down_exps.weight buffer type overriden to CUDA10
Tensor blk.40.ffn_up_exps.weight buffer type overriden to CUDA10
Tensor blk.40.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.40.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.40.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.41.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.41.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.41.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.41.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.41.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.41.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.41.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.41.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.41.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.41.attn_output.weight buffer type overriden to CUDA0
Tensor blk.41.ffn_gate_exps.weight buffer type overriden to CUDA11
Tensor blk.41.ffn_down_exps.weight buffer type overriden to CUDA11
Tensor blk.41.ffn_up_exps.weight buffer type overriden to CUDA11
Tensor blk.41.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.41.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.41.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.42.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.42.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.42.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.42.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.42.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.42.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.42.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.42.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.42.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.42.attn_output.weight buffer type overriden to CUDA0
Tensor blk.42.ffn_gate_exps.weight buffer type overriden to CUDA11
Tensor blk.42.ffn_down_exps.weight buffer type overriden to CUDA11
Tensor blk.42.ffn_up_exps.weight buffer type overriden to CUDA11
Tensor blk.42.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.42.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.42.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.43.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.43.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.43.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.43.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.43.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.43.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.43.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.43.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.43.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.43.attn_output.weight buffer type overriden to CUDA0
Tensor blk.43.ffn_gate_exps.weight buffer type overriden to CUDA11
Tensor blk.43.ffn_down_exps.weight buffer type overriden to CUDA11
Tensor blk.43.ffn_up_exps.weight buffer type overriden to CUDA11
Tensor blk.43.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.43.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.43.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.44.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.44.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.44.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.44.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.44.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.44.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.44.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.44.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.44.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.44.attn_output.weight buffer type overriden to CUDA0
Tensor blk.44.ffn_gate_exps.weight buffer type overriden to CUDA11
Tensor blk.44.ffn_down_exps.weight buffer type overriden to CUDA11
Tensor blk.44.ffn_up_exps.weight buffer type overriden to CUDA11
Tensor blk.44.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.44.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.44.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.45.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.45.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.45.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.45.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.45.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.45.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.45.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.45.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.45.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.45.attn_output.weight buffer type overriden to CUDA0
Tensor blk.45.ffn_gate_exps.weight buffer type overriden to CUDA12
Tensor blk.45.ffn_down_exps.weight buffer type overriden to CUDA12
Tensor blk.45.ffn_up_exps.weight buffer type overriden to CUDA12
Tensor blk.45.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.45.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.45.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.46.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.46.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.46.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.46.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.46.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.46.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.46.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.46.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.46.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.46.attn_output.weight buffer type overriden to CUDA0
Tensor blk.46.ffn_gate_exps.weight buffer type overriden to CUDA12
Tensor blk.46.ffn_down_exps.weight buffer type overriden to CUDA12
Tensor blk.46.ffn_up_exps.weight buffer type overriden to CUDA12
Tensor blk.46.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.46.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.46.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.47.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.47.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.47.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.47.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.47.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.47.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.47.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.47.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.47.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.47.attn_output.weight buffer type overriden to CUDA0
Tensor blk.47.ffn_gate_exps.weight buffer type overriden to CUDA12
Tensor blk.47.ffn_down_exps.weight buffer type overriden to CUDA12
Tensor blk.47.ffn_up_exps.weight buffer type overriden to CUDA12
Tensor blk.47.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.47.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.47.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.48.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.48.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.48.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.48.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.48.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.48.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.48.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.48.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.48.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.48.attn_output.weight buffer type overriden to CUDA0
Tensor blk.48.ffn_gate_exps.weight buffer type overriden to CUDA12
Tensor blk.48.ffn_down_exps.weight buffer type overriden to CUDA12
Tensor blk.48.ffn_up_exps.weight buffer type overriden to CUDA12
Tensor blk.48.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.48.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.48.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.49.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.49.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.49.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.49.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.49.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.49.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.49.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.49.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.49.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.49.attn_output.weight buffer type overriden to CUDA0
Tensor blk.49.ffn_gate_exps.weight buffer type overriden to CUDA13
Tensor blk.49.ffn_down_exps.weight buffer type overriden to CUDA13
Tensor blk.49.ffn_up_exps.weight buffer type overriden to CUDA13
Tensor blk.49.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.49.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.49.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.50.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.50.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.50.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.50.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.50.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.50.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.50.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.50.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.50.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.50.attn_output.weight buffer type overriden to CUDA0
Tensor blk.50.ffn_gate_exps.weight buffer type overriden to CUDA13
Tensor blk.50.ffn_down_exps.weight buffer type overriden to CUDA13
Tensor blk.50.ffn_up_exps.weight buffer type overriden to CUDA13
Tensor blk.50.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.50.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.50.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.51.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.51.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.51.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.51.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.51.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.51.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.51.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.51.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.51.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.51.attn_output.weight buffer type overriden to CUDA0
Tensor blk.51.ffn_gate_exps.weight buffer type overriden to CUDA13
Tensor blk.51.ffn_down_exps.weight buffer type overriden to CUDA13
Tensor blk.51.ffn_up_exps.weight buffer type overriden to CUDA13
Tensor blk.51.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.51.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.51.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.52.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.52.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.52.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.52.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.52.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.52.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.52.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.52.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.52.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.52.attn_output.weight buffer type overriden to CUDA0
Tensor blk.52.ffn_gate_exps.weight buffer type overriden to CUDA13
Tensor blk.52.ffn_down_exps.weight buffer type overriden to CUDA13
Tensor blk.52.ffn_up_exps.weight buffer type overriden to CUDA13
Tensor blk.52.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.52.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.52.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.53.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.53.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.53.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.53.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.53.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.53.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.53.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.53.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.53.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.53.attn_output.weight buffer type overriden to CUDA0
Tensor blk.53.ffn_gate_exps.weight buffer type overriden to CUDA14
Tensor blk.53.ffn_down_exps.weight buffer type overriden to CUDA14
Tensor blk.53.ffn_up_exps.weight buffer type overriden to CUDA14
Tensor blk.53.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.53.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.53.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.54.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.54.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.54.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.54.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.54.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.54.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.54.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.54.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.54.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.54.attn_output.weight buffer type overriden to CUDA0
Tensor blk.54.ffn_gate_exps.weight buffer type overriden to CUDA14
Tensor blk.54.ffn_down_exps.weight buffer type overriden to CUDA14
Tensor blk.54.ffn_up_exps.weight buffer type overriden to CUDA14
Tensor blk.54.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.54.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.54.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.55.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.55.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.55.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.55.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.55.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.55.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.55.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.55.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.55.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.55.attn_output.weight buffer type overriden to CUDA0
Tensor blk.55.ffn_gate_exps.weight buffer type overriden to CUDA14
Tensor blk.55.ffn_down_exps.weight buffer type overriden to CUDA14
Tensor blk.55.ffn_up_exps.weight buffer type overriden to CUDA14
Tensor blk.55.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.55.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.55.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.56.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.56.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.56.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.56.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.56.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.56.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.56.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.56.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.56.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.56.attn_output.weight buffer type overriden to CUDA0
Tensor blk.56.ffn_gate_exps.weight buffer type overriden to CUDA14
Tensor blk.56.ffn_down_exps.weight buffer type overriden to CUDA14
Tensor blk.56.ffn_up_exps.weight buffer type overriden to CUDA14
Tensor blk.56.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.56.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.56.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.57.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.57.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.57.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.57.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.57.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.57.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.57.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.57.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.57.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.57.attn_output.weight buffer type overriden to CUDA0
Tensor blk.57.ffn_gate_exps.weight buffer type overriden to CUDA15
Tensor blk.57.ffn_down_exps.weight buffer type overriden to CUDA15
Tensor blk.57.ffn_up_exps.weight buffer type overriden to CUDA15
Tensor blk.57.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.57.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.57.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.58.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.58.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.58.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.58.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.58.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.58.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.58.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.58.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.58.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.58.attn_output.weight buffer type overriden to CUDA0
Tensor blk.58.ffn_gate_exps.weight buffer type overriden to CUDA15
Tensor blk.58.ffn_down_exps.weight buffer type overriden to CUDA15
Tensor blk.58.ffn_up_exps.weight buffer type overriden to CUDA15
Tensor blk.58.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.58.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.58.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.59.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.59.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.59.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.59.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.59.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.59.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.59.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.59.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.59.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.59.attn_output.weight buffer type overriden to CUDA0
Tensor blk.59.ffn_gate_exps.weight buffer type overriden to CUDA15
Tensor blk.59.ffn_down_exps.weight buffer type overriden to CUDA15
Tensor blk.59.ffn_up_exps.weight buffer type overriden to CUDA15
Tensor blk.59.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.59.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.59.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.60.attn_norm.weight buffer type overriden to CUDA0
Tensor blk.60.attn_q_a_norm.weight buffer type overriden to CUDA0
Tensor blk.60.attn_kv_a_norm.weight buffer type overriden to CUDA0
Tensor blk.60.attn_q_a.weight buffer type overriden to CUDA0
Tensor blk.60.attn_q_b.weight buffer type overriden to CUDA0
Tensor blk.60.attn_kv_a_mqa.weight buffer type overriden to CUDA0
Tensor blk.60.attn_kv_b.weight buffer type overriden to CUDA0
Tensor blk.60.attn_k_b.weight buffer type overriden to CUDA0
Tensor blk.60.attn_v_b.weight buffer type overriden to CUDA0
Tensor blk.60.attn_output.weight buffer type overriden to CUDA0
Tensor blk.60.ffn_gate_exps.weight buffer type overriden to CUDA15
Tensor blk.60.ffn_down_exps.weight buffer type overriden to CUDA15
Tensor blk.60.ffn_up_exps.weight buffer type overriden to CUDA15
Tensor blk.60.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.60.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.60.ffn_up_shexp.weight buffer type overriden to CUDA0
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size =   938.98 MiB
llm_load_tensors:      CUDA0 buffer size = 13510.41 MiB
llm_load_tensors:      CUDA1 buffer size = 19516.11 MiB
llm_load_tensors:      CUDA2 buffer size = 19516.11 MiB
llm_load_tensors:      CUDA3 buffer size = 20412.11 MiB
llm_load_tensors:      CUDA4 buffer size = 20412.11 MiB
llm_load_tensors:      CUDA5 buffer size = 20412.11 MiB
llm_load_tensors:      CUDA6 buffer size = 20412.11 MiB
llm_load_tensors:      CUDA7 buffer size = 20405.08 MiB
llm_load_tensors:      CUDA8 buffer size = 20412.11 MiB
llm_load_tensors:      CUDA9 buffer size = 20412.11 MiB
llm_load_tensors:     CUDA10 buffer size = 20412.11 MiB
llm_load_tensors:     CUDA11 buffer size = 20412.11 MiB
llm_load_tensors:     CUDA12 buffer size = 20412.11 MiB
llm_load_tensors:     CUDA13 buffer size = 20412.11 MiB
llm_load_tensors:     CUDA14 buffer size = 20412.11 MiB
llm_load_tensors:     CUDA15 buffer size = 20398.08 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 2048
llama_new_context_with_model: n_batch    = 2028
llama_new_context_with_model: n_ubatch   = 32
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: mla_attn   = 2
llama_new_context_with_model: attn_max_b = 64
llama_new_context_with_model: fused_moe  = 0
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
llama_kv_cache_init:      CUDA0 KV buffer size =     9.00 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =     9.00 MiB
llama_kv_cache_init:      CUDA2 KV buffer size =     9.00 MiB
llama_kv_cache_init:      CUDA3 KV buffer size =     9.00 MiB
llama_kv_cache_init:      CUDA4 KV buffer size =     9.00 MiB
llama_kv_cache_init:      CUDA5 KV buffer size =     9.00 MiB
llama_kv_cache_init:      CUDA6 KV buffer size =     9.00 MiB
llama_kv_cache_init:      CUDA7 KV buffer size =     6.75 MiB
llama_kv_cache_init:      CUDA8 KV buffer size =     9.00 MiB
llama_kv_cache_init:      CUDA9 KV buffer size =     9.00 MiB
llama_kv_cache_init:     CUDA10 KV buffer size =     9.00 MiB
llama_kv_cache_init:     CUDA11 KV buffer size =     9.00 MiB
llama_kv_cache_init:     CUDA12 KV buffer size =     9.00 MiB
llama_kv_cache_init:     CUDA13 KV buffer size =     9.00 MiB
llama_kv_cache_init:     CUDA14 KV buffer size =     9.00 MiB
llama_kv_cache_init:     CUDA15 KV buffer size =     4.50 MiB
llama_new_context_with_model: KV self size  =  137.25 MiB, c^KV (f16):  137.25 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.49 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=4)
llama_new_context_with_model:      CUDA0 compute buffer size =  1468.25 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =    32.84 MiB
llama_new_context_with_model:      CUDA2 compute buffer size =    36.59 MiB
llama_new_context_with_model:      CUDA3 compute buffer size =    40.33 MiB
llama_new_context_with_model:      CUDA4 compute buffer size =    40.33 MiB
llama_new_context_with_model:      CUDA5 compute buffer size =    40.33 MiB
llama_new_context_with_model:      CUDA6 compute buffer size =    40.33 MiB
llama_new_context_with_model:      CUDA7 compute buffer size =    36.55 MiB
llama_new_context_with_model:      CUDA8 compute buffer size =    36.59 MiB
llama_new_context_with_model:      CUDA9 compute buffer size =    36.59 MiB
llama_new_context_with_model:     CUDA10 compute buffer size =    36.59 MiB
llama_new_context_with_model:     CUDA11 compute buffer size =    36.59 MiB
llama_new_context_with_model:     CUDA12 compute buffer size =    36.59 MiB
llama_new_context_with_model:     CUDA13 compute buffer size =    36.59 MiB
llama_new_context_with_model:     CUDA14 compute buffer size =    36.59 MiB
llama_new_context_with_model:     CUDA15 compute buffer size =    32.52 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =     1.88 MiB
llama_new_context_with_model: graph nodes  = 4029
llama_new_context_with_model: graph splits = 267

system_info: n_threads = 64 / 128 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | 
perplexity: tokenizing the input ..
perplexity: tokenization took 1204.36 ms
perplexity: calculating perplexity over 141 chunks, n_ctx=2028, batch_size=2028, n_seq=1
perplexity: 70.85 seconds per pass - ETA 2 hours 46.48 minutes
[1]1.5133,[2]1.2808,[3]1.2341,[4]1.6816,[5]1.7724,[6]1.7134,[7]1.7954,[8]1.9248,[9]2.1154,[10]2.2877,[11]2.4139,[9]2.1154,[12]2.3717,[13]2.5064,[14]2.5977,[15]2.7054,[16]2.8206,[17]2.7759,[18]2.8313,[19]2.8545,[20]2.7818,[21]2.7411,[22]2.6726,[23]2.5926,[24]2.5094,[25]2.4654,[26]2.5402,[27]2.6242,[28]2.7251,[29]2.6552,[30]2.5845,[31]2.5249,[32]2.4696,[33]2.4319,[34]2.4648,[35]2.5107,[36]2.5134,[37]2.5077,[38]2.5006,[39]2.4985,[40]2.5242,[41]2.5522,[42]2.6270,[43]2.6997,[44]2.6485,[45]2.5994,[46]2.6544,[47]2.6998,[48]2.7290,[49]2.7548,[50]2.7789,[51]2.7882,[52]2.8015,[53]2.8134,[54]2.8264,[55]2.8218,[56]2.8227,[57]2.8107,[58]2.8214,[59]2.8053,[60]2.8405,[61]2.8767,[62]2.9020,[63]2.8949,[64]2.9172,[65]2.9293,[66]2.9319,[67]2.9343,[68]2.9453,[69]2.9333,[70]2.9217,[71]2.9474,[72]2.9769,[73]2.9905,[74]2.9796,[75]2.9839,[76]2.9927,[77]3.0167,[78]2.9955,[79]3.0122,[80]3.0144,[81]3.0211,[82]3.0257,[83]3.0287,[84]3.0425,[85]3.0437,[86]3.0509,[87]3.0627,[88]3.0398,[89]3.0763,[90]3.1096,[91]3.1269,[92]3.1512,[93]3.1802,[94]3.2084,[95]3.2396,[96]3.2350,[97]3.2529,[98]3.2644,[99]3.2377,[100]3.2021,[101]3.1667,[102]3.1324,[103]3.0998,[104]3.0928,[105]3.0824,[106]3.0844,[107]3.0846,[108]3.0857,[109]3.0879,[110]3.0681,[111]3.0622,[112]3.0618,[113]3.0734,[114]3.0871,[115]3.0913,[116]3.1058,[117]3.1218,[118]3.1188,[119]3.1139,[120]3.1146,[121]3.1146,[122]3.1158,[123]3.1285,[124]3.1435,[125]3.1476,[126]3.1483,[127]3.1492,[128]3.1659,[129]3.1433,[130]3.1422,[131]3.1402,[132]3.1419,[133]3.1278,[134]3.1179,[135]3.1065,[136]3.1172,[137]3.1240,[138]3.1041,[139]3.0823,[140]3.0687,[141]3.0848,
Final estimate: PPL = 3.0848 +/- 0.01608

llama_print_timings:        load time =  704652.36 ms
llama_print_timings:      sample time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings: prompt eval time = 9852711.72 ms / 285948 tokens (   34.46 ms per token,    29.02 tokens per second)
llama_print_timings:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings:       total time = 9856368.39 ms / 285949 tokens
```

Final size on `llama-server` init: 

```
llm_load_print_meta: model size       = 311.346 GiB (3.980 BPW) 
llm_load_print_meta: repeating layers = 309.721 GiB (3.970 BPW, 670.196 B parameters)
```

---

👤 **jukofyork** commented the **2025-03-08** at **23:45:48**:<br>

@saood06 Mine was using the default chunk size of 512:

```
perplexity: calculating perplexity over 561 chunks, n_ctx=512, batch_size=2048, n_seq=4
```

I'm actually in the process of rewriting the `llama.cpp` code as I think it was suffering from some numerical problems.

I have the non-MLA version done now and running perplexity overnight, and will have the MLA version done over the next few days and put up a PR.

---

👤 **saood06** commented the **2025-03-09** at **01:02:39**:<br>

> @saood06 Mine was using the default chunk size of 512:
> 
> ```
> perplexity: calculating perplexity over 561 chunks, n_ctx=512, batch_size=2048, n_seq=4
> ```

Sorry, I missed that detail. Larger chunk sizes does mean lower ppl and thus not comparable.

---

👤 **jukofyork** commented the **2025-03-09** at **11:04:40**:<br>

This is for the non-MLA version that stores the decompressed K/V:

```
Final estimate: PPL = 3.3497 +/- 0.01848

llama_perf_context_print:        load time =   13347.43 ms
llama_perf_context_print: prompt eval time = 14395199.19 ms / 287232 tokens (   50.12 ms per token,    19.95 tokens per second)
llama_perf_context_print:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_perf_context_print:       total time = 14407917.86 ms / 287233 tokens
```

```cpp
static ggml_type llama_tensor_get_type(quantize_state_impl & qs, ggml_type new_type, const ggml_tensor * tensor, llama_ftype ftype) {
    const std::string name = ggml_get_name(tensor);
    if (name.find("_exps") != std::string::npos) {
        return name.find("ffn_down") != std::string::npos ? GGML_TYPE_Q6_K : GGML_TYPE_Q5_K;
    } else if (name.find("attn_") != std::string::npos && name.find("_output") == std::string::npos) {
        return name.find("attn_kv_b") != std::string::npos ? GGML_TYPE_Q2_K : GGML_TYPE_BF16;
    }
    return GGML_TYPE_Q8_0;
}
```

I've now got all the matrices split so should hopefully be able to find which are responsible for the numerical instabilities instead of using `BF16` for them all like this.

I'll post the MLA perplexity results in a couple of days when I've written and tested it.

---

👤 **davidsyoung** commented the **2025-03-09** at **11:23:17**:<br>

> This is for the non-MLA version that stores the decompressed K/V:
> 
> ```
> Final estimate: PPL = 3.3497 +/- 0.01848
> 
> llama_perf_context_print:        load time =   13347.43 ms
> llama_perf_context_print: prompt eval time = 14395199.19 ms / 287232 tokens (   50.12 ms per token,    19.95 tokens per second)
> llama_perf_context_print:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
> llama_perf_context_print:       total time = 14407917.86 ms / 287233 tokens
> ```
> 
> ```c++
> static ggml_type llama_tensor_get_type(quantize_state_impl & qs, ggml_type new_type, const ggml_tensor * tensor, llama_ftype ftype) {
>     const std::string name = ggml_get_name(tensor);
>     if (name.find("_exps") != std::string::npos) {
>         return name.find("ffn_down") != std::string::npos ? GGML_TYPE_Q6_K : GGML_TYPE_Q5_K;
>     } else if (name.find("attn_") != std::string::npos && name.find("_output") == std::string::npos) {
>         return GGML_TYPE_BF16;
>     }
>     return GGML_TYPE_Q8_0;
> }
> ```
> 
> I've now got all the attention matrices split up:
> 
> ```
> llama_model_loader: - type  f32:  361 tensors
> llama_model_loader: - type q8_0:  246 tensors
> llama_model_loader: - type q5_K:  116 tensors
> llama_model_loader: - type q6_K:   58 tensors
> llama_model_loader: - type bf16:  488 tensors
> print_info: file format = GGUF V3 (latest)
> print_info: file type   = Q5_K - Medium
> print_info: file size   = 467.54 GiB (5.98 BPW) 
> ```
> 
> so should hopefully be able to find which are responsible for the numerical instabilities instead of using `BF16` for them all like this.
> 
> I'll post the MLA perplexity results in a couple of days when I've written and tested it.

Which chunk size is this? I’ll see if I can replicate

---

👤 **jukofyork** commented the **2025-03-09** at **12:11:03**:<br>

> Which chunk size is this? I’ll see if I can replicate

Just the default. If you remove your `-ctx 2048` then it should work (check it says 561 chunks).

---

👤 **davidsyoung** commented the **2025-03-09** at **18:38:32**:<br>

```
root@1dcba5bcd62f:/app/build/bin# ./llama-perplexity -m /storage/DeepSeek-R1-GGroot@1dcba5bcd62f:/app/build/bin# ./llama-perplexity -m /storage/DeepSeek-R1-GGUF-IQ3_S.gguf -f /models/wiki.test.raw -fmoe -mla 2 -fa -c 512 -ub 512 --n-gpu-layers 100 -ts 41,23.5,26,24.5,23.5,25.5,24.4,23.5,25.5,24.5,23.5,25.5,24.5,23.5,25.5,30
main: build = 0 (unknown)
main: built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu
main: seed  = 1741529602
llama_model_loader: loaded meta data with 53 key-value pairs and 1147 tensors from /storage/DeepSeek-R1-GGUF-IQ3_S.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = unsloth_DeepSeek R1 BF16
llama_model_loader: - kv   3:                         general.size_label str              = 256x21B
llama_model_loader: - kv   4:                            general.license str              = mit
llama_model_loader: - kv   5:                   general.base_model.count u32              = 1
llama_model_loader: - kv   6:                  general.base_model.0.name str              = DeepSeek R1
llama_model_loader: - kv   7:          general.base_model.0.organization str              = Deepseek Ai
llama_model_loader: - kv   8:              general.base_model.0.repo_url str              = https://huggingface.co/deepseek-ai/De...
llama_model_loader: - kv   9:                               general.tags arr[str,3]       = ["deepseek", "unsloth", "transformers"]
llama_model_loader: - kv  10:                          general.languages arr[str,1]       = ["en"]
llama_model_loader: - kv  11:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv  12:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv  13:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv  14:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  15:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv  16:          deepseek2.attention.head_count_kv u32              = 128
llama_model_loader: - kv  17:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  18: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  19:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  20:                          general.file_type u32              = 26
llama_model_loader: - kv  21:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  22:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  23:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  24:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  25:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  26:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  27:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  28:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  29:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  30:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  31:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  32:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  33:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  34:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  35:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  36: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  37: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  38:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  39:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  40:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<�...
llama_model_loader: - kv  41:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  42:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  43:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  44:            tokenizer.ggml.padding_token_id u32              = 128815
llama_model_loader: - kv  45:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  46:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  47:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  48:               general.quantization_version u32              = 2
llama_model_loader: - kv  49:                      quantize.imatrix.file str              = /models/deepseek-config/imatrix.dat
llama_model_loader: - kv  50:                   quantize.imatrix.dataset str              = imatrix-training-full-3
llama_model_loader: - kv  51:             quantize.imatrix.entries_count i32              = 720
llama_model_loader: - kv  52:              quantize.imatrix.chunks_count i32              = 315
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  305 tensors
llama_model_loader: - type q5_K:   61 tensors
llama_model_loader: - type q6_K:    1 tensors
llama_model_loader: - type iq3_s:  419 tensors
loaded 127741 merges from merges.txt
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
llm_load_print_meta: model ftype      = IQ3_S - 3.4375 bpw
llm_load_print_meta: model params     = 672.050 B
llm_load_print_meta: model size       = 274.160 GiB (3.504 BPW) 
llm_load_print_meta: repeating layers = 273.081 GiB (3.500 BPW, 670.196 B parameters)
llm_load_print_meta: general.name     = unsloth_DeepSeek R1 BF16
llm_load_print_meta: BOS token        = 0 '<｜begin▁of▁sentence｜>'
llm_load_print_meta: EOS token        = 1 '<｜end▁of▁sentence｜>'
llm_load_print_meta: PAD token        = 128815 '<｜PAD▁TOKEN｜>'
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
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 16 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 1: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 2: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 3: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 4: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 5: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 6: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 7: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 8: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 9: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 10: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 11: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 12: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 13: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 14: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 15: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
llm_load_tensors: ggml ctx size =    7.94 MiB
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size =   379.74 MiB
llm_load_tensors:      CUDA0 buffer size = 20184.59 MiB
llm_load_tensors:      CUDA1 buffer size = 14413.91 MiB
llm_load_tensors:      CUDA2 buffer size = 19218.55 MiB
llm_load_tensors:      CUDA3 buffer size = 19218.55 MiB
llm_load_tensors:      CUDA4 buffer size = 14413.91 MiB
llm_load_tensors:      CUDA5 buffer size = 19218.55 MiB
llm_load_tensors:      CUDA6 buffer size = 19218.55 MiB
llm_load_tensors:      CUDA7 buffer size = 14413.91 MiB
llm_load_tensors:      CUDA8 buffer size = 19218.55 MiB
llm_load_tensors:      CUDA9 buffer size = 19218.55 MiB
llm_load_tensors:     CUDA10 buffer size = 14413.91 MiB
llm_load_tensors:     CUDA11 buffer size = 19218.55 MiB
llm_load_tensors:     CUDA12 buffer size = 19218.55 MiB
llm_load_tensors:     CUDA13 buffer size = 14413.91 MiB
llm_load_tensors:     CUDA14 buffer size = 19218.55 MiB
llm_load_tensors:     CUDA15 buffer size = 15138.89 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 2048
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 2
llama_new_context_with_model: attn_max_b = 0
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
llama_kv_cache_init:      CUDA0 KV buffer size =    15.75 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =     6.75 MiB
llama_kv_cache_init:      CUDA2 KV buffer size =     9.00 MiB
llama_kv_cache_init:      CUDA3 KV buffer size =     9.00 MiB
llama_kv_cache_init:      CUDA4 KV buffer size =     6.75 MiB
llama_kv_cache_init:      CUDA5 KV buffer size =     9.00 MiB
llama_kv_cache_init:      CUDA6 KV buffer size =     9.00 MiB
llama_kv_cache_init:      CUDA7 KV buffer size =     6.75 MiB
llama_kv_cache_init:      CUDA8 KV buffer size =     9.00 MiB
llama_kv_cache_init:      CUDA9 KV buffer size =     9.00 MiB
llama_kv_cache_init:     CUDA10 KV buffer size =     6.75 MiB
llama_kv_cache_init:     CUDA11 KV buffer size =     9.00 MiB
llama_kv_cache_init:     CUDA12 KV buffer size =     9.00 MiB
llama_kv_cache_init:     CUDA13 KV buffer size =     6.75 MiB
llama_kv_cache_init:     CUDA14 KV buffer size =     9.00 MiB
llama_kv_cache_init:     CUDA15 KV buffer size =     6.75 MiB
llama_new_context_with_model: KV self size  =  137.25 MiB, c^KV (f16):  137.25 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     1.97 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=4)
llama_new_context_with_model:      CUDA0 compute buffer size =   842.01 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =   810.01 MiB
llama_new_context_with_model:      CUDA2 compute buffer size =   810.01 MiB
llama_new_context_with_model:      CUDA3 compute buffer size =   810.01 MiB
llama_new_context_with_model:      CUDA4 compute buffer size =   810.01 MiB
llama_new_context_with_model:      CUDA5 compute buffer size =   810.01 MiB
llama_new_context_with_model:      CUDA6 compute buffer size =   810.01 MiB
llama_new_context_with_model:      CUDA7 compute buffer size =   810.01 MiB
llama_new_context_with_model:      CUDA8 compute buffer size =   810.01 MiB
llama_new_context_with_model:      CUDA9 compute buffer size =   810.01 MiB
llama_new_context_with_model:     CUDA10 compute buffer size =   810.01 MiB
llama_new_context_with_model:     CUDA11 compute buffer size =   810.01 MiB
llama_new_context_with_model:     CUDA12 compute buffer size =   810.01 MiB
llama_new_context_with_model:     CUDA13 compute buffer size =   810.01 MiB
llama_new_context_with_model:     CUDA14 compute buffer size =   810.01 MiB
llama_new_context_with_model:     CUDA15 compute buffer size =   810.02 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =    30.02 MiB
llama_new_context_with_model: graph nodes  = 3548
llama_new_context_with_model: graph splits = 17

system_info: n_threads = 64 / 128 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | 
perplexity: tokenizing the input ..
perplexity: tokenization took 1181.15 ms
perplexity: calculating perplexity over 561 chunks, n_ctx=512, batch_size=2048, n_seq=4
perplexity: 32.24 seconds per pass - ETA 1 hours 15.35 minutes
[1]2.6244,[2]3.4337,[3]2.4394,[4]2.0435,[5]1.8531,[6]1.7028,[7]1.6113,[8]1.5441,[9]1.4882,[10]1.4465,[11]1.4396,[12]1.4815,[13]1.4919,[14]1.6259,[15]1.7558,[16]1.8155,[17]1.9845,[18]2.1162,[19]2.0780,[20]2.0646,[21]2.1708,[22]2.1438,[23]2.1148,[24]2.1260,[25]2.0964,[26]2.0704,[27]2.1173,[28]2.1248,[29]2.1754,[30]2.2067,[31]2.2413,[32]2.2582,[33]2.2979,[34]2.3424,[35]2.3939,[36]2.4484,[37]2.4838,[38]2.5303,[39]2.5721,[40]2.6318,[41]2.6717,[42]2.6829,[43]2.7311,[44]2.7477,[45]2.8301,[46]2.8825,[47]2.8386,[48]2.7931,[49]2.7716,[50]2.7926,[51]2.8378,[52]2.8529,[53]2.9039,[54]2.9172,[55]2.9496,[56]2.9808,[57]2.9957,[58]3.0330,[59]3.0447,[60]3.0941,[61]3.1352,[62]3.1905,[63]3.2247,[64]3.2711,[65]3.2812,[66]3.2665,[67]3.2443,[68]3.2771,[69]3.2740,[70]3.2910,[71]3.3096,[72]3.3263,[73]3.3396,[74]3.3634,[75]3.3414,[76]3.2926,[77]3.2484,[78]3.2446,[79]3.2220,[80]3.2049,[81]3.1693,[82]3.1751,[83]3.1443,[84]3.1080,[85]3.0733,[86]3.0516,[87]3.0485,[88]3.0204,[89]3.0049,[90]2.9781,[91]2.9491,[92]2.9253,[93]2.8976,[94]2.8737,[95]2.8510,[96]2.8487,[97]2.8562,[98]2.8403,[99]2.8252,[100]2.8277,[101]2.8208,[102]2.8386,[103]2.8657,[104]2.8839,[105]2.8803,[106]2.9034,[107]2.9288,[108]2.9501,[109]2.9846,[110]3.0197,[111]3.0398,[112]3.0124,[113]2.9989,[114]2.9768,[115]2.9609,[116]2.9485,[117]2.9245,[118]2.9019,[119]2.8807,[120]2.8615,[121]2.8464,[122]2.8279,[123]2.8109,[124]2.7917,[125]2.7744,[126]2.7566,[127]2.7427,[128]2.7341,[129]2.7246,[130]2.7135,[131]2.7066,[132]2.7140,[133]2.7233,[134]2.7293,[135]2.7401,[136]2.7561,[137]2.7719,[138]2.7799,[139]2.7912,[140]2.7913,[141]2.7924,[142]2.7912,[143]2.7910,[144]2.7871,[145]2.7775,[146]2.7757,[147]2.7804,[148]2.7798,[149]2.7811,[150]2.7756,[151]2.7739,[152]2.7703,[153]2.7660,[154]2.7661,[155]2.7701,[156]2.7718,[157]2.7774,[158]2.7863,[159]2.7883,[160]2.7969,[161]2.8045,[162]2.8142,[163]2.8193,[164]2.8398,[165]2.8640,[166]2.8818,[167]2.8947,[168]2.9190,[169]2.9422,[170]2.9637,[171]2.9874,[172]2.9710,[173]2.9532,[174]2.9392,[175]2.9258,[176]2.9135,[177]2.9020,[178]2.8886,[179]2.8741,[180]2.8779,[181]2.8923,[182]2.9074,[183]2.9224,[184]2.9370,[185]2.9472,[186]2.9640,[187]2.9796,[188]2.9942,[189]3.0054,[190]3.0059,[191]3.0130,[192]3.0169,[193]3.0221,[194]3.0419,[195]3.0510,[196]3.0642,[197]3.0743,[198]3.0782,[199]3.0838,[200]3.0830,[201]3.0985,[202]3.0926,[203]3.0980,[204]3.1012,[205]3.1014,[206]3.1037,[207]3.1127,[208]3.1217,[209]3.1312,[210]3.1315,[211]3.1265,[212]3.1264,[213]3.1340,[214]3.1353,[215]3.1413,[216]3.1415,[217]3.1372,[218]3.1362,[219]3.1371,[220]3.1356,[221]3.1357,[222]3.1352,[223]3.1356,[224]3.1407,[225]3.1421,[226]3.1339,[227]3.1315,[228]3.1337,[229]3.1380,[230]3.1446,[231]3.1510,[232]3.1426,[233]3.1352,[234]3.1356,[235]3.1341,[236]3.1436,[237]3.1517,[238]3.1613,[239]3.1713,[240]3.1802,[241]3.1918,[242]3.2067,[243]3.2200,[244]3.2287,[245]3.2404,[246]3.2510,[247]3.2499,[248]3.2452,[249]3.2430,[250]3.2365,[251]3.2337,[252]3.2362,[253]3.2398,[254]3.2472,[255]3.2537,[256]3.2574,[257]3.2596,[258]3.2602,[259]3.2634,[260]3.2653,[261]3.2663,[262]3.2654,[263]3.2707,[264]3.2727,[265]3.2729,[266]3.2744,[267]3.2771,[268]3.2814,[269]3.2841,[270]3.2830,[271]3.2810,[272]3.2741,[273]3.2744,[274]3.2683,[275]3.2575,[276]3.2476,[277]3.2494,[278]3.2595,[279]3.2659,[280]3.2739,[281]3.2814,[282]3.2880,[283]3.2944,[284]3.3010,[285]3.3150,[286]3.3174,[287]3.3207,[288]3.3254,[289]3.3277,[290]3.3192,[291]3.3101,[292]3.3090,[293]3.3080,[294]3.3056,[295]3.3033,[296]3.3054,[297]3.3059,[298]3.3107,[299]3.3169,[300]3.3198,[301]3.3234,[302]3.3263,[303]3.3282,[304]3.3274,[305]3.3389,[306]3.3469,[307]3.3578,[308]3.3457,[309]3.3405,[310]3.3309,[311]3.3342,[312]3.3367,[313]3.3437,[314]3.3459,[315]3.3491,[316]3.3505,[317]3.3519,[318]3.3524,[319]3.3527,[320]3.3568,[321]3.3569,[322]3.3585,[323]3.3653,[324]3.3657,[325]3.3709,[326]3.3753,[327]3.3797,[328]3.3828,[329]3.3841,[330]3.3905,[331]3.3945,[332]3.3994,[333]3.3978,[334]3.3977,[335]3.3982,[336]3.3980,[337]3.3991,[338]3.3993,[339]3.4017,[340]3.4051,[341]3.4106,[342]3.4198,[343]3.4296,[344]3.4352,[345]3.4270,[346]3.4193,[347]3.4149,[348]3.4074,[349]3.4041,[350]3.4023,[351]3.4073,[352]3.4222,[353]3.4315,[354]3.4444,[355]3.4534,[356]3.4587,[357]3.4710,[358]3.4808,[359]3.4838,[360]3.4902,[361]3.4994,[362]3.5083,[363]3.5144,[364]3.5211,[365]3.5278,[366]3.5386,[367]3.5473,[368]3.5542,[369]3.5621,[370]3.5707,[371]3.5847,[372]3.5935,[373]3.5965,[374]3.6002,[375]3.6048,[376]3.6180,[377]3.6292,[378]3.6317,[379]3.6313,[380]3.6278,[381]3.6326,[382]3.6383,[383]3.6423,[384]3.6466,[385]3.6503,[386]3.6568,[387]3.6625,[388]3.6656,[389]3.6546,[390]3.6447,[391]3.6339,[392]3.6280,[393]3.6188,[394]3.6100,[395]3.6005,[396]3.5900,[397]3.5807,[398]3.5707,[399]3.5602,[400]3.5523,[401]3.5421,[402]3.5312,[403]3.5224,[404]3.5117,[405]3.5018,[406]3.4914,[407]3.4816,[408]3.4722,[409]3.4634,[410]3.4572,[411]3.4583,[412]3.4536,[413]3.4558,[414]3.4583,[415]3.4555,[416]3.4556,[417]3.4584,[418]3.4525,[419]3.4545,[420]3.4520,[421]3.4506,[422]3.4519,[423]3.4512,[424]3.4553,[425]3.4547,[426]3.4556,[427]3.4546,[428]3.4577,[429]3.4591,[430]3.4620,[431]3.4631,[432]3.4622,[433]3.4582,[434]3.4584,[435]3.4513,[436]3.4448,[437]3.4407,[438]3.4390,[439]3.4358,[440]3.4410,[441]3.4461,[442]3.4539,[443]3.4526,[444]3.4534,[445]3.4545,[446]3.4593,[447]3.4626,[448]3.4652,[449]3.4682,[450]3.4723,[451]3.4754,[452]3.4779,[453]3.4795,[454]3.4778,[455]3.4799,[456]3.4801,[457]3.4823,[458]3.4877,[459]3.4881,[460]3.4880,[461]3.4845,[462]3.4883,[463]3.4954,[464]3.5006,[465]3.4942,[466]3.4925,[467]3.4911,[468]3.4923,[469]3.4894,[470]3.4865,[471]3.4871,[472]3.4881,[473]3.4875,[474]3.4865,[475]3.4878,[476]3.4861,[477]3.4852,[478]3.4857,[479]3.4875,[480]3.4901,[481]3.4855,[482]3.4889,[483]3.4880,[484]3.4917,[485]3.4981,[486]3.5010,[487]3.5045,[488]3.5100,[489]3.5124,[490]3.5170,[491]3.5233,[492]3.5278,[493]3.5275,[494]3.5286,[495]3.5311,[496]3.5328,[497]3.5357,[498]3.5359,[499]3.5353,[500]3.5395,[501]3.5439,[502]3.5429,[503]3.5412,[504]3.5432,[505]3.5465,[506]3.5549,[507]3.5574,[508]3.5608,[509]3.5529,[510]3.5472,[511]3.5410,[512]3.5369,[513]3.5304,[514]3.5285,[515]3.5308,[516]3.5264,[517]3.5262,[518]3.5252,[519]3.5256,[520]3.5304,[521]3.5293,[522]3.5278,[523]3.5337,[524]3.5322,[525]3.5306,[526]3.5259,[527]3.5208,[528]3.5176,[529]3.5144,[530]3.5112,[531]3.5079,[532]3.5020,[533]3.4956,[534]3.4913,[535]3.4923,[536]3.4952,[537]3.4984,[538]3.5015,[539]3.5043,[540]3.5097,[541]3.5129,[542]3.5153,[543]3.5098,[544]3.5061,[545]3.5058,[546]3.4989,[547]3.4923,[548]3.4855,[549]3.4788,[550]3.4728,[551]3.4665,[552]3.4605,[553]3.4549,[554]3.4535,[555]3.4521,[556]3.4549,[557]3.4588,[558]3.4647,[559]3.4693,[560]3.4747,[561]3.4727,
Final estimate: PPL = 3.4727 +/- 0.01905

llama_print_timings:        load time = 7984522.50 ms
llama_print_timings:      sample time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings: prompt eval time = 1411983.15 ms / 287232 tokens (    4.92 ms per token,   203.42 tokens per second)
llama_print_timings:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings:       total time = 1421631.91 ms / 287233 tokens
```

Doing this from mobile so can’t format easily, sorry for length. This is IQ3_S standard format. Don’t have any handy quants available at the moment that doesn’t cause any NaN issues. 

This is with 512 chunks @jukofyork.

---

👤 **davidsyoung** commented the **2025-03-09** at **19:31:37**:<br>

I think I’ve found out why I was getting NaNs before. Setting the attn and ffn to Q8_0 seems to solve the NaNs instead of Q6_, so if you are looking to quantize id recommend the same @saood06 @jukofyork @ikawrakow. 

This is producing correct perplexity values:

```
./llama-quantize --imatrix /models/deepseek-config/imatrix.dat \
  --token-embedding-type q8_0 \
  --attn-q-type q8_0 \
  --attn-k-type q8_0 \
  --attn-v-type q8_0 \
  --attn-qkv-type q8_0 \
  --attn-output-type q8_0 \
  --ffn-gate-type q8_0 \
  --ffn-down-type q8_0 \
  --ffn-up-type q8_0 \
  --custom-q "\.ffn_.*_shexp\.weight=q6_K,output\.weight=q6_K" \
  --custom-q "blk\.3\.ffn_down_exps\.weight=q5_K,blk\.4\.ffn_down_exps\.weight=q5_K,blk\.5\.ffn_down_exps\.weight=q5_K,blk\.3\.ffn_up_exps\.weight=iq4_k,blk\.3\.ffn_gate_exps\.weight=iq4_k,blk\.4\.ffn_up_exps\.weight=iq4_k,blk\.4\.ffn_gate_exps\.weight=iq4_k,blk\.5\.ffn_up_exps\.weight=iq4_k,blk\.5\.ffn_gate_exps\.weight=iq4_k" \
  --custom-q "blk\.6\.ffn_down_exps\.weight=q5_K,blk\.7\.ffn_down_exps\.weight=q5_K,blk\.8\.ffn_down_exps\.weight=q5_K,blk\.6\.ffn_up_exps\.weight=iq4_k,blk\.6\.ffn_gate_exps\.weight=iq4_k,blk\.7\.ffn_up_exps\.weight=iq4_k,blk\.7\.ffn_gate_exps\.weight=iq4_k,blk\.8\.ffn_up_exps\.weight=iq4_k,blk\.8\.ffn_gate_exps\.weight=iq4_k" \
  --custom-q "blk\.9\.ffn_down_exps\.weight=iq4_k,blk\.10\.ffn_down_exps\.weight=iq4_k,blk\.11\.ffn_down_exps\.weight=iq4_k,blk\.12\.ffn_down_exps\.weight=iq4_k,blk\.9\.ffn_up_exps\.weight=iq3_s,blk\.9\.ffn_gate_exps\.weight=iq3_s,blk\.10\.ffn_up_exps\.weight=iq3_s,blk\.10\.ffn_gate_exps\.weight=iq3_s,blk\.11\.ffn_up_exps\.weight=iq3_s,blk\.11\.ffn_gate_exps\.weight=iq3_s,blk\.12\.ffn_up_exps\.weight=iq3_s,blk\.12\.ffn_gate_exps\.weight=iq3_s" \
  --custom-q "blk\.13\.ffn_down_exps\.weight=iq4_k,blk\.14\.ffn_down_exps\.weight=iq4_k,blk\.15\.ffn_down_exps\.weight=iq4_k,blk\.16\.ffn_down_exps\.weight=iq4_k,blk\.13\.ffn_up_exps\.weight=iq3_s,blk\.13\.ffn_gate_exps\.weight=iq3_s,blk\.14\.ffn_up_exps\.weight=iq3_s,blk\.14\.ffn_gate_exps\.weight=iq3_s,blk\.15\.ffn_up_exps\.weight=iq3_s,blk\.15\.ffn_gate_exps\.weight=iq3_s,blk\.16\.ffn_up_exps\.weight=iq3_s,blk\.16\.ffn_gate_exps\.weight=iq3_s" \
  --custom-q "blk\.17\.ffn_down_exps\.weight=iq4_k,blk\.18\.ffn_down_exps\.weight=iq4_k,blk\.19\.ffn_down_exps\.weight=iq4_k,blk\.20\.ffn_down_exps\.weight=iq4_k,blk\.17\.ffn_up_exps\.weight=iq3_s,blk\.17\.ffn_gate_exps\.weight=iq3_s,blk\.18\.ffn_up_exps\.weight=iq3_s,blk\.18\.ffn_gate_exps\.weight=iq3_s,blk\.19\.ffn_up_exps\.weight=iq3_s,blk\.19\.ffn_gate_exps\.weight=iq3_s,blk\.20\.ffn_up_exps\.weight=iq3_s,blk\.20\.ffn_gate_exps\.weight=iq3_s" \
  --custom-q "blk\.21\.ffn_down_exps\.weight=iq4_k,blk\.22\.ffn_down_exps\.weight=iq4_k,blk\.23\.ffn_down_exps\.weight=iq4_k,blk\.24\.ffn_down_exps\.weight=iq4_k,blk\.21\.ffn_up_exps\.weight=iq3_s,blk\.21\.ffn_gate_exps\.weight=iq3_s,blk\.22\.ffn_up_exps\.weight=iq3_s,blk\.22\.ffn_gate_exps\.weight=iq3_s,blk\.23\.ffn_up_exps\.weight=iq3_s,blk\.23\.ffn_gate_exps\.weight=iq3_s,blk\.24\.ffn_up_exps\.weight=iq3_s,blk\.24\.ffn_gate_exps\.weight=iq3_s" \
  --custom-q "blk\.25\.ffn_down_exps\.weight=iq4_k,blk\.26\.ffn_down_exps\.weight=iq4_k,blk\.27\.ffn_down_exps\.weight=iq4_k,blk\.28\.ffn_down_exps\.weight=iq4_k,blk\.25\.ffn_up_exps\.weight=iq3_s,blk\.25\.ffn_gate_exps\.weight=iq3_s,blk\.26\.ffn_up_exps\.weight=iq3_s,blk\.26\.ffn_gate_exps\.weight=iq3_s,blk\.27\.ffn_up_exps\.weight=iq3_s,blk\.27\.ffn_gate_exps\.weight=iq3_s,blk\.28\.ffn_up_exps\.weight=iq3_s,blk\.28\.ffn_gate_exps\.weight=iq3_s" \
  --custom-q "blk\.29\.ffn_down_exps\.weight=iq4_k,blk\.30\.ffn_down_exps\.weight=iq4_k,blk\.31\.ffn_down_exps\.weight=iq4_k,blk\.32\.ffn_down_exps\.weight=iq4_k,blk\.29\.ffn_up_exps\.weight=iq3_s,blk\.29\.ffn_gate_exps\.weight=iq3_s,blk\.30\.ffn_up_exps\.weight=iq3_s,blk\.30\.ffn_gate_exps\.weight=iq3_s,blk\.31\.ffn_up_exps\.weight=iq3_s,blk\.31\.ffn_gate_exps\.weight=iq3_s,blk\.32\.ffn_up_exps\.weight=iq3_s,blk\.32\.ffn_gate_exps\.weight=iq3_s" \
  --custom-q "blk\.33\.ffn_down_exps\.weight=iq4_k,blk\.34\.ffn_down_exps\.weight=iq4_k,blk\.35\.ffn_down_exps\.weight=iq4_k,blk\.36\.ffn_down_exps\.weight=iq4_k,blk\.33\.ffn_up_exps\.weight=iq3_s,blk\.33\.ffn_gate_exps\.weight=iq3_s,blk\.34\.ffn_up_exps\.weight=iq3_s,blk\.34\.ffn_gate_exps\.weight=iq3_s,blk\.35\.ffn_up_exps\.weight=iq3_s,blk\.35\.ffn_gate_exps\.weight=iq3_s,blk\.36\.ffn_up_exps\.weight=iq3_s,blk\.36\.ffn_gate_exps\.weight=iq3_s" \
  --custom-q "blk\.37\.ffn_down_exps\.weight=iq4_k,blk\.38\.ffn_down_exps\.weight=iq4_k,blk\.39\.ffn_down_exps\.weight=iq4_k,blk\.40\.ffn_down_exps\.weight=iq4_k,blk\.37\.ffn_up_exps\.weight=iq3_s,blk\.37\.ffn_gate_exps\.weight=iq3_s,blk\.38\.ffn_up_exps\.weight=iq3_s,blk\.38\.ffn_gate_exps\.weight=iq3_s,blk\.39\.ffn_up_exps\.weight=iq3_s,blk\.39\.ffn_gate_exps\.weight=iq3_s,blk\.40\.ffn_up_exps\.weight=iq3_s,blk\.40\.ffn_gate_exps\.weight=iq3_s" \
  --custom-q "blk\.41\.ffn_down_exps\.weight=iq4_k,blk\.42\.ffn_down_exps\.weight=iq4_k,blk\.43\.ffn_down_exps\.weight=iq4_k,blk\.44\.ffn_down_exps\.weight=iq4_k,blk\.41\.ffn_up_exps\.weight=iq3_s,blk\.41\.ffn_gate_exps\.weight=iq3_s,blk\.42\.ffn_up_exps\.weight=iq3_s,blk\.42\.ffn_gate_exps\.weight=iq3_s,blk\.43\.ffn_up_exps\.weight=iq3_s,blk\.43\.ffn_gate_exps\.weight=iq3_s,blk\.44\.ffn_up_exps\.weight=iq3_s,blk\.44\.ffn_gate_exps\.weight=iq3_s" \
  --custom-q "blk\.45\.ffn_down_exps\.weight=iq4_k,blk\.46\.ffn_down_exps\.weight=iq4_k,blk\.47\.ffn_down_exps\.weight=iq4_k,blk\.48\.ffn_down_exps\.weight=iq4_k,blk\.45\.ffn_up_exps\.weight=iq3_s,blk\.45\.ffn_gate_exps\.weight=iq3_s,blk\.46\.ffn_up_exps\.weight=iq3_s,blk\.46\.ffn_gate_exps\.weight=iq3_s,blk\.47\.ffn_up_exps\.weight=iq3_s,blk\.47\.ffn_gate_exps\.weight=iq3_s,blk\.48\.ffn_up_exps\.weight=iq3_s,blk\.48\.ffn_gate_exps\.weight=iq3_s" \
  --custom-q "blk\.49\.ffn_down_exps\.weight=iq4_k,blk\.50\.ffn_down_exps\.weight=iq4_k,blk\.51\.ffn_down_exps\.weight=iq4_k,blk\.52\.ffn_down_exps\.weight=iq4_k,blk\.49\.ffn_up_exps\.weight=iq3_s,blk\.49\.ffn_gate_exps\.weight=iq3_s,blk\.50\.ffn_up_exps\.weight=iq3_s,blk\.50\.ffn_gate_exps\.weight=iq3_s,blk\.51\.ffn_up_exps\.weight=iq3_s,blk\.51\.ffn_gate_exps\.weight=iq3_s,blk\.52\.ffn_up_exps\.weight=iq3_s,blk\.52\.ffn_gate_exps\.weight=iq3_s" \
  --custom-q "blk\.53\.ffn_down_exps\.weight=iq4_k,blk\.54\.ffn_down_exps\.weight=iq4_k,blk\.55\.ffn_down_exps\.weight=iq4_k,blk\.56\.ffn_down_exps\.weight=iq4_k,blk\.53\.ffn_up_exps\.weight=iq3_s,blk\.53\.ffn_gate_exps\.weight=iq3_s,blk\.54\.ffn_up_exps\.weight=iq3_s,blk\.54\.ffn_gate_exps\.weight=iq3_s,blk\.55\.ffn_up_exps\.weight=iq3_s,blk\.55\.ffn_gate_exps\.weight=iq3_s,blk\.56\.ffn_up_exps\.weight=iq3_s,blk\.56\.ffn_gate_exps\.weight=iq3_s" \
  --custom-q "blk\.57\.ffn_down_exps\.weight=iq4_k,blk\.58\.ffn_down_exps\.weight=iq4_k,blk\.59\.ffn_down_exps\.weight=iq4_k,blk\.60\.ffn_down_exps\.weight=iq4_k,blk\.57\.ffn_up_exps\.weight=iq3_s,blk\.57\.ffn_gate_exps\.weight=iq3_s,blk\.58\.ffn_up_exps\.weight=iq3_s,blk\.58\.ffn_gate_exps\.weight=iq3_s,blk\.59\.ffn_up_exps\.weight=iq3_s,blk\.59\.ffn_gate_exps\.weight=iq3_s,blk\.60\.ffn_up_exps\.weight=iq3_s,blk\.60\.ffn_gate_exps\.weight=iq3_s" \
  /storage/DeepSeek-R1-GGUF/unsloth_DeepSeek-R1-BF16-256x21B-F16-00001-of-00059.gguf \
  /models/DeepSeek-R1-GGUF/DeepSeek-R1-GGUF-IQ4_K__iq3_s-Q8.gguf \
  q8_0 64
```

---

👤 **ikawrakow** commented the **2025-03-10** at **05:25:18**:<br>

You are using `mla = 2`?
Do you get the NaNs also without MLA?

Yes, I changed the precision for the `K*Q` multiplication to `f32` because the model seemed too dumb. But I only changed it for token generation because with DeepSeek-Lite I'm getting the correct PPL, so I thought the numerical instability only applies to short contexts. I don't expect the PR to change the PPL results.

---

👤 **davidsyoung** commented the **2025-03-10** at **05:30:52**:<br>

> You are using `mla = 2`? Do you get the NaNs also without MLA?
> 
> Yes, I changed the precision for the `K*Q` multiplication to `f32` because the model seemed too dumb. But I only changed it for token generation because with DeepSeek-Lite I'm getting the correct PPL, so I thought the numerical instability only applies to short contexts. I don't expect the PR to change the PPL results.

Yes I get NaN’s with all combinations from what I can see. I detailed some of it in https://github.com/ikawrakow/ik_llama.cpp/issues/245. I believe it _may_ have to do with q6_K or some tensors not being set to q8_0 precision. 

Works with IQ3_M:

```
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  306 tensors
llama_model_loader: - type q5_K:   61 tensors
llama_model_loader: - type q6_K:    1 tensors
llama_model_loader: - type iq3_s:  407 tensors
llama_model_loader: - type iq4_k:   11 tensors
```

Doesn’t work - IQ4_K__iq3_s

```
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:   62 tensors
llama_model_loader: - type q5_K:    6 tensors
llama_model_loader: - type q6_K:  550 tensors
llama_model_loader: - type iq3_s:  104 tensors
llama_model_loader: - type iq4_k:   64 tensors
```

It seemed that the new quant I made lasts for longer without producing NaNs, and it has less q6_K. 

I want to test further, but we’ve had a power cut at home and the server is offline till I’m home later today.

---

👤 **ikawrakow** commented the **2025-03-10** at **05:57:41**:<br>

Try adding
```
--custom-q "\.attn_.*\.weight=q8_0"
```
to your quantization command. Also perhaps a good idea to replace
```
--custom-q "\.ffn_.*_shexp\.weight=q6_K,output\.weight=q6_K" \
```
with
```
--custom-q "\.ffn_.*_shexp\.weight=q5_K,output\.weight=q8_0" \
```

Do you know how many batches of what size were used to calculate the imatrix that you are using?

---

👤 **davidsyoung** commented the **2025-03-10** at **06:16:09**:<br>

Good idea. I’ll re-quant with these later today and update when done! 

I’m not sure on imatrix batch size. 

https://huggingface.co/mradermacher/DeepSeek-R1-i1-GGUF

Using from here.

---

👤 **orca-zhang** commented the **2025-03-14** at **05:32:46**:<br>

During the test, a lot of garbled characters appeared. When used with -fmoe, continuous DDDDDDD output appeared.

```
numactl --interleave=all ./build/bin/llama-cli -m /root/models/DeepSeek-R1-11446-Q2_K/DeepSeek-R1-11446-Q2_K-00001-of-00030.gguf -cnv -p "You are a helpful assistant." -fa --temp 0.6 --top-p 0.95 -s 3047 -if -mli -t 124 -nkvo -c 4096 -ngl 0 -mla 2 -ser 7,1
```

llm_load_print_meta: model type       = 671B
llm_load_print_meta: model ftype      = Q2_K - Medium
llm_load_print_meta: model params     = 672.050 B
llm_load_print_meta: model size       = 227.689 GiB (2.910 BPW)
llm_load_print_meta: repeating layers = 226.697 GiB (2.906 BPW, 670.196 B parameters)

llm_load_tensors: ggml ctx size =    0.47 MiB
llm_load_tensors: offloading 0 repeating layers to GPU
llm_load_tensors: offloaded 0/62 layers to GPU
llm_load_tensors:        CPU buffer size =  7738.41 MiB
llm_load_tensors:        CPU buffer size =  7982.63 MiB
llm_load_tensors:        CPU buffer size =  7982.63 MiB
llm_load_tensors:        CPU buffer size =  7982.63 MiB
llm_load_tensors:        CPU buffer size =  7982.63 MiB
llm_load_tensors:        CPU buffer size =  7982.63 MiB
llm_load_tensors:        CPU buffer size =  7982.63 MiB
llm_load_tensors:        CPU buffer size =  7982.63 MiB
llm_load_tensors:        CPU buffer size =  7982.63 MiB
llm_load_tensors:        CPU buffer size =  7982.63 MiB
llm_load_tensors:        CPU buffer size =  7982.63 MiB
llm_load_tensors:        CPU buffer size =  7982.63 MiB
llm_load_tensors:        CPU buffer size =  7982.63 MiB
llm_load_tensors:        CPU buffer size =  7982.63 MiB
llm_load_tensors:        CPU buffer size =  7982.63 MiB
llm_load_tensors:        CPU buffer size =  7982.63 MiB
llm_load_tensors:        CPU buffer size =  7982.63 MiB
llm_load_tensors:        CPU buffer size =  7982.63 MiB
llm_load_tensors:        CPU buffer size =  7982.63 MiB
llm_load_tensors:        CPU buffer size =  7982.63 MiB
llm_load_tensors:        CPU buffer size =  7982.63 MiB
llm_load_tensors:        CPU buffer size =  7982.63 MiB
llm_load_tensors:        CPU buffer size =  7982.63 MiB
llm_load_tensors:        CPU buffer size =  7982.63 MiB
llm_load_tensors:        CPU buffer size =  7982.63 MiB
llm_load_tensors:        CPU buffer size =  7982.63 MiB
llm_load_tensors:        CPU buffer size =  7982.63 MiB
llm_load_tensors:        CPU buffer size =  7982.63 MiB
llm_load_tensors:        CPU buffer size =  8707.58 MiB
llm_load_tensors:        CPU buffer size =  1176.05 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 4096
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 2
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 0
llama_new_context_with_model: ser        = 7, 1
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025

llama_kv_cache_init:  CUDA_Host KV buffer size =   274.50 MiB
llama_new_context_with_model: KV self size  =  274.50 MiB, c^KV (f16):  274.50 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.49 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =  1796.13 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =    22.01 MiB

> 9.8 vs 9.11?
<think>
Okay prontera tent might be the main hub now, but players still refer to it as pront. So why the version difference between 9.8 and 9.11? Let me think. Maybe it's a typo? Or perhaps different clients use different map names based on updates. Old Ragnarok Online had frequent updates, so some might have pront as 9.8 and others 9.11. Wait emails often have typos. Wait Natalies Dereck vs Dereck Natalies? Oh, maybe different sources label the same location differently. Wait Natalies could be Dereck's variant. Wait Natalies Dereck might be a different area altogether. Hmm. Maybe 9.8 is pront and Dereck is Natalies, but why numbers 9.8 vs 9.11? Wait chronological order? If pront is central and Dereck is elsewhere, but Natalies sounds like a name. Wait Natalies could be plural plural mishearing. Wait Natalies Dereck maybe it's a translation or client difference. Oh! Some clients have pront/map named prontera 9.8 vs pr domic domicmans dolphinsmans字典oor domic或许是 Mill数月 Mill人名人名 profilMiss interferonebemans Missbekebe Totebeyersrona MissebeebeedeebeMiss Omnrona Misseberonaebe和海晗 erectannotationmans Codes ellipteneinne impregn-platformOFFasuk domicss� Mill-platformronaariahronaebe benefits domicebemansariahbertebeebe domic班长 Sich Dome数年 antiviral Becsignronaanyaebebertiative anonymousronaebeeke Becety Oval Omn脚下ariahJBJBmans VirtMissyers attacking脚下的痞 domiciative domic erect domiciativeanyaariahadb MAG Omn和海 domiceberonaebeIUMoye erect Signature脚下的iativeebeekeiative Becador erectpeabecronayers intramronaebeanya Millyersebeebeebeebeebe sofebeZBronaMissabdMiss Pew Miss底下othebeebeebebert Omn impregnronaJBronaadeariah slipronaety erect Missebe antiviralene erectadorbec antiviral689ador也不行班长ronabecronaanyabecistarona Pew Subsronaeneronaevronabec脚下adorronabecronaronabecronarona Omn仇 domicrona689 BecganronaadorIUMrona693禧Miss Peweberonabertronaeberonaronaabd班长rona vergeronabertronabia ellipticalronaadbrona Missebeabdaea antiviralrijJB和海椭圆 Pew Omn antiviral surelyrona slip Goff脚下perianchendperianchendzetperianHallalerperian]**perianoyagger
>

---

```
numactl --interleave=all ./build/bin/llama-cli -m /root/models/DeepSeek-R1-11446-Q2_K/DeepSeek-R1-11446-Q2_K-00001-of-00030.gguf -cnv -p "You are a helpful assistant." -fa --temp 0.6 --top-p 0.95 -s 3047 -if -mli -t 124 -nkvo -c 4096 -ngl 0 -mla 2 -ser 6,1
```

> 9.8 vs 9.11?
<think>
Okay binanceso I need to compare Python 3.8.6 with Python 3.9.11. Wait策 but the user asked about "3.8 vs 9.11", but maybe they meant 3.8 vs 3.9.11. Let's check the versions.

First, I should figure out the release dates and key differences between these versions. Python 3ative3.8 was released in October 2019, while 3.9.11 is an update of the 3.9 series, which was released in August 2020. But 3.9.11 is a maintenance release with bug fixes and security updates.

Key features in 3.9 vs 3.8: Python 3.9 introduced new features like the merge diagnostics operators (| and |= for dicts), assignments in decorators, etc. Also年代new string methods KS萤 midsuffix SKIPsDoughey猫, improved type annotations, and more.

Performance improvements in 3.9 include more efficient handling of certain operations. For example年代the new parser in 3.9 allows niabaheweniri素有嵌入式嵌入嵌入式嵌入式嵌入式eal Corb嵌入式嵌入式iri inters嵌入式嵌入式嵌入式REF嵌入式素有ABLE081嵌入式REF inters嵌入式iriREF377CAM268CAM498ealiersiri嵌入式48嵌入式eeeREFREF嵌入式377嵌入式247嵌入式嵌入式08嵌入式REFREF08ASAASA嵌入式247257eeeREFFACE嵌入式ABLE498257嵌入式CAM嵌入式257otype Staffordestra嵌入式REF嵌入式CAM naz嵌入式REF080嵌入式 Chambersiri西斯 borderingiriefa081嵌入式080esterneeeirimCAM所属嵌入式REFeaeee嵌入式061257嵌入式257iri大雪嵌入式嵌入式嵌入式ASA Martialeal嵌入式嵌入式estra西斯嵌入式嵌入式eeeiri怪efa Alic257素有estraABLE reference嵌入式iriCAMiri退回嵌入式嵌入式eaestra257OdingleiriREF嵌入式嵌入式嵌入式嵌入式iri嵌入式eanasti257estra爱人498 Corbbabeee498080嵌入式wallingle Nazis嵌入式 FacesCAM嵌入式498498CAM嵌入式estra257素有REF fict嵌入式iri嵌入式REFola Corbestra Corb LeoneREF Emission嵌入式嵌入式iri嵌入式 tyl Petro08REFCAM嵌入式eee如下图所示嵌入式网点REFREF嵌入式247 fict inters嵌入式REF naz嵌入式 fict fict257iriestraalla081iri ChambersolaREF GobCAMREF Helper嵌入式yy Brideusestrairi KieREFolaREF tylREF嵌入式嵌入式 sealedeal tylREF谅嵌入式空空498iri tyl AAI嵌入式261 inters嵌入式eee嵌入式窃 gen generals暖 generativeoger老大كامabusabus卖dera retic generative MesaHarris Sain generative卖 dipdera凝 Mangroll卖的dera念念 Sain mutatedothe.op卖的deraothe卖ogerantzemon memor暖abus Sain genabus Generderalep generalsderaantz Sainoger deput aspir Sainothe Sain Sain Gener窃 Santiago Sell暖 stolenauf Sain dipdera Forces generativeothe Sainothe郎 generalslde郎ulanopf mutated SainPort manifest quoteabus自作 gen.opabusudal Tie manifest暖antz mutated卖的 manifestabus收回antz自作 Montreal暖 inner lic gen manifestantz是否是 manifestPartial Montreal Lect Mullegy plaque mutatedvesteraugh memorBLE manifestolk undersPartialPartial manifestvester unders Ley manifestgravity Sain自作 manifest卖的othe郎 demon CMepsionivoy CM Sain摩 gendet completeness manifest Ontario ration plaquesdial SainPartialPartial manifest-Geolk-selfderaGab dipdialjem manifest Muraolk Sain定义的Gab的颜色 blunt tripleDialstandingPartial plaque MendكامPartialolk賣大力 demon manifestPartial郎 Lectaugh SainPartialPartialelling直属olk Sain忠实 Sain Sain blinding Ontariolde Sain卖的定义的 squirrel completenessPartialPartialmissionolk自作 Chern completeness Shields domest MesaPartial Civ Mesa Ontario leftoverPartial plaquenad blinding Ontario lic Ontario自作 Sain annotationPartial Lect Ontario郎 quadru郎 Sain Ontario Sain Menderra郎vester spare-self Saindera Ontario completenessPartialPartialPartial證據 Beneffabprojectszers643zynonis涯证据zynDim Beneferts AlamATE Alyonisreckzyn证据人人zynerse Dediam清清ividzynprojectszynysty DeS格的ManualATE证据zyn extrapenisivid的水果直接将zynivid Ded格的停电涯 Benef直接将ebackDim Cenividzukivid Benef hypotheticalengesDim DeS AlyfaberseENEPrivacy墙上fabfolderALTlaidersefabervilleoniserse格的 Ded consadders Pasc款式 extrapivid的水果 expireerseonisauconis Beneferse Kenterseobreerse师大 Baltimoreerse极了 PPEerse墙上zynraeonis Perm然大 Benefonis涯 Dedvineividividzynenzzyn证据肩上accioystyystyterre Vetprojectsenth直接将ankarBE_[inkaguessprojectszukovidyticsysty肩上zynividteryATTerse在那iotaENE涯onisonisjos Fung１２projectsterrezukertserseervilleerts肩上onisividterre Grab的唯一odemcturepodonis extrapividonis颇-settingankarobreerse-meividnersprojectsividjos极了 pess burntaminesمارivid extrap pess|_{iota seedsividertsertsividividPieceonisertsprojectserse/textividprojects的水果 mapsersezukfabividertsyticsividsomonisyticsonisonis Warwick墙上erseervilleervillegyz signed Jacqugraphs的黑 Jacqu人人１２的人口 Jacqu�锦绣倍数erse的人口ervilleonisonis upliftjosfolder Pearlgraphsyg Norwegianonis停电 kinskas Moorkas Tran TF Structured Structured Kins Structured Structureddumparnation Structured kins Tran Structured Bodies昨日 origin Structured Cic Structured^- Structured origin mortal健康成长 originropic^- Structured tran Lesser originkas Structuredkas Structured Structured Structured Structured Bertrandkas Structuredkas不快kas Structured_o Structuredkaskamp Structured Structured Structuredkas Structuredkas Structured Structured Structuredkas Structured Structured Structuredkaskas Structured Structured Structured Structuredkas Structured允 Kins Structuredkas Structured Structured Structured Structuredkas Structured tonnekas Structured Structured Structured Structured Structured Structured Structuredkas Structured Structured Structured Structured Structured Structuredkamp Structured Structured Structured Structured Structuredkom Structuredjee补贴 Structured Structured Structuredkas Structured Structured Structured Structuredkas Structured Structured Structuredropic Structured Structured补贴kom Structured Structured Structured有那么补贴 Structured Structuredkas Structured Structured Structured Structured Structuredirim Structuredropic Structured Structured StructuredMos Structured Structured Structured Structuredkas Structured Structured Structured Structured Structured Structured Structured Structured Structured Structured Structured Structured Structured Structured Structured Structured Structured Structured cru Structured Structuredkas cru Structured Structured Structured Structured Structured Structured Structured Structured Structured Structured Structured Structuredkaskas Structured Structured Structured Structured Structured大家的 Structured Structured Structuredrency Structured忽略了kas Structured Structuredkas Structured mortalkas Structured Structured StructuredRevised Structured Structuredkaskas Structured Structured Structuredkas Structured Structured Structured Structuredkas Structured Structured三口ropic Structured允允ocha Structured Structured Structured Structured kins Structured Structured Structuredkaskas Structured Structured Structuredkas Structuredkaskas Structuredkas Structured Structured Structured cru locus Hels行政执法achal.decodeilot Helsachal Hels行政执法achal.decodemate永生 BSD.decodecter banachalcterontiachal BanCamphanCamp ban banCampabelsonti内脏achal Hels Ban Hels Helsachal BSD Ban永生 Ban ban ban Ban locus Lotus locus ban全域achalachal locusachalону banachalilot banachal.decodeachalCamp你呢 Banachalachal Ban永生resi永生二进制Campotype内脏永生achal内脏永生achal locushan banachal永生 ban LohCSI十进制永生 banachal ban永生 BSDachalabels locusону banrorcterachalachalachal丈 reputation永生行政执法Campachal locus banvement永生 ban banachalachalону Helsachal Sark永生Camp BSD locus Loh Helscter Lovedachalachalachal Hels ban永生内脏novilot Ban Banban永生 BanCampCamp永生 Lom

---

👤 **ikawrakow** commented the **2025-03-14** at **08:08:06**:<br>

Can you try building without CUDA? Thanks.

---

👤 **davidsyoung** commented the **2025-03-14** at **09:06:14**:<br>

Also worth trying a different quant. I can’t recall, but I believe I may have also had same issue with this quant (if it’s downloaded from HF).

---

👤 **orca-zhang** commented the **2025-03-18** at **05:42:45**:<br>

> Can you try building without CUDA? Thanks.

./buildCPU/bin/llama-cli -m /root/models/DeepSeek-R1-11446-Q2_K/DeepSeek-R1-11446-Q2_K-00001-of-00030.gguf -cnv -p "You are a helpful assistant." -fa --temp 0.6 --top-p 0.95 -s 3047 -if -mli -t 124 -nkvo -c 4096 -ngl 0 -mla 2 -ser 7,1

llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q2_K:  544 tensors
llama_model_loader: - type q3_K:  180 tensors
llama_model_loader: - type q6_K:    1 tensors
llama_model_loader: - type iq4_nl:   61 tensors

llm_load_print_meta: model type       = 671B
llm_load_print_meta: model ftype      = Q2_K - Medium
llm_load_print_meta: model params     = 672.050 B
llm_load_print_meta: model size       = 227.689 GiB (2.910 BPW)
llm_load_print_meta: repeating layers = 226.697 GiB (2.906 BPW, 670.196 B parameters)

....................................................................................................
llama_new_context_with_model: n_ctx      = 4096
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 2
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 0
llama_new_context_with_model: ser        = 7, 1
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init:        CPU KV buffer size =   274.50 MiB
llama_new_context_with_model: KV self size  =  274.50 MiB, c^KV (f16):  274.50 MiB, kv^T: not used
llama_new_context_with_model:        CPU  output buffer size =     0.49 MiB
llama_new_context_with_model:        CPU compute buffer size =   822.01 MiB
llama_new_context_with_model: graph nodes  = 3481
llama_new_context_with_model: graph splits = 1
main: chat template example: You are a helpful assistant

<｜User｜>Hello<｜Assistant｜>Hi there<｜end▁of▁sentence｜><｜User｜>How are you?<｜Assistant｜>

system_info: n_threads = 124 / 128 | AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 |
main: interactive mode on.
sampling:
	repeat_last_n = 64, repeat_penalty = 1.000, frequency_penalty = 0.000, presence_penalty = 0.000
	top_k = 40, tfs_z = 1.000, top_p = 0.950, min_p = 0.050, typical_p = 1.000, temp = 0.600
	mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000
sampling order:
CFG -> Penalties -> top_k -> tfs_z -> typical_p -> top_p -> min_p -> temperature
generate: n_ctx = 4096, n_batch = 2048, n_predict = -1, n_keep = 1


== Running in interactive mode. ==
 - Press Ctrl+C to interject at any time.
 - To return control to the AI, end your input with '\'.
 - To return control without starting a new line, end your input with '/'.

You are a helpful assistant.


> 9.8 vs 9.11?
<think> Earhart

</think>
**Era delimitada:** 9.11
**Consulta:** "9.8 vs 9.11?"

**Preg sacada de**
e
**Resposta:**

**Entiendo que te refieres a fechas históricas** (9.8 y 9.11), pero el formato no es claro. Aquí algunas posibilidades con base en fechas relevantes:

1. **9 de agosto** (9.8) vs **11 de septiembre** (9.11):**
   - **9 de agosto:** Destacan eventos como el **lanzamiento de la misión Souz translación (misión a Marte) en 1988**.
   - **11 de septiembre:** Conmemoración del **atentado terrorista del 11-S en 2001**, un evento clave en la historia moderna.

2. **En formato numérico** (9.8 vs 9.11), podría referirse a **versiones de software** o **códigos**, pero no hay referencias claras en ese contexto.

**Si es otra interpretación Hal electroparalle共建iativeicha Trent际becbecpole际听过hitbecayne/interayne际 Signature际ayneTRYbiaiative成都ayneTRYbec際aynemansaynepolehit shinepole SSpoleayne际ayneatively际bec泻ldonbec盆atively际bec剩余际ivatpoleatively际ativelypole Becativiativebecbecpole initiative Becativelypole shine盆iativesieshine措 Signature incomerad sitpole Trent scav际ldon际polepole际

> Ctrl+C