### 🐛 [#245](https://github.com/ikawrakow/ik_llama.cpp/issues/245) - Bug: Perplexity returns NaN with IQ4_KSS quantisation

| **Author** | `davidsyoung` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-03-07 |
| **Updated** | 2025-03-12 |

---

#### Description

### What happened?

I said I would open a separate issue for this instead of discussing under an irrelevant pull request - let me know if you'd rather me continue over there @ikawrakow.

So I have tracked down the bug with `llama-perplexity` returning NaN's. To be clear, this is with IQ4_KSS quantisation. I have ran ``llama-perplexity` with IQ3_M without any issues. Which, was also made with the same imatrix.dat.

The command that works under IQ3_M is as follows:

```
./llama-perplexity -m /models/DeepSeek-R1-GGUF/DeepSeek-R1-GGUF-IQ3_M.gguf -f /models/wiki.test.raw -fmoe -fa -c 2048 -ub 2048 --n-gpu-layers 100
```
---


I tried to initially replicate this across to IQ4_KSS, but it started to produce NaNs. From there, I tested no attention, mla, different combinations, etc to no prevail. Here are some combinations that were tested that produced NaNs: 

---

# -fa -ub 1024 -ot ... = NaN

```
root@887d1e7c1690:/app# ./llama-perplexity \
  -m /models/DeepSeek-R1-GGUF/DeepSeek-R1-GGUF-IQ4_KSS-v2.gguf \
  -f /models/wiki.test.raw \
  -fa \
  -c 2048 \
  -ub 1024 \
  -ngl 100 \
  -ot ...

...

perplexity: tokenizing the input ..
perplexity: tokenization took 1252.89 ms
perplexity: calculating perplexity over 140 chunks, n_ctx=2048, batch_size=2048, n_seq=1
perplexity: 15.37 seconds per pass - ETA 35.85 minutes
[1]nan,[2]nan,[3]nan,[4]nan,[5]nan,[6]nan,[7]nan,[8]nan,[9]nan,[10]nan,[11]nan,[12]nan,[13]nan,[14]nan,[15]nan,[16]nan,[17]nan,[18]nan,[19]nan,[20]nan,[21]nan,[22]nan,[23]nan,[24]nan,[25]nan,[26]nan,[27]nan,[28]nan,[29]nan,[30]nan,[31]nan,[32]nan,[33]nan,[34]nan,[35]nan,[36]nan,[37]nan
```

---

# -mla 2  -ub 512 --seed --temp --amb -ot ... = NaN

```
./llama-perplexity \
  -m /models/DeepSeek-R1-GGUF/DeepSeek-R1-GGUF-IQ4_KSS-v2.gguf \
  -f /models/wiki.test.raw \
  -mla 2 \
  -c 2048 \
  -ub 512 \
  -ngl 100 \
  --seed 3407 \
  --temp 0.5 \
  -amb 64 \
  -ot ... \
  -ts 24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24

system_info: n_threads = 64 / 128 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | 
perplexity: tokenizing the input ..
perplexity: tokenization took 1231.71 ms
perplexity: calculating perplexity over 140 chunks, n_ctx=2048, batch_size=2048, n_seq=1
perplexity: 22.04 seconds per pass - ETA 51.43 minutes
[1]nan,[2]nan,^C^C
```

---

# -fa -ub 8 --seed --temp --amb 64 -ot = Works!

```
./llama-perplexity \
  -m /models/DeepSeek-R1-GGUF/DeepSeek-R1-GGUF-IQ4_KSS-v2.gguf \
  -f /models/wiki.test.raw \
  -fa \
  -c 2048 \
  -ub 8 \
  -ngl 100 \
  --seed 3407 \
  --temp 0.5 \
  -amb 64 \
  -ot ...
  -ts 24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24

system_info: n_threads = 64 / 128 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | 
perplexity: tokenizing the input ..
perplexity: tokenization took 1211.1 ms
perplexity: calculating perplexity over 140 chunks, n_ctx=2048, batch_size=2048, n_seq=1
perplexity: 69.34 seconds per pass - ETA 2 hours 41.78 minutes
[1]1.5140,[2]1.2829,[3]1.2362,[4]1.6902,[5]1.7468,[6]1.7194,[7]1.8258,[8]1.9479,[9]2.1370,[10]2.3270,[11]2.4503,[12]2.3282,[13]2.4525,[14]2.5484,[15]2.6761,[16]2.7952,[17]2.7793,[18]2.8372,[19]2.7767,[20]2.6981,[21]2.6288,[22]2.5562,[23]2.4682,[24]2.4149
```

---

I figured it out when I read your comment here: https://github.com/ikawrakow/ik_llama.cpp/issues/103#issuecomment-2434735396

This quant was created with the following (I requanted the BF16-GGUF and this IQ4_KSS to be certain it wasn't a quantisation issue, but it could be the types here, namely IQ4_KSS possibly):

```
./llama-quantize --imatrix /models/deepseek-config/imatrix.dat  --token-embedding-type q8_0 /storage/DeepSeek-R1-GGUF/unsloth_DeepSeek-R1-BF16-256x21B-F16-00001-of-00059.gguf /models/DeepSeek-R1-GGUF/DeepSeek-R1-GGUF-IQ4_KSS-v2.gguf IQ4_KSS 64
```

The `imatrix.dat` is from https://huggingface.co/mradermacher/DeepSeek-R1-i1-GGUF from @schmorp.

---

I then decided to rebuild with `GGML_CUDA_FORCE_MMQ` / `LLAMA_CUDA_FORCE_MMQ` set, and then run to see if that would resolve with a higher `-ub` size. 

Unfortunately, no - produced NaNs.

Hopefully this is enough information for you to be able to possibly see what the issue is!

### Name and Version

main: build = 0 (unknown)
main: built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu
main: seed  = 3407

---

#### 💬 Conversation

👤 **davidsyoung** commented the **2025-03-07** at **19:50:29**:<br>

UPDATE: Can confirm it also works with `-ub 64` (still have `GGML_CUDA_FORCE_MMQ` enabled). Will continue to try different settings to narrow it down.

UPDATE 2: Can confirm also works with `-ub 128`, without `-amb`. Trying `-ub 512` now without `-amb`.

UPDATE 3: Doesn't work with `-ub 512`, without `-amb`. Trialling `-ub 256`.

UPDATE 4: Doesn't work with `-ub 256`.

Going back to `-ub 128`.

---

UPDATE 5: Started producing NaNs after 8 chunks at 4096 ctx with `-ub 64`.

Not too sure what this means.

```
perplexity: tokenizing the input ..
perplexity: tokenization took 1152.88 ms
perplexity: calculating perplexity over 70 chunks, n_ctx=4096, batch_size=2048, n_seq=1
perplexity: 60.48 seconds per pass - ETA 1 hours 10.55 minutes
[1]1.0918,[2]1.8117,[3]1.9102,[4]2.1285,[5]2.4849,[6]2.5949,[7]2.7723,[8]3.0115,[9]nan,[10]nan,[11]nan,[12]nan,^C^C
```

UPDATE 6: Tried removing `-fa` and it went a little longer but started producing NaNs again:

```
[1]1.5066,[2]1.2795,[3]1.2315,[4]1.6830,[5]1.7410,[6]1.7140,[7]1.8198,[8]1.9421,[9]2.1296,[10]2.3186,[11]2.4416,[12]2.3207,[13]2.4436,[14]2.5401,[15]2.6685,[16]2.7862,[17]2.7704,[18]2.8288,[19]nan,[20]nan,[21]nan,[22]nan,[23]nan,[24]nan,[25]nan
```


---

UPDATE 7:

`-ub 32` completed in the end. I did run this with `-mla 2`, but I don’t believe that was the solution given it failed above with a higher `-ub`.

```
perplexity: tokenizing the input ..
perplexity: tokenization took 1202.23 ms
perplexity: calculating perplexity over 140 chunks, n_ctx=2048, batch_size=2048, n_seq=1
perplexity: 65.08 seconds per pass - ETA 2 hours 31.85 minutes
[1]1.5091,[2]1.2810,[3]1.2387,[4]1.6906,[5]1.7516,[6]1.7225,[7]1.8288,[8]1.9517,[9]2.1405,[10]2.3297,[11]2.4523,[12]2.3328,[13]2.4554,[14]2.5518,[15]2.6808,[16]2.8002,[17]2.7836,[18]2.8415,[19]2.7820,[20]2.7049,[21]2.6368,[22]2.5643,[23]2.4760,[24]2.4234,[25]2.3868,[26]2.4654,[27]2.5406,[28]2.5428,[29]2.4865,[30]2.4271,[31]2.3721,[32]2.3269,[33]2.3127,[34]2.3525,[35]2.3884,[36]2.3891,[37]2.3959,[38]2.3918,[39]2.4025,[40]2.4321,[41]2.4859,[42]2.5627,[43]2.5913,[44]2.5467,[45]2.5188,[46]2.5701,[47]2.6229,[48]2.6445,[49]2.6922,[50]2.7100,[51]2.7326,[52]2.7553,[53]2.7585,[54]2.7733,[55]2.7738,[56]2.7869,[57]2.7900,[58]2.8088,[59]2.8216,[60]2.8548,[61]2.8961,[62]2.8999,[63]2.9024,[64]2.9205,[65]2.9293,[66]2.9411,[67]2.9497,[68]2.9344,[69]2.8968,[70]2.9245,[71]2.9534,[72]2.9626,[73]2.9373,[74]2.9410,[75]2.9588,[76]2.9646,[77]2.9660,[78]2.9710,[79]2.9800,[80]2.9861,[81]2.9895,[82]2.9952,[83]3.0084,[84]3.0102,[85]3.0235,[86]3.0479,[87]3.0258,[88]3.0555,[89]3.0848,[90]3.1080,[91]3.1284,[92]3.1570,[93]3.1884,[94]3.2194,[95]3.2202,[96]3.2380,[97]3.2502,[98]3.2188,[99]3.1830,[100]3.1477,[101]3.1139,[102]3.0818,[103]3.0735,[104]3.0623,[105]3.0637,[106]3.0649,[107]3.0674,[108]3.0695,[109]3.0481,[110]3.0463,[111]3.0431,[112]3.0536,[113]3.0666,[114]3.0722,[115]3.0821,[116]3.1002,[117]3.0995,[118]3.0992,[119]3.0996,[120]3.1027,[121]3.1039,[122]3.1167,[123]3.1333,[124]3.1369,[125]3.1438,[126]3.1436,[127]3.1524,[128]3.1348,[129]3.1284,[130]3.1338,[131]3.1426,[132]3.1261,[133]3.1132,[134]3.1202,[135]3.1335,[136]3.1231,[137]3.1000,[138]3.0781,[139]3.0815,[140]3.1010,
Final estimate: PPL = 3.1010 +/- 0.01626

llama_print_timings:        load time =  726885.88 ms
llama_print_timings:      sample time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings: prompt eval time = 8987465.73 ms / 286720 tokens (   31.35 ms per token,    31.90 tokens per second)
llama_print_timings:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings:       total time = 8991556.33 ms / 286721 tokens
```

---

👤 **davidsyoung** commented the **2025-03-07** at **19:50:29**:<br>

UPDATE: Can confirm it also works with `-ub 64` (still have `GGML_CUDA_FORCE_MMQ` enabled). Will continue to try different settings to narrow it down.

---

👤 **ikawrakow** commented the **2025-03-10** at **14:06:25**:<br>

I think there are precision issues in the MLA way of computing attention.

I wanted to calculate an imatrix with MLA enabled to test PR #250. I used the `fp16` version of DeepSeek-Lite and, boom, I got NaNs. I suspected the `K*Q` matrix multiplication as there have been precision issues with that for other models in the past (e.g., Phi-2 and Phi-3), so I set the precision of `K*Q` to `fp32`. The NaNs went away, but perplexity was much too high. 
This is only on CUDA. On the CPU the MLA imatrix calculation is perfectly fine. It is also OK if I use a `bf16` DeepSeek-Lite model on CUDA and CPU. If I convert DeepSeek-Lite directly from safetensors to `Q8_0` using `convert_hf_to_gguf.py`, the imatrix calculation with a chunk size of 2048 looks fine at first, but then I get NaNs at the 21st chunk. Additional strange observations:
* FlashMLA (`mla=2, fa=1`) works just fine with the `fp16` model.
* If I set the precision of all MLA matrix multiplications to `fp32`, I still get unreasonably high perplexity (around 50 instead of around 6). I verified that in the CUDA implementation tensors are indeed converted to `fp32` before performing the matrix multiplication. This would imply that information has been lost before the conversion to `fp32` (due to the limited range of `fp16`), either in the model weights or in the KV cache stored as `fp16`. But if that were true, then FlashMLA shouldn't be working either. But it does.    

So, it looks quite a bit more tricky than just setting the `K*Q` precision for `fp32`.

To come back to your use case, `IQ4_K` and `IQ4_KSS` don't have quantized matrix multiplications implemented (known as MMQ kernels). Hence, for these quantization types (and also `IQ2_KS, IQ2_K, IQ3_K, IQ4_KS, IQ5_K, IQ6_K`), matrix multiplications are done by first converting the quantized tensors to `fp16` and then using cuBLAS GEMM. So, given the observed numerical instabilities, these cannot be used for any attention tensors.

---

👤 **davidsyoung** commented the **2025-03-10** at **14:22:33**:<br>

Thank you for looking into this @ikawrakow - I have a quantisation of DeepSeek-R1 currently 50% complete with all attention tensors (as per your recommendations) set to q8_0 precision. 

Once it's complete, I'll run perplexity and report back and see if I get any NaNs.

---

👤 **davidsyoung** commented the **2025-03-10** at **16:55:16**:<br>

@ikawrakow Tried a new quant with all attention params being set to q8_0, no luck unfortunately. Starts producing NaNs at 10 chunks with `-ub 512` with `-fmoe -mla 2 -fa` with latest PR. Will try to run with some other combinations. Any suggestions to help you debug?

---

👤 **ikawrakow** commented the **2025-03-10** at **16:58:29**:<br>

I'm running out of ideas. In case you have it, can you post the quantization log?

---

👤 **davidsyoung** commented the **2025-03-10** at **17:02:08**:<br>

> I'm running out of ideas. In case you have it, can you post the quantization log?

Of course:

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
  --custom-q "\.attn_.*\.weight=q8_0" \
  --custom-q "\.ffn_.*_shexp\.weight=q5_K,output\.weight=q8_0" \
  --custom-q "blk\.3\.ffn_down_exps\.weight=q5_K,blk\.4\.ffn_down_exps\.weight=q5_K,blk\.5\.ffn_down_exps\.weight=q5_K,blk\.3\.ffn_up_exps\.weight=iq4_k,blk\.3\.ffn_gate_exps\.weight=iq4_k,blk\.4\.ffn_up_exps\.weight=iq4_k,blk\.4\.ffn_gate_exps\.weight=iq4_k,blk\.5\.ffn_up_exps\.weight=iq4_k,blk\.5\.ffn_gate_exps\.weight=iq4_k" \=17.0 ms
  --custom-q "blk\.6\.ffn_down_exps\.weight=q5_K,blk\.7\.ffn_down_exps\.weight=q5_K,blk\.8\.ffn_down_exps\.weight=q5_K,blk\.6\.ffn_up_exps\.weight=iq4_k,blk\.6\.ffn_gate_exps\.weight=iq4_k,blk\.7\.ffn_up_exps\.weight=iq4_k,blk\.7\.ffn_gate_exps\.weight=iq4_k,blk\.8\.ffn_up_exps\.weight=iq4_k,blk\.8\.ffn_gate_exps\.weight=iq4_k" \=15.0 ms
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
  /models/DeepSeek-R1-GGUF/DeepSeek-R1-GGUF-IQ4_K__IQ3_S_Q8.gguf \
  q8_0 64
Adding custom rule \.attn_.*\.weight -> q8_0
Adding custom rule \.ffn_.*_shexp\.weight -> q5_K
Adding custom rule output\.weight -> q8_0
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
Adding custom rule blk\.9\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.9\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.10\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.10\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.11\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.11\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.12\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.12\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.13\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.14\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.15\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.16\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.13\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.13\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.14\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.14\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.15\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.15\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.16\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.16\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.17\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.18\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.19\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.20\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.17\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.17\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.18\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.18\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.19\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.19\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.20\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.20\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.21\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.22\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.23\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.24\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.21\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.21\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.22\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.22\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.23\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.23\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.24\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.24\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.25\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.26\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.27\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.28\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.25\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.25\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.26\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.26\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.27\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.27\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.28\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.28\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.29\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.30\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.31\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.32\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.29\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.29\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.30\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.30\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.31\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.31\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.32\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.32\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.33\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.34\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.35\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.36\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.33\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.33\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.34\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.34\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.35\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.35\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.36\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.36\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.37\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.38\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.39\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.40\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.37\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.37\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.38\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.38\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.39\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.39\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.40\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.40\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.41\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.42\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.43\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.44\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.41\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.41\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.42\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.42\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.43\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.43\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.44\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.44\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.45\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.46\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.47\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.48\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.45\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.45\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.46\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.46\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.47\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.47\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.48\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.48\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.49\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.50\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.51\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.52\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.49\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.49\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.50\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.50\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.51\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.51\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.52\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.52\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.53\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.54\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.55\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.56\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.53\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.53\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.54\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.54\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.55\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.55\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.56\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.56\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.57\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.58\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.59\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.60\.ffn_down_exps\.weight -> iq4_k
Adding custom rule blk\.57\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.57\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.58\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.58\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.59\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.59\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.60\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.60\.ffn_gate_exps\.weight -> iq3_s
load_imatrix: imatrix dataset='imatrix-training-full-3'
load_imatrix: loaded 720 importance matrix entries from /models/deepseek-config/imatrix.dat computed on 315 chunks
prepare_imatrix: have 720 importance matrix entries
main: build = 0 (unknown)
main: built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu
main: quantizing '/storage/DeepSeek-R1-GGUF/unsloth_DeepSeek-R1-BF16-256x21B-F16-00001-of-00059.gguf' to '/models/DeepSeek-R1-GGUF/DeepSeek-R1-GGUF-IQ4_K__IQ3_S_Q8.gguf' as Q8_0 using 64 threads
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
llama_model_loader: - kv  40:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<�...
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
[   3/1147]                blk.0.ffn_down.weight - [18432,  7168,     1,     1], type =    f16, converting to q8_0 .. size =   252.00 MiB ->   133.88 MiB
[   4/1147]                blk.0.ffn_gate.weight - [ 7168, 18432,     1,     1], type =    f16, converting to q8_0 .. size =   252.00 MiB ->   133.88 MiB
[   5/1147]                  blk.0.ffn_up.weight - [ 7168, 18432,     1,     1], type =    f16, converting to q8_0 .. size =   252.00 MiB ->   133.88 MiB
[   6/1147]                blk.0.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[   7/1147]          blk.0.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[   8/1147]           blk.0.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.0.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[   9/1147]               blk.0.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.0.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[  10/1147]                blk.0.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.0.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.0.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  11/1147]                blk.0.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.0.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.0.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  12/1147]             blk.0.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.0.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[  13/1147]           blk.0.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  14/1147]                blk.0.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.0.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[  15/1147]                blk.0.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.0.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[  16/1147]               blk.1.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  17/1147]                blk.1.ffn_down.weight - [18432,  7168,     1,     1], type =    f16, converting to q8_0 .. size =   252.00 MiB ->   133.88 MiB
[  18/1147]                blk.1.ffn_gate.weight - [ 7168, 18432,     1,     1], type =    f16, converting to q8_0 .. size =   252.00 MiB ->   133.88 MiB
[  19/1147]                  blk.1.ffn_up.weight - [ 7168, 18432,     1,     1], type =    f16, converting to q8_0 .. size =   252.00 MiB ->   133.88 MiB
[  20/1147]                blk.1.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  21/1147]          blk.1.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  22/1147]           blk.1.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.1.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[  23/1147]               blk.1.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.1.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[  24/1147]                blk.1.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.1.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.1.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  25/1147]                blk.1.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.1.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.1.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  26/1147]             blk.1.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.1.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[  27/1147]           blk.1.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  28/1147]                blk.1.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.1.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[  29/1147]                blk.1.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.1.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[  30/1147]               blk.2.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  31/1147]                blk.2.ffn_down.weight - [18432,  7168,     1,     1], type =    f16, converting to q8_0 .. size =   252.00 MiB ->   133.88 MiB
[  32/1147]                blk.2.ffn_gate.weight - [ 7168, 18432,     1,     1], type =    f16, converting to q8_0 .. size =   252.00 MiB ->   133.88 MiB
[  33/1147]                  blk.2.ffn_up.weight - [ 7168, 18432,     1,     1], type =    f16, converting to q8_0 .. size =   252.00 MiB ->   133.88 MiB
[  34/1147]                blk.2.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  35/1147]          blk.2.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  36/1147]           blk.2.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.2.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[  37/1147]               blk.2.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.2.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[  38/1147]                blk.2.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.2.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.2.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  39/1147]                blk.2.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.2.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.2.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  40/1147]             blk.2.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.2.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[  41/1147]           blk.2.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  42/1147]                blk.2.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.2.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[  43/1147]                blk.2.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.2.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[  44/1147]               blk.3.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[  45/1147]            blk.3.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[  46/1147]          blk.3.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.3.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[  47/1147]          blk.3.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.3.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[  48/1147]            blk.3.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.3.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[  49/1147]          blk.3.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  50/1147]           blk.3.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.3.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[  51/1147]               blk.3.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.3.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[  52/1147]                blk.3.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.3.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.3.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  53/1147]                blk.3.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.3.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.3.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  54/1147]             blk.3.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.3.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[  55/1147]           blk.3.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  56/1147]                blk.3.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.3.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[  57/1147]                blk.3.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.3.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
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
[  65/1147]          blk.4.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.4.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[  66/1147]          blk.4.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.4.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[  67/1147]            blk.4.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.4.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[  68/1147]          blk.4.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  69/1147]           blk.4.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.4.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[  70/1147]               blk.4.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.4.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[  71/1147]                blk.4.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.4.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.4.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  72/1147]                blk.4.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.4.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.4.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  73/1147]             blk.4.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.4.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[  74/1147]           blk.4.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  75/1147]                blk.4.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.4.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[  76/1147]                blk.4.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.4.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[  77/1147]               blk.4.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  78/1147]           blk.4.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type q5_K for tensor blk.4.ffn_down_exps.weight
converting to q5_K .. size =  7168.00 MiB ->  2464.00 MiB
[  79/1147]           blk.4.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.4.ffn_gate_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[  80/1147]             blk.4.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.4.ffn_up_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[  81/1147]                blk.4.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  82/1147]          blk.5.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  83/1147]           blk.5.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.5.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[  84/1147]               blk.5.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.5.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[  85/1147]                blk.5.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.5.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.5.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  86/1147]                blk.5.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.5.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.5.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  87/1147]             blk.5.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.5.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[  88/1147]           blk.5.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  89/1147]                blk.5.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.5.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[  90/1147]                blk.5.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.5.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[  91/1147]               blk.5.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[  92/1147]            blk.5.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[  93/1147]          blk.5.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.5.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[  94/1147]          blk.5.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.5.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[  95/1147]            blk.5.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.5.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
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
[ 103/1147]          blk.6.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.6.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 104/1147]          blk.6.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.6.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 105/1147]            blk.6.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.6.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 106/1147]          blk.6.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 107/1147]           blk.6.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.6.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 108/1147]               blk.6.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.6.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 109/1147]                blk.6.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.6.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.6.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 110/1147]                blk.6.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.6.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.6.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 111/1147]             blk.6.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.6.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 112/1147]           blk.6.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 113/1147]                blk.6.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.6.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 114/1147]                blk.6.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.6.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
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
[ 122/1147]          blk.7.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.7.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 123/1147]          blk.7.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.7.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 124/1147]            blk.7.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.7.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 125/1147]          blk.7.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 126/1147]           blk.7.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.7.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 127/1147]               blk.7.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.7.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 128/1147]                blk.7.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.7.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.7.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 129/1147]                blk.7.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.7.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.7.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 130/1147]             blk.7.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.7.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 131/1147]           blk.7.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 132/1147]                blk.7.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.7.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 133/1147]                blk.7.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.7.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
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
[ 141/1147]          blk.8.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.8.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 142/1147]          blk.8.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.8.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 143/1147]            blk.8.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.8.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 144/1147]          blk.8.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 145/1147]           blk.8.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.8.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 146/1147]               blk.8.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.8.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 147/1147]                blk.8.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.8.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.8.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 148/1147]                blk.8.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.8.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.8.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 149/1147]             blk.8.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.8.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 150/1147]           blk.8.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 151/1147]                blk.8.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.8.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 152/1147]                blk.8.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.8.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
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
[ 160/1147]          blk.9.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.9.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 161/1147]          blk.9.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.9.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 162/1147]            blk.9.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.9.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 163/1147]          blk.9.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 164/1147]           blk.9.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.9.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 165/1147]               blk.9.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.9.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 166/1147]                blk.9.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.9.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.9.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 167/1147]                blk.9.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.9.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.9.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 168/1147]             blk.9.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.9.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 169/1147]           blk.9.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 170/1147]                blk.9.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.9.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 171/1147]                blk.9.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.9.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 172/1147]              blk.10.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 173/1147]           blk.10.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 174/1147]         blk.10.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.10.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 175/1147]         blk.10.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.10.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 176/1147]           blk.10.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.10.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 177/1147]         blk.10.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 178/1147]          blk.10.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.10.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 179/1147]              blk.10.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.10.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 180/1147]               blk.10.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.10.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.10.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 181/1147]               blk.10.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.10.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.10.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 182/1147]            blk.10.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.10.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 183/1147]          blk.10.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 184/1147]               blk.10.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.10.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 185/1147]               blk.10.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.10.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 186/1147]               blk.9.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 187/1147]           blk.9.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.9.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 188/1147]           blk.9.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.9.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 189/1147]             blk.9.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.9.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 190/1147]                blk.9.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 191/1147]              blk.10.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 192/1147]          blk.10.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.10.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 193/1147]          blk.10.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.10.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 194/1147]            blk.10.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.10.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 195/1147]               blk.10.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 196/1147]              blk.11.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 197/1147]           blk.11.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 198/1147]         blk.11.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.11.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 199/1147]         blk.11.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.11.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 200/1147]           blk.11.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.11.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 201/1147]         blk.11.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 202/1147]          blk.11.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.11.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 203/1147]              blk.11.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.11.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 204/1147]               blk.11.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.11.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.11.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 205/1147]               blk.11.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.11.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.11.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 206/1147]            blk.11.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.11.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 207/1147]          blk.11.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 208/1147]               blk.11.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.11.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 209/1147]               blk.11.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.11.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 210/1147]              blk.11.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 211/1147]          blk.11.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.11.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 212/1147]          blk.11.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.11.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 213/1147]            blk.11.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.11.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 214/1147]               blk.11.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 215/1147]              blk.12.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 216/1147]           blk.12.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 217/1147]         blk.12.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.12.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 218/1147]         blk.12.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.12.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 219/1147]           blk.12.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.12.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 220/1147]         blk.12.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 221/1147]          blk.12.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.12.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 222/1147]              blk.12.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.12.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 223/1147]               blk.12.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.12.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.12.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 224/1147]               blk.12.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.12.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.12.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 225/1147]            blk.12.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.12.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 226/1147]          blk.12.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 227/1147]               blk.12.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.12.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 228/1147]               blk.12.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.12.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 229/1147]              blk.12.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 230/1147]          blk.12.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.12.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 231/1147]          blk.12.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.12.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 232/1147]            blk.12.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.12.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 233/1147]               blk.12.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 234/1147]              blk.13.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 235/1147]           blk.13.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 236/1147]         blk.13.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.13.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 237/1147]         blk.13.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.13.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 238/1147]           blk.13.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.13.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 239/1147]         blk.13.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 240/1147]          blk.13.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.13.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 241/1147]              blk.13.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.13.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 242/1147]               blk.13.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.13.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.13.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 243/1147]               blk.13.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.13.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.13.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 244/1147]            blk.13.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.13.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 245/1147]          blk.13.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 246/1147]               blk.13.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.13.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 247/1147]               blk.13.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.13.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 248/1147]              blk.13.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 249/1147]          blk.13.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.13.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 250/1147]          blk.13.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.13.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 251/1147]            blk.13.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.13.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 252/1147]               blk.13.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 253/1147]              blk.14.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 254/1147]           blk.14.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 255/1147]         blk.14.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.14.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 256/1147]         blk.14.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.14.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 257/1147]           blk.14.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.14.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 258/1147]         blk.14.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 259/1147]          blk.14.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.14.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 260/1147]              blk.14.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.14.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 261/1147]               blk.14.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.14.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.14.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 262/1147]               blk.14.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.14.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.14.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 263/1147]            blk.14.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.14.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 264/1147]          blk.14.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 265/1147]               blk.14.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.14.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 266/1147]               blk.14.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.14.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 267/1147]              blk.14.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 268/1147]          blk.14.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.14.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 269/1147]          blk.14.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.14.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 270/1147]            blk.14.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.14.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 271/1147]               blk.14.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 272/1147]              blk.15.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 273/1147]           blk.15.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 274/1147]         blk.15.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.15.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 275/1147]         blk.15.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.15.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 276/1147]           blk.15.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.15.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 277/1147]         blk.15.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 278/1147]          blk.15.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.15.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 279/1147]              blk.15.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.15.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 280/1147]               blk.15.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.15.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.15.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 281/1147]               blk.15.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.15.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.15.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 282/1147]            blk.15.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.15.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 283/1147]          blk.15.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 284/1147]               blk.15.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.15.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 285/1147]               blk.15.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.15.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 286/1147]              blk.15.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 287/1147]          blk.15.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.15.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 288/1147]          blk.15.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.15.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 289/1147]            blk.15.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.15.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 290/1147]               blk.15.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 291/1147]              blk.16.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 292/1147]           blk.16.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 293/1147]         blk.16.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.16.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 294/1147]         blk.16.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.16.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 295/1147]           blk.16.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.16.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 296/1147]         blk.16.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 297/1147]          blk.16.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.16.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 298/1147]              blk.16.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.16.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 299/1147]               blk.16.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.16.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.16.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 300/1147]               blk.16.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.16.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.16.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 301/1147]            blk.16.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.16.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 302/1147]          blk.16.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 303/1147]               blk.16.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.16.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 304/1147]               blk.16.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.16.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 305/1147]              blk.16.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 306/1147]          blk.16.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.16.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 307/1147]          blk.16.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.16.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 308/1147]            blk.16.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.16.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 309/1147]               blk.16.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 310/1147]              blk.17.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 311/1147]           blk.17.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 312/1147]         blk.17.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.17.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 313/1147]         blk.17.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.17.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 314/1147]           blk.17.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.17.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 315/1147]         blk.17.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 316/1147]          blk.17.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.17.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 317/1147]              blk.17.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.17.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 318/1147]               blk.17.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.17.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.17.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 319/1147]               blk.17.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.17.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.17.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 320/1147]            blk.17.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.17.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 321/1147]          blk.17.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 322/1147]               blk.17.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.17.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 323/1147]               blk.17.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.17.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 324/1147]              blk.17.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 325/1147]          blk.17.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.17.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 326/1147]          blk.17.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.17.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 327/1147]            blk.17.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.17.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 328/1147]               blk.17.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 329/1147]              blk.18.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 330/1147]           blk.18.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 331/1147]         blk.18.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.18.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 332/1147]         blk.18.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.18.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 333/1147]           blk.18.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.18.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 334/1147]         blk.18.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 335/1147]          blk.18.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.18.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 336/1147]              blk.18.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.18.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 337/1147]               blk.18.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.18.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.18.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 338/1147]               blk.18.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.18.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.18.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 339/1147]            blk.18.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.18.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 340/1147]          blk.18.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 341/1147]               blk.18.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.18.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 342/1147]               blk.18.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.18.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 343/1147]              blk.18.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 344/1147]          blk.18.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.18.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 345/1147]          blk.18.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.18.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 346/1147]            blk.18.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.18.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 347/1147]               blk.18.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 348/1147]              blk.19.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 349/1147]           blk.19.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 350/1147]         blk.19.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.19.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 351/1147]         blk.19.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.19.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 352/1147]           blk.19.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.19.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 353/1147]         blk.19.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 354/1147]          blk.19.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.19.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 355/1147]              blk.19.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.19.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 356/1147]               blk.19.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.19.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.19.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 357/1147]               blk.19.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.19.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.19.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 358/1147]            blk.19.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.19.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 359/1147]          blk.19.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 360/1147]               blk.19.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.19.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 361/1147]               blk.19.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.19.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 362/1147]              blk.19.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 363/1147]          blk.19.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.19.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 364/1147]          blk.19.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.19.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 365/1147]            blk.19.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.19.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 366/1147]               blk.19.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 367/1147]              blk.20.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 368/1147]           blk.20.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 369/1147]         blk.20.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.20.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 370/1147]         blk.20.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.20.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 371/1147]           blk.20.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.20.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 372/1147]         blk.20.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 373/1147]          blk.20.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.20.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 374/1147]              blk.20.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.20.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 375/1147]               blk.20.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.20.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.20.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 376/1147]               blk.20.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.20.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.20.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 377/1147]            blk.20.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.20.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 378/1147]          blk.20.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 379/1147]               blk.20.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.20.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 380/1147]               blk.20.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.20.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 381/1147]              blk.20.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 382/1147]          blk.20.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.20.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 383/1147]          blk.20.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.20.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 384/1147]            blk.20.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.20.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 385/1147]               blk.20.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 386/1147]              blk.21.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 387/1147]           blk.21.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 388/1147]         blk.21.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.21.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 389/1147]         blk.21.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.21.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 390/1147]           blk.21.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.21.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 391/1147]         blk.21.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 392/1147]          blk.21.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.21.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 393/1147]              blk.21.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.21.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 394/1147]               blk.21.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.21.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.21.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 395/1147]               blk.21.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.21.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.21.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 396/1147]            blk.21.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.21.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 397/1147]          blk.21.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 398/1147]               blk.21.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.21.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 399/1147]               blk.21.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.21.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 400/1147]              blk.21.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 401/1147]          blk.21.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.21.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 402/1147]          blk.21.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.21.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 403/1147]            blk.21.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.21.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 404/1147]               blk.21.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 405/1147]              blk.22.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 406/1147]           blk.22.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 407/1147]         blk.22.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.22.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 408/1147]         blk.22.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.22.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 409/1147]           blk.22.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.22.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 410/1147]         blk.22.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 411/1147]          blk.22.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.22.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 412/1147]              blk.22.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.22.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 413/1147]               blk.22.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.22.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.22.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 414/1147]               blk.22.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.22.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.22.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 415/1147]            blk.22.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.22.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 416/1147]          blk.22.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 417/1147]               blk.22.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.22.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 418/1147]               blk.22.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.22.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 419/1147]              blk.22.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 420/1147]          blk.22.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.22.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 421/1147]          blk.22.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.22.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 422/1147]            blk.22.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.22.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 423/1147]               blk.22.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 424/1147]              blk.23.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 425/1147]           blk.23.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 426/1147]         blk.23.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.23.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 427/1147]         blk.23.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.23.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 428/1147]           blk.23.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.23.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 429/1147]         blk.23.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 430/1147]          blk.23.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.23.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 431/1147]              blk.23.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.23.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 432/1147]               blk.23.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.23.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.23.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 433/1147]               blk.23.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.23.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.23.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 434/1147]            blk.23.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.23.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 435/1147]          blk.23.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 436/1147]               blk.23.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.23.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 437/1147]               blk.23.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.23.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 438/1147]              blk.23.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 439/1147]          blk.23.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.23.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 440/1147]          blk.23.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.23.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 441/1147]            blk.23.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.23.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 442/1147]               blk.23.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 443/1147]              blk.24.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 444/1147]           blk.24.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 445/1147]         blk.24.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.24.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 446/1147]         blk.24.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.24.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 447/1147]           blk.24.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.24.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 448/1147]         blk.24.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 449/1147]          blk.24.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.24.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 450/1147]              blk.24.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.24.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 451/1147]               blk.24.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.24.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.24.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 452/1147]               blk.24.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.24.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.24.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 453/1147]            blk.24.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.24.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 454/1147]          blk.24.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 455/1147]               blk.24.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.24.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 456/1147]               blk.24.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.24.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 457/1147]              blk.24.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 458/1147]          blk.24.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.24.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 459/1147]          blk.24.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.24.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 460/1147]            blk.24.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.24.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 461/1147]               blk.24.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 462/1147]              blk.25.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 463/1147]           blk.25.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 464/1147]         blk.25.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.25.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 465/1147]         blk.25.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.25.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 466/1147]           blk.25.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.25.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 467/1147]         blk.25.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 468/1147]          blk.25.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.25.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 469/1147]              blk.25.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.25.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 470/1147]               blk.25.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.25.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.25.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 471/1147]               blk.25.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.25.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.25.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 472/1147]            blk.25.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.25.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 473/1147]          blk.25.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 474/1147]               blk.25.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.25.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 475/1147]               blk.25.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.25.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 476/1147]              blk.25.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 477/1147]          blk.25.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.25.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 478/1147]          blk.25.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.25.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 479/1147]            blk.25.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.25.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 480/1147]               blk.25.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 481/1147]              blk.26.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 482/1147]           blk.26.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 483/1147]         blk.26.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.26.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 484/1147]         blk.26.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.26.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 485/1147]           blk.26.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.26.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 486/1147]         blk.26.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 487/1147]          blk.26.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.26.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 488/1147]              blk.26.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.26.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 489/1147]               blk.26.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.26.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.26.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 490/1147]               blk.26.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.26.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.26.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 491/1147]            blk.26.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.26.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 492/1147]          blk.26.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 493/1147]               blk.26.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.26.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 494/1147]               blk.26.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.26.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 495/1147]              blk.26.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 496/1147]          blk.26.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.26.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 497/1147]          blk.26.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.26.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 498/1147]            blk.26.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.26.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 499/1147]               blk.26.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 500/1147]              blk.27.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 501/1147]           blk.27.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 502/1147]         blk.27.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.27.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 503/1147]         blk.27.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.27.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 504/1147]           blk.27.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.27.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 505/1147]         blk.27.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 506/1147]          blk.27.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.27.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 507/1147]              blk.27.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.27.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 508/1147]               blk.27.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.27.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.27.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 509/1147]               blk.27.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.27.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.27.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 510/1147]            blk.27.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.27.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 511/1147]          blk.27.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 512/1147]               blk.27.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.27.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 513/1147]               blk.27.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.27.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 514/1147]              blk.27.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 515/1147]          blk.27.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.27.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 516/1147]          blk.27.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.27.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 517/1147]            blk.27.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.27.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 518/1147]               blk.27.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 519/1147]              blk.28.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 520/1147]           blk.28.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 521/1147]         blk.28.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.28.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 522/1147]         blk.28.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.28.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 523/1147]           blk.28.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.28.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 524/1147]         blk.28.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 525/1147]          blk.28.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.28.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 526/1147]              blk.28.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.28.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 527/1147]               blk.28.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.28.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.28.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 528/1147]               blk.28.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.28.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.28.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 529/1147]            blk.28.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.28.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 530/1147]          blk.28.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 531/1147]               blk.28.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.28.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 532/1147]               blk.28.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.28.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 533/1147]              blk.28.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 534/1147]          blk.28.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.28.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 535/1147]          blk.28.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.28.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 536/1147]            blk.28.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.28.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 537/1147]               blk.28.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 538/1147]              blk.29.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 539/1147]           blk.29.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 540/1147]         blk.29.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.29.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 541/1147]         blk.29.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.29.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 542/1147]           blk.29.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.29.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 543/1147]         blk.29.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 544/1147]          blk.29.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.29.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 545/1147]              blk.29.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.29.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 546/1147]               blk.29.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.29.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.29.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 547/1147]               blk.29.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.29.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.29.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 548/1147]            blk.29.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.29.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 549/1147]          blk.29.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 550/1147]               blk.29.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.29.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 551/1147]               blk.29.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.29.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 552/1147]              blk.29.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 553/1147]          blk.29.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.29.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 554/1147]          blk.29.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.29.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 555/1147]            blk.29.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.29.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 556/1147]               blk.29.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 557/1147]              blk.30.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 558/1147]           blk.30.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 559/1147]         blk.30.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.30.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 560/1147]         blk.30.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.30.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 561/1147]           blk.30.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.30.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 562/1147]         blk.30.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 563/1147]          blk.30.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.30.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 564/1147]              blk.30.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.30.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 565/1147]               blk.30.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.30.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.30.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 566/1147]               blk.30.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.30.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.30.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 567/1147]            blk.30.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.30.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 568/1147]          blk.30.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 569/1147]               blk.30.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.30.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 570/1147]               blk.30.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.30.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 571/1147]              blk.30.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 572/1147]          blk.30.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.30.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 573/1147]          blk.30.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.30.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 574/1147]            blk.30.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.30.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 575/1147]               blk.30.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 576/1147]              blk.31.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 577/1147]           blk.31.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 578/1147]         blk.31.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.31.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 579/1147]         blk.31.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.31.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 580/1147]           blk.31.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.31.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 581/1147]         blk.31.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 582/1147]          blk.31.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.31.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 583/1147]              blk.31.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.31.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 584/1147]               blk.31.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.31.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.31.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 585/1147]               blk.31.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.31.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.31.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 586/1147]            blk.31.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.31.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 587/1147]          blk.31.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 588/1147]               blk.31.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.31.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 589/1147]               blk.31.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.31.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 590/1147]              blk.31.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 591/1147]          blk.31.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.31.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 592/1147]          blk.31.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.31.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 593/1147]            blk.31.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.31.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 594/1147]               blk.31.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 595/1147]              blk.32.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 596/1147]           blk.32.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 597/1147]         blk.32.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.32.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 598/1147]         blk.32.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.32.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 599/1147]           blk.32.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.32.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 600/1147]         blk.32.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 601/1147]          blk.32.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.32.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 602/1147]              blk.32.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.32.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 603/1147]               blk.32.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.32.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.32.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 604/1147]               blk.32.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.32.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.32.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 605/1147]            blk.32.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.32.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 606/1147]          blk.32.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 607/1147]               blk.32.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.32.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 608/1147]               blk.32.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.32.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 609/1147]              blk.32.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 610/1147]          blk.32.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.32.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 611/1147]          blk.32.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.32.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 612/1147]            blk.32.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.32.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 613/1147]               blk.32.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 614/1147]              blk.33.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 615/1147]           blk.33.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 616/1147]         blk.33.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.33.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 617/1147]         blk.33.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.33.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 618/1147]           blk.33.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.33.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 619/1147]         blk.33.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 620/1147]          blk.33.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.33.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 621/1147]              blk.33.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.33.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 622/1147]               blk.33.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.33.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.33.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 623/1147]               blk.33.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.33.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.33.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 624/1147]            blk.33.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.33.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 625/1147]          blk.33.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 626/1147]               blk.33.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.33.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 627/1147]               blk.33.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.33.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 628/1147]              blk.33.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 629/1147]          blk.33.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.33.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 630/1147]          blk.33.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.33.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 631/1147]            blk.33.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.33.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 632/1147]               blk.33.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 633/1147]              blk.34.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 634/1147]           blk.34.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 635/1147]         blk.34.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.34.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 636/1147]         blk.34.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.34.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 637/1147]           blk.34.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.34.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 638/1147]         blk.34.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 639/1147]          blk.34.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.34.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 640/1147]              blk.34.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.34.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 641/1147]               blk.34.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.34.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.34.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 642/1147]               blk.34.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.34.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.34.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 643/1147]            blk.34.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.34.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 644/1147]          blk.34.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 645/1147]               blk.34.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.34.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 646/1147]               blk.34.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.34.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 647/1147]              blk.34.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 648/1147]          blk.34.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.34.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 649/1147]          blk.34.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.34.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 650/1147]            blk.34.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.34.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 651/1147]               blk.34.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 652/1147]              blk.35.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 653/1147]           blk.35.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 654/1147]         blk.35.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.35.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 655/1147]         blk.35.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.35.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 656/1147]           blk.35.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.35.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 657/1147]         blk.35.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 658/1147]          blk.35.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.35.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 659/1147]              blk.35.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.35.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 660/1147]               blk.35.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.35.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.35.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 661/1147]               blk.35.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.35.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.35.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 662/1147]            blk.35.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.35.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 663/1147]          blk.35.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 664/1147]               blk.35.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.35.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 665/1147]               blk.35.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.35.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 666/1147]              blk.35.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 667/1147]          blk.35.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.35.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 668/1147]          blk.35.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.35.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 669/1147]            blk.35.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.35.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 670/1147]               blk.35.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 671/1147]              blk.36.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 672/1147]           blk.36.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 673/1147]         blk.36.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.36.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 674/1147]         blk.36.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.36.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 675/1147]           blk.36.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.36.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 676/1147]         blk.36.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 677/1147]          blk.36.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.36.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 678/1147]              blk.36.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.36.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 679/1147]               blk.36.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.36.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.36.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 680/1147]               blk.36.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.36.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.36.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 681/1147]            blk.36.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.36.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 682/1147]          blk.36.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 683/1147]               blk.36.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.36.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 684/1147]               blk.36.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.36.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 685/1147]              blk.36.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 686/1147]          blk.36.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.36.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 687/1147]          blk.36.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.36.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 688/1147]            blk.36.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.36.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 689/1147]               blk.36.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 690/1147]              blk.37.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 691/1147]           blk.37.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 692/1147]         blk.37.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.37.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 693/1147]         blk.37.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.37.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 694/1147]           blk.37.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.37.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 695/1147]         blk.37.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 696/1147]          blk.37.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.37.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 697/1147]              blk.37.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.37.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 698/1147]               blk.37.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.37.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.37.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 699/1147]               blk.37.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.37.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.37.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 700/1147]            blk.37.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.37.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 701/1147]          blk.37.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 702/1147]               blk.37.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.37.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 703/1147]               blk.37.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.37.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 704/1147]              blk.37.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 705/1147]          blk.37.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.37.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 706/1147]          blk.37.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.37.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 707/1147]            blk.37.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.37.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 708/1147]               blk.37.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 709/1147]              blk.38.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 710/1147]           blk.38.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 711/1147]         blk.38.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.38.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 712/1147]         blk.38.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.38.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 713/1147]           blk.38.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.38.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 714/1147]         blk.38.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 715/1147]          blk.38.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.38.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 716/1147]              blk.38.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.38.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 717/1147]               blk.38.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.38.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.38.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 718/1147]               blk.38.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.38.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.38.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 719/1147]            blk.38.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.38.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 720/1147]          blk.38.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 721/1147]               blk.38.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.38.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 722/1147]               blk.38.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.38.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 723/1147]              blk.38.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 724/1147]          blk.38.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.38.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 725/1147]          blk.38.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.38.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 726/1147]            blk.38.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.38.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 727/1147]               blk.38.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 728/1147]              blk.39.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 729/1147]           blk.39.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 730/1147]         blk.39.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.39.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 731/1147]         blk.39.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.39.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 732/1147]           blk.39.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.39.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 733/1147]         blk.39.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 734/1147]          blk.39.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.39.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 735/1147]              blk.39.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.39.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 736/1147]               blk.39.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.39.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.39.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 737/1147]               blk.39.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.39.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.39.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 738/1147]            blk.39.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.39.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 739/1147]          blk.39.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 740/1147]               blk.39.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.39.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 741/1147]               blk.39.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.39.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 742/1147]              blk.39.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 743/1147]          blk.39.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.39.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 744/1147]          blk.39.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.39.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 745/1147]            blk.39.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.39.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 746/1147]               blk.39.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 747/1147]              blk.40.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 748/1147]           blk.40.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 749/1147]         blk.40.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.40.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 750/1147]         blk.40.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.40.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 751/1147]           blk.40.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.40.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 752/1147]         blk.40.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 753/1147]          blk.40.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.40.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 754/1147]              blk.40.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.40.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 755/1147]               blk.40.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.40.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.40.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 756/1147]               blk.40.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.40.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.40.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 757/1147]            blk.40.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.40.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 758/1147]          blk.40.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 759/1147]               blk.40.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.40.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 760/1147]               blk.40.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.40.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 761/1147]              blk.40.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 762/1147]          blk.40.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.40.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 763/1147]          blk.40.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.40.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 764/1147]            blk.40.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.40.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 765/1147]               blk.40.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 766/1147]              blk.41.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 767/1147]           blk.41.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 768/1147]         blk.41.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.41.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 769/1147]         blk.41.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.41.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 770/1147]           blk.41.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.41.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 771/1147]         blk.41.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 772/1147]          blk.41.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.41.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 773/1147]              blk.41.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.41.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 774/1147]               blk.41.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.41.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.41.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 775/1147]               blk.41.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.41.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.41.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 776/1147]            blk.41.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.41.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 777/1147]          blk.41.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 778/1147]               blk.41.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.41.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 779/1147]               blk.41.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.41.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 780/1147]              blk.41.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 781/1147]          blk.41.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.41.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 782/1147]          blk.41.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.41.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 783/1147]            blk.41.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.41.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 784/1147]               blk.41.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 785/1147]              blk.42.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 786/1147]           blk.42.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 787/1147]         blk.42.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.42.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 788/1147]         blk.42.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.42.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 789/1147]           blk.42.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.42.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 790/1147]         blk.42.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 791/1147]          blk.42.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.42.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 792/1147]              blk.42.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.42.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 793/1147]               blk.42.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.42.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.42.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 794/1147]               blk.42.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.42.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.42.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 795/1147]            blk.42.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.42.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 796/1147]          blk.42.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 797/1147]               blk.42.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.42.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 798/1147]               blk.42.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.42.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 799/1147]              blk.42.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 800/1147]          blk.42.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.42.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 801/1147]          blk.42.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.42.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 802/1147]            blk.42.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.42.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 803/1147]               blk.42.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 804/1147]              blk.43.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 805/1147]           blk.43.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 806/1147]         blk.43.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.43.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 807/1147]         blk.43.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.43.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 808/1147]           blk.43.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.43.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 809/1147]         blk.43.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 810/1147]          blk.43.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.43.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 811/1147]              blk.43.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.43.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 812/1147]               blk.43.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.43.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.43.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 813/1147]               blk.43.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.43.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.43.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 814/1147]            blk.43.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.43.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 815/1147]          blk.43.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 816/1147]               blk.43.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.43.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 817/1147]               blk.43.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.43.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 818/1147]              blk.43.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 819/1147]          blk.43.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.43.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 820/1147]          blk.43.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.43.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 821/1147]            blk.43.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.43.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 822/1147]               blk.43.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 823/1147]              blk.44.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 824/1147]           blk.44.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 825/1147]         blk.44.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.44.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 826/1147]         blk.44.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.44.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 827/1147]           blk.44.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.44.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 828/1147]         blk.44.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 829/1147]          blk.44.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.44.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 830/1147]              blk.44.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.44.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 831/1147]               blk.44.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.44.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.44.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 832/1147]               blk.44.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.44.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.44.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 833/1147]            blk.44.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.44.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 834/1147]          blk.44.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 835/1147]               blk.44.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.44.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 836/1147]               blk.44.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.44.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 837/1147]              blk.44.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 838/1147]          blk.44.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.44.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 839/1147]          blk.44.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.44.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 840/1147]            blk.44.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.44.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 841/1147]               blk.44.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 842/1147]              blk.45.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 843/1147]           blk.45.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 844/1147]         blk.45.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.45.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 845/1147]         blk.45.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.45.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 846/1147]           blk.45.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.45.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 847/1147]         blk.45.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 848/1147]          blk.45.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.45.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 849/1147]              blk.45.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.45.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 850/1147]               blk.45.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.45.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.45.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 851/1147]               blk.45.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.45.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.45.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 852/1147]            blk.45.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.45.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 853/1147]          blk.45.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 854/1147]               blk.45.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.45.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 855/1147]               blk.45.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.45.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 856/1147]              blk.45.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 857/1147]          blk.45.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.45.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 858/1147]          blk.45.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.45.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 859/1147]            blk.45.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.45.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 860/1147]               blk.45.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 861/1147]              blk.46.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 862/1147]           blk.46.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 863/1147]         blk.46.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.46.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 864/1147]         blk.46.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.46.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 865/1147]           blk.46.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.46.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 866/1147]         blk.46.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 867/1147]          blk.46.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.46.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 868/1147]              blk.46.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.46.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 869/1147]               blk.46.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.46.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.46.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 870/1147]               blk.46.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.46.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.46.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 871/1147]            blk.46.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.46.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 872/1147]          blk.46.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 873/1147]               blk.46.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.46.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 874/1147]               blk.46.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.46.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 875/1147]              blk.46.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 876/1147]          blk.46.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.46.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 877/1147]          blk.46.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.46.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 878/1147]            blk.46.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.46.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 879/1147]               blk.46.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 880/1147]              blk.47.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 881/1147]           blk.47.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 882/1147]         blk.47.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.47.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 883/1147]         blk.47.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.47.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 884/1147]           blk.47.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.47.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 885/1147]         blk.47.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 886/1147]          blk.47.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.47.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 887/1147]              blk.47.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.47.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 888/1147]               blk.47.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.47.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.47.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 889/1147]               blk.47.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.47.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.47.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 890/1147]            blk.47.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.47.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 891/1147]          blk.47.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 892/1147]               blk.47.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.47.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 893/1147]               blk.47.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.47.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 894/1147]              blk.47.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 895/1147]          blk.47.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.47.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 896/1147]          blk.47.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.47.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 897/1147]            blk.47.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.47.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 898/1147]               blk.47.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 899/1147]              blk.48.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 900/1147]           blk.48.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 901/1147]         blk.48.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.48.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 902/1147]         blk.48.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.48.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 903/1147]           blk.48.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.48.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 904/1147]         blk.48.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 905/1147]          blk.48.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.48.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 906/1147]              blk.48.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.48.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 907/1147]               blk.48.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.48.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.48.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 908/1147]               blk.48.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.48.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.48.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 909/1147]            blk.48.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.48.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 910/1147]          blk.48.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 911/1147]               blk.48.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.48.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 912/1147]               blk.48.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.48.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 913/1147]              blk.48.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 914/1147]          blk.48.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.48.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 915/1147]          blk.48.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.48.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 916/1147]            blk.48.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.48.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 917/1147]               blk.48.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 918/1147]              blk.49.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 919/1147]           blk.49.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 920/1147]         blk.49.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.49.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 921/1147]         blk.49.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.49.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 922/1147]           blk.49.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.49.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 923/1147]         blk.49.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 924/1147]          blk.49.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.49.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 925/1147]              blk.49.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.49.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 926/1147]               blk.49.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.49.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.49.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 927/1147]               blk.49.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.49.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.49.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 928/1147]            blk.49.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.49.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 929/1147]          blk.49.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 930/1147]               blk.49.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.49.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 931/1147]               blk.49.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.49.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 932/1147]              blk.49.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 933/1147]          blk.49.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.49.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 934/1147]          blk.49.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.49.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 935/1147]            blk.49.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.49.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 936/1147]               blk.49.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 937/1147]              blk.50.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 938/1147]           blk.50.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 939/1147]         blk.50.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.50.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 940/1147]         blk.50.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.50.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 941/1147]           blk.50.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.50.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 942/1147]         blk.50.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 943/1147]          blk.50.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.50.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 944/1147]              blk.50.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.50.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 945/1147]               blk.50.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.50.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.50.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 946/1147]               blk.50.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.50.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.50.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 947/1147]            blk.50.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.50.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 948/1147]          blk.50.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 949/1147]               blk.50.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.50.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 950/1147]               blk.50.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.50.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 951/1147]              blk.50.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 952/1147]          blk.50.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.50.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 953/1147]          blk.50.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.50.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 954/1147]            blk.50.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.50.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 955/1147]               blk.50.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 956/1147]              blk.51.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 957/1147]           blk.51.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 958/1147]         blk.51.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.51.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 959/1147]         blk.51.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.51.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 960/1147]           blk.51.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.51.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 961/1147]         blk.51.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 962/1147]          blk.51.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.51.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 963/1147]              blk.51.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.51.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 964/1147]               blk.51.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.51.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.51.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 965/1147]               blk.51.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.51.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.51.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 966/1147]            blk.51.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.51.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 967/1147]          blk.51.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 968/1147]               blk.51.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.51.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 969/1147]               blk.51.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.51.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 970/1147]              blk.51.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 971/1147]          blk.51.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.51.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 972/1147]          blk.51.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.51.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 973/1147]            blk.51.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.51.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 974/1147]               blk.51.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 975/1147]              blk.52.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 976/1147]           blk.52.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 977/1147]         blk.52.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.52.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 978/1147]         blk.52.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.52.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 979/1147]           blk.52.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.52.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 980/1147]         blk.52.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 981/1147]          blk.52.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.52.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 982/1147]              blk.52.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.52.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 983/1147]               blk.52.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.52.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.52.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 984/1147]               blk.52.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.52.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.52.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 985/1147]            blk.52.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.52.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 986/1147]          blk.52.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 987/1147]               blk.52.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.52.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 988/1147]               blk.52.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.52.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 989/1147]              blk.52.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 990/1147]          blk.52.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.52.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[ 991/1147]          blk.52.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.52.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 992/1147]            blk.52.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.52.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 993/1147]               blk.52.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 994/1147]              blk.53.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 995/1147]           blk.53.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 996/1147]         blk.53.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.53.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 997/1147]         blk.53.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.53.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 998/1147]           blk.53.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.53.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 999/1147]         blk.53.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1000/1147]          blk.53.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.53.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[1001/1147]              blk.53.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.53.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[1002/1147]               blk.53.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.53.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.53.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1003/1147]               blk.53.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.53.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.53.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1004/1147]            blk.53.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.53.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[1005/1147]          blk.53.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1006/1147]               blk.53.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.53.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[1007/1147]               blk.53.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.53.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[1008/1147]              blk.53.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1009/1147]          blk.53.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.53.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[1010/1147]          blk.53.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.53.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[1011/1147]            blk.53.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.53.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[1012/1147]               blk.53.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1013/1147]              blk.54.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1014/1147]           blk.54.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1015/1147]         blk.54.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.54.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[1016/1147]         blk.54.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.54.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[1017/1147]           blk.54.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.54.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[1018/1147]         blk.54.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1019/1147]          blk.54.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.54.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[1020/1147]              blk.54.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.54.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[1021/1147]               blk.54.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.54.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.54.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1022/1147]               blk.54.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.54.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.54.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1023/1147]            blk.54.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.54.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[1024/1147]          blk.54.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1025/1147]               blk.54.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.54.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[1026/1147]               blk.54.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.54.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[1027/1147]              blk.54.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1028/1147]          blk.54.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.54.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[1029/1147]          blk.54.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.54.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[1030/1147]            blk.54.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.54.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[1031/1147]               blk.54.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1032/1147]              blk.55.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1033/1147]           blk.55.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1034/1147]         blk.55.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.55.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[1035/1147]         blk.55.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.55.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[1036/1147]           blk.55.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.55.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[1037/1147]         blk.55.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1038/1147]          blk.55.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.55.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[1039/1147]              blk.55.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.55.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[1040/1147]               blk.55.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.55.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.55.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1041/1147]               blk.55.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.55.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.55.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1042/1147]            blk.55.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.55.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[1043/1147]          blk.55.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1044/1147]               blk.55.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.55.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[1045/1147]               blk.55.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.55.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[1046/1147]              blk.55.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1047/1147]          blk.55.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.55.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[1048/1147]          blk.55.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.55.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[1049/1147]            blk.55.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.55.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[1050/1147]               blk.55.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1051/1147]              blk.56.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1052/1147]           blk.56.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1053/1147]         blk.56.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.56.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[1054/1147]         blk.56.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.56.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[1055/1147]           blk.56.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.56.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[1056/1147]         blk.56.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1057/1147]          blk.56.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.56.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[1058/1147]              blk.56.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.56.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[1059/1147]               blk.56.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.56.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.56.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1060/1147]               blk.56.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.56.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.56.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1061/1147]            blk.56.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.56.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[1062/1147]          blk.56.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1063/1147]               blk.56.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.56.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[1064/1147]               blk.56.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.56.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[1065/1147]              blk.56.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1066/1147]          blk.56.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.56.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[1067/1147]          blk.56.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.56.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[1068/1147]            blk.56.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.56.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[1069/1147]               blk.56.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1070/1147]              blk.57.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1071/1147]           blk.57.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1072/1147]         blk.57.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.57.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[1073/1147]         blk.57.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.57.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[1074/1147]           blk.57.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.57.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[1075/1147]         blk.57.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1076/1147]          blk.57.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.57.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[1077/1147]              blk.57.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.57.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[1078/1147]               blk.57.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.57.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.57.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1079/1147]               blk.57.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.57.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.57.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1080/1147]            blk.57.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.57.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[1081/1147]          blk.57.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1082/1147]               blk.57.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.57.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[1083/1147]               blk.57.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.57.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[1084/1147]              blk.57.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1085/1147]          blk.57.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.57.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[1086/1147]          blk.57.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.57.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[1087/1147]            blk.57.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.57.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[1088/1147]               blk.57.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1089/1147]              blk.58.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1090/1147]           blk.58.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1091/1147]         blk.58.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.58.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[1092/1147]         blk.58.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.58.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[1093/1147]           blk.58.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.58.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[1094/1147]         blk.58.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1095/1147]          blk.58.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.58.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[1096/1147]              blk.58.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.58.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[1097/1147]               blk.58.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.58.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.58.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1098/1147]               blk.58.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.58.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.58.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1099/1147]            blk.58.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.58.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[1100/1147]          blk.58.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1101/1147]               blk.58.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.58.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[1102/1147]               blk.58.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.58.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[1103/1147]              blk.58.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1104/1147]          blk.58.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.58.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[1105/1147]          blk.58.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.58.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[1106/1147]            blk.58.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.58.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[1107/1147]               blk.58.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1108/1147]              blk.59.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1109/1147]           blk.59.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1110/1147]         blk.59.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.59.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[1111/1147]         blk.59.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.59.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[1112/1147]           blk.59.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.59.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[1113/1147]         blk.59.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1114/1147]          blk.59.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.59.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[1115/1147]              blk.59.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.59.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[1116/1147]               blk.59.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.59.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.59.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1117/1147]               blk.59.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.59.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.59.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1118/1147]            blk.59.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.59.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[1119/1147]          blk.59.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1120/1147]               blk.59.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.59.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[1121/1147]               blk.59.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.59.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[1122/1147]              blk.59.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1123/1147]          blk.59.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.59.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[1124/1147]          blk.59.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.59.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[1125/1147]            blk.59.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.59.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[1126/1147]               blk.59.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1127/1147]              blk.60.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1128/1147]           blk.60.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1129/1147]         blk.60.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.60.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[1130/1147]         blk.60.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.60.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[1131/1147]           blk.60.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.60.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[1132/1147]         blk.60.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1133/1147]          blk.60.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.60.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[1134/1147]              blk.60.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.60.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[1135/1147]               blk.60.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.60.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.60.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1136/1147]               blk.60.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.60.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.60.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1137/1147]            blk.60.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.60.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[1138/1147]          blk.60.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1139/1147]               blk.60.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.60.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[1140/1147]               blk.60.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.60.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[1141/1147]                        output.weight - [ 7168, 129280,     1,     1], type =    f16, Using custom type q8_0 for tensor output.weight

====== llama_model_quantize_internal: did not find weights for output.weight
converting to q8_0 .. size =  1767.50 MiB ->   938.98 MiB
[1142/1147]              blk.60.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1143/1147]          blk.60.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_k for tensor blk.60.ffn_down_exps.weight
converting to iq4_k .. size =  7168.00 MiB ->  2016.00 MiB
[1144/1147]          blk.60.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.60.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[1145/1147]            blk.60.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.60.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[1146/1147]               blk.60.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1147/1147]                   output_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
llama_model_quantize_internal: model size  = 1282038.27 MB
llama_model_quantize_internal: quant size  = 321737.47 MB

main: quantize time = 12877811.18 ms
main:    total time = 12877811.18 ms
```

Perplexity run:
```
./llama-perplexity -m /models/DeepSeek-R1-GGUF/DeepSeek-R1-GGUF-IQ4_K__IQ3_S_Q8.gguf -f /models/wiki.test.raw -fmoe -mla 2 -fa -ts 24/24/24/24/24/24/24/24/24/24/24/24/24/24/24/24 -c 512 -ub 512 --n-gpu-layers 100 -ot "blk\.3\.ffn_(down|gate|up)_exps\.weight|blk\.4\.ffn_(down|gate|up)_exps\.weight|blk\.5\.ffn_(down|gate|up)_exps\.weight=CUDA0" -ot "blk\.6\.ffn_(down|gate|up)_exps\.weight|blk\.7\.ffn_(down|gate|up)_exps\.weight|blk\.8\.ffn_(down|gate|up)_exps\.weight=CUDA1" -ot "blk\.9\.ffn_(down|gate|up)_exps\.weight|blk\.10\.ffn_(down|gate|up)_exps\.weight|blk\.11\.ffn_(down|gate|up)_exps\.weight|blk\.12\.ffn_(down|gate|up)_exps\.weight=CUDA2" -ot "blk\.13\.ffn_(down|gate|up)_exps\.weight|blk\.14\.ffn_(down|gate|up)_exps\.weight|blk\.15\.ffn_(down|gate|up)_exps\.weight|blk\.16\.ffn_(down|gate|up)_exps\.weight=CUDA3" -ot "blk\.17\.ffn_(down|gate|up)_exps\.weight|blk\.18\.ffn_(down|gate|up)_exps\.weight|blk\.19\.ffn_(down|gate|up)_exps\.weight|blk\.20\.ffn_(down|gate|up)_exps\.weight=CUDA4" -ot "blk\.21\.ffn_(down|gate|up)_exps\.weight|blk\.22\.ffn_(down|gate|up)_exps\.weight|blk\.23\.ffn_(down|gate|up)_exps\.weight|blk\.24\.ffn_(down|gate|up)_exps\.weight=CUDA5" -ot "blk\.25\.ffn_(down|gate|up)_exps\.weight|blk\.26\.ffn_(down|gate|up)_exps\.weight|blk\.27\.ffn_(down|gate|up)_exps\.weight|blk\.28\.ffn_(down|gate|up)_exps\.weight=CUDA6" -ot "blk\.29\.ffn_(down|gate|up)_exps\.weight|blk\.30\.ffn_(down|gate|up)_exps\.weight|blk\.31\.ffn_(down|gate|up)_exps\.weight|blk\.32\.ffn_(down|gate|up)_exps\.weight=CUDA7" -ot "blk\.33\.ffn_(down|gate|up)_exps\.weight|blk\.34\.ffn_(down|gate|up)_exps\.weight|blk\.35\.ffn_(down|gate|up)_exps\.weight|blk\.36\.ffn_(down|gate|up)_exps\.weight=CUDA8" -ot "blk\.37\.ffn_(down|gate|up)_exps\.weight|blk\.38\.ffn_(down|gate|up)_exps\.weight|blk\.39\.ffn_(down|gate|up)_exps\.weight|blk\.40\.ffn_(down|gate|up)_exps\.weight=CUDA9" -ot "blk\.41\.ffn_(down|gate|up)_exps\.weight|blk\.42\.ffn_(down|gate|up)_exps\.weight|blk\.43\.ffn_(down|gate|up)_exps\.weight|blk\.44\.ffn_(down|gate|up)_exps\.weight=CUDA10" -ot "blk\.45\.ffn_(down|gate|up)_exps\.weight|blk\.46\.ffn_(down|gate|up)_exps\.weight|blk\.47\.ffn_(down|gate|up)_exps\.weight|blk\.48\.ffn_(down|gate|up)_exps\.weight=CUDA11" -ot "blk\.49\.ffn_(down|gate|up)_exps\.weight|blk\.50\.ffn_(down|gate|up)_exps\.weight|blk\.51\.ffn_(down|gate|up)_exps\.weight|blk\.52\.ffn_(down|gate|up)_exps\.weight=CUDA12" -ot "blk\.53\.ffn_(down|gate|up)_exps\.weight|blk\.54\.ffn_(down|gate|up)_exps\.weight|blk\.55\.ffn_(down|gate|up)_exps\.weight|blk\.56\.ffn_(down|gate|up)_exps\.weight=CUDA13" -ot "blk\.57\.ffn_(down|gate|up)_exps\.weight|blk\.58\.ffn_(down|gate|up)_exps\.weight|blk\.59\.ffn_(down|gate|up)_exps\.weight|blk\.60\.ffn_(down|gate|up)_exps\.weight=CUDA14" --seed 3407 --temp 0.5
```

---

👤 **davidsyoung** commented the **2025-03-10** at **17:06:29**:<br>

PPL run:
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
main: build = 0 (unknown)
main: built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu
main: seed  = 3407
llama_model_loader: loaded meta data with 54 key-value pairs and 1147 tensors from /models/DeepSeek-R1-GGUF/DeepSeek-R1-GGUF-IQ4_K__IQ3_S_Q8.gguf (version GGUF V3 (latest))
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
llama_model_loader: - kv  20:                          general.file_type u32              = 7
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
llama_model_loader: - type q8_0:  438 tensors
llama_model_loader: - type q5_K:  180 tensors
llama_model_loader: - type iq3_s:  104 tensors
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
llm_load_print_meta: model ftype      = Q8_0
llm_load_print_meta: model params     = 672.050 B
llm_load_print_meta: model size       = 314.197 GiB (4.016 BPW) 
llm_load_print_meta: repeating layers = 312.363 GiB (4.004 BPW, 670.196 B parameters)
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
Tensor blk.3.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.6.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.6.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.7.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.7.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.9.ffn_gate_exps.weight buffer type overriden to CUDA2
Tensor blk.9.ffn_down_exps.weight buffer type overriden to CUDA2
Tensor blk.9.ffn_up_exps.weight buffer type overriden to CUDA2
Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CUDA2
Tensor blk.10.ffn_down_exps.weight buffer type overriden to CUDA2
Tensor blk.10.ffn_up_exps.weight buffer type overriden to CUDA2
Tensor blk.11.ffn_gate_exps.weight buffer type overriden to CUDA2
Tensor blk.11.ffn_down_exps.weight buffer type overriden to CUDA2
Tensor blk.11.ffn_up_exps.weight buffer type overriden to CUDA2
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CUDA2
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CUDA2
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CUDA2
Tensor blk.13.ffn_gate_exps.weight buffer type overriden to CUDA3
Tensor blk.13.ffn_down_exps.weight buffer type overriden to CUDA3
Tensor blk.13.ffn_up_exps.weight buffer type overriden to CUDA3
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CUDA3
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CUDA3
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CUDA3
Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CUDA3
Tensor blk.15.ffn_down_exps.weight buffer type overriden to CUDA3
Tensor blk.15.ffn_up_exps.weight buffer type overriden to CUDA3
Tensor blk.16.ffn_gate_exps.weight buffer type overriden to CUDA3
Tensor blk.16.ffn_down_exps.weight buffer type overriden to CUDA3
Tensor blk.16.ffn_up_exps.weight buffer type overriden to CUDA3
Tensor blk.17.ffn_gate_exps.weight buffer type overriden to CUDA4
Tensor blk.17.ffn_down_exps.weight buffer type overriden to CUDA4
Tensor blk.17.ffn_up_exps.weight buffer type overriden to CUDA4
Tensor blk.18.ffn_gate_exps.weight buffer type overriden to CUDA4
Tensor blk.18.ffn_down_exps.weight buffer type overriden to CUDA4
Tensor blk.18.ffn_up_exps.weight buffer type overriden to CUDA4
Tensor blk.19.ffn_gate_exps.weight buffer type overriden to CUDA4
Tensor blk.19.ffn_down_exps.weight buffer type overriden to CUDA4
Tensor blk.19.ffn_up_exps.weight buffer type overriden to CUDA4
Tensor blk.20.ffn_gate_exps.weight buffer type overriden to CUDA4
Tensor blk.20.ffn_down_exps.weight buffer type overriden to CUDA4
Tensor blk.20.ffn_up_exps.weight buffer type overriden to CUDA4
Tensor blk.21.ffn_gate_exps.weight buffer type overriden to CUDA5
Tensor blk.21.ffn_down_exps.weight buffer type overriden to CUDA5
Tensor blk.21.ffn_up_exps.weight buffer type overriden to CUDA5
Tensor blk.22.ffn_gate_exps.weight buffer type overriden to CUDA5
Tensor blk.22.ffn_down_exps.weight buffer type overriden to CUDA5
Tensor blk.22.ffn_up_exps.weight buffer type overriden to CUDA5
Tensor blk.23.ffn_gate_exps.weight buffer type overriden to CUDA5
Tensor blk.23.ffn_down_exps.weight buffer type overriden to CUDA5
Tensor blk.23.ffn_up_exps.weight buffer type overriden to CUDA5
Tensor blk.24.ffn_gate_exps.weight buffer type overriden to CUDA5
Tensor blk.24.ffn_down_exps.weight buffer type overriden to CUDA5
Tensor blk.24.ffn_up_exps.weight buffer type overriden to CUDA5
Tensor blk.25.ffn_gate_exps.weight buffer type overriden to CUDA6
Tensor blk.25.ffn_down_exps.weight buffer type overriden to CUDA6
Tensor blk.25.ffn_up_exps.weight buffer type overriden to CUDA6
Tensor blk.26.ffn_gate_exps.weight buffer type overriden to CUDA6
Tensor blk.26.ffn_down_exps.weight buffer type overriden to CUDA6
Tensor blk.26.ffn_up_exps.weight buffer type overriden to CUDA6
Tensor blk.27.ffn_gate_exps.weight buffer type overriden to CUDA6
Tensor blk.27.ffn_down_exps.weight buffer type overriden to CUDA6
Tensor blk.27.ffn_up_exps.weight buffer type overriden to CUDA6
Tensor blk.28.ffn_gate_exps.weight buffer type overriden to CUDA6
Tensor blk.28.ffn_down_exps.weight buffer type overriden to CUDA6
Tensor blk.28.ffn_up_exps.weight buffer type overriden to CUDA6
Tensor blk.29.ffn_gate_exps.weight buffer type overriden to CUDA7
Tensor blk.29.ffn_down_exps.weight buffer type overriden to CUDA7
Tensor blk.29.ffn_up_exps.weight buffer type overriden to CUDA7
Tensor blk.30.ffn_gate_exps.weight buffer type overriden to CUDA7
Tensor blk.30.ffn_down_exps.weight buffer type overriden to CUDA7
Tensor blk.30.ffn_up_exps.weight buffer type overriden to CUDA7
Tensor blk.31.ffn_gate_exps.weight buffer type overriden to CUDA7
Tensor blk.31.ffn_down_exps.weight buffer type overriden to CUDA7
Tensor blk.31.ffn_up_exps.weight buffer type overriden to CUDA7
Tensor blk.32.ffn_gate_exps.weight buffer type overriden to CUDA7
Tensor blk.32.ffn_down_exps.weight buffer type overriden to CUDA7
Tensor blk.32.ffn_up_exps.weight buffer type overriden to CUDA7
Tensor blk.33.ffn_gate_exps.weight buffer type overriden to CUDA8
Tensor blk.33.ffn_down_exps.weight buffer type overriden to CUDA8
Tensor blk.33.ffn_up_exps.weight buffer type overriden to CUDA8
Tensor blk.34.ffn_gate_exps.weight buffer type overriden to CUDA8
Tensor blk.34.ffn_down_exps.weight buffer type overriden to CUDA8
Tensor blk.34.ffn_up_exps.weight buffer type overriden to CUDA8
Tensor blk.35.ffn_gate_exps.weight buffer type overriden to CUDA8
Tensor blk.35.ffn_down_exps.weight buffer type overriden to CUDA8
Tensor blk.35.ffn_up_exps.weight buffer type overriden to CUDA8
Tensor blk.36.ffn_gate_exps.weight buffer type overriden to CUDA8
Tensor blk.36.ffn_down_exps.weight buffer type overriden to CUDA8
Tensor blk.36.ffn_up_exps.weight buffer type overriden to CUDA8
Tensor blk.37.ffn_gate_exps.weight buffer type overriden to CUDA9
Tensor blk.37.ffn_down_exps.weight buffer type overriden to CUDA9
Tensor blk.37.ffn_up_exps.weight buffer type overriden to CUDA9
Tensor blk.38.ffn_gate_exps.weight buffer type overriden to CUDA9
Tensor blk.38.ffn_down_exps.weight buffer type overriden to CUDA9
Tensor blk.38.ffn_up_exps.weight buffer type overriden to CUDA9
Tensor blk.39.ffn_gate_exps.weight buffer type overriden to CUDA9
Tensor blk.39.ffn_down_exps.weight buffer type overriden to CUDA9
Tensor blk.39.ffn_up_exps.weight buffer type overriden to CUDA9
Tensor blk.40.ffn_gate_exps.weight buffer type overriden to CUDA9
Tensor blk.40.ffn_down_exps.weight buffer type overriden to CUDA9
Tensor blk.40.ffn_up_exps.weight buffer type overriden to CUDA9
Tensor blk.41.ffn_gate_exps.weight buffer type overriden to CUDA10
Tensor blk.41.ffn_down_exps.weight buffer type overriden to CUDA10
Tensor blk.41.ffn_up_exps.weight buffer type overriden to CUDA10
Tensor blk.42.ffn_gate_exps.weight buffer type overriden to CUDA10
Tensor blk.42.ffn_down_exps.weight buffer type overriden to CUDA10
Tensor blk.42.ffn_up_exps.weight buffer type overriden to CUDA10
Tensor blk.43.ffn_gate_exps.weight buffer type overriden to CUDA10
Tensor blk.43.ffn_down_exps.weight buffer type overriden to CUDA10
Tensor blk.43.ffn_up_exps.weight buffer type overriden to CUDA10
Tensor blk.44.ffn_gate_exps.weight buffer type overriden to CUDA10
Tensor blk.44.ffn_down_exps.weight buffer type overriden to CUDA10
Tensor blk.44.ffn_up_exps.weight buffer type overriden to CUDA10
Tensor blk.45.ffn_gate_exps.weight buffer type overriden to CUDA11
Tensor blk.45.ffn_down_exps.weight buffer type overriden to CUDA11
Tensor blk.45.ffn_up_exps.weight buffer type overriden to CUDA11
Tensor blk.46.ffn_gate_exps.weight buffer type overriden to CUDA11
Tensor blk.46.ffn_down_exps.weight buffer type overriden to CUDA11
Tensor blk.46.ffn_up_exps.weight buffer type overriden to CUDA11
Tensor blk.47.ffn_gate_exps.weight buffer type overriden to CUDA11
Tensor blk.47.ffn_down_exps.weight buffer type overriden to CUDA11
Tensor blk.47.ffn_up_exps.weight buffer type overriden to CUDA11
Tensor blk.48.ffn_gate_exps.weight buffer type overriden to CUDA11
Tensor blk.48.ffn_down_exps.weight buffer type overriden to CUDA11
Tensor blk.48.ffn_up_exps.weight buffer type overriden to CUDA11
Tensor blk.49.ffn_gate_exps.weight buffer type overriden to CUDA12
Tensor blk.49.ffn_down_exps.weight buffer type overriden to CUDA12
Tensor blk.49.ffn_up_exps.weight buffer type overriden to CUDA12
Tensor blk.50.ffn_gate_exps.weight buffer type overriden to CUDA12
Tensor blk.50.ffn_down_exps.weight buffer type overriden to CUDA12
Tensor blk.50.ffn_up_exps.weight buffer type overriden to CUDA12
Tensor blk.51.ffn_gate_exps.weight buffer type overriden to CUDA12
Tensor blk.51.ffn_down_exps.weight buffer type overriden to CUDA12
Tensor blk.51.ffn_up_exps.weight buffer type overriden to CUDA12
Tensor blk.52.ffn_gate_exps.weight buffer type overriden to CUDA12
Tensor blk.52.ffn_down_exps.weight buffer type overriden to CUDA12
Tensor blk.52.ffn_up_exps.weight buffer type overriden to CUDA12
Tensor blk.53.ffn_gate_exps.weight buffer type overriden to CUDA13
Tensor blk.53.ffn_down_exps.weight buffer type overriden to CUDA13
Tensor blk.53.ffn_up_exps.weight buffer type overriden to CUDA13
Tensor blk.54.ffn_gate_exps.weight buffer type overriden to CUDA13
Tensor blk.54.ffn_down_exps.weight buffer type overriden to CUDA13
Tensor blk.54.ffn_up_exps.weight buffer type overriden to CUDA13
Tensor blk.55.ffn_gate_exps.weight buffer type overriden to CUDA13
Tensor blk.55.ffn_down_exps.weight buffer type overriden to CUDA13
Tensor blk.55.ffn_up_exps.weight buffer type overriden to CUDA13
Tensor blk.56.ffn_gate_exps.weight buffer type overriden to CUDA13
Tensor blk.56.ffn_down_exps.weight buffer type overriden to CUDA13
Tensor blk.56.ffn_up_exps.weight buffer type overriden to CUDA13
Tensor blk.57.ffn_gate_exps.weight buffer type overriden to CUDA14
Tensor blk.57.ffn_down_exps.weight buffer type overriden to CUDA14
Tensor blk.57.ffn_up_exps.weight buffer type overriden to CUDA14
Tensor blk.58.ffn_gate_exps.weight buffer type overriden to CUDA14
Tensor blk.58.ffn_down_exps.weight buffer type overriden to CUDA14
Tensor blk.58.ffn_up_exps.weight buffer type overriden to CUDA14
Tensor blk.59.ffn_gate_exps.weight buffer type overriden to CUDA14
Tensor blk.59.ffn_down_exps.weight buffer type overriden to CUDA14
Tensor blk.59.ffn_up_exps.weight buffer type overriden to CUDA14
Tensor blk.60.ffn_gate_exps.weight buffer type overriden to CUDA14
Tensor blk.60.ffn_down_exps.weight buffer type overriden to CUDA14
Tensor blk.60.ffn_up_exps.weight buffer type overriden to CUDA14
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size =   938.98 MiB
llm_load_tensors:      CUDA0 buffer size = 21555.36 MiB
llm_load_tensors:      CUDA1 buffer size = 20458.12 MiB
llm_load_tensors:      CUDA2 buffer size = 21354.12 MiB
llm_load_tensors:      CUDA3 buffer size = 21354.12 MiB
llm_load_tensors:      CUDA4 buffer size = 21354.12 MiB
llm_load_tensors:      CUDA5 buffer size = 21354.12 MiB
llm_load_tensors:      CUDA6 buffer size = 21354.12 MiB
llm_load_tensors:      CUDA7 buffer size = 21111.59 MiB
llm_load_tensors:      CUDA8 buffer size = 21354.12 MiB
llm_load_tensors:      CUDA9 buffer size = 21354.12 MiB
llm_load_tensors:     CUDA10 buffer size = 21354.12 MiB
llm_load_tensors:     CUDA11 buffer size = 21354.12 MiB
llm_load_tensors:     CUDA12 buffer size = 21354.12 MiB
llm_load_tensors:     CUDA13 buffer size = 21354.12 MiB
llm_load_tensors:     CUDA14 buffer size = 21354.12 MiB
llm_load_tensors:     CUDA15 buffer size =  1424.07 MiB
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
llama_new_context_with_model:  CUDA_Host  output buffer size =     1.97 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=4)
llama_new_context_with_model:      CUDA0 compute buffer size =   915.00 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =   982.00 MiB
llama_new_context_with_model:      CUDA2 compute buffer size =   926.00 MiB
llama_new_context_with_model:      CUDA3 compute buffer size =   926.00 MiB
llama_new_context_with_model:      CUDA4 compute buffer size =   926.00 MiB
llama_new_context_with_model:      CUDA5 compute buffer size =   926.00 MiB
llama_new_context_with_model:      CUDA6 compute buffer size =   926.00 MiB
llama_new_context_with_model:      CUDA7 compute buffer size =   986.00 MiB
llama_new_context_with_model:      CUDA8 compute buffer size =  1042.00 MiB
llama_new_context_with_model:      CUDA9 compute buffer size =  1042.00 MiB
llama_new_context_with_model:     CUDA10 compute buffer size =  1042.00 MiB
llama_new_context_with_model:     CUDA11 compute buffer size =  1042.00 MiB
llama_new_context_with_model:     CUDA12 compute buffer size =  1042.00 MiB
llama_new_context_with_model:     CUDA13 compute buffer size =  1042.00 MiB
llama_new_context_with_model:     CUDA14 compute buffer size =  1042.00 MiB
llama_new_context_with_model:     CUDA15 compute buffer size =   912.02 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =    30.02 MiB
llama_new_context_with_model: graph nodes  = 3548
llama_new_context_with_model: graph splits = 65

system_info: n_threads = 64 / 128 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | 
perplexity: tokenizing the input ..
perplexity: tokenization took 1153.27 ms
perplexity: calculating perplexity over 561 chunks, n_ctx=512, batch_size=2048, n_seq=4
perplexity: 14.67 seconds per pass - ETA 34.30 minutes
[1]2.6227,[2]3.3777,[3]2.4231,[4]2.0272,[5]1.8467,[6]1.6985,[7]1.5984,[8]1.5279,[9]1.4757,[10]nan,[11]nan,[12]nan,[13]nan,[14]nan,[15]nan,[16]nan,[17]nan,[18]nan,[19]nan,[20]nan,[21]nan,[22]nan,[23]nan,[24]nan,^C
```

---

👤 **ikawrakow** commented the **2025-03-10** at **17:10:14**:<br>

Thanks. Don't see anything wrong.

Can you try with #251?

---

👤 **davidsyoung** commented the **2025-03-10** at **17:13:27**:<br>

> Thanks. Don't see anything wrong.
> 
> Can you try with [#251](https://github.com/ikawrakow/ik_llama.cpp/pull/251)?

Yes no problem, building - will report back. Also, I noticed that it had set n_seq=4 on it's own accord in the perplexity run. Could that be it?

---

👤 **ikawrakow** commented the **2025-03-10** at **17:18:41**:<br>

> I noticed that it had set n_seq=4 

No, this gets calculated internally. It is `n_batch / n_ctx`. If you use `n_ctx = 512` and don't change `n_batch` to 512 via `-b`, it will compute 4 chunks of 512 tokens in one batch, and you will see 4 PPL values printed at once.

---

👤 **davidsyoung** commented the **2025-03-10** at **17:19:20**:<br>

> > I noticed that it had set n_seq=4
> 
> No, this gets calculated internally. It is `n_batch / n_ctx`. If you use `n_ctx = 512` and don't change `n_batch` to 512 via `-b`, it will compute 4 chunks of 512 tokens in one batch, and you will see 4 PPL values printed at once.

ah, got it

---

👤 **davidsyoung** commented the **2025-03-10** at **17:32:26**:<br>

I'm afraid it is producing NaNs again with #251 @ikawrakow.

It starts producing NaNs on chunk 10 with `-fa` and without chunk 17. 

`-fa`:
`[1]2.6215,[2]3.3918,[3]2.4254,[4]2.0245,[5]1.8467,[6]1.6971,[7]1.5972,[8]1.5278,[9]1.4765,[10]nan,[11]nan,[12]nan,[13]nan,[14]nan,[15]nan,[16]nan,[17]nan,`

without `-fa`:
`[1]2.6160,[2]3.3842,[3]2.4246,[4]2.0259,[5]1.8470,[6]1.6980,[7]1.5990,[8]1.5281,[9]1.4770,[10]1.4340,[11]1.4205,[12]1.4415,[13]1.4523,[14]1.5825,[15]1.7121,[16]1.7733,[17]nan,[18]nan,[19]nan,[20]nan,[21]nan,[22]nan,[23]nan,[24]nan,^C`

---

👤 **davidsyoung** commented the **2025-03-10** at **17:32:26**:<br>

I'm afraid producing NaNs again.

It starts producing NaNs on chunk 10 with `-fa` and without chunk 17. 

`-fa`:
`[1]2.6215,[2]3.3918,[3]2.4254,[4]2.0245,[5]1.8467,[6]1.6971,[7]1.5972,[8]1.5278,[9]1.4765,[10]nan,[11]nan,[12]nan,[13]nan,[14]nan,[15]nan,[16]nan,[17]nan,`

without `-fa`:
`[1]2.6160,[2]3.3842,[3]2.4246,[4]2.0259,[5]1.8470,[6]1.6980,[7]1.5990,[8]1.5281,[9]1.4770,[10]1.4340,[11]1.4205,[12]1.4415,[13]1.4523,[14]1.5825,[15]1.7121,[16]1.7733,[17]nan,[18]nan,[19]nan,[20]nan,[21]nan,[22]nan,[23]nan,[24]nan,^C`

---

👤 **ikawrakow** commented the **2025-03-10** at **17:40:20**:<br>

Do you still have the `IQ3_S` quantization? Does it produce NaNs with that with `mla = 2, fa = 1`?

---

👤 **davidsyoung** commented the **2025-03-10** at **17:45:03**:<br>

> Do you still have the `IQ3_S` quantization? Does it produce NaNs with that with `mla = 2, fa = 1`?

I completed a run of that yesterday:

https://github.com/ikawrakow/ik_llama.cpp/pull/239#issuecomment-2709008864

In short, it didn't produce NaNs

---

👤 **davidsyoung** commented the **2025-03-10** at **17:48:23**:<br>

You can also rule out `-fmoe` being the issue, did a run with `fmoe = 0, fa = 0, mla = 2`, still produced NaN's after 16 chunks.

---

👤 **ikawrakow** commented the **2025-03-10** at **17:49:16**:<br>

Can we conclude from this that `IQ4_K` and `IQ4_KSS` do not work for DeepSeekR1? This would be really strange because I have tried `IQ4_K` on quite a few models, and it always was significantly better than `Q4_K`.

---

👤 **davidsyoung** commented the **2025-03-10** at **17:52:16**:<br>

> Can we conclude from this that `IQ4_K` and `IQ4_KSS` do not work for DeepSeekR1? This would be really strange because I have tried `IQ4_K` on quite a few models, and it always was significantly better than `Q4_K`.

Yeah, very possible. I mean, the model output seems good to me, but if NaN's are being produced for perplexity it makes me concerned that there's something wrong (ie it's masking it). Is there anything different that happens in perplexity vs model output?

---

👤 **ikawrakow** commented the **2025-03-10** at **17:52:16**:<br>

Te only thing that comes to mind at this point is to quantize the same model as this not working one, replacing `iq4_k` with `q4_K`.

---

👤 **davidsyoung** commented the **2025-03-10** at **17:53:45**:<br>

Also, if I do `-ub 32` it seems to work as per here: https://github.com/ikawrakow/ik_llama.cpp/issues/245#issuecomment-2707282221. 

It does make me think that it's not a model problem, and instead inference code somewhere? What path would be activated with `-ub 32` compared to what we're doing now?

---

👤 **davidsyoung** commented the **2025-03-10** at **18:03:19**:<br>

`mla = 0, fa = 0, fmoe = 0` produces NaN's after only 2 chunks.

```
[1]2.6133,[2]3.3819,[3]nan,[4]nan,[5]nan,[6]nan,[7]nan,[8]nan,[9]nan,[10]nan,[11]nan,[12]nan,
```

---

👤 **ikawrakow** commented the **2025-03-10** at **18:07:11**:<br>

`-ub 32` is exactly the same path as no `-ub` argument when running perplexity. The only difference is in the sizes of the matrices that get multiplied. As these are multi-threaded, and the way the work gets split up between the threads depends on the size of the matrices involved, results can change because of that. 

Token generation takes a slightly different path.

But if you get NaNs with `mla = 0`, this means that even standard attention is not working. But standard attention has been tested for so long with so many models, I find it extremely unlikely that the issue would be there. This really points to the MoE part. And as this worked with experts quantized with `IQ3_S`, it would mean it is `IQ4_KSS` and `IQ4_K` not working.

There is an actual difference between `IQ3_S` and `IQ4_K`. `IQ3_S` has a quantized matrix multiplication kernel (a.k.a. MMQ), `IQ4_K` does not. When there is no MMQ kernel, the matrix multiplication is done by first de-quantizing to `fp16`, and then using `fp16` matrix multiplication provided by cuBLAS. If the `fp16` range is not sufficient, we can get NaNs. As I did observe numerical issues with DeepSeek-Lite using the `fp16` model (see [here](https://github.com/ikawrakow/ik_llama.cpp/issues/245#issuecomment-2710726595), and as the issue did not go away when I changed the attention precision to `fp32`, it might as well be that `fp16` simply does not work for DeepSeekR1. If that would be true, non of the `IQX_K` quants can be used because they all don't have an MMQ kernel.

---

👤 **ikawrakow** commented the **2025-03-10** at **18:07:11**:<br>

`-ub 32` is exactly the same path as no `-ub` argument when running perplexity. The only difference is in the sizes of the matrices that get multiplied. As these are multi-threaded, and the way the work gets split up between the threads depends on the size of the matrices involved, results can change because of that. 

Token generation takes a slightly different path.

But if you get NaNs with `mla = 0`, this means that even standard attention is not working. But standard attention has been tested for so long with so many models, I find it extremely unlikely that the issue would be there. This really points to the MoE part. And as this worked with experts quantized with `IQ3_S`, it would mean it is `IQ4_KSS` and `IQ4_K` not working.

---

👤 **davidsyoung** commented the **2025-03-10** at **18:15:12**:<br>

Ah, I see.

I doubt this narrows it down, but.

It actually worked with `IQ3_M` as well, which I believe has some tensors as `IQ4_K`. 

https://github.com/ikawrakow/ik_llama.cpp/pull/239#issuecomment-2702105979

```
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  306 tensors
llama_model_loader: - type q5_K:   61 tensors
llama_model_loader: - type q6_K:    1 tensors
llama_model_loader: - type iq3_s:  407 tensors
llama_model_loader: - type iq4_k:   11 tensors
```

But not many tensors. 

---

Would it be hard to build a MMQ kernel for `IQX_K`?

---

👤 **davidsyoung** commented the **2025-03-10** at **18:15:12**:<br>

Ah, I see.

I doubt this narrows it down, but.

It actually worked with `IQ3_M` as well, which I believe has some tensors as `IQ4_K`. 

https://github.com/ikawrakow/ik_llama.cpp/pull/239#issuecomment-2702105979

```
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  306 tensors
llama_model_loader: - type q5_K:   61 tensors
llama_model_loader: - type q6_K:    1 tensors
llama_model_loader: - type iq3_s:  407 tensors
llama_model_loader: - type iq4_k:   11 tensors
```

But not many tensors. 

---

---

👤 **ikawrakow** commented the **2025-03-10** at **18:16:22**:<br>

Btw, what is the CPU in this system and how much RAM is there? A simple experiment to narrow it down would be to run the MoE part on the CPU (in case there is enough RAM and the CPU is not too slow)

---

👤 **davidsyoung** commented the **2025-03-10** at **18:17:46**:<br>

EPYC 7713 w/ 256GB DDR4 RAM. I don't think the experts will fit on the RAM sadly.

---

👤 **ikawrakow** commented the **2025-03-10** at **18:18:59**:<br>

> It actually worked with IQ3_M as well, which I believe has some tensors as IQ4_K.

Only 11. It is enough to have one misbehaving tensor (misbehaving when quantized with `IQ4_K`) in the experts to get the NaNs. It just so happens that the misbehaving tensor(s) were not part of the 11 in the `IQ3_M` mix.

I need to go get dinner now.

---

👤 **davidsyoung** commented the **2025-03-10** at **18:22:26**:<br>

Does this mean that the first 8 layers of the model are set to IQ4_K? Could be reading it wrong.

```
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M && (i_layer < n_layer/8 ||
                    (qs.model.hparams.n_expert == 8 && use_more_bits(i_layer, n_layer)))) {
            new_type = GGML_TYPE_IQ4_K;
```

These would likely be activated, right? But maybe just no issues in these tensors with that quant.

---

No panic at all, this isn't a priority in the grand scheme.

Could it possibly be an issue with the `-ot` commands? I need to run these to get the model loaded for me, and could splitting it over GPUs introduce issues?

---

👤 **davidsyoung** commented the **2025-03-10** at **18:30:45**:<br>

Could I replace `IQ4_K` with `IQ4_XS`? Or would that suffer the same type of issues as the `IQX_K` quants? Trying to find a suitable quant to replace, and slightly smaller wouldn't be the worst in terms of VRAM.

---

👤 **ikawrakow** commented the **2025-03-10** at **18:35:49**:<br>

> Could I replace `IQ4_K` with `IQ4_XS`? Or would that suffer the same type of issues as the `IQX_K` quants? Trying to find a suitable quant to replace, and slightly smaller wouldn't be the worst in terms of VRAM.

IQ4_XS has MMQ kernel, so yes, you can use that.

---

👤 **davidsyoung** commented the **2025-03-10** at **18:38:52**:<br>

> > Could I replace `IQ4_K` with `IQ4_XS`? Or would that suffer the same type of issues as the `IQX_K` quants? Trying to find a suitable quant to replace, and slightly smaller wouldn't be the worst in terms of VRAM.
> 
> IQ4_XS has MMQ kernel, so yes, you can use that.

Or would IQ4_NL be comparable to IQ4_K? 

UPDATE: Will quant with IQ4_XS, and hopefully get a NaN-Free PPL and go from there.

---

👤 **davidsyoung** commented the **2025-03-10** at **18:38:52**:<br>

> > Could I replace `IQ4_K` with `IQ4_XS`? Or would that suffer the same type of issues as the `IQX_K` quants? Trying to find a suitable quant to replace, and slightly smaller wouldn't be the worst in terms of VRAM.
> 
> IQ4_XS has MMQ kernel, so yes, you can use that.

Or would IQ4_NL be comparable to IQ4_K?

---

👤 **davidsyoung** commented the **2025-03-10** at **23:31:08**:<br>

Quanted with `IQ4_XS` as primary type:

```
root@13c28d802a57:/app/build/bin# ./llama-quantize --imatrix /models/deepseek-config/imatrix.dat \
  --token-embedding-type q8_0 \
  --attn-q-type q8_0 \
  --attn-k-type q8_0 \
  --attn-v-type q8_0 \
  --attn-qkv-type q8_0 \
  --attn-output-type q8_0 \
  --ffn-gate-type q8_0 \
  --ffn-down-type q8_0 \
  --ffn-up-type q8_0 \
  --custom-q "\.attn_.*\.weight=q8_0" \
  --custom-q "\.ffn_.*_shexp\.weight=q5_K,output\.weight=q8_0" \
  --custom-q "blk\.3\.ffn_down_exps\.weight=q5_K,blk\.4\.ffn_down_exps\.weight=q5_K,blk\.5\.ffn_down_exps\.weight=q5_K,blk\.3\.ffn_up_exps\.weight=iq4_xs,blk\.3\.ffn_gate_exps\.weight=iq4_xs,blk\.4\.ffn_up_exps\.weight=iq4_xs,blk\.4\.ffn_gate_exps\.weight=iq4_xs,blk\.5\.ffn_up_exps\.weight=iq4_xs,blk\.5\.ffn_gate_exps\.weight=iq4_xs" \
  --custom-q "blk\.6\.ffn_down_exps\.weight=q5_K,blk\.7\.ffn_down_exps\.weight=q5_K,blk\.8\.ffn_down_exps\.weight=q5_K,blk\.6\.ffn_up_exps\.weight=iq4_xs,blk\.6\.ffn_gate_exps\.weight=iq4_xs,blk\.7\.ffn_up_exps\.weight=iq4_xs,blk\.7\.ffn_gate_exps\.weight=iq4_xs,blk\.8\.ffn_up_exps\.weight=iq4_xs,blk\.8\.ffn_gate_exps\.weight=iq4_xs" \
  --custom-q "blk\.9\.ffn_down_exps\.weight=iq4_xs,blk\.10\.ffn_down_exps\.weight=iq4_xs,blk\.11\.ffn_down_exps\.weight=iq4_xs,blk\.12\.ffn_down_exps\.weight=iq4_xs,blk\.9\.ffn_up_exps\.weight=iq3_s,blk\.9\.ffn_gate_exps\.weight=iq3_s,blk\.10\.ffn_up_exps\.weight=iq3_s,blk\.10\.ffn_gate_exps\.weight=iq3_s,blk\.11\.ffn_up_exps\.weight=iq3_s,blk\.11\.ffn_gate_exps\.weight=iq3_s,blk\.12\.ffn_up_exps\.weight=iq3_s,blk\.12\.ffn_gate_exps\.weight=iq3_s" \
  --custom-q "blk\.13\.ffn_down_exps\.weight=iq4_xs,blk\.14\.ffn_down_exps\.weight=iq4_xs,blk\.15\.ffn_down_exps\.weight=iq4_xs,blk\.16\.ffn_down_exps\.weight=iq4_xs,blk\.13\.ffn_up_exps\.weight=iq3_s,blk\.13\.ffn_gate_exps\.weight=iq3_s,blk\.14\.ffn_up_exps\.weight=iq3_s,blk\.14\.ffn_gate_exps\.weight=iq3_s,blk\.15\.ffn_up_exps\.weight=iq3_s,blk\.15\.ffn_gate_exps\.weight=iq3_s,blk\.16\.ffn_up_exps\.weight=iq3_s,blk\.16\.ffn_gate_exps\.weight=iq3_s" \
  --custom-q "blk\.17\.ffn_down_exps\.weight=iq4_xs,blk\.18\.ffn_down_exps\.weight=iq4_xs,blk\.19\.ffn_down_exps\.weight=iq4_xs,blk\.20\.ffn_down_exps\.weight=iq4_xs,blk\.17\.ffn_up_exps\.weight=iq3_s,blk\.17\.ffn_gate_exps\.weight=iq3_s,blk\.18\.ffn_up_exps\.weight=iq3_s,blk\.18\.ffn_gate_exps\.weight=iq3_s,blk\.19\.ffn_up_exps\.weight=iq3_s,blk\.19\.ffn_gate_exps\.weight=iq3_s,blk\.20\.ffn_up_exps\.weight=iq3_s,blk\.20\.ffn_gate_exps\.weight=iq3_s" \
  --custom-q "blk\.21\.ffn_down_exps\.weight=iq4_xs,blk\.22\.ffn_down_exps\.weight=iq4_xs,blk\.23\.ffn_down_exps\.weight=iq4_xs,blk\.24\.ffn_down_exps\.weight=iq4_xs,blk\.21\.ffn_up_exps\.weight=iq3_s,blk\.21\.ffn_gate_exps\.weight=iq3_s,blk\.22\.ffn_up_exps\.weight=iq3_s,blk\.22\.ffn_gate_exps\.weight=iq3_s,blk\.23\.ffn_up_exps\.weight=iq3_s,blk\.23\.ffn_gate_exps\.weight=iq3_s,blk\.24\.ffn_up_exps\.weight=iq3_s,blk\.24\.ffn_gate_exps\.weight=iq3_s" \
  --custom-q "blk\.25\.ffn_down_exps\.weight=iq4_xs,blk\.26\.ffn_down_exps\.weight=iq4_xs,blk\.27\.ffn_down_exps\.weight=iq4_xs,blk\.28\.ffn_down_exps\.weight=iq4_xs,blk\.25\.ffn_up_exps\.weight=iq3_s,blk\.25\.ffn_gate_exps\.weight=iq3_s,blk\.26\.ffn_up_exps\.weight=iq3_s,blk\.26\.ffn_gate_exps\.weight=iq3_s,blk\.27\.ffn_up_exps\.weight=iq3_s,blk\.27\.ffn_gate_exps\.weight=iq3_s,blk\.28\.ffn_up_exps\.weight=iq3_s,blk\.28\.ffn_gate_exps\.weight=iq3_s" \
  --custom-q "blk\.29\.ffn_down_exps\.weight=iq4_xs,blk\.30\.ffn_down_exps\.weight=iq4_xs,blk\.31\.ffn_down_exps\.weight=iq4_xs,blk\.32\.ffn_down_exps\.weight=iq4_xs,blk\.29\.ffn_up_exps\.weight=iq3_s,blk\.29\.ffn_gate_exps\.weight=iq3_s,blk\.30\.ffn_up_exps\.weight=iq3_s,blk\.30\.ffn_gate_exps\.weight=iq3_s,blk\.31\.ffn_up_exps\.weight=iq3_s,blk\.31\.ffn_gate_exps\.weight=iq3_s,blk\.32\.ffn_up_exps\.weight=iq3_s,blk\.32\.ffn_gate_exps\.weight=iq3_s" \
  --custom-q "blk\.33\.ffn_down_exps\.weight=iq4_xs,blk\.34\.ffn_down_exps\.weight=iq4_xs,blk\.35\.ffn_down_exps\.weight=iq4_xs,blk\.36\.ffn_down_exps\.weight=iq4_xs,blk\.33\.ffn_up_exps\.weight=iq3_s,blk\.33\.ffn_gate_exps\.weight=iq3_s,blk\.34\.ffn_up_exps\.weight=iq3_s,blk\.34\.ffn_gate_exps\.weight=iq3_s,blk\.35\.ffn_up_exps\.weight=iq3_s,blk\.35\.ffn_gate_exps\.weight=iq3_s,blk\.36\.ffn_up_exps\.weight=iq3_s,blk\.36\.ffn_gate_exps\.weight=iq3_s" \
  --custom-q "blk\.37\.ffn_down_exps\.weight=iq4_xs,blk\.38\.ffn_down_exps\.weight=iq4_xs,blk\.39\.ffn_down_exps\.weight=iq4_xs,blk\.40\.ffn_down_exps\.weight=iq4_xs,blk\.37\.ffn_up_exps\.weight=iq3_s,blk\.37\.ffn_gate_exps\.weight=iq3_s,blk\.38\.ffn_up_exps\.weight=iq3_s,blk\.38\.ffn_gate_exps\.weight=iq3_s,blk\.39\.ffn_up_exps\.weight=iq3_s,blk\.39\.ffn_gate_exps\.weight=iq3_s,blk\.40\.ffn_up_exps\.weight=iq3_s,blk\.40\.ffn_gate_exps\.weight=iq3_s" \
  --custom-q "blk\.41\.ffn_down_exps\.weight=iq4_xs,blk\.42\.ffn_down_exps\.weight=iq4_xs,blk\.43\.ffn_down_exps\.weight=iq4_xs,blk\.44\.ffn_down_exps\.weight=iq4_xs,blk\.41\.ffn_up_exps\.weight=iq3_s,blk\.41\.ffn_gate_exps\.weight=iq3_s,blk\.42\.ffn_up_exps\.weight=iq3_s,blk\.42\.ffn_gate_exps\.weight=iq3_s,blk\.43\.ffn_up_exps\.weight=iq3_s,blk\.43\.ffn_gate_exps\.weight=iq3_s,blk\.44\.ffn_up_exps\.weight=iq3_s,blk\.44\.ffn_gate_exps\.weight=iq3_s" \
  --custom-q "blk\.45\.ffn_down_exps\.weight=iq4_xs,blk\.46\.ffn_down_exps\.weight=iq4_xs,blk\.47\.ffn_down_exps\.weight=iq4_xs,blk\.48\.ffn_down_exps\.weight=iq4_xs,blk\.45\.ffn_up_exps\.weight=iq3_s,blk\.45\.ffn_gate_exps\.weight=iq3_s,blk\.46\.ffn_up_exps\.weight=iq3_s,blk\.46\.ffn_gate_exps\.weight=iq3_s,blk\.47\.ffn_up_exps\.weight=iq3_s,blk\.47\.ffn_gate_exps\.weight=iq3_s,blk\.48\.ffn_up_exps\.weight=iq3_s,blk\.48\.ffn_gate_exps\.weight=iq3_s" \
  --custom-q "blk\.49\.ffn_down_exps\.weight=iq4_xs,blk\.50\.ffn_down_exps\.weight=iq4_xs,blk\.51\.ffn_down_exps\.weight=iq4_xs,blk\.52\.ffn_down_exps\.weight=iq4_xs,blk\.49\.ffn_up_exps\.weight=iq3_s,blk\.49\.ffn_gate_exps\.weight=iq3_s,blk\.50\.ffn_up_exps\.weight=iq3_s,blk\.50\.ffn_gate_exps\.weight=iq3_s,blk\.51\.ffn_up_exps\.weight=iq3_s,blk\.51\.ffn_gate_exps\.weight=iq3_s,blk\.52\.ffn_up_exps\.weight=iq3_s,blk\.52\.ffn_gate_exps\.weight=iq3_s" \
  --custom-q "blk\.53\.ffn_down_exps\.weight=iq4_xs,blk\.54\.ffn_down_exps\.weight=iq4_xs,blk\.55\.ffn_down_exps\.weight=iq4_xs,blk\.56\.ffn_down_exps\.weight=iq4_xs,blk\.53\.ffn_up_exps\.weight=iq3_s,blk\.53\.ffn_gate_exps\.weight=iq3_s,blk\.54\.ffn_up_exps\.weight=iq3_s,blk\.54\.ffn_gate_exps\.weight=iq3_s,blk\.55\.ffn_up_exps\.weight=iq3_s,blk\.55\.ffn_gate_exps\.weight=iq3_s,blk\.56\.ffn_up_exps\.weight=iq3_s,blk\.56\.ffn_gate_exps\.weight=iq3_s" \
  --custom-q "blk\.57\.ffn_down_exps\.weight=iq4_xs,blk\.58\.ffn_down_exps\.weight=iq4_xs,blk\.59\.ffn_down_exps\.weight=iq4_xs,blk\.60\.ffn_down_exps\.weight=iq4_xs,blk\.57\.ffn_up_exps\.weight=iq3_s,blk\.57\.ffn_gate_exps\.weight=iq3_s,blk\.58\.ffn_up_exps\.weight=iq3_s,blk\.58\.ffn_gate_exps\.weight=iq3_s,blk\.59\.ffn_up_exps\.weight=iq3_s,blk\.59\.ffn_gate_exps\.weight=iq3_s,blk\.60\.ffn_up_exps\.weight=iq3_s,blk\.60\.ffn_gate_exps\.weight=iq3_s" \
  /storage/DeepSeek-R1-GGUF/unsloth_DeepSeek-R1-BF16-256x21B-F16-00001-of-00059.gguf \
  /models/DeepSeek-R1-GGUF/DeepSeek-R1-GGUF-iq4_xs__iq3_s_q8.gguf \
  q8_0 64
Adding custom rule \.attn_.*\.weight -> q8_0
Adding custom rule \.ffn_.*_shexp\.weight -> q5_K
Adding custom rule output\.weight -> q8_0
Adding custom rule blk\.3\.ffn_down_exps\.weight -> q5_K
Adding custom rule blk\.4\.ffn_down_exps\.weight -> q5_K
Adding custom rule blk\.5\.ffn_down_exps\.weight -> q5_K
Adding custom rule blk\.3\.ffn_up_exps\.weight -> iq4_xs
Adding custom rule blk\.3\.ffn_gate_exps\.weight -> iq4_xs
Adding custom rule blk\.4\.ffn_up_exps\.weight -> iq4_xs
Adding custom rule blk\.4\.ffn_gate_exps\.weight -> iq4_xs
Adding custom rule blk\.5\.ffn_up_exps\.weight -> iq4_xs
Adding custom rule blk\.5\.ffn_gate_exps\.weight -> iq4_xs
Adding custom rule blk\.6\.ffn_down_exps\.weight -> q5_K
Adding custom rule blk\.7\.ffn_down_exps\.weight -> q5_K
Adding custom rule blk\.8\.ffn_down_exps\.weight -> q5_K
Adding custom rule blk\.6\.ffn_up_exps\.weight -> iq4_xs
Adding custom rule blk\.6\.ffn_gate_exps\.weight -> iq4_xs
Adding custom rule blk\.7\.ffn_up_exps\.weight -> iq4_xs
Adding custom rule blk\.7\.ffn_gate_exps\.weight -> iq4_xs
Adding custom rule blk\.8\.ffn_up_exps\.weight -> iq4_xs
Adding custom rule blk\.8\.ffn_gate_exps\.weight -> iq4_xs
Adding custom rule blk\.9\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.10\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.11\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.12\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.9\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.9\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.10\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.10\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.11\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.11\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.12\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.12\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.13\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.14\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.15\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.16\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.13\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.13\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.14\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.14\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.15\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.15\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.16\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.16\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.17\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.18\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.19\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.20\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.17\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.17\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.18\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.18\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.19\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.19\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.20\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.20\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.21\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.22\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.23\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.24\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.21\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.21\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.22\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.22\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.23\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.23\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.24\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.24\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.25\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.26\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.27\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.28\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.25\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.25\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.26\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.26\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.27\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.27\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.28\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.28\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.29\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.30\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.31\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.32\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.29\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.29\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.30\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.30\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.31\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.31\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.32\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.32\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.33\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.34\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.35\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.36\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.33\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.33\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.34\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.34\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.35\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.35\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.36\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.36\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.37\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.38\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.39\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.40\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.37\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.37\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.38\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.38\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.39\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.39\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.40\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.40\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.41\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.42\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.43\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.44\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.41\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.41\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.42\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.42\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.43\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.43\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.44\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.44\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.45\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.46\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.47\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.48\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.45\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.45\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.46\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.46\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.47\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.47\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.48\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.48\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.49\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.50\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.51\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.52\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.49\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.49\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.50\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.50\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.51\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.51\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.52\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.52\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.53\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.54\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.55\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.56\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.53\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.53\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.54\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.54\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.55\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.55\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.56\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.56\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.57\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.58\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.59\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.60\.ffn_down_exps\.weight -> iq4_xs
Adding custom rule blk\.57\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.57\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.58\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.58\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.59\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.59\.ffn_gate_exps\.weight -> iq3_s
Adding custom rule blk\.60\.ffn_up_exps\.weight -> iq3_s
Adding custom rule blk\.60\.ffn_gate_exps\.weight -> iq3_s
load_imatrix: imatrix dataset='imatrix-training-full-3'
load_imatrix: loaded 720 importance matrix entries from /models/deepseek-config/imatrix.dat computed on 315 chunks
prepare_imatrix: have 720 importance matrix entries
main: build = 0 (unknown)
main: built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu
main: quantizing '/storage/DeepSeek-R1-GGUF/unsloth_DeepSeek-R1-BF16-256x21B-F16-00001-of-00059.gguf' to '/models/DeepSeek-R1-GGUF/DeepSeek-R1-GGUF-iq4_xs__iq3_s_q8.gguf' as Q8_0 using 64 threads
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
llama_model_loader: - kv  40:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<�...
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
[   3/1147]                blk.0.ffn_down.weight - [18432,  7168,     1,     1], type =    f16, converting to q8_0 .. size =   252.00 MiB ->   133.88 MiB
[   4/1147]                blk.0.ffn_gate.weight - [ 7168, 18432,     1,     1], type =    f16, converting to q8_0 .. size =   252.00 MiB ->   133.88 MiB
[   5/1147]                  blk.0.ffn_up.weight - [ 7168, 18432,     1,     1], type =    f16, converting to q8_0 .. size =   252.00 MiB ->   133.88 MiB
[   6/1147]                blk.0.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[   7/1147]          blk.0.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[   8/1147]           blk.0.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.0.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[   9/1147]               blk.0.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.0.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[  10/1147]                blk.0.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.0.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.0.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  11/1147]                blk.0.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.0.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  12/1147]             blk.0.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.0.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[  13/1147]           blk.0.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  14/1147]                blk.0.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.0.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[  15/1147]                blk.0.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.0.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[  16/1147]               blk.1.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  17/1147]                blk.1.ffn_down.weight - [18432,  7168,     1,     1], type =    f16, converting to q8_0 .. size =   252.00 MiB ->   133.88 MiB
[  18/1147]                blk.1.ffn_gate.weight - [ 7168, 18432,     1,     1], type =    f16, converting to q8_0 .. size =   252.00 MiB ->   133.88 MiB
[  19/1147]                  blk.1.ffn_up.weight - [ 7168, 18432,     1,     1], type =    f16, converting to q8_0 .. size =   252.00 MiB ->   133.88 MiB
[  20/1147]                blk.1.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  21/1147]          blk.1.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  22/1147]           blk.1.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.1.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[  23/1147]               blk.1.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.1.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[  24/1147]                blk.1.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.1.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.1.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  25/1147]                blk.1.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.1.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  26/1147]             blk.1.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.1.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[  27/1147]           blk.1.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  28/1147]                blk.1.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.1.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[  29/1147]                blk.1.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.1.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[  30/1147]               blk.2.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  31/1147]                blk.2.ffn_down.weight - [18432,  7168,     1,     1], type =    f16, converting to q8_0 .. size =   252.00 MiB ->   133.88 MiB
[  32/1147]                blk.2.ffn_gate.weight - [ 7168, 18432,     1,     1], type =    f16, converting to q8_0 .. size =   252.00 MiB ->   133.88 MiB
[  33/1147]                  blk.2.ffn_up.weight - [ 7168, 18432,     1,     1], type =    f16, converting to q8_0 .. size =   252.00 MiB ->   133.88 MiB
[  34/1147]                blk.2.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  35/1147]          blk.2.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  36/1147]           blk.2.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.2.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[  37/1147]               blk.2.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.2.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[  38/1147]                blk.2.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.2.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.2.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  39/1147]                blk.2.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.2.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  40/1147]             blk.2.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.2.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[  41/1147]           blk.2.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  42/1147]                blk.2.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.2.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[  43/1147]                blk.2.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.2.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[  44/1147]               blk.3.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[  45/1147]            blk.3.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[  46/1147]          blk.3.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.3.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[  47/1147]          blk.3.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.3.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[  48/1147]            blk.3.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.3.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[  49/1147]          blk.3.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  50/1147]           blk.3.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.3.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[  51/1147]               blk.3.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.3.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[  52/1147]                blk.3.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.3.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.3.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  53/1147]                blk.3.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.3.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  54/1147]             blk.3.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.3.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[  55/1147]           blk.3.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  56/1147]                blk.3.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.3.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[  57/1147]                blk.3.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.3.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[  58/1147]               blk.3.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  59/1147]           blk.3.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type q5_K for tensor blk.3.ffn_down_exps.weight
converting to q5_K .. size =  7168.00 MiB ->  2464.00 MiB
[  60/1147]           blk.3.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.3.ffn_gate_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[  61/1147]             blk.3.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.3.ffn_up_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[  62/1147]                blk.3.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  63/1147]               blk.4.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[  64/1147]            blk.4.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[  65/1147]          blk.4.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.4.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[  66/1147]          blk.4.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.4.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[  67/1147]            blk.4.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.4.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[  68/1147]          blk.4.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  69/1147]           blk.4.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.4.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[  70/1147]               blk.4.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.4.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[  71/1147]                blk.4.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.4.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.4.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  72/1147]                blk.4.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.4.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  73/1147]             blk.4.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.4.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[  74/1147]           blk.4.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  75/1147]                blk.4.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.4.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[  76/1147]                blk.4.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.4.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[  77/1147]               blk.4.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  78/1147]           blk.4.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type q5_K for tensor blk.4.ffn_down_exps.weight
converting to q5_K .. size =  7168.00 MiB ->  2464.00 MiB
[  79/1147]           blk.4.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.4.ffn_gate_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[  80/1147]             blk.4.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.4.ffn_up_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[  81/1147]                blk.4.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  82/1147]          blk.5.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  83/1147]           blk.5.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.5.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[  84/1147]               blk.5.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.5.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[  85/1147]                blk.5.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.5.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.5.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  86/1147]                blk.5.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.5.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[  87/1147]             blk.5.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.5.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[  88/1147]           blk.5.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  89/1147]                blk.5.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.5.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[  90/1147]                blk.5.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.5.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[  91/1147]               blk.5.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[  92/1147]            blk.5.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[  93/1147]          blk.5.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.5.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[  94/1147]          blk.5.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.5.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[  95/1147]            blk.5.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.5.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[  96/1147]               blk.5.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  97/1147]           blk.5.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type q5_K for tensor blk.5.ffn_down_exps.weight
converting to q5_K .. size =  7168.00 MiB ->  2464.00 MiB
[  98/1147]           blk.5.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.5.ffn_gate_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[  99/1147]             blk.5.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.5.ffn_up_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 100/1147]                blk.5.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 101/1147]               blk.6.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 102/1147]            blk.6.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 103/1147]          blk.6.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.6.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 104/1147]          blk.6.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.6.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 105/1147]            blk.6.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.6.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 106/1147]          blk.6.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 107/1147]           blk.6.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.6.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 108/1147]               blk.6.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.6.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 109/1147]                blk.6.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.6.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.6.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 110/1147]                blk.6.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.6.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 111/1147]             blk.6.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.6.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 112/1147]           blk.6.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 113/1147]                blk.6.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.6.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 114/1147]                blk.6.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.6.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 115/1147]               blk.6.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 116/1147]           blk.6.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type q5_K for tensor blk.6.ffn_down_exps.weight
converting to q5_K .. size =  7168.00 MiB ->  2464.00 MiB
[ 117/1147]           blk.6.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.6.ffn_gate_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 118/1147]             blk.6.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.6.ffn_up_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 119/1147]                blk.6.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 120/1147]               blk.7.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 121/1147]            blk.7.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 122/1147]          blk.7.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.7.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 123/1147]          blk.7.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.7.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 124/1147]            blk.7.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.7.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 125/1147]          blk.7.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 126/1147]           blk.7.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.7.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 127/1147]               blk.7.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.7.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 128/1147]                blk.7.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.7.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.7.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 129/1147]                blk.7.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.7.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 130/1147]             blk.7.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.7.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 131/1147]           blk.7.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 132/1147]                blk.7.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.7.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 133/1147]                blk.7.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.7.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 134/1147]               blk.7.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 135/1147]           blk.7.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type q5_K for tensor blk.7.ffn_down_exps.weight
converting to q5_K .. size =  7168.00 MiB ->  2464.00 MiB
[ 136/1147]           blk.7.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.7.ffn_gate_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 137/1147]             blk.7.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.7.ffn_up_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 138/1147]                blk.7.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 139/1147]               blk.8.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 140/1147]            blk.8.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 141/1147]          blk.8.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.8.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 142/1147]          blk.8.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.8.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 143/1147]            blk.8.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.8.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 144/1147]          blk.8.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 145/1147]           blk.8.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.8.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 146/1147]               blk.8.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.8.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 147/1147]                blk.8.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.8.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.8.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 148/1147]                blk.8.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.8.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 149/1147]             blk.8.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.8.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 150/1147]           blk.8.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 151/1147]                blk.8.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.8.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 152/1147]                blk.8.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.8.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 153/1147]               blk.8.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 154/1147]           blk.8.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type q5_K for tensor blk.8.ffn_down_exps.weight
converting to q5_K .. size =  7168.00 MiB ->  2464.00 MiB
[ 155/1147]           blk.8.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.8.ffn_gate_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 156/1147]             blk.8.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.8.ffn_up_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 157/1147]                blk.8.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 158/1147]               blk.9.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 159/1147]            blk.9.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 160/1147]          blk.9.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.9.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 161/1147]          blk.9.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.9.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 162/1147]            blk.9.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.9.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 163/1147]          blk.9.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 164/1147]           blk.9.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.9.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 165/1147]               blk.9.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.9.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 166/1147]                blk.9.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.9.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.9.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 167/1147]                blk.9.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.9.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 168/1147]             blk.9.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.9.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 169/1147]           blk.9.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 170/1147]                blk.9.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.9.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 171/1147]                blk.9.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.9.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 172/1147]              blk.10.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 173/1147]           blk.10.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 174/1147]         blk.10.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.10.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 175/1147]         blk.10.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.10.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 176/1147]           blk.10.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.10.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 177/1147]         blk.10.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 178/1147]          blk.10.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.10.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 179/1147]              blk.10.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.10.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 180/1147]               blk.10.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.10.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.10.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 181/1147]               blk.10.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.10.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 182/1147]            blk.10.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.10.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 183/1147]          blk.10.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 184/1147]               blk.10.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.10.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 185/1147]               blk.10.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.10.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 186/1147]               blk.9.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 187/1147]           blk.9.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.9.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 188/1147]           blk.9.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.9.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 189/1147]             blk.9.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.9.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 190/1147]                blk.9.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 191/1147]              blk.10.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 192/1147]          blk.10.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.10.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 193/1147]          blk.10.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.10.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 194/1147]            blk.10.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.10.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 195/1147]               blk.10.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 196/1147]              blk.11.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 197/1147]           blk.11.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 198/1147]         blk.11.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.11.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 199/1147]         blk.11.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.11.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 200/1147]           blk.11.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.11.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 201/1147]         blk.11.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 202/1147]          blk.11.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.11.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 203/1147]              blk.11.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.11.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 204/1147]               blk.11.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.11.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.11.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 205/1147]               blk.11.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.11.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 206/1147]            blk.11.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.11.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 207/1147]          blk.11.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 208/1147]               blk.11.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.11.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 209/1147]               blk.11.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.11.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 210/1147]              blk.11.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 211/1147]          blk.11.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.11.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 212/1147]          blk.11.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.11.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 213/1147]            blk.11.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.11.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 214/1147]               blk.11.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 215/1147]              blk.12.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 216/1147]           blk.12.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 217/1147]         blk.12.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.12.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 218/1147]         blk.12.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.12.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 219/1147]           blk.12.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.12.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 220/1147]         blk.12.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 221/1147]          blk.12.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.12.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 222/1147]              blk.12.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.12.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 223/1147]               blk.12.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.12.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.12.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 224/1147]               blk.12.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.12.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 225/1147]            blk.12.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.12.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 226/1147]          blk.12.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 227/1147]               blk.12.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.12.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 228/1147]               blk.12.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.12.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 229/1147]              blk.12.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 230/1147]          blk.12.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.12.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 231/1147]          blk.12.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.12.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 232/1147]            blk.12.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.12.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 233/1147]               blk.12.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 234/1147]              blk.13.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 235/1147]           blk.13.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 236/1147]         blk.13.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.13.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 237/1147]         blk.13.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.13.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 238/1147]           blk.13.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.13.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 239/1147]         blk.13.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 240/1147]          blk.13.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.13.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 241/1147]              blk.13.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.13.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 242/1147]               blk.13.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.13.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.13.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 243/1147]               blk.13.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.13.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 244/1147]            blk.13.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.13.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 245/1147]          blk.13.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 246/1147]               blk.13.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.13.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 247/1147]               blk.13.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.13.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 248/1147]              blk.13.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 249/1147]          blk.13.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.13.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 250/1147]          blk.13.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.13.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 251/1147]            blk.13.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.13.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 252/1147]               blk.13.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 253/1147]              blk.14.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 254/1147]           blk.14.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 255/1147]         blk.14.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.14.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 256/1147]         blk.14.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.14.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 257/1147]           blk.14.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.14.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 258/1147]         blk.14.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 259/1147]          blk.14.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.14.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 260/1147]              blk.14.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.14.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 261/1147]               blk.14.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.14.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.14.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 262/1147]               blk.14.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.14.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 263/1147]            blk.14.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.14.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 264/1147]          blk.14.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 265/1147]               blk.14.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.14.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 266/1147]               blk.14.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.14.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 267/1147]              blk.14.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 268/1147]          blk.14.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.14.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 269/1147]          blk.14.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.14.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 270/1147]            blk.14.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.14.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 271/1147]               blk.14.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 272/1147]              blk.15.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 273/1147]           blk.15.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 274/1147]         blk.15.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.15.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 275/1147]         blk.15.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.15.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 276/1147]           blk.15.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.15.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 277/1147]         blk.15.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 278/1147]          blk.15.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.15.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 279/1147]              blk.15.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.15.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 280/1147]               blk.15.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.15.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.15.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 281/1147]               blk.15.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.15.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 282/1147]            blk.15.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.15.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 283/1147]          blk.15.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 284/1147]               blk.15.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.15.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 285/1147]               blk.15.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.15.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 286/1147]              blk.15.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 287/1147]          blk.15.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.15.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 288/1147]          blk.15.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.15.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 289/1147]            blk.15.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.15.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 290/1147]               blk.15.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 291/1147]              blk.16.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 292/1147]           blk.16.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 293/1147]         blk.16.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.16.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 294/1147]         blk.16.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.16.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 295/1147]           blk.16.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.16.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 296/1147]         blk.16.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 297/1147]          blk.16.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.16.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 298/1147]              blk.16.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.16.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 299/1147]               blk.16.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.16.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.16.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 300/1147]               blk.16.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.16.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 301/1147]            blk.16.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.16.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 302/1147]          blk.16.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 303/1147]               blk.16.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.16.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 304/1147]               blk.16.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.16.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 305/1147]              blk.16.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 306/1147]          blk.16.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.16.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 307/1147]          blk.16.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.16.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 308/1147]            blk.16.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.16.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 309/1147]               blk.16.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 310/1147]              blk.17.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 311/1147]           blk.17.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 312/1147]         blk.17.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.17.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 313/1147]         blk.17.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.17.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 314/1147]           blk.17.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.17.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 315/1147]         blk.17.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 316/1147]          blk.17.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.17.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 317/1147]              blk.17.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.17.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 318/1147]               blk.17.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.17.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.17.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 319/1147]               blk.17.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.17.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 320/1147]            blk.17.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.17.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 321/1147]          blk.17.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 322/1147]               blk.17.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.17.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 323/1147]               blk.17.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.17.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 324/1147]              blk.17.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 325/1147]          blk.17.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.17.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 326/1147]          blk.17.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.17.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 327/1147]            blk.17.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.17.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 328/1147]               blk.17.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 329/1147]              blk.18.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 330/1147]           blk.18.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 331/1147]         blk.18.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.18.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 332/1147]         blk.18.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.18.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 333/1147]           blk.18.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.18.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 334/1147]         blk.18.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 335/1147]          blk.18.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.18.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 336/1147]              blk.18.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.18.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 337/1147]               blk.18.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.18.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.18.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 338/1147]               blk.18.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.18.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 339/1147]            blk.18.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.18.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 340/1147]          blk.18.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 341/1147]               blk.18.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.18.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 342/1147]               blk.18.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.18.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 343/1147]              blk.18.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 344/1147]          blk.18.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.18.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 345/1147]          blk.18.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.18.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 346/1147]            blk.18.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.18.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 347/1147]               blk.18.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 348/1147]              blk.19.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 349/1147]           blk.19.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 350/1147]         blk.19.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.19.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 351/1147]         blk.19.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.19.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 352/1147]           blk.19.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.19.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 353/1147]         blk.19.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 354/1147]          blk.19.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.19.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 355/1147]              blk.19.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.19.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 356/1147]               blk.19.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.19.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.19.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 357/1147]               blk.19.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.19.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 358/1147]            blk.19.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.19.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 359/1147]          blk.19.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 360/1147]               blk.19.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.19.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 361/1147]               blk.19.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.19.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 362/1147]              blk.19.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 363/1147]          blk.19.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.19.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 364/1147]          blk.19.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.19.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 365/1147]            blk.19.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.19.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 366/1147]               blk.19.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 367/1147]              blk.20.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 368/1147]           blk.20.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 369/1147]         blk.20.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.20.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 370/1147]         blk.20.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.20.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 371/1147]           blk.20.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.20.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 372/1147]         blk.20.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 373/1147]          blk.20.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.20.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 374/1147]              blk.20.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.20.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 375/1147]               blk.20.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.20.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.20.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 376/1147]               blk.20.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.20.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 377/1147]            blk.20.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.20.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 378/1147]          blk.20.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 379/1147]               blk.20.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.20.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 380/1147]               blk.20.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.20.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 381/1147]              blk.20.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 382/1147]          blk.20.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.20.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 383/1147]          blk.20.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.20.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 384/1147]            blk.20.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.20.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 385/1147]               blk.20.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 386/1147]              blk.21.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 387/1147]           blk.21.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 388/1147]         blk.21.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.21.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 389/1147]         blk.21.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.21.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 390/1147]           blk.21.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.21.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 391/1147]         blk.21.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 392/1147]          blk.21.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.21.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 393/1147]              blk.21.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.21.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 394/1147]               blk.21.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.21.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.21.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 395/1147]               blk.21.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.21.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 396/1147]            blk.21.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.21.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 397/1147]          blk.21.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 398/1147]               blk.21.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.21.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 399/1147]               blk.21.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.21.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 400/1147]              blk.21.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 401/1147]          blk.21.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.21.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 402/1147]          blk.21.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.21.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 403/1147]            blk.21.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.21.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 404/1147]               blk.21.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 405/1147]              blk.22.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 406/1147]           blk.22.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 407/1147]         blk.22.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.22.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 408/1147]         blk.22.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.22.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 409/1147]           blk.22.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.22.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 410/1147]         blk.22.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 411/1147]          blk.22.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.22.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 412/1147]              blk.22.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.22.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 413/1147]               blk.22.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.22.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.22.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 414/1147]               blk.22.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.22.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 415/1147]            blk.22.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.22.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 416/1147]          blk.22.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 417/1147]               blk.22.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.22.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 418/1147]               blk.22.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.22.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 419/1147]              blk.22.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 420/1147]          blk.22.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.22.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 421/1147]          blk.22.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.22.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 422/1147]            blk.22.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.22.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 423/1147]               blk.22.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 424/1147]              blk.23.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 425/1147]           blk.23.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 426/1147]         blk.23.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.23.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 427/1147]         blk.23.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.23.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 428/1147]           blk.23.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.23.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 429/1147]         blk.23.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 430/1147]          blk.23.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.23.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 431/1147]              blk.23.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.23.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 432/1147]               blk.23.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.23.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.23.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 433/1147]               blk.23.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.23.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 434/1147]            blk.23.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.23.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 435/1147]          blk.23.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 436/1147]               blk.23.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.23.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 437/1147]               blk.23.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.23.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 438/1147]              blk.23.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 439/1147]          blk.23.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.23.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 440/1147]          blk.23.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.23.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 441/1147]            blk.23.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.23.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 442/1147]               blk.23.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 443/1147]              blk.24.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 444/1147]           blk.24.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 445/1147]         blk.24.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.24.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 446/1147]         blk.24.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.24.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 447/1147]           blk.24.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.24.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 448/1147]         blk.24.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 449/1147]          blk.24.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.24.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 450/1147]              blk.24.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.24.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 451/1147]               blk.24.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.24.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.24.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 452/1147]               blk.24.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.24.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 453/1147]            blk.24.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.24.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 454/1147]          blk.24.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 455/1147]               blk.24.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.24.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 456/1147]               blk.24.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.24.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 457/1147]              blk.24.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 458/1147]          blk.24.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.24.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 459/1147]          blk.24.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.24.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 460/1147]            blk.24.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.24.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 461/1147]               blk.24.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 462/1147]              blk.25.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 463/1147]           blk.25.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 464/1147]         blk.25.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.25.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 465/1147]         blk.25.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.25.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 466/1147]           blk.25.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.25.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 467/1147]         blk.25.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 468/1147]          blk.25.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.25.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 469/1147]              blk.25.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.25.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 470/1147]               blk.25.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.25.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.25.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 471/1147]               blk.25.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.25.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 472/1147]            blk.25.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.25.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 473/1147]          blk.25.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 474/1147]               blk.25.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.25.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 475/1147]               blk.25.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.25.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 476/1147]              blk.25.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 477/1147]          blk.25.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.25.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 478/1147]          blk.25.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.25.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 479/1147]            blk.25.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.25.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 480/1147]               blk.25.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 481/1147]              blk.26.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 482/1147]           blk.26.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 483/1147]         blk.26.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.26.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 484/1147]         blk.26.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.26.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 485/1147]           blk.26.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.26.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 486/1147]         blk.26.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 487/1147]          blk.26.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.26.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 488/1147]              blk.26.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.26.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 489/1147]               blk.26.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.26.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.26.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 490/1147]               blk.26.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.26.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 491/1147]            blk.26.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.26.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 492/1147]          blk.26.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 493/1147]               blk.26.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.26.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 494/1147]               blk.26.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.26.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 495/1147]              blk.26.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 496/1147]          blk.26.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.26.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 497/1147]          blk.26.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.26.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 498/1147]            blk.26.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.26.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 499/1147]               blk.26.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 500/1147]              blk.27.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 501/1147]           blk.27.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 502/1147]         blk.27.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.27.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 503/1147]         blk.27.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.27.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 504/1147]           blk.27.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.27.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 505/1147]         blk.27.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 506/1147]          blk.27.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.27.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 507/1147]              blk.27.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.27.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 508/1147]               blk.27.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.27.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.27.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 509/1147]               blk.27.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.27.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 510/1147]            blk.27.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.27.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 511/1147]          blk.27.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 512/1147]               blk.27.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.27.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 513/1147]               blk.27.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.27.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 514/1147]              blk.27.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 515/1147]          blk.27.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.27.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 516/1147]          blk.27.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.27.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 517/1147]            blk.27.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.27.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 518/1147]               blk.27.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 519/1147]              blk.28.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 520/1147]           blk.28.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 521/1147]         blk.28.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.28.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 522/1147]         blk.28.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.28.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 523/1147]           blk.28.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.28.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 524/1147]         blk.28.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 525/1147]          blk.28.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.28.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 526/1147]              blk.28.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.28.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 527/1147]               blk.28.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.28.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.28.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 528/1147]               blk.28.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.28.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 529/1147]            blk.28.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.28.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 530/1147]          blk.28.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 531/1147]               blk.28.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.28.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 532/1147]               blk.28.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.28.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 533/1147]              blk.28.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 534/1147]          blk.28.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.28.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 535/1147]          blk.28.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.28.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 536/1147]            blk.28.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.28.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 537/1147]               blk.28.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 538/1147]              blk.29.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 539/1147]           blk.29.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 540/1147]         blk.29.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.29.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 541/1147]         blk.29.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.29.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 542/1147]           blk.29.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.29.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 543/1147]         blk.29.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 544/1147]          blk.29.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.29.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 545/1147]              blk.29.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.29.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 546/1147]               blk.29.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.29.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.29.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 547/1147]               blk.29.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.29.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 548/1147]            blk.29.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.29.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 549/1147]          blk.29.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 550/1147]               blk.29.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.29.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 551/1147]               blk.29.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.29.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 552/1147]              blk.29.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 553/1147]          blk.29.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.29.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 554/1147]          blk.29.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.29.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 555/1147]            blk.29.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.29.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 556/1147]               blk.29.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 557/1147]              blk.30.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 558/1147]           blk.30.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 559/1147]         blk.30.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.30.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 560/1147]         blk.30.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.30.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 561/1147]           blk.30.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.30.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 562/1147]         blk.30.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 563/1147]          blk.30.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.30.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 564/1147]              blk.30.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.30.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 565/1147]               blk.30.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.30.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.30.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 566/1147]               blk.30.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.30.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 567/1147]            blk.30.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.30.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 568/1147]          blk.30.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 569/1147]               blk.30.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.30.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 570/1147]               blk.30.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.30.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 571/1147]              blk.30.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 572/1147]          blk.30.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.30.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 573/1147]          blk.30.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.30.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 574/1147]            blk.30.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.30.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 575/1147]               blk.30.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 576/1147]              blk.31.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 577/1147]           blk.31.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 578/1147]         blk.31.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.31.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 579/1147]         blk.31.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.31.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 580/1147]           blk.31.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.31.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 581/1147]         blk.31.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 582/1147]          blk.31.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.31.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 583/1147]              blk.31.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.31.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 584/1147]               blk.31.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.31.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.31.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 585/1147]               blk.31.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.31.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 586/1147]            blk.31.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.31.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 587/1147]          blk.31.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 588/1147]               blk.31.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.31.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 589/1147]               blk.31.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.31.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 590/1147]              blk.31.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 591/1147]          blk.31.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.31.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 592/1147]          blk.31.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.31.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 593/1147]            blk.31.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.31.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 594/1147]               blk.31.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 595/1147]              blk.32.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 596/1147]           blk.32.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 597/1147]         blk.32.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.32.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 598/1147]         blk.32.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.32.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 599/1147]           blk.32.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.32.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 600/1147]         blk.32.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 601/1147]          blk.32.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.32.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 602/1147]              blk.32.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.32.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 603/1147]               blk.32.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.32.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.32.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 604/1147]               blk.32.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.32.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 605/1147]            blk.32.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.32.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 606/1147]          blk.32.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 607/1147]               blk.32.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.32.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 608/1147]               blk.32.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.32.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 609/1147]              blk.32.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 610/1147]          blk.32.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.32.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 611/1147]          blk.32.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.32.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 612/1147]            blk.32.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.32.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 613/1147]               blk.32.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 614/1147]              blk.33.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 615/1147]           blk.33.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 616/1147]         blk.33.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.33.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 617/1147]         blk.33.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.33.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 618/1147]           blk.33.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.33.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 619/1147]         blk.33.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 620/1147]          blk.33.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.33.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 621/1147]              blk.33.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.33.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 622/1147]               blk.33.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.33.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.33.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 623/1147]               blk.33.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.33.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 624/1147]            blk.33.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.33.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 625/1147]          blk.33.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 626/1147]               blk.33.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.33.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 627/1147]               blk.33.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.33.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 628/1147]              blk.33.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 629/1147]          blk.33.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.33.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 630/1147]          blk.33.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.33.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 631/1147]            blk.33.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.33.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 632/1147]               blk.33.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 633/1147]              blk.34.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 634/1147]           blk.34.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 635/1147]         blk.34.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.34.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 636/1147]         blk.34.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.34.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 637/1147]           blk.34.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.34.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 638/1147]         blk.34.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 639/1147]          blk.34.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.34.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 640/1147]              blk.34.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.34.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 641/1147]               blk.34.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.34.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.34.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 642/1147]               blk.34.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.34.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 643/1147]            blk.34.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.34.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 644/1147]          blk.34.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 645/1147]               blk.34.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.34.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 646/1147]               blk.34.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.34.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 647/1147]              blk.34.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 648/1147]          blk.34.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.34.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 649/1147]          blk.34.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.34.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 650/1147]            blk.34.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.34.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 651/1147]               blk.34.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 652/1147]              blk.35.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 653/1147]           blk.35.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 654/1147]         blk.35.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.35.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 655/1147]         blk.35.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.35.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 656/1147]           blk.35.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.35.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 657/1147]         blk.35.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 658/1147]          blk.35.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.35.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 659/1147]              blk.35.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.35.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 660/1147]               blk.35.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.35.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.35.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 661/1147]               blk.35.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.35.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 662/1147]            blk.35.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.35.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 663/1147]          blk.35.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 664/1147]               blk.35.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.35.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 665/1147]               blk.35.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.35.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 666/1147]              blk.35.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 667/1147]          blk.35.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.35.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 668/1147]          blk.35.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.35.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 669/1147]            blk.35.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.35.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 670/1147]               blk.35.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 671/1147]              blk.36.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 672/1147]           blk.36.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 673/1147]         blk.36.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.36.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 674/1147]         blk.36.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.36.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 675/1147]           blk.36.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.36.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 676/1147]         blk.36.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 677/1147]          blk.36.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.36.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 678/1147]              blk.36.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.36.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 679/1147]               blk.36.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.36.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.36.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 680/1147]               blk.36.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.36.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 681/1147]            blk.36.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.36.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 682/1147]          blk.36.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 683/1147]               blk.36.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.36.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 684/1147]               blk.36.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.36.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 685/1147]              blk.36.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 686/1147]          blk.36.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.36.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 687/1147]          blk.36.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.36.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 688/1147]            blk.36.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.36.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 689/1147]               blk.36.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 690/1147]              blk.37.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 691/1147]           blk.37.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 692/1147]         blk.37.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.37.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 693/1147]         blk.37.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.37.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 694/1147]           blk.37.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.37.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 695/1147]         blk.37.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 696/1147]          blk.37.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.37.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 697/1147]              blk.37.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.37.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 698/1147]               blk.37.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.37.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.37.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 699/1147]               blk.37.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.37.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 700/1147]            blk.37.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.37.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 701/1147]          blk.37.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 702/1147]               blk.37.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.37.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 703/1147]               blk.37.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.37.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 704/1147]              blk.37.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 705/1147]          blk.37.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.37.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 706/1147]          blk.37.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.37.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 707/1147]            blk.37.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.37.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 708/1147]               blk.37.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 709/1147]              blk.38.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 710/1147]           blk.38.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 711/1147]         blk.38.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.38.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 712/1147]         blk.38.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.38.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 713/1147]           blk.38.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.38.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 714/1147]         blk.38.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 715/1147]          blk.38.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.38.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 716/1147]              blk.38.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.38.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 717/1147]               blk.38.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.38.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.38.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 718/1147]               blk.38.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.38.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 719/1147]            blk.38.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.38.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 720/1147]          blk.38.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 721/1147]               blk.38.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.38.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 722/1147]               blk.38.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.38.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 723/1147]              blk.38.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 724/1147]          blk.38.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.38.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 725/1147]          blk.38.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.38.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 726/1147]            blk.38.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.38.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 727/1147]               blk.38.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 728/1147]              blk.39.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 729/1147]           blk.39.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 730/1147]         blk.39.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.39.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 731/1147]         blk.39.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.39.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 732/1147]           blk.39.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.39.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 733/1147]         blk.39.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 734/1147]          blk.39.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.39.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 735/1147]              blk.39.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.39.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 736/1147]               blk.39.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.39.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.39.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 737/1147]               blk.39.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.39.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 738/1147]            blk.39.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.39.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 739/1147]          blk.39.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 740/1147]               blk.39.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.39.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 741/1147]               blk.39.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.39.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 742/1147]              blk.39.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 743/1147]          blk.39.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.39.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 744/1147]          blk.39.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.39.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 745/1147]            blk.39.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.39.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 746/1147]               blk.39.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 747/1147]              blk.40.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 748/1147]           blk.40.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 749/1147]         blk.40.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.40.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 750/1147]         blk.40.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.40.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 751/1147]           blk.40.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.40.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 752/1147]         blk.40.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 753/1147]          blk.40.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.40.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 754/1147]              blk.40.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.40.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 755/1147]               blk.40.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.40.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.40.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 756/1147]               blk.40.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.40.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 757/1147]            blk.40.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.40.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 758/1147]          blk.40.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 759/1147]               blk.40.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.40.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 760/1147]               blk.40.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.40.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 761/1147]              blk.40.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 762/1147]          blk.40.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.40.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 763/1147]          blk.40.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.40.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 764/1147]            blk.40.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.40.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 765/1147]               blk.40.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 766/1147]              blk.41.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 767/1147]           blk.41.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 768/1147]         blk.41.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.41.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 769/1147]         blk.41.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.41.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 770/1147]           blk.41.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.41.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 771/1147]         blk.41.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 772/1147]          blk.41.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.41.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 773/1147]              blk.41.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.41.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 774/1147]               blk.41.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.41.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.41.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 775/1147]               blk.41.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.41.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 776/1147]            blk.41.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.41.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 777/1147]          blk.41.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 778/1147]               blk.41.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.41.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 779/1147]               blk.41.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.41.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 780/1147]              blk.41.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 781/1147]          blk.41.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.41.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 782/1147]          blk.41.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.41.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 783/1147]            blk.41.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.41.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 784/1147]               blk.41.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 785/1147]              blk.42.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 786/1147]           blk.42.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 787/1147]         blk.42.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.42.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 788/1147]         blk.42.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.42.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 789/1147]           blk.42.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.42.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 790/1147]         blk.42.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 791/1147]          blk.42.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.42.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 792/1147]              blk.42.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.42.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 793/1147]               blk.42.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.42.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.42.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 794/1147]               blk.42.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.42.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 795/1147]            blk.42.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.42.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 796/1147]          blk.42.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 797/1147]               blk.42.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.42.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 798/1147]               blk.42.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.42.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 799/1147]              blk.42.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 800/1147]          blk.42.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.42.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 801/1147]          blk.42.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.42.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 802/1147]            blk.42.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.42.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 803/1147]               blk.42.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 804/1147]              blk.43.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 805/1147]           blk.43.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 806/1147]         blk.43.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.43.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 807/1147]         blk.43.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.43.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 808/1147]           blk.43.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.43.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 809/1147]         blk.43.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 810/1147]          blk.43.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.43.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 811/1147]              blk.43.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.43.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 812/1147]               blk.43.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.43.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.43.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 813/1147]               blk.43.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.43.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 814/1147]            blk.43.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.43.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 815/1147]          blk.43.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 816/1147]               blk.43.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.43.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 817/1147]               blk.43.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.43.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 818/1147]              blk.43.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 819/1147]          blk.43.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.43.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 820/1147]          blk.43.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.43.ffn_gate_exps.weight

Message from syslogd@Kingdom at Mar 10 21:23:25 ...
 kernel:[Hardware Error]: Corrected error, no action required.

Message from syslogd@Kingdom at Mar 10 21:23:25 ...
 kernel:[Hardware Error]: CPU:1 (19:1:1) MC18_STATUS[Over|CE|MiscV|AddrV|-|-|SyndV|CECC|-|-|-]: 0xdc2040000000011b

Message from syslogd@Kingdom at Mar 10 21:23:25 ...
 kernel:[Hardware Error]: Error Addr: 0x00000000a2a61fc0

Message from syslogd@Kingdom at Mar 10 21:23:25 ...
 kernel:[Hardware Error]: PPIN: 0x02b6b32442ad40cd

Message from syslogd@Kingdom at Mar 10 21:23:25 ...
 kernel:[Hardware Error]: IPID: 0x0000009600350f00, Syndrome: 0x51c900040a800101

Message from syslogd@Kingdom at Mar 10 21:23:25 ...
 kernel:[Hardware Error]: Unified Memory Controller Ext. Error Code: 0, DRAM ECC error.

Message from syslogd@Kingdom at Mar 10 21:23:25 ...
 kernel:[Hardware Error]: cache level: L3/GEN, tx: GEN, mem-tx: RD
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 821/1147]            blk.43.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.43.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 822/1147]               blk.43.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 823/1147]              blk.44.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 824/1147]           blk.44.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 825/1147]         blk.44.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.44.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 826/1147]         blk.44.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.44.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 827/1147]           blk.44.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.44.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 828/1147]         blk.44.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 829/1147]          blk.44.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.44.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 830/1147]              blk.44.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.44.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 831/1147]               blk.44.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.44.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.44.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 832/1147]               blk.44.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.44.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 833/1147]            blk.44.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.44.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 834/1147]          blk.44.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 835/1147]               blk.44.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.44.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 836/1147]               blk.44.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.44.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 837/1147]              blk.44.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 838/1147]          blk.44.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.44.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 839/1147]          blk.44.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.44.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 840/1147]            blk.44.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.44.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 841/1147]               blk.44.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 842/1147]              blk.45.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 843/1147]           blk.45.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 844/1147]         blk.45.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.45.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 845/1147]         blk.45.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.45.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 846/1147]           blk.45.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.45.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 847/1147]         blk.45.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 848/1147]          blk.45.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.45.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 849/1147]              blk.45.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.45.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 850/1147]               blk.45.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.45.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.45.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 851/1147]               blk.45.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.45.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 852/1147]            blk.45.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.45.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 853/1147]          blk.45.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 854/1147]               blk.45.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.45.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 855/1147]               blk.45.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.45.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 856/1147]              blk.45.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 857/1147]          blk.45.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.45.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB

Message from syslogd@Kingdom at Mar 10 21:28:53 ...
 kernel:[Hardware Error]: Corrected error, no action required.

Message from syslogd@Kingdom at Mar 10 21:28:53 ...
 kernel:[Hardware Error]: CPU:1 (19:1:1) MC18_STATUS[Over|CE|MiscV|AddrV|-|-|SyndV|CECC|-|-|-]: 0xdc2040000000011b

Message from syslogd@Kingdom at Mar 10 21:28:53 ...
 kernel:[Hardware Error]: Error Addr: 0x00000000a2a61fc0

Message from syslogd@Kingdom at Mar 10 21:28:53 ...
 kernel:[Hardware Error]: PPIN: 0x02b6b32442ad40cd

Message from syslogd@Kingdom at Mar 10 21:28:53 ...
 kernel:[Hardware Error]: IPID: 0x0000009600350f00, Syndrome: 0x51c900040a800101

Message from syslogd@Kingdom at Mar 10 21:28:53 ...
 kernel:[Hardware Error]: Unified Memory Controller Ext. Error Code: 0, DRAM ECC error.

Message from syslogd@Kingdom at Mar 10 21:28:53 ...
 kernel:[Hardware Error]: cache level: L3/GEN, tx: GEN, mem-tx: RD
[ 858/1147]          blk.45.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.45.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 859/1147]            blk.45.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.45.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 860/1147]               blk.45.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 861/1147]              blk.46.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 862/1147]           blk.46.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 863/1147]         blk.46.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.46.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 864/1147]         blk.46.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.46.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 865/1147]           blk.46.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.46.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 866/1147]         blk.46.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 867/1147]          blk.46.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.46.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 868/1147]              blk.46.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.46.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 869/1147]               blk.46.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.46.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.46.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 870/1147]               blk.46.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.46.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 871/1147]            blk.46.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.46.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 872/1147]          blk.46.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 873/1147]               blk.46.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.46.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 874/1147]               blk.46.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.46.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 875/1147]              blk.46.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 876/1147]          blk.46.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.46.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 877/1147]          blk.46.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.46.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 878/1147]            blk.46.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.46.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 879/1147]               blk.46.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 880/1147]              blk.47.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 881/1147]           blk.47.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 882/1147]         blk.47.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.47.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 883/1147]         blk.47.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.47.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 884/1147]           blk.47.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.47.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 885/1147]         blk.47.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 886/1147]          blk.47.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.47.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 887/1147]              blk.47.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.47.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 888/1147]               blk.47.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.47.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.47.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 889/1147]               blk.47.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.47.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 890/1147]            blk.47.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.47.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 891/1147]          blk.47.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 892/1147]               blk.47.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.47.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 893/1147]               blk.47.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.47.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 894/1147]              blk.47.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 895/1147]          blk.47.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.47.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 896/1147]          blk.47.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.47.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 897/1147]            blk.47.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.47.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 898/1147]               blk.47.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 899/1147]              blk.48.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 900/1147]           blk.48.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 901/1147]         blk.48.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.48.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 902/1147]         blk.48.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.48.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 903/1147]           blk.48.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.48.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 904/1147]         blk.48.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 905/1147]          blk.48.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.48.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 906/1147]              blk.48.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.48.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 907/1147]               blk.48.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.48.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.48.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 908/1147]               blk.48.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.48.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 909/1147]            blk.48.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.48.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 910/1147]          blk.48.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 911/1147]               blk.48.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.48.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 912/1147]               blk.48.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.48.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 913/1147]              blk.48.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 914/1147]          blk.48.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.48.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 915/1147]          blk.48.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.48.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 916/1147]            blk.48.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.48.ffn_up_exps.weight

Message from syslogd@Kingdom at Mar 10 21:39:49 ...
 kernel:[Hardware Error]: Corrected error, no action required.

Message from syslogd@Kingdom at Mar 10 21:39:49 ...
 kernel:[Hardware Error]: CPU:1 (19:1:1) MC18_STATUS[Over|CE|MiscV|AddrV|-|-|SyndV|CECC|-|-|-]: 0xdc2040000000011b

Message from syslogd@Kingdom at Mar 10 21:39:49 ...
 kernel:[Hardware Error]: Error Addr: 0x00000000a2a61fc0

Message from syslogd@Kingdom at Mar 10 21:39:49 ...
 kernel:[Hardware Error]: PPIN: 0x02b6b32442ad40cd

Message from syslogd@Kingdom at Mar 10 21:39:49 ...
 kernel:[Hardware Error]: IPID: 0x0000009600350f00, Syndrome: 0x51c900040a800101

Message from syslogd@Kingdom at Mar 10 21:39:49 ...
 kernel:[Hardware Error]: Unified Memory Controller Ext. Error Code: 0, DRAM ECC error.

Message from syslogd@Kingdom at Mar 10 21:39:49 ...
 kernel:[Hardware Error]: cache level: L3/GEN, tx: GEN, mem-tx: RD
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 917/1147]               blk.48.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 918/1147]              blk.49.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 919/1147]           blk.49.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 920/1147]         blk.49.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.49.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 921/1147]         blk.49.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.49.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 922/1147]           blk.49.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.49.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 923/1147]         blk.49.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 924/1147]          blk.49.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.49.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 925/1147]              blk.49.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.49.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 926/1147]               blk.49.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.49.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.49.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 927/1147]               blk.49.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.49.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 928/1147]            blk.49.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.49.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 929/1147]          blk.49.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 930/1147]               blk.49.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.49.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 931/1147]               blk.49.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.49.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 932/1147]              blk.49.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 933/1147]          blk.49.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.49.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 934/1147]          blk.49.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.49.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 935/1147]            blk.49.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.49.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 936/1147]               blk.49.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 937/1147]              blk.50.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 938/1147]           blk.50.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 939/1147]         blk.50.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.50.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 940/1147]         blk.50.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.50.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 941/1147]           blk.50.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.50.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 942/1147]         blk.50.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 943/1147]          blk.50.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.50.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 944/1147]              blk.50.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.50.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 945/1147]               blk.50.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.50.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.50.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 946/1147]               blk.50.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.50.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 947/1147]            blk.50.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.50.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 948/1147]          blk.50.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 949/1147]               blk.50.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.50.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 950/1147]               blk.50.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.50.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 951/1147]              blk.50.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 952/1147]          blk.50.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.50.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 953/1147]          blk.50.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.50.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB

Message from syslogd@Kingdom at Mar 10 21:45:16 ...
 kernel:[Hardware Error]: Corrected error, no action required.

Message from syslogd@Kingdom at Mar 10 21:45:16 ...
 kernel:[Hardware Error]: CPU:1 (19:1:1) MC18_STATUS[Over|CE|MiscV|AddrV|-|-|SyndV|CECC|-|-|-]: 0xdc2040000000011b

Message from syslogd@Kingdom at Mar 10 21:45:16 ...
 kernel:[Hardware Error]: Error Addr: 0x00000000a2a61fc0

Message from syslogd@Kingdom at Mar 10 21:45:16 ...
 kernel:[Hardware Error]: PPIN: 0x02b6b32442ad40cd

Message from syslogd@Kingdom at Mar 10 21:45:16 ...
 kernel:[Hardware Error]: IPID: 0x0000009600350f00, Syndrome: 0x51c900040a800101

Message from syslogd@Kingdom at Mar 10 21:45:16 ...
 kernel:[Hardware Error]: Unified Memory Controller Ext. Error Code: 0, DRAM ECC error.

Message from syslogd@Kingdom at Mar 10 21:45:16 ...
 kernel:[Hardware Error]: cache level: L3/GEN, tx: GEN, mem-tx: RD
[ 954/1147]            blk.50.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.50.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 955/1147]               blk.50.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 956/1147]              blk.51.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 957/1147]           blk.51.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 958/1147]         blk.51.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.51.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 959/1147]         blk.51.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.51.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 960/1147]           blk.51.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.51.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 961/1147]         blk.51.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 962/1147]          blk.51.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.51.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 963/1147]              blk.51.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.51.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 964/1147]               blk.51.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.51.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.51.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 965/1147]               blk.51.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.51.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 966/1147]            blk.51.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.51.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 967/1147]          blk.51.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 968/1147]               blk.51.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.51.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 969/1147]               blk.51.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.51.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 970/1147]              blk.51.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 971/1147]          blk.51.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.51.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 972/1147]          blk.51.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.51.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 973/1147]            blk.51.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.51.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 974/1147]               blk.51.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 975/1147]              blk.52.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 976/1147]           blk.52.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 977/1147]         blk.52.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.52.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 978/1147]         blk.52.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.52.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 979/1147]           blk.52.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.52.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 980/1147]         blk.52.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 981/1147]          blk.52.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.52.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[ 982/1147]              blk.52.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.52.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[ 983/1147]               blk.52.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.52.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.52.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 984/1147]               blk.52.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.52.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[ 985/1147]            blk.52.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.52.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[ 986/1147]          blk.52.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 987/1147]               blk.52.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.52.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[ 988/1147]               blk.52.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.52.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[ 989/1147]              blk.52.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 990/1147]          blk.52.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.52.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[ 991/1147]          blk.52.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.52.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB

Message from syslogd@Kingdom at Mar 10 21:50:44 ...
 kernel:[Hardware Error]: Corrected error, no action required.

Message from syslogd@Kingdom at Mar 10 21:50:44 ...
 kernel:[Hardware Error]: CPU:1 (19:1:1) MC18_STATUS[Over|CE|MiscV|AddrV|-|-|SyndV|CECC|-|-|-]: 0xdc2040000000011b

Message from syslogd@Kingdom at Mar 10 21:50:44 ...
 kernel:[Hardware Error]: Error Addr: 0x00000000a2a61fc0

Message from syslogd@Kingdom at Mar 10 21:50:44 ...
 kernel:[Hardware Error]: PPIN: 0x02b6b32442ad40cd

Message from syslogd@Kingdom at Mar 10 21:50:44 ...
 kernel:[Hardware Error]: IPID: 0x0000009600350f00, Syndrome: 0x51c900040a800101

Message from syslogd@Kingdom at Mar 10 21:50:44 ...
 kernel:[Hardware Error]: Unified Memory Controller Ext. Error Code: 0, DRAM ECC error.

Message from syslogd@Kingdom at Mar 10 21:50:44 ...
 kernel:[Hardware Error]: cache level: L3/GEN, tx: GEN, mem-tx: RD
[ 992/1147]            blk.52.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.52.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[ 993/1147]               blk.52.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 994/1147]              blk.53.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 995/1147]           blk.53.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 996/1147]         blk.53.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.53.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 997/1147]         blk.53.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.53.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 998/1147]           blk.53.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.53.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[ 999/1147]         blk.53.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1000/1147]          blk.53.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.53.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[1001/1147]              blk.53.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.53.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[1002/1147]               blk.53.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.53.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.53.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1003/1147]               blk.53.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.53.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1004/1147]            blk.53.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.53.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[1005/1147]          blk.53.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1006/1147]               blk.53.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.53.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[1007/1147]               blk.53.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.53.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[1008/1147]              blk.53.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1009/1147]          blk.53.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.53.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[1010/1147]          blk.53.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.53.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[1011/1147]            blk.53.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.53.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[1012/1147]               blk.53.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1013/1147]              blk.54.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1014/1147]           blk.54.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1015/1147]         blk.54.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.54.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[1016/1147]         blk.54.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.54.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[1017/1147]           blk.54.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.54.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[1018/1147]         blk.54.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1019/1147]          blk.54.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.54.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[1020/1147]              blk.54.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.54.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[1021/1147]               blk.54.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.54.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.54.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1022/1147]               blk.54.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.54.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1023/1147]            blk.54.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.54.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[1024/1147]          blk.54.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1025/1147]               blk.54.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.54.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[1026/1147]               blk.54.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.54.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[1027/1147]              blk.54.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1028/1147]          blk.54.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.54.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[1029/1147]          blk.54.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.54.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB

Message from syslogd@Kingdom at Mar 10 21:56:12 ...
 kernel:[Hardware Error]: Corrected error, no action required.

Message from syslogd@Kingdom at Mar 10 21:56:12 ...
 kernel:[Hardware Error]: CPU:1 (19:1:1) MC18_STATUS[Over|CE|MiscV|AddrV|-|-|SyndV|CECC|-|-|-]: 0xdc2040000000011b

Message from syslogd@Kingdom at Mar 10 21:56:12 ...
 kernel:[Hardware Error]: Error Addr: 0x00000000a2a61fc0

Message from syslogd@Kingdom at Mar 10 21:56:12 ...
 kernel:[Hardware Error]: PPIN: 0x02b6b32442ad40cd

Message from syslogd@Kingdom at Mar 10 21:56:12 ...
 kernel:[Hardware Error]: IPID: 0x0000009600350f00, Syndrome: 0x51c900040a800101

Message from syslogd@Kingdom at Mar 10 21:56:12 ...
 kernel:[Hardware Error]: Unified Memory Controller Ext. Error Code: 0, DRAM ECC error.

Message from syslogd@Kingdom at Mar 10 21:56:12 ...
 kernel:[Hardware Error]: cache level: L3/GEN, tx: GEN, mem-tx: RD
[1030/1147]            blk.54.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.54.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[1031/1147]               blk.54.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1032/1147]              blk.55.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1033/1147]           blk.55.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1034/1147]         blk.55.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.55.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[1035/1147]         blk.55.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.55.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[1036/1147]           blk.55.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.55.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[1037/1147]         blk.55.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1038/1147]          blk.55.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.55.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[1039/1147]              blk.55.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.55.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[1040/1147]               blk.55.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.55.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.55.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1041/1147]               blk.55.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.55.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1042/1147]            blk.55.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.55.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[1043/1147]          blk.55.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1044/1147]               blk.55.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.55.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[1045/1147]               blk.55.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.55.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[1046/1147]              blk.55.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1047/1147]          blk.55.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.55.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[1048/1147]          blk.55.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.55.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[1049/1147]            blk.55.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.55.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[1050/1147]               blk.55.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1051/1147]              blk.56.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1052/1147]           blk.56.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1053/1147]         blk.56.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.56.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[1054/1147]         blk.56.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.56.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[1055/1147]           blk.56.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.56.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[1056/1147]         blk.56.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1057/1147]          blk.56.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.56.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[1058/1147]              blk.56.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.56.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[1059/1147]               blk.56.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.56.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.56.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1060/1147]               blk.56.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.56.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1061/1147]            blk.56.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.56.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[1062/1147]          blk.56.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1063/1147]               blk.56.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.56.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[1064/1147]               blk.56.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.56.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[1065/1147]              blk.56.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1066/1147]          blk.56.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.56.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[1067/1147]          blk.56.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.56.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[1068/1147]            blk.56.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.56.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[1069/1147]               blk.56.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1070/1147]              blk.57.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1071/1147]           blk.57.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1072/1147]         blk.57.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.57.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[1073/1147]         blk.57.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.57.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[1074/1147]           blk.57.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.57.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[1075/1147]         blk.57.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1076/1147]          blk.57.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.57.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[1077/1147]              blk.57.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.57.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[1078/1147]               blk.57.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.57.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.57.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1079/1147]               blk.57.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.57.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1080/1147]            blk.57.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.57.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[1081/1147]          blk.57.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1082/1147]               blk.57.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.57.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[1083/1147]               blk.57.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.57.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[1084/1147]              blk.57.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1085/1147]          blk.57.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.57.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[1086/1147]          blk.57.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.57.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[1087/1147]            blk.57.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.57.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[1088/1147]               blk.57.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1089/1147]              blk.58.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1090/1147]           blk.58.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1091/1147]         blk.58.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.58.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[1092/1147]         blk.58.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.58.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[1093/1147]           blk.58.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.58.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[1094/1147]         blk.58.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1095/1147]          blk.58.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.58.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[1096/1147]              blk.58.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.58.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[1097/1147]               blk.58.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.58.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.58.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1098/1147]               blk.58.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.58.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1099/1147]            blk.58.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.58.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[1100/1147]          blk.58.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1101/1147]               blk.58.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.58.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[1102/1147]               blk.58.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.58.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[1103/1147]              blk.58.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1104/1147]          blk.58.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.58.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[1105/1147]          blk.58.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.58.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[1106/1147]            blk.58.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.58.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[1107/1147]               blk.58.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1108/1147]              blk.59.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1109/1147]           blk.59.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1110/1147]         blk.59.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.59.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[1111/1147]         blk.59.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.59.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[1112/1147]           blk.59.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.59.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[1113/1147]         blk.59.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1114/1147]          blk.59.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.59.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[1115/1147]              blk.59.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.59.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[1116/1147]               blk.59.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.59.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.59.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1117/1147]               blk.59.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.59.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1118/1147]            blk.59.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.59.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[1119/1147]          blk.59.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1120/1147]               blk.59.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.59.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[1121/1147]               blk.59.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.59.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[1122/1147]              blk.59.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1123/1147]          blk.59.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.59.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[1124/1147]          blk.59.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.59.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[1125/1147]            blk.59.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.59.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[1126/1147]               blk.59.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1127/1147]              blk.60.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[1128/1147]           blk.60.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[1129/1147]         blk.60.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =    f16, Using custom type q5_K for tensor blk.60.ffn_down_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[1130/1147]         blk.60.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.60.ffn_gate_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[1131/1147]           blk.60.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =    f16, Using custom type q5_K for tensor blk.60.ffn_up_shexp.weight
converting to q5_K .. size =    28.00 MiB ->     9.62 MiB
[1132/1147]         blk.60.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[1133/1147]          blk.60.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.60.attn_kv_a_mqa.weight
converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
[1134/1147]              blk.60.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.60.attn_kv_b.weight
converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
[1135/1147]               blk.60.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.60.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.60.attn_k_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1136/1147]               blk.60.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.60.attn_v_b.weight
converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
[1137/1147]            blk.60.attn_output.weight - [16384,  7168,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.60.attn_output.weight
converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
[1138/1147]          blk.60.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[1139/1147]               blk.60.attn_q_a.weight - [ 7168,  1536,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.60.attn_q_a.weight
converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
[1140/1147]               blk.60.attn_q_b.weight - [ 1536, 24576,     1,     1], type =    f16, Using custom type q8_0 for tensor blk.60.attn_q_b.weight
converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
[1141/1147]                        output.weight - [ 7168, 129280,     1,     1], type =    f16, Using custom type q8_0 for tensor output.weight

====== llama_model_quantize_internal: did not find weights for output.weight
converting to q8_0 .. size =  1767.50 MiB ->   938.98 MiB
[1142/1147]              blk.60.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1143/1147]          blk.60.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =    f16, Using custom type iq4_xs for tensor blk.60.ffn_down_exps.weight
converting to iq4_xs .. size =  7168.00 MiB ->  1904.00 MiB
[1144/1147]          blk.60.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.60.ffn_gate_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[1145/1147]            blk.60.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =    f16, Using custom type iq3_s for tensor blk.60.ffn_up_exps.weight
converting to iq3_s .. size =  7168.00 MiB ->  1540.00 MiB
[1146/1147]               blk.60.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[1147/1147]                   output_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
llama_model_quantize_internal: model size  = 1282038.27 MB
llama_model_quantize_internal: quant size  = 314569.47 MB

main: quantize time = 9971138.64 ms
main:    total time = 9971138.64 ms

```

Perplexity run with `fmoe = 1, mla = 2, fa = 1, ub = 512, c = 512`:

```
perplexity: tokenizing the input ..
perplexity: tokenization took 1195.26 ms
perplexity: calculating perplexity over 561 chunks, n_ctx=512, batch_size=2048, n_seq=4
perplexity: 11.69 seconds per pass - ETA 27.32 minutes
[1]2.5779,[2]3.3447,[3]2.4073,[4]2.0140,[5]1.8352,[6]1.6862,[7]1.5895,[8]1.5208,[9]1.4715,[10]1.4284,[11]1.4147,[12]1.4406,[13]1.4529,[14]1.5824,[15]1.7144,[16]1.7752,[17]1.9408,[18]2.0703,[19]2.0333,[20]2.0250,[21]2.1305,[22]2.1021,[23]2.0764,[24]2.0880,[25]2.0581,[26]2.0330,[27]2.0797,[28]2.0888,[29]2.1391,[30]2.1698,[31]2.2044,[32]2.2227,[33]2.2626,[34]2.3049,[35]2.3566,[36]2.4115,[37]2.4463,[38]2.4930,[39]2.5346,[40]2.5926,[41]2.6353,[42]2.6458,[43]2.6948,[44]2.7107,[45]2.7909,[46]2.8420,[47]2.8003,[48]2.7549,[49]2.7298,[50]2.7498,[51]2.7964,[52]2.8105,[53]2.8597,[54]2.8734,[55]2.9047,[56]2.9384,[57]2.9550,[58]2.9926,[59]3.0027,[60]3.0502,[61]3.0906,[62]3.1475,[63]3.1812,[64]3.2262,[65]3.2360,[66]3.2179,[67]3.1954,[68]3.2271,[69]3.2225,[70]3.2377,[71]3.2562,[72]3.2726,[73]3.2860,[74]3.3095,[75]3.2881,[76]3.2396,[77]3.1959,[78]3.1931,[79]3.1728,[80]3.1563,[81]3.1190,[82]3.1220,[83]3.0918,[84]3.0554,[85]3.0218,[86]2.9995,[87]2.9958,[88]2.9686,[89]2.9537,[90]2.9261,[91]2.8966,[92]2.8704,[93]2.8441,[94]2.8196,[95]2.7964,[96]2.7947,[97]2.8024,[98]2.7882,[99]2.7728,[100]2.7752,[101]2.7671,[102]2.7843,[103]2.8105,[104]2.8288,[105]2.8261,[106]2.8486,[107]2.8737,[108]2.8953,[109]2.9296,[110]2.9637,[111]2.9837,[112]2.9567,[113]2.9436,[114]2.9207,[115]2.9047,[116]2.8905,[117]2.8672,[118]2.8450,[119]2.8235,[120]2.8040,[121]2.7884,[122]2.7698,[123]2.7532,[124]2.7334,[125]2.7156,[126]2.6981,[127]2.6840,[128]2.6757,[129]2.6662,[130]2.6551,[131]2.6472,[132]2.6548,[133]2.6649,[134]2.6714,[135]2.6822,[136]2.6990,[137]2.7145,[138]2.7231,[139]2.7348,[140]2.7353,[141]2.7368,[142]2.7356,[143]2.7359,[144]2.7320,[145]2.7228,[146]2.7211,[147]2.7254,[148]2.7248,[149]2.7265,[150]2.7210,[151]2.7192,[152]2.7157,[153]2.7114,[154]2.7119,[155]2.7159,[156]2.7180,[157]2.7237,[158]2.7322,[159]2.7339,[160]2.7428,[161]2.7509,[162]2.7605,[163]2.7660,[164]2.7863,[165]2.8095,[166]2.8270,[167]2.8399,[168]2.8647,[169]2.8872,[170]2.9083,[171]2.9311,[172]2.9150,[173]2.8980,[174]2.8843,[175]2.8712,[176]2.8589,[177]2.8467,[178]2.8338,[179]2.8193,[180]2.8228,[181]2.8370,[182]2.8519,[183]2.8669,[184]2.8813,[185]2.8915,[186]2.9083,[187]2.9241,[188]2.9381,[189]2.9489,[190]2.9490,[191]2.9561,[192]2.9601,[193]2.9652,[194]2.9848,[195]2.9935,[196]3.0068,[197]3.0167,[198]3.0211,[199]3.0267,[200]3.0261,[201]3.0415,[202]3.0361,[203]3.0413,[204]3.0446,[205]3.0447,[206]3.0468,[207]3.0552,[208]3.0645,[209]3.0737,[210]3.0738,[211]3.0688,[212]3.0689,[213]3.0765,[214]3.0781,[215]3.0837,[216]3.0847,[217]3.0805,[218]3.0804,[219]3.0811,[220]3.0800,[221]3.0803,[222]3.0803,[223]3.0805,[224]3.0856,[225]3.0871,[226]3.0791,[227]3.0772,[228]3.0792,[229]3.0835,[230]3.0900,[231]3.0962,[232]3.0880,[233]3.0801,[234]3.0803,[235]3.0787,[236]3.0879,[237]3.0957,[238]3.1050,[239]3.1151,[240]3.1241,[241]3.1353,[242]3.1498,[243]3.1632,[244]3.1713,[245]3.1831,[246]3.1937,[247]3.1927,[248]3.1884,[249]3.1867,[250]3.1804,[251]3.1782,[252]3.1805,[253]3.1841,[254]3.1910,[255]3.1971,[256]3.2005,[257]3.2032,[258]3.2042,[259]3.2076,[260]3.2098,[261]3.2107,[262]3.2099,[263]3.2158,[264]3.2179,[265]3.2182,[266]3.2199,[267]3.2230,[268]3.2267,[269]3.2298,[270]3.2290,[271]3.2271,[272]3.2205,[273]3.2208,[274]3.2143,[275]3.2037,[276]3.1934,[277]3.1951,[278]3.2052,[279]3.2115,[280]3.2195,[281]3.2272,[282]3.2333,[283]3.2398,[284]3.2466,[285]3.2603,[286]3.2626,[287]3.2661,[288]3.2707,[289]3.2732,[290]3.2648,[291]3.2557,[292]3.2544,[293]3.2536,[294]3.2513,[295]3.2487,[296]3.2507,[297]3.2513,[298]3.2562,[299]3.2620,[300]3.2651,[301]3.2691,[302]3.2713,[303]3.2734,[304]3.2726,[305]3.2845,[306]3.2922,[307]3.3033,[308]3.2916,[309]3.2865,[310]3.2769,[311]3.2804,[312]3.2825,[313]3.2893,[314]3.2915,[315]3.2946,[316]3.2959,[317]3.2974,[318]3.2979,[319]3.2982,[320]3.3026,[321]3.3028,[322]3.3042,[323]3.3106,[324]3.3112,[325]3.3167,[326]3.3214,[327]3.3255,[328]3.3282,[329]3.3297,[330]3.3360,[331]3.3396,[332]3.3443,[333]3.3428,[334]3.3425,[335]3.3428,[336]3.3429,[337]3.3437,[338]3.3441,[339]3.3466,[340]3.3502,[341]3.3555,[342]3.3649,[343]3.3744,[344]3.3797,[345]3.3713,[346]3.3640,[347]3.3597,[348]3.3523,[349]3.3488,[350]3.3471,[351]3.3521,[352]3.3671,[353]3.3761,[354]3.3892,[355]3.3977,[356]3.4029,[357]3.4148,[358]3.4246,[359]3.4279,[360]3.4346,[361]3.4439,[362]3.4526,[363]3.4586,[364]3.4649,[365]3.4715,[366]3.4822,[367]3.4909,[368]3.4975,[369]3.5054,[370]3.5138,[371]3.5277,[372]3.5368,[373]3.5401,[374]3.5435,[375]3.5485,[376]3.5616,[377]3.5727,[378]3.5754,[379]3.5749,[380]3.5715,[381]3.5762,[382]3.5816,[383]3.5853,[384]3.5894,[385]3.5931,[386]3.5996,[387]3.6055,[388]3.6087,[389]3.5980,[390]3.5883,[391]3.5774,[392]3.5715,[393]3.5623,[394]3.5535,[395]3.5438,[396]3.5336,[397]3.5245,[398]3.5146,[399]3.5042,[400]3.4963,[401]3.4863,[402]3.4756,[403]3.4668,[404]3.4563,[405]3.4465,[406]3.4364,[407]3.4270,[408]3.4178,[409]3.4090,[410]3.4031,[411]3.4038,[412]3.3993,[413]3.4012,[414]3.4038,[415]3.4009,[416]3.4009,[417]3.4034,[418]3.3979,[419]3.3991,[420]3.3966,[421]3.3953,[422]3.3970,[423]3.3964,[424]3.4006,[425]3.4005,[426]3.4009,[427]3.3997,[428]3.4021,[429]3.4037,[430]3.4064,[431]3.4074,[432]3.4064,[433]3.4027,[434]3.4028,[435]3.3956,[436]3.3891,[437]3.3851,[438]3.3833,[439]3.3805,[440]3.3855,[441]3.3905,[442]3.3979,[443]3.3964,[444]3.3972,[445]3.3983,[446]3.4029,[447]3.4058,[448]3.4083,[449]3.4114,[450]3.4154,[451]3.4184,[452]3.4206,[453]3.4223,[454]3.4208,[455]3.4229,[456]3.4232,[457]3.4257,[458]3.4311,[459]3.4317,[460]3.4318,[461]3.4284,[462]3.4322,[463]3.4396,[464]3.4448,[465]3.4381,[466]3.4361,[467]3.4344,[468]3.4355,[469]3.4328,[470]3.4301,[471]3.4304,[472]3.4311,[473]3.4304,[474]3.4295,[475]3.4308,[476]3.4290,[477]3.4282,[478]3.4288,[479]3.4307,[480]3.4334,[481]3.4290,[482]3.4325,[483]3.4316,[484]3.4353,[485]3.4416,[486]3.4444,[487]3.4479,[488]3.4531,[489]3.4555,[490]3.4603,[491]3.4665,[492]3.4709,[493]3.4707,[494]3.4719,[495]3.4746,[496]3.4764,[497]3.4794,[498]3.4798,[499]3.4790,[500]3.4832,[501]3.4877,[502]3.4865,[503]3.4849,[504]3.4871,[505]3.4905,[506]3.4988,[507]3.5016,[508]3.5050,[509]3.4973,[510]3.4914,[511]3.4851,[512]3.4810,[513]3.4750,[514]3.4738,[515]3.4761,[516]3.4714,[517]3.4713,[518]3.4704,[519]3.4710,[520]3.4755,[521]3.4744,[522]3.4730,[523]3.4790,[524]3.4775,[525]3.4761,[526]3.4715,[527]3.4663,[528]3.4628,[529]3.4599,[530]3.4568,[531]3.4536,[532]3.4479,[533]3.4415,[534]3.4370,[535]3.4382,[536]3.4410,[537]3.4443,[538]3.4469,[539]3.4496,[540]3.4550,[541]3.4584,[542]3.4607,[543]3.4552,[544]3.4512,[545]3.4508,[546]3.4440,[547]3.4374,[548]3.4307,[549]3.4240,[550]3.4178,[551]3.4116,[552]3.4060,[553]3.4002,[554]3.3983,[555]3.3970,[556]3.3998,[557]3.4039,[558]3.4098,[559]3.4145,[560]3.4197,[561]3.4178,
Final estimate: PPL = 3.4178 +/- 0.01891

llama_print_timings:        load time =  708891.72 ms
llama_print_timings:      sample time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings: prompt eval time = 1469613.91 ms / 287232 tokens (    5.12 ms per token,   195.45 tokens per second)
llama_print_timings:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings:       total time = 1478240.60 ms / 287233 tokens
```

Thought perplexity wasn't great here, was hoping for better. 

Ran it at `c = 2048, ub = 2028, fa = 1, mla = 2, fmoe = 2` to see comparable to `IQ3_M` which achieved `Final estimate: PPL = 3.1464 +/- 0.01620` with same settings:

```
perplexity: tokenizing the input ..
perplexity: tokenization took 1174.64 ms
perplexity: calculating perplexity over 140 chunks, n_ctx=2048, batch_size=2048, n_seq=1
perplexity: 7.62 seconds per pass - ETA 17.77 minutes
[1]1.5191,[2]1.2928,[3]1.2408,[4]1.6923,[5]1.7504,[6]1.7209,[7]1.8237,[8]1.9455,[9]2.1337,[10]2.3217,[11]2.4469,[12]2.3274,[13]2.4507,[14]2.5489,[15]2.6763,[16]2.7962,[17]2.7824,[18]2.8417,[19]2.7812,[20]2.7006,[21]2.6333,[22]2.5627,[23]2.4751,[24]2.4184,[25]2.3854,[26]2.4627,[27]2.5380,[28]2.5401,[29]2.4842,[30]2.4252,[31]2.3702,[32]2.3267,[33]2.3103,[34]2.3489,[35]2.3848,[36]2.3849,[37]2.3911,[38]2.3871,[39]2.3979,[40]2.4271,[41]2.4818,[42]2.5601,[43]2.5891,[44]2.5454,[45]2.5170,[46]2.5691,[47]2.6215,[48]2.6434,[49]2.6904,[50]2.7084,[51]2.7301,[52]2.7528,[53]2.7560,[54]2.7706,[55]2.7710,[56]2.7837,[57]2.7865,[58]2.8053,[59]2.8187,[60]2.8509,[61]2.8924,[62]2.8963,[63]2.8979,[64]2.9161,[65]2.9244,[66]2.9365,[67]2.9452,[68]2.9299,[69]2.8925,[70]2.9199,[71]2.9491,[72]2.9585,[73]2.9341,[74]2.9376,[75]2.9550,[76]2.9611,[77]2.9624,[78]2.9674,[79]2.9764,[80]2.9831,[81]2.9870,[82]2.9929,[83]3.0063,[84]3.0084,[85]3.0215,[86]3.0459,[87]3.0233,[88]3.0526,[89]3.0818,[90]3.1045,[91]3.1252,[92]3.1539,[93]3.1856,[94]3.2169,[95]3.2175,[96]3.2352,[97]3.2469,[98]3.2157,[99]3.1800,[100]3.1453,[101]3.1116,[102]3.0791,[103]3.0714,[104]3.0611,[105]3.0620,[106]3.0633,[107]3.0655,[108]3.0675,[109]3.0452,[110]3.0435,[111]3.0404,[112]3.0506,[113]3.0635,[114]3.0694,[115]3.0790,[116]3.0973,[117]3.0966,[118]3.0955,[119]3.0956,[120]3.0985,[121]3.1000,[122]3.1126,[123]3.1293,[124]3.1330,[125]3.1403,[126]3.1400,[127]3.1487,[128]3.1310,[129]3.1252,[130]3.1303,[131]3.1391,[132]3.1221,[133]3.1092,[134]3.1161,[135]3.1289,[136]3.1184,[137]3.0953,[138]3.0734,[139]3.0769,[140]3.0966,
Final estimate: PPL = 3.0966 +/- 0.01608
```

At least no NaNs!

---

👤 **ikawrakow** commented the **2025-03-11** at **11:52:41**:<br>

> Thought perplexity wasn't great here, was hoping for better.

Why? `3.4178` is 2% higher than the PPL reported for the `Q5_K_XL` model, which is 480 GiB. Your model as per quantization log is  314569 MiV = 307 GiB. 2% increase in PPL for 56% reduction in model size is a pretty good result, actually. It basically means that the `Q5_K_XL` model is very far from the Pareto front in the model size vs model quality plane.

> Or would IQ4_NL be comparable to IQ4_K?

No, `IQ4_NL` is comparable to `IQ4_XS`. It is essentially the same quantization type, but `IQ4_XS` uses a more efficient bit packing for the block scales by utilizing "super-blocks" of size 256. The main reason for the existence of `IQ4_NL` is for using it to quantize tensors where the row size is not a multiple of 256 as required by `IQ4_XS`. E.g., in DeepSeek-Lite the `ffn_down_exps` row size is 1408, so one needs to use `IQ4_NL` instead of `IQ4_XS` for those.

> Would it be hard to build a MMQ kernel for IQ4_K?

Yes. `IQ4_K` departs too much from the bit packing used for the quants with MMQ kernels, so it is not possible to just adapt one of the existing kernels.

> EPYC 7713 w/ 256GB DDR4 RAM. I don't think the experts will fit on the RAM sadly.

Actually, a layer with `IQ4_K` for `ffn_dow_exps` and `IQ3_S` for `ffn_up/gate_exps` uses  5.03125 GiB for the experts, so you can fit 50 layers of such experts in 256 GiB of RAM. Let's assume it is better to have 16 GiB left so the process doesn't get killed due to OOM. This would be still 48 layers on the CPU. But if you have 48 layers on the CPU, you can use `Q5_K` or even `Q6_K` for the experts in the remaining 10 layers. You will lose performance, but you can run a much larger model that way.

---

👤 **davidsyoung** commented the **2025-03-11** at **19:29:11**:<br>

> > Thought perplexity wasn't great here, was hoping for better.
> 
> Why? `3.4178` is 2% higher than the PPL reported for the `Q5_K_XL` model, which is 480 GiB. Your model as per quantization log is 314569 MiV = 307 GiB. 2% increase in PPL for 56% reduction in model size is a pretty good result, actually. It basically means that the `Q5_K_XL` model is very far from the Pareto front in the model size vs model quality plane.
> 
> > Or would IQ4_NL be comparable to IQ4_K?
> 
> No, `IQ4_NL` is comparable to `IQ4_XS`. It is essentially the same quantization type, but `IQ4_XS` uses a more efficient bit packing for the block scales by utilizing "super-blocks" of size 256. The main reason for the existence of `IQ4_NL` is for using it to quantize tensors where the row size is not a multiple of 256 as required by `IQ4_XS`. E.g., in DeepSeek-Lite the `ffn_down_exps` row size is 1408, so one needs to use `IQ4_NL` instead of `IQ4_XS` for those.
> 
> > Would it be hard to build a MMQ kernel for IQ4_K?
> 
> Yes. `IQ4_K` departs too much from the bit packing used for the quants with MMQ kernels, so it is not possible to just adapt one of the existing kernels.
> 
> > EPYC 7713 w/ 256GB DDR4 RAM. I don't think the experts will fit on the RAM sadly.
> 
> Actually, a layer with `IQ4_K` for `ffn_dow_exps` and `IQ3_S` for `ffn_up/gate_exps` uses 5.03125 GiB for the experts, so you can fit 50 layers of such experts in 256 GiB of RAM. Let's assume it is better to have 16 GiB left so the process doesn't get killed due to OOM. This would be still 48 layers on the CPU. But if you have 48 layers on the CPU, you can use `Q5_K` or even `Q6_K` for the experts in the remaining 10 layers. You will lose performance, but you can run a much larger model that way.

This is really helpful to have context. I wasn't sure if the inference on the  `Q5_K_XL` was broken or not with it being on mainline `llama.cpp`. 

I might give that a go with running on CPU as well. Truthfully, I haven't spent much time testing the model outside of PPL. So likely I need to get some of that done first!