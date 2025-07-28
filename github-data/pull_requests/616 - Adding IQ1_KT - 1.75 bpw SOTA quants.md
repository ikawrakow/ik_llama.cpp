## 🔀 [Pull Request #616](https://github.com/ikawrakow/ik_llama.cpp/pull/616) - Adding IQ1_KT - 1.75 bpw SOTA quants

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/iq1_kt` |
| **Target Branch** | `main` |
| **Created** | 2025-07-16 |
| **Updated** | 2025-07-22 |
| **Merged** | 2025-07-20 |

---

## 📄 Description

With Kimi-2 at 1 trillion parameters being the new rage of the day, my guess is that even more local inference enthusiasts will reach to very low bit-per-weight (bpw) quantized models. The state of affairs in mainline `llama.cpp` for very low bpw quants is not good:
* Nothings has been done to improve quantization quality since I contributed [IQ1_S](https://github.com/ggml-org/llama.cpp/pull/5999) and [IQ1_M](https://github.com/ggml-org/llama.cpp/pull/6302) to mainline.
* `IQ1_M` does not even have a CUDA quantized matrix multiplication kernel (a.k.a, MMQ), which results in a disastrous prompt processing (PP) performance

The situation is better in `ik_llama.cpp` performance wise, but quantization quality improvements for the sub-2 bpw quants have been relatively minor.

Hence, this PR adds `IQ1_KT` - 1.75 bpw quantization type based on an integer trellis similar to `IQ2_KT, IQ3_KT` and `IQ4_KT`. `IQ1_KT` uses
* Per tensor row float scales
* Blocks of 32 weights with 4-bit block scales
* Groups of 8 quants per trellis sequence, each group requiring 13 bits.

Similar to the other `*_KT` quants
* Performance is excellent on CUDA for PP and TG
* PP performance is excellent on `AVX2/AVX512` and `ARM_NEON`
* TG performance is somewhat lower (~10-15%) than other quantization types of similar size
* TG performance is bad on `ARM_NEON`

As trellis quants performance is very low on Metal (at least for my 30-core M2-Max GPU), I didn't not even bother to add a Metal implementation.

To illustrate the quantization quality compared to other quantization types, the next graph shows `PPL(Q)/PPL(f16)-1` for LlaMA-3.1-8B-Instruct, which is notoriously hard to quantize. I have excluded the `IQ1_M` and `IQ1_S` data points as this would have extended the y-axis too much to be useful. We can see that `IQ1_KT` at 1.92 bpw provides nearly the same quality as `IQ2_XXS` at 2.13 bpw, so almost a 10% reduction in model size for comparable quantization quality. I have made the `IQ2_KL` data point magenta because it was also added very recently in PR [#602](https://github.com/ikawrakow/ik_llama.cpp/issues/602). 

<img width="792" height="612" alt="il31c" src="https://github.com/user-attachments/assets/c4a3109d-4390-4d41-b44f-17bb2445e89e" />

---

## 💬 Conversation

👤 **ubergarm** commented on **2025-07-16** at **15:50:24**

> With Kimi-2 at 1 trillion parameters being the new rage of the day, my guess is that even more local inference enthusiasts will reach to very low bit-per-weight (bpw) quantized models.

Indeed, people are asking me for sub 2bpw quants of Kimi-K2 already: https://huggingface.co/ubergarm/Kimi-K2-Instruct-GGUF/discussions/1#6876f91f7cf1ec76dfc9fa9e

I'm out of the office for a day or so, but will leave this IQ1_KT Kimi-K2 cooking with this recipe and see how it goes. Normally I leave ffn_down_exps slightly larger, but to get the size down gonna bonk *all* the routed exps down to 1.75bpw.

Guessing it will finish up around ~230GiB or so, still too large to fully offload on dual RTX 6000 PRO Blackwells haha...

<details>

<summary>👈 Secret Recipe</summary>

```bash
#!/usr/bin/env bash

custom="
## Attention [0-60] (GPU)
# Only ik's fork uses this, keep it q8_0 as its only for PP with -mla 3
blk\..*\.attn_kv_b\.weight=q8_0

# ideally k_b and v_b are smaller than q8_0 as they are is used for TG with -mla 3 (and ik's imatrix supports it)
# blk.*.attn_k_b.weight is not divisible by 256 so only supports qN_0 or iq4_nl
blk\..*\.attn_k_b\.weight=iq4_nl

# Balance of attn tensors
blk\..*\.attn_.*=iq4_kt

## First Single Dense Layer [0] (GPU)
blk\..*\.ffn_down\.weight=iq4_kt
blk\..*\.ffn_(gate|up)\.weight=iq3_kt

## Shared Expert [1-60] (GPU)
blk\..*\.ffn_down_shexp\.weight=iq4_kt
blk\..*\.ffn_(gate|up)_shexp\.weight=iq3_kt

## Routed Experts [1-60] (CPU)
blk\..*\.ffn_down_exps\.weight=iq1_kt
blk\..*\.ffn_(gate|up)_exps\.weight=iq1_kt

## Token embedding and output tensors (GPU)
token_embd\.weight=iq4_kt
output\.weight=iq5_ks
"

custom=$(
  echo "$custom" | grep -v '^#' | \
  sed -Ez 's:\n+:,:g;s:,$::;s:^,::'
)

numactl -N 1 -m 1 \
./build/bin/llama-quantize \
    --custom-q "$custom" \
    --imatrix /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat \
    /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/Kimi-K2-384x15B-Instruct-safetensors-BF16-00001-of-00045.gguf \
    /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/Kimi-K2-Instruct-IQ1_KT.gguf \
    IQ1_KT \
    192
```

</details>

---

👤 **ikawrakow** commented on **2025-07-16** at **17:10:01**

Thanks for cooking!

---

👤 **Nexesenex** commented on **2025-07-16** at **17:54:54**

@ikawrakow :
You might have forgotten the CUDA MMQ kernel file, now that you separate it from mmq.cuh. ^^

---

👤 **ubergarm** commented on **2025-07-16** at **17:59:44**

I'm on CPU only with this thing or now, so its doing perplexity now!

```
llama_model_quantize_internal: model size  = 1959011.30 MB
llama_model_quantize_internal: quant size  = 219322.63 MB

main: quantize time = 4557251.29 ms
main:    total time = 4557251.29 ms

llm_load_print_meta: model ftype      = IQ1_KT - 1.75 bpw
llm_load_print_meta: model params     = 1.027 T
llm_load_print_meta: model size       = 214.182 GiB (1.792 BPW)
llm_load_print_meta: repeating layers = 212.916 GiB (1.785 BPW, 1024.571 B parameters)
llm_load_print_meta: general.name     = Kimi K2 Instruct Bf16 Safetensors

llama_model_loader: - type  f32:  365 tensors
llama_model_loader: - type q8_0:   61 tensors
llama_model_loader: - type iq4_nl:   61 tensors
llama_model_loader: - type iq5_ks:    1 tensors
llama_model_loader: - type iq3_kt:  122 tensors
llama_model_loader: - type iq4_kt:  367 tensors
llama_model_loader: - type iq1_kt:  180 tensors

perplexity: tokenization took 633.399 ms
perplexity: calculating perplexity over 568 chunks, n_ctx=512, batch_size=2048, n_seq=4
perplexity: 15.64 seconds per pass - ETA 37.02 minutes
[1]3.3661,[2]4.4515,[3]3.6908,[4]3.6615,[5]3.3665,[6]3.2151,[7]3.2705,[8]3.2106,

llama_print_timings:        load time =   52736.67 ms
llama_print_timings:      sample time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings: prompt eval time = 1940171.96 ms / 290816 tokens (    6.67 ms per token,   149.89 tokens per second)
llama_print_timings:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings:       total time = 1958744.98 ms / 290817 tokens

Final estimate: PPL = 4.3623 +/- 0.02432
```

sweep benches later

---

👤 **ikawrakow** commented on **2025-07-16** at **19:26:04**

@Nexesenex Thanks! Added the forgotten file.

---

👤 **ubergarm** commented on **2025-07-16** at **20:42:55**

## -t 128 -tb 192 (of 192 cores)
main: n_kv_max = 12288, n_batch = 4096, n_ubatch = 4096, flash_attn = 1, n_gpu_layers = -1, n_threads = 128, n_threads_batch = 192

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |   17.377 |   235.72 |   75.213 |    13.61 |
|  4096 |   1024 |   4096 |   20.470 |   200.10 |   80.030 |    12.80 |
|  4096 |   1024 |   8192 |   21.570 |   189.89 |   82.918 |    12.35 |

## -t 192 -tb 192 (of 192 cores)

main: n_kv_max = 12288, n_batch = 4096, n_ubatch = 4096, flash_attn = 1, n_gpu_layers = -1, n_threads = 192, n_threads_batch = 192

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |   16.724 |   244.92 |   74.048 |    13.83 |
|  4096 |   1024 |   4096 |   19.215 |   213.17 |   80.421 |    12.73 |
|  4096 |   1024 |   8192 |   19.674 |   208.19 |   86.859 |    11.79 |

Sorry no graphs as I'm on laptop in a library. Huh I'm surprised adding more to `--threads` improved PP speeds?

>  -t,    --threads N              number of threads to use during generation (default: 384)
>  -tb,   --threads-batch N        number of threads to use during batch and prompt processing (default: same as --threads)

I'll fiddle with it more later.

I might call this one the `-smol-IQ1_KT` and try a more "normal" mix with IQ2_KT as ffn_down_exps to see how it fares PPL and speed-wise.

I'll not release any IQ1_KT until some further testing with CUDA and you are happy with everything. Fun!

*EDIT*
Kind of interesting to run https://github.com/amd/esmi_ib_library while its benchmarking. I'm running these benchmarks on  socket/numa 0 which is down a bank of ram hah. I gotta make a script to measure DDR bandwidth vs time to get a more clear view of what is happening. I haven't seen it go much over ~200GB/s but don't know how it is measuring exactly either. Socket 1 is busy cooking the slightly larger IQ1_KT.

```
============================= E-SMI ===================================

--------------------------------------
| CPU Family            | 0x1a (26 ) |
| CPU Model             | 0x11 (17 ) |
| NR_CPUS               | 768        |
| NR_SOCKETS            | 2          |
| THREADS PER CORE      | 2 (SMT ON) |
--------------------------------------

------------------------------------------------------------------------
| Sensor Name                    | Socket 0         | Socket 1         |
------------------------------------------------------------------------
| Energy (K Joules)              | 34137.254        | 40312.224        |
| Power (Watts)                  | 415.192          | 474.688          |
| PowerLimit (Watts)             | 500.000          | 500.000          |
| PowerLimitMax (Watts)          | 500.000          | 500.000          |
| C0 Residency (%)               | 67               | 87               |
| DDR Bandwidth                  |                  |                  |
|       DDR Max BW (GB/s)        | 528              | 576              |
|       DDR Utilized BW (GB/s)   | 207              | 56               |
|       DDR Utilized Percent(%)  | 39               | 10               |
| Current Active Freq limit      |                  |                  |
|        Freq limit (MHz)        | 3700             | 2818             |
|        Freq limit source       | Refer below[*0]  | Refer below[*1]  |
| Socket frequency range         |                  |                  |
|        Fmax (MHz)              | 3700             | 3700             |
|        Fmin (MHz)              | 600              | 600              |
------------------------------------------------------------------------
```

---

👤 **Nexesenex** commented on **2025-07-16** at **21:36:24**

@ikawrakow : Thanks!

constants.py could be updated as well, I guess.

And of course, thanks for this amazing development!
Having a viable sub-2bpw quant is quite a hit, for the attn_q of mono-GPU aimed quants of 70b/72b models, or even the ffns of the 253b nemotrons models (the highest models I could potentially run in full Cuda-MMQ offload on my 64GB VRAM with a decent quality thanks to the help of iq1_kt and iq2_kt).

---

👤 **ubergarm** commented on **2025-07-17** at **00:39:25**

Cooked a slightly larger version just for comparison. Same recipe as above except larger iq2_kt for ffn_down_exps so more like my "normal" recipes

```
llm_load_print_meta: model params     = 1.027 T
llm_load_print_meta: model size       = 228.948 GiB (1.915 BPW)
llm_load_print_meta: repeating layers = 227.682 GiB (1.909 BPW, 1024.571 B parameters)
llm_load_print_meta: general.name     = Kimi K2 Instruct Bf16 Safetensors

llama_model_loader: - type  f32:  365 tensors
llama_model_loader: - type q8_0:   61 tensors
llama_model_loader: - type iq4_nl:   61 tensors
llama_model_loader: - type iq5_ks:    1 tensors
llama_model_loader: - type iq2_kt:   60 tensors
llama_model_loader: - type iq3_kt:  122 tensors
llama_model_loader: - type iq4_kt:  367 tensors
llama_model_loader: - type iq1_kt:  120 tensors

llama_print_timings:        load time =   80560.40 ms
llama_print_timings:      sample time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings: prompt eval time = 1917998.73 ms / 290816 tokens (    6.60 ms per token,   151.62 tokens per second)
llama_print_timings:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings:       total time = 1936434.86 ms / 290817 tokens

Final estimate: PPL = 4.1310 +/- 0.02266
```

---

👤 **ubergarm** commented on **2025-07-17** at **01:11:45**

Okay last data point I made a "pure" Qwen3-14B-IQ1_KT same as this set: https://github.com/ikawrakow/ik_llama.cpp/pull/602#issuecomment-3065995863

CUDA backend full offload.

Final estimate: PPL = 13.4941 +/- 0.10484

```
llm_load_print_meta: model ftype      = IQ1_KT - 1.75 bpw
llm_load_print_meta: model params     = 14.768 B
llm_load_print_meta: model size       = 3.703 GiB (2.154 BPW)
llm_load_print_meta: repeating layers = 2.701 GiB (1.756 BPW, 13.212 B parameters

llama_model_loader: - type  f32:  161 tensors
llama_model_loader: - type q4_K:    1 tensors
llama_model_loader: - type q6_K:    1 tensors
llama_model_loader: - type iq1_kt:  280 tensors
```

I tried a few short chats and it is actually coherent and was able to complete some requests. It did get stuck in a loop trying think of a good joke, but wow- amazing a sub 2bpw "pure" quantized dense model works at all!

---

👤 **ikawrakow** commented on **2025-07-17** at **18:06:12**

> Final estimate: PPL = 4.3623 +/- 0.02432

IIRC, the `Q8_0` PPL is `2.95`? So this would be ~48% higher for the pure version, 40% for the 229 GiB `IQ1_KT/IQ2_KT` mix. I haven't seen a lot of Kimi-2 PPL values, so don't have a feel if this is good or not so good. I see Unsloth's `IQ1_M` is 304 GB (283 GiB), so there is a lot of room to use more bits for some of the tensors. 

One thing you could try is to simply take Unsloth's `IQ1_M` recipe, replace `IQ1_M` with `IQ1_KT`, and see how PPL compares. One could go a step further and replace 
* `Q4_K` with `IQ4_K`
* `Q5_K` with `IQ5_K`
* `IQ4_XS` with `IQ4_KS`
* `IQ2_XXS` with `IQ2_KT`
*  `IQ3_XXS` with `IQ3_KT`

The last two are +0.0625, so if one wanted to arrive at the exact same size, one needs to reduce the number of tensors using `IQ3_XXS` so that the +0.0625 bpw is recovered. Our perhaps replace some of the `IQ3_XXS` with `IQ2_KL`.

Does anyone know what is `TQ1_0`? When `TQ1_0` was added to `llama.cpp` it was BitNet specific, so totally not useful for a not-BitNet model. I also don't see any `TQ1_0` tensors. Why did they decide to call it `TQ1_0`?

---

👤 **ubergarm** commented on **2025-07-17** at **20:06:09**

> I haven't seen a lot of Kimi-2 PPL values, so don't have a feel if this is good or not so good.

Yeah I'm not 100% sure what command @magikRUKKOLA is using for imatrix, so hopefully his numbers are comparable to mine. He's been updating this graph here it seems: https://github.com/ikawrakow/ik_llama.cpp/discussions/477#discussioncomment-13776504

this is my test command for anyone curious

<details>

<summary>ubergarm latest perplexity methodology</summary>

Modify for CPU only or offload more CUDA layers etc. I've been using CPU only for my Kimi-K2-Instruct quants. The seed is not important. My older DeepSeek-V3 numbers were with `-ctk q8_0` but now I use full fp16 for reporting the PPLs.

```bash
$ wget https://github.com/user-attachments/files/19090237/wiki.test.raw.gz
$ gunzip wiki.test.raw.gz
$ du -h wiki.test.raw
1.3M	wiki.test.raw
$ sha1sum wiki.test.raw
6f1fe2054a940eebfc76b284b09680763b37f5ea  wiki.test.raw

./build/bin/llama-perplexity \
    --model "$model" \
    -f wiki.test.raw \
    -ctk fp16 \
    -fa -fmoe \
    -mla 3 -amb 512 \
    --ctx-size 512 \
    --ubatch-size 512 \
    -ngl 99 \
    -ot exps=CPU \
    --threads 24
```

Wait to the end for the Final PPL= value.

</details>

> One thing you could try

I'll try to make up a mix similar to your description test it out!

> Does anyone know what is TQ1_0? When TQ1_0 was added to llama.cpp it was BitNet specific, so totally not useful for a not-BitNet model. I also don't see any TQ1_0 tensors. Why did they decide to call it TQ1_0?

tl;dr; Apparently ollama and huggingface don't properly show quants with "unusual" file names. (which is why many of my quants don't show up in the side-bar tensor viewer on hf). Unsloth wanted to release quants in a similar BPW that just happened to be similar to Compilade's ternary only bitnet quantization type. So despite the unsloth TQ1_0 consisting of mostly IQ1_S and IQ3_S and no TQ1_0 tensors at all, they started using that name and seem to continue to be doing so.

I've called them out multiple times on reddit and github on the improper use of TQ1_0 beginning with R1-0528. Here is the most recent discussion where they continued to do it for Kimi-K2-Instruct:

* https://github.com/ggml-org/llama.cpp/pull/14654#issuecomment-3070114997
* https://github.com/ggml-org/llama.cpp/pull/14654#issuecomment-3070439724 
* https://github.com/ggml-org/llama.cpp/pull/14654#issuecomment-3071240401

And earlier on R1-0528:
* https://www.reddit.com/r/LocalLLaMA/comments/1l19yud/comment/mvjyw04/

Anyway, I'll go see what else I can cook up and try to compare some PPLs!

---

👤 **ubergarm** commented on **2025-07-18** at **06:18:49**

I've been grinding through perplexity on some quants, have at least one more to add (UD-IQ2_XXS) and would like to add @anikifoss 's larger model(s) as they roll out if he would like to calculate them, no pressure though!

*EDIT* Updated image and data to fixup badname TQ1_0 bpw and add some more data points. The v0.2 recipes are full q8_0 `attn/shexp/blk.0.ffn.*` versions. Some comments in the json data as well.

<details>

<summary>👈 raw data in json format</summary>

```json
[
  {
    "name": "q8_0",
    "ppl": "2.9507 +/- 0.01468",
    "size": 1016.623,
    "bpw": 8.504,
    "legend": "pure"
  },
  {
    "name": "IQ4_KS",
    "ppl": "3.0438 +/- 0.01536",
    "size": 550.428,
    "bpw": 4.604,
    "legend": "ubergarm",
    "comment": "v0.1 recipe"
  },
  {
    "name": "v0.2-IQ4_KS",
    "ppl": "2.9584 +/- 0.01473",
    "size": 554.421,
    "bpw": 4.638,
    "legend": "ubergarm",
    "comment": "v0.2 recipe - full q8_0 attn/shexp/blk.0.ffn"
  },
  {
    "name": "IQ3_KS",
    "ppl": "3.1395 +/- 0.01604",
    "size": 427.205,
    "bpw": 3.573,
    "legend": "ubergarm",
    "comment": "v0.1 recipe"
  },
  {
    "name": "v0.2-IQ3_KS",
    "ppl": "3.0226 +/- 0.01518",
    "size": 430.908,
    "bpw": 3.604,
    "legend": "ubergarm",
    "comment": "v0.2 recipe"
  },
  {
    "name": "PR624-IQ3_KS",
    "ppl": "3.1936 +/- 0.01638",
    "size": 427.205,
    "bpw": 3.573,
    "legend": "ubergarm"
  },
  {
    "name": "IQ2_KL",
    "ppl": "3.2741 +/- 0.01689",
    "size": 345.687,
    "bpw": 2.892,
    "legend": "ubergarm",
    "comment": "v0.1 recipe"
  },
  {
    "name": "PR624-IQ2_KL",
    "ppl": "3.3055 +/- 0.01709",
    "size": 345.687,
    "bpw": 2.892,
    "legend": "ubergarm"
  },
  {
    "name": "chonk-IQ2_KL",
    "ppl": "3.2095 +/- 0.01641",
    "size": 365.507,
    "bpw": 3.057,
    "legend": "ubergarm",
    "comment": "blk.(1|2|3|4|5|6|59|60).ffn_down_exps.weight=iq4_ks and blk.(1|2|3|4|5|6|59|60).ffn_(gate|up)_exps.weight=iq4_kss"
  },
  {
    "name": "PR624-chonk-IQ2_KL",
    "ppl": "3.2389 +/- 0.01661",
    "size": 365.507,
    "bpw": 3.057,
    "legend": "ubergarm",
    "comment": "blk.(1|2|3|4|5|6|59|60).ffn_down_exps.weight=iq4_ks and blk.(1|2|3|4|5|6|59|60).ffn_(gate|up)_exps.weight=iq4_kss"
  },
  {
    "name": "v0.2-IQ2_KL",
    "ppl": "3.1813 +/- 0.01619",
    "size": 349.389,
    "bpw": 2.923,
    "legend": "ubergarm",
    "comment": "v0.2 recipe - full q8_0 attn/shexp/blk.0.ffn"
  },
  {
    "name": "IQ2_KS",
    "ppl": "3.7922 +/- 0.02045",
    "size": 286.624,
    "bpw": 2.398,
    "legend": "ubergarm",
    "comment": "v0.1 recipe"
  },
  {
    "name": "PR624-IQ2_KS",
    "ppl": "3.7846 +/- 0.02040",
    "size": 286.624,
    "bpw": 2.398,
    "legend": "ubergarm"
  },
  {
    "name": "PR624-chonk-IQ2_KS",
    "ppl": "3.7313 +/- 0.01999",
    "size": 313.923,
    "bpw": 2.626,
    "legend": "ubergarm",
    "comment": "blk.(1|2|3|4|5|6|59|60).ffn_down_exps.weight=iq4_ks and blk.(1|2|3|4|5|6|59|60).ffn_(gate|up)_exps.weight=iq4_kss"
  },
  {
    "name": "PR624-v0.2-IQ2_KS",
    "ppl": "3.6827 +/- 0.01957",
    "size": 290.327,
    "bpw": 2.429,
    "legend": "ubergarm",
    "comment": "v0.2 recipe - full q8_0 attn/shexp/blk.0.ffn"
  },
  {
    "name": "v0.2-IQ1_KT",
    "ppl": "3.9734 +/- 0.02152",
    "size": 234.141,
    "bpw": 1.959,
    "legend": "ubergarm",
    "comment": "v0.2 recipe - full q8_0 attn/shexp/blk.0.ffn"
  },
  {
    "name": "IQ1_KT",
    "ppl": "4.1310 +/- 0.02266",
    "size": 228.948,
    "bpw": 1.915,
    "legend": "ubergarm"
  },
  {
    "name": "smol-IQ1_KT",
    "ppl": "4.3623 +/- 0.02432",
    "size": 214.182,
    "bpw": 1.792,
    "legend": "ubergarm"
  },
  {
    "name": "v0.2-smol-IQ1_KT",
    "ppl": "4.2187 +/- 0.02325",
    "size": 219.375,
    "bpw": 1.835,
    "legend": "ubergarm",
    "comment": "v0.2 recipe - full q8_0 attn/shexp/blk.0.ffn"
  },
  {
    "name": "DQ4_K",
    "ppl": "2.9691 +/- 0.01480",
    "size": 624.828,
    "bpw": 5.229,
    "legend": "anikifoss",
    "url": "https://huggingface.co/anikifoss/Kimi-K2-Instruct-DQ4_K"
  },
  {
    "name": "UD-IQ1_S",
    "ppl": "4.3331 +/- 0.02390",
    "size": 261.979,
    "bpw": 2.192,
    "legend": "unsloth",
    "comment": "ran this without -fmoe fwiw before PR630"
  },
  {
    "name": "badname-UD-TQ1_0",
    "ppl": "5.0150 +/- 0.02885",
    "size": 227.854,
    "bpw": 1.907,
    "legend": "unsloth",
    "comment": "this is not a TQ1_0 but an incorrect name"
  },
  {
    "name": "UD-IQ2_XXS",
    "ppl": "3.5258 +/- 0.01842",
    "size": 305.660,
    "bpw": 2.558,
    "legend": "unsloth"
  },
  {
    "name": "UD-IQ3_XXS",
    "ppl": "3.1535 +/- 0.01601",
    "size": 388.003,
    "bpw": 3.247,
    "legend": "unsloth"
  },
  {
    "name": "UD-Q4_K_XL",
    "ppl": "3.0612 +/- 0.01550",
    "size": 547.437,
    "bpw": 4.581,
    "legend": "unsloth"
  }
]
```

<img width="2000" height="1000" alt="image" src="https://github.com/user-attachments/assets/468a71f4-78a0-4db0-837a-fd7619bbef01" />

</details>

<img width="1795" height="1182" alt="ppl-Kimi-K2-v0 2-with-iq3_ks" src="https://github.com/user-attachments/assets/42dc1b23-fe4e-496b-ba15-a887534d1f6b" />

---

👤 **magikRUKKOLA** commented on **2025-07-18** at **17:15:11**

@ubergarm 

>     "name": "UD-IQ2_XXS",
>     "ppl": "3.5258 +/- 0.01842",
>     "size": 305.660,
>     "bpw": 2.558,
>     "legend": "unsloth",
>     "comment": "compare to magikRUKKOLA's measurement PPL = 3.1382"
>   },

~~But i never claimed that PPL for UD-IQ2_XXS is 3.1382.  That was related to the UD-IQ3_XXS.~~
Ah.  That was probably a copy/paste error.
Ok, I will retest the UD-IQ3_XXS.

---

👤 **ikawrakow** commented on **2025-07-18** at **17:20:53**

If `UD-IQ2_XXS` at 2.5 bpw had a PPL of 3.1382, we could all collectively forget about new quantization types and quant cooking, and just use that.

---

👤 **anikifoss** commented on **2025-07-18** at **17:21:12**

@ubergarm I'm distracted benchmarking MI50s for the next couple of days. I'll get perplexity calculations sometime next week or so.

---

👤 **ubergarm** commented on **2025-07-18** at **17:39:58**

@magikRUKKOLA 

Yeah, thanks, i was up a bit too late last night. Cleaning things up as much as I can while going along.

But yes the UD-IQ2_XXS and UD-IQ3_XXS seem pretty decent so far in my testing. I went ahead and ran the numbers on both myself and it came out just slightly above your reported value (for UD_IQ3_XXS).

I'm testing adding a little bit more weight to the early ffn_*_exps layers which is helping, but distracted by also testing the PR for tweaks that are affecting IQ3_KS. So gonna try to sort that out first before going too wild on new recipes hah...

If you want to see the full UD-IQ3_XXS here is the gguf dump showing them alternating tensors sizes for the same tensors up and down across throughout the layers. In looking at a few of their recipes the pattern seems different for different size models, so not sure exactly what they are using to decide on this, but I haven't read their blogs in a while.

<details>

<summary>UD-IQ3_XXS gguf-dump</summary>

```bash
INFO:gguf-dump:* Loading: /mnt/data/models/unsloth/Kimi-K2-Instruct-GGUF/UD-IQ3_XXS/Kimi-K2-Instruct-UD-IQ3_XXS-00001-of-00009.gguf
* File is LITTLE endian, script is running on a LITTLE endian host.
* Dumping 64 key/value pair(s)
      1: UINT32     |        1 | GGUF.version = 3
      2: UINT64     |        1 | GGUF.tensor_count = 134
      3: UINT64     |        1 | GGUF.kv_count = 61
      4: STRING     |        1 | general.architecture = 'deepseek2'
      5: STRING     |        1 | general.type = 'model'
      6: STRING     |        1 | general.name = 'Kimi-K2-Instruct'
      7: STRING     |        1 | general.finetune = 'Instruct'
      8: STRING     |        1 | general.basename = 'Kimi-K2-Instruct'
      9: STRING     |        1 | general.quantized_by = 'Unsloth'
     10: STRING     |        1 | general.size_label = '384x14B'
     11: STRING     |        1 | general.license = 'other'
     12: STRING     |        1 | general.license.name = 'modified-mit'
     13: STRING     |        1 | general.repo_url = 'https://huggingface.co/unsloth'
     14: UINT32     |        1 | general.base_model.count = 1
     15: STRING     |        1 | general.base_model.0.name = 'Kimi K2 Instruct'
     16: STRING     |        1 | general.base_model.0.organization = 'Moonshotai'
     17: STRING     |        1 | general.base_model.0.repo_url = 'https://huggingface.co/moonshotai/Kimi-K2-Instruct'
     18: [STRING]   |        1 | general.tags
     19: UINT32     |        1 | deepseek2.block_count = 61
     20: UINT32     |        1 | deepseek2.context_length = 131072
     21: UINT32     |        1 | deepseek2.embedding_length = 7168
     22: UINT32     |        1 | deepseek2.feed_forward_length = 18432
     23: UINT32     |        1 | deepseek2.attention.head_count = 64
     24: UINT32     |        1 | deepseek2.attention.head_count_kv = 1
     25: FLOAT32    |        1 | deepseek2.rope.freq_base = 50000.0
     26: FLOAT32    |        1 | deepseek2.attention.layer_norm_rms_epsilon = 9.999999974752427e-07
     27: UINT32     |        1 | deepseek2.expert_used_count = 8
     28: UINT32     |        1 | deepseek2.leading_dense_block_count = 1
     29: UINT32     |        1 | deepseek2.vocab_size = 163840
     30: UINT32     |        1 | deepseek2.attention.q_lora_rank = 1536
     31: UINT32     |        1 | deepseek2.attention.kv_lora_rank = 512
     32: UINT32     |        1 | deepseek2.attention.key_length = 576
     33: UINT32     |        1 | deepseek2.attention.value_length = 512
     34: UINT32     |        1 | deepseek2.attention.key_length_mla = 192
     35: UINT32     |        1 | deepseek2.attention.value_length_mla = 128
     36: UINT32     |        1 | deepseek2.expert_feed_forward_length = 2048
     37: UINT32     |        1 | deepseek2.expert_count = 384
     38: UINT32     |        1 | deepseek2.expert_shared_count = 1
     39: FLOAT32    |        1 | deepseek2.expert_weights_scale = 2.8269999027252197
     40: BOOL       |        1 | deepseek2.expert_weights_norm = True
     41: UINT32     |        1 | deepseek2.expert_gating_func = 2
     42: UINT32     |        1 | deepseek2.rope.dimension_count = 64
     43: STRING     |        1 | deepseek2.rope.scaling.type = 'yarn'
     44: FLOAT32    |        1 | deepseek2.rope.scaling.factor = 32.0
     45: UINT32     |        1 | deepseek2.rope.scaling.original_context_length = 4096
     46: FLOAT32    |        1 | deepseek2.rope.scaling.yarn_log_multiplier = 0.10000000149011612
     47: STRING     |        1 | tokenizer.ggml.model = 'gpt2'
     48: STRING     |        1 | tokenizer.ggml.pre = 'kimi-k2'
     49: [STRING]   |   163840 | tokenizer.ggml.tokens
     50: [INT32]    |   163840 | tokenizer.ggml.token_type
     51: [STRING]   |   163328 | tokenizer.ggml.merges
     52: UINT32     |        1 | tokenizer.ggml.bos_token_id = 163584
     53: UINT32     |        1 | tokenizer.ggml.eos_token_id = 163585
     54: UINT32     |        1 | tokenizer.ggml.padding_token_id = 163839
     55: STRING     |        1 | tokenizer.chat_template = '{%- if tools -%}\n  <|im_system|>tool_declare<|im_middle|>{{ '
     56: UINT32     |        1 | general.quantization_version = 2
     57: UINT32     |        1 | general.file_type = 23
     58: STRING     |        1 | quantize.imatrix.file = 'Kimi-K2-Instruct-GGUF/imatrix_unsloth.dat'
     59: STRING     |        1 | quantize.imatrix.dataset = 'unsloth_calibration_Kimi-K2-Instruct.txt'
     60: UINT32     |        1 | quantize.imatrix.entries_count = 667
     61: UINT32     |        1 | quantize.imatrix.chunks_count = 714
     62: UINT16     |        1 | split.no = 0
     63: INT32      |        1 | split.tensors.count = 1096
     64: UINT16     |        1 | split.count = 9
* Dumping 134 tensor(s)
      1: 1174405120 |  7168, 163840,     1,     1 | Q6_K    | output.weight
      2:       7168 |  7168,     1,     1,     1 | F32     | output_norm.weight
      3: 1174405120 |  7168, 163840,     1,     1 | Q4_K    | token_embd.weight
      4:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.0.attn_k_b.weight
      5:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.0.attn_kv_a_mqa.weight
      6:        512 |   512,     1,     1,     1 | F32     | blk.0.attn_kv_a_norm.weight
      7:       7168 |  7168,     1,     1,     1 | F32     | blk.0.attn_norm.weight
      8:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.0.attn_output.weight
      9:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.0.attn_q_a.weight
     10:       1536 |  1536,     1,     1,     1 | F32     | blk.0.attn_q_a_norm.weight
     11:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.0.attn_q_b.weight
     12:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.0.attn_v_b.weight
     13:  132120576 | 18432,  7168,     1,     1 | IQ4_XS  | blk.0.ffn_down.weight
     14:  132120576 |  7168, 18432,     1,     1 | IQ4_XS  | blk.0.ffn_gate.weight
     15:       7168 |  7168,     1,     1,     1 | F32     | blk.0.ffn_norm.weight
     16:  132120576 |  7168, 18432,     1,     1 | IQ4_XS  | blk.0.ffn_up.weight
     17:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.1.attn_k_b.weight
     18:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.1.attn_kv_a_mqa.weight
     19:        512 |   512,     1,     1,     1 | F32     | blk.1.attn_kv_a_norm.weight
     20:       7168 |  7168,     1,     1,     1 | F32     | blk.1.attn_norm.weight
     21:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.1.attn_output.weight
     22:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.1.attn_q_a.weight
     23:       1536 |  1536,     1,     1,     1 | F32     | blk.1.attn_q_a_norm.weight
     24:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.1.attn_q_b.weight
     25:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.1.attn_v_b.weight
     26:        384 |   384,     1,     1,     1 | F32     | blk.1.exp_probs_b.bias
     27: 5637144576 |  2048,  7168,   384,     1 | IQ4_XS  | blk.1.ffn_down_exps.weight
     28:   14680064 |  2048,  7168,     1,     1 | Q6_K    | blk.1.ffn_down_shexp.weight
     29: 5637144576 |  7168,  2048,   384,     1 | IQ3_S   | blk.1.ffn_gate_exps.weight
     30:    2752512 |  7168,   384,     1,     1 | F32     | blk.1.ffn_gate_inp.weight
     31:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.1.ffn_gate_shexp.weight
     32:       7168 |  7168,     1,     1,     1 | F32     | blk.1.ffn_norm.weight
     33: 5637144576 |  7168,  2048,   384,     1 | IQ3_S   | blk.1.ffn_up_exps.weight
     34:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.1.ffn_up_shexp.weight
     35:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.2.attn_k_b.weight
     36:    4128768 |  7168,   576,     1,     1 | Q5_K    | blk.2.attn_kv_a_mqa.weight
     37:        512 |   512,     1,     1,     1 | F32     | blk.2.attn_kv_a_norm.weight
     38:       7168 |  7168,     1,     1,     1 | F32     | blk.2.attn_norm.weight
     39:   58720256 |  8192,  7168,     1,     1 | Q5_K    | blk.2.attn_output.weight
     40:   11010048 |  7168,  1536,     1,     1 | Q5_K    | blk.2.attn_q_a.weight
     41:       1536 |  1536,     1,     1,     1 | F32     | blk.2.attn_q_a_norm.weight
     42:   18874368 |  1536, 12288,     1,     1 | Q5_K    | blk.2.attn_q_b.weight
     43:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.2.attn_v_b.weight
     44:        384 |   384,     1,     1,     1 | F32     | blk.2.exp_probs_b.bias
     45: 5637144576 |  2048,  7168,   384,     1 | IQ4_XS  | blk.2.ffn_down_exps.weight
     46:   14680064 |  2048,  7168,     1,     1 | Q6_K    | blk.2.ffn_down_shexp.weight
     47: 5637144576 |  7168,  2048,   384,     1 | IQ3_S   | blk.2.ffn_gate_exps.weight
     48:    2752512 |  7168,   384,     1,     1 | F32     | blk.2.ffn_gate_inp.weight
     49:   14680064 |  7168,  2048,     1,     1 | Q4_K    | blk.2.ffn_gate_shexp.weight
     50:       7168 |  7168,     1,     1,     1 | F32     | blk.2.ffn_norm.weight
     51: 5637144576 |  7168,  2048,   384,     1 | IQ3_S   | blk.2.ffn_up_exps.weight
     52:   14680064 |  7168,  2048,     1,     1 | Q4_K    | blk.2.ffn_up_shexp.weight
     53:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.3.attn_k_b.weight
     54:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.3.attn_kv_a_mqa.weight
     55:        512 |   512,     1,     1,     1 | F32     | blk.3.attn_kv_a_norm.weight
     56:       7168 |  7168,     1,     1,     1 | F32     | blk.3.attn_norm.weight
     57:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.3.attn_output.weight
     58:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.3.attn_q_a.weight
     59:       1536 |  1536,     1,     1,     1 | F32     | blk.3.attn_q_a_norm.weight
     60:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.3.attn_q_b.weight
     61:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.3.attn_v_b.weight
     62:        384 |   384,     1,     1,     1 | F32     | blk.3.exp_probs_b.bias
     63: 5637144576 |  2048,  7168,   384,     1 | IQ4_XS  | blk.3.ffn_down_exps.weight
     64:   14680064 |  2048,  7168,     1,     1 | Q5_K    | blk.3.ffn_down_shexp.weight
     65: 5637144576 |  7168,  2048,   384,     1 | IQ3_S   | blk.3.ffn_gate_exps.weight
     66:    2752512 |  7168,   384,     1,     1 | F32     | blk.3.ffn_gate_inp.weight
     67:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.3.ffn_gate_shexp.weight
     68:       7168 |  7168,     1,     1,     1 | F32     | blk.3.ffn_norm.weight
     69: 5637144576 |  7168,  2048,   384,     1 | IQ3_S   | blk.3.ffn_up_exps.weight
     70:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.3.ffn_up_shexp.weight
     71:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.4.attn_k_b.weight
     72:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.4.attn_kv_a_mqa.weight
     73:        512 |   512,     1,     1,     1 | F32     | blk.4.attn_kv_a_norm.weight
     74:       7168 |  7168,     1,     1,     1 | F32     | blk.4.attn_norm.weight
     75:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.4.attn_output.weight
     76:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.4.attn_q_a.weight
     77:       1536 |  1536,     1,     1,     1 | F32     | blk.4.attn_q_a_norm.weight
     78:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.4.attn_q_b.weight
     79:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.4.attn_v_b.weight
     80:        384 |   384,     1,     1,     1 | F32     | blk.4.exp_probs_b.bias
     81: 5637144576 |  2048,  7168,   384,     1 | IQ3_S   | blk.4.ffn_down_exps.weight
     82:   14680064 |  2048,  7168,     1,     1 | IQ4_XS  | blk.4.ffn_down_shexp.weight
     83: 5637144576 |  7168,  2048,   384,     1 | IQ3_S   | blk.4.ffn_gate_exps.weight
     84:    2752512 |  7168,   384,     1,     1 | F32     | blk.4.ffn_gate_inp.weight
     85:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.4.ffn_gate_shexp.weight
     86:       7168 |  7168,     1,     1,     1 | F32     | blk.4.ffn_norm.weight
     87: 5637144576 |  7168,  2048,   384,     1 | IQ3_S   | blk.4.ffn_up_exps.weight
     88:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.4.ffn_up_shexp.weight
     89:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.5.attn_k_b.weight
     90:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.5.attn_kv_a_mqa.weight
     91:        512 |   512,     1,     1,     1 | F32     | blk.5.attn_kv_a_norm.weight
     92:       7168 |  7168,     1,     1,     1 | F32     | blk.5.attn_norm.weight
     93:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.5.attn_output.weight
     94:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.5.attn_q_a.weight
     95:       1536 |  1536,     1,     1,     1 | F32     | blk.5.attn_q_a_norm.weight
     96:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.5.attn_q_b.weight
     97:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.5.attn_v_b.weight
     98:        384 |   384,     1,     1,     1 | F32     | blk.5.exp_probs_b.bias
     99: 5637144576 |  2048,  7168,   384,     1 | IQ4_XS  | blk.5.ffn_down_exps.weight
    100:   14680064 |  2048,  7168,     1,     1 | Q5_K    | blk.5.ffn_down_shexp.weight
    101: 5637144576 |  7168,  2048,   384,     1 | IQ3_S   | blk.5.ffn_gate_exps.weight
    102:    2752512 |  7168,   384,     1,     1 | F32     | blk.5.ffn_gate_inp.weight
    103:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.5.ffn_gate_shexp.weight
    104:       7168 |  7168,     1,     1,     1 | F32     | blk.5.ffn_norm.weight
    105: 5637144576 |  7168,  2048,   384,     1 | IQ3_S   | blk.5.ffn_up_exps.weight
    106:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.5.ffn_up_shexp.weight
    107:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.6.attn_k_b.weight
    108:    4128768 |  7168,   576,     1,     1 | Q5_K    | blk.6.attn_kv_a_mqa.weight
    109:        512 |   512,     1,     1,     1 | F32     | blk.6.attn_kv_a_norm.weight
    110:       7168 |  7168,     1,     1,     1 | F32     | blk.6.attn_norm.weight
    111:   58720256 |  8192,  7168,     1,     1 | Q5_K    | blk.6.attn_output.weight
    112:   11010048 |  7168,  1536,     1,     1 | Q5_K    | blk.6.attn_q_a.weight
    113:       1536 |  1536,     1,     1,     1 | F32     | blk.6.attn_q_a_norm.weight
    114:   18874368 |  1536, 12288,     1,     1 | Q5_K    | blk.6.attn_q_b.weight
    115:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.6.attn_v_b.weight
    116:        384 |   384,     1,     1,     1 | F32     | blk.6.exp_probs_b.bias
    117: 5637144576 |  2048,  7168,   384,     1 | IQ4_XS  | blk.6.ffn_down_exps.weight
    118:   14680064 |  2048,  7168,     1,     1 | Q6_K    | blk.6.ffn_down_shexp.weight
    119: 5637144576 |  7168,  2048,   384,     1 | IQ3_S   | blk.6.ffn_gate_exps.weight
    120:    2752512 |  7168,   384,     1,     1 | F32     | blk.6.ffn_gate_inp.weight
    121:   14680064 |  7168,  2048,     1,     1 | Q4_K    | blk.6.ffn_gate_shexp.weight
    122:       7168 |  7168,     1,     1,     1 | F32     | blk.6.ffn_norm.weight
    123: 5637144576 |  7168,  2048,   384,     1 | IQ3_S   | blk.6.ffn_up_exps.weight
    124:   14680064 |  7168,  2048,     1,     1 | Q4_K    | blk.6.ffn_up_shexp.weight
    125:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.7.attn_k_b.weight
    126:    4128768 |  7168,   576,     1,     1 | Q5_K    | blk.7.attn_kv_a_mqa.weight
    127:        512 |   512,     1,     1,     1 | F32     | blk.7.attn_kv_a_norm.weight
    128:       7168 |  7168,     1,     1,     1 | F32     | blk.7.attn_norm.weight
    129:   58720256 |  8192,  7168,     1,     1 | Q5_K    | blk.7.attn_output.weight
    130:   11010048 |  7168,  1536,     1,     1 | Q5_K    | blk.7.attn_q_a.weight
    131:       1536 |  1536,     1,     1,     1 | F32     | blk.7.attn_q_a_norm.weight
    132:   18874368 |  1536, 12288,     1,     1 | Q5_K    | blk.7.attn_q_b.weight
    133:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.7.attn_v_b.weight
    134:        384 |   384,     1,     1,     1 | F32     | blk.7.exp_probs_b.bias
INFO:gguf-dump:* Loading: /mnt/data/models/unsloth/Kimi-K2-Instruct-GGUF/UD-IQ3_XXS/Kimi-K2-Instruct-UD-IQ3_XXS-00002-of-00009.gguf
* File is LITTLE endian, script is running on a LITTLE endian host.
* Dumping 6 key/value pair(s)
      1: UINT32     |        1 | GGUF.version = 3
      2: UINT64     |        1 | GGUF.tensor_count = 128
      3: UINT64     |        1 | GGUF.kv_count = 3
      4: UINT16     |        1 | split.no = 1
      5: INT32      |        1 | split.tensors.count = 1096
      6: UINT16     |        1 | split.count = 9
* Dumping 128 tensor(s)
      1: 5637144576 |  2048,  7168,   384,     1 | IQ4_XS  | blk.7.ffn_down_exps.weight
      2:   14680064 |  2048,  7168,     1,     1 | Q5_K    | blk.7.ffn_down_shexp.weight
      3: 5637144576 |  7168,  2048,   384,     1 | IQ3_S   | blk.7.ffn_gate_exps.weight
      4:    2752512 |  7168,   384,     1,     1 | F32     | blk.7.ffn_gate_inp.weight
      5:   14680064 |  7168,  2048,     1,     1 | Q4_K    | blk.7.ffn_gate_shexp.weight
      6:       7168 |  7168,     1,     1,     1 | F32     | blk.7.ffn_norm.weight
      7: 5637144576 |  7168,  2048,   384,     1 | IQ3_S   | blk.7.ffn_up_exps.weight
      8:   14680064 |  7168,  2048,     1,     1 | Q4_K    | blk.7.ffn_up_shexp.weight
      9:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.8.attn_k_b.weight
     10:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.8.attn_kv_a_mqa.weight
     11:        512 |   512,     1,     1,     1 | F32     | blk.8.attn_kv_a_norm.weight
     12:       7168 |  7168,     1,     1,     1 | F32     | blk.8.attn_norm.weight
     13:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.8.attn_output.weight
     14:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.8.attn_q_a.weight
     15:       1536 |  1536,     1,     1,     1 | F32     | blk.8.attn_q_a_norm.weight
     16:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.8.attn_q_b.weight
     17:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.8.attn_v_b.weight
     18:        384 |   384,     1,     1,     1 | F32     | blk.8.exp_probs_b.bias
     19: 5637144576 |  2048,  7168,   384,     1 | IQ3_XXS | blk.8.ffn_down_exps.weight
     20:   14680064 |  2048,  7168,     1,     1 | IQ4_XS  | blk.8.ffn_down_shexp.weight
     21: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.8.ffn_gate_exps.weight
     22:    2752512 |  7168,   384,     1,     1 | F32     | blk.8.ffn_gate_inp.weight
     23:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.8.ffn_gate_shexp.weight
     24:       7168 |  7168,     1,     1,     1 | F32     | blk.8.ffn_norm.weight
     25: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.8.ffn_up_exps.weight
     26:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.8.ffn_up_shexp.weight
     27:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.9.attn_k_b.weight
     28:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.9.attn_kv_a_mqa.weight
     29:        512 |   512,     1,     1,     1 | F32     | blk.9.attn_kv_a_norm.weight
     30:       7168 |  7168,     1,     1,     1 | F32     | blk.9.attn_norm.weight
     31:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.9.attn_output.weight
     32:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.9.attn_q_a.weight
     33:       1536 |  1536,     1,     1,     1 | F32     | blk.9.attn_q_a_norm.weight
     34:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.9.attn_q_b.weight
     35:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.9.attn_v_b.weight
     36:        384 |   384,     1,     1,     1 | F32     | blk.9.exp_probs_b.bias
     37: 5637144576 |  2048,  7168,   384,     1 | IQ3_XXS | blk.9.ffn_down_exps.weight
     38:   14680064 |  2048,  7168,     1,     1 | IQ4_XS  | blk.9.ffn_down_shexp.weight
     39: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.9.ffn_gate_exps.weight
     40:    2752512 |  7168,   384,     1,     1 | F32     | blk.9.ffn_gate_inp.weight
     41:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.9.ffn_gate_shexp.weight
     42:       7168 |  7168,     1,     1,     1 | F32     | blk.9.ffn_norm.weight
     43: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.9.ffn_up_exps.weight
     44:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.9.ffn_up_shexp.weight
     45:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.10.attn_k_b.weight
     46:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.10.attn_kv_a_mqa.weight
     47:        512 |   512,     1,     1,     1 | F32     | blk.10.attn_kv_a_norm.weight
     48:       7168 |  7168,     1,     1,     1 | F32     | blk.10.attn_norm.weight
     49:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.10.attn_output.weight
     50:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.10.attn_q_a.weight
     51:       1536 |  1536,     1,     1,     1 | F32     | blk.10.attn_q_a_norm.weight
     52:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.10.attn_q_b.weight
     53:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.10.attn_v_b.weight
     54:        384 |   384,     1,     1,     1 | F32     | blk.10.exp_probs_b.bias
     55: 5637144576 |  2048,  7168,   384,     1 | IQ3_XXS | blk.10.ffn_down_exps.weight
     56:   14680064 |  2048,  7168,     1,     1 | IQ4_XS  | blk.10.ffn_down_shexp.weight
     57: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.10.ffn_gate_exps.weight
     58:    2752512 |  7168,   384,     1,     1 | F32     | blk.10.ffn_gate_inp.weight
     59:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.10.ffn_gate_shexp.weight
     60:       7168 |  7168,     1,     1,     1 | F32     | blk.10.ffn_norm.weight
     61: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.10.ffn_up_exps.weight
     62:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.10.ffn_up_shexp.weight
     63:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.11.attn_k_b.weight
     64:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.11.attn_kv_a_mqa.weight
     65:        512 |   512,     1,     1,     1 | F32     | blk.11.attn_kv_a_norm.weight
     66:       7168 |  7168,     1,     1,     1 | F32     | blk.11.attn_norm.weight
     67:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.11.attn_output.weight
     68:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.11.attn_q_a.weight
     69:       1536 |  1536,     1,     1,     1 | F32     | blk.11.attn_q_a_norm.weight
     70:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.11.attn_q_b.weight
     71:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.11.attn_v_b.weight
     72:        384 |   384,     1,     1,     1 | F32     | blk.11.exp_probs_b.bias
     73: 5637144576 |  2048,  7168,   384,     1 | IQ3_XXS | blk.11.ffn_down_exps.weight
     74:   14680064 |  2048,  7168,     1,     1 | IQ4_XS  | blk.11.ffn_down_shexp.weight
     75: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.11.ffn_gate_exps.weight
     76:    2752512 |  7168,   384,     1,     1 | F32     | blk.11.ffn_gate_inp.weight
     77:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.11.ffn_gate_shexp.weight
     78:       7168 |  7168,     1,     1,     1 | F32     | blk.11.ffn_norm.weight
     79: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.11.ffn_up_exps.weight
     80:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.11.ffn_up_shexp.weight
     81:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.12.attn_k_b.weight
     82:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.12.attn_kv_a_mqa.weight
     83:        512 |   512,     1,     1,     1 | F32     | blk.12.attn_kv_a_norm.weight
     84:       7168 |  7168,     1,     1,     1 | F32     | blk.12.attn_norm.weight
     85:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.12.attn_output.weight
     86:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.12.attn_q_a.weight
     87:       1536 |  1536,     1,     1,     1 | F32     | blk.12.attn_q_a_norm.weight
     88:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.12.attn_q_b.weight
     89:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.12.attn_v_b.weight
     90:        384 |   384,     1,     1,     1 | F32     | blk.12.exp_probs_b.bias
     91: 5637144576 |  2048,  7168,   384,     1 | IQ3_S   | blk.12.ffn_down_exps.weight
     92:   14680064 |  2048,  7168,     1,     1 | Q5_K    | blk.12.ffn_down_shexp.weight
     93: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.12.ffn_gate_exps.weight
     94:    2752512 |  7168,   384,     1,     1 | F32     | blk.12.ffn_gate_inp.weight
     95:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.12.ffn_gate_shexp.weight
     96:       7168 |  7168,     1,     1,     1 | F32     | blk.12.ffn_norm.weight
     97: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.12.ffn_up_exps.weight
     98:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.12.ffn_up_shexp.weight
     99:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.13.attn_k_b.weight
    100:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.13.attn_kv_a_mqa.weight
    101:        512 |   512,     1,     1,     1 | F32     | blk.13.attn_kv_a_norm.weight
    102:       7168 |  7168,     1,     1,     1 | F32     | blk.13.attn_norm.weight
    103:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.13.attn_output.weight
    104:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.13.attn_q_a.weight
    105:       1536 |  1536,     1,     1,     1 | F32     | blk.13.attn_q_a_norm.weight
    106:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.13.attn_q_b.weight
    107:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.13.attn_v_b.weight
    108:        384 |   384,     1,     1,     1 | F32     | blk.13.exp_probs_b.bias
    109: 5637144576 |  2048,  7168,   384,     1 | IQ3_XXS | blk.13.ffn_down_exps.weight
    110:   14680064 |  2048,  7168,     1,     1 | IQ4_XS  | blk.13.ffn_down_shexp.weight
    111: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.13.ffn_gate_exps.weight
    112:    2752512 |  7168,   384,     1,     1 | F32     | blk.13.ffn_gate_inp.weight
    113:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.13.ffn_gate_shexp.weight
    114:       7168 |  7168,     1,     1,     1 | F32     | blk.13.ffn_norm.weight
    115: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.13.ffn_up_exps.weight
    116:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.13.ffn_up_shexp.weight
    117:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.14.attn_k_b.weight
    118:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.14.attn_kv_a_mqa.weight
    119:        512 |   512,     1,     1,     1 | F32     | blk.14.attn_kv_a_norm.weight
    120:       7168 |  7168,     1,     1,     1 | F32     | blk.14.attn_norm.weight
    121:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.14.attn_output.weight
    122:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.14.attn_q_a.weight
    123:       1536 |  1536,     1,     1,     1 | F32     | blk.14.attn_q_a_norm.weight
    124:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.14.attn_q_b.weight
    125:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.14.attn_v_b.weight
    126:        384 |   384,     1,     1,     1 | F32     | blk.14.exp_probs_b.bias
    127: 5637144576 |  2048,  7168,   384,     1 | IQ3_XXS | blk.14.ffn_down_exps.weight
    128:   14680064 |  2048,  7168,     1,     1 | IQ4_XS  | blk.14.ffn_down_shexp.weight
INFO:gguf-dump:* Loading: /mnt/data/models/unsloth/Kimi-K2-Instruct-GGUF/UD-IQ3_XXS/Kimi-K2-Instruct-UD-IQ3_XXS-00003-of-00009.gguf
* File is LITTLE endian, script is running on a LITTLE endian host.
* Dumping 6 key/value pair(s)
      1: UINT32     |        1 | GGUF.version = 3
      2: UINT64     |        1 | GGUF.tensor_count = 126
      3: UINT64     |        1 | GGUF.kv_count = 3
      4: UINT16     |        1 | split.no = 2
      5: INT32      |        1 | split.tensors.count = 1096
      6: UINT16     |        1 | split.count = 9
* Dumping 126 tensor(s)
      1: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.14.ffn_gate_exps.weight
      2:    2752512 |  7168,   384,     1,     1 | F32     | blk.14.ffn_gate_inp.weight
      3:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.14.ffn_gate_shexp.weight
      4:       7168 |  7168,     1,     1,     1 | F32     | blk.14.ffn_norm.weight
      5: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.14.ffn_up_exps.weight
      6:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.14.ffn_up_shexp.weight
      7:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.15.attn_k_b.weight
      8:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.15.attn_kv_a_mqa.weight
      9:        512 |   512,     1,     1,     1 | F32     | blk.15.attn_kv_a_norm.weight
     10:       7168 |  7168,     1,     1,     1 | F32     | blk.15.attn_norm.weight
     11:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.15.attn_output.weight
     12:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.15.attn_q_a.weight
     13:       1536 |  1536,     1,     1,     1 | F32     | blk.15.attn_q_a_norm.weight
     14:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.15.attn_q_b.weight
     15:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.15.attn_v_b.weight
     16:        384 |   384,     1,     1,     1 | F32     | blk.15.exp_probs_b.bias
     17: 5637144576 |  2048,  7168,   384,     1 | IQ4_XS  | blk.15.ffn_down_exps.weight
     18:   14680064 |  2048,  7168,     1,     1 | Q6_K    | blk.15.ffn_down_shexp.weight
     19: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.15.ffn_gate_exps.weight
     20:    2752512 |  7168,   384,     1,     1 | F32     | blk.15.ffn_gate_inp.weight
     21:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.15.ffn_gate_shexp.weight
     22:       7168 |  7168,     1,     1,     1 | F32     | blk.15.ffn_norm.weight
     23: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.15.ffn_up_exps.weight
     24:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.15.ffn_up_shexp.weight
     25:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.16.attn_k_b.weight
     26:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.16.attn_kv_a_mqa.weight
     27:        512 |   512,     1,     1,     1 | F32     | blk.16.attn_kv_a_norm.weight
     28:       7168 |  7168,     1,     1,     1 | F32     | blk.16.attn_norm.weight
     29:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.16.attn_output.weight
     30:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.16.attn_q_a.weight
     31:       1536 |  1536,     1,     1,     1 | F32     | blk.16.attn_q_a_norm.weight
     32:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.16.attn_q_b.weight
     33:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.16.attn_v_b.weight
     34:        384 |   384,     1,     1,     1 | F32     | blk.16.exp_probs_b.bias
     35: 5637144576 |  2048,  7168,   384,     1 | IQ3_XXS | blk.16.ffn_down_exps.weight
     36:   14680064 |  2048,  7168,     1,     1 | IQ4_XS  | blk.16.ffn_down_shexp.weight
     37: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.16.ffn_gate_exps.weight
     38:    2752512 |  7168,   384,     1,     1 | F32     | blk.16.ffn_gate_inp.weight
     39:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.16.ffn_gate_shexp.weight
     40:       7168 |  7168,     1,     1,     1 | F32     | blk.16.ffn_norm.weight
     41: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.16.ffn_up_exps.weight
     42:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.16.ffn_up_shexp.weight
     43:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.17.attn_k_b.weight
     44:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.17.attn_kv_a_mqa.weight
     45:        512 |   512,     1,     1,     1 | F32     | blk.17.attn_kv_a_norm.weight
     46:       7168 |  7168,     1,     1,     1 | F32     | blk.17.attn_norm.weight
     47:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.17.attn_output.weight
     48:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.17.attn_q_a.weight
     49:       1536 |  1536,     1,     1,     1 | F32     | blk.17.attn_q_a_norm.weight
     50:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.17.attn_q_b.weight
     51:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.17.attn_v_b.weight
     52:        384 |   384,     1,     1,     1 | F32     | blk.17.exp_probs_b.bias
     53: 5637144576 |  2048,  7168,   384,     1 | IQ3_S   | blk.17.ffn_down_exps.weight
     54:   14680064 |  2048,  7168,     1,     1 | Q5_K    | blk.17.ffn_down_shexp.weight
     55: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.17.ffn_gate_exps.weight
     56:    2752512 |  7168,   384,     1,     1 | F32     | blk.17.ffn_gate_inp.weight
     57:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.17.ffn_gate_shexp.weight
     58:       7168 |  7168,     1,     1,     1 | F32     | blk.17.ffn_norm.weight
     59: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.17.ffn_up_exps.weight
     60:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.17.ffn_up_shexp.weight
     61:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.18.attn_k_b.weight
     62:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.18.attn_kv_a_mqa.weight
     63:        512 |   512,     1,     1,     1 | F32     | blk.18.attn_kv_a_norm.weight
     64:       7168 |  7168,     1,     1,     1 | F32     | blk.18.attn_norm.weight
     65:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.18.attn_output.weight
     66:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.18.attn_q_a.weight
     67:       1536 |  1536,     1,     1,     1 | F32     | blk.18.attn_q_a_norm.weight
     68:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.18.attn_q_b.weight
     69:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.18.attn_v_b.weight
     70:        384 |   384,     1,     1,     1 | F32     | blk.18.exp_probs_b.bias
     71: 5637144576 |  2048,  7168,   384,     1 | IQ3_S   | blk.18.ffn_down_exps.weight
     72:   14680064 |  2048,  7168,     1,     1 | Q5_K    | blk.18.ffn_down_shexp.weight
     73: 5637144576 |  7168,  2048,   384,     1 | IQ3_S   | blk.18.ffn_gate_exps.weight
     74:    2752512 |  7168,   384,     1,     1 | F32     | blk.18.ffn_gate_inp.weight
     75:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.18.ffn_gate_shexp.weight
     76:       7168 |  7168,     1,     1,     1 | F32     | blk.18.ffn_norm.weight
     77: 5637144576 |  7168,  2048,   384,     1 | IQ3_S   | blk.18.ffn_up_exps.weight
     78:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.18.ffn_up_shexp.weight
     79:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.19.attn_k_b.weight
     80:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.19.attn_kv_a_mqa.weight
     81:        512 |   512,     1,     1,     1 | F32     | blk.19.attn_kv_a_norm.weight
     82:       7168 |  7168,     1,     1,     1 | F32     | blk.19.attn_norm.weight
     83:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.19.attn_output.weight
     84:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.19.attn_q_a.weight
     85:       1536 |  1536,     1,     1,     1 | F32     | blk.19.attn_q_a_norm.weight
     86:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.19.attn_q_b.weight
     87:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.19.attn_v_b.weight
     88:        384 |   384,     1,     1,     1 | F32     | blk.19.exp_probs_b.bias
     89: 5637144576 |  2048,  7168,   384,     1 | IQ3_XXS | blk.19.ffn_down_exps.weight
     90:   14680064 |  2048,  7168,     1,     1 | IQ4_XS  | blk.19.ffn_down_shexp.weight
     91: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.19.ffn_gate_exps.weight
     92:    2752512 |  7168,   384,     1,     1 | F32     | blk.19.ffn_gate_inp.weight
     93:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.19.ffn_gate_shexp.weight
     94:       7168 |  7168,     1,     1,     1 | F32     | blk.19.ffn_norm.weight
     95: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.19.ffn_up_exps.weight
     96:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.19.ffn_up_shexp.weight
     97:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.20.attn_k_b.weight
     98:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.20.attn_kv_a_mqa.weight
     99:        512 |   512,     1,     1,     1 | F32     | blk.20.attn_kv_a_norm.weight
    100:       7168 |  7168,     1,     1,     1 | F32     | blk.20.attn_norm.weight
    101:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.20.attn_output.weight
    102:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.20.attn_q_a.weight
    103:       1536 |  1536,     1,     1,     1 | F32     | blk.20.attn_q_a_norm.weight
    104:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.20.attn_q_b.weight
    105:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.20.attn_v_b.weight
    106:        384 |   384,     1,     1,     1 | F32     | blk.20.exp_probs_b.bias
    107: 5637144576 |  2048,  7168,   384,     1 | IQ3_S   | blk.20.ffn_down_exps.weight
    108:   14680064 |  2048,  7168,     1,     1 | Q5_K    | blk.20.ffn_down_shexp.weight
    109: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.20.ffn_gate_exps.weight
    110:    2752512 |  7168,   384,     1,     1 | F32     | blk.20.ffn_gate_inp.weight
    111:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.20.ffn_gate_shexp.weight
    112:       7168 |  7168,     1,     1,     1 | F32     | blk.20.ffn_norm.weight
    113: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.20.ffn_up_exps.weight
    114:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.20.ffn_up_shexp.weight
    115:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.21.attn_k_b.weight
    116:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.21.attn_kv_a_mqa.weight
    117:        512 |   512,     1,     1,     1 | F32     | blk.21.attn_kv_a_norm.weight
    118:       7168 |  7168,     1,     1,     1 | F32     | blk.21.attn_norm.weight
    119:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.21.attn_output.weight
    120:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.21.attn_q_a.weight
    121:       1536 |  1536,     1,     1,     1 | F32     | blk.21.attn_q_a_norm.weight
    122:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.21.attn_q_b.weight
    123:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.21.attn_v_b.weight
    124:        384 |   384,     1,     1,     1 | F32     | blk.21.exp_probs_b.bias
    125: 5637144576 |  2048,  7168,   384,     1 | IQ3_XXS | blk.21.ffn_down_exps.weight
    126:   14680064 |  2048,  7168,     1,     1 | IQ4_XS  | blk.21.ffn_down_shexp.weight
INFO:gguf-dump:* Loading: /mnt/data/models/unsloth/Kimi-K2-Instruct-GGUF/UD-IQ3_XXS/Kimi-K2-Instruct-UD-IQ3_XXS-00004-of-00009.gguf
* File is LITTLE endian, script is running on a LITTLE endian host.
* Dumping 6 key/value pair(s)
      1: UINT32     |        1 | GGUF.version = 3
      2: UINT64     |        1 | GGUF.tensor_count = 130
      3: UINT64     |        1 | GGUF.kv_count = 3
      4: UINT16     |        1 | split.no = 3
      5: INT32      |        1 | split.tensors.count = 1096
      6: UINT16     |        1 | split.count = 9
* Dumping 130 tensor(s)
      1: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.21.ffn_gate_exps.weight
      2:    2752512 |  7168,   384,     1,     1 | F32     | blk.21.ffn_gate_inp.weight
      3:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.21.ffn_gate_shexp.weight
      4:       7168 |  7168,     1,     1,     1 | F32     | blk.21.ffn_norm.weight
      5: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.21.ffn_up_exps.weight
      6:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.21.ffn_up_shexp.weight
      7:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.22.attn_k_b.weight
      8:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.22.attn_kv_a_mqa.weight
      9:        512 |   512,     1,     1,     1 | F32     | blk.22.attn_kv_a_norm.weight
     10:       7168 |  7168,     1,     1,     1 | F32     | blk.22.attn_norm.weight
     11:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.22.attn_output.weight
     12:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.22.attn_q_a.weight
     13:       1536 |  1536,     1,     1,     1 | F32     | blk.22.attn_q_a_norm.weight
     14:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.22.attn_q_b.weight
     15:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.22.attn_v_b.weight
     16:        384 |   384,     1,     1,     1 | F32     | blk.22.exp_probs_b.bias
     17: 5637144576 |  2048,  7168,   384,     1 | IQ3_XXS | blk.22.ffn_down_exps.weight
     18:   14680064 |  2048,  7168,     1,     1 | IQ4_XS  | blk.22.ffn_down_shexp.weight
     19: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.22.ffn_gate_exps.weight
     20:    2752512 |  7168,   384,     1,     1 | F32     | blk.22.ffn_gate_inp.weight
     21:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.22.ffn_gate_shexp.weight
     22:       7168 |  7168,     1,     1,     1 | F32     | blk.22.ffn_norm.weight
     23: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.22.ffn_up_exps.weight
     24:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.22.ffn_up_shexp.weight
     25:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.23.attn_k_b.weight
     26:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.23.attn_kv_a_mqa.weight
     27:        512 |   512,     1,     1,     1 | F32     | blk.23.attn_kv_a_norm.weight
     28:       7168 |  7168,     1,     1,     1 | F32     | blk.23.attn_norm.weight
     29:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.23.attn_output.weight
     30:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.23.attn_q_a.weight
     31:       1536 |  1536,     1,     1,     1 | F32     | blk.23.attn_q_a_norm.weight
     32:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.23.attn_q_b.weight
     33:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.23.attn_v_b.weight
     34:        384 |   384,     1,     1,     1 | F32     | blk.23.exp_probs_b.bias
     35: 5637144576 |  2048,  7168,   384,     1 | IQ3_S   | blk.23.ffn_down_exps.weight
     36:   14680064 |  2048,  7168,     1,     1 | Q5_K    | blk.23.ffn_down_shexp.weight
     37: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.23.ffn_gate_exps.weight
     38:    2752512 |  7168,   384,     1,     1 | F32     | blk.23.ffn_gate_inp.weight
     39:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.23.ffn_gate_shexp.weight
     40:       7168 |  7168,     1,     1,     1 | F32     | blk.23.ffn_norm.weight
     41: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.23.ffn_up_exps.weight
     42:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.23.ffn_up_shexp.weight
     43:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.24.attn_k_b.weight
     44:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.24.attn_kv_a_mqa.weight
     45:        512 |   512,     1,     1,     1 | F32     | blk.24.attn_kv_a_norm.weight
     46:       7168 |  7168,     1,     1,     1 | F32     | blk.24.attn_norm.weight
     47:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.24.attn_output.weight
     48:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.24.attn_q_a.weight
     49:       1536 |  1536,     1,     1,     1 | F32     | blk.24.attn_q_a_norm.weight
     50:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.24.attn_q_b.weight
     51:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.24.attn_v_b.weight
     52:        384 |   384,     1,     1,     1 | F32     | blk.24.exp_probs_b.bias
     53: 5637144576 |  2048,  7168,   384,     1 | IQ3_XXS | blk.24.ffn_down_exps.weight
     54:   14680064 |  2048,  7168,     1,     1 | IQ4_XS  | blk.24.ffn_down_shexp.weight
     55: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.24.ffn_gate_exps.weight
     56:    2752512 |  7168,   384,     1,     1 | F32     | blk.24.ffn_gate_inp.weight
     57:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.24.ffn_gate_shexp.weight
     58:       7168 |  7168,     1,     1,     1 | F32     | blk.24.ffn_norm.weight
     59: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.24.ffn_up_exps.weight
     60:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.24.ffn_up_shexp.weight
     61:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.25.attn_k_b.weight
     62:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.25.attn_kv_a_mqa.weight
     63:        512 |   512,     1,     1,     1 | F32     | blk.25.attn_kv_a_norm.weight
     64:       7168 |  7168,     1,     1,     1 | F32     | blk.25.attn_norm.weight
     65:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.25.attn_output.weight
     66:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.25.attn_q_a.weight
     67:       1536 |  1536,     1,     1,     1 | F32     | blk.25.attn_q_a_norm.weight
     68:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.25.attn_q_b.weight
     69:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.25.attn_v_b.weight
     70:        384 |   384,     1,     1,     1 | F32     | blk.25.exp_probs_b.bias
     71: 5637144576 |  2048,  7168,   384,     1 | IQ3_XXS | blk.25.ffn_down_exps.weight
     72:   14680064 |  2048,  7168,     1,     1 | IQ4_XS  | blk.25.ffn_down_shexp.weight
     73: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.25.ffn_gate_exps.weight
     74:    2752512 |  7168,   384,     1,     1 | F32     | blk.25.ffn_gate_inp.weight
     75:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.25.ffn_gate_shexp.weight
     76:       7168 |  7168,     1,     1,     1 | F32     | blk.25.ffn_norm.weight
     77: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.25.ffn_up_exps.weight
     78:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.25.ffn_up_shexp.weight
     79:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.26.attn_k_b.weight
     80:    4128768 |  7168,   576,     1,     1 | Q6_K    | blk.26.attn_kv_a_mqa.weight
     81:        512 |   512,     1,     1,     1 | F32     | blk.26.attn_kv_a_norm.weight
     82:       7168 |  7168,     1,     1,     1 | F32     | blk.26.attn_norm.weight
     83:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.26.attn_output.weight
     84:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.26.attn_q_a.weight
     85:       1536 |  1536,     1,     1,     1 | F32     | blk.26.attn_q_a_norm.weight
     86:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.26.attn_q_b.weight
     87:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.26.attn_v_b.weight
     88:        384 |   384,     1,     1,     1 | F32     | blk.26.exp_probs_b.bias
     89: 5637144576 |  2048,  7168,   384,     1 | IQ3_XXS | blk.26.ffn_down_exps.weight
     90:   14680064 |  2048,  7168,     1,     1 | IQ4_XS  | blk.26.ffn_down_shexp.weight
     91: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.26.ffn_gate_exps.weight
     92:    2752512 |  7168,   384,     1,     1 | F32     | blk.26.ffn_gate_inp.weight
     93:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.26.ffn_gate_shexp.weight
     94:       7168 |  7168,     1,     1,     1 | F32     | blk.26.ffn_norm.weight
     95: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.26.ffn_up_exps.weight
     96:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.26.ffn_up_shexp.weight
     97:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.27.attn_k_b.weight
     98:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.27.attn_kv_a_mqa.weight
     99:        512 |   512,     1,     1,     1 | F32     | blk.27.attn_kv_a_norm.weight
    100:       7168 |  7168,     1,     1,     1 | F32     | blk.27.attn_norm.weight
    101:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.27.attn_output.weight
    102:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.27.attn_q_a.weight
    103:       1536 |  1536,     1,     1,     1 | F32     | blk.27.attn_q_a_norm.weight
    104:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.27.attn_q_b.weight
    105:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.27.attn_v_b.weight
    106:        384 |   384,     1,     1,     1 | F32     | blk.27.exp_probs_b.bias
    107: 5637144576 |  2048,  7168,   384,     1 | IQ3_S   | blk.27.ffn_down_exps.weight
    108:   14680064 |  2048,  7168,     1,     1 | Q5_K    | blk.27.ffn_down_shexp.weight
    109: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.27.ffn_gate_exps.weight
    110:    2752512 |  7168,   384,     1,     1 | F32     | blk.27.ffn_gate_inp.weight
    111:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.27.ffn_gate_shexp.weight
    112:       7168 |  7168,     1,     1,     1 | F32     | blk.27.ffn_norm.weight
    113: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.27.ffn_up_exps.weight
    114:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.27.ffn_up_shexp.weight
    115:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.28.attn_k_b.weight
    116:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.28.attn_kv_a_mqa.weight
    117:        512 |   512,     1,     1,     1 | F32     | blk.28.attn_kv_a_norm.weight
    118:       7168 |  7168,     1,     1,     1 | F32     | blk.28.attn_norm.weight
    119:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.28.attn_output.weight
    120:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.28.attn_q_a.weight
    121:       1536 |  1536,     1,     1,     1 | F32     | blk.28.attn_q_a_norm.weight
    122:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.28.attn_q_b.weight
    123:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.28.attn_v_b.weight
    124:        384 |   384,     1,     1,     1 | F32     | blk.28.exp_probs_b.bias
    125: 5637144576 |  2048,  7168,   384,     1 | IQ3_XXS | blk.28.ffn_down_exps.weight
    126:   14680064 |  2048,  7168,     1,     1 | IQ4_XS  | blk.28.ffn_down_shexp.weight
    127: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.28.ffn_gate_exps.weight
    128:    2752512 |  7168,   384,     1,     1 | F32     | blk.28.ffn_gate_inp.weight
    129:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.28.ffn_gate_shexp.weight
    130:       7168 |  7168,     1,     1,     1 | F32     | blk.28.ffn_norm.weight
INFO:gguf-dump:* Loading: /mnt/data/models/unsloth/Kimi-K2-Instruct-GGUF/UD-IQ3_XXS/Kimi-K2-Instruct-UD-IQ3_XXS-00005-of-00009.gguf
* File is LITTLE endian, script is running on a LITTLE endian host.
* Dumping 6 key/value pair(s)
      1: UINT32     |        1 | GGUF.version = 3
      2: UINT64     |        1 | GGUF.tensor_count = 138
      3: UINT64     |        1 | GGUF.kv_count = 3
      4: UINT16     |        1 | split.no = 4
      5: INT32      |        1 | split.tensors.count = 1096
      6: UINT16     |        1 | split.count = 9
* Dumping 138 tensor(s)
      1: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.28.ffn_up_exps.weight
      2:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.28.ffn_up_shexp.weight
      3:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.29.attn_k_b.weight
      4:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.29.attn_kv_a_mqa.weight
      5:        512 |   512,     1,     1,     1 | F32     | blk.29.attn_kv_a_norm.weight
      6:       7168 |  7168,     1,     1,     1 | F32     | blk.29.attn_norm.weight
      7:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.29.attn_output.weight
      8:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.29.attn_q_a.weight
      9:       1536 |  1536,     1,     1,     1 | F32     | blk.29.attn_q_a_norm.weight
     10:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.29.attn_q_b.weight
     11:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.29.attn_v_b.weight
     12:        384 |   384,     1,     1,     1 | F32     | blk.29.exp_probs_b.bias
     13: 5637144576 |  2048,  7168,   384,     1 | IQ3_XXS | blk.29.ffn_down_exps.weight
     14:   14680064 |  2048,  7168,     1,     1 | IQ4_XS  | blk.29.ffn_down_shexp.weight
     15: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.29.ffn_gate_exps.weight
     16:    2752512 |  7168,   384,     1,     1 | F32     | blk.29.ffn_gate_inp.weight
     17:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.29.ffn_gate_shexp.weight
     18:       7168 |  7168,     1,     1,     1 | F32     | blk.29.ffn_norm.weight
     19: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.29.ffn_up_exps.weight
     20:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.29.ffn_up_shexp.weight
     21:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.30.attn_k_b.weight
     22:    4128768 |  7168,   576,     1,     1 | Q6_K    | blk.30.attn_kv_a_mqa.weight
     23:        512 |   512,     1,     1,     1 | F32     | blk.30.attn_kv_a_norm.weight
     24:       7168 |  7168,     1,     1,     1 | F32     | blk.30.attn_norm.weight
     25:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.30.attn_output.weight
     26:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.30.attn_q_a.weight
     27:       1536 |  1536,     1,     1,     1 | F32     | blk.30.attn_q_a_norm.weight
     28:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.30.attn_q_b.weight
     29:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.30.attn_v_b.weight
     30:        384 |   384,     1,     1,     1 | F32     | blk.30.exp_probs_b.bias
     31: 5637144576 |  2048,  7168,   384,     1 | IQ3_XXS | blk.30.ffn_down_exps.weight
     32:   14680064 |  2048,  7168,     1,     1 | IQ4_XS  | blk.30.ffn_down_shexp.weight
     33: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.30.ffn_gate_exps.weight
     34:    2752512 |  7168,   384,     1,     1 | F32     | blk.30.ffn_gate_inp.weight
     35:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.30.ffn_gate_shexp.weight
     36:       7168 |  7168,     1,     1,     1 | F32     | blk.30.ffn_norm.weight
     37: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.30.ffn_up_exps.weight
     38:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.30.ffn_up_shexp.weight
     39:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.31.attn_k_b.weight
     40:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.31.attn_kv_a_mqa.weight
     41:        512 |   512,     1,     1,     1 | F32     | blk.31.attn_kv_a_norm.weight
     42:       7168 |  7168,     1,     1,     1 | F32     | blk.31.attn_norm.weight
     43:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.31.attn_output.weight
     44:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.31.attn_q_a.weight
     45:       1536 |  1536,     1,     1,     1 | F32     | blk.31.attn_q_a_norm.weight
     46:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.31.attn_q_b.weight
     47:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.31.attn_v_b.weight
     48:        384 |   384,     1,     1,     1 | F32     | blk.31.exp_probs_b.bias
     49: 5637144576 |  2048,  7168,   384,     1 | IQ3_XXS | blk.31.ffn_down_exps.weight
     50:   14680064 |  2048,  7168,     1,     1 | IQ4_XS  | blk.31.ffn_down_shexp.weight
     51: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.31.ffn_gate_exps.weight
     52:    2752512 |  7168,   384,     1,     1 | F32     | blk.31.ffn_gate_inp.weight
     53:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.31.ffn_gate_shexp.weight
     54:       7168 |  7168,     1,     1,     1 | F32     | blk.31.ffn_norm.weight
     55: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.31.ffn_up_exps.weight
     56:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.31.ffn_up_shexp.weight
     57:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.32.attn_k_b.weight
     58:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.32.attn_kv_a_mqa.weight
     59:        512 |   512,     1,     1,     1 | F32     | blk.32.attn_kv_a_norm.weight
     60:       7168 |  7168,     1,     1,     1 | F32     | blk.32.attn_norm.weight
     61:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.32.attn_output.weight
     62:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.32.attn_q_a.weight
     63:       1536 |  1536,     1,     1,     1 | F32     | blk.32.attn_q_a_norm.weight
     64:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.32.attn_q_b.weight
     65:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.32.attn_v_b.weight
     66:        384 |   384,     1,     1,     1 | F32     | blk.32.exp_probs_b.bias
     67: 5637144576 |  2048,  7168,   384,     1 | IQ3_XXS | blk.32.ffn_down_exps.weight
     68:   14680064 |  2048,  7168,     1,     1 | IQ4_XS  | blk.32.ffn_down_shexp.weight
     69: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.32.ffn_gate_exps.weight
     70:    2752512 |  7168,   384,     1,     1 | F32     | blk.32.ffn_gate_inp.weight
     71:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.32.ffn_gate_shexp.weight
     72:       7168 |  7168,     1,     1,     1 | F32     | blk.32.ffn_norm.weight
     73: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.32.ffn_up_exps.weight
     74:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.32.ffn_up_shexp.weight
     75:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.33.attn_k_b.weight
     76:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.33.attn_kv_a_mqa.weight
     77:        512 |   512,     1,     1,     1 | F32     | blk.33.attn_kv_a_norm.weight
     78:       7168 |  7168,     1,     1,     1 | F32     | blk.33.attn_norm.weight
     79:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.33.attn_output.weight
     80:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.33.attn_q_a.weight
     81:       1536 |  1536,     1,     1,     1 | F32     | blk.33.attn_q_a_norm.weight
     82:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.33.attn_q_b.weight
     83:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.33.attn_v_b.weight
     84:        384 |   384,     1,     1,     1 | F32     | blk.33.exp_probs_b.bias
     85: 5637144576 |  2048,  7168,   384,     1 | IQ3_XXS | blk.33.ffn_down_exps.weight
     86:   14680064 |  2048,  7168,     1,     1 | IQ4_XS  | blk.33.ffn_down_shexp.weight
     87: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.33.ffn_gate_exps.weight
     88:    2752512 |  7168,   384,     1,     1 | F32     | blk.33.ffn_gate_inp.weight
     89:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.33.ffn_gate_shexp.weight
     90:       7168 |  7168,     1,     1,     1 | F32     | blk.33.ffn_norm.weight
     91: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.33.ffn_up_exps.weight
     92:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.33.ffn_up_shexp.weight
     93:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.34.attn_k_b.weight
     94:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.34.attn_kv_a_mqa.weight
     95:        512 |   512,     1,     1,     1 | F32     | blk.34.attn_kv_a_norm.weight
     96:       7168 |  7168,     1,     1,     1 | F32     | blk.34.attn_norm.weight
     97:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.34.attn_output.weight
     98:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.34.attn_q_a.weight
     99:       1536 |  1536,     1,     1,     1 | F32     | blk.34.attn_q_a_norm.weight
    100:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.34.attn_q_b.weight
    101:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.34.attn_v_b.weight
    102:        384 |   384,     1,     1,     1 | F32     | blk.34.exp_probs_b.bias
    103: 5637144576 |  2048,  7168,   384,     1 | IQ3_XXS | blk.34.ffn_down_exps.weight
    104:   14680064 |  2048,  7168,     1,     1 | IQ4_XS  | blk.34.ffn_down_shexp.weight
    105: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.34.ffn_gate_exps.weight
    106:    2752512 |  7168,   384,     1,     1 | F32     | blk.34.ffn_gate_inp.weight
    107:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.34.ffn_gate_shexp.weight
    108:       7168 |  7168,     1,     1,     1 | F32     | blk.34.ffn_norm.weight
    109: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.34.ffn_up_exps.weight
    110:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.34.ffn_up_shexp.weight
    111:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.35.attn_k_b.weight
    112:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.35.attn_kv_a_mqa.weight
    113:        512 |   512,     1,     1,     1 | F32     | blk.35.attn_kv_a_norm.weight
    114:       7168 |  7168,     1,     1,     1 | F32     | blk.35.attn_norm.weight
    115:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.35.attn_output.weight
    116:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.35.attn_q_a.weight
    117:       1536 |  1536,     1,     1,     1 | F32     | blk.35.attn_q_a_norm.weight
    118:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.35.attn_q_b.weight
    119:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.35.attn_v_b.weight
    120:        384 |   384,     1,     1,     1 | F32     | blk.35.exp_probs_b.bias
    121: 5637144576 |  2048,  7168,   384,     1 | IQ3_XXS | blk.35.ffn_down_exps.weight
    122:   14680064 |  2048,  7168,     1,     1 | IQ4_XS  | blk.35.ffn_down_shexp.weight
    123: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.35.ffn_gate_exps.weight
    124:    2752512 |  7168,   384,     1,     1 | F32     | blk.35.ffn_gate_inp.weight
    125:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.35.ffn_gate_shexp.weight
    126:       7168 |  7168,     1,     1,     1 | F32     | blk.35.ffn_norm.weight
    127: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.35.ffn_up_exps.weight
    128:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.35.ffn_up_shexp.weight
    129:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.36.attn_k_b.weight
    130:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.36.attn_kv_a_mqa.weight
    131:        512 |   512,     1,     1,     1 | F32     | blk.36.attn_kv_a_norm.weight
    132:       7168 |  7168,     1,     1,     1 | F32     | blk.36.attn_norm.weight
    133:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.36.attn_output.weight
    134:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.36.attn_q_a.weight
    135:       1536 |  1536,     1,     1,     1 | F32     | blk.36.attn_q_a_norm.weight
    136:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.36.attn_q_b.weight
    137:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.36.attn_v_b.weight
    138:        384 |   384,     1,     1,     1 | F32     | blk.36.exp_probs_b.bias
INFO:gguf-dump:* Loading: /mnt/data/models/unsloth/Kimi-K2-Instruct-GGUF/UD-IQ3_XXS/Kimi-K2-Instruct-UD-IQ3_XXS-00007-of-00009.gguf
* File is LITTLE endian, script is running on a LITTLE endian host.
* Dumping 6 key/value pair(s)
      1: UINT32     |        1 | GGUF.version = 3
      2: UINT64     |        1 | GGUF.tensor_count = 130
      3: UINT64     |        1 | GGUF.kv_count = 3
      4: UINT16     |        1 | split.no = 6
      5: INT32      |        1 | split.tensors.count = 1096
      6: UINT16     |        1 | split.count = 9
* Dumping 130 tensor(s)
      1: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.43.ffn_gate_exps.weight
      2:    2752512 |  7168,   384,     1,     1 | F32     | blk.43.ffn_gate_inp.weight
      3:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.43.ffn_gate_shexp.weight
      4:       7168 |  7168,     1,     1,     1 | F32     | blk.43.ffn_norm.weight
      5: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.43.ffn_up_exps.weight
      6:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.43.ffn_up_shexp.weight
      7:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.44.attn_k_b.weight
      8:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.44.attn_kv_a_mqa.weight
      9:        512 |   512,     1,     1,     1 | F32     | blk.44.attn_kv_a_norm.weight
     10:       7168 |  7168,     1,     1,     1 | F32     | blk.44.attn_norm.weight
     11:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.44.attn_output.weight
     12:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.44.attn_q_a.weight
     13:       1536 |  1536,     1,     1,     1 | F32     | blk.44.attn_q_a_norm.weight
     14:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.44.attn_q_b.weight
     15:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.44.attn_v_b.weight
     16:        384 |   384,     1,     1,     1 | F32     | blk.44.exp_probs_b.bias
     17: 5637144576 |  2048,  7168,   384,     1 | IQ3_XXS | blk.44.ffn_down_exps.weight
     18:   14680064 |  2048,  7168,     1,     1 | IQ4_XS  | blk.44.ffn_down_shexp.weight
     19: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.44.ffn_gate_exps.weight
     20:    2752512 |  7168,   384,     1,     1 | F32     | blk.44.ffn_gate_inp.weight
     21:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.44.ffn_gate_shexp.weight
     22:       7168 |  7168,     1,     1,     1 | F32     | blk.44.ffn_norm.weight
     23: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.44.ffn_up_exps.weight
     24:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.44.ffn_up_shexp.weight
     25:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.45.attn_k_b.weight
     26:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.45.attn_kv_a_mqa.weight
     27:        512 |   512,     1,     1,     1 | F32     | blk.45.attn_kv_a_norm.weight
     28:       7168 |  7168,     1,     1,     1 | F32     | blk.45.attn_norm.weight
     29:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.45.attn_output.weight
     30:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.45.attn_q_a.weight
     31:       1536 |  1536,     1,     1,     1 | F32     | blk.45.attn_q_a_norm.weight
     32:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.45.attn_q_b.weight
     33:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.45.attn_v_b.weight
     34:        384 |   384,     1,     1,     1 | F32     | blk.45.exp_probs_b.bias
     35: 5637144576 |  2048,  7168,   384,     1 | IQ3_XXS | blk.45.ffn_down_exps.weight
     36:   14680064 |  2048,  7168,     1,     1 | IQ4_XS  | blk.45.ffn_down_shexp.weight
     37: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.45.ffn_gate_exps.weight
     38:    2752512 |  7168,   384,     1,     1 | F32     | blk.45.ffn_gate_inp.weight
     39:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.45.ffn_gate_shexp.weight
     40:       7168 |  7168,     1,     1,     1 | F32     | blk.45.ffn_norm.weight
     41: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.45.ffn_up_exps.weight
     42:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.45.ffn_up_shexp.weight
     43:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.46.attn_k_b.weight
     44:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.46.attn_kv_a_mqa.weight
     45:        512 |   512,     1,     1,     1 | F32     | blk.46.attn_kv_a_norm.weight
     46:       7168 |  7168,     1,     1,     1 | F32     | blk.46.attn_norm.weight
     47:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.46.attn_output.weight
     48:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.46.attn_q_a.weight
     49:       1536 |  1536,     1,     1,     1 | F32     | blk.46.attn_q_a_norm.weight
     50:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.46.attn_q_b.weight
     51:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.46.attn_v_b.weight
     52:        384 |   384,     1,     1,     1 | F32     | blk.46.exp_probs_b.bias
     53: 5637144576 |  2048,  7168,   384,     1 | IQ3_XXS | blk.46.ffn_down_exps.weight
     54:   14680064 |  2048,  7168,     1,     1 | IQ4_XS  | blk.46.ffn_down_shexp.weight
     55: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.46.ffn_gate_exps.weight
     56:    2752512 |  7168,   384,     1,     1 | F32     | blk.46.ffn_gate_inp.weight
     57:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.46.ffn_gate_shexp.weight
     58:       7168 |  7168,     1,     1,     1 | F32     | blk.46.ffn_norm.weight
     59: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.46.ffn_up_exps.weight
     60:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.46.ffn_up_shexp.weight
     61:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.47.attn_k_b.weight
     62:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.47.attn_kv_a_mqa.weight
     63:        512 |   512,     1,     1,     1 | F32     | blk.47.attn_kv_a_norm.weight
     64:       7168 |  7168,     1,     1,     1 | F32     | blk.47.attn_norm.weight
     65:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.47.attn_output.weight
     66:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.47.attn_q_a.weight
     67:       1536 |  1536,     1,     1,     1 | F32     | blk.47.attn_q_a_norm.weight
     68:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.47.attn_q_b.weight
     69:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.47.attn_v_b.weight
     70:        384 |   384,     1,     1,     1 | F32     | blk.47.exp_probs_b.bias
     71: 5637144576 |  2048,  7168,   384,     1 | IQ3_XXS | blk.47.ffn_down_exps.weight
     72:   14680064 |  2048,  7168,     1,     1 | IQ4_XS  | blk.47.ffn_down_shexp.weight
     73: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.47.ffn_gate_exps.weight
     74:    2752512 |  7168,   384,     1,     1 | F32     | blk.47.ffn_gate_inp.weight
     75:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.47.ffn_gate_shexp.weight
     76:       7168 |  7168,     1,     1,     1 | F32     | blk.47.ffn_norm.weight
     77: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.47.ffn_up_exps.weight
     78:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.47.ffn_up_shexp.weight
     79:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.48.attn_k_b.weight
     80:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.48.attn_kv_a_mqa.weight
     81:        512 |   512,     1,     1,     1 | F32     | blk.48.attn_kv_a_norm.weight
     82:       7168 |  7168,     1,     1,     1 | F32     | blk.48.attn_norm.weight
     83:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.48.attn_output.weight
     84:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.48.attn_q_a.weight
     85:       1536 |  1536,     1,     1,     1 | F32     | blk.48.attn_q_a_norm.weight
     86:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.48.attn_q_b.weight
     87:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.48.attn_v_b.weight
     88:        384 |   384,     1,     1,     1 | F32     | blk.48.exp_probs_b.bias
     89: 5637144576 |  2048,  7168,   384,     1 | IQ3_XXS | blk.48.ffn_down_exps.weight
     90:   14680064 |  2048,  7168,     1,     1 | IQ4_XS  | blk.48.ffn_down_shexp.weight
     91: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.48.ffn_gate_exps.weight
     92:    2752512 |  7168,   384,     1,     1 | F32     | blk.48.ffn_gate_inp.weight
     93:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.48.ffn_gate_shexp.weight
     94:       7168 |  7168,     1,     1,     1 | F32     | blk.48.ffn_norm.weight
     95: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.48.ffn_up_exps.weight
     96:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.48.ffn_up_shexp.weight
     97:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.49.attn_k_b.weight
     98:    4128768 |  7168,   576,     1,     1 | Q6_K    | blk.49.attn_kv_a_mqa.weight
     99:        512 |   512,     1,     1,     1 | F32     | blk.49.attn_kv_a_norm.weight
    100:       7168 |  7168,     1,     1,     1 | F32     | blk.49.attn_norm.weight
    101:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.49.attn_output.weight
    102:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.49.attn_q_a.weight
    103:       1536 |  1536,     1,     1,     1 | F32     | blk.49.attn_q_a_norm.weight
    104:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.49.attn_q_b.weight
    105:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.49.attn_v_b.weight
    106:        384 |   384,     1,     1,     1 | F32     | blk.49.exp_probs_b.bias
    107: 5637144576 |  2048,  7168,   384,     1 | IQ3_XXS | blk.49.ffn_down_exps.weight
    108:   14680064 |  2048,  7168,     1,     1 | Q6_K    | blk.49.ffn_down_shexp.weight
    109: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.49.ffn_gate_exps.weight
    110:    2752512 |  7168,   384,     1,     1 | F32     | blk.49.ffn_gate_inp.weight
    111:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.49.ffn_gate_shexp.weight
    112:       7168 |  7168,     1,     1,     1 | F32     | blk.49.ffn_norm.weight
    113: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.49.ffn_up_exps.weight
    114:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.49.ffn_up_shexp.weight
    115:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.50.attn_k_b.weight
    116:    4128768 |  7168,   576,     1,     1 | Q6_K    | blk.50.attn_kv_a_mqa.weight
    117:        512 |   512,     1,     1,     1 | F32     | blk.50.attn_kv_a_norm.weight
    118:       7168 |  7168,     1,     1,     1 | F32     | blk.50.attn_norm.weight
    119:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.50.attn_output.weight
    120:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.50.attn_q_a.weight
    121:       1536 |  1536,     1,     1,     1 | F32     | blk.50.attn_q_a_norm.weight
    122:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.50.attn_q_b.weight
    123:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.50.attn_v_b.weight
    124:        384 |   384,     1,     1,     1 | F32     | blk.50.exp_probs_b.bias
    125: 5637144576 |  2048,  7168,   384,     1 | IQ3_XXS | blk.50.ffn_down_exps.weight
    126:   14680064 |  2048,  7168,     1,     1 | IQ4_XS  | blk.50.ffn_down_shexp.weight
    127: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.50.ffn_gate_exps.weight
    128:    2752512 |  7168,   384,     1,     1 | F32     | blk.50.ffn_gate_inp.weight
    129:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.50.ffn_gate_shexp.weight
    130:       7168 |  7168,     1,     1,     1 | F32     | blk.50.ffn_norm.weight
INFO:gguf-dump:* Loading: /mnt/data/models/unsloth/Kimi-K2-Instruct-GGUF/UD-IQ3_XXS/Kimi-K2-Instruct-UD-IQ3_XXS-00008-of-00009.gguf
* File is LITTLE endian, script is running on a LITTLE endian host.
* Dumping 6 key/value pair(s)
      1: UINT32     |        1 | GGUF.version = 3
      2: UINT64     |        1 | GGUF.tensor_count = 122
      3: UINT64     |        1 | GGUF.kv_count = 3
      4: UINT16     |        1 | split.no = 7
      5: INT32      |        1 | split.tensors.count = 1096
      6: UINT16     |        1 | split.count = 9
* Dumping 122 tensor(s)
      1: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.50.ffn_up_exps.weight
      2:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.50.ffn_up_shexp.weight
      3:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.51.attn_k_b.weight
      4:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.51.attn_kv_a_mqa.weight
      5:        512 |   512,     1,     1,     1 | F32     | blk.51.attn_kv_a_norm.weight
      6:       7168 |  7168,     1,     1,     1 | F32     | blk.51.attn_norm.weight
      7:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.51.attn_output.weight
      8:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.51.attn_q_a.weight
      9:       1536 |  1536,     1,     1,     1 | F32     | blk.51.attn_q_a_norm.weight
     10:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.51.attn_q_b.weight
     11:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.51.attn_v_b.weight
     12:        384 |   384,     1,     1,     1 | F32     | blk.51.exp_probs_b.bias
     13: 5637144576 |  2048,  7168,   384,     1 | IQ3_XXS | blk.51.ffn_down_exps.weight
     14:   14680064 |  2048,  7168,     1,     1 | IQ4_XS  | blk.51.ffn_down_shexp.weight
     15: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.51.ffn_gate_exps.weight
     16:    2752512 |  7168,   384,     1,     1 | F32     | blk.51.ffn_gate_inp.weight
     17:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.51.ffn_gate_shexp.weight
     18:       7168 |  7168,     1,     1,     1 | F32     | blk.51.ffn_norm.weight
     19: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.51.ffn_up_exps.weight
     20:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.51.ffn_up_shexp.weight
     21:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.52.attn_k_b.weight
     22:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.52.attn_kv_a_mqa.weight
     23:        512 |   512,     1,     1,     1 | F32     | blk.52.attn_kv_a_norm.weight
     24:       7168 |  7168,     1,     1,     1 | F32     | blk.52.attn_norm.weight
     25:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.52.attn_output.weight
     26:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.52.attn_q_a.weight
     27:       1536 |  1536,     1,     1,     1 | F32     | blk.52.attn_q_a_norm.weight
     28:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.52.attn_q_b.weight
     29:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.52.attn_v_b.weight
     30:        384 |   384,     1,     1,     1 | F32     | blk.52.exp_probs_b.bias
     31: 5637144576 |  2048,  7168,   384,     1 | IQ3_XXS | blk.52.ffn_down_exps.weight
     32:   14680064 |  2048,  7168,     1,     1 | IQ4_XS  | blk.52.ffn_down_shexp.weight
     33: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.52.ffn_gate_exps.weight
     34:    2752512 |  7168,   384,     1,     1 | F32     | blk.52.ffn_gate_inp.weight
     35:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.52.ffn_gate_shexp.weight
     36:       7168 |  7168,     1,     1,     1 | F32     | blk.52.ffn_norm.weight
     37: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.52.ffn_up_exps.weight
     38:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.52.ffn_up_shexp.weight
     39:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.53.attn_k_b.weight
     40:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.53.attn_kv_a_mqa.weight
     41:        512 |   512,     1,     1,     1 | F32     | blk.53.attn_kv_a_norm.weight
     42:       7168 |  7168,     1,     1,     1 | F32     | blk.53.attn_norm.weight
     43:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.53.attn_output.weight
     44:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.53.attn_q_a.weight
     45:       1536 |  1536,     1,     1,     1 | F32     | blk.53.attn_q_a_norm.weight
     46:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.53.attn_q_b.weight
     47:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.53.attn_v_b.weight
     48:        384 |   384,     1,     1,     1 | F32     | blk.53.exp_probs_b.bias
     49: 5637144576 |  2048,  7168,   384,     1 | IQ3_S   | blk.53.ffn_down_exps.weight
     50:   14680064 |  2048,  7168,     1,     1 | IQ4_XS  | blk.53.ffn_down_shexp.weight
     51: 5637144576 |  7168,  2048,   384,     1 | IQ3_S   | blk.53.ffn_gate_exps.weight
     52:    2752512 |  7168,   384,     1,     1 | F32     | blk.53.ffn_gate_inp.weight
     53:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.53.ffn_gate_shexp.weight
     54:       7168 |  7168,     1,     1,     1 | F32     | blk.53.ffn_norm.weight
     55: 5637144576 |  7168,  2048,   384,     1 | IQ3_S   | blk.53.ffn_up_exps.weight
     56:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.53.ffn_up_shexp.weight
     57:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.54.attn_k_b.weight
     58:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.54.attn_kv_a_mqa.weight
     59:        512 |   512,     1,     1,     1 | F32     | blk.54.attn_kv_a_norm.weight
     60:       7168 |  7168,     1,     1,     1 | F32     | blk.54.attn_norm.weight
     61:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.54.attn_output.weight
     62:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.54.attn_q_a.weight
     63:       1536 |  1536,     1,     1,     1 | F32     | blk.54.attn_q_a_norm.weight
     64:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.54.attn_q_b.weight
     65:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.54.attn_v_b.weight
     66:        384 |   384,     1,     1,     1 | F32     | blk.54.exp_probs_b.bias
     67: 5637144576 |  2048,  7168,   384,     1 | IQ3_S   | blk.54.ffn_down_exps.weight
     68:   14680064 |  2048,  7168,     1,     1 | IQ4_XS  | blk.54.ffn_down_shexp.weight
     69: 5637144576 |  7168,  2048,   384,     1 | IQ3_S   | blk.54.ffn_gate_exps.weight
     70:    2752512 |  7168,   384,     1,     1 | F32     | blk.54.ffn_gate_inp.weight
     71:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.54.ffn_gate_shexp.weight
     72:       7168 |  7168,     1,     1,     1 | F32     | blk.54.ffn_norm.weight
     73: 5637144576 |  7168,  2048,   384,     1 | IQ3_S   | blk.54.ffn_up_exps.weight
     74:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.54.ffn_up_shexp.weight
     75:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.55.attn_k_b.weight
     76:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.55.attn_kv_a_mqa.weight
     77:        512 |   512,     1,     1,     1 | F32     | blk.55.attn_kv_a_norm.weight
     78:       7168 |  7168,     1,     1,     1 | F32     | blk.55.attn_norm.weight
     79:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.55.attn_output.weight
     80:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.55.attn_q_a.weight
     81:       1536 |  1536,     1,     1,     1 | F32     | blk.55.attn_q_a_norm.weight
     82:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.55.attn_q_b.weight
     83:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.55.attn_v_b.weight
     84:        384 |   384,     1,     1,     1 | F32     | blk.55.exp_probs_b.bias
     85: 5637144576 |  2048,  7168,   384,     1 | IQ4_XS  | blk.55.ffn_down_exps.weight
     86:   14680064 |  2048,  7168,     1,     1 | Q5_K    | blk.55.ffn_down_shexp.weight
     87: 5637144576 |  7168,  2048,   384,     1 | IQ3_S   | blk.55.ffn_gate_exps.weight
     88:    2752512 |  7168,   384,     1,     1 | F32     | blk.55.ffn_gate_inp.weight
     89:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.55.ffn_gate_shexp.weight
     90:       7168 |  7168,     1,     1,     1 | F32     | blk.55.ffn_norm.weight
     91: 5637144576 |  7168,  2048,   384,     1 | IQ3_S   | blk.55.ffn_up_exps.weight
     92:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.55.ffn_up_shexp.weight
     93:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.56.attn_k_b.weight
     94:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.56.attn_kv_a_mqa.weight
     95:        512 |   512,     1,     1,     1 | F32     | blk.56.attn_kv_a_norm.weight
     96:       7168 |  7168,     1,     1,     1 | F32     | blk.56.attn_norm.weight
     97:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.56.attn_output.weight
     98:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.56.attn_q_a.weight
     99:       1536 |  1536,     1,     1,     1 | F32     | blk.56.attn_q_a_norm.weight
    100:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.56.attn_q_b.weight
    101:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.56.attn_v_b.weight
    102:        384 |   384,     1,     1,     1 | F32     | blk.56.exp_probs_b.bias
    103: 5637144576 |  2048,  7168,   384,     1 | IQ3_S   | blk.56.ffn_down_exps.weight
    104:   14680064 |  2048,  7168,     1,     1 | IQ4_XS  | blk.56.ffn_down_shexp.weight
    105: 5637144576 |  7168,  2048,   384,     1 | IQ3_S   | blk.56.ffn_gate_exps.weight
    106:    2752512 |  7168,   384,     1,     1 | F32     | blk.56.ffn_gate_inp.weight
    107:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.56.ffn_gate_shexp.weight
    108:       7168 |  7168,     1,     1,     1 | F32     | blk.56.ffn_norm.weight
    109: 5637144576 |  7168,  2048,   384,     1 | IQ3_S   | blk.56.ffn_up_exps.weight
    110:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.56.ffn_up_shexp.weight
    111:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.57.attn_k_b.weight
    112:    4128768 |  7168,   576,     1,     1 | Q6_K    | blk.57.attn_kv_a_mqa.weight
    113:        512 |   512,     1,     1,     1 | F32     | blk.57.attn_kv_a_norm.weight
    114:       7168 |  7168,     1,     1,     1 | F32     | blk.57.attn_norm.weight
    115:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.57.attn_output.weight
    116:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.57.attn_q_a.weight
    117:       1536 |  1536,     1,     1,     1 | F32     | blk.57.attn_q_a_norm.weight
    118:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.57.attn_q_b.weight
    119:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.57.attn_v_b.weight
    120:        384 |   384,     1,     1,     1 | F32     | blk.57.exp_probs_b.bias
    121: 5637144576 |  2048,  7168,   384,     1 | IQ4_XS  | blk.57.ffn_down_exps.weight
    122:   14680064 |  2048,  7168,     1,     1 | Q6_K    | blk.57.ffn_down_shexp.weight
INFO:gguf-dump:* Loading: /mnt/data/models/unsloth/Kimi-K2-Instruct-GGUF/UD-IQ3_XXS/Kimi-K2-Instruct-UD-IQ3_XXS-00009-of-00009.gguf
* File is LITTLE endian, script is running on a LITTLE endian host.
* Dumping 6 key/value pair(s)
      1: UINT32     |        1 | GGUF.version = 3
      2: UINT64     |        1 | GGUF.tensor_count = 60
      3: UINT64     |        1 | GGUF.kv_count = 3
      4: UINT16     |        1 | split.no = 8
      5: INT32      |        1 | split.tensors.count = 1096
      6: UINT16     |        1 | split.count = 9
* Dumping 60 tensor(s)
      1: 5637144576 |  7168,  2048,   384,     1 | IQ3_S   | blk.57.ffn_gate_exps.weight
      2:    2752512 |  7168,   384,     1,     1 | F32     | blk.57.ffn_gate_inp.weight
      3:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.57.ffn_gate_shexp.weight
      4:       7168 |  7168,     1,     1,     1 | F32     | blk.57.ffn_norm.weight
      5: 5637144576 |  7168,  2048,   384,     1 | IQ3_S   | blk.57.ffn_up_exps.weight
      6:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.57.ffn_up_shexp.weight
      7:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.58.attn_k_b.weight
      8:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.58.attn_kv_a_mqa.weight
      9:        512 |   512,     1,     1,     1 | F32     | blk.58.attn_kv_a_norm.weight
     10:       7168 |  7168,     1,     1,     1 | F32     | blk.58.attn_norm.weight
     11:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.58.attn_output.weight
     12:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.58.attn_q_a.weight
     13:       1536 |  1536,     1,     1,     1 | F32     | blk.58.attn_q_a_norm.weight
     14:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.58.attn_q_b.weight
     15:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.58.attn_v_b.weight
     16:        384 |   384,     1,     1,     1 | F32     | blk.58.exp_probs_b.bias
     17: 5637144576 |  2048,  7168,   384,     1 | IQ3_S   | blk.58.ffn_down_exps.weight
     18:   14680064 |  2048,  7168,     1,     1 | IQ4_XS  | blk.58.ffn_down_shexp.weight
     19: 5637144576 |  7168,  2048,   384,     1 | IQ3_S   | blk.58.ffn_gate_exps.weight
     20:    2752512 |  7168,   384,     1,     1 | F32     | blk.58.ffn_gate_inp.weight
     21:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.58.ffn_gate_shexp.weight
     22:       7168 |  7168,     1,     1,     1 | F32     | blk.58.ffn_norm.weight
     23: 5637144576 |  7168,  2048,   384,     1 | IQ3_S   | blk.58.ffn_up_exps.weight
     24:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.58.ffn_up_shexp.weight
     25:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.59.attn_k_b.weight
     26:    4128768 |  7168,   576,     1,     1 | Q6_K    | blk.59.attn_kv_a_mqa.weight
     27:        512 |   512,     1,     1,     1 | F32     | blk.59.attn_kv_a_norm.weight
     28:       7168 |  7168,     1,     1,     1 | F32     | blk.59.attn_norm.weight
     29:   58720256 |  8192,  7168,     1,     1 | Q5_K    | blk.59.attn_output.weight
     30:   11010048 |  7168,  1536,     1,     1 | Q5_K    | blk.59.attn_q_a.weight
     31:       1536 |  1536,     1,     1,     1 | F32     | blk.59.attn_q_a_norm.weight
     32:   18874368 |  1536, 12288,     1,     1 | Q5_K    | blk.59.attn_q_b.weight
     33:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.59.attn_v_b.weight
     34:        384 |   384,     1,     1,     1 | F32     | blk.59.exp_probs_b.bias
     35: 5637144576 |  2048,  7168,   384,     1 | IQ4_XS  | blk.59.ffn_down_exps.weight
     36:   14680064 |  2048,  7168,     1,     1 | Q6_K    | blk.59.ffn_down_shexp.weight
     37: 5637144576 |  7168,  2048,   384,     1 | IQ3_S   | blk.59.ffn_gate_exps.weight
     38:    2752512 |  7168,   384,     1,     1 | F32     | blk.59.ffn_gate_inp.weight
     39:   14680064 |  7168,  2048,     1,     1 | Q4_K    | blk.59.ffn_gate_shexp.weight
     40:       7168 |  7168,     1,     1,     1 | F32     | blk.59.ffn_norm.weight
     41: 5637144576 |  7168,  2048,   384,     1 | IQ3_S   | blk.59.ffn_up_exps.weight
     42:   14680064 |  7168,  2048,     1,     1 | Q4_K    | blk.59.ffn_up_shexp.weight
     43:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.60.attn_k_b.weight
     44:    4128768 |  7168,   576,     1,     1 | Q5_K    | blk.60.attn_kv_a_mqa.weight
     45:        512 |   512,     1,     1,     1 | F32     | blk.60.attn_kv_a_norm.weight
     46:       7168 |  7168,     1,     1,     1 | F32     | blk.60.attn_norm.weight
     47:   58720256 |  8192,  7168,     1,     1 | Q5_K    | blk.60.attn_output.weight
     48:   11010048 |  7168,  1536,     1,     1 | Q5_K    | blk.60.attn_q_a.weight
     49:       1536 |  1536,     1,     1,     1 | F32     | blk.60.attn_q_a_norm.weight
     50:   18874368 |  1536, 12288,     1,     1 | Q5_K    | blk.60.attn_q_b.weight
     51:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.60.attn_v_b.weight
     52:        384 |   384,     1,     1,     1 | F32     | blk.60.exp_probs_b.bias
     53: 5637144576 |  2048,  7168,   384,     1 | IQ4_XS  | blk.60.ffn_down_exps.weight
     54:   14680064 |  2048,  7168,     1,     1 | Q6_K    | blk.60.ffn_down_shexp.weight
     55: 5637144576 |  7168,  2048,   384,     1 | IQ3_S   | blk.60.ffn_gate_exps.weight
     56:    2752512 |  7168,   384,     1,     1 | F32     | blk.60.ffn_gate_inp.weight
     57:   14680064 |  7168,  2048,     1,     1 | Q4_K    | blk.60.ffn_gate_shexp.weight
     58:       7168 |  7168,     1,     1,     1 | F32     | blk.60.ffn_norm.weight
     59: 5637144576 |  7168,  2048,   384,     1 | IQ3_S   | blk.60.ffn_up_exps.weight
     60:   14680064 |  7168,  2048,     1,     1 | Q4_K    | blk.60.ffn_up_shexp.weight
```

</details>

---

👤 **magikRUKKOLA** commented on **2025-07-19** at **01:30:36**

@ubergarm 

> Ok, I will retest the UD-IQ3_XXS.

Well, yeah, I retested the UD-IQ3-XXS from unsloth with the default settings and the results are below.

Final estimate: PPL = 3.1467 +/- 0.01596

Its possible I messed up the initial calculations due to non-default perplexity config.  So my initial value was 3.1382 which seems to be incorrect.  Thanks for letting me know!

```
export MALLOC_CONF="background_thread:true,percpu_arena:phycpu,metadata_thp:auto,dirty_decay_ms:10000,muzzy_decay_ms:60000"
export LD_PRELOAD=/usr/local/lib/libjemalloc.so

CUDA_VISIBLE_DEVICES="0,1" \
/opt/ik_llama.cpp/ik_llama.cpp/build/bin/llama-perplexity \
    -f /opt/ik_llama.cpp/wiki.test.raw \
    --model /opt/unsloth/Kimi-K2-Instruct-GGUF/UD-IQ3_XXS/Kimi-K2-Instruct-UD-IQ3_XXS-00001-of-00009.gguf \
    --alias unsloth/Kimi-K2-Instruct-UD-IQ3_XXS \
    --ctx-size $((512)) \
    -ub $((512)) \
    -ctk q8_0 \
    -mla 3 -fa \
    -amb 512 \
    -fmoe \
    --n-gpu-layers 99 \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads $(grep ^cpu\\scores /proc/cpuinfo | uniq | awk '{print $4}' | xargs -I{} echo "{}-0" | bc) \
    --host 0.0.0.0 \
    --port 8080 \
    --lookup-cache-dynamic /mnt/data/ik_llama.kv.dump
```

```

ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 1: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
main: build = 3808 (38012f72)
main: built with cc (Debian 14.2.0-19) 14.2.0 for x86_64-linux-gnu
main: seed  = 1752881437
llama_model_loader: additional 8 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 62 key-value pairs and 1096 tensors from /opt/unsloth/Kimi-K2-Instruct-GGUF/UD-IQ3_XXS/Kimi-K2-Ins
truct-UD-IQ3_XXS-00001-of-00009.gguf (version GGUF V3 (latest))
...
*** Your prompt processing speed will be crippled ***

Consider making your own ik_llama.cpp compatible model or
ask the model provider to make one for you,
==========================================================================
llm_load_vocab: special tokens cache size = 256
llm_load_vocab: token to piece cache size = 1.0607 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = deepseek2
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 163840
llm_load_print_meta: n_merges         = 163328
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 131072
llm_load_print_meta: n_embd           = 7168
llm_load_print_meta: n_layer          = 61
llm_load_print_meta: n_head           = 64
llm_load_print_meta: n_head_kv        = 64
llm_load_print_meta: n_rot            = 64
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 192
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: n_embd_k_gqa     = 12288
llm_load_print_meta: n_embd_v_gqa     = 8192
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 18432
llm_load_print_meta: n_expert         = 384
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = yarn
llm_load_print_meta: freq_base_train  = 50000.0
llm_load_print_meta: freq_scale_train = 0.03125
llm_load_print_meta: n_ctx_orig_yarn  = 4096
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = 671B
llm_load_print_meta: model ftype      = IQ3_XXS - 3.0625 bpw
llm_load_print_meta: model params     = 1.026 T
llm_load_print_meta: model size       = 388.003 GiB (3.247 BPW)
llm_load_print_meta: repeating layers = 386.491 GiB (3.242 BPW, 1024.059 B parameters)
llm_load_print_meta: general.name     = Kimi-K2-Instruct
llm_load_print_meta: BOS token        = 163584 '[BOS]'
llm_load_print_meta: EOS token        = 163586 '<|im_end|>'
llm_load_print_meta: PAD token        = 163839 '[PAD]'
llm_load_print_meta: LF token         = 128 'Ä'
llm_load_print_meta: EOT token        = 163586 '<|im_end|>'
llm_load_print_meta: max token length = 512
llm_load_print_meta: n_layer_dense_lead   = 1
llm_load_print_meta: n_lora_q             = 1536
llm_load_print_meta: n_lora_kv            = 512
llm_load_print_meta: n_ff_exp             = 2048
llm_load_print_meta: n_expert_shared      = 1
llm_load_print_meta: expert_weights_scale = 2.8
llm_load_print_meta: expert_weights_norm  = 1
llm_load_print_meta: expert_gating_func   = sigmoid
llm_load_print_meta: rope_yarn_log_mul    = 0.1000
llm_load_tensors: ggml ctx size =    1.35 MiB
...
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size = 44823.65 MiB
llm_load_tensors:        CPU buffer size = 47456.06 MiB
llm_load_tensors:        CPU buffer size = 45899.98 MiB
llm_load_tensors:        CPU buffer size = 46406.32 MiB
llm_load_tensors:        CPU buffer size = 45897.95 MiB
llm_load_tensors:        CPU buffer size = 45899.09 MiB
llm_load_tensors:        CPU buffer size = 45903.13 MiB
llm_load_tensors:        CPU buffer size = 46126.73 MiB
llm_load_tensors:        CPU buffer size = 26822.94 MiB
llm_load_tensors:        CPU buffer size =   630.00 MiB
llm_load_tensors:      CUDA0 buffer size =  2998.56 MiB
llm_load_tensors:      CUDA1 buffer size =  3632.72 MiB
....................................................................................................
============ llm_prepare_mla: need to compute 61 wkv_b tensors
...
llama_new_context_with_model: n_ctx      = 2048
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 3
llama_new_context_with_model: attn_max_b = 512
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 50000.0
llama_new_context_with_model: freq_scale = 0.03125
llama_kv_cache_init:      CUDA0 KV buffer size =    37.07 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =    35.87 MiB
llama_new_context_with_model: KV self size  =   72.91 MiB, c^KV (q8_0):   72.91 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     2.50 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      CUDA0 compute buffer size =   263.00 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =   334.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   162.01 MiB
llama_new_context_with_model: graph nodes  = 3586
llama_new_context_with_model: graph splits = 123

system_info: n_threads = 64 / 128 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 |
perplexity: tokenizing the input ..
perplexity: tokenization took 910.573 ms
perplexity: calculating perplexity over 568 chunks, n_ctx=512, batch_size=2048, n_seq=4
perplexity: 47.59 seconds per pass - ETA 1 hours 52.62 minutes
[1]2.4402,[2]3.2625,[3]2.7728,[4]2.7844,[5]2.4434,[6]2.2209,[7]2.2743,[8]2.1760,[9]2.1280,[10]2.1779,[11]2.1036,[12]2.0877,[13]2.0981,[14]2.1244,[15]2.2348,[16]2.3364,[17]2.4509,[18]2.6415,[19]2.6341,[20]2.6690,[21]2.7114,[22]2.6991,[23]2.6637,[24]2.6005,[25]2.5621,[26]2.5216,[27]2.4967,[28]2.5094,[29]2.4895,[30]2.5136,[31]2.5486,[32]2.5543,[33]2.5785,[34]2.6011,[35]2.6537,[36]2.6763,[37]2.7021,[38]2.7616,[39]2.7924,[40]2.8271,[41]2.8815,[42]2.9177,[43]2.9329,[44]2.9529,[45]3.0282,[46]3.0813,[47]3.0751,[48]3.0435,[49]3.0158,[50]3.0184,[51]3.0462,[52]3.0724,[53]3.1099,[54]3.1123,[55]3.1226,[56]3.1415,[57]3.1249,[58]3.1263,[59]3.1443,[60]3.1822,[61]3.2109,[62]3.2374,[63]3.2661,[64]3.2831,[65]3.2993,[66]3.2916,[67]3.2796,[68]3.2570,[69]3.2600,[70]3.2654,[71]3.2423,[72]3.2270,[73]3.2153,[74]3.2256,[75]3.2352,[76]3.2106,[77]3.1840,[78]3.1672,[79]3.1658,[80]3.1370,[81]3.1144,[82]3.0953,[83]3.1231,[84]3.1159,[85]3.0917,[86]3.0745,[87]3.0550,[88]3.0508,[89]3.0369,[90]3.0450,[91]3.0328,[92]3.0210,[93]3.0096,[94]2.9879,[95]2.9714,[96]2.9472,[97]2.9493,[98]2.9482,[99]2.9355,[100]2.9206,[101]2.9173,[102]2.9167,[103]2.9417,[104]2.9715,[105]3.0042,[106]3.0105,[107]3.0387,[108]3.0660,[109]3.0839,[110]3.1193,[111]3.1550,[112]3.1754,[113]3.1715,[114]3.1695,[115]3.1597,[116]3.1450,[117]3.1330,[118]3.1342,[119]3.1297,[120]3.1278,[121]3.1166,[122]3.1045,[123]3.0960,[124]3.0911,[125]3.0776,[126]3.0671,[127]3.0565,[128]3.0534,[129]3.0539,[130]3.0560,[131]3.0569,[132]3.0592,[133]3.0522,[134]3.0566,[135]3.0717,[136]3.0667,[137]3.0601,[138]3.0667,[139]3.0655,[140]3.0809,[141]3.0765,[142]3.0728,[143]3.0742,[144]3.0702,[145]3.0697,[146]3.0658,[147]3.0530,[148]3.0516,[149]3.0474,[150]3.0475,[151]3.0482,[152]3.0406,[153]3.0408,[154]3.0366,[155]3.0307,[156]3.0317,[157]3.0326,[158]3.0281,[159]3.0316,[160]3.0278,[161]3.0222,[162]3.0263,[163]3.0288,[164]3.0447,[165]3.0468,[166]3.0622,[167]3.0728,[168]3.0872,[169]3.1035,[170]3.1238,[171]3.1427,[172]3.1657,[173]3.1806,[174]3.1749,[175]3.1695,[176]3.1583,[177]3.1556,[178]3.1547,[179]3.1467,[180]3.1341,[181]3.1313,[182]3.1324,[183]3.1488,[184]3.1637,[185]3.1785,[186]3.1908,[187]3.2003,[188]3.2178,[189]3.2334,[190]3.2471,[191]3.2558,[192]3.2574,[193]3.2642,[194]3.2665,[195]3.2658,[196]3.2801,[197]3.2852,[198]3.2983,[199]3.3091,[200]3.3121,[201]3.3188,[202]3.3144,[203]3.3297,[204]3.3277,[205]3.3334,[206]3.3333,[207]3.3358,[208]3.3374,[209]3.3439,[210]3.3487,[211]3.3536,[212]3.3544,[213]3.3516,[214]3.3534,[215]3.3533,[216]3.3571,[217]3.3662,[218]3.3620,[219]3.3605,[220]3.3561,[221]3.3567,[222]3.3561,[223]3.3577,[224]3.3584,[225]3.3582,[226]3.3624,[227]3.3671,[228]3.3530,[229]3.3536,[230]3.3499,[231]3.3473,[232]3.3542,[233]3.3636,[234]3.3694,[235]3.3611,[236]3.3585,[237]3.3571,[238]3.3612,[239]3.3656,[240]3.3684,[241]3.3762,[242]3.3856,[243]3.3938,[244]3.4020,[245]3.4132,[246]3.4218,[247]3.4244,[248]3.4328,[249]3.4376,[250]3.4371,[251]3.4275,[252]3.4152,[253]3.4055,[254]3.4000,[255]3.3961,[256]3.3938,[257]3.3953,[258]3.3939,[259]3.3920,[260]3.3889,[261]3.3870,[262]3.3828,[263]3.3787,[264]3.3739,[265]3.3687,[266]3.3668,[267]3.3673,[268]3.3628,[269]3.3588,[270]3.3531,[271]3.3484,[272]3.3444,[273]3.3397,[274]3.3385,[275]3.3302,[276]3.3266,[277]3.3214,[278]3.3200,[279]3.3140,[280]3.3131,[281]3.3192,[282]3.3233,[283]3.3299,[284]3.3378,[285]3.3447,[286]3.3497,[287]3.3613,[288]3.3691,[289]3.3749,[290]3.3751,[291]3.3770,[292]3.3782,[293]3.3809,[294]3.3716,[295]3.3720,[296]3.3778,[297]3.3796,[298]3.3836,[299]3.3875,[300]3.3897,[301]3.3946,[302]3.3992,[303]3.3987,[304]3.3954,[305]3.3971,[306]3.3961,[307]3.3972,[308]3.4018,[309]3.4028,[310]3.4023,[311]3.4029,[312]3.3962,[313]3.3942,[314]3.3987,[315]3.4001,[316]3.3971,[317]3.3952,[318]3.3904,[319]3.3853,[320]3.3804,[321]3.3724,[322]3.3648,[323]3.3578,[324]3.3515,[325]3.3468,[326]3.3395,[327]3.3368,[328]3.3318,[329]3.3303,[330]3.3232,[331]3.3273,[332]3.3214,[333]3.3223,[334]3.3229,[335]3.3258,[336]3.3300,[337]3.3292,[338]3.3291,[339]3.3289,[340]3.3285,[341]3.3281,[342]3.3337,[343]3.3345,[344]3.3344,[345]3.3426,[346]3.3482,[347]3.3523,[348]3.3470,[349]3.3436,[350]3.3410,[351]3.3392,[352]3.3327,[353]3.3259,[354]3.3200,[355]3.3196,[356]3.3205,[357]3.3181,[358]3.3146,[359]3.3113,[360]3.3124,[361]3.3096,[362]3.3042,[363]3.3005,[364]3.2967,[365]3.2931,[366]3.2890,[367]3.2853,[368]3.2813,[369]3.2815,[370]3.2824,[371]3.2766,[372]3.2741,[373]3.2691,[374]3.2642,[375]3.2619,[376]3.2577,[377]3.2522,[378]3.2504,[379]3.2470,[380]3.2441,[381]3.2424,[382]3.2434,[383]3.2388,[384]3.2403,[385]3.2426,[386]3.2464,[387]3.2506,[388]3.2557,[389]3.2585,[390]3.2627,[391]3.2675,[392]3.2696,[393]3.2623,[394]3.2593,[395]3.2538,[396]3.2513,[397]3.2479,[398]3.2442,[399]3.2375,[400]3.2418,[401]3.2345,[402]3.2295,[403]3.2237,[404]3.2245,[405]3.2212,[406]3.2144,[407]3.2069,[408]3.2032,[409]3.1968,[410]3.1908,[411]3.1883,[412]3.1839,[413]3.1779,[414]3.1731,[415]3.1723,[416]3.1699,[417]3.1705,[418]3.1666,[419]3.1621,[420]3.1571,[421]3.1521,[422]3.1509,[423]3.1475,[424]3.1481,[425]3.1456,[426]3.1441,[427]3.1388,[428]3.1358,[429]3.1338,[430]3.1311,[431]3.1261,[432]3.1227,[433]3.1179,[434]3.1150,[435]3.1127,[436]3.1078,[437]3.1024,[438]3.0975,[439]3.0925,[440]3.0897,[441]3.0847,[442]3.0831,[443]3.0796,[444]3.0791,[445]3.0819,[446]3.0867,[447]3.0929,[448]3.0911,[449]3.0885,[450]3.0881,[451]3.0917,[452]3.0956,[453]3.0971,[454]3.1002,[455]3.1025,[456]3.1081,[457]3.1097,[458]3.1119,[459]3.1160,[460]3.1170,[461]3.1208,[462]3.1227,[463]3.1298,[464]3.1350,[465]3.1377,[466]3.1383,[467]3.1383,[468]3.1393,[469]3.1447,[470]3.1429,[471]3.1419,[472]3.1472,[473]3.1495,[474]3.1500,[475]3.1528,[476]3.1543,[477]3.1548,[478]3.1566,[479]3.1566,[480]3.1580,[481]3.1583,[482]3.1575,[483]3.1582,[484]3.1586,[485]3.1580,[486]3.1605,[487]3.1580,[488]3.1596,[489]3.1575,[490]3.1664,[491]3.1702,[492]3.1747,[493]3.1759,[494]3.1783,[495]3.1831,[496]3.1852,[497]3.1878,[498]3.1921,[499]3.1924,[500]3.1923,[501]3.1927,[502]3.1942,[503]3.1960,[504]3.1959,[505]3.1996,[506]3.2027,[507]3.2090,[508]3.2094,[509]3.2104,[510]3.2116,[511]3.2167,[512]3.2221,[513]3.2267,[514]3.2280,[515]3.2246,[516]3.2225,[517]3.2219,[518]3.2189,[519]3.2153,[520]3.2144,[521]3.2123,[522]3.2095,[523]3.2085,[524]3.2076,[525]3.2039,[526]3.2043,[527]3.2034,[528]3.2039,[529]3.2053,[530]3.2027,[531]3.2024,[532]3.2009,[533]3.1985,[534]3.1978,[535]3.1962,[536]3.1949,[537]3.1938,[538]3.1885,[539]3.1851,[540]3.1850,[541]3.1827,[542]3.1850,[543]3.1849,[544]3.1856,[545]3.1850,[546]3.1850,[547]3.1847,[548]3.1883,[549]3.1886,[550]3.1881,[551]3.1902,[552]3.1862,[553]3.1828,[554]3.1773,[555]3.1734,[556]3.1718,[557]3.1671,[558]3.1619,[559]3.1584,[560]3.1579,[561]3.1548,[562]3.1528,[563]3.1507,[564]3.1500,[565]3.1484,[566]3.1510,[567]3.1492,[568]3.1467,
Final estimate: PPL = 3.1467 +/- 0.01596

llama_print_timings:        load time =  126687.38 ms
llama_print_timings:      sample time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings: prompt eval time = 6458901.47 ms / 290816 tokens (   22.21 ms per token,    45.03 tokens per second)
llama_print_timings:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings:       total time = 6468857.58 ms / 290817 tokens
```

---

👤 **ThomasBaruzier** commented on **2025-07-19** at **15:59:44**

Thanks Iwan and Ubergram for the amazing work! You two motivated me to try Kimi on my "mere" 128GB + 3x3090 rig.  

@ubergarm, I tried using your imatrix and script to test this new quant, and I have a few questions if you don’t mind.  

Here’s the script I use - basically your recipe but with `blk\..*\.ffn_(gate|up)_exps\.weight` at `iq1_s_r4`.  

<details>
<summary>Script</summary>

```sh
#!/bin/bash

set -e

imatrix='/home/user/storage/gguf/Kimi-K2-Instruct/Kimi-K2-Instruct-Q8_0.imatrix'
input='/home/user/storage/gguf/Kimi-K2-Instruct/Kimi-K2-Instruct-Q8_0.gguf'
output='/home/user/nvme/gguf/Kimi-K2-Instruct/Kimi-K2-Instruct-IQ1_S.gguf'

custom="
## Attention [0-60] (GPU)
# Only ik's fork uses this, keep it q8_0 as its only for PP with -mla 3
blk\..*\.attn_kv_b\.weight=q8_0

# ideally k_b and v_b are smaller than q8_0 as they are is used for TG with -mla 3 (and ik's imatrix supports it)
# blk.*.attn_k_b.weight is not divisible by 256 so only supports qN_0 or iq4_nl
blk\..*\.attn_k_b\.weight=iq4_nl

# Balance of attn tensors
blk\..*\.attn_.*=iq4_kt

## First Single Dense Layer [0] (GPU)
blk\..*\.ffn_down\.weight=iq4_kt
blk\..*\.ffn_(gate|up)\.weight=iq3_kt

## Shared Expert [1-60] (GPU)
blk\..*\.ffn_down_shexp\.weight=iq4_kt
blk\..*\.ffn_(gate|up)_shexp\.weight=iq3_kt

## Routed Experts [1-60] (CPU)
blk\..*\.ffn_down_exps\.weight=iq1_kt
blk\..*\.ffn_(gate|up)_exps\.weight=iq1_s_r4

## Token embedding and output tensors (GPU)
token_embd\.weight=iq4_kt
output\.weight=iq5_ks
"

if [ -f "$output" ]; then
  read -p "Quant already exists: $output. Continue? (N/y): " x
  [ "$x" != y ] && exit 0
  rm -f "$output"
fi

get_screen() {
  if [ -z "$STY" ]; then
    log_path=$(readlink -f "$0")
    log_path="${log_path%/*}/logs/${log_path##*/}"
    log_path="${log_path%.*}.log"
    screen -ls | grep -q "$screen_name" && \
    echo 'Process already running.' && exit 1
    echo "Launching the $screen_name screen..."
    mkdir -p "${log_path%/*}"
    echo '------------------------------------' >> "$log_path"
    screen -mS "$screen_name" -L -Logfile "$log_path" bash "$0" "$@"
    exit 0
  fi
}

screen_name='ik-kimi'
get_screen

custom=$(
  echo "$custom" | grep -v '^#' | \
  sed -Ez 's:\n+:,:g;s:,$::;s:^,::'
)

/home/user/files/ai/llama/ik_llama.cpp/llama-quantize \
  --allow-requantize \
  --custom-q "$custom" \
  --imatrix "$imatrix" \
  "$input" "$output" \
  IQ1_KT 32
```
</details>

1) Which tensors are unnecessary for MLA 3? It seems there are a few suspicious warnings:  
   - `====== llama_model_quantize_internal: did not find weights for token_embd.weight`  
   - `converting to iq4_kt .. cluster_points: Oops. Cluster 4 has no points: 0 1 0 0`  
   - `cluster_points: 1 out of 625 clusters dir not have any points`  
   - `====== llama_model_quantize_internal: did not find weights for blk.0.attn_k_b.weight`  

It seems you already commented about `Oops. Cluster X has no points` in this repo, and it’s apparently harmless. However, could `token_embd.weight` be missing because I used Q8_0 as input? Note that the Q8_0 input was made from `convert_hf_to_gguf.py`:  
`python convert_hf_to_gguf.py --outfile /home/user/storage/gguf/Kimi-K2-Instruct/Kimi-K2-Instruct-Q8_0.gguf /home/user/storage/llm/Kimi-K2-Instruct-BF16/ --outtype q8_0 --model-name Kimi-K2-Instruct --split-max-size 9999G`

<details>
<summary>Full logs (so far)</summary>

```
Adding custom rule blk\..*\.attn_kv_b\.weight -> q8_0
Adding custom rule blk\..*\.attn_k_b\.weight -> iq4_nl
Adding custom rule blk\..*\.attn_.* -> iq4_kt
Adding custom rule blk\..*\.ffn_down\.weight -> iq4_kt
Adding custom rule blk\..*\.ffn_(gate|up)\.weight -> iq3_kt
Adding custom rule blk\..*\.ffn_down_shexp\.weight -> iq4_kt
Adding custom rule blk\..*\.ffn_(gate|up)_shexp\.weight -> iq3_kt
Adding custom rule blk\..*\.ffn_down_exps\.weight -> iq1_kt
Adding custom rule blk\..*\.ffn_(gate|up)_exps\.weight -> iq1_s_r4
Adding custom rule token_embd\.weight -> iq4_kt
Adding custom rule output\.weight -> iq5_ks
load_imatrix: imatrix dataset='ubergarm-imatrix-calibration-corpus-v02.txt'
load_imatrix: loaded 729 importance matrix entries from /home/user/storage/gguf/Kimi-K2-Instruct/Kimi-K2-Instruct-Q8_0.imatrix computed on 826 chunks
prepare_imatrix: have 729 importance matrix entries
main: build = 3818 (77eaa532)
main: built with cc (GCC) 15.1.1 20250425 for x86_64-pc-linux-gnu
main: quantizing '/home/user/storage/gguf/Kimi-K2-Instruct/Kimi-K2-Instruct-Q8_0.gguf' to '/home/user/nvme/gguf/Kimi-K2-Instruct/Kimi-K2-Instruct-IQ1_S.gguf' as IQ1_KT using 32 threads
llama_model_loader: loaded meta data with 50 key-value pairs and 1157 tensors from /home/user/storage/gguf/Kimi-K2-Instruct/Kimi-K2-Instruct-Q8_0.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Kimi-K2-Instruct
llama_model_loader: - kv   3:                           general.finetune str              = Instruct
llama_model_loader: - kv   4:                           general.basename str              = Kimi-K2
llama_model_loader: - kv   5:                         general.size_label str              = 384x15B
llama_model_loader: - kv   6:                            general.license str              = other
llama_model_loader: - kv   7:                       general.license.name str              = modified-mit
llama_model_loader: - kv   8:                   general.base_model.count u32              = 1
llama_model_loader: - kv   9:                  general.base_model.0.name str              = Kimi K2 Instruct
llama_model_loader: - kv  10:          general.base_model.0.organization str              = Moonshotai
llama_model_loader: - kv  11:              general.base_model.0.repo_url str              = https://huggingface.co/moonshotai/Kim...
llama_model_loader: - kv  12:                               general.tags arr[str,1]       = ["unsloth"]
llama_model_loader: - kv  13:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv  14:                   deepseek2.context_length u32              = 131072
llama_model_loader: - kv  15:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv  16:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  17:             deepseek2.attention.head_count u32              = 64
llama_model_loader: - kv  18:          deepseek2.attention.head_count_kv u32              = 64
llama_model_loader: - kv  19:                   deepseek2.rope.freq_base f32              = 50000.000000
llama_model_loader: - kv  20: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  21:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  22:                          general.file_type u32              = 7
llama_model_loader: - kv  23:        deepseek2.leading_dense_block_count u32              = 1
llama_model_loader: - kv  24:                       deepseek2.vocab_size u32              = 163840
llama_model_loader: - kv  25:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  26:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  27:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  28:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  29:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  30:                     deepseek2.expert_count u32              = 384
llama_model_loader: - kv  31:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  32:             deepseek2.expert_weights_scale f32              = 2.827000
llama_model_loader: - kv  33:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  34:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  35:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  36:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  37:              deepseek2.rope.scaling.factor f32              = 32.000000
llama_model_loader: - kv  38: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  39: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  40:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  41:                         tokenizer.ggml.pre str              = kimi-k2
llama_model_loader: - kv  42:                      tokenizer.ggml.tokens arr[str,163840]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  43:                  tokenizer.ggml.token_type arr[i32,163840]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  44:                      tokenizer.ggml.merges arr[str,163328]  = ["Ġ Ġ", "ĠĠ ĠĠ", "Ġ t", "i n",...
llama_model_loader: - kv  45:                tokenizer.ggml.bos_token_id u32              = 163584
llama_model_loader: - kv  46:                tokenizer.ggml.eos_token_id u32              = 163585
llama_model_loader: - kv  47:            tokenizer.ggml.padding_token_id u32              = 163839
llama_model_loader: - kv  48:                    tokenizer.chat_template str              = {%- if tools -%}\n  <|im_system|>tool_...
llama_model_loader: - kv  49:               general.quantization_version u32              = 2
llama_model_loader: - type  f32:  365 tensors
llama_model_loader: - type q8_0:  792 tensors
================================ Have weights data with 729 entries
[   1/1157]                    token_embd.weight - [ 7168, 163840,     1,     1], type =   q8_0, Using custom type iq4_kt for tensor token_embd.weight

====== llama_model_quantize_internal: did not find weights for token_embd.weight
converting to iq4_kt .. cluster_points: Oops. Cluster 4 has no points:  0 1 0 0
cluster_points: 1 out of 625 clusters dir not have any points
 size =  1190.00 MiB ->   560.62 MiB
[   2/1157]               blk.0.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[   3/1157]                blk.0.ffn_down.weight - [18432,  7168,     1,     1], type =   q8_0, Using custom type iq4_kt for tensor blk.0.ffn_down.weight
converting to iq4_kt .. size =   133.88 MiB ->    63.03 MiB
[   4/1157]                blk.0.ffn_gate.weight - [ 7168, 18432,     1,     1], type =   q8_0, Using custom type iq3_kt for tensor blk.0.ffn_gate.weight
converting to iq3_kt .. size =   133.88 MiB ->    49.29 MiB
[   5/1157]                  blk.0.ffn_up.weight - [ 7168, 18432,     1,     1], type =   q8_0, Using custom type iq3_kt for tensor blk.0.ffn_up.weight
converting to iq3_kt .. size =   133.88 MiB ->    49.29 MiB
[   6/1157]                blk.0.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[   7/1157]          blk.0.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[   8/1157]           blk.0.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, Using custom type iq4_kt for tensor blk.0.attn_kv_a_mqa.weight
converting to iq4_kt .. size =     4.18 MiB ->     1.97 MiB
[   9/1157]               blk.0.attn_kv_b.weight - [  512, 16384,     1,     1], type =   q8_0, Using custom type q8_0 for tensor blk.0.attn_kv_b.weight
size =    8.500 MB
[  10/1157]                blk.0.attn_k_b.weight - [  128, 32768,     1,     1], type =   q8_0, Using custom type iq4_nl for tensor blk.0.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.0.attn_k_b.weight
converting to iq4_nl .. size =     4.25 MiB ->     2.25 MiB
[  11/1157]                blk.0.attn_v_b.weight - [  512,  8192,     1,     1], type =   q8_0, Using custom type iq4_kt for tensor blk.0.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.0.attn_v_b.weight
converting to iq4_kt .. size =     4.25 MiB ->     2.03 MiB
[  12/1157]             blk.0.attn_output.weight - [ 8192,  7168,     1,     1], type =   q8_0, Using custom type iq4_kt for tensor blk.0.attn_output.weight
converting to iq4_kt .. size =    59.50 MiB ->    28.03 MiB
[  13/1157]           blk.0.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  14/1157]                blk.0.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, Using custom type iq4_kt for tensor blk.0.attn_q_a.weight
converting to iq4_kt .. size =    11.16 MiB ->     5.26 MiB
[  15/1157]                blk.0.attn_q_b.weight - [ 1536, 12288,     1,     1], type =   q8_0, Using custom type iq4_kt for tensor blk.0.attn_q_b.weight
converting to iq4_kt .. size =    19.12 MiB ->     9.05 MiB
[  16/1157]               blk.9.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  17/1157]           blk.9.ffn_down_exps.weight - [ 2048,  7168,   384,     1], type =   q8_0, Using custom type iq1_kt for tensor blk.9.ffn_down_exps.weight
converting to iq1_kt .. size =  5712.00 MiB ->  1186.50 MiB
[  18/1157]           blk.9.ffn_gate_exps.weight - [ 7168,  2048,   384,     1], type =   q8_0, Using custom type iq1_s_r4 for tensor blk.9.ffn_gate_exps.weight
converting to iq1_s_r4 .. size =  5712.00 MiB ->  1009.50 MiB
[  19/1157]             blk.9.ffn_up_exps.weight - [ 7168,  2048,   384,     1], type =   q8_0, Using custom type iq1_s_r4 for tensor blk.9.ffn_up_exps.weight
converting to iq1_s_r4 .. size =  5712.00 MiB ->  1009.50 MiB
[  20/1157]               blk.9.exp_probs_b.bias - [  384,     1,     1,     1], type =    f32, size =    0.001 MB
[  21/1157]            blk.9.ffn_gate_inp.weight - [ 7168,   384,     1,     1], type =    f32, size =   10.500 MB
[  22/1157]          blk.9.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, Using custom type iq4_kt for tensor blk.9.ffn_down_shexp.weight
converting to iq4_kt .. size =    14.88 MiB ->     7.03 MiB
[  23/1157]          blk.9.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, Using custom type iq3_kt for tensor blk.9.ffn_gate_shexp.weight
converting to iq3_kt .. size =    14.88 MiB ->     5.48 MiB
[  24/1157]            blk.9.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, Using custom type iq3_kt for tensor blk.9.ffn_up_shexp.weight
converting to iq3_kt .. size =    14.88 MiB ->     5.48 MiB
[  25/1157]                blk.9.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  26/1157]          blk.9.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  27/1157]           blk.9.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, Using custom type iq4_kt for tensor blk.9.attn_kv_a_mqa.weight
converting to iq4_kt .. size =     4.18 MiB ->     1.97 MiB
[  28/1157]               blk.9.attn_kv_b.weight - [  512, 16384,     1,     1], type =   q8_0, Using custom type q8_0 for tensor blk.9.attn_kv_b.weight
size =    8.500 MB
[  29/1157]                blk.9.attn_k_b.weight - [  128, 32768,     1,     1], type =   q8_0, Using custom type iq4_nl for tensor blk.9.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.9.attn_k_b.weight
converting to iq4_nl .. size =     4.25 MiB ->     2.25 MiB
[  30/1157]                blk.9.attn_v_b.weight - [  512,  8192,     1,     1], type =   q8_0, Using custom type iq4_kt for tensor blk.9.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.9.attn_v_b.weight
converting to iq4_kt .. size =     4.25 MiB ->     2.03 MiB
[  31/1157]             blk.9.attn_output.weight - [ 8192,  7168,     1,     1], type =   q8_0, Using custom type iq4_kt for tensor blk.9.attn_output.weight
converting to iq4_kt .. size =    59.50 MiB ->    28.03 MiB
[  32/1157]           blk.9.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  33/1157]                blk.9.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, Using custom type iq4_kt for tensor blk.9.attn_q_a.weight
converting to iq4_kt .. size =    11.16 MiB ->     5.26 MiB
[  34/1157]                blk.9.attn_q_b.weight - [ 1536, 12288,     1,     1], type =   q8_0, Using custom type iq4_kt for tensor blk.9.attn_q_b.weight
converting to iq4_kt .. size =    19.12 MiB ->     9.05 MiB
[  35/1157]              blk.10.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  36/1157]          blk.10.ffn_down_exps.weight - [ 2048,  7168,   384,     1], type =   q8_0, Using custom type iq1_kt for tensor blk.10.ffn_down_exps.weight
converting to iq1_kt .. size =  5712.00 MiB ->  1186.50 MiB
[  37/1157]          blk.10.ffn_gate_exps.weight - [ 7168,  2048,   384,     1], type =   q8_0, Using custom type iq1_s_r4 for tensor blk.10.ffn_gate_exps.weight
converting to iq1_s_r4 .. size =  5712.00 MiB ->  1009.50 MiB
[  38/1157]            blk.10.ffn_up_exps.weight - [ 7168,  2048,   384,     1], type =   q8_0, Using custom type iq1_s_r4 for tensor blk.10.ffn_up_exps.weight
converting to iq1_s_r4 .. size =  5712.00 MiB ->  1009.50 MiB
[  39/1157]              blk.10.exp_probs_b.bias - [  384,     1,     1,     1], type =    f32, size =    0.001 MB
[  40/1157]           blk.10.ffn_gate_inp.weight - [ 7168,   384,     1,     1], type =    f32, size =   10.500 MB
[  41/1157]         blk.10.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, Using custom type iq4_kt for tensor blk.10.ffn_down_shexp.weight
converting to iq4_kt .. size =    14.88 MiB ->     7.03 MiB
[  42/1157]         blk.10.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, Using custom type iq3_kt for tensor blk.10.ffn_gate_shexp.weight
converting to iq3_kt .. size =    14.88 MiB ->     5.48 MiB
[  43/1157]           blk.10.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, Using custom type iq3_kt for tensor blk.10.ffn_up_shexp.weight
converting to iq3_kt .. size =    14.88 MiB ->     5.48 MiB
[  44/1157]               blk.10.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  45/1157]         blk.10.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  46/1157]          blk.10.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, Using custom type iq4_kt for tensor blk.10.attn_kv_a_mqa.weight
converting to iq4_kt .. size =     4.18 MiB ->     1.97 MiB
[  47/1157]              blk.10.attn_kv_b.weight - [  512, 16384,     1,     1], type =   q8_0, Using custom type q8_0 for tensor blk.10.attn_kv_b.weight
size =    8.500 MB
[  48/1157]               blk.10.attn_k_b.weight - [  128, 32768,     1,     1], type =   q8_0, Using custom type iq4_nl for tensor blk.10.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.10.attn_k_b.weight
converting to iq4_nl .. size =     4.25 MiB ->     2.25 MiB
[  49/1157]               blk.10.attn_v_b.weight - [  512,  8192,     1,     1], type =   q8_0, Using custom type iq4_kt for tensor blk.10.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.10.attn_v_b.weight
converting to iq4_kt .. size =     4.25 MiB ->     2.03 MiB
[  50/1157]            blk.10.attn_output.weight - [ 8192,  7168,     1,     1], type =   q8_0, Using custom type iq4_kt for tensor blk.10.attn_output.weight
converting to iq4_kt .. size =    59.50 MiB ->    28.03 MiB
[  51/1157]          blk.10.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[  52/1157]               blk.10.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   q8_0, Using custom type iq4_kt for tensor blk.10.attn_q_a.weight
converting to iq4_kt .. size =    11.16 MiB ->     5.26 MiB
[  53/1157]               blk.10.attn_q_b.weight - [ 1536, 12288,     1,     1], type =   q8_0, Using custom type iq4_kt for tensor blk.10.attn_q_b.weight
converting to iq4_kt .. size =    19.12 MiB ->     9.05 MiB
[  54/1157]              blk.11.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  55/1157]          blk.11.ffn_down_exps.weight - [ 2048,  7168,   384,     1], type =   q8_0, Using custom type iq1_kt for tensor blk.11.ffn_down_exps.weight
converting to iq1_kt .. size =  5712.00 MiB ->  1186.50 MiB
[  56/1157]          blk.11.ffn_gate_exps.weight - [ 7168,  2048,   384,     1], type =   q8_0, Using custom type iq1_s_r4 for tensor blk.11.ffn_gate_exps.weight
converting to iq1_s_r4 .. size =  5712.00 MiB ->  1009.50 MiB
[  57/1157]            blk.11.ffn_up_exps.weight - [ 7168,  2048,   384,     1], type =   q8_0, Using custom type iq1_s_r4 for tensor blk.11.ffn_up_exps.weight
converting to iq1_s_r4 .. size =  5712.00 MiB ->  1009.50 MiB
[  58/1157]              blk.11.exp_probs_b.bias - [  384,     1,     1,     1], type =    f32, size =    0.001 MB
[  59/1157]           blk.11.ffn_gate_inp.weight - [ 7168,   384,     1,     1], type =    f32, size =   10.500 MB
[  60/1157]         blk.11.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   q8_0, Using custom type iq4_kt for tensor blk.11.ffn_down_shexp.weight
converting to iq4_kt .. size =    14.88 MiB ->     7.03 MiB
[  61/1157]         blk.11.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, Using custom type iq3_kt for tensor blk.11.ffn_gate_shexp.weight
converting to iq3_kt .. size =    14.88 MiB ->     5.48 MiB
[  62/1157]           blk.11.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   q8_0, Using custom type iq3_kt for tensor blk.11.ffn_up_shexp.weight
converting to iq3_kt .. size =    14.88 MiB ->     5.48 MiB
[  63/1157]               blk.11.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[  64/1157]         blk.11.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[  65/1157]          blk.11.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   q8_0, Using custom type iq4_kt for tensor blk.11.attn_kv_a_mqa.weight
converting to iq4_kt .. size =     4.18 MiB ->     1.97 MiB
[  66/1157]              blk.11.attn_kv_b.weight - [  512, 16384,     1,     1], type =   q8_0, Using custom type q8_0 for tensor blk.11.attn_kv_b.weight
size =    8.500 MB
[  67/1157]               blk.11.attn_k_b.weight - [  128, 32768,     1,     1], type =   q8_0, Using custom type iq4_nl for tensor blk.11.attn_k_b.weight

====== llama_model_quantize_internal: did not find weights for blk.11.attn_k_b.weight
converting to iq4_nl .. size =     4.25 MiB ->     2.25 MiB
[  68/1157]               blk.11.attn_v_b.weight - [  512,  8192,     1,     1], type =   q8_0, Using custom type iq4_kt for tensor blk.11.attn_v_b.weight

====== llama_model_quantize_internal: did not find weights for blk.11.attn_v_b.weight
converting to iq4_kt .. size =     4.25 MiB ->     2.03 MiB
[  69/1157]            blk.11.attn_output.weight - [ 8192,  7168,     1,     1], type =   q8_0, Using custom type iq4_kt for tensor blk.11.attn_output.weight
converting to iq4_kt .. 
```
</details>

2) How much accuracy do we lose by requantizing from Q8_0 instead of BF16?  

Thanks!

---

👤 **ikawrakow** commented on **2025-07-19** at **16:14:53**

> However, could token_embd.weight be missing because I used Q8_0 as input? 

One never calculates imatrix data for token embeddings. I should go and add a check to not print this warning to avoid people worrying about this.

The warnings about missing cluster points are harmless.

The warning about missing imatrix data for `attn_k_b` is not good. 

> How much accuracy do we lose by requantizing from Q8_0 instead of BF16?

Not much, if any.

---

👤 **ubergarm** commented on **2025-07-19** at **16:38:41**

> The warning about missing imatrix data for attn_k_b is not good.

Hrrm, I too see this for my Kimi-K2-Instruct quantize logs:

```bash
====== llama_model_quantize_internal: did not find weights for blk.5.attn_kv_b.weight
====== llama_model_quantize_internal: did not find weights for blk.5.attn_k_b.weight
====== llama_model_quantize_internal: did not find weights for blk.5.attn_v_b.weight
```

Looking back at my deepseek quantization logs it only has:
```bash
====== llama_model_quantize_internal: did not find weights for blk.47.attn_k_b.weight
```

(fwiw attn_k_b is not divisible by 256 so i've had to use like q5_0 or iq4_nl, might be related to the imatrix stuff, not sure)

The main difference is that for kimi-k2 imatrix i used `-mla 1` whereas with the older deepseek imatrix i did not specify `-mla` at all.

Also, yesterday I discovered that Kimi-K2-Instruct seems very sensitive to attn/shexp/blk.0.ffn.* or possibly just attn. I'm thinking it is because Kimi-K2 uses half the attn heads and 33% of the ffn dense layers as DeepSeek. So going back and requantizing my recipes with full q8_0 attn/shexp/blk.0.ffn.* is improving Perplexity a lot for a little BPW. 

So now I'm not sure if this is because of those architecture changes in Kimi-K2, or perhaps just my imatrix was not being properly applied to the MLA tensors? hrmm...

I'm updating the chart and data with what I have so far up above: https://github.com/ikawrakow/ik_llama.cpp/pull/616#issuecomment-3087170346

> Note that the Q8_0 input was made from convert_hf_to_gguf.py:
python convert_hf_to_gguf.py --outfile /home/user/storage/gguf/Kimi-K2-Instruct/Kimi-K2-Instruct-Q8_0.gguf /home/user/storage/llm/Kimi-K2-Instruct-BF16/ --outtype q8_0 --model-name Kimi-K2-Instruct --split-max-size 9999G

Oh interesting, I used `--outtype bf16` for my convert and then quantize from the full bf16. Its 2TB file size though so chews up disk space!

---

👤 **magikRUKKOLA** commented on **2025-07-19** at **22:39:56**

@ubergarm 

Here is my dump:
```bash
/opt/unsloth/Kimi-K2-Instruct-GGUF/UD-IQ3_XXS# find ./ -name "*gguf" | xargs -I{} gguf-dump "./{}" &> /tmp/dump.log
```

```diff
--- /tmp/dump2.log	2025-07-20 01:34:55.913286620 +0300
+++ /tmp/dump.log	2025-07-20 01:36:37.213790237 +0300
@@ -1,9 +1,9 @@
-INFO:gguf-dump:* Loading: /mnt/data/models/unsloth/Kimi-K2-Instruct-GGUF/UD-IQ3_XXS/Kimi-K2-Instruct-UD-IQ3_XXS-00001-of-00009.gguf
+INFO:gguf-dump:* Loading: ././Kimi-K2-Instruct-UD-IQ3_XXS-00001-of-00009.gguf
 * File is LITTLE endian, script is running on a LITTLE endian host.
-* Dumping 64 key/value pair(s)
+* Dumping 65 key/value pair(s)
       1: UINT32     |        1 | GGUF.version = 3
       2: UINT64     |        1 | GGUF.tensor_count = 134
-      3: UINT64     |        1 | GGUF.kv_count = 61
+      3: UINT64     |        1 | GGUF.kv_count = 62
       4: STRING     |        1 | general.architecture = 'deepseek2'
       5: STRING     |        1 | general.type = 'model'
       6: STRING     |        1 | general.name = 'Kimi-K2-Instruct'
@@ -15,10 +15,10 @@
      12: STRING     |        1 | general.license.name = 'modified-mit'
      13: STRING     |        1 | general.repo_url = 'https://huggingface.co/unsloth'
      14: UINT32     |        1 | general.base_model.count = 1
-     15: STRING     |        1 | general.base_model.0.name = 'Kimi K2 Instruct'
+     15: STRING     |        1 | general.base_model.0.name = 'Kimi K2 Instruct BF16'
      16: STRING     |        1 | general.base_model.0.organization = 'Moonshotai'
-     17: STRING     |        1 | general.base_model.0.repo_url = 'https://huggingface.co/moonshotai/Kimi-K2-Instruct'
-     18: [STRING]   |        1 | general.tags
+     17: STRING     |        1 | general.base_model.0.repo_url = 'https://huggingface.co/moonshotai/Kimi-K2-Instruct-BF16'
+     18: [STRING]   |       11 | general.tags = ['unsloth', 'unsloth', 'unsloth', 'unsloth', 'unsloth', 'unsloth', ...]
      19: UINT32     |        1 | deepseek2.block_count = 61
      20: UINT32     |        1 | deepseek2.context_length = 131072
      21: UINT32     |        1 | deepseek2.embedding_length = 7168
@@ -47,24 +47,25 @@
      44: FLOAT32    |        1 | deepseek2.rope.scaling.factor = 32.0
      45: UINT32     |        1 | deepseek2.rope.scaling.original_context_length = 4096
      46: FLOAT32    |        1 | deepseek2.rope.scaling.yarn_log_multiplier = 0.10000000149011612
-     47: STRING     |        1 | tokenizer.ggml.model = 'gpt2'
-     48: STRING     |        1 | tokenizer.ggml.pre = 'kimi-k2'
-     49: [STRING]   |   163840 | tokenizer.ggml.tokens
-     50: [INT32]    |   163840 | tokenizer.ggml.token_type
-     51: [STRING]   |   163328 | tokenizer.ggml.merges
-     52: UINT32     |        1 | tokenizer.ggml.bos_token_id = 163584
-     53: UINT32     |        1 | tokenizer.ggml.eos_token_id = 163585
-     54: UINT32     |        1 | tokenizer.ggml.padding_token_id = 163839
-     55: STRING     |        1 | tokenizer.chat_template = '{%- if tools -%}\n  <|im_system|>tool_declare<|im_middle|>{{ '
-     56: UINT32     |        1 | general.quantization_version = 2
-     57: UINT32     |        1 | general.file_type = 23
-     58: STRING     |        1 | quantize.imatrix.file = 'Kimi-K2-Instruct-GGUF/imatrix_unsloth.dat'
-     59: STRING     |        1 | quantize.imatrix.dataset = 'unsloth_calibration_Kimi-K2-Instruct.txt'
-     60: UINT32     |        1 | quantize.imatrix.entries_count = 667
-     61: UINT32     |        1 | quantize.imatrix.chunks_count = 714
-     62: UINT16     |        1 | split.no = 0
-     63: INT32      |        1 | split.tensors.count = 1096
-     64: UINT16     |        1 | split.count = 9
+     47: UINT32     |        1 | tokenizer.ggml.bos_token_id = 163584
+     48: UINT32     |        1 | tokenizer.ggml.eos_token_id = 163586
+     49: UINT32     |        1 | tokenizer.ggml.padding_token_id = 163839
+     50: STRING     |        1 | tokenizer.chat_template = "{% if tools -%}\n    {{ '<|im_system|>tool_declare<|im_mid..."
+     51: BOOL       |        1 | tokenizer.ggml.add_bos_token = False
+     52: STRING     |        1 | tokenizer.ggml.model = 'gpt2'
+     53: STRING     |        1 | tokenizer.ggml.pre = 'kimi-k2'
+     54: [STRING]   |   163840 | tokenizer.ggml.tokens = ['!', '"', '#', '$', '%', '&', ...]
+     55: [INT32]    |   163840 | tokenizer.ggml.token_type = [1, 1, 1, 1, 1, 1, ...]
+     56: [STRING]   |   163328 | tokenizer.ggml.merges = ['Ġ Ġ', 'ĠĠ ĠĠ', 'Ġ t', 'i n', 'ä ¸', 'Ġ a', ...]
+     57: UINT32     |        1 | general.quantization_version = 2
+     58: UINT32     |        1 | general.file_type = 23
+     59: STRING     |        1 | quantize.imatrix.file = 'Kimi-K2-Instruct-GGUF/imatrix_unsloth.dat'
+     60: STRING     |        1 | quantize.imatrix.dataset = 'unsloth_calibration_Kimi-K2-Instruct.txt'
+     61: UINT32     |        1 | quantize.imatrix.entries_count = 667
+     62: UINT32     |        1 | quantize.imatrix.chunks_count = 714
+     63: UINT16     |        1 | split.no = 0
+     64: INT32      |        1 | split.tensors.count = 1096
+     65: UINT16     |        1 | split.count = 9
 * Dumping 134 tensor(s)
       1: 1174405120 |  7168, 163840,     1,     1 | Q6_K    | output.weight
       2:       7168 |  7168,     1,     1,     1 | F32     | output_norm.weight
@@ -200,7 +201,7 @@
     132:   18874368 |  1536, 12288,     1,     1 | Q5_K    | blk.7.attn_q_b.weight
     133:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.7.attn_v_b.weight
     134:        384 |   384,     1,     1,     1 | F32     | blk.7.exp_probs_b.bias
-INFO:gguf-dump:* Loading: /mnt/data/models/unsloth/Kimi-K2-Instruct-GGUF/UD-IQ3_XXS/Kimi-K2-Instruct-UD-IQ3_XXS-00002-of-00009.gguf
+INFO:gguf-dump:* Loading: ././Kimi-K2-Instruct-UD-IQ3_XXS-00002-of-00009.gguf
 * File is LITTLE endian, script is running on a LITTLE endian host.
 * Dumping 6 key/value pair(s)
       1: UINT32     |        1 | GGUF.version = 3
@@ -338,7 +339,7 @@
     126:        384 |   384,     1,     1,     1 | F32     | blk.14.exp_probs_b.bias
     127: 5637144576 |  2048,  7168,   384,     1 | IQ3_XXS | blk.14.ffn_down_exps.weight
     128:   14680064 |  2048,  7168,     1,     1 | IQ4_XS  | blk.14.ffn_down_shexp.weight
-INFO:gguf-dump:* Loading: /mnt/data/models/unsloth/Kimi-K2-Instruct-GGUF/UD-IQ3_XXS/Kimi-K2-Instruct-UD-IQ3_XXS-00003-of-00009.gguf
+INFO:gguf-dump:* Loading: ././Kimi-K2-Instruct-UD-IQ3_XXS-00003-of-00009.gguf
 * File is LITTLE endian, script is running on a LITTLE endian host.
 * Dumping 6 key/value pair(s)
       1: UINT32     |        1 | GGUF.version = 3
@@ -474,7 +475,7 @@
     124:        384 |   384,     1,     1,     1 | F32     | blk.21.exp_probs_b.bias
     125: 5637144576 |  2048,  7168,   384,     1 | IQ3_XXS | blk.21.ffn_down_exps.weight
     126:   14680064 |  2048,  7168,     1,     1 | IQ4_XS  | blk.21.ffn_down_shexp.weight
-INFO:gguf-dump:* Loading: /mnt/data/models/unsloth/Kimi-K2-Instruct-GGUF/UD-IQ3_XXS/Kimi-K2-Instruct-UD-IQ3_XXS-00004-of-00009.gguf
+INFO:gguf-dump:* Loading: ././Kimi-K2-Instruct-UD-IQ3_XXS-00004-of-00009.gguf
 * File is LITTLE endian, script is running on a LITTLE endian host.
 * Dumping 6 key/value pair(s)
       1: UINT32     |        1 | GGUF.version = 3
@@ -614,7 +615,7 @@
     128:    2752512 |  7168,   384,     1,     1 | F32     | blk.28.ffn_gate_inp.weight
     129:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.28.ffn_gate_shexp.weight
     130:       7168 |  7168,     1,     1,     1 | F32     | blk.28.ffn_norm.weight
-INFO:gguf-dump:* Loading: /mnt/data/models/unsloth/Kimi-K2-Instruct-GGUF/UD-IQ3_XXS/Kimi-K2-Instruct-UD-IQ3_XXS-00005-of-00009.gguf
+INFO:gguf-dump:* Loading: ././Kimi-K2-Instruct-UD-IQ3_XXS-00005-of-00009.gguf
 * File is LITTLE endian, script is running on a LITTLE endian host.
 * Dumping 6 key/value pair(s)
       1: UINT32     |        1 | GGUF.version = 3
@@ -762,7 +763,145 @@
     136:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.36.attn_q_b.weight
     137:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.36.attn_v_b.weight
     138:        384 |   384,     1,     1,     1 | F32     | blk.36.exp_probs_b.bias
-INFO:gguf-dump:* Loading: /mnt/data/models/unsloth/Kimi-K2-Instruct-GGUF/UD-IQ3_XXS/Kimi-K2-Instruct-UD-IQ3_XXS-00007-of-00009.gguf
+INFO:gguf-dump:* Loading: ././Kimi-K2-Instruct-UD-IQ3_XXS-00006-of-00009.gguf
+* File is LITTLE endian, script is running on a LITTLE endian host.
+* Dumping 6 key/value pair(s)
+      1: UINT32     |        1 | GGUF.version = 3
+      2: UINT64     |        1 | GGUF.tensor_count = 128
+      3: UINT64     |        1 | GGUF.kv_count = 3
+      4: UINT16     |        1 | split.no = 5
+      5: INT32      |        1 | split.tensors.count = 1096
+      6: UINT16     |        1 | split.count = 9
+* Dumping 128 tensor(s)
+      1: 5637144576 |  2048,  7168,   384,     1 | IQ3_XXS | blk.36.ffn_down_exps.weight
+      2:   14680064 |  2048,  7168,     1,     1 | IQ4_XS  | blk.36.ffn_down_shexp.weight
+      3: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.36.ffn_gate_exps.weight
+      4:    2752512 |  7168,   384,     1,     1 | F32     | blk.36.ffn_gate_inp.weight
+      5:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.36.ffn_gate_shexp.weight
+      6:       7168 |  7168,     1,     1,     1 | F32     | blk.36.ffn_norm.weight
+      7: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.36.ffn_up_exps.weight
+      8:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.36.ffn_up_shexp.weight
+      9:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.37.attn_k_b.weight
+     10:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.37.attn_kv_a_mqa.weight
+     11:        512 |   512,     1,     1,     1 | F32     | blk.37.attn_kv_a_norm.weight
+     12:       7168 |  7168,     1,     1,     1 | F32     | blk.37.attn_norm.weight
+     13:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.37.attn_output.weight
+     14:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.37.attn_q_a.weight
+     15:       1536 |  1536,     1,     1,     1 | F32     | blk.37.attn_q_a_norm.weight
+     16:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.37.attn_q_b.weight
+     17:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.37.attn_v_b.weight
+     18:        384 |   384,     1,     1,     1 | F32     | blk.37.exp_probs_b.bias
+     19: 5637144576 |  2048,  7168,   384,     1 | IQ3_XXS | blk.37.ffn_down_exps.weight
+     20:   14680064 |  2048,  7168,     1,     1 | IQ4_XS  | blk.37.ffn_down_shexp.weight
+     21: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.37.ffn_gate_exps.weight
+     22:    2752512 |  7168,   384,     1,     1 | F32     | blk.37.ffn_gate_inp.weight
+     23:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.37.ffn_gate_shexp.weight
+     24:       7168 |  7168,     1,     1,     1 | F32     | blk.37.ffn_norm.weight
+     25: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.37.ffn_up_exps.weight
+     26:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.37.ffn_up_shexp.weight
+     27:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.38.attn_k_b.weight
+     28:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.38.attn_kv_a_mqa.weight
+     29:        512 |   512,     1,     1,     1 | F32     | blk.38.attn_kv_a_norm.weight
+     30:       7168 |  7168,     1,     1,     1 | F32     | blk.38.attn_norm.weight
+     31:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.38.attn_output.weight
+     32:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.38.attn_q_a.weight
+     33:       1536 |  1536,     1,     1,     1 | F32     | blk.38.attn_q_a_norm.weight
+     34:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.38.attn_q_b.weight
+     35:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.38.attn_v_b.weight
+     36:        384 |   384,     1,     1,     1 | F32     | blk.38.exp_probs_b.bias
+     37: 5637144576 |  2048,  7168,   384,     1 | IQ3_XXS | blk.38.ffn_down_exps.weight
+     38:   14680064 |  2048,  7168,     1,     1 | IQ4_XS  | blk.38.ffn_down_shexp.weight
+     39: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.38.ffn_gate_exps.weight
+     40:    2752512 |  7168,   384,     1,     1 | F32     | blk.38.ffn_gate_inp.weight
+     41:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.38.ffn_gate_shexp.weight
+     42:       7168 |  7168,     1,     1,     1 | F32     | blk.38.ffn_norm.weight
+     43: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.38.ffn_up_exps.weight
+     44:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.38.ffn_up_shexp.weight
+     45:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.39.attn_k_b.weight
+     46:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.39.attn_kv_a_mqa.weight
+     47:        512 |   512,     1,     1,     1 | F32     | blk.39.attn_kv_a_norm.weight
+     48:       7168 |  7168,     1,     1,     1 | F32     | blk.39.attn_norm.weight
+     49:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.39.attn_output.weight
+     50:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.39.attn_q_a.weight
+     51:       1536 |  1536,     1,     1,     1 | F32     | blk.39.attn_q_a_norm.weight
+     52:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.39.attn_q_b.weight
+     53:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.39.attn_v_b.weight
+     54:        384 |   384,     1,     1,     1 | F32     | blk.39.exp_probs_b.bias
+     55: 5637144576 |  2048,  7168,   384,     1 | IQ3_XXS | blk.39.ffn_down_exps.weight
+     56:   14680064 |  2048,  7168,     1,     1 | IQ4_XS  | blk.39.ffn_down_shexp.weight
+     57: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.39.ffn_gate_exps.weight
+     58:    2752512 |  7168,   384,     1,     1 | F32     | blk.39.ffn_gate_inp.weight
+     59:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.39.ffn_gate_shexp.weight
+     60:       7168 |  7168,     1,     1,     1 | F32     | blk.39.ffn_norm.weight
+     61: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.39.ffn_up_exps.weight
+     62:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.39.ffn_up_shexp.weight
+     63:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.40.attn_k_b.weight
+     64:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.40.attn_kv_a_mqa.weight
+     65:        512 |   512,     1,     1,     1 | F32     | blk.40.attn_kv_a_norm.weight
+     66:       7168 |  7168,     1,     1,     1 | F32     | blk.40.attn_norm.weight
+     67:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.40.attn_output.weight
+     68:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.40.attn_q_a.weight
+     69:       1536 |  1536,     1,     1,     1 | F32     | blk.40.attn_q_a_norm.weight
+     70:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.40.attn_q_b.weight
+     71:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.40.attn_v_b.weight
+     72:        384 |   384,     1,     1,     1 | F32     | blk.40.exp_probs_b.bias
+     73: 5637144576 |  2048,  7168,   384,     1 | IQ3_XXS | blk.40.ffn_down_exps.weight
+     74:   14680064 |  2048,  7168,     1,     1 | IQ4_XS  | blk.40.ffn_down_shexp.weight
+     75: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.40.ffn_gate_exps.weight
+     76:    2752512 |  7168,   384,     1,     1 | F32     | blk.40.ffn_gate_inp.weight
+     77:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.40.ffn_gate_shexp.weight
+     78:       7168 |  7168,     1,     1,     1 | F32     | blk.40.ffn_norm.weight
+     79: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.40.ffn_up_exps.weight
+     80:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.40.ffn_up_shexp.weight
+     81:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.41.attn_k_b.weight
+     82:    4128768 |  7168,   576,     1,     1 | Q6_K    | blk.41.attn_kv_a_mqa.weight
+     83:        512 |   512,     1,     1,     1 | F32     | blk.41.attn_kv_a_norm.weight
+     84:       7168 |  7168,     1,     1,     1 | F32     | blk.41.attn_norm.weight
+     85:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.41.attn_output.weight
+     86:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.41.attn_q_a.weight
+     87:       1536 |  1536,     1,     1,     1 | F32     | blk.41.attn_q_a_norm.weight
+     88:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.41.attn_q_b.weight
+     89:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.41.attn_v_b.weight
+     90:        384 |   384,     1,     1,     1 | F32     | blk.41.exp_probs_b.bias
+     91: 5637144576 |  2048,  7168,   384,     1 | IQ3_XXS | blk.41.ffn_down_exps.weight
+     92:   14680064 |  2048,  7168,     1,     1 | IQ4_XS  | blk.41.ffn_down_shexp.weight
+     93: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.41.ffn_gate_exps.weight
+     94:    2752512 |  7168,   384,     1,     1 | F32     | blk.41.ffn_gate_inp.weight
+     95:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.41.ffn_gate_shexp.weight
+     96:       7168 |  7168,     1,     1,     1 | F32     | blk.41.ffn_norm.weight
+     97: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.41.ffn_up_exps.weight
+     98:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.41.ffn_up_shexp.weight
+     99:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.42.attn_k_b.weight
+    100:    4128768 |  7168,   576,     1,     1 | IQ4_XS  | blk.42.attn_kv_a_mqa.weight
+    101:        512 |   512,     1,     1,     1 | F32     | blk.42.attn_kv_a_norm.weight
+    102:       7168 |  7168,     1,     1,     1 | F32     | blk.42.attn_norm.weight
+    103:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.42.attn_output.weight
+    104:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.42.attn_q_a.weight
+    105:       1536 |  1536,     1,     1,     1 | F32     | blk.42.attn_q_a_norm.weight
+    106:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.42.attn_q_b.weight
+    107:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.42.attn_v_b.weight
+    108:        384 |   384,     1,     1,     1 | F32     | blk.42.exp_probs_b.bias
+    109: 5637144576 |  2048,  7168,   384,     1 | IQ3_XXS | blk.42.ffn_down_exps.weight
+    110:   14680064 |  2048,  7168,     1,     1 | IQ4_XS  | blk.42.ffn_down_shexp.weight
+    111: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.42.ffn_gate_exps.weight
+    112:    2752512 |  7168,   384,     1,     1 | F32     | blk.42.ffn_gate_inp.weight
+    113:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.42.ffn_gate_shexp.weight
+    114:       7168 |  7168,     1,     1,     1 | F32     | blk.42.ffn_norm.weight
+    115: 5637144576 |  7168,  2048,   384,     1 | IQ3_XXS | blk.42.ffn_up_exps.weight
+    116:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.42.ffn_up_shexp.weight
+    117:    4194304 |   128,   512,    64,     1 | Q8_0    | blk.43.attn_k_b.weight
+    118:    4128768 |  7168,   576,     1,     1 | Q6_K    | blk.43.attn_kv_a_mqa.weight
+    119:        512 |   512,     1,     1,     1 | F32     | blk.43.attn_kv_a_norm.weight
+    120:       7168 |  7168,     1,     1,     1 | F32     | blk.43.attn_norm.weight
+    121:   58720256 |  8192,  7168,     1,     1 | IQ4_XS  | blk.43.attn_output.weight
+    122:   11010048 |  7168,  1536,     1,     1 | Q4_K    | blk.43.attn_q_a.weight
+    123:       1536 |  1536,     1,     1,     1 | F32     | blk.43.attn_q_a_norm.weight
+    124:   18874368 |  1536, 12288,     1,     1 | IQ4_XS  | blk.43.attn_q_b.weight
+    125:    4194304 |   512,   128,    64,     1 | Q8_0    | blk.43.attn_v_b.weight
+    126:        384 |   384,     1,     1,     1 | F32     | blk.43.exp_probs_b.bias
+    127: 5637144576 |  2048,  7168,   384,     1 | IQ3_XXS | blk.43.ffn_down_exps.weight
+    128:   14680064 |  2048,  7168,     1,     1 | IQ4_XS  | blk.43.ffn_down_shexp.weight
+INFO:gguf-dump:* Loading: ././Kimi-K2-Instruct-UD-IQ3_XXS-00007-of-00009.gguf
 * File is LITTLE endian, script is running on a LITTLE endian host.
 * Dumping 6 key/value pair(s)
       1: UINT32     |        1 | GGUF.version = 3
@@ -902,7 +1041,7 @@
     128:    2752512 |  7168,   384,     1,     1 | F32     | blk.50.ffn_gate_inp.weight
     129:   14680064 |  7168,  2048,     1,     1 | IQ4_XS  | blk.50.ffn_gate_shexp.weight
     130:       7168 |  7168,     1,     1,     1 | F32     | blk.50.ffn_norm.weight
-INFO:gguf-dump:* Loading: /mnt/data/models/unsloth/Kimi-K2-Instruct-GGUF/UD-IQ3_XXS/Kimi-K2-Instruct-UD-IQ3_XXS-00008-of-00009.gguf
+INFO:gguf-dump:* Loading: ././Kimi-K2-Instruct-UD-IQ3_XXS-00008-of-00009.gguf
 * File is LITTLE endian, script is running on a LITTLE endian host.
 * Dumping 6 key/value pair(s)
       1: UINT32     |        1 | GGUF.version = 3
@@ -1034,7 +1173,7 @@
     120:        384 |   384,     1,     1,     1 | F32     | blk.57.exp_probs_b.bias
     121: 5637144576 |  2048,  7168,   384,     1 | IQ4_XS  | blk.57.ffn_down_exps.weight
     122:   14680064 |  2048,  7168,     1,     1 | Q6_K    | blk.57.ffn_down_shexp.weight
-INFO:gguf-dump:* Loading: /mnt/data/models/unsloth/Kimi-K2-Instruct-GGUF/UD-IQ3_XXS/Kimi-K2-Instruct-UD-IQ3_XXS-00009-of-00009.gguf
+INFO:gguf-dump:* Loading: ././Kimi-K2-Instruct-UD-IQ3_XXS-00009-of-00009.gguf
 * File is LITTLE endian, script is running on a LITTLE endian host.
 * Dumping 6 key/value pair(s)
       1: UINT32     |        1 | GGUF.version = 3

```

[EDIT]:
hashes:
```
83ede91803add96bc6dedb654fb63d8623cbe49116ffa6157f7187038d0b844e  Kimi-K2-Instruct-UD-IQ3_XXS-00001-of-00009.gguf
0a13cd9912d8b2fd04d39db1e49c6b34e36b228b95511b5623211003903e1159  Kimi-K2-Instruct-UD-IQ3_XXS-00002-of-00009.gguf
7e756f2fb141dc6b9dc76905485b82997b03537594eeed6f00b000cb9ca8118e  Kimi-K2-Instruct-UD-IQ3_XXS-00003-of-00009.gguf
e22a7a9eabe7b65da81c61427ab0e878accc15abc1fc6996b01c904dc57f32b6  Kimi-K2-Instruct-UD-IQ3_XXS-00004-of-00009.gguf
67563c3134d19652726822daf641f5b83fd617e7ab1538965d51755323f11dcb  Kimi-K2-Instruct-UD-IQ3_XXS-00005-of-00009.gguf
952230068e615e8b5d04c67489b729196ddb323179b66e9838c3f1b9fe726636  Kimi-K2-Instruct-UD-IQ3_XXS-00006-of-00009.gguf
1628351d2b1659e4c7a52270955b9ce2c93d7781726318e53f5fe4152ddcb687  Kimi-K2-Instruct-UD-IQ3_XXS-00007-of-00009.gguf
6149283b6e36c5bd87db0fabe091ebec433e1f8126bfef847d3af39d652abf3a  Kimi-K2-Instruct-UD-IQ3_XXS-00008-of-00009.gguf
72aca833efedb0b00424eba6655a8c7319d27b1e9b87a4178b714c1b7770b53c  Kimi-K2-Instruct-UD-IQ3_XXS-00009-of-00009.gguf
```

```bash
/opt/unsloth/Kimi-K2-Instruct-GGUF/UD-IQ3_XXS# ls -lah *gguf
-rw-r--r-- 1 root root 46G Jul 16 11:17 Kimi-K2-Instruct-UD-IQ3_XXS-00001-of-00009.gguf
-rw-r--r-- 1 root root 47G Jul 16 12:35 Kimi-K2-Instruct-UD-IQ3_XXS-00002-of-00009.gguf
-rw-r--r-- 1 root root 45G Jul 16 13:50 Kimi-K2-Instruct-UD-IQ3_XXS-00003-of-00009.gguf
-rw-r--r-- 1 root root 46G Jul 17 02:54 Kimi-K2-Instruct-UD-IQ3_XXS-00004-of-00009.gguf
-rw-r--r-- 1 root root 45G Jul 16 16:36 Kimi-K2-Instruct-UD-IQ3_XXS-00005-of-00009.gguf
-rw-r--r-- 1 root root 45G Jul 16 18:00 Kimi-K2-Instruct-UD-IQ3_XXS-00006-of-00009.gguf
-rw-r--r-- 1 root root 45G Jul 16 19:24 Kimi-K2-Instruct-UD-IQ3_XXS-00007-of-00009.gguf
-rw-r--r-- 1 root root 46G Jul 17 04:22 Kimi-K2-Instruct-UD-IQ3_XXS-00008-of-00009.gguf
-rw-r--r-- 1 root root 27G Jul 17 02:34 Kimi-K2-Instruct-UD-IQ3_XXS-00009-of-00009.gguf
```

@ubergarm 

it seems like we are dealing with ~~two different revisions fo the same quant.~~ different tokenization configs embedded into GGUF (!?)
Also, your info regarding Kimi-K2-Instruct-UD-IQ3_XXS-00006-of-00009.gguf is missing.

[EDIT]:

```
* Extra key/value pair                                                     
  Old:  GGUF.kv_count = 61                                                    
  New:  GGUF.kv_count = 62                                                    
  The new dump contains one additional metadata field.                        
*  Extra metadata field                                                     
  New key #51:                                                                
    tokenizer.ggml.add_bos_token = False                                      
  This boolean flag tells downstream loaders not to automatically prepend the 
  BOS token ( <s> ) when the tokenizer is applied.                            
  Absence of this key (old dump) means “let the program decide” or default to 
  True ; with the key present and set to  False  the tokenizer will not add a 
  leading BOS token.                                                          
 * Base-model reference updated                                             
  Old:                                                                        
    general.base_model.0.name   = "Kimi K2 Instruct"                          
    general.base_model.0.repo_url = "https://huggingface.co/moonshotai/Kimi-K2-
  Instruct"                                                                   
  New:                                                                        
    general.base_model.0.name   = "Kimi K2 Instruct BF16"                     
    general.base_model.0.repo_url = "https://huggingface.co/moonshotai/Kimi-K2-
  Instruct-BF16"                                                              
  The quantized model now declares the BF16 checkpoint as its base instead of 
  the original one.                                                           
 * EOS token id changed                                                     
  Old:  tokenizer.ggml.eos_token_id = 163585                                  
  New:  tokenizer.ggml.eos_token_id = 163586                                  
  The end-of-sentence token id was shifted by one. This is usually caused by  
  re-generating the tokenizer vocabulary (or adding a new special token) after
  the base model was switched to the BF16 release.   
```

---

👤 **ikawrakow** commented on **2025-07-20** at **08:30:26**

> Hrrm, I too see this for my Kimi-K2-Instruct quantize logs:
>
>====== llama_model_quantize_internal: did not find weights for blk.5.attn_kv_b.weight
>====== llama_model_quantize_internal: did not find weights for blk.5.attn_k_b.weight
>====== llama_model_quantize_internal: did not find weights for blk.5.attn_v_b.weight

@ubergarm  As discussed elsewhere, it is expected that there is no imatrix data for `attn_kv_b`. But no imatrix data for `attn_k_b` and `attn_v_b` is unexpected if you used `-mla 1`. Could you please run the imatrix tool adding `--verbosity 2` to your command line? There will be a lot of output to `stdout` with that, so redirect to a log file and post the log here. You only need to run 1 batch so we see the names of all tensors where data is being captured.

---

👤 **ThomasBaruzier** commented on **2025-07-20** at **10:15:59**

> The warning about missing imatrix data for attn_k_b is not good.

> The main difference is that for kimi-k2 imatrix i used -mla 1 whereas with the older deepseek imatrix i did not specify -mla at all.

Is it related to how we produce the imatrix and what value of MLA we use while doing so? Edit: nvm I didn't read the answer above

> So going back and requantizing my recipes with full q8_0 attn/shexp/blk.0.ffn.*

I will try as well. Using your edited recipe, everything minus ffn gate up down is very small:

 | Name                          |   Count |   Size (MB) |   Total (MB) |
|:------------------------------|--------:|------------:|-------------:|
| blk.{i}.ffn_down_exps.weight  |      59 |   1186      |   70004      |
| blk.{i}.ffn_gate_exps.weight  |      59 |   1010      |   59560      |
| blk.{i}.ffn_up_exps.weight    |      59 |   1010      |   59560      |
| blk.{i}.attn_output.weight    |      59 |     28      |    1654      |
| output.weight                 |       1 |    736      |     736      |
| blk.{i}.ffn_gate_inp.weight   |      59 |     10.5    |     620      |
| blk.{i}.ffn_norm.weight       |       1 |    561      |     561      |
| blk.{i}.attn_q_b.weight       |      59 |      9.05   |     534      |
| blk.{i}.attn_kv_b.weight      |      59 |      8.5    |     502      |
| blk.{i}.ffn_down_shexp.weight |      59 |      7.03   |     415      |
| blk.{i}.ffn_gate_shexp.weight |      59 |      5.48   |     323      |
| blk.{i}.ffn_up_shexp.weight   |      59 |      5.48   |     323      |
| blk.{i}.attn_q_a.weight       |      59 |      5.26   |     310      |
| blk.{i}.attn_k_b.weight       |      59 |      2.25   |     133      |
| blk.{i}.attn_v_b.weight       |      59 |      2.03   |     120      |
| blk.{i}.attn_kv_a_mqa.weight  |      59 |      1.97   |     116      |
| blk.{i}.ffn_down.weight       |       1 |     63      |      63      |
| blk.{i}.ffn_gate.weight       |       1 |     49.3    |      49.3    |
| blk.{i}.ffn_up.weight         |       1 |     49.3    |      49.3    |
| blk.{i}.attn_norm.weight      |      60 |      0.0273 |       1.64   |
| blk.{i}.ffn_norm.weight       |      59 |      0.0273 |       1.61   |
| blk.{i}.attn_q_a_norm.weight  |      59 |      0.0059 |       0.346  |
| blk.{i}.attn_kv_a_norm.weight |      59 |      0.002  |       0.115  |
| blk.{i}.exp_probs_b.bias      |      59 |      0.0015 |       0.0864 |
| output_norm.weight            |       1 |      0.0273 |       0.0273 |
| token_embd.weight             |       1 |      0.002  |       0.002  |

For Q8_0:

| Name                          |   Count |   Size (MB) |   Total (MB) |
|:------------------------------|--------:|------------:|-------------:|
| blk.{i}.ffn_down_exps.weight  |      59 |   5712      |  337008      |
| blk.{i}.ffn_gate_exps.weight  |      59 |   5712      |  337008      |
| blk.{i}.ffn_up_exps.weight    |      59 |   5712      |  337008      |
| blk.{i}.attn_output.weight    |      59 |     59.5    |    3510      |
| token_embd.weight             |       1 |   1190      |    1190      |
| output.weight                 |       1 |   1190      |    1190      |
| blk.{i}.attn_q_b.weight       |      59 |     19.1    |    1128      |
| blk.{i}.ffn_down_shexp.weight |      58 |     14.9    |     863      |
| blk.{i}.ffn_gate_shexp.weight |      58 |     14.9    |     863      |
| blk.{i}.ffn_up_shexp.weight   |      58 |     14.9    |     863      |
| blk.{i}.attn_q_a.weight       |      59 |     11.2    |     658      |
| blk.{i}.ffn_gate_inp.weight   |      59 |     10.5    |     620      |
| blk.{i}.attn_kv_b.weight      |      59 |      8.5    |     502      |
| blk.{i}.attn_k_b.weight       |      59 |      4.25   |     251      |
| blk.{i}.attn_v_b.weight       |      59 |      4.25   |     251      |
| blk.{i}.attn_kv_a_mqa.weight  |      59 |      4.18   |     247      |
| blk.{i}.ffn_down.weight       |       1 |    134      |     134      |
| blk.{i}.ffn_gate.weight       |       1 |    134      |     134      |
| blk.{i}.ffn_up.weight         |       1 |    134      |     134      |
| blk.{i}.attn_norm.weight      |      60 |      0.0273 |       1.64   |
| blk.{i}.ffn_norm.weight       |      59 |      0.0273 |       1.61   |
| blk.{i}.attn_q_a_norm.weight  |      59 |      0.0059 |       0.346  |
| blk.{i}.attn_kv_a_norm.weight |      59 |      0.002  |       0.115  |
| blk.{i}.exp_probs_b.bias      |      59 |      0.0015 |       0.0864 |
| output_norm.weight            |       1 |      0.0273 |       0.0273 |

<details>
<summary>Script</summary>

`./llama-gguf Kimi-K2-Instruct-IQ1_S.gguf r n > kimi-tensors.txt`

`python tensors.py kimi-tensors.txt --md -m`

`tensors.py`:
```py
import re
import sys
import argparse
import pandas as pd

def format_size(size, unit):
    if not isinstance(size, (int, float)) or size < 0:
        return "Unknown"
    if unit == 'K': size /= 1024
    elif unit == 'M': size /= 1024 * 1024
    elif unit == 'G': size /= 1024 * 1024 * 1024
    if size == 0: return 0
    elif size < 0.1: return round(size, 4)
    elif size < 1: return round(size, 3)
    elif size < 10: return round(size, 2)
    elif size < 100: return round(size, 1)
    return round(size)

def create_tensor_report(file_contents, unit, md_output=False):
    tensor_regex = re.compile(r"tensor\[\d+\]: name = ([\w\.]+), offset = (\d+)")
    raw_tensors = [{"name": m.group(1), "offset": int(m.group(2))}
                   for line in file_contents.splitlines()
                   if (m := tensor_regex.search(line))]

    if not raw_tensors:
        print("Error: No tensors found.", file=sys.stderr)
        return

    raw_tensors.sort(key=lambda x: x["offset"])
    tensors = [{"name": t["name"], "size": raw_tensors[i+1]["offset"] - t["offset"]}
              for i, t in enumerate(raw_tensors[:-1])]
    tensors.append({"name": raw_tensors[-1]["name"], "size": -1})

    pattern_regex = re.compile(r"blk\.(\d+)\.")
    aggregated = {}
    for t in tensors[:-1]:
        p = pattern_regex.sub(r"blk.{i}.", t["name"])
        k = (p, t["size"])
        aggregated[k] = aggregated.get(k, 0) + 1

    last_p = pattern_regex.sub(r"blk.{i}.", tensors[-1]["name"])
    matched = False
    for p, s in aggregated:
        if p == last_p:
            aggregated[(p, s)] += 1
            matched = True
            break
    if not matched:
        aggregated[(last_p, -1)] = aggregated.get((last_p, -1), 0) + 1

    output = []
    for (p, s), c in aggregated.items():
        ts = c * s if s != -1 else -1
        fs = format_size(s, unit)
        ft = format_size(ts, unit)
        output.append([p, c, fs if isinstance(fs, str) else fs,
                      ft if isinstance(ft, str) else ft])

    output.sort(key=lambda x: x[3] if isinstance(x[3], (int, float)) else -1, reverse=True)
    units = {'B': 'Bytes', 'K': 'KB', 'M': 'MB', 'G': 'GB'}
    headers = ["Name", "Count", f"Size ({units.get(unit, 'Bytes')})",
              f"Total ({units.get(unit, 'Bytes')})"]

    if md_output:
        df = pd.DataFrame(output, columns=headers)
        print(df.to_markdown(index=False))
    else:
        print(*headers, sep=',')
        for row in output:
            print(*row, sep=',')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze GGUF tensor metadata.")
    parser.add_argument("filepath", help="Path to tensor metadata file.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-k", "--kb", action="store_true")
    group.add_argument("-m", "--mb", action="store_true")
    group.add_argument("-g", "--gb", action="store_true")
    parser.add_argument("--md", action="store_true", help="Output as markdown table")

    args = parser.parse_args()
    unit = 'B'
    if args.kb: unit = 'K'
    elif args.mb: unit = 'M'
    elif args.gb: unit = 'G'

    try:
        with open(args.filepath) as f:
            create_tensor_report(f.read(), unit, args.md)
    except FileNotFoundError:
        print(f"Error: File not found at '{args.filepath}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
```

</details>

Also, is there a way to get the tensor types from  llama-gguf? Or should I use something like gguf-py?

---

👤 **ubergarm** commented on **2025-07-20** at **15:18:58**

@ThomasBaruzier 

> everything minus ffn gate up down is very small

Yes, I like to imagine a person with the `attn/shexp/first N ffn dense layers` as their "head", and all the routed exps as their "body". DeepSeek has a very small "head" and a very large "body". Kimi-K2 has an even smaller tiny "head" and an even larger "body" haha... 

So perhaps one must be more careful when squishing that tiny "brain" lol... All metaphorical of course...

I would love to see a visualization of the relative sizes of say older llama vs deepseek vs kimi using visualization tool like https://github.com/ManimCommunity/manim/ ... too many things to do hah...

I'll test some more about that imatrix with `-mla 1` vs without `-mla` at all and get logs once `ssh` is back up for the remote rigs :crossed_fingers: 

> Also, is there a way to get the tensor types from llama-gguf? Or should I use something like gguf-py?

I didn't ever notice `build/bin/llama-gguf` even existed hah... Here is how I view gguf files similar to how @magikRUKKOLA is showing above:

```bash
cd ik_llama.cpp
# https://docs.astral.sh/uv/getting-started/installation/
uv venv ./venv --python 3.12 --python-preference=only-managed
source ./venv/bin/activate
uv pip install numpy==1.26.2 sentencepiece pyyaml  

python ./gguf-py/scripts/gguf_dump.py /models/mymodel.gguf
```

---

👤 **ubergarm** commented on **2025-07-20** at **16:08:14**

@ikawrakow 

> Could you please run the imatrix tool adding --verbosity 2 to your command line? There will be a lot of output to stdout with that, so redirect to a log file and post the log here. You only need to run 1 batch so we see the names of all tensors where data is being captured.

Just got access to the rig again after some storms cut short my cooking last night haha... Here are two command and logs for imatrix on Kimi-K2. One like I did with `-mla 1` and another omitting it. First full repeating layer chunk only.

<details>

<summary>👈 llama-imatrix -mla 1</summary>

```bash
model=/mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/Kimi-K2-Instruct-Q8_0.gguf

numactl --interleave=all \
./build/bin/llama-imatrix \
    -m "$model" \
    -f ubergarm-imatrix-calibration-corpus-v02.txt \
    -o /tmp/imatrix-test.dat \
    -mla 1 \
    --verbosity 2 \
    --ctx-size 512 \
    --layer-similarity \
    --numa distribute \
    --threads 384 \
    2>&1 | tee -a logs/imat-kimi-mla-1.log

llama_model_loader: loaded meta data with 42 key-value pairs and 1157 tensors from /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/Kimi-K2-Instruct-Q8_0.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Kimi K2 Instruct Bf16 Safetensors
llama_model_loader: - kv   3:                           general.finetune str              = Instruct-safetensors
llama_model_loader: - kv   4:                           general.basename str              = Kimi-K2
llama_model_loader: - kv   5:                         general.size_label str              = 384x15B
llama_model_loader: - kv   6:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv   7:                   deepseek2.context_length u32              = 131072
llama_model_loader: - kv   8:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv   9:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  10:             deepseek2.attention.head_count u32              = 64
llama_model_loader: - kv  11:          deepseek2.attention.head_count_kv u32              = 64
llama_model_loader: - kv  12:                   deepseek2.rope.freq_base f32              = 50000.000000
llama_model_loader: - kv  13: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  14:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  15:                          general.file_type u32              = 7
llama_model_loader: - kv  16:        deepseek2.leading_dense_block_count u32              = 1
llama_model_loader: - kv  17:                       deepseek2.vocab_size u32              = 163840
llama_model_loader: - kv  18:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  19:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  20:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  21:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  22:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  23:                     deepseek2.expert_count u32              = 384
llama_model_loader: - kv  24:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  25:             deepseek2.expert_weights_scale f32              = 2.827000
llama_model_loader: - kv  26:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  27:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  28:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  29:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  30:              deepseek2.rope.scaling.factor f32              = 32.000000
llama_model_loader: - kv  31: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  32: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  33:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  34:                         tokenizer.ggml.pre str              = kimi-k2
llama_model_loader: - kv  35:                      tokenizer.ggml.tokens arr[str,163840]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  36:                  tokenizer.ggml.token_type arr[i32,163840]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  37:                      tokenizer.ggml.merges arr[str,163328]  = ["Ġ Ġ", "ĠĠ ĠĠ", "Ġ t", "i n",...
llama_model_loader: - kv  38:                tokenizer.ggml.bos_token_id u32              = 163584
llama_model_loader: - kv  39:                tokenizer.ggml.eos_token_id u32              = 163585
llama_model_loader: - kv  40:                    tokenizer.chat_template str              = {% if tools -%}\n    {{ '<|im_system|>...
llama_model_loader: - kv  41:               general.quantization_version u32              = 2
llama_model_loader: - type  f32:  365 tensors
llama_model_loader: - type q8_0:  792 tensors
llm_load_vocab: special tokens cache size = 256
llm_load_vocab: token to piece cache size = 1.0607 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = deepseek2
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 163840
llm_load_print_meta: n_merges         = 163328
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 131072
llm_load_print_meta: n_embd           = 7168
llm_load_print_meta: n_layer          = 61
llm_load_print_meta: n_head           = 64
llm_load_print_meta: n_head_kv        = 64
llm_load_print_meta: n_rot            = 64
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 192
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: n_embd_k_gqa     = 12288
llm_load_print_meta: n_embd_v_gqa     = 8192
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 18432
llm_load_print_meta: n_expert         = 384
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = yarn
llm_load_print_meta: freq_base_train  = 50000.0
llm_load_print_meta: freq_scale_train = 0.03125
llm_load_print_meta: n_ctx_orig_yarn  = 4096
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = 671B
llm_load_print_meta: model ftype      = Q8_0
llm_load_print_meta: model params     = 1.027 T
llm_load_print_meta: model size       = 1016.623 GiB (8.504 BPW)
llm_load_print_meta: repeating layers = 1014.299 GiB (8.504 BPW, 1024.571 B parameters)
llm_load_print_meta: general.name     = Kimi K2 Instruct Bf16 Safetensors
llm_load_print_meta: BOS token        = 163584 '[BOS]'
llm_load_print_meta: EOS token        = 163585 '[EOS]'
llm_load_print_meta: LF token         = 128 'Ä'
llm_load_print_meta: EOT token        = 163586 '<|im_end|>'
llm_load_print_meta: max token length = 512
llm_load_print_meta: n_layer_dense_lead   = 1
llm_load_print_meta: n_lora_q             = 1536
llm_load_print_meta: n_lora_kv            = 512
llm_load_print_meta: n_ff_exp             = 2048
llm_load_print_meta: n_expert_shared      = 1
llm_load_print_meta: expert_weights_scale = 2.8
llm_load_print_meta: expert_weights_norm  = 1
llm_load_print_meta: expert_gating_func   = sigmoid
llm_load_print_meta: rope_yarn_log_mul    = 0.1000
llm_load_tensors: ggml ctx size =    0.47 MiB
llm_load_tensors:        CPU buffer size = 1041021.91 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 512
llama_new_context_with_model: n_batch    = 512
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: mla_attn   = 1
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 0
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 50000.0
llama_new_context_with_model: freq_scale = 0.03125
llama_kv_cache_init:        CPU KV buffer size =    64.81 MiB
llama_new_context_with_model: KV self size  =   64.81 MiB, c^KV (f16):   34.31 MiB, kv^T (f16):   30.50 MiB
llama_new_context_with_model:        CPU  output buffer size =     0.63 MiB
llama_new_context_with_model:        CPU compute buffer size =   334.00 MiB
llama_new_context_with_model: graph nodes  = 3827
llama_new_context_with_model: graph splits = 1

system_info: n_threads = 384 / 768 | AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 |
compute_imatrix: tokenizing the input ..
compute_imatrix: tokenization took 836.032 ms
compute_imatrix: computing over 826 chunks with batch_size 512
collect_imatrix[0]:       blk.0.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:            blk.0.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:            blk.0.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.0.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:         blk.0.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:            blk.0.ffn_gate.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:              blk.0.ffn_up.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:            blk.0.ffn_down.weight, MUL_MAT, 18432 x   512, 0
collect_imatrix[1]:       blk.1.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:            blk.1.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:            blk.1.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.1.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:         blk.1.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:        blk.1.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.1.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:         blk.1.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:       blk.1.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:      blk.1.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:        blk.1.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.1.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:       blk.2.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:            blk.2.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:            blk.2.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.2.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:         blk.2.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:        blk.2.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.2.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:         blk.2.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:       blk.2.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:      blk.2.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:        blk.2.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.2.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:       blk.3.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:            blk.3.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:            blk.3.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.3.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:         blk.3.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:        blk.3.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.3.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:         blk.3.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:       blk.3.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:      blk.3.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:        blk.3.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.3.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:       blk.4.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:            blk.4.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:            blk.4.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.4.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:         blk.4.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:        blk.4.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.4.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:         blk.4.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:       blk.4.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:      blk.4.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:        blk.4.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.4.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:       blk.5.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:            blk.5.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:            blk.5.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.5.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:         blk.5.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:        blk.5.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.5.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:         blk.5.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:       blk.5.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:      blk.5.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:        blk.5.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.5.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:       blk.6.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:            blk.6.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:            blk.6.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.6.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:         blk.6.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:        blk.6.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.6.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:         blk.6.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:       blk.6.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:      blk.6.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:        blk.6.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.6.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:       blk.7.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:            blk.7.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:            blk.7.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.7.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:         blk.7.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:        blk.7.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.7.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:         blk.7.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:       blk.7.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:      blk.7.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:        blk.7.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.7.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:       blk.8.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:            blk.8.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:            blk.8.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.8.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:         blk.8.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:        blk.8.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.8.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:         blk.8.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:       blk.8.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:      blk.8.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:        blk.8.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.8.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:       blk.9.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:            blk.9.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:            blk.9.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.9.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:         blk.9.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:        blk.9.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.9.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:         blk.9.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:       blk.9.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:      blk.9.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:        blk.9.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.9.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.10.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.10.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.10.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.10.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.10.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.10.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.10.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.10.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.10.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.10.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.10.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.10.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.11.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.11.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.11.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.11.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.11.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.11.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.11.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.11.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.11.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.11.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.11.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.11.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.12.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.12.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.12.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.12.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.12.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.12.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.12.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.12.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.12.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.12.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.12.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.12.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.13.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.13.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.13.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.13.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.13.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.13.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.13.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.13.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.13.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.13.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.13.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.13.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.14.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.14.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.14.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.14.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.14.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.14.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.14.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.14.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.14.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.14.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.14.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.14.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.15.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.15.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.15.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.15.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.15.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.15.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.15.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.15.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.15.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.15.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.15.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.15.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.16.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.16.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.16.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.16.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.16.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.16.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.16.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.16.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.16.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.16.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.16.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.16.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.17.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.17.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.17.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.17.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.17.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.17.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.17.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.17.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.17.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.17.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.17.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.17.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.18.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.18.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.18.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.18.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.18.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.18.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.18.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.18.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.18.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.18.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.18.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.18.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.19.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.19.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.19.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.19.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.19.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.19.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.19.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.19.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.19.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.19.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.19.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.19.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.20.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.20.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.20.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.20.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.20.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.20.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.20.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.20.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.20.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.20.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.20.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.20.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.21.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.21.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.21.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.21.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.21.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.21.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.21.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.21.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.21.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.21.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.21.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.21.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.22.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.22.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.22.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.22.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.22.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.22.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.22.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.22.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.22.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.22.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.22.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.22.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.23.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.23.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.23.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.23.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.23.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.23.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.23.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.23.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.23.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.23.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.23.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.23.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.24.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.24.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.24.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.24.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.24.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.24.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.24.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.24.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.24.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.24.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.24.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.24.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.25.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.25.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.25.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.25.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.25.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.25.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.25.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.25.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.25.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.25.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.25.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.25.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.26.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.26.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.26.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.26.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.26.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.26.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.26.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.26.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.26.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.26.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.26.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.26.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.27.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.27.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.27.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.27.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.27.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.27.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.27.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.27.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.27.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.27.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.27.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.27.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.28.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.28.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.28.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.28.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.28.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.28.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.28.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.28.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.28.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.28.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.28.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.28.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.29.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.29.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.29.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.29.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.29.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.29.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.29.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.29.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.29.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.29.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.29.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.29.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.30.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.30.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.30.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.30.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.30.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.30.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.30.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.30.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.30.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.30.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.30.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.30.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.31.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.31.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.31.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.31.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.31.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.31.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.31.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.31.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.31.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.31.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.31.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.31.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.32.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.32.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.32.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.32.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.32.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.32.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.32.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.32.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.32.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.32.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.32.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.32.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.33.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.33.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.33.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.33.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.33.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.33.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.33.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.33.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.33.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.33.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.33.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.33.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.34.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.34.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.34.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.34.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.34.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.34.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.34.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.34.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.34.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.34.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.34.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.34.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.35.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.35.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.35.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.35.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.35.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.35.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.35.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.35.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.35.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.35.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.35.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.35.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.36.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.36.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.36.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.36.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.36.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.36.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.36.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.36.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.36.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.36.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.36.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.36.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.37.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.37.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.37.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.37.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.37.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.37.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.37.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.37.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.37.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.37.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.37.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.37.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.38.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.38.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.38.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.38.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.38.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.38.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.38.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.38.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.38.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.38.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.38.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.38.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.39.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.39.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.39.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.39.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.39.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.39.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.39.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.39.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.39.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.39.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.39.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.39.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.40.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.40.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.40.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.40.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.40.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.40.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.40.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.40.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.40.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.40.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.40.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.40.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.41.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.41.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.41.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.41.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.41.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.41.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.41.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.41.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.41.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.41.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.41.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.41.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.42.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.42.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.42.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.42.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.42.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.42.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.42.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.42.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.42.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.42.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.42.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.42.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.43.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.43.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.43.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.43.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.43.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.43.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.43.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.43.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.43.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.43.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.43.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.43.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.44.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.44.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.44.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.44.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.44.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.44.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.44.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.44.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.44.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.44.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.44.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.44.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.45.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.45.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.45.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.45.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.45.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.45.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.45.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.45.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.45.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.45.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.45.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.45.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.46.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.46.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.46.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.46.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.46.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.46.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.46.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.46.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.46.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.46.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.46.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.46.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.47.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.47.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.47.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.47.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.47.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.47.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.47.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.47.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.47.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.47.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.47.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.47.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.48.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.48.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.48.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.48.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.48.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.48.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.48.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.48.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.48.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.48.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.48.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.48.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.49.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.49.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.49.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.49.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.49.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.49.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.49.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.49.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.49.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.49.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.49.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.49.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.50.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.50.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.50.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.50.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.50.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.50.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.50.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.50.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.50.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.50.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.50.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.50.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.51.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.51.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.51.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.51.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.51.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.51.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.51.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.51.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.51.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.51.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.51.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.51.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.52.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.52.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.52.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.52.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.52.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.52.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.52.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.52.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.52.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.52.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.52.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.52.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.53.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.53.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.53.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.53.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.53.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.53.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.53.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.53.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.53.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.53.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.53.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.53.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.54.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.54.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.54.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.54.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.54.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.54.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.54.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.54.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.54.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.54.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.54.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.54.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.55.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.55.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.55.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.55.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.55.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.55.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.55.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.55.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.55.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.55.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.55.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.55.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.56.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.56.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.56.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.56.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.56.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.56.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.56.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.56.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.56.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.56.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.56.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.56.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.57.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.57.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.57.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.57.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.57.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.57.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.57.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.57.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.57.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.57.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.57.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.57.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.58.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.58.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.58.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.58.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.58.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.58.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.58.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.58.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.58.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.58.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.58.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.58.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.59.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.59.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.59.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.59.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.59.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.59.ffn_gate_inp.weightcompute_imatrix: 190.09 seconds per pass - ETA 43 hours 36.88 minutes
, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.59.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.59.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.59.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.59.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.59.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.59.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.60.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.60.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.60.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]: blk.60.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.60.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.60.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.60.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.60.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.60.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.60.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.60.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.60.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:                    output.weight, MUL_MAT,  7168 x   512, 0
[1]75.3007,
```

</details>

<details>

<summary>👈 llama-imatrix (no mla) </summary>

```bash
model=/mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/Kimi-K2-Instruct-Q8_0.gguf

numactl --interleave=all \
./build/bin/llama-imatrix \
    -m "$model" \
    -f ubergarm-imatrix-calibration-corpus-v02.txt \
    -o /tmp/imatrix-test.dat \
    --verbosity 2 \
    --ctx-size 512 \
    --layer-similarity \
    --numa distribute \
    --threads 384 \
    2>&1 | tee -a logs/imat-kimi-no-mla.log

llama_model_loader: loaded meta data with 42 key-value pairs and 1157 tensors from /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/Kimi-K2-Instruct-Q8_0.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Kimi K2 Instruct Bf16 Safetensors
llama_model_loader: - kv   3:                           general.finetune str              = Instruct-safetensors
llama_model_loader: - kv   4:                           general.basename str              = Kimi-K2
llama_model_loader: - kv   5:                         general.size_label str              = 384x15B
llama_model_loader: - kv   6:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv   7:                   deepseek2.context_length u32              = 131072
llama_model_loader: - kv   8:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv   9:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  10:             deepseek2.attention.head_count u32              = 64
llama_model_loader: - kv  11:          deepseek2.attention.head_count_kv u32              = 64
llama_model_loader: - kv  12:                   deepseek2.rope.freq_base f32              = 50000.000000
llama_model_loader: - kv  13: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  14:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  15:                          general.file_type u32              = 7
llama_model_loader: - kv  16:        deepseek2.leading_dense_block_count u32              = 1
llama_model_loader: - kv  17:                       deepseek2.vocab_size u32              = 163840
llama_model_loader: - kv  18:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  19:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  20:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  21:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  22:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  23:                     deepseek2.expert_count u32              = 384
llama_model_loader: - kv  24:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  25:             deepseek2.expert_weights_scale f32              = 2.827000
llama_model_loader: - kv  26:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  27:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  28:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  29:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  30:              deepseek2.rope.scaling.factor f32              = 32.000000
llama_model_loader: - kv  31: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  32: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  33:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  34:                         tokenizer.ggml.pre str              = kimi-k2
llama_model_loader: - kv  35:                      tokenizer.ggml.tokens arr[str,163840]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  36:                  tokenizer.ggml.token_type arr[i32,163840]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  37:                      tokenizer.ggml.merges arr[str,163328]  = ["Ġ Ġ", "ĠĠ ĠĠ", "Ġ t", "i n",...
llama_model_loader: - kv  38:                tokenizer.ggml.bos_token_id u32              = 163584
llama_model_loader: - kv  39:                tokenizer.ggml.eos_token_id u32              = 163585
llama_model_loader: - kv  40:                    tokenizer.chat_template str              = {% if tools -%}\n    {{ '<|im_system|>...
llama_model_loader: - kv  41:               general.quantization_version u32              = 2
llama_model_loader: - type  f32:  365 tensors
llama_model_loader: - type q8_0:  792 tensors
llm_load_vocab: special tokens cache size = 256
llm_load_vocab: token to piece cache size = 1.0607 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = deepseek2
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 163840
llm_load_print_meta: n_merges         = 163328
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 131072
llm_load_print_meta: n_embd           = 7168
llm_load_print_meta: n_layer          = 61
llm_load_print_meta: n_head           = 64
llm_load_print_meta: n_head_kv        = 64
llm_load_print_meta: n_rot            = 64
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 192
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: n_embd_k_gqa     = 12288
llm_load_print_meta: n_embd_v_gqa     = 8192
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 18432
llm_load_print_meta: n_expert         = 384
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = yarn
llm_load_print_meta: freq_base_train  = 50000.0
llm_load_print_meta: freq_scale_train = 0.03125
llm_load_print_meta: n_ctx_orig_yarn  = 4096
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = 671B
llm_load_print_meta: model ftype      = Q8_0
llm_load_print_meta: model params     = 1.027 T
llm_load_print_meta: model size       = 1016.623 GiB (8.504 BPW)
llm_load_print_meta: repeating layers = 1014.299 GiB (8.504 BPW, 1024.571 B parameters)
llm_load_print_meta: general.name     = Kimi K2 Instruct Bf16 Safetensors
llm_load_print_meta: BOS token        = 163584 '[BOS]'
llm_load_print_meta: EOS token        = 163585 '[EOS]'
llm_load_print_meta: LF token         = 128 'Ä'
llm_load_print_meta: EOT token        = 163586 '<|im_end|>'
llm_load_print_meta: max token length = 512
llm_load_print_meta: n_layer_dense_lead   = 1
llm_load_print_meta: n_lora_q             = 1536
llm_load_print_meta: n_lora_kv            = 512
llm_load_print_meta: n_ff_exp             = 2048
llm_load_print_meta: n_expert_shared      = 1
llm_load_print_meta: expert_weights_scale = 2.8
llm_load_print_meta: expert_weights_norm  = 1
llm_load_print_meta: expert_gating_func   = sigmoid
llm_load_print_meta: rope_yarn_log_mul    = 0.1000
llm_load_tensors: ggml ctx size =    0.47 MiB
llm_load_tensors:        CPU buffer size = 1041021.91 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 512
llama_new_context_with_model: n_batch    = 512
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 0
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 50000.0
llama_new_context_with_model: freq_scale = 0.03125
llama_kv_cache_init:        CPU KV buffer size =  1220.00 MiB
llama_new_context_with_model: KV self size  = 1220.00 MiB, K (f16):  732.00 MiB, V (f16):  488.00 MiB
llama_new_context_with_model:        CPU  output buffer size =     0.63 MiB
llama_new_context_with_model:        CPU compute buffer size =   334.00 MiB
llama_new_context_with_model: graph nodes  = 3766
llama_new_context_with_model: graph splits = 1

system_info: n_threads = 384 / 768 | AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 |
compute_imatrix: tokenizing the input ..
compute_imatrix: tokenization took 840.818 ms
compute_imatrix: computing over 826 chunks with batch_size 512
collect_imatrix[0]:            blk.0.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:            blk.0.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:       blk.0.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.0.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:         blk.0.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:            blk.0.ffn_gate.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:              blk.0.ffn_up.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:            blk.0.ffn_down.weight, MUL_MAT, 18432 x   512, 0
collect_imatrix[1]:            blk.1.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:            blk.1.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:       blk.1.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.1.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:         blk.1.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:        blk.1.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.1.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:         blk.1.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:       blk.1.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:      blk.1.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:        blk.1.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.1.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:            blk.2.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:            blk.2.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:       blk.2.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.2.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:         blk.2.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:        blk.2.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.2.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:         blk.2.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:       blk.2.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:      blk.2.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:        blk.2.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.2.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:            blk.3.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:            blk.3.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:       blk.3.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.3.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:         blk.3.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:        blk.3.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.3.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:         blk.3.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:       blk.3.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:      blk.3.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:        blk.3.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.3.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:            blk.4.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:            blk.4.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:       blk.4.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.4.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:         blk.4.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:        blk.4.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.4.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:         blk.4.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:       blk.4.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:      blk.4.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:        blk.4.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.4.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:            blk.5.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:            blk.5.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:       blk.5.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.5.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:         blk.5.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:        blk.5.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.5.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:         blk.5.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:       blk.5.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:      blk.5.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:        blk.5.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.5.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:            blk.6.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:            blk.6.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:       blk.6.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.6.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:         blk.6.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:        blk.6.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.6.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:         blk.6.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:       blk.6.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:      blk.6.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:        blk.6.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.6.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:            blk.7.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:            blk.7.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:       blk.7.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.7.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:         blk.7.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:        blk.7.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.7.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:         blk.7.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:       blk.7.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:      blk.7.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:        blk.7.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.7.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:            blk.8.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:            blk.8.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:       blk.8.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.8.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:         blk.8.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:        blk.8.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.8.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:         blk.8.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:       blk.8.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:      blk.8.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:        blk.8.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.8.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:            blk.9.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:            blk.9.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:       blk.9.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.9.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:         blk.9.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:        blk.9.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.9.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:         blk.9.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:       blk.9.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:      blk.9.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:        blk.9.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.9.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.10.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.10.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.10.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.10.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.10.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.10.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.10.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.10.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.10.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.10.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.10.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.10.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.11.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.11.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.11.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.11.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.11.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.11.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.11.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.11.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.11.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.11.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.11.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.11.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.12.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.12.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.12.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.12.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.12.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.12.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.12.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.12.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.12.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.12.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.12.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.12.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.13.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.13.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.13.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.13.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.13.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.13.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.13.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.13.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.13.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.13.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.13.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.13.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.14.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.14.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.14.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.14.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.14.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.14.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.14.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.14.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.14.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.14.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.14.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.14.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.15.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.15.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.15.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.15.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.15.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.15.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.15.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.15.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.15.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.15.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.15.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.15.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.16.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.16.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.16.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.16.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.16.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.16.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.16.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.16.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.16.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.16.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.16.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.16.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.17.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.17.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.17.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.17.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.17.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.17.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.17.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.17.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.17.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.17.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.17.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.17.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.18.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.18.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.18.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.18.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.18.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.18.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.18.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.18.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.18.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.18.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.18.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.18.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.19.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.19.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.19.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.19.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.19.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.19.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.19.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.19.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.19.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.19.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.19.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.19.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.20.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.20.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.20.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.20.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.20.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.20.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.20.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.20.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.20.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.20.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.20.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.20.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.21.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.21.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.21.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.21.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.21.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.21.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.21.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.21.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.21.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.21.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.21.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.21.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.22.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.22.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.22.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.22.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.22.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.22.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.22.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.22.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.22.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.22.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.22.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.22.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.23.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.23.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.23.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.23.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.23.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.23.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.23.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.23.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.23.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.23.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.23.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.23.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.24.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.24.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.24.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.24.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.24.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.24.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.24.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.24.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.24.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.24.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.24.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.24.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.25.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.25.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.25.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.25.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.25.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.25.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.25.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.25.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.25.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.25.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.25.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.25.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.26.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.26.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.26.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.26.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.26.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.26.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.26.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.26.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.26.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.26.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.26.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.26.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.27.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.27.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.27.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.27.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.27.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.27.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.27.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.27.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.27.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.27.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.27.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.27.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.28.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.28.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.28.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.28.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.28.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.28.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.28.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.28.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.28.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.28.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.28.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.28.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.29.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.29.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.29.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.29.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.29.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.29.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.29.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.29.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.29.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.29.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.29.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.29.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.30.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.30.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.30.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.30.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.30.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.30.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.30.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.30.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.30.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.30.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.30.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.30.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.31.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.31.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.31.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.31.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.31.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.31.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.31.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.31.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.31.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.31.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.31.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.31.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.32.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.32.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.32.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.32.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.32.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.32.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.32.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.32.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.32.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.32.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.32.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.32.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.33.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.33.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.33.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.33.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.33.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.33.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.33.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.33.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.33.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.33.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.33.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.33.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.34.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.34.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.34.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.34.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.34.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.34.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.34.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.34.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.34.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.34.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.34.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.34.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.35.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.35.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.35.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.35.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.35.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.35.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.35.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.35.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.35.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.35.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.35.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.35.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.36.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.36.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.36.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.36.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.36.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.36.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.36.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.36.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.36.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.36.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.36.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.36.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.37.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.37.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.37.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.37.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.37.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.37.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.37.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.37.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.37.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.37.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.37.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.37.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.38.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.38.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.38.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.38.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.38.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.38.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.38.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.38.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.38.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.38.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.38.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.38.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.39.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.39.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.39.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.39.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.39.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.39.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.39.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.39.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.39.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.39.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.39.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.39.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.40.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.40.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.40.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.40.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.40.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.40.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.40.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.40.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.40.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.40.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.40.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.40.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.41.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.41.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.41.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.41.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.41.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.41.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.41.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.41.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.41.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.41.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.41.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.41.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.42.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.42.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.42.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.42.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.42.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.42.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.42.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.42.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.42.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.42.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.42.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.42.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.43.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.43.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.43.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.43.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.43.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.43.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.43.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.43.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.43.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.43.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.43.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.43.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.44.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.44.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.44.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.44.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.44.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.44.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.44.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.44.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.44.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.44.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.44.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.44.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.45.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.45.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.45.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.45.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.45.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.45.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.45.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.45.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.45.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.45.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.45.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.45.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.46.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.46.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.46.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.46.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.46.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.46.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.46.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.46.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.46.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.46.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.46.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.46.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.47.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.47.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.47.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.47.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.47.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.47.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.47.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.47.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.47.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.47.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.47.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.47.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.48.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.48.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.48.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.48.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.48.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.48.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.48.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.48.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.48.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.48.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.48.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.48.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.49.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.49.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.49.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.49.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.49.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.49.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.49.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.49.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.49.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.49.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.49.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.49.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.50.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.50.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.50.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.50.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.50.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.50.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.50.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.50.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.50.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.50.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.50.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.50.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.51.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.51.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.51.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.51.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.51.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.51.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.51.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.51.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.51.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.51.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.51.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.51.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.52.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.52.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.52.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.52.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.52.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.52.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.52.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.52.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.52.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.52.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.52.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.52.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.53.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.53.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.53.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.53.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.53.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.53.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.53.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.53.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.53.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.53.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.53.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.53.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.54.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.54.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.54.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.54.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.54.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.54.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.54.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.54.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.54.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.54.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.54.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.54.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.55.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.55.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.55.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.55.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.55.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.55.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.55.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.55.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.55.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.55.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.55.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.55.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.56.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.56.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.56.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.56.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.56.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.56.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.56.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.56.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.56.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.56.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.56.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.56.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.57.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.57.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.57.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.57.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.57.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.57.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.57.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.57.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.57.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.57.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.57.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.57.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.58.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.58.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.58.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.58.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.58.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.58.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.58.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.58.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.58.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.58.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.58.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.58.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.59.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.59.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.59.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.59.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.59.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.59.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:   compute_imatrix: 22.24 seconds per pass - ETA 5 hours 6.18 minutes
   blk.59.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.59.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.59.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.59.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.59.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.59.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:           blk.60.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:           blk.60.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[1]:      blk.60.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:          blk.60.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[1]:        blk.60.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[1]:       blk.60.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:      blk.60.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:        blk.60.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[1]:      blk.60.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:     blk.60.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:       blk.60.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[1]:     blk.60.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:                    output.weight, MUL_MAT,  7168 x   512, 0
[1]75.2142,collect_imatrix[1]:            blk.0.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[2]:            blk.0.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[2]:       blk.0.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[2]:           blk.0.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[2]:         blk.0.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[2]:            blk.0.ffn_gate.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[2]:              blk.0.ffn_up.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[2]:            blk.0.ffn_down.weight, MUL_MAT, 18432 x   512, 0
collect_imatrix[2]:            blk.1.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[2]:            blk.1.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[2]:       blk.1.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[2]:           blk.1.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[2]:         blk.1.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[2]:        blk.1.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[2]:       blk.1.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[2]:         blk.1.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[2]:       blk.1.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[2]:      blk.1.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[2]:        blk.1.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[2]:      blk.1.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[2]:            blk.2.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[2]:            blk.2.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[2]:       blk.2.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[2]:           blk.2.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[2]:         blk.2.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[2]:        blk.2.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[2]:       blk.2.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[2]:         blk.2.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[2]:       blk.2.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[2]:      blk.2.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[2]:        blk.2.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[2]:      blk.2.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[2]:            blk.3.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[2]:            blk.3.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[2]:       blk.3.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[2]:           blk.3.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[2]:         blk.3.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[2]:        blk.3.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[2]:       blk.3.ffn_gate_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[2]:         blk.3.ffn_up_exps.weight, MUL_MAT_ID,  7168 x   512, 0
collect_imatrix[2]:       blk.3.ffn_down_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[2]:      blk.3.ffn_gate_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[2]:        blk.3.ffn_up_shexp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[2]:      blk.3.ffn_down_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[2]:            blk.4.attn_q_a.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[2]:            blk.4.attn_q_b.weight, MUL_MAT,  1536 x   512, 0
collect_imatrix[2]:       blk.4.attn_kv_a_mqa.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[2]:           blk.4.attn_kv_b.weight, MUL_MAT,   512 x   512, 0
collect_imatrix[2]:         blk.4.attn_output.weight, MUL_MAT,  8192 x   512, 0
collect_imatrix[2]:        blk.4.ffn_gate_inp.weight, MUL_MAT,  7168 x   512, 0
collect_imatrix[2]:       blk.4.ffn_gate_exps.weight, MUL_MAT_ID,  71
```

</details>

---

👤 **ikawrakow** commented on **2025-07-20** at **16:18:21**

So, `attn_k_b` ends up having the name `attn_k_b.weight (reshaped)`. Somehow I thought I had taken care of that, but apparently not. We can still deal with it when using the imatrix. But `attn_v_b` is missing, which is very strange.

I'll look into it.

---

👤 **ThomasBaruzier** commented on **2025-07-20** at **16:59:15**

> Yes, I like to imagine a person with the attn/shexp/first N ffn dense layers as their "head", and all the routed exps as their "body". DeepSeek has a very small "head" and a very large "body". Kimi-K2 has an even smaller tiny "head" and an even larger "body" haha...

Funny analogy ahah. I guess we could try using some Q8_K_R8 for these tensors if one wanted pure cpu inference. I wonder how fast that would go. For cuda, I guess the best bet could be Q8_0 or Q6_K? Or maybe lower quants could be still fine if the PPL bump was due to missing tensor data in the imatrix?

> I didn't ever notice build/bin/llama-gguf even existed hah... Here is how I view gguf files similar to how @magikRUKKOLA is showing above

Thanks, I will check it out

---

👤 **ikawrakow** commented on **2025-07-20** at **17:23:02**

> I guess the best bet could be Q8_0 or Q6_K

`Q8_0` will be faster for PP, `Q6_K` for TG. As `Q6_K` is not the fastest quantization type on CUDA, you may want to try `Q6_0` - a highly overlooked quant - to get the best of both worlds.

---

👤 **ubergarm** commented on **2025-07-20** at **17:34:37**

> I wonder how fast that would go. 

I have some preliminary llama-sweep-bench with my original recipe Kimi-K2 quants on CPU only backend using the experimental AVX512 PR (on AMD Zen 5 CPU): https://github.com/ikawrakow/ik_llama.cpp/pull/612#issuecomment-3076539817

I plan to get at least one a/b test sweep-bench of my kimi-k2 v0.1 original recipe vs the v0.2 full q8_0 `attn/shexp/blk.0.ffn.*` on this same rig today and might release the updated quants if the speed hit is not too bad given the improvement Perplexity.

Of course I'll probably want to try a v0.3 recipe eventually after sorting out the MLA imatrix business :sweat_smile: ... Fortunately hf doesn't charge for the public storage :moneybag: :headstone: :hugs: ...

---

👤 **ikawrakow** commented on **2025-07-20** at **17:49:06**

Btw, what is causing this sudden surge in stars?

---

👤 **saood06** commented on **2025-07-22** at **14:50:08**

> Btw, what is causing this sudden surge in stars?

If I had to guess it was a mixture of Kimi, the people learning about this from the posts about Vulkan voting, organic growth.

(Also I'm glad the repo is back was going to reply to this before the incident).

---

👤 **ubergarm** commented on **2025-07-22** at **16:27:18**

Despite the outage, managed to release the worlds smallest Kimi-K2-Instruct as well as the best perplexity quants here: https://huggingface.co/ubergarm/Kimi-K2-Instruct-GGUF#quant-collection

Great job on the IQ1_KT, really impressive way to shrink down these behemoths models to run them in RAM on local rigs.

Welcome back!

---

👤 **ikawrakow** commented on **2025-07-22** at **17:30:41**

@ubergarm Looking at the graph in the linked HF repository, my guess is that one can cook models with size between `IQ1_KT` and `IQ2_KL` that push the Pareto frontier further away from Unsloth's quants using `IQ2_KT`. It looks like Unsloth's `IQ2_XXS` mix is performing surprisingly well in that size range.