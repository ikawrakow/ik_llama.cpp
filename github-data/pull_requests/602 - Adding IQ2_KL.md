## 🔀 [Pull Request #602](https://github.com/ikawrakow/ik_llama.cpp/pull/602) - Adding IQ2_KL

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/iq2_kl` |
| **Target Branch** | `main` |
| **Created** | 2025-07-12 |
| **Updated** | 2025-07-14 |
| **Merged** | 2025-07-14 |

---

## 📄 Description

### Motivation

* The gap between `IQ2_K/IQ2_S` (2.4375 bpw) and `IQ3_XXS` (3.0625 bpw) or `IQ3_KT` (3.125 bpw) is quite large. `Q2_K` (2.625 bpw), which should normally fill the gap, is a lower quality quantization type, so the gap remains unfilled. Hence, it would be useful to have a high quality quantization type that is about in the middle between `IQ2_K` and `IQ3_XXS`.
* Strangely enough, I had not realized until only quite recently that CUDA GEMM performance for quants with a block size of 16 is quite a bit lower than GEMM performance for blocks of 32. `IQ2_K, IQ2_S` and `Q2_K` all use blocks of 16, so there isn't a high CUDA PP performance quantization type in that bpw range. `IQ2_XXS, IQ2_KT` and `IQ2_KT` all have good CUDA PP performance, but they use `2.0625/2.125/2.1875` bpw, so are in a different quantization quality league as quantization errors increase very rapidly with decreasing bpw in that range.
* With models such as DeepSeek-V3/R1, Unsloth's `UD_Q2_K_XL` models have become very popular as for many people the resulting size is pretty much the maximum they can do with their hardware, while the quantization quality is closer to being really useful than smaller variants. Hence, a higher quality alternative to `Q2_K` with approximately the same bpw could become the goto quantization type for many users.   

Based on these observations and popular demand (hahaha, @Nexesenex was the only one asking for it), I decided to add `IQ2_KL`, a 2.6875 bpw quantization type with much better quality than the 2.625 bpw `Q2_K`.

 ### Some details

I wanted to have blocks of 32 for good CUDA PP performance (see above). Spending 5-6 bits per block scale leaves about 2.5 bpw for the quants if we want to be in the 2.6-2.7 bpw range, which disables the option of a direct int -> weight mapping. I did not want to use a full-fledged codebook as in the i-quans, as this kills CPU performance. But pairs of quants have 5 bits available, which corresponds to 32 distinct 2D points, which is still in the range that can be handled on the CPU via fast shuffle instructions (two `vqtbl2q_s8` instructions on NEON, 4 `_mm256_shuffle_epi8` instructions and two blends on `AVX2`). On CUDA this would need two lookups + shift/or to assemble a 32-but integer that can be used in `int8_t` dot products, so  also looking promising. So, then, 32 points in the 2D plane it is.

How do we get these 32 points? Here is what I did:
* Quantize a bunch of models using `IQ3_KS`, which uses 3 bits for the quants, so 6 bits per pair, so 64 distinct possibilities.
* Collect statistics $c_i$ about how often each of the 64 pairs (2D points) $x_i$ gets used (for this and the above, see changes in `examples/quantize-stats/quantize-stats.cpp`
* Pick 32 2D grid points $g_i$ such that

$$F = \sum c_i  d^2(x_i, G)$$

is minimized. Here, $d^2(x_i, G)$ is the minimum distance between the point $x_i$ and any point on the grid $G = \{ g_i \}$. Initially I wanted to have an elegant approach for finding the optimum solution, but at the end I just brute-forced it, so not publishing this code. The `IQ3_KS` values are non-uniformly distributed in `[-63, 47]`, and the resulting grid of 32 points looks quite interesting:

<img width="792" height="612" alt="u8" src="https://github.com/user-attachments/assets/b9f234a4-7185-4293-bd2f-9383cf0dab74" />

In this solution the locations of the grid points coincide with the `IQ3_KS` non-linear values. I did experiment with a grid where the points can take arbitrary `int8_t` values and this gives a lower value for $F$. However, when implemented in the quantization code, this alternative approach resulted in a higher quantization errors than what we get from the grid in the above figure, so I did not use that. My hand wavy explanation is that, when quantizing, we start with first finding an `IQ3_KS` solution, and then forcing the points not on the grid to a neighboring grid point, which kind of favors a grid where the grid points have the same co-ordinates as the `IQ3_KS` non-linear values.

### Quantization quality

I have done a fair share of experiments with this new quantization type with pretty good results, totally obliterating a similarly sized `Q2_K` quantization. But to not be told that "perplexity tells us nothing", I'm not adding these results here, and leaving it up to "quant cookers" to evaluate quantization quality in their favorite way. @Nexesenex, who apparently has been following the commits while I was working on the PR, has a comment [here](https://github.com/ikawrakow/ik_llama.cpp/commit/931bc412aef063037a6b2080f71dd844817176c8#commitcomment-161965520)

### Performance

I'll compare to `Q2_K`, the quantization type that `IQ2_KL` is looking to replace, and `IQ2_S`, an `i-quant` representative of slightly lower bpw. Using LlaMA-3.1-8B as an example with "pure" quantization (everything is `Q2_K/IQ2_KL` except for the output and token embedding tensors, which are `Q8_0`). The platforms used are
* `CUDA`: RTX-4080
* `Zen4`: Ryzen-7950X
* `AVX2`: Ryzen-5975WX
* `NEON`: M2-Max CPU
* `Metal`: M2-Max 30-core GPU

| Back-end  |  Type  | pp-512   |  tg-128 |
| ---: | ---: | ---: | ---: |
| CUDA  | IQ2_KL | 8483.36 | 164.04 |
|       | Q2_K   | 5819.40 | 169.76 |
|       | IQ2_S  | 6961.02 | 169.99 |
| Zen4  | IQ2_KL | 358.48  | 19.21  |
|       | Q2_K   | 352.16  | 19.62  |
|       | IQ2_S  | 357.00  | 19.23  |
| AVX2  | IQ2_KL | 310.85  | 16.79  |
|       | Q2_K   | 305.00  | 14.18  |
|       | IQ2_S  | 304.32  | 14.18  |
| NEON  | IQ2_KL | 161.97  | 26.80  |
|       | Q2_K   | 161.36  | 32.40  |
|       | IQ2_S  | 162.64  | 15.73  |
| Metal | IQ2_KL | 492.82  | 47.25  |
|       | Q2_K   | 511.45  | 58.36  |
|       | IQ2_S  | 471.22  | 37.62  |

---

## 💬 Conversation

👤 **Nexesenex** commented on **2025-07-12** at **13:44:15**

Thanks again, IK, for the quant and the explanations!

Already merged on my Croco, and it works like a charm on Cuda inference.

For the anecdote, I quantized a Miqu 70 b for my Mono-3090 back then, with mainline at the time :

llama_model_loader: - type  f32:  161 tensors
llama_model_loader: - type q5_K:   80 tensors (v)
llama_model_loader: - type q6_K:    1 tensors (out)
llama_model_loader: - type iq2_xxs:   80 tensors (q)
llama_model_loader: - type iq3_xxs:   80 tensors (a_o)
llama_model_loader: - type iq2_s:  320 tensors (ffns, k)
llama_model_loader: - type iq4_xs:    1 tensors (emb)
llm_load_print_meta: model size       = 20.711 GiB (2.579 BPW)
llm_load_print_meta: repeating layers = 20.381 GiB (2.558 BPW, 68.452 B parameters)
Final estimate: PPL = 4.3909 +/- 0.02255

And now, almost one year an a half later:
llama_model_loader: - type  f32:  161 tensors
llama_model_loader: - type q5_K:    1 tensors (emb)
llama_model_loader: - type iq5_k:    1 tensors (out)
llama_model_loader: - type iq4_ks:   80 tensors (v)
llama_model_loader: - type iq2_kt:   80 tensors (q)
llama_model_loader: - type iq3_kt:   80 tensors (k)
llama_model_loader: - type iq2_kl:  320 tensors (ffns, a_o)
llm_load_print_meta: model size       = 21.575 GiB (2.687 BPW)
llm_load_print_meta: repeating layers = 21.240 GiB (2.665 BPW, 68.452 B parameters)
Final estimate: PPL = 4.2293 +/- 0.02182

The recipe is a bit different, the size a bit higher, but Miqu's PPL being around 3.70 in q8_0, there's quite an overall jump in quality with all the work you did on IK_Llama, even if we account for recipe modulation, and even if you "deoptimized" some quants in respect for the legacy Llama models class to favor more tricky weights like L3 and the like.
I'm sure the ratio quality/weight can be even improved a bit more with some quant-strategy work, so I can have a 70b on 32k context with a good quality on a single 24GB GPU with the help of Quantized KV cache.

Anyway, IQ2_KL is SOTA imo, quality and speed-wise. Congratulations!

As for popular demand, the "people" might now wonder if the difference between IQ2_K/IQ2_S and IQ2_KL, for which you used you IQ3_KS, might be reproducible between IQ3_K/IQ3_S and an hypothetical IQ3_KL 3.6-3.8bpw, (with the help of IQ4_KS?). One might read with horror and contempt such an easy transposition, but now that the IQ2_S -> IQ3_KS gap has been quite well filled, remains the IQ3_K -> IQ4_KS gap (the IQ4_KSS that you so kindly developed after a popular request back then being more a side quant due to its complex packaging, in respect for a Cuda MMQ Kernel for example, from what I could understand).

The 3.5bpw quants have always been a bit tricky in my different tests, Q3_K now being obsolete, and IQ3_S / IQ3_K having somehow becoming subpar compared to the developments you made in the 4-4.5 bits and 2-2.75 bits range.

Btw, I listened to your intervention on Fosdem. It was nice to learn a bit about your background and to hear you, Iwan.

---

👤 **ikawrakow** commented on **2025-07-12** at **17:29:55**

> As for popular demand, the "people" might now wonder if the difference between IQ2_K/IQ2_S and IQ2_KL, for which you used you IQ3_KS, might be reproducible between IQ3_K/IQ3_S and an hypothetical IQ3_KL 3.6-3.8bpw, (with the help of IQ4_KS?).

Haha, I knew you will ask that. A similar approach does not work there because a pair of quants at 3.5 bpw is 7 bits, so 128 possibilities, so fast CPU shuffle instructions are not possible, and one would be back to slow lookup tables. Something else is needed for that gap.

To expand a bit more on that, a Trellis quant at 3.5 bpw (plus block scale bits) looks pretty promising. But the problem with Trellis quants is their lower CPU TG performance. Considering that most people seem to be using `ik_llama.cpp` for hybrid GPU/CPU inference with giant models such as DeepSeek, where TG runs on the CPU, this kind of sucks (and I have no sense of how people perceive the 10-20% TG performance gap).

---

👤 **Nexesenex** commented on **2025-07-12** at **18:56:45**

Well, I wondered if it would be that easy.. I'm so predictable indeed! ^^

As for a Trellis 3.5bpw, a 10% TG drop compared to what folks are using ain't too much of a big hassle, but 20% is really felt, that's for sure, especially in the single digits T/S. At least, that's my perception.
And as the context grows, the feeling grows also.

This being said, you bumped already the TG performances of Trellis on CPU, displacing the hard barrier towards the memory bandwidth. Sometimes we gain for free, sometimes we trade-off. And maybe you'll have another epiphany, says the profane!

Even without yet another TG bump for Trellis, considering the recent improvements about selecting the tensors you upload and those you don't for those using NVidia GPUs (on which Trellis is very competitive), also considering that most FTypes, especially those cooked by us enthusiasts, are not pure, the 20% drop might not be achieved often, because only some tensors and not other would be quantized in IQ3_KTL 3.5bpw.

Personally, I'd probably use an IQ3_KTL ggml_type for either the attn_k and attn_o, either the ffn_down, either the ffn_gate and up, either the attn_q, accordingly to the overall quant quality I'm searching for in respect for the size of the model and the context size desired.

IQ2_KT is a no brainer in its category, but IQ3_KS is quite competitive with IQ3_KT, and with a bigger delta bpw, IQ4_KS with IQ4_KT, including in quantization time! It's all about making a good mix between quality, size, and speed, not to speak about quantization time, between the available ggml_types to make an adequate FType.

As for the giant MOEs, they are an important niche in respect for all the work you accomplished on IKL, but the number of users able to run them is limited to well off enthusiasts and devs, academics with access to powerful workstations, and corpos/gov. And these giant models are most probably quite rarely ran on CPU only by those folks. ^^

That's my 2 cents.

---

👤 **saood06** commented on **2025-07-12** at **19:01:10**

>But to not be told that "perplexity tells us nothing", I'm not adding these results here, and leaving it up to "quant cookers" to evaluate quantization quality in their favorite way.

Is there any chance you could reconsider posting them? I think there is never going to be consensus on the best measure of quantization quality, because that differs based on user and use case, but the metrics you provided were useful for people to see roughly where quantization quality lies between quant types.

The open PR for the readme update has the additional benefit of making it easy to find and get to a PR where that is discussed as it is usually contained in the non row interleaved, CPU implementation of a quant, which is easy to get to from the table (as it is the first link in each column), and I do think it is quite useful for people who don't have strong opinions about PPL vs alternative metrics (which I believe is the majority).

I'm guessing you changed your mind around `IQ5_KS` as you never did post the details about that, and I meant to bring it up then but I didn't as I was hoping it was just delayed.

---

👤 **ubergarm** commented on **2025-07-12** at **19:31:47**

Great job with this `iq2_kl` and thanks for the explanation and details!

@saood06 

I had a bunch of old "pure" Qwen3-14B dense GGUFs sitting around from previous experiments so added a few of the new types including this sweet `iq2_kl` and its looking really good! ~Maybe I'll add a few more around that knee point like the `iq2_k` and `iq3_k`.~ *UPDATE*: Just added both.

<img width="1782" height="1181" alt="ik-sota-quants-iq2_kl" src="https://github.com/user-attachments/assets/f9c1ded4-5077-4396-82e9-a59bc58090a3" />

And yeah while the [`kt` trellis quants are a bit slower TG on CPU](https://github.com/ikawrakow/ik_llama.cpp/discussions/477#discussioncomment-13735744), they are quite handy for targeting certain tensors destined for GPU offload as @Nexesenex mentions.

Looking forward to incorporating this in some future mixes. I'll wait until its merged before releasing anything this time :skull: :joy:

---

👤 **ubergarm** commented on **2025-07-12** at **21:26:33**

Did some sweep benches fully offloaded on an older RTX A6000 GPU (not the new blackwell one). The new `iq2_kl` is looking like a nice blend speed for both PP and TG in this fully offloaded test.

<img width="4176" height="2217" alt="sweep-bench-pr602-iq2_kl" src="https://github.com/user-attachments/assets/1da58250-ce95-4ebe-93bb-73d174f4f735" />

---

👤 **Nexesenex** commented on **2025-07-12** at **21:34:13**

>      But to not be told that "perplexity tells us nothing", I'm not adding these results here, and leaving it up to "quant cookers" to evaluate quantization quality in their favorite way.
> 
> Is there any chance you could reconsider posting them? I think there is never going to be consensus on the best measure of quantization quality, because that differs based on user and use case, but the metrics you provided were useful for people for people to see roughly where quantization quality lies between quant types.

I forgot to comment that very line, @ikawrakow, and I second the request of @saood06.

Moreover, I do not understand the "perplexity tells us nothing" comment which has been made to you by -I don't know whom among my betters-. Non-withstanding the mathematical purity and/or sophistication of some other benchmark, Perplexity, aka. the "benchmark of the people", is a very clear indicator over the quality of the pretraining of a model, the damages made by its instruct training, and the quantization of the weights compared to their f16/bf16/f32 originals, and I could verify that on many models archs and finetunes/merges, both in use (including on long contexts) and with perplexity tests in several languages, English, French, but also Serbo-Croatian which is a minor one in pre-training.

The deltas in quantization (almost, to leave room for exceptions) always show in comparable (order of magnitude) proportions among different languages, even if they are not the same from one language to another, and so, both the baseline and the variation is relevant, if not the most relevant benchmark.

Being one of the bests in the field which pertains to IKL's developments, and the one who's pulling the work, I think that you can trust your own judgement over what is an adequate benchmark for your quants!
(Hell, I'm chatty today!)

@ubergarm : thanks for your clean benches!

---

👤 **ubergarm** commented on **2025-07-13** at **20:02:13**

I ran a few comparisons compiling CPU only on my local AMD 9950X 16x core gaming rig. Pretty cool to see TG coming within 95+% of theoretical max and even the IQ2_KT is coming within 80% of TG memory bandwidth limit as well.

<details>

<summary>👈 Details</summary>

```bash
# mlc
ALL Reads        :	85254.2 MB/sec

$ git checkout ik/iq2_kl
$ git rev-parse --short HEAD
f6d33e82
$ cmake -B build -DGGML_CUDA=0 -DGGML_VULKAN=0 -DGGML_BLAS=0
$ cmake --build build --config Release -j $(nproc)

./build/bin/llama-sweep-bench \
  --model "$model" \
  -fa \
  -c 4608 \
  --warmup-batch \
  --threads 16
```

# Q4_0 7.925 GiB (4.609 BPW)
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    1.907 |   268.42 |   12.233 |    10.46 |
|   512 |    128 |    512 |    1.982 |   258.26 |   12.458 |    10.27 |
|   512 |    128 |   1024 |    2.070 |   247.40 |   12.705 |    10.07 |
|   512 |    128 |   1536 |    2.129 |   240.45 |   12.948 |     9.89 |
|   512 |    128 |   2048 |    2.388 |   214.40 |   13.181 |     9.71 |
|   512 |    128 |   2560 |    2.478 |   206.65 |   14.166 |     9.04 |
|   512 |    128 |   3072 |    2.357 |   217.24 |   14.198 |     9.02 |
|   512 |    128 |   3584 |    2.433 |   210.40 |   13.938 |     9.18 |
|   512 |    128 |   4096 |    2.509 |   204.07 |   13.969 |     9.16 |

# IQ2_KL 5.141 GiB (2.990 BPW)
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    3.027 |   169.14 |    7.930 |    16.14 |
|   512 |    128 |    512 |    3.082 |   166.15 |    8.132 |    15.74 |
|   512 |    128 |   1024 |    3.139 |   163.10 |    8.394 |    15.25 |
|   512 |    128 |   1536 |    3.204 |   159.80 |    8.710 |    14.70 |
|   512 |    128 |   2048 |    3.292 |   155.53 |    8.961 |    14.28 |
|   512 |    128 |   2560 |    3.388 |   151.12 |    9.125 |    14.03 |
|   512 |    128 |   3072 |    3.440 |   148.85 |    9.364 |    13.67 |
|   512 |    128 |   3584 |    3.604 |   142.08 |    9.633 |    13.29 |
|   512 |    128 |   4096 |    3.589 |   142.64 |    9.933 |    12.89 |

# IQ2_KS 4.372 GiB (2.543 BPW)
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    2.969 |   172.48 |    6.724 |    19.04 |
|   512 |    128 |    512 |    3.031 |   168.94 |    6.918 |    18.50 |
|   512 |    128 |   1024 |    3.116 |   164.31 |    7.154 |    17.89 |
|   512 |    128 |   1536 |    3.184 |   160.80 |    7.420 |    17.25 |
|   512 |    128 |   2048 |    3.253 |   157.42 |    7.650 |    16.73 |
|   512 |    128 |   2560 |    3.408 |   150.23 |    7.962 |    16.08 |
|   512 |    128 |   3072 |    3.429 |   149.33 |    8.230 |    15.55 |
|   512 |    128 |   3584 |    3.498 |   146.38 |    8.474 |    15.11 |
|   512 |    128 |   4096 |    3.576 |   143.18 |    8.680 |    14.75 |

# IQ2_KT 4.280 GiB (2.489 BPW)
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    1.937 |   264.38 |    7.902 |    16.20 |
|   512 |    128 |    512 |    2.005 |   255.35 |    8.116 |    15.77 |
|   512 |    128 |   1024 |    2.076 |   246.59 |    8.322 |    15.38 |
|   512 |    128 |   1536 |    2.156 |   237.47 |    8.583 |    14.91 |
|   512 |    128 |   2048 |    2.246 |   227.98 |    8.743 |    14.64 |
|   512 |    128 |   2560 |    2.309 |   221.75 |    8.994 |    14.23 |
|   512 |    128 |   3072 |    2.381 |   215.07 |    9.213 |    13.89 |
|   512 |    128 |   3584 |    2.458 |   208.29 |    9.480 |    13.50 |
|   512 |    128 |   4096 |    2.529 |   202.48 |    9.723 |    13.16 |


</details>

<img width="4176" height="2329" alt="sweep-bench-pr602-cpu-only" src="https://github.com/user-attachments/assets/3f1bb727-65c5-4b4b-8c0e-09dda7b47baf" />

---

👤 **ikawrakow** commented on **2025-07-13** at **20:12:14**

It is strange that IQ2_KS/L have a lower PP performance. They are supposed to be ~20% faster than Q4_0

---

👤 **ubergarm** commented on **2025-07-13** at **20:50:43**

> It is strange that IQ2_KS/L have a lower PP performance. They are supposed to be ~20% faster than Q4_0

I was surprised when I saw the Q4_0 was faster on my Zen5 9950X. I just re-ran the benchmarks on a Thread Ripper Pro 24x core - same quants just using 24x cores now and more RAM bandwidth.

<img width="4176" height="2329" alt="sweep-bench-pr602-cpu-only-trpro" src="https://github.com/user-attachments/assets/ce33b47f-7ca1-4547-ad4b-29a4526b1e92" />


<details>

<summary>👈 Details</summary>

# Q4_0 7.925 GiB (4.609 BPW)
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    2.271 |   225.48 |    5.670 |    22.57 |
|   512 |    128 |    512 |    2.352 |   217.69 |    5.884 |    21.75 |
|   512 |    128 |   1024 |    2.432 |   210.50 |    6.019 |    21.27 |
|   512 |    128 |   1536 |    2.510 |   203.97 |    6.182 |    20.71 |
|   512 |    128 |   2048 |    2.591 |   197.63 |    6.359 |    20.13 |
|   512 |    128 |   2560 |    2.672 |   191.60 |    6.375 |    20.08 |
|   512 |    128 |   3072 |    2.759 |   185.54 |    6.727 |    19.03 |
|   512 |    128 |   3584 |    2.837 |   180.47 |    6.911 |    18.52 |
|   512 |    128 |   4096 |    2.918 |   175.49 |    6.895 |    18.57 |

# IQ2_KL 5.141 GiB (2.990 BPW)
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    1.657 |   308.97 |    3.845 |    33.29 |
|   512 |    128 |    512 |    1.737 |   294.69 |    4.047 |    31.63 |
|   512 |    128 |   1024 |    1.819 |   281.51 |    4.158 |    30.79 |
|   512 |    128 |   1536 |    1.901 |   269.27 |    4.335 |    29.53 |
|   512 |    128 |   2048 |    1.987 |   257.61 |    4.559 |    28.08 |
|   512 |    128 |   2560 |    2.065 |   247.92 |    4.547 |    28.15 |
|   512 |    128 |   3072 |    2.149 |   238.24 |    4.899 |    26.13 |
|   512 |    128 |   3584 |    2.232 |   229.41 |    5.120 |    25.00 |
|   512 |    128 |   4096 |    2.314 |   221.23 |    5.034 |    25.43 |

# IQ2_KS 4.372 GiB (2.543 BPW)
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    1.650 |   310.39 |    3.387 |    37.80 |
|   512 |    128 |    512 |    1.727 |   296.47 |    3.556 |    35.99 |
|   512 |    128 |   1024 |    1.807 |   283.29 |    3.703 |    34.56 |
|   512 |    128 |   1536 |    1.889 |   271.09 |    3.860 |    33.16 |
|   512 |    128 |   2048 |    1.975 |   259.26 |    4.045 |    31.64 |
|   512 |    128 |   2560 |    2.054 |   249.24 |    4.070 |    31.45 |
|   512 |    128 |   3072 |    2.137 |   239.56 |    4.385 |    29.19 |
|   512 |    128 |   3584 |    2.221 |   230.52 |    4.595 |    27.86 |
|   512 |    128 |   4096 |    2.304 |   222.18 |    4.522 |    28.31 |

# IQ2_KT 4.280 GiB (2.489 BPW)
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    2.276 |   224.94 |    6.348 |    20.16 |
|   512 |    128 |    512 |    2.355 |   217.37 |    6.527 |    19.61 |
|   512 |    128 |   1024 |    2.433 |   210.48 |    6.618 |    19.34 |
|   512 |    128 |   1536 |    2.512 |   203.81 |    6.808 |    18.80 |
|   512 |    128 |   2048 |    2.592 |   197.54 |    6.997 |    18.29 |
|   512 |    128 |   2560 |    2.673 |   191.55 |    7.000 |    18.29 |
|   512 |    128 |   3072 |    2.754 |   185.94 |    7.315 |    17.50 |
|   512 |    128 |   3584 |    2.835 |   180.58 |    7.623 |    16.79 |
|   512 |    128 |   4096 |    2.919 |   175.42 |    7.451 |    17.18 |

</details>

fwiw here are the cpu flags on both rigs:

```
# AMD Ryzen 9 9950X 16-Core Processor
fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl xtopology nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx_vnni avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold v_vmsave_vmload vgif v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq rdpid bus_lock_detect movdiri movdir64b overflow_recov succor smca fsrm avx512_vp2intersect flush_l1d amd_lbr_pmc_freeze

# AMD Ryzen Threadripper PRO 7965WX 24-Cores
fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl xtopology nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
```

The 9950x has `avx_vnni` whereas this Thread Ripper Pro does not fwiw.

---

👤 **ikawrakow** commented on **2025-07-14** at **05:23:05**

For PP `IQ2_KL` and `IQ2_KS` get repacked to `Q8_K_R8`, while `Q4_0` and `IQ2_KT` get repacked to `Q8_0_R8`. When the CPU has `avx512f, avx512vnni, avx512vl, avx512bw, avx512dq`, which both have, on my CPU (Ryzen-7950X, which has the same CPU flags as the 7965WX) GEMM with `Q8_K_R8` is ~20% faster than with `Q8_0_R8`. This would mean that somehow this is not true on the 9950X. Can we confirm that by using `Q8_K_R8` directly?
```
./bin/llama-quantize --token-embedding-type q8_0 $model $q8k_model q8_k_r8
./bin/llama-sweep-bench -m $q8k_model -c 4608 -t 16 -fa
```

Apart from this, yes, up to 100 GB/s or so memory bandwidth is fully saturated for TG. It still looks quite OK on the 795WX, where we are getting 160-180 GB/s. But beyond 200 GB/s something happens, and we cannot get anywhere close to the theoretical limit for the 400+ GB/s systems.

---

👤 **ubergarm** commented on **2025-07-14** at **12:19:28**

> Can we confirm that by using Q8_K_R8 directly?

Interesting, yes, let's measure: I made two test quants: `Q8_K_R8` and a `Q8_0_R8`, and ran them on both rigs. Tested both built cpu-only on `main@255c2204`. Also, just to be safe, I confirmed that the output looks clean on a 1shot chat with both quants on the 9950x.

<img width="4176" height="2329" alt="sweep-bench-activations-speed-testing" src="https://github.com/user-attachments/assets/a3be622c-58ae-46d0-a349-f65b939f47bd" />

<details>

<summary>👈 Details</summary>

```bash
# Qwen3-14B-Q8_K_R8.gguf
llama_model_loader: - type  f32:  161 tensors
llama_model_loader: - type q8_0:    1 tensors
llama_model_loader: - type q8_k_r8:  281 tensors

# Qwen3-14B-Q8_0_R8.gguf
llama_model_loader: - type  f32:  161 tensors
llama_model_loader: - type q8_0:    1 tensors
llama_model_loader: - type q8_0_r8:  281 tensors
```

# Q8_K_R8 7965WX 24x Core
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    1.632 |   313.78 |    9.270 |    13.81 |
|   512 |    128 |    512 |    1.713 |   298.91 |    9.397 |    13.62 |
|   512 |    128 |   1024 |    1.795 |   285.16 |    9.537 |    13.42 |
|   512 |    128 |   1536 |    1.881 |   272.13 |    9.651 |    13.26 |
|   512 |    128 |   2048 |    1.968 |   260.16 |    9.805 |    13.05 |
|   512 |    128 |   2560 |    2.052 |   249.47 |    9.844 |    13.00 |
|   512 |    128 |   3072 |    2.120 |   241.50 |   10.073 |    12.71 |
|   512 |    128 |   3584 |    2.210 |   231.72 |   10.260 |    12.48 |
|   512 |    128 |   4096 |    2.295 |   223.10 |   10.189 |    12.56 |

# Q8_K_R8 9950X 16x Core
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    2.855 |   179.32 |   21.399 |     5.98 |
|   512 |    128 |    512 |    2.945 |   173.84 |   21.614 |     5.92 |
|   512 |    128 |   1024 |    3.014 |   169.89 |   21.848 |     5.86 |
|   512 |    128 |   1536 |    3.093 |   165.54 |   22.090 |     5.79 |
|   512 |    128 |   2048 |    3.154 |   162.35 |   22.326 |     5.73 |
|   512 |    128 |   2560 |    3.228 |   158.60 |   22.562 |     5.67 |
|   512 |    128 |   3072 |    3.286 |   155.81 |   22.774 |     5.62 |
|   512 |    128 |   3584 |    3.378 |   151.57 |   23.050 |     5.55 |
|   512 |    128 |   4096 |    3.479 |   147.16 |   23.308 |     5.49 |

# Q8_0_R8 7965WX 24x Core
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    2.268 |   225.73 |   10.102 |    12.67 |
|   512 |    128 |    512 |    2.354 |   217.53 |   10.244 |    12.50 |
|   512 |    128 |   1024 |    2.435 |   210.29 |   10.413 |    12.29 |
|   512 |    128 |   1536 |    2.516 |   203.49 |   10.533 |    12.15 |
|   512 |    128 |   2048 |    2.601 |   196.82 |   10.679 |    11.99 |
|   512 |    128 |   2560 |    2.681 |   190.96 |   10.712 |    11.95 |
|   512 |    128 |   3072 |    2.763 |   185.32 |   10.986 |    11.65 |
|   512 |    128 |   3584 |    2.846 |   179.89 |   11.103 |    11.53 |
|   512 |    128 |   4096 |    2.930 |   174.73 |   11.063 |    11.57 |

# Q8_0_R8 9950X 16x Core
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    1.942 |   263.70 |   22.455 |     5.70 |
|   512 |    128 |    512 |    2.004 |   255.44 |   22.636 |     5.65 |
|   512 |    128 |   1024 |    2.071 |   247.28 |   22.860 |     5.60 |
|   512 |    128 |   1536 |    2.149 |   238.20 |   23.089 |     5.54 |
|   512 |    128 |   2048 |    2.221 |   230.57 |   23.395 |     5.47 |
|   512 |    128 |   2560 |    2.293 |   223.29 |   23.598 |     5.42 |
|   512 |    128 |   3072 |    2.375 |   215.58 |   23.796 |     5.38 |
|   512 |    128 |   3584 |    2.451 |   208.93 |   24.067 |     5.32 |
|   512 |    128 |   4096 |    2.523 |   202.96 |   24.249 |     5.28 |

</details>

---

👤 **ikawrakow** commented on **2025-07-14** at **12:36:48**

Wow, that's a bummer. Does a different compiler get used on the 9950X?

If you have time to experiment: can you comment out line 2675 in `iqk_gemm_kquants.cpp` (the line
```
    func16 = mul_mat_q8_k_r8_q8_k<16>;
```
 rebuild, and rerun the `Q8_K_R8` sweep-bench? Thanks.

---

👤 **ubergarm** commented on **2025-07-14** at **13:05:08**

Yeah the 9950x is bleeding edge ARCH box

```bash
$ lscpu | grep name
Model name:                              AMD Ryzen 9 9950X 16-Core Processor
$ ./build/bin/llama-sweep-bench --version
version: 3798 (255c2204)
built with cc (GCC) 15.1.1 20250425 for x86_64-pc-linux-gnu

$ lscpu | grep name
Model name:                           AMD Ryzen Threadripper PRO 7965WX 24-Cores
$ ./build/bin/llama-sweep-bench --version
version: 3798 (255c2204)
built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu
```

Got it, commenting this out on the 9950x and trying again adding one new plot to the above graph.
```
vi ./ggml/src/iqk/iqk_gemm_kquants.cpp
# line 2675
// #ifdef HAVE_FANCY_SIMD
//             func16 = mul_mat_q8_k_r8_q8_k<16>;
// #endif
```

So it is now much faster, albeit still a bit below the Q8_0_R8.

<img width="4176" height="2217" alt="sweep-bench-activations-speed-testing-fancy-simd" src="https://github.com/user-attachments/assets/5fe121c7-3adf-4d25-91d0-06babb43bd52" />

<details>

<summary>👈 Details</summary>

# Q8_K_R8 9950X 16x Core Comment Out mul_mat_q8_k_r8_q8_k<16>;
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    2.068 |   247.61 |   21.441 |     5.97 |
|   512 |    128 |    512 |    2.143 |   238.96 |   21.669 |     5.91 |
|   512 |    128 |   1024 |    2.227 |   229.95 |   21.925 |     5.84 |
|   512 |    128 |   1536 |    2.310 |   221.60 |   22.199 |     5.77 |
|   512 |    128 |   2048 |    2.392 |   214.01 |   22.458 |     5.70 |
|   512 |    128 |   2560 |    2.474 |   206.97 |   22.700 |     5.64 |
|   512 |    128 |   3072 |    2.552 |   200.64 |   22.971 |     5.57 |
|   512 |    128 |   3584 |    2.622 |   195.29 |   23.265 |     5.50 |
|   512 |    128 |   4096 |    2.700 |   189.66 |   23.514 |     5.44 |


</details>

---

👤 **ikawrakow** commented on **2025-07-14** at **13:27:58**

OK, this is much better, but not sure what to do with it. On my 7950X commenting out line 2675 leads to ~5% lower performance, I suspect this should be similar on your 7965WX. 

My best guess is that the compiler is misbehaving. The `Q8_K_R8` GEMM kernel uses more vector registers than available. This tends to give better performance than not using all vector registers (and it is very hard/next to impossible to setup the algorithm in a way that exactly 32 vector registers are used). But it seem GCC 15 absolutely dislikes this. Either this (very likely), or somehow the 9950X handles very badly register spillage (unlikely). 

The `Q8_0_R8` GEMM kernel uses 512-bit instructions on `AVX512`, the `Q8_K_R8` does not. The issue with 512-bit on my 7950X is that the 512-bit instructions get executed as two 256-bit instructions. As it does take extra effort to prepare the data in 512-bit registers, using 512-bit instructions on the Zen4 core is often slower. The Zen5 cores that the 9950X uses are the first AMD cores to have real 512-bit instructions. I guess, that's why the `Q8_0_R8` kernel is faster on the 9950X.

---

👤 **ubergarm** commented on **2025-07-14** at **15:38:46**

> it seem GCC 15 absolutely dislikes this

When I get a chance I'll try to compile with an older version and test again.

And yes this 9950X is one of the first to get that new 512-bit instruction, kinda cool to see it making a noticeable difference.

---

👤 **ikawrakow** commented on **2025-07-14** at **16:55:03**

OK, I think I'll merge this the way it is. I did try a few things but nothing resulted in an improvement (PPL and/or performance), so this is what it will be.