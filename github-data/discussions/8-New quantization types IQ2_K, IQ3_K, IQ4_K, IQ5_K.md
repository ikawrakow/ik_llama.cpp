### 🗣️ [#8](https://github.com/ikawrakow/ik_llama.cpp/discussions/8) - New quantization types IQ2_K, IQ3_K, IQ4_K, IQ5_K

| **Author** | `ikawrakow` |
| :--- | :--- |
| **Created** | 2024-08-01 |
| **Updated** | 2025-07-04 |

---

#### Description

## Why?

I can hear what some are thinking: "Are you crazy? Even more quantization types? Doesn't `llama.cpp` already have enough?"

That was what I was thinking too. Until LLaMA-3 came along, that is.

Quantization errors for LLaMA-3 models are much higher than they have been for all previous models I have experimented with. This is best illustrated with the graph below. LLaMA-3.1 is all the rage these days, but I don't have the ability to run LLaMA-3.1-405B, so I have settled for LLaMA-3.1-70B to generate the graph. We will measure quantization error `QError` of a quantization `Q` using perplexity `PPL` as
```
 QError =  PPL(Q)/PPL(fp16) - 1
```
As we are not evaluating model performance in language tasks, but are only interested in the performance of a quantized model compared to **the same** full precision model, there is no benefit from looking at commonly used language modeling / reasoning benchmarks, which a) are typically less sensitive to quantization errors than PPL and b) take much longer to evaluate.   
One could also use KL divergence, but KL divergence and `PPL` are closely related, and `PPL` is more convenient to calculate with `llama.cpp`, so `PPL` it is. 

 
![l31_70B](https://github.com/user-attachments/assets/e1e8e2ba-1e61-4913-9e86-bc682b227e25)

Blue symbols represent legacy quants (`Q4_0, Q4_1, Q5_0, Q5_1`), red symbols show results for k-quants, i-quants are depicted in black. To show how much higher the quantization error of LLaMA-3.1-70B is, I have included results for LLaMA-v2-70B shown in brown (just for k-quants as I have somehow lost the i-quants runs and did not feel like re-running the quite lengthy calculations). We see that there is basically about 1 bit-per-weight (bpw) gap between LLaMA-v2-70B and LLaMA-3.1-70B. I.e., it looks like the additional tokens used for training LLaMA-3 have paid off, the model has "learned" more from the data, and the model parameters in LLaMA-3.1 contain about 1 bpw extra information. This then results in a higher quantization error for a given bpw quantization budget.

We can now discuss the new quants shown with cyan circles. Please note that the y-axis is logarithmic so that the differences between the data points are quite large, even if they look fairly close to each other. For instance, the blue point around 5.5 bpw (`Q5_0`), which looks quite close to the red point (`Q5_K_S`), has a quantization error of 2.9% vs 1.9%. The cyan point around 5.5 bpw is `IQ5_K`, with a quantization error of 1.4%, i.e., `IQ5_K` has a quantization error that is 2.1X lower compared to `Q5_0`, and 40% lower compared to `Q5_K_S`. The cyan point around 4.5 bpw (`IQ4_K`) has a 2.7X lower quantization error compared to `Q4_0`, and 40% lower compared to `Q4_K_S`. So, even though `IQ4_K` and `IQ5_K` don't come anywhere close to what we used to have for 4- and 5-bit quantization in the pre-LLaMA-3.1 days, they do give a nice improvement compared to the SOTA in the 4+ bpw range.

"But what about the cyan points around 3.5 and 2.4 bpw? They are basically the same as i-quants!" - I hear you asking. These two exist for two reasons:
* My curiosity
* Much better inference performance compared to i-quants on the CPU and old GPU's. 

### Curiosity

i-quants are much better than k-quants in the sub-4-bpw range. i-quants in the sub-4-bpw range all use "codebooks" that encode groups of 8 or 4 model weights on the E8 or D4 lattice. The "codebook" idea comes originally from QuIP# and is also being used in, e.g., AQLM. I have been curious for some time to what extent the use of a "codebook" contributes to the better quantization quality of i-quants compared to k-quants. The "codebook" certainly acts as a kind of regularization to avoid/reduce overfitting: one only has a subset of all possible lattice points available in the "codebook" to represent a group of model weights, and hence the quantization algorithm cannot focus too much on individual quants, possibly missing more important model weights in the process. But is there more to it than just it being a regularization technique? I was curious and, as we can see in the above graph, it is indeed possible to match i-quants quantization accuracy with a non-linear quantization technique.  

### Performance

The use of a "codebook" requires a lookup in a fairly large table to convert the "codebook" index (which is stored in the quantized model) to actual quantized model weights when performing matrix multiplications. The lookup is handled quite OK by modern GPU's, but leads to a massive performance penalty on CPU's (and, from what I gather from `llama.cpp` user comments, also on older GPU's). The new `IQK` quants use a non-linear mapping between the quantized value stored in the model data (`0...15` for 4-bit quantization, `0...7` for 3-bit, etc.) and the actual model weight, which also needs a lookup table. But these lookup tables are much smaller (4, 8, 16, 32 `INT8` values for 2-, 3-, 4-, 5-bit quantization), so they fit into 1 or 2 SIMD registers, and thus can be handled very efficiently with SIMD instructions (`_mm256_shuffle_epi8` on `AVX2`, `vqtbl1q_s8` on `ARM_NEON`), resulting in a performance that is (nearly) the same as corresponding linear mapping between quants and model weights.

Let's look how this translates into observed inference performance. We compare `IQ2_K` to the matching `IQ2_XS`, and `IQ3_K` to the matching `IQ3_S` quants (matching in the sense that they use basically the same bpw and have very similar quantization accuracy).  The following table shows performance in tokens per second (t/s) for prompt processing (`pp512`, so a prompt of 512 tokens) and token generation (`tg128`, so generating 128 tokens one-by-one) between matching quants on `AVX2` (Ryzen-7950X) and `ARM_NEON` (M2-Max CPU). I have also added mainline `llama.cpp` results. The two values in the `Speedup` column are the `t/s` ratios between the new `IQK` quants and the corresponding i-quant in `llama.cpp` and in this repository. For instance, if we look at `IQ3_S` on the Ryzen-7950X, we see that `IQ3_K` will perform prompt processing 6.45 times faster than `llama.cpp`, and token generation speed will be 2.37X!   

| Case           | test  | threads | t/s llama.cpp | t/s this repo |   t/s iqk     |  Speedup    |   
| -------------- | ----- | ------: | ------------: | ------------: | ------------: | ----------: |
| 8B IQ2_XS AVX2 | pp512 |   16    |  46.45 ± 0.27 | 125.46 ± 0.43 | 194.64 ± 0.66 | 4.19 / 1.55 |
|                | tg128 |    4    |  10.88 ± 0.09 |  12.07 ± 0.07 |  21.46 ± 0.03 | 1.97 / 1.78 |
| 8B IQ3_S  AVX2 | pp512 |   16    |  28.04 ± 0.08 |  96.28 ± 0.45 | 180.77 ± 0.62 | 6.45 / 1.88 |
|                | tg128 |    4    |   6.80 ± 0.01 |  7.62 ± 0.10  |  16.10 ± 0.16 | 2.37 / 2.11 |
| 7B IQ2_XS NEON | pp512 |    8    |  22.77 ± 0.21 |  51.15 ± 0.24 |  60.60 ± 0.97 | 2.66 / 1.18 |
|                | tg128 |    8    |  18.19 ± 1.30 |  20.94 ± 0.19 |  28.24 ± 0.39 | 1.55 / 1.35 |
| 7B IQ3_S  NEON | pp512 |    8    |  12.08 ± 0.30 |  49.72 ± 0.06 |  55.65 ± 0.82 | 4.61 / 1.12 |
|                | tg128 |    8    |  10.32 ± 0.25 |  11.11 ± 0.37 |  20.33 ± 0.06 | 1.97 / 1.83 |

## What are non-linear quants anyway?

Will add later.

## IQ6_K?

Before LLaMA-3, `Q6_K` quantization always had a quantization error in the 0.1-0.15% range, i.e., it was basically as good as the full precision model. But for LLaMA-3.1-70B `Q6_K` quantization error is 0.65%! `Q8_0` does match the full precision model, but it uses 2 extra bpw. I have experimented with 6-bit non-linear quantization in the past, but `Q6_K` quantization error was so low that it was basically not possible to a see a benefit from the non-linearity. Given the much higher `Q6_K` quantization error for LLaMA-3 models, it may be worthwhile to resurrect 6-bit non-linear quantization.

**Update** See PR #14

---

#### 🗣️ Discussion

👤 **afsara-ben** replied the **2025-06-13** at **17:55:20**:<br>

@ikawrakow just found out your fork, wanted to clear my idea - K quants are block based and IQ quants are also block based in llama.cpp with a codebook. The IQn_K quants here is the same as IQ quants but with a non-linear mapping between the quantized weight and actual weight. Maybe its somewhere in the code but can you elaborate what the non-linear function is? And even if the lookup table is small (4x4grid instead of 256x256), the time to access it from L1 cache will still be the same because of memory bandwidth right?

> 👤 **ikawrakow** replied the **2025-06-13** at **18:56:53**:<br>
> Sub 4-bit i-quants use codebooks. `IQ4_XS` and `IQ4_NL`, which were added along with the codebook i-quants `IQ2_XXS, IQ2_S, IQ2_S, IQ3_XXS, IQ3_S` do not use a codebook, but a non-linear mapping for individual quants. They are both 4-bit, so the lookup table has just 16 entries, and the lookup adds negligible overhead.
> 
> The `IQX_K` quants also don't use a codebook. If fact, one of the main motivations to create them was to prove to myself that there is nothing special about codebooks. The main difference between `IQX_K` quants and `IQ4_XS/IQ4_NL` is in the use of an extra bit that selects between two lookup tables. `IQ4_KS`, which uses the exact same amount of bits per model weight as `IQ4_XS` (4.25) arrives at a lower quantization error than `IQ4_XS` that way. There are now the following `IQX_K` quants
> * `IQ2_KS` - blocks of 32 weights with a per tensor row scale. Lookup table is 2x4 entries, 2.1875 bpw
> * `IQ2_K` - blocks of 16 weights in super-blocks of 256. Lookup table is 2x4 entries, 2.375 bpw
> * `IQ3_K` - blocks of 16 weights in super-blocks of 256. Lookup table is 2x8 entries, 3.4375 bpw
> * `IQ4_KS` - blocks of 32 weights with a per tensor row scale. Lookup table is 2x16 entries, 4.25 bpw
> * `IQ4_K` - blocks of 16 weights in super-blocks of 256. Lookup table is 2x16 entries, 4.5 bpw
> * `IQ5_KS` - blocks of 32 weights with a per tensor row scale. Lookup table is 2x32 entries, 5.25 bpw
> * `IQ5_K` - blocks of 16 weights in super-blocks of 256. Lookup table is 2x32 entries, 5.5 bpw
> * `IQ6_K` - blocks of 16 weights in super-blocks of 256. Lookup table is 2x64 entries, 6.5 bpw
> 
> The sub-4 bpw `IQX_K` quants are much faster on the CPU than the corresponding i-quants and about on par with k-quants. On CUDA performance is more influenced by the block size than it is by the additional lookup required. If we take `IQ4_KS` as an example, it is faster than `Q4_0` (the quant that receives the largest amount of attention and love in mainline `llama.cpp`) for token generation, and only 3-4% slower for prompt processing. On the other hand, the quants that use blocks of 16 tend to be 20-25% slower for prompt processing than quants with blocks of 32 (due to me re-using the GEMM kernel that came from Jonahhes, and the block of 16 kernel not being as good as the block of 32 kernel). Token generation is memory bound, so speed is entirely determined by bpw, and none of the packing details or lookup tables matters that much.
> 
> Hope this answers your questions.
> 
> 👤 **afsara-ben** replied the **2025-06-13** at **20:51:18**:<br>
> thanks for your reply. What is the non-linear function that results in the lookup grid being smaller? Since it fits into 1/2 SIMD registers, so number of load requests is lower than what would be required for codebook? Additionally, will there be a Metal implementation of the `IQX_K` quants?
> 
> 👤 **ikawrakow** replied the **2025-06-14** at **03:03:48**:<br>
> Codebooks are for a group of quants, so much larger. Depending on quantization type the codebooks are between 256 and 2048 entries.
> 
> The non-linear function is a 3rd order polynomial. But since it acts on the quantized values it can only take a limited number of different values (4 for 2 bits, 8 for 3 bits, etc). These values can be rounded to the nearest 8-bit integer and put in a lookup table.
> 
> There is already a metal implementation for `IQX_K` quants. But since the Apple GPU is very low-end, performance is somewhat lower when I test on my M2-Max. The Metal back-end is not as well maintained as CPU and CUDA in `ik_llama.cpp`, so some of the advanced optimizations are not implemented there.
> 
> 👤 **afsara-ben** replied the **2025-06-17** at **23:29:17**:<br>
> thanks for the reply. if its not too much hassle, can you elaborate further how the kgrid matrices in the original IQ quants (PR [#4773]( https://github.com/ggml-org/llama.cpp/pull/4773))were generated ? I wanted to generate my own kgrid matrices so was wondering if there's a script that we can play with?

---

👤 **ikawrakow** replied the **2025-06-21** at **14:15:54**:<br>

@zhouwg 

Nice to meet you too.

I don't think I want to get involved with your dispute with the `llama.cpp` maintainers or discuss my reasons for leaving the `llama.cpp` project. 

Concerning a port of the `iqk` GEMM/GEMV implementation to  Qualcomm Hexagon cDSP: you are obviously free to make a port, and I can try to help as time permits. But be warned: adding this port to your ongoing PR will reduce its chance of getting accepted to zero.

> 👤 **ikawrakow** replied the **2025-06-22** at **13:52:00**:<br>
> You are likely not building the project correctly. `ik_lllama.cpp` is fast, but not 6 times faster than `llama.cpp` for `Q4_0`. What happens if you rebase on the latest main branch and run?
> 
> 👤 **ikawrakow** replied the **2025-06-22** at **14:42:43**:<br>
> So, why is the output correct now, but was gibberish before?
> 
> 👤 **ikawrakow** replied the **2025-06-22** at **14:52:22**:<br>
> But is correct with `-march=armv8.7-a+dotprod+fp16` ? And then PP-512 is 10 times faster than `llama.cpp`?
> 
> 👤 **ikawrakow** replied the **2025-06-22** at **15:02:12**:<br>
> What does `main_gpu=4` mean in the `llama.cpp` run?