### 🔀 [#208](https://github.com/ikawrakow/ik_llama.cpp/pull/208) - Q8_KV: 8-bit quantization type targeting the KV cache

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-02-18 |
| **Updated** | 2025-02-19 |

---

#### Description

What is `Q8_KV`? It is 8-bit quantization with a single scale per tensor row (so, no blocks at all). That may not be accurate enough for model quantization, but using it for KV cache quantization seems plausible, considering that there rows are defined by the head size, so contain 64, 80, 96, 128, 192, or 256 elements for all LLMs currently in circulation. We are not looking for KV cache size reduction but rather for improving inference performance for long contexts. This is especially relevant for MLA (DeepSeek) as in FA the kernels are highly optimized, so large improvements may not be really possible.

Caveat: everything is CPU only, there is no CUDA or Metal implementation.

The following changes are made:
* New types `Q8_KV` and `Q8_KV_R8` are added. `Q8_KV_R8` is `Q8_KV` with 8 interleaved rows
* Both can be used for model quantization, but quantization error is to high relative to the 8 bpw spent (it is roughly equivalent to 5 bpw). Prompt processing speed with these quants is great. On the M2-Max CPU we get 194 t/s for LlaMA-3-8B, so ~15% faster than `Q8_K_R8`, the so far fastest quantization type for prompt processing. On `AVX2/Zen4`   `Q8_KV_R8` is slightly slower than `Q8_K_R8`, which is somewhat surprising.
* Changes necessary to successfully store and use `Q8_KV` quants in the K cache. This required various fixes in `llama.cpp` and `ggml`. There were still places left where the number of bytes needed to store a row of size `N` are computed as `(N/B)*T`, where `B` is the type block size and `T` is the type size. This of course fails when the row has extra meta data. There is the function `ggml_row_size(ggml_type type, int64_t N)` to compute this, but I had missed a few places when adding the `IQK` quants. It also turned out that in quite a few places `ggml_row_size()` is not used correctly. E.g., for the KV cache we find `ggml_row_size(type_k, head_size*num_heads)` instead of `ggml_row_size(type_k, head_size)*num_heads`. Same issue was also present in the MoE matrix multiplication function.
* I couldn't get it to work for the V cache. But as the V cache can only be quantized when using FA, and as MLA was my main focus and I wasn't expecting performance gains from quantizing the V cache with `Q8_KV`, I didn't put too much effort into hunting down all places of incorrect `ggml_row_size()` usage.
* All necessary changes to be also able to use `Q8_KV` in FA. Here we get a minor speedup compared to `Q8_0` (1-2% at 16k tokens).

A quantization type such as `Q8_KV` has the distinct advantage of making the results of matrix multiplications 100% reproducible and independent of the hardware the calculation is being done on (the row x column dot products are performed using integer arithmetic, and only at the end the row scale is applied, so number of threads used and order of summation does not affect the final result). I know there is interest in that sort of thing, but I leave further exploration for another day.  

After all this, here is a comparison between the main branch and this PR for DeepSeek-Lite (acting as a surrogate for DeepSeek-R1) with MLA enabled. The V cache is `bf16`, the model is quantized with `IQ4_XS`, and the calculation is on a Ryzen-7950X CPU. The main branch uses `Q8_0` for the K cache, the PR uses `Q8_KV` 

| model                 |     params | mla |          test |   t/s (main)     |   t/s (PR)       |  Speedup |
| --------------------- | ---------: | --: | ------------: | ---------------: | ---------------: | -------: |
| deepseek2 16B IQ4_XS  |    15.76 B |   1 |         pp512 |    490.47 ± 1.12 |    507.33 ± 3.60 |  1.034   |   
| deepseek2 16B IQ4_XS  |    15.76 B |   1 |        pp1024 |    464.92 ± 1.44 |    491.55 ± 1.71 |  1.057   |
| deepseek2 16B IQ4_XS  |    15.76 B |   1 |        pp2048 |    416.22 ± 2.54 |    452.57 ± 5.00 |  1.087   |
| deepseek2 16B IQ4_XS  |    15.76 B |   1 |        pp4096 |    341.52 ± 1.70 |    388.29 ± 0.14 |  1.137   |
| deepseek2 16B IQ4_XS  |    15.76 B |   1 |        pp8192 |    252.49 ± 0.32 |    300.62 ± 0.12 |  1.191   |
| deepseek2 16B IQ4_XS  |    15.76 B |   1 |       pp16384 |    160.72 ± 3.78 |    207.43 ± 0.55 |  1.291   |

Here is a perplexity comparison between `Q8_0` and `Q8_KV` used for model and K cache quantization for DeepSeek-Lite with a context of 512 tokens. `PPL(fp16) =  6.7612`

| model quantization | K cache quantization | PPL |
| ----: | ---: | ---: |
| Q8_0 | Q8_0 | 6.7597 |
| Q8_0 | Q8_KV | 6.7699 |
| Q8_0 | Q6_0 | 6.7991 |
| Q8_KV | Q8_KV | 6.8317 |
| Q8_KV* | Q8_0 | 6.7843 |
| Q8_KV* | Q8_KV | 6.7947 |

I.e., using `Q8_KV` for K-cache quantization leads to a very minor loss of accuracy (certainly much better than `Q6_0`), but using `Q8_KV` to quantize the model weights results in much more significant accuracy loss. 

### Update

I have added the last 2 rows to the above table. In `Q8_KV*` the output and token embedding tensors are quantized with `Q8_0`, so most of the accuracy loss comes from these two tensors (and they have negligible impact on performance). I have also rerun the performance tests after merging PR #210. Here are the updated results:

| model          |     params | mla |          test |     t/s (main)   |    t/s (PR)      |  Speedup  |
| -------------- | ---------: | --: | ------------: | ---------------: | ---------------: | --------: |
| deepseek2 16B  |    15.76 B |   1 |         pp512 |    594.08 ± 0.19 |    628.58 ± 9.38 |  1.058    |   
| deepseek2 16B  |    15.76 B |   1 |        pp1024 |    554.24 ± 0.90 |    593.06 ± 2.71 |  1.070    |
| deepseek2 16B  |    15.76 B |   1 |        pp2048 |    487.52 ± 4.64 |    545.96 ± 0.82 |  1.120    |
| deepseek2 16B  |    15.76 B |   1 |        pp4096 |    394.07 ± 0.16 |    454.95 ± 0.84 |  1.154    |
| deepseek2 16B  |    15.76 B |   1 |        pp8192 |    279.55 ± 0.14 |    339.74 ± 0.64 |  1.215    |
| deepseek2 16B  |    15.76 B |   1 |        pp8192 |    175.21 ± 0.14 |    225.35 ± 0.30 |  1.286    |