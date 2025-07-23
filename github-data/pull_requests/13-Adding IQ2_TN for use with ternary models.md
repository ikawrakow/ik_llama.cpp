### 🔀 [#13](https://github.com/ikawrakow/ik_llama.cpp/pull/13) - Adding IQ2_TN for use with ternary models

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-08-06 |
| **Updated** | 2024-08-07 |

---

#### Description

They have abandoned the `Q1_3` and `Q2_2` quants in [PR-8151](https://github.com/ggerganov/llama.cpp/pull/8151) in `llama.cpp`, and have moved on to `TQ1_0` and `TQ2_0`. Like k-quants, these use blocks of 256 weights and utilize `Q8_K` for quantized dot products on the CPU. This removes support for [Bitnet b1.58](https://huggingface.co/1bitLLM/bitnet_b1_58-3B) (unless one adds padding to a multiple of 256), so they are now focussing on the [TriLM models](https://huggingface.co/collections/SpectraSuite/trilms-unpacked-668d5f62afe0f4036925b1d2). Unlike the previous `Q1_3` and `Q2_2`, where the quantized data only holds the ternary `-1/0/+1` values and the tensor scale is added via a separate `ggml_scale` operation, the new `TQ1_0` and `TQ2_0` include a scale in each block of 256. This basically wastes 0.0625 bpw, but has the advantage that one can simply reuse the standard `llama.cpp` computation graphs.

Based on the `PP-512` and `TG-128` figures posted in [PR-8151](https://github.com/ggerganov/llama.cpp/pull/8151), `TQ2_0` performance is much better than the earlier `Q2_0` attempt, so I became curious to see how @compilade's implementation compares to what we can do with `iqk_mul_mat` in this repo, and here we are.

The PR adds `IQ2_TN` (`TN` as `TriNet`). Implementation for `Zen4`, `AVX2`, `ARM_NEON`, `CUDA` and `Metal` is provided.

Let's look at some performance comparisons. We will focus on the largest TriLM model, which has ~4B parameters. Quantized with 2.0625 bpw the model size is 1.08 GiB.

### AVX2

`AVX2` was tested on a 32-core Ryzen-5975WX CPU. Not everybody has a 32-core CPU handy, so I have added performance values for fewer threads.

| threads |          test |  t/s (PR-8151)   |  t/s (this PR) | Speedup |
| ------: | ------------: | ---------------: | -------------: | ------: |
|    32   |         pp512 |    430.18 ± 0.56 |  490.73 ± 0.62 |  1.141  |
|    16   |         pp512 |    258.47 ± 0.21 |  306.37 ± 0.03 |  1.185  |
|     8   |         pp512 |    141.94 ± 0.04 |  175.45 ± 0.06 |  1.236  |
|     4   |         pp512 |     74.72 ± 0.02 |   91.78 ± 0.01 |  1.228  |
|     1   |         tg128 |     15.75 ± 0.01 |   15.71 ± 0.01 |  1.000  |
|     2   |         tg128 |     24.22 ± 0.02 |   26.50 ± 0.00 |  1.094  |
|     4   |         tg128 |     33.66 ± 0.14 |   41.63 ± 0.04 |  1.237  |
|     8   |         tg128 |     44.34 ± 0.07 |   48.62 ± 0.03 |  1.097  |
|    16   |         tg128 |     49.58 ± 0.05 |   48.09 ± 0.03 |  0.970  |

I would say @compilade has done remarkably well here, coming to within ~14% for PP performance. Although, for fewer than 32 threads, the performance gap increases to about ~23%. My guess is that the 23% is a more realistic value for the performance difference, and as the number of threads increases we see more the effect of `ggml` inefficiencies (thread synchronization, operations that do not scale with number of threads, etc.), which then narrows the gap. Nevertheless, even 23% is remarkable considering the performance differences for other quants (see main page). For TG the performance is the same for 1 thread (not much one can do there, the bit arrangement is so simple that there aren't many different ways to implement effectively with `AVX2`). The implementation in this PR then becomes faster, I guess due to better cache utilization. But this better per thread performance leads to too much memory bandwidth contention above 8 threads, so `TQ2_0` is able to arrive at a slightly better performance at 16 threads. 

### Zen4

I have also tested on a `Zen4` CPU (16-core Ryzen-7950X). `Zen4` implements some of the `AVX512` instruction set, and there is a dedicated implementation for that for `IQ2_TN`. The `TQ2_0` quants are implemented in pure `AVX2`, so one might think the performance comparison is unfair. But, at least as far as I know,  the `Zen4` core implements 512-bit instructions as two separate 256-bit instructions in hardware, so one does not gain much by operating on 512-bit wide vectors. The main advantage comes from having more vector registers (32 vs 16 on `AVX2`), but the way matrix multiplications are done in `ggml` (a series of vector x vector dot products), one cannot really take advantage of that. Anyway, here is the performance comparison on the Ryzen-7950X CPU

| threads |          test |  t/s (PR-8151)   |  t/s (this PR)   | Speedup |
| ------: | ------------: | ---------------: | ---------------: | ------: |
|      16 |         pp512 |    276.74 ± 0.75 |    429.97 ± 1.41 |  1.553  |
|       8 |         pp512 |    151.50 ± 0.46 |    250.88 ± 0.31 |  1.656  |
|       4 |         pp512 |     78.82 ± 0.64 |    131.29 ± 0.23 |  1.665  |
|       1 |         tg128 |     18.76 ± 0.40 |     20.11 ± 0.05 |  1.072  |
|       2 |         tg128 |     29.38 ± 0.05 |     35.69 ± 0.07 |  1.215  |
|       4 |         tg128 |     46.39 ± 0.04 |     48.62 ± 0.01 |  1.048  |
|       8 |         tg128 |     47.94 ± 0.03 |     48.28 ± 0.04 |  1.007  |

Here the PP performance gap is more significant at around 66%, reducing to 55% at 16 threads. If we look at TG performance for 1 thread, the ~7% performance difference comes from using `_mm512_dpbusd_epi32`, which is a fused multiply-add operation, whereas on `AVX2` one needs to use `_mm256_maddubs_epi16` followed by `_mm256_add_epi16` to accumulate the result. The TG performance gap then widens due to better cache utilization, and then decreases towards zero with increasing numbers of threads as the memory bandwidth is saturated. The 66% PP performance gap is hence the combination of the ~7% due to the use a fused multiply-add, and ~60% due to better utilization of vector registers while performing a multiplication of a row in the left matrix with several columns in the right matrix, where the unpacked quants for a block are held in vector registers.

### ARM_NEON

Here @compilade's implementation does not do very well, at least not on the M2-Max laptop where I have tested. But perhaps this is just due to the fact that @compilade used a Cortex A72 CPU in their development, and that CPU may as well behave very differently from the M2-Max.

| threads |          test |  t/s (PR-8151)   |  t/s (this PR)   | Speedup |
| ------: | ------------: | ---------------: | ---------------: | ------: |
|       8 |         pp512 |     79.15 ± 0.21 |    206.60 ± 0.14 | 2.610   |
|       2 |         tg128 |     17.61 ± 0.01 |     28.42 ± 0.05 | 1.614   |
|       4 |         tg128 |     32.40 ± 0.02 |     49.23 ± 0.09 | 1.519   |
|       8 |         tg128 |     51.64 ± 0.70 |     76.37 ± 0.22 | 1.479   |

### CUDA and Metal

There is no GPU implementation in PR-8151, so here just the performance values for this PR. `CUDA` is tested on RTX-4080, `Metal` on a 30-code M2-Max GPU.

| backend |          test |  t/s (this PR)   |
| ------: | ------------: | ---------------: |
| CUDA    |         pp512 |  9937     ± 81  |
| CUDA    |         tg128 |   299.19  ± 0.15 |
| Metal   |         pp512 |   891.52  ± 0.49 |
| Metal   |         tg128 |     98.52 ± 0.16 |
          
I have not bothered implementing the MMQ stuff, so CUDA PP performance is via dequantize and cuBLAS gemm.

---

#### 💬 Conversation

👤 **compilade** commented the **2024-08-06** at **17:00:57**:<br>

This is great!

> ARM_NEON
> Here @compilade's implementation does not do very well

Yeah, I did not particularly optimize the ARM_NEON implementation for recent ARM CPUs (yet), especially since I did not use `vdotq_s32` (although I was planning to), because the Cortex-A72 and the Cortex-A53 in the CPUs of my test machines do not support that and were faster with `vmlal_s8` than with `ggml_vdotq_s32`.

---

I see `IQ2_TN` mostly has the same format as `TQ2_0`, except that the float16 scale is before the packed weights instead of after.
But if I understand it correctly, both store the packed values in the same order and packed in the same way (same offset). Does that mean the Metal and CUDA implementations for `IQ2_TN` would also work for `TQ2_0`?

Do you have plans for `IQ2_TN` to replace `TQ2_0`, or is this something done in parallel to see how fast it can get with better matrix multiplication than lots of dot products?

Either way, I really appreciate your work on this. This was a pleasant surprise to see in my notifications.