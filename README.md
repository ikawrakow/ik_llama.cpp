# llama.cpp clone with better CPU performance

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## TL;DR

This repository is a clone of [llama.cpp](https://github.com/ggerganov/llama.cpp) with the following improvements
* Better implementation of CPU matrix multiplications (`AVX2` and `ARM_NEON`) for `fp16/fp32` and all k-, i-, and legacy `llama.cpp` quants, that leads to a significant improvement in prompt processing (PP) speed. Token generation (TG) also benefits, but to a lesser extent due to TG being memory bound
* Implementation of the [Bitnet b1.58](https://huggingface.co/1bitLLM/bitnet_b1_58-3B) model for the CPU (`AVX2` and `ARM_NEON`) and GPU (`CUDA` and `Metal`)
* Faster CPU inferrence for MoE models

If you are not already familiar with [llama.cpp](https://github.com/ggerganov/llama.cpp), it is better to start there. For those familiar with `llama.cpp`, everything works the same as `llama.cpp` (or at least the way `llama.cpp` worked when I last synced on June 21).

Note that I have published some, but not all, of the code in the respository in a series of [llamafile](https://github.com/Mozilla-Ocho/llamafile) PRs ([394](https://github.com/Mozilla-Ocho/llamafile/pull/394), [405](https://github.com/Mozilla-Ocho/llamafile/pull/405), [428](https://github.com/Mozilla-Ocho/llamafile/pull/428), [435](https://github.com/Mozilla-Ocho/llamafile/pull/435), [453](https://github.com/Mozilla-Ocho/llamafile/pull/453), and [464](https://github.com/Mozilla-Ocho/llamafile/pull/464)) 


## Why?

Mostly out of curiosity:
* Justine Tunney's `tinyBLAS`, which she contributed to `llama.cpp` in [PR 6414](https://github.com/ggerganov/llama.cpp/pull/6414), only works for `Q4_0`, `Q8_0` and `fp16/bf16` models. In the surrounding discussion about possibly extending `tinyBLAS` to k- and i-quants, she felt that k-quants are [not ammenable to block-tiling](https://github.com/ggerganov/llama.cpp/pull/6840#issuecomment-2072995387), which is required to improve performance. This statement piqued my curiosity, so here we are.
* Bitnet-1.58b has been one of the [most discussed topics](https://github.com/ggerganov/llama.cpp/issues/5761#issuecomment-2198380366) in the `llama.cpp` project, so eventually I decided to see how efficiently one can implement a tertiary model

Curiosity aside, improved CPU performance may be (or may become) important in practice. According to The Register, 70% of AI inferrence [is done on the CPU](https://www.theregister.com/2024/05/30/arm_cortex_x925_ai_cores/?td=rt-3a), at least in the Android world (but I haven't come around to actually comparing performancer on a phone). With ever increasing number of LLM model parameters, and with Meta's 400B model release imminent, the CPU may become the only viable option for people not willing (or not able to) rent/buy uber expensive GPU instances capable of running such models. Granted, one would need a pretty beefy computer to run a 400B model, and inference speed will be sluggish, but at least one will not need to spend the equivalent of a luxury apartmenty in the downtown of the city where I live to buy the GPU system capable of running the model.

## Bitnet-1.58B

Two implementations are provided
* `IQ1_BN` - uses 1.625 bits-per-weight (bpw)
* `IQ2_BN` - uses 2.0 bpw

`IQ2_BN` is faster for PP (CPU and GPU, although the PP performance difference on CUDA is very minor). `IQ1_BN` can arrive at a higher TG performance on the CPU (given enough threads) because of the smaller model size, but it is always slower on the GPU.

There is the unmerged [PR 8151](https://github.com/ggerganov/llama.cpp/pull/8151) in `llama.cpp` that implements Bitnet-1.58B for the CPU (`AVX` and `ARM_NEON`). The following table compares performance between this repo and `PR-8151` in `llama.cpp`.

## Performance comparison to llama.cpp

The results in the following tables are obtained with these parameters:
* Model is LLaMA-v3-8B for `AVX2` and LLaMA-v2-7B for `ARM_NEON`
* The `AVX2` CPU is a 16-core Ryzen-7950X
* The `ARM_NEON` CPU is M2-Max
* `tinyBLAS` is enabled in `llama.cpp`
* `llama.cpp` results are for `build: 081fe431 (3441)`, which was the current `llama.cpp` master branch master branch when I pulled on July 23 2024.
* The project is built without `CUDA` support, no `BLAS`, and Accelerate framework disabled
* Commandline is `./bin/llama-bench -m $model -p 512 -n 0 -t $num_threads -ngl 0` for prompt processing and `./bin/llama-bench -m $model -p 0 -n 128 -t $num_threads -ngl 0` for token generation tests 

### Prompt processing

Here I set the number of threads to be equal to the number of (performance) cores of the CPU, so 16 threads for the Ryzen-7950X and 8 threads for the M2-Max. The following table summarizes the results. To not make the table too long, I have listed only quantized models containing predominantly one quantization type (i.e., excluded the `QX_K - Medium` quants, which are typically a mix of `QX_K` and `Q(X+1)_K`, as well as `IQ2_S` and `IQ3_XS`).  

| Quantization          |       size | backend    | threads |          test | t/s (llama.cpp)  | t/s (iqk_mul_mat)| Speedup |
| --------------------- | ---------: | ---------- | ------: | ------------: | ---------------: | ---------------: | ------: |
| F16                   |  14.96 GiB | AVX2       |      16 |         pp512 |    112.37 ± 0.40 |    131.27 ± 0.38 |  1.168  |
| Q8_0                  |   7.95 GiB | AVX2       |      16 |         pp512 |    118.07 ± 0.53 |    134.00 ± 0.47 |  1.135  |
| Q4_0                  |   4.35 GiB | AVX2       |      16 |         pp512 |    104.46 ± 0.33 |    130.20 ± 0.29 |  1.246  |
| Q4_1                  |   4.77 GiB | AVX2       |      16 |         pp512 |     57.83 ± 0.24 |    160.69 ± 0.49 |  2.779  |
| Q5_0                  |   5.22 GiB | AVX2       |      16 |         pp512 |     53.50 ± 0.35 |    122.62 ± 0.48 |  2.292  |
| Q5_1                  |   5.64 GiB | AVX2       |      16 |         pp512 |     50.85 ± 0.36 |    147.15 ± 0.47 |  2.894  |
| Q2_K - Small          |   2.78 GiB | AVX2       |      16 |         pp512 |    110.11 ± 0.28 |    192.47 ± 1.35 |  1.748  |
| Q3_K - Small          |   3.41 GiB | AVX2       |      16 |         pp512 |     77.42 ± 0.36 |    181.64 ± 0.44 |  2.346  |
| Q4_K - Small          |   4.36 GiB | AVX2       |      16 |         pp512 |     98.92 ± 0.34 |    185.35 ± 0.39 |  1.874  |
| Q5_K - Small          |   5.21 GiB | AVX2       |      16 |         pp512 |     69.44 ± 0.31 |    179.62 ± 0.69 |  2.587  |
| Q6_K                  |   6.14 GiB | AVX2       |      16 |         pp512 |     74.89 ± 0.26 |    181.86 ± 0.55 |  2.428  |
| IQ2_XXS - 2.0625 bpw  |   2.23 GiB | AVX2       |      16 |         pp512 |     42.57 ± 0.16 |    126.63 ± 0.55 |  2.975  |
| IQ2_XS - 2.3125 bpw   |   2.42 GiB | AVX2       |      16 |         pp512 |     46.45 ± 0.27 |    125.46 ± 0.43 |  2.701  |
| IQ2_M - 2.7 bpw       |   2.74 GiB | AVX2       |      16 |         pp512 |     40.76 ± 0.18 |    113.07 ± 0.48 |  2.774  |
| IQ3_XXS - 3.0625 bpw  |   3.04 GiB | AVX2       |      16 |         pp512 |     31.95 ± 0.20 |    109.86 ± 0.45 |  3.438  |
| IQ3_S - 3.4375 bpw    |   3.42 GiB | AVX2       |      16 |         pp512 |     28.04 ± 0.08 |     96.28 ± 0.45 |  3.434  |
| IQ4_XS - 4.25 bpw     |   4.13 GiB | AVX2       |      16 |         pp512 |     68.98 ± 0.31 |    180.34 ± 0.55 |  2.614  |
| IQ4_NL - 4.5 bpw      |   4.35 GiB | AVX2       |      16 |         pp512 |     59.94 ± 0.21 |    129.06 ± 0.43 |  2.153  |

We see that `llama.cpp` achieves respectable performance for `fp16`, `Q8_0`, and `Q4_0`, being only up to 20% slower than this implementation. This is thanks to the use of Justine Tunney's `tinyBLAS`, which is utilized for these quantization types. For all other quants we observe performance gains in the `1.75X - 3.5X` range, which is not a small feat considering that the `ggml` matrix multiplication functions has been rewritten several times since `llama.cpp` was first published. 

## MoE models

There is [PR-6840](https://github.com/ggerganov/llama.cpp/pull/6840) from Justine Tunney in `llama.cpp`, but it has not been merged since April 23, so I'll compare performance to the master branch for Mixtral-8x7B. 

## To tile or not to tile

The common wisdom for efficient matrix multiplications is to use block tiling, and this is also used here for `fp16/fp32` matrices. But block tiling does not somehow magically reduce the amount of computation that needs to get done. Performance gains are simply due to the better utilization of memory caches. When dealing with quantized matrix multiplications, there is an additional factor that comes into play: the quantized data needs to be unpacked to 8-bit integers before being used in the matrix multiplication multiply-add operations. Depending on quantization type, this unpacking can represent a significant fraction of the overall computation cost. Hence, for best performance, one would want to reuse the unpacked quants as much as possible, thus spending some fraction of the available vector registers to hold the unpacked data. But when using block tiling, one also needs a certain number of vector registers for accumulating results. For instance, on `AVX2` (16 vector registers available), for `fp16/fp32` models best performance is achieved with `2 x 6` tiles (where the `2` refers to rows in the left matrix and is measured in units of the vector register size, so 16/8 floats for `fp16/fp32`, and `6` is for the number of columns in the right matrtix). Unpacking quantized data works best when done in blocks of 128 or 256 quants so that, if we wanted to keep unpacked unpacked quants for 2 rows, we would need at least 8 vector registers, thus being left with less than 8 registers for result accumulation, so at best `2 x 3` tiles. In practice one needs addition vector registers for various constants that are typically needed for de-quantization, so that, at the end, it becomes better to use `1 x N` "tiles", i.e., a row-wise multiplication where each row in the left matrix is multiplied with `N` columns in the right matrix, thus reusing the unpacked data `N` times. This (i.e., amortizing de-quantization cost) is the main mechanism for speding up quantized matrix multiplications. Having started with quantized matrices, and having gone from tiles to a row-wise implementation after some experimentation, I did try row-wise multiplication for float matrices first. Performance was not quite as good as for block-tiling, but I did get up to 90-95% of the speed of `tinyBLAS` that way before switching the `fp16/fp32` implementation to `2 x 6` (`AVX2`) or `5 x 5` (`AVX512` and `ARM_NEON`) block-tiles. But even for for `Q8_0 x Q8_0` multiplications, where there is basically no de-quantization cost, row-wise multiplication is faster than tiling (and beats `tinyBLAS`, which uses block-tiling also for `Q8_0`).          

