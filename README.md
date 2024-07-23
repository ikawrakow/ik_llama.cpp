# llama.cpp clone with better CPU performance

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

----

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#description">Description</a>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#get-the-code">Get the Code</a></li>
        <li><a href="#build">Build</a></li>
        <li><a href="#blas-build">BLAS Build</a></li>
        <li><a href="#prepare-and-quantize">Prepare and Quantize</a></li>
        <li><a href="#run-the-quantized-model">Run the quantized model</a></li>
        <li><a href="#memorydisk-requirements">Memory/Disk Requirements</a></li>
        <li><a href="#quantization">Quantization</a></li>
        <li><a href="#interactive-mode">Interactive mode</a></li>
        <li><a href="#constrained-output-with-grammars">Constrained output with grammars</a></li>
        <li><a href="#obtaining-and-using-the-facebook-llama-2-model">Obtaining and using the Facebook LLaMA 2 model</a></li>
        <li><a href="#seminal-papers-and-background-on-the-models">Seminal papers and background on the models</a></li>
        <li><a href="#perplexity-measuring-model-quality">Perplexity (measuring model quality)</a></li>
        <li><a href="#android">Android</a></li>
        <li><a href="#docker">Docker</a></li>
      </ul>
    </li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#coding-guidelines">Coding guidelines</a></li>
    <li><a href="#docs">Docs</a></li>
  </ol>
</details>

## TL;DR

This repository is a clone of [llama.cpp](https://github.com/ggerganov/llama.cpp) with the following improvements
* Better implementation of CPU matrix multiplications (`AVX2` and `ARM_NEON`) for `fp16/fp32` and all k-, i-, and legacy `llama.cpp` quants, that leads to a significant improvement in prompt processing (PP) speed. Token generation (TG) also benefits, but to a lesser extent due to TG being memory bound
* Implementation of the [Bitnet b1.58](https://huggingface.co/1bitLLM/bitnet_b1_58-3B) model for the CPU (`AVX2` and `ARM_NEON`) and GPU (`CUDA` and `Metal`)
* Faster CPU inferrence for MoE models

If you are not already familiar with [llama.cpp](https://github.com/ggerganov/llama.cpp), it is better to start there. For those familiar with `llama.cpp`, everything works the same as `llama.cpp` (or at least the way `llama.cpp` worked when I last synced on June 21).

Note that I have published some, but not all, of the code in the respository in a series of [llamafile](https://github.com/Mozilla-Ocho/llamafile) PRs ([394](https://github.com/Mozilla-Ocho/llamafile/pull/394), [405](https://github.com/Mozilla-Ocho/llamafile/pull/405), [428](https://github.com/Mozilla-Ocho/llamafile/pull/428), [435](https://github.com/Mozilla-Ocho/llamafile/pull/435), [453](https://github.com/Mozilla-Ocho/llamafile/pull/453), and [464](https://github.com/Mozilla-Ocho/llamafile/pull/464)) 

## Why

Mostly out of curiosity:
* Justine Tunney's `tinyBLAS`, which she contributed to `llama.cpp` in [PR 6414](https://github.com/ggerganov/llama.cpp/pull/6414), only works for `Q4_0`, `Q8_0` and `fp16/bf16` models. In the surrounding discussion about possibly extending `tinyBLAS` to k- and i-quants, she felt that k-quants are [not ammenable to block-tiling](https://github.com/ggerganov/llama.cpp/pull/6840#issuecomment-2072995387), which is required to improve performance. This statement piqued my curiosity, so here we are.
* Bitnet-1.58b has been one of the [most discussed topics](https://github.com/ggerganov/llama.cpp/issues/5761#issuecomment-2198380366) in the `llama.cpp` project, so eventually I decided to see how efficiently one can implement a tertiary model

Curiosity aside, improved CPU performance may be (or may become) important in practice. According to The Register, 70% of AI inferrence [is done on the CPU](https://www.theregister.com/2024/05/30/arm_cortex_x925_ai_cores/?td=rt-3a), at least in the Android world. With ever increasing number of LLM model parameters, and with Meta's 400B model release imminent, the CPU may become the only option for people not willing (or not able to) rent/buy uber expensive GPU instances capable of running such models. Granted, one would need a pretty beefy computer to run a 400B model, inference speed will be sluggsh, but at least one will not need to spend the equivalent of a luxury apartmenty in the downtown of the city where I live.

## Bitnet-1.58B

Two implementations are provided
* `IQ1_BN` - uses 1.625 bits-per-weight (bpw)
* `IQ2_BN` - uses 2.0 bpw

`IQ2_BN` is faster for PP. `IQ1_BN` can arrive at a higher TG performance on the CPU (given enough threads), but is always slower on the GPU.

There is the unmerged [PR 8151](https://github.com/ggerganov/llama.cpp/pull/8151) in `llama.cpp` that implements Bitnet-1.58B for the CPU (`AVX` and `ARM_NEON`). The following table compares performance between this repo and `PR-8151` in `llama.cpp`.

## Performance comparison to llama.cpp

The results in the following table are obtained with the following parameters:
* Model is LLaMA-v3-8B
* The `AVX2` CPU is a 16-core Ryzen-7950X
* The `ARM_NEON` CPU is M2-Max
* `tinyBLAS` is enabled in `llama.cpp`
* `llama.cpp` results are for `build: 081fe431 (3441)`, which is the master branch as of July 23 2024.

## MoE models

There is [PR-6840](https://github.com/ggerganov/llama.cpp/pull/6840) from Justine Tunney in `llama.cpp`, but it has not been merged since April 23, so I'll compare performance to the master branch for Mixtral-8x7B. 

## To tile or not to tile

