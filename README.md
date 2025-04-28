# ik_llama.cpp: llama.cpp fork with better CPU performance

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## TL;DR

This repository is a fork of [llama.cpp](https://github.com/ggerganov/llama.cpp) with better CPU and hybrid GPU/CPU performance.

## Latest News

* April 26 2025: GLM-4 support added
* April 26 2025: Command-A support added
* April 22 2025: Support for the latest Microsoft Bitnet model added
* April 17 2025: Better CPU Flash Attention token generation performance
* April 13 2025: `IQ1_M` quantization improvements
* April 10 2025: LLaMA-4 support added
* April 7 2025: `IQ2_XS` quantization improvements
* April 3 2025: Much faster MoE implementation on Metal
* April 1 2025: Quantization improvements for `Q2_K, Q4_K, Q5_K, Q4_1, Q5_1`
* March 28 2025: Quantization imrovements for `Q4_0, Q5_0, Q6_0, Q3_K, Q6_K, IQ4_XS, IQ4_NL`
* March 25 2025: Better MoE performance on CUDA
* March 23 2025: Better batched processing speed for DeepSeek models
* March 22 2025: Gemma3 support added
* March 21 2025: FlashMLA-3: fastest CPU-only inference for DeepSeek models
* March 18 2025: reduce compute buffer size
* March 17 2025: FlashMLA-2 performance improvements
* March 12 2025: Allow `Q8_0` KV cache with FlashMLA-2 on CUDA
* March 10 2025: Better TG performance for MoE models on CUDA
* March 9 2025: FlashMLA on CUDA
* March 8 2025: Faster FlashMLA CPU implementation
* March 7 2025: Custom quantization mixes using regular expressions
* March 5 2025: FlashMLA on CUDA
* March 3 2025: Introducing FlashMLA - MLA with Flash Attention
* March 1 2025: Smart Expert Reduction for faster DeepSeek inference
* Feb 27 2025: MLA without transposed cache
* Feb 25 2025: tensor overrides for better control where model weights are stored (GPU or CPU)
* Feb 23 2025: fused FFN ops for faster MoE inference
* Feb 23 2025: `sweep-bench` - better performance benchmarking
* Feb 20 2025: fast GEMM/GEMV for `IQ1_S`
* Feb 19 2025: `Q8_KV` - new type for 8-bit KV-cache quantization
* Feb 13 2025: allow `Q8_0` quantized cache with MLA
* Feb 11 2025: Flash Attention support for DeepSeek models
* Feb 9 2025: MLA for DeepSeek models
* Jan 23 2025: DeepSeek-V3 support added

### Contributing

Contributions in form of pull requests or issue submissions (bug reports, feature requests) are welcome.

### Licens

MIT
