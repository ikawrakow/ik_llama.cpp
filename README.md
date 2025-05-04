# ik_llama.cpp: llama.cpp fork with better CPU performance

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## TL;DR

This repository is a fork of [llama.cpp](https://github.com/ggerganov/llama.cpp) with better CPU and hybrid GPU/CPU performance, new SOTA quantization types, first-class Bitnet support, better DeepSeek performance via MLA, FlashMLA, fused MoE operations and tensor overrides for hybrid GPU/CPU inference, row-interleaved quant packing, etc.

>[!IMPORTANT]
>The new GGUFs for DeepSeek-V3/R1/Lite do not work in this repository. This is due to the backwards incompatibe change in mainline `llama.cpp` that [added MLA support](https://github.com/ggml-org/llama.cpp/pull/12801)
>2.5 months after MLA was available here, and worked with the original DeepSeek GGUFs. Please use the original GGUF or, if you don't have one, convert the HF safetnosrs using the Python conversion scrip in this repository.     

## Latest News

* May 4 2025: 🚀 Significant token generation performance improvement on CUDA with Flash Attention for GQA models. For details and benchmarks see [PR #370](https://github.com/ikawrakow/ik_llama.cpp/pull/370) 
* April 29 2025: Qwen3 support added
* April 26 2025: GLM-4 support added
* April 26 2025: Command-A support added
* April 22 2025: Support for the latest Microsoft Bitnet model added
* April 21 2025: ik_llama.cpp builds and runs successfully on Android (using termux)
* April 17 2025: 🚀 Better CPU Flash Attention token generation performance
* April 13 2025: `IQ1_M` quantization improvements
* April 10 2025: LLaMA-4 support added
* April 7 2025: `IQ2_XS` quantization improvements
* April 3 2025: 🚀 Much faster MoE implementation on Metal
* April 1 2025: Quantization improvements for `Q2_K, Q4_K, Q5_K, Q4_1, Q5_1`
* March 28 2025: Quantization imrovements for `Q4_0, Q5_0, Q6_0, Q3_K, Q6_K, IQ4_XS, IQ4_NL`
* March 25 2025: 🚀 Better MoE performance on CUDA
* March 23 2025: 🚀 Better batched processing speed for DeepSeek models
* March 22 2025: Gemma3 support added
* March 21 2025: 🚀 FlashMLA-3: fastest CPU-only inference for DeepSeek models
* March 18 2025: Reduce compute buffer size
* March 17 2025: 🚀 FlashMLA-2 performance improvements
* March 12 2025: Allow `Q8_0` KV cache with FlashMLA-2 on CUDA
* March 10 2025: 🚀 Better TG performance for MoE models on CUDA
* March 9 2025: 🚀 FlashMLA on CUDA
* March 8 2025: 🚀 Faster FlashMLA CPU implementation
* March 7 2025: Custom quantization mixes using regular expressions
* March 5 2025: 🚀 FlashMLA on CUDA
* March 3 2025: 🚀 Introducing FlashMLA - MLA with Flash Attention
* March 1 2025: Smart Expert Reduction for faster DeepSeek inference
* Feb 27 2025: MLA without transposed cache
* Feb 25 2025: Tensor overrides for better control where model weights are stored (GPU or CPU)
* Feb 23 2025: 🚀 Fused FFN ops for faster MoE inference
* Feb 23 2025: `sweep-bench` - better performance benchmarking
* Feb 20 2025: 🚀 Fast GEMM/GEMV for `IQ1_S`
* Feb 19 2025: `Q8_KV` - new type for 8-bit KV-cache quantization
* Feb 13 2025: Allow `Q8_0` quantized cache with MLA
* Feb 11 2025: 🚀 Flash Attention support for DeepSeek models
* Feb 9 2025: 🚀 MLA for DeepSeek models
* Jan 23 2025: DeepSeek-V3 support added

## Resources

There is no single point of reference describing all new `ik_llama.cpp` features. Pull requests often contain detailed information, so browsing the PRs is often the best way to learn about new features and how to use them. In addition
* [The Wiki page](https://github.com/ikawrakow/ik_llama.cpp/wiki) has performance comparisons to mainline `llama.cpp`
* [This guide](https://github.com/ikawrakow/ik_llama.cpp/discussions/258) is a good place to start if you came here because of DeepSeek models
* [This discussion](https://github.com/ikawrakow/ik_llama.cpp/discussions/266) is about running DeepSeek-V3/R1 on a 16 x 3090 setup
* [This discussion](https://github.com/ikawrakow/ik_llama.cpp/discussions/8) describes the new quantization types available in `ik_llama.cpp`

## Contributing

Contributions in form of pull requests, issue submissions (bug reports, feature requests), or general discussions, are welcome.

## License

MIT
