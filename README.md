# ik_llama.cpp: llama.cpp fork with better CPU performance

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## TL;DR

This repository is a fork of [llama.cpp](https://github.com/ggerganov/llama.cpp) with better CPU and hybrid GPU/CPU performance, new SOTA quantization types, first-class Bitnet support, better DeepSeek performance via MLA, FlashMLA, fused MoE operations and tensor overrides for hybrid GPU/CPU inference, row-interleaved quant packing, etc.

## Quickstart

```
# Install build dependencies and cuda toolkit as needed

# Clone
git clone https://github.com/ikawrakow/ik_llama.cpp
cd ik_llama.cpp

# Configure CUDA+CPU Backend
cmake -B ./build -DGGML_CUDA=ON -DGGML_BLAS=OFF

# *or* Configure CPU Only Backend
cmake -B ./build -DGGML_CUDA=OFF -DGGML_BLAS=OFF

# Build
cmake --build ./build --config Release -j $(nproc)

# Confirm
./build/bin/llama-server --version
version: 3597 (68a5b604)
```

See [this discussion](https://github.com/ikawrakow/ik_llama.cpp/discussions/258) for ik-llama specific parameters and enhancements.

## Latest News

* May 12 2025: User can now control if/which operations with tensors held in RAM are offloaded to the GPU. See [PR 405](https://github.com/ikawrakow/ik_llama.cpp/pull/405) 
* May 12 2025: Compatibility issues with mainline `llama.cpp` GGUFs for DeepSeek models with MLA enabled were resolved in [PR 394](https://github.com/ikawrakow/ik_llama.cpp/pull/394). The lower prompt processing performance resulting from using `llama.cpp`-style MLA GGUFs was recovered in [PR 409](https://github.com/ikawrakow/ik_llama.cpp/pull/409).
* May 11 2025: 🚀 Slightly faster flash attention for DeepSeek models on CUDA, along with extending compatibility to Touring or newer GPUs. See [PR 408](https://github.com/ikawrakow/ik_llama.cpp/pull/408)
* May 9 2025: Support for LlaMA-3-Nemotron models added, see [PR 377](https://github.com/ikawrakow/ik_llama.cpp/pull/377)
* May 7 2025: 🚀 Faster TG for DeepSeek models with GPU or hybrid GPU/CPU inference. See [PR 386](https://github.com/ikawrakow/ik_llama.cpp/pull/386) for details. Caveat: Ampere or newer Nvidia GPU required
* May 4 2025: 🚀 Significant token generation performance improvement on CUDA with Flash Attention for GQA models. For details and benchmarks see [PR #370](https://github.com/ikawrakow/ik_llama.cpp/pull/370) 
* April 29 2025: Qwen3 support added, see [PR 355](https://github.com/ikawrakow/ik_llama.cpp/pull/355)
* April 26 2025: GLM-4 support added, see [PR 344](https://github.com/ikawrakow/ik_llama.cpp/pull/344)
* April 26 2025: Command-A support added, see [PR 341](https://github.com/ikawrakow/ik_llama.cpp/pull/341)
* April 22 2025: Support for the latest Microsoft Bitnet model added, see [PR 337](https://github.com/ikawrakow/ik_llama.cpp/pull/337)
* April 21 2025: ik_llama.cpp builds and runs successfully on Android (using termux), see [PR 336](https://github.com/ikawrakow/ik_llama.cpp/pull/336)
* April 17 2025: 🚀 Better CPU Flash Attention token generation performance, see [PR 332](https://github.com/ikawrakow/ik_llama.cpp/pull/332)
* April 13 2025: `IQ1_M` quantization improvements, see [PR 327](https://github.com/ikawrakow/ik_llama.cpp/pull/327)
* April 10 2025: LLaMA-4 support added, see [PR 321](https://github.com/ikawrakow/ik_llama.cpp/pull/321). In the PR there are also some custom quantization recipes for L4-Scout provided.
* April 7 2025: `IQ2_XS` quantization improvements, see [PR 312](https://github.com/ikawrakow/ik_llama.cpp/pull/312)
* April 3 2025: 🚀 Much faster MoE implementation on Metal, see [PR 307](https://github.com/ikawrakow/ik_llama.cpp/pull/307) 
* April 1 2025: Quantization improvements for `Q2_K, Q4_K, Q5_K, Q4_1, Q5_1`, see [PR 302](https://github.com/ikawrakow/ik_llama.cpp/pull/302)
* March 28 2025: Quantization imrovements for `Q4_0, Q5_0, Q6_0, Q3_K, Q6_K, IQ4_XS, IQ4_NL`, see [PR 295](https://github.com/ikawrakow/ik_llama.cpp/pull/295)
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
