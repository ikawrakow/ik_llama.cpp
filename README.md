# ik_llama.cpp: llama.cpp fork with better CPU performance

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## TL;DR

This repository is a fork of [llama.cpp](https://github.com/ggerganov/llama.cpp) with better CPU and hybrid GPU/CPU performance, new SOTA quantization types, first-class Bitnet support, better DeepSeek performance via MLA, FlashMLA, fused MoE operations and tensor overrides for hybrid GPU/CPU inference, row-interleaved quant packing, etc.

## Latest News

### Model Support

| LlaMA-3-Nemotron | Qwen3 | GLM-4 | Command-A | bitnet-b1.58-2B-4T | LLaMA-4 | Gemma3 | DeepSeek-V3 |
|:----------------:|:-----:|:-----:|:---------:|:------------------:|:-------:|:------:|:-----------:|
[PR 377](https://github.com/ikawrakow/ik_llama.cpp/pull/377) | [PR 355](https://github.com/ikawrakow/ik_llama.cpp/pull/355) | [PR 344](https://github.com/ikawrakow/ik_llama.cpp/pull/344) | [PR 341](https://github.com/ikawrakow/ik_llama.cpp/pull/341) | [PR 337](https://github.com/ikawrakow/ik_llama.cpp/pull/337) | [PR 321](https://github.com/ikawrakow/ik_llama.cpp/pull/321) | [PR 276](https://github.com/ikawrakow/ik_llama.cpp/pull/276) | [PR 176](https://github.com/ikawrakow/ik_llama.cpp/pull/176) |
### Quantization

#### Quantization additions

##### Trellis quants (`IQ2_KT`, `IQ3_KT`, `IQ4_KT`)

Information and the original CUDA implementation in [PR 113](https://github.com/ikawrakow/ik_llama.cpp/pull/113). Additional implementations: Metal [PR 475](https://github.com/ikawrakow/ik_llama.cpp/pull/475), Neon [PR 471](https://github.com/ikawrakow/ik_llama.cpp/pull/471), CPU [PR 441](https://github.com/ikawrakow/ik_llama.cpp/pull/441)

##### IQK quants

Information about them can be found in [Discussion 8](https://github.com/ikawrakow/ik_llama.cpp/discussions/8).

|  | IQ2_KS | IQ2_K (R4) | IQ3_K (R4) | IQ4_KSS | IQ4_KS (R4) | IQ4_K (R4) | IQ5_KS (R4) | IQ5_K (R4) | IQ6_K |
|---------------------|:------:|:----------:|:----------:|:-------:|:-----------:|:----------:|:-----------:|:----------:|:-----:|
| CPU     | [85](https://github.com/ikawrakow/ik_llama.cpp/pull/85) | [7](https://github.com/ikawrakow/ik_llama.cpp/pull/7) ([146](https://github.com/ikawrakow/ik_llama.cpp/pull/146)) | [7](https://github.com/ikawrakow/ik_llama.cpp/pull/7) ([145](https://github.com/ikawrakow/ik_llama.cpp/pull/145)) | [89](https://github.com/ikawrakow/ik_llama.cpp/pull/89) | [83](https://github.com/ikawrakow/ik_llama.cpp/pull/83) ([150](https://github.com/ikawrakow/ik_llama.cpp/pull/150)) | [6](https://github.com/ikawrakow/ik_llama.cpp/pull/6) ([138](https://github.com/ikawrakow/ik_llama.cpp/pull/138)) | [422](https://github.com/ikawrakow/ik_llama.cpp/pull/422) ([426](https://github.com/ikawrakow/ik_llama.cpp/pull/426)) | [7](https://github.com/ikawrakow/ik_llama.cpp/pull/7) ([149](https://github.com/ikawrakow/ik_llama.cpp/pull/149)) | [14](https://github.com/ikawrakow/ik_llama.cpp/pull/14) |
| CUDA         | [418](https://github.com/ikawrakow/ik_llama.cpp/pull/418) | [418](https://github.com/ikawrakow/ik_llama.cpp/pull/418) ([461](https://github.com/ikawrakow/ik_llama.cpp/pull/461)) | [418](https://github.com/ikawrakow/ik_llama.cpp/pull/418) ([461](https://github.com/ikawrakow/ik_llama.cpp/pull/461)) | [89](https://github.com/ikawrakow/ik_llama.cpp/pull/89) | [83](https://github.com/ikawrakow/ik_llama.cpp/pull/493) ([493](https://github.com/ikawrakow/ik_llama.cpp/pull/493), [462](https://github.com/ikawrakow/ik_llama.cpp/pull/462)) | [417](https://github.com/ikawrakow/ik_llama.cpp/pull/417) ([461](https://github.com/ikawrakow/ik_llama.cpp/pull/461)) | [422](https://github.com/ikawrakow/ik_llama.cpp/pull/422) ([493](https://github.com/ikawrakow/ik_llama.cpp/pull/493), [462](https://github.com/ikawrakow/ik_llama.cpp/pull/462)) | [417](https://github.com/ikawrakow/ik_llama.cpp/pull/417) ([461](https://github.com/ikawrakow/ik_llama.cpp/pull/461)) | [417](https://github.com/ikawrakow/ik_llama.cpp/pull/417) |

##### Misc

`IQ1_S_R4`/`IQ1_M_R4`. (CPU: [PR 185](https://github.com/ikawrakow/ik_llama.cpp/pull/185)/[PR 187](https://github.com/ikawrakow/ik_llama.cpp/pull/187), CUDA: [PR 492](https://github.com/ikawrakow/ik_llama.cpp/pull/492)/[PR 494](https://github.com/ikawrakow/ik_llama.cpp/pull/494)). 
Note: These differ (and thus cannot be repacked) from `IQ1_S`/`IQ1_M`.

#### Quantization improvements

`IQ1_M` [PR 327](https://github.com/ikawrakow/ik_llama.cpp/pull/327), `IQ2_XS` [PR 312](https://github.com/ikawrakow/ik_llama.cpp/pull/312), `Q2_K, Q4_K, Q5_K, Q4_1, Q5_1` [PR 302](https://github.com/ikawrakow/ik_llama.cpp/pull/302), `Q4_0, Q5_0, Q6_0, Q3_K, Q6_K, IQ4_XS, IQ4_NL` [PR 295](https://github.com/ikawrakow/ik_llama.cpp/pull/295)

#### Quantization performance improvements 

* Faster CPU prompt processing for Trellis quants and MoE models. [PR 488](https://github.com/ikawrakow/ik_llama.cpp/pull/488)
* Trellis quants: faster CPU prompt processing [PR 482](https://github.com/ikawrakow/ik_llama.cpp/pull/482).
* Minor (~2%) `iq2_ks` TG performance improvement on CUDA [PR 468](https://github.com/ikawrakow/ik_llama.cpp/pull/468)
* Faster `IQ3_KT` and `IQ4_KT` [PR 453](https://github.com/ikawrakow/ik_llama.cpp/pull/453)
* Zen4: Faster PP for `IQ2_KS, IQ4_KS, IQ5_KS` [PR 428](https://github.com/ikawrakow/ik_llama.cpp/pull/428)
* Fast GEMM/GEMV for `IQ1_S` [PR 212](https://github.com/ikawrakow/ik_llama.cpp/pull/212)

### Features

* Legacy quants conversion schemes in `convert_hf_to_gguf.py` [PR 449](https://github.com/ikawrakow/ik_llama.cpp/pull/449), `Q6_0` in [PR 483](https://github.com/ikawrakow/ik_llama.cpp/pull/483)
* June 8 2025: Webui updated (legacy still available when `--path ./examples/server/public_legacy` is passed) [PR 481](https://github.com/ikawrakow/ik_llama.cpp/pull/481)
* June 8 2025: RPC improvements [PR 480](https://github.com/ikawrakow/ik_llama.cpp/pull/480)
* June 7 2025: Add an endpoint that lists all the saved prompt caches to server [PR 502](https://github.com/ikawrakow/ik_llama.cpp/pull/502)
* June 6 2025: Make prompt cache saving and restoring MLA aware [PR 497](https://github.com/ikawrakow/ik_llama.cpp/pull/497)
* June 3 2025: Added samplers, XTC [PR 486](https://github.com/ikawrakow/ik_llama.cpp/pull/486), top-n σ [PR 489](https://github.com/ikawrakow/ik_llama.cpp/pull/489).
* May 22 2025: Refactor `iqk_mul_mat.cpp` which speeds up compilation time significantly. [PR 435](https://github.com/ikawrakow/ik_llama.cpp/pull/435)
* May 17 2025: Option to enable or disable the CPU FA kernels [PR 429](https://github.com/ikawrakow/ik_llama.cpp/pull/429).
* May 12 2025: User can now control if/which operations with tensors held in RAM are offloaded to the GPU. See [PR 405](https://github.com/ikawrakow/ik_llama.cpp/pull/405) 
* May 12 2025: Compatibility issues with mainline `llama.cpp` GGUFs for DeepSeek models with MLA enabled were resolved in [PR 394](https://github.com/ikawrakow/ik_llama.cpp/pull/394). The lower prompt processing performance resulting from using `llama.cpp`-style MLA GGUFs was recovered in [PR 409](https://github.com/ikawrakow/ik_llama.cpp/pull/409).
* April 21 2025: ik_llama.cpp builds and runs successfully on Android (using termux), see [PR 336](https://github.com/ikawrakow/ik_llama.cpp/pull/336)
* March 1 2025: Smart Expert Reduction for faster DeepSeek inference [PR 239](https://github.com/ikawrakow/ik_llama.cpp/pull/239) 
* Feb 25 2025: Tensor overrides for better control where model weights are stored (GPU or CPU) [PR 232](https://github.com/ikawrakow/ik_llama.cpp/pull/232)
* Feb 23 2025: `sweep-bench` - better performance benchmarking [PR 225](https://github.com/ikawrakow/ik_llama.cpp/pull/225)
* Feb 19 2025: `Q8_KV` - new type for 8-bit KV-cache quantization [PR 208](https://github.com/ikawrakow/ik_llama.cpp/pull/208)
* March 7 2025: Custom quantization mixes using regular expressions [PR 244](https://github.com/ikawrakow/ik_llama.cpp/pull/244)

### Performance improvements

* May 13 2025: Better CPU FA performance for DeepSeek-Lite. [PR 410](https://github.com/ikawrakow/ik_llama.cpp/pull/410)
* May 11 2025: Slightly faster flash attention for DeepSeek models on CUDA, along with extending compatibility to Touring or newer GPUs. [PR 408](https://github.com/ikawrakow/ik_llama.cpp/pull/408)
* May 4 2025: Significant token generation performance improvement on CUDA with Flash Attention for GQA models. For details and benchmarks. [PR 370](https://github.com/ikawrakow/ik_llama.cpp/pull/370) 
* April 17 2025: Better CPU Flash Attention token generation performance. [PR 332](https://github.com/ikawrakow/ik_llama.cpp/pull/332)
* April 3 2025: Much faster MoE implementation on Metal. [PR 307](https://github.com/ikawrakow/ik_llama.cpp/pull/307) 
* March 25 2025: Better MoE performance on CUDA [PR 283](https://github.com/ikawrakow/ik_llama.cpp/pull/283)
* March 23 2025: Better batched processing speed for DeepSeek models [PR 282](https://github.com/ikawrakow/ik_llama.cpp/pull/282)
* March 18 2025: Reduce compute buffer size [PR 237](https://github.com/ikawrakow/ik_llama.cpp/pull/237)
* March 10 2025: Better TG performance for MoE models on CUDA [PR 248](https://github.com/ikawrakow/ik_llama.cpp/pull/248)
* Feb 23 2025: Fused FFN ops for faster MoE inference [PR 229](https://github.com/ikawrakow/ik_llama.cpp/pull/229)

### Flash-MLA

* May 7 2025: 🚀 FlashMLA-3 for DeepSeek models on CUDA. [PR 386](https://github.com/ikawrakow/ik_llama.cpp/pull/386). Caveat: Ampere or newer Nvidia GPU required
* March 21 2025: 🚀 FlashMLA-3: fastest CPU-only inference for DeepSeek models [PR 273](https://github.com/ikawrakow/ik_llama.cpp/pull/273)
* March 17 2025: 🚀 FlashMLA-2 performance improvements [PR 253](https://github.com/ikawrakow/ik_llama.cpp/pull/253)
* March 12 2025: Allow `Q8_0` KV cache with FlashMLA-2 on CUDA [PR 265](https://github.com/ikawrakow/ik_llama.cpp/pull/265)
* March 9 2025: 🚀 FlashMLA on CUDA [PR 247](https://github.com/ikawrakow/ik_llama.cpp/pull/247)
* March 8 2025: 🚀 Faster FlashMLA CPU implementation [PR 243](https://github.com/ikawrakow/ik_llama.cpp/pull/243)
* March 3 2025: 🚀 Introducing FlashMLA - MLA with Flash Attention [PR 240](https://github.com/ikawrakow/ik_llama.cpp/pull/240)
* Feb 27 2025: MLA without transposed cache [PR 235](https://github.com/ikawrakow/ik_llama.cpp/pull/235)
* Feb 13 2025: Allow `Q8_0` quantized cache with MLA [PR 206](https://github.com/ikawrakow/ik_llama.cpp/pull/206)
* Feb 11 2025: 🚀 Flash Attention support for DeepSeek models [PR 200](https://github.com/ikawrakow/ik_llama.cpp/pull/200)
* Feb 9 2025: 🚀 MLA for DeepSeek models [PR 188](https://github.com/ikawrakow/ik_llama.cpp/pull/188)

### Fixes

* Fix bug in MMVQ kernel [PR 446](https://github.com/ikawrakow/ik_llama.cpp/pull/446)
* Fix AVX2 implementation of `IQ4_K, IQ4_KS, IQ5_K, IQ6_K` [PR 427](https://github.com/ikawrakow/ik_llama.cpp/pull/427) 
* Fix standard attention on the CPU [PR 421](https://github.com/ikawrakow/ik_llama.cpp/pull/421) 
* Fix imatrix calculation for MLA models [PR 411](https://github.com/ikawrakow/ik_llama.cpp/pull/411)
* Fix new CUDA FA on Touring [PR 413](https://github.com/ikawrakow/ik_llama.cpp/pull/413)
* Fix SER. CPU: [PR 415](https://github.com/ikawrakow/ik_llama.cpp/pull/415) CUDA: [PR 416](https://github.com/ikawrakow/ik_llama.cpp/pull/416)

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
