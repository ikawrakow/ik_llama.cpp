### 🔀 [#53](https://github.com/ikawrakow/ik_llama.cpp/pull/53) - Quantization mixes tweaks

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-09-14 |
| **Updated** | 2024-09-14 |

---

#### Description

This PR changes quantization type selection for some quantization types. This leads to a lower PPL **and** a smaller quantized model size for Gemma-2 models.

The following table shows a comparison between the main branch and this PR for Gemma2-9b in terms of bits-per-weight (bpw) and quantization error (QError) defined as `PPL(Q)/PPL(fp16)-1`

| Type  |  bpw (main) | QError (main) | bpw (PR) | QError (PR) |
| ---: | ---: | ---: | ---: | ---: |
| IQ1_M | 2.20 |  78.04% | 2.15 | 67.55% |
| IQ2_XXS | 2.44 | 41.79% | 2.37 | 38.64% |
| IQ2_XS | 2.65 | 29.58% | 2.58 | 26.72% |
| IQ2_S | 2.77 | 22.12% | 2.68 | 21.82% |
| IQ2_M | 2.97 | 15.22% | 2.87 | 15.12% |
| IQ3_XXS | 3.28 | 8.46% | 3.19 | 8.07% |
| IQ3_S | 3.75 | 4.79% | 3.68 | 3.97% |
| IQ4_XS | 4.48 | 1.56% | 4.42 | 1.33% |

Basically, because Gemma models use the same tensor for token embeddings and output, so it needs to be quantized with more bits, but the tensor is very large because of the large vocabulary, quantized models end up with significantly more bpw for the entire model compared to the bpw of the main quantization type. The idea here is to wuantize `output.weight` with one of the new quantization types (`IQ4_K` for 2- and low-3-bit quantization, `IQ5_K` for the others), and use a higher bpw for the `attn_v` tensor (`IQ3_K`, IQ4_K`, or `IQ5_K`, depending on quantization type).