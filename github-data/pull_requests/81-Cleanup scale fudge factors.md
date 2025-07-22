### 🔀 [#81](https://github.com/ikawrakow/ik_llama.cpp/pull/81) - Cleanup scale fudge factors

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-10-04 |
| **Updated** | 2024-10-04 |

---

#### Description

Low-bit quants often benefit from a fudge factor applied to the (super-)block scale. When I was developing `IQ2_K` and `IQ3_K` it was faster to change the fudge factor in `ggml-cuda/convert.cu` and recompile than to change it in the quantization function and re-quantize. But when I was ready, I forgot to move the `IQ2_K` and `IQ3_K` fudge factors to quantization, so they remained in the CUDA dequantization function (and hence weren't applied anywhere else). This PR fixes this.