### 🔀 [#61](https://github.com/ikawrakow/ik_llama.cpp/pull/61) - Adding ability to have meta data per tensor row

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-09-25 |
| **Updated** | 2024-09-27 |

---

#### Description

`ggml` is very opinionated on the topic of tensor data layout - things must be organized in blocks of a known size, the number of elements in a block must be fixed, etc. There are many places where it is assumed that a contiguous tensor row with `ne` elements occupies `ne * ts / bs` bytes, where `ts` is the "type size" and `bs` is the "block size". This is not very useful when one wants to have some meta data per tensor or per row (e.g., tensor or row scale, quant values in a K-means clustering based quantization, etc.).

This PR adds the ability to have per row meta data. As a POC, `IQ1_TN` and `IQ2_TN` are changed to have a row-wise block scale, which reduces the quantized model size to `1.625` (`IQ1_TN`) or `2.0` (`IQ2_TN`) bpw from `1.6875` or `2.0625` bpw.

There are a few places left in the CUDA Flash Attention implementation where the `ne * ts / bs` assumption is used. But as we are not using quants with row meta data for quantized KV cache, this should be OK for now.

This is a breaking change. Previously created `IQ1_TN` and `IQ2_TN` models need to be re-quantized.