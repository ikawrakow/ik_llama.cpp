### 🔀 [#79](https://github.com/ikawrakow/ik_llama.cpp/pull/79) - Do not quantize activations if not necessary

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-10-04 |
| **Updated** | 2024-10-04 |

---

#### Description

It has always bugged me that `ggml` unnecessarily repeats the "quantization" of activations when the corresponding matrix multiplication cannot be done directly. E.g., `Q`, `K` and `V` all multiply the input to the self-attention layer. Similarly, `ffn_up` and `ffn_gate` multiply the same activations for parallel FFNs. "Quantization" is in quotes, because it applies to `fp16` and `bf16` tensors when the matrix multiplication function used does not work directly with `fp32` activations. There are typically 7 tensors per layer in a transformer model, so basically 3 out of 7 "quantizations" are unnecessary.

This PR remedies this unfortunate situation by storing "quantized" activations in a dedicated part of the work buffer (so the data cannot be trashed by other ops that also need a work buffer), and by remembering the name of the last tensor that was quantized. I was hoping that by avoiding the unnecessary quantization we can also skip the thread synchronization barrier that we have in `ggml_compute_forward_mul_mat` after quantization, but I guess I'm missing something because skipping the barrier may hang the inference pipeline, so for now the barrier is still there.

Quantization takes a relatively small fraction of the overall graph evaluation time, so performance gains are typically in the ~1% range. But for a `bf16` model with a long context I'm finding a non-trivial performance improvement when running on a CPU with native `bf16` support (Ryzen-7950X). Here is a comparison for LLaMA-3.1-8B with a context of 8192 tokens

| model                          |       size |     params | backend    | threads | type_k | type_v | fa |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -----: | -----: | -: | ------------: | ---------------: |
| llama 8B BF16 (main)                 |  14.96 GiB |     8.03 B | CPU        |      16 |   bf16 |   bf16 |  1 |        pp8192 |    178.64 ± 0.69 |
| llama 8B BF16 (PR)                 |  14.96 GiB |     8.03 B | CPU        |      16 |   bf16 |   bf16 |  1 |        pp8192 |    188.28 ± 0.49 |
   
5.4% gain in performance is nothing to sneeze at, especially considering how minor the necessary code change is.