### 🔀 [#234](https://github.com/ikawrakow/ik_llama.cpp/pull/234) - Faster MLA on CUDA

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-02-26 |
| **Updated** | 2025-02-27 |

---

#### Description

The CUDA code absolutely does not like MLA. On the main branch MLA attention is in the range of 15-20% slower than the standard attention implementation. The issue is with the `wk_b x q_nope` and `wv_b x qkv_compressed` operations. For TG they require two tensor multiplications of shapes $(N_h \times N_t \times K)$ and $(N_h \times 1 \times K)$, where $N_h$ is the head size, $N_t$ is the number of tokens in the KV cache, and $K$ is the number of heads. These get computed as $K$ consecutive $(N_h \times N_t) \times (N_h \times 1)$ matrix-vector multiplications. To add insult to injury, for `wk_b x q_nope` where `q_nope` is not contiguous, we get $K$ copies (one for each `q_nope` row) to contiguous memory, followed by quantization for a single row (when `wk_b` is quantized), followed by the actual GEMV, i.e., $3 K$ CUDA kernel launches. The associated overhead by far exceeds the time needed for the actual matrix multiplications, so the computation becomes extremely slow compared to what it could be.

This PR fixes the inefficiency by adding a special purpose kernel that performs the $K$ GEMV in one go. It is a bit of a hack and I should try to consolidate with the regular `ggml_cuda_op_mul_mat_vec_q` implementation, but it should do for now. In addition, the PR adds a new  `quantize_tensor_q8_1_cuda` method that operates on non-contiguous tensors that have a single row. This allows the `q_nope` quantization for the `qk_b x q_nope` multiplication to be done with a single call.

These two changes result in a significant speedup of the MLA attention computation on CUDA. For `IQ4_NL` quantized DeepSeek-Lite with all layers processed on the GPU we get a TG-128 increase of 31%. For the hybrid calculations where the experts are computed on the CPU we get a 15% speedup. MLA is now (nearly) on par with standard attention for short contexts and outperforms it with increasing context length. Here is a table comparing standard to MLA attention in this PR for hybrid CPU/GPU inference as a function of context length. The CPU is Ryzen-7950X, and the GPU is RTX-4080

| model                |          test |   t/s (std)      |  t/s (MLA, this PR)|  Speedup |
| -------------------- | ------------: | ---------------: | -----------------: | -------: |
| deepseek2 16B IQ4_NL |    tg64@pp128 |     52.99 ± 0.03 |       52.43 ± 0.04 |  0.989   |
| deepseek2 16B IQ4_NL |    tg64@pp256 |     52.77 ± 0.09 |       52.26 ± 0.07 |  0.990   |
| deepseek2 16B IQ4_NL |    tg64@pp512 |     51.58 ± 1.19 |       51.93 ± 0.10 |  1.007   |
| deepseek2 16B IQ4_NL |   tg64@pp1024 |     50.75 ± 0.56 |       51.73 ± 0.07 |  1.019   |
| deepseek2 16B IQ4_NL |   tg64@pp2048 |     49.96 ± 0.28 |       51.29 ± 0.05 |  1.027   |
| deepseek2 16B IQ4_NL |   tg64@pp4096 |     47.94 ± 0.58 |       50.23 ± 0.05 |  1.048   |
| deepseek2 16B IQ4_NL |   tg64@pp8192 |     43.77 ± 0.34 |       48.04 ± 0.04 |  1.098   |
| deepseek2 16B IQ4_NL |  tg64@pp16384 |     37.76 ± 0.15 |       44.62 ± 0.17 |  1.182   |

---

#### 💬 Conversation

👤 **davidsyoung** commented the **2025-02-27** at **16:17:26**:<br>

@ikawrakow Seeing a significant speed increase from this, with also transposed KV cache. From 12t/s to 17.25t/s, and seeing less of a drop off on speed as well at longer PP tokens. Full CUDA 15x3090 Q2_K MLA.

Really nice!