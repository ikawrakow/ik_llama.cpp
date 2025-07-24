### 🔀 [#233](https://github.com/ikawrakow/ik_llama.cpp/pull/233) - Slightly faster CUDA MLA

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-02-26 |
| **Updated** | 2025-02-27 |

---

#### Description

The CUDA code absolutely does not like MLA. The issue is with the `wk_b x q_nope` and `wv_b x qkv_compressed` operations. For TG they require two tensor multiplications of shapes $(N_h \times N_t \times K)$ and $(N_h \times 1 \times K)$, where $N_h$ is the head size, $N_t$ is the number of tokens in the KV cache, and $K$ is the number of heads. These get computed as $K$ consecutive $(N_h \times N_t) \times ($N_h \times 1)$ matrix-vector multiplications. To add insult to injury, for `wk_b x q_nope` where `q_nope` is not contiguous, we get $K$ copies (one for each `q_nope` row) to contiguous memory, followed by quantization for a single row (when `wk_b` is quantized), followed by the actual GEMV, i.e., $3 K$ CUDA kernel launches. The associated overhead by far exceeds the time needed for the actual matrix multiplications, so the computation becomes extremely slow compared to what it could be.

This PR improves the situation slightly by making `q_nope` contiguous before `ggml_mul_mat(ctx, wk_b, q_nope)`. For DeepSeek-Lite we gain ~7% in performance when running the entire model on the GPU, and about 4% when running experts on the CPU and everything else on the GPU.

I did attempt to implement a computation of the entire tensor multiplication with a single kernel launch, but I'm failing so far. The TG speed is improved and matches standard attention performance, but I get gibberish output (and so far haven't seen what is wrong). So, for now, adding just this relatively minor improvement.

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-02-26** at **17:27:37**:<br>

Closing in favor of #234

---

👤 **davidsyoung** commented the **2025-02-27** at **16:16:55**:<br>

@ikawrakow Seeing a significant speed increase from this, with also transposed KV cache. From 12t/s to 17.25t/s, and seeing less of a drop off on speed as well at longer PP tokens. Full CUDA 15x3090 Q2_K MLA.

Really nice!