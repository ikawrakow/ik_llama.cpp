### 🔀 [#173](https://github.com/ikawrakow/ik_llama.cpp/pull/173) - More Flash Attention improvements

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-01-19 |
| **Updated** | 2025-01-20 |

---

#### Description

This PR further improves the Flash Attention implementation as follows:
* Slightly faster `V * softmax(K * Q)` implementation. This benefits all V-cache types
* Faster implementation when the K-cache is quantized with `Q8_0` via run-time-repacking to `Q8_0_R4`.

The following graph shows prompt processing speed as a function of prompt length for LLaMA-3.1-8B quantized with `IQ4_XS` on a Ryzem-7950X CPU. The PR results are shown with black (`BF16` KV-cache) and red (`Q8_0` KV-cache) triangles, circles are used for the main branch.  I have reused the graph from the last post in #25 by just adding the results for this PR, so mainline `llama.cpp` performance is shown as well. I'm particularly pleased with the fact that `Q8_0` KV-cache is now on per or even slightly better than the natively supported 16-bit float type as `Q8_0` quantized KV-cache is basically lossless while reducing required memory by 2X.

For reference, with a `Q8_K_R8`-quantized model we achieve 380 t/s for 512 tokens, and 150 t/s for 32k tokens.   

![pp512_vs_ctx](https://github.com/user-attachments/assets/cc1e7ce5-c596-47b0-a56a-912a196d2e38)

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-01-20** at **06:57:27**:<br>

Here is the performance relative to a GPU (RTX-4080) for the above graph

![pp_gpu_vs_cpu1](https://github.com/user-attachments/assets/b103b599-b4e6-4775-8c2a-b7fff69fe61c). We observe the ratio now decreasing with increasing prompt length $\Rightarrow$ the utilization of available FLOPs in the FA implementation is now better on the CPU compared to the GPU.