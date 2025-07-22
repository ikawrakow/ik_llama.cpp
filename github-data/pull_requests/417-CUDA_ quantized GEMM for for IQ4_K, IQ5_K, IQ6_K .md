### 🔀 [#417](https://github.com/ikawrakow/ik_llama.cpp/pull/417) - CUDA: quantized GEMM for for IQ4_K, IQ5_K, IQ6_K 

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-14 |
| **Updated** | 2025-05-14 |

---

#### Description

This PR follows in the footsteps of #374, and is the next step towards complete implementation of quantized matrix multiplications (a.k.a. MMQ) for the `IQX_K` quants.

We get in the range of 15% performance improvement compared to the existing implementation that dequantizes to `fp16` and then uses cuBLAS to perform the matrix multiplications.

Another benefit is avoiding the numerical issues observed for DeepSeek models when using `fp16` arithmetic (see #261). It also potentially leads to CUDA compute buffer size reduction because the intermediate buffer for the dequantized tensor is not required. 

I have reused the existing matrix multiplication kernels, providing only the unpacking of the quantized data into the tiles used in the kernels. As such, performance is largely determined by the kernel (blocks of 16 or blocks of 32), and the unpacking cost (converting the packed data into `int8_t` values ready for matrix multiplications). This is best illustrated with the following graph. Model is LLaMA-3.1-8B, GPU is RTX-4080. All quantizations are done using `--output-tensor-type q6_K --pure`.

`Q4_0` is the fastest (black circles). It uses a "type-0" kernel for a block size of 32. Next is `IQ4_KS` (red circles), which uses the same kernel as `Q4_0`. The ~10% lower performance is due to the higher unpacking cost. Next is `Q3_K` (green circles), which has low unpacking cost (at least when compared to `IQX_K` quants), but uses the kernel for a block size of 16. We see a ~30% drop in performance compared to `Q4_0` because of that. Then come the `IQ4_K` (blue circles), `IQ5_K` (magenta circles) and `IQ6_K` (cyan circles) in this PR. They all use the kernel for block size 16, but are ~7-9% slower than `Q3_K` due to the higher unpacking cost. `IQ4_K, IQ5_K` and `IQ6_K` on the main branch are shown with squares in corresponding colors to illustrate the performance gain in this PR. The matrix multiplication kernels are inherited from mainline `llama.cpp`. Based on the graph, it would make sense to try to optimize two aspects of these kernels:
* As `Q4_0` receives a huge amount of attention in `llama.cpp`, most likely the block size 32 kernel was optimized for it. `Q4_0` is a very simple quant, so unpacking cost is (nearly) negligible. When unpacking host is high, it makes sense to reuse a tile more times to amortize the unpacking cost. This is what I have done in the CPU implementation where most quantization types are on par with `Q4_0` (or even outperform it)
* The ~30% drop in performance for blocks of 16 does not seem reasonable. In the CPU implementation quants with blocks of 16 are at most ~10% slower than quants using blocks of 32

Such efforts are left for a future PR.     
  
![z16](https://github.com/user-attachments/assets/b970d150-d0a3-4c37-896d-d3db7a4fe2a1)

---

#### 💬 Conversation

👤 **ubergarm** commented the **2025-05-14** at **14:42:37**:<br>

This is great to see the CUDA performance of the new iqX_k quants relative to each other.

Appreciate the speed boost, can confirm my mixed Qwen3-30B-A3B quant just got faster PP with this PR:

![ik-pr417-sweep-bench](https://github.com/user-attachments/assets/9c27032e-551b-4a51-a374-5ccba823fd10)

- type  f32:  241 tensors
- type q8_0:    6 tensors - token_embd, output, and I juiced `blk.0.attn_*` to q8_0 for funzies given lowest cosine similar score
- type iq4_k:   96 tensors - ffn_(gate|up)_exps
- type iq5_k:   48 tensors - ffn_down_exps
- type iq6_k:  188 tensors - balance of attn_*