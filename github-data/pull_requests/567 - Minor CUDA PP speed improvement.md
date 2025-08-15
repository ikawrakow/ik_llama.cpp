### 🔀 [#567](https://github.com/ikawrakow/ik_llama.cpp/pull/567) - Minor CUDA PP speed improvement

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-07-01 |
| **Updated** | 2025-07-02 |

---

#### Description

I was actually trying to improve MMQ performance for quants with a block-size of 16, but ended up with a small improvement of the MMQ kernel for blocks of 32. Just 1-2% kind of improvement, so nothing earth shattering.

Here a `sweep-bench` graph for LlaMA-3.1-8B on RTX-4080 for `Q4_0` and `IQ4_KS`. The `IQ4_KS` improvement is slightly larger because I added a tweak to the tile loading kernel in addition of taking advantage of the slightly faster tile multiplication kernel. 

 
![u4](https://github.com/user-attachments/assets/26ab1293-3298-4d45-a3fd-6abdbc082bd6)

---

#### 💬 Conversation

👤 **Nexesenex** commented the **2025-07-02** at **03:05:58**:<br>

No problem on my side on Miqu Q5_K_M (full offload w/MMQ on 3 GPUs) and Wizard 8x22b IQ3_S mix (same test) after adapting this PR to Croco.cpp (mainline's fork).
Perfs are similar, with maybe a 0.5-1% bonus (still in the margin of variation of my bench results, but not downward, upward).

Can the iq4_ks versant of that PR be valid on the other quants' MMQ kernels using currently
`const int k0 = 8 * (threadIdx.x / 4) + threadIdx.x % 4;`
such as iq4_xs and iq4_nl?

---

👤 **ikawrakow** commented the **2025-07-02** at **07:11:23**:<br>

> Can the iq4_ks versant of that PR be valid on the other quants' MMQ kernels

Not sure, one needs to try.

Larger gains would come from rewriting the MMQ implementation to have the x-tiles be reused more times. Currently `Q4_0` MMQ is almost 10% faster than `IQ4_KS`. This does not make any sense. Yes, unpacking `IQ4_KS` is more expensive than unpacking `Q4_0`, but one should be able to fully amortize the unpacking cost in large matrix multiplications. This is what happens on the CPU, where all quants using the same unpacked GEMM kernel have the same performance (to within 1-2%).  I think the reason we see this on CUDA is that there all optimizations are made with `Q4_0` as the main optimization target.  As `Q4_0` is very simple, and it costs next to nothing to unpack, the remaining MMQ logic is tailored for very cheap unpacking, to the detriment of all other quantization types.