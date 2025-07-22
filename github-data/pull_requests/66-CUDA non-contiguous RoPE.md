### 🔀 [#66](https://github.com/ikawrakow/ik_llama.cpp/pull/66) - CUDA non-contiguous RoPE

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-09-28 |
| **Updated** | 2024-09-28 |

---

#### Description

In this way we can avoid the Q, K, V copies being made after multiplication with the QKV tensor in, e.g., Phi-3.5-mini (see #65 for details). This results in a 6-7% speedup of PP-512(Phi-3.5-mini) on CUDA (RTX-4080). There is also a 2-3% gain on Metal (M2-Max GPU).

Here is the combined effect of this PR and PR #65 on CUDA (RTX-4080) and Metal (M2-Max 30-core GPU) for Phi-3.5-mini:

| model        | backend    | ngl | threads |          test |   t/s (llama.cpp)    |  t/s (this PR)   |  Speedup |
| -------------| ---------- | --: | ------: | ------------: | -------------------: | ---------------: | -------: |
| phi3 3B F16  | Metal      | 100 |       4 |         pp512 |       1003.22 ± 1.31 |   1063.84 ± 0.63 |  1.060   |
| phi3 3B F16  | Metal      | 100 |       4 |         tg128 |         39.32 ± 0.07 |     41.70 ± 0.06 |  1.061   |
| phi3 3B F16  | CUDA       | 100 |       1 |         pp512 |     11280.47 ± 26.75 | 13770.42 ± 84.46 |  1.221   |
| phi3 3B F16  | CUDA       | 100 |       1 |         tg128 |         79.84 ± 0.03 |     81.50 ± 0.02 |  1.021   |

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2024-09-28** at **12:42:05**:<br>

So, I see that there are a lot of models that can potentially benefit from this PR as the pattern
```
qkv = ggml_mul_mat(...);
Q = ggml_cont(..., qkv, ...);
K = ggml_cont(..., qkv, ...);
V = ggml_cont(..., qkv, ...);
```
is quite common in `llama.cpp`. But replacing the copies that make `Q, K` and `V` contiguous with appropriate views requires testing (it is easy to screw things up), and I don't feel like fetching `N` models and trying at this point. So, for now, just Phi-3(.5) benefits.