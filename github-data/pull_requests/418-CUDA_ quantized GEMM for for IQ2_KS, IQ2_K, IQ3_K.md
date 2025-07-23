### 🔀 [#418](https://github.com/ikawrakow/ik_llama.cpp/pull/418) - CUDA: quantized GEMM for for IQ2_KS, IQ2_K, IQ3_K

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-14 |
| **Updated** | 2025-05-15 |

---

#### Description

This PR is a follow up of #417 and (almost) completes the quantized matrix multiplication (a.k.a. MMQ) implementation for `IQX_K` quants. The only one missing is `IQ4_KSS`, but I don't think I'll do that one as the packing is much too complicated.

There are larger performance gains for `IQ2_KS`  (~35%) than for `IQ2_K` and `IQ3_K` (~10%). This is due to `IQ2_KS` having blocks of 32 and thus being able to use the more efficient GEMM kernel (see discussion in #417).

The graph illustrates the performance improvements for the same setup as in #417. 

![z17](https://github.com/user-attachments/assets/5aac9e16-569a-4d02-9001-8c76965bd7a6)

Looking at this graph and in the graph in #417, I almost feel like adding `IQ3_KS` and `IQ5_KS` as 3- and 5-bit quants with blocks of 32.

---

#### 💬 Conversation

👤 **ubergarm** commented the **2025-05-14** at **19:24:21**:<br>

Wow the IQ2_KS improved around 35%!? The 32 block `_KS` variants have a nice speedup. 

I'd probably try out the larger IQ3_KS and especially IQ5_KS for some mixes in the future if you decide to add them.