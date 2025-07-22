### 🐛 [#415](https://github.com/ikawrakow/ik_llama.cpp/pull/415) - Fix SER (CPU)

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-13 |
| **Updated** | 2025-05-13 |

---

#### Description

There have been reports that Smart Expert Reduction (SER) can produce garbage.

This PR (hopefully and finally) fixes the CPU implementation.

The issue was that when fewer experts are used than specified by the number of active experts, there are some rows in the experts matrix multiplication results that have not been set to any value. Normally this should not be an issue as these rows get multiplied by zero before being summed up to obtain the final experts result. But if there are `Inf` or `NaN` values in the rows that were not computed, then we get NaNs, and this leads to garbage output. If there are `Inf` and `NaN` values is a matter of luck that depends on what happened before in the computation, as the same memory is used by other operations to store results. This is why the issue does not always manifests itself (but yes, if one has a long enough conversation, the `DDDDDDDD` or `GGGGGGGGG` output will eventually show up).

A similar fix is required for the CUDA implementation. This is left for a follow up PR.

---

#### 💬 Conversation

👤 **ubergarm** commented the **2025-05-13** at **15:04:17**:<br>

Hah, our sleep schedules are just off, I just tested this compiling CPU only and it indeed fixes the issue when using `-ser 6,1`.

Without the fix I saw:
```
/home/w/projects/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed/home/w/projects/ik_llama.cpp/ggml/src/iqk/iqk_mul_m
at.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
/home/w/projects/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:16600: GGML_ASSERT(fms.S[j] > 0) failed
```

With this PR it works fine.

Thanks!

I'll peep the CUDA one next.