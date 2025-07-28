## 🔀 [Pull Request #112](https://github.com/ikawrakow/ik_llama.cpp/pull/112) - Faster MoE inference

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/multi_add` |
| **Target Branch** | `main` |
| **Created** | 2024-10-31 |
| **Updated** | 2025-06-23 |
| **Merged** | 2024-10-31 |

---

## 📄 Description

This PR
* Adds a new op `GGML_MULTI_ADD` used to sum up the contributions of the selected experts. It results in, e.g., a 7% improvement of token generation speed for Granite-1B-MoE on CUDA (RTX-4080).
* Fixes a massive inefficiency in the Metal implementation of MoE matrix multiplications (`kernel_mul_mm_id`). This leads to a nearly 6-fold prompt processing speedup for Granite-1B-MoE on Metal. But even for a much larger model such as Mixtral-8x7B the speedup is nearly a factor of 2 compared to current mainline `llama.cpp` (build: `8f275a7c (3989)`).

---

## 💬 Conversation

👤 **Nexesenex** commented on **2025-06-23** at **12:59:59**

Hey IK.

```
    if (n_expert_used == 1) {
        return ggml_cont(ctx, ggml_view_2d(ctx, experts, n_embd, n_tokens, experts->nb[2], 0));
    }
    if (n_expert_used == 2) {
        return ggml_add(ctx, ggml_view_2d(ctx, experts, n_embd, n_tokens, experts->nb[2], 0),
                             ggml_view_2d(ctx, experts, n_embd, n_tokens, experts->nb[2], experts->nb[1]));
    }
    return ggml_multi_add(ctx, ggml_view_2d(ctx, experts, n_embd, n_tokens, experts->nb[2], 0), n_expert_used);
```

What of the case if expert_used >= 3?

For example, on Mixtral 8x22b, there's a perplexity benefit to use 3 experts instead of 2 (-2% PPL 512).

---

👤 **ikawrakow** commented on **2025-06-23** at **13:05:43**

Well, if it is not 1 or 2, then we handle it via `multi_add`, which handles adding together any number of contributions.

---

👤 **Nexesenex** commented on **2025-06-23** at **13:08:58**

Oh silly me, I just read too fast the code, I understand now.
Sorry!