### 🔀 [#71](https://github.com/ikawrakow/ik_llama.cpp/pull/71) - iqk_mul_mat: better srategy when nrc_y not divisible by ny

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-10-01 |
| **Updated** | 2024-12-09 |

---

#### Description

In the llamafile repository @Djip007 has posted [PP results](https://github.com/Mozilla-Ocho/llamafile/discussions/549#discussioncomment-10780156) for short prompt lengths in steps of 1, and one sees a sharp drop in performance for 9 tokens for `Q6_K` and `Q5_K_M`. Why? For these quants llamafile uses `iqk_mul_mat` that I have contributed there, so the matrix multiplication is done using 1x8 tiles. The way it is implemented there (and also here on the main branch) is that first we multiply with 8 columns from the right matrix and then have a second pass to multiple with the remaining 9th column. This second pass is much slower, so overall performance drops. I was of course aware that there will be this effect, and always meant to investigate it, but never did. Now that we have it published, it is time to fix it via this PR.

When the number of columns `N` in the right matrix is not divisible by the maximum tile size `n_max`, a better strategy for performing the matrix multiplication is this:
* `M = (N + n_max - 1)/n_max` is the number of passes we need for the full matrix multiplication (loops over B-columns tiles)
* Let `n = N/M` (integer division). We will take `m` passes with a tile size of `n`, and `(M - m)` passes with a tile size of `n+1`
* `n*m + (n+1)*(M-m)` must be equal `N`, so we get `m = M * (n+1) - N`

This strategy is implemented in this PR. The following graph shows performance (tokens per second) for LLaMA-3.2-3B as a function of prompt length for the main branch (black) and this PR (red). This is for a `bf16` model where the tile size is `5 x 5`, so we see the main branch being equivalent to this PR for prompt length <= 5 (single pass) and then for 10, 15, 20, 25 and 30 tokens, but being significantly lower for prompt lengths that are not a multiple of 5. The PR shows a nice smooth increase in performance as one would expect.

![iqk_strategy](https://github.com/user-attachments/assets/bb776c03-3a9f-4358-b2f4-b5b9b2f2fc43)

---

#### 💬 Conversation

👤 **Djip007** commented the **2024-11-26** at **19:09:21**:<br>

I what thinking to do something for that to (on tinyBLAS) but not that way. Good to see that it work, I may use it in some other case... 
Good JOB!

Will you do the same on tinyBLAS for non the other case (FP16/BF16/...) ?

---

👤 **ikawrakow** commented the **2024-11-27** at **15:34:24**:<br>

> Will you do the same on tinyBLAS for non the other case (FP16/BF16/...) ?

In my case all matrix multiplications are driven by the same function, so this change benefits all types. I think in tinyBLAS one needs to do it for every version of `mnpack`

---

👤 **Djip007** commented the **2024-12-09** at **22:08:55**:<br>

OK I think I figure how to do it for FP16/BF16/FP32 on tinyblas...
https://github.com/Mozilla-Ocho/llamafile/discussions/654

some bench are WIP but for now it look good.