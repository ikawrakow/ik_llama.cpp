## 🔀 [Pull Request #181](https://github.com/ikawrakow/ik_llama.cpp/pull/181) - Various

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/bench_gp` |
| **Target Branch** | `main` |
| **Created** | 2025-01-29 |
| **Updated** | 2025-01-29 |
| **Merged** | 2025-01-29 |

---

## 📄 Description

PR started by me adding the `-gp` option to `llama-bench` as per https://github.com/ggerganov/llama.cpp/pull/11126 because I wanted to test TG performance after a long prompt to be able to compare to the MLA attention implementation in  https://github.com/ggerganov/llama.cpp/pull/11446.

But then I noticed that the repacked `Q8_0` and `Q4_0` quants do not work for row tensor sizes that are not a multiple of 128 (4 x block size of 32), which is the case for some of the tensors in Deepseek2-Lite that I used for testing, so I fixed that.

And than I was comparing performance after the fix on `Llama-3.2-1B`, and noticed that FA with `Q8_0` K-cache does not work.  `Llama-3.2-1B` has a head size of 64 and there was a comment in the code that `Q8_0` does not work for a head sizes less than 128, so I fixed that as well.