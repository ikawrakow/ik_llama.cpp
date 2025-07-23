### 🔀 [#56](https://github.com/ikawrakow/ik_llama.cpp/pull/56) - BF16 support on Metal

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-09-16 |
| **Updated** | 2024-09-17 |

---

#### Description

It is slightly slower than `fp16`, but definitely a massive improvement compared to not having `bf16` support at al. ~Didn't put any effort into optimizing the matrix x vector kernel, so it is likely one can improve `bf16` TG performance~. 

| model                          |       size |     params | backend    | ngl |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------------: | ---------------: |
| llama 8B BF16                  |  14.96 GiB |     8.03 B | Metal      | 100 |         pp512 |    538.84 ± 0.26 |
| llama 8B F16                   |  14.96 GiB |     8.03 B | Metal      | 100 |         pp512 |    587.26 ± 0.39 |
| llama 8B BF16                  |  14.96 GiB |     8.03 B | Metal      | 100 |         tg128 |     21.64 ± 0.05 |
| llama 8B F16                   |  14.96 GiB |     8.03 B | Metal      | 100 |         tg128 |     21.77 ± 0.03 |