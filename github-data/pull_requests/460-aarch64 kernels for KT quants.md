### 🔀 [#460](https://github.com/ikawrakow/ik_llama.cpp/pull/460) - aarch64 kernels for KT quants

| **Author** | `andrewkchan` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-26 |
| **Updated** | 2025-05-30 |

---

#### Description

This adds aarch64 kernels for the KT quants added in https://github.com/ikawrakow/ik_llama.cpp/pull/441.

All benchmarks are done on my 14-inch 2023 M3 Macbook Pro with 6 threads on Llama-3.1-8B-Instruct.

**Performance sweeps:**

IQ2_KT:

PP | TG | N_KV | T_PP s | S_PP t/s | T_TG s | S_TG t/s
-- | -- | -- | -- | -- | -- | --
512 | 128 | 0 | 8.925 | 57.37 | 40.254 | 3.18
512 | 128 | 512 | 8.301 | 61.68 | 43.609 | 2.94
512 | 128 | 1024 | 8.035 | 63.72 | 36.382 | 3.52
512 | 128 | 1536 | 7.037 | 72.76 | 40.407 | 3.17
512 | 128 | 2048 | 10.026 | 51.07 | 32.519 | 3.94

IQ3_KT:

PP | TG | N_KV | T_PP s | S_PP t/s | T_TG s | S_TG t/s
-- | -- | -- | -- | -- | -- | --
512 | 128 | 0 | 11.348 | 45.12 | 69.893 | 1.83
512 | 128 | 512 | 9.895 | 51.74 | 37.603 | 3.40
512 | 128 | 1024 | 8.937 | 57.29 | 42.072 | 3.04
512 | 128 | 1536 | 10.940 | 46.80 | 36.691 | 3.49
512 | 128 | 2048 | 9.552 | 53.60 | 36.397 | 3.52

IQ4_KT:

PP | TG | N_KV | T_PP s | S_PP t/s | T_TG s | S_TG t/s
-- | -- | -- | -- | -- | -- | --
512 | 128 | 0 | 8.022 | 63.83 | 60.247 | 2.12
512 | 128 | 512 | 8.473 | 60.42 | 54.940 | 2.33
512 | 128 | 1024 | 8.174 | 62.64 | 48.575 | 2.64
512 | 128 | 1536 | 9.337 | 54.84 | 47.700 | 2.68
512 | 128 | 2048 | 9.766 | 52.43 | 142.519 | 0.90

For comparison, I get ~18.3 t/s on IQ2_K, so it is considerably slower, but maybe still acceptable. Metal kernels should be better!

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [X] Low
  - [ ] Medium
  - [ ] High