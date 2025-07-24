### 🔀 [#83](https://github.com/ikawrakow/ik_llama.cpp/pull/83) - New SOTA quantization: 4.25 bpw IQ4_KS

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-10-09 |
| **Updated** | 2024-10-09 |

---

#### Description

It is similar to `IQ4_K` with the following difference
* Blocks of 32 instead of blocks of 16
* Row-wise `float` scale instead of per block instead of per super-block `ggml_half`
* 7-bit block scales instead of 6-bit - needed to ensure enough precision when using per row float scale

It ends up being 4.25 bpw, so the same as `IQ4_XS`. Why add it then? Because it has a lower quantization error than `IQ4_XS`. For some models the difference is quite significant. The following table gives some examples. Quantization error `Qerr` is defined as `PPL(Q)/PPL(f16)-1`

| Model | Qerr(IQ4_XS) | Qerr(IQ4_KS) |
| :------- | ---: | ---: |
| LLaMA-3.1-8B |  2.82% | 2.68% |
| LLaMA-3.1-8B-Instruct | 2.54% | 1.85% |
| LLaMA-3.2-3B-Instruct | 2.45% | 2.13% |
| Qwen-2.5-7B-Instruct | 2.31% | 1.62% |
| Qwen-2.5-32B-Instruct | 2.17% | 1.82% |
| Nemo-Instruct-2407 | 1.592% | 1.579% |
| Gemma-2-9B | 1.33% | 0.92% |
| Gemma-2-27B-Instruct | 1.23% | 0.72% |

Performance is similar to `IQ4_XS` or even slightly better, except for TG on the M2-Max GPU, where it is ~2% slower (Apple Silicon does not like non-sequential memory access, but having the row scale stored at the beginning of the row causes an additional memory jump in the dot product kernel).

The PR also adds a new quantization mix - `IQ3_KL` (`L` for "large"). It fills the gap between `IQ4_K` and `IQ4_K` (and now `IQ4_KS`). The following graph illustrates where this new mix sits for LLaMA-3.1-8B-Instruct.

![il31_8B](https://github.com/user-attachments/assets/5ece2ee2-23e6-4e9e-8502-27c91423a2f9)