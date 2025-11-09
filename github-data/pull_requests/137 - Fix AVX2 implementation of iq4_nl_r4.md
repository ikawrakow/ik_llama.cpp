## 🔀 [Pull Request #137](https://github.com/ikawrakow/ik_llama.cpp/pull/137) - Fix AVX2 implementation of iq4_nl_r4

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/fix_avx2_iq4_nl_r4` |
| **Target Branch** | `main` |
| **Created** | 2024-12-11 |
| **Updated** | 2024-12-11 |
| **Merged** | 2024-12-11 |

---

## 📄 Description

The implementation was using  `_mm256_maddubs_epi16`, which overflows (and gets saturated) with the unsigned version of the non-linear quants `IQ4_NL` lookup table. This PR fixes it without a noticeable performance loss.