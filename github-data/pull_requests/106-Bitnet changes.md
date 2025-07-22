### 🔀 [#106](https://github.com/ikawrakow/ik_llama.cpp/pull/106) - Bitnet changes

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-10-24 |
| **Updated** | 2024-10-25 |

---

#### Description

* Change `IQ1_BN` and `IQ2_BN` to have per row scales. In that way we can handle Bitnet models with and without separate tensor scales
* Remove `IQ1_TN` and `IQ2_TN`. With the above change these are now redundant. `IQ1_BN` and `IQ2_BN` are also faster, so no reason to keep these around
* Change `build_bitnet()` to use the standard `llm_build_kv()` function for the self attention portion. I was hoping this would also allow to use FA, but nope, the Bitnet models have a strange head size of 100 that is not supported by the FA implementations.

Everything works except - can you guess? - Metal. There is something wrong with the dot product kernels and I simply don't see what. I have to fix Metal before merging.

On CUDA (RTX-4080) we now get 368 t/s for TG-128 with the 3.3B Bitnet model (`IQ2_BN`). When I first added Bitnet support we were at ~320 t/s, so quite an improvement since then. 

**Update**

I wasted quite some time trying to figure out why the Bitnet changes don't work on Metal. At the end it turned out that it is PR #98 that breaks the Metal back-end. So, this PR reverts #98.

@agray3 Do you have the ability to investigate why #98 breaks the Metal back-end?