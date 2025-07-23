### 🔀 [#518](https://github.com/ikawrakow/ik_llama.cpp/pull/518) - IQ3_S: much faster CPU prompt processing

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-11 |
| **Updated** | 2025-06-12 |

---

#### Description

As PRs #515, #516, #517.

Here a sweep-bench with this PR for LlaMA-3.1-8B on a Ryzen-7950X CPU

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    1.733 |   295.36 |    8.239 |    15.54 |
|   512 |    128 |    512 |    1.805 |   283.62 |    8.398 |    15.24 |
|   512 |    128 |   1024 |    1.857 |   275.73 |    8.561 |    14.95 |
|   512 |    128 |   1536 |    1.905 |   268.74 |    8.430 |    15.18 |
|   512 |    128 |   2048 |    1.954 |   261.97 |    8.563 |    14.95 |

I haven't done this for a while, but I think for this one worth looking at mainline `llama.cpp` (build: `5635 (3069e3169)`)

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |   18.261 |    28.04 |    7.933 |    16.14 |
|   512 |    128 |    512 |   18.708 |    27.37 |    8.335 |    15.36 |
|   512 |    128 |   1024 |   19.048 |    26.88 |    8.547 |    14.98 |
|   512 |    128 |   1536 |   19.480 |    26.28 |    8.739 |    14.65 |
|   512 |    128 |   2048 |   19.670 |    26.03 |    8.912 |    14.36 |

10X faster PP here!