### 🐛 [#537](https://github.com/ikawrakow/ik_llama.cpp/pull/537) - Update CMakeLists.txt to fix NDEBUG handling

| **Author** | `iSevenDays` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-18 |
| **Updated** | 2025-06-19 |

---

#### Description

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [x] Low
  - [ ] Medium
  - [ ] High


without my change

| PP  | TG  | N_KV | T_PP s | S_PP t/s | T_TG s | S_TG t/s | | --- | --- | ---- | ------ | -------- | ------ | -------- | ggml_backend_cuda_graph_compute: disabling CUDA graphs due to mul_mat_id ggml_backend_cuda_graph_compute: disabling CUDA graphs due to too many consecutive updates
|  8192 |   2048 |      0 |   54.433 |   150.50 |  414.061 |     4.95 |
|  8192 |   2048 |   8192 |   64.162 |   127.68 |  428.767 |     4.78 |

after my change to CMakeLists.txt

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  8192 |   2048 |      0 |   58.363 |   140.36 |  405.040 |     5.06 |
|  8192 |   2048 |   8192 |   63.752 |   128.50 |  423.548 |     4.84 |
|  8192 |   2048 |  16384 |   69.712 |   117.51 |  431.367 |     4.75 |

---

#### 💬 Conversation

👤 **ikawrakow** submitted a review the **2025-06-19** at **07:18:05**: ✅ `APPROVED`<br>

So, in the latest tool chains someone decided that the `NDEBUG` is not set when making a release build? Contrary to the established practice of the last 30 years?

---

👤 **iSevenDays** commented the **2025-06-19** at **07:32:42**:<br>

Yes, thanks for merging the fix quickly :)