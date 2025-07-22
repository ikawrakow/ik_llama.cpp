### 🔀 [#404](https://github.com/ikawrakow/ik_llama.cpp/pull/404) - TG improvements for MoE models

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-10 |
| **Updated** | 2025-05-10 |

---

#### Description

This PR does 3 things:
* Removes an unnecessary device to host copy of selected experts IDs on CUDA. This results in a few percent improvement of CUDA TG speed for MoE models
* Fixes bugs related to Smart Experts Reduction (SER, see #239). The issue was that the `GGML_OP_GET_ROWS` op implementation did not consider disabled experts for float tensors. As a result, when combining the results of the experts garbage weights were used for the disabled experts, which could lead to NaNs.  
* Further improves CUDA TG performance with SER enabled. Here the `ggml_cuda_op_mul_mat_vec_q_id` function did not consider that an expert may be disabled, and needlessly calculated the matrix-vector multiplication for disabled experts.

Prompt processing is not eaffected by these changes.

Here is a graph obtained with `sweep-bench` showing TG performance as a function of the number of tokens in the KV cache `N_KV`. The model is DeepSeek-Lite quantized to `Q4_0`. The GPU is RTX-4080. Black symbols are without using SER, red symbols are with `-ser 4,1`. The command line is
```
./bin/llama-sweep-bench -m $model -t 1 -ngl 100 -fmoe -mla 3 -fa -b 4096 -ub 4096 [-ser 4,1]
```
  
![z8](https://github.com/user-attachments/assets/e6408f60-63dc-438d-824c-4bee9bb5120e)