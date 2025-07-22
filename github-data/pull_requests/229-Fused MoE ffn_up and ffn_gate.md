### 🔀 [#229](https://github.com/ikawrakow/ik_llama.cpp/pull/229) - Fused MoE ffn_up and ffn_gate

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-02-23 |
| **Updated** | 2025-02-23 |

---

#### Description

In all MoE models one has the following sequence of operations as part of the feed forward network (simplified):
```
up   = ggml_mul_mat_id(up_exps, cur, selected_experts)
gate = ggml_mul_mat_id(gate_exps, cur, selected_experts)
act  = ggml_silu(gate) or ggml_gelu(gate)
par  = ggml_mul(up, act)
down = ggml_mul_mat_id(down_exps, par)
```
Each of the `ggml_mul_mat_id` operations requires a search of activated experts (which is the same for all 3). Also, `up` and `gate` have the same second operand so that, if they are quantized, the quantization is unnecessarily repeated. There is a barrier after each operation. On CUDA there is no implementation of indirect matrix multiplication, so each `ggml_mul_mat_id` op triggers a copy of the rows of the second operand to a contiguous memory block, actual matrix multiplication, and then another copy from the contiguous matrix multiplication result to the non-contiguous op result. All of this adds overhead thus reducing performance.

This PR adds new `ggml` op that fuses the `up, gate` and `act` operations. On CUDA, if the next op in the computation graph is the `par` op, it is auto-fused as well. The `down` operation is not included for now, but a future PR may do so.

This is relevant for the performance of the large DeepSeekV3/R1 models. I don't have the means to run DeepSeekV3/R1, hence using DeepSeek-Lite (very similar architecture but only16B parameters with 2.4B active parameters). For this model, we gain ~3-4% in prompt processing (PP) speed and 1-2% for token generation (TG) when running on the CPU. The performance gains are much more significant on CUDA - about 26% speedup for PP and 7% for TG. On my RTX-4080 I now get `PP-512 = 4400 t/s`  for DeepSeek-Lite. This is still much to low compared to a dense model with 2.4B parameters (one should get in the range of 15,000 t/s), but quite a bit better than the 3450 t/s one gets on the main branch (and also in mainline `llama.cpp`). 

As the new op is not implemented on all platforms (Metal is missing), it is enabled via a command line option that is off by default. To turn on, use `-fmoe` or `--fused-moe`.

Obviously this option cannot be used when computing an imatrix because than the intermediate results remain in temporary work buffers, hence will not be propagated to collect activation statistics for the `up_exps` and `gate_exps` tensors.