### 📝 [#196](https://github.com/ikawrakow/ik_llama.cpp/issues/196) - Refactor: remove usage of Q8_1 for activation quantization

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-02-09 |
| **Updated** | 2025-03-27 |

---

#### Description

### Background Description

Some models can produce activations that are beyond the range of `fp16`. In that scenario, usage of `Q8_1` to quantize the activations can be futile, see discussion in #194.

Hence, it would be prudent to change all quantization types using `Q8_1` for matrix multiplications to use something else.
Alternatively, one may replace the `fp16` block scale and block sum in `Q8_1` with `bf16`. 

### Possible Refactor Approaches

_No response_