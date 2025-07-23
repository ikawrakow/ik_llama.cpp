### 🐛 [#540](https://github.com/ikawrakow/ik_llama.cpp/pull/540) - Fix missed block_q8_x2 bf16 -> i16 change

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-19 |
| **Updated** | 2025-06-19 |

---

#### Description

See #538

The story behind this bug:

Many years ago, the committee designing the `AVX` instruction set decided to use the most unhelpful instruction for performing dot products between `int8` SIMD vectors: the left operand in the instruction had to be an unsigned integer. That decision propagated into `AVX2` and `AVX512`. When using this in the context of quantized LLMs, where quantized model weights are signed integers, we have two options to deal with this situation: 
1. Remove the signs of the left operand and apply the same signs to the right operand
2. Add a constant to the left operand such that it becomes unsigned. Undo the applied constant by subtracting the constant times the sum of the quants in the right operand

Option 2 is faster, but cannot be used on `AVX2` when the quants span the full `int8_t` range as the dot product produces a SIMD vector with `int16_t` values containing the sum of pairs, and that can overflow (e.g., 255*127 + 255*127). But on `AVX512` the dot product sums 4 products into an `int32_t` avoiding overflow in intermediate results, so we use the faster option 2. For this we have the `Q8_1` type, which contains the block scale and the sum of the quants in the block times the block scale as `fp16`. This worked fine until DeepSeek came along, and we started getting NaNs because the sum was occasionally overflowing the `fp16` range. We then switched to using `Q8_2`, which is the same `Q8_1`, except that block scale and sum are stored as `bf16`, which resolved the NaNs with DeepSeek. But when working on PR #534, I noticed that PPL for `Q4_0` became significantly higher, and that was due to not enough precision in the `bf16` block sum. So, I changed again to have the block sum stored as `int16_t` (which is exact), and then converted to `fp32` at run time. I thought I did adapt all places where `Q8_2` or `Q8_2_X4` is used, but no, I missed one place in the tail of the `Q8_0_R8 x Q8_2_X4` dot product. In that product we go over groups of 4 blocks of 32 quants, and then have a tail handling the leftover. In the vast majority of cases there are no leftovers, but in the DeepSeek FlashMLA, we run into this forgotten corner. The PR fixes that.