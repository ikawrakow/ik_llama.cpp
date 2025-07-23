### 🔀 [#76](https://github.com/ikawrakow/ik_llama.cpp/pull/76) - iq4_nl: faster quantization

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-10-02 |
| **Updated** | 2024-10-02 |

---

#### Description

Speeds up CPU flash attention using `IQ4_NL`.

**Of note**: I noticed `Q8_0` cannot be used for V-cache when head size is not divisible by 128. This is because of
* My change to `quantize_row_q8_0` to store data in groups of 4 blocks. This speeds up legacy quants and `IQ4_NL` matrix multiplications
* The fact that when `V` is stored into the cache, it is treated as being a contiguous 2D tensor. As a result, the groups-of-4 storage strategy is applied. But when used in FA, the `V` tensor is viewed as a non-contiguous 3D tensor with second and third dimension permuted, so for heads that are not a multiple of 128, data in groups-of-4 ends up in different heads. 

To fix this, one would need to
* Revert the change to `quantize_row_q8_0`
* Introduce a new quantization type for usage as the vector dot type of legacy quants and `IQ4_NL` where data is stored in groups-of-4.
* Remember to use this new type rather than `Q8_0` for K-cache, as groups of 4 is exactly  what we need for the K-cache to have a more performant implementation.

I don't like this, so will not do.

Considering that the CUDA FA implementation does not support `Q8_0` for heads other than 128, I think it is OK to have this limitation on `Q8_0` usage for V-cache in the CPU implementation. From my not very thorough experimentation, it seems better/no quantization for K-cache is much more important. In the few models I tried, `Q8_0` for K-cache and `IQ4_NL` for V-cache beets `Q5_1` for K- and V-cache by a significant margin while using only 8% more memory.