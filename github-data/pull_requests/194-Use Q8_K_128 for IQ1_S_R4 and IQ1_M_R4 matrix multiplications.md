### 🔀 [#194](https://github.com/ikawrakow/ik_llama.cpp/pull/194) - Use Q8_K_128 for IQ1_S_R4 and IQ1_M_R4 matrix multiplications

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-02-08 |
| **Updated** | 2025-02-09 |

---

#### Description

@saood06 is still observing NaNs for DeepSeek-R1 quantized with `IQ1_S_R4`. As I don't see what else could be wrong, I'm making the following hypothesis:

1. Given the discussions about DeepSeek-R1 becoming "dumb" when `fp16` is used for some of the attention tensors, I hypothesize that there are activations that go beyond the range of `fp16` floats, which get truncated when converted from `fp32` for `fp16` for multiplications with some `fp16` model tensor.
2. If this is the case, using `Q8_1` as quantization type for activations, as `IQ1_S_R4` does, can be futile:
  * Suppose there is some block of 32 activations that has a maximum $x_{\rm max} > {\rm f16}_{\rm max}$
  * Suppose that the block scale $d = x_{\rm max}/127$ is in the `f16`  range. This is likely to be the case as `Q8_0` attention tensors are reported to behave better than `fp16`.
  * In `Q8_1` we also compute $s = d \sum q_i$, where $q_i$ are the 8-bit quants. The scaled sum $s$ is also stored as `fp16`. If one gets unlucky, it can overflow, despite $d$ being in range
  * If this occurs, we will get a completely bogus result for the `IQ1_S_R4` dot product with this block. To make the calculation more efficient on `AVX2`, we use ternary quants $0, 1, 2$ (instead of $-1, 0, 1$) to multiply the Q8 quants (so we can use `_mm256_maddubs_epi16`) , and then recover the correct result by subtracting $s$ from the result. But if $s$ is wrong (truncated because outside the `fp16` range), this does not work and we get a wrong result.
 
To test this hypothesis, this draft PR uses `Q8_K_128` for `IQ1_S_R4` and `IQ1_M_R4` matrix multiplications. `Q8_K_128` is a new 8-bit  quantization type similar to `Q8_K` but with blocks of 128 (so I can test with DeepSeek-Lite). It is draft because I haven't done the `ARM_NEON` implementation. `Q8_K_128` uses a 32-bit float scale, and the sums over blocks of 32 are stored as `int16_t` without multiplying with $d$, hence we cannot run into 16-bit float range issues. Perplexity for DeepSeek-Lite is slightly lower compared to using `Q8_1`, which indicates that there may be non-fatal truncation effects also there (normally one expects a slightly higher accuracy from using `Q8_0` or `Q8_1` because of the smaller block size).

Would appreciate if this gets tested with DeepSeek-R1.

---

#### 💬 Conversation

👤 **saood06** commented the **2025-02-08** at **21:39:38**:<br>

@ikawrakow 
>Would appreciate if this gets tested with DeepSeek-R1.

Done.

[1]3.7099,[2]4.6162,[3]3.5438,[4]3.4199,[5]3.5375,[6]3.5710,[7]3.5428,[8]3.6748,[9]3.7417,[10]3.6724,[11]3.7879,[12]3.9602,[13]4.0477,[14]4.1439,[15]4.2809,[16]4.1981,[17]4.3853,[18]4.5141,[19]4.4493,[20]4.3848,[21]4.4664,[22]4.3290,[23]4.1912,[24]4.1799,[25]4.0693,[26]4.0135,[27]4.0672,[28]4.0459,[29]4.1110,[30]4.1116,[31]4.1261,[32]4.1192,[33]4.1756,[34]4.2340,[35]4.3112,[36]4.3722,[37]4.3822,[38]4.4260,[39]4.4568,[40]4.5164,[41]4.5661,[42]4.5563,[43]4.5975,[44]4.5821,[45]4.6738,[46]4.7199,[47]4.7029,[48]4.6934,[49]4.6900,[50]4.7087,[51]4.7637,[52]4.7736,[53]4.8515,[54]4.8776,[55]4.9119,[56]4.9504,[57]4.9769,[58]5.0124,[59]5.0024,[60]5.0545,[61]5.1015,[62]5.1639,[63]5.2095,[64]5.2599,

No more `NaN`'s, nice! It's impressive how quickly you found the race condition and this issue.

---

👤 **ikawrakow** commented the **2025-02-09** at **06:02:29**:<br>

Thank you for this! The decisive hint to solve it was the discussion about DeepSeek-R1 being dumb with `fp16` attention tensors that you alerted me to.