### 🔀 [#141](https://github.com/ikawrakow/ik_llama.cpp/pull/141) - Q8_K_R8: Fastest quantized matrix multiplications

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-12-14 |
| **Updated** | 2024-12-14 |

---

#### Description

This PR adds `Q8_K_R8` - 8-rows interleaved version of `Q8_K`. With that, we break the world record in prompt processing speed. Here is what we get for PP-512 with LLaMA-3.1-8B on `Zen4` (Ryzen-7950X), `AVX2` (Ryzen-5975WX) and `ARM_NEON` (M2-Max):

| Platform | PP-512 (Q8_0_R4) | PP-512 (Q8_K_R8) | Speedup |
| ---: | ---: | ---: | ---: |
| ARM_NEON | 128.29 ± 1.50 | 172.52 ± 4.17 | 1.345 |
| Zen4            | 268.98 ± 0.31 | 368.85 ± 0.73 | 1.371 |
| AVX2            | 234.40 ± 0.60 | 293.72 ± 0.34 | 1.253 | 

On the Ryzen-7950X, which provides native `bf16` support, this is nearly 60% faster than `bf16`. On the M2-Max, which has native `fp16` support, `Q8_K_R8` is 87% faster than `fp16`!
  
**Note on AVX2**: In the `AVX2` implementation one needs to use the `_mm256_madd_epi16(x, y)` instruction, where `x` holds unsigned 8-bit integers and `y` has signed 8-bit integers. In the initial implementation I forgot for the 177'th time that the unsigned integers still need to be within `0...127`, else adding up two adjacent products (as the instruction does) may overflow the `int16_t` range (and gets silently truncated if it does), so I was making the `Q8_K_R8` quants unsigned (simply `xor 0x80`). This implementation resulted in 354 t/s on the Ryzen-5975WX. Sadly, one needs to "unsign" the `Q8_K_R8` quants with `_mm256_sign_epi8(x, x)`, and then apply the sign to the activation quants before taking the dot product. This is quite costly and `AVX2` performance drops to 293 t/s. Being curious about the effect that the `int16_t` overflow might have, I computed LLaMA-3.1-8B-Instruct perplexity (context 512 tokens) with the original and with the correct implementation. I get `PPL = 7.3725` with the overflowing variant, and `PPL = 7.3443` with the correct implementation. I.e., the effect is small but noticeable.