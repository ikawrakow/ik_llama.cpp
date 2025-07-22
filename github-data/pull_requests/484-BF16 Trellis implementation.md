### 🔀 [#484](https://github.com/ikawrakow/ik_llama.cpp/pull/484) - BF16 Trellis implementation

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-02 |
| **Updated** | 2025-06-19 |

---

#### Description

This PR adds a `bf16` CPU implementation for the trellis quants `IQ2_KT, IQ3_KT` and `IQ4_KT` for CPUs with native `bf16` support.

We get massive gains in prompt processing speeds, and a ~5-10% gain in TG performance. On my Ryzen-7950X CPU that supports `bf16`, all 3 types now have PP-512 in the range of 230-240 t/s for 8B LLaMA-3. This makes them comparable to row-interleaved quants (where PP-512 performance on this CPU is in the 240-300 t/s range).

TG-128 performance for 8B LlaMA-3 on the Ryzen-7950X changes as follows

| type | f32 t/s | bf16 t/s|
|---: | ---: | ---: |
| IQ2_KT | 12.17 | 12.65 |
| IQ3_KT | 10.54 | 11.22 |
| IQ4_KT | 8.39 | 9.45 |

PP-512 performance for 8B LlaMA-3 on the Ryzen-7950X changes as follows

| type | f32 t/s | bf16 t/s|
|---: | ---: | ---: |
| IQ2_KT | 132.47 | 233.96 |
| IQ3_KT | 127.80 | 233.37 |
| IQ4_KT | 126.31 | 243.17 |

A similar optimization can be done for CPUs with native `fp16` support, but as I don't have access to one of those, this is not implemented for now.

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-06-03** at **04:22:51**:<br>

Thank for testing.

Yes, this assert is always associated with a NaN somewhere else. I ran into NaNs with the `fp16` implementation on NEON, and had to be extra careful with under- and overflows and what needs to be computed with `fp32`. But I wouldn't have thought there could be similar issues with `bf16`.

Looking at the low GPU TG performance, my guess is that you need to explicitly enable `F16` on CUDA (`cmake -DGGML_CUDA_F16=ON`).

---

👤 **ikawrakow** commented the **2025-06-03** at **07:10:14**:<br>

I hadn't tested this PR with a DeepSeek model. Testing now I see DeepSeek-Lite breaks with `bf16` precision. I don't get NaNs but I get extremely high perplexity values and gibberish in TG.

---

👤 **ikawrakow** commented the **2025-06-19** at **07:26:25**:<br>

Closing in favor of #529