### 🔀 [#247](https://github.com/ikawrakow/ik_llama.cpp/pull/247) - FlashMLA on CUDA

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-03-08 |
| **Updated** | 2025-03-09 |

---

#### Description

This PR adds FlasMLA on CUDA. It is enabled via `-mla 2 -fa`.

I observe a very strange slow down for TG that is caused by a very slow `ffn_gate_exps` matrix multiplication. As I was not able to resolve what causes this, for now TG will got via the regular `mla = 2` route, so TG performance remains the same as we had with `mla = 2, fa = 0`.

Prompt processing speed is massively improved for long contexts, and is almost on par with standard FA. The following table shows a comparison between `mla = 2` without FA and FlashMLA. Model is `IQ4_NL` quantized DeepSeek-Lite, GPU is RTX-4080. `fmoe` is on, `u_batch = 2048`

| model                | mla | fmoe |          test |    t/s (no FA)   |   t/s (FlashMLA) |  Speedup |
| ---------------------| --: | ---: | ------------: | ---------------: | ---------------: | -------: |
| deepseek2 16B IQ4_NL |   2 |    1 |         pp512 |  4027.80 ± 63.97 |  4529.65 ± 73.42 |  1.124   |
| deepseek2 16B IQ4_NL |   2 |    1 |        pp1024 |  5304.63 ± 32.33 |  6228.89 ± 46.93 |  1.174   |
| deepseek2 16B IQ4_NL |   2 |    1 |        pp2048 |  5841.37 ± 10.99 |  7684.09 ± 27.38 |  1.315   |
| deepseek2 16B IQ4_NL |   2 |    1 |        pp4096 |  5013.22 ± 12.50 |  7176.75 ± 28.25 |  1.432   |
| deepseek2 16B IQ4_NL |   2 |    1 |        pp8192 |   4006.03 ± 6.73 |  6400.43 ± 17.39 |  1.600   |
| deepseek2 16B IQ4_NL |   2 |    1 |       pp16384 |   2883.92 ± 8.53 |  5216.29 ± 20.36 |  1.809   |

The KV cache is the same size as `mla = 2` without FA (i.e., the smallest possible). One no longer needs to worry about controlling the maximum compute buffer size via `-amb`.

**Caveats:**
* Only `f16` KV cache can be used for now. As explained in PR #246 we need to convert the KV cache to `fp32` to be able to do the required operations, and the CUDA back-end does not yet support this conversion for quantized data types.
* There is an avoidable increase in compute buffer size that is proportional to the maximum context length (to hold the KV cache converted to `f32` and other intermediate results. This is required on every GPU that performs attention computations.  For DeepSeek-Lite and context length of 32k tokens the CUDA compute buffer is 1404 MiB. It shuldn't be much bigger for DeepSeekV3/R1.

---

#### 💬 Conversation

👤 **davidsyoung** commented the **2025-03-08** at **23:33:14**:<br>

Thank you very much for this. Working on getting layers balanced best I can to give this a proper run. Will report back.

---

👤 **saood06** commented the **2025-03-09** at **03:49:55**:<br>

@davidsyoung I actually just realized for your setup you might be able to fit the AWQ version of Deepseek R1, with a tensor parallel of 16 using [sglang](https://github.com/sgl-project/sglang), it would be interesting to see how the performance compares as it is that is actually the recommend backed for DeepSeek, and they now have Multi-token prediction support with speculative decoding which is an optimization that is not present here (and would actually require another change to the GGUF as the MTP layer is not in the current GGUF file (similar to the situation with the tensors added for MLA attention).

---

👤 **davidsyoung** commented the **2025-03-09** at **08:56:11**:<br>

> @davidsyoung I actually just realized for your setup you might be able to fit the AWQ version of Deepseek R1, with a tensor parallel of 16 using [sglang](https://github.com/sgl-project/sglang), it would be interesting to see how the performance compares as it is that is actually the recommend backed for DeepSeek, and they now have Multi-token prediction support with speculative decoding which is an optimization that is not present here (and would actually require another change to the GGUF as the MTP layer is not in the current GGUF file (similar to the situation with the tensors added for MLA attention).

It’s very possible! It depends on how much additional usage there is outside of the AWQ itself. From quick check, with my 16x3090 I have 384gb VRAM, whereas the AWQ file from looking on HF is 365gb. That could just about fit, but unsure of the possibility with additional usage. 

I’m currently away from server at the moment until Mon/Tues, and I’ll see if I can load it then. The way vLLM loads on the GPUs at the same time causes transient spikes across all cards, which is pretty hard to control. 

It’s possible it could be fine, but being away from server means there’s a chance I can’t restart it without a hard reset so physical access is important 😄 

But, tbh, at the rate @ikawrakow has been going here it wouldn’t surprise me if we’d see MTP much sooner rather than later!

---

👤 **ikawrakow** commented the **2025-03-09** at **09:03:04**:<br>

> But, tbh, at the rate @ikawrakow has been going here it wouldn’t surprise me if we’d see MTP much sooner rather than later!

I have been wondering about that. Why has nobody added the MTP layer to the `llama.cpp` GGUF?

---

👤 **saood06** commented the **2025-03-09** at **10:52:15**:<br>

> I have been wondering about that. Why has nobody added the MTP layer to the `llama.cpp` GGUF?

Adding the MTP to the GGUF is trivial, having a performant integrated implementation is difficult. 

Mainline has speculative support in server, so it would be a bit easier but looking at existing inference software and how they implemented it (1) sglang, which implemented a custom strategy based on Eagle-2. Llama.cpp never adopted support for Eagle, or Eagle-2 based speculative decoding even though issues were created and there was demand for it. (2) vLLM implementation is here https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/deepseek_mtp.py . This looks simpler, at the cost of performance as it is less performant than sglang's MTP, but it would still require work to implement here.