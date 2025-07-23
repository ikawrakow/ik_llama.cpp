### 🔀 [#101](https://github.com/ikawrakow/ik_llama.cpp/pull/101) - Enable q6_0 in flash attention

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-10-21 |
| **Updated** | 2024-10-22 |

---

#### Description

As with `IQ4_NL`, just for head size of 128 for now. Without `GGML_CUDA_FA_ALL_QUANTS` set, only `Q6_0 + Q5_0` and `Q8_0 + Q6_0` are included. With this the VRAM poor have better options for selecting the best possible (as allowed by VRAM, model size, context length) quantized KV-cache from

| K-cache | V-cache | BPV |
| -------: | --------: | ----: |
| Q4_0 | Q4_0 | 4.5 |
| IQ4_NL | IQ4_NL | 4.5 |
| Q6_0 | Q5_0 | 6.0 |
| Q8_0 | IQ4_NL | 6.5 |
| Q8_0 | Q6_0 | 7.5 |
| Q8_0 | Q8_0 | 8.5 |
| F16 | F16 | 16.0 |

---

#### 💬 Conversation

👤 **Nexesenex** commented the **2024-10-21** at **18:14:38**:<br>

Merged in my fork of Kobold CPP. K q6_0 V q5_0 works like a charm. I also activated 16/6, 6/iq4_nl, as well as 8/6 and 6/6, I'll test them tonight or tomorrow.

Thank you (very very much) and congratulation for this, IK, I'm delighted to have those options and thus the best inference quality I can get right now, and I'm gonna release soon an updated version of my fork, with the proper credits of course, so everyone interested and not too scared by downloading my patchwork can enjoy the fruit of your labors on these KV Quants, as some already enjoyed a bit more speed on CPU due to some of your commits that I was able to merge a few months ago!