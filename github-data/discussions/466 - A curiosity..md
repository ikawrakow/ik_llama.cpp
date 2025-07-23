### 🗣️ [#466](https://github.com/ikawrakow/ik_llama.cpp/discussions/466) - A curiosity.

| **Author** | `Nexesenex` |
| :--- | :--- |
| **Created** | 2025-05-28 |
| **Updated** | 2025-06-08 |

---

#### Description

I made a little fork of Llama.cpp mainline, integrating some commits of IK_Llama, and able to quantize (for now) in q6_0, IQ3_K, IQ4_K, IQ5_K and IQ6_K.
It's based on b5474 for now, and now I can use the wonderful q6_0 and IQ6_K for any model supported by mainline.
Here's the first alpha : https://github.com/Nexesenex/croco.cpp/releases/tag/v0.01

Edit : https://github.com/Nexesenex/croco.cpp/releases/tag/NXS_v0.04_b5525

Edit 2 : https://github.com/Nexesenex/croco.cpp/releases/tag/v1.93040_b5600_RMv1.11.8 (with NXS_Llama_v0.13_b5600), an attempt to make work the R4 quants supported on Cuda.

---

#### 🗣️ Discussion

👤 **VinnyG9** replied the **2025-05-28** at **20:14:51**:<br>

any performance numberos?

> 👤 **Nexesenex** replied the **2025-05-29** at **07:05:33**:<br>
> None, it barely works for a part of its purpose, which is to quantize models with some IQ quants within the mainline framework.
> PPL test work also, as well as Cuda inference for Gemma 3 in 0.04. And that's it for now. ^^