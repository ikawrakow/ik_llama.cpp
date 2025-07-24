### 🗣️ [#594](https://github.com/ikawrakow/ik_llama.cpp/discussions/594) - Is AVX2 a hard requirement on x64?

| **Author** | `SmallAndSoft` |
| :--- | :--- |
| **Created** | 2025-07-08 |
| **Updated** | 2025-07-09 |

---

#### Description

I am getting compilation errors on the older CPU with just AVX even if I want to offload everything to CUDA GPU.

---

#### 🗣️ Discussion

👤 **ikawrakow** replied the **2025-07-09** at **08:41:22**:<br>

Yes, `AVX2` or better is a hard requirement on `x86_64`. I think `llama.cpp` is a better option for older hardware.

> 👤 **SmallAndSoft** replied the **2025-07-09** at **08:45:07**:<br>
> Thank you for reply. Yes, I just wanted to try your advanced quants on GPU. It is sad that AVX2 is required even if CPU will be doing next to nothing.