### 🗣️ [#556](https://github.com/ikawrakow/ik_llama.cpp/discussions/556) - ik_llama.cpp for Armv8.0

| **Author** | `NotAHero04` |
| :--- | :--- |
| **Created** | 2025-06-25 |
| **Updated** | 2025-06-26 |

---

#### Description

I managed to port ik_llama.cpp to my phone which has a Snapdragon 680 CPU. Although under heavy emulation, it's still much faster than mainline llama.cpp. All of the tests are done using Qwen 3 0.6B model.
![Screenshot_2025_0625_135810](https://github.com/user-attachments/assets/39bd5d8e-d1eb-4dd4-9342-888733cc8fe2)
What works:
- Quants: legacy quants (tested Q4_0, Q8_0), i-quants (IQ4_XS), k-quants (Q4_K_M), iqk-quants (IQ4_KS, IQ5_K).
- Flash attention.
![Screenshot_2025_0625_141018](https://github.com/user-attachments/assets/e31a73c5-1bf9-4bc3-bdd6-303539748765)

What doesn't work:
- Trellis quants (tested IQ4_KT), though it might be specific to model or to my quantization. I'll test it more tonight.
- Repacking (both online and quantized forms, tested Q4_0_R8 and Q8_0_R8).
![Screenshot_2025_0625_141636](https://github.com/user-attachments/assets/21da3aed-d8a8-406e-82f7-ac6cef6d8a76)
If anyone is interested, I'll publish a fork. It just adds emulation for some NEON dot product and float16 arithmetic intrinsics. (mainline also has some level of emulation for v8.0)

---

#### 🗣️ Discussion

👤 **ikawrakow** replied the **2025-06-25** at **07:52:27**:<br>

Nice 😄 

The repacked variants don't work because the emulation for `vdotq_laneq_s32` is incorrect, or is there some other issue? But I guess it may not be worth putting too much effort into this as one would need to use `vgetq_lane_X`, which will make the dot products quite slow, I think.

---

👤 **NotAHero04** replied the **2025-06-25** at **14:37:21**:<br>

I did a fresh recompile and repacking works now! Unfortunately IQ4_KT still doesn't work :(
![Screenshot_2025_0625_213454](https://github.com/user-attachments/assets/ecdfd5e3-c7c0-41ce-affa-c35f59d68dfa)

---

👤 **ikawrakow** replied the **2025-06-25** at **15:30:22**:<br>

The `*_KT` quants are very slow on my M2-Max CPU, so it may not be worth putting the effort to make them work on a v8.0 phone.

> 👤 **NotAHero04** replied the **2025-06-26** at **09:18:15**:<br>
> So the KT quants do work after all, I just have to get the model from my PC. And yes, it is unbearably slow. (Q4_0 is 3x faster in TG)
> ![Screenshot_20250626_155507](https://github.com/user-attachments/assets/e0a54dc0-4285-470a-b333-5aba063566b0)

---

👤 **ikawrakow** replied the **2025-06-26** at **16:57:03**:<br>

Yes, the `*_kt` quants performance is very competitive on a GPU, nearly competitive on the two `x86_64` CPU's that I have available, 2X slower than corresponding size quant on the M2-Max CPU, and ridiculously slow on the M2-Max GPU.

But nice you have made all this work!