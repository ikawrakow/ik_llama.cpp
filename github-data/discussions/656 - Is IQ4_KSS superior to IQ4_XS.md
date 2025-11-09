## 🗣️ [Discussion #656](https://github.com/ikawrakow/ik_llama.cpp/discussions/656) - Is IQ4_KSS superior to IQ4_XS?

| **Author** | `tarruda` |
| :--- | :--- |
| **State** | ✅ **Answered** |
| **Created** | 2025-07-27 |
| **Updated** | 2025-07-27 |

---

## 📄 Description

I've been running unsloth's IQ4_XS quant of Qwen 235B Instruct with upstream llama.cpp, as that is the highest quant I can fit on my 128GB mac studio with 32k context.

Just read about ik_llama.cpp on reddit and saw that someone uploaded a IQ4_KSS version, which apparently uses less memory.

I'm curious how this new IQ4_KSS compares with IQ4_XS. Should I download the new quant or are they going to be too close to notice any differences?

---

## 💬 Discussion

👤 **saood06** commented on **2025-07-27** at **01:08:29**

>Should I download the new quant or are they going to be too close to notice any differences?

There really isn't a good way to know this without using it as every different use and user changes what differences matter.

>I'm curious how this new IQ4_KSS compares with IQ4_XS.

Well there is a lot of info about this in the IQ4_KSS PR here: [#89](https://github.com/ikawrakow/ik_llama.cpp/issues/89) and I would also recommend reading this [#83](https://github.com/ikawrakow/ik_llama.cpp/issues/83) as this PR shows an equivalent size newer quant vs IQ4_XS. Another user asked a similar thing here: https://github.com/ikawrakow/ik_llama.cpp/discussions/334#discussioncomment-13666631 and this is also worth reading

---

👤 **tarruda** commented on **2025-07-27** at **10:38:50**

Thanks @saood06 I will give it a shot as IQ4_KSS allows for extra context on my setup.

Unrelated to the original question, but does ik_llama.cpp CPU inference improvements introduces some sort of regression on Apple silicon metal inference? I did build ik_llama.cpp locally and ran llama-bench on my existing IQ4_XS weights, and it seems to have half the pp512 speed:

`llama.cpp`(@ca0ef2dddb022cb1337d775cd05cd27d7808aff4):
```
% ./build/bin/llama-bench -m ~/weights/unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF/iq4_xs/Qwen3-235B-A22B-Instruct-2507-IQ4_XS-00001-of-00003.gguf
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen3moe 235B.A22B IQ4_XS - 4.25 bpw | 116.86 GiB |   235.09 B | Metal,BLAS |      16 |           pp512 |        148.58 ± 0.73 |
| qwen3moe 235B.A22B IQ4_XS - 4.25 bpw | 116.86 GiB |   235.09 B | Metal,BLAS |      16 |           tg128 |         18.30 ± 0.00 |
```

`ik_llama.cpp`(@ae0ba31fd078282fe6ac675176862ed6955c52dc):

```
% ./ik-build/bin/llama-bench -m ~/weights/unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF/iq4_xs/Qwen3-235B-A22B-Instruct-2507-IQ4_XS-00001-of-00003.gguf
| model                          |       size |     params | backend    | ngl |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------------: | ---------------: |
| qwen3moe ?B IQ4_XS - 4.25 bpw  | 116.86 GiB |   235.09 B | Metal      |  99 |         pp512 |     76.09 ± 0.40 |
| qwen3moe ?B IQ4_XS - 4.25 bpw  | 116.86 GiB |   235.09 B | Metal      |  99 |         tg128 |     16.74 ± 0.00 |
```

This is on a Mac Studio M1 Ultra with 125GB VRAM. I noticed that the ik version doesn't have "BLAS"  in the backend column.

> 👤 **ikawrakow** replied on **2025-07-27** at **15:35:18**
> 
> I haven't done much work on the Metal back-end, so it is possible `llama.cpp` is now faster. But if so, it definitely is not a "regression". If you check out a `llama.cpp` version from August of last year (last time `ik_llama.cpp` was synced with upstream), I'm pretty sure `ik_llama.cpp` will be faster by a large margin compared to that. The performance improvements in `llama.cpp` on Metal for MoE models such as Qwen3-235B-A22B are relatively recent.

> 👤 **tarruda** replied on **2025-07-27** at **17:07:39**
> 
> I initially assumed that ik_llama.cpp was actively pulling changes from llama.cpp, and had no idea it diverged from llama.cpp almost one year ago. I used the term regression" under the assumption that the CPU inference improvements done came at the cost of reduced performance in other cases. Clearly I was mistaken.
> 
> Thanks for the clarification @ikawrakow, and for your amazing contributions to local inference of LLMs.