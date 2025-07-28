## 🗣️ [Discussion #18](https://github.com/ikawrakow/ik_llama.cpp/discussions/18) - CPU beating GPU in token generation speed

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2024-08-13 |
| **Updated** | 2025-04-03 |

---

## 📄 Description

The [TriLM](https://huggingface.co/collections/SpectraSuite/trilms-unpacked-668d5f62afe0f4036925b1d2) ternary models are available in various sizes, so I was curious to look into prompt processing (PP) and token generation (TG) speed when the model is small enough to fit in the CPU cache. I have a Ryzen-7950X CPU with 64 MiB of L3 cache, and the 99M parameter TriLM model is 46 MiB when quantized with `IQ2_TN`. So, without further ado, lets look at a comparison between the Ryzen-7950X and an RTX-4080 in this case:

| backend    | threads |          test |              t/s |
| ---------- | ------: | ------------: | ---------------: |
| Ryzen-7950X        |      16 |        pp1500 |  8268.11 ± 48.34 |
| Ryzen-7950X        |       4 |         tg500   |  1016.65 ± 22.17 |
| Ryzen-7950X        |       8 |         tg500   |  1224.83 ± 32.28 |
| Ryzen-7950X        |      16 |         tg500   |  1240.54 ± 25.74  |
| RTX-4080              |      -   |         pp1500 |  110388 ± 250 |
| RTX-4080              |      -   |         tg500 |  1136.64 ± 4.99 |

The GPU is still much faster than the CPU for prompt processing (although the difference, which is typically a factor of ~30 between this specific GPU and CPU, has shrunk to just a factor of 13), but now the CPU beats the GPU in TG speed!

I also have an M2-Max laptop (the version with a 30-core GPU). Here is what we get:

| backend    | threads |          test |              t/s |
| ---------- | ------: | ------------: | ---------------: |
| M2-Max CPU     |      8 |        pp1500 |  5209.27 ± 21.48 |
| M2-Max CPU        |       2 |         tg500   |  692.87 ± 1.74 |
| M2-Max CPU        |       4 |         tg500   |  841.48 ± 5.96 |
| M2-Max CPU        |       8 |         tg500   |  894.73 ± 10.03 |
| M2-Max GPU         |      4   |         pp1500 |  25824 ± 562 |
| M2-Max GPU         |      4  |         tg500 |  464.86 ± 3.85  |

Also here the GPU is faster for PP (but just 5X faster), but the CPU wipes the floor with the GPU for TG, beating it close to 2X using all 8 threads, and 1.5X with just 2 threads!

---

## 💬 Discussion

👤 **ikawrakow** commented on **2024-09-02** at **13:20:54**

Now that we have efficient Flash Attention (FA) implementation on the CPU via PR [#32](https://github.com/ikawrakow/ik_llama.cpp/issues/32), we can compare again performance between the CPU and GPU for this tiny 99M parameter model. We get

|  model                          |       size |     params | backend    | ngl | threads | fa |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -: | ------------: | ---------------: |
| IQ2_BN - 2.06 bpw TriLM |  45.89 MiB |    99.76 M | CUDA       | 100 |       1 |  1 |        pp1500 | 156827.38 ± 727 |
| IQ2_BN - 2.06 bpw TriLM |  45.89 MiB |    99.76 M | CUDA       | 100 |       1 |  1 |         tg500 |  1496.37 ± 36.79 |
| IQ2_BN - 2.06 bpw TriLM |  45.89 MiB |    99.76 M | CPU        |      0 | 16 |  1 |        pp1500 | 12133.80 ± 51.45 |
| IQ2_BN - 2.06 bpw TriLM |  45.89 MiB |    99.76 M | CPU        |      0 | 16 |  1 |         tg500 |   1509.52 ± 9.65 |

TG speed is now about the same, which is still quite remarkable.

FA has improved CPU prompt processing speed by almost 50%, TG by 22%.

> 👤 **saood06** replied on **2025-04-02** at **10:36:44**
> 
> Is there a chance SpargeAttn could be implemented here. Code [here](https://github.com/thu-ml/SpargeAttn), Paper [here](https://arxiv.org/abs/2502.18137). 
> 
> If it could would it benefit speed on CPU?

> 👤 **ikawrakow** replied on **2025-04-02** at **13:44:09**
> 
> Other than the paper, is there any evidence that this works as advertised? If I did nothing else but implementing breakthroughs announced on arXiv, the day still wouldn't have enough hours.

> 👤 **saood06** replied on **2025-04-03** at **00:24:39**
> 
> >Other than the paper, is there any evidence that this works as advertised?
> 
> Not really (there are multiple ComfyUI custom nodes that port support but not much on people using it), the paper looked interesting to me and the idea makes sense to me, but the implementation they have looks premature. The same group put out SageAttention/SageAttention2 which has been widely adopted (mostly for image/video models) and the performance matched the paper but SpargeAttn has gotten interest but not much adoption because of the state of the implmentation. 
> 
> >If I did nothing else but implementing breakthroughs announced on arXiv, the day still wouldn't have enough hours.
> 
> Sorry.

---

👤 **ikawrakow** commented on **2024-09-08** at **07:16:59**

With PR [#42](https://github.com/ikawrakow/ik_llama.cpp/issues/42) we get this

| model                          |       size |     params | backend    | threads | fa |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -: | ------------: | ---------------: |
| IQ2_BN - 2.06 bpw TriLM |  45.89 MiB |    99.76 M | CPU        |      16 |  1 |        pp1500 | 12906.95 ± 61.04 |
| IQ2_BN - 2.06 bpw TriLM |  45.89 MiB |    99.76 M | CPU        |      16 |  1 |         tg512 |  1563.62 ± 12.55 |

I.e., 56% improvement for PP and 26% improvement for TG since the original post from Aug 13!

I see [PR-8151](https://github.com/ggerganov/llama.cpp/pull/8151), which provides dedicated quantization for the TriLM ternary models in mainline `llama.cpp`, has been merged. Here is what we get for `TQ2_0` that corresponds to our `IQ2_TN`

| model                          |       size |     params | backend    | threads | fa |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -: | ------------: | -------------------: |
| TQ2_0 - 2.06 bpw ternary |  45.89 MiB |    99.76 M | CPU        |      16 |  1 |        pp1500 |      5187.34 ± 11.69 |
| TQ2_0 - 2.06 bpw ternary |  45.89 MiB |    99.76 M | CPU        |      16 |  0 |        pp1500 |      5281.54 ± 53.33 |
| TQ2_0 - 2.06 bpw ternary |  45.89 MiB |    99.76 M | CPU        |      16 |  1 |         tg500 |      1156.25 ± 18.14 |
| TQ2_0 - 2.06 bpw ternary |  45.89 MiB |    99.76 M | CPU        |      16 |  0 |         tg500 |      1041.27 ± 21.30 |

Our version is 2.44X faster for PP and 35% faster for TG.