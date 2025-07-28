## 🔀 [Pull Request #566](https://github.com/ikawrakow/ik_llama.cpp/pull/566) - Adding IQ3_KS quants

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/iq3_ks_v2` |
| **Target Branch** | `main` |
| **Created** | 2025-07-01 |
| **Updated** | 2025-07-02 |
| **Merged** | 2025-07-02 |

---

## 📄 Description

This PR adds `IQ3_KS` - 3.1875 bpw quants with a block size of 32. This makes the `IQX_KS` quant series complete

| type | bpw |
| ---: | ---: |
| IQ2_KS | 2.1875 |
| IQ3_KS | 3.1875 |
| IQ4_KS | 4.25 |
| IQ5_KS | 5.25 |

CUDA and CPU performance are very good, Metal is not so great.

Here a few sweep-benches for LlaMA-3.1-8B-Instruct

### RTX-4080

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |    512 |    0.065 |  7932.94 |    0.887 |   144.38 |
|   512 |    128 |   1024 |    0.066 |  7725.27 |    0.893 |   143.35 |
|   512 |    128 |   1536 |    0.068 |  7551.51 |    0.908 |   141.02 |
|   512 |    128 |   2048 |    0.069 |  7404.30 |    0.924 |   138.59 |
|   512 |    128 |   2560 |    0.072 |  7098.39 |    0.939 |   136.30 |
|   512 |    128 |   3072 |    0.074 |  6873.96 |    0.955 |   134.08 |
|   512 |    128 |   3584 |    0.074 |  6890.43 |    0.969 |   132.07 |
|   512 |    128 |   4096 |    0.077 |  6620.20 |    0.987 |   129.64 |
|   512 |    128 |   4608 |    0.079 |  6445.44 |    1.000 |   128.00 |
|   512 |    128 |   5120 |    0.081 |  6350.94 |    1.026 |   124.82 |
|   512 |    128 |   5632 |    0.083 |  6175.82 |    1.033 |   123.97 |
|   512 |    128 |   6144 |    0.084 |  6071.67 |    1.043 |   122.77 |
|   512 |    128 |   6656 |    0.086 |  5944.16 |    1.057 |   121.15 |
|   512 |    128 |   7168 |    0.088 |  5810.65 |    1.071 |   119.46 |
|   512 |    128 |   7680 |    0.090 |  5693.89 |    1.087 |   117.77 |

### Ryzen-7950X (Zen4)

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    1.423 |   359.79 |    7.616 |    16.81 |
|   512 |    128 |    512 |    1.479 |   346.15 |    7.800 |    16.41 |
|   512 |    128 |   1024 |    1.537 |   333.06 |    7.979 |    16.04 |
|   512 |    128 |   1536 |    1.603 |   319.47 |    7.939 |    16.12 |
|   512 |    128 |   2048 |    1.661 |   308.29 |    7.984 |    16.03 |
|   512 |    128 |   2560 |    1.722 |   297.39 |    8.071 |    15.86 |
|   512 |    128 |   3072 |    1.778 |   287.90 |    8.154 |    15.70 |
|   512 |    128 |   3584 |    1.841 |   278.04 |    8.241 |    15.53 |

### Ryzen-5975WX

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    1.697 |   301.64 |    6.933 |    18.46 |
|   512 |    128 |    512 |    1.760 |   290.91 |    7.062 |    18.13 |
|   512 |    128 |   1024 |    1.834 |   279.19 |    7.217 |    17.74 |
|   512 |    128 |   1536 |    1.910 |   268.03 |    7.414 |    17.26 |
|   512 |    128 |   2048 |    1.985 |   257.88 |    7.555 |    16.94 |
|   512 |    128 |   2560 |    2.062 |   248.26 |    7.666 |    16.70 |
|   512 |    128 |   3072 |    2.140 |   239.29 |    7.810 |    16.39 |
|   512 |    128 |   3584 |    2.217 |   230.98 |    7.987 |    16.03 |

### M2-Max CPU

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    3.119 |   164.13 |    5.410 |    23.66 |
|   512 |    128 |    512 |    3.322 |   154.14 |    5.487 |    23.33 |
|   512 |    128 |   1024 |    3.614 |   141.66 |    5.658 |    22.62 |
|   512 |    128 |   1536 |    3.872 |   132.23 |    5.735 |    22.32 |
|   512 |    128 |   2048 |    4.089 |   125.21 |    5.911 |    21.65 |

### M2-Max 30-core GPU

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    1.088 |   470.79 |    3.255 |    39.33 |
|   512 |    128 |    512 |    1.106 |   462.77 |    3.411 |    37.53 |
|   512 |    128 |   1024 |    1.126 |   454.85 |    3.579 |    35.77 |
|   512 |    128 |   1536 |    1.153 |   444.08 |    3.762 |    34.03 |
|   512 |    128 |   2048 |    1.178 |   434.48 |    3.965 |    32.28 |
|   512 |    128 |   2560 |    1.207 |   424.23 |    4.118 |    31.08 |
|   512 |    128 |   3072 |    1.235 |   414.51 |    4.290 |    29.84 |
|   512 |    128 |   3584 |    1.265 |   404.69 |    4.461 |    28.69 |

---

## 💬 Conversation

👤 **ikawrakow** commented on **2025-07-02** at **07:27:42**

Let's merge this so people don't get crashes when trying to run `IQ3_KS` models with the main branch.

---

👤 **Nexesenex** commented on **2025-07-02** at **13:09:32**

On a SicariusSicariiStuff_Nano_Imp_1B-bf16 Llama 3.2 1B model I had on my drive.

PPL 512 wikitest eng.
IQ3_XXS, output Q6_K : 39.9308 +/- 0.36424
IQ3_KS V1 (your old branch), output Q6_K : 37.4846 +/- 0.34625
IQ3_KS V2 (this one), output IQ5_K : 35.3730 +/- 0.32563 (that's a clear improvement)
Q3_K, output Q6_K : 31.3082 +/- 0.28644
IQ3_S, output Q6_K : 34.0241 +/- 0.31115
IQ3_K, output Q6_K : 33.2313 +/- 0.30001

IQ3_KT FTYPE
llama_model_loader: - type  f32:   34 tensors
llama_model_loader: - type q5_K:    1 tensors (output)
llama_model_loader: - type iq3_s:    1 tensors (embeddings)
llama_model_loader: - type iq3_k:   16 tensors (attn_output)
llama_model_loader: - type iq5_k:   16 tensors (attn_v)
llama_model_loader: - type iq3_kt:   80 tensors
PPL 512 Wikitest eng : PPL = 35.3279 +/- 0.32366 (IQ3_KS_v2 compete quite well on this model)

Also, merged successfully on Croco.cpp, and it infers properly.

---

👤 **Nexesenex** commented on **2025-07-02** at **14:05:49**

@Ikawrakow : You brought us SOTA quants in the 2.1-2.2x bpw and 3.1x-3.2 bpw range with the KS and KT quants (and so, IQ2_XXS, XS, and IQ3_XXS are close to obsolescence now), and IQ2_K/IQ2_S remain on duty, but there's now a SOTA quants gap in the 2.4-3.1bpw range.

Would it be possible, mathematically wise, and interesting for you to develop a new IQ2_KL quant (in the 2.6875-2.75bpw range?), and offer a much more performant alternative to IQ2_S and IQ2_K, in line of what you developed recently?

---

👤 **ikawrakow** commented on **2025-07-02** at **14:25:59**

I have been thinking about this, but don't have a good idea how to spend the extra bits (extra compared to, e.g., `IQ2_KS`) without making inference inefficient. A larger Trellis quant based on `IQ2_KT` is also tricky as at 2 bpw (excluding block scales) we are already at 65k possibilities for a group of 8 quants, so going to 2.5 bpw would increase this to a million, which would make quantization time prohibitive. But if I go to groups of 4, that's only 1024 possibilities at 2.5 bpw, so this is not going to be very good. I could make a hybrid thing between `IQ2_KS` and a codebook (as the i-quants), but that brings CPU performance down, which is not good as most `ik_llama.cpp` users use it for the giant MoE models where TG runs on the CPU. But yes, If I come up with a good idea for a ~2.6-2.7 bpw quant, I will add it.

---

👤 **Nexesenex** commented on **2025-07-02** at **15:01:59**

Thanks for the explanation, I understand that the alternatives you have atm are quite unpractical.

In any case, thank you for the IQ3_KS (and the Cuda MMQ Kernels you kindly provided for most quants), it completes the KS quants lot, which is more practical to quantize than the already very demanding indeed Trellis lot. I'm very happy with all of this, compared to what mainline limits itself to atm.