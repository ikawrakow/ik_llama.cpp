## 📌 [Issue #345](https://github.com/ikawrakow/ik_llama.cpp/issues/345) - build question newbie

| **Author** | `VinnyG9` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-04-25 |
| **Updated** | 2025-04-30 |

---

## 📄 Description

hello, i just found this repo and I'm getting incredible performance on my rock5b SBC 

i saw some build flags flowing around like
DGGML_NATIVE=1
OpenMP
flax-vectors-something
tinyblas 

I'm wondering what they do and if I'm missing any other one to squeeze even more performance

here are some quick numbers i got
```
k_llama.cpp$
user@rock-5b:/srv/dev-disk-by-uuid-0444eaaf-0405-4373-ad45-74f5ca64d1df/fast/github/ik_llama.cpp$ ./build/bin/llama-bench -m models/bitnet1582b4t-iq2_bn.gguf  -m models/bitnet1582b4t-iq2_bn_r4.gguf -m models/deepcogito_cogito-v1-preview-llama-3B-IQ4_NL.gguf -m models/deepcogito_cogito-v1-preview-llama-3B-Q4_0.gguf -m models/deepcogito_cogito-v1-preview-llama-3B-Q4_K_M.gguf -m models/deepcogito_cogito-v1-preview-llama-3B-Q4_K_S.gguf -p 64,128,256,512,1024 -n 64,128,256,512,1024 -t 4 -rtr 1
| model                          |       size |     params | backend
   | threads | rtr |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --: | ------------: | ---------------: |
============ Repacked 211 tensors
| bitnet-25 2B IQ2_BN - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B | CPU        |       4 |   1 |          pp64 |    318.86 ± 6.89 |
| bitnet-25 2B IQ2_BN - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B | CPU        |       4 |   1 |         pp128 |    238.43 ± 0.36 |
| bitnet-25 2B IQ2_BN - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B | CPU        |       4 |   1 |         pp256 |    158.87 ± 0.16 |
| bitnet-25 2B IQ2_BN - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B | CPU        |       4 |   1 |         pp512 |     98.19 ± 0.11 |
| bitnet-25 2B IQ2_BN - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B | CPU        |       4 |   1 |        pp1024 |     70.59 ± 0.04 |
| bitnet-25 2B IQ2_BN - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B | CPU        |       4 |   1 |          tg64 |    161.93 ± 0.04 |
| bitnet-25 2B IQ2_BN - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B | CPU        |       4 |   1 |         tg128 |    150.32 ± 0.47 |
| bitnet-25 2B IQ2_BN - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B | CPU        |       4 |   1 |         tg256 |    131.80 ± 0.06 |
| bitnet-25 2B IQ2_BN - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B | CPU        |       4 |   1 |         tg512 |    106.54 ± 0.03 |
| bitnet-25 2B IQ2_BN - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B | CPU        |       4 |   1 |        tg1024 |     74.70 ± 0.08 |
============ Repacked 1 tensors
| bitnet-25 2B IQ2_BN_R4 - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B
| CPU        |       4 |   1 |          pp64 |    318.16 ± 0.97 |
| bitnet-25 2B IQ2_BN_R4 - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B
| CPU        |       4 |   1 |         pp128 |    236.25 ± 1.11 |
| bitnet-25 2B IQ2_BN_R4 - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B
| CPU        |       4 |   1 |         pp256 |    157.40 ± 0.17 |
| bitnet-25 2B IQ2_BN_R4 - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B
| CPU        |       4 |   1 |         pp512 |     97.44 ± 0.10 |
| bitnet-25 2B IQ2_BN_R4 - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B
| CPU        |       4 |   1 |        pp1024 |     70.36 ± 0.04 |
| bitnet-25 2B IQ2_BN_R4 - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B
| CPU        |       4 |   1 |          tg64 |    162.03 ± 0.04 |
| bitnet-25 2B IQ2_BN_R4 - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B
| CPU        |       4 |   1 |         tg128 |    150.46 ± 0.04 |
| bitnet-25 2B IQ2_BN_R4 - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B
| CPU        |       4 |   1 |         tg256 |    131.58 ± 1.27 |
| bitnet-25 2B IQ2_BN_R4 - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B
| CPU        |       4 |   1 |         tg512 |    106.38 ± 0.22 |
| bitnet-25 2B IQ2_BN_R4 - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B
| CPU        |       4 |   1 |        tg1024 |     74.93 ± 0.03 |
============ Repacked 197 tensors
| llama ?B IQ4_NL - 4.5 bpw      |   1.98 GiB |     3.61 B | CPU
   |       4 |   1 |          pp64 |    312.00 ± 0.70 |
| llama ?B IQ4_NL - 4.5 bpw      |   1.98 GiB |     3.61 B | CPU
   |       4 |   1 |         pp128 |    228.23 ± 0.85 |
| llama ?B IQ4_NL - 4.5 bpw      |   1.98 GiB |     3.61 B | CPU
   |       4 |   1 |         pp256 |    150.19 ± 0.27 |
| llama ?B IQ4_NL - 4.5 bpw      |   1.98 GiB |     3.61 B | CPU
   |       4 |   1 |         pp512 |     90.48 ± 0.15 |
| llama ?B IQ4_NL - 4.5 bpw      |   1.98 GiB |     3.61 B | CPU
   |       4 |   1 |        pp1024 |     64.53 ± 0.04 |
| llama ?B IQ4_NL - 4.5 bpw      |   1.98 GiB |     3.61 B | CPU
   |       4 |   1 |          tg64 |    170.81 ± 0.05 |
| llama ?B IQ4_NL - 4.5 bpw      |   1.98 GiB |     3.61 B | CPU
   |       4 |   1 |         tg128 |    155.30 ± 0.03 |
| llama ?B IQ4_NL - 4.5 bpw      |   1.98 GiB |     3.61 B | CPU
   |       4 |   1 |         tg256 |    130.97 ± 0.09 |
| llama ?B IQ4_NL - 4.5 bpw      |   1.98 GiB |     3.61 B | CPU
   |       4 |   1 |         tg512 |     96.60 ± 0.17 |
| llama ?B IQ4_NL - 4.5 bpw      |   1.98 GiB |     3.61 B | CPU
   |       4 |   1 |        tg1024 |     59.32 ± 0.03 |
============ Repacked 194 tensors
| llama ?B Q4_0                  |   1.99 GiB |     3.61 B | CPU
   |       4 |   1 |          pp64 |    142.40 ± 0.18 |
| llama ?B Q4_0                  |   1.99 GiB |     3.61 B | CPU
   |       4 |   1 |         pp128 |    122.02 ± 0.12 |
| llama ?B Q4_0                  |   1.99 GiB |     3.61 B | CPU
   |       4 |   1 |         pp256 |     95.33 ± 0.11 |
| llama ?B Q4_0                  |   1.99 GiB |     3.61 B | CPU
   |       4 |   1 |         pp512 |     67.30 ± 0.08 |
| llama ?B Q4_0                  |   1.99 GiB |     3.61 B | CPU
   |       4 |   1 |        pp1024 |     51.75 ± 0.03 |
| llama ?B Q4_0                  |   1.99 GiB |     3.61 B | CPU
   |       4 |   1 |          tg64 |    101.11 ± 0.05 |
| llama ?B Q4_0                  |   1.99 GiB |     3.61 B | CPU
   |       4 |   1 |         tg128 |     95.60 ± 0.01 |
| llama ?B Q4_0                  |   1.99 GiB |     3.61 B | CPU
   |       4 |   1 |         tg256 |     84.97 ± 0.02 |
| llama ?B Q4_0                  |   1.99 GiB |     3.61 B | CPU
   |       4 |   1 |         tg512 |     69.57 ± 0.06 |
| llama ?B Q4_0                  |   1.99 GiB |     3.61 B | CPU
   |       4 |   1 |        tg1024 |     48.06 ± 0.03 |
============ Repacked 197 tensors
| llama ?B Q4_K - Medium         |   2.08 GiB |     3.61 B | CPU
   |       4 |   1 |          pp64 |    309.64 ± 0.78 |
| llama ?B Q4_K - Medium         |   2.08 GiB |     3.61 B | CPU
   |       4 |   1 |         pp128 |    227.22 ± 1.13 |
| llama ?B Q4_K - Medium         |   2.08 GiB |     3.61 B | CPU
   |       4 |   1 |         pp256 |    149.46 ± 0.34 |
| llama ?B Q4_K - Medium         |   2.08 GiB |     3.61 B | CPU
   |       4 |   1 |         pp512 |     90.10 ± 0.12 |
| llama ?B Q4_K - Medium         |   2.08 GiB |     3.61 B | CPU
   |       4 |   1 |        pp1024 |     64.23 ± 0.05 |
| llama ?B Q4_K - Medium         |   2.08 GiB |     3.61 B | CPU
   |       4 |   1 |          tg64 |    164.21 ± 0.07 |
| llama ?B Q4_K - Medium         |   2.08 GiB |     3.61 B | CPU
   |       4 |   1 |         tg128 |    149.79 ± 0.07 |
| llama ?B Q4_K - Medium         |   2.08 GiB |     3.61 B | CPU
   |       4 |   1 |         tg256 |    125.76 ± 0.06 |
| llama ?B Q4_K - Medium         |   2.08 GiB |     3.61 B | CPU
   |       4 |   1 |         tg512 |     94.72 ± 0.08 |
| llama ?B Q4_K - Medium         |   2.08 GiB |     3.61 B | CPU
   |       4 |   1 |        tg1024 |     58.99 ± 0.07 |
============ Repacked 197 tensors
| llama ?B Q4_K - Small          |   1.99 GiB |     3.61 B | CPU
   |       4 |   1 |          pp64 |    310.07 ± 1.15 |
| llama ?B Q4_K - Small          |   1.99 GiB |     3.61 B | CPU
   |       4 |   1 |         pp128 |    226.93 ± 0.88 |
| llama ?B Q4_K - Small          |   1.99 GiB |     3.61 B | CPU
   |       4 |   1 |         pp256 |    149.10 ± 0.58 |
| llama ?B Q4_K - Small          |   1.99 GiB |     3.61 B | CPU
   |       4 |   1 |         pp512 |     90.04 ± 0.12 |
| llama ?B Q4_K - Small          |   1.99 GiB |     3.61 B | CPU
   |       4 |   1 |        pp1024 |     64.23 ± 0.05 |
| llama ?B Q4_K - Small          |   1.99 GiB |     3.61 B | CPU
   |       4 |   1 |          tg64 |    164.18 ± 0.04 |
| llama ?B Q4_K - Small          |   1.99 GiB |     3.61 B | CPU
   |       4 |   1 |         tg128 |    150.28 ± 0.07 |
| llama ?B Q4_K - Small          |   1.99 GiB |     3.61 B | CPU
   |       4 |   1 |         tg256 |    125.84 ± 0.04 |
| llama ?B Q4_K - Small          |   1.99 GiB |     3.61 B | CPU
   |       4 |   1 |         tg512 |     94.57 ± 0.12 |
| llama ?B Q4_K - Small          |   1.99 GiB |     3.61 B | CPU
   |       4 |   1 |        tg1024 |     58.67 ± 0.05 |
build: c9eec172 (3644)
```

8B


```
build/bin/llama-bench -m models/deepcogito_cogito-v1-preview-llama-8B-IQ4_NL.gguf -p 64,128,256,512 -n
64,128,256,512 -t 4 -rtr 1
| model                          |       size |     params | backend
   | threads | rtr |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --: | ------------: | ---------------: |
============ Repacked 225 tensors
| llama 8B IQ4_NL - 4.5 bpw      |   4.35 GiB |     8.03 B | CPU
   |       4 |   1 |          pp64 |    183.79 ± 3.47 |
| llama 8B IQ4_NL - 4.5 bpw      |   4.35 GiB |     8.03 B | CPU
   |       4 |   1 |         pp128 |    139.43 ± 0.79 |
| llama 8B IQ4_NL - 4.5 bpw      |   4.35 GiB |     8.03 B | CPU
   |       4 |   1 |         pp256 |     94.39 ± 0.20 |
| llama 8B IQ4_NL - 4.5 bpw      |   4.35 GiB |     8.03 B | CPU
   |       4 |   1 |         pp512 |     57.99 ± 0.04 |
| llama 8B IQ4_NL - 4.5 bpw      |   4.35 GiB |     8.03 B | CPU
   |       4 |   1 |          tg64 |    110.81 ± 0.03 |
| llama 8B IQ4_NL - 4.5 bpw      |   4.35 GiB |     8.03 B | CPU
   |       4 |   1 |         tg128 |    100.95 ± 0.03 |
| llama 8B IQ4_NL - 4.5 bpw      |   4.35 GiB |     8.03 B | CPU
   |       4 |   1 |         tg256 |     85.88 ± 0.10 |
| llama 8B IQ4_NL - 4.5 bpw      |   4.35 GiB |     8.03 B | CPU
   |       4 |   1 |         tg512 |     65.49 ± 0.03 |
```
this is like 2000% improvement 

Thank you very much 🙏

---

## 💬 Conversation

👤 **VinnyG9** commented on **2025-04-25** at **06:04:04**

llama3.2/gemma3 are way worse on tg but same pp performance 
 
now when trying to chat with cogito I'm getting only this, any tips what's going on?

```
.Form comntSTSTSTSTnt g gSTntntSTSTSTST g g g gSTSTSTSTSTST gnt g g gntnt gST null gST g nullSTnt g g gntST gSTST gST gST g null null gntSTST g gSTnt g gSTSTntntSTSTST g gSTSTSTST g null gSTntSTST g gSTSTnt g gntntSTST null g g gSTSTST nullST g gSTSTntSTntntSTSTST gntST null g null g nullnt
llama_print_timings:        load time =    2283.50 ms
llama_print_timings:      sample time =      16.71 ms /   128 runs   (    0.13 ms per token,  7661.00 tokens per second)
llama_print_timings: prompt eval time =      38.83 ms /     5 tokens (    7.77 ms per token,   128.76 tokens per second)
llama_print_timings:        eval time =    1211.22 ms /   127 runs   (    9.54 ms per token,   104.85 tokens per second)
llama_print_timings:       total time =    1293.44 ms /   132 tokens
Log end

```

---

👤 **ikawrakow** commented on **2025-04-25** at **06:56:13**

What is cogito?

---

👤 **saood06** commented on **2025-04-25** at **07:00:09**

> What is cogito?

I'm assuming he's referring to this: https://huggingface.co/collections/deepcogito/cogito-v1-preview-67eb105721081abe4ce2ee53

---

👤 **mcm007** commented on **2025-04-25** at **07:07:28**

The t/s looks too high for a SBC, maybe the .gguf model is corrupt?


Results on "11th Gen Intel(R) Core(TM) i7-1185G7 @ 3.00GHz" with [bartowski cogito-v1-preview-llama-8B IQ4_NL](https://huggingface.co/bartowski/deepcogito_cogito-v1-preview-llama-8B-GGUF/blob/main/deepcogito_cogito-v1-preview-llama-8B-IQ4_NL.gguf) which produces good output:

```
./llama-bench -m /models1/deepcogito_cogito-v1-preview-llama-8B-IQ4_NL.gguf -p 64,128,256,512 -n 64,128,256,512 -t 4 -rtr 1
| model                          |       size |     params | backend    | threads | rtr |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --: | ------------: | ---------------: |
============ Repacked 225 tensors
| llama 8B IQ4_NL - 4.5 bpw      |   4.35 GiB |     8.03 B | CPU        |       4 |   1 |          pp64 |     31.93 ± 1.24 |
| llama 8B IQ4_NL - 4.5 bpw      |   4.35 GiB |     8.03 B | CPU        |       4 |   1 |         pp128 |     23.98 ± 9.64 |
| llama 8B IQ4_NL - 4.5 bpw      |   4.35 GiB |     8.03 B | CPU        |       4 |   1 |         pp256 |     21.24 ± 5.84 |
| llama 8B IQ4_NL - 4.5 bpw      |   4.35 GiB |     8.03 B | CPU        |       4 |   1 |         pp512 |     21.92 ± 2.58 |
| llama 8B IQ4_NL - 4.5 bpw      |   4.35 GiB |     8.03 B | CPU        |       4 |   1 |          tg64 |      7.97 ± 1.48 |
| llama 8B IQ4_NL - 4.5 bpw      |   4.35 GiB |     8.03 B | CPU        |       4 |   1 |         tg128 |      8.34 ± 0.62 |
| llama 8B IQ4_NL - 4.5 bpw      |   4.35 GiB |     8.03 B | CPU        |       4 |   1 |         tg256 |      8.86 ± 0.02 |
| llama 8B IQ4_NL - 4.5 bpw      |   4.35 GiB |     8.03 B | CPU        |       4 |   1 |         tg512 |      7.67 ± 0.79 |

build: 55fb9c81 (3643)
```

---

👤 **saood06** commented on **2025-04-25** at **07:26:13**

Also here are the tables from the first post
<details>
  <summary>Click me for Table 1</summary>

| model                          |       size |     params | backend   | threads | rtr |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --: | ------------: | ---------------: |
| bitnet-25 2B IQ2_BN - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B | CPU        |       4 |   1 |          pp64 |    318.86 ± 6.89 |
| bitnet-25 2B IQ2_BN - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B | CPU        |       4 |   1 |         pp128 |    238.43 ± 0.36 |
| bitnet-25 2B IQ2_BN - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B | CPU        |       4 |   1 |         pp256 |    158.87 ± 0.16 |
| bitnet-25 2B IQ2_BN - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B | CPU        |       4 |   1 |         pp512 |     98.19 ± 0.11 |
| bitnet-25 2B IQ2_BN - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B | CPU        |       4 |   1 |        pp1024 |     70.59 ± 0.04 |
| bitnet-25 2B IQ2_BN - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B | CPU        |       4 |   1 |          tg64 |    161.93 ± 0.04 |
| bitnet-25 2B IQ2_BN - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B | CPU        |       4 |   1 |         tg128 |    150.32 ± 0.47 |
| bitnet-25 2B IQ2_BN - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B | CPU        |       4 |   1 |         tg256 |    131.80 ± 0.06 |
| bitnet-25 2B IQ2_BN - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B | CPU        |       4 |   1 |         tg512 |    106.54 ± 0.03 |
| bitnet-25 2B IQ2_BN - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B | CPU        |       4 |   1 |        tg1024 |     74.70 ± 0.08 |
| bitnet-25 2B IQ2_BN_R4 - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B| CPU        |       4 |   1 |          pp64 |    318.16 ± 0.97 |
| bitnet-25 2B IQ2_BN_R4 - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B| CPU        |       4 |   1 |         pp128 |    236.25 ± 1.11 |
| bitnet-25 2B IQ2_BN_R4 - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B| CPU        |       4 |   1 |         pp256 |    157.40 ± 0.17 |
| bitnet-25 2B IQ2_BN_R4 - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B| CPU        |       4 |   1 |         pp512 |     97.44 ± 0.10 |
| bitnet-25 2B IQ2_BN_R4 - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B| CPU        |       4 |   1 |        pp1024 |     70.36 ± 0.04 |
| bitnet-25 2B IQ2_BN_R4 - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B| CPU        |       4 |   1 |          tg64 |    162.03 ± 0.04 |
| bitnet-25 2B IQ2_BN_R4 - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B| CPU        |       4 |   1 |         tg128 |    150.46 ± 0.04 |
| bitnet-25 2B IQ2_BN_R4 - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B| CPU        |       4 |   1 |         tg256 |    131.58 ± 1.27 |
| bitnet-25 2B IQ2_BN_R4 - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B| CPU        |       4 |   1 |         tg512 |    106.38 ± 0.22 |
| bitnet-25 2B IQ2_BN_R4 - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B| CPU        |       4 |   1 |        tg1024 |     74.93 ± 0.03 |
| llama ?B IQ4_NL - 4.5 bpw      |   1.98 GiB |     3.61 B | CPU   |       4 |   1 |          pp64 |    312.00 ± 0.70 |
| llama ?B IQ4_NL - 4.5 bpw      |   1.98 GiB |     3.61 B | CPU   |       4 |   1 |         pp128 |    228.23 ± 0.85 |
| llama ?B IQ4_NL - 4.5 bpw      |   1.98 GiB |     3.61 B | CPU   |       4 |   1 |         pp256 |    150.19 ± 0.27 |
| llama ?B IQ4_NL - 4.5 bpw      |   1.98 GiB |     3.61 B | CPU   |       4 |   1 |         pp512 |     90.48 ± 0.15 |
| llama ?B IQ4_NL - 4.5 bpw      |   1.98 GiB |     3.61 B | CPU   |       4 |   1 |        pp1024 |     64.53 ± 0.04 |
| llama ?B IQ4_NL - 4.5 bpw      |   1.98 GiB |     3.61 B | CPU   |       4 |   1 |          tg64 |    170.81 ± 0.05 |
| llama ?B IQ4_NL - 4.5 bpw      |   1.98 GiB |     3.61 B | CPU   |       4 |   1 |         tg128 |    155.30 ± 0.03 |
| llama ?B IQ4_NL - 4.5 bpw      |   1.98 GiB |     3.61 B | CPU   |       4 |   1 |         tg256 |    130.97 ± 0.09 |
| llama ?B IQ4_NL - 4.5 bpw      |   1.98 GiB |     3.61 B | CPU   |       4 |   1 |         tg512 |     96.60 ± 0.17 |
| llama ?B IQ4_NL - 4.5 bpw      |   1.98 GiB |     3.61 B | CPU   |       4 |   1 |        tg1024 |     59.32 ± 0.03 |
| llama ?B Q4_0                  |   1.99 GiB |     3.61 B | CPU   |       4 |   1 |          pp64 |    142.40 ± 0.18 |
| llama ?B Q4_0                  |   1.99 GiB |     3.61 B | CPU   |       4 |   1 |         pp128 |    122.02 ± 0.12 |
| llama ?B Q4_0                  |   1.99 GiB |     3.61 B | CPU   |       4 |   1 |         pp256 |     95.33 ± 0.11 |
| llama ?B Q4_0                  |   1.99 GiB |     3.61 B | CPU   |       4 |   1 |         pp512 |     67.30 ± 0.08 |
| llama ?B Q4_0                  |   1.99 GiB |     3.61 B | CPU   |       4 |   1 |        pp1024 |     51.75 ± 0.03 |
| llama ?B Q4_0                  |   1.99 GiB |     3.61 B | CPU   |       4 |   1 |          tg64 |    101.11 ± 0.05 |
| llama ?B Q4_0                  |   1.99 GiB |     3.61 B | CPU   |       4 |   1 |         tg128 |     95.60 ± 0.01 |
| llama ?B Q4_0                  |   1.99 GiB |     3.61 B | CPU   |       4 |   1 |         tg256 |     84.97 ± 0.02 |
| llama ?B Q4_0                  |   1.99 GiB |     3.61 B | CPU   |       4 |   1 |         tg512 |     69.57 ± 0.06 |
| llama ?B Q4_0                  |   1.99 GiB |     3.61 B | CPU   |       4 |   1 |        tg1024 |     48.06 ± 0.03 |
| llama ?B Q4_K - Medium         |   2.08 GiB |     3.61 B | CPU   |       4 |   1 |          pp64 |    309.64 ± 0.78 |
| llama ?B Q4_K - Medium         |   2.08 GiB |     3.61 B | CPU   |       4 |   1 |         pp128 |    227.22 ± 1.13 |
| llama ?B Q4_K - Medium         |   2.08 GiB |     3.61 B | CPU   |       4 |   1 |         pp256 |    149.46 ± 0.34 |
| llama ?B Q4_K - Medium         |   2.08 GiB |     3.61 B | CPU   |       4 |   1 |         pp512 |     90.10 ± 0.12 |
| llama ?B Q4_K - Medium         |   2.08 GiB |     3.61 B | CPU   |       4 |   1 |        pp1024 |     64.23 ± 0.05 |
| llama ?B Q4_K - Medium         |   2.08 GiB |     3.61 B | CPU   |       4 |   1 |          tg64 |    164.21 ± 0.07 |
| llama ?B Q4_K - Medium         |   2.08 GiB |     3.61 B | CPU   |       4 |   1 |         tg128 |    149.79 ± 0.07 |
| llama ?B Q4_K - Medium         |   2.08 GiB |     3.61 B | CPU   |       4 |   1 |         tg256 |    125.76 ± 0.06 |
| llama ?B Q4_K - Medium         |   2.08 GiB |     3.61 B | CPU   |       4 |   1 |         tg512 |     94.72 ± 0.08 |
| llama ?B Q4_K - Medium         |   2.08 GiB |     3.61 B | CPU   |       4 |   1 |        tg1024 |     58.99 ± 0.07 |
| llama ?B Q4_K - Small          |   1.99 GiB |     3.61 B | CPU   |       4 |   1 |          pp64 |    310.07 ± 1.15 |
| llama ?B Q4_K - Small          |   1.99 GiB |     3.61 B | CPU   |       4 |   1 |         pp128 |    226.93 ± 0.88 |
| llama ?B Q4_K - Small          |   1.99 GiB |     3.61 B | CPU   |       4 |   1 |         pp256 |    149.10 ± 0.58 |
| llama ?B Q4_K - Small          |   1.99 GiB |     3.61 B | CPU   |       4 |   1 |         pp512 |     90.04 ± 0.12 |
| llama ?B Q4_K - Small          |   1.99 GiB |     3.61 B | CPU   |       4 |   1 |        pp1024 |     64.23 ± 0.05 |
| llama ?B Q4_K - Small          |   1.99 GiB |     3.61 B | CPU   |       4 |   1 |          tg64 |    164.18 ± 0.04 |
| llama ?B Q4_K - Small          |   1.99 GiB |     3.61 B | CPU   |       4 |   1 |         tg128 |    150.28 ± 0.07 |
| llama ?B Q4_K - Small          |   1.99 GiB |     3.61 B | CPU   |       4 |   1 |         tg256 |    125.84 ± 0.04 |
| llama ?B Q4_K - Small          |   1.99 GiB |     3.61 B | CPU   |       4 |   1 |         tg512 |     94.57 ± 0.12 |
| llama ?B Q4_K - Small          |   1.99 GiB |     3.61 B | CPU   |       4 |   1 |        tg1024 |     58.67 ± 0.05 |
</details>

<details>
  <summary>Click me for Table 2</summary>

| model                          |       size |     params | backend   | threads | rtr |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --: | ------------: | ---------------: |
| llama 8B IQ4_NL - 4.5 bpw      |   4.35 GiB |     8.03 B | CPU   |       4 |   1 |          pp64 |    183.79 ± 3.47 |
| llama 8B IQ4_NL - 4.5 bpw      |   4.35 GiB |     8.03 B | CPU   |       4 |   1 |         pp128 |    139.43 ± 0.79 |
| llama 8B IQ4_NL - 4.5 bpw      |   4.35 GiB |     8.03 B | CPU   |       4 |   1 |         pp256 |     94.39 ± 0.20 |
| llama 8B IQ4_NL - 4.5 bpw      |   4.35 GiB |     8.03 B | CPU   |       4 |   1 |         pp512 |     57.99 ± 0.04 |
| llama 8B IQ4_NL - 4.5 bpw      |   4.35 GiB |     8.03 B | CPU   |       4 |   1 |          tg64 |    110.81 ± 0.03 |
| llama 8B IQ4_NL - 4.5 bpw      |   4.35 GiB |     8.03 B | CPU   |       4 |   1 |         tg128 |    100.95 ± 0.03 |
| llama 8B IQ4_NL - 4.5 bpw      |   4.35 GiB |     8.03 B | CPU   |       4 |   1 |         tg256 |     85.88 ± 0.10 |
| llama 8B IQ4_NL - 4.5 bpw      |   4.35 GiB |     8.03 B | CPU   |       4 |   1 |         tg512 |     65.49 ± 0.03 |

</details>

---

👤 **VinnyG9** commented on **2025-04-25** at **07:27:20**

> The t/s looks too high for a SBC, maybe the .gguf model is corrupt?
> 
> Results on "11th Gen Intel(R) Core(TM) i7-1185G7 @ 3.00GHz" with [bartowski cogito-v1-preview-llama-8B IQ4_NL](https://huggingface.co/bartowski/deepcogito_cogito-v1-preview-llama-8B-GGUF/blob/main/deepcogito_cogito-v1-preview-llama-8B-IQ4_NL.gguf) which produces good output:
> 
> ```
> ./llama-bench -m /models1/deepcogito_cogito-v1-preview-llama-8B-IQ4_NL.gguf -p 64,128,256,512 -n 64,128,256,512 -t 4 -rtr 1
> | model                          |       size |     params | backend    | threads | rtr |          test |              t/s |
> | ------------------------------ | ---------: | ---------: | ---------- | ------: | --: | ------------: | ---------------: |
> ============ Repacked 225 tensors
> | llama 8B IQ4_NL - 4.5 bpw      |   4.35 GiB |     8.03 B | CPU        |       4 |   1 |          pp64 |     31.93 ± 1.24 |
> | llama 8B IQ4_NL - 4.5 bpw      |   4.35 GiB |     8.03 B | CPU        |       4 |   1 |         pp128 |     23.98 ± 9.64 |
> | llama 8B IQ4_NL - 4.5 bpw      |   4.35 GiB |     8.03 B | CPU        |       4 |   1 |         pp256 |     21.24 ± 5.84 |
> | llama 8B IQ4_NL - 4.5 bpw      |   4.35 GiB |     8.03 B | CPU        |       4 |   1 |         pp512 |     21.92 ± 2.58 |
> | llama 8B IQ4_NL - 4.5 bpw      |   4.35 GiB |     8.03 B | CPU        |       4 |   1 |          tg64 |      7.97 ± 1.48 |
> | llama 8B IQ4_NL - 4.5 bpw      |   4.35 GiB |     8.03 B | CPU        |       4 |   1 |         tg128 |      8.34 ± 0.62 |
> | llama 8B IQ4_NL - 4.5 bpw      |   4.35 GiB |     8.03 B | CPU        |       4 |   1 |         tg256 |      8.86 ± 0.02 |
> | llama 8B IQ4_NL - 4.5 bpw      |   4.35 GiB |     8.03 B | CPU        |       4 |   1 |         tg512 |      7.67 ± 0.79 |
> 
> build: 55fb9c81 (3643)
> ```

I'm using the same one from bartowski, if you need me to test any models just say I'm on 300mbit connection

ive tested about a dozen models they all show crazy performance no idea why
I'm new to these things do you set any tokenizer/template/whatever?

all i did was 
build (cmake w/ no extra flags)
download model/config files via wget
run/bench

---

👤 **VinnyG9** commented on **2025-04-25** at **07:40:54**

yeah, they all output gibberish 

main llama.cpp works no problem

---

👤 **ikawrakow** commented on **2025-04-25** at **07:48:15**

I'm not familiar with this space, so had to look up what "rock 5b" is. According to [this](https://bret.dk/radxa-rock-5b-review-powerful-rk3588-sbc/) it has one Cortex-A76 and one Cortex-A55 CPU. For this the performance numbers look too high. Which means that most likely the `iqk` matrix multiplications that I have added do not get invoked, and it falls back to stock `ggml` implementation (`ggml` is the inference library behind `llama.cpp`). Most likely something goes wrong there, which leads to crazy performance and gibberish output. I did try to maintain this use case (the fallback to stock `ggml`) in a working condition for a while, but I think it is broken now.

I assume you are running Linux on this board? Can you do `cat /proc/cpuinfo`?

---

👤 **saood06** commented on **2025-04-25** at **07:56:57**

> I'm not familiar with this space, so had to look up what "rock 5b" is. According to [this](https://bret.dk/radxa-rock-5b-review-powerful-rk3588-sbc/) it has one Cortex-A76 and one Cortex-A55 CPU. For this the performance numbers look too high.

It has eight cores in total, "Quad Cortex®-A76 @ 2.2~2.4GHz and a Quad Cortex®-A55 @ 1.8GHz" from [what I think is the official product page](https://radxa.com/products/rock5/5b/#techspec). But even with that the performance still seems too high.

---

👤 **VinnyG9** commented on **2025-04-25** at **08:05:16**

> 
> I assume you are running Linux on this board? Can you do `cat /proc/cpuinfo`?

yup, that's the same soc orange pi 5+ uses, which I've seen somebody running here in this repo

```
processor       : 6
BogoMIPS        : 48.00
Features        : fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics
fphp asimdhp cpuid asimdrdm lrcpc dcpop asimddp
CPU implementer : 0x41
CPU architecture: 8
CPU variant     : 0x4
CPU part        : 0xd0b
CPU revision    : 0
processor       : 7
BogoMIPS        : 48.00
Features        : fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics
fphp asimdhp cpuid asimdrdm lrcpc dcpop asimddp
CPU implementer : 0x41
CPU architecture: 8
CPU variant     : 0x4
CPU part        : 0xd0b
CPU revision    : 0
```

i get same performance on q4km and iq4nl are you sure it has to do with iqk mm?

---

👤 **VinnyG9** commented on **2025-04-25** at **08:12:29**

> 
>  But even with that the performance still seems too high.

yeah i tried bitnet.cpp first = 0.5t/s
then came here expecting like 10-15t/s @8b
oh boy

but this little board is pretty capable I'm running 30 containers load average is ~0.5
NPU does 16x simultaneous 1080p@30fps transcodes at 18w 
fun board

maybe it's some dependency missing? but main runs normally...

---

👤 **ikawrakow** commented on **2025-04-25** at **08:16:42**

> i get same performance on q4km and iq4nl are you sure it has to do with iqk mm?

You are getting 110 t/s for tg64 with  LLaMA-8B. This is GPU territory (I get 130 t/s on my RTX-4080, 55 t/s on the M2-Max 30-core GPU). So, most likely the matrix multiplications don't get done at all.

The CPU flags look completely unfamiliar, so I cannot deduce from there if the `NEON` extensions get automatically enabled (required for this repo to work correctly on ARM CPUs).

---

👤 **VinnyG9** commented on **2025-04-25** at **08:22:04**

> > i get same performance on q4km and iq4nl are you sure it has to do with iqk mm?
> 
> You are getting 110 t/s for tg64 with LLaMA-8B. This is GPU territory (I get 130 t/s on my RTX-4080, 55 t/s on the M2-Max 30-core GPU). So, most likely the matrix multiplications don't get done at all.
> 
> The CPU flags look completely unfamiliar, so I cannot deduce from there if the `NEON` extensions get automatically enabled (required for this repo to work correctly on ARM CPUs).

they do

![Image](https://github.com/user-attachments/assets/d8726b49-ff66-4a73-b1aa-828052f05ea5)

---

👤 **ikawrakow** commented on **2025-04-25** at **08:24:21**

So, I'm finding that the `asimddp` CPU feature that you have should enable `__ARM_FEATURE_DOTPROD`. With that things should work correctly. 

What is the compiler being used?

---

👤 **VinnyG9** commented on **2025-04-25** at **08:27:08**

> So, I'm finding that the `asimddp` CPU feature that you have should enable `__ARM_FEATURE_DOTPROD`. With that things should work correctly.
> 
> What is the compiler being used?

gcc-12.2
i noticed bitnet.cpp uses clang-19 would that help?

DOTPROD shows enabled on main

![Image](https://github.com/user-attachments/assets/fa5ffc53-a552-4dbd-a590-1fde3b833ae1)

could the llamafile feature interfere?

---

👤 **ikawrakow** commented on **2025-04-25** at **08:41:23**

Mainline `llama.cpp` has now a much more sophisticated CPU feature detection than this project that got added after I forked. Here things are more on the "do it yourself" level. What you can do to see if the features added by this repo are working, add
```
printf("iqk is not enabled\n");
```
just before [this line](https://github.com/ikawrakow/ik_llama.cpp/blob/f176122a3d50c781414458b498b9426086a91647/ggml/src/iqk/iqk_mul_mat.cpp#L17563). Rebuild and run. If you see the messages, then something is not woking as expected.

---

👤 **ikawrakow** commented on **2025-04-25** at **08:56:26**

> could the llamafile feature interfere?

Normally no, but you can disable it just in case with `-DGGML_LLAMAFILE=0`

---

👤 **VinnyG9** commented on **2025-04-25** at **09:01:18**

> Mainline `llama.cpp` has now a much more sophisticated CPU feature detection than this project that got added after I forked. Here things are more on the "do it yourself" level. What you can do to see if the features added by this repo are working, add
> 
> ```
> printf("iqk is not enabled\n");
> ```
> 
> just before [this line](https://github.com/ikawrakow/ik_llama.cpp/blob/f176122a3d50c781414458b498b9426086a91647/ggml/src/iqk/iqk_mul_mat.cpp#L17563). Rebuild and run. If you see the messages, then something is not woking as expected.

like this?
![Image](https://github.com/user-attachments/assets/94a24876-b7d1-4b48-a31c-bafb173025f6)

i get a build error

---

👤 **ikawrakow** commented on **2025-04-25** at **09:02:45**

Yes.

---

👤 **ikawrakow** commented on **2025-04-25** at **09:03:57**

Sorry, also add the same `printf` line in the `iqk_mul_mat` function just above that.

---

👤 **VinnyG9** commented on **2025-04-25** at **09:06:13**

![Image](https://github.com/user-attachments/assets/44acfb74-4db2-46fd-8fe6-760bea9a653a)

---

👤 **ikawrakow** commented on **2025-04-25** at **09:08:36**

Then you need to add `#include <cstdio>` near the beginning of the file.

---

👤 **VinnyG9** commented on **2025-04-25** at **09:09:40**

> Then you need to add `#include <cstdio>` near the beginning of the file.

i did

edit: readded at the literal beginning xD
now it errors but keeps building 
![Image](https://github.com/user-attachments/assets/fc3746a4-c16c-43ae-b0f7-80988bbfbc30)

---

👤 **ikawrakow** commented on **2025-04-25** at **09:12:34**

The warning is harmless. What happens after you run it?

---

👤 **VinnyG9** commented on **2025-04-25** at **09:15:41**

> The warning is harmless. What happens after you run it?

floods the terminal with "iqk is not enabled"

---

👤 **ikawrakow** commented on **2025-04-25** at **09:18:13**

OK, so we know that the build does not work on your system. Your CPU supports the necessary features, so we need to understand why the compiler is not enabling them, so we can fix it.

---

👤 **VinnyG9** commented on **2025-04-25** at **09:20:29**

> OK, so we know that the build does not work on your system. Your CPU supports the necessary features, so we need to understand why the compiler is not enabling them, so we can fix it.

i can try with clang19?

---

👤 **ikawrakow** commented on **2025-04-25** at **09:22:44**

Yes, you can try building with `clang`, maybe this will fix it. But if not, I guess I need to add the ability to manually set compiler flags.

---

👤 **VinnyG9** commented on **2025-04-25** at **09:24:15**

i got this with clang build setup
not sure why as I'd seen openmp found earlier

```
- Could NOT find OpenMP_C (missing: OpenMP_C_FLAGS OpenMP_C_LIB_NAMES)
-- Could NOT find OpenMP_CXX (missing: OpenMP_CXX_FLAGS OpenMP_CXX_LIB_NAMES)
-- Could NOT find OpenMP (missing: OpenMP_C_FOUND OpenMP_CXX_FOUND)
CMake Warning at ggml/src/CMakeLists.txt:167 (message):
  OpenMP not found
```

---

👤 **ikawrakow** commented on **2025-04-25** at **09:25:33**

`OpenMP` is not really required. On my M2-Max laptop it actually hurts performance.

---

👤 **VinnyG9** commented on **2025-04-25** at **09:29:02**

same error on clang19

![Image](https://github.com/user-attachments/assets/23b03ccd-7faf-4f2e-b77d-7b724d9bcd4a)[](url)

---

👤 **ikawrakow** commented on **2025-04-25** at **09:46:07**

So, I made PR [#347](https://github.com/ikawrakow/ik_llama.cpp/issues/347) 

Can you try
```
git fetch
git checkout ik/arch_flags
cmake -DGGML_ARCH_FLAGS="-march=armv8.2-a+dotprod+fp16" (plus other things you want to add)
```

---

👤 **VinnyG9** commented on **2025-04-25** at **11:05:45**

![Image](https://github.com/user-attachments/assets/eef76f83-ec01-4617-9727-ffafed4a299f)

yup¡!!!!!!! working now, have yet to do the printf test and find how to disable openmp explicitly but iq4nl and q4km are running at least

numbers i got

         main         |    this PR
IQ4NL.  pp 43 tg 12 |  pp 38 tg 12
Q4KM.  pp 15 tg 10 |  pp 36 tg 12
Q4_0.    pp 50 tg 12 |   output gibberish 



also not able to use the -fa flag

---

👤 **ikawrakow** commented on **2025-04-25** at **11:14:20**

Great. Not sure what could be wrong with `Q4_0` as it does work on my M2-Max. Mainline has done optimizations for `Q4_0` and `IQ4_NL` on ARM, so for these there will not be much difference (my implementation is faster than theirs on the M2-Max, but I guess my optimizations are too aggressive for the A76, so mainline ends up being faster for these two quants on a lower spec Arm CPU). 

> also not able to use the -fa flag

Why? What happens?

---

👤 **VinnyG9** commented on **2025-04-25** at **18:28:37**

> Great. Not sure what could be wrong with `Q4_0` as it does work on my M2-Max. Mainline has done optimizations for `Q4_0` and `IQ4_NL` on ARM, so for these there will not be much difference (my implementation is faster than theirs on the M2-Max, but I guess my optimizations are too aggressive for the A76, so mainline ends up being faster for these two quants on a lower spec Arm CPU).
> 
> > also not able to use the -fa flag
> 
> Why? What happens?

sorry the battery ran out


edit: tested your new commits "fix fa on arm" and Q4_0 fix the latter works but -fa throws an error during generation
```
malloc(): invalid size (unsorted)
Aborted
```
is it really a cpu feature?

offtopic: from what i got reading llamacpp issues llamafile enables tinyblas? it works independent from GGML_BLAS being on or off? any point in trying e.g BLIS?

---

👤 **VinnyG9** commented on **2025-04-25** at **19:53:33**

got some decent performance with bitnet new model, however if i disable OpenMP, tg drops to 16t/s:

ik_llama.cpp$  build/bin/llama-bench -m ../models/bitnet1582b4t-iq2_bn.gguf -m ../models/bitnet1582b4t-iq2_bn_r4.gguf -p 64,128,256,512 -n 64,128,256,512 -t 4 -rtr 1
| model                          |       size |     params | backend    | threads | rtr |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --: | ------------: | ---------------: |
============ Repacked 211 tensors
| bitnet-25 2B IQ2_BN - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B | CPU        |       4 |   1 |          pp64 |     80.85 ± 0.06 |
| bitnet-25 2B IQ2_BN - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B | CPU        |       4 |   1 |         pp128 |     78.62 ± 0.03 |
| bitnet-25 2B IQ2_BN - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B | CPU        |       4 |   1 |         pp256 |     74.35 ± 0.03 |
| bitnet-25 2B IQ2_BN - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B | CPU        |       4 |   1 |         pp512 |     68.22 ± 0.04 |
| bitnet-25 2B IQ2_BN - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B | CPU        |       4 |   1 |          tg64 |     28.37 ± 0.02 |
| bitnet-25 2B IQ2_BN - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B | CPU        |       4 |   1 |         tg128 |     28.09 ± 0.03 |
| bitnet-25 2B IQ2_BN - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B | CPU        |       4 |   1 |         tg256 |     27.72 ± 0.02 |
| bitnet-25 2B IQ2_BN - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B | CPU        |       4 |   1 |         tg512 |     25.58 ± 0.77 |
============ Repacked 1 tensors
| bitnet-25 2B IQ2_BN_R4 - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B | CPU        |       4 |   1 |          pp64 |     79.62 ± 0.02 |
| bitnet-25 2B IQ2_BN_R4 - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B | CPU        |       4 |   1 |         pp128 |     77.85 ± 0.02 |
| bitnet-25 2B IQ2_BN_R4 - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B | CPU        |       4 |   1 |         pp256 |     73.56 ± 0.05 |
| bitnet-25 2B IQ2_BN_R4 - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B | CPU        |       4 |   1 |         pp512 |     67.69 ± 0.04 |
| bitnet-25 2B IQ2_BN_R4 - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B | CPU        |       4 |   1 |          tg64 |     28.02 ± 0.10 |
| bitnet-25 2B IQ2_BN_R4 - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B | CPU        |       4 |   1 |         tg128 |     26.48 ± 0.74 |
| bitnet-25 2B IQ2_BN_R4 - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B | CPU        |       4 |   1 |         tg256 |     25.95 ± 0.06 |
| bitnet-25 2B IQ2_BN_R4 - 2.00 bpw Bitnet | 934.16 MiB |     2.74 B | CPU        |       4 |   1 |         tg512 |     25.08 ± 0.05 |
build: 77089208 (3648)

this board seems to top at ~25GB/s which is nearly half the expected for lpddr4x dual channel, so the CPU is bottlenecking
but at least speed didn't drop much with longer text

---

👤 **VinnyG9** commented on **2025-04-25** at **21:26:36**

>
> ```
>      main         |    this PR
> 
> IQ4NL. pp 43 tg 12 | pp 38 tg 12 
> Q4KM. pp 15 tg 10 | pp 36 tg 12 
> Q4_0. pp 50 tg 12 | output gibberish
> ```

i was able to improve performance on all quantas by changing the
```
cmake -B build -DGGML_AR
CH_FLAGS="-march=armv8.2-a+dotprod+fp16"
```
to

```
cmake -B build -DGGML_AR
CH_FLAGS="-march=armv8.2-a+dotprod+fp16+noi8mm+nosve+nosme"
```

| quant | before | after |
| IQ4NL | pp 38 tg 12 | pp 43 tg 12 |
| Q4KM | pp 36 tg 12 | pp 42 tg 12 |
| Q4_0 | null | pp 40 tg 12 |

nosme actually only worked on main only on clang 

can someone explain why I'm not benefitting from the arm repack thing? like is not IQ4_NL supposed to run faster?

---

👤 **saood06** commented on **2025-04-26** at **00:36:22**

>nosme actually only worked on main only on clang

So for ik_llama.cpp was there a difference between clang and gcc now that you got it working?

---

👤 **ikawrakow** commented on **2025-04-26** at **06:12:06**

> can someone explain why I'm not benefitting from the arm repack thing? like is not IQ4_NL supposed to run faster?

You are benefiting. But 
you are also comparing to the two quants where in mainline they do the same kind of repacking as done here (`Q4_0` and `IQ4_NL`).  `Q4_K` is 2.4X faster here for PP as they don't have repacking on Arm for `Q4_K`. You can try `IQ4_XS`, which is probably the best choice for your board if you are using 4-bit quantization. If you go to lower bpw quants you will find much larger performance differences.

I find it interesting that explicitly disabling some features with `-march=armv8.2-a+dotprod+fp16+noi8mm+nosve+nosme` produces faster code. I'm not making use of any of these, so it must be the compiler inserting such instructions.

---

👤 **ikawrakow** commented on **2025-04-26** at **07:30:23**

> got some decent performance with bitnet new model, however if i disable OpenMP, tg drops to 16t/s:

I guess, if OpenMP is useful or not is more a matter of OS than a matter of CPU. OpenMP is indeed better on Linux, but I do get better performance without OpenMP on the M2-Max (macOS).

> this board seems to top at ~25GB/s which is nearly half the expected for lpddr4x dual channel, so the CPU is bottlenecking
but at least speed didn't drop much with longer text

I think you can only get more bandwidth utilized if both CPUs get used. Unfortunately the multi-threading implementation inherited from mainline is not useful for systems with a mix of fast and slow CPU cores. The work is simply split into chunks of equal size, so the slowest core determines how long it will take to compute. Improving this is one of the things that I want to do eventually.

---

👤 **VinnyG9** commented on **2025-04-26** at **17:55:29**

> > nosme actually only worked on main only on clang
> 
> So for ik_llama.cpp was there a difference between clang and gcc now that you got it working?

only OpenMP because clang couldn't find it, and the nosme as mentioned
i'll do more tests later on the desktop/teslas as well




> I think you can only get more bandwidth utilized if both CPUs get used. Unfortunately the multi-threading implementation inherited from mainline is not useful for systems with a mix of fast and slow CPU cores. The work is simply split into chunks of equal size, so the slowest core determines how long it will take to compute. Improving this is one of the things that I want to do eventually.

interesting info, for some reason the embedding model runs at the full bandwidth(~400x120=48Gb/s)

llama.cpp$ build/bin/llama-bench -m ../models/embed/bge-m3-Q4_0.gguf -p 64,128,256,512 -n 64,128,256,512 -t 4 -embd 1
| model                          |       size |     params | backend    | threads |       embd |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ---------: | ------------: | -------------------: |
| bert 335M Q4_0                 | 395.50 MiB |   566.70 M | CPU        |       4 |          1 |          pp64 |        441.44 ± 0.22 |
| bert 335M Q4_0                 | 395.50 MiB |   566.70 M | CPU        |       4 |          1 |         pp128 |        409.75 ± 0.21 |
| bert 335M Q4_0                 | 395.50 MiB |   566.70 M | CPU        |       4 |          1 |         pp256 |        349.64 ± 0.22 |
| bert 335M Q4_0                 | 395.50 MiB |   566.70 M | CPU        |       4 |          1 |         pp512 |        270.87 ± 0.17 |
| bert 335M Q4_0                 | 395.50 MiB |   566.70 M | CPU        |       4 |          1 |          tg64 |        117.99 ± 1.29 |
| bert 335M Q4_0                 | 395.50 MiB |   566.70 M | CPU        |       4 |          1 |         tg128 |        117.28 ± 0.03 |
| bert 335M Q4_0                 | 395.50 MiB |   566.70 M | CPU        |       4 |          1 |         tg256 |        115.44 ± 0.25 |
| bert 335M Q4_0                 | 395.50 MiB |   566.70 M | CPU        |       4 |          1 |         tg512 |        118.04 ± 0.11 |

---

👤 **ikawrakow** commented on **2025-04-27** at **06:13:34**

If your 566M parameter Bert model is something like [this one](https://huggingface.co/blogcncom/bge-m3-Q4_0-GGUF), 200 MiB out of 400 MiB are token embeddings. Only a tiny fraction of these 200 MiB gets actually used (~1000 bytes per generated token), so effectively you are running a 200 MiB model, so memory bandwidth utilized during TG is `120 t/s x 0.2 GiB = 24 GiB/s.`

---

👤 **VinnyG9** commented on **2025-04-30** at **04:45:02**

> If your 566M parameter Bert model is something like [this one](https://huggingface.co/blogcncom/bge-m3-Q4_0-GGUF), 200 MiB out of 400 MiB are token embeddings. Only a tiny fraction of these 200 MiB gets actually used (~1000 bytes per generated token), so effectively you are running a 200 MiB model, so memory bandwidth utilized during TG is `120 t/s x 0.2 GiB = 24 GiB/s.`

that's exactly it, thanks for the correction