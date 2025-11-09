## 🔀 [Pull Request #595](https://github.com/ikawrakow/ik_llama.cpp/pull/595) - CUDA: Faster prompt processing for several quantization types 

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/apply_cuda_faster_iq3k` |
| **Target Branch** | `main` |
| **Created** | 2025-07-09 |
| **Updated** | 2025-07-10 |
| **Merged** | 2025-07-10 |

---

## 📄 Description

This PR slightly improves prompt processing speed for `IQ3_K, IQ3_K_R4, IQ4_KS, IQ4_KS_R4, IQ4_K, IQ4_K_R4` and `IQ4_XS`.

Here some PP-512 results for LlaMA-3.1-8B on RTX-4080

 | model              |          test |    t/s (main)    |    t/s (PR)      |  Speedup |
| ------------------ | ------------: | ---------------: | ---------------: | -------: |
| llama 8B IQ3_K     |         pp512 |  6467.57 ± 18.48 |  6628.75 ± 14.24 |  1.025   |   
| llama 8B IQ3_K_R4  |         pp512 |  6102.36 ± 14.63 |  6464.58 ± 10.89 |  1.059   |   
| llama 8B IQ4_K     |         pp512 |  6442.38 ± 17.97 |  6625.94 ± 22.90 |  1.028   |   
| llama 8B IQ4_K_R4  |         pp512 |  6391.48 ± 16.77 |  6450.58 ± 11.54 |  1.009   |   
| llama 8B IQ4_KS    |         pp512 |  7732.35 ± 26.04 |  8074.07 ± 16.37 |  1.044   |   
| llama 8B IQ4_KS_R  |         pp512 |  7912.27 ± 21.10 |  8178.74 ± 28.14 |  1.034   |   
| llama 8B IQ4_XS    |         pp512 |  7748.68 ± 20.75 |  8149.86 ± 28.13 |  1.051   |

---

## 💬 Conversation

👤 **Nexesenex** commented on **2025-07-09** at **14:42:26**

Test in full Cuda offload on 3 Ampere GPUs (3090-3090-RTXA4000), TS 3-3-2+output, MMQ, and BBS 128 (on Croco.cpp) :

No trouble on my end for merging, compiling, and infering, 

I tested on a 111b Command-A model quantized as such :
llama_model_loader: - type iq5_ks:    1 tensors
llama_model_loader: - type iq4_ks_r4:  320 tensors
llama_model_loader: - type iq5_ks_r4:  128 tensors
Gross average PP : around 445 t/s, 435 before.

On a Mistral 123b :
llama_model_loader: - type  f32:  177 tensors
llama_model_loader: - type iq3_k:  352 tensors
llama_model_loader: - type iq5_k:   89 tensors
llama_model_loader: - type iq6_k:    1 tensors
llama_model_loader: - type iq4_ks:  176 tensors
Gross average PP : 340 t/s, 330 before.

---

👤 **ikawrakow** commented on **2025-07-09** at **14:49:20**

Thanks for testing. Yes, it is a 1-5% kind of improvement, nothing major.

---

👤 **ubergarm** commented on **2025-07-09** at **20:14:20**

Seeing roughly 1.2~2.6% speed-up on a `Qwen3-14B-IQ3_K` mix of mostly iq4_k and iq3_k fully offloaded on my home rig 3090TI FE. I checked at default batch sizes and also `-ub 4096 -b 4096` where it was still faster albeit slightly less gains vs default batch sizes.

![sweep-bench-PR595](https://github.com/user-attachments/assets/892ccba1-b896-4a1b-8118-ce112f199c59)

Nice!