## 🔀 [Pull Request #511](https://github.com/ikawrakow/ik_llama.cpp/pull/511) - New IQ2_KT

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Source Branch** | `ik/new_iq2kt` |
| **Target Branch** | `main` |
| **Created** | 2025-06-09 |
| **Updated** | 2025-06-18 |
| **Labels** | `Breaking change` |

---

## 📄 Description

This PR uses the new trellis introduced in [#505](https://github.com/ikawrakow/ik_llama.cpp/issues/505) and applies it to `IQ2_KT`.

This leads to a slightly higher PPL for the models where the `IQ2_KT` on the main branch works, but is more stable and there are no longer NaNs for the models where the existing `IQ2_KT` was failing (Qwen3-30B-A3B and DeepSeek-Lite).

Performance is also great, except on the Apple GPU, where it is slower than the original `IQ2_KT` implementation. But on CUDA and on the CPU there are massive performance gains. Here an example of LLaMA-3.1-8B on RTX-4080 and Ryzen-7950X

| model            |       size |     params | backend    | fa |          test |              t/s |
| ---------------- | ---------: | ---------: | ---------- | -: | ------------: | ---------------: |
| llama 8B IQ2_KT  |   2.41 GiB |     8.03 B | CUDA       |  1 |         pp512 |  8972.05 ± 85.75 |
| llama 8B IQ2_KT  |   2.41 GiB |     8.03 B | CUDA       |  1 |         tg128 |    205.51 ± 0.22 |
| llama 8B IQ2_KT  |   2.41 GiB |     8.03 B | CPU        |  1 |         pp512 |    299.96 ± 4.58 |
| llama 8B IQ2_KT  |   2.41 GiB |     8.03 B | CPU        |  1 |         tg128 |     20.54 ± 0.18 |

---

## 💬 Conversation

👤 **ubergarm** commented on **2025-06-10** at **18:41:50**

Just kicked the tires on this PR and looks good so far!

1. It compiles fine.
2. I managed to quantize [OpenBuddy-R1-0528-Distill-Qwen3-32B-Preview0-QAT](https://huggingface.co/OpenBuddy/OpenBuddy-R1-0528-Distill-Qwen3-32B-Preview0-QAT) using a variety of quants including `iq2_kt` and `iq4_kt` from this PR.

There is not a lot of info about this model, and honestly it doesn't behave like a 4bpw QAT and they don't have much details (i asked on their hf repo). Their chat template stuff seems wonky too, (but that is unrelated to this PR). (might need to use the `tokenizer_config.json -> "chat_template"` JINJA template (also in the GGUF kv metadata) and make a new `llama_chat_apply_template_internal` case...) [*EDIT* made a rough chat template patch to test it [here](https://github.com/ubergarm/ik_llama.cpp/tree/ug/openbuddy_chat_template), but initial impressions is it might not be worth adding unless there is other demand]

Anyway, the important thing is the new `iq2_kt` and` iq4_kt` are functional, able to quantize using normal imatrix, runs full perplexity clean with no `nan` on CUDA RTXA6000, and outputs okay looking text (no gibberish) down to the `iq2_kt` even.

![ppl-OpenBuddy](https://github.com/user-attachments/assets/7ec38680-880b-4a78-ade9-4fbda3930abc)

I'll run some sweep benches too for speed comparisons.

---

👤 **ubergarm** commented on **2025-06-10** at **19:31:18**

Speed benchmarks on Single CUDA RTX A6000 48GB VRAM fully offloaded.

![sweep-bench-OpenBuddy-32B](https://github.com/user-attachments/assets/2a8b0e8e-3149-4f0e-971c-267cce56096d)

<details>

<summary>👈 Logs</summary>

```bash
git checkout ik/new_iq2kt

cmake -B ./build -DGGML_CUDA=ON -DGGML_BLAS=OFF -DGGML_SCHED_MAX_COPIES=1 -DGGML_CUDA_F16=ON
cmake --build ./build --config Release -j $(nproc)

#model=/mnt/raid/models/ubergarm/OpenBuddy-R1-0528-Distill-Qwen3-32B-Preview0-QAT-GGUF/DeepSeek-R1-0528-Distill-Qwen3-32B-Preview0-QAT-BF16-00001-of-00002.gguf
#model=/mnt/raid/models/ubergarm/OpenBuddy-R1-0528-Distill-Qwen3-32B-Preview0-QAT-GGUF/DeepSeek-R1-0528-Distill-Qwen3-32B-Preview0-QAT-Q4_0.gguf
#model=/mnt/raid/models/ubergarm/OpenBuddy-R1-0528-Distill-Qwen3-32B-Preview0-QAT-GGUF/DeepSeek-R1-0528-Distill-Qwen3-32B-Preview0-QAT-Q4_K.gguf
#model=/mnt/raid/models/ubergarm/OpenBuddy-R1-0528-Distill-Qwen3-32B-Preview0-QAT-GGUF/DeepSeek-R1-0528-Distill-Qwen3-32B-Preview0-QAT-IQ4_K.gguf
#model=/mnt/raid/models/ubergarm/OpenBuddy-R1-0528-Distill-Qwen3-32B-Preview0-QAT-GGUF/DeepSeek-R1-0528-Distill-Qwen3-32B-Preview0-QAT-IQ4_KS.gguf
#model=/mnt/raid/models/ubergarm/OpenBuddy-R1-0528-Distill-Qwen3-32B-Preview0-QAT-GGUF/DeepSeek-R1-0528-Distill-Qwen3-32B-Preview0-QAT-IQ4_KT.gguf
model=/mnt/raid/models/ubergarm/OpenBuddy-R1-0528-Distill-Qwen3-32B-Preview0-QAT-GGUF/DeepSeek-R1-0528-Distill-Qwen3-32B-Preview0-QAT-IQ2_KT.gguf

CUDA_VISIBLE_DEVICES="0" \
./build/bin/llama-sweep-bench \
    --model "$model" \
    --ctx-size 17408 \
    -ctk f16 -ctv f16 \
    -fa \
    -ngl 99 \
    --warmup-batch \
    --threads 1
```

## Q4_0
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    0.372 |  1376.00 |    3.810 |    33.59 |
|   512 |    128 |    512 |    0.380 |  1347.27 |    3.855 |    33.20 |
|   512 |    128 |   1024 |    0.390 |  1313.32 |    3.853 |    33.22 |
|   512 |    128 |   1536 |    0.398 |  1284.87 |    3.871 |    33.07 |
|   512 |    128 |   2048 |    0.405 |  1262.75 |    3.906 |    32.77 |
|   512 |    128 |   2560 |    0.416 |  1231.28 |    3.939 |    32.50 |
|   512 |    128 |   3072 |    0.426 |  1201.25 |    3.971 |    32.23 |
|   512 |    128 |   3584 |    0.435 |  1178.30 |    4.004 |    31.96 |
|   512 |    128 |   4096 |    0.442 |  1157.73 |    4.041 |    31.67 |
|   512 |    128 |   4608 |    0.450 |  1136.84 |    4.076 |    31.40 |
|   512 |    128 |   5120 |    0.460 |  1113.95 |    4.113 |    31.12 |
|   512 |    128 |   5632 |    0.469 |  1091.93 |    4.192 |    30.54 |
|   512 |    128 |   6144 |    0.478 |  1072.20 |    4.195 |    30.51 |
|   512 |    128 |   6656 |    0.485 |  1055.47 |    4.218 |    30.35 |
|   512 |    128 |   7168 |    0.492 |  1039.78 |    4.228 |    30.27 |
|   512 |    128 |   7680 |    0.501 |  1021.45 |    4.254 |    30.09 |
|   512 |    128 |   8192 |    0.510 |  1004.30 |    4.276 |    29.94 |
|   512 |    128 |   8704 |    0.519 |   986.79 |    4.300 |    29.77 |
|   512 |    128 |   9216 |    0.526 |   972.49 |    4.331 |    29.56 |
|   512 |    128 |   9728 |    0.534 |   958.15 |    4.358 |    29.37 |
|   512 |    128 |  10240 |    0.542 |   944.16 |    4.378 |    29.24 |
|   512 |    128 |  10752 |    0.550 |   931.00 |    4.466 |    28.66 |
|   512 |    128 |  11264 |    0.558 |   917.95 |    4.493 |    28.49 |
|   512 |    128 |  11776 |    0.565 |   906.58 |    4.473 |    28.61 |
|   512 |    128 |  12288 |    0.574 |   891.64 |    4.485 |    28.54 |
|   512 |    128 |  12800 |    0.581 |   881.06 |    4.511 |    28.37 |
|   512 |    128 |  13312 |    0.588 |   870.85 |    4.538 |    28.21 |
|   512 |    128 |  13824 |    0.596 |   859.14 |    4.561 |    28.07 |
|   512 |    128 |  14336 |    0.603 |   849.39 |    4.584 |    27.92 |
|   512 |    128 |  14848 |    0.615 |   832.78 |    4.614 |    27.74 |
|   512 |    128 |  15360 |    0.622 |   823.76 |    4.639 |    27.59 |
|   512 |    128 |  15872 |    0.629 |   814.41 |    4.663 |    27.45 |
|   512 |    128 |  16384 |    0.640 |   800.14 |    4.740 |    27.00 |

## Q4_K
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    0.398 |  1285.23 |    3.876 |    33.02 |
|   512 |    128 |    512 |    0.409 |  1251.94 |    3.923 |    32.63 |
|   512 |    128 |   1024 |    0.418 |  1223.79 |    3.920 |    32.65 |
|   512 |    128 |   1536 |    0.428 |  1195.37 |    3.939 |    32.50 |
|   512 |    128 |   2048 |    0.435 |  1175.93 |    3.974 |    32.21 |
|   512 |    128 |   2560 |    0.446 |  1148.89 |    4.005 |    31.96 |
|   512 |    128 |   3072 |    0.456 |  1122.26 |    4.039 |    31.69 |
|   512 |    128 |   3584 |    0.464 |  1103.45 |    4.075 |    31.41 |
|   512 |    128 |   4096 |    0.474 |  1081.26 |    4.111 |    31.13 |
|   512 |    128 |   4608 |    0.482 |  1062.08 |    4.145 |    30.88 |
|   512 |    128 |   5120 |    0.489 |  1045.97 |    4.182 |    30.61 |
|   512 |    128 |   5632 |    0.498 |  1028.66 |    4.265 |    30.01 |
|   512 |    128 |   6144 |    0.507 |  1010.81 |    4.267 |    29.99 |
|   512 |    128 |   6656 |    0.515 |   994.16 |    4.292 |    29.82 |
|   512 |    128 |   7168 |    0.524 |   977.04 |    4.293 |    29.82 |
|   512 |    128 |   7680 |    0.532 |   962.24 |    4.319 |    29.64 |
|   512 |    128 |   8192 |    0.540 |   947.85 |    4.343 |    29.47 |
|   512 |    128 |   8704 |    0.549 |   932.32 |    4.369 |    29.30 |
|   512 |    128 |   9216 |    0.558 |   917.14 |    4.399 |    29.10 |
|   512 |    128 |   9728 |    0.566 |   905.25 |    4.420 |    28.96 |
|   512 |    128 |  10240 |    0.573 |   892.89 |    4.446 |    28.79 |
|   512 |    128 |  10752 |    0.581 |   880.91 |    4.538 |    28.20 |
|   512 |    128 |  11264 |    0.590 |   867.99 |    4.566 |    28.03 |
|   512 |    128 |  11776 |    0.598 |   856.83 |    4.545 |    28.16 |
|   512 |    128 |  12288 |    0.606 |   844.92 |    4.555 |    28.10 |
|   512 |    128 |  12800 |    0.613 |   834.67 |    4.580 |    27.94 |
|   512 |    128 |  13312 |    0.622 |   823.72 |    4.606 |    27.79 |
|   512 |    128 |  13824 |    0.629 |   814.58 |    4.628 |    27.66 |
|   512 |    128 |  14336 |    0.636 |   804.84 |    4.653 |    27.51 |
|   512 |    128 |  14848 |    0.644 |   795.55 |    4.682 |    27.34 |
|   512 |    128 |  15360 |    0.652 |   785.29 |    4.704 |    27.21 |
|   512 |    128 |  15872 |    0.660 |   775.65 |    4.728 |    27.07 |
|   512 |    128 |  16384 |    0.668 |   766.55 |    4.807 |    26.63 |

## IQ4_K
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    0.488 |  1049.09 |    4.469 |    28.64 |
|   512 |    128 |    512 |    0.495 |  1033.90 |    4.513 |    28.36 |
|   512 |    128 |   1024 |    0.506 |  1012.60 |    4.536 |    28.22 |
|   512 |    128 |   1536 |    0.515 |   994.74 |    4.575 |    27.98 |
|   512 |    128 |   2048 |    0.527 |   972.36 |    4.630 |    27.65 |
|   512 |    128 |   2560 |    0.537 |   953.95 |    4.685 |    27.32 |
|   512 |    128 |   3072 |    0.545 |   938.94 |    4.732 |    27.05 |
|   512 |    128 |   3584 |    0.557 |   919.28 |    4.779 |    26.78 |
|   512 |    128 |   4096 |    0.566 |   905.20 |    4.828 |    26.51 |
|   512 |    128 |   4608 |    0.574 |   891.86 |    4.871 |    26.28 |
|   512 |    128 |   5120 |    0.584 |   876.47 |    4.916 |    26.04 |
|   512 |    128 |   5632 |    0.593 |   863.50 |    4.999 |    25.60 |
|   512 |    128 |   6144 |    0.601 |   851.51 |    5.017 |    25.51 |
|   512 |    128 |   6656 |    0.611 |   838.57 |    5.050 |    25.35 |
|   512 |    128 |   7168 |    0.618 |   828.94 |    5.060 |    25.30 |
|   512 |    128 |   7680 |    0.626 |   817.85 |    5.089 |    25.15 |
|   512 |    128 |   8192 |    0.636 |   805.25 |    5.117 |    25.02 |
|   512 |    128 |   8704 |    0.644 |   795.42 |    5.140 |    24.90 |
|   512 |    128 |   9216 |    0.652 |   784.96 |    5.169 |    24.76 |
|   512 |    128 |   9728 |    0.660 |   775.28 |    5.195 |    24.64 |
|   512 |    128 |  10240 |    0.669 |   765.28 |    5.221 |    24.52 |
|   512 |    128 |  10752 |    0.677 |   755.78 |    5.307 |    24.12 |
|   512 |    128 |  11264 |    0.684 |   748.31 |    5.334 |    24.00 |
|   512 |    128 |  11776 |    0.693 |   739.19 |    5.320 |    24.06 |
|   512 |    128 |  12288 |    0.700 |   731.07 |    5.339 |    23.97 |
|   512 |    128 |  12800 |    0.708 |   723.07 |    5.360 |    23.88 |
|   512 |    128 |  13312 |    0.717 |   713.84 |    5.386 |    23.77 |
|   512 |    128 |  13824 |    0.723 |   707.75 |    5.406 |    23.68 |
|   512 |    128 |  14336 |    0.732 |   699.50 |    5.433 |    23.56 |
|   512 |    128 |  14848 |    0.740 |   691.91 |    5.454 |    23.47 |
|   512 |    128 |  15360 |    0.748 |   684.68 |    5.478 |    23.37 |
|   512 |    128 |  15872 |    0.754 |   678.60 |    5.496 |    23.29 |
|   512 |    128 |  16384 |    0.762 |   671.95 |    5.562 |    23.01 |

## IQ4_KS
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    0.414 |  1236.19 |    3.999 |    32.00 |
|   512 |    128 |    512 |    0.423 |  1209.12 |    4.043 |    31.66 |
|   512 |    128 |   1024 |    0.432 |  1185.42 |    4.052 |    31.59 |
|   512 |    128 |   1536 |    0.442 |  1159.53 |    4.075 |    31.41 |
|   512 |    128 |   2048 |    0.451 |  1135.91 |    4.114 |    31.12 |
|   512 |    128 |   2560 |    0.461 |  1111.57 |    4.152 |    30.83 |
|   512 |    128 |   3072 |    0.469 |  1091.64 |    4.183 |    30.60 |
|   512 |    128 |   3584 |    0.479 |  1067.94 |    4.222 |    30.31 |
|   512 |    128 |   4096 |    0.488 |  1048.82 |    4.265 |    30.01 |
|   512 |    128 |   4608 |    0.498 |  1027.90 |    4.300 |    29.77 |
|   512 |    128 |   5120 |    0.506 |  1011.36 |    4.337 |    29.52 |
|   512 |    128 |   5632 |    0.514 |   996.39 |    4.420 |    28.96 |
|   512 |    128 |   6144 |    0.525 |   975.51 |    4.427 |    28.91 |
|   512 |    128 |   6656 |    0.532 |   962.19 |    4.454 |    28.74 |
|   512 |    128 |   7168 |    0.541 |   946.79 |    4.458 |    28.71 |
|   512 |    128 |   7680 |    0.549 |   931.88 |    4.484 |    28.55 |
|   512 |    128 |   8192 |    0.558 |   917.89 |    4.511 |    28.38 |
|   512 |    128 |   8704 |    0.566 |   905.17 |    4.536 |    28.22 |
|   512 |    128 |   9216 |    0.574 |   892.08 |    4.565 |    28.04 |
|   512 |    128 |   9728 |    0.582 |   879.27 |    4.586 |    27.91 |
|   512 |    128 |  10240 |    0.591 |   865.73 |    4.613 |    27.75 |
|   512 |    128 |  10752 |    0.599 |   855.02 |    4.703 |    27.22 |
|   512 |    128 |  11264 |    0.608 |   842.76 |    4.729 |    27.07 |
|   512 |    128 |  11776 |    0.614 |   833.86 |    4.712 |    27.16 |
|   512 |    128 |  12288 |    0.625 |   819.51 |    4.723 |    27.10 |
|   512 |    128 |  12800 |    0.630 |   812.88 |    4.750 |    26.95 |
|   512 |    128 |  13312 |    0.639 |   801.28 |    4.774 |    26.81 |
|   512 |    128 |  13824 |    0.648 |   790.22 |    4.795 |    26.70 |
|   512 |    128 |  14336 |    0.655 |   781.86 |    4.822 |    26.55 |
|   512 |    128 |  14848 |    0.663 |   772.15 |    4.848 |    26.40 |
|   512 |    128 |  15360 |    0.670 |   763.86 |    4.871 |    26.28 |
|   512 |    128 |  15872 |    0.678 |   755.06 |    4.895 |    26.15 |
|   512 |    128 |  16384 |    0.686 |   745.93 |    4.973 |    25.74 |

## IQ4_KT
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    0.409 |  1253.26 |    3.866 |    33.11 |
|   512 |    128 |    512 |    0.416 |  1229.86 |    3.916 |    32.69 |
|   512 |    128 |   1024 |    0.425 |  1203.83 |    3.928 |    32.59 |
|   512 |    128 |   1536 |    0.434 |  1180.87 |    3.945 |    32.44 |
|   512 |    128 |   2048 |    0.442 |  1158.85 |    3.977 |    32.18 |
|   512 |    128 |   2560 |    0.450 |  1137.03 |    4.008 |    31.94 |
|   512 |    128 |   3072 |    0.459 |  1114.55 |    4.058 |    31.54 |
|   512 |    128 |   3584 |    0.467 |  1096.28 |    4.094 |    31.27 |
|   512 |    128 |   4096 |    0.478 |  1072.10 |    4.127 |    31.01 |
|   512 |    128 |   4608 |    0.485 |  1054.76 |    4.156 |    30.80 |
|   512 |    128 |   5120 |    0.493 |  1038.25 |    4.195 |    30.52 |
|   512 |    128 |   5632 |    0.501 |  1021.18 |    4.271 |    29.97 |
|   512 |    128 |   6144 |    0.509 |  1005.50 |    4.275 |    29.94 |
|   512 |    128 |   6656 |    0.517 |   990.30 |    4.302 |    29.76 |
|   512 |    128 |   7168 |    0.525 |   975.22 |    4.313 |    29.68 |
|   512 |    128 |   7680 |    0.532 |   961.73 |    4.330 |    29.56 |
|   512 |    128 |   8192 |    0.541 |   946.23 |    4.347 |    29.45 |
|   512 |    128 |   8704 |    0.548 |   933.76 |    4.367 |    29.31 |
|   512 |    128 |   9216 |    0.556 |   920.76 |    4.398 |    29.11 |
|   512 |    128 |   9728 |    0.563 |   908.69 |    4.417 |    28.98 |
|   512 |    128 |  10240 |    0.572 |   895.58 |    4.443 |    28.81 |
|   512 |    128 |  10752 |    0.579 |   883.69 |    4.525 |    28.29 |
|   512 |    128 |  11264 |    0.586 |   873.18 |    4.551 |    28.12 |
|   512 |    128 |  11776 |    0.594 |   861.57 |    4.542 |    28.18 |
|   512 |    128 |  12288 |    0.601 |   851.28 |    4.558 |    28.08 |
|   512 |    128 |  12800 |    0.609 |   841.04 |    4.580 |    27.95 |
|   512 |    128 |  13312 |    0.617 |   830.42 |    4.589 |    27.89 |
|   512 |    128 |  13824 |    0.625 |   819.83 |    4.609 |    27.77 |
|   512 |    128 |  14336 |    0.632 |   810.68 |    4.629 |    27.65 |
|   512 |    128 |  14848 |    0.640 |   799.72 |    4.667 |    27.42 |
|   512 |    128 |  15360 |    0.649 |   789.30 |    4.677 |    27.37 |
|   512 |    128 |  15872 |    0.653 |   783.95 |    4.702 |    27.22 |
|   512 |    128 |  16384 |    0.664 |   771.36 |    4.764 |    26.87 |

## IQ2_KT
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    0.365 |  1400.95 |    2.737 |    46.76 |
|   512 |    128 |    512 |    0.373 |  1372.12 |    2.780 |    46.04 |
|   512 |    128 |   1024 |    0.381 |  1342.95 |    2.786 |    45.95 |
|   512 |    128 |   1536 |    0.389 |  1316.39 |    2.800 |    45.72 |
|   512 |    128 |   2048 |    0.399 |  1283.34 |    2.833 |    45.18 |
|   512 |    128 |   2560 |    0.407 |  1257.53 |    2.866 |    44.65 |
|   512 |    128 |   3072 |    0.415 |  1234.09 |    2.891 |    44.27 |
|   512 |    128 |   3584 |    0.423 |  1210.97 |    2.927 |    43.73 |
|   512 |    128 |   4096 |    0.431 |  1188.16 |    2.962 |    43.21 |
|   512 |    128 |   4608 |    0.440 |  1162.72 |    2.991 |    42.80 |
|   512 |    128 |   5120 |    0.450 |  1138.34 |    3.043 |    42.06 |
|   512 |    128 |   5632 |    0.457 |  1119.22 |    3.110 |    41.15 |
|   512 |    128 |   6144 |    0.466 |  1098.93 |    3.118 |    41.06 |
|   512 |    128 |   6656 |    0.475 |  1078.81 |    3.147 |    40.67 |
|   512 |    128 |   7168 |    0.484 |  1057.24 |    3.151 |    40.62 |
|   512 |    128 |   7680 |    0.491 |  1042.58 |    3.168 |    40.40 |
|   512 |    128 |   8192 |    0.497 |  1029.54 |    3.196 |    40.05 |
|   512 |    128 |   8704 |    0.508 |  1008.46 |    3.225 |    39.69 |
|   512 |    128 |   9216 |    0.515 |   993.52 |    3.252 |    39.36 |
|   512 |    128 |   9728 |    0.521 |   982.11 |    3.279 |    39.04 |
|   512 |    128 |  10240 |    0.531 |   964.15 |    3.291 |    38.89 |
|   512 |    128 |  10752 |    0.539 |   949.54 |    3.361 |    38.08 |
|   512 |    128 |  11264 |    0.547 |   935.45 |    3.388 |    37.78 |
|   512 |    128 |  11776 |    0.555 |   923.02 |    3.386 |    37.80 |
|   512 |    128 |  12288 |    0.564 |   907.81 |    3.398 |    37.67 |
|   512 |    128 |  12800 |    0.570 |   897.93 |    3.420 |    37.42 |
|   512 |    128 |  13312 |    0.581 |   881.98 |    3.441 |    37.20 |
|   512 |    128 |  13824 |    0.586 |   873.12 |    3.456 |    37.04 |
|   512 |    128 |  14336 |    0.595 |   860.52 |    3.478 |    36.80 |
|   512 |    128 |  14848 |    0.602 |   850.42 |    3.504 |    36.53 |
|   512 |    128 |  15360 |    0.609 |   841.32 |    3.523 |    36.33 |
|   512 |    128 |  15872 |    0.617 |   829.62 |    3.547 |    36.08 |
|   512 |    128 |  16384 |    0.623 |   821.93 |    3.627 |    35.29 |

</details>

Nice job, the `IQ2_KT` is quite speedy (relative to the ~4bpw quants)!

Somewhat related I [saw further discussions](https://github.com/turboderp-org/exllamav3/pull/26#issuecomment-2957155162) on optimizing QTIP style quants by using pre-computed Hessians for each layer/tensor. Zero pressure to look or distract, just interesting folks are already uploading Hessians for some models.

---

👤 **ikawrakow** commented on **2025-06-11** at **14:36:11**

> Somewhat related I https://github.com/turboderp-org/exllamav3/pull/26#issuecomment-2957155162 on optimizing QTIP style quants by using pre-computed Hessians for each layer/tensor. Zero pressure to look or distract, just interesting folks are already uploading Hessians for some models.

This is the sort of thing we do not want to do here. It leads to overfitting, needs a huge amount of compute, which makes it inaccessible for the average enthusiast, so basically only good for pushing out yet another paper to arXiv.

---

👤 **louiehelm** commented on **2025-06-11** at **17:03:36**

Great work! Love seeing improved performance on the trellis quants ik.

Some alternate MCG multipliers (with no addition) have lower PPL than QTIP 3INST defaults:

### Meta-Llama-3.1-8B-Instruct
| **Quantization** | **Version** | **PPL** |
|------------------|-------------|---------|
| **f32** | - | 7.3210 |
| **IQ2_KT** | [#511](https://github.com/ikawrakow/ik_llama.cpp/issues/511) default | 11.0029 |
| | 0xCBAC1FED (3417055213) | 10.9466 |
| **IQ3_KT** | [#511](https://github.com/ikawrakow/ik_llama.cpp/issues/511) default | 8.1319 |
| | 0xCBAC1FED (3417055213) | 8.0776 |
| **IQ4_KT** | [#511](https://github.com/ikawrakow/ik_llama.cpp/issues/511) default | 7.5620 |
| | 0xCBAC1FED (3417055213) | 7.5591 |

Just chiming in because it might be a great time to take the 0.5% higher fidelity of ditching the default QTIP multiplier+addition params if you're already introducing a breaking change to IQx_KT quants anyway. For IQ2_K, this gains back a good chunk of what was lost by switching to your new decoder scheme, while also making IQ3_KT and IQ4_KT both better than [#511](https://github.com/ikawrakow/ik_llama.cpp/issues/511) and in some cases even better than prior versions.

Also, ka = `0xCBAC1FED`  and kb = 0 is a more well-tested distribution than 3INST defaults and currently the best known so far. Obviously if this change is added kb can be deleted rather than updated to 0 (for a small speed boost). This is how to test it further with more models to confirm PPL shows improvements more broadly:

`./test_IQ2_KT.sh 3417055213`

```
#!/bin/sh

find . -type f \( -name "*.cpp" -o -name "*.cu" -o -name "*.cuh" \) -exec sed -i "s/ ka = 89226354/ ka = $1/g" {} +
find . -type f \( -name "*.cpp" -o -name "*.cu" -o -name "*.cuh" \) -exec sed -i "s/ kb = 64248484/ kb = 0/g" {} +
cmake -B build -DGGML_CUDA=ON -DGGML_RPC=OFF -DGGML_BLAS=OFF  -DGGML_SCHED_MAX_COPIES=1
cmake --build build --config Release -j $(nproc)
find . -type f \( -name "*.cpp" -o -name "*.cu" -o -name "*.cuh" \) -exec sed -i "s/ ka = $1/ ka = 89226354/g" {} +
find . -type f \( -name "*.cpp" -o -name "*.cu" -o -name "*.cuh" \) -exec sed -i "s/ kb = 0/ kb = 64248484/g" {} +

build/bin/llama-quantize --imatrix ~/llms/Meta-Llama-3.1-8B-Instruct-f32-imatrix.dat ~/llms/Meta-Llama-3.1-8B-Instruct-f32.gguf Meta-Llama-3.1-8B-Instruct-IQ2_KT.gguf IQ2_KT
# build/bin/llama-perplexity -m ~/llms/Meta-Llama-3.1-8B-Instruct-f32.gguf -f ~/llms/wiki.test.raw  --ctx-size 512 --ubatch-size 512 -fa -ngl 99 --seed 1337 # BASELINE TEST

build/bin/llama-perplexity -m Meta-Llama-3.1-8B-Instruct-IQ2_KT.gguf -f ~/llms/wiki.test.raw --ctx-size 512 --ubatch-size 512 -fa -ngl 99 --seed 1337

rm -f Meta-Llama-3.1-8B-Instruct-IQ2_KT.gguf
```

---

👤 **ikawrakow** commented on **2025-06-12** at **08:16:34**

@louiehelm Thank you for the comment, looks very promising. It should also improve performance slightly by saving one integer addition.

Do I understand correctly that you applied the new multiplier to PR [#511](https://github.com/ikawrakow/ik_llama.cpp/issues/511) instead of the original implementation on the main branch?

Did you also try models other than LlaMA-3.1-8B-Instruct?

---

👤 **louiehelm** commented on **2025-06-12** at **22:27:27**

Yes initial tests above were on [#511](https://github.com/ikawrakow/ik_llama.cpp/issues/511). Needs more testing... Qwen3 1.7B IQ2_KT = 2.5% lower PPL.... Magistral 24B IQ2_KT = 50% lower PPL [default model bugged perhaps?]

---

👤 **Nexesenex** commented on **2025-06-13** at **10:20:31**

On Gemma 3 27b qat unquantized (iq2_kt for ffn_up, ffn_gate, attn_q, attn_k and attn_o, iq4_ks for ffn_down, q4_0 for attn_v, and q6 for embed/output), I obtained an almost equivalent perplexity wikitest 512 between the original couple ka/kb and louiehelm's.

But on a Llama 3.3 70b type model (iq2_kt for the ffns, attn_q and attn_o, q6 for embedding, iq5_ks_r4 for output and attn_v, and iq4_ks_r4 for attn_k), the final wikitest 512 perplexity is 1% lower with ka = 3417055213 and kb = 0 compared to the original couple.

With an IQ3_KT with a Cuda MMQ Kernel, and ffn_down/attn_o in iq3_KT, a Llama 3 70b on mono 24GB GPU will become really viable.

---

👤 **ikawrakow** commented on **2025-06-13** at **10:25:19**

> But on a Llama 3.3 70b type model (iq2_kt for the ffns, attn_q and attn_o), the final wikitest 512 perplexity is 1% lower with ka = 3417055213 and kb = 0 compared to the original couple.

1% of what? Can you give the specific PPL values?

---

👤 **Nexesenex** commented on **2025-06-13** at **10:32:43**

> > But on a Llama 3.3 70b type model (iq2_kt for the ffns, attn_q and attn_o), the final wikitest 512 perplexity is 1% lower with ka = 3417055213 and kb = 0 compared to the original couple.
> 
> 1% of what? Can you give the specific PPL values?

Here is :

For Llama 3.3 70b type model (a merge, not the original 3.3 70b ; iq2_kt for the ffns, attn_q and attn_o, q6 for embedding, iq5_ks_r4 for output and attn_v, and iq4_ks_r4 for attn_k).
- final wikitest 512 perplexity with ka = 89226354 and kb = 64248484. -> 6.1443 +/- 0.03805
- final wikitest 512 perplexity is 1% lower with ka = 3417055213 and kb = 0. -> 6.0739 +/- 0.03762

For Gemma 3 27b qat unquantized (iq2_kt for ffn_up, ffn_gate, attn_q, attn_k and attn_o, iq4_ks for ffn_down, q4_0 for attn_v, and q6 for embed/output).
- final wikitest 512 perplexity with ka = 89226354 and kb = 64248484. -> 8.9993 +/- 0.06887 (and the intermediate values are often lower by 0.01-0.03).
- final wikitest 512 perplexity with ka = 3417055213 and kb = 0. -> 9.0001 +/- 0.06897

---

👤 **ikawrakow** commented on **2025-06-13** at **16:59:17**

Did you also try `IQ4_KT`?

I tried LlaMA-3.1-8B-Instruct and PPL goes up by ~0.5%, which is a lot for 4 bit. `IQ2_KT` has 30-40% quantization error, so 1% improvement is not that much. But `IQ4_KT` has 2.5% quantization error, so a 0.5% increase is not good. Strangely enough, with this multiplier `IQ4_KT` quantization takes much longer, while `IQ2_KT` quantization becomes faster.

I only changed the CUDA implementation so I can run PPL. When I make the change in the CPU code I'll push to a new branch. Probably tomorrow.

---

👤 **ubergarm** commented on **2025-06-13** at **18:52:10**

> Did you also try IQ4_KT?

Just got home and tried louiehelm's 0xCBAC1FED patch on this PR511.

### Patch

<details>

<summary>👈 `0xCBAC1FED` Patch</summary>

```diff
diff --git a/ggml/src/ggml-cuda/convert.cu b/ggml/src/ggml-cuda/convert.cu
index a602e47d..45de337e 100644
--- a/ggml/src/ggml-cuda/convert.cu
+++ b/ggml/src/ggml-cuda/convert.cu
@@ -341,15 +341,15 @@ inline __device__ int nearest_int(float fval) {
 }
 
 int __device__ __forceinline__ trellis_next_int(uint32_t& val) {
-    constexpr uint32_t ka = 89226354;
-    constexpr uint32_t kb = 64248484;
+    constexpr uint32_t ka = 3417055213;
+    constexpr uint32_t kb = 0;
     val = ka*val + kb;
     return ggml_cuda_dp4a(val & 0x3f3f3f3f, 0x01010101, -126);
 }
 
 float __device__ __forceinline__ trellis_next(uint32_t& val) {
-    constexpr uint32_t ka = 89226354;
-    constexpr uint32_t kb = 64248484;
+    constexpr uint32_t ka = 3417055213;
+    constexpr uint32_t kb = 0;
     constexpr uint32_t kmask = 0x8fff8fff;
     constexpr uint32_t km32 = 0x3b603b60;
     uint32_t s;
diff --git a/ggml/src/ggml-cuda/dmmv.cu b/ggml/src/ggml-cuda/dmmv.cu
index 50e6458d..5e0226ed 100644
--- a/ggml/src/ggml-cuda/dmmv.cu
+++ b/ggml/src/ggml-cuda/dmmv.cu
@@ -16,8 +16,8 @@ static_assert(K_QUANTS_PER_ITERATION == 1 || K_QUANTS_PER_ITERATION == 2, "K_QUA
 #endif
 
 static __device__ __forceinline__ uint32_t trellis_next(uint32_t& val) {
-    constexpr uint32_t ka = 89226354;
-    constexpr uint32_t kb = 64248484;
+    constexpr uint32_t ka = 3417055213;
+    constexpr uint32_t kb = 0;
     constexpr uint32_t kmask = 0x8fff8fff;
     constexpr uint32_t km32 = 0x3b603b60;
     val = ka*val + kb;
diff --git a/ggml/src/ggml-cuda/iqk_mmvq.cu b/ggml/src/ggml-cuda/iqk_mmvq.cu
index df1cea89..34402358 100644
--- a/ggml/src/ggml-cuda/iqk_mmvq.cu
+++ b/ggml/src/ggml-cuda/iqk_mmvq.cu
@@ -398,8 +398,8 @@ __device__ __forceinline__ void vec_dot_iq4_ks_q8_1(
 __device__ __forceinline__ void vec_dot_iq4_kt_q8_1(
     const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {
 
-    constexpr uint32_t ka = 89226354;
-    constexpr uint32_t kb = 64248484;
+    constexpr uint32_t ka = 3417055213;
+    constexpr uint32_t kb = 0;
     constexpr uint32_t km = 0x3f3f3f3f;
 
     float scale = *(const float *)vbq;
@@ -436,8 +436,8 @@ __device__ __forceinline__ void vec_dot_iq4_kt_q8_1(
 __device__ __forceinline__ void vec_dot_iq2_kt_q8_1(
     const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {
 
-    constexpr uint32_t ka = 89226354;
-    constexpr uint32_t kb = 64248484;
+    constexpr uint32_t ka = 3417055213;
+    constexpr uint32_t kb = 0;
     constexpr uint32_t km = 0x3f3f3f3f;
 
     float scale = *(const float *)vbq;
diff --git a/ggml/src/ggml-cuda/mmq.cuh b/ggml/src/ggml-cuda/mmq.cuh
index e2c76a85..2b5a6df5 100644
--- a/ggml/src/ggml-cuda/mmq.cuh
+++ b/ggml/src/ggml-cuda/mmq.cuh
@@ -2799,8 +2799,8 @@ template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinlin
 template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq4_kt(
     const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {
 
-    constexpr uint32_t ka = 89226354;
-    constexpr uint32_t kb = 64248484;
+    constexpr uint32_t ka = 3417055213;
+    constexpr uint32_t kb = 0;
     constexpr uint32_t km = 0x3f3f3f3f;
 
 #ifdef INT8_MMA_AVAILABLE
@@ -2872,8 +2872,8 @@ template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinlin
 template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq2_kt(
     const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {
 
-    constexpr uint32_t ka = 89226354;
-    constexpr uint32_t kb = 64248484;
+    constexpr uint32_t ka = 3417055213;
+    constexpr uint32_t kb = 0;
     constexpr uint32_t km = 0x3f3f3f3f;
 
 #ifdef INT8_MMA_AVAILABLE
diff --git a/ggml/src/iqk/iqk_gemm_ktquants.cpp b/ggml/src/iqk/iqk_gemm_ktquants.cpp
index 8b8cae14..41b9b2d6 100644
--- a/ggml/src/iqk/iqk_gemm_ktquants.cpp
+++ b/ggml/src/iqk/iqk_gemm_ktquants.cpp
@@ -14,8 +14,8 @@
 namespace {
 
 inline uint32_t trellis_next(uint32_t& val) {
-    constexpr uint32_t ka = 89226354;
-    constexpr uint32_t kb = 64248484;
+    constexpr uint32_t ka = 3417055213;
+    constexpr uint32_t kb = 0;
     constexpr uint32_t kmask = 0x8fff8fff;
     constexpr uint32_t km32 = 0x3b603b60;
     val = val*ka + kb;
@@ -31,8 +31,8 @@ inline float trellis_gen(uint32_t& val, uint32_t* s) {
 struct Trellis1 {
     constexpr static uint32_t kmask = 0x8fff8fff;
     constexpr static uint32_t km32 = 0x3b603b60;
-    constexpr static uint32_t ka = 89226354;
-    constexpr static uint32_t kb = 64248484;
+    constexpr static uint32_t ka = 3417055213;
+    constexpr static uint32_t kb = 0;
     constexpr static uint32_t ka1 = ka*ka;
     constexpr static uint32_t kb1 = kb*ka+kb;
     constexpr static uint32_t ka2 = ka1*ka;
@@ -76,8 +76,8 @@ inline __m256 trellis_gen8(__m256i i8) {
 struct Trellis2 {
     constexpr static uint32_t kmask = 0x8fff8fff;
     constexpr static uint32_t km32 = 0x3b603b60;
-    constexpr static uint32_t ka = 89226354;
-    constexpr static uint32_t kb = 64248484;
+    constexpr static uint32_t ka = 3417055213;
+    constexpr static uint32_t kb = 0;
     constexpr static uint32_t ka1 = ka*ka;
     constexpr static uint32_t kb1 = kb*ka+kb;
     constexpr static uint32_t ka2 = ka1*ka;
@@ -100,8 +100,8 @@ struct Trellis2 {
 
 template <bool is_8 = false>
 struct Trellis3 {
-    constexpr static uint32_t ka = 89226354;
-    constexpr static uint32_t kb = 64248484;
+    constexpr static uint32_t ka = 3417055213;
+    constexpr static uint32_t kb = 0;
     constexpr static uint32_t ka1 = ka*ka;
     constexpr static uint32_t kb1 = kb*ka+kb;
     constexpr static uint32_t ka2 = ka1*ka;
@@ -913,8 +913,8 @@ namespace {
 struct Trellis1 {
     constexpr static uint32_t kmask = 0x8fff8fff;
     constexpr static uint32_t km32 = 0x3b603b60;
-    constexpr static uint32_t ka = 89226354;
-    constexpr static uint32_t kb = 64248484;
+    constexpr static uint32_t ka = 3417055213;
+    constexpr static uint32_t kb = 0;
     constexpr static uint32_t ka1 = ka*ka;
     constexpr static uint32_t kb1 = kb*ka+kb;
     constexpr static uint32_t ka2 = ka1*ka;
@@ -1419,8 +1419,8 @@ void mul_mat_iq4_kt_F32_T(int n, const void * vx, size_t bx, const DataInfo& inf
 }
 
 struct Trellis3 {
-    constexpr static uint32_t ka = 89226354;
-    constexpr static uint32_t kb = 64248484;
+    constexpr static uint32_t ka = 3417055213;
+    constexpr static uint32_t kb = 0;
     constexpr static uint32_t ka1 = ka*ka;
     constexpr static uint32_t kb1 = kb*ka+kb;
     constexpr static uint32_t ka2 = ka1*ka;
diff --git a/ggml/src/iqk/iqk_quantize.cpp b/ggml/src/iqk/iqk_quantize.cpp
index b6bff0a1..7c052989 100644
--- a/ggml/src/iqk/iqk_quantize.cpp
+++ b/ggml/src/iqk/iqk_quantize.cpp
@@ -7454,8 +7454,8 @@ public:
     inline float find_best_inverse_scale(const float * xb, const float * weight, const int * best_idx) const;
 
     static inline void set_values(uint32_t i, float * result, float scale, int offset = 4096) {
-        constexpr uint32_t ka = 89226354;
-        constexpr uint32_t kb = 64248484;
+        constexpr uint32_t ka = 3417055213;
+        constexpr uint32_t kb = 0;
         uint32_t x = i + offset;
         if constexpr (is_int) {
             uint32_t s;
```

</details>

### Data
Here is the comparison of the same [OpenBuddy-R1-0528-Distill-Qwen3-32B-Preview0-QAT](https://huggingface.co/OpenBuddy/OpenBuddy-R1-0528-Distill-Qwen3-32B-Preview0-QAT) used above between regular PR511 and the patched version.

#### PR511 (from above)
* IQ4_KT
  - `7.0114 +/- 0.04516`
  - `main: quantize time = 1465481.74 ms` 24.42 min
* IQ2_KT (token_embd|output)@iq4_kt
  - `8.7412 +/- 0.05859`
  - `main: quantize time = 865473.26 ms` 14.42 min

#### 0xCBAC1FED Patch
* IQ4_KT
  - `7.0210 +/- 0.04529`
  - `main: quantize time = 1518609.40 ms` 25.31 min
* IQ2_KT (token_embd|output)@iq4_kt
  - `8.6883 +/- 0.05866`
  - `main: quantize time = 877350.58 ms` 14.62 min

### Comparison
* IQ4_KT
  - Patched version is ~0.14% "worse" perplexity
  - Patched version quantized ~3.6% slower
* IQ2_KT (token_embd|output)@iq4_kt
  - Patched version is ~0.61% "better" perplexity
  - Patched version quantized ~1.4% slower

### Conclusion
Well, its hard to say for a single run given the deltas seem within the margin of error. I'm not sure if it is possible/worthwhile to save the `ka`/`kb` values into the GGUF metadata and load them per model to support both? This would allow any future discovered magic numbers as well (couldn't optimize away kb=0 though).

---

👤 **ikawrakow** commented on **2025-06-18** at **13:21:51**

Closing in favor of [#529](https://github.com/ikawrakow/ik_llama.cpp/issues/529)