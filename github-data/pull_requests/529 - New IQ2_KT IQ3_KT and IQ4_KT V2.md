## 🔀 [Pull Request #529](https://github.com/ikawrakow/ik_llama.cpp/pull/529) - New IQ2_KT, IQ3_KT and IQ4_KT, V2

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/new_iq2kt_v2` |
| **Target Branch** | `main` |
| **Created** | 2025-06-14 |
| **Updated** | 2025-06-18 |
| **Merged** | 2025-06-18 |
| **Labels** | `Breaking change` |

---

## 📄 Description

This PR is the combination of [#505](https://github.com/ikawrakow/ik_llama.cpp/issues/505) and [#511](https://github.com/ikawrakow/ik_llama.cpp/issues/511), but rebased on current main, and using @louiehelm's alternative multiplier (see comments in [#511](https://github.com/ikawrakow/ik_llama.cpp/issues/511)). 

I was curios to see if not having an extra addition per step when generating the trellis sequence will have a pefromance impact, so made a proper change rather than just blindly replacing the two constants using `sed`. On CUDA performance impact is negligible, on `AVX2` we see 1-2% improvement.

With the latest commits I have also adapted `IQ3_KT` to the integer trellis.

---

## 💬 Conversation

👤 **ubergarm** commented on **2025-06-14** at **17:55:02**

## tl;dr;
Something seems off as perplexity on both of my new tests for this PR529 are much higher now than a previous attempt around June 3rd with commits around [PR484](https://github.com/ikawrakow/ik_llama.cpp/pull/484). I double checked my logs and commands and confirmed using the same imatrix etc so not sure what is going on. I've been compiling for CPU only fwiw. Details below.

## Experiment
Okay, doing a fresh test using this new PR529 on DeepSeek-R1-0528. I made two almost identical quants that differ only in the commit used to quantize/test/benchmark. Quantization was done roughly simultaneously, one on each socket of a dual socket intel xeon 6980P.

### Common Recipe

* 218.877 GiB (2.798 BPW)
* type  f32:  361 tensors
* type q5_0:   61 tensors - `attn_k_b`
* type iq2_kt:  116 tensors - `ffn_(gate|up)_exps`
* type iq4_kt:  609 tensors - everything else

### Test Cases

1. `ik/new_iq2kt_v2@e5a06688`
  * `mix-IQ4_KT-0xCBAC1FED`
  * including louiehelm's multiplier
  * quantize time = 15666814.63 ms - 4.35 hours
  * Final estimate: PPL = 13.2237 +/- 0.09673
  * *UPDATE w/ 6408b94 CPU fix* `Final estimate: PPL = 3.5808 +/- 0.01943`
```
INFO [                    main] build info | tid="135292499650880" timestamp=1749922901 build=3776 commit="e5a06688"
INFO [                    main] system info | tid="135292499650880" timestamp=1749922901 n_threads=80 n_threads_batch=128 total_threads=512 system_inf
o="AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 |
 F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
```

2. `ik/new_iq2kt_v2@b1416bf0` 
  * `mix-IQ4_KT-og`
  * two commits earlier, *without* louiehelm's multiplier
  * quantize time = 15890223.61 ms - 4.41 hours
  * Final estimate: PPL = 13.1972 +/- 0.09621
```
INFO [                    main] build info | tid="133117239363904" timestamp=1749922843 build=3774 commit="b1416bf0"
INFO [                    main] system info | tid="133117239363904" timestamp=1749922843 n_threads=80 n_threads_batch=128 total_threads=512 system_inf
o="AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 |
 F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
```

<details>

<summary>👈 Perplexity Command</summary>

### Perplexity
```bash
#model=/mnt/raid/models/ubergarm/DeepSeek-R1-0528-GGUF/DeepSeek-R1-0528-mix-IQ4_KT-og.gguf
model=/mnt/raid/models/ubergarm/DeepSeek-R1-0528-GGUF/DeepSeek-R1-0528-mix-IQ4_KT-0xCBAC1FED.gguf

numactl -N 0 -m 0 \
./build/bin/llama-perplexity \
    --model "$model" \
    -ctk f16 \
    -mla 3 -fa \
    -amb 512 \
    -fmoe \
    -f wiki.test.raw \
    --seed 1337 \
    --no-mmap \
    --threads 128 \
    --numa numactl
```

</details>

### llama-sweep-bench

![sweep-bench-pr529](https://github.com/user-attachments/assets/1edccf5f-b72b-4eb5-837c-a82eb233dc3d)

<details>

<summary>👈 llama-sweep-bench logs</summary>

```bash
#!/usr/bin/env bash

#model=/mnt/raid/models/ubergarm/DeepSeek-R1-0528-GGUF/DeepSeek-R1-0528-mix-IQ4_KT-0xCBAC1FED.gguf
model=/mnt/raid/models/ubergarm/DeepSeek-R1-0528-GGUF/DeepSeek-R1-0528-mix-IQ4_KT-og.gguf

numactl -N 1 -m 1 \
./build/bin/llama-sweep-bench \
    --model "$model" \
    -c 8704 \
    -ctk f16 \
    -mla 3 -fa \
    -fmoe \
    --no-mmap \
    --threads 80 \
    --threads-batch 128 \
    --numa numactl \
    --warmup-batch
```

#### DeepSeek-R1-0528-mix-IQ4_KT-og b1416bf0
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    4.884 |   104.83 |   20.927 |     6.12 |
|   512 |    128 |    512 |    5.235 |    97.80 |   26.596 |     4.81 |
|   512 |    128 |   1024 |    5.670 |    90.31 |   20.470 |     6.25 |
|   512 |    128 |   1536 |    6.087 |    84.12 |   27.983 |     4.57 |
|   512 |    128 |   2048 |    7.085 |    72.27 |   29.016 |     4.41 |
|   512 |    128 |   2560 |    7.906 |    64.76 |   28.767 |     4.45 |
|   512 |    128 |   3072 |    7.373 |    69.44 |   28.416 |     4.50 |
|   512 |    128 |   3584 |    8.216 |    62.32 |   20.452 |     6.26 |
|   512 |    128 |   4096 |    9.451 |    54.17 |   19.672 |     6.51 |
|   512 |    128 |   4608 |    9.573 |    53.49 |   20.232 |     6.33 |
|   512 |    128 |   5120 |    9.966 |    51.37 |   20.479 |     6.25 |
|   512 |    128 |   5632 |   10.774 |    47.52 |   24.437 |     5.24 |
|   512 |    128 |   6144 |   11.816 |    43.33 |   22.064 |     5.80 |
|   512 |    128 |   6656 |   12.937 |    39.58 |   21.809 |     5.87 |
|   512 |    128 |   7168 |   12.519 |    40.90 |   27.118 |     4.72 |
|   512 |    128 |   7680 |   13.039 |    39.27 |   29.001 |     4.41 |
|   512 |    128 |   8192 |   13.726 |    37.30 |   22.418 |     5.71 |

#### DeepSeek-R1-0528-mix-IQ4_KT-0xCBAC1FED e5a06688
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    4.699 |   108.96 |   20.847 |     6.14 |
|   512 |    128 |    512 |    6.078 |    84.24 |   25.537 |     5.01 |
|   512 |    128 |   1024 |    5.355 |    95.62 |   21.164 |     6.05 |
|   512 |    128 |   1536 |    5.858 |    87.41 |   25.636 |     4.99 |
|   512 |    128 |   2048 |    6.297 |    81.31 |   26.767 |     4.78 |
|   512 |    128 |   2560 |    7.047 |    72.66 |   24.938 |     5.13 |
|   512 |    128 |   3072 |    7.227 |    70.85 |   19.698 |     6.50 |
|   512 |    128 |   3584 |    8.237 |    62.16 |   23.806 |     5.38 |
|   512 |    128 |   4096 |    8.208 |    62.38 |   19.898 |     6.43 |
|   512 |    128 |   4608 |    9.000 |    56.89 |   23.857 |     5.37 |
|   512 |    128 |   5120 |    9.589 |    53.39 |   21.710 |     5.90 |
|   512 |    128 |   5632 |   10.344 |    49.50 |   25.217 |     5.08 |
|   512 |    128 |   6144 |   11.087 |    46.18 |   23.658 |     5.41 |
|   512 |    128 |   6656 |   12.194 |    41.99 |   24.630 |     5.20 |
|   512 |    128 |   7168 |   11.945 |    42.86 |   21.043 |     6.08 |
|   512 |    128 |   7680 |   12.517 |    40.91 |   22.126 |     5.79 |
|   512 |    128 |   8192 |   13.231 |    38.70 |   22.478 |     5.69 |

</details>

### Conclusion
Huh, the perplexity of ~13.2 on *both* of these seems surprisingly "bad" relative to my earlier first test I had done with a smaller mix `IQ2_KT` 196.696 GiB (2.514 BPW) with `Final estimate: PPL = 3.6378 +/- 0.01997` and my other various mixes on huggingface shown in the graph below.

```
# https://github.com/ikawrakow/ik_llama.cpp/pull/484
# quantized on June 2nd, 2025: build = 3726 (061d064b)
# perplexity on June 3rd, 2025: build = 3724 (626f49ab)
# compiled and tested CPU-only on 24x core 7965WX
system_info: n_threads = 24 / 48 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NE
ON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1
```
- type  f32:  361 tensors
- type q5_0:   61 tensors - `attn_k_b`
- type iq2_kt:  116 tensors - `ffn_(gate|up)_exps`
- type iq3_kt:   58 tensors - `ffn_down_exps`
- type iq4_kt:  551 tensors - everything else

![ppl-r1-0528-dq4_k_r4-ubergarm](https://github.com/user-attachments/assets/4e3ea5ca-ddec-4052-bc06-0df185dc1b47)

I did confirm that both these new test case models give a reasonable answer for my usual `Count from 1 to 10 in French.` with no gibberish output.

Not sure what to try next... Possibly try comparing perplexity of a smaller quant on this PR529 vs older PR484 with exact same recipe? Maybe the new recipe is actually worse despite being larger?

Happy to provide more logs as requested. Thanks!

---

👤 **ubergarm** commented on **2025-06-14** at **21:30:05**

Okay, did one more faster experiment using the *same* recipe/imatrix for Qwen3-30B-A3B moe. Something is off between this PR529 and main's implementation of a "pure" `iq4_kt` when checking llama-perplexity compiled CPU only:

* PR529@e5a06688
  - Final estimate: PPL = 142.3478 +/- 1.47226
  - total time =  479886.25 ms / 299009 tokens
* main@6fc5bbb6
  - Final estimate: PPL = 9.3612 +/- 0.07518
  - total time =  585627.38 ms / 299009 tokens
* PR511 ik/new_iq2kt@c8cf1280
   - Odd, quantized one to sanity check this, but now getting `GGML_ASSERT(fms.S[j] > 0)` on the intel xeon 6980P... hrmm... i used the Thread Ripper Pro in my testing over on that PR yesterday compiled with CUDA...

- Qwen3-30B-A3B
- 14.344 GiB (4.035 BPW)
- type  f32:  241 tensors
- type iq4_kt:  338 tensors

---

👤 **ubergarm** commented on **2025-06-15** at **15:54:01**

Okay, back to the basics as my sanity is thin. I used the Thread Ripper Pro 24x Core with RTX A6000 GPUs to test. 

### tl;dr;
The CUDA implementation of this PR529 seems to give reasonable perplexity. However compiling CPU-only gives *much* higher perplexity testing the same quant.

### Experiment
1. I cooked a "pure" `iq4_kt` Qwen3-30B-A3B 14.344 GiB (4.035 BPW) quant using this `PR529@e5a06688` 
2. Compiled with CUDA
  * `cmake -B ./build -DGGML_CUDA=ON -DGGML_BLAS=OFF -DGGML_SCHED_MAX_COPIES=1 -DGGML_CUDA_IQK_FORCE_BF16=1 -DGGML_CUDA_F16=ON`
  * `system_info: n_threads = 1 / 48 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 |`
  * `Final estimate: PPL = 9.2514 +/- 0.07376`
3. Compiled with CPU *only*
  * `cmake -B build -DGGML_CUDA=OFF -DGGML_BLAS=OFF`
  * `system_info: n_threads = 24 / 48 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 |`
  * `Final estimate: PPL = 922.0458 +/- 10.91332`

---

👤 **ikawrakow** commented on **2025-06-15** at **16:06:42**

PPL = 922 means I have a bug in the CPU implementation. I haven't come around to check.

---

👤 **ubergarm** commented on **2025-06-15** at **16:14:28**

All good no rush. Just wanted to re-create the issue on a "known working" system for my own peace of mind hah.

If it is useful for anyone else testing, I'll make this experimental [Qwen3-30B-A3B-IQ4_KT-PR529-e5a06688.gguf](http://emptyduck.com/Qwen3-30B-A3B-IQ4_KT-PR529-e5a06688.gguf) available from my personal server for a few days. ~15GiB with sha256sum `c47dd5298181806608fe6dc585d7f1ba2387788881a68be85ff42655e03ce453`

---

👤 **ikawrakow** commented on **2025-06-16** at **12:02:54**

The CPU bug is fixed now.

I get quite a bit lower PPL using
```
./bin/llama-quantize --imatrix qwen3_imat_unsloth.dat --output-tensor-type q8_0 --token-embedding-type q8_0 --pure
```
(didn't want to risk something going wrong in the output tensor or the token embeddings)

* CUDA: `Final estimate: PPL = 9.0801 +/- 0.07115`
* CPU:     `Final estimate: PPL = 9.0781 +/- 0.07113`

---

👤 **ubergarm** commented on **2025-06-16** at **14:40:28**

Aye, that did the trick for qwen3moe:

* CUDA: `Final estimate: PPL = 9.2514 +/- 0.07376`
* CPU: `Final estimate: PPL = 9.2557 +/- 0.07382`

I'll come back around with some more results soon thanks!

---

👤 **ikawrakow** commented on **2025-06-16** at **14:42:50**

But why is your PPL so much higher?

---

👤 **ubergarm** commented on **2025-06-16** at **14:57:16**

> But why is your PPL so much higher?

This was my quant from yesterday "pure" `iq4_kt`:
```
llama_model_loader: - type  f32:  241 tensors
llama_model_loader: - type iq4_kt:  338 tensors
```

I'll use your command with my imatrix now and test again.
```
./bin/llama-quantize --imatrix qwen3_imat_unsloth.dat --output-tensor-type q8_0 --token-embedding-type q8_0 --pure
```

I'm assuming the higher bpw output/token_embd accounts for most of the discrepancy.

---

*UPDATE*

Results with the IQ4_KT using q8_0 for embedding/output are still higher for me. Discrepency could be because you use the unsloth imatrix dat. My imatrix dat is older using only [imatrix calibration_data_v5_rc.txt](https://gist.github.com/tristandruyen/9e207a95c7d75ddf37525d353e00659c#file-calibration_data_v5_rc-txt).

My newer imatrix corpus adds extra data in an attempt to activate more experts, but I never went back and updated my Qwen3-30B-A3Bs with it. I believe both unsloth and bartowski used bigger corpus for qwen3moe due to issues quantizing at lower BPW with their usual corpus text.

* GPU: `Final estimate: PPL = 9.2301 +/- 0.07378`
* CPU: `Final estimate: PPL = 9.2279 +/- 0.07375`

---

👤 **ikawrakow** commented on **2025-06-16** at **16:30:16**

> Results with the IQ4_KT using q8_0 for embedding/output are still higher for me.

Must be the imatrix, then. I used the one [from Unsloth](https://huggingface.co/unsloth/Qwen3-30B-A3B-128K-GGUF/blob/main/imatrix_unsloth.dat), which produced the lowest PPL in my Qwen3 quantization experiments ([#359](https://github.com/ikawrakow/ik_llama.cpp/issues/359))

---

👤 **Nexesenex** commented on **2025-06-17** at **01:34:47**

llama-perplexity -m Configurable-Llama-3.1-8B-Instruct_iMat-IQ3_KT_Nv2_embed_q6_0_output&attn_v_iq5ksr4_attn_k_iq4ksr4.gguf -f wiki.test.raw -ngl 150 -b 512 -mg 0 -ts 40,0,0 --no-mmap -fa -c 512
llama_model_loader: - type  f32:   66 tensors
llama_model_loader: - type q6_0:    1 tensors
llama_model_loader: - type iq3_kt:  160 tensors
llama_model_loader: - type iq4_ks_r4:   32 tensors
llama_model_loader: - type iq5_ks_r4:   33 tensors
llm_load_print_meta: model ftype      = IQ3_KT - 3.125 bpw
llm_load_print_meta: model size       = 3.315 GiB (3.546 BPW)
llm_load_print_meta: repeating layers = 2.596 GiB (3.195 BPW, 6.980 B parameters)

Final estimate: PPL = 8.1431 +/- 0.05213

IQ3_KT's PPL works for me on CUDA. It also infers on both CPU and CUDA.

llama-perplexity -m Configurable-Llama-3.1-8B-Instruct_iMat-IQ3_XXS_embed_q6_0_output&attn_v_iq5ksr4_attn_k_iq4ksr4.gguf -f wiki.test.raw -ngl 150 -b 512 -mg 0 -ts 40,0,0 --no-mmap -fa -c 512
llama_model_loader: - type  f32:   66 tensors
llama_model_loader: - type iq3_xxs:  160 tensors
llama_model_loader: - type q6_0:    1 tensors
llama_model_loader: - type iq4_ks_r4:   32 tensors
llama_model_loader: - type iq5_ks_r4:   33 tensors
llm_load_print_meta: model ftype      = IQ3_XXS - 3.0625 bpw
llm_load_print_meta: model size       = 3.261 GiB (3.489 BPW)
llm_load_print_meta: repeating layers = 2.542 GiB (3.129 BPW, 6.980 B parameters)

Final estimate: PPL = 8.4642 +/- 0.05423

IQ3_XXS has some serious competition, quant quality wise.

Same recipe, but with IQ3_S tensors instead of IQ3_KT/IQ3_XXS : Final estimate: PPL = 7.9331 +/- 0.05065
With IQ3_K : Final estimate: PPL = 7.9098 +/- 0.05097
With Q3_K : Final estimate: PPL = 8.1488 +/- 0.05292 (IQ3_KT went below!)

Note: this version of Llama 8B gives a PPL of 7.3287 +/- 0.04703 for Q8_0, so very close to the original.

---

👤 **ubergarm** commented on **2025-06-17** at **03:53:27**

> With the latest commits I have also adapted IQ3_KT to the integer trellis.

I saw this and started cooking asap targeting ~3.5bpw for [some recent requests on :hugs: ](https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF/discussions/7). Not releasing anything yet, just experimenting for funzies.

* `DeepSeek-R1-0528-IQ3_KT`
  - 272.527 GiB (3.483 BPW)
  - quantize time = 8 hours 48 minutes
  - `Final estimate: PPL = 3.3056 +/- 0.01758`
  - (beats the "unsloth dynamic" 275.576GiB `UD-Q3_K_XL` at `3.3341 +/- 0.01784`)
  - `Cor(ln(PPL(Q)), ln(PPL(base))):  99.69%` on `ubergarm-kld-test-corpus-short.txt`
  -    f32:  361 tensors
  -   q5_0:   61 tensors `attn_k_b`
  -   q8_0:    1 tensors `token_embd`
  - iq5_ks:  550 tensors `attn/shexp`
  - iq3_kt:  116 tensors `ffn_(gate|up)_exps`
  - iq4_kt:   58 tensors `ffn_down_exps`

About the largest size quant fitting 256GB RAM ~48+GB VRAM rigs. I'm offloading additional 7 or 8 `exps` layers each on dual RTX A6000's using ~43+GB out of 48GB VRAM each with the remaining routed `exps` on CPU/RAM.

<details>

<summary>👈 2x GPU offload Perplexity Command </summary>

```bash
./build/bin/llama-perplexity \
    --model "$model" \
    -f wiki.test.raw \
    --seed 1337 \
    -ctk f16 \
    -mla 3 -fa \
    -fmoe \
    -amb 512 \
    -ngl 99 \
    -ot "blk\.(3|4|5|6|7|8|9|10)\.ffn_.*=CUDA0" \
    -ot "blk\.(11|12|13|14|15|16|17|18)\.ffn_.*=CUDA1" \
    -ot exps=CPU \
    --threads 24
```

</details>

![ppl-r1-0528-iq3_kt-ubergarm](https://github.com/user-attachments/assets/86a6e9e0-6544-48a6-a324-27489af9f7d9)
*NOTE*: the [iq2_kt is the earlier implementation from PR484](https://github.com/ikawrakow/ik_llama.cpp/pull/484#issuecomment-2932521414) *without* louiehelm's magic number and a slightly different mix.

<details>

<summary>👈 llama-sweep-bench-data and screenshot</summary>

`nvitop` showing the CPU utilization saturating ~44% (24 / 48 threads). It has a similar pattern of alternating CPU <-> GPU utilization (maybe during TG/PP phases respectively?) that I've seen on other similar quants running like this. Interesting the TG curve is fairly flat though I only had the patience to run out to ~16k context. PP definitely benefits greatly from larger batches.

![ik_llama-cpp-DeepSeep-R1-0528-IQ3_KT-screenshot-smaller](https://github.com/user-attachments/assets/f1deeee3-f138-475c-95a9-7bd12874d40b)

```bash
cmake -B ./build -DGGML_CUDA=ON -DGGML_BLAS=OFF -DGGML_SCHED_MAX_COPIES=1 -DGGML_CUDA_IQK_FORCE_BF16=1 -DGGML_CUDA_F16=ON
cmake --build ./build --config Release -j $(nproc)

model=/mnt/raid/models/ubergarm/DeepSeek-R1-0528-GGUF/DeepSeek-R1-0528-IQ3_KT.gguf
./build/bin/llama-sweep-bench \
    --model "$model" \
    --ctx-size 20480 \
    -ctk f16 \
    -mla 3 -fa \
    -fmoe \
    -amb 512 \
    -ngl 99 \
    -ot "blk\.(3|4|5|6|7|8|9|10)\.ffn_.*=CUDA0" \
    -ot "blk\.(11|12|13|14|15|16|17|18)\.ffn_.*=CUDA1" \
    -ot exps=CPU \
    -ub 2048 -b 2048 \
    --warmup-batch \
    --threads 24
```

## 16 exps offload default batches
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    6.831 |    74.95 |   13.710 |     9.34 |
|   512 |    128 |    512 |    6.711 |    76.30 |   13.129 |     9.75 |
|   512 |    128 |   1024 |    6.741 |    75.96 |   12.887 |     9.93 |
|   512 |    128 |   1536 |    7.085 |    72.27 |   12.860 |     9.95 |
|   512 |    128 |   2048 |    7.694 |    66.54 |   13.063 |     9.80 |
|   512 |    128 |   2560 |    7.037 |    72.76 |   13.036 |     9.82 |
|   512 |    128 |   3072 |    6.970 |    73.46 |   13.064 |     9.80 |
|   512 |    128 |   3584 |    6.969 |    73.47 |   13.229 |     9.68 |
|   512 |    128 |   4096 |    7.094 |    72.17 |   13.086 |     9.78 |
|   512 |    128 |   4608 |    7.291 |    70.22 |   13.104 |     9.77 |
|   512 |    128 |   5120 |    7.220 |    70.92 |   13.104 |     9.77 |
|   512 |    128 |   5632 |    7.343 |    69.73 |   13.250 |     9.66 |
|   512 |    128 |   6144 |    7.392 |    69.26 |   13.332 |     9.60 |
|   512 |    128 |   6656 |    7.524 |    68.04 |   13.352 |     9.59 |
|   512 |    128 |   7168 |    7.558 |    67.74 |   13.297 |     9.63 |
|   512 |    128 |   7680 |    7.655 |    66.88 |   13.322 |     9.61 |
|   512 |    128 |   8192 |    7.838 |    65.32 |   13.649 |     9.38 |
|   512 |    128 |   8704 |    7.876 |    65.01 |   13.644 |     9.38 |
|   512 |    128 |   9216 |    7.971 |    64.23 |   13.474 |     9.50 |
|   512 |    128 |   9728 |    8.085 |    63.33 |   13.476 |     9.50 |
|   512 |    128 |  10240 |    8.154 |    62.79 |   13.504 |     9.48 |
|   512 |    128 |  10752 |    8.756 |    58.47 |   13.686 |     9.35 |
|   512 |    128 |  11264 |    8.333 |    61.44 |   13.716 |     9.33 |
|   512 |    128 |  11776 |    8.451 |    60.59 |   13.703 |     9.34 |
|   512 |    128 |  12288 |    8.552 |    59.87 |   13.707 |     9.34 |
|   512 |    128 |  12800 |    8.653 |    59.17 |   13.981 |     9.16 |
|   512 |    128 |  13312 |    8.745 |    58.55 |   13.844 |     9.25 |
|   512 |    128 |  13824 |    8.784 |    58.29 |   13.890 |     9.22 |
|   512 |    128 |  14336 |    8.906 |    57.49 |   13.918 |     9.20 |
|   512 |    128 |  14848 |    9.000 |    56.89 |   13.900 |     9.21 |
|   512 |    128 |  15360 |    9.067 |    56.47 |   14.015 |     9.13 |
|   512 |    128 |  15872 |    9.760 |    52.46 |   13.957 |     9.17 |
|   512 |    128 |  16384 |    9.405 |    54.44 |   14.125 |     9.06 |

## 16 exps offload 2048 batches
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  2048 |    512 |      0 |   11.513 |   177.89 |   51.919 |     9.86 |
|  2048 |    512 |   2048 |   11.932 |   171.65 |   51.601 |     9.92 |
|  2048 |    512 |   4096 |   12.194 |   167.95 |   52.321 |     9.79 |
|  2048 |    512 |   6144 |   12.645 |   161.96 |   53.345 |     9.60 |
|  2048 |    512 |   8192 |   12.945 |   158.21 |   54.023 |     9.48 |
|  2048 |    512 |  10240 |   13.333 |   153.60 |   54.153 |     9.45 |
|  2048 |    512 |  12288 |   13.804 |   148.37 |   55.268 |     9.26 |
|  2048 |    512 |  14336 |   14.197 |   144.26 |   56.150 |     9.12 |
|  2048 |    512 |  16384 |   14.855 |   137.87 |   56.782 |     9.02 |
|  2048 |    512 |  18432 |   15.578 |   131.47 |   57.078 |     8.97 |

## 14 exps offload 4096 batches
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |   15.067 |   271.85 |  105.623 |     9.69 |
|  4096 |   1024 |   4096 |   16.295 |   251.36 |  107.134 |     9.56 |
|  4096 |   1024 |   8192 |   18.532 |   221.02 |  110.084 |     9.30 |
|  4096 |   1024 |  12288 |   20.982 |   195.22 |  112.232 |     9.12 |
|  4096 |   1024 |  16384 |   23.490 |   174.37 |  115.404 |     8.87 |

## 14 exps offload 8192 batches
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  8192 |   2048 |      0 |   25.344 |   323.23 |  211.439 |     9.69 |
|  8192 |   2048 |   8192 |   34.622 |   236.61 |  221.261 |     9.26 |
|  8192 |   2048 |  16384 |   43.623 |   187.79 |  231.458 |     8.85 |

</details>

![sweep-bench-pr529-iq3_kt](https://github.com/user-attachments/assets/4014c2c7-46d0-4721-8dda-d86084714c68)

---

👤 **ikawrakow** commented on **2025-06-18** at **13:20:49**

Time to merge this.