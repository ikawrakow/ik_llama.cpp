### 🔀 [#541](https://github.com/ikawrakow/ik_llama.cpp/pull/541) - Perhaps slightly faster trellis quants

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-19 |
| **Updated** | 2025-06-21 |

---

#### Description

The PR adds some optimizations to the GEMV implementation of the `IQ2_KT, IQ3_KT, IQ4_KT` quants.

On my Ryzen-7950X I don't notice much of a difference when running with 16 threads as the calculation is (nearly) memory bound. But when testing with fewer threads, I see quite significant gains in TG performance compared to the main branch. Here some results for LlaMA-3.1-8B-Instruct

### IQ2_KT

| model             |       size | threads |          test |      t/s (main)  |     t/s (PR)     |  Speedup |
| ----------------- | ---------: | ------: | ------------: | ---------------: | ---------------: | -------: |
| llama 8B IQ2_KT   |   2.77 GiB |       2 |         tg128 |      3.28 ± 0.00 |      4.11 ± 0.00 |   1.253  |
| llama 8B IQ2_KT   |   2.77 GiB |       4 |         tg128 |      6.28 ± 0.01 |      7.86 ± 0.00 |   1.251  |
| llama 8B IQ2_KT   |   2.77 GiB |       8 |         tg128 |     11.38 ± 0.00 |     14.02 ± 0.01 |   1.232  |

### IQ3_KT

| model            |       size | threads |          test |       t/s (main) |      t/s (PR)    |  Speedup |
| ---------------- | ---------: | ------: | ------------: | ---------------: | ---------------: | -------: |
| llama 8B IQ3_KT  |   3.58 GiB |       2 |         tg128 |      2.87 ± 0.00 |      3.92 ± 0.00 |   1.366  |
| llama 8B IQ3_KT  |   3.58 GiB |       4 |         tg128 |      5.58 ± 0.00 |      7.50 ± 0.00 |   1.344  |
| llama 8B IQ3_KT  |   3.58 GiB |       8 |         tg128 |     10.20 ± 0.00 |     13.42 ± 0.01 |   1.316  |

### IQ4_KT

| model            |       size | threads |          test |     t/s (main)   |   t/s (PR)       |  Speedup |
| ---------------- | ---------: | ------: | ------------: | ---------------: | ---------------: | -------: |
| llama 8B IQ4_KT  |   4.30 GiB |       2 |         tg128 |      2.26 ± 0.00 |      3.27 ± 0.00 |   1.447  |
| llama 8B IQ4_KT  |   4.30 GiB |       4 |         tg128 |      4.38 ± 0.00 |      6.25 ± 0.01 |   1.427  |
| llama 8B IQ4_KT  |   4.30 GiB |       8 |         tg128 |      8.11 ± 0.00 |     11.30 ± 0.00 |   1.393  |

@ubergarm 

In your performance testing on the 6980P system `iqX_kt` quants were very far from saturating memory bandwidth, so perhaps you will see bigger gains there than I see on my system when using all cores.

---

#### 💬 Conversation

👤 **ubergarm** commented the **2025-06-19** at **20:04:52**:<br>

My usual library spot was closed today so sitting outside in the sun trying to grab some quick llama-sweep-bench numbers:

* 7965WX running R1-0528-IQ3_KT
  - PR541@93209939
    *  10.58 TG tok/sec
  - main@144ee1c4
    * 8.61 TG tok/sec

So on the AMD Thread Ripper Pro seeing improvementn from 8.61 up to 10.58 TG tok/sec so 1.229x improvement speedup! Great considering this is also using CUDA offload.

<details>

<summary>llama-sweep-bench command</summary>

```bash
./build/bin/llama-sweep-bench \
    --model "$model" \
    --no-mmap \
    --ctx-size 8704 \
    -ctk f16 \
    -mla 3 -fa \
    -fmoe \
    -amb 512 \
    -ngl 99 \
    -ot "blk\.(3|4|5|6|7|8|9)\.ffn_.*=CUDA0" \
    -ot "blk\.(10|11|12|13|14|15|16)\.ffn_.*=CUDA1" \
    -ot exps=CPU \
    --warmup-batch \
    --threads 24
```
main: n_kv_max = 8704, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = 99, n_threads = 24, n_threads_batch = 24

## version: 3764 (93209939) (PR541)
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    6.297 |    81.30 |   12.097 |    10.58 |
|   512 |    128 |    512 |    6.421 |    79.74 |   12.570 |    10.18 |
|   512 |    128 |   1024 |    6.515 |    78.59 |   12.184 |    10.51 |
|   512 |    128 |   1536 |    7.365 |    69.52 |   12.578 |    10.18 |

## version: 3761 (144ee1c4) (main)
main: n_kv_max = 8704, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = 99, n_threads = 24, n_threads_batch = 24

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    9.495 |    53.92 |   14.858 |     8.61 |
|   512 |    128 |    512 |    9.533 |    53.71 |   14.918 |     8.58 |
|   512 |    128 |   1024 |    9.535 |    53.70 |   15.376 |     8.32 |
|   512 |    128 |   1536 |    9.911 |    51.66 |   14.961 |     8.56 |
|   512 |    128 |   2048 |   10.121 |    50.59 |   14.990 |     8.54 |
|   512 |    128 |   2560 |   10.010 |    51.15 |   15.133 |     8.46 |
|   512 |    128 |   3072 |   10.075 |    50.82 |   15.551 |     8.23 |
|   512 |    128 |   3584 |   10.190 |    50.24 |   15.575 |     8.22 |
|   512 |    128 |   4096 |   10.712 |    47.80 |   15.185 |     8.43 |
|   512 |    128 |   4608 |   10.329 |    49.57 |   15.294 |     8.37 |

</details>

I'll try to get some numbers on the big 6980P pure-CPU soon!

---

👤 **ubergarm** commented the **2025-06-20** at **02:04:52**:<br>

Okay, back at a desk with my laptop for a little while. Here is a quick comparison for a mixed R1-0528-IQ3_KT quant.

* Intel Xeon 6980P
* Single Socket
* CPU-Only compiled
* First line of llama-sweep-bench PP512/TG128/N_KV0
* 272.527 GiB (3.483 BPW)
- type  f32:  361 tensors
- type q5_0:   61 tensors `attn_k_b`
- type q8_0:    1 tensors `token_embd`
- type iq5_ks:  550 tensors `shexp/dense/attn`
- type iq3_kt:  116 tensors `ffn_(gate|up)_exps`
- type iq4_kt:   58 tensors `ffn_down_exps`

| TG tok/sec `main@144ee1c4` | TG tok/sec `PR541@93209939` | speed-up |
| --- | --- | --- |
| 6.29 | 8.22 | 1.309x |

Given not every tensor is `kt` type, actual speed-ups are likely higher. I don't have a good set of pure `kt`'s to test easily like you did above, but my limited testing suggests a big improvement in TG for all three `kt` quant types in both MoE and dense models.

I spot checked using less threads for TG and it was slower, so using `--threads 128 --threads-batch 128` seemed best. There is some little variation in multiple runs as well.

Finally, I didn't expect this, but it seems like PP increased *a lot* as well!!?? At the default batch size PP went from 36.75 up to 117.38,  a ~3.19x speedup!!? I didn't track the code-path to see if the new avx512 and other code is used for PP as well as TG? The effect is not as dramatic at higher batch sizes, but still holds as being faster at ub 4096.

No graphs, tonight, but some data in the fold below showing the effect.

<details>

<summary>👈 llama-sweep-bench command and data </summary>

```bash
model=/mnt/raid/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ3_KT/DeepSeek-R1-0528-IQ3_KT-00001-of-00006.gguf

# adjust -ub 2048 -b 2048
# also adjust -c to large enough for batch size or it will segfault out

numactl -N 0 -m 0 \
./build/bin/llama-sweep-bench \
    --model "$model" \
    -c 1536 \
    -ctk q8_0 \
    -mla 3 -fa \
    -fmoe \
    --no-mmap \
    --threads 128 \
    --threads-batch 128 \
    --numa numactl \
    --warmup-batch
```

## main@144ee1c4

main: n_kv_max = 1536, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = -1, n_threads = 128, n_threads_batch = 128
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |   13.957 |    36.68 |   20.376 |     6.28 |
|   512 |    128 |    512 |   14.501 |    35.31 |   20.703 |     6.18 |
|   512 |    128 |   1024 |   14.865 |    34.44 |   22.955 |     5.58 |

main: n_kv_max = 6144, n_batch = 2048, n_ubatch = 2048, flash_attn = 1, n_gpu_layers = -1, n_threads = 128, n_threads_batch = 128
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  2048 |    512 |      0 |   24.211 |    84.59 |   83.107 |     6.16 |
|  2048 |    512 |   2048 |   27.291 |    75.04 |   94.896 |     5.40 |
|  2048 |    512 |   4096 |   30.650 |    66.82 |   95.730 |     5.35 |

main: n_kv_max = 12288, n_batch = 4096, n_ubatch = 4096, flash_attn = 1, n_gpu_layers = -1, n_threads = 128, n_threads_batch = 128

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |   37.711 |   108.62 |  174.245 |     5.88 |
|  4096 |   1024 |   4096 |   49.629 |    82.53 |  196.952 |     5.20 |
|  4096 |   1024 |   8192 |   59.777 |    68.52 |  199.099 |     5.14 |

---

## PR541@93209939 

main: n_kv_max = 1536, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = -1, n_threads = 128, n_threads_batch = 128
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    4.362 |   117.38 |   15.564 |     8.22 |
|   512 |    128 |    512 |    4.729 |   108.26 |   17.158 |     7.46 |
|   512 |    128 |   1024 |    4.942 |   103.60 |   19.407 |     6.60 |

main: n_kv_max = 6144, n_batch = 2048, n_ubatch = 2048, flash_attn = 1, n_gpu_layers = -1, n_threads = 128, n_threads_batch = 128
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  2048 |    512 |      0 |   14.727 |   139.07 |   65.669 |     7.80 |
|  2048 |    512 |   2048 |   18.297 |   111.93 |   81.433 |     6.29 |
|  2048 |    512 |   4096 |   21.476 |    95.36 |   82.792 |     6.18 |

main: n_kv_max = 12288, n_batch = 4096, n_ubatch = 4096, flash_attn = 1, n_gpu_layers = -1, n_threads = 128, n_threads_batch = 128

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |   31.189 |   131.33 |  137.583 |     7.44 |
|  4096 |   1024 |   4096 |   42.929 |    95.41 |  162.857 |     6.29 |
|  4096 |   1024 |   8192 |   53.700 |    76.28 |  149.823 |     6.83 |

</details>

fwiw here is the output of `lscpu | grep Flags` on the 6980P

<details>

<summary>👈 6980P CPU flags</summary>

```
$ lscpu | grep Flags
Flags:                                fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf tsc_known_freq pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb cat_l3 cat_l2 cdp_l3 intel_ppin cdp_l2 ssbd mba ibrs ibpb stibp ibrs_enhanced tpr_shadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb intel_pt avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local split_lock_detect user_shstk avx_vnni avx512_bf16 wbnoinvd dtherm ida arat pln pts hwp hwp_act_window hwp_epp hwp_pkg_req vnmi avx512vbmi umip pku ospke waitpkg avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid bus_lock_detect cldemote movdiri movdir64b enqcmd fsrm md_clear serialize tsxldtrk pconfig arch_lbr ibt amx_bf16 avx512_fp16 amx_tile amx_int8 flush_l1d arch_capabilities
```

</details>

Thanks!

---

👤 **Nexesenex** commented the **2025-06-20** at **02:48:36**:<br>

Confirmed for me for IQ3_KT.

Llama 8b.
llama_model_loader: - type  f32:   66 tensors
llama_model_loader: - type q6_0:    1 tensors (embeds)
llama_model_loader: - type iq3_kt:  160 tensors
llama_model_loader: - type iq4_ks_r4:   32 tensors (attn_k)
llama_model_loader: - type iq5_ks_r4:   33 tensors (attn_v, output)

Before patch : TG 3.27 t/s.
After patch : TG 4.79 t/s.

Rig : Ryzen 5700G, AVX2, 4*8GB DDR4 2666mhz.

---

👤 **ikawrakow** commented the **2025-06-20** at **04:44:01**:<br>

Thank you for testing!

> Finally, I didn't expect this, but it seems like PP increased a lot as well!!?? At the default batch size PP went from 36.75 up to 117.38, 

This is not supposed to happen. It is a mixture of experts, so the new path can get invoked when an expert ends up processing fewer than (currently) 32 tokens. But at least on my end this works fine, even if I disable the repacking to `Q8_0_R8` altogether. So, I guess, something else must be broken.

---

👤 **ubergarm** commented the **2025-06-20** at **21:43:10**:<br>

Okay, here are the perplexities as run on the thread ripper pro. I ran all the Qwen3-14B quants on a single RTX A6000 to use the CUDA implementation, and then the three `KT` quant again compiled CPU-only to confirm things line up as expected. All tests run on PR541@5b677c3c

| Quant | Size   | BPW    | Perplexity | Perplexity | Change |
| ---   | ---    | ---    | ---        | ---        | ---    |
|       | GiB    |        | CUDA       | CPU        | Percent|
| BF16  | 27.509 | 16.000 |  9.0133    |            |        |
| Q8_0  | 14.615 |  8.501 |  9.0281    |            |        |
| Q4_0  |  7.925 |  4.609 |  9.1455    |            |        |
| IQ4_KT|  7.164 |  4.167 |  9.0973    |  9.1005    | +0.035% |
| IQ3_KT|  5.818 |  3.384 |  9.5184    |  9.5244    | +0.063% |
| IQ2_KT|  4.280 |  2.489 | 11.2557    | 11.2631    | +0.066% |

So looks like the CPU implementation is within the margin of error though shows a *very slight* increase in perplexity over the CUDA implementation.

<details>

<summary>👈 Perplexity command and data including error values</summary>

```bash
# For CPU remove `-ngl` and increase threads
CUDA_VISIBLE_DEVICES="0" \
    ./build/bin/llama-perplexity \
    --model "$model" \
    -fa \
    -f wiki.test.raw \
    --seed 1337 \
    -ngl 99 \
    --threads 1
```

## CUDA
* BF16
  - Final estimate: PPL = 9.0133 +/- 0.07115
* Q8_0
  - Final estimate: PPL = 9.0281 +/- 0.07136
* Q4_0
  - Final estimate: PPL = 9.1455 +/- 0.07215
* IQ4_KT
  - Final estimate: PPL = 9.0973 +/- 0.07157
* IQ3_KT
  - Final estimate: PPL = 9.5184 +/- 0.07579
* IQ2_KT
  - Final estimate: PPL = 11.2557 +/- 0.08946

## CPU
* IQ4_KT
  - Final estimate: PPL = 9.1005 +/- 0.07161
* IQ3_KT
  - Final estimate: PPL = 9.5244 +/- 0.07586
* IQ2_KT
  - Final estimate: PPL = 11.2631 +/- 0.08954

</details>


## Conclusion

Overall PR looks like a great speed improvement for token generation of KT quants. Given they still seem CPU bottle-necked at least in this specific test, I'd likely choose the 4bpw version over the smaller sizes when targeting tensors destined for CPU/RAM; because it generates about as fast while keeping more quality.

Makes me wonder when a 5bpw or 6bpw version would begin to be RAM bandwidth bottle-necked again, but probably heavily dependent on the specific model and hardware. An iq6_kt might be equally RAM / CPU bottlenecked and achieve ~25 tok/sec TG on the ~512GB/s 6980P. 512 / (27.509 * (6/8))

Anyway, very cool stuff! Thanks!

---

👤 **ubergarm** commented the **2025-06-20** at **23:32:59**:<br>

I was too curious to see how it it performed on the AMD Thread Ripper Pro.. Interestingly, there was more variability in the generation speed than with the Xeon 6980P. So I take back my conclusion above about always reaching for the 4bpw... lol...

Here is the graph and numbers below. Cheers!

![sweep-bench-pr541-thread-ripper-qwen3-14b](https://github.com/user-attachments/assets/af56c28a-8dd3-43a0-b4f1-2847d71433d2)

<details>

<summary>👈 sweep-bench command and data</summary>

```bash
./build/bin/llama-sweep-bench \
        --model "$model" \
        --ctx-size 8704 \
        -ctk q8_0 -ctv q8_0 \
        -fa \
        --no-mmap \
        --warmup-batch \
        --threads 24 \
```

## IQ4_KT PR541@5b677c3c
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    2.330 |   219.70 |    7.633 |    16.77 |
|   512 |    128 |    512 |    2.384 |   214.75 |    7.785 |    16.44 |
|   512 |    128 |   1024 |    3.593 |   142.50 |    7.870 |    16.26 |
|   512 |    128 |   1536 |    2.495 |   205.20 |    8.024 |    15.95 |
|   512 |    128 |   2048 |    2.548 |   200.94 |    7.986 |    16.03 |
|   512 |    128 |   2560 |    2.611 |   196.11 |    8.056 |    15.89 |
|   512 |    128 |   3072 |    2.744 |   186.60 |    8.193 |    15.62 |
|   512 |    128 |   3584 |    2.712 |   188.77 |    8.251 |    15.51 |
|   512 |    128 |   4096 |    2.781 |   184.13 |    8.257 |    15.50 |
|   512 |    128 |   4608 |    2.818 |   181.69 |    8.392 |    15.25 |
|   512 |    128 |   5120 |    2.877 |   177.94 |    8.562 |    14.95 |
|   512 |    128 |   5632 |    2.928 |   174.88 |    8.382 |    15.27 |
|   512 |    128 |   6144 |    2.987 |   171.42 |    8.711 |    14.69 |
|   512 |    128 |   6656 |    3.039 |   168.45 |    8.864 |    14.44 |
|   512 |    128 |   7168 |    3.097 |   165.32 |    8.737 |    14.65 |
|   512 |    128 |   7680 |    3.147 |   162.72 |    8.738 |    14.65 |
|   512 |    128 |   8192 |    3.208 |   159.60 |    8.992 |    14.24 |

## IQ3_KT PR541@5b677c3c
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    2.279 |   224.68 |    6.590 |    19.42 |
|   512 |    128 |    512 |    2.334 |   219.40 |    6.725 |    19.03 |
|   512 |    128 |   1024 |    2.390 |   214.23 |    6.813 |    18.79 |
|   512 |    128 |   1536 |    2.446 |   209.36 |    6.914 |    18.51 |
|   512 |    128 |   2048 |    2.502 |   204.60 |    6.953 |    18.41 |
|   512 |    128 |   2560 |    2.558 |   200.13 |    7.028 |    18.21 |
|   512 |    128 |   3072 |    2.612 |   196.05 |    7.201 |    17.77 |
|   512 |    128 |   3584 |    2.671 |   191.70 |    7.217 |    17.74 |
|   512 |    128 |   4096 |    2.720 |   188.24 |    7.230 |    17.70 |
|   512 |    128 |   4608 |    2.776 |   184.44 |    7.364 |    17.38 |
|   512 |    128 |   5120 |    2.836 |   180.54 |    7.475 |    17.12 |
|   512 |    128 |   5632 |    2.885 |   177.47 |    7.342 |    17.43 |
|   512 |    128 |   6144 |    2.950 |   173.58 |    7.842 |    16.32 |
|   512 |    128 |   6656 |    2.995 |   170.98 |    7.761 |    16.49 |
|   512 |    128 |   7168 |    3.054 |   167.64 |    7.590 |    16.86 |
|   512 |    128 |   7680 |    3.101 |   165.10 |    7.605 |    16.83 |
|   512 |    128 |   8192 |    3.164 |   161.83 |    8.007 |    15.99 |

## IQ2_KT PR541@5b677c3c
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    2.260 |   226.51 |    6.357 |    20.13 |
|   512 |    128 |    512 |    2.316 |   221.06 |    6.504 |    19.68 |
|   512 |    128 |   1024 |    2.371 |   215.90 |    6.596 |    19.41 |
|   512 |    128 |   1536 |    2.428 |   210.90 |    6.672 |    19.19 |
|   512 |    128 |   2048 |    2.482 |   206.29 |    6.713 |    19.07 |
|   512 |    128 |   2560 |    2.539 |   201.65 |    6.783 |    18.87 |
|   512 |    128 |   3072 |    2.593 |   197.47 |    6.934 |    18.46 |
|   512 |    128 |   3584 |    2.650 |   193.18 |    6.958 |    18.40 |
|   512 |    128 |   4096 |    2.708 |   189.09 |    6.974 |    18.36 |
|   512 |    128 |   4608 |    2.761 |   185.41 |    7.116 |    17.99 |
|   512 |    128 |   5120 |    2.820 |   181.58 |    7.274 |    17.60 |
|   512 |    128 |   5632 |    2.865 |   178.71 |    7.085 |    18.07 |
|   512 |    128 |   6144 |    2.930 |   174.72 |    7.480 |    17.11 |
|   512 |    128 |   6656 |    2.985 |   171.50 |    7.469 |    17.14 |
|   512 |    128 |   7168 |    3.042 |   168.32 |    7.465 |    17.15 |
|   512 |    128 |   7680 |    3.085 |   165.95 |    7.465 |    17.15 |
|   512 |    128 |   8192 |    3.146 |   162.72 |    7.742 |    16.53 |

## IQ4_KT main@1843ed22
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    2.325 |   220.18 |   10.114 |    12.66 |
|   512 |    128 |    512 |    2.381 |   215.03 |   10.263 |    12.47 |
|   512 |    128 |   1024 |    2.435 |   210.30 |   10.355 |    12.36 |
|   512 |    128 |   1536 |    2.490 |   205.62 |   10.435 |    12.27 |
|   512 |    128 |   2048 |    2.544 |   201.25 |   10.488 |    12.20 |
|   512 |    128 |   2560 |    2.599 |   197.01 |   10.556 |    12.13 |
|   512 |    128 |   3072 |    2.657 |   192.68 |   10.665 |    12.00 |
|   512 |    128 |   3584 |    2.711 |   188.89 |   10.735 |    11.92 |
|   512 |    128 |   4096 |    2.766 |   185.12 |   10.757 |    11.90 |
|   512 |    128 |   4608 |    2.820 |   181.55 |   10.887 |    11.76 |
|   512 |    128 |   5120 |    2.877 |   177.94 |   10.981 |    11.66 |
|   512 |    128 |   5632 |    2.933 |   174.59 |   10.864 |    11.78 |
|   512 |    128 |   6144 |    2.993 |   171.09 |   11.155 |    11.47 |
|   512 |    128 |   6656 |    3.045 |   168.16 |   11.238 |    11.39 |
|   512 |    128 |   7168 |    3.105 |   164.90 |   11.260 |    11.37 |
|   512 |    128 |   7680 |    3.146 |   162.75 |   11.261 |    11.37 |
|   512 |    128 |   8192 |    3.219 |   159.07 |   11.497 |    11.13 |

## IQ3_KT main@1843ed22
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    2.278 |   224.73 |    8.885 |    14.41 |
|   512 |    128 |    512 |    2.334 |   219.39 |    9.031 |    14.17 |
|   512 |    128 |   1024 |    2.388 |   214.37 |    9.129 |    14.02 |
|   512 |    128 |   1536 |    2.444 |   209.48 |    9.207 |    13.90 |
|   512 |    128 |   2048 |    2.855 |   179.35 |    9.299 |    13.76 |
|   512 |    128 |   2560 |    2.558 |   200.13 |    9.364 |    13.67 |
|   512 |    128 |   3072 |    2.616 |   195.72 |    9.440 |    13.56 |
|   512 |    128 |   3584 |    2.666 |   192.04 |    9.513 |    13.45 |
|   512 |    128 |   4096 |    2.719 |   188.31 |    9.510 |    13.46 |
|   512 |    128 |   4608 |    2.774 |   184.55 |    9.681 |    13.22 |
|   512 |    128 |   5120 |    2.832 |   180.80 |    9.763 |    13.11 |
|   512 |    128 |   5632 |    2.885 |   177.45 |    9.656 |    13.26 |
|   512 |    128 |   6144 |    2.942 |   174.05 |    9.899 |    12.93 |
|   512 |    128 |   6656 |    3.001 |   170.59 |   10.007 |    12.79 |
|   512 |    128 |   7168 |    3.055 |   167.61 |   10.057 |    12.73 |
|   512 |    128 |   7680 |    3.108 |   164.72 |   10.078 |    12.70 |
|   512 |    128 |   8192 |    3.169 |   161.54 |   10.248 |    12.49 |

## IQ2_KT main@1843ed22
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    2.257 |   226.84 |    8.466 |    15.12 |
|   512 |    128 |    512 |    2.312 |   221.45 |    8.614 |    14.86 |
|   512 |    128 |   1024 |    2.374 |   215.67 |    8.673 |    14.76 |
|   512 |    128 |   1536 |    2.425 |   211.09 |    8.781 |    14.58 |
|   512 |    128 |   2048 |    2.482 |   206.32 |    8.821 |    14.51 |
|   512 |    128 |   2560 |    2.536 |   201.88 |    8.899 |    14.38 |
|   512 |    128 |   3072 |    2.591 |   197.60 |    9.070 |    14.11 |
|   512 |    128 |   3584 |    2.648 |   193.35 |    9.091 |    14.08 |
|   512 |    128 |   4096 |    2.702 |   189.48 |    9.087 |    14.09 |
|   512 |    128 |   4608 |    2.751 |   186.15 |    9.211 |    13.90 |
|   512 |    128 |   5120 |    2.813 |   182.01 |    9.401 |    13.62 |
|   512 |    128 |   5632 |    2.860 |   179.05 |    9.028 |    14.18 |
|   512 |    128 |   6144 |    2.922 |   175.21 |    9.615 |    13.31 |
|   512 |    128 |   6656 |    2.968 |   172.48 |    9.611 |    13.32 |
|   512 |    128 |   7168 |    3.030 |   168.95 |    9.330 |    13.72 |
|   512 |    128 |   7680 |    3.079 |   166.27 |    9.333 |    13.71 |
|   512 |    128 |   8192 |    3.143 |   162.89 |    9.883 |    12.95 |

</details>

---

👤 **ubergarm** commented the **2025-06-21** at **00:37:44**:<br>

I'm happy enough with the performance now to release the `R1-0528-IQ3_KT` on hugging face as *experimental* with the warning that there could potentially still be breaking changes. But a few others folks would be able to test as well. It lines up nicely in terms of perplexity and size, has a tight KLD max delta P, and now generates comparable to the slightly larger `IQ3_K_R4` as [shown in this discussion benchmark on huggingface](https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF/discussions/7)

over and out!

---

👤 **ikawrakow** commented the **2025-06-21** at **14:32:10**:<br>

@ubergarm Thank you for the extensive testing!

Based on the tests, this looks like a winner, so merging.