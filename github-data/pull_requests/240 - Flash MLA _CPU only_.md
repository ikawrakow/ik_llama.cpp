### 🔀 [#240](https://github.com/ikawrakow/ik_llama.cpp/pull/240) - Flash MLA (CPU only)

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-03-03 |
| **Updated** | 2025-03-03 |

---

#### Description

This PR adds Flash Attention for MLA for the CPU back-end. This should be of interest to people running DeepSeeklV3/R1 on the CPU.

Benefits:
* Reduced KV cache size - only the K-cache is required. Hence, the KV cache can be quantized. One can achieve the same with `-mla 2`, but this comes at a significant performance penalty (the transposed view of the cache needs to be computed on each compute graph evaluation)
* Reduced compute buffer size - the `K*Q` tensor, which is the major contributor to compute buffer size for long contexts, never materializes. One can keep the compute buffer size to a desired maximum size using the `-amb` option, but this comes with the inconvenience of having to think about compute buffer sizes, and a small performance penalty for large contexts
* Same or slightly better prompt processing performance compared to just `-mla 1` (but performance for long contexts is still lower than standard attention with FA)
* The same or nearly the same token generation performance

Here is a what we get for KV cache and compute buffer size for DeepSeek-Lite with just MLA for a context of 65k tokens
```
./bin/llama-cli -m $model ... -c 65536 -ctk q8_KV -mla 1
llama_kv_cache_init:        CPU KV buffer size =  2713,50 MiB
llama_new_context_with_model: KV self size  = 2713,50 MiB, c^KV (q8_KV):  985,50 MiB, kv^T (f16): 1728,00 MiB
llama_new_context_with_model:        CPU  output buffer size =     0,39 MiB
llama_new_context_with_model:        CPU compute buffer size =  2228,01 MiB
llama_new_context_with_model: graph nodes  = 1449
llama_new_context_with_model: graph splits = 1
```

And here the same with FA enabled
```
./bin/llama-cli -m $model ... -c 65536 -ctk q8_KV -mla 1 -fa
llama_kv_cache_init:        CPU KV buffer size =   985,50 MiB
llama_new_context_with_model: KV self size  =  985,50 MiB, c^KV (q8_KV):  985,50 MiB, kv^T: not used
llama_new_context_with_model:        CPU  output buffer size =     0,39 MiB
llama_new_context_with_model:        CPU compute buffer size =   240,01 MiB
llama_new_context_with_model: graph nodes  = 1342
llama_new_context_with_model: graph splits = 1
```
For DeepSeekV3/R1 KV cache will be `61/27 = 2.26X` larger. Without FA, the compute buffer would be 8X larger (8X more heads), with FA it would be only marginally larger (due to the larger embedding size).

Just for fun, here is what we need without MLA:
```
./bin/llama-cli -m $model ... -ctk q8_KV -mla 0 -fa
llama_kv_cache_init:        CPU KV buffer size = 12312,00 MiB
llama_new_context_with_model: KV self size  = 12312,00 MiB, K (q8_KV): 5400,00 MiB, V (f16): 6912,00 MiB
llama_new_context_with_model:        CPU  output buffer size =     0,39 MiB
llama_new_context_with_model:        CPU compute buffer size =   214,01 MiB
llama_new_context_with_model: graph nodes  = 1315
llama_new_context_with_model: graph splits = 1
```
And now without MLA and without FA (i.e., what one has available in mainline `llama.cpp`)
```
./bin/llama-cli -m $model ... -ctk q8_KV 
llama_kv_cache_init:        CPU KV buffer size = 12312,00 MiB
llama_new_context_with_model: KV self size  = 12312,00 MiB, K (q8_KV): 5400,00 MiB, V (f16): 6912,00 MiB
llama_new_context_with_model:        CPU  output buffer size =     0,39 MiB
llama_new_context_with_model:        CPU compute buffer size =  2200,01 MiB
llama_new_context_with_model: graph nodes  = 1422
llama_new_context_with_model: graph splits = 1
```
Hahaha - 14.2 GiB. For DeepSeekV3/R1 scale KV cache size by 2.26 and compute buffer size by 8, so 44 GiB.

 
Anyway, here is a performance comparison between FlashMLA and regular MLA for DeepSeek-Lite on a Ryzen-7950X (Zen4) and a Ryzen-5975WX (AVX2)

| model                | platform   | type_k | mla | rtr | fmoe |      test |  t/s (no FA)     |     t/s (FA)     |
| ---------------------| ---------- | -----: | --: | --: | ---: | --------: | ---------------: | ---------------: |
| deepseek2 16B IQ4_NL | Zen4       |  q8_KV |   1 |   1 |    1 |     pp512 |    603.88 ± 2.13 |    616.65 ± 2.81 |
| deepseek2 16B IQ4_NL |            |  q8_KV |   1 |   1 |    1 |    pp1024 |    575.34 ± 2.60 |    579.28 ± 0.65 |
| deepseek2 16B IQ4_NL |            |  q8_KV |   1 |   1 |    1 |    pp2048 |    520.35 ± 3.50 |    518.01 ± 4.12 |
| deepseek2 16B IQ4_NL |            |  q8_KV |   1 |   1 |    1 |    pp4096 |    425.10 ± 0.83 |    433.62 ± 0.38 |
| deepseek2 16B IQ4_NL |            |  q8_KV |   1 |   1 |    1 |    pp8192 |    311.88 ± 0.70 |    309.52 ± 0.37 |
| deepseek2 16B IQ4_NL |            |  q8_KV |   1 |   1 |    1 |   pp16384 |    198.67 ± 2.81 |    181.15 ± 1.47 |
| deepseek2 16B IQ4_NL | AVX2       |  q8_KV |   1 |   1 |    1 |     pp512 |    551.07 ± 3.32 |    571.88 ± 2.92 |
| deepseek2 16B IQ4_NL |            |  q8_KV |   1 |   1 |    1 |    pp1024 |    520.66 ± 3.82 |    551.12 ± 1.85 |
| deepseek2 16B IQ4_NL |            |  q8_KV |   1 |   1 |    1 |    pp2048 |    473.37 ± 3.58 |    504.35 ± 0.92 |
| deepseek2 16B IQ4_NL |            |  q8_KV |   1 |   1 |    1 |    pp4096 |    395.86 ± 3.17 |    421.14 ± 0.58 |
| deepseek2 16B IQ4_NL |            |  q8_KV |   1 |   1 |    1 |    pp8192 |    302.35 ± 1.82 |    315.33 ± 0.49 |
| deepseek2 16B IQ4_NL |            |  q8_KV |   1 |   1 |    1 |   pp16384 |    186.79 ± 0.90 |    193.28 ± 2.92 |

I.e., about the same on `Zen4` and slightly better on vanilla `AVX2`. I think the lower performance at 16k tokens can be improved, but I leave this for another PR.

Here the same but for TG as a function of tokens in the KV cache

| model                | platform   | type_k | mla | rtr | fmoe |          test |    t/s (no FA)   |    t/s (FA)      | 
| ---------------------| ---------- | -----: | --: | --: | ---: | ------------: | ---------------: | ---------------: | 
| deepseek2 16B IQ4_NL | Zen4       |  q8_KV |   1 |   1 |    1 |    tg64@pp128 |     32.21 ± 0.01 |     32.32 ± 0.02 | 
| deepseek2 16B IQ4_NL |            |  q8_KV |   1 |   1 |    1 |    tg64@pp256 |     32.07 ± 0.02 |     32.11 ± 0.06 | 
| deepseek2 16B IQ4_NL |            |  q8_KV |   1 |   1 |    1 |    tg64@pp512 |     31.40 ± 0.03 |     31.82 ± 0.06 | 
| deepseek2 16B IQ4_NL |            |  q8_KV |   1 |   1 |    1 |   tg64@pp1024 |     31.18 ± 0.01 |     31.37 ± 0.00 | 
| deepseek2 16B IQ4_NL |            |  q8_KV |   1 |   1 |    1 |   tg64@pp2048 |     30.05 ± 0.01 |     30.49 ± 0.07 | 
| deepseek2 16B IQ4_NL |            |  q8_KV |   1 |   1 |    1 |   tg64@pp4096 |     28.17 ± 0.06 |     28.83 ± 0.04 | 
| deepseek2 16B IQ4_NL |            |  q8_KV |   1 |   1 |    1 |   tg64@pp8192 |     25.16 ± 0.01 |     26.00 ± 0.13 | 
| deepseek2 16B IQ4_NL | AVX2       |  q8_KV |   1 |   1 |    1 |    tg64@pp128 |     31.21 ± 0.01 |     31.30 ± 0.00 |
| deepseek2 16B IQ4_NL |            |  q8_KV |   1 |   1 |    1 |    tg64@pp256 |     31.26 ± 0.02 |     30.63 ± 0.02 |
| deepseek2 16B IQ4_NL |            |  q8_KV |   1 |   1 |    1 |    tg64@pp512 |     30.79 ± 0.02 |     30.22 ± 0.00 |
| deepseek2 16B IQ4_NL |            |  q8_KV |   1 |   1 |    1 |   tg64@pp1024 |     30.02 ± 0.00 |     29.09 ± 0.00 |
| deepseek2 16B IQ4_NL |            |  q8_KV |   1 |   1 |    1 |   tg64@pp2048 |     28.89 ± 0.00 |     27.38 ± 0.02 |
| deepseek2 16B IQ4_NL |            |  q8_KV |   1 |   1 |    1 |   tg64@pp4096 |     27.01 ± 0.00 |     25.07 ± 0.01 |
| deepseek2 16B IQ4_NL |            |  q8_KV |   1 |   1 |    1 |   tg64@pp8192 |     23.40 ± 0.01 |     21.30 ± 0.00 |

I.e., very slightly better on `Zen4`  and slightly slower on vanilla `AVX2`. 

Supported KV caches are:
* `F16`
* `BF16` (if CPU has native support for `BF16` instructions
* `Q8_0`
* `Q8_KV` - the fastest option
* `Q6_0`

I didn't allow lower quantization than `Q6_0` because a) quality loss becomes significant; b) build time becomes too long as one adds additional quantization types; and c) KV cache is now so much smaller compared to standard attention that it does not make sense to be stingy with KV cache bits.