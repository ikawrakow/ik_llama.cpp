### 🗣️ [#491](https://github.com/ikawrakow/ik_llama.cpp/discussions/491) - -rtr actually hurts prompt t/s for large ubatch?

| **Author** | `Ph0rk0z` |
| :--- | :--- |
| **Created** | 2025-06-03 |
| **Updated** | 2025-06-11 |

---

#### Description

I had long assumed that -RTR was a universal speedup and just like repacking, it would help your performance always. Seems that is not the case.

<details>
<summary> Qwen 235b command line </summary>

```
    CUDA_VISIBLE_DEVICES=0,1,2,3 numactl --interleave=all ./bin/llama-sweep-bench \
    -m Smoothie-Qwen3-235B-A22B.IQ4_XS.gguf \
    -t 48 \
    -c 32768 \
    --numa distribute \
    -ngl 95 \
    -ctk q8_0 \
    -ctv q8_0 \
    -fa \
    -fmoe \
    -amb 64 \
    -b 4096 \
    -ub 4096 \
    -ot "blk\.(0|1|2|3|4|5|6|7|8|9|10|11|12|13|)\.ffn_.*_exps.=CUDA0" \
    -ot "blk\.(14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29)\.ffn_.*_exps.=CUDA1" \
    -ot "blk\.(30|31|32|33|34|35|36|37|38|39|40|41|42|43|44|45|)\.ffn_.*_exps.=CUDA2" \
    -ot "blk\.(46|47|48|49|50|51|52|53|54|55|56|57|58|59)\.ffn_.*_exps.=CUDA3" \
    -ot "\.ffn_.*_exps.=CPU"
```

</details>
<details><summary>No RTR Buffers</summary>

```
llama_kv_cache_init:      CUDA0 KV buffer size =   816.01 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =   816.01 MiB
llama_kv_cache_init:      CUDA2 KV buffer size =   816.01 MiB
llama_kv_cache_init:      CUDA3 KV buffer size =   748.01 MiB
llama_new_context_with_model: KV self size  = 3196.00 MiB, K (q8_0): 1598.00 MiB, V (q8_0): 1598.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.58 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      CUDA0 compute buffer size =  1856.02 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =  1094.02 MiB
llama_new_context_with_model:      CUDA2 compute buffer size =   836.00 MiB
llama_new_context_with_model:      CUDA3 compute buffer size =  2502.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   576.05 MiB
llama_new_context_with_model: graph nodes  = 3672
llama_new_context_with_model: graph splits = 183
main: n_kv_max = 32768, n_batch = 4096, n_ubatch = 4096, flash_attn = 1, n_gpu_layers = 95, n_threads = 48, n_threads_batch = 48
```
</details>

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |   14.283 |   286.78 |   65.942 |    15.53 |
|  4096 |   1024 |   4096 |   14.803 |   276.70 |   68.941 |    14.85 |
|  4096 |   1024 |   8192 |   15.461 |   264.92 |   73.586 |    13.92 |
|  4096 |   1024 |  12288 |   15.831 |   258.74 |   77.875 |    13.15 |
|  4096 |   1024 |  16384 |   16.185 |   253.08 |   81.513 |    12.56 |
|  4096 |   1024 |  20480 |   16.926 |   241.99 |   85.266 |    12.01 |

<details><summary>Buffers with RTR</summary>

```
llama_kv_cache_init:      CUDA0 KV buffer size =   816.01 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =   816.01 MiB
llama_kv_cache_init:      CUDA2 KV buffer size =   816.01 MiB
llama_kv_cache_init:      CUDA3 KV buffer size =   748.01 MiB
llama_new_context_with_model: KV self size  = 3196.00 MiB, K (q8_0): 1598.00 MiB, V (q8_0): 1598.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.58 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      CUDA0 compute buffer size =  1664.02 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =  1094.02 MiB
llama_new_context_with_model:      CUDA2 compute buffer size =  1024.02 MiB
llama_new_context_with_model:      CUDA3 compute buffer size =  2502.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =  1024.05 MiB
llama_new_context_with_model: graph nodes  = 3672
llama_new_context_with_model: graph splits = 149

main: n_kv_max = 32768, n_batch = 4096, n_ubatch = 4096, flash_attn = 1, n_gpu_layers = 95, n_threads = 48, n_threads_batch = 48
```
</details>

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |   24.221 |   169.11 |   59.405 |    17.24 |
|  4096 |   1024 |   4096 |   24.852 |   164.82 |   62.359 |    16.42 |
|  4096 |   1024 |   8192 |   25.570 |   160.19 |   67.178 |    15.24 |
|  4096 |   1024 |  12288 |   26.293 |   155.78 |   71.996 |    14.22 |
|  4096 |   1024 |  16384 |   26.979 |   151.82 |   76.468 |    13.39 | 


It's even worse on deepseek where my prompt speeds were cut in half while losing about 1.5t/s of TG only. Another thing of note is that no repacking causes much more large transfers to the GPU. I saw rates of up to 16GBs going between cards and I assume the system?

Peculiar thing though, for smaller batches:

<details> <summary> 235b ub 1024 </summary>


```
CUDA_VISIBLE_DEVICES=0,1,2,3 numactl --interleave=all ./bin/llama-sweep-bench \
    -m Smoothie-Qwen3-235B-A22B.IQ4_XS.gguf \
    -t 48 \
    -c 32768 \
    --numa distribute \
    -ngl 95 \
    -ctk q8_0 \
    -ctv q8_0 \
    -fa \
    -rtr \
    -fmoe \
    -amb 512 \
    -ub 1024 \
    -ot "blk\.(0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15)\.ffn_.*_exps.=CUDA0" \
    -ot "blk\.(16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32)\.ffn_.*_exps.=CUDA1" \
    -ot "blk\.(33|34|35|36|37|38|39|40|41|42|43|44|45|46|47|48|49)\.ffn_.*_exps.=CUDA2" \
    -ot "blk\.(50|51|52|53|54|55|56|57|58|59|60|61|62|63|64|65|66)\.ffn_.*_exps.=CUDA3" \
    -ot "\.ffn_.*_exps.=CPU"
```


</details>

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  1024 |    256 |      0 |    5.432 |   188.50 |   13.878 |    18.45 |
|  1024 |    256 |   1024 |    5.402 |   189.55 |   14.069 |    18.20 |
|  1024 |    256 |   2048 |    5.434 |   188.43 |   14.268 |    17.94 |
|  1024 |    256 |   3072 |    5.514 |   185.71 |   14.499 |    17.66 |
|  1024 |    256 |   4096 |    5.543 |   184.74 |   14.655 |    17.47 |
|  1024 |    256 |   5120 |    5.566 |   183.96 |   15.034 |    17.03 |
|  1024 |    256 |   6144 |    5.624 |   182.08 |   15.241 |    16.80 |
|  1024 |    256 |   7168 |    5.700 |   179.64 |   15.547 |    16.47 |
|  1024 |    256 |   8192 |    5.732 |   178.66 |   15.836 |    16.17 |
|  1024 |    256 |   9216 |    5.820 |   175.96 |   16.136 |    15.87 |
|  1024 |    256 |  10240 |    5.812 |   176.18 |   16.415 |    15.60 |
|  1024 |    256 |  11264 |    5.888 |   173.92 |   16.751 |    15.28 |
|  1024 |    256 |  12288 |    5.907 |   173.37 |   16.951 |    15.10 |
|  1024 |    256 |  13312 |    5.994 |   170.84 |   17.151 |    14.93 |
|  1024 |    256 |  14336 |    5.998 |   170.72 |   17.394 |    14.72 |
|  1024 |    256 |  15360 |    6.043 |   169.46 |   17.623 |    14.53 |
|  1024 |    256 |  16384 |    6.139 |   166.80 |   17.983 |    14.24 |


Without -rtr, this makes ~120 prompt at most. Anyone know the why or noticed something similar?

---

#### 🗣️ Discussion

👤 **Ph0rk0z** replied the **2025-06-04** at **15:59:57**:<br>

I played around with offline repacking next. Oh boy.

Offline repacking on 4096 batch. 

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |   24.349 |   168.22 |   69.065 |    14.83 |
|  4096 |   1024 |   4096 |   24.815 |   165.06 |   71.880 |    14.25 |
|  4096 |   1024 |   8192 |   25.604 |   159.97 |   76.457 |    13.39 |
|  4096 |   1024 |  12288 |   26.288 |   155.81 |   80.361 |    12.74 |

It seems like performance here is identical to using -rtr. Debuff to text generation likely from mmap.


Ok.. so let's try it in a configuration where repacking previously helped like the last one in the previous post. Only 6 layers are incorrectly packed and everything has gone into the toilet.

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  1024 |    256 |      0 |    6.992 |   146.46 |  192.370 |     1.33 |
|  1024 |    256 |   1024 |    6.969 |   146.95 |  192.509 |     1.33 |

Then I indiscriminately repacked the whole model to see what would happen. It got just as bad. Lots of transfers.Could be related to offload policy? I didn't even bother waiting for the first iteration it took so long. CPU running at 10 cores from the 1000% usage.


And finally I packed the model correctly AND used the configuration that produced a speed gain.

with mmap

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  1024 |    256 |      0 |    6.306 |   162.40 |   15.561 |    16.45 |
|  1024 |    256 |   1024 |    5.993 |   170.87 |   15.743 |    16.26 |
|  1024 |    256 |   2048 |    6.004 |   170.54 |   15.897 |    16.10 |
|  1024 |    256 |   3072 |    5.882 |   174.10 |   16.071 |    15.93 |
|  1024 |    256 |   4096 |    6.295 |   162.67 |   16.253 |    15.75 |
|  1024 |    256 |   5120 |    6.144 |   166.67 |   16.608 |    15.41 |
|  1024 |    256 |   6144 |    6.143 |   166.70 |   16.833 |    15.21 |
|  1024 |    256 |   7168 |    6.280 |   163.07 |   17.086 |    14.98 |
|  1024 |    256 |   8192 |    6.298 |   162.58 |   17.373 |    14.74 |

no mmap

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  1024 |    256 |      0 |    5.759 |   177.82 |   14.442 |    17.73 |
|  1024 |    256 |   1024 |    5.639 |   181.59 |   14.523 |    17.63 |
|  1024 |    256 |   2048 |    5.867 |   174.53 |   14.656 |    17.47 |
|  1024 |    256 |   3072 |    5.900 |   173.56 |   14.833 |    17.26 |
|  1024 |    256 |   4096 |    6.026 |   169.92 |   15.031 |    17.03 |
|  1024 |    256 |   5120 |    6.069 |   168.73 |   15.389 |    16.63 |
|  1024 |    256 |   6144 |    5.849 |   175.07 |   15.564 |    16.45 |
|  1024 |    256 |   7168 |    5.943 |   172.31 |   15.939 |    16.06 |
|  1024 |    256 |   8192 |    6.154 |   166.39 |   16.184 |    15.82 |

Does it help to cache the model first? Let's run with mmap again....


|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  1024 |    256 |      0 |    6.441 |   158.99 |   15.466 |    16.55 |
|  1024 |    256 |   1024 |    6.111 |   167.56 |   15.717 |    16.29 |
|  1024 |    256 |   2048 |    5.875 |   174.30 |   15.810 |    16.19 |
|  1024 |    256 |   3072 |    6.029 |   169.84 |   16.001 |    16.00 |
|  1024 |    256 |   4096 |    6.150 |   166.52 |   16.170 |    15.83 |
|  1024 |    256 |   5120 |    6.010 |   170.39 |   16.537 |    15.48 |
|  1024 |    256 |   6144 |    6.008 |   170.44 |   16.727 |    15.30 |
|  1024 |    256 |   7168 |    6.332 |   161.73 |   17.038 |    15.02 |
|  1024 |    256 |   8192 |    6.277 |   163.13 |   17.328 |    14.77 |

NOPE!

**So the point to the whole story, if anyone cares, is that even a few mis-packed layers will tank your speeds. Feels like there is no point to posting R4/R8 quants because the user will have to repack them anyway unless using the EXACT configuration of the author. What am I missing here?**


As a bonus.. let's find where RTR starts to help prompt processing...

First I'll take a new baseline because it seems textgen is not working so good after packing/loading/etc. Could be I need to drop caches?

4096 no rtr/no-mmap Baseline

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |   14.588 |   280.78 |   71.871 |    14.25 |
|  4096 |   1024 |   4096 |   14.877 |   275.33 |   74.257 |    13.79 |
|  4096 |   1024 |   8192 |   15.500 |   264.25 |   78.862 |    12.98 |
|  4096 |   1024 |  12288 |   15.919 |   257.30 |   83.039 |    12.33 |
|  4096 |   1024 |  16384 |   16.476 |   248.60 |   87.030 |    11.77 |

That's the highest we will get for now.


2048 without RTR with no-mmap

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  2048 |    512 |      0 |   11.606 |   176.47 |   35.719 |    14.33 |
|  2048 |    512 |   2048 |   11.586 |   176.77 |   36.388 |    14.07 |
|  2048 |    512 |   4096 |   11.683 |   175.30 |   37.146 |    13.78 |
|  2048 |    512 |   6144 |   11.813 |   173.37 |   38.241 |    13.39 |
|  2048 |    512 |   8192 |   11.950 |   171.38 |   39.246 |    13.05 |
|  2048 |    512 |  10240 |   12.194 |   167.95 |   40.579 |    12.62 |
|  2048 |    512 |  12288 |   12.208 |   167.75 |   41.348 |    12.38 |
|  2048 |    512 |  14336 |   12.412 |   165.00 |   42.410 |    12.07 |
|  2048 |    512 |  16384 |   12.407 |   165.07 |   43.277 |    11.83 |

2048 with rtr

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  2048 |    512 |      0 |   13.308 |   153.89 |   32.755 |    15.63 |
|  2048 |    512 |   2048 |   13.167 |   155.54 |   33.466 |    15.30 |
|  2048 |    512 |   4096 |   13.308 |   153.89 |   34.117 |    15.01 |
|  2048 |    512 |   6144 |   13.351 |   153.40 |   35.396 |    14.47 |
|  2048 |    512 |   8192 |   13.539 |   151.27 |   36.420 |    14.06 |
|  2048 |    512 |  10240 |   14.000 |   146.28 |   37.873 |    13.52 |
|  2048 |    512 |  12288 |   14.011 |   146.17 |   38.719 |    13.22 |
|  2048 |    512 |  14336 |   14.113 |   145.11 |   39.612 |    12.93 |
|  2048 |    512 |  16384 |   14.596 |   140.32 |   40.743 |    12.57 |

So still a debuff to prompt processing and a mild gain to t/g

Let's try something else....

2048/1024 -rtr

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  1024 |    256 |      0 |    6.837 |   149.78 |   16.543 |    15.47 |
|  1024 |    256 |   1024 |    6.830 |   149.93 |   16.713 |    15.32 |
|  1024 |    256 |   2048 |    6.885 |   148.73 |   16.821 |    15.22 |
|  1024 |    256 |   3072 |    7.085 |   144.54 |   17.057 |    15.01 |
|  1024 |    256 |   4096 |    6.899 |   148.42 |   17.248 |    14.84 |
|  1024 |    256 |   5120 |    7.106 |   144.10 |   17.608 |    14.54 |
|  1024 |    256 |   6144 |    6.760 |   151.47 |   17.794 |    14.39 |
|  1024 |    256 |   7168 |    7.181 |   142.60 |   18.080 |    14.16 |
|  1024 |    256 |   8192 |    7.154 |   143.13 |   18.325 |    13.97 |

2048/1024 -no rtr and no-mmap

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  1024 |    256 |      0 |    9.905 |   103.38 |   17.792 |    14.39 |
|  1024 |    256 |   1024 |    9.711 |   105.45 |   17.938 |    14.27 |
|  1024 |    256 |   2048 |    9.793 |   104.56 |   18.090 |    14.15 |
|  1024 |    256 |   3072 |    9.786 |   104.64 |   18.292 |    14.00 |
|  1024 |    256 |   4096 |    9.824 |   104.24 |   18.465 |    13.86 |
|  1024 |    256 |   5120 |    9.854 |   103.92 |   18.844 |    13.59 |
|  1024 |    256 |   6144 |    9.874 |   103.71 |   19.033 |    13.45 |
|  1024 |    256 |   7168 |    9.930 |   103.12 |   19.309 |    13.26 |
|  1024 |    256 |   8192 |   10.060 |   101.79 |   19.568 |    13.08 |

Ok.. now prompt processing finally fell.. the original observed effect.


So then -rtr or repacking is only useful in the case of ub being half the batch size? It does allow you to generate text a little bit faster in every test at least.

---

👤 **ikawrakow** replied the **2025-06-04** at **16:48:34**:<br>

Perhaps to understand how repacked quants behave on the CPU and CUDA, it is easier to take a smaller model that would completely fit one GPU, quantize with with `--pure` to your favorite quant and corresponding repacked variant, and then
* Run fully offloaded to the GPU
* Run CPU-only

It is an easy exercise, does not require an imatrix as you are not after the best possible quantization quality, and if you pick a model that is not too large, it is very quick to do.

Without having understood what the repacking does or does not do for you, it becomes very hard to sort out the big models with partial offloads, offload policy, numa, what runs on the GPU or CPU when and why, etc.

> 👤 **Ph0rk0z** replied the **2025-06-04** at **17:17:17**:<br>
> Worth a try. I will have to. I'm repacking exactly what I don't put on GPU and watching the layers in quantize, i.e which become _R8. One other metric would be to do 4096/2048 and see if it really is correlated to half batch size or bound to the 1024 size.
> 
> Is there a way to print exactly what tensors are repacked by RTR? I could be missing some tiny layers it did on it's own by using the regex offline.
> 
> Textgen is back to 18.x t/s after I dropped caches but prompt processing benchmarks hold universally through my tests.
> 
> 👤 **Ph0rk0z** replied the **2025-06-05** at **11:48:40**:<br>
> So I got it to print the tensors. The one that gets repacked by RTR and not offline repacking is token_embd. I had issues moving that tensor to either CPU or GPU manually.
> 
> Also notice that quantize will repack to R8, is there a difference between that and R4 as far as the various cuda implementations you are adding?
> 
> 👤 **ikawrakow** replied the **2025-06-05** at **11:56:57**:<br>
> `token_embd.weight` is never repacked and always stays on the CPU. It should not go to the GPU, and it should not get repacked. If you managed to make it repack, that's a bug, and you should tell me how you did it.
> 
> For some quantization one gets better CPU performance by interleaving 8 rows, so these are the `_R8` quants. `Q4_0`, `Q8_0` and `IQ4_XS` get repacked to `_R8`, all others are `_R4`. Some of those that are `_R4` would benefit from being `_R8`, but I haven't done it, and now that there are `_R4` quantized models floating around the Internet, I don't want to break backwards compatibility (and I don't want to carry `_R4` and `_R8` version of the same quantization type), so it will stay like this.
> 
> 👤 **Ph0rk0z** replied the **2025-06-05** at **12:49:05**:<br>
> I uncommented your line near where it says REPACKED XX Tensors which purportedly printed what was repacked. Everything else matches what I sent to CPU. Either the print is incorrect or it repacked it. 
> 
> Its strange too because I had tried to find layers to to throw on the CPU for just a few MB since my command line was OOM at 22k. Finally settled on 10 ffn_gate_inp towards the end. When I put token_embd=CPU I'd get a crash on qwen right away.
> 
> I just realized that *all* of my quants are IQ something. Wonder if it's related. Also tried offload policy from -1 to 29, negligible speed differences all around. Got deepseek lite a while ago which fits on one GPU but it's also IQ4_XS. Perhaps I should download a Q4_K instead.
> 
> edit:I enabled a further debug printout that says what got repacked to what and emb isn't there.

---

👤 **Ph0rk0z** replied the **2025-06-06** at **17:29:36**:<br>

Finally got around to testing a smaller model. Non IQ quant as well.

<details><summary>DeepSeek-V2-Lite-Chat.i1-Q4_K_M</summary>

    CUDA_VISIBLE_DEVICES= numactl --interleave=all ./bin/llama-sweep-bench \
    -m DeepSeek-V2-Lite-Chat.i1-Q4_K_M.gguf \
    -t 48 \
    -c 32768 \
    --numa distribute \
    -ngl 0 \
    -ctk q8_0 \
    -ctv q8_0 \
    -fa \
    -fmoe \
    -rtr \
    -b 4096 \
    -ub 4096 

</details>

No RTR 48c CPU distribute, cache on GPU

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |    2.955 |  1386.18 |   36.494 |    28.06 |
|  4096 |   1024 |   4096 |    3.047 |  1344.07 |   60.110 |    17.04 |
|  4096 |   1024 |   8192 |    3.338 |  1227.20 |   82.831 |    12.36 |
|  4096 |   1024 |  12288 |    3.611 |  1134.32 |  103.469 |     9.90 |
|  4096 |   1024 |  16384 |    3.861 |  1060.81 |  125.330 |     8.17 |


RTR 48c CPU distribute, Cache on GPU (iqk_repack_tensor(output.weight): q6_K -> q6_k_r4. 102400 rows, 3200 chunks, 48 threads)

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |   11.081 |   369.65 |   32.316 |    31.69 |
|  4096 |   1024 |   4096 |   13.410 |   305.44 |   53.593 |    19.11 |
|  4096 |   1024 |   8192 |   15.889 |   257.79 |   74.674 |    13.71 |


24 cores, numa isolate + RTR + no interleave

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |   19.223 |   213.08 |   30.327 |    33.76 |
|  4096 |   1024 |   4096 |   23.378 |   175.21 |   64.052 |    15.99 |
|  4096 |   1024 |   8192 |   28.008 |   146.25 |   97.014 |    10.56 |


24 cores, no interleave + no rtr + numa isolate


|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |    3.352 |  1221.83 |   46.758 |    21.90 |
|  4096 |   1024 |   4096 |    3.448 |  1187.76 |   81.010 |    12.64 |
|  4096 |   1024 |   8192 |    3.730 |  1098.15 |  113.951 |     8.99 |


GPU Fully

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |    0.730 |  5613.13 |    7.402 |   138.33 |
|  4096 |   1024 |   4096 |    0.863 |  4745.09 |   10.398 |    98.48 |
|  4096 |   1024 |   8192 |    1.115 |  3674.86 |   13.378 |    76.55 |

No GPU full cores no rtr

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |   13.485 |   303.75 |   36.449 |    28.09 |
|  4096 |   1024 |   4096 |   15.527 |   263.81 |   58.686 |    17.45 |
|  4096 |   1024 |   8192 |   18.000 |   227.55 |   79.114 |    12.94 |


No GPU full cores RTR

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |   10.863 |   377.07 |   33.246 |    30.80 |
|  4096 |   1024 |   4096 |   13.005 |   314.95 |   54.394 |    18.83 |
|  4096 |   1024 |   8192 |   15.463 |   264.88 |   75.656 |    13.53 |


It looks like on this system, RTR only helps when there is no GPU involved or the ubatch is 1024 (previous tests). In every other case, RTR lowers the prompt processing by a lot but improves TG.

> 👤 **ciprianveg** replied the **2025-06-10** at **16:08:25**:<br>
> I noticed it too, and iQ3_XXS_UD pp speed is affected by rtr much more than other quants, it drops from 250t/s to 26t/s, cca 10x slower. q2_xl_ud drops only from 245 to 140t/s. I am using no-mmap and swap disabled..
> 
> It is a pitty because while dropping pp speed 90%, it increases the generation speed by 40%.
> 
> i have a TR 3955 and 2x3090.
> built with: cmake -B build -DGGML_CUDA=ON -DGGML_RPC=OFF -DGGML_BLAS=OFF  -DGGML_SCHED_MAX_COPIES=1   -DGGML_CUDA_IQK_FORCE_BF16=1
> 
> started with:
> -ctx-size 71680 \
>     -ctk q8_0 \
>     -mla 3 \
>     -fa \
>     -amb 512 \
>     -fmoe \
>     --temp 0.6 \
>     --top_p 0.95 \
>     --min_p 0.01 \
>     --n-gpu-layers 63 \
>     -ot "blk\.[0-3]\.ffn_up_exps=CUDA0,blk\.[0-3]\.ffn_gate_exps=CUDA0,blk\.[0-3]\.ffn_down_exps=CUDA0"  \
>     -ot "blk\.1[0-1]\.ffn_up_exps=CUDA1,blk\.1[0-1]\.ffn_gate_exps=CUDA1,blk\.1[0]\.ffn_down_exps=CUDA1"    \
>     --override-tensor exps=CPU \
>     --parallel 1 \
>     --threads 16 \
>     --threads-batch 15 \
>     --host 0.0.0.0 --port 5002   \
>     --ubatch-size 7168 --batch-size 7168  --no-mmap
>     
>     BUT, if i build it with: cmake -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DGGML_CUDA=ON -DGGML_SCHED_MAX_COPIES=1
>     
>     no pp decrease anymore, but no tg speed increase, too..
> 
> 👤 **Ph0rk0z** replied the **2025-06-11** at **11:40:47**:<br>
> Could it be using BLAS instead of cuda when built with it? While ubatch size 1024 isn't as good as 4096+, it gives me a happy medium to use the RTR's textgen speed increase.