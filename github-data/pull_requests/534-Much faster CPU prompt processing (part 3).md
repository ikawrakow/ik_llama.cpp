### 🔀 [#534](https://github.com/ikawrakow/ik_llama.cpp/pull/534) - Much faster CPU prompt processing (part 3)

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-18 |
| **Updated** | 2025-06-19 |

---

#### Description

This PR is a follow up of #531 and #533, and adds much faster GEMM for the remaining non-interleaved quants: `Q2_K, IQ4_XS, IQ4_NL, Q4_0, Q4_1, Q5_0, Q5_1, Q6_0, Q8_0`.

Here is a PP-512 performance comparison between the main branch and this PR for LLaMA-3.1-8B-Instruct on a Ryzen-7950X CPU:

| type | main (t/s)  | PR (t/s)  | Speedup |
| ---: | ---: | ---: | ---: |
| Q2_K   | 202.1 | 364.2 | 1.802 |
| IQ4_XS | 178.0 | 363.2 | 2.040 |
| IQ4_NL | 136.6 | 293.5 | 2.149 |
| Q4_0 | 155.6 | 300.9 | 1.934 |
| Q4_1 | 135.1 | 253.5 | 1.876 |
| Q5_0 | 147.5 | 293.4 | 1.989 |
| Q5_1 | 124.9 | 253.5 | 2.030 |
| Q6_0 | 129.0 | 296.2 | 2.296 |
| Q8_0 | 145.9 | 293.5 | 2.012 |

We observe gains in the range of 2X for all types. In case anyone is wondering why we see 3 performance levels, this is simply due to the quantization type to which the data gets repacked:
* `Q2_K` and `IQ4_XS` get repacked to `Q8_K_R8`, and hence have a higher performance due to the faster `Q8_K_R8 x Q8_K` GEMM
* `IQ4_NL, Q4_0, Q5_0, Q6_0, Q8_0` get repacked to `Q8_0_R8`, so `Q8_0_R8 x Q8_2_X4` GEMM gets used, and they all end up with PP-512 in tghe 290-300 t/s range
* `Q4_1` and `Q5_1` get repacked to `Q8_1_R8` (they must due to being "type-1" quants), and that results in the lower performance around 250 t/s

---

#### 💬 Conversation

👤 **Nexesenex** submitted a review the **2025-06-18** at **13:46:15**: 💬 `COMMENTED`<br>

`
                float d = _mm_cvtss_f32(max4/127.f);
             
`
This line (2077) in idk_gemm_kquants.cpp provokes this error in MSVS 22 (Win 11) :

binary '/': '__m128' does not define this operator or a conversion to a type acceptable to the predefined operator.

I compile with AVX2 and FMA enabled.

---

👤 **Nexesenex** submitted a review the **2025-06-18** at **13:46:15**: 💬 `COMMENTED`<br>

`
                float d = _mm_cvtss_f32(max4/127.f);
             
`
This line (2077) in idk_gemm_kquants.cpp) provokes this error in MSVS 22 (Win 11) :

binary '/': '__m128' does not define this operator or a conversion to a type acceptable to the predefined operator.

I compile with AVX2 and FMA enabled.

---

👤 **ikawrakow** commented the **2025-06-18** at **13:49:36**:<br>

Should be fixed now.

---

👤 **Nexesenex** commented the **2025-06-18** at **14:05:57**:<br>

@ikawrakow : It is, thank you!

---

👤 **ubergarm** commented the **2025-06-18** at **15:25:16**:<br>

This 3 part refresh on PP performance across so many quants is epic, appreciate your explaining the details in your PR notes.

* `IQ4_NL`

Great to see this one in there too, I ran into it yesterday playing with [moonshotai/Kimi-Dev-72B](https://huggingface.co/moonshotai/Kimi-Dev-72B) which is a fine-tune of Qwen-2.5-72B architecture.

Turns out for those models the `ffn_down.weight, shape = {29568, 8192}` the column size is not divisible by 256, which sent me back over a year in time a year to your earlier notes:

> IQ4_NL: 4-bit non-linear quants with blocks of 32
> The main purpose of this PR is to provide a 4-bit quantization type that can be used when k- and i-quants that use blocks of 256 are not available (because the number of columns in some tensors are not a multiple of 256).
> https://github.com/ggml-org/llama.cpp/pull/5590#issue-2142529815

I saw some notes on [vLLM about padding out 29568 + 128 intermediate size before quantization](https://github.com/QwenLM/Qwen2.5-VL/issues/230#issuecomment-2370831542) and I believe turboderp's exllamav3 `EXL3` blocks of 128x128 weights and supports padding.

Are there any quantization/padding options I have to deal with this `ffn_down` tensor? In existing GGUFs seems like folks tend to leave it at `Q8_0` or `Q5_1` or use `IQ4_NL` as I was doing in my testing. 

I'll need to re-run some llama-sweep-bench testing, but I made a shotgun collection of experimental quants of this dense 72B hoping to find a good mix for 16-24GB VRAM hybrid inferencing.

While the prompt processing speeds are excellent (especially given probably less than 32k context), the token generation speeds seem bottlenecked by RAM i/o. The solution there is use a smaller size quant to fit more layers on GPU, but that directly eats into Perplexity score. I'm still feeling around for that "knee" point in the curve to get a fair trade-off in TG and Perplexity.

No wonder many folks are choosing MoEs for hybrid inference over dense 72Bs. Moe's fewer active weights during TG yield faster speeds with larger overall parameter size models.

![ppl-Kimi-Dev-72B](https://github.com/user-attachments/assets/34329f87-afb4-4765-b6ad-1884873bd8c0)

---

👤 **ikawrakow** commented the **2025-06-18** at **16:07:25**:<br>

> No wonder many folks are choosing MoEs for hybrid inference over dense 72Bs. Moe's fewer active weights during TG yield faster speeds with larger overall parameter size models.

TG performance of MoE models is far away from what is theoretically possible. If I look at your 6980P system, IIRC it has in the range of 512 GB/s memory bandwidth per node. So that, running DeepSeek on a single node because we haven't learnt how to do the NUMA thing effectively, and getting 10 t/s for 20 GB worth of active parameters means we are a factor of 2.5X away from what should be achievable. I do fully saturate memory bandwidth of my systems with the dense models I can run, so I was hoping that one can get that with a 70B dense model as well (on a higher bandwidth system). If so, quantized at 4 bpw one should be getting in the range of 15 t/s TG on your rig for this 70B dense model running CPU only.

> Turns out for those models the ffn_down.weight, shape = {29568, 8192} the column size is not divisible by 256, which sent me back over a year in time a year to your earlier notes:

If I was the Emperor of the Universe, I would put people creating models with strange tensor dimensions in prison. They haven't heard that modern computing architectures strongly prefer to operate on data sizes that are a high power of 2? And I mean, do they really believe that it makes a difference if the FFN tensors were 29440 or 29696 instead of 29568? Hahaha.

> Are there any quantization/padding options I have to deal with this ffn_down tensor? In existing GGUFs seems like folks tend to leave it at Q8_0 or Q5_1 or use IQ4_NL as I was doing in my testing.

Padding was discussed back in the day, but the idea was discarded. After all, it is `ggml` we are talking about. There used to be k-quants with a super-block size of 64, but as it was burdensome to maintain both, at some point the block of 64 variant got thrown out. In any case, yes, you need to use one of the quants with a block size of 32. `IQ4_NL` if you are targeting a lower bpw version, `Q5_0` or `Q6_0` for higher bpw quantization. I was thinking to make the trellis quants with a block size of 32, but that is much more tedious when handling the block scales, so I didn't do it. Maybe I should change them before trellis models become available?

---

👤 **saood06** commented the **2025-06-18** at **16:41:02**:<br>

> TG performance of MoE models is far away from what is theoretically possible. If I look at your 6980P system, IIRC it has in the range of 512 GB/s memory bandwidth per node. So that, running DeepSeek on a single node because we haven't learnt how to do the NUMA thing effectively, and getting 10 t/s for 20 GB worth of active parameters means we are a factor of 2.5X away from what should be achievable.

I do think now that we have the -ot, if the GGUF were changed to split up the experts and you launched it with `numactl --membind=[...]  --cpunodebind=[...]`, that might help (due to NUMA aware, expert parallelism).

---

👤 **ubergarm** commented the **2025-06-18** at **23:51:50**:<br>

@ikawrakow 

Always appreciate your insights, and these new prompt processing numbers are looking great on avx2 CPUs!

> I was hoping that one can get that with a 70B dense model as well (on a higher bandwidth system).

I ran `sweep-bench` for a few of my ~4 BPW 72B Dense models shown in the graph above on two big remote rigs compiled CPU-only. I was kinda surprised by the results.

![sweep-bench-Kimi-Dev-72B](https://github.com/user-attachments/assets/57f894f4-3103-4099-8d88-f36f4865d6d4)

My impression is that the big 6980P CPU is not saturating the expected ~512GB socket RAM bandwidth during generation. As you mentioned it could hit theoretically ~15 tok/sec (512 GB bandwidth / 32GB model size = 16 tok/sec). While the 24x Core 7965WX Thread Ripper Pro is doing better, it too has 4x CCDs configured as a single NUMA node via NPS1.

I spot checked using 80 and 64 threads for TG on the Intel Xeon 6980P, but less threads led to slower generation for this benchmark. Perhaps because its 3x CCDs are configured as a single NUMA node via BIOS config `SNC=Disable`. Though probably won't be able to reboot it to try, though the model *would fit* in the 256GB RAM if configured as one NUMA node per CCD.

Assuming the benchmarked ~512GB/s RAM bandwidth on the 6980P and let's call it ~256 GB/s on the Thread Ripper Pro are accurate, the potential token generation breakdown looks like this:

| Rig | Model | Theoretical | Measured | Yield |
| --- | --- | --- | --- | --- |
| | | tok/sec | tok/sec | % |
| 6980P | Q4_0 | 13.4 | 5.47 | 40.8% |
| " | smol-IQ3_K | 15.9 | 6.05 | 38.1% |
| " | IQ3_KT | 16.8 | 3.76 | 22.4% |
| 7965WX | Q4_0 | 6.7 | 4.74 | 70.7% |
| " | smol-IQ3_K | 7.9 | 5.61 | 71.0% |
| " | IQ3_KT | 8.4 | 3.06 | 36.4% |

I want to like the ~70B dense models, but man they are difficult to get good TG without offloading the whole thing to VRAM... I could try my home AMD 9950X given it would fit, even with lower absolute TG speeds it could be more "efficient" given native single NUMA node...

<details>

<summary>👈 Commands, Data, Model Descriptions</summary>

#### Q4_0
*extra* pure
- 38.095 GiB (4.501 BPW)
- type  f32:  401 tensors
- type q4_0:  562 tensors everything including embedding/output

### smol-IQ3_K
(its called `smol` just to match its PPL value from previous graph)
- 32.273 GiB (3.813 BPW)
- type  f32:  401 tensors
- type q4_K:    1 tensors embedding
- type q6_K:    1 tensors output
- type iq4_nl:   80 tensors down
- type iq3_k:  320 tensors (q|o) (gate|up)
- type iq4_k:  160 tensors (k|v)
 
### IQ3_KT
using the most recent PR merged into main
- 30.417 GiB (3.594 BPW)
- type  f32:  401 tensors
- type q4_K:    1 tensors embedding
- type q6_K:    1 tensors output
- type iq4_nl:   80 tensors down
- type iq3_kt:  320 tensors (q|o) (gate|up)
- type iq4_kt:  160 tensors (k|v)

```bash
# on the Thread Ripper Pro I removed numactl stuff and used 24 threads.
numactl -N 0 -m 0 \
    ./build/bin/llama-sweep-bench \
        --model "$model" \
        --ctx-size 6144 \
        -ctk q8_0 -ctv q8_0 \
        -fa \
        --no-mmap \
        -ub 2048 -b 2048 \
        --warmup-batch \
        --threads 128 \
        --threads-batch 128 \
        --numa numactl
```

## 6980P Q4_0 -t 128
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  2048 |    512 |      0 |   17.241 |   118.79 |   93.585 |     5.47 |
|  2048 |    512 |   2048 |   18.073 |   113.32 |   95.782 |     5.35 |
|  2048 |    512 |   4096 |   19.067 |   107.41 |   97.443 |     5.25 |

## 6980P smol-IQ3_K -t 128
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  2048 |    512 |      0 |   17.715 |   115.61 |   84.592 |     6.05 |
|  2048 |    512 |   2048 |   18.753 |   109.21 |   85.094 |     6.02 |
|  2048 |    512 |   4096 |   19.438 |   105.36 |   86.905 |     5.89 |

## 6980P IQ3_KT -t 128
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  2048 |    512 |      0 |   17.356 |   118.00 |  136.233 |     3.76 |
|  2048 |    512 |   2048 |   18.462 |   110.93 |  139.345 |     3.67 |
|  2048 |    512 |   4096 |   18.944 |   108.11 |  140.283 |     3.65 |

## 7965WX Q4_0 -t 24
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  2048 |    512 |      0 |   44.916 |    45.60 |  108.030 |     4.74 |
|  2048 |    512 |   2048 |   47.595 |    43.03 |  110.270 |     4.64 |
|  2048 |    512 |   4096 |   50.202 |    40.80 |  113.182 |     4.52 |

## 7965WX smol-IQ3_K -t 24
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  2048 |    512 |      0 |   35.626 |    57.49 |   91.275 |     5.61 |
|  2048 |    512 |   2048 |   38.347 |    53.41 |   93.747 |     5.46 |
|  2048 |    512 |   4096 |   40.987 |    49.97 |   96.587 |     5.30 |

## 7965WX IQ3_KT -t 24
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  2048 |    512 |      0 |   44.884 |    45.63 |  167.161 |     3.06 |
|  2048 |    512 |   2048 |   47.600 |    43.03 |  169.435 |     3.02 |
|  2048 |    512 |   4096 |   50.176 |    40.82 |  172.420 |     2.97 |

</details>

> Padding was discussed back in the day

I was checking how [bullerwins dealt with the goofy dimensions ffn_down.](https://huggingface.co/bullerwins/Kimi-Dev-72B-GGUF/discussions/1#6852fc43cd6b6db96eb0980e). Given they use `Q8_0` I was surprised to hear the log mentioned padding:

```
29 568 / 256 = 115 full blocks  (115 × 256 = 29 440)
remainder              128 elements (padded to 256)
```

I didn't look into it further, and used `IQ4_NL` for the above test quants which is a reasonable size for these quants.

> Maybe I should change them before trellis models become available?

Right, related to the `iqN_kt` quants merged in [PR529](https://github.com/ikawrakow/ik_llama.cpp/pull/529), I haven't released anything yet. Going through the trouble to make block size 32 might not be worth it unless cursed sized tensor column dimensions becomes more prevalent as `iq4_nl` seems pretty solid. Not sure how changing the block size would effect TG performance as well.

The PP performance on the `iqN_kt` quants is amazing, about the highest despite being on the [B Tier Q8_0_R8 mul_mat list](https://github.com/ikawrakow/ik_llama.cpp/pull/495#issuecomment-2985633815), but I noticed the TG performance is lagging behind the other quants which I assume is to extra CPU overhead dealing with them?

Same with DeepSeek-R1-0528 which I run here offloading `-ngl 99 -ot exps=CPU` plus 16 more layers on dual RTX A6000 GPU (to not OOM RAM), on the Thread Ripper Pro, 24 core, default batch sizes:

#### IQ3_KS_R4 300.938 GiB (3.847 BPW)
* 12.39 tok/sec TG
- type  f32:  361 tensors
- type q8_0:  612 tensors attn/shexp/embedding
- type iq3_k_r4:  116 tensors (gate|up)
- type iq4_ks_r4:   58 tensors down

#### IQ3_KT 272.527 GiB (3.483 BPW)
* 8.61 tok/sec TG
- type  f32:  361 tensors
- type q5_0:   61 tensors attn_k_b
- type q8_0:    1 tensors embedding
- type iq5_ks:  550 tensors attn/shexp
- type iq3_kt:  116 tensors down
- type iq4_kt:   58 tensors (gate|up)

<details>

<summary>llama-sweep-bench details and data</summary>

Ignore the PP given this was low batch sizes so not a good comparison.

```bash
#model=/mnt/raid/hf/DeepSeek-R1-0528-GGUF/IQ3_K_R4/DeepSeek-R1-0528-IQ3_K_R4-00001-of-00007.gguf
model=/mnt/raid/hf/DeepSeek-R1-0528-GGUF/IQ3_KT/DeepSeek-R1-0528-IQ3_KT-00001-of-00006.gguf

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

## IQ3_KS_R4
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    4.470 |   114.55 |   10.332 |    12.39 |
|   512 |    128 |    512 |    5.504 |    93.03 |   10.412 |    12.29 |
|   512 |    128 |   1024 |    4.614 |   110.96 |   10.451 |    12.25 |
|   512 |    128 |   1536 |    4.825 |   106.12 |   10.475 |    12.22 |
|   512 |    128 |   2048 |    4.863 |   105.28 |   10.470 |    12.23 |
|   512 |    128 |   2560 |    4.969 |   103.04 |   10.621 |    12.05 |
|   512 |    128 |   3072 |    5.238 |    97.74 |   10.666 |    12.00 |
|   512 |    128 |   3584 |    5.130 |    99.81 |   10.684 |    11.98 |
|   512 |    128 |   4096 |    5.972 |    85.73 |   10.785 |    11.87 |
|   512 |    128 |   4608 |    5.392 |    94.96 |   10.715 |    11.95 |
|   512 |    128 |   5120 |    5.399 |    94.83 |   10.718 |    11.94 |
|   512 |    128 |   5632 |    5.490 |    93.27 |   10.882 |    11.76 |
|   512 |    128 |   6144 |    5.593 |    91.54 |   10.883 |    11.76 |
|   512 |    128 |   6656 |    5.602 |    91.39 |   10.919 |    11.72 |
|   512 |    128 |   7168 |    5.707 |    89.71 |   10.921 |    11.72 |
|   512 |    128 |   7680 |    5.803 |    88.23 |   10.924 |    11.72 |
|   512 |    128 |   8192 |    5.904 |    86.73 |   11.204 |    11.42 |

## IQ3_KT
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    9.604 |    53.31 |   14.861 |     8.61 |
|   512 |    128 |    512 |    9.337 |    54.83 |   14.948 |     8.56 |
|   512 |    128 |   1024 |    9.430 |    54.30 |   15.232 |     8.40 |
|   512 |    128 |   1536 |    9.929 |    51.57 |   15.232 |     8.40 |
|   512 |    128 |   2048 |   10.088 |    50.76 |   15.035 |     8.51 |
|   512 |    128 |   2560 |   10.250 |    49.95 |   15.132 |     8.46 |
|   512 |    128 |   3072 |   10.542 |    48.57 |   15.189 |     8.43 |
|   512 |    128 |   3584 |   10.404 |    49.21 |   15.242 |     8.40 |
|   512 |    128 |   4096 |   10.858 |    47.15 |   15.204 |     8.42 |
|   512 |    128 |   4608 |   10.433 |    49.08 |   15.234 |     8.40 |
|   512 |    128 |   5120 |   10.389 |    49.29 |   15.638 |     8.19 |
|   512 |    128 |   5632 |   10.889 |    47.02 |   15.753 |     8.13 |
|   512 |    128 |   6144 |   10.754 |    47.61 |   15.448 |     8.29 |
|   512 |    128 |   6656 |   10.670 |    47.98 |   15.482 |     8.27 |
|   512 |    128 |   7168 |   10.681 |    47.94 |   15.796 |     8.10 |
|   512 |    128 |   7680 |   10.804 |    47.39 |   15.812 |     8.10 |
|   512 |    128 |   8192 |   11.206 |    45.69 |   15.643 |     8.18 |

</details>

So given DeepSeek-R1-671B has active 37B during generation and the theoretical max bandwidth on the 256GB/s Thread Ripper Pro we can use the calculate the GiB of the active parameters and get theoretical max TG as above.

`256 / (37 * (BPW/8)`

| Rig | Model | Theoretical | Measured | Yield |
| --- | --- | --- | --- | --- |
| | | tok/sec | tok/sec | % |
| 7965WX | IQ3_KS_R4 | 14.4 | 12.4 | 86.0% |
| " | IQ3_KT | 15.9 | 8.6 | 54.1% |

Thanks again for these great PP speed-ups and your time and patience with these long posts haha.. I gotta eat some dinner now, cheers!