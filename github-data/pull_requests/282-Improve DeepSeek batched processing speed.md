### 🔀 [#282](https://github.com/ikawrakow/ik_llama.cpp/pull/282) - Improve DeepSeek batched processing speed

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-03-23 |
| **Updated** | 2025-03-23 |

---

#### Description

I was looking into the batched processing performance dips observed by @saood06 [here](https://github.com/ikawrakow/ik_llama.cpp/pull/277#issuecomment-2745952185) and I saw this for DeepSeek-Lite:

![batched0](https://github.com/user-attachments/assets/63d465fc-bf18-403c-839b-c68f392ed1f7)

Commandline was
```
./bin/llama-batched-bench -m junk1.bin -npp 512 -ntg 128 -npl 4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120 -pps -fmoe -fa -mla 3 -t 16
```
It took me a while to figure out the reason for the dramatic drop in performance between a batch size of 16 and a batch size of 20. I was suspecting that something goes wrong how the work is being distributed between the threads. But at the end it turned out that it is due to the way the compute graph is built: when `n_token > n_head` we switch to "PP optimized" processing, which means we go from FA with `Dk = 576, Dv = 512` to `Dk = 192, Dv = 128`, which requires two additional matrix multiplications. For DeepSeek-Lite `n_head = 16`, so with steps of 4 for the batch size 20 is exactly where the switch is made. I'm not sure what the rationale was for selecting this specific transition point (the optimization came from the [mainline llama.cpp PR](https://github.com/ggml-org/llama.cpp/pull/11446), but it clearly kills performance. If we look at prompt processing performance using "PP optimized" vs "TG optimized" DeepSeek compute graphs, we see this picture:

 
![pp_opt](https://github.com/user-attachments/assets/8b981565-9eb9-4bf4-b35e-48e6ed5ec028)

I.e., "TG optimized" is better than "PP optimized" for prompt lengths up to 64 tokens, and is not too far behind at 128 tokens. So, we can easily solve the performance drop by using "TG optimized" up to `n_prompt = 128`. By doing that, we get this result:

 
![batched](https://github.com/user-attachments/assets/79859f66-1147-4173-8ada-6916e7f07286)

The calculations take quite some time, so I didn't have the patience to run beyond batch size of 100 to see the exact crossover point. But eyeballing the graph, it looks like 128 is a good choice for DeepSeek-Lite. DeepSeek-V3/R1 have 128 heads, so this PR will not change the behavior for this models. But it isn't clear to me if one shouldn't use a larger threshold for the "TG optimized" -> "PP optimized" transition.

Concerning DeepSeek-R1, there is a small change in this PR that I hope will reduce the performance dips observed by @saood06

---

#### 💬 Conversation

👤 **saood06** commented the **2025-03-23** at **11:32:00**:<br>

>Concerning DeepSeek-R1, there is a small change in this PR that I hope will reduce the performance dips observed by @saood06

Running sweep bench and will post full results with graph when they finish, but right now but early results look promising showing

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |   49.636 |    10.32 |   39.574 |     3.23 |
|   512 |    128 |    512 |   57.011 |     8.98 |   43.246 |     2.96 |
|   512 |    128 |   1024 |   62.986 |     8.13 |   42.916 |     2.98 |
|   512 |    128 |   1536 |   63.400 |     8.08 |   44.014 |     2.91 |
|   512 |    128 |   2048 |   66.228 |     7.73 |   47.167 |     2.71 |
|   512 |    128 |   2560 |   72.508 |     7.06 |   46.553 |     2.75 |
|   512 |    128 |   3072 |   74.616 |     6.86 |   47.772 |     2.68 |
|   512 |    128 |   3584 |   80.675 |     6.35 |   50.907 |     2.51 |
|   512 |    128 |   4096 |   87.558 |     5.85 |   50.432 |     2.54 |
|   512 |    128 |   4608 |   88.584 |     5.78 |   53.859 |     2.38 |
|   512 |    128 |   5120 |   92.838 |     5.52 |   54.277 |     2.36 |
|   512 |    128 |   5632 |   99.437 |     5.15 |   54.257 |     2.36 |


I see you pushed another commit, should I stop this test and recompile and run the new commit?

---

👤 **ikawrakow** commented the **2025-03-23** at **11:34:40**:<br>

> I see you pushed another commit, should I stop this test and recompile and run the new commit?

This will only affect results for `B > 128`, so beyond the range where you are testing, so no need to rerun.

---

👤 **ikawrakow** commented the **2025-03-23** at **11:51:34**:<br>

What would be very interesting is to run PP benchmarks with DeepSeek-V3/R1 with `./bin/llama-bench -mla 3 -fa 1 -fmoe 1 -p 32,64,128,192,256,320,384,448,512,576,640,704,768` with
* [This line](https://github.com/ikawrakow/ik_llama.cpp/blob/5a4855e61c05b0c54ecad3f4155074d8f344b6f6/src/llama.cpp#L13899) changed to `pp_opt = true`;
* The same line changed to `pp_opt = false`;

This will help understand if the crossover between "TG optimized" and "PP optimized" is somehow dependent on the number of heads, or if it is just a (perhaps somewhat computer dependent) constant. I can see arguments for both options, so the only way to understand is to just test.

---

👤 **saood06** commented the **2025-03-23** at **13:28:00**:<br>

> What would be very interesting is to run PP benchmarks with DeepSeek-V3/R1 with `./bin/llama-bench -mla 3 -fa 1 -fmoe 1 -p 32,64,128,192,256,320,384,448,512,576,640,704,768` with
> 
>     * [This line](https://github.com/ikawrakow/ik_llama.cpp/blob/5a4855e61c05b0c54ecad3f4155074d8f344b6f6/src/llama.cpp#L13899) changed to `pp_opt = true`;
> 
>     * The same line changed to `pp_opt = false`;
> 
> 
> This will help understand if the crossover between "TG optimized" and "PP optimized" is somehow dependent on the number of heads, or if it is just a (perhaps somewhat computer dependent) constant. I can see arguments for both options, so the only way to understand is to just test.

Running now, each config is going to take ~50 minutes.

---

👤 **saood06** commented the **2025-03-23** at **16:56:44**:<br>

@ikawrakow Here's the benchmark you asked for:

On https://github.com/ikawrakow/ik_llama.cpp/pull/282/commits/d12f4a12aa0f2a31b20d08e2a8f500eb6b441459 with `bool pp_opt = n_tokens > n_head;`


| model                          |       size |     params | backend    | threads | fa | mla | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -: | --: | ---: | ------------: | ---------------: |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      48 |  1 |   3 |    1 |          pp32 |     10.30 ± 0.12 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      48 |  1 |   3 |    1 |          pp64 |     10.46 ± 0.66 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      48 |  1 |   3 |    1 |         pp128 |     11.25 ± 0.69 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      48 |  1 |   3 |    1 |         pp192 |      9.35 ± 0.34 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      48 |  1 |   3 |    1 |         pp256 |      9.46 ± 0.13 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      48 |  1 |   3 |    1 |         pp320 |      9.15 ± 0.29 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      48 |  1 |   3 |    1 |         pp384 |      9.43 ± 0.33 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      48 |  1 |   3 |    1 |         pp448 |     10.05 ± 0.16 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      48 |  1 |   3 |    1 |         pp512 |     10.30 ± 0.11 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      48 |  1 |   3 |    1 |         pp576 |      9.97 ± 0.20 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      48 |  1 |   3 |    1 |         pp640 |      9.62 ± 0.20 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      48 |  1 |   3 |    1 |         pp704 |      9.43 ± 0.14 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      48 |  1 |   3 |    1 |         pp768 |      9.51 ± 0.16 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      48 |  1 |   3 |    1 |         tg128 |      2.84 ± 0.00 |

On ec4bc75f with `bool pp_opt = true;`

| model                          |       size |     params | backend    | threads | fa | mla | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -: | --: | ---: | ------------: | ---------------: |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      48 |  1 |   3 |    1 |          pp32 |      9.15 ± 0.06 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      48 |  1 |   3 |    1 |          pp64 |      9.91 ± 0.61 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      48 |  1 |   3 |    1 |         pp128 |     11.20 ± 0.38 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      48 |  1 |   3 |    1 |         pp192 |      9.25 ± 0.48 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      48 |  1 |   3 |    1 |         pp256 |      9.11 ± 0.29 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      48 |  1 |   3 |    1 |         pp320 |      8.96 ± 0.18 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      48 |  1 |   3 |    1 |         pp384 |      9.17 ± 0.12 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      48 |  1 |   3 |    1 |         pp448 |      9.93 ± 0.13 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      48 |  1 |   3 |    1 |         pp512 |     10.07 ± 0.31 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      48 |  1 |   3 |    1 |         pp576 |      9.66 ± 0.21 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      48 |  1 |   3 |    1 |         pp640 |      9.37 ± 0.10 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      48 |  1 |   3 |    1 |         pp704 |      9.26 ± 0.11 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      48 |  1 |   3 |    1 |         pp768 |      9.44 ± 0.20 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      48 |  1 |   3 |    1 |         tg128 |      0.99 ± 0.02 |

On ec4bc75f with `bool pp_opt = false;`

| model                          |       size |     params | backend    | threads | fa | mla | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -: | --: | ---: | ------------: | ---------------: |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      48 |  1 |   3 |    1 |          pp32 |     10.09 ± 0.17 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      48 |  1 |   3 |    1 |          pp64 |     10.09 ± 0.53 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      48 |  1 |   3 |    1 |         pp128 |     10.50 ± 0.60 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      48 |  1 |   3 |    1 |         pp192 |      8.79 ± 0.37 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      48 |  1 |   3 |    1 |         pp256 |      8.70 ± 0.12 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      48 |  1 |   3 |    1 |         pp320 |      8.39 ± 0.17 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      48 |  1 |   3 |    1 |         pp384 |      8.74 ± 0.09 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      48 |  1 |   3 |    1 |         pp448 |      8.85 ± 0.15 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      48 |  1 |   3 |    1 |         pp512 |      9.48 ± 0.15 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      48 |  1 |   3 |    1 |         pp576 |      9.28 ± 0.02 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      48 |  1 |   3 |    1 |         pp640 |      8.89 ± 0.30 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      48 |  1 |   3 |    1 |         pp704 |      8.67 ± 0.10 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      48 |  1 |   3 |    1 |         pp768 |      8.69 ± 0.13 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      48 |  1 |   3 |    1 |         tg128 |      2.87 ± 0.00 |

I'm going to reboot my machine now to enable 1GB hugepages and mitigations=off and run a sweep-bench to see if TG performance increases.

---

👤 **ikawrakow** commented the **2025-03-23** at **17:10:32**:<br>

Thanks, this is great! It looks like a threshold of 128 tokens is not a bad choice for DeepSeek-R1 as well.