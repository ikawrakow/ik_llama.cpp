### [Discussion #223](https://github.com/ikawrakow/ik_llama.cpp/discussions/223) - Recent performance testing with DeepSeek R1

| **Author** | `bitbottrap` |
| :--- | :--- |
| **Created** | 2025-02-22 |
| **Updated** | 2025-03-14 |

---

#### Description

I'm open to a more rigorous set of tests using accepted benchmark files. Just point me to them. I can run this periodically if it's scripted. Available are 2x24GB GPUs and 1TB of RAM on an Epyc CPU. 

Tested with:
commit 4b45b82e67d9362e7522e5c7107e9d99219e0432 (HEAD -> main, origin/main, origin/HEAD)
Author: Iwan Kawrakow <iwan.kawrakow@gmail.com>
Date:   Thu Feb 20 17:42:07 2025 +0200
Honor attn_output specified in the command line also for low-bit quants

DeepSeek R1 Q4_K_M

Only the MLA configuration worked at 163840 token context. Everything else was OOM.


Attention Type | rtr | CUDA | Context Size | KV Quant | Load Time (ms) | Tokens/Second (Prompt Eval) | Tokens/Second (Eval) | Notes
-- | -- | -- | -- | -- | -- | -- | -- | --
flash |  |  | 8192 | Q8 | 87751 | 43.22 | 1.68 |  
flash | X |  | 8192 | Q8 | 249508 | 58.58 | 1.89 |  
flash |  |  | 8192 |   | 146536 | 44.26 | 2.18 |  
flash | X |  | 8192 |   | 259598 | 52.65 | 2.18 |  
mla |  |  | 8192 |   | 74651 | 32.76 | 5.21 |  
mla | X |  | 8192 |   | 0 | 0 | 0 | FAIL, core dump
standard |  |  | 8192 |   | 94564 | 39.74 | 4.86 |  
standard | X |  | 8192 |   | 254080 | 48.15 | 4.87 |  
flash |  |  | 65536 |   | 249237 | 43.44 | 2.05 |  
flash | X |  | 65536 |   | 422931 | 55.18 | 2.06 |  
flash |  |  | 128000 |   | 416902 | 41.61 | 2.1 |  
flash | X |  | 128000 |   | 593555 | 50.35 | 2.12 |  
mla |  |  | 128000 |   | 274483 | 32.18 | 5.24 |  
standard |  |  | 128000 |   | 612123 | 39.96 | 4.81 |  
standard | X |  | 128000 |   | 731429 | 49.46 | 4.7 |  
flash |  |  | 163840 | Q8 | 413241 | 47.44 | 1.74 |  
flash | X |  | 163840 | Q8 | 444949 | 57.90 | 1.75 |  
mla |  |  | 163840 |   | 83955 | 31.3 | 5.25 |  
mla | X |  | 163840 |   | 0 | 0 | 0 | FAIL
flash |  | X | 8192 |   | 0 | 0 | 0 | fail: ggml_cuda_flash_attn_ext_wmma_f16: Unhandled head size 192
flash | X | X | 8192 |   | 397501 | 49.35 | 2.16 |  
mla |  | X | 8192 |   | 95964 | 22.77 | 5.22 | FAIL, garbage output
mla | X | X | 8192 |   | 0 | 0 | 0 | FAIL, core dump
standard | X | X | 8192 |   | 396659 | 50.17 | 4.84 |  
standard |   | X | 8192 |   | 126521 | 21.5 | 4.68 |

---

#### 🗣️ Discussion

👤 **saood06** commented on **2025-02-23** at **01:03:00**

Thank you so much for these results.

Also was the test conducted the same as before with a 500 token prompt and a 300 token response, or something different?

>I'm open to a more rigorous set of tests using accepted benchmark files.

I can make a branch containing what fairydreaming used to evaluate PP and TG performance.

From it's readme:

>Benchmark the prompt processing and token generation performance of `llama.cpp`
by doing a sweep over a whole context size and gathering performance metrics
in each ubatch-sized window. Only a single token sequence is used.
>[...]
>The purpose of the benchmark is to visualize how the performance changes with
the context size without averaging the metrics values over the whole context.

---

👤 **saood06** commented on **2025-02-23** at **01:36:47**

The fairydreaming benchmark includes a script that contains a python script that generates a graph that would display multiple configurations against each other here are two examples of it's output from fairydreaming ( [1](https://preview.redd.it/o2uxzg63x3he1.png?width=989&format=png&auto=webp&s=dc2743353f3d5a86258aa51efc7e18853e3911a0) and [2](https://www.reddit.com/r/LocalLLaMA/comments/1igpwzl/paradigm_shift/mawmoq0/) )

We could tell you what configs to run and then you just pass all the jsonl output from each config into the script and it outputs a graph.

Edit: Fixed image link to show PP instead of TG graph

> 👤 **saood06** replied on **2025-02-23** at **04:59:50**
> 
> >Being that the model is so large and the most value would probably be pushing limits of context tests take a while.
> 
> I understand my system is far weaker than yours (the highest PP I've seen is 11), and I've done overnight benchmarks so I do appreciate you doing this. I just created #225 for an easy to use but thorough benchmark, that will output nice graphs.
> 
> >But if there were a smaller number of runs, say up to 30ish that's reasonable to run overnight by request.
> 
> @ikawrakow Can you pick any runs you would like to see?

---

👤 **ikawrakow** commented on **2025-02-23** at **05:57:41**

Thank you for this!

What is the hardware configuration? (EPYC model, single or dual socket, how many RAM sticks and what type)

How many threads do you use when running the benchmarks?

I think the most pressing issue is to understand why TG performance with FA enabled is so low. Is it possible to run one FA configuration with varying number of threads (e.g., `llama_bench -m $model -p 0 -n 64 -t 2,4,8,16,...,max_threads`?

The MLA failures are also concerning, but solving them would require debugging.

CUDA does not support FA with different K and V head sizes and in the DeepSeekV3/R1 models, so no need to run those. I guess, I should add a check for that.

Run time repacking seems to be adding 2-3 minutes to the load time. This is better than I expected but I guess it could be very annoying if used regularly. I should try to optimize or perhaps create a tool to repack an existing model.

---

👤 **bitbottrap** commented on **2025-02-23** at **15:30:00**

Epyc 7773X (64 cores, 128 threads), one socket, 8x128GB RAM

For the above I used 63 threads as a balance between prefill and generation.

Is the run time repacking equivalent of using Q4_K_S versus quantizing a model with Q4_K_R4? Also, there is no repacking for Q4_K_M? If so, some of the comparisons are off as the models being compared are in fact different.

I don't think repacking time is important for such a large model. Can't imagine loading it on demand in many environments.

Here is a table of the benchmarks you asked for above.

threads | std | flash | mla
-- | -- | -- | --
2 | 0.99 | 0.92 | 0.99
4 | 1.89 | 1.7 | 1.86
8 | 3.25 | 2.89 | 3.26
16 | 4.6 | 4.04 | 4.64
24 | 4.81 | 4.03 | 4.82
32 | 4.81 | 4.17 | 4.8
48 | 4.75 | 4.08 | 4.75
64 | 4.69 | 4.14 | 4.73
96 | 4.56 | 4.05 | 4.64
128 | 4.49 | 4.11 | 4.59

---

👤 **ikawrakow** commented on **2025-02-23** at **16:08:15**

Thanks!

So, what is the difference between the above and the original table? Here we see FA having lower performance than std/MLA, but only 10-20% lower and not 2.5x lower as in the original table. FA having slightly lower TG performance is in line with the expectation. Its main benefit is prefill performance, so depending on context (number of tokens generated vs prompt length), it will often win against std or MLA in terms of total processing time. But not when TG performance is 2.5X lower...

> For the above I used 63 threads as a balance between prefill and generation.

63 or 64? 63 is really bad as suddenly number of rows in tensors is no longer a multiple of the number of threads, so threads process different portions, and one likely even ends up with false sharing (threads writing into the same cache line, triggering cache syncs with potentially disastrous effects on performance). You see a little bit of that in the FA column above at 24, 48 and 96 threads, but these are still relatively "nice" thread numbers compared to 63.

> Is the run time repacking equivalent of using Q4_K_S versus quantizing a model with Q4_K_R4?

Run-time-repacking (rtr) does not change the mix of quantization types. `Q4_K_M` is a mix of `Q4_K` and `Q5_K`, so after rtr we will have a corresponding mix of `Q4_K_R4` and `Q5_K_R4`. If you select `Q4_K_R4` as the quantization type during quantization, then yes, you basically end up with the same as `Q4_K_S` after rtr.

> Epyc 7773X (64 cores, 128 threads), one socket, 8x128GB RAM

OK, so this is Zen3, so using vanilla AVX2 implementation. If the information I find on the Internet is correct, it should have ~200 GB/s memory bandwidth. We have 37B active parameters at about 4.8 bpw for `Q4_K_M`, so about 22 GB of model weights are active, so we should be getting in the range of 8-9 t/s for TG. I wonder where is the bottleneck. I'm able to 100% saturate the memory bandwidth on a Ryzen-7950X (Zen4 core), Ryzen-5975WX (Zen3 core) and M2-Max with the models I can run.

---

👤 **ikawrakow** commented on **2025-02-24** at **14:35:34**

Really curious to see what happens with PR #232.

---

👤 **ikawrakow** commented on **2025-02-26** at **13:08:06**

> If you've got a command line that works with 1 or 2 24GB GPUs I'll start it up

Basically whatever command you use for your standard testing, but add `-ngl 999 -ot "\.ffn_.*_exps\.=CPU"`. My concept is that the non-expert tensors of DeepSeekV3/R1 (~17B) fit on a single 24GB GPU when quantized. I don't think `llama.cpp` (and by inheritance `ik_llama.cpp`) benefits from multiple GPU's performance wise, so the only benefit from using both GPU's would be the ability to process larger contexts (assuming one can meaningfully split the layers, but I have never played with that as I don't have access to a multi-GPU system).

>  One thing bothers me and that's the official llama.cpp doesn't like the standard quants that are generated. I used the evshiron convert_hf_to_gguf.py and llama.cpp complains about "wrong number of tensors; expected 1147, got 1025"

This bothers me too, but that's how it got implemented in this unmerged [llama.cpp PR](https://github.com/ggml-org/llama.cpp/pull/11446) where the MLA implementation here originally came from (but there have been quite a few improvements compared to the PR in `llama.cpp`). Basically, the tensors `wkv_b` get split into `wk_b` and `wv_b` by the `convert_hf_to_gguf.py` script, so there are more tensors in the GGUF produced by `ik_llama.cpp` compared to mainline. I have thought about removing this change from `convert_hf_to_gguf.py` and performing the split on-the-fly while loading the model. But then we run into issues with the imatrix stuff because `wk_b` and `wv_b` will not have entries in the imatrix file (so, no low-bit quantization is possible). It is also not possible to take an existing imatrix and split its `wkv_b` entries because `wv_b` is transposed. From my perspective `llama.cpp` goes too far in treating situations that, although unexpected, can be gracefully handled into fatal errors. In this particular case, all tensors that `llama.cpp` needs to run the model are present, so the presence of the additional `wk_b` and `wv_b` tensors shouldn't result in an error. But I guess that's what happens in a project with many users and few regular contributors who have the big picture.

On KV cache size: To match KTransformers, `ik_llama.cpp` must be able to handle a context of 8K tokens. Based on the figures you provide for a context of 163k tokens, 8K tokens will require ~1 GiB if left as `f16`, or 765 MiB if the K cache is quantized with `Q8_0`. Let's assume the non-experts are quantized with 6.5 bpw on average (for DeepSeekV3/R1 it is useful to use more bits for the attention tensors and shared experts). 17B * 6.5 bpw = 13.5 GiB. So, there would be ~10 GiB left for KV cache and compute buffers I don't know how much compute buffers are required for DeepSeekV3/R1, but it seems you will be able to go to 32K or perhaps 65K tokens with MLA. Going beyond that will require splitting the model between the two GPUs.

Of note: MLA is ~20% slower than standard attention for less than a few hundred tokens in the cache. It becomes competitive performance wise only beyond 16k tokens. With MLA there are two matrix multiplications that are extremely slow on CUDA. I'm trying to improve that but no luck so far.

> 👤 **ikawrakow** replied on **2025-02-26** at **17:33:19**
> 
> Oh, and adding `-fmoe` (or `-fmoe 1` with `llama-bench`) is useful too. This fuses the MoE matrix multiplications. Speedup is not dramatic, but we do get a few percent speedup for prefill and 1-2% for TG.

---

👤 **bitbottrap** commented on **2025-03-14** at **14:54:37**

So I was going to try and get a bunch of benchmarks with recent code and I encountered a problem using any GPU offloading. This was a feature that was working, but poorly, last time I did some hand testing.

The model is DeepSeek R1 Q8_0

| Configuration | Prompt Eval Time (tokens/s) | Eval Time (tokens/s) | Notes |
|-------------------------------|----------------------------|---------------------|---------------------------------|
| -mla 1 | 37.00 | 3.52 | |
| -mla 1 -fa | N/A | N/A | Segmentation fault (core dumped)|
| -mla 1 -fmoe | 37.55 | 3.53 | |
| -mla 1 -rtr | 43.58 | 3.50 | |
| -mla 1 -rtr -fmoe | 44.37 | 3.51 | |
| -mla 2 | 38.52 | 3.49 | |
| -mla 2 -fa | N/A | N/A | NO TEXT GENERATED |
| -mla 2 -fa -fmoe | N/A | N/A | NO TEXT GENERATED |
| -mla 2 -rtr | 45.41 | 3.47 | |
| -mla 2 -rtr -fmoe | N/A | N/A |Killed/crashed |
| -mla 2 -fmoe | 38.79 | 3.49 | |

Command lines like these with GPU offloading failed:
CUDA_VISIBLE_DEVICES=0 ~/llmla/ik_llama.cpp/build/bin/llama-cli -mla 2 -ngl 0 -b 1024 -ub 1024 -m DeepSeek-R1-Q8_0.gguf -c 8192 -t 64 --mlock -n 300 -f /mnt/data/prompt-prefill-benchmark.txt
CUDA error: out of memory

CUDA_VISIBLE_DEVICES=0 ~/llmla/ik_llama.cpp/build/bin/llama-cli -mla 1 -rtr -b 1024 -ub 1024 -m DeepSeek-R1-Q8_0.gguf -c 8192 -t 64 --mlock -n 300 -f /mnt/data/prompt-prefill-benchmark.txt -ngl 999 -ot "\.ffn_.*_exps\.=CPU"
died