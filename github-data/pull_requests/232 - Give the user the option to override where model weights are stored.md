### 🔀 [#232](https://github.com/ikawrakow/ik_llama.cpp/pull/232) - Give the user the option to override where model weights are stored

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-02-24 |
| **Updated** | 2025-02-27 |

---

#### Description

It seems this PR amounts to most of the "secret sauce" of KTransformers. 

We add a command line option to override where model weights are stored using regular expressions. This allows to keep the MoE experts on the CPU and to offload only the attention and not repeating layers to the GPU. The PR is inspired by https://github.com/ggml-org/llama.cpp/pull/11397, but `ik_llama.cpp` has now diverged so much from mainline that I had to do most of it new.

Unfortunately I cannot test with DeepSeekV3/R1, but here is what I get for DeepSeek-Lite (very similar MoE architecture) using
```
./bin/llama-bench -m deepseek_lite.gguf -p 512 -n 128 -t 16 -ngl 100 -rtr 1 -ot "\.ffn_.*_exps\.=CPU"
```

| model                          |       size |  threads |     test |  t/s (CPU only)  | t/s (CPU+GPU)    |   Speedup | 
| ------------------------------ | ---------: | ------: | -------: | ---------------: | ---------------: | --------: |
| deepseek2 16B Q4_K - Medium    |   9.78 GiB |      16 |    pp512 |    631.03 ± 4.89 |  1066.42 ± 29.88 |  1.690    |
| deepseek2 16B Q4_K - Medium    |   9.78 GiB |       16 |    tg128 |     28.70 ± 0.03 |     45.28 ± 0.05 |  1.578    |

The argument to the new `-ot` or `--override-tensor` option is
```
regular_expression=backend_name
```
In the above example we first ask all model layers to be offloaded to the GPU (`-ngl 100`), but then override all model tensors that match the regular expression `\.ffn_.*_exps\.` to be kept on the CPU (and also not offloaded to the GPU to perform operations on them).

The PR is still a bit rough around the edges (not much error handling, `mmap` gets disabled for the tensors with buffer type override, etc.), but throwing it out there to get feedback. 
Would love to hear from someone having a GPU with enough VRAM to fit all DeepSeekV3/R1 model weights on the GPU except the experts.

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-02-25** at **06:34:34**:<br>

Here some results using `IQ4_NL`

| model                 | threads | mla | rtr | fmoe |          test |              t/s |
| --------------------- | ------: | --: | --: | ---: | ------------: | ---------------: |
| deepseek2 16B IQ4_NL  |       8 |   1 |   1 |    1 |    tg64@pp128 |     53.08 ± 0.03 |
| deepseek2 16B IQ4_NL  |       8 |   1 |   1 |    1 |    tg64@pp256 |     52.87 ± 0.07 |
| deepseek2 16B IQ4_NL  |       8 |   1 |   1 |    1 |    tg64@pp512 |     52.53 ± 0.04 |
| deepseek2 16B IQ4_NL  |       8 |   1 |   1 |    1 |   tg64@pp1024 |     51.48 ± 0.10 |
| deepseek2 16B IQ4_NL  |       8 |   1 |   1 |    1 |   tg64@pp2048 |     50.40 ± 0.04 |
| deepseek2 16B IQ4_NL  |       8 |   1 |   1 |    1 |   tg64@pp4096 |     48.39 ± 0.13 |
| deepseek2 16B IQ4_NL  |       8 |   1 |   1 |    1 |   tg64@pp8192 |     44.00 ± 0.02 |

| model                 | mla | rtr | fmoe |          test |              t/s | 
| --------------------- | --: | --: | ---: | ------------: | ---------------: | 
| deepseek2 16B IQ4_NL  |   1 |   1 |    1 |         pp512 |   1172.35 ± 2.91 |   
| deepseek2 16B IQ4_NL  |   1 |   1 |    1 |        pp1024 |   1167.57 ± 1.75 |   
| deepseek2 16B IQ4_NL  |   1 |   1 |    1 |        pp2048 |   1148.17 ± 1.45 |   
| deepseek2 16B IQ4_NL  |   1 |   1 |    1 |        pp4096 |   1125.10 ± 1.52 |   
| deepseek2 16B IQ4_NL  |   1 |   1 |    1 |        pp8192 |   1067.71 ± 5.17 |
| deepseek2 16B IQ4_NL  |   1 |   1 |    1 |       pp16384 |    974.12 ± 0.85 | 

So, with attention running on the GPU, MLA is competitive with standard also for PP. Given the reduced KV cache size with MLA, it becomes the best option for this setup (CPU computes experts matrix multiplications, GPU computes everything else). 

Dumping some timing info for TG, in a run with 5 tg128 evaluations I get
* 55.5 t/s, so (640 tokens)/(55.5 tokens/second) = 11.53 seconds total evaluation time
* 8.42 seconds for computing the MoE experts matrix multiplications on the CPU
* 1.23 seconds for computing everything else on the GPU
* Hence, 11.53 - 8.42 - 1.23 = 1.88 seconds are spent in the `ggml` back-end on synchronization and copying data between CPU and GPU. This is ~16% of total evaluation time (!!!), and I think this is very far from optimal, so there is much room for improvement there. If this cost can be optimized out, we will be getting in the range of 65 t/s
* The experts in DeepSeek-Lite are `2048 x 1408`. We have `ffn_up, ffn_gate` and `ffn_down`, 6 active experts, and 25 experts layers. So, this is `2048 x 1408 x 3 x 6 x 25 = 1.298B` weights involved in the CPU calculation. Model is quantized with `IQ4_NL`, so 4.5 bits per weight, so `1298 x 4.5 / 8 = 730 MB` of data needs to be fetched from RAM per evaluated token. 640 tokens evaluated in 8.42 seconds is 0.01316 seconds per token. Hence, the memory bandwidth utilized during CPU computation is `730 MB / 0.01316 seconds = 55.5 GB/s`. The system (Ryzen-7950X) has 64 GB/s theoretical memory bandwidth, but 60 GB/s is the best one gets in practice for TG (with dense models). I.e., for this 6 active, 64 total experts MoE model we are at 90%+ of memory bandwidth utilization

---

👤 **saood06** commented the **2025-02-25** at **06:48:52**:<br>

>Hence, 11.53 - 8.42 - 1.23 = 1.88 seconds are spent in the ggml back-end on synchronization and copying data between CPU and GPU. This is ~16% of total evaluation time (!!!), and I think this is very far from optimal, so there is much room for improvement there. If this cost can be optimized out, we will be getting in the range of 65 t/s

Is the cost call overhead or throughput?

>Here is the op timing breakdown for 5 x tg128 runs

Also how do you generate these op timing breakdowns?

---

👤 **ikawrakow** commented the **2025-02-25** at **07:53:14**:<br>

> Is the cost call overhead or throughput?

I don't know. Haven't gone into the back-end code to break it down. But I suspect most of it is synchronization inefficiencies as there isn't much data to be sent back-and-fort when doing TG. 

> Also how do you generate these op timing breakdowns?

I set `IK_PRINT_TIMING` to 1 in `ggml.c` or `ggml-cuda.cu` and rebuild. Then I run the benchmark. This produces a lot of output. I have a simple program to read this output and prepare the timing statistics. I found this to be more reliable and easier to use than `perf`.

---

👤 **ikawrakow** commented the **2025-02-25** at **10:17:13**:<br>

> Is the cost call overhead or throughput?

For TG cost for copying data back-and-fort is negligible. Here is a rough breakdown of the 16% overhead:
* ~1% to build the graph and set the KQ mask
* ~2.3% to start the CPU threads evaluating the MoE experts matrix multiplications (this happens on every graph split, so 25 times per token). Here a thread pool might help to reduce this cost (but waking up threads blocked on a wait condition is not free either)
* ~13% is spent in `ggml_backend_sched_compute_splits()` up to the point where `ggml_backend_graph_compute_async()` is called. Out of these 13% about 2% go into copying data back-and-fort (including synchronization cost). To sort out the remaining 11% I need to actually understand what the code does (which I don't very well at this point)

For PP copying data back-and-fort is more significant. I tested with a context of 1024 and I see about 11% spent in `ggml_backend_sched_compute_splits()` before calling `ggml_backend_graph_compute_async()`. Out of these 11%, about 6% are spent on copying data between the GPU and the CPU (with the copy to the GPU taking most of the time).

---

👤 **ikawrakow** commented the **2025-02-26** at **06:55:34**:<br>

### Update:

I made a mistake in the above. I was using a model file that did not have the additional tensors required for MLA. But `llama-bench` swallows the output of the model loading, so I didn't see the warning that MLA is turned off. I have updated the tables to show `mla = 0`. Here is the actual TG performance with MLA enabled:

 | model                | mla | rtr | fmoe |          test |              t/s |
| -------------------- | --: | --: | ---: | ------------: | ---------------: |
| deepseek2 16B IQ4_NL |   1 |   1 |    1 |    tg64@pp128 |     46.16 ± 0.05 |
| deepseek2 16B IQ4_NL |   1 |   1 |    1 |    tg64@pp256 |     46.10 ± 0.14 |
| deepseek2 16B IQ4_NL |   1 |   1 |    1 |    tg64@pp512 |     45.87 ± 0.01 |
| deepseek2 16B IQ4_NL |   1 |   1 |    1 |   tg64@pp1024 |     45.77 ± 0.06 |
| deepseek2 16B IQ4_NL |   1 |   1 |    1 |   tg64@pp2048 |     45.37 ± 0.04 |
| deepseek2 16B IQ4_NL |   1 |   1 |    1 |   tg64@pp4096 |     44.60 ± 0.04 |
| deepseek2 16B IQ4_NL |   1 |   1 |    1 |   tg64@pp8192 |     43.10 ± 0.06 |

So, ~20% slower than standard attention. CUDA does not like MLA. I need to investigate why.

---

👤 **orca-zhang** commented the **2025-02-27** at **17:03:36**:<br>

I have observed the same phenomenon as you. After a single inference is completed, there is a lot of D2H copy work. Currently, I also use multiple parallel processing to "bypass" the solution you mentioned. I am not sure if we don't need to cache the results, can we directly abandon this part of the work? I would like to hear your opinion.

PS: I am actually a rookie who has only been exposed to the llama.cpp source code for a week.