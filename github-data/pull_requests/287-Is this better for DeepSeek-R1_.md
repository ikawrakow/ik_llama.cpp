### 🔀 [#287](https://github.com/ikawrakow/ik_llama.cpp/pull/287) - Is this better for DeepSeek-R1?

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-03-24 |
| **Updated** | 2025-04-03 |

---

#### Description

This PR implements MoE matrix multiplications on the CPU with a different strategy for distributing the work among the threads. I observe a very slight performance improvement for DeepSeek-Lite (~1%). I'm wondering if this could have more impact for DeepSeek-R1.

What is the difference?

In the implementation on the main branch all threads participate in each matrix multiplication for the involved experts, and the multiplications are performed one after the other.

In this PR we have MoE matrix multiplications be performed in parallel, with each multiplication being done by fewer threads. My thinking is that in this way we may better utilize the available memory bandwidth, as threads are accessing different tensors, which may be stored in different memory banks/be accessed via different memory controllers. On my Ryzen-7950X test system I'm maxing out the available memory bandwidth, so there cannot be much impact from this change. But on an EPYC or Xeon with 400+ GB/s available, the benchmark results we are getting for DeepSeek-R1 are far from saturating the memory bandwidth, so perhaps this PR could have a positive impact on TG performance.

To be most effective, the number of threads used should be a multiple of the number of activated experts (8 for DeepSeek-R1), so 8, 16, 24, 32, etc.

---

#### 💬 Conversation

👤 **saood06** commented the **2025-03-24** at **22:09:34**:<br>

I still haven't restarted my machine (in order to test hugepages, and mitigations being off) so when I have some time, I'll test this with sweep-bench and see how it compares to the results I last got.

---

👤 **ubergarm** commented the **2025-03-25** at **05:15:59**:<br>

Oh this looks interesting. Hopefully the 6980P frees up tomorrow to gives this branch a proper test given that rig has a lot of RAM bandwidth that seems under-utilized.

I gave this branch a very quick try on the 7965WX 24-Core with `-mla 2` and offloading some layers to GPU as usual. Not sure if this even applies to `-mla 2`.

Not super conclusive, but tg might be slightly improved with pp about the same in this test :point_down: 

<details>

<summary>Quick Test Results</summary>

## command
```bash
CUDA_VISIBLE_DEVICES="0," \
./build/bin/llama-bench \
    --model /mnt/raid/models/ubergarm/DeepSeek-R1-GGUF/DeepSeek-R1-IQ2_K_R4.gguf \
    -ctk q8_0 \
    -mla 2 -fa 1 \
    -amb 512 \
    -fmoe 1 \
    -p 512,4096 -n 0 \
    -gp 512,64 \
    -gp 4096,64 \
    -r 2 \
    --n-gpu-layers 63 \
    --override-tensor exps=CPU \
    --threads 24
```

## this experimental branch

Device 0: NVIDIA RTX A6000, compute capability 8.6, VMM: yes

| model                          |       size |     params | backend    | ngl | type_k | fa | mla |   amb | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -: | --: | ----: | ---: | ------------: | ---------------: |
| deepseek2 671B IQ2_K_R4 - 2.375 bpw | 226.00 GiB |   672.05 B | CUDA       |  63 |   q8_0 |  1 |   2 |   512 |    1 |         pp512 |    105.92 ± 0.50 |
| deepseek2 671B IQ2_K_R4 - 2.375 bpw | 226.00 GiB |   672.05 B | CUDA       |  63 |   q8_0 |  1 |   2 |   512 |    1 |        pp4096 |    100.30 ± 0.01 |
| deepseek2 671B IQ2_K_R4 - 2.375 bpw | 226.00 GiB |   672.05 B | CUDA       |  63 |   q8_0 |  1 |   2 |   512 |    1 |    tg64@pp512 |     10.70 ± 0.00 |
| deepseek2 671B IQ2_K_R4 - 2.375 bpw | 226.00 GiB |   672.05 B | CUDA       |  63 |   q8_0 |  1 |   2 |   512 |    1 |   tg64@pp4096 |     10.05 ± 0.03 |

build: be46f3ef (3608)

---

## main branch

Device 0: NVIDIA RTX A6000, compute capability 8.6, VMM: yes

| model                          |       size |     params | backend    | ngl | type_k | fa | mla |   amb | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -: | --: | ----: | ---: | ------------: | ---------------: |
| deepseek2 671B IQ2_K_R4 - 2.375 bpw | 226.00 GiB |   672.05 B | CUDA       |  63 |   q8_0 |  1 |   2 |   512 |    1 |         pp512 |    106.01 ± 0.50 |
| deepseek2 671B IQ2_K_R4 - 2.375 bpw | 226.00 GiB |   672.05 B | CUDA       |  63 |   q8_0 |  1 |   2 |   512 |    1 |        pp4096 |     99.68 ± 0.28 |
| deepseek2 671B IQ2_K_R4 - 2.375 bpw | 226.00 GiB |   672.05 B | CUDA       |  63 |   q8_0 |  1 |   2 |   512 |    1 |    tg64@pp512 |     10.15 ± 0.02 |
| deepseek2 671B IQ2_K_R4 - 2.375 bpw | 226.00 GiB |   672.05 B | CUDA       |  63 |   q8_0 |  1 |   2 |   512 |    1 |   tg64@pp4096 |      9.63 ± 0.01 |

build: f9307d79 (3607)

</details>

---

👤 **saood06** commented the **2025-03-25** at **09:08:27**:<br>

For me early results show regression, I dropped the caches and tested it, I'll let this run fully and post the graph but initial results below (build daa3b00c):

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |   58.226 |     8.79 |   44.387 |     2.88 |
|   512 |    128 |    512 |   58.371 |     8.77 |   49.335 |     2.59 |
|   512 |    128 |   1024 |   64.067 |     7.99 |   47.876 |     2.67 |
|   512 |    128 |   1536 |   66.120 |     7.74 |   49.035 |     2.61 |
|   512 |    128 |   2048 |   68.724 |     7.45 |   52.119 |     2.46 |
|   512 |    128 |   2560 |   70.648 |     7.25 |   51.798 |     2.47 |
|   512 |    128 |   3072 |   77.060 |     6.64 |   53.143 |     2.41 |
|   512 |    128 |   3584 |   78.354 |     6.53 |   55.939 |     2.29 |
|   512 |    128 |   4096 |   84.516 |     6.06 |   57.200 |     2.24 |
|   512 |    128 |   4608 |   88.221 |     5.80 |   56.947 |     2.25 |
|   512 |    128 |   5120 |   91.967 |     5.57 |   59.165 |     2.16 |
|   512 |    128 |   5632 |   93.136 |     5.50 |   59.594 |     2.15 |


For reference build d12f4a12 results below (truncated to same amount):
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

Oddly I also did a preliminary run before dropping the cache and oddly enough that performed better than after dropping but still worse than my previous one table below for reference (also build daa3b00c):

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |   50.972 |    10.04 |   41.870 |     3.06 |
|   512 |    128 |    512 |   56.608 |     9.04 |   44.729 |     2.86 |

Also while watching the CPU usage while it was loading the model into the cache it was different, it now had bursts of CPU activity then stretches around 3-4x as long with far lower CPU usage, the disk I/O was also fluctuating a lot more, but it did finish the load from cache in a similar time as expected for 48 threads.

---

👤 **saood06** commented the **2025-03-25** at **10:21:38**:<br>

Full results still show regression in TG:

![performance_comparison_tg](https://github.com/user-attachments/assets/61501c6b-1039-4e2e-8f04-1cc36c8eda05)

Although PP does improve a bit at contexts above ~5K:

![performance_comparison_pp](https://github.com/user-attachments/assets/c9a62ebc-e65f-4281-a0fa-d9978cc32f68)


Full results for this in table form:


|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |   58.226 |     8.79 |   44.387 |     2.88 |
|   512 |    128 |    512 |   58.371 |     8.77 |   49.335 |     2.59 |
|   512 |    128 |   1024 |   64.067 |     7.99 |   47.876 |     2.67 |
|   512 |    128 |   1536 |   66.120 |     7.74 |   49.035 |     2.61 |
|   512 |    128 |   2048 |   68.724 |     7.45 |   52.119 |     2.46 |
|   512 |    128 |   2560 |   70.648 |     7.25 |   51.798 |     2.47 |
|   512 |    128 |   3072 |   77.060 |     6.64 |   53.143 |     2.41 |
|   512 |    128 |   3584 |   78.354 |     6.53 |   55.939 |     2.29 |
|   512 |    128 |   4096 |   84.516 |     6.06 |   57.200 |     2.24 |
|   512 |    128 |   4608 |   88.221 |     5.80 |   56.947 |     2.25 |
|   512 |    128 |   5120 |   91.967 |     5.57 |   59.165 |     2.16 |
|   512 |    128 |   5632 |   93.136 |     5.50 |   59.594 |     2.15 |
|   512 |    128 |   6144 |   98.209 |     5.21 |   61.134 |     2.09 |
|   512 |    128 |   6656 |  102.257 |     5.01 |   63.292 |     2.02 |
|   512 |    128 |   7168 |  106.199 |     4.82 |   65.389 |     1.96 |
|   512 |    128 |   7680 |  106.290 |     4.82 |   65.561 |     1.95 |
|   512 |    128 |   8192 |  113.897 |     4.50 |   67.017 |     1.91 |
|   512 |    128 |   8704 |  117.766 |     4.35 |   67.738 |     1.89 |
|   512 |    128 |   9216 |  120.040 |     4.27 |   69.176 |     1.85 |
|   512 |    128 |   9728 |  124.898 |     4.10 |   72.930 |     1.76 |
|   512 |    128 |  10240 |  130.148 |     3.93 |   71.870 |     1.78 |
|   512 |    128 |  10752 |  133.752 |     3.83 |   73.079 |     1.75 |
|   512 |    128 |  11264 |  136.896 |     3.74 |   74.614 |     1.72 |
|   512 |    128 |  11776 |  141.029 |     3.63 |   76.383 |     1.68 |
|   512 |    128 |  12288 |  146.294 |     3.50 |   77.357 |     1.65 |
|   512 |    128 |  12800 |  147.800 |     3.46 |   78.471 |     1.63 |
|   512 |    128 |  13312 |  150.277 |     3.41 |   79.927 |     1.60 |
|   512 |    128 |  13824 |  153.251 |     3.34 |   81.628 |     1.57 |
|   512 |    128 |  14336 |  157.735 |     3.25 |   82.132 |     1.56 |
|   512 |    128 |  14848 |  160.234 |     3.20 |   84.146 |     1.52 |
|   512 |    128 |  15360 |  166.087 |     3.08 |   85.433 |     1.50 |
|   512 |    128 |  15872 |  167.285 |     3.06 |   88.591 |     1.44 |

---

👤 **ikawrakow** commented the **2025-03-25** at **11:14:42**:<br>

@saood06 Thanks for the results, but the tests are for batched processing. #287 is not supposed to influence batches in any way, it only does something different when we have exactly one token to process (as in TG). I suspect you end up having different results because of the warm up, which is TG. It seems in your case this leads to a less optimal distribution of model weights across memory banks, so you see a lower performance in your batched experiments. But with the small batches being used here, and a MoE model with so many experts, many of the experts will "see" just a single token in the batch, so I guess I could apply a similar optimization also there.

---

👤 **saood06** commented the **2025-03-25** at **12:06:03**:<br>

> @saood06 Thanks for the results, but the tests are for batched processing. #287 is not supposed to influence batches in any way, it only does something different when we have exactly one token to process (as in TG). I suspect you end up having different results because of the warm up, which is TG. It seems in your case this leads to a less optimal distribution of model weights across memory banks, so you see a lower performance in your batched experiments. But with the small batches being used here, and a MoE model with so many experts, many of the experts will "see" just a single token in the batch, so I guess I could apply a similar optimization also there.

I'm not testing batched performance, the TG values given for sweep-bench should be identical to the `-gp` option that you added in llama-bench.

The benefit is that it measures at intervals while growing and reusing the context, which makes it feasible for me to measure TG and PP performance and see how it changes at different context depths.

Doing the same with llama-bench's -gp would take much longer as my PP speed is so slow.

---

👤 **ikawrakow** commented the **2025-03-25** at **12:32:55**:<br>

> I'm not testing batched performance

So, not using `llama-batched-bench`? But then, if that wasn't batched inference, why would `N_KV` be so large?

---

👤 **saood06** commented the **2025-03-25** at **12:50:04**:<br>

> So, not using `llama-batched-bench`? 

No, all my recent benchmarks have been with the llama-sweep-bench.

>But then, if that wasn't batched inference, why would `N_KV` be so large?

The N_KV in the table is the equivalent to the first argument of gp. It is the depth at which you are testing TG/PP performance.

The PP and TG numbers is the equivalent to the second argument of gp. It is how many tokens of PP/TG you are doing at the given depth.

I used to use llama-batched-bench at batch size of 1 to get these numbers (and even told fairydreaming that `-gp` is redundant because that also gives you PP numbers), but llama-sweep-bench is more efficient as it grows the context as the test progresses instead of just starting from zero.

This benchmark does really reflect how llama-server feels for PP and TG across the tested context range.

---

👤 **saood06** commented the **2025-03-25** at **13:19:05**:<br>

@ikawrakow 

SORRY, I accidentally edited your comment instead of replying.

---

👤 **ikawrakow** commented the **2025-03-25** at **13:25:48**:<br>

OK, thanks. I'll wait for more detailed results from @ubergarm. If they are positive, I'll make it a compile time option (it is difficult to propagate a parameter to `ggml` CPU backend). If they are negative or inconclusive, I'll discard the PR.

---

👤 **saood06** commented the **2025-03-25** at **14:19:47**:<br>

I just pushed a fix to the [readme](https://github.com/ikawrakow/ik_llama.cpp/blob/98a264a2ea21761322847ac562f58d986ef6c512/examples/sweep-bench/README.md) so you can read it at the link. 

It goes over what the benchmark does and the definition of each header.

---

👤 **saood06** commented the **2025-03-25** at **14:27:36**:<br>

>(Squeezing this in while copying over the new deepseek-v3 q8_0_r8 for imatrix making given updated info over on that thread!)

How far did the BF16 one get overnight?

---

👤 **ubergarm** commented the **2025-03-25** at **20:17:57**:<br>

@saood06 

> I have been using (MLA-3, FA on, 48 threads, fmoe on)

> Looking at the results 64 cores with this PR is the best performing option, so both of your rigs do see a bump in speed while mine does not.

Yeah it is interesting, seems like for me there is a regression for non optimal number of threads though. Did you try a quick check of say 32 and 40 threads for a single setting? Just brainstorming...

Too many irons in the fire today lol, jumping back over to the thread on `imatrix` as that seems to actually be cooking now :crossed_fingers:

---

👤 **saood06** commented the **2025-03-25** at **20:26:52**:<br>

> Yeah it is interesting, seems like for me there is a regression for non optimal number of threads though. Did you try a quick check of say 32 and 40 threads for a single setting? Just brainstorming...
> 
> Too many irons in the fire today lol, jumping back over to the thread on `imatrix` as that seems to actually be cooking now 🤞

Not on this PR maybe that will help, as all previous testing showed bad results at 32. I don't feel like dropping my cache right now and testing that, but I might later. The behavior change during warmup does make me feel like the problem is deeper.

---

👤 **ubergarm** commented the **2025-03-26** at **00:10:52**:<br>

Haha, okay so I used `DeepSeek-V3-0324-IQ2_K_R4-bartowski-imat.gguf` to cook up some graphs and copy pasted my actual markdown `llama-bench` output into the `graph.py` and ran it without linting or anything and here is what we got.

It is complex, basically this PR is  7~12% better for pp and ~5% better for tg *only* when the number of threads is dialed in. Otherwise it is 3~20% worse than baseline main.

I would have to run more intervals near the peak e.g. 56 and 72 threads to confirm 64 is peak for this rig and config.

Gotta say I'm impressed `V3-0324` one-shotted that! Not perfect graphs, but it actually saved me some time! lol...

![llama-bench-testing-plot-pr287](https://github.com/user-attachments/assets/9e0ef00b-8910-4675-8f7a-f61eb80704f5)

The auto-generated code python:
<details>
<summary>plot.py</summary>

```bash
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

def parse_markdown_table(markdown_text):
    # Extract the table part from the markdown
    table_lines = []
    in_table = False
    for line in markdown_text.split('\n'):
        if line.startswith('|') and '----' not in line:
            table_lines.append(line)

    # Clean and parse the table
    rows = []
    for line in table_lines:
        # Remove leading/trailing | and strip whitespace
        cleaned = line.strip('|').strip()
        # Split by | and strip whitespace from each cell
        cells = [cell.strip() for cell in cleaned.split('|')]
        rows.append(cells)

    # Create DataFrame
    if not rows:
        return pd.DataFrame()

    headers = rows[0]
    data = rows[1:]
    df = pd.DataFrame(data, columns=headers)

    # Clean numeric columns
    numeric_cols = ['size', 'params', 'threads', 'type_k', 'fa', 'mla', 'amb', 'mmap', 'fmoe', 't/s']
    for col in numeric_cols:
        if col in df.columns:
            # Extract numeric part (handle GiB, B, etc.)
            if col in ['size', 'params']:
                df[col] = df[col].str.extract(r'([\d.]+)')[0].astype(float)
            elif col == 't/s':
                # Extract the numeric part before ± if present
                df[col] = df[col].str.extract(r'([\d.]+)')[0].astype(float)
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

# Sample data (you would replace this with your actual markdown)
pr_markdown = """## This PR branch `ik/deepseek_is_this_better@daa3b00`
| model                          |       size |     params | backend    | threads | type_k | fa | mla |   amb | mmap | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -----: | -: | --: | ----: | ---: | ---: | ------------: | ---------------: |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |      32 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |         pp512 |     56.67 ± 3.68 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |      32 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |        pp8192 |     39.15 ± 0.20 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |      32 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |       pp16384 |     28.63 ± 0.06 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |      32 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |    tg64@pp512 |      7.22 ± 0.00 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |      32 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |   tg64@pp8192 |      6.05 ± 0.03 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |      32 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |  tg64@pp16384 |      3.94 ± 0.01 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |      64 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |         pp512 |    105.04 ± 3.36 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |      64 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |        pp8192 |     69.45 ± 1.17 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |      64 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |       pp16384 |     51.00 ± 0.33 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |      64 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |    tg64@pp512 |      9.65 ± 0.00 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |      64 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |   tg64@pp8192 |      7.86 ± 0.00 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |      64 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |  tg64@pp16384 |      6.14 ± 0.11 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |      88 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |         pp512 |    112.03 ± 1.78 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |      88 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |        pp8192 |     70.51 ± 2.83 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |      88 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |       pp16384 |     55.87 ± 2.67 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |      88 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |    tg64@pp512 |      9.43 ± 0.00 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |      88 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |   tg64@pp8192 |      7.32 ± 0.01 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |      88 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |  tg64@pp16384 |      6.02 ± 0.03 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |     128 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |         pp512 |   127.07 ± 12.23 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |     128 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |        pp8192 |     76.89 ± 2.53 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |     128 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |       pp16384 |     55.11 ± 0.19 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |     128 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |    tg64@pp512 |      8.49 ± 0.02 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |     128 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |   tg64@pp8192 |      6.84 ± 0.19 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |     128 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |  tg64@pp16384 |      5.61 ± 0.14 |"""

baseline_markdown = """## Baseline `main@98a264a2`
| model                          |       size |     params | backend    | threads | type_k | fa | mla |   amb | mmap | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -----: | -: | --: | ----: | ---: | ---: | ------------: | ---------------: |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |      32 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |         pp512 |     62.14 ± 0.68 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |      32 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |        pp8192 |     41.03 ± 0.20 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |      32 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |       pp16384 |     29.36 ± 0.68 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |      32 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |    tg64@pp512 |      7.78 ± 0.01 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |      32 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |   tg64@pp8192 |      6.15 ± 0.01 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |      32 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |  tg64@pp16384 |      4.57 ± 0.03 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |      64 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |         pp512 |     96.11 ± 0.54 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |      64 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |        pp8192 |     64.43 ± 0.01 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |      64 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |       pp16384 |     45.32 ± 0.83 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |      64 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |    tg64@pp512 |      9.14 ± 0.03 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |      64 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |   tg64@pp8192 |      7.45 ± 0.02 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |      64 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |  tg64@pp16384 |      5.76 ± 0.02 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |      88 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |         pp512 |    116.98 ± 0.62 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |      88 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |        pp8192 |     81.51 ± 2.21 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |      88 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |       pp16384 |     58.54 ± 0.27 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |      88 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |    tg64@pp512 |      9.37 ± 0.00 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |      88 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |   tg64@pp8192 |      7.31 ± 0.06 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |      88 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |  tg64@pp16384 |      5.88 ± 0.19 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |     128 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |         pp512 |    139.62 ± 3.28 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |     128 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |        pp8192 |     95.89 ± 0.11 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |     128 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |       pp16384 |     69.04 ± 0.48 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |     128 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |    tg64@pp512 |      8.64 ± 0.05 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |     128 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |   tg64@pp8192 |      7.31 ± 0.05 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |     128 |   q8_0 |  1 |   3 |  1024 |    0 |    1 |  tg64@pp16384 |      5.97 ± 0.05 |"""

# Parse the tables
pr_df = parse_markdown_table(pr_markdown)
baseline_df = parse_markdown_table(baseline_markdown)

# Merge the data for comparison
comparison_df = pr_df.merge(baseline_df, on=['threads', 'test'], suffixes=('_pr', '_baseline'))

# Calculate performance difference
comparison_df['t/s_diff'] = comparison_df['t/s_pr'] - comparison_df['t/s_baseline']
comparison_df['t/s_pct_diff'] = (comparison_df['t/s_diff'] / comparison_df['t/s_baseline']) * 100

# Create plots
plt.figure(figsize=(15, 10))

# Plot 1: Performance comparison by test type and thread count
plt.subplot(2, 2, 1)
for test in comparison_df['test'].unique():
    test_data = comparison_df[comparison_df['test'] == test]
    plt.plot(test_data['threads'], test_data['t/s_pr'], 'o-', label=f'{test} (PR)')
    plt.plot(test_data['threads'], test_data['t/s_baseline'], 'x--', label=f'{test} (Baseline)')
plt.title('Performance Comparison by Test Type')
plt.xlabel('Thread Count')
plt.ylabel('Tokens per Second (t/s)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Plot 2: Performance difference (absolute)
plt.subplot(2, 2, 2)
for test in comparison_df['test'].unique():
    test_data = comparison_df[comparison_df['test'] == test]
    plt.plot(test_data['threads'], test_data['t/s_diff'], 'o-', label=test)
plt.title('Performance Difference (PR - Baseline)')
plt.xlabel('Thread Count')
plt.ylabel('Tokens per Second Difference (t/s)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Plot 3: Performance difference (percentage)
plt.subplot(2, 2, 3)
for test in comparison_df['test'].unique():
    test_data = comparison_df[comparison_df['test'] == test]
    plt.plot(test_data['threads'], test_data['t/s_pct_diff'], 'o-', label=test)
plt.title('Performance Difference Percentage')
plt.xlabel('Thread Count')
plt.ylabel('Percentage Difference (%)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Plot 4: Bar chart of average performance difference by test type
plt.subplot(2, 2, 4)
avg_diff = comparison_df.groupby('test')['t/s_diff'].mean()
avg_diff.plot(kind='bar')
plt.title('Average Performance Difference by Test Type')
plt.xlabel('Test Type')
plt.ylabel('Average Tokens per Second Difference (t/s)')
plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()
#plt.show()
plt.savefig('plot.png', bbox_inches='tight')

# Print summary statistics
print("Summary Statistics:")
print(f"Average performance difference: {comparison_df['t/s_diff'].mean():.2f} t/s")
print(f"Median performance difference: {comparison_df['t/s_diff'].median():.2f} t/s")
print(f"Maximum improvement: {comparison_df['t/s_diff'].max():.2f} t/s")
print(f"Maximum regression: {comparison_df['t/s_diff'].min():.2f} t/s")
```

<details>

---

👤 **saood06** commented the **2025-03-26** at **00:55:36**:<br>

> Haha, okay so I used `DeepSeek-V3-0324-IQ2_K_R4-bartowski-imat.gguf` to cook up some graphs and copy pasted my actual markdown `llama-bench` output into the `graph.py` and ran it without linting or anything and here is what we got.
> 
>[...]
> Gotta say I'm impressed `V3-0324` one-shotted that! Not perfect graphs, but it actually saved me some time! lol...

Ya that does seem nice. It might be clearer if you separate out the two test types and make two images.

>It is complex, basically this PR is 7~12% better for pp and ~5% better for tg only when the number of threads is dialed in. Otherwise it is 3~20% worse than baseline main.
>
>I would have to run more intervals near the peak e.g. 56 and 72 threads to confirm 64 is peak for this rig and config.

Sounds like a good time to try sweep-bench, it will give you a lot of data points much quicker than -gp 16384,64 and that way you can also see the curves and if there are any dips, just run ./llama-sweep with the settings you want to test (as mentioned before only llama-bench has special cli argument handling), and just set context to 16896.

Then just save the resulting markdown into a file and give it the filename of what you want it to say in the legend for that configuration.

---

👤 **ubergarm** commented the **2025-03-26** at **02:02:03**:<br>

> Sounds like a good time to try sweep-bench

Okay, I gave it a try, but possibly I didn't build the right version given I was testing this branch. It looks like I could just run `llama-sweep-bench` a few times varying threads to get the curves?

I guess I have a few questions:

1. `./build/bin/llama-sweep-bench --help` didn't show anything. I think it uses parameters out of common like `llama-server` and not like `llama-bench` as you mentioned above.
2. Does it output results as it goes to stdout or do I need to specify a file to save it to? I didn't find the output, but it seemed to run for a while and I saw CPU usage with 64 threads.
3. I'm not exactly sure how to compare its outputs to `llama-bench` `pp` and `tg` numbers, as I don't have a good conception of what varying `N_KV` exactly does. I read the README, but if I see an example maybe it would click in my brain.

I guess the first thing is I need to find where the output goes. Also the output log looks a bit wonky at the end like it does for me sometimes, not sure if that is due to piping stderr/stdout into tee or what...

<summary>

<details>Full llama-sweep-bench logs</details>

```bash
$ git branch
* ik/deepseek_is_this_better

$ ./build/bin/llama-sweep-bench --version
version: 3609 (daa3b00c)
built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu

$ numactl -N 0 -m 0 \
./build/bin/llama-sweep-bench \
    --no-mmap \
    --model /mnt/ai/models/unsloth/repack/DeepSeek-R1-Q4_K_R4.gguf \
    -ctk q8_0 \
    -mla 3 -fa \
    -amb 1024 \
    -fmoe \
    -c 16896 \
    -ub 512 \
    --threads 64 \
    --numa numactl 2>&1 | tee -a sweep-bench-test.log

llama_model_loader: loaded meta data with 45 key-value pairs and 1025 tensors from /mnt/ai/models/unsloth/repack/DeepSeek-R1-Q4_K_R4.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = DeepSeek R1 BF16
llama_model_loader: - kv   3:                       general.quantized_by str              = Unsloth
llama_model_loader: - kv   4:                         general.size_label str              = 256x20B
llama_model_loader: - kv   5:                           general.repo_url str              = https://huggingface.co/unsloth
llama_model_loader: - kv   6:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv   7:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv   8:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv   9:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  10:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv  11:          deepseek2.attention.head_count_kv u32              = 128
llama_model_loader: - kv  12:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  13: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  14:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  15:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  16:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  17:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  18:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  19:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  20:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  21:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  22:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  23:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  24:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  25:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  26:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  27:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  28:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  29:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  30: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  31: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  32:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  33:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  34:                      tokenizer.ggml.tokens arr[str,129280]  = ["
llama_model_loader: - kv  35:                  tokenizer.ggml.token_type arr[i32,129280]  = [3
llama_model_loader: - kv  36:                      tokenizer.ggml.merges arr[str,127741]  = ["
llama_model_loader: - kv  37:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  38:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  39:            tokenizer.ggml.padding_token_id u32              = 128815
llama_model_loader: - kv  40:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  41:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  42:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  43:               general.quantization_version u32              = 2
llama_model_loader: - kv  44:                          general.file_type u32              = 214
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q4_K:    1 tensors
llama_model_loader: - type q4_k_r4:  605 tensors
llama_model_loader: - type q6_k_r4:   58 tensors
llm_load_vocab: special tokens cache size = 819
llm_load_vocab: token to piece cache size = 0.8223 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = deepseek2
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 129280
llm_load_print_meta: n_merges         = 127741
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 163840
llm_load_print_meta: n_embd           = 7168
llm_load_print_meta: n_layer          = 61
llm_load_print_meta: n_head           = 128
llm_load_print_meta: n_head_kv        = 128
llm_load_print_meta: n_rot            = 64
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_embd_head_k    = 192
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: n_embd_k_gqa     = 24576
llm_load_print_meta: n_embd_v_gqa     = 16384
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 18432
llm_load_print_meta: n_expert         = 256
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = yarn
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 0.025
llm_load_print_meta: n_ctx_orig_yarn  = 4096
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = 671B
llm_load_print_meta: model ftype      = Q4_K_R4
llm_load_print_meta: model params     = 671.026 B
llm_load_print_meta: model size       = 376.650 GiB (4.822 BPW) 
llm_load_print_meta: repeating layers = 375.457 GiB (4.820 BPW, 669.173 B parameters)
llm_load_print_meta: general.name     = DeepSeek R1 BF16
llm_load_print_meta: BOS token        = 0 '<｜begin▁of▁sentence｜>'
llm_load_print_meta: EOS token        = 1 '<｜end▁of▁sentence｜>'
llm_load_print_meta: PAD token        = 128815 '<｜PAD▁TOKEN｜>'
llm_load_print_meta: LF token         = 131 'Ä'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_layer_dense_lead   = 3
llm_load_print_meta: n_lora_q             = 1536
llm_load_print_meta: n_lora_kv            = 512
llm_load_print_meta: n_ff_exp             = 2048
llm_load_print_meta: n_expert_shared      = 1
llm_load_print_meta: expert_weights_scale = 2.5
llm_load_print_meta: expert_weights_norm  = 1
llm_load_print_meta: expert_gating_func   = sigmoid
llm_load_print_meta: rope_yarn_log_mul    = 0.1000
llm_load_tensors: ggml ctx size =    0.42 MiB
llm_load_tensors:        CPU buffer size = 385689.63 MiB
....................................................................................................
============ llm_load_tensors: need to compute 61 wk_b tensors
Computed blk.0.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.1.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.2.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.3.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.4.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.5.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.6.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.7.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.8.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.9.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.10.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.11.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.12.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.13.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.14.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.15.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.16.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.17.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.18.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.19.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.20.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.21.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.22.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.23.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.24.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.25.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.26.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.27.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.28.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.29.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.30.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.31.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.32.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.33.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.34.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.35.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.36.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.37.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.38.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.39.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.40.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.41.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.42.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.43.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.44.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.45.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.46.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.47.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.48.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.49.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.50.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.51.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.52.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.53.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Collama_new_context_with_model: n_ctx      = 16896
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 3
llama_new_context_with_model: attn_max_b = 1024
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init: layer 0: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 1: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 2: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 3: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 4: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 5: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 6: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 7: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 8: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 9: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 10: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 11: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 12: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 13: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 14: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 15: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 16: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 17: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 18: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 19: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 20: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 21: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 22: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 23: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 24: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 25: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 26: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 27: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 28: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 29: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 30: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 31: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 32: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 33: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 34: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 35: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 36: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 37: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 38: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 39: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 40: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 41: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 42: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 43: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 44: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 45: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 46: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 47: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 48: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 49: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 50: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 51: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 52: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 53: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 54: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 55: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 56: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 57: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 58: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 59: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 60: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init:        CPU KV buffer size =   601.54 MiB
llama_new_context_with_model: KV self size  =  601.54 MiB, c^KV (q8_0):  601.54 MiB, kv^T: not used
llama_new_context_with_model:        CPU  output buffer size =     0.49 MiB
llama_new_context_with_model:        CPU compute buffer size =  1383.76 MiB
llama_new_context_with_model: graph nodes  = 5500
llama_new_context_with_model: graph splits = 1
mputed blk.54.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.55.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.56.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.57.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.58.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.59.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
Computed blk.60.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
```

</summary>

---

👤 **saood06** commented the **2025-03-26** at **02:27:43**:<br>

> > Sounds like a good time to try sweep-bench
> 
> Okay, I gave it a try, but possibly I didn't build the right version given I was testing this branch.

Yes this branch has the old version, you should merge in the new version. A lot of the instructions I will give below are specific to the new version. The old one is functional but is a lot more cumbersome to use.

>It looks like I could just run `llama-sweep-bench` a few times varying threads to get the curves?

Not quite, so for example here is one of my outputs

```
./llama-sweep-bench -m /mnt/sda/opensourcerelease_DeepSeek-R1-bf16/opensourcerelease_DeepSeek-R1-Bf16-256x21B-IQ4_K_R4.gguf -mla 3 -fa -fmoe --numa distribute -t 48 -c 16384
llama_model_loader: loaded meta data with 52 key-value pairs and 1147 tensors from /mnt/sda/opensourcerelease_DeepSeek-R1-bf16/opensourcerelease_DeepSeek-R1-Bf16-256x21B-IQ4_K_R4.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = opensourcerelease_DeepSeek R1 Bf16
llama_model_loader: - kv   3:                         general.size_label str              = 256x21B
llama_model_loader: - kv   4:                            general.license str              = mit
llama_model_loader: - kv   5:                   general.base_model.count u32              = 1
llama_model_loader: - kv   6:                  general.base_model.0.name str              = DeepSeek R1
llama_model_loader: - kv   7:          general.base_model.0.organization str              = Deepseek Ai
llama_model_loader: - kv   8:              general.base_model.0.repo_url str              = https://huggingface.co/deepseek-ai/De...
llama_model_loader: - kv   9:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv  10:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv  11:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv  12:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  13:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv  14:          deepseek2.attention.head_count_kv u32              = 128
llama_model_loader: - kv  15:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  16: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  17:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  18:                          general.file_type u32              = 340
llama_model_loader: - kv  19:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  20:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  21:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  22:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  23:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  24:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  25:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  26:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  27:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  28:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  29:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  30:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  31:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  32:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  33:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  34: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  35: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  36:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  37:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  38:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<▒...
llama_model_loader: - kv  39:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  40:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
llama_model_loader: - kv  41:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  42:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  43:            tokenizer.ggml.padding_token_id u32              = 1
llama_model_loader: - kv  44:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  45:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  46:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  47:               general.quantization_version u32              = 2
llama_model_loader: - kv  48:                      quantize.imatrix.file str              = /mnt/sda/mradermacher_DeepSeek-R1-GGU...
llama_model_loader: - kv  49:                   quantize.imatrix.dataset str              = imatrix-training-full-3
llama_model_loader: - kv  50:             quantize.imatrix.entries_count i32              = 720
llama_model_loader: - kv  51:              quantize.imatrix.chunks_count i32              = 315
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q5_0:   61 tensors
llama_model_loader: - type q5_K:   61 tensors
llama_model_loader: - type q6_K:    1 tensors
llama_model_loader: - type iq4_k:    1 tensors
llama_model_loader: - type iq4_k_r4:  662 tensors
llm_load_vocab: special tokens cache size = 818
llm_load_vocab: token to piece cache size = 0.8223 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = deepseek2
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 129280
llm_load_print_meta: n_merges         = 127741
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 163840
llm_load_print_meta: n_embd           = 7168
llm_load_print_meta: n_layer          = 61
llm_load_print_meta: n_head           = 128
llm_load_print_meta: n_head_kv        = 128
llm_load_print_meta: n_rot            = 64
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_embd_head_k    = 192
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: n_embd_k_gqa     = 24576
llm_load_print_meta: n_embd_v_gqa     = 16384
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 18432
llm_load_print_meta: n_expert         = 256
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = yarn
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 0.025
llm_load_print_meta: n_ctx_orig_yarn  = 4096
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = 671B
llm_load_print_meta: model ftype      = IQ4_K_R4 - 4.5 bpw
llm_load_print_meta: model params     = 672.050 B
llm_load_print_meta: model size       = 353.526 GiB (4.519 BPW)
llm_load_print_meta: repeating layers = 352.333 GiB (4.516 BPW, 670.196 B parameters)
llm_load_print_meta: general.name     = opensourcerelease_DeepSeek R1 Bf16
llm_load_print_meta: BOS token        = 0 '<｜begin▁of▁sentence｜>'
llm_load_print_meta: EOS token        = 1 '<｜end▁of▁sentence｜>'
llm_load_print_meta: PAD token        = 1 '<｜end▁of▁sentence｜>'
llm_load_print_meta: LF token         = 131 'Ä'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_layer_dense_lead   = 3
llm_load_print_meta: n_lora_q             = 1536
llm_load_print_meta: n_lora_kv            = 512
llm_load_print_meta: n_ff_exp             = 2048
llm_load_print_meta: n_expert_shared      = 1
llm_load_print_meta: expert_weights_scale = 2.5
llm_load_print_meta: expert_weights_norm  = 1
llm_load_print_meta: expert_gating_func   = sigmoid
llm_load_print_meta: rope_yarn_log_mul    = 0.1000
llm_load_tensors: ggml ctx size =    0.47 MiB
llm_load_tensors: offloading 0 repeating layers to GPU
llm_load_tensors: offloaded 0/62 layers to GPU
llm_load_tensors:        CPU buffer size = 362010.72 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 16384
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 3
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init: layer 0: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 1: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 2: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 3: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 4: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 5: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 6: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 7: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 8: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 9: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 10: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 11: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 12: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 13: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 14: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 15: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 16: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 17: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 18: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 19: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 20: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 21: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 22: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 23: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 24: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 25: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 26: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 27: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 28: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 29: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 30: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 31: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 32: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 33: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 34: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 35: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 36: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 37: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 38: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 39: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 40: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 41: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 42: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 43: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 44: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 45: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 46: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 47: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 48: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 49: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 50: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 51: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 52: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 53: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 54: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 55: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 56: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 57: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 58: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 59: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 60: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init:        CPU KV buffer size =  1098.00 MiB
llama_new_context_with_model: KV self size  = 1098.00 MiB, c^KV (f16): 1098.00 MiB, kv^T: not used
llama_new_context_with_model:        CPU  output buffer size =     0.49 MiB
llama_new_context_with_model:        CPU compute buffer size =  3258.01 MiB
llama_new_context_with_model: graph nodes  = 3487
llama_new_context_with_model: graph splits = 1

main: n_kv_max = 16384, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = -1, n_threads = 48, n_threads_batch = 48

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |   49.094 |    10.43 |   39.605 |     3.23 |
|   512 |    128 |    512 |   56.509 |     9.06 |   43.036 |     2.97 |
|   512 |    128 |   1024 |   63.248 |     8.10 |   44.641 |     2.87 |
|   512 |    128 |   1536 |   65.444 |     7.82 |   46.500 |     2.75 |
[...]
```

I would take the resulting table and write it to a file and call it result1, and result2 and so on.

Then I run `python sweep-bench-plot.py result1 result2 result3` and that would make `performance_comparison_tg.png` and `performance_comparison_pp.png`

> 
> I guess I have a few questions:
> 
>     1. `./build/bin/llama-sweep-bench --help` didn't show anything. I think it uses parameters out of common like `llama-server` and not like `llama-bench` as you mentioned above.

Yes, the -help is not very good, and the old version's print_usage also never printed to the screen only to the log file (I did not pay much attention to it printing to the screen when I originally ported as the old python only supported jsonl which wasn't really human readable anyway, and so it only going to a log file [which according to the [documentation](https://github.com/ikawrakow/ik_llama.cpp/blob/a22250df93fd833a6cb7f310b159ad1b54e4d582/common/log.h#L24) should be different for each pid, but for me it always overwrote the same log file], I switched them to LOG_TEE like most of the other examples, which goes both to the output and a log file in the fixed version.

>     2. Does it output results as it goes to stdout or do I need to specify a file to save it to? I didn't find the output, but it seemed to run for a while and I saw CPU usage with 64 threads.

The new one should, the old one didn't which I found annoying, it uses the LOG function which writes to llama.log (or a file like it)

3. I'm not exactly sure how to compare its outputs to `llama-bench` `pp` and `tg` numbers, as I don't have a good conception of what varying `N_KV` exactly does. I read the README, but if I see an example maybe it would click in my brain.

Think of N_KV as how deep in the context the you are measuring from from, and TG/PP is how many tokens. So in a row if the `N_KV` is 8192 and the `TG` is 128, the  `S_TG t/s`  resulting value is equivalent to `-gp 8192,128`.

> I guess the first thing is I need to find where the output goes. Also the output log looks a bit wonky at the end like it does for me sometimes, not sure if that is due to piping stderr/stdout into tee or what...

Sorry again, I forgot this branch had the old version, I should have warned you before reccomending, like I mentioned above it is only going to a log file in the old version, but you would have a far easier time just using the updated version where it also goes to the screen in the form of a markdown table and the script now makes graphs from the markdown output instead of the jsonl output.

---

👤 **ikawrakow** commented the **2025-03-26** at **07:24:39**:<br>

OK, this does not look like it is helping.

---

👤 **saood06** commented the **2025-03-29** at **07:34:32**:<br>

> OK, this does not look like it is helping.

It helped both of ubergarm's system under it's best configuration for TG, beating mainline in it's best configuration.

I'll test my system more thoroughly with this in different configurations later, I may be able to find a configuration that works on my system.

---

👤 **saood06** commented the **2025-04-03** at **05:36:15**:<br>

I tested at 24 threads this branch still loses to main (and main loses to main at 48 threads), but again it had the same odd behavior where this branch performed better when cache is warmed up with main than if cache is warmed up with it's own code.