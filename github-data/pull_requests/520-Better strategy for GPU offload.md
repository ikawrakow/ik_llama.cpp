### 🔀 [#520](https://github.com/ikawrakow/ik_llama.cpp/pull/520) - Better strategy for GPU offload

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-11 |
| **Updated** | 2025-06-12 |

---

#### Description

In a hybrid GPU/CPU situation, the decision if to offload model weights residing in RAM to the GPU to perform matrix multiplications is a tricky business. On the master branch (and also in mainline `llama.cpp`) a simply heuristics is used: if the batch size is `>= 32` and the operation is supported, it is offloaded to the GPU. This heuristics comes from the experience with dense models (but even then, the correct decision will depend on the speed of the CPU, the GPU, and the PCI-E bandwidth).

This heuristics is definitely not meaningful for MoE models. In a MoE model with $N_{\rm tot}$ total routed experts and $N_A$ active experts, the matrix multiplication for each expert will contain, on average, $N_A/N_{\rm tot} N_b$ tokens, where $N_b$ is the batch (or rather, u-batch, size).  For a model such as DeepSeek-R1/V3 with $N_A = 8, N_{\rm tot} = 256$, a batch size of 32 will result in a single token per expert on average, so offloading gigabytes of data to the GPU does not make sense at all.

This PR adds the above consideration. MoE matrix multiplications will only be offloaded if

$$N_b \ge \frac{N_{\rm tot}}{N_A} N_{\rm min}$$

where $N_{\rm min}$ is the minimum batch size for dense models (hard-coded to 32 on the main branch). To allow for setup/model specific adjustment, a compile time option is added that allows to change $N_{\rm min}$ via
```
cmake -DGGML_CUDA_MIN_BATCH_OFFLOAD=new_value ...
```
The default value for `GGML_CUDA_MIN_BATCH_OFFLOAD` is left at 32. With this, MoE matrix multiplications will not get offloaded for DeepSeelk-R1/V3 unless the batch size is $\ge 1024$. For Qwen3-235B-A22B the minimumbtach size for offload becomes 512 tokens.

As a reminder, in addition to this PR in `ik_llama.cpp` GPU offload can be disabled via `-op 26,0,27,0,29,0`. 

As a quick example, the following tables contain `llama-bench` results for `PP-4096` using `IQ4_KS` quantized DeepSeek-Lite, with all experts left on the CPU.  

On the main branch we get this:

| model                |     params | n_ubatch | fa | mla | rtr | fmoe |          test |              t/s |
| -------------------- | ---------: | -------: | -: | --: | --: | ---: | ------------: | ---------------: |
| deepseek2 16B IQ4_KS |    15.76 B |      128 |  1 |   3 |   1 |    1 |        pp4096 |    344.75 ± 1.52 |
| deepseek2 16B IQ4_KS |    15.76 B |      256 |  1 |   3 |   1 |    1 |        pp4096 |   604.47 ± 10.39 |
| deepseek2 16B IQ4_KS |    15.76 B |      512 |  1 |   3 |   1 |    1 |        pp4096 |   973.29 ± 14.90 |
| deepseek2 16B IQ4_KS |    15.76 B |     1024 |  1 |   3 |   1 |    1 |        pp4096 |   1427.88 ± 9.06 |
| deepseek2 16B IQ4_KS |    15.76 B |     2048 |  1 |   3 |   1 |    1 |        pp4096 |  1804.31 ± 70.77 |
| deepseek2 16B IQ4_KS |    15.76 B |     4096 |  1 |   3 |   1 |    1 |        pp4096 | 1878.12 ± 139.24 |

With this PR we get this:

| model                |     params | n_ubatch | fa | mla | rtr | fmoe |          test |              t/s |
| -------------------- | ---------: | -------: | -: | --: | --: | ---: | ------------: | ---------------: |
| deepseek2 16B IQ4_KS |    15.76 B |      128 |  1 |   3 |   1 |    1 |        pp4096 |    723.34 ± 2.93 |
| deepseek2 16B IQ4_KS |    15.76 B |      256 |  1 |   3 |   1 |    1 |        pp4096 |    955.96 ± 3.76 |  
| deepseek2 16B IQ4_KS |    15.76 B |      512 |  1 |   3 |   1 |    1 |        pp4096 |   974.72 ± 12.17 |  
| deepseek2 16B IQ4_KS |    15.76 B |     1024 |  1 |   3 |   1 |    1 |        pp4096 |  1410.79 ± 20.59 |  
| deepseek2 16B IQ4_KS |    15.76 B |     2048 |  1 |   3 |   1 |    1 |        pp4096 |   1838.61 ± 2.46 |  
| deepseek2 16B IQ4_KS |    15.76 B |     4096 |  1 |   3 |   1 |    1 |        pp4096 |  2071.28 ± 37.94 |  

We see massively better performance for small u-batch` sizes (important for a more fluid interaction with the LLM as not all prompts are so long). For this model offload kicks in at `64/6*32 = 341` tokens, so for batch sizes of 512 and above the two results are the same.

If I change `GGML_CUDA_MIN_BATCH_OFFLOAD` to 64, min batch size for offload becomes 682 tokens, and we get this result:

| model                |     params | n_ubatch | fa | mla | rtr | fmoe |          test |              t/s |
| -------------------- | ---------: | -------: | -: | --: | --: | ---: | ------------: | ---------------: |
| deepseek2 16B IQ4_KS |    15.76 B |      128 |  1 |   3 |   1 |    1 |        pp4096 |    737.72 ± 7.27 |
| deepseek2 16B IQ4_KS |    15.76 B |      256 |  1 |   3 |   1 |    1 |        pp4096 |    968.12 ± 5.75 |  
| deepseek2 16B IQ4_KS |    15.76 B |      512 |  1 |   3 |   1 |    1 |        pp4096 |  1081.28 ± 28.45 |  
| deepseek2 16B IQ4_KS |    15.76 B |     1024 |  1 |   3 |   1 |    1 |        pp4096 |   1428.79 ± 3.19 |  
| deepseek2 16B IQ4_KS |    15.76 B |     2048 |  1 |   3 |   1 |    1 |        pp4096 | 1844.95 ±   9.59 |  
| deepseek2 16B IQ4_KS |    15.76 B |     4096 |  1 |   3 |   1 |    1 |        pp4096 |  2052.55 ± 78.42 |  

We see that for my setup, even batches of 512 tokens are better left on the CPU (for this specific quantization type).

Please play with this PR and let me know if it is useful to get merged.

---

#### 💬 Conversation

👤 **quasar-of-mikus** commented the **2025-06-11** at **20:40:59**:<br>

Looks quite good for setups like mine where PCIe bandwidth is low and prompt length is short.

128gb ddr4 3200 2ch
2x 3090 PCIe 3.0 x8 x8
DeepSeek-V3-0324-IQ1_S_R4.gguf 
Default value for DGGML_CUDA_MIN_BATCH_OFFLOAD=32,

For an existing context of 1400 + added prompt of 34 tokens, the difference was waiting a mere 3 seconds instead of 23 seconds until the first tokens:
Main: ~1.5t/s pp
PR: 9-10t/s pp


PR:
| model                          |       size |     params | backend    | ngl | threads | n_batch | n_ubatch | fa | mla |   amb | ts           | mmap | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | ------: | -------: | -: | --: | ----: | ------------ | ---: | ---: | ------------: | ---------------: |
| deepseek2 671B IQ1_S_R4 - 1.5 bpw | 130.20 GiB |   672.05 B | CUDA       | 999 |      18 |    4096 |     4096 |  1 |   3 |   512 | 23.00/23.00  |    0 |    1 |          pp16 |      7.81 ± 0.55 |
| deepseek2 671B IQ1_S_R4 - 1.5 bpw | 130.20 GiB |   672.05 B | CUDA       | 999 |      18 |    4096 |     4096 |  1 |   3 |   512 | 23.00/23.00  |    0 |    1 |          pp32 |     10.61 ± 0.34 |
| deepseek2 671B IQ1_S_R4 - 1.5 bpw | 130.20 GiB |   672.05 B | CUDA       | 999 |      18 |    4096 |     4096 |  1 |   3 |   512 | 23.00/23.00  |    0 |    1 |          pp64 |     13.31 ± 0.16 |
| deepseek2 671B IQ1_S_R4 - 1.5 bpw | 130.20 GiB |   672.05 B | CUDA       | 999 |      18 |    4096 |     4096 |  1 |   3 |   512 | 23.00/23.00  |    0 |    1 |         pp128 |     17.58 ± 0.20 |
| deepseek2 671B IQ1_S_R4 - 1.5 bpw | 130.20 GiB |   672.05 B | CUDA       | 999 |      18 |    4096 |     4096 |  1 |   3 |   512 | 23.00/23.00  |    0 |    1 |         pp256 |     19.66 ± 0.08 |
| deepseek2 671B IQ1_S_R4 - 1.5 bpw | 130.20 GiB |   672.05 B | CUDA       | 999 |      18 |    4096 |     4096 |  1 |   3 |   512 | 23.00/23.00  |    0 |    1 |         pp512 |     21.24 ± 0.10 |
| deepseek2 671B IQ1_S_R4 - 1.5 bpw | 130.20 GiB |   672.05 B | CUDA       | 999 |      18 |    4096 |     4096 |  1 |   3 |   512 | 23.00/23.00  |    0 |    1 |        pp1024 |     52.75 ± 0.37 |
| deepseek2 671B IQ1_S_R4 - 1.5 bpw | 130.20 GiB |   672.05 B | CUDA       | 999 |      18 |    4096 |     4096 |  1 |   3 |   512 | 23.00/23.00  |    0 |    1 |        pp2048 |     97.01 ± 0.59 |
| deepseek2 671B IQ1_S_R4 - 1.5 bpw | 130.20 GiB |   672.05 B | CUDA       | 999 |      18 |    4096 |     4096 |  1 |   3 |   512 | 23.00/23.00  |    0 |    1 |        pp4096 |    165.89 ± 0.63 |
build: cdcb324f (3743)


Main, note the very low speeds for pp16 to pp256:
| model                          |       size |     params | backend    | ngl | threads | n_batch | n_ubatch | fa | mla |   amb | ts           | mmap | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | ------: | -------: | -: | --: | ----: | ------------ | ---: | ---: | ------------: | ---------------: |
| deepseek2 671B IQ1_S_R4 - 1.5 bpw | 130.20 GiB |   672.05 B | CUDA       | 999 |      18 |    4096 |     4096 |  1 |   3 |   512 | 23.00/23.00  |    0 |    1 |          pp16 |      7.81 ± 0.40 |
| deepseek2 671B IQ1_S_R4 - 1.5 bpw | 130.20 GiB |   672.05 B | CUDA       | 999 |      18 |    4096 |     4096 |  1 |   3 |   512 | 23.00/23.00  |    0 |    1 |          pp32 |      1.89 ± 0.01 |
| deepseek2 671B IQ1_S_R4 - 1.5 bpw | 130.20 GiB |   672.05 B | CUDA       | 999 |      18 |    4096 |     4096 |  1 |   3 |   512 | 23.00/23.00  |    0 |    1 |          pp64 |      3.69 ± 0.01 |
| deepseek2 671B IQ1_S_R4 - 1.5 bpw | 130.20 GiB |   672.05 B | CUDA       | 999 |      18 |    4096 |     4096 |  1 |   3 |   512 | 23.00/23.00  |    0 |    1 |         pp128 |      7.44 ± 0.01 |
| deepseek2 671B IQ1_S_R4 - 1.5 bpw | 130.20 GiB |   672.05 B | CUDA       | 999 |      18 |    4096 |     4096 |  1 |   3 |   512 | 23.00/23.00  |    0 |    1 |         pp256 |     14.47 ± 0.03 |
| deepseek2 671B IQ1_S_R4 - 1.5 bpw | 130.20 GiB |   672.05 B | CUDA       | 999 |      18 |    4096 |     4096 |  1 |   3 |   512 | 23.00/23.00  |    0 |    1 |         pp512 |     27.94 ± 0.10 |
| deepseek2 671B IQ1_S_R4 - 1.5 bpw | 130.20 GiB |   672.05 B | CUDA       | 999 |      18 |    4096 |     4096 |  1 |   3 |   512 | 23.00/23.00  |    0 |    1 |        pp1024 |     52.96 ± 0.18 |
| deepseek2 671B IQ1_S_R4 - 1.5 bpw | 130.20 GiB |   672.05 B | CUDA       | 999 |      18 |    4096 |     4096 |  1 |   3 |   512 | 23.00/23.00  |    0 |    1 |        pp2048 |     97.27 ± 0.25 |
| deepseek2 671B IQ1_S_R4 - 1.5 bpw | 130.20 GiB |   672.05 B | CUDA       | 999 |      18 |    4096 |     4096 |  1 |   3 |   512 | 23.00/23.00  |    0 |    1 |        pp4096 |    166.23 ± 0.19 |
build: 3f54b497 (3742)

---

👤 **ikawrakow** commented the **2025-06-12** at **04:44:22**:<br>

Here the above data illustrated in a graph:

![batch_strategy](https://github.com/user-attachments/assets/d8acdbe1-8963-4db5-a8ed-d23db1c0e877)