### 🔀 [#533](https://github.com/ikawrakow/ik_llama.cpp/pull/533) - Much faster CPU prompt processing (part 2)

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-17 |
| **Updated** | 2025-06-18 |

---

#### Description

This PR is a follow up of #531 and applies the technique to `IQK` quants.

Here is a PP-512 performance comparison for LlaMA-3.1-8B-Instruct on a Ryzen-7950X CPU between the main branch and this PR:

| model            |       size |          test |     t/s (main)   |    t/s (PR)      |   Speedup |
| ---------------- | ---------: | ------------: | ---------------: | ---------------: | --------: |
| llama 8B IQ2_KS  |   2.05 GiB |         pp512 |    203.08 ± 0.39 |    372.48 ± 3.69 |  1.834    |    
| llama 8B IQ2_K   |   2.22 GiB |         pp512 |    195.04 ± 2.44 |    365.58 ± 4.25 |  1.874    |    
| llama 8B IQ3_K   |   3.21 GiB |         pp512 |    167.65 ± 0.53 |    354.90 ± 3.44 |  2.117    |    
| llama 8B IQ4_KS  |   3.98 GiB |         pp512 |    198.28 ± 0.57 |    362.81 ± 1.74 |  1.830    |    
| llama 8B IQ4_K   |   4.21 GiB |         pp512 |    177.08 ± 1.71 |    360.58 ± 1.96 |  2.036    |    
| llama 8B IQ5_KS  |   4.91 GiB |         pp512 |    182.40 ± 1.62 |    358.66 ± 3.39 |  1.966    |    
| llama 8B IQ5_K   |   5.14 GiB |         pp512 |    158.74 ± 0.87 |    354.68 ± 0.75 |  2.234    |    
| llama 8B IQ6_K   |   6.19 GiB |         pp512 |    147.07 ± 0.80 |    353.20 ± 3.48 |  2.402    | 

To put things into perspective, the fastest mainline `llama.cpp` quant on this CPU is `Q4_0`, and I get **170 t/s** with today's build (`build: 860a9e4ee (5688)`).

For a bit of history, when [PR 6414](https://github.com/ggml-org/llama.cpp/pull/6414) was added to `llama.cpp`, it received 92 :+1:, 32 :tada:, 34 :heart:, and 30 :rocket:. It only supported `Q4_0` and `Q8_0`, and speedup compared to the master branch at the time was in the range of 40-50%, for a PP-512 of **135 t/s** on the Ryzen-7950X CPU used for the above table.  There was a [blog post](https://justine.lol/matmul/) received with [great fanfare on HN](https://news.ycombinator.com/item?id=39890262).

---

#### 💬 Conversation

👤 **ubergarm** commented the **2025-06-17** at **16:45:36**:<br>

Thanks, this is huge. I feel like this will make ~70B dense models much better for hybrid inferencing on home rigs. Hope to try some quants soon!

---

👤 **Nexesenex** commented the **2025-06-17** at **18:31:50**:<br>

Very impressive, @ikawrakow!
All your recent commits motivates me to put more of IK_Llama on my Kobold.Cpp fork.
I already have overall twice its CPU PP perfs thanks to your amazing work, and I merged most of your quants, including the last Trellis!
Way to make an enthusiast happy!