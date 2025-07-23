### 🔀 [#180](https://github.com/ikawrakow/ik_llama.cpp/pull/180) - Deepseek MLA Optimizations

| **Author** | `saood06` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-01-29 |
| **Updated** | 2025-02-10 |

---

#### Description

Very direct port of https://github.com/ggerganov/llama.cpp/pull/11446

Tested working with Q4_K_S on dual socket Xeon E5-2690 v3, performance compared with llama.cpp below.
| model                          |       size |     params |         test |              llama.cpp t/s |              ik_llama.cpp t/s |
| ------------------------------ | ---------: | ---------: |  ------------: | ---------------: | ---------------: |
| deepseek2 671B Q4_K - Small    | 355.33 GiB |   672.05 B |          pp512 |      7.63  |      8.53 |
| deepseek2 671B Q4_K - Small    | 355.33 GiB |   672.05 B |         tg128 |     2.74  |      3.11  |

Tests in: https://github.com/ikawrakow/ik_llama.cpp/pull/180#issuecomment-2624940338

This PR also contains things I missed in my last PR in the convert_hf_to_gguf.py.

@ikawrakow 
Is there any chance to convert old imatrix files (such as [this](https://huggingface.co/mradermacher/DeepSeek-R1-i1-GGUF/blob/main/imatrix.dat)) to include the components you get from splitting kv_b included in it. I'm not sure how impactful missing them would be as right now it obviously prints "did not find weights for attn_k_b.weight/attn_v_b.weight". I do not have the capability to generate new imatrix.dat files, and it would be nice if it wasn't needed as it is quite resource intensive to do.


- Self-reported review complexity:
  - [X] Low
  - [ ] Medium
  - [ ] High

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-01-29** at **09:16:02**:<br>

Here is how much time is being spent in the various matrix multiplications in the attention part when processing a prompt of 8192 tokens:

| result tensor |  time (s) |
| ------------: | ---------: |
| kq                  |  4.116 |
| kqv                 |  2.372 |
| kqv_out             |  0.458 |
| kv                  |  0.253 |
| kv_pe_compresseed   |  0.219 |
| q                   |  0.687 |
| total               |  8.107 |

And here is with this PR:

| result tensor |  time (s) |
| ------------: | ---------: |
| kq_nope             |  8.343 |
| kq_pe               |  2.495 |
| kqv                 |  0.401 |
| kqv_compressed      |  7.120 |
| kqv_out             |  0.473 |
| kv_pe_compresseed   |  0.224 |
| q                   |  0.693 |
| q_nope2             |  0.240 |
| total               |  19.989 |

I.e., attention is 2.5X slower with the PR. In addition, I'm finding that on the main branch `0.114` seconds are spent in `GGML_OP_ADD` operations, and `0.194` seconds for `GGML_OP_CONT`. In this PR  `3.320` seconds go into `GGML_OP_ADD`, and `2.701` seconds into `GGML_OP_CONT` (basically making copies). For reference, total processing time is `27.73` seconds on main and `45.47` seconds with the PR.

Maybe this can be useful when trying to optimize.

---

👤 **saood06** commented the **2025-01-29** at **09:28:49**:<br>

>This hurts prompt processing (a.k.a prefill) speed very significantly.
>[...]
>I think we need to either try to understand why the attention part is so much slower when processing batches of tokens and fix it, or simply wait for @fairydreaming to fix their PR.

Changed to draft. PP does seem to have regressions, I'll have direct comparisons against old version soon, generating an iq4_k_r4 quant now (PP in main for me was 11.5 t/s for iq4_k and  9.8 t/s for iq4_k_r4 at pp512,  9.22 t/s at PP1024 for IQ4_K).

>Maybe this can be useful when trying to optimize.

Thank you for the op time breakdown. 

I was drawn in to this PR for the TG benefits, it should have also been a draft for the reason that it would mean GGUF's wouldn't be cross compatible, as this is also a draft in llama.cpp. I just want to have it here because it does optimize for a workload where TG dominates, and R1 as a reasoning model it often does.

---

👤 **ikawrakow** commented the **2025-01-29** at **09:33:33**:<br>

@saood06 Perhaps a good way to move forward is to add an additional architecture (`deepseek-mla` or similar), but keep the original  `deepseek2/3`. In this way, depending on use case, one can choose the improved TG speed after long prompts or the better PP speed when generating a few tokens after processing a long prompt.

---

👤 **saood06** commented the **2025-01-29** at **10:21:32**:<br>

>Perhaps a good way to move forward is to add an additional architecture (deepseek-mla or similar), but keep the original deepseek2/3. In this way, depending on use case, one can choose the improved TG speed after long prompts or the better PP speed when generating a few tokens after processing a long prompt.

I'll do that. I'll still leave it in a draft as I'm waiting to see how it progresses in llama.cpp, and for me to more thoroughly evaluate how it performs at long prompt lengths vs main.

---

👤 **ikawrakow** commented the **2025-01-29** at **11:40:16**:<br>

So, as far as I can tell, the attention implementation in this PR leads to ~3X more multiply-adds (madds) when performing matrix multiplications. For prompt processing here we need `2 x 512 x 16 x n_token^2` madds, whereas the original implementation requires  `(192 + 128) x 16 x n_token^2` madds. For TG, the PR still requires 3X more madds, namely `2 x 512 x n_prompt` madds here vs `(192 + 128) x 16 x n_prompt` on main. The only reason TG ends up being faster here is the shape of the tensors: On main it is 16 matrix multiplications each being `192 x n_prompt  * 192 x 1` (`K*Q`) or `n_prompt x 128 * n_prompt x 1` (`V*softmax(K*Q)`). I.e., we have 16 GEMVs, which are 100% memory bound on modern CPU's. In this PR the TG shapes are  `512 x n_prompt * 512 x 16` and `n_prompt x 512 * n_prompt x 16`, so real GEMMs with much higher FLOPs, so we end up needing less time despite doing more work. Hence, the way it is implemented, there is no way one can recover PP performance.

These figures are of course specific to the Deepseek2-Lite model. It may be different for a much larger model where rank-512 decomposition may really be "low-rank". It isn't for this model relative to the head sizes, number of heads, and hidden dimension.

---

👤 **fairydreaming** commented the **2025-01-29** at **12:49:35**:<br>

@ikawrakow I think applying the trick with "absorbing" matrices mentioned in the DeepSeek V2 paper shall fix this, I'm working on that.

---

👤 **ikawrakow** commented the **2025-01-29** at **13:14:33**:<br>

@fairydreaming 

Great!

Btw, I observe that `attn_kv_b.weight` is still present in the model. Is it needed, given that we now have `attn_k_b.weight` and `attn_v_b.weight` ?

---

👤 **fairydreaming** commented the **2025-01-30** at **11:23:08**:<br>

@ikawrakow Unfortunately the idea with speeding things up thanks to the matrix absorption is wrong: https://github.com/ggerganov/llama.cpp/pull/11446#issuecomment-2624177134

I'm not sure why they mentioned it in the DeepSeek paper.

Regarding other possible optimizations do you know how much work is needed to add support for multiplication of transposed matrices to ggml_mul_mat()? The problem is that I use kv cache for multiplication both directly and then in transposed form. I got around this problem by storing kv cache in both regular and transposed forms, but it doubles the amount of required memory.

---

👤 **fairydreaming** commented the **2025-01-30** at **12:39:37**:<br>

> @fairydreaming

> Out of curiosity, did you ever try this repository with your Epyc CPU?

Sure, I checked it a while ago (before the optimization work):

Regular llama.cpp:

```
$ ./build/bin/llama-bench --numa distribute -t 32 -m /mnt/md0/models/deepseek-v3-Q4_K_S.gguf
| model                          |       size |     params | backend    | threads |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
| deepseek2 671B Q4_K - Small    | 353.90 GiB |   671.03 B | CPU        |      32 |         pp512 |         26.08 ± 0.23 |
| deepseek2 671B Q4_K - Small    | 353.90 GiB |   671.03 B | CPU        |      32 |         tg128 |          9.57 ± 0.03 |
```

ik_llama.cpp:

```
$ ./llama-bench --numa distribute -t 32 -m /mnt/md0/models/deepseek-v3-Q4_K_S.gguf
| model                          |       size |     params | backend    | threads |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | ---------------: |
| deepseek2 671B Q4_K - Small    | 353.90 GiB |   671.03 B | CPU        |      32 |         pp512 |     49.47 ± 0.11 |
| deepseek2 671B Q4_K - Small    | 353.90 GiB |   671.03 B | CPU        |      32 |         tg128 |     10.01 ± 0.09 |
```

Generation was ~4.6% faster, while prompt processing was ~90% faster, impressive!

---

👤 **ikawrakow** commented the **2025-01-30** at **13:42:04**:<br>

10 t/s TG for Deepseek-R1 - wow! 

PP should be ~50% faster now for `Q4_K_S`. 

I'm playing with Deepseek-Lite and I'm finding that the CUDA performance is pretty bad - 3500 t/s for PP-512 and 142 t/s for TG-128 on an RTX-4080. This is for `IQ4_XS` fully offloaded to the GPU. On my Ryzen-7950X CPU I'm getting PP-512 = 525 t/s, TG-128 = 36 t/s. So, less than 7X slower for PP (normally the RTX-4080 is ~25X faster) and less than 4X slower for TG (despite the paltry 64 GB/s memory bandwidth for the Ryzen-7950X). So, I guess, your Epyc system wipes the floor with any GPU setup using partial GPU offload of Deepseek-R1.

---

👤 **saood06** commented the **2025-01-30** at **16:15:26**:<br>

I ran batched-bench at batch size 1 with TG at 32 at various PP to show PP performance and TG performance at different context lengths. Batched-bench numbers are noisy because they do not use repetitions like llama-bench and this model on this machine seems to have some variance, but all data is shown after dropping the cache's and running the model until it is fully in the page cache.

IQ4_K_R4 with this PR:

|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
|   128 |     32 |    1 |    160 |   22.569 |     5.67 |   10.237 |     3.13 |   32.806 |     4.88 |
|   256 |     32 |    1 |    288 |   38.648 |     6.62 |   10.699 |     2.99 |   49.347 |     5.84 |
|   512 |     32 |    1 |    544 |   76.447 |     6.70 |   10.793 |     2.96 |   87.240 |     6.24 |
|  1024 |     32 |    1 |   1056 |  144.100 |     7.11 |   10.788 |     2.97 |  154.888 |     6.82 |
|  2048 |     32 |    1 |   2080 |  312.306 |     6.56 |   12.624 |     2.53 |  324.930 |     6.40 |
|  4096 |     32 |    1 |   4128 |  745.760 |     5.49 |   12.929 |     2.48 |  758.688 |     5.44 |
|  8192 |     32 |    1 |   8224 | 2023.859 |     4.05 |   16.017 |     2.00 | 2039.877 |     4.03 |

IQ4_K_R4 on main:
|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
|   128 |     32 |    1 |    160 |   20.958 |     6.11 |   10.999 |     2.91 |   31.956 |     5.01 |
|   256 |     32 |    1 |    288 |   38.777 |     6.60 |   11.780 |     2.72 |   50.558 |     5.70 |
|   512 |     32 |    1 |    544 |   63.574 |     8.05 |   12.474 |     2.57 |   76.047 |     7.15 |
|  1024 |     32 |    1 |   1056 |  118.630 |     8.63 |   14.462 |     2.21 |  133.092 |     7.93 |
|  2048 |     32 |    1 |   2080 |  258.999 |     7.91 |   18.241 |     1.75 |  277.239 |     7.50 |
|  4096 |     32 |    1 |   4128 |  574.593 |     7.13 |   26.023 |     1.23 |  600.616 |     6.87 |
|  8192 |     32 |    1 |   8224 | 1391.722 |     5.89 |   43.056 |     0.74 | 1434.778 |     5.73 |


Looking at the 8K context results, PP does drop from 5.89 to 4.05, but TG jumps from 0.74 to 2.00. At q8_0 (results below) PP again drops 6.06 to 4.03, but TG benefits going from 0.99 to 1.94. I would test/run this model at even higher context, but I would either need a smaller quant or to use RPC (for reference the KV cache at n_ctx of 8224 is 40,233.55 MiB)

<details>
  <summary>Expand to see more runs with q8_0 and q6_0 K cache tested as well</summary>

  PR with q6_0 K cache:

|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
|   128 |     32 |    1 |    160 |   14.948 |     8.56 |   10.498 |     3.05 |   25.446 |     6.29 |
|   256 |     32 |    1 |    288 |   35.061 |     7.30 |   10.430 |     3.07 |   45.491 |     6.33 |
|   512 |     32 |    1 |    544 |   69.842 |     7.33 |   10.936 |     2.93 |   80.778 |     6.73 |
|  1024 |     32 |    1 |   1056 |  142.141 |     7.20 |   11.083 |     2.89 |  153.224 |     6.89 |
|  2048 |     32 |    1 |   2080 |  313.431 |     6.53 |   11.415 |     2.80 |  324.846 |     6.40 |
|  4096 |     32 |    1 |   4128 |  763.385 |     5.37 |   12.964 |     2.47 |  776.349 |     5.32 |
|  8192 |     32 |    1 |   8224 | 2076.578 |     3.94 |   16.371 |     1.95 | 2092.948 |     3.93 |


  PR with q8_0 K cache:

|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
|   128 |     32 |    1 |    160 |   15.804 |     8.10 |   10.288 |     3.11 |   26.092 |     6.13 |
|   256 |     32 |    1 |    288 |   34.806 |     7.35 |   10.436 |     3.07 |   45.242 |     6.37 |
|   512 |     32 |    1 |    544 |   69.839 |     7.33 |   10.597 |     3.02 |   80.437 |     6.76 |
|  1024 |     32 |    1 |   1056 |  141.519 |     7.24 |   10.909 |     2.93 |  152.428 |     6.93 |
|  2048 |     32 |    1 |   2080 |  310.669 |     6.59 |   11.430 |     2.80 |  322.099 |     6.46 |
|  4096 |     32 |    1 |   4128 |  751.935 |     5.45 |   12.970 |     2.47 |  764.905 |     5.40 |
|  8192 |     32 |    1 |   8224 | 2031.924 |     4.03 |   16.499 |     1.94 | 2048.424 |     4.01 |
  
  Second run of PR without K cache quantization:

|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
|   128 |     32 |    1 |    160 |   20.898 |     6.12 |   10.378 |     3.08 |   31.276 |     5.12 |
|   256 |     32 |    1 |    288 |   40.503 |     6.32 |   10.407 |     3.07 |   50.910 |     5.66 |
|   512 |     32 |    1 |    544 |   70.978 |     7.21 |   10.629 |     3.01 |   81.607 |     6.67 |
|  1024 |     32 |    1 |   1056 |  144.713 |     7.08 |   10.879 |     2.94 |  155.592 |     6.79 |
|  2048 |     32 |    1 |   2080 |  311.658 |     6.57 |   11.718 |     2.73 |  323.376 |     6.43 |
|  4096 |     32 |    1 |   4128 |  754.120 |     5.43 |   12.996 |     2.46 |  767.116 |     5.38 |
|  8192 |     32 |    1 |   8224 | 2037.022 |     4.02 |   16.437 |     1.95 | 2053.458 |     4.00 |

  main with q6_0 K cache:

|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
|   128 |     32 |    1 |    160 |   18.503 |     6.92 |   10.480 |     3.05 |   28.983 |     5.52 |
|   256 |     32 |    1 |    288 |   31.320 |     8.17 |   10.858 |     2.95 |   42.178 |     6.83 |
|   512 |     32 |    1 |    544 |   57.909 |     8.84 |   11.459 |     2.79 |   69.368 |     7.84 |
|  1024 |     32 |    1 |   1056 |  118.199 |     8.66 |   12.679 |     2.52 |  130.878 |     8.07 |
|  2048 |     32 |    1 |   2080 |  250.592 |     8.17 |   15.486 |     2.07 |  266.078 |     7.82 |
|  4096 |     32 |    1 |   4128 |  541.938 |     7.56 |   20.315 |     1.58 |  562.253 |     7.34 |
|  8192 |     32 |    1 |   8224 | 1353.169 |     6.05 |   30.144 |     1.06 | 1383.313 |     5.95 |




  main with q8_0 K cache:

|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
|   128 |     32 |    1 |    160 |   16.825 |     7.61 |   10.586 |     3.02 |   27.411 |     5.84 |
|   256 |     32 |    1 |    288 |   33.362 |     7.67 |   10.894 |     2.94 |   44.255 |     6.51 |
|   512 |     32 |    1 |    544 |   54.048 |     9.47 |   11.869 |     2.70 |   65.917 |     8.25 |
|  1024 |     32 |    1 |   1056 |  109.381 |     9.36 |   13.128 |     2.44 |  122.509 |     8.62 |
|  2048 |     32 |    1 |   2080 |  238.006 |     8.60 |   15.567 |     2.06 |  253.574 |     8.20 |
|  4096 |     32 |    1 |   4128 |  553.239 |     7.40 |   21.099 |     1.52 |  574.339 |     7.19 |
|  8192 |     32 |    1 |   8224 | 1351.138 |     6.06 |   32.240 |     0.99 | 1383.377 |     5.94 |



</details>

>I think one should make Flash Attention work with different K and V head sizes. 

If that happened it would also have the benefit of allowing V cache quantization (not sure why FA is needed for that), which this model could really benefit from in it's current implementation which uses the space of MHA. A proper MLA implementation would take up far less space. 

>I'm playing with Deepseek-Lite and I'm finding that the CUDA performance is pretty bad

Other people have reported poor performance even for the larger Deepseek models with TG at 10-14 t/s (although with an IQ1 based quant) even fully offloaded with datacenter GPU's, and around the same performance for a 192GB Mac.

>So, I guess, your Epyc system wipes the floor with any GPU setup using partial GPU offload of Deepseek-R1.

Partial offload is reported benefited by this: https://github.com/ggerganov/llama.cpp/pull/11397 and it is something I plan to test/use.

---

👤 **ikawrakow** commented the **2025-01-30** at **17:12:27**:<br>

> not sure why FA is needed for that

Because without FA `V` gets transposed, which would break the quantization blocks if `V` was quantized. It gets transposed because in that way the matrix multiplication with `softmax(K*Q^T)` is much faster. With FA, `V` is not transposed, which allows to quantize it. But, at least on the CPU, performance suffers quite a bit because of that. E.g., for a large context where all this matters, I see about 37% of the FA compute time to be spent for `K*Q^T`, about 10% for `softmax(K*Q^T)`, and the remaining 53% for `V*softmax(K*Q^T)`. I.e., the matrix multiplication with the not transposed `V` is ~50% slower compared to `K*Q^T`, although both multiplications require the same number of multiply-adds.

> Other people have reported poor performance even for the larger Deepseek models with TG at 10-14 t/s (although with an IQ1 based quant) even fully offloaded with datacenter GPU's, and around the same performance for a 192GB Mac.

I just made Deepseek-Lite also work on my Mac (M2-Max). I get TG-128 = 70 t/s on the CPU using `IQ4_NL_R4`, so basically half of an RTX-4080. Mainline `llama.cpp` gets 80 t/s on the M2-Max GPU (30 core version) and 63 t/s on the CPU for `IQ4_NL`. PP-512 is even more interesting: I get 292 t/s on the CPU, mainline `llama.cpp` manages  205 t/s on the CPU, but just 60 t/s on the GPU! So, there is some very serious bottleneck there, both on `CUDA` and `Metal`, for the Deepseek models.

---

👤 **fairydreaming** commented the **2025-02-01** at **08:09:20**:<br>

> So, as far as I can tell, the attention implementation in this PR leads to ~3X more multiply-adds (madds) when performing matrix multiplications. For prompt processing here we need `2 x 512 x 16 x n_token^2` madds, whereas the original implementation requires `(192 + 128) x 16 x n_token^2` madds. For TG, the PR still requires 3X more madds, namely `2 x 512 x n_prompt` madds here vs `(192 + 128) x 16 x n_prompt` on main. The only reason TG ends up being faster here is the shape of the tensors: On main it is 16 matrix multiplications each being `192 x n_prompt * 192 x 1` (`K*Q`) or `n_prompt x 128 * n_prompt x 1` (`V*softmax(K*Q)`). I.e., we have 16 GEMVs, which are 100% memory bound on modern CPU's. In this PR the TG shapes are `512 x n_prompt * 512 x 16` and `n_prompt x 512 * n_prompt x 16`, so real GEMMs with much higher FLOPs, so we end up needing less time despite doing more work. Hence, the way it is implemented, there is no way one can recover PP performance.

This is something that I kind of intuitively expected, I mean the whole point of DeepSeek MLA is to reduce KV cache memory size by storing the "compressed" latent representation of KV vectors, but we still have to perform additional calculations to "decompress" and use them to calculate attentions scores and attention output.

---

👤 **saood06** commented the **2025-02-09** at **15:02:19**:<br>

This is superseded by #188. Closing

---

👤 **jukofyork** commented the **2025-02-10** at **16:48:36**:<br>

@saood06

Just saw your linked post.

I see you have a slightly faster prompt processing speed, but what I'm confused about is why when I have everything on the GPU apart from the 3 sets of non-shared experts' tensors, why batch processing it's gaining anything hardly, eg:

- I can get 3.5 -5 tokens per second for token generation with careful NUMA placement and 30 threads of a 2-CPU system with ~78GB/s per node.
- I can only get 9-10 tokens per second when using a batch of 1024+ and it should be pulling each set of tensors from RAM to VRAM and doing the work for the 1024 tokens in parallel. IMO this shouild be showing speeds like what KTrasnformers is, but it's nothing like this and I'm near 100% sure there will be some glaring flaw in the way this is handled ***if*** I could actually profile the GGML stuff and see clearly WTF is going on to cause this!

---

👤 **jukofyork** commented the **2025-02-10** at **17:15:49**:<br>

> > I can only get 9-10 tokens per second for prompt processing when using a batch of 1024+ and it should be pulling each set of tensors from RAM to VRAM and doing the work for the 1024 tokens in parallel with 15x the memory bandwidth and 100x+ the compute. IMO this should be showing speeds like what KTrasnformers is, but it's nothing like this and I'm near 100% sure there will be some glaring flaw in the way this is handled if I could actually profile the GGML stuff and see clearly WTF is going on to cause this!
> 
> Can you try this fork, without MLA and this PR: #200 which adds FA support. This should be the fastest prompt processing you can do. Fairydreaming on his system with this fork without MLA and without FA and more optimizations reported 50 tok/s. [#180 (comment)](https://github.com/ikawrakow/ik_llama.cpp/pull/180#issuecomment-2624398627)
> 
> If you want to try MLA, just use the -mla flag, which will turn MLA on.

Thanks - I will do, but it will probably be a couple of days due to running another experiment.