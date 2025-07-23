### 🔀 [#178](https://github.com/ikawrakow/ik_llama.cpp/pull/178) - Interleave 8 rows (Q8_0, IQ4_XS)

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-01-26 |
| **Updated** | 2025-01-31 |

---

#### Description

One can get better performance on `AVX2/Zen4` by interleaving 8 instead of 4 rows. I did not do it earlier because in my previous attempts performance on `ARM` suffered significantly. But in this PR I found an `ARM_NEON` implementation for 8 interleaved rows for `Q8_0` and `IQ4_XS` that is not slower or is even slightly faster than 4 interleaved rows.

Run-time-repacking from `Q8_0/IQ4_XS` will of course work, but models quantized to `Q8_0_R4` or `IQ4_XS_R4` will stop working, so putting it out there for testing and feedback.

I did not rename the types to `_R8` yet but will in case this gets merged.

Below is a graph showing prompt processing (a.k.a. prefill) performance for LLaMA-3.1-8B quantized with `IQ4_XS` on a Ryzen-7950X CPU. The cyan symbols are the results with this PR. We now get over 300 t/s for prompts  less than 1000 tokens. 

![pp512_vs_ctx](https://github.com/user-attachments/assets/e532b929-894a-4187-9290-7a84b5286919)

@saood06 Can you test if this improves `IQ4_XS_R4` performance on your system?

---

#### 💬 Conversation

👤 **saood06** commented the **2025-01-26** at **17:03:11**:<br>

@ikawrakow 

Tested on my Xeon E5-2683 v4 machine via llama-bench.

| model                          |       size |     params | fa | rtr |          test |            master t/s |            PR t/s |
| ------------------------------ | ---------: |  ---------- |  -: | --: | ------------: | ---------------: | ---------------: |
| llama 70B IQ4_XS - 4.25 bpw    |  34.30 GiB |    68.98 B |  1 |   1 |         pp512 |      7.00 |      7.10 |


If you want me to test on my other machine (dual socket Xeon E5-2690 v3) or other models let me know. 

Also any chance you can sync the RPC code (mostly care about #11047 and to a lesser degree #9389 and #11424/#9296), if not I'll do it when I have some free time and submit a PR.

---

👤 **saood06** commented the **2025-01-27** at **13:06:04**:<br>

Testing the batch performance difference showing the peak numbers


IQ4_XS_R8:
|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
|   128 |    128 |   14 |   1920 |   18.944 |     6.76 |  272.880 |     6.57 |  291.824 |     6.58 |

IQ4_XS_R4:
|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
|   128 |    128 |   11 |   1536 |   19.367 |     6.61 |  220.288 |     6.39 |  239.655 |     6.41 |

---

👤 **ikawrakow** commented the **2025-01-27** at **13:28:46**:<br>

So, it looks like a small (~2%) improvement. OK to merge? (IIRC, you had this giant R1 model that will become useless after the merge if it is `IQ4_XS_R4`.

---

👤 **saood06** commented the **2025-01-27** at **14:12:11**:<br>

> So, it looks like a small (~2%) improvement.

Yes, it is an improvement, (there is an edge case where R4 was better and that was at batch size 4).

>OK to merge? (IIRC, you had this giant R1 model that will become useless after the merge if it is `IQ4_XS_R4`.

Yes, it is okay to merge. That model is an IQ4_K_R4 (and IQ4_K), not IQ4_XS, as I prefer your quants over the mainline ones. Which is why I didn't have comparison data for it to mainline.

On the note of the R1 quant this PR [llama.cpp/pull/11446](https://github.com/ggerganov/llama.cpp/pull/11446) will make me reconvert anyway, I want to use it and also it is easy to grab it now before the KV refactor it is waiting for to implement MLA KV cache. I was going to bring that up anyway in the Deepseek PR because it is a change to the the GGUF for Deepseek.

#11397 is also showing significant improvements to Deepseek.

---

👤 **ikawrakow** commented the **2025-01-27** at **15:41:40**:<br>

> On the note of R1, this PR 11446 will make me reconvert anyway

What is being measured in the graph in this PR? It says "Token generation rate", but what tool is being used?

---

👤 **fairydreaming** commented the **2025-01-27** at **19:42:36**:<br>

> > On the note of R1, this PR 11446 will make me reconvert anyway
> 
> What is being measured in the graph in this PR? It says "Token generation rate", but what tool is being used?

That would be my modified llama-bench from this PR: https://github.com/ggerganov/llama.cpp/pull/11126
It allows to measure token generation rate after processing a prompt of given size.

---

👤 **ikawrakow** commented the **2025-01-28** at **14:06:19**:<br>

@fairydreaming  Thanks for the clarification. 

I played a bit with your PR 11466. TG after a long prompt looks great compared to `llama.cpp`, but it seems this comes at the expense of a much reduced prompt processing speed? Here is what I get on my Ryzen-7950X

* **llama.cpp** 

| model                          |       size |     params | backend    | threads |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
| deepseek2 16B F16              |  29.26 GiB |    15.71 B | CPU        |      16 |         pp256 |        150.29 ± 0.31 |
| deepseek2 16B F16              |  29.26 GiB |    15.71 B | CPU        |      16 |         pp512 |        153.23 ± 0.13 |
| deepseek2 16B F16              |  29.26 GiB |    15.71 B | CPU        |      16 |        pp1024 |        149.27 ± 0.22 |
| deepseek2 16B F16              |  29.26 GiB |    15.71 B | CPU        |      16 |        pp4096 |        133.74 ± 0.20 |
| deepseek2 16B F16              |  29.26 GiB |    15.71 B | CPU        |      16 |        pp8192 |        117.74 ± 0.03 |

* **PR 11466**

| model                          |       size |     params | backend    | threads |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
| deepseek2 16B F16              |  29.37 GiB |    15.76 B | CPU        |      16 |         pp256 |        142.08 ± 0.27 |
| deepseek2 16B F16              |  29.37 GiB |    15.76 B | CPU        |      16 |         pp512 |        140.53 ± 0.03 |
| deepseek2 16B F16              |  29.37 GiB |    15.76 B | CPU        |      16 |        pp1024 |        133.17 ± 0.12 |
| deepseek2 16B F16              |  29.37 GiB |    15.76 B | CPU        |      16 |        pp4096 |        101.17 ± 0.10 |
| deepseek2 16B F16              |  29.37 GiB |    15.76 B | CPU        |      16 |        pp8192 |         77.08 ± 0.08 |

(I did not have the patience to wait for the 16k tokens benchmark to finish).

---

👤 **fairydreaming** commented the **2025-01-28** at **14:12:33**:<br>

@ikawrakow Yup, I noticed this. I'm planning to reorganize tensor dimensions for the prompt processing in the PR, hopefully this will fix the issue.

---

👤 **saood06** commented the **2025-01-29** at **09:03:52**:<br>

@fairydreaming 
> It allows to measure token generation rate after processing a prompt of given size.

Can't this be done already with batched-bench by setting a batch size of 1, and it has the benefit of showing PP speed as well.

>it helped, but only a bit (pp rate is 6-8% higher with these changes), it's still slower than the original implementation.

Can you push that change? For my use cases the TG benefits outweigh the loss in PP, I'll try looking into the performance as well.

---

👤 **fairydreaming** commented the **2025-01-29** at **10:09:22**:<br>

@saood06

> @fairydreaming
> 
> > It allows to measure token generation rate after processing a prompt of given size.
> 
> Can't this be done already with batched-bench by setting a batch size of 1, and it has the benefit of showing PP speed as well.

That is correct.

> > it helped, but only a bit (pp rate is 6-8% higher with these changes), it's still slower than the original implementation.
> 
> Can you push that change? For my use cases the TG benefits outweigh the loss in PP, I'll try looking into the performance as well.

Pushed.

---

👤 **saood06** commented the **2025-01-30** at **19:32:55**:<br>

@ikawrakow 
>I did not rename the types to _R8 yet but will in case this gets merged.

---

👤 **ikawrakow** commented the **2025-01-31** at **06:31:03**:<br>

Will do when I come back from FOSDEM.