### 🔀 [#200](https://github.com/ikawrakow/ik_llama.cpp/pull/200) - DeepSeek FA support (CPU only)

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-02-10 |
| **Updated** | 2025-02-11 |

---

#### Description

This PR adds FA support for models where K and V head sizes are different, such as DeepSeek-R1 and DeepSeek-Lite. It only works with the standard attention mechanism, I have yet to look into FA with MLA.

We get a nice speedup for PP, increasing with context length, but TG is not faster. I want to play some more with it, but throwing it out there if someone wants to try. For sure this allows longer contexts to be processed as `-ctk q8_0 -ctv q8_0` seems perfectly adequate.

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-02-11** at **09:08:44**:<br>

So, I did get some minor FA speed improvements for TG, but I don't see what else one could do, so I'll merge it.

Here is a performance comparison between baseline (`Q8_0` K-cache, no FA, no MLA), MLA (`Q8_0` K-cache) and FA (`Q8_0` K and V cache) for DeepSeek-Lite running on a Ryzen-7950X CPU. Both graphs show the MLA and FA performance ratio to baseline.  

First graph is prompt processing speed. We see FA giving a ~40% performance boost at 16k tokens compared to baseline. MLA is 2X slower than baseline and 2.8X slower than FA at 16k tokens.

![ds2_pp](https://github.com/user-attachments/assets/426446de-5371-4305-8ac1-4da5e3501145)

The second graph is token generation speed (TG-64) after a prompt of a given length (i.e., TG speed as a function of the number of tokens in the KV cache). We do get some performance gains for very long prompts from FA (~10% at 16k tokens), but by far not as much as from MLA. MLA is 1.57X faster than baseline and 1.43X faster than FA at 16k tokens. 
  
![ds2_tg](https://github.com/user-attachments/assets/0b9fefcc-2f83-4b8f-8734-fd24c2104fe5)

---

👤 **ikawrakow** commented the **2025-02-11** at **10:33:34**:<br>

Recently I read somewhere that for the "common enterprise workflow" (whatever that means) the number of generated tokens is typically only about 10% of the prompt tokens. I don't know if that is true, but for the sake of argument, let's assume for a moment that it is. In that case the best way to measure overall model performance is to use `llama-bench -pg Npp,Ntg`, where `Ntg=0.1*Npp` is the number of generated tokens and `Npp` is the number of prompt tokens.  The following graph shows `PG` performance as a function of prompt length. The black symbols are mainline `llama.cpp build b9ab0a4d (4687)` (most current version as of today), the red symbols are for baseline `ik_llama.cpp` (no FA, no MLA), the green symbols are for MLA, and the blue symbols are for FA from this PR. The model is DeepSeek-Lite quantized with `IQ4_XS`. All use `Q8_0` for K cache, FA uses `Q8_0` also for V cache. All runs are on a Ryzen-7950X CPU. If we buy the claim that `Ntg ~ 0.1*Npp` in the "typical enterprise workflow", then there is no benefit from MLA over baseline, while FA is ~26% better for long prompts. Mainline `llama.cpp` is, as usual, slower. 1.45X for short prompts, increasing to 1.7X slower for prompts with 16k tokens.

![ds2_pg](https://github.com/user-attachments/assets/910f830d-31a6-4d66-8df9-b90e30b8f68d)