### 🐛 [#202](https://github.com/ikawrakow/ik_llama.cpp/pull/202) - Fix imatrix overprotectiveness

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-02-11 |
| **Updated** | 2025-02-12 |

---

#### Description

I hear reports that people are having trouble creating imatrix data for models with many experts (e.g., DeepSeek-R1, Arctic). For such models it may be very hard to activate all experts in all layers, which it turns out leads to the data for **the entire** tensor containing experts with missing data to be not stored in the imatrix file. Which then prevents usage of the imatrix data for low-bit quantization of such models.

It wasn't like this when I added the imatrix to `llama.cpp`, but it turns out the protection police has been at work and has added these checks, which I then inherited when syncing with upstream. Thanks to @saood06 for making me aware of this unfortunate situation.

This PR reduces the powers of the protection police. If a tensor is found that has partial contributions to the imatrix data, instead of simply skipping it, we now
* Check if it is a tensor containing experts
* If so, count how many experts are missing data
* If less than 5% of the experts are missing data, we
   - Warn the user, but still store the data in the imatrix file
   - Set the imatrix weights to 1 for the experts missing data

The rationale behind this approach is that if an expert was never activated after processing a significant amount of calibration data, this expert cannot be very important, so we can afford to quantize it with low bpw quants even without guidance on the importance of columns of this expert.

Strictly speaking it would be better to leave the zeros in the imatrix data of experts that have never been activated. But this would require to go and add proper protection against all-zeros imatrices, along with the appropriate corrective action, for all quants, and not just for `IQ1_S_R4` as I did in #191. So, for now we go with same-importance columns for never activated experts.

---

#### 💬 Conversation

👤 **saood06** commented the **2025-02-11** at **17:09:17**:<br>

>for the entire tensor containing experts

Not entirely related to this, but do you know why GGUF stores all the experts together? (I just checked the initial PR in mainline for an MoE and no rationale was given for this).

I plan to port over code that lets you override where certain tensors are allocated which allows you to store non-shared experts on RAM and all else on VRAM. If the experts were not consolidated into one large tensor this could easily allow for expert parallelism which would benefit NUMA systems.

---

👤 **ikawrakow** commented the **2025-02-11** at **17:16:38**:<br>

> but do you know why GGUF stores all the experts together?

No I don't. The initial MoE implementation was not like that, and then it got changed. I have kept the ability to use the original version in my fork (so I don't need to re-download MoE models that were created before the change).