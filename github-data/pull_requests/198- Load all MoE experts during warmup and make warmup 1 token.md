### 🔀 [#198](https://github.com/ikawrakow/ik_llama.cpp/pull/198) -  Load all MoE experts during warmup and make warmup 1 token

| **Author** | `saood06` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-02-09 |
| **Updated** | 2025-02-10 |

---

#### Description

First commit is a port of: https://github.com/ggerganov/llama.cpp/pull/11571

The second commit is based on what fairydreaming has reported here https://github.com/ggerganov/llama.cpp/discussions/11733 and also unify's warmup to always be one token.

This allows warmup to actually warmup an MoE model as all experts are exercised.

---

#### 💬 Conversation

👤 **ikawrakow** submitted a review the **2025-02-10** at **07:12:56**: ✅ `APPROVED`<br>

LGTM, but it does nothing on the single socket computers I have currently available, so relying on the comments in the linked PR and issue that this really improves things on NUMA systems.

---

👤 **saood06** commented the **2025-02-10** at **14:52:48**:<br>

> LGTM, but it does nothing on the single socket computers I have currently available, so relying on the comments in the linked PR and issue that this really improves things on NUMA systems.

The first commit, should work on any system to help MoE loading (Deepseek is the most noticeable because of it's large size and expert count but it should help, but all MoE should benefit) . It is only the the second commit is designed to benefit NUMA systems.