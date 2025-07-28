## 🔀 [Pull Request #115](https://github.com/ikawrakow/ik_llama.cpp/pull/115) - MMQ for Q6_0

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/q60_mmq` |
| **Target Branch** | `main` |
| **Created** | 2024-11-20 |
| **Updated** | 2024-11-21 |
| **Merged** | 2024-11-21 |

---

## 📄 Description

Add MMQ kernel for `Q6_0`.

@Nexesenex

---

## 💬 Conversation

👤 **Nexesenex** commented on **2024-11-20** at **19:42:56**

Tested successfully on IK_LLama, PPL is 0.1% above Q6_K on a pure quant of Sheared Llama 2.7b.
Thanks IK. I'll play with the Qwen models in the next days.

Edit : testing right now a Rhys 78b (based on Qwen 2 72b), with Q5_K ftype, attn_v in Q6_K, and the whole ffdown in q6_0/5_1/5_0

Broadly, 5_1 has 0.2% ppl more than 5_0, and 5.0 0.05% ppl more than 6_0.
q5_1 underperforms q5_0 nowadays in most of my tests on various models. q6_0 replaces it adequately for the models incompatible with Q6_K, Qwen 2 is not "dense" enough to showcase a real benefit but nevertheless it's there.

Thanks again for your fast help, IK.