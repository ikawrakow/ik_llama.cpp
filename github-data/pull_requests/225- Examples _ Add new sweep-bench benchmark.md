### 🔀 [#225](https://github.com/ikawrakow/ik_llama.cpp/pull/225) -  Examples : Add new sweep-bench benchmark

| **Author** | `saood06` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-02-23 |
| **Updated** | 2025-04-26 |

---

#### Description

Port of https://github.com/ggml-org/llama.cpp/commit/9488fbf1e4334b8f189b38a7d224b8e6c1a7b22b

This is a good tool to benchmark with as requested by #223. 

As a very quick demo I generated this, just by running this ( ```./llama-sweep-bench -c 2048 -ub 512 -m  WizardLM-2-8x22B-IQ4_K_R4.gguf -ctk q8_KV -ctv q8_0 -fa --output-format jsonl ``` and then sweep-bench-plot.py with the output).

![performance_comparison_pp](https://github.com/user-attachments/assets/4a53b14d-d6a1-4e3a-99ac-5c3802c1e044)

![performance_comparison_tg](https://github.com/user-attachments/assets/b8b3cd9a-675d-415a-89b4-e334ed6ab825)

- Self-reported review complexity:
  - [X] Low
  - [ ] Medium
  - [ ] High

---

#### 💬 Conversation

👤 **ikawrakow** submitted a review the **2025-02-23** at **06:00:18**: ✅ `APPROVED`<br>

Thank you for this - can be very useful.

---

👤 **ubergarm** commented the **2025-04-26** at **18:01:12**:<br>

@saood06 thanks I'm a convert to `llama-sweep-bench`! It is indeed very useful.

I pushed a branch on my personal mainline llama.cpp fork just to use for testing performance across forks. I don't plan to open a PR to mainline, but just left it up there in case anyone else is using it. I'm guessing ik has something similar as we were comparing the new GLM-4 performance. 

Thanks!