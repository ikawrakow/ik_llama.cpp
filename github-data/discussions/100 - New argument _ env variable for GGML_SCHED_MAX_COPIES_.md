### 🗣️ [#100](https://github.com/ikawrakow/ik_llama.cpp/discussions/100) - New argument / env variable for GGML_SCHED_MAX_COPIES?

| **Author** | `Nexesenex` |
| :--- | :--- |
| **Created** | 2024-10-21 |
| **Updated** | 2024-10-21 |

---

#### Description

@ikawrakow, could you set up a CLI argument (or at least an env variable, it's much simpler I guess but I'm failing to do it right) to determine GGML_SCHED_MAX_COPIES without recompiling? It impacts VRAM occupation and performances, and it'd be great to set that up conveniently for benching and customized use.

---

#### 🗣️ Discussion

👤 **ikawrakow** replied the **2024-10-21** at **08:29:25**:<br>

I haven't looked into this at all. What is it good for?

---

👤 **Nexesenex** replied the **2024-10-21** at **09:36:22**:<br>

It's supposed to go faster inference on multi-GPU I guess. Mainline sets it at 4, I set it at 1, because I didn't notice much improvement back in the days, but I noticed more vram consumption and gpu load.