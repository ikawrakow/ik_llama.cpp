### [Pull Request #486](https://github.com/ikawrakow/ik_llama.cpp/pull/486) - Adding the XTC sampler

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Created** | 2025-06-03 |
| **Updated** | 2025-06-03 |
| **Merged** | 2025-06-03 |

---

#### Description

Given popular demand, here is the XTC sampler.

Same usage as in mainline:
* `x` to add with `--sampling-seq`
* `xtc` to add with `--samplers`
* `--xtc-probability` to set the probability
* `--xtc-threshold` to set the threshold

---

#### 💬 Conversation

👤 **saood06** submitted a review: 💬 `COMMENTED` on **2025-06-03** at **09:34:48**

_No content provided._

---

👤 **saood06** submitted a review: 💬 `COMMENTED` on **2025-06-03** at **09:35:50**

_No content provided._

---

👤 **ikawrakow** commented during a code review on `common/common.cpp` on **2025-06-03** at **09:39:08**

Oh, I forgot to update those, thanks!

As per mainline implementation, the disabling threshold is 0.5

---

👤 **ikawrakow** submitted a review: 💬 `COMMENTED` on **2025-06-03** at **09:39:08**

_No content provided._

---

👤 **saood06** commented during a code review on `common/common.cpp` on **2025-06-03** at **09:44:33**

>As per mainline implementation, the disabling threshold is 0.5

Yeah, I forgot and only remembered after commenting (after reading the rest of the commit). I was referencing mainline which makes the mistake of saying 1.0 here (but >0.5 in other places). Sorry.

---

👤 **saood06** submitted a review: 💬 `COMMENTED` on **2025-06-03** at **09:44:33**

_No content provided._