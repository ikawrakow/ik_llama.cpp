### 🔀 [#489](https://github.com/ikawrakow/ik_llama.cpp/pull/489) - Adding top-n-sigma sampler

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-03 |
| **Updated** | 2025-06-03 |

---

#### Description

Given popular demand, adding top-n $\sigma$ sampler.

Set to off by default.

* Add to sampling chain using `--sampling-chain ...n...` or `--samplers ...top-n-sigma...`
* Set parameter using `--top-n-sigma value`

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-06-03** at **10:04:08**:<br>

Sure, will do.

What else do people want for sampling?

DRY?

---

👤 **saood06** commented the **2025-06-03** at **10:23:49**:<br>

>What else do people want for sampling?
>
> DRY?

That does seem to be more popular than the other two you just added (based on what I've seen reported in other places). Looking at the `main/README.md` of mainline that is the only one that is missing. (We also have TFS which was removed in mainline due to low usage and bugs).

I do personally think DRY is the best repeat penalty (of the ones that are publicly used), and so I would use it if I ever encounter looping again (but I wouldn't ever turn it on unless needed, since it does definitely affect quality if left on and there is no looping you want to avoid). I fortunately haven't seen looping in a while (and I think it is because newer models have this issue a lot less if at all)

---

👤 **saood06** submitted a review the **2025-06-03** at **10:38:23**: 💬 `COMMENTED`

---

👤 **saood06** submitted a review the **2025-06-03** at **10:41:21**: 💬 `COMMENTED`

---

👤 **Ph0rk0z** commented the **2025-06-03** at **11:33:23**:<br>

Yep, DRY is good. XTC threshold is usually .1 and below to get anything meaningful out of it. Not sure how that compares here. Super interesting to how this one is going to compare to the one I stole from mainline.

---

👤 **saood06** submitted a review the **2025-06-03** at **12:22:28**: 💬 `COMMENTED`