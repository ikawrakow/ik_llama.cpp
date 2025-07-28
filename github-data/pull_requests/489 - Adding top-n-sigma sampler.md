## 🔀 [Pull Request #489](https://github.com/ikawrakow/ik_llama.cpp/pull/489) - Adding top-n-sigma sampler

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/sampling-top-n-sigma` |
| **Target Branch** | `main` |
| **Created** | 2025-06-03 |
| **Updated** | 2025-06-03 |
| **Merged** | 2025-06-03 |

---

## 📄 Description

Given popular demand, adding top-n $\sigma$ sampler.

Set to off by default.

* Add to sampling chain using `--sampling-chain ...n...` or `--samplers ...top-n-sigma...`
* Set parameter using `--top-n-sigma value`

---

## 💬 Conversation

👤 **saood06** commented on **2025-06-03** at **09:48:28**

Since this PR is still open could the documentation for this and XTC be added to [examples/server/README.md](https://github.com/ikawrakow/ik_llama.cpp/blob/ccb265c01676aad9ae5860ba50e74e61dfcd1cf8/examples/server/README.md) and [examples/main/README.md](https://github.com/ikawrakow/ik_llama.cpp/blob/ccb265c01676aad9ae5860ba50e74e61dfcd1cf8/examples/main/README.md).

---

👤 **ikawrakow** commented on **2025-06-03** at **10:04:08**

Sure, will do.

What else do people want for sampling?

DRY?

---

👤 **saood06** commented on **2025-06-03** at **10:23:49**

>What else do people want for sampling?
>
> DRY?

That does seem to be more popular than the other two you just added (based on what I've seen reported in other places). Looking at the `main/README.md` of mainline that is the only one that is missing. (We also have TFS which was removed in mainline due to low usage and bugs).

I do personally think DRY is the best repeat penalty (of the ones that are publicly used), and so I would use it if I ever encounter looping again (but I wouldn't ever turn it on unless needed, since it does definitely affect quality if left on and there is no looping you want to avoid). I fortunately haven't seen looping in a while (and I think it is because newer models have this issue a lot less if at all)

---

👤 **saood06** started a conversation on `examples/main/README.md` on **2025-06-03** at **10:38:23**

Maybe add something like the following:

XTC probability sets how likely the XTC sampler is to engage.
XTC threshold is the lower-bound for what probability is needed for a token to be considered a "Top choice" and when engaged only the lowest probability top choice is kept. 

And maybe change ### XTC Sampling to ### XTC Sampling (Exclude Top Choices) since the description above refers to the full name

---

👤 **saood06** started a conversation on `examples/main/README.md` on **2025-06-03** at **10:41:21**

Maybe add something letting people know that increasing top-n-sigma results in more tokens being considered, while decreasing it makes less tokens be considered as not all users will be able to figure that out from the mathematical description you provided.

---

👤 **Ph0rk0z** commented on **2025-06-03** at **11:33:23**

Yep, DRY is good. XTC threshold is usually .1 and below to get anything meaningful out of it. Not sure how that compares here. Super interesting how this one is going to compare to the one I stole from mainline.

---

👤 **saood06** started a conversation on `examples/main/README.md` on **2025-06-03** at **12:22:28**

"conrolled" -> controlled

This isn't really accurate, as the lowest "top choice" is retained. As it is written it makes it seem like it removes all tokens with probability greater than the threshold.

Also I think the conditions for it to be turned off should be consistent instead of having the probability one in the beginning and the threshold one at the bottom

---

👤 **ikawrakow** commented on **2025-06-03** at **13:38:26**

Why don't you make your changes on top of the PR? Or, we merge the way it is and you make a new PR with better description.

---

👤 **saood06** commented on **2025-06-03** at **14:04:51**

> Or, we merge the way it is and you make a new PR with better description.

Sure. I can do that.