### 🔀 [#443](https://github.com/ikawrakow/ik_llama.cpp/pull/443) - Streamline a bit the quant strategies

| **Author** | `Nexesenex` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-22 |
| **Updated** | 2025-05-22 |

---

#### Description

Unlike last time..

No change over the existing patterns, except for the bump for attn_k and attn_v for the models with 4 and 6 experts (several frankensteins seen on HF, and which also use GQA).
The rest is applying the existing patterns to the new IQ_K quants.
Also, a Q8_0 for attn_q slipped into the MOEs 8 experts rule, I removed it, because that tensor is much bigger than attn_k or attn_v.

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [x] Low
  - [ ] Medium
  - [ ] High

---

#### 💬 Conversation

👤 **ikawrakow** commented during a code review the **2025-05-22** at **06:46:59** on `src/llama.cpp`:<br>

Why do we want to limit to `<= 8` experts?

---

👤 **ikawrakow** commented during a code review the **2025-05-22** at **06:48:18** on `src/llama.cpp`:<br>

Why limit to `<= 8` experts?

---

👤 **ikawrakow** commented during a code review the **2025-05-22** at **06:54:53** on `src/llama.cpp`:<br>

So, I see you added the condition for `Q5_K_S` just above but I have forgotten why we want to have it. Can you remind me? I was wondering not too long ago why a model quantized with `Q5_K_S` ended up having less the 5.5 bpw (but didn't check). Why is the decision to reduce the number of bits dependent on the vocabulary size?

---

👤 **ikawrakow** commented during a code review the **2025-05-22** at **06:55:55** on `src/llama.cpp`:<br>

`<= 8`?

---

👤 **ikawrakow** submitted a review the **2025-05-22** at **06:58:25**: 💬 `COMMENTED`<br>

Looks OK apart from the `<= 8` condition for MoE models. I don't think it is needed.

This may make it more convenient for some people, but I basically just use `--custom-q` these days.

---

👤 **Nexesenex** commented during a code review the **2025-05-22** at **13:46:33** on `src/llama.cpp`:<br>

Oh, I just not wanted to step on bigger MOEs because I didn't test any.
I left that to your discretion.

---

👤 **Nexesenex** submitted a review the **2025-05-22** at **13:46:34**: 💬 `COMMENTED`

---

👤 **Nexesenex** submitted a review the **2025-05-22** at **13:48:21**: 💬 `COMMENTED`

---

👤 **Nexesenex** commented during a code review the **2025-05-22** at **13:48:21** on `src/llama.cpp`:<br>

I just did not want to step on bigger MOEs because I didn't test any.
I left that to your discretion. But ofc if it's fine with your we can remove that second condition.

---

👤 **Nexesenex** submitted a review the **2025-05-22** at **13:54:19**: 💬 `COMMENTED`

---

👤 **Nexesenex** commented during a code review the **2025-05-22** at **13:54:19** on `src/llama.cpp`:<br>

I added this back then because attn_q endures very well a smaller quant on Llama 3 models, with no perplexity bump or even a drop around 0.005 on L3 (and Also Mistral 123b models).
I also observed this with IQ4_XS -> IQ3_S for attn_q.
I take benefit of this to bump attn_v instead on L3, which is very sensitive to it.
At the time, you agreed with the principle.

---

👤 **Nexesenex** submitted a review the **2025-05-22** at **13:54:45**: 💬 `COMMENTED`

---

👤 **Nexesenex** commented during a code review the **2025-05-22** at **13:54:45** on `src/llama.cpp`:<br>

Ok, I will remove this <= part!

---

👤 **ikawrakow** submitted a review the **2025-05-22** at **15:04:41**: ✅ `APPROVED`