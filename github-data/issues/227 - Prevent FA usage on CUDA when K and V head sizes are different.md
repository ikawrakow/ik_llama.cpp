### 📝 [#227](https://github.com/ikawrakow/ik_llama.cpp/issues/227) - Prevent FA usage on CUDA when K and V head sizes are different

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-02-23 |
| **Updated** | 2025-03-20 |

---

#### Description

CUDA FA is not implemented when K and V head sizes are different (e.g., DeepSeekV3/R1/Lite), and leads to random error messages being displayed to the user or garbage output. Since the user may not know this detail, it is better to prevent CUDA FA usage in such cases.

---

#### 💬 Conversation

👤 **saood06** commented the **2025-03-20** at **01:41:17**:<br>

Can this be closed now, I think https://github.com/ikawrakow/ik_llama.cpp/pull/268 handled the only case left where CUDA was not supported.

---

👤 **ikawrakow** commented the **2025-03-20** at **16:33:31**:<br>

Yes, closing it.