## 🔀 [Pull Request #77](https://github.com/ikawrakow/ik_llama.cpp/pull/77) - Adding Q6_0

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/add_q60` |
| **Target Branch** | `main` |
| **Created** | 2024-10-02 |
| **Updated** | 2024-10-21 |
| **Merged** | 2024-10-02 |

---

## 📄 Description

Main motivation was to see how it performs for quantized kv-cache. Disappointingly, it is slightly worse than `Q8_0` for K-cache and `IQ4_NL` for V-cache (this `Q8_0`+`IQ4_NL` combo needs the exact same memory as `Q6_0` for both caches).

Nevertheless, with a block size of 32 it is the same as the other legacy quants, beets `Q5_0` and `Q5_1` with a significant margin for PPL (it is almost as good as `Q6_K`), performance on Metal is quite a bit better than `Q5_0` and `Q5_1`, etc. So that, once I did the work to implement and test, why not add it?

---

## 💬 Conversation

👤 **Nexesenex** commented on **2024-10-21** at **09:42:19**

You should test the combo -ctk q6_0 -ctv q5_0.
After a few PPL tests, it seems to be a keeper for me, to replace q5_1 - q5_0 and be quite close to the K q8_0 mixes in term of quality with much less VRAM occupation.

ON L3.1 8B Q5_K

PPL 512 211 chunks- K q5_1 / V q5_0 : 7.4175 - 46 MB

PPL 512 211 chunks - K q6_0 / V q5_0 : 7.3995 - 48 MB -> choice of the jury

PPL 512 211 chunks - K q8_0 / V iq4_nl : 7.4078 - 52MB