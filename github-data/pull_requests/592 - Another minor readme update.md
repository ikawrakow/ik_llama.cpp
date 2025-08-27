## 🔀 [Pull Request #592](https://github.com/ikawrakow/ik_llama.cpp/pull/592) - Another minor readme update

| **Author** | `saood06` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Source Branch** | `s6/readme-minor2` |
| **Target Branch** | `main` |
| **Created** | 2025-07-08 |
| **Updated** | 2025-07-09 |

---

## 📄 Description

I think this looks cleaner.

It does remove mentions to: `IQ1_S_R4` [PR 492](https://github.com/ikawrakow/ik_llama.cpp/pull/492), `IQ1_M_R4` [PR 494](https://github.com/ikawrakow/ik_llama.cpp/pull/494).

They didn't belong in that section, but now I don't know where it would go at all (Features?).

---

## 💬 Conversation

👤 **ikawrakow** commented on **2025-07-09** at **12:00:32**

> They didn't belong in that section, but now I don't know where it would go at all (Features?).

They can go under "Quantization additions". `IQ1_M_R4` and `IQ1_S_R4` are distinct quantization types, not just repacked `IQ1_M` and `IQ1_S`.

Not sure if the tabular format for the new models works well. The table is quite squeezed already, and now Hunyuan has been added and dits.llm1 is pending. Do you know how you want to reformat/change to accommodate additional models?

---

👤 **saood06** commented on **2025-07-09** at **19:45:59**

> They can go under "Quantization additions". `IQ1_M_R4` and `IQ1_S_R4` are distinct quantization types, not just repacked `IQ1_M` and `IQ1_S`.

Added them (by making a Misc) section.

> Not sure if the tabular format for the new models works well. The table is quite squeezed already, and now Hunyuan has been added and dits.llm1 is pending. Do you know how you want to reformat/change to accommodate additional models?

I agree that it doesn't work well, but I wanted to try something to get rid of the block of text. I do think on top of just accommodating new models, this section might not belong in the "Latest News", since that makes it not mention all of the model supported inherited by mainline (and thus may confuse a user to thinking the listed models are the only models supported).