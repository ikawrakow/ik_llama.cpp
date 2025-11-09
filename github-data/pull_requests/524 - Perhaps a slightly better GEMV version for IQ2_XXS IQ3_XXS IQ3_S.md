## 🔀 [Pull Request #524](https://github.com/ikawrakow/ik_llama.cpp/pull/524) - Perhaps a slightly better GEMV version for IQ2_XXS, IQ3_XXS, IQ3_S

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/iq_gemv_tweaks` |
| **Target Branch** | `main` |
| **Created** | 2025-06-12 |
| **Updated** | 2025-06-13 |
| **Merged** | 2025-06-13 |

---

## 📄 Description

Closes [#523](https://github.com/ikawrakow/ik_llama.cpp/issues/523) 

@ciprianveg  @Ph0rk0z

Does this work better for you?

---

## 💬 Conversation

👤 **ciprianveg** commented on **2025-06-12** at **20:29:16**

> Ref [#523](https://github.com/ikawrakow/ik_llama.cpp/issues/523)
> 
> @ciprianveg @Ph0rk0z
> 
> Does this work better for you?

Yes, it does! :)

---

👤 **Ph0rk0z** commented on **2025-06-12** at **20:47:02**

I'm seeing 10s again so this one is a winner. Wish I knew why RTR isn't helpful since so many other people appear to have huge benefits from it at all batch sizes. Plus it buffs TG even more. I'm still using GCC 11, can it be related to that compiler thing?