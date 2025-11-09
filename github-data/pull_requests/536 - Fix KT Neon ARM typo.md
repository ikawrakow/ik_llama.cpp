## 🔀 [Pull Request #536](https://github.com/ikawrakow/ik_llama.cpp/pull/536) - Fix KT Neon / ARM typo

| **Author** | `louiehelm` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `main` |
| **Target Branch** | `main` |
| **Created** | 2025-06-18 |
| **Updated** | 2025-06-18 |
| **Merged** | 2025-06-18 |

---

## 📄 Description

Removes errant ";" in front of 0xCBAC1FED in non-x86 code

```
error: expected primary-expression before ';' token
     constexpr static uint32_t ka = ;0xCBAC1FED;
                                    ^
error: expected unqualified-id before numeric constant
     constexpr static uint32_t ka = ;0xCBAC1FED;
                                    ^
```

---

## 💬 Conversation

👤 **ikawrakow** approved this pull request ✅ on **2025-06-18** at **16:53:19**

---

👤 **ikawrakow** commented on **2025-06-18** at **16:54:57**

Thank you for this. Are you using an ARM CPU? I haven't checked if it works there.

---

👤 **louiehelm** commented on **2025-06-18** at **17:05:31**

No I don't have ARM CPU unfortunately. Just cross-compiled to see if all code paths would build then fixed that line so it could at least compile. Ready for someone who actually has ARM to test it now.