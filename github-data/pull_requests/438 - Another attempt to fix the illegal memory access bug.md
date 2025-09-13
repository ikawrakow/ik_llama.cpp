## 🔀 [Pull Request #438](https://github.com/ikawrakow/ik_llama.cpp/pull/438) - Another attempt to fix the illegal memory access bug

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Source Branch** | `ik/desperate_bug_fix_attempt` |
| **Target Branch** | `main` |
| **Created** | 2025-05-20 |
| **Updated** | 2025-05-23 |

---

## 📄 Description

Attempt to fix [#398](https://github.com/ikawrakow/ik_llama.cpp/issues/398), [#425](https://github.com/ikawrakow/ik_llama.cpp/issues/425)

My hopes are not very high, but it is better to try.
* More extensive check that we can really also fuse the `ffn_down` operation. The change does nothing for me, but I also never have a crash, so let's try that.
* Picked up a few changes from the mainline `llama.cpp` back-end. None of the changes seems very promising, but let's still try.

Please let me know if this fixes the illegal memory access