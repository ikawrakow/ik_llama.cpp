## 🔀 [Pull Request #542](https://github.com/ikawrakow/ik_llama.cpp/pull/542) - Fix NEON build

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/fix_neon_build` |
| **Target Branch** | `main` |
| **Created** | 2025-06-19 |
| **Updated** | 2025-06-19 |
| **Merged** | 2025-06-19 |

---

## 📄 Description

I did not pay attention to the `ARM_NEON` build with the recent PP performance improvement PRs, so now the main branch does not even build. This PR fixes that (but nothing will be working).