### [Pull Request #620](https://github.com/ikawrakow/ik_llama.cpp/pull/620) - Bump Windows max open files from 512 to 2048

| **Author** | `Thireus` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `patch-2` |
| **Target Branch** | `main` |
| **Created** | 2025-07-16 |
| **Updated** | 2025-07-17 |
| **Merged** | 2025-07-17 |

---

#### Description

Allows up to 2048 shards to be loaded on Windows builds, from the current default of 512. This change is specific to Windows, it instructs the Windows OS that the binary requires 2048 of max opened files. This is the equivalent to Linux's `ulimit -n`.

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [x] Low
  - [ ] Medium
  - [ ] High