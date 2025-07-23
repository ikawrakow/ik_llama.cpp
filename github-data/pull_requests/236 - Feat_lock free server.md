### ✨ [#236](https://github.com/ikawrakow/ik_llama.cpp/pull/236) - Feat/lock free server

| **Author** | `orca-zhang` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2025-02-27 |
| **Updated** | 2025-03-19 |

---

#### Description

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [x] Low
  - [ ] Medium
  - [ ] High

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-02-27** at **11:43:27**:<br>

Thank you for this PR.

LGTM, but as I never use the server and I'm not familiar with the code, I have assigned @saood06 to review it.

---

👤 **orca-zhang** commented the **2025-02-27** at **17:02:24**:<br>

Hi Ikawrakow,

Please accept my apologies for the accidental PR submission during my preliminary testing phase. I'm currently conducting informal experiments **without rigorous benchmarking**, and cannot yet confirm the actual utility of these code changes.

During my evaluation of DeepSeek-R1-671B performance, I observed occasionnally perceptible latency in Time-to-First-Token (TTFT) measurements within the llama.cpp implementation. This preliminary observation coincided with identifying a potentially prolonged lock duration in the execution flow while reviewing the codebase and profiling results which are early-stage findings requiring further validation.

Thank you for your continued dedication to maintaining this exceptional codebase. I'm consistently impressed by the engineering rigor demonstrated in this project.

---

👤 **saood06** commented during a code review the **2025-02-27** at **19:55:22** on `examples/server/atomic_hash_map.hpp`:<br>

This is Apache, while this project is MIT.

---

👤 **saood06** submitted a review the **2025-02-27** at **19:55:23**: 💬 `COMMENTED`

---

👤 **saood06** commented the **2025-02-27** at **19:57:11**:<br>

>Please accept my apologies for the accidental PR submission during my preliminary testing phase. I'm currently conducting informal experiments without rigorous benchmarking, and cannot yet confirm the actual utility of these code changes.

You can set this to be a draft PR until it is ready to be reviewed, but for now I did leave a comment on the license mismatch from some of the code in your PR.