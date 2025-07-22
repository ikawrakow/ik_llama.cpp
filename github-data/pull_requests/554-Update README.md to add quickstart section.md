### 🔀 [#554](https://github.com/ikawrakow/ik_llama.cpp/pull/554) - Update README.md to add quickstart section

| **Author** | `jwinpbe` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2025-06-25 |
| **Updated** | 2025-06-25 |

---

#### Description

add quickstart section using ubergarm's discussion post. Scrolling to the discussion every time I want to remember how to build the damn thing is a minor inconvienience so this pull request is both useful and self-serving. Thanks <3



- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [X] Low
  - [ ] Medium
  - [ ] High

---

#### 💬 Conversation

👤 **saood06** commented the **2025-06-25** at **04:44:23**:<br>

The quickstart section seems like a very oversimplified version of the `docs/build.md` file (which I just noticed should be updated to reference `ik_llama.cpp` not `llama.cpp`.

I do think a Quick Start section similar to mainline could be beneficial but I still think it should go after the News section (which still needs to be shorter), and reference `docs/build.md`.

---

👤 **saood06** submitted a review the **2025-06-25** at **17:48:05**: 💬 `COMMENTED`

---

👤 **saood06** commented during a code review the **2025-06-25** at **17:48:05** on `README.md`:<br>

`-DGGML_BLAS=OFF`

Is not needed, it is off by default.

---

👤 **saood06** submitted a review the **2025-06-25** at **17:48:42**: 💬 `COMMENTED`

---

👤 **saood06** commented during a code review the **2025-06-25** at **17:48:42** on `README.md`:<br>

Same as above

---

👤 **jwinpbe** commented the **2025-06-25** at **21:25:24**:<br>

> Why do I see the latest news as being changed in the diff?

i'm hiding the fact that this is a drive by pull request by making it extremely amateurish (read: i am not used to using the github webui and didn't know i could just edit the readme on the main branch and make a new branch from the edit)