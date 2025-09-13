## 🔀 [Pull Request #554](https://github.com/ikawrakow/ik_llama.cpp/pull/554) - Update README.md to add quickstart section

| **Author** | `jwinpbe` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Source Branch** | `s6/docs_update` |
| **Target Branch** | `s6/docs_update` |
| **Created** | 2025-06-25 |
| **Updated** | 2025-06-25 |

---

## 📄 Description

add quickstart section using ubergarm's discussion post. Scrolling to the discussion every time I want to remember how to build the damn thing is a minor inconvienience so this pull request is both useful and self-serving. Thanks <3



- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [X] Low
  - [ ] Medium
  - [ ] High

---

## 💬 Conversation

👤 **saood06** commented on **2025-06-25** at **04:44:23**

The quickstart section seems like a very oversimplified version of the [`docs/build.md`](https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/build.md) file (which I just noticed should be updated to reference `ik_llama.cpp` not `llama.cpp`.

I do think a Quick Start section similar to mainline could be beneficial but I still think it should go after the News section (which still needs to be shorter), and reference `docs/build.md`.

---

👤 **jwinpbe** commented on **2025-06-25** at **04:54:37**

I'll happily defer to your judgement -- I see you updating documents all the time. I don't want to iterate over the news section as I don't feel like that's my call. Thanks again.

---

👤 **ikawrakow** commented on **2025-06-25** at **14:38:23**

Why do I see the latest news as being changed in the diff?

---

👤 **saood06** started a conversation on `README.md` on **2025-06-25** at **17:48:05**

`-DGGML_BLAS=OFF`

Is not needed, it is off by default.

---

👤 **saood06** started a conversation on `README.md` on **2025-06-25** at **17:48:42**

Same as above

---

👤 **saood06** commented on **2025-06-25** at **18:00:28**

> Why do I see the latest news as being changed in the diff?

Because this PR is targeting an old branch, and manually pulled in the changes from main.

---

👤 **saood06** commented on **2025-06-25** at **18:07:40**

>I don't want to iterate over the news section as I don't feel like that's my call. 

I'd be curious about your opinions. I ran out of ideas on how to condense it, and it is also just good to hear the perspective of someone else.

---

👤 **jwinpbe** commented on **2025-06-25** at **21:25:24**

> Why do I see the latest news as being changed in the diff?

i'm hiding the fact that this is a drive by pull request by making it extremely amateurish (read: i am not used to using the github webui and didn't know i could just edit the readme on the main branch and make a new branch from the edit)