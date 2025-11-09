## 🔀 [Pull Request #444](https://github.com/ikawrakow/ik_llama.cpp/pull/444) - gguf-split : update

| **Author** | `Nexesenex` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `updated_split` |
| **Target Branch** | `main` |
| **Created** | 2025-05-22 |
| **Updated** | 2025-05-23 |
| **Merged** | 2025-05-23 |

---

## 📄 Description

Among the useful stuff on mainline, there's the updates of gguf-split.

I encountered some crashes on my KCPP fork reading the split ggufs made with IK Llama back in January. I thus merged these two commits in my IK_Llama fork to produce my split ggufs, and the crashes were gone in my KCPP fork. I kept using this ever since.

Those two commits are pre-GGUF v14 refactor, and there was not a single conflict with the merge, nor problem to compile on the current IK_LLama.

So I just put it here!

-----

gguf-split : improve --split and --merge logic ([#9619](https://github.com/ikawrakow/ik_llama.cpp/issues/9619))

* make sure params --split and --merge are not specified at same time

* update gguf-split params parse logic

* Update examples/gguf-split/gguf-split.cpp

Co-authored-by: Xuan Son Nguyen <thichthat@gmail.com>
Co-authored-by: slaren <slarengh@gmail.com>

---------

gguf-split : add basic checks ([#9499](https://github.com/ikawrakow/ik_llama.cpp/issues/9499))

* gguf-split : do not overwrite existing files when merging

* gguf-split : error when too many arguments are passed

Authored-by: slaren <slarengh@gmail.com>


- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [ ] Low
  - [x] Medium
  - [ ] High

---

## 💬 Conversation

👤 **ikawrakow** approved this pull request ✅ on **2025-05-23** at **05:07:35**