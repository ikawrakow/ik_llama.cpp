## 🔀 [Pull Request #637](https://github.com/ikawrakow/ik_llama.cpp/pull/637) - Add GitHub data backup

| **Author** | `ThomasBaruzier` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `tb/github-data` |
| **Target Branch** | `main` |
| **Created** | 2025-07-22 |
| **Updated** | 2025-07-22 |
| **Merged** | 2025-07-22 |

---

## 📄 Description

Hello,

The last two days have been pretty stressful, but I’m glad to see the repo back up!

To prepare for any future unexpected outages, I’m sharing what I’ve been working on while the repo was down. For now, here’s a complete archive of all discussions, issues, and pull requests from before the takedown. I’ll also push the scraping and formatting code soon.

This backup will also allow people to use the data directly for RAG use, in case the takedown was caused by scraping for this purpose (seems unlikely, but we don't know).

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [x] Low
  - [ ] Medium
  - [ ] High

---

## 💬 Conversation

👤 **ikawrakow** approved this pull request ✅ on **2025-07-22** at **16:18:31**

Wow, thank you for this!

---

👤 **ikawrakow** commented on **2025-07-22** at **16:30:05**

So, now we have a copy of the discussions, issues, and PRs
* On [Codeberg](https://codeberg.org/ikawrakow/illama.git)
* On [GitLab](https://gitlab.com/ikawrakow-group/ik_llama.cpp.git)

It would be great to also get your scraping tool so we can update and backup regularly.

---

👤 **ThomasBaruzier** commented on **2025-07-22** at **16:42:24**

Nice!

I’ll clean up and refactor my code to make it resumable (currently, everything is scraped from scratch, which isn’t ideal). If successful, we could even set it up as a GitHub Actions workflow, triggered on every commit push to the repo.

That said, frequent runs might clutter the commit history, so we can revisit the approach later.

Expect a new PR for this in the next few days!