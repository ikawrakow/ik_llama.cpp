## 📌 [Issue #638](https://github.com/ikawrakow/ik_llama.cpp/issues/638) - "github_data" dir contains filename causing issues on Windows

| **Author** | `sousekd` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-07-23 |
| **Updated** | 2025-07-23 |

---

## 📄 Description

Commit ab7d193 makes it difficult to work with the repo on Windows, as it contains files with long names and/or invalid characters causing all kinds of `git` issues.

---

## 💬 Conversation

👤 **ikawrakow** commented on **2025-07-23** at **09:07:14**

What are the invalid characters? We can ask @ThomasBaruzier to recreate, but he will need more specific requirements to change the scripts.

---

👤 **ThomasBaruzier** commented on **2025-07-23** at **09:30:13**

I don't have access to a windows machine at the moment. I will sanitize the filenames more strictly and limit them to 100 chars instead of 200.

---

👤 **ikawrakow** commented on **2025-07-23** at **09:43:53**

> I don't have access to a windows machine at the moment. I will sanitize the filenames more strictly and limit them to 100 chars instead of 200

Isn't it better to just name the files "Issue_XXX", "Discussion_XXX" and "PR_XXX", where `XXX` is the number of the issue/PR/discussion? There is of course some value in having the title in the file name, but if this causes problems, I can just use grep to see the titles.

---

👤 **saood06** commented on **2025-07-23** at **09:45:31**

>Isn't it better to just name the files "Issue_XXX", "Discussion_XXX" and "PR_XXX", where XXX is the number of the issue/PR/discussion? There is of course some value in having the title in the file name, but if this causes problems, I can just use grep to see the titles.

If you do that you could have a .json file with a dictionary.

---

👤 **ThomasBaruzier** commented on **2025-07-23** at **10:11:27**

> Isn't it better to just name the files "Issue_XXX", "Discussion_XXX" and "PR_XXX", where `XXX` is the number of the issue/PR/discussion? There is of course some value in having the title in the file name, but if this causes problems, I can just use grep to see the titles.

I already went ahead and sanitized the filenames to allow only `a-zA-Z0-9_.\- ` with an 80 character limit before seeing your message. I can switch to your proposed approach if you prefer, though I’m fairly confident the current fix is sufficient.

---

👤 **ikawrakow** commented on **2025-07-23** at **10:36:47**

@sousekd  Can you check if [#640](https://github.com/ikawrakow/ik_llama.cpp/issues/640) fixes your problem? Thanks.

---

👤 **sousekd** commented on **2025-07-23** at **11:29:42**

@ikawrakow @ThomasBaruzier Sorry I was off. **Yes** it fixes the problem. Thank you.

The issue was:

`error: cannot stat 'github-data/issues/383-Bug_ Loading DeepSeek R1T Chimera causes _llama_model_load_ error loading model_ check_tensor_dims_ tensor 'blk.0.attn_q_b.weight' has wrong shape; expected  1536, 73728, got  1536, 24576,     1, ': Filename too long`