## 🗣️ [Discussion #636](https://github.com/ikawrakow/ik_llama.cpp/discussions/636) - Unsuspended

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2025-07-22 |
| **Updated** | 2025-07-22 |

---

## 📄 Description

Creating a pre-release after the account was unsuspended

<hr /><em>This discussion was created from the release <a href='https://github.com/ikawrakow/ik_llama.cpp/releases/tag/t0002'>Unsuspended</a>.</em>

---

## 💬 Discussion

👤 **ikawrakow** commented on **2025-07-22** at **13:39:53**

On Sunday (July 20) around noon my son sent me a message in our chat with this picture, asking what is up with `ik_llama.cpp`

![stars](https://github.com/user-attachments/assets/1d79d35c-fef3-43ec-8d1b-491a91bc2022)

The same day around 7:45 PM Central European Time my account was suspended. I did not receive any notification from GitHub about what I might have done wrong, or about the suspension.

The issue was discussed on Reddit [here](https://www.reddit.com/r/LocalLLaMA/comments/1m4vw29/ikllamacpp_repository_gone_or_it_is_only_me/?share_id=wf4Cj-LIu-gKy8VWcIZQb&utm_content=1&utm_medium=android_app&utm_name=androidcss&utm_source=share&utm_term=1), and in the [Qwen3 repository](https://huggingface.co/ikawrakow/Qwen3-30B-A3B/discussions/2) on my HF account.

Today (July 22) around 2:30 PM my account was unsuspended, again without any notification.

I would like to thank everybody for the support in the last two days!

Feel free to add comments about this strange event here.

> 👤 **jeffzhou2000** replied on **2025-07-22** at **14:37:41**
> 
> it seems all previous discussions gone.

> 👤 **saood06** replied on **2025-07-22** at **14:44:41**
> 
> > it seems all previous discussions gone.
> 
> I don't think they are gone. If you remove the "is:open" filter they are all there. For some reason they are in a weird state seemingly until someone replies in them again like https://github.com/ikawrakow/ik_llama.cpp/discussions/477 which I can see as open.

> 👤 **jeffzhou2000** replied on **2025-07-22** at **14:46:36**
> 
> thanks for help and I'll try.

> 👤 **saood06** replied on **2025-07-22** at **15:01:04**
> 
> >Feel free to add comments about this strange event here.
> 
> My account was just suspended (and similar to yours everything it touched was gone) the moment I dropped [this](https://github.com/ikawrakow/ik_llama.cpp/pull/616#issuecomment-3103156198) comment and came back right as soon as I started (but did not complete) the form asking for reinstatement.
> 
> I'm not sure what is causing Github to flag accounts but hopefully it does not happen to others that try to interact with this repo.

> 👤 **ikawrakow** replied on **2025-07-22** at **15:40:42**
> 
> Maybe it has something to do with you being a collaborator in this evil repository?
> Others have made comments without their account getting suspended.

> 👤 **saood06** replied on **2025-07-22** at **18:39:41**
> 
> > Maybe it has something to do with you being a collaborator in this evil repository? Others have made comments without their account getting suspended.
> 
> I have no idea, all I know is that the moment I provided and confirmed my phone number to support (which I'm fairly certain was never associated with my account before), and got to the reinstatement form my account came back even before I filled it out. Very strange.

---

👤 **ikawrakow** commented on **2025-07-22** at **14:45:32**

For some reason discussions do not show up. What you need to do is delete "is:open" in the search field and hit enter. This will let you see all discussions.

I'll try to fix it.

---

👤 **ikawrakow** commented on **2025-07-22** at **14:47:33**

Hahaha. To make a discussion be visible, I need to close it, and then re-open it.

It will take a while to make all visible again.

---

👤 **ikawrakow** commented on **2025-07-22** at **15:39:05**

Here is the import in Gitlab: https://gitlab.com/ikawrakow-group/ik_llama.cpp

Codeberg import fails. Tried several times, and each time it fails with a different error.
So, for now, just code without PRs and issues: https://codeberg.org/ikawrakow/illama.
I renamed it to `illama` so I can import the GitHub repo as `ik_llama.cpp`. If it keeps failing, I'll rename it back.

---

👤 **ikawrakow** commented on **2025-07-22** at **17:12:11**

Now that I have the repository in 3 different places, I decided to check how contribution counting compares between them.

### GitHub:

<img width="436" height="217" alt="Screenshot 2025-07-22 at 6 50 08 PM" src="https://github.com/user-attachments/assets/c7426c28-0762-42e7-a909-fc66b652fef0" />

### GitLab

Does not use the `.mailmap` file I added, so it lists 2 different Kawrakows:

<img width="1333" height="963" alt="Screenshot 2025-07-22 at 6 53 04 PM" src="https://github.com/user-attachments/assets/8994de42-7605-4ad4-ac38-1f165e38ab9e" />

### Codeberg

<img width="511" height="245" alt="Screenshot 2025-07-22 at 6 53 35 PM" src="https://github.com/user-attachments/assets/d9c442a5-81d5-47d9-8bf8-446415a4392e" />

Hahaha, how hard can that be?
```
git shortlog -se | sort -n -r
```

Codeberg is the only one that gets the number of commits correctly (GitLab does that too, but is unable to merge the 3 Kawrakow emails). GitHub must be using an LLM that hallucinates an answer :-)

> 👤 **mcm007** replied on **2025-07-22** at **20:49:42**
> 
> > ... must be using an LLM that hallucinates an answer :-)
> 
> Or a too lower quant that was not cooked by @ubergarm 😄