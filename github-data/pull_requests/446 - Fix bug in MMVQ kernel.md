### 🐛 [#446](https://github.com/ikawrakow/ik_llama.cpp/pull/446) - Fix bug in MMVQ kernel

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-23 |
| **Updated** | 2025-05-24 |

---

#### Description

After a very long bug hunt, this PR should hopefully fix #389, #398, #425.

Thanks to everybody who tested my previous bug fix attempts!
Huge kudos to @ciprianveg who was instrumental in finding the bug!

The bug was in the CUDA matrix-vector multiplication kernel (a.k.a., MMVQ). It only shows up when the kernel processes 2 or 3 tokens. Hence, it was not observed during TG, and only showed up during PP when an expert in a MoE model ended up with having to process just 2 or 3 tokens from the batch (which is rare).

I believe all other changes I made in #442 are not necessary, but please test this PR to confirm.

Closes #389 
Closes #398
Closes #425

---

#### 💬 Conversation

👤 **ciprianveg** commented the **2025-05-23** at **11:29:36**:<br>

Thank you for the fix!🍻

On Fri, 23 May 2025, 12:17 Kawrakow, ***@***.***> wrote:

> After a very long bug hunt, this PR should hopefully fix #389
> <https://github.com/ikawrakow/ik_llama.cpp/issues/389>, #398
> <https://github.com/ikawrakow/ik_llama.cpp/issues/398>, #425
> <https://github.com/ikawrakow/ik_llama.cpp/issues/425>.
>
> Thanks to everybody who tested my previous bug fix attempts!
> Huge kudos to @ciprianveg <https://github.com/ciprianveg> who was
> instrumental in finding the bug!
>
> The bug was in the CUDA matrix-vector multiplication kernel (a.k.a.,
> MMVQ). It only shows up when the kernel processes 2 or 3 tokens. Hence, it
> was not observed during TG, and only showed up during PP when an expert in
> a MoE model ended up with having to process just 2 or 3 tokens from the
> batch (which is rare).
>
> I believe all other changes I made in #442
> <https://github.com/ikawrakow/ik_llama.cpp/pull/442> are not necessary,
> but please test this PR to confirm.
>
> Closes #389 <https://github.com/ikawrakow/ik_llama.cpp/issues/389>
> Closes #398 <https://github.com/ikawrakow/ik_llama.cpp/issues/398>
> Closes #425 <https://github.com/ikawrakow/ik_llama.cpp/issues/425>
> ------------------------------
> You can view, comment on, or merge this pull request online at:
>
>   https://github.com/ikawrakow/ik_llama.cpp/pull/446
> Commit Summary
>
>    - 193a15b
>    <https://github.com/ikawrakow/ik_llama.cpp/pull/446/commits/193a15b465abf913cd1260e422d7c7dbecd27e19>
>    Fix bug in MMVQ kernel
>
> File Changes
>
> (1 file <https://github.com/ikawrakow/ik_llama.cpp/pull/446/files>)
>
>    - *M* ggml/src/ggml-cuda/mmvq.cu
>    <https://github.com/ikawrakow/ik_llama.cpp/pull/446/files#diff-215515d65e174fb02240522a4bb36f5c8f974d129f7a8d1aa6026a4dbd8dff12>
>    (5)
>
> Patch Links:
>
>    - https://github.com/ikawrakow/ik_llama.cpp/pull/446.patch
>    - https://github.com/ikawrakow/ik_llama.cpp/pull/446.diff
>
> —
> Reply to this email directly, view it on GitHub
> <https://github.com/ikawrakow/ik_llama.cpp/pull/446>, or unsubscribe
> <https://github.com/notifications/unsubscribe-auth/AJTBYK7WCU4ARPNJHW3ML4D273RTLAVCNFSM6AAAAAB5YG7EMGVHI2DSMVQWIX3LMV43ASLTON2WKOZTGA4DKNZVGEYDGNA>
> .
> You are receiving this because you were mentioned.Message ID:
> ***@***.***>
>

---

👤 **ikawrakow** commented the **2025-05-23** at **15:25:05**:<br>

I think I'll merge this now. It fixes a real bug, so it should be merged irrespective of it fixing #389, #398, #425.

---

👤 **Panchovix** commented the **2025-05-23** at **16:00:18**:<br>

Amazing, thanks for all your work!

---

👤 **p4s2wd** commented the **2025-05-24** at **05:12:04**:<br>

Thank you!

---

👤 **pt13762104** commented the **2025-05-24** at **09:31:08**:<br>

It's working fine now, thank you for your patience