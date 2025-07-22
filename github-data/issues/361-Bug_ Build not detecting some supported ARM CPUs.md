### 🐛 [#361](https://github.com/ikawrakow/ik_llama.cpp/issues/361) - Bug: Build not detecting some supported ARM CPUs

| **Author** | `saood06` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2025-04-30 |
| **Updated** | 2025-05-02 |

---

#### Description

### What happened?

This was reported in #345 and I was also able to reproduce it on an Android device, there is a workaround with #347 but ideally you should not need to set the architecture flag manually. This does not seem to affect the Apple ARM devices.

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-05-02** at **05:23:08**:<br>

We can add something along the lines of mainline's automatic CPU feature detection. But I also have the experience that since they added the feature, mainline runs slower on my M2-Max CPU as it enables the `i8mm` CPU feature, but my guess is that this is emulated and not an actual feature of the M2 CPU.

---

👤 **saood06** commented the **2025-05-02** at **05:38:14**:<br>

> We can add something along the lines of mainline's automatic CPU feature detection. 

Yes, I just created the issue since I hadn't looked into it fully.

>But I also have the experience that since they added the feature, mainline runs slower on my M2-Max CPU as it enables the `i8mm` CPU feature, but my guess is that this is emulated and not an actual feature of the M2 CPU.

That aligns with what was reported in #345 where the user had better performance with `-march=armv8.2-a+dotprod+fp16+noi8mm+nosve+nosme` over just `"-march=armv8.2-a+dotprod+fp16"`. So it may not be just the M2 CPU. I'm not very familiar with the actual hardware implementation of the recent ARM extensions so I can't really say.