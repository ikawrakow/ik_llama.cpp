### 🔀 [#193](https://github.com/ikawrakow/ik_llama.cpp/pull/193) - RPC sync

| **Author** | `saood06` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-02-08 |
| **Updated** | 2025-06-15 |

---

#### Description

I grabbed all of the changes needed for [llama.cpp/pull/11047](https://github.com/ggerganov/llama.cpp/pull/11047) , which was https://github.com/ggerganov/llama.cpp/pull/9912 and https://github.com/ggerganov/llama.cpp/pull/9040 

This compiles, but has not been tested yet.

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-02-08** at **13:23:08**:<br>

I never use RPC, have never looked into the RPC code, so I'll have to rely on you for self-review and testing.

---

👤 **saood06** commented the **2025-02-10** at **16:40:34**:<br>

@jukofyork 
>I strongly suspect something funky is going on 

There is, see this comment: https://github.com/ikawrakow/ik_llama.cpp/pull/180#issuecomment-2625090660


This fork has much faster PP speeds, has Deepseek MLA support with a flag (-mla), this PR should allow RPC to work, and I'm working on porting the add option to override model tensor buffers.

---

👤 **saood06** commented the **2025-02-27** at **23:11:54**:<br>

This has been tested, and does not currently work. I'm not sure why as the errors I'm getting seem to have never been encountered by people on llama.cpp.

---

👤 **saood06** submitted a review the **2025-02-27** at **23:14:23**: 💬 `COMMENTED`

---

👤 **saood06** commented during a code review the **2025-02-27** at **23:14:23** on `ggml/src/ggml-rpc.cpp`:<br>

The RPC client crashes here, which happens as the RPC server hits an issue.

---

👤 **saood06** submitted a review the **2025-02-27** at **23:17:32**: 💬 `COMMENTED`

---

👤 **saood06** commented during a code review the **2025-02-27** at **23:17:32** on `ggml/src/ggml-rpc.cpp`:<br>

I'm fairly certain this is where the RPC server is crashing, although it doesn't print the message as I never ran with GGML_DEBUG on.

---

👤 **saood06** commented the **2025-04-12** at **04:39:37**:<br>

> @saood06
> 
> I just came across another [llama.cpp fork called prima.cpp](https://github.com/Lizonghang/prima.cpp?tab=readme-ov-file#-key-features) which claims to have improved support for multi-device distributed inferencing.
> 
> I haven't tried it, just saw it on reddit today. Might be worth a shot given your GPU is in a different system than your big RAM box.

Thanks for the link, it is interesting. I think it would work for dense models but not as well for MoE because as far as I can tell it doesn't handle `-ot` ([this](https://github.com/Lizonghang/prima.cpp/commit/631daadd92bfd27504c89d14ff6cd3d4ae007d53) commit looks relevant) . I'd also need windows support which is on the roadmap (but I might see what the issue is by trying to build it on my machine, and see if I can fix it), and the GPU machine has to run windows (my big RAM box runs clear linux, and I have other servers that run FreeBSD and Proxmox).

---

👤 **saood06** commented the **2025-06-15** at **11:26:50**:<br>

Closed as superseded by #480 / #506