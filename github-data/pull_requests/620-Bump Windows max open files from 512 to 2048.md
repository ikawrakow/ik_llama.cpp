### 🔀 [#620](https://github.com/ikawrakow/ik_llama.cpp/pull/620) - Bump Windows max open files from 512 to 2048

| **Author** | `Thireus` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-07-16 |
| **Updated** | 2025-07-17 |

---

#### Description

Allows up to 2048 shards to be loaded on Windows builds, from the current default of 512. This change is specific to Windows, it instructs the Windows OS that the binary requires 2048 of max opened files. This is the equivalent to Linux's `ulimit -n`.

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [x] Low
  - [ ] Medium
  - [ ] High

---

#### 💬 Conversation

👤 **ikawrakow** submitted a review the **2025-07-17** at **05:39:22**: 💬 `COMMENTED`

---

👤 **ikawrakow** commented during a code review the **2025-07-17** at **05:39:22** on `src/llama.cpp`:<br>

Don't you want to make this dependent on the value of `GGML_MAX_CONTEXTS` instead of it being simply set to 2048?

I don't know much about Windows, but if I understand correctly the description of the `_setmaxstdio` function, it changes the max. number of files that can be open at the same time at the stream I/O level. The default for this is 512. The Microsoft engineers must have had a reason to keep it at 512 instead of just setting it to the 8192 limit of the low I/O level. If they did have a reason, then my thinking is that ot would be wise to not increase the stream I/O limit unless necessary. It only becomes necessary if we want to use more than 512 shards, which is only possible if we have changed the value of `GGML_MAX_CONTEXTS`.

---

👤 **saood06** submitted a review the **2025-07-17** at **06:03:53**: 💬 `COMMENTED`

---

👤 **ikawrakow** submitted a review the **2025-07-17** at **06:35:12**: 💬 `COMMENTED`

---

👤 **ikawrakow** commented during a code review the **2025-07-17** at **06:35:12** on `src/llama.cpp`:<br>

If we are sure that limitations in `CreateProcess` implementation is the only reason, then it wouldn't be an issue as `llama.cpp` is not actually spawning new processes. A file handle leak each time one starts a `llama.cpp` process is not too bad either: one simply needs to reboot their Windows box from time to time just like in the old days. Just joking. If there is indeed a file handle leak, then it is even more important to make the increase conditional upon `GGML_MAX_CONTEXTS > 512`.

---

👤 **Thireus** submitted a review the **2025-07-17** at **06:38:36**: 💬 `COMMENTED`

---

👤 **Thireus** commented during a code review the **2025-07-17** at **06:38:36** on `src/llama.cpp`:<br>

Change made. Please let me know if this is now acceptable.

---

👤 **saood06** submitted a review the **2025-07-17** at **06:44:27**: 💬 `COMMENTED`

---

👤 **saood06** commented during a code review the **2025-07-17** at **06:44:27** on `src/llama.cpp`:<br>

> If we are sure that limitations in `CreateProcess` implementation is the only reason, then it wouldn't be an issue as `llama.cpp` is not actually spawning new processes. A file handle leak each time one starts a `llama.cpp` process is not too bad either: one simply needs to reboot their Windows box from time to time just like in the old days. Just joking. If there is indeed a file handle leak, then it is even more important to make the increase conditional upon `GGML_MAX_CONTEXTS > 512`.

I wouldn't take the "leak" part seriously as it is from "10 Dec 2006", just included that because it mentioned the handles. Win32 should only be needed if models large enough and people have 2048 limits (instead of 8192).

---

👤 **ikawrakow** submitted a review the **2025-07-17** at **06:50:15**: ✅ `APPROVED`