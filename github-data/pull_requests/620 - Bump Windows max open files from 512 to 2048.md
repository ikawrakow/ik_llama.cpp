## 🔀 [Pull Request #620](https://github.com/ikawrakow/ik_llama.cpp/pull/620) - Bump Windows max open files from 512 to 2048

| **Author** | `Thireus` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `patch-2` |
| **Target Branch** | `main` |
| **Created** | 2025-07-16 |
| **Updated** | 2025-07-17 |
| **Merged** | 2025-07-17 |

---

## 📄 Description

Allows up to 2048 shards to be loaded on Windows builds, from the current default of 512. This change is specific to Windows, it instructs the Windows OS that the binary requires 2048 of max opened files. This is the equivalent to Linux's `ulimit -n`.

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [x] Low
  - [ ] Medium
  - [ ] High

---

## 💬 Conversation

👤 **ikawrakow** started a conversation on `src/llama.cpp` on **2025-07-17** at **05:39:22**

Don't you want to make this dependent on the value of `GGML_MAX_CONTEXTS` instead of it being simply set to 2048?

I don't know much about Windows, but if I understand correctly the description of the `_setmaxstdio` function, it changes the max. number of files that can be open at the same time at the stream I/O level. The default for this is 512. The Microsoft engineers must have had a reason to keep it at 512 instead of just setting it to the 8192 limit of the low I/O level. If they did have a reason, then my thinking is that it would be wise to not increase the stream I/O limit unless necessary. It only becomes necessary if we want to use more than 512 shards, which is only possible if we have changed the value of `GGML_MAX_CONTEXTS`.

> 👤 **saood06** replied on **2025-07-17** at **06:03:52**
> 
> I agree, and this is what I was saying here: https://github.com/ikawrakow/ik_llama.cpp/pull/611#issuecomment-3072281429
> 
> >The default for this is 512. The Microsoft engineers must have had a reason to keep it at 512 instead of just setting it to the 8192 limit of the low I/O level.
> 
> Since this came up, I've looked into it, best reason I found was this (from a time when the true maximum was 2048):
> 
> >I believe the limit has to do with the ability to inherit the open files from a CreateProcess call. The CreateProcess has only 2048 slots for passing handles (both on 32-bit and 64-bit). You can debug a program and step into the system, exec, or spawn CRT functions to see the limit of the 2048 slots.
> >
> >If you use the Win32 file API (CreateFile, WriteFile, ReadFile, CloseHandle, etc.), then you don't have a limit on open files (well, you do but I believe it is based on your resources like memory).
> 
> Source: https://stackoverflow.com/questions/1803552/setmaxstdio-max-open-files-is-2048-only
> 
> alongside this corroborating piece from https://bugs.mysql.com/bug.php?id=24509 (they also mention Win32 on that page):
> 
> >It's a hard windows limit due to the fact of using posix-like
> >functions in some places.  I will open 2nd bug report about a 
> >handle leak when that 2048 limit is hit.
> 
> If 2048/8192+ is wanted Win32 API might be needed (not sure how big a change that would be).

> 👤 **ikawrakow** replied on **2025-07-17** at **06:35:12**
> 
> If we are sure that limitations in `CreateProcess` implementation is the only reason, then it wouldn't be an issue as `llama.cpp` is not actually spawning new processes. A file handle leak each time one starts a `llama.cpp` process is not too bad either: one simply needs to reboot their Windows box from time to time just like in the old days. Just joking. If there is indeed a file handle leak, then it is even more important to make the increase conditional upon `GGML_MAX_CONTEXTS > 512`.

> 👤 **Thireus** replied on **2025-07-17** at **06:38:36**
> 
> Change made. Please let me know if this is now acceptable.

> 👤 **saood06** replied on **2025-07-17** at **06:44:27**
> 
> > If there is indeed a file handle leak, then it is even more important to make the increase conditional upon `GGML_MAX_CONTEXTS > 512`.
> 
> I wouldn't take the "leak" part seriously as it is from "10 Dec 2006", just included that because it mentioned the handles. Win32 should only be needed if models large enough (much more than deepseek) and people have 2048 limits (instead of 8192).

---

👤 **ikawrakow** approved this pull request ✅ on **2025-07-17** at **06:50:15**