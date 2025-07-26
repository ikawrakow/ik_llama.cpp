### [Pull Request #611](https://github.com/ikawrakow/ik_llama.cpp/pull/611) - Bump GGML_MAX_CONTEXTS to allow loading more shards

| **Author** | `Thireus` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `patch-1` |
| **Target Branch** | `main` |
| **Created** | 2025-07-15 |
| **Updated** | 2025-07-16 |
| **Merged** | 2025-07-16 |

---

#### Description

This var prevents more than 64 shards from being loaded - Specifically relevant for large models such as DeepSeek R1.

I have tested it extensively for a few weeks - see https://github.com/Thireus/ik_llama.cpp/commit/a66490410a366a9605234b94d67f3d9b7b389140

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [x] Low
  - [ ] Medium
  - [ ] High

---

#### 🔀 Conversation

👤 **saood06** commented on **2025-07-15** at **01:19:45**

Would it make sense to also include this https://github.com/Thireus/ik_llama.cpp/commit/65dd65c10d2dc24cdddbd6255c3841c6a6c1038c as well for Windows users?

---

👤 **ikawrakow** commented on **2025-07-15** at **06:26:41**

How about this:
```c++
#ifndef GGML_MAX_CONTEXTS
#define GGML_MAX_CONTEXTS 64
#endif
```
along with a `cmake` variable that can be used to set `GGML_MAX_CONTEXTS`? You can then build the tool suite with whatever number of contexts you like (the way things are going, soon even 2048 may not be enough).  

I see that `GGML_MAX_CONTEXTS` is not used anywhere else apart from `ggml.c`, so strictly speaking it should not be the the `ggml` public API header (but this is of course not your fault or the issue handled by the PR).

---

👤 **Thireus** commented on **2025-07-15** at **06:35:54**

> How about this:
> 
> ```c++
> 
> #ifndef GGML_MAX_CONTEXTS
> 
> #define GGML_MAX_CONTEXTS 64
> 
> #endif
> 
> ```
> 
> along with a `cmake` variable that can be used to set `GGML_MAX_CONTEXTS`? You can then build the tool suite with whatever number of contexts you like (the way things are going, soon even 2048 may not be enough).  
> 
> 
> 
> I see that `GGML_MAX_CONTEXTS` is not used anywhere else apart from `ggml.c`, so strictly speaking it should not be the the `ggml` public API header (but this is of course not your fault or the issue handled by the PR). 

Still adds friction if users don't know they have to change it, so will need to be made explicit but I'm ok with this compromise since there aren't official pre-compiled versions here yet (less chance of people not knowing how to compile, and the Win binaries I distribute already come with 2048 set).

Thank you.

---

👤 **saood06** commented on **2025-07-15** at **06:37:41**

> along with a `cmake` variable that can be used to set `GGML_MAX_CONTEXTS`? You can then build the tool suite with whatever number of contexts you like (the way things are going, soon even 2048 may not be enough).

For a dynamic `GGML_MAX_CONTEXTS` can the windows commit I describe can be set according to this limit (capped at 8192), and included?

---

👤 **ikawrakow** commented on **2025-07-15** at **06:44:17**

> For a dynamic GGML_MAX_CONTEXTS can the windows commit I describe can be set according to this limit (capped at 8192), and included?

Don't understand this comment. Which windows commit and when is there dynamic `GGML_MAX_CONTEXTS`?

---

👤 **saood06** commented on **2025-07-15** at **06:58:21**

@ikawrakow 

>Which windows commit 

[Thireus@65dd65c](https://github.com/Thireus/ik_llama.cpp/commit/65dd65c10d2dc24cdddbd6255c3841c6a6c1038c)

>and when is there dynamic GGML_MAX_CONTEXTS?

And dynamic in the sense that if `GGML_MAX_CONTEXTS` is below 512 then nothing needs to be set (as 512 is the default), if built above 8192, set to only 8192 (as 8192 is the Windows hard upper limit [and even this is not guaranteed see [this](https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/setmaxstdio?view=msvc-160) for more info]).

---

👤 **Thireus** commented on **2025-07-15** at **08:30:07**

https://github.com/Thireus/ik_llama.cpp/commit/65dd65c10d2dc24cdddbd6255c3841c6a6c1038c would be a separate pull request as this is a different limitation (OS limitation for number of opened files), that code is required for Windows while other platforms (linux, macos) can use ulimit to lift the limitation.

---

👤 **saood06** commented on **2025-07-16** at **00:31:03**

> [Thireus@65dd65c](https://github.com/Thireus/ik_llama.cpp/commit/65dd65c10d2dc24cdddbd6255c3841c6a6c1038c) would be a separate pull request as this is a different limitation (OS limitation for number of opened files), that code is required for Windows while other platforms (linux, macos) can use ulimit to lift the limitation.

Thanks.

---

👤 **Thireus** commented on **2025-07-16** at **23:47:10**

@saood06 - https://github.com/ikawrakow/ik_llama.cpp/pull/620