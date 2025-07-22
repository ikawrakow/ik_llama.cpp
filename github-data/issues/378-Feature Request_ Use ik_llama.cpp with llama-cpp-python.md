### ✨ [#378](https://github.com/ikawrakow/ik_llama.cpp/issues/378) - Feature Request: Use ik_llama.cpp with llama-cpp-python

| **Author** | `kadongre` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2025-05-04 |
| **Updated** | 2025-05-25 |

---

#### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Enable python interface for ik_llama


### Motivation

Enable python interface for ik_llama


### Possible Implementation

The install instructions of llama-cpp-python indicates that it builds its own version of llama.cpp or there is an alternative to using the Wheels interface/API 
Would be useful to leverage any of these mechanisms for ik_llama to utilize the current llama-cpp-python interface

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-05-04** at **15:40:03**:<br>

I'm not a Python person. `ik_llama.cpp` is a fork of `llama.cpp` and hence has inherited whatever Python bindings were there in June of last year. But I have no idea if they still work and, if not, what needs to get done.

---

👤 **saood06** commented the **2025-05-04** at **16:28:41**:<br>

He is asking about `llama-cpp-python` which is it's own project that pulls in llama.cpp as a submodule: https://github.com/abetlen/llama-cpp-python/tree/main/vendor

---

👤 **ikawrakow** commented the **2025-05-04** at **16:42:48**:<br>

I see. Is it even possible to have `ik_llama.cpp` live as a sub-module in that project? Mainline has been very busy pushing pieces of code from here to there, renaming functions, changing interfaces for no actual benefit, etc. So, my guess is that it will not be easy, if it is even possible.

---

👤 **Ph0rk0z** commented the **2025-05-14** at **16:17:57**:<br>

Besides stuff like -ot and other new features, can just grab the revision from around the forking. IIRC, something around 3.0. They all have tags. Then it's a matter of adding most missing function names in ~2 places. Make it pull ik_llama instead of llama.cpp as the sub-module. 

All the bindings do is call C++ functions from the library. Not sure why you'd want to embark on such a journey but it doesn't look too bad.

---

👤 **ikawrakow** commented the **2025-05-14** at **16:35:11**:<br>

You want to do it?

---

👤 **Ph0rk0z** commented the **2025-05-14** at **16:51:04**:<br>

I was going to do it to maybe use ik_llama with textgen webui but it's a whole separate repo. Out of scope from here. It's been just as easy to run llama-server.. the only reason to bother is to use HF sampling instead of built in. IK is missing nsigma sampler and --cache-reuse stuff, textgen at least has context shifting in hf_llama.cpp mode.

---

👤 **Ph0rk0z** commented the **2025-05-14** at **16:51:04**:<br>

I was going to do it to maybe use ik_llama with textgen webui but it's a whole separate repo. Out of scope from here. It's been just as easy to run llama-server.. the only reason to bother is to use HF sampling instead of built in. IK is missing nsigma and --cache-reuse stuff, textgen at least has context shifting in hf_llama.cpp mode.

---

👤 **saood06** commented the **2025-05-25** at **05:05:19**:<br>

@ikawrakow 

I agree with @Ph0rk0z this issue seems out of scope here, as solving it involves making a new repo/fork/branch of `llama-cpp-python`. Can this be closed?