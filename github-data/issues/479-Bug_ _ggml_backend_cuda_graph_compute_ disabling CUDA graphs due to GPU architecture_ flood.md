### 🐛 [#479](https://github.com/ikawrakow/ik_llama.cpp/issues/479) - Bug: \"ggml_backend_cuda_graph_compute: disabling CUDA graphs due to GPU architecture\" flood

| **Author** | `pt13762104` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-31 |
| **Updated** | 2025-05-31 |

---

#### Description

### What happened?

I used my GTX 1660 Ti (which probably doesn't support CUDA graphs). The message "ggml_backend_cuda_graph_compute: disabling CUDA graphs due to GPU architecture" is flooded thousands of times instead of only once. 

### Name and Version

version: 3719 (7239ce6b)
built with cc (GCC) 15.1.1 20250521 (Red Hat 15.1.1-2) for x86_64-redhat-linux

### What operating system are you seeing the problem on?

_No response_

### Relevant log output

```shell

```

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-05-31** at **13:56:56**:<br>

So, in this repository `GGML_CUDA_USE_GRAPHS` is off by default. You have explicitly enabled it, but are using a GPU that does not support CUDA graphs and are not satisfied with the observed behavior.

There are 3 possible ways how the application could behave:
1. Flood your terminal with messages that CUDA graphs are not supported (observed behavior)
2. Abort the execution with an error message telling you that CUDA graphs are not supported
3. Silently disable CUDA graphs (or perhaps print 1 warning that you will not notice between all the other log)

1 and 2 are equivalent. You just rebuild the app with `-DGGML_CUDA_USE_GRAPHS=OFF`.

But it seems you think 3 is better?

---

👤 **pt13762104** commented the **2025-05-31** at **13:59:34**:<br>

Oh, I build it with -DCMAKE_CUDA_ARCHITECTURES="75", didn't know such flags existed. Thank you