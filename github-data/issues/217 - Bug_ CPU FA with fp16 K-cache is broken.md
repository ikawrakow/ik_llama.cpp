### 🐛 [#217](https://github.com/ikawrakow/ik_llama.cpp/issues/217) - Bug: CPU FA with fp16 K-cache is broken

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-02-21 |
| **Updated** | 2025-02-22 |

---

#### Description

### What happened?

Running HellaSwag with flash attention enabled and using `fp16` for K-cache produces much lower scores than no FA or FA using `Q8_0` or `bf16` for K-cache.

### Name and Version

Latest



### What operating system are you seeing the problem on?

_No response_

### Relevant log output

```shell

```