### 🐛 [#34](https://github.com/ikawrakow/ik_llama.cpp/issues/34) - Bug: FA fails when processing prompt lengths that are not a multiple of 8

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-09-02 |
| **Updated** | 2024-09-02 |

---

#### Description

### What happened?

Assert 
```
iqk_mul_mat.cpp:6163: GGML_ASSERT(S[j] > 0) failed
```

### Name and Version

version: 3408 (57808fd4)

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
Noticed with Gemma2-2b
```