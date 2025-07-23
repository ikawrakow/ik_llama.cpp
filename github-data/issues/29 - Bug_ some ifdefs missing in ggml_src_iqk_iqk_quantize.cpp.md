### 🐛 [#29](https://github.com/ikawrakow/ik_llama.cpp/issues/29) - Bug: some ifdefs missing in ggml/src/iqk/iqk_quantize.cpp

| **Author** | `whoreson` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-08-30 |
| **Updated** | 2024-09-01 |

---

#### Description

### What happened?

```
#if GGML_USE_IQK_MULMAT
if (iqk_mul_mat...yadda-yadda
```
#if blocks are missing in a few places so it doesn't compile when GGML_NO_IQMULMAT=1 is specified.

### Name and Version

-

### What operating system are you seeing the problem on?

Other? (Please let us know in description)

### Relevant log output

```shell
-
```

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2024-08-31** at **05:56:47**:<br>

Thanks for the bug report.

Clearly I'm never using `iqk_mul_mat` disabled :-)

It should be fixed via #31

---

👤 **ikawrakow** commented the **2024-09-01** at **09:24:57**:<br>

I think I can close this now.