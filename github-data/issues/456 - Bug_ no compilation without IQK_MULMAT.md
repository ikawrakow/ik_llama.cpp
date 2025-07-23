### 🐛 [#456](https://github.com/ikawrakow/ik_llama.cpp/issues/456) - Bug: no compilation without IQK_MULMAT

| **Author** | `Nexesenex` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-25 |
| **Updated** | 2025-05-25 |

---

#### Description

### What happened?

IK_LLama won't compile without IQK_Mulmat's compilation activated.

### Name and Version

Last version, PR446 merged

### What operating system are you seeing the problem on?

Win11, MSVS

### Relevant log output

```shell
The cause is probably in ggml.c

Line 15044/45 :

"#if GGML_USE_IQK_MULMAT
static void ggml_compute_forward_mul_mat_id_up_gate("

So, the OP "GGML_OP_MOE_FUSED_UP_GATE" involving
ggml_"compute_forward_mul_mat_id_up_gate",
OP which is not under the condition "#if GGML_USE_IQK_MULMAT",
will not be compiled because "static void ggml_compute_forward_mul_mat_id_up_gate" is not available.
```

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-05-25** at **04:30:11**:<br>

It no longer works without `GGML_USE_IQK_MULMAT`, so I'll just remove that option.

---

👤 **Nexesenex** commented the **2025-05-25** at **12:27:17**:<br>

Et voilà!