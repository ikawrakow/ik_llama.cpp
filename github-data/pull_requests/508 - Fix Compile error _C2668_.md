### 🐛 [#508](https://github.com/ikawrakow/ik_llama.cpp/pull/508) - Fix Compile error (C2668)

| **Author** | `Gaolingx` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-09 |
| **Updated** | 2025-06-10 |

---

#### Description

The compiler(msvc) reports error: `..iqk_quantize.cpp(568,12): error C2668: "'anonymous-namespace'::hsum_float_4”: 对重载函数的调用不明确..` , I found some functions defined repeatedly and move these to `iqk_common.h`, It can be compiled successfully, but on linux doesn't seem to get the error...

![61f16a4264ac7586d17a7a7e39754920](https://github.com/user-attachments/assets/1be364ee-494e-4bfc-b2f8-9e116c3a6c82)

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [x] Low
  - [ ] Medium
  - [ ] High

---

#### 💬 Conversation

👤 **ikawrakow** submitted a review the **2025-06-09** at **15:12:45**: 💬 `COMMENTED`

---

👤 **ikawrakow** submitted a review the **2025-06-10** at **05:30:02**: ✅ `APPROVED`