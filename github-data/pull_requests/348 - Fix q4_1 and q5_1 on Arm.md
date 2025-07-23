### 🐛 [#348](https://github.com/ikawrakow/ik_llama.cpp/pull/348) - Fix q4_1 and q5_1 on Arm

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-04-25 |
| **Updated** | 2025-04-25 |

---

#### Description

When I changed the `vet_dot_type` for `q8_1_x4` to `q8_2_x4` for the quants using `q8_1_x4` I forgot to also make the change for the `ARM_NEON` implementation. As a result `q4_1` and `q5_1` are currently broken. But because `q4_0/q5_0` will use `q4_1/q5_1` for a few `ffn_down` layers, `q4_0` and `q5_0` are broken as well.

Looking at the implementation, changing to use `q8_2_x4` would be too a major change. Hence, just go back to using `q8_1_x4` on Arm. If this results in some models not working correctly, then simply don't use legacy quants for those models.