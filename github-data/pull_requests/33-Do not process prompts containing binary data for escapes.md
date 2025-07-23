### 🔀 [#33](https://github.com/ikawrakow/ik_llama.cpp/pull/33) - Do not process prompts containing binary data for escapes

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-09-02 |
| **Updated** | 2024-09-02 |

---

#### Description

The multiple choice evaluation has been broken in `llama.cpp` via commit `6ff13987a`, and this PR fixes it.

The multiple choice evaluation uses binary data stored in `params.prompt`. Commit `6ff13987a` adds prompt escape character processing, which modifies the binary data and renders it unusable. To preserve whatever utility `6ff13987a` might have added, we add a flag indicating if the data stored in `params.prompt` is binary and, if so, avoid the escape processing.