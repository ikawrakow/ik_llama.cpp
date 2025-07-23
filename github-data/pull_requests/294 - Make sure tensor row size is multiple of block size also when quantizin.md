### 🔀 [#294](https://github.com/ikawrakow/ik_llama.cpp/pull/294) - Make sure tensor row size is multiple of block size also when quantizing with --pure

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-03-27 |
| **Updated** | 2025-03-27 |

---

#### Description

`ffn_down_exps` row sizes are not a multiple of 256 in DeepSeek-Lite. When using `--pure` with `llama-quantize` this leads to a crash. I got tired of having to do custom quantization overrides in that case, so this PR adds the check for divisibility by the quantization block size also for `--pure`, and uses the fallback quantization type if necessary.