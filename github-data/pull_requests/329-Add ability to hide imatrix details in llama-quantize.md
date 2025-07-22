### 🔀 [#329](https://github.com/ikawrakow/ik_llama.cpp/pull/329) - Add ability to hide imatrix details in llama-quantize

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-04-14 |
| **Updated** | 2025-04-14 |

---

#### Description

Simply add `--hide-imatrix` to the command line when quantizing. This will store "top_secret" in the imatrix data file name and calibration dataset fields, and zeros in the batch size and number of chunks used to compute the imatrix. Example:
```
llama_model_loader: - kv  29:                      quantize.imatrix.file str              = top_secret
llama_model_loader: - kv  30:                   quantize.imatrix.dataset str              = top_secret
llama_model_loader: - kv  31:             quantize.imatrix.entries_count i32              = 0
llama_model_loader: - kv  32:              quantize.imatrix.chunks_count i32              = 0
```

Why? Someone publishing quantized models may not want to reveal the details of the imatrix they have used.