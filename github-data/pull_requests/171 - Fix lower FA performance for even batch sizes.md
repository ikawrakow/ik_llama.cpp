### 🐛 [#171](https://github.com/ikawrakow/ik_llama.cpp/pull/171) - Fix lower FA performance for even batch sizes

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-01-12 |
| **Updated** | 2025-01-12 |

---

#### Description

This PR fixes the lower performance for even batch sizes reported in #164. The graph shows a t/s comparison between the main branch and this PR using
```
./bin/llama-batched-bench -m some_model.gguf -pps -t 16 -npp 256 -ntg 128 -npl 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 -c 4096 -rtr -fa 
```
for LLaMA-3.1-8B-Instruct quantized with `IQ4_XS` on a Ryzen-7950X CPU. We see the strange zig zag  behavior with FA enabled is no longer there. For fun I have also added the latest `llama.cpp` performance for this model on this CPU (`llama.cpp` build: `4465 (9a483999)`). The performance difference for a batch size of 16 is a factor of 2.7X.

![batches](https://github.com/user-attachments/assets/eae98329-b921-4a65-b5ca-ef2b81ee82d9)