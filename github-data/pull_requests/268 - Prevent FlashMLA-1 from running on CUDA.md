## 🔀 [Pull Request #268](https://github.com/ikawrakow/ik_llama.cpp/pull/268) - Prevent FlashMLA-1 from running on CUDA

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/avoid_cuda_mla_1` |
| **Target Branch** | `main` |
| **Created** | 2025-03-19 |
| **Updated** | 2025-03-19 |
| **Merged** | 2025-03-19 |

---

## 📄 Description

It is not supported, so let's not spam the user with messages about that by not allowing it to run on the GPU in the first place.

Interestingly enough, with this I can use `-ot attn_k=CPU,attn_v=CPU -mla 1 -fa -rtr -ctk q8_0 -nkvo` to run attention computations on the CPU using FlashMLA-1 with `Q8_0` KV cache stored on the host. For DeepSeek-Lite I get 134 t/s, which is about 25% slower than `ik_llama.cpp` with full GPU offload, and about the same as mainline `llama.cpp` with all layers offloaded to the GPU.  For a context of 65k tokens, this uses 1032 MiB of KV cache (will be 2.6X larger for DeepSeek-R1) and has a CUDA compute buffer of just 242 MiB!