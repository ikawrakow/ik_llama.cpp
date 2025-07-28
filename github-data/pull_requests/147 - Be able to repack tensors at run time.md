## 🔀 [Pull Request #147](https://github.com/ikawrakow/ik_llama.cpp/pull/147) - Be able to repack tensors at run time

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/run_time_repack` |
| **Target Branch** | `main` |
| **Created** | 2024-12-17 |
| **Updated** | 2024-12-17 |
| **Merged** | 2024-12-17 |

---

## 📄 Description

It is a bit of a hack as I didn't see a good way to figure out if tensors may be uploaded to a GPU later on. But if running on the CPU it works fine. Just use
```
-rtr or --run-time-repack
```
and all tensors types that have a corresponding type with interleaved rows will be repacked. 

**Note**: turning on run time repacking will automatically turn off `mmap`.