### 🐛 [#358](https://github.com/ikawrakow/ik_llama.cpp/issues/358) - Bug: IQK_FA_ALL_QUANTS causes failure to compile

| **Author** | `saood06` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-04-30 |
| **Updated** | 2025-04-30 |

---

#### Description

### What happened?

cmake .. -DGGML_RPC=ON -DGGML_IQK_FA_ALL_QUANTS=1; cmake --build . --config Release -j 48 Fails

cmake .. -DGGML_RPC=ON; cmake --build . --config Release -j 48 Works

### Name and Version

9ba362706c998902752caf31d99fe077ed7d4faa

### What operating system are you seeing the problem on?

Clear Linux OS

### Relevant log output

[compile_errors3.txt](https://github.com/user-attachments/files/19971488/compile_errors3.txt)