### 🐛 [#300](https://github.com/ikawrakow/ik_llama.cpp/issues/300) - Bug: IQK_FA_ALL_QUANTS causes failure to compile

| **Author** | `saood06` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-03-31 |
| **Updated** | 2025-04-01 |

---

#### Description

### What happened?

cmake .. -DGGML_RPC=ON -DGGML_IQK_FA_ALL_QUANTS=1; cmake --build . --config Release -j 48 Fails

cmake .. -DGGML_RPC=ON; cmake --build . --config Release -j 48 Works

### Name and Version

Git commit hash: 23b0addb34d8942baedc6f968460560392feadd3

### What operating system are you seeing the problem on?

Clear Linux OS

### Relevant log output

[compile_errors2.txt](https://github.com/user-attachments/files/19534579/compile_errors2.txt)

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-03-31** at **11:53:55**:<br>

Sorry I broke it again. I'll look into it in a moment.

I guess, it would be useful to have CI, but with all the tests that need to be run I'll exhaust the free minutes really quickly.