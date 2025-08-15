### 🐛 [#224](https://github.com/ikawrakow/ik_llama.cpp/issues/224) - Bug: IQK_FA_ALL_QUANTS causes failure to compile

| **Author** | `saood06` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-02-23 |
| **Updated** | 2025-02-23 |

---

#### Description

### What happened?

cmake .. -DGGML_RPC=ON -DGGML_IQK_FA_ALL_QUANTS=1; cmake --build . --config Release -j 48 Fails

cmake .. -DGGML_RPC=ON; cmake --build . --config Release -j 48 Works





### Name and Version

Git commit hash: 49261058442cfe382dab3270fcd86652296a75c0

### What operating system are you seeing the problem on?

Clear Linux OS 42780

### Relevant log output


[compile_errors.txt](https://github.com/user-attachments/files/18927384/compile_errors.txt)