### 🔀 [#279](https://github.com/ikawrakow/ik_llama.cpp/pull/279) - Fighting with cmake

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-03-22 |
| **Updated** | 2025-03-22 |

---

#### Description

`cmake` has the unpleasant habit of using "response" files to put stuff such as list of include directories. But that confuses `vim` (or at least it does the way I have set it up) when I edit CUDA files. I had tricked `cmake` into not using "response" files, but instead adding all `nvcc` command line options into `compile_commands.json`. But at some point that stopped working, I guess after a system update. I hate it, so this PR restores the desired behavior. I had to add 
```
            set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
            set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
            set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)
```
another time at the end of the block related to CUDA in `CMakeLists.txt`, else something was making my request at the beginning of the CUDA block to be ignored.