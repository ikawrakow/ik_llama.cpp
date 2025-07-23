### 📝 [#254](https://github.com/ikawrakow/ik_llama.cpp/issues/254) - Split-mode row

| **Author** | `davidsyoung` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2025-03-12 |
| **Updated** | 2025-03-13 |

---

#### Description

### What happened?

With the experts being quite large on bigger MoE models, if we were able to split by row instead of layers, it'd allow a much more even balancing of the model across multiple cards.

Is `-split-mode row` something that we can get working? As of right now, it doesn't seem to work with DeepSeek V3/R1.

### Name and Version

Current main

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell

```

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-03-12** at **17:19:03**:<br>

Would be nice, I agree. 

Here 3 examples from the CUDA code where the comments/asserts say that split tensors are not supported. 
 
https://github.com/ikawrakow/ik_llama.cpp/blob/3f23ed68f17583a8ee63afd0c214f5b39226226c/ggml/src/ggml-cuda.cu#L731

https://github.com/ikawrakow/ik_llama.cpp/blob/3f23ed68f17583a8ee63afd0c214f5b39226226c/ggml/src/ggml-cuda.cu#L2228

https://github.com/ikawrakow/ik_llama.cpp/blob/3f23ed68f17583a8ee63afd0c214f5b39226226c/ggml/src/ggml-cuda.cu#L2228

Most noticeably, there is clearly no support for MoE models with split tensors. This is not code I wrote, it is inherited from upstream.

---

👤 **davidsyoung** commented the **2025-03-13** at **17:42:30**:<br>

Hmm, yeah, it seems as though there's not a lot we can do in that case with splitting MoE based tensors.