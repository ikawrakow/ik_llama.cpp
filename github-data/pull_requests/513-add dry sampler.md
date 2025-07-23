### 🔀 [#513](https://github.com/ikawrakow/ik_llama.cpp/pull/513) - add dry sampler

| **Author** | `firecoperana` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-10 |
| **Updated** | 2025-06-19 |

---

#### Description

I test this using the example in https://github.com/vllm-project/vllm/pull/11368 and it looks ok.

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [ ] Low
  - [ ] Medium
  - [x] High

---

#### 💬 Conversation

👤 **saood06** commented the **2025-06-10** at **02:57:13**:<br>

This already looks so much better than #504 just from looking at how much more similar it is to the reference implementation.

It was taking time testing that because it looked like it had a lot of edge cases that would lead to issues or at least bugs (some more minor than others).

---

👤 **ikawrakow** submitted a review the **2025-06-10** at **05:42:27**: 💬 `COMMENTED`

---

👤 **ikawrakow** commented during a code review the **2025-06-10** at **05:42:27** on `examples/rpc/CMakeLists.txt`:<br>

Why do we need this?

---

👤 **ikawrakow** submitted a review the **2025-06-10** at **05:42:44**: 💬 `COMMENTED`

---

👤 **ikawrakow** commented during a code review the **2025-06-10** at **05:42:44** on `examples/server/CMakeLists.txt`:<br>

Why is this needed?

---

👤 **ikawrakow** submitted a review the **2025-06-10** at **05:47:23**: 💬 `COMMENTED`

---

👤 **ikawrakow** commented during a code review the **2025-06-10** at **05:47:23** on `src/llama.cpp`:<br>

The DRY sampler only depends on the vocabulary, not the entire model. Wouldn't it have been better to define the interface that way (taking a pointer to vocabulary instead of model)?

---

👤 **firecoperana** submitted a review the **2025-06-10** at **12:39:44**: 💬 `COMMENTED`

---

👤 **firecoperana** submitted a review the **2025-06-10** at **12:40:23**: 💬 `COMMENTED`

---

👤 **firecoperana** commented during a code review the **2025-06-10** at **12:40:23** on `src/llama.cpp`:<br>

I can change it.

---

👤 **firecoperana** submitted a review the **2025-06-10** at **12:49:08**: 💬 `COMMENTED`

---

👤 **ikawrakow** commented the **2025-06-10** at **13:38:46**:<br>

@saood06 Any other comments?

---

👤 **saood06** commented the **2025-06-11** at **05:35:49**:<br>

Tried to build this to test and got this:

```cpp
/ik_llama.cpp/src/../include/llama.h:1240:54: error: unknown type name ‘llama_sampler_dry’
 1240 |     void llama_sample_dry(struct llama_context* ctx, llama_sampler_dry* smpl, llama_token_data_array* candidates_p);
      |                                                      ^~~~~~~~~~~~~~~~~
```

---

👤 **firecoperana** commented the **2025-06-12** at **02:58:18**:<br>

> > Can you clean the build folder and try again?
> 
> This was with a clean build folder.
> 
> > It compiles fine for me. Build command I use. cmake -B build -DGGML_CUDA=ON -DGGML_BLAS=OFF -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=ON -DGGML_SCHED_MAX_COPIES=1
> 
> Maybe it is because you set `-DLLAMA_BUILD_TESTS=OFF`, sorry I should have given you more of the compile error log.
> 
> ```
> In file included from /home/saood06/ik_main/ik_llama.cpp/tests/test-c.c:1:
> /home/saood06/ik_main/ik_llama.cpp/src/../include/llama.h:1240:54: error: unknown type name ‘llama_sampler_dry’
>  1240 |     void llama_sample_dry(struct llama_context* ctx, llama_sampler_dry * smpl, llama_token_data_array* candidates_p);
>       |                                                      ^~~~~~~~~~~~~~~~~
> gmake[2]: *** [tests/CMakeFiles/test-c.dir/build.make:79: tests/CMakeFiles/test-c.dir/test-c.c.o] Error 1
> gmake[1]: *** [CMakeFiles/Makefile2:2688: tests/CMakeFiles/test-c.dir/all] Error 2
> gmake[1]: *** Waiting for unfinished jobs....
> ```

Should be good this time.

---

👤 **ikawrakow** submitted a review the **2025-06-19** at **07:24:21**: ✅ `APPROVED`