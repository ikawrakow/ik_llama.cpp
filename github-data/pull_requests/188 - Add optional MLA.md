### [Pull Request #188](https://github.com/ikawrakow/ik_llama.cpp/pull/188) - Add optional MLA

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/mla` |
| **Target Branch** | `main` |
| **Created** | 2025-02-06 |
| **Updated** | 2025-02-11 |
| **Merged** | 2025-02-09 |

---

#### Description

This PR is derived from [#180](https://github.com/ikawrakow/ik_llama.cpp/issues/180). The difference to [#180](https://github.com/ikawrakow/ik_llama.cpp/issues/180) is that MLA is made optional. It is off by default, and can be turned on using the added `-mla` or `--use-mla` command line option.

Rationale: MLA improves TG speed, especially when there is a long context. But it also makes prompt processing significantly slower. Hence, MLA is made optional since advantage/disadvantage is use case dependent.

Being able to select or deselect MLA at run time is possible due to the fact that [#180](https://github.com/ikawrakow/ik_llama.cpp/issues/180) leaves the original `wkv_b` tensor and its decomposition into `wk_b` and `wv_b` in the model. This is somewhat wasteful, but these tensors are not very large and now come handy to easily select between the two attention implementations. 

In addition:
* It is now possible to use a model converted without this PR so that the `wk_b` and `wk_v` tensors are missing. In this case MLA will be disabled even if requested on the command line
* Eliminated some unnecessary copies (`ggml_cont`). This repo has supported non-contiguous RoPE for a while and con-contiguous RMS norm on CUDA was added in [#190](https://github.com/ikawrakow/ik_llama.cpp/issues/190) (the CPU has always supported non-contiguous RMS norm).

---

#### 🔀 Conversation

👤 **saood06** commented on **2025-02-08** at **11:23:52**

There were some other change's in the gguf-py/gguf/tensor_mapping.py that are in https://github.com/saood06/ik_llama.cpp/pull/1 that I missed porting over earlier. 

The next thing I was going to do was remove the old KV from being allocated, I hadn't gotten around to it, as I had a workaround from the mmap KV cache feature, but it should be a relatively simple fix, when I have more time I'll look into it.

---

👤 **saood06** commented on **2025-02-08** at **19:51:36**

@ikawrakow I made #195 to merge into this with the things mentioned.

---

👤 **ikawrakow** commented on **2025-02-09** at **11:09:23**

I think we can merge this now.

---

👤 **ikawrakow** commented on **2025-02-09** at **17:48:32**

> BF16 makes sense when it is faster, but I never bothered as I'm assuming it would come with a large quality loss.

Why? Most modern models are trained in `bf16`, so `bf16` will be better than `fp16`. But if the CPU does not have native `bf16` support it will be somewhat slower. 

> Once this is merged I'll make PR's for the warmup MoE fix and then the mmap KV allocator .

Sounds good.

---

👤 **saood06** commented on **2025-02-09** at **18:28:01**

> > BF16 makes sense when it is faster, but I never bothered as I'm assuming it would come with a large quality loss.
> 
> Why? Most modern models are trained in `bf16`, so `bf16` will be better than `fp16`. But if the CPU does not have native `bf16` support it will be somewhat slower.
> 
I mispoke, I meant I never bothered quantizing the MLA version down to Q4 or Q6 as I did with the non MLA solution. I know most models are bf16 native (Deepseek was FP8 native which I had to upscale to BF16 before making the GGUF), and I would use BF16 if I had a modern processor with support for it. 

The old solution was MHA, which quantizes down very well, and is large enough to warrant it. Heavy GQA does not, MLA is sized like heavy GQA and is also small enough where I'm fine leaving it in F16 and not smaller and not BF16 as my CPU is old and doesn't do BF16 well.