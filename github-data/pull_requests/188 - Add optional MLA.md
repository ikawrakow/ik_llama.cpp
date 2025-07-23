### 🔀 [#188](https://github.com/ikawrakow/ik_llama.cpp/pull/188) - Add optional MLA

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-02-06 |
| **Updated** | 2025-02-11 |

---

#### Description

This PR is derived from #180. The difference to #180 is that MLA is made optional. It is off by default, and can be turned on using the added `-mla` or `--use-mla` command line option.

Rationale: MLA improves TG speed, especially when there is a long context. But it also makes prompt processing significantly slower. Hence, MLA is made optional since advantage/disadvantage is use case dependent.

Being able to select or deselect MLA at run time is possible due to the fact that #180 leaves the original `wkv_b` tensor and its decomposition into `wk_b` and `wv_b` in the model. This is somewhat wasteful, but these tensors are not very large and now come handy to easily select between the two attention implementations. 

In addition:
* It is now possible to use a model converted without this PR so that the `wk_b` and `wk_v` tensors are missing. In this case MLA will be disabled even if requested on the command line
* Eliminated some unnecessary copies (`ggml_cont`). This repo has supported non-contiguous RoPE for a while and con-contiguous RMS norm on CUDA was added in #190 (the CPU has always supported non-contiguous RMS norm).

---

#### 💬 Conversation

👤 **saood06** commented the **2025-02-08** at **11:23:52**:<br>

There were some other change's in the gguf-py/gguf/tensor_mapping.py that are in that branch that I missed porting over earlier. 

The next thing I was going to do was remove the old KV from being allocated, I hadn't gotten around to it, as I had a workaround from the mmap KV cache feature, but it should be a relatively simple fix, when I have more time I'll look into it.

---

👤 **saood06** commented the **2025-02-08** at **19:51:36**:<br>

@ikawrakow I made #195 to merge into this with the things mentioned.

---

👤 **ikawrakow** commented the **2025-02-09** at **11:09:23**:<br>

I think we can merge this now.

---

👤 **saood06** submitted a review the **2025-02-09** at **17:28:01**: ✅ `APPROVED`<br>

LGTM, good catch on applying cache quantization, it was something I had missed. BF16 makes sense when it is faster, but I never bothered as I'm assuming it would come with a large quality loss. 

Once this is merged I'll make PR's for the warmup MoE fix and then the mmap KV allocator .

Testing was a bit of a pain without the warmup MoE fix as loading in experts takes much longer (and it is already quite long as this server has no SSD only HDD) and takes many runs instead of just one warmup, PP seems slightly lower compared to my local testing branch but that might just be variance, or from the mmap KV allocator that I have yet to make a PR for.

---

👤 **ikawrakow** commented the **2025-02-09** at **17:48:32**:<br>

> BF16 makes sense when it is faster, but I never bothered as I'm assuming it would come with a large quality loss.

Why? Most modern models are trained in `bf16`, so `bf16` will be better than `fp16`. But if the CPU does not have native `bf16` support it will be somewhat slower. 

> Once this is merged I'll make PR's for the warmup MoE fix and then the mmap KV allocator .

Sounds good.

---

👤 **saood06** commented the **2025-02-09** at **18:28:01**:<br>

> > BF16 makes sense when it is faster, but I never bothered as I'm assuming it would come with a large quality loss.
> 
> Why? Most modern models are trained in `bf16`, so `bf16` will be better than `fp16`. But if the CPU does not have native `bf16` support it will be somewhat slower.
> 
I mispoke, I meant I never bothered quantizing the MLA version down to Q4 or Q6 as I did with the non MLA solution. I know most models are bf16 native (Deepseek was FP8 native which I had to upscale to BF16 before making the GGUF), and I would use BF16 if I had a modern processor with support for it. 

The old solution was MHA, which quantizes down very well, and is large enough to warrant it. Heavy GQA does not, MLA is sized like GQA and also small enough where I'm fine leaving it in F16, as my CPU is old and doesn't do BF16 but if I had a modern CPU I would use BF16.

---

👤 **saood06** submitted a review the **2025-02-11** at **20:15:12**: 💬 `COMMENTED`

---

👤 **saood06** commented during a code review the **2025-02-11** at **20:20:39** on `src/llama.cpp`:<br>

With the above change only one of these should be allocated so that is the only one that should be displayed as KV self size

---

👤 **saood06** submitted a review the **2025-02-11** at **20:20:40**: 💬 `COMMENTED`