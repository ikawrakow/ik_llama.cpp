### 📝 [#463](https://github.com/ikawrakow/ik_llama.cpp/issues/463) - Research: V100 Flash Attention Implementation

| **Author** | `sempervictus` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-26 |
| **Updated** | 2025-05-29 |

---

#### Description

### Research Stage

- [ ] Background Research (Let's try to avoid reinventing the wheel)
- [x] Hypothesis Formed (How do you think this will work and it's effect?)
- [x] Strategy / Implementation Forming
- [ ] Analysis of results
- [ ] Debrief / Documentation (So people in the future can learn from us)

### Previous existing literature and research

This is a copy of https://github.com/ollama/ollama/issues/10859 but i think relevant to this fork's objectives.

i stumbled across an initial implementation of flash attention for the V100: https://github.com/ZRayZzz/flash-attention-v100/ or the apparently updated fork @ https://github.com/Coloured-glaze/flash-attention-v100. Bots say the readme translates to:

> # Flash_Attention_V100  
> Flash Attention only supports GPUs with the Ampere architecture or newer. Since it does not support the Volta architecture (as used in the V100), I created this version of Flash Attention specifically for V100 out of personal interest, following the CUTLASS tutorials and the Flash Attention 2 paper. However, due to time constraints and limited hardware resources, thorough performance tuning was not possible. As a result, the performance of this repository does not match that of PyTorch's attention implementation. Currently, the forward pass is approximately 40% faster than PyTorch, but the backward pass is about 20% slower, offsetting the gains. Additionally, this implementation does not account for boundary conditions, so sequence lengths must be padded to multiples of 32 using right padding. This will not affect normal training; simply ignore the padded positions when computing the loss.
> 
> ## Installation  
> Before installing, ensure you have:  
> - PyTorch >= 2.0.1  
> - CUDA >= 11.6  
> - Linux OS  
> - CUTLASS source code  
> 
> Modify line 146 in `setup.py` to point to the location where you downloaded the CUTLASS source code:  
> ```python
> include_dirs=[
>     Path(this_dir) / "include",
>     "/home/user/cutlass/include",
> ],
> ```
> 
> After making this change, install the package using:  
> ```bash
> python setup.py install --user
> ```
> 
> ## Usage  
> ```python
> from flash_attn_v100 import flash_attn_func  
> q = torch.empty((Z, N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=1).requires_grad_()  
> k = torch.empty((Z, N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=1).requires_grad_()  
> v = torch.empty((Z, N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=1).requires_grad_()  
> cuda_out = flash_attn_func(q, k, v, sm_scale, causal)  
> ```
> 
> ## References  
> - [Flash-Attention](https://github.com/Dao-AILab/flash-attention)  
> - [CUTLASS](https://github.com/NVIDIA/cutlass)

### Hypothesis

If this effort can be ported (and performance regression resolved), it would open up use of _runtime_ memory-hungry models to far more people on commodity hardware

### Implementation

Unfortunately not familiar enough with llama.cpp's innards to propose a porting strategy and no point in posting bot-generated content anyone here can produce :-)

### Analysis

_No response_

### Relevant log output

```shell

```

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-05-28** at **11:42:36**:<br>

So, my concept is that the flash attention implementation supports Volta, except for the case of DeepSeek models with MLA enabled where Touring or newer is required. The DeepSeek attention architecture has different K- and V-head sizes. Is this supported by the quoted implementation? The usage example suggests that it is not supported.

But apart from this, support for old hardware is not a focus of this project. Mainline `llama.cpp` is covering the old/exotic hardware use case much better than this project.

---

👤 **sempervictus** commented the **2025-05-28** at **17:20:32**:<br>

@ikawrakow thanks for jumping in. This is a class of hardware still very common in academia and much more available to aspiring developers than a data haul of water-cooled B200s so i'm hoping an exception can be made for putting talented effort toward a an area of runtime logic which underpins a lot of the operating mechanics/capability to include KV quantization. If anything, the optimal use of memory in those devices is difference between being able and unable to load a model (not being able to fit runtime memory into a single device apparently prevents loading of a model that would otherwise fit into multiple devices just fine). So far with our V100s we've see flash attention unsupported messages with every model loaded - llama3/4, phi, falcon, DS, qwen.

---

👤 **ikawrakow** commented the **2025-05-29** at **06:09:35**:<br>

@sempervictus 

Water-cooled B-200s are not a focus here either. This is a hobby project, and I develop/test on commodity hardware that I have access to, which does not include GPUs released 8 years ago. Your chances really are much better in the [llama.cpp project](https://github.com/ggml-org/llama.cpp)

---

👤 **sempervictus** commented the **2025-05-29** at **08:49:16**:<br>

Thank you