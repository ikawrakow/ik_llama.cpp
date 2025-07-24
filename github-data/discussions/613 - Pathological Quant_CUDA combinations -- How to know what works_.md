### 🗣️ [#613](https://github.com/ikawrakow/ik_llama.cpp/discussions/613) - Pathological Quant/CUDA combinations -- How to know what works?

| **Author** | `usrlocalben` |
| :--- | :--- |
| **Created** | 2025-07-15 |
| **Updated** | 2025-07-15 |

---

#### Description

Some quants/tensors seem to be incompatible with CUDA. My current example is a Q6_K (unsloth) quant of Kimi K2. If I leave all routed exp on CPU, I can get e.g. TG=~9tps. There's some VRAM remaining (RTX 8000, Turing, 48GB) so I can put a few e.g. up_exps on GPU. When doing this TG drops to 1tps or worse.

I've seen this phenomena before, trying to offload routed experts with some other quant types (w DeepSeek R1/V3) My understanding (I think somewhere @ubergarm explained it) is that some quants are not supported on CUDA and therefore must be converted before use **per token**.

PP throughput (~80tps) is not noticeably affected, presumably because of batching. (b=ub=4096)

Good outcome, ~9tps TG
```
-mla 2 -fa -fmoe
-b 4096 -ub 4096
-ctk f16 -c 64000
--n-gpu-layers 99
-ot exps=CPU
-op 26,0,27,0,29,0
-m /path/to/Kimi-K2-Instruct-Q6_K-00001-of-00018.gguf
```

if I change to 
```
-ot "blk\.(1|2|3|4|5|6)\.ffn_up.*=CUDA0"
-ot exps=CPU
```

TG drops to 1tps or worse.

Assuming the idea is correct, Q6_K is a pathological quant type (at least on Turing) -- how to know this? How can I know what my options are when building GGUFs that match my offload/cpu arrangement? 

edit: I shouldn't say they are not _supported_, but they aren't integrated into a kernel for the required op.

---

#### 🗣️ Discussion

👤 **ikawrakow** replied the **2025-07-15** at **17:52:16**:<br>

`Q6_K` has been around forever and hence is a well supported quant on all platforms. So, it is not that.

Instead, you absolutely do not want to split up `ffn_up` and `ffn_gate` when using `-fmoe`. Try
```
-ot "blk\.(1|2|3)\.ffn_up_exps=CUDA0,blk\.(1|2|3)\.ffn_gate_exps=CUDA0"
```
instead.

If you split `ffn_up` and `ffn_gate` and there is a fused `ffn_up/ffn_gate` op where `ffn_up` is on the GPU but `ffn_gate` is on the CPU, whatever the back-end decides to do (run the op on the GPU or the CPU), the tensors need to be copied from the GPU to the CPU or vice versa. This totally kills TG performance.

> 👤 **usrlocalben** replied the **2025-07-15** at **18:37:07**:<br>
> > `Q6_K` has been around forever and hence is a well supported quant on all platforms.
> 
> I did find it surprising. If instead it were IQxxx I proably might not have been inspired to ask/write.
> 
> 
> > Instead, you absolutely do not want to split up `ffn_up` and `ffn_gate` when using `-fmoe`. Try
> 
> Makes so much sense it should have been obvious 😥
> 
> Thanks
> 
> 👤 **usrlocalben** replied the **2025-07-15** at **18:42:37**:<br>
> Furthermore, now I see why I've observed that particular offload pattern mentioned in various places.
> 
> I'll have to revisit some of my previous quant layouts and invocations. I had mixed gate/up offloads arbitrarily to optimally fill VRAM and didn't realize I was creating pathological arrangements.

---

👤 **ikawrakow** replied the **2025-07-15** at **18:07:47**:<br>

One more thing: if you have enough VRAM to use batch and u-batch of 4096, you should try removing `-op 26,0,27,0,29,0` to see how this affects your PP performance. Depending on GPU vs CPU speed, this may give you a non-negligible boost in PP performance for long prompts (longer than 4k tokens).

> 👤 **usrlocalben** replied the **2025-07-15** at **18:39:48**:<br>
> In the same testing prior to posting I did a fresh a/b test w & w/o this and it _still_ improve things, maybe 1.5x (I just tossed the measurements). I did notice the recent change to the heuristics wrt. offloading but enforcing the -op policy is still an improvement for my hw combo.