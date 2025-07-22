### 🗣️ [#599](https://github.com/ikawrakow/ik_llama.cpp/discussions/599) - mla matrix absorbtion

| **Author** | `magikRUKKOLA` |
| :--- | :--- |
| **Created** | 2025-07-11 |
| **Updated** | 2025-07-15 |

---

#### Description

As a prefill optimization for the long context as implemented in ktransformers.  I found some cool docs.  Will leave it here.

https://github.com/madsys-dev/deepseekv2-profile/blob/main/workspace/blog/optimizing-mla.md

deepseek r1 **explanation**:

The **matrix absorption technique** in DeepSeek-V2's MLA (Multi-head Latent Attention) mechanism is a clever mathematical optimization that avoids explicitly decompressing the compressed KV cache, significantly reducing computation and memory overhead. Here's a step-by-step explanation:

### 1. **Core Problem**
Traditional MLA implementations:
- Store **compressed KV representations** (small memory footprint)
- But require **decompression** before attention calculation:
  ```math
  k_t^C = W^{UK} \cdot c_t^{KV} \quad \text{(expensive operation)}
  ```
  ```math
  v_t = W^{UV} \cdot c_t^{KV} \quad \text{(expensive operation)}
  ```

### 2. **Key Insight: Matrix Associativity**
Matrix multiplication is associative. Instead of decompressing KV, **absorb the decompression matrices** into adjacent operations:
- **K-absorption**: Fuse decompression into Q projection
- **V-absorption**: Fuse decompression into output projection

---

### 3. **K-Absorption (for Attention Scores)**
**Original computation** for non-RoPE attention scores:
```math
{q_t^C}^\top k_t^C = (W^{UQ} c_t^Q)^\top (W^{UK} c_t^{KV})
```

**Absorbed version** using associativity:
```math
{q_t^C}^\top k_t^C = \underbrace{(c_t^Q)^\top}_{\text{input}} \cdot \underbrace{(W^{UQ})^\top W^{UK}}_{\text{precomputed}} \cdot \underbrace{c_t^{KV}}_{\text{cached}}
```

**Why this helps**:
- Avoids explicit computation of full-dimensional `k_t^C`
- Replaces large matrix multiplication with smaller operations
- **FLOPs reduction**: From 33.64 MFLOP/token → 0.28 MFLOP/token

---

### 4. **V-Absorption (for Attention Output)**
**Original computation**:
```math
o = a \cdot v_t = a \cdot (W^{UV} \cdot c_t^{KV})
```
```math
u = W^O \cdot o
```

**Absorbed version** using Einstein summation:
```python
# Instead of decompressing V explicitly:
attn_output = torch.einsum('bhql,blc->bhqc', attn_weights, compressed_kv)  # Weighted sum in compressed space
attn_output = torch.einsum('bhqc,hdc->bhqd', attn_output, W_UV)            # Late decompression
u = torch.einsum('hdD,bhqd->bqD', W_O, attn_output)                        # Output projection
```

**Why this helps**:
- Avoids materializing full `v_t` (128× larger than `c_t^{KV}`)
- Fuses decompression with weighted sum
- Minimizes intermediate memory

---

### 5. **Move Elision Optimization**
**Additional problem**: Original code concatenates RoPE/non-RoPE parts of Q/K, creating large temporary tensors.

**Solution**: Compute attention scores in two parts and sum:
```python
attn_weights = (
    torch.matmul(q_pe, k_pe.transpose(2, 3))   # RoPE part
    + torch.einsum('bhqc,blc->bhql', q_nope, compressed_kv)  # Non-RoPE (absorbed)
)
```
- **Eliminates** memory-hungry concatenation ops
- **Avoids** storing full Q/K tensors (192-dimensional)

---

### 6. **Why Not Pre-Absorb All Matrices?**
Theoretically, you could precompute:
```math
W_{\text{new}}^{UQ} = (W^{UQ})^\top W^{UK} \quad \text{and} \quad W_{\text{new}}^O = W^O W^{UV}
```
But this is **inefficient** because:
- `W_{\text{new}}^{UQ}` would be a large low-rank matrix (1536×512)
- `W_{\text{new}}^O` would be massive (5120×512 per head)
- **Actual implementation** (sequential small ops) is faster:
  ![](https://i.imgur.com/5XJzZ9l.png)

---

### 7. **Performance Impact**
| Optimization                     | KV Cache Size | Compute (FLOPs/token) | Speedup (vs baseline) |
|----------------------------------|---------------|----------------------|----------------------|
| Baseline (CacheDecompressed)     | 81.92 KB      | 0.08 M               | 1.0×                 |
| CacheCompressed (no absorption)  | 1.15 KB       | 33.64 M              | 0.3×                 |
| **Absorption + Move Elision**    | **1.15 KB**   | **0.28 M**           | **1.2×**             |

**Key wins**:
- 71× smaller KV cache than decompressed version
- 120× less computation than non-absorbed compression
- Enables larger batch sizes (critical for GPU efficiency)

---

### Summary
Matrix absorption works by:
1. **Exploiting associativity** to "push" decompression matrices into adjacent operations
2. **Late decompression** of V during output projection
3. **Splitting attention scores** to avoid tensor concatenation
4. **Leveraging compressed representations** throughout computation

This transforms MLA from a memory-bound problem into a compute-bound one, better utilizing modern GPU tensor cores while maintaining 98.6% KV cache compression.

---

#### 🗣️ Discussion

👤 **ikawrakow** replied the **2025-07-11** at **12:16:34**:<br>

@magikRUKKOLA 

You may want to check #246, #260, #273. 

As far as I can tell, #246, which explains the basic idea of reducing the amount of multiply-adds when using MLA, precedes the linked doc by about a month, and is surprisingly similar to what they wrote.

#260 explains the `-amb` option, which limits the amount of intermediate compute buffer storage required.

#273 is the best MLA version in `ik_llama.cpp`. The MLA=2 variant (explained in #246) is used for prompt processing, the original MLA (MLA=1) is used for token generation. The main reason it took a while to arrive at #273 was the struggle to implement the MLA=1 case efficiently on CUDA (and the struggle was due to the much larger than usual attention head sizes of 576 and 512).

If you look at all merged PRs, you will see that it has been quite a journey to arrive at what we have today for doing fast DeepSeek inference.

---

👤 **ubergarm** replied the **2025-07-11** at **22:10:57**:<br>

A new model with MLA just dropped only 1000B-A32B https://huggingface.co/moonshotai/Kimi-K2-Instruct .... :sob: lol...

> 👤 **magikRUKKOLA** replied the **2025-07-11** at **23:35:48**:<br>
> @ubergarm 
> > A new model with MLA just dropped only 1000B-A32B https://huggingface.co/moonshotai/Kimi-K2-Instruct .... 😭 lol...
> 
> ```
> Paper Link (co**mm**ing soon)
> ```
> 
> Yeah, I am so excited too! :D
> So the minimum requirements are the 512GB RAM and 48GB VRAM to run some IQ2 quant lol. (?)  I guess its time to upgrade.
> 
> quote:
> > Agentic Intelligence: Specifically designed for **tool use**, reasoning, and autonomous problem-solving.
> 
> I suggest that the setup how the tool usage can be applied with ik_llama.cpp should be documented somewhere.  Basically we need a MITM-tool to translate JSON<->TOOL_CALL_TOKENS.  And that's about it.
> 
> 👤 **ewhacc** replied the **2025-07-12** at **09:59:24**:<br>
> @ubergarm 
> Are you going to cook quants?  ^^;   It uses deekseek architecture, so I'm hoping it runs in ik_llama.cpp flawlessly.
> 
> I have 512G RAM and would like to test IQ2.  I thought 256G is the best because using 512G (with higher bits) is too slow.  I was wrong.  Kimi-K2 keep the active experts the same but almost doubled the weights.  I guess tg speed is about the same, but pp will be slower.
> 
> I'm downloading original FP8 now.  I don't know why I'm doing this... ^^
> 
> 👤 **ubergarm** replied the **2025-07-12** at **15:43:55**:<br>
> @ewhacc 
> 
> I haven't looked to see if existing methods for going from fp8 safetensors to bf16 GGUFs would work on that model yet. I use the evshrion llama.cpp fork (from fairydreamings original MLA fork) plus triton-cpu to convert deepseek 671B without a GPU on a big RAM box. That is the first challenge.
> 
> Next you'll need over 1TB RAM to inference the Q8_0 to make an imatrix. I don't have access to the big RAM box right now, so I can't do this step at the moment. Plus its a pain to free up like 4TB disk space lol...
> 
> Keep us posted, I'm sure people will want to run this monster eventually
> 
> 👤 **ubergarm** replied the **2025-07-12** at **15:45:52**:<br>
> @magikRUKKOLA 
> 
> > I suggest that the setup how the tool usage can be applied with ik_llama.cpp should be documented somewhere. Basically we need a MITM-tool to translate JSON<->TOOL_CALL_TOKENS. And that's about it.
> 
> One guy put together a function calling wrapper thing, not sure if it is applicable here: https://github.com/ikawrakow/ik_llama.cpp/issues/407#issuecomment-2953602943
> 
> I haven't tried it personally.
> 
> 👤 **magikRUKKOLA** replied the **2025-07-12** at **20:58:08**:<br>
> > @magikRUKKOLA
> > 
> > One guy put together a function calling wrapper thing, not sure if it is applicable here: [#407 (comment)](https://github.com/ikawrakow/ik_llama.cpp/issues/407#issuecomment-2953602943)
> > 
> 
> Yeah, I noticed.  I suggest some docs should be created on how to provide a frontend for the ik_llama.cpp to support the tool calling.  But first let me observe what solution would be the most elegant.
> 
> 👤 **magikRUKKOLA** replied the **2025-07-12** at **21:05:55**:<br>
> @ewhacc 
> 
> > I have 512G RAM and would like to test IQ2.
> 
> I just noticed that IQ4_KS_R4 of Deepseek R1 is 368 GiB.  So
> 
> ```
> echo "scale=2;368*(1000/671)"|bc
> 548.32
> ```
> 
> So the kimi k2 with a similar quant might fit within the 512 GB RAM.  Or, the IQ3 quant should fit.
> 
> But... but... something should be done with the attention mechanism (for the prefill) to reduce the VRAM usage.  I am currently looking at flashinfer.  That is the exact reason of instability in ktransofmers.  Its a hurdle. :)
> 
> > I thought 256G is the best because using 512G (with higher bits) is too slow.  I was wrong.
> 
> Yeah, I made a same mistake.
> Small tip/note -- it you chose to use DDR4 don't buy 3200 MT/s (unless its for Lenovo machines).  The Samsung 2666 MT/s ECC overclocks with 1.35V great with crazy timings.  But you would have to install the additional fans and the heatsinks on top of the RAM.  Also, Gigabyte MC62-G40-00 suck -- it doesn't allow overclocking.
> 
> 👤 **magikRUKKOLA** replied the **2025-07-13** at **14:09:14**:<br>
> 621GB Q4_K quant dropped!
> 
> https://huggingface.co/KVCache-ai/Kimi-K2-Instruct-GGUF
> 
> Can't wait for the Q3 quant to try out on 512GB RAM. :)  Also setting up the water cooling for the four RTX 3090 to be able to connect [four of them] without the risers (to support as much context as possible).

---

👤 **ewhacc** replied the **2025-07-13** at **11:25:36**:<br>

@ubergarm

> I haven't looked to see if existing methods for going from fp8 safetensors to bf16 GGUFs would work on that model yet. I use the evshrion llama.cpp fork (from fairydreamings original MLA fork) plus triton-cpu to convert deepseek 671B without a GPU on a big RAM box. That is the first challenge.

I just tried fp8_cast_bf16.py but got VRAM OOM.  I didn't think this will be big challenge but 1st one is getting tough.  I will try with more VRAM, and perhaps will try evshrion llama.cpp too.  Thanks a lot for help.  I'm just giving a try your recipes.

> Next you'll need over 1TB RAM to inference the Q8_0 to make an imatrix.

Hmm, this one is what I  worried and wanted to ask.  Well, time to wake my xeon box (it's too loud).  BTW, isn't it possible to make imatrix directly from BF16?  Making Q8_0 is a must?   Ha ha, it's a long and big way to go.   FP8 -> BF16 -> Q8_0 -> imatrix -> Q2

Edit: I'm trying evshiron llama.cpp, which seems to have a direct conversion from fp8 to q8_0.

Edit:  Failed to get q8_0.    I don't know it needs 1T RAM, but seems not a RAM problem (tried on 512M)
python ev_llama.cpp/convert_hf_to_gguf.py Kimi-K2-Instruct --outfile  Kimi-K2-Instruct-q8 --outtype q8_0
ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)

> 👤 **ubergarm** replied the **2025-07-13** at **16:29:49**:<br>
> @ewhacc 
> 
> > I just tried fp8_cast_bf16.py but got VRAM OOM.
> 
> Right for the `fp8_cast_bf16.py` script from deepseek approach it is quite long. `fp8 safetensors -> bf16 safetensors -> bf16 GGUF -> Q8_0 -> imatrix -> Q2`. I believe this is the method used for mainline MLA quants of deepseek. Not sure if this works for the slightly different arch Kimi-K2 1000B-A32B or not. 
> 
> Regarding OOMing with this method, [i have some notes in a discussion with fairydreaming about using triton-cpu instead for using RAM without GPU](https://github.com/ggml-org/llama.cpp/discussions/11989#discussioncomment-13555486) that I just dug up. Also found a patch that might prevent VRAM OOM on 4090 series cards [here on hugginface](https://huggingface.co/deepseek-ai/DeepSeek-V3/discussions/17).
> 
> > BTW, isn't it possible to make imatrix directly from BF16?
> 
> Yes, if you can run inferencing with the 2TB VRAM+RAM bf16 GGUF, then you could use it directly for imatrix. I haven't tested the quality difference in terms of perplexity, but I believe the Q8_0 is sufficient given it is quite similar to the native fp8.
> 
> >  I'm trying evshiron llama.cpp, which seems to have a direct conversion from fp8 to q8_0.
> 
> Yes this is my usual method. Not sure it would work with Kimi-K2 though without some modifications. I assume you got `triton-cpu` to build (this is one of the more difficult steps of the process). Notes on building triton-cpu [here where @saood06 helped fix a build bug for them](https://github.com/triton-lang/triton-cpu/issues/237#issuecomment-2878180022).
> 
> My script then is to convert the fp8 safetensors directly to bf16 GGUF is:
> ```bash
> # evshiron/llama.cpp@63b7e8aa
> source venv/bin/activate
> python \
>       llama.cpp/convert_hf_to_gguf.py \
>       --outtype bf16 \
>       --split-max-size 50G \
>       --outfile /models/ubergarm/DeepSeek-TNG-R1T2-Chimera-GGUF/ \
>       /models/tngtech/DeepSeek-TNG-R1T2-Chimera/
> ```
> 
> If you're still getting that error, you might have to poke around in `convert_hf_to_gguf.py` search where it says `triton` for the `deepseek-v3` part. Might need to look at the recent Kimi-K2 PR https://github.com/ggml-org/llama.cpp/pull/14654 and add that to the evshiron fork or something.
> 
> I don't have access to enough RAM at the moment. Maybe will in the next few weeks :crossed_fingers: 
> 
> Thanks for blazing the trail! And feel free to open a new discussion/issue specific to Kimi-K2 etc...
> 
> 👤 **magikRUKKOLA** replied the **2025-07-13** at **18:17:56**:<br>
> > I don't have access to enough RAM at the moment. Maybe will in the next few weeks 🤞
> 
> Hey bro, are you in EU?  I can drop you some 1TB DDR5 RAM with a huge discount.
> 
> 👤 **ubergarm** replied the **2025-07-13** at **18:38:54**:<br>
> @magikRUKKOLA 
> 
> Oh man, thanks for the offer, no I'm in east coast usa currently. wendell at level1techs.com is hooking me up with access to a new remote rig he's assembling that is a big dual socket 1.5TB beast that should be online sooner than I expected!
> 
> 👤 **ewhacc** replied the **2025-07-14** at **00:16:44**:<br>
> @ubergarm 
> You had have gone through all this tough process. Thank so much for sharing experience.
> 
> > Yes, if you can run inferencing with the 2TB VRAM+RAM bf16 GGUF, then you could use it directly for imatrix. I haven't tested the quality difference in terms of perplexity, but I believe the Q8_0 is sufficient given it is quite similar to the native fp8.
> 
> Oops, 2TB.  Sounds like going through Q8_0 is a must.
> 
> 👤 **ubergarm** replied the **2025-07-14** at **02:53:30**:<br>
> @ewhacc 
> 
> So Wendell just hooked me up with remote access to a big dual socket AMD CPU rig with 42TB of kioxia flash storage i put into two RAID0 arrays and with almost 1.5TB RAM - (no GPUs). So working through it now using the "mainline" method of casting the fp8 safetensors to bf16 safetensors first. 
> 
> If I can get that working, I'll try to see if it is possible to adapt the evshiron fork to do the same MLA treatment to Kimi-K2 as it does for deepseek models and do the direct fp8 safetensors -> bf16 GGUF
> 
> A few folks working on it also here feel free to join with your findings: https://huggingface.co/gabriellarson/Kimi-K2-Instruct-GGUF/discussions/1
> 
> 👤 **ewhacc** replied the **2025-07-14** at **03:12:39**:<br>
> @ubergarm 
> > A few folks working on it also here feel free to join with your findings: https://huggingface.co/gabriellarson/Kimi-K2-Instruct-GGUF/discussions/1
> 
> Thanks for inviting. I see you already started there :)

---

👤 **ewhacc** replied the **2025-07-13** at **11:30:20**:<br>

@magikRUKKOLA 
>  Small tip/note -- it you chose to use DDR4 don't buy 3200 MT/s (unless its for Lenovo machines). The Samsung 2666 MT/s ECC overclocks with 1.35V great with crazy timings. But you would have to install the additional fans and the heatsinks on top of the RAM. Also, Gigabyte MC62-G40-00 suck -- it doesn't allow overclocking.

Thank you for the tip.  Yeah, I have temped to overclock DDR4, and even DDR5.  But, I have to check my board allow it.  Yes, RAM also needs cooling, my DDR5 gets hot when I use R1.

---

👤 **magikRUKKOLA** replied the **2025-07-15** at **19:59:27**:<br>

ATTN!  Below is not a joke.  Its an actual latest commit for the flashinfer.  Please pay attention:

```diff
-        return self.run_return_lse(q, paged_kv_cache, k_scale, v_scale)
+        return self.run_return_lse(q, paged_kv_cache, k_scale=k_scale, v_scale=v_scale)
```

Lets read the explanation:

```
fix: correctly pass k_scale and v_scale to run() in forward_return_lse
```
MORE!
```
Bug Fix: Corrected an issue in BatchPrefillWithPagedKVCacheWrapper.forward_return_lse where k_scale and v_scale were incorrectly passed as positional arguments instead of keyword arguments to run_return_lse(). This resolves a **silent misbehavior or potential runtime error** caused by functools.partialmethod expecting keyword-only arguments.
```

the comments from the **maintainer**!!

```
Great catch, left some comments for suggestions :)
```

I mean, this doesn't make sense.  I am not really sure its real.