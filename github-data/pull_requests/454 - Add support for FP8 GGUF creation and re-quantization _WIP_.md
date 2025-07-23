### 🔀 [#454](https://github.com/ikawrakow/ik_llama.cpp/pull/454) - Add support for FP8 GGUF creation and re-quantization (WIP)

| **Author** | `saood06` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-24 |
| **Updated** | 2025-06-15 |

---

#### Description

The goal of this is to be able to directly handle FP8 (more specifically E4M3) native models by creating an FP8 GGUF, which can then be quantized into a GGUF that can be used for inferencing (inference on FP8 is beyond the scope of this PR similar to #169).

Currently only the FP8 GGUF creation is implemented (which involved including the weight_scale_inv and FP8_E4M3 quant methods). Tested with [this](https://huggingface.co/Qwen/Qwen3-0.6B-FP8/tree/main) tiny model for now which successfully created a GGUF and was able to dump it.

I picked that model as I think it follows the same usage of scale as Deepseek and is tiny. Other models such as [this](https://huggingface.co/RedHatAI/Mistral-Nemo-Instruct-2407-FP8/tree/main) seem like they handle scale differently and so that is definitely something that should be addressed if we want to support all FP8 native models but I'm leaving that for later.

I will attempt to add the quantization support later (handling the scale) but wanted to create this draft PR now in case there is any feedback on this idea or approach. 

(Also the enum value for FP8_E4M3 is set to 999 for now as I don't know where it should slot in)

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-05-24** at **11:34:38**:<br>

I was thinking that it would be useful to support `fp8`.

---

👤 **ikawrakow** commented the **2025-05-25** at **04:37:33**:<br>

Btw, thinking about the matrix multiplication implementation, it seems one will need to multiply the activations with the `fp8` scales before doing the multiplication with the `fp8` tensor. The alternative would be to just directly convert to `Q8_0`, folding the scales into the `Q8_0` quants.

---

👤 **saood06** commented the **2025-05-25** at **04:44:32**:<br>

> Btw, thinking about the matrix multiplication implementation, it seems one will need to multiply the activations with the `fp8` scales before doing the multiplication with the `fp8` tensor. 

That is the approach I had in mind for when I come back to finish it (might be tonight if things go well and nothing else takes my time).

>The alternative would be to just directly convert to `Q8_0`, folding the scales into the `Q8_0` quants.

Interesting. I hadn't considered that. I'm still going to attempt the first approach as it is what makes most sense to me, even if this approach is better.

---

👤 **whatever1983** commented the **2025-06-15** at **10:24:27**:<br>

Yo, guys, I just bumped into a surprise today:

https://hf-mirror.com/nvidia/DeepSeek-V3-0324-FP4

Benchmark 	DeepSeek V3-0324 	DeepSeek V3-0324-FP4
MMMU Pro 	82 	82.9
GPQA Diamond 	66 	67.2
LiveCodeBench 	41 	52.23  (Is this a typo?  42.23 is more likely, but 52 is possible)
AIME 2024 	52 	49.3
MATH-500 	94 	94.4
MGSM 	92 	92.8

Apparently, the Nvidia official FP4 quantization of DS V3-0324 has a LiveCodeBench boost from 41 to 52.   So I thought about it for a while: how can FP4 quant of a model can boost coding that much.  The only explanation I can think of is this: The TensorRT model optimizer is actually doing one pass fine tuning.   Nvidia's finetuning Token count is probably already in the multi-trillion tokens range.  So a TensorRT model optimization run using much higher quality trillions of tokens than the FP8 model you are tuning to FP4 is like a one pass finetune that can actually boost model performance greatly. 

This raises some serious implications: 

1.  We should probably quant from FP4 native instead of FP8 or BF16, if the FP4 finetune performs way better than the original.
2. If FP4 is more than perfect, runs better than FP8 and BF16,  quantization is dead.  Any thing 2bit/3bit would wreck coding performance too much, anything 5bit-8bit to represent FP4 is a waste.  FP4 to other forms of 4bit data representation ie IQ4K, IQ4XS isn't lossless.
3.  So instead of quantization, using higher precision hardware units(BF16, FP8) to run FP4 models losslessly is probably the way forward.    On the CPU side, before Zen6 gets fp4 or Intel Xeon gets AMX-FP4, we can use BF16 to emulate FP4.  I don't care if we take a 30% hit in performance running FP4 to BF16.  It is lossless.  On the CUDA side, we can just use "Chitu" kernels which runs FP4 with Ampere level hardware: https://kkgithub.com/thu-pacman/chitu
4.  Before FP8 conversion is finished,  it is already deprecated.  Blackwell FP4 training or quantized FP4 are already here.  So I think FP4 actually now takes priority vs FP8 now.  Half the VRAM with potential Nvidia tuned higher performance is killer.

---

👤 **ikawrakow** commented the **2025-06-15** at **11:05:51**:<br>

>  FP4 to other forms of 4bit data representation ie IQ4K, IQ4XS isn't lossless.

It is absolutely lossless. All you need to do is use an alternative lookup table that matches NVidia's 16 distinct `fp4` values, and voila, you have `fp4` implementation. The model browser on HF doesn't show the tensors, so I cannot see the `fp4` block size they have used or the tensor types, but for sure one of the `IQ4` quants will do the trick. One wouldn't need to quantize as 4 bits are 4 bits, so can be used directly. We will not need to wait for Zen6 or AMX-FP4, or use "Chitu" kernels, it will just work with a minor change.

> Any thing 2bit/3bit would wreck coding performance too much

This sounds surprising. My bet is that one will be able to get a much higher quality 2- or 3-bit quantization from an `fp4` model than from `fp8` or `bf16`. 

> Before FP8 conversion is finished, it is already deprecated. 

You will not be the first and I'm sure you will not be the last to declare something deprecated, which then is alive and kicking long time after its presumed death. The model seems to clock in at around 420 GB, so that's more like 5 bpw than 4. Presumably because some of the tensors are not `fp4`? Or perhaps because they have used blocks of 8 with `fp8` scales? Either way, a quantized version of DeepSeek-V3-0324 at 5 bpw is effectively lossless.

---

👤 **saood06** commented the **2025-06-15** at **11:24:59**:<br>

Closing as even though the approach could work, my attempt was wrong.

---

👤 **saood06** commented the **2025-06-15** at **12:04:33**:<br>

> why would you close this issue?

This isn't an issue, it's a PR with code that is wrong, and other than looking at it for reference to the approach there is very little value to building off of it.

>Llama.cpp main branch also refused to accept an already implemented FP8 code path for months, which is a mistake.

That has nothing to do with this.

> Taking a hit for AIME 2024 and getting a boost for LiveCodeBench is a great tradeoff. Nvidia obviously has more coding finetuning data than math data. Coders have a $150K-$300K/year salary compared to mathematician's at what? $60-80K/year. So any boost in coding is worth more than AIME or graduate level reasoning.

I think there is no boost. The benchmark numbers have margins of error and often times different testing approaches.

There is absolutely zero evidence I could find or that you provided that suggests they did some form of QAT or just fine-tuning after quantization to recover accuracy. 

Getting an ~425 GB quant of deepseek to perform about on par with unquantized is not really that impressive.

Look at this https://github.com/ikawrakow/ik_llama.cpp/discussions/477#discussioncomment-13339067, the graph only goes up to ~370 GB and yet approaches 0 loss.

---

👤 **saood06** commented the **2025-06-15** at **14:24:47**:<br>

> The quant size isn't that impressive. The thing is if you run the 370GB non-FP4 quant on an EPYC with 512GB ram, you get 10-20 tokens/s with a 24GB VRAM GPU. That's a 1000W platform you run at home. that's 50w-100w per token generated a sec.
> 
> 8x FP4 accelerated GPUs might cost $400K each at 10KW each generating 21K tokens/s on 8x GB200s. That's 2w per token generated per sec. a 25-50x reduction in power density. Assume a DDR4 Based EPYC with 24G VRAM GPU at $5K, or a DDR5 Based EPYC with 24G 4090 at $10K, nvidia is 40 times more expensive cap ex but generates 1000 times tokens(21K vs 21 tokens/s). So per token generated is 25 times less at the capex.

You are just stating GPUs are more power efficient at doing matrix multiplications than CPUs.

I focused on loss/quality as that seemed to be the major point of your messages about the quality of their fp4 quant vs unquantized.

> I am sorry for the mathematics. This order of magnitude difference is turning us into a shared structure where the API endpoint steal all your code output. If you have to run LLMs at home or privately, you'd hope that future CPU/GPU both have FP4 transformers capabilities.

How one chooses to use models is up to them. I personally use API/cloud offerings for things that I am comfortable with the privacy loss and/or do not care about manually sampling via looking at token probabilities ( I know there are certain API offerings that do offer that, but it is not offered by the services I prefer to use).

> Just the cost perspective, you are 25x times off.

How can I be 25x times off if I made no claim about cost (let alone a numeric one). I even stated that the model linked is useful for it's performance on specific hardware.


> Then there is the quality. PPL means nothing. 0.005 difference in PPL could mean a difference between code that runs in VS code, or code that doesn't. There is a difference for code even at IQ6K, q8_0, BF16 levels even though PPL is 0.25% different. 

Yes I am aware of that, there was even an example [here](https://github.com/ikawrakow/ik_llama.cpp/issues/383#issuecomment-2882600098) where performance collapse was observed even though PPL looked good, but the problem is there are infinite valid ways to measure quality, and benchmarking takes time (especially for large models). NVIDIA seemingly didn't even bother to run benchmarks on the unquantized version (and like I said chose to use third party that were far lower than the official numbers which makes their quant look far better than it should).

> I don't know about you, but running non-perfect quants non-FP4 accelerated on home EPYC servers is not fun. I am running it. Waiting for 8K thinking tokens before first useful code token pops out at 10 tokens/s, that's a 10 minute wait. 

I have shared my performance numbers [here](https://github.com/ikawrakow/ik_llama.cpp/issues/296#issuecomment-2774330966), so not only do I have experience dealing with what you are talking about, but my situation is FAR worse.

>(And if you are throwing money at API endpoints to buy your own life's worth, might as well pay for 2700 elo o4-mini instead of a 2000 elo deepseek) Computing is suppose to accelerate productivity, not waiting for a reasoning models for minutes.

I'm not sure I follow your point. If your goal is to use the best model available, that is often not open-weight so at that point there is no option besides just using an API. So I'm not really sure how local inference software improvements help with that situation where local inference isn't an option.

>If FP4 is guaranteed to be perfect, why waste the joules running 8bits or 16bits? Those Trillion parameter models are not meant to run on SSDs like DeepSeek would like you to believe it can.
> Hence the need to run FP4 perfectly at Petaflops scale, not custom quants non-perfectly at Teraflops scale.

Not sure what you mean by FP4 guaranteed to be perfect, and I'm not sure where the Deepseek team advocated or supported running on SSDs. (All the officially recommended inference software is GPU only or GPU focused).

The FP4 model you linked is a lossy quantization of Deepseek, and thus could easily be considered a custom quant of Deepseek.

If I wanted that PR here, I would port it, test it, and then make a PR. Otherwise you are just waiting and hoping someone else cares enough to do the steps listed above.

---

👤 **ikawrakow** commented the **2025-06-15** at **14:36:25**:<br>

> Is there anyway for ik to accept this pull from the main branch? 

@whatever1983 

Are you asking if I would accept a PR adding `fp8` support if you prepared one for `ik_llama.cpp` based on the linked PR in `llama.cpp`?

---

👤 **whatever1983** commented the **2025-06-15** at **14:49:50**:<br>

@ikawrakow 

Might need some minor mods.  The code in llama.cpp main branch seems decent.  Besides the GGML version difference, why don't you try a merge first?   At least the conversion scripts all work.  Running FP8 on 40 series and 50 series need additional CUDA code.   Running on CPU needs BF16 casts

All I am saying is that at least the repo maintainer  needs to be willing to accept the importance of those data formats.  Because current/future hardware can do Petaflops on those formats.  B200/GB10 and recently announced MI350X and the 432GB MI450X in 2026 can run the FP4 in a single GPU FP4 accelerated.  You need to be forward looking.