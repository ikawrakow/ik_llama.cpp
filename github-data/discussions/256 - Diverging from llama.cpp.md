### 🗣️ [#256](https://github.com/ikawrakow/ik_llama.cpp/discussions/256) - Diverging from llama.cpp

| **Author** | `arnfaldur` |
| :--- | :--- |
| **Created** | 2025-03-14 |
| **Updated** | 2025-03-14 |

---

#### Description

I just discovered this fork yesterday and would like to understand the situation better. This message is addressed to @ikawrakow

I was very excited to discover that you were still innovating on quantizations but I'm confused as to why it's happening on a fork with little desire (https://github.com/ikawrakow/ik_llama.cpp/issues/133) to upstream the developments. I researched the history of this fork and many of the discussions that lead to it's creation (like the curiosity about Justine's tinyBLAS doubts), but have still not found a satisfactory answer.

## Underutilization

The **very impressive** developments occurring on this fork seem to me to be underutilized. The `llama.cpp` community is huge and all those people could be enjoying the new `IQn_K` quants. But as it stands, most people don't know about them. Bartowski and his peers aren't uploading `IQn_K` quants to hugging face, and even if someone were to go through the effort of making them themselves, using them is considerably harder as there are no build instructions here, and the build process has changed upstream.

There is of course the possibility that you don't care about mass adoption of your quants, in which case the last paragraph isn't relevant. I completely respect that disposition, if that is the case.

I would be surprised if that was the case however. Why share the work on this fork if not for others to use? A potential answer would be that you prefer a smaller, more technical community that is less concerned about mass adoption and compatibility. That is certainly valid but there are some downsides, e.g. no Bartowski quants, slower support for new models, and no development of secondary tools like the server. You might not care about those things either. I do, but I can also solve them myself with mild effort.

## The quants of `llama.cpp`

A defining feature of llama.cpp is it's popular model format and it's supported quantizations. I know that many people always wait for Bartowski's speedy quantizations for new models and pick their preferred quants from there, just like I do. As I understand it you contributed every one of these quantization schemes, many of wich were SOTA or near SOTA at the time of publishing. In light of that, your efforts were instrumental in making `llama.cpp` into what it is today. Especially considering that quantization quality is probably the most important aspect of running models in RAM constrained environments, which is the point of `llama.cpp`.

As is likely evident, I think it is a big loss to the commons that these new quants and optimizations aren't available upstream. 

I still want to emphasize that I believe that there is a valid reason for the fork's creation and I would be very interested in hearing that reason.

## Resolution

In light of the importance of the past contributions to `llama.cpp`, I also want to know if you would ever consider upstreaming them, and importantly, under what conditions you would be willing to do that. The maintainers of `llama.cpp` should see the value in the work on this fork and want to get it upstreamed and I hope that they would be willing to accommodate you and do what ever it takes to make you happy to contribute.

I'm sorry if this is a bit much, but I think it's very important and I was honestly shocked to discover this and that nobody is talking about this. Maybe I care more about quants than most `llama.cpp` users 🤷

---

#### 🗣️ Discussion

👤 **ikawrakow** replied the **2025-03-14** at **06:06:08**:<br>

Hello @arnfaldur, 

I'm hacking here to keep my brain utilized and to have some fun. Definitely not looking for fame and/or mass adoption of this repository. A few people have found it useful, this is good enough for me (and, if it did become popular, I'm not sure I want to spend my time supporting non-technical users). I will not be upstreaming stuff to `llama.cpp`, but obviously with this repo being MIT licensed, upstream is free to take from here whatever they find useful. In addition to the `IQX_K` quants, there are a lot of things here that are better than upstream. In no particular order
* CPU Flash Attention implementation that, unlike upstream, actually improves performance. By quite a margin for very long contexts. Oh, it also works for models where the K head size is different from the V head size (DeepSeek models)
* GPU Flash Attention for different K and V head sizes
* MLA in 2 variants, very relevant for DeepSeekV3/R1 CPU and GPU inference
* What I believe are the fastest quantized matrix multiplications on the planet
* Row interleaving for (almost) all quantization types, which leads to much better CPU performance. Upstream has some of that, but just for `IQ4_0`, `Q8_0`, and `IQ4_NL`, but even for those performance here is quite a bit better, even on `ARM` CPUs.
* Selective tensor offloading to the GPU. Very useful when the model does not fit in VRAM, and one can offload specific tensors to the GPU(s). This replicates what KTransformers have done
* Support for Bitnet models with much better performance than `llama.cpp` and even the 12k stars Bitnet repository from Microsoft
* Much more comprehensive `bf16` support. CUDA support for `bf16` was added not too long ago in upstream, but mine beats it by a factor of 2 for prompt processing
* Various fused operations. This includes fusing of experts (relevant for MoE models). Gemma2 performance is quite a bit better than upstream because of that on CPU, GPU, Metal (but I guess this is no longer relevant with Gemma3 now released)
* Support for custom quantization schemes

---

👤 **bitbottrap** replied the **2025-03-14** at **14:40:37**:<br>

I completely agree that some of this stuff needs to get into llama.cpp. And I completely understand why ikawrakow does not want to be personally responsible for it.

I'm not sure what the focus is over there in llama.cpp land but it's very active. I just don't see a lot of the core stuff being improved on like it is here.