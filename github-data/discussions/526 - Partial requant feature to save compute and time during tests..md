### ✨ [#526](https://github.com/ikawrakow/ik_llama.cpp/discussions/526) - Partial requant feature to save compute and time during tests.

| **Author** | `Nexesenex` |
| :--- | :--- |
| **Created** | 2025-06-13 |
| **Updated** | 2025-07-13 |

---

#### Description

Hey,

Could it be possible to have a partial requant feature?

For (a generic) example, one quantizes a IQ2_KT .gguf, but with ffn_down in IQ2_S and the output in IQ5_KS_R4.
Then, one wants to requantize the same model with the same IQ2_KT broad quant strategy, but with ffn_down in IQ3_XXS and the output in IQ5_K.

Could a feature be implemented so the first quantized model is used as a secondary source to the original source, in order import all the already quantized tensors in IQ2_KT from this secondary source, copy them in the destination .gguf, and only requantize from the original source those tensors which the type has been changed in the quantization command?

That could save a lot of time and compute during tests.

---

#### 🗣️ Discussion

👤 **saood06** replied the **2025-06-13** at **12:49:01**:<br>

People do similar things a lot by making scripts that leveraging gguf-py. (Some notable examples was updating the gemma QAT to use use Q6_K instead of fp16 for the embeddings table, manually making deepseek R1-T chimera from a V3 and R1 GGUF, etc.).

I've thought to add support to the C/C++ code to do this, but it seems unnecessary given how flexible gguf-py is.

There has been effort made to keep gguf-py current with all the quant types (see #458 and #298).

---

👤 **ikawrakow** replied the **2025-06-13** at **12:53:13**:<br>

It would be useful, right? When I'm actively experimenting with quantization mixes I wish I had this feature. But implementing it basically means to re-implement quantization, so I have not done it.

The alternative is to run a second quantization where only the tensors that you want to change are quantized (using `--custom-q`), and then, as @saood06 mentions, use gguf-py to stitch the two models together (although, I don't think there is an easy out-of-the-box way of doing, or is there?)

> 👤 **Nexesenex** replied the **2025-06-13** at **12:59:46**:<br>
> Well, I'm not well versed in gguf.py, so I'd trust Saood's word on that.
> It seems to be quite the hassle still, and a proper and straight implementation of such feature would imho be critically important, because it would save time, which is irrecoverable, and compute/money/natural resources, which are not infinite for either one, either all.
> 
> 👤 **saood06** replied the **2025-06-13** at **13:01:45**:<br>
> >(although, I don't think there is an easy out-of-the-box way of doing, or is there?)
> 
> A script that does so would really not be that difficult to make especially if you reference the existing ones (that are designed for specific one-off situations).
> 
> I do think it is trivial enough where it is very likely of the smaller coding oriented models could one-shot a working version (especially if given the references of the notable examples mentioned above).
> 
> I do think a polished version would make sense in `gguf-py/scripts` if one gets made and wants to be shared. I haven't done that with any of the one's I have seen in the wild or made myself as they are not generic and handle very specific needs.
> 
> 👤 **saood06** replied the **2025-06-13** at **13:15:09**:<br>
> I have actually thought about this before, and thought the most polished version would be to add this functionality both as a standalone script (taking in some regex similar to `--custom-q`, `-ot`,  `--repack-pattern`, etc.) and in the GGUF Editor GUI : https://github.com/ggml-org/llama.cpp/pull/12930 (which has yet to be ported here).
> 
> I never did it as it was always so easy to make one-off scripts for my gguf-py needs, and I thought it wasn't something that many other people would care about or use, but I guess I was wrong.
> 
> 👤 **Nexesenex** replied the **2025-06-13** at **14:20:40**:<br>
> Well, we are actually several folks testing new quants on different models, and so the idea might be quite popoular, ideally in C or at least in Python. I'll try by myself if no one comes with an out of the box solution, but need to read all those references and understand more about what I'm getting into, because I'm far far behind you guys about the know-how.
> 
> 👤 **saood06** replied the **2025-06-13** at **14:48:47**:<br>
> > Well, we are actually several folks testing new quants on different models, and so the idea might be quite popoular, ideally in C or at least in Python.
> 
> Yeah. I floated this idea a long time ago to a certain quant maker (who pumps out a lot of quants) as it would (and still could) save them a lot of wasted compute, but this was before I knew about gguf-py.
> 
> >I'll try by myself if no one comes with an out of the box solution
> 
> Nice, if you don't end up getting something working by the time I finish up polishing the frontend I use to be good enough for a public release I'll do it.
> 
> >but need to read all those references and understand more
> 
> Here's two I mentioned, [QAT embed swap](https://huggingface.co/stduhpf/google-gemma-3-27b-it-qat-q4_0-gguf-small/blob/main/swap_embeds.py), [DIY chimera merge](https://gist.github.com/city96/a05cb7ec6664a5085efb007497f2049b). I know I've seen more but these are the first two that came to mind.
> 
> 👤 **saood06** replied the **2025-06-13** at **15:03:21**:<br>
> Also I just remembered there was another hacky idea I had to do this which involved abusing the gguf-split system to isolate any tensors you want to experiment with which would allow you to swap them out (and test many combinations). 
> 
> The best implementation of this could in theory minimize both the amount of space taken (should be easy) and the amount of files written to (this seems like it would be much more difficult, quantizing only select tensors with gguf-py might not be too bad, but that would limit it to only the tensors it can quantize to, and doing it with `quantize.cpp` means adding that functionality to it which may be difficult).
> 
> 👤 **Nexesenex** replied the **2025-06-13** at **16:10:32**:<br>
> > Also I just remembered there was another hacky idea I had to do this which involved abusing the gguf-split system to isolate any tensors you want to experiment with which would allow you to swap them out (and test many combinations).
> > 
> > The best implementation of this could in theory minimize both the amount of space taken (should be easy) and the amount of files written to (this seems like it would be much more difficult, quantizing only select tensors with gguf-py might not be too bad, but that would limit it to only the tensors it can quantize to, and doing it with `quantize.cpp` means adding that functionality to it which may be difficult).
> 
> Lol, I was just thinking about this 1h ago. (Why don't I simply split the gguf in as many tensor as there is..), and then it becomes a matter of naming. I was contemplating over that a long time ago already, tensor-series based gguf, gguf as directory and so on. But actually, it can already be tried as things are.

---

👤 **saood06** replied the **2025-07-12** at **21:45:17**:<br>

@Nexesenex 

Have you seen this: https://github.com/Thireus/GGUF-Tool-Suite? I haven't fully gone through the code yet, but I think it seems to accomplish at least some of the goals you described here (taking the path of using the gguf-split system).

> 👤 **Nexesenex** replied the **2025-07-12** at **22:04:37**:<br>
> > @Nexesenex
> > 
> > Have you seen this: https://github.com/Thireus/GGUF-Tool-Suite? I haven't fully gone through the code yet, but I think it seems to accomplish at least some of the goals you described here (taking the path of using the gguf-split system).
> 
> You will laugh. I discovered his fork of IKL today, and didn't discover yet his tools suite. Thanks for the heads-up, I will dive into it asap! :)
> 
> 👤 **saood06** replied the **2025-07-12** at **23:30:04**:<br>
> >Thanks for the heads-up, I will dive into it asap! :)
> 
> Let me know your thoughts, e.g. if it does meet your goals, will you use it, will you change/fork it, etc.
> 
> 👤 **Nexesenex** replied the **2025-07-13** at **02:32:53**:<br>
> > > Thanks for the heads-up, I will dive into it asap! :)
> > 
> > Let me know your thoughts, e.g. if it does meet your goals, will you use it, will you change/fork it, etc.
> 
> Sure thing.