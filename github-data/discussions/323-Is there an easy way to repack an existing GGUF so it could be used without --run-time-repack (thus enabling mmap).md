### 🗣️ [#323](https://github.com/ikawrakow/ik_llama.cpp/discussions/323) - Is there an easy way to repack an existing GGUF so it could be used without --run-time-repack (thus enabling mmap)

| **Author** | `Lissanro` |
| :--- | :--- |
| **Created** | 2025-04-10 |
| **Updated** | 2025-05-21 |

---

#### Description

DeepSeek-V3-0324-GGUF-UD-Q4_K_XL works great for me when I load it using --run-time-repack, I get more than 7 tokens/s with EPYC 7763 and 1TB of 3200MHz RAM + 4x3090 GPUs. But this unfortunately disables mmap and requires a lot of compute on each reload - and if I need to switch models often in some tasks (for example, a separate model to process input images and describe them, then continue with DeepSeek V3), it slows things down.

So, what I am looking for, is it possible to repack DeepSeek-V3-0324-GGUF-UD-Q4_K_XL offline to a new GGUF which would work well with ik_llama.cpp and I ould load it without the --run-time-repack?

I know there are some existing quants made specifically for ik_llama.cpp, like https://huggingface.co/ubergarm/DeepSeek-V3-0324-GGUF, but I noticed that DeepSeek-V3-0324-GGUF-IQ4_K_R4 for example gives me 4-5 tokens/s at most, my guess because it quantized very differently, even though it has about the same size. This also suggests that creating my own quant  from scratch may be very difficult - not only I have to download the full size models for V3 and R1 (which would take weeks via 4G connection I have), but I also may end up with a quant that does not perform as good as the original Unsloth quant, since I do not have any experience with creating GGUF quants. This is why I would prefer to find a way to repack an existing quant, rather than trying to create one from scratch, if that is possible?

In case it matters, here is the command I use to run the model (I specify only -ctk q8_0 because my understanding -ctv does not have any effect when due to enabled optimizations V cache is not actually used):

```
taskset -c 0-63 ~/pkgs/ik_llama.cpp/build/bin/llama-server \
--model ~/models/DeepSeek-V3-0324-GGUF-UD-Q4_K_XL/DeepSeek-V3-0324-UD-Q4_K_XL-00001-of-00009.gguf \
--ctx-size 81920 --n-gpu-layers 62 --tensor-split 25,25,25,25 \
-mla 2 -fa -ctk q8_0 -amb 2048 -fmoe -rtr \
--override-tensor "ffn_down_exps=CPU, ffn_up_exps=CPU, gate_exps=CPU" \
--threads 64 --host 0.0.0.0 --port 5000
```

This command utilizes about 20GB of VRAM on each 24GB GPU. The main issue is that I am yet to figure out a way how to repack this GGUF so I could run without the -rtr option. I would appreciate any help how to resolve this?

---

#### 🗣️ Discussion

👤 **ikawrakow** replied the **2025-04-10** at **15:31:47**:<br>

You can use
```
./bin/llama-quantize --repack --repack-pattern exps ~/models/DeepSeek-V3-0324-GGUF-UD-Q4_K_XL/DeepSeek-V3-0324-UD-Q4_K_XL-00001-of-00009.gguf repacked_model_file_name q4_k_r4
```
The command will not overwrite the existing model, so you need to have enough free disk space for both models.

In your command that starts the server, you can simplify to
```
--override-tensor exps=CPU
```
It is a regular expression, so it is equivalent to explicitly listing
```
--override-tensor "ffn_down_exps=CPU, ffn_up_exps=CPU, gate_exps=CPU"
```

More generally, you can use `--repack-pattern` in the `llama-quantize` command by simply copying the regular expressions from the `--override-tensor` argument and removing the `=CPU` from it. So,
```
./bin/llama-quantize --repack --repack-pattern "ffn_down_exps,ffn_up_exps,gate_exps" etc.
```
is equivalent.

> 👤 **ikawrakow** replied the **2025-04-10** at **15:36:25**:<br>
> I have never repacked (or quantized) a multi-part GGUF, so I don't know if `llama-quantize` does the right thing to load all parts. In case it does not, you may need to concatenate the parts into a single file
> ```
> cat file1 file2 ... fileN >>combined_file
> ```
> 
> 👤 **saood06** replied the **2025-04-10** at **23:00:39**:<br>
> >In case it does not, you may need to concatenate the parts into a single file
> > 
> > ```
> > cat file1 file2 ... fileN >>combined_file
> > ```
> 
> Files split in the gguf-split way need to be merged via gguf-split.

---

👤 **ubergarm** replied the **2025-04-10** at **22:05:30**:<br>

> I noticed that DeepSeek-V3-0324-GGUF-IQ4_K_R4 for example gives me 4-5 tokens/s at most, my guess because it quantized very differently, even though it has about the same size.

A few thoughts here:

1. My quant was designed to be a bit heavy in the non-routed experts to give better quality output. You can trade-off some quality for extra speed by adding `-ser 6,1` as detailed in [PR#239](https://github.com/ikawrakow/ik_llama.cpp/pull/239).
2. My quant is designed to offload just over 17GiB weights to VRAM plus context cache. However, it looks like you have 96 GB VRAM (4x GPUs?). Using `-ot exps=CPU` shouldn't fill up 20GB VRAM on 4x cards (80GB)?. Designing a quant specific to multiple-gpu setups like yours is more tricky as you want to offload some of the routed `exps` layers which need to be quantized in a way suited for GPU inferencing.

So yeah, like ik mentions, you will want to use `./bin/llama-quantize --repack --repack-pattern "ffn_down_exps,ffn_up_exps,gate_exps" etc.` and figure out ahead of time the size of the tensors/layers you want to offload onto GPU (and don't repack those), and only repack the remaining routed experts `exps` layers going into RAM for CPU inferencing. In other words the repacked `q4_k_r4` is for running on CPU RAM. Don't repack the tensors/layers you're running on GPU.

Haha hope I didn't confuse too much. This is indeed a more straight-forward way than rolling your own quant, which would have the same steps but more.

Cheers!

---

👤 **Lissanro** replied the **2025-04-11** at **10:49:26**:<br>

@ikawrakow 
Thank you, I was able to convert based on the suggested command, but the issue is, performance of the converted quant is very low, so I cannot really use it yet. I would appreciate any help to figure out how to convert it in the same way like -rtr option does, but to a file permanently, so I can use mmap and load without -rtr option.

With the original Unsloth quant and -rtr option, I get more than 7 tokens/s, while with converted quant without -rtr option, I get 4-5 tokens/s. Maybe it converted some tensors to more compute intensive equivalents? Perhaps there are other options besides 

The command I used was:

```
~/pkgs/ik_llama.cpp/build/bin/llama-quantize --repack --repack-pattern exps ~/models/DeepSeek-V3-0324-GGUF-UD-Q4_K_XL/DeepSeek-V3-0324-UD-Q4_K_XL-00001-of-00009.gguf /tmp/DeepSeek-V3-0324-GGUF-UD-Q4_K_R4.gguf q4_k_r4
main: build = 3630 (5f44f4b3)
main: built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu
main: quantizing '/home/lissanro/pkgs/text-generation-webui/models/DeepSeek-V3-0324-GGUF-UD-Q4_K_XL/DeepSeek-V3-0324-UD-Q4_K_XL-00001-of-00009.gguf' to '/mnt/secondary/tmp/DeepSeek-V3-0324-GGUF-UD-Q4_K_R4.gguf' as Q4_K_R4
llama_model_loader: additional 8 GGUFs metadata loaded.
...
```

Here is full conversion log which includes all the output during the conversion:
https://pastebin.com/P7QEQsKy

Three runs using the original Unsloth quant with -rtr option (timings line only for each run):

```
INFO [           print_timings] generation eval time =   31669.99 ms /   230 runs   (  137.70 ms per token,     7.26 tokens per second) | tid="128283826724864" timestamp=1744362775 id_slot=0 id_task=0 t_token_generation=31669.991 n_decoded=230 t_token=137.69561304347826 n_tokens_second=7.262395496102289
INFO [           print_timings] generation eval time =   37422.90 ms /   273 runs   (  137.08 ms per token,     7.29 tokens per second) | tid="128283826724864" timestamp=1744362939 id_slot=0 id_task=232 t_token_generation=37422.898 n_decoded=273 t_token=137.08021245421247 n_tokens_second=7.2949989068190275
INFO [           print_timings] generation eval time =   39311.07 ms /   297 runs   (  132.36 ms per token,     7.56 tokens per second) | tid="128283826724864" timestamp=1744364349 id_slot=0 id_task=507 t_token_generation=39311.072 n_decoded=297 t_token=132.36051178451177 n_tokens_second=7.555123401366415
```

Three runs using the same prompt with the converted quant (without the -rtr option):

```
INFO [           print_timings] generation eval time =   67077.44 ms /   287 runs   (  233.72 ms per token,     4.28 tokens per second) | tid="140159021387776" timestamp=1744366116 id_slot=0 id_task=0 t_token_generation=67077.444 n_decoded=287 t_token=233.71931707317074 n_tokens_second=4.278636496644088
INFO [           print_timings] generation eval time =   67416.24 ms /   342 runs   (  197.12 ms per token,     5.07 tokens per second) | tid="140159021387776" timestamp=1744366192 id_slot=0 id_task=289 t_token_generation=67416.242 n_decoded=342 t_token=197.12351461988303 n_tokens_second=5.072961497913218
INFO [           print_timings] generation eval time =   76603.74 ms /   303 runs   (  252.82 ms per token,     3.96 tokens per second) | tid="140159021387776" timestamp=1744366731 id_slot=0 id_task=633 t_token_generation=76603.741 n_decoded=303 t_token=252.81762706270626 n_tokens_second=3.955420401726856
```

---

👤 **Lissanro** replied the **2025-04-11** at **10:52:18**:<br>

@saood06 
It seems my own quant converted from the Unsloth one also loses a lot of performance, so it may not be something specific to your quant. I am not sure what the issue is yet. It is worth mentioning that my EPYC 7763 64-core CPU is under full load during inference with either quant, so my guess something in the converted quants hits CPU bottleneck, which is not present when using Unsloth quant with -rtr option.

As of VRAM usage, I think it depends on context length. To be more precise, with 80K context I get around 19 gigabytes VRAM utilization on each GPU, so around 76-80 VRAM usage in total. If I try to increase context size too much, I get CUDA OOM errors, confirming it is using VRAM for context.

Maybe I could put some additional ffn_down_exps, ffn_up_exps or ffn_gate_exps on each GPU, but not sure which of them is more beneficial to put in VRAM yet. I already experimented with blk.3.ffn_gate_exps=CUDA0, ... and so on, but since I cannot put too many of them due to having not that much VRAM free, I did not notice difference in performance. I did not try with non-gate ones yet.

With my workflow that involves loading 72B vision model in VRAM, processing images, then load V3, not being able to get mmap working with good performance is the biggest bottleneck at the moment. I am still trying to figure out if there are options I could try to achieve the same kind of conversion -rtr option does, to create a new GGUF that would work the same in terms of performance but would not require -rtr anymore.

---

👤 **ikawrakow** replied the **2025-04-11** at **10:58:58**:<br>

The offline repacking command should produce a result that is 100% equivalent to what happens with online repacking.

But the two runs will not be equivalent as memory will be allocated and assigned to tensors in a different way. I have seen performance differences between offline and online repacking on my hardware, but never as large as you are reporting.

Can you try dropping caches before using the offline repacked model?
```
echo 3 | sudo tee /proc/sys/vm/drop_caches
```

---

👤 **ikawrakow** replied the **2025-04-11** at **11:10:46**:<br>

> Maybe I could put some additional ffn_down_exps, ffn_up_exps or ffn_gate_exps on each GPU, but not sure which of them is more beneficial to put in VRAM yet. I already experimented with blk.3.ffn_gate_exps=CUDA0, ... and so on, but since I cannot put too many of them due to having not that much VRAM free, I did not notice difference in performance. I did not try with non-gate ones yet.

If you have spare VRAM, the best strategy is to put the `ffn_up_exps` and `ffn_gate_exps` of a given number of layers in VRAM (how many layers depends on how much VRAM you have left and how big the tensors are). This brings more benefit than putting just one of the experts tensors or all 3 of the experts tensors, especially when you are using `-fmoe`. I'm currently running some experiments with LlaMA-4-Scout on my low-end  hardware (Ryzen-5975WX + RTX 4080), and I use
```
-ot "blk\.[0-9]\.ffn_up_exps=CUDA0,blk\.[0-9]\.ffn_gate_exps=CUDA0,blk\.1[0-9]\.ffn_up_exps=CUDA0,blk\.1[0-9]\.ffn_gate_exps=CUDA0,exps=CPU" -ngl 100
```
to have all attention and shared experts tensors plus the first 20 layers of `ffn_up_exps` and `ffn_gate_exps` on the GPU, with all remaining experts on the CPU.

---

👤 **Lissanro** replied the **2025-04-11** at **11:35:48**:<br>

First, I load the repacked model with -rtr option - obviously should be unnecessary, but I was curious if it makes a difference, and to my surprise, it did, I got good performance again (full log: https://pastebin.com/5d6R2GDG):

```
INFO [           print_timings] generation eval time =   46791.42 ms /   341 runs   (  137.22 ms per token,     7.29 tokens per second) | tid="127320811921408" timestamp=1744369176 id_slot=0 id_task=0 t_token_generation=46791.423 n_decoded=341 t_token=137.2182492668622 n_tokens_second=7.287660390238612
INFO [           print_timings] generation eval time =   36683.23 ms /   274 runs   (  133.88 ms per token,     7.47 tokens per second) | tid="127320811921408" timestamp=1744369220 id_slot=0 id_task=343 t_token_generation=36683.233 n_decoded=274 t_token=133.88041240875913 n_tokens_second=7.469352551341372
```

Then, I ran `echo 3 | sudo tee /proc/sys/vm/drop_caches`, this left me with 704 GB of memory free of cache. I also have no swap file and my system has 1TB of RAM in total, so plenty of memory for 378GB quant (the size of the converted quant). After it fully loaded, I still have 322GB of completely free memory. But, the performance become quite bad (from almost 7.5 tokens/s down to less than 4 tokens/s; full log: https://pastebin.com/K4PYP52t):

```
INFO [           print_timings] generation eval time =   75071.14 ms /   270 runs   (  278.04 ms per token,     3.60 tokens per second) | tid="140708181868544" timestamp=1744369869 id_slot=0 id_task=0 t_token_generation=75071.144 n_decoded=270 t_token=278.04127407407407 n_tokens_second=3.5965883242701087
INFO [           print_timings] generation eval time =   73892.48 ms /   268 runs   (  275.72 ms per token,     3.63 tokens per second) | tid="140708181868544" timestamp=1744369983 id_slot=0 id_task=272 t_token_generation=73892.479 n_decoded=268 t_token=275.7182052238806 n_tokens_second=3.626891445880439
```

I tried adding --mlock, but the performance did not improve much (still was getting at most 4-5 tokens/s no matter how many times I tried).

Since -rtr option disables mmap, I decided to disable it explicitly with --no-mmap and run without -rtr option, to see if it is mmap that ruins the performance:

```
INFO [           print_timings] generation eval time =   42764.35 ms /   314 runs   (  136.19 ms per token,     7.34 tokens per second) | tid="129645145907200" timestamp=1744370957 id_slot=0 id_task=0 t_token_generation=42764.346 n_decoded=314 t_token=136.19218471337578 n_tokens_second=7.342565229455397
```

...and with the repacked quant and --no-mmap option, performance was back to normal. So, it seems something about mmap that drastically reduces performance. Nothing wrong with the quant file then. Very strange. In theory, I would expect the performance to be about the same, since either way the same memory is used and I have plenty of it free.

Please let me know if there are some kind of performance profiling or additional logging I could do on my side.

As of putting more ffn_up_exps and ffn_gate_exps on GPU, I will try that with as much layers as I can, thank you very much for the suggestion.

> 👤 **ubergarm** replied the **2025-04-11** at **14:20:23**:<br>
> @Lissanro 
> 
> > --no-mmap option, performance was back to normal. So, it seems something about mmap that drastically reduces performance. Nothing wrong with the quant file then.
> 
> If you are benchmarking while using mmap, you have to throw away the first full run results typically as the benchmarks start running before the model is loaded into page cache. You can check by watching your disk i/o and `cached` inside of `btop`. You will notice with mmap disabled, it takes longer to start up and finish allocating the entire model into RAM. When using mmap, it starts much quicker but runs slower in the beginning. This is normal expected behavior for all inference engines I've used.
> 
> Also, depending on how your system is configured, when not using mmap() you may be taking advantage of transparent huge pages automatically under the hood. You can check that with `numastat -m -p $(pidof llama-server)` or llama-bench etc... It seems to be system dependent on how this effects performance.
> 
> Keep us posted once you come up with a multi-gpu command line to override `ffn_up_exps` and `ffn_gate_exps` tensors onto each GPU as ik mentions above. I wanted to document that somewhere to help others as many of the questions I see are how to use more VRAM correctly when using `-ot`.
> 
> Thanks!
> 
> 👤 **ubergarm** replied the **2025-04-11** at **19:08:55**:<br>
> @Lissanro 
> 
> Also, using the above examples I'm slowly learning how to better use `-ot` myself. I have a few examples now on [discussion #258](https://github.com/ikawrakow/ik_llama.cpp/discussions/258#discussioncomment-12807746) which you could use to target `CUDA0` `CUDA1` etc to craft the best command for your rig.

---

👤 **Lissanro** replied the **2025-04-13** at **03:57:01**:<br>

I was able to achieve similar speed with mmap after resetting my BIOS, and changing only absolutely necessary settings. Before that, no matter what I did, it ran at 30%-50% reduced speed. Not sure exactly what setting was messing up results, maybe performance tuning settings for memory throughput.

But all good now, this is my current performance with mmap enabled using repacked quant (this is with around 2.5K token long fill in the context window):

```
INFO [           print_timings] generation eval time =    1400.35 ms /    11 runs   (  127.30 ms per token,     7.86 tokens per second) | tid="124902137237504" timestamp=1744499973 id_slot=0 id_task=835 t_token_generation=1400.348 n_decoded=11 t_token=127.30436363636363 n_tokens_second=7.85519028127294
```

With 32K filled, I get lesser performance but still good:

```
INFO [           print_timings] generation eval time =   76081.15 ms /   387 runs   (  196.59 ms per token,     5.09 tokens per second) | tid="132320194224128" timestamp=1744494220 id_slot=0 id_task=2362 t_token_generation=76081.154 n_decoded=387 t_token=196.5921291989664 n_tokens_second=5.086673632736959
```

I did not save exact stats for 64K+ context fill, but it was slightly above 3 tokens/s for output. Input generally was within 50-80 tokens/s range. Reloading model with mmap enabled takes about 45 seconds, which is great.

My final command to repack R1 and V3 was like this:

```
~/pkgs/ik_llama.cpp/build/bin/llama-quantize --repack \
--repack-pattern "(^blk\.[7-9]|\d\d).ffn_(up|gate)_exps|ffn_down_exps" \
/mnt/secondary/neuro/DeepSeek-R1-GGUF_Q4_K_M-163840seq/DeepSeek-R1-Q4_K_M-00001-of-00011.gguf \
/home/lissanro/neuro/DeepSeek-R1-GGUF_Q4_K_M-163840seq/DeepSeek-R1-GGUF_Q4_K_M_R4.gguf \
q4_k_r4
```

The pattern in llama-quantize crafted in a way that avoids repacking tensors I intent to use on GPUs. This the command I use to run it:

```
taskset -c 0-63 ~/pkgs/ik_llama.cpp/build/bin/llama-server \
--model /home/lissanro/neuro/DeepSeek-R1-GGUF_Q4_K_M-163840seq/DeepSeek-R1-GGUF_Q4_K_M_R4.gguf \
--ctx-size 73728 --n-gpu-layers 62 --tensor-split 25,25,25,25 -mla 2 -fa -ctk q8_0 -amb 1024 -fmoe \
-ot "blk\.3\.ffn_up_exps=CUDA0, blk\.3\.ffn_gate_exps=CUDA0" \
-ot "blk\.4\.ffn_up_exps=CUDA1, blk\.4\.ffn_gate_exps=CUDA1" \
-ot "blk\.5\.ffn_up_exps=CUDA2, blk\.5\.ffn_gate_exps=CUDA2" \
-ot "blk\.6\.ffn_up_exps=CUDA3, blk\.6\.ffn_gate_exps=CUDA3" \
-ot "ffn_down_exps=CPU, ffn_up_exps=CPU, gate_exps=CPU" \
--threads 64 --host 0.0.0.0 --port 5000
```

I also noticed that I need to specify CPU overrides last rather than first for CUDA overrides to have an effect. I used multiple -ot arguments since a single one could not understand multi-line format, but with many -ot, I can use multiple lines in my script for better readability. Putting ffn_up_exps and ffn_gate_exps from blocks 3-6 on my GPUs (one pair per GPU) is all that I could fit, I had even reduce context length to 72K (73728).

Thank you so very much, @ikawrakow and @ubergarm , for helping me to figure this out!

---

👤 **Ph0rk0z** replied the **2025-05-17** at **18:57:32**:<br>

So to repack I do inverse of my cuda regex? Can quant type also be converted? Or does it just become same_R4? MMAP or not, the entire model gets cached on my system, at least for qwen 235b sizes.

---

👤 **Lissanro** replied the **2025-05-21** at **05:27:22**:<br>

@Ph0rk0z 
You need to craft a regex for R4 repacking happen in way that covers all tensors you plan to keep on CPU, but does not affect tensors that you plan running on GPU (GPU tensors need to be kept non-R4). You can refer to regexes in my previous message to see how repack regex differs.

> 👤 **Ph0rk0z** replied the **2025-05-21** at **11:25:07**:<br>
> Yea I assume it's just see which layers are on GPU and then exclude them. So if you pick 1,2,3,4 make a not 1,2,3,4 regex. Funny enough we have AI for this. But I have IQ4_XS, so what does that become? IQ4_XS_R4? Or can it repack to something else?
> 
> 👤 **ikawrakow** replied the **2025-05-21** at **11:29:29**:<br>
> > Or can it repack to something else?
> 
> No. The repacking is only to the  corresponding row-interleaved type. Repacking to something else would result in quality loss.