### 🗣️ [#286](https://github.com/ikawrakow/ik_llama.cpp/discussions/286) - Testing `deepseek-ai/DeepSeek-V3-0324` model support.

| **Author** | `ubergarm` |
| :--- | :--- |
| **Created** | 2025-03-24 |
| **Updated** | 2025-04-02 |

---

#### Description

I saw today a new model [deepseek-ai/DeepSeek-V3-0324](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324) that may run on this fork?

Zero pressure for anyone to spend time on this, just experimenting to satisfy my curiosity.

I figure might as well download it and see if it magically "just works" using my [existing R1 custom quant procedure](https://github.com/ikawrakow/ik_llama.cpp/discussions/258).

The main two issues I imagine might crop up without knowing anything:
* Might need a special imatrix file (maybe [this one from mradermacher](https://huggingface.co/mradermacher/DeepSeek-V3-i1-GGUF/resolve/main/imatrix.dat) for earlier V3 will still work?)
* 14B of the Multi-Token Prediction (MTP) Module weights

> 5.4.3. Multi-Token Prediction Evaluation
Instead of predicting just the next single token, DeepSeek-V3 predicts the next 2 tokens through
the MTP technique. Combined with the framework of speculative decoding (Leviathan et al.,
2023; Xia et al., 2023), it can significantly accelerate the decoding speed of the model. A natural
question arises concerning the acceptance rate of the additionally predicted token. Based on
our evaluation, the acceptance rate of the second token prediction ranges between 85% and 90%
across various generation topics, demonstrating consistent reliability. This high acceptance rate
enables DeepSeek-V3 to achieve a significantly improved decoding speed, delivering 1.8 times
TPS (Tokens Per Second). -https://arxiv.org/pdf/2412.19437

Well, I'll update this discussion after it finishes downloading and I give it the old college try haha...

Curious if anyone else has any luck and if this new model is "better" at coding like some are speculating over on [r/LocalLlama](https://www.reddit.com/r/LocalLLaMA/comments/1jisuq4/deepseek_v30324_has_caught_up_to_sonnet_37_in_my/)... Who knows!

---

#### 🗣️ Discussion

👤 **saood06** replied the **2025-03-24** at **22:03:22**:<br>

> I saw today a new model [deepseek-ai/DeepSeek-V3-0324](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324) that may run on this fork?
>[...]
>I figure might as well download it and see if it magically "just works" using my https://github.com/ikawrakow/ik_llama.cpp/discussions/258.

The config.json is the same (same architecture/same config) so ik_llama.cpp will behave the same (besides the updated weights, which affect output). This is just another finetune. 

There are cases where finetuned model does change the config (see Qwen with the base being 128K , and the instruct tunes being only 32k with them recommending: "To handle extensive inputs exceeding 32,768 tokens, we utilize [YaRN](https://arxiv.org/abs/2309.00071), a technique for enhancing model length extrapolation, ensuring optimal performance on lengthy texts.", but this is not one of those cases, and even in that cases the finetune did not change the architecture (which is what matters for conversion) just config.



> The main two issues I imagine might crop up without knowing anything:
> 
>     * Might need a special imatrix file (maybe [this one from mradermacher](https://huggingface.co/mradermacher/DeepSeek-V3-i1-GGUF/resolve/main/imatrix.dat) for earlier V3 will still work?)
> 
>     * 14B of the Multi-Token Prediction (MTP) Module weights
> 

For the first point, the linked imatrix will work but I do not recommended it as even though that imatrix was generated on the same model type and so it will apply, the model weights are different and that affects the imatrix data. (Edit: The mradermacher team is already working on quanting and imatrixing that model)

The second point, those weights were present in the other releases such as V3, V3-BASE, and R1, and the conversion just does not include them as llama.cpp and ik_llama.cpp both have do not support the MTP, it is a similar situation with what happened with the MLA tensors, where once support was added the conversion script was updated to include them which required reconverting.

> Curious if anyone else has any luck and if this new model is "better" at coding like some are speculating over on [r/LocalLlama](https://www.reddit.com/r/LocalLLaMA/comments/1jisuq4/deepseek_v30324_has_caught_up_to_sonnet_37_in_my/)... Who knows!

I'm curious, and will have to make room for it on my server. I know this is slightly off topic but I'd be curious to hear your experience with this (and any of the other Deepseek models you've tried).

> 👤 **ubergarm** replied the **2025-03-25** at **00:00:24**:<br>
> > This is just another finetune.
> 
> Great, might have a chance at getting it to work!
> 
> > For the first point, the linked imatrix will work but I do not recommended it 
> 
> I see, thanks for the tip. I see now some discussions from over a year ago about making imatrix files and will give it a go.
> 
> >  The mradermacher team is already working on quanting and imatrixing that model
> 
> Ahh yes, I see [mradermacher/DeepSeek-V3-0324-GGUF](https://huggingface.co/mradermacher/DeepSeek-V3-0324-GGUF) is rolling in as we speak! I'm almost done with the `fp8` and will make the `bf16` GGUF from that. Not sure how long generating an imatrix will take, but might have something working by end of tomorrow if it goes smoothly!
> 
> > your experience with this (and any of the other Deepseek models you've tried)
> 
> Yeah will keep you posted with new V3. I'm only now experimenting with using longer context ~30-40k by copy pasting in code, man pages, documentation, etc.  Using R1 at `Q4` today I was trying to understand how to potentially have `llm_load_tensors()` allocate N copies of ctx_buffs (one on each N NUMA nodes). It helped me understand a bit more the relationship between `src/llama.cpp` and `ggml/src/ggml-backend.c`, but didn't give magic working code haha... It did help me update `CMakeLists.txt` to get it building linking with libnuma library. I've also had some luck with it refactoring python code especially creating uniform style comments and adding static typing. Even QwQ-32B could write a decent 1-shot flappy bird when given a detailed prompt to follow haha... 
> 
> One supposed success story is about [airbnb refactoring javascript test code](https://medium.com/airbnb-engineering/accelerating-large-scale-test-migration-with-llms-9565c208023b) to use a different library. Hard to say how much "tech debt" was incurred if any, but I too am curious to hear of any successful uses of ai for actually useful coding.
> 
> 👤 **saood06** replied the **2025-03-25** at **02:39:47**:<br>
> > > This is just another finetune.
> > 
> > Great, might have a chance at getting it to work!
> > 
> 
> I'd be more surprised if it didn't work for you.
> 
> 
> > > For the first point, the linked imatrix will work but I do not recommended it
> > 
> > I see, thanks for the tip. I see now some discussions from over a year ago about making imatrix files and will give it a go.
> > 
> > > The mradermacher team is already working on quanting and imatrixing that model
> > 
> > Ahh yes, I see [mradermacher/DeepSeek-V3-0324-GGUF](https://huggingface.co/mradermacher/DeepSeek-V3-0324-GGUF) is rolling in as we speak! I'm almost done with the `fp8` and will make the `bf16` GGUF from that. Not sure how long generating an imatrix will take, but might have something working by end of tomorrow if it goes smoothly!
> 
> Have you decided on a calibration dataset? The discussions  [here](https://github.com/ikawrakow/ik_llama.cpp/pull/185#issuecomment-2640263710) and [here](https://github.com/ikawrakow/ik_llama.cpp/discussions/140#discussioncomment-12126789) on the nature of an MoE  model and size of the calibration dataset might be interesting to you, this was also discussed by the mradermacher team [here](https://huggingface.co/mradermacher/BabyHercules-4x150M-GGUF/discussions/3#6758d52499eea0c4b65d0475) [I think there may be other times I've seen it discussed, but don't recall exactly where so that I can link them to you]). I know bartowski's calibration dataset is public, but the longer team mradermacher dataset is not (but they do proactively imatrix a lot of quants, and I've never seen them deny a request to imatrix a model, so it not being public doesn't matter as much).
> 
> Also this https://github.com/ikawrakow/ik_llama.cpp/pull/250 if you haven't seen it is obviously relevant to you, (team mradermacher use a very lightly modified llama.cpp and not ik_llama.cpp).
> 
> The time it takes is obviously dependent on your hardware, I know for team mradermacher it takes them around 20 hours for these large Deepseek models but their situation is FAR from ideal (which is just another reason why I'm so grateful for them). They RPC multiple machines together, and also have to enable GGML_CUDA_ENABLE_UNIFIED_MEMORY and they state "Keep in mind that half the performance is lost due to using GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 while allocating memory far larger than the available GPU memory instead of -ngl 0 duetoi (sic) this not beeing (sic) supported for RPC servers."
> 
> I should probably queue my download (and will finally test the triton dequant method myself), so that I'm ready by the time I have access to imatrix.dat files.
> 
> > Yeah will keep you posted with new V3. 
> 
> Thanks.
> 
> >I'm only now experimenting with using longer context ~30-40k by copy pasting in code, man pages, documentation, etc. 
> 
> My local machine's PP is so slow, which is why I use free cloud hosted access for general and technical things (it also is more convenient as loading the model in takes me ~30 minutes), I only use my server for more creative tasks where I look at the top 10 token probs for many of the tokens generated and manually steer the model.
> 
> I don't remember exactly how much I used V3, but I know I very briefly tried V3-Base and it catastrophically failed a test prompt I'd given it, and had no desire to use it ever since, V3 I remember trying more, but wasn't really impressed and with how slow inference of it was at the time for me, it felt like for creative tasks it was just a worse Mistral Large 2407 (both sharing similar tradeoffs of being really intelligent with good prompt following but very dry and boring).
> 
> I'd be interested to hear any of your feedback in non-coding usages as well for models.
> 
> >Using R1 at `Q4` today I was trying to understand how to potentially have `llm_load_tensors()` allocate N copies of ctx_buffs (one on each N NUMA nodes). It helped me understand a bit more the relationship between `src/llama.cpp` and `ggml/src/ggml-backend.c`, but didn't give magic working code haha... It did help me update `CMakeLists.txt` to get it building linking with libnuma library. I've also had some luck with it refactoring python code especially creating uniform style comments and adding static typing. Even QwQ-32B could write a decent 1-shot flappy bird when given a detailed prompt to follow haha...
> 
> Nice, I'd be curious if your numa optimizations bear any fruit. 
> 
> > One supposed success story is about [airbnb refactoring javascript test code](https://medium.com/airbnb-engineering/accelerating-large-scale-test-migration-with-llms-9565c208023b) to use a different library. Hard to say how much "tech debt" was incurred if any, but I too am curious to hear of any successful uses of ai for actually useful coding.
> 
> Thank you for the linked article, was a good read, another success story I know of is here: https://github.com/ggml-org/llama.cpp/pull/11453, "Surprisingly, 99% of the code in this PR is written by DeekSeek-R1."
> 
> 👤 **ubergarm** replied the **2025-03-25** at **05:02:40**:<br>
> I'm half asleep and didn't see this reply it pretty late. I appreciate the encouragement and pointers to good existing discussions!
> 
> I got the new `V3-0324` bf16 cranked out pretty quickly, but it didn't sink in that `bin/llama-imatrix` would have to run the full ~1.34TB model lmao... Of course the 256GB + 96GB VRAM system OOMd almost immedeately. So I copied everything to the 1.5TB RAM dual xeon 6980P and am giving that a go while I sleep.
> 
> > Have you decided on a calibration dataset?
> 
> I found some year old discussions that led me to this gist [calibration_data_v5_rc.txt](https://gist.github.com/tristandruyen/9e207a95c7d75ddf37525d353e00659c) which has a mix of languages and code. No idea if it will work well for this big MoE. I could have gone with standard `wiki.text.raw`, but seems like using something is better than nothing. I'll be happy if it even works haha...
> 
> > weighted/imatrix quants seem not to be available (by me) at this time. If they do not show up a week or so after the static ones, I have probably not planned for them. Feel free to request them by opening a Community Discussion. [mradermacher/DeepSeek-V3-0324-GGUF](https://huggingface.co/mradermacher/DeepSeek-V3-0324-GGUF)
> 
> Interesting note on mradermacher's model card readme :point_up_2: 
> 
> > They RPC multiple machines together,
> 
> Wow, sounds like quite a chore to imatrix these big models. Oh yeah, I can see why, just got my first imatrix chunk in lol:
> 
> ```
> ~/projects/ik_llama.cpp$ git rev-parse --short HEAD
> f9307d79
> 
> $ numactl --interleave=all \
> ./build/bin/llama-imatrix \
>     --verbosity 1 \
>     -m /mnt/ai/models/deepseek-ai/DeepSeek-V3-0324-bf16-GGUF/DeepSeek-256x21B-V3-0324-BF16-00001-of-00030.gguf \
>     -f calibration_data_v5_rc.txt \
>     -o /mnt/ai/models/deepseek-ai/DeepSeek-V3-0324-bf16-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-calibration-data-v5-rc.dat \
>     --ctx-size 512 \
>     -ctk q8_0 \ # <--- see below for updated commands
>     -mla 3 -fa \# <--- see below for updated commands
>     -amb 512 \# <--- see below for updated commands
>     -fmoe \# <--- see below for updated commands
>     --numa distribute \
>     --threads 192
> 
> llama_model_loader: - type  f32:  361 tensors
> llama_model_loader: - type bf16:  786 tensors
> 
> llama_kv_cache_init:        CPU KV buffer size =    18.23 MiB
> llama_new_context_with_model: KV self size  =   18.23 MiB, c^KV (q8_0):   18.23 MiB, kv^T: not used
> llama_new_context_with_model:        CPU  output buffer size =     0.49 MiB
> llama_new_context_with_model:        CPU compute buffer size =   266.50 MiB
> llama_new_context_with_model: graph nodes  = 3487
> llama_new_context_with_model: graph splits = 1
> 
> system_info: n_threads = 192 / 512 | AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | A
> VX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1
> | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 |
> compute_imatrix: tokenizing the input ..
> compute_imatrix: tokenization took 310.937 ms
> compute_imatrix: computing over 213 chunks with batch_size 512
> 
> [1]59.8267,[2]10.6927,[3]5.8694,[4]3.7855,[5]2.9690,[6]2.5103,[7]2.2235,[8]2.0239,[9]1.9107,
> save_imatrix: entry '             blk.25.ffn_down_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.12.ffn_down_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
> 
> save_imatrix: stored collected data after 10 chunks in /mnt/ai/models/deepseek-ai/DeepSeek-V3-0324-bf16-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-calibration-data-v5-rc.dat
> [10]1.8245,
> ```
> 
> Huh it still seems to be reading mmap off of this slower disk into cache and is barely hitting 20% total CPU utilization so hopefully it speeds up a bit more haha...
> 
> Okie, gotta sleep, exciting times!
> 
> 👤 **saood06** replied the **2025-03-25** at **05:35:18**:<br>
> > I'm half asleep and didn't see this reply it pretty late. I appreciate the encouragement and pointers to good existing discussions!
> > 
> > I got the new `V3-0324` bf16 cranked out pretty quickly, but it didn't sink in that `bin/llama-imatrix` would have to run the full ~1.34TB model lmao... 
> 
> You can just quantize to Q8_0 statically, and then use that for imatrix, which should finish a lot quicker, and since Deepseek is FP8 native, Q8_0 should be fine for imatrix (I know team mradermacher uses Q8_0 for these models, and in the past has done imatrix calculations on even smaller quants for other models, but that seems behind them for now [there is no indication of what quant was used on the model pages, and if that practice had continued I would have requested that be added], but they will requant models they have in the past and people have reported much better quants and this may play a part in it])
> 
> > > Have you decided on a calibration dataset?
> > 
> > I found some year old discussions that led me to this gist [calibration_data_v5_rc.txt](https://gist.github.com/tristandruyen/9e207a95c7d75ddf37525d353e00659c) which has a mix of languages and code. No idea if it will work well for this big MoE. I could have gone with standard `wiki.text.raw`, but seems like using something is better than nothing. I'll be happy if it even works haha...
> 
> Ah that one, I'm familiar with it, and I think you made a good choice (comparing to the publicly available ones I know of, I know there are others like the team mradermacher that are better but aren't publicly available).
> 
> 
> > Interesting note on mradermacher's model card readme 👆
> 
> They always have that, they handle a LOT of quants and so they script the whole process including creating and updating README.md.
> 
> I have personally asked them about the status of this imatrix, which is where I got my information on their status with it from.
> 
> > > They RPC multiple machines together,
> > 
> > Wow, sounds like quite a chore to imatrix these big models. Oh yeah, I can see why, just got my first imatrix chunk in lol:
> 
> I agree it does, but fortunately for us they seem to enjoy doing it.
> > 
> >```
> > compute_imatrix: 779.93 seconds per pass - ETA 46 hours 8.75 minutes
> > [1]59.8267,[2]10.6927,
> > ```
> > 
> > Huh it still seems to be reading mmap off of this slower disk into cache and is barely hitting 20% total CPU utilization so hopefully it speeds up a bit more haha...
> 
> I do too, and any chance you would be willing to post the resulting imatrix.dat file which will be just ~1 GB? I still will probably use the mradermacher one, ~~but yours will have the additional MLA tensor which I might be able to merge into theirs~~, but it would be fun to size golf the smallest functional Deepseek model and see how much the quality of the imatrix matters on a model that small.
> 
> > Okie, gotta sleep, exciting times!
> 
> I agree, I have a lot of theories about what they will do with Deepseek-R2. I really like the model, but reading their papers they have done an amazing job at optimizations when it comes to get the most out of the hardware and on the choices of model architecture (MLA, MoE with a good amount of experts [I can't say it's a lot when [this](https://arxiv.org/abs/2407.04153) exists], a shared expert [qwen 3 seems to be dropping this for their MoE which is interesting], etc.) , but the actual RL tuning seems like there are a LOT of low hanging fruit and obvious and large improvements that can be done.
> 
> Edit: Corrected mistake about imatrix
> 
> 👤 **ubergarm** replied the **2025-03-25** at **14:51:52**:<br>
> > You can just quantize to Q8_0 statically, and then use that for imatrix
> 
> Ahh, that is good news, running across both CPU sockets NUMA nodes is not performant to fit the whole bf16 haha... You asked in another thread about how it went. I had to quickly restart it due to forgetting to set directory permissions to write the imatrix.dat file, and that second time it estimated 11 hours. I killed it before finishing though after reading more of these notes.
> 
> <details>
> 
> <summary>Incomplete imatrix logs</summary>
> 
> ```bash
> compute_imatrix: tokenizing the input ..
> compute_imatrix: tokenization took 313.572 ms
> compute_imatrix: computing over 213 chunks with batch_size 512
> compute_imatrix: 200.99 seconds per pass - ETA 11 hours 53.50 minutes
> [1]59.8267,[2]10.6927,[3]5.8694,[4]3.7855,[5]2.9690,[6]2.5103,[7]2.2235,[8]2.0239,[9]1.9107,
> save_imatrix: entry '             blk.25.ffn_down_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.12.ffn_down_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
> 
> save_imatrix: stored collected data after 10 chunks in /mnt/ai/models/deepseek-ai/DeepSeek-V3-0324-bf16-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-calibration-data-v5-r
> c.dat
> [10]1.8245,[11]2.0340,[12]2.0895,[13]2.1034,[14]2.1467,[15]2.0421,[16]1.9542,[17]1.8831,[18]1.8202,[19]1.7779,
> save_imatrix: stored collected data after 20 chunks in /mnt/ai/models/deepseek-ai/DeepSeek-V3-0324-bf16-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-calibration-data-v5-r
> c.dat
> [20]1.7348,[21]1.7019,[22]1.6643,[23]1.6350,[24]1.6226,[25]1.6104,[26]1.5849,[27]1.6841,[28]1.7585,[29]1.8246,
> save_imatrix: stored collected data after 30 chunks in /mnt/ai/models/deepseek-ai/DeepSeek-V3-0324-bf16-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-calibration-data-v5-r
> c.dat
> [30]1.8229,[31]1.8362,[32]1.8357,[33]1.8132,[34]1.8491,[35]1.8247,[36]1.8247,[37]1.8135,[38]1.8235,[39]1.8101,
> save_imatrix: stored collected data after 40 chunks in /mnt/ai/models/deepseek-ai/DeepSeek-V3-0324-bf16-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-calibration-data-v5-r
> c.dat
> [40]1.7868,[41]1.7635,[42]1.7438,[43]1.7319,[44]1.7185,[45]1.7052,[46]1.7007,[47]1.6944,[48]1.6837,[49]1.6732,
> save_imatrix: stored collected data after 50 chunks in /mnt/ai/models/deepseek-ai/DeepSeek-V3-0324-bf16-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-calibration-data-v5-r
> c.dat
> [50]1.6671,[51]1.6644,[52]1.6646,[53]1.6693,[54]1.6833,[55]1.6800,[56]1.6701,[57]1.6783,[58]1.6811,[59]1.6924,
> save_imatrix: stored collected data after 60 chunks in /mnt/ai/models/deepseek-ai/DeepSeek-V3-0324-bf16-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-calibration-data-v5-r
> c.dat
> [60]1.6872,[61]1.7256,[62]1.7581,[63]1.7904,[64]1.8218,[65]1.8703,[66]1.8824,[67]1.9172,[68]1.9465,[69]2.0022,
> save_imatrix: stored collected data after 70 chunks in /mnt/ai/models/deepseek-ai/DeepSeek-V3-0324-bf16-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-calibration-data-v5-r
> c.dat
> [70]2.0549,[71]2.0852,[72]2.1154,[73]2.1277,[74]2.1428,[75]2.1718,[76]2.2021,[77]2.2196,[78]2.2177,[79]2.2324,
> save_imatrix: stored collected data after 80 chunks in /mnt/ai/models/deepseek-ai/DeepSeek-V3-0324-bf16-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-calibration-data-v5-r
> c.dat
> [80]2.2556,[81]2.2916,[82]2.3254,[83]2.3361,[84]2.3665,[85]2.3747,[86]2.3745,[87]2.4037,[88]2.4361,[89]2.4919,
> save_imatrix: stored collected data after 90 chunks in /mnt/ai/models/deepseek-ai/DeepSeek-V3-0324-bf16-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-calibration-data-v5-r
> c.dat
> [90]2.5123,[91]2.5145,[92]2.5212,[93]2.5367,[94]2.5471,[95]2.5800,[96]2.5691,[97]2.6079,[98]2.6339,[99]2.6236,
> save_imatrix: stored collected data after 100 chunks in /mnt/ai/models/deepseek-ai/DeepSeek-V3-0324-bf16-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-calibration-data-v5-
> rc.dat
> [100]2.6563,[101]2.7033,[102]2.7351,[103]2.7763,[104]2.8043,[105]2.8335,[106]2.8704,[107]2.8624,[108]2.8809,[109]2.8875,
> save_imatrix: stored collected data after 110 chunks in /mnt/ai/models/deepseek-ai/DeepSeek-V3-0324-bf16-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-calibration-data-v5-
> rc.dat
> [110]2.8934,[111]2.8903,[112]2.9198,[113]2.9459,[114]2.9543,[115]2.9385,[116]2.9127,[117]2.9070,[118]2.9173,[119]2.9029,
> save_imatrix: stored collected data after 120 chunks in /mnt/ai/models/deepseek-ai/DeepSeek-V3-0324-bf16-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-calibration-data-v5-
> rc.dat
> [120]2.8798,[121]2.8762,[122]2.8762,[123]2.8841,[124]2.8896,[125]2.8964,[126]2.9037,[127]2.9059,[128]2.9361,[129]2.9503,
> save_imatrix: stored collected data after 130 chunks in /mnt/ai/models/deepseek-ai/DeepSeek-V3-0324-bf16-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-calibration-data-v5-
> rc.dat
> [130]2.9259,[131]2.9021,[132]2.8789,[133]2.8561,[134]2.8580,[135]2.8584,[136]2.8844,[137]2.9166,[138]2.9356,^C^C
> 
> # only ~130 MiB after 8 hours or so...
> $ ls -la imatrix-ubergarm-DeepSeek-V3-0324-bf16-calibration-data-v5-rc.dat
> 135382908 Mar 25 13:19 imatrix-ubergarm-DeepSeek-V3-0324-bf16-calibration-data-v5-rc.dat
> ```
> 
> </details>
> 
> > it would be fun to size golf the smallest functional Deepseek model
> 
> Hah, yeah, I'm too am wondering which of the non MoE layers I can shrink down from `q8_0` a bit to free up enough space to fit 64k context in under 24GB VRAM along with them all using `-ot exps=CPU`. Yes, if I can get a valid imatrix.dat I'm happy to upload it onto huggingface along with all details to re-create including what fork/git sha/data file used etc.
> 
> Will see how much I can get through today, and I am out of office next couple days. Could leave imatrix running probably if there is a special llama fork to use as you referenced or if the input file is not enough chunks to give the ~1GiB dat file (tbh I'm just learning how it even works so just winging it lol).
> 
> 👤 **saood06** replied the **2025-03-25** at **15:04:57**:<br>
> > > You can just quantize to Q8_0 statically, and then use that for imatrix
> > 
> > ETA 11 hours 53.50 minutes
> 
> A lot faster than I expected.
> 
> > > it would be fun to size golf the smallest functional Deepseek model
> > 
> > Hah, yeah, I'm too am wondering which of the non MoE layers I can shrink down from `q8_0` a bit to free up enough space to fit 64k context in under 24GB VRAM along with them all using `-ot exps=CPU`.
> 
> [IQ6_K](https://github.com/ikawrakow/ik_llama.cpp/pull/14) is a very good quant, would be worth experimenting with.
> 
> >Yes, if I can get a valid imatrix.dat I'm happy to upload it onto huggingface along with all details to re-create including what fork/git sha/data file used etc.
> 
> Thank you
> 
> > Could leave imatrix running probably if there is a special llama fork to use as you referenced.
> 
> I recommend you to stick to this repo, the team mradermacher have very specialized needs and thus need to track llama.cpp's bleeding edge religiously, they took a fix ikawrakow wrote to fix an issue they were seeing, and just ported that over to llama.cpp alongside an extra example that allows you to calculate exact footprint required so they can do automated job scheduler that is resource aware.
> 
> 👤 **bartowski1182** replied the **2025-04-01** at **23:17:46**:<br>
> @ubergarm I wouldn't give too much thought to the imatrix dataset, there have been a lot of people recently who have tried iterating and experimenting on the one that I use, in particular related to different languages, and found shockingly minimal (if any) impact on the results of a target language by including that language in the dataset.
> 
> it seems clear that, as Kalomaze suggested way way back, the randomness/diversity of the data is much more important than the quality, because if ANYTHING was going to be altered by using a different imatrix set, surely it would be completely different languages.
> 
> for models the size of DeepSeek you can probably even go all the way down to Q4_K_M, I know mradermacher mentions going down to Q4_K_S, IQ3_XS or even Q2_K, and that was there before these monster models existed
> 
> that said, all this discussion about people with their massive xeon clusters and multiple servers RPCed together really tells me i need to find a sponsor.. 😂
> 
> 👤 **saood06** replied the **2025-04-02** at **00:23:04**:<br>
> > @ubergarm I wouldn't give too much thought to the imatrix dataset, there have been a lot of people recently who have tried iterating and experimenting on the one that I use, in particular related to different languages, and found shockingly minimal (if any) impact on the results of a target language by including that language in the dataset.
> 
> This paper also confirms that https://arxiv.org/abs/2503.03592
> 
> "Further, the usage of importance matrices written in non-English does not significantly improve performance on non-English datasets and might in fact slightly harm it. However, this reduction in performance is not statistically significant."
> 
> > it seems clear that, as Kalomaze suggested way way back, the randomness/diversity of the data is much more important than the quality, because if ANYTHING was going to be altered by using a different imatrix set, surely it would be completely different languages.
> 
> Yes, but I still tend toward team mradermacher's imatrix.dat because it is longer, and that matters a lot more in a model like deepseek where the calibration data is effectively spread out over the experts. I do think the difference is minimal (unless a catastrophic failure occurs but that would require a smaller dataset than yours).

---

👤 **ikawrakow** replied the **2025-03-25** at **06:32:51**:<br>

> [!IMPORTANT]
> To calculate the imatrix, please do not use any of the `mla, fa, fmoe` or `amb` options. With these, some of the tensors will not get imatrix data collected. 


As @saood06 pointed out, `Q8_0` is good enough to collect imatrix data. 

> Also this https://github.com/ikawrakow/ik_llama.cpp/pull/250 if you haven't seen it is obviously relevant to you,

This has been superseded by #259. The additional 2 tensors needed for MLA (`attn_k_b` and `attn_v_b`) are computed on the fly from `attn_kv_b` when loading the model (if missing). So, the best strategy is to use standard attention for imatrix calculations, which will give imatrix data to `attn_kv_b`, so this tensor will get a better quantization. `attn_k_b` is a transposed version of half of `attn_kv_b`. It gets computed by converting `attn_kv_b` to `fp32`, transposing that, and then quantizing to `Q8_0`, so (nearly) lossless. `attn_v_b` is just a view of the other half of `attn_kv_b`, so it uses the `attn_kv_b` data directly.

> 👤 **saood06** replied the **2025-03-25** at **07:01:42**:<br>
> Sorry I forgot about the implications of that PR, updated my comment to reflect it.
> 
> 👤 **ubergarm** replied the **2025-03-25** at **14:58:40**:<br>
> Great, thanks for the help and pro-tips!
> 
> Copying over the V3-0324 `q8_0_r8` to the xeon 6980P now and will leave this running and hope to get an imatrix.dat for further smaller quants. I've removed `mla, fa, fmoe, amb` options and am unsure on `-ctk q8_0` so will just remove it too.
> 
> What is left is basically all defaults.
> 
> ```
> numactl -N 0 -m 0 \
> ./build/bin/llama-imatrix \
>     --verbosity 1 \
>     -m /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-Q8_0_R8.gguf \
>     -f calibration_data_v5_rc.txt \
>     -o imatrix-DeepSeek-V3-0324.dat \
>     --ctx-size 512 \
>     --numa numactl \
>     --threads 128
> ```
> 
> 👤 **ubergarm** replied the **2025-03-25** at **17:00:15**:<br>
> Oof, starting getting NaN's computing imatrix on the `q8_0_r8`... Gonna pause rushing on this and go back and look at [Issue 285](https://github.com/ikawrakow/ik_llama.cpp/issues/285#issuecomment-2750335421) which I assume may be related.
> 
> <details>
> 
> <summary>llama-imatrix NaNs</summary>
> 
> ```
> llama_model_loader: loaded meta data with 46 key-value pairs and 1147 tensors from /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-Q8_0_R8.gguf (vers
> ion GGUF V3 (latest))
> llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
> llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
> llama_model_loader: - kv   1:                               general.type str              = model
> llama_model_loader: - kv   2:                               general.name str              = DeepSeek V3 0324
> llama_model_loader: - kv   3:                            general.version str              = V3-0324
> llama_model_loader: - kv   4:                           general.basename str              = DeepSeek
> llama_model_loader: - kv   5:                         general.size_label str              = 256x21B
> llama_model_loader: - kv   6:                            general.license str              = mit
> llama_model_loader: - kv   7:                      deepseek2.block_count u32              = 61
> llama_model_loader: - kv   8:                   deepseek2.context_length u32              = 163840
> llama_model_loader: - kv   9:                 deepseek2.embedding_length u32              = 7168
> llama_model_loader: - kv  10:              deepseek2.feed_forward_length u32              = 18432
> llama_model_loader: - kv  11:             deepseek2.attention.head_count u32              = 128
> llama_model_loader: - kv  12:          deepseek2.attention.head_count_kv u32              = 128
> llama_model_loader: - kv  13:                   deepseek2.rope.freq_base f32              = 10000.000000
> llama_model_loader: - kv  14: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
> llama_model_loader: - kv  15:                deepseek2.expert_used_count u32              = 8
> llama_model_loader: - kv  16:                          general.file_type u32              = 207
> llama_model_loader: - kv  17:        deepseek2.leading_dense_block_count u32              = 3
> llama_model_loader: - kv  18:                       deepseek2.vocab_size u32              = 129280
> llama_model_loader: - kv  19:            deepseek2.attention.q_lora_rank u32              = 1536
> llama_model_loader: - kv  20:           deepseek2.attention.kv_lora_rank u32              = 512
> llama_model_loader: - kv  21:             deepseek2.attention.key_length u32              = 192
> llama_model_loader: - kv  22:           deepseek2.attention.value_length u32              = 128
> llama_model_loader: - kv  23:       deepseek2.expert_feed_forward_length u32              = 2048
> llama_model_loader: - kv  24:                     deepseek2.expert_count u32              = 256
> llama_model_loader: - kv  25:              deepseek2.expert_shared_count u32              = 1
> llama_model_loader: - kv  26:             deepseek2.expert_weights_scale f32              = 2.500000
> llama_model_loader: - kv  27:              deepseek2.expert_weights_norm bool             = true
> llama_model_loader: - kv  28:               deepseek2.expert_gating_func u32              = 2
> llama_model_loader: - kv  29:             deepseek2.rope.dimension_count u32              = 64
> llama_model_loader: - kv  30:                deepseek2.rope.scaling.type str              = yarn
> llama_model_loader: - kv  31:              deepseek2.rope.scaling.factor f32              = 40.000000
> llama_model_loader: - kv  32: deepseek2.rope.scaling.original_context_length u32              = 4096
> llama_model_loader: - kv  33: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
> llama_model_loader: - kv  34:                       tokenizer.ggml.model str              = gpt2
> llama_model_loader: - kv  35:                         tokenizer.ggml.pre str              = deepseek-v3
> llama_model_loader: - kv  36:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<...
> llama_model_loader: - kv  37:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
> llama_model_loader: - kv  38:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
> llama_model_loader: - kv  39:                tokenizer.ggml.bos_token_id u32              = 0
> llama_model_loader: - kv  40:                tokenizer.ggml.eos_token_id u32              = 1
> llama_model_loader: - kv  41:            tokenizer.ggml.padding_token_id u32              = 1
> llama_model_loader: - kv  42:               tokenizer.ggml.add_bos_token bool             = true
> llama_model_loader: - kv  43:               tokenizer.ggml.add_eos_token bool             = false
> llama_model_loader: - kv  44:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
> llama_model_loader: - kv  45:               general.quantization_version u32              = 2
> llama_model_loader: - type  f32:  361 tensors
> llama_model_loader: - type q8_0:  612 tensors
> llama_model_loader: - type q8_0_r8:  174 tensors
> llm_load_vocab: special tokens cache size = 818
> llm_load_vocab: token to piece cache size = 0.8223 MB
> llm_load_print_meta: format           = GGUF V3 (latest)
> llm_load_print_meta: arch             = deepseek2
> llm_load_print_meta: vocab type       = BPE
> llm_load_print_meta: n_vocab          = 129280
> llm_load_print_meta: n_merges         = 127741
> llm_load_print_meta: vocab_only       = 0
> llm_load_print_meta: n_ctx_train      = 163840
> llm_load_print_meta: n_embd           = 7168
> llm_load_print_meta: n_layer          = 61
> llm_load_print_meta: n_head           = 128
> llm_load_print_meta: n_head_kv        = 128
> llm_load_print_meta: n_rot            = 64
> llm_load_print_meta: n_swa            = 0
> llm_load_print_meta: n_embd_head_k    = 192
> llm_load_print_meta: n_embd_head_v    = 128
> llm_load_print_meta: n_gqa            = 1
> llm_load_print_meta: n_embd_k_gqa     = 24576
> llm_load_print_meta: n_embd_v_gqa     = 16384
> llm_load_print_meta: f_norm_eps       = 0.0e+00
> llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
> llm_load_print_meta: f_clamp_kqv      = 0.0e+00
> llm_load_print_meta: f_max_alibi_bias = 0.0e+00
> llm_load_print_meta: f_logit_scale    = 0.0e+00
> llm_load_print_meta: n_ff             = 18432
> llm_load_print_meta: n_expert         = 256
> llm_load_print_meta: n_expert_used    = 8
> llm_load_print_meta: causal attn      = 1
> llm_load_print_meta: pooling type     = 0
> llm_load_print_meta: rope type        = 0
> llm_load_print_meta: rope scaling     = yarn
> llm_load_print_meta: freq_base_train  = 10000.0
> llm_load_print_meta: freq_scale_train = 0.025
> llm_load_print_meta: n_ctx_orig_yarn  = 4096
> llm_load_print_meta: rope_finetuned   = unknown
> llm_load_print_meta: ssm_d_conv       = 0
> llm_load_print_meta: ssm_d_inner      = 0
> llm_load_print_meta: ssm_d_state      = 0
> llm_load_print_meta: ssm_dt_rank      = 0
> llm_load_print_meta: model type       = 671B
> llm_load_print_meta: model ftype      = Q8_0_R8 - 8.5 bpw
> llm_load_print_meta: model params     = 672.050 B
> llm_load_print_meta: model size       = 665.308 GiB (8.504 BPW)
> llm_load_print_meta: repeating layers = 663.474 GiB (8.504 BPW, 670.196 B parameters)
> llm_load_print_meta: general.name     = DeepSeek V3 0324
> llm_load_print_meta: BOS token        = 0 '<｜begin▁of▁sentence｜>'
> llm_load_print_meta: EOS token        = 1 '<｜end▁of▁sentence｜>'
> llm_load_print_meta: PAD token        = 1 '<｜end▁of▁sentence｜>'
> llm_load_print_meta: LF token         = 131 'Ä'
> llm_load_print_meta: max token length = 256
> llm_load_print_meta: n_layer_dense_lead   = 3
> llm_load_print_meta: n_lora_q             = 1536
> llm_load_print_meta: n_lora_kv            = 512
> llm_load_print_meta: n_ff_exp             = 2048
> llm_load_print_meta: n_expert_shared      = 1
> llm_load_print_meta: expert_weights_scale = 2.5
> llm_load_print_meta: expert_weights_norm  = 1
> llm_load_print_meta: expert_gating_func   = sigmoid
> llm_load_print_meta: rope_yarn_log_mul    = 0.1000
> llm_load_tensors: ggml ctx size =    0.47 MiB
> llm_load_tensors:        CPU buffer size = 681274.97 MiB
> ....................................................................................................
> llama_new_context_with_model: n_ctx      = 512
> llama_new_context_with_model: n_batch    = 512
> llama_new_context_with_model: n_ubatch   = 512
> llama_new_context_with_model: flash_attn = 0
> llama_new_context_with_model: mla_attn   = 0
> llama_new_context_with_model: attn_max_b = 0
> llama_new_context_with_model: fused_moe  = 0
> llama_new_context_with_model: ser        = -1, 0
> llama_new_context_with_model: freq_base  = 10000.0
> llama_new_context_with_model: freq_scale = 0.025
> llama_kv_cache_init:        CPU KV buffer size =  2440.00 MiB
> llama_new_context_with_model: KV self size  = 2440.00 MiB, K (f16): 1464.00 MiB, V (f16):  976.00 MiB
> llama_new_context_with_model:        CPU  output buffer size =     0.49 MiB
> llama_new_context_with_model:        CPU compute buffer size =   283.01 MiB
> llama_new_context_with_model: graph nodes  = 3724
> llama_new_context_with_model: graph splits = 1
> 
> system_info: n_threads = 128 / 512 | AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE
>  = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 |
> compute_imatrix: tokenizing the input ..
> compute_imatrix: tokenization took 311.034 ms
> compute_imatrix: computing over 213 chunks with batch_size 512
> compute_imatrix: 421.99 seconds per pass - ETA 24 hours 58.07 minutes
> [1]61.0342,[2]10.8003,[3]5.8859,[4]3.7958,[5]2.9744,[6]2.5129,[7]2.2236,[8]2.0237,[9]1.9134,
> save_imatrix: entry '             blk.60.ffn_down_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.60.ffn_gate_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.60.ffn_up_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.31.ffn_gate_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.31.ffn_up_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.31.ffn_down_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
> 
> save_imatrix: stored collected data after 10 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324.dat
> [10]1.8254,[11]2.0341,[12]2.0883,[13]2.1015,[14]2.1441,[15]2.0399,[16]1.9523,[17]1.8810,[18]1.8182,[19]1.7755,
> save_imatrix: stored collected data after 20 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324.dat
> [20]1.7330,[21]1.7002,[22]1.6627,[23]1.6335,[24]1.6215,[25]1.6091,[26]1.5832,[27]1.6825,[28]1.7562,[29]1.8217,
> save_imatrix: stored collected data after 30 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324.dat
> [30]1.8190,[31]1.8323,[32]1.8320,[33]1.8095,[34]1.8453,[35]1.8213,[36]1.8208,[37]1.8093,[38]nan,[39]nan,
> save_imatrix: stored collected data after 40 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324.dat
> [40]nan,[41]nan,[42]nan,[43]nan,[44]nan,[45]nan,[46]nan,[47]nan,[48]nan,[49]nan,
> save_imatrix: stored collected data after 50 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324.dat
> [50]nan,[51]nan,[52]nan,[53]nan,[54]nan,[55]nan,[56]nan,[57]nan,[58]nan,[59]nan,
> save_imatrix: stored collected data after 60 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324.dat
> [60]nan,[61]nan,[62]nan,[63]nan,[64]nan,[65]nan,[66]nan,[67]nan,[68]nan,^C^C
> ```
> 
> </details>
> 
> 👤 **ikawrakow** replied the **2025-03-25** at **17:12:23**:<br>
> So, this is unfortunate, but also helpful as it excludes the `fmoe` optimization as a cause. Oops, not actually helpful as now I'm completely at a loss what could be causing the NaNs.
> 
> 👤 **ubergarm** replied the **2025-03-25** at **17:31:08**:<br>
> @ikawrakow 
> 
> Thanks for looking. I'm running the perplexity again as per 285 currently. Will update that one as soon as data starts coming in.
> 
> I realized my `V3-0324`quantize script left attention and non MoE layers as `q8_0`. The MoE layers were `q8_0_r8` which is a bit odd of a mix given this one was intended for CPU only.
> 
> <details>
> <summary>Block 42 logs for the quantization used for the imatrix giving Nans</summary>
> 
> ```
> [ 785/1147]              blk.42.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
> [ 786/1147]           blk.42.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
> [ 787/1147]         blk.42.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   bf16, Using custom type q8_0 for tensor blk.42.ffn_down_shexp.weight
> converting to q8_0 .. size =    28.00 MiB ->    14.88 MiB
> [ 788/1147]         blk.42.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   bf16, Using custom type q8_0 for tensor blk.42.ffn_gate_shexp.weight
> converting to q8_0 .. size =    28.00 MiB ->    14.88 MiB
> [ 789/1147]           blk.42.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   bf16, Using custom type q8_0 for tensor blk.42.ffn_up_shexp.weight
> converting to q8_0 .. size =    28.00 MiB ->    14.88 MiB
> [ 790/1147]         blk.42.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
> [ 791/1147]          blk.42.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   bf16, Using custom type q8_0 for tensor blk.42.attn_kv_a_mqa.weight
> converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
> [ 792/1147]              blk.42.attn_kv_b.weight - [  512, 32768,     1,     1], type =   bf16, Using custom type q8_0 for tensor blk.42.attn_kv_b.weight
> converting to q8_0 .. size =    32.00 MiB ->    17.00 MiB
> [ 793/1147]               blk.42.attn_k_b.weight - [  128, 65536,     1,     1], type =   bf16, Using custom type q8_0 for tensor blk.42.attn_k_b.weight
> converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
> [ 794/1147]               blk.42.attn_v_b.weight - [  512, 16384,     1,     1], type =   bf16, Using custom type q8_0 for tensor blk.42.attn_v_b.weight
> converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
> [ 795/1147]            blk.42.attn_output.weight - [16384,  7168,     1,     1], type =   bf16, Using custom type q8_0 for tensor blk.42.attn_output.weight
> converting to q8_0 .. size =   224.00 MiB ->   119.00 MiB
> [ 796/1147]          blk.42.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
> [ 797/1147]               blk.42.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   bf16, Using custom type q8_0 for tensor blk.42.attn_q_a.weight
> converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
> [ 798/1147]               blk.42.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   bf16, Using custom type q8_0 for tensor blk.42.attn_q_b.weight
> converting to q8_0 .. size =    72.00 MiB ->    38.25 MiB
> [ 799/1147]              blk.42.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
> [ 800/1147]          blk.42.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.42.ffn_down_exps.weight
> converting to q8_0_r8 .. size =  7168.00 MiB ->  3808.00 MiB
> [ 801/1147]          blk.42.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.42.ffn_gate_exps.weight
> converting to q8_0_r8 .. size =  7168.00 MiB ->  3808.00 MiB
> [ 802/1147]            blk.42.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   bf16, Using custom type q8_0_r8 for tensor blk.42.ffn_up_exps.weight
> converting to q8_0_r8 .. size =  7168.00 MiB ->  3808.00 MiB
> [ 803/1147]               blk.42.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
> ```
> 
> </details>
> 
> 👤 **saood06** replied the **2025-03-25** at **17:41:06**:<br>
> > Oof, starting getting NaN's computing imatrix on the `q8_0_r8`... Gonna pause rushing on this and go back and look at [Issue 285](https://github.com/ikawrakow/ik_llama.cpp/issues/285#issuecomment-2750335421) which I assume may be related.
> 
> Are you going to go back to the BF16, or use llama.cpp with the Q8_0 to generate an imatrix?
> 
> 👤 **ubergarm** replied the **2025-03-25** at **17:45:20**:<br>
> @saood06 
> 
> Well, mainline llama.cpp will *not* work with my mixed `q8_0`/`q8_0_r8` quant. So either:
> 
>   * Option A: i wait for the `bf16` which will take forever
>   * Option B: whip out another more simple `q8_0` everything and copy it over and use mainline llama.cpp...
> 
> I've started down Option B for now and with some luck I can get the imatrix.dat uploaded by tomorrow morning before I head out.
> 
> 👤 **ikawrakow** replied the **2025-03-25** at **18:11:31**:<br>
> If you do option B and make simple `Q8_0`, then it would be useful to 1st try `ik_llama.cpp` with that. That will help narrow down the problem. If you don't get NaNs, it is somehow related to `Q8_0_R8`, and you can keep going with `ik_llama.cpp`. If you do get NaNs, you can stop it and use mainline.
> 
> Btw, on a CPU with native `bf16` support, running `imatrix` with a `bf16` model should be only marginally slower than `Q8_0`.
> 
> 👤 **saood06** replied the **2025-03-25** at **18:24:50**:<br>
> > Btw, on a CPU with native `bf16` support, running `imatrix` with a `bf16` model should be only marginally slower than `Q8_0`.
> 
> Under normal conditions yes, but going to bf16 forces him onto both numa sockets, I'm interested to know what speed llama.cpp would give though compared to this since he's going down that path now.
> 
> 👤 **ikawrakow** replied the **2025-03-25** at **18:30:46**:<br>
> > Under normal conditions yes, but going to bf16 forces him onto both numa sockets
> 
> And why would 2 sockets be bad for performance? It is PP, not TG, memory bandwidth and latency should be mostly irrelevant. With batches of 512, each piece of data that gets fetched from memory gets used 512 times for computations.
> 
> 👤 **ikawrakow** replied the **2025-03-25** at **18:34:44**:<br>
> Ah, it is a MoE model with 256 experts. Batches of 512 result in many experts doing multiplication with just a handful of rows. So, I guess, there will be a larger penalty due to memory access patterns. Still, I don't expect it to be slower than 1 socket. Or?
> 
> 👤 **ubergarm** replied the **2025-03-25** at **20:24:25**:<br>
> > simple Q8_0, then it would be useful to 1st try ik_llama.cpp with that
> 
> Ooh I almost thought we had it... Was about to update and just got first `nan`:
> 
> <details>
> 
> <summary>Attempt ik_llama.cpp imatrix with all `q8_0` quant Logs</summary>
> 
> ```bash
> # Double checked ik_llama.cpp main@98a264a2 also threw nan beginning chunk 38, so no difference.
> 
> # was on a test branch ik_llama.cp ik/deepseek_is_this_better
> $ git rev-parse --short HEAD
> daa3b00c
> 
> $ numactl -N 1 -m 1 \
> ./build/bin/llama-imatrix \
>     --verbosity 1 \
>     -m /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-Q8_0.gguf \
>     -f calibration_data_v5_rc.txt \
>     -o /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-daa3b00c.dat \
>     --ctx-size 512 \
>     --numa numactl \
>     --threads 128
> 
> llama_model_loader: loaded meta data with 46 key-value pairs and 1147 tensors from /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-Q8_0.gguf (version GGUF V3 (late
> st))
> llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
> llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
> llama_model_loader: - kv   1:                               general.type str              = model
> llama_model_loader: - kv   2:                               general.name str              = DeepSeek V3 0324
> llama_model_loader: - kv   3:                            general.version str              = V3-0324
> llama_model_loader: - kv   4:                           general.basename str              = DeepSeek
> llama_model_loader: - kv   5:                         general.size_label str              = 256x21B
> llama_model_loader: - kv   6:                            general.license str              = mit
> llama_model_loader: - kv   7:                      deepseek2.block_count u32              = 61
> ...
> llama_model_loader: - type  f32:  361 tensors
> llama_model_loader: - type q8_0:  786 tensors
> llm_load_vocab: special tokens cache size = 818
> llm_load_vocab: token to piece cache size = 0.8223 MB
> llm_load_print_meta: format           = GGUF V3 (latest)
> llm_load_print_meta: arch             = deepseek2
> llm_load_print_meta: vocab type       = BPE
> ...
> llm_load_tensors: ggml ctx size =    0.47 MiB
> llm_load_tensors:        CPU buffer size = 681274.97 MiB
> ....................................................................................................
> llama_new_context_with_model: n_ctx      = 512
> llama_new_context_with_model: n_batch    = 512
> llama_new_context_with_model: n_ubatch   = 512
> llama_new_context_with_model: flash_attn = 0
> llama_new_context_with_model: mla_attn   = 0
> llama_new_context_with_model: attn_max_b = 0
> llama_new_context_with_model: fused_moe  = 0
> llama_new_context_with_model: ser        = -1, 0
> llama_new_context_with_model: freq_base  = 10000.0
> llama_new_context_with_model: freq_scale = 0.025
> llama_kv_cache_init:        CPU KV buffer size =  2440.00 MiB
> llama_new_context_with_model: KV self size  = 2440.00 MiB, K (f16): 1464.00 MiB, V (f16):  976.00 MiB
> llama_new_context_with_model:        CPU  output buffer size =     0.49 MiB
> llama_new_context_with_model:        CPU compute buffer size =   283.01 MiB
> llama_new_context_with_model: graph nodes  = 3724
> llama_new_context_with_model: graph splits = 1
> ...
> system_info: n_threads = 128 / 512 | AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 |
> compute_imatrix: tokenizing the input ..
> compute_imatrix: tokenization took 315.999 ms
> compute_imatrix: computing over 213 chunks with batch_size 512
> compute_imatrix: 161.29 seconds per pass - ETA 9 hours 32.55 minutes
> [1]60.7582,[2]10.7798,[3]5.8765,[4]3.7890,[5]2.9716,[6]2.5104,[7]2.2220,[8]2.0224,[9]1.9119,
> save_imatrix: entry '             blk.60.ffn_down_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.60.ffn_gate_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.60.ffn_up_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.25.ffn_down_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '               blk.25.ffn_up_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
> save_imatrix: entry '             blk.25.ffn_gate_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
> 
> save_imatrix: stored collected data after 10 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-daa3b00c.dat
> [10]1.8245,[11]2.0331,[12]2.0874,[13]2.1014,[14]2.1468,[15]2.0425,[16]1.9547,[17]1.8833,[18]1.8205,[19]1.7774,
> save_imatrix: stored collected data after 20 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-daa3b00c.dat
> [20]1.7345,[21]1.7015,[22]1.6640,[23]1.6345,[24]1.6219,[25]1.6099,[26]1.5840,[27]1.6832,[28]1.7571,[29]1.8226,
> save_imatrix: stored collected data after 30 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-daa3b00c.dat
> [30]1.8203,[31]1.8337,[32]1.8334,[33]1.8108,[34]1.8468,[35]1.8225,[36]1.8218,[37]1.8108,[38]nan,[39]nan,
> save_imatrix: stored collected data after 40 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-daa3b00c.dat
> [40]nan,[41]nan,[42]nan,[43]nan,[44]nan,[45]nan,[46]nan,[47]nan,[48]nan,[49]nan,
> save_imatrix: stored collected data after 50 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-daa3b00c.dat
> ```
> 
> </details>
> 
> So gonna stop and try mainline for now. Can keep tracking this over in #285 as it may be related.
> 
> 👤 **ubergarm** replied the **2025-03-25** at **20:52:22**:<br>
> Double oof mainline is complaining despite it all being `q8_0`... 
> 
> ```
> build: 4958 (ef19c717) with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu
> llama_model_loader: loaded meta data with 46 key-value pairs and 1147 tensors from /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-Q8_0.gguf (version
>  GGUF V3 (latest))
> ...
> llama_model_loader: - type  f32:  361 tensors
> llama_model_loader: - type q8_0:  786 tensors
> print_info: file format = GGUF V3 (latest)
> print_info: file type   = Q8_0
> print_info: file size   = 665.31 GiB (8.50 BPW)
> ...
> load_tensors: tensor 'token_embd.weight' (q8_0) (and 535 others) cannot be used with preferred buffer type AMX, using CPU instead
> llama_model_load: error loading model: done_getting_tensors: wrong number of tensors; expected 1147, got 1025
> llama_model_load_from_file_impl: failed to load model
> ```
> 
> Not sure what is up there, as i confirmed the `ik_llama.cpp` `llama-quantize` did all 1147 layers and the file size is correct so it all copied over via rsync.
> 
> I went ahead and used mainline to make the `q8_0` without any *custom* stuff in it and am copying that over. Gotta get that sweet sweet imatrix.dat lol...
> 
> *EDIT* Huh [bartowski/deepseek-ai_DeepSeek-V3-0324-GGUF](https://huggingface.co/bartowski/deepseek-ai_DeepSeek-V3-0324-GGUF/tree/main) has had an imatrix.dat there since yesterday... lol... okay... well... i'll still give this a go for fun and report back...
> 
> 👤 **saood06** replied the **2025-03-25** at **20:56:26**:<br>
> > Double oof mainline is complaining despite it all being `q8_0`...
> > 
> 
> That is expected.
> 
> #259 says:
> >In principle we could remove the preparation of wk_v and wk_b from convert_hf_to_gguf.py, but I decided have some more thorough testing in the wild before doing so.
> 
> Those extra tensors support the MLA branch of llama.cpp (since we derived MLA support from that originally), so maybe try that one.
> 
> 👤 **saood06** replied the **2025-03-25** at **21:03:52**:<br>
> Mentioned in the original port PR.
> 
> https://github.com/ikawrakow/ik_llama.cpp/pull/180#issuecomment-2621112020
> 
> It is still a choice of what our converter outputs, should we be compliant with the MLA PR as that allows you to compare feature performance across both, or to support the main branch of llama.cpp even though they have a PR with that feature.
> 
> 👤 **ubergarm** replied the **2025-03-25** at **21:49:13**:<br>
> Oooh right right the fairydreaming fork PR! I never tried that as I found this fork before learning how to roll my own MLA quant... Thanks, I'll try that quick while also copying over another mainline `q8_0` for insurance haha...  Also finally rolling my new usual `q8_0` on GPU and MoEs on CPU with `iq3_k_r4/iq2_k_r4` quant with bartowski's imatrix just to compare perpelxity if i get the itch haha...
> 
> 👤 **saood06** replied the **2025-03-25** at **22:29:21**:<br>
> >Also finally rolling my new usual q8_0 on GPU and MoEs on CPU with iq3_k_r4/iq2_k_r4 quant with bartowski's imatrix just to compare perpelxity if i get the itch haha...
> 
> I will let my IQ4_K_R4 quantize overnight, I grabbed everything I need  for V3 0324 ( bartowski's imatrix and a Q8_0).
> 
> 👤 **ubergarm** replied the **2025-03-25** at **22:52:07**:<br>
> > I will let my IQ4_K_R4 quantize overnight, I grabbed everything I need for V3 0324 ( bartowski's imatrix and a Q8_0).
> 
> Nice, happy cooking!
> 
> So I managed to build that [fairydreaming/deepseek2-mla-exp@76543311](https://github.com/fairydreaming/llama.cpp/tree/deepseek2-mla-exp) and have `llama-perplexity` running on the plain `q8_0` I made with `ik_llama.cpp`.
> 
> The output is different and it seems to be skipping 10-15% of the tensors due to partial 99.9% data... 
> 
> Took just over 2 hours to run the imatrix, but not sure that is is valid.
> 
> <details>
> <summary>imatrix logs</summary>
> 
> warning this is pretty long:
> 
> ```bash
> numactl -N 1 -m 1 \
> ./build/bin/llama-imatrix \
>     --verbosity 1 \
>     -m /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-Q8_0.gguf \
>     -f calibration_data_v5_rc.txt \
>     -o /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-fairydreaming-llamacpp-76543311.dat \
>     --ctx-size 512 \
>     --numa numactl \
>     --threads 128
> 
> build: 4553 (76543311) with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu
> llama_model_loader: loaded meta data with 46 key-value pairs and 1147 tensors from /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-Q8_0.gguf (version GGUF V3 (latest))
> llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
> llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
> llama_model_loader: - kv   1:                               general.type str              = model
> llama_model_loader: - kv   2:                               general.name str              = DeepSeek V3 0324
> llama_model_loader: - kv   3:                            general.version str              = V3-0324
> llama_model_loader: - kv   4:                           general.basename str              = DeepSeek
> llama_model_loader: - kv   5:                         general.size_label str              = 256x21B
> llama_model_loader: - kv   6:                            general.license str              = mit
> llama_model_loader: - kv   7:                      deepseek2.block_count u32              = 61
> llama_model_loader: - kv   8:                   deepseek2.context_length u32              = 163840
> llama_model_loader: - kv   9:                 deepseek2.embedding_length u32              = 7168
> llama_model_loader: - kv  10:              deepseek2.feed_forward_length u32              = 18432
> llama_model_loader: - kv  11:             deepseek2.attention.head_count u32              = 128
> llama_model_loader: - kv  12:          deepseek2.attention.head_count_kv u32              = 128
> llama_model_loader: - kv  13:                   deepseek2.rope.freq_base f32              = 10000.000000
> llama_model_loader: - kv  14: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
> llama_model_loader: - kv  15:                deepseek2.expert_used_count u32              = 8
> llama_model_loader: - kv  16:                          general.file_type u32              = 7
> llama_model_loader: - kv  17:        deepseek2.leading_dense_block_count u32              = 3
> llama_model_loader: - kv  18:                       deepseek2.vocab_size u32              = 129280
> llama_model_loader: - kv  19:            deepseek2.attention.q_lora_rank u32              = 1536
> llama_model_loader: - kv  20:           deepseek2.attention.kv_lora_rank u32              = 512
> llama_model_loader: - kv  21:             deepseek2.attention.key_length u32              = 192
> llama_model_loader: - kv  22:           deepseek2.attention.value_length u32              = 128
> llama_model_loader: - kv  23:       deepseek2.expert_feed_forward_length u32              = 2048
> llama_model_loader: - kv  24:                     deepseek2.expert_count u32              = 256
> llama_model_loader: - kv  25:              deepseek2.expert_shared_count u32              = 1
> llama_model_loader: - kv  26:             deepseek2.expert_weights_scale f32              = 2.500000
> llama_model_loader: - kv  27:              deepseek2.expert_weights_norm bool             = true
> llama_model_loader: - kv  28:               deepseek2.expert_gating_func u32              = 2
> llama_model_loader: - kv  29:             deepseek2.rope.dimension_count u32              = 64
> llama_model_loader: - kv  30:                deepseek2.rope.scaling.type str              = yarn
> llama_model_loader: - kv  31:              deepseek2.rope.scaling.factor f32              = 40.000000
> llama_model_loader: - kv  32: deepseek2.rope.scaling.original_context_length u32              = 4096
> llama_model_loader: - kv  33: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
> llama_model_loader: - kv  34:                       tokenizer.ggml.model str              = gpt2
> llama_model_loader: - kv  35:                         tokenizer.ggml.pre str              = deepseek-v3
> llama_model_loader: - kv  36:                      tokenizer.ggml.tokens arr[str,129280]  = ["
> llama_model_loader: - kv  37:                  tokenizer.ggml.token_type arr[i32,129280]  = [3
> llama_model_loader: - kv  38:                      tokenizer.ggml.merges arr[str,127741]  = ["
> llama_model_loader: - kv  39:                tokenizer.ggml.bos_token_id u32              = 0
> llama_model_loader: - kv  40:                tokenizer.ggml.eos_token_id u32              = 1
> llama_model_loader: - kv  41:            tokenizer.ggml.padding_token_id u32              = 1
> llama_model_loader: - kv  42:               tokenizer.ggml.add_bos_token bool             = true
> llama_model_loader: - kv  43:               tokenizer.ggml.add_eos_token bool             = false
> llama_model_loader: - kv  44:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
> llama_model_loader: - kv  45:               general.quantization_version u32              = 2
> llama_model_loader: - type  f32:  361 tensors
> llama_model_loader: - type q8_0:  786 tensors
> print_info: file format = GGUF V3 (latest)
> print_info: file type   = Q8_0
> print_info: file size   = 665.31 GiB (8.50 BPW) 
> init_tokenizer: initializing tokenizer for type 2
> load: control token: 128000 '<
> load: special_eos_id is not in special_eog_ids - the tokenizer config may be incorrect
> load: special tokens cache size = 818
> load: token to piece cache size = 0.8223 MB
> print_info: arch             = deepseek2
> print_info: vocab_only       = 0
> print_info: n_ctx_train      = 163840
> print_info: n_embd           = 7168
> print_info: n_layer          = 61
> print_info: n_head           = 128
> print_info: n_head_kv        = 128
> print_info: n_rot            = 64
> print_info: n_swa            = 0
> print_info: n_embd_head_k    = 192
> print_info: n_embd_head_v    = 128
> print_info: n_gqa            = 1
> print_info: n_embd_k_gqa     = 24576
> print_info: n_embd_v_gqa     = 16384
> print_info: f_norm_eps       = 0.0e+00
> print_info: f_norm_rms_eps   = 1.0e-06
> print_info: f_clamp_kqv      = 0.0e+00
> print_info: f_max_alibi_bias = 0.0e+00
> print_info: f_logit_scale    = 0.0e+00
> print_info: n_ff             = 18432
> print_info: n_expert         = 256
> print_info: n_expert_used    = 8
> print_info: causal attn      = 1
> print_info: pooling type     = 0
> print_info: rope type        = 0
> print_info: rope scaling     = yarn
> print_info: freq_base_train  = 10000.0
> print_info: freq_scale_train = 0.025
> print_info: n_ctx_orig_yarn  = 4096
> print_info: rope_finetuned   = unknown
> print_info: ssm_d_conv       = 0
> print_info: ssm_d_inner      = 0
> print_info: ssm_d_state      = 0
> print_info: ssm_dt_rank      = 0
> print_info: ssm_dt_b_c_rms   = 0
> print_info: model type       = 671B
> print_info: model params     = 672.05 B
> print_info: general.name     = DeepSeek V3 0324
> print_info: n_layer_dense_lead   = 3
> print_info: n_lora_q             = 1536
> print_info: n_lora_kv            = 512
> print_info: n_ff_exp             = 2048
> print_info: n_expert_shared      = 1
> print_info: expert_weights_scale = 2.5
> print_info: expert_weights_norm  = 1
> print_info: expert_gating_func   = sigmoid
> print_info: rope_yarn_log_mul    = 0.1000
> print_info: vocab type       = BPE
> print_info: n_vocab          = 129280
> print_info: n_merges         = 127741
> print_info: BOS token        = 0 '<｜begin▁of▁sentence｜>'
> print_info: EOS token        = 1 '<｜end▁of▁sentence｜>'
> print_info: EOT token        = 1 '<｜end▁of▁sentence｜>'
> print_info: PAD token        = 1 '<｜end▁of▁sentence｜>'
> print_info: LF token         = 131 'Ä'
> print_info: FIM PRE token    = 128801 '<｜fim▁begin｜>'
> print_info: FIM SUF token    = 128800 '<｜fim▁hole｜>'
> print_info: FIM MID token    = 128802 '<｜fim▁end｜>'
> print_info: EOG token        = 1 '<｜end▁of▁sentence｜>'
> print_info: max token length = 256
> load_tensors: tensor 'token_embd.weight' (q8_0) (and 535 others) cannot be used with preferred buffer type AMX, using CPU instead
> load_tensors:          AMX model buffer size = 19373.39 MiB
> load_tensors:   CPU_Mapped model buffer size = 681274.97 MiB
> ggml_backend_amx_buffer_set_tensor: amx repack tensor output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.0.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.0.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.0.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.0.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.0.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.0.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.0.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.0.ffn_gate.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.0.ffn_down.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.0.ffn_up.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.1.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.1.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.1.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.1.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.1.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.1.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.1.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.1.ffn_gate.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.1.ffn_down.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.1.ffn_up.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.2.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.2.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.2.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.2.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.2.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.2.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.2.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.2.ffn_gate.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.2.ffn_down.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.2.ffn_up.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.3.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.3.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.3.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.3.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.3.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.3.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.3.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.3.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.3.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.3.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.4.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.4.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.4.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.4.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.4.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.4.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.4.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.4.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.4.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.4.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.5.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.5.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.5.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.5.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.5.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.5.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.5.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.5.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.5.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.5.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.6.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.6.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.6.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.6.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.6.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.6.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.6.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.6.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.6.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.6.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.7.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.7.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.7.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.7.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.7.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.7.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.7.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.7.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.7.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.7.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.8.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.8.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.8.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.8.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.8.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.8.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.8.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.8.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.8.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.8.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.9.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.9.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.9.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.9.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.9.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.9.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.9.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.9.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.9.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.9.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.10.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.10.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.10.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.10.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.10.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.10.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.10.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.10.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.10.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.10.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.11.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.11.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.11.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.11.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.11.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.11.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.11.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.11.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.11.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.11.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.12.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.12.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.12.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.12.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.12.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.12.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.12.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.12.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.12.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.12.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.13.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.13.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.13.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.13.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.13.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.13.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.13.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.13.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.13.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.13.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.14.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.14.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.14.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.14.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.14.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.14.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.14.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.14.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.14.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.14.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.15.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.15.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.15.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.15.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.15.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.15.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.15.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.15.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.15.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.15.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.16.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.16.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.16.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.16.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.16.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.16.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.16.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.16.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.16.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.16.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.17.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.17.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.17.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.17.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.17.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.17.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.17.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.17.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.17.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.17.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.18.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.18.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.18.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.18.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.18.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.18.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.18.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.18.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.18.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.18.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.19.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.19.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.19.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.19.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.19.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.19.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.19.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.19.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.19.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.19.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.20.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.20.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.20.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.20.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.20.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.20.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.20.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.20.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.20.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.20.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.21.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.21.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.21.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.21.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.21.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.21.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.21.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.21.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.21.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.21.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.22.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.22.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.22.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.22.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.22.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.22.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.22.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.22.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.22.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.22.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.23.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.23.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.23.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.23.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.23.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.23.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.23.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.23.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.23.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.23.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.24.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.24.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.24.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.24.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.24.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.24.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.24.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.24.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.24.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.24.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.25.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.25.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.25.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.25.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.25.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.25.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.25.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.25.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.25.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.25.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.26.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.26.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.26.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.26.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.26.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.26.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.26.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.26.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.26.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.26.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.27.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.27.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.27.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.27.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.27.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.27.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.27.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.27.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.27.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.27.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.28.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.28.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.28.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.28.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.28.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.28.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.28.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.28.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.28.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.28.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.29.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.29.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.29.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.29.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.29.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.29.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.29.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.29.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.29.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.29.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.30.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.30.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.30.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.30.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.30.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.30.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.30.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.30.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.30.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.30.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.31.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.31.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.31.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.31.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.31.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.31.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.31.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.31.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.31.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.31.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.32.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.32.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.32.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.32.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.32.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.32.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.32.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.32.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.32.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.32.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.33.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.33.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.33.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.33.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.33.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.33.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.33.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.33.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.33.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.33.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.34.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.34.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.34.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.34.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.34.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.34.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.34.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.34.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.34.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.34.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.35.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.35.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.35.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.35.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.35.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.35.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.35.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.35.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.35.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.35.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.36.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.36.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.36.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.36.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.36.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.36.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.36.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.36.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.36.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.36.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.37.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.37.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.37.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.37.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.37.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.37.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.37.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.37.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.37.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.37.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.38.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.38.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.38.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.38.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.38.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.38.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.38.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.38.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.38.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.38.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.39.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.39.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.39.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.39.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.39.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.39.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.39.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.39.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.39.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.39.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.40.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.40.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.40.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.40.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.40.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.40.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.40.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.40.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.40.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.40.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.41.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.41.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.41.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.41.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.41.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.41.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.41.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.41.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.41.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.41.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.42.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.42.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.42.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.42.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.42.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.42.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.42.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.42.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.42.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.42.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.43.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.43.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.43.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.43.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.43.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.43.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.43.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.43.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.43.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.43.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.44.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.44.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.44.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.44.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.44.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.44.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.44.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.44.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.44.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.44.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.45.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.45.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.45.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.45.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.45.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.45.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.45.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.45.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.45.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.45.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.46.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.46.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.46.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.46.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.46.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.46.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.46.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.46.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.46.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.46.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.47.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.47.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.47.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.47.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.47.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.47.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.47.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.47.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.47.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.47.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.48.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.48.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.48.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.48.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.48.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.48.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.48.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.48.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.48.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.48.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.49.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.49.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.49.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.49.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.49.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.49.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.49.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.49.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.49.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.49.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.50.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.50.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.50.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.50.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.50.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.50.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.50.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.50.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.50.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.50.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.51.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.51.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.51.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.51.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.51.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.51.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.51.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.51.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.51.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.51.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.52.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.52.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.52.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.52.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.52.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.52.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.52.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.52.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.52.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.52.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.53.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.53.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.53.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.53.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.53.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.53.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.53.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.53.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.53.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.53.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.54.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.54.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.54.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.54.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.54.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.54.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.54.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.54.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.54.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.54.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.55.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.55.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.55.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.55.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.55.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.55.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.55.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.55.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.55.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.55.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.56.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.56.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.56.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.56.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.56.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.56.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.56.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.56.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.56.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.56.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.57.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.57.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.57.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.57.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.57.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.57.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.57.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.57.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.57.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.57.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.58.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.58.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.58.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.58.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.58.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.58.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.58.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.58.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.58.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.58.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.59.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.59.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.59.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.59.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.59.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.59.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.59.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.59.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.59.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.59.ffn_up_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.60.attn_q_a.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.60.attn_q_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.60.attn_kv_a_mqa.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.60.attn_kv_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.60.attn_k_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.60.attn_v_b.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.60.attn_output.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.60.ffn_gate_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.60.ffn_down_shexp.weight of type q8_0
> ggml_backend_amx_buffer_set_tensor: amx repack tensor blk.60.ffn_up_shexp.weight of type q8_0
> llama_init_from_model: n_seq_max     = 1
> llama_init_from_model: n_ctx         = 512
> llama_init_from_model: n_ctx_per_seq = 512
> llama_init_from_model: n_batch       = 512
> llama_init_from_model: n_ubatch      = 512
> llama_init_from_model: flash_attn    = 0
> llama_init_from_model: freq_base     = 10000.0
> llama_init_from_model: freq_scale    = 0.025
> llama_init_from_model: n_ctx_per_seq (512) < n_ctx_train (163840) -- the full capacity of the model will not be utilized
> llama_kv_cache_init: kv_size = 512, offload = 1, type_k = 'f16', type_v = 'f16', n_layer = 61, can_shift = 0
> llama_kv_cache_init: layer 0: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 0: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 1: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 1: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 2: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 2: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 3: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 3: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 4: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 4: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 5: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 5: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 6: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 6: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 7: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 7: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 8: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 8: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 9: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 9: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 10: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 10: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 11: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 11: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 12: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 12: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 13: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 13: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 14: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 14: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 15: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 15: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 16: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 16: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 17: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 17: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 18: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 18: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 19: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 19: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 20: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 20: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 21: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 21: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 22: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 22: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 23: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 23: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 24: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 24: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 25: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 25: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 26: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 26: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 27: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 27: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 28: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 28: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 29: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 29: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 30: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 30: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 31: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 31: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 32: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 32: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 33: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 33: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 34: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 34: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 35: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 35: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 36: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 36: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 37: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 37: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 38: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 38: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 39: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 39: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 40: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 40: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 41: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 41: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 42: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 42: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 43: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 43: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 44: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 44: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 45: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 45: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 46: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 46: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 47: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 47: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 48: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 48: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 49: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 49: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 50: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 50: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 51: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 51: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 52: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 52: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 53: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 53: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 54: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 54: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 55: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 55: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 56: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 56: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 57: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 57: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 58: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 58: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 59: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 59: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 60: n_embd_k_gqa = 24576, n_embd_v_gqa = 16384
> llama_kv_cache_init: layer 60: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init:        CPU KV buffer size =  2504.81 MiB
> llama_init_from_model: KV self size  = 2440.00 MiB, K (f16): 1464.00 MiB, V (f16):  976.00 MiB
> llama_init_from_model: KV self size  =   34.31 MiB, K^R (f16):    3.81 MiB, c^KV (f16):   30.50 MiB
> llama_init_from_model:        CPU  output buffer size =     0.49 MiB
> llama_init_from_model:        CPU compute buffer size =   379.01 MiB
> llama_init_from_model: graph nodes  = 5208 (with bs=512), 5330 (with bs=1)
> llama_init_from_model: graph splits = 1
> common_init_from_params: KV cache shifting is not supported for this model, disabling KV cache shifting
> common_init_from_params: setting dry_penalty_last_n to ctx_size = 512
> 
> system_info: n_threads = 128 (n_threads_batch = 128) / 512 | CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | AMX_INT8 = 1 | LLAMAFILE = 1 | OPENMP = 1 | AARCH64_REPACK = 1 | 
> compute_imatrix: tokenizing the input ..
> compute_imatrix: tokenization took 314.286 ms
> compute_imatrix: computing over 213 chunks with batch_size 512
> compute_imatrix: 38.56 seconds per pass - ETA 2 hours 16.87 minutes
> [1]10099620.0329,[2]5891181.7767,[3]6287837.4629,[4]6347458.4866,[5]6814823.2533,[6]6098823.6402,[7]6208734.2134,[8]6229710.3740,[9]5927383.6219,
> save_imatrix: entry '             blk.60.ffn_down_exps.weight' has partial data (83.98%) - skipping
> save_imatrix: entry '             blk.59.ffn_down_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '             blk.59.ffn_gate_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '               blk.59.ffn_up_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '             blk.60.ffn_gate_exps.weight' has partial data (83.98%) - skipping
> save_imatrix: entry '             blk.58.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.57.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.56.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.56.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.50.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.50.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.23.ffn_up_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '               blk.22.ffn_up_exps.weight' has partial data (96.48%) - skipping
> save_imatrix: entry '             blk.33.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.12.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.58.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.6.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.21.ffn_gate_exps.weight' has partial data (97.27%) - skipping
> save_imatrix: entry '               blk.19.ffn_up_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.18.ffn_down_exps.weight' has partial data (96.48%) - skipping
> save_imatrix: entry '             blk.17.ffn_gate_exps.weight' has partial data (96.09%) - skipping
> save_imatrix: entry '               blk.58.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.16.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.9.ffn_up_exps.weight' has partial data (97.66%) - skipping
> save_imatrix: entry '             blk.15.ffn_gate_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '               blk.15.ffn_up_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '               blk.10.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.20.ffn_up_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '             blk.45.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.8.ffn_down_exps.weight' has partial data (97.27%) - skipping
> save_imatrix: entry '             blk.14.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.14.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.10.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.40.ffn_up_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.60.ffn_up_exps.weight' has partial data (83.98%) - skipping
> save_imatrix: entry '             blk.25.ffn_down_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.26.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.27.ffn_down_exps.weight' has partial data (96.88%) - skipping
> save_imatrix: entry '             blk.16.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.13.ffn_down_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '                blk.5.ffn_up_exps.weight' has partial data (95.70%) - skipping
> save_imatrix: entry '             blk.20.ffn_gate_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '             blk.36.ffn_down_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '              blk.5.ffn_down_exps.weight' has partial data (95.70%) - skipping
> save_imatrix: entry '               blk.30.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '                blk.7.ffn_up_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '               blk.50.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.36.ffn_gate_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.21.ffn_up_exps.weight' has partial data (97.27%) - skipping
> save_imatrix: entry '              blk.7.ffn_gate_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '               blk.14.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.23.ffn_down_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '              blk.4.ffn_gate_exps.weight' has partial data (94.14%) - skipping
> save_imatrix: entry '               blk.25.ffn_up_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.17.ffn_up_exps.weight' has partial data (96.09%) - skipping
> save_imatrix: entry '             blk.17.ffn_down_exps.weight' has partial data (96.09%) - skipping
> save_imatrix: entry '             blk.21.ffn_down_exps.weight' has partial data (97.27%) - skipping
> save_imatrix: entry '             blk.22.ffn_gate_exps.weight' has partial data (96.48%) - skipping
> save_imatrix: entry '              blk.9.ffn_gate_exps.weight' has partial data (97.66%) - skipping
> save_imatrix: entry '             blk.37.ffn_down_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '             blk.18.ffn_gate_exps.weight' has partial data (96.48%) - skipping
> save_imatrix: entry '               blk.57.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.9.ffn_down_exps.weight' has partial data (97.66%) - skipping
> save_imatrix: entry '             blk.12.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.3.ffn_down_exps.weight' has partial data (94.14%) - skipping
> save_imatrix: entry '             blk.12.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.3.ffn_gate_exps.weight' has partial data (94.14%) - skipping
> save_imatrix: entry '               blk.13.ffn_up_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '               blk.34.ffn_up_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.32.ffn_up_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '              blk.6.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.15.ffn_down_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '             blk.11.ffn_down_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.32.ffn_gate_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.24.ffn_gate_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '             blk.23.ffn_gate_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '                blk.3.ffn_up_exps.weight' has partial data (94.14%) - skipping
> save_imatrix: entry '             blk.10.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.7.ffn_down_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '                blk.6.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.11.ffn_up_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.11.ffn_gate_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.37.ffn_up_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '             blk.22.ffn_down_exps.weight' has partial data (96.48%) - skipping
> save_imatrix: entry '              blk.4.ffn_down_exps.weight' has partial data (94.14%) - skipping
> save_imatrix: entry '               blk.24.ffn_up_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '             blk.24.ffn_down_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '               blk.18.ffn_up_exps.weight' has partial data (96.48%) - skipping
> save_imatrix: entry '             blk.25.ffn_gate_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.26.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.26.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '                blk.4.ffn_up_exps.weight' has partial data (94.14%) - skipping
> save_imatrix: entry '               blk.27.ffn_up_exps.weight' has partial data (96.88%) - skipping
> save_imatrix: entry '             blk.27.ffn_gate_exps.weight' has partial data (96.88%) - skipping
> save_imatrix: entry '             blk.38.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.31.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.28.ffn_up_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.28.ffn_gate_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.29.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.29.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.29.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.30.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '                blk.8.ffn_up_exps.weight' has partial data (97.27%) - skipping
> save_imatrix: entry '             blk.43.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.36.ffn_up_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.40.ffn_gate_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.13.ffn_gate_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '             blk.31.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.16.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.39.ffn_up_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.33.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.33.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.5.ffn_gate_exps.weight' has partial data (95.70%) - skipping
> save_imatrix: entry '             blk.20.ffn_down_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '             blk.34.ffn_gate_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.34.ffn_down_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.30.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.35.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.35.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.40.ffn_down_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.37.ffn_gate_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '              blk.8.ffn_gate_exps.weight' has partial data (97.27%) - skipping
> save_imatrix: entry '             blk.39.ffn_down_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.38.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.39.ffn_gate_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.28.ffn_down_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.35.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.43.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.19.ffn_gate_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.43.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.45.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.45.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.38.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.19.ffn_down_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.31.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.57.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.32.ffn_down_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.56.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: storing only 521 out of 659 entries
> 
> save_imatrix: stored collected data after 10 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-fairydreaming-llamacpp-76543311.dat
> [10]5566393.4069,[11]5275015.3751,[12]5172372.6126,[13]5246273.3072,[14]5279623.9749,[15]5174838.5077,[16]5336073.7993,[17]5263611.8912,[18]5433651.3703,[19]5220894.9876,
> save_imatrix: entry '             blk.60.ffn_down_exps.weight' has partial data (86.33%) - skipping
> save_imatrix: entry '             blk.59.ffn_down_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.59.ffn_gate_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.59.ffn_up_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.60.ffn_gate_exps.weight' has partial data (86.33%) - skipping
> save_imatrix: entry '             blk.58.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.56.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.56.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.23.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.22.ffn_up_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '             blk.58.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.6.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.21.ffn_gate_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.19.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.18.ffn_down_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '             blk.17.ffn_gate_exps.weight' has partial data (97.27%) - skipping
> save_imatrix: entry '               blk.58.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '                blk.9.ffn_up_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.15.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.15.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.20.ffn_up_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '             blk.45.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.8.ffn_down_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '             blk.14.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.14.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.40.ffn_up_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.60.ffn_up_exps.weight' has partial data (86.33%) - skipping
> save_imatrix: entry '             blk.25.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.26.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.27.ffn_down_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '             blk.13.ffn_down_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '                blk.5.ffn_up_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '             blk.20.ffn_gate_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '             blk.36.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.5.ffn_down_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '               blk.30.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '                blk.7.ffn_up_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '             blk.36.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.21.ffn_up_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '              blk.7.ffn_gate_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '               blk.14.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.23.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.4.ffn_gate_exps.weight' has partial data (95.31%) - skipping
> save_imatrix: entry '               blk.25.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.17.ffn_up_exps.weight' has partial data (97.27%) - skipping
> save_imatrix: entry '             blk.17.ffn_down_exps.weight' has partial data (97.27%) - skipping
> save_imatrix: entry '             blk.21.ffn_down_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.22.ffn_gate_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '              blk.9.ffn_gate_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.37.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.18.ffn_gate_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '              blk.9.ffn_down_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '              blk.3.ffn_down_exps.weight' has partial data (96.48%) - skipping
> save_imatrix: entry '              blk.3.ffn_gate_exps.weight' has partial data (96.48%) - skipping
> save_imatrix: entry '               blk.13.ffn_up_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '               blk.34.ffn_up_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.32.ffn_up_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '              blk.6.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.15.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.11.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.32.ffn_gate_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.24.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.23.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '                blk.3.ffn_up_exps.weight' has partial data (96.48%) - skipping
> save_imatrix: entry '              blk.7.ffn_down_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '                blk.6.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.11.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.11.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.37.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.22.ffn_down_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '              blk.4.ffn_down_exps.weight' has partial data (95.31%) - skipping
> save_imatrix: entry '               blk.24.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.24.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.18.ffn_up_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '             blk.25.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.26.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.26.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.4.ffn_up_exps.weight' has partial data (95.31%) - skipping
> save_imatrix: entry '               blk.27.ffn_up_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '             blk.27.ffn_gate_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '             blk.38.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.31.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.29.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.29.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.29.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.30.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '                blk.8.ffn_up_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '               blk.36.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.40.ffn_gate_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.13.ffn_gate_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '             blk.31.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.39.ffn_up_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '              blk.5.ffn_gate_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '             blk.20.ffn_down_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '             blk.34.ffn_gate_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.34.ffn_down_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.30.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.35.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.35.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.40.ffn_down_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.37.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.8.ffn_gate_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '             blk.39.ffn_down_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.38.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.39.ffn_gate_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.35.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.19.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.45.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.45.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.38.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.19.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.31.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.32.ffn_down_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.56.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: storing only 545 out of 659 entries
> 
> save_imatrix: stored collected data after 20 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-fairydreaming-llamacpp-76543311.dat
> [20]5142641.1471,[21]5124800.0424,[22]5078210.0759,[23]5156119.0865,[24]5199447.2924,[25]5189607.3987,[26]5182308.5024,[27]5157808.2319,[28]5114499.0719,[29]5106314.6561,
> save_imatrix: entry '             blk.60.ffn_down_exps.weight' has partial data (87.11%) - skipping
> save_imatrix: entry '             blk.59.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.59.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.59.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.60.ffn_gate_exps.weight' has partial data (87.11%) - skipping
> save_imatrix: entry '             blk.58.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.23.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.22.ffn_up_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.58.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.6.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.21.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.19.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.18.ffn_down_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '             blk.17.ffn_gate_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '               blk.58.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.9.ffn_up_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.20.ffn_up_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '             blk.45.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.8.ffn_down_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.14.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.14.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.40.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.60.ffn_up_exps.weight' has partial data (87.11%) - skipping
> save_imatrix: entry '             blk.25.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.27.ffn_down_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.13.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.5.ffn_up_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.20.ffn_gate_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '              blk.5.ffn_down_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.30.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '                blk.7.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.21.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.7.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.14.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.23.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.4.ffn_gate_exps.weight' has partial data (96.88%) - skipping
> save_imatrix: entry '               blk.25.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.17.ffn_up_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '             blk.17.ffn_down_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '             blk.21.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.22.ffn_gate_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '              blk.9.ffn_gate_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.37.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.18.ffn_gate_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '              blk.9.ffn_down_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '              blk.3.ffn_down_exps.weight' has partial data (97.66%) - skipping
> save_imatrix: entry '              blk.3.ffn_gate_exps.weight' has partial data (97.66%) - skipping
> save_imatrix: entry '               blk.13.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.34.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.32.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.6.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.32.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.24.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.23.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.3.ffn_up_exps.weight' has partial data (97.66%) - skipping
> save_imatrix: entry '              blk.7.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.6.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.37.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.22.ffn_down_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '              blk.4.ffn_down_exps.weight' has partial data (96.88%) - skipping
> save_imatrix: entry '               blk.24.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.24.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.18.ffn_up_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '             blk.25.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.4.ffn_up_exps.weight' has partial data (96.88%) - skipping
> save_imatrix: entry '               blk.27.ffn_up_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.27.ffn_gate_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.38.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.30.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '                blk.8.ffn_up_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.40.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.13.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.39.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.5.ffn_gate_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.20.ffn_down_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '             blk.34.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.34.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.30.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.35.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.35.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.40.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.37.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.8.ffn_gate_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.39.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.38.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.39.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.35.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.19.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.45.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.45.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.38.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.19.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.32.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: storing only 566 out of 659 entries
> 
> save_imatrix: stored collected data after 30 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-fairydreaming-llamacpp-76543311.dat
> [30]5117005.6825,[31]5316050.9054,[32]5253677.1003,[33]5251844.7907,[34]5270371.6274,[35]5185687.4168,[36]5164607.5507,[37]5298843.8882,[38]5370264.5450,[39]5520150.9112,
> save_imatrix: entry '             blk.60.ffn_down_exps.weight' has partial data (87.89%) - skipping
> save_imatrix: entry '             blk.60.ffn_gate_exps.weight' has partial data (87.89%) - skipping
> save_imatrix: entry '             blk.58.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.22.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.58.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.6.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.21.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.19.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.18.ffn_down_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.17.ffn_gate_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.58.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.9.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.20.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.45.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.8.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.40.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.60.ffn_up_exps.weight' has partial data (87.89%) - skipping
> save_imatrix: entry '             blk.25.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.27.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.13.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.5.ffn_up_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.20.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.5.ffn_down_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.30.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.7.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.21.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.7.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.4.ffn_gate_exps.weight' has partial data (97.27%) - skipping
> save_imatrix: entry '               blk.25.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.17.ffn_up_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.17.ffn_down_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.21.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.22.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.9.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.37.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.18.ffn_gate_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '              blk.9.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.3.ffn_down_exps.weight' has partial data (97.66%) - skipping
> save_imatrix: entry '              blk.3.ffn_gate_exps.weight' has partial data (97.66%) - skipping
> save_imatrix: entry '               blk.13.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.34.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.6.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.24.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.3.ffn_up_exps.weight' has partial data (97.66%) - skipping
> save_imatrix: entry '              blk.7.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.6.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.37.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.22.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.4.ffn_down_exps.weight' has partial data (97.27%) - skipping
> save_imatrix: entry '               blk.24.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.24.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.18.ffn_up_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.25.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.4.ffn_up_exps.weight' has partial data (97.27%) - skipping
> save_imatrix: entry '               blk.27.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.27.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.38.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.30.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.8.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.40.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.13.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.39.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.5.ffn_gate_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.20.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.34.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.34.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.30.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.35.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.35.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.40.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.37.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.8.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.39.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.38.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.39.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.35.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.19.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.45.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.45.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.38.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.19.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: storing only 578 out of 659 entries
> 
> save_imatrix: stored collected data after 40 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-fairydreaming-llamacpp-76543311.dat
> [40]5535641.6861,[41]5563674.2120,[42]5597934.7002,[43]5927878.9452,[44]5819467.9645,[45]5842418.8376,[46]5832158.4790,[47]5804593.0437,[48]5720860.9853,[49]5990902.2935,
> save_imatrix: entry '             blk.60.ffn_down_exps.weight' has partial data (88.28%) - skipping
> save_imatrix: entry '             blk.60.ffn_gate_exps.weight' has partial data (88.28%) - skipping
> save_imatrix: entry '               blk.22.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.6.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.21.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.19.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.18.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.17.ffn_gate_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '                blk.9.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.20.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.45.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.8.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.40.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.60.ffn_up_exps.weight' has partial data (88.28%) - skipping
> save_imatrix: entry '             blk.25.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.27.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.13.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.5.ffn_up_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.20.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.5.ffn_down_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.30.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.7.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.21.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.7.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.4.ffn_gate_exps.weight' has partial data (97.27%) - skipping
> save_imatrix: entry '               blk.25.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.17.ffn_up_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.17.ffn_down_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.21.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.22.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.9.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.37.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.18.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.9.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.3.ffn_down_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '              blk.3.ffn_gate_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '               blk.13.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.34.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.6.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.24.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.3.ffn_up_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '              blk.7.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.6.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.37.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.22.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.4.ffn_down_exps.weight' has partial data (97.27%) - skipping
> save_imatrix: entry '               blk.24.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.24.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.18.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.25.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.4.ffn_up_exps.weight' has partial data (97.27%) - skipping
> save_imatrix: entry '               blk.27.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.27.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.38.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.30.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.8.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.40.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.13.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.39.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.5.ffn_gate_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.20.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.34.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.34.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.30.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.35.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.35.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.40.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.37.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.8.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.39.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.38.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.39.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.35.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.19.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.45.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.45.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.38.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.19.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: storing only 581 out of 659 entries
> 
> save_imatrix: stored collected data after 50 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-fairydreaming-llamacpp-76543311.dat
> [50]5954228.4011,[51]5899743.8001,[52]5822579.9684,[53]5820037.7718,[54]5832302.6714,[55]5854089.5515,[56]5936771.8378,[57]5919873.8985,[58]5857793.5739,[59]5746723.3960,
> save_imatrix: entry '             blk.60.ffn_down_exps.weight' has partial data (89.45%) - skipping
> save_imatrix: entry '             blk.60.ffn_gate_exps.weight' has partial data (89.45%) - skipping
> save_imatrix: entry '               blk.22.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.6.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.21.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.19.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.18.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.17.ffn_gate_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '                blk.9.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.20.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.45.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.8.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.40.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.60.ffn_up_exps.weight' has partial data (89.45%) - skipping
> save_imatrix: entry '             blk.25.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.27.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.5.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.20.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.5.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.30.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.21.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.4.ffn_gate_exps.weight' has partial data (97.66%) - skipping
> save_imatrix: entry '               blk.25.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.17.ffn_up_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.17.ffn_down_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '             blk.21.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.22.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.9.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.37.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.18.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.9.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.3.ffn_down_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '              blk.3.ffn_gate_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '               blk.34.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.6.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.24.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.3.ffn_up_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '                blk.6.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.37.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.22.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.4.ffn_down_exps.weight' has partial data (97.66%) - skipping
> save_imatrix: entry '               blk.24.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.24.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.18.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.25.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.4.ffn_up_exps.weight' has partial data (97.66%) - skipping
> save_imatrix: entry '               blk.27.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.27.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.38.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.30.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.8.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.40.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.39.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.5.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.20.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.34.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.34.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.30.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.35.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.35.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.40.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.37.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.8.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.39.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.38.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.39.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.35.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.19.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.45.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.45.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.38.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.19.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: storing only 587 out of 659 entries
> 
> save_imatrix: stored collected data after 60 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-fairydreaming-llamacpp-76543311.dat
> [60]5794106.3600,[61]5757117.7366,[62]5767844.4712,[63]5793165.3504,[64]5810712.0438,[65]5851623.1812,[66]5854376.2815,[67]5776990.0658,[68]5797318.3634,[69]5824867.0818,
> save_imatrix: entry '             blk.60.ffn_down_exps.weight' has partial data (89.45%) - skipping
> save_imatrix: entry '             blk.60.ffn_gate_exps.weight' has partial data (89.45%) - skipping
> save_imatrix: entry '               blk.22.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.6.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.18.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.17.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.20.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.45.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.8.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.40.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.60.ffn_up_exps.weight' has partial data (89.45%) - skipping
> save_imatrix: entry '             blk.25.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.27.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.5.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.20.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.5.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.30.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.4.ffn_gate_exps.weight' has partial data (97.66%) - skipping
> save_imatrix: entry '               blk.25.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.17.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.17.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.22.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.37.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.18.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.3.ffn_down_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '              blk.3.ffn_gate_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '               blk.34.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.6.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.3.ffn_up_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '                blk.6.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.37.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.22.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.4.ffn_down_exps.weight' has partial data (97.66%) - skipping
> save_imatrix: entry '               blk.18.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.25.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.4.ffn_up_exps.weight' has partial data (97.66%) - skipping
> save_imatrix: entry '               blk.27.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.27.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.38.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.30.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.8.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.40.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.39.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.5.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.20.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.34.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.34.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.30.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.35.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.35.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.40.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.37.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.8.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.39.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.38.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.39.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.35.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.45.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.45.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.38.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: storing only 599 out of 659 entries
> 
> save_imatrix: stored collected data after 70 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-fairydreaming-llamacpp-76543311.dat
> [70]5818905.6407,[71]5801987.8419,[72]5806722.0852,[73]5761175.5716,[74]5824874.4234,[75]5809799.4348,[76]5813982.9251,[77]5786950.5852,[78]5798986.4011,[79]5781810.0004,
> save_imatrix: entry '             blk.60.ffn_down_exps.weight' has partial data (89.84%) - skipping
> save_imatrix: entry '             blk.60.ffn_gate_exps.weight' has partial data (89.84%) - skipping
> save_imatrix: entry '               blk.22.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.18.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.17.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.20.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.45.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.40.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.60.ffn_up_exps.weight' has partial data (89.84%) - skipping
> save_imatrix: entry '             blk.25.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.27.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.5.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.20.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.5.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.30.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.4.ffn_gate_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '               blk.25.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.17.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.17.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.22.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.37.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.18.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.3.ffn_down_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '              blk.3.ffn_gate_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.34.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.3.ffn_up_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.37.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.22.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.4.ffn_down_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '               blk.18.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.25.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.4.ffn_up_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '               blk.27.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.27.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.38.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.30.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.40.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.39.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.5.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.20.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.34.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.34.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.30.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.35.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.35.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.40.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.37.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.39.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.38.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.39.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.35.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.45.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.45.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.38.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: storing only 605 out of 659 entries
> 
> save_imatrix: stored collected data after 80 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-fairydreaming-llamacpp-76543311.dat
> [80]5774801.5351,[81]5779856.7665,[82]5837701.5609,[83]5860968.2119,[84]5922526.5202,[85]5922493.9059,[86]5911194.9571,[87]5920235.1279,[88]6092682.0673,[89]6177472.5774,
> save_imatrix: entry '             blk.60.ffn_down_exps.weight' has partial data (89.84%) - skipping
> save_imatrix: entry '             blk.60.ffn_gate_exps.weight' has partial data (89.84%) - skipping
> save_imatrix: entry '               blk.22.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.18.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.17.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.20.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.45.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.40.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.60.ffn_up_exps.weight' has partial data (89.84%) - skipping
> save_imatrix: entry '             blk.25.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.27.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.5.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.20.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.5.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.30.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.4.ffn_gate_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '               blk.25.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.17.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.17.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.22.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.37.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.18.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.3.ffn_down_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '              blk.3.ffn_gate_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.34.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.3.ffn_up_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.37.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.22.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.4.ffn_down_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '               blk.18.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.25.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.4.ffn_up_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '               blk.27.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.27.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.38.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.30.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.40.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.39.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.5.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.20.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.34.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.34.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.30.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.35.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.35.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.40.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.37.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.39.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.38.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.39.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.35.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.45.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.45.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.38.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: storing only 605 out of 659 entries
> 
> save_imatrix: stored collected data after 90 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-fairydreaming-llamacpp-76543311.dat
> [90]6175753.4634,[91]6190801.3912,[92]6215677.0108,[93]6199976.0739,[94]6211377.7885,[95]6239406.1645,[96]6231140.0390,[97]6298915.6241,[98]6316318.3182,[99]6300166.6172,
> save_imatrix: entry '             blk.60.ffn_down_exps.weight' has partial data (90.62%) - skipping
> save_imatrix: entry '             blk.60.ffn_gate_exps.weight' has partial data (90.62%) - skipping
> save_imatrix: entry '               blk.22.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.18.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.17.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.20.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.45.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.40.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.60.ffn_up_exps.weight' has partial data (90.62%) - skipping
> save_imatrix: entry '             blk.25.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.27.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.5.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.20.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.5.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.30.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.4.ffn_gate_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '               blk.25.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.17.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.17.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.22.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.37.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.18.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.3.ffn_down_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '              blk.3.ffn_gate_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.34.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.3.ffn_up_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.37.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.22.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.4.ffn_down_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '               blk.18.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.25.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.4.ffn_up_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '               blk.27.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.27.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.38.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.30.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.40.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.39.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.5.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.20.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.34.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.34.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.30.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.35.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.35.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.40.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.37.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.39.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.38.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.39.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.35.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.45.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.45.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.38.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: storing only 605 out of 659 entries
> 
> save_imatrix: stored collected data after 100 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-fairydreaming-llamacpp-76543311.dat
> [100]6301109.9866,[101]6324614.7445,[102]6351151.6391,[103]6325758.2099,[104]6363894.8439,[105]6424071.1082,[106]6446152.2034,[107]6428068.0699,[108]6469490.7110,[109]6463960.7281,
> save_imatrix: entry '             blk.60.ffn_down_exps.weight' has partial data (91.02%) - skipping
> save_imatrix: entry '             blk.60.ffn_gate_exps.weight' has partial data (91.02%) - skipping
> save_imatrix: entry '               blk.22.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.18.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.17.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.20.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.45.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.40.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.60.ffn_up_exps.weight' has partial data (91.02%) - skipping
> save_imatrix: entry '             blk.25.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.27.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.5.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.20.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.5.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.30.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.4.ffn_gate_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '               blk.25.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.17.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.17.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.22.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.37.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.18.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.3.ffn_down_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '              blk.3.ffn_gate_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.34.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.3.ffn_up_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.37.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.22.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.4.ffn_down_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '               blk.18.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.25.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.4.ffn_up_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '               blk.27.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.27.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.38.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.30.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.40.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.39.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.5.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.20.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.34.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.34.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.30.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.35.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.35.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.40.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.37.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.39.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.38.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.39.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.35.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.45.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.45.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.38.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: storing only 605 out of 659 entries
> 
> save_imatrix: stored collected data after 110 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-fairydreaming-llamacpp-76543311.dat
> [110]6507330.6507,[111]6513544.1907,[112]6513858.2784,[113]6517686.6742,[114]6491948.3340,[115]6486339.1162,[116]6494041.7263,[117]6465172.7777,[118]6482285.5172,[119]6504278.7278,
> save_imatrix: entry '             blk.60.ffn_down_exps.weight' has partial data (91.02%) - skipping
> save_imatrix: entry '             blk.60.ffn_gate_exps.weight' has partial data (91.02%) - skipping
> save_imatrix: entry '               blk.22.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.18.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.17.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.20.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.45.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.40.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.60.ffn_up_exps.weight' has partial data (91.02%) - skipping
> save_imatrix: entry '             blk.27.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.5.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.20.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.5.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.30.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.4.ffn_gate_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '               blk.17.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.17.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.22.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.37.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.18.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.3.ffn_down_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '              blk.3.ffn_gate_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.34.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.3.ffn_up_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.37.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.22.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.4.ffn_down_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '               blk.18.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.4.ffn_up_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '               blk.27.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.27.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.38.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.30.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.40.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.39.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.5.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.20.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.34.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.34.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.30.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.35.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.35.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.40.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.37.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.39.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.38.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.39.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.35.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.45.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.45.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.38.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: storing only 608 out of 659 entries
> 
> save_imatrix: stored collected data after 120 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-fairydreaming-llamacpp-76543311.dat
> [120]6476396.8031,[121]6432330.9894,[122]6410350.7446,[123]6445130.3947,[124]6455481.2395,[125]6458371.6557,[126]6459901.8849,[127]6455678.6998,[128]6420879.9721,[129]6427934.7804,
> save_imatrix: entry '             blk.60.ffn_down_exps.weight' has partial data (91.02%) - skipping
> save_imatrix: entry '             blk.60.ffn_gate_exps.weight' has partial data (91.02%) - skipping
> save_imatrix: entry '               blk.22.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.17.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.20.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.45.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.40.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.60.ffn_up_exps.weight' has partial data (91.02%) - skipping
> save_imatrix: entry '             blk.27.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.5.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.20.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.5.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.30.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.4.ffn_gate_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '               blk.17.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.17.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.22.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.37.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.3.ffn_down_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '              blk.3.ffn_gate_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.34.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.3.ffn_up_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.37.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.22.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.4.ffn_down_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '                blk.4.ffn_up_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '               blk.27.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.27.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.38.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.30.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.40.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.39.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.5.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.20.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.34.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.34.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.30.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.35.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.35.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.40.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.37.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.39.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.38.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.39.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.35.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.45.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.45.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.38.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: storing only 611 out of 659 entries
> 
> save_imatrix: stored collected data after 130 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-fairydreaming-llamacpp-76543311.dat
> [130]6476450.2766,[131]6486585.4775,[132]6480990.7546,[133]6505868.5457,[134]6485907.5957,[135]6506688.5466,[136]6508671.7673,[137]6539504.1261,[138]6551740.8463,[139]6538556.2192,
> save_imatrix: entry '             blk.60.ffn_down_exps.weight' has partial data (91.02%) - skipping
> save_imatrix: entry '             blk.60.ffn_gate_exps.weight' has partial data (91.02%) - skipping
> save_imatrix: entry '               blk.22.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.17.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.20.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.45.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.40.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.60.ffn_up_exps.weight' has partial data (91.02%) - skipping
> save_imatrix: entry '             blk.27.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.5.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.20.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.5.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.30.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.4.ffn_gate_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '               blk.17.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.17.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.22.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.37.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.3.ffn_down_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '              blk.3.ffn_gate_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.34.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.3.ffn_up_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.37.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.22.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.4.ffn_down_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '                blk.4.ffn_up_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '               blk.27.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.27.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.38.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.30.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.40.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.39.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.5.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.20.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.34.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.34.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.30.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.35.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.35.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.40.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.37.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.39.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.38.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.39.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.35.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.45.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.45.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.38.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: storing only 611 out of 659 entries
> 
> save_imatrix: stored collected data after 140 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-fairydreaming-llamacpp-76543311.dat
> [140]6495577.0276,[141]6538584.4614,[142]6531648.2379,[143]6521259.6067,[144]6512718.8535,[145]6505180.7797,[146]6499838.5139,[147]6507039.8622,[148]6509619.3561,[149]6490918.4871,
> save_imatrix: entry '             blk.60.ffn_down_exps.weight' has partial data (91.02%) - skipping
> save_imatrix: entry '             blk.60.ffn_gate_exps.weight' has partial data (91.02%) - skipping
> save_imatrix: entry '               blk.22.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.17.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.20.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.45.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.40.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.60.ffn_up_exps.weight' has partial data (91.02%) - skipping
> save_imatrix: entry '             blk.27.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.5.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.20.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.5.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.30.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.4.ffn_gate_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '               blk.17.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.17.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.22.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.37.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.3.ffn_down_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '              blk.3.ffn_gate_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.34.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.3.ffn_up_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.37.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.22.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.4.ffn_down_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '                blk.4.ffn_up_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '               blk.27.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.27.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.38.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.30.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.40.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.39.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.5.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.20.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.34.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.34.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.30.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.35.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.35.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.40.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.37.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.39.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.38.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.39.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.35.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.45.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.45.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.38.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: storing only 611 out of 659 entries
> 
> save_imatrix: stored collected data after 150 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-fairydreaming-llamacpp-76543311.dat
> [150]6492792.8736,[151]6482801.5910,[152]6474426.1334,[153]6468425.1052,[154]6462858.2172,[155]6466011.0127,[156]6465413.5613,[157]6448193.7327,[158]6458209.8959,[159]6443204.7725,
> save_imatrix: entry '             blk.60.ffn_down_exps.weight' has partial data (91.02%) - skipping
> save_imatrix: entry '             blk.60.ffn_gate_exps.weight' has partial data (91.02%) - skipping
> save_imatrix: entry '             blk.17.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.20.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.45.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.40.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.60.ffn_up_exps.weight' has partial data (91.02%) - skipping
> save_imatrix: entry '             blk.27.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.5.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.20.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.5.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.30.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.4.ffn_gate_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '               blk.17.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.17.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.37.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.3.ffn_down_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '              blk.3.ffn_gate_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.34.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.3.ffn_up_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.37.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.4.ffn_down_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '                blk.4.ffn_up_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '               blk.27.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.27.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.38.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.30.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.40.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.39.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.5.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.20.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.34.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.34.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.30.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.35.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.35.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.40.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.37.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.39.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.38.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.39.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.35.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.45.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.45.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.38.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: storing only 614 out of 659 entries
> 
> save_imatrix: stored collected data after 160 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-fairydreaming-llamacpp-76543311.dat
> [160]6429567.2711,[161]6427941.9669,[162]6415660.0404,[163]6392363.2083,[164]6401388.0519,[165]6375799.2275,[166]6377644.0495,[167]6410331.8695,[168]6426651.1307,[169]6435519.7345,
> save_imatrix: entry '             blk.60.ffn_down_exps.weight' has partial data (91.41%) - skipping
> save_imatrix: entry '             blk.60.ffn_gate_exps.weight' has partial data (91.41%) - skipping
> save_imatrix: entry '             blk.17.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.20.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.45.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.40.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.60.ffn_up_exps.weight' has partial data (91.41%) - skipping
> save_imatrix: entry '             blk.27.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.5.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.20.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.5.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.30.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.4.ffn_gate_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '               blk.17.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.17.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.37.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.3.ffn_down_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '              blk.3.ffn_gate_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.34.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.3.ffn_up_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.37.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.4.ffn_down_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '                blk.4.ffn_up_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '               blk.27.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.27.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.38.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.30.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.40.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.39.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.5.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.20.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.34.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.34.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.30.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.35.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.35.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.40.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.37.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.39.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.38.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.39.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.35.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.45.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.45.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.38.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: storing only 614 out of 659 entries
> 
> save_imatrix: stored collected data after 170 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-fairydreaming-llamacpp-76543311.dat
> [170]6428769.8846,[171]6419611.4694,[172]6411973.9782,[173]6451479.6905,[174]6457102.8772,[175]6458849.4832,[176]6439765.8921,[177]6417731.4902,[178]6403329.4114,[179]6383100.6990,
> save_imatrix: entry '             blk.60.ffn_down_exps.weight' has partial data (91.41%) - skipping
> save_imatrix: entry '             blk.60.ffn_gate_exps.weight' has partial data (91.41%) - skipping
> save_imatrix: entry '             blk.17.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.20.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.45.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.40.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.60.ffn_up_exps.weight' has partial data (91.41%) - skipping
> save_imatrix: entry '             blk.27.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.5.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.20.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.5.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.30.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.4.ffn_gate_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '               blk.17.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.17.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.37.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.3.ffn_down_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '              blk.3.ffn_gate_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.34.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.3.ffn_up_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.37.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.4.ffn_down_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '                blk.4.ffn_up_exps.weight' has partial data (98.05%) - skipping
> save_imatrix: entry '               blk.27.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.27.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.38.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.30.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.40.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.39.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.5.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.20.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.34.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.34.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.30.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.35.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.35.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.40.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.37.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.39.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.38.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.39.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.35.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.45.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.45.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.38.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: storing only 614 out of 659 entries
> 
> save_imatrix: stored collected data after 180 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-fairydreaming-llamacpp-76543311.dat
> [180]6391487.5872,[181]6383063.6482,[182]6373383.3402,[183]6363548.5557,[184]6376991.9936,[185]6370068.3041,[186]6379028.8137,[187]6365403.7968,[188]6366442.4816,[189]6359016.9848,
> save_imatrix: entry '             blk.60.ffn_down_exps.weight' has partial data (91.41%) - skipping
> save_imatrix: entry '             blk.60.ffn_gate_exps.weight' has partial data (91.41%) - skipping
> save_imatrix: entry '             blk.17.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.20.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.45.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.40.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.60.ffn_up_exps.weight' has partial data (91.41%) - skipping
> save_imatrix: entry '             blk.27.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.5.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.20.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.5.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.30.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.4.ffn_gate_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '               blk.17.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.17.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.37.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.3.ffn_down_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '              blk.3.ffn_gate_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.34.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.3.ffn_up_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.37.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.4.ffn_down_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '                blk.4.ffn_up_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '               blk.27.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.27.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.38.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.30.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.40.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.39.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.5.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.20.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.34.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.34.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.30.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.35.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.35.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.40.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.37.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.39.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.38.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.39.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.35.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.45.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.45.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.38.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: storing only 614 out of 659 entries
> 
> save_imatrix: stored collected data after 190 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-fairydreaming-llamacpp-76543311.dat
> [190]6366114.0439,[191]6347658.8125,[192]6350871.2032,[193]6345914.8868,[194]6353433.2773,[195]6344337.2567,[196]6359380.0162,[197]6356840.3002,[198]6349110.5048,[199]6336902.1625,
> save_imatrix: entry '             blk.60.ffn_down_exps.weight' has partial data (91.41%) - skipping
> save_imatrix: entry '             blk.60.ffn_gate_exps.weight' has partial data (91.41%) - skipping
> save_imatrix: entry '             blk.17.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.20.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.45.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.40.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.60.ffn_up_exps.weight' has partial data (91.41%) - skipping
> save_imatrix: entry '             blk.27.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.5.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.20.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.5.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.30.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.4.ffn_gate_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '               blk.17.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.17.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.37.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.3.ffn_down_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '              blk.3.ffn_gate_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.34.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.3.ffn_up_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.37.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.4.ffn_down_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '                blk.4.ffn_up_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '               blk.27.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.27.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.38.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.30.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.40.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.39.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.5.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.20.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.34.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.34.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.30.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.35.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.35.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.40.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.37.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.39.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.38.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.39.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.35.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.45.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.45.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.38.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: storing only 614 out of 659 entries
> 
> save_imatrix: stored collected data after 200 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-fairydreaming-llamacpp-76543311.dat
> [200]6330401.3814,[201]6374236.3913,[202]6386271.5405,[203]6390608.9919,[204]6428234.2483,[205]6440615.5978,[206]6438135.2383,[207]6458495.0429,[208]6450338.4535,[209]6443037.4588,
> save_imatrix: entry '             blk.60.ffn_down_exps.weight' has partial data (91.41%) - skipping
> save_imatrix: entry '             blk.60.ffn_gate_exps.weight' has partial data (91.41%) - skipping
> save_imatrix: entry '             blk.17.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.20.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.45.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.40.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.60.ffn_up_exps.weight' has partial data (91.41%) - skipping
> save_imatrix: entry '             blk.27.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.5.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.20.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.5.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.30.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.4.ffn_gate_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '               blk.17.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.17.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.37.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.3.ffn_down_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '              blk.3.ffn_gate_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.34.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.3.ffn_up_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.37.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.4.ffn_down_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '                blk.4.ffn_up_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '               blk.27.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.27.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.38.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.30.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.40.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.39.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.5.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.20.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.34.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.34.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.30.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.35.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.35.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.40.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.37.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.39.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.38.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.39.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.35.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.45.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.45.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.38.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: storing only 614 out of 659 entries
> 
> save_imatrix: stored collected data after 210 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-fairydreaming-llamacpp-76543311.dat
> [210]6447980.5077,[211]6475482.7036,[212]6484583.7694,[213]6476309.6415,
> Final estimate: PPL = 6476309.6415 +/- 108643.32717
> 
> save_imatrix: entry '             blk.60.ffn_down_exps.weight' has partial data (91.41%) - skipping
> save_imatrix: entry '             blk.60.ffn_gate_exps.weight' has partial data (91.41%) - skipping
> save_imatrix: entry '               blk.20.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.45.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.40.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.60.ffn_up_exps.weight' has partial data (91.41%) - skipping
> save_imatrix: entry '             blk.27.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.5.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.20.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.5.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.30.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.4.ffn_gate_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '             blk.37.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.3.ffn_down_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '              blk.3.ffn_gate_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.34.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '                blk.3.ffn_up_exps.weight' has partial data (98.83%) - skipping
> save_imatrix: entry '               blk.37.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '              blk.4.ffn_down_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '                blk.4.ffn_up_exps.weight' has partial data (98.44%) - skipping
> save_imatrix: entry '               blk.27.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.27.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.38.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.30.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.40.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.39.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '              blk.5.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.20.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.34.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.34.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.30.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '               blk.35.ffn_up_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.35.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.40.ffn_down_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.37.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.39.ffn_down_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.38.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.39.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '             blk.35.ffn_gate_exps.weight' has partial data (99.22%) - skipping
> save_imatrix: entry '               blk.45.ffn_up_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.45.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: entry '             blk.38.ffn_gate_exps.weight' has partial data (99.61%) - skipping
> save_imatrix: storing only 617 out of 659 entries
> 
> save_imatrix: stored collected data after 213 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-fairydreaming-llamacpp-76543311.dat
> 
> llama_perf_context_print:        load time =   40703.46 ms
> llama_perf_context_print: prompt eval time = 7249119.69 ms / 109056 tokens (   66.47 ms per token,    15.04 tokens per second)
> llama_perf_context_print:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
> llama_perf_context_print:       total time = 7322634.52 ms / 109057 tokens
> ```
> </details>
> 
> 👤 **saood06** replied the **2025-03-25** at **23:00:04**:<br>
> >The output is different and it seems to be skipping 10-15% of the tensors due to partial 99.9% data...
> 
> This is to be expected as long as this `storing only 605 out of 659 entries` number keeps trending up to 659 you should be good.
> 
> The warnings here are different because of https://github.com/ikawrakow/ik_llama.cpp/pull/202
> 
> Llama.cpp is more strict, but hopefully by the time you get through all your chunks they should all be activated.
> 
> The real concern though is the perplexity numbers, they seem way too high, even though I've never made an imatrix that still looks concerning.
> 
> Edit: Actually maybe this will prove a clue to what is wrong as this implementation also seems unstable.
> 
> 👤 **ubergarm** replied the **2025-03-25** at **23:46:34**:<br>
> > The real concern though is the perplexity numbers, they seem way too high
> 
> Yeah I was wondering why they are so much higher than on this fork. They did seem to get trend smaller quickly at first though hah... Still running, I'll paste the rest of the logs when it is done
> 
> > Edit: Actually maybe this will prove a clue to what is wrong as this implementation also seems unstable.
> 
> Right, if it isn't working either, something is odd. I wonder how bartowski made his? I compared it and it has a different sha256 than another older V3 I saw on hugging face publicly, so not sure the details. One of the threads you had linked me above he mentions a flag or another branch maybe. Anyway...
> 
> ## First Test
> So I went ahead and used the mysterious bartowski importance matrix data to cook my first `IQ2_K_R4`
> 
> <details>
> <summary>`DeepSeek-V3-0324-IQ2_K_R4-bartowski-imat.gguf`</summary>
> 
> ```
> 227G    /mnt/raid/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-IQ2_K_R4-bartowski-imat.gguf
> 
> llama_model_loader: - type  f32:  361 tensors
> llama_model_loader: - type q8_0:  612 tensors
> llama_model_loader: - type iq2_k_r4:  116 tensors
> llama_model_loader: - type iq3_k_r4:   58 tensors
> ...
> llm_load_tensors: offloaded 62/62 layers to GPU
> llm_load_tensors:        CPU buffer size = 228404.85 MiB
> llm_load_tensors:        CPU buffer size =   938.98 MiB
> llm_load_tensors:      CUDA0 buffer size = 17744.02 MiB
> ....................................................................................................
> llama_new_context_with_model: n_ctx      = 65536
> llama_new_context_with_model: n_batch    = 2048
> llama_new_context_with_model: n_ubatch   = 512
> llama_new_context_with_model: flash_attn = 1
> llama_new_context_with_model: mla_attn   = 2
> llama_new_context_with_model: attn_max_b = 512
> llama_new_context_with_model: fused_moe  = 1
> llama_new_context_with_model: ser        = -1, 0
> llama_new_context_with_model: freq_base  = 10000.0
> llama_new_context_with_model: freq_scale = 0.025
> ...
> llama_kv_cache_init:      CUDA0 KV buffer size =  2333.28 MiB
> llama_new_context_with_model: KV self size  = 2333.25 MiB, c^KV (q8_0): 2333.25 MiB, kv^T: not used
> llama_new_context_with_model:  CUDA_Host  output buffer size =     0.99 MiB
> llama_new_context_with_model:      CUDA0 compute buffer size =  6081.00 MiB
> llama_new_context_with_model:  CUDA_Host compute buffer size =   240.01 MiB
> llama_new_context_with_model: graph nodes  = 13613
> llama_new_context_with_model: graph splits = 118
> 
> INFO [           print_timings] prompt eval time     =   38158.44 ms /  3682 tokens (   10.36 ms per token,    96.49 tokens per second) | tid="139663225118720" timestamp=1742946073 id_slot=0 id_task=0 t_prompt_processing=38158.439 n_prompt_tokens_processed=3682 t_token=10.363508690928843 n_tokens_second=96.4924167888524
> INFO [           print_timings] generation eval time =  444729.93 ms /  4907 runs   (   90.63 ms per token,    11.03 tokens per second) | tid="139663225118720" timestamp=1742946073 id_slot=0 id_task=0 t_token_generation=444729.926 n_decoded=4907 t_token=90.63173547992663 n_tokens_second=11.033662708814427
> INFO [           print_timings]           total time =  482888.36 ms | tid="139663225118720" timestamp=1742946073 id_slot=0 id_task=0 t_prompt_processing=38158.439 t_token_generation=444729.926 t_total=482888.365
> ```
> 
> </details>
> 
> Running at 64k context it is using `26732MiB`... I wonder what the least damaging `q8_0`s to knock down in the GPU layers to fit this in 24GB VRAM. Would need to shave off just over 2GiB of tensors out of a total of ~17.33 so like maybe dense layers to q6 might do it... probably need a spreadsheet lol...
> 
> Looks anecdotally like around 95 tok/sec pp on a <~4k prompt and 11 tok/sec generation. Generation seems a bit slower while copying markdown table logs haha... Initial impression is I don't miss `<think>` as it gets right to the point haha... I'll test to see if it can make any graphs of my log data! Oh right and set `temperature=0.3`.
> 
> 👤 **saood06** replied the **2025-03-26** at **01:18:42**:<br>
> > Right, if it isn't working either, something is odd. I wonder how bartowski made his? 
> 
> Using main llama.cpp, it seems that the MLA attention is causing problems.
> 
> >I compared it and it has a different sha256 than another older V3 I saw on hugging face publicly
> 
> Well yes, I've seen plenty of people make imatrix for the Deepseek V3 family of models, Q8_0 on main should work, and there's a good chance BF16 and Q6_K would work here.
> > ## First Test
> > 
> > So I went ahead and used the mysterious bartowski importance matrix data to cook my first `IQ2_K_R4`
> > `DeepSeek-V3-0324-IQ2_K_R4-bartowski-imat.gguf`
> 
> Nice.
> 
> > Running at 64k context it is using `26732MiB`... I wonder what the least damaging `q8_0`s to knock down in the GPU layers to fit this in 24GB VRAM. Would need to shave off just over 2GiB of tensors out of a total of ~17.33 so like maybe dense layers to q6 might do it... probably need a spreadsheet lol...
> 
> By my napkin math you'd need to set around 60% to Q6_K (or set some weights even lower).
> 
> Take some inspiration from the low bit quant recipes [here](https://github.com/ikawrakow/ik_llama.cpp/blob/a22250df93fd833a6cb7f310b159ad1b54e4d582/src/llama.cpp#L16765)  or in the unsloth repo [here](https://github.com/ggml-org/llama.cpp/compare/master...unslothai:llama.cpp:master).
> 
> The code might be a bit spread out, but it is very easy to understand, and I'm sure it will help you find the 2GiB you need to cut.
> 
> > Looks anecdotally like around 95 tok/sec pp on a <~4k prompt and 11 tok/sec generation. Generation seems a bit slower while copying markdown table logs haha... 
> 
> That sounds so nice, I'm struggling here with my PP at 0 context (~10.5) slower than your TG at 4k.
> 
> >Initial impression is I don't miss `<think>` as it gets right to the point haha... I'll test to see if it can make any graphs of my log data! 
> 
> That does sound like a nice use, python graphing always felt a bit more tedious to me. I used to use a lot of R for graphing.
> 
> >Oh right and set `temperature=0.3`.
> 
> What have you been running at, and did 0.3 feel appropriate? (also anything else in the sampler chain, like top p/k, min p, mirostat etc.)
> 
> 👤 **ubergarm** replied the **2025-03-26** at **02:23:57**:<br>
> > it seems that the MLA attention is causing problems
> 
> Yeah, good point, using mainline without MLA is probably fine. I got the files copied over, but didn't try running it as I just went with bartowski's without MLA for now then. Makes sense after you explain it.
> 
> > The code might be a bit spread out, but it is very easy to understand, and I'm sure it will help you find the 2GiB you need to cut.
> 
> Ahh okay, I had seen that unsloth fork before, but now having quantized the model enough times here, I can understand what is happening now. And right looks like `q6_k` for `ffn_down.weight` in the first 3 dense layers and `ffn_down_shexp.weight` shared experts is a good place to start trimming a bit.
> 
> > I'm struggling here with my PP at 0 context (~10.5)
> 
> Hrmm, I didn't actually bench it just did one `llama-server` call API call. Will kick the tires on it more later this week and get a more proper benchmark.
> 
> > What have you been running at, and did 0.3 feel appropriate?
> 
> I use a small custom python chat client that uses `litellm` to hit the OpenAI API chat endpoint. The first time i forgot and left it at R1 default of `0.6` which possibly had some funky code generation or my terminal got borked. I set it to `0.3` and re-ran while not resizing my terminal and things looks good. The only things I ever specify are `top_p=0.95` and `temperature` as mentioned above. I generally keep it simple for coding generations.
> 
> In the past I have played with samplers more, especially when trying to reduce slop and increase creativity in writing. I would increase temperature, adjust `top_p`, `min_p`, `top_k`, and even played around a bit with the more specialized samplers like [xtc](https://github.com/ggml-org/llama.cpp/blob/master/examples/main/README.md#xtc-sampling). Anymore I haven't fussed with it much, and spend more time adding variance into the prompt like example clips etc.
> 
> 👤 **ubergarm** replied the **2025-03-26** at **02:28:19**:<br>
> @saood06 
> 
> I got a perplexity run for the `DeepSeek-V3-0324-IQ2_K_R4-bartowski-imat.gguf`.
> 
> <details>
> 
> <summary>llama-perplexity Logs</summary>
> 
> ```bash
> CUDA_VISIBLE_DEVICES="0," \
> ./build/bin/llama-perplexity \
>     --model /mnt/raid/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-IQ2_K_R4-bartowski-imat.gguf \
>     -ctk q8_0 \
>     -mla 2 -fa \
>     -amb 512 \
>     -fmoe \
>     --ctx-size 512 \
>     --ubatch-size 512 \
>     -f wiki.test.raw \
>     --seed 1337 \
>     --n-gpu-layers 63 \
>     --override-tensor exps=CPU \
>     --threads 24
> 
> ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
> ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
> ggml_cuda_init: found 1 CUDA devices:
>   Device 0: NVIDIA RTX A6000, compute capability 8.6, VMM: yes
> main: build = 3608 (98a264a2)
> main: built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu
> main: seed  = 1337
> llama_model_loader: loaded meta data with 50 key-value pairs and 1147 tensors from /mnt/raid/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-IQ2_K_R4-bartowski-imat.gguf (version GGUF V3 (latest))
> llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
> llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
> llama_model_loader: - kv   1:                               general.type str              = model
> llama_model_loader: - kv   2:                               general.name str              = DeepSeek V3 0324
> llama_model_loader: - kv   3:                            general.version str              = V3-0324
> llama_model_loader: - kv   4:                           general.basename str              = DeepSeek
> llama_model_loader: - kv   5:                         general.size_label str              = 256x21B
> llama_model_loader: - kv   6:                            general.license str              = mit
> llama_model_loader: - kv   7:                      deepseek2.block_count u32              = 61
> llama_model_loader: - kv   8:                   deepseek2.context_length u32              = 163840
> llama_model_loader: - kv   9:                 deepseek2.embedding_length u32              = 7168
> llama_model_loader: - kv  10:              deepseek2.feed_forward_length u32              = 18432
> llama_model_loader: - kv  11:             deepseek2.attention.head_count u32              = 128
> llama_model_loader: - kv  12:          deepseek2.attention.head_count_kv u32              = 128
> llama_model_loader: - kv  13:                   deepseek2.rope.freq_base f32              = 10000.000000
> llama_model_loader: - kv  14: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
> llama_model_loader: - kv  15:                deepseek2.expert_used_count u32              = 8
> llama_model_loader: - kv  16:                          general.file_type u32              = 338
> llama_model_loader: - kv  17:        deepseek2.leading_dense_block_count u32              = 3
> llama_model_loader: - kv  18:                       deepseek2.vocab_size u32              = 129280
> llama_model_loader: - kv  19:            deepseek2.attention.q_lora_rank u32              = 1536
> llama_model_loader: - kv  20:           deepseek2.attention.kv_lora_rank u32              = 512
> llama_model_loader: - kv  21:             deepseek2.attention.key_length u32              = 192
> llama_model_loader: - kv  22:           deepseek2.attention.value_length u32              = 128
> llama_model_loader: - kv  23:       deepseek2.expert_feed_forward_length u32              = 2048
> llama_model_loader: - kv  24:                     deepseek2.expert_count u32              = 256
> llama_model_loader: - kv  25:              deepseek2.expert_shared_count u32              = 1
> llama_model_loader: - kv  26:             deepseek2.expert_weights_scale f32              = 2.500000
> llama_model_loader: - kv  27:              deepseek2.expert_weights_norm bool             = true
> llama_model_loader: - kv  28:               deepseek2.expert_gating_func u32              = 2
> llama_model_loader: - kv  29:             deepseek2.rope.dimension_count u32              = 64
> llama_model_loader: - kv  30:                deepseek2.rope.scaling.type str              = yarn
> llama_model_loader: - kv  31:              deepseek2.rope.scaling.factor f32              = 40.000000
> llama_model_loader: - kv  32: deepseek2.rope.scaling.original_context_length u32              = 4096
> llama_model_loader: - kv  33: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
> llama_model_loader: - kv  34:                       tokenizer.ggml.model str              = gpt2
> llama_model_loader: - kv  35:                         tokenizer.ggml.pre str              = deepseek-v3
> llama_model_loader: - kv  36:                      tokenizer.ggml.tokens arr[str,129280]  = ["
> llama_model_loader: - kv  37:                  tokenizer.ggml.token_type arr[i32,129280]  = [3
> llama_model_loader: - kv  38:                      tokenizer.ggml.merges arr[str,127741]  = ["
> llama_model_loader: - kv  39:                tokenizer.ggml.bos_token_id u32              = 0
> llama_model_loader: - kv  40:                tokenizer.ggml.eos_token_id u32              = 1
> llama_model_loader: - kv  41:            tokenizer.ggml.padding_token_id u32              = 1
> llama_model_loader: - kv  42:               tokenizer.ggml.add_bos_token bool             = true
> llama_model_loader: - kv  43:               tokenizer.ggml.add_eos_token bool             = false
> llama_model_loader: - kv  44:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
> llama_model_loader: - kv  45:               general.quantization_version u32              = 2
> llama_model_loader: - kv  46:                      quantize.imatrix.file str              = /mnt/raid/models/ubergarm/DeepSeek-V3...
> llama_model_loader: - kv  47:                   quantize.imatrix.dataset str              = /workspace/calibration_datav3.txt
> llama_model_loader: - kv  48:             quantize.imatrix.entries_count i32              = 720
> llama_model_loader: - kv  49:              quantize.imatrix.chunks_count i32              = 124
> llama_model_loader: - type  f32:  361 tensors
> llama_model_loader: - type q8_0:  612 tensors
> llama_model_loader: - type iq2_k_r4:  116 tensors
> llama_model_loader: - type iq3_k_r4:   58 tensors
> llm_load_vocab: special tokens cache size = 818
> llm_load_vocab: token to piece cache size = 0.8223 MB
> llm_load_print_meta: format           = GGUF V3 (latest)
> llm_load_print_meta: arch             = deepseek2
> llm_load_print_meta: vocab type       = BPE
> llm_load_print_meta: n_vocab          = 129280
> llm_load_print_meta: n_merges         = 127741
> llm_load_print_meta: vocab_only       = 0
> llm_load_print_meta: n_ctx_train      = 163840
> llm_load_print_meta: n_embd           = 7168
> llm_load_print_meta: n_layer          = 61
> llm_load_print_meta: n_head           = 128
> llm_load_print_meta: n_head_kv        = 128
> llm_load_print_meta: n_rot            = 64
> llm_load_print_meta: n_swa            = 0
> llm_load_print_meta: n_embd_head_k    = 192
> llm_load_print_meta: n_embd_head_v    = 128
> llm_load_print_meta: n_gqa            = 1
> llm_load_print_meta: n_embd_k_gqa     = 24576
> llm_load_print_meta: n_embd_v_gqa     = 16384
> llm_load_print_meta: f_norm_eps       = 0.0e+00
> llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
> llm_load_print_meta: f_clamp_kqv      = 0.0e+00
> llm_load_print_meta: f_max_alibi_bias = 0.0e+00
> llm_load_print_meta: f_logit_scale    = 0.0e+00
> llm_load_print_meta: n_ff             = 18432
> llm_load_print_meta: n_expert         = 256
> llm_load_print_meta: n_expert_used    = 8
> llm_load_print_meta: causal attn      = 1
> llm_load_print_meta: pooling type     = 0
> llm_load_print_meta: rope type        = 0
> llm_load_print_meta: rope scaling     = yarn
> llm_load_print_meta: freq_base_train  = 10000.0
> llm_load_print_meta: freq_scale_train = 0.025
> llm_load_print_meta: n_ctx_orig_yarn  = 4096
> llm_load_print_meta: rope_finetuned   = unknown
> llm_load_print_meta: ssm_d_conv       = 0
> llm_load_print_meta: ssm_d_inner      = 0
> llm_load_print_meta: ssm_d_state      = 0
> llm_load_print_meta: ssm_dt_rank      = 0
> llm_load_print_meta: model type       = 671B
> llm_load_print_meta: model ftype      = IQ2_K_R4 - 2.375 bpw
> llm_load_print_meta: model params     = 672.050 B
> llm_load_print_meta: model size       = 226.003 GiB (2.889 BPW) 
> llm_load_print_meta: repeating layers = 224.169 GiB (2.873 BPW, 670.196 B parameters)
> llm_load_print_meta: general.name     = DeepSeek V3 0324
> llm_load_print_meta: BOS token        = 0 '<
> llm_load_print_meta: EOS token        = 1 '<
> llm_load_print_meta: PAD token        = 1 '<
> llm_load_print_meta: LF token         = 131 '
> llm_load_print_meta: max token length = 256
> llm_load_print_meta: n_layer_dense_lead   = 3
> llm_load_print_meta: n_lora_q             = 1536
> llm_load_print_meta: n_lora_kv            = 512
> llm_load_print_meta: n_ff_exp             = 2048
> llm_load_print_meta: n_expert_shared      = 1
> llm_load_print_meta: expert_weights_scale = 2.5
> llm_load_print_meta: expert_weights_norm  = 1
> llm_load_print_meta: expert_gating_func   = sigmoid
> llm_load_print_meta: rope_yarn_log_mul    = 0.1000
> llm_load_tensors: ggml ctx size =    0.93 MiB
> Tensor blk.3.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.3.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.3.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.4.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.4.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.4.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.5.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.5.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.5.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.6.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.6.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.6.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.7.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.7.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.8.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.8.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.9.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.9.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.9.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.10.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.10.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.11.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.11.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.11.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.12.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.12.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.13.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.13.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.13.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.14.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.14.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.15.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.15.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.16.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.16.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.16.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.17.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.17.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.17.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.18.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.18.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.18.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.19.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.19.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.19.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.20.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.20.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.20.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.21.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.21.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.21.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.22.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.22.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.22.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.23.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.23.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.23.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.24.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.24.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.24.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.25.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.25.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.25.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.26.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.26.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.26.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.27.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.27.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.27.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.28.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.28.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.28.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.29.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.29.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.29.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.30.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.30.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.30.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.31.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.31.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.31.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.32.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.32.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.32.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.33.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.33.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.33.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.34.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.34.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.34.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.35.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.35.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.35.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.36.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.36.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.36.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.37.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.37.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.37.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.38.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.38.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.38.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.39.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.39.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.39.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.40.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.40.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.40.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.41.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.41.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.41.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.42.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.42.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.42.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.43.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.43.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.43.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.44.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.44.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.44.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.45.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.45.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.45.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.46.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.46.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.46.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.47.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.47.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.47.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.48.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.48.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.48.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.49.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.49.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.49.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.50.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.50.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.50.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.51.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.51.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.51.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.52.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.52.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.52.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.53.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.53.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.53.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.54.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.54.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.54.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.55.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.55.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.55.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.56.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.56.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.56.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.57.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.57.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.57.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.58.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.58.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.58.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.59.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.59.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.59.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.60.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.60.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.60.ffn_up_exps.weight buffer type overriden to CPU
> llm_load_tensors: offloading 61 repeating layers to GPU
> llm_load_tensors: offloading non-repeating layers to GPU
> llm_load_tensors: offloaded 62/62 layers to GPU
> llm_load_tensors:        CPU buffer size = 228404.85 MiB
> llm_load_tensors:        CPU buffer size =   938.98 MiB
> llm_load_tensors:      CUDA0 buffer size = 17744.02 MiB
> ....................................................................................................
> llama_new_context_with_model: n_ctx      = 2048
> llama_new_context_with_model: n_batch    = 2048
> llama_new_context_with_model: n_ubatch   = 512
> llama_new_context_with_model: flash_attn = 1
> llama_new_context_with_model: mla_attn   = 2
> llama_new_context_with_model: attn_max_b = 512
> llama_new_context_with_model: fused_moe  = 1
> llama_new_context_with_model: ser        = -1, 0
> llama_new_context_with_model: freq_base  = 10000.0
> llama_new_context_with_model: freq_scale = 0.025
> llama_kv_cache_init: layer 0: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 1: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 2: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 3: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 4: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 5: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 6: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 7: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 8: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 9: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 10: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 11: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 12: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 13: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 14: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 15: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 16: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 17: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 18: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 19: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 20: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 21: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 22: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 23: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 24: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 25: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 26: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 27: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 28: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 29: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 30: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 31: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 32: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 33: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 34: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 35: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 36: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 37: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 38: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 39: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 40: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 41: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 42: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 43: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 44: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 45: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 46: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 47: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 48: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 49: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 50: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 51: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 52: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 53: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 54: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 55: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 56: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 57: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 58: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 59: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init: layer 60: n_embd_head_qk_rope = 64, kv_lora_rank = 512
> llama_kv_cache_init:      CUDA0 KV buffer size =    72.94 MiB
> llama_new_context_with_model: KV self size  =   72.91 MiB, c^KV (q8_0):   72.91 MiB, kv^T: not used
> llama_new_context_with_model:  CUDA_Host  output buffer size =     1.97 MiB
> llama_new_context_with_model:      CUDA0 compute buffer size =   503.00 MiB
> llama_new_context_with_model:  CUDA_Host compute buffer size =   162.01 MiB
> llama_new_context_with_model: graph nodes  = 3548
> llama_new_context_with_model: graph splits = 118
> 
> system_info: n_threads = 24 / 48 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | 
> perplexity: tokenizing the input ..
> perplexity: tokenization took 603.222 ms
> perplexity: calculating perplexity over 561 chunks, n_ctx=512, batch_size=2048, n_seq=4
> perplexity: 21.36 seconds per pass - ETA 49.93 minutes
> [1]2.7483,[2]3.4794,[3]2.4909,[4]2.1074,[5]1.8962,[6]1.7783,[7]1.6837,[8]1.6355,[9]1.5876,[10]1.5483,[11]1.5395,[12]1.5801,[13]1.5988,[14]1.7261,[15]1.8556,[16]1.9082,[17]2.0746,[18]2.2056,[19]2.1612,[20]2.1513,[21]2.2527,[22]2.2216,[23]2.1876,[24]2.2030,[25]2.1696,[26]2.1413,[27]2.1883,[28]2.1959,[29]2.2508,[30]2.2836,[31]2.3181,[32]2.3356,[33]2.3736,[34]2.4222,[35]2.4707,[36]2.5274,[37]2.5627,[38]2.6117,[39]2.6492,[40]2.7112,[41]2.7527,[42]2.7671,[43]2.8188,[44]2.8333,[45]2.9155,[46]2.9651,[47]2.9285,[48]2.8841,[49]2.8647,[50]2.8868,[51]2.9293,[52]2.9409,[53]2.9979,[54]3.0117,[55]3.0426,[56]3.0775,[57]3.0936,[58]3.1311,[59]3.1417,[60]3.1892,[61]3.2299,[62]3.2819,[63]3.3130,[64]3.3598,[65]3.3684,[66]3.3600,[67]3.3364,[68]3.3635,[69]3.3653,[70]3.3813,[71]3.3980,[72]3.4090,[73]3.4226,[74]3.4430,[75]3.4208,[76]3.3732,[77]3.3304,[78]3.3280,[79]3.3110,[80]3.2976,[81]3.2636,[82]3.2673,[83]3.2412,[84]3.2080,[85]3.1748,[86]3.1550,[87]3.1544,[88]3.1305,[89]3.1191,[90]3.0982,[91]3.0734,[92]3.0482,[93]3.0226,[94]3.0016,[95]2.9838,[96]2.9867,[97]2.9964,[98]2.9867,[99]2.9703,[100]2.9705,[101]2.9617,[102]2.9795,[103]3.0050,[104]3.0240,[105]3.0202,[106]3.0452,[107]3.0700,[108]3.0908,[109]3.1247,[110]3.1578,[111]3.1785,[112]3.1510,[113]3.1388,[114]3.1178,[115]3.1026,[116]3.0946,[117]3.0731,[118]3.0522,[119]3.0303,[120]3.0082,[121]2.9920,[122]2.9723,[123]2.9542,[124]2.9340,[125]2.9151,[126]2.8995,[127]2.8870,[128]2.8808,[129]2.8711,[130]2.8591,[131]2.8508,[132]2.8567,[133]2.8654,[134]2.8722,[135]2.8830,[136]2.8983,[137]2.9117,[138]2.9191,[139]2.9298,[140]2.9290,[141]2.9287,[142]2.9250,[143]2.9245,[144]2.9198,[145]2.9112,[146]2.9079,[147]2.9108,[148]2.9088,[149]2.9090,[150]2.9016,[151]2.8981,[152]2.8942,[153]2.8890,[154]2.8872,[155]2.8907,[156]2.8905,[157]2.8956,[158]2.9035,[159]2.9057,[160]2.9145,[161]2.9222,[162]2.9324,[163]2.9401,[164]2.9608,[165]2.9842,[166]3.0019,[167]3.0142,[168]3.0395,[169]3.0630,[170]3.0854,[171]3.1072,[172]3.0901,[173]3.0724,[174]3.0592,[175]3.0470,[176]3.0346,[177]3.0236,[178]3.0116,[179]2.9990,[180]3.0020,[181]3.0163,[182]3.0316,[183]3.0458,[184]3.0591,[185]3.0688,[186]3.0844,[187]3.1002,[188]3.1135,[189]3.1235,[190]3.1236,[191]3.1304,[192]3.1324,[193]3.1371,[194]3.1578,[195]3.1676,[196]3.1807,[197]3.1905,[198]3.1943,[199]3.1997,[200]3.1978,[201]3.2123,[202]3.2067,[203]3.2113,[204]3.2130,[205]3.2123,[206]3.2151,[207]3.2234,[208]3.2330,[209]3.2418,[210]3.2409,[211]3.2351,[212]3.2358,[213]3.2438,[214]3.2449,[215]3.2502,[216]3.2501,[217]3.2437,[218]3.2428,[219]3.2435,[220]3.2430,[221]3.2431,[222]3.2424,[223]3.2435,[224]3.2480,[225]3.2495,[226]3.2401,[227]3.2381,[228]3.2392,[229]3.2428,[230]3.2486,[231]3.2548,[232]3.2463,[233]3.2388,[234]3.2411,[235]3.2407,[236]3.2495,[237]3.2577,[238]3.2667,[239]3.2770,[240]3.2857,[241]3.2967,[242]3.3118,[243]3.3242,[244]3.3325,[245]3.3449,[246]3.3558,[247]3.3540,[248]3.3493,[249]3.3463,[250]3.3386,[251]3.3357,[252]3.3368,[253]3.3399,[254]3.3463,[255]3.3518,[256]3.3550,[257]3.3570,[258]3.3577,[259]3.3604,[260]3.3626,[261]3.3630,[262]3.3613,[263]3.3665,[264]3.3688,[265]3.3688,[266]3.3702,[267]3.3718,[268]3.3750,[269]3.3781,[270]3.3761,[271]3.3741,[272]3.3672,[273]3.3670,[274]3.3599,[275]3.3496,[276]3.3389,[277]3.3408,[278]3.3509,[279]3.3566,[280]3.3644,[281]3.3713,[282]3.3765,[283]3.3830,[284]3.3891,[285]3.4030,[286]3.4050,[287]3.4076,[288]3.4125,[289]3.4144,[290]3.4059,[291]3.3983,[292]3.3996,[293]3.3995,[294]3.3984,[295]3.3976,[296]3.3995,[297]3.4007,[298]3.4060,[299]3.4120,[300]3.4146,[301]3.4181,[302]3.4199,[303]3.4212,[304]3.4197,[305]3.4316,[306]3.4384,[307]3.4493,[308]3.4376,[309]3.4318,[310]3.4224,[311]3.4256,[312]3.4285,[313]3.4348,[314]3.4366,[315]3.4396,[316]3.4407,[317]3.4420,[318]3.4424,[319]3.4431,[320]3.4473,[321]3.4471,[322]3.4483,[323]3.4544,[324]3.4548,[325]3.4600,[326]3.4642,[327]3.4678,[328]3.4699,[329]3.4713,[330]3.4774,[331]3.4809,[332]3.4845,[333]3.4829,[334]3.4826,[335]3.4825,[336]3.4818,[337]3.4827,[338]3.4829,[339]3.4850,[340]3.4883,[341]3.4935,[342]3.5026,[343]3.5117,[344]3.5166,[345]3.5086,[346]3.5018,[347]3.4991,[348]3.4919,[349]3.4879,[350]3.4866,[351]3.4912,[352]3.5062,[353]3.5152,[354]3.5281,[355]3.5373,[356]3.5434,[357]3.5550,[358]3.5654,[359]3.5686,[360]3.5746,[361]3.5840,[362]3.5923,[363]3.5976,[364]3.6040,[365]3.6092,[366]3.6196,[367]3.6283,[368]3.6348,[369]3.6423,[370]3.6504,[371]3.6639,[372]3.6730,[373]3.6761,[374]3.6794,[375]3.6839,[376]3.6965,[377]3.7076,[378]3.7101,[379]3.7095,[380]3.7065,[381]3.7114,[382]3.7170,[383]3.7201,[384]3.7242,[385]3.7279,[386]3.7334,[387]3.7392,[388]3.7423,[389]3.7315,[390]3.7217,[391]3.7116,[392]3.7059,[393]3.6972,[394]3.6889,[395]3.6801,[396]3.6704,[397]3.6616,[398]3.6514,[399]3.6417,[400]3.6326,[401]3.6221,[402]3.6115,[403]3.6025,[404]3.5914,[405]3.5811,[406]3.5703,[407]3.5606,[408]3.5518,[409]3.5432,[410]3.5377,[411]3.5389,[412]3.5343,[413]3.5378,[414]3.5410,[415]3.5389,[416]3.5393,[417]3.5411,[418]3.5354,[419]3.5369,[420]3.5338,[421]3.5329,[422]3.5342,[423]3.5343,[424]3.5386,[425]3.5382,[426]3.5391,[427]3.5383,[428]3.5413,[429]3.5422,[430]3.5454,[431]3.5466,[432]3.5450,[433]3.5412,[434]3.5414,[435]3.5353,[436]3.5298,[437]3.5255,[438]3.5239,[439]3.5220,[440]3.5266,[441]3.5320,[442]3.5397,[443]3.5371,[444]3.5375,[445]3.5384,[446]3.5429,[447]3.5460,[448]3.5482,[449]3.5507,[450]3.5543,[451]3.5577,[452]3.5598,[453]3.5612,[454]3.5596,[455]3.5619,[456]3.5619,[457]3.5641,[458]3.5690,[459]3.5692,[460]3.5689,[461]3.5654,[462]3.5689,[463]3.5763,[464]3.5818,[465]3.5753,[466]3.5740,[467]3.5729,[468]3.5752,[469]3.5726,[470]3.5698,[471]3.5703,[472]3.5711,[473]3.5702,[474]3.5688,[475]3.5697,[476]3.5685,[477]3.5675,[478]3.5682,[479]3.5701,[480]3.5727,[481]3.5687,[482]3.5723,[483]3.5715,[484]3.5747,[485]3.5809,[486]3.5840,[487]3.5871,[488]3.5925,[489]3.5946,[490]3.5996,[491]3.6058,[492]3.6104,[493]3.6101,[494]3.6106,[495]3.6129,[496]3.6146,[497]3.6176,[498]3.6180,[499]3.6172,[500]3.6211,[501]3.6253,[502]3.6245,[503]3.6228,[504]3.6248,[505]3.6278,[506]3.6359,[507]3.6388,[508]3.6421,[509]3.6343,[510]3.6298,[511]3.6239,[512]3.6201,[513]3.6142,[514]3.6132,[515]3.6159,[516]3.6115,[517]3.6118,[518]3.6110,[519]3.6116,[520]3.6161,[521]3.6149,[522]3.6130,[523]3.6186,[524]3.6170,[525]3.6155,[526]3.6112,[527]3.6059,[528]3.6038,[529]3.6006,[530]3.5978,[531]3.5944,[532]3.5882,[533]3.5818,[534]3.5783,[535]3.5787,[536]3.5817,[537]3.5848,[538]3.5879,[539]3.5906,[540]3.5962,[541]3.5996,[542]3.6024,[543]3.5978,[544]3.5938,[545]3.5936,[546]3.5867,[547]3.5807,[548]3.5740,[549]3.5678,[550]3.5622,[551]3.5568,[552]3.5513,[553]3.5456,[554]3.5451,[555]3.5437,[556]3.5461,[557]3.5497,[558]3.5558,[559]3.5599,[560]3.5653,[561]3.5630,
> llama_print_timings:        load time =   11044.74 ms
> llama_print_timings:      sample time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
> llama_print_timings: prompt eval time = 2990770.13 ms / 287232 tokens (   10.41 ms per token,    96.04 tokens per second)
> llama_print_timings:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
> llama_print_timings:       total time = 2994359.67 ms / 287233 tokens
> 
> Final estimate: PPL = 3.5630 +/- 0.02004
> ```
> 
> </details>
> 
> 👤 **saood06** replied the **2025-03-26** at **02:45:43**:<br>
> > > The code might be a bit spread out, but it is very easy to understand, and I'm sure it will help you find the 2GiB you need to cut.
> > 
> > Ahh okay, I had seen that unsloth fork before, but now having quantized the model enough times here, I can understand what is happening now. And right looks like `q6_k` for `ffn_down.weight` in the first 3 dense layers and `ffn_down_shexp.weight` shared experts is a good place to start trimming a bit.
> 
> Nice, it's good your experience let you understand it, before https://github.com/ikawrakow/ik_llama.cpp/pull/244 existed I would modify the code to generate custom blends and seeing all the recipes (and other bits and pieces I had picked up) was helpful for me, so glad it can be helpful for you.
> 
> 
> > > What have you been running at, and did 0.3 feel appropriate?
> > 
> > I use a small custom python chat client that uses `litellm` to hit the OpenAI API chat endpoint.
> 
> Interesting, I use [mikupad](https://github.com/lmg-anon/mikupad) which is really nice, but from using it a lot, I have a long wishlist of things it doesn't do and might either modify it more than I already have or just make a new thing from scratch architected with all my wants in mind.
> 
> >The first time i forgot and left it at R1 default of `0.6` which possibly had some funky code generation or my terminal got borked. I set it to `0.3` and re-ran while not resizing my terminal and things looks good. The only things I ever specify are `top_p=0.95` and `temperature` as mentioned above. I generally keep it simple for coding generations.
> > 
> 
> I also like to keep it simple in general, temperature and just a little min_p to cull the garbage tokens.
> 
> > In the past I have played with samplers more, especially when trying to reduce slop and increase creativity in writing. I would increase temperature, adjust `top_p`, `min_p`, `top_k`, and even played around a bit with the more specialized samplers like [xtc](https://github.com/ggml-org/llama.cpp/blob/master/examples/main/README.md#xtc-sampling). Anymore I haven't fussed with it much, and spend more time adding variance into the prompt like example clips etc.
> 
> I never played around with samplers much, as I never really liked what increasing temperature did, and too low wasn't nearly as bad but made the model too stiff, and so I would have to put more effort into steering it.
> 
> 👤 **saood06** replied the **2025-03-26** at **04:18:34**:<br>
> > Initial impression is I don't miss `<think>` as it gets right to the point
> 
> Ya it does take time to do, also did you also follow the recommendation of removing them after the round like this:
> 
> ![TbMD7HZZGoeitlEo1p8Ur](https://github.com/user-attachments/assets/5aa8667e-347e-47be-ba4c-863591b07a67)
> 
> Removing the thinking as recommended for multi round causes a lot of prompt reprocessing which takes time on my machine. All the more reason I'm looking forward to DeepSeek-V3-0324
> 
> 👤 **ubergarm** replied the **2025-03-26** at **14:48:48**:<br>
> > Interesting, I use [mikupad](https://github.com/lmg-anon/mikupad) which is really nice, but ...
> 
> Oh nice, a single html sounds cool. I want to re-write my little `dchat.py` app to remove litellm dependency and simply use async http directly as it is such a thin layer and I would prefer to have more transparency. It uses a simple status bar `enlighten` and `deepseek-tokenizer` to dynamically update tok/sec estimate on the client using async streaming response. I'd like to add [primp](https://github.com/deedy5/primp) directly to it, which I use for my "agentic" stuff like web search and scraping - it delivers fairly clean markdown ready to feed to LLMs.
> 
> > also did you also follow the recommendation of removing them after the round 
> 
> Yeah definitely important. I use a naieve `re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)` to rip it out as the client keeps track of the chat thread. Works great unless I'm having it try to refactor itself lol...
> 
> Unrelated, I got my quant downloaded and running locally on the 9950x 96GB RAM + 3090TI 24GB VRAM box with initial test showing almost 2 tok/sec pp and over 4 tok/sec tg (note using `-ser`):
> ```bash
> ./build/bin/llama-server \
>     --model /mnt/ai/models/ubergarm/DeepSeek-R1-GGUF/DeepSeek-V3-0324-IQ2_K_R4-bartowski-imat.gguf \
>     --alias ubergarm/DeepSeek-V3-0324-IQ2_K_R4 \
>     --ctx-size 32768 \
>     -ctk q8_0 \
>     -mla 2 -fa \
>     -amb 512 \
>     -fmoe \
>     -ser 6,1 \
>     --n-gpu-layers 63 \
>     --override-tensor exps=CPU \
>     --parallel 1 \
>     --threads 16 \
>     --host 127.0.0.1 \
>     --port 8080
> ```
> 
> Gotta head out for a night or two, hope to leave a test running and possibly check in via laptop to track updates. Cheers and curious to hear how your iq4 works out!
> 
> 👤 **saood06** replied the **2025-03-27** at **04:00:05**:<br>
> > > Interesting, I use [mikupad](https://github.com/lmg-anon/mikupad) which is really nice, but ...
> > 
> > Oh nice, a single html sounds cool. 
> 
> I actually use the optional server. This way I have access to chat history on all my devices, and the browser is spared storing it (my current DB file is over 8 GB).
> 
> >I want to re-write my little `dchat.py` app to remove litellm dependency and simply use async http directly as it is such a thin layer and I would prefer to have more transparency. 
> 
> That sounds nice. Newer builds of llama.cpp and ik_llama.cpp may differ in some ways, see https://github.com/lmg-anon/mikupad/issues/104 and some of the other issues in the mikupad repo.
> 
> >It uses a simple status bar `enlighten` and `deepseek-tokenizer` to dynamically update tok/sec estimate on the client using async streaming response.
> 
> Mikupad also roughly calculates and displays tok/sec which is nice. 
> 
> You may want to look at how mikupad leverages the llama-server's tokenizer and detokinizer endpoints [here](https://github.com/lmg-anon/mikupad/blob/main/mikupad.html#L1660)
> 
> >I'd like to add [primp](https://github.com/deedy5/primp) directly to it, which I use for my "agentic" stuff like web search and scraping - it delivers fairly clean markdown ready to feed to LLMs.
> 
> Sounds interesting, when you have something do you mind sharing the source in some way?
> 
> > > also did you also follow the recommendation of removing them after the round
> > 
> > Yeah definitely important. I use a naieve `re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)` to rip it out as the client keeps track of the chat thread. Works great unless I'm having it try to refactor itself lol...
> 
> Mikupad has a find and replace that can take a regex so I do about the same, but just manually before sending the next reply as I often do edit the think and response sections of a reply as they are happening.
> 
> > Unrelated, I got my quant downloaded and running locally on the 9950x 96GB RAM + 3090TI 24GB VRAM box with initial test showing almost 2 tok/sec pp and over 4 tok/sec tg (note using `-ser`):
> 
> Nice, PP being slower than TG is odd. Is that because of the ser?
> 
> > Gotta head out for a night or two, hope to leave a test running and possibly check in via laptop to track updates. Cheers and curious to hear how your iq4 works out!
> 
> It finished.
> 
> ```
> llama_model_quantize_internal: model size  = 680237.97 MB
> llama_model_quantize_internal: quant size  = 364082.97 MB
> 
> main: quantize time = 13350534.07 ms
> main:    total time = 13350534.07 ms
> ```
> 
> Thanks, I'll let you know my experience with it.
> 
> 
> Edit: Performance is lower for this mix vs my first (and fastest) R1 mix, I do think it is almost certainly because I did make this mix a bit bigger, but looking into if the runtime computed tensors in #259 may be loaded in a way that is not ideal for my system, I could maybe try loading them into my mmap buffer type from #290.
> 
> First mix of V3_0324:
> (
> llama_model_loader: - type  f32:  361 tensors
> llama_model_loader: - type q8_0:  246 tensors
> llama_model_loader: - type iq4_k_r4:  357 tensors
> llama_model_loader: - type iq5_k_r4:   61 tensors
> llm_load_print_meta: model params     = 671.026 B //this is lower because of MLA tensor exclusion
> llm_load_print_meta: model size       = 355.550 GiB (4.551 BPW)
> llm_load_print_meta: repeating layers = 353.716 GiB (4.541 BPW, 669.173 B parameters)
> )
> 
> |    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
> |-------|--------|--------|----------|----------|----------|----------|
> |   512 |    128 |      0 |   50.431 |    10.15 |   45.268 |     2.83 |
> |   512 |    128 |    512 |   61.857 |     8.28 |   47.996 |     2.67 |
> |   512 |    128 |   1024 |   62.828 |     8.15 |   49.111 |     2.61 |
> |   512 |    128 |   1536 |   64.459 |     7.94 |   50.553 |     2.53 |
> |   512 |    128 |   2048 |   72.170 |     7.09 |   53.913 |     2.37 |
> |   512 |    128 |   2560 |   73.997 |     6.92 |   53.007 |     2.41 |
> 
> R1 fast mix for reference 
> (
> llama_model_loader: - type  f32:  361 tensors
> llama_model_loader: - type q5_0:   61 tensors
> llama_model_loader: - type q5_K:   61 tensors
> llama_model_loader: - type q6_K:    1 tensors
> llama_model_loader: - type iq4_k:    1 tensors
> llama_model_loader: - type iq4_k_r4:  662 tensors
> llm_load_print_meta: model params     = 672.050 B //this is higher because of MLA tensor inclusion
> llm_load_print_meta: model size       = 353.526 GiB (4.519 BPW)
> llm_load_print_meta: repeating layers = 352.333 GiB (4.516 BPW, 670.196 B parameters)
> )
> :
> 
> |    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
> |-------|--------|--------|----------|----------|----------|----------|
> |   512 |    128 |      0 |   49.636 |    10.32 |   39.574 |     3.23 |
> |   512 |    128 |    512 |   57.011 |     8.98 |   43.246 |     2.96 |
> |   512 |    128 |   1024 |   62.986 |     8.13 |   42.916 |     2.98 |
> |   512 |    128 |   1536 |   63.400 |     8.08 |   44.014 |     2.91 |
> |   512 |    128 |   2048 |   66.228 |     7.73 |   47.167 |     2.71 |
> |   512 |    128 |   2560 |   72.508 |     7.06 |   46.553 |     2.75 |
> 
> Edit 2:
> 
> Comparing against another deep context run (where it took 2 hours to load ~26k tokens), it did TG and PP that far out better than my fast quant with a build from early Feb. The optimizations since then such as  PP improvements on PP from MLA-3 mode with   FA, and TG improvements, with FA helping as sweep bench showed the crossing over point at 8K where FA on is better) even though it is at a quant disadvantage.
> 
> I do want to make a fast quant (much closer to pure iq4_k_r4) and see how much better it is.
> 
> Edit 3: Made a pure IQ4_K_R4 mix using the team mradermacher imatrix. It is not functional (but it was fast).
> 
> Overall first impressions though, I do think R1 is better, but the performance benefits of not having thinking tokens, and not having to reprocess the prompt so often due to removing the thinking tokens, means I actually think the new V3 is useful to me. The same can't be said about the old V3 even though it also has those performance benefits.
> 
> 👤 **ubergarm** replied the **2025-03-30** at **04:12:33**:<br>
> > You may want to look at how mikupad leverages the llama-server's tokenizer and detokinizer endpoints 
> 
> Oh that is a nice feature, I didn't realize that endpoint existed! Good to know there may be some differences in the API endpoint as well. I'm happy to share the `dchat.py` after I get it to a place I'm happy enough to release it.
> 
> > Nice, PP being slower than TG is odd. Is that because of the ser?
> 
> I don't think so, but I haven't tested. Basically I'm too impatient to do a proper `llama-bench` on my local rig, but anecdotally I've seen pp go up a bit more to 3-4 tok/sec in short prompts. Been using the faster remote servers mostly haha...
> 
> >  Made a pure IQ4_K_R4 mix
> 
> oh interesting, I was trying to follow the discusson about `--pure`, and found one of the original PRs introducing it on mainline a while back, but I'm honestly not sure that I would want to use it with R1 or V3 given it seems best to make attention higher quant than the experts rather than a single "pure" quant? Maybe I don't understand how it works, or it might apply more to dense models?
> 
> >  iq4_k_r4 and see how much better it is.
> 
> Yeah, that quant has my eye too for a good quality CPU only quant I had in mind... Maybe `iq4_k_r4` for `down_exps` and `iq3_k_r4` for `(gate|up)_exps`... Or what would be the next best size up from `iq4_k_r4`, possibly `IQ5_K_R4` ? Hrmm... Yeah might try that with `q8_0_r8` for all token embedding, attention, dense layers, and shared experts. Maybe can get fairly close to the full `q8_0` perplexity `Final estimate: PPL = 3.2454 +/- 0.01773` with more speed ideally.
> 
> > the new V3 is useful to me
> 
> Yeah, agreed it is nice to just get the answer without all that thinking latency hah.. :crossed_fingers: Fingers crossed that R2 is magically better with the same architecture if they drop that soon hah...
> 
> 👤 **saood06** replied the **2025-03-30** at **05:10:16**:<br>
> >I'm happy to share the `dchat.py` after I get it to a place I'm happy enough to release it.
> 
> Thank you, let me know whenever that is.
> 
> > > Nice, PP being slower than TG is odd. Is that because of the ser?
> > 
> > I don't think so, but I haven't tested. Basically I'm too impatient to do a proper `llama-bench` on my local rig, but anecdotally I've seen pp go up a bit more to 3-4 tok/sec in short prompts. Been using the faster remote servers mostly haha...
> 
> and TG is above 4? I gave ser 7,1 an attempt, I resumed a chat mid system reply and it couldn't finish it only giving gibberish, turned ser off and it worked like usual, maybe ser 7,0.4 might be more stable?
> 
> > > Made a pure IQ4_K_R4 mix
> > 
> > oh interesting, I was trying to follow the discusson about `--pure`, and found one of the original PRs introducing it on mainline a while back, but I'm honestly not sure that I would want to use it with R1 or V3 given it seems best to make attention higher quant than the experts rather than a single "pure" quant?
> 
> I've done many IQ4_K_R4 mixes and my personal favorites for my use cases are the ones closest to pure that have the fastest TG, the PPL benefits from straying away for me don't seem to match the value of IQ4_K_R4, which has really good quality/size and performance characteristics on my machine.
> 
> >Maybe I don't understand how it works, or it might apply more to dense models?
> 
> I don't know, I've stuck with the standard recipes for other models, it's only deepseek where I've experimented a lot with mixes.
> 
> > > iq4_k_r4 and see how much better it is.
> > 
> > Yeah, that quant has my eye too for a good quality CPU only quant I had in mind... Maybe `iq4_k_r4` for `down_exps` and `iq3_k_r4` for `(gate|up)_exps`... Or what would be the next best size up from `iq4_k_r4`, possibly `IQ5_K_R4` ?
> 
> https://github.com/ikawrakow/ik_llama.cpp/pull/149 and https://github.com/ikawrakow/ik_llama.cpp/pull/157 and https://github.com/ikawrakow/ik_llama.cpp/pull/138 have performance metrics for some quants, and https://github.com/ikawrakow/ik_llama.cpp/issues/293 has some info about IQ5_K_R4.
> 
>  >Hrmm... Yeah might try that with `q8_0_r8` for all token embedding, attention, dense layers, and shared experts. Maybe can get fairly close to the full `q8_0` perplexity `Final estimate: PPL = 3.2454 +/- 0.01773` with more speed ideally.
> 
> If my near pure mix that is currently cooking is functional and fast, I wonder if it would have acceptably close PPL for you and also high speed on your CPU system.
> 
> Edit: It is broken, going to try again, also this may be worth looking at for you https://github.com/ikawrakow/ik_llama.cpp/pull/141 
> 
> > > the new V3 is useful to me
> > 
> > Yeah, agreed it is nice to just get the answer without all that thinking latency hah.. 🤞 Fingers crossed that R2 is magically better with the same architecture if they drop that soon hah...
> 
> It is, but if R2 is good enough I know I'll go back to dealing with the latency.
> 
> 👤 **ubergarm** replied the **2025-03-30** at **16:49:03**:<br>
> @saood06 
> 
> > First mix of V3_0324:
> > llama_model_loader: - type f32: 361 tensors
> > llama_model_loader: - type q8_0: 246 tensors
> > llama_model_loader: - type iq4_k_r4: 357 tensors
> > llama_model_loader: - type iq5_k_r4: 61 tensors
> > llm_load_print_meta: model params = 671.026 B //this is lower because of MLA tensor exclusion
> > llm_load_print_meta: model size = 355.550 GiB (4.551 BPW)
> > llm_load_print_meta: repeating layers = 353.716 GiB (4.541 BPW, 669.173 B parameters)
> 
> > some info about IQ5_K_R4.
> 
> Hrmm, I see you used `llama-sweep-bench` on your "first mix", but did you ever check perplexity or try to inference with it? 
> 
> Reason I'm asking is that I made a quant overnight using `iq5_k_r4` and checking perplexity this morning it is very high (not NaN but possibly numerical instability) and also it doesn't inference correctly and just replies with `AlrightAlrightAlrightAlright` hah... 
> 
> I've opened an issue about it to track relevant information easier, feel free to chime in if you have any thoughts. https://github.com/ikawrakow/ik_llama.cpp/issues/296
> 
> > It is broken, going to try again
> 
> Hrm, so your `--pure` mix didn't work? I'm curious how it broke and what you are changing to try again?
> 
> Also I noticed that `python gguf-py/scripts/gguf_dump.py --markdown  /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-CPU-IQ4_K_R4.gguf` doesn't have support for the new quant types so it barfs. I'll keep that in the back of my head for a rainy day to possibly try to update it. More of a convenience than anything else.
> 
> Thanks for sharing all your quant cooking experience and tips!
> 
> 👤 **saood06** replied the **2025-03-30** at **19:34:21**:<br>
> > Hrmm, I see you used `llama-sweep-bench` on your "first mix", but did you ever check perplexity or try to inference with it?
> 
> Assuming you mean the V3_0324, I have not checked perplexity (and I haven't for any other V3_0324 mix), but I do use it for inference as it is my only quant of V3_0324 that functions for inference.
> 
> Also as I've been using V3 more, it feels like a distillation, where it lacks a lot of "breadth" or variety, in a way that I've only seen from distills before. I don't like it, if this continues I may end up back on R1.
> 
> I made all further mixes to try and improve speed (and decided to swap to using a different imatrix file).
> 
> > 
> > Reason I'm asking is that I made a quant overnight using `iq5_k_r4` and checking perplexity this morning it is very high (not NaN but possibly numerical instability) and also it doesn't inference correctly and just replies with `AlrightAlrightAlrightAlright` hah...
> > 
> > I've opened an issue about it to track relevant information easier, feel free to chime in if you have any thoughts. #296
> 
> I will reply over there.
> 
> > 
> > > It is broken, going to try again
> > 
> > Hrm, so your `--pure` mix didn't work? I'm curious how it broke and what you are changing to try again?
> 
> I went into more detail [here](https://github.com/ikawrakow/ik_llama.cpp/pull/295#issuecomment-2762814972) and a few comments following that.
> 
> > Also I noticed that `python gguf-py/scripts/gguf_dump.py --markdown /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-CPU-IQ4_K_R4.gguf` doesn't have support for the new quant types so it barfs.
> 
> Make an issue for it, I've looked into gguf-py before, so I might PR a fix for it when I can.
> 
> >I'll keep that in the back of my head for a rainy day to possibly try to update it. More of a convenience than anything else.
> 
> Or you can make a PR yourself instead of an issue if you want to.
> 
> > Thanks for sharing all your quant cooking experience and tips!
> 
> Thanks for doing the same, I would do more experiments but quanting takes time, and also hogs my server so I can't do inference or other things.

---

👤 **saood06** replied the **2025-03-25** at **15:51:30**:<br>

@ubergarm 

Just saw this "In our web and application environments, the temperature parameter $T_{model}$ is set to 0.3. " and they even go as far to encourage users to use that by "Thus, if you call V3 via API, temperature 1.0 equals to the model temperature 0.3.", so I think you might want to experiment with that temperature.

> 👤 **ubergarm** replied the **2025-03-25** at **16:03:34**:<br>
> Ahh, interesting, yeah R1 suggested default was 0.6 or somthing iirc.
> 
> Does specifying temperature matter for making the imatrix? Guessing it does not, so will continue trying to make imatrix with default command above.
> 
> But when I go to actually test a final quant, thanks for this important detail to set `temp=0.3`!
> 
> 👤 **saood06** replied the **2025-03-25** at **16:54:05**:<br>
> > But when I go to actually test a final quant, thanks for this important detail to set `temp=0.3`!
> 
> Ya I'm in the middle of downloading. This model seems interesting to try out.
> 
> 👤 **saood06** replied the **2025-03-25** at **20:34:02**:<br>
> On this topic what are your preferred samplers? I use just temp, and min_p but this https://github.com/ggml-org/llama.cpp/pull/11223 has caught my eye a bit (seems like it might be a slight improvement over min_p)

---

👤 **saood06** replied the **2025-03-25** at **19:07:02**:<br>

> 14B of the Multi-Token Prediction (MTP) Module weights

@ikawrakow 

Is this something you have looked into? I think even a basic implementation should offer 50% improvement.

There is also jukofyork who is making draft model's (see [here](https://huggingface.co/jukofyork/DeepSeek-R1-DRAFT-0.5B-GGUF)) that can be used with llama.cpp's already existing generic drafting implementation, I'm watching that to see how much performance uplift people end up reporting on that.

> 👤 **ikawrakow** replied the **2025-03-26** at **05:05:55**:<br>
> > > 14B of the Multi-Token Prediction (MTP) Module weights
> > 
> > @ikawrakow
> > 
> > Is this something you have looked into? I think even a basic implementation should offer 50% improvement.
> > 
> > There is also jukofyork who is making draft model's (see [here](https://huggingface.co/jukofyork/DeepSeek-R1-DRAFT-0.5B-GGUF)) that can be used with llama.cpp's already existing generic drafting implementation, I'm watching that to see how much performance uplift people end up reporting on that.
> 
> No, I haven't looked into how it works. I'm surprised MPT has not been implemented in mainline.
> 
> 👤 **jukofyork** replied the **2025-03-31** at **22:05:13**:<br>
> > There is also jukofyork who is making draft model's (see [here](https://huggingface.co/jukofyork/DeepSeek-R1-DRAFT-0.5B-GGUF)) that can be used with llama.cpp's already existing generic drafting implementation, I'm watching that to see how much performance uplift people end up reporting on that.
> 
> @saood06 I haven't released anything yet as wasn't really happy with the results, but somebody linked me this paper:
> 
> https://arxiv.org/html/2411.11055v1
> 
> and I'm retrying after seeing this:
> 
> ![Screenshot_20250331-190526](https://github.com/user-attachments/assets/a6349545-ec76-4644-be19-22b2c6280a3d)
> 
> With 30% raw code data in the mix now.
> 
> 👤 **saood06** replied the **2025-04-01** at **00:10:00**:<br>
> @jukofyork 
> 
> Thanks for the update.

---

👤 **ikawrakow** replied the **2025-03-26** at **05:03:12**:<br>

> [210]6447980.5077,[211]6475482.7036,[212]6484583.7694,[213]6476309.6415,

The imatrix computation that gave these final perplexity values is useless. It means mainline is not working with `Q8_0` either for DeepSeek-V3 (the difference between a NaN PPL and a PPL of 6 million is marginal, if any).

> 👤 **saood06** replied the **2025-03-26** at **05:08:32**:<br>
> > It means mainline is not working with `Q8_0` either for DeepSeek-V3 (the difference between a NaN PPL and a PPL of 6 million is marginal, if any).
> 
> That's the MLA PR on llama.cpp that is not working, llama.cpp main works as it has been used a lot to do imatrix for the large Deepseek V3/R1 models.
> 
> 👤 **ikawrakow** replied the **2025-03-26** at **06:01:14**:<br>
> It looked like this is @ubergarm's imatrix run? It ran to completion with 213 chunks.
> 
> 👤 **saood06** replied the **2025-03-26** at **06:19:31**:<br>
> > It looked like this is @ubergarm's imatrix run? It ran to completion with 213 chunks.
> 
> Yes and that run was on the dairy dreaming PR see below:
> 
> > So I managed to build that [fairydreaming/deepseek2-mla-exp@76543311](https://github.com/fairydreaming/llama.cpp/tree/deepseek2-mla-exp) and have `llama-perplexity` running on the plain `q8_0` I made with `ik_llama.cpp`.
> 
> 👤 **ubergarm** replied the **2025-03-26** at **21:23:34**:<br>
> Okay, using PR#291 I was able to compute an importance matrix on a `V3-0324` static `q8_0` quant. I made the `bf16` GGUF using [evshiron/llama.cpp](https://github.com/evshiron/llama.cpp) as outlined in my notes from the original deepseek-ai `fp8`.
> 
> I'm not clear if this computes imatrix for the MLA tensors as well? If so, then would this be better to use than the bartowski imatrix computed on mainline?
> 
> Anyway, @saood06 if you are interested, I haven't had time to test it yet, but just uploaded it to [ubergarm/DeepSeek-V3-0324-GGUF](https://huggingface.co/ubergarm/DeepSeek-V3-0324-GGUF) hf repo. I hope to eventually upload a quant or two that I like for this fork to that repo.
> 
> Perplexty value and partial logs from computing imatrix on [PR#291 here](https://github.com/ikawrakow/ik_llama.cpp/pull/291#issuecomment-2755540202)
> 
> Cheers!
> 
> 👤 **saood06** replied the **2025-03-27** at **03:32:08**:<br>
> > Anyway, @saood06 if you are interested, I haven't had time to test it yet, but just uploaded it to [ubergarm/DeepSeek-V3-0324-GGUF](https://huggingface.co/ubergarm/DeepSeek-V3-0324-GGUF) hf repo. I hope to eventually upload a quant or two that I like for this fork to that repo.
> 
> Thanks, I would have used your imatrix over bartowski as I think your dataset is better, but I just finished up the quant and don't feel like making another. Once team mradermacher uploads one I may end up making additional quants using both theirs and yours.
> 
> Also the forum link on your huggingface readme from L1T caught my eye, I used to hang around there a good amount, haven't in a while, I should go back.
> 
> 👤 **ubergarm** replied the **2025-03-29** at **18:43:50**:<br>
> > Thanks, I would have used your imatrix over bartowski as I think your dataset is better, but I just finished up the quant and don't feel like making another. Once team mradermacher uploads one I may end up making additional quants using both theirs and yours.
> 
> So I did manage to do a comparison against both imatrix datasets by making two otherwise identical quants and comparing perplexity against `wiki.text.raw`: [here](https://gist.github.com/tristandruyen/9e207a95c7d75ddf37525d353e00659c?permalink_comment_id=5519433#gistcomment-5519433)
> 
> They are pretty close, and bartowski's started off better in the beginning, but the final value the new one I used was slightly better which was interesting.
> 
> Also, I finished and uploaded my `V3-0324` quant and did a comparison across top quant cookers recipes over in [this discussion](https://github.com/ikawrakow/ik_llama.cpp/discussions/288#discussioncomment-12663525)
> 
> The other tip I saw was by [unsloth in r/LocalLLama post](https://www.reddit.com/r/LocalLLaMA/comments/1jk0qjs/178bit_deepseekv30324_230gb_unsloth_dynamic_gguf/) suggesting turn down temp to 0 and min-p to 0.01 when generating code or math. I've seen folks anecdotally suggesting `V3-0324` hallucinates more but might just be the default temps are too high, not sure.
> 
> 👤 **saood06** replied the **2025-03-30** at **01:22:27**:<br>
> > So I did manage to do a comparison against both imatrix datasets by making two otherwise identical quants and comparing perplexity against `wiki.text.raw`: [here](https://gist.github.com/tristandruyen/9e207a95c7d75ddf37525d353e00659c?permalink_comment_id=5519433#gistcomment-5519433)
> 
> Nice, thanks for the additional data point on imatrix dataset quality.
> 
> >Also, I finished and uploaded my V3-0324 quant and did a comparison across top quant cookers recipes over in https://github.com/ikawrakow/ik_llama.cpp/discussions/288#discussioncomment-12663525
> 
> I'm working on making my 3rd quant of V3-0324 (a lot more info on my V3-0324 quants [here](https://github.com/ikawrakow/ik_llama.cpp/discussions/286#discussioncomment-12635966)