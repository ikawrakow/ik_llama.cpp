## 🔀 [Pull Request #336](https://github.com/ikawrakow/ik_llama.cpp/pull/336) - Fix termux/android build

| **Author** | `saood06` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `s6/termux_fix` |
| **Target Branch** | `main` |
| **Created** | 2025-04-20 |
| **Updated** | 2025-04-30 |
| **Merged** | 2025-04-21 |

---

## 📄 Description

@ikawrakow

Sorry this is a mess, but this does get it to build now on my android device where I was able to replicate the compile error (my device does not support __ARM_FEATURE_DOTPROD so even though it now builds, it does not use the IQK stuff, but I may be able to confirm it works later on a device that that does support dotprod later).

I did catch the additional issue of the changed iqk_flash_attn_noalibi definition in the case where your building this repo and IQK_IMPLEMENT is not defined because my device doesn't support dotprod.

Fixes [#159](https://github.com/ikawrakow/ik_llama.cpp/issues/159)

---

## 💬 Conversation

👤 **ikawrakow** commented on **2025-04-20** at **08:59:26**

Thank you for this.

So, the issue on Android was that no visibility was specified for the iqk functions, Android apparently uses hidden visibility by default, so the linker does not find the iqk functions.

I guess we need an `IQK_API` macro similar to `GGML_API`. Or one can just reuse `GGML_API` as the `iqk` stuff gets built as part of the `ggml` library.

---

👤 **saood06** commented on **2025-04-20** at **09:20:04**

> Thank you for this.

It would be interesting to benchmark it, but I can't since my phone doesn't support IQK. My main motivation was thinking about doing a release (I can build on Windows, Linux, and now Android but I haven't done many non-native builds, and don't have access to a mac).
 
> So, the issue on Android was that no visibility was specified for the iqk functions, Android apparently uses hidden visibility by default, so the linker does not find the iqk functions.

Yes, that and the definition fix for the iqk_flash_attn_noalibi.

> I guess we need an `IQK_API` macro similar to `GGML_API`. 

That should work.

>Or one can just reuse `GGML_API` as the `iqk` stuff gets built as part of the `ggml` library.

"Attempt fix 3" was my last try at that, I couldn't get it to work.

---

👤 **saood06** commented on **2025-04-21** at **03:39:42**

Cleaned it up using an `IQK_API` macro.

---

👤 **ikawrakow** started a conversation on `ggml/src/iqk/iqk_config.h` on **2025-04-21** at **06:11:32**

To have this also work for a static built, it should be
```c++
#ifdef GGML_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef GGML_BUILD
#            define IQK_API __declspec(dllexport)
#        else
#            define IQK_API __declspec(dllimport)
#        endif
#    else
#        define IQK_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define IQK_API
#endif
```

> 👤 **saood06** replied on **2025-04-21** at **07:11:44**
> 
> Changed.

---

👤 **ikawrakow** started a conversation on `ggml/src/iqk/iqk_flash_attn.cpp` on **2025-04-21** at **06:15:05**

Do we really need to repeat `extern "C" IQK_API` here?

> 👤 **saood06** replied on **2025-04-21** at **07:12:00**
> 
> Changed

---

👤 **ikawrakow** approved this pull request ✅ on **2025-04-21** at **06:27:52**

I wonder if something else apart from the dot product is needed to have the iqk functions work on your phone. I see that I have consistently used  `ggml_vdotq_s32`, whiere `ggml` provided an implementation when `__ARM_FEATURE_DOTPROD` is not available.  The one known missing ingredient without `__ARM_FEATURE_DOTPROD ` is `vdotq_laneq_s32`. But is there something else missing? If  `vdotq_laneq_s32` was the only missing thing, one could add an implementation, and then one would be able to use `iqk` stuff on generic `__aarch64__`. I don't have an Android phone myself, so was never compelled to try.

---

👤 **saood06** commented on **2025-04-21** at **07:13:59**

>I don't have an Android phone myself, so was never compelled to try.

I do have an android device, but I don't plan on using ik_llama on it, the limited RAM and slow CPU/GPU make it not worthwhile for me.

I made the two suggested changes, and it compiles.

---

👤 **ikawrakow** commented on **2025-04-21** at **07:19:58**

So now we need to find someone with a modern phone willing to test. I would be really curious to compare the performance to Vulkan. The GPUs on many of the phones are quite underpowered, and the `llama.cpp` Vulkan implementation is not particularly performant (although it seems to have been improving lately), so now that it builds on Android, running `ik_llama.cpp` on the CPU is possibly a viable alternative to Vulkan.

---

👤 **saood06** commented on **2025-04-21** at **07:38:30**

> So now we need to find someone with a modern phone willing to test.

I should be able to get temporary access to a modern phone. I want to test the new Bitnet model (that needs to be ported) as that does seem like a really good fit for mobile use, and also a really good showcase of ik_llama.cpp.

>I would be really curious to compare the performance to Vulkan. The GPUs on many of the phones are quite underpowered, and the `llama.cpp` Vulkan implementation is not particularly performant (although it seems to have been improving lately), so now that it builds on Android, running `ik_llama.cpp` on the CPU is possibly a viable alternative to Vulkan.

Yes, Vulkan and [this OpenCL backend](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENCL.md#todo),  which was introduced after this repo forked (this repo is actually in an awkward middle where it has neither the old or the new OpenCL).

Do you have a model/quant in mind you would want ran across the 3 backends?

---

👤 **ikawrakow** commented on **2025-04-21** at **08:45:24**

> Do you have a model/quant in mind you would want ran across the 3 backends?

Including Android? Then something small like LLaMA-3B using `IQ4_XS` or `IQ4_KS`. Bitnet would be good too.

---

👤 **saood06** commented on **2025-04-24** at **13:47:00**

I had a little bit of time with a Galaxy S22 (1×3.00 GHz Cortex-X2 & 3×2.40 GHz Cortex-A710 & 4×1.70 GHz Cortex-A510).

`~/ik_llama.cpp/build1 $ bin/llama-sweep-bench -m ~/ggml-model-iq2_bn_r4.gguf  -t 4`

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    6.081 |    84.20 |    3.537 |    36.18 |
|   512 |    128 |    512 |    8.509 |    60.17 |    4.594 |    27.86 |
|   512 |    128 |   1024 |   12.571 |    40.73 |    5.461 |    23.44 |
|   512 |    128 |   1536 |   16.879 |    30.33 |    6.582 |    19.45 |
|   512 |    128 |   2048 |   20.344 |    25.17 |    7.640 |    16.75 |
|   512 |    128 |   2560 |   29.417 |    17.40 |   10.138 |    12.63 |
|   512 |    128 |   3072 |   34.477 |    14.85 |   11.348 |    11.28 |
|   512 |    128 |   3584 |   38.911 |    13.16 |   12.595 |    10.16 |

Flash attention did worse:
`~/ik_llama.cpp/build1 $ bin/llama-sweep-bench -m ~/ggml-model-iq2_bn_r4.gguf -fa -t 4`

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    9.496 |    53.92 |    3.954 |    32.38 |
|   512 |    128 |    512 |   19.082 |    26.83 |    7.029 |    18.21 |
|   512 |    128 |   1024 |   27.123 |    18.88 |   10.393 |    12.32 |
|   512 |    128 |   1536 |   32.178 |    15.91 |   14.209 |     9.01 |
|   512 |    128 |   2048 |   40.818 |    12.54 |   16.617 |     7.70 |
|   512 |    128 |   2560 |   48.743 |    10.50 |   20.061 |     6.38 |
|   512 |    128 |   3072 |   55.976 |     9.15 |   25.354 |     5.05 |
|   512 |    128 |   3584 |   76.750 |     6.67 |   27.247 |     4.70 |

I'll be able to test more with it again later.

---

👤 **saood06** commented on **2025-04-30** at **07:37:58**

I was able to test a bit more and turns out the results I got above are meaningless as the model returns gibberish. I have to build with arch flags manually set (and armv9 caused illegal instructions even though this device supports it, but `armv8.2-a+dotprod+fp16` worked). The new build was tested working with the test prompt in cli returning coherent results (and the much longer compile time showed it was actually compiling iqk_mul_mat.cpp), but performance numbers were wildly inconsistent between runs (even using taskset to try and force it to only be on the performant cores helped a bit but still was very inconsistent).

Best result I was able to get was with 4 threads and FA off but I haven't managed to get another result close (even with those same settings for FA and thread number)

`bin/llama-sweep-bench -m ~/ggml-model-iq2_bn_r4.gguf -t 4`

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |   10.261 |    49.90 |    5.130 |    24.95 |
|   512 |    128 |    512 |   11.840 |    43.24 |    6.445 |    19.86 |
|   512 |    128 |   1024 |   16.336 |    31.34 |    6.925 |    18.48 |
|   512 |    128 |   1536 |   13.914 |    36.80 |    7.685 |    16.66 |
|   512 |    128 |   2048 |   14.825 |    34.54 |    8.168 |    15.67 |
|   512 |    128 |   2560 |   17.940 |    28.54 |    8.694 |    14.72 |
|   512 |    128 |   3072 |   19.040 |    26.89 |    8.911 |    14.36 |
|   512 |    128 |   3584 |   20.549 |    24.92 |    9.319 |    13.74 |

---

👤 **ikawrakow** commented on **2025-04-30** at **08:28:12**

Do you know how `BitNet.cpp` does on this device?

---

👤 **saood06** commented on **2025-04-30** at **08:47:23**

> Do you know how `BitNet.cpp` does on this device?

I don't and I really want to but I until I find a way to get more consistent performance numbers on the device, I'm not sure any meaningful comparisons could be made. The issue does seem like a mix of the system scheduler, thermal throttling, and core assignment (and there might even be more issues). Using taskset does seem to help the core assignment issue, but results still fluctuate an incredible amount. 

I wanted to provide the flash attention numbers as well, but I'm not sure if I just can't get a good run, or if flash attention is worse on this device.

---

👤 **ikawrakow** commented on **2025-04-30** at **09:06:45**

So, my Arm optimizations are totally based on the M2 chip. Your results and what was reported in [#345](https://github.com/ikawrakow/ik_llama.cpp/issues/345) may indicate that they may not really be optimal for lower end Arm processors. For instance, I often use more vector registers than available. On the M2-Max this register spillage is better (faster) than not using all vector registers. But the lower end chips may not handle this very well (common wisdom is that one should avoid register spillage). Or perhaps the compiler is not producing optimum code. Have you tried `clang` (which is what I use for the M2)?

I guess, if I want to become serious with supporting mobile devices, I should get myself a Raspberry Pi to play with. Or perhaps the Rock 5b board.

I haven't done any experiments on that sort of CPU for a long time. But I think around 2016 or so I did experiment with a bunch of heavy duty number crunching algorithms on my Android phone at the time (don't remember what the CPU was). It was actually quite impressive, being only about 3 times slower than my desktop PC at the time. But only for a short period of time. After a minute or two, performance would totally disintegrate, and would not come back without a reboot even after long periods of letting the phone sit idle. This is now almost 10 years ago and mobile phone CPUs have improved a lot since then, but I'm not surprised you are observing issues with performance sustaining over longer periods.

---

👤 **saood06** commented on **2025-04-30** at **09:31:06**

>For instance, I often use more vector registers than available. On the M2-Max this register spillage is better (faster) than not using all vector registers. But the lower end chips may not handle this very well (common wisdom is that one should avoid register spillage).

Interesting.

> Or perhaps the compiler is not producing optimum code. Have you tried `clang` (which is what I use for the M2)?

I have only tried clang on this device (and I'm still not sure why the `armv9-a` build gives illegal instruction even though my CPU supports that instruction set).

> I guess, if I want to become serious with supporting mobile devices, I should get myself a Raspberry Pi to play with. Or perhaps the Rock 5b board.

The Raspberry Pi 5 has a 4×2.40GHz Cortex-A76, which is far worse than the (1×3.00 GHz Cortex-X2 & 3×2.40 GHz Cortex-A710 + ...) of the phone I am using. The Apple cores though are definitely nicer (but they take up a lot more die area).

> I haven't done any experiments on that sort of CPU for a long time. But I think around 2016 or so I did experiment with a bunch of heavy duty number crunching algorithms on my Android phone at the time (don't remember what the CPU was). It was actually quite impressive, being only about 3 times slower than my desktop PC at the time.

It really is impressive how much compute mobile devices have.

>But only for a short period of time. After a minute or two, performance would totally disintegrate, and would not come back without a reboot even after long periods of letting the phone sit idle. This is now almost 10 years ago and mobile phone CPUs have improved a lot since then, but I'm not surprised you are observing issues with performance sustaining over longer periods.

If it was just throttling that would make it easy, but the fast run I posted wasn't even the first full run, and the phone was already noticeably warm by that point. The SoC in that phone is notorious for throttling though, so that probably played a part.