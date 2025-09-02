## 🗣️ [Discussion #632](https://github.com/ikawrakow/ik_llama.cpp/discussions/632) - Row-interleaved (R4) Quant Performance on Dual Xeon CPU-only Inference

| **Author** | `rkozuch` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2025-07-20 |
| **Updated** | 2025-07-24 |

---

## 📄 Description

Comparing @ubergarm R4 quants to Unsloth's UD quants, I get significantly _worse_ performance in ik_llama.cpp with the R4 quants:

| model                                  |  fa | pp512 t/s | tg128 t/s | UD vs R4 |
| -------------------------------------- | --: | --------: | --------: | -------- |
| DeepSeek-R1-0528-UD-Q4_K_XL-00001-of-00008.gguf |   0 |     20.62 |      2.44 | UD +33% faster     |
| DeepSeek-R1-0528-IQ4_KS_R4-00001-of-00009.gguf    |   0 |     17.10 |      1.84 |          |
| DeepSeek-R1-0528-UD-Q4_K_XL-00001-of-00008.gguf |   1 |     21.36 |      1.51 | UD +15% faster    |
| DeepSeek-R1-0528-IQ4_KS_R4-00001-of-00009.gguf    |   1 |     17.53 |      1.31 |          |

These aren't exactly the same quants, but the difference seems significant.
Am I doing something wrong? And am I making the best use of these ik_llama-tuned R4 quants?

(Also, I usually get around 3.5 tg t/s on the UD R1 quant, but the pp and tg varies between reboots and cache clears, for some reason.)

---

# Details
Run command for the above table:
```
numactl --interleave=all --cpunodebind=0,1 ./build_build8b/bin/llama-bench -m /mnt/vm-shared-files/models/unsloth/DeepSeek-R1-0528-UD-Q4_K_XL-00001-of-00008.gguf -m /mnt/vm-shared-files/models/ubergram/DeepSeek-R1-0528-IQ4_KS_R4-00001-of-00009.gguf --numa distribute -p 512 -n 128 -r 10 -fa 0,1 -mla 0 --split-mode none -t 46 -rtr 0 -fmoe 1 -o json
```

(I know that `-mla 3` is [recommended for best performance](https://github.com/ikawrakow/ik_llama.cpp/pull/273), but I get segfaults on the UD quant with anything other than `-mla 0`.)

My hardware is an old Dell PowerEdge R740, with:
- 2 x 12-core Xeon Gold 6226 CPUs @ 2.70GHz (iDRAC showing a current speed of 2700 MT/s)
- 20 x 32GB DDR4 (640GB Total) RAM (rated at 2933 MT/s) 
- No GPU

ik_llama.cpp config:
- build number `3736`, commit `1eabdb42`
- Compiled from source, with the following CMAKE flags:
```
-DCMAKE_BUILD_TYPE=Release -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Intel10_64lp -DGGML_NATIVE=ON '-DCMAKE_C_FLAGS=-O3 -march=cascadelake -mtune=cascadelake -ffast-math -fno-finite-math-only -fopenmp' '-DCMAKE_CXX_FLAGS=-O3 -march=cascadelake -mtune=cascadelake -ffast-math -fno-finite-math-only -fopenmp' -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_AVX_VNNI=ON -DGGML_AVX512=ON -DGGML_AVX512_VNNI=ON -DGGML_OPENMP=ON
```

---

This may or may not be relevant: This is all running on a `Ubuntu 24.04.2 Server` VM hosted on `Proxmox Virtual Environment 8.4.0` on the Dell R740. I've configured the VM to use the host CPU, with 1:1 mappings to cores, with NUMA enabled on both Host and Guest and equal memory bindings across the two NUMA nodes. Even if this was a performance impediment, I'd still expect Ubergarm's R4 to perform better _relatively_ to UD in ik_llama.cpp, right?

---

## 💬 Discussion

👤 **ikawrakow** commented on **2025-07-20** at **12:58:55**

So, things are changing fast in this repository.

Yes, it used to be true that row-interleaved quants offer better PP performance. But then I optimized non-interleaved quants in PRs [#531](https://github.com/ikawrakow/ik_llama.cpp/issues/531), [#533](https://github.com/ikawrakow/ik_llama.cpp/issues/533), [#534](https://github.com/ikawrakow/ik_llama.cpp/issues/534) (AVX2) and [#549](https://github.com/ikawrakow/ik_llama.cpp/issues/549), [#550](https://github.com/ikawrakow/ik_llama.cpp/issues/550), [#552](https://github.com/ikawrakow/ik_llama.cpp/issues/552) (ARM_NEON), so now non-interleaved quants have a better PP performance.

The better TG performance is unexpected. I haven't checked these models closely, but I wouldn't be surprised if the `IQ4_KS_R4` model uses more bits in terms of active parameters than `UD-Q4_K_XL`. If so, that could explain the difference in TG performance.

> but I get segfaults on the UD quant with anything other than -mla 0

Please file an issue with your command. If you could run in the debugger and do a backtrace when it crashes, that would be great!

> 👤 **rkozuch** replied on **2025-07-24** at **12:02:00**
> 
> Changing fast indeed. Was not expecting the whole repo to disappear overnight 😅. Welcome back!
> 
> I misremembered the error and it was not Deepseek R1 but instead V3 causing the segfaults, and only in `llama-bench`, for some reason.
> 
> > If you could run in the debugger and do a backtrace when it crashes, that would be great!
> 
> No worries, but I'm not familiar with the debugger; if you can point me in the right direction on how I can do a backtrace, I can do that over the weekend. Cheers

---

👤 **ubergarm** commented on **2025-07-20** at **14:57:40**

@rkozuch 

Yup as ik says things are moving fast. In lieu of the recent optimizations for non-interleaved quants I have moved away from releasing the `_R4` flavors. The non-interleaved quants tend to see a large boost in PP using larger batch sizes like `-ub 4096 -b 4096` even on MoE architecture models.

Some folks have been re-mixing my recipes using the newer quantizations available and non-interleaved forms with good results as described here: https://github.com/ikawrakow/ik_llama.cpp/pull/616#issuecomment-3087170346

---

👤 **Ph0rk0z** commented on **2025-07-24** at **11:41:19**

I have dual xeon *with* GPUs and in my experience -RTR or static packed quants only helped prompt processing at certain batch sizes. Otherwise it would lower speeds. In my case, TG would improve a bit, so I'm surprised yours did not.