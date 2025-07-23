### 📝 [#353](https://github.com/ikawrakow/ik_llama.cpp/issues/353) - Binaries releases for Windows ?

| **Author** | `lbarasc` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2025-04-28 |
| **Updated** | 2025-06-06 |

---

#### Description

Hi,

Can you release binaries for windows working on different types of CPU (avx,avx2 etc...) ?

Thank you.

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-04-29** at **13:55:36**:<br>

If this repository gains more momentum and there are users testing on Windows and providing feedback, sure, we can consider releasing Windows binaries. 

But in the meantime
* I don't have access to a Windows machine
* This is just a hobby project that does not have the funds to go out and rent something in the cloud
* I don't feel OK releasing builds that were never tested

Another thing is that this project does not aim at providing the broad hardware support that mainline `llama.cpp` offers. The optimizations here are targeted towards newer CPUs and GPUs. For instance, a CPU old enough to not support `AVX2` will not benefit at all from this project compared to mainline `llama.cpp`.

---

👤 **PmNz8** commented the **2025-04-30** at **22:54:13**:<br>

I managed to compile from source for Windows cpu, but not for cuda - it is above my skills level. Having (best automatically) compiled binaries available on github would be great! I can always test some binaries if that would be helpful, one of my machine runs intel with avx512 (rocket lake), the other is AMD zen 3 + Nvidia ada.

---

👤 **saood06** commented the **2025-05-01** at **07:32:23**:<br>

>     * I don't have access to a Windows machine
>     * I don't feel OK releasing builds that were never tested

If you want to do occasional releases (since we don't have CI like mainline does that generates over a dozen Windows builds), I can provide the Windows builds made with MVSC 2019 and CUDA v12.1 with AVX2 that have been tested and also Android builds. I could try cross compiling with AVX512 but they wouldn't be tested. ( I know [this](https://www.intel.com/content/www/us/en/developer/articles/tool/software-development-emulator.html) exists but I've never used it and so don't know how much of a slowdown it would have).

---

👤 **SpookyT00th** commented the **2025-05-01** at **22:11:05**:<br>

I noticed you mentioned that this is intended to support newer GPUs. Do you know if the Nvidia V100 (Volta Architecture) is supported?  also, does this support tensor parallelism? i want to fit this model across 128GB VRAM : https://huggingface.co/ubergarm/Qwen3-235B-A22B-GGUF

---

👤 **SpookyT00th** commented the **2025-05-01** at **22:11:05**:<br>

I noticed you mentioned that this is intended to support newer GPUs. Do you know if the Nvidia V100 (Volta Architecture) is supported?

---

👤 **saood06** commented the **2025-05-02** at **03:05:53**:<br>

>also, does this support tensor parallelism? i want to fit this model across 128GB VRAM : https://huggingface.co/ubergarm/Qwen3-235B-A22B-GGUF

For MoE models such as the one you linked, `-split-mode row` does not function, see https://github.com/ikawrakow/ik_llama.cpp/issues/254

---

👤 **sousekd** commented the **2025-05-29** at **20:39:13**:<br>

I would be happy to test on AMD Epyc Turin + RTX 4090 / RTX Pro 6000, if builds are provided.

---

👤 **Thireus** commented the **2025-06-03** at **17:54:35**:<br>

If anyone wants to give a go to the build I've created, and report back if it works decently... https://github.com/Thireus/ik_llama.cpp/releases

Using CUDA 12.8 (and Blackwell compatible) + `-DGGML_AVX512=ON -DGGML_SCHED_MAX_COPIES=1 -DGGML_CUDA_IQK_FORCE_BF16=1`
See https://github.com/Thireus/ik_llama.cpp/blob/main/.github/workflows/release.yml#L448-L450

---

👤 **lbarasc** commented the **2025-06-03** at **19:25:40**:<br>

Well thank you !! i will test this on my server.

---

👤 **ikawrakow** commented the **2025-06-05** at **07:05:32**:<br>

How is the testing going here?

@Thireus 

On `x86_64` the CPU implementation has basically two implementation paths:
* Vanilla `AVX2`, so `/arch:AVX2` for MSVC.
* "Fancy AVX512", which requires `/arch:AVX512`, plus `__AVX512VNNI__`, `__AVX512VL__`, `__AVX512BW__` and `__AVX512DQ__` being defined (if they are not defined, the implementation will use vanilla `AVX2`). These are supported on Zen4/Zen5 CPUs, and I guess some recent Intel CPUs. On Linux they will get defined with `-march=native` if the CPU supports them, not sure how this works under Windows.

There is also GEMM/GEMV implementation for CPUs natively supporting `bf16` (e.g., Zen4/Zen5 and some recent Intel CPUs). To be turned on it requires `__AVX512BF16__` to be defined.

So, to cover pre-build binaries for Windows users, one would need 6 different builds: vanilla `AVX2`, fancy `AVX512` without `bf16`, fancy `AVX512` with `bf16`, with or without CUDA (without CUDA for the users who don't have a supported GPU and don't want to get involved with installing CUDA toolkits and such so the app can run).

---

👤 **PmNz8** commented the **2025-06-06** at **19:01:35**:<br>

@Thireus for me your binaries do not run. I try something simple like .\llama-cli.exe -m "D:\LLMs\bartowski\Qwen_Qwen3-4B-GGUF\Qwen_Qwen3-4B-Q8_0.gguf" and all I get in the log is: 

```
[1749236397] Log start
[1749236397] Cmd: C:\Users\dawidgaming\Downloads\ik_llama-main-b3770-5a8bb97-bin-win-cuda-12.8-x64\llama-cli.exe -m D:\LLMs\bartowski\Qwen_Qwen3-4B-GGUF\Qwen_Qwen3-4B-Q8_0.gguf
[1749236397] main: build = 1 (5a8bb97)
[1749236397] main: built with MSVC 19.29.30159.0 for 
[1749236397] main: seed  = 1749236397
[1749236397] main: llama backend init
[1749236397] main: load the model and apply lora adapter, if any
```
Then it just shuts down.

Windows 11 + RTX 4090 @ 576.52 drivers.

---

👤 **PmNz8** commented the **2025-06-06** at **19:01:35**:<br>

@Thireus for me your binaries do not run. I try something simple like .\llama-cli.exe -m "D:\LLMs\bartowski\Qwen_Qwen3-4B-GGUF\Qwen_Qwen3-4B-Q8_0.gguf" and all I get in the log is: 

```
[1749236397] Log start
[1749236397] Cmd: C:\Users\dawidgaming\Downloads\ik_llama-main-b3770-5a8bb97-bin-win-cuda-12.8-x64\llama-cli.exe -m D:\LLMs\bartowski\Qwen_Qwen3-4B-GGUF\Qwen_Qwen3-4B-Q8_0.gguf
[1749236397] main: build = 1 (5a8bb97)
[1749236397] main: built with MSVC 19.29.30159.0 for 
[1749236397] main: seed  = 1749236397
[1749236397] main: llama backend init
[1749236397] main: load the model and apply lora adapter, if any
```

---

👤 **kiron111** commented the **2025-06-06** at **19:55:45**:<br>

> If anyone wants to give a go to the build I've created, and report back if it works decently... https://github.com/Thireus/ik_llama.cpp/releases
> 
> Using CUDA 12.8 (and Blackwell compatible) + `-DGGML_AVX512=ON -DGGML_SCHED_MAX_COPIES=1 -DGGML_CUDA_IQK_FORCE_BF16=1` See https://github.com/Thireus/ik_llama.cpp/blob/main/.github/workflows/release.yml#L448-L450

Thanks
it's great, I 've just stuck in compiling cuda version....failed for hours