### 🐛 [#472](https://github.com/ikawrakow/ik_llama.cpp/issues/472) - Bug: Don't build ggml-aarch64 regardless of CPU arch type

| **Author** | `FullstackSensei` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-29 |
| **Updated** | 2025-05-31 |

---

#### Description

### What happened?

Building ik_llama.cpp always builds ggml-aarch64. This takes almost as much time to build on my system as the rest of ik_llama.cpp's build. and I'm building on 96 cores with 190 threads!!! It's unnecessary when building for x64.

I think it is done because it is hard-coded in [Here](https://github.com/ikawrakow/ik_llama.cpp/blob/1eac9e8487646ee7af00d6d91e10c0cc21ab38c1/ggml/src/CMakeLists.txt#L1376). Seems it came from a merge from llama.cpp last year, but llama.cpp doesn't always build for aarch64.

### Name and Version

llama-cli --version
version: 3717 (1eac9e84)
built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell

```

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-05-30** at **06:11:12**:<br>

Really? Then I guess you need to file a bug report with your compiler vendor. Here is what I see
```
cd ggml/src

time gcc -O3 -march=native -c -I../include -I. ggml-aarch64.c 

real	0m0.164s
user	0m0.130s
sys	0m0.025s
```

The file is ~2.2 LoC, but most of it is between `#ifdef's`, so I'm surprised it takes that long.

Is it possible you think it takes a long time because you see it being compiled at the same time as the `iqk_*.cpp` files, which do indeed take some time to compile.

---

👤 **Ph0rk0z** commented the **2025-05-30** at **11:45:48**:<br>

His compiler isn't broken. I saw this same behavior and though to post about it but just accepted it. The aarch64 is added to cmakelists for everything and some of the quants require symbols from it. I tried to remove it and server wouldn't run due to missing symbols. It is an include in I think ggml.c and those iqk files. I see those already compile and then it sticks on ggml-aarch64. 

It could be a visual bug as you say, but then are those iqk files working on aarch64 specific quant functions? Something is obviously linked to where aarch64 stuff is mandatory for x86.

---

👤 **ikawrakow** commented the **2025-05-30** at **12:27:35**:<br>

> It could be a visual bug as you say, but then are those iqk files working on aarch64 specific quant functions?

It is a "visual bug". When you make a fresh build the first thing that needs to happen is to build the `libggml.so` shared object (DLL on Windows). This involves compiling `ggml.c, ggml-quants.c, ggml-aarch64.c` and the `iqk_*.cpp` files. When building with many threads, all these are compiled in parallel. The next step is linking the generated object files, which cannot proceed until all compilations have finished. Hence, you see `ggml-aarch64.c` being compiled, but it is not `ggml-aarch.c` compilation blocking progress, its compilation is done in a small fraction of a second (0.16 seconds on my CPU).

The file name is of course misleading. `ggml-aarch64.c` does not contain only `__aarch64__` specific code. In this fork it contains `ARM_NEON` implementation for the `Q4_0_4_4` and `Q4_0_8_8` quants, plus scalar implementation for these for other platforms. The file also exists in mainline `llama.cpp` and there it contains SIMD implementations for more quantization types for `ARM_NEON` and, last I checked, `AVX2/AVX512`. I personally find it quite amusing that the `llama.cpp` developers would spend days in a row renaming functions (to make the API more intuitive as they say), and yet will have a source file named [ggml-cpu-aarch64.cpp](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/ggml-cpu-aarch64.cpp) that is not `__aarch64__` specific (it used to be `ggml-aarch64.c` but got renamed to `ggml-cpu-aarch64.cpp` at some point. You can click the link and marvel at the massive amount of `AVX2` code in that file). There is a PR right now in `llama.cpp` attempting to split `ggml-aarch64.c` into multiple platform-specific files.

In principle I could remove this file, but I find it handy for benchmarking my `ARM_NEON` implementation against `Q4_0_X_Y`, which is as fast as it gets on NEON in mainline land. If I wanted to enable `ggml-aarch64` only on `__aarch64__`, it would require a lot of `#ifdef's` all over the place to avoid having the `Q4_0_X_Y` quantization types mentioned. Given the 0.16 seconds compilation time I don't see the point of it.

---

👤 **Ph0rk0z** commented the **2025-05-30** at **14:13:27**:<br>

When I took it out, it did seem to go much faster and those Q4_0_4_4/Q4_0_8_8 functions popped up warnings. I compile for all cache quantizations too with like -j 90. There are points where it just sits on very little CPU usage for quite a while and this is one that comes up. No clue what it's doing during that time.

---

👤 **ikawrakow** commented the **2025-05-30** at **15:09:23**:<br>

https://github.com/user-attachments/assets/da575fd8-ba9e-41c6-bbb9-658672b47b78

---

👤 **FullstackSensei** commented the **2025-05-30** at **20:47:54**:<br>

The underlying issue is that building ik_llama.cpp takes ~2x (or more?) the time it takes to build llama.cpp on the same machine with the same build options. I was trying to help find the underlying issue since it does seem to stall at ggml-aarch64 with very low CPU utilization. I genuinely don't care whether there's an ARM build also tucked in there. The issue is the long build times which make updating ik_llama.cpp or testing branches/forks a lot more painful than it needs to be.

@ikawrakow, obviously you know the codebase. I was trying to help debug the issue since that is where the build stops for quite a while, and help pinpoint where the issue might be. I don't think anyone asked for proof that ggml-aarch64 is not the issue, but we also don't know the codebase nor the build process as well as you do.

I'm no expert in cmake, but if there's anything I can do to help diagnose the issue, I'd be happy to help if you can give some guidance or instructions on what to do.

---

👤 **ikawrakow** commented the **2025-05-31** at **05:18:47**:<br>

> The underlying issue is that building ik_llama.cpp takes ~2x (or more?) the time it takes to build llama.cpp on the same machine with the same build options.

There are 2 main contributing factors to the longer build times:
* The matrix multiplication and flash attention kernels that I have added in `ik_llama.cpp`. These are ~18 kLOC of heavily templated C++ code, so take a while to compile. Prior to PR #435 they used to be in a single file that took 2.5 minutes to compile on my CPU. It shouldn't be so bad after #435, but they still do they a while (~20 seconds on my CPU). No progress can be made in the build process until these have been compiled and linked as they are part of the `ggml` library that everything depends on.
* Compiling `llama.cpp` (a ~23 kLOC C++ source file). This takes ~50 seconds on my CPU. In mainline `llama.cpp` they have refactored their former `llama.cpp` source file into multiple files, which allows this part to be done in parallel. I know I should do something similar here, just haven't come around to do it.

I just measured how long it takes to build `ik_llama.cpp` and `llama.cpp` from scratch with `ccache` disabled and without CUDA (the CUDA code is in a league of its own here and in mainline). Result:
* `ik_llama.cpp`:   84 seconds
* `llama.cpp`: 41 seconds

So, excluding the 50 seconds taken by `llama.cpp` compilation, the remainder in `ik_llama.cpp` is just ~35 seconds.

---

👤 **saood06** commented the **2025-05-31** at **23:08:49**:<br>

> The file name is of course misleading. `ggml-aarch64.c` does not contain only `__aarch64__` specific code. In this fork it contains `ARM_NEON` implementation for the `Q4_0_4_4` and `Q4_0_8_8` quants, plus scalar implementation for these for other platforms. 
> 
> In principle I could remove this file, but [...] If I wanted to enable `ggml-aarch64` only on `__aarch64__`, it would require a lot of `#ifdef's` all over the place to avoid having the `Q4_0_X_Y` quantization types mentioned. Given the 0.16 seconds compilation time I don't see the point of it.

Instead of refactoring or removing it since I agree with the reasons against both, why not just rename the file to something that is less misleading.