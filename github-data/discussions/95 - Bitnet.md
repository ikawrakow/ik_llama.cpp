### 🗣️ [#95](https://github.com/ikawrakow/ik_llama.cpp/discussions/95) - Bitnet

| **Author** | `ikawrakow` |
| :--- | :--- |
| **Created** | 2024-10-19 |
| **Updated** | 2025-04-22 |

---

#### Description

A Microsoft team has released [CPU inference code](https://github.com/microsoft/BitNet) for 1.58-bit Bitnets. The repo, based 100% on `llama.cpp`, and only adding Bitnet CPU kernels (`ARM_NEON, AVX2`)  has 2.1k stars as of this writing. As per @Dampfinchen ["this is just insanity"](https://github.com/ggerganov/llama.cpp/discussions/9945).

Well, here we have had Bitnet inference for while. For CPU and GPU. Faster than Microsoft's by quite some margin.

There is a screen recording in their repo demoing the 3.3B Bitnet model writing a 900 token essay and achieving 71 t/s on **M2 Ultra**.   Here is a screen recording from my **M2-Max laptop** (~1/2 the computing power and memory bandwidth of M2 Ultra) getting 74 t/s on the same prompt.

https://github.com/user-attachments/assets/889090a2-4c09-4392-99d6-31a76cf54dc1

And here it is running on the M2-Max 30-core GPU

https://github.com/user-attachments/assets/4c08fa07-177a-4462-b4d8-9ce512733fb3

Finally, here running on RTX-4080


https://github.com/user-attachments/assets/e240fd80-9747-470f-8282-3f53bfacff4b

The prompt is very short (9 tokens), but it is still worth noting that Microsoft's implementation processes the prompt at a rate of 85 t/s, while here we get 157 t/s with half the computing power.

---

#### 🗣️ Discussion

👤 **ikawrakow** replied the **2024-10-19** at **08:44:58**:<br>

I was curious to see Microsoft's Bitnet performance on `X86_64`. So, cloned their repo and followed the setup instructions. The setup script  downloaded the `fp32` Bitnet-1.58-3B version, so 13.2 GB instead of 6.6. It also demands `clang-18`, so I had to install that first (even though `llama.cpp` definitely does not require `clang`, and even less `clang-18` to be built, and at a quick glance neither do the added ternary kernels). Their "end-to-end" test script `e2e_benchmark.py` does not do much more than just run the familiar `llama-bench`. Here is what I get on my Ryzen-7950X CPU

| model                          |       size |     params | backend    | threads |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
| bitnet 3B I2_S - 2 bpw ternary | 873.66 MiB |     3.32 B | CPU        |      16 |         pp512 |         28.19 ± 0.12 |
| bitnet 3B I2_S - 2 bpw ternary | 873.66 MiB |     3.32 B | CPU        |      16 |         tg128 |         20.84 ± 0.03 |

The script warns that this is a debug build, but going to the `build` folder and checking shows that, nope, it is a release build. 28 t/s for PP-512 on a 3B ternary model? Hahaha.

Here is what I get with this repo:
| model                          |       size |     params | backend    | threads |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | ---------------: |
| bitnet 3B IQ2_BN - 2.00 bpw Bitnet | 977.42 MiB |     3.43 B | CPU        |      16 |         pp512 |    620.63 ± 3.16 |
| bitnet 3B IQ2_BN - 2.00 bpw Bitnet | 977.42 MiB |     3.43 B | CPU        |       4 |         tg128 |     56.27 ± 0.27 |


22X (!!!) difference in prompt processing speed.  2.8X difference in token generation (TG) speed. TG is memory bound, so let's check what we get with just 1 thread. First theirs (be patient if you try it):

| model                          |       size |     params | backend    | threads |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
| bitnet 3B I2_S - 2 bpw ternary | 873.66 MiB |     3.32 B | CPU        |       1 |         tg128 |          2.01 ± 0.01 |

Then ours

| model                          |       size |     params | backend    | threads |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | ---------------: |
| bitnet 3B IQ2_BN - 2.00 bpw Bitnet | 977.42 MiB |     3.43 B | CPU        |       1 |         tg128 |     25.72 ± 0.11 |

Aha. 12.8X.

Perhaps they did not turn on `AVX2/AVX512` while building? Let's try this
```
python run_inference.py -m models/bitnet_b1_58-3B/ggml-model-i2_s.gguf -p "I believe the meaning of life is" -t 16
...
system_info: n_threads = 16 (n_threads_batch = 16) / 32 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | RISCV_VECT = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | 

sampler seed: 2909124194
sampler params: 
	repeat_last_n = 64, repeat_penalty = 1.000, frequency_penalty = 0.000, presence_penalty = 0.000
	top_k = 40, tfs_z = 1.000, top_p = 0.950, min_p = 0.050, typical_p = 1.000, temp = 0.800
	mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000
sampler chain: logits -> logit-bias -> penalties -> top-k -> tail-free -> typical -> top-p -> min-p -> temp-ext -> softmax -> dist 
generate: n_ctx = 2048, n_batch = 1, n_predict = 128, n_keep = 1

 I believe the meaning of life is . really, ... ... ... ... "..., or. ... what a...... ... ... ... just a by we or close... ar is is it is (... m ... is o to _ more _ _ full _ k _ _ good
 _ _ ( _ R _ ) P P _ and the a, the *’ P R
 B F F ( F F F F B V V
 Com Im Str
 American T



,

 
 ter “ ! M M B P IN IN S P P P O PA PA V ST IN AS B BE PA EHER B BTER B B PA

llama_perf_sampler_print:    sampling time =      15.96 ms /   136 runs   (    0.12 ms per token,  8521.84 tokens per second)
llama_perf_context_print:        load time =     390.49 ms
llama_perf_context_print: prompt eval time =     380.52 ms /     8 tokens (   47.56 ms per token,    21.02 tokens per second)
llama_perf_context_print:        eval time =    6114.10 ms /   127 runs   (   48.14 ms per token,    20.77 tokens per second)
llama_perf_context_print:       total time =    6530.61 ms /   135 tokens
```

Oops. `AVX2` and `AVX512` are both on, and we get gibberish.  

Perhaps `clang` is mis-compiling the code? Or maybe something went wrong with the `clang-18` installation? Let's try `GCC`.
```
mkdir build1 && cd build1
cmake ..
-- The C compiler identification is GNU 11.4.0
-- The CXX compiler identification is GNU 11.4.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Success
-- Found Threads: TRUE  
CMake Error at src/CMakeLists.txt:9 (message):
  Clang is required for Bitnet.cpp compilation


-- Configuring incomplete, errors occurred!
```
Arghh. Comment out the `clang` check in `src/CMakeLists.txt` and retry. Now it builds successfully after
```
cmake ..
make -j
```

Running `llama-cli` gives much better performance - 52 t/s - but still gibberish output. PP-512 is also much better - 300 t/s. That's what I would expect from a run-of-the-mill `AVX2/AVX512` implementation. Still very far from being competitive.

---

👤 **ikawrakow** replied the **2024-10-19** at **15:19:26**:<br>

OK, here is apples-to-apples performance comparison on my M2-Max laptop between Microsoft's `I2_S` and `IQ2_BN` here. I used their `generate-dummy-bitnet-model.py` tool to generate fake Bitnet models of different sizes and ran `llama-bench`. Did not go beyond 30B because generating the 30B model almost exhausted my patience. Their code crashes with segmentation fault on PP-512 tests, so just TG-128.

| Model | t/s (MS I2_S) | t/s (IQ2_BN)   | Speedup |
| ----- | ------------: | -------------: | ------: |
| 125M  |  639.39 ± 10.74 | 947.67 ± 34.86 | 1.482 |
| 350M  |  286.92 ± 1.35  | 426.03 ± 6.64  | 1.485 |
| 1B    |  144.62 ± 3.96  | 225.76 ± 7.70  | 1.561 |
| 1.5B  |  120.12 ± 1.31  | 170.55 ± 8.35  | 1.420 |
| 2.7B  |   84.25 ± 0.43  | 115.52 ± 3.13  | 1.371 |
| 3.8B  |   64.74 ± 0.22  |  86.58 ± 2.83  | 1.337 |
| 7B    |   39.14 ± 0.67  |  51.37 ± 0.82  | 1.312 |
| 13B   |   24.04 ± 0.03  |  30.21 ± 0.18  | 1.257 |   
| 30B   |   11.22 ± 0.05  |  13.57 ± 0.03  | 1.209 |  

The difference in performance decreases with model size, but that's just a matter of memory bandwidth saturation for `IQ2_BN`. The 30B model is 7.45 GiB, so at 13.6 t/s this is 101 GiB/s to fetch the model weights from RAM, which is basically as good as it gets on the M2-Max CPU.

> 👤 **saood06** replied the **2025-04-22** at **08:05:03**:<br>
> Interesting to see the TG number here for 2.7B (115.52 t/s)  is double the performance you got for bitnet2b_2501 (62.33 t/s) which is 2.741 B parameters. Do you know what makes the different architecture twice as slow?
> 
> 👤 **ikawrakow** replied the **2025-04-22** at **08:19:46**:<br>
> This is running on my M2-Max laptop. The M2 has 400 GB/s memory bandwidth. Unfortunately only about 100 GB/s are given to the CPU, the other 300 GB/s are reserved for the GPU (but there are model/quant combinations where I can get up to 110-115 GB/s running CPU-only). As a result the M2-Max has a much better TG performance than a consumer level `x86_64` CPU - nearly twice the TG performance of the Ryzen-7950X. Another interesting thing about the M2-Max is that the silicon spent on the GPU is basically a waste. If it had been spent to double the number of CPU cores, and all of the 400 GB/s had been given to the CPU, that hypothetical CPU would be wiping the floor with the Apple GPU (well, at least for TG, PP would be still 2X lower than the GPU).
> 
> 👤 **saood06** replied the **2025-04-22** at **08:31:01**:<br>
> >This is running on my M2-Max laptop. 
> 
> Sorry, I skipped over that when looking back at this thread.
> 
> 👤 **saood06** replied the **2025-04-22** at **08:42:18**:<br>
> > This is running on my M2-Max laptop. The M2 has 400 GB/s memory bandwidth. Unfortunately only about 100 GB/s are given to the CPU, the other 300 GB/s are reserved for the GPU (but there are model/quant combinations where I can get up to 110-115 GB/s running CPU-only). As a result the M2-Max has a much better TG performance than a consumer level `x86_64` CPU - nearly twice the TG performance of the Ryzen-7950X. Another interesting thing about the M2-Max is that the silicon spent on the GPU is basically a waste. If it had been spent to double the number of CPU cores, and all of the 400 GB/s had been given to the CPU, that hypothetical CPU would be wiping the floor with the Apple GPU (well, at least for TG, PP would be still 2X lower than the GPU).
> 
> Hmm, I know this is for the M1-Max but this https://www.anandtech.com/show/17024/apple-m1-max-performance-review/2 goes over the memory bandwith situation in a lot of depth.
> 
> I'm surprised you tap out at 115 GB/s given what is shown in the linked article.
> 
> The silicon design of the Apple chips has always been interesting to me, I've been following it since the early designs from the iPhone.
> 
> 👤 **ikawrakow** replied the **2025-04-22** at **09:24:20**:<br>
> The article is about the M1 chips? Yes, I have seen benchmarks such as this article. But we are not interested in shoving some data from here to there (which the benchmark does). We are interested in getting some data to the CPU and actually doing something with it.  Here the M2-Max CPU maxes out at 110-115 GB/s, being around 100 GB/s most of the time. For PP I get about 2 TFLOPS out of the M2-Max CPU, so that's 250 GB/s of multiply-add processing power (fused multiply-add counting as 2 ops and needing 4 bytes of data per op), so processing power is not what limit us to ~100 GB/s in TG.
> 
> 👤 **saood06** replied the **2025-04-22** at **09:38:31**:<br>
> >Here the M2-Max CPU maxes out at 110-115 GB/s, being around 100 GB/s most of the time.
> 
> This shows something similar.
> 
> ![a901d026-a1f1-4da4-a410-16c507517571_1256x585](https://github.com/user-attachments/assets/50765a5e-5b5d-4bcf-9aa8-60d4b25bbeff) from https://old.chipsandcheese.com/2023/10/31/a-brief-look-at-apples-m2-pro-igpu/
> 
> This article shows the GPU capping out around 200 GB/s though as the article is more focused on it.
> 
> ![cf2abde5-a4cc-4638-8380-f45cf13c2bc7_1005x497](https://github.com/user-attachments/assets/df0857d8-cbc0-4cc1-9564-9cf4e35eefbb)
> 
> It is a rather impressive chip.
> 
> 👤 **ikawrakow** replied the **2025-04-22** at **10:35:47**:<br>
> Yes, it is. I wish AMD/Intel would finally follow suit, and would give their consumer level chips more memory bandwidth.
> 
> 👤 **saood06** replied the **2025-04-22** at **10:53:44**:<br>
> The cores are also a lot wider, Intel/AMD were stuck on 4-wide for so long, and look at Apple at 9-wide.
> 
> ![image](https://github.com/user-attachments/assets/fa2b157a-365f-4cc7-9ab3-226f65f4c6fb)
> 
> Golden cove from Intel shown below is 6-wide.
> 
> ![3036f76f-f8e9-476b-8bd7-f3be4aadbc88_768x622](https://github.com/user-attachments/assets/8a0583c8-4ced-4669-9ac2-73d777374b6c)

---

👤 **saood06** replied the **2025-04-15** at **14:27:18**:<br>

They updated the repo with the first Official model (all previous models were just supported models, and had far less training) https://huggingface.co/microsoft/bitnet-b1.58-2B-4T it looks competitive at it's size as it was trained with 4T tokens.

> 👤 **ikawrakow** replied the **2025-04-15** at **15:22:22**:<br>
> Good to know. But has something changed since the preliminary models were published (i.e., do I need to make changes to the Bitnet implementation)?
> 
> 👤 **saood06** replied the **2025-04-15** at **15:27:41**:<br>
> I don't think so, they published the i2_s GGUF [here](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/tree/main) which you already did the work supporting converting to a type from this repo in #169.
> 
> 👤 **saood06** replied the **2025-04-20** at **14:24:15**:<br>
> I think I was wrong, [this](https://github.com/microsoft/BitNet/pull/167) adds the new architecture, seems simple enough to port though (might be interesting to test on Android).