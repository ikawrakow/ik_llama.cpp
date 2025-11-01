## 🔀 [Pull Request #435](https://github.com/ikawrakow/ik_llama.cpp/pull/435) - Refactor iqk_mul_mat.cpp

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/refactor_iqk` |
| **Target Branch** | `main` |
| **Created** | 2025-05-20 |
| **Updated** | 2025-05-23 |
| **Merged** | 2025-05-22 |

---

## 📄 Description

I have been putting all matrix multiplication (GEMM) and flash attention (FA) kernels into `iqk_mul_mat.cpp`. With time it became a giant source file (~18 kLOC) containing heavily templated C++ code. The result: extremely long compilations times (over 2 minutes on a high end CPU, with some users reporting 30 minutes on an Android phone).

This PR splits `iqk_mul_mat.cpp` into multiple files:
* `iqk/iqk_gemm_floats.cpp` - contains GEMM kernels operating on float tensors
* `iqk/iqk_gemm_1bit.cpp` - contains GEMM kernels for BitNet and `IQ1_S, IQ1_M` (along with repacked variants)
* `iqk/iqk_gemm_kquants.cpp` - contains GEMM kernels for k-quants and repacked k-quants
* `iqk/iqk_gemm_iquants.cpp` - contains GEMM kernels for i-quants and repacked i-quants
* `iqk/iqk_gemm_iqk_quants.cpp` - GEMM kernels for `IQX_K` and repacked
* `iqk/iqk_gemm_legacy_quants.cpp` - GEMM kenels for legacy quants (`Q4_0`, etc.) and repacked
* `iqk/iqk_mul_mat.cpp` now contains just the GEMM business logic and compiles very fast
* `iqk/fa/iqk_fa_templates.h` - FA templates that get included in the FA `*.cpp` files
* `iqk/fa/iqk_fa_*_*.cpp` - FA template instantiations for specific combinations of K and V attention head sizes

With this, a fresh build with of the `iqk` folder (with files compiled in parallel) takes
* ~17 seconds on a Ryzen-7950X (Zen4)
* ~15 seconds on a Ryzen-5975WX (AVX2)
* ~13 seconds on a M2-Max (ARM_NEON)

The Zen4 build is longer because we have additional kernels for `bf16` not supported natively by the other two platforms.
The GEMM files compile in 5-6 seconds each, so the FA instantiations dominate the build time. One could split them further, but for now I can live with compile times in the range of 15 seconds.

It is a massive change. Testing of all types (50+ when row-interleaved quants are included) on `AVX2, Zen4` and `ARM_NEON` took quite some time. I hope to have covered all possible combinations, but still would appreciate additional testing from people using `ik_llama.cpp` for CPU-only inference.

Closes [#183](https://github.com/ikawrakow/ik_llama.cpp/issues/183)

---

## 💬 Conversation

👤 **saood06** commented on **2025-05-20** at **07:20:58**

>I hope to have covered all possible combinations, but still would appreciate additional testing from people using ik_llama.cpp for CPU-only inference.

Testing the build time: ~7 min compared to ~18 minutes before on my dual socket Xeon E5-2690 v3. It used more threads but still nowhere near saturating my available ones for a large amount of the time. It may have a lower peak memory footprint but I will have to measure that better to tell.

Tested with my standard `cmake .. -DGGML_RPC=ON -DGGML_IQK_FA_ALL_QUANTS=1; cmake --build . --config Release -j 48`

---

👤 **ikawrakow** commented on **2025-05-20** at **07:30:51**

> It used more threads but still nowhere near saturating my available ones for a large amount of the time. 

It cannot saturate your 48 cores. It needs to build `libggml.so` first, and this is what it takes to do that:
```
[  2%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml.c.o
[  3%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml-alloc.c.o
[  4%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml-quants.c.o
[  4%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml-backend.c.o
[  5%] Building CXX object ggml/src/CMakeFiles/ggml.dir/llamafile/sgemm.cpp.o
[  5%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_mul_mat.cpp.o
[  6%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_flash_attn.cpp.o
[  6%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/fa/iqk_fa_576_512.cpp.o
[  7%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/fa/iqk_fa_192_128.cpp.o
[  8%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/fa/iqk_fa_128_128.cpp.o
[  8%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/fa/iqk_fa_256_256.cpp.o
[  8%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/fa/iqk_fa_96_96.cpp.o
[  9%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/fa/iqk_fa_64_64.cpp.o
[  9%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_gemm_floats.cpp.o
[ 10%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_gemm_kquants.cpp.o
[ 10%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_gemm_iquants.cpp.o
[ 12%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_gemm_iqk_quants.cpp.o
[ 12%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_gemm_1bit.cpp.o
[ 12%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_gemm_legacy_quants.cpp.o
[ 13%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_quantize.cpp.o
```
With all quants enabled for FA the above takes 36 seconds on my `AVX2` box.

Compiling `llama.cpp` is another piece that takes quite some time, so it should get refactored as well.

---

👤 **saood06** commented on **2025-05-20** at **07:51:53**

>It cannot saturate your 48 cores. It needs to build libggml.so first, and this is what it takes to do that:

I know and I'm not expecting it to, but it still did have a much higher usage overall. (I use this machine to do a lot of cross-compiling and builds of other software so I understand what the output of cmake means and I was monitoring it alongside btop). 

>Compiling llama.cpp is another piece that takes quite some time, so it should get refactored as well.

That piece is fast enough on my machine iqk_mul_mat.cpp was the majority of the time spent before.

Thank you for this, it is a very welcome speed improvement.

---

👤 **cmoncure** commented on **2025-05-22** at **18:23:28**

This commit results in a significant performance regression for me, established by git bisect.

My TG drops by about 30% on DeepSeek.

b94cd3b632a78dfb46b18d52b84be66bcf26166a is the first bad commit
commit b94cd3b632a78dfb46b18d52b84be66bcf26166a (HEAD)
Author: Kawrakow <iwankawrakow@gmail.com>
Date:   Thu May 22 10:05:51 2025 +0300

    Refactor iqk_mul_mat.cpp ([#435](https://github.com/ikawrakow/ik_llama.cpp/issues/435))

---

👤 **ikawrakow** commented on **2025-05-23** at **05:09:34**

> This commit results in a significant performance regression for me, established by git bisect.

Please file an issue with all the relevant details.