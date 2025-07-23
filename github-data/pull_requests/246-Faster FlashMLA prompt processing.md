### 🔀 [#246](https://github.com/ikawrakow/ik_llama.cpp/pull/246) - Faster FlashMLA prompt processing

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-03-08 |
| **Updated** | 2025-03-08 |

---

#### Description

MLA as used in the DeepSeek models is great for token generation (TG), but prompt processing (PP) speed is much lower compared to standard attention even with FA enabled.

This PR improves FlashMLA speed by a large margin. FlashMLA is CPU only, but the PR paves the way to perhaps also get it on CUDA (but this is left for a future PR).

The following table compares FlashMLA PP speed for DeepSeek-Lite quantized as `IQ4_NL` between the main branch and this PR. CPU is Ryzen-7950X, the cache is quantized with `Q8_0`, `fmoe` is on.

| model                |          test |     t/s (main)   |   t/s (PR)       |  Speedup |
| ---------------------| ------------: | ---------------: | ---------------: | -------: |
| deepseek2 16B IQ4_NL |         pp512 |    605.29 ± 4.92 |    681.72 ± 1.12 |  1.126   |
| deepseek2 16B IQ4_NL |        pp1024 |    568.79 ± 0.75 |    648.71 ± 1.48 |  1.141   |
| deepseek2 16B IQ4_NL |        pp2048 |    509.15 ± 4.38 |    598.99 ± 0.83 |  1.176   |
| deepseek2 16B IQ4_NL |        pp4096 |    420.10 ± 0.82 |    514.62 ± 2.68 |  1.225   |
| deepseek2 16B IQ4_NL |        pp8192 |    293.24 ± 2.09 |    399.14 ± 5.89 |  1.361   |
| deepseek2 16B IQ4_NL |       pp16384 |    170.66 ± 0.76 |    269.01 ± 4.64 |  1.576   |

For reference, here is a comparison between standard attention with FA enabled and FlashMLA with this PR

| model                |          test | t/s (standard FA)|   t/s (PR)       |  Speedup |
| ---------------------| ------------: | ---------------: | ---------------: | -------: |
| deepseek2 16B IQ4_NL |         pp512 |    675.89 ± 7.49 |    681.72 ± 1.12 |  1.009   |   
| deepseek2 16B IQ4_NL |        pp1024 |    658.84 ± 1.08 |    648.71 ± 1.48 |  0.985   |   
| deepseek2 16B IQ4_NL |        pp2048 |    635.75 ± 1.70 |    598.99 ± 0.83 |  0.942   |   
| deepseek2 16B IQ4_NL |        pp4096 |    591.13 ± 0.06 |    514.62 ± 2.68 |  0.871   |   
| deepseek2 16B IQ4_NL |        pp8192 |    515.03 ± 2.53 |    399.14 ± 5.89 |  0.775   |   
| deepseek2 16B IQ4_NL |       pp16384 |    400.24 ± 0.74 |    269.01 ± 4.64 |  0.672   |   

I.e., still quite a bit slower than standard attention with FA enabled for long contexts, but much better than the original implementation.

The new functionality is enabled via `-mla 2 -fa` as command line arguments. I know, it is getting confusing, so here is a summary of what happens with the different `mla` and `fa` combinations:

* `mla = 0, fa = 0`: standard attention without FA. Works on the CPU and on CUDA. Large K- and V-cache required. The V cache cannot be quantized
* `mla = 0, fa = 1`: standard attention with FA. Works on the CPU and on CUDA. Large K- and V-cache required. The V cache can be quantized. Best PP performance, TG performance is slightly lower than standard attention without FA
* `mla = 1, fa = 0`: MLA attention. Works on the CPU and on CUDA. Smaller K- and smaller transposed V cache required.  The V cache cannot be quantized. Great TG performance, pathetic TG performance. 
* `mla = 1, fa = 1`: FlashMLA. Works only on the CPU. Only small K cache required. Great TG performance, slightly less pathetic PP performance
* `mla = 2, fa = 0`: FlashMLA . Works only on the CPU and on CUDA. Only small K cache required (the transposed V cache is computed on the fly). Great TG performance (but slightly lower than `mla = 1` for long contexts), pathetic PP performance.
* `mla = 2, fa = 1`: FlashMLA from this PR. Works only on CPU. Only small K cache required. Great TG performance, more acceptable PP performance. 

### Background

Let $X$ and $Q$ be the activations and the query after projection with their corresponding MQA tensors and after applying rotational position encoding (RoPE). In standard attention one computes (apart from scaling factors and masks that I'll omit for simplicity)

$$K = W_k X, \quad\quad V = W_v X,\quad\quad R = V_{\rm cache} {\rm softmax}(K_{\rm cache} Q)$$

In practice the $W_k$ and $W_v$ tensors are combined into $W_{kv}$ (the tensor `wkv_b` in `llama.cpp`), one computes $Y = W_{kv} X$, and the tensors $K$ and $V$ are views into $Y$. The matrix multiplication with $W_{kv}$ is performed only for the tokens in the batch being processed, the results are stored in the cache, and the tensors $V_{\rm cache}$ and $K_{\rm cache}$ are views into the KV cache.

With MLA one computes

$$Q' = W_k^T Q,\quad\quad R = W_v \left[ V_{\rm cache}  {\rm softmax}(K_{\rm cache} Q' \right]$$

where one stores $X$ directly into the K-cache, and $K_{\rm cache}$ is an appropriate view into the cache. $V_{\rm cache}$ is a transposed version of $K_{\rm cache}$ with FA is not used, or a slightly different view into the K-cache with FA or `mla=2`. The benefit of doing this reordering of the operations is that the cache becomes much smaller. But as these are not square matrices, the amount of multiply-adds (madds in the following) does depend on the order of the matrix multiplications. If we denote the number of madds in the standard attention implementation wit $N$, for the DeepSeek models the number of madds with MLA is $(576 + 512)/(192 + 128) \times N = 3.4 \times N$. Why is TG with MLA faster than with standard attention if one needs to do more computation? The difference comes from the shapes of the various matrices involved. TG with standard attention results in the tensor $Q$ being of shape $M \times 1 \times L$, so all multiplications are matrix-vector (a.k.a. GEMV), which are memory bound on basically any modern system (CPU or GPU). With MLA the shape of $Q'$ is $M' \times L$, so the calculation involves matrix-matrix multiplications (a.k.a. GEMM), which are much faster per madd, so one ends up with a better performance despite having computed more madds. But for PP in both cases we are dealing with GEMMs, so the `3.4X` more madds makes MLA PP processing slower. As an example, for 8k tokens with standard attention and FA, about 25% of the time is spent in the flash attention computation. We can estimate the expected MLA PP performance to be `0.75 + 0.25 x 3.4 = 1.6` times slower. From the above tables we see that in practice it is `515 t/s / 293 t/s = 1.75` times slower. As there are some other differences in the performed matrix multiplications, our back-of-the-envelope estimate comes quite close to the observed behavior.

So, how can we improve? We can rearrange the computation back to standard attention. The only difference: as we are storing $X$ into the cache, we need to multiply $W_{kv}$ with the **entire** content of the cache. This seems pretty  stupid at first glance (and I had had the idea to rearrange the multiplications quite a while ago but discarded it because of that), but if one sits down and counts the actual madds that are required, one finds that for DeepSeek this results in $(192 + 3 \times 128)/(192 + 128) = 1.8 \times N$ more madds than standard attention. I.e., we still need more madds, but significantly less madds than the existing MLA implementation. What about TG? We save the day by applying the rearranged matrix multiplications only if the number of tokens in the batch is greater than 1 (or some suitably chosen threshold). In this way we keep the good TG performance, keep the reduced cache size, and get improved prompt processing speed.

---

#### 💬 Conversation

👤 **davidsyoung** commented the **2025-03-08** at **14:58:12**:<br>

Getting a linking error on `iqk_flash_attn_noalibi`:

> 129.5 c++ -std=c++17 -fPIC -O3 -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wmissing-declarations -Wmissing-noreturn -pthread -fopenmp  -march=native -mtune=native -Wno-array-bounds -Wno-format-truncation -Wextra-semi -Iggml/include -Iggml/src -Iinclude -Isrc -Icommon -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_OPENMP -DGGML_USE_IQK_MULMAT -DGGML_USE_LLAMAFILE -DGGML_USE_CUDA -I/usr/local/cuda/include -I/usr/local/cuda/targets/x86_64-linux/include -DGGML_CUDA_USE_GRAPHS  -DLLAMA_USE_CURL ggml/src/iqk/iqk_quantize.o ggml/src/iqk/iqk_mul_mat.o ggml/src/llamafile/sgemm.o ggml/src/ggml-cuda.o ggml/src/ggml-cuda/acc.o ggml/src/ggml-cuda/arange.o ggml/src/ggml-cuda/argsort.o ggml/src/ggml-cuda/binbcast.o ggml/src/ggml-cuda/clamp.o ggml/src/ggml-cuda/concat.o ggml/src/ggml-cuda/conv-transpose-1d.o ggml/src/ggml-cuda/convert.o ggml/src/ggml-cuda/cpy.o ggml/src/ggml-cuda/diagmask.o ggml/src/ggml-cuda/dmmv.o ggml/src/ggml-cuda/fattn-tile-f16.o ggml/src/ggml-cuda/fattn-tile-f32.o ggml/src/ggml-cuda/fattn.o ggml/src/ggml-cuda/getrows.o ggml/src/ggml-cuda/im2col.o ggml/src/ggml-cuda/iqk_mmvq.o ggml/src/ggml-cuda/mmq.o ggml/src/ggml-cuda/mmvq.o ggml/src/ggml-cuda/norm.o ggml/src/ggml-cuda/pad.o ggml/src/ggml-cuda/pool2d.o ggml/src/ggml-cuda/quantize.o ggml/src/ggml-cuda/rope.o ggml/src/ggml-cuda/scale.o ggml/src/ggml-cuda/softcap.o ggml/src/ggml-cuda/softmax.o ggml/src/ggml-cuda/sumrows.o ggml/src/ggml-cuda/tsembd.o ggml/src/ggml-cuda/unary.o ggml/src/ggml-cuda/upscale.o ggml/src/ggml-cuda/template-instances/fattn-wmma-f16-instance-kqfloat-cpb16.o ggml/src/ggml-cuda/template-instances/fattn-wmma-f16-instance-kqfloat-cpb32.o ggml/src/ggml-cuda/template-instances/fattn-wmma-f16-instance-kqhalf-cpb16.o ggml/src/ggml-cuda/template-instances/fattn-wmma-f16-instance-kqhalf-cpb32.o ggml/src/ggml-cuda/template-instances/fattn-wmma-f16-instance-kqhalf-cpb8.o ggml/src/ggml-cuda/template-instances/mmq-instance-iq1_s.o ggml/src/ggml-cuda/template-instances/mmq-instance-iq2_s.o ggml/src/ggml-cuda/template-instances/mmq-instance-iq2_xs.o ggml/src/ggml-cuda/template-instances/mmq-instance-iq2_xxs.o ggml/src/ggml-cuda/template-instances/mmq-instance-iq3_s.o ggml/src/ggml-cuda/template-instances/mmq-instance-iq3_xxs.o ggml/src/ggml-cuda/template-instances/mmq-instance-iq4_nl.o ggml/src/ggml-cuda/template-instances/mmq-instance-iq4_xs.o ggml/src/ggml-cuda/template-instances/mmq-instance-q2_k.o ggml/src/ggml-cuda/template-instances/mmq-instance-q3_k.o ggml/src/ggml-cuda/template-instances/mmq-instance-q4_0.o ggml/src/ggml-cuda/template-instances/mmq-instance-q4_1.o ggml/src/ggml-cuda/template-instances/mmq-instance-q4_k.o ggml/src/ggml-cuda/template-instances/mmq-instance-q5_0.o ggml/src/ggml-cuda/template-instances/mmq-instance-q5_1.o ggml/src/ggml-cuda/template-instances/mmq-instance-q5_k.o ggml/src/ggml-cuda/template-instances/mmq-instance-q6_0.o ggml/src/ggml-cuda/template-instances/mmq-instance-q6_k.o ggml/src/ggml-cuda/template-instances/mmq-instance-q8_0.o ggml/src/ggml-cuda/template-instances/fattn-vec-f16-instance-hs128-q4_0-q4_0.o ggml/src/ggml-cuda/template-instances/fattn-vec-f32-instance-hs128-q4_0-q4_0.o ggml/src/ggml-cuda/template-instances/fattn-vec-f16-instance-hs128-q8_0-q8_0.o ggml/src/ggml-cuda/template-instances/fattn-vec-f16-instance-hs192-q8_0-q8_0.o ggml/src/ggml-cuda/template-instances/fattn-vec-f32-instance-hs128-q8_0-q8_0.o ggml/src/ggml-cuda/template-instances/fattn-vec-f32-instance-hs192-q8_0-q8_0.o ggml/src/ggml-cuda/template-instances/fattn-vec-f16-instance-hs128-f16-f16.o ggml/src/ggml-cuda/template-instances/fattn-vec-f16-instance-hs192-f16-f16.o ggml/src/ggml-cuda/template-instances/fattn-vec-f16-instance-hs256-f16-f16.o ggml/src/ggml-cuda/template-instances/fattn-vec-f16-instance-hs64-f16-f16.o ggml/src/ggml-cuda/template-instances/fattn-vec-f32-instance-hs128-f16-f16.o ggml/src/ggml-cuda/template-instances/fattn-vec-f32-instance-hs192-f16-f16.o ggml/src/ggml-cuda/template-instances/fattn-vec-f32-instance-hs256-f16-f16.o ggml/src/ggml-cuda/template-instances/fattn-vec-f32-instance-hs64-f16-f16.o ggml/src/ggml-cuda/template-instances/fattn-vec-f16-instance-hs128-q8_0-iq4_nl.o ggml/src/ggml-cuda/template-instances/fattn-vec-f32-instance-hs128-q8_0-iq4_nl.o ggml/src/ggml.o ggml/src/ggml-alloc.o ggml/src/ggml-backend.o ggml/src/ggml-quants.o ggml/src/ggml-aarch64.o src/llama.o src/llama-vocab.o src/llama-grammar.o src/llama-sampling.o src/unicode.o src/unicode-data.o common/common.o common/console.o common/ngram-cache.o common/sampling.o common/train.o common/grammar-parser.o common/build-info.o common/json-schema-to-grammar.o -Iexamples/server examples/server/server.o -o llama-server -lcuda -lcublas -lculibos -lcudart -lcublasLt -lpthread -ldl -lrt -L/usr/local/cuda/lib64 -L/usr/lib64 -L/usr/local/cuda/targets/x86_64-linux/lib -L/usr/local/cuda/lib64/stubs -L/usr/lib/wsl/lib  -lcurl 
129.7 /usr/bin/ld: ggml/src/ggml.o: in function `ggml_compute_forward_flash_attn_ext_f16':
129.7 ggml.c:(.text+0xb96b): undefined reference to `iqk_flash_attn_noalibi'
130.1 collect2: error: ld returned 1 exit status
130.1 make: *** [Makefile:1462: llama-server] Error 1

---

👤 **ikawrakow** commented the **2025-03-08** at **15:07:14**:<br>

Are you using `cmake` to build? The object file for the new file that I added (`iqk_flash_attn.cpp`) is missing from the link command. It should be automatically added with `cmake`.

---

👤 **davidsyoung** commented the **2025-03-08** at **15:20:58**:<br>

> Are you using `cmake` to build? The object file for the new file that I added (`iqk_flash_attn.cpp`) is missing from the link command. It should be automatically added with `cmake`.

Ah, I think that'll fix it. I was using the `full-cuda.Dockerfile` to run and I believe it was using a version of `make` still from previously forked `llama.cpp`.