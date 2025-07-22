### 🔀 [#511](https://github.com/ikawrakow/ik_llama.cpp/pull/511) - New IQ2_KT

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-09 |
| **Updated** | 2025-06-18 |

---

#### Description

This PR uses the new trellis introduced in #505 and applies it to `IQ2_KT`.

This leads to a slightly higher PPL for the models where the `IQ2_KT` on the main branch works, but is more stable and there are no longer NaNs for the models where the existing `IQ2_KT` was failing (Qwen3-30B-A3B and DeepSeek-Lite).

Performance is also great, except on the Apple GPU, where it is slower than the original `IQ2_KT` implementation. But on CUDA and on the CPU there are massive performance gains. Here an example of LLaMA-3.1-8B on RTX-4080 and Ryzen-7950X

| model            |       size |     params | backend    | fa |          test |              t/s |
| ---------------- | ---------: | ---------: | ---------- | -: | ------------: | ---------------: |
| llama 8B IQ2_KT  |   2.41 GiB |     8.03 B | CUDA       |  1 |         pp512 |  8972.05 ± 85.75 |
| llama 8B IQ2_KT  |   2.41 GiB |     8.03 B | CUDA       |  1 |         tg128 |    205.51 ± 0.22 |
| llama 8B IQ2_KT  |   2.41 GiB |     8.03 B | CPU        |  1 |         pp512 |    299.96 ± 4.58 |
| llama 8B IQ2_KT  |   2.41 GiB |     8.03 B | CPU        |  1 |         tg128 |     20.54 ± 0.18 |

---

#### 💬 Conversation

👤 **ubergarm** commented the **2025-06-10** at **18:41:50**:<br>

Just kicked the tires on this PR and looks good so far!

1. It compiles fine.
2. I managed to quantize [OpenBuddy-R1-0528-Distill-Qwen3-32B-Preview0-QAT](https://huggingface.co/OpenBuddy/OpenBuddy-R1-0528-Distill-Qwen3-32B-Preview0-QAT) using a variety of quants including `iq2_kt` and `iq4_kt` from this PR.

There is not a lot of info about this model, and honestly it doesn't behave like a 4bpw QAT and they don't have much details (i'll ask on their hf). Their chat tokenizing stuff seems wonky too, (but that is unrelated to this PR). (might need to stuff the `tokenizer_config.json -> "chat_template"` into the GGUF kv metadata.)

Anyway, the important thing is the new `iq2_kt` and` iq4_kt` are functional, able to quantize using normal imatrix, runs full perplexity clean with no `nan`, and outputs okay looking text (no gibberish) down to the `iq2_kt` even.

![ppl-OpenBuddy](https://github.com/user-attachments/assets/7ec38680-880b-4a78-ade9-4fbda3930abc)

I'll run some sweep benches too for speed comparisons.

---

👤 **ikawrakow** commented the **2025-06-11** at **14:36:11**:<br>

> Somewhat related I https://github.com/turboderp-org/exllamav3/pull/26#issuecomment-2957155162 on optimizing QTIP style quants by using pre-computed Hessians for each layer/tensor. Zero pressure to look or distract, just interesting folks are already uploading Hessians for some models.

This is the sort of thing we do not want to do here. It leads to overfitting, needs a huge amount of compute, which makes it inaccessible for the average enthusiast, so basically only good for pushing out yet another paper to arXiv.

---

👤 **louiehelm** commented the **2025-06-11** at **17:03:36**:<br>

Great work! Love seeing improved performance on the trellis quants ik.

Some alternate MCG multipliers (with no addition) have lower PPL than QTIP 3INST defaults:

### Meta-Llama-3.1-8B-Instruct
| **Quantization** | **Version** | **PPL** |
|------------------|-------------|---------|
| **f32** | - | 7.3210 |
| **IQ2_KT** | #511 default | 11.0029 |
| | 0xCBAC1FED (3417055213) | 10.9466 |
| **IQ3_KT** | #511 default | 8.1319 |
| | 0xCBAC1FED (3417055213) | 8.0776 |
| **IQ4_KT** | #511 default | 7.5620 |
| | 0xCBAC1FED (3417055213) | 7.5591 |

Just chiming in because it might be a great time to take the 0.5% higher fidelity of ditching the default QTIP multiplier+addition params if you're already introducing a breaking change to IQx_KT quants anyway. For IQ2_K, this gains back a good chunk of what was lost by switching to your new decoder scheme, while also making IQ3_KT and IQ4_KT both better than #511 and in some cases even better than prior versions.

Also, ka = `0xCBAC1FED`  and kb = 0 is a more well-tested distribution than 3INST defaults and currently the best known so far. Obviously if this change is added kb can be deleted rather than updated to 0 (for a small speed boost). This is how to test it further with more models to confirm PPL shows improvements more broadly:

`./test_IQ2_KT.sh 3417055213`

```
#!/bin/sh

find . -type f \( -name "*.cpp" -o -name "*.cu" -o -name "*.cuh" \) -exec sed -i "s/ ka = 89226354/ ka = $1/g" {} +
find . -type f \( -name "*.cpp" -o -name "*.cu" -o -name "*.cuh" \) -exec sed -i "s/ kb = 64248484/ kb = 0/g" {} +
cmake -B build -DGGML_CUDA=ON -DGGML_RPC=OFF -DGGML_BLAS=OFF  -DGGML_SCHED_MAX_COPIES=1
cmake --build build --config Release -j $(nproc)
find . -type f \( -name "*.cpp" -o -name "*.cu" -o -name "*.cuh" \) -exec sed -i "s/ ka = $1/ ka = 89226354/g" {} +
find . -type f \( -name "*.cpp" -o -name "*.cu" -o -name "*.cuh" \) -exec sed -i "s/ kb = 0/ kb = 64248484/g" {} +

build/bin/llama-quantize --imatrix ~/llms/Meta-Llama-3.1-8B-Instruct-f32-imatrix.dat ~/llms/Meta-Llama-3.1-8B-Instruct-f32.gguf Meta-Llama-3.1-8B-Instruct-IQ2_KT.gguf IQ2_KT
# build/bin/llama-perplexity -m ~/llms/Meta-Llama-3.1-8B-Instruct-f32.gguf -f ~/llms/wiki.test.raw  --ctx-size 512 --ubatch-size 512 -fa -ngl 99 --seed 1337 # BASELINE TEST

build/bin/llama-perplexity -m Meta-Llama-3.1-8B-Instruct-IQ2_KT.gguf -f ~/llms/wiki.test.raw --ctx-size 512 --ubatch-size 512 -fa -ngl 99 --seed 1337

rm -f Meta-Llama-3.1-8B-Instruct-IQ2_KT.gguf
```

---

👤 **louiehelm** commented the **2025-06-12** at **22:27:27**:<br>

Yes initial tests above were on #511. Needs more testing... Qwen3 1.7B IQ2_KT = 2.5% lower PPL.... Magistral 24B IQ2_KT = 50% lower PPL [default model bugged perhaps?]

---

👤 **Nexesenex** commented the **2025-06-13** at **10:32:43**:<br>

> > But on a Llama 3.3 70b type model (iq2_kt for the ffns, attn_q and attn_o), the final wikitest 512 perplexity is 1% lower with ka = 3417055213 and kb = 0 compared to the original couple.
> 
> 1% of what? Can you give the specific PPL values?

Here is :

For Llama 3.3 70b type model (iq2_kt for the ffns, attn_q and attn_o, q6 for embedding, iq5_ks_r4 for output and attn_v, and iq4_ks_r4 for attn_k).
- final wikitest 512 perplexity is 1% lower with ka = 89226354 and kb = 64248484. Final estimate: PPL = 6.1443 +/- 0.03805
- final wikitest 512 perplexity is 1% lower with ka = 3417055213 and kb = 0. Final estimate: PPL = 6.0739 +/- 0.03762

---

👤 **ikawrakow** commented the **2025-06-13** at **16:59:17**:<br>

Did you also try `IQ4_KT`?

I tried LlaMA-3.1-8B-Instruct and PPL goes up by ~0.5%, which is a lot for 4 bit. `IQ2_KT` has 30-40% quantization error, so 1% improvement is not that much. But `IQ4_KT` has 2.5% quantization error, so a 0.5% increase is not good. Strangely enough, with this multiplier `IQ4_KT` quantization takes much longer, while `IQ2_KT` quantization becomes faster.

I only changed the CUDA implementation so I can run PPL. When I make the change in the CPU code I'll push to a new branch. Probably tomorrow.

---

👤 **ubergarm** commented the **2025-06-13** at **18:52:10**:<br>

> Did you also try IQ4_KT?

Just got home and tried louiehelm's 0xCBAC1FED patch on this PR511.


### Patch

<details>

<summary>👈 `0xCBAC1FED` Patch</summary>

```bash
diff --git a/ggml/src/ggml-cuda/convert.cu b/ggml/src/ggml-cuda/convert.cu
index a602e47d..45de337e 100644
--- a/ggml/src/ggml-cuda/convert.cu
+++ b/ggml/src/ggml-cuda/convert.cu
@@ -341,15 +341,15 @@ inline __device__ int nearest_int(float fval) {
 }
 
 int __device__ __forceinline__ trellis_next_int(uint32_t& val) {
-    constexpr uint32_t ka = 89226354;
-    constexpr uint32_t kb = 64248484;
+    constexpr uint32_t ka = 3417055213;
+    constexpr uint32_t kb = 0;
     val = ka*val + kb;
     return ggml_cuda_dp4a(val & 0x3f3f3f3f, 0x01010101, -126);
 }
 
 float __device__ __forceinline__ trellis_next(uint32_t& val) {
-    constexpr uint32_t ka = 89226354;
-    constexpr uint32_t kb = 64248484;
+    constexpr uint32_t ka = 3417055213;
+    constexpr uint32_t kb = 0;
     constexpr uint32_t kmask = 0x8fff8fff;
     constexpr uint32_t km32 = 0x3b603b60;
     uint32_t s;
diff --git a/ggml/src/ggml-cuda/dmmv.cu b/ggml/src/ggml-cuda/dmmv.cu
index 50e6458d..5e0226ed 100644
--- a/ggml/src/ggml-cuda/dmmv.cu
+++ b/ggml/src/ggml-cuda/dmmv.cu
@@ -16,8 +16,8 @@ static_assert(K_QUANTS_PER_ITERATION == 1 || K_QUANTS_PER_ITERATION == 2, "K_QUA
 #endif
 
 static __device__ __forceinline__ uint32_t trellis_next(uint32_t& val) {
-    constexpr uint32_t ka = 89226354;
-    constexpr uint32_t kb = 64248484;
+    constexpr uint32_t ka = 3417055213;
+    constexpr uint32_t kb = 0;
     constexpr uint32_t kmask = 0x8fff8fff;
     constexpr uint32_t km32 = 0x3b603b60;
     val = ka*val + kb;
diff --git a/ggml/src/ggml-cuda/iqk_mmvq.cu b/ggml/src/ggml-cuda/iqk_mmvq.cu
index df1cea89..34402358 100644
--- a/ggml/src/ggml-cuda/iqk_mmvq.cu
+++ b/ggml/src/ggml-cuda/iqk_mmvq.cu
@@ -398,8 +398,8 @@ __device__ __forceinline__ void vec_dot_iq4_ks_q8_1(
 __device__ __forceinline__ void vec_dot_iq4_kt_q8_1(
     const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {
 
-    constexpr uint32_t ka = 89226354;
-    constexpr uint32_t kb = 64248484;
+    constexpr uint32_t ka = 3417055213;
+    constexpr uint32_t kb = 0;
     constexpr uint32_t km = 0x3f3f3f3f;
 
     float scale = *(const float *)vbq;
@@ -436,8 +436,8 @@ __device__ __forceinline__ void vec_dot_iq4_kt_q8_1(
 __device__ __forceinline__ void vec_dot_iq2_kt_q8_1(
     const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {
 
-    constexpr uint32_t ka = 89226354;
-    constexpr uint32_t kb = 64248484;
+    constexpr uint32_t ka = 3417055213;
+    constexpr uint32_t kb = 0;
     constexpr uint32_t km = 0x3f3f3f3f;
 
     float scale = *(const float *)vbq;
diff --git a/ggml/src/ggml-cuda/mmq.cuh b/ggml/src/ggml-cuda/mmq.cuh
index e2c76a85..2b5a6df5 100644
--- a/ggml/src/ggml-cuda/mmq.cuh
+++ b/ggml/src/ggml-cuda/mmq.cuh
@@ -2799,8 +2799,8 @@ template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinlin
 template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq4_kt(
     const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {
 
-    constexpr uint32_t ka = 89226354;
-    constexpr uint32_t kb = 64248484;
+    constexpr uint32_t ka = 3417055213;
+    constexpr uint32_t kb = 0;
     constexpr uint32_t km = 0x3f3f3f3f;
 
 #ifdef INT8_MMA_AVAILABLE
@@ -2872,8 +2872,8 @@ template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinlin
 template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq2_kt(
     const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {
 
-    constexpr uint32_t ka = 89226354;
-    constexpr uint32_t kb = 64248484;
+    constexpr uint32_t ka = 3417055213;
+    constexpr uint32_t kb = 0;
     constexpr uint32_t km = 0x3f3f3f3f;
 
 #ifdef INT8_MMA_AVAILABLE
diff --git a/ggml/src/iqk/iqk_gemm_ktquants.cpp b/ggml/src/iqk/iqk_gemm_ktquants.cpp
index 8b8cae14..41b9b2d6 100644
--- a/ggml/src/iqk/iqk_gemm_ktquants.cpp
+++ b/ggml/src/iqk/iqk_gemm_ktquants.cpp
@@ -14,8 +14,8 @@
 namespace {
 
 inline uint32_t trellis_next(uint32_t& val) {
-    constexpr uint32_t ka = 89226354;
-    constexpr uint32_t kb = 64248484;
+    constexpr uint32_t ka = 3417055213;
+    constexpr uint32_t kb = 0;
     constexpr uint32_t kmask = 0x8fff8fff;
     constexpr uint32_t km32 = 0x3b603b60;
     val = val*ka + kb;
@@ -31,8 +31,8 @@ inline float trellis_gen(uint32_t& val, uint32_t* s) {
 struct Trellis1 {
     constexpr static uint32_t kmask = 0x8fff8fff;
     constexpr static uint32_t km32 = 0x3b603b60;
-    constexpr static uint32_t ka = 89226354;
-    constexpr static uint32_t kb = 64248484;
+    constexpr static uint32_t ka = 3417055213;
+    constexpr static uint32_t kb = 0;
     constexpr static uint32_t ka1 = ka*ka;
     constexpr static uint32_t kb1 = kb*ka+kb;
     constexpr static uint32_t ka2 = ka1*ka;
@@ -76,8 +76,8 @@ inline __m256 trellis_gen8(__m256i i8) {
 struct Trellis2 {
     constexpr static uint32_t kmask = 0x8fff8fff;
     constexpr static uint32_t km32 = 0x3b603b60;
-    constexpr static uint32_t ka = 89226354;
-    constexpr static uint32_t kb = 64248484;
+    constexpr static uint32_t ka = 3417055213;
+    constexpr static uint32_t kb = 0;
     constexpr static uint32_t ka1 = ka*ka;
     constexpr static uint32_t kb1 = kb*ka+kb;
     constexpr static uint32_t ka2 = ka1*ka;
@@ -100,8 +100,8 @@ struct Trellis2 {
 
 template <bool is_8 = false>
 struct Trellis3 {
-    constexpr static uint32_t ka = 89226354;
-    constexpr static uint32_t kb = 64248484;
+    constexpr static uint32_t ka = 3417055213;
+    constexpr static uint32_t kb = 0;
     constexpr static uint32_t ka1 = ka*ka;
     constexpr static uint32_t kb1 = kb*ka+kb;
     constexpr static uint32_t ka2 = ka1*ka;
@@ -913,8 +913,8 @@ namespace {
 struct Trellis1 {
     constexpr static uint32_t kmask = 0x8fff8fff;
     constexpr static uint32_t km32 = 0x3b603b60;
-    constexpr static uint32_t ka = 89226354;
-    constexpr static uint32_t kb = 64248484;
+    constexpr static uint32_t ka = 3417055213;
+    constexpr static uint32_t kb = 0;
     constexpr static uint32_t ka1 = ka*ka;
     constexpr static uint32_t kb1 = kb*ka+kb;
     constexpr static uint32_t ka2 = ka1*ka;
@@ -1419,8 +1419,8 @@ void mul_mat_iq4_kt_F32_T(int n, const void * vx, size_t bx, const DataInfo& inf
 }
 
 struct Trellis3 {
-    constexpr static uint32_t ka = 89226354;
-    constexpr static uint32_t kb = 64248484;
+    constexpr static uint32_t ka = 3417055213;
+    constexpr static uint32_t kb = 0;
     constexpr static uint32_t ka1 = ka*ka;
     constexpr static uint32_t kb1 = kb*ka+kb;
     constexpr static uint32_t ka2 = ka1*ka;
diff --git a/ggml/src/iqk/iqk_quantize.cpp b/ggml/src/iqk/iqk_quantize.cpp
index b6bff0a1..7c052989 100644
--- a/ggml/src/iqk/iqk_quantize.cpp
+++ b/ggml/src/iqk/iqk_quantize.cpp
@@ -7454,8 +7454,8 @@ public:
     inline float find_best_inverse_scale(const float * xb, const float * weight, const int * best_idx) const;
 
     static inline void set_values(uint32_t i, float * result, float scale, int offset = 4096) {
-        constexpr uint32_t ka = 89226354;
-        constexpr uint32_t kb = 64248484;
+        constexpr uint32_t ka = 3417055213;
+        constexpr uint32_t kb = 0;
         uint32_t x = i + offset;
         if constexpr (is_int) {
             uint32_t s;
```

</details>

### Data
Here is the comparison of the same [OpenBuddy-R1-0528-Distill-Qwen3-32B-Preview0-QAT](https://huggingface.co/OpenBuddy/OpenBuddy-R1-0528-Distill-Qwen3-32B-Preview0-QAT) used above between regular PR511 and the patched version.

#### PR511 (from above)
* IQ4_KT
  - `7.0114 +/- 0.04516`
  - `main: quantize time = 1465481.74 ms` 24.42 min
* IQ2_KT (token_embd|output)@iq4_kt
  - `8.7412 +/- 0.05859`
  - `main: quantize time = 865473.26 ms` 14.42 min

#### 0xCBAC1FED Patch
* IQ4_KT
  - `7.0210 +/- 0.04529`
  - `main: quantize time = 1518609.40 ms` 25.31 min
* IQ2_KT (token_embd|output)@iq4_kt
  - `8.6883 +/- 0.05866`
  - `main: quantize time = 877350.58 ms` 14.62 min

### Comparison
* IQ4_KT
  - Patched version is ~0.14% "worse" perplexity
  - Patched version quantized ~3.6% slower
* IQ4_KT (token_embd|output)@iq4_kt
  - Patched version is ~0.61% "better" perplexity
  - Patched version quantized ~1.4% slower

### Conclusion
Well, its hard to say for a single run given the deltas seem within the margin of error. I'm not sure if it is possible/worthwhile to save the `ka`/`kb` values into the GGUF metadata and load them per model to support both? This would allow any future discovered magic numbers as well (couldn't optimize away kb=0 though).

---

👤 **ikawrakow** commented the **2025-06-18** at **13:21:51**:<br>

Closing in favor of #529