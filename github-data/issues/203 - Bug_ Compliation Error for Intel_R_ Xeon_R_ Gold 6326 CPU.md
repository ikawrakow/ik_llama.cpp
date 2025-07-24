### 🐛 [#203](https://github.com/ikawrakow/ik_llama.cpp/issues/203) - Bug: Compliation Error for Intel(R) Xeon(R) Gold 6326 CPU

| **Author** | `Flying-Cloud` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-02-12 |
| **Updated** | 2025-02-12 |

---

#### Description

### What happened?

Hello! I found some error when build ik_llama.cpp project. Running the command 'cmake --build build --config Release'
I found errors in that the cpu in my system Intel(R) Xeon(R) Gold 6326 CPU does not support AVX512BF16 but do support other AVX512 features.
So when compling iqk_mul_mat.cpp, encounter errors for BF16 data.
Can you help me fix this error, or some suggestions for me to fix. Thanks!
```
llm/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp: In instantiation of ‘{anonymous}::QFBase::Data {anonymous}::QFT<Float, nrc_in>::load1(int, int) const [with Float = ggml_bf16_t; int nrc_in = 1; {anonymous}::QFBase::Data = __vector(16) float]’:
llm/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:8249:10:   required from ‘void {anonymous}::mul_mat_Qx_Qy_MxN(int, const char*, size_t, int, const {anonymous}::DataInfo&) [with Qy = {anonymous}::QFT<ggml_bf16_t, 1>; Qx = {anonymous}::QFT<ggml_bf16_t, 5>; size_t = long unsigned int]’
llm/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:8362:65:   required from ‘void {anonymous}::mul_mat_fX_fY_T(int, const void*, size_t, const {anonymous}::DataInfo&, int) [with int nrc_y = 1; FloatX = ggml_bf16_t; FloatY = ggml_bf16_t; size_t = long unsigned int]’
llm/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:8643:17:   required from ‘void {anonymous}::set_mul_mat_f({anonymous}::MulMat&) [with FloatX = ggml_bf16_t; FloatY = ggml_bf16_t]’
ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:8685:76:   required from here
ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:8173:68: error: no matching function for call to ‘{anonymous}::QFT<ggml_bf16_t, 1>::load(const ggml_bf16_t*) const’
 8173 |     IQK_ALWAYS_INLINE Data load1(int iy, int i) const { return load(y[iy] + k_step*i); }
```

### Name and Version

Intel(R) Xeon(R) Gold 6326 CPU Ubuntu 20.04

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell

```

---

#### 💬 Conversation

👤 **Flying-Cloud** commented the **2025-02-12** at **08:13:39**:<br>

I have added the overload function for bf16 as follows, which resolved the compilation issue in iqk_mul_mat.cpp.
I am not quite sure if it is right functionally but it did fix the compliation bug

```   
static inline Data load(const ggml_bf16_t * x) {
        // Load BF16 data into __m256i
        __m256i bf16_data = _mm256_loadu_si256((const __m256i *)x);
        // Convert BF16 to FP32 by shifting left 16 bits
        __m512i bf16_extended = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_data), 16);
        // Cast to __m512 (FP32)
        return _mm512_castsi512_ps(bf16_extended);
    }
```

---

👤 **ikawrakow** commented the **2025-02-12** at **08:18:53**:<br>

Yes, this is the right fix. I have disabled `BF16` on my CPU and tested that PR #204 works correctly (not a very thorough testing, but token generation and perplexity seem fine).

Thank you for the report! It is always helpful when things get tested on more diverse systems. Let me know if #204 works correctly for you.

---

👤 **ikawrakow** commented the **2025-02-12** at **08:18:53**:<br>

Yes, this is the right fix. I have disabled `BF16` on my CPU and tested that PR #204 works correctly (not a very thorough testing, but token generation and perplexity seem fine).

---

👤 **Flying-Cloud** commented the **2025-02-12** at **11:28:50**:<br>

> Yes, this is the right fix. I have disabled `BF16` on my CPU and tested that PR [#204](https://github.com/ikawrakow/ik_llama.cpp/pull/204) works correctly (not a very thorough testing, but token generation and perplexity seem fine).
> 
> Thank you for the report! It is always helpful when things get tested on more diverse systems. Let me know if [#204](https://github.com/ikawrakow/ik_llama.cpp/pull/204) works correctly for you.

Lines 16082 in iqk_mul_mat.cpp  should be changed from
```
#ifdef HAVE_FANCY_SIMD
        case GGML_TYPE_BF16: {
            HelperBF16<Dv, k_step> vh(v, stride_v);
            iqk_flash_helper<Dk, Dv, k_step>(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, mask, scale, softcap, qkv);
        } break;
#endif
```
to
```
#if defined(HAVE_FANCY_SIMD) && defined(__AVX512BF16__)
        case GGML_TYPE_BF16: {
            HelperBF16<D, k_step> vh(v, stride_v);
            iqk_flash_helper<D, k_step>(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, mask, scale, softcap, qkv);
        } break;
#endif
```
Otherwise, there will still be error that HelperBF16 not defined

---

👤 **ikawrakow** commented the **2025-02-12** at **11:49:04**:<br>

Do you want to submit a PR (I'll close #204 if you do). Or do you want me to add it to #204?

---

👤 **Flying-Cloud** commented the **2025-02-12** at **11:51:48**:<br>

For convenience, add it to #204 is fined. There is no other issue when add these two codes, thanks for your effort