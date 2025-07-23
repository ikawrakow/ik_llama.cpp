### 📝 [#214](https://github.com/ikawrakow/ik_llama.cpp/issues/214) - AVX512 build error

| **Author** | `pt13762104` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-02-21 |
| **Updated** | 2025-02-21 |

---

#### Description

When building for AVX512, this error occurs:
```cpp
/home/why/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp: In member function '__m256i {anonymous}::DequantizerIQ6K::make_one(__m256i, __m256i) const':
/home/why/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:2750:114: warning: overflow in conversion from 'int' to 'char' changes value from '255' to '-1' [-Woverflow]
 2750 |         auto mask1 = _mm256_andnot_si256(_mm256_or_si256(mask4, _mm256_or_si256(mask2, mask3)), _mm256_set1_epi8(0xff));
      |                                                                                                                  ^~~~
/home/why/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp: At global scope:
/home/why/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:2846:48: warning: overflow in conversion from 'int' to 'short int' changes value from '65534' to '-2' [-Woverflow]
 2846 |     const __m256i bmask    = _mm256_set1_epi16(0xfffe);
      |                                                ^~~~~~
/home/why/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp: In member function 'void {anonymous}::QFT<Float, nrc_in>::load_r4(int, int, {anonymous}::QFBase::Data*) const':
/home/why/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:8550:42: error: cannot convert '{anonymous}::QFBase::Data' {aka '__m512'} to '__m256'
 8550 |         auto t0 = _mm256_unpacklo_ps(xv[0], xv[1]);
      |                                      ~~~~^
      |                                          |
      |                                          {anonymous}::QFBase::Data {aka __m512}
In file included from /usr/lib/gcc/x86_64-redhat-linux/14/include/immintrin.h:43,
                 from /home/why/ik_llama.cpp/ggml/src/./ggml-impl.h:449,
                 from /home/why/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:24:
/usr/lib/gcc/x86_64-redhat-linux/14/include/avxintrin.h:1100:28: note:   initializing argument 1 of '__m256 _mm256_unpacklo_ps(__m256, __m256)'
 1100 | _mm256_unpacklo_ps (__m256 __A, __m256 __B)
      |                     ~~~~~~~^~~
/home/why/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:8551:42: error: cannot convert '{anonymous}::QFBase::Data' {aka '__m512'} to '__m256'
 8551 |         auto t1 = _mm256_unpacklo_ps(xv[2], xv[3]);
      |                                      ~~~~^
      |                                          |
      |                                          {anonymous}::QFBase::Data {aka __m512}
/usr/lib/gcc/x86_64-redhat-linux/14/include/avxintrin.h:1100:28: note:   initializing argument 1 of '__m256 _mm256_unpacklo_ps(__m256, __m256)'
 1100 | _mm256_unpacklo_ps (__m256 __A, __m256 __B)
      |                     ~~~~~~~^~~
/home/why/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:8552:42: error: cannot convert '{anonymous}::QFBase::Data' {aka '__m512'} to '__m256'
 8552 |         auto t2 = _mm256_unpackhi_ps(xv[0], xv[1]);
      |                                      ~~~~^
      |                                          |
      |                                          {anonymous}::QFBase::Data {aka __m512}
/usr/lib/gcc/x86_64-redhat-linux/14/include/avxintrin.h:1094:28: note:   initializing argument 1 of '__m256 _mm256_unpackhi_ps(__m256, __m256)'
 1094 | _mm256_unpackhi_ps (__m256 __A, __m256 __B)
      |                     ~~~~~~~^~~
/home/why/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:8553:42: error: cannot convert '{anonymous}::QFBase::Data' {aka '__m512'} to '__m256'
 8553 |         auto t3 = _mm256_unpackhi_ps(xv[2], xv[3]);
      |                                      ~~~~^
      |                                          |
      |                                          {anonymous}::QFBase::Data {aka __m512}
/usr/lib/gcc/x86_64-redhat-linux/14/include/avxintrin.h:1094:28: note:   initializing argument 1 of '__m256 _mm256_unpackhi_ps(__m256, __m256)'
 1094 | _mm256_unpackhi_ps (__m256 __A, __m256 __B)
      |                     ~~~~~~~^~~
```
I have tried multiple copies of GCC 14, they produce the same result. The AVX2 builds fine, it's AVX512 that have trouble building.

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-02-21** at **07:09:29**:<br>

Does #215 fix it?

---

👤 **pt13762104** commented the **2025-02-21** at **07:41:00**:<br>

I'll try, thanks

---

👤 **pt13762104** commented the **2025-02-21** at **07:41:00**:<br>

I'll try

---

👤 **pt13762104** commented the **2025-02-21** at **07:51:50**:<br>

It doesn't...

---

👤 **ikawrakow** commented the **2025-02-21** at **07:53:16**:<br>

What is the new compilation error?

---

👤 **pt13762104** commented the **2025-02-21** at **07:59:04**:<br>

Seems like that fixed it, my bad

---

👤 **ikawrakow** commented the **2025-02-21** at **10:35:38**:<br>

@pt13762104 I think #216 really fixes it. Can you try? Thanks.

---

👤 **pt13762104** commented the **2025-02-21** at **11:05:47**:<br>

I'll try to run a model to see if it's working

---

👤 **pt13762104** commented the **2025-02-21** at **13:31:25**:<br>

It seemed to work fine, the models run, it compiles nicely...

---

👤 **ikawrakow** commented the **2025-02-21** at **13:33:09**:<br>

OK, thanks! I'll merge #216