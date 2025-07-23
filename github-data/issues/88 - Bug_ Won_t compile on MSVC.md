### 🐛 [#88](https://github.com/ikawrakow/ik_llama.cpp/issues/88) - Bug: Won't compile on MSVC

| **Author** | `saood06` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-10-14 |
| **Updated** | 2024-10-19 |

---

#### Description

### What happened?

As mentioned in #82 this does not compile with MSVC. I was able to get through the issues and make it compile on my machine, no PR right now, but if this issue stays open long enough I will create one with an actual fix. 

Here's the git diff of the changes I made:
```diff
diff --git a/ggml/src/iqk/iqk_mul_mat.cpp b/ggml/src/iqk/iqk_mul_mat.cpp
index 66d26a25..3a40a4b7 100644
--- a/ggml/src/iqk/iqk_mul_mat.cpp
+++ b/ggml/src/iqk/iqk_mul_mat.cpp
@@ -252,7 +252,7 @@ const uint64_t keven_signs[128] = {

 }

-#if defined __x86_64__
+#if defined _M_X64

 #if defined HAVE_FANCY_SIMD
     #undef HAVE_FANCY_SIMD
@@ -7024,10 +7024,10 @@ struct F16 {
     static inline float reduce_max(Data data) { return hmax_float_8(data); }
     static inline float reduce_add(Data data) { return hsum_float_8(data); }
     template <int k_step> static inline float reduce_max(const Data * data) {
-        return reduce_T<k_step, _mm256_max_ps, &F16::reduce_max>(data);
+        return reduce_T1<k_step, &F16::reduce_max>(data);
     }
     template <int k_step> static inline float reduce_add(const Data * data) {
-        return reduce_T<k_step, _mm256_add_ps, &F16::reduce_add>(data);
+        return reduce_T2<k_step, &F16::reduce_add>(data);
     }
 #else
     using Data = float16x8_t;
@@ -7065,18 +7065,34 @@ struct F16 {
         return reduce_T<k_step, vaddq_f16, &F16::reduce_add>(data);
     }
 #endif
-    template <int k_step, Data (*Op_combine)(Data, Data), float (*Op)(Data)>
-    static float reduce_T(const Data * data) {
+    template <int k_step, float (*Op)(Data)>
+    static float reduce_T1(const Data * data) {
         float result;
         if constexpr (k_step/block_size == 1) {
             result = Op(data[0]);
         }
         else if constexpr (k_step/block_size == 2) {
-            result = Op(Op_combine(data[0], data[1]));
+            result = Op(_mm256_max_ps(data[0], data[1]));
         }
         else {
-            auto vmax = Op_combine(data[0], data[1]);
-            for (int l = 2; l < k_step/block_size; ++l) vmax = Op_combine(vmax, data[l]);
+            auto vmax = _mm256_max_ps(data[0], data[1]);
+            for (int l = 2; l < k_step/block_size; ++l) vmax = _mm256_max_ps(vmax, data[l]);
+            result = Op(vmax);
+        }
+        return result;
+    }
+    template <int k_step, float (*Op)(Data)>
+    static float reduce_T2(const Data * data) {
+        float result;
+        if constexpr (k_step/block_size == 1) {
+            result = Op(data[0]);
+        }
+        else if constexpr (k_step/block_size == 2) {
+            result = Op(_mm256_add_ps(data[0], data[1]));
+        }
+        else {
+            auto vmax = _mm256_add_ps(data[0], data[1]);
+            for (int l = 2; l < k_step/block_size; ++l) vmax = _mm256_add_ps(vmax, data[l]);
             result = Op(vmax);
         }
         return result;
````

For reference the error messages for the error with reduce_T:
..\ik_llama.cpp\ggml\src\iqk\iqk_mul_mat.cpp(7027,16): error C2672: '`anonymous-namespace'::F16::reduce_T': no matching overloaded function found [..\ik_llama.cpp\build-rpc-cuda1-ik\ggml\src\ggml.vcxproj]
..\ik_llama.cpp\ggml\src\iqk\iqk_mul_mat.cpp(7027,1): error C7551: '`anonymous-namespace'::F16::reduce_T': template parameter 'Op_combine': '_mm256_max_ps': purely intrinsic functions have no address for use as a non-type template argument [..\ik_llama.cpp\build-rpc-cuda1-ik\ggml\src\ggml.vcxproj]
..\ik_llama.cpp\ggml\src\iqk\iqk_mul_mat.cpp(7030,16): error C2672: '`anonymous-namespace'::F16::reduce_T': no matching overloaded function found [..\ik_llama.cpp\build-rpc-cuda1-ik\ggml\src\ggml.vcxproj]
..\ik_llama.cpp\ggml\src\iqk\iqk_mul_mat.cpp(7030,1): error C7551: '`anonymous-namespace'::F16::reduce_T': template parameter 'Op_combine': '_mm256_add_ps': purely intrinsic functions have no address for use as a non-type template argument [..\ik_llama.cpp\build-rpc-cuda1-ik\ggml\src\ggml.vcxproj]



### Name and Version

version: 3459 (baab1d9a)
built with MSVC 19.28.29335.0 for x64

### What operating system are you seeing the problem on?

Windows

### Relevant log output

_No response_

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2024-10-15** at **05:52:32**:<br>

Thanks for the fix. This is the only issue MSVC has with the 10k+ LOC that I have added? This is a pleasant surprise.

Please submit a PR. As I don't have the ability to test on Windows, the issue will stay open until someone else fixes it.

---

👤 **Nexesenex** commented the **2024-10-17** at **18:48:34**:<br>

@saood06 : It worked perfectly for me, thanks.

---

👤 **ikawrakow** commented the **2024-10-19** at **18:00:25**:<br>

Fixed via #93