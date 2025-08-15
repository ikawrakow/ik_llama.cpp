### 📝 [#447](https://github.com/ikawrakow/ik_llama.cpp/issues/447) - Compilation Error: Error C2676

| **Author** | `quasar-of-mikus` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-23 |
| **Updated** | 2025-05-23 |

---

#### Description

Got this when trying to compile the latest commit. The last time I ran a build was commit `2ec2229` and that was successful.
Windows 10
```
# usual command
cmake -B ./build -DGGML_CUDA=ON -DGGML_BLAS=OFF -DGGML_SCHED_MAX_COPIES=1
cmake --build ./build --config Release -j 20
```

Log:
```
  iqk_gemm_kquants.cpp
  iqk_gemm_ktquants.cpp
C:\Textgen\ik_llama.cpp\ggml\src\iqk\iqk_gemm_ktquants.cpp(47,61): error C2676: binary '^': '__m256i' does not define t
his operator or a conversion to a type acceptable to the predefined operator [C:\Textgen\ik_llama.cpp\build\ggml\src\gg
ml.vcxproj]
C:\Textgen\ik_llama.cpp\ggml\src\iqk\iqk_gemm_ktquants.cpp(83,46): error C2676: binary '^': '__m256i' does not define t
his operator or a conversion to a type acceptable to the predefined operator [C:\Textgen\ik_llama.cpp\build\ggml\src\gg
ml.vcxproj]
C:\Textgen\ik_llama.cpp\ggml\src\iqk\iqk_gemm_ktquants.cpp(120,65): error C2676: binary '^': '__m256i' does not define
this operator or a conversion to a type acceptable to the predefined operator [C:\Textgen\ik_llama.cpp\build\ggml\src\g
gml.vcxproj]
  iqk_gemm_iquants.cpp
  iqk_gemm_iqk_quants.cpp
C:\Textgen\ik_llama.cpp\ggml\src\iqk\iqk_gemm_iqk_quants.cpp(810,84): warning C4244: 'argument': conversion from 'const
 uint16_t' to 'char', possible loss of data [C:\Textgen\ik_llama.cpp\build\ggml\src\ggml.vcxproj]
C:\Textgen\ik_llama.cpp\ggml\src\iqk\iqk_gemm_iqk_quants.cpp(1279,34): message : see reference to function template ins
tantiation '__m256i `anonymous-namespace'::DequantizerIQ2KS::new_block<Q8<1,block_q8_K>>(int,const Q8 &,__m256 *)' bein
g compiled [C:\Textgen\ik_llama.cpp\build\ggml\src\ggml.vcxproj]
          with
          [
              Q8=Q8<1,block_q8_K>
          ]
C:\Textgen\ik_llama.cpp\ggml\src\iqk\iqk_gemm_iqk_quants.cpp(2050,1): message : see reference to function template inst
antiation 'void `anonymous-namespace'::mul_mat_qX_K_q8_K_T<Dequantizer,1>(int,const void *,size_t,const DataInfo &,int)
' being compiled [C:\Textgen\ik_llama.cpp\build\ggml\src\ggml.vcxproj]
          with
          [
              Dequantizer=`anonymous-namespace'::DequantizerIQ2KS
          ]
C:\Textgen\ik_llama.cpp\ggml\src\iqk\iqk_gemm_iqk_quants.cpp(2070,13): message : see reference to function template ins
tantiation 'void `anonymous-namespace'::set_functions<`anonymous-namespace'::DequantizerIQ2KS>(std::array<mul_mat_t,8>
&)' being compiled [C:\Textgen\ik_llama.cpp\build\ggml\src\ggml.vcxproj]
  iqk_gemm_1bit.cpp
  iqk_gemm_legacy_quants.cpp
  iqk_quantize.cpp
  Generating Code...

C:\Textgen\ik_llama.cpp>
```

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-05-23** at **12:10:01**:<br>

Does #448 fix it?

---

👤 **quasar-of-mikus** commented the **2025-05-23** at **12:30:39**:<br>

Yep, it compiles and runs fine with that PR. Don't know if this is related but I saw this message come up even though it built:
```
C:\Textgen\ik_llama.cpp\examples\quantize-stats\quantize-stats.cpp(554,1): error C3493: 'kBlockSize' cannot be implicit
ly captured because no default capture mode has been specified [C:\Textgen\ik_llama.cpp\build\examples\quantize-stats\l
lama-quantize-stats.vcxproj]
C:\Textgen\ik_llama.cpp\examples\quantize-stats\quantize-stats.cpp(555,1): error C3493: 'kGroupSize' cannot be implicit
ly captured because no default capture mode has been specified [C:\Textgen\ik_llama.cpp\build\examples\quantize-stats\l
lama-quantize-stats.vcxproj]
C:\Textgen\ik_llama.cpp\examples\quantize-stats\quantize-stats.cpp(679,1): error C3493: 'kNg' cannot be implicitly capt
ured because no default capture mode has been specified [C:\Textgen\ik_llama.cpp\build\examples\quantize-stats\llama-qu
antize-stats.vcxproj]
C:\Textgen\ik_llama.cpp\examples\quantize-stats\quantize-stats.cpp(694,5): error C2064: term does not evaluate to a fun
ction taking 0 arguments [C:\Textgen\ik_llama.cpp\build\examples\quantize-stats\llama-quantize-stats.vcxproj]
C:\Textgen\ik_llama.cpp\examples\quantize-stats\quantize-stats.cpp(722,1): error C3493: 'kBlockSize' cannot be implicit
ly captured because no default capture mode has been specified [C:\Textgen\ik_llama.cpp\build\examples\quantize-stats\l
lama-quantize-stats.vcxproj]
C:\Textgen\ik_llama.cpp\examples\quantize-stats\quantize-stats.cpp(777,1): error C3493: 'kNumVal' cannot be implicitly
captured because no default capture mode has been specified [C:\Textgen\ik_llama.cpp\build\examples\quantize-stats\llam
a-quantize-stats.vcxproj]
C:\Textgen\ik_llama.cpp\examples\quantize-stats\quantize-stats.cpp(821,5): error C2064: term does not evaluate to a fun
ction taking 0 arguments [C:\Textgen\ik_llama.cpp\build\examples\quantize-stats\llama-quantize-stats.vcxproj]
  llama-gguf.vcxproj -> C:\Textgen\ik_llama.cpp\build\bin\Release\llama-gguf.exe
  llama-gguf-hash.vcxproj -> C:\Textgen\ik_llama.cpp\build\bin\Release\llama-gguf-hash.exe
  llama-bench-matmult.vcxproj -> C:\Textgen\ik_llama.cpp\build\bin\Release\llama-bench-matmult.exe
```

---

👤 **ikawrakow** commented the **2025-05-23** at **12:56:37**:<br>

These are in the `quantize-stats` tool that fails to build (but everything else build correctly).
Somehow MSVC disagrees with GCC and clang on the scope of `constexpr`'s. Can you check if the commit I just pushed fixes it? Thanks.

---

👤 **quasar-of-mikus** commented the **2025-05-23** at **13:14:15**:<br>

No, on commit [f015390](https://github.com/ikawrakow/ik_llama.cpp/pull/448/commits/f015390efa54b21752e3a76c212c93614cfff7ca) I am still getting an error, same as last time minus an error for `kBlockSize`:
```
C:\Textgen\ik_llama.cpp\examples\quantize-stats\quantize-stats.cpp(555,1): error C3493: 'kGroupSize' cannot be implicit
ly captured because no default capture mode has been specified [C:\Textgen\ik_llama.cpp\build\examples\quantize-stats\l
lama-quantize-stats.vcxproj]
C:\Textgen\ik_llama.cpp\examples\quantize-stats\quantize-stats.cpp(678,1): error C3493: 'kNg' cannot be implicitly capt
ured because no default capture mode has been specified [C:\Textgen\ik_llama.cpp\build\examples\quantize-stats\llama-qu
antize-stats.vcxproj]
C:\Textgen\ik_llama.cpp\examples\quantize-stats\quantize-stats.cpp(693,5): error C2064: term does not evaluate to a fun
ction taking 0 arguments [C:\Textgen\ik_llama.cpp\build\examples\quantize-stats\llama-quantize-stats.vcxproj]
C:\Textgen\ik_llama.cpp\examples\quantize-stats\quantize-stats.cpp(780,1): error C3493: 'kNumVal' cannot be implicitly
captured because no default capture mode has been specified [C:\Textgen\ik_llama.cpp\build\examples\quantize-stats\llam
a-quantize-stats.vcxproj]
C:\Textgen\ik_llama.cpp\examples\quantize-stats\quantize-stats.cpp(824,5): error C2064: term does not evaluate to a fun
ction taking 0 arguments [C:\Textgen\ik_llama.cpp\build\examples\quantize-stats\llama-quantize-stats.vcxproj]
  llama-gguf.vcxproj -> C:\Textgen\ik_llama.cpp\build\bin\Release\llama-gguf.exe
  llama-gguf-hash.vcxproj -> C:\Textgen\ik_llama.cpp\build\bin\Release\llama-gguf-hash.exe
  llama-bench-matmult.vcxproj -> C:\Textgen\ik_llama.cpp\build\bin\Release\llama-bench-matmult.exe
```

---

👤 **quasar-of-mikus** commented the **2025-05-23** at **13:14:15**:<br>

No, I am still getting an error, same as last time minus an error for `kBlockSize`:
```
C:\Textgen\ik_llama.cpp\examples\quantize-stats\quantize-stats.cpp(555,1): error C3493: 'kGroupSize' cannot be implicit
ly captured because no default capture mode has been specified [C:\Textgen\ik_llama.cpp\build\examples\quantize-stats\l
lama-quantize-stats.vcxproj]
C:\Textgen\ik_llama.cpp\examples\quantize-stats\quantize-stats.cpp(678,1): error C3493: 'kNg' cannot be implicitly capt
ured because no default capture mode has been specified [C:\Textgen\ik_llama.cpp\build\examples\quantize-stats\llama-qu
antize-stats.vcxproj]
C:\Textgen\ik_llama.cpp\examples\quantize-stats\quantize-stats.cpp(693,5): error C2064: term does not evaluate to a fun
ction taking 0 arguments [C:\Textgen\ik_llama.cpp\build\examples\quantize-stats\llama-quantize-stats.vcxproj]
C:\Textgen\ik_llama.cpp\examples\quantize-stats\quantize-stats.cpp(780,1): error C3493: 'kNumVal' cannot be implicitly
captured because no default capture mode has been specified [C:\Textgen\ik_llama.cpp\build\examples\quantize-stats\llam
a-quantize-stats.vcxproj]
C:\Textgen\ik_llama.cpp\examples\quantize-stats\quantize-stats.cpp(824,5): error C2064: term does not evaluate to a fun
ction taking 0 arguments [C:\Textgen\ik_llama.cpp\build\examples\quantize-stats\llama-quantize-stats.vcxproj]
  llama-gguf.vcxproj -> C:\Textgen\ik_llama.cpp\build\bin\Release\llama-gguf.exe
  llama-gguf-hash.vcxproj -> C:\Textgen\ik_llama.cpp\build\bin\Release\llama-gguf-hash.exe
  llama-bench-matmult.vcxproj -> C:\Textgen\ik_llama.cpp\build\bin\Release\llama-bench-matmult.exe
```

---

👤 **ikawrakow** commented the **2025-05-23** at **13:29:23**:<br>

And now?

I never work on Windows, but from what I hear from `llama.cpp` users `clang` produces faster code than MSVC.

---

👤 **quasar-of-mikus** commented the **2025-05-23** at **13:44:54**:<br>

It works now, no more errors during compilation.
>from what I hear from llama.cpp users clang produces faster code than MSVC.

Cool, I'll compare with clang sometime