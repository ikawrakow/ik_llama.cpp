### 🐛 [#160](https://github.com/ikawrakow/ik_llama.cpp/issues/160) - Bug: Can't compile on MSVC 2022

| **Author** | `Nexesenex` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-12-22 |
| **Updated** | 2024-12-23 |

---

#### Description

### What happened?

Screenshot and log below.                                                                   

### Name and Version

PR 158 merged (sunday 22/12/2024 at 3PM).
Main branch, no modification.

![2024-12-22 15_13_31-ik_llama cpp fks - Microsoft Visual Studio](https://github.com/user-attachments/assets/b4ed5da5-b702-468d-acb0-feefac558fac)

### What operating system are you seeing the problem on?

Windows 11

### Relevant log output

```shell
>------ Build All started: Project: ik_llama.cpp, Configuration: x64-Release-MMQ ------
  [1/135] Building C object tests\CMakeFiles\test-c.dir\test-c.c.obj
  [2/135] Building C object ggml\src\CMakeFiles\ggml.dir\ggml-aarch64.c.obj
  [3/135] Generating build details from Git
  -- Found Git: C:/Program Files/Git/cmd/git.exe (found version "2.47.0.windows.2")
  [4/135] Building CXX object common\CMakeFiles\build_info.dir\build-info.cpp.obj
  [5/135] Building CXX object ggml\src\CMakeFiles\ggml.dir\iqk\iqk_quantize.cpp.obj
  FAILED: ggml/src/CMakeFiles/ggml.dir/iqk/iqk_quantize.cpp.obj 
  P:\PROGRA~1\MICROS~1\2022\COMMUN~1\VC\Tools\MSVC\1442~1.344\bin\Hostx64\x64\cl.exe  /nologo /TP -DGGML_BUILD -DGGML_CUDA_DMMV_X=32 -DGGML_CUDA_F16 -DGGML_CUDA_FORCE_MMQ -DGGML_CUDA_MMV_Y=1 -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 -DGGML_CUDA_USE_GRAPHS -DGGML_SCHED_MAX_COPIES=1 -DGGML_SHARED -DGGML_USE_CUDA -DGGML_USE_IQK_MULMAT -DGGML_USE_LLAMAFILE -DGGML_USE_OPENMP -DK_QUANTS_PER_ITERATION=2 -D_CRT_SECURE_NO_WARNINGS -D_XOPEN_SOURCE=600 -Dggml_EXPORTS -IQ:\GitHub\ik_llama.cpp.fks\ggml\src\..\include -IQ:\GitHub\ik_llama.cpp.fks\ggml\src\. -external:IP:\NVIDIAGPUCT\CUDA\v12.6\include -external:W0 /DWIN32 /D_WINDOWS /W3 /GR /EHsc /MD /O2 /Ob2 /DNDEBUG -std:c++17 /arch:AVX2 -openmp /showIncludes /Foggml\src\CMakeFiles\ggml.dir\iqk\iqk_quantize.cpp.obj /Fdggml\src\CMakeFiles\ggml.dir\ /FS -c Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_quantize.cpp
Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_quantize.cpp(5752): error C3493: 'kChunk' cannot be implicitly captured because no default capture mode has been specified
Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_quantize.cpp(5762): error C2064: term does not evaluate to a function taking 0 arguments
  [6/135] Building CXX object examples\gguf\CMakeFiles\llama-gguf.dir\gguf.cpp.obj
Q:\GitHub\ik_llama.cpp.fks\examples\gguf\gguf.cpp(69): warning C4244: '=': conversion from 'int' to 'float', possible loss of data
  [7/135] Building CXX object examples\gguf-hash\CMakeFiles\llama-gguf-hash.dir\gguf-hash.cpp.obj
Q:\GitHub\ik_llama.cpp.fks\examples\gguf-hash\gguf-hash.cpp(383): warning C4267: 'argument': conversion from 'size_t' to 'uint32_t', possible loss of data
Q:\GitHub\ik_llama.cpp.fks\examples\gguf-hash\gguf-hash.cpp(412): warning C4267: 'argument': conversion from 'size_t' to 'uint32_t', possible loss of data
Q:\GitHub\ik_llama.cpp.fks\examples\gguf-hash\gguf-hash.cpp(453): warning C4267: 'argument': conversion from 'size_t' to 'uint32_t', possible loss of data
  [8/135] Building CXX object ggml\src\CMakeFiles\ggml.dir\iqk\iqk_mul_mat.cpp.obj
  FAILED: ggml/src/CMakeFiles/ggml.dir/iqk/iqk_mul_mat.cpp.obj 
  P:\PROGRA~1\MICROS~1\2022\COMMUN~1\VC\Tools\MSVC\1442~1.344\bin\Hostx64\x64\cl.exe  /nologo /TP -DGGML_BUILD -DGGML_CUDA_DMMV_X=32 -DGGML_CUDA_F16 -DGGML_CUDA_FORCE_MMQ -DGGML_CUDA_MMV_Y=1 -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 -DGGML_CUDA_USE_GRAPHS -DGGML_SCHED_MAX_COPIES=1 -DGGML_SHARED -DGGML_USE_CUDA -DGGML_USE_IQK_MULMAT -DGGML_USE_LLAMAFILE -DGGML_USE_OPENMP -DK_QUANTS_PER_ITERATION=2 -D_CRT_SECURE_NO_WARNINGS -D_XOPEN_SOURCE=600 -Dggml_EXPORTS -IQ:\GitHub\ik_llama.cpp.fks\ggml\src\..\include -IQ:\GitHub\ik_llama.cpp.fks\ggml\src\. -external:IP:\NVIDIAGPUCT\CUDA\v12.6\include -external:W0 /DWIN32 /D_WINDOWS /W3 /GR /EHsc /MD /O2 /Ob2 /DNDEBUG -std:c++17 /arch:AVX2 -openmp /showIncludes /Foggml\src\CMakeFiles\ggml.dir\iqk\iqk_mul_mat.cpp.obj /Fdggml\src\CMakeFiles\ggml.dir\ /FS -c Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp
Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(143): warning C4267: 'initializing': conversion from 'size_t' to 'int', possible loss of data
Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(1826): warning C4309: 'argument': truncation of constant value
Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(1922): warning C4309: 'argument': truncation of constant value
Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(7159): warning C4065: switch statement contains 'default' but no 'case' labels
Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(7170): warning C4065: switch statement contains 'default' but no 'case' labels
Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2612): error C2676: binary '|': '__m256i' does not define this operator or a conversion to a type acceptable to the predefined operator
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2612): note: the template instantiation context (the oldest one first) is
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(7580): note: see reference to function template instantiation 'void `anonymous-namespace'::mul_mat_q5_0_r4_q8_1<1>(int,const void *,size_t,const `anonymous-namespace'::DataInfo &,int)' being compiled
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2704): note: see reference to function template instantiation 'void `anonymous-namespace'::mul_mat_q5_0_r4_q8_1_avx2<1>(int,const void *,size_t,const `anonymous-namespace'::DataInfo &,int)' being compiled
Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2613): error C2676: binary '|': '__m256i' does not define this operator or a conversion to a type acceptable to the predefined operator
Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2614): error C2676: binary '|': '__m256i' does not define this operator or a conversion to a type acceptable to the predefined operator
Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2615): error C2676: binary '|': '__m256i' does not define this operator or a conversion to a type acceptable to the predefined operator
Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2618): error C3536: 'q1': cannot be used before it is initialized
Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2618): error C2664: '__m256i _mm256_maddubs_epi16(__m256i,__m256i)': cannot convert argument 1 from 'int' to '__m256i'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2618): note: '__m256i::__m256i': no overloaded function could convert all the argument types
  P:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\include\immintrin.h(56): note: could be '__m256i::__m256i(__m256i &&)'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2618): note: '__m256i::__m256i(__m256i &&)': cannot convert argument 1 from 'int' to '__m256i &&'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2618): note: Reason: cannot convert from 'int' to '__m256i'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2618): note: Conversion requires a second user-defined-conversion operator or constructor
  P:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\include\immintrin.h(56): note: or       '__m256i::__m256i(const __m256i &)'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2618): note: '__m256i::__m256i(const __m256i &)': cannot convert argument 1 from 'int' to 'const __m256i &'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2618): note: Reason: cannot convert from 'int' to 'const __m256i'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2618): note: Conversion requires a second user-defined-conversion operator or constructor
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2618): note: while trying to match the argument list '(int)'
  P:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\include\immintrin.h(1548): note: see declaration of '_mm256_maddubs_epi16'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2618): note: while trying to match the argument list '(int, __m256i)'
Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2619): error C3536: 'q2': cannot be used before it is initialized
Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2619): error C2664: '__m256i _mm256_maddubs_epi16(__m256i,__m256i)': cannot convert argument 1 from 'int' to '__m256i'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2619): note: '__m256i::__m256i': no overloaded function could convert all the argument types
  P:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\include\immintrin.h(56): note: could be '__m256i::__m256i(__m256i &&)'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2619): note: '__m256i::__m256i(__m256i &&)': cannot convert argument 1 from 'int' to '__m256i &&'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2619): note: Reason: cannot convert from 'int' to '__m256i'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2619): note: Conversion requires a second user-defined-conversion operator or constructor
  P:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\include\immintrin.h(56): note: or       '__m256i::__m256i(const __m256i &)'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2619): note: '__m256i::__m256i(const __m256i &)': cannot convert argument 1 from 'int' to 'const __m256i &'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2619): note: Reason: cannot convert from 'int' to 'const __m256i'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2619): note: Conversion requires a second user-defined-conversion operator or constructor
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2619): note: while trying to match the argument list '(int)'
  P:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\include\immintrin.h(1548): note: see declaration of '_mm256_maddubs_epi16'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2619): note: while trying to match the argument list '(int, __m256i)'
Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2620): error C3536: 'q3': cannot be used before it is initialized
Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2620): error C2664: '__m256i _mm256_maddubs_epi16(__m256i,__m256i)': cannot convert argument 1 from 'int' to '__m256i'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2620): note: '__m256i::__m256i': no overloaded function could convert all the argument types
  P:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\include\immintrin.h(56): note: could be '__m256i::__m256i(__m256i &&)'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2620): note: '__m256i::__m256i(__m256i &&)': cannot convert argument 1 from 'int' to '__m256i &&'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2620): note: Reason: cannot convert from 'int' to '__m256i'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2620): note: Conversion requires a second user-defined-conversion operator or constructor
  P:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\include\immintrin.h(56): note: or       '__m256i::__m256i(const __m256i &)'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2620): note: '__m256i::__m256i(const __m256i &)': cannot convert argument 1 from 'int' to 'const __m256i &'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2620): note: Reason: cannot convert from 'int' to 'const __m256i'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2620): note: Conversion requires a second user-defined-conversion operator or constructor
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2620): note: while trying to match the argument list '(int)'
  P:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\include\immintrin.h(1548): note: see declaration of '_mm256_maddubs_epi16'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2620): note: while trying to match the argument list '(int, __m256i)'
Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2621): error C3536: 'q4': cannot be used before it is initialized
Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2621): error C2664: '__m256i _mm256_maddubs_epi16(__m256i,__m256i)': cannot convert argument 1 from 'int' to '__m256i'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2621): note: '__m256i::__m256i': no overloaded function could convert all the argument types
  P:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\include\immintrin.h(56): note: could be '__m256i::__m256i(__m256i &&)'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2621): note: '__m256i::__m256i(__m256i &&)': cannot convert argument 1 from 'int' to '__m256i &&'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2621): note: Reason: cannot convert from 'int' to '__m256i'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2621): note: Conversion requires a second user-defined-conversion operator or constructor
  P:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\include\immintrin.h(56): note: or       '__m256i::__m256i(const __m256i &)'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2621): note: '__m256i::__m256i(const __m256i &)': cannot convert argument 1 from 'int' to 'const __m256i &'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2621): note: Reason: cannot convert from 'int' to 'const __m256i'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2621): note: Conversion requires a second user-defined-conversion operator or constructor
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2621): note: while trying to match the argument list '(int)'
  P:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\include\immintrin.h(1548): note: see declaration of '_mm256_maddubs_epi16'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2621): note: while trying to match the argument list '(int, __m256i)'
Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2622): error C3536: 'sumi1': cannot be used before it is initialized
Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2622): error C3536: 'sumi2': cannot be used before it is initialized
Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2622): error C2664: '__m256i _mm256_add_epi16(__m256i,__m256i)': cannot convert argument 1 from 'int' to '__m256i'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2622): note: '__m256i::__m256i': no overloaded function could convert all the argument types
  P:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\include\immintrin.h(56): note: could be '__m256i::__m256i(__m256i &&)'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2622): note: '__m256i::__m256i(__m256i &&)': cannot convert argument 1 from 'int' to '__m256i &&'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2622): note: Reason: cannot convert from 'int' to '__m256i'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2622): note: Conversion requires a second user-defined-conversion operator or constructor
  P:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\include\immintrin.h(56): note: or       '__m256i::__m256i(const __m256i &)'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2622): note: '__m256i::__m256i(const __m256i &)': cannot convert argument 1 from 'int' to 'const __m256i &'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2622): note: Reason: cannot convert from 'int' to 'const __m256i'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2622): note: Conversion requires a second user-defined-conversion operator or constructor
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2622): note: while trying to match the argument list '(int)'
  P:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\include\immintrin.h(1517): note: see declaration of '_mm256_add_epi16'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2622): note: while trying to match the argument list '(int, int)'
Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2624): error C3536: 'sumi': cannot be used before it is initialized
Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2624): error C2664: '__m256 _mm256_cvtepi32_ps(__m256i)': cannot convert argument 1 from 'int' to '__m256i'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2624): note: '__m256i::__m256i': no overloaded function could convert all the argument types
  P:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\include\immintrin.h(56): note: could be '__m256i::__m256i(__m256i &&)'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2624): note: '__m256i::__m256i(__m256i &&)': cannot convert argument 1 from 'int' to '__m256i &&'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2624): note: Reason: cannot convert from 'int' to '__m256i'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2624): note: Conversion requires a second user-defined-conversion operator or constructor
  P:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\include\immintrin.h(56): note: or       '__m256i::__m256i(const __m256i &)'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2624): note: '__m256i::__m256i(const __m256i &)': cannot convert argument 1 from 'int' to 'const __m256i &'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2624): note: Reason: cannot convert from 'int' to 'const __m256i'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2624): note: Conversion requires a second user-defined-conversion operator or constructor
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2624): note: while trying to match the argument list '(int)'
  P:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\include\immintrin.h(574): note: see declaration of '_mm256_cvtepi32_ps'
  Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2624): note: while trying to match the argument list '(int)'
Q:\GitHub\ik_llama.cpp.fks\ggml\src\iqk\iqk_mul_mat.cpp(2624): fatal error C1003: error count exceeds 100; stopping compilation
  [9/135] Building CXX object src\CMakeFiles\llama.dir\llama-sampling.cpp.obj
Q:\GitHub\ik_llama.cpp.fks\src\llama-sampling.cpp(26): warning C4244: '=': conversion from 'time_t' to 'uint32_t', possible loss of data
Q:\GitHub\ik_llama.cpp.fks\src\llama-sampling.cpp(70): warning C4267: '=': conversion from 'size_t' to 'int32_t', possible loss of data
Q:\GitHub\ik_llama.cpp.fks\src\llama-sampling.cpp(405): warning C4244: '=': conversion from 'double' to 'float', possible loss of data
Q:\GitHub\ik_llama.cpp.fks\src\llama-sampling.cpp(409): warning C4244: '/=': conversion from 'double' to 'float', possible loss of data
Q:\GitHub\ik_llama.cpp.fks\src\llama-sampling.cpp(510): warning C4244: 'initializing': conversion from 'float' to 'int32_t', possible loss of data
Q:\GitHub\ik_llama.cpp.fks\src\llama-sampling.cpp(510): warning C4244: 'initializing': conversion from 'float' to 'const int32_t', possible loss of data
Q:\GitHub\ik_llama.cpp.fks\src\llama-sampling.cpp(530): warning C4244: 'argument': conversion from 'const int32_t' to 'float', possible loss of data
  [10/135] Building CXX object src\CMakeFiles\llama.dir\llama-grammar.cpp.obj
  [11/135] Building CXX object src\CMakeFiles\llama.dir\llama-vocab.cpp.obj
Q:\GitHub\ik_llama.cpp.fks\src\llama-vocab.cpp(138): warning C4244: 'return': conversion from 'long' to 'uint8_t', possible loss of data
Q:\GitHub\ik_llama.cpp.fks\src\llama-vocab.cpp(211): warning C4267: 'argument': conversion from 'size_t' to 'int', possible loss of data
Q:\GitHub\ik_llama.cpp.fks\src\llama-vocab.cpp(211): warning C4267: 'argument': conversion from 'size_t' to 'int', possible loss of data
Q:\GitHub\ik_llama.cpp.fks\src\llama-vocab.cpp(515): warning C4267: 'argument': conversion from 'size_t' to 'int', possible loss of data
Q:\GitHub\ik_llama.cpp.fks\src\llama-vocab.cpp(515): warning C4267: 'argument': conversion from 'size_t' to 'int', possible loss of data
Q:\GitHub\ik_llama.cpp.fks\src\llama-vocab.cpp(555): warning C4267: '=': conversion from 'size_t' to 'llm_symbol::index', possible loss of data
Q:\GitHub\ik_llama.cpp.fks\src\llama-vocab.cpp(558): warning C4267: '=': conversion from 'size_t' to 'int', possible loss of data
Q:\GitHub\ik_llama.cpp.fks\src\llama-vocab.cpp(652): warning C4267: 'initializing': conversion from 'size_t' to 'int', possible loss of data
Q:\GitHub\ik_llama.cpp.fks\src\llama-vocab.cpp(652): warning C4267: 'initializing': conversion from 'size_t' to 'const int', possible loss of data
Q:\GitHub\ik_llama.cpp.fks\src\llama-vocab.cpp(1515): warning C4267: 'return': conversion from 'size_t' to 'int32_t', possible loss of data
  [12/135] Building CXX object examples\llava\CMakeFiles\llava.dir\llava.cpp.obj
Q:\GitHub\ik_llama.cpp.fks\examples\llava\llava.cpp(346): warning C4244: 'initializing': conversion from 'double' to 'float', possible loss of data
  [13/135] Building CXX object src\CMakeFiles\llama.dir\unicode.cpp.obj
  [14/135] Building CXX object common\CMakeFiles\common.dir\common.cpp.obj
  [15/135] Building CXX object src\CMakeFiles\llama.dir\unicode-data.cpp.obj
```