### 📝 [#380](https://github.com/ikawrakow/ik_llama.cpp/issues/380) - Drop at the start of generation

| **Author** | `intulint` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-04 |
| **Updated** | 2025-05-25 |

---

#### Description

After the generation starts, the server crashes. This only happens on the Qwen3-30B-A3B, and I checked different quant. Regular dense models work, including other dense qwen3. 
What could be the problem? I liked the acceleration in dense models, I thought moe would fly. 
But it doesn't work. It crashes without an error, it just goes to the command line when generation starts.

win10, Microsoft Visual Studio\2022, main branch

cmake -B ./build -DGGML_CUDA=OFF -DGGML_BLAS=OFF
cmake --build ./build --config Release -j 16

./llama-server.exe -t 7 -c 4096 -m F:\llm\Qwen3-30B-A3B-Q5_K_M.gguf

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-05-05** at **05:12:28**:<br>

Can you post the output of the above commands (including the `cmake` commands)? Thanks.

---

👤 **intulint** commented the **2025-05-05** at **10:10:19**:<br>

Sure, but it turned out to be a lot of text. I also noticed that it takes a long time to assemble in a single thread of unicode.cpp
  unicode-data.cpp. I don't know if this is normal or not.
From a third-party frontend, generation does not occur at all and the program exits. If you connect from the native server, then about 140 tokens are generated and again it crashes without messages.


**********************************************************************
** Visual Studio 2022 Developer Command Prompt v17.13.6
** Copyright (c) 2022 Microsoft Corporation
**********************************************************************

C:\Program Files\Microsoft Visual Studio\2022\Community>cd C:\neuro\ik_llama.cpp

C:\neuro\ik_llama.cpp>git pull
Already up to date.

C:\neuro\ik_llama.cpp>cmake -B ./build -DGGML_CUDA=OFF -DGGML_BLAS=OFF
-- Building for: Visual Studio 17 2022
-- Selecting Windows SDK version 10.0.20348.0 to target Windows 10.0.19045.
-- The C compiler identification is MSVC 19.43.34810.0
-- The CXX compiler identification is MSVC 19.43.34810.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.43.34808/bin/Hostx64/x64/cl.exe - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.43.34808/bin/Hostx64/x64/cl.exe - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Found Git: C:/Program Files/Git/cmd/git.exe (found version "2.47.1.windows.2")
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Failed
-- Looking for pthread_create in pthreads
-- Looking for pthread_create in pthreads - not found
-- Looking for pthread_create in pthread
-- Looking for pthread_create in pthread - not found
-- Found Threads: TRUE
-- Found OpenMP_C: -openmp (found version "2.0")
-- Found OpenMP_CXX: -openmp (found version "2.0")
-- Found OpenMP: TRUE (found version "2.0")
-- OpenMP found
-- Using optimized iqk matrix multiplications
-- Using llamafile
-- ccache found, compilation results will be cached. Disable with GGML_CCACHE=OFF.
-- CMAKE_SYSTEM_PROCESSOR: AMD64
-- CMAKE_GENERATOR_PLATFORM:
-- x86 detected
-- Performing Test HAS_AVX_1
-- Performing Test HAS_AVX_1 - Success
-- Performing Test HAS_AVX2_1
-- Performing Test HAS_AVX2_1 - Success
-- Performing Test HAS_FMA_1
-- Performing Test HAS_FMA_1 - Success
-- Performing Test HAS_AVX512_1
-- Performing Test HAS_AVX512_1 - Failed
-- Performing Test HAS_AVX512_2
-- Performing Test HAS_AVX512_2 - Failed
-- Configuring done (24.9s)
-- Generating done (1.9s)
-- Build files have been written to: C:/neuro/ik_llama.cpp/build

C:\neuro\ik_llama.cpp>cmake --build ./build --config Release -j 16
Версия MSBuild 17.13.19+0d9f5a35a для .NET Framework

  1>Checking Build System
  Building Custom Rule C:/neuro/ik_llama.cpp/examples/gguf-hash/CMakeLists.txt
  Building Custom Rule C:/neuro/ik_llama.cpp/examples/gguf-hash/CMakeLists.txt
  Generating build details from Git
  Building Custom Rule C:/neuro/ik_llama.cpp/ggml/src/CMakeLists.txt
  Building Custom Rule C:/neuro/ik_llama.cpp/examples/gguf-hash/CMakeLists.txt
  -- Found Git: C:/Program Files/Git/cmd/git.exe (found version "2.47.1.windows.2")
  sha1.c
  xxhash.c
  sha256.c
  ggml.c
  Building Custom Rule C:/neuro/ik_llama.cpp/common/CMakeLists.txt
  build-info.cpp
  ggml-alloc.c
  sha1.vcxproj -> C:\neuro\ik_llama.cpp\build\examples\gguf-hash\sha1.dir\Release\sha1.lib
  build_info.vcxproj -> C:\neuro\ik_llama.cpp\build\common\build_info.dir\Release\build_info.lib
  sha256.vcxproj -> C:\neuro\ik_llama.cpp\build\examples\gguf-hash\sha256.dir\Release\sha256.lib
  ggml-backend.c
  xxhash.vcxproj -> C:\neuro\ik_llama.cpp\build\examples\gguf-hash\xxhash.dir\Release\xxhash.lib
  ggml-quants.c
C:\Program Files (x86)\Windows Kits\10\Include\10.0.20348.0\ucrt\assert.h(21,9): warning C4005: 'static_assert': mac
ro redefinition [C:\neuro\ik_llama.cpp\build\ggml\src\ggml.vcxproj]
  (compiling source file '../../../ggml/src/ggml-quants.c')
      C:\neuro\ik_llama.cpp\ggml\src\ggml-common.h(69,9):
      see previous definition of 'static_assert'

  ggml-aarch64.c
C:\Program Files (x86)\Windows Kits\10\Include\10.0.20348.0\ucrt\assert.h(21,9): warning C4005: 'static_assert': mac
ro redefinition [C:\neuro\ik_llama.cpp\build\ggml\src\ggml.vcxproj]
  (compiling source file '../../../ggml/src/ggml-aarch64.c')
      C:\neuro\ik_llama.cpp\ggml\src\ggml-common.h(69,9):
      see previous definition of 'static_assert'

  Generating Code...
  sgemm.cpp
  iqk_mul_mat.cpp
C:\neuro\ik_llama.cpp\ggml\src\iqk\iqk_mul_mat.cpp(177,16): warning C4267: 'initializing': conversion from 'size_t'
to 'int', possible loss of data [C:\neuro\ik_llama.cpp\build\ggml\src\ggml.vcxproj]
C:\neuro\ik_llama.cpp\ggml\src\iqk\iqk_mul_mat.cpp(260,16): warning C4267: 'initializing': conversion from 'size_t'
to 'int', possible loss of data [C:\neuro\ik_llama.cpp\build\ggml\src\ggml.vcxproj]
C:\neuro\ik_llama.cpp\ggml\src\iqk\iqk_mul_mat.cpp(9584,9): warning C4065: switch statement contains 'default' but n
o 'case' labels [C:\neuro\ik_llama.cpp\build\ggml\src\ggml.vcxproj]
C:\neuro\ik_llama.cpp\ggml\src\iqk\iqk_mul_mat.cpp(3049,84): warning C4244: 'argument': conversion from 'const uint1
6_t' to 'char', possible loss of data [C:\neuro\ik_llama.cpp\build\ggml\src\ggml.vcxproj]
      C:\neuro\ik_llama.cpp\ggml\src\iqk\iqk_mul_mat.cpp(3049,84):
      the template instantiation context (the oldest one first) is
          C:\neuro\ik_llama.cpp\ggml\src\iqk\iqk_mul_mat.cpp(9649,21):
          see reference to function template instantiation 'void `anonymous-namespace'::MulMat::set_functions<`anony
  mous-namespace'::DequantizerIQ2KS>(`anonymous-namespace'::MulMat &)' being compiled
          C:\neuro\ik_llama.cpp\ggml\src\iqk\iqk_mul_mat.cpp(9511,30):
          see reference to function template instantiation 'void `anonymous-namespace'::mul_mat_qX_K_q8_K_T<Dequanti
  zer,1>(int,const void *,size_t,const `anonymous-namespace'::DataInfo &,int)' being compiled
          with
          [
              Dequantizer=`anonymous-namespace'::DequantizerIQ2KS
          ]
          C:\neuro\ik_llama.cpp\ggml\src\iqk\iqk_mul_mat.cpp(3240,35):
          see reference to function template instantiation '__m256i `anonymous-namespace'::DequantizerIQ2KS::new_blo
  ck<`anonymous-namespace'::Q8<1,block_q8_K>>(int,const Q8 &,__m256 *)' being compiled
          with
          [
              Q8=`anonymous-namespace'::Q8<1,block_q8_K>
          ]

  iqk_flash_attn.cpp
C:\neuro\ik_llama.cpp\ggml\src\iqk\iqk_flash_attn.cpp(88,24): warning C4244: '=': conversion from 'uint64_t' to 'int
', possible loss of data [C:\neuro\ik_llama.cpp\build\ggml\src\ggml.vcxproj]
  iqk_quantize.cpp
  Generating Code...
  Auto build dll exports
     Creating library C:/neuro/ik_llama.cpp/build/ggml/src/Release/ggml.lib and object C:/neuro/ik_llama.cpp/build/g
  gml/src/Release/ggml.exp
  ggml.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\ggml.dll
  Building Custom Rule C:/neuro/ik_llama.cpp/src/CMakeLists.txt
  llama.cpp
C:\neuro\ik_llama.cpp\src\llama.cpp(2635,40): warning C4305: 'initializing': truncation from 'double' to 'float' [C:
\neuro\ik_llama.cpp\build\src\llama.vcxproj]
C:\neuro\ik_llama.cpp\src\llama.cpp(5511,17): warning C4065: switch statement contains 'default' but no 'case' label
s [C:\neuro\ik_llama.cpp\build\src\llama.vcxproj]
C:\neuro\ik_llama.cpp\src\llama.cpp(5520,17): warning C4065: switch statement contains 'default' but no 'case' label
s [C:\neuro\ik_llama.cpp\build\src\llama.vcxproj]
C:\neuro\ik_llama.cpp\src\llama.cpp(8970,24): warning C4477: 'printf' : format string '%ld' requires an argument of
type 'long', but variadic argument 2 has type 'int64_t' [C:\neuro\ik_llama.cpp\build\src\llama.vcxproj]
      C:\neuro\ik_llama.cpp\src\llama.cpp(8970,24):
      consider using '%lld' in the format string
      C:\neuro\ik_llama.cpp\src\llama.cpp(8970,24):
      consider using '%Id' in the format string
      C:\neuro\ik_llama.cpp\src\llama.cpp(8970,24):
      consider using '%I64d' in the format string

C:\neuro\ik_llama.cpp\src\llama.cpp(8970,24): warning C4477: 'printf' : format string '%ld' requires an argument of
type 'long', but variadic argument 3 has type 'int64_t' [C:\neuro\ik_llama.cpp\build\src\llama.vcxproj]
      C:\neuro\ik_llama.cpp\src\llama.cpp(8970,24):
      consider using '%lld' in the format string
      C:\neuro\ik_llama.cpp\src\llama.cpp(8970,24):
      consider using '%Id' in the format string
      C:\neuro\ik_llama.cpp\src\llama.cpp(8970,24):
      consider using '%I64d' in the format string

C:\neuro\ik_llama.cpp\src\llama.cpp(8970,24): warning C4477: 'printf' : format string '%ld' requires an argument of
type 'long', but variadic argument 4 has type 'int64_t' [C:\neuro\ik_llama.cpp\build\src\llama.vcxproj]
      C:\neuro\ik_llama.cpp\src\llama.cpp(8970,24):
      consider using '%lld' in the format string
      C:\neuro\ik_llama.cpp\src\llama.cpp(8970,24):
      consider using '%Id' in the format string
      C:\neuro\ik_llama.cpp\src\llama.cpp(8970,24):
      consider using '%I64d' in the format string

  llama-vocab.cpp
C:\neuro\ik_llama.cpp\src\llama-vocab.cpp(138,26): warning C4244: 'return': conversion from 'long' to 'uint8_t', pos
sible loss of data [C:\neuro\ik_llama.cpp\build\src\llama.vcxproj]
C:\neuro\ik_llama.cpp\src\llama-vocab.cpp(211,35): warning C4267: 'argument': conversion from 'size_t' to 'int', pos
sible loss of data [C:\neuro\ik_llama.cpp\build\src\llama.vcxproj]
C:\neuro\ik_llama.cpp\src\llama-vocab.cpp(211,30): warning C4267: 'argument': conversion from 'size_t' to 'int', pos
sible loss of data [C:\neuro\ik_llama.cpp\build\src\llama.vcxproj]
C:\neuro\ik_llama.cpp\src\llama-vocab.cpp(543,39): warning C4267: 'argument': conversion from 'size_t' to 'int', pos
sible loss of data [C:\neuro\ik_llama.cpp\build\src\llama.vcxproj]
C:\neuro\ik_llama.cpp\src\llama-vocab.cpp(543,34): warning C4267: 'argument': conversion from 'size_t' to 'int', pos
sible loss of data [C:\neuro\ik_llama.cpp\build\src\llama.vcxproj]
C:\neuro\ik_llama.cpp\src\llama-vocab.cpp(583,82): warning C4267: '=': conversion from 'size_t' to 'llm_symbol::inde
x', possible loss of data [C:\neuro\ik_llama.cpp\build\src\llama.vcxproj]
C:\neuro\ik_llama.cpp\src\llama-vocab.cpp(586,61): warning C4267: '=': conversion from 'size_t' to 'int', possible l
oss of data [C:\neuro\ik_llama.cpp\build\src\llama.vcxproj]
C:\neuro\ik_llama.cpp\src\llama-vocab.cpp(680,37): warning C4267: 'initializing': conversion from 'size_t' to 'int',
 possible loss of data [C:\neuro\ik_llama.cpp\build\src\llama.vcxproj]
C:\neuro\ik_llama.cpp\src\llama-vocab.cpp(680,25): warning C4267: 'initializing': conversion from 'size_t' to 'const
 int', possible loss of data [C:\neuro\ik_llama.cpp\build\src\llama.vcxproj]
C:\neuro\ik_llama.cpp\src\llama-vocab.cpp(1543,20): warning C4267: 'return': conversion from 'size_t' to 'int32_t',
possible loss of data [C:\neuro\ik_llama.cpp\build\src\llama.vcxproj]
  llama-grammar.cpp
  llama-sampling.cpp
C:\neuro\ik_llama.cpp\src\llama-sampling.cpp(26,20): warning C4244: '=': conversion from 'time_t' to 'uint32_t', pos
sible loss of data [C:\neuro\ik_llama.cpp\build\src\llama.vcxproj]
C:\neuro\ik_llama.cpp\src\llama-sampling.cpp(70,23): warning C4267: '=': conversion from 'size_t' to 'int32_t', poss
ible loss of data [C:\neuro\ik_llama.cpp\build\src\llama.vcxproj]
C:\neuro\ik_llama.cpp\src\llama-sampling.cpp(405,33): warning C4244: '=': conversion from 'double' to 'float', possi
ble loss of data [C:\neuro\ik_llama.cpp\build\src\llama.vcxproj]
C:\neuro\ik_llama.cpp\src\llama-sampling.cpp(409,34): warning C4244: '/=': conversion from 'double' to 'float', poss
ible loss of data [C:\neuro\ik_llama.cpp\build\src\llama.vcxproj]
C:\neuro\ik_llama.cpp\src\llama-sampling.cpp(510,34): warning C4244: 'initializing': conversion from 'float' to 'int
32_t', possible loss of data [C:\neuro\ik_llama.cpp\build\src\llama.vcxproj]
C:\neuro\ik_llama.cpp\src\llama-sampling.cpp(510,27): warning C4244: 'initializing': conversion from 'float' to 'con
st int32_t', possible loss of data [C:\neuro\ik_llama.cpp\build\src\llama.vcxproj]
C:\neuro\ik_llama.cpp\src\llama-sampling.cpp(530,61): warning C4244: 'argument': conversion from 'const int32_t' to
'float', possible loss of data [C:\neuro\ik_llama.cpp\build\src\llama.vcxproj]
  unicode.cpp
  unicode-data.cpp
  Generating Code...
  Auto build dll exports
     Creating library C:/neuro/ik_llama.cpp/build/src/Release/llama.lib and object C:/neuro/ik_llama.cpp/build/src/R
  elease/llama.exp
  llama.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\llama.dll
  Building Custom Rule C:/neuro/ik_llama.cpp/examples/llava/CMakeLists.txt
  llava.cpp
C:\neuro\ik_llama.cpp\examples\llava\llava.cpp(346,24): warning C4244: 'initializing': conversion from 'double' to '
float', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
  clip.cpp
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(590,32): warning C4267: 'initializing': conversion from 'size_t' to 'i
nt', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(590,26): warning C4267: 'initializing': conversion from 'size_t' to 'c
onst int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(824,149): warning C4244: 'argument': conversion from 'int64_t' to 'int
', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(824,130): warning C4244: 'argument': conversion from 'int64_t' to 'int
', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(824,111): warning C4244: 'argument': conversion from 'int64_t' to 'int
', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(824,92): warning C4244: 'argument': conversion from 'int64_t' to 'int'
, possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(838,23): warning C4244: 'initializing': conversion from 'int64_t' to '
int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(838,43): warning C4244: 'initializing': conversion from 'int64_t' to '
int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(872,149): warning C4244: 'argument': conversion from 'int64_t' to 'int
', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(872,130): warning C4244: 'argument': conversion from 'int64_t' to 'int
', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(872,111): warning C4244: 'argument': conversion from 'int64_t' to 'int
', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(872,92): warning C4244: 'argument': conversion from 'int64_t' to 'int'
, possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(887,23): warning C4244: 'initializing': conversion from 'int64_t' to '
int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(887,43): warning C4244: 'initializing': conversion from 'int64_t' to '
int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1218,27): warning C4267: 'initializing': conversion from 'size_t' to '
int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1010,9): warning C4297: 'clip_model_load': function assumed not to thr
ow an exception but does [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
      C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1010,9):
      __declspec(nothrow), throw(), noexcept(true), or noexcept was specified on the function

C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1429,13): warning C4297: 'clip_model_load': function assumed not to th
row an exception but does [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
      C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1429,13):
      __declspec(nothrow), throw(), noexcept(true), or noexcept was specified on the function

C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1529,48): warning C4267: 'argument': conversion from 'size_t' to 'int'
, possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1627,58): warning C4244: 'argument': conversion from 'int' to 'float',
 possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1627,46): warning C4244: 'argument': conversion from 'int' to 'float',
 possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1627,88): warning C4244: 'argument': conversion from 'int' to 'float',
 possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1627,77): warning C4244: 'argument': conversion from 'int' to 'float',
 possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1627,98): warning C4244: 'argument': conversion from 'float' to 'const
 unsigned __int64', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1627,137): warning C4244: 'argument': conversion from 'int' to 'float'
, possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1627,125): warning C4244: 'argument': conversion from 'int' to 'float'
, possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1627,163): warning C4244: 'argument': conversion from 'int' to 'float'
, possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1627,154): warning C4244: 'argument': conversion from 'int' to 'float'
, possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1627,173): warning C4244: 'argument': conversion from 'float' to 'cons
t unsigned __int64', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1627,103): warning C4244: '=': conversion from 'int' to 'float', possi
ble loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1628,58): warning C4244: 'argument': conversion from 'int' to 'float',
 possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1628,46): warning C4244: 'argument': conversion from 'int' to 'float',
 possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1628,88): warning C4244: 'argument': conversion from 'int' to 'float',
 possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1628,77): warning C4244: 'argument': conversion from 'int' to 'float',
 possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1628,98): warning C4244: 'argument': conversion from 'float' to 'const
 unsigned __int64', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1628,137): warning C4244: 'argument': conversion from 'int' to 'float'
, possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1628,125): warning C4244: 'argument': conversion from 'int' to 'float'
, possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1628,163): warning C4244: 'argument': conversion from 'int' to 'float'
, possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1628,154): warning C4244: 'argument': conversion from 'int' to 'float'
, possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1628,173): warning C4244: 'argument': conversion from 'float' to 'cons
t unsigned __int64', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1628,103): warning C4244: '=': conversion from 'int' to 'float', possi
ble loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1629,58): warning C4244: 'argument': conversion from 'int' to 'float',
 possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1629,46): warning C4244: 'argument': conversion from 'int' to 'float',
 possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1629,88): warning C4244: 'argument': conversion from 'int' to 'float',
 possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1629,77): warning C4244: 'argument': conversion from 'int' to 'float',
 possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1629,98): warning C4244: 'argument': conversion from 'float' to 'const
 unsigned __int64', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1629,137): warning C4244: 'argument': conversion from 'int' to 'float'
, possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1629,125): warning C4244: 'argument': conversion from 'int' to 'float'
, possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1629,163): warning C4244: 'argument': conversion from 'int' to 'float'
, possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1629,154): warning C4244: 'argument': conversion from 'int' to 'float'
, possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1629,173): warning C4244: 'argument': conversion from 'float' to 'cons
t unsigned __int64', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1629,103): warning C4244: '=': conversion from 'int' to 'float', possi
ble loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1630,58): warning C4244: 'argument': conversion from 'int' to 'float',
 possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1630,46): warning C4244: 'argument': conversion from 'int' to 'float',
 possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1630,84): warning C4244: 'argument': conversion from 'int' to 'float',
 possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1630,75): warning C4244: 'argument': conversion from 'int' to 'float',
 possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1630,94): warning C4244: 'argument': conversion from 'float' to 'const
 unsigned __int64', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1632,45): warning C4244: '=': conversion from 'double' to 'float', pos
sible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1633,40): warning C4244: '=': conversion from 'double' to 'float', pos
sible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1634,60): warning C4244: '=': conversion from 'double' to 'float', pos
sible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1642,45): warning C4244: '=': conversion from 'double' to 'float', pos
sible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1643,40): warning C4244: '=': conversion from 'double' to 'float', pos
sible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1644,60): warning C4244: '=': conversion from 'double' to 'float', pos
sible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1647,49): warning C4244: 'initializing': conversion from 'const _Ty' t
o 'uint8_t', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1647,49): warning C4244:         with [C:\neuro\ik_llama.cpp\build\exa
mples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1647,49): warning C4244:         [ [C:\neuro\ik_llama.cpp\build\exampl
es\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1647,49): warning C4244:             _Ty=float [C:\neuro\ik_llama.cpp\
build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1647,49): warning C4244:         ] [C:\neuro\ik_llama.cpp\build\exampl
es\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1647,39): warning C4244: 'initializing': conversion from 'const _Ty' t
o 'const uint8_t', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1647,39): warning C4244:         with [C:\neuro\ik_llama.cpp\build\exa
mples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1647,39): warning C4244:         [ [C:\neuro\ik_llama.cpp\build\exampl
es\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1647,39): warning C4244:             _Ty=float [C:\neuro\ik_llama.cpp\
build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1647,39): warning C4244:         ] [C:\neuro\ik_llama.cpp\build\exampl
es\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1648,68): warning C4244: '=': conversion from 'float' to '_Ty', possib
le loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1648,68): warning C4244:         with [C:\neuro\ik_llama.cpp\build\exa
mples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1648,68): warning C4244:         [ [C:\neuro\ik_llama.cpp\build\exampl
es\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1648,68): warning C4244:             _Ty=uint8_t [C:\neuro\ik_llama.cp
p\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1648,68): warning C4244:         ] [C:\neuro\ik_llama.cpp\build\exampl
es\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1821,21): warning C4244: 'initializing': conversion from 'double' to '
float', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1838,32): warning C4244: 'initializing': conversion from 'double' to '
float', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1838,27): warning C4244: 'initializing': conversion from 'double' to '
const float', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1839,63): warning C4244: 'initializing': conversion from 'double' to '
float', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1839,23): warning C4244: 'initializing': conversion from 'double' to '
const float', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1840,30): warning C4244: 'initializing': conversion from 'double' to '
int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1840,24): warning C4244: 'initializing': conversion from 'double' to '
const int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1903,32): warning C4244: 'initializing': conversion from 'double' to '
float', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1903,27): warning C4244: 'initializing': conversion from 'double' to '
const float', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1904,63): warning C4244: 'initializing': conversion from 'double' to '
float', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1904,23): warning C4244: 'initializing': conversion from 'double' to '
const float', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1905,30): warning C4244: 'initializing': conversion from 'double' to '
int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(1905,24): warning C4244: 'initializing': conversion from 'double' to '
const int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(2077,44): warning C4244: 'initializing': conversion from 'const _Ty' t
o 'uint8_t', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(2077,44): warning C4244:         with [C:\neuro\ik_llama.cpp\build\exa
mples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(2077,44): warning C4244:         [ [C:\neuro\ik_llama.cpp\build\exampl
es\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(2077,44): warning C4244:             _Ty=float [C:\neuro\ik_llama.cpp\
build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(2077,44): warning C4244:         ] [C:\neuro\ik_llama.cpp\build\exampl
es\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(2077,34): warning C4244: 'initializing': conversion from 'const _Ty' t
o 'const uint8_t', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(2077,34): warning C4244:         with [C:\neuro\ik_llama.cpp\build\exa
mples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(2077,34): warning C4244:         [ [C:\neuro\ik_llama.cpp\build\exampl
es\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(2077,34): warning C4244:             _Ty=float [C:\neuro\ik_llama.cpp\
build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(2077,34): warning C4244:         ] [C:\neuro\ik_llama.cpp\build\exampl
es\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(2157,11): warning C4267: 'initializing': conversion from 'size_t' to '
int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(2158,11): warning C4267: 'initializing': conversion from 'size_t' to '
int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(2162,24): warning C4244: '=': conversion from 'double' to '_Ty', possi
ble loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(2162,24): warning C4244:         with [C:\neuro\ik_llama.cpp\build\exa
mples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(2162,24): warning C4244:         [ [C:\neuro\ik_llama.cpp\build\exampl
es\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(2162,24): warning C4244:             _Ty=float [C:\neuro\ik_llama.cpp\
build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(2162,24): warning C4244:         ] [C:\neuro\ik_llama.cpp\build\exampl
es\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(2184,11): warning C4267: 'initializing': conversion from 'size_t' to '
int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(2185,11): warning C4267: 'initializing': conversion from 'size_t' to '
int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(2259,20): warning C4267: 'initializing': conversion from 'size_t' to '
int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(2320,47): warning C4244: '=': conversion from 'double' to 'int', possi
ble loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(2539,68): warning C4244: 'return': conversion from 'int64_t' to 'int',
 possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(2542,56): warning C4244: 'return': conversion from 'int64_t' to 'int',
 possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(2545,46): warning C4244: 'return': conversion from 'int64_t' to 'int',
 possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(2548,46): warning C4244: 'return': conversion from 'int64_t' to 'int',
 possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(2555,5): warning C4297: 'clip_n_mmproj_embd': function assumed not to
throw an exception but does [C:\neuro\ik_llama.cpp\build\examples\llava\llava.vcxproj]
      C:\neuro\ik_llama.cpp\examples\llava\clip.cpp(2555,5):
      __declspec(nothrow), throw(), noexcept(true), or noexcept was specified on the function

  Generating Code...
  llava.vcxproj -> C:\neuro\ik_llama.cpp\build\examples\llava\llava.dir\Release\llava.lib
  Building Custom Rule C:/neuro/ik_llama.cpp/common/CMakeLists.txt
  Building Custom Rule C:/neuro/ik_llama.cpp/examples/benchmark/CMakeLists.txt
  Building Custom Rule C:/neuro/ik_llama.cpp/examples/quantize-stats/CMakeLists.txt
  Building Custom Rule C:/neuro/ik_llama.cpp/examples/llava/CMakeLists.txt
  Building Custom Rule C:/neuro/ik_llama.cpp/examples/llava/CMakeLists.txt
  Building Custom Rule C:/neuro/ik_llama.cpp/examples/gguf/CMakeLists.txt
  Building Custom Rule C:/neuro/ik_llama.cpp/tests/CMakeLists.txt
  common.cpp
  benchmark-matmult.cpp
  gguf.cpp
  quantize-stats.cpp
     Creating library C:/neuro/ik_llama.cpp/build/examples/llava/Release/llava_shared.lib and object C:/neuro/ik_lla
  ma.cpp/build/examples/llava/Release/llava_shared.exp
  llava_static.vcxproj -> C:\neuro\ik_llama.cpp\build\examples\llava\Release\llava_static.lib
  test-c.c
  Building Custom Rule C:/neuro/ik_llama.cpp/examples/gguf-hash/CMakeLists.txt
  llava_shared.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\llava_shared.dll
C:\neuro\ik_llama.cpp\examples\gguf\gguf.cpp(69,31): warning C4244: '=': conversion from 'int' to 'float', possible
loss of data [C:\neuro\ik_llama.cpp\build\examples\gguf\llama-gguf.vcxproj]
  test-c.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\test-c.exe
  gguf-hash.cpp
  llama-gguf.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\llama-gguf.exe
  llama-bench-matmult.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\llama-bench-matmult.exe
C:\neuro\ik_llama.cpp\common\common.cpp(328,30): warning C4996: 'strdup': The POSIX name for this item is deprecated
. Instead, use the ISO C and C++ conformant name: _strdup. See online help for details. [C:\neuro\ik_llama.cpp\build
\common\common.vcxproj]
C:\neuro\ik_llama.cpp\examples\gguf-hash\gguf-hash.cpp(383,55): warning C4267: 'argument': conversion from 'size_t'
to 'uint32_t', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\gguf-hash\llama-gguf-hash.vcxproj]
C:\neuro\ik_llama.cpp\examples\gguf-hash\gguf-hash.cpp(412,80): warning C4267: 'argument': conversion from 'size_t'
to 'uint32_t', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\gguf-hash\llama-gguf-hash.vcxproj]
C:\neuro\ik_llama.cpp\examples\gguf-hash\gguf-hash.cpp(453,78): warning C4267: 'argument': conversion from 'size_t'
to 'uint32_t', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\gguf-hash\llama-gguf-hash.vcxproj]
  llama-gguf-hash.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\llama-gguf-hash.exe
  llama-quantize-stats.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\llama-quantize-stats.exe
  sampling.cpp
C:\neuro\ik_llama.cpp\common\sampling.cpp(105,45): warning C4267: 'initializing': conversion from 'size_t' to 'int',
 possible loss of data [C:\neuro\ik_llama.cpp\build\common\common.vcxproj]
C:\neuro\ik_llama.cpp\common\sampling.cpp(105,20): warning C4267: 'initializing': conversion from 'size_t' to 'const
 int', possible loss of data [C:\neuro\ik_llama.cpp\build\common\common.vcxproj]
  console.cpp
C:\neuro\ik_llama.cpp\common\console.cpp(253,30): warning C4267: 'initializing': conversion from 'size_t' to 'DWORD'
, possible loss of data [C:\neuro\ik_llama.cpp\build\common\common.vcxproj]
C:\neuro\ik_llama.cpp\common\console.cpp(407,28): warning C4267: 'initializing': conversion from 'size_t' to 'int',
possible loss of data [C:\neuro\ik_llama.cpp\build\common\common.vcxproj]
  grammar-parser.cpp
  json-schema-to-grammar.cpp
C:\neuro\ik_llama.cpp\common\json-schema-to-grammar.cpp(139,46): warning C4267: 'argument': conversion from 'size_t'
 to 'int', possible loss of data [C:\neuro\ik_llama.cpp\build\common\common.vcxproj]
C:\neuro\ik_llama.cpp\common\json-schema-to-grammar.cpp(139,37): warning C4267: 'argument': conversion from 'size_t'
 to 'int', possible loss of data [C:\neuro\ik_llama.cpp\build\common\common.vcxproj]
C:\neuro\ik_llama.cpp\common\json-schema-to-grammar.cpp(154,50): warning C4267: 'argument': conversion from 'size_t'
 to 'int', possible loss of data [C:\neuro\ik_llama.cpp\build\common\common.vcxproj]
C:\neuro\ik_llama.cpp\common\json-schema-to-grammar.cpp(154,41): warning C4267: 'argument': conversion from 'size_t'
 to 'int', possible loss of data [C:\neuro\ik_llama.cpp\build\common\common.vcxproj]
C:\neuro\ik_llama.cpp\common\json-schema-to-grammar.cpp(234,29): warning C4267: 'argument': conversion from 'size_t'
 to 'int', possible loss of data [C:\neuro\ik_llama.cpp\build\common\common.vcxproj]
C:\neuro\ik_llama.cpp\common\json-schema-to-grammar.cpp(245,33): warning C4267: 'argument': conversion from 'size_t'
 to 'int', possible loss of data [C:\neuro\ik_llama.cpp\build\common\common.vcxproj]
C:\neuro\ik_llama.cpp\common\json-schema-to-grammar.cpp(558,60): warning C4101: 'e': unreferenced local variable [C:
\neuro\ik_llama.cpp\build\common\common.vcxproj]
  train.cpp
  ngram-cache.cpp
C:\neuro\ik_llama.cpp\common\ngram-cache.cpp(20,50): warning C4244: 'argument': conversion from 'int64_t' to 'const
int', possible loss of data [C:\neuro\ik_llama.cpp\build\common\common.vcxproj]
C:\neuro\ik_llama.cpp\common\ngram-cache.cpp(100,16): warning C4267: 'initializing': conversion from 'size_t' to 'in
t', possible loss of data [C:\neuro\ik_llama.cpp\build\common\common.vcxproj]
C:\neuro\ik_llama.cpp\common\ngram-cache.cpp(147,34): warning C4267: 'initializing': conversion from 'size_t' to 'in
t', possible loss of data [C:\neuro\ik_llama.cpp\build\common\common.vcxproj]
C:\neuro\ik_llama.cpp\common\ngram-cache.cpp(147,24): warning C4267: 'initializing': conversion from 'size_t' to 'co
nst int', possible loss of data [C:\neuro\ik_llama.cpp\build\common\common.vcxproj]
C:\neuro\ik_llama.cpp\common\ngram-cache.cpp(156,82): warning C4267: 'initializing': conversion from 'size_t' to 'in
t', possible loss of data [C:\neuro\ik_llama.cpp\build\common\common.vcxproj]
C:\neuro\ik_llama.cpp\common\ngram-cache.cpp(156,38): warning C4267: 'initializing': conversion from 'size_t' to 'co
nst int', possible loss of data [C:\neuro\ik_llama.cpp\build\common\common.vcxproj]
C:\neuro\ik_llama.cpp\common\ngram-cache.cpp(170,77): warning C4267: 'initializing': conversion from 'size_t' to 'in
t', possible loss of data [C:\neuro\ik_llama.cpp\build\common\common.vcxproj]
C:\neuro\ik_llama.cpp\common\ngram-cache.cpp(170,38): warning C4267: 'initializing': conversion from 'size_t' to 'co
nst int', possible loss of data [C:\neuro\ik_llama.cpp\build\common\common.vcxproj]
C:\neuro\ik_llama.cpp\common\ngram-cache.cpp(202,50): warning C4267: 'initializing': conversion from 'size_t' to 'in
t32_t', possible loss of data [C:\neuro\ik_llama.cpp\build\common\common.vcxproj]
C:\neuro\ik_llama.cpp\common\ngram-cache.cpp(202,31): warning C4267: 'initializing': conversion from 'size_t' to 'co
nst int32_t', possible loss of data [C:\neuro\ik_llama.cpp\build\common\common.vcxproj]
  Generating Code...
  common.vcxproj -> C:\neuro\ik_llama.cpp\build\common\Release\common.lib
  Building Custom Rule C:/neuro/ik_llama.cpp/examples/llava/CMakeLists.txt
  Building Custom Rule C:/neuro/ik_llama.cpp/tests/CMakeLists.txt
  Building Custom Rule C:/neuro/ik_llama.cpp/tests/CMakeLists.txt
  Building Custom Rule C:/neuro/ik_llama.cpp/tests/CMakeLists.txt
  Building Custom Rule C:/neuro/ik_llama.cpp/examples/lookup/CMakeLists.txt
  Building Custom Rule C:/neuro/ik_llama.cpp/tests/CMakeLists.txt
  Building Custom Rule C:/neuro/ik_llama.cpp/tests/CMakeLists.txt
  Building Custom Rule C:/neuro/ik_llama.cpp/tests/CMakeLists.txt
  Building Custom Rule C:/neuro/ik_llama.cpp/tests/CMakeLists.txt
  Building Custom Rule C:/neuro/ik_llama.cpp/tests/CMakeLists.txt
  Building Custom Rule C:/neuro/ik_llama.cpp/examples/gguf-split/CMakeLists.txt
  Building Custom Rule C:/neuro/ik_llama.cpp/tests/CMakeLists.txt
  Building Custom Rule C:/neuro/ik_llama.cpp/tests/CMakeLists.txt
  Building Custom Rule C:/neuro/ik_llama.cpp/examples/sweep-bench/CMakeLists.txt
  Building Custom Rule C:/neuro/ik_llama.cpp/examples/tokenize/CMakeLists.txt
  lookup-merge.cpp
  llava-cli.cpp
  test-sampling.cpp
  test-json-schema-to-grammar.cpp
  test-quantize-fns.cpp
  test-quantize-perf.cpp
  Building Custom Rule C:/neuro/ik_llama.cpp/tests/CMakeLists.txt
  Building Custom Rule C:/neuro/ik_llama.cpp/tests/CMakeLists.txt
C:\neuro\ik_llama.cpp\tests\test-sampling.cpp(157,34): warning C4244: 'argument': conversion from 'llama_token' to '
float', possible loss of data [C:\neuro\ik_llama.cpp\build\tests\test-sampling.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-sampling.cpp(164,45): warning C4267: 'initializing': conversion from 'size_t' to 'l
lama_token', possible loss of data [C:\neuro\ik_llama.cpp\build\tests\test-sampling.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-sampling.cpp(164,36): warning C4267: 'initializing': conversion from 'size_t' to 'c
onst llama_token', possible loss of data [C:\neuro\ik_llama.cpp\build\tests\test-sampling.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-sampling.cpp(179,38): warning C4267: 'initializing': conversion from 'size_t' to 'i
nt', possible loss of data [C:\neuro\ik_llama.cpp\build\tests\test-sampling.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-sampling.cpp(179,24): warning C4267: 'initializing': conversion from 'size_t' to 'c
onst int', possible loss of data [C:\neuro\ik_llama.cpp\build\tests\test-sampling.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-sampling.cpp(189,67): warning C4267: 'initializing': conversion from 'size_t' to 'i
nt', possible loss of data [C:\neuro\ik_llama.cpp\build\tests\test-sampling.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-sampling.cpp(189,39): warning C4267: 'initializing': conversion from 'size_t' to 'c
onst int', possible loss of data [C:\neuro\ik_llama.cpp\build\tests\test-sampling.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-sampling.cpp(190,55): warning C4244: 'initializing': conversion from 'float' to 'in
t', possible loss of data [C:\neuro\ik_llama.cpp\build\tests\test-sampling.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-sampling.cpp(190,48): warning C4244: 'initializing': conversion from 'float' to 'co
nst int', possible loss of data [C:\neuro\ik_llama.cpp\build\tests\test-sampling.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-sampling.cpp(192,33): warning C4267: '=': conversion from 'size_t' to 'llama_token'
, possible loss of data [C:\neuro\ik_llama.cpp\build\tests\test-sampling.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-sampling.cpp(212,31): warning C4244: 'initializing': conversion from 'float' to 'in
t', possible loss of data [C:\neuro\ik_llama.cpp\build\tests\test-sampling.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-sampling.cpp(216,34): warning C4244: '=': conversion from 'float' to 'llama_token',
 possible loss of data [C:\neuro\ik_llama.cpp\build\tests\test-sampling.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-sampling.cpp(229,12): warning C4477: 'printf' : format string '%05ld' requires an a
rgument of type 'long', but variadic argument 2 has type 'const size_t' [C:\neuro\ik_llama.cpp\build\tests\test-samp
ling.vcxproj]
      C:\neuro\ik_llama.cpp\tests\test-sampling.cpp(229,12):
      consider using '%zd' in the format string

  Building Custom Rule C:/neuro/ik_llama.cpp/examples/export-lora/CMakeLists.txt
C:\neuro\ik_llama.cpp\tests\test-sampling.cpp(275,49): warning C4305: 'argument': truncation from 'double' to 'const
 float' [C:\neuro\ik_llama.cpp\build\tests\test-sampling.vcxproj]
  test-tokenizer-1-spm.cpp
  Building Custom Rule C:/neuro/ik_llama.cpp/tests/CMakeLists.txt
  test-rope.cpp
  gguf-split.cpp
  test-tokenizer-0.cpp
  test-model-load-cancel.cpp
  get-model.cpp
  Building Custom Rule C:/neuro/ik_llama.cpp/tests/CMakeLists.txt
  Generating Code...
  get-model.cpp
  get-model.cpp
  Generating Code...
  Generating Code...
C:\neuro\ik_llama.cpp\examples\llava\llava-cli.cpp(89,105): warning C4267: 'argument': conversion from 'size_t' to '
int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llama-llava-cli.vcxproj]
  get-model.cpp
  get-model.cpp
  Generating Code...
  Generating Code...
  sweep-bench.cpp
  export-lora.cpp
  tokenize.cpp
  test-backend-ops.cpp
  test-grad0.cpp
  test-chat-template.cpp
  get-model.cpp
  Generating Code...
  test-grammar-integration.cpp
  Building Custom Rule C:/neuro/ik_llama.cpp/examples/passkey/CMakeLists.txt
  test-tokenizer-1-bpe.cpp
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(601,20): warning C4267: 'initializing': conversion from 'size_t' to
 'int', possible loss of data [C:\neuro\ik_llama.cpp\build\tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(632,24): warning C4244: 'initializing': conversion from 'int64_t' t
o 'double', possible loss of data [C:\neuro\ik_llama.cpp\build\tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(778,87): warning C4244: 'argument': conversion from 'const _Ty' to
'int', possible loss of data [C:\neuro\ik_llama.cpp\build\tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(778,87): warning C4244:         with [C:\neuro\ik_llama.cpp\build\t
ests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(778,87): warning C4244:         [ [C:\neuro\ik_llama.cpp\build\test
s\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(778,87): warning C4244:             _Ty=int64_t [C:\neuro\ik_llama.
cpp\build\tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(778,87): warning C4244:         ] [C:\neuro\ik_llama.cpp\build\test
s\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(778,75): warning C4244: 'argument': conversion from 'const _Ty' to
'int', possible loss of data [C:\neuro\ik_llama.cpp\build\tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(778,75): warning C4244:         with [C:\neuro\ik_llama.cpp\build\t
ests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(778,75): warning C4244:         [ [C:\neuro\ik_llama.cpp\build\test
s\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(778,75): warning C4244:             _Ty=int64_t [C:\neuro\ik_llama.
cpp\build\tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(778,75): warning C4244:         ] [C:\neuro\ik_llama.cpp\build\test
s\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(778,63): warning C4244: 'argument': conversion from 'const _Ty' to
'int', possible loss of data [C:\neuro\ik_llama.cpp\build\tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(778,63): warning C4244:         with [C:\neuro\ik_llama.cpp\build\t
ests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(778,63): warning C4244:         [ [C:\neuro\ik_llama.cpp\build\test
s\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(778,63): warning C4244:             _Ty=int64_t [C:\neuro\ik_llama.
cpp\build\tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(778,63): warning C4244:         ] [C:\neuro\ik_llama.cpp\build\test
s\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(778,51): warning C4244: 'argument': conversion from 'const _Ty' to
'int', possible loss of data [C:\neuro\ik_llama.cpp\build\tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(778,51): warning C4244:         with [C:\neuro\ik_llama.cpp\build\t
ests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(778,51): warning C4244:         [ [C:\neuro\ik_llama.cpp\build\test
s\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(778,51): warning C4244:             _Ty=int64_t [C:\neuro\ik_llama.
cpp\build\tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(778,51): warning C4244:         ] [C:\neuro\ik_llama.cpp\build\test
s\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(814,87): warning C4244: 'argument': conversion from 'const _Ty' to
'int', possible loss of data [C:\neuro\ik_llama.cpp\build\tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(814,87): warning C4244:         with [C:\neuro\ik_llama.cpp\build\t
ests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(814,87): warning C4244:         [ [C:\neuro\ik_llama.cpp\build\test
s\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(814,87): warning C4244:             _Ty=int64_t [C:\neuro\ik_llama.
cpp\build\tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(814,87): warning C4244:         ] [C:\neuro\ik_llama.cpp\build\test
s\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(814,75): warning C4244: 'argument': conversion from 'const _Ty' to
'int', possible loss of data [C:\neuro\ik_llama.cpp\build\tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(814,75): warning C4244:         with [C:\neuro\ik_llama.cpp\build\t
ests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(814,75): warning C4244:         [ [C:\neuro\ik_llama.cpp\build\test
s\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(814,75): warning C4244:             _Ty=int64_t [C:\neuro\ik_llama.
cpp\build\tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(814,75): warning C4244:         ] [C:\neuro\ik_llama.cpp\build\test
s\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(814,63): warning C4244: 'argument': conversion from 'const _Ty' to
'int', possible loss of data [C:\neuro\ik_llama.cpp\build\tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(814,63): warning C4244:         with [C:\neuro\ik_llama.cpp\build\t
ests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(814,63): warning C4244:         [ [C:\neuro\ik_llama.cpp\build\test
s\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(814,63): warning C4244:             _Ty=int64_t [C:\neuro\ik_llama.
cpp\build\tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(814,63): warning C4244:         ] [C:\neuro\ik_llama.cpp\build\test
s\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(814,51): warning C4244: 'argument': conversion from 'const _Ty' to
'int', possible loss of data [C:\neuro\ik_llama.cpp\build\tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(814,51): warning C4244:         with [C:\neuro\ik_llama.cpp\build\t
ests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(814,51): warning C4244:         [ [C:\neuro\ik_llama.cpp\build\test
s\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(814,51): warning C4244:             _Ty=int64_t [C:\neuro\ik_llama.
cpp\build\tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(814,51): warning C4244:         ] [C:\neuro\ik_llama.cpp\build\test
s\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(1280,85): warning C4244: 'argument': conversion from 'const int' to
 'float', possible loss of data [C:\neuro\ik_llama.cpp\build\tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(1280,81): warning C4244: 'argument': conversion from 'const int' to
 'float', possible loss of data [C:\neuro\ik_llama.cpp\build\tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(1431,35): warning C4244: '=': conversion from 'int' to '_Ty', possi
ble loss of data [C:\neuro\ik_llama.cpp\build\tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(1431,35): warning C4244:         with [C:\neuro\ik_llama.cpp\build\
tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(1431,35): warning C4244:         [ [C:\neuro\ik_llama.cpp\build\tes
ts\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(1431,35): warning C4244:             _Ty=float [C:\neuro\ik_llama.c
pp\build\tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(1431,35): warning C4244:         ] [C:\neuro\ik_llama.cpp\build\tes
ts\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(1504,94): warning C4244: 'argument': conversion from 'const _Ty' to
 'int', possible loss of data [C:\neuro\ik_llama.cpp\build\tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(1504,94): warning C4244:         with [C:\neuro\ik_llama.cpp\build\
tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(1504,94): warning C4244:         [ [C:\neuro\ik_llama.cpp\build\tes
ts\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(1504,94): warning C4244:             _Ty=int64_t [C:\neuro\ik_llama
.cpp\build\tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(1504,94): warning C4244:         ] [C:\neuro\ik_llama.cpp\build\tes
ts\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(1504,83): warning C4244: 'argument': conversion from 'const _Ty' to
 'int', possible loss of data [C:\neuro\ik_llama.cpp\build\tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(1504,83): warning C4244:         with [C:\neuro\ik_llama.cpp\build\
tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(1504,83): warning C4244:         [ [C:\neuro\ik_llama.cpp\build\tes
ts\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(1504,83): warning C4244:             _Ty=int64_t [C:\neuro\ik_llama
.cpp\build\tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(1504,83): warning C4244:         ] [C:\neuro\ik_llama.cpp\build\tes
ts\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(1504,73): warning C4244: 'argument': conversion from 'const _Ty' to
 'int', possible loss of data [C:\neuro\ik_llama.cpp\build\tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(1504,73): warning C4244:         with [C:\neuro\ik_llama.cpp\build\
tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(1504,73): warning C4244:         [ [C:\neuro\ik_llama.cpp\build\tes
ts\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(1504,73): warning C4244:             _Ty=int64_t [C:\neuro\ik_llama
.cpp\build\tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(1504,73): warning C4244:         ] [C:\neuro\ik_llama.cpp\build\tes
ts\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(1504,62): warning C4244: 'argument': conversion from 'const _Ty' to
 'int', possible loss of data [C:\neuro\ik_llama.cpp\build\tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(1504,62): warning C4244:         with [C:\neuro\ik_llama.cpp\build\
tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(1504,62): warning C4244:         [ [C:\neuro\ik_llama.cpp\build\tes
ts\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(1504,62): warning C4244:             _Ty=int64_t [C:\neuro\ik_llama
.cpp\build\tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(1504,62): warning C4244:         ] [C:\neuro\ik_llama.cpp\build\tes
ts\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(1677,77): warning C4244: 'argument': conversion from 'const int64_t
' to 'float', possible loss of data [C:\neuro\ik_llama.cpp\build\tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(2377,32): warning C4244: 'initializing': conversion from 'const _El
em' to 'float', possible loss of data [C:\neuro\ik_llama.cpp\build\tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(2377,32): warning C4244:         with [C:\neuro\ik_llama.cpp\build\
tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(2377,32): warning C4244:         [ [C:\neuro\ik_llama.cpp\build\tes
ts\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(2377,32): warning C4244:             _Elem=int [C:\neuro\ik_llama.c
pp\build\tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(2377,32): warning C4244:         ] [C:\neuro\ik_llama.cpp\build\tes
ts\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(2383,125): warning C4244: 'argument': conversion from 'float' to 'i
nt', possible loss of data [C:\neuro\ik_llama.cpp\build\tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(2386,129): warning C4244: 'argument': conversion from 'float' to 'i
nt', possible loss of data [C:\neuro\ik_llama.cpp\build\tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(2387,129): warning C4244: 'argument': conversion from 'float' to 'i
nt', possible loss of data [C:\neuro\ik_llama.cpp\build\tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(2388,129): warning C4244: 'argument': conversion from 'float' to 'i
nt', possible loss of data [C:\neuro\ik_llama.cpp\build\tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(2392,129): warning C4244: 'argument': conversion from 'float' to 'i
nt', possible loss of data [C:\neuro\ik_llama.cpp\build\tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(2393,129): warning C4244: 'argument': conversion from 'float' to 'i
nt', possible loss of data [C:\neuro\ik_llama.cpp\build\tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(2394,129): warning C4244: 'argument': conversion from 'float' to 'i
nt', possible loss of data [C:\neuro\ik_llama.cpp\build\tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(2395,129): warning C4244: 'argument': conversion from 'float' to 'i
nt', possible loss of data [C:\neuro\ik_llama.cpp\build\tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(2396,129): warning C4244: 'argument': conversion from 'float' to 'i
nt', possible loss of data [C:\neuro\ik_llama.cpp\build\tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-backend-ops.cpp(2399,125): warning C4244: 'argument': conversion from 'float' to 'i
nt', possible loss of data [C:\neuro\ik_llama.cpp\build\tests\test-backend-ops.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-chat-template.cpp(117,143): warning C4267: 'argument': conversion from 'size_t' to
'int32_t', possible loss of data [C:\neuro\ik_llama.cpp\build\tests\test-chat-template.vcxproj]
C:\neuro\ik_llama.cpp\tests\test-chat-template.cpp(131,32): warning C4267: 'argument': conversion from 'size_t' to '
int32_t', possible loss of data [C:\neuro\ik_llama.cpp\build\tests\test-chat-template.vcxproj]
C:\neuro\ik_llama.cpp\examples\gguf-split\gguf-split.cpp(257,68): warning C4267: 'argument': conversion from 'size_t
' to 'uint16_t', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\gguf-split\llama-gguf-split.vcxproj]
C:\neuro\ik_llama.cpp\examples\gguf-split\gguf-split.cpp(278,16): warning C4477: 'printf' : format string '%ld' requ
ires an argument of type 'long', but variadic argument 1 has type 'unsigned __int64' [C:\neuro\ik_llama.cpp\build\ex
amples\gguf-split\llama-gguf-split.vcxproj]
      C:\neuro\ik_llama.cpp\examples\gguf-split\gguf-split.cpp(278,16):
      consider using '%zd' in the format string

C:\neuro\ik_llama.cpp\examples\gguf-split\gguf-split.cpp(288,20): warning C4477: 'printf' : format string '%ld' requ
ires an argument of type 'long', but variadic argument 3 has type 'size_t' [C:\neuro\ik_llama.cpp\build\examples\ggu
f-split\llama-gguf-split.vcxproj]
      C:\neuro\ik_llama.cpp\examples\gguf-split\gguf-split.cpp(288,20):
      consider using '%zd' in the format string

C:\neuro\ik_llama.cpp\examples\gguf-split\gguf-split.cpp(295,21): warning C4267: 'initializing': conversion from 'si
ze_t' to 'int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\gguf-split\llama-gguf-split.vcxproj]
C:\neuro\ik_llama.cpp\examples\gguf-split\gguf-split.cpp(369,17): warning C4267: 'initializing': conversion from 'si
ze_t' to 'int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\gguf-split\llama-gguf-split.vcxproj]
  Building Custom Rule C:/neuro/ik_llama.cpp/tests/CMakeLists.txt
  Building Custom Rule C:/neuro/ik_llama.cpp/examples/save-load-state/CMakeLists.txt
  test-llama-grammar.cpp
  Building Custom Rule C:/neuro/ik_llama.cpp/examples/simple/CMakeLists.txt
C:\neuro\ik_llama.cpp\examples\export-lora\export-lora.cpp(254,16): warning C4477: 'printf' : format string '%ld' re
quires an argument of type 'long', but variadic argument 2 has type 'size_t' [C:\neuro\ik_llama.cpp\build\examples\e
xport-lora\llama-export-lora.vcxproj]
      C:\neuro\ik_llama.cpp\examples\export-lora\export-lora.cpp(254,16):
      consider using '%zd' in the format string

C:\neuro\ik_llama.cpp\examples\export-lora\export-lora.cpp(255,16): warning C4477: 'printf' : format string '%ld' re
quires an argument of type 'long', but variadic argument 2 has type 'unsigned __int64' [C:\neuro\ik_llama.cpp\build\
examples\export-lora\llama-export-lora.vcxproj]
      C:\neuro\ik_llama.cpp\examples\export-lora\export-lora.cpp(255,16):
      consider using '%zd' in the format string

C:\neuro\ik_llama.cpp\examples\export-lora\export-lora.cpp(337,24): warning C4477: 'printf' : format string '%ld' re
quires an argument of type 'long', but variadic argument 2 has type 'size_t' [C:\neuro\ik_llama.cpp\build\examples\e
xport-lora\llama-export-lora.vcxproj]
      C:\neuro\ik_llama.cpp\examples\export-lora\export-lora.cpp(337,24):
      consider using '%zd' in the format string

C:\neuro\ik_llama.cpp\examples\tokenize\tokenize.cpp(94,77): warning C4267: 'argument': conversion from 'size_t' to
'int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\tokenize\llama-tokenize.vcxproj]
C:\neuro\ik_llama.cpp\examples\tokenize\tokenize.cpp(98,57): warning C4267: 'argument': conversion from 'size_t' to
'int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\tokenize\llama-tokenize.vcxproj]
C:\neuro\ik_llama.cpp\examples\tokenize\tokenize.cpp(150,91): warning C4267: 'argument': conversion from 'size_t' to
 'int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\tokenize\llama-tokenize.vcxproj]
C:\neuro\ik_llama.cpp\examples\tokenize\tokenize.cpp(155,25): warning C4267: 'initializing': conversion from 'size_t
' to 'int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\tokenize\llama-tokenize.vcxproj]
C:\neuro\ik_llama.cpp\examples\tokenize\tokenize.cpp(172,52): warning C4267: 'argument': conversion from 'size_t' to
 'int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\tokenize\llama-tokenize.vcxproj]
C:\neuro\ik_llama.cpp\examples\tokenize\tokenize.cpp(185,31): warning C4267: 'initializing': conversion from 'size_t
' to 'int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\tokenize\llama-tokenize.vcxproj]
C:\neuro\ik_llama.cpp\examples\tokenize\tokenize.cpp(185,20): warning C4267: 'initializing': conversion from 'size_t
' to 'const int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\tokenize\llama-tokenize.vcxproj]
C:\neuro\ik_llama.cpp\examples\tokenize\tokenize.cpp(399,16): warning C4477: 'printf' : format string '%ld' requires
 an argument of type 'long', but variadic argument 1 has type 'unsigned __int64' [C:\neuro\ik_llama.cpp\build\exampl
es\tokenize\llama-tokenize.vcxproj]
      C:\neuro\ik_llama.cpp\examples\tokenize\tokenize.cpp(399,16):
      consider using '%zd' in the format string

  get-model.cpp
  passkey.cpp
  test-autorelease.cpp
  save-load-state.cpp
  simple.cpp
  Generating Code...
  get-model.cpp
  test-tokenizer-1-spm.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\test-tokenizer-1-spm.exe
  llama-lookup-merge.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\llama-lookup-merge.exe
C:\neuro\ik_llama.cpp\tests\test-llama-grammar.cpp(205,20): warning C4267: '=': conversion from 'size_t' to 'uint32_
t', possible loss of data [C:\neuro\ik_llama.cpp\build\tests\test-llama-grammar.vcxproj]
  get-model.cpp
  Generating Code...
  Generating Code...
  get-model.cpp
C:\neuro\ik_llama.cpp\examples\save-load-state\save-load-state.cpp(45,69): warning C4267: 'argument': conversion fro
m 'size_t' to 'int32_t', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\save-load-state\llama-save-load
-state.vcxproj]
C:\neuro\ik_llama.cpp\examples\save-load-state\save-load-state.cpp(46,26): warning C4267: '+=': conversion from 'siz
e_t' to 'int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\save-load-state\llama-save-load-state.vcx
proj]
C:\neuro\ik_llama.cpp\examples\simple\simple.cpp(64,45): warning C4267: 'initializing': conversion from 'size_t' to
'int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\simple\llama-simple.vcxproj]
C:\neuro\ik_llama.cpp\examples\simple\simple.cpp(64,24): warning C4267: 'initializing': conversion from 'size_t' to
'const int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\simple\llama-simple.vcxproj]
C:\neuro\ik_llama.cpp\examples\simple\simple.cpp(92,48): warning C4267: 'argument': conversion from 'size_t' to 'lla
ma_pos', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\simple\llama-simple.vcxproj]
  Generating Code...
C:\neuro\ik_llama.cpp\examples\passkey\passkey.cpp(29,23): warning C4244: 'argument': conversion from 'time_t' to 'u
nsigned int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\passkey\llama-passkey.vcxproj]
C:\neuro\ik_llama.cpp\examples\passkey\passkey.cpp(94,80): warning C4267: 'initializing': conversion from 'size_t' t
o 'int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\passkey\llama-passkey.vcxproj]
C:\neuro\ik_llama.cpp\examples\passkey\passkey.cpp(94,31): warning C4267: 'initializing': conversion from 'size_t' t
o 'const int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\passkey\llama-passkey.vcxproj]
C:\neuro\ik_llama.cpp\examples\passkey\passkey.cpp(96,46): warning C4267: 'initializing': conversion from 'size_t' t
o 'int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\passkey\llama-passkey.vcxproj]
C:\neuro\ik_llama.cpp\examples\passkey\passkey.cpp(96,28): warning C4267: 'initializing': conversion from 'size_t' t
o 'const int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\passkey\llama-passkey.vcxproj]
  get-model.cpp
  Building Custom Rule C:/neuro/ik_llama.cpp/examples/lookup/CMakeLists.txt
  Building Custom Rule C:/neuro/ik_llama.cpp/pocs/vdot/CMakeLists.txt
  Generating Code...
  Building Custom Rule C:/neuro/ik_llama.cpp/examples/retrieval/CMakeLists.txt
     Creating library C:/neuro/ik_llama.cpp/build/examples/llava/Release/llama-llava-cli.lib and object C:/neuro/ik_
  llama.cpp/build/examples/llava/Release/llama-llava-cli.exp
  lookup.cpp
  test-sampling.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\test-sampling.exe
  q8dot.cpp
  test-grad0.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\test-grad0.exe
  test-rope.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\test-rope.exe
  llama-llava-cli.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\llama-llava-cli.exe
  retrieval.cpp
  test-quantize-fns.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\test-quantize-fns.exe
  test-tokenizer-1-bpe.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\test-tokenizer-1-bpe.exe
  test-autorelease.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\test-autorelease.exe
  llama-tokenize.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\llama-tokenize.exe
  test-tokenizer-0.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\test-tokenizer-0.exe
  get-model.cpp
C:\neuro\ik_llama.cpp\examples\lookup\lookup.cpp(56,102): warning C4267: 'argument': conversion from 'size_t' to 'in
t', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\lookup\llama-lookup.vcxproj]
C:\neuro\ik_llama.cpp\examples\lookup\lookup.cpp(92,33): warning C4267: 'initializing': conversion from 'size_t' to
'int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\lookup\llama-lookup.vcxproj]
C:\neuro\ik_llama.cpp\examples\lookup\lookup.cpp(92,23): warning C4267: 'initializing': conversion from 'size_t' to
'const int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\lookup\llama-lookup.vcxproj]
C:\neuro\ik_llama.cpp\examples\lookup\lookup.cpp(105,16): warning C4267: 'initializing': conversion from 'size_t' to
 'int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\lookup\llama-lookup.vcxproj]
C:\neuro\ik_llama.cpp\examples\lookup\lookup.cpp(210,57): warning C4267: 'argument': conversion from 'size_t' to 'll
ama_pos', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\lookup\llama-lookup.vcxproj]
C:\neuro\ik_llama.cpp\examples\lookup\lookup.cpp(214,35): warning C4267: '+=': conversion from 'size_t' to 'int', po
ssible loss of data [C:\neuro\ik_llama.cpp\build\examples\lookup\llama-lookup.vcxproj]
C:\neuro\ik_llama.cpp\examples\retrieval\retrieval.cpp(79,43): warning C4267: 'argument': conversion from 'size_t' t
o 'llama_pos', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\retrieval\llama-retrieval.vcxproj]
C:\neuro\ik_llama.cpp\examples\retrieval\retrieval.cpp(146,12): warning C4477: 'printf' : format string '%ld' requir
es an argument of type 'long', but variadic argument 1 has type 'unsigned __int64' [C:\neuro\ik_llama.cpp\build\exam
ples\retrieval\llama-retrieval.vcxproj]
      C:\neuro\ik_llama.cpp\examples\retrieval\retrieval.cpp(146,12):
      consider using '%zd' in the format string

C:\neuro\ik_llama.cpp\examples\retrieval\retrieval.cpp(214,37): warning C4267: 'initializing': conversion from 'size
_t' to 'int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\retrieval\llama-retrieval.vcxproj]
C:\neuro\ik_llama.cpp\examples\retrieval\retrieval.cpp(214,24): warning C4267: 'initializing': conversion from 'size
_t' to 'const int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\retrieval\llama-retrieval.vcxproj]
C:\neuro\ik_llama.cpp\examples\retrieval\retrieval.cpp(215,49): warning C4244: 'argument': conversion from 'const ui
nt64_t' to 'int32_t', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\retrieval\llama-retrieval.vcxproj]
C:\neuro\ik_llama.cpp\examples\retrieval\retrieval.cpp(263,59): warning C4244: 'argument': conversion from 'const ui
nt64_t' to 'int32_t', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\retrieval\llama-retrieval.vcxproj]
  Generating Code...
  Building Custom Rule C:/neuro/ik_llama.cpp/examples/gritlm/CMakeLists.txt
  Building Custom Rule C:/neuro/ik_llama.cpp/examples/llava/CMakeLists.txt
  Building Custom Rule C:/neuro/ik_llama.cpp/examples/main/CMakeLists.txt
  Building Custom Rule C:/neuro/ik_llama.cpp/pocs/vdot/CMakeLists.txt
  test-chat-template.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\test-chat-template.exe
  Building Custom Rule C:/neuro/ik_llama.cpp/examples/perplexity/CMakeLists.txt
  Building Custom Rule C:/neuro/ik_llama.cpp/examples/cvector-generator/CMakeLists.txt
  Building Custom Rule C:/neuro/ik_llama.cpp/examples/embedding/CMakeLists.txt
  Building Custom Rule C:/neuro/ik_llama.cpp/tests/CMakeLists.txt
  gritlm.cpp
  minicpmv-cli.cpp
  vdot.cpp
  main.cpp
  perplexity.cpp
  Building Custom Rule C:/neuro/ik_llama.cpp/examples/convert-llama2c-to-ggml/CMakeLists.txt
C:\neuro\ik_llama.cpp\examples\gritlm\gritlm.cpp(23,43): warning C4267: 'initializing': conversion from 'size_t' to
'int32_t', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\gritlm\llama-gritlm.vcxproj]
C:\neuro\ik_llama.cpp\examples\gritlm\gritlm.cpp(23,30): warning C4267: 'initializing': conversion from 'size_t' to
'const int32_t', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\gritlm\llama-gritlm.vcxproj]
C:\neuro\ik_llama.cpp\examples\gritlm\gritlm.cpp(30,82): warning C4267: 'initializing': conversion from 'size_t' to
'int32_t', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\gritlm\llama-gritlm.vcxproj]
C:\neuro\ik_llama.cpp\examples\gritlm\gritlm.cpp(30,30): warning C4267: 'initializing': conversion from 'size_t' to
'const int32_t', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\gritlm\llama-gritlm.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\minicpmv-cli.cpp(198,27): warning C4244: 'initializing': conversion from 'doubl
e' to 'float', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llama-minicpmv-cli.vcxproj]
C:\neuro\ik_llama.cpp\examples\llava\minicpmv-cli.cpp(204,30): warning C4244: 'initializing': conversion from 'doubl
e' to 'float', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llava\llama-minicpmv-cli.vcxproj]
C:\neuro\ik_llama.cpp\examples\gritlm\gritlm.cpp(77,65): warning C4244: 'argument': conversion from 'uint64_t' to 'i
nt', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\gritlm\llama-gritlm.vcxproj]
  Building Custom Rule C:/neuro/ik_llama.cpp/examples/speculative/CMakeLists.txt
  cvector-generator.cpp
  embedding.cpp
  test-quantize-perf.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\test-quantize-perf.exe
  convert-llama2c-to-ggml.cpp
  test-model-load-cancel.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\test-model-load-cancel.exe
  llama-gguf-split.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\llama-gguf-split.exe
  llama-retrieval.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\llama-retrieval.exe
  test-backend-ops.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\test-backend-ops.exe
  test-json-schema-to-grammar.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\test-json-schema-to-grammar.exe
  test-grammar-parser.cpp
  llama-q8dot.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\llama-q8dot.exe
  llama-lookup.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\llama-lookup.exe
  llama-simple.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\llama-simple.exe
  llama-export-lora.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\llama-export-lora.exe
  speculative.cpp
C:\neuro\ik_llama.cpp\tests\test-grammar-parser.cpp(39,73): warning C4267: 'argument': conversion from 'size_t' to '
unsigned int', possible loss of data [C:\neuro\ik_llama.cpp\build\tests\test-grammar-parser.vcxproj]
C:\neuro\ik_llama.cpp\examples\main\main.cpp(399,19): warning C4804: '>': unsafe use of type 'bool' in operation [C:
\neuro\ik_llama.cpp\build\examples\main\llama-cli.vcxproj]
C:\neuro\ik_llama.cpp\examples\cvector-generator\pca.hpp(29,43): warning C4267: 'argument': conversion from 'size_t'
 to 'int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\cvector-generator\llama-cvector-generator.vcx
proj]
  (compiling source file '../../examples/cvector-generator/cvector-generator.cpp')

C:\neuro\ik_llama.cpp\examples\cvector-generator\pca.hpp(41,23): warning C4305: 'initializing': truncation from 'dou
ble' to 'float' [C:\neuro\ik_llama.cpp\build\examples\cvector-generator\llama-cvector-generator.vcxproj]
  (compiling source file '../../examples/cvector-generator/cvector-generator.cpp')

C:\neuro\ik_llama.cpp\examples\cvector-generator\pca.hpp(318,26): warning C4267: '=': conversion from 'size_t' to 'i
nt', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\cvector-generator\llama-cvector-generator.vcxproj]
  (compiling source file '../../examples/cvector-generator/cvector-generator.cpp')

C:\neuro\ik_llama.cpp\examples\cvector-generator\pca.hpp(319,39): warning C4267: '=': conversion from 'size_t' to 'i
nt', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\cvector-generator\llama-cvector-generator.vcxproj]
  (compiling source file '../../examples/cvector-generator/cvector-generator.cpp')

C:\neuro\ik_llama.cpp\examples\cvector-generator\cvector-generator.cpp(99,41): warning C4244: 'argument': conversion
 from 'float' to 'const unsigned __int64', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\cvector-gener
ator\llama-cvector-generator.vcxproj]
C:\neuro\ik_llama.cpp\examples\cvector-generator\cvector-generator.cpp(100,41): warning C4244: 'argument': conversio
n from 'float' to 'const unsigned __int64', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\cvector-gene
rator\llama-cvector-generator.vcxproj]
C:\neuro\ik_llama.cpp\examples\cvector-generator\cvector-generator.cpp(101,50): warning C4244: 'argument': conversio
n from 'float' to 'const unsigned __int64', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\cvector-gene
rator\llama-cvector-generator.vcxproj]
C:\neuro\ik_llama.cpp\examples\cvector-generator\cvector-generator.cpp(106,60): warning C4244: 'argument': conversio
n from 'float' to 'const unsigned __int64', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\cvector-gene
rator\llama-cvector-generator.vcxproj]
C:\neuro\ik_llama.cpp\examples\cvector-generator\cvector-generator.cpp(117,24): warning C4244: 'initializing': conve
rsion from 'int64_t' to 'int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\cvector-generator\llama-c
vector-generator.vcxproj]
C:\neuro\ik_llama.cpp\examples\cvector-generator\cvector-generator.cpp(127,45): warning C4305: 'argument': truncatio
n from 'double' to 'float' [C:\neuro\ik_llama.cpp\build\examples\cvector-generator\llama-cvector-generator.vcxproj]
C:\neuro\ik_llama.cpp\examples\cvector-generator\cvector-generator.cpp(133,28): warning C4267: 'initializing': conve
rsion from 'size_t' to 'int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\cvector-generator\llama-cv
ector-generator.vcxproj]
C:\neuro\ik_llama.cpp\examples\cvector-generator\cvector-generator.cpp(135,20): warning C4244: 'initializing': conve
rsion from 'int64_t' to 'int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\cvector-generator\llama-c
vector-generator.vcxproj]
C:\neuro\ik_llama.cpp\examples\cvector-generator\cvector-generator.cpp(232,24): warning C4267: 'initializing': conve
rsion from 'size_t' to 'int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\cvector-generator\llama-cv
ector-generator.vcxproj]
C:\neuro\ik_llama.cpp\examples\cvector-generator\cvector-generator.cpp(342,73): warning C4267: 'argument': conversio
n from 'size_t' to 'int32_t', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\cvector-generator\llama-cv
ector-generator.vcxproj]
C:\neuro\ik_llama.cpp\examples\cvector-generator\cvector-generator.cpp(355,71): warning C4267: 'argument': conversio
n from 'size_t' to 'int32_t', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\cvector-generator\llama-cv
ector-generator.vcxproj]
C:\neuro\ik_llama.cpp\examples\cvector-generator\cvector-generator.cpp(450,29): warning C4267: '=': conversion from
'size_t' to 'int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\cvector-generator\llama-cvector-gener
ator.vcxproj]
  get-model.cpp
  Generating Code...
  Building Custom Rule C:/neuro/ik_llama.cpp/examples/eval-callback/CMakeLists.txt
  Building Custom Rule C:/neuro/ik_llama.cpp/examples/gbnf-validator/CMakeLists.txt
  Generating colorthemes.css.hpp
  test-llama-grammar.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\test-llama-grammar.exe
C:\neuro\ik_llama.cpp\examples\speculative\speculative.cpp(47,27): warning C4244: '=': conversion from 'time_t' to '
uint32_t', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\speculative\llama-speculative.vcxproj]
C:\neuro\ik_llama.cpp\examples\speculative\speculative.cpp(154,33): warning C4267: 'initializing': conversion from '
size_t' to 'int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\speculative\llama-speculative.vcxproj]
C:\neuro\ik_llama.cpp\examples\speculative\speculative.cpp(154,23): warning C4267: 'initializing': conversion from '
size_t' to 'const int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\speculative\llama-speculative.vc
xproj]
C:\neuro\ik_llama.cpp\examples\speculative\speculative.cpp(175,20): warning C4267: 'initializing': conversion from '
size_t' to 'int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\speculative\llama-speculative.vcxproj]
C:\neuro\ik_llama.cpp\examples\speculative\speculative.cpp(176,20): warning C4267: 'initializing': conversion from '
size_t' to 'int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\speculative\llama-speculative.vcxproj]
C:\neuro\ik_llama.cpp\examples\speculative\speculative.cpp(244,102): warning C4267: 'argument': conversion from 'siz
e_t' to '_Ty', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\speculative\llama-speculative.vcxproj]
C:\neuro\ik_llama.cpp\examples\speculative\speculative.cpp(244,102): warning C4267:         with [C:\neuro\ik_llama.
cpp\build\examples\speculative\llama-speculative.vcxproj]
C:\neuro\ik_llama.cpp\examples\speculative\speculative.cpp(244,102): warning C4267:         [ [C:\neuro\ik_llama.cpp
\build\examples\speculative\llama-speculative.vcxproj]
C:\neuro\ik_llama.cpp\examples\speculative\speculative.cpp(244,102): warning C4267:             _Ty=unsigned int [C:
\neuro\ik_llama.cpp\build\examples\speculative\llama-speculative.vcxproj]
C:\neuro\ik_llama.cpp\examples\speculative\speculative.cpp(244,102): warning C4267:         ] [C:\neuro\ik_llama.cpp
\build\examples\speculative\llama-speculative.vcxproj]
C:\neuro\ik_llama.cpp\examples\speculative\speculative.cpp(260,33): warning C4244: 'initializing': conversion from '
double' to 'float', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\speculative\llama-speculative.vcxpro
j]
  Generating style.css.hpp
  llama-passkey.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\llama-passkey.exe
  Building Custom Rule C:/neuro/ik_llama.cpp/examples/lookup/CMakeLists.txt
  eval-callback.cpp
  gbnf-validator.cpp
  Generating theme-beeninorder.css.hpp
  test-grammar-integration.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\test-grammar-integration.exe
  Generating theme-ketivah.css.hpp
     Creating library C:/neuro/ik_llama.cpp/build/examples/llava/Release/llama-minicpmv-cli.lib and object C:/neuro/
  ik_llama.cpp/build/examples/llava/Release/llama-minicpmv-cli.exp
  lookup-stats.cpp
  Building Custom Rule C:/neuro/ik_llama.cpp/examples/infill/CMakeLists.txt
  Generating theme-mangotango.css.hpp
  llama-gritlm.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\llama-gritlm.exe
  llama-minicpmv-cli.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\llama-minicpmv-cli.exe
  infill.cpp
  Generating theme-playground.css.hpp
  test-grammar-parser.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\test-grammar-parser.exe
  llama-embedding.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\llama-embedding.exe
C:\neuro\ik_llama.cpp\examples\eval-callback\eval-callback.cpp(134,73): warning C4267: 'argument': conversion from '
size_t' to 'int32_t', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\eval-callback\llama-eval-callback.
vcxproj]
  llama-save-load-state.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\llama-save-load-state.exe
  Building Custom Rule C:/neuro/ik_llama.cpp/examples/batched/CMakeLists.txt
C:\neuro\ik_llama.cpp\examples\lookup\lookup-stats.cpp(66,33): warning C4267: 'initializing': conversion from 'size_
t' to 'int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\lookup\llama-lookup-stats.vcxproj]
C:\neuro\ik_llama.cpp\examples\lookup\lookup-stats.cpp(66,23): warning C4267: 'initializing': conversion from 'size_
t' to 'const int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\lookup\llama-lookup-stats.vcxproj]
  Generating theme-polarnight.css.hpp
C:\neuro\ik_llama.cpp\examples\lookup\lookup-stats.cpp(92,39): warning C4267: '+=': conversion from 'size_t' to 'int
', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\lookup\llama-lookup-stats.vcxproj]
  Building Custom Rule C:/neuro/ik_llama.cpp/examples/batched-bench/CMakeLists.txt
  Generating theme-snowstorm.css.hpp
  Generating index.html.hpp
  batched.cpp
  batched-bench.cpp
  llama-convert-llama2c-to-ggml.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\llama-convert-llama2c-to-ggml.exe
  llama-cvector-generator.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\llama-cvector-generator.exe
  llama-gbnf-validator.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\llama-gbnf-validator.exe
  llama-perplexity.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\llama-perplexity.exe
  llama-sweep-bench.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\llama-sweep-bench.exe
  llama-eval-callback.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\llama-eval-callback.exe
  Generating index-new.html.hpp
C:\neuro\ik_llama.cpp\examples\batched\batched.cpp(57,45): warning C4267: 'initializing': conversion from 'size_t' t
o 'int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\batched\llama-batched.vcxproj]
C:\neuro\ik_llama.cpp\examples\batched\batched.cpp(57,24): warning C4267: 'initializing': conversion from 'size_t' t
o 'const int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\batched\llama-batched.vcxproj]
C:\neuro\ik_llama.cpp\examples\batched\batched.cpp(96,50): warning C4267: 'argument': conversion from 'size_t' to 'i
nt32_t', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\batched\llama-batched.vcxproj]
C:\neuro\ik_llama.cpp\examples\batched\batched.cpp(105,48): warning C4267: 'argument': conversion from 'size_t' to '
llama_pos', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\batched\llama-batched.vcxproj]
  Building Custom Rule C:/neuro/ik_llama.cpp/examples/lookahead/CMakeLists.txt
  Generating index.js.hpp
  Generating completion.js.hpp
  Generating system-prompts.js.hpp
  llama-lookup-stats.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\llama-lookup-stats.exe
  lookahead.cpp
  Building Custom Rule C:/neuro/ik_llama.cpp/examples/baby-llama/CMakeLists.txt
  Generating prompt-formats.js.hpp
  llama-batched-bench.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\llama-batched-bench.exe
  Generating json-schema-to-grammar.mjs.hpp
  baby-llama.cpp
  llama-batched.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\llama-batched.exe
  Building Custom Rule C:/neuro/ik_llama.cpp/examples/server/CMakeLists.txt
  server.cpp
  llama-infill.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\llama-infill.exe
C:\neuro\ik_llama.cpp\examples\lookahead\lookahead.cpp(90,33): warning C4267: 'initializing': conversion from 'size_
t' to 'int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\lookahead\llama-lookahead.vcxproj]
C:\neuro\ik_llama.cpp\examples\lookahead\lookahead.cpp(90,23): warning C4267: 'initializing': conversion from 'size_
t' to 'const int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\lookahead\llama-lookahead.vcxproj]
C:\neuro\ik_llama.cpp\examples\lookahead\lookahead.cpp(107,16): warning C4267: 'initializing': conversion from 'size
_t' to 'int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\lookahead\llama-lookahead.vcxproj]
  llama-speculative.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\llama-speculative.exe
C:\neuro\ik_llama.cpp\examples\lookahead\lookahead.cpp(364,129): warning C4267: 'argument': conversion from 'size_t'
 to 'int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\lookahead\llama-lookahead.vcxproj]
  llama-cli.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\llama-cli.exe
  llama-vdot.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\llama-vdot.exe
  Building Custom Rule C:/neuro/ik_llama.cpp/examples/quantize/CMakeLists.txt
  llama-baby-llama.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\llama-baby-llama.exe
C:\neuro\ik_llama.cpp\examples\server\utils.hpp(171,16): warning C4267: 'initializing': conversion from 'size_t' to
'int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\server\llama-server.vcxproj]
  (compiling source file '../../../examples/server/server.cpp')

C:\neuro\ik_llama.cpp\examples\server\utils.hpp(182,52): warning C4267: '=': conversion from 'size_t' to 'uint8_t',
possible loss of data [C:\neuro\ik_llama.cpp\build\examples\server\llama-server.vcxproj]
  (compiling source file '../../../examples/server/server.cpp')

C:\neuro\ik_llama.cpp\examples\server\utils.hpp(203,48): warning C4267: '=': conversion from 'size_t' to 'uint8_t',
possible loss of data [C:\neuro\ik_llama.cpp\build\examples\server\llama-server.vcxproj]
  (compiling source file '../../../examples/server/server.cpp')

  quantize.cpp
  Building Custom Rule C:/neuro/ik_llama.cpp/examples/parallel/CMakeLists.txt
  parallel.cpp
  Building Custom Rule C:/neuro/ik_llama.cpp/examples/lookup/CMakeLists.txt
  llama-lookahead.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\llama-lookahead.exe
C:\neuro\ik_llama.cpp\examples\parallel\parallel.cpp(163,21): warning C4267: '=': conversion from 'size_t' to 'int32
_t', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\parallel\llama-parallel.vcxproj]
C:\neuro\ik_llama.cpp\examples\parallel\parallel.cpp(169,55): warning C4267: 'initializing': conversion from 'size_t
' to 'int32_t', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\parallel\llama-parallel.vcxproj]
C:\neuro\ik_llama.cpp\examples\parallel\parallel.cpp(169,35): warning C4267: 'initializing': conversion from 'size_t
' to 'const int32_t', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\parallel\llama-parallel.vcxproj]
C:\neuro\ik_llama.cpp\examples\parallel\parallel.cpp(263,68): warning C4267: 'argument': conversion from 'size_t' to
 'llama_pos', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\parallel\llama-parallel.vcxproj]
C:\neuro\ik_llama.cpp\examples\parallel\parallel.cpp(271,58): warning C4267: '=': conversion from 'size_t' to 'int32
_t', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\parallel\llama-parallel.vcxproj]
  lookup-create.cpp
  Building Custom Rule C:/neuro/ik_llama.cpp/examples/imatrix/CMakeLists.txt
  imatrix.cpp
C:\neuro\ik_llama.cpp\examples\server\server.cpp(361,48): warning C4244: '+=': conversion from 'const double' to 'ui
nt64_t', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\server\llama-server.vcxproj]
C:\neuro\ik_llama.cpp\examples\server\server.cpp(362,48): warning C4244: '+=': conversion from 'const double' to 'ui
nt64_t', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\server\llama-server.vcxproj]
C:\neuro\ik_llama.cpp\examples\server\server.cpp(368,43): warning C4244: '+=': conversion from 'const double' to 'ui
nt64_t', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\server\llama-server.vcxproj]
C:\neuro\ik_llama.cpp\examples\server\server.cpp(369,43): warning C4244: '+=': conversion from 'const double' to 'ui
nt64_t', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\server\llama-server.vcxproj]
C:\neuro\ik_llama.cpp\examples\server\server.cpp(842,37): warning C4267: 'initializing': conversion from 'size_t' to
 'int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\server\llama-server.vcxproj]
C:\neuro\ik_llama.cpp\examples\server\server.cpp(845,29): warning C4267: 'initializing': conversion from 'size_t' to
 'int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\server\llama-server.vcxproj]
  Building Custom Rule C:/neuro/ik_llama.cpp/examples/llama-bench/CMakeLists.txt
C:\neuro\ik_llama.cpp\examples\server\server.cpp(1570,73): warning C4267: 'initializing': conversion from 'size_t' t
o 'int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\server\llama-server.vcxproj]
C:\neuro\ik_llama.cpp\examples\server\server.cpp(1570,32): warning C4267: 'initializing': conversion from 'size_t' t
o 'const int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\server\llama-server.vcxproj]
C:\neuro\ik_llama.cpp\examples\lookup\lookup-create.cpp(39,96): warning C4267: 'argument': conversion from 'size_t'
to 'int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\lookup\llama-lookup-create.vcxproj]
  llama-bench.cpp
C:\neuro\ik_llama.cpp\examples\server\server.cpp(1969,103): warning C4267: 'argument': conversion from 'size_t' to '
llama_pos', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\server\llama-server.vcxproj]
C:\neuro\ik_llama.cpp\examples\server\server.cpp(2001,71): warning C4267: 'argument': conversion from 'size_t' to 'l
lama_pos', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\server\llama-server.vcxproj]
C:\neuro\ik_llama.cpp\examples\server\server.cpp(2083,66): warning C4267: '=': conversion from 'size_t' to 'int32_t'
, possible loss of data [C:\neuro\ik_llama.cpp\build\examples\server\llama-server.vcxproj]
C:\neuro\ik_llama.cpp\examples\server\server.cpp(2143,74): warning C4267: '=': conversion from 'size_t' to 'int32_t'
, possible loss of data [C:\neuro\ik_llama.cpp\build\examples\server\llama-server.vcxproj]
C:\neuro\ik_llama.cpp\examples\server\server.cpp(2167,58): warning C4267: '=': conversion from 'size_t' to 'int32_t'
, possible loss of data [C:\neuro\ik_llama.cpp\build\examples\server\llama-server.vcxproj]
C:\neuro\ik_llama.cpp\examples\server\server.cpp(2203,46): warning C4805: '!=': unsafe mix of type 'int32_t' and typ
e 'bool' in operation [C:\neuro\ik_llama.cpp\build\examples\server\llama-server.vcxproj]
C:\neuro\ik_llama.cpp\examples\server\server.cpp(2253,97): warning C4267: 'argument': conversion from 'size_t' to 'l
lama_pos', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\server\llama-server.vcxproj]
C:\neuro\ik_llama.cpp\examples\server\server.cpp(2421,57): warning C4267: 'argument': conversion from 'size_t' to 'i
nt32_t', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\server\llama-server.vcxproj]
C:\neuro\ik_llama.cpp\examples\server\server.cpp(3363,21): warning C4267: 'initializing': conversion from 'size_t' t
o 'int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\server\llama-server.vcxproj]
  llama-parallel.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\llama-parallel.exe
  llama-quantize.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\llama-quantize.exe
C:\neuro\ik_llama.cpp\examples\llama-bench\llama-bench.cpp(409,30): warning C4996: 'strdup': The POSIX name for this
 item is deprecated. Instead, use the ISO C and C++ conformant name: _strdup. See online help for details. [C:\neuro
\ik_llama.cpp\build\examples\llama-bench\llama-bench.vcxproj]
  llama-lookup-create.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\llama-lookup-create.exe
C:\neuro\ik_llama.cpp\examples\llama-bench\llama-bench.cpp(1235,31): warning C4267: '=': conversion from 'size_t' to
 'int', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llama-bench\llama-bench.vcxproj]
C:\neuro\ik_llama.cpp\examples\llama-bench\llama-bench.cpp(92,13): warning C4244: 'initializing': conversion from 'd
ouble' to 'T', possible loss of data [C:\neuro\ik_llama.cpp\build\examples\llama-bench\llama-bench.vcxproj]
C:\neuro\ik_llama.cpp\examples\llama-bench\llama-bench.cpp(92,13): warning C4244:         with [C:\neuro\ik_llama.cp
p\build\examples\llama-bench\llama-bench.vcxproj]
C:\neuro\ik_llama.cpp\examples\llama-bench\llama-bench.cpp(92,13): warning C4244:         [ [C:\neuro\ik_llama.cpp\b
uild\examples\llama-bench\llama-bench.vcxproj]
C:\neuro\ik_llama.cpp\examples\llama-bench\llama-bench.cpp(92,13): warning C4244:             T=uint64_t [C:\neuro\i
k_llama.cpp\build\examples\llama-bench\llama-bench.vcxproj]
C:\neuro\ik_llama.cpp\examples\llama-bench\llama-bench.cpp(92,13): warning C4244:         ] [C:\neuro\ik_llama.cpp\b
uild\examples\llama-bench\llama-bench.vcxproj]
      C:\neuro\ik_llama.cpp\examples\llama-bench\llama-bench.cpp(92,13):
      the template instantiation context (the oldest one first) is
          C:\neuro\ik_llama.cpp\examples\llama-bench\llama-bench.cpp(1145,18):
          see reference to function template instantiation 'T stdev<uint64_t>(const std::vector<uint64_t,std::alloca
  tor<uint64_t>> &)' being compiled
          with
          [
              T=uint64_t
          ]

  llama-imatrix.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\llama-imatrix.exe
  llama-bench.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\llama-bench.exe
  llama-server.vcxproj -> C:\neuro\ik_llama.cpp\build\bin\Release\llama-server.exe
  Building Custom Rule C:/neuro/ik_llama.cpp/CMakeLists.txt

C:\neuro\ik_llama.cpp>

------------------------------------------------------------------

PS C:\neuro\ik_llama.cpp\build\bin\Release> ./llama-server.exe -t 7 -c 4096 -m F:\llm\Qwen3-30B-A3B-Q5_K_M.gguf
INFO [                    main] build info | tid="11116" timestamp=1746438993 build=3667 commit="e3fec173"
INFO [                    main] system info | tid="11116" timestamp=1746438993 n_threads=7 n_threads_batch=-1 total_threads=16 system_info="AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: loaded meta data with 35 key-value pairs and 579 tensors from F:\llm\Qwen3-30B-A3B-Q5_K_M.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3-30B-A3B
llama_model_loader: - kv   3:                           general.basename str              = Qwen3-30B-A3B
llama_model_loader: - kv   4:                       general.quantized_by str              = Unsloth
llama_model_loader: - kv   5:                         general.size_label str              = 30B-A3B
llama_model_loader: - kv   6:                           general.repo_url str              = https://huggingface.co/unsloth
llama_model_loader: - kv   7:                       qwen3moe.block_count u32              = 48
llama_model_loader: - kv   8:                    qwen3moe.context_length u32              = 40960
llama_model_loader: - kv   9:                  qwen3moe.embedding_length u32              = 2048
llama_model_loader: - kv  10:               qwen3moe.feed_forward_length u32              = 6144
llama_model_loader: - kv  11:              qwen3moe.attention.head_count u32              = 32
llama_model_loader: - kv  12:           qwen3moe.attention.head_count_kv u32              = 4
llama_model_loader: - kv  13:                    qwen3moe.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  14:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  15:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  16:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  17:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  18:                      qwen3moe.expert_count u32              = 128
llama_model_loader: - kv  19:        qwen3moe.expert_feed_forward_length u32              = 768
llama_model_loader: - kv  20:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  21:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  22:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  23:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  24:                      tokenizer.ggml.merges arr[str,151387]  = ["─а ─а", "─а─а ─а─а", "i n", "─а t",...
llama_model_loader: - kv  25:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  26:            tokenizer.ggml.padding_token_id u32              = 151654
llama_model_loader: - kv  27:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  28:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - kv  29:               general.quantization_version u32              = 2
llama_model_loader: - kv  30:                          general.file_type u32              = 17
llama_model_loader: - kv  31:                      quantize.imatrix.file str              = Qwen3-30B-A3B-GGUF/imatrix_unsloth.dat
llama_model_loader: - kv  32:                   quantize.imatrix.dataset str              = unsloth_calibration_Qwen3-30B-A3B.txt
llama_model_loader: - kv  33:             quantize.imatrix.entries_count i32              = 384
llama_model_loader: - kv  34:              quantize.imatrix.chunks_count i32              = 32
llama_model_loader: - type  f32:  241 tensors
llama_model_loader: - type q5_K:  289 tensors
llama_model_loader: - type q6_K:   49 tensors
llm_load_vocab: special tokens cache size = 26
llm_load_vocab: token to piece cache size = 0.9311 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = qwen3moe
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 151936
llm_load_print_meta: n_merges         = 151387
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 40960
llm_load_print_meta: n_embd           = 2048
llm_load_print_meta: n_layer          = 48
llm_load_print_meta: n_head           = 32
llm_load_print_meta: n_head_kv        = 4
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 8
llm_load_print_meta: n_embd_k_gqa     = 512
llm_load_print_meta: n_embd_v_gqa     = 512
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 6144
llm_load_print_meta: n_expert         = 128
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 2
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 1000000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 40960
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = ?B
llm_load_print_meta: model ftype      = Q5_K - Medium
llm_load_print_meta: model params     = 30.532 B
llm_load_print_meta: model size       = 20.228 GiB (5.691 BPW)
llm_load_print_meta: repeating layers = 19.791 GiB (5.684 BPW, 29.910 B parameters)
llm_load_print_meta: general.name     = Qwen3-30B-A3B
llm_load_print_meta: BOS token        = 11 ','
llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
llm_load_print_meta: PAD token        = 151654 '<|vision_pad|>'
llm_load_print_meta: LF token         = 148848 '├Д─м'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp         = 768
llm_load_tensors: ggml ctx size =    0.25 MiB
llm_load_tensors:        CPU buffer size = 20713.44 MiB
...................................................................................................
llama_new_context_with_model: n_ctx      = 4096
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 0
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:        CPU KV buffer size =   384.00 MiB
llama_new_context_with_model: KV self size  =  384.00 MiB, K (f16):  192.00 MiB, V (f16):  192.00 MiB
llama_new_context_with_model:        CPU  output buffer size =     1.16 MiB
llama_new_context_with_model:        CPU compute buffer size =   304.75 MiB
llama_new_context_with_model: graph nodes  = 2165
llama_new_context_with_model: graph splits = 1
INFO [                    init] initializing slots | tid="11116" timestamp=1746439008 n_slots=1
INFO [                    init] new slot | tid="11116" timestamp=1746439008 id_slot=0 n_ctx_slot=4096
INFO [                    main] model loaded | tid="11116" timestamp=1746439008
INFO [                    main] chat template | tid="11116" timestamp=1746439008 chat_example="<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there<|im_end|>\n<|im_start|>user\nHow are you?<|im_end|>\n<|im_start|>assistant\n" built_in=true
INFO [                    main] HTTP server listening | tid="11116" timestamp=1746439008 hostname="127.0.0.1" port="8080" n_threads_http="15"
INFO [            update_slots] all slots are idle | tid="11116" timestamp=1746439008
INFO [      log_server_request] request | tid="19268" timestamp=1746439081 remote_addr="127.0.0.1" remote_port=63234 status=404 method="GET" path="/models" params={}
INFO [   launch_slot_with_task] slot is processing task | tid="11116" timestamp=1746439086 id_slot=0 id_task=0
INFO [            update_slots] kv cache rm [p0, end) | tid="11116" timestamp=1746439086 id_slot=0 id_task=0 p0=0
PS C:\neuro\ik_llama.cpp\build\bin\Release>

------------------------------------------------------------------

PS C:\neuro\ik_llama.cpp\build\bin\Release> ./llama-server.exe -t 7 -c 4096 -m F:\llm\Qwen3-30B-A3B-Q5_K_M.gguf
INFO [                    main] build info | tid="21556" timestamp=1746439373 build=3667 commit="e3fec173"
INFO [                    main] system info | tid="21556" timestamp=1746439373 n_threads=7 n_threads_batch=-1 total_threads=16 system_info="AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: loaded meta data with 35 key-value pairs and 579 tensors from F:\llm\Qwen3-30B-A3B-Q5_K_M.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3-30B-A3B
llama_model_loader: - kv   3:                           general.basename str              = Qwen3-30B-A3B
llama_model_loader: - kv   4:                       general.quantized_by str              = Unsloth
llama_model_loader: - kv   5:                         general.size_label str              = 30B-A3B
llama_model_loader: - kv   6:                           general.repo_url str              = https://huggingface.co/unsloth
llama_model_loader: - kv   7:                       qwen3moe.block_count u32              = 48
llama_model_loader: - kv   8:                    qwen3moe.context_length u32              = 40960
llama_model_loader: - kv   9:                  qwen3moe.embedding_length u32              = 2048
llama_model_loader: - kv  10:               qwen3moe.feed_forward_length u32              = 6144
llama_model_loader: - kv  11:              qwen3moe.attention.head_count u32              = 32
llama_model_loader: - kv  12:           qwen3moe.attention.head_count_kv u32              = 4
llama_model_loader: - kv  13:                    qwen3moe.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  14:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  15:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  16:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  17:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  18:                      qwen3moe.expert_count u32              = 128
llama_model_loader: - kv  19:        qwen3moe.expert_feed_forward_length u32              = 768
llama_model_loader: - kv  20:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  21:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  22:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  23:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  24:                      tokenizer.ggml.merges arr[str,151387]  = ["─а ─а", "─а─а ─а─а", "i n", "─а t",...
llama_model_loader: - kv  25:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  26:            tokenizer.ggml.padding_token_id u32              = 151654
llama_model_loader: - kv  27:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  28:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - kv  29:               general.quantization_version u32              = 2
llama_model_loader: - kv  30:                          general.file_type u32              = 17
llama_model_loader: - kv  31:                      quantize.imatrix.file str              = Qwen3-30B-A3B-GGUF/imatrix_unsloth.dat
llama_model_loader: - kv  32:                   quantize.imatrix.dataset str              = unsloth_calibration_Qwen3-30B-A3B.txt
llama_model_loader: - kv  33:             quantize.imatrix.entries_count i32              = 384
llama_model_loader: - kv  34:              quantize.imatrix.chunks_count i32              = 32
llama_model_loader: - type  f32:  241 tensors
llama_model_loader: - type q5_K:  289 tensors
llama_model_loader: - type q6_K:   49 tensors
llm_load_vocab: special tokens cache size = 26
llm_load_vocab: token to piece cache size = 0.9311 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = qwen3moe
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 151936
llm_load_print_meta: n_merges         = 151387
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 40960
llm_load_print_meta: n_embd           = 2048
llm_load_print_meta: n_layer          = 48
llm_load_print_meta: n_head           = 32
llm_load_print_meta: n_head_kv        = 4
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 8
llm_load_print_meta: n_embd_k_gqa     = 512
llm_load_print_meta: n_embd_v_gqa     = 512
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 6144
llm_load_print_meta: n_expert         = 128
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 2
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 1000000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 40960
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = ?B
llm_load_print_meta: model ftype      = Q5_K - Medium
llm_load_print_meta: model params     = 30.532 B
llm_load_print_meta: model size       = 20.228 GiB (5.691 BPW)
llm_load_print_meta: repeating layers = 19.791 GiB (5.684 BPW, 29.910 B parameters)
llm_load_print_meta: general.name     = Qwen3-30B-A3B
llm_load_print_meta: BOS token        = 11 ','
llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
llm_load_print_meta: PAD token        = 151654 '<|vision_pad|>'
llm_load_print_meta: LF token         = 148848 '├Д─м'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp         = 768
llm_load_tensors: ggml ctx size =    0.25 MiB
llm_load_tensors:        CPU buffer size = 20713.44 MiB
...................................................................................................
llama_new_context_with_model: n_ctx      = 4096
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 0
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:        CPU KV buffer size =   384.00 MiB
llama_new_context_with_model: KV self size  =  384.00 MiB, K (f16):  192.00 MiB, V (f16):  192.00 MiB
llama_new_context_with_model:        CPU  output buffer size =     1.16 MiB
llama_new_context_with_model:        CPU compute buffer size =   304.75 MiB
llama_new_context_with_model: graph nodes  = 2165
llama_new_context_with_model: graph splits = 1
INFO [                    init] initializing slots | tid="21556" timestamp=1746439379 n_slots=1
INFO [                    init] new slot | tid="21556" timestamp=1746439379 id_slot=0 n_ctx_slot=4096
INFO [                    main] model loaded | tid="21556" timestamp=1746439379
INFO [                    main] chat template | tid="21556" timestamp=1746439379 chat_example="<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there<|im_end|>\n<|im_start|>user\nHow are you?<|im_end|>\n<|im_start|>assistant\n" built_in=true
INFO [                    main] HTTP server listening | tid="21556" timestamp=1746439379 hostname="127.0.0.1" port="8080" n_threads_http="15"
INFO [            update_slots] all slots are idle | tid="21556" timestamp=1746439379
INFO [      log_server_request] request | tid="16816" timestamp=1746439384 remote_addr="127.0.0.1" remote_port=57484 status=200 method="GET" path="/" params={}
INFO [      log_server_request] request | tid="15152" timestamp=1746439384 remote_addr="127.0.0.1" remote_port=61232 status=200 method="GET" path="/completion.js" params={}
INFO [      log_server_request] request | tid="19108" timestamp=1746439384 remote_addr="127.0.0.1" remote_port=61590 status=200 method="GET" path="/json-schema-to-grammar.mjs" params={}
INFO [      log_server_request] request | tid="16816" timestamp=1746439384 remote_addr="127.0.0.1" remote_port=57484 status=200 method="GET" path="/index.js" params={}
INFO [      log_server_request] request | tid="16816" timestamp=1746439384 remote_addr="127.0.0.1" remote_port=57484 status=404 method="GET" path="/favicon.ico" params={}
INFO [   launch_slot_with_task] slot is processing task | tid="21556" timestamp=1746439391 id_slot=0 id_task=0
INFO [            update_slots] kv cache rm [p0, end) | tid="21556" timestamp=1746439391 id_slot=0 id_task=0 p0=0
INFO [           print_timings] prompt eval time     =    1253.52 ms /    50 tokens (   25.07 ms per token,    39.89 tokens per second) | tid="21556" timestamp=1746439402 id_slot=0 id_task=0 t_prompt_processing=1253.524 n_prompt_tokens_processed=50 t_token=25.070479999999996 n_tokens_second=39.88754902179775
INFO [           print_timings] generation eval time =   10483.45 ms /   120 runs   (   87.36 ms per token,    11.45 tokens per second) | tid="21556" timestamp=1746439402 id_slot=0 id_task=0 t_token_generation=10483.451 n_decoded=120 t_token=87.36209166666666 n_tokens_second=11.44661237983561
INFO [           print_timings]           total time =   11736.97 ms | tid="21556" timestamp=1746439402 id_slot=0 id_task=0 t_prompt_processing=1253.524 t_token_generation=10483.451 t_total=11736.974999999999
INFO [            update_slots] slot released | tid="21556" timestamp=1746439402 id_slot=0 id_task=0 n_ctx=4096 n_past=169 n_system_tokens=0 n_cache_tokens=169 truncated=false
INFO [            update_slots] all slots are idle | tid="21556" timestamp=1746439402
INFO [      log_server_request] request | tid="17584" timestamp=1746439402 remote_addr="127.0.0.1" remote_port=64288 status=200 method="POST" path="/completion" params={}
INFO [            update_slots] all slots are idle | tid="21556" timestamp=1746439402
INFO [   launch_slot_with_task] slot is processing task | tid="21556" timestamp=1746439409 id_slot=0 id_task=122
INFO [            update_slots] kv cache rm [p0, end) | tid="21556" timestamp=1746439409 id_slot=0 id_task=122 p0=49
PS C:\neuro\ik_llama.cpp\build\bin\Release>

---

👤 **intulint** commented the **2025-05-05** at **10:14:49**:<br>

Even the benchmark crashes during generation. I don't know what the problem is, but it seems to be related to what happens during generation.

PS C:\neuro\ik_llama.cpp\build\bin\Release> .\llama-sweep-bench.exe -m F:\llm\Qwen3-30B-A3B-Q5_K_M.gguf -c 4096 -t 7
llama_model_loader: loaded meta data with 35 key-value pairs and 579 tensors from F:\llm\Qwen3-30B-A3B-Q5_K_M.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3-30B-A3B
llama_model_loader: - kv   3:                           general.basename str              = Qwen3-30B-A3B
llama_model_loader: - kv   4:                       general.quantized_by str              = Unsloth
llama_model_loader: - kv   5:                         general.size_label str              = 30B-A3B
llama_model_loader: - kv   6:                           general.repo_url str              = https://huggingface.co/unsloth
llama_model_loader: - kv   7:                       qwen3moe.block_count u32              = 48
llama_model_loader: - kv   8:                    qwen3moe.context_length u32              = 40960
llama_model_loader: - kv   9:                  qwen3moe.embedding_length u32              = 2048
llama_model_loader: - kv  10:               qwen3moe.feed_forward_length u32              = 6144
llama_model_loader: - kv  11:              qwen3moe.attention.head_count u32              = 32
llama_model_loader: - kv  12:           qwen3moe.attention.head_count_kv u32              = 4
llama_model_loader: - kv  13:                    qwen3moe.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  14:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  15:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  16:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  17:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  18:                      qwen3moe.expert_count u32              = 128
llama_model_loader: - kv  19:        qwen3moe.expert_feed_forward_length u32              = 768
llama_model_loader: - kv  20:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  21:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  22:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  23:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  24:                      tokenizer.ggml.merges arr[str,151387]  = ["─а ─а", "─а─а ─а─а", "i n", "─а t",...
llama_model_loader: - kv  25:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  26:            tokenizer.ggml.padding_token_id u32              = 151654
llama_model_loader: - kv  27:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  28:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - kv  29:               general.quantization_version u32              = 2
llama_model_loader: - kv  30:                          general.file_type u32              = 17
llama_model_loader: - kv  31:                      quantize.imatrix.file str              = Qwen3-30B-A3B-GGUF/imatrix_unsloth.dat
llama_model_loader: - kv  32:                   quantize.imatrix.dataset str              = unsloth_calibration_Qwen3-30B-A3B.txt
llama_model_loader: - kv  33:             quantize.imatrix.entries_count i32              = 384
llama_model_loader: - kv  34:              quantize.imatrix.chunks_count i32              = 32
llama_model_loader: - type  f32:  241 tensors
llama_model_loader: - type q5_K:  289 tensors
llama_model_loader: - type q6_K:   49 tensors
llm_load_vocab: special tokens cache size = 26
llm_load_vocab: token to piece cache size = 0.9311 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = qwen3moe
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 151936
llm_load_print_meta: n_merges         = 151387
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 40960
llm_load_print_meta: n_embd           = 2048
llm_load_print_meta: n_layer          = 48
llm_load_print_meta: n_head           = 32
llm_load_print_meta: n_head_kv        = 4
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 8
llm_load_print_meta: n_embd_k_gqa     = 512
llm_load_print_meta: n_embd_v_gqa     = 512
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 6144
llm_load_print_meta: n_expert         = 128
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 2
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 1000000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 40960
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = ?B
llm_load_print_meta: model ftype      = Q5_K - Medium
llm_load_print_meta: model params     = 30.532 B
llm_load_print_meta: model size       = 20.228 GiB (5.691 BPW)
llm_load_print_meta: repeating layers = 19.791 GiB (5.684 BPW, 29.910 B parameters)
llm_load_print_meta: general.name     = Qwen3-30B-A3B
llm_load_print_meta: BOS token        = 11 ','
llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
llm_load_print_meta: PAD token        = 151654 '<|vision_pad|>'
llm_load_print_meta: LF token         = 148848 '├Д─м'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp         = 768
llm_load_tensors: ggml ctx size =    0.25 MiB
llm_load_tensors:        CPU buffer size = 20713.44 MiB
...................................................................................................
llama_new_context_with_model: n_ctx      = 4096
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 0
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:        CPU KV buffer size =   384.00 MiB
llama_new_context_with_model: KV self size  =  384.00 MiB, K (f16):  192.00 MiB, V (f16):  192.00 MiB
llama_new_context_with_model:        CPU  output buffer size =     0.58 MiB
llama_new_context_with_model:        CPU compute buffer size =   304.75 MiB
llama_new_context_with_model: graph nodes  = 2165
llama_new_context_with_model: graph splits = 1

main: n_kv_max = 4096, n_batch = 2048, n_ubatch = 512, flash_attn = 0, n_gpu_layers = -1, n_threads = 7, n_threads_batch = 7

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |   10.780 |    47.49 |    8.250 |    15.51 |
PS C:\neuro\ik_llama.cpp\build\bin\Release>

---

👤 **ikawrakow** commented the **2025-05-05** at **10:22:33**:<br>

Can you try running with `-t 8`?

If that works, try also adding `-fa -rtr -fmoe`.

---

👤 **ikawrakow** commented the **2025-05-05** at **10:22:33**:<br>

Can you try running with `-t 8`?

If that works, try also adding `-fa -rtr`.

---

👤 **intulint** commented the **2025-05-05** at **10:42:45**:<br>

8 cores make no difference.
-fa -rtr -fmoe Finally it works, but I noticed that every time before writing a comma the generation stops for half a second. The first time I see this. 
In the llama.cpp avx2 release, generation is much faster.

PS C:\neuro\ik_llama.cpp\build\bin\Release> ./llama-server.exe -t 8 -c 4096 -m F:\llm\Qwen3-30B-A3B-Q5_K_M.gguf
INFO [                    main] build info | tid="11244" timestamp=1746440931 build=3667 commit="e3fec173"
INFO [                    main] system info | tid="11244" timestamp=1746440931 n_threads=8 n_threads_batch=-1 total_threads=16 system_info="AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: loaded meta data with 35 key-value pairs and 579 tensors from F:\llm\Qwen3-30B-A3B-Q5_K_M.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3-30B-A3B
llama_model_loader: - kv   3:                           general.basename str              = Qwen3-30B-A3B
llama_model_loader: - kv   4:                       general.quantized_by str              = Unsloth
llama_model_loader: - kv   5:                         general.size_label str              = 30B-A3B
llama_model_loader: - kv   6:                           general.repo_url str              = https://huggingface.co/unsloth
llama_model_loader: - kv   7:                       qwen3moe.block_count u32              = 48
llama_model_loader: - kv   8:                    qwen3moe.context_length u32              = 40960
llama_model_loader: - kv   9:                  qwen3moe.embedding_length u32              = 2048
llama_model_loader: - kv  10:               qwen3moe.feed_forward_length u32              = 6144
llama_model_loader: - kv  11:              qwen3moe.attention.head_count u32              = 32
llama_model_loader: - kv  12:           qwen3moe.attention.head_count_kv u32              = 4
llama_model_loader: - kv  13:                    qwen3moe.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  14:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  15:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  16:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  17:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  18:                      qwen3moe.expert_count u32              = 128
llama_model_loader: - kv  19:        qwen3moe.expert_feed_forward_length u32              = 768
llama_model_loader: - kv  20:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  21:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  22:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  23:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  24:                      tokenizer.ggml.merges arr[str,151387]  = ["─а ─а", "─а─а ─а─а", "i n", "─а t",...
llama_model_loader: - kv  25:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  26:            tokenizer.ggml.padding_token_id u32              = 151654
llama_model_loader: - kv  27:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  28:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - kv  29:               general.quantization_version u32              = 2
llama_model_loader: - kv  30:                          general.file_type u32              = 17
llama_model_loader: - kv  31:                      quantize.imatrix.file str              = Qwen3-30B-A3B-GGUF/imatrix_unsloth.dat
llama_model_loader: - kv  32:                   quantize.imatrix.dataset str              = unsloth_calibration_Qwen3-30B-A3B.txt
llama_model_loader: - kv  33:             quantize.imatrix.entries_count i32              = 384
llama_model_loader: - kv  34:              quantize.imatrix.chunks_count i32              = 32
llama_model_loader: - type  f32:  241 tensors
llama_model_loader: - type q5_K:  289 tensors
llama_model_loader: - type q6_K:   49 tensors
llm_load_vocab: special tokens cache size = 26
llm_load_vocab: token to piece cache size = 0.9311 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = qwen3moe
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 151936
llm_load_print_meta: n_merges         = 151387
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 40960
llm_load_print_meta: n_embd           = 2048
llm_load_print_meta: n_layer          = 48
llm_load_print_meta: n_head           = 32
llm_load_print_meta: n_head_kv        = 4
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 8
llm_load_print_meta: n_embd_k_gqa     = 512
llm_load_print_meta: n_embd_v_gqa     = 512
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 6144
llm_load_print_meta: n_expert         = 128
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 2
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 1000000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 40960
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = ?B
llm_load_print_meta: model ftype      = Q5_K - Medium
llm_load_print_meta: model params     = 30.532 B
llm_load_print_meta: model size       = 20.228 GiB (5.691 BPW)
llm_load_print_meta: repeating layers = 19.791 GiB (5.684 BPW, 29.910 B parameters)
llm_load_print_meta: general.name     = Qwen3-30B-A3B
llm_load_print_meta: BOS token        = 11 ','
llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
llm_load_print_meta: PAD token        = 151654 '<|vision_pad|>'
llm_load_print_meta: LF token         = 148848 '├Д─м'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp         = 768
llm_load_tensors: ggml ctx size =    0.25 MiB
llm_load_tensors:        CPU buffer size = 20713.44 MiB
...................................................................................................
llama_new_context_with_model: n_ctx      = 4096
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 0
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:        CPU KV buffer size =   384.00 MiB
llama_new_context_with_model: KV self size  =  384.00 MiB, K (f16):  192.00 MiB, V (f16):  192.00 MiB
llama_new_context_with_model:        CPU  output buffer size =     1.16 MiB
llama_new_context_with_model:        CPU compute buffer size =   304.75 MiB
llama_new_context_with_model: graph nodes  = 2165
llama_new_context_with_model: graph splits = 1
INFO [                    init] initializing slots | tid="11244" timestamp=1746440937 n_slots=1
INFO [                    init] new slot | tid="11244" timestamp=1746440937 id_slot=0 n_ctx_slot=4096
INFO [                    main] model loaded | tid="11244" timestamp=1746440937
INFO [                    main] chat template | tid="11244" timestamp=1746440937 chat_example="<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there<|im_end|>\n<|im_start|>user\nHow are you?<|im_end|>\n<|im_start|>assistant\n" built_in=true
INFO [                    main] HTTP server listening | tid="11244" timestamp=1746440937 hostname="127.0.0.1" port="8080" n_threads_http="15"
INFO [            update_slots] all slots are idle | tid="11244" timestamp=1746440937
INFO [   launch_slot_with_task] slot is processing task | tid="11244" timestamp=1746440956 id_slot=0 id_task=0
INFO [            update_slots] kv cache rm [p0, end) | tid="11244" timestamp=1746440956 id_slot=0 id_task=0 p0=0
PS C:\neuro\ik_llama.cpp\build\bin\Release>

--------------------------------------------------------

PS C:\neuro\ik_llama.cpp\build\bin\Release> ./llama-server.exe -t 8 -c 4096 -m F:\llm\Qwen3-30B-A3B-Q5_K_M.gguf -fa -rtr -fmoe
INFO [                    main] build info | tid="12376" timestamp=1746441162 build=3667 commit="e3fec173"
INFO [                    main] system info | tid="12376" timestamp=1746441162 n_threads=8 n_threads_batch=-1 total_threads=16 system_info="AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: loaded meta data with 35 key-value pairs and 579 tensors from F:\llm\Qwen3-30B-A3B-Q5_K_M.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3-30B-A3B
llama_model_loader: - kv   3:                           general.basename str              = Qwen3-30B-A3B
llama_model_loader: - kv   4:                       general.quantized_by str              = Unsloth
llama_model_loader: - kv   5:                         general.size_label str              = 30B-A3B
llama_model_loader: - kv   6:                           general.repo_url str              = https://huggingface.co/unsloth
llama_model_loader: - kv   7:                       qwen3moe.block_count u32              = 48
llama_model_loader: - kv   8:                    qwen3moe.context_length u32              = 40960
llama_model_loader: - kv   9:                  qwen3moe.embedding_length u32              = 2048
llama_model_loader: - kv  10:               qwen3moe.feed_forward_length u32              = 6144
llama_model_loader: - kv  11:              qwen3moe.attention.head_count u32              = 32
llama_model_loader: - kv  12:           qwen3moe.attention.head_count_kv u32              = 4
llama_model_loader: - kv  13:                    qwen3moe.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  14:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  15:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  16:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  17:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  18:                      qwen3moe.expert_count u32              = 128
llama_model_loader: - kv  19:        qwen3moe.expert_feed_forward_length u32              = 768
llama_model_loader: - kv  20:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  21:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  22:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  23:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  24:                      tokenizer.ggml.merges arr[str,151387]  = ["─а ─а", "─а─а ─а─а", "i n", "─а t",...
llama_model_loader: - kv  25:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  26:            tokenizer.ggml.padding_token_id u32              = 151654
llama_model_loader: - kv  27:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  28:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - kv  29:               general.quantization_version u32              = 2
llama_model_loader: - kv  30:                          general.file_type u32              = 17
llama_model_loader: - kv  31:                      quantize.imatrix.file str              = Qwen3-30B-A3B-GGUF/imatrix_unsloth.dat
llama_model_loader: - kv  32:                   quantize.imatrix.dataset str              = unsloth_calibration_Qwen3-30B-A3B.txt
llama_model_loader: - kv  33:             quantize.imatrix.entries_count i32              = 384
llama_model_loader: - kv  34:              quantize.imatrix.chunks_count i32              = 32
llama_model_loader: - type  f32:  241 tensors
llama_model_loader: - type q5_K:  289 tensors
llama_model_loader: - type q6_K:   49 tensors
llm_load_vocab: special tokens cache size = 26
llm_load_vocab: token to piece cache size = 0.9311 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = qwen3moe
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 151936
llm_load_print_meta: n_merges         = 151387
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 40960
llm_load_print_meta: n_embd           = 2048
llm_load_print_meta: n_layer          = 48
llm_load_print_meta: n_head           = 32
llm_load_print_meta: n_head_kv        = 4
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 8
llm_load_print_meta: n_embd_k_gqa     = 512
llm_load_print_meta: n_embd_v_gqa     = 512
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 6144
llm_load_print_meta: n_expert         = 128
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 2
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 1000000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 40960
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = ?B
llm_load_print_meta: model ftype      = Q5_K - Medium
llm_load_print_meta: model params     = 30.532 B
llm_load_print_meta: model size       = 20.228 GiB (5.691 BPW)
llm_load_print_meta: repeating layers = 19.791 GiB (5.684 BPW, 29.910 B parameters)
llm_load_print_meta: general.name     = Qwen3-30B-A3B
llm_load_print_meta: BOS token        = 11 ','
llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
llm_load_print_meta: PAD token        = 151654 '<|vision_pad|>'
llm_load_print_meta: LF token         = 148848 '├Д─м'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp         = 768
llm_load_tensors: ggml ctx size =    0.25 MiB
llm_load_tensors:        CPU buffer size = 20713.44 MiB
...................................................................................................
============ Repacked 337 tensors
llama_new_context_with_model: n_ctx      = 4096
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:        CPU KV buffer size =   384.00 MiB
llama_new_context_with_model: KV self size  =  384.00 MiB, K (f16):  192.00 MiB, V (f16):  192.00 MiB
llama_new_context_with_model:        CPU  output buffer size =     1.16 MiB
llama_new_context_with_model:        CPU compute buffer size =   300.75 MiB
llama_new_context_with_model: graph nodes  = 1878
llama_new_context_with_model: graph splits = 1
INFO [                    init] initializing slots | tid="12376" timestamp=1746441190 n_slots=1
INFO [                    init] new slot | tid="12376" timestamp=1746441190 id_slot=0 n_ctx_slot=4096
INFO [                    main] model loaded | tid="12376" timestamp=1746441190
INFO [                    main] chat template | tid="12376" timestamp=1746441190 chat_example="<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there<|im_end|>\n<|im_start|>user\nHow are you?<|im_end|>\n<|im_start|>assistant\n" built_in=true
INFO [                    main] HTTP server listening | tid="12376" timestamp=1746441190 hostname="127.0.0.1" port="8080" n_threads_http="15"
INFO [            update_slots] all slots are idle | tid="12376" timestamp=1746441190
INFO [   launch_slot_with_task] slot is processing task | tid="12376" timestamp=1746441214 id_slot=0 id_task=0
INFO [            update_slots] kv cache rm [p0, end) | tid="12376" timestamp=1746441214 id_slot=0 id_task=0 p0=0
INFO [           print_timings] prompt eval time     =     767.18 ms /    51 tokens (   15.04 ms per token,    66.48 tokens per second) | tid="12376" timestamp=1746441236 id_slot=0 id_task=0 t_prompt_processing=767.178 n_prompt_tokens_processed=51 t_token=15.04270588235294 n_tokens_second=66.47740159389348
INFO [           print_timings] generation eval time =   21654.80 ms /   288 runs   (   75.19 ms per token,    13.30 tokens per second) | tid="12376" timestamp=1746441236 id_slot=0 id_task=0 t_token_generation=21654.802 n_decoded=288 t_token=75.19028472222222 n_tokens_second=13.299590548091828
INFO [           print_timings]           total time =   22421.98 ms | tid="12376" timestamp=1746441236 id_slot=0 id_task=0 t_prompt_processing=767.178 t_token_generation=21654.802 t_total=22421.98
INFO [            update_slots] slot released | tid="12376" timestamp=1746441236 id_slot=0 id_task=0 n_ctx=4096 n_past=338 n_system_tokens=0 n_cache_tokens=338 truncated=false
INFO [            update_slots] all slots are idle | tid="12376" timestamp=1746441236
INFO [      log_server_request] request | tid="21628" timestamp=1746441236 remote_addr="127.0.0.1" remote_port=65237 status=200 method="POST" path="/completion" params={}
INFO [            update_slots] all slots are idle | tid="12376" timestamp=1746441236
INFO [   launch_slot_with_task] slot is processing task | tid="12376" timestamp=1746441247 id_slot=0 id_task=290
INFO [            update_slots] kv cache rm [p0, end) | tid="12376" timestamp=1746441247 id_slot=0 id_task=290 p0=50
INFO [           print_timings] prompt eval time     =    4001.53 ms /   296 tokens (   13.52 ms per token,    73.97 tokens per second) | tid="12376" timestamp=1746441271 id_slot=0 id_task=290 t_prompt_processing=4001.527 n_prompt_tokens_processed=296 t_token=13.518672297297297 n_tokens_second=73.9717612801313
INFO [           print_timings] generation eval time =   19925.00 ms /   245 runs   (   81.33 ms per token,    12.30 tokens per second) | tid="12376" timestamp=1746441271 id_slot=0 id_task=290 t_token_generation=19924.999 n_decoded=245 t_token=81.32652653061224 n_tokens_second=12.296111031172448
INFO [           print_timings]           total time =   23926.53 ms | tid="12376" timestamp=1746441271 id_slot=0 id_task=290 t_prompt_processing=4001.527 t_token_generation=19924.999 t_total=23926.525999999998
INFO [            update_slots] slot released | tid="12376" timestamp=1746441271 id_slot=0 id_task=290 n_ctx=4096 n_past=590 n_system_tokens=0 n_cache_tokens=590 truncated=false
INFO [            update_slots] all slots are idle | tid="12376" timestamp=1746441271
INFO [      log_server_request] request | tid="21948" timestamp=1746441271 remote_addr="127.0.0.1" remote_port=50253 status=200 method="POST" path="/completion" params={}
INFO [            update_slots] all slots are idle | tid="12376" timestamp=1746441271
INFO [   launch_slot_with_task] slot is processing task | tid="12376" timestamp=1746441283 id_slot=0 id_task=537
INFO [            update_slots] kv cache rm [p0, end) | tid="12376" timestamp=1746441283 id_slot=0 id_task=537 p0=3
INFO [           print_timings] prompt eval time     =    7425.26 ms /   523 tokens (   14.20 ms per token,    70.44 tokens per second) | tid="12376" timestamp=1746441292 id_slot=0 id_task=537 t_prompt_processing=7425.256 n_prompt_tokens_processed=523 t_token=14.197430210325049 n_tokens_second=70.43528196199566
INFO [           print_timings] generation eval time =    1970.69 ms /    24 runs   (   82.11 ms per token,    12.18 tokens per second) | tid="12376" timestamp=1746441292 id_slot=0 id_task=537 t_token_generation=1970.687 n_decoded=24 t_token=82.11195833333333 n_tokens_second=12.178494098758453
INFO [           print_timings]           total time =    9395.94 ms | tid="12376" timestamp=1746441292 id_slot=0 id_task=537 t_prompt_processing=7425.256 t_token_generation=1970.687 t_total=9395.943
INFO [            update_slots] slot released | tid="12376" timestamp=1746441292 id_slot=0 id_task=537 n_ctx=4096 n_past=549 n_system_tokens=0 n_cache_tokens=549 truncated=false
INFO [            update_slots] all slots are idle | tid="12376" timestamp=1746441292
INFO [      log_server_request] request | tid="14164" timestamp=1746441292 remote_addr="127.0.0.1" remote_port=55394 status=200 method="POST" path="/completion" params={}
INFO [            update_slots] all slots are idle | tid="12376" timestamp=1746441292
INFO [      log_server_request] request | tid="20768" timestamp=1746441292 remote_addr="127.0.0.1" remote_port=64794 status=200 method="POST" path="/tokenize" params={}
INFO [      log_server_request] request | tid="18372" timestamp=1746441301 remote_addr="127.0.0.1" remote_port=51189 status=404 method="GET" path="/models" params={}
INFO [      log_server_request] request | tid="18372" timestamp=1746441303 remote_addr="127.0.0.1" remote_port=51189 status=404 method="GET" path="/models" params={}
INFO [   launch_slot_with_task] slot is processing task | tid="12376" timestamp=1746441304 id_slot=0 id_task=563
INFO [            update_slots] kv cache rm [p0, end) | tid="12376" timestamp=1746441304 id_slot=0 id_task=563 p0=0
INFO [           print_timings] prompt eval time     =    6708.66 ms /   512 tokens (   13.10 ms per token,    76.32 tokens per second) | tid="12376" timestamp=1746441368 id_slot=0 id_task=563 t_prompt_processing=6708.662 n_prompt_tokens_processed=512 t_token=13.10285546875 n_tokens_second=76.3192421976245
INFO [           print_timings] generation eval time =   56613.50 ms /   647 runs   (   87.50 ms per token,    11.43 tokens per second) | tid="12376" timestamp=1746441368 id_slot=0 id_task=563 t_token_generation=56613.499 n_decoded=647 t_token=87.50154404945904 n_tokens_second=11.428369760364042
INFO [           print_timings]           total time =   63322.16 ms | tid="12376" timestamp=1746441368 id_slot=0 id_task=563 t_prompt_processing=6708.662 t_token_generation=56613.499 t_total=63322.16100000001
INFO [            update_slots] slot released | tid="12376" timestamp=1746441368 id_slot=0 id_task=563 n_ctx=4096 n_past=1158 n_system_tokens=0 n_cache_tokens=0 truncated=false
INFO [            update_slots] all slots are idle | tid="12376" timestamp=1746441368
INFO [      log_server_request] request | tid="18372" timestamp=1746441368 remote_addr="127.0.0.1" remote_port=51189 status=200 method="POST" path="/chat/completions" params={}
INFO [            update_slots] all slots are idle | tid="12376" timestamp=1746441368

---------------------------------------------------

PS C:\neuro\llama-avx2> ./llama-server.exe -t 8 -c 4096 -m F:\llm\Qwen3-30B-A3B-Q5_K_M.gguf
build: 5273 (8ae5ebcf) with MSVC 19.43.34808.0 for x64
system info: n_threads = 8, n_threads_batch = 8, total_threads = 16

system_info: n_threads = 8 (n_threads_batch = 8) / 16 | CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | BMI2 = 1 | LLAMAFILE = 1 | OPENMP = 1 | AARCH64_REPACK = 1 |

main: binding port with default address family
main: HTTP server is listening, hostname: 127.0.0.1, port: 8080, http threads: 15
main: loading model
srv    load_model: loading model 'F:\llm\Qwen3-30B-A3B-Q5_K_M.gguf'
llama_model_loader: loaded meta data with 35 key-value pairs and 579 tensors from F:\llm\Qwen3-30B-A3B-Q5_K_M.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3-30B-A3B
llama_model_loader: - kv   3:                           general.basename str              = Qwen3-30B-A3B
llama_model_loader: - kv   4:                       general.quantized_by str              = Unsloth
llama_model_loader: - kv   5:                         general.size_label str              = 30B-A3B
llama_model_loader: - kv   6:                           general.repo_url str              = https://huggingface.co/unsloth
llama_model_loader: - kv   7:                       qwen3moe.block_count u32              = 48
llama_model_loader: - kv   8:                    qwen3moe.context_length u32              = 40960
llama_model_loader: - kv   9:                  qwen3moe.embedding_length u32              = 2048
llama_model_loader: - kv  10:               qwen3moe.feed_forward_length u32              = 6144
llama_model_loader: - kv  11:              qwen3moe.attention.head_count u32              = 32
llama_model_loader: - kv  12:           qwen3moe.attention.head_count_kv u32              = 4
llama_model_loader: - kv  13:                    qwen3moe.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  14:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  15:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  16:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  17:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  18:                      qwen3moe.expert_count u32              = 128
llama_model_loader: - kv  19:        qwen3moe.expert_feed_forward_length u32              = 768
llama_model_loader: - kv  20:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  21:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  22:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  23:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  24:                      tokenizer.ggml.merges arr[str,151387]  = ["─а ─а", "─а─а ─а─а", "i n", "─а t",...
llama_model_loader: - kv  25:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  26:            tokenizer.ggml.padding_token_id u32              = 151654
llama_model_loader: - kv  27:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  28:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - kv  29:               general.quantization_version u32              = 2
llama_model_loader: - kv  30:                          general.file_type u32              = 17
llama_model_loader: - kv  31:                      quantize.imatrix.file str              = Qwen3-30B-A3B-GGUF/imatrix_unsloth.dat
llama_model_loader: - kv  32:                   quantize.imatrix.dataset str              = unsloth_calibration_Qwen3-30B-A3B.txt
llama_model_loader: - kv  33:             quantize.imatrix.entries_count i32              = 384
llama_model_loader: - kv  34:              quantize.imatrix.chunks_count i32              = 32
llama_model_loader: - type  f32:  241 tensors
llama_model_loader: - type q5_K:  289 tensors
llama_model_loader: - type q6_K:   49 tensors
print_info: file format = GGUF V3 (latest)
print_info: file type   = Q5_K - Medium
print_info: file size   = 20.23 GiB (5.69 BPW)
load: special tokens cache size = 26
load: token to piece cache size = 0.9311 MB
print_info: arch             = qwen3moe
print_info: vocab_only       = 0
print_info: n_ctx_train      = 40960
print_info: n_embd           = 2048
print_info: n_layer          = 48
print_info: n_head           = 32
print_info: n_head_kv        = 4
print_info: n_rot            = 128
print_info: n_swa            = 0
print_info: n_swa_pattern    = 1
print_info: n_embd_head_k    = 128
print_info: n_embd_head_v    = 128
print_info: n_gqa            = 8
print_info: n_embd_k_gqa     = 512
print_info: n_embd_v_gqa     = 512
print_info: f_norm_eps       = 0.0e+00
print_info: f_norm_rms_eps   = 1.0e-06
print_info: f_clamp_kqv      = 0.0e+00
print_info: f_max_alibi_bias = 0.0e+00
print_info: f_logit_scale    = 0.0e+00
print_info: f_attn_scale     = 0.0e+00
print_info: n_ff             = 6144
print_info: n_expert         = 128
print_info: n_expert_used    = 8
print_info: causal attn      = 1
print_info: pooling type     = 0
print_info: rope type        = 2
print_info: rope scaling     = linear
print_info: freq_base_train  = 1000000.0
print_info: freq_scale_train = 1
print_info: n_ctx_orig_yarn  = 40960
print_info: rope_finetuned   = unknown
print_info: ssm_d_conv       = 0
print_info: ssm_d_inner      = 0
print_info: ssm_d_state      = 0
print_info: ssm_dt_rank      = 0
print_info: ssm_dt_b_c_rms   = 0
print_info: model type       = 30B.A3B
print_info: model params     = 30.53 B
print_info: general.name     = Qwen3-30B-A3B
print_info: n_ff_exp         = 768
print_info: vocab type       = BPE
print_info: n_vocab          = 151936
print_info: n_merges         = 151387
print_info: BOS token        = 11 ','
print_info: EOS token        = 151645 '<|im_end|>'
print_info: EOT token        = 151645 '<|im_end|>'
print_info: PAD token        = 151654 '<|vision_pad|>'
print_info: LF token         = 198 '─К'
print_info: FIM PRE token    = 151659 '<|fim_prefix|>'
print_info: FIM SUF token    = 151661 '<|fim_suffix|>'
print_info: FIM MID token    = 151660 '<|fim_middle|>'
print_info: FIM PAD token    = 151662 '<|fim_pad|>'
print_info: FIM REP token    = 151663 '<|repo_name|>'
print_info: FIM SEP token    = 151664 '<|file_sep|>'
print_info: EOG token        = 151643 '<|endoftext|>'
print_info: EOG token        = 151645 '<|im_end|>'
print_info: EOG token        = 151662 '<|fim_pad|>'
print_info: EOG token        = 151663 '<|repo_name|>'
print_info: EOG token        = 151664 '<|file_sep|>'
print_info: max token length = 256
load_tensors: loading model tensors, this can take a while... (mmap = true)
load_tensors: offloading 0 repeating layers to GPU
load_tensors: offloaded 0/49 layers to GPU
load_tensors:   CPU_Mapped model buffer size = 20713.44 MiB
...................................................................................................
llama_context: constructing llama_context
llama_context: n_seq_max     = 1
llama_context: n_ctx         = 4096
llama_context: n_ctx_per_seq = 4096
llama_context: n_batch       = 2048
llama_context: n_ubatch      = 512
llama_context: causal_attn   = 1
llama_context: flash_attn    = 0
llama_context: freq_base     = 1000000.0
llama_context: freq_scale    = 1
llama_context: n_ctx_per_seq (4096) < n_ctx_train (40960) -- the full capacity of the model will not be utilized
llama_context:        CPU  output buffer size =     0.58 MiB
llama_kv_cache_unified: kv_size = 4096, type_k = 'f16', type_v = 'f16', n_layer = 48, can_shift = 1, padding = 32
llama_kv_cache_unified:        CPU KV buffer size =   384.00 MiB
llama_kv_cache_unified: KV self size  =  384.00 MiB, K (f16):  192.00 MiB, V (f16):  192.00 MiB
llama_context:        CPU compute buffer size =   300.75 MiB
llama_context: graph nodes  = 3126
llama_context: graph splits = 1
common_init_from_params: setting dry_penalty_last_n to ctx_size = 4096
common_init_from_params: warming up the model with an empty run - please wait ... (--no-warmup to disable)
srv  log_server_r: request: GET / 127.0.0.1 503
srv  log_server_r: request: GET / 127.0.0.1 503
srv          init: initializing slots, n_slots = 1
slot         init: id  0 | task -1 | new slot n_ctx_slot = 4096
main: model loaded
main: chat template, chat_template: {%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for forward_message in messages %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- set message = messages[index] %}
    {%- set tool_start = '<tool_response>' %}
    {%- set tool_start_length = tool_start|length %}
    {%- set start_of_message = message.content[:tool_start_length] %}
    {%- set tool_end = '</tool_response>' %}
    {%- set tool_end_length = tool_end|length %}
    {%- set start_pos = (message.content|length) - tool_end_length %}
    {%- if start_pos < 0 %}
        {%- set start_pos = 0 %}
    {%- endif %}
    {%- set end_of_message = message.content[start_pos:] %}
    {%- if ns.multi_step_tool and message.role == "user" and not(start_of_message == tool_start and end_of_message == tool_end) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set content = message.content %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is defined and message.reasoning_content is not none %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in message.content %}
                {%- set content = (message.content.split('</think>')|last).lstrip('\n') %}
                {%- set reasoning_content = (message.content.split('</think>')|first).rstrip('\n') %}
                {%- set reasoning_content = (reasoning_content.split('<think>')|last).lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}, example_format: '<|im_start|>system
You are a helpful assistant<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there<|im_end|>
<|im_start|>user
How are you?<|im_end|>
<|im_start|>assistant
'
main: server is listening on http://127.0.0.1:8080 - starting the main loop
srv  update_slots: all slots are idle
srv  log_server_r: request: GET / 127.0.0.1 200
srv  params_from_: Chat format: Content-only
slot launch_slot_: id  0 | task 0 | processing task
slot update_slots: id  0 | task 0 | new prompt, n_ctx_slot = 4096, n_keep = 0, n_prompt_tokens = 20
slot update_slots: id  0 | task 0 | kv cache rm [0, end)
slot update_slots: id  0 | task 0 | prompt processing progress, n_past = 20, n_tokens = 20, progress = 1.000000
slot update_slots: id  0 | task 0 | prompt done, n_past = 20, n_tokens = 20
slot      release: id  0 | task 0 | stop processing: n_past = 67, truncated = 0
slot print_timing: id  0 | task 0 |
prompt eval time =     713.89 ms /    20 tokens (   35.69 ms per token,    28.02 tokens per second)
       eval time =    3163.91 ms /    48 tokens (   65.91 ms per token,    15.17 tokens per second)
      total time =    3877.80 ms /    68 tokens
srv  update_slots: all slots are idle
srv  log_server_r: request: POST /v1/chat/completions 127.0.0.1 200

---

👤 **ikawrakow** commented the **2025-05-05** at **11:00:59**:<br>

So, with `-rtr -fa -fmoe` it works, but TG is slow (slower than `llama.cpp`). How much slower?
What about prompt processing, or when you have a few thousand tokens in the KV cache?
Is the `llama.cpp` build done with MSVC or with GCC/clang?

Without these flags it does not work. If you try `-rtr -fmoe` and `-fa -fmoe` separately, this will help me pinpoint the issue.

---

👤 **intulint** commented the **2025-05-05** at **11:05:55**:<br>

The speeds are in my message above, it is of course long, but I tried to give all the information

---

👤 **intulint** commented the **2025-05-05** at **11:15:26**:<br>

-fa -fmoe  - works, but also pauses before displaying commas. The speed is also low

INFO [           print_timings] prompt eval time     =    9586.72 ms /   512 tokens (   18.72 ms per token,    53.41 tokens per second) | tid="16952" timestamp=1746443401 id_slot=0 id_task=354 t_prompt_processing=9586.721 n_prompt_tokens_processed=512 t_token=18.724064453125 n_tokens_second=53.407207740790625
INFO [           print_timings] generation eval time =   40935.66 ms /   426 runs   (   96.09 ms per token,    10.41 tokens per second) | tid="16952" timestamp=1746443401 id_slot=0 id_task=354 t_token_generation=40935.658 n_decoded=426 t_token=96.09309389671363 n_tokens_second=10.406575118445634

-rtr -fmoe - falling

---

👤 **ikawrakow** commented the **2025-05-05** at **11:15:51**:<br>

Ah, OK. I see
* `ik_llama.cpp`: PP = 76.3 t/s (512 tokens), TG = 11.4 t/s (647 tokens)
* `llama.cpp`: PP = 28.02 t/s (20 tokens), TG = 15.17 t/s (48 tokens)

Correct? I think it would be more fair to compare for the same (or at least similar) number of tokens generated and same number of tokens in the prompt.

---

👤 **intulint** commented the **2025-05-05** at **11:35:12**:<br>

llama.cpp ~ 1000 - 500
prompt eval time =   35744.63 ms /  1053 tokens (   33.95 ms per token,    29.46 tokens per second)
       eval time =   33454.47 ms /   426 tokens (   78.53 ms per token,    12.73 tokens per second)

ik_llama.cpp -fa -fmoe  ~ 1000 - 500

INFO [           print_timings] prompt eval time     =   20147.56 ms /  1057 tokens (   19.06 ms per token,    52.46 tokens per second) | tid="5624" timestamp=1746444960 id_slot=0 id_task=0 t_prompt_processing=20147.559 n_prompt_tokens_processed=1057 t_token=19.06107757805109 n_tokens_second=52.46293111736265
INFO [           print_timings] generation eval time =   40472.90 ms /   422 runs   (   95.91 ms per token,    10.43 tokens per second) | tid="5624" timestamp=1746444960 id_slot=0 id_task=0 t_token_generation=40472.905 n_decoded=422 t_token=95.90735781990522 n_tokens_second=10.426728696642853

---

👤 **ikawrakow** commented the **2025-05-05** at **11:41:03**:<br>

OK, thanks. I'll look into the failure without flash attention.

> -fa -rtr -fmoe Finally it works, but I noticed that every time before writing a comma the generation stops for half a second. 

Sorry for asking, but in what language is your conversation? I'm asking because a pause before a comma may indicate a performance issue in the token id -> utf-8 conversion code. I haven't looked at that since I forked `llama.cpp` last June, and they may have improved since then.

---

👤 **intulint** commented the **2025-05-05** at **11:43:33**:<br>

This is a good question, I somehow didn't pay attention to what language the pauses in generation are in. Usually Russian, but also English. I'll check now. We need generation in English, right? Or is it important that the entire context is in one language?

---

👤 **ikawrakow** commented the **2025-05-05** at **11:46:02**:<br>

> Or is it important that the entire context is in one language?

I don't know. Just looking for clues what could be slowing it down.

---

👤 **intulint** commented the **2025-05-05** at **11:54:19**:<br>

I launched it only in English and looked more closely, a pause in generation appears after or before the comma is displayed. It lasts a noticeable fraction of a second, and generation continues. Usually in such places - "Okay, the", "So, if", "than B, the"

---

👤 **intulint** commented the **2025-05-05** at **11:56:28**:<br>

To avoid confusion, I checked in 2 frontends. I noticed pauses only on commas.

---

👤 **ikawrakow** commented the **2025-05-05** at **11:57:24**:<br>

Interesting. I don't observe such effects on my Linux box. Are the sampling parameters exactly the same?

---

👤 **intulint** commented the **2025-05-05** at **12:01:40**:<br>

In the native front the servers are standard as far as I understand. I only changed the max tokens when measuring the speed. It didn't affect the pauses.

![Image](https://github.com/user-attachments/assets/a162dca1-b3d3-46ed-8ad6-eff1eeb2d6cc)

![Image](https://github.com/user-attachments/assets/0953b999-b438-4a3f-b16b-7d5f2734e0e9)

---

👤 **intulint** commented the **2025-05-05** at **12:16:22**:<br>

Maybe it's a compiler version? I don't know much, but as I understand it, a fresh one was used during assembly. I remember there were messages during assembly about changing the format of variables and that data loss could occur.

---

👤 **ikawrakow** commented the **2025-05-05** at **12:17:11**:<br>

For reference, here is what I get on my vanilla AVX2 Linux box using 8 threads with the commands
```
./bin/llama-sweep-bench -m Qwen_Qwen3-30B-A3B-Q5_K_M.gguf -c 4096 -t 8 -fa -ctk q8_0 -ctv q8_0 -rtr -fmoe
```

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    3.081 |   166.16 |    5.223 |    24.51 |
|   512 |    128 |    512 |    3.331 |   153.69 |    5.502 |    23.26 |
|   512 |    128 |   1024 |    3.606 |   141.97 |    5.740 |    22.30 |
|   512 |    128 |   1536 |    3.873 |   132.20 |    5.984 |    21.39 |
|   512 |    128 |   2048 |    4.154 |   123.25 |    6.212 |    20.61 |
|   512 |    128 |   2560 |    4.419 |   115.87 |    6.443 |    19.87 |
|   512 |    128 |   3072 |    4.691 |   109.15 |    6.685 |    19.15 |
|   512 |    128 |   3584 |    4.959 |   103.26 |    6.906 |    18.54 |

The model is [this one from Bartowski](https://huggingface.co/bartowski/Qwen_Qwen3-30B-A3B-GGUF/blob/main/Qwen_Qwen3-30B-A3B-Q5_K_M.gguf)

The CPU has a Zen3 core, so I'm not expecting it to be faster than a reasonably up-to-date AVX2 capable CPU. 

In my case it also works without issues with just `-c 4096 -t 8`.

So, something goes seriously wrong with the Windows build. 

Not sure how to debug. I don't have access to a Windows box.

---

👤 **intulint** commented the **2025-05-05** at **12:23:26**:<br>

Got it. I'll try to figure out how and by how much to downgrade the compiler, maybe that will help. If not, I don't know what to do next, I'll run it with llama.cpp.

---

👤 **ikawrakow** commented the **2025-05-05** at **12:31:36**:<br>

You can try building with `GCC or clang`. I cannot give you instructions how one does that as it is a long time since I last did that, so I have forgotten. But IIRC, the GCC build was running ~40% faster than the MSVC build. It wasn't an LLM, but it did involve algorithms with heavy number crunching. It must have been around 2017-2018, so don't know if MSVC has improved since then.

---

👤 **intulint** commented the **2025-05-05** at **12:33:50**:<br>

>Is the llama.cpp build done with MSVC or with GCC/clang?

I have written a script that downloads the latest official releases; I have never compiled such large projects myself before.

By the way, yes, we found the parameters under which it starts.
PS C:\neuro\ik_llama.cpp\build\bin\Release> .\llama-sweep-bench.exe -m F:\llm\Qwen3-30B-A3B-Q5_K_M.gguf -c 4096 -t 8 -fa -fmoe
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    9.384 |    54.56 |    8.596 |    14.89 |
|   512 |    128 |    512 |   10.704 |    47.83 |    8.700 |    14.71 |
|   512 |    128 |   1024 |   10.833 |    47.26 |    8.572 |    14.93 |
|   512 |    128 |   1536 |   11.697 |    43.77 |    8.849 |    14.47 |
|   512 |    128 |   2048 |   12.257 |    41.77 |    9.372 |    13.66 |
|   512 |    128 |   2560 |   13.290 |    38.53 |    9.859 |    12.98 |
|   512 |    128 |   3072 |   14.514 |    35.28 |   11.724 |    10.92 |
|   512 |    128 |   3584 |   14.406 |    35.54 |   10.795 |    11.86 |

---

👤 **intulint** commented the **2025-05-05** at **12:33:50**:<br>

>Is the llama.cpp build done with MSVC or with GCC/clang?
I have written a script that downloads the latest official releases; I have never compiled such large projects myself before.

By the way, yes, we found the parameters under which it starts.
PS C:\neuro\ik_llama.cpp\build\bin\Release> .\llama-sweep-bench.exe -m F:\llm\Qwen3-30B-A3B-Q5_K_M.gguf -c 4096 -t 8 -fa -fmoe
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    9.384 |    54.56 |    8.596 |    14.89 |
|   512 |    128 |    512 |   10.704 |    47.83 |    8.700 |    14.71 |
|   512 |    128 |   1024 |   10.833 |    47.26 |    8.572 |    14.93 |
|   512 |    128 |   1536 |   11.697 |    43.77 |    8.849 |    14.47 |
|   512 |    128 |   2048 |   12.257 |    41.77 |    9.372 |    13.66 |
|   512 |    128 |   2560 |   13.290 |    38.53 |    9.859 |    12.98 |
|   512 |    128 |   3072 |   14.514 |    35.28 |   11.724 |    10.92 |
|   512 |    128 |   3584 |   14.406 |    35.54 |   10.795 |    11.86 |

---

👤 **intulint** commented the **2025-05-05** at **12:35:11**:<br>

Got it, I'll try it in the evening if I figure it out.

---

👤 **ikawrakow** commented the **2025-05-05** at **12:46:18**:<br>

You didn't say what your CPU was, so here another reference point from me on a more recent CPU (Ryzen-7950X). Again using 8 threads to be comparable to yours, same command as above:

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    1.874 |   273.19 |    5.253 |    24.37 |
|   512 |    128 |    512 |    1.993 |   256.92 |    5.414 |    23.64 |
|   512 |    128 |   1024 |    2.131 |   240.24 |    5.523 |    23.17 |
|   512 |    128 |   1536 |    2.273 |   225.30 |    5.620 |    22.77 |
|   512 |    128 |   2048 |    2.417 |   211.83 |    5.721 |    22.37 |
|   512 |    128 |   2560 |    2.549 |   200.86 |    5.821 |    21.99 |
|   512 |    128 |   3072 |    2.688 |   190.46 |    5.925 |    21.60 |
|   512 |    128 |   3584 |    2.828 |   181.02 |    6.013 |    21.29 |

In comparison, mainline `llama.cpp` on the same computer (just pulled and rebuilt)

### With flash attention

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    6.668 |    76.79 |    5.408 |    23.67 |
|   512 |    128 |    512 |    8.692 |    58.91 |    6.007 |    21.31 |
|   512 |    128 |   1024 |   10.831 |    47.27 |    6.781 |    18.88 |
|   512 |    128 |   1536 |   12.907 |    39.67 |    7.603 |    16.84 |
|   512 |    128 |   2048 |   14.947 |    34.26 |    8.544 |    14.98 |
|   512 |    128 |   2560 |   16.958 |    30.19 |    9.603 |    13.33 |
|   512 |    128 |   3072 |   19.009 |    26.93 |   10.614 |    12.06 |
|   512 |    128 |   3584 |   21.115 |    24.25 |   11.577 |    11.06 |

### Without flash attnetion

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    6.246 |    81.98 |    5.522 |    23.18 |
|   512 |    128 |    512 |    6.696 |    76.46 |    5.781 |    22.14 |
|   512 |    128 |   1024 |    7.157 |    71.54 |    6.009 |    21.30 |
|   512 |    128 |   1536 |    7.639 |    67.02 |    6.207 |    20.62 |
|   512 |    128 |   2048 |    8.089 |    63.30 |    6.468 |    19.79 |
|   512 |    128 |   2560 |    8.577 |    59.70 |    6.708 |    19.08 |
|   512 |    128 |   3072 |    9.010 |    56.82 |    7.012 |    18.25 |
|   512 |    128 |   3584 |    9.498 |    53.91 |    7.144 |    17.92 |

---

👤 **intulint** commented the **2025-05-05** at **12:59:45**:<br>

Ah, indeed. This is an assembly on an old server processor 1660v4 with 4 memory channels, 32 GB in total. The speeds during generation are quite good, since the memory gives somewhere around 55 GB/s. Of course, this is not comparable with modern processors.

---

👤 **saood06** commented the **2025-05-05** at **22:30:50**:<br>

> You can try building with `GCC or clang`. I cannot give you instructions how one does that as it is a long time since I last did that, so I have forgotten. 

The easiest way I found to use non MSVC to compile this on Windows was with https://github.com/skeeto/w64devkit but I don't use that as I can't compile there with CUDA (and my Nvidia GPU is the only advantage of my Windows machine), and it wasn't any faster on my machine even for CPU only from what I remember.

---

👤 **alex1284B** commented the **2025-05-14** at **16:37:33**:<br>

I think I have a similar problem, Qwen3 does not produce valid output after two lines of tokens, I tried different quantz IQ_K Q6, the same problems.  But Qwen2.5 is fine. Base llama.cpp works fine  also. Linux, only CPU.
I'm not sure but the line of samplers is different than base llama.cpp
CFG -> Penalties -> top_k -> tfs_z -> typical_p -> top_p -> min_p -> temperature
vs 
sampler chain: logits -> logit-bias -> penalties -> dry -> top-k -> typical -> top-p -> min-p -> xtc -> temp-ext -> dist

`ik_llama.cpp$ ./build/bin/llama-cli --color -m /home/ollama/models/gguf/Qwen3-30B-A3B-Q6_K_L.gguf  --threads 12 --temp 0.6 --min-p 0 --top-k 20 --top-p 0.95 -p "<|im_start|>user\nA drinks machine offers three selections - Tea, Coffee or Random but the machine has been wired up wrongly so that each button does not give what it claims. If each drink costs 50p, how much minimum money do you have to put into the machine to work out which button gives which selection ?<|im_end|>\n<|im_start|>assistant\n"
Log start
main: build = 3693 (0435b68e)
main: built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu
main: seed  = 1747238169
llama_model_loader: loaded meta data with 41 key-value pairs and 579 tensors from /home/ollama/models/gguf/Qwen3-30B-A3B-Q6_K_L.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3 30B A3B
llama_model_loader: - kv   3:                           general.basename str              = Qwen3
llama_model_loader: - kv   4:                         general.size_label str              = 30B-A3B
llama_model_loader: - kv   5:                            general.license str              = apache-2.0
llama_model_loader: - kv   6:                       general.license.link str              = https://huggingface.co/Qwen/Qwen3-30B...
llama_model_loader: - kv   7:                   general.base_model.count u32              = 1
llama_model_loader: - kv   8:                  general.base_model.0.name str              = Qwen3 30B A3B Base
llama_model_loader: - kv   9:          general.base_model.0.organization str              = Qwen
llama_model_loader: - kv  10:              general.base_model.0.repo_url str              = https://huggingface.co/Qwen/Qwen3-30B...
llama_model_loader: - kv  11:                               general.tags arr[str,1]       = ["text-generation"]
llama_model_loader: - kv  12:                       qwen3moe.block_count u32              = 48
llama_model_loader: - kv  13:                    qwen3moe.context_length u32              = 32768
llama_model_loader: - kv  14:                  qwen3moe.embedding_length u32              = 2048
llama_model_loader: - kv  15:               qwen3moe.feed_forward_length u32              = 6144
llama_model_loader: - kv  16:              qwen3moe.attention.head_count u32              = 32
llama_model_loader: - kv  17:           qwen3moe.attention.head_count_kv u32              = 4
llama_model_loader: - kv  18:                    qwen3moe.rope.freq_base f32              = 1000000,000000
llama_model_loader: - kv  19:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0,000001
llama_model_loader: - kv  20:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  21:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  22:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  23:                      qwen3moe.expert_count u32              = 128
llama_model_loader: - kv  24:        qwen3moe.expert_feed_forward_length u32              = 768
llama_model_loader: - kv  25:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  26:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  27:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  28:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  29:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv  30:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  31:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  32:                tokenizer.ggml.bos_token_id u32              = 151643
llama_model_loader: - kv  33:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  34:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - kv  35:               general.quantization_version u32              = 2
llama_model_loader: - kv  36:                          general.file_type u32              = 18
llama_model_loader: - kv  37:                      quantize.imatrix.file str              = /models_out/Qwen3-30B-A3B-GGUF/Qwen_Q...
llama_model_loader: - kv  38:                   quantize.imatrix.dataset str              = /training_data/calibration_datav3.txt
llama_model_loader: - kv  39:             quantize.imatrix.entries_count i32              = 384
llama_model_loader: - kv  40:              quantize.imatrix.chunks_count i32              = 209
llama_model_loader: - type  f32:  241 tensors
llama_model_loader: - type q8_0:   50 tensors
llama_model_loader: - type q6_K:  288 tensors
llm_load_vocab: special tokens cache size = 26
llm_load_vocab: token to piece cache size = 0,9311 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = qwen3moe
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 151936
llm_load_print_meta: n_merges         = 151387
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 32768
llm_load_print_meta: n_embd           = 2048
llm_load_print_meta: n_layer          = 48
llm_load_print_meta: n_head           = 32
llm_load_print_meta: n_head_kv        = 4
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 8
llm_load_print_meta: n_embd_k_gqa     = 512
llm_load_print_meta: n_embd_v_gqa     = 512
llm_load_print_meta: f_norm_eps       = 0,0e+00
llm_load_print_meta: f_norm_rms_eps   = 1,0e-06
llm_load_print_meta: f_clamp_kqv      = 0,0e+00
llm_load_print_meta: f_max_alibi_bias = 0,0e+00
llm_load_print_meta: f_logit_scale    = 0,0e+00
llm_load_print_meta: n_ff             = 6144
llm_load_print_meta: n_expert         = 128
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 2
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 1000000,0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 32768
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = ?B
llm_load_print_meta: model ftype      = Q6_K
llm_load_print_meta: model params     = 30,532 B
llm_load_print_meta: model size       = 23,515 GiB (6,616 BPW) 
llm_load_print_meta: repeating layers = 22,900 GiB (6,577 BPW, 29,910 B parameters)
llm_load_print_meta: general.name     = Qwen3 30B A3B
llm_load_print_meta: BOS token        = 151643 '<|endoftext|>'
llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
llm_load_print_meta: PAD token        = 151643 '<|endoftext|>'
llm_load_print_meta: LF token         = 148848 'ÄĬ'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp         = 768
llm_load_tensors: ggml ctx size =    0,25 MiB
llm_load_tensors:        CPU buffer size = 24079,77 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 0
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000,0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:        CPU KV buffer size =  3072,00 MiB
llama_new_context_with_model: KV self size  = 3072,00 MiB, K (f16): 1536,00 MiB, V (f16): 1536,00 MiB
llama_new_context_with_model:        CPU  output buffer size =     0,58 MiB
llama_new_context_with_model:        CPU compute buffer size =  2136,01 MiB
llama_new_context_with_model: graph nodes  = 2165
llama_new_context_with_model: graph splits = 1

system_info: n_threads = 12 / 24 | AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | 
sampling: 
	repeat_last_n = 64, repeat_penalty = 1,000, frequency_penalty = 0,000, presence_penalty = 0,000
	top_k = 20, tfs_z = 1,000, top_p = 0,950, min_p = 0,000, typical_p = 1,000, temp = 0,600
	mirostat = 0, mirostat_lr = 0,100, mirostat_ent = 5,000
sampling order: 
CFG -> Penalties -> top_k -> tfs_z -> typical_p -> top_p -> min_p -> temperature 
generate: n_ctx = 32768, n_batch = 2048, n_predict = -1, n_keep = 0


user
A drinks machine offers three selections - Tea, Coffee or Random but the machine has been wired up wrongly so that each button does not give what it claims. If each drink costs 50p, how much minimum money do you have to put into the machine to work out which button gives which selection ?
assistant
<think>
Okay, so there's this drinks machine with three buttons: Tea, Coffee, and Random. But the problem is, each button is wired up incorrectly. That means if you press Tea, it's not going to give Tea; same with Coffee. And Random is also not giving a random selection. So, the challenge is to figure out how much money you need to put in to determine which button actually gives which drink. Each drink costs 50p, so we need to find the minimum amount required.

First, let me try to understand the problem better. The machine has three buttons, each labeled incorrectly. So, the Tea button doesn't give Tea, Coffee doesn't give Coffee, and Random doesn't give a random selection. Wait, but what does. So… well... okay. [. So. So,. Wait, well? Wait,. Wait, but then. Wait,. Wait, but now. Wait, So,,. Wait, Hey,. But... So, let me.\? So. So,. So, Also,ab. But the.\ the.\. Let of, Wait,. Hmm. Let is the \,... So,. Let is probably. So, let. Let, let, actually., and go.,.\,.\,,.\, But, wait,… So,\n). So,... etc. So, \ If,… but… but I'm, \ is the the same thing., \, \ The is the the you.,., \ But, \ So, \, \, \. So, \,.\a. So,!\n't sure, \, \ But, \,.\ but the same. the question. So, \ is the problem. So, \ I'm,.\

But,.\, \ So, \ you can you can you have to figure out of, \ the same, \, \, \, \ but there's a lot. So, \, \ Let you get the problem. So, \ I think that's, \, \ but I think that is the problem. So,.\

But, \. So, \ the question. you are you can you can you know,, \, \ but I'm. But, \,.\ The problem is the answer the problem. But, \ I'm the answer the previous. but I'm. But, so you need to be careful, but I am I'm a problem. But, the problem. I have to see, I'm, I'm a bit, I know, but the actual, but, that. Then, but, but I'm a lot. So, but, but I don. So, but, I don! So, but I'm not, but, but, but, the number. But, but I can it's a bit, let me, I don… let me. So, but I can you need to see, in your answer. Let, but, but I need to the, but, but, I don, so, but, but I think it's a bit, but I think I have to be, I'm just that's not, and! So, in my, I have to be you know, but I can you need to solve this is a bit of \ what's the

Okay! It to be, I have to see, etc, I'm, you are you, andab, and, etc, I'll, I'm not. So, and \ I can you are you, I need to the other than, but I can you, I know, I need to make. But, I don, I think of. But, but, I have to make it's, you, I can I can I'm not, the the to me have you, but, I don… I think, I don… but, I am the which, I have to see, I'm going to be it's, I'm a person, I've been, no, I think. For, but I'm. If, I'm. I'm, the all, that, I'm just to be, I think I don. I don the the same, I will, but I am it's a new. But, but, I'm, or, but, but, but, but, but, but, but. But, but I don, I have been confused. So, but, no, in this is the same, I don? But, I think, but, I think, but, you can't, I want to you. That, but this is, but I can you, I mean, I need to. So, but, but the same, I'm, I’m, but, I can you, I'm on the, I'm just, I can I know, I'm in the, I have to me you, but, but, but I'm not. I don… but, I need to be. I need to know, the question, I think, but, but, but I have to say, I'm not to the only, but, no, I think, I'm going to think, but that, you, and and I'm, but, I have you! It, I think, that. I can you \ I was a) the question, I need, is, or, I have the. The problem.

The thing, but, it have to be, I was a lot, but, I know how is the way, but, but, I have to see, I’m not, I think. But, and! Let, I have you! I will be it's. It, but, I, and, I want to be, I don, I'm. I'm, I need to the problem. It, that.



 I need to have you… I have to make, but, and. I need to. So, but, but, if, I'm going to be, but, I have, or, I think about, that, but, I have to get, but, I'm, that, but, and
, and! I'm, I need to be, I just, I need to the, but, that, but, but, that, I don, I think, but, I don! I'm, in this is a very, the, what is that, you. I'm not. But, I was, I think that's a lot, the, that, I'm going to be the, but, it, I need to say, I'm, and. So, but, I'm, but, I have to be, I am, but, is a problem, I need. I’m in the problem, that, you! I think, I'm, I am, but it, I'm not, I think, if I, in the, in the, that, and, but the, but, I can't. But, I, I'm trying

llama_print_timings:        load time =    1206,27 ms
llama_print_timings:      sample time =      49,64 ms /  1459 runs   (    0,03 ms per token, 29392,21 tokens per second)
llama_print_timings: prompt eval time =     337,36 ms /    69 tokens (    4,89 ms per token,   204,53 tokens per second)
llama_print_timings:        eval time =   60951,79 ms /  1458 runs   (   41,81 ms per token,    23,92 tokens per second)
llama_print_timings:       total time =   61937,29 ms /  1527  tokens`

---

👤 **ikawrakow** commented the **2025-05-14** at **16:57:33**:<br>

@alex1284B 

I tried your prompt and I see that it does not work. But of you add `-fa -fmoe`, then it works. Please create a separate issue for this. Thanks.

---

👤 **alex1284B** commented the **2025-05-14** at **17:23:47**:<br>

Thank you, I probably missed these options for starting. My bad.

---

👤 **ikawrakow** commented the **2025-05-25** at **07:10:25**:<br>

Closed via #420