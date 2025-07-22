### ✨ [#159](https://github.com/ikawrakow/ik_llama.cpp/issues/159) - Feature Request: steps how to compile as cmake i struction on the origi al repo not work here.

| **Author** | `ajiekc905` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-12-22 |
| **Updated** | 2025-04-21 |

---

#### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

cmake -B build
cmake --build build --config Release

i"m trying to compile it for use under termux / android.
Thanks


[ 17%] Linking CXX executable ../bin/test-tokenizer-0
ld.lld: error: undefined reference: iqk_mul_mat
>>> referenced by ../ggml/src/libggml.so (disallowed by --no-allow-shlib-undefined)
                                               ld.lld: error: undefined reference: iqk_mul_mat_moe
>>> referenced by ../ggml/src/libggml.so (disallowed by --no-allow-shlib-undefined)           
ld.lld: error: undefined reference: iqk_flash_attn_noalibi                                    >>> referenced by ../ggml/src/libggml.so (disallowed by --no-allow-shlib-undefined)           c++: error: linker command failed with exit code 1 (use -v to see invocation)
make[2]: *** [tests/CMakeFiles/test-tokenizer-0.dir/build.make:104: bin/test-tokenizer-0] Error 1
make[1]: *** [CMakeFiles/Makefile2:2100: tests/CMakeFiles/test-tokenizer-0.dir/all] Error 2   make: *** [Makefile:146: all] Error 2


### Motivation

It implements optimizations and bitnet to work on limited resources on cpu which is exactly termux case.

### Possible Implementation

_No response_

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2024-12-22** at **15:04:29**:<br>

* Does the `Makefile` work?
* Can you post the full output of `make -j` and/or `cmake -B build`? 
* What is the CPU? Does it support `__ARM_FEATURE_DOTPROD` (if `ARM`) or AVX2 (if `x86`)

---

👤 **ajiekc905** commented the **2024-12-27** at **14:01:24**:<br>

Sorry for the delay, was no reception / internet.
**make -j **
~ $ cd ik_llama.cpp/
~/ik_llama.cpp $ git pull
Already up to date.
~/ik_llama.cpp $ make -j
[  1%] Built target build_info
[  1%] Built target sha256
[  3%] Built target xxhash
[  3%] Built target sha1
[  9%] Built target ggml
[ 10%] Linking CXX executable ../../bin/llama-gguf-hash
[ 11%] Linking CXX executable ../../bin/llama-gguf
[ 15%] Built target llama
[ 16%] Building CXX object common/CMakeFiles/common.dir/sampling.cpp.o
[ 16%] Building CXX object common/CMakeFiles/common.dir/console.cpp.o
[ 17%] Building CXX object common/CMakeFiles/common.dir/json-schema-to-grammar.cpp.o
[ 18%] Building CXX object common/CMakeFiles/common.dir/grammar-parser.cpp.o
[ 18%] Building CXX object common/CMakeFiles/common.dir/common.cpp.o
[ 18%] Building CXX object common/CMakeFiles/common.dir/train.cpp.o
[ 18%] Building CXX object examples/quantize-stats/CMakeFiles/llama-quantize-stats.dir/quantize-stats.cpp.o
[ 19%] Building CXX object examples/benchmark/CMakeFiles/llama-bench-matmult.dir/benchmark-matmult.cpp.o
[ 20%] Building C object tests/CMakeFiles/test-c.dir/test-c.c.o
[ 21%] Building CXX object common/CMakeFiles/common.dir/ngram-cache.cpp.o
[ 22%] Building CXX object examples/llava/CMakeFiles/llava.dir/llava.cpp.o
[ 22%] Building CXX object examples/llava/CMakeFiles/llava.dir/clip.cpp.o
ld.lld: error: undefined reference: iqk_mul_mat
>>> referenced by ../../ggml/src/libggml.so (disallowed by --no-allow-shlib-undefined)

ld.lld: error: undefined reference: iqk_mul_mat_moe
>>> referenced by ../../ggml/src/libggml.so (disallowed by --no-allow-shlib-undefined)

ld.lld: error: undefined reference: iqk_flash_attn_noalibi
>>> referenced by ../../ggml/src/libggml.so (disallowed by --no-allow-shlib-undefined)
c++: error: linker command failed with exit code 1 (use -v to see invocation)
ld.lld: error: undefined reference: iqk_mul_mat
>>> referenced by ../../ggml/src/libggml.so (disallowed by --no-allow-shlib-undefined)

ld.lld: error: undefined reference: iqk_mul_mat_moe
>>> referenced by ../../ggml/src/libggml.so (disallowed by --no-allow-shlib-undefined)

ld.lld: error: undefined reference: iqk_flash_attn_noalibi
>>> referenced by ../../ggml/src/libggml.so (disallowed by --no-allow-shlib-undefined)
c++: error: linker command failed with exit code 1 (use -v to see invocation)
make[2]: *** [examples/gguf/CMakeFiles/llama-gguf.dir/build.make:102: bin/llama-gguf] Error 1
make[1]: *** [CMakeFiles/Makefile2:3237: examples/gguf/CMakeFiles/llama-gguf.dir/all] Error 2
make[1]: *** Waiting for unfinished jobs....
make[2]: *** [examples/gguf-hash/CMakeFiles/llama-gguf-hash.dir/build.make:108: bin/llama-gguf-hash] Error 1
make[1]: *** [CMakeFiles/Makefile2:3074: examples/gguf-hash/CMakeFiles/llama-gguf-hash.dir/all] Error 2
[ 22%] Linking C executable ../bin/test-c
[ 22%] Built target test-c
[ 23%] Linking CXX executable ../../bin/llama-bench-matmult
ld.lld: error: undefined reference: iqk_mul_mat
>>> referenced by ../../ggml/src/libggml.so (disallowed by --no-allow-shlib-undefined)

ld.lld: error: undefined reference: iqk_mul_mat_moe
>>> referenced by ../../ggml/src/libggml.so (disallowed by --no-allow-shlib-undefined)

ld.lld: error: undefined reference: iqk_flash_attn_noalibi
>>> referenced by ../../ggml/src/libggml.so (disallowed by --no-allow-shlib-undefined)
c++: error: linker command failed with exit code 1 (use -v to see invocation)
make[2]: *** [examples/benchmark/CMakeFiles/llama-bench-matmult.dir/build.make:105: bin/llama-bench-matmult] Error 1
make[1]: *** [CMakeFiles/Makefile2:2864: examples/benchmark/CMakeFiles/llama-bench-matmult.dir/all] Error 2
[ 24%] Linking CXX executable ../../bin/llama-quantize-stats
ld.lld: error: undefined reference: iqk_mul_mat
>>> referenced by ../../ggml/src/libggml.so (disallowed by --no-allow-shlib-undefined)

ld.lld: error: undefined reference: iqk_mul_mat_moe
>>> referenced by ../../ggml/src/libggml.so (disallowed by --no-allow-shlib-undefined)

ld.lld: error: undefined reference: iqk_flash_attn_noalibi
>>> referenced by ../../ggml/src/libggml.so (disallowed by --no-allow-shlib-undefined)
c++: error: linker command failed with exit code 1 (use -v to see invocation)
make[2]: *** [examples/quantize-stats/CMakeFiles/llama-quantize-stats.dir/build.make:105: bin/llama-quantize-stats] Error 1
make[1]: *** [CMakeFiles/Makefile2:3897: examples/quantize-stats/CMakeFiles/llama-quantize-stats.dir/all] Error 2
[ 24%] Built target llava
/data/data/com.termux/files/home/ik_llama.cpp/common/common.cpp:1913:35: warning: 'codecvt_utf8<char32_t>' is deprecated [-Wdeprecated-declarations]
 1913 |         std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
      |                                   ^
/data/data/com.termux/files/usr/include/c++/v1/codecvt:194:28: note: 'codecvt_utf8<char32_t>' has been explicitly marked deprecated here
  194 | class _LIBCPP_TEMPLATE_VIS _LIBCPP_DEPRECATED_IN_CXX17 codecvt_utf8 : public __codecvt_utf8<_Elem> {
      |                            ^
/data/data/com.termux/files/usr/include/c++/v1/__config:942:41: note: expanded from macro '_LIBCPP_DEPRECATED_IN_CXX17'
  942 | #    define _LIBCPP_DEPRECATED_IN_CXX17 _LIBCPP_DEPRECATED
      |                                         ^
/data/data/com.termux/files/usr/include/c++/v1/__config:915:49: note: expanded from macro '_LIBCPP_DEPRECATED'
  915 | #      define _LIBCPP_DEPRECATED __attribute__((__deprecated__))
      |                                                 ^
/data/data/com.termux/files/home/ik_llama.cpp/common/common.cpp:1913:14: warning: 'wstring_convert<std::codecvt_utf8<char32_t>, char32_t>' is deprecated [-Wdeprecated-declarations]
 1913 |         std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
      |              ^
/data/data/com.termux/files/usr/include/c++/v1/locale:3114:28: note: 'wstring_convert<std::codecvt_utf8<char32_t>, char32_t>' has been explicitly marked deprecated here
 3114 | class _LIBCPP_TEMPLATE_VIS _LIBCPP_DEPRECATED_IN_CXX17 wstring_convert {
      |                            ^
/data/data/com.termux/files/usr/include/c++/v1/__config:942:41: note: expanded from macro '_LIBCPP_DEPRECATED_IN_CXX17'
  942 | #    define _LIBCPP_DEPRECATED_IN_CXX17 _LIBCPP_DEPRECATED
      |                                         ^
/data/data/com.termux/files/usr/include/c++/v1/__config:915:49: note: expanded from macro '_LIBCPP_DEPRECATED'
  915 | #      define _LIBCPP_DEPRECATED __attribute__((__deprecated__))
      |                                                 ^
In file included from /data/data/com.termux/files/home/ik_llama.cpp/common/common.cpp:5:
In file included from /data/data/com.termux/files/home/ik_llama.cpp/common/common.h:7:
In file included from /data/data/com.termux/files/home/ik_llama.cpp/common/sampling.h:5:
In file included from /data/data/com.termux/files/home/ik_llama.cpp/common/grammar-parser.h:14:
In file included from /data/data/com.termux/files/usr/include/c++/v1/vector:325:
In file included from /data/data/com.termux/files/usr/include/c++/v1/__format/formatter_bool.h:20:
In file included from /data/data/com.termux/files/usr/include/c++/v1/__format/formatter_integral.h:35:
/data/data/com.termux/files/usr/include/c++/v1/locale:3257:1: warning: 'wstring_convert<std::codecvt_utf8<char32_t>, char32_t>' is deprecated [-Wdeprecated-declarations]
 3257 | wstring_convert<_Codecvt, _Elem, _WideAlloc, _ByteAlloc>::to_bytes(const _Elem* __frm, const _Elem* __frm_end) {
      | ^
/data/data/com.termux/files/usr/include/c++/v1/locale:3161:12: note: in instantiation of member function 'std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t>::to_bytes' requested here
 3161 |     return to_bytes(__wstr.data(), __wstr.data() + __wstr.size());
      |            ^
/data/data/com.termux/files/home/ik_llama.cpp/common/common.cpp:1918:52: note: in instantiation of member function 'std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t>::to_bytes' requested here
 1918 |         std::string filename_reencoded = converter.to_bytes(filename_utf32);
      |                                                    ^
/data/data/com.termux/files/usr/include/c++/v1/locale:3114:28: note: 'wstring_convert<std::codecvt_utf8<char32_t>, char32_t>' has been explicitly marked deprecated here
 3114 | class _LIBCPP_TEMPLATE_VIS _LIBCPP_DEPRECATED_IN_CXX17 wstring_convert {
      |                            ^
/data/data/com.termux/files/usr/include/c++/v1/__config:942:41: note: expanded from macro '_LIBCPP_DEPRECATED_IN_CXX17'
  942 | #    define _LIBCPP_DEPRECATED_IN_CXX17 _LIBCPP_DEPRECATED
      |                                         ^
/data/data/com.termux/files/usr/include/c++/v1/__config:915:49: note: expanded from macro '_LIBCPP_DEPRECATED'
  915 | #      define _LIBCPP_DEPRECATED __attribute__((__deprecated__))
      |                                                 ^
3 warnings generated.
[ 24%] Linking CXX static library libcommon.a
[ 24%] Built target common
make: *** [Makefile:146: all] Error 2
~/ik_llama.cpp $



** cmake -B build **
~/ik_llama.cpp $ git pull
Already up to date.
~/ik_llama.cpp $ cmake -B build
-- OpenMP found
-- Using optimized iqk matrix multiplications
-- Using llamafile
-- ccache found, compilation results will be cached. Disable with GGML_CCACHE=OFF.
-- CMAKE_SYSTEM_PROCESSOR: aarch64
-- ARM detected
-- Configuring done (1.4s)
-- Generating done (1.8s)
-- Build files have been written to: /data/data/com.termux/files/home/ik_llama.cpp/build
~/ik_llama.cpp $ ls
AUTHORS                ci                             grammars                       llama-gguf           llama-q8dot            mypy.ini
CMakeCache.txt         cmake                          include                        llama-gguf-hash      llama-quantize         pocs
CMakeFiles             cmake_install.cmake            libllava.a                     llama-gguf-split     llama-quantize-stats   poetry.lock
CMakeLists.txt         common                         llama-baby-llama               llama-gritlm         llama-retrieval        prompts
CMakePresets.json      compile.log                    llama-batched                  llama-imatrix        llama-save-load-state  pyproject.toml
CONTRIBUTING.md        compile_commands.json          llama-batched-bench            llama-infill         llama-server           pyrightconfig.json
CTestTestfile.cmake    convert_hf_to_gguf.py          llama-bench                    llama-llava-cli      llama-simple           requirements
DartConfiguration.tcl  convert_hf_to_gguf_update.py   llama-benchmark-matmult        llama-lookahead      llama-speculative      requirements.txt
LICENSE                convert_llama_ggml_to_gguf.py  llama-cli                      llama-lookup         llama-tokenize         run.sh
Makefile               convert_lora_to_gguf.py        llama-config.cmake             llama-lookup-create  llama-vdot             scripts
Package.swift          docs                           llama-convert-llama2c-to-ggml  llama-lookup-merge   llama-version.cmake    server
README.md              examples                       llama-cvector-generator        llama-lookup-stats   llama.pc               spm-headers
Testing                flake.lock                     llama-embedding                llama-minicpmv-cli   log.log                src
bartowski.sh           flake.nix                      llama-eval-callback            llama-parallel       main                   tests
bin                    ggml                           llama-export-lora              llama-passkey        media                  up.sh
build                  gguf-py                        llama-gbnf-validator           llama-perplexity     models
~/ik_llama.cpp $ ./llama-cli
Illegal instruction


It look like the compiler use sve / sve2 which is not implemented in Qualcom 8 gen 1, 2, 3. Cmake compilation use to fail too. 

This is how look like cmake for **original llama** repository.

~/llama.cpp $ git pull
Already up to date.
~/llama.cpp $ cmake -B build
-- ccache found, compilation results will be cached. Disable with GGML_CCACHE=OFF.
-- CMAKE_SYSTEM_PROCESSOR: aarch64
-- Including CPU backend
-- ARM detected
-- ARM -mcpu not found, -mcpu=native will be used
-- ARM feature DOTPROD enabled
-- ARM feature MATMUL_INT8 enabled
-- ARM feature FMA enabled
-- ARM feature FP16_VECTOR_ARITHMETIC enabled
-- Adding CPU backend variant ggml-cpu: -mcpu=native+dotprod+i8mm+nosve
-- Configuring done (1.4s)
-- Generating done (2.1s)
-- Build files have been written to: /data/data/com.termux/files/home/llama.cpp/build

~/llama.cpp $ ./llama-cli
build: 74 (d79d8f3) with clang version 19.1.6 for aarch64-unknown-linux-android24
main: llama backend init
main: load the model and apply lora adapter, if any
gguf_init_from_file: failed to open 'models/7B/ggml-model-f16.gguf': 'No such file or directory'
llama_model_load: error loading model: llama_model_loader: failed to load model from models/7B/ggml-model-f16.gguf

llama_load_model_from_file: failed to load model
common_init_from_params: failed to load model 'models/7B/ggml-model-f16.gguf'
main: error: unable to load model

---

👤 **ikawrakow** commented the **2024-12-27** at **17:50:50**:<br>

Thanks, but this doesn't show the part where ggml is being built. I think you need to do 'make clean' first.

---

👤 **ajiekc905** commented the **2024-12-28** at **00:58:23**:<br>

~/ik_llama.cpp $ make clean
~/ik_llama.cpp $ make --jobs=1 VERBOSE=0
/data/data/com.termux/files/usr/bin/cmake -S/data/data/com.termux/files/home/ik_llama.cpp -B/data/data/com.termux/files/home/ik_llama.cpp --check-build-system CMakeFiles/Makefile.cmake 0
/data/data/com.termux/files/usr/bin/cmake -E cmake_progress_start /data/data/com.termux/files/home/ik_llama.cpp/CMakeFiles /data/data/com.termux/files/home/ik_llama.cpp//CMakeFiles/progress.marks
make  -f CMakeFiles/Makefile2 all
make[1]: Entering directory '/data/data/com.termux/files/home/ik_llama.cpp'
make  -f ggml/src/CMakeFiles/ggml.dir/build.make ggml/src/CMakeFiles/ggml.dir/depend
make[2]: Entering directory '/data/data/com.termux/files/home/ik_llama.cpp'
cd /data/data/com.termux/files/home/ik_llama.cpp && /data/data/com.termux/files/usr/bin/cmake -E cmake_depends "Unix Makefiles" /data/data/com.termux/files/home/ik_llama.cpp /data/data/com.termux/files/home/ik_llama.cpp/ggml/src /data/data/com.termux/files/home/ik_llama.cpp /data/data/com.termux/files/home/ik_llama.cpp/ggml/src /data/data/com.termux/files/home/ik_llama.cpp/ggml/src/CMakeFiles/ggml.dir/DependInfo.cmake "--color="
make[2]: Leaving directory '/data/data/com.termux/files/home/ik_llama.cpp'
make  -f ggml/src/CMakeFiles/ggml.dir/build.make ggml/src/CMakeFiles/ggml.dir/build
make[2]: Entering directory '/data/data/com.termux/files/home/ik_llama.cpp'
[  1%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml-aarch64.c.o
cd /data/data/com.termux/files/home/ik_llama.cpp/ggml/src && ccache /data/data/com.termux/files/usr/bin/cc -DGGML_BUILD -DGGML_SCHED_MAX_COPIES=4 -DGGML_SHARED -DGGML_USE_IQK_MULMAT -DGGML_USE_LLAMAFILE -DGGML_USE_OPENMP -D_XOPEN_SOURCE=600 -Dggml_EXPORTS -I/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/../include -I/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/. -O2 -g -DNDEBUG -std=gnu11 -fPIC -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes -Werror=implicit-int -Werror=implicit-function-declaration -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wunreachable-code-break -Wunreachable-code-return -Wdouble-promotion -pthread -fopenmp=libomp -MD -MT ggml/src/CMakeFiles/ggml.dir/ggml-aarch64.c.o -MF CMakeFiles/ggml.dir/ggml-aarch64.c.o.d -o CMakeFiles/ggml.dir/ggml-aarch64.c.o -c /data/data/com.termux/files/home/ik_llama.cpp/ggml/src/ggml-aarch64.c
[  2%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml-alloc.c.o
cd /data/data/com.termux/files/home/ik_llama.cpp/ggml/src && ccache /data/data/com.termux/files/usr/bin/cc -DGGML_BUILD -DGGML_SCHED_MAX_COPIES=4 -DGGML_SHARED -DGGML_USE_IQK_MULMAT -DGGML_USE_LLAMAFILE -DGGML_USE_OPENMP -D_XOPEN_SOURCE=600 -Dggml_EXPORTS -I/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/../include -I/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/. -O2 -g -DNDEBUG -std=gnu11 -fPIC -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes -Werror=implicit-int -Werror=implicit-function-declaration -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wunreachable-code-break -Wunreachable-code-return -Wdouble-promotion -pthread -fopenmp=libomp -MD -MT ggml/src/CMakeFiles/ggml.dir/ggml-alloc.c.o -MF CMakeFiles/ggml.dir/ggml-alloc.c.o.d -o CMakeFiles/ggml.dir/ggml-alloc.c.o -c /data/data/com.termux/files/home/ik_llama.cpp/ggml/src/ggml-alloc.c
[  2%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml-backend.c.o
cd /data/data/com.termux/files/home/ik_llama.cpp/ggml/src && ccache /data/data/com.termux/files/usr/bin/cc -DGGML_BUILD -DGGML_SCHED_MAX_COPIES=4 -DGGML_SHARED -DGGML_USE_IQK_MULMAT -DGGML_USE_LLAMAFILE -DGGML_USE_OPENMP -D_XOPEN_SOURCE=600 -Dggml_EXPORTS -I/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/../include -I/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/. -O2 -g -DNDEBUG -std=gnu11 -fPIC -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes -Werror=implicit-int -Werror=implicit-function-declaration -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wunreachable-code-break -Wunreachable-code-return -Wdouble-promotion -pthread -fopenmp=libomp -MD -MT ggml/src/CMakeFiles/ggml.dir/ggml-backend.c.o -MF CMakeFiles/ggml.dir/ggml-backend.c.o.d -o CMakeFiles/ggml.dir/ggml-backend.c.o -c /data/data/com.termux/files/home/ik_llama.cpp/ggml/src/ggml-backend.c
[  3%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml-quants.c.o
cd /data/data/com.termux/files/home/ik_llama.cpp/ggml/src && ccache /data/data/com.termux/files/usr/bin/cc -DGGML_BUILD -DGGML_SCHED_MAX_COPIES=4 -DGGML_SHARED -DGGML_USE_IQK_MULMAT -DGGML_USE_LLAMAFILE -DGGML_USE_OPENMP -D_XOPEN_SOURCE=600 -Dggml_EXPORTS -I/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/../include -I/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/. -O2 -g -DNDEBUG -std=gnu11 -fPIC -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes -Werror=implicit-int -Werror=implicit-function-declaration -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wunreachable-code-break -Wunreachable-code-return -Wdouble-promotion -pthread -fopenmp=libomp -MD -MT ggml/src/CMakeFiles/ggml.dir/ggml-quants.c.o -MF CMakeFiles/ggml.dir/ggml-quants.c.o.d -o CMakeFiles/ggml.dir/ggml-quants.c.o -c /data/data/com.termux/files/home/ik_llama.cpp/ggml/src/ggml-quants.c
[  4%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml.c.o
cd /data/data/com.termux/files/home/ik_llama.cpp/ggml/src && ccache /data/data/com.termux/files/usr/bin/cc -DGGML_BUILD -DGGML_SCHED_MAX_COPIES=4 -DGGML_SHARED -DGGML_USE_IQK_MULMAT -DGGML_USE_LLAMAFILE -DGGML_USE_OPENMP -D_XOPEN_SOURCE=600 -Dggml_EXPORTS -I/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/../include -I/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/. -O2 -g -DNDEBUG -std=gnu11 -fPIC -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes -Werror=implicit-int -Werror=implicit-function-declaration -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wunreachable-code-break -Wunreachable-code-return -Wdouble-promotion -pthread -fopenmp=libomp -MD -MT ggml/src/CMakeFiles/ggml.dir/ggml.c.o -MF CMakeFiles/ggml.dir/ggml.c.o.d -o CMakeFiles/ggml.dir/ggml.c.o -c /data/data/com.termux/files/home/ik_llama.cpp/ggml/src/ggml.c
/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/ggml.c:2643:5: warning: implicit conversion increases floating-point precision: 'float32_t' (aka 'float') to 'ggml_float' (aka 'double') [-Wdouble-promotion]
 2643 |     GGML_F16_VEC_REDUCE(sumf, sum);
      |     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/ggml.c:1743:41: note: expanded from macro 'GGML_F16_VEC_REDUCE'
 1743 |     #define GGML_F16_VEC_REDUCE         GGML_F32Cx4_REDUCE
      |                                         ^
/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/ggml.c:1733:38: note: expanded from macro 'GGML_F32Cx4_REDUCE'
 1733 |     #define GGML_F32Cx4_REDUCE       GGML_F32x4_REDUCE
      |                                      ^
/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/ggml.c:1663:11: note: expanded from macro 'GGML_F32x4_REDUCE'
 1663 |     res = GGML_F32x4_REDUCE_ONE(x[0]);         \
      |         ~ ^~~~~~~~~~~~~~~~~~~~~~~~~~~
/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/ggml.c:1648:34: note: expanded from macro 'GGML_F32x4_REDUCE_ONE'
 1648 | #define GGML_F32x4_REDUCE_ONE(x) vaddvq_f32(x)
      |                                  ^~~~~~~~~~~~~
/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/ggml.c:2691:9: warning: implicit conversion increases floating-point precision: 'float32_t' (aka 'float') to 'ggml_float' (aka 'double') [-Wdouble-promotion]
 2691 |         GGML_F16_VEC_REDUCE(sumf[k], sum[k]);
      |         ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/ggml.c:1743:41: note: expanded from macro 'GGML_F16_VEC_REDUCE'
 1743 |     #define GGML_F16_VEC_REDUCE         GGML_F32Cx4_REDUCE
      |                                         ^
/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/ggml.c:1733:38: note: expanded from macro 'GGML_F32Cx4_REDUCE'
 1733 |     #define GGML_F32Cx4_REDUCE       GGML_F32x4_REDUCE
      |                                      ^
/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/ggml.c:1663:11: note: expanded from macro 'GGML_F32x4_REDUCE'
 1663 |     res = GGML_F32x4_REDUCE_ONE(x[0]);         \
      |         ~ ^~~~~~~~~~~~~~~~~~~~~~~~~~~
/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/ggml.c:1648:34: note: expanded from macro 'GGML_F32x4_REDUCE_ONE'
 1648 | #define GGML_F32x4_REDUCE_ONE(x) vaddvq_f32(x)
      |                                  ^~~~~~~~~~~~~
2 warnings generated.
[  5%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_mul_mat.cpp.o
cd /data/data/com.termux/files/home/ik_llama.cpp/ggml/src && ccache /data/data/com.termux/files/usr/bin/c++ -DGGML_BUILD -DGGML_SCHED_MAX_COPIES=4 -DGGML_SHARED -DGGML_USE_IQK_MULMAT -DGGML_USE_LLAMAFILE -DGGML_USE_OPENMP -D_XOPEN_SOURCE=600 -Dggml_EXPORTS -I/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/../include -I/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/. -O2 -g -DNDEBUG -std=gnu++17 -fPIC -Wmissing-declarations -Wmissing-noreturn -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wunreachable-code-break -Wunreachable-code-return -Wmissing-prototypes -Wextra-semi -pthread -fopenmp=libomp -MD -MT ggml/src/CMakeFiles/ggml.dir/iqk/iqk_mul_mat.cpp.o -MF CMakeFiles/ggml.dir/iqk/iqk_mul_mat.cpp.o.d -o CMakeFiles/ggml.dir/iqk/iqk_mul_mat.cpp.o -c /data/data/com.termux/files/home/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp
/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:14187:6: warning: no previous prototype for function 'iqk_mul_mat' [-Wmissing-prototypes]
 14187 | bool iqk_mul_mat(int, long, long, long, int, const void *, long, int, const void *, long, float *, long, int, int) {
       |      ^
/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:14187:1: note: declare 'static' if the function is not intended to be used outside of this translation unit
 14187 | bool iqk_mul_mat(int, long, long, long, int, const void *, long, int, const void *, long, float *, long, int, int) {
       | ^
       | static
/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:14191:6: warning: no previous prototype for function 'iqk_mul_mat_moe' [-Wmissing-prototypes]
 14191 | bool iqk_mul_mat_moe(long, long, long, int, int, const void *, long, int, const void *, long, float *, long, long,
       |      ^
/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:14191:1: note: declare 'static' if the function is not intended to be used outside of this translation unit
 14191 | bool iqk_mul_mat_moe(long, long, long, int, int, const void *, long, int, const void *, long, float *, long, long,
       | ^
       | static
/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:14196:6: warning: no previous prototype for function 'iqk_flash_attn_noalibi' [-Wmissing-prototypes]
 14196 | bool iqk_flash_attn_noalibi([[maybe_unused]] int int_type_k,         // type of k
       |      ^
/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:14196:1: note: declare 'static' if the function is not intended to be used outside of this translation unit
 14196 | bool iqk_flash_attn_noalibi([[maybe_unused]] int int_type_k,         // type of k
       | ^
       | static
3 warnings generated.
[  5%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_quantize.cpp.o
cd /data/data/com.termux/files/home/ik_llama.cpp/ggml/src && ccache /data/data/com.termux/files/usr/bin/c++ -DGGML_BUILD -DGGML_SCHED_MAX_COPIES=4 -DGGML_SHARED -DGGML_USE_IQK_MULMAT -DGGML_USE_LLAMAFILE -DGGML_USE_OPENMP -D_XOPEN_SOURCE=600 -Dggml_EXPORTS -I/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/../include -I/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/. -O2 -g -DNDEBUG -std=gnu++17 -fPIC -Wmissing-declarations -Wmissing-noreturn -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wunreachable-code-break -Wunreachable-code-return -Wmissing-prototypes -Wextra-semi -pthread -fopenmp=libomp -MD -MT ggml/src/CMakeFiles/ggml.dir/iqk/iqk_quantize.cpp.o -MF CMakeFiles/ggml.dir/iqk/iqk_quantize.cpp.o.d -o CMakeFiles/ggml.dir/iqk/iqk_quantize.cpp.o -c /data/data/com.termux/files/home/ik_llama.cpp/ggml/src/iqk/iqk_quantize.cpp
[  5%] Building CXX object ggml/src/CMakeFiles/ggml.dir/llamafile/sgemm.cpp.o
cd /data/data/com.termux/files/home/ik_llama.cpp/ggml/src && ccache /data/data/com.termux/files/usr/bin/c++ -DGGML_BUILD -DGGML_SCHED_MAX_COPIES=4 -DGGML_SHARED -DGGML_USE_IQK_MULMAT -DGGML_USE_LLAMAFILE -DGGML_USE_OPENMP -D_XOPEN_SOURCE=600 -Dggml_EXPORTS -I/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/../include -I/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/. -O2 -g -DNDEBUG -std=gnu++17 -fPIC -Wmissing-declarations -Wmissing-noreturn -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wunreachable-code-break -Wunreachable-code-return -Wmissing-prototypes -Wextra-semi -pthread -fopenmp=libomp -MD -MT ggml/src/CMakeFiles/ggml.dir/llamafile/sgemm.cpp.o -MF CMakeFiles/ggml.dir/llamafile/sgemm.cpp.o.d -o CMakeFiles/ggml.dir/llamafile/sgemm.cpp.o -c /data/data/com.termux/files/home/ik_llama.cpp/ggml/src/llamafile/sgemm.cpp
[  6%] Linking CXX shared library libggml.so
cd /data/data/com.termux/files/home/ik_llama.cpp/ggml/src && /data/data/com.termux/files/usr/bin/cmake -E cmake_link_script CMakeFiles/ggml.dir/link.txt --verbose=0
make[2]: Leaving directory '/data/data/com.termux/files/home/ik_llama.cpp'
[  6%] Built target ggml
make  -f src/CMakeFiles/llama.dir/build.make src/CMakeFiles/llama.dir/depend
make[2]: Entering directory '/data/data/com.termux/files/home/ik_llama.cpp'
cd /data/data/com.termux/files/home/ik_llama.cpp && /data/data/com.termux/files/usr/bin/cmake -E cmake_depends "Unix Makefiles" /data/data/com.termux/files/home/ik_llama.cpp /data/data/com.termux/files/home/ik_llama.cpp/src /data/data/com.termux/files/home/ik_llama.cpp /data/data/com.termux/files/home/ik_llama.cpp/src /data/data/com.termux/files/home/ik_llama.cpp/src/CMakeFiles/llama.dir/DependInfo.cmake "--color="
make[2]: Leaving directory '/data/data/com.termux/files/home/ik_llama.cpp'
make  -f src/CMakeFiles/llama.dir/build.make src/CMakeFiles/llama.dir/build
make[2]: Entering directory '/data/data/com.termux/files/home/ik_llama.cpp'
[  6%] Building CXX object src/CMakeFiles/llama.dir/llama-grammar.cpp.o
cd /data/data/com.termux/files/home/ik_llama.cpp/src && ccache /data/data/com.termux/files/usr/bin/c++ -DLLAMA_BUILD -DLLAMA_SHARED -Dllama_EXPORTS -I/data/data/com.termux/files/home/ik_llama.cpp/src/. -I/data/data/com.termux/files/home/ik_llama.cpp/src/../include -I/data/data/com.termux/files/home/ik_llama.cpp/src/../ggml/src -I/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/../include -O2 -g -DNDEBUG -std=gnu++17 -fPIC -MD -MT src/CMakeFiles/llama.dir/llama-grammar.cpp.o -MF CMakeFiles/llama.dir/llama-grammar.cpp.o.d -o CMakeFiles/llama.dir/llama-grammar.cpp.o -c /data/data/com.termux/files/home/ik_llama.cpp/src/llama-grammar.cpp
[  7%] Building CXX object src/CMakeFiles/llama.dir/llama-sampling.cpp.o
cd /data/data/com.termux/files/home/ik_llama.cpp/src && ccache /data/data/com.termux/files/usr/bin/c++ -DLLAMA_BUILD -DLLAMA_SHARED -Dllama_EXPORTS -I/data/data/com.termux/files/home/ik_llama.cpp/src/. -I/data/data/com.termux/files/home/ik_llama.cpp/src/../include -I/data/data/com.termux/files/home/ik_llama.cpp/src/../ggml/src -I/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/../include -O2 -g -DNDEBUG -std=gnu++17 -fPIC -MD -MT src/CMakeFiles/llama.dir/llama-sampling.cpp.o -MF CMakeFiles/llama.dir/llama-sampling.cpp.o.d -o CMakeFiles/llama.dir/llama-sampling.cpp.o -c /data/data/com.termux/files/home/ik_llama.cpp/src/llama-sampling.cpp
[  8%] Building CXX object src/CMakeFiles/llama.dir/llama-vocab.cpp.o
cd /data/data/com.termux/files/home/ik_llama.cpp/src && ccache /data/data/com.termux/files/usr/bin/c++ -DLLAMA_BUILD -DLLAMA_SHARED -Dllama_EXPORTS -I/data/data/com.termux/files/home/ik_llama.cpp/src/. -I/data/data/com.termux/files/home/ik_llama.cpp/src/../include -I/data/data/com.termux/files/home/ik_llama.cpp/src/../ggml/src -I/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/../include -O2 -g -DNDEBUG -std=gnu++17 -fPIC -MD -MT src/CMakeFiles/llama.dir/llama-vocab.cpp.o -MF CMakeFiles/llama.dir/llama-vocab.cpp.o.d -o CMakeFiles/llama.dir/llama-vocab.cpp.o -c /data/data/com.termux/files/home/ik_llama.cpp/src/llama-vocab.cpp
[  8%] Building CXX object src/CMakeFiles/llama.dir/llama.cpp.o
cd /data/data/com.termux/files/home/ik_llama.cpp/src && ccache /data/data/com.termux/files/usr/bin/c++ -DLLAMA_BUILD -DLLAMA_SHARED -Dllama_EXPORTS -I/data/data/com.termux/files/home/ik_llama.cpp/src/. -I/data/data/com.termux/files/home/ik_llama.cpp/src/../include -I/data/data/com.termux/files/home/ik_llama.cpp/src/../ggml/src -I/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/../include -O2 -g -DNDEBUG -std=gnu++17 -fPIC -MD -MT src/CMakeFiles/llama.dir/llama.cpp.o -MF CMakeFiles/llama.dir/llama.cpp.o.d -o CMakeFiles/llama.dir/llama.cpp.o -c /data/data/com.termux/files/home/ik_llama.cpp/src/llama.cpp
[  8%] Building CXX object src/CMakeFiles/llama.dir/unicode-data.cpp.o
cd /data/data/com.termux/files/home/ik_llama.cpp/src && ccache /data/data/com.termux/files/usr/bin/c++ -DLLAMA_BUILD -DLLAMA_SHARED -Dllama_EXPORTS -I/data/data/com.termux/files/home/ik_llama.cpp/src/. -I/data/data/com.termux/files/home/ik_llama.cpp/src/../include -I/data/data/com.termux/files/home/ik_llama.cpp/src/../ggml/src -I/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/../include -O2 -g -DNDEBUG -std=gnu++17 -fPIC -MD -MT src/CMakeFiles/llama.dir/unicode-data.cpp.o -MF CMakeFiles/llama.dir/unicode-data.cpp.o.d -o CMakeFiles/llama.dir/unicode-data.cpp.o -c /data/data/com.termux/files/home/ik_llama.cpp/src/unicode-data.cpp
[  9%] Building CXX object src/CMakeFiles/llama.dir/unicode.cpp.o
cd /data/data/com.termux/files/home/ik_llama.cpp/src && ccache /data/data/com.termux/files/usr/bin/c++ -DLLAMA_BUILD -DLLAMA_SHARED -Dllama_EXPORTS -I/data/data/com.termux/files/home/ik_llama.cpp/src/. -I/data/data/com.termux/files/home/ik_llama.cpp/src/../include -I/data/data/com.termux/files/home/ik_llama.cpp/src/../ggml/src -I/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/../include -O2 -g -DNDEBUG -std=gnu++17 -fPIC -MD -MT src/CMakeFiles/llama.dir/unicode.cpp.o -MF CMakeFiles/llama.dir/unicode.cpp.o.d -o CMakeFiles/llama.dir/unicode.cpp.o -c /data/data/com.termux/files/home/ik_llama.cpp/src/unicode.cpp
/data/data/com.termux/files/home/ik_llama.cpp/src/unicode.cpp:203:31: warning: 'codecvt_utf8<wchar_t>' is deprecated [-Wdeprecated-declarations]
  203 |     std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
      |                               ^
/data/data/com.termux/files/usr/include/c++/v1/codecvt:194:28: note: 'codecvt_utf8<wchar_t>' has been explicitly marked deprecated here
  194 | class _LIBCPP_TEMPLATE_VIS _LIBCPP_DEPRECATED_IN_CXX17 codecvt_utf8 : public __codecvt_utf8<_Elem> {
      |                            ^
/data/data/com.termux/files/usr/include/c++/v1/__config:942:41: note: expanded from macro '_LIBCPP_DEPRECATED_IN_CXX17'
  942 | #    define _LIBCPP_DEPRECATED_IN_CXX17 _LIBCPP_DEPRECATED
      |                                         ^
/data/data/com.termux/files/usr/include/c++/v1/__config:915:49: note: expanded from macro '_LIBCPP_DEPRECATED'
  915 | #      define _LIBCPP_DEPRECATED __attribute__((__deprecated__))
      |                                                 ^
/data/data/com.termux/files/home/ik_llama.cpp/src/unicode.cpp:203:10: warning: 'wstring_convert<std::codecvt_utf8<wchar_t>>' is deprecated [-Wdeprecated-declarations]
  203 |     std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
      |          ^
/data/data/com.termux/files/usr/include/c++/v1/locale:3114:28: note: 'wstring_convert<std::codecvt_utf8<wchar_t>>' has been explicitly marked deprecated here
 3114 | class _LIBCPP_TEMPLATE_VIS _LIBCPP_DEPRECATED_IN_CXX17 wstring_convert {
      |                            ^
/data/data/com.termux/files/usr/include/c++/v1/__config:942:41: note: expanded from macro '_LIBCPP_DEPRECATED_IN_CXX17'
  942 | #    define _LIBCPP_DEPRECATED_IN_CXX17 _LIBCPP_DEPRECATED
      |                                         ^
/data/data/com.termux/files/usr/include/c++/v1/__config:915:49: note: expanded from macro '_LIBCPP_DEPRECATED'
  915 | #      define _LIBCPP_DEPRECATED __attribute__((__deprecated__))
      |                                                 ^
2 warnings generated.
[ 10%] Linking CXX shared library libllama.so
cd /data/data/com.termux/files/home/ik_llama.cpp/src && /data/data/com.termux/files/usr/bin/cmake -E cmake_link_script CMakeFiles/llama.dir/link.txt --verbose=0
make[2]: Leaving directory '/data/data/com.termux/files/home/ik_llama.cpp'
[ 10%] Built target llama
make  -f common/CMakeFiles/build_info.dir/build.make common/CMakeFiles/build_info.dir/depend
make[2]: Entering directory '/data/data/com.termux/files/home/ik_llama.cpp'
[ 10%] Generating build details from Git
/data/data/com.termux/files/usr/bin/cmake -DMSVC= -DCMAKE_C_COMPILER_VERSION=19.1.6 -DCMAKE_C_COMPILER_ID=Clang -DCMAKE_VS_PLATFORM_NAME= -DCMAKE_C_COMPILER=/data/data/com.termux/files/usr/bin/cc -P /data/data/com.termux/files/home/ik_llama.cpp/common/cmake/build-info-gen-cpp.cmake
-- Found Git: /data/data/com.termux/files/usr/bin/git (found version "2.47.1")
cd /data/data/com.termux/files/home/ik_llama.cpp && /data/data/com.termux/files/usr/bin/cmake -E cmake_depends "Unix Makefiles" /data/data/com.termux/files/home/ik_llama.cpp /data/data/com.termux/files/home/ik_llama.cpp/common /data/data/com.termux/files/home/ik_llama.cpp /data/data/com.termux/files/home/ik_llama.cpp/common /data/data/com.termux/files/home/ik_llama.cpp/common/CMakeFiles/build_info.dir/DependInfo.cmake "--color="
make[2]: Leaving directory '/data/data/com.termux/files/home/ik_llama.cpp'
make  -f common/CMakeFiles/build_info.dir/build.make common/CMakeFiles/build_info.dir/build
make[2]: Entering directory '/data/data/com.termux/files/home/ik_llama.cpp'
[ 11%] Building CXX object common/CMakeFiles/build_info.dir/build-info.cpp.o
cd /data/data/com.termux/files/home/ik_llama.cpp/common && ccache /data/data/com.termux/files/usr/bin/c++   -O2 -g -DNDEBUG -std=gnu++17 -fPIC -MD -MT common/CMakeFiles/build_info.dir/build-info.cpp.o -MF CMakeFiles/build_info.dir/build-info.cpp.o.d -o CMakeFiles/build_info.dir/build-info.cpp.o -c /data/data/com.termux/files/home/ik_llama.cpp/common/build-info.cpp
make[2]: Leaving directory '/data/data/com.termux/files/home/ik_llama.cpp'
[ 11%] Built target build_info
make  -f common/CMakeFiles/common.dir/build.make common/CMakeFiles/common.dir/depend
make[2]: Entering directory '/data/data/com.termux/files/home/ik_llama.cpp'
cd /data/data/com.termux/files/home/ik_llama.cpp && /data/data/com.termux/files/usr/bin/cmake -E cmake_depends "Unix Makefiles" /data/data/com.termux/files/home/ik_llama.cpp /data/data/com.termux/files/home/ik_llama.cpp/common /data/data/com.termux/files/home/ik_llama.cpp /data/data/com.termux/files/home/ik_llama.cpp/common /data/data/com.termux/files/home/ik_llama.cpp/common/CMakeFiles/common.dir/DependInfo.cmake "--color="
make[2]: Leaving directory '/data/data/com.termux/files/home/ik_llama.cpp'
make  -f common/CMakeFiles/common.dir/build.make common/CMakeFiles/common.dir/build
make[2]: Entering directory '/data/data/com.termux/files/home/ik_llama.cpp'
[ 11%] Building CXX object common/CMakeFiles/common.dir/common.cpp.o
cd /data/data/com.termux/files/home/ik_llama.cpp/common && ccache /data/data/com.termux/files/usr/bin/c++  -I/data/data/com.termux/files/home/ik_llama.cpp/common/. -I/data/data/com.termux/files/home/ik_llama.cpp/src/. -I/data/data/com.termux/files/home/ik_llama.cpp/src/../include -I/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/../include -O2 -g -DNDEBUG -std=gnu++17 -fPIC -pthread -MD -MT common/CMakeFiles/common.dir/common.cpp.o -MF CMakeFiles/common.dir/common.cpp.o.d -o CMakeFiles/common.dir/common.cpp.o -c /data/data/com.termux/files/home/ik_llama.cpp/common/common.cpp
/data/data/com.termux/files/home/ik_llama.cpp/common/common.cpp:1913:35: warning: 'codecvt_utf8<char32_t>' is deprecated [-Wdeprecated-declarations]
 1913 |         std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
      |                                   ^
/data/data/com.termux/files/usr/include/c++/v1/codecvt:194:28: note: 'codecvt_utf8<char32_t>' has been explicitly marked deprecated here
  194 | class _LIBCPP_TEMPLATE_VIS _LIBCPP_DEPRECATED_IN_CXX17 codecvt_utf8 : public __codecvt_utf8<_Elem> {
      |                            ^
/data/data/com.termux/files/usr/include/c++/v1/__config:942:41: note: expanded from macro '_LIBCPP_DEPRECATED_IN_CXX17'
  942 | #    define _LIBCPP_DEPRECATED_IN_CXX17 _LIBCPP_DEPRECATED
      |                                         ^
/data/data/com.termux/files/usr/include/c++/v1/__config:915:49: note: expanded from macro '_LIBCPP_DEPRECATED'
  915 | #      define _LIBCPP_DEPRECATED __attribute__((__deprecated__))
      |                                                 ^
/data/data/com.termux/files/home/ik_llama.cpp/common/common.cpp:1913:14: warning: 'wstring_convert<std::codecvt_utf8<char32_t>, char32_t>' is deprecated [-Wdeprecated-declarations]
 1913 |         std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
      |              ^
/data/data/com.termux/files/usr/include/c++/v1/locale:3114:28: note: 'wstring_convert<std::codecvt_utf8<char32_t>, char32_t>' has been explicitly marked deprecated here
 3114 | class _LIBCPP_TEMPLATE_VIS _LIBCPP_DEPRECATED_IN_CXX17 wstring_convert {
      |                            ^
/data/data/com.termux/files/usr/include/c++/v1/__config:942:41: note: expanded from macro '_LIBCPP_DEPRECATED_IN_CXX17'
  942 | #    define _LIBCPP_DEPRECATED_IN_CXX17 _LIBCPP_DEPRECATED
      |                                         ^
/data/data/com.termux/files/usr/include/c++/v1/__config:915:49: note: expanded from macro '_LIBCPP_DEPRECATED'
  915 | #      define _LIBCPP_DEPRECATED __attribute__((__deprecated__))
      |                                                 ^
In file included from /data/data/com.termux/files/home/ik_llama.cpp/common/common.cpp:5:
In file included from /data/data/com.termux/files/home/ik_llama.cpp/common/common.h:7:
In file included from /data/data/com.termux/files/home/ik_llama.cpp/common/sampling.h:5:
In file included from /data/data/com.termux/files/home/ik_llama.cpp/common/grammar-parser.h:14:
In file included from /data/data/com.termux/files/usr/include/c++/v1/vector:325:
In file included from /data/data/com.termux/files/usr/include/c++/v1/__format/formatter_bool.h:20:
In file included from /data/data/com.termux/files/usr/include/c++/v1/__format/formatter_integral.h:35:
/data/data/com.termux/files/usr/include/c++/v1/locale:3257:1: warning: 'wstring_convert<std::codecvt_utf8<char32_t>, char32_t>' is deprecated [-Wdeprecated-declarations]
 3257 | wstring_convert<_Codecvt, _Elem, _WideAlloc, _ByteAlloc>::to_bytes(const _Elem* __frm, const _Elem* __frm_end) {
      | ^
/data/data/com.termux/files/usr/include/c++/v1/locale:3161:12: note: in instantiation of member function 'std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t>::to_bytes' requested here
 3161 |     return to_bytes(__wstr.data(), __wstr.data() + __wstr.size());
      |            ^
/data/data/com.termux/files/home/ik_llama.cpp/common/common.cpp:1918:52: note: in instantiation of member function 'std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t>::to_bytes' requested here
 1918 |         std::string filename_reencoded = converter.to_bytes(filename_utf32);
      |                                                    ^
/data/data/com.termux/files/usr/include/c++/v1/locale:3114:28: note: 'wstring_convert<std::codecvt_utf8<char32_t>, char32_t>' has been explicitly marked deprecated here
 3114 | class _LIBCPP_TEMPLATE_VIS _LIBCPP_DEPRECATED_IN_CXX17 wstring_convert {
      |                            ^
/data/data/com.termux/files/usr/include/c++/v1/__config:942:41: note: expanded from macro '_LIBCPP_DEPRECATED_IN_CXX17'
  942 | #    define _LIBCPP_DEPRECATED_IN_CXX17 _LIBCPP_DEPRECATED
      |                                         ^
/data/data/com.termux/files/usr/include/c++/v1/__config:915:49: note: expanded from macro '_LIBCPP_DEPRECATED'
  915 | #      define _LIBCPP_DEPRECATED __attribute__((__deprecated__))
      |                                                 ^
3 warnings generated.
[ 12%] Building CXX object common/CMakeFiles/common.dir/sampling.cpp.o
cd /data/data/com.termux/files/home/ik_llama.cpp/common && ccache /data/data/com.termux/files/usr/bin/c++  -I/data/data/com.termux/files/home/ik_llama.cpp/common/. -I/data/data/com.termux/files/home/ik_llama.cpp/src/. -I/data/data/com.termux/files/home/ik_llama.cpp/src/../include -I/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/../include -O2 -g -DNDEBUG -std=gnu++17 -fPIC -pthread -MD -MT common/CMakeFiles/common.dir/sampling.cpp.o -MF CMakeFiles/common.dir/sampling.cpp.o.d -o CMakeFiles/common.dir/sampling.cpp.o -c /data/data/com.termux/files/home/ik_llama.cpp/common/sampling.cpp
[ 12%] Building CXX object common/CMakeFiles/common.dir/console.cpp.o
cd /data/data/com.termux/files/home/ik_llama.cpp/common && ccache /data/data/com.termux/files/usr/bin/c++  -I/data/data/com.termux/files/home/ik_llama.cpp/common/. -I/data/data/com.termux/files/home/ik_llama.cpp/src/. -I/data/data/com.termux/files/home/ik_llama.cpp/src/../include -I/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/../include -O2 -g -DNDEBUG -std=gnu++17 -fPIC -pthread -MD -MT common/CMakeFiles/common.dir/console.cpp.o -MF CMakeFiles/common.dir/console.cpp.o.d -o CMakeFiles/common.dir/console.cpp.o -c /data/data/com.termux/files/home/ik_llama.cpp/common/console.cpp
[ 13%] Building CXX object common/CMakeFiles/common.dir/grammar-parser.cpp.o
cd /data/data/com.termux/files/home/ik_llama.cpp/common && ccache /data/data/com.termux/files/usr/bin/c++  -I/data/data/com.termux/files/home/ik_llama.cpp/common/. -I/data/data/com.termux/files/home/ik_llama.cpp/src/. -I/data/data/com.termux/files/home/ik_llama.cpp/src/../include -I/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/../include -O2 -g -DNDEBUG -std=gnu++17 -fPIC -pthread -MD -MT common/CMakeFiles/common.dir/grammar-parser.cpp.o -MF CMakeFiles/common.dir/grammar-parser.cpp.o.d -o CMakeFiles/common.dir/grammar-parser.cpp.o -c /data/data/com.termux/files/home/ik_llama.cpp/common/grammar-parser.cpp
[ 14%] Building CXX object common/CMakeFiles/common.dir/json-schema-to-grammar.cpp.o
cd /data/data/com.termux/files/home/ik_llama.cpp/common && ccache /data/data/com.termux/files/usr/bin/c++  -I/data/data/com.termux/files/home/ik_llama.cpp/common/. -I/data/data/com.termux/files/home/ik_llama.cpp/src/. -I/data/data/com.termux/files/home/ik_llama.cpp/src/../include -I/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/../include -O2 -g -DNDEBUG -std=gnu++17 -fPIC -pthread -MD -MT common/CMakeFiles/common.dir/json-schema-to-grammar.cpp.o -MF CMakeFiles/common.dir/json-schema-to-grammar.cpp.o.d -o CMakeFiles/common.dir/json-schema-to-grammar.cpp.o -c /data/data/com.termux/files/home/ik_llama.cpp/common/json-schema-to-grammar.cpp
[ 14%] Building CXX object common/CMakeFiles/common.dir/train.cpp.o
cd /data/data/com.termux/files/home/ik_llama.cpp/common && ccache /data/data/com.termux/files/usr/bin/c++  -I/data/data/com.termux/files/home/ik_llama.cpp/common/. -I/data/data/com.termux/files/home/ik_llama.cpp/src/. -I/data/data/com.termux/files/home/ik_llama.cpp/src/../include -I/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/../include -O2 -g -DNDEBUG -std=gnu++17 -fPIC -pthread -MD -MT common/CMakeFiles/common.dir/train.cpp.o -MF CMakeFiles/common.dir/train.cpp.o.d -o CMakeFiles/common.dir/train.cpp.o -c /data/data/com.termux/files/home/ik_llama.cpp/common/train.cpp
[ 15%] Building CXX object common/CMakeFiles/common.dir/ngram-cache.cpp.o
cd /data/data/com.termux/files/home/ik_llama.cpp/common && ccache /data/data/com.termux/files/usr/bin/c++  -I/data/data/com.termux/files/home/ik_llama.cpp/common/. -I/data/data/com.termux/files/home/ik_llama.cpp/src/. -I/data/data/com.termux/files/home/ik_llama.cpp/src/../include -I/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/../include -O2 -g -DNDEBUG -std=gnu++17 -fPIC -pthread -MD -MT common/CMakeFiles/common.dir/ngram-cache.cpp.o -MF CMakeFiles/common.dir/ngram-cache.cpp.o.d -o CMakeFiles/common.dir/ngram-cache.cpp.o -c /data/data/com.termux/files/home/ik_llama.cpp/common/ngram-cache.cpp
[ 15%] Linking CXX static library libcommon.a
cd /data/data/com.termux/files/home/ik_llama.cpp/common && /data/data/com.termux/files/usr/bin/cmake -P CMakeFiles/common.dir/cmake_clean_target.cmake
cd /data/data/com.termux/files/home/ik_llama.cpp/common && /data/data/com.termux/files/usr/bin/cmake -E cmake_link_script CMakeFiles/common.dir/link.txt --verbose=0
make[2]: Leaving directory '/data/data/com.termux/files/home/ik_llama.cpp'
[ 15%] Built target common
make  -f tests/CMakeFiles/test-tokenizer-0.dir/build.make tests/CMakeFiles/test-tokenizer-0.dir/depend
make[2]: Entering directory '/data/data/com.termux/files/home/ik_llama.cpp'
cd /data/data/com.termux/files/home/ik_llama.cpp && /data/data/com.termux/files/usr/bin/cmake -E cmake_depends "Unix Makefiles" /data/data/com.termux/files/home/ik_llama.cpp /data/data/com.termux/files/home/ik_llama.cpp/tests /data/data/com.termux/files/home/ik_llama.cpp /data/data/com.termux/files/home/ik_llama.cpp/tests /data/data/com.termux/files/home/ik_llama.cpp/tests/CMakeFiles/test-tokenizer-0.dir/DependInfo.cmake "--color="
make[2]: Leaving directory '/data/data/com.termux/files/home/ik_llama.cpp'
make  -f tests/CMakeFiles/test-tokenizer-0.dir/build.make tests/CMakeFiles/test-tokenizer-0.dir/build
make[2]: Entering directory '/data/data/com.termux/files/home/ik_llama.cpp'
[ 16%] Building CXX object tests/CMakeFiles/test-tokenizer-0.dir/test-tokenizer-0.cpp.o
cd /data/data/com.termux/files/home/ik_llama.cpp/tests && ccache /data/data/com.termux/files/usr/bin/c++  -I/data/data/com.termux/files/home/ik_llama.cpp/common/. -I/data/data/com.termux/files/home/ik_llama.cpp/src/. -I/data/data/com.termux/files/home/ik_llama.cpp/src/../include -I/data/data/com.termux/files/home/ik_llama.cpp/ggml/src/../include -O2 -g -DNDEBUG -std=gnu++17 -pthread -MD -MT tests/CMakeFiles/test-tokenizer-0.dir/test-tokenizer-0.cpp.o -MF CMakeFiles/test-tokenizer-0.dir/test-tokenizer-0.cpp.o.d -o CMakeFiles/test-tokenizer-0.dir/test-tokenizer-0.cpp.o -c /data/data/com.termux/files/home/ik_llama.cpp/tests/test-tokenizer-0.cpp
[ 17%] Linking CXX executable ../bin/test-tokenizer-0
cd /data/data/com.termux/files/home/ik_llama.cpp/tests && /data/data/com.termux/files/usr/bin/cmake -E cmake_link_script CMakeFiles/test-tokenizer-0.dir/link.txt --verbose=0
ld.lld: error: undefined reference: iqk_mul_mat
>>> referenced by ../ggml/src/libggml.so (disallowed by --no-allow-shlib-undefined)

ld.lld: error: undefined reference: iqk_mul_mat_moe
>>> referenced by ../ggml/src/libggml.so (disallowed by --no-allow-shlib-undefined)

ld.lld: error: undefined reference: iqk_flash_attn_noalibi
>>> referenced by ../ggml/src/libggml.so (disallowed by --no-allow-shlib-undefined)
c++: error: linker command failed with exit code 1 (use -v to see invocation)
make[2]: *** [tests/CMakeFiles/test-tokenizer-0.dir/build.make:104: bin/test-tokenizer-0] Error 1
make[2]: Leaving directory '/data/data/com.termux/files/home/ik_llama.cpp'
make[1]: *** [CMakeFiles/Makefile2:2132: tests/CMakeFiles/test-tokenizer-0.dir/all] Error 2
make[1]: Leaving directory '/data/data/com.termux/files/home/ik_llama.cpp'
make: *** [Makefile:146: all] Error 2

---

👤 **ajiekc905** commented the **2024-12-28** at **01:17:18**:<br>

I could be wrong but it looks like iqk_mul_mat not properly compiled and/or linked.  error: undefined reference: iqk_mul_mat_moe, error: undefined reference: iqk_flash_attn_noalibi, error: undefined reference: iqk_mul_mat

---

👤 **ikawrakow** commented the **2024-12-28** at **17:27:54**:<br>

I'm travelling without my laptop to dig in deeper, but perhaps adding `-DGGML_NATIVE=1` to cmake could help.