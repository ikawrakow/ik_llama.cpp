### 🐛 [#30](https://github.com/ikawrakow/ik_llama.cpp/issues/30) - Bug: Appcrash on Windows 7 with GGML_USE_IQK_MULMAT

| **Author** | `whoreson` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-08-30 |
| **Updated** | 2024-09-19 |

---

#### Description

### What happened?

Trying latest HEAD with: Fimbulvetr-11B-v2-Q8_0.gguf (or L3-8B-Stheno-v3.1-Q8_0-imat.gguf, or SFR-Iterative-DPO-LLaMA-3-8B-R-Q8_0.gguf for example):
```
llama_new_context_with_model: graph nodes  = 1542
llama_new_context_with_model: graph splits = 1
[New Thread 5152.0x115c]
[New Thread 5152.0xc44]
[New Thread 5152.0x99c]

Thread 3 received signal SIGSEGV, Segmentation fault.
[Switching to Thread 5152.0xc44]
0x00000000004118f6 in (anonymous namespace)::Sum4<block_q8_0, block_q8_0_x4, (an
onymous namespace)::SignedDot, false>::compute(long long __vector(4) const*, blo
ck_q8_0 const*) const ()
(gdb) bt
#0  0x00000000004118f6 in (anonymous namespace)::Sum4<block_q8_0, block_q8_0_x4,
 (anonymous namespace)::SignedDot, false>::compute(long long __vector(4) const*,
 block_q8_0 const*) const ()
#1  0x0000000000431dcd in void (anonymous namespace)::mul_mat_qX_q8_Helper<(anon
ymous namespace)::Q8_0_Unpacker, (anonymous namespace)::AccumT<(anonymous namesp
ace)::MinusType0, 2, true>, (anonymous namespace)::ScaleHelperQ8_0, block_q8_0,
2>(int, void const*, unsigned long long, (anonymous namespace)::DataInfo const&,
 block_q8_0 const**, int) ()
#2  0x000000000045319a in void (anonymous namespace)::mul_mat_qX_0_q8_0_T<(anony
mous namespace)::Q8_0_Unpacker, 2>(int, void const*, unsigned long long, (anonym
ous namespace)::DataInfo const&, int) ()
#3  0x000000000040f9fa in (anonymous namespace)::MulMat::mul_mat_NxM(int, void c
onst*, unsigned long long, (anonymous namespace)::DataInfo&, int, int) ()
#4  0x00000000004a1a3e in iqk_mul_mat ()
#5  0x00000000004dda7e in ggml_compute_forward_mul_mat (params=0x4844fda0,
    dst=0x347e1250) at ggml/src/ggml.c:12973
#6  0x00000000004ef622 in ggml_compute_forward (params=0x4844fda0,
    tensor=0x347e1250) at ggml/src/ggml.c:17689
#7  0x00000000004f478d in ggml_graph_compute_thread (data=0x4844fe20)
    at ggml/src/ggml.c:19765
#8  0x00000000004ffddb in ggml_graph_compute._omp_fn.0 ()
    at ggml/src/ggml.c:19816
#9  0x000000006360cf98 in omp_in_final ()
   from C:\util\Strawberry\c\bin\libgomp-1.dll
```

Crashes here without even trying to load and malloc the GGUF. After disabling this code block:
```ggml.c:12967
#if GGML_USE_IQK_MULMAT
    if (src1->type != vec_dot_type && dst->type == GGML_TYPE_F32) {
        const size_t row_size = ggml_row_size(vec_dot_type, ne10);
        for (int64_t i13 = 0; i13 < ne13; i13++)
            for (int64_t i12 = 0; i12 < ne12; i12++)
                if (!iqk_mul_mat(ne01, ne11, ne00,
                            src0->type, (const char *)src0->data + i12/r2*nb02 +
                            vec_dot_type, (const char *)wdata + (i12*ne11 + i13*
                            (float *)((char *)dst->data + i12*nb2 + i13*nb3), nb
                            ith, nth)) goto IQK_MulMat_Not_Available2;
        return;
    }
IQK_MulMat_Not_Available2:;
#endif
```

... seems to make it work with these files, but still crashes with Fimbulvetr Q4_1. Works with stable-code-3b-q5_k_m.gguf even without any modification, though. Also everything works on Linux. This is a Win7 PC with Strawberry Perl's gcc version 8.3.0 (x86_64-posix-seh, Built by strawberryperl.com project).

Stock llama.cpp works.

Seems really weird, any hints on debugging this?

### Name and Version

c7e99c88a2de7489ba2a1539b1a9025912010b70

### What operating system are you seeing the problem on?

Windows

### Relevant log output

_No response_

---

#### 💬 Conversation

👤 **whoreson** commented the **2024-08-30** at **20:30:11**:<br>

Q4_1 crash backtrace:
```
llama_new_context_with_model: graph splits = 1
[New Thread 5064.0x680]
[New Thread 5064.0x5a8]
[New Thread 5064.0x1268]

Thread 2 received signal SIGSEGV, Segmentation fault.
[Switching to Thread 5064.0x680]
quantize_row_q8_1 (x=0x367058c0, vy=0x37e0ca0, k=4096)
    at ggml/src/ggml-quants.c:1397
1397                y4[i4].d[ir+4] = GGML_FP32_TO_FP16(d * hsum_i32_8(_mm256_add
_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3))));
(gdb) bt
#0  quantize_row_q8_1 (x=0x367058c0, vy=0x37e0ca0, k=4096)
    at ggml/src/ggml-quants.c:1397
#1  0x00000000004dd7c9 in ggml_compute_forward_mul_mat (params=0x4810fda0,
    dst=0x346a1250) at ggml/src/ggml.c:12945
#2  0x00000000004ef622 in ggml_compute_forward (params=0x4810fda0,
    tensor=0x346a1250) at ggml/src/ggml.c:17689
#3  0x00000000004f478d in ggml_graph_compute_thread (data=0x4810fe20)
    at ggml/src/ggml.c:19765
#4  0x00000000004ffddb in ggml_graph_compute._omp_fn.0 ()
    at ggml/src/ggml.c:19816
#5  0x000000006360cf98 in omp_in_final ()
   from C:\util\Strawberry\c\bin\libgomp-1.dll
#6  0x0000000064944ae4 in pthread_create_wrapper ()
   from C:\util\Strawberry\c\bin\libwinpthread-1.dll
#7  0x000007fefd2d42bf in sqrt () from C:\Windows\system32\msvcrt.dll
#8  0x000007fefd2d7459 in msvcrt!_beginthreadex ()
   from C:\Windows\system32\msvcrt.dll
#9  0x0000000076da652d in KERNEL32!BaseThreadInitThunk ()
   from C:\Windows\system32\kernel32.dll
#10 0x0000000076fdc521 in ntdll!RtlUserThreadStart ()
   from C:\Windows\SYSTEM32\ntdll.dll
#11 0x0000000000000000 in ?? ()
Backtrace stopped: previous frame inner to this frame (corrupt stack?)
```

Seems to be different perhaps?.. Still, works with stock llama.cpp.

---

👤 **ikawrakow** commented the **2024-08-31** at **05:59:09**:<br>

Can you post your `system_info` message when these crashes happen? It should look something like this
```
system_info: n_threads = 16 / 32 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 |
```

Thanks!

---

👤 **whoreson** commented the **2024-08-31** at **08:22:16**:<br>

```
INFO [                    main] system info | tid="1" timestamp=1725092503 n_thr
eads=4 n_threads_batch=-1 total_threads=4 system_info="AVX = 1 | AVX_VNNI = 0 |
AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FM
A = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD =
0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1
```

---

👤 **ikawrakow** commented the **2024-08-31** at **10:50:07**:<br>

I was suspecting something I might have missed between `AVX2` and `AVX`, but no, you have `AVX2`.

I have no access to a Windows box, and even less to Windows 7 with GCC 8.3, so not sure how to debug.

With the second crash you posted a bt (the one during quantization), what are the values of `k`, `nb`, `i4` and `ir`?

---

👤 **whoreson** commented the **2024-08-31** at **11:56:33**:<br>

Hmm no, all of these are results of llama-cli, not quantize.

```
1397                y4[i4].d[ir+4] = GGML_FP32_TO_FP16(d * hsum_i32_8(_mm256_add
_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3))));
(gdb) p k
$1 = 4096
(gdb) p nb
$2 = 128
(gdb) p i4
$3 = 0
(gdb) p ir
$4 = 0
```

---

👤 **ikawrakow** commented the **2024-08-31** at **12:22:17**:<br>

Then `y4` must be `null`?

---

👤 **whoreson** commented the **2024-08-31** at **14:33:20**:<br>

```
(gdb) p y4
$5 = (block_q8_1_x4 * restrict) 0x3870ca0
```

---

👤 **ikawrakow** commented the **2024-08-31** at **15:57:55**:<br>

So
* `y4` is not null
* We attempt to store data into bytes `12...16` of the memory block pointed to by `y4`. The memory block is 4608 bytes (the row size of `Q8_1`-quantized tensor row with 4096 elements), so we are not having an out-of-bounds access
* We get `SIGSEGV`, so we are attempting to write to memory not accessible to us
* Hence, `y4` is somehow pointing to outside of our process address space
* As this is not possible to happen in this specific function, there are two options
  - We overwrote memory somewhere else, thus corrupting the pointer passed into the crashing function. A bug like this can only be meaningfully debugged with an address sanitizer or `valgrind`. Is one of those available on this Windows box?
  - GCC miscompiled the code. You mention that the program sometimes crashes even before loading the model, so this kind of supports this possibility

---

👤 **whoreson** commented the **2024-08-31** at **19:21:42**:<br>

Ehm, looks like it's not gonna be that easy... Just tried with TDM-GCC's gcc version 10.3.0 (tdm64-1), and the results are the same.

---

👤 **whoreson** commented the **2024-08-31** at **19:29:10**:<br>

Hmm... Could it be related that I've been disabling the -muse-unaligned-vector-move assembler flag? I don't have a recent enough binutils for it, and llama.cpp's been working so far...

---

👤 **whoreson** commented the **2024-08-31** at **19:46:57**:<br>

Alas, no... Same crash with latest mingw's gcc 14.1 and binutils 2.42.

---

👤 **ikawrakow** commented the **2024-09-01** at **09:34:15**:<br>

If you tried 3 different compiler versions and the crash persists, then it is more likely that it is a bug in the code that somehow only shows up on Windows (any Windows or just Windows 7?).

I see [here](https://github.com/google/sanitizers/wiki/AddressSanitizerWindowsPort) that one can use the address sanitizer with `clang` for Windows.  If you can get it going that way, this might help find the problem.

---

👤 **whoreson** commented the **2024-09-01** at **19:57:45**:<br>

Okay "good news", I've compiled it with the same TDM-GCC on a Windows 11 box (with -mno-avx512f, because it's a much newer CPU), and it crashes there too.

It works when compiled with the default AVX512 setting.

---

👤 **ikawrakow** commented the **2024-09-02** at **08:54:50**:<br>

Do you find it important to disable AVX512?

---

👤 **whoreson** commented the **2024-09-02** at **16:31:29**:<br>

Well since the Windows 7 PC in question is only AVX2, I kinda absolutely have to, in order to maintain the comparison...

So it'd seem to me that there's some AVX2 bug going on on all Windows OSes? I'll check if I can do some address sanitizing checks, but sounds extremely painful.

---

👤 **whoreson** commented the **2024-09-02** at **16:38:57**:<br>

I can set up an rdesktop access if that's at all helpful.

---

👤 **ikawrakow** commented the **2024-09-02** at **17:31:21**:<br>

`-march=native` does not work? This enables the features your CPU supports. If you are setting this manually, you need `FMA` and `F16C` in addition to `AVX2`

---

👤 **whoreson** commented the **2024-09-03** at **18:21:16**:<br>

Err, I think you misunderstood. I'm using the default flags as usual. In order to test the AVX2 code on the PC which has Windows 11 (to check if it's a 7 vs 11 issue), I had to disable AVX512 on that box - naturally.

---

👤 **whoreson** commented the **2024-09-14** at **17:00:21**:<br>

> I can set up an rdesktop access if that's at all helpful.

Sooo... no?

---

👤 **ikawrakow** commented the **2024-09-15** at **06:25:32**:<br>

We can try, but I'm not very hopeful as I haven't touched a Windows computer for 10+ years. What is the Linux rdesktop client one uses these days? I'm on Ubuntu 22.04.

---

👤 **whoreson** commented the **2024-09-15** at **08:41:29**:<br>

Well, it's called just that, "rdesktop". It works fine. I'll set it up then. Err, can github do private messages? If not, I have Telegram.

---

👤 **ikawrakow** commented the **2024-09-15** at **10:01:30**:<br>

As far as I can tell the private message feature has been removed from Githib. I don't have Telegram. I made my email address public. If you fetch the latest main branch the last commit will have my email.

---

👤 **whoreson** commented the **2024-09-15** at **11:45:28**:<br>

Cool, just sent you an e-mail (from s*.t*@gmail).

---

👤 **ikawrakow** commented the **2024-09-19** at **08:49:48**:<br>

So, I used the provided `rdesktop` access to try to debug - without success. Supporting exotic systems (and yes, a Windows 7 box in the year 2024 is an exotic system on my book) is not one of the goals here - you are much better served with the mainline `llama.cpp` project.