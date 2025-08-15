### 🐛 [#59](https://github.com/ikawrakow/ik_llama.cpp/issues/59) - Bug: GGML Compilation Error: undefined references to `iqk_mul_mat'

| **Author** | `ndavidson19` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-09-18 |
| **Updated** | 2024-09-26 |

---

#### Description

### What happened?

When running `make llama-server` or `make llama-bench` I observe the following error:

```
/usr/bin/ld: ggml/src/ggml.o: in function `ggml_compute_forward_flash_attn_ext_f16':
ggml.c:(.text+0xbdde): undefined reference to `iqk_flash_attn_noalibi'
/usr/bin/ld: ggml/src/ggml.o: in function `ggml_compute_forward_mul_mat':
ggml.c:(.text+0x13aac): undefined reference to `iqk_mul_mat'
/usr/bin/ld: ggml.c:(.text+0x14ae6): undefined reference to `iqk_mul_mat'
/usr/bin/ld: ggml.c:(.text+0x15109): undefined reference to `iqk_mul_mat'
/usr/bin/ld: ggml/src/ggml.o: in function `ggml_compute_forward_mul_mat_id':
ggml.c:(.text+0x15c49): undefined reference to `iqk_mul_mat_moe'
/usr/bin/ld: ggml/src/ggml-quants.o: in function `ggml_vec_dot_q4_0_q8_0':
ggml-quants.c:(.text+0x24a06): undefined reference to `iqk_mul_mat'
/usr/bin/ld: ggml/src/ggml-quants.o: in function `ggml_vec_dot_q4_1_q8_1':
ggml-quants.c:(.text+0x24b86): undefined reference to `iqk_mul_mat'
/usr/bin/ld: ggml/src/ggml-quants.o: in function `ggml_vec_dot_q5_0_q8_0':
ggml-quants.c:(.text+0x24d16): undefined reference to `iqk_mul_mat'
/usr/bin/ld: ggml/src/ggml-quants.o: in function `ggml_vec_dot_q5_1_q8_1':
ggml-quants.c:(.text+0x24ee6): undefined reference to `iqk_mul_mat'
/usr/bin/ld: ggml/src/ggml-quants.o: in function `ggml_vec_dot_q8_0_q8_0':
ggml-quants.c:(.text+0x250d6): undefined reference to `iqk_mul_mat'
/usr/bin/ld: ggml/src/ggml-quants.o:ggml-quants.c:(.text+0x28c26): more undefined references to `iqk_mul_mat' follow
collect2: error: ld returned 1 exit status
make: *** [Makefile:1458: llama-server] Error 1
```

## System Specs
```
Architecture:            x86_64
  CPU op-mode(s):        32-bit, 64-bit
  Address sizes:         40 bits physical, 48 bits virtual
  Byte Order:            Little Endian
CPU(s):                  24
  On-line CPU(s) list:   0-23
Vendor ID:               GenuineIntel
  Model name:            Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz
    CPU family:          6
    Model:               85
    Thread(s) per core:  1
    Core(s) per socket:  1
    Socket(s):           24
    Stepping:            4
    BogoMIPS:            3990.62
    Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon nopl xtopology tsc_reliable nonstop_tsc cpuid tsc_known_freq pni pclmulqdq ssse3 fma cx16 p
                         cid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm 3dnowprefetch pti ssbd ibrs ibpb stibp fsgsbase smep arat md_clear flush_l1d arch_capabilities
```

---

What is interesting however is how this is failing only on this server and not for my other server with the following CPU in which I get >50% improvements in prompt processing and token generation.

Side Note: Thank you for all the great work with the llama.cpp project and the open-source community!

```
Architecture:             x86_64
  CPU op-mode(s):         32-bit, 64-bit
  Address sizes:          46 bits physical, 57 bits virtual
  Byte Order:             Little Endian
CPU(s):                   160
  On-line CPU(s) list:    0-159
Vendor ID:                GenuineIntel
  Model name:             Intel(R) Xeon(R) Platinum 8380 CPU @ 2.30GHz
    CPU family:           6
    Model:                106
    Thread(s) per core:   2
    Core(s) per socket:   40
    Socket(s):            2
    Stepping:             6
    CPU max MHz:          3400.0000
    CPU min MHz:          800.0000
    BogoMIPS:             4600.00
    Flags:                fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni 
                          pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb cat_l3 invpcid_single intel_ppin ssbd
                           mba ibrs ibpb stibp ibrs_enhanced tpr_shadow vnmi flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb intel_pt avx512cd sha_ni avx512bw avx512vl
                           xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local split_lock_detect wbnoinvd dtherm ida arat pln pts hwp hwp_act_window hwp_epp hwp_pkg_req avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni a
                          vx512_bitalg tme avx512_vpopcntdq la57 rdpid fsrm md_clear pconfig flush_l1d arch_capabilities
```


### Name and Version

./llama-server --version
version: 3432 (12bbdb8c)
built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu (Working)

The other server will not build

### What operating system are you seeing the problem on?

Linux

### Relevant log output

_No response_

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2024-09-19** at **06:55:56**:<br>

I use `cmake`, so the Makefile is less solid than it shou be. Have you tried `make clean && make -j`? I'm away for a few days and will look at the problem when I come back.

---

👤 **ndavidson19** commented the **2024-09-19** at **16:39:19**:<br>

Same error happens with those commands. No rush will try to build via `cmake` on this particular server.

---

👤 **ikawrakow** commented the **2024-09-21** at **16:05:29**:<br>

So, I don't really see what could be wrong with the `Makefile`. The `Makefile`, inherited from `llama.cpp`, is of course useless as it does not reflect the actual build artifact dependencies. E.g., here is what we have as a build rule for `ggml.o`, which is the core of the whole system
```
ggml/src/ggml.o: \
    ggml/src/ggml.c \
    ggml/include/ggml.h
    $(CC)  $(CFLAGS)   -c $< -o $@
```
In reality `ggml.o` depends on several other files as one gets via
```
gcc -Iggml/include -Iggml/src -MM ggml/src/ggml.c
ggml.o: ggml/src/ggml.c ggml/src/ggml-impl.h ggml/include/ggml.h \
  ggml/src/ggml-quants.h ggml/src/ggml-common.h ggml/src/ggml-aarch64.h
```
But `make clean && make` does produce the correct build, both in mainline `llama.cpp` and in this repository, so the failure you get on this one server is a bit mysterious.

Can you post the full output of the `make` command?
Thanks!

---

👤 **ikawrakow** commented the **2024-09-26** at **16:20:39**:<br>

I'm not getting a response, and without the full output of the `make` command it is not possible to see what might be going wrong, so closing.