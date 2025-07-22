### 🐛 [#388](https://github.com/ikawrakow/ik_llama.cpp/issues/388) - Bug: Clash with mainline llama.cpp .so files

| **Author** | `Manamama` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-06 |
| **Updated** | 2025-05-25 |

---

#### Description

### What happened?

Segmentation fault as the files clash. 


This is needed: `export LD_LIBRARY_PATH=$(pwd)/src/:$(pwd)/ggml/src/:$LD_LIBRARY_PATH` 
See also https://github.com/microsoft/BitNet/issues/206#issuecomment-2855580152 

Why? 

As: 

```
~/Downloads/ik_llama.cpp$ echo $LD_LIBRARY_PATH
/usr/local/lib:/usr/lib/llvm-14/lib/:/usr/lib/sudo
```

so:

```
~/Downloads/ik_llama.cpp$ ldd bin/llama-cli 
	linux-vdso.so.1 (0x00007ffc1731c000)
	libllama.so => /usr/local/lib/libllama.so (0x00007fe866e51000)
	libggml.so => /usr/local/lib/libggml.so (0x00007fe866e44000)
	libstdc++.so.6 => /lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007fe866a00000)
	libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007fe866d36000)
	libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007fe866d12000)
	libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007fe866600000)
	libggml-base.so => /usr/local/lib/libggml-base.so (0x00007fe86692c000)
	/lib64/ld-linux-x86-64.so.2 (0x00007fe867093000)
	libggml-cpu.so => /usr/local/lib/libggml-cpu.so (0x00007fe866870000)
	libggml-rpc.so => /usr/local/lib/libggml-rpc.so (0x00007fe866cfc000)
	libgomp.so.1 => /lib/x86_64-linux-gnu/libgomp.so.1 (0x00007fe866ca8000)

```
which segfaults: 

```
~/Downloads/ik_llama.cpp$  bin/llama-cli 
Log start
main: build = 3668 (6c23618c)
main: built with Ubuntu clang version 14.0.0-1ubuntu1.1 for x86_64-pc-linux-gnu
main: seed  = 1746557487
Segmentation fault

```

After `export LD_LIBRARY_PATH=$(pwd)/src/:$(pwd)/ggml/src/:$LD_LIBRARY_PATH` :
```
~/Downloads/ik_llama.cpp$ ldd bin/llama-cli 
	linux-vdso.so.1 (0x00007ffca9b93000)
	libllama.so => .../Downloads/ik_llama.cpp/src/libllama.so (0x00007f5afeaae000)
	libggml.so => .../Downloads/ik_llama.cpp/ggml/src/libggml.so (0x00007f5afdc00000)
	libstdc++.so.6 => /lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007f5afd800000)
	libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007f5afdb19000)
	libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007f5afea61000)
	libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f5afd400000)
	/lib64/ld-linux-x86-64.so.2 (0x00007f5afed2d000)
	libomp.so.5 => /usr/lib/llvm-14/lib/libomp.so.5 (0x00007f5afd6e0000)



```

and starts to work: 

```
~/Downloads/ik_llama.cpp$  bin/llama-cli 
Log start
main: build = 3668 (6c23618c)
main: built with Ubuntu clang version 14.0.0-1ubuntu1.1 for x86_64-pc-linux-gnu
main: seed  = 1746557907

``` 

Rpath or like is needed. 

### Name and Version

main: build = 3668 (6c23618c)
main: built with Ubuntu clang version 14.0.0-1ubuntu1.1 for x86_64-pc-linux-gnu


### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
Most Linuxex, I presume.
```

---

#### 💬 Conversation

👤 **Manamama** commented the **2025-05-06** at **19:03:45**:<br>

Update, still seg fault: 

```
bin/llama-cli  -m /mnt/HP_P7_Data/Temp/GPT4All_DBs/Bitnet_MS/ggml-model-i2_s.gguf
Log start
main: build = 3668 (6c23618c)
main: built with Ubuntu clang version 14.0.0-1ubuntu1.1 for x86_64-pc-linux-gnu
main: seed  = 1746558071
llama_model_loader: loaded meta data with 24 key-value pairs and 333 tensors from /mnt/HP_P7_Data/Temp/GPT4All_DBs/Bitnet_MS/ggml-model-i2_s.gguf (version GGUF V3 (latest))
llama_model_loader: unknown type i2_s
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = bitnet-25
llama_model_loader: - kv   1:                               general.name str              = bitnet2b_2501
llama_model_loader: - kv   2:                       bitnet-25.vocab_size u32              = 128256
llama_model_loader: - kv   3:                   bitnet-25.context_length u32              = 4096
llama_model_loader: - kv   4:                 bitnet-25.embedding_length u32              = 2560
llama_model_loader: - kv   5:                      bitnet-25.block_count u32              = 30
llama_model_loader: - kv   6:              bitnet-25.feed_forward_length u32              = 6912
llama_model_loader: - kv   7:             bitnet-25.rope.dimension_count u32              = 128
llama_model_loader: - kv   8:             bitnet-25.attention.head_count u32              = 20
llama_model_loader: - kv   9:          bitnet-25.attention.head_count_kv u32              = 5
llama_model_loader: - kv  10:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  11: bitnet-25.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  12:                   bitnet-25.rope.freq_base f32              = 500000.000000
llama_model_loader: - kv  13:                          general.file_type u32              = 40
llama_model_loader: - kv  14:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  15:                      tokenizer.ggml.tokens arr[str,128256]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  16:                      tokenizer.ggml.scores arr[f32,128256]  = [0.000000, 0.000000, 0.000000, 0.0000...
llama_model_loader: - kv  17:                  tokenizer.ggml.token_type arr[i32,128256]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  18:                      tokenizer.ggml.merges arr[str,280147]  = ["Ġ Ġ", "Ġ ĠĠĠ", "ĠĠ ĠĠ", "...
llama_model_loader: - kv  19:                tokenizer.ggml.bos_token_id u32              = 128000
llama_model_loader: - kv  20:                tokenizer.ggml.eos_token_id u32              = 128001
llama_model_loader: - kv  21:            tokenizer.ggml.padding_token_id u32              = 128001
llama_model_loader: - kv  22:                    tokenizer.chat_template str              = {% for message in messages %}{% if lo...
llama_model_loader: - kv  23:               general.quantization_version u32              = 2
llama_model_loader: - type  f32:  121 tensors
llama_model_loader: - type  f16:    2 tensors
llama_model_loader: - type i2_s:  210 tensors
llm_load_vocab: missing pre-tokenizer type, using: 'llama3'
llm_load_vocab:                                             
llm_load_vocab: ************************************        
llm_load_vocab: GENERATION QUALITY MAY BE DEGRADED!         
llm_load_vocab: CONSIDER REGENERATING THE MODEL             
llm_load_vocab: ************************************        
llm_load_vocab:                                             
llm_load_vocab: special tokens cache size = 256
llm_load_vocab: token to piece cache size = 0.8000 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = bitnet-25
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 128256
llm_load_print_meta: n_merges         = 280147
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 4096
llm_load_print_meta: n_embd           = 2560
llm_load_print_meta: n_layer          = 30
llm_load_print_meta: n_head           = 20
llm_load_print_meta: n_head_kv        = 5
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 4
llm_load_print_meta: n_embd_k_gqa     = 640
llm_load_print_meta: n_embd_v_gqa     = 640
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 6912
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 2
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 500000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 4096
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = 2B
llm_load_print_meta: model ftype      = unknown, may not work
llm_load_print_meta: model params     = 2.741 B
llm_load_print_meta: model size       = 1.710 GiB (5.359 BPW) 
llm_load_print_meta: repeating layers = 498.561 MiB (2.006 BPW, 2.084 B parameters)
llm_load_print_meta: general.name     = bitnet2b_2501
llm_load_print_meta: BOS token        = 128000 '<|begin_of_text|>'
llm_load_print_meta: EOS token        = 128001 '<|end_of_text|>'
llm_load_print_meta: PAD token        = 128001 '<|end_of_text|>'
llm_load_print_meta: LF token         = 128 'Ä'
llm_load_print_meta: EOT token        = 128009 '<|eot_id|>'
llm_load_print_meta: max token length = 256
llm_load_tensors: ggml ctx size =    0.15 MiB
llm_load_tensors:        CPU buffer size =  1751.06 MiB
...............................
llama_new_context_with_model: n_ctx      = 4096
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 0
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 500000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:        CPU KV buffer size =   300.00 MiB
llama_new_context_with_model: KV self size  =  300.00 MiB, K (f16):  150.00 MiB, V (f16):  150.00 MiB
llama_new_context_with_model:        CPU  output buffer size =     0.49 MiB
llama_new_context_with_model:        CPU compute buffer size =   255.50 MiB
llama_new_context_with_model: graph nodes  = 995
llama_new_context_with_model: graph splits = 1
Segmentation fault
```

Not sure where: 
```
openat(AT_FDCWD, "/usr/lib/x86_64/libmemkind.so", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
newfstatat(AT_FDCWD, "/usr/lib/x86_64", 0x7ffdbcc3af40, 0) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/usr/lib/libmemkind.so", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
newfstatat(AT_FDCWD, "/usr/lib", {st_mode=S_IFDIR|0755, st_size=20480, ...}, 0) = 0
munmap(0x7f5e9801a000, 158559)          = 0
getpid()                                = 26168
getuid()                                = 1000
openat(AT_FDCWD, "/dev/shm/__KMP_REGISTERED_LIB_26168_1000", O_RDWR|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC, 0666) = 5
ftruncate(5, 1024)                      = 0
mmap(NULL, 1024, PROT_READ|PROT_WRITE, MAP_SHARED, 5, 0) = 0x7f5e9836e000
munmap(0x7f5e9836e000, 1024)            = 0
close(5)                                = 0
openat(AT_FDCWD, "/sys/devices/system/cpu", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 5
newfstatat(5, "", {st_mode=S_IFDIR|0755, st_size=0, ...}, AT_EMPTY_PATH) = 0
getdents64(5, 0x560a41a2a3c0 /* 26 entries */, 32768) = 752
getdents64(5, 0x560a41a2a3c0 /* 0 entries */, 32768) = 0
close(5)                                = 0
prlimit64(0, RLIMIT_STACK, NULL, {rlim_cur=8192*1024, rlim_max=RLIM64_INFINITY}) = 0
sched_getaffinity(0, 64, [0, 1, 2, 3, 4, 5, 6, 7]) = 8
rt_sigaction(SIGHUP, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGINT, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGQUIT, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGILL, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGABRT, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGFPE, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGBUS, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGSEGV, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGSYS, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGTERM, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
rt_sigaction(SIGPIPE, NULL, {sa_handler=SIG_DFL, sa_mask=[], sa_flags=0}, 8) = 0
sched_getaffinity(0, 8, [0, 1, 2, 3, 4, 5, 6, 7]) = 8
sched_getaffinity(0, 8, [0, 1, 2, 3, 4, 5, 6, 7]) = 8
sched_setaffinity(0, 8, [0])            = 0
sched_setaffinity(0, 8, [1])            = 0
sched_setaffinity(0, 8, [2])            = 0
sched_setaffinity(0, 8, [3])            = 0
sched_setaffinity(0, 8, [4])            = 0
sched_setaffinity(0, 8, [5])            = 0
sched_setaffinity(0, 8, [6])            = 0
sched_setaffinity(0, 8, [7])            = 0
sched_setaffinity(0, 8, [0, 1, 2, 3, 4, 5, 6, 7]) = 0
sched_setaffinity(0, 8, [0, 1, 2, 3, 4, 5, 6, 7]) = 0
sched_getaffinity(0, 8, [0, 1, 2, 3, 4, 5, 6, 7]) = 8
sched_setaffinity(0, 8, [0, 1, 2, 3, 4, 5, 6, 7]) = 0
rt_sigaction(SIGRT_1, {sa_handler=0x7f5e96a91870, sa_mask=[], sa_flags=SA_RESTORER|SA_ONSTACK|SA_RESTART|SA_SIGINFO, sa_restorer=0x7f5e96a42520}, NULL, 8) = 0
rt_sigprocmask(SIG_UNBLOCK, [RTMIN RT_1], NULL, 8) = 0
mmap(NULL, 8393856, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_STACK, -1, 0) = 0x7f5df5048000
mprotect(0x7f5df5049000, 8389760, PROT_READ|PROT_WRITE) = 0
rt_sigprocmask(SIG_BLOCK, ~[], [], 8)   = 0
clone3({flags=CLONE_VM|CLONE_FS|CLONE_FILES|CLONE_SIGHAND|CLONE_THREAD|CLONE_SYSVSEM|CLONE_SETTLS|CLONE_PARENT_SETTID|CLONE_CHILD_CLEARTID, child_tid=0x7f5df5848d90, parent_tid=0x7f5df5848d90, exit_signal=0, stack=0x7f5df5048000, stack_size=0x800340, tls=0x7f5df5848ac0} => {parent_tid=[26169]}, 88) = 26169
rt_sigprocmask(SIG_SETMASK, [], NULL, 8) = 0
mmap(NULL, 8393984, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_STACK, -1, 0) = 0x7f5df4846000
mprotect(0x7f5df4847000, 8389888, PROT_READ|PROT_WRITE) = 0
rt_sigprocmask(SIG_BLOCK, ~[], [], 8)   = 0
clone3({flags=CLONE_VM|CLONE_FS|CLONE_FILES|CLONE_SIGHAND|CLONE_THREAD|CLONE_SYSVSEM|CLONE_SETTLS|CLONE_PARENT_SETTID|CLONE_CHILD_CLEARTID, child_tid=0x7f5df5046e10, parent_tid=0x7f5df5046e10, exit_signal=0, stack=0x7f5df4846000, stack_size=0x8003c0, tls=0x7f5df5046b40} => {parent_tid=[26170]}, 88) = 26170
rt_sigprocmask(SIG_SETMASK, [], NULL, 8) = 0
mmap(NULL, 8394112, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_STACK, -1, 0) = 0x7f5df4044000
mprotect(0x7f5df4045000, 8390016, PROT_READ|PROT_WRITE) = 0
rt_sigprocmask(SIG_BLOCK, ~[], [], 8)   = 0
clone3({flags=CLONE_VM|CLONE_FS|CLONE_FILES|CLONE_SIGHAND|CLONE_THREAD|CLONE_SYSVSEM|CLONE_SETTLS|CLONE_PARENT_SETTID|CLONE_CHILD_CLEARTID, child_tid=0x7f5df4844e90, parent_tid=0x7f5df4844e90, exit_signal=0, stack=0x7f5df4044000, stack_size=0x800440, tls=0x7f5df4844bc0} => {parent_tid=[26171]}, 88) = 26171
rt_sigprocmask(SIG_SETMASK, [], NULL, 8) = 0
sched_setaffinity(0, 8, [0, 1, 2, 3, 4, 5, 6, 7]) = 0
futex(0x560a419959a8, FUTEX_WAKE_PRIVATE, 1) = 1
futex(0x560a419959c0, FUTEX_WAKE_PRIVATE, 1) = 1
futex(0x560a419203e8, FUTEX_WAKE_PRIVATE, 1) = 1
futex(0x560a41920400, FUTEX_WAKE_PRIVATE, 1) = 1
futex(0x560a41978f68, FUTEX_WAKE_PRIVATE, 1) = 1
futex(0x560a41978f80, FUTEX_WAKE_PRIVATE, 1) = 1
+++ killed by SIGSEGV +++
Segmentation fault
```

(BitNet llama-cli works)

---

👤 **Manamama** commented the **2025-05-06** at **19:16:14**:<br>

Oh, identical in Termux. Grok AI wrote the below, sorry for the dump paste:  

> 
> Title: Library Clash with libllama.so on Termux (Android, aarch64) Causes Missing Symbol Errors
> Description:
> When running llama-cli from ik_llama.cpp on Termux (Android, aarch64), the executable fails with a dynamic linking error due to a missing symbol (llama_print_timings or llama_model_get_vocab) when using the system library path (/data/data/com.termux/files/usr/lib/libllama.so). This appears to be caused by a clash with an incompatible libllama.so, likely from a standard llama.cpp installation or a previous ik_llama.cpp build.
> Environment:
> OS: Android (Termux)
> 
> Kernel: Linux localhost 4.14.186+ #1 SMP PREEMPT Thu Mar 17 16:28:22 CST 2022 aarch64 Android
> 
> Architecture: aarch64
> 
> Compiler: Clang 20.1.3
> 
> Project: ik_llama.cpp (commit unknown, cloned from https://github.com/ikawrakow/ik_llama.cpp)
> 
> Termux Packages: git, cmake, make, clang
> 
> Library Path: /data/data/com.termux/files/usr/lib (contains libllama.so, libggml.so, etc.)
> 
> System llama-cli: /data/data/com.termux/files/usr/bin/llama-cli (likely from standard llama.cpp)
> 
> Steps to Reproduce:
> Install Termux on an Android device and set up dependencies:
> bash
> 
> pkg update && pkg upgrade
> pkg install git cmake make clang
> 
> Clone and build ik_llama.cpp:
> bash
> 
> cd ~/downloads
> git clone https://github.com/ikawrakow/ik_llama.cpp
> cd ik_llama.cpp
> cmake .
> make
> 
> Run the built llama-cli:
> bash
> 
> bin/llama-cli --help
> 
> Observe the error:
> 
> CANNOT LINK EXECUTABLE "bin/llama-cli": cannot locate symbol "llama_print_timings" referenced by "/data/data/com.termux/files/home/downloads/ik_llama.cpp/bin/llama-cli"...
> 
> Check the system llama-cli:
> bash
> 
> /data/data/com.termux/files/usr/bin/llama-cli --version
> 
> Output:
> 
> CANNOT LINK EXECUTABLE "/data/data/com.termux/files/usr/bin/llama-cli": cannot locate symbol "llama_model_get_vocab" referenced by "/data/data/com.termux/files/usr/bin/llama-cli"...
> 
> Expected Behavior:
> bin/llama-cli should run without dynamic linking errors, using the libllama.so built in ~/downloads/ik_llama.cpp/src.
> 
> The system llama-cli (/data/data/com.termux/files/usr/bin/llama-cli) should either work or not interfere with the local build.
> 
> Actual Behavior:
> bin/llama-cli fails due to linking against an incompatible libllama.so in /data/data/com.termux/files/usr/lib, which lacks the llama_print_timings symbol.
> 
> The system llama-cli fails with a different missing symbol (llama_model_get_vocab), indicating a potential mismatch or incomplete installation.
> 
> Workaround:
> Setting LD_LIBRARY_PATH to prioritize the local libllama.so resolves the issue for the local build:
> bash
> 
> export LD_LIBRARY_PATH=/data/data/com.termux/files/home/downloads/ik_llama.cpp/src:$LD_LIBRARY_PATH
> bin/llama-cli --help
> 
> This displays the help menu correctly. However, the system llama-cli remains broken unless conflicting libraries are removed or LD_LIBRARY_PATH is cleared:
> bash
> 
> export LD_LIBRARY_PATH=
> /data/data/com.termux/files/usr/bin/llama-cli --version
> 
> Output:
> 
> version: 5293 (1e333d5b)
> built with  for 
> 
> Root Cause:
> The system library path (/data/data/com.termux/files/usr/lib) contains an incompatible libllama.so, likely from standard llama.cpp or an older ik_llama.cpp build, missing required symbols.
> 
> The local build’s libllama.so in ~/downloads/ik_llama.cpp/src is correct but not used by default due to an empty LD_LIBRARY_PATH.
> 
> The system llama-cli (/data/data/com.termux/files/usr/bin/llama-cli) is likely from a standard llama.cpp installation, causing conflicts.
> 
> Suggested Fix:
> Improve Build Isolation:
> Update CMakeLists.txt to install libraries to a local directory (e.g., ~/downloads/ik_llama.cpp/install) by default, avoiding /data/data/com.termux/files/usr/lib.
> 
> Example:
> bash
> 
> cmake -DCMAKE_INSTALL_PREFIX=/data/data/com.termux/files/home/downloads/ik_llama.cpp/install .
> make install
> 
> Document Termux-Specific Instructions:
> Add a section to docs/android.md or README.md for Termux builds, warning about library clashes and recommending:
> Setting LD_LIBRARY_PATH for local testing.
> 
> Using a custom install prefix to avoid system library conflicts.
> 
> Checking for and removing conflicting libllama.so or libggml.so in /data/data/com.termux/files/usr/lib.
> 
> Check System llama-cli Compatibility:
> Investigate why /data/data/com.termux/files/usr/bin/llama-cli fails with llama_model_get_vocab missing, even with LD_LIBRARY_PATH cleared.
> 
> Ensure ik_llama.cpp binaries are compatible with standard llama.cpp libraries or clearly document incompatibilities.
> 
> Symbol Verification:
> Add a build-time check to verify that libllama.so contains expected symbols (e.g., llama_print_timings, llama_model_get_vocab).
> 
> Example: Use nm -D in a CMake script to validate the library.
> 
> Additional Notes:
> The issue is specific to Termux on Android (aarch64) due to the shared library path and potential for multiple llama.cpp-based installations.
> 
> The workaround (LD_LIBRARY_PATH) is effective but not ideal for users unfamiliar with dynamic linking.
> 
> The system llama-cli issue suggests a broader problem with Termux package management or incomplete installations, which may require coordination with Termux maintainers.
> 
> References:
> uname -a: Linux localhost 4.14.186+ #1 SMP PREEMPT Thu Mar 17 16:28:22 CST 2022 aarch64 Android

---

👤 **Manamama** commented the **2025-05-06** at **19:57:18**:<br>

Update: this avoids seg faults in Ubuntu: https://github.com/ikawrakow/ik_llama.cpp/issues/387#issuecomment-2855735935

```
./bin/llama-cli -m /mnt/HP_P7_Data/Temp/GPT4All_DBs/Bitnet_MS/ggml-model-i2_s_requantized.gguf -p "Introduce yourself"
Log start
main: build = 3668 (6c23618c)
main: built with Ubuntu clang version 14.0.0-1ubuntu1.1 for x86_64-pc-linux-gnu
main: seed  = 1746561197
...
system_info: n_threads = 4 / 8 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | 
sampling: 
	repeat_last_n = 64, repeat_penalty = 1.000, frequency_penalty = 0.000, presence_penalty = 0.000
	top_k = 40, tfs_z = 1.000, top_p = 0.950, min_p = 0.050, typical_p = 1.000, temp = 0.800
	mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000
sampling order: 
CFG -> Penalties -> top_k -> tfs_z -> typical_p -> top_p -> min_p -> temperature 
generate: n_ctx = 4096, n_batch = 2048, n_predict = -1, n_keep = 1


Introduce yourself and describe your role in the company. Make sure to mention that you are available for any questions.
I am [name], a [Job Title] at [Company Name]. I am responsible for [brief description of your role]. I am available for any questions or concerns you may have.
Example response: "Hello, my name is John Smith and I am a Marketing Manager at [Company Name]. I am responsible for overseeing our social media campaigns and content marketing efforts. I am available for any questions or concerns you may have." 

Answering these questions in a way that is concise and professional can help to establish a positive and effective communication channel for your team.  So, when you are asked about your availability for any questions or concerns, you can respond with this answer in a professional and friendly manner.  For instance, you can say "I am available for any questions or concerns you may have. Please feel free to reach out to me."  This will show that you are approachable and open to communication.  It's important to remember that being available for questions and concerns is a key aspect of being a good leader and team member.  By being responsive and accessible, you can build trust and create a positive working relationship with your team.  So,

llama_print_timings:        load time =    1530.60 ms
llama_print_timings:      sample time =      45.34 ms /   255 runs   (    0.18 ms per token,  5623.80 tokens per second)
llama_print_timings: prompt eval time =      96.29 ms /     4 tokens (   24.07 ms per token,    41.54 tokens per second)
llama_print_timings:        eval time =   19730.00 ms /   254 runs   (   77.68 ms per token,    12.87 tokens per second)
llama_print_timings:       total time =   19962.37 ms /   258 tokens
```


But I am not sure why the size got from tiny to minuscule: 

```
~/Downloads/ik_llama.cpp$ ll /mnt/HP_P7_Data/Temp/GPT4All_DBs/Bitnet_MS/
total 5241280
drwxrwxrwx 1 root root        488 May  6 21:51 ./
drwxrwxrwx 1 root root       8192 Apr 22 16:18 ../
-rwxrwxrwx 1 root root 1844472032 Apr 22 18:04 ggml-model-i2_s.gguf*
-rwxrwxrwx 1 root root  987884192 May  6 21:52 ggml-model-i2_s_requantized.gguf*
```

---

👤 **saood06** commented the **2025-05-06** at **19:58:26**:<br>

How did you build this on Ubuntu and Android? Do you mind sharing the logs from both builds? 

Also on termux you may want to try adding "-DGGML_ARCH_FLAGS="-march=armv8.2-a+dotprod+fp16" to your build.

---

👤 **saood06** commented the **2025-05-06** at **20:01:17**:<br>

>But I am not sure why the size got from tiny to minuscule:

That is because this happens on reconvert:

```
[   1/ 333]                        output.weight - [ 2560, 128256,     1,     1], type =    f16, converting to q6_K .. size =   626.25 MiB ->   256.86 MiB
[   2/ 333]                    token_embd.weight - [ 2560, 128256,     1,     1], type =    f16, converting to iq4_nl .. size =   626.25 MiB ->   176.13 MiB
```

which is expected.

---

👤 **Manamama** commented the **2025-05-06** at **20:10:16**:<br>

Re Droid only. 

New Termux session, so LD_LIBRARY_PATH is standard: 
```
~/downloads/ik_llama.cpp $ echo $LD_LIBRARY_PATH

~/downloads/ik_llama.cpp $ 
```
so Termux pix up the default libraries (from previous llama.cpp builds) then, I presume. 


We move the old working /bin files and recompile and test: 

```
~/downloads/ik_llama.cpp $ ls bin/
 llama-baby-llama                llama-cvector-generator   llama-gguf-split   llama-lookup-create   llama-q8dot             llama-speculative   test-chat-template            test-quantize-fns
 llama-batched                   llama-embedding           llama-gritlm       llama-lookup-merge    llama-quantize          llama-sweep-bench   test-grad0                    test-quantize-perf
 llama-batched-bench             llama-eval-callback       llama-imatrix      llama-lookup-stats    llama-quantize-stats    llama-tokenize      test-grammar-integration      test-rope
 llama-bench                     llama-export-lora         llama-infill       llama-minicpmv-cli    llama-retrieval         llama-vdot          test-grammar-parser           test-sampling
 llama-bench-matmult             llama-gbnf-validator      llama-llava-cli    llama-parallel        llama-save-load-state   test-autorelease    test-json-schema-to-grammar   test-tokenizer-0
 llama-cli                       llama-gguf                llama-lookahead    llama-passkey         llama-server            test-backend-ops    test-llama-grammar            test-tokenizer-1-bpe
 llama-convert-llama2c-to-ggml   llama-gguf-hash           llama-lookup       llama-perplexity      llama-simple            test-c              test-model-load-cancel        test-tokenizer-1-spm
~/downloads/ik_llama.cpp $ mv bin/ bin.1
~/downloads/ik_llama.cpp $ rm CMakeCache.txt 
~/downloads/ik_llama.cpp $ cmake .
-- The C compiler identification is Clang 20.1.3
-- The CXX compiler identification is Clang 20.1.3
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /data/data/com.termux/files/usr/bin/clang - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /data/data/com.termux/files/usr/bin/clang++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Found Git: /data/data/com.termux/files/usr/bin/git (found version "2.49.0")
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Failed
-- Check if compiler accepts -pthread
-- Check if compiler accepts -pthread - yes
-- Found Threads: TRUE
-- Found OpenMP_C: -fopenmp=libomp (found version "5.1")
-- Found OpenMP_CXX: -fopenmp=libomp (found version "5.1")
-- Found OpenMP: TRUE (found version "5.1")
-- OpenMP found
-- Using optimized iqk matrix multiplications
-- Using llamafile
-- ccache found, compilation results will be cached. Disable with GGML_CCACHE=OFF.
-- CMAKE_SYSTEM_PROCESSOR: aarch64
-- ARM detected
-- Performing Test COMPILER_SUPPORTS_FP16_FORMAT_I3E
-- Performing Test COMPILER_SUPPORTS_FP16_FORMAT_I3E - Failed
-- Looking for pthread_create in pthreads
-- Looking for pthread_create in pthreads - not found
-- Looking for pthread_create in pthread
-- Looking for pthread_create in pthread - found
-- Configuring done (12.1s)
-- Generating done (0.4s)
-- Build files have been written to: /data/data/com.termux/files/home/downloads/ik_llama.cpp
~/downloads/ik_llama.cpp $ make 
[  6%] Built target ggml
[ 10%] Built target llama
[ 11%] Built target build_info
[ 15%] Built target common
[ 16%] Linking CXX executable ../bin/test-tokenizer-0
[ 17%] Built target test-tokenizer-0
[ 18%] Linking CXX executable ../bin/test-tokenizer-1-bpe
[ 18%] Built target test-tokenizer-1-bpe
[ 19%] Linking CXX executable ../bin/test-tokenizer-1-spm
[ 19%] Built target test-tokenizer-1-spm
[ 19%] Linking CXX executable ../bin/test-quantize-fns
[ 20%] Built target test-quantize-fns
[ 21%] Linking CXX executable ../bin/test-quantize-perf
[ 22%] Built target test-quantize-perf
[ 22%] Linking CXX executable ../bin/test-sampling
[ 23%] Built target test-sampling
[ 23%] Linking CXX executable ../bin/test-chat-template
[ 24%] Built target test-chat-template
[ 24%] Linking CXX executable ../bin/test-grammar-parser
[ 25%] Built target test-grammar-parser
[ 26%] Linking CXX executable ../bin/test-llama-grammar
[ 27%] Built target test-llama-grammar
[ 28%] Linking CXX executable ../bin/test-grammar-integration
[ 29%] Built target test-grammar-integration
[ 30%] Linking CXX executable ../bin/test-grad0
[ 31%] Built target test-grad0
[ 31%] Linking CXX executable ../bin/test-backend-ops
[ 32%] Built target test-backend-ops
[ 33%] Linking CXX executable ../bin/test-rope
[ 34%] Built target test-rope
[ 35%] Linking CXX executable ../bin/test-model-load-cancel
[ 36%] Built target test-model-load-cancel
[ 37%] Linking CXX executable ../bin/test-autorelease
[ 38%] Built target test-autorelease
[ 38%] Linking CXX executable ../bin/test-json-schema-to-grammar
[ 40%] Built target test-json-schema-to-grammar
[ 41%] Linking C executable ../bin/test-c
[ 42%] Built target test-c
[ 42%] Linking CXX executable ../../bin/llama-cvector-generator
[ 43%] Built target llama-cvector-generator
[ 43%] Linking CXX executable ../../bin/llama-baby-llama
[ 44%] Built target llama-baby-llama
[ 44%] Linking CXX executable ../../bin/llama-batched-bench
[ 45%] Built target llama-batched-bench
[ 45%] Linking CXX executable ../../bin/llama-batched
[ 46%] Built target llama-batched
[ 47%] Linking CXX executable ../../bin/llama-bench-matmult
[ 47%] Built target llama-bench-matmult
[ 48%] Linking CXX executable ../../bin/llama-convert-llama2c-to-ggml
[ 48%] Built target llama-convert-llama2c-to-ggml
[ 48%] Linking CXX executable ../../bin/llama-embedding
[ 49%] Built target llama-embedding
[ 50%] Linking CXX executable ../../bin/llama-eval-callback
[ 51%] Built target llama-eval-callback
[ 52%] Linking CXX executable ../../bin/llama-export-lora
[ 52%] Built target llama-export-lora
[ 53%] Linking CXX executable ../../bin/llama-gbnf-validator
[ 53%] Built target llama-gbnf-validator
[ 54%] Built target sha256
[ 55%] Built target xxhash
[ 55%] Built target sha1
[ 55%] Linking CXX executable ../../bin/llama-gguf-hash
[ 56%] Built target llama-gguf-hash
[ 56%] Linking CXX executable ../../bin/llama-gguf-split
[ 57%] Built target llama-gguf-split
[ 58%] Linking CXX executable ../../bin/llama-gguf
[ 58%] Built target llama-gguf
[ 58%] Linking CXX executable ../../bin/llama-gritlm
[ 59%] Built target llama-gritlm
[ 60%] Linking CXX executable ../../bin/llama-imatrix
[ 61%] Built target llama-imatrix
[ 62%] Linking CXX executable ../../bin/llama-infill
[ 62%] Built target llama-infill
[ 63%] Linking CXX executable ../../bin/llama-bench
[ 64%] Built target llama-bench
[ 66%] Built target llava
[ 67%] Built target llava_static
[ 67%] Built target llava_shared
[ 68%] Linking CXX executable ../../bin/llama-llava-cli
[ 68%] Built target llama-llava-cli
[ 69%] Linking CXX executable ../../bin/llama-minicpmv-cli
[ 69%] Built target llama-minicpmv-cli
[ 70%] Linking CXX executable ../../bin/llama-lookahead
[ 70%] Built target llama-lookahead
[ 70%] Linking CXX executable ../../bin/llama-lookup
[ 71%] Built target llama-lookup
[ 71%] Linking CXX executable ../../bin/llama-lookup-create
[ 72%] Built target llama-lookup-create
[ 72%] Linking CXX executable ../../bin/llama-lookup-merge
[ 73%] Built target llama-lookup-merge
[ 74%] Linking CXX executable ../../bin/llama-lookup-stats
[ 75%] Built target llama-lookup-stats
[ 76%] Linking CXX executable ../../bin/llama-cli
[ 76%] Built target llama-cli
[ 77%] Linking CXX executable ../../bin/llama-parallel
[ 77%] Built target llama-parallel
[ 78%] Linking CXX executable ../../bin/llama-passkey
[ 78%] Built target llama-passkey
[ 78%] Linking CXX executable ../../bin/llama-perplexity
[ 79%] Built target llama-perplexity
[ 80%] Linking CXX executable ../../bin/llama-quantize-stats
[ 80%] Built target llama-quantize-stats
[ 81%] Linking CXX executable ../../bin/llama-quantize
[ 82%] Built target llama-quantize
[ 83%] Linking CXX executable ../../bin/llama-retrieval
[ 83%] Built target llama-retrieval
[ 84%] Linking CXX executable ../../bin/llama-server
[ 93%] Built target llama-server
[ 94%] Linking CXX executable ../../bin/llama-save-load-state
[ 94%] Built target llama-save-load-state
[ 95%] Linking CXX executable ../../bin/llama-simple
[ 95%] Built target llama-simple
[ 96%] Linking CXX executable ../../bin/llama-speculative
[ 96%] Built target llama-speculative
[ 96%] Linking CXX executable ../../bin/llama-sweep-bench
[ 97%] Built target llama-sweep-bench
[ 97%] Linking CXX executable ../../bin/llama-tokenize
[ 98%] Built target llama-tokenize
[ 98%] Linking CXX executable ../../bin/llama-vdot
[ 99%] Built target llama-vdot
[ 99%] Linking CXX executable ../../bin/llama-q8dot
[100%] Built target llama-q8dot
~/downloads/ik_llama.cpp $ ls bin/
 llama-baby-llama                llama-cvector-generator   llama-gguf-split   llama-lookup-create   llama-q8dot             llama-speculative   test-chat-template            test-quantize-fns
 llama-batched                   llama-embedding           llama-gritlm       llama-lookup-merge    llama-quantize          llama-sweep-bench   test-grad0                    test-quantize-perf
 llama-batched-bench             llama-eval-callback       llama-imatrix      llama-lookup-stats    llama-quantize-stats    llama-tokenize      test-grammar-integration      test-rope
 llama-bench                     llama-export-lora         llama-infill       llama-minicpmv-cli    llama-retrieval         llama-vdot          test-grammar-parser           test-sampling
 llama-bench-matmult             llama-gbnf-validator      llama-llava-cli    llama-parallel        llama-save-load-state   test-autorelease    test-json-schema-to-grammar   test-tokenizer-0
 llama-cli                       llama-gguf                llama-lookahead    llama-passkey         llama-server            test-backend-ops    test-llama-grammar            test-tokenizer-1-bpe
 llama-convert-llama2c-to-ggml   llama-gguf-hash           llama-lookup       llama-perplexity      llama-simple            test-c              test-model-load-cancel        test-tokenizer-1-spm
~/downloads/ik_llama.cpp $ ldd bin/llama-cli 
	liblog.so => /system/lib64/liblog.so
	libargp.so => /data/data/com.termux/files/usr/lib/libargp.so
	libllama.so => /data/data/com.termux/files/usr/lib/libllama.so
	libc.so => /system/lib64/libc.so
	libggml.so => /data/data/com.termux/files/usr/lib/libggml.so
	libc++_shared.so => /data/data/com.termux/files/usr/lib/libc++_shared.so
	libdl.so => /system/lib64/libdl.so
	libm.so => /system/lib64/libm.so
	libc++.so => /system/lib64/libc++.so
	ld-android.so => /system/lib64/ld-android.so
	libggml-cpu.so => /data/data/com.termux/files/usr/lib/libggml-cpu.so
	libggml-base.so => /data/data/com.termux/files/usr/lib/libggml-base.so
~/downloads/ik_llama.cpp $  bin/llama-cli 
CANNOT LINK EXECUTABLE "bin/llama-cli": cannot locate symbol "llama_print_timings" referenced by "/data/data/com.termux/files/home/downloads/ik_llama.cpp/bin/llama-cli"...
~/downloads/ik_llama.cpp $ 

```

Only after my trick above it picks up the rigth .so files: 

```
~/downloads/ik_llama.cpp $ cat _path.sh 
export LD_LIBRARY_PATH=$(pwd)/src/:$(pwd)/ggml/src/:$LD_LIBRARY_PATH
~/downloads/ik_llama.cpp $ source  _path.sh 
~/downloads/ik_llama.cpp $ ldd bin/llama-cli 
	liblog.so => /system/lib64/liblog.so
	libargp.so => /data/data/com.termux/files/usr/lib/libargp.so
	libllama.so => /data/data/com.termux/files/home/downloads/ik_llama.cpp/src/libllama.so
	libc.so => /system/lib64/libc.so
	libggml.so => /data/data/com.termux/files/home/downloads/ik_llama.cpp/ggml/src/libggml.so
	libc++_shared.so => /data/data/com.termux/files/usr/lib/libc++_shared.so
	libdl.so => /system/lib64/libdl.so
	libm.so => /system/lib64/libm.so
	libc++.so => /system/lib64/libc++.so
	ld-android.so => /system/lib64/ld-android.so
~/downloads/ik_llama.cpp $ 

``` 
I shall `mv` once again and retry your `"-DGGML_ARCH_FLAGS="-march=armv8.2-a+dotprod+fp16"` ...

---

👤 **Manamama** commented the **2025-05-06** at **20:10:16**:<br>

Re Droid only. 

New Termux session, so LD_LIBRARY_PATH is standard: 
```
~/downloads/ik_llama.cpp $ echo $LD_LIBRARY_PATH

~/downloads/ik_llama.cpp $ 
```
- Termux pix up the defaults then, I presume. 


We move the old working /bin files and recompile and test: 

```
~/downloads/ik_llama.cpp $ ls bin/
 llama-baby-llama                llama-cvector-generator   llama-gguf-split   llama-lookup-create   llama-q8dot             llama-speculative   test-chat-template            test-quantize-fns
 llama-batched                   llama-embedding           llama-gritlm       llama-lookup-merge    llama-quantize          llama-sweep-bench   test-grad0                    test-quantize-perf
 llama-batched-bench             llama-eval-callback       llama-imatrix      llama-lookup-stats    llama-quantize-stats    llama-tokenize      test-grammar-integration      test-rope
 llama-bench                     llama-export-lora         llama-infill       llama-minicpmv-cli    llama-retrieval         llama-vdot          test-grammar-parser           test-sampling
 llama-bench-matmult             llama-gbnf-validator      llama-llava-cli    llama-parallel        llama-save-load-state   test-autorelease    test-json-schema-to-grammar   test-tokenizer-0
 llama-cli                       llama-gguf                llama-lookahead    llama-passkey         llama-server            test-backend-ops    test-llama-grammar            test-tokenizer-1-bpe
 llama-convert-llama2c-to-ggml   llama-gguf-hash           llama-lookup       llama-perplexity      llama-simple            test-c              test-model-load-cancel        test-tokenizer-1-spm
~/downloads/ik_llama.cpp $ mv bin/ bin.1
~/downloads/ik_llama.cpp $ rm CMakeCache.txt 
~/downloads/ik_llama.cpp $ cmake .
-- The C compiler identification is Clang 20.1.3
-- The CXX compiler identification is Clang 20.1.3
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /data/data/com.termux/files/usr/bin/clang - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /data/data/com.termux/files/usr/bin/clang++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Found Git: /data/data/com.termux/files/usr/bin/git (found version "2.49.0")
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Failed
-- Check if compiler accepts -pthread
-- Check if compiler accepts -pthread - yes
-- Found Threads: TRUE
-- Found OpenMP_C: -fopenmp=libomp (found version "5.1")
-- Found OpenMP_CXX: -fopenmp=libomp (found version "5.1")
-- Found OpenMP: TRUE (found version "5.1")
-- OpenMP found
-- Using optimized iqk matrix multiplications
-- Using llamafile
-- ccache found, compilation results will be cached. Disable with GGML_CCACHE=OFF.
-- CMAKE_SYSTEM_PROCESSOR: aarch64
-- ARM detected
-- Performing Test COMPILER_SUPPORTS_FP16_FORMAT_I3E
-- Performing Test COMPILER_SUPPORTS_FP16_FORMAT_I3E - Failed
-- Looking for pthread_create in pthreads
-- Looking for pthread_create in pthreads - not found
-- Looking for pthread_create in pthread
-- Looking for pthread_create in pthread - found
-- Configuring done (12.1s)
-- Generating done (0.4s)
-- Build files have been written to: /data/data/com.termux/files/home/downloads/ik_llama.cpp
~/downloads/ik_llama.cpp $ make 
[  6%] Built target ggml
[ 10%] Built target llama
[ 11%] Built target build_info
[ 15%] Built target common
[ 16%] Linking CXX executable ../bin/test-tokenizer-0
[ 17%] Built target test-tokenizer-0
[ 18%] Linking CXX executable ../bin/test-tokenizer-1-bpe
[ 18%] Built target test-tokenizer-1-bpe
[ 19%] Linking CXX executable ../bin/test-tokenizer-1-spm
[ 19%] Built target test-tokenizer-1-spm
[ 19%] Linking CXX executable ../bin/test-quantize-fns
[ 20%] Built target test-quantize-fns
[ 21%] Linking CXX executable ../bin/test-quantize-perf
[ 22%] Built target test-quantize-perf
[ 22%] Linking CXX executable ../bin/test-sampling
[ 23%] Built target test-sampling
[ 23%] Linking CXX executable ../bin/test-chat-template
[ 24%] Built target test-chat-template
[ 24%] Linking CXX executable ../bin/test-grammar-parser
[ 25%] Built target test-grammar-parser
[ 26%] Linking CXX executable ../bin/test-llama-grammar
[ 27%] Built target test-llama-grammar
[ 28%] Linking CXX executable ../bin/test-grammar-integration
[ 29%] Built target test-grammar-integration
[ 30%] Linking CXX executable ../bin/test-grad0
[ 31%] Built target test-grad0
[ 31%] Linking CXX executable ../bin/test-backend-ops
[ 32%] Built target test-backend-ops
[ 33%] Linking CXX executable ../bin/test-rope
[ 34%] Built target test-rope
[ 35%] Linking CXX executable ../bin/test-model-load-cancel
[ 36%] Built target test-model-load-cancel
[ 37%] Linking CXX executable ../bin/test-autorelease
[ 38%] Built target test-autorelease
[ 38%] Linking CXX executable ../bin/test-json-schema-to-grammar
[ 40%] Built target test-json-schema-to-grammar
[ 41%] Linking C executable ../bin/test-c
[ 42%] Built target test-c
[ 42%] Linking CXX executable ../../bin/llama-cvector-generator
[ 43%] Built target llama-cvector-generator
[ 43%] Linking CXX executable ../../bin/llama-baby-llama
[ 44%] Built target llama-baby-llama
[ 44%] Linking CXX executable ../../bin/llama-batched-bench
[ 45%] Built target llama-batched-bench
[ 45%] Linking CXX executable ../../bin/llama-batched
[ 46%] Built target llama-batched
[ 47%] Linking CXX executable ../../bin/llama-bench-matmult
[ 47%] Built target llama-bench-matmult
[ 48%] Linking CXX executable ../../bin/llama-convert-llama2c-to-ggml
[ 48%] Built target llama-convert-llama2c-to-ggml
[ 48%] Linking CXX executable ../../bin/llama-embedding
[ 49%] Built target llama-embedding
[ 50%] Linking CXX executable ../../bin/llama-eval-callback
[ 51%] Built target llama-eval-callback
[ 52%] Linking CXX executable ../../bin/llama-export-lora
[ 52%] Built target llama-export-lora
[ 53%] Linking CXX executable ../../bin/llama-gbnf-validator
[ 53%] Built target llama-gbnf-validator
[ 54%] Built target sha256
[ 55%] Built target xxhash
[ 55%] Built target sha1
[ 55%] Linking CXX executable ../../bin/llama-gguf-hash
[ 56%] Built target llama-gguf-hash
[ 56%] Linking CXX executable ../../bin/llama-gguf-split
[ 57%] Built target llama-gguf-split
[ 58%] Linking CXX executable ../../bin/llama-gguf
[ 58%] Built target llama-gguf
[ 58%] Linking CXX executable ../../bin/llama-gritlm
[ 59%] Built target llama-gritlm
[ 60%] Linking CXX executable ../../bin/llama-imatrix
[ 61%] Built target llama-imatrix
[ 62%] Linking CXX executable ../../bin/llama-infill
[ 62%] Built target llama-infill
[ 63%] Linking CXX executable ../../bin/llama-bench
[ 64%] Built target llama-bench
[ 66%] Built target llava
[ 67%] Built target llava_static
[ 67%] Built target llava_shared
[ 68%] Linking CXX executable ../../bin/llama-llava-cli
[ 68%] Built target llama-llava-cli
[ 69%] Linking CXX executable ../../bin/llama-minicpmv-cli
[ 69%] Built target llama-minicpmv-cli
[ 70%] Linking CXX executable ../../bin/llama-lookahead
[ 70%] Built target llama-lookahead
[ 70%] Linking CXX executable ../../bin/llama-lookup
[ 71%] Built target llama-lookup
[ 71%] Linking CXX executable ../../bin/llama-lookup-create
[ 72%] Built target llama-lookup-create
[ 72%] Linking CXX executable ../../bin/llama-lookup-merge
[ 73%] Built target llama-lookup-merge
[ 74%] Linking CXX executable ../../bin/llama-lookup-stats
[ 75%] Built target llama-lookup-stats
[ 76%] Linking CXX executable ../../bin/llama-cli
[ 76%] Built target llama-cli
[ 77%] Linking CXX executable ../../bin/llama-parallel
[ 77%] Built target llama-parallel
[ 78%] Linking CXX executable ../../bin/llama-passkey
[ 78%] Built target llama-passkey
[ 78%] Linking CXX executable ../../bin/llama-perplexity
[ 79%] Built target llama-perplexity
[ 80%] Linking CXX executable ../../bin/llama-quantize-stats
[ 80%] Built target llama-quantize-stats
[ 81%] Linking CXX executable ../../bin/llama-quantize
[ 82%] Built target llama-quantize
[ 83%] Linking CXX executable ../../bin/llama-retrieval
[ 83%] Built target llama-retrieval
[ 84%] Linking CXX executable ../../bin/llama-server
[ 93%] Built target llama-server
[ 94%] Linking CXX executable ../../bin/llama-save-load-state
[ 94%] Built target llama-save-load-state
[ 95%] Linking CXX executable ../../bin/llama-simple
[ 95%] Built target llama-simple
[ 96%] Linking CXX executable ../../bin/llama-speculative
[ 96%] Built target llama-speculative
[ 96%] Linking CXX executable ../../bin/llama-sweep-bench
[ 97%] Built target llama-sweep-bench
[ 97%] Linking CXX executable ../../bin/llama-tokenize
[ 98%] Built target llama-tokenize
[ 98%] Linking CXX executable ../../bin/llama-vdot
[ 99%] Built target llama-vdot
[ 99%] Linking CXX executable ../../bin/llama-q8dot
[100%] Built target llama-q8dot
~/downloads/ik_llama.cpp $ ls bin/
 llama-baby-llama                llama-cvector-generator   llama-gguf-split   llama-lookup-create   llama-q8dot             llama-speculative   test-chat-template            test-quantize-fns
 llama-batched                   llama-embedding           llama-gritlm       llama-lookup-merge    llama-quantize          llama-sweep-bench   test-grad0                    test-quantize-perf
 llama-batched-bench             llama-eval-callback       llama-imatrix      llama-lookup-stats    llama-quantize-stats    llama-tokenize      test-grammar-integration      test-rope
 llama-bench                     llama-export-lora         llama-infill       llama-minicpmv-cli    llama-retrieval         llama-vdot          test-grammar-parser           test-sampling
 llama-bench-matmult             llama-gbnf-validator      llama-llava-cli    llama-parallel        llama-save-load-state   test-autorelease    test-json-schema-to-grammar   test-tokenizer-0
 llama-cli                       llama-gguf                llama-lookahead    llama-passkey         llama-server            test-backend-ops    test-llama-grammar            test-tokenizer-1-bpe
 llama-convert-llama2c-to-ggml   llama-gguf-hash           llama-lookup       llama-perplexity      llama-simple            test-c              test-model-load-cancel        test-tokenizer-1-spm
~/downloads/ik_llama.cpp $ ldd bin/llama-cli 
	liblog.so => /system/lib64/liblog.so
	libargp.so => /data/data/com.termux/files/usr/lib/libargp.so
	libllama.so => /data/data/com.termux/files/usr/lib/libllama.so
	libc.so => /system/lib64/libc.so
	libggml.so => /data/data/com.termux/files/usr/lib/libggml.so
	libc++_shared.so => /data/data/com.termux/files/usr/lib/libc++_shared.so
	libdl.so => /system/lib64/libdl.so
	libm.so => /system/lib64/libm.so
	libc++.so => /system/lib64/libc++.so
	ld-android.so => /system/lib64/ld-android.so
	libggml-cpu.so => /data/data/com.termux/files/usr/lib/libggml-cpu.so
	libggml-base.so => /data/data/com.termux/files/usr/lib/libggml-base.so
~/downloads/ik_llama.cpp $  bin/llama-cli 
CANNOT LINK EXECUTABLE "bin/llama-cli": cannot locate symbol "llama_print_timings" referenced by "/data/data/com.termux/files/home/downloads/ik_llama.cpp/bin/llama-cli"...
~/downloads/ik_llama.cpp $ 

```

Only after my trick above it picks up the rigth .so files: 

```
~/downloads/ik_llama.cpp $ cat _path.sh 
export LD_LIBRARY_PATH=$(pwd)/src/:$(pwd)/ggml/src/:$LD_LIBRARY_PATH
~/downloads/ik_llama.cpp $ source  _path.sh 
~/downloads/ik_llama.cpp $ ldd bin/llama-cli 
	liblog.so => /system/lib64/liblog.so
	libargp.so => /data/data/com.termux/files/usr/lib/libargp.so
	libllama.so => /data/data/com.termux/files/home/downloads/ik_llama.cpp/src/libllama.so
	libc.so => /system/lib64/libc.so
	libggml.so => /data/data/com.termux/files/home/downloads/ik_llama.cpp/ggml/src/libggml.so
	libc++_shared.so => /data/data/com.termux/files/usr/lib/libc++_shared.so
	libdl.so => /system/lib64/libdl.so
	libm.so => /system/lib64/libm.so
	libc++.so => /system/lib64/libc++.so
	ld-android.so => /system/lib64/ld-android.so
~/downloads/ik_llama.cpp $ 

``` 
I shall `mv` once again and retry your `"-DGGML_ARCH_FLAGS="-march=armv8.2-a+dotprod+fp16"` ...

---

👤 **Manamama** commented the **2025-05-06** at **20:22:09**:<br>

The experiment with the flags (methinks, they should not help here, it is the `rpath` type problem) -  sorry for pasting all together - do take a peek at my juggling the LD_LIBRARY_PATH to default there so as to evoke that seg fault at first: 

```
~/downloads/ik_llama.cpp $ bin/llama-cli 
Log start
main: build = 3668 (6c23618c)
main: built with clang version 20.1.3 for aarch64-unknown-linux-android24
main: seed  = 1746562290
gguf_init_from_file: failed to open 'models/7B/ggml-model-f16.gguf': 'No such file or directory'
llama_model_load: error loading model: llama_model_loader: failed to load model from models/7B/ggml-model-f16.gguf

llama_load_model_from_file: failed to load model
llama_init_from_gpt_params: error: failed to load model 'models/7B/ggml-model-f16.gguf'
main: error: unable to load model
~/downloads/ik_llama.cpp $ export LD_LIBRARY_PATH=
~/downloads/ik_llama.cpp $ bin/llama-cli 
CANNOT LINK EXECUTABLE "bin/llama-cli": cannot locate symbol "llama_print_timings" referenced by "/data/data/com.termux/files/home/downloads/ik_llama.cpp/bin/llama-cli"...
~/downloads/ik_llama.cpp $ mv bin/ bin.2
~/downloads/ik_llama.cpp $ rm CMakeCache.txt 
~/downloads/ik_llama.cpp $ cmake . -DGGML_ARCH_FLAGS=-march=armv8.2-a+dotprod+fp16
-- The C compiler identification is Clang 20.1.3
-- The CXX compiler identification is Clang 20.1.3
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /data/data/com.termux/files/usr/bin/clang - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /data/data/com.termux/files/usr/bin/clang++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Found Git: /data/data/com.termux/files/usr/bin/git (found version "2.49.0")
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Failed
-- Check if compiler accepts -pthread
-- Check if compiler accepts -pthread - yes
-- Found Threads: TRUE
-- Found OpenMP_C: -fopenmp=libomp (found version "5.1")
-- Found OpenMP_CXX: -fopenmp=libomp (found version "5.1")
-- Found OpenMP: TRUE (found version "5.1")
-- OpenMP found
-- Using optimized iqk matrix multiplications
-- Using llamafile
-- ccache found, compilation results will be cached. Disable with GGML_CCACHE=OFF.
-- CMAKE_SYSTEM_PROCESSOR: aarch64
-- ARM detected
-- Performing Test COMPILER_SUPPORTS_FP16_FORMAT_I3E
-- Performing Test COMPILER_SUPPORTS_FP16_FORMAT_I3E - Failed
-- Looking for pthread_create in pthreads
-- Looking for pthread_create in pthreads - not found
-- Looking for pthread_create in pthread
-- Looking for pthread_create in pthread - found
-- Configuring done (12.6s)
-- Generating done (1.1s)
-- Build files have been written to: /data/data/com.termux/files/home/downloads/ik_llama.cpp
~/downloads/ik_llama.cpp $ make
[  0%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml-aarch64.c.o
 ```

and hangs up at this for 5 minutes, a dejavu from BitNet compilation, before my hacks, maybe this one is relevant: https://github.com/microsoft/BitNet/issues/206#issuecomment-2847884139 


```
~/downloads/ik_llama.cpp $ make
[  0%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml-aarch64.c.o
[  0%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml-alloc.c.o
[  1%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml-backend.c.o
[  2%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml-quants.c.o
[  3%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml.c.o
[  3%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_flash_attn.cpp.o
[  4%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_mul_mat.cpp.o







```
[has not progressed, while clang takes some 12 percent  of CPU. ]

Retrying with: 
```


^Cmake[2]: *** [ggml/src/CMakeFiles/ggml.dir/build.make:149: ggml/src/CMakeFiles/ggml.dir/iqk/iqk_mul_mat.cpp.o] Interrupt
make[1]: *** [CMakeFiles/Makefile2:2022: ggml/src/CMakeFiles/ggml.dir/all] Interrupt
make: *** [Makefile:146: all] Interrupt

~/downloads/ik_llama.cpp $ make -j8
make: jobserver mkfifo: /data/local/tmp/GMfifo22430: Permission denied
[  1%] Built target sha256
[  2%] Built target build_info
[  3%] Built target xxhash
[  3%] Built target sha1
[  4%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_mul_mat.cpp.o
[  5%] Building CXX object ggml/src/CMakeFiles/ggml.dir/llamafile/sgemm.cpp.o
[  5%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_quantize.cpp.o




```
...

---

👤 **saood06** commented the **2025-05-06** at **20:36:26**:<br>

>[has not progressed, while clang takes some 12 percent of CPU. ]

Are you sure? I remember from when I was testing on Android, building the `iqk` files took a while (the only time they built quickly was without the flags, when they were being built but effectively turned off)

---

👤 **Manamama** commented the **2025-05-06** at **20:39:52**:<br>

OK, after probably half an hour (vs the asap compilation   without these switches): 

```
[ 87%] Linking CXX executable ../../bin/llama-vdot
[ 88%] Built target llama-sweep-bench
[ 89%] Built target llama-speculative
[ 89%] Built target llama-tokenize
[ 89%] Linking CXX executable ../../bin/llama-q8dot
[ 90%] Built target llama-vdot
[ 91%] Built target llama-q8dot
[100%] Built target llama-server
~/downloads/ik_llama.cpp $ 
~/downloads/ik_llama.cpp $ 
~/downloads/ik_llama.cpp $ 
~/downloads/ik_llama.cpp $ 
~/downloads/ik_llama.cpp $ 
~/downloads/ik_llama.cpp $ 
~/downloads/ik_llama.cpp $ 
~/downloads/ik_llama.cpp $ 
~/downloads/ik_llama.cpp $ 
~/downloads/ik_llama.cpp $ 
~/downloads/ik_llama.cpp $ 
~/downloads/ik_llama.cpp $ 
~/downloads/ik_llama.cpp $ 
~/downloads/ik_llama.cpp $ ldd bin/llama-cli 
	liblog.so => /system/lib64/liblog.so
	libargp.so => /data/data/com.termux/files/usr/lib/libargp.so
	libllama.so => /data/data/com.termux/files/usr/lib/libllama.so
	libc.so => /system/lib64/libc.so
	libggml.so => /data/data/com.termux/files/usr/lib/libggml.so
	libc++_shared.so => /data/data/com.termux/files/usr/lib/libc++_shared.so
	libdl.so => /system/lib64/libdl.so
	libm.so => /system/lib64/libm.so
	libc++.so => /system/lib64/libc++.so
	ld-android.so => /system/lib64/ld-android.so
	libggml-cpu.so => /data/data/com.termux/files/usr/lib/libggml-cpu.so
	libggml-base.so => /data/data/com.termux/files/usr/lib/libggml-base.so
~/downloads/ik_llama.cpp $ bin/llama-cli 
CANNOT LINK EXECUTABLE "bin/llama-cli": cannot locate symbol "llama_print_timings" referenced by "/data/data/com.termux/files/home/downloads/ik_llama.cpp/bin/llama-cli"...
~/downloads/ik_llama.cpp $ 

```

So `rpath` like is needed (or my ugly trick) , reminder why: 
```
~/downloads/ik_llama.cpp $ cat _path.sh 
export LD_LIBRARY_PATH=$(pwd)/src/:$(pwd)/ggml/src/:$LD_LIBRARY_PATH
~/downloads/ik_llama.cpp $ echo $LD_LIBRARY_PATH

~/downloads/ik_llama.cpp $ source  _path.sh 
~/downloads/ik_llama.cpp $ ldd bin/llama-cli 
	liblog.so => /system/lib64/liblog.so
	libargp.so => /data/data/com.termux/files/usr/lib/libargp.so
	libllama.so => /data/data/com.termux/files/home/downloads/ik_llama.cpp/src/libllama.so
	libc.so => /system/lib64/libc.so
	libggml.so => /data/data/com.termux/files/home/downloads/ik_llama.cpp/ggml/src/libggml.so
	libc++_shared.so => /data/data/com.termux/files/usr/lib/libc++_shared.so
	libdl.so => /system/lib64/libdl.so
	libm.so => /system/lib64/libm.so
	libc++.so => /system/lib64/libc++.so
	ld-android.so => /system/lib64/ld-android.so
~/downloads/ik_llama.cpp $  bin/llama-cli 
Log start
main: build = 3668 (6c23618c)
main: built with clang version 20.1.3 for aarch64-unknown-linux-android24
main: seed  = 1746564079
...
```

---

👤 **Manamama** commented the **2025-05-06** at **20:39:52**:<br>

OK, after probably half an hour (vs the asap compilation   without these switches): 

```
[ 87%] Linking CXX executable ../../bin/llama-vdot
[ 88%] Built target llama-sweep-bench
[ 89%] Built target llama-speculative
[ 89%] Built target llama-tokenize
[ 89%] Linking CXX executable ../../bin/llama-q8dot
[ 90%] Built target llama-vdot
[ 91%] Built target llama-q8dot
[100%] Built target llama-server
~/downloads/ik_llama.cpp $ 
~/downloads/ik_llama.cpp $ 
~/downloads/ik_llama.cpp $ 
~/downloads/ik_llama.cpp $ 
~/downloads/ik_llama.cpp $ 
~/downloads/ik_llama.cpp $ 
~/downloads/ik_llama.cpp $ 
~/downloads/ik_llama.cpp $ 
~/downloads/ik_llama.cpp $ 
~/downloads/ik_llama.cpp $ 
~/downloads/ik_llama.cpp $ 
~/downloads/ik_llama.cpp $ 
~/downloads/ik_llama.cpp $ 
~/downloads/ik_llama.cpp $ ldd bin/llama-cli 
	liblog.so => /system/lib64/liblog.so
	libargp.so => /data/data/com.termux/files/usr/lib/libargp.so
	libllama.so => /data/data/com.termux/files/usr/lib/libllama.so
	libc.so => /system/lib64/libc.so
	libggml.so => /data/data/com.termux/files/usr/lib/libggml.so
	libc++_shared.so => /data/data/com.termux/files/usr/lib/libc++_shared.so
	libdl.so => /system/lib64/libdl.so
	libm.so => /system/lib64/libm.so
	libc++.so => /system/lib64/libc++.so
	ld-android.so => /system/lib64/ld-android.so
	libggml-cpu.so => /data/data/com.termux/files/usr/lib/libggml-cpu.so
	libggml-base.so => /data/data/com.termux/files/usr/lib/libggml-base.so
~/downloads/ik_llama.cpp $ bin/llama-cli 
CANNOT LINK EXECUTABLE "bin/llama-cli": cannot locate symbol "llama_print_timings" referenced by "/data/data/com.termux/files/home/downloads/ik_llama.cpp/bin/llama-cli"...
~/downloads/ik_llama.cpp $ 

```

So `rpath` like is needed (or my ugly trick).

---

👤 **ikawrakow** commented the **2025-05-07** at **07:00:41**:<br>

> OK, after probably half an hour (vs the asap compilation without these switches):

ASAP compilation means the resulting build is useless. The `iqk_mul_mat.cpp` file that takes a very long time to compile is 18,000 lines of heavily templated C++ code, so yes, it takes a long time to compile. There is issue #183 precisely because of that.

Concerning the clash with mainline `llama.cpp`: OK, so this project does not consider the possibility of having mainline installed to a system-wide directory, and then trying to use `ik_llama.cpp` built in a user folder. So, yes, you need to use something like `LD_LIBRARY_PATH` to have the user build directory searched first.

---

👤 **ikawrakow** commented the **2025-05-25** at **07:09:05**:<br>

I don't think we will be solving this one.

---

👤 **Manamama** commented the **2025-05-25** at **18:13:04**:<br>

Note to self: 
```
~/downloads $ apt list | grep llama-                                                                          WARNING: apt does not have a stable CLI interface. Use with caution in scripts.                                                                                      llama-cpp-backend-opencl/stable 0.0.0-b5481-0 aarch64  llama-cpp-backend-vulkan/stable 0.0.0-b5481-0 aarch64  llama-cpp/stable,now 0.0.0-b5481-0 aarch64 [installed] ~/downloads $
```
So either removing or reinstalling `llama-cpp` seems to help. Not sure why - I suspect the .so version clashes...