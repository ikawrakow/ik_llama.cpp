### 🐛 [#464](https://github.com/ikawrakow/ik_llama.cpp/issues/464) - Bug: The streaming every couple of rows blocks for 5-8s

| **Author** | `ciprianveg` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2025-05-27 |
| **Updated** | 2025-06-01 |

---

#### Description

### What happened?

Although I obtained good sweep-bench results for 235b UD_Q5_XL as shown below, and with the q4 quant they were 20% faster, in both cases, this annoying blocking happens every couple of rows. I tried changing from 16 threads to 12, but same thing happens. Wilth main llama, is like 25% slower, but is cursive.
My system is a TR 3955wx with 16 cores, 256 ddr4 3200, 2x3090..
Any ideas?   
./build/bin/llama-sweep-bench --model /home/ciprian/ai/models/Qwen3-235B-UD_Q5_XL/Qwen3-235B-A22B-UD-Q5_K_XL-00001-of-00004.gguf --alias Qwen3-235B-A22B-UD-Q5_K_XL -fa -fmoe  -ctk q8_0 -ctv q8_0 -c 40960  --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0 --presence-penalty 0.5 -ot "blk\.[0-9]\.ffn_up_exps=CUDA0,blk\.[0-9]\.ffn_gate_exps=CUDA0,blk\.2[0-4]\.ffn_up_exps=CUDA0,blk\.2[0-4]\.ffn_gate_exps=CUDA0,blk\.1[0-9]\.ffn_up_exps=CUDA1,blk\.1[0-9]\.ffn_gate_exps=CUDA1,blk\.2[5-8]\.ffn_up_exps=CUDA1,blk\.2[5-8]\.ffn_gate_exps=CUDA1,exps=CPU"  -ngl 99 --threads 16 --host 0.0.0.0 --port 5002    --ubatch-size 4096 --batch-size 4096
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |   11.730 |   349.19 |  133.500 |     7.67 |
|  4096 |   1024 |   4096 |   12.079 |   339.11 |  136.944 |     7.48 |
|  4096 |   1024 |   8192 |   12.514 |   327.33 |  140.286 |     7.30 |
|  4096 |   1024 |  12288 |   13.038 |   314.17 |  144.478 |     7.09 |
|  4096 |   1024 |  16384 |   13.545 |   302.40 |  148.595 |     6.89 |
|  4096 |   1024 |  20480 |   13.943 |   293.76 |  151.881 |     6.74 |
|  4096 |   1024 |  24576 |   14.767 |   277.38 |  154.643 |     6.62 |
|  4096 |   1024 |  28672 |   15.621 |   262.21 |  158.355 |     6.47 |
|  4096 |   1024 |  32768 |   16.561 |   247.32 |  161.875 |     6.33 |
|  4096 |   1024 |  36864 |   17.658 |   231.97 |  166.160 |     6.16 |

### Name and Version

llama-server -model /home/ciprian/ai/models/Qwen3-235B-UD_Q5_XL/Qwen3-235B-A22B-UD-Q5_K_XL-00001-of-00004.gguf --alias Qwen3-235B-A22B-UD-Q5_K_XL -fa -fmoe  -ctk q8_0 -ctv q8_0 -c 40960  --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0 --presence-penalty 0.5 -ot "blk\.[0-9]\.ffn_up_exps=CUDA0,blk\.[0-9]\.ffn_gate_exps=CUDA0,blk\.2[0-4]\.ffn_up_exps=CUDA0,blk\.2[0-4]\.ffn_gate_exps=CUDA0,blk\.1[0-9]\.ffn_up_exps=CUDA1,blk\.1[0-9]\.ffn_gate_exps=CUDA1,blk\.2[5-8]\.ffn_up_exps=CUDA1,blk\.2[5-8]\.ffn_gate_exps=CUDA1,exps=CPU"  -ngl 99 --threads 16 --host 0.0.0.0 --port 5002    --ubatch-size 4096 --batch-size 4096

### What operating system are you seeing the problem on?

_No response_

### Relevant log output

```shell

```

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-05-28** at **05:17:25**:<br>

Not sure. Do you get many tokens at once after the 5-8 seconds pause, or it just did nothing for 5-8 seconds?

---

👤 **ciprianveg** commented the **2025-05-28** at **06:19:29**:<br>

It looks like it did nothing,  sometimes a second 5-8 s pause comes after
just 2 words, other times after 2 rows of text. I tried also with 2048
ubatch size and with using amb 512, no difference. For my hardware, what
would be the most suitable build params. I am now setting gpu ggml sched
copies to 1, cublast off and ggml cuda on

On Wed, 28 May 2025, 08:17 Kawrakow, ***@***.***> wrote:

> *ikawrakow* left a comment (ikawrakow/ik_llama.cpp#464)
> <https://github.com/ikawrakow/ik_llama.cpp/issues/464#issuecomment-2914966626>
>
> Not sure. Do you get many tokens at once after the 5-8 seconds pause, or
> it just did nothing for 5-8 seconds?
>
> —
> Reply to this email directly, view it on GitHub
> <https://github.com/ikawrakow/ik_llama.cpp/issues/464#issuecomment-2914966626>,
> or unsubscribe
> <https://github.com/notifications/unsubscribe-auth/AJTBYK5CNGBBG5RUZEC6VBT3AVBHVAVCNFSM6AAAAAB6BBXI7KVHI2DSMVQWIX3LMV43OSLTON2WKQ3PNVWWK3TUHMZDSMJUHE3DMNRSGY>
> .
> You are receiving this because you authored the thread.Message ID:
> ***@***.***>
>

---

👤 **ikawrakow** commented the **2025-05-28** at **07:56:05**:<br>

I'm trying to understand the root cause for this strange behavior. Can you reproduce it using `llama-cli` ?

---

👤 **ciprianveg** commented the **2025-05-28** at **10:01:30**:<br>

I will try this evening and let you know

---

👤 **ciprianveg** commented the **2025-05-28** at **13:06:09**:<br>

Something that maybe can give a clue is that my system is cpu limited, i have 8 channels ddr4 3200 ram but the memory read speed is limited to 85Mb/s instead of the theoretical >200Mb/s because the 16 cores are not enough to read at that speed. This is against the standard cpu systems where memory speed is the limiter, not the cpu..

---

👤 **ciprianveg** commented the **2025-05-28** at **16:52:25**:<br>

same issue also with llama-cli

---

👤 **ikawrakow** commented the **2025-05-28** at **17:22:10**:<br>

Is there disc activity during the pause? Have you looked at process activity during the pause? Are you running llama.cpp with the exact same parameters (apart from -fmoe)? Is there another memory hungry process running (e.g., another llama.cpp server)?

---

👤 **ciprianveg** commented the **2025-05-28** at **17:27:38**:<br>

Llama.cpp runs with exact params except fmoe. I have 256Gb ram and almost
100gb free. No other memory hungry process..

On Wed, 28 May 2025, 20:22 Kawrakow, ***@***.***> wrote:

> *ikawrakow* left a comment (ikawrakow/ik_llama.cpp#464)
> <https://github.com/ikawrakow/ik_llama.cpp/issues/464#issuecomment-2917078274>
>
> Is there disc activity during the pause? Have you looked at process
> activity during the pause? Are you running llama.cpp with the exact same
> parameters (apart from -fmoe)? Is there another memory hungry process
> running (e.g., another llama.cpp server)?
>
> —
> Reply to this email directly, view it on GitHub
> <https://github.com/ikawrakow/ik_llama.cpp/issues/464#issuecomment-2917078274>,
> or unsubscribe
> <https://github.com/notifications/unsubscribe-auth/AJTBYK64C6EOOT2D5LADL733AXWFPAVCNFSM6AAAAAB6BBXI7KVHI2DSMVQWIX3LMV43OSLTON2WKQ3PNVWWK3TUHMZDSMJXGA3TQMRXGQ>
> .
> You are receiving this because you authored the thread.Message ID:
> ***@***.***>
>

---

👤 **ikawrakow** commented the **2025-05-29** at **04:19:32**:<br>

What about the first two questions? Is the CPU busy during the pauses or just sitting there doing nothing? But at the end it might be easier to just run in the debugger and when it pauses, hit Ctrl-C, type `bt`, and post the backtrace here.

---

👤 **ciprianveg** commented the **2025-05-29** at **05:35:10**:<br>

1. Disk activity, no
2. Top shows llama server between 100-500% when it works and same when it pauses

---

👤 **kirnat** commented the **2025-05-29** at **09:12:01**:<br>

Check your PCIe traffic with nvtop or similar when the pause happens. Does it happen if you don't offload any experts to the GPUs?

---

👤 **ikawrakow** commented the **2025-05-29** at **09:31:47**:<br>

To test the hypothesis that it gets stuck on copying tensors to the GPU, you can run with `-op 26,0,27,0,29,0`. This disables offloading tensors to the GPU for any type of matrix multiplication.

But running in the debugger, interrupting with Ctrl-C when it gets stuck, and sending the backtrace will hopefully also diagnose where (in which function) it hangs for so long.

---

👤 **ciprianveg** commented the **2025-05-29** at **09:44:38**:<br>

XXXXXXXXXXXXXXXXXXXXX Setting offload policy for op MUL_MAT to OFF
XXXXXXXXXXXXXXXXXXXXX Setting offload policy for op MUL_MAT_ID to OFF
XXXXXXXXXXXXXXXXXXXXX Setting offload policy for op MOE_FUSED_UP_GATE to OFF
XXXXXXXXXXXXXXXXXXXXXXXXXXXX offload(MUL_MAT) = 0
XXXXXXXXXXXXXXXXXXXXXXXXXXXX offload(MUL_MAT_ID) = 0
XXXXXXXXXXXXXXXXXXXXXXXXXXXX offload(MOE_FUSED_UP_GATE) = 0

same issue

---

👤 **ciprianveg** commented the **2025-05-29** at **09:54:13**:<br>

Thread 1 "llama-server" received signal SIGINT, Interrupt.
Download failed: Invalid argument.  Continuing without source file ./nptl/./nptl/pthread_mutex_lock.c.
0x00007fffee4a014c in lll_mutex_lock_optimized (mutex=0x55555899a0d8) at ./nptl/pthread_mutex_lock.c:48
warning: 48	./nptl/pthread_mutex_lock.c: No such file or directory

this is from debug

also, with nvtop, when pause happens, the gpus transfer speed is around 1,8GB/s and as soon as it unblocks drops to 50-100MB/s

---

👤 **ciprianveg** commented the **2025-05-29** at **09:54:13**:<br>

Thread 1 "llama-server" received signal SIGINT, Interrupt.
Download failed: Invalid argument.  Continuing without source file ./nptl/./nptl/pthread_mutex_lock.c.
0x00007fffee4a014c in lll_mutex_lock_optimized (mutex=0x55555899a0d8) at ./nptl/pthread_mutex_lock.c:48
warning: 48	./nptl/pthread_mutex_lock.c: No such file or directory

---

👤 **ciprianveg** commented the **2025-05-29** at **13:02:59**:<br>

it happened also with ngl 0, with nothing sent to gpus, only that being slower, like 2-3tok/s also the pause was longer, cca 20s

llama-server --model /home/ciprian/ai/models/Qwen3-235B-UD_Q4_XL/Qwen3-235B-A22B-UD-Q4_K_XL-00001-of-00003.gguf --alias Qwen3-235B-A22B-UD-Q4_K_XL -fa  -ctk q8_0 -ctv q8_0 -c 36864  --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0 --presence-penalty 0.5 -ngl 0 --threads 16 --host 0.0.0.0 --port 5002    --ubatch-size 4096 --batch-size 4096

---

👤 **ikawrakow** commented the **2025-05-29** at **13:22:24**:<br>

If you want to test if the pauses happen when running CPU only, you need to say `CUDA_VISIBLE_DEVICES="" ./bin/llama-server...`. Or just make a build with CUDA disabled.

The debug session above was not useful as the main thread is the server thread, so we don't see where the computation hangs. To get the desired backtrace you need to run `llama-cli`.

> the gpus transfer speed is around 1,8GB/s and as soon as it unblocks drops to 50-100MB/s

Isn't this kind of slow? But even at that rate in 5 seconds it will transfer ~9 GB to the GPU. A `Q5_K` quantized Qwen3-235-A22B layer is in the range of 1.8 GB, so it is transferring 5 layers worth of tensors?

Or is this all happening when your context gets full?

---

👤 **ciprianveg** commented the **2025-05-29** at **13:59:44**:<br>

debug on llama-cli ctrl+c when paused i don't think is helpful:
Thread 1 "llama-cli" received signal SIGINT, Interrupt.
0x00007fffe5391028 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
(gdb)

---

👤 **ikawrakow** commented the **2025-05-29** at **15:42:57**:<br>

I guess you need
```
thread apply all bt
```

---

👤 **ciprianveg** commented the **2025-05-29** at **17:40:41**:<br>

Hi @ikawrakow:
0x00007fffe5391024 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
(gdb) thread apply all bt

Thread 21 (Thread 0x7fff647db000 (LWP 18073) "llama-cli"):
#0  futex_wait (addr=0x55555f628314, val=60160) at ../../../src/libgomp/config/linux/x86/futex.h:97
#1  do_wait (addr=<optimized out>, val=60160) at ../../../src/libgomp/config/linux/wait.h:67
#2  gomp_barrier_wait_end (bar=0x55555f628310, state=60160) at ../../../src/libgomp/config/linux/bar.c:48
#3  0x00007ffff7c87779 in gomp_simple_barrier_wait (bar=<optimized out>) at ../../../src/libgomp/config/posix/simple-bar.h:60
#4  gomp_thread_start (xdata=<optimized out>) at ../../../src/libgomp/team.c:133
#5  0x00007fffee49caa4 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
#6  0x00007fffee529c3c in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78

Thread 20 (Thread 0x7fff64fdc000 (LWP 18072) "llama-cli"):
#0  futex_wait (addr=0x55555f628314, val=60160) at ../../../src/libgomp/config/linux/x86/futex.h:97
#1  do_wait (addr=<optimized out>, val=60160) at ../../../src/libgomp/config/linux/wait.h:67
#2  gomp_barrier_wait_end (bar=0x55555f628310, state=60160) at ../../../src/libgomp/config/linux/bar.c:48
#3  0x00007ffff7c87779 in gomp_simple_barrier_wait (bar=<optimized out>) at ../../../src/libgomp/config/posix/simple-bar.h:60
#4  gomp_thread_start (xdata=<optimized out>) at ../../../src/libgomp/team.c:133
#5  0x00007fffee49caa4 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
#6  0x00007fffee529c3c in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78

Thread 19 (Thread 0x7fff657dd000 (LWP 18071) "llama-cli"):
#0  futex_wait (addr=0x55555f628314, val=60160) at ../../../src/libgomp/config/linux/x86/futex.h:97
--Type <RET> for more, q to quit, c to continue without paging--
#1  do_wait (addr=<optimized out>, val=60160) at ../../../src/libgomp/config/linux/wait.h:67
#2  gomp_barrier_wait_end (bar=0x55555f628310, state=60160) at ../../../src/libgomp/config/linux/bar.c:48
#3  0x00007ffff7c87779 in gomp_simple_barrier_wait (bar=<optimized out>) at ../../../src/libgomp/config/posix/simple-bar.h:60
#4  gomp_thread_start (xdata=<optimized out>) at ../../../src/libgomp/team.c:133
#5  0x00007fffee49caa4 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
#6  0x00007fffee529c3c in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78

Thread 18 (Thread 0x7fff65fde000 (LWP 18070) "llama-cli"):
#0  futex_wait (addr=0x55555f628314, val=60160) at ../../../src/libgomp/config/linux/x86/futex.h:97
#1  do_wait (addr=<optimized out>, val=60160) at ../../../src/libgomp/config/linux/wait.h:67
#2  gomp_barrier_wait_end (bar=0x55555f628310, state=60160) at ../../../src/libgomp/config/linux/bar.c:48
#3  0x00007ffff7c87779 in gomp_simple_barrier_wait (bar=<optimized out>) at ../../../src/libgomp/config/posix/simple-bar.h:60
#4  gomp_thread_start (xdata=<optimized out>) at ../../../src/libgomp/team.c:133
#5  0x00007fffee49caa4 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
#6  0x00007fffee529c3c in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78

Thread 17 (Thread 0x7fff667df000 (LWP 18069) "llama-cli"):
#0  futex_wait (addr=0x55555f628314, val=60160) at ../../../src/libgomp/config/linux/x86/futex.h:97
#1  do_wait (addr=<optimized out>, val=60160) at ../../../src/libgomp/config/linux/wait.h:67
#2  gomp_barrier_wait_end (bar=0x55555f628310, state=60160) at ../../../src/libgomp/config/linux/bar.c:48
#3  0x00007ffff7c87779 in gomp_simple_barrier_wait (bar=<optimized out>) at ../../../src/libgomp/config/posix/simple-bar.h:6--Type <RET> for more, q to quit, c to continue without paging--
0
#4  gomp_thread_start (xdata=<optimized out>) at ../../../src/libgomp/team.c:133
#5  0x00007fffee49caa4 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
#6  0x00007fffee529c3c in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78

Thread 16 (Thread 0x7fff66fe0000 (LWP 18068) "llama-cli"):
#0  futex_wait (addr=0x55555f628314, val=60160) at ../../../src/libgomp/config/linux/x86/futex.h:97
#1  do_wait (addr=<optimized out>, val=60160) at ../../../src/libgomp/config/linux/wait.h:67
#2  gomp_barrier_wait_end (bar=0x55555f628310, state=60160) at ../../../src/libgomp/config/linux/bar.c:48
#3  0x00007ffff7c87779 in gomp_simple_barrier_wait (bar=<optimized out>) at ../../../src/libgomp/config/posix/simple-bar.h:60
#4  gomp_thread_start (xdata=<optimized out>) at ../../../src/libgomp/team.c:133
#5  0x00007fffee49caa4 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
#6  0x00007fffee529c3c in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78

Thread 15 (Thread 0x7fff677e1000 (LWP 18067) "llama-cli"):
#0  futex_wait (addr=0x55555f628314, val=60160) at ../../../src/libgomp/config/linux/x86/futex.h:97
#1  do_wait (addr=<optimized out>, val=60160) at ../../../src/libgomp/config/linux/wait.h:67
#2  gomp_barrier_wait_end (bar=0x55555f628310, state=60160) at ../../../src/libgomp/config/linux/bar.c:48
#3  0x00007ffff7c87779 in gomp_simple_barrier_wait (bar=<optimized out>) at ../../../src/libgomp/config/posix/simple-bar.h:60
#4  gomp_thread_start (xdata=<optimized out>) at ../../../src/libgomp/team.c:133
#5  0x00007fffee49caa4 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
--Type <RET> for more, q to quit, c to continue without paging--
#6  0x00007fffee529c3c in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78

Thread 14 (Thread 0x7fff67fe2000 (LWP 18066) "llama-cli"):
#0  futex_wait (addr=0x55555f628314, val=60160) at ../../../src/libgomp/config/linux/x86/futex.h:97
#1  do_wait (addr=<optimized out>, val=60160) at ../../../src/libgomp/config/linux/wait.h:67
#2  gomp_barrier_wait_end (bar=0x55555f628310, state=60160) at ../../../src/libgomp/config/linux/bar.c:48
#3  0x00007ffff7c87779 in gomp_simple_barrier_wait (bar=<optimized out>) at ../../../src/libgomp/config/posix/simple-bar.h:60
#4  gomp_thread_start (xdata=<optimized out>) at ../../../src/libgomp/team.c:133
#5  0x00007fffee49caa4 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
#6  0x00007fffee529c3c in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78

Thread 13 (Thread 0x7fff687e3000 (LWP 18065) "llama-cli"):
#0  futex_wait (addr=0x55555f628314, val=60160) at ../../../src/libgomp/config/linux/x86/futex.h:97
#1  do_wait (addr=<optimized out>, val=60160) at ../../../src/libgomp/config/linux/wait.h:67
#2  gomp_barrier_wait_end (bar=0x55555f628310, state=60160) at ../../../src/libgomp/config/linux/bar.c:48
#3  0x00007ffff7c87779 in gomp_simple_barrier_wait (bar=<optimized out>) at ../../../src/libgomp/config/posix/simple-bar.h:60
#4  gomp_thread_start (xdata=<optimized out>) at ../../../src/libgomp/team.c:133
#5  0x00007fffee49caa4 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
#6  0x00007fffee529c3c in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78

Thread 12 (Thread 0x7fff68fe4000 (LWP 18064) "llama-cli"):
#0  futex_wait (addr=0x55555f628314, val=60160) at ../../../src/libgomp/config/linux/x86/futex.h:97
#1  do_wait (addr=<optimized out>, val=60160) at ../../../src/libgomp/config/linux/wait.h:67
#2  gomp_barrier_wait_end (bar=0x55555f628310, state=60160) at ../../../src/libgomp/config/linux/bar.c:48
#3  0x00007ffff7c87779 in gomp_simple_barrier_wait (bar=<optimized out>) at ../../../src/libgomp/config/posix/simple-bar.h:6--Type <RET> for more, q to quit, c to continue without paging--
0
#4  gomp_thread_start (xdata=<optimized out>) at ../../../src/libgomp/team.c:133
#5  0x00007fffee49caa4 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
#6  0x00007fffee529c3c in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78

Thread 11 (Thread 0x7fff697e5000 (LWP 18063) "llama-cli"):
#0  futex_wait (addr=0x55555f628314, val=60160) at ../../../src/libgomp/config/linux/x86/futex.h:97
#1  do_wait (addr=<optimized out>, val=60160) at ../../../src/libgomp/config/linux/wait.h:67
#2  gomp_barrier_wait_end (bar=0x55555f628310, state=60160) at ../../../src/libgomp/config/linux/bar.c:48
#3  0x00007ffff7c87779 in gomp_simple_barrier_wait (bar=<optimized out>) at ../../../src/libgomp/config/posix/simple-bar.h:60
#4  gomp_thread_start (xdata=<optimized out>) at ../../../src/libgomp/team.c:133
#5  0x00007fffee49caa4 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
#6  0x00007fffee529c3c in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78

Thread 10 (Thread 0x7fff69fe6000 (LWP 18062) "llama-cli"):
#0  futex_wait (addr=0x55555f628314, val=60160) at ../../../src/libgomp/config/linux/x86/futex.h:97
#1  do_wait (addr=<optimized out>, val=60160) at ../../../src/libgomp/config/linux/wait.h:67
#2  gomp_barrier_wait_end (bar=0x55555f628310, state=60160) at ../../../src/libgomp/config/linux/bar.c:48
#3  0x00007ffff7c87779 in gomp_simple_barrier_wait (bar=<optimized out>) at ../../../src/libgomp/config/posix/simple-bar.h:60
#4  gomp_thread_start (xdata=<optimized out>) at ../../../src/libgomp/team.c:133
#5  0x00007fffee49caa4 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
#6  0x00007fffee529c3c in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78

Thread 9 (Thread 0x7fff6a7e7000 (LWP 18061) "llama-cli"):
#0  futex_wait (addr=0x55555f628314, val=60160) at ../../../src/libgomp/config/linux/x86/futex.h:97
--Type <RET> for more, q to quit, c to continue without paging--
#1  do_wait (addr=<optimized out>, val=60160) at ../../../src/libgomp/config/linux/wait.h:67
#2  gomp_barrier_wait_end (bar=0x55555f628310, state=60160) at ../../../src/libgomp/config/linux/bar.c:48
#3  0x00007ffff7c87779 in gomp_simple_barrier_wait (bar=<optimized out>) at ../../../src/libgomp/config/posix/simple-bar.h:60
#4  gomp_thread_start (xdata=<optimized out>) at ../../../src/libgomp/team.c:133
#5  0x00007fffee49caa4 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
#6  0x00007fffee529c3c in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78

Thread 8 (Thread 0x7fff6afe8000 (LWP 18060) "llama-cli"):
#0  futex_wait (addr=0x55555f628314, val=60160) at ../../../src/libgomp/config/linux/x86/futex.h:97
#1  do_wait (addr=<optimized out>, val=60160) at ../../../src/libgomp/config/linux/wait.h:67
#2  gomp_barrier_wait_end (bar=0x55555f628310, state=60160) at ../../../src/libgomp/config/linux/bar.c:48
#3  0x00007ffff7c87779 in gomp_simple_barrier_wait (bar=<optimized out>) at ../../../src/libgomp/config/posix/simple-bar.h:60
#4  gomp_thread_start (xdata=<optimized out>) at ../../../src/libgomp/team.c:133
#5  0x00007fffee49caa4 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
#6  0x00007fffee529c3c in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78

Thread 7 (Thread 0x7fff6b7e9000 (LWP 18059) "llama-cli"):
#0  futex_wait (addr=0x55555f628314, val=60160) at ../../../src/libgomp/config/linux/x86/futex.h:97
#1  do_wait (addr=<optimized out>, val=60160) at ../../../src/libgomp/config/linux/wait.h:67
#2  gomp_barrier_wait_end (bar=0x55555f628310, state=60160) at ../../../src/libgomp/config/linux/bar.c:48
#3  0x00007ffff7c87779 in gomp_simple_barrier_wait (bar=<optimized out>) at ../../../src/libgomp/config/posix/simple-bar.h:60
#4  gomp_thread_start (xdata=<optimized out>) at ../../../src/libgomp/team.c:133
#5  0x00007fffee49caa4 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
#6  0x00007fffee529c3c in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78
--Type <RET> for more, q to quit, c to continue without paging--

Thread 6 (Thread 0x7fffa0afa000 (LWP 18018) "llama-cli"):
#0  0x00007fffee498d71 in __futex_abstimed_wait_common64 (private=32767, cancel=true, abstime=0x7fffa0ad6800, op=393, expected=0, futex_word=0x555555cccca0) at ./nptl/futex-internal.c:57
#1  __futex_abstimed_wait_common (cancel=true, private=32767, abstime=0x7fffa0ad6800, clockid=0, expected=0, futex_word=0x555555cccca0) at ./nptl/futex-internal.c:87
#2  __GI___futex_abstimed_wait_cancelable64 (futex_word=futex_word@entry=0x555555cccca0, expected=expected@entry=0, clockid=clockid@entry=0, abstime=abstime@entry=0x7fffa0ad6800, private=private@entry=0) at ./nptl/futex-internal.c:139
#3  0x00007fffee49bc8e in __pthread_cond_wait_common (abstime=0x7fffa0ad6800, clockid=0, mutex=0x555555cc7d30, cond=0x555555cccc78) at ./nptl/pthread_cond_wait.c:503
#4  ___pthread_cond_timedwait64 (cond=0x555555cccc78, mutex=0x555555cc7d30, abstime=0x7fffa0ad6800) at ./nptl/pthread_cond_wait.c:652
#5  0x00007fffe53cadfa in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#6  0x00007fffe546e143 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#7  0x00007fffee49caa4 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
#8  0x00007fffee529c3c in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78

Thread 5 (Thread 0x7fffa231c000 (LWP 18017) "cuda-EvtHandlr"):
#0  0x00007fffee51b4cd in __GI___poll (fds=0x7fff70000c20, nfds=10, timeout=100) at ../sysdeps/unix/sysv/linux/poll.c:29
#1  0x00007fffe547644f in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#2  0x00007fffe553a80f in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#3  0x00007fffe546e143 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#4  0x00007fffee49caa4 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
#5  0x00007fffee529c3c in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78

Thread 4 (Thread 0x7fffa2d1d000 (LWP 18016) "llama-cli"):
#0  0x00007fffee498d71 in __futex_abstimed_wait_common64 (private=0, cancel=true, abstime=0x7fffa2cf9800, op=393, expected=0--Type <RET> for more, q to quit, c to continue without paging--
, futex_word=0x555555d20600) at ./nptl/futex-internal.c:57
#1  __futex_abstimed_wait_common (cancel=true, private=0, abstime=0x7fffa2cf9800, clockid=0, expected=0, futex_word=0x555555d20600) at ./nptl/futex-internal.c:87
#2  __GI___futex_abstimed_wait_cancelable64 (futex_word=futex_word@entry=0x555555d20600, expected=expected@entry=0, clockid=clockid@entry=0, abstime=abstime@entry=0x7fffa2cf9800, private=private@entry=0) at ./nptl/futex-internal.c:139
#3  0x00007fffee49bc8e in __pthread_cond_wait_common (abstime=0x7fffa2cf9800, clockid=0, mutex=0x555555cd1320, cond=0x555555d205d8) at ./nptl/pthread_cond_wait.c:503
#4  ___pthread_cond_timedwait64 (cond=0x555555d205d8, mutex=0x555555cd1320, abstime=0x7fffa2cf9800) at ./nptl/pthread_cond_wait.c:652
#5  0x00007fffe53cadfa in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#6  0x00007fffe546e143 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#7  0x00007fffee49caa4 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
#8  0x00007fffee529c3c in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78

Thread 3 (Thread 0x7fffa453f000 (LWP 18015) "cuda-EvtHandlr"):
#0  0x00007fffee51b4cd in __GI___poll (fds=0x7fff88000c20, nfds=10, timeout=100) at ../sysdeps/unix/sysv/linux/poll.c:29
#1  0x00007fffe547644f in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#2  0x00007fffe553a80f in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#3  0x00007fffe546e143 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#4  0x00007fffee49caa4 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
#5  0x00007fffee529c3c in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78

Thread 2 (Thread 0x7fffb2dff000 (LWP 18008) "cuda00001400006"):
#0  0x00007fffee51b4cd in __GI___poll (fds=0x555555cd4240, nfds=3, timeout=-1) at ../sysdeps/unix/sysv/linux/poll.c:29
#1  0x00007fffe547644f in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#2  0x00007fffe553a80f in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#3  0x00007fffe546e143 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
--Type <RET> for more, q to quit, c to continue without paging--
#4  0x00007fffee49caa4 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
#5  0x00007fffee529c3c in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78

Thread 1 (Thread 0x7ffff7c4d000 (LWP 18005) "llama-cli"):
#0  0x00007fffe5391024 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#1  0x00007fffe543328a in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#2  0x00007fffe5583eae in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#3  0x00007fffe5585a4c in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#4  0x00007fffe56e29f9 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#5  0x00007fffe5341556 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#6  0x00007fffe5341a70 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#7  0x00007fffe5342407 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#8  0x00007fffe54ebfe9 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#9  0x00007fffee0481a9 in ?? () from /usr/local/cuda-12.8/lib64/libcudart.so.12
#10 0x00007fffee017058 in ?? () from /usr/local/cuda-12.8/lib64/libcudart.so.12
#11 0x00007fffee07693c in cudaMemcpyAsync () from /usr/local/cuda-12.8/lib64/libcudart.so.12
#12 0x00007fffeee271e5 in ggml_backend_cuda_buffer_set_tensor(ggml_backend_buffer*, ggml_tensor*, void const*, unsigned long, unsigned long) () from /home/ciprian/ai/ik_llama.cpp/build/ggml/src/libggml.so
#13 0x00007fffeecc0bfc in ggml_backend_sched_graph_compute_async () from /home/ciprian/ai/ik_llama.cpp/build/ggml/src/libggml.so
#14 0x00007ffff7e8e522 in llama_decode () from /home/ciprian/ai/ik_llama.cpp/build/src/libllama.so
#15 0x0000555555573b55 in main ()

---

👤 **ikawrakow** commented the **2025-05-30** at **06:41:25**:<br>

OK, so we see it being stuck on a call to `cudaMemcpyAsync` copying data from the host to the GPU. No idea why. Or why the transfer rate is just 1.8 GB/s.

---

👤 **ciprianveg** commented the **2025-05-30** at **18:28:15**:<br>

Strange, with deepseek i2k from ubergarm it works perfectly..

---

👤 **ikawrakow** commented the **2025-05-31** at **05:31:40**:<br>

Thanks for the update.

I really don't know what could be causing the pauses and, unlike the illegal memory access bug, nobody else has reported a similar problem.

---

👤 **pt13762104** commented the **2025-05-31** at **11:33:44**:<br>

I also found this problem on my Pc with Qwen3 30B Q4_K_XL. It just stops for a few seconds, then it might be slow or not... unlike llama.cpp.

---

👤 **ciprianveg** commented the **2025-06-01** at **18:03:32**:<br>

Another feedback: i tried the 235b iq3 quant done by @ubergarm and it works fine. Maybe the issue is caused by the unsloth UD XL q3, q4 and q6 quants