## 📌 [Issue #464](https://github.com/ikawrakow/ik_llama.cpp/issues/464) - Bug: The streaming every couple of rows blocks for 5-8s

| **Author** | `ciprianveg` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-27 |
| **Updated** | 2025-07-23 |

---

## 📄 Description

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

## 💬 Conversation

👤 **ikawrakow** commented on **2025-05-28** at **05:17:25**

Not sure. Do you get many tokens at once after the 5-8 seconds pause, or it just did nothing for 5-8 seconds?

---

👤 **ciprianveg** commented on **2025-05-28** at **06:19:29**

It looks like it did nothing,  sometimes a second 5-8 s pause comes after
just 2 words, other times after 2 rows of text. I tried also with 2048
ubatch size and with using amb 512, no difference. For my hardware, what
would be the most suitable build params. I am now setting gpu ggml sched
copies to 1, cublast off and ggml cuda on

On Wed, 28 May 2025, 08:17 Kawrakow, ***@***.***> wrote:

> *ikawrakow* left a comment (ikawrakow/ik_llama.cpp[#464](https://github.com/ikawrakow/ik_llama.cpp/issues/464))
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

👤 **ikawrakow** commented on **2025-05-28** at **07:56:05**

I'm trying to understand the root cause for this strange behavior. Can you reproduce it using `llama-cli` ?

---

👤 **ciprianveg** commented on **2025-05-28** at **10:01:30**

I will try this evening and let you know

---

👤 **ciprianveg** commented on **2025-05-28** at **13:06:09**

Something that maybe can give a clue is that my system is cpu limited, i have 8 channels ddr4 3200 ram but the memory read speed is limited to 85Mb/s instead of the theoretical >200Mb/s because the 16 cores are not enough to read at that speed. This is against the standard cpu systems where memory speed is the limiter, not the cpu..

---

👤 **ciprianveg** commented on **2025-05-28** at **16:52:25**

same issue also with llama-cli

---

👤 **ikawrakow** commented on **2025-05-28** at **17:22:10**

Is there disc activity during the pause? Have you looked at process activity during the pause? Are you running llama.cpp with the exact same parameters (apart from -fmoe)? Is there another memory hungry process running (e.g., another llama.cpp server)?

---

👤 **ciprianveg** commented on **2025-05-28** at **17:27:38**

Llama.cpp runs with exact params except fmoe. I have 256Gb ram and almost
100gb free. No other memory hungry process..

On Wed, 28 May 2025, 20:22 Kawrakow, ***@***.***> wrote:

> *ikawrakow* left a comment (ikawrakow/ik_llama.cpp[#464](https://github.com/ikawrakow/ik_llama.cpp/issues/464))
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

👤 **ikawrakow** commented on **2025-05-29** at **04:19:32**

What about the first two questions? Is the CPU busy during the pauses or just sitting there doing nothing? But at the end it might be easier to just run in the debugger and when it pauses, hit Ctrl-C, type `bt`, and post the backtrace here.

---

👤 **ciprianveg** commented on **2025-05-29** at **05:35:10**

1. Disk activity, no
2. Top shows llama server between 100-500% when it works and same when it pauses

---

👤 **kirnat** commented on **2025-05-29** at **09:12:01**

Check your PCIe traffic with nvtop or similar when the pause happens. Does it happen if you don't offload any experts to the GPUs?

---

👤 **ikawrakow** commented on **2025-05-29** at **09:31:47**

To test the hypothesis that it gets stuck on copying tensors to the GPU, you can run with `-op 26,0,27,0,29,0`. This disables offloading tensors to the GPU for any type of matrix multiplication.

But running in the debugger, interrupting with Ctrl-C when it gets stuck, and sending the backtrace will hopefully also diagnose where (in which function) it hangs for so long.

---

👤 **ciprianveg** commented on **2025-05-29** at **09:44:38**

XXXXXXXXXXXXXXXXXXXXX Setting offload policy for op MUL_MAT to OFF
XXXXXXXXXXXXXXXXXXXXX Setting offload policy for op MUL_MAT_ID to OFF
XXXXXXXXXXXXXXXXXXXXX Setting offload policy for op MOE_FUSED_UP_GATE to OFF
XXXXXXXXXXXXXXXXXXXXXXXXXXXX offload(MUL_MAT) = 0
XXXXXXXXXXXXXXXXXXXXXXXXXXXX offload(MUL_MAT_ID) = 0
XXXXXXXXXXXXXXXXXXXXXXXXXXXX offload(MOE_FUSED_UP_GATE) = 0

same issue

---

👤 **ciprianveg** commented on **2025-05-29** at **09:54:13**

Thread 1 "llama-server" received signal SIGINT, Interrupt.
Download failed: Invalid argument.  Continuing without source file ./nptl/./nptl/pthread_mutex_lock.c.
0x00007fffee4a014c in lll_mutex_lock_optimized (mutex=0x55555899a0d8) at ./nptl/pthread_mutex_lock.c:48
warning: 48	./nptl/pthread_mutex_lock.c: No such file or directory

this is from debug

also, with nvtop, when pause happens, the gpus transfer speed is around 1,8GB/s and as soon as it unblocks drops to 50-100MB/s

---

👤 **ciprianveg** commented on **2025-05-29** at **13:02:59**

it happened also with ngl 0, with nothing sent to gpus, only that being slower, like 2-3tok/s also the pause was longer, cca 20s

llama-server --model /home/ciprian/ai/models/Qwen3-235B-UD_Q4_XL/Qwen3-235B-A22B-UD-Q4_K_XL-00001-of-00003.gguf --alias Qwen3-235B-A22B-UD-Q4_K_XL -fa  -ctk q8_0 -ctv q8_0 -c 36864  --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0 --presence-penalty 0.5 -ngl 0 --threads 16 --host 0.0.0.0 --port 5002    --ubatch-size 4096 --batch-size 4096

---

👤 **ikawrakow** commented on **2025-05-29** at **13:22:24**

If you want to test if the pauses happen when running CPU only, you need to say `CUDA_VISIBLE_DEVICES="" ./bin/llama-server...`. Or just make a build with CUDA disabled.

The debug session above was not useful as the main thread is the server thread, so we don't see where the computation hangs. To get the desired backtrace you need to run `llama-cli`.

> the gpus transfer speed is around 1,8GB/s and as soon as it unblocks drops to 50-100MB/s

Isn't this kind of slow? But even at that rate in 5 seconds it will transfer ~9 GB to the GPU. A `Q5_K` quantized Qwen3-235-A22B layer is in the range of 1.8 GB, so it is transferring 5 layers worth of tensors?

Or is this all happening when your context gets full?

---

👤 **ciprianveg** commented on **2025-05-29** at **13:59:44**

debug on llama-cli ctrl+c when paused i don't think is helpful:
Thread 1 "llama-cli" received signal SIGINT, Interrupt.
0x00007fffe5391028 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
(gdb)

---

👤 **ikawrakow** commented on **2025-05-29** at **15:42:57**

I guess you need
```
thread apply all bt
```

---

👤 **ciprianveg** commented on **2025-05-29** at **17:40:41**

Hi @ikawrakow:
0x00007fffe5391024 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
(gdb) thread apply all bt

Thread 21 (Thread 0x7fff647db000 (LWP 18073) "llama-cli"):
[#0](https://github.com/ikawrakow/ik_llama.cpp/issues/0)  futex_wait (addr=0x55555f628314, val=60160) at ../../../src/libgomp/config/linux/x86/futex.h:97
[#1](https://github.com/ikawrakow/ik_llama.cpp/issues/1)  do_wait (addr=<optimized out>, val=60160) at ../../../src/libgomp/config/linux/wait.h:67
[#2](https://github.com/ikawrakow/ik_llama.cpp/issues/2)  gomp_barrier_wait_end (bar=0x55555f628310, state=60160) at ../../../src/libgomp/config/linux/bar.c:48
[#3](https://github.com/ikawrakow/ik_llama.cpp/issues/3)  0x00007ffff7c87779 in gomp_simple_barrier_wait (bar=<optimized out>) at ../../../src/libgomp/config/posix/simple-bar.h:60
[#4](https://github.com/ikawrakow/ik_llama.cpp/issues/4)  gomp_thread_start (xdata=<optimized out>) at ../../../src/libgomp/team.c:133
[#5](https://github.com/ikawrakow/ik_llama.cpp/issues/5)  0x00007fffee49caa4 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
[#6](https://github.com/ikawrakow/ik_llama.cpp/issues/6)  0x00007fffee529c3c in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78

Thread 20 (Thread 0x7fff64fdc000 (LWP 18072) "llama-cli"):
[#0](https://github.com/ikawrakow/ik_llama.cpp/issues/0)  futex_wait (addr=0x55555f628314, val=60160) at ../../../src/libgomp/config/linux/x86/futex.h:97
[#1](https://github.com/ikawrakow/ik_llama.cpp/issues/1)  do_wait (addr=<optimized out>, val=60160) at ../../../src/libgomp/config/linux/wait.h:67
[#2](https://github.com/ikawrakow/ik_llama.cpp/issues/2)  gomp_barrier_wait_end (bar=0x55555f628310, state=60160) at ../../../src/libgomp/config/linux/bar.c:48
[#3](https://github.com/ikawrakow/ik_llama.cpp/issues/3)  0x00007ffff7c87779 in gomp_simple_barrier_wait (bar=<optimized out>) at ../../../src/libgomp/config/posix/simple-bar.h:60
[#4](https://github.com/ikawrakow/ik_llama.cpp/issues/4)  gomp_thread_start (xdata=<optimized out>) at ../../../src/libgomp/team.c:133
[#5](https://github.com/ikawrakow/ik_llama.cpp/issues/5)  0x00007fffee49caa4 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
[#6](https://github.com/ikawrakow/ik_llama.cpp/issues/6)  0x00007fffee529c3c in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78

Thread 19 (Thread 0x7fff657dd000 (LWP 18071) "llama-cli"):
[#0](https://github.com/ikawrakow/ik_llama.cpp/issues/0)  futex_wait (addr=0x55555f628314, val=60160) at ../../../src/libgomp/config/linux/x86/futex.h:97
--Type <RET> for more, q to quit, c to continue without paging--
[#1](https://github.com/ikawrakow/ik_llama.cpp/issues/1)  do_wait (addr=<optimized out>, val=60160) at ../../../src/libgomp/config/linux/wait.h:67
[#2](https://github.com/ikawrakow/ik_llama.cpp/issues/2)  gomp_barrier_wait_end (bar=0x55555f628310, state=60160) at ../../../src/libgomp/config/linux/bar.c:48
[#3](https://github.com/ikawrakow/ik_llama.cpp/issues/3)  0x00007ffff7c87779 in gomp_simple_barrier_wait (bar=<optimized out>) at ../../../src/libgomp/config/posix/simple-bar.h:60
[#4](https://github.com/ikawrakow/ik_llama.cpp/issues/4)  gomp_thread_start (xdata=<optimized out>) at ../../../src/libgomp/team.c:133
[#5](https://github.com/ikawrakow/ik_llama.cpp/issues/5)  0x00007fffee49caa4 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
[#6](https://github.com/ikawrakow/ik_llama.cpp/issues/6)  0x00007fffee529c3c in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78

Thread 18 (Thread 0x7fff65fde000 (LWP 18070) "llama-cli"):
[#0](https://github.com/ikawrakow/ik_llama.cpp/issues/0)  futex_wait (addr=0x55555f628314, val=60160) at ../../../src/libgomp/config/linux/x86/futex.h:97
[#1](https://github.com/ikawrakow/ik_llama.cpp/issues/1)  do_wait (addr=<optimized out>, val=60160) at ../../../src/libgomp/config/linux/wait.h:67
[#2](https://github.com/ikawrakow/ik_llama.cpp/issues/2)  gomp_barrier_wait_end (bar=0x55555f628310, state=60160) at ../../../src/libgomp/config/linux/bar.c:48
[#3](https://github.com/ikawrakow/ik_llama.cpp/issues/3)  0x00007ffff7c87779 in gomp_simple_barrier_wait (bar=<optimized out>) at ../../../src/libgomp/config/posix/simple-bar.h:60
[#4](https://github.com/ikawrakow/ik_llama.cpp/issues/4)  gomp_thread_start (xdata=<optimized out>) at ../../../src/libgomp/team.c:133
[#5](https://github.com/ikawrakow/ik_llama.cpp/issues/5)  0x00007fffee49caa4 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
[#6](https://github.com/ikawrakow/ik_llama.cpp/issues/6)  0x00007fffee529c3c in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78

Thread 17 (Thread 0x7fff667df000 (LWP 18069) "llama-cli"):
[#0](https://github.com/ikawrakow/ik_llama.cpp/issues/0)  futex_wait (addr=0x55555f628314, val=60160) at ../../../src/libgomp/config/linux/x86/futex.h:97
[#1](https://github.com/ikawrakow/ik_llama.cpp/issues/1)  do_wait (addr=<optimized out>, val=60160) at ../../../src/libgomp/config/linux/wait.h:67
[#2](https://github.com/ikawrakow/ik_llama.cpp/issues/2)  gomp_barrier_wait_end (bar=0x55555f628310, state=60160) at ../../../src/libgomp/config/linux/bar.c:48
[#3](https://github.com/ikawrakow/ik_llama.cpp/issues/3)  0x00007ffff7c87779 in gomp_simple_barrier_wait (bar=<optimized out>) at ../../../src/libgomp/config/posix/simple-bar.h:6--Type <RET> for more, q to quit, c to continue without paging--
0
[#4](https://github.com/ikawrakow/ik_llama.cpp/issues/4)  gomp_thread_start (xdata=<optimized out>) at ../../../src/libgomp/team.c:133
[#5](https://github.com/ikawrakow/ik_llama.cpp/issues/5)  0x00007fffee49caa4 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
[#6](https://github.com/ikawrakow/ik_llama.cpp/issues/6)  0x00007fffee529c3c in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78

Thread 16 (Thread 0x7fff66fe0000 (LWP 18068) "llama-cli"):
[#0](https://github.com/ikawrakow/ik_llama.cpp/issues/0)  futex_wait (addr=0x55555f628314, val=60160) at ../../../src/libgomp/config/linux/x86/futex.h:97
[#1](https://github.com/ikawrakow/ik_llama.cpp/issues/1)  do_wait (addr=<optimized out>, val=60160) at ../../../src/libgomp/config/linux/wait.h:67
[#2](https://github.com/ikawrakow/ik_llama.cpp/issues/2)  gomp_barrier_wait_end (bar=0x55555f628310, state=60160) at ../../../src/libgomp/config/linux/bar.c:48
[#3](https://github.com/ikawrakow/ik_llama.cpp/issues/3)  0x00007ffff7c87779 in gomp_simple_barrier_wait (bar=<optimized out>) at ../../../src/libgomp/config/posix/simple-bar.h:60
[#4](https://github.com/ikawrakow/ik_llama.cpp/issues/4)  gomp_thread_start (xdata=<optimized out>) at ../../../src/libgomp/team.c:133
[#5](https://github.com/ikawrakow/ik_llama.cpp/issues/5)  0x00007fffee49caa4 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
[#6](https://github.com/ikawrakow/ik_llama.cpp/issues/6)  0x00007fffee529c3c in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78

Thread 15 (Thread 0x7fff677e1000 (LWP 18067) "llama-cli"):
[#0](https://github.com/ikawrakow/ik_llama.cpp/issues/0)  futex_wait (addr=0x55555f628314, val=60160) at ../../../src/libgomp/config/linux/x86/futex.h:97
[#1](https://github.com/ikawrakow/ik_llama.cpp/issues/1)  do_wait (addr=<optimized out>, val=60160) at ../../../src/libgomp/config/linux/wait.h:67
[#2](https://github.com/ikawrakow/ik_llama.cpp/issues/2)  gomp_barrier_wait_end (bar=0x55555f628310, state=60160) at ../../../src/libgomp/config/linux/bar.c:48
[#3](https://github.com/ikawrakow/ik_llama.cpp/issues/3)  0x00007ffff7c87779 in gomp_simple_barrier_wait (bar=<optimized out>) at ../../../src/libgomp/config/posix/simple-bar.h:60
[#4](https://github.com/ikawrakow/ik_llama.cpp/issues/4)  gomp_thread_start (xdata=<optimized out>) at ../../../src/libgomp/team.c:133
[#5](https://github.com/ikawrakow/ik_llama.cpp/issues/5)  0x00007fffee49caa4 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
--Type <RET> for more, q to quit, c to continue without paging--
[#6](https://github.com/ikawrakow/ik_llama.cpp/issues/6)  0x00007fffee529c3c in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78

Thread 14 (Thread 0x7fff67fe2000 (LWP 18066) "llama-cli"):
[#0](https://github.com/ikawrakow/ik_llama.cpp/issues/0)  futex_wait (addr=0x55555f628314, val=60160) at ../../../src/libgomp/config/linux/x86/futex.h:97
[#1](https://github.com/ikawrakow/ik_llama.cpp/issues/1)  do_wait (addr=<optimized out>, val=60160) at ../../../src/libgomp/config/linux/wait.h:67
[#2](https://github.com/ikawrakow/ik_llama.cpp/issues/2)  gomp_barrier_wait_end (bar=0x55555f628310, state=60160) at ../../../src/libgomp/config/linux/bar.c:48
[#3](https://github.com/ikawrakow/ik_llama.cpp/issues/3)  0x00007ffff7c87779 in gomp_simple_barrier_wait (bar=<optimized out>) at ../../../src/libgomp/config/posix/simple-bar.h:60
[#4](https://github.com/ikawrakow/ik_llama.cpp/issues/4)  gomp_thread_start (xdata=<optimized out>) at ../../../src/libgomp/team.c:133
[#5](https://github.com/ikawrakow/ik_llama.cpp/issues/5)  0x00007fffee49caa4 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
[#6](https://github.com/ikawrakow/ik_llama.cpp/issues/6)  0x00007fffee529c3c in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78

Thread 13 (Thread 0x7fff687e3000 (LWP 18065) "llama-cli"):
[#0](https://github.com/ikawrakow/ik_llama.cpp/issues/0)  futex_wait (addr=0x55555f628314, val=60160) at ../../../src/libgomp/config/linux/x86/futex.h:97
[#1](https://github.com/ikawrakow/ik_llama.cpp/issues/1)  do_wait (addr=<optimized out>, val=60160) at ../../../src/libgomp/config/linux/wait.h:67
[#2](https://github.com/ikawrakow/ik_llama.cpp/issues/2)  gomp_barrier_wait_end (bar=0x55555f628310, state=60160) at ../../../src/libgomp/config/linux/bar.c:48
[#3](https://github.com/ikawrakow/ik_llama.cpp/issues/3)  0x00007ffff7c87779 in gomp_simple_barrier_wait (bar=<optimized out>) at ../../../src/libgomp/config/posix/simple-bar.h:60
[#4](https://github.com/ikawrakow/ik_llama.cpp/issues/4)  gomp_thread_start (xdata=<optimized out>) at ../../../src/libgomp/team.c:133
[#5](https://github.com/ikawrakow/ik_llama.cpp/issues/5)  0x00007fffee49caa4 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
[#6](https://github.com/ikawrakow/ik_llama.cpp/issues/6)  0x00007fffee529c3c in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78

Thread 12 (Thread 0x7fff68fe4000 (LWP 18064) "llama-cli"):
[#0](https://github.com/ikawrakow/ik_llama.cpp/issues/0)  futex_wait (addr=0x55555f628314, val=60160) at ../../../src/libgomp/config/linux/x86/futex.h:97
[#1](https://github.com/ikawrakow/ik_llama.cpp/issues/1)  do_wait (addr=<optimized out>, val=60160) at ../../../src/libgomp/config/linux/wait.h:67
[#2](https://github.com/ikawrakow/ik_llama.cpp/issues/2)  gomp_barrier_wait_end (bar=0x55555f628310, state=60160) at ../../../src/libgomp/config/linux/bar.c:48
[#3](https://github.com/ikawrakow/ik_llama.cpp/issues/3)  0x00007ffff7c87779 in gomp_simple_barrier_wait (bar=<optimized out>) at ../../../src/libgomp/config/posix/simple-bar.h:6--Type <RET> for more, q to quit, c to continue without paging--
0
[#4](https://github.com/ikawrakow/ik_llama.cpp/issues/4)  gomp_thread_start (xdata=<optimized out>) at ../../../src/libgomp/team.c:133
[#5](https://github.com/ikawrakow/ik_llama.cpp/issues/5)  0x00007fffee49caa4 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
[#6](https://github.com/ikawrakow/ik_llama.cpp/issues/6)  0x00007fffee529c3c in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78

Thread 11 (Thread 0x7fff697e5000 (LWP 18063) "llama-cli"):
[#0](https://github.com/ikawrakow/ik_llama.cpp/issues/0)  futex_wait (addr=0x55555f628314, val=60160) at ../../../src/libgomp/config/linux/x86/futex.h:97
[#1](https://github.com/ikawrakow/ik_llama.cpp/issues/1)  do_wait (addr=<optimized out>, val=60160) at ../../../src/libgomp/config/linux/wait.h:67
[#2](https://github.com/ikawrakow/ik_llama.cpp/issues/2)  gomp_barrier_wait_end (bar=0x55555f628310, state=60160) at ../../../src/libgomp/config/linux/bar.c:48
[#3](https://github.com/ikawrakow/ik_llama.cpp/issues/3)  0x00007ffff7c87779 in gomp_simple_barrier_wait (bar=<optimized out>) at ../../../src/libgomp/config/posix/simple-bar.h:60
[#4](https://github.com/ikawrakow/ik_llama.cpp/issues/4)  gomp_thread_start (xdata=<optimized out>) at ../../../src/libgomp/team.c:133
[#5](https://github.com/ikawrakow/ik_llama.cpp/issues/5)  0x00007fffee49caa4 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
[#6](https://github.com/ikawrakow/ik_llama.cpp/issues/6)  0x00007fffee529c3c in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78

Thread 10 (Thread 0x7fff69fe6000 (LWP 18062) "llama-cli"):
[#0](https://github.com/ikawrakow/ik_llama.cpp/issues/0)  futex_wait (addr=0x55555f628314, val=60160) at ../../../src/libgomp/config/linux/x86/futex.h:97
[#1](https://github.com/ikawrakow/ik_llama.cpp/issues/1)  do_wait (addr=<optimized out>, val=60160) at ../../../src/libgomp/config/linux/wait.h:67
[#2](https://github.com/ikawrakow/ik_llama.cpp/issues/2)  gomp_barrier_wait_end (bar=0x55555f628310, state=60160) at ../../../src/libgomp/config/linux/bar.c:48
[#3](https://github.com/ikawrakow/ik_llama.cpp/issues/3)  0x00007ffff7c87779 in gomp_simple_barrier_wait (bar=<optimized out>) at ../../../src/libgomp/config/posix/simple-bar.h:60
[#4](https://github.com/ikawrakow/ik_llama.cpp/issues/4)  gomp_thread_start (xdata=<optimized out>) at ../../../src/libgomp/team.c:133
[#5](https://github.com/ikawrakow/ik_llama.cpp/issues/5)  0x00007fffee49caa4 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
[#6](https://github.com/ikawrakow/ik_llama.cpp/issues/6)  0x00007fffee529c3c in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78

Thread 9 (Thread 0x7fff6a7e7000 (LWP 18061) "llama-cli"):
[#0](https://github.com/ikawrakow/ik_llama.cpp/issues/0)  futex_wait (addr=0x55555f628314, val=60160) at ../../../src/libgomp/config/linux/x86/futex.h:97
--Type <RET> for more, q to quit, c to continue without paging--
[#1](https://github.com/ikawrakow/ik_llama.cpp/issues/1)  do_wait (addr=<optimized out>, val=60160) at ../../../src/libgomp/config/linux/wait.h:67
[#2](https://github.com/ikawrakow/ik_llama.cpp/issues/2)  gomp_barrier_wait_end (bar=0x55555f628310, state=60160) at ../../../src/libgomp/config/linux/bar.c:48
[#3](https://github.com/ikawrakow/ik_llama.cpp/issues/3)  0x00007ffff7c87779 in gomp_simple_barrier_wait (bar=<optimized out>) at ../../../src/libgomp/config/posix/simple-bar.h:60
[#4](https://github.com/ikawrakow/ik_llama.cpp/issues/4)  gomp_thread_start (xdata=<optimized out>) at ../../../src/libgomp/team.c:133
[#5](https://github.com/ikawrakow/ik_llama.cpp/issues/5)  0x00007fffee49caa4 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
[#6](https://github.com/ikawrakow/ik_llama.cpp/issues/6)  0x00007fffee529c3c in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78

Thread 8 (Thread 0x7fff6afe8000 (LWP 18060) "llama-cli"):
[#0](https://github.com/ikawrakow/ik_llama.cpp/issues/0)  futex_wait (addr=0x55555f628314, val=60160) at ../../../src/libgomp/config/linux/x86/futex.h:97
[#1](https://github.com/ikawrakow/ik_llama.cpp/issues/1)  do_wait (addr=<optimized out>, val=60160) at ../../../src/libgomp/config/linux/wait.h:67
[#2](https://github.com/ikawrakow/ik_llama.cpp/issues/2)  gomp_barrier_wait_end (bar=0x55555f628310, state=60160) at ../../../src/libgomp/config/linux/bar.c:48
[#3](https://github.com/ikawrakow/ik_llama.cpp/issues/3)  0x00007ffff7c87779 in gomp_simple_barrier_wait (bar=<optimized out>) at ../../../src/libgomp/config/posix/simple-bar.h:60
[#4](https://github.com/ikawrakow/ik_llama.cpp/issues/4)  gomp_thread_start (xdata=<optimized out>) at ../../../src/libgomp/team.c:133
[#5](https://github.com/ikawrakow/ik_llama.cpp/issues/5)  0x00007fffee49caa4 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
[#6](https://github.com/ikawrakow/ik_llama.cpp/issues/6)  0x00007fffee529c3c in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78

Thread 7 (Thread 0x7fff6b7e9000 (LWP 18059) "llama-cli"):
[#0](https://github.com/ikawrakow/ik_llama.cpp/issues/0)  futex_wait (addr=0x55555f628314, val=60160) at ../../../src/libgomp/config/linux/x86/futex.h:97
[#1](https://github.com/ikawrakow/ik_llama.cpp/issues/1)  do_wait (addr=<optimized out>, val=60160) at ../../../src/libgomp/config/linux/wait.h:67
[#2](https://github.com/ikawrakow/ik_llama.cpp/issues/2)  gomp_barrier_wait_end (bar=0x55555f628310, state=60160) at ../../../src/libgomp/config/linux/bar.c:48
[#3](https://github.com/ikawrakow/ik_llama.cpp/issues/3)  0x00007ffff7c87779 in gomp_simple_barrier_wait (bar=<optimized out>) at ../../../src/libgomp/config/posix/simple-bar.h:60
[#4](https://github.com/ikawrakow/ik_llama.cpp/issues/4)  gomp_thread_start (xdata=<optimized out>) at ../../../src/libgomp/team.c:133
[#5](https://github.com/ikawrakow/ik_llama.cpp/issues/5)  0x00007fffee49caa4 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
[#6](https://github.com/ikawrakow/ik_llama.cpp/issues/6)  0x00007fffee529c3c in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78
--Type <RET> for more, q to quit, c to continue without paging--

Thread 6 (Thread 0x7fffa0afa000 (LWP 18018) "llama-cli"):
[#0](https://github.com/ikawrakow/ik_llama.cpp/issues/0)  0x00007fffee498d71 in __futex_abstimed_wait_common64 (private=32767, cancel=true, abstime=0x7fffa0ad6800, op=393, expected=0, futex_word=0x555555cccca0) at ./nptl/futex-internal.c:57
[#1](https://github.com/ikawrakow/ik_llama.cpp/issues/1)  __futex_abstimed_wait_common (cancel=true, private=32767, abstime=0x7fffa0ad6800, clockid=0, expected=0, futex_word=0x555555cccca0) at ./nptl/futex-internal.c:87
[#2](https://github.com/ikawrakow/ik_llama.cpp/issues/2)  __GI___futex_abstimed_wait_cancelable64 (futex_word=futex_word@entry=0x555555cccca0, expected=expected@entry=0, clockid=clockid@entry=0, abstime=abstime@entry=0x7fffa0ad6800, private=private@entry=0) at ./nptl/futex-internal.c:139
[#3](https://github.com/ikawrakow/ik_llama.cpp/issues/3)  0x00007fffee49bc8e in __pthread_cond_wait_common (abstime=0x7fffa0ad6800, clockid=0, mutex=0x555555cc7d30, cond=0x555555cccc78) at ./nptl/pthread_cond_wait.c:503
[#4](https://github.com/ikawrakow/ik_llama.cpp/issues/4)  ___pthread_cond_timedwait64 (cond=0x555555cccc78, mutex=0x555555cc7d30, abstime=0x7fffa0ad6800) at ./nptl/pthread_cond_wait.c:652
[#5](https://github.com/ikawrakow/ik_llama.cpp/issues/5)  0x00007fffe53cadfa in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
[#6](https://github.com/ikawrakow/ik_llama.cpp/issues/6)  0x00007fffe546e143 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
[#7](https://github.com/ikawrakow/ik_llama.cpp/issues/7)  0x00007fffee49caa4 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
[#8](https://github.com/ikawrakow/ik_llama.cpp/issues/8)  0x00007fffee529c3c in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78

Thread 5 (Thread 0x7fffa231c000 (LWP 18017) "cuda-EvtHandlr"):
[#0](https://github.com/ikawrakow/ik_llama.cpp/issues/0)  0x00007fffee51b4cd in __GI___poll (fds=0x7fff70000c20, nfds=10, timeout=100) at ../sysdeps/unix/sysv/linux/poll.c:29
[#1](https://github.com/ikawrakow/ik_llama.cpp/issues/1)  0x00007fffe547644f in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
[#2](https://github.com/ikawrakow/ik_llama.cpp/issues/2)  0x00007fffe553a80f in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
[#3](https://github.com/ikawrakow/ik_llama.cpp/issues/3)  0x00007fffe546e143 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
[#4](https://github.com/ikawrakow/ik_llama.cpp/issues/4)  0x00007fffee49caa4 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
[#5](https://github.com/ikawrakow/ik_llama.cpp/issues/5)  0x00007fffee529c3c in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78

Thread 4 (Thread 0x7fffa2d1d000 (LWP 18016) "llama-cli"):
[#0](https://github.com/ikawrakow/ik_llama.cpp/issues/0)  0x00007fffee498d71 in __futex_abstimed_wait_common64 (private=0, cancel=true, abstime=0x7fffa2cf9800, op=393, expected=0--Type <RET> for more, q to quit, c to continue without paging--
, futex_word=0x555555d20600) at ./nptl/futex-internal.c:57
[#1](https://github.com/ikawrakow/ik_llama.cpp/issues/1)  __futex_abstimed_wait_common (cancel=true, private=0, abstime=0x7fffa2cf9800, clockid=0, expected=0, futex_word=0x555555d20600) at ./nptl/futex-internal.c:87
[#2](https://github.com/ikawrakow/ik_llama.cpp/issues/2)  __GI___futex_abstimed_wait_cancelable64 (futex_word=futex_word@entry=0x555555d20600, expected=expected@entry=0, clockid=clockid@entry=0, abstime=abstime@entry=0x7fffa2cf9800, private=private@entry=0) at ./nptl/futex-internal.c:139
[#3](https://github.com/ikawrakow/ik_llama.cpp/issues/3)  0x00007fffee49bc8e in __pthread_cond_wait_common (abstime=0x7fffa2cf9800, clockid=0, mutex=0x555555cd1320, cond=0x555555d205d8) at ./nptl/pthread_cond_wait.c:503
[#4](https://github.com/ikawrakow/ik_llama.cpp/issues/4)  ___pthread_cond_timedwait64 (cond=0x555555d205d8, mutex=0x555555cd1320, abstime=0x7fffa2cf9800) at ./nptl/pthread_cond_wait.c:652
[#5](https://github.com/ikawrakow/ik_llama.cpp/issues/5)  0x00007fffe53cadfa in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
[#6](https://github.com/ikawrakow/ik_llama.cpp/issues/6)  0x00007fffe546e143 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
[#7](https://github.com/ikawrakow/ik_llama.cpp/issues/7)  0x00007fffee49caa4 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
[#8](https://github.com/ikawrakow/ik_llama.cpp/issues/8)  0x00007fffee529c3c in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78

Thread 3 (Thread 0x7fffa453f000 (LWP 18015) "cuda-EvtHandlr"):
[#0](https://github.com/ikawrakow/ik_llama.cpp/issues/0)  0x00007fffee51b4cd in __GI___poll (fds=0x7fff88000c20, nfds=10, timeout=100) at ../sysdeps/unix/sysv/linux/poll.c:29
[#1](https://github.com/ikawrakow/ik_llama.cpp/issues/1)  0x00007fffe547644f in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
[#2](https://github.com/ikawrakow/ik_llama.cpp/issues/2)  0x00007fffe553a80f in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
[#3](https://github.com/ikawrakow/ik_llama.cpp/issues/3)  0x00007fffe546e143 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
[#4](https://github.com/ikawrakow/ik_llama.cpp/issues/4)  0x00007fffee49caa4 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
[#5](https://github.com/ikawrakow/ik_llama.cpp/issues/5)  0x00007fffee529c3c in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78

Thread 2 (Thread 0x7fffb2dff000 (LWP 18008) "cuda00001400006"):
[#0](https://github.com/ikawrakow/ik_llama.cpp/issues/0)  0x00007fffee51b4cd in __GI___poll (fds=0x555555cd4240, nfds=3, timeout=-1) at ../sysdeps/unix/sysv/linux/poll.c:29
[#1](https://github.com/ikawrakow/ik_llama.cpp/issues/1)  0x00007fffe547644f in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
[#2](https://github.com/ikawrakow/ik_llama.cpp/issues/2)  0x00007fffe553a80f in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
[#3](https://github.com/ikawrakow/ik_llama.cpp/issues/3)  0x00007fffe546e143 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
--Type <RET> for more, q to quit, c to continue without paging--
[#4](https://github.com/ikawrakow/ik_llama.cpp/issues/4)  0x00007fffee49caa4 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
[#5](https://github.com/ikawrakow/ik_llama.cpp/issues/5)  0x00007fffee529c3c in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78

Thread 1 (Thread 0x7ffff7c4d000 (LWP 18005) "llama-cli"):
[#0](https://github.com/ikawrakow/ik_llama.cpp/issues/0)  0x00007fffe5391024 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
[#1](https://github.com/ikawrakow/ik_llama.cpp/issues/1)  0x00007fffe543328a in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
[#2](https://github.com/ikawrakow/ik_llama.cpp/issues/2)  0x00007fffe5583eae in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
[#3](https://github.com/ikawrakow/ik_llama.cpp/issues/3)  0x00007fffe5585a4c in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
[#4](https://github.com/ikawrakow/ik_llama.cpp/issues/4)  0x00007fffe56e29f9 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
[#5](https://github.com/ikawrakow/ik_llama.cpp/issues/5)  0x00007fffe5341556 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
[#6](https://github.com/ikawrakow/ik_llama.cpp/issues/6)  0x00007fffe5341a70 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
[#7](https://github.com/ikawrakow/ik_llama.cpp/issues/7)  0x00007fffe5342407 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
[#8](https://github.com/ikawrakow/ik_llama.cpp/issues/8)  0x00007fffe54ebfe9 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
[#9](https://github.com/ikawrakow/ik_llama.cpp/issues/9)  0x00007fffee0481a9 in ?? () from /usr/local/cuda-12.8/lib64/libcudart.so.12
[#10](https://github.com/ikawrakow/ik_llama.cpp/issues/10) 0x00007fffee017058 in ?? () from /usr/local/cuda-12.8/lib64/libcudart.so.12
[#11](https://github.com/ikawrakow/ik_llama.cpp/issues/11) 0x00007fffee07693c in cudaMemcpyAsync () from /usr/local/cuda-12.8/lib64/libcudart.so.12
[#12](https://github.com/ikawrakow/ik_llama.cpp/issues/12) 0x00007fffeee271e5 in ggml_backend_cuda_buffer_set_tensor(ggml_backend_buffer*, ggml_tensor*, void const*, unsigned long, unsigned long) () from /home/ciprian/ai/ik_llama.cpp/build/ggml/src/libggml.so
[#13](https://github.com/ikawrakow/ik_llama.cpp/issues/13) 0x00007fffeecc0bfc in ggml_backend_sched_graph_compute_async () from /home/ciprian/ai/ik_llama.cpp/build/ggml/src/libggml.so
[#14](https://github.com/ikawrakow/ik_llama.cpp/issues/14) 0x00007ffff7e8e522 in llama_decode () from /home/ciprian/ai/ik_llama.cpp/build/src/libllama.so
[#15](https://github.com/ikawrakow/ik_llama.cpp/issues/15) 0x0000555555573b55 in main ()

---

👤 **ikawrakow** commented on **2025-05-30** at **06:41:25**

OK, so we see it being stuck on a call to `cudaMemcpyAsync` copying data from the host to the GPU. No idea why. Or why the transfer rate is just 1.8 GB/s.

---

👤 **ciprianveg** commented on **2025-05-30** at **18:28:15**

Strange, with deepseek i2k from ubergarm it works perfectly..

---

👤 **ikawrakow** commented on **2025-05-31** at **05:31:40**

Thanks for the update.

I really don't know what could be causing the pauses and, unlike the illegal memory access bug, nobody else has reported a similar problem.

---

👤 **pt13762104** commented on **2025-05-31** at **11:33:44**

I also found this problem on my Pc with Qwen3 30B Q4_K_XL. It just stops for a few seconds, then it might be slow or not... unlike llama.cpp.

---

👤 **ciprianveg** commented on **2025-06-01** at **18:03:32**

Another feedback: i tried the 235b iq3 quant done by @ubergarm and it works fine. Maybe the issue is caused by the unsloth UD XL q3, q4 and q6 quants

---

👤 **samteezy** commented on **2025-07-22** at **15:40:24**

I have some context to add here that may help, idk. I'm running a very recent build of `ik_llama` from maybe a week ago if that.

I've been running `Qwen3-30B-A3B-128K-UD-Q5_K_XL.gguf` with `ik_llama` using Vulkan and experiencing lower performance than regular `llama.cpp`, but it's mainly been due to stuttering/pausing during text generation. I noticed that [this pull](https://github.com/ikawrakow/ik_llama.cpp/pull/573) mentions some odd behavior around commas, and what I'm experiencing seems similar, though my delays are more like around a second or less rather than the 5-8 sec mentioned here, I assume because I'm GPU-accelerated. (I have the experts offloaded to CPU).

So today I tried running all three with the same prompt:

`Qwen3-30B-A3B-128K-UD-Q5_K_XL.gguf` from unsloth - dynamic quant
`Qwen3-30B-A3B-128K-Q4_K_M.gguf` also from unsloth - not a dynamic quant
`Qwen3-30B-A3B-Q4_K_M.gguf` directly from Qwen

With these settings (all are run via `llama-swap`):

```
cmd: |
      /root/llama-builds/ik_llama.cpp/bin/llama-server
      --port ${PORT}
      --flash-attn
      -ctk q8_0 -ctv q8_0
      --threads 17
      --n-gpu-layers 0 -sm none --main-gpu 1
      -m /mnt/models/unsloth/Qwen3-30B-A3B-128K-UD-Q5_K_XL.gguf
      -fmoe
      -ot ".*ffn_.*_exps\.weight=CPU"
      --temp 0.7
      --min-p 0
      --top-p 0.8
      --top-k 20
      --ctx-size 128000
      --presence-penalty 0.1
```

The Qwen official GGUF runs without stutter or issue with near identical performance to `llama.cpp`, but both the dynamic and "regular" quants from unsloth experience the pausing issue. So I don't think it's related to the dynamic quants, but something else with unsloth's edits/fixes.

---

👤 **ciprianveg** commented on **2025-07-22** at **17:24:19**

I can confirm it is comma related.

---

👤 **saood06** commented on **2025-07-22** at **18:42:24**

> I can confirm it is comma related.

I collected as much info as I could about this bug [here](https://github.com/ikawrakow/ik_llama.cpp/pull/573#issuecomment-3033895399).

---

👤 **ciprianveg** commented on **2025-07-22** at **21:08:18**

I tried adding to my llama-server command --override-kv tokenizer.ggml.bos_token_id=int:-1, but it still pauses every comma..

---

👤 **saood06** commented on **2025-07-22** at **21:14:57**

> I tried adding to my llama-server command --override-kv tokenizer.ggml.bos_token_id=int:-1, but it still pauses every comma..

Which is to be expected. The only reason that was relevant was with dots the BOS token upon being loaded was a comma which caused it to present the underlying issue. 

Changing that didn't fix the issue with the comma, it just prevented an incorrect situation that just happened to intersect with the comma bug.

---

👤 **ubergarm** commented on **2025-07-23** at **04:43:24**

Hrmm, I possibly just observed this "brief pause after generating a `,` character" in very early test of https://huggingface.co/ubergarm/Qwen3-Coder-480B-A35B-Instruct-GGUF/ on the IQ2_KS. Not sure, and I'm too sleepy to look into it. Just dropping a note to my future self to look more into it. 

I also added a note there on the discussing https://huggingface.co/ubergarm/Qwen3-Coder-480B-A35B-Instruct-GGUF/discussions/1#688067cdf4ffc5de61c3f86a linking back to here suggesting trying that tokenizer trick which probably won't work and has nothing to do with it lol.

Goodnight and I hope no new MoEs drop while I'm asleep. 💀 😹

---

👤 **saood06** commented on **2025-07-23** at **05:53:09**

>I also added a note there on the discussing https://huggingface.co/ubergarm/Qwen3-Coder-480B-A35B-Instruct-GGUF/discussions/1#688067cdf4ffc5de61c3f86a linking back to here suggesting trying that tokenizer trick which probably won't work and has nothing to do with it lol.

I replied above that it doesn't. The model linked has `tokenizer.ggml.add_bos_token = false`. Turning the BOS token off in two ways (`-1` and `false`), will not resolve the comma bug.

---

👤 **ikawrakow** commented on **2025-07-23** at **06:18:22**

I can confirm the issue with the Qwen3-30B-A3B `Q4_K_M` model from Unsloth (although the pauses are much shorter, more like a fraction of a second, but still noticable at the 50 t/s speed I have). It does not happen if I quantize the model myself from the official Qwen3-30B-A3B `bf16` model using the exact same `Q4_K_M` recipe. 

As the pauses are too short to be able to reliably interrupt at exactly that point in a debugger, I decided to see if I can make it easier to break at a pause. So, I asked the model to write 100 commas in a row. The pauses occur while it is thinking, but when it starts writing the commas there are no pauses. Then I though may be it is comma followed by space, so I asked it to write comma followed by space 100 times. Same thing - pauses while thinking, no pauses while writing `, , , ...`.

I'll try to debug today, let's see how it goes.

---

👤 **ikawrakow** commented on **2025-07-23** at **06:21:28**

@samteezy  The Vulkan backend does not have an implementation for the `-fmoe` command line argument. The result is that the fused `ffn_up+ffn_gate` op that is enabled by `-fmoe` will be run on the CPU. I'm also wondering why you use `-ngl 0`. My guess is that `-ngl 100 -ot exps=CPU` without the `-fmoe` will produce a better performance with Vulkan.

---

👤 **ikawrakow** commented on **2025-07-23** at **07:45:39**

So, with the model that I quantize myself I see
```
llm_load_print_meta: BOS token        = 151643 '<|endoftext|>'
```

But with the model from Unsloth I see
```
llm_load_print_meta: BOS token        = 11 ','
```

If I run
```
./bin/llama-server --override-kv tokenizer.ggml.bos_token_id=int:151643 $other_args
```
the pauses are gone.

If I run the Unsloth model in mainline, it also shows the BOS token as 11, but for whatever reason it does not cause an issue there. Perhaps somewhere in `ik_llama.cpp` the `add_bos_token = false` is being ignored?

---

👤 **gapeleon** commented on **2025-07-23** at **07:47:29**

Forgive me if this is already known; but the comma-pause bug seems to depend on what directly precedes the comma.

You can see this with the following prompt:

```
Repeat the following, verbatim:

**Fast**:

`punctuation + comma + space`  eg:
", ")
 ;, test

**Slow**:
`letter + comma + space` eg:
word, 

`number + comma + space` eg:
1, 2,

`Newline followed by comma` :
,

```

I haven't checked the different token ids yet.

---

👤 **ikawrakow** commented on **2025-07-23** at **07:51:31**

This is not known, at least not known by me.

But do you still see the pauses if you add `--override-kv tokenizer.ggml.bos_token_id=int:151643` to your command line?

---

👤 **gapeleon** commented on **2025-07-23** at **08:06:22**

I just tested, no more pauses after adding that 👍

---

👤 **ikawrakow** commented on **2025-07-23** at **08:29:47**

OK, I found the culprit. It is the change in warm up that was added in [#198](https://github.com/ikawrakow/ik_llama.cpp/issues/198).

Basically, to improve the experience for MoE models, [#198](https://github.com/ikawrakow/ik_llama.cpp/issues/198) changed the warm up to use all experts. But the warm up is being detected with this line of code
```
bool is_warming_up = (batch.n_tokens == 1 && (batch.token[0] == ((bos != -1) ? bos : eos)));
```
which works out to `true` when we have a comma and bos token is set to `,`. So, each time there is a comma, all 128 experts get used, which makes the calculation basically 16 times slower.

This is not done in mainline, so the issue does not exist even when the BOS token is set to be a comma.

@saood06 Do you want to fix it, or should I?

---

👤 **saood06** commented on **2025-07-23** at **08:56:37**

>@saood06 Do you want to fix it, or should I?

I can add support for the llama_add_bos_token_impl to the warmup code (mainline warmup code that they merged in is very different it is post refactor).

Did the models you quantize have `tokenizer.ggml.add_bos_token = false`?

---

👤 **ikawrakow** commented on **2025-07-23** at **09:01:03**

> Did the models you quantize have tokenizer.ggml.add_bos_token = false

Yes. But the BOS token id was set to 151643 instead of 11. I see the timestamp of the `bf16` GGUF to be April 29, but I do not remember if I used the `ik_llama.cpp` or the `llama.cpp` `convert_hf_to_gguf.py` script to create it from the safetensors.

---

👤 **samteezy** commented on **2025-07-23** at **09:09:07**

> @samteezy  The Vulkan backend does not have an implementation for the `-fmoe` command line argument. The result is that the fused `ffn_up+ffn_gate` op that is enabled by `-fmoe` will be run on the CPU. I'm also wondering why you use `-ngl 0`. My guess is that `-ngl 100 -ot exps=CPU` without the `-fmoe` will produce a better performance with Vulkan. 

Yeah, I realized that after I posted - I was just mucking about with various settings to see how performance changed. Normally on mainline I'm offloading experts to CPU as needed and leave ngl set to 99. I hadn't quite understood what -fmoe did yet, thanks for the detail.

---

👤 **saood06** commented on **2025-07-23** at **09:15:38**

I just checked [this](https://huggingface.co/ubergarm/Qwen3-Coder-480B-A35B-Instruct-GGUF/tree/main/IQ2_KS?show_file_info=IQ2_KS%2FQwen3-480B-A35B-Instruct-IQ2_KS-00001-of-00004.gguf) which also has it false. So at least the ones being reported should be covered.

---

👤 **ikawrakow** commented on **2025-07-23** at **09:59:00**

This should be fixed on latest main (after merging [#639](https://github.com/ikawrakow/ik_llama.cpp/issues/639)). You should not need to override the BOS token ID.

But if there are still pauses, let me know.