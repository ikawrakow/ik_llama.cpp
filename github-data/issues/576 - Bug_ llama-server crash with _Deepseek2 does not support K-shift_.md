### 🐛 [#576](https://github.com/ikawrakow/ik_llama.cpp/issues/576) - Bug: llama-server crash with \"Deepseek2 does not support K-shift\"

| **Author** | `ewhacc` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-07-03 |
| **Updated** | 2025-07-04 |

---

#### Description

### What happened?

llama-server crashed with a message "llama.cpp:18430: Deepseek2 does not support K-shift"
It was during jobs using ubergarm's DeepSeek-V3-0324-IQ2_K_R4.

Relaunched it, it keeps going.  So, It's not reproducible.
In what circumstance, will "Deepseek2 does not support K-shift" be shown?

### Name and Version

$ ik_llama.cpp/build/bin/llama-server --version
version: 3774 (bce7697d)
built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
# Build
cmake -B ./build -DGGML_CUDA=ON -DGGML_BLAS=OFF -DGGML_SCHED_MAX_COPIES=1 -DGGML_CUDA_IQK_FORCE_BF16=1 -DGGML_CUDA_F16=ON

# Run
llama-server --model $model_path \
    --alias DeepSeek-V3-0324 \
    --ctx-size 98304 \
    -mla 3 -fa -amb 512 -fmoe \
    -b 4096 -ub 4096 \
    --n-gpu-layers 63 \
    --override-tensor exps=CPU \
    --parallel 2 --threads 32 \
    --host 0.0.0.0 --port 5000

# Log
INFO [   launch_slot_with_task] slot is processing task | tid="138385419128832" timestamp=1751529199 id_slot=1 id_task=106779
INFO [            update_slots] kv cache rm [p0, end) | tid="138385419128832" timestamp=1751529199 id_slot=1 id_task=106779 p0=9
INFO [            update_slots] kv cache rm [p0, end) | tid="138385419128832" timestamp=1751529229 id_slot=1 id_task=106779 p0=4105
INFO [            update_slots] kv cache rm [p0, end) | tid="138385419128832" timestamp=1751529259 id_slot=1 id_task=106779 p0=8201
INFO [            update_slots] kv cache rm [p0, end) | tid="138385419128832" timestamp=1751529290 id_slot=1 id_task=106779 p0=12297
INFO [            update_slots] kv cache rm [p0, end) | tid="138385419128832" timestamp=1751529321 id_slot=1 id_task=106779 p0=16393
INFO [            update_slots] kv cache rm [p0, end) | tid="138385419128832" timestamp=1751529351 id_slot=1 id_task=106779 p0=20489
INFO [            update_slots] kv cache rm [p0, end) | tid="138385419128832" timestamp=1751529382 id_slot=1 id_task=106779 p0=24585
INFO [            update_slots] kv cache rm [p0, end) | tid="138385419128832" timestamp=1751529413 id_slot=1 id_task=106779 p0=28681
INFO [            update_slots] kv cache rm [p0, end) | tid="138385419128832" timestamp=1751529444 id_slot=1 id_task=106779 p0=32777
INFO [            update_slots] kv cache rm [p0, end) | tid="138385419128832" timestamp=1751529475 id_slot=1 id_task=106779 p0=36873
INFO [            update_slots] kv cache rm [p0, end) | tid="138385419128832" timestamp=1751529506 id_slot=1 id_task=106779 p0=40969
INFO [            update_slots] kv cache rm [p0, end) | tid="138385419128832" timestamp=1751529537 id_slot=1 id_task=106779 p0=45065
INFO [            update_slots] slot context shift | tid="138385419128832" timestamp=1751529662 id_slot=1 id_task=106779 n_keep=1 n_left=49150 n_discard=24575 n_ctx=98304 n_past=49151 n_system_tokens=0 n_cache_tokens=49151
/home/..../ik_llama.cpp/src/llama.cpp:18430: Deepseek2 does not support K-shift
Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
ptrace: Operation not permitted.
No stack.
The program is not being run.
```

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-07-03** at **11:38:54**:<br>

> In what circumstance, will "Deepseek2 does not support K-shift" be shown?

When you reach the maximum context length.

---

👤 **ewhacc** commented the **2025-07-03** at **18:15:28**:<br>

> When you reach the maximum context length.

Did I reach the maximum context length?  p0=45065 just before crash.

n_keep=1 n_left=49150 n_discard=24575 n_ctx=98304 n_past=49151 n_system_tokens=0 n_cache_tokens=49151

Crashed again for the different prompt, but at the same p0=45065.

It was ok with R1.  I'm going to check with R1 again.

---

👤 **saood06** commented the **2025-07-03** at **22:29:38**:<br>

> > When you reach the maximum context length.
> 
> Did I reach the maximum context length? p0=45065 just before crash.
> 
> n_keep=1 n_left=49150 n_discard=24575 n_ctx=98304 n_past=49151 n_system_tokens=0 n_cache_tokens=49151
> 
> Crashed again for the different prompt, but at the same p0=45065.
> 

Yes. 

You set `--parallel 2`, which makes your max context per slot (with 0 system tokens) to 49,152 (`98304 / 2`). Your batch size is 4,096 and so you'd expect to see the last reported context length to be between 45,056 - 49,152, which `45065` falls into. That is the current way slots handle context limit, the cap is set to (`n_ctx` - `n_system_tokens`) divided by the number of slots.

---

👤 **saood06** commented the **2025-07-03** at **22:29:38**:<br>

> > When you reach the maximum context length.
> 
> Did I reach the maximum context length? p0=45065 just before crash.
> 
> n_keep=1 n_left=49150 n_discard=24575 n_ctx=98304 n_past=49151 n_system_tokens=0 n_cache_tokens=49151
> 
> Crashed again for the different prompt, but at the same p0=45065.
> 

Yes. 

You set `--parallel 2`, which makes your max context per slot (with 0 system tokens) to 49,152 (`98304 / 2`). Your batch size is 4,096 and so you'd expect to see the last reported context length to be between 45,056 - 49,152, which `45065` falls into.

---

👤 **ewhacc** commented the **2025-07-04** at **05:16:41**:<br>

@saood06 

Thank so much!   Yeah, that is the difference from my previous run.

I suspected `--parallel 2` but didn't know it divides the context length.