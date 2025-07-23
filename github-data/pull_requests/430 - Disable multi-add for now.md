### 🔀 [#430](https://github.com/ikawrakow/ik_llama.cpp/pull/430) - Disable multi-add for now

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-18 |
| **Updated** | 2025-05-23 |

---

#### Description

There have been several crash reports (#398, #425) for large MoE models when using hybrid GPU/CPU inference. As I don't have the hardware to run such large models I'm not able to debug. But with help from @nux, who ran `llama-server` in the debugger on his computer and gave me a backtrace along with a few variables values (see [this](https://github.com/ikawrakow/ik_llama.cpp/issues/425#issuecomment-2888464768) and [this](https://github.com/ikawrakow/ik_llama.cpp/issues/425#issuecomment-2888500696), my hypothesis is that the problem is with the multi-add operation that I added to `ik_llama.cpp`.

I'm of course not sure if the hypothesis is correct as it is based on very scarce evidence. Hence I would appreciate if the people reporting a problem test this PR and let me know if it fixes the problem, so pinging @Panchovix, @ciprianveg, @pt13762104, @schynce, @p4s2wd

### Background

What is multi-add? In MoE models the contributions of the routed experts need to be added together. In mainline `llama.cpp` this is done via `N-1` consecutive `GGML_OP_ADD` operations, where `N` is the number of active experts. This is not a problem when `N` is small as in the early MoE models (e.g., Mixtral8x7 with 2 active experts). But more recent models such as DeepSeek-V3/R1 and Qwen3-235B-A22B have 8 active experts, so this means 7 additional graph nodes with 7 additional synchronization points, causing a non-negligible overhead. Hence, I added the multi-add operation, which adds `N` tensors in one go. The operation works fine if everything is done on one device. But it looks like things can go wrong when data needs to be copied between devices. I don't observe the problem when using the smaller siblings of these models (Qwen-22B-A3B and DeepSeek-Lite) in my setup with hybrid GPU/CPU inference, but looking at my implementation it appears there could be an issue. The PR reverts to the original implementation, so it will result in a small performance penalty (2-3% with the models I can run).

---

#### 💬 Conversation

👤 **schynce** commented the **2025-05-18** at **10:10:42**:<br>

Hi!

I tested the ik/disable_multi_add branch, but it unfortunately did not solve the issue.

Running this command:

```
./llama-server --model /mnt/Qwen3-235B-A22B-IQ4_XS-00001-of-00003.gguf --alias Qwen3-235B-A22B-IQ4_XS \
-fa -fmoe -rtr -c 40960 -ctk q8_0 -ctv q8_0 --threads 8 --no-kv-offload \
-ot "blk\.\d+\.attn=CUDA2" \
-ot "blk\.(0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17)\.=CUDA0" \
-ot "blk\.(18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35)\.=CUDA1" \
-ot "blk\.(36|37|38|39|40|41|42|43|44|45|46|47|48|49|50|51)\.=CUDA2"
```

Results in the following:

```
llm_load_tensors: offloading 0 repeating layers to GPU
llm_load_tensors: offloaded 0/95 layers to GPU
llm_load_tensors:  CUDA_Host buffer size = 52313.37 MiB
llm_load_tensors:      CUDA0 buffer size = 22068.28 MiB
llm_load_tensors:      CUDA1 buffer size = 22068.28 MiB
llm_load_tensors:      CUDA2 buffer size = 23042.94 MiB
....................................................................................................
============ Repacked 127 tensors
llama_new_context_with_model: n_ctx      = 40960
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:  CUDA_Host KV buffer size =  3995.00 MiB
llama_new_context_with_model: KV self size  = 3995.00 MiB, K (q8_0): 1997.50 MiB, V (q8_0): 1997.50 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     1.16 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =   104.50 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =   104.50 MiB
llama_new_context_with_model:      CUDA2 compute buffer size =   189.25 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   304.75 MiB
llama_new_context_with_model: graph nodes  = 4894
llama_new_context_with_model: graph splits = 432
INFO [                    init] initializing slots | tid="140536058970112" timestamp=1747562221 n_slots=1
INFO [                    init] new slot | tid="140536058970112" timestamp=1747562221 id_slot=0 n_ctx_slot=40960
INFO [                    main] model loaded | tid="140536058970112" timestamp=1747562221
INFO [                    main] chat template | tid="140536058970112" timestamp=1747562221 chat_example="<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there<|im_end|>\n<|im_start|>user\nHow are you?<|im_end|>\n<|im_start|>assistant\n" built_in=true
INFO [                    main] HTTP server listening | tid="140536058970112" timestamp=1747562221 n_threads_http="15" port="5000" hostname="127.0.0.1"
INFO [            update_slots] all slots are idle | tid="140536058970112" timestamp=1747562221
INFO [      log_server_request] request | tid="140533654548480" timestamp=1747562221 remote_addr="127.0.0.1" remote_port=54622 status=200 method="GET" path="/v1/models" params={}
INFO [      log_server_request] request | tid="140533646155776" timestamp=1747562221 remote_addr="127.0.0.1" remote_port=36468 status=200 method="GET" path="/v1/models" params={}
INFO [      log_server_request] request | tid="140533637763072" timestamp=1747562325 remote_addr="127.0.0.1" remote_port=39456 status=200 method="GET" path="/v1/models" params={}
INFO [      log_server_request] request | tid="140533629370368" timestamp=1747562325 remote_addr="127.0.0.1" remote_port=39462 status=200 method="GET" path="/v1/models" params={}
INFO [      log_server_request] request | tid="140533620977664" timestamp=1747562329 remote_addr="127.0.0.1" remote_port=60618 status=200 method="GET" path="/v1/models" params={}
INFO [   launch_slot_with_task] slot is processing task | tid="140536058970112" timestamp=1747562329 id_slot=0 id_task=0
INFO [            update_slots] kv cache rm [p0, end) | tid="140536058970112" timestamp=1747562329 id_slot=0 id_task=0 p0=0
INFO [           print_timings] prompt eval time     =    1336.82 ms /    18 tokens (   74.27 ms per token,    13.46 tokens per second) | tid="140536058970112" timestamp=1747562444 id_slot=0 id_task=0 t_prompt_processing=1336.819 n_prompt_tokens_processed=18 t_token=74.26772222222222 n_tokens_second=13.464799647521467
INFO [           print_timings] generation eval time =  113540.01 ms /   817 runs   (  138.97 ms per token,     7.20 tokens per second) | tid="140536058970112" timestamp=1747562444 id_slot=0 id_task=0 t_token_generation=113540.008 n_decoded=817 t_token=138.97185801713587 n_tokens_second=7.195701448250735
INFO [           print_timings]           total time =  114876.83 ms | tid="140536058970112" timestamp=1747562444 id_slot=0 id_task=0 t_prompt_processing=1336.819 t_token_generation=113540.008 t_total=114876.827
INFO [            update_slots] slot released | tid="140536058970112" timestamp=1747562444 id_slot=0 id_task=0 n_ctx=40960 n_past=834 n_system_tokens=0 n_cache_tokens=834 truncated=false
INFO [            update_slots] all slots are idle | tid="140536058970112" timestamp=1747562444
INFO [      log_server_request] request | tid="140533612584960" timestamp=1747562444 remote_addr="127.0.0.1" remote_port=56040 status=200 method="POST" path="/v1/chat/completions" params={}
INFO [            update_slots] all slots are idle | tid="140536058970112" timestamp=1747562444
INFO [      log_server_request] request | tid="140533604192256" timestamp=1747562479 remote_addr="127.0.0.1" remote_port=47776 status=200 method="GET" path="/v1/models" params={}
INFO [      log_server_request] request | tid="140533595799552" timestamp=1747562484 remote_addr="127.0.0.1" remote_port=47790 status=200 method="GET" path="/v1/models" params={}
INFO [   launch_slot_with_task] slot is processing task | tid="140536058970112" timestamp=1747562484 id_slot=0 id_task=819
INFO [            update_slots] kv cache rm [p0, end) | tid="140536058970112" timestamp=1747562484 id_slot=0 id_task=819 p0=1
CUDA error: an illegal memory access was encountered
  current device: 2, in function ggml_backend_cuda_synchronize at /home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu:3067
  cudaStreamSynchronize(cuda_ctx->stream())
/home/user/ik_llama.cpp/ggml/src/ggml-cuda.cu:110: CUDA error
Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
ptrace: Operation not permitted.
No stack.
The program is not being run.
Aborted (core dumped)
```

I tested once again just to be sure, and I can confirm that this command does *not* crash:

```
./llama-server --model /mnt/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf --alias Qwen3-235B-A22B-mix-IQ3_K \
-fa -fmoe -rtr -c 40960 -ctk q8_0 -ctv q8_0 --threads 7 --no-kv-offload \
-ot "blk\.\d+\.attn=CUDA2" \
-ot "blk\.(0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20)\.=CUDA0" \
-ot "blk\.(21|22|23|24|25|26|27|28|29|30|31|32|33|34|35|36|37|38|39|40|41)\.=CUDA1" \
-ot "blk\.(42|43|44|45|46|47|48|49|50|51|52|53|54|55|56|57)\.=CUDA2"
```

Also, as suggested in #398 by @Ph0rk0z, running without -fa seems to not crash:

```
./llama-server --model /mnt/Qwen3-235B-A22B-IQ4_XS-00001-of-00003.gguf --alias Qwen3-235B-A22B-IQ4_XS \
-fmoe -rtr -c 40960 --threads 7 --no-kv-offload \
-ot "blk\.\d+\.attn=CUDA2" \
-ot "blk\.(0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17)\.=CUDA0" \
-ot "blk\.(18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35)\.=CUDA1" \
-ot "blk\.(36|37|38|39|40|41|42|43|44|45|46|47|48|49|50)\.=CUDA2"
```

---

👤 **ikawrakow** commented the **2025-05-18** at **11:50:50**:<br>

To be honest, I don't understand what could be wrong.

---

👤 **ChicoPinto70** commented the **2025-05-18** at **21:33:03**:<br>

If I may, I have the same problem running DeepSeekV3 0324. My workaround to avoid this bug is, change the rtr for no_map, use tensor split to the two gpus not connect to the monitor and, in the deepseek case, use MLA 3 instead 2.

My system is a 2xE5 2699v3, 256Gb DDR4 in octachannel, 3xRTX3090, running Ubuntu 24.04 LTS. The command is:

CUDA_VISIBLE_DEVICES="1,2,0" ./build/bin/llama-server --alias unsloth/DeepSeek-V3-0324-UD-IQ2_XXS     --model /home/chico/.lmstudio/models/unsloth/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-UD-IQ2_XXS-00001-of-00005.gguf   -ngl 64 -c 131072 -mla 3 -fa -amb 512 -fmoe -t 32 -ctk q8_0 -ot "blk\.[0-7]\..*_exps\.=CUDA2,exps=CPU"  --host 127.0.0.1 --port 1234  --parallel 1  --numa distribute -ser 7,1 -b 4096 -ub 4096 --no-mmap -ts 1,1,0

I hope it helps.

---

👤 **Ph0rk0z** commented the **2025-05-18** at **21:56:16**:<br>

It happened to me much more when I undervolted hard and had nvidia HDMI audio devices compete for BAR space. Now that I fixed those issues, I am not seeing this a whole lot if at all.

---

👤 **ciprianveg** commented the **2025-05-18** at **21:58:30**:<br>

It isn't a hardware issue, llama.cpp is not experiencing this issue with same settings

---

👤 **schynce** commented the **2025-05-18** at **22:45:19**:<br>

> It isn't a hardware issue, llama.cpp is not experiencing this issue with same settings

I can also confirm that llama.cpp runs fine with the same settings (just without -fmoe and -rtr).

---

👤 **Ph0rk0z** commented the **2025-05-19** at **00:24:03**:<br>

llama.cpp doesn't have fmoe or rtr and has a different fa implementation. Exllama didn't crash on me either :D  
If hardware instability makes it easier to reproduce it could be related. Check nothing funny is in journal or dmesg.

---

👤 **ikawrakow** commented the **2025-05-19** at **06:35:14**:<br>

`ik_llama.cpp` is faster then `llama.cpp`, else you wouldn't be here. If there is a hardware issue or a driver bug, or a bug that exists in `ik_llama.cpp` and in `llama.cpp`, the probability to trigger the problem is likely to be higher when the computation goes faster.

But if the bug is in `ik_llama.cpp` only, I have no going hypothesis what it could be.