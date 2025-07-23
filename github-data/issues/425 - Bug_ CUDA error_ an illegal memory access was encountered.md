### 🐛 [#425](https://github.com/ikawrakow/ik_llama.cpp/issues/425) - Bug: CUDA error: an illegal memory access was encountered

| **Author** | `nux` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-15 |
| **Updated** | 2025-05-23 |

---

#### Description

### What happened?

Not sure if this is a problem with me or ik_llama - but getting this while starting prompt processing (ubergarm's deepseek-v3)

May 15 08:57:29 red llama-swap[80783]: INFO [   launch_slot_with_task] slot is processing task | tid="139638925832192" timestamp=1747317449 id_slot=0 id_task=3
May 15 08:57:29 red llama-swap[80783]: INFO [            update_slots] kv cache rm [p0, end) | tid="139638925832192" timestamp=1747317449 id_slot=0 id_task=3 p0=0
May 15 08:57:36 red kernel: NVRM: Xid (PCI:0000:01:00): 31, pid=80798, name=llama-server, Ch 00000008, intr 00000000. MMU Fault: ENGINE GRAPHICS GPC1 GPCCLIENT_T1_3 faulted @ 0x7e9f_4f200000. Fault is of type FAULT_PDE ACCESS_TYPE_VIRT_READ
May 15 08:57:36 red llama-swap[80783]: CUDA error: an illegal memory access was encountered
May 15 08:57:36 red llama-swap[80783]:   current device: 0, in function ggml_backend_cuda_synchronize at /home/nux/dev/ik_llama.cpp/ggml/src/ggml-cuda.cu:3067
May 15 08:57:36 red llama-swap[80783]:   cudaStreamSynchronize(cuda_ctx->stream())
May 15 08:57:36 red llama-swap[80783]: /home/nux/dev/ik_llama.cpp/ggml/src/ggml-cuda.cu:110: CUDA error
May 15 08:57:36 red kernel: llama-server[80906]: segfault at 204803fe0 ip 00007f00399189d7 sp 00007ffc4a6104f0 error 4 in libcuda.so.575.51.03[7f00395c5000+e97000] likely on CPU 11 (core 11, socket 0)
May 15 08:57:36 red kernel: Code: ef e8 9d c9 ca ff 83 3d 7e 57 2f 05 01 49 8b 1c 24 76 0a 8b 05 86 57 2f 05 85 c0 74 56 49 8b 44 24 10 41 8b 4c 24 24 48 8b 13 <8b> 00 41 39 c6 74 52 8b b3 40 40 00 00 48 89 f0 89 8c b3 44 40 00



### Name and Version

./build/bin/llama-server --version
version: 3697 (34ae71c4)
built with cc (Debian 12.2.0-14) 12.2.0 for x86_64-linux-gnu

### What operating system are you seeing the problem on?

_No response_

### Relevant log output

```shell

```

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-05-15** at **14:03:32**:<br>

What was the command line?
Are you running this model for the first time? If not, did you experience this error on an earlier `ik_llama.cpp` version?

---

👤 **nux** commented the **2025-05-15** at **14:15:21**:<br>

Here is the command I am running:
/home/nux/dev/ik_llama.cpp/build/bin/llama-server --model /mnt/nvme/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-IQ4_K_R4/DeepSeek-V3-0324-IQ4_K_R4-00001-of-00010.gguf --ctx-size 32768 -mla 2 -fa -amb 512 -fmoe --temp 0.3 --min-p 0.05 --n-gpu-layers 63 --override-tensor "exps=CPU" --parallel 1 --threads 32 --host 0.0.0.0 --port 8081

This is the model I use primarily - been working well for a while now. I pulled it out of my normal llama-swap setup and running manually....

It worked when I sent a random benchmark to it (solobench). When I attempt to redo a prompt sent from open-webui it crashed again:
INFO [            update_slots] kv cache rm [p0, end) | tid="139648547147776" timestamp=1747318222 id_slot=0 id_task=0 p0=0
CUDA error: an illegal memory access was encountered
  current device: 0, in function ggml_backend_cuda_synchronize at /home/nux/dev/ik_llama.cpp/ggml/src/ggml-cuda.cu:3067
  cudaStreamSynchronize(cuda_ctx->stream())
/home/nux/dev/ik_llama.cpp/ggml/src/ggml-cuda.cu:110: CUDA error


I will attempt to send the same prompt that I am sending in open webui, with the cli client I used when it worked...

Odd. The prompt was pasting some php code and a warning it was throwing.

Do you want me to try and get the prompt posted for you? Would try and remove parts of prompt I don't really want to post on github, and see if it still crashes.

---

👤 **Panchovix** commented the **2025-05-15** at **14:18:06**:<br>

If you try without -fmoe, does it works?

---

👤 **nux** commented the **2025-05-15** at **14:19:31**:<br>

Nope:

INFO [            update_slots] kv cache rm [p0, end) | tid="140102707097600" timestamp=1747318723 id_slot=0 id_task=0 p0=0
CUDA error: an illegal memory access was encountered
  current device: 0, in function ggml_backend_cuda_synchronize at /home/nux/dev/ik_llama.cpp/ggml/src/ggml-cuda.cu:3067
  cudaStreamSynchronize(cuda_ctx->stream())
/home/nux/dev/ik_llama.cpp/ggml/src/ggml-cuda.cu:110: CUDA error
Aborted

Command I ran was:

nux@red ~/dev/ik_llama.cpp $ /home/nux/dev/ik_llama.cpp/build/bin/llama-server --model /mnt/nvme/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-IQ4_K_R4/DeepSeek-V3-0324-IQ4_K_R4-00001-of-00010.gguf --alias ubergarm/DeepSeek-R1-V3-0324-IQ4_K_R4 --alias "deepseek-v3" --ctx-size 32768 -mla 2 -fa -amb 512 --temp 0.3 --min-p 0.05 --n-gpu-layers 63 --override-tensor "exps=CPU" --parallel 1 --threads 32 --host 0.0.0.0 --port 8080

---

👤 **nux** commented the **2025-05-15** at **14:19:54**:<br>

Would you like me to try with llama.cpp vanilla? Err...I'm not sure that model loads there. Perhaps I could try other models if you think it would be useful

---

👤 **Panchovix** commented the **2025-05-15** at **14:21:22**:<br>

I think R4 doesn't work on llamacpp, yeah. You can try with unsloth quants there https://huggingface.co/unsloth/DeepSeek-V3-0324-GGUF-UD.

---

👤 **ikawrakow** commented the **2025-05-15** at **14:24:51**:<br>

There is a place in the log that looks like this:
```
llama_model_loader: - type  f32:   66 tensors
llama_model_loader: - type q6_K:    1 tensors
llama_model_loader: - type iq5_ks:  225 tensors
```
Seeing this will be helpful.

---

👤 **nux** commented the **2025-05-15** at **14:29:04**:<br>

llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  612 tensors
llama_model_loader: - type iq4_k_r4:  116 tensors
llama_model_loader: - type iq5_k_r4:   58 tensors

I ran the same prompt through: unsloth/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf with ik_llama /home/nux/dev/ik_llama.cpp/build/bin/llama-server --model /mnt/nvme/models/unsloth/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf --ctx-size 32768 -mla 2 -fa -amb 512 --temp 0.3 --min-p 0.05 --n-gpu-layers 63 --override-tensor "exps=CPU" --parallel 1 --threads 32 --host 0.0.0.0 --port 8080
and it worked fine.

It looks like I do have unsloth/DeepSeek-V3-0324-GGUF/UD-Q4_K_XL on a network storage. If you want me to test that I can.

Edit: I will throw another prompt at the model I had a problem with for some other php stuff and see how it goes. At this point if it's that one prompt and everything else works fine...we could close this for now

---

👤 **nux** commented the **2025-05-15** at **14:35:38**:<br>

Worked with another php related prompt (first one had a ~80 line function pasted in, this one was only 5 lines). Odd...

---

👤 **ikawrakow** commented the **2025-05-15** at **14:36:20**:<br>

> It looks like I do have unsloth/DeepSeek-V3-0324-GGUF/UD-Q4_K_XL on a network storage. If you want me to test that I can.

If you have time, yes, this can be helpful.

But based on the described symptoms
* It was working before with the same model where we get illegal memory access now
* There are no tensors that were computed on the CPU before and are now computed on the GPU

I have no hypothesis what changed. You can try using `-mla 3` instead of `-mla 2` as this is now supported on CUDA. It may make your TG speed better (especially for long context), but it also eliminates two matrix multiplications that are done in the FA kernel.

---

👤 **nux** commented the **2025-05-15** at **18:04:11**:<br>

Interesting...I've been trying various combinations of models/parameters, and so far here's what I have:

ik_llama crashes with ds v3 with unsloth or ubergarms variant. 

If I run it fully on CPU it doesnt crash:
/home/nux/dev/ik_llama.cpp/build/bin/llama-server --model /mnt/nvme/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-IQ4_K_R4/DeepSeek-V3-0324-IQ4_K_R4-00001-of-00010.gguf --alias "deepseek-v3" --ctx-size 32768 -fa --temp 0.3 --min-p 0.05 --n-gpu-layers 0 --override-tensor "exps=CPU" --parallel 1 --threads 32 --host 0.0.0.0 --port 8080 

If I put a single layer on GPU it does crash:
/home/nux/dev/ik_llama.cpp/build/bin/llama-server --model /mnt/nvme/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-IQ4_K_R4/DeepSeek-V3-0324-IQ4_K_R4-00001-of-00010.gguf --alias "deepseek-v3" --ctx-size 32768 -fa --temp 0.3 --min-p 0.05 --n-gpu-layers 1 --override-tensor "exps=CPU" --parallel 1 --threads 32 --host 0.0.0.0 --port 8080 

I left off all the mla amb fmoe stuff. Going to see if it crashes with vanilla llama.cpp

ik_llama.cpp:
CUDA error: an illegal memory access was encountered
  current device: 0, in function ggml_backend_cuda_synchronize at /home/nux/dev/ik_llama.cpp/ggml/src/ggml-cuda.cu:3067
  cudaStreamSynchronize(cuda_ctx->stream())
/home/nux/dev/ik_llama.cpp/ggml/src/ggml-cuda.cu:110: CUDA error
Aborted
nux@red ~/dev/ik_llama.cpp $ /home/nux/dev/ik_llama.cpp/build/bin/llama-server --model /mnt/amp/models/unsloth/DeepSeek-V3-0324-GGUF/UD-Q4_K_XL/DeepSeek-V3-0324-UD-Q4_K_XL-00001-of-00009.gguf --alias "deepseek-v3" --ctx-size 32768 -fa --temp 0.3 --min-p 0.05 --n-gpu-layers 1 --override-tensor "exps=CPU" --parallel 1 --threads 32 --host 0.0.0.0 --port 8080


llama.cpp:
/home/nux/dev/llama.cpp/build/bin/llama-server --model /mnt/amp/models/unsloth/DeepSeek-V3-0324-GGUF/UD-Q4_K_XL/DeepSeek-V3-0324-UD-Q4_K_XL-00001-of-00009.gguf --alias deepseek-v3 --ctx-size 32768 -fa --temp 0.3 --min-p 0.05 --n-gpu-layers 1 --override-tensor exps=CPU --parallel 1 --threads 32 --host 0.0.0.0 --port 8080

and it works. 
prompt eval time =   21301.26 ms /   515 tokens (   41.36 ms per token,    24.18 tokens per second)
eval time =   98803.52 ms /   626 tokens (  157.83 ms per token,     6.34 tokens per second)
total time =  120104.78 ms /  1141 tokens

---

👤 **ciprianveg** commented the **2025-05-17** at **07:10:26**:<br>

> Here is the command I am running: /home/nux/dev/ik_llama.cpp/build/bin/llama-server --model /mnt/nvme/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-IQ4_K_R4/DeepSeek-V3-0324-IQ4_K_R4-00001-of-00010.gguf --ctx-size 32768 -mla 2 -fa -amb 512 -fmoe --temp 0.3 --min-p 0.05 --n-gpu-layers 63 --override-tensor "exps=CPU" --parallel 1 --threads 32 --host 0.0.0.0 --port 8081
> 
> This is the model I use primarily - been working well for a while now. I pulled it out of my normal llama-swap setup and running manually....
> 
> It worked when I sent a random benchmark to it (solobench). When I attempt to redo a prompt sent from open-webui it crashed again: INFO [ update_slots] kv cache rm [p0, end) | tid="139648547147776" timestamp=1747318222 id_slot=0 id_task=0 p0=0 CUDA error: an illegal memory access was encountered current device: 0, in function ggml_backend_cuda_synchronize at /home/nux/dev/ik_llama.cpp/ggml/src/ggml-cuda.cu:3067 cudaStreamSynchronize(cuda_ctx->stream()) /home/nux/dev/ik_llama.cpp/ggml/src/ggml-cuda.cu:110: CUDA error
> 
> I will attempt to send the same prompt that I am sending in open webui, with the cli client I used when it worked...
> 
> Odd. The prompt was pasting some php code and a warning it was throwing.
> 
> Do you want me to try and get the prompt posted for you? Would try and remove parts of prompt I don't really want to post on github, and see if it still crashes.

Similar thing happens to me, it worked 2 days ago, i rebuilt it with latest sources yesterday.  I am using qwen3 235b q4 ud xl. Llama-sweep-bench works fine but when i send a real prompt via open web ui, it crashes.. On the last days since it worked I changed nvidia driver from 535 to 550,  cuda version from 12.2 to 12.6, i pulled latest changes yesterday and rebuilt..

---

👤 **ikawrakow** commented the **2025-05-17** at **07:36:58**:<br>

@ciprianveg Can you also give the build for the last version that worked, tell us if the crash happens during PP or during TG, and post the line from the log where it says where the illegal memory access was encountered? Thanks. Also, is it a single GPU or a multi-GPU setup?

---

👤 **ciprianveg** commented the **2025-05-17** at **08:23:13**:<br>

Hello, it was built from main 20 h ago, now i rebuilt from main 30m ago with latest changes (from 2h ago) and same error:
INFO [            update_slots] kv cache rm [p0, end) | tid="136731577430016" timestamp=1747469764 id_slot=0 id_task=0 p0=0
VERB [            update_slots] prompt processing progress | tid="136731577430016" timestamp=1747469764 id_slot=0 n_past=33 n_ctx=20480 n_tokens=33 progress=1.0
VERB [            update_slots] prompt done | tid="136731577430016" timestamp=1747469764 id_slot=0 n_past=33 n_ctx=20480 n_tokens=33
VERB [            update_slots] decoding batch | tid="136731577430016" timestamp=1747469764 n_tokens=33
CUDA error: an illegal memory access was encountered
  current device: 2, in function ggml_backend_cuda_synchronize at /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu:3067
  cudaStreamSynchronize(cuda_ctx->stream())
/home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu:110: CUDA error
Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
ptrace: Operation not permitted.
No stack.
The program is not being run.
Aborted (core dumped)

This was the test query: Tell me a random fun fact about the Roman Empire

what is strange is the with exact same command th llama-sweeb-bench works ok:
main: n_kv_max = 20480, n_batch = 2048, n_ubatch = 2048, flash_attn = 1, n_gpu_layers = 99, n_threads = 16, n_threads_batch = 16

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  2048 |    512 |      0 |   12.843 |   159.47 |   51.788 |     9.89 |
|  2048 |    512 |   2048 |   13.000 |   157.54 |   51.361 |     9.97 |


last main pull done, that worked was 3 days ago..

---

👤 **ciprianveg** commented the **2025-05-17** at **08:23:13**:<br>

Hello, it was built from main 20 h ago, now i rebuilt from main 30m ago with latest changes (from 2h ago) and same error:
INFO [            update_slots] kv cache rm [p0, end) | tid="136731577430016" timestamp=1747469764 id_slot=0 id_task=0 p0=0
VERB [            update_slots] prompt processing progress | tid="136731577430016" timestamp=1747469764 id_slot=0 n_past=33 n_ctx=20480 n_tokens=33 progress=1.0
VERB [            update_slots] prompt done | tid="136731577430016" timestamp=1747469764 id_slot=0 n_past=33 n_ctx=20480 n_tokens=33
VERB [            update_slots] decoding batch | tid="136731577430016" timestamp=1747469764 n_tokens=33
CUDA error: an illegal memory access was encountered
  current device: 2, in function ggml_backend_cuda_synchronize at /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu:3067
  cudaStreamSynchronize(cuda_ctx->stream())
/home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu:110: CUDA error
Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
ptrace: Operation not permitted.
No stack.
The program is not being run.
Aborted (core dumped)

This was the test query: Tell me a random fun fact about the Roman Empire

---

👤 **ciprianveg** commented the **2025-05-17** at **08:31:25**:<br>

it happens with both Qwen3-235B-A22B-UD-Q3_K_XL and Qwen3-235B-A22B-UD-Q4_K_XL.  I am using 2 3090 gpus and 2 A4000, built  with 1 copy of cache parameter. I think the multiple gpus can be the issue.. but iti is very strange that llama sweep bench works
..

---

👤 **ciprianveg** commented the **2025-05-17** at **08:31:25**:<br>

it happens with both Qwen3-235B-A22B-UD-Q3_K_XL and Qwen3-235B-A22B-UD-Q4_K_XL

---

👤 **ikawrakow** commented the **2025-05-17** at **08:37:44**:<br>

Strange. Nothing really changed since 3 days ago that could affect your use case.
The illegal memory access is triggered in the back-end, so most likely when data is being copied from the CPU to the GPU.

What happens if you do
```
git checkout 0c57f84dc41aa756dae7b1aaee0d3db6ecc14300
```
to checkout the last version from 4 days ago, and then build & run as usual?

---

👤 **ciprianveg** commented the **2025-05-17** at **08:40:15**:<br>

I will try and let you know. I added 2 more gpus to my first 2... maybe it
also matters

On Sat, 17 May 2025, 11:38 Kawrakow, ***@***.***> wrote:

> *ikawrakow* left a comment (ikawrakow/ik_llama.cpp#425)
> <https://github.com/ikawrakow/ik_llama.cpp/issues/425#issuecomment-2888226845>
>
> Strange. Nothing really changed since 3 days ago that could affect your
> use case.
> The illegal memory access is triggered in the back-end, so most likely
> when data is being copied from the CPU to the GPU.
>
> What happens if you do
>
> git checkout 0c57f84dc41aa756dae7b1aaee0d3db6ecc14300
>
> to checkout the last version from 4 days ago, and then build & run as
> usual?
>
> —
> Reply to this email directly, view it on GitHub
> <https://github.com/ikawrakow/ik_llama.cpp/issues/425#issuecomment-2888226845>,
> or unsubscribe
> <https://github.com/notifications/unsubscribe-auth/AJTBYK7H62CRHBF3PCILEAD263YO5AVCNFSM6AAAAAB5GG6KRWVHI2DSMVQWIX3LMV43OSLTON2WKQ3PNVWWK3TUHMZDQOBYGIZDMOBUGU>
> .
> You are receiving this because you were mentioned.Message ID:
> ***@***.***>
>

---

👤 **ciprianveg** commented the **2025-05-17** at **09:15:05**:<br>

i checked out and built the above version from 4 days ago and the same error, so it looks like it has to do with multiple gpus..

---

👤 **ikawrakow** commented the **2025-05-17** at **09:19:01**:<br>

OK, it is the bug that happens with multiple GPUs and partial offload (multi-GPU with full offload is known to work) that has been reported by several users. It is a bug that I currently cannot solve because I don't have access to a multi-GPU system.

---

👤 **ciprianveg** commented the **2025-05-17** at **09:22:18**:<br>

i tried same command, on llama.cpp, without -fmoe (obvious) and it works, much slower pp process peed but it works. On ik_llama same error happens with or without -fmoe param.

---

👤 **ciprianveg** commented the **2025-05-17** at **09:22:18**:<br>

i treied same command, on llama.cpp, without -fmoe (obvious) and it works, much slower pp process peed but it works

---

👤 **ciprianveg** commented the **2025-05-17** at **09:25:06**:<br>

what is very strange is that the sweep-bench works, till the max cache length set, so what can be different?

---

👤 **ikawrakow** commented the **2025-05-17** at **09:33:11**:<br>

Are you exceeding the max cache size and it crashes then? Or does it crash before?

---

👤 **ciprianveg** commented the **2025-05-17** at **09:34:12**:<br>

llama-sweep-bench works till it exceeds the max cache size

---

👤 **ikawrakow** commented the **2025-05-17** at **09:37:03**:<br>

> llama-sweep-bench works till it exceeds the max cache size

Yes, I got that part. So, I'm wondering if `llama-server` crashes after the max. cache size is reached or before?

---

👤 **ikawrakow** commented the **2025-05-17** at **09:56:36**:<br>

> llama-sweep-bench works till it exceeds the max cache size

OK, this gives me another idea. Can you try running `sweep-bench` with some unusual u-batch size? Add e.g., `-ub 873` to the `sweep-bench` command. If this crashes, I would finally have an indication where to look for the problem. There have been several bug fixes in `llama.cpp` very recently related to clearing compute buffers and padding, so maybe it is just that. I cannot easily pick up their bug fixes as the code bases have massively diverged, but at least I would know where to try.

---

👤 **ciprianveg** commented the **2025-05-17** at **10:11:51**:<br>

I tried with unusual ub it still works, also with unusual nbatch and it works..
main: n_kv_max = 20480, n_batch = 1234, n_ubatch = 873, flash_attn = 1, n_gpu_layers = 99, n_threads = 16, n_threads_batch = 16

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   873 |    218 |      0 |   10.950 |    79.72 |   21.436 |    10.17 |

---

👤 **ikawrakow** commented the **2025-05-17** at **10:17:34**:<br>

OK, this is becoming a real puzzle. Have you tried `llama-cli` ?

---

👤 **ciprianveg** commented the **2025-05-17** at **14:44:57**:<br>

llama-cli seems to work, but is not webui issue as it appeared also from other client

---

👤 **nux** commented the **2025-05-17** at **15:00:59**:<br>

Was reading latest comments on this and wanted to point out I have a single GPU. If you want me to test any more stuff let me know

---

👤 **ciprianveg** commented the **2025-05-17** at **15:02:55**:<br>

On  one gpu the issue doesn't happen

---

👤 **ikawrakow** commented the **2025-05-17** at **15:19:32**:<br>

It seems the issue only occurs when using `llama-server`. 

If someone would build with `-DCMAKE_BUILD_TYPE=RelWithDebInfo`, run it in the debugger
```
gdb --args your_command_that_triggers_the_crash_goes_here
```
and would send the backtrace when it crashes, that would be very useful.

---

👤 **nux** commented the **2025-05-17** at **15:43:50**:<br>

#0  __pthread_kill_implementation (threadid=<optimized out>, signo=signo@entry=6,
    no_tid=no_tid@entry=0) at ./nptl/pthread_kill.c:44
#1  0x00007fffeb8a9f4f in __pthread_kill_internal (signo=6, threadid=<optimized out>)
    at ./nptl/pthread_kill.c:78
#2  0x00007fffeb85afb2 in __GI_raise (sig=sig@entry=6) at ../sysdeps/posix/raise.c:26
#3  0x00007fffeb845472 in __GI_abort () at ./stdlib/abort.c:79
#4  0x000055555558ff52 in ggml_abort (
    file=0x55555634ba10 "/home/nux/dev/ik_llama.cpp/ggml/src/ggml-cuda.cu", line=110,
    fmt=<optimized out>) at /home/nux/dev/ik_llama.cpp/ggml/src/ggml.c:270
#5  0x0000555555810534 in ggml_cuda_error (
    stmt=stmt@entry=0x55555634c128 "cudaStreamSynchronize(cuda_ctx->stream())",
    func=func@entry=0x55555634b5bc "ggml_backend_cuda_synchronize",
    file=file@entry=0x55555634ba10 "/home/nux/dev/ik_llama.cpp/ggml/src/ggml-cuda.cu",
    line=line@entry=3067, msg=0x7ffff7c95d68 "an illegal memory access was encountered")
    at /home/nux/dev/ik_llama.cpp/ggml/src/ggml-cuda.cu:110
#6  0x0000555555810f0a in ggml_backend_cuda_synchronize (backend=<optimized out>)
    at /home/nux/dev/ik_llama.cpp/ggml/src/ggml-cuda.cu:3067
#7  0x00005555557f627b in ggml_backend_synchronize (backend=0x555566e6d9b0)
    at /home/nux/dev/ik_llama.cpp/ggml/src/ggml-backend.c:273
#8  ggml_backend_sched_compute_splits (sched=0x5555647fdcb0)
    at /home/nux/dev/ik_llama.cpp/ggml/src/ggml-backend.c:1833
#9  ggml_backend_sched_graph_compute_async (sched=0x5555647fdcb0, graph=<optimized out>)
    at /home/nux/dev/ik_llama.cpp/ggml/src/ggml-backend.c:2043
#10 0x00005555556fef93 in llama_graph_compute (n_threads=32, gf=0x7f9f020fc030, lctx=...)
    at /home/nux/dev/ik_llama.cpp/src/llama.cpp:17694
#11 llama_decode_internal (batch_all=..., lctx=...)
    at /home/nux/dev/ik_llama.cpp/src/llama.cpp:17910
#12 llama_decode (ctx=0x555563ffcf60, batch=...) at /home/nux/dev/ik_llama.cpp/src/llama.cpp:22305
#13 0x000055555567ad49 in server_context::update_slots (this=0x7fffffffda30)
--Type <RET> for more, q to quit, c to continue without paging--
    at /home/nux/dev/ik_llama.cpp/examples/server/server.cpp:2355
#14 0x0000555555655b4a in std::function<void ()>::operator()() const (this=0x7fffffffe650)
    at /usr/include/c++/12/bits/std_function.h:591
#15 server_queue::start_loop (this=this@entry=0x7fffffffe568)
    at /home/nux/dev/ik_llama.cpp/examples/server/server.cpp:501
#16 0x00005555555936d0 in main (argc=<optimized out>, argv=<optimized out>)
    at /home/nux/dev/ik_llama.cpp/examples/server/server.cpp:3509

---

👤 **nux** commented the **2025-05-17** at **15:43:50**:<br>

`
#0  __pthread_kill_implementation (threadid=<optimized out>, signo=signo@entry=6,
    no_tid=no_tid@entry=0) at ./nptl/pthread_kill.c:44
#1  0x00007fffeb8a9f4f in __pthread_kill_internal (signo=6, threadid=<optimized out>)
    at ./nptl/pthread_kill.c:78
#2  0x00007fffeb85afb2 in __GI_raise (sig=sig@entry=6) at ../sysdeps/posix/raise.c:26
#3  0x00007fffeb845472 in __GI_abort () at ./stdlib/abort.c:79
#4  0x000055555558ff52 in ggml_abort (
    file=0x55555634ba10 "/home/nux/dev/ik_llama.cpp/ggml/src/ggml-cuda.cu", line=110,
    fmt=<optimized out>) at /home/nux/dev/ik_llama.cpp/ggml/src/ggml.c:270
#5  0x0000555555810534 in ggml_cuda_error (
    stmt=stmt@entry=0x55555634c128 "cudaStreamSynchronize(cuda_ctx->stream())",
    func=func@entry=0x55555634b5bc "ggml_backend_cuda_synchronize",
    file=file@entry=0x55555634ba10 "/home/nux/dev/ik_llama.cpp/ggml/src/ggml-cuda.cu",
    line=line@entry=3067, msg=0x7ffff7c95d68 "an illegal memory access was encountered")
    at /home/nux/dev/ik_llama.cpp/ggml/src/ggml-cuda.cu:110
#6  0x0000555555810f0a in ggml_backend_cuda_synchronize (backend=<optimized out>)
    at /home/nux/dev/ik_llama.cpp/ggml/src/ggml-cuda.cu:3067
#7  0x00005555557f627b in ggml_backend_synchronize (backend=0x555566e6d9b0)
    at /home/nux/dev/ik_llama.cpp/ggml/src/ggml-backend.c:273
#8  ggml_backend_sched_compute_splits (sched=0x5555647fdcb0)
    at /home/nux/dev/ik_llama.cpp/ggml/src/ggml-backend.c:1833
#9  ggml_backend_sched_graph_compute_async (sched=0x5555647fdcb0, graph=<optimized out>)
    at /home/nux/dev/ik_llama.cpp/ggml/src/ggml-backend.c:2043
#10 0x00005555556fef93 in llama_graph_compute (n_threads=32, gf=0x7f9f020fc030, lctx=...)
    at /home/nux/dev/ik_llama.cpp/src/llama.cpp:17694
#11 llama_decode_internal (batch_all=..., lctx=...)
    at /home/nux/dev/ik_llama.cpp/src/llama.cpp:17910
#12 llama_decode (ctx=0x555563ffcf60, batch=...) at /home/nux/dev/ik_llama.cpp/src/llama.cpp:22305
#13 0x000055555567ad49 in server_context::update_slots (this=0x7fffffffda30)
--Type <RET> for more, q to quit, c to continue without paging--
    at /home/nux/dev/ik_llama.cpp/examples/server/server.cpp:2355
#14 0x0000555555655b4a in std::function<void ()>::operator()() const (this=0x7fffffffe650)
    at /usr/include/c++/12/bits/std_function.h:591
#15 server_queue::start_loop (this=this@entry=0x7fffffffe568)
    at /home/nux/dev/ik_llama.cpp/examples/server/server.cpp:501
#16 0x00005555555936d0 in main (argc=<optimized out>, argv=<optimized out>)
    at /home/nux/dev/ik_llama.cpp/examples/server/server.cpp:3509
`

---

👤 **nux** commented the **2025-05-17** at **15:46:08**:<br>

[llama-server-bt-full.txt](https://github.com/user-attachments/files/20265607/llama-server-bt-full.txt) Or is this better?

---

👤 **ikawrakow** commented the **2025-05-17** at **16:21:42**:<br>

@nux Thank you for the backtrace. I cannot diagnose what has happened from it alone. I could now start asking you to give me the values of some variables, but this is really too tedious. But perhaps just one thing: 
```
frame 8
p *input
```

---

👤 **nux** commented the **2025-05-17** at **16:34:05**:<br>

Yes I can do that - how exactly do I get that for you? I had to look up that I have to type run into gdb the first time. Never used gdb before.

---

👤 **ikawrakow** commented the **2025-05-17** at **16:41:44**:<br>

When it crashes, and the backtrace is the same as before, you can select the frame where it is in the ` ggml_backend_sched_compute_splits` function. You do this by typing `frame 8` (8 was the frame index in the backtrace you sent). And then you type `p *input`. This will output the content of the `input` tensor. The code is basically iterating over the inputs of the next operation in the graph, and copying data to the appropriate back-end if needed, and I want to see what is the tensor being processed when the crash happens.

But I have to go now, I'll look at the outcome tomorrow.

---

👤 **ikawrakow** commented the **2025-05-17** at **16:41:44**:<br>

When it crashes, and the backtrace is the same as before, you can select the frame where it is in the ` ggml_backend_sched_compute_splits` function. You do this by typing `frame 8` (8 was the frame index in the backtrace you sent). And then you type `p *input`. This will output the content of the `input` tensor. The code is basically iterating over the inputs of the next operation in the graph, and copying data to the appropriate back-end if needed, and I want to see what is the tensor being processed when the crash happens.

---

👤 **nux** commented the **2025-05-17** at **17:19:05**:<br>

(gdb) frame 8
#8  ggml_backend_sched_compute_splits (sched=0x5555647fdcb0)
    at /home/nux/dev/ik_llama.cpp/ggml/src/ggml-backend.c:1833
1833                        ggml_backend_synchronize(input_backend);
(gdb) p *input
$1 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x5555641bc4e0, ne = {7168,
    1, 1, 1}, nb = {4, 28672, 28672, 28672}, op = GGML_OP_RESHAPE, op_params = {
    0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {0x7f9f02436990, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0}, view_src = 0x7f9f02436990, view_offs = 0, data = 0x7f7820000000,
  name = "ffn_moe_weighted-60\000d)", '\000' <repeats 41 times>, extra = 0x0}

---

👤 **ikawrakow** commented the **2025-05-18** at **06:19:46**:<br>

@nux Thank you! Based on the above, I have added PR #430. Hopefully this fixes it.

---

👤 **ciprianveg** commented the **2025-05-18** at **07:27:52**:<br>

cd ik_llama.cpp/
   git checkout disable_multi_add
   git fetch origin
   git checkout ik/disable_multi_add
   git pull origin ik/disable_multi_add
   cmake -B build -DGGML_CUDA=ON -DGGML_RPC=OFF -DGGML_BLAS=OFF -DGGML_SCHED_MAX_COPIES=1
   cmake --build ./build --config Release -j $(nproc)
   ./build/bin/llama-server --model /home/ciprian/ai/models/Qwen3-235B-UD_Q4_XL/Qwen3-235B-A22B-UD-Q4_K_XL-00001-of-00003.gguf --alias Qwen3-235B-A22B-UD-Q4_K_XL -fa -fmoe -ctk q8_0 -ctv q8_0 -c 20480  -ot "blk.(?:[x]|[5-9][0-9]).ffn.*=CPU" -ngl 99 --threads 16 --host 0.0.0.0 --port 5002   --no-mmap --ubatch-size 3072 --batch-size 3072 -ts 68,70,60,240 -v
same issue: (maybe it has something todo with the chat template considering the sweep-bench and cli are working fine?)

INFO [            update_slots] kv cache rm [p0, end) | tid="124177210875904" timestamp=1747553203 id_slot=0 id_task=0 p0=0
VERB [            update_slots] prompt processing progress | tid="124177210875904" timestamp=1747553203 id_slot=0 n_past=18 n_ctx=20480 n_tokens=18 progress=1.0
VERB [            update_slots] prompt done | tid="124177210875904" timestamp=1747553203 id_slot=0 n_past=18 n_ctx=20480 n_tokens=18
VERB [            update_slots] decoding batch | tid="124177210875904" timestamp=1747553203 n_tokens=18
CUDA error: an illegal memory access was encountered
  current device: 2, in function ggml_backend_cuda_synchronize at /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu:3067
  cudaStreamSynchronize(cuda_ctx->stream())
/home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu:110: CUDA error
Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
ptrace: Operation not permitted.

Same command works on llama.cpp

---

👤 **ciprianveg** commented the **2025-05-18** at **07:27:52**:<br>

1990  cd ik_llama.cpp/
 1991  git checkout disable_multi_add
 1992  git fetch origin
 1993  git checkout ik/disable_multi_add
 1994  git pull origin ik/disable_multi_add
 1996  history | grep cmake
 1997  cmake -B build -DGGML_CUDA=ON -DGGML_RPC=OFF -DGGML_BLAS=OFF -DGGML_SCHED_MAX_COPIES=1
 1998  cmake --build ./build --config Release -j $(nproc)
 1999  ./build/bin/llama-server --model /home/ciprian/ai/models/Qwen3-235B-UD_Q4_XL/Qwen3-235B-A22B-UD-Q4_K_XL-00001-of-00003.gguf --alias Qwen3-235B-A22B-UD-Q4_K_XL -fa -fmoe -ctk q8_0 -ctv q8_0 -c 20480  -ot "blk.(?:[x]|[5-9][0-9]).ffn.*=CPU" -ngl 99 --threads 16 --host 0.0.0.0 --port 5002   --no-mmap --ubatch-size 3072 --batch-size 3072 -ts 68,70,60,240 -v
same issue: (maybe it has something todo with the chat template considering the sweep-bench and cli are working fine?)

INFO [            update_slots] kv cache rm [p0, end) | tid="124177210875904" timestamp=1747553203 id_slot=0 id_task=0 p0=0
VERB [            update_slots] prompt processing progress | tid="124177210875904" timestamp=1747553203 id_slot=0 n_past=18 n_ctx=20480 n_tokens=18 progress=1.0
VERB [            update_slots] prompt done | tid="124177210875904" timestamp=1747553203 id_slot=0 n_past=18 n_ctx=20480 n_tokens=18
VERB [            update_slots] decoding batch | tid="124177210875904" timestamp=1747553203 n_tokens=18
CUDA error: an illegal memory access was encountered
  current device: 2, in function ggml_backend_cuda_synchronize at /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu:3067
  cudaStreamSynchronize(cuda_ctx->stream())
/home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu:110: CUDA error
Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
ptrace: Operation not permitted.

---

👤 **ikawrakow** commented the **2025-05-18** at **07:42:09**:<br>

@ciprianveg Thanks for testing. Are you willing to do a similar debugging session?
```
cmake --build ./build --config RelWithDebInfo -j $(nproc)
gsb --args ./build/bin/llama-server --model /home/ciprian/ai/models/Qwen3-235B-UD_Q4_XL/Qwen3-235B-A22B-UD-Q4_K_XL-00001-of-00003.gguf --alias Qwen3-235B-A22B-UD-Q4_K_XL -fa -fmoe -ctk q8_0 -ctv q8_0 -c 20480 -ot "blk.(?:[x]|[5-9][0-9]).ffn.*=CPU" -ngl 99 --threads 16 --host 0.0.0.0 --port 5002 --no-mmap --ubatch-size 3072 --batch-size 3072 -ts 68,70,60,240 -v
```
When it crashes, `type backtrace` and post the output.

---

👤 **ciprianveg** commented the **2025-05-18** at **08:00:14**:<br>

sure:
VERB [            update_slots] decoding batch | tid="140737203113984" timestamp=1747555159 n_tokens=18
CUDA error: an illegal memory access was encountered
  current device: 2, in function ggml_backend_cuda_synchronize at /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu:3067
  cudaStreamSynchronize(cuda_ctx->stream())
/home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu:110: CUDA error
[Detaching after fork from child process 59562]
Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
warning: process 59376 is already traced by process 59323
ptrace: Operation not permitted.
No stack.
The program is not being run.

Thread 1 "llama-server" received signal SIGABRT, Aborted.
Download failed: Invalid argument.  Continuing without source file ./nptl/./nptl/pthread_kill.c.
__pthread_kill_implementation (no_tid=0, signo=6, threadid=<optimized out>) at ./nptl/pthread_kill.c:44
warning: 44	./nptl/pthread_kill.c: No such file or directory
(gdb) backtrace
#0  __pthread_kill_implementation (no_tid=0, signo=6, threadid=<optimized out>) at ./nptl/pthread_kill.c:44
#1  __pthread_kill_internal (signo=6, threadid=<optimized out>) at ./nptl/pthread_kill.c:78
#2  __GI___pthread_kill (threadid=<optimized out>, signo=signo@entry=6) at ./nptl/pthread_kill.c:89
#3  0x00007fffee84527e in __GI_raise (sig=sig@entry=6) at ../sysdeps/posix/raise.c:26
#4  0x00007fffee8288ff in __GI_abort () at ./stdlib/abort.c:79
#5  0x00007fffef0333a5 in ggml_abort (file=0x7fffefa4cfc0 "/home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu", line=110, fmt=0x7fffefa35a7c "CUDA error")
    at /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml.c:270
#6  0x00007fffef18ed67 in ggml_cuda_error (stmt=stmt@entry=0x7fffefa4d698 "cudaStreamSynchronize(cuda_ctx->stream())", func=func@entry=0x7fffefa35b77 "ggml_backend_cuda_synchronize", 
    file=file@entry=0x7fffefa4cfc0 "/home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu", line=line@entry=3067, msg=0x7fffee48ece8 "an illegal memory access was encountered")
    at /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu:110
#7  0x00007fffef18f8aa in ggml_backend_cuda_synchronize (backend=<optimized out>) at /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu:3067
#8  0x00007fffef0aeed8 in ggml_backend_sched_compute_splits (sched=0x55555655d7c0) at /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-backend.c:1837
#9  ggml_backend_sched_graph_compute_async (sched=0x55555655d7c0, graph=<optimized out>) at /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-backend.c:2043
#10 0x00007ffff7ea3803 in llama_graph_compute (n_threads=16, gf=0x7fdfa06fb030, lctx=...) at /home/ciprian/ai/ik_llama.cpp/src/llama.cpp:17688
#11 llama_decode_internal (batch_all=..., lctx=...) at /home/ciprian/ai/ik_llama.cpp/src/llama.cpp:17904
#12 llama_decode (ctx=0x55555b677230, batch=...) at /home/ciprian/ai/ik_llama.cpp/src/llama.cpp:22299
#13 0x0000555555608122 in server_context::update_slots (this=0x7fffffffccc0) at /home/ciprian/ai/ik_llama.cpp/examples/server/server.cpp:2355
#14 0x00005555555e235b in std::function<void ()>::operator()() const (this=0x7fffffffd8e0) at /usr/include/c++/13/bits/std_function.h:591
#15 server_queue::start_loop (this=this@entry=0x7fffffffd7f8) at /home/ciprian/ai/ik_llama.cpp/examples/server/server.cpp:501
#16 0x000055555557e3dc in main (argc=<optimized out>, argv=<optimized out>) at /home/ciprian/ai/ik_llama.cpp/examples/server/server.cpp:3509
(gdb)

---

👤 **ciprianveg** commented the **2025-05-18** at **08:01:05**:<br>

this is from ik/disable_multi_add branch

---

👤 **ikawrakow** commented the **2025-05-18** at **08:11:02**:<br>

OK, now
```
frame 8
```
and then I need to see as much as possible. 
```
p sched->n_splits
p i
p *ggml_backend_sched_split
p *input_backend
p *split_backend
p split_backend_id
p split->n_inputs
p j
p *input
p *input_backend

if j > 0
p *split->inputs[0]
p *split->inputs[1], etc., up to j
```

---

👤 **ciprianveg** commented the **2025-05-18** at **08:13:48**:<br>

(gdb) frame 8
#8  0x00007fffef0aeed8 in ggml_backend_sched_compute_splits (sched=0x55555655d7c0) at /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-backend.c:1837
1837	                        ggml_backend_synchronize(split_backend);
(gdb)

---

👤 **ikawrakow** commented the **2025-05-18** at **08:16:19**:<br>

And the second part with `p sched->n_splits` etc.?

---

👤 **ciprianveg** commented the **2025-05-18** at **08:20:09**:<br>

(gdb) p sched->n_splits
$1 = 93
(gdb) p i
$2 = 4
(gdb) p *ggml_backend_sched_split
No symbol "ggml_backend_sched_split" in current context.
(gdb) p *input_backend
$3 = {guid = 0x7ffff7be9cf0 <guid>, iface = {get_name = 0x7fffef0aaf50 <ggml_backend_cpu_name>, free = 0x7fffef0aa960 <ggml_backend_cpu_free>, 
    get_default_buffer_type = 0x7fffef0ac0d0 <ggml_backend_cpu_get_default_buffer_type>, set_tensor_async = 0x0, get_tensor_async = 0x0, cpy_tensor_async = 0x0, synchronize = 0x0, 
    graph_plan_create = 0x7fffef0aaa90 <ggml_backend_cpu_graph_plan_create>, graph_plan_free = 0x7fffef0aa940 <ggml_backend_cpu_graph_plan_free>, graph_plan_update = 0x0, 
    graph_plan_compute = 0x7fffef0aac40 <ggml_backend_cpu_graph_plan_compute>, graph_compute = 0x7fffef0aab80 <ggml_backend_cpu_graph_compute>, 
    supports_op = 0x7fffef0aaf20 <ggml_backend_cpu_supports_op>, supports_buft = 0x7fffef0ab000 <ggml_backend_cpu_supports_buft>, offload_op = 0x0, event_new = 0x0, event_free = 0x0, 
    event_record = 0x0, event_wait = 0x0, event_synchronize = 0x0}, context = 0x55555b06ff20}
(gdb) p *split_backend
$4 = {guid = 0x7ffff7be9d40 <ggml_backend_cuda_guid()::guid>, iface = {get_name = 0x7fffef18dd80 <ggml_backend_cuda_name(ggml_backend_t)>, 
    free = 0x7fffef18f6c0 <ggml_backend_cuda_free(ggml_backend_t)>, get_default_buffer_type = 0x7fffef191140 <ggml_backend_cuda_get_default_buffer_type(ggml_backend_t)>, 
    set_tensor_async = 0x7fffef191000 <ggml_backend_cuda_set_tensor_async(ggml_backend_t, ggml_tensor*, void const*, size_t, size_t)>, 
    get_tensor_async = 0x7fffef190ec0 <ggml_backend_cuda_get_tensor_async(ggml_backend_t, ggml_tensor const*, void*, size_t, size_t)>, 
    cpy_tensor_async = 0x7fffef18fa90 <ggml_backend_cuda_cpy_tensor_async(ggml_backend_t, ggml_backend_t, ggml_tensor const*, ggml_tensor*)>, 
    synchronize = 0x7fffef18f820 <ggml_backend_cuda_synchronize(ggml_backend_t)>, graph_plan_create = 0x0, graph_plan_free = 0x0, graph_plan_update = 0x0, graph_plan_compute = 0x0, 
    graph_compute = 0x7fffef19c550 <ggml_backend_cuda_graph_compute(ggml_backend_t, ggml_cgraph*)>, 
    supports_op = 0x7fffef190550 <ggml_backend_cuda_supports_op(ggml_backend_t, ggml_tensor const*)>, 
    supports_buft = 0x7fffef18e670 <ggml_backend_cuda_supports_buft(ggml_backend_t, ggml_backend_buffer_type_t)>, 
    offload_op = 0x7fffef18dd90 <ggml_backend_cuda_offload_op(ggml_backend_t, ggml_tensor const*)>, event_new = 0x7fffef18f610 <ggml_backend_cuda_event_new(ggml_backend_t)>, 
    event_free = 0x7fffef18f5c0 <ggml_backend_cuda_event_free(ggml_backend_event_t)>, event_record = 0x7fffef18f8e0 <ggml_backend_cuda_event_record(ggml_backend_event_t)>, 
    event_wait = 0x7fffef18f9a0 <ggml_backend_cuda_event_wait(ggml_backend_t, ggml_backend_event_t)>, 
    event_synchronize = 0x7fffef18f570 <ggml_backend_cuda_event_synchronize(ggml_backend_event_t)>}, context = 0x55555b658b40}
(gdb) p split_backend_id
$5 = 3
(gdb) p split->n_inputs
$6 = 3
(gdb) p j
$7 = 1
(gdb) p *input
$8 = {type = GGML_TYPE_I32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x555555b61530, ne = {18, 1, 1, 1}, nb = {4, 72, 72, 72}, op = GGML_OP_NONE, op_params = {0 <repeats 16 times>}, 
  flags = 1, grad = 0x0, src = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0, data = 0x7fde4dff6080, name = "inp_pos", '\000' <repeats 56 times>, 
  extra = 0x0}
(gdb) p *input_backend
$9 = {guid = 0x7ffff7be9cf0 <guid>, iface = {get_name = 0x7fffef0aaf50 <ggml_backend_cpu_name>, free = 0x7fffef0aa960 <ggml_backend_cpu_free>, 
    get_default_buffer_type = 0x7fffef0ac0d0 <ggml_backend_cpu_get_default_buffer_type>, set_tensor_async = 0x0, get_tensor_async = 0x0, cpy_tensor_async = 0x0, synchronize = 0x0, 
    graph_plan_create = 0x7fffef0aaa90 <ggml_backend_cpu_graph_plan_create>, graph_plan_free = 0x7fffef0aa940 <ggml_backend_cpu_graph_plan_free>, graph_plan_update = 0x0, 
    graph_plan_compute = 0x7fffef0aac40 <ggml_backend_cpu_graph_plan_compute>, graph_compute = 0x7fffef0aab80 <ggml_backend_cpu_graph_compute>, 
    supports_op = 0x7fffef0aaf20 <ggml_backend_cpu_supports_op>, supports_buft = 0x7fffef0ab000 <ggml_backend_cpu_supports_buft>, offload_op = 0x0, event_new = 0x0, event_free = 0x0, 
    event_record = 0x0, event_wait = 0x0, event_synchronize = 0x0}, context = 0x55555b06ff20}
(gdb) p *split->inputs[0]
$10 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x55555af7abc0, ne = {4096, 18, 1, 1}, nb = {4, 16384, 294912, 294912}, op = GGML_OP_ADD, op_params = {
    0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {0x7fdfa09c8420, 0x7fdfa09c5900, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0, 
  data = 0x7fded6124080, name = "l_out-42", '\000' <repeats 55 times>, extra = 0x0}
(gdb) p *split->inputs[1]
$11 = {type = GGML_TYPE_I32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x555555b61530, ne = {18, 1, 1, 1}, nb = {4, 72, 72, 72}, op = GGML_OP_NONE, op_params = {
    0 <repeats 16 times>}, flags = 1, grad = 0x0, src = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0, data = 0x7fde4dff6080, 
  name = "inp_pos", '\000' <repeats 56 times>, extra = 0x0}
(gdb) p *split->inputs[2]
$12 = {type = GGML_TYPE_F16, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x55555b65a540, ne = {256, 32, 1, 1}, nb = {2, 512, 16384, 16384}, op = GGML_OP_CPY, op_params = {
    0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {0x7fe3040ba310, 0x7fdfa08ff750, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0, 
  data = 0x7fdf42050080, name = "KQ_mask (copy)", '\000' <repeats 49 times>, extra = 0x0}
(gdb) p *split->inputs[3]
Cannot access memory at address 0x0
(gdb)

---

👤 **ikawrakow** commented the **2025-05-18** at **09:03:55**:<br>

Don't know. Thanks for helping.

It is attempting to copy the inputs for layer 43 to a GPU. They consist of the result of layer 42 (`l_out-42`), the input positions (`inp_pos`), and the KQ mask (`KQ_mask (copy)`). As `inp_pos` and `KQ_mask (copy)` have been successfully copied 42 times before, the issue cannot be with them, so it must be with the result of layer 42. It looks like `l_out-42` was computed on the CPU. It is a simple ADD operation, so the likelihood of something going wrong there is zero.

---

👤 **ciprianveg** commented the **2025-05-19** at **12:30:36**:<br>

Hello,  some feedback that might help: With 3 gpus it is working, and considering that is faster than llama.cpp with 4 gpus, it is a win for me. Just fyi, it is not the gpu, because i put all the 3 gpus combination among all my gpus, to be sure i do not have a deffective one and they worked. Maybe because the last pcie is at lower speed and lags behind the rest? and maybe in llama.cpp lower speed being achieved it is still speedy enough? 

Non related question, is there a downside to set a large u_batch, n_batch? setting  u_batch =3072, n_batch=3072 increased the pp speed from 80t/s (when they were set to 1024) to 180t/s

---

👤 **ikawrakow** commented the **2025-05-19** at **12:48:12**:<br>

> Non related question, is there a downside to set a large u_batch, n_batch? setting u_batch =3072, n_batch=3072 increased the pp speed from 80t/s (when they were set to 1024) to 180t/s

For MoE models there is no downside other than needing a larger CUDA compute buffer, so it is just a matter of having enough VRAM. If you do have enough VRAM, then try `-b 4096 -ub 4096`, this should give you another 10-20% boost in PP speed.

For dense models the performance starts going down at some point as you increase u-batch size. At what point it starts going down depends on the GPU. The default choice of batch=2048, u-batch=512 is nearly optimum for dense models.  

The reason MoE models are different from dense models are the experts. If you use a u-batch size of 512 with DeepSeek-V3/R1, there will be `512 * 8 = 4096` total experts activated, so each expert will have to process on average just `4096 / 256 = 16` rows. Matrix multiplications with just 16 rows are much slower than matrix multiplications with 512 rows. This is why for MoE models PP speed increases with u-batch size. But I wouldn't go beyond 4096 as there are likely bugs (Johannes just very recently fixed a bug in `llama.cpp` that showed up at u-batch = 8192, which is likely also present here. His fix is not directly transferable to `ik_llama.cpp` because of the different way the MoE matrix multiplications are computed here).

---

👤 **ikawrakow** commented the **2025-05-19** at **13:06:22**:<br>

> Hello, some feedback that might help: With 3 gpus it is working,

Thanks for letting me know.

I'm maybe grasping at straws here, but is it possible that your power supply cannot manage  when all 4 GPUs start getting driven really hard? There is also a report from another user that they need to disable the GPU driving their monitor to have `ik_llama.cpp` working (see [here](https://github.com/ikawrakow/ik_llama.cpp/pull/430#issuecomment-2889222797))

---

👤 **ikawrakow** commented the **2025-05-19** at **13:11:59**:<br>

Also related to `u-batch`: If you don't have enough VRAM to go to batch=u-batch=4096, but PP performance is important to you, you may keep one extra layer per GPU on the CPU so you can use the larger u-batch. This will slightly slow down TG, but the decrease in TG performance with fewer layers offloaded to the GPU is quite modest, so you may still prefer the increase in PP performance.

---

👤 **ciprianveg** commented the **2025-05-19** at **13:15:24**:<br>

> > Hello, some feedback that might help: With 3 gpus it is working,
> 
> Thanks for letting me know.
> 
> I'm maybe grasping at straws here, but is it possible that your power supply cannot manage when all 4 GPUs start getting driven really hard? There is also a report from another user that they need to disable the GPU driving their monitor to have `ik_llama.cpp` working (see [here](https://github.com/ikawrakow/ik_llama.cpp/pull/430#issuecomment-2889222797))

I don't think the power is the issue, nvidia-smi shows the power usage very low, like between 80-150w per card, I guess the gpus are waiting after the cpu..

---

👤 **Lissanro** commented the **2025-05-20** at **11:13:14**:<br>

I think I have the same issue, seems to happen periodically. I am using the following command:

```
/pkgs/ik_llama.cpp/build/bin/llama-server \
--model /mnt/neuro/models/DeepSeek-R1T-Chimera-256x21B-IQ4_K_R4-163840seq/DeepSeek-R1T-Chimera-256x21B-IQ4_K_R4-163840seq.gguf \
--ctx-size 81920 --n-gpu-layers 62 --tensor-split 25,23,26,26 -mla 3 -fa -ctk q8_0 -amb 1024 -fmoe \
-ot "blk\.3\.ffn_up_exps=CUDA0, blk\.3\.ffn_gate_exps=CUDA0" \
-ot "blk\.4\.ffn_up_exps=CUDA1, blk\.4\.ffn_gate_exps=CUDA1" \
-ot "blk\.5\.ffn_up_exps=CUDA2, blk\.5\.ffn_gate_exps=CUDA2" \
-ot "blk\.6\.ffn_up_exps=CUDA3, blk\.6\.ffn_gate_exps=CUDA3" \
-ot "ffn_down_exps=CPU, ffn_up_exps=CPU, gate_exps=CPU" \
--threads 64 --host 0.0.0.0 --port 5000
```

Few lines of log before the error and the error itself look very similar to this bug report:

```
INFO [      log_server_request] request | tid="139488642715648" timestamp=1747701084 remote_addr="127.0.0.1" remote_port=57838 status=200 method="POST" path="/completion" params={}
INFO [            update_slots] all slots are idle | tid="139972738117632" timestamp=1747701084
INFO [   launch_slot_with_task] slot is processing task | tid="139972738117632" timestamp=1747726885 id_slot=0 id_task=11339
INFO [            update_slots] kv cache rm [p0, end) | tid="139972738117632" timestamp=1747726886 id_slot=0 id_task=11339 p0=47064
CUDA error: an illegal memory access was encountered
  current device: 0, in function ggml_backend_cuda_synchronize at /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu:3067
  cudaStreamSynchronize(cuda_ctx->stream())
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu:110: CUDA error
```

I am using 4x3090 GPUs on EPYC 7763 with 1TB 3200MHz RAM. I am using 2880W server grade PSU to power the video cards and online UPS, and GPUs are stable in all other tasks, including passing overnight memtest_vulkan testing (which verifies VRAM integrity). In case additional debug information from my side could be of help, please let me know.

---

👤 **Lissanro** commented the **2025-05-20** at **11:13:14**:<br>

I think I have the same issue, seems to happen periodically. I am using the following command:

```
/pkgs/ik_llama.cpp/build/bin/llama-server \
--model /mnt/neuro/models/DeepSeek-R1T-Chimera-256x21B-IQ4_K_R4-163840seq/DeepSeek-R1T-Chimera-256x21B-IQ4_K_R4-163840seq.gguf \
--ctx-size 81920 --n-gpu-layers 62 --tensor-split 25,23,26,26 -mla 3 -fa -ctk q8_0 -amb 1024 -fmoe \
-ot "blk\.3\.ffn_up_exps=CUDA0, blk\.3\.ffn_gate_exps=CUDA0" \
-ot "blk\.4\.ffn_up_exps=CUDA1, blk\.4\.ffn_gate_exps=CUDA1" \
-ot "blk\.5\.ffn_up_exps=CUDA2, blk\.5\.ffn_gate_exps=CUDA2" \
-ot "blk\.6\.ffn_up_exps=CUDA3, blk\.6\.ffn_gate_exps=CUDA3" \
-ot "ffn_down_exps=CPU, ffn_up_exps=CPU, gate_exps=CPU" \
--threads 64 --host 0.0.0.0 --port 5000
```

Few lines of log before the error and the error itself look very similar to this bug report:

```
INFO [      log_server_request] request | tid="139488642715648" timestamp=1747701084 remote_addr="127.0.0.1" remote_port=57838 status=200 method="POST" path="/completion" params={}
INFO [            update_slots] all slots are idle | tid="139972738117632" timestamp=1747701084
INFO [   launch_slot_with_task] slot is processing task | tid="139972738117632" timestamp=1747726885 id_slot=0 id_task=11339
INFO [            update_slots] kv cache rm [p0, end) | tid="139972738117632" timestamp=1747726886 id_slot=0 id_task=11339 p0=47064
CUDA error: an illegal memory access was encountered
  current device: 0, in function ggml_backend_cuda_synchronize at /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu:3067
  cudaStreamSynchronize(cuda_ctx->stream())
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu:110: CUDA error
```

I am using 4x3090 GPUs on EPYC 7763 with 1TB 3200MHz RAM. I am using server grade PSU to power the video cards and online UPS, and GPUs are stable in all other tasks, including passing overnight memtest_vulkan testing (which verifies VRAM integrity). In case additional debug information from my side could be of help, please let me know.

---

👤 **ikawrakow** commented the **2025-05-20** at **14:26:32**:<br>

@Lissanro All the experts in this mode use `*_R4` quants? If so, why are you offloading them to the GPUs? The data will have to be copied back to the CPU to do the matrix multiplications.

To all participants: Does #438 help?

---

👤 **ikawrakow** commented the **2025-05-20** at **14:26:32**:<br>

@Lissanro All the experts in this mode use `*_R4` quants? If so, why are you offloading them to the GPUs? The data will have to be copied back to the CPU to do the matrix multiplications.

@all Does #438 help?

---

👤 **nux** commented the **2025-05-20** at **14:56:58**:<br>

Just rebuilt and tried and got the error:
May 20 09:47:03 red llama-swap[1412]: CUDA error: an illegal memory access was encountered
May 20 09:47:03 red llama-swap[1412]:   current device: 0, in function ggml_backend_cuda_synchronize at /home/nux/dev/ik_llama.cpp/ggml/src/ggml-cuda.cu:3073
May 20 09:47:03 red llama-swap[1412]:   cudaStreamSynchronize(cuda_ctx->stream())
May 20 09:47:03 red llama-swap[1412]: /home/nux/dev/ik_llama.cpp/ggml/src/ggml-cuda.cu:110: CUDA error

Hmmm interesting...I sent the prompt that caused my crash, but removed a single line of code from the prompt that had php regex. And it worked. The line was:
while ( preg_match('#<(s(?:cript|tyle))[^>]*>#i', $input, $match, PREG_OFFSET_CAPTURE, $offset) ) {

I sent another prompt with only the regex and it didn't crash....hmm

---

👤 **Panchovix** commented the **2025-05-20** at **14:57:15**:<br>

I will try to test ASAP, I'm on vacations so my time is a bit more limited to try it via ssh

---

👤 **ciprianveg** commented the **2025-05-20** at **15:21:17**:<br>

same error:

CUDA error: an illegal memory access was encountered
  current device: 2, in function ggml_backend_cuda_synchronize at /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu:3075
  cudaStreamSynchronize(cuda_ctx->stream())
/home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu:110: CUDA error
Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf

---

👤 **ikawrakow** commented the **2025-05-20** at **15:23:43**:<br>

OK, thanks. So #438 does not fix it.

---

👤 **ciprianveg** commented the **2025-05-20** at **16:07:36**:<br>

@ikawrakow  can it have something to do with not sanitizing the prompt? it would explain why in bench and cli it doesnt happen..
openwebui appends the "/no_prompt" and some tools. It is strange that I removed "\no_think" from the prompt and it didn't crash..  Cand be also related to exact prompt length and how it is split..

---

👤 **ikawrakow** commented the **2025-05-20** at **16:25:22**:<br>

@ciprianveg I don't know. The crash reports are inconsistent with any hypothesis that I had. And in my own testing I'm just not able to crash it. Some users have found workarounds. For some users it does not crash. I have no idea what it is.

---

👤 **ciprianveg** commented the **2025-05-20** at **16:54:27**:<br>

Workarounds other than limiting the no of gpus?

---

👤 **nux** commented the **2025-05-20** at **17:03:07**:<br>

I only have one GPU. If I put a single layer -ngl 1 on the gpu it will crash for me. https://github.com/ikawrakow/ik_llama.cpp/issues/425#issuecomment-2884657811

---

👤 **ikawrakow** commented the **2025-05-21** at **04:43:40**:<br>

> I only have one GPU. If I put a single layer -ngl 1 on the gpu it will crash for me. [#425 (comment)](https://github.com/ikawrakow/ik_llama.cpp/issues/425#issuecomment-2884657811)

This is what makes it even more confusing. Everybody else reporting a crash has more than one GPU. I have one GPU and can never make it fail. I almost always use partial offload as only toy models fit on my 16 GB GPU.

---

👤 **Lissanro** commented the **2025-05-21** at **05:24:08**:<br>

@ikawrakow 
> All the experts in this mode use *_R4 quants? If so, why are you offloading them to the GPUs? The data will have to be copied back to the CPU to do the matrix multiplications.

I am using -mla 3 mode but please let me know if am I doing something wrong? I first create a normal IQ4_K_M quant without _R4, then selectively repack to _R4 only tensors that I plan to keep on CPU using commands mentioned in this message (using --repack-pattern): https://github.com/ikawrakow/ik_llama.cpp/discussions/323#discussioncomment-12816641

```
~/pkgs/ik_llama.cpp/build/bin/llama-quantize --repack \
--repack-pattern "(^blk\.[7-9]|\d\d).ffn_(up|gate)_exps|ffn_down_exps" \
/path/to/IQ4_K_M.gguf \
/path/to/IQ4_K_M_R4.gguf \
IQ4_K_R4
```

I am getting rather low 35 tokens/s input processing though, used to be 50+, but I thought this is because of IQ quant. I saw mention of suggestion to increase -ub, but I could only set it to 640 at most (even 1024 seems to try to allocate almost 20 GB on each GPU, which would leave no room for 64K-80K context I need at q8_0 cache).

In the meantime, I will keep testing using the latest patch to see if the crash still occurs. Based on what others reported, seems to be the case so I will not be surprised if I get the crash again, but for me it not always happens, only after multiple messages. I never managed to crash it on the first try yet, which makes it hard to reproduce.

---

👤 **ikawrakow** commented the **2025-05-21** at **06:02:16**:<br>

Please use branch in PR #442 and post the CUDA call trace that will be printed when the application crashes.

---

👤 **ikawrakow** commented the **2025-05-21** at **06:18:06**:<br>

@Lissanro 

If you are observing such huge compute buffers, you most likely need to rebuild using `-DGGML_SCHED_MAX_COPIES=1`.

There was also PR #405, which changed the GPU offload policy. After that PR the fused experts operation that gets used when `-fmoe` is specified gets offloaded to the GPU for PP. This speeds up PP quite a bit especially, if you use a large value for u-batch. But the offloading will only happen if the tensors are not repacked. After rebuilding with `-DGGML_SCHED_MAX_COPIES=1` you can try using your not repacked model with `-b 4096 -ub 4096`. If you don't have enough VRAM, you can offload fewer tensors to the GPU. The larger u-batch will increase PP speed with a very modest impact on TG performance due to the fewer experts offloaded to the GPU. With experts ops offloaded to the GPU it is also better to offload all 3 types of experts (as opposed to pre-#405, where it was better to offload more layers of `ffn_up_exps` and `ffn_gate_exps`).

The downside of the above is that you will increase the probability for a crash. But if you use #442, this may help debug the issue.

---

👤 **Lissanro** commented the **2025-05-21** at **13:24:25**:<br>

@ikawrakow Thank you, I recompiled with `-DGGML_SCHED_MAX_COPIES=1` as you suggested and now can use `-b 4096 -ub 4096`, and I had room to add more tensors as well:

```
/pkgs/ik_llama.cpp/build/bin/llama-server \
--model /mnt/neuro/models/DeepSeek-R1T-Chimera-256x21B-IQ4_K-163840seq/DeepSeek-R1T-Chimera-256x21B-IQ4_K-163840seq.gguf \
--ctx-size 81920 --n-gpu-layers 62 --tensor-split 25,23,26,26 -mla 3 -fa -ctk q8_0 -amb 1024 -fmoe -b 4096 -ub 4096 \
-ot "blk\.3\.ffn_up_exps=CUDA0, blk\.3\.ffn_gate_exps=CUDA0, blk\.3\.ffn_down_exps=CUDA0" \
-ot "blk\.4\.ffn_up_exps=CUDA1, blk\.4\.ffn_gate_exps=CUDA1, blk\.4\.ffn_down_exps=CUDA1" \
-ot "blk\.5\.ffn_up_exps=CUDA2, blk\.5\.ffn_gate_exps=CUDA2, blk\.5\.ffn_down_exps=CUDA2" \
-ot "blk\.6\.ffn_up_exps=CUDA3, blk\.6\.ffn_gate_exps=CUDA3, blk\.6\.ffn_down_exps=CUDA3" \
-ot "ffn_down_exps=CPU, ffn_up_exps=CPU, gate_exps=CPU" \
--threads 64 --host 0.0.0.0 --port 5000
```

Now I am getting 100-105 tokens/s for input processing, with little impact on generation speed - which is excellent, given I often work with long context tasks and long prompts.

By the way, is my understanding correct that repacking no longer necessary, or is there still some benefit to repack CPU-only tensors as R4?

---

Unfortunately, the issue is still there (I have applied #442 for debugging):

```txt
CUDA error: an illegal memory access was encountered
  current device: 0, in function ggml_backend_cuda_synchronize at /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu:3085
  cudaStreamSynchronize(cuda_ctx->stream())
========================== CUDA trace: 5239365 previous calls
     5239364: function ggml_cuda_get_device, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
     5239363: function ggml_cuda_op_mul_mat, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1755
     5239362: function ggml_cuda_op_mul_mat_cublas, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1388
     5239361: function ggml_cuda_op_mul_mat_cublas, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1387
     5239360: function ggml_cuda_get_device, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
     5239359: function ggml_cuda_set_device, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
     5239358: function ggml_cuda_set_device, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
     5239357: function ggml_cuda_op_mul_mat, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1755
     5239356: function ggml_cuda_get_device, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
     5239355: function ggml_cuda_get_device, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
     5239354: function ggml_cuda_get_device, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
     5239353: function ggml_cuda_set_device, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
     5239352: function ggml_cuda_op_mul_mat, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1632
     5239351: function ggml_cuda_set_device, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
     5239350: function ggml_cuda_op_mul_mat, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1755
     5239349: function ggml_cuda_get_device, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
     5239348: function ggml_cuda_get_device, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
     5239347: function ggml_cuda_op_mul_mat, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1745
     5239346: function ggml_cuda_op_mul_mat, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1735
     5239345: function ggml_cuda_op_mul_mat, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1755
     5239344: function ggml_cuda_get_device, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
     5239343: function ggml_cuda_get_device, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
     5239342: function ggml_cuda_op_mul_mat, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1745
     5239341: function ggml_cuda_op_mul_mat, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1735
     5239340: function ggml_cuda_op_mul_mat, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1755
     5239339: function ggml_cuda_get_device, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
     5239338: function ggml_cuda_get_device, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
     5239337: function ggml_cuda_op_mul_mat, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1745
     5239336: function ggml_cuda_op_mul_mat, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1735
     5239335: function ggml_cuda_op_mul_mat, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1755
     5239334: function ggml_cuda_get_device, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
     5239333: function ggml_cuda_get_device, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
     5239332: function ggml_cuda_get_device, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu:122: CUDA error
```

As far as I can tell, probability of it happening is about the same as before. What I noticed though, it seems to never happen on the first try, usually when I try to regenerate a message, or maybe on the next message. It is also hard to reproduce - using exactly the same input prompt, sometimes I can regenerate messages all I want, sometimes it crashes on the second try.

For some reason, if I let it generate without thinking first, then try to force thinking by specifying `<think>` as the start of a reply, and then regenerate a message, it is very likely to crash (`<think>` by itself does not cause the crash, if AI's reply starts with it, and I then regenerate, then it does not crash usually regardless if the next message with or without thinking). Not sure yet if this is truly affects probability of the crash or just few coincidences, but I thought I mention this - I tried few times with different prompts and seems like generating first message without thinking, then with thinking, is the fastest way to trigger the bug.

Another observation, does not seem to depend on context length. Both short (less than 1K) and long (40K+) context seem to have about the same probability of the crash.

---

👤 **Lissanro** commented the **2025-05-21** at **13:24:25**:<br>

@ikawrakow Thank you, I recompiled with `-DGGML_SCHED_MAX_COPIES=1` as you suggested and now can use `-b 4096 -ub 4096`, and I had room to add more tensors as well:

```
/pkgs/ik_llama.cpp/build/bin/llama-server \
--model /mnt/neuro/models/DeepSeek-R1T-Chimera-256x21B-IQ4_K-163840seq/DeepSeek-R1T-Chimera-256x21B-IQ4_K-163840seq.gguf \
--ctx-size 81920 --n-gpu-layers 62 --tensor-split 25,23,26,26 -mla 3 -fa -ctk q8_0 -amb 1024 -fmoe -b 4096 -ub 4096 \
-ot "blk\.3\.ffn_up_exps=CUDA0, blk\.3\.ffn_gate_exps=CUDA0, blk\.3\.ffn_down_exps=CUDA0" \
-ot "blk\.4\.ffn_up_exps=CUDA1, blk\.4\.ffn_gate_exps=CUDA1, blk\.4\.ffn_down_exps=CUDA1" \
-ot "blk\.5\.ffn_up_exps=CUDA2, blk\.5\.ffn_gate_exps=CUDA2, blk\.5\.ffn_down_exps=CUDA2" \
-ot "blk\.6\.ffn_up_exps=CUDA3, blk\.6\.ffn_gate_exps=CUDA3, blk\.6\.ffn_down_exps=CUDA3" \
-ot "ffn_down_exps=CPU, ffn_up_exps=CPU, gate_exps=CPU" \
--threads 64 --host 0.0.0.0 --port 5000
```

Now I am getting 100-105 tokens/s for input processing, with little impact on generation speed - which is excellent, given I often work with long context tasks and long prompts.

By the way, is my understanding correct that repacking no longer necessary, or is there still some benefit to repack CPU-only tensors as R4?

---

Unfortunately, the issue is still there (I have applied #439 and #442):

```txt
CUDA error: an illegal memory access was encountered
  current device: 0, in function ggml_backend_cuda_synchronize at /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu:3085
  cudaStreamSynchronize(cuda_ctx->stream())
========================== CUDA trace: 5239365 previous calls
     5239364: function ggml_cuda_get_device, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
     5239363: function ggml_cuda_op_mul_mat, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1755
     5239362: function ggml_cuda_op_mul_mat_cublas, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1388
     5239361: function ggml_cuda_op_mul_mat_cublas, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1387
     5239360: function ggml_cuda_get_device, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
     5239359: function ggml_cuda_set_device, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
     5239358: function ggml_cuda_set_device, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
     5239357: function ggml_cuda_op_mul_mat, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1755
     5239356: function ggml_cuda_get_device, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
     5239355: function ggml_cuda_get_device, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
     5239354: function ggml_cuda_get_device, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
     5239353: function ggml_cuda_set_device, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
     5239352: function ggml_cuda_op_mul_mat, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1632
     5239351: function ggml_cuda_set_device, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
     5239350: function ggml_cuda_op_mul_mat, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1755
     5239349: function ggml_cuda_get_device, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
     5239348: function ggml_cuda_get_device, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
     5239347: function ggml_cuda_op_mul_mat, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1745
     5239346: function ggml_cuda_op_mul_mat, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1735
     5239345: function ggml_cuda_op_mul_mat, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1755
     5239344: function ggml_cuda_get_device, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
     5239343: function ggml_cuda_get_device, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
     5239342: function ggml_cuda_op_mul_mat, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1745
     5239341: function ggml_cuda_op_mul_mat, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1735
     5239340: function ggml_cuda_op_mul_mat, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1755
     5239339: function ggml_cuda_get_device, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
     5239338: function ggml_cuda_get_device, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
     5239337: function ggml_cuda_op_mul_mat, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1745
     5239336: function ggml_cuda_op_mul_mat, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1735
     5239335: function ggml_cuda_op_mul_mat, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1755
     5239334: function ggml_cuda_get_device, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
     5239333: function ggml_cuda_get_device, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
     5239332: function ggml_cuda_get_device, file /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu:122: CUDA error
```

As far as I can tell, probability of it happening is about the same as before. What I noticed though, it seems to never happen on the first try, usually when I try to regenerate a message, or maybe on the next message. It is also hard to reproduce - using exactly the same input prompt, sometimes I can regenerate messages all I want, sometimes it crashes on the second try.

For some reason, if I let it generate without thinking first, then try to force thinking by specifying "<think>" as the start of a reply, and then regenerate a message, it is very likely to crash ("<think>" by itself does not cause the crash, if AI's reply starts with it, and I then regenerate, then it does not crash usually regardless if the next message with or without thinking). Not sure yet if this is truly affects probability of the crash or just few coincidences, but I thought I mention this - I tried few times with different prompts and seems like generating first message without thinking, then with thinking, is the fastest way to trigger the bug.

Another observation, does not seem to depend on context length. Both short (less than 1K) and long (40K+) context seem to have about the same probability of the crash.

---

👤 **ikawrakow** commented the **2025-05-21** at **13:48:35**:<br>

> By the way, is my understanding correct that repacking no longer necessary, or is there still some benefit to repack CPU-only tensors as R4?

It depends where the matrix multiplications for PP are done (TG is always done where the tensors are, but for TG there is little benefit from repacking). If they are done on CUDA, then don't repack. If they are left to run on the CPU, then repack. One example where I think not offloading the experts multiplications to CUDA would be beneficial is LlaMA-4 Maverick. This model has 128 experts, but only one is active. Hence, offloading to the GPU is likely to be slower than just running on the CPU. For the DeepSeek and Qwen3 MoE models for large batches it is better to offload to the GPU. But if your workflow is such that the prompts are not very long (so the large u-batch is not actually used), it would be faster to not offload and compute on the CPU, so in that case it would be useful to repack. Complicated, I know.

---

👤 **ikawrakow** commented the **2025-05-21** at **15:15:52**:<br>

I have added a trace to synchronize calls in the ggml-backend to #442 if someone wants to try.

---

👤 **ciprianveg** commented the **2025-05-21** at **15:58:44**:<br>

Hi @ikawrakow, here it is:

CUDA error: an illegal memory access was encountered
  current device: 2, in function ggml_backend_sched_compute_splits at /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-backend.c:1820
  cudaStreamSynchronize
========================== CUDA trace: 346129 previous calls
      346128: function ggml_backend_cuda_cpy_tensor_async, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 3074
      346127: function ggml_backend_cuda_cpy_tensor_async, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 3071
      346126: function ggml_backend_cuda_cpy_tensor_async, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 3061
      346125: function ggml_backend_sched_compute_splits, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-backend.c, line 1828
      346124: function ggml_cuda_up_gate_unary, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2773
      346123: function ggml_cuda_up_gate_unary, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2764
      346122: function ggml_cuda_op_mul_mat, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1755
      346121: function ggml_cuda_op_mul_mat_vec_q, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda/mmvq.cu, line 593
      346120: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      346119: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      346118: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      346117: function ggml_cuda_set_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      346116: function ggml_cuda_op_mul_mat, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1632
      346115: function ggml_cuda_set_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      346114: function ggml_cuda_up_gate_unary, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2743
      346113: function ggml_cuda_up_gate_unary, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2739
      346112: function ggml_cuda_op_mul_mat, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1755
      346111: function ggml_cuda_op_mul_mat_vec_q, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda/mmvq.cu, line 593
      346110: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      346109: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      346108: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      346107: function ggml_cuda_set_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      346106: function ggml_cuda_op_mul_mat, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1632
      346105: function ggml_cuda_set_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      346104: function ggml_cuda_up_gate_unary, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2735
      346103: function ggml_cuda_op_mul_mat, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1755
      346102: function ggml_cuda_op_mul_mat_vec_q, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda/mmvq.cu, line 593
      346101: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      346100: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      346099: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      346098: function ggml_cuda_set_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      346097: function ggml_cuda_op_mul_mat, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1632
      346096: function ggml_backend_cuda_cpy_tensor_async, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 3074
/home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu:122: CUDA error

---

👤 **maxious** commented the **2025-05-21** at **15:59:52**:<br>

same here
```
CUDA error: an illegal memory access was encountered
  current device: 0, in function ggml_backend_sched_compute_splits at /home/maxious/ik_llama.cpp/ggml/src/ggml-backend.c:1820
  cudaStreamSynchronize
========================== CUDA trace: 652627 previous calls
      652626: function ggml_backend_cuda_cpy_tensor_async, file /home/maxious/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 3074
      652625: function ggml_backend_cuda_cpy_tensor_async, file /home/maxious/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 3071
      652624: function ggml_backend_cuda_cpy_tensor_async, file /home/maxious/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 3061
      652623: function ggml_backend_sched_compute_splits, file /home/maxious/ik_llama.cpp/ggml/src/ggml-backend.c, line 1828
      652622: function ggml_cuda_up_gate_unary, file /home/maxious/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2773
      652621: function ggml_cuda_up_gate_unary, file /home/maxious/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2764
      652620: function ggml_cuda_op_mul_mat, file /home/maxious/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1755
      652619: function ggml_cuda_op_mul_mat_vec_q, file /home/maxious/ik_llama.cpp/ggml/src/ggml-cuda/mmvq.cu, line 593
      652618: function ggml_cuda_get_device, file /home/maxious/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      652617: function ggml_cuda_get_device, file /home/maxious/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      652616: function ggml_cuda_get_device, file /home/maxious/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      652615: function ggml_cuda_set_device, file /home/maxious/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      652614: function ggml_cuda_op_mul_mat, file /home/maxious/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1632
      652613: function ggml_cuda_set_device, file /home/maxious/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      652612: function ggml_cuda_up_gate_unary, file /home/maxious/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2743
      652611: function ggml_cuda_up_gate_unary, file /home/maxious/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2739
      652610: function ggml_cuda_op_mul_mat, file /home/maxious/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1755
      652609: function ggml_cuda_op_mul_mat_vec_q, file /home/maxious/ik_llama.cpp/ggml/src/ggml-cuda/mmvq.cu, line 593
      652608: function ggml_cuda_get_device, file /home/maxious/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      652607: function ggml_cuda_get_device, file /home/maxious/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      652606: function ggml_cuda_get_device, file /home/maxious/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      652605: function ggml_cuda_set_device, file /home/maxious/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      652604: function ggml_cuda_op_mul_mat, file /home/maxious/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1632
      652603: function ggml_cuda_set_device, file /home/maxious/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      652602: function ggml_cuda_up_gate_unary, file /home/maxious/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2735
      652601: function ggml_cuda_op_mul_mat, file /home/maxious/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1755
      652600: function ggml_cuda_op_mul_mat_vec_q, file /home/maxious/ik_llama.cpp/ggml/src/ggml-cuda/mmvq.cu, line 593
      652599: function ggml_cuda_get_device, file /home/maxious/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      652598: function ggml_cuda_get_device, file /home/maxious/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      652597: function ggml_cuda_get_device, file /home/maxious/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      652596: function ggml_cuda_set_device, file /home/maxious/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      652595: function ggml_cuda_op_mul_mat, file /home/maxious/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1632
      652594: function ggml_backend_cuda_cpy_tensor_async, file /home/maxious/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 3074
/home/maxious/ik_llama.cpp/ggml/src/ggml-cuda.cu:122: CUDA error
========================== CUDA trace: 652627 previous calls

```

---

👤 **ikawrakow** commented the **2025-05-21** at **17:04:08**:<br>

In both of these data is copied from one device to another. Then the back-end attempts to synchronize before copying the next tensor, and that's where it crashes.

I cannot figure out out also from this.

I could try printf debugging (will flood your terminals with printouts), but it is getting late here, so tomorrow.

---

👤 **ciprianveg** commented the **2025-05-21** at **18:17:31**:<br>

do these suggestions make sense or are hallucinations: https://chat.qwen.ai/s/b35fc22c-a36c-4b50-a296-6058ba15f313?fev=0.0.95

---

👤 **ikawrakow** commented the **2025-05-22** at **06:45:01**:<br>

If you are not tired of testing, there are new changes on #442

---

👤 **ciprianveg** commented the **2025-05-22** at **07:06:53**:<br>

Hi @ikawrakow, this is the log:
ggml_backend_cuda_synchronize: curent device is 3, context device is 0
ggml_backend_cuda_synchronize: reverting device to 3
ggml_backend_cuda_synchronize: curent device is 3, context device is 1
ggml_backend_cuda_synchronize: reverting device to 3
ggml_backend_cuda_synchronize: curent device is 3, context device is 2
ggml_backend_cuda_synchronize: reverting device to 3
ggml_backend_cuda_synchronize: curent device is 3, context device is 0
ggml_backend_cuda_synchronize: reverting device to 3
ggml_backend_cuda_synchronize: curent device is 3, context device is 0
ggml_backend_cuda_synchronize: reverting device to 3
ggml_backend_cuda_synchronize: curent device is 0, context device is 1
ggml_backend_cuda_synchronize: reverting device to 0
ggml_backend_cuda_cpy_tensor_async: attempt to copy from device 0 to device 1 without access enabled
ggml_backend_cuda_synchronize: curent device is 0, context device is 1
ggml_backend_cuda_synchronize: reverting device to 0
ggml_backend_cuda_synchronize: curent device is 0, context device is 1
ggml_backend_cuda_synchronize: reverting device to 0
ggml_backend_cuda_cpy_tensor_async: attempt to copy on device 0 while current device is 1
ggml_backend_cuda_cpy_tensor_async: attempt to copy from device 0 to device 1 without access enabled
ggml_backend_cuda_synchronize: curent device is 0, context device is 1
ggml_backend_cuda_synchronize: reverting device to 0
ggml_backend_cuda_synchronize: curent device is 1, context device is 2
ggml_backend_cuda_synchronize: reverting device to 1
ggml_backend_cuda_cpy_tensor_async: attempt to copy from device 1 to device 2 without access enabled
ggml_backend_cuda_synchronize: curent device is 1, context device is 2
ggml_backend_cuda_synchronize: reverting device to 1
ggml_backend_cuda_synchronize: curent device is 1, context device is 2
ggml_backend_cuda_synchronize: reverting device to 1
ggml_backend_cuda_cpy_tensor_async: attempt to copy on device 0 while current device is 2
ggml_backend_cuda_cpy_tensor_async: attempt to copy from device 0 to device 2 without access enabled
ggml_backend_cuda_synchronize: curent device is 0, context device is 2
ggml_backend_cuda_synchronize: reverting device to 0
ggml_backend_cuda_synchronize: curent device is 2, context device is 3
ggml_backend_cuda_synchronize: reverting device to 2
ggml_backend_cuda_cpy_tensor_async: attempt to copy from device 2 to device 3 without access enabled
CUDA error: an illegal memory access was encountered
  current device: 2, in function ggml_backend_sched_compute_splits at /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-backend.c:1835
  cudaStreamSynchronize
========================== CUDA trace: 347020 previous calls
      347019: function ggml_backend_cuda_cpy_tensor_async, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 3070
      347018: function ggml_backend_cuda_cpy_tensor_async, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 3055
      347017: function ggml_backend_cuda_synchronize, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 3120
      347016: function ggml_backend_sched_compute_splits, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-backend.c, line 1828
      347015: function ggml_backend_cuda_synchronize, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 3107
      347014: function ggml_cuda_up_gate_unary, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2774
      347013: function ggml_cuda_up_gate_unary, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2765
      347012: function ggml_cuda_op_mul_mat, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1756
      347011: function ggml_cuda_op_mul_mat_vec_q, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda/mmvq.cu, line 593
      347010: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      347009: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      347008: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      347007: function ggml_cuda_set_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      347006: function ggml_cuda_op_mul_mat, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1632
      347005: function ggml_cuda_set_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      347004: function ggml_cuda_up_gate_unary, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2744
      347003: function ggml_cuda_up_gate_unary, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2740
      347002: function ggml_cuda_op_mul_mat, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1756
      347001: function ggml_cuda_op_mul_mat_vec_q, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda/mmvq.cu, line 593
      347000: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      346999: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      346998: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      346997: function ggml_cuda_set_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      346996: function ggml_cuda_op_mul_mat, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1632
      346995: function ggml_cuda_set_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      346994: function ggml_cuda_up_gate_unary, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2736
      346993: function ggml_cuda_op_mul_mat, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1756
      346992: function ggml_cuda_op_mul_mat_vec_q, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda/mmvq.cu, line 593
      346991: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      346990: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      346989: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      346988: function ggml_cuda_set_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      346987: function ggml_backend_cuda_cpy_tensor_async, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 3070
/home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu:122: CUDA error
Could not attach to process.  If your uid matches the uid of the target

---

👤 **ikawrakow** commented the **2025-05-22** at **07:15:50**:<br>

Thanks!

What if you build with `-DGGML_CUDA_NO_PEER_COPY=1` ?

---

👤 **ciprianveg** commented the **2025-05-22** at **07:31:23**:<br>

i built it like this:
cmake -B build -DGGML_CUDA=ON -DGGML_RPC=OFF -DGGML_BLAS=OFF  -DGGML_SCHED_MAX_COPIES=1 -DGGML_CUDA_NO_PEER_COPY=1
but my command fails now:
./build/bin/llama-server --model /home/ciprian/ai/models/Qwen3-235B-UD_Q4_XL/Qwen3-235B-A22B-UD-Q4_K_XL-00001-of-00003.gguf --alias Qwen3-235B-A22B-UD-Q4_K_XL -fa -fmoe -ctk q8_0 -ctv q8_0 -c 10000 -ot "blk.(?:[x]|[5-9][0-9]).ffn.*=CPU" -ngl 99 --threads 16 --host 0.0.0.0 --port 5002 -amb 1024 --no-mmap --ubatch-size 2048 --batch-size 2048 -ts 68,70,60,240
llm_load_tensors: offloaded 95/95 layers to GPU
llm_load_tensors:        CPU buffer size = 58072.69 MiB
llm_load_tensors:  CUDA_Host buffer size =   333.84 MiB
llm_load_tensors:      CUDA0 buffer size = 20349.23 MiB
llm_load_tensors:      CUDA1 buffer size = 20140.23 MiB
llm_load_tensors:      CUDA2 buffer size = 17379.92 MiB
llm_load_tensors:      CUDA3 buffer size = 11628.83 MiB
.............................................Segmentation fault (core dumped)

---

👤 **ikawrakow** commented the **2025-05-22** at **07:36:22**:<br>

OK, then discard `DGGML_CUDA_NO_PEER_COPY=1`. There was another peer to peer copy without a check, so pushed a new commit.

The thing I don't understand is how this can work in `llama.cpp` when I don't see anywhere peer to peer access being enabled.

---

👤 **ciprianveg** commented the **2025-05-22** at **07:44:58**:<br>

i build without: -DGGML_CUDA_NO_PEER_COPY=1 and i still get the loading seg fault(should i delete all build dir to start from 0?):
llm_load_tensors: offloaded 95/95 layers to GPU
llm_load_tensors:        CPU buffer size = 30168.47 MiB
llm_load_tensors:  CUDA_Host buffer size =   333.84 MiB
llm_load_tensors:      CUDA0 buffer size = 16901.02 MiB
llm_load_tensors:      CUDA1 buffer size = 20613.89 MiB
llm_load_tensors:      CUDA2 buffer size = 18553.33 MiB
llm_load_tensors:      CUDA3 buffer size = 12339.69 MiB
.............................../startQwen235Q3UDXL.sh: line 2: 18560 Segmentation fault      (core dumped) ./build/bin/llama-server --model /home/ciprian/ai/models/Qwen3-235B-UD_Q3_XL/Qwen3-235B-A22B-UD-Q3_K_XL-00001-of-00003.gguf --alias Qwen3-235B-A22B-UD-Q3_K_XL -fa -fmoe -ctk q8_0 -ctv q8_0 -c 16384 -ot "blk.(?:[x]|[6-8][0-9]).ffn.*=CPU" -ngl 99 --threads 16 --host 0.0.0.0 --port 5002 -amb 2048 --no-mmap --ubatch-size 2048 --batch-size 2048 -ts 21,26,24,56

---

👤 **ikawrakow** commented the **2025-05-22** at **08:13:56**:<br>

Are you using `ccache`? My experience with `ccache` is that it does get confused and does not always rebuild correctly.

If you don't have anything of value in the build folder, yes, just delete it and rebuild.

---

👤 **ikawrakow** commented the **2025-05-22** at **08:14:40**:<br>

Oh, and pull another time.

---

👤 **ciprianveg** commented the **2025-05-22** at **08:53:00**:<br>

@ikawrakow Done:
INFO [            update_slots] kv cache rm [p0, end) | tid="134731138850816" timestamp=1747903765 id_slot=0 id_task=0 p0=0
ggml_backend_cuda_synchronize: curent device is 3, context device is 0
ggml_backend_cuda_synchronize: reverting device to 3
ggml_backend_cuda_synchronize: curent device is 3, context device is 1
ggml_backend_cuda_synchronize: reverting device to 3
ggml_backend_cuda_synchronize: curent device is 3, context device is 2
ggml_backend_cuda_synchronize: reverting device to 3
ggml_backend_cuda_synchronize: curent device is 3, context device is 0
ggml_backend_cuda_synchronize: reverting device to 3
ggml_backend_cuda_synchronize: curent device is 3, context device is 0
ggml_backend_cuda_synchronize: reverting device to 3
ggml_backend_cuda_synchronize: curent device is 0, context device is 1
ggml_backend_cuda_synchronize: reverting device to 0
ggml_backend_cuda_cpy_tensor_async: attempt to copy from device 0 to device 1 without access enabled
ggml_backend_cuda_synchronize: curent device is 0, context device is 1
ggml_backend_cuda_synchronize: reverting device to 0
ggml_backend_cuda_buffer_cpy_tensor: attempt to copy from device 0 to device 1 without access enabled
ggml_backend_cuda_cpy_tensor_async: attempt to copy on device 0 while current device is 1
ggml_backend_cuda_cpy_tensor_async: attempt to copy from device 0 to device 1 without access enabled
ggml_backend_cuda_synchronize: curent device is 0, context device is 1
ggml_backend_cuda_synchronize: reverting device to 0
ggml_backend_cuda_buffer_cpy_tensor: attempt to copy from device 0 to device 1 without access enabled
ggml_backend_cuda_synchronize: curent device is 1, context device is 2
ggml_backend_cuda_synchronize: reverting device to 1
ggml_backend_cuda_cpy_tensor_async: attempt to copy from device 1 to device 2 without access enabled
ggml_backend_cuda_synchronize: curent device is 1, context device is 2
ggml_backend_cuda_synchronize: reverting device to 1
ggml_backend_cuda_buffer_cpy_tensor: attempt to copy from device 1 to device 2 without access enabled
ggml_backend_cuda_cpy_tensor_async: attempt to copy on device 0 while current device is 2
ggml_backend_cuda_cpy_tensor_async: attempt to copy from device 0 to device 2 without access enabled
ggml_backend_cuda_synchronize: curent device is 0, context device is 2
ggml_backend_cuda_synchronize: reverting device to 0
ggml_backend_cuda_buffer_cpy_tensor: attempt to copy from device 0 to device 2 without access enabled
ggml_backend_cuda_synchronize: curent device is 2, context device is 3
ggml_backend_cuda_synchronize: reverting device to 2
ggml_backend_cuda_cpy_tensor_async: attempt to copy from device 2 to device 3 without access enabled
CUDA error: an illegal memory access was encountered
  current device: 2, in function ggml_backend_sched_compute_splits at /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-backend.c:1835
  cudaStreamSynchronize
========================== CUDA trace: 347264 previous calls
      347263: function ggml_backend_cuda_cpy_tensor_async, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 3078
      347262: function ggml_backend_cuda_cpy_tensor_async, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 3063
      347261: function ggml_backend_cuda_synchronize, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 3132
      347260: function ggml_backend_sched_compute_splits, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-backend.c, line 1828
      347259: function ggml_backend_cuda_synchronize, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 3119
      347258: function ggml_cuda_up_gate_unary, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2782
      347257: function ggml_cuda_up_gate_unary, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2773
      347256: function ggml_cuda_op_mul_mat, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1764
      347255: function ggml_cuda_op_mul_mat_vec_q, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda/mmvq.cu, line 593
      347254: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      347253: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      347252: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      347251: function ggml_cuda_set_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      347250: function ggml_cuda_op_mul_mat, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1640
      347249: function ggml_cuda_set_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      347248: function ggml_cuda_up_gate_unary, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2752
      347247: function ggml_cuda_up_gate_unary, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2748
      347246: function ggml_cuda_op_mul_mat, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1764
      347245: function ggml_cuda_op_mul_mat_vec_q, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda/mmvq.cu, line 593
      347244: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      347243: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      347242: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      347241: function ggml_cuda_set_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      347240: function ggml_cuda_op_mul_mat, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1640
      347239: function ggml_cuda_set_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      347238: function ggml_cuda_up_gate_unary, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2744
      347237: function ggml_cuda_op_mul_mat, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1764
      347236: function ggml_cuda_op_mul_mat_vec_q, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda/mmvq.cu, line 593
      347235: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      347234: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      347233: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      347232: function ggml_cuda_set_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      347231: function ggml_backend_cuda_cpy_tensor_async, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 3078
/home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu:122: CUDA error
Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
ptrace: Operation not permitted.
No stack.
The program is not being run.
./startQwen235Q4UDXL.sh: line 2: 32862 Aborted                 (core dumped) ./build/bin/llama-server --model /home/ciprian/ai/models/Qwen3-235B-UD_Q4_XL/Qwen3-235B-A22B-UD-Q4_K_XL-00001-of-00003.gguf --alias Qwen3-235B-A22B-UD-Q4_K_XL -fa -fmoe -ctk q4_0 -ctv q4_0 -c 32768 -ot "blk.(?:[x]|[5-9][0-9]).ffn.*=CPU" -ngl 99 --threads 16 --host 0.0.0.0 --port 5002 -amb 1024 --no-mmap --ubatch-size 2048 --batch-size 2048 -ts 68,70,60,240


and also a load time i got a lot of logs, but it loaded ok:




llm_load_tensors: offloaded 95/95 layers to GPU
llm_load_tensors:        CPU buffer size = 58072.69 MiB
llm_load_tensors:  CUDA_Host buffer size =   333.84 MiB
llm_load_tensors:      CUDA0 buffer size = 20349.23 MiB
llm_load_tensors:      CUDA1 buffer size = 20140.23 MiB
llm_load_tensors:      CUDA2 buffer size = 17379.92 MiB
llm_load_tensors:      CUDA3 buffer size = 11628.83 MiB
.............................................Failed to enable peer access from 0 to 1: peer access is not supported between these two devicesFailed to enable peer access from 0 to 2: peer access is not supported between these two devicesFailed to enable peer access from 0 to 3: peer access is not supported between these two devices................Failed to enable peer access from 1 to 0: peer access is not supported between these two devicesFailed to enable peer access from 1 to 2: peer access is not supported between these two devicesFailed to enable peer access from 1 to 3: peer access is not supported between these two devices...............Failed to enable peer access from 2 to 0: peer access is not supported between these two devicesFailed to enable peer access from 2 to 1: peer access is not supported between these two devicesFailed to enable peer access from 2 to 3: peer access is not supported between these two devices..............Failed to enable peer access from 3 to 0: peer access is not supported between these two devicesFailed to enable peer access from 3 to 1: peer access is not supported between these two devicesFailed to enable peer access from 3 to 2: peer access is not supported between these two devices..........
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 2048
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 1024
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
Failed to enable peer access from 0 to 1: peer access is not supported between these two devicesFailed to enable peer access from 0 to 2: peer access is not supported between these two devicesFailed to enable peer access from 0 to 3: peer access is not supported between these two devicesFailed to enable peer access from 1 to 0: peer access is not supported between these two devicesFailed to enable peer access from 1 to 2: peer access is not supported between these two devicesFailed to enable peer access from 1 to 3: peer access is not supported between these two devicesFailed to enable peer access from 2 to 0: peer access is not supported between these two devicesFailed to enable peer access from 2 to 1: peer access is not supported between these two devicesFailed to enable peer access from 2 to 3: peer access is not supported between these two devicesFailed to enable peer access from 3 to 0: peer access is not supported between these two devicesFailed to enable peer access from 3 to 1: peer access is not supported between these two devicesFailed to enable peer access from 3 to 2: peer access is not supported between these two devicesllama_kv_cache_init:      CUDA0 KV buffer size =   270.00 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =   270.00 MiB
llama_kv_cache_init:      CUDA2 KV buffer size =   234.00 MiB
llama_kv_cache_init:      CUDA3 KV buffer size =   918.01 MiB
llama_new_context_with_model: KV self size  = 1692.00 MiB, K (q4_0):  846.00 MiB, V (q4_0):  846.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     1.16 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
ggml_cuda_host_malloc: failed to allocate 288.02 MiB of pinned memory: invalid argument
llama_new_context_with_model:      CUDA0 compute buffer size =  1027.02 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =   608.01 MiB
llama_new_context_with_model:      CUDA2 compute buffer size =   608.01 MiB
llama_new_context_with_model:      CUDA3 compute buffer size =  1251.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   288.02 MiB
llama_new_context_with_model: graph nodes  = 3672
llama_new_context_with_model: graph splits = 225
ggml_backend_cuda_synchronize: curent device is 3, context device is 0
ggml_backend_cuda_synchronize: reverting device to 3
ggml_backend_cuda_synchronize: curent device is 3, context device is 1
ggml_backend_cuda_synchronize: reverting device to 3
ggml_backend_cuda_synchronize: curent device is 3, context device is 2
ggml_backend_cuda_synchronize: reverting device to 3
ggml_backend_cuda_synchronize: curent device is 3, context device is 0
ggml_backend_cuda_synchronize: reverting device to 3
ggml_backend_cuda_synchronize: curent device is 3, context device is 0
ggml_backend_cuda_synchronize: reverting device to 3
ggml_backend_cuda_synchronize: curent device is 0, context device is 1
ggml_backend_cuda_synchronize: reverting device to 0
ggml_backend_cuda_cpy_tensor_async: attempt to copy from device 0 to device 1 without access enabled
ggml_backend_cuda_synchronize: curent device is 0, context device is 1
ggml_backend_cuda_synchronize: reverting device to 0
ggml_backend_cuda_buffer_cpy_tensor: attempt to copy from device 0 to device 1 without access enabled
ggml_backend_cuda_cpy_tensor_async: attempt to copy on device 0 while current device is 1
ggml_backend_cuda_cpy_tensor_async: attempt to copy from device 0 to device 1 without access enabled
ggml_backend_cuda_synchronize: curent device is 0, context device is 1
ggml_backend_cuda_synchronize: reverting device to 0
ggml_backend_cuda_buffer_cpy_tensor: attempt to copy from device 0 to device 1 without access enabled
ggml_backend_cuda_synchronize: curent device is 1, context device is 2
ggml_backend_cuda_synchronize: reverting device to 1
ggml_backend_cuda_cpy_tensor_async: attempt to copy from device 1 to device 2 without access enabled
ggml_backend_cuda_synchronize: curent device is 1, context device is 2
ggml_backend_cuda_synchronize: reverting device to 1
ggml_backend_cuda_buffer_cpy_tensor: attempt to copy from device 1 to device 2 without access enabled
ggml_backend_cuda_cpy_tensor_async: attempt to copy on device 0 while current device is 2
ggml_backend_cuda_cpy_tensor_async: attempt to copy from device 0 to device 2 without access enabled
ggml_backend_cuda_synchronize: curent device is 0, context device is 2
ggml_backend_cuda_synchronize: reverting device to 0
ggml_backend_cuda_buffer_cpy_tensor: attempt to copy from device 0 to device 2 without access enabled
ggml_backend_cuda_synchronize: curent device is 2, context device is 3
ggml_backend_cuda_synchronize: reverting device to 2
ggml_backend_cuda_cpy_tensor_async: attempt to copy from device 2 to device 3 without access enabled
ggml_backend_cuda_synchronize: curent device is 2, context device is 3
ggml_backend_cuda_synchronize: reverting device to 2
ggml_backend_cuda_buffer_cpy_tensor: attempt to copy from device 2 to device 3 without access enabled
ggml_backend_cuda_cpy_tensor_async: attempt to copy on device 0 while current device is 3
ggml_backend_cuda_cpy_tensor_async: attempt to copy from device 0 to device 3 without access enabled
ggml_backend_cuda_synchronize: curent device is 0, context device is 3
ggml_backend_cuda_synchronize: reverting device to 0
ggml_backend_cuda_buffer_cpy_tensor: attempt to copy from device 0 to device 3 without access enabled
ggml_backend_cuda_synchronize: curent device is 3, context device is 0
ggml_backend_cuda_synchronize: reverting device to 3
ggml_backend_cuda_synchronize: curent device is 3, context device is 0
ggml_backend_cuda_synchronize: reverting device to 3
ggml_backend_cuda_cpy_tensor_async: attempt to copy on device 3 while current device is 0
ggml_backend_cuda_cpy_tensor_async: attempt to copy from device 3 to device 0 without access enabled
ggml_backend_cuda_synchronize: curent device is 3, context device is 0
ggml_backend_cuda_synchronize: reverting device to 3
ggml_backend_cuda_buffer_cpy_tensor: attempt to copy from device 3 to device 0 without access enabled
ggml_backend_cuda_synchronize: curent device is 0, context device is 3
ggml_backend_cuda_synchronize: reverting device to 0
ggml_backend_cuda_cpy_tensor_async: attempt to copy from device 0 to device 3 without access enabled
ggml_backend_cuda_synchronize: curent device is 0, context device is 3
ggml_backend_cuda_synchronize: reverting device to 0
ggml_backend_cuda_buffer_cpy_tensor: attempt to copy from device 0 to device 3 without access enabled
ggml_backend_cuda_synchronize: curent device is 3, context device is 0
ggml_backend_cuda_synchronize: reverting device to 3
ggml_backend_cuda_synchronize: curent device is 3, context device is 0
ggml_backend_cuda_synchronize: reverting device to 3

---

👤 **ikawrakow** commented the **2025-05-22** at **09:41:44**:<br>

So, there is no peer-to-peer access for your devices?

OK, so then let's try to follow the other Qwen3 suggestion: use `cuda-memcheck your_server_command`

---

👤 **ciprianveg** commented the **2025-05-22** at **11:30:31**:<br>

it is a lot of output from compute-sanitizer:

ggml_backend_cuda_buffer_cpy_tensor: attempt to copy from device 0 to device 2 without access enabled
ggml_backend_cuda_synchronize: curent device is 2, context device is 3
ggml_backend_cuda_synchronize: reverting device to 2
ggml_backend_cuda_cpy_tensor_async: attempt to copy from device 2 to device 3 without access enabled
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (8,0,0) in block (779,0,0)
=========     Address 0x7f788a3f7a0c is out of bounds
=========     and is 50,701 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (9,0,0) in block (779,0,0)
=========     Address 0x7f788a3f7a0c is out of bounds
=========     and is 50,701 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (10,0,0) in block (779,0,0)
=========     Address 0x7f788a3f7a0c is out of bounds
=========     and is 50,701 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (11,0,0) in block (779,0,0)
=========     Address 0x7f788a3f7a0c is out of bounds
=========     and is 50,701 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (12,0,0) in block (779,0,0)
=========     Address 0x7f788a3f7a0e is out of bounds
=========     and is 50,703 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (13,0,0) in block (779,0,0)
=========     Address 0x7f788a3f7a0e is out of bounds
=========     and is 50,703 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (14,0,0) in block (779,0,0)
=========     Address 0x7f788a3f7a0e is out of bounds
=========     and is 50,703 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (15,0,0) in block (779,0,0)
=========     Address 0x7f788a3f7a0e is out of bounds
=========     and is 50,703 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (24,0,0) in block (779,0,0)
=========     Address 0x7f788a3f7a9c is out of bounds
=========     and is 50,845 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (25,0,0) in block (779,0,0)
=========     Address 0x7f788a3f7a9c is out of bounds
=========     and is 50,845 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (26,0,0) in block (779,0,0)
=========     Address 0x7f788a3f7a9c is out of bounds
=========     and is 50,845 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (27,0,0) in block (779,0,0)
=========     Address 0x7f788a3f7a9c is out of bounds
=========     and is 50,845 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (28,0,0) in block (779,0,0)
=========     Address 0x7f788a3f7a9e is out of bounds
=========     and is 50,847 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (29,0,0) in block (779,0,0)
=========     Address 0x7f788a3f7a9e is out of bounds
=========     and is 50,847 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (30,0,0) in block (779,0,0)
=========     Address 0x7f788a3f7a9e is out of bounds
=========     and is 50,847 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (31,0,0) in block (779,0,0)
=========     Address 0x7f788a3f7a9e is out of bounds
=========     and is 50,847 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (8,1,0) in block (779,0,0)
=========     Address 0x7f788a3f7b2c is out of bounds
=========     and is 50,989 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (9,1,0) in block (779,0,0)
=========     Address 0x7f788a3f7b2c is out of bounds
=========     and is 50,989 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (10,1,0) in block (779,0,0)
=========     Address 0x7f788a3f7b2c is out of bounds
=========     and is 50,989 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (11,1,0) in block (779,0,0)
=========     Address 0x7f788a3f7b2c is out of bounds
=========     and is 50,989 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (12,1,0) in block (779,0,0)
=========     Address 0x7f788a3f7b2e is out of bounds
=========     and is 50,991 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (13,1,0) in block (779,0,0)
=========     Address 0x7f788a3f7b2e is out of bounds
=========     and is 50,991 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (14,1,0) in block (779,0,0)
=========     Address 0x7f788a3f7b2e is out of bounds
=========     and is 50,991 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (15,1,0) in block (779,0,0)
=========     Address 0x7f788a3f7b2e is out of bounds
=========     and is 50,991 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (24,1,0) in block (779,0,0)
=========     Address 0x7f788a3f7bbc is out of bounds
=========     and is 51,133 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (25,1,0) in block (779,0,0)
=========     Address 0x7f788a3f7bbc is out of bounds
=========     and is 51,133 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (26,1,0) in block (779,0,0)
=========     Address 0x7f788a3f7bbc is out of bounds
=========     and is 51,133 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (27,1,0) in block (779,0,0)
=========     Address 0x7f788a3f7bbc is out of bounds
=========     and is 51,133 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (28,1,0) in block (779,0,0)
=========     Address 0x7f788a3f7bbe is out of bounds
=========     and is 51,135 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (29,1,0) in block (779,0,0)
=========     Address 0x7f788a3f7bbe is out of bounds
=========     and is 51,135 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (30,1,0) in block (779,0,0)
=========     Address 0x7f788a3f7bbe is out of bounds
=========     and is 51,135 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (31,1,0) in block (779,0,0)
=========     Address 0x7f788a3f7bbe is out of bounds
=========     and is 51,135 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (8,1,0) in block (776,0,0)
=========     Address 0x7f788a3f452c is out of bounds
=========     and is 37,165 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (9,1,0) in block (776,0,0)
=========     Address 0x7f788a3f452c is out of bounds
=========     and is 37,165 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (10,1,0) in block (776,0,0)
=========     Address 0x7f788a3f452c is out of bounds
=========     and is 37,165 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (11,1,0) in block (776,0,0)
=========     Address 0x7f788a3f452c is out of bounds
=========     and is 37,165 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (12,1,0) in block (776,0,0)
=========     Address 0x7f788a3f452e is out of bounds
=========     and is 37,167 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (13,1,0) in block (776,0,0)
=========     Address 0x7f788a3f452e is out of bounds
=========     and is 37,167 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (14,1,0) in block (776,0,0)
=========     Address 0x7f788a3f452e is out of bounds
=========     and is 37,167 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (15,1,0) in block (776,0,0)
=========     Address 0x7f788a3f452e is out of bounds
=========     and is 37,167 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (24,1,0) in block (776,0,0)
=========     Address 0x7f788a3f45bc is out of bounds
=========     and is 37,309 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (25,1,0) in block (776,0,0)
=========     Address 0x7f788a3f45bc is out of bounds
=========     and is 37,309 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (26,1,0) in block (776,0,0)
=========     Address 0x7f788a3f45bc is out of bounds
=========     and is 37,309 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (27,1,0) in block (776,0,0)
=========     Address 0x7f788a3f45bc is out of bounds
=========     and is 37,309 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (28,1,0) in block (776,0,0)
=========     Address 0x7f788a3f45be is out of bounds
=========     and is 37,311 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (29,1,0) in block (776,0,0)
=========     Address 0x7f788a3f45be is out of bounds
=========     and is 37,311 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (30,1,0) in block (776,0,0)
=========     Address 0x7f788a3f45be is out of bounds
=========     and is 37,311 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (31,1,0) in block (776,0,0)
=========     Address 0x7f788a3f45be is out of bounds
=========     and is 37,311 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (8,2,0) in block (776,0,0)
=========     Address 0x7f788a3f464c is out of bounds
=========     and is 37,453 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (9,2,0) in block (776,0,0)
=========     Address 0x7f788a3f464c is out of bounds
=========     and is 37,453 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (10,2,0) in block (776,0,0)
=========     Address 0x7f788a3f464c is out of bounds
=========     and is 37,453 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (11,2,0) in block (776,0,0)
=========     Address 0x7f788a3f464c is out of bounds
=========     and is 37,453 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (12,2,0) in block (776,0,0)
=========     Address 0x7f788a3f464e is out of bounds
=========     and is 37,455 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (13,2,0) in block (776,0,0)
=========     Address 0x7f788a3f464e is out of bounds
=========     and is 37,455 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (14,2,0) in block (776,0,0)
=========     Address 0x7f788a3f464e is out of bounds
=========     and is 37,455 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (15,2,0) in block (776,0,0)
=========     Address 0x7f788a3f464e is out of bounds
=========     and is 37,455 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (24,2,0) in block (776,0,0)
=========     Address 0x7f788a3f46dc is out of bounds
=========     and is 37,597 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (25,2,0) in block (776,0,0)
=========     Address 0x7f788a3f46dc is out of bounds
=========     and is 37,597 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (26,2,0) in block (776,0,0)
=========     Address 0x7f788a3f46dc is out of bounds
=========     and is 37,597 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (27,2,0) in block (776,0,0)
=========     Address 0x7f788a3f46dc is out of bounds
=========     and is 37,597 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (28,2,0) in block (776,0,0)
=========     Address 0x7f788a3f46de is out of bounds
=========     and is 37,599 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (29,2,0) in block (776,0,0)
=========     Address 0x7f788a3f46de is out of bounds
=========     and is 37,599 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (30,2,0) in block (776,0,0)
=========     Address 0x7f788a3f46de is out of bounds
=========     and is 37,599 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (31,2,0) in block (776,0,0)
=========     Address 0x7f788a3f46de is out of bounds
=========     and is 37,599 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (8,0,0) in block (776,0,0)
=========     Address 0x7f788a3f440c is out of bounds
=========     and is 36,877 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (9,0,0) in block (776,0,0)
=========     Address 0x7f788a3f440c is out of bounds
=========     and is 36,877 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (10,0,0) in block (776,0,0)
=========     Address 0x7f788a3f440c is out of bounds
=========     and is 36,877 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (11,0,0) in block (776,0,0)
=========     Address 0x7f788a3f440c is out of bounds
=========     and is 36,877 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (12,0,0) in block (776,0,0)
=========     Address 0x7f788a3f440e is out of bounds
=========     and is 36,879 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
========= Invalid __global__ read of size 2 bytes
=========     at void mul_mat_vec_q<(ggml_type)12, (int)2, (int)4>(const void *, const void *, float *, const char *, int, int, int, int, unsigned long, unsigned long, unsigned long, long)+0x540
=========     by thread (13,0,0) in block (776,0,0)
=========     Address 0x7f788a3f440e is out of bounds
=========     and is 36,879 bytes after the nearest allocation at 0x7f744c000000 of size 18,224,165,888 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame:  [0x2f285f] in libcuda.so.1
=========         Host Frame:  [0x13e88] in libcudart.so.12
=========         Host Frame: cudaLaunchKernel [0x79f87] in libcudart.so.12
=========         Host Frame: void mul_mat_vec_q_cuda_T<(ggml_type)12, 4>(void const*, void const*, float*, char const*, int, int, int, int, int, int, unsigned long, unsigned long, unsigned long, long, CUstream_st*) [0x1a08d2] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*) [0x203c6b] in libggml.so
=========         Host Frame: ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void*, long, long, long, long, ggml_type, CUstream_st*)) [0x2278e4] in libggml.so
=========         Host Frame: ggml_cuda_up_gate_unary(ggml_backend_cuda_context&, ggml_tensor*, ggml_tensor*) [0x230cf7] in libggml.so
=========         Host Frame: ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) [0x235688] in libggml.so
=========         Host Frame: ggml_backend_sched_graph_compute_async [0xc0743] in libggml.so
=========         Host Frame: llama_decode [0xa4391] in libllama.so
=========         Host Frame: server_context::update_slots() [0xc0ceb] in llama-server
=========         Host Frame: server_queue::start_loop() [0x9105c] in llama-server
=========         Host Frame: main [0x2cd94] in llama-server
========= 
CUDA error: unspecified launch failure
  current device: 2, in function ggml_backend_sched_compute_splits at /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-backend.c:1835
  cudaStreamSynchronize
========================== CUDA trace: 347264 previous calls
      347263: function ggml_backend_cuda_cpy_tensor_async, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 3078
      347262: function ggml_backend_cuda_cpy_tensor_async, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 3063
      347261: function ggml_backend_cuda_synchronize, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 3132
      347260: function ggml_backend_sched_compute_splits, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-backend.c, line 1828
      347259: function ggml_backend_cuda_synchronize, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 3119
      347258: function ggml_cuda_up_gate_unary, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2782
      347257: function ggml_cuda_up_gate_unary, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2773
      347256: function ggml_cuda_op_mul_mat, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1764
      347255: function ggml_cuda_op_mul_mat_vec_q, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda/mmvq.cu, line 593
      347254: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      347253: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      347252: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      347251: function ggml_cuda_set_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      347250: function ggml_cuda_op_mul_mat, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1640
      347249: function ggml_cuda_set_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      347248: function ggml_cuda_up_gate_unary, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2752
      347247: function ggml_cuda_up_gate_unary, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2748
      347246: function ggml_cuda_op_mul_mat, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1764
      347245: function ggml_cuda_op_mul_mat_vec_q, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda/mmvq.cu, line 593
      347244: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      347243: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      347242: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      347241: function ggml_cuda_set_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      347240: function ggml_cuda_op_mul_mat, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1640
      347239: function ggml_cuda_set_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      347238: function ggml_cuda_up_gate_unary, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2744
      347237: function ggml_cuda_op_mul_mat, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1764
      347236: function ggml_cuda_op_mul_mat_vec_q, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda/mmvq.cu, line 593
      347235: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      347234: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      347233: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      347232: function ggml_cuda_set_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      347231: function ggml_backend_cuda_cpy_tensor_async, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 3078
/home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu:122: CUDA error
Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
ptrace: Operation not permitted.
No stack.
The program is not being run.
========= Error: process didn't terminate successfully
========= Target application returned an error
========= ERROR SUMMARY: 1263 errors
========= ERROR SUMMARY: 1163 errors were not printed. Use --print-limit option to adjust the number of printed errors

---

👤 **ikawrakow** commented the **2025-05-22** at **12:31:43**:<br>

Thank you for this. You are using UD-Q4_K_XL ?

---

👤 **ciprianveg** commented the **2025-05-22** at **12:33:49**:<br>

Yes. Same thing happens also with UD-Q3_K_XL, in ik_llama only. Do you want me to test with another 235b model? A non UD one?

---

👤 **ikawrakow** commented the **2025-05-22** at **13:44:32**:<br>

So, the only hypothesis I can make is that somehow the tensor metadata for one of the tensors is incorrect (else we cannot get the out of bounds access reported by the sanitizer). That's why I asked for the model. In UD-XL the `ffn_down` experts are quantized with more bits than `ffn_up` and `ffn_gate` in the first few layers. If we somehow are using the metadata (quantization type, etc.) for such a tensor in later layers, than we can get the out-of-bounds access.

To confirm, I have pushed another change that checks for an error in  `ggml_cuda_up_gate_unary` and prints the tensor metadata.

---

👤 **ciprianveg** commented the **2025-05-22** at **14:06:23**:<br>

@ikawrakow, logs:


gml_backend_cuda_cpy_tensor_async: attempt to copy from device 0 to device 1 without access enabled
ggml_backend_cuda_buffer_cpy_tensor: attempt to copy from device 0 to device 1 without access enabled
ggml_backend_cuda_cpy_tensor_async: attempt to copy from device 0 to device 1 without access enabled
ggml_backend_cuda_buffer_cpy_tensor: attempt to copy from device 0 to device 1 without access enabled
ggml_backend_cuda_cpy_tensor_async: attempt to copy from device 1 to device 2 without access enabled
ggml_backend_cuda_buffer_cpy_tensor: attempt to copy from device 1 to device 2 without access enabled
ggml_backend_cuda_cpy_tensor_async: attempt to copy from device 0 to device 2 without access enabled
ggml_backend_cuda_buffer_cpy_tensor: attempt to copy from device 0 to device 2 without access enabled
CUDA error: an illegal memory access was encountered
  current device: 2, in function prepare_row_mappigs at /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu:2243
  cudaMemcpyAsync(ids_host.data(), ids_dev, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream)
========================== CUDA trace: 397764 previous calls
      397763: function ggml_cuda_mul_mat_id, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2288
      397762: function ggml_cuda_mul_mat_id, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2451
      397761: function ggml_cuda_op_mul_mat, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1764
      397760: function ggml_cuda_op_mul_mat_vec_q, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda/mmvq.cu, line 593
      397759: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      397758: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      397757: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      397756: function ggml_cuda_set_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      397755: function ggml_cuda_op_mul_mat, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1640
      397754: function ggml_cuda_set_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      397753: function ggml_cuda_mul_mat_id, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2423
      397752: function ggml_cuda_mul_mat_id, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2451
      397751: function ggml_cuda_op_mul_mat, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1764
      397750: function ggml_cuda_op_mul_mat_vec_q, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda/mmvq.cu, line 593
      397749: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      397748: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      397747: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      397746: function ggml_cuda_set_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      397745: function ggml_cuda_op_mul_mat, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1640
      397744: function ggml_cuda_set_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      397743: function ggml_cuda_mul_mat_id, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2423
      397742: function ggml_cuda_mul_mat_id, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2451
      397741: function ggml_cuda_op_mul_mat, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1764
      397740: function ggml_cuda_op_mul_mat_vec_q, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda/mmvq.cu, line 593
      397739: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      397738: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      397737: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      397736: function ggml_cuda_set_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      397735: function ggml_cuda_op_mul_mat, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1640
      397734: function ggml_cuda_set_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      397733: function ggml_cuda_mul_mat_id, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2423
      397732: function ggml_cuda_mul_mat_id, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2451
      397731: function ggml_cuda_mul_mat_id, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2288
/home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu:122: CUDA error
Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
ptrace: Operation not permitted.
No stack.
The program is not being run.
./../llama.cpp/startQwen235Q4UDXL.sh: line 1: 22738 Aborted                 (core dumped) ./build/bin/llama-server --model /home/ciprian/ai/models/Qwen3-235B-UD_Q4_XL/Qwen3-235B-A22B-UD-Q4_K_XL-00001-of-00003.gguf --alias Qwen3-235B-A22B-UD-Q4_K_XL -fa -ctk q4_0 -ctv q4_0 -c 40960 --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0 --presence-penalty 0.5 -ot "blk.(?:[x]|[5-9][0-9]).ffn.*=CPU" -ngl 99 --threads 16 --host 0.0.0.0 --port 5002 --no-mmap --ubatch-size 3072 --batch-size 3072 -ts 68,70,60,240 --main-gpu 0

---

👤 **ikawrakow** commented the **2025-05-22** at **14:11:23**:<br>

This is a new. What is different to the previous times?

---

👤 **ciprianveg** commented the **2025-05-22** at **14:22:44**:<br>

Just git pull and rebuilt..

---

👤 **ikawrakow** commented the **2025-05-22** at **14:24:29**:<br>

You left out `-fmoe`

---

👤 **ciprianveg** commented the **2025-05-22** at **14:45:09**:<br>

@ikawrakow, you are right:

ggml_backend_cuda_buffer_cpy_tensor: attempt to copy from device 0 to device 2 without access enabled
========================================== Error in ggml_cuda_up_gate_unary. Device = 2
Devices: 2, 2, 2, 2. Current: 2
src0_1: blk.42.ffn_up_exps.weight, q4_K, 4096 x 1536 x 128
src0_2: blk.42.ffn_gate_exps.weight, q4_K, 4096 x 1536 x 128
src1  : ffn_moe_weighted-42, f32, 4096 x 1 x 26
nb0_1 : 144 x 2304 x 3538944
nb0_2 : 144 x 2304 x 3538944
src0_n: blk.42.ffn_down_exps.weight, q4_K, 1536 x 4096 x 128
next  : ffn_moe_down-42, f32, 4096 x 8 x 26
nxt_nb: 144 x 864 x 3538944
next devices: 2, 2
/home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu:2825: Fatal error
Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
ptrace: Operation not permitted.
No stack.
The program is not being run.
./startQwen235Q4UDXL.sh: line 2: 24499 Aborted                 (core dumped) ./build/bin/llama-server --model /home/ciprian/ai/models/Qwen3-235B-UD_Q4_XL/Qwen3-235B-A22B-UD-Q4_K_XL-00001-of-00003.gguf --alias Qwen3-235B-A22B-UD-Q4_K_XL -fa -fmoe -ctk q4_0 -ctv q4_0 -c 40960 --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0 --presence-penalty 0.5 -ot "blk.(?:[x]|[5-9][0-9]).ffn.*=CPU" -ngl 99 --threads 16 --host 0.0.0.0 --port 5002 --no-mmap --ubatch-size 3072 --batch-size 3072 -ts 68,70,60,240 --main-gpu 0

---

👤 **ikawrakow** commented the **2025-05-22** at **15:04:05**:<br>

Are you tired of testing yet? I have pushed another change.

---

👤 **ikawrakow** commented the **2025-05-22** at **15:27:06**:<br>

Btw, with the regex you are using for the tensor overrides, the small `ffn` tensors (`ffn_gate_inp` and `ffn_norm`) remain on the CPU. This results in more graph splits. Testing with Qwen3-30B-A3B with a single RTX-4080, I get

* TG = 70.4 t/s using `-ot "blk\.[3-4][0-9].ffn_.*_exps=CPU"`. There are 38 graph splits
* TG = 66.7 t/s using `-ot "blk\.[3-4][0-9].ffn.*=CPU". There are 74 graph splits.

PP is the same.

---

👤 **ciprianveg** commented the **2025-05-22** at **15:32:27**:<br>

I will rebuild, change the regex and retest, in about an hour, i am out a bit..

On Thu, 22 May 2025, 18:27 Kawrakow, ***@***.***> wrote:

> *ikawrakow* left a comment (ikawrakow/ik_llama.cpp#425)
> <https://github.com/ikawrakow/ik_llama.cpp/issues/425#issuecomment-2901663601>
>
> Btw, with the regex you are using for the tensor overrides, the small ffn
> tensors (ffn_gate_inp and ffn_norm) remain on the CPU. This results in
> more graph splits. Testing with Qwen3-30B-A3B with a single RTX-4080, I get
>
>    - TG = 70.4 t/s using -ot "blk\.[3-4][0-9].ffn_.*_exps=CPU". There are
>    38 graph splits
>    - TG = 66.7 t/s using `-ot "blk.[3-4][0-9].ffn.*=CPU". There are 74
>    graph splits.
>
> PP is the same.
>
> —
> Reply to this email directly, view it on GitHub
> <https://github.com/ikawrakow/ik_llama.cpp/issues/425#issuecomment-2901663601>,
> or unsubscribe
> <https://github.com/notifications/unsubscribe-auth/AJTBYK4GGWBORL2H6XHYIA327XUGBAVCNFSM6AAAAAB5GG6KRWVHI2DSMVQWIX3LMV43OSLTON2WKQ3PNVWWK3TUHMZDSMBRGY3DGNRQGE>
> .
> You are receiving this because you were mentioned.Message ID:
> ***@***.***>
>

---

👤 **ciprianveg** commented the **2025-05-22** at **15:32:27**:<br>

I will change the regex and retest, in about an hour, i am out a bit..

On Thu, 22 May 2025, 18:27 Kawrakow, ***@***.***> wrote:

> *ikawrakow* left a comment (ikawrakow/ik_llama.cpp#425)
> <https://github.com/ikawrakow/ik_llama.cpp/issues/425#issuecomment-2901663601>
>
> Btw, with the regex you are using for the tensor overrides, the small ffn
> tensors (ffn_gate_inp and ffn_norm) remain on the CPU. This results in
> more graph splits. Testing with Qwen3-30B-A3B with a single RTX-4080, I get
>
>    - TG = 70.4 t/s using -ot "blk\.[3-4][0-9].ffn_.*_exps=CPU". There are
>    38 graph splits
>    - TG = 66.7 t/s using `-ot "blk.[3-4][0-9].ffn.*=CPU". There are 74
>    graph splits.
>
> PP is the same.
>
> —
> Reply to this email directly, view it on GitHub
> <https://github.com/ikawrakow/ik_llama.cpp/issues/425#issuecomment-2901663601>,
> or unsubscribe
> <https://github.com/notifications/unsubscribe-auth/AJTBYK4GGWBORL2H6XHYIA327XUGBAVCNFSM6AAAAAB5GG6KRWVHI2DSMVQWIX3LMV43OSLTON2WKQ3PNVWWK3TUHMZDSMBRGY3DGNRQGE>
> .
> You are receiving this because you were mentioned.Message ID:
> ***@***.***>
>

---

👤 **ciprianveg** commented the **2025-05-22** at **17:43:10**:<br>

Hi @ikawrakow, here it is:
ggml_backend_cuda_buffer_cpy_tensor: attempt to copy from device 0 to device 2 without access enabled
CUDA error: an illegal memory access was encountered
  current device: 2, in function ggml_cuda_up_gate_unary at /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu:2751
  cudaStreamSynchronize(stream)
========================== CUDA trace: 354700 previous calls
      354699: function ggml_cuda_up_gate_unary, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2750
      354698: function ggml_cuda_op_mul_mat, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1764
      354697: function ggml_cuda_op_mul_mat_vec_q, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda/mmvq.cu, line 593
      354696: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      354695: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      354694: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      354693: function ggml_cuda_set_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      354692: function ggml_cuda_op_mul_mat, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1640
      354691: function ggml_cuda_set_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      354690: function ggml_cuda_up_gate_unary, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2729
      354689: function ggml_cuda_up_gate_unary, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2791
      354688: function ggml_cuda_up_gate_unary, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2782
      354687: function ggml_cuda_up_gate_unary, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2781
      354686: function ggml_cuda_op_mul_mat, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1764
      354685: function ggml_cuda_op_mul_mat_vec_q, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda/mmvq.cu, line 593
      354684: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      354683: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      354682: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      354681: function ggml_cuda_set_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      354680: function ggml_cuda_op_mul_mat, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1640
      354679: function ggml_cuda_set_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      354678: function ggml_cuda_up_gate_unary, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2760
      354677: function ggml_cuda_up_gate_unary, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2759
      354676: function ggml_cuda_up_gate_unary, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2755
      354675: function ggml_cuda_op_mul_mat, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1764
      354674: function ggml_cuda_op_mul_mat_vec_q, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda/mmvq.cu, line 593
      354673: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      354672: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      354671: function ggml_cuda_get_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 140
      354670: function ggml_cuda_set_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      354669: function ggml_cuda_op_mul_mat, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 1640
      354668: function ggml_cuda_set_device, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 129
      354667: function ggml_cuda_up_gate_unary, file /home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu, line 2750
/home/ciprian/ai/ik_llama.cpp/ggml/src/ggml-cuda.cu:122: CUDA error
Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
ptrace: Operation not permitted.
No stack.
The program is not being run.
./startQwen235Q4UDXL.sh: line 2: 33332 Aborted                 (core dumped) ./build/bin/llama-server --model /home/ciprian/ai/models/Qwen3-235B-UD_Q4_XL/Qwen3-235B-A22B-UD-Q4_K_XL-00001-of-00003.gguf --alias Qwen3-235B-A22B-UD-Q4_K_XL -fa -fmoe -ctk q4_0 -ctv q4_0 -c 40960 --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0 --presence-penalty 0.5 -ot "blk.(?:[x]|[5-9][0-9]).ffn.*=CPU" -ngl 99 --threads 16 --host 0.0.0.0 --port 5002 --no-mmap --ubatch-size 3072 --batch-size 3072 -ts 68,70,60,240 --main-gpu 0

---

👤 **ciprianveg** commented the **2025-05-22** at **18:09:34**:<br>

and also thanks for the regex tip, i got a 6% increase in gen speed.

---

👤 **ikawrakow** commented the **2025-05-23** at **05:06:33**:<br>

Hopefully the last change fixes it...

There really was a bug showing up when 2 or 3 tokens are processed.

---

👤 **ciprianveg** commented the **2025-05-23** at **08:17:10**:<br>

I won't be able to test it till tomorrow evening..

---

👤 **Lissanro** commented the **2025-05-23** at **08:19:53**:<br>

I rebuilt from the latest git, and it crashed when regenarating reply by getting triggered the same way as before, so unfortunately seem to be no change on my end. However, for some strange reason applying #442 "fixes" the bug. Below I provide detailed debug info.

First, I generate reply without thinking, which works fine, then with the `<think>` tag, which crashes it; if I start generating first message with `<think>` then the bug usually does not trigger when I try to regenerate it. May be it has nothing to do with the thinking mode, but slightly bigger partial match in the cache when the next message regenerates, forcing slightly different timings? Just regenerating non-thinking replies or thinking replies may not trigger it at all, but so far, generating non-thinking then thinking reply triggers it in all of cases that I have tried regardless if prompt is less than 1K tokens or 40K+ tokens long. Since I tried relatively few times, I am not yet 100% sure if is the most reliable way to trigger it, but so far it does it for me:

```
CUDA error: an illegal memory access was encountered
  current device: 0, in function ggml_backend_cuda_synchronize at /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu:3074
  cudaStreamSynchronize(cuda_ctx->stream())
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml-cuda.cu:110: CUDA error
```

With #442 applied the bug does not trigger anymore (or becomes much less probable to happen), but I get a lot of warnings like both before I send my first prompt, and after:

```
ggml_backend_cuda_cpy_tensor_async: attempt to copy from device 0 to device 1 without access enabled
ggml_backend_cuda_buffer_cpy_tensor: attempt to copy from device 0 to device 1 without access enabled
```

Full log (most of repeated lines replaced with "..." since they look the same) after generating first reply: https://pastebin.com/8F1YNFyw

Second log after generating the second reply with the `<think>` tag, which usually triggers the bug without #442 applied: https://pastebin.com/VUgDKehw

My only guess, #442 changes timings somehow and workarounds the bug in most cases. Just to be sure, I tried rebuilding without the patch, and the bug is back again, very reproducible using the method described above, no matter the content of the prompt as far as I can tell.

Previously, I tried with older #442 version and the bug still could trigger (I shared the debug output here in the previous messages), so I guess updated version #442 started to work as a workaround.

Also, I wonder if it is supposed to attempt to copy from device to device without access enabled? Maybe fixing this warning could lead to an actual fix?

---

👤 **ikawrakow** commented the **2025-05-23** at **08:23:31**:<br>

The bug is fixed on #442, but only as of this morning European time.

It is not fixed on the main branch. I wanted to first have confirmation that the last change in #442 actually fixes it before making a fresh bug fix PR.

---

👤 **ikawrakow** commented the **2025-05-23** at **08:47:09**:<br>

> Also, I wonder if it is supposed to attempt to copy from device to device without access enabled? Maybe fixing this warning could lead to an actual fix?

So, this was a wisdom from Qwen3. But the only place in mainline `llama.cpp` where peer-to-peer access is explicitly enabled or disabled is when using split mode row, which is not the case here. Considering that mainline works, these checks are not required.

The bug was in the matrix-vector multiplication kernel. It only shows up when the number of rows being processed (i.e., tokens) is 2 or 3 (the matrix-vector kernel confusingly processes up to 8 rows). This is not used during TG, and only triggers if an expert ends up with 2 or 3 rows, which is rare. I think all other changes on #442 are not required. The reason it took me so long to find is my lack of GPU experience (and my laziness to actually read the CUDA API specification). I realized only yesterday that checking for an error after launching a CUDA kernel does not tell us that the kernel was successfully executed, but only tells us that the kernel was successfully **queued** for execution. If there is a bug in the kernel (e.g., illegal memory access), the resulting error will get reported in some later call. Hence we were observing the illegal memory access error in synchronization calls, which made me think that there was something wrong in the back-end, data copying between devices, etc.  So, most of what Qwen3 wrote were useless hallucinations. But at the end Qwen3 was actually useful, as the hallucinations were what made me go and read the CUDA programming guide.

---

👤 **Lissanro** commented the **2025-05-23** at **08:56:21**:<br>

> The bug is fixed on https://github.com/ikawrakow/ik_llama.cpp/pull/442, but only as of this morning European time.

I see, I guess I got confused by "CUDA call tracer #442" title, and did not pay enough attention to notice it also adds fixes, not just call traces. My apologies.

In order to confirm what fixed the bug, I rebuilt with only [Fix bug in MMVQ kernel](https://github.com/ikawrakow/ik_llama.cpp/pull/442/commits/b79be8a191c10883a84d725ae9e70ec693ab3b6b) applied, and the bug seems to be fixed as far as I can tell using just this one commit.