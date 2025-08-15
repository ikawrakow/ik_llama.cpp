### 📝 [#381](https://github.com/ikawrakow/ik_llama.cpp/issues/381) - ik_llama.cpp/ggml/src/ggml-cuda/fattn.cu:66: fatal error after latest

| **Author** | `nux` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-05 |
| **Updated** | 2025-05-05 |

---

#### Description

did git pull and tried llama-bench:
~/dev/ik_llama.cpp $ ./build/bin/llama-bench -m /mnt/nvme/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-IQ4_K_R4/DeepSeek-V3-0324-IQ4_K_R4-00001-of-00010.gguf -p 512 -t 32 -mla 2 -fa 1 -fmoe 1 -ngl 99 --override-tensor "exps=CPU" -amb 512 -ctk q8_0 -ctv q8_0
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
| model                          |       size |     params | backend    | ngl | type_k | type_v | fa | mla |   amb | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | -: | --: | ----: | ---: | ------------: | ---------------: |
/home/nux/dev/ik_llama.cpp/ggml/src/ggml-cuda/fattn.cu:66: fatal error
Aborted

Tried llama-server and it gave the response, but got the same error. Here is output from llama-swap logs/stream:
INFO [            update_slots] all slots are idle | tid="140042111852544" timestamp=1746415718
INFO [      log_server_request] request | tid="139593198325760" timestamp=1746415718 remote_addr="127.0.0.1" remote_port=37478 status=200 method="POST" path="/v1/chat/completions" params={}
INFO [            update_slots] all slots are idle | tid="140042111852544" timestamp=1746415718
[DEBUG] Process [deepseek-v3] request /v1/chat/completions - start: 11.260061323s, total: 1m10.727098281s
[INFO] Request 127.0.0.1 "POST /v1/chat/completions HTTP/1.1" 200 136633 "Python/3.11 aiohttp/3.11.11" 1m10.727207572s
[INFO] Request 127.0.0.1 "GET /v1/models HTTP/1.1" 200 597 "Python/3.11 aiohttp/3.11.11" 55.89µs
[DEBUG] No-swap, using existing process for model [deepseek-v3]
INFO [   launch_slot_with_task] slot is processing task | tid="140042111852544" timestamp=1746415718 id_slot=0 id_task=670
INFO [            update_slots] kv cache rm [p0, end) | tid="140042111852544" timestamp=1746415718 id_slot=0 id_task=670 p0=0
/home/nux/dev/ik_llama.cpp/ggml/src/ggml-cuda/fattn.cu:66: fatal error
[INFO] Request 127.0.0.1 "POST /v1/chat/completions HTTP/1.1" 502 54 "Python/3.11 aiohttp/3.11.11" 2.77018715s
[INFO] Request 127.0.0.1 "GET /v1/models HTTP/1.1" 200 597 "Python/3.11 aiohttp/3.11.11" 81.941µs
[DEBUG] No-swap, using existing process for model [deepseek-v3]
[INFO] Request 127.0.0.1 "POST /v1/chat/completions HTTP/1.1" 502 103 "Python/3.11 aiohttp/3.11.11" 273.281µs

Command to run server is: 

/home/nux/dev/ik_llama.cpp/build/bin/llama-server
      --model /mnt/nvme/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-IQ4_K_R4/DeepSeek-V3-0324-IQ4_K_R4-00001-of-00010.gguf
      --alias ubergarm/DeepSeek-R1-V3-0324-IQ4_K_R4
      --ctx-size 32768 -mla 2 -fa -amb 512 -fmoe --temp 0.3 -ctk q8_0
      --min-p 0.05 --n-gpu-layers 63 --override-tensor "exps=CPU"
      --parallel 1 --threads 32 --host 127.0.0.1 --port 8081


Using ubergarm/DeepSeek-V3-0324-GGUF with ik_llama.cpp. 
Using CPU with most of model in memory, with a 3090. Been using ubergarm/DeepSeek-V3-0324-GGUF for a while with no issues. 
Can give more info if needed. 
Did a llama-bench before git pull and rebuilding.
$ cat ../commands/ik_bench-dsv3.txt
./build/bin/llama-bench -m /mnt/nvme/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-IQ4_K_R4/DeepSeek-V3-0324-IQ4_K_R4-00001-of-00010.gguf -p 512 -t 32 -mla 2 -fa 1 -fmoe 1 -ngl 99 --override-tensor "exps=CPU" -amb 512

| model                          |       size |     params | backend    | ngl | fa | mla |   amb | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | --: | ----: | ---: | ------------: | ---------------: |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 386.18 GiB |   672.05 B | CUDA       |  99 |  1 |   2 |   512 |    1 |         pp512 |     78.93 ± 0.04 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 386.18 GiB |   672.05 B | CUDA       |  99 |  1 |   2 |   512 |    1 |         tg128 |      9.98 ± 0.06 |

build: 1ea1df4b (3659)


Built with
cmake -B build -DGGML_CUDA_FA_ALL_QUANTS=ON -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON

cmake --build build --config Release -j --clean-first

dmesg -T shows:
[Sun May  4 22:19:21 2025] llama-bench[2015]: segfault at 204803fe0 ip 00007efcfc1189d7 sp 00007ffe21474280 error 4 in libcuda.so.575.51.03[7efcfbdc5000+e97000] likely on CPU 21 (core 5, socket 1)
[Sun May  4 22:19:21 2025] Code: ef e8 9d c9 ca ff 83 3d 7e 57 2f 05 01 49 8b 1c 24 76 0a 8b 05 86 57 2f 05 85 c0 74 56 49 8b 44 24 10 41 8b 4c 24 24 48 8b 13 <8b> 00 41 39 c6 74 52 8b b3 40 40 00 00 48 89 f0 89 8c b3 44 40 00

Can give more info if needed. Tried to put this on reddit post but got "Server error. Try again later." Apologize if this is not correct spot for this.

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-05-05** at **05:43:02**:<br>

Thank you for the bug report. PR #370 broke it. Can you check if it works for you now? Thanks.

As a side note: The row-interleaved quants (`*_R4, *_R8`) are not ideal when running on the GPU as there is no CUDA support for them. The effect will be that all calculations will be run on the CPU, and your GPU will be acting as a very expensive RAM module. If you are using partial offload to the GPU, the better option is to use a model without row-interleaved quants, and to specify `-rtr` on the command line. In that case, the tensors that are not offloaded to the GPU will get run-time repacked to row-interleaved for better performance (but this will make model loading time longer).

---

👤 **nux** commented the **2025-05-05** at **06:55:21**:<br>

I rebuilt with the latest changes and it works

On that side note - I've stuck with ubergarm/DeepSeek-V3-0324-GGUF IQ4_K_R4 as it's worked. Would love to hear recommendation on what I should look into or direction I should go for a (dual epyc) 768GB ram 3090 setup. Still quite new to this.

Will consider bug report closed - thanks!

---

👤 **nux** commented the **2025-05-05** at **06:55:21**:<br>

I rebuilt with the latest changes and it works

On that side note - I've stuck with ubergarm/DeepSeek-V3-0324-GGUF IQ4_K_R4 as it's worked. Would love to hear recommendation on what I should look into or direction I should go for a 768GB ram 3090 setup. Still quite new to this.

Will consider bug report closed - thanks!

---

👤 **ikawrakow** commented the **2025-05-05** at **07:09:19**:<br>

If you are new to this and don't want to get involved with making your own quantized models, perhaps we should ask @ubergarm to publish his models without row interleaving so they can be run efficiently with with full/partial GPU offload.

---

👤 **ikawrakow** commented the **2025-05-05** at **07:20:54**:<br>

What you can try in the meantime is to see if you get better performance by running CPU-only.

Build the project without CUDA:
```
cmake -DGGML_CUDA=OFF other_cmake_args
```
and then run as you have done above but without the `-ngl 99` argument and using `-mla 3` instead of `-mla 2`.

---

👤 **nux** commented the **2025-05-05** at **14:49:30**:<br>

I will look into making my own quantized models

I do see this on (https://huggingface.co/ubergarm/DeepSeek-V3-0324-GGUF)

> So far these are my best recipes offering the lowest perplexity per GiB models suitable for a wide variety of CPU+GPU or CPU only rigs.
> IQ4_K_R4 4.936 BPW
> 
> Special mix IQ5_K_R4/IQ4_K_R4 routed experts with all other layers full q8_0 for CPU+GPU offload or --run-time-repack for max speed CPU only rigs. Great for big 384+ GB RAM rig with 24GB+ GPU

Did something change or a misunderstanding somewhere? 

Thanks!

---

👤 **ikawrakow** commented the **2025-05-05** at **15:04:10**:<br>

> Did something change or a misunderstanding somewhere?

Oh, I see these have all attention tensors quantized with `Q8_0`. Sorry, didn't pay attention. Yes, these are good for hybrid CPU/GPU inference the way you are running it.

---

👤 **ubergarm** commented the **2025-05-05** at **15:21:39**:<br>

Thanks, yeah going forward I've started to release non-repacked quants as a lot of multi-gpu people were complaining. Then folks who want can offline-repack themselves which seems a bit more flexible for general audience.