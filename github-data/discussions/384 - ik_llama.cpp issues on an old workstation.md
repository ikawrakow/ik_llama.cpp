### 🗣️ [#384](https://github.com/ikawrakow/ik_llama.cpp/discussions/384) - ik_llama.cpp issues on an old workstation

| **Author** | `matt23654` |
| :--- | :--- |
| **Created** | 2025-05-06 |
| **Updated** | 2025-05-06 |

---

#### Description

Hi! So I have managed to get ubergarm's 235B quant to work on a 6 year old workstation with 2*2080TI's, 64GB RAM and a pretty fast (new) SSD. 

I have encountered some wierd issues with trying to use multiple GPUs though:

- Just using one device and offloading all experts to CPU works.
- The problems start when I try to keep some MoE experts on GPUs...
- Trying to use 2 devices with -sm layer and putting the first few layers entirely on GPU results in a crash on load where for some reason CUDA tries to allocate 170GB of VRAM:

```
llama_new_context_with_model: n_ctx      = 8192
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:      CUDA0 KV buffer size =   768.00 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =   736.00 MiB
llama_new_context_with_model: KV self size  = 1504.00 MiB, K (f16):  752.00 MiB, V (f16):  752.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     1.16 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=4)
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 167771.94 MiB on device 0: cudaMalloc failed: out of memory
ggml_gallocr_reserve_n: failed to allocate CUDA0 buffer of size 175921630208
llama_new_context_with_model: failed to allocate compute buffers
```

- Trying to use -sm row results either in illegal memory access if I specifically pin some expert weights to CUDA1, or the ``GGML_ASSERT(!ggml_backend_buffer_is_cuda_split(src0_1->buffer) && "mul_mat_id does not support split buffers")`` error if I do not. Incidentally I think the last one is because split buffers and 3d tensors are not supported by llama.cpp.

Command used (some variation of):

```
build/bin/llama-server -m ~/.cache/huggingface/hub/models--ubergarm--Qwen3-235B-A22B-GGUF/snapshots/073738969f80d41f288cbfd6a29523769336bee8/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf -ngl 99 --temp 0.6 --top-k 20 --top-p 0.95 --min-p 0 --presence-penalty 1.5 -c 8192 -ot "^blk\.[0-2]\.=CUDA1" -ot "^blk\.[3-9]\.ffn_.*_exps\.=CPU" -ot "[1-9][0-9]\.ffn_.*_exps\.=CPU" --host 127.0.0.1 --port 4000 -fa -fmoe -sm row -mg 0 -v
```

Am I just doing something wrong or is there some genuine bug here?

---

#### 🗣️ Discussion

👤 **ikawrakow** replied the **2025-05-06** at **11:31:27**:<br>

Split mode "row" does not work for MoE models (and I'm not sure if it works for dense models as I don't have access to a multi-GPU system, so have not tested since forking). I'm pretty sure split mode "row" does not work for MoE models in mainline `llama.cpp` either.

With two or more GPU's you may need a more complicated tensor override recipe to get the best possible performance out of the system. For two identical GPU's I think you could start by using
```
-ngl 99 -ot exps=CPU -ts 50,50
```
note how much VRAM this has used on each GPU, and then change to e.g.
```
-ngl 99 -ts 50,50 -ot "blk\.[0-1]\.ffn=CUDA0,blk\.[2-3]\.ffn=CUDA1,exps=CPU
```
(I'm just guessing, as I don't have access to a multi-GPU system).

Note that the tensor overrides are processed in the order they were defined on the command line. So, in the above example, we don't need to be specific about experts tensor layers going to the CPU because the ones that we want to stay on the GPU (layers 0,1 on CUDA0, layers 2,3 on CUDA1) were already handled, so all remaining experts go to the CPU.

If the GPUs are different, then it may be better to just manually define with `-ot` which tensors go where.

> 👤 **matt23654** replied the **2025-05-06** at **13:54:09**:<br>
> Hi @ikawrakow !
> 
> No matter what I do ``-sm layer`` just doesnt seem to work with 2 devices. A variation of your first command segfaults:
> 
> ``build/bin/llama-server -m ~/.cache/huggingface/hub/models--ubergarm--Qwen3-235B-A22B-GGUF/snapshots/073738969f80d41f288cbfd6a29523769336bee8/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf -ngl 99 --temp 0.6 --top-k 20 --top-p 0.95 --min-p 0 --presence-penalty 1.5 -c 8192 --host 127.0.0.1 --port 4000 -fa -fmoe -sm layer -v -ts 50,50 -ot "exps=CPU"``
> 
> ...
> 
> ```
> llama_new_context_with_model: mla_attn   = 0
> llama_new_context_with_model: attn_max_b = 0
> llama_new_context_with_model: fused_moe  = 1
> llama_new_context_with_model: ser        = -1, 0
> llama_new_context_with_model: freq_base  = 1000000.0
> llama_new_context_with_model: freq_scale = 1
> llama_kv_cache_init:      CUDA0 KV buffer size =   768.00 MiB
> llama_kv_cache_init:      CUDA1 KV buffer size =   736.00 MiB
> llama_new_context_with_model: KV self size  = 1504.00 MiB, K (f16):  752.00 MiB, V (f16):  752.00 MiB
> llama_new_context_with_model:  CUDA_Host  output buffer size =     1.16 MiB
> llama_new_context_with_model: pipeline parallelism enabled (n_copies=4)
> ggml_backend_cuda_buffer_type_alloc_buffer: allocating 173219.94 MiB on device 0: cudaMalloc failed: out of memory
> ggml_gallocr_reserve_n: failed to allocate CUDA0 buffer of size 181634272256
> llama_new_context_with_model: failed to allocate compute buffers
> llama_init_from_gpt_params: error: failed to create context with model '~/.cache/huggingface/hub/models--ubergarm--Qwen3-235B-A22B-GGUF/snapshots/073738969f80d41f288cbfd6a29523769336bee8/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf'
>  ERR [              load_model] unable to load model | tid="127462866935808" timestamp=1746539401 model="~/.cache/huggingface/hub/models--ubergarm--Qwen3-235B-A22B-GGUF/snapshots/073738969f80d41f288cbfd6a29523769336bee8/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf"
> Segmentation fault (core dumped)
> ```
> 
> I don't know why it wants to allocate such a huge amount of memory. It doesn't do that with one device or with ``-sm row`` (as mentioned row doesn't work if I try to put any MoE expert tensors on the GPUs).
> 
> 👤 **ubergarm** replied the **2025-05-06** at **13:57:01**:<br>
> @matt23654 
> 
> First I'm  not sure where this came from but a lot of folks keep using `-ot "^blk\.[3-9]\.ffn_.*_exps\.=CPU"` which misses some other ffn layers without the `exps` as the naming convention on Qwen3 is a bit different than DeepSeek for example.
> 
> 
> One other tip for multi-gpu is to recompile with `-DGGML_SCHED_MAX_COPIES=1`
> 
> Look here for more discussions and examples: https://huggingface.co/ubergarm/Qwen3-235B-A22B-GGUF/discussions/1#681642d4a383b2fb9aa3bd8c
> 
> Keep us posted how you get along, as some others have reported success with multi-gpu once they get the arguments just right for their specific systems!
> 
> 👤 **matt23654** replied the **2025-05-06** at **15:19:56**:<br>
> Thanks @ubergarm ! For some reason ``-DGGML_SCHED_MAX_COPIES=1`` works and it no longer tries allocating 170GB of VRAM. I'm getting ~15 tok/s PP and ~6 tok/s generation. Not too bad really for a very old computer offloading from SSD! Specs: i9-9940X, 64GB quad channel ram, 2*2080Ti. I also offloaded all the ffn tensors as suggested.
> 
> I'm guessing that I can't really expect to get a lot of PP speed with SSD offloading and an old CPU (i9-9940X)?
> 
> 👤 **ikawrakow** replied the **2025-05-06** at **16:32:43**:<br>
> @matt23654 I'm curious what happens if you add `-rtr` to your command line. Model loading will take longer, but possibly this will improve your PP performance (PP being only 2.5 times faster than TG does not sound right).
> 
> 👤 **matt23654** replied the **2025-05-06** at **19:59:06**:<br>
> @ikawrakow So there definitely looks to be something a bit wierd going on, maybe because of the SSD, but ``-rtr`` didn't really change PP speed. I've also tried compiling with OpenBLAS, but that somehow seems to have made it slower (yay!).
> 
> The CPU is less active during PP than during regular inference, so I can only assume that somehow the SSD is bottlenecking it. The SSD bandwidth on its own should only be about 0.5tok/s peak, I think the reason generation is so fast is that Qwen isn't choosing experts uniformly and so the kernel caching is making it far closer to the quad-channel ram speed instead. That's my theory, anyway.
> 
> 👤 **ubergarm** replied the **2025-05-06** at **20:44:40**:<br>
> You might be able to get some more out of it, not sure your what your final command was, but give this a try:
> ```
> # do *not* use BLAS and set -DGGML_SCHED_MAX_COPIES=1
> cmake -B build -DGGML_CUDA=ON -DGGML_RPC=OFF -DGGML_BLAS=OFF  -DGGML_SCHED_MAX_COPIES=1
> cmake --build build --config Release -j $(nproc)
> 
> # 1. -sm layer seems default, so i removed it
> # 2. you didn't specify threads? set that to number of physical cores or experiment, i'll assume -t 16
> # 3. try the more simple to understand version regex of listing each ffn layer to each CUDA, increase if u have VRAM
> # 4. explicitly put all other ffn to CPU just so you see it print out on startup
> # 5. use quantized kv cache e.g. q8_0 or q4_0 
> 
> $ build/bin/llama-server \
>     -m ~/.cache/huggingface/hub/models--ubergarm--Qwen3-235B-A22B-GGUF/snapshots/073738969f80d41f288cbfd6a29523769336bee8/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf \
>     -c 8192 \
>     -ctk q8_0 -ctv q8_0 \
>     -fa \
>     -fmoe \
>     -ngl 99 \
>     -ts 50,50 \
>     -ot "blk\.(0|1)\.ffn.*=CUDA0" \
>     -ot "blk\.(2|3)\.ffn.*=CUDA1" \
>     -ot "ffn.*=CPU" \
>     -t 16 \
>     --temp 0.6 \
>     --top-k 20 \
>     --top-p 0.95 \
>     --min-p 0 \
>     --presence-penalty 1.5 \
>     -v \
>     --host 127.0.0.1 \
>     --port 4000
> ```
> 
> If you have more VRAM (assuming like 11GB per GPU?), then try to add one more layer each until you OOM, or use the extra e.g.
> ```
>     -ot "blk\.(0|1|2)\.ffn.*=CUDA0" \
>     -ot "blk\.(3|4|5)\.ffn.*=CUDA1" \
> ```
> 
> Or u can use the extra VRAM for more context etc...
> Curious if you get anything more out of that, and share you updated command whenever. Cheers!
> 
> *EDIT*: I removed `-rtr` because you don't have enough RAM to use that as it disables mmap. You can look into doing the offline tensor repack of the weights not offloaded to GPU so you can get the benefits of the repacked `_R4` and also mmap() to run despite only 64GB RAM.
> 
> So your system is a bit more complex of a setup to get max speed.