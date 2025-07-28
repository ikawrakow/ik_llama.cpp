## 🔀 [Pull Request #259](https://github.com/ikawrakow/ik_llama.cpp/pull/259) - Prepare wk_b tensors of DeepSeek models on the fly

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/prepare_wk_b` |
| **Target Branch** | `main` |
| **Created** | 2025-03-15 |
| **Updated** | 2025-03-17 |
| **Merged** | 2025-03-17 |

---

## 📄 Description

This enables usage of MLA also for model files that were converted with mainline `llama.cpp` and hence to not contain the tensors required for MLA.

MLA requires two additional tensors per layer: `wk_v` and `wk_b`. `wk_v` is just a view of half of the `wkv_b` tensor, so it is not actually necessary to have it in the model file. `wk_b` is a transposed version of the other half of `wkv_b`. If `wk_b` is missing in the model file, this PR computes it while loading the model. The newly created tensors are stored on the same back-end where the corresponding `wkv_b` tensor is stored. 

In principle we could remove the preparation of `wk_v` and `wk_b` from `convert_hf_to_gguf.py`, but I decided have some more thorough testing in the wild before doing so. 

Oh, when `wkv_b` is not quantized, `wk_b` uses the same type as `wkv_b` (`fp16` or `bf16`). But if `wkb_b` is quantized, then `wk_b` becomes `Q8_0`, irrespectively of the `wkv_b` type. Transposing a quantized tensor requires dequantization to `fp32`, so to avoid a potential precision loss if `wkv_b` was quantized with low bpw, we simply use `Q8_0` for `wk_b`.

---

## 💬 Conversation

👤 **ubergarm** commented on **2025-03-15** at **16:27:08**

Thanks for pushing this branch, I decided to try this first before downloading/generating my own MLA quant.

Not sure if it only works for certain quantizations? It throws an assertion error for me when trying the unsloth R1 671B `UD-Q2_K_XL`. Here are the details:

```
# Build the experimental branch  `ik/prepare_wk_b`
# Debugging symbols and CUDA backend enabled
git pull
git checkout ik/prepare_wk_b
cmake -B ./build -DCMAKE_BUILD_TYPE=Debug -DGGML_CUDA=ON -DGGML_BLAS=OFF
cmake --build ./build --config Debug -j $(nproc)

git rev-parse --short HEAD
1324de97

./build/bin/llama-server --version
version: 3594 (1324de97)
built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu

# try it with existing non-MLA quant
CUDA_VISIBLE_DEVICES="0," \
gdb ./build/bin/llama-server
(gdb) run \
      --alias unsloth/DeepSeek-R1-UD-Q2_K_XL \
      --model /mnt/raid/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-UD-Q2_K_XL/DeepSeek-R1-UD-Q2_K_XL-00001-of-00005.gguf \
      --ctx-size 4096 \
      --parallel 1 \
      -mla 2 -fa \
      -amb 2048 \
      -fmoe \
      -rtr \
      --n-gpu-layers 63 \
      --override-tensor exps=CPU \
      --threads 24 \
      --host 127.0.0.1 \
      --port 8080

.
.
.
Tensor blk.60.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size = 205716.00 MiB
llm_load_tensors:  CUDA_Host buffer size =   497.11 MiB
llm_load_tensors:      CUDA0 buffer size =  9885.95 MiB
....................................................................................................
============ llm_load_tensors: need to compute 61 wk_b tensors
llama-server: /home/w/projects/ik_llama.cpp/ggml/src/ggml.c:4306: ggml_row_size: Assertion `ne % ggml_blck_size(type) == 0' failed.

Thread 1 "llama-server" received signal SIGABRT, Aborted.
Download failed: Invalid argument.  Continuing without source file ./nptl/./nptl/pthread_kill.c.
__pthread_kill_implementation (no_tid=0, signo=6, threadid=<optimized out>) at ./nptl/pthread_kill.c:44
warning: 44     ./nptl/pthread_kill.c: No such file or directory
(gdb) bt
#0  __pthread_kill_implementation (no_tid=0, signo=6, threadid=<optimized out>) at ./nptl/pthread_kill.c:44
#1  __pthread_kill_internal (signo=6, threadid=<optimized out>) at ./nptl/pthread_kill.c:78
#2  __GI___pthread_kill (threadid=<optimized out>, signo=signo@entry=6) at ./nptl/pthread_kill.c:89
#3  0x00007fffd6e4527e in __GI_raise (sig=sig@entry=6) at ../sysdeps/posix/raise.c:26
#4  0x00007fffd6e288ff in __GI_abort () at ./stdlib/abort.c:79
#5  0x00007fffd6e2881b in __assert_fail_base (fmt=0x7fffd6fd01e8 "%s%s%s:%u: %s%sAssertion `%s' failed.\n%n",
    assertion=assertion@entry=0x7fffd81b1ed8 "ne % ggml_blck_size(type) == 0",
    file=file@entry=0x7fffd81b11a0 "/home/w/projects/ik_llama.cpp/ggml/src/ggml.c", line=line@entry=4306,
    function=function@entry=0x7fffd81b6c58 <__PRETTY_FUNCTION__.74> "ggml_row_size") at ./assert/assert.c:96
#6  0x00007fffd6e3b517 in __assert_fail (assertion=0x7fffd81b1ed8 "ne % ggml_blck_size(type) == 0",
    file=0x7fffd81b11a0 "/home/w/projects/ik_llama.cpp/ggml/src/ggml.c", line=4306, function=0x7fffd81b6c58 <__PRETTY_FUNCTION__.74> "ggml_row_size")
    at ./assert/assert.c:105
#7  0x00007fffd76634b9 in ggml_row_size (type=GGML_TYPE_Q6_K, ne=128) at /home/w/projects/ik_llama.cpp/ggml/src/ggml.c:4306
#8  0x00007ffff7a9ad7b in llm_load_tensors (ml=..., model=..., n_gpu_layers=63, split_mode=LLAMA_SPLIT_MODE_LAYER, main_gpu=0,
    tensor_split=0x7fffffffd0f0, use_mlock=false, progress_callback=0x7ffff7ac1229 <_FUN(float, void*)>, progress_callback_user_data=0x7fffffffbc08)
    at /home/w/projects/ik_llama.cpp/src/llama.cpp:8160
#9  0x00007ffff7aadedc in llama_model_load (
    fname="/mnt/raid/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-UD-Q2_K_XL/DeepSeek-R1-UD-Q2_K_XL-00001-of-00005.gguf", model=..., params=...)
    at /home/w/projects/ik_llama.cpp/src/llama.cpp:8343
#10 0x00007ffff7ac1451 in llama_load_model_from_file (
    path_model=0x5555566a4cb0 "/mnt/raid/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-UD-Q2_K_XL/DeepSeek-R1-UD-Q2_K_XL-00001-of-00005.gguf",
    params=...) at /home/w/projects/ik_llama.cpp/src/llama.cpp:18134
#11 0x000055555572fcbe in llama_init_from_gpt_params (params=...) at /home/w/projects/ik_llama.cpp/common/common.cpp:2197
#12 0x000055555561ff52 in server_context::load_model (this=0x7fffffffd080, params_=...)
    at /home/w/projects/ik_llama.cpp/examples/server/server.cpp:682
#13 0x00005555555f2eec in main (argc=26, argv=0x7fffffffdf18) at /home/w/projects/ik_llama.cpp/examples/server/server.cpp:2628
```

---

👤 **ikawrakow** commented on **2025-03-15** at **16:37:09**

Sorry about that. Hope the fix I just pushed will work.

---

👤 **ubergarm** commented on **2025-03-15** at **17:11:41**

All good, happy to try this out. Great, it does startup okay now!

However, I tried 64k context and threw about 8k prompt at it, and the generation seem wonky. Same for shorter prompts and also at 8k context.

I'm happy to download and try a smaller working test quant, or try any other combination of arguments etc.

#### Observations

* 64k context uses about 34GiB of 48GiB VRAM
* 8k context uses about 14GiB of 48GiB VRAM
* Same issue with and without `-rtr`

#### Long Prompt Test with 64k context
```
>>> User:

(ask a question and copy paste in about 8k from a book)

>>> Assistant:

<think>QQZZJJQQHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH^C
Response cancelled.
```

#### Short Prompt Test with 64k context
```
>>> User:

Count from 1 to 10 in French.

>>> Assistant:

<think>zzzzbbkk and kAAHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH^C
Response cancelled.
```

#### Short Prompt Test with 8k context
```
>>> User:

Count from 1 to 10 in French.

>>> Assistant:

<think>SS and AAkk,	.0
-
kk,	.3
>
HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH^C
Response cancelled.
```

#### Server with 64k context
```bash
$ ./build/bin/llama-server --version
version: 3595 (fc03b9ad)

CUDA_VISIBLE_DEVICES="0," \
./build/bin/llama-server \
    --alias unsloth/DeepSeek-R1-UD-Q2_K_XL \
    --model /mnt/raid/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-UD-Q2_K_XL/DeepSeek-R1-UD-Q2_K_XL-00001-of-00005.gguf \
    --ctx-size 65536 \
    --parallel 1 \
    -mla 2 -fa \
    -amb 2048 \
    -fmoe \
    -rtr \
    --n-gpu-layers 63 \
    --override-tensor exps=CPU \
    --threads 24 \
    --host 127.0.0.1 \
    --port 8080

.
.
.
Tensor blk.60.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size = 205716.00 MiB
llm_load_tensors:  CUDA_Host buffer size =   497.11 MiB
llm_load_tensors:      CUDA0 buffer size =  9885.95 MiB
....................................................................................................
============ llm_load_tensors: need to compute 61 wk_b tensors
Computed blk.0.attn_k_b.weight as 128 x 512 x 128
Computed blk.1.attn_k_b.weight as 128 x 512 x 128
Computed blk.2.attn_k_b.weight as 128 x 512 x 128
Computed blk.3.attn_k_b.weight as 128 x 512 x 128
.
.
.
Computed blk.58.attn_k_b.weight as 128 x 512 x 128
Computed blk.59.attn_k_b.weight as 128 x 512 x 128
Computed blk.60.attn_k_b.weight as 128 x 512 x 128
============ Repacked 174 tensors
llama_new_context_with_model: n_ctx      = 65536
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 2
llama_new_context_with_model: attn_max_b = 2048
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init: layer 0: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 1: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 2: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 3: n_embd_head_qk_rope = 64, kv_lora_rank = 512
.
.
.
llama_kv_cache_init: layer 58: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 59: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 60: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init:      CUDA0 KV buffer size =  4392.00 MiB
llama_new_context_with_model: KV self size  = 4392.00 MiB, c^KV (f16): 4392.00 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.99 MiB
llama_new_context_with_model:      CUDA0 compute buffer size = 19857.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   240.01 MiB
llama_new_context_with_model: graph nodes  = 3548
llama_new_context_with_model: graph splits = 118
INFO [                    init] initializing slots | tid="136342914363392" timestamp=1742057505 n_slots=1
INFO [                    init] new slot | tid="136342914363392" timestamp=1742057505 id_slot=0 n_ctx_slot=65536
INFO [                    main] model loaded | tid="136342914363392" timestamp=1742057505
INFO [                    main] chat template | tid="136342914363392" timestamp=1742057505 chat_example="You are a helpful assistant\n\n<｜User｜>Hell
o<｜Assistant｜>Hi there<｜end▁of▁sentence｜><｜User｜>How are you?<｜Assistant｜>" built_in=true
INFO [                    main] HTTP server listening | tid="136342914363392" timestamp=1742057505 n_threads_http="47" port="8080" hostname="127.0.0.1
"
INFO [            update_slots] all slots are idle | tid="136342914363392" timestamp=1742057505
INFO [      log_server_request] request | tid="136329442553856" timestamp=1742057524 remote_addr="127.0.0.1" remote_port=45946 status=200 method="GET"
 path="/v1/models" params={}
INFO [   launch_slot_with_task] slot is processing task | tid="136342914363392" timestamp=1742057604 id_slot=0 id_task=0
INFO [            update_slots] kv cache rm [p0, end) | tid="136342914363392" timestamp=1742057604 id_slot=0 id_task=0 p0=0
INFO [            update_slots] kv cache rm [p0, end) | tid="136342914363392" timestamp=1742057622 id_slot=0 id_task=0 p0=2048
INFO [            update_slots] kv cache rm [p0, end) | tid="136342914363392" timestamp=1742057643 id_slot=0 id_task=0 p0=4096
INFO [            update_slots] kv cache rm [p0, end) | tid="136342914363392" timestamp=1742057665 id_slot=0 id_task=0 p0=6144
INFO [            update_slots] kv cache rm [p0, end) | tid="136342914363392" timestamp=1742057691 id_slot=0 id_task=0 p0=8192
INFO [      log_server_request] request | tid="136329450946560" timestamp=1742057722 remote_addr="127.0.0.1" remote_port=56568 status=200 method="POST
" path="/v1/chat/completions" params={}
INFO [            update_slots] slot released | tid="136342914363392" timestamp=1742057722 id_slot=0 id_task=0 n_ctx=65536 n_past=8988 n_system_tokens
=0 n_cache_tokens=8988 truncated=false
INFO [            update_slots] all slots are idle | tid="136342914363392" timestamp=1742057722
```

---

👤 **ubergarm** commented on **2025-03-15** at **17:17:59**

Confirmed similar wonky generations using `./build/bin/llama-cli` to take my client out of the picture.

Also currently trying some other combinations. This one with `-mla 1` spammed the logs like so:

```
CUDA_VISIBLE_DEVICES="0," \
./build/bin/llama-cli \
    --alias unsloth/DeepSeek-R1-UD-Q2_K_XL \
    --model /mnt/raid/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-UD-Q2_K_XL/DeepSeek-R1-UD-Q2_K_XL-00001-of-00005.gguf \
    --ctx-size 8192 \
    --parallel 1 \
    -mla 1 -fa \
    --n-gpu-layers 63 \
    --override-tensor exps=CPU \
    --threads 24

Unsupported KV type combination for head_sizes 576 / 512
Unsupported KV type combination for head_sizes 576 / 512
Unsupported KV type combination for head_sizes 576 / 512
Unsupported KV type combination for head_sizes 576 / 512
```

No pressure to stay up late looking at this, I'm having fun. Enjoy your weekend!

---

👤 **ikawrakow** commented on **2025-03-15** at **17:41:33**

Yes, I see similar behavior with DeepSeek-Lite. I broke something somewhere and need to investigate. I got confused and tested with options that did not actually trigger the usage of the computed tensors.

---

👤 **saood06** commented on **2025-03-16** at **00:44:48**

> Also currently trying some other combinations. This one with `-mla 1` spammed the logs like so:
> 
> ```
> CUDA_VISIBLE_DEVICES="0," \
> ./build/bin/llama-cli \
>     --alias unsloth/DeepSeek-R1-UD-Q2_K_XL \
>     --model /mnt/raid/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-UD-Q2_K_XL/DeepSeek-R1-UD-Q2_K_XL-00001-of-00005.gguf \
>     --ctx-size 8192 \
>     --parallel 1 \
>     -mla 1 -fa \
>     --n-gpu-layers 63 \
>     --override-tensor exps=CPU \
>     --threads 24
> 
> Unsupported KV type combination for head_sizes 576 / 512
> Unsupported KV type combination for head_sizes 576 / 512
> Unsupported KV type combination for head_sizes 576 / 512
> Unsupported KV type combination for head_sizes 576 / 512
> ```

I think this is because -mla 1 -fa is currently only supported on the CPU and not on CUDA

---

👤 **ikawrakow** commented on **2025-03-16** at **06:25:30**

@ubergarm Thank you for playing with this, it is very helpful. 

I think I finally fixed the issue with `mla = 2`, so It should work now with Unsloth's models (or any other model created with mainline `llama.cpp`).

I'm surprised by the giant CUDA compute buffer for a context of 65k. This basically renders the `mla=2, fa=1` option useless for anyone not being lucky enough to have a 48 GB GPU. The KV buffer size is exactly as expected (`576 * n_ctx * 61 * sizeof(f16)`. For long contexts most of the compute buffer goes into operations with the KV cache in **one layer**, so I was expecting it to be only marginally larger than the 2788 MiB I observe at 65k tokens for DeepSeek-Lite as the cache size per layer is the same. I guess, I need to look into this more closely.

`-mla 1 -fa` only works on the CPU. I haven't been able to adapt the existing FA kernel to work correctly with head sizes > 256. I guess, I need to write a new CUDA kernel for this case.

---

👤 **ubergarm** commented on **2025-03-16** at **14:38:44**

@ikawrakow 

I appreciate all your discussions in the various PRs, each one a treasure trove of knowledge!

> I think I finally fixed the issue with mla = 2, so It should work now with Unsloth's models (or any other model created with mainline llama.cpp).

I'll give this a try again and confirm. If it works, then I can easily compare perplexity of my new custom quants against the unsloth one I have been using with similar `mla=2 fa=1` options.

> `-mla 1 -fa` only works on the CPU.

Perfect, I'll add a note in my rough guide. I still haven't fully grokk'd the implications of `-mla 1` vs `-mla 2` yet so I'll eventually compare them both on CPU and simply use `-mla 2` for CUDA no problemo.

---

👤 **ubergarm** commented on **2025-03-16** at **15:03:50**

Looks good!

The most recent patch seems to work on the unsloth `UD-Q2_K_XL` quant I have been using with `-mla 2 -fa` etc. The output generations look good for a few simple tests including an ~8k prompt with results shown below.

#### Update Branch
```bash
# update
git checkout ik/prepare_wk_b
git pull
git rev-parse --short HEAD
f2fb15de
# rebuild and confirm
./build/bin/llama-server --version
version: 3596 (f2fb15de)
```

#### Test
```bash
# Uses about 21GiB VRAM @ 32k context
CUDA_VISIBLE_DEVICES="0," \
./build/bin/llama-server \
    --alias unsloth/DeepSeek-R1-UD-Q2_K_XL \
    --model /mnt/raid/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-UD-Q2_K_XL/DeepSeek-R1-UD-Q2_K_XL-00001-of-00005.gguf \
    --ctx-size 32768 \
    -ctk q8_0 -ctv q8_0 \
    -mla 2 -fa \
    -amb 2048 \
    -fmoe \
    --n-gpu-layers 63 \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 24 \
    --host 127.0.0.1 \
    --port 8080
```

#### Logs
Open the details fold for complete logs.
:point_down: 
<details>

<summary>Collapsed Logs</summary>

#### Server
Running script containing above command.
```bash
$ ./myscripts/api-server-DeepSeek-R1-UD-Q2_K_XL.sh
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA RTX A6000, compute capability 8.6, VMM: yes
INFO [                    main] build info | tid="137362671300608" timestamp=1742136822 build=3596 commit="f2fb15de"
INFO [                    main] system info | tid="137362671300608" timestamp=1742136822 n_threads=24 n_threads_batch=-1 total_threads=48 system_info=
"AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F
16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: additional 4 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 48 key-value pairs and 1025 tensors from /mnt/raid/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-UD-Q2_K_XL/De
epSeek-R1-UD-Q2_K_XL-00001-of-00005.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = DeepSeek R1 BF16
llama_model_loader: - kv   3:                       general.quantized_by str              = Unsloth
llama_model_loader: - kv   4:                         general.size_label str              = 256x20B
llama_model_loader: - kv   5:                           general.repo_url str              = https://huggingface.co/unsloth
llama_model_loader: - kv   6:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv   7:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv   8:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv   9:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  10:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv  11:          deepseek2.attention.head_count_kv u32              = 128
llama_model_loader: - kv  12:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  13: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  14:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  15:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  16:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  17:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  18:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  19:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  20:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  21:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  22:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  23:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  24:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  25:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  26:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  27:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  28:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  29:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  30: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  31: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  32:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  33:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  34:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<...
llama_model_loader: - kv  35:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  36:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
llama_model_loader: - kv  37:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  38:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  39:            tokenizer.ggml.padding_token_id u32              = 128815
llama_model_loader: - kv  40:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  41:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  42:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  43:               general.quantization_version u32              = 2
llama_model_loader: - kv  44:                          general.file_type u32              = 10
llama_model_loader: - kv  45:                                   split.no u16              = 0
llama_model_loader: - kv  46:                        split.tensors.count i32              = 1025
llama_model_loader: - kv  47:                                split.count u16              = 5
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q2_K:  171 tensors
llama_model_loader: - type q3_K:    3 tensors
llama_model_loader: - type q4_K:  306 tensors
llama_model_loader: - type q6_K:  184 tensors
llm_load_vocab: special tokens cache size = 819
llm_load_vocab: token to piece cache size = 0.8223 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = deepseek2
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 129280
llm_load_print_meta: n_merges         = 127741
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 163840
llm_load_print_meta: n_embd           = 7168
llm_load_print_meta: n_layer          = 61
llm_load_print_meta: n_head           = 128
llm_load_print_meta: n_head_kv        = 128
llm_load_print_meta: n_rot            = 64
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_embd_head_k    = 192
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: n_embd_k_gqa     = 24576
llm_load_print_meta: n_embd_v_gqa     = 16384
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 18432
llm_load_print_meta: n_expert         = 256
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = yarn
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 0.025
llm_load_print_meta: n_ctx_orig_yarn  = 4096
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = 671B
llm_load_print_meta: model ftype      = Q2_K - Medium
llm_load_print_meta: model params     = 671.026 B
llm_load_print_meta: model size       = 211.034 GiB (2.701 BPW)
llm_load_print_meta: repeating layers = 209.841 GiB (2.694 BPW, 669.173 B parameters)
llm_load_print_meta: general.name     = DeepSeek R1 BF16
llm_load_print_meta: BOS token        = 0 '<｜begin▁of▁sentence｜>'
llm_load_print_meta: EOS token        = 1 '<｜end▁of▁sentence｜>'
llm_load_print_meta: PAD token        = 128815 '<｜PAD▁TOKEN｜>'
llm_load_print_meta: LF token         = 131 'Ä'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_layer_dense_lead   = 3
llm_load_print_meta: n_lora_q             = 1536
llm_load_print_meta: n_lora_kv            = 512
llm_load_print_meta: n_ff_exp             = 2048
llm_load_print_meta: n_expert_shared      = 1
llm_load_print_meta: expert_weights_scale = 2.5
llm_load_print_meta: expert_weights_norm  = 1
llm_load_print_meta: expert_gating_func   = sigmoid
llm_load_print_meta: rope_yarn_log_mul    = 0.1000
llm_load_tensors: ggml ctx size =    0.85 MiB
Tensor blk.3.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CPU
.
.
.
Tensor blk.59.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size = 205716.00 MiB
llm_load_tensors:        CPU buffer size =   497.11 MiB
llm_load_tensors:      CUDA0 buffer size =  9885.95 MiB
....................................................................................................
============ llm_load_tensors: need to compute 61 wk_b tensors
Computed blk.0.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.1.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.2.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.3.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.4.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.5.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.6.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.7.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.8.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.9.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.10.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.11.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.12.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.13.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.14.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.15.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.16.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.17.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.18.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.19.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.20.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.21.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.22.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.23.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.24.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.25.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.26.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.27.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.28.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.29.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.30.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.31.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.32.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.33.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.34.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.35.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.36.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.37.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.38.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.39.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.40.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.41.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.42.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.43.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.44.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.45.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.46.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.47.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.48.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.49.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.50.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.51.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.52.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.53.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.54.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.55.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.56.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.57.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.58.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.59.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.60.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 2
llama_new_context_with_model: attn_max_b = 2048
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init: layer 0: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 1: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 2: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 3: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 4: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 5: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 6: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 7: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 8: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 9: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 10: n_embd_head_qk_rope = 64, kv_lora_rank = 512
.
.
.
llama_kv_cache_init: layer 58: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 59: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 60: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init:      CUDA0 KV buffer size =  1166.65 MiB
llama_new_context_with_model: KV self size  = 1166.62 MiB, c^KV (q8_0): 1166.62 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.99 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =  8470.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =    78.01 MiB
llama_new_context_with_model: graph nodes  = 3548
llama_new_context_with_model: graph splits = 118
INFO [                    init] initializing slots | tid="137362671300608" timestamp=1742136993 n_slots=1
INFO [                    init] new slot | tid="137362671300608" timestamp=1742136993 id_slot=0 n_ctx_slot=32768
INFO [                    main] model loaded | tid="137362671300608" timestamp=1742136993
INFO [                    main] chat template | tid="137362671300608" timestamp=1742136993 chat_example="You are a helpful assistant\n\n<｜User｜>Hell
o<｜Assistant｜>Hi there<｜end▁of▁sentence｜><｜User｜>How are you?<｜Assistant｜>" built_in=true
INFO [                    main] HTTP server listening | tid="137362671300608" timestamp=1742136993 n_threads_http="47" port="8080" hostname="127.0.0.1
"
INFO [            update_slots] all slots are idle | tid="137362671300608" timestamp=1742136993
INFO [      log_server_request] request | tid="137360887316480" timestamp=1742137013 remote_addr="127.0.0.1" remote_port=35958 status=200 method="GET"
 path="/v1/models" params={}
INFO [   launch_slot_with_task] slot is processing task | tid="137362671300608" timestamp=1742137018 id_slot=0 id_task=0
INFO [            update_slots] kv cache rm [p0, end) | tid="137362671300608" timestamp=1742137018 id_slot=0 id_task=0 p0=0
INFO [           print_timings] prompt eval time     =     739.81 ms /    13 tokens (   56.91 ms per token,    17.57 tokens per second) | tid="1373626
71300608" timestamp=1742137056 id_slot=0 id_task=0 t_prompt_processing=739.81 n_prompt_tokens_processed=13 t_token=56.90846153846154 n_tokens_second=1
7.572079317662645
INFO [           print_timings] generation eval time =   37448.69 ms /   549 runs   (   68.21 ms per token,    14.66 tokens per second) | tid="1373626
71300608" timestamp=1742137056 id_slot=0 id_task=0 t_token_generation=37448.694 n_decoded=549 t_token=68.21255737704918 n_tokens_second=14.66005730400
1041
INFO [           print_timings]           total time =   38188.50 ms | tid="137362671300608" timestamp=1742137056 id_slot=0 id_task=0 t_prompt_process
ing=739.81 t_token_generation=37448.694 t_total=38188.504
INFO [            update_slots] slot released | tid="137362671300608" timestamp=1742137056 id_slot=0 id_task=0 n_ctx=32768 n_past=561 n_system_tokens=
0 n_cache_tokens=561 truncated=false
INFO [            update_slots] all slots are idle | tid="137362671300608" timestamp=1742137056
INFO [      log_server_request] request | tid="137349061144576" timestamp=1742137056 remote_addr="127.0.0.1" remote_port=39278 status=200 method="POST
" path="/v1/chat/completions" params={}
INFO [            update_slots] all slots are idle | tid="137362671300608" timestamp=1742137056
INFO [      log_server_request] request | tid="137349052751872" timestamp=1742137139 remote_addr="127.0.0.1" remote_port=52170 status=200 method="GET"
 path="/v1/models" params={}
INFO [   launch_slot_with_task] slot is processing task | tid="137362671300608" timestamp=1742137148 id_slot=0 id_task=551
INFO [            update_slots] kv cache rm [p0, end) | tid="137362671300608" timestamp=1742137148 id_slot=0 id_task=551 p0=2
INFO [            update_slots] kv cache rm [p0, end) | tid="137362671300608" timestamp=1742137179 id_slot=0 id_task=551 p0=2050
INFO [            update_slots] kv cache rm [p0, end) | tid="137362671300608" timestamp=1742137211 id_slot=0 id_task=551 p0=4098
INFO [            update_slots] kv cache rm [p0, end) | tid="137362671300608" timestamp=1742137247 id_slot=0 id_task=551 p0=6146
INFO [            update_slots] kv cache rm [p0, end) | tid="137362671300608" timestamp=1742137285 id_slot=0 id_task=551 p0=8194
INFO [           print_timings] prompt eval time     =  146792.23 ms /  8693 tokens (   16.89 ms per token,    59.22 tokens per second) | tid="137362671300608" timestamp=1742137370 id_slot=0 id_task=551 t_prompt_processing=146792.227 n_prompt_tokens_processed=8693 t_token=16.88625641320603 n_tokens_second=59.2197569153304
INFO [           print_timings] generation eval time =   75395.69 ms /   907 runs   (   83.13 ms per token,    12.03 tokens per second) | tid="137362671300608" timestamp=1742137370 id_slot=0 id_task=551 t_token_generation=75395.694 n_decoded=907 t_token=83.12645424476295 n_tokens_second=12.029864729410143
INFO [           print_timings]           total time =  222187.92 ms | tid="137362671300608" timestamp=1742137370 id_slot=0 id_task=551 t_prompt_processing=146792.227 t_token_generation=75395.694 t_total=222187.92100000003
INFO [            update_slots] slot released | tid="137362671300608" timestamp=1742137370 id_slot=0 id_task=551 n_ctx=32768 n_past=9601 n_system_tokens=0 n_cache_tokens=9601 truncated=false
INFO [            update_slots] all slots are idle | tid="137362671300608" timestamp=1742137370
INFO [      log_server_request] request | tid="137349044359168" timestamp=1742137370 remote_addr="127.0.0.1" remote_port=35304 status=200 method="POST" path="/v1/chat/completions" params={}
INFO [            update_slots] all slots are idle | tid="137362671300608" timestamp=1742137370
```

</details>
:point_up:

---

👤 **ubergarm** commented on **2025-03-16** at **15:19:51**

> The KV buffer size is exactly as expected `(576 * n_ctx * 61 * sizeof(f16))`

#### VRAM Usage vs `--ctx-size`
A few examples running exact command as above and varying only context length. Note I was using `-ctk q8_0 -ctv q8_0`:
```
#####
## --ctx-size 65536
30410MiB

llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size = 205716.00 MiB
llm_load_tensors:        CPU buffer size =   497.11 MiB
llm_load_tensors:      CUDA0 buffer size =  9885.95 MiB
...
llama_kv_cache_init:      CUDA0 KV buffer size =  2333.28 MiB
llama_new_context_with_model: KV self size  = 2333.25 MiB, c^KV (q8_0): 2333.25 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.99 MiB
llama_new_context_with_model:      CUDA0 compute buffer size = 16785.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   142.01 MiB
llama_new_context_with_model: graph nodes  = 3548
llama_new_context_with_model: graph splits = 118

#####
## --ctx-size 32768
20930MiB

llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size = 205716.00 MiB
llm_load_tensors:        CPU buffer size =   497.11 MiB
llm_load_tensors:      CUDA0 buffer size =  9885.95 MiB
...
llama_kv_cache_init:      CUDA0 KV buffer size =  1166.65 MiB
llama_new_context_with_model: KV self size  = 1166.62 MiB, c^KV (q8_0): 1166.62 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.99 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =  8470.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =    78.01 MiB
llama_new_context_with_model: graph nodes  = 3548
llama_new_context_with_model: graph splits = 118

#####
## --ctx-size 16384
16146MiB VRAM

llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size = 205716.00 MiB
llm_load_tensors:        CPU buffer size =   497.11 MiB
llm_load_tensors:      CUDA0 buffer size =  9885.95 MiB
...
llama_kv_cache_init:      CUDA0 KV buffer size =   583.34 MiB
llama_new_context_with_model: KV self size  =  583.31 MiB, c^KV (q8_0):  583.31 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.99 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =  4270.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =    64.01 MiB
llama_new_context_with_model: graph nodes  = 3548
llama_new_context_with_model: graph splits = 118
```

---

👤 **ubergarm** commented on **2025-03-16** at **21:49:12**

Confirmed it is working with three different unsloth quants on that intel6980P. Fastest CPU only speeds I've been able to achieve with this rig!

#### Benchmarks
🪄✨👇
<details>

<summary> Dual Socket Intel Xeon 6980P </summary>

## Single Socket
```
$ git rev-parse --short HEAD
f2fb15de
$ ./build/bin/llama-server --version
version: 3596 (f2fb15de)

$ sudo powerprofilesctl set performance
$ echo 0 | sudo tee /proc/sys/kernel/numa_balancing
$ cat /sys/kernel/mm/transparent_hugepage/enabled
[always] madvise never

# ran this with various number of threads for unsloth Q8_0, Q4_K_M, and UD-Q2_K_XL
numactl -N 0 -m 0 \
./build/bin/llama-bench \
    --model /mnt/ai/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-Q8_0/DeepSeek-R1.Q8_0-00001-of-00015.gguf \
    -ctk f16 -ctv f16 \
    -mla 2 -fa 1 \
    -amb 2048 \
    -fmoe 1 \
    -rtr 1 \
    --numa numactl \
    --threads 43,64,86,128
```


| model                          |       size |     params | backend    | threads | fa | mla |   amb | rtr | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -: | --: | ----: | --: | ---: | ------------: | ---------------: |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      64 |  1 |   2 |  2048 |   1 |    1 |         pp512 |     93.08 ± 0.76 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      64 |  1 |   2 |  2048 |   1 |    1 |         tg128 |     10.02 ± 0.00 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      86 |  1 |   2 |  2048 |   1 |    1 |         pp512 |    114.34 ± 0.67 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      86 |  1 |   2 |  2048 |   1 |    1 |         tg128 |      9.87 ± 0.00 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |     128 |  1 |   2 |  2048 |   1 |    1 |         pp512 |    143.04 ± 7.88 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |     128 |  1 |   2 |  2048 |   1 |    1 |         tg128 |      9.07 ± 0.00 |
| model                          |       size |     params | backend    | threads | fa | mla |   amb | rtr | fmoe |          test |              t/s |
| deepseek2 671B Q8_0            | 664.29 GiB |   671.03 B | CPU        |      43 |  1 |   2 |  2048 |   1 |    1 |         pp512 |     77.28 ± 0.14 |
| deepseek2 671B Q8_0            | 664.29 GiB |   671.03 B | CPU        |      43 |  1 |   2 |  2048 |   1 |    1 |         tg128 |      6.50 ± 0.00 |
| deepseek2 671B Q8_0            | 664.29 GiB |   671.03 B | CPU        |      64 |  1 |   2 |  2048 |   1 |    1 |         pp512 |    107.43 ± 6.55 |
| deepseek2 671B Q8_0            | 664.29 GiB |   671.03 B | CPU        |      64 |  1 |   2 |  2048 |   1 |    1 |         tg128 |      7.52 ± 0.00 |
| deepseek2 671B Q8_0            | 664.29 GiB |   671.03 B | CPU        |      86 |  1 |   2 |  2048 |   1 |    1 |         pp512 |    110.24 ± 4.70 |
| deepseek2 671B Q8_0            | 664.29 GiB |   671.03 B | CPU        |      86 |  1 |   2 |  2048 |   1 |    1 |         tg128 |      7.37 ± 0.00 |
| deepseek2 671B Q8_0            | 664.29 GiB |   671.03 B | CPU        |     128 |  1 |   2 |  2048 |   1 |    1 |         pp512 |    152.62 ± 6.02 |
| deepseek2 671B Q8_0            | 664.29 GiB |   671.03 B | CPU        |     128 |  1 |   2 |  2048 |   1 |    1 |         tg128 |      7.01 ± 0.00 |
| model                          |       size |     params | backend    | threads | fa | mla |   amb | rtr | fmoe |          test |              t/s |
| deepseek2 671B Q2_K - Medium   | 211.03 GiB |   671.03 B | CPU        |      64 |  1 |   2 |  2048 |   1 |    1 |         pp512 |    101.23 ± 0.11 |
| deepseek2 671B Q2_K - Medium   | 211.03 GiB |   671.03 B | CPU        |      64 |  1 |   2 |  2048 |   1 |    1 |         tg128 |      9.47 ± 0.01 |
| deepseek2 671B Q2_K - Medium   | 211.03 GiB |   671.03 B | CPU        |      43 |  1 |   2 |  2048 |   1 |    1 |         pp512 |     76.69 ± 0.14 |
| deepseek2 671B Q2_K - Medium   | 211.03 GiB |   671.03 B | CPU        |      43 |  1 |   2 |  2048 |   1 |    1 |         tg128 |      8.37 ± 0.00 |
| deepseek2 671B Q2_K - Medium   | 211.03 GiB |   671.03 B | CPU        |      64 |  1 |   2 |  2048 |   1 |    1 |         pp512 |     98.91 ± 0.19 |
| deepseek2 671B Q2_K - Medium   | 211.03 GiB |   671.03 B | CPU        |      64 |  1 |   2 |  2048 |   1 |    1 |         tg128 |      9.32 ± 0.01 |
| deepseek2 671B Q2_K - Medium   | 211.03 GiB |   671.03 B | CPU        |      86 |  1 |   2 |  2048 |   1 |    1 |         pp512 |    118.22 ± 0.55 |
| deepseek2 671B Q2_K - Medium   | 211.03 GiB |   671.03 B | CPU        |      86 |  1 |   2 |  2048 |   1 |    1 |         tg128 |      9.63 ± 0.00 |
| deepseek2 671B Q2_K - Medium   | 211.03 GiB |   671.03 B | CPU        |     128 |  1 |   2 |  2048 |   1 |    1 |         pp512 |   147.49 ± 12.00 |
| deepseek2 671B Q2_K - Medium   | 211.03 GiB |   671.03 B | CPU        |     128 |  1 |   2 |  2048 |   1 |    1 |         tg128 |      9.94 ± 0.00 |
| deepseek2 671B Q2_K - Medium   | 211.03 GiB |   671.03 B | CPU        |     172 |  1 |   2 |  2048 |   1 |    1 |         pp512 |    113.38 ± 0.68 |
| deepseek2 671B Q2_K - Medium   | 211.03 GiB |   671.03 B | CPU        |     172 |  1 |   2 |  2048 |   1 |    1 |         tg128 |      8.78 ± 0.00 |


#### Compre `-mla 1,2`

```
numactl -N 0 -m 0 \
./build/bin/llama-bench \
    --model /mnt/ai/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00001-of-00009.gguf \
    -rtr 1 \
    -ctk f16 -ctv f16 \
    -mla 2,1 -fa 1 \
    -amb 2048 \
    -fmoe 1 \
    --numa numactl \
    --threads 43,64,86,128

Computed blk.0.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
.
.
.
Computed blk.60.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
============ Repacked 663 tensors
```

| model                          |       size |     params | backend    | threads | fa | mla |   amb | rtr | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -: | --: | ----: | --: | ---: | ------------: | ---------------: |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      43 |  1 |   2 |  2048 |   1 |    1 |         pp512 |     70.20 ± 0.22 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      43 |  1 |   2 |  2048 |   1 |    1 |         tg128 |      8.52 ± 0.00 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      64 |  1 |   2 |  2048 |   1 |    1 |         pp512 |     92.37 ± 0.21 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      64 |  1 |   2 |  2048 |   1 |    1 |         tg128 |      9.75 ± 0.01 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      86 |  1 |   2 |  2048 |   1 |    1 |         pp512 |    115.09 ± 0.45 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      86 |  1 |   2 |  2048 |   1 |    1 |         tg128 |      9.32 ± 0.00 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |     128 |  1 |   2 |  2048 |   1 |    1 |         pp512 |    143.12 ± 7.15 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |     128 |  1 |   2 |  2048 |   1 |    1 |         tg128 |      8.97 ± 0.00 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      43 |  1 |   2 |  2048 |   1 |    1 |         pp512 |     70.20 ± 0.22 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      43 |  1 |   2 |  2048 |   1 |    1 |         tg128 |      8.52 ± 0.00 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      64 |  1 |   2 |  2048 |   1 |    1 |         pp512 |     92.37 ± 0.21 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      64 |  1 |   2 |  2048 |   1 |    1 |         tg128 |      9.75 ± 0.01 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      86 |  1 |   2 |  2048 |   1 |    1 |         pp512 |    115.09 ± 0.45 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      86 |  1 |   2 |  2048 |   1 |    1 |         tg128 |      9.32 ± 0.00 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |     128 |  1 |   2 |  2048 |   1 |    1 |         pp512 |    143.12 ± 7.15 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |     128 |  1 |   2 |  2048 |   1 |    1 |         tg128 |      8.97 ± 0.00 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      43 |  1 |   1 |  2048 |   1 |    1 |         pp512 |     51.82 ± 0.07 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      43 |  1 |   1 |  2048 |   1 |    1 |         tg128 |      4.44 ± 0.01 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      64 |  1 |   1 |  2048 |   1 |    1 |         pp512 |     83.13 ± 2.56 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      64 |  1 |   1 |  2048 |   1 |    1 |         tg128 |     10.26 ± 0.00 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      86 |  1 |   1 |  2048 |   1 |    1 |         pp512 |     79.87 ± 0.08 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      86 |  1 |   1 |  2048 |   1 |    1 |         tg128 |      6.08 ± 0.02 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |     128 |  1 |   1 |  2048 |   1 |    1 |         pp512 |    125.96 ± 7.73 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |     128 |  1 |   1 |  2048 |   1 |    1 |         tg128 |      9.66 ± 0.00 |


## Dual Socket
#### Test One
```
sudo powerprofilesctl set performance
# *this time try with and without setting numa_balancing*
$ echo 1 | sudo tee /proc/sys/kernel/numa_balancing
$ cat /sys/kernel/mm/transparent_hugepage/enabled
[always] madvise never

./build/bin/llama-bench \
    --model /mnt/ai/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00001-of-00009.gguf \
    -rtr 1 \
    -ctk f16 -ctv f16 \
    -mla 2,1 -fa 1 \
    -amb 2048 \
    -fmoe 1 \
    --numa distribute \
    --threads 64,86,128,172,256

Computed blk.0.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
.
.
.
Computed blk.60.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
============ Repacked 663 tensors
```
**Without NUMA Balancing**
| model                          |       size |     params | backend    | threads | fa | mla |   amb | rtr | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -: | --: | ----: | --: | ---: | ------------: | ---------------: |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      64 |  1 |   2 |  2048 |   1 |    1 |         pp512 |     84.75 ± 0.68 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      64 |  1 |   2 |  2048 |   1 |    1 |         tg128 |      6.84 ± 0.01 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      86 |  1 |   2 |  2048 |   1 |    1 |         pp512 |     99.78 ± 0.31 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      86 |  1 |   2 |  2048 |   1 |    1 |         tg128 |      7.00 ± 0.00 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |     128 |  1 |   2 |  2048 |   1 |    1 |         pp512 |    135.28 ± 0.43 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |     128 |  1 |   2 |  2048 |   1 |    1 |         tg128 |      6.99 ± 0.00 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |     172 |  1 |   2 |  2048 |   1 |    1 |         pp512 |    129.16 ± 3.46 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |     172 |  1 |   2 |  2048 |   1 |    1 |         tg128 |      6.22 ± 0.00 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |     256 |  1 |   2 |  2048 |   1 |    1 |         pp512 |    166.44 ± 5.03 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |     256 |  1 |   2 |  2048 |   1 |    1 |         tg128 |      5.02 ± 0.02 |

** With NUMA Balancing**
| model                          |       size |     params | backend    | threads | fa | mla |   amb | rtr | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -: | --: | ----: | --: | ---: | ------------: | ---------------: |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      64 |  1 |   2 |  2048 |   1 |    1 |         pp512 |     84.70 ± 1.59 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      64 |  1 |   2 |  2048 |   1 |    1 |         tg128 |      6.99 ± 0.00 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      86 |  1 |   2 |  2048 |   1 |    1 |         pp512 |    100.58 ± 0.10 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      86 |  1 |   2 |  2048 |   1 |    1 |         tg128 |      6.98 ± 0.01 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |     128 |  1 |   2 |  2048 |   1 |    1 |         pp512 |    135.53 ± 0.37 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |     128 |  1 |   2 |  2048 |   1 |    1 |         tg128 |      6.82 ± 0.01 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |     172 |  1 |   2 |  2048 |   1 |    1 |         pp512 |    136.60 ± 2.23 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |     172 |  1 |   2 |  2048 |   1 |    1 |         tg128 |      6.02 ± 0.12 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |     256 |  1 |   2 |  2048 |   1 |    1 |         pp512 |   160.48 ± 12.80 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |     256 |  1 |   2 |  2048 |   1 |    1 |         tg128 |      5.08 ± 0.03 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      64 |  1 |   1 |  2048 |   1 |    1 |         pp512 |     74.27 ± 4.43 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      64 |  1 |   1 |  2048 |   1 |    1 |         tg128 |      7.43 ± 0.11 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      86 |  1 |   1 |  2048 |   1 |    1 |         pp512 |     72.91 ± 1.65 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      86 |  1 |   1 |  2048 |   1 |    1 |         tg128 |      5.38 ± 0.22 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |     128 |  1 |   1 |  2048 |   1 |    1 |         pp512 |    106.80 ± 5.28 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |     128 |  1 |   1 |  2048 |   1 |    1 |         tg128 |      7.24 ± 0.36 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |     172 |  1 |   1 |  2048 |   1 |    1 |         pp512 |    106.76 ± 2.56 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |     172 |  1 |   1 |  2048 |   1 |    1 |         tg128 |      5.69 ± 0.01 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |     256 |  1 |   1 |  2048 |   1 |    1 |         pp512 |   144.27 ± 14.69 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |     256 |  1 |   1 |  2048 |   1 |    1 |         tg128 |      5.34 ± 0.37 |

## Test Two
Try `numactl --interleave`
```bash
Current power profile is: performance
Set numa balancing to be:
0

Computed blk.0.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
.
.
.
Computed blk.60.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU
============ Repacked 663 tensors

build: f2fb15de (3596)
```

| model                          |       size |     params | backend    | threads | fa | mla |   amb | rtr | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -: | --: | ----: | --: | ---: | ------------: | ---------------: 
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      42 |  1 |   2 |  2048 |   1 |    1 |         pp512 |     56.47 ± 0.09 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      42 |  1 |   2 |  2048 |   1 |    1 |         tg128 |      6.71 ± 0.02 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      64 |  1 |   2 |  2048 |   1 |    1 |         pp512 |     93.50 ± 0.21 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      64 |  1 |   2 |  2048 |   1 |    1 |         tg128 |      8.09 ± 0.01 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      86 |  1 |   2 |  2048 |   1 |    1 |         pp512 |    109.02 ± 0.15 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      86 |  1 |   2 |  2048 |   1 |    1 |         tg128 |      8.04 ± 0.01 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |     128 |  1 |   2 |  2048 |   1 |    1 |         pp512 |    149.25 ± 0.50 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |     128 |  1 |   2 |  2048 |   1 |    1 |         tg128 |      7.66 ± 0.03 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |     172 |  1 |   2 |  2048 |   1 |    1 |         pp512 |    152.62 ± 0.34 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |     172 |  1 |   2 |  2048 |   1 |    1 |         tg128 |      6.93 ± 0.00 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |     256 |  1 |   2 |  2048 |   1 |    1 |         pp512 |    182.26 ± 8.22 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |     256 |  1 |   2 |  2048 |   1 |    1 |         tg128 |      5.74 ± 0.00 |

Now exactly the same with:
```
Set numa balancing to be:
0
```
| model                          |       size |     params | backend    | threads | fa | mla |   amb | rtr | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -: | --: | ----: | --: | ---: | ------------: | ---------------: |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      42 |  1 |   2 |  2048 |   1 |    1 |         pp512 |     56.00 ± 0.21 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      42 |  1 |   2 |  2048 |   1 |    1 |         tg128 |      6.60 ± 0.01 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      64 |  1 |   2 |  2048 |   1 |    1 |         pp512 |     92.35 ± 0.21 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      64 |  1 |   2 |  2048 |   1 |    1 |         tg128 |      7.83 ± 0.04 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      86 |  1 |   2 |  2048 |   1 |    1 |         pp512 |    104.96 ± 0.35 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |      86 |  1 |   2 |  2048 |   1 |    1 |         tg128 |      7.82 ± 0.01 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |     128 |  1 |   2 |  2048 |   1 |    1 |         pp512 |    141.52 ± 0.78 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |     128 |  1 |   2 |  2048 |   1 |    1 |         tg128 |      7.52 ± 0.04 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |     172 |  1 |   2 |  2048 |   1 |    1 |         pp512 |    147.92 ± 0.38 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |     172 |  1 |   2 |  2048 |   1 |    1 |         tg128 |      6.75 ± 0.01 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |     256 |  1 |   2 |  2048 |   1 |    1 |         pp512 |    182.15 ± 8.15 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | CPU        |     256 |  1 |   2 |  2048 |   1 |    1 |         tg128 |      5.58 ± 0.00 |

</details>