### 📝 [#521](https://github.com/ikawrakow/ik_llama.cpp/issues/521) - When offloading semi layers to some GPUs with -ot, TG t/s performance tanks (CUDA + CPU, DeepSeek V3-R1), while not on main llamacpp.

| **Author** | `Panchovix` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-11 |
| **Updated** | 2025-07-10 |

---

#### Description

Hi there, thanks for your work!

I noticed something, when running Deepseek R1 0528. When using parts of a layer to some GPUs, TG t/s tanks, but PP t/s looks normal.

PC info:
Ryzen 7 7800X3D
192GB RAM
Fedora 41
```
./llama-server --list-devices
ggml_cuda_init: found 7 CUDA devices:
  Device 0: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
  Device 1: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
  Device 2: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
  Device 3: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
  Device 4: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 5: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 6: NVIDIA RTX A6000, compute capability 8.6, VMM: yes
```


Command to run is:

ikllamacpp:

```
./llama-sweep-bench -m '/models_llm/DeepSeek-R1-0528-UD-Q3_K_XL-00001-of-00007.gguf' -c 16384 --no-mmap -ngl 999 \
-ot "blk.(0|1|2|3|4|5|6|7).ffn.=CUDA0" \
-ot "blk.(8|9|10|11).ffn.=CUDA1" \
-ot "blk.(12|13|14|15).ffn.=CUDA2" \
-ot "blk.(16|17|18|19|20).ffn.=CUDA3" \
-ot "blk.(21|22|23).ffn.=CUDA4" \
-ot "blk.(24|25|26).ffn.=CUDA5" \
-ot "blk.(27|28|29|30|31|32|33|34).ffn.=CUDA6" \
-ot "blk.35.ffn_(norm|gate_inp|gate_shexp|down_shexp|up_shexp).weight=CUDA4" \
-ot "blk.35.ffn_gate_exps.weight=CUDA4" \
-ot "blk.36.ffn_(norm|gate_inp|gate_shexp|down_shexp|up_shexp).weight=CUDA5" \
-ot "blk.36.ffn_gate_exps.weight=CUDA5" \
-ot "ffn.*=CPU" \
-fa -mg 0 -ub 2048 -fmoe -mla 1
```

speeds look like:

```
main: n_kv_max = 16384, n_batch = 2048, n_ubatch = 2048, flash_attn = 1, n_gpu_layers = 999, n_threads = 8, n_threads_batch = 8

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  2048 |    512 |      0 |   13.489 |   151.83 |  326.413 |     1.57 |
|  2048 |    512 |   2048 |   12.965 |   157.96 |  326.891 |     1.57 |
|  2048 |    512 |   4096 |   13.751 |   148.93 |  327.513 |     1.56 |
|  2048 |    512 |   6144 |   14.467 |   141.56 |  328.236 |     1.56 |
|  2048 |    512 |   8192 |   15.263 |   134.18 |  329.009 |     1.56 |
```

On main llamacpp, I can't quite load llama-bench with the same settings (and I think I can't use llama-sweep-bench, or is not there) but ran the server independently and tried to fit inside the 8192 tokens.

Loaded with

```
./llama-server -m '/models_llm/DeepSeek-R1-0528-UD-Q3_K_XL-00001-of-00007.gguf' -c 16384 --no-mmap -ngl 999 \
-ot "blk.(0|1|2|3|4|5|6|7).ffn.=CUDA0" \
-ot "blk.(8|9|10|11).ffn.=CUDA1" \
-ot "blk.(12|13|14|15).ffn.=CUDA2" \
-ot "blk.(16|17|18|19|20).ffn.=CUDA3" \
-ot "blk.(21|22|23).ffn.=CUDA4" \
-ot "blk.(24|25|26).ffn.=CUDA5" \
-ot "blk.(27|28|29|30|31|32|33|34).ffn.=CUDA6" \
-ot "blk.35.ffn_(norm|gate_inp|gate_shexp|down_shexp|up_shexp).weight=CUDA4" \
-ot "blk.35.ffn_gate_exps.weight=CUDA4" \
-ot "blk.36.ffn_(norm|gate_inp|gate_shexp|down_shexp|up_shexp).weight=CUDA5" \
-ot "blk.36.ffn_gate_exps.weight=CUDA5" \
-ot "ffn.*=CPU" \
-fa -mg 0 -ub 2048
```

Speeds are:

```
slot launch_slot_: id  0 | task 1048 | processing task
slot update_slots: id  0 | task 1048 | new prompt, n_ctx_slot = 16384, n_keep = 0, n_prompt_tokens = 1276
slot update_slots: id  0 | task 1048 | kv cache rm [536, end)
slot update_slots: id  0 | task 1048 | prompt processing progress, n_past = 1276, n_tokens = 740, progress = 0.579937
slot update_slots: id  0 | task 1048 | prompt done, n_past = 1276, n_tokens = 740
slot      release: id  0 | task 1048 | stop processing: n_past = 2413, truncated = 0
slot print_timing: id  0 | task 1048 | 
prompt eval time =    9258.01 ms /   740 tokens (   12.51 ms per token,    79.93 tokens per second)
       eval time =  155399.04 ms /  1138 tokens (  136.55 ms per token,     7.32 tokens per second)
...
srv  params_from_: Chat format: Content-only
slot launch_slot_: id  0 | task 2187 | processing task
slot update_slots: id  0 | task 2187 | new prompt, n_ctx_slot = 16384, n_keep = 0, n_prompt_tokens = 3312
slot update_slots: id  0 | task 2187 | kv cache rm [1276, end)
slot update_slots: id  0 | task 2187 | prompt processing progress, n_past = 3312, n_tokens = 2036, progress = 0.614734
slot update_slots: id  0 | task 2187 | prompt done, n_past = 3312, n_tokens = 2036
slot      release: id  0 | task 2187 | stop processing: n_past = 4610, truncated = 0
slot print_timing: id  0 | task 2187 | 
prompt eval time =   12816.60 ms /  2036 tokens (    6.29 ms per token,   158.86 tokens per second)
       eval time =  179147.95 ms /  1299 tokens (  137.91 ms per token,     7.25 tokens per second)
...
srv  params_from_: Chat format: Content-only
slot launch_slot_: id  0 | task 3487 | processing task
slot update_slots: id  0 | task 3487 | new prompt, n_ctx_slot = 16384, n_keep = 0, n_prompt_tokens = 5481
slot update_slots: id  0 | task 3487 | kv cache rm [3312, end)
slot update_slots: id  0 | task 3487 | prompt processing progress, n_past = 5360, n_tokens = 2048, progress = 0.373654
slot update_slots: id  0 | task 3487 | kv cache rm [5360, end)
slot update_slots: id  0 | task 3487 | prompt processing progress, n_past = 5481, n_tokens = 121, progress = 0.395731
slot update_slots: id  0 | task 3487 | prompt done, n_past = 5481, n_tokens = 121
slot      release: id  0 | task 3487 | stop processing: n_past = 7383, truncated = 0
slot print_timing: id  0 | task 3487 | 
prompt eval time =   21481.40 ms /  2169 tokens (    9.90 ms per token,   100.97 tokens per second)
       eval time =  266511.08 ms /  1903 tokens (  140.05 ms per token,     7.14 tokens per second)
...
srv  params_from_: Chat format: Content-only
slot launch_slot_: id  0 | task 5392 | processing task
slot update_slots: id  0 | task 5392 | new prompt, n_ctx_slot = 16384, n_keep = 0, n_prompt_tokens = 8232
slot update_slots: id  0 | task 5392 | kv cache rm [5481, end)
slot update_slots: id  0 | task 5392 | prompt processing progress, n_past = 7529, n_tokens = 2048, progress = 0.248785
slot update_slots: id  0 | task 5392 | kv cache rm [7529, end)
slot update_slots: id  0 | task 5392 | prompt processing progress, n_past = 8232, n_tokens = 703, progress = 0.334184
slot update_slots: id  0 | task 5392 | prompt done, n_past = 8232, n_tokens = 703
slot      release: id  0 | task 5392 | stop processing: n_past = 10227, truncated = 0
slot print_timing: id  0 | task 5392 | 
prompt eval time =   24427.19 ms /  2751 tokens (    8.88 ms per token,   112.62 tokens per second)
       eval time =  281851.24 ms /  1996 tokens (  141.21 ms per token,     7.08 tokens per second)
```

When running complete tensors (i.e.:)

```
./llama-server -m '/models_llm/DeepSeek-R1-0528-UD-Q3_K_XL-00001-of-00007.gguf' -c 32768 --no-mmap -ngl 999 -ot "blk.(0|1|2|3|4|5|6|7).ffn.=CUDA0" -ot "blk.(8|9|10|11).ffn.=CUDA1" -ot "blk.(12|13|14|15).ffn.=CUDA2" -ot "blk.(16|17|18|19|20).ffn.=CUDA3"  -ot "blk.(20|21|22|23).ffn.=CUDA4" -ot "blk.(24|25|26).ffn.=CUDA5" -ot "blk.(27|28|29|30|31|32|33|34).ffn.=CUDA6" -ot "ffn.*=CPU" -fa -mg 0 -ub 2048 -mla 1 -fmoe
```

I get the expected speed for TG t/s.

I can test or give more info if is needed.

---

#### 💬 Conversation

👤 **Ph0rk0z** commented the **2025-06-11** at **21:47:02**:<br>

If you do fmoe, some of the layers are fused. Do you also see high GPU usage? When I played with this, the up/gate had to be together and then downs could be on a different card. I could tank my prompt processing or my textgen depending on what I chose.

---

👤 **Panchovix** commented the **2025-06-12** at **00:00:44**:<br>

@Ph0rk0z Perfect, it was that! Disabling fmoe makes it work correctly

```
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  2048 |    512 |      0 |   10.990 |   186.35 |   58.726 |     8.72 |
|  2048 |    512 |   2048 |   10.805 |   189.53 |   59.120 |     8.66 |
|  2048 |    512 |   4096 |   11.567 |   177.05 |   59.698 |     8.58 |
|  2048 |    512 |   6144 |   12.275 |   166.84 |   60.586 |     8.45 |
```

I haven't checked GPU usage actually, but I assume it is pretty low as PCIe between GPUs is not optimal at all at X4.

---

👤 **Ph0rk0z** commented the **2025-06-12** at **11:17:06**:<br>

GPU usage gets high when you cause it to bounce between 2 GPUs and produce a bottleneck.

---

👤 **Panchovix** commented the **2025-06-13** at **17:30:21**:<br>

@Ph0rk0z It seems to peg the main GPU when doing PP at 100%, then, while inferencing, usage seems to bounce on some GPUs at ~90% each at the start, but then it drops to 10-30% per GPU.

---

👤 **Ph0rk0z** commented the **2025-06-14** at **11:54:57**:<br>

Then you're not locked up. On mine when the TG became this slow it was doing >50% on only 2 gpu and did it the entire time generating.

---

👤 **Ph0rk0z** commented the **2025-06-14** at **11:54:57**:<br>

Then you're not locked up. On mine when the TG became this slow it was doing >50% on only 2 gpu and did it the entire time.

---

👤 **ubergarm** commented the **2025-07-10** at **02:33:12**:<br>

@Panchovix 

> Disabling fmoe makes it work correctly

As we discussed over on [ubergarm/DeepSeek-TNG-R1T2-Chimera-GGUF](https://huggingface.co/ubergarm/DeepSeek-TNG-R1T2-Chimera-GGUF/discussions/2#686ef0ccc5d154595fd460df) I'd recommend to definitely keep using `-fmoe` for any MoE including DeepSeek and to *avoid* splitting `ffn_(gate|up)` tensors between different GPUs/CPU to take advantage of this optimization.

Just leaving this here if anyone else stumbles across this in the future. 

Finally, thanks for all your help testing and tuning with your unique collection of GPUs! Thanks!