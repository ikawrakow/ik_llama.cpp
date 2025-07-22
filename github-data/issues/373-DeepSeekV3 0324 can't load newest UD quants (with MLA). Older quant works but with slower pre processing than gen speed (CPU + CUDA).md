### 📝 [#373](https://github.com/ikawrakow/ik_llama.cpp/issues/373) - DeepSeekV3 0324 can't load newest UD quants (with MLA). Older quant works but with slower pre processing than gen speed (CPU + CUDA)

| **Author** | `Panchovix` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-04 |
| **Updated** | 2025-05-09 |

---

#### Description

Hi there!

Following a bit from https://github.com/ikawrakow/ik_llama.cpp/issues/305, I managed to make CUDA + CPU work MLA as long as you set the experts on CPU and the active parameters all on GPU.

So I can load the older quant from unsloth (https://huggingface.co/unsloth/DeepSeek-V3-0324-GGUF/tree/main/UD-Q2_K_XL) with

```
./llama-server -m '/llm/DeepSeek-V3-0324-UD-Q2_K_XL-00001-of-00006.gguf' -c 16384 --no-mmap --no-warmup -v -ngl 99 --override-tensor 'blk\.(2[5-9]|[3-6][0-9])\..*_exps\.=CPU' --override-tensor 'blk\.([1-6])\..*_exps\.=CUDA0' --override-tensor 'blk\.([7-9]|1[0])\..*_exps\.=CUDA1' --override-tensor 'blk\.(1[1-5])\..*_exps\.=CUDA2' --override-tensor 'blk\.(1[6-9]|2[0-4])\..*_exps\.=CUDA3' -fmoe -amb 512 -mla 2
```

But pre processing speeds are severly affected. I can't load with the same parameters as cache uses ~80GB at f16. With ctk/ctv 4 loads but quality is really not good.

```
INFO [           print_timings] prompt eval time     =  795446.55 ms /  3781 tokens (  210.38 ms per token,     4.75 tokens per second) | tid="140556999061504" timestamp=1746316599 id_slot=0 id_task=0 t_prompt_processing=795446.549 n_prompt_tokens_processed=3781 t_token=210.37993890505157 n_tokens_second=4.753304926337671
INFO [           print_timings] generation eval time =   42540.22 ms /   360 runs   (  118.17 ms per token,     8.46 tokens per second) | tid="140556999061504" timestamp=1746316599 id_slot=0 id_task=0 t_token_generation=42540.225 n_decoded=360 t_token=118.16729166666666 n_tokens_second=8.462578653497955
```

While, trying to use the newer quants that have MLA "out of the box" after llamacpp PR (https://huggingface.co/unsloth/DeepSeek-V3-0324-GGUF-UD/tree/main/UD-Q2_K_XL), I get this issue.

```
llama_model_load: error loading model: check_tensor_dims: tensor 'blk.0.attn_q_b.weight' has wrong shape; expected  1536, 73728, got  1536, 24576,     1,     1
llama_load_model_from_file: failed to load model
```

For comparison, normal llamacpp with latest UD quant I get these speeds

```
prompt eval time =  146999.55 ms /  3070 tokens (   47.88 ms per token,    20.88 tokens per second)
       eval time =   34334.69 ms /   257 tokens (  133.60 ms per token,     7.49 tokens per second)
```

Ran it with

```
./llama-server -m '/home/GGUFs/DeepSeek-V3-0324-UD-Q2_K_XL-00001-of-00006.gguf' -c 16384 --no-mmap --no-warmup -v -ngl 99 --override-tensor 'blk\.(2[5-9]|[3-6][0-9])\..*_exps\.=CPU' --override-tensor 'blk\.([1-6])\..*_exps\.=CUDA0' --override-tensor 'blk\.([7-9]|1[0])\..*_exps\.=CUDA1' --override-tensor 'blk\.(1[1-5])\..*_exps\.=CUDA2' --override-tensor 'blk\.(1[6-9]|2[0-4])\..*_exps\.=CUDA3'
```

---

#### 💬 Conversation

👤 **clockworkwhale** commented the **2025-05-04** at **01:38:06**:<br>

Confirmed I am also getting the exact same "check_tensor_dims: tensor 'blk.0.attn_q_b.weight' has wrong shape" error when attempting to load the newer quants with ik_llama.

---

👤 **ikawrakow** commented the **2025-05-04** at **04:15:58**:<br>

Please file an issue with mainline `llama.cpp` and/or the creators of the quantized model. MLA implementation existed here long before mainline `llama.cpp` had one, and they decided to make it incompatible with existing GGUFs. The implementation here works with the original GGUFs, and creates the tensors necessary for MLA on-the-fly during model load. The same could have (and should have) be done in mainline.

---

👤 **Panchovix** commented the **2025-05-09** at **19:17:25**:<br>

Closing as it is fixed now on https://github.com/ikawrakow/ik_llama.cpp/commit/43a154d8b8b0e9217114577442cecb224a488d45