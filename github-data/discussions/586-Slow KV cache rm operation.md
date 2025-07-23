### 🗣️ [#586](https://github.com/ikawrakow/ik_llama.cpp/discussions/586) - Slow KV cache rm operation

| **Author** | `jneloexpirements` |
| :--- | :--- |
| **Created** | 2025-07-05 |
| **Updated** | 2025-07-05 |

---

#### Description

Is this related to #451 ?
I am running DeepSeek-R1-V3-0324-IQ4_K_R4 (ubergarm's Q4) quant and while the token generation is decent (i have seen 12 tps at 0, around 66% when it goes to)

I use intel Xeon QYFS, 512GB DDR5 4800 RAM, and a RTX PRO 6000.
I run the command below and also for real use case change it from sweep-bench to server with host/port
```
CUDA_VISIBLE_DEVICES="0," \
./build/bin/llama-sweep-bench \
    --model /mnt/x/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-IQ4_K_R4-00001-of-00010.gguf \
    --alias ubergarm/DeepSeek-R1-V3-0324-IQ4_K_R4 \
    --ctx-size 98304 \
    -ctk q8_0 \
    -mla 3 -fa \
    -amb 8192 \
    -fmoe \
    --temp 0.3 \
    --min-p 0.05 \
    --n-gpu-layers 63 \
    -ot "blk\.[3-9]\.ffn_.*=CUDA0" \
    -ot exps=CPU \
    -ub 8192 -b 8192 \
    --parallel 1 \
    --threads 57
```
The above command puts VRAM usage to 90376 out of 97887 MiB.
```
....................................................................................................
llama_new_context_with_model: n_ctx      = 98304
llama_new_context_with_model: n_batch    = 8192
llama_new_context_with_model: n_ubatch   = 8192
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 3
llama_new_context_with_model: attn_max_b = 8192
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init:      CUDA0 KV buffer size =  3499.90 MiB
llama_new_context_with_model: KV self size  = 3499.88 MiB, c^KV (q8_0): 3499.88 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.49 MiB
ggml_cuda_host_malloc: failed to allocate 3296.09 MiB of pinned memory: invalid argument
llama_new_context_with_model:      CUDA0 compute buffer size = 20496.03 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =  3296.09 MiB
llama_new_context_with_model: graph nodes  = 4219
llama_new_context_with_model: graph splits = 104

```
The raw PP seems to be proper and not irregularly slow from sweep-bench (in this example and also past ones)
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  8192 |   2048 |      0 |   65.721 |   124.65 |  173.995 |    11.77 |
|  8192 |   2048 |   8192 |   69.385 |   118.07 |  190.416 |    10.76 |
|  8192 |   2048 |  16384 |   73.025 |   112.18 |  199.023 |    10.29 |
|  8192 |   2048 |  24576 |   76.688 |   106.82 |  204.607 |    10.01 |
|  8192 |   2048 |  32768 |   79.945 |   102.47 |  208.366 |     9.83 |

I can tolerate the TG but...

In real use cases however which are RAG heavy (feeding it long documents, then chatting for a while on it and websearch) and I like to flip flop between conversations, I have to wait for 2-5 minutes for KV cache removal.
```
INFO [            update_slots] kv cache rm [p0, end) | tid="125357154684928" timestamp=1751624758 id_slot=0 id_task=12104 p0=8410
INFO [           print_timings] prompt eval time     =  128443.90 ms / 10172 tokens (   12.63 ms per token,    79.19 tokens per second) |  timestamp=1751624830 id_slot=0 id_task=12104 t_prompt_processing=128443.905 n_prompt_tokens_processed=10172 t_token=12.627202615021627 n_tokens_second=79.19410422783393
INFO [           print_timings] generation eval time =   10688.65 ms /   122 runs   (   87.61 ms per token,    11.41 tokens per second) | timestamp=1751624830 id_slot=0 id_task=12104 t_token_generation=10688.646 n_decoded=122 t_token=87.6118524590164 n_tokens_second=11.413980779230597

```
The time it took to for KV removal was around 3 minutes thats imo too slow. even if it is 8192 I tried with 4096 2048 or any number KV is just too slow.

1. Does `ggml_cuda_host_malloc: failed to allocate 3296.09 MiB of pinned memory: invalid argument` have anything to do with that? How to fix this problem?
2. Is 60-120 SPP for 4096/8192 batch expected for systems that offload Dense to GPU and experts to CPU?
3. Is KV removal operation tied to PP or is it a separate thing?

Any help is appreciated so that I can mitigate before-generation slowdowns