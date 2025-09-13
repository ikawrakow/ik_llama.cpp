## 📌 [Issue #655](https://github.com/ikawrakow/ik_llama.cpp/issues/655) - Bug: warning: failed to munlock buffer: Cannot allocate memory

| **Author** | `magikRUKKOLA` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2025-07-26 |

---

## 📄 Description

### What happened?

Any idea what's going on?


```

export MALLOC_CONF="background_thread:true,percpu_arena:phycpu,metadata_thp:auto,dirty_decay_ms:10000,muzzy_decay_ms:60000"
export LD_PRELOAD=/usr/local/lib/libjemalloc.so

ulimit -n 9999
ulimit -l unlimited

CUDA_VISIBLE_DEVICES="0,1" \
/opt/ik_llama.cpp/ik_llama.cpp/build/bin/llama-sweep-bench \
    --warmup-batch \
    --model /opt/ubergarm/Kimi-K2-Instruct-GGUF/IQ3_KS/Kimi-K2-Instruct-IQ3_KS-00001-of-00010.gguf \
    --alias ubergarm/Kimi-K2-Instruct-GGUF_IQ3_KS \
    --ctx-size $((128 * 1024)) \
    -b $((32 * 512)) -ub $((16 * 512)) \
    --mlock \
    --seed 3407 \
    --temp 0.5 --top-k 0 --top-p 1.0 --min-p 0.1 --repeat-penalty 1.0 \
    -ctk q8_0 \
    -mla 3 -fa \
    -amb 512 \
    --main-gpu 1 \
    --tensor-split 2,8 \
    --split-mode layer \
    --override-tensor exps=CPU \
    --n-gpu-layers 99 \
    --threads $(grep ^cpu\\scores /proc/cpuinfo | uniq | awk '{print $4}' | xargs -I{} echo "{}-0" | bc) \
    --host 0.0.0.0 \
    --port 8080 \
    --lookup-cache-dynamic /mnt/data/ik_llama.kv.dump
```


### Name and Version

/opt/ik_llama.cpp/ik_llama.cpp/build/bin/llama-sweep-bench --version
version: 3822 (4e9c78c0)
built with cc (Debian 14.2.0-19) 14.2.0 for x86_64-linux-gnu

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
....................................................................................................
llama_new_context_with_model: n_ctx      = 131072
llama_new_context_with_model: n_batch    = 16384
llama_new_context_with_model: n_ubatch   = 8192
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 3
llama_new_context_with_model: attn_max_b = 512
llama_new_context_with_model: fused_moe  = 0
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 50000.0
llama_new_context_with_model: freq_scale = 0.03125
llama_kv_cache_init:      CUDA0 KV buffer size =   994.51 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =  3672.02 MiB
llama_new_context_with_model: KV self size  = 4666.50 MiB, c^KV (q8_0): 4666.50 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.62 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      CUDA0 compute buffer size = 12432.03 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =  8672.06 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =  4320.09 MiB
llama_new_context_with_model: graph nodes  = 13771
llama_new_context_with_model: graph splits = 231

main: n_kv_max = 131072, n_batch = 16384, n_ubatch = 8192, flash_attn = 1, n_gpu_layers = 99, n_threads = 64, n_threads_batch = 64

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  8192 |   2048 |      0 |   35.383 |   231.52 |  199.517 |    10.26 |
|  8192 |   2048 |   8192 |   39.030 |   209.89 |  208.009 |     9.85 |
|  8192 |   2048 |  16384 |   43.585 |   187.96 |  218.848 |     9.36 |
|  8192 |   2048 |  24576 |   48.235 |   169.84 |  226.935 |     9.02 |
|  8192 |   2048 |  32768 |   52.680 |   155.50 |  236.670 |     8.65 |
|  8192 |   2048 |  40960 |   57.465 |   142.56 |  245.538 |     8.34 |
|  8192 |   2048 |  49152 |   62.096 |   131.93 |  254.381 |     8.05 |
|  8192 |   2048 |  57344 |   66.846 |   122.55 |  264.281 |     7.75 |
|  8192 |   2048 |  65536 |   71.637 |   114.35 |  272.132 |     7.53 |
|  8192 |   2048 |  73728 |   76.372 |   107.26 |  280.351 |     7.31 |
|  8192 |   2048 |  81920 |   81.235 |   100.84 |  283.917 |     7.21 |
|  8192 |   2048 |  90112 |   86.135 |    95.11 |  292.227 |     7.01 |
|  8192 |   2048 |  98304 |   91.048 |    89.97 |  300.119 |     6.82 |
|  8192 |   2048 | 106496 |   95.891 |    85.43 |  309.025 |     6.63 |
|  8192 |   2048 | 114688 |  100.902 |    81.19 |  317.808 |     6.44 |
|  8192 |   2048 | 122880 |  105.924 |    77.34 |  325.710 |     6.29 |
warning: failed to munlock buffer: Cannot allocate memory
warning: failed to munlock buffer: Cannot allocate memory
warning: failed to munlock buffer: Cannot allocate memory
warning: failed to munlock buffer: Cannot allocate memory
warning: failed to munlock buffer: Cannot allocate memory
warning: failed to munlock buffer: Cannot allocate memory
warning: failed to munlock buffer: Cannot allocate memory
warning: failed to munlock buffer: Cannot allocate memory
warning: failed to munlock buffer: Cannot allocate memory
warning: failed to munlock buffer: Cannot allocate memory
```