### 🔀 [#494](https://github.com/ikawrakow/ik_llama.cpp/pull/494) - IQ1_M_R4 CUDA implementation

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-05 |
| **Updated** | 2025-06-05 |

---

#### Description

To help the quest for the world's smallest DeepSeek model, this PR adds CUDA implementation for `IQ1_M_R4`.

GEMM is done via dequantize+cuBLAS, so may require `cmake -DGGML_CUDA_IQK_FORCE_BF16=ON`.

Performance is on par or even tiny bit better than `IQ1_M`.

Here sweep bench for LlaMA-3-8B on RTX-4080

### IQ1_M

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  2048 |    512 |      0 |    0.347 |  5909.51 |    2.466 |   207.66 |
|  2048 |    512 |   2048 |    0.329 |  6216.59 |    2.657 |   192.69 |
|  2048 |    512 |   4096 |    0.356 |  5745.00 |    2.928 |   174.88 |
|  2048 |    512 |   6144 |    0.384 |  5332.11 |    3.162 |   161.91 |
|  2048 |    512 |   8192 |    0.411 |  4983.68 |    3.380 |   151.50 |
|  2048 |    512 |  10240 |    0.438 |  4678.79 |    3.634 |   140.88 |
|  2048 |    512 |  12288 |    0.466 |  4398.46 |    3.830 |   133.68 |
|  2048 |    512 |  14336 |    0.494 |  4149.40 |    4.095 |   125.03 |

### IQ1_M_R4 (PR)

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  2048 |    512 |      0 |    0.338 |  6058.78 |    2.440 |   209.81 |
|  2048 |    512 |   2048 |    0.323 |  6337.42 |    2.639 |   193.99 |
|  2048 |    512 |   4096 |    0.350 |  5859.50 |    2.914 |   175.71 |
|  2048 |    512 |   6144 |    0.379 |  5409.73 |    3.151 |   162.47 |
|  2048 |    512 |   8192 |    0.405 |  5054.63 |    3.371 |   151.90 |
|  2048 |    512 |  10240 |    0.432 |  4742.62 |    3.618 |   141.52 |
|  2048 |    512 |  12288 |    0.458 |  4471.08 |    3.804 |   134.59 |
|  2048 |    512 |  14336 |    0.486 |  4210.13 |    4.067 |   125.90 |

---

#### 💬 Conversation

👤 **ubergarm** commented the **2025-06-05** at **15:26:27**:<br>

Amazing, you've done it! The pieces of the puzzle are in place. Congrats, ik, on the world's smallest working DeepSeek-R1-0528 quant! :tada: 

With the new DDR5 2x64GB DIMM kits becoming available, an AM5 gaming class rig + GPU can barely fit this little beast!

![thud-sweep-R1-0528-IQ1_S_R4-PR494](https://github.com/user-attachments/assets/5d566460-6d52-46b3-9f72-f5c25c3065a1)

I'm going to double check that `llama-perplexity` still runs clean, but great speed with partial offload is now working!

<details>

<summary>👈 Commands and Logs</summary>

#### Pull and Build
```bash
git branch | grep '*'
* ik/cuda_iq1_m_r4

git rev-parse --short HEAD
8ed7825f

cmake -B ./build -DGGML_CUDA=ON -DGGML_BLAS=OFF -DGGML_SCHED_MAX_COPIES=1 -DGGML_CUDA_IQK_FORCE_BF16=1
cmake --build ./build --config Release -j $(nproc)
```

#### llama-sweep-bench

```bash
model=/mnt/raid/hf/DeepSeek-R1-0528-GGUF/IQ1_S_R4/DeepSeek-R1-0528-IQ1_S_R4-00001-of-00003.gguf

./build/bin/llama-sweep-bench \
  --model "$model" \
  -c 16384 \
  -ctk f16 \
  -mla 3 -fa \
  -amb 512 \
  -fmoe \
  -ngl 99 \
  -ot "blk\.(3|4|5|6|7|8|9|10|11|12|13|13|14|15|16|17|18|19|20)\.ffn_.*=CUDA0" \
  -ot "blk\.(21|22|23|24|25|26|27|28|29|30|31|32|33|34|35|36|37|38)\.ffn_.*=CUDA1" \
  -ot exps=CPU \
  -b 4096 -ub 4096 \
  --warmup-batch \
  --threads 24

ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA RTX A6000, compute capability 8.6, VMM: yes
  Device 1: NVIDIA RTX A6000, compute capability 8.6, VMM: yes

llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q4_0:   61 tensors
llama_model_loader: - type iq4_ks:  551 tensors
llama_model_loader: - type iq1_s_r4:  116 tensors
llama_model_loader: - type iq1_m_r4:   58 tensors

llm_load_print_meta: model type       = 671B
llm_load_print_meta: model ftype      = IQ1_S_R4 - 1.5 bpw
llm_load_print_meta: model params     = 672.050 B
llm_load_print_meta: model size       = 130.203 GiB (1.664 BPW)
llm_load_print_meta: repeating layers = 129.285 GiB (1.657 BPW, 670.196 B parameters)
llm_load_print_meta: general.name     = DeepSeek R1 0528

llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size =  5994.06 MiB
llm_load_tensors:        CPU buffer size = 44211.82 MiB
llm_load_tensors:        CPU buffer size =   469.99 MiB
llm_load_tensors:      CUDA0 buffer size = 42859.65 MiB
llm_load_tensors:      CUDA1 buffer size = 43061.37 MiB

llama_kv_cache_init:      CUDA0 KV buffer size =   576.00 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =   522.00 MiB
llama_new_context_with_model: KV self size  = 1098.00 MiB, c^KV (f16): 1098.00 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.49 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      CUDA0 compute buffer size =  2824.02 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =  2520.01 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   368.05 MiB
llama_new_context_with_model: graph nodes  = 5500
llama_new_context_with_model: graph splits = 111
```

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |    9.959 |   411.28 |   70.744 |    14.47 |
|  4096 |   1024 |   4096 |   12.460 |   328.73 |   73.277 |    13.97 |
|  4096 |   1024 |   8192 |   14.947 |   274.04 |   76.418 |    13.40 |
|  4096 |   1024 |  12288 |   17.442 |   234.84 |   78.654 |    13.02 |

</details>