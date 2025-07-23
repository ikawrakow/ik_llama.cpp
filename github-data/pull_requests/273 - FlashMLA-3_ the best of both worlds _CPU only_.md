### 🔀 [#273](https://github.com/ikawrakow/ik_llama.cpp/pull/273) - FlashMLA-3: the best of both worlds (CPU only)

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-03-20 |
| **Updated** | 2025-07-12 |

---

#### Description

For DeepSeek models `mla=1` has a very good TG but low PP performance. `mla=2` has better PP performance, but TG performance rapidly decreases with number of tokens in the KV cache. `mla=0` (i.e., standard attention) has the best PP performance, but TG is even lower than `mla=2`. In addition, standard attention requires a much larger KV cache than `mla = 1,2`. Here are two graphs comparing PP and TG performance of `mla=0,1,2` for DeepSeek-Lite. In all cases FA is enabled, the KV cache is quantized with `Q8_0`, the model weights are quantized with `IQ4_NL`, and the calculations are run on a Ryzen-7950X CPU. The second graph is TG speed as a function of the number of tokens in the KV cache (obtained using `llama-bench -gp Np,64`). Note the logarithmic x-axis for both graphs.

![pp](https://github.com/user-attachments/assets/6d016a80-5e6a-45f1-9f6a-367fa2928cd2)


![tg](https://github.com/user-attachments/assets/0206d0e5-e525-4bca-94f9-0d482448ead2)

Since `mla=1` and `mla=2` use the same KV cache (actually, just K-cache as `V` gets computed from the K-cache), we can take the best parts of `mla=1` and `mla=2`, and create `mla=3`, where prompt processing is done with the `mla=2` approach, while TG is performed with `mla=1`.

Why do we need yet another option? Simply because the CUDA backend does not support `mla=1`,  and the `ggml` back-end is very opinionated about where operations should run, with its opinions often being difficult to predict. Hence, when building the graph with more than one compute backend available, one cannot easily predict if the operation(s) will be run on the CPU or on the other compute backend, so it is easier to just have another option for this that the user can turn on via command line arguments.

Coming back to the above graphs, `mla=3` PP performance is given by the blue curve in the first graph, and TG performance by the red curve in the second graph.

---

#### 💬 Conversation

👤 **ubergarm** commented the **2025-03-20** at **20:55:44**:<br>

Clever idea to combine the best of both worlds, PP with `-mla 2` and TG with `-mla 1`!

So reading closely it sounds like `-mla 3` is for CPU *only*?

> the CUDA backend does not support mla=1

fwiw I before thinking I just compiled and tried to run it with CUDA backend it outputs seemed off (was slower than usual and throwing occasional `DDDDDDD` in llama-server).

I hope to kick the tires on this with the intel 6980P tomorrow. Also that `IQ4_NL` might be good for a hybrid quant on that rig... So many toys to play with, thanks!

---

👤 **ikawrakow** commented the **2025-03-21** at **06:23:02**:<br>

> So reading closely it sounds like -mla 3 would also be for CPU only?

Yes, it is CPU only. Based on the above graphs, this is what I would recommend for CPU-only inference.

>  Also it would throw long strings of DDDDDDDD... So yeah sounds like what you said, not for CUDA backend. haha...

Strange. It does run correctly on my end. The unsupported FA variant (head sizes 576 and 512) gets run on the CPU. I tried and was surprised to see that performance for DeepSeek-Lite is only marginally lower compared to all attention computed on the CPU:
```
./bin/llama-cli -m $model-s 1234 -n 128 -p "I believe the meaning of life is" -t 8 -ngl 100 -ot exps=CPU -mla 3 -fa -fmoe -c 32768
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4080, compute capability 8.9, VMM: yes
Log start
main: build = 3597 (1b62d0fa)

...

llama_kv_cache_init:      CUDA0 KV buffer size =   972.00 MiB
llama_new_context_with_model: KV self size  =  972.00 MiB, c^KV (f16):  972.00 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.39 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =   884.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =    68.01 MiB
llama_new_context_with_model: graph nodes  = 1369
llama_new_context_with_model: graph splits = 54

system_info: n_threads = 8 / 32 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | 
sampling: 
	repeat_last_n = 64, repeat_penalty = 1.000, frequency_penalty = 0.000, presence_penalty = 0.000
	top_k = 40, tfs_z = 1.000, top_p = 0.950, min_p = 0.050, typical_p = 1.000, temp = 0.800
	mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000
sampling order: 
CFG -> Penalties -> top_k -> tfs_z -> typical_p -> top_p -> min_p -> temperature 
generate: n_ctx = 32768, n_batch = 2048, n_predict = 128, n_keep = 1


I believe the meaning of life is to be happy, and that to be happy you need to be free.
I was born in 1970, so the 1980s was my childhood. In the 1980s, my country was governed by a corrupt communist 
dictator. I remember watching TV and hearing the news that a man had been murdered, and the state security had
been sent to find the killer. I remember being excited, I was fascinated by the idea of having to kill someone. I remember 
thinking that I could be a killer. I remember feeling that this was a great way to spend my time. I remember feeling
llama_print_timings:        load time =    1153.30 ms
llama_print_timings:      sample time =       3.44 ms /   128 runs   (    0.03 ms per token, 37209.30 tokens per second)
llama_print_timings: prompt eval time =      75.22 ms /     8 tokens (    9.40 ms per token,   106.36 tokens per second)
llama_print_timings:        eval time =    2365.88 ms /   127 runs   (   18.63 ms per token,    53.68 tokens per second)
llama_print_timings:       total time =    2452.52 ms /   135 tokens
Log end
```
In comparison, the same command but using `-mla 2` gives me 55 t/s.

---

👤 **saood06** commented the **2025-03-21** at **10:59:27**:<br>

Would it be possible to use FA for PP and no FA for TG as that would be the best of both worlds for my AVX-2 system?

Did some testing to get a baseline to later compare against the HugePage mmap version, and PP is the best I've seen for IQ4_K_R4 when FA is turned on (IQ4_K seems like it would still perform better given I had gotten 11.5 t/s before MLA was even implemented but I don't have that quant anymore, and still not sure why it performed better than IQ4_K_R4 especially now that I've seen others use the repacked quants without this issue).

Results with FA off:
[
  {
    "build_commit": "ddc8eee1",
    "build_number": 3599,
    "cuda": false,
    "vulkan": false,
    "kompute": false,
    "metal": false,
    "sycl": false,
    "rpc": "0",
    "gpu_blas": false,
    "blas": false,
    "cpu_info": "Intel(R) Xeon(R) CPU E5-2690 v3 @ 2.60GHz",
    "gpu_info": "",
    "model_filename": "/mnt/sda/opensourcerelease_DeepSeek-R1-bf16/opensourcerelease_DeepSeek-R1-Bf16-256x21B-IQ4_K_R4.gguf",
    "model_type": "deepseek2 671B IQ4_K_R4 - 4.5 bpw",
    "model_size": 379595751424,
    "model_n_params": 672049829376,
    "n_batch": 2048,
    "n_ubatch": 512,
    "n_threads": 48,
    "type_k": "f16",
    "type_v": "f16",
    "n_gpu_layers": 99,
    "split_mode": "layer",
    "main_gpu": 0,
    "no_kv_offload": false,
    "flash_attn": false,
    "mla_attn": 3,
    "attn_max_batch": 0,
    "ser": "-1,0",
    "tensor_split": "0.00",
    "use_mmap": true,
    "embeddings": false,
    "repack": false,
    "fused_moe": true,
    "n_prompt": 512,
    "n_gen": 0,
    "test_time": "2025-03-21T10:11:53Z",
    "avg_ns": 62419796060,
    "stddev_ns": 1009107912,
    "avg_ts": 8.204253,
    "stddev_ts": 0.133555,
    "test": "pp512",
    "samples_ns": [ 63297959014, 60973578738, 61863802862, 63348978014, 62614661674 ],
    "samples_ts": [ 8.08873, 8.39708, 8.27625, 8.08221, 8.177 ]
  },
  {
    "build_commit": "ddc8eee1",
    "build_number": 3599,
    "cuda": false,
    "vulkan": false,
    "kompute": false,
    "metal": false,
    "sycl": false,
    "rpc": "0",
    "gpu_blas": false,
    "blas": false,
    "cpu_info": "Intel(R) Xeon(R) CPU E5-2690 v3 @ 2.60GHz",
    "gpu_info": "",
    "model_filename": "/mnt/sda/opensourcerelease_DeepSeek-R1-bf16/opensourcerelease_DeepSeek-R1-Bf16-256x21B-IQ4_K_R4.gguf",
    "model_type": "deepseek2 671B IQ4_K_R4 - 4.5 bpw",
    "model_size": 379595751424,
    "model_n_params": 672049829376,
    "n_batch": 2048,
    "n_ubatch": 512,
    "n_threads": 48,
    "type_k": "f16",
    "type_v": "f16",
    "n_gpu_layers": 99,
    "split_mode": "layer",
    "main_gpu": 0,
    "no_kv_offload": false,
    "flash_attn": false,
    "mla_attn": 3,
    "attn_max_batch": 0,
    "ser": "-1,0",
    "tensor_split": "0.00",
    "use_mmap": true,
    "embeddings": false,
    "repack": false,
    "fused_moe": true,
    "n_prompt": 0,
    "n_gen": 128,
    "test_time": "2025-03-21T10:17:10Z",
    "avg_ns": 43130895818,
    "stddev_ns": 98868993,
    "avg_ts": 2.967723,
    "stddev_ts": 0.006819,
    "test": "tg128",
    "samples_ns": [ 42963040991, 43127461276, 43187501491, 43164440227, 43212035108 ],
    "samples_ts": [ 2.9793, 2.96795, 2.96382, 2.9654, 2.96214 ]
  }
]

Results with FA on (first PP result can be ignored as there was still some model loading since I saw disk activity):
[
  {
    "build_commit": "ddc8eee1",
    "build_number": 3599,
    "cuda": false,
    "vulkan": false,
    "kompute": false,
    "metal": false,
    "sycl": false,
    "rpc": "0",
    "gpu_blas": false,
    "blas": false,
    "cpu_info": "Intel(R) Xeon(R) CPU E5-2690 v3 @ 2.60GHz",
    "gpu_info": "",
    "model_filename": "/mnt/sda/opensourcerelease_DeepSeek-R1-bf16/opensourcerelease_DeepSeek-R1-Bf16-256x21B-IQ4_K_R4.gguf",
    "model_type": "deepseek2 671B IQ4_K_R4 - 4.5 bpw",
    "model_size": 379595751424,
    "model_n_params": 672049829376,
    "n_batch": 2048,
    "n_ubatch": 512,
    "n_threads": 48,
    "type_k": "f16",
    "type_v": "f16",
    "n_gpu_layers": 99,
    "split_mode": "layer",
    "main_gpu": 0,
    "no_kv_offload": false,
    "flash_attn": true,
    "mla_attn": 3,
    "attn_max_batch": 0,
    "ser": "-1,0",
    "tensor_split": "0.00",
    "use_mmap": true,
    "embeddings": false,
    "repack": false,
    "fused_moe": true,
    "n_prompt": 512,
    "n_gen": 0,
    "test_time": "2025-03-21T09:12:50Z",
    "avg_ns": 51626433358,
    "stddev_ns": 1523685588,
    "avg_ts": 9.949408,
    "stddev_ts": 0.608194,
    "test": "pp512",
    "samples_ns": [ 57560377324, 49541849406, 50790455805, 49287972241, 50951512017 ],
    "samples_ts": [ 8.89501, 10.3347, 10.0806, 10.3879, 10.0488 ]
  },
  {
    "build_commit": "ddc8eee1",
    "build_number": 3599,
    "cuda": false,
    "vulkan": false,
    "kompute": false,
    "metal": false,
    "sycl": false,
    "rpc": "0",
    "gpu_blas": false,
    "blas": false,
    "cpu_info": "Intel(R) Xeon(R) CPU E5-2690 v3 @ 2.60GHz",
    "gpu_info": "",
    "model_filename": "/mnt/sda/opensourcerelease_DeepSeek-R1-bf16/opensourcerelease_DeepSeek-R1-Bf16-256x21B-IQ4_K_R4.gguf",
    "model_type": "deepseek2 671B IQ4_K_R4 - 4.5 bpw",
    "model_size": 379595751424,
    "model_n_params": 672049829376,
    "n_batch": 2048,
    "n_ubatch": 512,
    "n_threads": 48,
    "type_k": "f16",
    "type_v": "f16",
    "n_gpu_layers": 99,
    "split_mode": "layer",
    "main_gpu": 0,
    "no_kv_offload": false,
    "flash_attn": true,
    "mla_attn": 3,
    "attn_max_batch": 0,
    "ser": "-1,0",
    "tensor_split": "0.00",
    "use_mmap": true,
    "embeddings": false,
    "repack": false,
    "fused_moe": true,
    "n_prompt": 0,
    "n_gen": 128,
    "test_time": "2025-03-21T09:42:59Z",
    "avg_ns": 46505789499,
    "stddev_ns": 38516020,
    "avg_ts": 2.752347,
    "stddev_ts": 0.002282,
    "test": "tg128",
    "samples_ns": [ 46438924546, 46531577743, 46531048518, 46509540044, 46517856647 ],
    "samples_ts": [ 2.75631, 2.75082, 2.75085, 2.75212, 2.75163 ]
  }
]

---

👤 **ikawrakow** commented the **2025-03-21** at **11:38:12**:<br>

> Would it be possible to use FA for PP and no FA for TG as that would be the best of both worlds for my AVX-2 system?

I think it is the number of threads that you are using that leads to a lower TG performance. The efficient path is not taken when the number of threads is not a power of 2. Can you try TG with 32 threads to confirm before I try to make changes?

---

👤 **saood06** commented the **2025-03-21** at **11:44:19**:<br>

> I think it is the number of threads that you are using that leads to a lower TG performance. The efficient path is not taken when the number of threads is not a power of 2. Can you try TG with 32 threads to confirm before I try to make changes?

I already had ran some tests with 16,24,32,48 threads with FA on, results below but this is without dropping the caches like I normally do before changing thread counts.

| model                          |       size |     params | backend    | threads | fa | mla | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -: | --: | ---: | ------------: | ---------------: |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      16 |  1 |   3 |    1 |         pp512 |      8.29 ± 1.05 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      16 |  1 |   3 |    1 |         tg128 |      2.62 ± 0.03 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      24 |  1 |   3 |    1 |         pp512 |      9.49 ± 0.10 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      24 |  1 |   3 |    1 |         tg128 |      2.53 ± 0.00 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      32 |  1 |   3 |    1 |         pp512 |      6.89 ± 0.05 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      32 |  1 |   3 |    1 |         tg128 |      2.68 ± 0.01 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      48 |  1 |   3 |    1 |         pp512 |     10.27 ± 0.10 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 353.53 GiB |   672.05 B | CPU        |      48 |  1 |   3 |    1 |         tg128 |      2.61 ± 0.04 |

Sorry, won't be available to run more tests till tommorow.

---

👤 **ikawrakow** commented the **2025-03-21** at **13:20:58**:<br>

Here some results for all combinations of `mla=1,2,3; fa=0,1` on a Risen-5975WX (i.e., 32 Zen3 cores, so vanilla `AVX2` is being used).

```
 ./bin/llama-bench -m junk1.bin -p 0 -n 0 -gp 128,64 -gp 256,64 -gp 512,64 -gp 1024,64 -gp 2048,64 -gp 4096,64 -gp 8192,64 -r 2 -fmoe 1 -mla 1,2,3 -fa 0,1 -t 32 -ctk q8_0
```

| model                   |     params | threads | type_k | fa | mla | fmoe |          test |              t/s |        
| ----------------------- | ---------: | ------: | -----: | -: | --: | ---: | ------------: | ---------------: |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  0 |   1 |    1 |    tg64@pp128 |     34.04 ± 0.03 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  0 |   1 |    1 |    tg64@pp256 |     33.58 ± 0.03 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  0 |   1 |    1 |    tg64@pp512 |     33.34 ± 0.03 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  0 |   1 |    1 |   tg64@pp1024 |     32.76 ± 0.01 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  0 |   1 |    1 |   tg64@pp2048 |     31.45 ± 0.01 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  0 |   1 |    1 |   tg64@pp4096 |     29.25 ± 0.02 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  0 |   1 |    1 |   tg64@pp8192 |     25.58 ± 0.01 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  0 |   2 |    1 |    tg64@pp128 |     33.64 ± 0.02 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  0 |   2 |    1 |    tg64@pp256 |     32.94 ± 0.00 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  0 |   2 |    1 |    tg64@pp512 |     31.92 ± 0.01 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  0 |   2 |    1 |   tg64@pp1024 |     29.92 ± 0.02 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  0 |   2 |    1 |   tg64@pp2048 |     27.27 ± 0.03 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  0 |   2 |    1 |   tg64@pp4096 |     22.59 ± 0.02 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  0 |   2 |    1 |   tg64@pp8192 |     14.65 ± 0.05 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  0 |   3 |    1 |    tg64@pp128 |     33.67 ± 0.04 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  0 |   3 |    1 |    tg64@pp256 |     32.87 ± 0.01 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  0 |   3 |    1 |    tg64@pp512 |     31.86 ± 0.03 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  0 |   3 |    1 |   tg64@pp1024 |     29.89 ± 0.01 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  0 |   3 |    1 |   tg64@pp2048 |     27.29 ± 0.01 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  0 |   3 |    1 |   tg64@pp4096 |     22.62 ± 0.16 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  0 |   3 |    1 |   tg64@pp8192 |     14.70 ± 0.00 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  1 |   1 |    1 |    tg64@pp128 |     34.04 ± 0.02 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  1 |   1 |    1 |    tg64@pp256 |     33.46 ± 0.05 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  1 |   1 |    1 |    tg64@pp512 |     33.11 ± 0.03 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  1 |   1 |    1 |   tg64@pp1024 |     32.43 ± 0.00 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  1 |   1 |    1 |   tg64@pp2048 |     31.02 ± 0.02 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  1 |   1 |    1 |   tg64@pp4096 |     29.08 ± 0.01 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  1 |   1 |    1 |   tg64@pp8192 |     26.02 ± 0.02 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  1 |   2 |    1 |    tg64@pp128 |     33.07 ± 0.05 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  1 |   2 |    1 |    tg64@pp256 |     32.17 ± 0.01 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  1 |   2 |    1 |    tg64@pp512 |     31.32 ± 0.02 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  1 |   2 |    1 |   tg64@pp1024 |     29.82 ± 0.01 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  1 |   2 |    1 |   tg64@pp2048 |     26.84 ± 0.01 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  1 |   2 |    1 |   tg64@pp4096 |     22.79 ± 0.01 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  1 |   2 |    1 |   tg64@pp8192 |     17.13 ± 0.01 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  1 |   3 |    1 |    tg64@pp128 |     33.84 ± 0.03 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  1 |   3 |    1 |    tg64@pp256 |     33.46 ± 0.00 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  1 |   3 |    1 |    tg64@pp512 |     33.17 ± 0.01 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  1 |   3 |    1 |   tg64@pp1024 |     32.48 ± 0.01 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  1 |   3 |    1 |   tg64@pp2048 |     31.18 ± 0.02 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  1 |   3 |    1 |   tg64@pp4096 |     29.13 ± 0.00 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  1 |   3 |    1 |   tg64@pp8192 |     26.12 ± 0.03 |

---

👤 **saood06** commented the **2025-03-22** at **04:25:04**:<br>

> Here some results for all combinations of `mla=1,2,3; fa=0,1` on a Risen-5975WX (i.e., 32 Zen3 cores, so vanilla `AVX2` is being used).
> 
> ```
>  ./bin/llama-bench -m junk1.bin -p 0 -n 0 -gp 128,64 -gp 256,64 -gp 512,64 -gp 1024,64 -gp 2048,64 -gp 4096,64 -gp 8192,64 -r 2 -fmoe 1 -mla 1,2,3 -fa 0,1 -t 32 -ctk q8_0
> ```
> [Selected entries of your table below, not in block quotes as that breaks the markdown formatting]

| model                   |     params | threads | type_k | fa | mla | fmoe |          test |              t/s |        
| ----------------------- | ---------: | ------: | -----: | -: | --: | ---: | ------------: | ---------------: |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  0 |   1 |    1 |   tg64@pp8192 |     25.58 ± 0.01 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  0 |   2 |    1 |   tg64@pp8192 |     14.65 ± 0.05 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  0 |   3 |    1 |   tg64@pp8192 |     14.70 ± 0.00 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  1 |   1 |    1 |   tg64@pp8192 |     26.02 ± 0.02 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  1 |   2 |    1 |   tg64@pp8192 |     17.13 ± 0.01 |
| deepseek2 16B IQ4_NL_R4 |    15.76 B |      32 |   q8_0 |  1 |   3 |    1 |   tg64@pp8192 |     26.12 ± 0.03 |

Looking at your results with FA off, MLA-3 is similar to the lower TG of MLA-2 and not the faster MLA-1, with FA MLA-3 is similar to the faster MLA-1. 

>The difference between dropping and not dropping caches is almost the same as the difference between FA off and FA on? Hope we are not chasing our tale here.

That test was done to check the performance at 16 threads, and to get more insight into the behavior from not dropping the caches when changing thread count since I've known it's bad but haven't done enough testing to understand the variation in severity of the impact of it. The model takes 20-30 minutes to load in depending on thread count (with higher thread count taking longer).

Interestingly PP performance seems to be unaffected by not dropping the cache as the values at 32 and 48 threads match the results with dropping the cache.

>But when you come around to test again, I recommend to try -ctk q8_0. I think the fp16 ->fp32 conversion on your CPU is very slow, and this disproportionally affects the speed of the attention calculations when the KV cache is fp16. 

I ran more tests (new tests run on commit 3d6e25c8 ) and put the results (including the 48 thread results from above) in a table for easy viewing.

| threads | type_k | fa | mla | fmoe |          test |             avg t/s |        stddev t/s
 ------: | -----: | -: | --: | ---: | ------------: | ---------------: | ---------------: |
| 32 | f16 | 0 | 3 | 1 | pp512 | 6.222884 | 0.085403 |
| 32 | f16 | 0 | 3 | 1 | tg128 | 2.927266 | 0.003848 |
| 32 | f16 | 1 | 3 | 1 | pp512 | 6.784420 | 0.282985 |
| 32 | f16 | 1 | 3 | 1 | tg128 | 2.830131 | 0.014125 |
| 32 | q8_0 | 0 | 3 | 1 | pp512 | 6.304752 | 0.079066 |
| 32 | q8_0 | 0 | 3 | 1 | tg128 | 2.934792 | 0.017285 |
| 32 | q8_0 | 1 | 3 | 1 | pp512 | 6.880018 | 0.047091 |
| 32 | q8_0 | 1 | 3 | 1 | tg128 | 2.824385 | 0.011719 |
| 32 | q8_KV | 0 | 3 | 1 | pp512 | 6.211539 | 0.022591 |
| 32 | q8_KV | 0 | 3 | 1 | tg128 | 2.948649 | 0.018792 |
| 48 | f16 | 0 | 3 | 1 | pp512 | 8.204253 | 0.133555 |
| 48 | f16 | 0 | 3 | 1 | tg128 | 2.967723 | 0.006819 |
| 48 | f16 | 1 | 3 | 1 | pp512 | 10.213** | 0.17310** |
| 48 | f16 | 1 | 3 | 1 | tg128 | 2.752347 | 0.002282 |

No results for q8_KV with FA on as it crashed hitting this assert `iqk_mul_mat.cpp:421: GGML_ASSERT(Nx%num_rows == 0) failed`

As you can see the best result for TG of those tested is still 48 threads with FA off and f16 type_k, and for PP it is also 48 threads but with FA on and f16 type_k. Going to q8_0 or q8_KV did help slightly when tested with 32 threads.

PP performance at 32 threads is inline with my testing without dropping the cache where it performed far worse than all other tested thread counts, not really sure why that is, so even if 32 threads was ideal for TG it would come at a steep penalty for PP.

>For tg128 there should be barely any difference between the different mla/fa options.

I know tg128 is not the best test, I prefer to do longer tests, and also test deeper into the KV cache but I was just planning to grab a baseline to see if the HugePage mmap changes can get anywhere close to the +50% TG uplift orca-zhang saw on his machine.

Also https://github.com/ikawrakow/ik_llama.cpp/pull/240 you reported FA degraded MLA-1 performance on AVX2, which is what made me test FA on and off (although I was surprised by seeing a difference with just tg128 as your results both here and there), I forgot that you improved that with https://github.com/ikawrakow/ik_llama.cpp/pull/243, but as shown above the situation I see is different (could it be because of the size of the model?).

---

👤 **ikawrakow** commented the **2025-03-22** at **07:03:25**:<br>

> Looking at your results with FA off, MLA-3 is similar to the lower TG of MLA-2 and not the faster MLA-1, with FA MLA-3 is similar to the faster MLA-1. Is that what is expected?

Yes. With FA off, for TG MLA-3 is identical to MLA-2. With FA on, it is identical to MLA-1.

---

👤 **saood06** commented the **2025-03-22** at **10:21:11**:<br>

Ran MLA-3 with FA through a much longer test via sweep-bench, will do the other 5 combinations as well.

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |   49.300 |    10.39 |   41.575 |     3.08 |
|   512 |    128 |    512 |   56.224 |     9.11 |   43.899 |     2.92 |
|   512 |    128 |   1024 |   62.094 |     8.25 |   50.923 |     2.51 |
|   512 |    128 |   1536 |   66.510 |     7.70 |   57.158 |     2.24 |
|   512 |    128 |   2048 |   67.585 |     7.58 |   49.648 |     2.58 |
|   512 |    128 |   2560 |   70.106 |     7.30 |   71.653 |     1.79 |
|   512 |    128 |   3072 |   75.708 |     6.76 |   78.948 |     1.62 |
|   512 |    128 |   3584 |   78.358 |     6.53 |   50.780 |     2.52 |
|   512 |    128 |   4096 |   81.845 |     6.26 |   89.474 |     1.43 |
|   512 |    128 |   4608 |   85.695 |     5.97 |   94.354 |     1.36 |
|   512 |    128 |   5120 |   90.736 |     5.64 |   57.370 |     2.23 |
|   512 |    128 |   5632 |   95.275 |     5.37 |  103.264 |     1.24 |
|   512 |    128 |   6144 |   99.108 |     5.17 |  110.374 |     1.16 |
|   512 |    128 |   6656 |  101.478 |     5.05 |   58.461 |     2.19 |
|   512 |    128 |   7168 |  105.490 |     4.85 |  122.629 |     1.04 |
|   512 |    128 |   7680 |  108.935 |     4.70 |  135.901 |     0.94 |
|   512 |    128 |   8192 |  114.398 |     4.48 |   61.164 |     2.09 |
|   512 |    128 |   8704 |  115.502 |     4.43 |  135.792 |     0.94 |
|   512 |    128 |   9216 |  122.377 |     4.18 |  143.546 |     0.89 |
|   512 |    128 |   9728 |  121.992 |     4.20 |   65.858 |     1.94 |
|   512 |    128 |  10240 |  125.463 |     4.08 |  152.709 |     0.84 |
|   512 |    128 |  10752 |  133.142 |     3.85 |  159.024 |     0.80 |
|   512 |    128 |  11264 |  138.752 |     3.69 |   70.149 |     1.82 |
|   512 |    128 |  11776 |  139.309 |     3.68 |  167.620 |     0.76 |
|   512 |    128 |  12288 |  145.077 |     3.53 |  174.769 |     0.73 |
|   512 |    128 |  12800 |  148.735 |     3.44 |   73.611 |     1.74 |
|   512 |    128 |  13312 |  150.444 |     3.40 |  180.752 |     0.71 |

The results are not ideal because of the issue with the TG performance often dropping lower but this is something I've experienced many times before with llama-server as well where I would workaround it by just canceling generation and sending requests until it wouldn't hit this issue. This bug seems like it's because it is bouncing around threads and thus resulting in lower CPU usage as I think I saw that when watching btop while it happened, but I may be wrong.

---

👤 **saood06** commented the **2025-03-22** at **22:38:01**:<br>

Here are all 6 configurations (all at 48 threads with fmoe turned on) graphed.

![performance_comparison_pp-1](https://github.com/user-attachments/assets/cb40a59c-568e-4129-9524-8e9884c72689)

![performance_comparison_tg](https://github.com/user-attachments/assets/71e94a75-d06a-4670-956c-c0ce23bf95e2)

The MLA-3 FA on results are only up to 13312 while all other results are up to 15872.

MLA-3 FA on configuration (excluding the strange bug) does seem like the best of both worlds even before https://github.com/ikawrakow/ik_llama.cpp/pull/277 as it matches the strongest performing configuration in both PP and TG.

Raw results:
MLA-1 FA on

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |   56.372 |     9.08 |   41.637 |     3.07 |
|   512 |    128 |    512 |   65.906 |     7.77 |   44.449 |     2.88 |
|   512 |    128 |   1024 |   75.631 |     6.77 |   51.104 |     2.50 |
|   512 |    128 |   1536 |   84.515 |     6.06 |   56.877 |     2.25 |
|   512 |    128 |   2048 |   92.765 |     5.52 |   48.265 |     2.65 |
|   512 |    128 |   2560 |  104.452 |     4.90 |   89.489 |     1.43 |
|   512 |    128 |   3072 |  114.392 |     4.48 |   78.147 |     1.64 |
|   512 |    128 |   3584 |  122.741 |     4.17 |   52.674 |     2.43 |
|   512 |    128 |   4096 |  131.675 |     3.89 |   78.033 |     1.64 |
|   512 |    128 |   4608 |  141.033 |     3.63 |   82.457 |     1.55 |
|   512 |    128 |   5120 |  149.885 |     3.42 |   55.784 |     2.29 |
|   512 |    128 |   5632 |  158.856 |     3.22 |   90.373 |     1.42 |
|   512 |    128 |   6144 |  168.300 |     3.04 |   94.076 |     1.36 |
|   512 |    128 |   6656 |  181.462 |     2.82 |   58.954 |     2.17 |
|   512 |    128 |   7168 |  187.150 |     2.74 |  103.445 |     1.24 |
|   512 |    128 |   7680 |  196.882 |     2.60 |  106.750 |     1.20 |
|   512 |    128 |   8192 |  206.121 |     2.48 |   63.281 |     2.02 |
|   512 |    128 |   8704 |  212.475 |     2.41 |  114.532 |     1.12 |
|   512 |    128 |   9216 |  222.311 |     2.30 |  118.826 |     1.08 |
|   512 |    128 |   9728 |  233.403 |     2.19 |   65.968 |     1.94 |
|   512 |    128 |  10240 |  243.954 |     2.10 |  124.580 |     1.03 |
|   512 |    128 |  10752 |  250.691 |     2.04 |  128.195 |     1.00 |
|   512 |    128 |  11264 |  258.130 |     1.98 |   71.721 |     1.78 |
|   512 |    128 |  11776 |  267.407 |     1.91 |  135.833 |     0.94 |
|   512 |    128 |  12288 |  277.375 |     1.85 |  140.668 |     0.91 |
|   512 |    128 |  12800 |  285.441 |     1.79 |   73.901 |     1.73 |
|   512 |    128 |  13312 |  296.597 |     1.73 |  148.917 |     0.86 |
|   512 |    128 |  13824 |  304.513 |     1.68 |  151.734 |     0.84 |
|   512 |    128 |  14336 |  313.140 |     1.64 |   77.420 |     1.65 |
|   512 |    128 |  14848 |  321.383 |     1.59 |  161.674 |     0.79 |
|   512 |    128 |  15360 |  330.559 |     1.55 |  163.908 |     0.78 |
|   512 |    128 |  15872 |  338.761 |     1.51 |   80.965 |     1.58 |


MLA-1 FA off

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |   52.362 |     9.78 |   39.171 |     3.27 |
|   512 |    128 |    512 |   60.041 |     8.53 |   40.495 |     3.16 |
|   512 |    128 |   1024 |   66.452 |     7.70 |   41.637 |     3.07 |
|   512 |    128 |   1536 |   71.559 |     7.15 |   44.482 |     2.88 |
|   512 |    128 |   2048 |   74.518 |     6.87 |   43.680 |     2.93 |
|   512 |    128 |   2560 |   79.878 |     6.41 |   45.378 |     2.82 |
|   512 |    128 |   3072 |   85.570 |     5.98 |   46.669 |     2.74 |
|   512 |    128 |   3584 |   89.800 |     5.70 |   47.966 |     2.67 |
|   512 |    128 |   4096 |   98.576 |     5.19 |   49.332 |     2.59 |
|   512 |    128 |   4608 |  108.627 |     4.71 |   50.382 |     2.54 |
|   512 |    128 |   5120 |  112.797 |     4.54 |   52.691 |     2.43 |
|   512 |    128 |   5632 |  126.354 |     4.05 |   53.285 |     2.40 |
|   512 |    128 |   6144 |  136.373 |     3.75 |   55.482 |     2.31 |
|   512 |    128 |   6656 |  145.487 |     3.52 |   56.918 |     2.25 |
|   512 |    128 |   7168 |  152.475 |     3.36 |   59.291 |     2.16 |
|   512 |    128 |   7680 |  157.011 |     3.26 |   60.613 |     2.11 |
|   512 |    128 |   8192 |  164.186 |     3.12 |   61.650 |     2.08 |
|   512 |    128 |   8704 |  172.213 |     2.97 |   63.285 |     2.02 |
|   512 |    128 |   9216 |  179.342 |     2.85 |   65.066 |     1.97 |
|   512 |    128 |   9728 |  184.866 |     2.77 |   66.739 |     1.92 |
|   512 |    128 |  10240 |  189.532 |     2.70 |   68.594 |     1.87 |
|   512 |    128 |  10752 |  200.580 |     2.55 |   70.216 |     1.82 |
|   512 |    128 |  11264 |  206.011 |     2.49 |   74.366 |     1.72 |
|   512 |    128 |  11776 |  210.935 |     2.43 |   73.921 |     1.73 |
|   512 |    128 |  12288 |  219.023 |     2.34 |   75.357 |     1.70 |
|   512 |    128 |  12800 |  229.901 |     2.23 |   78.950 |     1.62 |
|   512 |    128 |  13312 |  234.175 |     2.19 |   79.112 |     1.62 |
|   512 |    128 |  13824 |  243.651 |     2.10 |   79.621 |     1.61 |
|   512 |    128 |  14336 |  252.523 |     2.03 |   83.572 |     1.53 |
|   512 |    128 |  14848 |  258.125 |     1.98 |   83.176 |     1.54 |
|   512 |    128 |  15360 |  266.951 |     1.92 |   84.145 |     1.52 |
|   512 |    128 |  15872 |  274.193 |     1.87 |   85.428 |     1.50 |



MLA-2 FA on

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |   48.774 |    10.50 |   39.587 |     3.23 |
|   512 |    128 |    512 |   56.517 |     9.06 |   41.636 |     3.07 |
|   512 |    128 |   1024 |   62.483 |     8.19 |   43.358 |     2.95 |
|   512 |    128 |   1536 |   66.271 |     7.73 |   45.037 |     2.84 |
|   512 |    128 |   2048 |   65.885 |     7.77 |   48.797 |     2.62 |
|   512 |    128 |   2560 |   70.072 |     7.31 |   49.303 |     2.60 |
|   512 |    128 |   3072 |   76.580 |     6.69 |   51.587 |     2.48 |
|   512 |    128 |   3584 |   77.433 |     6.61 |   53.760 |     2.38 |
|   512 |    128 |   4096 |   82.779 |     6.19 |   55.922 |     2.29 |
|   512 |    128 |   4608 |   84.483 |     6.06 |   57.871 |     2.21 |
|   512 |    128 |   5120 |   92.774 |     5.52 |   59.870 |     2.14 |
|   512 |    128 |   5632 |   93.801 |     5.46 |   64.068 |     2.00 |
|   512 |    128 |   6144 |   95.289 |     5.37 |   66.614 |     1.92 |
|   512 |    128 |   6656 |  101.627 |     5.04 |   69.262 |     1.85 |
|   512 |    128 |   7168 |  106.607 |     4.80 |   71.099 |     1.80 |
|   512 |    128 |   7680 |  108.579 |     4.72 |   72.970 |     1.75 |
|   512 |    128 |   8192 |  114.884 |     4.46 |   76.877 |     1.66 |
|   512 |    128 |   8704 |  116.458 |     4.40 |   78.309 |     1.63 |
|   512 |    128 |   9216 |  122.505 |     4.18 |   79.273 |     1.61 |
|   512 |    128 |   9728 |  120.222 |     4.26 |   82.697 |     1.55 |
|   512 |    128 |  10240 |  133.184 |     3.84 |   84.714 |     1.51 |
|   512 |    128 |  10752 |  132.524 |     3.86 |   88.663 |     1.44 |
|   512 |    128 |  11264 |  137.127 |     3.73 |   91.123 |     1.40 |
|   512 |    128 |  11776 |  138.639 |     3.69 |   93.269 |     1.37 |
|   512 |    128 |  12288 |  141.845 |     3.61 |   94.465 |     1.36 |
|   512 |    128 |  12800 |  143.882 |     3.56 |   96.995 |     1.32 |
|   512 |    128 |  13312 |  149.154 |     3.43 |  102.144 |     1.25 |
|   512 |    128 |  13824 |  152.665 |     3.35 |  103.466 |     1.24 |
|   512 |    128 |  14336 |  158.567 |     3.23 |  105.759 |     1.21 |
|   512 |    128 |  14848 |  161.432 |     3.17 |  107.325 |     1.19 |
|   512 |    128 |  15360 |  162.770 |     3.15 |  110.936 |     1.15 |
|   512 |    128 |  15872 |  166.575 |     3.07 |  113.067 |     1.13 |



MLA-2 FA off

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |   50.630 |    10.11 |   38.945 |     3.29 |
|   512 |    128 |    512 |   61.614 |     8.31 |   40.749 |     3.14 |
|   512 |    128 |   1024 |   65.128 |     7.86 |   42.490 |     3.01 |
|   512 |    128 |   1536 |   69.541 |     7.36 |   44.866 |     2.85 |
|   512 |    128 |   2048 |   73.857 |     6.93 |   46.628 |     2.75 |
|   512 |    128 |   2560 |   81.255 |     6.30 |   48.725 |     2.63 |
|   512 |    128 |   3072 |   83.896 |     6.10 |   50.649 |     2.53 |
|   512 |    128 |   3584 |   94.061 |     5.44 |   52.687 |     2.43 |
|   512 |    128 |   4096 |   98.347 |     5.21 |   55.033 |     2.33 |
|   512 |    128 |   4608 |  111.448 |     4.59 |   57.147 |     2.24 |
|   512 |    128 |   5120 |  120.595 |     4.25 |   59.680 |     2.14 |
|   512 |    128 |   5632 |  130.825 |     3.91 |   61.763 |     2.07 |
|   512 |    128 |   6144 |  139.542 |     3.67 |   67.220 |     1.90 |
|   512 |    128 |   6656 |  146.483 |     3.50 |   66.623 |     1.92 |
|   512 |    128 |   7168 |  150.188 |     3.41 |   68.854 |     1.86 |
|   512 |    128 |   7680 |  157.738 |     3.25 |   71.535 |     1.79 |
|   512 |    128 |   8192 |  164.418 |     3.11 |   76.463 |     1.67 |
|   512 |    128 |   8704 |  170.963 |     2.99 |   76.542 |     1.67 |
|   512 |    128 |   9216 |  177.897 |     2.88 |   79.228 |     1.62 |
|   512 |    128 |   9728 |  185.886 |     2.75 |   80.453 |     1.59 |
|   512 |    128 |  10240 |  191.639 |     2.67 |   84.522 |     1.51 |
|   512 |    128 |  10752 |  199.377 |     2.57 |   85.961 |     1.49 |
|   512 |    128 |  11264 |  204.889 |     2.50 |   89.789 |     1.43 |
|   512 |    128 |  11776 |  211.540 |     2.42 |   92.103 |     1.39 |
|   512 |    128 |  12288 |  220.448 |     2.32 |   92.519 |     1.38 |
|   512 |    128 |  12800 |  230.541 |     2.22 |   95.078 |     1.35 |
|   512 |    128 |  13312 |  233.450 |     2.19 |  100.113 |     1.28 |
|   512 |    128 |  13824 |  243.031 |     2.11 |  102.234 |     1.25 |
|   512 |    128 |  14336 |  251.980 |     2.03 |  103.885 |     1.23 |
|   512 |    128 |  14848 |  256.868 |     1.99 |  107.598 |     1.19 |
|   512 |    128 |  15360 |  266.032 |     1.92 |  109.378 |     1.17 |
|   512 |    128 |  15872 |  273.106 |     1.87 |  111.869 |     1.14 |



MLA-3 FA off

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |   52.826 |     9.69 |   38.995 |     3.28 |
|   512 |    128 |    512 |   60.517 |     8.46 |   40.854 |     3.13 |
|   512 |    128 |   1024 |   64.679 |     7.92 |   42.588 |     3.01 |
|   512 |    128 |   1536 |   70.026 |     7.31 |   44.923 |     2.85 |
|   512 |    128 |   2048 |   73.916 |     6.93 |   47.864 |     2.67 |
|   512 |    128 |   2560 |   79.430 |     6.45 |   48.791 |     2.62 |
|   512 |    128 |   3072 |   82.989 |     6.17 |   50.803 |     2.52 |
|   512 |    128 |   3584 |   89.584 |     5.72 |   52.880 |     2.42 |
|   512 |    128 |   4096 |  101.278 |     5.06 |   55.031 |     2.33 |
|   512 |    128 |   4608 |  110.789 |     4.62 |   57.182 |     2.24 |
|   512 |    128 |   5120 |  124.281 |     4.12 |   59.242 |     2.16 |
|   512 |    128 |   5632 |  131.453 |     3.89 |   62.172 |     2.06 |
|   512 |    128 |   6144 |  139.561 |     3.67 |   64.478 |     1.99 |
|   512 |    128 |   6656 |  147.034 |     3.48 |   66.423 |     1.93 |
|   512 |    128 |   7168 |  152.453 |     3.36 |   68.449 |     1.87 |
|   512 |    128 |   7680 |  158.548 |     3.23 |   73.672 |     1.74 |
|   512 |    128 |   8192 |  164.658 |     3.11 |   73.802 |     1.73 |
|   512 |    128 |   8704 |  171.058 |     2.99 |   74.993 |     1.71 |
|   512 |    128 |   9216 |  178.295 |     2.87 |   80.705 |     1.59 |
|   512 |    128 |   9728 |  186.087 |     2.75 |   82.645 |     1.55 |
|   512 |    128 |  10240 |  190.243 |     2.69 |   83.655 |     1.53 |
|   512 |    128 |  10752 |  199.190 |     2.57 |   84.720 |     1.51 |
|   512 |    128 |  11264 |  205.033 |     2.50 |   90.305 |     1.42 |
|   512 |    128 |  11776 |  212.679 |     2.41 |   92.204 |     1.39 |
|   512 |    128 |  12288 |  220.020 |     2.33 |   93.821 |     1.36 |
|   512 |    128 |  12800 |  228.681 |     2.24 |   97.448 |     1.31 |
|   512 |    128 |  13312 |  233.225 |     2.20 |  100.463 |     1.27 |
|   512 |    128 |  13824 |  243.440 |     2.10 |  100.816 |     1.27 |
|   512 |    128 |  14336 |  249.817 |     2.05 |  104.079 |     1.23 |
|   512 |    128 |  14848 |  255.171 |     2.01 |  106.178 |     1.21 |
|   512 |    128 |  15360 |  263.535 |     1.94 |  110.075 |     1.16 |
|   512 |    128 |  15872 |  271.336 |     1.89 |  113.361 |     1.13 |




MLA-3 FA on

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |   49.300 |    10.39 |   41.575 |     3.08 |
|   512 |    128 |    512 |   56.224 |     9.11 |   43.899 |     2.92 |
|   512 |    128 |   1024 |   62.094 |     8.25 |   50.923 |     2.51 |
|   512 |    128 |   1536 |   66.510 |     7.70 |   57.158 |     2.24 |
|   512 |    128 |   2048 |   67.585 |     7.58 |   49.648 |     2.58 |
|   512 |    128 |   2560 |   70.106 |     7.30 |   71.653 |     1.79 |
|   512 |    128 |   3072 |   75.708 |     6.76 |   78.948 |     1.62 |
|   512 |    128 |   3584 |   78.358 |     6.53 |   50.780 |     2.52 |
|   512 |    128 |   4096 |   81.845 |     6.26 |   89.474 |     1.43 |
|   512 |    128 |   4608 |   85.695 |     5.97 |   94.354 |     1.36 |
|   512 |    128 |   5120 |   90.736 |     5.64 |   57.370 |     2.23 |
|   512 |    128 |   5632 |   95.275 |     5.37 |  103.264 |     1.24 |
|   512 |    128 |   6144 |   99.108 |     5.17 |  110.374 |     1.16 |
|   512 |    128 |   6656 |  101.478 |     5.05 |   58.461 |     2.19 |
|   512 |    128 |   7168 |  105.490 |     4.85 |  122.629 |     1.04 |
|   512 |    128 |   7680 |  108.935 |     4.70 |  135.901 |     0.94 |
|   512 |    128 |   8192 |  114.398 |     4.48 |   61.164 |     2.09 |
|   512 |    128 |   8704 |  115.502 |     4.43 |  135.792 |     0.94 |
|   512 |    128 |   9216 |  122.377 |     4.18 |  143.546 |     0.89 |
|   512 |    128 |   9728 |  121.992 |     4.20 |   65.858 |     1.94 |
|   512 |    128 |  10240 |  125.463 |     4.08 |  152.709 |     0.84 |
|   512 |    128 |  10752 |  133.142 |     3.85 |  159.024 |     0.80 |
|   512 |    128 |  11264 |  138.752 |     3.69 |   70.149 |     1.82 |
|   512 |    128 |  11776 |  139.309 |     3.68 |  167.620 |     0.76 |
|   512 |    128 |  12288 |  145.077 |     3.53 |  174.769 |     0.73 |
|   512 |    128 |  12800 |  148.735 |     3.44 |   73.611 |     1.74 |
|   512 |    128 |  13312 |  150.444 |     3.40 |  180.752 |     0.71 |

---

👤 **magikRUKKOLA** commented the **2025-07-12** at **00:39:46**:<br>

@ikawrakow 
> Simply because the CUDA backend does not support `mla=1`, and the `ggml` back-end is very opinionated about where operations should run, with its opinions often being difficult to predict.

That's good to know!  Can you please share more info regarding this issue?