### 🔀 [#270](https://github.com/ikawrakow/ik_llama.cpp/pull/270) - Honor mmap setting when using tensor overrides

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-03-19 |
| **Updated** | 2025-03-19 |

---

#### Description

The reason why `mmap` was disabled when using tensor overrides is this:
* When the command line argument is parsed (and the override buffer is set to `CPU`), we get  the buffer type returned by `ggml_backend_cpu_buffer_type()`
* The tensor loading logic uses `llama_default_buffer_type_cpu(true)` instead to see if a buffer is a CPU buffer and hence can be memory mapped.
* When CUDA (or some other backend) is enabled, `llama_default_buffer_type_cpu(true)` returns a different buffer type (`CUDA_Host` in the case of the CUDA backend).
* As a result, the tensors set to be stored in the CPU memory buffer are not memory mapped

This PR fixes that by asking the buffer type to be either `llama_default_buffer_type_cpu(true)` or `ggml_backend_cpu_buffer_type()` to be eligible for using `mmap`.

Note, however, that `-rtr` still disables `mmap` because otherwise the model would be overwritten with the repacked tensors.

---

#### 💬 Conversation

👤 **ubergarm** commented the **2025-03-19** at **19:52:45**:<br>

Wow sweet! I just got back home and saw this, pull'd and rebuilt and got my custom quant running locally on the 9950X + 96GB DDR5-6400 RAM + 3090TI 24GB! Got about 3 tok/sec generation on a quick initial test.

This quant is heavy (`q8_0` on the GPU offload tensors) but still fits 32k context with enough left-over for x windows! Better perplexity than the unsloth `UD-Q2_K_XL` too.

Amazing that `mmap()` and Linux page cache can serve ~238GiB model weights off of a PCIe Gen 5 Crucial T700 2TB NVMe and 2x48GB tuned DIMMs.

This setup might benefit from `-ser 6,1` too! Plenty to try out, thanks!

```bash
./build/bin/llama-server \
    --alias ubergarm/DeepSeek-R1-Q2_K_R4 \
    --model /mnt/ai/models/ubergarm/DeepSeek-R1-GGUF/DeepSeek-R1-GGUF-Q2_K_R4.gguf \
    --ctx-size 32768 \
    -ctk q8_0 \
    -mla 2 -fa \
    -amb 512 \
    -fmoe \
    --n-gpu-layers 63 \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 16 \
    --host 127.0.0.1 \
    --port 8080

...
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  612 tensors
llama_model_loader: - type q2_k_r4:  116 tensors
llama_model_loader: - type q3_k_r4:   58 tensors
...
```